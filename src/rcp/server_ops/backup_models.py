"""Strict, secret-free plans and manifests for team-server backups."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp.config import (
    AGENT_EXECUTION_PROFILES,
    AgentExecutionProfile,
    AgentPermissions,
    Manifest,
)
from rcp.core.transition_models import GraphHeadRef
from rcp.limits import (
    BACKUP_DIAGNOSTIC_MAX_CHARS,
    BACKUP_INVENTORY_MAX_ENTRIES,
)
from rcp.providers import PROVIDERS, ProviderId
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.models import redact_server_text
from rcp.skill_registry import SkillDefaults

BACKUP_MANIFEST_SCHEMA_VERSION = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_OPENSSH_FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{20,64}={0,2}")
_ALIAS = re.compile(r"[a-z][a-z0-9-]{0,47}")
_HOST = re.compile(r"[A-Za-z0-9_.@:-]{0,255}")
_ACCOUNT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")

BACKUP_APP_DATA_DATABASE = "rcp.sqlite3"
BACKUP_APP_DATA_CAPTURED = frozenset({"project-sources"})
BACKUP_APP_DATA_DEFERRED = frozenset()
BACKUP_APP_DATA_EXCLUSIONS = frozenset(
    {
        "bootstrap-manifests",
        "chat-attachments",
        "paper-snapshots",
        "project-caches",
        "project-snapshots",
        "rcp-server.json",
        "rcp.lock",
        "rcp.sqlite3-journal",
        "rcp.sqlite3-shm",
        "rcp.sqlite3-wal",
        "run-stage",
        "session-slices",
        "source-cache",
        "state-cache",
        "transfer-exports",
        "transfer-inbox",
    }
)

BACKUP_RESEARCH_DELEGATED_ROOTS = frozenset({"chat", "facts", "paper"})
BACKUP_RESEARCH_EXCLUSIONS = frozenset(
    {
        ".agent-run.lock",
        ".append.lock",
        ".chat.lock",
        ".publish",
        ".refresh.lock",
        "coverage.json",
        "cursors.json",
        "glossary.json",
        "graph.json",
        "proposals.json",
        "research.md",
    }
)
BACKUP_RESEARCH_CANONICAL_ROOTS = frozenset(
    {"branches", "manifest.toml", "patches", "scope-base.json"}
)
BACKUP_MATERIALIZED_NAMES = frozenset(
    {
        "coverage.json",
        "cursors.json",
        "glossary.json",
        "graph.json",
        "proposals.json",
        "research.md",
    }
)

BackupFileGroup = Literal[
    "sqlite_snapshot",
    "canonical",
    "chat",
    "paper_introduction",
    "fact",
    "kept_artifact",
    "legacy_kept_result_view",
    "imported_provider_history",
]
BackupCanonicalFileKind = Literal[
    "manifest",
    "scope_base",
    "main_patch",
    "branch_metadata",
    "branch_patch",
    "branch_merge_receipt",
]


def _canonical_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase, hyphenated canonical UUID4")
    return value


def _safe_line(value: str, *, label: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be one bounded nonempty line")
    if redact_server_text(value) != value:
        raise ValueError(f"{label} cannot contain credential-shaped text")
    return value


def _relative_path(value: str, *, label: str) -> str:
    _safe_line(value, label=label)
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or path in {PurePosixPath("."), PurePosixPath("")}:
        raise ValueError(f"{label} must be a nonempty relative path")
    if any(part in {"", ".", ".."} for part in path.parts) or str(path) != value:
        raise ValueError(f"{label} must be normalized and cannot traverse")
    return value


def _absolute_path(value: str, *, label: str) -> str:
    _safe_line(value, label=label)
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError(f"{label} must be a normalized absolute non-root path")
    if str(path) != value:
        raise ValueError(f"{label} must be normalized")
    return value


def _direct_entry_name(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 4096
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be one bounded direct-entry name")
    return value


def _aware_time(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return value


def _canonical_source_kind(path: PurePosixPath) -> BackupCanonicalFileKind | None:
    parts = path.parts
    if path == PurePosixPath("manifest.toml"):
        return "manifest"
    if path == PurePosixPath("scope-base.json"):
        return "scope_base"
    if len(parts) == 2 and parts[0] == "patches":
        return "main_patch" if re.fullmatch(r"[0-9]{6}\.json", parts[1]) else None
    if len(parts) == 3 and parts[0] == "patches" and parts[1].startswith("batch-"):
        return "main_patch" if re.fullmatch(r"[0-9]{6}\.json", parts[2]) else None
    if len(parts) == 3 and parts[0] == "branches":
        try:
            branch_id = uuid.UUID(parts[1])
        except ValueError:
            return None
        if branch_id.version == 4 and str(branch_id) == parts[1] and parts[2] == "branch.json":
            return "branch_metadata"
        return None
    if len(parts) != 4 or parts[0] != "branches":
        return None
    try:
        branch_id = uuid.UUID(parts[1])
    except ValueError:
        return None
    if branch_id.version != 4 or str(branch_id) != parts[1]:
        return None
    if parts[2] == "patches" and re.fullmatch(r"[0-9]{6}\.json", parts[3]):
        return "branch_patch"
    if parts[2] == "merges" and re.fullmatch(r"[0-9a-f]{64}\.json", parts[3]):
        return "branch_merge_receipt"
    return None


def _is_canonical_project_source(path: PurePosixPath) -> bool:
    if not path.parts or path.parts[0] != ".research":
        return False
    return _canonical_source_kind(PurePosixPath(*path.parts[1:])) is not None


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _StrictBackupModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


class BackupFileEntry(_StrictBackupModel):
    """One exact regular file included in the plaintext archive."""

    archive_path: str
    source_relative_path: str
    group: BackupFileGroup
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("archive_path", "source_relative_path")
    @classmethod
    def validate_relative_path(cls, value: str, info) -> str:
        return _relative_path(value, label=info.field_name.replace("_", " "))

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("backup file digest must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def forbid_materialized_and_secret_roots(self) -> BackupFileEntry:
        source = PurePosixPath(self.source_relative_path)
        if self.group == "canonical":
            if not _is_canonical_project_source(source):
                raise ValueError("canonical backup entries must be retained research inputs")
            if ".publish" in source.parts:
                raise ValueError("canonical backup entries cannot include publication staging")
        elif self.group == "chat" and not (
            len(source.parts) == 3
            and source.parts[:2] == (".research", "chat")
            and source.suffix == ".jsonl"
        ):
            raise ValueError("chat backup entries must be canonical chat JSONL files")
        elif self.group == "paper_introduction" and source != PurePosixPath(
            ".research/paper/introduction.md"
        ):
            raise ValueError("Paper backup entries must name the canonical introduction")
        elif self.group == "fact" and not (
            len(source.parts) > 2 and source.parts[:2] == (".research", "facts")
        ):
            raise ValueError("fact backup entries must stay below .research/facts")
        elif self.group == "kept_artifact" and not (
            len(source.parts) == 2 and source.parts[0] == "artifacts"
        ):
            raise ValueError("kept artifacts must be direct repository artifact files")
        elif self.group == "legacy_kept_result_view" and not (
            len(source.parts) == 2 and source.parts[0] == "views"
        ):
            raise ValueError("legacy kept views must be direct repository view files")
        elif self.group == "sqlite_snapshot" and source != PurePosixPath("rcp.sqlite3"):
            raise ValueError("the SQLite snapshot must retain its fixed source name")
        elif self.group == "imported_provider_history":
            parts = source.parts
            valid = parts == ("provider-history", "manifest.json") or (
                len(parts) == 3
                and parts[0] == "provider-history"
                and parts[1] in PROVIDERS
                and _SHA256.fullmatch(parts[2]) is not None
            )
            if not valid:
                raise ValueError("imported provider-history entries require owned sealed paths")
        forbidden = {".git", "credentials", "run-stage", "chat-attachments"}
        if forbidden.intersection(source.parts):
            raise ValueError("backup entries cannot include credentials, source Git, or stages")
        return self


class BackupImportedProviderSourceFile(_StrictBackupModel):
    provider: ProviderId
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("backup imported-source digest must be lowercase SHA-256")
        return value


class BackupImportedProviderSourceInventory(_StrictBackupModel):
    project_id: str
    files: tuple[BackupImportedProviderSourceFile, ...]
    payload_size_bytes: int = Field(ge=0)
    fingerprint: str

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="backup imported-source project identity")

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("backup imported-source fingerprint must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_inventory(self) -> BackupImportedProviderSourceInventory:
        keys = [(item.provider, item.sha256) for item in self.files]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("backup imported-source files must be sorted and unique")
        if sum(item.size_bytes for item in self.files) != self.payload_size_bytes:
            raise ValueError("backup imported-source byte total differs from its files")
        expected = _canonical_json_sha256(
            {
                "project_id": self.project_id,
                "files": [item.model_dump(mode="json") for item in self.files],
            }
        )
        if self.fingerprint != expected:
            raise ValueError("backup imported-source fingerprint differs from its files")
        return self


class BackupCanonicalSourceFile(_StrictBackupModel):
    """One observed canonical input; O2b must revalidate it while copying."""

    relative_path: str
    kind: BackupCanonicalFileKind
    observed_size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _relative_path(value, label="canonical source path")

    @model_validator(mode="after")
    def validate_kind(self) -> BackupCanonicalSourceFile:
        expected = _canonical_source_kind(PurePosixPath(self.relative_path))
        if expected is None or self.kind != expected:
            raise ValueError("canonical backup source kind does not match its path")
        return self


class BackupBranchSourcePlan(_StrictBackupModel):
    branch_id: str
    head: GraphHeadRef
    files: tuple[BackupCanonicalSourceFile, ...]

    @field_validator("branch_id")
    @classmethod
    def validate_branch_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="graph branch identity")

    @model_validator(mode="after")
    def validate_branch(self) -> BackupBranchSourcePlan:
        if self.head.target.kind != "branch" or self.head.target.branch_id != self.branch_id:
            raise ValueError("backup branch head must name its exact branch")
        if sum(item.kind == "branch_metadata" for item in self.files) != 1:
            raise ValueError("backup branch plans require their canonical metadata")
        if len({item.relative_path for item in self.files}) != len(self.files):
            raise ValueError("backup branch plans cannot repeat a canonical file")
        for item in self.files:
            parts = PurePosixPath(item.relative_path).parts
            if len(parts) < 3 or parts[:2] != ("branches", self.branch_id):
                raise ValueError("backup branch files must belong to their exact branch")
        revisions = [
            int(PurePosixPath(item.relative_path).stem)
            for item in self.files
            if item.kind == "branch_patch"
        ]
        if len(revisions) != len(set(revisions)):
            raise ValueError("backup branch plans cannot repeat a Patch revision")
        if revisions and max(revisions) != self.head.revision:
            raise ValueError("backup branch head does not match its retained Patches")
        return self


class BackupCanonicalSourcePlan(_StrictBackupModel):
    main_observed_revision: int = Field(ge=0)
    main_files: tuple[BackupCanonicalSourceFile, ...]
    branches: tuple[BackupBranchSourcePlan, ...]
    delegated_roots: tuple[str, ...]
    excluded_roots: tuple[str, ...]
    excluded_canonical_paths: tuple[str, ...]
    unclassified_roots: tuple[str, ...]
    observed_canonical_bytes: int = Field(ge=0)

    @field_validator("delegated_roots", "excluded_roots", "unclassified_roots")
    @classmethod
    def validate_root_names(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        for name in value:
            _direct_entry_name(name, label=info.field_name.replace("_", " "))
        return value

    @field_validator("excluded_canonical_paths")
    @classmethod
    def validate_excluded_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _relative_path(path, label="excluded canonical path")
        return value

    @model_validator(mode="after")
    def validate_inventory(self) -> BackupCanonicalSourcePlan:
        if sum(item.kind == "manifest" for item in self.main_files) != 1:
            raise ValueError("canonical backup inventory requires one manifest")
        if sum(item.kind == "scope_base" for item in self.main_files) > 1:
            raise ValueError("canonical backup inventory cannot repeat scope provenance")
        if any(
            item.kind not in {"manifest", "scope_base", "main_patch"} for item in self.main_files
        ):
            raise ValueError("main backup inventory cannot contain branch files")
        names = [item.relative_path for item in self.main_files]
        names.extend(item.relative_path for branch in self.branches for item in branch.files)
        if len(names) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise ValueError("canonical backup inventory exceeds its entry bound")
        if len(names) != len(set(names)):
            raise ValueError("canonical backup inventory cannot repeat a source file")
        if self.observed_canonical_bytes != sum(
            item.observed_size_bytes for item in self.main_files
        ) + sum(item.observed_size_bytes for branch in self.branches for item in branch.files):
            raise ValueError("canonical backup inventory byte total does not match its entries")
        main_revisions = [
            int(PurePosixPath(item.relative_path).stem)
            for item in self.main_files
            if item.kind == "main_patch"
        ]
        if len(main_revisions) != len(set(main_revisions)):
            raise ValueError("canonical backup inventory repeats a main Patch revision")
        if main_revisions != list(range(1, self.main_observed_revision + 1)):
            raise ValueError("canonical backup main head does not match retained Patches")
        for values, label in (
            (self.delegated_roots, "delegated research root"),
            (self.excluded_roots, "excluded research root"),
            (self.excluded_canonical_paths, "excluded canonical path"),
            (self.unclassified_roots, "unclassified research root"),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{label} names must be sorted and unique")
        branch_ids = [branch.branch_id for branch in self.branches]
        if tuple(sorted(branch_ids)) != tuple(branch_ids) or len(set(branch_ids)) != len(
            branch_ids
        ):
            raise ValueError("backup branch plans must be sorted and unique")
        return self

    @property
    def complete(self) -> bool:
        return not self.unclassified_roots


class BackupAppDataCapturePlan(_StrictBackupModel):
    data_dir: str
    database_path: str | None
    database_unavailable_reason: str | None
    excluded_entries: tuple[str, ...]
    captured_entries: tuple[str, ...] = ()
    deferred_entries: tuple[str, ...]
    unclassified_entries: tuple[str, ...]

    @field_validator(
        "excluded_entries",
        "captured_entries",
        "deferred_entries",
        "unclassified_entries",
    )
    @classmethod
    def validate_entry_names(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        for name in value:
            _direct_entry_name(name, label=info.field_name.replace("_", " "))
        return value

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, value: str) -> str:
        return _absolute_path(value, label="backup data directory")

    @field_validator("database_path")
    @classmethod
    def validate_database_path(cls, value: str | None) -> str | None:
        return None if value is None else _absolute_path(value, label="backup database path")

    @field_validator("database_unavailable_reason")
    @classmethod
    def validate_database_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_line(
            value,
            label="backup database diagnostic",
            maximum=BACKUP_DIAGNOSTIC_MAX_CHARS,
        )

    @model_validator(mode="after")
    def validate_plan(self) -> BackupAppDataCapturePlan:
        if (self.database_path is None) == (self.database_unavailable_reason is None):
            raise ValueError("backup database availability must carry exactly one result")
        for values, label in (
            (self.excluded_entries, "excluded app-data entry"),
            (self.captured_entries, "captured app-data entry"),
            (self.deferred_entries, "deferred app-data entry"),
            (self.unclassified_entries, "unclassified app-data entry"),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{label} names must be sorted and unique")
        all_names = (
            *self.excluded_entries,
            *self.captured_entries,
            *self.deferred_entries,
            *self.unclassified_entries,
        )
        if len(all_names) != len(set(all_names)):
            raise ValueError("an app-data entry cannot have two backup classifications")
        return self

    @property
    def complete(self) -> bool:
        return bool(
            self.database_path is not None
            and not self.deferred_entries
            and not self.unclassified_entries
        )


class BackupManifestMachine(_StrictBackupModel):
    alias: str
    host: str
    os_account: str
    provider_paths: dict[ProviderId, str]

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _ALIAS.fullmatch(value) is None:
            raise ValueError("backup manifest machine alias is invalid")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        if _HOST.fullmatch(value) is None or redact_server_text(value) != value:
            raise ValueError("backup manifest machine host is invalid")
        return value

    @field_validator("os_account")
    @classmethod
    def validate_account(cls, value: str) -> str:
        if _ACCOUNT.fullmatch(value) is None:
            raise ValueError("backup manifest machine account is invalid")
        return value

    @field_validator("provider_paths")
    @classmethod
    def validate_provider_paths(cls, value: dict[ProviderId, str]) -> dict[ProviderId, str]:
        for path in value.values():
            _absolute_path(path, label="backup manifest provider path")
        return value


class BackupManifestRepository(_StrictBackupModel):
    alias: str
    machine: str
    path: str

    @field_validator("alias", "machine")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _ALIAS.fullmatch(value) is None:
            raise ValueError("backup manifest repository reference is invalid")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _absolute_path(value, label="backup manifest repository path")


class BackupManifestAgentProfile(_StrictBackupModel):
    profile: AgentExecutionProfile
    provider: ProviderId
    runtime: str
    model: str
    reasoning: str
    run_on: str
    permissions: AgentPermissions

    @field_validator("runtime", "reasoning", "run_on")
    @classmethod
    def validate_bounded_field(cls, value: str, info) -> str:
        return _safe_line(value, label=f"backup agent {info.field_name}", maximum=200)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if len(value) > 200 or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("backup agent model must be one bounded line")
        if redact_server_text(value) != value:
            raise ValueError("backup agent model cannot contain credential-shaped text")
        return value


class BackupManifestSources(_StrictBackupModel):
    claude_roots: tuple[str, ...]
    codex_roots: tuple[str, ...]
    remote_claude_roots: tuple[str, ...]
    remote_codex_roots: tuple[str, ...]

    @field_validator(
        "claude_roots",
        "codex_roots",
        "remote_claude_roots",
        "remote_codex_roots",
    )
    @classmethod
    def validate_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for root in value:
            _safe_line(root, label="provider-history root")
        return value


class BackupManifestConfiguration(_StrictBackupModel):
    name: str
    machines: tuple[BackupManifestMachine, ...]
    repositories: tuple[BackupManifestRepository, ...]
    project_truth_scope: tuple[str, ...]
    state_repository: str
    default_run_truth_scope: tuple[str, ...]
    default_auto_research_invocation_ceiling: int = Field(ge=1)
    skill_defaults: SkillDefaults
    agent_profiles: tuple[BackupManifestAgentProfile, ...]
    sources: BackupManifestSources

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _safe_line(value, label="backup project name", maximum=120)

    @field_validator("state_repository")
    @classmethod
    def validate_state_repository(cls, value: str) -> str:
        if _ALIAS.fullmatch(value) is None:
            raise ValueError("backup state repository alias is invalid")
        return value

    @field_validator("project_truth_scope", "default_run_truth_scope")
    @classmethod
    def validate_scope(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(_ALIAS.fullmatch(alias) is None for alias in value):
            raise ValueError("backup repository scope is invalid")
        if len(value) != len(set(value)):
            raise ValueError("backup repository scope cannot repeat an alias")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> BackupManifestConfiguration:
        machines = {machine.alias for machine in self.machines}
        repositories = {repository.alias for repository in self.repositories}
        if len(machines) != len(self.machines) or len(repositories) != len(self.repositories):
            raise ValueError("backup manifest aliases must be unique")
        if any(repository.machine not in machines for repository in self.repositories):
            raise ValueError("backup manifest repository names an unknown machine")
        if self.state_repository not in repositories:
            raise ValueError("backup manifest state repository is unknown")
        if not set(self.project_truth_scope).issubset(repositories):
            raise ValueError("backup project scope names an unknown repository")
        if self.state_repository not in self.project_truth_scope:
            raise ValueError("backup state repository must remain in project scope")
        if not set(self.default_run_truth_scope).issubset(self.project_truth_scope):
            raise ValueError("backup default run scope must be a project subset")
        profiles = [profile.profile for profile in self.agent_profiles]
        if set(profiles) != set(AGENT_EXECUTION_PROFILES) or len(profiles) != len(set(profiles)):
            raise ValueError("backup manifest must carry every agent profile exactly once")
        if any(profile.run_on not in machines for profile in self.agent_profiles):
            raise ValueError("backup agent profile names an unknown machine")
        return self

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> BackupManifestConfiguration:
        return cls(
            name=manifest.name,
            machines=tuple(
                BackupManifestMachine(
                    alias=machine.alias,
                    host=machine.host,
                    os_account=machine.os_account,
                    provider_paths=dict(machine.provider_paths),
                )
                for machine in manifest.machines
            ),
            repositories=tuple(
                BackupManifestRepository(
                    alias=repository.alias,
                    machine=repository.machine,
                    path=repository.path,
                )
                for repository in manifest.repositories
            ),
            project_truth_scope=tuple(manifest.project.truth_scope),
            state_repository=manifest.state.repository,
            default_run_truth_scope=tuple(manifest.agent.default_run_truth_scope),
            default_auto_research_invocation_ceiling=(
                manifest.agent.default_auto_research_invocation_ceiling
            ),
            skill_defaults=manifest.agent.skill_defaults,
            agent_profiles=tuple(
                BackupManifestAgentProfile(
                    profile=profile,
                    provider=manifest.agent_profile(profile).provider,
                    runtime=manifest.agent_profile(profile).runtime,
                    model=manifest.agent_profile(profile).model,
                    reasoning=manifest.agent_profile(profile).reasoning,
                    run_on=manifest.agent_profile(profile).run_on,
                    permissions=manifest.agent_profile(profile).permissions,
                )
                for profile in AGENT_EXECUTION_PROFILES
            ),
            sources=BackupManifestSources(
                claude_roots=tuple(manifest.sources.claude_roots),
                codex_roots=tuple(manifest.sources.codex_roots),
                remote_claude_roots=tuple(manifest.sources.remote_claude_roots),
                remote_codex_roots=tuple(manifest.sources.remote_codex_roots),
            ),
        )

    @property
    def sha256(self) -> str:
        return _canonical_json_sha256(self.model_dump(mode="json"))


class BackupRecoveryMachine(_StrictBackupModel):
    alias: str
    location: Literal["local", "ssh"]
    host: str
    os_account: str
    resolved_central_root: str

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _ALIAS.fullmatch(value) is None:
            raise ValueError("backup recovery machine alias is invalid")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        if _HOST.fullmatch(value) is None or redact_server_text(value) != value:
            raise ValueError("backup recovery machine host is invalid")
        return value

    @field_validator("os_account")
    @classmethod
    def validate_account(cls, value: str) -> str:
        if _ACCOUNT.fullmatch(value) is None:
            raise ValueError("backup recovery machine account is invalid")
        return value

    @field_validator("resolved_central_root")
    @classmethod
    def validate_root(cls, value: str) -> str:
        return _absolute_path(value, label="backup recovery central root")

    @model_validator(mode="after")
    def validate_location(self) -> BackupRecoveryMachine:
        if self.location == "local" and self.host:
            raise ValueError("a local backup recovery machine cannot name an SSH host")
        if self.location == "ssh" and not self.host:
            raise ValueError("an SSH backup recovery machine requires its route host")
        return self


class BackupRecoveryRepository(_StrictBackupModel):
    alias: str
    repository: GitHubRepositoryRef
    machine_alias: str
    resolved_path: str
    git_commit: str
    deploy_key_label: str
    public_key_fingerprint: str

    @field_validator("alias", "machine_alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _ALIAS.fullmatch(value) is None:
            raise ValueError("backup recovery repository reference is invalid")
        return value

    @field_validator("resolved_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _absolute_path(value, label="backup recovery repository path")

    @field_validator("git_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("backup recovery Git commit must be a full lowercase object id")
        return value

    @field_validator("deploy_key_label")
    @classmethod
    def validate_key_label(cls, value: str) -> str:
        return _safe_line(value, label="backup recovery deploy-key label", maximum=255)

    @field_validator("public_key_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if _OPENSSH_FINGERPRINT.fullmatch(value) is None:
            raise ValueError("backup recovery public-key fingerprint is invalid")
        return value


class BackupCheckoutRecoveryDescriptor(_StrictBackupModel):
    request_id: str
    project_id: str
    home_space_id: str
    completed_at: datetime
    final_review_digest: str
    configuration: BackupManifestConfiguration
    configuration_sha256: str
    machines: tuple[BackupRecoveryMachine, ...]
    repositories: tuple[BackupRecoveryRepository, ...]

    @field_validator("request_id", "project_id", "home_space_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: datetime) -> datetime:
        return _aware_time(value, label="provisioning completion time")

    @field_validator("final_review_digest", "configuration_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{info.field_name.replace('_', ' ')} must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_descriptor(self) -> BackupCheckoutRecoveryDescriptor:
        if self.configuration_sha256 != self.configuration.sha256:
            raise ValueError("backup manifest configuration digest does not match its payload")
        machine_map = {machine.alias: machine for machine in self.machines}
        repository_map = {repository.alias: repository for repository in self.repositories}
        if len(machine_map) != len(self.machines) or len(repository_map) != len(self.repositories):
            raise ValueError("backup recovery aliases must be unique")
        configured_machines = {machine.alias: machine for machine in self.configuration.machines}
        configured_repositories = {
            repository.alias: repository for repository in self.configuration.repositories
        }
        if machine_map.keys() != configured_machines.keys():
            raise ValueError("backup recovery machines differ from canonical configuration")
        if repository_map.keys() != configured_repositories.keys():
            raise ValueError("backup recovery repositories differ from canonical configuration")
        for alias, machine in machine_map.items():
            configured = configured_machines[alias]
            if (machine.host, machine.os_account) != (
                configured.host,
                configured.os_account,
            ):
                raise ValueError("backup recovery route differs from canonical configuration")
        for alias, repository in repository_map.items():
            configured = configured_repositories[alias]
            if (
                repository.machine_alias != configured.machine
                or repository.resolved_path != configured.path
            ):
                raise ValueError("backup recovery checkout differs from canonical configuration")
            machine = machine_map[repository.machine_alias]
            expected = (
                PurePosixPath(machine.resolved_central_root)
                / self.project_id
                / "repositories"
                / alias
            )
            if repository.resolved_path != str(expected):
                raise ValueError("backup recovery checkout is not derived from its central root")
            expected_label = f"rcp:{self.home_space_id}:{self.project_id}:{alias}"
            if repository.deploy_key_label != expected_label:
                raise ValueError("backup recovery deploy-key label is not derived")
        return self


class BackupProjectCapture(_StrictBackupModel):
    project_id: str
    home_space_id: str | None
    locator: str | None
    status: Literal["captured", "uncaptured"]
    main_head: GraphHeadRef | None = None
    branch_heads: tuple[GraphHeadRef, ...] = ()
    files: tuple[BackupFileEntry, ...] = ()
    recovery: BackupCheckoutRecoveryDescriptor | None = None
    unavailable_kind: (
        Literal[
            "inventory_failure",
            "remote_unreachable",
            "capture_failure",
        ]
        | None
    ) = None
    unavailable_reason: str | None = None
    unavailable_at: datetime | None = None
    total_bytes: int = Field(ge=0)

    @field_validator("project_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("home_space_id")
    @classmethod
    def validate_home_space_id(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_uuid4(value, label="home space identity")

    @field_validator("locator")
    @classmethod
    def validate_locator(cls, value: str | None) -> str | None:
        return None if value is None else _absolute_path(value, label="project locator")

    @field_validator("unavailable_reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_line(
            value,
            label="project backup diagnostic",
            maximum=BACKUP_DIAGNOSTIC_MAX_CHARS,
        )

    @field_validator("unavailable_at")
    @classmethod
    def validate_unavailable_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_time(value, label="project unavailable time")

    @model_validator(mode="after")
    def validate_capture(self) -> BackupProjectCapture:
        if len(self.files) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise ValueError("project backup file inventory exceeds its entry bound")
        if self.status == "captured":
            if (
                self.main_head is None
                or self.locator is None
                or self.home_space_id is None
                or self.main_head.target.kind != "main"
                or self.recovery is None
                or self.unavailable_kind is not None
                or self.unavailable_reason is not None
                or self.unavailable_at is not None
            ):
                raise ValueError("a captured project requires heads and recovery without failure")
            _absolute_path(self.locator, label="project locator")
            if (
                self.recovery.project_id != self.project_id
                or self.recovery.home_space_id != self.home_space_id
            ):
                raise ValueError("project capture and recovery identity differ")
            if not any(
                entry.group == "canonical"
                and entry.source_relative_path == ".research/manifest.toml"
                for entry in self.files
            ):
                raise ValueError("a captured project requires its canonical manifest")
        else:
            if any(
                (
                    self.main_head is not None,
                    bool(self.branch_heads),
                    bool(self.files),
                    self.unavailable_reason is None,
                    self.unavailable_at is None,
                )
            ):
                raise ValueError("an uncaptured project carries only its failure proof")
            if self.unavailable_kind == "remote_unreachable":
                if self.recovery is None or self.locator is None or self.home_space_id is None:
                    raise ValueError(
                        "an unreachable remote project requires its captured recovery descriptor"
                    )
            elif self.recovery is not None:
                raise ValueError(
                    "only an unreachable remote project may retain its recovery descriptor"
                )
        branch_ids: list[str] = []
        for head in self.branch_heads:
            if head.target.kind != "branch" or head.target.branch_id is None:
                raise ValueError("project branch heads must name graph branches")
            branch_ids.append(head.target.branch_id)
        if tuple(sorted(branch_ids)) != tuple(branch_ids) or len(set(branch_ids)) != len(
            branch_ids
        ):
            raise ValueError("project branch heads must be sorted and unique")
        paths = [entry.archive_path for entry in self.files]
        sources = [entry.source_relative_path for entry in self.files]
        if len(paths) != len(set(paths)) or len(sources) != len(set(sources)):
            raise ValueError("project backup files cannot repeat a source or archive path")
        if any(entry.group == "sqlite_snapshot" for entry in self.files):
            raise ValueError("project captures cannot contain the space database snapshot")
        main_revisions = sorted(
            int(PurePosixPath(entry.source_relative_path).stem)
            for entry in self.files
            if entry.group == "canonical"
            and len(PurePosixPath(entry.source_relative_path).parts) in {3, 4}
            and PurePosixPath(entry.source_relative_path).parts[:2] == (".research", "patches")
        )
        if self.main_head is not None and main_revisions != list(
            range(1, self.main_head.revision + 1)
        ):
            raise ValueError("project main head does not match its captured Patches")
        metadata_branches = {
            PurePosixPath(entry.source_relative_path).parts[2]
            for entry in self.files
            if entry.group == "canonical"
            and len(PurePosixPath(entry.source_relative_path).parts) == 4
            and PurePosixPath(entry.source_relative_path).parts[:2] == (".research", "branches")
        }
        if metadata_branches != set(branch_ids):
            raise ValueError("project branch heads and canonical metadata differ")
        branch_head_map = {
            head.target.branch_id: head.revision
            for head in self.branch_heads
            if head.target.branch_id is not None
        }
        branch_revisions: dict[str, list[int]] = {}
        merge_branches: set[str] = set()
        for entry in self.files:
            source = PurePosixPath(entry.source_relative_path)
            if (
                entry.group == "canonical"
                and len(source.parts) == 5
                and source.parts[:2] == (".research", "branches")
                and source.parts[3] == "patches"
            ):
                branch_revisions.setdefault(source.parts[2], []).append(int(source.stem))
            elif (
                entry.group == "canonical"
                and len(source.parts) == 5
                and source.parts[:2] == (".research", "branches")
                and source.parts[3] == "merges"
            ):
                merge_branches.add(source.parts[2])
        if not set(branch_revisions).issubset(metadata_branches):
            raise ValueError("project branch Patches require their captured metadata")
        if not merge_branches.issubset(metadata_branches):
            raise ValueError("project branch merge receipts require their captured metadata")
        for branch_id, revisions in branch_revisions.items():
            if len(revisions) != len(set(revisions)):
                raise ValueError("project branch history repeats a Patch revision")
            if max(revisions) != branch_head_map[branch_id]:
                raise ValueError("project branch head does not match its captured Patches")
        if sum(entry.size_bytes for entry in self.files) != self.total_bytes:
            raise ValueError("project backup byte total does not match its files")
        return self


class BackupImportedProviderSourceCapture(_StrictBackupModel):
    project_id: str
    inventory: BackupImportedProviderSourceInventory
    present: bool
    files: tuple[BackupFileEntry, ...]
    total_bytes: int = Field(ge=0)

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="imported provider source project identity")

    @model_validator(mode="after")
    def validate_capture(self) -> BackupImportedProviderSourceCapture:
        if self.inventory.project_id != self.project_id:
            raise ValueError("backup imported-source inventory names another project")
        if not self.present:
            if self.inventory.files or self.files or self.total_bytes:
                raise ValueError("an absent imported-source root cannot carry files")
            return self
        expected_sources = {"provider-history/manifest.json"}
        expected_sources.update(
            f"provider-history/{item.provider}/{item.sha256}" for item in self.inventory.files
        )
        sources = {entry.source_relative_path for entry in self.files}
        if sources != expected_sources or len(sources) != len(self.files):
            raise ValueError("backup imported-source files differ from their sealed inventory")
        expected_archive_prefix = f"project-sources/{self.project_id}/"
        if any(
            entry.group != "imported_provider_history"
            or entry.archive_path != expected_archive_prefix + entry.source_relative_path
            for entry in self.files
        ):
            raise ValueError("backup imported-source archive paths escaped their project owner")
        inventory_files = {
            f"provider-history/{item.provider}/{item.sha256}": item for item in self.inventory.files
        }
        for entry in self.files:
            if entry.source_relative_path == "provider-history/manifest.json":
                payload = self.inventory.model_dump_json(indent=2).encode() + b"\n"
                if (entry.sha256, entry.size_bytes) != (
                    hashlib.sha256(payload).hexdigest(),
                    len(payload),
                ):
                    raise ValueError("backup imported-source manifest bytes are not canonical")
                continue
            item = inventory_files[entry.source_relative_path]
            if (entry.sha256, entry.size_bytes) != (item.sha256, item.size_bytes):
                raise ValueError("backup imported-source bytes differ from their inventory")
        if sum(entry.size_bytes for entry in self.files) != self.total_bytes:
            raise ValueError("backup imported-source byte total differs from its files")
        return self


class BackupArchiveManifest(_StrictBackupModel):
    schema_version: Literal[BACKUP_MANIFEST_SCHEMA_VERSION] = BACKUP_MANIFEST_SCHEMA_VERSION
    space_id: str
    space_name: str
    rcp_source_commit: str
    database_schema_sha256: str
    captured_at: datetime
    sqlite_snapshot: BackupFileEntry
    encryption_recipient_fingerprint: str
    installation_id: str
    source_deploy_key_label: str | None = None
    source_public_key_fingerprint: str | None = None
    excluded_app_data_entries: tuple[str, ...]
    captured_app_data_entries: tuple[str, ...] = ()
    uncaptured_app_data_entries: tuple[str, ...]
    projects: tuple[BackupProjectCapture, ...]
    imported_sources: tuple[BackupImportedProviderSourceCapture, ...] = ()
    status: Literal["complete", "partial"]
    total_bytes: int = Field(ge=0)

    @field_validator(
        "excluded_app_data_entries",
        "captured_app_data_entries",
        "uncaptured_app_data_entries",
    )
    @classmethod
    def validate_app_data_names(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        for name in value:
            _direct_entry_name(name, label=info.field_name.replace("_", " "))
        return value

    @field_validator("space_id", "installation_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("space_name")
    @classmethod
    def validate_space_name(cls, value: str) -> str:
        return _safe_line(value, label="backup space name", maximum=120)

    @field_validator("rcp_source_commit")
    @classmethod
    def validate_source_commit(cls, value: str) -> str:
        if _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("backup RCP source commit must be a full lowercase object id")
        return value

    @field_validator("database_schema_sha256", "encryption_recipient_fingerprint")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{info.field_name.replace('_', ' ')} must be lowercase SHA-256")
        return value

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        return _aware_time(value, label="backup capture time")

    @field_validator("source_deploy_key_label")
    @classmethod
    def validate_source_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_line(value, label="source deploy-key label", maximum=255)

    @field_validator("source_public_key_fingerprint")
    @classmethod
    def validate_source_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and _OPENSSH_FINGERPRINT.fullmatch(value) is None:
            raise ValueError("source public-key fingerprint is invalid")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> BackupArchiveManifest:
        if self.sqlite_snapshot.group != "sqlite_snapshot":
            raise ValueError("backup manifest requires one SQLite snapshot entry")
        expected_source_label = f"rcp-source:{self.installation_id}"
        if (self.source_deploy_key_label is None) != (self.source_public_key_fingerprint is None):
            raise ValueError("source deploy-key revocation fields must be both present or absent")
        if (
            self.source_deploy_key_label is not None
            and self.source_deploy_key_label != expected_source_label
        ):
            raise ValueError("source deploy-key label is not derived from the installation")
        project_ids = [project.project_id for project in self.projects]
        if len(project_ids) != len(set(project_ids)):
            raise ValueError("backup manifest cannot repeat a project")
        if any(
            project.status == "captured" and project.home_space_id != self.space_id
            for project in self.projects
        ):
            raise ValueError("backup manifest cannot capture a project from another space")
        imported_ids = [capture.project_id for capture in self.imported_sources]
        if (
            tuple(sorted(imported_ids)) != tuple(imported_ids)
            or len(imported_ids) != len(set(imported_ids))
            or not set(imported_ids).issubset(project_ids)
        ):
            raise ValueError("backup imported-source captures must name unique archived projects")
        partial = bool(
            self.uncaptured_app_data_entries
            or any(project.status == "uncaptured" for project in self.projects)
        )
        if self.status != ("partial" if partial else "complete"):
            raise ValueError("backup manifest status does not match its capture results")
        entries = [self.sqlite_snapshot]
        entries.extend(entry for project in self.projects for entry in project.files)
        entries.extend(entry for capture in self.imported_sources for entry in capture.files)
        paths = [entry.archive_path for entry in entries]
        if len(paths) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise ValueError("backup manifest exceeds its entry bound")
        if len(paths) != len(set(paths)):
            raise ValueError("backup archive paths must be globally unique")
        if sum(entry.size_bytes for entry in entries) != self.total_bytes:
            raise ValueError("backup manifest byte total does not match its entries")
        for values, label in (
            (self.excluded_app_data_entries, "excluded app-data entry"),
            (self.captured_app_data_entries, "captured app-data entry"),
            (self.uncaptured_app_data_entries, "uncaptured app-data entry"),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{label} names must be sorted and unique")
        classified = (
            *self.excluded_app_data_entries,
            *self.captured_app_data_entries,
            *self.uncaptured_app_data_entries,
        )
        if len(classified) != len(set(classified)):
            raise ValueError("an app-data entry cannot have multiple archive classifications")
        if (
            any(capture.present for capture in self.imported_sources)
            and "project-sources" not in self.captured_app_data_entries
        ):
            raise ValueError("imported provider sources require an explicit captured app-data root")
        return self


def inspect_app_data_capture_plan(data_dir: Path) -> BackupAppDataCapturePlan:
    """Classify every direct app-data child without following or copying it."""

    root = data_dir.resolve()
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"Could not inspect the backup data directory: {exc}") from exc
    if not stat.S_ISDIR(root_mode):
        raise ValueError("The backup data directory is not a regular directory.")
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ValueError(f"Could not enumerate the backup data directory: {exc}") from exc
    if len(entries) > BACKUP_INVENTORY_MAX_ENTRIES:
        raise ValueError("The backup app-data inventory exceeds its entry bound.")

    excluded: list[str] = []
    captured: list[str] = []
    deferred: list[str] = []
    unclassified: list[str] = []
    database_path: str | None = None
    database_reason: str | None = None
    for entry in entries:
        if entry.name == BACKUP_APP_DATA_DATABASE:
            try:
                mode = entry.lstat().st_mode
            except OSError:
                database_reason = "The application database could not be inspected."
            else:
                if stat.S_ISREG(mode):
                    database_path = str(entry)
                else:
                    database_reason = "The application database is not a safe regular file."
        elif entry.name in BACKUP_APP_DATA_EXCLUSIONS:
            excluded.append(entry.name)
        elif entry.name in BACKUP_APP_DATA_CAPTURED:
            captured.append(entry.name)
        elif entry.name in BACKUP_APP_DATA_DEFERRED:
            deferred.append(entry.name)
        else:
            unclassified.append(entry.name)
    if database_path is None and database_reason is None:
        database_reason = "The application database is missing."
    return BackupAppDataCapturePlan(
        data_dir=str(root),
        database_path=database_path,
        database_unavailable_reason=database_reason,
        excluded_entries=tuple(excluded),
        captured_entries=tuple(captured),
        deferred_entries=tuple(deferred),
        unclassified_entries=tuple(unclassified),
    )


__all__ = [
    "BACKUP_APP_DATA_DATABASE",
    "BACKUP_APP_DATA_CAPTURED",
    "BACKUP_APP_DATA_DEFERRED",
    "BACKUP_APP_DATA_EXCLUSIONS",
    "BACKUP_MANIFEST_SCHEMA_VERSION",
    "BACKUP_MATERIALIZED_NAMES",
    "BACKUP_RESEARCH_CANONICAL_ROOTS",
    "BACKUP_RESEARCH_DELEGATED_ROOTS",
    "BACKUP_RESEARCH_EXCLUSIONS",
    "BackupAppDataCapturePlan",
    "BackupArchiveManifest",
    "BackupBranchSourcePlan",
    "BackupCanonicalSourceFile",
    "BackupCanonicalSourcePlan",
    "BackupCheckoutRecoveryDescriptor",
    "BackupFileEntry",
    "BackupImportedProviderSourceFile",
    "BackupImportedProviderSourceInventory",
    "BackupImportedProviderSourceCapture",
    "BackupManifestConfiguration",
    "BackupProjectCapture",
    "BackupRecoveryMachine",
    "BackupRecoveryRepository",
    "inspect_app_data_capture_plan",
]
