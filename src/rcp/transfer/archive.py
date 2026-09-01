"""Versioned project-transfer manifest and closed source inventories."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import stat
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from rcp.core.models import DISPLAY_NAME_MAX_LENGTH, AuthorizedHuman, normalize_display_name
from rcp.core.transition_models import GraphHeadRef
from rcp.limits import (
    PROJECT_TRANSFER_DIAGNOSTIC_MAX_CHARS,
    PROJECT_TRANSFER_DIAGNOSTIC_MAX_COUNT,
    PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES,
    PROJECT_TRANSFER_MANIFEST_MAX_BYTES,
)
from rcp.server_ops.models import redact_server_text

TRANSFER_ARCHIVE_SCHEMA_VERSION = 1
TRANSFER_ARCHIVE_CODEC = "rcp-transfer-v1"

# These are transfer-specific classifications, not a generic file-root registry.
# Their tests compare them with the concrete backup/root owners so a new durable
# root cannot silently enter or disappear from transfer policy.
TRANSFER_APP_DATA_TYPED_ROOTS = frozenset({"rcp.sqlite3"})
TRANSFER_APP_DATA_PROJECT_SOURCE_ROOTS = frozenset({"project-sources"})
TRANSFER_APP_DATA_CONTROL_ROOTS = frozenset({"transfer-exports", "transfer-inbox"})
TRANSFER_APP_DATA_EXCLUDED_ROOTS = frozenset(
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
    }
)
TRANSFER_RESEARCH_PROVENANCE_ROOTS = frozenset({"manifest.toml"})
TRANSFER_RESEARCH_CANONICAL_ROOTS = frozenset({"branches", "patches", "scope-base.json"})
TRANSFER_RESEARCH_DELEGATED_ROOTS = frozenset({"chat", "facts", "paper"})
TRANSFER_RESEARCH_EXCLUDED_ROOTS = frozenset(
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
TRANSFER_GLOBAL_TABLES = frozenset(
    {
        "provider_skill_inventories",
        "space_identity",
        "space_users",
        "team_bootstrap_codes",
        "team_invitations",
        "team_member_tokens",
        "team_sessions",
    }
)

TransferArchiveGroup = Literal[
    "source_manifest_provenance",
    "canonical_history",
    "operational_records",
    "rcp_chat",
    "paper_introduction",
    "fact",
    "kept_artifact",
    "legacy_kept_result_view",
    "provider_history",
    "source_release_proof",
]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}")
_DIAGNOSTIC_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_PROJECT_LINK_COLUMNS = frozenset({"project_id", "canonical_project_id", "proposed_project_id"})


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


def _relative_path(value: str, *, label: str) -> PurePosixPath:
    _safe_line(value, label=label)
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path in {PurePosixPath("."), PurePosixPath("")}
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ValueError(f"{label} must be one normalized relative path")
    return path


def _canonical_history_path(path: PurePosixPath) -> bool:
    parts = path.parts
    if path == PurePosixPath("canonical/scope-base.json"):
        return True
    if len(parts) == 3 and parts[:2] == ("canonical", "patches"):
        return re.fullmatch(r"[0-9]{6}\.json", parts[2]) is not None
    if len(parts) == 4 and parts[:2] == ("canonical", "patches"):
        return (
            parts[2].startswith("batch-") and re.fullmatch(r"[0-9]{6}\.json", parts[3]) is not None
        )
    if len(parts) not in {4, 5} or parts[:2] != ("canonical", "branches"):
        return False
    try:
        branch_id = uuid.UUID(parts[2])
    except ValueError:
        return False
    if branch_id.version != 4 or str(branch_id) != parts[2]:
        return False
    if len(parts) == 4:
        return parts[3] == "branch.json"
    if parts[3] == "patches":
        return re.fullmatch(r"[0-9]{6}\.json", parts[4]) is not None
    if parts[3] == "merges":
        return _SHA256.fullmatch(PurePosixPath(parts[4]).stem) is not None and parts[4].endswith(
            ".json"
        )
    return False


def _main_patch_revision(path: PurePosixPath) -> int | None:
    if len(path.parts) in {3, 4} and path.parts[:2] == ("canonical", "patches"):
        return int(path.stem)
    return None


def _branch_entry(path: PurePosixPath) -> tuple[str, str, int | None] | None:
    if len(path.parts) not in {4, 5} or path.parts[:2] != ("canonical", "branches"):
        return None
    branch_id = path.parts[2]
    if len(path.parts) == 4 and path.parts[3] == "branch.json":
        return branch_id, "metadata", None
    if len(path.parts) == 5 and path.parts[3] == "patches":
        return branch_id, "patch", int(path.stem)
    if len(path.parts) == 5 and path.parts[3] == "merges":
        return branch_id, "merge", None
    return None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


class _StrictTransferModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )


class TransferArchiveEntry(_StrictTransferModel):
    """One exact payload file, never an executable source path or session binding."""

    archive_path: str
    group: TransferArchiveGroup
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("archive_path")
    @classmethod
    def validate_archive_path(cls, value: str) -> str:
        _relative_path(value, label="transfer archive path")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("transfer entry digest must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_group_path(self) -> TransferArchiveEntry:
        path = PurePosixPath(self.archive_path)
        parts = path.parts
        valid = False
        if self.group == "source_manifest_provenance":
            valid = path == PurePosixPath("provenance/manifest.toml")
        elif self.group == "canonical_history":
            valid = _canonical_history_path(path)
        elif self.group == "operational_records":
            valid = len(parts) == 2 and parts[0] == "records" and path.suffix == ".jsonl"
        elif self.group == "rcp_chat":
            valid = len(parts) == 2 and parts[0] == "chats" and path.suffix == ".jsonl"
        elif self.group == "paper_introduction":
            valid = path == PurePosixPath("paper/introduction.md")
        elif self.group == "fact":
            valid = len(parts) >= 2 and parts[0] == "facts"
        elif self.group == "kept_artifact":
            valid = len(parts) == 2 and parts[0] == "artifacts"
        elif self.group == "legacy_kept_result_view":
            valid = len(parts) == 2 and parts[0] == "result-views"
        elif self.group == "provider_history":
            valid = (
                len(parts) == 3
                and parts[0] == "provider-history"
                and _SAFE_NAME.fullmatch(parts[1]) is not None
                and _SHA256.fullmatch(parts[2]) is not None
                and parts[2] == self.sha256
            )
        elif self.group == "source_release_proof":
            valid = path == PurePosixPath("control/source-release-proof.bin")
        if not valid:
            raise ValueError("transfer entry path does not match its declared group")
        if {".git", "credentials", "run-stage", "chat-attachments"}.intersection(parts):
            raise ValueError("transfer entries cannot contain Git, credentials, stages, or inputs")
        return self


class TransferGraphTarget(_StrictTransferModel):
    kind: Literal["main", "branch"] = "main"
    branch_id: str | None = None

    @field_validator("branch_id")
    @classmethod
    def validate_branch_id(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_uuid4(value, label="graph branch identity")

    @model_validator(mode="after")
    def validate_target(self) -> TransferGraphTarget:
        if self.kind == "main" and self.branch_id is not None:
            raise ValueError("main transfer heads cannot carry a branch identity")
        if self.kind == "branch" and self.branch_id is None:
            raise ValueError("branch transfer heads require a branch identity")
        return self


class TransferGraphHead(_StrictTransferModel):
    target: TransferGraphTarget = Field(default_factory=TransferGraphTarget)
    revision: int = Field(ge=0)
    transition_id: str | None = None

    @field_validator("transition_id")
    @classmethod
    def validate_transition_id(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("transfer graph transition identity must be lowercase SHA-256")
        return value

    @classmethod
    def capture(cls, head: GraphHeadRef) -> TransferGraphHead:
        return cls.model_validate_json(head.model_dump_json())


class TransferArchiveActor(_StrictTransferModel):
    space_id: str
    user_id: str
    display_name: str = Field(min_length=1, max_length=DISPLAY_NAME_MAX_LENGTH)

    @field_validator("space_id", "user_id")
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("display_name", mode="before")
    @classmethod
    def validate_display_name(cls, value: object) -> str:
        return normalize_display_name(value)

    @classmethod
    def capture(cls, actor: AuthorizedHuman) -> TransferArchiveActor:
        return cls.model_validate_json(actor.model_dump_json())


class TransferArchiveAttribution(_StrictTransferModel):
    archive_actor_id: str
    source_actor: TransferArchiveActor

    @field_validator("archive_actor_id")
    @classmethod
    def validate_archive_actor_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="archive actor identity")


class TransferArchiveDiagnostic(_StrictTransferModel):
    severity: Literal["warning"] = "warning"
    code: str
    message: str

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if _DIAGNOSTIC_CODE.fullmatch(value) is None:
            raise ValueError("transfer diagnostic code is invalid")
        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _safe_line(
            value,
            label="transfer diagnostic",
            maximum=PROJECT_TRANSFER_DIAGNOSTIC_MAX_CHARS,
        )


class TransferArchiveManifest(_StrictTransferModel):
    schema_version: Literal[TRANSFER_ARCHIVE_SCHEMA_VERSION] = TRANSFER_ARCHIVE_SCHEMA_VERSION
    archive_codec: Literal[TRANSFER_ARCHIVE_CODEC] = TRANSFER_ARCHIVE_CODEC
    project_id: str
    source_space_id: str
    target_space_id: str
    source_request_id: str
    target_request_id: str
    source_rcp_version: str
    source_schema_generation: int = Field(ge=1)
    source_configuration_sha256: str
    source_manifest_sha256: str
    source_release_proof_sha256: str
    target_activation_proof_sha256: str
    main_head: TransferGraphHead
    branch_heads: tuple[TransferGraphHead, ...]
    attributions: tuple[TransferArchiveAttribution, ...]
    diagnostics: tuple[TransferArchiveDiagnostic, ...]
    entries: tuple[TransferArchiveEntry, ...]
    payload_size_bytes: int = Field(ge=0)
    created_at: datetime

    @field_validator(
        "project_id",
        "source_space_id",
        "target_space_id",
        "source_request_id",
        "target_request_id",
    )
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator(
        "source_configuration_sha256",
        "source_manifest_sha256",
        "source_release_proof_sha256",
        "target_activation_proof_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("transfer manifest digest must be lowercase SHA-256")
        return value

    @field_validator("source_rcp_version")
    @classmethod
    def validate_source_version(cls, value: str) -> str:
        return _safe_line(value, label="source RCP version", maximum=255)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("transfer manifest time must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> TransferArchiveManifest:
        if self.source_space_id == self.target_space_id:
            raise ValueError("transfer archive must cross spaces")
        if self.main_head.target.kind != "main":
            raise ValueError("transfer manifest main head must name the main graph")
        if not self.attributions:
            raise ValueError("transfer manifest requires explicit historical attribution")
        actor_ids = [item.archive_actor_id for item in self.attributions]
        source_actors = [
            (item.source_actor.space_id, item.source_actor.user_id) for item in self.attributions
        ]
        if any(
            item.source_actor.space_id not in {self.source_space_id, self.target_space_id}
            for item in self.attributions
        ):
            raise ValueError("transfer attribution belongs to an unrelated space")
        if len(actor_ids) != len(set(actor_ids)) or len(source_actors) != len(set(source_actors)):
            raise ValueError("transfer attribution mappings must be one-to-one")
        if tuple(sorted(actor_ids)) != tuple(actor_ids):
            raise ValueError("transfer attribution mappings must be sorted")
        if len(self.diagnostics) > PROJECT_TRANSFER_DIAGNOSTIC_MAX_COUNT:
            raise ValueError("transfer manifest has too many diagnostics")
        if len(self.entries) > PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES:
            raise ValueError("transfer manifest exceeds its entry bound")
        paths = [entry.archive_path for entry in self.entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("transfer archive paths must be sorted and unique")
        if sum(entry.size_bytes for entry in self.entries) != self.payload_size_bytes:
            raise ValueError("transfer payload byte total does not match its entries")
        provenance = [
            entry for entry in self.entries if entry.group == "source_manifest_provenance"
        ]
        if len(provenance) != 1 or provenance[0].sha256 != self.source_manifest_sha256:
            raise ValueError("transfer manifest must bind one source-manifest provenance entry")
        controls = [entry for entry in self.entries if entry.group == "source_release_proof"]
        if (
            len(controls) != 1
            or controls[0].size_bytes != 32
            or controls[0].sha256 != self.source_release_proof_sha256
        ):
            raise ValueError("transfer manifest must bind the exact source-release proof entry")
        self._validate_canonical_heads()
        return self

    def _validate_canonical_heads(self) -> None:
        canonical = [entry for entry in self.entries if entry.group == "canonical_history"]
        paths = [PurePosixPath(entry.archive_path) for entry in canonical]
        if paths.count(PurePosixPath("canonical/scope-base.json")) != 1:
            raise ValueError("transfer canonical history requires one scope provenance file")
        main_revisions = sorted(
            revision for path in paths if (revision := _main_patch_revision(path)) is not None
        )
        if main_revisions != list(range(1, self.main_head.revision + 1)):
            raise ValueError("transfer main head does not match its retained Patches")
        branch_ids = [head.target.branch_id for head in self.branch_heads]
        if any(
            head.target.kind != "branch" or head.target.branch_id is None
            for head in self.branch_heads
        ):
            raise ValueError("transfer branch heads must name exact graph branches")
        if branch_ids != sorted(branch_ids) or len(branch_ids) != len(set(branch_ids)):
            raise ValueError("transfer branch heads must be sorted and unique")
        observed: dict[str, dict[str, object]] = {}
        for path in paths:
            item = _branch_entry(path)
            if item is None:
                continue
            branch_id, kind, revision = item
            values = observed.setdefault(branch_id, {"metadata": 0, "patches": []})
            if kind == "metadata":
                values["metadata"] = int(values["metadata"]) + 1
            elif kind == "patch":
                patches = values["patches"]
                assert isinstance(patches, list)
                patches.append(revision)
        if set(observed) != set(branch_ids):
            raise ValueError("transfer branch heads do not match canonical branch entries")
        heads = {head.target.branch_id: head for head in self.branch_heads}
        for branch_id, values in observed.items():
            patches = sorted(values["patches"])
            assert isinstance(patches, list)
            contiguous_tail = not patches or patches == list(
                range(patches[0], heads[branch_id].revision + 1)
            )
            if values["metadata"] != 1 or not contiguous_tail:
                raise ValueError("transfer branch head does not match its retained inputs")

    def canonical_bytes(self) -> bytes:
        payload = _canonical_json_bytes(self.model_dump(mode="json"))
        if len(payload) > PROJECT_TRANSFER_MANIFEST_MAX_BYTES:
            raise ValueError("transfer manifest exceeds its byte bound")
        return payload

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class TransferArchiveEnvelope(_StrictTransferModel):
    """External seal receipt for the exact encoded archive bytes."""

    archive_codec: Literal[TRANSFER_ARCHIVE_CODEC] = TRANSFER_ARCHIVE_CODEC
    manifest_sha256: str
    manifest_size_bytes: int = Field(ge=1)
    payload_size_bytes: int = Field(ge=0)
    archive_sha256: str
    archive_size_bytes: int = Field(ge=1)

    @field_validator("manifest_sha256", "archive_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("transfer envelope digest must be lowercase SHA-256")
        return value

    @classmethod
    def bind(
        cls,
        manifest: TransferArchiveManifest,
        *,
        archive_sha256: str,
        archive_size_bytes: int,
    ) -> TransferArchiveEnvelope:
        manifest_bytes = manifest.canonical_bytes()
        return cls(
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            manifest_size_bytes=len(manifest_bytes),
            payload_size_bytes=manifest.payload_size_bytes,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size_bytes,
        )

    def verify_manifest(self, manifest: TransferArchiveManifest) -> None:
        manifest_bytes = manifest.canonical_bytes()
        if (
            self.manifest_sha256 != hashlib.sha256(manifest_bytes).hexdigest()
            or self.manifest_size_bytes != len(manifest_bytes)
            or self.payload_size_bytes != manifest.payload_size_bytes
        ):
            raise ValueError("transfer envelope does not match its manifest")


class TransferRootInventory(_StrictTransferModel):
    typed_entries: tuple[str, ...] = ()
    project_source_entries: tuple[str, ...] = ()
    control_entries: tuple[str, ...] = ()
    provenance_entries: tuple[str, ...] = ()
    canonical_entries: tuple[str, ...] = ()
    delegated_entries: tuple[str, ...] = ()
    excluded_entries: tuple[str, ...] = ()
    unclassified_entries: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_inventory(self) -> TransferRootInventory:
        groups = (
            self.typed_entries,
            self.project_source_entries,
            self.control_entries,
            self.provenance_entries,
            self.canonical_entries,
            self.delegated_entries,
            self.excluded_entries,
            self.unclassified_entries,
        )
        for values in groups:
            if tuple(sorted(set(values))) != values:
                raise ValueError("transfer root inventory groups must be sorted and unique")
        flattened = [item for values in groups for item in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError("transfer root inventory cannot classify one entry twice")
        return self

    @property
    def complete(self) -> bool:
        return not self.unclassified_entries


class TransferTableInventory(_StrictTransferModel):
    project_linked_tables: tuple[str, ...]
    global_tables: tuple[str, ...]
    unclassified_tables: tuple[str, ...]

    @model_validator(mode="after")
    def validate_inventory(self) -> TransferTableInventory:
        groups = (self.project_linked_tables, self.global_tables, self.unclassified_tables)
        for values in groups:
            if tuple(sorted(set(values))) != values:
                raise ValueError("transfer table inventory groups must be sorted and unique")
        flattened = [table for values in groups for table in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError("transfer table inventory cannot classify one table twice")
        return self

    @property
    def complete(self) -> bool:
        return not self.unclassified_tables


def _direct_root_names(root: Path, *, label: str) -> tuple[str, ...]:
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise ValueError(f"Could not inspect the {label} root") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"The {label} root must be a directory")
    try:
        names = tuple(sorted(entry.name for entry in root.iterdir()))
    except OSError as exc:
        raise ValueError(f"Could not enumerate the {label} root") from exc
    if len(names) > PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES:
        raise ValueError(f"The {label} root inventory exceeds its entry bound")
    return names


def inspect_transfer_app_data_roots(data_dir: Path) -> TransferRootInventory:
    names = _direct_root_names(data_dir, label="transfer app-data")
    return TransferRootInventory(
        typed_entries=tuple(name for name in names if name in TRANSFER_APP_DATA_TYPED_ROOTS),
        project_source_entries=tuple(
            name for name in names if name in TRANSFER_APP_DATA_PROJECT_SOURCE_ROOTS
        ),
        control_entries=tuple(name for name in names if name in TRANSFER_APP_DATA_CONTROL_ROOTS),
        excluded_entries=tuple(name for name in names if name in TRANSFER_APP_DATA_EXCLUDED_ROOTS),
        unclassified_entries=tuple(
            name
            for name in names
            if name
            not in (
                TRANSFER_APP_DATA_TYPED_ROOTS
                | TRANSFER_APP_DATA_PROJECT_SOURCE_ROOTS
                | TRANSFER_APP_DATA_CONTROL_ROOTS
                | TRANSFER_APP_DATA_EXCLUDED_ROOTS
            )
        ),
    )


def inspect_transfer_research_roots(research_dir: Path) -> TransferRootInventory:
    names = _direct_root_names(research_dir, label="transfer research")
    return TransferRootInventory(
        provenance_entries=tuple(
            name for name in names if name in TRANSFER_RESEARCH_PROVENANCE_ROOTS
        ),
        canonical_entries=tuple(
            name for name in names if name in TRANSFER_RESEARCH_CANONICAL_ROOTS
        ),
        delegated_entries=tuple(
            name for name in names if name in TRANSFER_RESEARCH_DELEGATED_ROOTS
        ),
        excluded_entries=tuple(name for name in names if name in TRANSFER_RESEARCH_EXCLUDED_ROOTS),
        unclassified_entries=tuple(
            name
            for name in names
            if name
            not in (
                TRANSFER_RESEARCH_PROVENANCE_ROOTS
                | TRANSFER_RESEARCH_CANONICAL_ROOTS
                | TRANSFER_RESEARCH_DELEGATED_ROOTS
                | TRANSFER_RESEARCH_EXCLUDED_ROOTS
            )
        ),
    )


def _schema_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
    }


def inspect_project_linked_tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Discover direct and foreign-key-linked project tables without reading rows."""

    tables = _schema_tables(connection)

    def pragma_rows(kind: str, table: str) -> list[sqlite3.Row]:
        identifier = table.replace('"', '""')
        return connection.execute(f'PRAGMA {kind}("{identifier}")').fetchall()

    linked = {
        table
        for table in tables
        if _PROJECT_LINK_COLUMNS.intersection(
            str(row[1]) for row in pragma_rows("table_info", table)
        )
    }
    while True:
        children = {
            table
            for table in tables
            if any(str(row[2]) in linked for row in pragma_rows("foreign_key_list", table))
        }
        if children <= linked:
            return tuple(sorted(linked))
        linked.update(children)


def inspect_transfer_table_inventory(connection: sqlite3.Connection) -> TransferTableInventory:
    """Partition the entire application schema so every later table is visible."""

    tables = _schema_tables(connection)
    project = set(inspect_project_linked_tables(connection))
    global_tables = tables.intersection(TRANSFER_GLOBAL_TABLES)
    return TransferTableInventory(
        project_linked_tables=tuple(sorted(project)),
        global_tables=tuple(sorted(global_tables)),
        unclassified_tables=tuple(sorted(tables - project - global_tables)),
    )


__all__ = [
    "TRANSFER_APP_DATA_CONTROL_ROOTS",
    "TRANSFER_APP_DATA_EXCLUDED_ROOTS",
    "TRANSFER_APP_DATA_PROJECT_SOURCE_ROOTS",
    "TRANSFER_APP_DATA_TYPED_ROOTS",
    "TRANSFER_ARCHIVE_CODEC",
    "TRANSFER_ARCHIVE_SCHEMA_VERSION",
    "TRANSFER_GLOBAL_TABLES",
    "TRANSFER_RESEARCH_CANONICAL_ROOTS",
    "TRANSFER_RESEARCH_DELEGATED_ROOTS",
    "TRANSFER_RESEARCH_EXCLUDED_ROOTS",
    "TRANSFER_RESEARCH_PROVENANCE_ROOTS",
    "TransferArchiveActor",
    "TransferArchiveAttribution",
    "TransferArchiveDiagnostic",
    "TransferArchiveEntry",
    "TransferArchiveEnvelope",
    "TransferArchiveManifest",
    "TransferGraphHead",
    "TransferGraphTarget",
    "TransferRootInventory",
    "TransferTableInventory",
    "inspect_project_linked_tables",
    "inspect_transfer_app_data_roots",
    "inspect_transfer_research_roots",
    "inspect_transfer_table_inventory",
]
