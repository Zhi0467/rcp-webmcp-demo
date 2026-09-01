from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rcp.artifacts import validate_artifact_bytes
from rcp.config import DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING, AgentExecutionProfile
from rcp.core.authority import (
    AgentDispatchAuthority,
)
from rcp.core.models import (
    DISPLAY_NAME_MAX_LENGTH,
    AuthorizedHuman,
    normalize_display_name,
)
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.limits import (
    CHAT_ARTIFACT_MAX_FILE_BYTES,
    MEMBER_REMOVAL_PREVIEW_MAX_ITEMS,
    TEAM_ENROLLMENT_CODE_MAX_LENGTH,
    WATCHER_ERROR_BACKOFF_SECONDS,
    WATCHER_HEALTHY_INTERVAL_SECONDS,
    WATCHER_SCHEDULE_JITTER_RATIO,
)
from rcp.providers import (
    ProviderId,
    ProviderSkill,
    legacy_runtime_id,
    require_runtime_id,
    runtime_label,
)
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.server_ops.models import (
    ExternalServiceTarget,
    MachineTarget,
    MessageText,
    ServerStep,
    redact_server_text,
)
from rcp.skill_registry import SkillReference

if TYPE_CHECKING:
    pass


SpaceKind = Literal["personal", "team"]
SpaceUserKind = Literal["local_owner", "team_member"]
SPACE_NAME_MAX_LENGTH = 120


class TeamAuthenticationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SpaceUserRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str
    identity_kind: SpaceUserKind
    display_name: str | None = Field(default=None, max_length=DISPLAY_NAME_MAX_LENGTH)
    created_at: str
    updated_at: str
    removal_started_at: str | None = None
    removed_at: str | None = None

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        try:
            return _canonical_uuid4(value, label="user identity")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("display_name", mode="before")
    @classmethod
    def normalize_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_display_name(value)

    @model_validator(mode="after")
    def removal_is_ordered(self) -> SpaceUserRecord:
        if self.removed_at is not None and self.removal_started_at is None:
            raise ValueError("a removed member requires its durable access fence")
        return self


class MemberRemovalPreviewRecord(BaseModel):
    """One exact, digest-bound preview of a member-removal consequence set."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    member: SpaceUserRecord
    last_authenticating_member: bool
    project_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    orphaned_project_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    active_task_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    active_episode_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    active_token_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    browser_session_count: int = Field(ge=0)
    space_invitation_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    project_invitation_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    boundary_sha256: str

    @model_validator(mode="after")
    def inventory_is_canonical(self) -> MemberRemovalPreviewRecord:
        for values in (
            self.project_ids,
            self.orphaned_project_ids,
            self.active_task_ids,
            self.active_episode_ids,
            self.active_token_ids,
            self.space_invitation_ids,
            self.project_invitation_ids,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("member-removal preview inventories must be sorted and unique")
        if not set(self.orphaned_project_ids).issubset(self.project_ids):
            raise ValueError("member-removal orphaned projects must belong to the target")
        if not re.fullmatch(r"[0-9a-f]{64}", self.boundary_sha256):
            raise ValueError("member-removal preview boundary must be lowercase SHA-256")
        return self


class TeamMemberAuthorityRecord(BaseModel):
    """Nonsecret, exact authority retained by one active team member."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    member_id: str
    display_name: str | None
    active_token_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)

    @model_validator(mode="after")
    def authority_is_canonical(self) -> TeamMemberAuthorityRecord:
        _canonical_uuid4(self.member_id, label="member identity")
        if self.active_token_ids != tuple(sorted(set(self.active_token_ids))):
            raise ValueError("active team token identities must be sorted and unique")
        for token_id in self.active_token_ids:
            _canonical_uuid4(token_id, label="team token identity")
        return self


class TeamInvitationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    invitation_id: str
    created_by: str
    created_at: str
    expires_at: str
    consumed_at: str | None = None
    consumed_by: str | None = None
    failed_attempts: int
    locked_at: str | None = None
    revoked_at: str | None = None


class ProjectMemberRecord(BaseModel):
    """One person's membership of one project.

    Membership is operational authority inside RCP. It binds the durable
    ``user_id`` and never a display name, so a member exists before they have
    chosen one. It lives in SQLite and never in ``.research/``.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    project_id: str
    user_id: str
    seated_at: str
    seated_by: str | None = None

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        try:
            return _canonical_uuid4(value, label="user identity")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc


class ProjectInvitationRecord(BaseModel):
    """One in-product invitation to join one project.

    It carries no code, no expiry, and no lockout, because it grants no
    credential — the person is already enrolled in the space.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    invitation_id: str
    project_id: str
    invited_user_id: str
    invited_by: str
    created_at: str
    response: Literal["accepted", "declined", "revoked"] | None = None
    responded_at: str | None = None


class ProjectRecord(BaseModel):
    project_id: str
    home_space_id: str | None = None
    locator: str
    name: str
    state_location: str
    state_remote: bool
    added_at: str
    last_opened_at: str | None = None
    revision: int | None = None
    primary_question: str | None = None
    attention_count: int = 0
    last_refresh_at: str | None = None
    reachable: bool | None = None
    error: str | None = None
    retired_at: str | None = None
    retired_transfer_request_id: str | None = None

    @field_validator("home_space_id")
    @classmethod
    def validate_home_space_id(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None
        try:
            home_space_id = _canonical_uuid4(value, label="project home space identity")
            _canonical_uuid4(info.data.get("project_id"), label="canonical project identity")
            return home_space_id
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc


class ProjectStageRecord(BaseModel):
    host: str
    root: str


ProjectProvisioningKind = Literal["create_team_project", "incoming_transfer"]
ProjectProvisioningStatus = Literal[
    "waiting_for_server_setup",
    "setup_in_progress",
    "operator_action_needed",
    "ready_for_review",
    "completed",
    "cancelled",
]
ProjectProvisioningCheckStatus = Literal[
    "pending",
    "checking",
    "operator_action_needed",
    "ready",
]
ProjectProvisioningCancellationDisposition = Literal[
    "nothing_to_remove",
    "request_owned_state_removed",
    "prepared_state_preserved",
    "operator_cleanup_confirmed",
]
ProjectProvisioningCheckoutDisposition = Literal[
    "request_created",
    "reused_existing",
]
ProjectTransferSide = Literal["source", "target"]
ProjectTransferPhase = Literal[
    "awaiting_link",
    "linked",
    "target_admitted",
    "source_released",
    "source_fenced",
    "archive_bound",
    "target_activated",
    "cleanup_acknowledged",
    "completed",
    "operator_action_needed",
]
ProjectTransferProofKind = Literal["source_release", "target_activation"]
ProjectTransferProofState = Literal[
    "unexposed",
    "exposed",
    "acknowledged",
    "consumed",
]

_PROVISIONING_ALIAS = re.compile(r"[a-z][a-z0-9-]{0,47}")
_PROVISIONING_HOST = re.compile(r"[A-Za-z0-9_.@:-]{1,255}")
_PROVISIONING_ACCOUNT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")
_PROVISIONING_RUNTIME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}")
_PROVISIONING_PHASE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PROVISIONING_RECEIPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_OPENSSH_FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{20,64}={0,2}")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _provisioning_absolute_path(value: str, *, label: str) -> str:
    if len(value) > 4096 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"{label} must be one bounded safe path")
    if redact_server_text(value) != value:
        raise ValueError(f"{label} cannot contain credential-shaped text")
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError(f"{label} must be a specific absolute path")
    if str(path) != value:
        raise ValueError(f"{label} must be normalized")
    return value


def _provisioning_resume_matches_request(argv: tuple[str, ...], request_id: str) -> bool:
    command_tails = (
        ("server", "project", "provision", request_id),
        ("server", "provider", "check", "--request", request_id),
    )
    executable_prefixes = (
        ("rcp",),
        (str(DEFAULT_SERVER_LAYOUT.cli_wrapper),),
        ("sudo", "-u", "rcp", "-H", str(DEFAULT_SERVER_LAYOUT.cli_wrapper)),
        ("sudo", "-n", "-u", "rcp", "-H", str(DEFAULT_SERVER_LAYOUT.cli_wrapper)),
    )
    return any(
        argv in {prefix + tail, prefix + tail + ("--machine-readable",)}
        for prefix in executable_prefixes
        for tail in command_tails
    )


class _StrictProvisioningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")


class ProjectProvisioningMachineIntent(_StrictProvisioningModel):
    alias: str
    location: Literal["local", "ssh"]
    host: str = ""
    os_account: str
    central_root: str | None = None

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _PROVISIONING_ALIAS.fullmatch(value) is None:
            raise ValueError("provisioning machine alias is invalid")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        if value and _PROVISIONING_HOST.fullmatch(value) is None:
            raise ValueError("provisioning SSH host is invalid")
        if redact_server_text(value) != value:
            raise ValueError("provisioning SSH host cannot contain credential-shaped text")
        return value

    @field_validator("os_account")
    @classmethod
    def validate_os_account(cls, value: str) -> str:
        if _PROVISIONING_ACCOUNT.fullmatch(value) is None:
            raise ValueError("provisioning operating-system account is invalid")
        return value

    @field_validator("central_root")
    @classmethod
    def validate_central_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _provisioning_absolute_path(value, label="provisioning central root")

    @model_validator(mode="after")
    def validate_location(self) -> ProjectProvisioningMachineIntent:
        if self.location == "local":
            if self.host:
                raise ValueError("the server-local provisioning machine cannot name an SSH host")
            if self.os_account != DEFAULT_SERVER_LAYOUT.service_account:
                raise ValueError("the server-local provisioning machine must execute as rcp")
            if self.central_root != str(DEFAULT_SERVER_LAYOUT.projects_root):
                raise ValueError("the server-local provisioning root is fixed by the installation")
        elif not self.host:
            raise ValueError("an SSH provisioning machine requires a configured host")
        return self


class ProjectProvisioningMachineRecord(ProjectProvisioningMachineIntent):
    resolved_central_root: str | None = None

    @field_validator("resolved_central_root")
    @classmethod
    def validate_resolved_central_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _provisioning_absolute_path(value, label="resolved provisioning central root")

    @model_validator(mode="after")
    def resolved_root_matches_intent(self) -> ProjectProvisioningMachineRecord:
        if (
            self.central_root is not None
            and self.resolved_central_root is not None
            and self.resolved_central_root != self.central_root
        ):
            raise ValueError("resolved provisioning root must match the reviewed central root")
        return self


class ProjectProvisioningGitCheckRecord(_StrictProvisioningModel):
    status: ProjectProvisioningCheckStatus = "pending"
    commit: str | None = None
    write_verified: bool = False
    deploy_key_label: str | None = Field(default=None, max_length=255)
    public_key_fingerprint: str | None = None
    checked_at: str | None = None
    diagnostic: MessageText | None = None

    @field_validator("deploy_key_label")
    @classmethod
    def validate_deploy_key_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value.strip()
            or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("provisioning deploy-key label must be one nonempty safe line")
        if redact_server_text(value) != value:
            raise ValueError("provisioning deploy-key label cannot contain credential-shaped text")
        return value

    @field_validator("commit")
    @classmethod
    def validate_commit(cls, value: str | None) -> str | None:
        if value is not None and _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("provisioning Git commit must be a lowercase full object id")
        return value

    @field_validator("public_key_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and _OPENSSH_FINGERPRINT.fullmatch(value) is None:
            raise ValueError("provisioning public-key fingerprint is invalid")
        return value

    @model_validator(mode="after")
    def ready_means_read_write_proven(self) -> ProjectProvisioningGitCheckRecord:
        if self.status == "ready" and (
            self.commit is None
            or not self.write_verified
            or self.deploy_key_label is None
            or self.public_key_fingerprint is None
            or self.checked_at is None
        ):
            raise ValueError(
                "a ready Git check requires a commit, write proof, deploy-key label, "
                "fingerprint, and check time"
            )
        if self.write_verified and self.commit is None:
            raise ValueError("a Git write proof must name the commit it proved")
        return self


class ProjectProvisioningRepositoryIntent(_StrictProvisioningModel):
    alias: str
    repository: GitHubRepositoryRef
    machine_alias: str

    @field_validator("alias", "machine_alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _PROVISIONING_ALIAS.fullmatch(value) is None:
            raise ValueError("provisioning repository or machine alias is invalid")
        return value


class ProjectProvisioningRepositoryRecord(ProjectProvisioningRepositoryIntent):
    intended_path: str | None
    resolved_path: str | None = None
    checkout_disposition: ProjectProvisioningCheckoutDisposition | None = None
    git_check: ProjectProvisioningGitCheckRecord = Field(
        default_factory=ProjectProvisioningGitCheckRecord
    )

    @field_validator("intended_path", "resolved_path")
    @classmethod
    def validate_path(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _provisioning_absolute_path(value, label=info.field_name.replace("_", " "))


class ProjectProvisioningProviderIntent(_StrictProvisioningModel):
    profile: AgentExecutionProfile
    provider: ProviderId
    runtime_id: str
    model: str = Field(max_length=200)
    reasoning: str = Field(min_length=1, max_length=80)
    machine_alias: str

    @field_validator("runtime_id")
    @classmethod
    def validate_runtime_id(cls, value: str) -> str:
        if _PROVISIONING_RUNTIME.fullmatch(value) is None:
            raise ValueError("provisioning provider runtime id is invalid")
        if redact_server_text(value) != value:
            raise ValueError("provisioning provider runtime cannot contain credential-shaped text")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("provisioning provider model must be one safe line")
        if redact_server_text(value) != value:
            raise ValueError("provisioning provider model cannot contain credential-shaped text")
        return value

    @field_validator("reasoning", "machine_alias")
    @classmethod
    def validate_plain_value(cls, value: str, info: ValidationInfo) -> str:
        pattern = (
            _PROVISIONING_ALIAS
            if info.field_name == "machine_alias"
            else re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
        )
        if pattern.fullmatch(value) is None:
            raise ValueError(f"provisioning provider {info.field_name} is invalid")
        return value


class ProjectProvisioningProviderCheckRecord(ProjectProvisioningProviderIntent):
    status: ProjectProvisioningCheckStatus = "pending"
    binary_path: str | None = None
    version: str | None = Field(default=None, max_length=240)
    resolved_runtime_id: str | None = None
    execution_account: str | None = None
    checked_at: str | None = None
    diagnostic: MessageText | None = None

    @field_validator("binary_path")
    @classmethod
    def validate_binary_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _provisioning_absolute_path(value, label="provider executable path")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value.strip()
            or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
            or redact_server_text(value) != value
        ):
            raise ValueError("provider version must be one nonsecret safe line")
        return value

    @field_validator("resolved_runtime_id")
    @classmethod
    def validate_resolved_runtime_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ProjectProvisioningProviderIntent.validate_runtime_id(value)

    @field_validator("execution_account")
    @classmethod
    def validate_execution_account(cls, value: str | None) -> str | None:
        if value is not None and _PROVISIONING_ACCOUNT.fullmatch(value) is None:
            raise ValueError("provider execution account is invalid")
        return value

    @model_validator(mode="after")
    def ready_has_check_time(self) -> ProjectProvisioningProviderCheckRecord:
        if self.status == "ready" and self.checked_at is None:
            raise ValueError("a ready provider check requires its check time")
        proof = (
            self.binary_path,
            self.version,
            self.resolved_runtime_id,
            self.execution_account,
        )
        if any(value is not None for value in proof) and (
            self.status != "ready" or any(value is None for value in proof)
        ):
            raise ValueError("provider readiness proof fields must be complete and ready")
        return self


class ProjectProvisioningRequestRecord(_StrictProvisioningModel):
    request_id: str
    kind: ProjectProvisioningKind
    status: ProjectProvisioningStatus
    target_space_id: str
    authorized_by: AuthorizedHuman
    proposed_project_id: str
    name: str | None = Field(default=None, max_length=120)
    state_repository: str | None = None
    project_truth_scope: list[str] = Field(default_factory=list, max_length=64)
    default_run_truth_scope: list[str] = Field(default_factory=list, max_length=64)
    default_auto_research_invocation_ceiling: int = Field(
        default=DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING,
        ge=1,
    )
    machines: list[ProjectProvisioningMachineRecord] = Field(min_length=1, max_length=32)
    repositories: list[ProjectProvisioningRepositoryRecord] = Field(min_length=1, max_length=64)
    provider_checks: list[ProjectProvisioningProviderCheckRecord] = Field(
        min_length=1, max_length=32
    )
    retryable_diagnostic: MessageText | None = None
    operator_action: ServerStep | None = None
    final_review_digest: str | None = None
    cancellation_disposition: ProjectProvisioningCancellationDisposition | None = None
    revision: int = Field(ge=0)
    created_at: str
    updated_at: str
    setup_started_at: str | None = None
    ready_at: str | None = None
    completed_at: str | None = None
    cancelled_at: str | None = None

    @field_validator("request_id", "target_space_id", "proposed_project_id")
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("final_review_digest")
    @classmethod
    def validate_review_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("provisioning final-review digest must be lowercase SHA-256")
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or any(
            ord(character) < 32 or ord(character) == 127 for character in normalized
        ):
            raise ValueError("provisioning project name must be one nonempty safe line")
        if redact_server_text(normalized) != normalized:
            raise ValueError("provisioning project name cannot contain credential-shaped text")
        return normalized

    @field_validator("state_repository")
    @classmethod
    def validate_state_repository(cls, value: str | None) -> str | None:
        if value is not None and _PROVISIONING_ALIAS.fullmatch(value) is None:
            raise ValueError("provisioning state repository alias is invalid")
        return value

    @field_validator("project_truth_scope", "default_run_truth_scope")
    @classmethod
    def validate_scope(cls, value: list[str], info: ValidationInfo) -> list[str]:
        if any(_PROVISIONING_ALIAS.fullmatch(alias) is None for alias in value):
            raise ValueError(f"provisioning {info.field_name} contains an invalid alias")
        if len(value) != len(set(value)):
            raise ValueError(f"provisioning {info.field_name} must not repeat an alias")
        return value

    @property
    def configuration_complete(self) -> bool:
        return bool(
            self.name is not None
            and self.state_repository is not None
            and self.project_truth_scope
            and self.default_run_truth_scope
        )

    @model_validator(mode="after")
    def validate_request(self) -> ProjectProvisioningRequestRecord:
        if self.target_space_id != self.authorized_by.space_id:
            raise ValueError("provisioning authorizer must belong to the target space")
        machine_map = {machine.alias: machine for machine in self.machines}
        if len(machine_map) != len(self.machines):
            raise ValueError("provisioning machine aliases must be unique")
        repository_aliases = [repository.alias for repository in self.repositories]
        if len(set(repository_aliases)) != len(repository_aliases):
            raise ValueError("provisioning repository aliases must be unique")
        repository_identities = [repository.repository.identity for repository in self.repositories]
        if len(set(repository_identities)) != len(repository_identities):
            raise ValueError("one GitHub repository cannot appear twice in a provisioning request")
        has_any_configuration = bool(
            self.name is not None
            or self.state_repository is not None
            or self.project_truth_scope
            or self.default_run_truth_scope
            or self.default_auto_research_invocation_ceiling
            != DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING
        )
        if has_any_configuration and not self.configuration_complete:
            raise ValueError("provisioning project configuration must be complete")
        if self.configuration_complete:
            repository_set = set(repository_aliases)
            assert self.state_repository is not None
            if self.state_repository not in repository_set:
                raise ValueError("provisioning state repository must name a declared repository")
            if not set(self.project_truth_scope).issubset(repository_set):
                raise ValueError("provisioning project truth scope names an unknown repository")
            if self.state_repository not in self.project_truth_scope:
                raise ValueError("provisioning state repository must remain in project truth scope")
            if not set(self.default_run_truth_scope).issubset(set(self.project_truth_scope)):
                raise ValueError("provisioning default run truth scope must be a project subset")
        for repository in self.repositories:
            machine = machine_map.get(repository.machine_alias)
            if machine is None:
                raise ValueError("provisioning repository names an unknown machine")
            if machine.central_root is None:
                if repository.intended_path is not None:
                    raise ValueError("default remote repository path cannot be guessed")
            else:
                intended = (
                    PurePosixPath(machine.central_root)
                    / self.proposed_project_id
                    / "repositories"
                    / repository.alias
                )
                if repository.intended_path != str(intended):
                    raise ValueError("provisioning repository intended path is not derived")
            if machine.resolved_central_root is None:
                if (
                    repository.resolved_path is not None
                    or repository.checkout_disposition is not None
                ):
                    raise ValueError("repository path cannot resolve before its central root")
            elif repository.resolved_path is not None:
                resolved = (
                    PurePosixPath(machine.resolved_central_root)
                    / self.proposed_project_id
                    / "repositories"
                    / repository.alias
                )
                if repository.resolved_path != str(resolved):
                    raise ValueError("provisioning repository resolved path is not derived")
                if self.configuration_complete and repository.checkout_disposition is None:
                    raise ValueError("resolved repository path requires its checkout disposition")
            elif repository.checkout_disposition is not None:
                raise ValueError("checkout disposition requires a resolved repository path")
            expected_key_label = (
                f"rcp:{self.target_space_id}:{self.proposed_project_id}:{repository.alias}"
            )
            if (
                repository.git_check.deploy_key_label is not None
                and repository.git_check.deploy_key_label != expected_key_label
            ):
                raise ValueError("provisioning deploy-key label is not derived from the request")
        profiles = [check.profile for check in self.provider_checks]
        if len(set(profiles)) != len(profiles):
            raise ValueError("provisioning provider profiles must be unique")
        if any(check.machine_alias not in machine_map for check in self.provider_checks):
            raise ValueError("provisioning provider check names an unknown machine")
        if self.status == "operator_action_needed":
            if self.operator_action is None:
                raise ValueError("operator-action provisioning requires one structured action")
            if self.operator_action.state != "operator_action_needed":
                raise ValueError("durable provisioning action must be an operator-action step")
            if not _provisioning_resume_matches_request(
                self.operator_action.resume_argv,
                self.request_id,
            ):
                raise ValueError("provisioning operator action must resume this exact request")
            target = self.operator_action.target
            if isinstance(target, ExternalServiceTarget):
                matching_repository = next(
                    (
                        repository
                        for repository in self.repositories
                        if target.service == "github.com"
                        and target.resource == repository.repository.identity
                        and target.destination_url == repository.repository.settings_url
                        and target.required_authority_role == "repository administrator"
                    ),
                    None,
                )
                if matching_repository is None:
                    raise ValueError(
                        "provisioning external action must target one declared GitHub repository"
                    )
            elif isinstance(target, MachineTarget):
                if not any(
                    machine.os_account == target.os_account
                    and (machine.location == "local" or machine.host == target.host)
                    for machine in self.machines
                ):
                    raise ValueError(
                        "provisioning machine action must target one declared execution account"
                    )
        elif self.operator_action is not None:
            raise ValueError("only operator-action provisioning may retain an operator action")
        review_status = self.status in {"ready_for_review", "completed"}
        if review_status:
            if self.final_review_digest is None or self.ready_at is None:
                raise ValueError("reviewable provisioning requires a digest and ready time")
            repository_machines = {repository.machine_alias for repository in self.repositories}
            if any(
                machine.alias in repository_machines and machine.resolved_central_root is None
                for machine in self.machines
            ):
                raise ValueError("reviewable provisioning requires every checkout central root")
            if self.configuration_complete and any(
                repository.resolved_path is None or repository.checkout_disposition is None
                for repository in self.repositories
            ):
                raise ValueError("reviewable provisioning requires every central checkout")
            if any(repository.git_check.status != "ready" for repository in self.repositories):
                raise ValueError("reviewable provisioning requires every Git check")
            if any(check.status != "ready" for check in self.provider_checks):
                raise ValueError("reviewable provisioning requires every provider check")
        elif self.final_review_digest is not None or self.ready_at is not None:
            raise ValueError("only reviewable provisioning may retain a final-review digest")
        if self.status == "completed" and self.completed_at is None:
            raise ValueError("completed provisioning requires its completion time")
        if self.status != "completed" and self.completed_at is not None:
            raise ValueError("only completed provisioning may retain a completion time")
        if self.status == "cancelled":
            if self.cancelled_at is None or self.cancellation_disposition is None:
                raise ValueError("cancelled provisioning requires time and explicit disposition")
        elif self.cancelled_at is not None:
            raise ValueError("only cancelled provisioning may retain a cancellation time")
        if self.cancellation_disposition is not None and self.status not in {
            "operator_action_needed",
            "cancelled",
        }:
            raise ValueError("cancellation disposition belongs only to cancellation handling")
        return self


class ProjectProvisioningStepReceiptRecord(_StrictProvisioningModel):
    request_id: str
    receipt_id: str
    phase: str
    from_status: ProjectProvisioningStatus
    to_status: ProjectProvisioningStatus
    transition_sha256: str
    resulting_revision: int = Field(ge=1)
    created_at: str

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        try:
            return _canonical_uuid4(value, label="provisioning request identity")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("receipt_id")
    @classmethod
    def validate_receipt_id(cls, value: str) -> str:
        if _PROVISIONING_RECEIPT_ID.fullmatch(value) is None:
            raise ValueError("provisioning receipt id is invalid")
        return value

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, value: str) -> str:
        if _PROVISIONING_PHASE.fullmatch(value) is None:
            raise ValueError("provisioning receipt phase is invalid")
        return value

    @field_validator("transition_sha256")
    @classmethod
    def validate_transition_digest(cls, value: str) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("provisioning transition digest must be lowercase SHA-256")
        return value


class ProjectTransferRepositorySource(_StrictProvisioningModel):
    alias: str
    repository: GitHubRepositoryRef
    machine_alias: str

    @field_validator("alias", "machine_alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _PROVISIONING_ALIAS.fullmatch(value) is None:
            raise ValueError("transfer repository or machine alias is invalid")
        return value


class ProjectTransferSourceConfiguration(_StrictProvisioningModel):
    source_rcp_version: str = Field(min_length=1, max_length=120)
    source_schema_generation: int = Field(ge=1)
    supported_archive_codecs: tuple[str, ...] = Field(min_length=1, max_length=16)
    machine_aliases: tuple[str, ...] = Field(min_length=1, max_length=32)
    repositories: tuple[ProjectTransferRepositorySource, ...] = Field(
        min_length=1,
        max_length=64,
    )
    state_repository: str
    project_truth_scope: tuple[str, ...] = Field(min_length=1, max_length=64)
    default_run_truth_scope: tuple[str, ...] = Field(min_length=1, max_length=64)
    source_manifest_sha256: str

    @field_validator("source_rcp_version")
    @classmethod
    def validate_rcp_version(cls, value: str) -> str:
        if (
            value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
            or redact_server_text(value) != value
        ):
            raise ValueError("source RCP version must be one nonsecret safe line")
        return value

    @field_validator("supported_archive_codecs")
    @classmethod
    def validate_archive_codecs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            _PROVISIONING_RUNTIME.fullmatch(item) is None for item in value
        ):
            raise ValueError("transfer archive codecs must be unique safe identifiers")
        return value

    @field_validator("machine_aliases")
    @classmethod
    def validate_machine_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            _PROVISIONING_ALIAS.fullmatch(alias) is None for alias in value
        ):
            raise ValueError("transfer machine aliases must be unique safe identifiers")
        return value

    @field_validator("state_repository")
    @classmethod
    def validate_state_repository(cls, value: str) -> str:
        if _PROVISIONING_ALIAS.fullmatch(value) is None:
            raise ValueError("transfer state repository alias is invalid")
        return value

    @field_validator("project_truth_scope", "default_run_truth_scope")
    @classmethod
    def validate_scope(cls, value: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            _PROVISIONING_ALIAS.fullmatch(alias) is None for alias in value
        ):
            raise ValueError(f"transfer {info.field_name} must contain unique repository aliases")
        return value

    @field_validator("source_manifest_sha256")
    @classmethod
    def validate_manifest_digest(cls, value: str) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("source manifest digest must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_repository_provenance(self) -> ProjectTransferSourceConfiguration:
        aliases = [repository.alias for repository in self.repositories]
        identities = [repository.repository.identity for repository in self.repositories]
        if len(aliases) != len(set(aliases)):
            raise ValueError("transfer repository aliases must be unique")
        if len(identities) != len(set(identities)):
            raise ValueError("one GitHub repository cannot appear twice in transfer provenance")
        if not {repository.machine_alias for repository in self.repositories}.issubset(
            self.machine_aliases
        ):
            raise ValueError("transfer repository names an unknown historical machine alias")
        alias_set = set(aliases)
        if self.state_repository not in alias_set:
            raise ValueError("transfer state repository must name a declared repository")
        if not set(self.project_truth_scope).issubset(alias_set):
            raise ValueError("transfer project truth scope names an unknown repository")
        if self.state_repository not in self.project_truth_scope:
            raise ValueError("transfer state repository must remain in project truth scope")
        if not set(self.default_run_truth_scope).issubset(self.project_truth_scope):
            raise ValueError("transfer default run truth scope must be a project subset")
        return self


class ProjectTransferResolvedPath(_StrictProvisioningModel):
    repository_alias: str
    machine_alias: str
    path: str

    @field_validator("repository_alias", "machine_alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _PROVISIONING_ALIAS.fullmatch(value) is None:
            raise ValueError("resolved transfer alias is invalid")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _provisioning_absolute_path(value, label="resolved transfer repository path")


class ProjectTransferRepositoryBinding(_StrictProvisioningModel):
    alias: str
    repository: GitHubRepositoryRef

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if _PROVISIONING_ALIAS.fullmatch(value) is None:
            raise ValueError("linked transfer repository alias is invalid")
        return value


class ProjectTransferLinkReceipt(_StrictProvisioningModel):
    source_request_id: str
    target_request_id: str
    project_id: str
    source_space_id: str
    target_space_id: str
    source_configuration_sha256: str
    target_repositories: tuple[ProjectTransferRepositoryBinding, ...] = Field(
        min_length=1,
        max_length=64,
    )
    accepted_schema_generation: int = Field(ge=1)
    accepted_archive_codec: str
    source_release_proof_sha256: str
    target_activation_proof_sha256: str
    created_at: str

    @field_validator(
        "source_request_id",
        "target_request_id",
        "project_id",
        "source_space_id",
        "target_space_id",
    )
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator(
        "source_configuration_sha256",
        "source_release_proof_sha256",
        "target_activation_proof_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("transfer link digest must be lowercase SHA-256")
        return value

    @field_validator("accepted_archive_codec")
    @classmethod
    def validate_archive_codec(cls, value: str) -> str:
        if _PROVISIONING_RUNTIME.fullmatch(value) is None:
            raise ValueError("accepted transfer archive codec is invalid")
        return value

    @model_validator(mode="after")
    def validate_link(self) -> ProjectTransferLinkReceipt:
        if self.source_space_id == self.target_space_id:
            raise ValueError("a transfer link must cross spaces")
        aliases = [item.alias for item in self.target_repositories]
        identities = [item.repository.identity for item in self.target_repositories]
        if len(aliases) != len(set(aliases)):
            raise ValueError("linked target repository aliases must be unique")
        if aliases != sorted(aliases):
            raise ValueError("linked target repositories must be ordered by alias")
        if len(identities) != len(set(identities)):
            raise ValueError("one target repository cannot appear twice in a transfer link")
        return self


class ProjectTransferTargetAdmissionReceipt(_StrictProvisioningModel):
    source_request_id: str
    target_request_id: str
    project_id: str
    source_space_id: str
    target_space_id: str
    admitted_by: AuthorizedHuman
    source_configuration_sha256: str
    target_preparation_revision: int = Field(ge=0)
    target_preparation_sha256: str
    resolved_paths: tuple[ProjectTransferResolvedPath, ...] = Field(
        min_length=1,
        max_length=64,
    )
    accepted_schema_generation: int = Field(ge=1)
    accepted_archive_codec: str
    source_release_proof_sha256: str
    target_activation_proof_sha256: str
    created_at: str

    @field_validator(
        "source_request_id",
        "target_request_id",
        "project_id",
        "source_space_id",
        "target_space_id",
    )
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator(
        "source_configuration_sha256",
        "target_preparation_sha256",
        "source_release_proof_sha256",
        "target_activation_proof_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("transfer receipt digest must be lowercase SHA-256")
        return value

    @field_validator("accepted_archive_codec")
    @classmethod
    def validate_archive_codec(cls, value: str) -> str:
        if _PROVISIONING_RUNTIME.fullmatch(value) is None:
            raise ValueError("accepted transfer archive codec is invalid")
        return value

    @model_validator(mode="after")
    def admission_actor_matches_target(self) -> ProjectTransferTargetAdmissionReceipt:
        if self.source_space_id == self.target_space_id:
            raise ValueError("a transfer must cross spaces")
        if self.admitted_by.space_id != self.target_space_id:
            raise ValueError("target admission actor must belong to the target space")
        aliases = [item.repository_alias for item in self.resolved_paths]
        if len(aliases) != len(set(aliases)):
            raise ValueError("target admission paths must have unique repository aliases")
        return self


class ProjectTransferSourceReleaseReceipt(_StrictProvisioningModel):
    source_request_id: str
    target_request_id: str
    project_id: str
    source_space_id: str
    target_space_id: str
    released_by: AuthorizedHuman
    source_configuration_sha256: str
    target_admission_sha256: str
    target_preparation_revision: int = Field(ge=0)
    target_preparation_sha256: str
    source_head: GraphHeadRef
    accepted_schema_generation: int = Field(ge=1)
    accepted_archive_codec: str
    source_release_proof_sha256: str
    target_activation_proof_sha256: str
    created_at: str

    @field_validator(
        "source_request_id",
        "target_request_id",
        "project_id",
        "source_space_id",
        "target_space_id",
    )
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator(
        "source_configuration_sha256",
        "target_admission_sha256",
        "target_preparation_sha256",
        "source_release_proof_sha256",
        "target_activation_proof_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("transfer receipt digest must be lowercase SHA-256")
        return value

    @field_validator("accepted_archive_codec")
    @classmethod
    def validate_archive_codec(cls, value: str) -> str:
        if _PROVISIONING_RUNTIME.fullmatch(value) is None:
            raise ValueError("accepted transfer archive codec is invalid")
        return value

    @model_validator(mode="after")
    def release_actor_matches_source(self) -> ProjectTransferSourceReleaseReceipt:
        if self.source_space_id == self.target_space_id:
            raise ValueError("a transfer must cross spaces")
        if self.released_by.space_id != self.source_space_id:
            raise ValueError("source release actor must belong to the source space")
        if self.source_head.target.kind != "main":
            raise ValueError("source release must bind the main canonical head")
        return self


class ProjectTransferCleanupAcknowledgment(_StrictProvisioningModel):
    """Public receipt produced only after the source verifies target activation."""

    source_request_id: str
    target_request_id: str
    project_id: str
    source_space_id: str
    target_space_id: str
    source_release_proof_sha256: str
    target_activation_proof_sha256: str
    archive_sha256: str
    source_fence_head: GraphHeadRef

    @field_validator(
        "source_request_id",
        "target_request_id",
        "project_id",
        "source_space_id",
        "target_space_id",
    )
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator(
        "source_release_proof_sha256",
        "target_activation_proof_sha256",
        "archive_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("transfer cleanup receipt digest must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_spaces(self) -> ProjectTransferCleanupAcknowledgment:
        if self.source_space_id == self.target_space_id:
            raise ValueError("a transfer cleanup receipt must cross spaces")
        if self.source_fence_head.target.kind != "main":
            raise ValueError("transfer cleanup must bind the fenced main head")
        return self


_PROJECT_TRANSFER_PHASE_RANK: dict[ProjectTransferPhase, int] = {
    "awaiting_link": 0,
    "linked": 1,
    "target_admitted": 2,
    "source_released": 3,
    "source_fenced": 4,
    "archive_bound": 5,
    "target_activated": 6,
    "cleanup_acknowledged": 7,
    "completed": 8,
    "operator_action_needed": 9,
}

_PROJECT_TRANSFER_RESTORABLE_TARGET_PHASES: frozenset[ProjectTransferPhase] = frozenset(
    {
        "linked",
        "target_admitted",
        "source_released",
        "archive_bound",
        "target_activated",
        "cleanup_acknowledged",
    }
)


class ProjectTransferRequestRecord(_StrictProvisioningModel):
    request_id: str
    side: ProjectTransferSide
    phase: ProjectTransferPhase
    linked_request_id: str | None = None
    project_id: str
    source_space_id: str
    target_space_id: str
    initiated_by: AuthorizedHuman
    source_configuration: ProjectTransferSourceConfiguration
    source_configuration_sha256: str
    accepted_schema_generation: int | None = Field(default=None, ge=1)
    accepted_archive_codec: str | None = None
    source_release_proof_sha256: str
    target_activation_proof_sha256: str | None = None
    link_receipt: ProjectTransferLinkReceipt | None = None
    proof_state: ProjectTransferProofState = "unexposed"
    proof_acknowledgement_sha256: str | None = None
    target_admission_receipt: ProjectTransferTargetAdmissionReceipt | None = None
    source_release_receipt: ProjectTransferSourceReleaseReceipt | None = None
    source_fence_head: GraphHeadRef | None = None
    archive_sha256: str | None = None
    archive_size_bytes: int | None = Field(default=None, ge=1)
    restore_resume_phase: ProjectTransferPhase | None = None
    restore_diagnostic: MessageText | None = None
    revision: int = Field(ge=0)
    created_at: str
    updated_at: str

    @field_validator(
        "request_id",
        "linked_request_id",
        "project_id",
        "source_space_id",
        "target_space_id",
    )
    @classmethod
    def validate_identifier(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator(
        "source_configuration_sha256",
        "source_release_proof_sha256",
        "target_activation_proof_sha256",
        "proof_acknowledgement_sha256",
        "archive_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("transfer request digest must be lowercase SHA-256")
        return value

    @field_validator("accepted_archive_codec")
    @classmethod
    def validate_archive_codec(cls, value: str | None) -> str | None:
        if value is not None and _PROVISIONING_RUNTIME.fullmatch(value) is None:
            raise ValueError("accepted transfer archive codec is invalid")
        return value

    @model_validator(mode="after")
    def validate_protocol_shape(self) -> ProjectTransferRequestRecord:
        if self.source_space_id == self.target_space_id:
            raise ValueError("a transfer must cross spaces")
        expected_actor_space = (
            self.source_space_id if self.side == "source" else self.target_space_id
        )
        if self.initiated_by.space_id != expected_actor_space:
            raise ValueError("transfer initiator belongs to the wrong space")
        if self.phase == "operator_action_needed":
            if (
                self.side != "target"
                or self.restore_resume_phase not in _PROJECT_TRANSFER_RESTORABLE_TARGET_PHASES
                or self.restore_diagnostic is None
            ):
                raise ValueError(
                    "restored transfer action must preserve one unfinished target phase"
                )
            effective_phase = self.restore_resume_phase
        else:
            if self.restore_resume_phase is not None or self.restore_diagnostic is not None:
                raise ValueError("ordinary transfer state cannot carry restore action metadata")
            effective_phase = self.phase
        linked_fields = (
            self.linked_request_id,
            self.accepted_schema_generation,
            self.accepted_archive_codec,
            self.target_activation_proof_sha256,
        )
        if effective_phase == "awaiting_link":
            if (
                self.side != "source"
                or any(value is not None for value in linked_fields)
                or self.link_receipt is not None
            ):
                raise ValueError("only an unlinked source request may await its target")
        elif any(value is None for value in linked_fields) or self.link_receipt is None:
            raise ValueError("a linked transfer request requires its exact link receipt")
        if (
            self.accepted_schema_generation is not None
            and self.accepted_schema_generation
            != self.source_configuration.source_schema_generation
        ):
            raise ValueError("accepted transfer schema must match the bound source schema")
        if (
            self.accepted_archive_codec is not None
            and self.accepted_archive_codec
            not in self.source_configuration.supported_archive_codecs
        ):
            raise ValueError("accepted transfer codec is not supported by the source")
        rank = _PROJECT_TRANSFER_PHASE_RANK[effective_phase]
        if rank >= _PROJECT_TRANSFER_PHASE_RANK["target_admitted"]:
            if self.target_admission_receipt is None:
                raise ValueError("admitted transfer state requires its target receipt")
        elif self.target_admission_receipt is not None:
            raise ValueError("target admission receipt appears before admission")
        if rank >= _PROJECT_TRANSFER_PHASE_RANK["source_released"]:
            if self.source_release_receipt is None:
                raise ValueError("released transfer state requires its source receipt")
        elif self.source_release_receipt is not None:
            raise ValueError("source release receipt appears before release")
        if rank >= _PROJECT_TRANSFER_PHASE_RANK["source_fenced"]:
            if self.side == "source" and self.source_fence_head is None:
                raise ValueError("fenced source transfer state requires its canonical head")
        elif self.source_fence_head is not None:
            raise ValueError("source fence head appears before the fence")
        if rank >= _PROJECT_TRANSFER_PHASE_RANK["archive_bound"]:
            if self.archive_sha256 is None or self.archive_size_bytes is None:
                raise ValueError("archive-bound transfer state requires exact archive metadata")
            if self.source_fence_head is None:
                raise ValueError("archive binding requires the exact fenced source head")
        elif self.archive_sha256 is not None or self.archive_size_bytes is not None:
            raise ValueError("transfer archive metadata appears before its binding")
        if self.proof_state in {"acknowledged", "consumed"}:
            if self.proof_acknowledgement_sha256 is None:
                raise ValueError("acknowledged transfer proof requires its receipt digest")
        elif self.proof_acknowledgement_sha256 is not None:
            raise ValueError("transfer proof acknowledgment appears before acknowledgment")
        return self


class ProjectTransferImportRecord(_StrictProvisioningModel):
    """Receipt for one atomic, history-only target-side transfer import.

    This is transfer-control state, not project history.  In particular, the
    imported rows are deliberately usable only for display until a later
    publication transaction binds the receipt to a canonical/file readback
    digest.
    """

    request_id: str
    project_id: str
    archive_manifest_sha256: str
    target_manifest_sha256: str
    operational_payload_sha256: str
    status: Literal["database_imported", "complete"]
    event_id_map: dict[str, int]
    receipt_id_map: dict[str, int]
    publication_sha256: str | None = None
    created_at: str
    completed_at: str | None = None

    @field_validator("request_id", "project_id")
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=f"project transfer import {info.field_name}")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator(
        "archive_manifest_sha256",
        "target_manifest_sha256",
        "operational_payload_sha256",
        "publication_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("project transfer import digest must be lowercase SHA-256")
        return value

    @field_validator("event_id_map", "receipt_id_map")
    @classmethod
    def validate_id_map(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key.strip() or target_id < 1 for key, target_id in value.items()):
            raise ValueError("project transfer import id maps must contain positive target ids")
        return dict(sorted(value.items()))

    @field_validator("created_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            _required_timestamp(value)
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ProjectTransferImportRecord:
        if self.status == "database_imported":
            if self.publication_sha256 is not None or self.completed_at is not None:
                raise ValueError("an imported transfer receipt is not complete")
        elif self.publication_sha256 is None or self.completed_at is None:
            raise ValueError("a complete transfer receipt requires publication readback")
        if self.completed_at is not None and _required_timestamp(
            self.completed_at
        ) < _required_timestamp(self.created_at):
            raise ValueError("project transfer import completion precedes creation")
        return self


ProjectTransferUploadStatus = Literal["active", "complete", "consumed", "invalidated"]


class ProjectTransferUploadCompleteReceipt(_StrictProvisioningModel):
    """The durable byte-boundary receipt for one target-side upload."""

    request_id: str
    project_id: str
    archive_sha256: str
    archive_size_bytes: int = Field(ge=1)
    lease_boundary_sha256: str
    completed_at: str

    @field_validator("request_id", "project_id")
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=f"project transfer upload {info.field_name}")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("archive_sha256", "lease_boundary_sha256")
    @classmethod
    def validate_digest(cls, value: str, info: ValidationInfo) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError(f"project transfer upload {info.field_name} must be lowercase SHA-256")
        return value

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: str) -> str:
        _required_timestamp(value)
        return value


class ProjectTransferUploadRecord(_StrictProvisioningModel):
    """One request-bound target upload lease or its terminal receipt.

    Upload rows are operational transfer control, not project history.  An
    invalidated row is retained as an audit boundary but cannot be completed;
    a later target re-entry may replace it with a fresh lease for the same
    immutable request archive.
    """

    request_id: str
    project_id: str
    archive_sha256: str
    archive_size_bytes: int = Field(ge=1)
    lease_boundary_sha256: str
    status: ProjectTransferUploadStatus
    receipt: ProjectTransferUploadCompleteReceipt | None = None
    created_at: str
    updated_at: str
    invalidated_at: str | None = None

    @field_validator("request_id", "project_id")
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=f"project transfer upload {info.field_name}")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("archive_sha256", "lease_boundary_sha256")
    @classmethod
    def validate_digest(cls, value: str, info: ValidationInfo) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError(f"project transfer upload {info.field_name} must be lowercase SHA-256")
        return value

    @field_validator("created_at", "updated_at", "invalidated_at")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            _required_timestamp(value)
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ProjectTransferUploadRecord:
        if _required_timestamp(self.updated_at) < _required_timestamp(self.created_at):
            raise ValueError("project transfer upload update precedes creation")
        if self.status == "active":
            if self.receipt is not None or self.invalidated_at is not None:
                raise ValueError("active project transfer upload cannot retain a terminal receipt")
        elif self.status in {"complete", "consumed"}:
            if self.receipt is None or self.invalidated_at is not None:
                raise ValueError("completed project transfer upload requires its receipt")
            if (
                self.receipt.request_id != self.request_id
                or self.receipt.project_id != self.project_id
                or self.receipt.archive_sha256 != self.archive_sha256
                or self.receipt.archive_size_bytes != self.archive_size_bytes
                or self.receipt.lease_boundary_sha256 != self.lease_boundary_sha256
            ):
                raise ValueError("project transfer upload receipt does not match its lease")
            if _required_timestamp(self.receipt.completed_at) < _required_timestamp(
                self.created_at
            ):
                raise ValueError("project transfer upload completion precedes creation")
        elif self.receipt is not None or self.invalidated_at is None:
            raise ValueError(
                "invalidated project transfer upload requires only its invalidation time"
            )
        return self


class ProjectTransferRegisteredProject(_StrictProvisioningModel):
    """The stable catalog boundary committed by target activation."""

    project_id: str
    home_space_id: str
    locator: str
    name: str
    state_location: str
    state_remote: bool
    revision: int | None = Field(default=None, ge=0)
    registered_at: str

    @field_validator("project_id", "home_space_id")
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("registered_at")
    @classmethod
    def validate_registered_at(cls, value: str) -> str:
        _required_timestamp(value)
        return value


class ProjectTransferActivationReceipt(_StrictProvisioningModel):
    """One compound target-home activation boundary, without either raw proof."""

    target_request_id: str
    source_request_id: str
    project_id: str
    source_space_id: str
    target_space_id: str
    archive_sha256: str
    source_fence_head: GraphHeadRef
    source_release_proof_sha256: str
    target_activation_proof_sha256: str
    upload_lease_boundary_sha256: str
    upload_archive_sha256: str
    upload_archive_size_bytes: int = Field(ge=1)
    upload_completed_at: str
    archive_manifest_sha256: str
    target_manifest_sha256: str
    operational_payload_sha256: str
    publication_sha256: str
    import_completed_at: str
    provisioning_request_id: str
    provisioning_revision: int = Field(ge=1)
    provisioning_final_review_sha256: str
    provisioning_completed_at: str
    admitted_by: AuthorizedHuman
    registered_project: ProjectTransferRegisteredProject
    first_member: ProjectMemberRecord
    activated_at: str

    @field_validator(
        "target_request_id",
        "source_request_id",
        "project_id",
        "source_space_id",
        "target_space_id",
        "provisioning_request_id",
    )
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator(
        "archive_sha256",
        "source_release_proof_sha256",
        "target_activation_proof_sha256",
        "upload_lease_boundary_sha256",
        "upload_archive_sha256",
        "archive_manifest_sha256",
        "target_manifest_sha256",
        "operational_payload_sha256",
        "publication_sha256",
        "provisioning_final_review_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str, info: ValidationInfo) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be lowercase SHA-256")
        return value

    @field_validator(
        "upload_completed_at",
        "import_completed_at",
        "provisioning_completed_at",
        "activated_at",
    )
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        _required_timestamp(value)
        return value

    @model_validator(mode="after")
    def validate_boundary(self) -> ProjectTransferActivationReceipt:
        if self.source_space_id == self.target_space_id:
            raise ValueError("a project transfer activation must cross spaces")
        if self.provisioning_request_id != self.target_request_id:
            raise ValueError("target activation must complete its linked provisioning request")
        if self.admitted_by.space_id != self.target_space_id:
            raise ValueError("target activation actor belongs to another space")
        if (
            self.registered_project.project_id != self.project_id
            or self.registered_project.home_space_id != self.target_space_id
        ):
            raise ValueError("target activation project registration does not match its home")
        if (
            self.first_member.project_id != self.project_id
            or self.first_member.user_id != self.admitted_by.user_id
            or self.first_member.seated_by != self.admitted_by.user_id
        ):
            raise ValueError("target activation first member does not match its admitting actor")
        if self.source_fence_head.target.kind != "main":
            raise ValueError("target activation must bind the main source fence")
        if self.upload_archive_sha256 != self.archive_sha256:
            raise ValueError("target activation upload does not match its bound archive")
        activated_at = _required_timestamp(self.activated_at)
        for completed_at in (
            self.upload_completed_at,
            self.import_completed_at,
            self.provisioning_completed_at,
            self.registered_project.registered_at,
            self.first_member.seated_at,
        ):
            if _required_timestamp(completed_at) > activated_at:
                raise ValueError("target activation precedes one of its committed boundaries")
        return self


class ProjectTransferRestoreReentryReceipt(_StrictProvisioningModel):
    """One reviewed replacement-machine relay boundary after restore."""

    target_request_id: str
    source_request_id: str
    project_id: str
    source_space_id: str
    target_space_id: str
    restored_revision: int = Field(ge=0)
    resume_phase: Literal["archive_bound"]
    provisioning_revision: int = Field(ge=1)
    provisioning_final_review_sha256: str
    confirmed_by: AuthorizedHuman
    archive_sha256: str
    archive_size_bytes: int = Field(ge=1)
    replacement_lease_boundary_sha256: str
    created_at: str

    @field_validator(
        "target_request_id",
        "source_request_id",
        "project_id",
        "source_space_id",
        "target_space_id",
    )
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        try:
            return _canonical_uuid4(value, label=info.field_name.replace("_", " "))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator(
        "provisioning_final_review_sha256",
        "archive_sha256",
        "replacement_lease_boundary_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str, info: ValidationInfo) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be lowercase SHA-256")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        _required_timestamp(value)
        return value

    @model_validator(mode="after")
    def validate_boundary(self) -> ProjectTransferRestoreReentryReceipt:
        if self.source_space_id == self.target_space_id:
            raise ValueError("a restored transfer re-entry must cross spaces")
        if self.confirmed_by.space_id != self.target_space_id:
            raise ValueError("restored transfer confirmer belongs to another space")
        return self


class ProviderSkillInventoryRecord(BaseModel):
    """One durable last-known provider-native skill inventory."""

    provider: str
    host: str
    configured_binary: str
    resolved_binary: str | None = None
    provider_version: str | None = None
    command: list[str] = Field(default_factory=list)
    protocol: str | None = None
    skills: list[ProviderSkill] = Field(default_factory=list)
    inventory_hash: str | None = None
    status: Literal["refreshing", "fresh", "stale", "unavailable"]
    diagnostic: str | None = None
    refreshed_at: str | None = None
    updated_at: str


AgentTaskKind = Literal[
    "seed",
    "refresh",
    "node_chat",
    "project_chat",
    "paper_coach",
    "auto_research",
    "branch_merge",
    "episode_report",
]
AgentTaskStatus = Literal[
    "queued",
    "running",
    "pausing",
    "paused",
    "succeeded",
    "failed",
    "interrupted",
]
AgentTaskReceiptTier = Literal["summary", "diagnostic", "trace"]

# A task is still moving through these; every other status is terminal. "pausing"
# belongs here because the pause has been requested but not yet observed, so a
# caller that treats it as settled reads a state the task is about to leave.
ACTIVE_AGENT_TASK_STATUSES: frozenset[AgentTaskStatus] = frozenset({"queued", "running", "pausing"})
# A turn in one of these states is waiting on a person, not on the machine.
AWAITING_HUMAN_AGENT_TASK_STATUSES: frozenset[AgentTaskStatus] = frozenset(
    {"paused", "failed", "interrupted"}
)

# One table owns the status transitions used by the durable task lifecycle.
# Recovery actions are deliberately not represented here: Resume and Retry
# create child attempts and have additional native-session requirements.
AGENT_TASK_TRANSITIONS: dict[AgentTaskStatus, frozenset[AgentTaskStatus]] = {
    "running": frozenset({"queued"}),
    "pausing": frozenset({"queued", "running"}),
    "paused": frozenset({"queued", "running", "pausing"}),
    "succeeded": frozenset({"queued", "running", "pausing"}),
    "failed": frozenset({"queued", "running", "pausing"}),
    "interrupted": frozenset({"queued", "running", "pausing"}),
}

_EXPERIMENT_EPISODE_CONTEXT_CANDIDATE_ROLE = "experiment_episode_context_candidate"
_MISSING_EXPERIMENT_EPISODE_CONTEXT_DIAGNOSTIC = (
    "This Experiment-loop turn cannot be resumed or retried because its pre-migration "
    "root has no retained episode context candidate. Use Stop loop and press Run to start "
    "a fresh episode."
)


class AgentTaskEventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: int
    operation_id: str
    created_at: str
    level: Literal["info", "warning", "error"]
    message: str
    event_kind: Literal["message", "command"] = "message"
    command_id: str | None = None
    episode_id: str | None = None
    command_verb: str | None = None
    command_phase: Literal["start", "exit"] | None = None
    idempotency_key: str | None = None
    payload: dict[str, object] | None = None


class AgentTaskReceiptRecord(BaseModel):
    receipt_id: int
    operation_id: str
    created_at: str
    tier: AgentTaskReceiptTier
    category: str
    payload: dict[str, object]


class AgentTaskContractRecord(BaseModel):
    operation_id: str
    role: str
    created_at: str
    sha256: str
    content: str


class ChatSessionContextRecord(BaseModel):
    """Durable RCP context baseline bound to one native provider session."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    execution_machine: str = Field(min_length=1)
    native_session_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    kind: Literal["node_chat", "project_chat"]
    chat_id: str = Field(min_length=1)
    node_id: str | None = None
    protocol_version: int = Field(ge=1)
    snapshot_json: str
    snapshot_sha256: str = Field(min_length=1)
    committed_operation_id: str = Field(min_length=1)
    created_at: str
    updated_at: str


class AgentTaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    project_id: str
    kind: AgentTaskKind
    status: AgentTaskStatus
    request: dict[str, object]
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    status_message: str
    error: str | None = None
    applied_revision: int | None = None
    result: dict[str, object] | None = None
    attempt: int = 1
    parent_operation_id: str | None = None
    episode_id: str | None = None
    runtime_id: str = ""
    #: How that runtime is named to a human. Derived here so a surface reporting
    #: what actually ran never maps a durable id back to the registry itself.
    runtime_label: str = ""
    native_session_id: str | None = None
    history_only: bool = False
    stage_host: str | None = None
    stage_root: str | None = None
    graph_target: GraphTargetRef = Field(default_factory=GraphTargetRef)
    write_scope_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    estimate_seconds: float = 300.0
    estimate_samples: int = 0
    phase: str = "queued"
    last_activity_at: str | None = None
    authorized_by: AuthorizedHuman | None = None
    dispatch_authority: AgentDispatchAuthority | None = None
    elapsed_seconds: float = 0.0
    progress: float = 0.0
    can_pause: bool = False
    can_resume: bool = False
    can_retry: bool = False
    # The lifecycle questions a surface asks about a task, answered here so no
    # reader has to ask them of `status` itself.
    active: bool = False
    queued: bool = False
    pausing: bool = False
    awaiting_human: bool = False
    paused: bool = False
    failed: bool = False
    settled: bool = False
    finished: bool = False
    status_label: str = ""
    visible: bool = True

    @model_validator(mode="after")
    def validate_provider_runtime(self) -> AgentTaskRecord:
        provider = self.request.get("provider")
        if not isinstance(provider, str) or not provider:
            if self.runtime_id:
                raise ValueError("an agent runtime requires a provider")
            return self
        try:
            legacy = legacy_runtime_id(provider)
        except ValueError:
            # An old row may name a provider RCP no longer supports. Nothing can
            # launch it, so it needs no runtime identity; keep it readable for
            # project deletion and forensic export instead of failing every read.
            return self
        if not self.runtime_id:
            self.runtime_id = legacy
        require_runtime_id(provider, self.runtime_id)
        self.runtime_label = runtime_label(provider, self.runtime_id)
        return self


# Fields a stored task carries because the row projection computed them, not
# because a caller supplied them. Equality between a requested task and its
# committed twin ignores these.
AGENT_TASK_PROJECTION_FIELDS: frozenset[str] = frozenset(
    {
        "elapsed_seconds",
        "progress",
        "can_pause",
        "can_resume",
        "can_retry",
        "active",
        "queued",
        "pausing",
        "awaiting_human",
        "paused",
        "failed",
        "settled",
        "finished",
        "status_label",
    }
)


class ResultViewRecord(BaseModel):
    """Private binding and lifecycle metadata for one conversation result view."""

    model_config = ConfigDict(extra="forbid", strict=True)

    view_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    project_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    origin_operation_id: str = Field(min_length=1)
    latest_operation_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str
    reasoning: str
    run_on: str = Field(min_length=1)
    native_session_id: str = Field(min_length=1)
    stage_host: str
    stage_root: str = Field(min_length=1)
    source_name: str = Field(min_length=1, max_length=255)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(gt=0, le=CHAT_ARTIFACT_MAX_FILE_BYTES)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)
    kept_filename: str | None = Field(default=None, min_length=1, max_length=255)
    kept_at: str | None = Field(default=None, min_length=1)

    @field_validator("source_name")
    @classmethod
    def source_name_is_plain_html(cls, value: str) -> str:
        return _plain_html_name(value, label="result view source name")

    @field_validator("kept_filename")
    @classmethod
    def kept_filename_is_plain_html(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _plain_html_name(value, label="kept result view filename")

    @field_validator("created_at", "updated_at", "expires_at", "kept_at")
    @classmethod
    def timestamps_are_parseable(cls, value: str | None) -> str | None:
        if value is not None:
            _required_timestamp(value)
        return value

    @model_validator(mode="after")
    def lifecycle_is_coherent(self) -> ResultViewRecord:
        if (self.kept_filename is None) != (self.kept_at is None):
            raise ValueError("a kept result view requires both its filename and kept_at")
        created_at = _required_timestamp(self.created_at)
        updated_at = _required_timestamp(self.updated_at)
        expires_at = _required_timestamp(self.expires_at)
        if updated_at < created_at:
            raise ValueError("result view updated_at precedes created_at")
        if expires_at < created_at:
            raise ValueError("result view expires_at precedes created_at")
        if self.kept_at is not None and _required_timestamp(self.kept_at) < created_at:
            raise ValueError("result view kept_at precedes created_at")
        return self


AutoResearchRole = Literal["orchestrator", "worker"]
AutoResearchMessageRole = Literal["human", "orchestrator", "worker"]
AutoResearchRecoveryStatus = Literal["pending", "admitted", "exhausted", "blocked"]
AutoResearchRecoveryMode = Literal["exact", "clean", "blocked"]
AutoResearchChildExperimentState = Literal["pending", "running", "cancelled", "terminal"]
AutoResearchChildAdmissionState = Literal["accepted", "reflected", "cancelled"]
AutoResearchLifecycleNoticeState = Literal["pending", "delivered", "acknowledged"]
AutoResearchInboxReceiptMode = Literal["harvest", "clear"]
AutoResearchFinishDisposition = Literal["blocked", "completed"]
AutoResearchCommandFileKind = Literal["apply", "instruction", "goal"]
AutoResearchFinishBlockerKind = Literal[
    "spawned_work",
    "experiment_episode",
    "experiment_replacement",
    "lifecycle_notice",
    "child_admission",
]

EpisodeMode = Literal["auto_research", "experiment_loop"]
EpisodeStatus = Literal[
    "queued",
    "running",
    "stopping",
    "wrapping_up",
    "needs_action",
    "completed",
    "stopped",
    "failed",
]
EpisodeEnding = Literal["completed", "exhausted", "stopped", "failed", "human_pause"]
EpisodeWrapupState = Literal[
    "not_started",
    "pending",
    "running",
    "ready",
    "failed",
    "skipped",
    "legacy_unavailable",
]
EpisodeReportAttemptStatus = Literal["queued", "running", "succeeded", "failed"]


class EpisodeRecord(BaseModel):
    """The mode-neutral parent and lifecycle for one bounded episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    project_id: str
    mode: EpisodeMode
    control_node_id: str | None = None
    graph_target: GraphTargetRef = Field(default_factory=GraphTargetRef)
    graph_base_head: GraphHeadRef | None = None
    root_operation_id: str | None = None
    status: EpisodeStatus
    invocation_ceiling: int = Field(ge=1)
    invocations_used: int = Field(default=0, ge=0)
    authorized_by: AuthorizedHuman | None = None
    stop_requested_at: str | None = None
    stop_settled_at: str | None = None
    ending: EpisodeEnding | None = None
    ending_diagnostic: str | None = None
    wrapup_state: EpisodeWrapupState = "not_started"
    wrapup_error: str | None = None
    report_attempts_used: int = Field(default=0, ge=0, le=3)
    created_at: str
    updated_at: str
    ended_at: str | None = None

    @model_validator(mode="after")
    def lifecycle_is_coherent(self) -> EpisodeRecord:
        if self.invocations_used > self.invocation_ceiling:
            raise ValueError("episode invocations used exceed the authorized ceiling")
        if self.ending == "stopped" and self.wrapup_state != "skipped":
            raise ValueError("a stopped episode must skip report generation")
        if self.wrapup_state == "skipped" and self.ending != "stopped":
            raise ValueError("only a stopped episode may skip report generation")
        if self.mode == "experiment_loop" and not self.control_node_id:
            raise ValueError("an Experiment-loop episode requires its control node")
        if self.mode == "auto_research" and self.control_node_id is not None:
            raise ValueError("an Auto-research episode cannot carry an Experiment control node")
        if self.graph_target.kind == "main" and self.graph_base_head is not None:
            raise ValueError("a main-target episode cannot carry a branch base head")
        if self.graph_target.kind == "branch":
            if self.graph_base_head is None or self.graph_base_head.target.kind != "main":
                raise ValueError("a branch-target episode requires its immutable main base head")
            if self.mode == "auto_research" and self.graph_target.branch_id != self.episode_id:
                raise ValueError("an Auto-research episode must own its same-id graph branch")
        if self.wrapup_state in {"ready", "failed"} and self.ending is None:
            raise ValueError("a terminal episode wrap-up requires its semantic ending")
        return self

    @property
    def invocations_remaining(self) -> int:
        return max(0, self.invocation_ceiling - self.invocations_used)


class EpisodeBudgetMeter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invocation_ceiling: int = Field(ge=1)
    invocations_used: int = Field(ge=0)
    invocations_remaining: int = Field(ge=0)
    observed_input_tokens: int = Field(default=0, ge=0)
    observed_generated_tokens: int = Field(default=0, ge=0)


class EpisodeInvocationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    operation_id: str
    invocation_number: int = Field(ge=1)
    created_at: str


class EpisodeReportAttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    episode_id: str
    attempt_number: int = Field(ge=1, le=3)
    allocation_operation_id: str
    status: EpisodeReportAttemptStatus
    error: str | None = None
    created_at: str
    updated_at: str
    finished_at: str | None = None


class EpisodeReportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    episode_id: str
    attempt_id: str
    allocation_operation_id: str
    ending: EpisodeEnding
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    html: str
    created_at: str

    @field_validator("html")
    @classmethod
    def html_is_a_bounded_utf8_artifact(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("episode report HTML contains NUL bytes")
        if len(value.encode("utf-8")) > CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("episode report HTML exceeds the artifact size limit")
        return value

    @model_validator(mode="after")
    def digest_matches_html(self) -> EpisodeReportRecord:
        if hashlib.sha256(self.html.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("episode report HTML does not match its digest")
        return self


class EpisodeWrapupRecord(BaseModel):
    """The immutable restart fence for one episode's hidden report allocation."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    ending: EpisodeEnding | None
    partial: bool
    concluding_operation_id: str | None = None
    allocation_operation_id: str | None = None
    provider: str | None = None
    run_on: str | None = None
    execution_host: str | None = None
    native_session_id: str | None = None
    stage_host: str | None = None
    stage_root: str | None = None
    skill_id: str | None = None
    skill_version: str | None = None
    output_name: str | None = None
    output_path: str | None = None
    receipt_json: str
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    state: EpisodeWrapupState
    diagnostic: str | None = None
    created_at: str
    updated_at: str
    finished_at: str | None = None

    @model_validator(mode="after")
    def restart_fence_is_coherent(self) -> EpisodeWrapupRecord:
        try:
            receipt = json.loads(self.receipt_json)
        except json.JSONDecodeError as exc:
            raise ValueError("episode wrap-up receipt is invalid JSON") from exc
        if not isinstance(receipt, dict):
            raise ValueError("episode wrap-up receipt must be a JSON object")
        compact = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        if compact != self.receipt_json:
            raise ValueError("episode wrap-up receipt must use canonical compact JSON")
        if hashlib.sha256(self.receipt_json.encode("utf-8")).hexdigest() != self.receipt_sha256:
            raise ValueError("episode wrap-up receipt does not match its digest")
        if self.state == "skipped" and self.ending != "stopped":
            raise ValueError("only a stopped episode may skip its wrap-up")
        if self.ending == "stopped" and self.state != "skipped":
            raise ValueError("a stopped episode must skip its wrap-up")
        if self.state != "legacy_unavailable" and self.ending is None:
            raise ValueError("a new episode wrap-up requires its semantic ending")
        if self.output_name is not None:
            _plain_html_name(self.output_name, label="episode report output name")
        return self


class EpisodeInvocationCeilingReached(ValueError):
    pass


class EpisodeNotRunning(ValueError):
    pass


class EpisodeReportAttemptLimitReached(ValueError):
    pass


class EpisodeReportConflict(ValueError):
    pass


class AutoResearchStateRecord(BaseModel):
    """Mode-specific state attached to one generic Auto-research episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    starting_instruction: str | None = Field(default=None, max_length=16_000)
    created_at: str
    updated_at: str


class AutoResearchInvocationRecord(BaseModel):
    """One Auto-research task and the operational allocation it belongs to."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    operation_id: str
    allocation_operation_id: str
    role: AutoResearchRole
    actor_operation_id: str
    control_node_id: str | None = None
    created_at: str


class AutoResearchChildWorkRecord(BaseModel):
    """One ordinary Work actor admitted and routed by an Auto-research parent."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    episode_id: str
    project_id: str
    control_node_id: str
    root_operation_id: str
    current_operation_id: str
    admitted_by_operation_id: str
    instruction: str = Field(min_length=1, max_length=16_000)
    instruction_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    stop_requested_at: str | None = None
    created_at: str
    updated_at: str

    @model_validator(mode="after")
    def instruction_matches_digest(self) -> AutoResearchChildWorkRecord:
        if not self.instruction.strip():
            raise ValueError("an Auto-research child Work instruction must not be blank")
        if hashlib.sha256(self.instruction.encode("utf-8")).hexdigest() != self.instruction_sha256:
            raise ValueError("the child Work instruction does not match its digest")
        return self


class AutoResearchChildExperimentRecord(BaseModel):
    """Parent routing and immutable launch intent for one child Experiment episode."""

    model_config = ConfigDict(extra="forbid")

    child_episode_id: str
    auto_research_episode_id: str
    project_id: str
    control_node_id: str
    state: AutoResearchChildExperimentState
    replaces_episode_id: str | None = None
    request: dict[str, object]
    goal_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    parent_operation_id: str
    terminal_diagnostic: str | None = None
    created_at: str
    updated_at: str


class AutoResearchExperimentAllowance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=5)
    used: int = Field(ge=0)
    remaining: int = Field(ge=0)

    @model_validator(mode="after")
    def accounting_is_coherent(self) -> AutoResearchExperimentAllowance:
        if self.used > self.total or self.remaining != self.total - self.used:
            raise ValueError("the child Experiment allowance accounting is inconsistent")
        return self


class AutoResearchChildAdmissionRecord(BaseModel):
    """A durable command admission awaiting or naming its reflected child route."""

    model_config = ConfigDict(extra="forbid")

    admission_id: str
    episode_id: str
    project_id: str
    child_kind: Literal["work", "experiment"]
    child_id: str
    state: AutoResearchChildAdmissionState
    created_at: str
    updated_at: str


class AutoResearchLifecycleNoticeRecord(BaseModel):
    """An RCP-authored lifecycle fact, deliberately separate from agent mail."""

    model_config = ConfigDict(extra="forbid")

    notice_id: str
    episode_id: str
    source_kind: str
    source_id: str
    source_event: str
    source_attempt: int = Field(default=1, ge=1)
    state: AutoResearchLifecycleNoticeState = "pending"
    payload: dict[str, object]
    created_at: str
    delivered_at: str | None = None
    delivery_operation_id: str | None = None
    acknowledged_at: str | None = None
    acknowledged_by: str | None = None

    @model_validator(mode="after")
    def delivery_state_is_coherent(self) -> AutoResearchLifecycleNoticeRecord:
        if (self.delivered_at is None) != (self.delivery_operation_id is None):
            raise ValueError("a lifecycle delivery requires both its time and operation")
        if (self.acknowledged_at is None) != (self.acknowledged_by is None):
            raise ValueError("a lifecycle acknowledgment requires both its time and actor")
        expected = (
            "acknowledged"
            if self.acknowledged_at is not None
            else "delivered"
            if self.delivered_at is not None
            else "pending"
        )
        if self.state != expected:
            raise ValueError("the lifecycle notice state does not match its timestamps")
        return self


class AutoResearchInboxReceiptRecord(BaseModel):
    """The exact lifecycle-notice snapshot acknowledged by one keyed inbox effect."""

    model_config = ConfigDict(extra="forbid")

    effect_id: str
    episode_id: str
    mode: AutoResearchInboxReceiptMode
    notice_ids: list[str]
    count: int = Field(ge=0)
    notices: list[AutoResearchLifecycleNoticeRecord] = Field(default_factory=list)
    acknowledged_by: str
    created_at: str

    @model_validator(mode="after")
    def result_matches_mode(self) -> AutoResearchInboxReceiptRecord:
        if self.count != len(self.notice_ids) or len(set(self.notice_ids)) != self.count:
            raise ValueError("an inbox receipt count must match its unique notice ids")
        if self.mode == "clear" and self.notices:
            raise ValueError("a clear receipt must not retain notice bodies")
        if self.mode == "harvest" and [item.notice_id for item in self.notices] != self.notice_ids:
            raise ValueError("a harvest receipt body must match its notice ids in order")
        return self


class AutoResearchFinishReceiptRecord(BaseModel):
    """The complete immutable result of one keyed guarded-Finish decision."""

    model_config = ConfigDict(extra="forbid")

    effect_id: str
    episode_id: str
    actor_operation_id: str = Field(min_length=1)
    disposition: AutoResearchFinishDisposition
    blocker_count: int = Field(ge=0)
    result: dict[str, object]
    result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: str

    @model_validator(mode="after")
    def result_matches_decision(self) -> AutoResearchFinishReceiptRecord:
        compact = json.dumps(
            self.result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if hashlib.sha256(compact.encode("utf-8")).hexdigest() != self.result_sha256:
            raise ValueError("the guarded-Finish result does not match its digest")
        if self.result.get("episode_id") != self.episode_id:
            raise ValueError("the guarded-Finish result belongs to another episode")
        blockers = self.result.get("blockers")
        if self.disposition == "blocked":
            if set(self.result) != {"episode_id", "blockers"} or not isinstance(blockers, list):
                raise ValueError("a blocked Finish receipt requires its complete blocker array")
            parsed = [AutoResearchFinishBlocker.model_validate(item) for item in blockers]
            if self.blocker_count == 0 or len(parsed) != self.blocker_count:
                raise ValueError("a blocked Finish receipt count must match its blockers")
        elif (
            set(self.result) != {"episode_id", "status", "ending"}
            or self.blocker_count != 0
            or blockers is not None
            or self.result.get("ending") != "completed"
        ):
            raise ValueError("a completed Finish receipt requires its fenced episode result")
        return self


class AutoResearchCommandFileRecord(BaseModel):
    """Immutable text snapshotted before a keyed staged command takes effect."""

    model_config = ConfigDict(extra="forbid")

    command_id: str
    episode_id: str
    operation_id: str
    kind: AutoResearchCommandFileKind
    filename: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content: str
    created_at: str

    @field_validator("filename")
    @classmethod
    def filename_is_direct(cls, value: str) -> str:
        if not value or value in {".", ".."} or "\x00" in value or Path(value).name != value:
            raise ValueError("a staged command snapshot requires one direct filename")
        return value

    @model_validator(mode="after")
    def content_matches_digest(self) -> AutoResearchCommandFileRecord:
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("the staged command snapshot does not match its digest")
        return self


class AutoResearchApplyResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apply_id: str
    episode_id: str
    operation_id: str
    patch_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    result: dict[str, object]
    created_at: str


class AutoResearchFinishBlocker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: AutoResearchFinishBlockerKind
    blocker_id: str
    state: str
    action: str


class AutoResearchExperimentAllowanceReached(ValueError):
    def __init__(self, allowance: AutoResearchExperimentAllowance) -> None:
        self.allowance = allowance
        super().__init__(
            "the Auto-research child Experiment allowance is exhausted "
            f"({allowance.used}/{allowance.total})"
        )


class AutoResearchRecoveryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recovery_id: str
    episode_id: str
    operation_id: str
    failure_kind: str
    retry_mode: AutoResearchRecoveryMode
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(ge=1)
    status: AutoResearchRecoveryStatus
    next_attempt_at: str | None = None
    diagnostic: str
    admitted_operation_id: str | None = None
    created_at: str
    updated_at: str


class AutoResearchMessageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    episode_id: str
    sender_role: AutoResearchMessageRole
    sender_task_id: str | None = None
    authorized_by: AuthorizedHuman | None = None
    recipient_task_id: str
    control_node_id: str | None = None
    body: str = Field(min_length=1, max_length=16_000)
    created_at: str
    delivered_at: str | None = None
    delivery_operation_id: str | None = None

    @field_validator("body")
    @classmethod
    def message_body_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Auto-research message body must not be blank")
        return stripped

    @model_validator(mode="after")
    def only_human_messages_carry_human_identity(self) -> AutoResearchMessageRecord:
        if self.sender_role != "human" and self.authorized_by is not None:
            raise ValueError("an agent Auto-research message cannot claim a human sender snapshot")
        return self


class AutoResearchActorBinding(BaseModel):
    """Canonical actor identity plus the newest task carrying its native session."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    actor_operation_id: str
    role: AutoResearchRole
    control_node_id: str | None = None
    current_operation_id: str
    native_session_id: str | None = None
    stage_host: str | None = None
    stage_root: str | None = None


class AutoResearchActorBusy(ValueError):
    """One Auto-research actor already has an unresolved leaf."""

    def __init__(self, actor_operation_id: str, operation_id: str) -> None:
        self.actor_operation_id = actor_operation_id
        self.operation_id = operation_id
        super().__init__(
            f"Auto-research actor {actor_operation_id} already has unresolved task {operation_id}."
        )


class AgentCommandInvocationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str
    episode_id: str | None = None
    operation_id: str
    verb: str
    idempotency_key: str | None = None
    started_at: str
    start_payload: dict[str, object]
    exited_at: str | None = None
    status: Literal["ok", "invalid", "unavailable"] | None = None
    exit_payload: dict[str, object] | None = None


class ExperimentEpisodeRecord(BaseModel):
    """Joined projection of an Experiment episode parent and its mode state.

    The binding is what an automatic watcher wake resumes. It is committed only
    by a mechanically successful joint handoff, so a failed first invocation
    never leaves a session an automatic wake would try to continue. A graph-only
    rejection is still a truthful accepted operational handoff. Project,
    control-node, and Stop fields are read from ``episodes``; they are not
    duplicated in the mode-specific child row.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    project_id: str
    control_node_id: str
    graph_target: GraphTargetRef = Field(default_factory=GraphTargetRef)
    provider: str | None = None
    execution_machine: str | None = None
    execution_host: str = ""
    native_session_id: str | None = None
    stage_host: str | None = None
    stage_root: str | None = None
    chat_id: str | None = None
    last_turn_operation_id: str | None = None
    last_turn_invocation: int | None = Field(default=None, ge=1)
    last_graph_result: str | None = None
    last_watcher_ids: list[str] = Field(default_factory=list)
    context_baseline: dict[str, object] = Field(default_factory=dict)
    session_diagnostic: str | None = None
    stop_requested_at: str | None = None
    stop_settled_at: str | None = None
    created_at: str
    updated_at: str

    @property
    def session_bound(self) -> bool:
        """Whether an automatic wake has a complete binding to resume."""

        return bool(
            self.native_session_id
            and self.provider
            and self.execution_machine
            and self.stage_root
            and self.chat_id
        )


class ExperimentLoopRuntime(BaseModel):
    """Operational state of the newest bounded episode for one Experiment."""

    episode_id: str | None = None
    invocations_used: int = Field(default=0, ge=0)
    invocation_ceiling: int | None = Field(default=None, ge=1)
    control_revision: int | None = Field(default=None, ge=0)
    active: bool = False
    # A parent row still occupying this Experiment, which is what admission
    # refuses a second episode against. Wider than `active`: a settled turn that
    # armed nothing leaves the parent live with no work left to wake it.
    episode_live: bool = False
    paused: bool = False
    task_active: bool = False
    detached_work_active: bool = False
    watcher_degraded: bool = False
    watcher_completion_pending: bool = False
    episode_exited: bool = False
    decision_bundle: list[dict[str, object]] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    stop_requested: bool = False
    stop_settled: bool = False
    session_bound: bool = False
    session_diagnostic: str | None = None
    provider: str | None = None
    model: str | None = None
    reasoning: str | None = None
    run_on: str | None = None
    execution_host: str | None = None
    run_truth_scope: list[str] | None = None
    chat_id: str | None = None
    current_operation_id: str | None = None
    current_status: str | None = None
    current_phase: str | None = None
    current_status_message: str | None = None
    current_last_activity_at: str | None = None
    current_invocation: int | None = Field(default=None, ge=1)


class ExperimentEpisodeProjectionSnapshot(BaseModel):
    """One transactionally coherent Experiment episode read model input."""

    model_config = ConfigDict(extra="forbid")

    episode: EpisodeRecord
    tasks: list[AgentTaskRecord] = Field(default_factory=list)
    budget: EpisodeBudgetMeter
    report: EpisodeReportRecord | None = None


class ExperimentControlProjectionSnapshot(BaseModel):
    """Runtime and episode inputs observed in one SQLite read transaction."""

    model_config = ConfigDict(extra="forbid")

    runtime: ExperimentLoopRuntime
    episode: ExperimentEpisodeProjectionSnapshot | None = None
    latest_report_episode_id: str | None = None


AgentUsageCountReason = Literal["counted", "duplicate", "invalid"]


class AgentUsageRecord(BaseModel):
    usage_id: str
    project_id: str
    operation_id: str
    task_kind: AgentTaskKind
    provider: str
    model: str | None = None
    provider_profile: str
    provider_event_type: str
    dedupe_key: str
    counted: bool
    count_reason: AgentUsageCountReason
    created_at: str
    processed_input_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_write_input_tokens: int = Field(default=0, ge=0)
    reasoning_output_tokens: int = Field(default=0, ge=0)
    reported_input_tokens: int | None = Field(default=None, ge=0)
    reported_output_tokens: int | None = Field(default=None, ge=0)
    reported_total_tokens: int | None = Field(default=None, ge=0)
    provider_fields: dict[str, object] = Field(default_factory=dict)


class AgentUsageCell(BaseModel):
    task_kind: AgentTaskKind
    provider: str
    processed_input_tokens: int = 0
    generated_tokens: int = 0
    cached_input_tokens: int = 0
    counted_records: int = 0


class AgentUsageMetric(BaseModel):
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_share: float = 0.0
    block_percent: float = 5.0
    block_tokens: float = 0.0
    cells: list[AgentUsageCell] = Field(default_factory=list)


class AgentUsageSnapshot(BaseModel):
    project_id: str
    input_processed: AgentUsageMetric
    generated: AgentUsageMetric
    counted_records: int = 0
    excluded_records: int = 0
    records: list[AgentUsageRecord] = Field(default_factory=list)


WatcherStatus = Literal["active", "degraded", "completed", "stopped"]


class WatcherClaimConflict(ValueError):
    """A watcher delivery already won the atomic claim."""


class ResultViewConflict(ValueError):
    """A result-view revision was based on bytes that are no longer current."""


class WatcherStopRequest(BaseModel):
    """An Experiment agent's narrow request to retire one staged observer."""

    model_config = ConfigDict(extra="forbid")

    stop_watcher_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("stop_watcher_id", "reason")
    @classmethod
    def is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("watcher stop fields must not be blank")
        return stripped


class WatcherContinuation(BaseModel):
    """RCP-bound policy needed to create a fresh Work wake."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None
    reasoning: str | None = None
    run_on: str
    run_truth_scope: list[str] | None = None
    patch_kind: Literal["work", "experiment_loop"] = "work"
    control_node_id: str | None = None
    control_revision: int | None = Field(default=None, ge=0)
    control_episode_id: str | None = None
    control_invocation: int | None = Field(default=None, ge=1)
    control_invocation_ceiling: int | None = Field(default=None, ge=1)
    control_decision_bundle: list[dict[str, object]] = Field(default_factory=list)
    control_completion_criteria: list[str] = Field(default_factory=list)
    workflow_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    invoked_workflow_ids: list[str] = Field(default_factory=list)
    invoked_skill_ids: list[str] = Field(default_factory=list)
    resolved_skill_packages: list[SkillReference] = Field(default_factory=list)


class NodeStatusGraphCondition(BaseModel):
    """Wake when one canonical node reaches any named status."""

    model_config = ConfigDict(extra="forbid", strict=True)

    node_id: str = Field(min_length=1)
    status_in: list[str] = Field(min_length=1)

    @field_validator("node_id")
    @classmethod
    def node_id_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("graph condition node_id must not be blank")
        return stripped

    @field_validator("status_in")
    @classmethod
    def statuses_are_unique_and_not_blank(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("graph condition statuses must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("graph condition statuses must be unique")
        return sorted(normalized)


class ProposalResolvedGraphCondition(BaseModel):
    """Wake when a Proposal related to one canonical node is resolved."""

    model_config = ConfigDict(extra="forbid", strict=True)

    node_id: str = Field(min_length=1)
    proposal_resolved: Literal[True]

    @field_validator("proposal_resolved", mode="before")
    @classmethod
    def proposal_resolved_is_literal_true(cls, value: object) -> object:
        if type(value) is not bool or value is not True:
            raise ValueError("proposal_resolved must be the JSON literal true")
        return value

    @field_validator("node_id")
    @classmethod
    def node_id_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("graph condition node_id must not be blank")
        return stripped


GraphCondition = Annotated[
    NodeStatusGraphCondition | ProposalResolvedGraphCondition,
    Field(union_mode="left_to_right"),
]


class WatcherDeliveryRecord(BaseModel):
    """Durable delivery state shared by external and graph watchers."""

    model_config = ConfigDict(extra="forbid")

    watcher_id: str
    project_id: str
    origin_operation_id: str
    origin_task_kind: Literal["node_chat", "project_chat", "auto_research"]
    chat_id: str
    node_id: str | None = None
    episode_id: str | None = None
    graph_target: GraphTargetRef = Field(default_factory=GraphTargetRef)
    execution_host: str = ""
    continuation: WatcherContinuation
    status: WatcherStatus = "active"
    created_at: str
    completed_at: str | None = None
    notified: bool = False
    notification_operation_id: str | None = None
    stopped_by: Literal["human", "loop", "agent"] | None = None
    stop_reason: str | None = None
    stopped_at: str | None = None
    stop_operation_id: str | None = None


class WatcherRecord(WatcherDeliveryRecord):
    """Durable external observer checked from a fresh login shell."""

    check_command: str
    log_path: str
    cwd: str
    last_checked_at: str | None = None
    last_exit_code: int | None = None
    last_error: str | None = None
    next_check_at: str | None = None
    consecutive_error_count: int = Field(default=0, ge=0)
    group_id: str | None = None
    group_label: str | None = None


class GraphWatcherRecord(WatcherDeliveryRecord):
    """Durable canonical-graph condition with no shell-check fields."""

    condition: GraphCondition
    armed_revision: int | None = Field(default=None, ge=0)
    last_evaluated_at: str | None = None
    status: Literal["active", "completed", "stopped"] = "active"

    @property
    def last_checked_at(self) -> str | None:
        return self.last_evaluated_at

    @property
    def last_exit_code(self) -> None:
        return None

    @property
    def last_error(self) -> None:
        return None

    @property
    def next_check_at(self) -> None:
        return None

    @property
    def consecutive_error_count(self) -> int:
        return 0

    @property
    def group_id(self) -> None:
        return None

    @property
    def group_label(self) -> None:
        return None


StoredWatcherRecord = WatcherRecord | GraphWatcherRecord


class ExperimentWatcherResourceRecord(BaseModel):
    """The current node-and-episode owner of one Experiment watcher file."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    control_node_id: str
    episode_id: str
    graph_target: GraphTargetRef = Field(default_factory=GraphTargetRef)
    execution_host: str
    wake_task_kind: Literal["node_chat"]
    wake_chat_id: str
    continuation: WatcherContinuation
    watcher_snapshot_token: str


def watcher_next_check_at(
    watcher_id: str,
    checked_at: str,
    consecutive_error_count: int,
) -> str:
    """Return one durable, identity-jittered watcher due time."""

    if consecutive_error_count < 0:
        raise ValueError("watcher error count cannot be negative")
    try:
        base = datetime.fromisoformat(checked_at)
    except ValueError as exc:
        raise ValueError("watcher check time must be ISO 8601") from exc
    if consecutive_error_count == 0:
        delay = WATCHER_HEALTHY_INTERVAL_SECONDS
    else:
        delay = WATCHER_ERROR_BACKOFF_SECONDS[
            min(consecutive_error_count - 1, len(WATCHER_ERROR_BACKOFF_SECONDS) - 1)
        ]
    fraction = int.from_bytes(hashlib.sha256(watcher_id.encode("utf-8")).digest()[:8], "big")
    unit = fraction / ((1 << 64) - 1)
    jitter = 1 + WATCHER_SCHEDULE_JITTER_RATIO * (2 * unit - 1)
    return (base + timedelta(seconds=delay * jitter)).isoformat()


_EXPERIMENT_EPISODE_PINNED_FIELDS = (
    "run_on",
    "run_truth_scope",
    "chat_id",
    "control_node_id",
    "control_revision",
    "control_episode_id",
    "control_invocation_ceiling",
    "control_decision_bundle",
    "control_completion_criteria",
)


def _experiment_pinned_value(request: dict[str, object], field: str) -> object:
    value = request.get(field)
    if field == "run_truth_scope" and isinstance(value, list):
        return sorted({str(item) for item in value})
    return value


def _canonical_uuid4(value: object, *, label: str) -> str:
    identifier = str(value)
    try:
        parsed = uuid.UUID(identifier)
    except ValueError as exc:
        raise RuntimeError(f"RCP {label} is invalid.") from exc
    if str(parsed) != identifier or parsed.version != 4:
        raise RuntimeError(f"RCP {label} is not a canonical UUIDv4.")
    return identifier


def _canonical_space_id(value: object) -> str:
    return _canonical_uuid4(value, label="space identity")


def _stored_space_kind(value: object) -> SpaceKind:
    if value == "personal" or value == "team":
        return value
    raise RuntimeError("RCP space kind is invalid.")


def normalize_space_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("space name must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError("space name must not be blank")
    if any(character in normalized for character in ("\n", "\r", "\u2028", "\u2029")):
        raise ValueError("space name must be a single line")
    if len(normalized) > SPACE_NAME_MAX_LENGTH:
        raise ValueError(f"space name must be at most {SPACE_NAME_MAX_LENGTH} characters")
    return normalized


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _new_member_token() -> tuple[str, str]:
    token = f"rcp_{secrets.token_urlsafe(32)}"
    return token, _sha256(token)


def _new_session_token() -> tuple[str, str]:
    token = f"rcp_session_{secrets.token_urlsafe(32)}"
    return token, _sha256(token)


def _new_enrollment_code(kind: Literal["bootstrap", "invite"]) -> tuple[str, str, str]:
    code_id = secrets.token_urlsafe(12)
    secret = secrets.token_urlsafe(32)
    return f"rcp_{kind}_{code_id}.{secret}", code_id, _sha256(secret)


def _parse_enrollment_code(
    code: str,
) -> tuple[Literal["bootstrap", "invite"], str, str] | None:
    if not isinstance(code, str) or len(code) > TEAM_ENROLLMENT_CODE_MAX_LENGTH or "." not in code:
        return None
    public, secret = code.split(".", 1)
    if not secret:
        return None
    for kind in ("bootstrap", "invite"):
        prefix = f"rcp_{kind}_"
        if public.startswith(prefix) and len(public) > len(prefix):
            return kind, public[len(prefix) :], _sha256(secret)
    return None


def _discard_failed_team_initialization(path: Path, expected_space_id: str) -> None:
    """Remove only the unopened team database created by this failed init attempt."""

    if not path.exists():
        return
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            identity = connection.execute(
                "SELECT space_id, space_kind FROM space_identity WHERE singleton = 1"
            ).fetchone()
            user_count = connection.execute("SELECT COUNT(*) FROM space_users").fetchone()[0]
    except (OSError, sqlite3.Error):
        return
    if identity != (expected_space_id, "team") or user_count != 0:
        return
    for candidate in (path, path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm")):
        candidate.unlink(missing_ok=True)


_PROJECT_ID_TABLES = (
    "projects",
    "project_members",
    "paper_drafts",
    "writing_sessions",
    "chat_session_contexts",
    "result_views",
    "graph_runs",
    "episodes",
    "agent_usage",
    "watchers",
    "graph_watcher_reconciliation",
    "auto_research_child_work",
    "auto_research_child_experiments",
    "auto_research_child_admissions",
)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _plain_html_name(value: str, *, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\x00" in value:
        raise ValueError(f"{label} must be a plain base name")
    if Path(value).suffix.casefold() != ".html":
        raise ValueError(f"{label} must end in .html")
    return value


def _required_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("result view timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("result view timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _result_view_is_visible(
    record: ResultViewRecord,
    *,
    as_of: datetime | None,
) -> bool:
    if record.kept_filename is not None:
        return True
    return _required_timestamp(record.expires_at) > _result_view_reference_time(as_of)


def _result_view_reference_time(as_of: datetime | None) -> datetime:
    current = as_of or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("result view visibility time must include a timezone")
    return current.astimezone(UTC)


def _validated_result_view_html(record: ResultViewRecord, data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("result view HTML must be bytes")
    if len(data) > CHAT_ARTIFACT_MAX_FILE_BYTES:
        raise ValueError("result view HTML exceeds its byte limit")
    if len(data) != record.size_bytes:
        raise ValueError("result view HTML size does not match its metadata")
    if hashlib.sha256(data).hexdigest() != record.content_sha256:
        raise ValueError("result view HTML digest does not match its metadata")
    if validate_artifact_bytes(record.source_name, data) != "text/html":
        raise ValueError("result view must be HTML")
    return data.decode("utf-8")


def _result_view_html_bytes(record: ResultViewRecord, html: object) -> bytes:
    if not isinstance(html, str):
        raise ValueError("stored result view HTML is invalid")
    data = html.encode("utf-8")
    _validated_result_view_html(record, data)
    return data


__all__ = [
    "ACTIVE_AGENT_TASK_STATUSES",
    "AGENT_TASK_TRANSITIONS",
    "AgentCommandInvocationRecord",
    "AgentTaskContractRecord",
    "AgentTaskEventRecord",
    "AgentTaskKind",
    "AgentTaskReceiptRecord",
    "AgentTaskReceiptTier",
    "AgentTaskRecord",
    "AgentTaskStatus",
    "AgentUsageCell",
    "AgentUsageCountReason",
    "AgentUsageMetric",
    "AgentUsageRecord",
    "AgentUsageSnapshot",
    "AutoResearchActorBinding",
    "AutoResearchActorBusy",
    "AutoResearchApplyResultRecord",
    "AutoResearchChildAdmissionRecord",
    "AutoResearchChildAdmissionState",
    "AutoResearchChildExperimentRecord",
    "AutoResearchChildExperimentState",
    "AutoResearchChildWorkRecord",
    "AutoResearchCommandFileKind",
    "AutoResearchCommandFileRecord",
    "AutoResearchExperimentAllowance",
    "AutoResearchExperimentAllowanceReached",
    "AutoResearchFinishBlocker",
    "AutoResearchFinishBlockerKind",
    "AutoResearchFinishDisposition",
    "AutoResearchFinishReceiptRecord",
    "AutoResearchInboxReceiptMode",
    "AutoResearchInboxReceiptRecord",
    "AutoResearchInvocationRecord",
    "AutoResearchLifecycleNoticeRecord",
    "AutoResearchLifecycleNoticeState",
    "AutoResearchMessageRecord",
    "AutoResearchMessageRole",
    "AutoResearchRecoveryMode",
    "AutoResearchRecoveryRecord",
    "AutoResearchRecoveryStatus",
    "AutoResearchRole",
    "AutoResearchStateRecord",
    "ChatSessionContextRecord",
    "ExperimentEpisodeRecord",
    "ExperimentEpisodeProjectionSnapshot",
    "ExperimentControlProjectionSnapshot",
    "ExperimentLoopRuntime",
    "EpisodeBudgetMeter",
    "EpisodeEnding",
    "EpisodeInvocationCeilingReached",
    "EpisodeInvocationRecord",
    "EpisodeMode",
    "EpisodeNotRunning",
    "EpisodeRecord",
    "EpisodeReportAttemptLimitReached",
    "EpisodeReportAttemptRecord",
    "EpisodeReportAttemptStatus",
    "EpisodeReportConflict",
    "EpisodeReportRecord",
    "EpisodeStatus",
    "EpisodeWrapupState",
    "EpisodeWrapupRecord",
    "ExperimentWatcherResourceRecord",
    "GraphCondition",
    "GraphWatcherRecord",
    "MemberRemovalPreviewRecord",
    "TeamMemberAuthorityRecord",
    "NodeStatusGraphCondition",
    "ProjectInvitationRecord",
    "ProjectMemberRecord",
    "ProjectProvisioningCancellationDisposition",
    "ProjectProvisioningCheckoutDisposition",
    "ProjectProvisioningCheckStatus",
    "ProjectProvisioningGitCheckRecord",
    "ProjectProvisioningKind",
    "ProjectProvisioningMachineIntent",
    "ProjectProvisioningMachineRecord",
    "ProjectProvisioningProviderCheckRecord",
    "ProjectProvisioningProviderIntent",
    "ProjectProvisioningRepositoryIntent",
    "ProjectProvisioningRepositoryRecord",
    "ProjectProvisioningRequestRecord",
    "ProjectProvisioningStatus",
    "ProjectProvisioningStepReceiptRecord",
    "ProjectTransferPhase",
    "ProjectTransferActivationReceipt",
    "ProjectTransferProofKind",
    "ProjectTransferProofState",
    "ProjectTransferRepositoryBinding",
    "ProjectTransferCleanupAcknowledgment",
    "ProjectTransferImportRecord",
    "ProjectTransferLinkReceipt",
    "ProjectTransferRepositorySource",
    "ProjectTransferRequestRecord",
    "ProjectTransferRegisteredProject",
    "ProjectTransferRestoreReentryReceipt",
    "ProjectTransferResolvedPath",
    "ProjectTransferSide",
    "ProjectTransferSourceConfiguration",
    "ProjectTransferSourceReleaseReceipt",
    "ProjectTransferTargetAdmissionReceipt",
    "ProjectTransferUploadCompleteReceipt",
    "ProjectTransferUploadRecord",
    "ProjectTransferUploadStatus",
    "ProjectRecord",
    "ProjectStageRecord",
    "ProposalResolvedGraphCondition",
    "ProviderSkillInventoryRecord",
    "ResultViewConflict",
    "ResultViewRecord",
    "SPACE_NAME_MAX_LENGTH",
    "SpaceKind",
    "SpaceUserKind",
    "SpaceUserRecord",
    "StoredWatcherRecord",
    "TeamAuthenticationError",
    "TeamInvitationRecord",
    "WatcherClaimConflict",
    "WatcherContinuation",
    "WatcherDeliveryRecord",
    "WatcherRecord",
    "WatcherStatus",
    "WatcherStopRequest",
    "normalize_space_name",
    "watcher_next_check_at",
]
