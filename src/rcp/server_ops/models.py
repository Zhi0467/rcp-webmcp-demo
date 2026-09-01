"""Bounded, secret-safe records shared by server CLI renderers."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SERVER_CLI_PROTOCOL_VERSION = 1
# One project plan may contain start + three steps for each of 64 repositories
# + one check for each of 32 provider records + final review.
SERVER_CLI_MAX_STEPS = 256
SERVER_CLI_MAX_EVENTS = 1 + (SERVER_CLI_MAX_STEPS * 4)
SERVER_CLI_MAX_ACTIONS = 16
SERVER_CLI_MAX_FIELDS = 48
SERVER_CLI_MAX_FIELD_CHARS = 2048
SERVER_CLI_MAX_ARGV = 64
SERVER_CLI_MAX_ARG_CHARS = 4096
SERVER_CLI_MAX_EXECUTION_BYTES = 1024 * 1024

ServerCommandName = Literal[
    "server install",
    "server doctor",
    "server provider check",
    "server project provision",
    "server project transfer-import",
    "server backup configure",
    "server backup run",
    "server restore",
    "server member remove",
    "server update",
]
ServerStepState = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "operator_action_needed",
    "unavailable",
]

_PEM_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?-----END [^-\n]*PRIVATE KEY-----",
    re.DOTALL,
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_RCP_CREDENTIAL = re.compile(r"\brcp_(?:bootstrap|member)_[A-Za-z0-9_.-]+")
_GITHUB_CREDENTIAL = re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{10,}")
_AGE_IDENTITY = re.compile(r"\bAGE-SECRET-KEY-1[A-Z0-9]+")
_PROVIDER_CREDENTIAL = re.compile(
    r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{12,}|xox[baprs]-[A-Za-z0-9-]{8,})"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passphrase|secret|token|authorization|private[_ -]?key)"
    r"(\s*(?::|=)\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SENSITIVE_FIELD = re.compile(
    r"(?i)(?:^|_)(?:password|passphrase|secret|token|authorization|private_key|credential|"
    r"identity|api_key|access_key|session|cookie)(?:_|$)"
)
_SENSITIVE_ARG_FLAG = re.compile(
    r"(?i)^--?(?:password|passphrase|secret|token|authorization|private-key|api-key|"
    r"access-key|session|cookie|identity)(?:=|$)"
)
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PHASE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def redact_server_text(value: str) -> str:
    """Remove credential-shaped values before they can enter a CLI event."""

    redacted = _PEM_PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", value)
    redacted = _BEARER_TOKEN.sub("Bearer [REDACTED]", redacted)
    redacted = _RCP_CREDENTIAL.sub("[REDACTED RCP CREDENTIAL]", redacted)
    redacted = _GITHUB_CREDENTIAL.sub("[REDACTED GITHUB CREDENTIAL]", redacted)
    redacted = _AGE_IDENTITY.sub("[REDACTED AGE IDENTITY]", redacted)
    redacted = _PROVIDER_CREDENTIAL.sub("[REDACTED PROVIDER CREDENTIAL]", redacted)
    return _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", redacted)


def _safe_text(value: str) -> str:
    value = _single_line_text(value)
    return redact_server_text(value.strip())


def _single_line_text(value: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("server CLI text cannot contain control characters")
    return value


def _safe_argv_token(value: str) -> str:
    _single_line_text(value)
    if len(value) > SERVER_CLI_MAX_ARG_CHARS:
        raise ValueError(f"argv tokens cannot exceed {SERVER_CLI_MAX_ARG_CHARS} characters")
    if redact_server_text(value) != value:
        raise ValueError("argv tokens cannot contain credential-shaped values")
    return value


ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
    AfterValidator(_safe_text),
]
PurposeText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=600),
    AfterValidator(_safe_text),
]
MessageText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
    AfterValidator(_safe_text),
]
SuccessText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
    AfterValidator(_safe_text),
]
ArgvToken = Annotated[
    str,
    StringConstraints(min_length=1, max_length=SERVER_CLI_MAX_ARG_CHARS),
    AfterValidator(_safe_argv_token),
]


def canonical_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase, hyphenated canonical UUID4")
    return value


def absolute_path(value: str, *, label: str) -> str:
    try:
        _single_line_text(value)
    except ValueError as exc:
        raise ValueError(f"{label} cannot contain control characters") from exc
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return str(path)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")


class ServerCommandRequest(_StrictModel):
    """Validated operation identity, independent of its output renderer."""

    command: ServerCommandName
    team_name: ShortText | None = None
    request_id: str | None = None
    project_id: str | None = None
    member_id: str | None = None
    member_confirmed_boundary: str | None = None
    archive_path: str | None = None
    recovery_identity_file: str | None = None
    restore_confirmed_data_dir: str | None = None
    restore_old_authority_disposition: (
        Literal["old-machine-destroyed", "old-machine-fenced-and-credentials-revoked"] | None
    ) = None
    restore_confirmed_old_authority: str | None = None
    restore_confirmed_member_roster: str | None = None
    restore_stale_member_id: str | None = None
    backup_destination: str | None = None
    backup_schedule: str | None = None
    backup_retention: int | None = None
    backup_age_recipient: str | None = None
    backup_confirmed: bool | None = None
    update_confirmed_commit: str | None = None

    @field_validator("request_id", "project_id", "member_id", "restore_stale_member_id")
    @classmethod
    def validate_identifier(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("archive_path", "recovery_identity_file", "restore_confirmed_data_dir")
    @classmethod
    def validate_path(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return absolute_path(value, label=info.field_name.replace("_", " "))

    @field_validator("update_confirmed_commit")
    @classmethod
    def validate_update_commit(cls, value: str | None) -> str | None:
        if value is not None and _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("update confirmed commit must be a full lowercase Git object id")
        return value

    @field_validator(
        "member_confirmed_boundary",
        "restore_confirmed_old_authority",
        "restore_confirmed_member_roster",
    )
    @classmethod
    def validate_member_boundary(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("member confirmed boundary must be a lowercase SHA-256 digest")
        return value

    @field_validator("backup_destination")
    @classmethod
    def validate_backup_destination(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from rcp.server_ops.config import validate_backup_destination

        return validate_backup_destination(value)

    @field_validator("backup_schedule")
    @classmethod
    def validate_backup_schedule(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from rcp.server_ops.config import validate_backup_schedule

        return validate_backup_schedule(value)

    @field_validator("backup_retention")
    @classmethod
    def validate_backup_retention(cls, value: int | None) -> int | None:
        if value is None:
            return None
        from rcp.server_ops.config import validate_backup_retention

        return validate_backup_retention(value)

    @field_validator("backup_age_recipient")
    @classmethod
    def validate_backup_age_recipient(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from rcp.server_ops.config import validate_age_recipient

        return validate_age_recipient(value)

    @model_validator(mode="after")
    def fields_match_command(self) -> ServerCommandRequest:
        supplied = {
            "team_name": self.team_name,
            "request_id": self.request_id,
            "project_id": self.project_id,
            "member_id": self.member_id,
            "member_confirmed_boundary": self.member_confirmed_boundary,
            "archive_path": self.archive_path,
            "recovery_identity_file": self.recovery_identity_file,
            "restore_confirmed_data_dir": self.restore_confirmed_data_dir,
            "restore_old_authority_disposition": self.restore_old_authority_disposition,
            "restore_confirmed_old_authority": self.restore_confirmed_old_authority,
            "restore_confirmed_member_roster": self.restore_confirmed_member_roster,
            "restore_stale_member_id": self.restore_stale_member_id,
            "backup_destination": self.backup_destination,
            "backup_schedule": self.backup_schedule,
            "backup_retention": self.backup_retention,
            "backup_age_recipient": self.backup_age_recipient,
            "backup_confirmed": self.backup_confirmed,
            "update_confirmed_commit": self.update_confirmed_commit,
        }
        expected: set[str]
        if self.command == "server install":
            expected = {"team_name"}
        elif self.command == "server provider check":
            if (self.request_id is None) == (self.project_id is None):
                raise ValueError("provider check requires exactly one request or project selector")
            expected = {"request_id" if self.request_id is not None else "project_id"}
        elif self.command in {
            "server project provision",
            "server project transfer-import",
        }:
            expected = {"request_id"}
        elif self.command == "server member remove":
            expected = {"member_id"}
            if self.member_confirmed_boundary is not None:
                expected.add("member_confirmed_boundary")
        elif self.command == "server restore":
            expected = {"archive_path", "recovery_identity_file"}
            if self.restore_confirmed_data_dir is not None:
                expected.add("restore_confirmed_data_dir")
            optional_restore_fields = {
                "restore_old_authority_disposition": self.restore_old_authority_disposition,
                "restore_confirmed_old_authority": self.restore_confirmed_old_authority,
                "restore_confirmed_member_roster": self.restore_confirmed_member_roster,
                "restore_stale_member_id": self.restore_stale_member_id,
            }
            expected.update(
                name for name, value in optional_restore_fields.items() if value is not None
            )
            if self.restore_confirmed_data_dir is None and any(
                value is not None for value in optional_restore_fields.values()
            ):
                raise ValueError(
                    "restore authority review requires the exact confirmed data directory"
                )
            if (self.restore_old_authority_disposition is None) != (
                self.restore_confirmed_old_authority is None
            ):
                raise ValueError(
                    "restore old-authority disposition and confirmation must be supplied together"
                )
            if (
                self.restore_stale_member_id is not None
                and self.restore_confirmed_member_roster is not None
            ):
                raise ValueError(
                    "restore cannot remove a stale member and confirm the pre-removal roster"
                )
        elif self.command == "server backup configure":
            expected = {
                "backup_destination",
                "backup_schedule",
                "backup_retention",
                "backup_age_recipient",
                "backup_confirmed",
            }
            if self.backup_confirmed is not True:
                raise ValueError("backup configure requires explicit confirmation")
        elif self.command == "server update":
            expected = (
                {"update_confirmed_commit"} if self.update_confirmed_commit is not None else set()
            )
        else:
            expected = set()
        missing = sorted(name for name in expected if supplied[name] is None)
        unexpected = sorted(
            name for name, value in supplied.items() if value is not None and name not in expected
        )
        if missing:
            raise ValueError(f"{self.command} is missing {', '.join(missing)}")
        if unexpected:
            raise ValueError(f"{self.command} does not accept {', '.join(unexpected)}")
        return self


class MachineTarget(_StrictModel):
    kind: Literal["machine"] = "machine"
    host: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
    os_account: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
    ]

    @field_validator("host", "os_account")
    @classmethod
    def validate_terminal_text(cls, value: str) -> str:
        _single_line_text(value)
        if redact_server_text(value) != value:
            raise ValueError("machine targets cannot contain credential-shaped values")
        return value


class ExternalServiceTarget(_StrictModel):
    kind: Literal["external_service"] = "external_service"
    service: ShortText
    resource: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
        AfterValidator(_safe_text),
    ]
    destination_url: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)
    ]
    required_authority_role: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
        AfterValidator(_safe_text),
    ]

    @field_validator("destination_url")
    @classmethod
    def validate_destination_url(cls, value: str) -> str:
        _single_line_text(value)
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("external-service destinations must be credential-free HTTPS URLs")
        if redact_server_text(value) != value:
            raise ValueError(
                "external-service destinations cannot contain credential-shaped values"
            )
        return value


ServerStepTarget: TypeAlias = Annotated[
    MachineTarget | ExternalServiceTarget,
    Field(discriminator="kind"),
]


class CommandAction(_StrictModel):
    kind: Literal["command"] = "command"
    argv: tuple[ArgvToken, ...] = Field(min_length=1, max_length=SERVER_CLI_MAX_ARGV)

    @field_validator("argv")
    @classmethod
    def reject_secret_bearing_flags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_SENSITIVE_ARG_FLAG.match(token) for token in value):
            raise ValueError("operator argv cannot accept raw credential flags")
        return value


class ExternalAction(_StrictModel):
    kind: Literal["external"] = "external"
    instruction: MessageText


OperatorAction: TypeAlias = Annotated[
    CommandAction | ExternalAction,
    Field(discriminator="kind"),
]


class NonsecretField(_StrictModel):
    name: str
    value: str | int | bool

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _FIELD_NAME.fullmatch(value):
            raise ValueError("nonsecret field names must be lowercase snake_case")
        if _SENSITIVE_FIELD.search(value):
            raise ValueError("credential-shaped fields cannot enter server CLI events")
        return value

    @field_validator("value")
    @classmethod
    def redact_value(cls, value: str | int | bool) -> str | int | bool:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("nonsecret string fields cannot be empty")
            if len(value) > SERVER_CLI_MAX_FIELD_CHARS:
                raise ValueError(
                    f"nonsecret string fields cannot exceed {SERVER_CLI_MAX_FIELD_CHARS} characters"
                )
            return _safe_text(value)
        return value


class ServerStep(_StrictModel):
    number: int = Field(ge=1, le=SERVER_CLI_MAX_STEPS)
    title: ShortText
    purpose: PurposeText
    performed_by: Literal["system", "human"]
    target: ServerStepTarget
    phase: str
    state: ServerStepState
    expected_success: SuccessText
    message: MessageText
    actions: tuple[OperatorAction, ...] = Field(default=(), max_length=SERVER_CLI_MAX_ACTIONS)
    fields: tuple[NonsecretField, ...] = Field(default=(), max_length=SERVER_CLI_MAX_FIELDS)
    resume_argv: tuple[ArgvToken, ...] = Field(default=(), max_length=SERVER_CLI_MAX_ARGV)

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, value: str) -> str:
        if not _PHASE_NAME.fullmatch(value):
            raise ValueError("phase must be lowercase snake_case")
        return value

    @model_validator(mode="after")
    def operator_action_is_explicit(self) -> ServerStep:
        field_names = [field.name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("nonsecret field names must be unique within one step")
        has_operator_contract = bool(self.actions or self.resume_argv)
        if any(_SENSITIVE_ARG_FLAG.match(token) for token in self.resume_argv):
            raise ValueError("resume argv cannot accept raw credential flags")
        if self.state == "operator_action_needed":
            if self.performed_by != "human":
                raise ValueError("operator-action steps must be performed by a human")
            if not self.actions or not self.resume_argv:
                raise ValueError("operator-action steps require actions and resume argv")
        elif has_operator_contract:
            raise ValueError("only operator-action steps may carry actions or resume argv")
        return self


class ServerPlanEvent(_StrictModel):
    version: Literal[1] = SERVER_CLI_PROTOCOL_VERSION
    event: Literal["plan"] = "plan"
    command: ServerCommandName
    timestamp: datetime
    steps: tuple[ServerStep, ...] = Field(min_length=1, max_length=SERVER_CLI_MAX_STEPS)

    @field_validator("timestamp")
    @classmethod
    def timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("server CLI event timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def plan_is_complete_and_pending(self) -> ServerPlanEvent:
        if [step.number for step in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("plan steps must be consecutively numbered from one")
        if any(step.state != "pending" for step in self.steps):
            raise ValueError("plan steps must begin pending")
        return self


class ServerStepEvent(_StrictModel):
    version: Literal[1] = SERVER_CLI_PROTOCOL_VERSION
    event: Literal["step"] = "step"
    command: ServerCommandName
    timestamp: datetime
    step: ServerStep

    @field_validator("timestamp")
    @classmethod
    def timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("server CLI event timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def event_is_not_pending(self) -> ServerStepEvent:
        if self.step.state == "pending":
            raise ValueError("step events cannot remain pending")
        return self


ServerCommandEvent: TypeAlias = Annotated[
    ServerPlanEvent | ServerStepEvent,
    Field(discriminator="event"),
]


def server_event_stream_size(events: tuple[ServerCommandEvent, ...]) -> int:
    """Return the exact UTF-8 byte count of the NDJSON event stream."""

    return sum(len(event.model_dump_json().encode("utf-8")) + 1 for event in events)


def validate_server_event_prefix(events: tuple[ServerCommandEvent, ...]) -> None:
    """Validate an event prefix before it is emitted to a human or desktop."""

    if not events:
        raise ValueError("a server CLI event stream must begin with a complete plan")
    if len(events) > SERVER_CLI_MAX_EVENTS:
        raise ValueError(f"server CLI event streams cannot exceed {SERVER_CLI_MAX_EVENTS} events")
    _validate_event_sequence(events, require_terminal=False, exit_code=None)
    if server_event_stream_size(events) > SERVER_CLI_MAX_EXECUTION_BYTES:
        raise ValueError(f"server CLI execution exceeds {SERVER_CLI_MAX_EXECUTION_BYTES} bytes")


def _validate_event_sequence(
    events: tuple[ServerCommandEvent, ...],
    *,
    require_terminal: bool,
    exit_code: int | None,
) -> None:
    plan = events[0]
    if not isinstance(plan, ServerPlanEvent):
        raise ValueError("the first server CLI event must be the complete plan")
    planned = {step.number: step for step in plan.steps}
    latest: dict[int, ServerStepState] = {}
    last_number = 0
    terminated = False
    previous_timestamp = plan.timestamp
    for event in events[1:]:
        if not isinstance(event, ServerStepEvent):
            raise ValueError("the complete plan may appear only once at the beginning")
        if event.command != plan.command:
            raise ValueError("every event must name the planned command")
        if event.timestamp < previous_timestamp:
            raise ValueError("server CLI event timestamps cannot move backwards")
        previous_timestamp = event.timestamp
        expected = planned.get(event.step.number)
        if expected is None:
            raise ValueError("step events must refer to a numbered plan step")
        if terminated:
            raise ValueError("no step event may follow a terminal failure or pause")
        if event.step.number < last_number:
            raise ValueError("step events must follow plan order")
        if event.step.number > last_number:
            if any(latest.get(number) != "succeeded" for number in range(1, event.step.number)):
                raise ValueError("a step cannot begin before every earlier step succeeds")
            last_number = event.step.number
        for field in (
            "title",
            "purpose",
            "target",
            "phase",
            "expected_success",
        ):
            if getattr(event.step, field) != getattr(expected, field):
                raise ValueError(f"step events cannot change planned {field}")
        if event.step.performed_by != expected.performed_by and not (
            expected.performed_by == "system"
            and event.step.performed_by == "human"
            and event.step.state == "operator_action_needed"
        ):
            raise ValueError("only an operator-action pause may transfer responsibility to a human")
        previous = latest.get(event.step.number)
        if event.step.state == "running" and previous is not None:
            raise ValueError("a step can start only once")
        if event.step.state == "succeeded" and previous != "running":
            raise ValueError("a step must start before it succeeds")
        if previous in {"succeeded", "failed", "operator_action_needed", "unavailable"}:
            raise ValueError("a completed or paused step cannot emit another event")
        latest[event.step.number] = event.step.state
        terminated = event.step.state in {"failed", "operator_action_needed", "unavailable"}
    if not require_terminal:
        return
    final = events[-1]
    if not isinstance(final, ServerStepEvent) or final.step.state == "running":
        raise ValueError("a returned execution must end in a terminal or paused step event")
    if exit_code == 0 and any(latest.get(step.number) != "succeeded" for step in plan.steps):
        raise ValueError("a successful execution must complete every planned step")
    if exit_code != 0 and all(latest.get(step.number) == "succeeded" for step in plan.steps):
        raise ValueError("a fully successful execution must return exit code zero")


class ServerCommandExecution(_StrictModel):
    events: tuple[ServerCommandEvent, ...] = Field(min_length=2, max_length=SERVER_CLI_MAX_EVENTS)
    exit_code: int = Field(ge=0, le=125)

    @model_validator(mode="after")
    def events_match_the_initial_plan(self) -> ServerCommandExecution:
        _validate_event_sequence(self.events, require_terminal=True, exit_code=self.exit_code)
        if server_event_stream_size(self.events) > SERVER_CLI_MAX_EXECUTION_BYTES:
            raise ValueError(f"server CLI execution exceeds {SERVER_CLI_MAX_EXECUTION_BYTES} bytes")
        return self


__all__ = [
    "CommandAction",
    "ExternalAction",
    "ExternalServiceTarget",
    "MachineTarget",
    "NonsecretField",
    "OperatorAction",
    "SERVER_CLI_MAX_ACTIONS",
    "SERVER_CLI_MAX_ARGV",
    "SERVER_CLI_MAX_EVENTS",
    "SERVER_CLI_MAX_EXECUTION_BYTES",
    "SERVER_CLI_MAX_FIELD_CHARS",
    "SERVER_CLI_MAX_FIELDS",
    "SERVER_CLI_MAX_STEPS",
    "SERVER_CLI_PROTOCOL_VERSION",
    "ServerCommandEvent",
    "ServerCommandExecution",
    "ServerCommandName",
    "ServerCommandRequest",
    "ServerPlanEvent",
    "ServerStep",
    "ServerStepEvent",
    "ServerStepState",
    "ServerStepTarget",
    "absolute_path",
    "canonical_uuid4",
    "redact_server_text",
    "server_event_stream_size",
    "validate_server_event_prefix",
]
