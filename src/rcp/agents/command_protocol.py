from __future__ import annotations

import importlib.resources
import json
import re
from functools import lru_cache
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from rcp.storage import GraphCondition

COMMAND_PROTOCOL_VERSION = 1
COMMAND_OK = 0
COMMAND_INVALID = 1
COMMAND_UNAVAILABLE = 2

CommandVerb = Literal[
    "validate",
    "apply",
    "status",
    "spawn",
    "pause",
    "resume",
    "stop",
    "message",
    "watch_graph",
    "episode",
    "inbox",
    "finish",
]
CommandStatus = Literal["ok", "invalid", "unavailable"]
MutatingCommandVerb = Literal[
    "apply",
    "spawn",
    "pause",
    "resume",
    "stop",
    "message",
    "watch_graph",
    "episode",
    "inbox",
    "finish",
]

MUTATING_COMMAND_VERBS: frozenset[CommandVerb] = frozenset(
    {
        "apply",
        "spawn",
        "pause",
        "resume",
        "stop",
        "message",
        "watch_graph",
        "episode",
        "inbox",
        "finish",
    }
)

_MAILBOX_ID = re.compile(r"^[a-f0-9]{32}$")
_REQUEST_ID = re.compile(r"^[a-f0-9]{32}$")
_CREDENTIAL = re.compile(r"^[a-f0-9]{64}$")


class CommandCredential(BaseModel):
    """One staged secret whose server-side binding expires with a provider turn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = COMMAND_PROTOCOL_VERSION
    mailbox_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    token: str = Field(pattern=r"^[a-f0-9]{64}$")


class ValidateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    patch: str


class ApplyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    patch_file: Literal["patch.json"]


class StatusArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    worker_id: str | None = Field(default=None, min_length=1, max_length=200)
    episode_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("worker_id", "episode_id")
    @classmethod
    def normalize_target_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("status target id must not be blank")
        return stripped

    @model_validator(mode="after")
    def target_is_unambiguous(self) -> StatusArguments:
        if self.worker_id is not None and self.episode_id is not None:
            raise ValueError("status accepts either a worker id or an episode id, not both")
        return self


_SAFE_WORKSPACE_FILENAME = r"^[A-Za-z0-9._-]+$"


class SpawnArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seat_node_id: str = Field(min_length=1, max_length=200)
    instruction_file: str = Field(
        min_length=1,
        max_length=255,
        pattern=_SAFE_WORKSPACE_FILENAME,
    )

    @field_validator("seat_node_id")
    @classmethod
    def strip_nonblank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("seat node id must not be blank")
        return stripped


class WorkerControlArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    worker_id: str = Field(min_length=1, max_length=200)


class MessageArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    recipient_task_id: str | None = Field(default=None, min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=16_000)

    @field_validator("body")
    @classmethod
    def strip_nonblank_body(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message body must not be blank")
        return stripped


class WatchGraphArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    condition: GraphCondition
    reason: str = Field(min_length=1, max_length=2_000)

    @field_validator("reason")
    @classmethod
    def strip_nonblank_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("watch reason must not be blank")
        return stripped


class ExperimentKickoffArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["kick_off_experiment"]
    node_id: str = Field(min_length=1, max_length=200)
    goal_file: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=_SAFE_WORKSPACE_FILENAME,
    )
    invocation_limit: int | None = Field(default=None, gt=0)

    @field_validator("node_id")
    @classmethod
    def strip_nonblank_node_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Experiment node id must not be blank")
        return stripped


class EpisodeControlArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["stop", "resume"]
    episode_id: str = Field(min_length=1, max_length=200)

    @field_validator("episode_id")
    @classmethod
    def strip_nonblank_episode_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("episode id must not be blank")
        return stripped


EpisodeArguments: TypeAlias = Annotated[
    ExperimentKickoffArguments | EpisodeControlArguments,
    Field(discriminator="action"),
]


class InboxHarvestArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["harvest"]


class InboxClearArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["clear"]


InboxArguments: TypeAlias = Annotated[
    InboxHarvestArguments | InboxClearArguments,
    Field(discriminator="action"),
]


class FinishArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = COMMAND_PROTOCOL_VERSION
    mailbox_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    credential: str = Field(pattern=r"^[a-f0-9]{64}$")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("idempotency_key")
    @classmethod
    def normalize_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("idempotency key must not be blank")
        return stripped


class ValidateCommandRequest(_CommandRequest):
    verb: Literal["validate"]
    arguments: ValidateArguments


class ApplyCommandRequest(_CommandRequest):
    verb: Literal["apply"]
    arguments: ApplyArguments


class StatusCommandRequest(_CommandRequest):
    verb: Literal["status"]
    arguments: StatusArguments = Field(default_factory=StatusArguments)


class SpawnCommandRequest(_CommandRequest):
    verb: Literal["spawn"]
    arguments: SpawnArguments


class PauseCommandRequest(_CommandRequest):
    verb: Literal["pause"]
    arguments: WorkerControlArguments


class ResumeCommandRequest(_CommandRequest):
    verb: Literal["resume"]
    arguments: WorkerControlArguments


class StopCommandRequest(_CommandRequest):
    verb: Literal["stop"]
    arguments: WorkerControlArguments


class MessageCommandRequest(_CommandRequest):
    verb: Literal["message"]
    arguments: MessageArguments


class WatchGraphCommandRequest(_CommandRequest):
    verb: Literal["watch_graph"]
    arguments: WatchGraphArguments


class EpisodeCommandRequest(_CommandRequest):
    verb: Literal["episode"]
    arguments: EpisodeArguments


class InboxCommandRequest(_CommandRequest):
    verb: Literal["inbox"]
    arguments: InboxArguments


class FinishCommandRequest(_CommandRequest):
    verb: Literal["finish"]
    arguments: FinishArguments = Field(default_factory=FinishArguments)


CommandRequest: TypeAlias = Annotated[
    ValidateCommandRequest
    | ApplyCommandRequest
    | StatusCommandRequest
    | SpawnCommandRequest
    | PauseCommandRequest
    | ResumeCommandRequest
    | StopCommandRequest
    | MessageCommandRequest
    | WatchGraphCommandRequest
    | EpisodeCommandRequest
    | InboxCommandRequest
    | FinishCommandRequest,
    Field(discriminator="verb"),
]
COMMAND_REQUEST_ADAPTER = TypeAdapter(CommandRequest)


class CommandResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = COMMAND_PROTOCOL_VERSION
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    status: CommandStatus
    message: str | None = Field(default=None, max_length=2_000)
    result: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def outcome_has_a_diagnostic(self) -> CommandResponse:
        if self.status != "ok" and not (self.message or "").strip():
            raise ValueError("an unsuccessful command response requires a diagnostic")
        return self

    @property
    def exit_code(self) -> int:
        return {
            "ok": COMMAND_OK,
            "invalid": COMMAND_INVALID,
            "unavailable": COMMAND_UNAVAILABLE,
        }[self.status]


def validate_command_request(value: str | bytes) -> CommandRequest:
    """Validate one exact request envelope, including verb-specific arguments."""

    return COMMAND_REQUEST_ADAPTER.validate_json(value)


def command_requires_idempotency_key(verb: CommandVerb) -> bool:
    return verb in MUTATING_COMMAND_VERBS


@lru_cache(maxsize=1)
def staged_command_client_source() -> str:
    """Load the one tested stdlib client source shipped to local or SSH stages."""

    return (
        importlib.resources.files("rcp.agents")
        .joinpath("staged_command_client.py")
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def staged_command_broker_source() -> str:
    """Load the stdlib broker that authenticates one live provider process tree."""

    return (
        importlib.resources.files("rcp.agents")
        .joinpath("staged_command_broker.py")
        .read_text(encoding="utf-8")
    )


def command_authentication_payload(document: str) -> bytes:
    """Canonical bytes covered by an Auto-research broker's per-request HMAC.

    The broker signs the request exactly as the client wrote it, so verification
    has to canonicalize that same text. Rebuilding the payload from the validated
    model instead would re-serialize whatever validation normalized — a sorted
    ``status_in``, a stripped string, a filled default — and the two sides would
    stop agreeing for reasons that have nothing to do with authenticity.
    """

    value = json.loads(document)
    if not isinstance(value, dict):
        raise ValueError("command request must be one JSON object")
    value.pop("credential", None)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def request_identity_is_well_formed(request: CommandRequest) -> bool:
    return (
        _MAILBOX_ID.fullmatch(request.mailbox_id) is not None
        and _REQUEST_ID.fullmatch(request.request_id) is not None
        and _CREDENTIAL.fullmatch(request.credential) is not None
    )
