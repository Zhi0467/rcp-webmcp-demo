"""Typed, history-only operational records for project transfer archives."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from rcp.transfer.archive import (
    TransferArchiveAttribution,
    TransferGraphHead,
    TransferGraphTarget,
)

TRANSFER_RECORD_SCHEMA_VERSION = 1

TRANSFER_RECORD_TABLES = frozenset(
    {
        "agent_usage",
        "auto_research_apply_results",
        "auto_research_child_admissions",
        "auto_research_child_experiments",
        "auto_research_child_work",
        "auto_research_child_work_attempts",
        "auto_research_command_files",
        "auto_research_episodes",
        "auto_research_experiment_invocations",
        "auto_research_finish_receipts",
        "auto_research_inbox_receipts",
        "auto_research_invocations",
        "auto_research_lifecycle_notices",
        "auto_research_messages",
        "auto_research_recoveries",
        "episode_invocations",
        "episode_report_attempts",
        "episode_reports",
        "episode_wrapups",
        "episodes",
        "experiment_episode_state",
        "graph_run_contracts",
        "graph_run_events",
        "graph_run_outputs",
        "graph_run_receipts",
        "graph_runs",
        "paper_drafts",
        "watchers",
    }
)

TRANSFER_EXCLUDED_PROJECT_TABLES = frozenset(
    {
        "chat_session_contexts",
        "graph_watcher_reconciliation",
        "project_aliases",
        "project_invitations",
        "project_members",
        "project_provisioning_requests",
        "project_provisioning_step_receipts",
        "project_transfer_activations",
        "project_transfer_import_configurations",
        "project_transfer_proofs",
        "project_transfer_imports",
        "project_transfer_requests",
        "project_transfer_restore_reentries",
        "project_transfer_uploads",
        "projects",
        "result_views",
        "writing_sessions",
    }
)

TRANSFER_EXECUTABLE_JSON_FIELDS = frozenset(
    {
        "continuation",
        "continuation_json",
        "can_retry",
        "repairable",
        "retryable",
        "dispatch_authority",
        "dispatch_authority_json",
        "execution_host",
        "execution_machine",
        "native_session_id",
        "session_id",
        "next_attempt_at",
        "next_check_at",
        "output_path",
        "stage_host",
        "stage_root",
        "run_on",
        "attachment_set_id",
        "attachment_client_id",
        "attachment_batch_id",
        "attachments",
        "accepted_handoff",
        "artifact_context",
        "check_command",
        "cwd",
        "delivery_operation_id",
        "log_path",
        "notification_operation_id",
        "result_view",
        "runtime_id",
        "watcher_ids",
        "watcher_snapshot_token",
        "write_scope_fingerprint",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_UUID4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


def _aware_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("transfer record timestamp must be ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("transfer record timestamp must include a timezone")
    return value


AwareTimestamp = Annotated[str, AfterValidator(_aware_timestamp)]


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _validate_history_json(value: JsonValue) -> None:
    if isinstance(value, dict):
        forbidden = TRANSFER_EXECUTABLE_JSON_FIELDS.intersection(value)
        if forbidden:
            raise ValueError(
                f"transfer history JSON contains executable field {sorted(forbidden)[0]}"
            )
        for child in value.values():
            _validate_history_json(child)
    elif isinstance(value, list):
        for child in value:
            _validate_history_json(child)


def sanitize_transfer_history_json(value: JsonValue) -> JsonValue:
    """Copy inert JSON while dropping every recognized source execution binding."""

    if isinstance(value, dict):
        return {
            key: sanitize_transfer_history_json(child)
            for key, child in value.items()
            if key not in TRANSFER_EXECUTABLE_JSON_FIELDS
        }
    if isinstance(value, list):
        return [sanitize_transfer_history_json(child) for child in value]
    return value


class _StrictTransferRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


class TransferJsonDocument(_StrictTransferRecord):
    """Immutable canonical JSON after recursive live-binding exclusion."""

    canonical_json: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("transfer JSON digest must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_document(self) -> TransferJsonDocument:
        try:
            parsed: JsonValue = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("transfer history JSON is invalid") from exc
        canonical = _canonical_json(parsed)
        if canonical != self.canonical_json:
            raise ValueError("transfer history JSON must be canonical")
        _validate_history_json(parsed)
        if hashlib.sha256(canonical.encode()).hexdigest() != self.sha256:
            raise ValueError("transfer history JSON does not match its digest")
        return self

    @classmethod
    def capture(cls, value: JsonValue) -> TransferJsonDocument:
        _validate_history_json(value)
        canonical = _canonical_json(value)
        return cls(canonical_json=canonical, sha256=hashlib.sha256(canonical.encode()).hexdigest())

    @classmethod
    def capture_sanitized(cls, value: JsonValue) -> TransferJsonDocument:
        return cls.capture(sanitize_transfer_history_json(value))

    def value(self) -> JsonValue:
        return json.loads(self.canonical_json)


class TransferLocalId(_StrictTransferRecord):
    """Stable archive identity for a source-local integer or composite id."""

    archive_id: str
    source_table: Literal["graph_run_events", "graph_run_receipts"]
    source_id: str = Field(min_length=1, max_length=255)

    @field_validator("archive_id")
    @classmethod
    def validate_archive_id(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except ValueError as exc:
            raise ValueError("archive-local record identity must be a canonical UUID") from exc
        if str(parsed) != value:
            raise ValueError("archive-local record identity must be a canonical UUID")
        return value


class TransferArtifactReference(_StrictTransferRecord):
    artifact_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=255)
    size_bytes: int | None = Field(default=None, ge=0)
    content_sha256: str | None = None
    expires_at: AwareTimestamp | None = None
    kept_filename: str | None = Field(default=None, min_length=1, max_length=255)
    kept_at: AwareTimestamp | None = None

    @field_validator("content_sha256")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("artifact reference digest must be lowercase SHA-256")
        return value

    @field_validator("source_name", "kept_filename")
    @classmethod
    def validate_filename(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("artifact history requires one direct filename")
        return value

    @model_validator(mode="after")
    def validate_kept_reference(self) -> TransferArtifactReference:
        if (self.kept_filename is None) != (self.kept_at is None):
            raise ValueError("kept artifact history requires both its filename and kept time")
        return self


class TransferTaskEvent(_StrictTransferRecord):
    identity: TransferLocalId
    created_at: AwareTimestamp
    level: Literal["info", "warning", "error"]
    message: str
    event_kind: Literal["message", "command"] = "message"
    command_id: str | None = None
    command_verb: str | None = None
    command_phase: Literal["start", "exit"] | None = None
    payload: TransferJsonDocument | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> TransferTaskEvent:
        if self.identity.source_table != "graph_run_events":
            raise ValueError("task event identity belongs to the wrong source table")
        if self.event_kind == "message" and any(
            value is not None for value in (self.command_id, self.command_verb, self.command_phase)
        ):
            raise ValueError("message history cannot carry command execution fields")
        return self


class TransferTaskReceipt(_StrictTransferRecord):
    identity: TransferLocalId
    created_at: AwareTimestamp
    tier: Literal["summary", "diagnostic", "trace"]
    category: str = Field(min_length=1)
    payload: TransferJsonDocument

    @model_validator(mode="after")
    def validate_identity(self) -> TransferTaskReceipt:
        if self.identity.source_table != "graph_run_receipts":
            raise ValueError("task receipt identity belongs to the wrong source table")
        return self


class TransferTaskUsage(_StrictTransferRecord):
    usage_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str | None = None
    provider_profile: str = Field(min_length=1)
    provider_event_type: str = Field(min_length=1)
    counted: bool
    count_reason: Literal["counted", "duplicate", "invalid"]
    processed_input_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_write_input_tokens: int = Field(ge=0)
    reasoning_output_tokens: int = Field(ge=0)
    reported_input_tokens: int | None = Field(default=None, ge=0)
    reported_output_tokens: int | None = Field(default=None, ge=0)
    reported_total_tokens: int | None = Field(default=None, ge=0)
    provider_fields: TransferJsonDocument
    created_at: AwareTimestamp


class TransferTaskContract(_StrictTransferRecord):
    role: str = Field(min_length=1)
    content: str
    sha256: str
    created_at: AwareTimestamp

    @model_validator(mode="after")
    def validate_content(self) -> TransferTaskContract:
        if hashlib.sha256(self.content.encode()).hexdigest() != self.sha256:
            raise ValueError("task contract content does not match its digest")
        return self


class TransferTaskOutput(_StrictTransferRecord):
    created_at: AwareTimestamp
    patch: TransferJsonDocument


class TransferRunRequestHistory(_StrictTransferRecord):
    shape: Literal["run"] = "run"
    provider: str | None = None
    model: str | None = None
    reasoning: str | None = None
    run_truth_scope: tuple[str, ...] | None = None
    chat_scope: Literal["node", "project"] = "node"
    node_id: str | None = None
    message: str | None = None
    chat_id: str | None = None
    mode: Literal["discuss", "work"] = "discuss"
    trigger: Literal["human", "orchestrator", "experiment_run", "watcher"] = "human"
    patch_kind: Literal["work", "experiment_loop"] = "work"
    control_node_id: str | None = None
    control_revision: int | None = Field(default=None, ge=0)
    control_episode_id: str | None = None
    control_invocation: int | None = Field(default=None, ge=1)
    control_invocation_ceiling: int | None = Field(default=None, ge=1)
    control_decision_bundle: tuple[TransferJsonDocument, ...] = ()
    control_completion_criteria: tuple[str, ...] = ()
    episode_id: str | None = None
    workflow_ids: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    invoked_workflow_ids: tuple[str, ...] = ()
    invoked_skill_ids: tuple[str, ...] = ()
    invoked_provider_skill_names: tuple[str, ...] = ()


class TransferPaperCoachRequestHistory(_StrictTransferRecord):
    shape: Literal["paper_coach"] = "paper_coach"
    message: str
    provider: str | None = None
    model: str | None = None
    reasoning: str | None = None
    workflow_ids: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    invoked_workflow_ids: tuple[str, ...] = ()
    invoked_skill_ids: tuple[str, ...] = ()
    invoked_provider_skill_names: tuple[str, ...] = ()


class TransferAutoResearchRequestHistory(_StrictTransferRecord):
    shape: Literal["auto_research"] = "auto_research"
    episode_id: str = Field(min_length=1)
    role: Literal["orchestrator", "worker"]
    provider: str | None = None
    model: str | None = None
    reasoning: str | None = None
    run_truth_scope: tuple[str, ...] | None = None
    actor_operation_id: str | None = None
    instruction: str | None = Field(default=None, max_length=16_000)
    control_node_id: str | None = None
    wake_cause: Literal["watcher", "graph_condition", "message", "lifecycle"] | None = None
    workflow_ids: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    invoked_workflow_ids: tuple[str, ...] = ()
    invoked_skill_ids: tuple[str, ...] = ()
    invoked_provider_skill_names: tuple[str, ...] = ()


class TransferEpisodeReportRequestHistory(_StrictTransferRecord):
    shape: Literal["episode_report"] = "episode_report"
    episode_id: str = Field(min_length=1)
    provider: str
    model: str
    reasoning: str


TransferTaskRequestHistory = Annotated[
    TransferRunRequestHistory
    | TransferPaperCoachRequestHistory
    | TransferAutoResearchRequestHistory
    | TransferEpisodeReportRequestHistory,
    Field(discriminator="shape"),
]


def _optional_string(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError("task request history expected a string")


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("task request history expected a string list")
    return tuple(value)


def _request_common(request: Mapping[str, object]) -> dict[str, object]:
    return {
        "provider": _optional_string(request.get("provider")),
        "model": _optional_string(request.get("model")),
        "reasoning": _optional_string(request.get("reasoning")),
        "workflow_ids": _string_tuple(request.get("workflow_ids")),
        "skill_ids": _string_tuple(request.get("skill_ids")),
        "invoked_workflow_ids": _string_tuple(request.get("invoked_workflow_ids")),
        "invoked_skill_ids": _string_tuple(request.get("invoked_skill_ids")),
        "invoked_provider_skill_names": _string_tuple(request.get("invoked_provider_skill_names")),
    }


def capture_task_request_history(
    kind: Literal[
        "seed",
        "refresh",
        "node_chat",
        "project_chat",
        "paper_coach",
        "auto_research",
        "branch_merge",
        "episode_report",
    ],
    request: Mapping[str, object],
) -> TransferTaskRequestHistory:
    """Project one persisted request into inert history, never a resume request."""

    common = _request_common(request)
    if kind == "paper_coach":
        return TransferPaperCoachRequestHistory(
            **common,
            message=_optional_string(request.get("message")) or "",
        )
    if kind == "auto_research":
        return TransferAutoResearchRequestHistory(
            **common,
            episode_id=_optional_string(request.get("episode_id")) or "",
            role=request.get("role"),
            run_truth_scope=_string_tuple(request.get("run_truth_scope")) or None,
            actor_operation_id=_optional_string(request.get("actor_operation_id")),
            instruction=_optional_string(request.get("instruction")),
            control_node_id=_optional_string(request.get("control_node_id")),
            wake_cause=request.get("wake_cause"),
        )
    if kind == "episode_report":
        return TransferEpisodeReportRequestHistory(
            episode_id=_optional_string(request.get("episode_id")) or "",
            provider=_optional_string(request.get("provider")) or "",
            model=_optional_string(request.get("model")) or "",
            reasoning=_optional_string(request.get("reasoning")) or "",
        )

    decision_bundle = request.get("control_decision_bundle")
    if not isinstance(decision_bundle, list):
        decision_bundle = []
    return TransferRunRequestHistory(
        **common,
        run_truth_scope=_string_tuple(request.get("run_truth_scope")) or None,
        chat_scope=request.get("chat_scope", "node"),
        node_id=_optional_string(request.get("node_id")),
        message=_optional_string(request.get("message")),
        chat_id=_optional_string(request.get("chat_id")),
        mode=request.get("mode", "discuss"),
        trigger=request.get("trigger", "human"),
        patch_kind=request.get("patch_kind", "work"),
        control_node_id=_optional_string(request.get("control_node_id")),
        control_revision=request.get("control_revision"),
        control_episode_id=_optional_string(request.get("control_episode_id")),
        control_invocation=request.get("control_invocation"),
        control_invocation_ceiling=request.get("control_invocation_ceiling"),
        control_decision_bundle=tuple(
            TransferJsonDocument.capture(item) for item in decision_bundle
        ),
        control_completion_criteria=_string_tuple(request.get("control_completion_criteria")),
        episode_id=_optional_string(request.get("episode_id")),
    )


class TransferAssistantHistory(_StrictTransferRecord):
    """Exact labelled output plus an honest bucket for pre-label source rows."""

    answer: str | None = None
    trace_messages: tuple[str, ...] = ()
    legacy_unlabelled_lines: tuple[str, ...] = ()


class TransferTaskRecord(_StrictTransferRecord):
    operation_id: str = Field(min_length=1)
    kind: Literal[
        "seed",
        "refresh",
        "node_chat",
        "project_chat",
        "paper_coach",
        "auto_research",
        "branch_merge",
        "episode_report",
    ]
    status: Literal["succeeded", "failed", "interrupted"]
    request: TransferTaskRequestHistory
    assistant: TransferAssistantHistory = Field(default_factory=TransferAssistantHistory)
    error: str | None = None
    applied_revision: int | None = Field(default=None, ge=0)
    graph_updates: tuple[TransferJsonDocument, ...] = ()
    attempt: int = Field(ge=1)
    parent_operation_id: str | None = None
    episode_id: str | None = None
    graph_target: TransferGraphTarget = Field(default_factory=TransferGraphTarget)
    authorized_by_attribution_id: str | None = None
    created_at: AwareTimestamp
    updated_at: AwareTimestamp
    started_at: AwareTimestamp | None = None
    finished_at: AwareTimestamp
    status_message: str
    events: tuple[TransferTaskEvent, ...] = ()
    receipts: tuple[TransferTaskReceipt, ...] = ()
    usage: tuple[TransferTaskUsage, ...] = ()
    contracts: tuple[TransferTaskContract, ...] = ()
    output: TransferTaskOutput | None = None
    artifacts: tuple[TransferArtifactReference, ...] = ()
    visible: bool = True
    history_only: Literal[True] = True

    @field_validator("authorized_by_attribution_id")
    @classmethod
    def validate_attribution_id(cls, value: str | None) -> str | None:
        if value is not None and _UUID4.fullmatch(value) is None:
            raise ValueError("task attribution must name one archive actor")
        return value

    @model_validator(mode="after")
    def validate_task(self) -> TransferTaskRecord:
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("successful task history cannot carry an error")
        event_ids = [item.identity.archive_id for item in self.events]
        receipt_ids = [item.identity.archive_id for item in self.receipts]
        if len(event_ids) != len(set(event_ids)) or len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError("task history cannot repeat local record mappings")
        expected_shape = {
            "paper_coach": "paper_coach",
            "auto_research": "auto_research",
            "episode_report": "episode_report",
        }.get(self.kind, "run")
        if self.request.shape != expected_shape:
            raise ValueError("task history request shape does not match its task kind")
        return self


class TransferWatcherRecord(_StrictTransferRecord):
    watcher_id: str = Field(min_length=1)
    kind: Literal["external", "graph"]
    origin_operation_id: str = Field(min_length=1)
    origin_task_kind: Literal["node_chat", "project_chat", "auto_research"]
    chat_id: str = Field(min_length=1)
    node_id: str | None = None
    episode_id: str | None = None
    graph_target: TransferGraphTarget = Field(default_factory=TransferGraphTarget)
    status: Literal["completed", "stopped"]
    graph_condition: TransferJsonDocument | None = None
    last_checked_at: AwareTimestamp | None = None
    last_exit_code: int | None = None
    last_error: str | None = None
    consecutive_error_count: int = Field(ge=0)
    group_id: str | None = None
    group_label: str | None = None
    stopped_by: Literal["human", "loop", "agent"] | None = None
    stop_reason: str | None = None
    created_at: AwareTimestamp
    completed_at: AwareTimestamp | None = None
    stopped_at: AwareTimestamp | None = None
    stop_operation_id: str | None = None

    @model_validator(mode="after")
    def validate_terminal_watcher(self) -> TransferWatcherRecord:
        if self.status == "completed" and self.completed_at is None:
            raise ValueError("completed watcher history requires its completion time")
        if self.status == "stopped" and self.stopped_at is None:
            raise ValueError("stopped watcher history requires its stop time")
        if self.kind == "graph" and self.graph_condition is None:
            raise ValueError("graph watcher history requires its condition")
        if self.kind == "external" and self.graph_condition is not None:
            raise ValueError("external watcher history cannot carry a graph condition")
        return self


class TransferEpisodeInvocation(_StrictTransferRecord):
    operation_id: str = Field(min_length=1)
    invocation_number: int = Field(ge=1)
    created_at: AwareTimestamp


class TransferEpisodeReportAttempt(_StrictTransferRecord):
    attempt_id: str = Field(min_length=1)
    attempt_number: int = Field(ge=1, le=3)
    allocation_operation_id: str = Field(min_length=1)
    status: Literal["succeeded", "failed"]
    error: str | None = None
    created_at: AwareTimestamp
    updated_at: AwareTimestamp
    finished_at: AwareTimestamp


class TransferEpisodeReport(_StrictTransferRecord):
    report_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    allocation_operation_id: str = Field(min_length=1)
    ending: Literal["completed", "exhausted", "stopped", "failed", "human_pause"]
    sha256: str
    html: str
    created_at: AwareTimestamp

    @model_validator(mode="after")
    def validate_report(self) -> TransferEpisodeReport:
        if hashlib.sha256(self.html.encode()).hexdigest() != self.sha256:
            raise ValueError("episode report content does not match its digest")
        return self


class TransferEpisodeWrapup(_StrictTransferRecord):
    ending: Literal["completed", "exhausted", "stopped", "failed", "human_pause"] | None
    partial: bool
    concluding_operation_id: str | None = None
    allocation_operation_id: str | None = None
    provider: str | None = None
    skill_id: str | None = None
    skill_version: str | None = None
    receipt: TransferJsonDocument
    state: Literal["ready", "failed", "skipped", "legacy_unavailable"]
    diagnostic: str | None = None
    created_at: AwareTimestamp
    updated_at: AwareTimestamp
    finished_at: AwareTimestamp | None = None

    @model_validator(mode="after")
    def validate_terminal_wrapup(self) -> TransferEpisodeWrapup:
        if self.state == "skipped" and self.ending != "stopped":
            raise ValueError("only a stopped episode can carry a skipped wrap-up")
        if self.ending == "stopped" and self.state != "skipped":
            raise ValueError("a stopped episode must carry a skipped wrap-up")
        if self.state in {"ready", "failed"} and self.ending is None:
            raise ValueError("terminal wrap-up history requires its episode ending")
        return self


class TransferExperimentEpisodeHistory(_StrictTransferRecord):
    provider: str | None = None
    chat_id: str | None = None
    last_turn_operation_id: str | None = None
    last_turn_invocation: int | None = Field(default=None, ge=1)
    last_graph_result: str | None = None
    last_watcher_ids: tuple[str, ...] = ()
    session_diagnostic: str | None = None


class TransferAutoResearchInvocation(_StrictTransferRecord):
    operation_id: str = Field(min_length=1)
    allocation_operation_id: str = Field(min_length=1)
    role: Literal["orchestrator", "worker"]
    actor_operation_id: str = Field(min_length=1)
    control_node_id: str | None = None
    created_at: AwareTimestamp


class TransferAutoResearchMessage(_StrictTransferRecord):
    message_id: str = Field(min_length=1)
    sender_role: Literal["human", "orchestrator", "worker"]
    sender_task_id: str | None = None
    authorized_by_attribution_id: str | None = None
    recipient_task_id: str = Field(min_length=1)
    control_node_id: str | None = None
    body: str = Field(min_length=1, max_length=16_000)
    disposition: Literal["delivered", "cancelled"]
    created_at: AwareTimestamp
    delivered_at: AwareTimestamp | None = None

    @field_validator("authorized_by_attribution_id")
    @classmethod
    def validate_attribution_id(cls, value: str | None) -> str | None:
        if value is not None and _UUID4.fullmatch(value) is None:
            raise ValueError("Auto-research message attribution must name one archive actor")
        return value

    @model_validator(mode="after")
    def validate_message(self) -> TransferAutoResearchMessage:
        if self.sender_role == "human" and self.authorized_by_attribution_id is None:
            raise ValueError("human Auto-research history requires attribution")
        if self.sender_role != "human" and self.authorized_by_attribution_id is not None:
            raise ValueError("agent Auto-research history cannot claim human attribution")
        if self.disposition == "delivered" and self.delivered_at is None:
            raise ValueError("delivered Auto-research history requires its delivery time")
        return self


class TransferAutoResearchRecovery(_StrictTransferRecord):
    recovery_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    failure_kind: str = Field(min_length=1)
    retry_mode: Literal["exact", "clean", "blocked"]
    attempts: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    status: Literal["admitted", "exhausted", "blocked"]
    diagnostic: str
    admitted_operation_id: str | None = None
    created_at: AwareTimestamp
    updated_at: AwareTimestamp


class TransferAutoResearchChildWorkAttempt(_StrictTransferRecord):
    operation_id: str = Field(min_length=1)
    allocation_operation_id: str = Field(min_length=1)
    created_at: AwareTimestamp


class TransferAutoResearchChildWork(_StrictTransferRecord):
    worker_id: str = Field(min_length=1)
    control_node_id: str = Field(min_length=1)
    root_operation_id: str = Field(min_length=1)
    final_operation_id: str = Field(min_length=1)
    admitted_by_operation_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=16_000)
    instruction_sha256: str
    attempts: tuple[TransferAutoResearchChildWorkAttempt, ...]
    created_at: AwareTimestamp
    updated_at: AwareTimestamp

    @model_validator(mode="after")
    def validate_instruction(self) -> TransferAutoResearchChildWork:
        if hashlib.sha256(self.instruction.encode()).hexdigest() != self.instruction_sha256:
            raise ValueError("child Work instruction does not match its digest")
        return self


class TransferAutoResearchChildExperimentRequest(_StrictTransferRecord):
    goal: str | None = None
    invocation_limit: int | None = Field(default=None, ge=1)


class TransferAutoResearchChildExperiment(_StrictTransferRecord):
    child_episode_id: str = Field(min_length=1)
    control_node_id: str = Field(min_length=1)
    state: Literal["cancelled", "terminal"]
    replaces_episode_id: str | None = None
    request: TransferAutoResearchChildExperimentRequest
    goal_sha256: str | None = None
    parent_operation_id: str = Field(min_length=1)
    terminal_diagnostic: str | None = None
    invocations: tuple[TransferAutoResearchExperimentInvocation, ...] = ()
    created_at: AwareTimestamp
    updated_at: AwareTimestamp

    @field_validator("goal_sha256")
    @classmethod
    def validate_goal_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("child Experiment goal digest must be lowercase SHA-256")
        return value


class TransferAutoResearchExperimentInvocation(_StrictTransferRecord):
    operation_id: str = Field(min_length=1)
    created_at: AwareTimestamp


class TransferAutoResearchChildAdmission(_StrictTransferRecord):
    admission_id: str = Field(min_length=1)
    child_kind: Literal["work", "experiment"]
    child_id: str = Field(min_length=1)
    state: Literal["reflected", "cancelled"]
    created_at: AwareTimestamp
    updated_at: AwareTimestamp


class TransferAutoResearchLifecycleNotice(_StrictTransferRecord):
    notice_id: str = Field(min_length=1)
    source_kind: Literal[
        "worker",
        "experiment_task",
        "experiment_episode",
        "experiment_replacement",
    ]
    source_id: str = Field(min_length=1)
    source_event: str = Field(min_length=1)
    source_attempt: int = Field(ge=1)
    payload: TransferJsonDocument
    created_at: AwareTimestamp
    delivered_at: AwareTimestamp
    acknowledged_at: AwareTimestamp
    acknowledged_by: str = Field(min_length=1)


class TransferAutoResearchInboxReceipt(_StrictTransferRecord):
    effect_id: str = Field(min_length=1)
    mode: Literal["harvest", "clear"]
    result: TransferJsonDocument
    acknowledged_by: str = Field(min_length=1)
    created_at: AwareTimestamp


class TransferAutoResearchFinishReceipt(_StrictTransferRecord):
    effect_id: str = Field(min_length=1)
    actor_operation_id: str = Field(min_length=1)
    disposition: Literal["blocked", "completed"]
    blocker_count: int = Field(ge=0)
    result: TransferJsonDocument
    result_sha256: str
    created_at: AwareTimestamp

    @model_validator(mode="after")
    def validate_result(self) -> TransferAutoResearchFinishReceipt:
        if self.result.sha256 != self.result_sha256:
            raise ValueError("Auto-research Finish result does not match its stored digest")
        return self


class TransferAutoResearchApplyResult(_StrictTransferRecord):
    apply_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    patch_sha256: str
    result: TransferJsonDocument
    created_at: AwareTimestamp

    @field_validator("patch_sha256")
    @classmethod
    def validate_patch_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("Auto-research Apply digest must be lowercase SHA-256")
        return value


class TransferAutoResearchCommand(_StrictTransferRecord):
    command_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    kind: Literal["apply", "instruction", "goal"]
    filename: str = Field(min_length=1, max_length=255)
    sha256: str
    content: str
    created_at: AwareTimestamp

    @model_validator(mode="after")
    def validate_content(self) -> TransferAutoResearchCommand:
        if self.filename in {".", ".."} or "/" in self.filename or "\\" in self.filename:
            raise ValueError("Auto-research command history requires one direct filename")
        if hashlib.sha256(self.content.encode()).hexdigest() != self.sha256:
            raise ValueError("Auto-research command content does not match its digest")
        return self


class TransferAutoResearchHistory(_StrictTransferRecord):
    starting_instruction: str | None = Field(default=None, max_length=16_000)
    created_at: AwareTimestamp
    updated_at: AwareTimestamp
    invocations: tuple[TransferAutoResearchInvocation, ...] = ()
    messages: tuple[TransferAutoResearchMessage, ...] = ()
    recoveries: tuple[TransferAutoResearchRecovery, ...] = ()
    child_work: tuple[TransferAutoResearchChildWork, ...] = ()
    child_experiments: tuple[TransferAutoResearchChildExperiment, ...] = ()
    child_admissions: tuple[TransferAutoResearchChildAdmission, ...] = ()
    lifecycle_notices: tuple[TransferAutoResearchLifecycleNotice, ...] = ()
    inbox_receipts: tuple[TransferAutoResearchInboxReceipt, ...] = ()
    finish_receipts: tuple[TransferAutoResearchFinishReceipt, ...] = ()
    apply_results: tuple[TransferAutoResearchApplyResult, ...] = ()
    commands: tuple[TransferAutoResearchCommand, ...] = ()


class TransferEpisodeRecord(_StrictTransferRecord):
    episode_id: str = Field(min_length=1)
    mode: Literal["auto_research", "experiment_loop"]
    control_node_id: str | None = None
    graph_target: TransferGraphTarget = Field(default_factory=TransferGraphTarget)
    graph_base_head: TransferGraphHead | None = None
    root_operation_id: str | None = None
    status: Literal["completed", "stopped", "failed"]
    invocation_ceiling: int = Field(ge=1)
    invocations_used: int = Field(ge=0)
    authorized_by_attribution_id: str | None = None
    ending: Literal["completed", "exhausted", "stopped", "failed", "human_pause"]
    ending_diagnostic: str | None = None
    wrapup_state: Literal[
        "not_started",
        "ready",
        "failed",
        "skipped",
        "legacy_unavailable",
    ]
    wrapup_error: str | None = None
    report_attempts_used: int = Field(ge=0, le=3)
    created_at: AwareTimestamp
    updated_at: AwareTimestamp
    ended_at: AwareTimestamp
    invocations: tuple[TransferEpisodeInvocation, ...] = ()
    report_attempts: tuple[TransferEpisodeReportAttempt, ...] = ()
    wrapup: TransferEpisodeWrapup | None = None
    report: TransferEpisodeReport | None = None
    experiment: TransferExperimentEpisodeHistory | None = None
    auto_research: TransferAutoResearchHistory | None = None

    @model_validator(mode="after")
    def validate_episode(self) -> TransferEpisodeRecord:
        if self.invocations_used > self.invocation_ceiling:
            raise ValueError("episode history exceeds its invocation ceiling")
        if self.mode == "experiment_loop":
            if (
                self.control_node_id is None
                or self.experiment is None
                or self.auto_research is not None
            ):
                raise ValueError("Experiment history requires only its sanitized mode record")
        elif (
            self.control_node_id is not None
            or self.auto_research is None
            or self.experiment is not None
        ):
            raise ValueError("Auto-research history requires only its sanitized mode record")
        if self.mode == "auto_research" and (
            self.graph_target.kind != "branch" or self.graph_target.branch_id != self.episode_id
        ):
            raise ValueError("Auto-research history must retain its same-id graph branch")
        if self.graph_target.kind == "main" and self.graph_base_head is not None:
            raise ValueError("main-target episode history cannot carry a branch base head")
        if self.graph_target.kind == "branch" and (
            self.graph_base_head is None or self.graph_base_head.target.kind != "main"
        ):
            raise ValueError("branch episode history requires its immutable main base head")
        if len(self.invocations) != self.invocations_used:
            raise ValueError("episode invocation history does not match its used count")
        if len(self.report_attempts) != self.report_attempts_used:
            raise ValueError("episode report attempts do not match their used count")
        if self.wrapup is None:
            if self.wrapup_state != "not_started":
                raise ValueError("terminal wrap-up state requires its retained wrap-up record")
            if self.report_attempts or self.report is not None:
                raise ValueError("report history requires its retained episode wrap-up")
            return self
        if self.wrapup.state != self.wrapup_state:
            raise ValueError("episode and retained wrap-up states disagree")
        if self.wrapup.ending != self.ending:
            raise ValueError("episode and retained wrap-up endings disagree")
        if self.report is None:
            if self.wrapup_state == "ready":
                raise ValueError("a ready episode wrap-up requires its report")
            if any(attempt.status == "succeeded" for attempt in self.report_attempts):
                raise ValueError("a succeeded report attempt requires its report")
            return self
        attempts = {attempt.attempt_id: attempt for attempt in self.report_attempts}
        attempt = attempts.get(self.report.attempt_id)
        if self.wrapup_state != "ready" or attempt is None or attempt.status != "succeeded":
            raise ValueError("an episode report requires its succeeded ready attempt")
        if (
            self.report.ending != self.ending
            or self.report.allocation_operation_id != attempt.allocation_operation_id
            or self.report.allocation_operation_id != self.wrapup.allocation_operation_id
        ):
            raise ValueError("episode report lineage disagrees with its wrap-up")
        return self


class TransferPaperDraft(_StrictTransferRecord):
    content: str
    base_hash: str | None = None
    ancestor_content: str | None = None
    cursor_state: str | None = None
    updated_at: AwareTimestamp


class TransferRecordBundle(_StrictTransferRecord):
    schema_version: Literal[TRANSFER_RECORD_SCHEMA_VERSION] = TRANSFER_RECORD_SCHEMA_VERSION
    project_id: str = Field(min_length=1)
    attributions: tuple[TransferArchiveAttribution, ...]
    tasks: tuple[TransferTaskRecord, ...]
    watchers: tuple[TransferWatcherRecord, ...]
    episodes: tuple[TransferEpisodeRecord, ...]
    paper_draft: TransferPaperDraft | None = None

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        if _UUID4.fullmatch(value) is None:
            raise ValueError("transfer record bundle project identity must be a canonical UUID4")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> TransferRecordBundle:
        attribution_ids = {item.archive_actor_id for item in self.attributions}
        references = {
            item.authorized_by_attribution_id
            for item in (*self.tasks, *self.episodes)
            if item.authorized_by_attribution_id is not None
        }
        references.update(
            message.authorized_by_attribution_id
            for episode in self.episodes
            if episode.auto_research is not None
            for message in episode.auto_research.messages
            if message.authorized_by_attribution_id is not None
        )
        if not references <= attribution_ids:
            raise ValueError("transfer records reference an unknown archive attribution")
        task_ids = {item.operation_id for item in self.tasks}
        watcher_ids = {item.watcher_id for item in self.watchers}
        episode_ids = {item.episode_id for item in self.episodes}
        for records, identities, label in (
            (self.tasks, task_ids, "task"),
            (self.watchers, watcher_ids, "watcher"),
            (self.episodes, episode_ids, "episode"),
        ):
            if len(records) != len(identities):
                raise ValueError(f"transfer bundle repeats one {label} identity")

        local_ids = [
            child.identity.archive_id
            for task in self.tasks
            for child in (*task.events, *task.receipts)
        ]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("transfer bundle repeats one archive-local record identity")

        def require(reference: str | None, identities: set[str], label: str) -> None:
            if reference is not None and reference not in identities:
                raise ValueError(f"transfer records reference an unknown {label}")

        for task in self.tasks:
            require(task.parent_operation_id, task_ids, "parent task")
            require(task.episode_id, episode_ids, "episode")
        for watcher in self.watchers:
            require(watcher.origin_operation_id, task_ids, "watcher origin task")
            require(watcher.episode_id, episode_ids, "watcher episode")
            require(watcher.stop_operation_id, task_ids, "watcher stop task")
        for episode in self.episodes:
            require(episode.root_operation_id, task_ids, "episode root task")
            for invocation in episode.invocations:
                require(invocation.operation_id, task_ids, "episode invocation task")
            for attempt in episode.report_attempts:
                require(attempt.allocation_operation_id, task_ids, "report allocation task")
            if episode.wrapup is not None:
                require(
                    episode.wrapup.concluding_operation_id,
                    task_ids,
                    "wrap-up concluding task",
                )
                require(
                    episode.wrapup.allocation_operation_id,
                    task_ids,
                    "wrap-up allocation task",
                )
            if episode.report is not None:
                require(
                    episode.report.allocation_operation_id,
                    task_ids,
                    "report allocation task",
                )
                if episode.report.attempt_id not in {
                    item.attempt_id for item in episode.report_attempts
                }:
                    raise ValueError("episode report references an unknown report attempt")
            if episode.experiment is not None:
                require(
                    episode.experiment.last_turn_operation_id,
                    task_ids,
                    "Experiment last-turn task",
                )
                unknown_watchers = set(episode.experiment.last_watcher_ids) - watcher_ids
                if unknown_watchers:
                    raise ValueError("Experiment history references an unknown watcher")
            if episode.auto_research is not None:
                self._validate_auto_research_references(
                    episode.auto_research,
                    task_ids=task_ids,
                    episode_ids=episode_ids,
                    require=require,
                )
        return self

    @staticmethod
    def _validate_auto_research_references(
        history: TransferAutoResearchHistory,
        *,
        task_ids: set[str],
        episode_ids: set[str],
        require: Callable[[str | None, set[str], str], None],
    ) -> None:
        child_work_ids = {item.worker_id for item in history.child_work}
        child_experiment_ids = {item.child_episode_id for item in history.child_experiments}
        if len(child_work_ids) != len(history.child_work):
            raise ValueError("Auto-research history repeats one child Work identity")
        if len(child_experiment_ids) != len(history.child_experiments):
            raise ValueError("Auto-research history repeats one child Experiment identity")

        for invocation in history.invocations:
            for reference in (
                invocation.operation_id,
                invocation.allocation_operation_id,
                invocation.actor_operation_id,
            ):
                require(reference, task_ids, "Auto-research invocation task")
        for message in history.messages:
            require(message.sender_task_id, task_ids, "Auto-research sender task")
            if message.recipient_task_id not in task_ids | child_work_ids:
                raise ValueError("Auto-research message references an unknown recipient")
        for recovery in history.recoveries:
            require(recovery.operation_id, task_ids, "Auto-research recovery task")
            require(
                recovery.admitted_operation_id,
                task_ids,
                "Auto-research admitted task",
            )
        for child in history.child_work:
            for reference in (
                child.root_operation_id,
                child.final_operation_id,
                child.admitted_by_operation_id,
            ):
                require(reference, task_ids, "Auto-research child Work task")
            for attempt in child.attempts:
                require(
                    attempt.operation_id,
                    task_ids,
                    "Auto-research child Work attempt",
                )
                require(
                    attempt.allocation_operation_id,
                    task_ids,
                    "Auto-research child Work allocation",
                )
        for child in history.child_experiments:
            require(
                child.parent_operation_id,
                task_ids,
                "Auto-research child Experiment parent task",
            )
            require(
                child.child_episode_id,
                episode_ids,
                "Auto-research child episode",
            )
            require(
                child.replaces_episode_id,
                episode_ids,
                "replaced child episode",
            )
            for invocation in child.invocations:
                require(
                    invocation.operation_id,
                    task_ids,
                    "Auto-research child Experiment invocation",
                )
        for admission in history.child_admissions:
            identities = child_work_ids if admission.child_kind == "work" else child_experiment_ids
            if admission.child_id not in identities:
                raise ValueError("Auto-research admission references an unknown child")
        for notice in history.lifecycle_notices:
            notice_sources = {
                "worker": child_work_ids,
                "experiment_task": task_ids,
                "experiment_episode": child_experiment_ids,
                "experiment_replacement": child_experiment_ids,
            }
            if notice.source_id not in notice_sources[notice.source_kind]:
                raise ValueError("Auto-research notice references an unknown lifecycle source")
            require(
                notice.acknowledged_by,
                task_ids,
                "Auto-research notice acknowledging task",
            )
        for receipt in history.inbox_receipts:
            require(
                receipt.acknowledged_by,
                task_ids,
                "Auto-research inbox acknowledging task",
            )
        for receipt in history.finish_receipts:
            require(
                receipt.actor_operation_id,
                task_ids,
                "Auto-research Finish actor task",
            )
        for result in history.apply_results:
            require(result.operation_id, task_ids, "Auto-research Apply task")
        for command in history.commands:
            require(command.operation_id, task_ids, "Auto-research command task")


def validate_transfer_table_policy(project_linked_tables: tuple[str, ...]) -> None:
    """Fail until every current project table has one explicit transfer disposition."""

    current = set(project_linked_tables)
    classified = TRANSFER_RECORD_TABLES | TRANSFER_EXCLUDED_PROJECT_TABLES
    if current != classified:
        missing = sorted(current - classified)
        stale = sorted(classified - current)
        detail = missing[0] if missing else stale[0]
        raise ValueError(f"project transfer table policy is incomplete at {detail}")


__all__ = [
    "TRANSFER_EXCLUDED_PROJECT_TABLES",
    "TRANSFER_EXECUTABLE_JSON_FIELDS",
    "TRANSFER_RECORD_SCHEMA_VERSION",
    "TRANSFER_RECORD_TABLES",
    "TransferArtifactReference",
    "TransferAssistantHistory",
    "TransferAutoResearchHistory",
    "TransferEpisodeRecord",
    "TransferJsonDocument",
    "TransferPaperDraft",
    "TransferRecordBundle",
    "TransferTaskOutput",
    "TransferTaskRecord",
    "TransferWatcherRecord",
    "capture_task_request_history",
    "sanitize_transfer_history_json",
    "validate_transfer_table_policy",
]
