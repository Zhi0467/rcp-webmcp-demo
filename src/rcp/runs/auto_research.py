from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp.agents.command_protocol import (
    MUTATING_COMMAND_VERBS,
    ApplyArguments,
    ApplyCommandRequest,
    CommandRequest,
    CommandResponse,
    EpisodeArguments,
    EpisodeCommandRequest,
    EpisodeControlArguments,
    ExperimentKickoffArguments,
    FinishCommandRequest,
    InboxArguments,
    InboxCommandRequest,
    MessageArguments,
    MessageCommandRequest,
    PauseCommandRequest,
    ResumeCommandRequest,
    SpawnArguments,
    SpawnCommandRequest,
    StatusArguments,
    StopCommandRequest,
    ValidateArguments,
    WatchGraphArguments,
    WatchGraphCommandRequest,
    command_requires_idempotency_key,
)
from rcp.limits import (
    AGENT_COMMAND_EVENT_MAX_BYTES,
    AGENT_TASK_RECEIPT_MAX_BYTES,
    AUTO_RESEARCH_APPLY_MAX_PER_TURN,
    AUTO_RESEARCH_PROMPT_FILE_MAX_BYTES,
    PATCH_SELF_CHECK_MAX_REQUEST_BYTES,
)
from rcp.providers import ProviderId, ProviderSkillReference
from rcp.skill_registry import SkillReference
from rcp.storage import (
    AgentCommandInvocationRecord,
    AgentTaskRecord,
    AppStore,
    AutoResearchChildAdmissionRecord,
    AutoResearchCommandFileRecord,
    AutoResearchMessageRecord,
    EpisodeNotRunning,
    EpisodeRecord,
)

if TYPE_CHECKING:
    from rcp.runs.episodes.wrapup import EpisodeWrapupSpec

AutoResearchActorRole = Literal["orchestrator", "worker"]
AutoResearchWakeCause = Literal["watcher", "graph_condition", "message", "lifecycle"]
AutoResearchWakeAdmission = Callable[
    [AgentTaskRecord, AutoResearchActorRole, AutoResearchWakeCause],
    AgentTaskRecord | None,
]
AutoResearchCommandFileReader = Callable[[str, int], str]
AutoResearchCommandFileConsumer = Callable[[str, str], bool]
AutoResearchCommandStateRefresher = Callable[[], tuple[int, str, str]]


class AutoResearchStartRequest(BaseModel):
    """Human-supplied and profile-resolved inputs for one new Auto-research episode."""

    model_config = ConfigDict(extra="forbid")

    invocation_ceiling: int = Field(ge=1)
    starting_instruction: str | None = Field(default=None, max_length=16_000)
    provider: ProviderId | None = None
    run_truth_scope: list[str] | None = None
    model: str | None = None
    reasoning: str | None = None
    run_on: str | None = None
    workflow_ids: list[str] | None = None
    skill_ids: list[str] | None = None
    invoked_workflow_ids: list[str] = Field(default_factory=list)
    invoked_skill_ids: list[str] = Field(default_factory=list)
    invoked_provider_skill_names: list[str] = Field(default_factory=list)
    resolved_provider_skills: list[ProviderSkillReference] = Field(default_factory=list)
    resolved_skill_packages: list[SkillReference] | None = None

    @model_validator(mode="after")
    def normalize_starting_instruction(self) -> AutoResearchStartRequest:
        if self.starting_instruction is not None:
            instruction = self.starting_instruction.strip()
            self.starting_instruction = instruction or None
        return self


class AutoResearchRunRequest(BaseModel):
    """One operational provider invocation inside an Auto-research episode.

    ``role`` is actor attribution, not a wake category. Watcher, graph-condition,
    message, and lifecycle delivery resume the same actor while spending another
    unit from the auto_research pot. Lifecycle delivery is root-orchestrator only.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    role: AutoResearchActorRole
    provider: ProviderId | None = None
    run_truth_scope: list[str] | None = None
    model: str | None = None
    reasoning: str | None = None
    run_on: str | None = None
    session_id: str | None = None
    actor_operation_id: str | None = None
    instruction: str | None = Field(default=None, max_length=16_000)
    control_node_id: str | None = None
    wake_cause: AutoResearchWakeCause | None = None
    watcher_ids: list[str] = Field(default_factory=list)
    workflow_ids: list[str] | None = None
    skill_ids: list[str] | None = None
    invoked_workflow_ids: list[str] = Field(default_factory=list)
    invoked_skill_ids: list[str] = Field(default_factory=list)
    invoked_provider_skill_names: list[str] = Field(default_factory=list)
    resolved_provider_skills: list[ProviderSkillReference] = Field(default_factory=list)
    resolved_skill_packages: list[SkillReference] | None = None

    @model_validator(mode="after")
    def role_fields_are_coherent(self) -> AutoResearchRunRequest:
        if self.actor_operation_id is not None:
            actor_operation_id = self.actor_operation_id.strip()
            if not actor_operation_id:
                raise ValueError("an Auto-research actor operation id must not be blank")
            self.actor_operation_id = actor_operation_id
        if self.instruction is not None:
            instruction = self.instruction.strip()
            self.instruction = instruction or None
        if self.role == "worker" and not self.control_node_id:
            raise ValueError(
                "an Auto-research worker must name the Experiment or Blocker seating it"
            )
        if self.wake_cause is not None and self.session_id is None:
            raise ValueError("an Auto-research wake must resume its saved native session")
        if self.wake_cause == "lifecycle" and self.role != "orchestrator":
            raise ValueError("only the Auto-research orchestrator may receive lifecycle facts")
        if len(self.watcher_ids) != len(set(self.watcher_ids)):
            raise ValueError("an Auto-research wake cannot repeat watcher ids")
        if self.watcher_ids and self.wake_cause not in {"watcher", "graph_condition"}:
            raise ValueError("only an Auto-research watcher wake may carry watcher ids")
        return self


def auto_research_root_request(
    request: AutoResearchStartRequest,
    *,
    episode_id: str,
) -> AutoResearchRunRequest:
    """Capture a resolved Auto-research start as its first orchestrator turn."""

    values = request.model_dump(mode="json", exclude={"invocation_ceiling", "starting_instruction"})
    return AutoResearchRunRequest.model_validate(
        {
            **values,
            "episode_id": episode_id,
            "role": "orchestrator",
            "instruction": request.starting_instruction,
        }
    )


class AutoResearchEndingSignal(BaseModel):
    """One durable mode ending handed to central episode settlement."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    ending: Literal["completed", "exhausted", "failed", "human_pause"]
    partial: bool
    diagnostic: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def partial_matches_ending(self) -> AutoResearchEndingSignal:
        if self.partial != (self.ending != "completed"):
            raise ValueError("an Auto-research ending signal has inconsistent partial state")
        return self


def fence_auto_research_ending(
    store: AppStore,
    episode_id: str,
    ending: Literal["completed", "exhausted", "failed", "human_pause"],
    *,
    diagnostic: str | None = None,
) -> AutoResearchEndingSignal:
    """Fence new work and return the mode signal central settlement consumes."""

    episode = _auto_research_episode(store, episode_id)
    store.fence_auto_research_ending_and_settle_watchers(
        episode.episode_id,
        ending,
        diagnostic=diagnostic,
    )
    return AutoResearchEndingSignal(
        episode_id=episode_id,
        ending=ending,
        partial=ending != "completed",
        diagnostic=diagnostic,
    )


def auto_research_completion_signal(
    store: AppStore,
    episode_id: str,
    *,
    diagnostic: str | None = None,
) -> AutoResearchEndingSignal:
    return fence_auto_research_ending(
        store,
        episode_id,
        "completed",
        diagnostic=diagnostic,
    )


def auto_research_exhaustion_signal(
    store: AppStore,
    episode_id: str,
    *,
    diagnostic: str | None = None,
) -> AutoResearchEndingSignal:
    return fence_auto_research_ending(
        store,
        episode_id,
        "exhausted",
        diagnostic=diagnostic,
    )


def auto_research_failure_signal(
    store: AppStore,
    episode_id: str,
    *,
    diagnostic: str,
) -> AutoResearchEndingSignal:
    return fence_auto_research_ending(
        store,
        episode_id,
        "failed",
        diagnostic=diagnostic,
    )


def request_auto_research_stop(store: AppStore, episode_id: str) -> EpisodeRecord:
    """Persist Stop and retain every Auto watcher as one atomic boundary."""

    _auto_research_episode(store, episode_id)
    return store.request_auto_research_stop_and_settle_watchers(episode_id)


def settle_auto_research_stop(
    store: AppStore,
    episode_id: str,
    *,
    diagnostic: str | None = None,
) -> EpisodeRecord | None:
    """Settle Stop once all already-authorized Auto work is quiescent."""

    episode = _auto_research_episode(store, episode_id)
    if episode.stop_requested_at is None:
        raise EpisodeNotRunning("the Auto-research episode has no durable Stop request")
    if not store.auto_research_is_quiescent(episode_id):
        return None
    return store.mark_episode_stop_skipped(episode_id, diagnostic=diagnostic)


def auto_research_wrapup_spec(
    store: AppStore,
    signal: AutoResearchEndingSignal,
) -> EpisodeWrapupSpec:
    """Build a compact receipt and select the root actor's exact latest task."""

    from rcp.runs.episodes.wrapup import EpisodeWrapupSpec

    episode = _auto_research_episode(store, signal.episode_id)
    if episode.ending != signal.ending or episode.ending_diagnostic != signal.diagnostic:
        raise ValueError("the Auto-research ending signal differs from its durable fence")
    if episode.root_operation_id is None:
        raise ValueError("the Auto-research episode has no root orchestrator actor")
    binding = store.auto_research_actor_binding(episode.root_operation_id)
    if binding.episode_id != episode.episode_id or binding.role != "orchestrator":
        raise ValueError("the Auto-research root actor binding is inconsistent")
    continuation = store.agent_task(binding.current_operation_id)
    if continuation is None:
        raise ValueError("the Auto-research root actor lost its latest continuation task")

    state = store.auto_research_state(episode.episode_id)
    meter = store.episode_budget_meter(episode.episode_id)
    tasks = store.auto_research_tasks(episode.episode_id)
    task_statuses: dict[str, int] = {}
    actor_rows: dict[str, dict[str, object]] = {}
    graph_results: list[dict[str, object]] = []
    for task in tasks:
        task_statuses[task.status] = task_statuses.get(task.status, 0) + 1
        invocation = store.auto_research_invocation(task.operation_id)
        if invocation is not None:
            actor_rows[invocation.actor_operation_id] = {
                "actor_operation_id": _receipt_text(invocation.actor_operation_id, 160),
                "role": invocation.role,
                "control_node_id": _receipt_text(invocation.control_node_id, 240),
                "latest_operation_id": _receipt_text(task.operation_id, 160),
                "latest_status": task.status,
                "latest_attempt": task.attempt,
            }
        graph_update = task.result.get("graph_update") if isinstance(task.result, dict) else None
        if isinstance(graph_update, dict):
            graph_results.append(
                {
                    "operation_id": _receipt_text(task.operation_id, 160),
                    "status": _receipt_text(graph_update.get("status"), 80),
                    "applied_revision": graph_update.get("applied_revision"),
                }
            )

    _, events = store.auto_research_event_history(episode.episode_id, limit=64)
    command_facts = [
        {
            "operation_id": _receipt_text(event.operation_id, 160),
            "verb": _receipt_text(event.command_verb, 80),
            "phase": event.command_phase,
            "level": event.level,
        }
        for event in events
        if event.event_kind == "command"
    ][-16:]
    work_routes = store.auto_research_child_works(episode.episode_id)
    child_work: list[dict[str, object]] = []
    for route in work_routes[-16:]:
        current = store.agent_task(route.current_operation_id)
        child_work.append(
            {
                "worker_id": _receipt_text(route.worker_id, 160),
                "control_node_id": _receipt_text(route.control_node_id, 240),
                "current_operation_id": _receipt_text(route.current_operation_id, 160),
                "status": current.status if current is not None else "missing",
                "attempt": current.attempt if current is not None else None,
                "stop_requested": route.stop_requested_at is not None,
            }
        )
    experiment_routes = store.auto_research_child_experiments(episode.episode_id)
    child_experiments: list[dict[str, object]] = []
    for route in experiment_routes[-16:]:
        child = store.episode(route.child_episode_id)
        child_experiments.append(
            {
                "episode_id": _receipt_text(route.child_episode_id, 160),
                "control_node_id": _receipt_text(route.control_node_id, 240),
                "route_state": route.state,
                "status": child.status if child is not None else route.state,
                "ending": child.ending if child is not None else None,
                "replaces_episode_id": _receipt_text(route.replaces_episode_id, 160),
                "diagnostic": _receipt_text(route.terminal_diagnostic, 480),
            }
        )
    lifecycle_notices = store.auto_research_lifecycle_notices(episode.episode_id)
    lifecycle_counts = {"pending": 0, "delivered": 0, "acknowledged": 0}
    for notice in lifecycle_notices:
        lifecycle_counts[notice.state] += 1
    lifecycle_facts = [
        {
            "notice_id": _receipt_text(notice.notice_id, 160),
            "source_kind": _receipt_text(notice.source_kind, 80),
            "source_id": _receipt_text(notice.source_id, 160),
            "source_event": _receipt_text(notice.source_event, 80),
            "source_attempt": notice.source_attempt,
            "state": notice.state,
            "payload": _receipt_text(
                json.dumps(
                    notice.payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                800,
            ),
        }
        for notice in lifecycle_notices[-16:]
    ]
    experiment_allowance = store.auto_research_experiment_allowance(episode.episode_id)
    pending_admissions = store.pending_auto_research_child_admissions(episode.episode_id)
    receipt: dict[str, object] = {
        "starting_instruction": _receipt_text(
            state.starting_instruction if state is not None else None,
            1_200,
        ),
        "operational_meter": {
            "ceiling": meter.invocation_ceiling,
            "used": meter.invocations_used,
            "remaining": meter.invocations_remaining,
            "observed_input_tokens": meter.observed_input_tokens,
            "observed_generated_tokens": meter.observed_generated_tokens,
        },
        "task_status_counts": dict(sorted(task_statuses.items())),
        "actors": list(actor_rows.values())[-16:],
        "omitted_actor_count": max(0, len(actor_rows) - 16),
        "command_facts": command_facts,
        "graph_results": graph_results[-16:],
        "experiment_allowance": experiment_allowance.model_dump(mode="json"),
        "child_work": child_work,
        "omitted_child_work_count": max(0, len(work_routes) - len(child_work)),
        "child_experiments": child_experiments,
        "omitted_child_experiment_count": max(0, len(experiment_routes) - len(child_experiments)),
        "lifecycle": {
            "counts": lifecycle_counts,
            "facts": lifecycle_facts,
            "omitted_fact_count": max(0, len(lifecycle_notices) - len(lifecycle_facts)),
        },
        "pending_child_admission_ids": [
            _receipt_text(item.admission_id, 160) for item in pending_admissions[:16]
        ],
        "omitted_pending_child_admission_count": max(0, len(pending_admissions) - 16),
    }
    if _receipt_size(receipt) > AGENT_TASK_RECEIPT_MAX_BYTES:
        receipt["actors"] = list(actor_rows.values())[-8:]
        receipt["command_facts"] = command_facts[-8:]
        receipt["graph_results"] = graph_results[-8:]
        receipt["child_work"] = child_work[-8:]
        receipt["omitted_child_work_count"] = max(0, len(work_routes) - 8)
        receipt["child_experiments"] = child_experiments[-8:]
        receipt["omitted_child_experiment_count"] = max(0, len(experiment_routes) - 8)
        lifecycle = receipt["lifecycle"]
        assert isinstance(lifecycle, dict)
        lifecycle["facts"] = lifecycle_facts[-8:]
        lifecycle["omitted_fact_count"] = max(0, len(lifecycle_notices) - 8)
        receipt["starting_instruction"] = _receipt_text(
            state.starting_instruction if state is not None else None,
            480,
        )
    if _receipt_size(receipt) > AGENT_TASK_RECEIPT_MAX_BYTES:
        raise ValueError("the compact Auto-research ending receipt exceeds its storage boundary")
    return EpisodeWrapupSpec(
        episode_id=episode.episode_id,
        ending=signal.ending,
        partial=signal.partial,
        continuation_operation_id=continuation.operation_id,
        receipt=receipt,
        diagnostic=signal.diagnostic,
    )


def _auto_research_episode(store: AppStore, episode_id: str) -> EpisodeRecord:
    episode = store.episode(episode_id)
    if episode is None:
        raise KeyError(episode_id)
    if episode.mode != "auto_research" or store.auto_research_state(episode_id) is None:
        raise ValueError("the episode is not a canonical Auto-research episode")
    return episode


def _receipt_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"


def _receipt_size(value: dict[str, object]) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


class PendingAutoResearchMail(BaseModel):
    """Unclaimed hearsay-only messages awaiting one atomic wake admission."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    recipient_task_id: str
    messages: list[AutoResearchMessageRecord]
    graph_authority: Literal["none"] = "none"

    @property
    def message_ids(self) -> list[str]:
        return [message.message_id for message in self.messages]


def pending_auto_research_mail(
    store: AppStore,
    *,
    episode_id: str,
    recipient_task_id: str,
) -> PendingAutoResearchMail:
    """Read one recipient's undelivered mail without claiming a wake path."""

    recipient = store.agent_task(recipient_task_id)
    if recipient is None:
        raise KeyError(recipient_task_id)
    if recipient.episode_id != episode_id:
        raise ValueError("The mail recipient is outside this Auto-research episode.")
    messages = store.pending_auto_research_messages(episode_id, recipient_task_id)
    return PendingAutoResearchMail(
        episode_id=episode_id,
        recipient_task_id=recipient_task_id,
        messages=messages,
    )


class AutoResearchCommandInvalid(ValueError):
    """A staged command is well-formed but not permitted or applicable."""


class AutoResearchCommandUnavailable(RuntimeError):
    """A staged command could not reach the authoritative effect boundary."""


class AutoResearchCommandEffectResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "invalid", "unavailable"] = "ok"
    message: str | None = Field(default=None, max_length=2_000)
    result: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unsuccessful_result_has_a_diagnostic(self) -> AutoResearchCommandEffectResult:
        if self.status != "ok" and not (self.message or "").strip():
            raise ValueError("An unsuccessful Auto-research command requires a diagnostic.")
        try:
            encoded = json.dumps(
                {
                    "status": self.status,
                    "result": self.result,
                    **({"diagnostic": self.message} if self.message else {}),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("The Auto-research command result must be valid JSON.") from exc
        if len(encoded) > AGENT_COMMAND_EVENT_MAX_BYTES:
            raise ValueError("The Auto-research command result exceeds the event ledger limit.")
        return self


@dataclass(frozen=True)
class AutoResearchCommandContext:
    episode: EpisodeRecord
    task: AgentTaskRecord
    request: AutoResearchRunRequest
    command_file: AutoResearchCommandFile | None = None
    consume_command_file: AutoResearchCommandFileConsumer | None = None
    refresh_command_state: AutoResearchCommandStateRefresher | None = None


@dataclass(frozen=True)
class AutoResearchCommandFile:
    kind: Literal["apply", "instruction", "goal"]
    filename: str
    text: str
    sha256: str


AutoResearchValidateCommand = Callable[
    [AutoResearchCommandContext, ValidateArguments],
    AutoResearchCommandEffectResult,
]
AutoResearchApplyCommand = Callable[
    [AutoResearchCommandContext, ApplyArguments, str],
    AutoResearchCommandEffectResult,
]
AutoResearchStatusCommand = Callable[
    [AutoResearchCommandContext, StatusArguments],
    AutoResearchCommandEffectResult,
]
AutoResearchSpawnCommand = Callable[
    [AutoResearchCommandContext, SpawnArguments, str],
    AutoResearchCommandEffectResult,
]
AutoResearchWorkerCommand = Callable[
    [AutoResearchCommandContext, str],
    AutoResearchCommandEffectResult,
]
AutoResearchWorkerResumeCommand = Callable[
    [AutoResearchCommandContext, str, str],
    AutoResearchCommandEffectResult,
]
AutoResearchMessageCommand = Callable[
    [AutoResearchCommandContext, MessageArguments, str],
    AutoResearchCommandEffectResult,
]
AutoResearchWatchGraphCommand = Callable[
    [AutoResearchCommandContext, WatchGraphArguments, str],
    AutoResearchCommandEffectResult,
]
AutoResearchFinishCommand = Callable[
    [AutoResearchCommandContext, str], AutoResearchCommandEffectResult
]
AutoResearchEpisodeCommand = Callable[
    [AutoResearchCommandContext, EpisodeArguments, str],
    AutoResearchCommandEffectResult,
]
AutoResearchInboxCommand = Callable[
    [AutoResearchCommandContext, InboxArguments, str],
    AutoResearchCommandEffectResult,
]
AutoResearchUnknownCommandReconciler = Callable[
    [AutoResearchCommandContext, CommandRequest, str | None],
    AutoResearchCommandEffectResult | None,
]
AutoResearchSeatNodeType = Callable[[str, str], str | None]
AutoResearchWorkerLookup = Callable[
    [AutoResearchCommandContext, str],
    AgentTaskRecord,
]
AutoResearchSpawnVerifier = Callable[
    [AutoResearchCommandContext, SpawnArguments, str],
    AgentTaskRecord,
]


def _unsupported_apply(
    _context: AutoResearchCommandContext,
    _arguments: ApplyArguments,
    _planned_apply_id: str,
) -> AutoResearchCommandEffectResult:
    raise AutoResearchCommandUnavailable("In-turn Apply is not available in this runtime.")


def _unsupported_episode(
    _context: AutoResearchCommandContext,
    _arguments: EpisodeArguments,
    _planned_episode_effect_id: str,
) -> AutoResearchCommandEffectResult:
    raise AutoResearchCommandUnavailable("Experiment episode control is not available.")


def _unsupported_inbox(
    _context: AutoResearchCommandContext,
    _arguments: InboxArguments,
    _planned_inbox_effect_id: str,
) -> AutoResearchCommandEffectResult:
    raise AutoResearchCommandUnavailable("The lifecycle inbox is not available.")


@dataclass(frozen=True)
class AutoResearchCommandEffects:
    """Injected graph/run effects behind the staged transport protocol.

    This seam lets API composition bind existing validator, watcher, and
    BackgroundAgentTasks behavior without making an execution host call RCP over
    HTTP and without adding another wake implementation here.
    """

    validate: AutoResearchValidateCommand
    status: AutoResearchStatusCommand
    spawn: AutoResearchSpawnCommand
    pause: AutoResearchWorkerCommand
    resume: AutoResearchWorkerResumeCommand
    stop: AutoResearchWorkerCommand
    message: AutoResearchMessageCommand
    watch_graph: AutoResearchWatchGraphCommand
    finish: AutoResearchFinishCommand
    seat_node_type: AutoResearchSeatNodeType
    reconcile_unknown: AutoResearchUnknownCommandReconciler
    apply: AutoResearchApplyCommand = _unsupported_apply
    episode: AutoResearchEpisodeCommand = _unsupported_episode
    inbox: AutoResearchInboxCommand = _unsupported_inbox
    worker_lookup: AutoResearchWorkerLookup | None = None
    verify_spawn: AutoResearchSpawnVerifier | None = None


@dataclass
class _AutoResearchApplyAdmissionLocks:
    guard: threading.Lock = field(default_factory=threading.Lock)
    by_operation: dict[str, threading.Lock] = field(default_factory=dict)

    def for_operation(self, operation_id: str) -> threading.Lock:
        with self.guard:
            return self.by_operation.setdefault(operation_id, threading.Lock())


class AutoResearchCommandDispatcher:
    """Audit, deduplicate, reconcile, and dispatch one staged client call."""

    def __init__(
        self,
        store: AppStore,
        effects: AutoResearchCommandEffects,
        *,
        command_file_reader: AutoResearchCommandFileReader | None = None,
        command_file_consumer: AutoResearchCommandFileConsumer | None = None,
        command_state_refresher: AutoResearchCommandStateRefresher | None = None,
        _apply_admission_locks: _AutoResearchApplyAdmissionLocks | None = None,
    ) -> None:
        self.store = store
        self.effects = effects
        self.command_file_reader = command_file_reader
        self.command_file_consumer = command_file_consumer
        self.command_state_refresher = command_state_refresher
        self._apply_admission_locks = _apply_admission_locks or _AutoResearchApplyAdmissionLocks()

    def with_command_files(
        self,
        *,
        reader: AutoResearchCommandFileReader,
        consumer: AutoResearchCommandFileConsumer,
        refresher: AutoResearchCommandStateRefresher,
    ) -> AutoResearchCommandDispatcher:
        """Bind this turn's exact reusable stage to file-backed commands."""

        return AutoResearchCommandDispatcher(
            self.store,
            self.effects,
            command_file_reader=reader,
            command_file_consumer=consumer,
            command_state_refresher=refresher,
            _apply_admission_locks=self._apply_admission_locks,
        )

    def dispatch(self, operation_id: str, request: CommandRequest) -> CommandResponse:
        context = self._context(operation_id)
        if request.mailbox_id == "":  # already schema-validated; keeps the binding explicit here
            raise AutoResearchCommandInvalid("The Auto-research command mailbox is missing.")

        planned_worker_id = (
            _planned_worker_id(context.episode.episode_id, request.idempotency_key)
            if isinstance(request, SpawnCommandRequest) and request.idempotency_key is not None
            else None
        )
        planned_message_id = (
            _planned_effect_id(
                context.episode.episode_id,
                "message",
                request.idempotency_key,
            )
            if isinstance(request, MessageCommandRequest) and request.idempotency_key is not None
            else None
        )
        planned_watcher_id = (
            _planned_effect_id(
                context.episode.episode_id,
                "watch_graph",
                request.idempotency_key,
            )
            if isinstance(request, WatchGraphCommandRequest) and request.idempotency_key is not None
            else None
        )
        planned_apply_id = (
            _planned_effect_id(
                context.episode.episode_id,
                "apply",
                request.idempotency_key,
            )
            if isinstance(request, ApplyCommandRequest) and request.idempotency_key is not None
            else None
        )
        planned_resume_operation_id = (
            _planned_effect_id(
                context.episode.episode_id,
                "resume",
                request.idempotency_key,
            )
            if isinstance(request, ResumeCommandRequest) and request.idempotency_key is not None
            else None
        )
        planned_episode_effect_id = (
            _planned_effect_id(
                context.episode.episode_id,
                "episode",
                request.idempotency_key,
            )
            if isinstance(request, EpisodeCommandRequest) and request.idempotency_key is not None
            else None
        )
        planned_inbox_effect_id = (
            _planned_effect_id(
                context.episode.episode_id,
                "inbox",
                request.idempotency_key,
            )
            if isinstance(request, InboxCommandRequest) and request.idempotency_key is not None
            else None
        )
        planned_finish_effect_id = (
            _planned_effect_id(
                context.episode.episode_id,
                "finish",
                request.idempotency_key,
            )
            if isinstance(request, FinishCommandRequest) and request.idempotency_key is not None
            else None
        )
        arguments = request.arguments.model_dump(mode="json")
        if request.verb == "validate":
            patch = request.arguments.patch
            encoded_patch = patch.encode("utf-8")
            arguments = {
                "patch_byte_length": len(encoded_patch),
                "patch_sha256": hashlib.sha256(encoded_patch).hexdigest(),
            }
        start_payload = {
            "request_id": request.request_id,
            "arguments": arguments,
        }
        if planned_worker_id is not None:
            start_payload["planned_worker_id"] = planned_worker_id
        if planned_message_id is not None:
            start_payload["planned_message_id"] = planned_message_id
        if planned_watcher_id is not None:
            start_payload["planned_watcher_id"] = planned_watcher_id
        if planned_apply_id is not None:
            start_payload["planned_apply_id"] = planned_apply_id
        if planned_resume_operation_id is not None:
            start_payload["planned_resume_operation_id"] = planned_resume_operation_id
        if planned_episode_effect_id is not None:
            start_payload["planned_episode_effect_id"] = planned_episode_effect_id
        if planned_inbox_effect_id is not None:
            start_payload["planned_inbox_effect_id"] = planned_inbox_effect_id
        if planned_finish_effect_id is not None:
            start_payload["planned_finish_effect_id"] = planned_finish_effect_id
        prior = (
            self.store.agent_command_by_key(
                context.episode.episode_id,
                request.idempotency_key,
            )
            if request.idempotency_key is not None
            else None
        )

        if prior is not None:
            attempt = self._start_retry_attempt(context, request, start_payload, prior)
            return self._dispatch_retry(context, request, prior, attempt)

        keyed_apply = (
            isinstance(request, ApplyCommandRequest) and request.idempotency_key is not None
        )
        admission_lock = (
            self._apply_admission_locks.for_operation(operation_id) if keyed_apply else None
        )
        if admission_lock is not None:
            admission_lock.acquire()
        try:
            if keyed_apply:
                locked_prior = self.store.agent_command_by_key(
                    context.episode.episode_id,
                    request.idempotency_key,
                )
                if locked_prior is not None:
                    admission_lock.release()
                    admission_lock = None
                    attempt = self._start_retry_attempt(
                        context,
                        request,
                        start_payload,
                        locked_prior,
                    )
                    return self._dispatch_retry(context, request, locked_prior, attempt)

            command_file: AutoResearchCommandFile | None = None
            command_file_error: AutoResearchCommandEffectResult | None = None
            apply_limit_reached = keyed_apply and (
                self.store.auto_research_apply_admission_count(operation_id)
                >= AUTO_RESEARCH_APPLY_MAX_PER_TURN
            )
            if not apply_limit_reached:
                try:
                    command_file = self._snapshot_command_file(request)
                except AutoResearchCommandInvalid as exc:
                    command_file_error = AutoResearchCommandEffectResult(
                        status="invalid",
                        message=str(exc),
                    )
                except (AutoResearchCommandUnavailable, OSError) as exc:
                    command_file_error = AutoResearchCommandEffectResult(
                        status="unavailable",
                        message=str(exc),
                    )
            if command_file is not None:
                start_payload["command_file"] = {
                    "filename": command_file.filename,
                    "byte_length": len(command_file.text.encode("utf-8")),
                    "sha256": command_file.sha256,
                }

            if (
                command_file_error is not None
                and command_file_error.status == "unavailable"
                and request.idempotency_key is not None
            ):
                # The keyed intent does not exist until its immutable file bytes
                # can be committed with the command start.  Keep the failed read
                # visible as an unkeyed audit attempt, but leave the key and Apply
                # admission slot free for an exact later call.  Invalid file
                # content remains a canonical keyed result below.
                command_id = self._unused_command_id(request.request_id)
                audit_payload = {
                    **start_payload,
                    "attempted_idempotency_key": request.idempotency_key,
                    "pre_admission_unavailable": True,
                }
                invocation = self.store.start_agent_command(
                    operation_id=operation_id,
                    command_id=command_id,
                    episode_id=context.episode.episode_id,
                    verb=request.verb,
                    idempotency_key=None,
                    payload=audit_payload,
                )
                return self._finish(
                    invocation.command_id,
                    request.request_id,
                    command_file_error,
                )

            command_id = self._unused_command_id(request.request_id)
            admitted_at = self.store.now()
            file_snapshot = (
                AutoResearchCommandFileRecord(
                    command_id=command_id,
                    episode_id=context.episode.episode_id,
                    operation_id=operation_id,
                    kind=command_file.kind,
                    filename=command_file.filename,
                    sha256=command_file.sha256,
                    content=command_file.text,
                    created_at=admitted_at,
                )
                if command_file is not None
                else None
            )
            child_admission = self._child_admission(
                context,
                request,
                planned_worker_id=planned_worker_id,
                planned_episode_effect_id=planned_episode_effect_id,
                created_at=admitted_at,
                command_file_error=command_file_error,
            )
            try:
                invocation = self.store.start_agent_command(
                    operation_id=operation_id,
                    command_id=command_id,
                    episode_id=context.episode.episode_id,
                    verb=request.verb,
                    idempotency_key=request.idempotency_key,
                    payload=start_payload,
                    file_snapshot=file_snapshot,
                    child_admission=child_admission,
                    apply_admission_limit=(
                        AUTO_RESEARCH_APPLY_MAX_PER_TURN if keyed_apply else None
                    ),
                )
            except ValueError:
                # Another client may have won the auto_research-wide key between the
                # read above and this insert. Preserve that invocation as the
                # effect record while giving this client call its own ledger pair.
                raced = (
                    self.store.agent_command_by_key(
                        context.episode.episode_id,
                        request.idempotency_key,
                    )
                    if request.idempotency_key is not None
                    else None
                )
                if raced is None:
                    raise
                if admission_lock is not None:
                    admission_lock.release()
                    admission_lock = None
                attempt = self._start_retry_attempt(context, request, start_payload, raced)
                return self._dispatch_retry(context, request, raced, attempt)
        finally:
            if admission_lock is not None:
                admission_lock.release()
        if invocation.command_id != command_id:
            # ``start_agent_command`` returns the winning keyed invocation on
            # a race without inserting this call. Record this call separately.
            attempt = self._start_retry_attempt(context, request, start_payload, invocation)
            return self._dispatch_retry(context, request, invocation, attempt)

        if (
            isinstance(request, ApplyCommandRequest)
            and invocation.start_payload.get("apply_admitted") is False
        ):
            return self._finish(
                invocation.command_id,
                request.request_id,
                _apply_admission_limit_outcome(),
            )

        if command_requires_idempotency_key(request.verb) and request.idempotency_key is None:
            return self._finish(
                invocation.command_id,
                request.request_id,
                AutoResearchCommandEffectResult(
                    status="invalid",
                    message=f"Agent command {request.verb} requires an idempotency key.",
                ),
            )

        if command_file_error is not None:
            return self._finish(
                invocation.command_id,
                request.request_id,
                command_file_error,
            )

        try:
            outcome = self._execute(
                replace(context, command_file=command_file),
                request,
                planned_worker_id=planned_worker_id,
                planned_message_id=planned_message_id,
                planned_watcher_id=planned_watcher_id,
                planned_apply_id=planned_apply_id,
                planned_resume_operation_id=planned_resume_operation_id,
                planned_episode_effect_id=planned_episode_effect_id,
                planned_inbox_effect_id=planned_inbox_effect_id,
                planned_finish_effect_id=planned_finish_effect_id,
            )
        except AutoResearchCommandInvalid as exc:
            outcome = AutoResearchCommandEffectResult(status="invalid", message=str(exc))
        except (AutoResearchCommandUnavailable, OSError) as exc:
            outcome = AutoResearchCommandEffectResult(status="unavailable", message=str(exc))
        except (KeyError, ValueError) as exc:
            outcome = AutoResearchCommandEffectResult(status="invalid", message=str(exc))
        if outcome.status == "invalid" and child_admission is not None:
            self._cancel_known_child_admission(child_admission.admission_id)
        return self._finish(invocation.command_id, request.request_id, outcome)

    def _child_admission(
        self,
        context: AutoResearchCommandContext,
        request: CommandRequest,
        *,
        planned_worker_id: str | None,
        planned_episode_effect_id: str | None,
        created_at: str,
        command_file_error: AutoResearchCommandEffectResult | None,
    ) -> AutoResearchChildAdmissionRecord | None:
        """Bind a fresh child intent to the command start and immutable file snapshot."""

        if command_file_error is not None:
            return None
        if isinstance(request, SpawnCommandRequest):
            if planned_worker_id is None:
                return None
            child_kind: Literal["work", "experiment"] = "work"
            child_id = planned_worker_id
        elif isinstance(request, EpisodeCommandRequest) and isinstance(
            request.arguments,
            ExperimentKickoffArguments,
        ):
            if planned_episode_effect_id is None:
                return None
            child_kind = "experiment"
            child_id = planned_episode_effect_id
        else:
            return None
        return AutoResearchChildAdmissionRecord(
            admission_id=child_id,
            episode_id=context.episode.episode_id,
            project_id=context.task.project_id,
            child_kind=child_kind,
            child_id=child_id,
            state="accepted",
            created_at=created_at,
            updated_at=created_at,
        )

    def _cancel_known_child_admission(self, admission_id: str) -> None:
        admission = self.store.auto_research_child_admission(admission_id)
        if admission is not None and admission.state == "accepted":
            self.store.cancel_auto_research_child_admission(admission_id)

    def _snapshot_command_file(
        self,
        request: CommandRequest,
    ) -> AutoResearchCommandFile | None:
        spec: tuple[Literal["apply", "instruction", "goal"], str, int, bool] | None = None
        if isinstance(request, ApplyCommandRequest):
            spec = (
                "apply",
                request.arguments.patch_file,
                PATCH_SELF_CHECK_MAX_REQUEST_BYTES,
                False,
            )
        elif isinstance(request, SpawnCommandRequest):
            spec = (
                "instruction",
                request.arguments.instruction_file,
                AUTO_RESEARCH_PROMPT_FILE_MAX_BYTES,
                True,
            )
        elif (
            isinstance(request, EpisodeCommandRequest)
            and isinstance(request.arguments, ExperimentKickoffArguments)
            and request.arguments.goal_file is not None
        ):
            spec = (
                "goal",
                request.arguments.goal_file,
                AUTO_RESEARCH_PROMPT_FILE_MAX_BYTES,
                True,
            )
        if spec is None:
            return None
        if self.command_file_reader is None:
            raise AutoResearchCommandUnavailable(
                "The exact Auto-research command workspace is unavailable."
            )
        kind, filename, max_bytes, require_nonblank = spec
        try:
            text = self.command_file_reader(filename, max_bytes)
        except FileNotFoundError as exc:
            raise AutoResearchCommandInvalid(
                f"Command file {filename!r} does not exist in the orchestrator workspace."
            ) from exc
        except ValueError as exc:
            raise AutoResearchCommandInvalid(str(exc)) from exc
        if require_nonblank and not text.strip():
            raise AutoResearchCommandInvalid(f"Command file {filename!r} must not be blank.")
        encoded = text.encode("utf-8")
        return AutoResearchCommandFile(
            kind=kind,
            filename=filename,
            text=text,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def _start_retry_attempt(
        self,
        context: AutoResearchCommandContext,
        request: CommandRequest,
        start_payload: dict[str, object],
        prior: AgentCommandInvocationRecord,
    ) -> AgentCommandInvocationRecord:
        """Give every client retry its own start/exit pair without owning the key."""

        attempt_payload = {
            **start_payload,
            "idempotency_key": request.idempotency_key,
            "deduplicates_command_id": prior.command_id,
        }
        return self.store.start_agent_command(
            operation_id=context.task.operation_id,
            command_id=self._unused_command_id(request.request_id),
            episode_id=context.episode.episode_id,
            verb=request.verb,
            idempotency_key=None,
            payload=attempt_payload,
        )

    def _dispatch_retry(
        self,
        retry_context: AutoResearchCommandContext,
        request: CommandRequest,
        prior: AgentCommandInvocationRecord,
        attempt: AgentCommandInvocationRecord,
    ) -> CommandResponse:
        """Resolve one keyed retry from the original durable request intent."""

        try:
            original_context = self._context(prior.operation_id)
            if original_context.episode.episode_id != retry_context.episode.episode_id:
                raise AutoResearchCommandUnavailable(
                    "The original command task no longer belongs to this Auto-research episode."
                )
        except (
            AutoResearchCommandInvalid,
            AutoResearchCommandUnavailable,
            KeyError,
            ValueError,
        ) as exc:
            return self._finish(
                attempt.command_id,
                request.request_id,
                AutoResearchCommandEffectResult(status="unavailable", message=str(exc)),
            )

        try:
            original_actor = self._canonical_command_actor(original_context)
            retry_actor = self._canonical_command_actor(retry_context)
        except (
            AutoResearchCommandInvalid,
            AutoResearchCommandUnavailable,
            KeyError,
            ValueError,
        ) as exc:
            return self._finish(
                attempt.command_id,
                request.request_id,
                AutoResearchCommandEffectResult(status="unavailable", message=str(exc)),
            )
        if retry_actor != original_actor:
            return self._finish(
                attempt.command_id,
                request.request_id,
                AutoResearchCommandEffectResult(
                    status="invalid",
                    message=(
                        "An idempotency key may be replayed only by the same canonical "
                        "Auto-research actor and role."
                    ),
                ),
            )

        try:
            recorded_request = self._recorded_request(request, prior)
        except AutoResearchCommandInvalid as exc:
            return self._finish(
                attempt.command_id,
                request.request_id,
                AutoResearchCommandEffectResult(status="invalid", message=str(exc)),
            )
        except (AutoResearchCommandUnavailable, KeyError, ValueError) as exc:
            outcome = AutoResearchCommandEffectResult(status="unavailable", message=str(exc))
            if prior.exited_at is None:
                outcome = self._finish_original_unknown(prior.command_id, outcome)
            return self._finish(attempt.command_id, request.request_id, outcome)

        original_was_unknown = prior.exited_at is None
        if (
            isinstance(recorded_request, ApplyCommandRequest)
            and prior.start_payload.get("apply_admitted") is False
        ):
            outcome = _apply_admission_limit_outcome()
            if original_was_unknown:
                outcome = self._finish_original_unknown(prior.command_id, outcome)
            return self._finish(attempt.command_id, request.request_id, outcome)
        if not original_was_unknown:
            try:
                recorded = _effect_from_recorded_invocation(prior)
                if recorded.status != "unavailable":
                    outcome = self._completed_retry_outcome(
                        original_context,
                        recorded_request,
                        prior,
                    )
                    return self._finish(attempt.command_id, request.request_id, outcome)
            except (
                AutoResearchCommandInvalid,
                AutoResearchCommandUnavailable,
                KeyError,
                ValueError,
            ) as exc:
                outcome = AutoResearchCommandEffectResult(status="unavailable", message=str(exc))
                return self._finish(attempt.command_id, request.request_id, outcome)

        try:
            original_context = replace(
                original_context,
                command_file=self._recorded_command_file(prior, recorded_request),
            )
            reconciled = self._reconcile_unknown(
                original_context,
                recorded_request,
                prior.start_payload,
            )
            if reconciled is None:
                if not self._may_reexecute_after_unproven_outcome(recorded_request):
                    raise AutoResearchCommandUnavailable(
                        "Interrupted command outcome is unknown and could not be proven; "
                        "it was not re-executed."
                    )
                planned_worker_id = (
                    self._recorded_planned_worker_id(
                        original_context,
                        recorded_request,
                        prior.start_payload,
                    )
                    if isinstance(recorded_request, SpawnCommandRequest)
                    else None
                )
                reconciled = self._execute(
                    original_context,
                    recorded_request,
                    planned_worker_id=planned_worker_id,
                    planned_message_id=(
                        self._recorded_planned_effect_id(
                            original_context,
                            recorded_request,
                            prior.start_payload,
                        )
                        if isinstance(recorded_request, MessageCommandRequest)
                        else None
                    ),
                    planned_watcher_id=(
                        self._recorded_planned_effect_id(
                            original_context,
                            recorded_request,
                            prior.start_payload,
                        )
                        if isinstance(recorded_request, WatchGraphCommandRequest)
                        else None
                    ),
                    planned_apply_id=(
                        self._recorded_planned_effect_id(
                            original_context,
                            recorded_request,
                            prior.start_payload,
                        )
                        if isinstance(recorded_request, ApplyCommandRequest)
                        else None
                    ),
                    planned_resume_operation_id=(
                        self._recorded_planned_effect_id(
                            original_context,
                            recorded_request,
                            prior.start_payload,
                        )
                        if isinstance(recorded_request, ResumeCommandRequest)
                        else None
                    ),
                    planned_episode_effect_id=(
                        self._recorded_planned_effect_id(
                            original_context,
                            recorded_request,
                            prior.start_payload,
                        )
                        if isinstance(recorded_request, EpisodeCommandRequest)
                        else None
                    ),
                    planned_inbox_effect_id=(
                        self._recorded_planned_effect_id(
                            original_context,
                            recorded_request,
                            prior.start_payload,
                        )
                        if isinstance(recorded_request, InboxCommandRequest)
                        else None
                    ),
                    planned_finish_effect_id=(
                        self._recorded_planned_effect_id(
                            original_context,
                            recorded_request,
                            prior.start_payload,
                        )
                        if isinstance(recorded_request, FinishCommandRequest)
                        else None
                    ),
                )
            outcome = reconciled
        except AutoResearchCommandInvalid as exc:
            outcome = AutoResearchCommandEffectResult(status="invalid", message=str(exc))
        except (AutoResearchCommandUnavailable, OSError, KeyError, ValueError) as exc:
            outcome = AutoResearchCommandEffectResult(status="unavailable", message=str(exc))

        if original_was_unknown:
            outcome = self._finish_original_unknown(prior.command_id, outcome)
        return self._finish(attempt.command_id, request.request_id, outcome)

    @staticmethod
    def _may_reexecute_after_unproven_outcome(request: CommandRequest) -> bool:
        """Name effects whose deterministic identity or monotonicity makes retry safe."""

        return isinstance(
            request,
            (
                SpawnCommandRequest,
                ApplyCommandRequest,
                PauseCommandRequest,
                ResumeCommandRequest,
                StopCommandRequest,
                FinishCommandRequest,
                MessageCommandRequest,
                WatchGraphCommandRequest,
                InboxCommandRequest,
            ),
        ) or (
            isinstance(request, EpisodeCommandRequest)
            and isinstance(
                request.arguments,
                (ExperimentKickoffArguments, EpisodeControlArguments),
            )
        )

    def _recorded_command_file(
        self,
        prior: AgentCommandInvocationRecord,
        request: CommandRequest,
    ) -> AutoResearchCommandFile | None:
        requires_file = isinstance(request, (ApplyCommandRequest, SpawnCommandRequest)) or (
            isinstance(request, EpisodeCommandRequest)
            and isinstance(request.arguments, ExperimentKickoffArguments)
            and request.arguments.goal_file is not None
        )
        stored = self.store.auto_research_command_file(prior.command_id)
        if stored is None:
            if requires_file:
                raise AutoResearchCommandUnavailable(
                    "The original command lost its immutable file snapshot."
                )
            return None
        metadata = prior.start_payload.get("command_file")
        if not isinstance(metadata, dict) or (
            metadata.get("filename") != stored.filename
            or metadata.get("sha256") != stored.sha256
            or metadata.get("byte_length") != len(stored.content.encode("utf-8"))
        ):
            raise AutoResearchCommandUnavailable(
                "The original command file snapshot does not match its audit record."
            )
        return AutoResearchCommandFile(
            kind=stored.kind,
            filename=stored.filename,
            text=stored.content,
            sha256=stored.sha256,
        )

    @staticmethod
    def _recorded_request(
        request: CommandRequest,
        prior: AgentCommandInvocationRecord,
    ) -> CommandRequest:
        if prior.verb != request.verb:
            raise AutoResearchCommandInvalid(
                "This idempotency key was already used for another command verb."
            )
        recorded_arguments = prior.start_payload.get("arguments")
        current_arguments = request.arguments.model_dump(mode="json")
        if recorded_arguments != current_arguments:
            raise AutoResearchCommandInvalid(
                "This idempotency key was already used with different command arguments."
            )
        if not isinstance(recorded_arguments, dict):
            raise AutoResearchCommandUnavailable(
                "The original command has no valid recorded arguments to reconcile."
            )
        try:
            arguments = type(request.arguments).model_validate(recorded_arguments)
        except ValueError as exc:
            raise AutoResearchCommandUnavailable(
                "The original command's recorded arguments are invalid."
            ) from exc
        return request.model_copy(
            update={
                "arguments": arguments,
                "idempotency_key": prior.idempotency_key,
            }
        )

    def _completed_retry_outcome(
        self,
        context: AutoResearchCommandContext,
        request: CommandRequest,
        prior: AgentCommandInvocationRecord,
    ) -> AutoResearchCommandEffectResult:
        self._recorded_planned_effect_id(context, request, prior.start_payload)
        recorded = _effect_from_recorded_invocation(prior)
        if not isinstance(request, SpawnCommandRequest):
            return recorded
        if recorded.status != "ok":
            return recorded
        planned_worker_id = self._recorded_planned_worker_id(
            context,
            request,
            prior.start_payload,
        )
        context = replace(
            context,
            command_file=self._recorded_command_file(prior, request),
        )
        worker = self.store.agent_task(planned_worker_id)
        if worker is not None:
            worker = self._verify_spawn_worker(context, request.arguments, planned_worker_id)
            return AutoResearchCommandEffectResult(
                message="The existing Auto-research worker was returned for this Spawn key.",
                result=_worker_command_result(worker, disposition="existing"),
            )
        raise AutoResearchCommandUnavailable(
            "Completed spawn has no durable canonical worker to return."
        )

    def _finish_original_unknown(
        self,
        command_id: str,
        outcome: AutoResearchCommandEffectResult,
    ) -> AutoResearchCommandEffectResult:
        try:
            self._record_finish(command_id, outcome)
            return outcome
        except ValueError:
            # A concurrent retry may have resolved the original first. Its
            # durable exit is authoritative for this retry too.
            recorded = self.store.agent_command(command_id)
            if recorded is None or recorded.exited_at is None:
                raise
            return _effect_from_recorded_invocation(recorded)

    def _unused_command_id(self, preferred: str) -> str:
        if self.store.agent_command(preferred) is None:
            return preferred
        while True:
            candidate = uuid.uuid4().hex
            if self.store.agent_command(candidate) is None:
                return candidate

    def _context(self, operation_id: str) -> AutoResearchCommandContext:
        task = self.store.agent_task(operation_id)
        if task is None:
            raise KeyError(operation_id)
        if task.kind != "auto_research" or task.episode_id is None:
            raise AutoResearchCommandInvalid("agent command requires an Auto-research task")
        episode = self.store.episode(task.episode_id)
        if episode is None or episode.mode != "auto_research":
            raise KeyError(task.episode_id)
        request = AutoResearchRunRequest.model_validate(task.request)
        return AutoResearchCommandContext(
            episode=episode,
            task=task,
            request=request,
            consume_command_file=self.command_file_consumer,
            refresh_command_state=self.command_state_refresher,
        )

    def _canonical_command_actor(
        self,
        context: AutoResearchCommandContext,
    ) -> tuple[str, str]:
        binding = self.store.auto_research_actor_binding(context.task.operation_id)
        role = self.store.auto_research_invocation_role(context.task.operation_id)
        if role is None or role != binding.role or binding.episode_id != context.episode.episode_id:
            raise AutoResearchCommandUnavailable(
                "The Auto-research command task has no coherent canonical actor role."
            )
        return binding.actor_operation_id, role

    def _reconcile_unknown(
        self,
        context: AutoResearchCommandContext,
        request: CommandRequest,
        start_payload: dict[str, object],
    ) -> AutoResearchCommandEffectResult | None:
        if isinstance(request, SpawnCommandRequest):
            planned_worker_id = self._recorded_planned_worker_id(
                context,
                request,
                start_payload,
            )
            worker = self.store.agent_task(planned_worker_id)
            if worker is None:
                # The durable task row is the spawn commit point. Its absence means
                # the earlier attempt did not create a worker, so the same planned
                # id may be attempted; an existing row is never restarted.
                return None
            worker = self._verify_spawn_worker(context, request.arguments, planned_worker_id)
            return AutoResearchCommandEffectResult(
                message="The existing Auto-research worker was recovered after interrupted Spawn.",
                result=_worker_command_result(worker, disposition="existing"),
            )
        planned_effect_id = self._recorded_planned_effect_id(
            context,
            request,
            start_payload,
        )
        return self.effects.reconcile_unknown(context, request, planned_effect_id)

    @staticmethod
    def _recorded_planned_worker_id(
        context: AutoResearchCommandContext,
        request: SpawnCommandRequest,
        start_payload: dict[str, object],
    ) -> str:
        planned_worker_id = start_payload.get("planned_worker_id")
        expected_worker_id = _planned_worker_id(
            context.episode.episode_id,
            request.idempotency_key,
        )
        if planned_worker_id != expected_worker_id:
            raise AutoResearchCommandUnavailable(
                "Interrupted spawn has no valid deterministic worker id to reconcile."
            )
        return expected_worker_id

    @staticmethod
    def _recorded_planned_effect_id(
        context: AutoResearchCommandContext,
        request: CommandRequest,
        start_payload: dict[str, object],
    ) -> str | None:
        if isinstance(request, MessageCommandRequest):
            field = "planned_message_id"
            verb: Literal[
                "apply",
                "message",
                "watch_graph",
                "episode",
                "inbox",
                "resume",
                "finish",
            ] = "message"
        elif isinstance(request, WatchGraphCommandRequest):
            field = "planned_watcher_id"
            verb = "watch_graph"
        elif isinstance(request, ApplyCommandRequest):
            field = "planned_apply_id"
            verb = "apply"
        elif isinstance(request, ResumeCommandRequest):
            field = "planned_resume_operation_id"
            verb = "resume"
        elif isinstance(request, EpisodeCommandRequest):
            field = "planned_episode_effect_id"
            verb = "episode"
        elif isinstance(request, InboxCommandRequest):
            field = "planned_inbox_effect_id"
            verb = "inbox"
        elif isinstance(request, FinishCommandRequest):
            field = "planned_finish_effect_id"
            verb = "finish"
        else:
            return None
        planned_effect_id = start_payload.get(field)
        expected_effect_id = _planned_effect_id(
            context.episode.episode_id,
            verb,
            request.idempotency_key,
        )
        if planned_effect_id != expected_effect_id:
            raise AutoResearchCommandUnavailable(
                f"Interrupted {verb} has no valid deterministic effect id to reconcile."
            )
        return expected_effect_id

    def _execute(
        self,
        context: AutoResearchCommandContext,
        request: CommandRequest,
        *,
        planned_worker_id: str | None,
        planned_message_id: str | None,
        planned_watcher_id: str | None,
        planned_apply_id: str | None,
        planned_resume_operation_id: str | None,
        planned_episode_effect_id: str | None,
        planned_inbox_effect_id: str | None,
        planned_finish_effect_id: str | None,
    ) -> AutoResearchCommandEffectResult:
        retrospective_worker_reply = request.verb == "message" and context.request.role == "worker"
        if request.verb in MUTATING_COMMAND_VERBS and not retrospective_worker_reply:
            episode = self.store.episode(context.episode.episode_id)
            if episode is None or episode.mode != "auto_research":
                raise AutoResearchCommandUnavailable(
                    "The Auto-research episode is no longer available."
                )
            if (
                episode.status != "running"
                or episode.ending is not None
                or episode.stop_requested_at is not None
            ):
                raise AutoResearchCommandUnavailable(
                    "The Auto-research episode is no longer accepting mutating commands."
                )
        if request.verb == "validate":
            return self.effects.validate(context, request.arguments)
        if request.verb == "status":
            return self.effects.status(context, request.arguments)
        if request.verb == "message":
            assert planned_message_id is not None
            recipient_task_id = request.arguments.recipient_task_id
            if context.request.role == "worker":
                if recipient_task_id not in {None, context.episode.root_operation_id}:
                    raise AutoResearchCommandInvalid(
                        "An Auto-research worker may reply only to its orchestrator."
                    )
                return self.effects.message(context, request.arguments, planned_message_id)
            if context.request.role == "orchestrator":
                if recipient_task_id is None:
                    raise AutoResearchCommandInvalid(
                        "The Auto-research orchestrator must name the worker it is messaging."
                    )
                worker = self._require_worker(context, recipient_task_id)
                if (
                    self.effects.worker_lookup is None
                    and recipient_task_id
                    != self.store.auto_research_actor_binding(
                        worker.operation_id
                    ).actor_operation_id
                ):
                    raise AutoResearchCommandInvalid(
                        "The Auto-research orchestrator must address a worker by its stable worker ID."
                    )
                return self.effects.message(context, request.arguments, planned_message_id)
            raise AutoResearchCommandInvalid("Only an Auto-research actor can send messages.")
        if context.request.role != "orchestrator":
            raise AutoResearchCommandInvalid(
                "Only the Auto-research orchestrator may issue mutating staged commands."
            )
        if request.verb == "apply":
            assert isinstance(request, ApplyCommandRequest)
            assert planned_apply_id is not None
            return self.effects.apply(context, request.arguments, planned_apply_id)
        if request.verb == "spawn":
            assert planned_worker_id is not None
            node_type = self.effects.seat_node_type(
                context.task.project_id,
                request.arguments.seat_node_id,
            )
            if node_type is None or node_type.casefold() not in {"experiment", "blocker"}:
                raise AutoResearchCommandInvalid(
                    "Auto-research workers may be seated only on Experiments and Blockers."
                )
            outcome = self.effects.spawn(context, request.arguments, planned_worker_id)
            if outcome.status != "ok":
                return outcome
            worker = self._verify_spawn_worker(context, request.arguments, planned_worker_id)
            if not outcome.result:
                outcome = outcome.model_copy(
                    update={"result": _worker_command_result(worker, disposition="created")}
                )
            return outcome
        if request.verb == "pause":
            assert isinstance(request, PauseCommandRequest)
            self._require_worker(context, request.arguments.worker_id)
            return self.effects.pause(context, request.arguments.worker_id)
        if request.verb == "resume":
            assert isinstance(request, ResumeCommandRequest)
            assert planned_resume_operation_id is not None
            self._require_worker(context, request.arguments.worker_id)
            return self.effects.resume(
                context,
                request.arguments.worker_id,
                planned_resume_operation_id,
            )
        if request.verb == "stop":
            assert isinstance(request, StopCommandRequest)
            self._require_worker(context, request.arguments.worker_id)
            return self.effects.stop(context, request.arguments.worker_id)
        if request.verb == "watch_graph":
            assert planned_watcher_id is not None
            return self.effects.watch_graph(context, request.arguments, planned_watcher_id)
        if request.verb == "episode":
            assert isinstance(request, EpisodeCommandRequest)
            assert planned_episode_effect_id is not None
            return self.effects.episode(
                context,
                request.arguments,
                planned_episode_effect_id,
            )
        if request.verb == "inbox":
            assert isinstance(request, InboxCommandRequest)
            assert planned_inbox_effect_id is not None
            return self.effects.inbox(context, request.arguments, planned_inbox_effect_id)
        if request.verb == "finish":
            assert isinstance(request, FinishCommandRequest)
            assert planned_finish_effect_id is not None
            return self.effects.finish(context, planned_finish_effect_id)
        raise AssertionError(f"unhandled auto_research command verb: {request.verb}")

    def _require_worker(
        self,
        context: AutoResearchCommandContext,
        operation_id: str,
    ) -> AgentTaskRecord:
        if self.effects.worker_lookup is not None:
            return self.effects.worker_lookup(context, operation_id)
        worker = self.store.agent_task(operation_id)
        if worker is None or worker.episode_id != context.episode.episode_id:
            raise AutoResearchCommandInvalid(
                "The worker control target is outside this Auto-research episode."
            )
        worker_request = AutoResearchRunRequest.model_validate(worker.request)
        if worker_request.role != "worker":
            raise AutoResearchCommandInvalid(
                "The worker control target is not an Auto-research worker."
            )
        return worker

    def _verify_spawn_worker(
        self,
        context: AutoResearchCommandContext,
        arguments: SpawnArguments,
        planned_worker_id: str,
    ) -> AgentTaskRecord:
        """Mechanically prove the canonical worker row matches the spawn intent."""

        if self.effects.verify_spawn is not None:
            return self.effects.verify_spawn(context, arguments, planned_worker_id)

        worker = self.store.agent_task(planned_worker_id)
        if worker is None:
            raise AutoResearchCommandUnavailable(
                "Auto-research Spawn returned without durably creating its planned worker."
            )
        if (
            worker.kind != "auto_research"
            or worker.project_id != context.episode.project_id
            or worker.episode_id != context.episode.episode_id
            or worker.parent_operation_id != context.task.operation_id
        ):
            raise AutoResearchCommandUnavailable(
                "Auto-research Spawn created a worker with incorrect parent lineage."
            )
        if self.store.auto_research_invocation_role(worker.operation_id) != "worker":
            raise AutoResearchCommandUnavailable(
                "Auto-research Spawn did not record the canonical worker invocation role."
            )
        try:
            worker_request = AutoResearchRunRequest.model_validate(worker.request)
        except ValueError as exc:
            raise AutoResearchCommandUnavailable(
                "Auto-research Spawn created a worker with an invalid run request."
            ) from exc
        if (
            worker_request.episode_id != context.episode.episode_id
            or worker_request.role != "worker"
            or worker_request.control_node_id != arguments.seat_node_id
        ):
            raise AutoResearchCommandUnavailable(
                "Auto-research Spawn created a worker that does not match its recorded seat."
            )
        return worker

    def _finish(
        self,
        command_id: str,
        response_request_id: str,
        outcome: AutoResearchCommandEffectResult,
    ) -> CommandResponse:
        outward_result = self._outward_command_result(command_id, outcome)
        self._record_finish(command_id, outcome)
        return CommandResponse(
            request_id=response_request_id,
            status=outcome.status,
            message=outcome.message,
            result=outward_result,
        )

    def _outward_command_result(
        self,
        command_id: str,
        outcome: AutoResearchCommandEffectResult,
    ) -> dict[str, object]:
        """Hydrate unbounded protocol output without copying it into the event ledger."""

        receipt_id = outcome.result.get("finish_receipt_id")
        if receipt_id is None:
            return outcome.result
        if not isinstance(receipt_id, str):
            raise AutoResearchCommandUnavailable("The guarded-Finish receipt identity is invalid.")
        receipt = self.store.auto_research_finish_receipt(receipt_id)
        invocation = self.store.agent_command(command_id)
        if receipt is None or invocation is None or invocation.episode_id is None:
            raise AutoResearchCommandUnavailable(
                "The guarded-Finish receipt is unavailable for this command."
            )
        context = self._context(invocation.operation_id)
        actor_operation_id, actor_role = self._canonical_command_actor(context)
        expected_compact = {
            "finish_receipt_id": receipt.effect_id,
            "disposition": receipt.disposition,
            "blocker_count": receipt.blocker_count,
            "digest": receipt.result_sha256,
        }
        if (
            actor_role != "orchestrator"
            or receipt.episode_id != invocation.episode_id
            or receipt.actor_operation_id != actor_operation_id
            or outcome.result != expected_compact
        ):
            raise AutoResearchCommandUnavailable(
                "The guarded-Finish receipt does not match this canonical command actor."
            )
        return receipt.result

    def _record_finish(
        self,
        command_id: str,
        outcome: AutoResearchCommandEffectResult,
    ) -> None:
        payload: dict[str, object] = {"result": outcome.result}
        if outcome.message:
            payload["diagnostic"] = outcome.message
        self.store.finish_agent_command(
            command_id,
            status=outcome.status,
            payload=payload,
            message=outcome.message or f"Agent command completed with {outcome.status}.",
        )


def _planned_worker_id(episode_id: str, idempotency_key: str | None) -> str:
    if idempotency_key is None:
        raise ValueError("a spawn command requires an idempotency key")
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode_id}:spawn:{idempotency_key}",
        )
    )


def _apply_admission_limit_outcome() -> AutoResearchCommandEffectResult:
    return AutoResearchCommandEffectResult(
        status="invalid",
        message=(
            "This provider turn already reached its "
            f"{AUTO_RESEARCH_APPLY_MAX_PER_TURN}-Apply limit; "
            "use the refreshed result or finish this turn before applying again."
        ),
    )


def _planned_effect_id(
    episode_id: str,
    verb: Literal[
        "apply",
        "message",
        "watch_graph",
        "episode",
        "inbox",
        "resume",
        "finish",
    ],
    idempotency_key: str | None,
) -> str:
    if idempotency_key is None:
        raise ValueError(f"a {verb} command requires an idempotency key")
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rcp:auto_research:{episode_id}:{verb}:{idempotency_key}",
        )
    )


def _worker_command_result(
    worker: AgentTaskRecord,
    *,
    disposition: Literal["created", "existing"],
) -> dict[str, object]:
    return {
        "worker_id": worker.operation_id,
        "status": worker.status,
        "disposition": disposition,
    }


def _effect_from_recorded_invocation(
    invocation: AgentCommandInvocationRecord,
) -> AutoResearchCommandEffectResult:
    status = invocation.status
    payload = invocation.exit_payload
    if status not in {"ok", "invalid", "unavailable"} or not isinstance(payload, dict):
        raise AutoResearchCommandUnavailable(
            "The recorded Auto-research command exit is incomplete."
        )
    recorded_result = payload.get("result")
    result = (
        dict(recorded_result)
        if isinstance(recorded_result, dict)
        else {key: value for key, value in payload.items() if key not in {"status", "diagnostic"}}
    )
    diagnostic = payload.get("diagnostic")
    message = diagnostic if isinstance(diagnostic, str) else None
    if status != "ok" and message is None:
        message = "The recorded Auto-research command did not complete successfully."
    return AutoResearchCommandEffectResult(
        status=status,
        message=message,
        result=result,
    )
