from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast, get_args

from rcp.agents import AgentEvent, AgentProcessControl
from rcp.agents.write_scope import ProjectWriteScope
from rcp.artifacts import AgentArtifactDescriptor
from rcp.core.authority import require_dispatch
from rcp.core.models import AuthorizedHuman, GraphState
from rcp.core.transition_models import GraphTargetRef
from rcp.limits import CHAT_ARTIFACT_MAX_COUNT, GRAPH_UPDATE_HISTORY_MAX_COUNT
from rcp.providers import classify_terminal_error, require_runtime_id
from rcp.runs.auto_research import (
    AutoResearchRunRequest,
    AutoResearchWakeAdmission,
    PendingAutoResearchMail,
)
from rcp.runs.auto_research_admission import (
    auto_research_admission_exhausted,
    auto_research_for_request,
    ensure_auto_research_wake_spawned,
    proven_committed_auto_research_dispatches,
    proven_reserved_auto_research_roots,
    retry_auto_research_task,
)
from rcp.runs.auto_research_mail import auto_research_mail_claim_prefix
from rcp.runs.auto_research_recovery import (
    AutoResearchOrchestratorTerminalFailure,
    record_structural_failure,
)
from rcp.runs.branch_merge_request import BranchMergeRunRequest
from rcp.runs.episodes.report import restart_interrupted_episode_reports
from rcp.runs.experiment_recovery import (
    preflight_experiment_episode_recovery,
    record_bound_experiment_session_limit,
    restart_stopping_experiment_recoveries,
    retry_experiment_loop,
)
from rcp.runs.task_policy import (
    AgentTaskContinuation,
    AgentTaskRequest,
    DispatchAuthorityResolver,
    load_stored_request,
    resolved_dispatch_authority,
    skill_update,
    task_graph_capable,
)
from rcp.runs.tasks.episode_report import EpisodeReportRunRequest
from rcp.service import (
    CoachRequest,
    GraphUpdateResult,
    RunRequest,
    resolve_dispatch_authority,
)
from rcp.skill_registry import SkillSelection
from rcp.storage import (
    AgentTaskKind,
    AgentTaskRecord,
    AppStore,
    EpisodeInvocationCeilingReached,
    EpisodeRecord,
)
from rcp.transport import RemoteRunStage

_AGENT_TASK_CONTINUATIONS = frozenset(get_args(AgentTaskContinuation))

# A watcher wake reuses a native session without being task Resume: it is a new
# task at the next invocation, so it must never inherit Resume's same-invocation
# parent/child recovery semantics.
_NATIVE_CHECKPOINT_CONTINUATIONS = frozenset(
    {
        "resume",
        "retry",
        "graph_repair",
        "watcher_wake",
        "graph_condition_wake",
        "message_wake",
        "lifecycle_wake",
        "auto_research_continuation",
        "episode_report",
    }
)


class StartupEffectBlocked(RuntimeError):
    """A candidate tried to cross the closed startup-effect boundary."""


class StartupEffectFence:
    """One shared gate for startup verification and later update cutover.

    The fence is deliberately an in-process object rather than an environment
    flag. Callers hold the exact object that records an attempted effect, and
    releasing it changes the same gate consulted by the background engine.
    Candidate rehearsal never releases it.
    """

    def __init__(self, reason: str) -> None:
        if not reason or reason != reason.strip():
            raise ValueError("startup-effect fence reason must be one nonempty line")
        self.reason = reason
        self._active = True
        self._attempts: list[str] = []
        self._release_callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def attempted_effects(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._attempts)

    def require_open(self, effect: str) -> None:
        if not effect or effect != effect.strip():
            raise ValueError("startup effect name must be one nonempty line")
        with self._lock:
            if not self._active:
                return
            self._attempts.append(effect)
        raise StartupEffectBlocked(
            f"Startup effects are closed for {self.reason}; blocked {effect}."
        )

    def release(self) -> None:
        with self._lock:
            if self._attempts:
                raise StartupEffectBlocked(
                    "Startup effects cannot open after a blocked effect was attempted."
                )
            if not self._active:
                return
            self._active = False
            callbacks = tuple(self._release_callbacks)
            self._release_callbacks.clear()
        for callback in callbacks:
            callback()

    def on_release(self, callback: Callable[[], None]) -> None:
        """Run one callback when this exact fence opens, including late registration."""

        with self._lock:
            if self._active:
                self._release_callbacks.append(callback)
                return
        callback()


class RuntimeAdmissionGate(Protocol):
    """The installed-service owner may close new launches during maintenance."""

    def require_open(self, effect: str) -> None: ...


@dataclass(frozen=True)
class StartupRecoveryPlan:
    """Read-only inventory of work ordinary startup would reconcile."""

    active_operation_ids: tuple[str, ...]
    stopping_experiment_operation_ids: tuple[str, ...]
    report_episode_ids: tuple[str, ...]
    auto_research_recovery_operation_ids: tuple[str, ...]
    active_watcher_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "active_operation_ids": self.active_operation_ids,
            "stopping_experiment_operation_ids": self.stopping_experiment_operation_ids,
            "report_episode_ids": self.report_episode_ids,
            "auto_research_recovery_operation_ids": (self.auto_research_recovery_operation_ids),
            "active_watcher_ids": self.active_watcher_ids,
        }


@dataclass
class AgentTaskExecution:
    operation_id: str
    store: AppStore
    control: AgentProcessControl
    runtime_id: str = ""
    stage_host: str | None = None
    stage_root: str | None = None
    write_scope_fingerprint: str | None = None
    continuation: AgentTaskContinuation = "fresh"
    retry_feedback: tuple[str, ...] = ()
    applied_revision: int | None = None
    applied_graph_state: GraphState | None = None
    armed_graph_watchers: bool = False

    @property
    def reuses_native_checkpoint(self) -> bool:
        return self.continuation in _NATIVE_CHECKPOINT_CONTINUATIONS

    def checkpoint_stage(self, host: str, root: str) -> None:
        self.stage_host = host or None
        self.stage_root = root
        self.store.checkpoint_agent_task(
            self.operation_id,
            stage_host=host or None,
            stage_root=root,
        )
        self.store.record_agent_task_receipt(
            self.operation_id,
            "stage_checkpoint",
            {"remote": bool(host), "stage_available": bool(root)},
            tier="diagnostic",
        )

    def bind_write_scope(self, scope: ProjectWriteScope) -> None:
        if self.stage_root is None:
            raise ValueError(
                "agent task must checkpoint its exact stage before write-scope binding"
            )
        self.store.bind_agent_task_write_scope(
            self.operation_id,
            project_id=scope.project_id,
            stage_host=self.stage_host or "",
            stage_root=scope.stage_root,
            fingerprint=scope.fingerprint,
        )
        self.write_scope_fingerprint = scope.fingerprint


AgentTaskStream = Callable[
    [str, AgentTaskKind, AgentTaskRequest, AgentTaskExecution], AsyncIterator[str]
]
AgentTaskStreamClosedHook = Callable[
    [str, AgentTaskKind, AgentTaskRequest, AgentTaskExecution], None
]
AgentTaskSettledHook = Callable[[str, AgentTaskKind, AgentTaskRequest, AgentTaskExecution], None]
AutoResearchAdmissionExhaustedHook = Callable[[EpisodeRecord], None]


@dataclass(frozen=True)
class AgentTaskOutcome:
    applied_revision: int | None
    messages: list[str]
    artifacts: list[AgentArtifactDescriptor]
    graph_update: GraphUpdateResult | None = None
    graph_updates: list[GraphUpdateResult] = field(default_factory=list)


class TaskPaused(RuntimeError):
    def __init__(
        self,
        message: str,
        messages: list[str] | None = None,
        artifacts: list[AgentArtifactDescriptor] | None = None,
    ) -> None:
        self.messages = list(messages or [])
        self.artifacts = list(artifacts or [])
        super().__init__(message)


class TaskFailed(RuntimeError):
    """A task that failed after producing output worth keeping.

    A chat turn can answer the human and only then have its graph change rejected.
    The answer is already written and already useful, so it travels with the
    failure instead of being dropped with the stream.
    """

    def __init__(
        self,
        message: str,
        messages: list[str],
        artifacts: list[AgentArtifactDescriptor],
    ) -> None:
        super().__init__(message)
        self.messages = messages
        self.artifacts = artifacts


class BackgroundAgentTasks:
    def __init__(
        self,
        store: AppStore,
        stream: AgentTaskStream,
        on_stream_closed: AgentTaskStreamClosedHook | None = None,
        on_task_settled: AgentTaskSettledHook | None = None,
        on_auto_research_admission_exhausted: AutoResearchAdmissionExhaustedHook | None = None,
        dispatch_authority_resolver: DispatchAuthorityResolver | None = None,
        startup_effect_fence: StartupEffectFence | None = None,
        runtime_admission_gate: RuntimeAdmissionGate | None = None,
    ) -> None:
        self.store = store
        self.stream = stream
        self.on_stream_closed = on_stream_closed
        self.on_task_settled = on_task_settled
        self.on_auto_research_admission_exhausted = on_auto_research_admission_exhausted
        self.dispatch_authority_resolver = dispatch_authority_resolver or resolve_dispatch_authority
        self.startup_effect_fence = startup_effect_fence
        self.runtime_admission_gate = runtime_admission_gate
        self._controls: dict[str, AgentProcessControl] = {}
        self._workers: dict[str, threading.Thread] = {}
        self._controls_lock = threading.Lock()
        self._watcher_delivery_lock = threading.Lock()
        self._accepting_watcher_deliveries = not (
            startup_effect_fence is not None and startup_effect_fence.active
        )

    def plan_startup_recovery(self) -> StartupRecoveryPlan:
        """Describe recovery work without changing a row or resolving a stage."""

        projects = self.store.projects()
        tasks = [
            task
            for project in projects
            for task in self.store.all_project_agent_tasks(project.project_id)
        ]
        watchers = [
            watcher
            for project in projects
            for watcher in self.store.active_graph_watchers(project.project_id)
        ]
        return StartupRecoveryPlan(
            active_operation_ids=tuple(sorted(task.operation_id for task in tasks if task.active)),
            stopping_experiment_operation_ids=tuple(
                sorted(
                    task.operation_id
                    for task in self.store.stopping_experiment_recovery_candidates()
                )
            ),
            report_episode_ids=tuple(
                sorted(episode.episode_id for episode in self.store.episodes_awaiting_report())
            ),
            auto_research_recovery_operation_ids=tuple(
                sorted(task.operation_id for task in self.store.auto_research_recovery_candidates())
            ),
            active_watcher_ids=tuple(sorted(watcher.watcher_id for watcher in watchers)),
        )

    def _require_startup_effects_open(self, effect: str) -> None:
        if self.startup_effect_fence is not None:
            self.startup_effect_fence.require_open(effect)
        if self.runtime_admission_gate is not None:
            self.runtime_admission_gate.require_open(effect)

    def recover_at_startup(self) -> None:
        """Reconcile work a previous process left behind. Called once, by the lifespan.

        This ran in ``__init__`` until constructing the engine stopped writing to
        the store.  It stays on this class as a waypoint: when the job-specific
        owners move out it becomes startup orchestration that calls each owner,
        and the wrong-way calls below leave with them.

        Errors are deliberately not caught.  A constructor exception used to fail
        ``create_app``; this must still fail startup rather than open the app over
        state nobody reconciled.
        """

        self._require_startup_effects_open("startup recovery")
        preserved_dispatches = proven_committed_auto_research_dispatches(self)
        reserved_roots = proven_reserved_auto_research_roots(self)
        self.store.interrupt_active_agent_tasks(
            preserve_operation_ids={
                *[item.operation_id for item in preserved_dispatches],
                *[task.operation_id for _episode, task, _request in reserved_roots],
            }
        )
        restart_stopping_experiment_recoveries(self)
        self.store.settle_ready_experiment_loop_stops()
        restart_interrupted_episode_reports(self)

    def start(
        self,
        project_id: str,
        kind: AgentTaskKind,
        request: AgentTaskRequest,
        *,
        operation_id: str | None = None,
        authorized_by: AuthorizedHuman | None = None,
        stage_host: str | None = None,
        stage_root: str | None = None,
    ) -> AgentTaskRecord:
        self._require_startup_effects_open("provider task dispatch")
        if kind == "auto_research":
            raise ValueError(
                "Use start_auto_research so its episode and root are created atomically."
            )
        if kind == "branch_merge":
            raise ValueError(
                "Use start_branch_merge so its ended episode and exact branch are checked "
                "atomically."
            )
        if kind == "episode_report":
            raise ValueError(
                "Use start_episode_report so the existing hidden allocation is preserved."
            )
        experiment_root = (
            isinstance(request, RunRequest)
            and request.patch_kind == "experiment_loop"
            and request.trigger == "experiment_run"
        )
        if (
            isinstance(request, RunRequest)
            and request.patch_kind == "experiment_loop"
            and not experiment_root
        ):
            raise ValueError(
                "An Experiment watcher wake or recovery must use its dedicated admission path."
            )
        if kind in {"seed", "refresh"} and request.session_id:
            raise ValueError(
                "Seed and refresh sessions can only be resumed from an RCP background "
                "task checkpoint."
            )
        result_view_revision = (
            isinstance(request, RunRequest)
            and request.result_view is not None
            and request.result_view.action == "revise"
        )
        if result_view_revision and (not request.session_id or not stage_root):
            raise ValueError(
                "A result-view revision requires its saved native session and exact stage."
            )
        if not result_view_revision and (stage_host is not None or stage_root is not None):
            raise ValueError("Only a result-view revision may inherit a saved stage on start.")
        self._validate_request_type(kind, request)
        request_data = request.model_dump(mode="json")
        estimate, samples = self.store.agent_task_estimate(project_id, kind, request_data)
        return self._create_and_spawn(
            project_id,
            kind,
            request,
            estimate_seconds=estimate,
            estimate_samples=samples,
            operation_id=operation_id,
            authorized_by=authorized_by,
            stage_host=stage_host,
            stage_root=stage_root,
        )

    def resume(
        self,
        operation_id: str,
        *,
        skills: SkillSelection | None = None,
        authorized_by: AuthorizedHuman | None = None,
    ) -> AgentTaskRecord:
        self._require_startup_effects_open("provider task resume")
        previous = self._require_operation(operation_id)
        if previous.kind == "episode_report":
            raise ValueError("Episode report recovery is automatic and has no Resume control.")
        if not previous.can_resume or not previous.native_session_id:
            raise ValueError(
                "This task has no resumable native agent checkpoint. Retry it instead."
            )
        if not self._session_is_rcp_owned(previous):
            raise ValueError(
                "This task's native session was not checkpointed or validated by RCP. "
                "Retry it instead."
            )
        preflight_experiment_episode_recovery(self, previous)
        request = self._request_from_record(previous).model_copy(
            update={"session_id": previous.native_session_id, **skill_update(skills)}
        )
        continuation: AgentTaskContinuation = (
            "graph_repair"
            if self.store.agent_task_continuation_cause(previous.operation_id) == "graph_repair"
            else "resume"
        )
        return self._create_and_spawn(
            previous.project_id,
            previous.kind,
            request,
            parent=previous,
            continuation=continuation,
            estimate_seconds=previous.estimate_seconds,
            estimate_samples=previous.estimate_samples,
            stage_host=previous.stage_host,
            stage_root=previous.stage_root,
            authorized_by=authorized_by,
        )

    def retry(
        self,
        operation_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning: str | None = None,
        run_on: str | None = None,
        skills: SkillSelection | None = None,
        authorized_by: AuthorizedHuman | None = None,
    ) -> AgentTaskRecord:
        self._require_startup_effects_open("provider task retry")
        previous = self._require_operation(operation_id)
        if previous.kind == "episode_report":
            raise ValueError("Episode report recovery is automatic and has no Retry control.")
        if not previous.can_retry:
            raise ValueError("Only a paused, interrupted, or failed task can be retried.")
        original = self._request_from_record(previous)
        if isinstance(original, AutoResearchRunRequest):
            return retry_auto_research_task(
                self,
                previous,
                original,
                provider=provider,
                model=model,
                reasoning=reasoning,
                run_on=run_on,
                skills=skills,
            )
        preflight_experiment_episode_recovery(self, previous, request=original)
        graph_repair = (
            self.store.agent_task_continuation_cause(previous.operation_id) == "graph_repair"
        )
        if isinstance(original, RunRequest) and original.patch_kind == "experiment_loop":
            if run_on is not None:
                raise ValueError(
                    "Experiment-loop recovery cannot change its pinned execution machine."
                )
            if not graph_repair:
                return retry_experiment_loop(
                    self,
                    previous,
                    original,
                    provider=provider,
                    model=model,
                    reasoning=reasoning,
                    skills=skills,
                    authorized_by=authorized_by,
                )
        updates = {
            key: value
            for key, value in {
                "provider": provider,
                "model": model,
                "reasoning": reasoning,
                "run_on": run_on,
            }.items()
            if value is not None
        }
        request = type(original).model_validate(
            {
                **original.model_dump(mode="json"),
                **updates,
                **skill_update(skills, mode="json"),
                "session_id": None,
            }
        )
        same_provider = request.provider == original.provider
        same_model = request.model == original.model
        same_reasoning = request.reasoning == original.reasoning
        same_execution_host = request.run_on == original.run_on
        session_limit = self._failure_is_session_limit(previous)
        continuation_context_unavailable = self._continuation_context_is_unavailable(previous)
        owned_checkpoint = (
            bool(previous.native_session_id)
            and bool(previous.stage_root)
            and self._session_is_rcp_owned(previous)
        )
        result_view_revision = bool(
            isinstance(original, RunRequest)
            and original.result_view is not None
            and original.result_view.action == "revise"
        )
        must_reuse_saved_session = graph_repair or result_view_revision
        if must_reuse_saved_session:
            problem = None
            stage_available: bool | None = True
            if owned_checkpoint and previous.stage_host:
                stage_available = RemoteRunStage(previous.stage_host).directory_exists(
                    previous.stage_root or ""
                )
            elif owned_checkpoint and previous.stage_root:
                stage = Path(previous.stage_root)
                stage_available = stage.is_dir() and not stage.is_symlink()
            if session_limit:
                problem = "the native provider session reached its limit"
            elif continuation_context_unavailable:
                problem = "the saved continuation context is unavailable"
            elif result_view_revision and (
                not same_provider or not same_model or not same_reasoning or not same_execution_host
            ):
                problem = "the pinned provider, model, reasoning, or execution machine changed"
            elif not same_provider or not same_execution_host:
                problem = "the pinned provider or execution machine changed"
            elif not owned_checkpoint:
                problem = "the prior task has no complete RCP-owned session and stage"
            elif stage_available is not True:
                problem = "the saved provider workspace is unavailable"
            if problem is not None:
                detail = (
                    "This result-view revision cannot start a fresh provider session because "
                    f"{problem}. The existing view was not redrawn; start a new result view "
                    "instead."
                    if result_view_revision
                    else "This patch-only graph repair cannot start a full Work turn because "
                    f"{problem}. Start a new Work turn instead."
                )
                raise ValueError(detail)
            assert previous.native_session_id is not None
            request = request.model_copy(update={"session_id": previous.native_session_id})
            return self._create_and_spawn(
                previous.project_id,
                previous.kind,
                request,
                parent=previous,
                continuation="graph_repair" if graph_repair else "retry",
                estimate_seconds=previous.estimate_seconds,
                estimate_samples=previous.estimate_samples,
                stage_host=previous.stage_host,
                stage_root=previous.stage_root,
                authorized_by=authorized_by,
            )
        retry_same_provider = (
            previous.status == "failed"
            and same_provider
            and same_execution_host
            and owned_checkpoint
            and not session_limit
            and not continuation_context_unavailable
        )
        if retry_same_provider:
            request = request.model_copy(update={"session_id": previous.native_session_id})
            return self._create_and_spawn(
                previous.project_id,
                previous.kind,
                request,
                parent=previous,
                continuation="retry",
                estimate_seconds=previous.estimate_seconds,
                estimate_samples=previous.estimate_samples,
                stage_host=previous.stage_host,
                stage_root=previous.stage_root,
                authorized_by=authorized_by,
            )
        estimate, samples = self.store.agent_task_estimate(
            previous.project_id,
            previous.kind,
            request.model_dump(mode="json"),
        )
        retried = self._create_and_spawn(
            previous.project_id,
            previous.kind,
            request,
            parent=previous,
            continuation="handoff",
            estimate_seconds=estimate,
            estimate_samples=samples,
            authorized_by=authorized_by,
        )
        if same_provider and session_limit:
            self.store.record_agent_task_receipt(
                retried.operation_id,
                "native_resume_skipped",
                {"classification": "session_limit"},
                tier="diagnostic",
            )
            self.store.record_agent_task_event(
                retried.operation_id,
                "The provider session limit was exhausted; starting a clean retry.",
                level="warning",
            )
        elif previous.status == "failed" and same_provider and not retry_same_provider:
            reason = (
                "execution host changed"
                if not same_execution_host
                else "the saved continuation context is unavailable"
                if continuation_context_unavailable
                else "the prior task has no complete RCP-owned native checkpoint and stage"
            )
            self.store.record_agent_task_receipt(
                retried.operation_id,
                "native_resume_unavailable",
                {"reason": reason},
                tier="diagnostic",
            )
            self.store.record_agent_task_event(
                retried.operation_id,
                f"Native resume is unavailable because {reason}; starting a clean retry.",
                level="warning",
            )
        return retried

    def repair_graph_update(
        self,
        operation_id: str,
        *,
        authorized_by: AuthorizedHuman | None = None,
    ) -> AgentTaskRecord:
        """Create one idempotent patch-only continuation for a rejected Work result."""

        self._require_startup_effects_open("provider graph-repair dispatch")
        previous = self._require_operation(operation_id)
        if previous.kind not in {"node_chat", "project_chat"}:
            raise ValueError("Only a conversation Work task can repair a graph update.")
        request = self._request_from_record(previous)
        if not isinstance(request, RunRequest) or request.mode != "work":
            raise ValueError("Only a Work turn can repair a graph update.")
        if (
            not previous.native_session_id
            or not previous.stage_root
            or not self._session_is_rcp_owned(previous)
        ):
            raise ValueError(
                "The rejected graph update has no retained RCP-owned session and stage. "
                "Start a new Work turn instead."
            )
        request = request.model_copy(
            update={"session_id": previous.native_session_id, "message": None}
        )
        return self._create_and_spawn(
            previous.project_id,
            previous.kind,
            request,
            parent=previous,
            continuation="graph_repair",
            claim_graph_repair_parent=True,
            estimate_seconds=previous.estimate_seconds,
            estimate_samples=previous.estimate_samples,
            stage_host=previous.stage_host,
            stage_root=previous.stage_root,
            authorized_by=authorized_by,
        )

    def pause(self, operation_id: str) -> AgentTaskRecord:
        self._require_startup_effects_open("provider task pause")
        current = self._require_operation(operation_id)
        if current.kind == "episode_report":
            raise ValueError("Episode report generation has no manual Pause control.")
        record = self.store.request_agent_task_pause(operation_id)
        self._signal_agent_task_pause(operation_id)
        return record

    def request_member_removal_pause(self, operation_id: str) -> AgentTaskRecord:
        """Fence a task without terminating an already-running provider turn."""

        current = self._require_operation(operation_id)
        if not current.active:
            return current
        record = self.store.request_agent_task_pause(
            operation_id,
            requested_by="member_removal",
        )
        # A durably queued row normally owns a worker that will observe
        # ``pausing`` before provider dispatch. A crash fixture or recovered
        # admission may have no worker; terminalize only that unowned row.
        with self._controls_lock:
            worker = self._workers.get(operation_id)
        if worker is None or not worker.is_alive():
            self.store.pause_agent_task(
                operation_id,
                detail="Paused because the authorizing member was removed.",
            )
            settled = self.store.agent_task(operation_id)
            assert settled is not None
            return settled
        return record

    def _signal_agent_task_pause(self, operation_id: str) -> None:
        """Best-effort re-signal of a pause whose durable intent already exists."""

        with self._controls_lock:
            control = self._controls.get(operation_id)
        if control is not None:
            control.request_pause()

    def shutdown(self, *, timeout: float = 7.0) -> None:
        """Pause live subprocesses before the web process exits."""
        with self._watcher_delivery_lock:
            self._accepting_watcher_deliveries = False
            with self._controls_lock:
                active = list(self._controls.items())
                workers = [self._workers.get(operation_id) for operation_id, _ in active]
        for operation_id, control in active:
            with suppress(ValueError):
                self.store.request_agent_task_pause(operation_id, requested_by="shutdown")
            control.request_pause()
        deadline = time.monotonic() + timeout
        for worker in workers:
            if worker is None or worker is threading.current_thread():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            worker.join(remaining)

    def accept_watcher_notifications(self) -> None:
        """Open automatic delivery admission for one app lifespan."""

        self._require_startup_effects_open("watcher delivery admission")
        with self._watcher_delivery_lock:
            self._accepting_watcher_deliveries = True

    def close_watcher_notifications(self) -> None:
        """Close automatic wake admission without stopping durable watchers."""

        with self._watcher_delivery_lock:
            self._accepting_watcher_deliveries = False

    def runtime_is_idle(self) -> bool:
        """Report whether every already-launched provider worker has settled."""

        with self._controls_lock:
            return not self._workers and not self._controls

    def _create_and_spawn(
        self,
        project_id: str,
        kind: AgentTaskKind,
        request: AgentTaskRequest,
        *,
        estimate_seconds: float,
        estimate_samples: int,
        parent: AgentTaskRecord | None = None,
        continuation: AgentTaskContinuation = "fresh",
        stage_host: str | None = None,
        stage_root: str | None = None,
        operation_id: str | None = None,
        authorized_by: AuthorizedHuman | None = None,
        auto_research_mail_delivery: PendingAutoResearchMail | None = None,
        auto_research_wake_admission: AutoResearchWakeAdmission | None = None,
        claim_graph_repair_parent: bool = False,
    ) -> AgentTaskRecord | None:
        """Insert one admitted task row and start it.

        Deliberately not split by surface.  It reads like four surfaces sharing
        one function, but 82% of it is universal row assembly, and no two owners
        have ever collided here — which is the failure a split would prevent.
        The Auto-research branch and its two parameters leave when Auto-research
        admission moves out and inserts its own row; that is a consequence of
        the move, not separate work.  Splitting the rest was measured and
        rejected.
        """

        self._require_startup_effects_open("provider task admission")
        episode: EpisodeRecord | None = None
        task_graph_target = parent.graph_target if parent is not None else GraphTargetRef()
        if isinstance(request, BranchMergeRunRequest):
            raise TypeError("BranchMergeRunRequest requires start_branch_merge.")
        if isinstance(request, AutoResearchRunRequest):
            if kind != "auto_research":
                raise TypeError("AutoResearchRunRequest requires auto_research task kind.")
            episode = auto_research_for_request(self, request.episode_id, request)
            authorized_by = episode.authorized_by
            task_graph_target = episode.graph_target
        elif isinstance(request, EpisodeReportRunRequest):
            raise TypeError("EpisodeReportRunRequest requires an existing hidden allocation.")
        elif isinstance(request, RunRequest) and request.patch_kind == "experiment_loop":
            stored_episode = self.store.episode(request.control_episode_id or "")
            if stored_episode is not None:
                if (
                    stored_episode.mode != "experiment_loop"
                    or stored_episode.project_id != project_id
                    or stored_episode.control_node_id != request.control_node_id
                ):
                    raise ValueError("The Experiment task changed its episode parent scope.")
                if stored_episode.authorized_by is None:
                    # The episode's own human is the authority for every turn in
                    # it, so a current human pressing Resume/Retry cannot stand in.
                    # Episodes created before that snapshot was recorded therefore
                    # have no recoverable authority; a fresh Run does.
                    raise ValueError(
                        "This Experiment-loop turn cannot be resumed or retried because its "
                        "episode predates the recorded human authorizer. Press Run on the "
                        "Experiment to start a fresh episode."
                    )
                authorized_by = stored_episode.authorized_by
                task_graph_target = stored_episode.graph_target
            elif parent is not None or request.trigger != "experiment_run":
                raise ValueError("The Experiment continuation lost its episode parent.")
        elif auto_research_mail_delivery is not None or auto_research_wake_admission is not None:
            raise ValueError("Only Auto-research may use Auto-research wake admission.")
        if task_graph_capable(kind, request) and authorized_by is None:
            raise ValueError("A patch-capable agent task requires a human authorizer snapshot.")
        if authorized_by is None:
            raise ValueError("An ordinary agent task requires a human authorizer snapshot.")
        if authorized_by is not None and not authorized_by.display_name.strip():
            raise ValueError("A human authorizer snapshot must include a nonblank display name.")
        if claim_graph_repair_parent and (parent is None or continuation != "graph_repair"):
            raise ValueError("Only an initial graph-repair admission can claim its parent.")
        operation_id = operation_id or str(uuid.uuid4())
        dispatch_authority = resolved_dispatch_authority(
            self.store,
            self.dispatch_authority_resolver,
            kind,
            request,
            project_id=project_id,
            parent=parent,
            operation_id=operation_id,
            continuation=continuation,
        )
        now = self.store.now()
        verb = (
            "repair its graph update"
            if continuation == "graph_repair"
            else "resume"
            if continuation == "resume"
            else "retry"
            if continuation in {"retry", "handoff"}
            else "start"
        )
        recovery = continuation in {"resume", "retry", "handoff", "graph_repair"}
        task_record = AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            episode_id=(
                request.episode_id
                if isinstance(request, AutoResearchRunRequest)
                else request.control_episode_id
                if isinstance(request, RunRequest) and request.patch_kind == "experiment_loop"
                else None
            ),
            graph_target=task_graph_target,
            kind=kind,
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message=f"Waiting for the background worker to {verb}.",
            attempt=(parent.attempt + 1) if parent and recovery else 1,
            parent_operation_id=parent.operation_id if parent else None,
            native_session_id=request.session_id,
            stage_host=stage_host,
            stage_root=stage_root,
            estimate_seconds=estimate_seconds,
            estimate_samples=estimate_samples,
            phase="queued",
            last_activity_at=now,
            authorized_by=authorized_by,
            dispatch_authority=dispatch_authority,
        )
        if auto_research_mail_delivery is not None:
            selected_messages = auto_research_mail_claim_prefix(
                episode_id=auto_research_mail_delivery.episode_id,
                recipient_task_id=auto_research_mail_delivery.recipient_task_id,
                delivery_operation_id=task_record.operation_id,
                delivered_at=task_record.created_at,
                messages=auto_research_mail_delivery.messages,
            )
            if not selected_messages:
                return None
            auto_research_mail_delivery = auto_research_mail_delivery.model_copy(
                update={"messages": selected_messages}
            )
        if isinstance(request, AutoResearchRunRequest):
            assert episode is not None
            try:
                if continuation in {"resume", "retry"}:
                    record = self.store.create_auto_research_recovery_task(
                        task_record,
                        continuation_cause=continuation,
                    )
                elif auto_research_mail_delivery is not None:
                    record = self.store.create_auto_research_message_wake_task(
                        task_record,
                        role=request.role,
                        recipient_task_id=auto_research_mail_delivery.recipient_task_id,
                        message_ids=auto_research_mail_delivery.message_ids,
                    )
                elif auto_research_wake_admission is not None:
                    assert request.wake_cause is not None
                    record = auto_research_wake_admission(
                        task_record,
                        request.role,
                        request.wake_cause,
                    )
                else:
                    record = self.store.create_auto_research_agent_task(
                        task_record,
                        role=request.role,
                        continuation_cause=continuation,
                    )
            except EpisodeInvocationCeilingReached:
                auto_research_admission_exhausted(self, episode)
                raise
            if record is None:
                if auto_research_wake_admission is None and auto_research_mail_delivery is None:
                    raise RuntimeError(
                        "Auto-research admission returned no task outside a watcher or mail wake"
                    )
                return None
            if (
                record.operation_id != task_record.operation_id
                or record.project_id != episode.project_id
                or record.episode_id != episode.episode_id
            ):
                raise ValueError("Auto-research wake admission returned another task lineage.")
        elif isinstance(request, RunRequest) and request.patch_kind == "experiment_loop":
            if request.trigger == "experiment_run" and parent is None:
                record = self.store.create_experiment_episode_with_invocation(
                    task_record,
                    request.watcher_ids,
                )
            elif parent is not None and continuation in {
                "resume",
                "retry",
                "handoff",
                "graph_repair",
            }:
                if claim_graph_repair_parent:
                    record = self.store.create_experiment_graph_repair_task(
                        parent.operation_id,
                        task_record,
                    )
                else:
                    record = self.store.create_experiment_recovery_task(
                        task_record,
                        continuation_cause=continuation,
                    )
            else:
                raise ValueError("An Experiment watcher wake must use start_watcher_notification.")
        else:
            if claim_graph_repair_parent:
                assert parent is not None
                record = self.store.create_agent_task_graph_repair(
                    parent.operation_id,
                    task_record,
                )
            else:
                record = self.store.create_agent_task(
                    task_record,
                    continuation_cause=continuation,
                )
        if isinstance(request, AutoResearchRunRequest) and request.wake_cause is not None:
            return ensure_auto_research_wake_spawned(
                self,
                request.episode_id,
                operation_id=record.operation_id,
            )
        return self.launch_admitted(record.operation_id)

    def launch_admitted(self, operation_id: str) -> AgentTaskRecord:
        """Launch one task whose durable admission already committed.

        Admission and provider dispatch are separate durability boundaries.  The
        caller therefore supplies only the operation identity: the request,
        continuation cause, parent, and launch bindings all come back from the
        durable row and its admission receipt.  This is also the startup repair
        seam for a task that was admitted before the process disappeared.
        """

        self._require_startup_effects_open("provider task launch")
        record = self._require_operation(operation_id)
        with self._controls_lock:
            if operation_id in self._workers:
                return self._require_operation(operation_id)
        if record.status != "queued":
            return record

        try:
            request = self._request_from_record(record)
            self._validate_request_type(record.kind, request)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The admitted task has no valid persisted request for its kind."
            ) from exc
        if request.model_dump(mode="json") != record.request:
            raise ValueError("The admitted task request failed its persisted roundtrip.")

        intent = self.store.agent_task_admission_intent(operation_id)
        if intent is None:
            raise ValueError("The admitted task has no durable admission intent.")
        cause = intent.get("continuation_cause")
        if not isinstance(cause, str) or cause not in _AGENT_TASK_CONTINUATIONS:
            raise ValueError("The admitted task has no valid continuation cause.")
        intent_parent = intent.get("parent_operation_id")
        if "parent_operation_id" in intent:
            if intent_parent != record.parent_operation_id:
                raise ValueError("The admission intent changed its exact parent operation.")
        elif intent.get("has_parent") != (record.parent_operation_id is not None):
            raise ValueError("The legacy admission intent changed its parent presence.")

        parent = None
        if record.parent_operation_id is not None:
            parent = self.store.agent_task(record.parent_operation_id)
            if parent is None:
                raise ValueError("The admitted task lost its exact persisted parent.")

        continuation = cast(AgentTaskContinuation, cause)
        self._validate_launch_admission(
            record,
            request,
            parent=parent,
        )

        # The proof can race another in-process launcher.  Re-check the registry
        # after a negative proof so a claimed or already-advanced row is treated
        # as an idempotent duplicate rather than as an ambiguous dispatch.
        if not self.store.agent_task_dispatch_was_proven_not_started(operation_id):
            latest = self._require_operation(operation_id)
            with self._controls_lock:
                if operation_id in self._workers:
                    return self._require_operation(operation_id)
            if latest.status != "queued":
                return latest
            raise ValueError(
                "The admitted task has an ambiguous or already-started dispatch attempt."
            )

        return self._spawn_record(
            record,
            request,
            continuation=continuation,
            parent=parent,
        )

    def _validate_launch_admission(
        self,
        record: AgentTaskRecord,
        request: AgentTaskRequest,
        *,
        parent: AgentTaskRecord | None,
    ) -> None:
        """Validate immutable bindings before the first dispatch receipt."""

        if record.authorized_by is None or not record.authorized_by.display_name.strip():
            raise ValueError("The admitted task lost its human authorizer snapshot.")
        if record.native_session_id != request.session_id:
            raise ValueError("The admitted task request and native session do not agree.")
        if request.provider:
            require_runtime_id(request.provider, record.runtime_id)
        elif record.runtime_id:
            raise ValueError("The admitted task has a runtime without a provider.")
        if record.stage_host is not None and record.stage_root is None:
            raise ValueError("The admitted task has an incoherent execution stage binding.")
        if record.stage_root is not None and not record.stage_root.strip():
            raise ValueError("The admitted task has an empty execution stage root.")
        if record.write_scope_fingerprint is not None:
            raise ValueError("A queued admitted task cannot already carry a write-scope binding.")

        if record.kind == "episode_report":
            if record.dispatch_authority is not None:
                raise ValueError("An episode report task cannot carry dispatch authority.")
        else:
            if record.dispatch_authority is None:
                raise ValueError("The admitted task has no dispatch authority.")
            require_dispatch(record.dispatch_authority)

        request_episode_id = (
            request.episode_id
            if isinstance(
                request,
                (AutoResearchRunRequest, BranchMergeRunRequest, EpisodeReportRunRequest),
            )
            else request.control_episode_id
            if isinstance(request, RunRequest) and request.patch_kind == "experiment_loop"
            else None
        )
        if request_episode_id is not None and request_episode_id != record.episode_id:
            raise ValueError("The admitted task request changed its exact episode identity.")
        if record.episode_id is None:
            if record.graph_target.kind != "main":
                raise ValueError("A task without an episode must target the main graph.")
        else:
            episode = self.store.episode(record.episode_id)
            if episode is None:
                raise ValueError("The admitted task lost its exact episode parent.")
            if (
                episode.project_id != record.project_id
                or episode.graph_target != record.graph_target
            ):
                raise ValueError("The admitted task changed its episode project or graph target.")

        expected_parent = record.parent_operation_id
        if (expected_parent is None) != (parent is None):
            raise ValueError("The admitted task changed its parent presence.")
        if parent is not None:
            if expected_parent != parent.operation_id:
                raise ValueError("The admitted task changed its exact parent operation.")
            if parent.project_id != record.project_id or parent.graph_target != record.graph_target:
                raise ValueError("The admitted task changed its parent project or graph target.")

    def _spawn_record(
        self,
        record: AgentTaskRecord,
        request: AgentTaskRequest,
        *,
        continuation: AgentTaskContinuation,
        parent: AgentTaskRecord | None = None,
    ) -> AgentTaskRecord:
        operation_id = record.operation_id
        with self._controls_lock:
            if operation_id in self._workers:
                return self._require_operation(operation_id)
            current = self._validated_spawn_record(record, request, parent=parent)
            if current.status != "queued":
                return current
            control = AgentProcessControl()
            worker = threading.Thread(
                target=self._run,
                args=(current, request, control, continuation),
                name=f"rcp-{current.kind}-{operation_id[:8]}",
                daemon=True,
            )
            # Registry membership is the one in-process dispatch claim.  It is
            # installed before any observable launch work so concurrent command
            # reconciliation cannot create a second thread for the same row.
            self._controls[operation_id] = control
            self._workers[operation_id] = worker
            dispatch_attempt_id = str(uuid.uuid4())
            try:
                self.store.record_agent_task_receipt(
                    operation_id,
                    "operation_dispatch_attempt",
                    {"dispatch_attempt_id": dispatch_attempt_id},
                    tier="diagnostic",
                )
                self._record_spawn_dispatch(
                    current,
                    request,
                    continuation=continuation,
                    parent=parent,
                )
                worker.start()
            except Exception:
                with suppress(Exception):
                    self.store.record_agent_task_receipt(
                        operation_id,
                        "operation_dispatch_failed_before_start",
                        {"dispatch_attempt_id": dispatch_attempt_id},
                        tier="diagnostic",
                    )
                if self._workers.get(operation_id) is worker:
                    self._controls.pop(operation_id, None)
                    self._workers.pop(operation_id, None)
                raise
            with suppress(Exception):
                self.store.record_agent_task_receipt(
                    operation_id,
                    "operation_dispatch_started",
                    {"dispatch_attempt_id": dispatch_attempt_id},
                    tier="diagnostic",
                )
        return current

    def _validated_spawn_record(
        self,
        record: AgentTaskRecord,
        request: AgentTaskRequest,
        *,
        parent: AgentTaskRecord | None,
    ) -> AgentTaskRecord:
        current = self._require_operation(record.operation_id)
        if (
            current.project_id != record.project_id
            or current.episode_id != record.episode_id
            or current.graph_target != record.graph_target
            or current.kind != record.kind
            or current.request != record.request
            or current.request != request.model_dump(mode="json")
            or current.attempt != record.attempt
            or current.parent_operation_id != record.parent_operation_id
            or current.runtime_id != record.runtime_id
            or (parent is None) != (current.parent_operation_id is None)
            or (parent is not None and current.parent_operation_id != parent.operation_id)
            or current.native_session_id != record.native_session_id
            or current.stage_host != record.stage_host
            or current.stage_root != record.stage_root
            or current.write_scope_fingerprint != record.write_scope_fingerprint
            or current.authorized_by != record.authorized_by
            or current.dispatch_authority != record.dispatch_authority
        ):
            raise ValueError("The committed task changed before background dispatch.")
        try:
            current_request = self._request_from_record(current)
            self._validate_request_type(current.kind, current_request)
        except (TypeError, ValueError) as exc:
            raise ValueError("The committed task lost its persisted request contract.") from exc
        if current_request.model_dump(mode="json") != current.request:
            raise ValueError("The committed task request failed its persisted roundtrip.")
        return current

    def _record_spawn_dispatch(
        self,
        record: AgentTaskRecord,
        request: AgentTaskRequest,
        *,
        continuation: AgentTaskContinuation,
        parent: AgentTaskRecord | None,
    ) -> None:
        operation_id = record.operation_id
        reuses_native_checkpoint = continuation in _NATIVE_CHECKPOINT_CONTINUATIONS
        already_recorded = any(
            receipt.category == "operation_created"
            for receipt in self.store.agent_task_receipts(operation_id)
        )
        if already_recorded:
            return
        self.store.record_agent_task_receipt(
            operation_id,
            "operation_created",
            {
                "kind": record.kind,
                "attempt": record.attempt,
                "has_parent": parent is not None,
                "continuation_cause": continuation,
                "resumed": reuses_native_checkpoint,
            },
        )
        if (
            parent
            and isinstance(request, AutoResearchRunRequest)
            and continuation not in {"resume", "retry", "handoff", "graph_repair"}
        ):
            label = {
                "fresh": f"Auto-research {request.role} turn",
                "watcher_wake": "Auto-research watcher wake",
                "graph_condition_wake": "Auto-research graph-condition wake",
                "message_wake": "Auto-research message wake",
                "lifecycle_wake": "Auto-research lifecycle wake",
                "auto_research_continuation": "Auto-research human-authorized continuation",
            }[continuation]
            self.store.record_agent_task_event(
                operation_id,
                f"{label} queued from task {parent.operation_id[:8]}.",
            )
        elif parent and isinstance(request, EpisodeReportRunRequest):
            self.store.record_agent_task_event(
                operation_id,
                "Wrapping up visualization and report",
            )
        elif parent:
            action = (
                "Repairing the graph update from"
                if continuation == "graph_repair"
                else "Resuming"
                if continuation == "resume"
                else "Retrying"
            )
            feedback = "" if continuation == "resume" else " with prior failure diagnostics"
            self.store.record_agent_task_event(
                operation_id,
                f"{action} task {parent.operation_id[:8]} as attempt {record.attempt}{feedback}.",
            )
        elif (
            isinstance(request, RunRequest)
            and bool(request.watcher_ids)
            and request.trigger in {"watcher", "experiment_run"}
        ):
            self.store.record_agent_task_receipt(
                operation_id,
                "watcher_notification",
                {"watcher_ids": request.watcher_ids},
            )
            event = (
                "Pending watcher completion reauthorized by human Run."
                if request.trigger == "experiment_run"
                else "Watcher completion queued."
            )
            self.store.record_agent_task_event(operation_id, event)
        else:
            self.store.record_agent_task_event(operation_id, "Agent task queued.")

    def _run(
        self,
        record: AgentTaskRecord,
        request: AgentTaskRequest,
        control: AgentProcessControl,
        continuation: AgentTaskContinuation,
    ) -> None:
        operation_id = record.operation_id
        current = self.store.agent_task(operation_id)
        if current is None:
            self._forget_control(operation_id)
            return
        if current.status == "pausing" or control.pause_requested.is_set():
            self.store.pause_agent_task(operation_id)
            execution = AgentTaskExecution(
                operation_id=operation_id,
                store=self.store,
                control=control,
                runtime_id=record.runtime_id,
                stage_host=record.stage_host,
                stage_root=record.stage_root,
                write_scope_fingerprint=record.write_scope_fingerprint,
                continuation=continuation,
            )
            try:
                self._task_settled(record, request, execution)
            finally:
                self._forget_control(operation_id)
            return
        self.store.mark_agent_task_running(operation_id)
        execution = AgentTaskExecution(
            operation_id=operation_id,
            store=self.store,
            control=control,
            runtime_id=record.runtime_id,
            stage_host=record.stage_host,
            stage_root=record.stage_root,
            write_scope_fingerprint=record.write_scope_fingerprint,
            continuation=continuation,
            retry_feedback=(
                self._retry_feedback(record) if continuation in {"retry", "handoff"} else ()
            ),
        )
        try:
            try:
                outcome = asyncio.run(
                    self._consume(record.project_id, record.kind, request, execution)
                )
            finally:
                self._stream_closed(record, request, execution)
        except TaskPaused as exc:
            result: dict[str, object] | None = None
            if exc.messages or exc.artifacts:
                result = {"messages": exc.messages}
                if exc.artifacts:
                    result["artifacts"] = [item.model_dump(mode="json") for item in exc.artifacts]
            self.store.pause_agent_task(
                operation_id,
                detail=str(exc) or None,
                result=result,
            )
        except Exception as exc:  # The persisted task is the API error boundary.
            if (
                isinstance(request, AutoResearchRunRequest)
                and request.role == "orchestrator"
                and isinstance(exc, AutoResearchOrchestratorTerminalFailure)
            ):
                record_structural_failure(
                    self,
                    operation_id=operation_id,
                    diagnostic=str(exc),
                )
            elif (
                isinstance(request, EpisodeReportRunRequest)
                or (isinstance(request, AutoResearchRunRequest) and request.role == "orchestrator")
            ) and isinstance(exc, TaskFailed):
                self.store.record_agent_task_receipt(
                    operation_id,
                    "provider_terminal_error",
                    {
                        "provider": request.provider,
                        "classification": classify_terminal_error(str(exc)),
                    },
                    tier="diagnostic",
                )
            self.store.record_agent_task_receipt(
                operation_id,
                "operation_exception",
                {"exception_type": type(exc).__name__},
                tier="diagnostic",
            )
            partial = exc.messages if isinstance(exc, TaskFailed) else []
            artifacts = exc.artifacts if isinstance(exc, TaskFailed) else []
            result: dict[str, object] = {"messages": partial}
            if artifacts:
                result["artifacts"] = [item.model_dump(mode="json") for item in artifacts]
            if isinstance(exc, TaskFailed):
                # The one call from the general engine into one job type's
                # policy, kept on purpose.  It fires after a provider failure
                # deep inside a running worker, so there is no caller to invert.
                # Replacing it with a generic failure event that Experiment
                # policy subscribes to would be a dispatch registry under
                # another name, and would hide the behaviour rather than
                # decouple it.  One honest named call is the better trade —
                # which is why the callee moved out and this call did not.
                record_bound_experiment_session_limit(self, record, request, str(exc))
            current = self.store.agent_task(operation_id)
            report_already_finalized = (
                isinstance(request, EpisodeReportRunRequest)
                and current is not None
                and current.status in {"succeeded", "failed"}
            )
            if not report_already_finalized:
                self.store.fail_agent_task(
                    operation_id,
                    str(exc),
                    result=result if partial or artifacts else None,
                )
        else:
            # Only ingest runs owe a graph revision. A chat turn answers a
            # question; changing the graph is the exception, not the contract.
            if record.kind in {"seed", "refresh"} and outcome.applied_revision is None:
                if control.pause_requested.is_set():
                    self.store.pause_agent_task(operation_id)
                else:
                    self.store.record_agent_task_receipt(
                        operation_id,
                        "missing_applied_revision",
                        {"agent_stream_completed": True},
                        tier="diagnostic",
                    )
                    self.store.fail_agent_task(
                        operation_id,
                        "The agent stopped without applying a graph revision.",
                    )
            else:
                result: dict[str, object] = {"messages": outcome.messages}
                if outcome.artifacts:
                    result["artifacts"] = [
                        item.model_dump(mode="json") for item in outcome.artifacts
                    ]
                if outcome.graph_update is not None:
                    result["graph_update"] = outcome.graph_update.model_dump(mode="json")
                if outcome.graph_updates:
                    result["graph_updates"] = [
                        item.model_dump(mode="json") for item in outcome.graph_updates
                    ]
                current = self.store.agent_task(operation_id)
                report_already_finalized = (
                    isinstance(request, EpisodeReportRunRequest)
                    and current is not None
                    and current.status in {"succeeded", "failed"}
                )
                if not report_already_finalized:
                    self.store.complete_agent_task(
                        operation_id,
                        applied_revision=outcome.applied_revision,
                        result=result,
                    )
        finally:
            try:
                if isinstance(request, RunRequest) and request.patch_kind == "experiment_loop":
                    self.store.settle_ready_experiment_loop_stops()
            finally:
                try:
                    self._task_settled(record, request, execution)
                finally:
                    self._forget_control(operation_id)

    def _stream_closed(
        self,
        record: AgentTaskRecord,
        request: AgentTaskRequest,
        execution: AgentTaskExecution,
    ) -> None:
        if self.on_stream_closed is None:
            return
        try:
            self.on_stream_closed(record.project_id, record.kind, request, execution)
        except Exception as exc:
            # An observer must never replace the stream's actual paused, failed,
            # or completed verdict.
            with suppress(Exception):
                self.store.record_agent_task_receipt(
                    execution.operation_id,
                    "stream_closed_callback_failed",
                    {"exception_type": type(exc).__name__},
                    tier="diagnostic",
                )

    def _task_settled(
        self,
        record: AgentTaskRecord,
        request: AgentTaskRequest,
        execution: AgentTaskExecution,
    ) -> None:
        if isinstance(request, EpisodeReportRunRequest):
            return
        if self.on_task_settled is not None:
            try:
                self.on_task_settled(record.project_id, record.kind, request, execution)
            except Exception as exc:
                # Delivery observation runs after the task verdict and must never
                # replace it, including when the agent stream itself failed.
                with suppress(Exception):
                    self.store.record_agent_task_receipt(
                        execution.operation_id,
                        "task_settled_callback_failed",
                        {"exception_type": type(exc).__name__},
                        tier="diagnostic",
                    )

    async def _consume(
        self,
        project_id: str,
        kind: AgentTaskKind,
        request: AgentTaskRequest,
        execution: AgentTaskExecution,
    ) -> AgentTaskOutcome:
        applied_revision: int | None = None
        messages: list[str] = []
        artifacts: list[AgentArtifactDescriptor] = []
        graph_update: GraphUpdateResult | None = None
        graph_updates: list[GraphUpdateResult] = []
        # `aclosing` so an error or a pause closes the run generator here rather
        # than leaving it suspended for the garbage collector: its `finally` is
        # what releases the canonical run lock and retains the scratch folder.
        async with aclosing(self.stream(project_id, kind, request, execution)) as stream:
            async for frame in stream:
                event = _event_from_sse(frame)
                if event.usage is not None:
                    usage_record = self.store.record_agent_usage(
                        execution.operation_id,
                        event.usage,
                    )
                    self.store.record_agent_task_receipt(
                        execution.operation_id,
                        "provider_usage",
                        {
                            "usage_id": usage_record.usage_id,
                            "counted": usage_record.counted,
                            "count_reason": usage_record.count_reason,
                            "provider_profile": usage_record.provider_profile,
                            "processed_input_tokens": usage_record.processed_input_tokens,
                            "generated_tokens": usage_record.generated_tokens,
                        },
                        tier="diagnostic",
                    )
                if event.event == "error":
                    raise TaskFailed(event.text or "The agent task failed.", messages, artifacts)
                if event.event == "paused":
                    raise TaskPaused(event.text, messages, artifacts)
                if event.event == "runtime_fallback":
                    self.store.record_agent_task_receipt(
                        execution.operation_id,
                        "provider_runtime_fallback",
                        _runtime_fallback_payload(event.text),
                        tier="diagnostic",
                    )
                    continue
                if event.event == "runtime":
                    if not request.provider:
                        raise TaskFailed(
                            "The provider runtime has no admitted provider.",
                            messages,
                            artifacts,
                        )
                    try:
                        self.store.checkpoint_agent_task_runtime(
                            execution.operation_id,
                            provider=request.provider,
                            runtime_id=event.text,
                        )
                    except ValueError as exc:
                        # A requeue of this same operation can legitimately reach
                        # a different runtime than the one already recorded. That
                        # is a verdict about this task, not an internal fault.
                        raise TaskFailed(str(exc), messages, artifacts) from exc
                    execution.runtime_id = event.text
                    continue
                if event.event == "session" and event.session_id:
                    self.store.checkpoint_agent_task(
                        execution.operation_id,
                        native_session_id=event.session_id,
                    )
                    self.store.record_agent_task_receipt(
                        execution.operation_id,
                        "native_agent_checkpoint",
                        {
                            "provider": request.provider,
                            "runtime_id": execution.runtime_id,
                            "run_on": request.run_on,
                            "native_session_id": event.session_id,
                            "continuation_cause": execution.continuation,
                            "resumed": execution.reuses_native_checkpoint,
                        },
                        tier="diagnostic",
                    )
                    self.store.update_agent_task_message(
                        execution.operation_id,
                        "Agent task is running.",
                        phase="agent",
                    )
                if event.event == "message":
                    revision = _applied_revision(event.text)
                    parsed_graph_updates = _graph_updates(event.text)
                    if parsed_graph_updates is not None:
                        graph_updates.extend(parsed_graph_updates)
                        graph_updates = graph_updates[-GRAPH_UPDATE_HISTORY_MAX_COUNT:]
                        if parsed_graph_updates:
                            graph_update = parsed_graph_updates[-1]
                    parsed_graph_update = _graph_update(event.text)
                    if parsed_graph_update is not None:
                        graph_update = parsed_graph_update
                    if revision is None:
                        revision_candidates = (
                            [parsed_graph_update] if parsed_graph_update is not None else []
                        ) + list(reversed(parsed_graph_updates or []))
                        revision = next(
                            (
                                update.applied_revision
                                for update in revision_candidates
                                if update.applied_revision is not None
                            ),
                            None,
                        )
                    if revision is not None:
                        applied_revision = revision
                        # A retained Experiment patch can report its historical
                        # commit id while the captured state is the current
                        # idempotent result. Keep that state and revision paired.
                        if execution.applied_graph_state is None:
                            execution.applied_revision = revision
                        self.store.update_agent_task_message(
                            execution.operation_id,
                            "Applying the graph update.",
                            phase="applying",
                            event=True,
                        )
                    elif (
                        parsed_graph_update is None
                        and parsed_graph_updates is None
                        and event.text.strip()
                        and len(messages) < 32
                    ):
                        messages.append(event.text.strip()[:16_000])
                if event.event == "answer" and event.text.strip() and len(messages) < 32:
                    messages.append(event.text.strip()[:16_000])
                if (
                    event.event == "artifact"
                    and event.artifact is not None
                    and len(artifacts) < CHAT_ARTIFACT_MAX_COUNT
                    and event.artifact.artifact_id not in {item.artifact_id for item in artifacts}
                ):
                    artifacts.append(event.artifact)
                if event.event == "raw" and event.text.startswith(
                    "Omitted oversized provider event"
                ):
                    self.store.record_agent_task_event(
                        execution.operation_id,
                        event.text,
                        level="warning",
                    )
                    self.store.record_agent_task_receipt(
                        execution.operation_id,
                        "provider_event_omitted",
                        {"reason": "provider_event_exceeded_stream_limit"},
                        tier="trace",
                    )
        return AgentTaskOutcome(
            applied_revision=(
                applied_revision if applied_revision is not None else execution.applied_revision
            ),
            messages=messages,
            artifacts=artifacts,
            graph_update=graph_update,
            graph_updates=graph_updates,
        )

    def _require_operation(self, operation_id: str) -> AgentTaskRecord:
        record = self.store.agent_task(operation_id)
        if record is None:
            raise KeyError(operation_id)
        return record

    def _session_is_rcp_owned(self, record: AgentTaskRecord) -> bool:
        if record.kind in {"node_chat", "project_chat", "paper_coach"}:
            return bool(record.native_session_id)
        seen: set[str] = set()
        current = record
        while current.operation_id not in seen:
            seen.add(current.operation_id)
            request = self._request_from_record(current)
            if request.session_id is None:
                return bool(current.native_session_id)
            if not current.parent_operation_id:
                return False
            parent = self.store.agent_task(current.parent_operation_id)
            if (
                parent is None
                or parent.project_id != current.project_id
                or parent.kind != current.kind
                or parent.native_session_id != request.session_id
            ):
                return False
            current = parent
        return False

    def _retry_feedback(self, record: AgentTaskRecord) -> tuple[str, ...]:
        feedback: list[str] = []
        seen_operations: set[str] = set()
        seen_errors: set[str] = set()
        parent_id = record.parent_operation_id
        while parent_id and parent_id not in seen_operations and len(feedback) < 3:
            seen_operations.add(parent_id)
            parent = self.store.agent_task(parent_id)
            if (
                parent is None
                or parent.project_id != record.project_id
                or parent.kind != record.kind
            ):
                break
            if parent.error:
                detail = " ".join(parent.error.split())[:1600]
                if detail and detail not in seen_errors:
                    feedback.append(
                        f"Attempt {parent.attempt} ({parent.status}) failed with: {detail}"
                    )
                    seen_errors.add(detail)
            parent_id = parent.parent_operation_id
        return tuple(feedback)

    def _failure_is_session_limit(self, record: AgentTaskRecord) -> bool:
        classified_receipt = any(
            receipt.category == "provider_terminal_error"
            and receipt.payload.get("classification") == "session_limit"
            for receipt in self.store.agent_task_receipts(record.operation_id)
        )
        return classified_receipt or (
            bool(record.error) and classify_terminal_error(record.error or "") == "session_limit"
        )

    def _continuation_context_is_unavailable(self, record: AgentTaskRecord) -> bool:
        return any(
            receipt.category == "continuation_context_unavailable"
            and receipt.payload.get("retry_required") is True
            for receipt in self.store.agent_task_receipts(record.operation_id)
        )

    @staticmethod
    def _request_from_record(record: AgentTaskRecord) -> AgentTaskRequest:
        model: type[AgentTaskRequest] = {
            "paper_coach": CoachRequest,
            "auto_research": AutoResearchRunRequest,
            "branch_merge": BranchMergeRunRequest,
            "episode_report": EpisodeReportRunRequest,
        }.get(record.kind, RunRequest)
        return load_stored_request(model, record.request, operation_id=record.operation_id)

    @staticmethod
    def _validate_request_type(kind: AgentTaskKind, request: AgentTaskRequest) -> None:
        if kind == "paper_coach" and not isinstance(request, CoachRequest):
            raise TypeError("paper_coach requires a CoachRequest")
        if kind == "auto_research" and not isinstance(request, AutoResearchRunRequest):
            raise TypeError("auto_research requires an AutoResearchRunRequest")
        if kind == "branch_merge" and not isinstance(request, BranchMergeRunRequest):
            raise TypeError("branch_merge requires a BranchMergeRunRequest")
        if kind == "episode_report" and not isinstance(request, EpisodeReportRunRequest):
            raise TypeError("episode_report requires an EpisodeReportRunRequest")
        if kind not in {
            "paper_coach",
            "auto_research",
            "branch_merge",
            "episode_report",
        } and not isinstance(request, RunRequest):
            raise TypeError(f"{kind} requires a RunRequest")

    def _forget_control(self, operation_id: str) -> None:
        with self._controls_lock:
            self._controls.pop(operation_id, None)
            self._workers.pop(operation_id, None)


def _runtime_fallback_payload(text: str) -> dict[str, object]:
    """Keep an unparsable diagnostic rather than losing why a runtime was skipped."""

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = None
    return payload if isinstance(payload, dict) else {"detail": text[:400]}


def _event_from_sse(frame: str) -> AgentEvent:
    data = next(
        (line[6:] for line in frame.splitlines() if line.startswith("data: ")),
        "",
    )
    if not data:
        raise ValueError("The background task emitted an invalid event.")
    return AgentEvent.model_validate_json(data)


def _applied_revision(text: str) -> int | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or "applied_revision" not in value:
        return None
    try:
        return int(value["applied_revision"])
    except (TypeError, ValueError):
        return None


def _graph_update(text: str) -> GraphUpdateResult | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or "graph_update" not in value:
        return None
    try:
        return GraphUpdateResult.model_validate(value["graph_update"])
    except (TypeError, ValueError):
        return None


def _graph_updates(text: str) -> list[GraphUpdateResult] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or "graph_updates" not in value:
        return None
    raw_updates = value["graph_updates"]
    if not isinstance(raw_updates, list):
        return None
    try:
        return [GraphUpdateResult.model_validate(item) for item in raw_updates]
    except (TypeError, ValueError):
        return None
