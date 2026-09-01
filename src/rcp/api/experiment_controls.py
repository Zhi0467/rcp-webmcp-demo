from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

from rcp.api.episodes import EpisodeResponse, EpisodeTaskResponse
from rcp.control import (
    ExperimentControlState,
    ExperimentOperationalState,
    ExperimentSessionBinding,
    derive_experiment_control_state,
)
from rcp.core.models import (
    CLOSED_EXPERIMENT_STATUSES,
    Experiment,
    ExperimentDecisionPin,
    GraphState,
)
from rcp.core.transition_models import GraphTargetRef
from rcp.storage import AppStore, ExperimentLoopRuntime
from rcp.storage.models import (
    ACTIVE_AGENT_TASK_STATUSES,
    AWAITING_HUMAN_AGENT_TASK_STATUSES,
)

ExperimentLoopHealth = Literal[
    "starting",
    "agent_active",
    "waiting_on_watchers",
    "degraded",
    "stopping",
    "wrapping_up",
    "failed",
    "human_stopped",
    "paused_at_limit",
    "needs_action",
    "completed",
]
ExperimentRecommendationKind = Literal[
    "wait",
    "resume",
    "retry",
    "keep_loop",
    "start_episode",
    "stop_and_restart",
    "resolve_requirements",
    "open_report",
    "review",
    "none",
]
ExperimentRunSection = Literal["running", "actionable", "completed"]
ExperimentTaskControlKind = Literal["resume", "retry"]


class ExperimentControlResponse(ExperimentControlState):
    """One Experiment's graph control plus its decided Runs lifecycle."""

    model_config = ConfigDict(extra="forbid", strict=True)

    episode: EpisodeResponse | None
    health: ExperimentLoopHealth
    recommendation: ExperimentRecommendationKind
    run_section: ExperimentRunSection
    live: bool
    can_start: bool
    can_stop: bool
    stop_pending: bool
    task_control: ExperimentTaskControlKind | None
    can_switch_provider: bool
    can_open_report: bool
    report_episode_id: str | None
    node_closed: bool


def _experiment_control(
    store: AppStore,
    project_id: str,
    state: GraphState,
    experiment_id: str,
    *,
    graph_target: GraphTargetRef,
) -> tuple[ExperimentLoopRuntime, ExperimentControlState]:
    """Derive one Experiment's operational and semantic control state together.

    Deriving is also where a graceful stop is reconciled, so the same joint
    handoff settles identically after a restart without anyone replaying it.
    """

    runtime = store.experiment_loop_runtime(
        project_id,
        experiment_id,
        graph_target=graph_target,
    )
    if runtime.stop_requested and not runtime.stop_settled and not runtime.task_active:
        store.settle_experiment_loop_stop(
            project_id,
            experiment_id,
            episode_id=runtime.episode_id,
            graph_target=graph_target,
        )
        runtime = store.experiment_loop_runtime(
            project_id,
            experiment_id,
            graph_target=graph_target,
        )
    return runtime, _experiment_control_from_runtime(state, experiment_id, runtime)


def _experiment_control_for_target(
    store: AppStore,
    project_id: str,
    state: GraphState,
    experiment_id: str,
    *,
    graph_target: GraphTargetRef,
) -> tuple[ExperimentLoopRuntime, ExperimentControlState]:
    """Derive and reconcile one exact target-bound operational runtime."""

    runtime = store.experiment_loop_runtime_for_target(
        project_id,
        experiment_id,
        graph_target,
    )
    if runtime.stop_requested and not runtime.stop_settled and not runtime.task_active:
        store.settle_experiment_loop_stop(
            project_id,
            experiment_id,
            episode_id=runtime.episode_id,
            graph_target=graph_target,
        )
        runtime = store.experiment_loop_runtime_for_target(
            project_id,
            experiment_id,
            graph_target,
        )
    return runtime, _experiment_control_from_runtime(state, experiment_id, runtime)


def _experiment_control_from_runtime(
    state: GraphState,
    experiment_id: str,
    runtime: ExperimentLoopRuntime,
) -> ExperimentControlState:
    """Combine graph authority with one already-projected operational runtime."""

    pins = [ExperimentDecisionPin.model_validate(item) for item in runtime.decision_bundle]
    return derive_experiment_control_state(
        state,
        experiment_id,
        {experiment_id} if runtime.active else set(),
        episode_id=runtime.episode_id,
        invocations_used=runtime.invocations_used,
        invocation_ceiling=runtime.invocation_ceiling,
        paused=runtime.paused,
        detached_work_active=runtime.detached_work_active,
        episode_decision_bundle=pins if runtime.episode_id is not None else None,
        operational=_experiment_operational_state(runtime),
    )


def _experiment_control_response(
    state: GraphState,
    experiment_id: str,
    runtime: ExperimentLoopRuntime,
    episode_payload: dict[str, object] | EpisodeResponse | None,
    *,
    latest_report_episode_id: str | None = None,
) -> ExperimentControlResponse:
    """Publish the complete Experiment control consumed by Runs.

    The canonical Experiment remains editable graph data. This response owns only
    operational conclusions whose inputs are already backend state, so a browser
    cannot grow a second lifecycle by combining the raw fields differently.
    """

    node = state.nodes.get(experiment_id)
    if not isinstance(node, Experiment):
        raise ValueError(f"Node {experiment_id!r} is not an Experiment.")
    control = _experiment_control_from_runtime(state, experiment_id, runtime)
    episode = (
        episode_payload
        if isinstance(episode_payload, EpisodeResponse)
        else EpisodeResponse.model_validate(episode_payload)
        if episode_payload is not None
        else None
    )
    task = _experiment_control_task(episode)
    task_control = _experiment_task_control(episode)
    queued = task.queued if task is not None else control.operational.current_queued
    active = task.active if task is not None else control.operational.current_active
    awaiting_human = (
        task.awaiting_human if task is not None else control.operational.current_awaiting_human
    )
    has_valid_recovery = task_control is not None or (
        task is None and control.operational.task_active and awaiting_human
    )
    health = _experiment_run_health(
        node,
        control,
        episode,
        queued=queued,
        active=active,
        awaiting_human=awaiting_human,
        has_valid_recovery=has_valid_recovery,
    )
    recommendation = _experiment_recommendation(
        control,
        episode,
        health,
        active=active,
        awaiting_human=awaiting_human,
        task_control=task_control,
    )
    run_section = _experiment_run_section(health, awaiting_human=awaiting_human)
    ended = episode is not None and episode.ending is not None
    live = not ended and bool(
        control.operational.episode_live
        or active
        or control.operational.task_active
        or control.operational.detached_work_active
        or control.operational.watcher_completion_pending
    )
    stop_pending = control.operational.stop_requested and not control.operational.stop_settled
    can_stop = bool(
        control.episode_id
        and not ended
        and not control.operational.stop_requested
        and (live or awaiting_human)
    )
    report_episode_id = latest_report_episode_id or (
        episode.episode_id
        if episode is not None and episode.report is not None and episode.wrapup_state == "ready"
        else None
    )
    can_open_report = report_episode_id is not None
    can_switch_provider = bool(task_control is not None and task is not None and task.can_retry)
    return ExperimentControlResponse.model_validate(
        {
            **control.model_dump(mode="json"),
            "episode": episode.model_dump(mode="json") if episode is not None else None,
            "health": health,
            "recommendation": recommendation,
            "run_section": run_section,
            "live": live,
            "can_start": control.ready,
            "can_stop": can_stop,
            "stop_pending": stop_pending,
            "task_control": task_control,
            "can_switch_provider": can_switch_provider,
            "can_open_report": can_open_report,
            "report_episode_id": report_episode_id,
            "node_closed": node.status in CLOSED_EXPERIMENT_STATUSES,
        }
    )


def _experiment_control_task(episode: EpisodeResponse | None) -> EpisodeTaskResponse | None:
    if episode is None or episode.current_control_task_id is None:
        return None
    return next(
        (task for task in episode.tasks if task.operation_id == episode.current_control_task_id),
        None,
    )


def _experiment_task_control(
    episode: EpisodeResponse | None,
) -> ExperimentTaskControlKind | None:
    if episode is None or episode.task_control not in {"resume", "retry"}:
        return None
    return episode.task_control


def _experiment_run_health(
    node: Experiment,
    control: ExperimentControlState,
    episode: EpisodeResponse | None,
    *,
    queued: bool,
    active: bool,
    awaiting_human: bool,
    has_valid_recovery: bool,
) -> ExperimentLoopHealth:
    operational = control.operational
    stop_requested = operational.stop_requested
    if stop_requested and not operational.stop_settled:
        return "needs_action" if has_valid_recovery else "stopping"
    if (
        node.status in CLOSED_EXPERIMENT_STATUSES
        and episode is not None
        and episode.health == "stopped"
    ):
        return "completed"
    if stop_requested and operational.stop_settled and awaiting_human:
        return "human_stopped"
    if episode is not None and episode.wrapup_state in {"pending", "running"}:
        return "wrapping_up"
    if episode is not None and episode.health == "failed":
        return "failed"
    if episode is not None and episode.health == "stopped":
        return "human_stopped"
    if episode is not None and episode.ending is not None:
        if episode.ending == "completed":
            return "completed"
        if node.status in CLOSED_EXPERIMENT_STATUSES:
            return "completed"
        if episode.ending == "exhausted":
            return "paused_at_limit"
        if episode.ending == "human_pause":
            return "needs_action"
    if queued:
        return "starting"
    if active:
        return "agent_active"
    if awaiting_human or operational.task_active:
        return "needs_action"

    completion_pending = operational.watcher_completion_pending
    detached_work_active = operational.detached_work_active
    can_wake = bool(
        not stop_requested
        and control.invocations_remaining > 0
        and not operational.episode_exited
        and not operational.session.diagnostic
        and not control.graph_reasons
    )
    if (completion_pending or detached_work_active) and control.invocations_remaining <= 0:
        return "paused_at_limit"
    if completion_pending and not can_wake:
        return "needs_action"
    if detached_work_active and not can_wake:
        return "needs_action"
    if operational.watcher_degraded:
        return "degraded"
    if completion_pending or detached_work_active:
        return "waiting_on_watchers"
    if node.status in CLOSED_EXPERIMENT_STATUSES:
        return "completed"
    if stop_requested:
        return "human_stopped"
    if control.invocations_remaining <= 0 and control.invocations_used > 0:
        return "paused_at_limit"
    return "needs_action"


def _experiment_recommendation(
    control: ExperimentControlState,
    episode: EpisodeResponse | None,
    health: ExperimentLoopHealth,
    *,
    active: bool,
    awaiting_human: bool,
    task_control: ExperimentTaskControlKind | None,
) -> ExperimentRecommendationKind:
    if health in {"stopping", "wrapping_up"}:
        return "wait"
    if episode is not None and episode.wrapup_state == "ready":
        return "open_report"
    if episode is not None and episode.wrapup_state == "legacy_unavailable":
        return "none"
    if health == "failed":
        return "none"
    if active:
        return "wait"
    if task_control is not None:
        return task_control
    if health == "degraded":
        return "keep_loop"
    if health == "waiting_on_watchers":
        return "wait"
    if health == "completed":
        return "none"
    if control.graph_reasons:
        return "resolve_requirements"
    ended = episode is not None and episode.ending is not None
    if (
        control.episode_id
        and not ended
        and not control.operational.stop_requested
        and (awaiting_human or control.operational.episode_live)
    ):
        return "stop_and_restart"
    if health in {"paused_at_limit", "human_stopped"}:
        return "start_episode" if control.ready else "review"
    if control.ready:
        return "start_episode"
    return "review"


def _experiment_run_section(
    health: ExperimentLoopHealth,
    *,
    awaiting_human: bool,
) -> ExperimentRunSection:
    if health == "stopping" and awaiting_human:
        return "actionable"
    if health in {
        "starting",
        "agent_active",
        "waiting_on_watchers",
        "degraded",
        "stopping",
        "wrapping_up",
    }:
        return "running"
    if health in {"human_stopped", "paused_at_limit", "needs_action"}:
        return "actionable"
    return "completed"


def _experiment_operational_state(runtime: ExperimentLoopRuntime) -> ExperimentOperationalState:
    """Project the loop runtime onto the operational block Runs reads.

    The native session id itself stays in the backend; whether one is bound is
    the only part of it the human needs.
    """

    return ExperimentOperationalState(
        task_active=runtime.task_active,
        detached_work_active=runtime.detached_work_active,
        watcher_degraded=runtime.watcher_degraded,
        watcher_completion_pending=runtime.watcher_completion_pending,
        episode_exited=runtime.episode_exited,
        episode_live=runtime.episode_live,
        stop_requested=runtime.stop_requested,
        stop_settled=runtime.stop_settled,
        chat_id=runtime.chat_id,
        current_operation_id=runtime.current_operation_id,
        current_status=runtime.current_status,
        current_queued=runtime.current_status == "queued",
        current_active=runtime.current_status in ACTIVE_AGENT_TASK_STATUSES,
        current_awaiting_human=runtime.current_status in AWAITING_HUMAN_AGENT_TASK_STATUSES,
        current_phase=runtime.current_phase,
        current_status_message=runtime.current_status_message,
        current_last_activity_at=runtime.current_last_activity_at,
        current_invocation=runtime.current_invocation,
        session=ExperimentSessionBinding(
            provider=runtime.provider,
            model=runtime.model,
            reasoning=runtime.reasoning,
            run_on=runtime.run_on,
            execution_host=runtime.execution_host,
            run_truth_scope=runtime.run_truth_scope,
            native_session_bound=runtime.session_bound,
            diagnostic=runtime.session_diagnostic,
        ),
    )


__all__ = [
    "_experiment_control",
    "_experiment_control_for_target",
    "_experiment_control_from_runtime",
    "_experiment_control_response",
    "_experiment_operational_state",
    "ExperimentControlResponse",
]
