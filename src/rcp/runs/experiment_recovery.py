"""Recovery and retry for a bounded Experiment loop.

Experiment-loop policy: what a Stop fence may let finish, what a retry of a
session-bound turn is allowed to change, and what a provider session limit means
for the episode that hit it. None of it generalises to another task kind.

It takes the engine because recovery ends in ordinary admission — the engine
owns the launch gate and the request decoding these need.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rcp.core.models import AuthorizedHuman
from rcp.providers import classify_terminal_error
from rcp.runs.task_policy import AgentTaskRequest, skill_update
from rcp.service import RunRequest
from rcp.skill_registry import SkillSelection
from rcp.storage import AgentTaskRecord
from rcp.transport import RemoteRunStage

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks

_EXPERIMENT_SESSION_LIMIT_DIAGNOSTIC = (
    "The provider session reached its limit. Retry the same provider to recheck the limit and "
    "resume this episode, or switch provider to continue this same episode and invocation."
)


def restart_stopping_experiment_recoveries(tasks: BackgroundAgentTasks) -> None:
    """Let an already-authorized Experiment turn finish behind its Stop fence.

    Process restart converts a live turn to ``interrupted``.  Stop must keep
    new invocations fenced, but that interruption must not strand the turn
    that was already authorized: when its exact RCP-owned session and stage
    are still usable, recover it without spending another invocation.  A
    concrete continuation problem is persisted so the existing Stop adapter
    can take its established abandonment path instead.
    """

    for previous in tasks.store.stopping_experiment_recovery_candidates():
        try:
            request = tasks._request_from_record(previous)
            if (
                not isinstance(request, RunRequest)
                or request.patch_kind != "experiment_loop"
                or not request.control_episode_id
                or not request.control_node_id
            ):
                continue
            episode = tasks.store.episode(request.control_episode_id)
            if (
                episode is None
                or episode.mode != "experiment_loop"
                or episode.project_id != previous.project_id
                or episode.control_node_id != request.control_node_id
                or episode.stop_requested_at is None
                or episode.stop_settled_at is not None
            ):
                continue
            problem = tasks.store.experiment_episode_recovery_context_problem(previous.operation_id)
            if problem is None:
                if tasks._failure_is_session_limit(previous):
                    problem = "the saved provider session reached its limit"
                elif tasks._continuation_context_is_unavailable(previous):
                    problem = "the saved continuation context is unavailable"
                elif (
                    not previous.native_session_id
                    or not previous.stage_root
                    or not tasks._session_is_rcp_owned(previous)
                ):
                    problem = "the turn has no complete RCP-owned session and stage"
                elif episode.authorized_by is None:
                    problem = "the episode lost its human authorizer snapshot"
                else:
                    try:
                        if previous.stage_host:
                            available = RemoteRunStage(previous.stage_host).directory_exists(
                                previous.stage_root
                            )
                        else:
                            stage = Path(previous.stage_root)
                            available = stage.is_dir() and not stage.is_symlink()
                    except Exception:
                        # A remote transport outage is not evidence that the
                        # saved continuation is unusable. Leave Stop pending
                        # for a later process/reconciliation pass.
                        continue
                    if available is None:
                        continue
                    if available is not True:
                        problem = "the saved provider workspace is unavailable"
            if problem is not None:
                tasks.store.record_experiment_episode_diagnostic(
                    episode_id=episode.episode_id,
                    project_id=episode.project_id,
                    control_node_id=request.control_node_id,
                    diagnostic=(
                        "Stop loop cannot finish its already-authorized turn because "
                        + problem.rstrip(".")
                        + "."
                    ),
                )
                continue
            recovered = (
                tasks.retry(
                    previous.operation_id,
                    authorized_by=episode.authorized_by,
                )
                if previous.status == "failed"
                else tasks.resume(previous.operation_id)
            )
            tasks.store.record_agent_task_receipt(
                recovered.operation_id,
                "experiment_stop_recovery",
                {
                    "episode_id": episode.episode_id,
                    "recovered_operation_id": previous.operation_id,
                },
                tier="summary",
            )
            tasks.store.record_agent_task_event(
                recovered.operation_id,
                "Resuming the already-authorized Experiment turn so its graceful Stop can settle.",
            )
        except (KeyError, RuntimeError, ValueError):
            # Startup recovery is best effort. A transaction race or a
            # temporarily unreachable remote stage remains retryable on the
            # next reconciliation rather than preventing the app from opening.
            continue


def retry_experiment_loop(
    tasks: BackgroundAgentTasks,
    previous: AgentTaskRecord,
    original: RunRequest,
    *,
    provider: str | None,
    model: str | None,
    reasoning: str | None,
    skills: SkillSelection | None,
    authorized_by: AuthorizedHuman | None,
) -> AgentTaskRecord:
    """Recover an Experiment attempt without starting or spending a new episode turn."""

    if not original.control_episode_id:
        raise ValueError("Experiment-loop recovery is missing its episode id.")
    episode = tasks.store.experiment_episode(original.control_episode_id)
    binding_task = (
        tasks.store.agent_task(episode.last_turn_operation_id)
        if episode is not None and episode.last_turn_operation_id
        else None
    )
    binding_request = (
        tasks._request_from_record(binding_task) if binding_task is not None else original
    )
    if not isinstance(binding_request, RunRequest):
        raise ValueError("The Experiment episode binding does not belong to a Work task.")

    active_run_on = (
        episode.execution_machine if episode is not None and episode.session_bound else None
    )
    baseline = {
        **original.model_dump(mode="json"),
        "run_on": active_run_on or binding_request.run_on,
        "session_id": None,
        **skill_update(skills, mode="json"),
    }
    requested_config = {
        key: value
        for key, value in {
            "provider": provider,
            "model": model,
            "reasoning": reasoning,
        }.items()
        if value is not None
    }
    request = RunRequest.model_validate({**baseline, **requested_config})
    config_changed = any(
        requested_config.get(field, baseline.get(field)) != baseline.get(field)
        for field in requested_config
    )
    if config_changed:
        estimate, samples = tasks.store.agent_task_estimate(
            previous.project_id,
            previous.kind,
            request.model_dump(mode="json"),
        )
        return tasks._create_and_spawn(
            previous.project_id,
            previous.kind,
            request,
            parent=previous,
            continuation="handoff",
            estimate_seconds=estimate,
            estimate_samples=samples,
            authorized_by=authorized_by,
        )

    active_config = (
        binding_request.provider,
        binding_request.model,
        binding_request.reasoning,
    )
    current_config = (original.provider, original.model, original.reasoning)
    previous_checkpoint = bool(
        previous.native_session_id and previous.stage_root and tasks._session_is_rcp_owned(previous)
    )
    use_active_binding = bool(
        not previous_checkpoint
        and episode is not None
        and episode.session_bound
        and current_config == active_config
    )
    session_id = episode.native_session_id if use_active_binding else previous.native_session_id
    stage_host = episode.stage_host if use_active_binding else previous.stage_host
    stage_root = episode.stage_root if use_active_binding else previous.stage_root
    owned_checkpoint = previous_checkpoint or use_active_binding
    stage_available: bool | None = True
    if owned_checkpoint and stage_host:
        stage_available = RemoteRunStage(stage_host).directory_exists(stage_root or "")
    elif owned_checkpoint and stage_root:
        stage = Path(stage_root)
        stage_available = stage.is_dir() and not stage.is_symlink()
    if owned_checkpoint and stage_available is True:
        return tasks._create_and_spawn(
            previous.project_id,
            previous.kind,
            request.model_copy(update={"session_id": session_id}),
            parent=previous,
            continuation="retry",
            estimate_seconds=previous.estimate_seconds,
            estimate_samples=previous.estimate_samples,
            stage_host=stage_host,
            stage_root=stage_root,
            authorized_by=authorized_by,
        )

    reason = (
        "the saved provider workspace is unavailable"
        if owned_checkpoint
        else "the episode has no complete RCP-owned native checkpoint and stage"
    )
    if episode is not None and episode.session_bound and current_config == active_config:
        detail = (
            f"This continuation cannot start a fresh provider session because {reason}. "
            "Switch provider to continue this same episode, or use Stop loop to abandon it."
        )
        if original.control_node_id:
            tasks.store.record_experiment_episode_diagnostic(
                episode_id=original.control_episode_id,
                project_id=previous.project_id,
                control_node_id=original.control_node_id,
                diagnostic=detail,
            )
            if episode.stop_requested_at is not None:
                tasks.store.settle_experiment_loop_stop(
                    previous.project_id,
                    original.control_node_id,
                    episode_id=episode.episode_id,
                    graph_target=episode.graph_target,
                )
        raise ValueError(detail)
    estimate, samples = tasks.store.agent_task_estimate(
        previous.project_id,
        previous.kind,
        request.model_dump(mode="json"),
    )
    retried = tasks._create_and_spawn(
        previous.project_id,
        previous.kind,
        request,
        parent=previous,
        continuation="handoff",
        estimate_seconds=estimate,
        estimate_samples=samples,
        authorized_by=authorized_by,
    )
    tasks.store.record_agent_task_receipt(
        retried.operation_id,
        "native_resume_unavailable",
        {"reason": reason},
        tier="diagnostic",
    )
    tasks.store.record_agent_task_event(
        retried.operation_id,
        f"Native resume is unavailable because {reason}; continuing this episode with a "
        "provisional provider session.",
        level="warning",
    )
    return retried


def record_bound_experiment_session_limit(
    tasks: BackgroundAgentTasks,
    record: AgentTaskRecord,
    request: AgentTaskRequest,
    error: str,
) -> None:
    """Persist a bound episode's terminal provider limit before human recovery acts."""

    if (
        not isinstance(request, RunRequest)
        or request.patch_kind != "experiment_loop"
        or not request.control_episode_id
        or not request.control_node_id
        or classify_terminal_error(error) != "session_limit"
    ):
        return
    episode = tasks.store.experiment_episode(request.control_episode_id)
    if (
        episode is None
        or episode.project_id != record.project_id
        or episode.control_node_id != request.control_node_id
        or not episode.session_bound
    ):
        return
    tasks.store.record_experiment_episode_diagnostic(
        episode_id=request.control_episode_id,
        project_id=episode.project_id,
        control_node_id=request.control_node_id,
        diagnostic=_EXPERIMENT_SESSION_LIMIT_DIAGNOSTIC,
    )


def preflight_experiment_episode_recovery(
    tasks: BackgroundAgentTasks,
    record: AgentTaskRecord,
    *,
    request: AgentTaskRequest | None = None,
) -> None:
    """Refuse a legacy Experiment recovery before it creates or launches a child."""

    original = request or tasks._request_from_record(record)
    if not isinstance(original, RunRequest) or original.patch_kind != "experiment_loop":
        return
    problem = tasks.store.experiment_episode_recovery_context_problem(record.operation_id)
    if problem is None:
        return
    assert original.control_episode_id is not None
    assert original.control_node_id is not None
    tasks.store.record_experiment_episode_diagnostic(
        episode_id=original.control_episode_id,
        project_id=record.project_id,
        control_node_id=original.control_node_id,
        diagnostic=problem,
    )
    episode = tasks.store.experiment_episode(original.control_episode_id)
    if episode is not None and episode.stop_requested_at is not None:
        tasks.store.settle_experiment_loop_stop(
            record.project_id,
            original.control_node_id,
            episode_id=episode.episode_id,
            graph_target=episode.graph_target,
        )
    raise ValueError(problem)
