"""Admission for a watcher-attributed Work turn.

Watcher policy, not engine plumbing: consuming completed watchers atomically and
deciding what an Experiment wake may carry is specific to watcher delivery.

The delivery-admission flag it reads belongs to the engine — ``shutdown`` clears
it and ``accept_watcher_notifications`` sets it, and those two stay there as one
matched pair.  This module reads it under the engine's own lock rather than
copying the fence.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from rcp.core.models import AuthorizedHuman
from rcp.runs.task_policy import resolved_dispatch_authority
from rcp.service import RunRequest
from rcp.storage import AgentTaskRecord, EpisodeRecord

if TYPE_CHECKING:
    from rcp.background import BackgroundAgentTasks


def start_watcher_notification(
    tasks: BackgroundAgentTasks,
    project_id: str,
    kind: Literal["node_chat", "project_chat"],
    request: RunRequest,
    watcher_ids: list[str],
    *,
    authorized_by: AuthorizedHuman,
    episode_stage_host: str | None = None,
    episode_stage_root: str | None = None,
    admission_fence: Callable[[Callable[[], None]], bool] | None = None,
) -> AgentTaskRecord | None:
    """Atomically consume completed watchers and start their attributed Work turn.

    An Experiment-loop wake carries the episode's native session so the turn
    continues that bounded session. It is still a new task at the next
    invocation, so it uses the `watcher_wake` cause rather than Resume.
    """

    if not authorized_by.display_name.strip():
        raise ValueError("A watcher notification requires a named human authorizer snapshot.")

    source_watchers = [tasks.store.watcher(watcher_id) for watcher_id in watcher_ids]
    if any(item is None for item in source_watchers):
        raise ValueError("A watcher notification requires every durable watcher record.")
    resolved_watchers = [item for item in source_watchers if item is not None]
    graph_targets = {item.graph_target.key: item.graph_target for item in resolved_watchers}
    if len(graph_targets) != 1:
        raise ValueError("A watcher notification cannot cross graph targets.")
    graph_target = next(iter(graph_targets.values()))
    branch_episode_ids = {
        item.episode_id for item in resolved_watchers if item.episode_id is not None
    }
    if graph_target.kind == "branch" and len(branch_episode_ids) != 1:
        raise ValueError("A branch watcher notification requires one exact episode lineage.")

    experiment_reauthorization = (
        request.trigger == "experiment_run"
        and request.patch_kind == "experiment_loop"
        and request.control_invocation == 1
        and bool(request.watcher_ids)
    )
    experiment_wake = request.trigger == "watcher" and request.patch_kind == "experiment_loop"
    if (
        (request.trigger != "watcher" and not experiment_reauthorization)
        or request.mode != "work"
        or (request.session_id and not experiment_wake)
    ):
        raise ValueError("A watcher notification must be a fresh watcher-attributed Work turn.")
    if experiment_wake and (not request.session_id or not episode_stage_root):
        raise ValueError(
            "An Experiment watcher wake requires its episode's session and exact stage."
        )
    episode: EpisodeRecord | None = None
    if experiment_wake:
        episode = tasks.store.episode(request.control_episode_id or "")
        if (
            episode is None
            or episode.mode != "experiment_loop"
            or episode.project_id != project_id
            or episode.graph_target != graph_target
        ):
            raise ValueError("The Experiment watcher wake lost its episode parent.")
        if episode.authorized_by is None:
            raise ValueError("The Experiment episode lost its human authorizer snapshot.")
        authorized_by = episode.authorized_by
    if request.watcher_ids != watcher_ids:
        raise ValueError("The watcher notification request must name its watcher records.")
    if not isinstance(request, RunRequest):
        raise TypeError(f"{kind} requires a RunRequest")
    request_data = request.model_dump(mode="json")
    estimate, samples = tasks.store.agent_task_estimate(project_id, kind, request_data)
    dispatch_authority = resolved_dispatch_authority(
        tasks.store,
        tasks.dispatch_authority_resolver,
        kind,
        request,
        project_id=project_id,
    )
    operation_id = str(uuid.uuid4())
    now = tasks.store.now()
    record = AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        episode_id=request.control_episode_id
        if experiment_wake or experiment_reauthorization
        else next(iter(branch_episode_ids))
        if graph_target.kind == "branch"
        else None,
        graph_target=graph_target,
        kind=kind,
        status="queued",
        request=request_data,
        created_at=now,
        updated_at=now,
        status_message="Waiting to deliver a watcher update.",
        native_session_id=request.session_id if experiment_wake else None,
        stage_host=episode_stage_host if experiment_wake else None,
        stage_root=episode_stage_root if experiment_wake else None,
        estimate_seconds=estimate,
        estimate_samples=samples,
        phase="queued",
        last_activity_at=now,
        authorized_by=authorized_by,
        dispatch_authority=dispatch_authority,
    )
    started: AgentTaskRecord | None = None

    def claim_and_spawn() -> None:
        nonlocal started
        with tasks._watcher_delivery_lock:
            if not tasks._accepting_watcher_deliveries:
                return
            if experiment_reauthorization:
                stored = tasks.store.create_experiment_episode_with_invocation(
                    record,
                    watcher_ids,
                )
            elif experiment_wake:
                stored = tasks.store.create_experiment_watcher_invocation(
                    record,
                    watcher_ids,
                )
            else:
                stored = tasks.store.create_watcher_notification_task(
                    record,
                    watcher_ids,
                    continuation_cause="fresh",
                )
            if stored is None:
                return
            started = tasks.launch_admitted(stored.operation_id)

    if admission_fence is not None:
        if not admission_fence(claim_and_spawn):
            return None
    else:
        claim_and_spawn()
    return started
