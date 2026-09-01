from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from rcp.api.dependencies import (
    get_background_tasks,
    get_catalog,
    get_experiment_operation_lock,
    get_identity_access,
    get_project_service,
    get_store,
    require_project_membership,
    require_project_write_admission,
    require_registered_project,
)
from rcp.api.episodes import _episode_for_http
from rcp.api.experiment_controls import _experiment_control, _experiment_control_for_target
from rcp.api.identity import IdentityAccess
from rcp.background import BackgroundAgentTasks
from rcp.control import ExperimentControlState
from rcp.core.models import Experiment
from rcp.core.transition_models import GraphTargetRef
from rcp.keyed_locks import KeyedLocks
from rcp.projects import ProjectCatalog
from rcp.runs.experiment_admission import (
    experiment_start_message,
    fresh_experiment_run_request,
    resolve_experiment_node_work_request,
)
from rcp.runs.experiment_loop import experiment_watcher_delivery_request
from rcp.runs.watcher_admission import start_watcher_notification
from rcp.service import RunRequest
from rcp.storage import AppStore, EpisodeNotRunning, EpisodeRecord
from rcp.transport import StateUnavailable

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
StoreDependency = Annotated[AppStore, Depends(get_store)]
IdentityAccessDependency = Annotated[IdentityAccess, Depends(get_identity_access)]
BackgroundTasksDependency = Annotated[BackgroundAgentTasks, Depends(get_background_tasks)]
ExperimentOperationLockDependency = Annotated[
    KeyedLocks,
    Depends(get_experiment_operation_lock),
]


@router.post(
    "/api/projects/{project_id}/experiments/{node_id:path}/run",
    status_code=202,
    dependencies=[Depends(require_project_write_admission)],
)
def run_experiment(
    project_id: str,
    node_id: str,
    body: dict[str, object],
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityAccessDependency,
    background_tasks: BackgroundTasksDependency,
) -> dict[str, object]:
    authorized_by = identity_access.require_patch_capable_identity(request)
    service = get_project_service(catalog, project_id)
    try:
        state = service.history.state()
        node = state.nodes.get(node_id)
        if not isinstance(node, Experiment):
            raise HTTPException(status_code=404, detail="Experiment not found")
        runtime, control = _experiment_control(
            store,
            project_id,
            state,
            node_id,
            graph_target=GraphTargetRef(),
        )
        if not control.ready:
            raise HTTPException(status_code=409, detail=" ".join(control.reasons))
        supplied = RunRequest.model_validate(body)
        if supplied.result_view is not None:
            raise ValueError("Result views require an ordinary node Work turn.")
        if not supplied.chat_id:
            raise ValueError("Run requires a chat_id")
        uuid.UUID(supplied.chat_id)
        episode_id = str(uuid.uuid4())
        pending_group = (
            None
            if runtime.stop_requested and runtime.stop_settled
            else store.completed_experiment_watcher_group(
                project_id,
                node_id,
                graph_target=GraphTargetRef(),
            )
        )
        if pending_group is not None:
            experiment_request = experiment_watcher_delivery_request(
                pending_group,
                trigger="experiment_run",
                episode_id=episode_id,
                invocation=1,
                invocation_ceiling=node.invocation_ceiling,
                control_revision=state.revision,
                decision_bundle=control.governing_decisions,
                completion_criteria=list(node.completion_criteria),
            )
            experiment_request = experiment_request.model_copy(
                update={
                    "run_truth_scope": supplied.run_truth_scope,
                    "chat_scope": "node",
                    "node_id": node_id,
                    "message": experiment_start_message(supplied.message, node_id),
                    "chat_id": supplied.chat_id,
                    "session_id": None,
                }
            )
            experiment_request = resolve_experiment_node_work_request(service, experiment_request)
            record = start_watcher_notification(
                background_tasks,
                project_id,
                "node_chat",
                experiment_request,
                [item.watcher_id for item in pending_group],
                authorized_by=authorized_by,
            )
            if record is None:
                raise ValueError(
                    "The pending watcher completion could not be claimed because its "
                    "conversation is active."
                )
            return record.model_dump(mode="json")
        experiment_request = fresh_experiment_run_request(
            service,
            supplied,
            node=node,
            state_revision=state.revision,
            control=control,
            episode_id=episode_id,
        )
        record = background_tasks.start(
            project_id,
            "node_chat",
            experiment_request,
            authorized_by=authorized_by,
        )
    except ValueError as exc:
        status = 409 if "already running" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return record.model_dump(mode="json")


@router.post("/api/projects/{project_id}/experiments/{node_id:path}/watchers/stop")
def stop_experiment_watchers(
    project_id: str,
    node_id: str,
    *,
    catalog: CatalogDependency,
) -> list[dict[str, object]]:
    """Reject the retired bulk watcher control in favor of graceful Stop loop."""

    require_registered_project(catalog, project_id)
    raise HTTPException(
        status_code=409,
        detail="This control was retired. Use Stop loop for the current Experiment episode.",
    )


def stop_bound_experiment_episode(
    project_id: str,
    episode: EpisodeRecord,
    *,
    store: AppStore,
    catalog: ProjectCatalog,
) -> ExperimentControlState:
    """Stop one exact current loop against the graph target it actually controls."""

    node_id = episode.control_node_id
    if episode.mode != "experiment_loop" or node_id is None:
        raise HTTPException(status_code=409, detail="This is not an Experiment-loop episode.")
    main_service = get_project_service(catalog, project_id)
    try:
        target_service = (
            main_service
            if episode.graph_target.kind == "main"
            else main_service.for_graph_target(
                episode.graph_target,
                expected_episode_id=episode.graph_target.branch_id,
            )
        )
        materialization = target_service.history.current_materialization()
        state = materialization.state
        head = target_service.history.head_ref(materialization)
    except (KeyError, OSError, StateUnavailable, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if head.target != episode.graph_target:
        raise HTTPException(
            status_code=409,
            detail="The Experiment episode no longer resolves to its exact graph target.",
        )
    if not isinstance(state.nodes.get(node_id), Experiment):
        raise HTTPException(status_code=404, detail="Experiment not found")
    if episode.graph_target.kind == "branch":
        route = store.auto_research_child_experiment(episode.episode_id)
        if route is None or route.auto_research_episode_id != episode.graph_target.branch_id:
            raise HTTPException(
                status_code=409,
                detail="The branch Experiment lost its Auto-research parent binding.",
            )
    runtime = store.experiment_loop_runtime_for_target(
        project_id,
        node_id,
        episode.graph_target,
    )
    if runtime.episode_id != episode.episode_id:
        raise HTTPException(
            status_code=409,
            detail="Only the current exact Experiment episode can be stopped.",
        )
    try:
        store.request_experiment_loop_stop(
            project_id,
            node_id,
            episode_id=episode.episode_id,
            graph_target=episode.graph_target,
        )
    except EpisodeNotRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _, control = _experiment_control_for_target(
        store,
        project_id,
        state,
        node_id,
        graph_target=episode.graph_target,
    )
    return control


# Register this after ``.../watchers/stop``: ``{node_id:path}`` is greedy, so
# the stop-loop route would otherwise swallow that path.
@router.post("/api/projects/{project_id}/experiments/{node_id:path}/stop")
def stop_experiment_loop(
    project_id: str,
    node_id: str,
    request: Request,
    episode_id: str | None = Query(default=None),
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityAccessDependency,
    experiment_operation_lock: ExperimentOperationLockDependency,
) -> dict[str, object]:
    """Finish the current turn, then disable automatic continuation.

    The stop is durable before this returns, so no unclaimed watcher can win a
    wake afterwards. It never cancels the live task, kills external work,
    deletes a watcher, or changes what the Experiment means, and calling it
    again changes nothing.
    """

    identity_access.require_patch_capable_identity(request)
    with experiment_operation_lock(project_id):
        if episode_id is None:
            runtime = store.experiment_loop_runtime(
                project_id,
                node_id,
            )
            episode = store.episode(runtime.episode_id) if runtime.episode_id is not None else None
            if episode is None or episode.project_id != project_id:
                raise HTTPException(status_code=404, detail="Experiment episode not found")
            if episode.graph_target != GraphTargetRef():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This Experiment runs on an episode graph branch; select its exact "
                        "episode before stopping the loop."
                    ),
                )
        else:
            episode = _episode_for_http(store, catalog, project_id, episode_id)
            if episode.control_node_id != node_id:
                raise HTTPException(status_code=404, detail="Experiment episode not found")
        control = stop_bound_experiment_episode(
            project_id,
            episode,
            store=store,
            catalog=catalog,
        )
    return control.model_dump(mode="json")


__all__ = [
    "router",
    "run_experiment",
    "stop_bound_experiment_episode",
    "stop_experiment_loop",
    "stop_experiment_watchers",
]
