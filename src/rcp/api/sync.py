from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from rcp.api.dependencies import (
    get_catalog,
    get_identity_access,
    get_project_display_cache,
    get_project_service,
    get_store,
    get_watcher_delivery,
    project_write_admission,
    require_project_membership,
)
from rcp.api.identity import IdentityAccess
from rcp.core.transition_models import GraphTargetRef
from rcp.core.transitions import current_project_projection
from rcp.history import PatchRejected, RevisionConflict
from rcp.projects import ProjectCatalog, ProjectDisplayCache
from rcp.service import GraphSyncRequest, NodeEditConflict
from rcp.storage import AppStore
from rcp.watchers import WatcherDelivery

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
IdentityDependency = Annotated[IdentityAccess, Depends(get_identity_access)]
StoreDependency = Annotated[AppStore, Depends(get_store)]
DisplayCacheDependency = Annotated[ProjectDisplayCache, Depends(get_project_display_cache)]
WatcherDeliveryDependency = Annotated[WatcherDelivery, Depends(get_watcher_delivery)]


@router.post("/api/projects/{project_id}/sync")
def sync_graph(
    project_id: str,
    body: GraphSyncRequest,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    project_display_cache: DisplayCacheDependency,
    watcher_delivery: WatcherDeliveryDependency,
):
    authorized_by = identity_access.require_patch_capable_identity(request)
    try:
        with project_write_admission(project_id, request) as canonical_project_id:
            service = get_project_service(catalog, canonical_project_id)
            transition = service.sync_graph_transition(
                body,
                active_control_node_ids=store.active_experiment_control_ids(
                    canonical_project_id,
                    graph_target=GraphTargetRef(),
                ),
                authorized_by=authorized_by,
            )
            if transition is None:
                current = service.history.current_materialization()
                head = service.history.head_ref(current)
                projection = current_project_projection(
                    current.state,
                    transition_id=head.transition_id,
                    target=head.target,
                )
            else:
                projection = transition.projection
            state = projection.graph
            payload = state.model_dump(mode="json")
            payload.update(
                project_display_cache.transition_payload(
                    canonical_project_id,
                    projection,
                    reconcile_operational=True,
                )
            )
        watcher_delivery.evaluate_graph_wake_boundary(
            canonical_project_id,
            state,
            source="human Sync",
        )
        return payload
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Missing graph object: {exc.args[0]}") from exc
    except NodeEditConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PatchRejected:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/projects/{project_id}/sync/preview")
def preview_graph_sync(
    project_id: str,
    body: GraphSyncRequest,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    project_display_cache: DisplayCacheDependency,
):
    authorized_by = identity_access.require_patch_capable_identity(request)
    service = get_project_service(catalog, project_id)
    try:
        prepared = service.preview_sync_graph(
            body,
            active_control_node_ids=store.active_experiment_control_ids(
                project_id,
                graph_target=GraphTargetRef(),
            ),
            authorized_by=authorized_by,
        )
        assert prepared.patch.transition is not None
        return {
            "projection": project_display_cache.transition_payload(
                project_id,
                prepared.projection,
                reconcile_operational=False,
            ),
            "transition": prepared.patch.transition.model_dump(mode="json"),
        }
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Missing graph object: {exc.args[0]}",
        ) from exc
    except RevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NodeEditConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PatchRejected:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


__all__ = [
    "preview_graph_sync",
    "router",
    "sync_graph",
]
