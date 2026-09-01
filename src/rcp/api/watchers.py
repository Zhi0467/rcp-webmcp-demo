from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from rcp.api.dependencies import (
    get_catalog,
    get_store,
    get_watcher_poller,
    require_project_membership,
    require_registered_project,
)
from rcp.projects import ProjectCatalog
from rcp.storage import AppStore, WatcherClaimConflict
from rcp.watchers import WatcherPoller

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
StoreDependency = Annotated[AppStore, Depends(get_store)]
WatcherPollerDependency = Annotated[WatcherPoller, Depends(get_watcher_poller)]


@router.get("/api/projects/{project_id}/watchers")
def project_watchers(
    project_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> list[dict[str, object]]:
    require_registered_project(catalog, project_id)
    return [record.model_dump(mode="json") for record in store.watchers(project_id)]


@router.post("/api/projects/{project_id}/watchers/{watcher_id}/check")
def check_watcher_now(
    project_id: str,
    watcher_id: str,
    *,
    catalog: CatalogDependency,
    watcher_poller: WatcherPollerDependency,
) -> dict[str, object]:
    require_registered_project(catalog, project_id)
    try:
        watcher = watcher_poller.check_now(project_id, watcher_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Watcher not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return watcher.model_dump(mode="json")


@router.post("/api/projects/{project_id}/watchers/{watcher_id}/stop")
def stop_watcher(
    project_id: str,
    watcher_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> dict[str, object]:
    require_registered_project(catalog, project_id)
    watcher = store.watcher(watcher_id)
    if watcher is None or watcher.project_id != project_id:
        raise HTTPException(status_code=404, detail="Watcher not found")
    if watcher.continuation.patch_kind == "experiment_loop":
        raise HTTPException(
            status_code=409,
            detail="Use Stop loop to stop an Experiment loop and its watchers gracefully.",
        )
    try:
        stopped = store.stop_watchers(project_id, [watcher_id])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Watcher not found") from exc
    except WatcherClaimConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return stopped[0].model_dump(mode="json")


__all__ = [
    "check_watcher_now",
    "project_watchers",
    "router",
    "stop_watcher",
]
