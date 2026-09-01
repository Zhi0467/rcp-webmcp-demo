from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from rcp.api.dependencies import (
    get_background_tasks,
    get_catalog,
    get_identity_access,
    get_project_display_cache,
    get_project_service,
    get_store,
    require_project_membership,
    require_project_write_admission,
    require_registered_project,
)
from rcp.api.identity import IdentityAccess
from rcp.background import BackgroundAgentTasks
from rcp.config import load_manifest
from rcp.projects import ProjectCatalog, ProjectDisplayCache
from rcp.providers import profile_for
from rcp.repository_preview import (
    REPOSITORY_PREVIEW_CSP,
    load_repository_source_for_path,
    repository_source_document,
)
from rcp.runs.membership_fence import fence_episodes_for_departed_member
from rcp.service import ProjectSettingsRequest
from rcp.storage import AgentUsageSnapshot, AppStore
from rcp.transport import StateUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
DisplayCacheDependency = Annotated[ProjectDisplayCache, Depends(get_project_display_cache)]
IdentityDependency = Annotated[IdentityAccess, Depends(get_identity_access)]
BackgroundTasksDependency = Annotated[BackgroundAgentTasks, Depends(get_background_tasks)]
StoreDependency = Annotated[AppStore, Depends(get_store)]


class ProjectInviteRequest(BaseModel):
    """Who is being invited. The server derives the inviter from the session."""

    user_id: str


@router.get("/api/projects/{project_id}")
async def project(
    project_id: str,
    *,
    project_display_cache: DisplayCacheDependency,
    catalog: CatalogDependency,
) -> dict[str, object]:
    cached = project_display_cache.cached_project_snapshot(project_id)
    if cached is not None:
        return cached
    try:
        generation = catalog.reserve_cached_snapshot_generation(project_id)
        service, snapshot = await asyncio.to_thread(project_display_cache.open_snapshot, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except (FileNotFoundError, OSError, ValueError, StateUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        committed = catalog.commit_cached_snapshot(
            project_id,
            snapshot,
            generation=generation,
            patch_log_head=service.history.workspace.cached_patch_log_head(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Could not update display snapshot for %s: %s", project_id, exc)
    else:
        if not committed:
            latest = project_display_cache.cached_project_snapshot(project_id)
            if latest is not None:
                return latest
    return snapshot


@router.get("/api/projects/{project_id}/members")
def project_members(
    project_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> list[dict[str, object]]:
    canonical = catalog.resolve_project_id(project_id)
    users = {user.user_id: user for user in store.space_users()}
    return [
        {
            "user_id": record.user_id,
            "display_name": (
                users[record.user_id].display_name if record.user_id in users else None
            ),
            "seated_at": record.seated_at,
        }
        for record in store.project_members(canonical)
    ]


@router.post("/api/projects/{project_id}/invitations", status_code=201)
def invite_project_member(
    project_id: str,
    body: ProjectInviteRequest,
    request: Request,
    *,
    catalog: CatalogDependency,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, object]:
    canonical = catalog.resolve_project_id(project_id)
    # The server derives the inviter from the session; the body names only
    # who is being invited.
    inviter = identity_access.acting_user(request)
    try:
        invitation = store.invite_to_project(
            canonical,
            body.user_id,
            invited_by=inviter.user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return invitation.model_dump(mode="json")


@router.post("/api/projects/{project_id}/leave", status_code=204)
def leave_project(
    project_id: str,
    request: Request,
    *,
    background_tasks: BackgroundTasksDependency,
    catalog: CatalogDependency,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> Response:
    canonical = catalog.resolve_project_id(project_id)
    leaving = identity_access.acting_user(request)
    try:
        store.leave_project(canonical, leaving.user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    fence_episodes_for_departed_member(store, background_tasks, canonical, leaving.user_id)
    return Response(status_code=204)


@router.get("/api/projects/{project_id}/cached")
def cached_project(
    project_id: str,
    *,
    project_display_cache: DisplayCacheDependency,
) -> dict[str, object]:
    snapshot = project_display_cache.cached_project_snapshot(project_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Cached project snapshot not found")
    return snapshot


@router.get("/api/projects/{project_id}/cached/revision")
async def cached_project_revision(
    project_id: str,
    *,
    project_display_cache: DisplayCacheDependency,
) -> dict[str, object]:
    snapshot = project_display_cache.cached_project_snapshot(project_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Cached project snapshot not found")
    project_display_cache.schedule_project_reconciliation(project_id)
    return {
        "revision": snapshot["revision"],
        "snapshot_freshness": snapshot["snapshot_freshness"],
        "last_remote_sync_at": snapshot["last_remote_sync_at"],
    }


@router.get("/api/projects/{project_id}/readiness")
def project_readiness(
    project_id: str,
    refresh: bool = False,
    *,
    catalog: CatalogDependency,
) -> dict[str, object]:
    try:
        return catalog.readiness_snapshot(project_id, refresh=refresh)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/projects/{project_id}/graph")
def graph(
    project_id: str,
    *,
    catalog: CatalogDependency,
) -> dict[str, object]:
    return get_project_service(catalog, project_id).graph_snapshot()


@router.get("/api/projects/{project_id}/revision")
def project_revision(
    project_id: str,
    *,
    catalog: CatalogDependency,
) -> dict[str, int]:
    service = get_project_service(catalog, project_id)
    return {"revision": service.history.current_accepted_revision()}


@router.get("/api/projects/{project_id}/repositories/files/preview")
@router.head("/api/projects/{project_id}/repositories/files/preview")
def preview_repository_file(
    project_id: str,
    request: Request,
    path: str = Query(min_length=1),
    line: int | None = Query(default=None, ge=1),
    *,
    store: StoreDependency,
) -> Response:
    record = store.project(project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        manifest = load_manifest(record.locator)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        source = load_repository_source_for_path(manifest, path)
        document = repository_source_document(source, line=line)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        b"" if request.method == "HEAD" else document,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": REPOSITORY_PREVIEW_CSP,
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.put(
    "/api/projects/{project_id}/settings",
    dependencies=[Depends(require_project_write_admission)],
)
def update_project_settings(
    project_id: str,
    body: ProjectSettingsRequest,
    *,
    project_display_cache: DisplayCacheDependency,
) -> dict[str, object]:
    try:
        snapshot = project_display_cache.update_settings(project_id, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return snapshot


@router.post(
    "/api/projects/{project_id}/machines/{machine_alias}/providers/{provider}/resolve",
    dependencies=[Depends(require_project_write_admission)],
)
def resolve_project_provider_path(
    project_id: str,
    machine_alias: str,
    provider: str,
    *,
    project_display_cache: DisplayCacheDependency,
) -> dict[str, object]:
    try:
        profile_for(provider)
        result = project_display_cache.resolve_provider_path(
            project_id,
            machine_alias,
            provider,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@router.get("/api/projects/{project_id}/sources")
def sources(
    project_id: str,
    refresh: bool = False,
    *,
    catalog: CatalogDependency,
):
    service = get_project_service(catalog, project_id)
    return service.index_snapshot(refresh=refresh).model_dump(mode="json")


@router.delete("/api/projects/{project_id}/caches")
def clear_rebuildable_caches(
    project_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
):
    service = get_project_service(catalog, project_id)
    if store.has_active_agent_task(project_id):
        raise HTTPException(
            status_code=409,
            detail="This project's cache cannot be cleared while its agent task is active.",
        )
    return service.clear_rebuildable_caches()


@router.get("/api/projects/{project_id}/usage", response_model=AgentUsageSnapshot)
def agent_usage(
    project_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> AgentUsageSnapshot:
    require_registered_project(catalog, project_id)
    return store.agent_usage_snapshot(project_id)


__all__ = [
    "ProjectInviteRequest",
    "agent_usage",
    "cached_project",
    "cached_project_revision",
    "clear_rebuildable_caches",
    "graph",
    "invite_project_member",
    "leave_project",
    "preview_repository_file",
    "project",
    "project_members",
    "project_readiness",
    "project_revision",
    "resolve_project_provider_path",
    "router",
    "sources",
    "update_project_settings",
]
