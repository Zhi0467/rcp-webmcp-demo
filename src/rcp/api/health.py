from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from rcp import __version__
from rcp.api.dependencies import (
    HealthComposition,
    get_catalog,
    get_health_composition,
    get_store,
)
from rcp.api.project_provisioning import project_creation_control
from rcp.projects import ProjectCatalog
from rcp.storage import AppStore

router = APIRouter()

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
HealthCompositionDependency = Annotated[
    HealthComposition,
    Depends(get_health_composition),
]
StoreDependency = Annotated[AppStore, Depends(get_store)]


@router.get("/api/health")
async def health(
    *,
    catalog: CatalogDependency,
    composition: HealthCompositionDependency,
    store: StoreDependency,
) -> dict[str, object]:
    with store.connection() as connection:
        active_agent_tasks = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM graph_runs
                WHERE status IN ('queued', 'running', 'pausing')
                """
            ).fetchone()[0]
        )
    payload: dict[str, object] = {
        "status": "ok",
        "version": __version__,
        "space_id": composition.space_id,
        "space_kind": composition.space_kind,
        "space_name": store.space_name,
        "instance_id": composition.instance_metadata.instance_id,
        "pid": composition.instance_metadata.pid,
        "data_dir_id": composition.instance_metadata.data_dir_id,
        "owner_kind": composition.instance_metadata.owner_kind,
        "running_commit": composition.instance_metadata.running_commit,
        "web_build_id": composition.instance_metadata.web_build_id,
        "active_agent_tasks": active_agent_tasks,
        "projects": len(catalog.cards()),
        "agent_mode": composition.agent_mode,
        "project_creation": project_creation_control(composition.space_kind).model_dump(
            mode="json"
        ),
    }
    if composition.default_project_name is not None:
        payload["project"] = composition.default_project_name
    return payload


__all__ = ["health", "router"]
