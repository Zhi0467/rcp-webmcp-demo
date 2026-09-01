from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from rcp.api.dependencies import (
    get_catalog,
    get_project_service,
    require_project_membership,
    require_project_write_admission,
)
from rcp.projects import ProjectCatalog

router = APIRouter(dependencies=[Depends(require_project_membership)])


class PaperSaveRequest(BaseModel):
    content: str
    base_hash: str | None = None


@router.get("/api/projects/{project_id}/paper")
def get_paper(
    project_id: str,
    catalog: Annotated[ProjectCatalog, Depends(get_catalog)],
):
    paper = get_project_service(catalog, project_id).paper
    return paper.snapshot().model_dump(mode="json")


@router.post(
    "/api/projects/{project_id}/paper/create",
    dependencies=[Depends(require_project_write_admission)],
)
def create_paper(
    project_id: str,
    catalog: Annotated[ProjectCatalog, Depends(get_catalog)],
):
    paper = get_project_service(catalog, project_id).paper
    return paper.create().model_dump(mode="json")


@router.put(
    "/api/projects/{project_id}/paper",
    dependencies=[Depends(require_project_write_admission)],
)
def save_paper(
    project_id: str,
    body: PaperSaveRequest,
    catalog: Annotated[ProjectCatalog, Depends(get_catalog)],
):
    paper = get_project_service(catalog, project_id).paper
    return paper.save(body.content, body.base_hash).model_dump(mode="json")


@router.get("/api/projects/{project_id}/paper/sessions")
def paper_sessions(
    project_id: str,
    catalog: Annotated[ProjectCatalog, Depends(get_catalog)],
):
    paper = get_project_service(catalog, project_id).paper
    return [item.model_dump(mode="json") for item in paper.sessions()]


__all__ = [
    "PaperSaveRequest",
    "create_paper",
    "get_paper",
    "paper_sessions",
    "router",
    "save_paper",
]
