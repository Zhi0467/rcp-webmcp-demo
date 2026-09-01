from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from rcp.api.dependencies import (
    get_attachment_store,
    get_catalog,
    get_project_service,
    require_project_membership,
    require_registered_project,
)
from rcp.attachments import ChatAttachmentStore, ChatAttachmentUpload
from rcp.limits import CHAT_PAGE_DEFAULT_LIMIT, CHAT_PAGE_MAX_LIMIT
from rcp.projects import ProjectCatalog
from rcp.service import ChatSummaryPage, ChatTranscript

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
AttachmentStoreDependency = Annotated[ChatAttachmentStore, Depends(get_attachment_store)]


@router.post(
    "/api/projects/{project_id}/chats/{chat_id}/attachments",
    response_model=ChatAttachmentUpload,
)
def upload_chat_attachment(
    project_id: str,
    chat_id: str,
    file: Annotated[UploadFile, File()],
    client_id: Annotated[str, Form()],
    attachment_set_id: Annotated[str | None, Form()] = None,
    *,
    catalog: CatalogDependency,
    attachment_store: AttachmentStoreDependency,
) -> ChatAttachmentUpload:
    require_registered_project(catalog, project_id)
    try:
        return attachment_store.add(
            project_id=project_id,
            chat_id=chat_id,
            client_id=client_id,
            filename=file.filename or "",
            media_type=file.content_type,
            source=file.file,
            attachment_set_id=attachment_set_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        file.file.close()


@router.delete(
    "/api/projects/{project_id}/chats/{chat_id}/attachments/{attachment_id}",
)
def remove_chat_attachment(
    project_id: str,
    chat_id: str,
    attachment_id: str,
    client_id: str,
    attachment_set_id: str,
    *,
    catalog: CatalogDependency,
    attachment_store: AttachmentStoreDependency,
) -> dict[str, bool]:
    require_registered_project(catalog, project_id)
    try:
        attachment_store.remove(
            project_id=project_id,
            chat_id=chat_id,
            client_id=client_id,
            attachment_set_id=attachment_set_id,
            attachment_id=attachment_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"removed": True}


@router.get(
    "/api/projects/{project_id}/chats",
    response_model=ChatSummaryPage,
)
def chats(
    project_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(
        default=CHAT_PAGE_DEFAULT_LIMIT,
        ge=1,
        le=CHAT_PAGE_MAX_LIMIT,
    ),
    *,
    catalog: CatalogDependency,
) -> ChatSummaryPage:
    service = get_project_service(catalog, project_id)
    return service.chat_summaries(offset=offset, limit=limit)


@router.get(
    "/api/projects/{project_id}/chats/{chat_id}",
    response_model=ChatTranscript,
)
def chat(
    project_id: str,
    chat_id: str,
    *,
    catalog: CatalogDependency,
) -> ChatTranscript:
    service = get_project_service(catalog, project_id)
    try:
        transcript = service.chat_transcript(chat_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if transcript is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return transcript


__all__ = [
    "chat",
    "chats",
    "remove_chat_attachment",
    "router",
    "upload_chat_attachment",
]
