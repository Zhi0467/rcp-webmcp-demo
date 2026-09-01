from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from rcp.api.dependencies import (
    get_catalog,
    get_project_service,
    get_result_view_keep_locks,
    get_store,
    require_project_membership,
    require_project_write_admission,
    require_registered_project,
)
from rcp.artifacts import ResultViewDescriptor, html_preview_document
from rcp.keyed_locks import KeyedLocks
from rcp.projects import ProjectCatalog
from rcp.storage import AppStore, ResultViewConflict, ResultViewRecord
from rcp.transport import StateUnavailable

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
StoreDependency = Annotated[AppStore, Depends(get_store)]
ResultViewKeepLocksDependency = Annotated[KeyedLocks, Depends(get_result_view_keep_locks)]


@router.get(
    "/api/projects/{project_id}/result-views",
    response_model=list[ResultViewDescriptor],
)
def result_views(
    project_id: str,
    experiment_id: str | None = None,
    chat_id: str | None = None,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> list[ResultViewDescriptor]:
    require_registered_project(catalog, project_id)
    return store.list_result_view_descriptors(
        project_id,
        experiment_id=experiment_id,
        chat_id=chat_id,
    )


@router.get("/api/projects/{project_id}/result-views/{view_id}/preview")
@router.head("/api/projects/{project_id}/result-views/{view_id}/preview")
async def preview_result_view(
    project_id: str,
    view_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> Response:
    require_registered_project(catalog, project_id)
    _, data = await asyncio.to_thread(
        _load_visible_result_view_bytes,
        store,
        project_id,
        view_id,
    )
    try:
        document, csp = html_preview_document(data, result_view_gestures=True)
    except (UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=410, detail="Result view unavailable") from exc
    encoded = document.encode("utf-8")
    headers = {
        "Cache-Control": "no-store",
        "Content-Length": str(len(encoded)),
        "Content-Security-Policy": csp,
        "X-Content-Type-Options": "nosniff",
    }
    return Response(
        b"" if request.method == "HEAD" else encoded,
        media_type="text/html",
        headers=headers,
    )


@router.post(
    "/api/projects/{project_id}/result-views/{view_id}/keep",
    response_model=ResultViewDescriptor,
    dependencies=[Depends(require_project_write_admission)],
)
def keep_result_view(
    project_id: str,
    view_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    result_view_keep_locks: ResultViewKeepLocksDependency,
) -> ResultViewDescriptor:
    require_registered_project(catalog, project_id)
    with result_view_keep_locks(view_id):
        record = _visible_result_view_record(store, project_id, view_id)
        if record.kept_filename is not None:
            return store.result_view_descriptor(record)
        if store.has_active_result_view_revision(record):
            raise HTTPException(
                status_code=409,
                detail="Wait for the active result view revision before keeping it.",
            )
        data = _read_result_view_bytes_for_http(store, record)
        service = get_project_service(catalog, project_id)
        project_name = catalog.card(project_id)["name"]
        if not isinstance(project_name, str):
            raise HTTPException(status_code=503, detail="Result view Keep unavailable")
        try:
            kept_filename = service.history.workspace.keep_result_view(
                source_name=record.source_name,
                project_name=project_name,
                data=data,
            )
        except (FileNotFoundError, OSError, StateUnavailable, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail="Result view Keep unavailable",
            ) from exc
        try:
            kept = store.mark_result_view_kept(
                view_id,
                expected_content_sha256=record.content_sha256,
                kept_filename=kept_filename,
                kept_at=store.now(),
            )
        except ResultViewConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="The result view changed before Keep completed.",
            ) from exc
        return store.result_view_descriptor(kept)


def _visible_result_view_record(
    store: AppStore,
    project_id: str,
    view_id: str,
) -> ResultViewRecord:
    as_of = datetime.now(UTC)
    record = store.result_view_for_diagnostics(view_id)
    if record is None or record.project_id != project_id:
        raise HTTPException(status_code=404, detail="Result view not found")
    if record.kept_filename is None and store.result_view(view_id, as_of=as_of) is None:
        raise HTTPException(status_code=410, detail="Result view expired")
    return record


def _load_visible_result_view_bytes(
    store: AppStore,
    project_id: str,
    view_id: str,
) -> tuple[ResultViewRecord, bytes]:
    record = _visible_result_view_record(store, project_id, view_id)
    return record, _read_result_view_bytes_for_http(store, record)


def _read_result_view_bytes_for_http(
    store: AppStore,
    record: ResultViewRecord,
) -> bytes:
    try:
        return store.result_view_bytes(
            record.view_id,
            expected_content_sha256=record.content_sha256,
        )
    except (FileNotFoundError, OSError, StateUnavailable) as exc:
        raise HTTPException(status_code=503, detail="Result view storage unavailable") from exc
    except (KeyError, ResultViewConflict, ValueError) as exc:
        raise HTTPException(status_code=410, detail="Result view unavailable") from exc


__all__ = [
    "keep_result_view",
    "preview_result_view",
    "result_views",
    "router",
]
