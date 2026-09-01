from __future__ import annotations

import asyncio
import hashlib
import threading
import uuid
from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from rcp.api.dependencies import (
    get_attachment_store,
    get_background_tasks,
    get_catalog,
    get_experiment_admission,
    get_identity_access,
    get_project_service,
    get_result_view_keep_locks,
    get_store,
    require_project_membership,
    require_project_write_admission,
    require_registered_project,
)
from rcp.api.identity import IdentityAccess
from rcp.api.task_requests import _resolved_auto_research_request, _resolved_graph_request
from rcp.artifacts import (
    ARTIFACT_MEDIA_TYPES,
    AgentArtifactDescriptor,
    artifact_viewer_document,
    descriptor_for,
    html_preview_document,
    read_local_regular_file,
    validate_artifact_bytes,
)
from rcp.attachments import ChatAttachmentStore
from rcp.background import AgentTaskRequest, BackgroundAgentTasks
from rcp.core.models import Experiment
from rcp.keyed_locks import ExperimentAdmission, KeyedLocks
from rcp.limits import CHAT_ARTIFACT_MAX_FILE_BYTES
from rcp.projects import ProjectCatalog
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.chat import _logical_chat_turn_operation_id
from rcp.runs.task_policy import load_stored_request, task_graph_capable
from rcp.runs.tasks.coach import _resolved_coach_request
from rcp.service import CoachRequest, ProjectService, RunRequest
from rcp.skill_registry import SkillSelection
from rcp.storage import AgentTaskKind, AgentTaskRecord, AppStore
from rcp.transport import RemoteRunStage, StateUnavailable
from rcp.transport.state import StateWorkspace

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
IdentityDependency = Annotated[IdentityAccess, Depends(get_identity_access)]
StoreDependency = Annotated[AppStore, Depends(get_store)]
AttachmentStoreDependency = Annotated[ChatAttachmentStore, Depends(get_attachment_store)]
BackgroundTasksDependency = Annotated[BackgroundAgentTasks, Depends(get_background_tasks)]
ExperimentAdmissionDependency = Annotated[
    ExperimentAdmission,
    Depends(get_experiment_admission),
]
ResultViewKeepLocksDependency = Annotated[
    KeyedLocks,
    Depends(get_result_view_keep_locks),
]


class RetryAgentTaskRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    reasoning: str | None = None
    run_on: str | None = None


class AgentArtifactResponse(AgentArtifactDescriptor):
    """One stored descriptor plus backend-owned availability decisions."""

    available: bool
    unavailable_reason: str | None
    can_open: bool
    can_download: bool
    can_keep: bool
    can_revise: bool


def _agent_artifact_response(
    record: AgentTaskRecord,
    descriptor: AgentArtifactDescriptor,
) -> AgentArtifactResponse:
    kept = descriptor.kept_filename is not None
    retained_stage = bool(record.stage_root) and not record.history_only
    available = kept or retained_stage
    unavailable_reason = (
        None
        if available
        else "Artifact bytes were not retained with this task history."
        if record.history_only
        else "Artifact bytes are no longer available."
    )
    return AgentArtifactResponse(
        **descriptor.model_dump(mode="python"),
        available=available,
        unavailable_reason=unavailable_reason,
        can_open=available,
        can_download=available,
        can_keep=available and not kept and not record.history_only,
        can_revise=(
            available
            and not record.history_only
            and bool(record.native_session_id)
            and bool(record.stage_root)
        ),
    )


def _agent_task_response(record: AgentTaskRecord) -> dict[str, object]:
    response = record.model_dump(mode="json")
    result = response.get("result")
    stored_artifacts = record.result.get("artifacts") if record.result else None
    if not isinstance(result, dict):
        return response
    if record.history_only:
        graph_update = result.get("graph_update")
        if isinstance(graph_update, dict):
            graph_update["repairable"] = False
        graph_updates = result.get("graph_updates")
        if isinstance(graph_updates, list):
            for update in graph_updates:
                if isinstance(update, dict):
                    update["repairable"] = False
    if not isinstance(stored_artifacts, list):
        return response
    projected: list[object] = []
    for raw in stored_artifacts:
        try:
            descriptor = AgentArtifactDescriptor.model_validate(raw)
        except (TypeError, ValueError):
            projected.append(raw)
            continue
        projected.append(_agent_artifact_response(record, descriptor).model_dump(mode="json"))
    result["artifacts"] = projected
    return response


def _reject_history_only_control(record: AgentTaskRecord) -> None:
    if record.history_only:
        raise HTTPException(
            status_code=409,
            detail="This task is retained as history and cannot be controlled or continued.",
        )


@router.post(
    "/api/projects/{project_id}/tasks/{kind}",
    status_code=202,
    dependencies=[Depends(require_project_write_admission)],
)
def start_agent_task(
    project_id: str,
    kind: AgentTaskKind,
    body: dict[str, object],
    http_request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    attachment_store: AttachmentStoreDependency,
    background_tasks: BackgroundTasksDependency,
    result_view_keep_locks: ResultViewKeepLocksDependency,
) -> dict[str, object]:
    if kind in {"auto_research", "branch_merge", "episode_report"}:
        raise HTTPException(
            status_code=405,
            detail="Use the project episode endpoint for Auto-research and branch merge.",
        )
    authorized_by = identity_access.require_patch_capable_identity(http_request)
    service = get_project_service(catalog, project_id)
    admission_lock: threading.Lock | None = None
    result_view_stage_host: str | None = None
    result_view_stage_root: str | None = None
    try:
        request = _validated_task_request(service, kind, body)
        if isinstance(request, RunRequest):
            if request.result_view is not None and request.result_view.action == "revise":
                admission_lock = result_view_keep_locks(request.result_view.view_id)
                admission_lock.acquire()
            request = _admit_result_view_request(
                store,
                service,
                project_id,
                kind,
                request,
            )
            request = _admit_artifact_context_request(
                store,
                service,
                project_id,
                kind,
                request,
            )
            if request.result_view is not None and request.result_view.action == "revise":
                view = store.result_view(request.result_view.view_id)
                if view is None:
                    raise ValueError("The result view is unavailable or expired.")
                result_view_stage_host = view.stage_host or None
                result_view_stage_root = view.stage_root
        if kind in {"node_chat", "project_chat"}:
            assert isinstance(request, RunRequest)
            assert request.chat_id is not None
            if store.has_resumable_paused_chat_task(
                project_id,
                kind,
                request.chat_id,
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This conversation has a paused turn. Resume or retry it before "
                        "starting a new turn."
                    ),
                )
        operation_id = str(uuid.uuid4())
        claimed_set: tuple[str, str] | None = None
        if kind in {"node_chat", "project_chat"}:
            assert isinstance(request, RunRequest)
            supplied = (request.attachment_set_id, request.attachment_client_id)
            if any(supplied) and not all(supplied):
                raise ValueError(
                    "Chat attachments require both attachment_set_id and attachment_client_id."
                )
            if request.attachment_set_id and request.attachment_client_id:
                assert request.chat_id is not None
                claimed = attachment_store.claim(
                    project_id=project_id,
                    chat_id=request.chat_id,
                    client_id=request.attachment_client_id,
                    attachment_set_id=request.attachment_set_id,
                    operation_id=operation_id,
                )
                claimed_set = (claimed.attachment_batch_id, operation_id)
                request = request.model_copy(
                    update={
                        "attachment_set_id": None,
                        "attachment_client_id": None,
                        "attachment_batch_id": claimed.attachment_batch_id,
                        "attachments": claimed.attachments,
                    }
                )
        try:
            record = background_tasks.start(
                project_id,
                kind,
                request,
                operation_id=operation_id,
                authorized_by=authorized_by,
                stage_host=result_view_stage_host,
                stage_root=result_view_stage_root,
            )
        except BaseException:
            if claimed_set is not None and store.agent_task(operation_id) is None:
                attachment_store.release(*claimed_set)
            raise
    except ValueError as exc:
        status = 409 if "already running" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    finally:
        if admission_lock is not None:
            admission_lock.release()
    return record.model_dump(mode="json")


@router.get("/api/projects/{project_id}/tasks")
def agent_tasks(
    project_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> list[dict[str, object]]:
    require_registered_project(catalog, project_id)
    return [_agent_task_response(record) for record in store.agent_tasks(project_id)]


@router.get("/api/projects/{project_id}/tasks/{operation_id}")
def agent_task(
    project_id: str,
    operation_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> dict[str, object]:
    require_registered_project(catalog, project_id)
    record = store.agent_task(operation_id)
    if record is None or record.project_id != project_id or not record.visible:
        raise HTTPException(status_code=404, detail="Agent task not found")
    detail = _agent_task_response(record)
    detail["events"] = [
        event.model_dump(mode="json") for event in store.agent_task_events(operation_id)
    ]
    detail["debug_receipts"] = [
        receipt.model_dump(mode="json") for receipt in store.agent_task_receipts(operation_id)
    ]
    detail["contracts"] = [
        contract.model_dump(mode="json") for contract in store.agent_task_contracts(operation_id)
    ]
    return detail


@router.get("/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/content")
@router.head("/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/content")
async def content_agent_artifact(
    project_id: str,
    operation_id: str,
    artifact_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> Response:
    service = get_project_service(catalog, project_id)
    descriptor, data = await asyncio.to_thread(
        _load_agent_artifact,
        store,
        service.history.workspace,
        project_id,
        operation_id,
        artifact_id,
        "open",
    )
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if descriptor.media_type == "text/html":
        try:
            document, csp = html_preview_document(data)
        except Exception as exc:
            # Rendering is an optional preview boundary. A malformed document
            # or renderer defect makes only this attachment unavailable.
            raise HTTPException(status_code=410, detail="Preview unavailable") from exc
        headers["Content-Security-Policy"] = csp
        return Response(
            b"" if request.method == "HEAD" else document,
            media_type="text/html",
            headers=headers,
        )
    headers["Content-Security-Policy"] = "default-src 'none'; sandbox"
    return Response(
        b"" if request.method == "HEAD" else data,
        media_type=descriptor.media_type,
        headers=headers,
    )


@router.get("/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/preview")
@router.head("/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/preview")
async def preview_agent_artifact(
    project_id: str,
    operation_id: str,
    artifact_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> Response:
    """Keep the old desktop route on the unified shell after source updates."""

    # Retained clients also used this URL as the src for small inline images.
    # A browser image request is distinguishable from a viewer navigation by
    # its Accept header, so keep that bounded compatibility without restoring
    # the old raw-preview entrance for ordinary navigation.
    if "image/" in request.headers.get("accept", "").casefold():
        return await content_agent_artifact(
            project_id,
            operation_id,
            artifact_id,
            request,
            catalog=catalog,
            store=store,
        )
    return await _artifact_viewer_response(
        project_id,
        operation_id,
        artifact_id,
        catalog=catalog,
        store=store,
        head=request.method == "HEAD",
    )


@router.get("/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/download")
@router.head("/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/download")
async def download_agent_artifact(
    project_id: str,
    operation_id: str,
    artifact_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> Response:
    service = get_project_service(catalog, project_id)
    descriptor, data = await asyncio.to_thread(
        _load_agent_artifact,
        store,
        service.history.workspace,
        project_id,
        operation_id,
        artifact_id,
        "download",
    )
    suffix = Path(descriptor.name).suffix.casefold()
    fallback = f"artifact{suffix}" if suffix in ARTIFACT_MEDIA_TYPES else "artifact"
    disposition = (
        f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(descriptor.name, safe='')}"
    )
    return Response(
        b"" if request.method == "HEAD" else data,
        media_type=descriptor.media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": disposition,
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/viewer")
async def view_agent_artifact(
    project_id: str,
    operation_id: str,
    artifact_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> Response:
    return await _artifact_viewer_response(
        project_id,
        operation_id,
        artifact_id,
        catalog=catalog,
        store=store,
    )


async def _artifact_viewer_response(
    project_id: str,
    operation_id: str,
    artifact_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    head: bool = False,
) -> Response:
    service = get_project_service(catalog, project_id)
    descriptor, _ = await asyncio.to_thread(
        _load_agent_artifact,
        store,
        service.history.workspace,
        project_id,
        operation_id,
        artifact_id,
        "open",
    )
    record = store.agent_task(operation_id)
    chat_id = record.request.get("chat_id") if record is not None else None
    if not isinstance(chat_id, str):
        raise HTTPException(status_code=410, detail="Artifact chat unavailable")
    content_url = (
        f"/api/projects/{quote(project_id, safe='')}/tasks/{quote(operation_id, safe='')}"
        f"/artifacts/{quote(artifact_id, safe='')}/content"
    )
    keep_url = (
        f"/api/projects/{quote(project_id, safe='')}/tasks/{quote(operation_id, safe='')}"
        f"/artifacts/{quote(artifact_id, safe='')}/keep"
        if descriptor.can_keep
        else None
    )
    document, csp = artifact_viewer_document(
        preview_url=content_url,
        keep_url=keep_url,
        project_id=project_id,
        chat_id=chat_id if descriptor.can_revise else None,
        operation_id=operation_id,
        descriptor=descriptor,
    )
    return Response(
        b"" if head else document,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": csp,
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/api/projects/{project_id}/tasks/{operation_id}/artifacts/{artifact_id}/keep",
    dependencies=[Depends(require_project_write_admission)],
)
async def keep_agent_artifact(
    project_id: str,
    operation_id: str,
    artifact_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> dict[str, object]:
    service = get_project_service(catalog, project_id)
    descriptor, data = await asyncio.to_thread(
        _load_agent_artifact,
        store,
        service.history.workspace,
        project_id,
        operation_id,
        artifact_id,
        "keep",
    )
    project_name = catalog.card(project_id)["name"]
    if not isinstance(project_name, str):
        raise HTTPException(status_code=503, detail="Artifact Keep unavailable")
    try:
        kept_filename = await asyncio.to_thread(
            service.history.workspace.keep_artifact,
            source_name=descriptor.name,
            project_name=project_name,
            data=data,
        )
        kept = store.mark_agent_artifact_kept(
            operation_id,
            artifact_id,
            kept_filename=kept_filename,
            kept_at=store.now(),
        )
    except (FileNotFoundError, OSError, StateUnavailable, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Artifact Keep unavailable") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    updated = store.agent_task(operation_id)
    assert updated is not None
    return _agent_artifact_response(updated, kept).model_dump(mode="json")


@router.post("/api/projects/{project_id}/tasks/{operation_id}/pause", status_code=202)
def pause_agent_task(
    project_id: str,
    operation_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    background_tasks: BackgroundTasksDependency,
) -> dict[str, object]:
    get_project_service(catalog, project_id)
    record = store.agent_task(operation_id)
    if record is None or record.project_id != project_id or not record.visible:
        raise HTTPException(status_code=404, detail="Agent task not found")
    _reject_history_only_control(record)
    try:
        return background_tasks.pause(operation_id).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/projects/{project_id}/tasks/{operation_id}/resume",
    status_code=202,
    dependencies=[Depends(require_project_write_admission)],
)
def resume_agent_task(
    project_id: str,
    operation_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    background_tasks: BackgroundTasksDependency,
    experiment_admission: ExperimentAdmissionDependency,
    result_view_keep_locks: ResultViewKeepLocksDependency,
) -> dict[str, object]:
    previous = store.agent_task(operation_id)
    if previous is None or previous.project_id != project_id or not previous.visible:
        raise HTTPException(status_code=404, detail="Agent task not found")
    _reject_history_only_control(previous)
    if previous.kind == "branch_merge":
        raise HTTPException(
            status_code=409,
            detail="Dispatch a new Merge to main task from the episode detail.",
        )
    authorized_by = identity_access.require_patch_capable_identity(request)
    service = get_project_service(catalog, project_id)
    result_view_resume_lock: threading.Lock | None = None
    try:
        if previous.kind not in {"paper_coach", "auto_research"}:
            stored_request = load_stored_request(
                RunRequest, previous.request, operation_id=previous.operation_id
            )
            if (
                stored_request.result_view is not None
                and stored_request.result_view.action == "revise"
            ):
                result_view_resume_lock = result_view_keep_locks(stored_request.result_view.view_id)
                result_view_resume_lock.acquire()
                _admit_result_view_request(
                    store,
                    service,
                    project_id,
                    previous.kind,
                    stored_request,
                )
        experiment_admission.require_current(service, previous.request)
        skills = _validate_stored_task_request(service, previous.kind, previous.request)
        return background_tasks.resume(
            operation_id,
            skills=skills,
            authorized_by=authorized_by,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        if result_view_resume_lock is not None:
            result_view_resume_lock.release()


@router.post(
    "/api/projects/{project_id}/tasks/{operation_id}/repair-graph-update",
    status_code=202,
    dependencies=[Depends(require_project_write_admission)],
)
def repair_agent_task_graph_update(
    project_id: str,
    operation_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    background_tasks: BackgroundTasksDependency,
    experiment_admission: ExperimentAdmissionDependency,
) -> dict[str, object]:
    previous = store.agent_task(operation_id)
    if previous is None or previous.project_id != project_id or not previous.visible:
        raise HTTPException(status_code=404, detail="Agent task not found")
    _reject_history_only_control(previous)
    if previous.kind == "branch_merge":
        raise HTTPException(
            status_code=409,
            detail="Dispatch a new Merge to main task from the episode detail.",
        )
    authorized_by = (
        identity_access.require_patch_capable_identity(request)
        if task_graph_capable(previous.kind, previous.request)
        else None
    )
    service = get_project_service(catalog, project_id)
    try:
        experiment_admission.require_current(service, previous.request)
        return background_tasks.repair_graph_update(
            operation_id,
            authorized_by=authorized_by,
        ).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent task not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/projects/{project_id}/tasks/{operation_id}/retry",
    status_code=202,
    dependencies=[Depends(require_project_write_admission)],
)
def retry_agent_task(
    project_id: str,
    operation_id: str,
    request: Request,
    body: RetryAgentTaskRequest | None = None,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    background_tasks: BackgroundTasksDependency,
    experiment_admission: ExperimentAdmissionDependency,
    result_view_keep_locks: ResultViewKeepLocksDependency,
) -> dict[str, object]:
    previous = store.agent_task(operation_id)
    if previous is None or previous.project_id != project_id or not previous.visible:
        raise HTTPException(status_code=404, detail="Agent task not found")
    _reject_history_only_control(previous)
    if previous.kind == "branch_merge":
        raise HTTPException(
            status_code=409,
            detail="Dispatch a new Merge to main task from the episode detail.",
        )
    authorized_by = identity_access.require_patch_capable_identity(request)
    service = get_project_service(catalog, project_id)
    result_view_retry_lock: threading.Lock | None = None
    try:
        overrides = body.model_dump(exclude_none=True) if body is not None else {}
        if previous.request.get("patch_kind") == "experiment_loop" and "run_on" in overrides:
            raise ValueError("Experiment-loop recovery cannot change its pinned execution machine.")
        if previous.kind == "auto_research":
            candidate = load_stored_request(
                AutoResearchRunRequest,
                {**previous.request, **overrides},
                operation_id=previous.operation_id,
            )
        else:
            request_type = CoachRequest if previous.kind == "paper_coach" else RunRequest
            candidate = load_stored_request(
                request_type,
                {**previous.request, **overrides, "session_id": None},
                operation_id=previous.operation_id,
            )
        if (
            isinstance(candidate, RunRequest)
            and candidate.result_view is not None
            and candidate.result_view.action == "revise"
        ):
            result_view_retry_lock = result_view_keep_locks(candidate.result_view.view_id)
            result_view_retry_lock.acquire()
            _admit_result_view_request(
                store,
                service,
                project_id,
                previous.kind,
                candidate,
            )
        candidate_payload = candidate.model_dump(mode="json")
        experiment_admission.require_current(service, candidate_payload)
        skills = _validate_stored_task_request(
            service,
            previous.kind,
            candidate_payload,
        )
        if previous.kind == "auto_research":
            _require_auto_research_retry_target_ready(
                service,
                AutoResearchRunRequest.model_validate(candidate),
            )
        return background_tasks.retry(
            operation_id,
            skills=skills,
            authorized_by=authorized_by,
            **overrides,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        if result_view_retry_lock is not None:
            result_view_retry_lock.release()


def _load_agent_artifact(
    store: AppStore,
    workspace: StateWorkspace,
    project_id: str,
    operation_id: str,
    artifact_id: str,
    action: Literal["open", "download", "keep"],
) -> tuple[AgentArtifactResponse, bytes]:
    """Resolve an attachment only through its persisted task descriptor and stage."""
    record = store.agent_task(operation_id)
    if (
        record is None
        or record.project_id != project_id
        or record.kind not in {"node_chat", "project_chat"}
    ):
        raise HTTPException(status_code=404, detail="Agent task not found")
    descriptor = _agent_artifact_descriptor(record, artifact_id)
    projected = _agent_artifact_response(record, descriptor)
    allowed = {
        "open": projected.can_open,
        "download": projected.can_download,
        "keep": projected.can_keep,
    }[action]
    if not allowed:
        raise HTTPException(
            status_code=410 if action in {"open", "download"} else 409,
            detail=projected.unavailable_reason or f"Artifact {action} unavailable",
        )
    try:
        scope_id = _logical_chat_turn_operation_id(store, record.operation_id)
        expected_descriptor = descriptor_for(scope_id, descriptor.name)
        if (
            expected_descriptor.artifact_id != descriptor.artifact_id
            or expected_descriptor.name != descriptor.name
            or expected_descriptor.media_type != descriptor.media_type
        ):
            raise ValueError("artifact descriptor does not match its task scope")
        if descriptor.kept_filename is not None:
            data = workspace.read_kept_artifact(
                descriptor.kept_filename,
                max_bytes=CHAT_ARTIFACT_MAX_FILE_BYTES,
            )
        elif not record.stage_root:
            raise FileNotFoundError(descriptor.name)
        elif record.stage_host:
            stage = RemoteRunStage(record.stage_host).attach_artifact_source(record.stage_root)
            data = stage.read_artifact_bytes(
                scope_id,
                descriptor.name,
                max_bytes=CHAT_ARTIFACT_MAX_FILE_BYTES,
            )
        else:
            directory = Path(record.stage_root) / "turns" / scope_id / "artifacts"
            data = read_local_regular_file(
                directory,
                descriptor.name,
                max_bytes=CHAT_ARTIFACT_MAX_FILE_BYTES,
            )
        media_type = validate_artifact_bytes(descriptor.name, data)
        if media_type != descriptor.media_type:
            raise ValueError("artifact media type changed")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="Preview unavailable") from exc
    except StateUnavailable as exc:
        raise HTTPException(status_code=503, detail="Preview unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=410, detail="Preview unavailable") from exc
    return projected, data


def _agent_artifact_descriptor(
    record: AgentTaskRecord,
    artifact_id: str,
) -> AgentArtifactDescriptor:
    artifacts = record.result.get("artifacts") if record.result else None
    if isinstance(artifacts, list):
        for raw in artifacts:
            try:
                descriptor = AgentArtifactDescriptor.model_validate(raw)
            except (TypeError, ValueError):
                continue
            if descriptor.artifact_id == artifact_id:
                return descriptor
    raise HTTPException(status_code=404, detail="Artifact not found")


def _admit_result_view_request(
    store: AppStore,
    service: ProjectService,
    project_id: str,
    kind: AgentTaskKind,
    request: RunRequest,
) -> RunRequest:
    intent = request.result_view
    if intent is None:
        return request
    if (
        kind != "node_chat"
        or request.chat_scope != "node"
        or request.mode != "work"
        or request.trigger != "human"
        or request.patch_kind != "work"
        or request.control_node_id is not None
        or request.watcher_ids
    ):
        raise ValueError("Result views require an ordinary node Work turn.")
    if request.node_id is None or not isinstance(
        service.history.state().nodes.get(request.node_id),
        Experiment,
    ):
        raise ValueError("Result views require an Experiment node.")
    if intent.action == "create":
        return request

    record = store.result_view(intent.view_id)
    if record is None or record.project_id != project_id:
        raise ValueError("The result view is unavailable or expired.")
    if record.kept_filename is not None:
        raise ValueError("A kept result view cannot be revised.")
    if record.experiment_id != request.node_id or record.chat_id != request.chat_id:
        raise ValueError("The result view does not belong to this Experiment conversation.")

    pinned = RunRequest.model_validate(
        {
            **request.model_dump(mode="python"),
            "provider": record.provider,
            "model": record.model,
            "reasoning": record.reasoning,
            "run_on": record.run_on,
            "session_id": record.native_session_id,
        }
    )
    return _resolved_graph_request(service, kind, pinned)


def _admit_artifact_context_request(
    store: AppStore,
    service: ProjectService,
    project_id: str,
    kind: AgentTaskKind,
    request: RunRequest,
) -> RunRequest:
    context = request.artifact_context
    if context is None:
        return request
    if request.result_view is not None or kind not in {"node_chat", "project_chat"}:
        raise ValueError("Artifact context belongs to one ordinary chat turn.")
    origin = store.agent_task(context.operation_id)
    if context.source == "episode_report":
        report = store.episode_report(context.episode_id or "")
        wrapup = store.episode_wrapup(context.episode_id or "")
        expected_artifact_id = (
            hashlib.sha256(report.report_id.encode("utf-8")).hexdigest()[:24]
            if report is not None
            else None
        )
        if (
            report is None
            or wrapup is None
            or wrapup.concluding_operation_id != context.operation_id
            or expected_artifact_id != context.artifact_id
            or origin is None
            or origin.project_id != project_id
            or origin.request.get("chat_id") != request.chat_id
        ):
            raise ValueError("The episode report does not belong to this chat.")
        descriptor = None
    else:
        descriptor = None
    if (
        origin is None
        or origin.project_id != project_id
        or origin.kind != kind
        or origin.request.get("chat_id") != request.chat_id
        or origin.request.get("node_id") != request.node_id
    ):
        raise ValueError("The artifact does not belong to this chat.")
    artifacts = origin.result.get("artifacts") if origin and origin.result else None
    if context.source == "task" and isinstance(artifacts, list):
        for raw in artifacts:
            try:
                candidate = AgentArtifactDescriptor.model_validate(raw)
            except (TypeError, ValueError):
                continue
            if candidate.artifact_id == context.artifact_id:
                descriptor = candidate
                break
    if context.source == "task" and descriptor is None:
        raise ValueError("The artifact is unavailable.")
    if context.source == "task":
        assert descriptor is not None
        artifact = _agent_artifact_response(origin, descriptor)
        if not artifact.can_revise:
            raise ValueError(
                artifact.unavailable_reason
                or "The artifact's native session is unavailable. Start a fresh session "
                "explicitly before asking about it."
            )
    pinned_values = {
        "provider": origin.request.get("provider"),
        "model": origin.request.get("model"),
        "reasoning": origin.request.get("reasoning"),
        "run_on": origin.request.get("run_on"),
        "session_id": origin.native_session_id,
    }
    required_values = (
        pinned_values["provider"],
        pinned_values["reasoning"],
        pinned_values["run_on"],
        pinned_values["session_id"],
    )
    if not all(isinstance(value, str) and value for value in required_values) or not isinstance(
        pinned_values["model"], str
    ):
        raise ValueError(
            "The artifact's native session is unavailable. Start a fresh session explicitly "
            "before asking about it."
        )
    pinned = RunRequest.model_validate({**request.model_dump(mode="python"), **pinned_values})
    return _resolved_graph_request(service, kind, pinned)


def _validated_task_request(
    service: ProjectService,
    kind: AgentTaskKind,
    body: dict[str, object],
) -> AgentTaskRequest:
    if kind == "paper_coach":
        return _resolved_coach_request(service, CoachRequest.model_validate(body))

    request = RunRequest.model_validate(body).model_copy(
        update={
            "trigger": "human",
            "patch_kind": "work",
            "control_node_id": None,
            "control_revision": None,
            "control_episode_id": None,
            "control_invocation": None,
            "control_invocation_ceiling": None,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
            "watcher_ids": [],
            "attachment_batch_id": None,
            "attachments": [],
        }
    )
    if request.result_view is not None:
        raise ValueError(
            "Result views are ordinary task artifacts now. Ask the chat to create or revise "
            "the artifact through the unified viewer."
        )
    if kind in {"seed", "refresh"}:
        service.history.require_writable()
        if request.session_id:
            raise ValueError(
                "Seed and refresh sessions can only be resumed from an RCP background "
                "task checkpoint."
            )
        return _resolved_graph_request(service, kind, request)

    chat_scope: Literal["node", "project"] = "node" if kind == "node_chat" else "project"
    request = request.model_copy(
        update={
            "chat_scope": chat_scope,
            "node_id": request.node_id if chat_scope == "node" else None,
        }
    )
    if not request.message or not request.message.strip() or not request.chat_id:
        raise ValueError("Chat requires a chat_id and message")
    if chat_scope == "node":
        if not request.node_id:
            raise ValueError("Node chat requires a node_id")
        if request.node_id not in service.history.state().nodes:
            raise HTTPException(status_code=404, detail="Node not found")
    try:
        uuid.UUID(request.chat_id)
    except ValueError as exc:
        raise ValueError("chat_id must be a UUID") from exc
    # Artifact-context admission resolves the exact execution profile and native
    # session recorded by the originating turn. Do not first resolve transient
    # settings from the currently open chat; stale settings must not prevent a
    # valid origin-session continuation.
    if request.artifact_context is not None:
        return request
    return _resolved_graph_request(service, kind, request)


def _validate_stored_task_request(
    service: ProjectService,
    kind: AgentTaskKind,
    body: dict[str, object],
) -> SkillSelection | None:
    """Validate a stored request and return any package-selection refresh it needs."""

    if kind == "auto_research":
        auto_research_request = AutoResearchRunRequest.model_validate(body)
        resolved_auto_research = _resolved_auto_research_request(
            service,
            auto_research_request,
        )
        return service.resolve_skill_selection(cast(RunRequest, resolved_auto_research))
    if kind == "paper_coach":
        resolved_coach = _resolved_coach_request(service, CoachRequest.model_validate(body))
        return service.resolve_skill_selection(resolved_coach)
    request = RunRequest.model_validate(body)
    if kind in {"seed", "refresh"}:
        service.history.require_writable()
    resolved_run = _resolved_graph_request(service, kind, request)
    return service.resolve_skill_selection(resolved_run)


def _require_auto_research_retry_target_ready(
    service: ProjectService,
    request: AutoResearchRunRequest,
) -> None:
    """Recheck the pinned provider target before Retry can allocate a child task."""

    if request.provider is None or request.run_on is None:
        raise ValueError("Auto-research Retry requires its pinned provider and execution machine.")
    machine = service.manifest.machine_map.get(request.run_on)
    if machine is None:
        raise ValueError(f"unknown execution machine: {request.run_on}")
    binary = machine.provider_paths.get(request.provider)
    readiness = service.launcher.readiness(
        request.provider,
        host=machine.host,
        binary=binary,
        refresh=True,
    )
    if readiness.installed and readiness.authenticated:
        return
    diagnostic = (
        readiness.reason or f"{request.provider} is not ready on {request.run_on}"
    ).strip()
    if diagnostic.endswith("."):
        diagnostic = diagnostic[:-1]
    raise ValueError(
        f"Auto-research Retry cannot start: {diagnostic}. The current task was left unchanged."
    )


__all__ = [
    "RetryAgentTaskRequest",
    "agent_task",
    "agent_tasks",
    "content_agent_artifact",
    "download_agent_artifact",
    "pause_agent_task",
    "preview_agent_artifact",
    "repair_agent_task_graph_update",
    "resume_agent_task",
    "retry_agent_task",
    "router",
    "start_agent_task",
    "view_agent_artifact",
]
