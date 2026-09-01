from __future__ import annotations

import hashlib
import logging
from functools import partial
from typing import Annotated, Literal, cast
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

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
from rcp.api.episodes import (
    EpisodeMessageBody,
    EpisodeResponse,
    ReauthorizeEpisodeBody,
    StartEpisodeBody,
    _episode_for_http,
    serialize_episode,
    serialize_episodes,
)
from rcp.api.experiments import stop_bound_experiment_episode
from rcp.api.identity import IdentityAccess
from rcp.artifacts import AgentArtifactDescriptor, artifact_viewer_document, html_preview_document
from rcp.background import BackgroundAgentTasks
from rcp.keyed_locks import KeyedLocks
from rcp.projects import ProjectCatalog
from rcp.runs.auto_research import AutoResearchStartRequest, settle_auto_research_stop
from rcp.runs.auto_research_admission import (
    start_auto_research,
    stop_auto_research,
)
from rcp.runs.auto_research_delivery import (
    deliver_pending_auto_research_lifecycle,
    deliver_pending_auto_research_mail,
    record_auto_research_message,
)
from rcp.runs.branch_merge_admission import start_branch_merge
from rcp.runs.branch_merge_request import BranchMergeRunRequest
from rcp.service import ProjectService, RunRequest
from rcp.storage import AppStore, AutoResearchMessageRecord, EpisodeNotRunning

from .episode_branches import (
    ensure_auto_research_graph_target,
    graph_branch_summaries,
    graph_branch_summary,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_project_membership)])

CatalogDependency = Annotated[ProjectCatalog, Depends(get_catalog)]
IdentityDependency = Annotated[IdentityAccess, Depends(get_identity_access)]
StoreDependency = Annotated[AppStore, Depends(get_store)]
BackgroundTasksDependency = Annotated[BackgroundAgentTasks, Depends(get_background_tasks)]
ExperimentOperationLockDependency = Annotated[
    KeyedLocks,
    Depends(get_experiment_operation_lock),
]


def _branch_summaries(
    store: AppStore,
    catalog: ProjectCatalog,
) -> partial:
    return partial(graph_branch_summaries, store=store, catalog=catalog)


def _branch_summary(
    store: AppStore,
    catalog: ProjectCatalog,
) -> partial:
    return partial(graph_branch_summary, store=store, catalog=catalog)


@router.get(
    "/api/projects/{project_id}/episodes",
    response_model=list[EpisodeResponse],
)
def episodes(
    project_id: str,
    mode: Literal["auto_research", "experiment_loop"] | None = None,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> list[EpisodeResponse]:
    require_registered_project(catalog, project_id)
    return serialize_episodes(
        store,
        project_id,
        mode=mode,
        branch_summaries=_branch_summaries(store, catalog),
    )


@router.post(
    "/api/projects/{project_id}/episodes",
    response_model=EpisodeResponse,
    status_code=202,
    dependencies=[Depends(require_project_write_admission)],
)
def start_episode(
    project_id: str,
    body: StartEpisodeBody,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    background_tasks: BackgroundTasksDependency,
) -> EpisodeResponse:
    authorized_by = identity_access.require_patch_capable_identity(request)
    service = get_project_service(catalog, project_id)
    try:
        start_request = _resolved_auto_research_start_request(service, body)
        service.history.require_writable()
        graph_base_head = service.history.head_ref()
        episode, _ = start_auto_research(
            background_tasks,
            project_id,
            start_request,
            authorized_by=authorized_by,
            graph_base_head=graph_base_head,
            ensure_graph_target=partial(
                ensure_auto_research_graph_target,
                catalog=catalog,
            ),
        )
        return serialize_episode(
            store,
            project_id,
            episode,
            branch_summary=_branch_summary(store, catalog),
        )
    except ValueError as exc:
        live = any(
            episode.mode == "auto_research"
            and episode.status in {"queued", "running", "stopping", "wrapping_up"}
            for episode in store.episodes(project_id)
        )
        status = 409 if live else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post(
    "/api/projects/{project_id}/episodes/{episode_id}/stop",
    response_model=EpisodeResponse,
)
def stop_episode(
    project_id: str,
    episode_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    background_tasks: BackgroundTasksDependency,
    experiment_operation_lock: ExperimentOperationLockDependency,
) -> EpisodeResponse:
    identity_access.require_patch_capable_identity(request)
    episode = _episode_for_http(store, catalog, project_id, episode_id)
    if episode.mode == "auto_research":
        try:
            stop_auto_research(background_tasks, episode.episode_id)
            settle_auto_research_stop(store, episode.episode_id)
        except EpisodeNotRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    elif episode.mode == "experiment_loop":
        with experiment_operation_lock(project_id):
            stop_bound_experiment_episode(
                project_id,
                episode,
                store=store,
                catalog=catalog,
            )
    else:
        raise HTTPException(status_code=409, detail="This episode cannot be stopped.")
    current = store.episode(episode.episode_id)
    if current is None:
        raise RuntimeError("The stopped episode could not be reloaded.")
    return serialize_episode(
        store,
        project_id,
        current,
        branch_summary=_branch_summary(store, catalog),
    )


@router.post(
    "/api/projects/{project_id}/episodes/{episode_id}/merge",
    response_model=EpisodeResponse,
    status_code=202,
    dependencies=[Depends(require_project_write_admission)],
)
def merge_episode_branch(
    project_id: str,
    episode_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    background_tasks: BackgroundTasksDependency,
) -> EpisodeResponse:
    authorized_by = identity_access.require_patch_capable_identity(request)
    episode = _episode_for_http(store, catalog, project_id, episode_id)
    if episode.mode != "auto_research" or episode.graph_target.kind != "branch":
        raise HTTPException(
            status_code=409,
            detail="Only an Auto-research graph branch can merge to main.",
        )
    service = get_project_service(catalog, project_id)
    try:
        summary = graph_branch_summary(episode, store=store, catalog=catalog)
        if not summary.merge_eligible:
            raise ValueError(
                "This graph branch is active, unchanged, already merged, or otherwise "
                "not merge eligible."
            )
        service.history.require_writable()
        merge_request = _resolved_branch_merge_request(service, episode.episode_id)
        start_branch_merge(
            background_tasks,
            project_id,
            merge_request,
            authorized_by=authorized_by,
        )
        current = store.episode(episode.episode_id)
        if current is None:
            raise RuntimeError("The branch merge episode could not be reloaded.")
        return serialize_episode(
            store,
            project_id,
            current,
            branch_summary=_branch_summary(store, catalog),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/projects/{project_id}/episodes/{episode_id}/reauthorize",
    response_model=EpisodeResponse,
    status_code=202,
    dependencies=[Depends(require_project_write_admission)],
)
def reauthorize_episode(
    project_id: str,
    episode_id: str,
    body: ReauthorizeEpisodeBody,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    background_tasks: BackgroundTasksDependency,
) -> EpisodeResponse:
    authorized_by = identity_access.require_patch_capable_identity(request)
    episode = _episode_for_http(store, catalog, project_id, episode_id)
    if not (
        episode.mode == "auto_research"
        and episode.status == "needs_action"
        and episode.ending == "exhausted"
        and episode.wrapup_state in {"ready", "failed", "legacy_unavailable"}
    ):
        raise HTTPException(
            status_code=409,
            detail="Only an exhausted, settled Auto-research episode can be reauthorized.",
        )
    state = store.auto_research_state(episode.episode_id)
    if state is None:
        raise HTTPException(status_code=409, detail="Auto-research state is unavailable.")
    service = get_project_service(catalog, project_id)
    try:
        service.history.require_writable()
        start_request = _resolved_auto_research_start_request(
            service,
            StartEpisodeBody(
                mode="auto_research",
                invocation_ceiling=body.invocation_ceiling,
                starting_instruction=state.starting_instruction,
            ),
        )
        graph_base_head = service.history.head_ref()
        fresh, _ = start_auto_research(
            background_tasks,
            project_id,
            start_request,
            authorized_by=authorized_by,
            graph_base_head=graph_base_head,
            ensure_graph_target=partial(
                ensure_auto_research_graph_target,
                catalog=catalog,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return serialize_episode(
        store,
        project_id,
        fresh,
        branch_summary=_branch_summary(store, catalog),
    )


@router.get(
    "/api/projects/{project_id}/episodes/{episode_id}/messages",
    response_model=list[AutoResearchMessageRecord],
)
def episode_messages(
    project_id: str,
    episode_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> list[AutoResearchMessageRecord]:
    episode = _episode_for_http(store, catalog, project_id, episode_id)
    if episode.mode != "auto_research":
        raise HTTPException(status_code=409, detail="This episode has no Auto-research mail.")
    return store.auto_research_messages(episode.episode_id)


@router.post(
    "/api/projects/{project_id}/episodes/{episode_id}/messages",
    response_model=AutoResearchMessageRecord,
    status_code=201,
    dependencies=[Depends(require_project_write_admission)],
)
def send_episode_message(
    project_id: str,
    episode_id: str,
    body: EpisodeMessageBody,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    identity_access: IdentityDependency,
    background_tasks: BackgroundTasksDependency,
) -> AutoResearchMessageRecord:
    authorized_by = identity_access.require_patch_capable_identity(request)
    episode = _episode_for_http(store, catalog, project_id, episode_id)
    if episode.mode != "auto_research":
        raise HTTPException(status_code=409, detail="This episode has no Auto-research mail.")
    if episode.status != "running" or episode.ending is not None:
        raise HTTPException(status_code=409, detail="Episode is not accepting new mail")
    if episode.root_operation_id is None:
        raise HTTPException(status_code=409, detail="Episode orchestrator is unavailable")
    try:
        saved = record_auto_research_message(
            store,
            episode_id=episode.episode_id,
            sender_role="human",
            sender_task_id=None,
            authorized_by=authorized_by,
            recipient_task_id=episode.root_operation_id,
            body=body.body,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        started = deliver_pending_auto_research_lifecycle(
            background_tasks,
            episode_id=episode.episode_id,
        )
        if started is None:
            deliver_pending_auto_research_mail(
                background_tasks,
                episode_id=episode.episode_id,
                recipient_task_id=episode.root_operation_id,
            )
    except Exception as exc:
        logger.warning(
            "Could not deliver durable Auto-research message %s immediately: %s",
            saved.message_id,
            exc,
        )
    current = store.auto_research_message(saved.message_id)
    if current is None:
        raise RuntimeError("The durable episode message could not be reloaded.")
    return current


@router.get("/api/projects/{project_id}/episodes/{episode_id}/report/content")
@router.head("/api/projects/{project_id}/episodes/{episode_id}/report/content")
def content_episode_report(
    project_id: str,
    episode_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> Response:
    episode = _episode_for_http(store, catalog, project_id, episode_id)
    report = None if episode.ending == "stopped" else store.episode_report(episode.episode_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Episode report not found")
    try:
        document, csp = html_preview_document(
            report.html.encode("utf-8"),
        )
    except (UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=410, detail="Episode report unavailable") from exc
    encoded = document.encode("utf-8")
    return Response(
        b"" if request.method == "HEAD" else encoded,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "Content-Length": str(len(encoded)),
            "Content-Security-Policy": csp,
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/api/projects/{project_id}/episodes/{episode_id}/report/preview")
@router.head("/api/projects/{project_id}/episodes/{episode_id}/report/preview")
def preview_episode_report(
    project_id: str,
    episode_id: str,
    request: Request,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> Response:
    """Keep the old desktop route on the unified shell after source updates."""

    return _episode_report_viewer_response(
        project_id,
        episode_id,
        catalog=catalog,
        store=store,
        head=request.method == "HEAD",
    )


@router.get("/api/projects/{project_id}/episodes/{episode_id}/report/viewer")
def view_episode_report(
    project_id: str,
    episode_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
) -> Response:
    return _episode_report_viewer_response(
        project_id,
        episode_id,
        catalog=catalog,
        store=store,
    )


def _episode_report_viewer_response(
    project_id: str,
    episode_id: str,
    *,
    catalog: CatalogDependency,
    store: StoreDependency,
    head: bool = False,
) -> Response:
    episode = _episode_for_http(store, catalog, project_id, episode_id)
    report = None if episode.ending == "stopped" else store.episode_report(episode.episode_id)
    wrapup = store.episode_wrapup(episode.episode_id)
    if report is None or wrapup is None or wrapup.concluding_operation_id is None:
        raise HTTPException(status_code=404, detail="Episode report not found")
    origin = store.agent_task(wrapup.concluding_operation_id)
    chat_id = origin.request.get("chat_id") if origin is not None else None
    if not isinstance(chat_id, str):
        chat_id = None
    artifact_id = hashlib.sha256(report.report_id.encode("utf-8")).hexdigest()[:24]
    descriptor = AgentArtifactDescriptor(
        artifact_id=artifact_id,
        name="episode-report.html",
        media_type="text/html",
        size_bytes=len(report.html.encode("utf-8")),
    )
    content_url = (
        f"/api/projects/{quote(project_id, safe='')}/episodes/"
        f"{quote(episode_id, safe='')}/report/content"
    )
    document, csp = artifact_viewer_document(
        preview_url=content_url,
        keep_url=None,
        project_id=project_id,
        chat_id=chat_id,
        operation_id=wrapup.concluding_operation_id,
        descriptor=descriptor,
        source="episode_report",
        episode_id=episode_id,
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


def _resolved_auto_research_start_request(
    service: ProjectService,
    body: StartEpisodeBody,
) -> AutoResearchStartRequest:
    profile = service.resolve_agent_profile("orchestrator")
    request = AutoResearchStartRequest(
        invocation_ceiling=body.invocation_ceiling,
        starting_instruction=body.starting_instruction,
        provider=profile.provider,
        model=profile.model,
        reasoning=profile.reasoning,
        run_on=profile.run_on,
        run_truth_scope=list(service.manifest.agent.default_run_truth_scope),
    )
    resolved = service.resolve_skill_request(cast(RunRequest, request))
    if not isinstance(resolved, AutoResearchStartRequest):
        raise TypeError("Auto-research skill resolution changed the start request type.")
    return resolved


def _resolved_branch_merge_request(
    service: ProjectService,
    episode_id: str,
) -> BranchMergeRunRequest:
    profile = service.resolve_agent_profile("orchestrator")
    return BranchMergeRunRequest(
        episode_id=episode_id,
        provider=profile.provider,
        model=profile.model,
        reasoning=profile.reasoning,
        run_on=profile.run_on,
        run_truth_scope=sorted(set(service.manifest.agent.default_run_truth_scope)),
        chat_scope="project",
        mode="work",
        trigger="human",
        patch_kind="work",
    )


__all__ = [
    "content_episode_report",
    "episode_messages",
    "episodes",
    "merge_episode_branch",
    "preview_episode_report",
    "reauthorize_episode",
    "router",
    "send_episode_message",
    "start_episode",
    "stop_episode",
    "view_episode_report",
]
