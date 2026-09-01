from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from fastapi import HTTPException, Request

from rcp.agents import AgentLauncher
from rcp.api.identity import IdentityAccess
from rcp.attachments import ChatAttachmentStore
from rcp.background import BackgroundAgentTasks
from rcp.keyed_locks import ExperimentAdmission, KeyedLocks
from rcp.projects import ProjectCatalog, ProjectDisplayCache
from rcp.server_ops.backup import BackupArchiveReceipt
from rcp.server_ops.doctor import ServerDoctorReport
from rcp.server_runtime import ServerMetadata
from rcp.service import ProjectService
from rcp.setup import ProjectSetupManager
from rcp.storage import AppStore, SpaceKind
from rcp.watchers import WatcherDelivery, WatcherPoller


@dataclass(frozen=True, slots=True)
class HealthComposition:
    """Startup identity and mode values exposed to the health route."""

    instance_metadata: ServerMetadata
    agent_mode: Literal["acceptance", "provider"]
    default_project_name: str | None
    space_id: str
    space_kind: SpaceKind


@dataclass(frozen=True, slots=True)
class ServerStatusComposition:
    """Concrete read owners used by the read-only Server Settings route."""

    doctor_reader: Callable[[], ServerDoctorReport]
    protected_backup_reader: Callable[[ServerDoctorReport], BackupArchiveReceipt | None]
    restore_completed_at_reader: Callable[[], datetime | None]
    clock: Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class ApiServices:
    """Composition-only runtime services exposed to API dependencies."""

    store: AppStore
    catalog: ProjectCatalog
    identity_access: IdentityAccess
    attachment_store: ChatAttachmentStore
    watcher_poller: WatcherPoller
    result_view_keep_locks: KeyedLocks
    project_display_cache: ProjectDisplayCache
    watcher_delivery: WatcherDelivery
    experiment_operation_lock: KeyedLocks
    background_tasks: BackgroundAgentTasks
    experiment_admission: ExperimentAdmission
    launcher: AgentLauncher
    setup: ProjectSetupManager
    health_composition: HealthComposition
    server_status_composition: ServerStatusComposition


def _api_services(request: Request) -> ApiServices:
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, ApiServices):
        raise RuntimeError("API services have not been configured.")
    return services


def get_store(request: Request) -> AppStore:
    return _api_services(request).store


def get_catalog(request: Request) -> ProjectCatalog:
    return _api_services(request).catalog


def get_identity_access(request: Request) -> IdentityAccess:
    return _api_services(request).identity_access


def get_launcher(request: Request) -> AgentLauncher:
    return _api_services(request).launcher


def get_setup(request: Request) -> ProjectSetupManager:
    return _api_services(request).setup


def get_health_composition(request: Request) -> HealthComposition:
    return _api_services(request).health_composition


def get_server_status_composition(request: Request) -> ServerStatusComposition:
    return _api_services(request).server_status_composition


def get_attachment_store(request: Request) -> ChatAttachmentStore:
    return _api_services(request).attachment_store


def get_watcher_poller(request: Request) -> WatcherPoller:
    return _api_services(request).watcher_poller


def get_result_view_keep_locks(request: Request) -> KeyedLocks:
    return _api_services(request).result_view_keep_locks


def get_project_display_cache(request: Request) -> ProjectDisplayCache:
    return _api_services(request).project_display_cache


def get_watcher_delivery(request: Request) -> WatcherDelivery:
    return _api_services(request).watcher_delivery


def get_experiment_operation_lock(request: Request) -> KeyedLocks:
    return _api_services(request).experiment_operation_lock


def get_background_tasks(request: Request) -> BackgroundAgentTasks:
    return _api_services(request).background_tasks


def get_experiment_admission(request: Request) -> ExperimentAdmission:
    return _api_services(request).experiment_admission


def get_project_service(catalog: ProjectCatalog, project_id: str) -> ProjectService:
    try:
        return catalog.open(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def require_registered_project(catalog: ProjectCatalog, project_id: str) -> None:
    try:
        catalog.card(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


def require_project_membership(project_id: str, request: Request) -> str:
    canonical = get_catalog(request).resolve_project_id(project_id)
    member = get_identity_access(request).acting_user(request)
    if not get_store(request).is_project_member(canonical, member.user_id):
        # A refusal is indistinguishable from an unknown project. A 403 would
        # confirm the project exists, which is the one thing a non-member must
        # not learn.
        raise HTTPException(status_code=404, detail="Project not found")
    return canonical


@contextmanager
def project_write_admission(project_id: str, request: Request) -> Iterator[str]:
    """Hold the one process-local admission fence across a human mutation."""

    canonical = get_catalog(request).resolve_project_id(project_id)
    with get_experiment_operation_lock(request)(canonical):
        try:
            get_store(request).require_project_accepts_new_work(canonical)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        yield canonical


def require_project_write_admission(project_id: str, request: Request) -> Iterator[str]:
    """Dependency form of the project write-admission fence."""

    with project_write_admission(project_id, request) as canonical:
        yield canonical


__all__ = [
    "ApiServices",
    "HealthComposition",
    "ServerStatusComposition",
    "get_attachment_store",
    "get_background_tasks",
    "get_catalog",
    "get_identity_access",
    "get_launcher",
    "get_experiment_operation_lock",
    "get_experiment_admission",
    "get_health_composition",
    "get_server_status_composition",
    "get_project_service",
    "get_project_display_cache",
    "get_result_view_keep_locks",
    "get_setup",
    "get_store",
    "get_watcher_delivery",
    "get_watcher_poller",
    "project_write_admission",
    "require_registered_project",
    "require_project_membership",
    "require_project_write_admission",
]
