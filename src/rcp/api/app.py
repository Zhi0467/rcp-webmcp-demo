from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager, suppress
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rcp import __version__
from rcp.agents import AcceptanceAgentLauncher, AgentLauncher, ProviderReadiness
from rcp.agents.command_protocol import SpawnArguments
from rcp.api.chats import router as chats_router
from rcp.api.dependencies import (
    ApiServices,
    HealthComposition,
    ServerStatusComposition,
    require_project_membership,
)
from rcp.api.dependencies import (
    get_project_service as _project_service,
)
from rcp.api.episode_branches import (
    ensure_auto_research_graph_target as _ensure_auto_research_graph_target,
)
from rcp.api.episode_branches import (
    graph_branch_summary as _graph_branch_summary,
)
from rcp.api.episode_routes import router as episode_router
from rcp.api.episodes import (
    _episode_for_http,
    serialize_episode,
)
from rcp.api.experiment_controls import _experiment_control_response
from rcp.api.experiments import router as experiments_router
from rcp.api.health import router as health_router
from rcp.api.history import router as history_router
from rcp.api.identity import IdentityAccess, TrustedPrincipalResolver
from rcp.api.index import membership_router as index_membership_router
from rcp.api.index import router as index_router
from rcp.api.paper import router as paper_router
from rcp.api.project_provisioning import router as project_provisioning_router
from rcp.api.project_state import router as project_state_router
from rcp.api.result_views import router as result_views_router
from rcp.api.server_status import router as server_status_router
from rcp.api.sync import router as sync_router
from rcp.api.task_requests import _resolved_graph_request, resolved_agent_surface
from rcp.api.tasks import router as tasks_router
from rcp.api.team import router as team_router
from rcp.api.watchers import router as watchers_router
from rcp.attachments import ChatAttachmentStore
from rcp.background import (
    AgentTaskExecution,
    AgentTaskRequest,
    BackgroundAgentTasks,
    StartupEffectFence,
)
from rcp.control import admit_experiment_watcher_invocation
from rcp.history import PatchRejected, ReplayHalted
from rcp.keyed_locks import ExperimentAdmission, KeyedLocks
from rcp.limits import (
    SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS,
    SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
    TEAM_PUBLIC_AUTH_REQUEST_MAX_BYTES,
)
from rcp.projects import ProjectCatalog, ProjectDisplayCache
from rcp.provider_skills import ProviderSkillInventoryManager
from rcp.providers import configured_runtime_id
from rcp.runs.auto_research import (
    AutoResearchCommandContext,
    AutoResearchCommandDispatcher,
    AutoResearchCommandEffectResult,
    AutoResearchRunRequest,
)
from rcp.runs.auto_research_admission import (
    reconcile_committed_auto_research_dispatches,
    reconcile_reserved_auto_research_roots,
)
from rcp.runs.auto_research_child_reconcile import (
    reconcile_pending_auto_research_child_admissions,
)
from rcp.runs.auto_research_delivery import (
    deliver_auto_research_watcher_group,
    reconcile_pending_auto_research_lifecycle,
    reconcile_pending_auto_research_mail,
)
from rcp.runs.auto_research_effects import auto_research_command_effects
from rcp.runs.auto_research_experiments import AutoResearchExperimentCoordinator
from rcp.runs.auto_research_recovery import reconcile_orphaned_auto_research_failures
from rcp.runs.branch_merge_request import BranchMergeRunRequest
from rcp.runs.episodes.reconcile import EpisodeReconciler
from rcp.runs.experiment_loop import (
    experiment_watcher_delivery_request,
    preflight_episode_wake,
)
from rcp.runs.shared import _sweep_stale_stages
from rcp.runs.task_policy import task_experiment_episode_id, task_graph_capable
from rcp.runs.tasks.auto_research_child_work import stream_auto_research_child_work_run
from rcp.runs.tasks.auto_research_stream import (
    stream_auto_research_orchestrator_run,
    stream_auto_research_worker_run,
)
from rcp.runs.tasks.branch_merge import stream_branch_merge_task
from rcp.runs.tasks.coach import stream_coach
from rcp.runs.tasks.discuss import stream_discuss_run
from rcp.runs.tasks.episode_report import EpisodeReportRunRequest, stream_episode_report_run
from rcp.runs.tasks.experiment_loop import stream_experiment_loop_task
from rcp.runs.tasks.graph import stream_graph_run
from rcp.runs.tasks.work import _apply_work_patch, _validate_work_patch_live, stream_work_run
from rcp.runs.transition_event_reconciliation import reconcile_accepted_graph_boundaries
from rcp.runs.watcher_admission import start_watcher_notification
from rcp.server_ops.backup import BackupArchiveReceipt, latest_protected_backup_receipt
from rcp.server_ops.backup_capture import BackupCaptureCoordinator
from rcp.server_ops.control import (
    SERVER_CONTROL_OPERATIONS,
    ServerControlBackupCaptureResult,
    ServerControlError,
    ServerControlMemberAdvanceResult,
    ServerControlMemberPlanResult,
    ServerControlPeer,
    ServerControlProbeResult,
    ServerControlProjectPlanResult,
    ServerControlProjectStepResult,
    ServerControlProjectTransferActivationResult,
    ServerControlProjectTransferUploadResult,
    ServerControlProviderCheckResult,
    ServerControlProviderPlanResult,
    ServerControlRequest,
    ServerControlRestoreResult,
    ServerControlServer,
    ServerControlUpdateResult,
)
from rcp.server_ops.doctor import LinuxServerDoctorMachine, ServerDoctorReport
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.members import MemberRemovalCoordinator
from rcp.server_ops.project_provision import ProjectProvisionCoordinator
from rcp.server_ops.provider_readiness import ProviderReadinessCoordinator
from rcp.server_ops.update_cutover import (
    RuntimeAdmissionGate,
    UpdateAdmissionClosed,
    UpdateCutoverRefused,
    UpdateRuntimeBoundary,
    UpdateServiceCoordinator,
    load_update_runtime_boundary,
)
from rcp.server_runtime import ServerMetadata, data_dir_identity, remove_server_metadata
from rcp.service import (
    CoachRequest,
    ProjectService,
    RunRequest,
)
from rcp.setup import ProjectSetupManager
from rcp.sources import (
    REMOTE_SOURCE_CACHE_LIMITS,
    SESSION_SLICE_CACHE_LIMITS,
    RebuildableCache,
    discover_project_cache_roots,
    legacy_shared_cache_roots,
)
from rcp.storage import (
    AgentTaskKind,
    AppStore,
    GraphWatcherRecord,
    StoredWatcherRecord,
    TeamAuthenticationError,
)
from rcp.transfer.target import (
    TargetTransferActivationCoordinator,
    TargetTransferUploadCoordinator,
)
from rcp.transport import RemoteRunStage, StateUnavailable
from rcp.watchers import (
    GraphWatcherRetryRegistry,
    WatcherDelivery,
    WatcherPoller,
    WatcherRetryWorker,
    ready_graph_watcher_groups,
)
from rcp.web_assets import web_dist_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rcp.server_ops.restore import RestoreOperationJournal


def _terminate_uncommitted_restore_process() -> None:
    """Exit cleanly so Restart=on-failure leaves the disabled unit stopped."""

    os.kill(os.getpid(), signal.SIGTERM)


def _installed_rollback_journals(update_root: Path) -> tuple[Path, ...]:
    """Inspect rollback state without importing the rehearsal cycle at module load."""

    from rcp.server_ops.update_checkpoint import (
        UpdateCheckpointRefused,
        unfinished_rollback_journals,
    )

    try:
        return unfinished_rollback_journals(update_root, expected_uid=os.geteuid())
    except (OSError, UpdateCheckpointRefused) as exc:
        raise RuntimeError(
            "Installed update recovery state is unsafe; run sudo rcp server update."
        ) from exc


def inspect_installed_replacement_startup(
    app_data: Path,
    server_layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> tuple[bool, RestoreOperationJournal | None]:
    """Refuse incomplete replacement before a server can touch its data root."""

    installed_layout = app_data.resolve(strict=False) == server_layout.data_dir.resolve(
        strict=False
    )
    if not installed_layout:
        return False, None

    restore_boundary = None
    if server_layout.restore_operations_root.exists():
        from rcp.server_ops.restore import RestoreRefused, unfinished_restore_operation

        try:
            pending_restore = unfinished_restore_operation(
                server_layout,
                expected_uid=os.geteuid(),
            )
        except (OSError, RestoreRefused) as exc:
            raise RuntimeError(
                "Installed restore state is unsafe; keep the service stopped and run sudo rcp "
                "server restore."
            ) from exc
        if pending_restore is not None:
            if pending_restore.phase != "activation_ready":
                raise RuntimeError(
                    "Installed replacement restore is incomplete; run sudo rcp server restore."
                )
            restore_boundary = pending_restore

    if server_layout.update_checkpoints_root.exists():
        pending_rollback_journals = _installed_rollback_journals(
            server_layout.update_checkpoints_root
        )
        if pending_rollback_journals:
            raise RuntimeError(
                "Installed rollback restoration is incomplete; run sudo rcp server update."
            )
        from rcp.server_ops.update_cutover import (
            UpdateCutoverRefused,
            update_operation_needing_recovery,
        )

        try:
            pending_update = update_operation_needing_recovery(
                server_layout.update_checkpoints_root,
                expected_uid=os.geteuid(),
            )
        except (OSError, UpdateCutoverRefused) as exc:
            raise RuntimeError(
                "Installed update recovery state is unsafe; run sudo rcp server update."
            ) from exc
        if pending_update is not None and pending_update[1].state in {
            "rollback_restoring",
            "repair_required",
        }:
            raise RuntimeError(
                "Installed rollback cutover is incomplete; run sudo rcp server update."
            )
    return True, restore_boundary


class TeamPublicAuthBodyLimit:
    """Bound unauthenticated credential requests before JSON parsing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in {
            "/api/team/enroll",
            "/api/team/session/exchange",
        }:
            await self.app(scope, receive, send)
            return

        messages: list[Message] = []
        total = 0
        more_body = True
        while more_body:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                more_body = False
                continue
            total += len(message.get("body", b""))
            if total > TEAM_PUBLIC_AUTH_REQUEST_MAX_BYTES:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": {
                            "code": "team_auth_request_too_large",
                            "message": "The team authentication request is too large.",
                        }
                    },
                )
                await response(scope, receive, send)
                return
            more_body = message.get("more_body", False)

        message_index = 0

        async def replay() -> Message:
            nonlocal message_index
            if message_index >= len(messages):
                return {"type": "http.request", "body": b"", "more_body": False}
            message = messages[message_index]
            message_index += 1
            return message

        await self.app(scope, replay, send)


class _LazyProjectService:
    """Compatibility handle that opens the default project only when inspected."""

    def __init__(self, catalog: ProjectCatalog, project_id: str) -> None:
        object.__setattr__(self, "_catalog", catalog)
        object.__setattr__(self, "_project_id", project_id)

    def _resolve(self) -> ProjectService:
        return self._catalog.open(self._project_id)

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._resolve(), name, value)


def create_app(
    manifest_path: str | None = None,
    data_dir: Path | None = None,
    *,
    instance_metadata: ServerMetadata | None = None,
    acceptance_agent: bool = False,
    trusted_principal_resolver: TrustedPrincipalResolver | None = None,
    startup_effect_fence: StartupEffectFence | None = None,
    server_layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
    server_doctor_reader: Callable[[], ServerDoctorReport] | None = None,
    server_protected_backup_reader: (
        Callable[[ServerDoctorReport], BackupArchiveReceipt | None] | None
    ) = None,
    server_restore_completed_at_reader: Callable[[], datetime | None] | None = None,
    server_status_clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    # macOS exposes /tmp through /private/tmp. Keep every cache and manifest
    # pointer in the same canonical spelling so relative canonical-state paths
    # do not fail after an otherwise successful remote chat.
    app_data = (data_dir or default_data_dir()).expanduser().resolve()
    identity = instance_metadata or ServerMetadata.create(
        app_data,
        host="127.0.0.1",
        port=8421,
        owner_kind="embedded",
    )
    if identity.data_dir_id != data_dir_identity(app_data):
        raise ValueError("Server metadata does not identify this RCP data directory.")
    update_runtime_boundary: UpdateRuntimeBoundary | None = None
    installed_update_layout, restore_runtime_boundary = inspect_installed_replacement_startup(
        app_data,
        server_layout,
    )
    runtime_admission_gate = RuntimeAdmissionGate()
    background_admission_gate = RuntimeAdmissionGate()
    if (
        identity.control_socket is not None
        and identity.running_commit is not None
        and installed_update_layout
    ):
        update_runtime_boundary = load_update_runtime_boundary(
            running_commit=identity.running_commit,
            layout=server_layout,
            expected_uid=os.geteuid(),
        )
        if update_runtime_boundary is not None and restore_runtime_boundary is not None:
            raise RuntimeError(
                "Installed update and restore activation boundaries overlap; keep the service "
                "stopped and inspect both journals."
            )
        if update_runtime_boundary is not None or restore_runtime_boundary is not None:
            admission_reason = (
                "Server restore activation"
                if restore_runtime_boundary is not None
                else "Server update maintenance"
            )
            runtime_admission_gate = RuntimeAdmissionGate(closed=True, reason=admission_reason)
            background_admission_gate = RuntimeAdmissionGate(closed=True, reason=admission_reason)
            if startup_effect_fence is None:
                reason = (
                    f"server restore {restore_runtime_boundary.operation_id}"
                    if restore_runtime_boundary is not None
                    else f"server update {update_runtime_boundary.receipt.operation_id}"
                )
                startup_effect_fence = StartupEffectFence(reason)
    store = AppStore(app_data / "rcp.sqlite3")
    space_id = store.space_id
    space_kind = store.space_kind
    control_server: ServerControlServer | None = None
    provider_readiness_coordinator: ProviderReadinessCoordinator | None = None
    project_provision_coordinator: ProjectProvisionCoordinator | None = None
    target_transfer_upload_coordinator: TargetTransferUploadCoordinator | None = None
    target_transfer_activation_coordinator: TargetTransferActivationCoordinator | None = None
    backup_capture_coordinator: BackupCaptureCoordinator | None = None
    member_removal_coordinator: MemberRemovalCoordinator | None = None
    update_service_coordinator: UpdateServiceCoordinator | None = None
    startup_effect_runtime_event = threading.Event()
    startup_effect_release_error: list[str | None] = [None]
    restore_activation_committed = threading.Event()

    def capture_sqlite_for_control() -> ServerControlBackupCaptureResult:
        assert backup_capture_coordinator is not None
        publication = backup_capture_coordinator.capture_sqlite()
        receipt = publication.receipt
        return ServerControlBackupCaptureResult(
            instance_id=identity.instance_id,
            pid=identity.pid,
            data_dir_id=identity.data_dir_id,
            space_id=space_id,
            capture_id=receipt.capture_id,
            receipt_path=str(publication.receipt_path),
            receipt_sha256=publication.receipt_sha256,
            snapshot_sha256=receipt.sqlite_snapshot.sha256,
            status=receipt.status,
            project_count=len(receipt.projects),
            uncaptured_project_count=sum(
                project.status == "uncaptured" for project in receipt.projects
            ),
        )

    def update_control_result(
        receipt,
        digest: str,
        *,
        capture: ServerControlBackupCaptureResult | None = None,
        verification_sha256: str | None = None,
    ) -> ServerControlUpdateResult:
        if identity.running_commit is None:
            raise ServerControlError(
                "operation_refused",
                "Update maintenance requires one installed running release commit.",
            )
        return ServerControlUpdateResult(
            instance_id=identity.instance_id,
            pid=identity.pid,
            data_dir_id=identity.data_dir_id,
            space_id=space_id,
            operation_id=receipt.operation_id,
            operation_state=receipt.state,
            receipt_sha256=digest,
            running_commit=identity.running_commit,
            capture=capture,
            verification_sha256=verification_sha256,
        )

    def commit_restore_activation(
        *,
        operation_id: str,
        boundary_sha256: str,
    ) -> ServerControlRestoreResult:
        from rcp.server_ops.restore import (
            RestoreActivationProjectReadback,
            RestoreActivationReadback,
            RestoreRefused,
            read_restore_journal,
            restore_activation_boundary,
            write_restore_journal,
        )

        if restore_runtime_boundary is None or startup_effect_fence is None:
            raise ServerControlError(
                "operation_refused",
                "This process does not own a replacement restore activation boundary.",
            )
        try:
            current = read_restore_journal(server_layout, expected_uid=os.geteuid())
        except (OSError, RestoreRefused) as exc:
            raise ServerControlError(
                "operation_refused", "The replacement restore journal is unavailable."
            ) from exc
        if current.operation_id != operation_id or current.phase != "activation_ready":
            raise ServerControlError(
                "operation_refused", "The replacement restore operation or phase changed."
            )
        if restore_activation_boundary(current) != boundary_sha256:
            raise ServerControlError(
                "operation_refused", "The replacement activation boundary changed."
            )
        if not startup_effect_fence.active or startup_effect_fence.attempted_effects:
            raise ServerControlError(
                "operation_refused",
                "The replacement process crossed its startup-effect fence before verification.",
            )
        if not runtime_admission_gate.closed or not background_admission_gate.closed:
            raise ServerControlError(
                "operation_refused", "Replacement admission opened before verification."
            )
        startup = background_tasks.plan_startup_recovery().as_dict()
        if any(startup.values()):
            raise ServerControlError(
                "operation_refused",
                "Detached work still appears recoverable on the replacement server.",
            )
        projects: list[RestoreActivationProjectReadback] = []
        publications = {item.project_id: item for item in current.project_publications}
        for capture in sorted(current.manifest.projects, key=lambda item: item.project_id):
            record = store.project(capture.project_id)
            if record is None:
                raise ServerControlError(
                    "operation_refused", "A restored project disappeared before activation."
                )
            if capture.status == "uncaptured":
                if record.reachable:
                    raise ServerControlError(
                        "operation_refused",
                        "An explicitly uncaptured project became reachable before activation.",
                    )
                continue
            publication = publications.get(capture.project_id)
            if publication is None or not record.reachable:
                raise ServerControlError(
                    "operation_refused", "A captured project failed final activation readback."
                )
            projects.append(
                RestoreActivationProjectReadback(
                    project_id=capture.project_id,
                    main_revision=publication.main_revision,
                    reachable=True,
                )
            )
        if identity.running_commit is None:
            raise ServerControlError(
                "operation_refused", "The installed replacement has no running source commit."
            )
        startup_sha256 = hashlib.sha256(
            json.dumps(
                startup,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        readback = RestoreActivationReadback(
            instance_id=identity.instance_id,
            pid=identity.pid,
            data_dir_id=identity.data_dir_id,
            space_id=space_id,
            running_commit=identity.running_commit,
            startup_recovery_sha256=startup_sha256,
            projects=tuple(projects),
            activated_at=datetime.now(UTC),
        )
        completed = current.model_copy(
            update={
                "phase": "complete",
                "activation_readback": readback,
                "updated_at": datetime.now(UTC),
            }
        )
        write_restore_journal(
            completed,
            server_layout,
            uid=os.geteuid(),
            gid=os.getegid(),
        )
        try:
            background_admission_gate.reopen()
            startup_effect_fence.release()
            if not startup_effect_runtime_event.wait(SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS):
                raise RuntimeError("deferred replacement startup did not complete")
            if startup_effect_release_error[0]:
                raise RuntimeError(startup_effect_release_error[0])
            runtime_admission_gate.reopen()
            restore_activation_committed.set()
        except BaseException as exc:
            write_restore_journal(
                current.model_copy(update={"updated_at": datetime.now(UTC)}),
                server_layout,
                uid=os.geteuid(),
                gid=os.getegid(),
            )
            raise ServerControlError(
                "operation_refused",
                "The verified replacement could not start its deferred runtime.",
            ) from exc
        return ServerControlRestoreResult(
            instance_id=identity.instance_id,
            pid=identity.pid,
            data_dir_id=identity.data_dir_id,
            space_id=space_id,
            operation_id=operation_id,
            boundary_sha256=boundary_sha256,
            readback=readback,
        )

    if identity.control_socket is not None:
        if space_kind != "team" or identity.owner_kind != "cli":
            raise ValueError(
                "A private control socket is available only to an installed CLI-owned team service."
            )

        def dispatch_server_control(
            request: ServerControlRequest,
            _peer: ServerControlPeer,
        ) -> (
            ServerControlProbeResult
            | ServerControlMemberPlanResult
            | ServerControlMemberAdvanceResult
            | ServerControlProviderPlanResult
            | ServerControlProviderCheckResult
            | ServerControlProjectPlanResult
            | ServerControlProjectStepResult
            | ServerControlProjectTransferActivationResult
            | ServerControlProjectTransferUploadResult
            | ServerControlBackupCaptureResult
            | ServerControlUpdateResult
            | ServerControlRestoreResult
        ):
            update_operations = {
                "update_maintenance_enter",
                "update_candidate_verify",
                "update_fence_release",
                "update_maintenance_abort",
            }
            root_operations = {*update_operations, "restore_activation_commit"}
            continuation_operations = {"project_transfer_upload_complete"}
            if request.operation not in {
                "probe",
                *root_operations,
                *continuation_operations,
            }:
                try:
                    background_admission_gate.require_open(
                        f"server-control operation {request.operation}"
                    )
                except UpdateAdmissionClosed as exc:
                    raise ServerControlError("operation_refused", str(exc)) from exc
            if (
                startup_effect_fence is not None
                and startup_effect_fence.active
                and request.operation not in {"probe", *root_operations, *continuation_operations}
            ):
                startup_effect_fence.require_open(f"server-control operation {request.operation}")
            if request.operation in root_operations and _peer.uid != 0:
                raise ServerControlError(
                    "operation_refused",
                    "This maintenance operation requires the root server coordinator.",
                )
            if request.operation in update_operations and update_service_coordinator is None:
                raise ServerControlError(
                    "operation_refused",
                    "This process does not own the installed server update layout.",
                )
            match request.operation:
                case "probe":
                    return ServerControlProbeResult(
                        instance_id=identity.instance_id,
                        pid=identity.pid,
                        data_dir_id=identity.data_dir_id,
                        space_id=space_id,
                        operations=SERVER_CONTROL_OPERATIONS,
                        pending_member_removals=(
                            member_removal_coordinator.pending_snapshots()
                            if member_removal_coordinator is not None
                            else ()
                        ),
                    )
                case "member_removal_plan":
                    assert member_removal_coordinator is not None
                    assert request.selector_id is not None
                    return member_removal_coordinator.plan(request.selector_id)
                case "member_removal_advance":
                    assert member_removal_coordinator is not None
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None
                    return member_removal_coordinator.advance(
                        request.selector_id,
                        boundary_sha256=request.boundary_sha256,
                    )
                case "provider_readiness_plan":
                    assert provider_readiness_coordinator is not None
                    assert request.selector_kind is not None and request.selector_id is not None
                    return provider_readiness_coordinator.plan(
                        request.selector_kind,
                        request.selector_id,
                    )
                case "provider_readiness_check":
                    assert provider_readiness_coordinator is not None
                    assert request.selector_kind is not None and request.selector_id is not None
                    assert request.boundary_sha256 is not None and request.target_id is not None
                    return provider_readiness_coordinator.check(
                        request.selector_kind,
                        request.selector_id,
                        boundary_sha256=request.boundary_sha256,
                        target_id=request.target_id,
                    )
                case "project_provision_plan":
                    assert project_provision_coordinator is not None
                    assert request.selector_id is not None
                    return project_provision_coordinator.plan(request.selector_id)
                case "project_provision_step":
                    assert project_provision_coordinator is not None
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None and request.target_id is not None
                    return project_provision_coordinator.advance(
                        request.selector_id,
                        boundary_sha256=request.boundary_sha256,
                        target_id=request.target_id,
                    )
                case "project_transfer_upload_plan":
                    assert target_transfer_upload_coordinator is not None
                    assert request.selector_id is not None
                    try:
                        return target_transfer_upload_coordinator.plan(request.selector_id)
                    except ValueError as exc:
                        raise ServerControlError("operation_refused", str(exc)) from exc
                case "project_transfer_upload_complete":
                    assert target_transfer_upload_coordinator is not None
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None
                    try:
                        return target_transfer_upload_coordinator.complete(
                            request.selector_id,
                            lease_boundary_sha256=request.boundary_sha256,
                        )
                    except ValueError as exc:
                        raise ServerControlError("operation_refused", str(exc)) from exc
                case "project_transfer_activate":
                    assert target_transfer_activation_coordinator is not None
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None
                    try:
                        activation = target_transfer_activation_coordinator.activate(
                            request.selector_id,
                            lease_boundary_sha256=request.boundary_sha256,
                        )
                    except ValueError as exc:
                        raise ServerControlError("operation_refused", str(exc)) from exc
                    return ServerControlProjectTransferActivationResult(
                        instance_id=identity.instance_id,
                        pid=identity.pid,
                        data_dir_id=identity.data_dir_id,
                        space_id=space_id,
                        target_request_id=activation.target_request_id,
                        source_request_id=activation.source_request_id,
                        project_id=activation.project_id,
                        archive_sha256=activation.archive_sha256,
                        upload_lease_boundary_sha256=(activation.upload_lease_boundary_sha256),
                        archive_manifest_sha256=activation.archive_manifest_sha256,
                        target_manifest_sha256=activation.target_manifest_sha256,
                        publication_sha256=activation.publication_sha256,
                        activated_at=activation.activated_at,
                    )
                case "backup_sqlite_capture":
                    return capture_sqlite_for_control()
                case "restore_activation_commit":
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None
                    return commit_restore_activation(
                        operation_id=request.selector_id,
                        boundary_sha256=request.boundary_sha256,
                    )
                case "update_maintenance_enter":
                    assert update_service_coordinator is not None
                    assert target_transfer_upload_coordinator is not None
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None
                    # Control requests are dispatched serially. Refuse before
                    # closing admission so this same loop can accept the active
                    # upload's completion continuation.
                    if not target_transfer_upload_coordinator.uploads_idle():
                        raise ServerControlError(
                            "operation_refused",
                            "An active project transfer upload must complete before server "
                            "update maintenance can begin.",
                        )
                    try:
                        receipt, digest, capture = update_service_coordinator.enter_maintenance(
                            operation_id=request.selector_id,
                            receipt_sha256=request.boundary_sha256,
                            timeout=SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS,
                        )
                    except UpdateCutoverRefused as exc:
                        raise ServerControlError("operation_refused", str(exc)) from exc
                    return update_control_result(receipt, digest, capture=capture)
                case "update_candidate_verify":
                    assert update_service_coordinator is not None
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None
                    try:
                        receipt, digest, verification = (
                            update_service_coordinator.verify_running_release(
                                operation_id=request.selector_id,
                                receipt_sha256=request.boundary_sha256,
                            )
                        )
                    except UpdateCutoverRefused as exc:
                        raise ServerControlError("operation_refused", str(exc)) from exc
                    return update_control_result(
                        receipt,
                        digest,
                        verification_sha256=verification,
                    )
                case "update_fence_release":
                    assert update_service_coordinator is not None
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None
                    try:
                        receipt, digest = update_service_coordinator.release_fence(
                            operation_id=request.selector_id,
                            receipt_sha256=request.boundary_sha256,
                            timeout=SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
                        )
                    except UpdateCutoverRefused as exc:
                        raise ServerControlError("operation_refused", str(exc)) from exc
                    return update_control_result(receipt, digest)
                case "update_maintenance_abort":
                    assert update_service_coordinator is not None
                    assert request.selector_id is not None
                    assert request.boundary_sha256 is not None
                    try:
                        receipt, digest = update_service_coordinator.abort_before_switch(
                            operation_id=request.selector_id,
                            receipt_sha256=request.boundary_sha256,
                            timeout=SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
                        )
                    except UpdateCutoverRefused as exc:
                        raise ServerControlError("operation_refused", str(exc)) from exc
                    return update_control_result(receipt, digest)
            raise AssertionError(f"Unhandled server control operation {request.operation!r}")

        control_server = ServerControlServer(
            Path(identity.control_socket),
            instance_id=identity.instance_id,
            owner_uid=os.geteuid(),
            owner_gid=os.getegid(),
            handler=dispatch_server_control,
        )
    identity_access = IdentityAccess(
        store,
        space_id=space_id,
        space_kind=space_kind,
        trusted_principal_resolver=trusted_principal_resolver,
    )
    set_team_session_cookie = identity_access.set_team_session_cookie
    resolve_team_user = identity_access.resolve_team_user
    launcher = AcceptanceAgentLauncher() if acceptance_agent else AgentLauncher()
    if control_server is not None:
        provider_readiness_coordinator = ProviderReadinessCoordinator(
            store,
            launcher,
            identity,
        )
        project_provision_coordinator = ProjectProvisionCoordinator(
            store,
            identity,
            provider_readiness_coordinator,
        )
        target_transfer_upload_coordinator = TargetTransferUploadCoordinator(
            store,
            app_data,
            identity,
        )
        backup_capture_coordinator = BackupCaptureCoordinator(store, app_data, identity)
    agent_mode: Literal["acceptance", "provider"] = "acceptance" if acceptance_agent else "provider"
    provider_skills = ProviderSkillInventoryManager(store)
    catalog = ProjectCatalog(app_data, store, launcher, provider_skills)
    attachment_store = ChatAttachmentStore(app_data / "chat-attachments")

    ensure_auto_research_graph_target = partial(
        _ensure_auto_research_graph_target,
        catalog=catalog,
    )
    graph_branch_summary = partial(
        _graph_branch_summary,
        store=store,
        catalog=catalog,
    )

    project_display_cache = ProjectDisplayCache(
        store,
        catalog,
        serialize_episode=lambda project_id, episode, projection_snapshot: serialize_episode(
            store,
            project_id,
            episode,
            branch_summary=graph_branch_summary,
            projection_snapshot=projection_snapshot,
        ).model_dump(mode="json"),
        project_experiment_control=lambda state, experiment_id, runtime, episode, report_episode_id: (
            _experiment_control_response(
                state,
                experiment_id,
                runtime,
                episode,
                latest_report_episode_id=report_episode_id,
            ).model_dump(mode="json")
        ),
        logger=logger,
    )
    refresh_cached_project_after_stream = project_display_cache.refresh_cached_project_after_stream

    setup = ProjectSetupManager(app_data, catalog, launcher)
    if control_server is not None:
        target_transfer_activation_coordinator = TargetTransferActivationCoordinator(
            store,
            catalog,
            setup,
            app_data,
        )
    default_record = (
        catalog.register(manifest_path, identity_action="adopted") if manifest_path else None
    )
    default_project_id = default_record.project_id if default_record else None
    default_project_name = default_record.name if default_record else None
    default_state_host = (
        catalog.state_host(default_project_id) if default_project_id is not None else ""
    )
    default_service = (
        _LazyProjectService(catalog, default_project_id) if default_project_id is not None else None
    )
    # FastAPI may enter and exit the synchronous admission dependency on
    # different worker threads, so this must remain a primitive Lock, not RLock.
    experiment_operation_lock = KeyedLocks()
    result_view_keep_lock = KeyedLocks()
    experiment_admission = ExperimentAdmission(
        experiment_operation_lock,
        _experiment_control_node_id,
    )
    graph_watcher_retry = GraphWatcherRetryRegistry()

    async def background_task_stream(
        project_id: str,
        kind: AgentTaskKind,
        request: AgentTaskRequest,
        execution: AgentTaskExecution,
    ) -> AsyncIterator[str]:
        service = _project_service(catalog, project_id)
        task = store.agent_task(execution.operation_id)
        if task is None or task.project_id != project_id:
            raise ValueError("The agent stream lost its durable project task.")
        profile = service.resolve_agent_profile(
            resolved_agent_surface(
                store,
                kind,
                request,
                parent_operation_id=task.parent_operation_id,
            ),
            provider=request.provider,
            model=request.model,
            reasoning=request.reasoning,
            run_on=request.run_on,
        )
        execution.runtime_id = configured_runtime_id(profile.provider, profile.runtime)
        if task.graph_target.kind == "branch" and kind != "branch_merge":
            service = service.for_graph_target(
                task.graph_target,
                expected_episode_id=task.graph_target.branch_id,
            )
        if kind == "branch_merge":
            if not isinstance(request, BranchMergeRunRequest):
                raise TypeError("A branch merge task requires its pinned merge request.")
            episode = store.episode(request.episode_id)
            if episode is None or episode.project_id != project_id:
                raise ValueError("The branch merge task lost its Auto-research episode.")
            async with aclosing(
                stream_branch_merge_task(
                    service,
                    launcher,
                    request,
                    app_data,
                    episode=episode,
                    task=task,
                    execution=execution,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
            return
        if kind == "paper_coach":
            assert isinstance(request, CoachRequest)
            async with aclosing(
                stream_coach(
                    service,
                    launcher,
                    service.paper,
                    request,
                    app_data,
                    execution=execution,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
            return
        if kind == "episode_report":
            if not isinstance(request, EpisodeReportRunRequest):
                raise TypeError("An episode report task requires its frozen report request.")
            async with aclosing(
                stream_episode_report_run(service, launcher, request, execution)
            ) as stream:
                async for frame in stream:
                    yield frame
            return
        if kind == "auto_research":
            if not isinstance(request, AutoResearchRunRequest):
                raise TypeError("An Auto-research task requires its operational run request.")
            if request.run_on is None:
                raise ValueError("An Auto-research turn has no pinned execution machine.")
            execution_machine = service.manifest.machine_map.get(request.run_on)
            if execution_machine is None:
                raise ValueError(f"unknown execution machine: {request.run_on}")

            def validate_auto_research_patch(context, arguments) -> AutoResearchCommandEffectResult:
                validated = _validate_work_patch_live(
                    service,
                    arguments.patch,
                    run_truth_scope=list(context.request.run_truth_scope or ()),
                    source_operation_id=context.task.operation_id,
                    profile=(
                        "orchestrator" if context.request.role == "orchestrator" else "ordinary"
                    ),
                )
                result = validated.model_dump(mode="json")
                if validated.status == "valid":
                    return AutoResearchCommandEffectResult(result=result)
                diagnostic = (
                    validated.messages[0]
                    if validated.messages
                    else f"Auto-research Patch validation is {validated.status}."
                )
                return AutoResearchCommandEffectResult(
                    status=validated.status,
                    message=diagnostic,
                    result=result,
                )

            def apply_auto_research_patch(context, patch_text, source_effect_id):
                result, failure = _apply_work_patch(
                    service,
                    execution,
                    patch_text,
                    run_truth_scope=list(
                        context.request.run_truth_scope
                        or service.manifest.agent.default_run_truth_scope
                    ),
                    profile="orchestrator",
                    source_operation_id=context.task.operation_id,
                    source_effect_id=source_effect_id,
                )
                if failure is not None:
                    return None, failure.message, failure.correctable
                return result, None, False

            effects = auto_research_command_effects(
                store=store,
                background=background_tasks,
                validate=validate_auto_research_patch,
                worker_request_factory=lambda context, arguments, instruction, worker_id: (
                    _auto_research_worker_request(
                        service,
                        context,
                        arguments,
                        instruction,
                        worker_id,
                    )
                ),
                graph_state=service.history.state,
                execution_host=execution_machine.host,
                apply_patch=apply_auto_research_patch,
                on_graph_applied=lambda: evaluate_graph_wake_boundary(
                    project_id,
                    None,
                    graph_target=task.graph_target,
                    source="Auto-research in-turn Apply",
                ),
                on_watcher_ready=lambda ready_project_id: evaluate_graph_wake_boundary(
                    ready_project_id,
                    None,
                    graph_target=task.graph_target,
                    source="Auto-research graph condition",
                ),
                experiment_coordinator=auto_research_experiment_coordinator,
            )
            dispatcher = AutoResearchCommandDispatcher(store, effects)
            stream_auto_research = (
                stream_auto_research_orchestrator_run
                if request.role == "orchestrator"
                else stream_auto_research_worker_run
            )
            async with aclosing(
                stream_auto_research(
                    service,
                    launcher,
                    request,
                    app_data,
                    execution,
                    command_dispatcher=dispatcher,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
            return
        assert isinstance(request, RunRequest)
        if kind in {"node_chat", "project_chat"}:
            if request.mode == "work":
                child_route = store.auto_research_child_work_for_operation(execution.operation_id)
                if child_route is not None:
                    async with aclosing(
                        stream_auto_research_child_work_run(
                            service,
                            launcher,
                            request,
                            app_data,
                            execution,
                            route=child_route,
                        )
                    ) as stream:
                        async for frame in stream:
                            yield frame
                    return
                if request.patch_kind == "experiment_loop":
                    async with aclosing(
                        stream_experiment_loop_task(
                            service,
                            launcher,
                            request,
                            app_data,
                            execution=execution,
                        )
                    ) as stream:
                        async for frame in stream:
                            yield frame
                    return
                async with aclosing(
                    stream_work_run(
                        service,
                        launcher,
                        request,
                        app_data,
                        execution=execution,
                    )
                ) as stream:
                    async for frame in stream:
                        yield frame
                return
            async with aclosing(
                stream_discuss_run(
                    service,
                    launcher,
                    request,
                    app_data,
                    execution=execution,
                )
            ) as stream:
                async for frame in stream:
                    yield frame
            return
        async with aclosing(
            stream_graph_run(
                service,
                launcher,
                kind,
                request,
                app_data,
                execution=execution,
            )
        ) as stream:
            async for frame in stream:
                yield frame

    background_tasks = BackgroundAgentTasks(
        store,
        background_task_stream,
        on_stream_closed=refresh_cached_project_after_stream,
        startup_effect_fence=startup_effect_fence,
        runtime_admission_gate=background_admission_gate,
    )
    if control_server is not None:
        member_removal_coordinator = MemberRemovalCoordinator(
            store,
            background_tasks,
            identity,
        )
    auto_research_experiment_coordinator = AutoResearchExperimentCoordinator(
        store,
        background_tasks,
        project_service=lambda project_id, episode_id: _project_service(
            catalog,
            project_id,
        ).for_graph_target(
            _episode_for_http(store, catalog, project_id, episode_id).graph_target,
            expected_episode_id=episode_id,
        ),
        operation_lock=experiment_operation_lock,
    )

    def restart_child_worker_request(
        context: AutoResearchCommandContext,
        arguments: SpawnArguments,
        instruction: str,
        worker_id: str,
    ) -> RunRequest:
        service = _project_service(catalog, context.task.project_id).for_graph_target(
            context.episode.graph_target,
            expected_episode_id=context.episode.episode_id,
        )
        return _auto_research_worker_request(
            service,
            context,
            arguments,
            instruction,
            worker_id,
        )

    def restart_child_seat_node_type(
        project_id: str,
        episode_id: str,
        node_id: str,
    ) -> str | None:
        episode = store.episode(episode_id)
        if episode is None or episode.project_id != project_id:
            raise ValueError("Auto-research child admission lost its branch episode.")
        service = _project_service(catalog, project_id).for_graph_target(
            episode.graph_target,
            expected_episode_id=(
                episode.graph_target.branch_id if episode.graph_target.kind == "branch" else None
            ),
        )
        node = service.history.state().nodes.get(node_id)
        return node.type if node is not None else None

    # Reconciliation runs on the 5s watcher poll, so a stuck admission would log
    # a dozen identical warnings a minute. Keep only each active admission's last
    # reason: log transitions (including A -> B -> A) and forget settled entries.
    reported_child_deferrals: dict[str, tuple[str, str]] = {}
    reported_child_deferrals_lock = threading.Lock()

    def reconcile_auto_research_children(episode_id: str | None = None):
        reconciliation = reconcile_pending_auto_research_child_admissions(
            store,
            background_tasks,
            auto_research_experiment_coordinator,
            worker_request_factory=restart_child_worker_request,
            seat_node_type=restart_child_seat_node_type,
            episode_id=episode_id,
        )
        current_deferrals = {
            deferral.admission_id: (deferral.episode_id, deferral.reason)
            for deferral in reconciliation.deferrals
        }
        with reported_child_deferrals_lock:
            if episode_id is None:
                stale_ids = set(reported_child_deferrals) - set(current_deferrals)
            else:
                stale_ids = {
                    admission_id
                    for admission_id, (reported_episode_id, _reason) in (
                        reported_child_deferrals.items()
                    )
                    if reported_episode_id == episode_id and admission_id not in current_deferrals
                }
            for admission_id in stale_ids:
                reported_child_deferrals.pop(admission_id, None)
            changed_deferrals = [
                deferral
                for deferral in reconciliation.deferrals
                if reported_child_deferrals.get(deferral.admission_id)
                != (deferral.episode_id, deferral.reason)
            ]
            for deferral in changed_deferrals:
                reported_child_deferrals[deferral.admission_id] = (
                    deferral.episode_id,
                    deferral.reason,
                )
        for deferral in changed_deferrals:
            logger.warning(
                "Auto-research child admission %s (%s) in episode %s is still unreflected "
                "and blocks finish: %s",
                deferral.admission_id,
                deferral.child_kind,
                deferral.episode_id,
                deferral.reason,
            )
        return reconciliation

    episode_reconciler = EpisodeReconciler(store, background_tasks, logger=logger)
    reconcile_auto_research_wrapup = episode_reconciler.reconcile_auto_research_wrapup
    reconcile_auto_research_episode = episode_reconciler.reconcile_auto_research_episode
    reconcile_auto_research_recovery_pass = episode_reconciler.reconcile_auto_research_recovery_pass
    reconcile_experiment_episode = episode_reconciler.reconcile_experiment_episode

    watcher_delivery = WatcherDelivery(
        store,
        retry=graph_watcher_retry,
        project_service=lambda project_id: _project_service(catalog, project_id),
        graph_project_service=lambda project_id, target: _project_service(
            catalog, project_id
        ).for_graph_target(
            target,
            expected_episode_id=(target.branch_id if target.kind == "branch" else None),
        ),
        generic_request=_generic_watcher_delivery_request,
        experiment_operation_lock=experiment_operation_lock,
        experiment_admission=experiment_admission,
        deliver_auto_research_group=lambda group: deliver_auto_research_watcher_group(
            background_tasks,
            group,
        ),
        preflight_episode_wake=preflight_episode_wake,
        admit_experiment_watcher_invocation=admit_experiment_watcher_invocation,
        experiment_watcher_request=experiment_watcher_delivery_request,
        start_watcher_notification=lambda *args, **kwargs: start_watcher_notification(
            background_tasks, *args, **kwargs
        ),
        state_unavailable=lambda exc: isinstance(exc, StateUnavailable),
        task_graph_capable=task_graph_capable,
        task_experiment_episode_id=task_experiment_episode_id,
        reconcile_experiment_episode=episode_reconciler.reconcile_experiment_episode,
        reconcile_graph_boundaries=reconcile_accepted_graph_boundaries,
        ready_graph_groups=lambda candidate_store, project_id: ready_graph_watcher_groups(
            candidate_store,
            project_id,
        ),
        logger=logger,
    )
    deliver_watcher_group = watcher_delivery.deliver_watcher_group
    evaluate_graph_wake_boundary = watcher_delivery.evaluate_graph_wake_boundary
    sweep_graph_conditions_at_startup = watcher_delivery.sweep_graph_conditions_at_startup
    retry_graph_wakes_after_poll = watcher_delivery.retry_graph_wakes_after_poll

    def after_task_settled(
        project_id: str,
        kind: AgentTaskKind,
        request: AgentTaskRequest,
        execution: AgentTaskExecution,
    ) -> None:
        # `finally`, because the Auto-research half ran even when the generic
        # half raised while the engine owned both. It keeps its own diagnostic
        # and still lets the generic failure reach the engine's receipt.
        try:
            watcher_delivery.evaluate_graph_conditions_after_task(
                project_id,
                kind,
                request,
                execution,
            )
            for episode in store.episodes(project_id):
                if episode.mode == "auto_research":
                    reconcile_auto_research_children(episode.episode_id)
                    auto_research_experiment_coordinator.reconcile(episode.episode_id)
                    reconcile_pending_auto_research_lifecycle(
                        background_tasks,
                        episode_id=episode.episode_id,
                    )
                    reconcile_pending_auto_research_mail(
                        background_tasks,
                        episode_id=episode.episode_id,
                    )
        finally:
            try:
                if isinstance(request, AutoResearchRunRequest):
                    episode_reconciler.settle_auto_research_task(request, execution)
            finally:
                if member_removal_coordinator is not None:
                    member_removal_coordinator.reconcile_pending()

    background_tasks.on_task_settled = after_task_settled
    background_tasks.on_auto_research_admission_exhausted = lambda episode: (
        reconcile_auto_research_episode(
            episode.episode_id,
            source="budget exhaustion",
            operation_id=episode.root_operation_id,
        )
    )

    graph_watcher_retry_worker = WatcherRetryWorker(retry_graph_wakes_after_poll)

    def after_watcher_poll() -> None:
        graph_watcher_retry_worker.signal()
        reconcile_auto_research_recovery_pass()
        auto_research_episode_ids: list[str] = []
        for project in store.projects():
            for episode in store.episodes(project.project_id):
                if episode.mode == "auto_research":
                    auto_research_episode_ids.append(episode.episode_id)
                    reconcile_auto_research_children(episode.episode_id)
                    auto_research_experiment_coordinator.reconcile(episode.episode_id)
                    reconcile_auto_research_episode(
                        episode.episode_id,
                        source="watcher poll",
                    )
                elif episode.mode == "experiment_loop":
                    reconcile_experiment_episode(
                        episode.episode_id,
                        source="watcher poll",
                    )
        for episode_id in auto_research_episode_ids:
            reconcile_pending_auto_research_lifecycle(
                background_tasks,
                episode_id=episode_id,
            )
            reconcile_pending_auto_research_mail(
                background_tasks,
                episode_id=episode_id,
            )

    watcher_poller = WatcherPoller(
        store,
        on_completed=deliver_watcher_group,
        on_poll_completed=after_watcher_poll,
    )
    health_composition = HealthComposition(
        instance_metadata=identity,
        agent_mode=agent_mode,
        default_project_name=default_project_name,
        space_id=space_id,
        space_kind=space_kind,
    )

    if server_doctor_reader is None:
        doctor_machine = LinuxServerDoctorMachine(server_layout)
        server_doctor_reader = doctor_machine.inspect

    if server_restore_completed_at_reader is None:

        def server_restore_completed_at_reader() -> datetime | None:
            from rcp.server_ops.restore import read_restore_journal_if_present

            journal = read_restore_journal_if_present(
                server_layout,
                expected_uid=os.geteuid(),
            )
            if journal is None or journal.phase != "complete":
                return None
            if journal.activation_readback is None:
                raise RuntimeError("completed restore has no activation readback")
            return journal.activation_readback.activated_at

    if server_protected_backup_reader is None:

        def server_protected_backup_reader(
            report: ServerDoctorReport,
        ) -> BackupArchiveReceipt | None:
            if report.installation_id is None or report.backup_destination is None:
                return None
            return latest_protected_backup_receipt(
                Path(report.backup_destination),
                installation_id=report.installation_id,
                expected_uid=os.geteuid(),
            )

    server_status_composition = ServerStatusComposition(
        doctor_reader=server_doctor_reader,
        protected_backup_reader=server_protected_backup_reader,
        restore_completed_at_reader=server_restore_completed_at_reader,
        clock=server_status_clock or (lambda: datetime.now(UTC)),
    )
    services = ApiServices(
        store=store,
        catalog=catalog,
        identity_access=identity_access,
        launcher=launcher,
        setup=setup,
        attachment_store=attachment_store,
        watcher_poller=watcher_poller,
        result_view_keep_locks=result_view_keep_lock,
        project_display_cache=project_display_cache,
        watcher_delivery=watcher_delivery,
        experiment_operation_lock=experiment_operation_lock,
        background_tasks=background_tasks,
        experiment_admission=experiment_admission,
        health_composition=health_composition,
        server_status_composition=server_status_composition,
    )

    async def warm_provider_capabilities() -> None:
        try:
            targets = await asyncio.to_thread(catalog.provider_targets)

            def mark_refreshing() -> None:
                for provider, host, binary in targets:
                    provider_skills.mark_refreshing(provider, host, binary)

            def probe(provider: str, host: str, binary: str | None) -> None:
                try:
                    readiness = launcher.readiness(provider, host=host, binary=binary)
                except Exception as exc:
                    # An unexpected readiness exception still has to finish
                    # this startup generation. Feeding the diagnostic through
                    # the inventory manager preserves any last-good skills as
                    # stale instead of leaving the target stuck refreshing.
                    readiness = ProviderReadiness(
                        provider=provider,
                        installed=False,
                        authenticated=False,
                        binary_path=binary,
                        path_state="unreachable" if host else "missing",
                        reason=str(exc),
                    )
                    provider_skills.refresh(provider, host, binary, readiness)
                    raise
                provider_skills.refresh(provider, host, binary, readiness)

            # Mark the whole startup inventory before beginning any provider
            # process so the UI never mistakes a prior process's cache for a
            # completed refresh in this one. All SQLite and CLI/SSH work stays
            # off the event loop.
            await asyncio.to_thread(mark_refreshing)

            results = await asyncio.gather(
                *(
                    asyncio.to_thread(probe, provider, host, binary)
                    for provider, host, binary in targets
                ),
                return_exceptions=True,
            )
            for target, result in zip(targets, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Could not warm provider capability %s: %s", target, result)
        except Exception as exc:
            # Warming is an optimization; readiness remains authoritative and
            # can retry an individual capability when explicitly requested.
            logger.warning("Could not warm provider capabilities: %s", exc)

    async def sweep_remote_run_stages() -> None:
        try:
            await asyncio.to_thread(RemoteRunStage(default_state_host).sweep)
        except Exception as exc:
            # Stale scratch cleanup is best-effort and must not delay app
            # availability when the project's remote machine is unavailable.
            logger.warning("Could not sweep remote run stages: %s", exc)

    startup_maintenance: list[asyncio.Task[None]] = []
    runtime_loop: list[asyncio.AbstractEventLoop | None] = [None]

    def pause_update_runtime_owners(timeout: float) -> None:
        """Stop process-owned pollers and wait for already-scheduled async reads."""

        watcher_poller.stop()
        graph_watcher_retry_worker.stop(timeout=timeout)
        if watcher_poller.is_running() or graph_watcher_retry_worker.is_running():
            raise UpdateCutoverRefused("Timed out stopping watcher polling at the update boundary.")
        loop = runtime_loop[0]
        if loop is None or loop.is_closed():
            raise UpdateCutoverRefused(
                "The app runtime loop is unavailable at the update boundary."
            )

        async def wait_for_scheduled_reads() -> None:
            pending = {
                task
                for task in (
                    *startup_maintenance,
                    *project_display_cache.reconciliation_tasks.values(),
                )
                if not task.done()
            }
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        future = asyncio.run_coroutine_threadsafe(wait_for_scheduled_reads(), loop)
        try:
            future.result(timeout=timeout)
        except TimeoutError as exc:
            raise UpdateCutoverRefused(
                "Timed out waiting for scheduled runtime reads to settle."
            ) from exc

    def resume_update_runtime_owners() -> None:
        graph_watcher_retry_worker.start()
        watcher_poller.start()

    if control_server is not None and installed_update_layout:
        assert target_transfer_upload_coordinator is not None
        update_service_coordinator = UpdateServiceCoordinator(
            layout=server_layout,
            instance_metadata=identity,
            space_id=space_id,
            admission=runtime_admission_gate,
            background_admission=background_admission_gate,
            background=background_tasks,
            capture_sqlite=capture_sqlite_for_control,
            catalog=catalog,
            store=store,
            startup_effect_fence=startup_effect_fence,
            runtime_started=startup_effect_runtime_event,
            runtime_error=lambda: startup_effect_release_error[0],
            pause_runtime_owners=pause_update_runtime_owners,
            resume_runtime_owners=resume_update_runtime_owners,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        startup_maintenance.clear()
        runtime_loop[0] = asyncio.get_running_loop()
        control_started = False
        runtime_started = False
        runtime_start_lock = asyncio.Lock()

        async def start_deferred_runtime() -> None:
            nonlocal control_started, runtime_started
            async with runtime_start_lock:
                if runtime_started:
                    return
                # This is the one ordinary startup sequence. Normal startup calls
                # it immediately; a cutover candidate calls it after the shared
                # effect fence opens. Recovery must precede every other owner.
                background_tasks.recover_at_startup()
                if member_removal_coordinator is not None:
                    member_removal_coordinator.reconcile_pending()
                background_tasks.accept_watcher_notifications()
                store.prune_operational_storage()
                await asyncio.to_thread(
                    reconcile_reserved_auto_research_roots,
                    background_tasks,
                    ensure_auto_research_graph_target,
                )
                await asyncio.to_thread(
                    reconcile_committed_auto_research_dispatches,
                    background_tasks,
                )
                child_reconciliation = await asyncio.to_thread(
                    reconcile_auto_research_children,
                )
                if child_reconciliation.cancelled:
                    logger.warning(
                        "Cancelled %s unlaunchable Auto-research child admission(s) at startup.",
                        child_reconciliation.cancelled,
                    )
                for project in store.projects():
                    for episode in store.episodes(project.project_id):
                        if episode.mode == "auto_research":
                            await asyncio.to_thread(
                                auto_research_experiment_coordinator.reconcile,
                                episode.episode_id,
                            )
                orphaned_endings = await asyncio.to_thread(
                    reconcile_orphaned_auto_research_failures,
                    background_tasks,
                )
                for ending in orphaned_endings:
                    await asyncio.to_thread(
                        reconcile_auto_research_wrapup,
                        ending,
                        source="startup failure recovery",
                    )
                try:
                    await asyncio.to_thread(
                        reconcile_pending_auto_research_lifecycle,
                        background_tasks,
                    )
                    await asyncio.to_thread(
                        reconcile_pending_auto_research_mail,
                        background_tasks,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not reconcile pending Auto-research lifecycle or mail at startup: %s",
                        exc,
                    )
                for project in store.projects():
                    for episode in store.episodes(project.project_id):
                        if episode.mode == "auto_research":
                            await asyncio.to_thread(
                                reconcile_auto_research_episode,
                                episode.episode_id,
                                source="startup",
                            )
                        elif episode.mode == "experiment_loop":
                            await asyncio.to_thread(
                                reconcile_experiment_episode,
                                episode.episode_id,
                                source="startup",
                            )
                await asyncio.to_thread(reconcile_auto_research_recovery_pass)
                if member_removal_coordinator is not None:
                    await asyncio.to_thread(member_removal_coordinator.reconcile_pending)
                _sweep_stale_stages(app_data / "run-stage", now=time.time())
                attachment_store.sweep()
                cache_roots = [
                    *discover_project_cache_roots(app_data),
                    legacy_shared_cache_roots(app_data),
                ]
                for source_root, slice_root in cache_roots:
                    RebuildableCache(
                        source_root,
                        REMOTE_SOURCE_CACHE_LIMITS,
                        layout="files",
                    ).sweep()
                    RebuildableCache(
                        slice_root,
                        SESSION_SLICE_CACHE_LIMITS,
                        layout="directories",
                    ).sweep()
                startup_maintenance.append(asyncio.create_task(warm_provider_capabilities()))
                if default_state_host:
                    startup_maintenance.append(asyncio.create_task(sweep_remote_run_stages()))
                await asyncio.to_thread(sweep_graph_conditions_at_startup)
                graph_watcher_retry_worker.start()
                watcher_poller.start()
                if control_server is not None and not control_started:
                    control_server.start()
                    control_started = True
                runtime_started = True
                app.state.startup_effect_runtime_started = True
                startup_effect_runtime_event.set()

        release_task: asyncio.Task[None] | None = None
        restore_timeout_task: asyncio.Task[None] | None = None
        fenced_startup = startup_effect_fence is not None and startup_effect_fence.active
        # Preserve ordinary startup's fail-fast boundary: if recovery fails, do
        # not run the shutdown path over state that never completed startup.
        if not fenced_startup:
            await start_deferred_runtime()
        try:
            if fenced_startup:
                app.state.startup_recovery_plan = background_tasks.plan_startup_recovery().as_dict()
                if control_server is not None:
                    control_server.start()
                    control_started = True
                released = asyncio.Event()
                loop = asyncio.get_running_loop()

                def notify_release() -> None:
                    if not loop.is_closed():
                        loop.call_soon_threadsafe(released.set)

                startup_effect_fence.on_release(notify_release)

                async def resume_after_release() -> None:
                    await released.wait()
                    try:
                        await start_deferred_runtime()
                    except BaseException as exc:
                        startup_effect_release_error[0] = str(exc)
                        app.state.startup_effect_release_error = startup_effect_release_error[0]
                        logger.exception("Deferred startup failed after the effect fence opened.")

                release_task = asyncio.create_task(resume_after_release())
                app.state.startup_effect_release_task = release_task
                if restore_runtime_boundary is not None:

                    async def stop_uncommitted_restore_after_timeout() -> None:
                        await asyncio.sleep(SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS)
                        if restore_activation_committed.is_set():
                            return
                        logger.error(
                            "Restore activation %s was not committed before its bounded startup "
                            "timeout; stopping the disabled replacement process.",
                            restore_runtime_boundary.operation_id,
                        )
                        _terminate_uncommitted_restore_process()

                    restore_timeout_task = asyncio.create_task(
                        stop_uncommitted_restore_after_timeout()
                    )
                    app.state.restore_activation_timeout_task = restore_timeout_task
            yield
        finally:
            if restore_timeout_task is not None and not restore_timeout_task.done():
                restore_timeout_task.cancel()
                with suppress(asyncio.CancelledError):
                    await restore_timeout_task
            if release_task is not None and not release_task.done():
                release_task.cancel()
                with suppress(asyncio.CancelledError):
                    await release_task
            if control_started and control_server is not None:
                control_server.stop()
            for task in startup_maintenance:
                task.cancel()
            for task in list(project_display_cache.reconciliation_tasks.values()):
                task.cancel()
            for task in startup_maintenance:
                with suppress(asyncio.CancelledError):
                    await task
            for task in list(project_display_cache.reconciliation_tasks.values()):
                with suppress(asyncio.CancelledError):
                    await task
            watcher_poller.stop()
            graph_watcher_retry_worker.stop()
            background_tasks.shutdown()
            # A PyInstaller one-file backend runs under a bootloader supervisor
            # whose signal exit can skip the CLI context manager's ``finally``.
            # Source reload workers share metadata owned by the outer supervisor,
            # so they must leave it in place across worker restarts.
            if getattr(sys, "frozen", False):
                remove_server_metadata(app_data, instance_id=identity.instance_id)
            runtime_loop[0] = None

    app = FastAPI(title="RCP", version=__version__, lifespan=lifespan)
    app.state.services = services
    app.state.catalog = catalog
    app.state.provider_skills = provider_skills
    app.state.setup = setup
    app.state.default_project_id = default_project_id
    app.state.service = default_service
    app.state.data_dir = app_data
    app.state.background_tasks = background_tasks
    app.state.project_reconciliation_tasks = project_display_cache.reconciliation_tasks
    app.state.watcher_poller = watcher_poller
    app.state.graph_watcher_retry_worker = graph_watcher_retry_worker
    app.state.instance_metadata = identity
    app.state.server_control = control_server
    app.state.space_id = space_id
    app.state.space_kind = space_kind
    app.state.launcher = launcher
    app.state.agent_mode = agent_mode
    app.state.startup_effect_fence = startup_effect_fence
    app.state.startup_recovery_plan = None
    app.state.startup_effect_runtime_started = False
    app.state.startup_effect_runtime_event = startup_effect_runtime_event
    app.state.startup_effect_release_task = None
    app.state.startup_effect_release_error = None
    app.state.restore_activation_timeout_task = None
    app.state.update_runtime_boundary = update_runtime_boundary
    app.state.runtime_admission_gate = runtime_admission_gate
    app.state.background_admission_gate = background_admission_gate
    if space_kind == "team":
        app.add_middleware(TeamPublicAuthBodyLimit)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def enforce_update_maintenance(request: Request, call_next):
        try:
            with runtime_admission_gate.mutation(f"HTTP {request.method} {request.scope['path']}"):
                return await call_next(request)
        except UpdateAdmissionClosed as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": {
                        "code": (
                            "server_restore_activation"
                            if restore_runtime_boundary is not None
                            else "server_update_maintenance"
                        ),
                        "message": str(exc),
                    }
                },
                headers={"Retry-After": "5"},
            )

    @app.middleware("http")
    async def require_current_instance(request: Request, call_next):
        project_prefix = "/api/projects/"
        path = request.scope["path"]
        if path.startswith(project_prefix):
            project_id, separator, rest = path[len(project_prefix) :].partition("/")
            canonical_project_id = catalog.resolve_project_id(project_id)
            if canonical_project_id != project_id:
                canonical_path = f"{project_prefix}{canonical_project_id}"
                if separator:
                    canonical_path = f"{canonical_path}/{rest}"
                request.scope["path"] = canonical_path
                path = canonical_path

        public_team_api_paths = {
            "/api/health",
            "/api/team/enroll",
            "/api/team/session/exchange",
        }
        session_ending_paths = {
            "/api/team/session/logout",
            "/api/team/credential/rotate",
            "/api/team/credential/revoke",
        }
        path_parts = path.split("/")
        native_team_transfer_path = (
            len(path_parts) == 7
            and path_parts[1:5] == ["api", "native", "project-transfers", "target-requests"]
            and (
                (request.method == "GET" and path_parts[6:] == ["activation-proof"])
                or (request.method == "POST" and path_parts[6:] == ["cleanup-acknowledgment"])
            )
        )
        if (
            space_kind == "team"
            and request.method != "OPTIONS"
            and (path == "/api" or path.startswith("/api/"))
            and path not in public_team_api_paths
            and not native_team_transfer_path
        ):
            try:
                resolve_team_user(request)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers=exc.headers,
                )
            if trusted_principal_resolver is None and request.method not in {
                "GET",
                "HEAD",
                "OPTIONS",
            }:
                origin = request.headers.get("origin")
                expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
                if origin is not None and origin.rstrip("/") != expected_origin.rstrip("/"):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": {
                                "code": "team_origin_invalid",
                                "message": "The request origin does not match this team server.",
                            }
                        },
                    )
                media_type = request.headers.get("content-type", "").partition(";")[0].strip()
                attachment_upload = (
                    request.method == "POST"
                    and len(path_parts) == 7
                    and path_parts[1:3] == ["api", "projects"]
                    and path_parts[4] == "chats"
                    and path_parts[6] == "attachments"
                )
                allowed_media_type = media_type.lower() == "application/json" or (
                    attachment_upload and media_type.lower() == "multipart/form-data"
                )
                if not allowed_media_type:
                    return JSONResponse(
                        status_code=415,
                        content={
                            "detail": {
                                "code": "team_json_required",
                                "message": (
                                    "Authenticated team mutations require JSON, except for "
                                    "the bounded attachment upload."
                                ),
                            }
                        },
                    )
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            pinned_instance = request.headers.get("X-RCP-Instance-ID")
            if pinned_instance and pinned_instance != identity.instance_id:
                response = JSONResponse(
                    status_code=409,
                    content={
                        "detail": (
                            "RCP was replaced by another backend instance. Refresh before "
                            "making changes."
                        ),
                        "instance_id": identity.instance_id,
                    },
                )
                session = getattr(request.state, "team_session", None)
                if isinstance(session, str) and path not in session_ending_paths:
                    set_team_session_cookie(response, session)
                return response
        response = await call_next(request)
        session = getattr(request.state, "team_session", None)
        if isinstance(session, str) and path not in session_ending_paths:
            set_team_session_cookie(response, session)
        return response

    @app.exception_handler(PatchRejected)
    async def patch_rejected(_: Request, exc: PatchRejected) -> JSONResponse:
        status = 409 if any(item.code == "stale-node-edit" for item in exc.report.messages) else 422
        return JSONResponse(
            status_code=status,
            content={"detail": [item.model_dump(mode="json") for item in exc.report.messages]},
        )

    @app.exception_handler(StateUnavailable)
    async def state_unavailable(_: Request, exc: StateUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ReplayHalted)
    async def replay_halted(_: Request, exc: ReplayHalted) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
                "failed_revision": exc.failed_revision,
                "coherent_revision": exc.coherent_revision,
                "code": exc.code,
            },
        )

    @app.exception_handler(TeamAuthenticationError)
    async def team_authentication_error(_: Request, exc: TeamAuthenticationError) -> JSONResponse:
        status_by_code = {
            "enrollment_code_invalid": 401,
            "team_token_invalid": 401,
            "enrollment_code_consumed": 409,
            "enrollment_code_expired": 410,
            "enrollment_code_locked": 429,
        }
        return JSONResponse(
            status_code=status_by_code.get(exc.code, 401),
            content={"detail": {"code": exc.code, "message": str(exc)}},
        )

    # Exposed so the route-enumeration test can prove membership is attached,
    # rather than trusting that every project route was declared in one place.
    app.state.project_membership_dependency = require_project_membership

    app.include_router(health_router)
    app.include_router(server_status_router)
    app.include_router(team_router)
    app.include_router(index_router)
    app.include_router(index_membership_router)
    app.include_router(project_provisioning_router)
    app.include_router(project_state_router)
    app.include_router(episode_router)
    app.include_router(experiments_router)
    app.include_router(chats_router)
    app.include_router(history_router)
    app.include_router(paper_router)
    app.include_router(result_views_router)
    app.include_router(sync_router)
    app.include_router(tasks_router)
    app.include_router(watchers_router)

    web_dist = web_dist_path()
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app


def _generic_watcher_delivery_request(group: list[StoredWatcherRecord]) -> RunRequest:
    first = group[0]
    continuation = first.continuation
    if continuation.patch_kind != "work":
        raise ValueError("A generic watcher cannot carry Experiment-loop authority.")
    watcher_ids = [item.watcher_id for item in group]
    details = "\n".join(
        (
            f"- graph condition `{item.watcher_id}`: `{item.condition.model_dump_json()}`"
            if isinstance(item, GraphWatcherRecord)
            else f"- external watcher `{item.watcher_id}`: `{item.log_path}`"
        )
        for item in group
    )
    return RunRequest(
        provider=continuation.provider,
        model=continuation.model,
        reasoning=continuation.reasoning,
        run_on=continuation.run_on,
        run_truth_scope=continuation.run_truth_scope,
        chat_scope="node" if first.origin_task_kind == "node_chat" else "project",
        node_id=first.node_id,
        message=(
            "RCP watcher update: the following external or canonical graph conditions are "
            f"ready. Inspect their current authoritative state and continue the Work turn.\n{details}"
        ),
        chat_id=first.chat_id,
        session_id=None,
        mode="work",
        trigger="watcher",
        patch_kind="work",
        workflow_ids=continuation.workflow_ids,
        skill_ids=continuation.skill_ids,
        invoked_workflow_ids=[],
        invoked_skill_ids=[],
        resolved_skill_packages=continuation.resolved_skill_packages,
        watcher_ids=watcher_ids,
    )


def _experiment_control_node_id(
    request: RunRequest | CoachRequest | dict[str, object],
) -> str | None:
    if isinstance(request, CoachRequest):
        return None
    if isinstance(request, RunRequest):
        patch_kind = request.patch_kind
        node_id = request.control_node_id
    else:
        patch_kind = request.get("patch_kind")
        node_id = request.get("control_node_id")
    if patch_kind != "experiment_loop":
        return None
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("A bounded experiment-loop task must name its control node.")
    return node_id


def _auto_research_worker_request(
    service: ProjectService,
    _context: AutoResearchCommandContext,
    arguments: SpawnArguments,
    instruction: str,
    worker_id: str,
) -> RunRequest:
    """Resolve a spawned child through the current ordinary node-Work profile."""

    return _resolved_graph_request(
        service,
        "node_chat",
        RunRequest(
            chat_id=worker_id,
            chat_scope="node",
            node_id=arguments.seat_node_id,
            message=instruction,
            mode="work",
            trigger="orchestrator",
            patch_kind="work",
        ),
    )


def default_data_dir() -> Path:
    override = os.environ.get("RCP_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / "research-control-panel"
