"""Read-only installed-team-service diagnosis and exact release identity."""

from __future__ import annotations

import os
import pwd
import re
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from rcp.limits import SERVER_INSTALL_PROBE_TIMEOUT_SECONDS
from rcp.server_ops.cli import CallerIdentity, PreparedServerCommand, ServerEventEmitter
from rcp.server_ops.config import InstalledServerConfig, load_installed_server_config
from rcp.server_ops.control import (
    ServerControlClient,
    ServerControlError,
    ServerControlMemberSnapshot,
    ServerControlProbeResult,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout, server_service_unit_text
from rcp.server_ops.models import (
    MachineTarget,
    NonsecretField,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
)
from rcp.server_runtime import (
    ServerMetadata,
    ServerMetadataError,
    data_dir_identity,
    metadata_path,
    read_server_metadata,
    web_build_identity,
)

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,79}")
_NO_VALUE = "none"

DoctorOverallState = Literal[
    "healthy",
    "update_available",
    "candidate_pending",
    "restart_pending",
    "problems",
]
DoctorReleaseState = Literal[
    "aligned",
    "candidate_pending",
    "restart_pending",
    "inconsistent",
    "unavailable",
]
DoctorSourceState = Literal[
    "aligned",
    "update_available",
    "local_ahead",
    "diverged",
    "unavailable",
]
DoctorBackupState = Literal[
    "not_configured",
    "never_run",
    "protected",
    "partial",
    "failure",
    "unavailable",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ServerDoctorReport(_StrictModel):
    """One secret-free structured readback rendered by both CLI modes."""

    overall_state: DoctorOverallState
    installation_id: str | None
    service_account: str
    data_dir: str
    source_root: str
    releases_root: str
    configured_origin: str | None
    configured_branch: str | None
    source_public_key_fingerprint: str | None
    managed_main_head: str | None
    upstream_head: str | None
    candidate_commit: str | None
    current_commit: str | None
    running_commit: str | None
    release_state: DoctorReleaseState
    source_state: DoctorSourceState
    current_web_build_id: str | None
    running_web_build_id: str | None
    service_active_state: str
    service_unit_file_state: str
    service_main_pid: int | None
    reload_mode: Literal["disabled", "unknown"]
    space_id: str | None
    instance_id: str | None
    process_pid: int | None
    data_dir_id: str
    control_socket_status: str
    provider_check_status: Literal["available", "unavailable"]
    dependencies_ready: bool
    dependency_versions: str
    problems: tuple[str, ...]
    backup_status: DoctorBackupState = "not_configured"
    backup_destination: str | None = None
    backup_schedule: str | None = None
    backup_retention: int | None = None
    backup_recipient_fingerprint: str | None = None
    backup_timer_active_state: str = "not_configured"
    backup_timer_unit_file_state: str = "not_configured"
    last_backup_at: datetime | None = None
    last_backup_archive: str | None = None
    last_backup_captured_bytes: int | None = None
    last_backup_protected_projects: int | None = None
    last_backup_uncaptured_projects: int | None = None
    last_backup_failure: str | None = None
    update_operation_state: str = "none"
    update_candidate_commit: str | None = None
    update_restored_commit: str | None = None
    update_failure: str | None = None

    @field_validator(
        "managed_main_head",
        "upstream_head",
        "candidate_commit",
        "current_commit",
        "running_commit",
        "update_candidate_commit",
        "update_restored_commit",
    )
    @classmethod
    def validate_commit(cls, value: str | None) -> str | None:
        if value is not None and _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("doctor commit identities must be full lowercase Git object ids")
        return value

    @field_validator("problems")
    @classmethod
    def validate_problems(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("doctor problems must be unique")
        if any(
            not problem
            or len(problem) > 160
            or any(ord(character) < 32 or ord(character) == 127 for character in problem)
            for problem in value
        ):
            raise ValueError("doctor problems must be bounded one-line messages")
        return value

    @field_validator("backup_recipient_fingerprint")
    @classmethod
    def validate_backup_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("doctor backup recipient fingerprint must be lowercase SHA-256")
        return value

    @field_validator("last_backup_at")
    @classmethod
    def validate_last_backup_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("doctor backup time requires a timezone")
        return value

    @model_validator(mode="after")
    def state_matches_problems(self) -> ServerDoctorReport:
        if (self.overall_state == "problems") != bool(self.problems):
            raise ValueError("doctor overall state must agree with its problem list")
        return self

    def fields(self) -> tuple[NonsecretField, ...]:
        return (
            NonsecretField(name="overall_state", value=self.overall_state),
            NonsecretField(name="installation_id", value=_shown(self.installation_id)),
            NonsecretField(name="service_account", value=self.service_account),
            NonsecretField(name="data_dir", value=self.data_dir),
            NonsecretField(name="source_root", value=self.source_root),
            NonsecretField(name="releases_root", value=self.releases_root),
            NonsecretField(name="configured_origin", value=_shown(self.configured_origin)),
            NonsecretField(name="configured_branch", value=_shown(self.configured_branch)),
            NonsecretField(
                name="source_public_key_fingerprint",
                value=_shown(self.source_public_key_fingerprint),
            ),
            NonsecretField(name="managed_main_head", value=_shown(self.managed_main_head)),
            NonsecretField(name="upstream_head", value=_shown(self.upstream_head)),
            NonsecretField(name="candidate_commit", value=_shown(self.candidate_commit)),
            NonsecretField(name="current_commit", value=_shown(self.current_commit)),
            NonsecretField(name="running_commit", value=_shown(self.running_commit)),
            NonsecretField(name="release_state", value=self.release_state),
            NonsecretField(name="source_state", value=self.source_state),
            NonsecretField(name="current_web_build_id", value=_shown(self.current_web_build_id)),
            NonsecretField(name="running_web_build_id", value=_shown(self.running_web_build_id)),
            NonsecretField(name="service_active_state", value=self.service_active_state),
            NonsecretField(name="service_unit_file_state", value=self.service_unit_file_state),
            NonsecretField(name="service_main_pid", value=_shown(self.service_main_pid)),
            NonsecretField(name="reload_mode", value=self.reload_mode),
            NonsecretField(name="space_id", value=_shown(self.space_id)),
            NonsecretField(name="instance_id", value=_shown(self.instance_id)),
            NonsecretField(name="process_pid", value=_shown(self.process_pid)),
            NonsecretField(name="data_dir_id", value=self.data_dir_id),
            NonsecretField(name="control_socket_status", value=self.control_socket_status),
            NonsecretField(name="provider_check_status", value=self.provider_check_status),
            NonsecretField(name="dependencies_ready", value=self.dependencies_ready),
            NonsecretField(name="dependency_versions", value=self.dependency_versions),
            NonsecretField(name="backup_status", value=self.backup_status),
            NonsecretField(name="backup_destination", value=_shown(self.backup_destination)),
            NonsecretField(name="backup_schedule", value=_shown(self.backup_schedule)),
            NonsecretField(name="backup_retention", value=_shown(self.backup_retention)),
            NonsecretField(
                name="backup_recipient_fingerprint",
                value=_shown(self.backup_recipient_fingerprint),
            ),
            NonsecretField(
                name="backup_timer_active_state",
                value=self.backup_timer_active_state,
            ),
            NonsecretField(
                name="backup_timer_unit_file_state",
                value=self.backup_timer_unit_file_state,
            ),
            NonsecretField(
                name="last_backup_at",
                value=(self.last_backup_at.isoformat() if self.last_backup_at else _NO_VALUE),
            ),
            NonsecretField(name="last_backup_archive", value=_shown(self.last_backup_archive)),
            NonsecretField(
                name="last_backup_captured_bytes",
                value=_shown(self.last_backup_captured_bytes),
            ),
            NonsecretField(
                name="last_backup_protected_projects",
                value=_shown(self.last_backup_protected_projects),
            ),
            NonsecretField(
                name="last_backup_uncaptured_projects",
                value=_shown(self.last_backup_uncaptured_projects),
            ),
            NonsecretField(name="last_backup_failure", value=_shown(self.last_backup_failure)),
            NonsecretField(name="update_operation_state", value=self.update_operation_state),
            NonsecretField(
                name="update_candidate_commit",
                value=_shown(self.update_candidate_commit),
            ),
            NonsecretField(
                name="update_restored_commit",
                value=_shown(self.update_restored_commit),
            ),
            NonsecretField(name="update_failure", value=_shown(self.update_failure)),
            NonsecretField(name="problems", value=_problem_text(self.problems)),
        )


@dataclass(frozen=True)
class _BackupDoctorSummary:
    status: DoctorBackupState
    destination: str | None = None
    schedule: str | None = None
    retention: int | None = None
    recipient_fingerprint: str | None = None
    timer_active_state: str = "not_configured"
    timer_unit_file_state: str = "not_configured"
    last_at: datetime | None = None
    archive: str | None = None
    captured_bytes: int | None = None
    protected_projects: int | None = None
    uncaptured_projects: int | None = None
    failure: str | None = None


@dataclass(frozen=True)
class _DoctorUpdateSummary:
    state: str
    candidate_commit: str | None = None
    restored_commit: str | None = None
    failure: str | None = None


class ServerDoctorMachine(Protocol):
    def inspect(self) -> ServerDoctorReport: ...


class ReadOnlyRunner(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


ConfigLoader = Callable[[Path], InstalledServerConfig]
MetadataReader = Callable[[Path], ServerMetadata]
ControlProbe = Callable[[ServerMetadata, int], ServerControlProbeResult]


def prepare_doctor_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    machine: ServerDoctorMachine | None = None,
) -> PreparedServerCommand:
    if request.command != "server doctor":
        raise ValueError("prepare_doctor_command requires one server doctor request")
    target = MachineTarget(host=identity.host, os_account="rcp")
    pending = ServerStep(
        number=1,
        title="Inspect the installed team server",
        purpose=(
            "Read exact source, release, process, service, filesystem, control, and dependency "
            "identity without changing machine or application state."
        ),
        performed_by="system",
        target=target,
        phase="server_doctor",
        state="pending",
        expected_success=(
            "One secret-free report distinguishes aligned, update, restart, and problem states."
        ),
        message="RCP will inspect the installed team server without changing it.",
    )
    plan = ServerPlanEvent(command=request.command, timestamp=datetime.now(UTC), steps=(pending,))
    resolved_machine = machine or LinuxServerDoctorMachine()

    def execute(emitter: ServerEventEmitter, _input_stream: BinaryIO) -> None:
        emitter.emit_step(
            pending.model_copy(
                update={
                    "state": "running",
                    "message": "Reading installed identities and checking their exact relationships.",
                }
            )
        )
        report = resolved_machine.inspect()
        succeeded = report.overall_state != "problems"
        emitter.emit_step(
            pending.model_copy(
                update={
                    "state": "succeeded" if succeeded else "failed",
                    "message": (
                        f"The installed team server is {report.overall_state.replace('_', ' ')}."
                        if succeeded
                        else (
                            f"Doctor found {len(report.problems)} problem(s); the report names "
                            "every completed readback without changing state."
                        )
                    ),
                    "fields": report.fields(),
                }
            )
        )

    return PreparedServerCommand(plan=plan, execute=execute)


class LinuxServerDoctorMachine:
    """Concrete read-only Ubuntu/systemd implementation for the installed layout."""

    def __init__(
        self,
        layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
        *,
        config_loader: ConfigLoader | None = None,
        metadata_reader: MetadataReader | None = None,
        control_probe: ControlProbe | None = None,
        runner: ReadOnlyRunner | None = None,
        service_identity: tuple[int, int] | None = None,
        root_identity: tuple[int, int] = (0, 0),
    ) -> None:
        self.layout = layout
        self._config_loader = config_loader or load_installed_server_config
        self._metadata_reader = metadata_reader or read_server_metadata
        self._control_probe = control_probe or _probe_control
        self._runner = runner or _run_read_only
        self._service_identity = service_identity
        self._root_identity = root_identity

    def inspect(self) -> ServerDoctorReport:
        problems: list[str] = []

        def add_problem(message: str) -> None:
            _add_problem(problems, message)

        service_uid, service_gid = self._resolve_service_identity(add_problem)
        root_uid, root_gid = self._root_identity
        config = self._load_config(add_problem)

        self._inspect_fixed_paths(
            config,
            service_uid=service_uid,
            service_gid=service_gid,
            root_uid=root_uid,
            root_gid=root_gid,
            add_problem=add_problem,
        )
        managed_head, upstream_head, source_state = self._inspect_source(
            config,
            service_uid=service_uid,
            service_gid=service_gid,
            add_problem=add_problem,
        )
        current_commit = self._inspect_current_release(
            service_uid=service_uid,
            service_gid=service_gid,
            root_uid=root_uid,
            root_gid=root_gid,
            add_problem=add_problem,
        )
        current_web_build_id = self._inspect_release(
            current_commit,
            label="current",
            service_uid=service_uid,
            service_gid=service_gid,
            add_problem=add_problem,
        )
        active_state, unit_file_state, main_pid, reload_mode = self._inspect_service(add_problem)
        restore_pending = self._inspect_restore(
            service_uid=service_uid,
            add_problem=add_problem,
        )
        if restore_pending and (active_state != "inactive" or main_pid not in {None, 0}):
            add_problem("unfinished replacement restore requires rcp.service to remain stopped")
        (
            metadata,
            probe,
            control_status,
        ) = self._inspect_process(
            service_uid=service_uid,
            main_pid=main_pid,
            add_problem=add_problem,
        )
        running_commit = metadata.running_commit if metadata is not None else None
        running_web_build_id = metadata.web_build_id if metadata is not None else None
        if running_commit is not None:
            observed_running_web = self._inspect_release(
                running_commit,
                label="running",
                service_uid=service_uid,
                service_gid=service_gid,
                add_problem=add_problem,
            )
            if observed_running_web != running_web_build_id:
                add_problem("running Web bundle differs from the process startup identity")

        release_state = release_relationship(managed_head, current_commit, running_commit)
        if release_state == "inconsistent":
            add_problem("managed, current, and running commits have an inconsistent relationship")
        candidate_commit = (
            managed_head
            if managed_head is not None
            and running_commit is not None
            and managed_head != running_commit
            else None
        )
        dependencies_ready, dependency_versions = self._inspect_dependencies(
            current_commit,
            add_problem,
        )
        backup = self._inspect_backup(
            config,
            service_uid=service_uid,
            add_problem=add_problem,
        )
        update = self._inspect_update(
            service_uid=service_uid,
            installation_id=config.installation_id if config is not None else None,
            add_problem=add_problem,
        )
        provider_check_status: Literal["available", "unavailable"] = (
            "available"
            if probe is not None
            and "provider_readiness_plan" in probe.operations
            and "provider_readiness_check" in probe.operations
            else "unavailable"
        )
        if control_status == "healthy" and provider_check_status == "unavailable":
            add_problem("private control socket does not offer provider readiness")
        if control_status == "healthy" and (
            probe is None
            or "member_removal_plan" not in probe.operations
            or "member_removal_advance" not in probe.operations
        ):
            add_problem("private control socket does not offer member removal")
        if probe is not None:
            for pending in getattr(probe, "pending_member_removals", ()):
                for problem in _member_removal_problems(pending):
                    add_problem(problem)
        overall_state = _overall_state(
            problems,
            release_state=release_state,
            source_state=source_state,
        )
        return ServerDoctorReport(
            overall_state=overall_state,
            installation_id=config.installation_id if config is not None else None,
            service_account=self.layout.service_account,
            data_dir=str(self.layout.data_dir),
            source_root=str(self.layout.source_checkout),
            releases_root=str(self.layout.releases_root),
            configured_origin=config.source.origin if config is not None else None,
            configured_branch=config.source.branch if config is not None else None,
            source_public_key_fingerprint=(
                config.source.public_key_fingerprint if config is not None else None
            ),
            managed_main_head=managed_head,
            upstream_head=upstream_head,
            candidate_commit=candidate_commit,
            current_commit=current_commit,
            running_commit=running_commit,
            release_state=release_state,
            source_state=source_state,
            current_web_build_id=current_web_build_id,
            running_web_build_id=running_web_build_id,
            service_active_state=active_state,
            service_unit_file_state=unit_file_state,
            service_main_pid=main_pid,
            reload_mode=reload_mode,
            space_id=probe.space_id if probe is not None else None,
            instance_id=metadata.instance_id if metadata is not None else None,
            process_pid=metadata.pid if metadata is not None else None,
            data_dir_id=data_dir_identity(self.layout.data_dir),
            control_socket_status=control_status,
            provider_check_status=provider_check_status,
            dependencies_ready=dependencies_ready,
            dependency_versions=dependency_versions,
            backup_status=backup.status,
            backup_destination=backup.destination,
            backup_schedule=backup.schedule,
            backup_retention=backup.retention,
            backup_recipient_fingerprint=backup.recipient_fingerprint,
            backup_timer_active_state=backup.timer_active_state,
            backup_timer_unit_file_state=backup.timer_unit_file_state,
            last_backup_at=backup.last_at,
            last_backup_archive=backup.archive,
            last_backup_captured_bytes=backup.captured_bytes,
            last_backup_protected_projects=backup.protected_projects,
            last_backup_uncaptured_projects=backup.uncaptured_projects,
            last_backup_failure=backup.failure,
            update_operation_state=update.state,
            update_candidate_commit=update.candidate_commit,
            update_restored_commit=update.restored_commit,
            update_failure=update.failure,
            problems=tuple(problems),
        )

    def _inspect_restore(
        self,
        *,
        service_uid: int,
        add_problem: Callable[[str], None],
    ) -> bool:
        from rcp.server_ops.restore import RestoreRefused, unfinished_restore_operation

        try:
            operation = unfinished_restore_operation(
                self.layout,
                expected_uid=service_uid,
            )
        except (OSError, RestoreRefused):
            add_problem("restore operation state is unsafe; preserve it and rerun server restore")
            return True
        if operation is None:
            return False
        add_problem("unfinished replacement restore requires sudo rcp server restore re-entry")
        return True

    def _inspect_update(
        self,
        *,
        service_uid: int,
        installation_id: str | None,
        add_problem: Callable[[str], None],
    ) -> _DoctorUpdateSummary:
        from rcp.server_ops.update_checkpoint import (
            UpdateCheckpointRefused,
            unfinished_rollback_journals,
        )
        from rcp.server_ops.update_cutover import (
            UpdateCutoverRefused,
            update_operation_receipts,
        )

        try:
            operations = update_operation_receipts(
                self.layout.update_checkpoints_root,
                expected_uid=service_uid,
            )
            journals = unfinished_rollback_journals(
                self.layout.update_checkpoints_root,
                expected_uid=service_uid,
            )
        except (OSError, UpdateCheckpointRefused, UpdateCutoverRefused):
            add_problem("update maintenance receipts or rollback journals are unsafe")
            return _DoctorUpdateSummary(state="unavailable")
        if journals:
            add_problem("unfinished update rollback requires sudo rcp server update re-entry")
        if not operations:
            return _DoctorUpdateSummary(state="none")
        _path, latest, _digest = max(
            operations,
            key=lambda item: (item[1].updated_at, item[1].operation_id),
        )
        if installation_id is not None and latest.installation_id != installation_id:
            add_problem("latest update receipt belongs to another server installation")
        runtime_failure = getattr(latest, "runtime_failure", None)
        if not latest.terminal:
            add_problem("unfinished source update requires sudo rcp server update re-entry")
        elif latest.state in {"committed", "rolled_back"} and runtime_failure is not None:
            add_problem(
                "selected source release needs safe runtime restart via sudo rcp server update"
            )
        return _DoctorUpdateSummary(
            state=latest.state,
            candidate_commit=latest.candidate_commit,
            restored_commit=(latest.base_commit if latest.state == "rolled_back" else None),
            failure=runtime_failure or latest.failure,
        )

    def _resolve_service_identity(
        self,
        add_problem: Callable[[str], None],
    ) -> tuple[int, int]:
        if self._service_identity is not None:
            return self._service_identity
        try:
            account = pwd.getpwnam(self.layout.service_account)
        except KeyError:
            add_problem("the rcp service account is unavailable")
            return -1, -1
        return account.pw_uid, account.pw_gid

    def _load_config(
        self,
        add_problem: Callable[[str], None],
    ) -> InstalledServerConfig | None:
        try:
            config = self._config_loader(self.layout.config_path)
        except (OSError, ValueError):
            add_problem("installed server configuration is missing or invalid")
            return None
        if config.paths.model_dump() != self.layout.recorded_paths():
            add_problem("installed server configuration does not name the fixed layout")
        if config.service_account != self.layout.service_account:
            add_problem("installed server configuration names the wrong service account")
        if config.service_unit != self.layout.service_unit_name:
            add_problem("installed server configuration names the wrong systemd unit")
        return config

    def _inspect_fixed_paths(
        self,
        config: InstalledServerConfig | None,
        *,
        service_uid: int,
        service_gid: int,
        root_uid: int,
        root_gid: int,
        add_problem: Callable[[str], None],
    ) -> None:
        service_directories = (
            (self.layout.service_home, "service home"),
            (self.layout.server_root, "server root"),
            (self.layout.releases_root, "releases root"),
            (self.layout.data_dir, "data directory"),
            (self.layout.projects_root, "projects root"),
            (self.layout.credentials_root, "credentials root"),
            (self.layout.update_checkpoints_root, "update checkpoints root"),
            (self.layout.restore_operations_root, "restore operations root"),
            (self.layout.codex_state_root, "Codex state root"),
            (self.layout.claude_state_root, "Claude state root"),
            (self.layout.ssh_state_root, "SSH state root"),
        )
        for path, label in service_directories:
            _check_path(
                path,
                label=label,
                kind="directory",
                uid=service_uid,
                gid=service_gid,
                mode=0o700,
                add_problem=add_problem,
            )
        _check_path(
            self.layout.source_checkout,
            label="managed source checkout",
            kind="directory",
            uid=service_uid,
            gid=service_gid,
            mode=None,
            add_problem=add_problem,
        )
        _check_path(
            self.layout.config_path.parent,
            label="server configuration directory",
            kind="directory",
            uid=root_uid,
            gid=service_gid,
            mode=0o750,
            add_problem=add_problem,
        )
        for path, label, mode in (
            (self.layout.config_path, "server configuration", 0o640),
            (self.layout.cli_wrapper, "CLI wrapper", 0o755),
            (self.layout.systemd_unit, "systemd unit", 0o644),
        ):
            expected_gid = service_gid if path == self.layout.config_path else root_gid
            _check_path(
                path,
                label=label,
                kind="file",
                uid=root_uid,
                gid=expected_gid,
                mode=mode,
                add_problem=add_problem,
            )
        _check_path(
            self.layout.runtime_dir,
            label="runtime directory",
            kind="directory",
            uid=service_uid,
            gid=service_gid,
            mode=0o700,
            add_problem=add_problem,
        )
        _check_path(
            self.layout.control_socket,
            label="control socket",
            kind="socket",
            uid=service_uid,
            gid=service_gid,
            mode=0o600,
            add_problem=add_problem,
        )
        for path, label in (
            (self.layout.data_dir / "rcp.sqlite3", "application database"),
            (self.layout.data_dir / "rcp.lock", "application instance lock"),
            (metadata_path(self.layout.data_dir), "server metadata"),
        ):
            _check_path(
                path,
                label=label,
                kind="file",
                uid=service_uid,
                gid=service_gid,
                mode=0o600,
                add_problem=add_problem,
            )
        for suffix in ("-wal", "-shm"):
            path = self.layout.data_dir / f"rcp.sqlite3{suffix}"
            if os.path.lexists(path):
                _check_path(
                    path,
                    label=f"application database {suffix.removeprefix('-')}",
                    kind="file",
                    uid=service_uid,
                    gid=service_gid,
                    mode=0o600,
                    add_problem=add_problem,
                )
        if config is not None and config.source.authentication == "deploy_key":
            for path, label, mode in (
                (
                    self.layout.credentials_root / "source_ed25519",
                    "source private key",
                    0o600,
                ),
                (
                    self.layout.credentials_root / "source_ed25519.pub",
                    "source public key",
                    0o644,
                ),
            ):
                _check_path(
                    path,
                    label=label,
                    kind="file",
                    uid=service_uid,
                    gid=service_gid,
                    mode=mode,
                    add_problem=add_problem,
                )

    def _inspect_source(
        self,
        config: InstalledServerConfig | None,
        *,
        service_uid: int,
        service_gid: int,
        add_problem: Callable[[str], None],
    ) -> tuple[str | None, str | None, DoctorSourceState]:
        if config is None:
            return None, None, "unavailable"
        source = self.layout.source_checkout
        safe = all(
            (
                _check_path(
                    self.layout.service_home,
                    label="service home",
                    kind="directory",
                    uid=service_uid,
                    gid=service_gid,
                    mode=0o700,
                    add_problem=add_problem,
                ),
                _check_path(
                    self.layout.server_root,
                    label="server root",
                    kind="directory",
                    uid=service_uid,
                    gid=service_gid,
                    mode=0o700,
                    add_problem=add_problem,
                ),
                _check_path(
                    source,
                    label="managed source checkout",
                    kind="directory",
                    uid=service_uid,
                    gid=service_gid,
                    mode=None,
                    add_problem=add_problem,
                ),
            )
        )
        if not safe:
            return None, None, "unavailable"
        origin = self._git_text(source, ("remote", "get-url", "origin"))
        branch = self._git_text(source, ("symbolic-ref", "--short", "HEAD"))
        dirty = self._git_text(source, ("status", "--porcelain", "--untracked-files=all"))
        managed = self._git_commit(source, "HEAD")
        upstream = self._git_commit(source, "origin/main")
        if origin is None or origin != config.source.origin:
            add_problem("managed source origin differs from installed configuration")
        if branch is None or branch != config.source.branch:
            add_problem("managed source is not checked out on configured main")
        if dirty is None:
            add_problem("managed source cleanliness could not be read")
        elif dirty:
            add_problem("managed source has tracked or untracked changes")
        if managed is None:
            add_problem("managed source HEAD is unavailable or invalid")
        if upstream is None:
            add_problem("the last fetched origin/main identity is unavailable or invalid")
        if managed is None or upstream is None:
            return managed, upstream, "unavailable"
        if managed == upstream:
            return managed, upstream, "aligned"
        if self._git_is_ancestor(source, managed, upstream):
            return managed, upstream, "update_available"
        if self._git_is_ancestor(source, upstream, managed):
            add_problem("managed main is ahead of the last fetched origin/main")
            return managed, upstream, "local_ahead"
        add_problem("managed main and the last fetched origin/main have diverged")
        return managed, upstream, "diverged"

    def _inspect_current_release(
        self,
        *,
        service_uid: int,
        service_gid: int,
        root_uid: int,
        root_gid: int,
        add_problem: Callable[[str], None],
    ) -> str | None:
        current = self.layout.current_release
        try:
            info = current.lstat()
            target = Path(os.readlink(current))
        except (FileNotFoundError, OSError):
            add_problem("current release pointer is missing or unreadable")
            return None
        if not stat.S_ISLNK(info.st_mode) or (info.st_uid, info.st_gid) != (root_uid, root_gid):
            add_problem("current release pointer has the wrong type or owner")
            return None
        if (
            not target.is_absolute()
            or target.parent != self.layout.releases_root
            or _FULL_GIT_COMMIT.fullmatch(target.name) is None
            or target != self.layout.release_dir(target.name)
        ):
            add_problem("current release pointer does not name one canonical release")
            return None
        if not all(
            (
                _check_path(
                    self.layout.service_home,
                    label="service home",
                    kind="directory",
                    uid=service_uid,
                    gid=service_gid,
                    mode=0o700,
                    add_problem=add_problem,
                ),
                _check_path(
                    self.layout.server_root,
                    label="server root",
                    kind="directory",
                    uid=service_uid,
                    gid=service_gid,
                    mode=0o700,
                    add_problem=add_problem,
                ),
                _check_path(
                    self.layout.releases_root,
                    label="releases root",
                    kind="directory",
                    uid=service_uid,
                    gid=service_gid,
                    mode=0o700,
                    add_problem=add_problem,
                ),
                _check_path(
                    target,
                    label="current release directory",
                    kind="directory",
                    uid=service_uid,
                    gid=service_gid,
                    mode=None,
                    add_problem=add_problem,
                ),
            )
        ):
            return None
        return target.name

    def _inspect_release(
        self,
        commit: str | None,
        *,
        label: str,
        service_uid: int,
        service_gid: int,
        add_problem: Callable[[str], None],
    ) -> str | None:
        if commit is None:
            return None
        release = self.layout.release_dir(commit)
        if not _check_path(
            release,
            label=f"{label} release directory",
            kind="directory",
            uid=service_uid,
            gid=service_gid,
            mode=None,
            add_problem=add_problem,
        ):
            return None
        head = self._git_commit(release, "HEAD")
        if head != commit:
            add_problem(f"{label} release Git identity differs from its directory name")
        dirty = self._git_text(release, ("status", "--porcelain", "--untracked-files=all"))
        if dirty is None:
            add_problem(f"{label} release cleanliness could not be read")
        elif dirty:
            add_problem(f"{label} release has tracked or untracked changes")
        web_root = release / "web" / "dist"
        artifacts_safe = True
        for path, artifact, kind in (
            (release / ".venv" / "bin" / "rcp", "Python entry point", "file"),
            (web_root, "Web bundle", "directory"),
            (web_root / "index.html", "Web entry point", "file"),
        ):
            artifacts_safe = (
                _check_descendant_path(
                    path,
                    root=release,
                    label=f"{label} release {artifact}",
                    kind=kind,
                    uid=service_uid,
                    gid=service_gid,
                    mode=None,
                    add_problem=add_problem,
                )
                and artifacts_safe
            )
        if not artifacts_safe:
            return None
        try:
            return web_build_identity(web_root)
        except ServerMetadataError:
            add_problem(f"{label} Web bundle is missing, unsafe, or exceeds its bound")
            return None

    def _inspect_service(
        self,
        add_problem: Callable[[str], None],
    ) -> tuple[str, str, int | None, Literal["disabled", "unknown"]]:
        active = self._systemd_property("ActiveState")
        enabled = self._systemd_property("UnitFileState")
        raw_pid = self._systemd_property("MainPID")
        reload_needed = self._systemd_property("NeedDaemonReload")
        fragment_path = self._systemd_property("FragmentPath", max_length=4096)
        drop_ins = self._systemd_property("DropInPaths", allow_empty=True, max_length=4096)
        main_pid = int(raw_pid) if raw_pid is not None and raw_pid.isdigit() else None
        if active != "active":
            add_problem("rcp.service is not active")
        if enabled != "enabled":
            add_problem("rcp.service is not enabled")
        if main_pid is None or main_pid <= 0:
            add_problem("rcp.service does not report one live main process")
            main_pid = None
        try:
            unit = self.layout.systemd_unit.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            unit = None
        reload_mode: Literal["disabled", "unknown"] = "unknown"
        loaded_unit_matches = (
            reload_needed == "no"
            and fragment_path == str(self.layout.systemd_unit)
            and drop_ins == ""
        )
        if not loaded_unit_matches:
            add_problem("systemd has not loaded the exact unit without overrides")
        if (
            unit == server_service_unit_text()
            and "ExecReload=" not in unit
            and "--reload" not in unit
            and loaded_unit_matches
        ):
            reload_mode = "disabled"
        else:
            add_problem("installed systemd unit differs from the non-reloading service contract")
        return active or "unavailable", enabled or "unavailable", main_pid, reload_mode

    def _inspect_process(
        self,
        *,
        service_uid: int,
        main_pid: int | None,
        add_problem: Callable[[str], None],
    ) -> tuple[ServerMetadata | None, ServerControlProbeResult | None, str]:
        try:
            metadata = self._metadata_reader(self.layout.data_dir)
        except (OSError, ServerMetadataError, ValueError):
            add_problem("running server metadata is missing or invalid")
            return None, None, "unavailable"
        expected_data_dir_id = data_dir_identity(self.layout.data_dir)
        if metadata.data_dir_id != expected_data_dir_id:
            add_problem("running process metadata names a different data directory")
        if metadata.owner_kind != "cli":
            add_problem("running process is not owned by the installed CLI service")
        if metadata.control_socket != str(self.layout.control_socket):
            add_problem("running process metadata names a different control socket")
        if (
            metadata.data_dir_id != expected_data_dir_id
            or metadata.owner_kind != "cli"
            or metadata.control_socket != str(self.layout.control_socket)
        ):
            return metadata, None, "identity_mismatch"
        if metadata.running_commit is None or metadata.web_build_id is None:
            add_problem("running process did not publish an exact release and Web identity")
        if main_pid is not None and metadata.pid != main_pid:
            add_problem("systemd and running metadata name different process ids")
        try:
            probe = self._control_probe(metadata, service_uid)
        except (OSError, ServerControlError, ServerMetadataError, ValueError):
            add_problem("private control socket did not authenticate the running process")
            return metadata, None, "unavailable"
        if (
            probe.instance_id != metadata.instance_id
            or probe.pid != metadata.pid
            or probe.data_dir_id != expected_data_dir_id
        ):
            add_problem("control socket identity differs from running metadata")
            return metadata, probe, "identity_mismatch"
        return metadata, probe, "healthy"

    def _inspect_dependencies(
        self,
        current_commit: str | None,
        add_problem: Callable[[str], None],
    ) -> tuple[bool, str]:
        probes: tuple[tuple[str, tuple[str, ...], Callable[[str], bool]], ...] = (
            ("git", ("git", "--version"), lambda value: value.startswith("git version ")),
            ("node", ("node", "--version"), lambda value: _major(value) == 24),
            ("npm", ("npm", "--version"), lambda value: _major(value) is not None),
            ("uv", ("uv", "--version"), lambda value: value.startswith("uv ")),
            ("age", ("age", "--version"), lambda value: _major(value) == 1),
            ("ssh", ("ssh", "-V"), lambda value: value.startswith("OpenSSH_")),
        )
        versions: list[str] = []
        ready = True
        for name, argv, validator in probes:
            value = self._version(argv)
            if value is None or not validator(value):
                ready = False
                versions.append(f"{name}=unavailable")
            else:
                versions.append(f"{name}={_version_token(value)}")
        python = (
            self.layout.release_dir(current_commit) / ".venv" / "bin" / "python"
            if current_commit is not None
            else None
        )
        python_value = self._version((str(python), "--version")) if python is not None else None
        if python_value is None or not python_value.startswith("Python 3.12."):
            ready = False
            versions.append("python=unavailable")
        else:
            versions.append(f"python={_version_token(python_value)}")
        if not ready:
            add_problem("one or more installed runtime dependencies are unavailable or unsupported")
        return ready, ",".join(versions)

    def _inspect_backup(
        self,
        config: InstalledServerConfig | None,
        *,
        service_uid: int,
        add_problem: Callable[[str], None],
    ) -> _BackupDoctorSummary:
        if config is None or config.backup is None:
            return _BackupDoctorSummary(status="not_configured")
        import hashlib

        from rcp.server_ops.backup import (
            BackupRunRefused,
            read_backup_archive_receipt,
            read_backup_outcome,
        )

        backup = config.backup
        active = self._systemd_property("ActiveState", unit="rcp-backup.timer")
        enabled = self._systemd_property("UnitFileState", unit="rcp-backup.timer")
        if active != "active" or enabled != "enabled":
            add_problem("configured backup timer is not both active and enabled")
        common = {
            "destination": backup.destination,
            "schedule": backup.schedule,
            "retention": backup.retention,
            "recipient_fingerprint": hashlib.sha256(
                backup.age_recipient.encode("ascii")
            ).hexdigest(),
            "timer_active_state": active or "unavailable",
            "timer_unit_file_state": enabled or "unavailable",
        }
        try:
            outcome = read_backup_outcome(self.layout, expected_uid=service_uid)
        except FileNotFoundError:
            add_problem("configured backup has no durable run status")
            return _BackupDoctorSummary(status="never_run", **common)
        except (OSError, ValueError):
            add_problem("configured backup status is missing, unsafe, or invalid")
            return _BackupDoctorSummary(status="unavailable", **common)
        if (
            outcome.installation_id != config.installation_id
            or outcome.destination != backup.destination
        ):
            add_problem("backup status belongs to another installation or destination")
            return _BackupDoctorSummary(status="unavailable", **common)

        archive = outcome.archive
        archive_path = None
        captured_bytes = None
        protected_projects = None
        uncaptured_projects = None
        if archive is not None:
            archive_path = str(Path(backup.destination) / archive.archive_name)
            captured_bytes = archive.captured_bytes
            protected_projects = archive.protected_project_count
            uncaptured_projects = archive.uncaptured_project_count
            try:
                observed = read_backup_archive_receipt(
                    Path(f"{archive_path}.receipt.json"),
                    expected_destination=Path(backup.destination),
                    expected_installation_id=config.installation_id,
                    expected_uid=service_uid,
                    verify_digest=False,
                    expected_receipt_sha256=outcome.archive_receipt_sha256,
                )
            except (BackupRunRefused, OSError, ValueError):
                add_problem("last protected backup no longer matches its archive receipt")
            else:
                if observed != archive:
                    add_problem("backup status and archive receipt disagree")

        if outcome.status == "failure":
            add_problem("the last protected backup failed; inspect last_backup_failure")
        elif outcome.status == "partial":
            add_problem("the last protected backup is partial; inspect uncaptured projects")
        return _BackupDoctorSummary(
            status=outcome.status,
            last_at=outcome.completed_at,
            archive=archive_path,
            captured_bytes=captured_bytes,
            protected_projects=protected_projects,
            uncaptured_projects=uncaptured_projects,
            failure=outcome.failure,
            **common,
        )

    def _git_text(self, root: Path, argv: tuple[str, ...]) -> str | None:
        result = self._runner(
            ("git", "-c", f"safe.directory={root}", "-C", str(root), *argv),
            cwd=root,
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
            return None
        return value

    def _git_commit(self, root: Path, revision: str) -> str | None:
        value = self._git_text(root, ("rev-parse", "--verify", f"{revision}^{{commit}}"))
        return value if value is not None and _FULL_GIT_COMMIT.fullmatch(value) else None

    def _git_is_ancestor(self, root: Path, first: str, second: str) -> bool:
        result = self._runner(
            (
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                first,
                second,
            ),
            cwd=root,
        )
        return result.returncode == 0

    def _systemd_property(
        self,
        name: str,
        *,
        unit: str | None = None,
        allow_empty: bool = False,
        max_length: int = 120,
    ) -> str | None:
        result = self._runner(
            (
                "systemctl",
                "show",
                f"--property={name}",
                "--value",
                unit or self.layout.service_unit_name,
            )
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        if not value and allow_empty:
            return ""
        return value if value and "\n" not in value and len(value) <= max_length else None

    def _version(self, argv: tuple[str, ...]) -> str | None:
        result = self._runner(argv)
        if result.returncode != 0:
            return None
        lines = (result.stdout or result.stderr).strip().splitlines()
        if not lines or len(lines[-1]) > 240:
            return None
        return lines[-1]


def _member_removal_problems(
    snapshot: ServerControlMemberSnapshot,
) -> tuple[str, ...]:
    return (
        f"member removal remains in progress: {snapshot.member_id}",
        *(
            f"member removal has a live task: {operation_id}"
            for operation_id in snapshot.active_task_ids
        ),
        *(
            f"member removal has a live episode: {episode_id}"
            for episode_id in snapshot.active_episode_ids
        ),
    )


def release_relationship(
    managed: str | None,
    current: str | None,
    running: str | None,
) -> DoctorReleaseState:
    if managed is None or current is None or running is None:
        return "unavailable"
    if managed == current == running:
        return "aligned"
    if current == running and managed != current:
        return "candidate_pending"
    if managed == current and running != current:
        return "restart_pending"
    return "inconsistent"


def _overall_state(
    problems: list[str],
    *,
    release_state: DoctorReleaseState,
    source_state: DoctorSourceState,
) -> DoctorOverallState:
    if problems:
        return "problems"
    if release_state == "candidate_pending":
        return "candidate_pending"
    if release_state == "restart_pending":
        return "restart_pending"
    if source_state == "update_available":
        return "update_available"
    return "healthy"


def _probe_control(metadata: ServerMetadata, expected_uid: int) -> ServerControlProbeResult:
    return ServerControlClient(metadata, expected_server_uid=expected_uid).probe()


def _run_read_only(
    argv: tuple[str, ...],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {
        "HOME": str(DEFAULT_SERVER_LAYOUT.service_home),
        "USER": DEFAULT_SERVER_LAYOUT.service_account,
        "LOGNAME": DEFAULT_SERVER_LAYOUT.service_account,
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(argv, 126, "", "")


def _check_path(
    path: Path,
    *,
    label: str,
    kind: Literal["directory", "file", "socket"],
    uid: int,
    gid: int,
    mode: int | None,
    add_problem: Callable[[str], None],
) -> bool:
    try:
        info = path.lstat()
    except OSError:
        add_problem(f"{label} is missing or unreadable")
        return False
    kind_matches = {
        "directory": stat.S_ISDIR,
        "file": stat.S_ISREG,
        "socket": stat.S_ISSOCK,
    }[kind](info.st_mode)
    mode_matches = mode is None or stat.S_IMODE(info.st_mode) == mode
    if not kind_matches or (info.st_uid, info.st_gid) != (uid, gid) or not mode_matches:
        add_problem(f"{label} has the wrong type, owner, group, or mode")
        return False
    return True


def _check_descendant_path(
    path: Path,
    *,
    root: Path,
    label: str,
    kind: Literal["directory", "file", "socket"],
    uid: int,
    gid: int,
    mode: int | None,
    add_problem: Callable[[str], None],
) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        add_problem(f"{label} is outside its release directory")
        return False
    current = root
    for component in relative.parts[:-1]:
        current /= component
        try:
            info = current.lstat()
        except OSError:
            add_problem(f"{label} has missing or unreadable parent directories")
            return False
        if not stat.S_ISDIR(info.st_mode) or (info.st_uid, info.st_gid) != (uid, gid):
            add_problem(f"{label} has unsafe parent-directory ancestry")
            return False
    return _check_path(
        path,
        label=label,
        kind=kind,
        uid=uid,
        gid=gid,
        mode=mode,
        add_problem=add_problem,
    )


def _major(value: str) -> int | None:
    match = re.search(r"(?<![0-9])v?([0-9]+)(?:\.[0-9]+)", value)
    return int(match.group(1)) if match else None


def _version_token(value: str) -> str:
    for token in value.replace(",", " ").split():
        normalized = token.removeprefix("v")
        if _SAFE_VERSION.fullmatch(normalized) and any(character.isdigit() for character in token):
            return normalized
    return "recognized"


def _shown(value: str | int | None) -> str | int:
    return _NO_VALUE if value is None else value


def _add_problem(problems: list[str], message: str) -> None:
    if message not in problems:
        problems.append(message)


def _problem_text(problems: tuple[str, ...]) -> str:
    if not problems:
        return _NO_VALUE
    selected: list[str] = []
    total = 0
    for problem in problems:
        addition = len(problem) + (2 if selected else 0)
        if total + addition > 1800:
            remaining = len(problems) - len(selected)
            selected.append(f"{remaining} additional checks failed")
            break
        selected.append(problem)
        total += addition
    return "; ".join(selected)


__all__ = [
    "LinuxServerDoctorMachine",
    "ServerDoctorReport",
    "prepare_doctor_command",
    "release_relationship",
]
