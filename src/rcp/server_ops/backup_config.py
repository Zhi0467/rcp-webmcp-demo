"""Root-owned backup configuration and verified systemd schedule activation."""

from __future__ import annotations

import fcntl
import importlib.resources
import os
import pwd
import stat
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Protocol, TypeVar

from rcp.limits import SERVER_BACKUP_CONFIGURATION_TIMEOUT_SECONDS
from rcp.server_ops.cli import (
    CallerIdentity,
    PreparedServerCommand,
    ServerEventEmitter,
)
from rcp.server_ops.config import (
    DEFAULT_BACKUP_SCHEDULE,
    InstalledServerConfig,
    ServerBackupConfig,
    load_installed_server_config,
    write_installed_server_config,
)
from rcp.server_ops.install import (
    InstallRefused,
    enable_backup_timer,
    fence_backup_timer_before_unit_change,
    install_backup_unit_files,
    read_systemd_unit_state,
    reload_and_disable_backup_timer,
    run_backup_service_once,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.models import (
    MachineTarget,
    NonsecretField,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
)

_TIMER_PLACEHOLDER = "@RCP_BACKUP_ON_CALENDAR@"
_UNIT_MODE = 0o644
_LOCK_MODE = 0o600
_LOCK_NAME = ".server-configuration.lock"
_PENDING_NAME = ".backup-configuration.pending.toml"


class BackupConfigurationRefused(InstallRefused):
    """A known configuration refusal whose secret-free message may be rendered."""


class _ReportedBackupConfigurationFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class BackupConfigurationReadback:
    config: ServerBackupConfig
    timer_active_state: str
    timer_unit_file_state: str


class BackupConfigurationMachine(Protocol):
    def validate_destination(self, config: ServerBackupConfig) -> None: ...

    def persist_and_install(self, config: ServerBackupConfig) -> None: ...

    def readback(self, config: ServerBackupConfig) -> BackupConfigurationReadback: ...


def prepare_backup_configure_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    machine: BackupConfigurationMachine | None = None,
) -> PreparedServerCommand:
    if (
        request.command != "server backup configure"
        or request.backup_destination is None
        or request.backup_schedule is None
        or request.backup_retention is None
        or request.backup_age_recipient is None
        or request.backup_confirmed is not True
    ):
        raise ValueError("prepare_backup_configure_command requires one confirmed configuration")
    config = ServerBackupConfig(
        destination=request.backup_destination,
        schedule=request.backup_schedule,
        retention=request.backup_retention,
        age_recipient=request.backup_age_recipient,
    )
    target = MachineTarget(host=identity.host, os_account="root")
    plan = ServerPlanEvent(
        command=request.command,
        timestamp=datetime.now(UTC),
        steps=_configuration_plan(target),
    )
    resolved_machine = machine or LinuxBackupConfigurationMachine()

    def execute(emitter: ServerEventEmitter, _input_stream: BinaryIO) -> None:
        _execute_configuration(emitter, resolved_machine, config)

    return PreparedServerCommand(plan=plan, execute=execute)


def _configuration_plan(target: MachineTarget) -> tuple[ServerStep, ...]:
    return (
        ServerStep(
            number=1,
            title="Confirm the backup policy",
            purpose=(
                "Record the operator's explicit destination, public recovery recipient, daily "
                "server-local schedule, and archive retention count."
            ),
            performed_by="human",
            target=target,
            phase="backup_policy_confirm",
            state="pending",
            expected_success="The command carries all four reviewed values and --confirm.",
            message="RCP will record the explicitly confirmed backup policy.",
        ),
        ServerStep(
            number=2,
            title="Prove the destination is writable",
            purpose=(
                "Require one existing safe directory in which the rcp account can create and "
                "remove an exclusive probe file."
            ),
            performed_by="system",
            target=target,
            phase="backup_destination_probe",
            state="pending",
            expected_success="The exact destination is a safe directory writable by rcp.",
            message="RCP will probe the configured destination as the rcp account.",
        ),
        ServerStep(
            number=3,
            title="Publish configuration and prove the first backup",
            purpose=(
                "Atomically replace the versioned machine config, render the timer from its "
                "same schedule, keep it disabled during one real backup, then enable it only "
                "after that archive passes readback."
            ),
            performed_by="system",
            target=target,
            phase="backup_configuration_publish",
            state="pending",
            expected_success=(
                "The root-owned config and units are exact, the first backup passed, and the "
                "timer is enabled."
            ),
            message="RCP will publish the policy, prove one backup, and activate its timer.",
        ),
        ServerStep(
            number=4,
            title="Read back the active schedule",
            purpose=(
                "Prove the stored values, rendered OnCalendar, and loaded enabled timer agree "
                "after the first protected archive."
            ),
            performed_by="system",
            target=target,
            phase="backup_configuration_readback",
            state="pending",
            expected_success=("Config and timer agree, and the timer is active and enabled."),
            message="RCP will read back the config, timer text, and systemd state.",
        ),
    )


def _execute_configuration(
    emitter: ServerEventEmitter,
    machine: BackupConfigurationMachine,
    config: ServerBackupConfig,
) -> None:
    planned = emitter.events[0]
    if not isinstance(planned, ServerPlanEvent):  # pragma: no cover - emitter owns this
        raise AssertionError("backup configuration requires its complete plan")
    steps = planned.steps
    try:
        _complete_step(
            emitter,
            steps[0],
            running="Checking the four explicitly confirmed nonsecret values.",
            succeeded="The operator explicitly confirmed the complete backup policy.",
            fields=_configuration_fields(config),
        )
        _run_step(
            emitter,
            steps[1],
            running=f"Probing {config.destination} as the rcp account.",
            operation=lambda: machine.validate_destination(config),
            succeeded="The rcp account created and removed an exclusive destination probe.",
        )
        _run_step(
            emitter,
            steps[2],
            running="Publishing the policy, running one backup, and enabling the verified timer.",
            operation=lambda: machine.persist_and_install(config),
            succeeded="The config, first protected archive, and enabled timer were published.",
        )
        readback = _run_step(
            emitter,
            steps[3],
            running="Reading back machine config, timer text, and systemd timer state.",
            operation=lambda: machine.readback(config),
            succeeded="The stored policy and active systemd schedule agree exactly.",
            fields=lambda value: (
                *_configuration_fields(value.config),
                NonsecretField(name="timer_active_state", value=value.timer_active_state),
                NonsecretField(name="timer_unit_file_state", value=value.timer_unit_file_state),
            ),
        )
        _ = readback
    except _ReportedBackupConfigurationFailure:
        return


_T = TypeVar("_T")


def _run_step(
    emitter: ServerEventEmitter,
    planned: ServerStep,
    *,
    running: str,
    operation,
    succeeded: str,
    fields=lambda _value: (),
) -> _T:
    emitter.emit_step(planned.model_copy(update={"state": "running", "message": running}))
    try:
        value = operation()
    except BackupConfigurationRefused as exc:
        emitter.emit_step(planned.model_copy(update={"state": "failed", "message": str(exc)}))
        raise _ReportedBackupConfigurationFailure from exc
    emitter.emit_step(
        planned.model_copy(
            update={
                "state": "succeeded",
                "message": succeeded,
                "fields": tuple(fields(value)),
            }
        )
    )
    return value


def _complete_step(
    emitter: ServerEventEmitter,
    planned: ServerStep,
    *,
    running: str,
    succeeded: str,
    fields: tuple[NonsecretField, ...],
) -> None:
    emitter.emit_step(planned.model_copy(update={"state": "running", "message": running}))
    emitter.emit_step(
        planned.model_copy(update={"state": "succeeded", "message": succeeded, "fields": fields})
    )


def _configuration_fields(config: ServerBackupConfig) -> tuple[NonsecretField, ...]:
    return (
        NonsecretField(name="destination", value=config.destination),
        NonsecretField(name="schedule", value=config.schedule),
        NonsecretField(name="retention", value=config.retention),
        NonsecretField(name="age_recipient", value=config.age_recipient),
    )


class LinuxBackupConfigurationMachine:
    def __init__(self, layout: ServerLayout = DEFAULT_SERVER_LAYOUT) -> None:
        self.layout = layout

    def validate_destination(self, config: ServerBackupConfig) -> None:
        destination = Path(config.destination)
        _reject_symlink_ancestry(destination)
        try:
            info = destination.stat()
        except OSError as exc:
            raise BackupConfigurationRefused(
                "The backup destination does not exist or cannot be inspected. Create the exact "
                "directory, make it writable by rcp, and rerun the same command."
            ) from exc
        if not stat.S_ISDIR(info.st_mode):
            raise BackupConfigurationRefused(
                "The backup destination is not a directory. Choose one existing local or mounted "
                "filesystem directory and rerun the same command."
            )
        try:
            account = pwd.getpwnam(self.layout.service_account)
        except KeyError as exc:
            raise BackupConfigurationRefused(
                "The rcp service account is missing. Rerun server install before backup configure."
            ) from exc
        if account.pw_uid == 0 or Path(account.pw_dir) != self.layout.service_home:
            raise BackupConfigurationRefused(
                "The rcp account identity differs from the installed server layout. Rerun "
                "server install and correct the account before backup configure."
            )
        argv = (
            "runuser",
            "--user",
            account.pw_name,
            "--",
            "env",
            "-i",
            f"HOME={account.pw_dir}",
            "PATH=/usr/bin:/bin",
            sys.executable,
            "-m",
            "rcp.server_ops.backup_config",
            "--probe-destination",
            config.destination,
        )
        try:
            completed = subprocess.run(
                argv,
                cwd=self.layout.service_home,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=SERVER_BACKUP_CONFIGURATION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackupConfigurationRefused(
                "RCP could not run the bounded destination probe as rcp. Inspect runuser and "
                "the destination permissions, then rerun the same command."
            ) from exc
        if completed.returncode != 0:
            raise BackupConfigurationRefused(
                "The rcp account could not create and remove a private file in the backup "
                "destination. Correct that directory's access and rerun the same command."
            )

    def persist_and_install(self, config: ServerBackupConfig) -> None:
        try:
            with backup_configuration_lock(self.layout):
                recover_pending_backup_configuration(self.layout)
                installed = load_installed_server_config(self.layout.config_path)
                updated = InstalledServerConfig.model_validate(
                    {**installed.model_dump(mode="python"), "backup": config}
                )
                fence_backup_timer_before_unit_change()
                _write_pending_backup_configuration(updated, self.layout)
                _converge_installed_backup_configuration(updated, self.layout)
                _clear_pending_backup_configuration(self.layout)
        except BackupConfigurationRefused:
            raise
        except InstallRefused as exc:
            raise BackupConfigurationRefused(str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise BackupConfigurationRefused(
                "RCP could not finish publishing the backup configuration. Its pending marker "
                "was retained and the timer remains untrusted; inspect the root-owned config "
                "and units, then rerun the same command to recover automatically."
            ) from exc

    def readback(self, config: ServerBackupConfig) -> BackupConfigurationReadback:
        try:
            return _readback_backup_configuration(config, self.layout)
        except BackupConfigurationRefused:
            raise
        except (InstallRefused, OSError, UnicodeError, ValueError) as exc:
            raise BackupConfigurationRefused(
                "Backup configuration readback differs from the requested inert schedule. "
                "Inspect server.toml and rcp-backup.timer, then rerun the same command."
            ) from exc


@contextmanager
def backup_configuration_lock(
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> Iterator[None]:
    """Serialize root-owned config/unit convergence on one stable inode."""

    lock_path = layout.config_path.parent / _LOCK_NAME
    descriptor = -1
    try:
        _reject_configuration_path_ancestry(lock_path.parent)
        ownership = _root_ownership()
        flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, _LOCK_MODE)
            os.fchown(descriptor, *ownership)
            os.fchmod(descriptor, _LOCK_MODE)
            os.fsync(descriptor)
            _fsync_directory(lock_path.parent)
        except FileExistsError:
            descriptor = os.open(lock_path, flags)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_uid, info.st_gid) != ownership
            or stat.S_IMODE(info.st_mode) != _LOCK_MODE
        ):
            raise BackupConfigurationRefused(
                "The server-configuration lock has unexpected ownership or mode. Inspect "
                "/etc/rcp and rerun the same command."
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupConfigurationRefused(
                "Another server-configuration operation is running. Wait for it to finish, "
                "then rerun the same command."
            ) from exc
    except BackupConfigurationRefused:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        raise
    except (KeyError, OSError) as exc:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        raise BackupConfigurationRefused(
            "RCP could not acquire the root-owned server-configuration lock. Inspect /etc/rcp "
            "and rerun the same command."
        ) from exc

    try:
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def recover_pending_backup_configuration(
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> None:
    """Complete one journaled config/timer publication while the caller holds the lock."""

    pending_path = _pending_backup_configuration_path(layout)
    if not pending_path.exists() and not pending_path.is_symlink():
        return
    try:
        pending = load_installed_server_config(pending_path)
        current = load_installed_server_config(layout.config_path)
        if pending.backup is None or _without_backup(pending) != _without_backup(current):
            raise ValueError("pending backup configuration does not match this installation")
        fence_backup_timer_before_unit_change()
        _converge_installed_backup_configuration(pending, layout)
        _clear_pending_backup_configuration(layout)
    except BackupConfigurationRefused:
        raise
    except InstallRefused as exc:
        raise BackupConfigurationRefused(str(exc)) from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise BackupConfigurationRefused(
            "RCP found an unfinished backup configuration that could not be recovered. The "
            "timer remains untrusted; inspect the pending config and units, then rerun."
        ) from exc


def activate_configured_backup_timer(
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> BackupConfigurationReadback | None:
    """Restore a configured schedule after install has safely fenced its units."""

    try:
        with backup_configuration_lock(layout):
            recover_pending_backup_configuration(layout)
            installed = load_installed_server_config(layout.config_path)
            config = installed.backup
            if config is None:
                return None
            try:
                return _readback_backup_configuration(config, layout, expected_enabled=True)
            except BackupConfigurationRefused:
                _readback_backup_configuration(config, layout, expected_enabled=False)
            try:
                run_backup_service_once()
                enable_backup_timer()
                return _readback_backup_configuration(config, layout, expected_enabled=True)
            except (InstallRefused, OSError, ValueError):
                with suppress(InstallRefused, OSError):
                    fence_backup_timer_before_unit_change()
                raise
    except BackupConfigurationRefused:
        raise
    except InstallRefused as exc:
        raise BackupConfigurationRefused(str(exc)) from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise BackupConfigurationRefused(
            "RCP could not restore the configured backup schedule after service installation. "
            "The timer remains disabled; inspect its unit and backup status, then rerun install."
        ) from exc


def _converge_installed_backup_configuration(
    installed: InstalledServerConfig,
    layout: ServerLayout,
) -> None:
    config = installed.backup
    if config is None:
        raise ValueError("backup convergence requires configured policy")
    install_backup_unit_files(
        service_content=backup_service_unit_text(),
        timer_content=render_backup_timer_unit(config.schedule),
        layout=layout,
    )
    reload_and_disable_backup_timer()
    write_installed_server_config(installed, layout.config_path)
    _readback_backup_configuration(config, layout, expected_enabled=False)
    try:
        run_backup_service_once()
        enable_backup_timer()
        _readback_backup_configuration(config, layout, expected_enabled=True)
    except (InstallRefused, OSError, ValueError):
        with suppress(InstallRefused, OSError):
            fence_backup_timer_before_unit_change()
        raise


def _readback_backup_configuration(
    config: ServerBackupConfig,
    layout: ServerLayout,
    *,
    expected_enabled: bool = True,
) -> BackupConfigurationReadback:
    installed = load_installed_server_config(layout.config_path)
    if installed.backup != config:
        raise ValueError("stored backup configuration differs")
    timer_path = layout.systemd_unit.parent / "rcp-backup.timer"
    _require_root_unit(timer_path, render_backup_timer_unit(config.schedule))
    active, enabled = read_systemd_unit_state("rcp-backup.timer")
    expected = ("active", "enabled") if expected_enabled else ("inactive", "disabled")
    if (active, enabled) != expected:
        state = "active and enabled" if expected_enabled else "inactive and disabled"
        raise BackupConfigurationRefused(
            f"The backup timer is not both {state}. Disable it, inspect rcp-backup.timer, then "
            "rerun the same command."
        )
    return BackupConfigurationReadback(
        config=installed.backup,
        timer_active_state=active,
        timer_unit_file_state=enabled,
    )


def _write_pending_backup_configuration(
    installed: InstalledServerConfig,
    layout: ServerLayout,
) -> None:
    pending_path = _pending_backup_configuration_path(layout)
    if pending_path.exists() or pending_path.is_symlink():
        raise BackupConfigurationRefused(
            "An unfinished backup configuration appeared while the operation lock was held. "
            "Stop and inspect /etc/rcp before retrying."
        )
    write_installed_server_config(installed, pending_path)


def _clear_pending_backup_configuration(layout: ServerLayout) -> None:
    pending_path = _pending_backup_configuration_path(layout)
    try:
        pending_path.unlink()
        _fsync_directory(pending_path.parent)
    except OSError as exc:
        raise BackupConfigurationRefused(
            "The backup configuration is exact, but RCP could not clear its pending "
            "marker. Inspect /etc/rcp and rerun the same command."
        ) from exc


def _pending_backup_configuration_path(layout: ServerLayout) -> Path:
    return layout.config_path.parent / _PENDING_NAME


def _without_backup(config: InstalledServerConfig) -> InstalledServerConfig:
    return config.model_copy(update={"backup": None})


def _root_ownership() -> tuple[int, int]:
    root = pwd.getpwnam("root")
    if root.pw_uid != 0:
        raise BackupConfigurationRefused("The root account must retain uid 0.")
    return root.pw_uid, root.pw_gid


def _reject_configuration_path_ancestry(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise BackupConfigurationRefused(
                "The server-configuration directory ancestry contains a symlink. Inspect "
                "/etc/rcp and rerun the same command."
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def backup_service_unit_text() -> str:
    return (
        importlib.resources.files("rcp.server_ops")
        .joinpath("assets", "rcp-backup.service")
        .read_text(encoding="utf-8")
    )


def render_backup_timer_unit(schedule: str | None) -> str:
    resolved = schedule or DEFAULT_BACKUP_SCHEDULE
    from rcp.server_ops.config import validate_backup_schedule

    local_time = validate_backup_schedule(resolved)
    template = (
        importlib.resources.files("rcp.server_ops")
        .joinpath("assets", "rcp-backup.timer")
        .read_text(encoding="utf-8")
    )
    if template.count(_TIMER_PLACEHOLDER) != 1:
        raise RuntimeError("the backup timer asset must contain exactly one schedule placeholder")
    rendered = template.replace(_TIMER_PLACEHOLDER, f"*-*-* {local_time}:00")
    if _TIMER_PLACEHOLDER in rendered:
        raise RuntimeError("the backup timer retained an unresolved schedule placeholder")
    return rendered


def _require_root_unit(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("backup timer is not one regular root-managed file")
    info = path.stat()
    if (info.st_uid, info.st_gid) != (0, 0) or stat.S_IMODE(info.st_mode) != _UNIT_MODE:
        raise ValueError("backup timer has unexpected ownership or mode")
    if path.read_text(encoding="utf-8") != expected:
        raise ValueError("backup timer text differs from the configured schedule")


def _reject_symlink_ancestry(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise BackupConfigurationRefused(
                f"The backup destination ancestry contains a symlink at {candidate}. Choose an "
                "explicit real directory and rerun the same command."
            )


def _probe_destination_as_current_user(destination: Path) -> int:
    descriptor = -1
    directory_descriptor = -1
    name = f".rcp-backup-write-probe-{uuid.uuid4().hex}"
    try:
        directory_descriptor = os.open(
            destination,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        os.write(descriptor, b"rcp backup destination probe\n")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.unlink(name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
        return 0
    except OSError:
        return 1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            with suppress(FileNotFoundError):
                os.unlink(name, dir_fd=directory_descriptor)
            os.close(directory_descriptor)


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "--probe-destination":
        return 2
    return _probe_destination_as_current_user(Path(argv[1]))


if __name__ == "__main__":  # pragma: no cover - exercised through the root coordinator
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "BackupConfigurationMachine",
    "BackupConfigurationReadback",
    "BackupConfigurationRefused",
    "LinuxBackupConfigurationMachine",
    "activate_configured_backup_timer",
    "backup_configuration_lock",
    "backup_service_unit_text",
    "prepare_backup_configure_command",
    "recover_pending_backup_configuration",
    "render_backup_timer_unit",
]
