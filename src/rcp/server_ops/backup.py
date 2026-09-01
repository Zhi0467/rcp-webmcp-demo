"""Create, verify, retain, and report protected team-server backups."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp.limits import (
    BACKUP_COPY_BUFFER_BYTES,
    BACKUP_RECEIPT_MAX_BYTES,
    SERVER_BACKUP_CONFIGURATION_TIMEOUT_SECONDS,
)
from rcp.server_ops.backup_capture import (
    BackupSQLiteCaptureReceipt,
    read_backup_sqlite_capture_receipt,
)
from rcp.server_ops.backup_models import BackupArchiveManifest, BackupFileEntry
from rcp.server_ops.backup_project_files import (
    BackupProjectFileCaptureCoordinator,
    BackupProjectFileCapturePublication,
)
from rcp.server_ops.cli import CallerIdentity, PreparedServerCommand, ServerEventEmitter
from rcp.server_ops.config import (
    InstalledServerConfig,
    ServerBackupConfig,
    load_installed_server_config,
)
from rcp.server_ops.control import ServerControlBackupCaptureResult, ServerControlClient
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.models import (
    MachineTarget,
    NonsecretField,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
    redact_server_text,
)

BACKUP_ARCHIVE_FORMAT = "rcp-team-backup-tar-age-v1"
BACKUP_ARCHIVE_RECEIPT_SCHEMA_VERSION = 1
BACKUP_OUTCOME_SCHEMA_VERSION = 1

_AGE_HEADER = b"age-encryption.org/v1\n"
_AGE_VERSION = re.compile(r"(?<![0-9])v?([0-9]+)\.([0-9]+)\.([0-9]+)(?![0-9])")
_ARCHIVE_NAME = re.compile(
    r"rcp-team-backup-v1-[0-9]{8}T[0-9]{12}Z-"
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.tar\.age"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_MODE = 0o600
_STATUS_NAME = "backup-status.json"
_LOCK_NAME = ".backup-run.lock"


class BackupRunRefused(RuntimeError):
    """One secret-free backup failure that may be shown to an operator."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


def _canonical_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase canonical UUID4")
    return value


def _aware_time(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} requires a timezone")
    return value


def _safe_diagnostic(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 240
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or redact_server_text(value) != value
    ):
        raise ValueError("backup diagnostics must be one bounded secret-free line")
    return value


class BackupArchiveReceipt(_StrictModel):
    """Immutable proof that one ciphertext was published and read back."""

    schema_version: Literal[BACKUP_ARCHIVE_RECEIPT_SCHEMA_VERSION] = (
        BACKUP_ARCHIVE_RECEIPT_SCHEMA_VERSION
    )
    archive_format: Literal[BACKUP_ARCHIVE_FORMAT] = BACKUP_ARCHIVE_FORMAT
    installation_id: str
    space_id: str
    capture_id: str
    destination: str
    archive_name: str
    captured_at: datetime
    protected_at: datetime
    capture_status: Literal["complete", "partial"]
    age_version: str
    age_recipient_fingerprint: str
    archive_sha256: str
    archive_size_bytes: int = Field(gt=0)
    manifest_sha256: str
    captured_bytes: int = Field(ge=0)
    project_count: int = Field(ge=0)
    protected_project_count: int = Field(ge=0)
    uncaptured_project_count: int = Field(ge=0)
    readback: Literal["passed"] = "passed"

    @field_validator("installation_id", "space_id", "capture_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("captured_at", "protected_at")
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _aware_time(value, label=info.field_name.replace("_", " "))

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path == Path("/") or ".." in path.parts or str(path) != value:
            raise ValueError("backup receipt destination must be absolute and normalized")
        return value

    @field_validator("archive_name")
    @classmethod
    def validate_archive_name(cls, value: str) -> str:
        if _ARCHIVE_NAME.fullmatch(value) is None:
            raise ValueError("backup archive name is invalid")
        return value

    @field_validator("age_version")
    @classmethod
    def validate_age_version(cls, value: str) -> str:
        matched = _AGE_VERSION.fullmatch(value)
        if matched is None or int(matched.group(1)) != 1:
            raise ValueError("backup receipt requires one age 1.x version")
        return value

    @field_validator("age_recipient_fingerprint", "archive_sha256", "manifest_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("backup receipt digests must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> BackupArchiveReceipt:
        if not self.archive_name.endswith(f"-{self.capture_id}.tar.age"):
            raise ValueError("backup archive name is not bound to its capture")
        if self.protected_at < self.captured_at:
            raise ValueError("backup protection cannot predate capture")
        if self.protected_project_count + self.uncaptured_project_count != self.project_count:
            raise ValueError("backup project counts do not match")
        if self.capture_status == "complete" and self.uncaptured_project_count:
            raise ValueError("a complete backup cannot contain an uncaptured project")
        return self


class BackupRunOutcome(_StrictModel):
    """Mutable last-run status stored outside the backed-up data directory."""

    schema_version: Literal[BACKUP_OUTCOME_SCHEMA_VERSION] = BACKUP_OUTCOME_SCHEMA_VERSION
    operation_id: str
    installation_id: str
    destination: str
    started_at: datetime
    completed_at: datetime
    status: Literal["protected", "partial", "failure"]
    archive: BackupArchiveReceipt | None = None
    archive_receipt_sha256: str | None = None
    failure: str | None = None
    retention_deleted_archives: tuple[str, ...] = ()

    @field_validator("operation_id", "installation_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _aware_time(value, label=info.field_name.replace("_", " "))

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        return BackupArchiveReceipt.validate_destination(value)

    @field_validator("archive_receipt_sha256")
    @classmethod
    def validate_receipt_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("archive receipt digest must be lowercase SHA-256")
        return value

    @field_validator("failure")
    @classmethod
    def validate_failure(cls, value: str | None) -> str | None:
        return None if value is None else _safe_diagnostic(value)

    @field_validator("retention_deleted_archives")
    @classmethod
    def validate_deleted_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value or any(
            _ARCHIVE_NAME.fullmatch(name) is None for name in value
        ):
            raise ValueError("deleted backup archive names must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> BackupRunOutcome:
        if self.completed_at < self.started_at:
            raise ValueError("backup outcome cannot complete before it starts")
        if self.status in {"protected", "partial"}:
            if (
                self.archive is None
                or self.archive_receipt_sha256 is None
                or self.failure is not None
            ):
                raise ValueError("successful backup outcomes require exact archive proof")
            expected = "protected" if self.archive.capture_status == "complete" else "partial"
            if self.status != expected:
                raise ValueError("backup outcome status differs from its archive")
        elif self.failure is None:
            raise ValueError("failed backup outcomes require a diagnostic")
        if (self.archive is None) != (self.archive_receipt_sha256 is None):
            raise ValueError("backup archive and receipt digest must be present together")
        if self.archive is not None and (
            self.archive.installation_id != self.installation_id
            or self.archive.destination != self.destination
        ):
            raise ValueError("backup outcome and archive identity differ")
        return self


class BackupRetentionPlan(_StrictModel):
    destination: str
    kept_archives: tuple[str, ...]
    delete_archives: tuple[str, ...]

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        return BackupArchiveReceipt.validate_destination(value)

    @field_validator("kept_archives", "delete_archives")
    @classmethod
    def validate_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value or any(
            _ARCHIVE_NAME.fullmatch(name) is None for name in value
        ):
            raise ValueError("retention archive names must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> BackupRetentionPlan:
        if set(self.kept_archives).intersection(self.delete_archives):
            raise ValueError("retention cannot keep and delete the same archive")
        return self


@dataclass(frozen=True)
class ProtectedBackupArchive:
    receipt: BackupArchiveReceipt
    archive_path: Path
    receipt_path: Path
    receipt_sha256: str


class BackupCaptureControl(Protocol):
    def capture_backup_sqlite(self) -> ServerControlBackupCaptureResult: ...


class BackupRunMachine(Protocol):
    def run(self) -> BackupRunOutcome: ...


def prepare_backup_run_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    machine: BackupRunMachine | None = None,
) -> PreparedServerCommand:
    if request.command != "server backup run":
        raise ValueError("prepare_backup_run_command requires one backup run request")
    target = MachineTarget(host=identity.host, os_account="rcp")
    pending = ServerStep(
        number=1,
        title="Capture and protect the team space",
        purpose=(
            "Capture the live database and typed project history without pausing work, stream one "
            "deterministic archive through age 1.x, read it back, and apply proven retention."
        ),
        performed_by="system",
        target=target,
        phase="backup_run",
        state="pending",
        expected_success="One atomic age archive has a durable readback receipt and status.",
        message="RCP will run the configured protected-backup workflow.",
    )
    plan = ServerPlanEvent(command=request.command, timestamp=datetime.now(UTC), steps=(pending,))
    resolved_machine = machine or LinuxBackupRunMachine()

    def execute(emitter: ServerEventEmitter, _input_stream: BinaryIO) -> None:
        emitter.emit_step(
            pending.model_copy(
                update={
                    "state": "running",
                    "message": "Capturing typed state, encrypting it, and proving the published bytes.",
                }
            )
        )
        try:
            outcome = resolved_machine.run()
        except BackupRunRefused as exc:
            emitter.emit_step(pending.model_copy(update={"state": "failed", "message": str(exc)}))
            return
        assert outcome.archive is not None
        emitter.emit_step(
            pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": (
                        "The complete team space is protected and read back."
                        if outcome.status == "protected"
                        else "A partial archive is protected; its manifest names every omission."
                    ),
                    "fields": _outcome_fields(outcome),
                }
            )
        )

    return PreparedServerCommand(plan=plan, execute=execute)


def _outcome_fields(outcome: BackupRunOutcome) -> tuple[NonsecretField, ...]:
    archive = outcome.archive
    assert archive is not None
    return (
        NonsecretField(name="backup_status", value=outcome.status),
        NonsecretField(
            name="archive_path", value=str(Path(outcome.destination) / archive.archive_name)
        ),
        NonsecretField(name="archive_sha256", value=archive.archive_sha256),
        NonsecretField(name="captured_bytes", value=archive.captured_bytes),
        NonsecretField(name="protected_projects", value=archive.protected_project_count),
        NonsecretField(name="uncaptured_projects", value=archive.uncaptured_project_count),
        NonsecretField(name="age_version", value=archive.age_version),
        NonsecretField(name="retention_deleted", value=len(outcome.retention_deleted_archives)),
    )


class LinuxBackupRunMachine:
    """Concrete service-account composition of capture, encryption, and retention."""

    def __init__(
        self,
        layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
        *,
        control: BackupCaptureControl | None = None,
        age_executable: str = "age",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.layout = layout
        self.control = control
        self.age_executable = age_executable
        self.clock = clock or (lambda: datetime.now(UTC))

    def run(self) -> BackupRunOutcome:
        from rcp.server_ops.restore import RestoreRefused, unfinished_restore_operation

        try:
            restore = unfinished_restore_operation(
                self.layout,
                expected_uid=os.geteuid(),
            )
        except (OSError, RestoreRefused) as exc:
            raise BackupRunRefused(
                "Restore machine state is unsafe. Keep the service stopped and resume server "
                "restore before another protected backup."
            ) from exc
        if restore is not None:
            raise BackupRunRefused(
                "An unfinished replacement restore blocks protected backup until its final "
                "service readback completes."
            )
        started_at = self.clock()
        operation_id = str(uuid.uuid4())
        config = _load_backup_configuration(self.layout)
        protected: ProtectedBackupArchive | None = None
        with backup_run_lock(self.layout):
            try:
                age_version = require_age_1x(self.age_executable)
                control = self.control or ServerControlClient.from_data_dir(
                    self.layout.data_dir,
                    expected_server_uid=os.geteuid(),
                )
                sqlite_result = control.capture_backup_sqlite()
                project_publication = BackupProjectFileCaptureCoordinator(
                    self.layout.data_dir
                ).capture(
                    Path(sqlite_result.receipt_path),
                    expected_sha256=sqlite_result.receipt_sha256,
                )
                sqlite_receipt = read_backup_sqlite_capture_receipt(
                    Path(sqlite_result.receipt_path),
                    expected_sha256=sqlite_result.receipt_sha256,
                )
                manifest = build_archive_manifest(
                    installed=config,
                    sqlite_receipt=sqlite_receipt,
                    project_publication=project_publication,
                )
                protected = protect_backup_archive(
                    installed=config,
                    manifest=manifest,
                    capture_root=Path(sqlite_result.receipt_path).parent,
                    age_version=age_version,
                    age_executable=self.age_executable,
                    protected_at=self.clock(),
                )
                retention = plan_backup_retention(
                    config.backup,
                    installation_id=config.installation_id,
                    expected_uid=os.geteuid(),
                )
                deleted = apply_backup_retention(
                    retention,
                    installation_id=config.installation_id,
                    expected_uid=os.geteuid(),
                )
                discard_backup_capture_root(
                    Path(sqlite_result.receipt_path).parent,
                    data_dir=self.layout.data_dir,
                    capture_id=sqlite_result.capture_id,
                )
                completed_at = self.clock()
                outcome = BackupRunOutcome(
                    operation_id=operation_id,
                    installation_id=config.installation_id,
                    destination=config.backup.destination,
                    started_at=started_at,
                    completed_at=completed_at,
                    status=(
                        "protected" if protected.receipt.capture_status == "complete" else "partial"
                    ),
                    archive=protected.receipt,
                    archive_receipt_sha256=protected.receipt_sha256,
                    retention_deleted_archives=deleted,
                )
                write_backup_outcome(outcome, self.layout)
                return outcome
            except BackupRunRefused as exc:
                self._record_failure(
                    operation_id,
                    started_at,
                    config,
                    str(exc),
                    protected=protected,
                )
                raise
            except Exception as exc:
                message = (
                    "The protected backup failed unexpectedly. Inspect the service log and the "
                    "retained private capture stage, then rerun the same command."
                )
                self._record_failure(
                    operation_id,
                    started_at,
                    config,
                    message,
                    protected=protected,
                )
                raise BackupRunRefused(message) from exc

    def _record_failure(
        self,
        operation_id: str,
        started_at: datetime,
        config: InstalledServerConfig,
        message: str,
        *,
        protected: ProtectedBackupArchive | None,
    ) -> None:
        assert config.backup is not None
        outcome = BackupRunOutcome(
            operation_id=operation_id,
            installation_id=config.installation_id,
            destination=config.backup.destination,
            started_at=started_at,
            completed_at=self.clock(),
            status="failure",
            archive=protected.receipt if protected is not None else None,
            archive_receipt_sha256=(protected.receipt_sha256 if protected is not None else None),
            failure=message,
        )
        try:
            write_backup_outcome(outcome, self.layout)
        except (OSError, ValueError) as exc:
            raise BackupRunRefused(
                "The backup failed and RCP could not publish its durable status. Inspect the "
                "server root and service log before retrying."
            ) from exc


def _load_backup_configuration(layout: ServerLayout) -> InstalledServerConfig:
    try:
        installed = load_installed_server_config(layout.config_path)
    except (OSError, ValueError) as exc:
        raise BackupRunRefused(
            "The installed server configuration is missing or invalid. Run backup configure as "
            "root before retrying."
        ) from exc
    if installed.backup is None:
        raise BackupRunRefused(
            "Protected backup is not configured. Run backup configure as root before retrying."
        )
    if installed.paths.model_dump() != layout.recorded_paths():
        raise BackupRunRefused("The installed backup configuration names another server layout.")
    _validate_destination_boundary(Path(installed.backup.destination))
    return installed


def require_age_1x(executable: str = "age") -> str:
    try:
        completed = subprocess.run(
            (executable, "--version"),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=SERVER_BACKUP_CONFIGURATION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BackupRunRefused(
            "The upstream age CLI is unavailable. Install age >=1.0.0,<2.0.0 and retry."
        ) from exc
    combined = f"{completed.stdout}\n{completed.stderr}"
    matched = _AGE_VERSION.search(combined)
    if completed.returncode != 0 or matched is None or int(matched.group(1)) != 1:
        raise BackupRunRefused(
            "The upstream age CLI must be >=1.0.0,<2.0.0. Install age 1.x and retry."
        )
    return ".".join(matched.groups())


def build_archive_manifest(
    *,
    installed: InstalledServerConfig,
    sqlite_receipt: BackupSQLiteCaptureReceipt,
    project_publication: BackupProjectFileCapturePublication,
) -> BackupArchiveManifest:
    if installed.backup is None:
        raise ValueError("archive manifest construction requires backup configuration")
    project_receipt = project_publication.receipt
    if (
        project_receipt.capture_id != sqlite_receipt.capture_id
        or project_receipt.space_id != sqlite_receipt.space_id
        or project_receipt.sqlite_snapshot_sha256 != sqlite_receipt.sqlite_snapshot.sha256
    ):
        raise BackupRunRefused("The project-file capture differs from its SQLite boundary.")
    source_label = None
    source_fingerprint = None
    if installed.source.authentication == "deploy_key":
        source_label = f"rcp-source:{installed.installation_id}"
        source_fingerprint = installed.source.public_key_fingerprint
    uncaptured_app_data = tuple(
        sorted(
            {
                *sqlite_receipt.app_data_plan.deferred_entries,
                *sqlite_receipt.app_data_plan.unclassified_entries,
            }
        )
    )
    manifest = BackupArchiveManifest(
        space_id=sqlite_receipt.space_id,
        space_name=sqlite_receipt.space_name,
        rcp_source_commit=sqlite_receipt.rcp_source_commit,
        database_schema_sha256=sqlite_receipt.database_schema_sha256,
        captured_at=sqlite_receipt.captured_at,
        sqlite_snapshot=sqlite_receipt.sqlite_snapshot,
        encryption_recipient_fingerprint=_recipient_fingerprint(installed.backup.age_recipient),
        installation_id=installed.installation_id,
        source_deploy_key_label=source_label,
        source_public_key_fingerprint=source_fingerprint,
        excluded_app_data_entries=sqlite_receipt.app_data_plan.excluded_entries,
        captured_app_data_entries=sqlite_receipt.app_data_plan.captured_entries,
        uncaptured_app_data_entries=uncaptured_app_data,
        projects=project_receipt.projects,
        imported_sources=project_receipt.imported_sources,
        status=("partial" if project_receipt.status == "partial" else "complete"),
        total_bytes=(
            sqlite_receipt.sqlite_snapshot.size_bytes
            + sum(project.total_bytes for project in project_receipt.projects)
            + sum(capture.total_bytes for capture in project_receipt.imported_sources)
        ),
    )
    if (manifest.status == "partial") != (
        sqlite_receipt.status == "partial" or project_receipt.status == "partial"
    ):
        raise BackupRunRefused("The final backup manifest does not explain its capture status.")
    return manifest


def protect_backup_archive(
    *,
    installed: InstalledServerConfig,
    manifest: BackupArchiveManifest,
    capture_root: Path,
    age_version: str,
    age_executable: str = "age",
    protected_at: datetime | None = None,
) -> ProtectedBackupArchive:
    config = installed.backup
    if config is None:
        raise ValueError("archive protection requires backup configuration")
    if (
        manifest.installation_id != installed.installation_id
        or manifest.encryption_recipient_fingerprint != _recipient_fingerprint(config.age_recipient)
    ):
        raise BackupRunRefused(
            "The backup manifest does not match this installation and configured recipient."
        )
    protected_time = protected_at or datetime.now(UTC)
    _aware_time(protected_time, label="backup protection time")
    try:
        capture_info = capture_root.lstat()
    except OSError as exc:
        raise BackupRunRefused("The private backup capture root is unavailable.") from exc
    if (
        not stat.S_ISDIR(capture_info.st_mode)
        or capture_info.st_uid != os.geteuid()
        or stat.S_IMODE(capture_info.st_mode) & 0o077
    ):
        raise BackupRunRefused("The private backup capture root has unsafe ownership or mode.")
    destination = Path(config.destination)
    _validate_destination_boundary(destination)
    archive_name = _archive_name(manifest.captured_at, manifest.sqlite_snapshot, capture_root)
    archive_path = destination / archive_name
    temporary_name = f".{archive_name}.{uuid.uuid4().hex}.partial"
    temporary_path = destination / temporary_name
    descriptor = -1
    process: subprocess.Popen[bytes] | None = None
    stderr_file = tempfile.TemporaryFile()  # noqa: SIM115 - closed by the shared failure fence
    try:
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            _ARCHIVE_MODE,
        )
        process = subprocess.Popen(
            (
                age_executable,
                "--encrypt",
                "--recipient",
                config.age_recipient,
            ),
            stdin=subprocess.PIPE,
            stdout=descriptor,
            stderr=stderr_file,
        )
        if process.stdin is None:  # pragma: no cover - subprocess contract
            raise OSError("age stdin was not created")
        try:
            _write_deterministic_archive(process.stdin, manifest, capture_root)
        finally:
            process.stdin.close()
        returncode = process.wait()
        if returncode != 0:
            raise BackupRunRefused(
                "age 1.x could not encrypt the captured archive. Inspect the service log and "
                "destination capacity, then retry."
            )
        os.fchmod(descriptor, _ARCHIVE_MODE)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _require_age_header(temporary_path)
        try:
            os.link(temporary_path, archive_path, follow_symlinks=False)
        except FileExistsError as exc:
            raise BackupRunRefused(
                "The generated backup archive name already exists; no existing archive was replaced."
            ) from exc
        except OSError as exc:
            raise BackupRunRefused(
                "The backup filesystem could not atomically publish the protected archive."
            ) from exc
        temporary_path.unlink()
        _fsync_directory(destination)
        archive_sha256, archive_size = _readback_ciphertext(
            archive_path,
            expected_uid=os.geteuid(),
        )
        manifest_bytes = _manifest_bytes(manifest)
        captured_projects = sum(project.status == "captured" for project in manifest.projects)
        receipt = BackupArchiveReceipt(
            installation_id=installed.installation_id,
            space_id=manifest.space_id,
            capture_id=capture_root.name.removeprefix("backup-"),
            destination=str(destination),
            archive_name=archive_name,
            captured_at=manifest.captured_at,
            protected_at=protected_time,
            capture_status=manifest.status,
            age_version=age_version,
            age_recipient_fingerprint=manifest.encryption_recipient_fingerprint,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            captured_bytes=manifest.total_bytes,
            project_count=len(manifest.projects),
            protected_project_count=captured_projects,
            uncaptured_project_count=len(manifest.projects) - captured_projects,
        )
        receipt_path = destination / f"{archive_name}.receipt.json"
        receipt_sha256 = _publish_new_json(receipt_path, receipt)
        return ProtectedBackupArchive(
            receipt=receipt,
            archive_path=archive_path,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
        )
    except BackupRunRefused:
        raise
    except (BrokenPipeError, OSError, tarfile.TarError, ValueError) as exc:
        raise BackupRunRefused(
            "RCP could not stream and verify the protected archive. The private capture stage "
            "was retained for diagnosis and retry."
        ) from exc
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            if process.poll() is None:
                process.kill()
                process.wait()
        if descriptor >= 0:
            os.close(descriptor)
        stderr_file.close()
        temporary_path.unlink(missing_ok=True)


def _archive_name(captured_at: datetime, sqlite: BackupFileEntry, capture_root: Path) -> str:
    capture_id = capture_root.name.removeprefix("backup-")
    _canonical_uuid4(capture_id, label="backup capture identity")
    if sqlite.group != "sqlite_snapshot":
        raise ValueError("archive naming requires the SQLite capture entry")
    stamp = captured_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"rcp-team-backup-v1-{stamp}-{capture_id}.tar.age"


def _manifest_bytes(manifest: BackupArchiveManifest) -> bytes:
    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_deterministic_archive(
    stream: BinaryIO,
    manifest: BackupArchiveManifest,
    capture_root: Path,
) -> None:
    manifest_bytes = _manifest_bytes(manifest)
    with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        _add_bytes_to_tar(archive, "manifest.json", manifest_bytes)
        entries = (
            manifest.sqlite_snapshot,
            *(entry for project in manifest.projects for entry in project.files),
            *(entry for capture in manifest.imported_sources for entry in capture.files),
        )
        for entry in sorted(entries, key=lambda item: item.archive_path):
            source = (
                capture_root / "rcp.sqlite3"
                if entry.group == "sqlite_snapshot"
                else capture_root.joinpath(*Path(entry.archive_path).parts)
            )
            _add_verified_file_to_tar(archive, source, entry)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o400
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _add_bytes_to_tar(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))


class _DigestingReader:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        amount = BACKUP_COPY_BUFFER_BYTES if size < 0 else min(size, BACKUP_COPY_BUFFER_BYTES)
        data = os.read(self.descriptor, amount)
        self.digest.update(data)
        self.size += len(data)
        return data


def _add_verified_file_to_tar(
    archive: tarfile.TarFile,
    source: Path,
    entry: BackupFileEntry,
) -> None:
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        path_before = source.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (path_before.st_dev, path_before.st_ino)
            or before.st_size != entry.size_bytes
        ):
            raise BackupRunRefused("A captured backup input changed before archive streaming.")
        reader = _DigestingReader(descriptor)
        archive.addfile(_tar_info(entry.archive_path, entry.size_bytes), reader)
        after = os.fstat(descriptor)
        path_after = source.lstat()
        compared = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            reader.size != entry.size_bytes
            or reader.digest.hexdigest() != entry.sha256
            or any(getattr(before, name) != getattr(after, name) for name in compared)
            or any(getattr(after, name) != getattr(path_after, name) for name in compared)
        ):
            raise BackupRunRefused("A captured backup input changed during archive streaming.")
    finally:
        os.close(descriptor)


def _require_age_header(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if os.read(descriptor, len(_AGE_HEADER)) != _AGE_HEADER:
            raise BackupRunRefused("age returned output that is not the required v1 file format.")
    finally:
        os.close(descriptor)


def _readback_ciphertext(path: Path, *, expected_uid: int) -> tuple[str, int]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != expected_uid
            or stat.S_IMODE(before.st_mode) != _ARCHIVE_MODE
        ):
            raise BackupRunRefused("The published backup archive has unsafe ownership or mode.")
        prefix = os.read(descriptor, len(_AGE_HEADER))
        digest.update(prefix)
        size += len(prefix)
        if prefix != _AGE_HEADER:
            raise BackupRunRefused("The published backup archive is not an age v1 file.")
        while True:
            chunk = os.read(descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != after.st_size:
            raise BackupRunRefused("The protected archive changed during integrity readback.")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def plan_backup_retention(
    config: ServerBackupConfig,
    *,
    installation_id: str,
    expected_uid: int,
) -> BackupRetentionPlan:
    destination = Path(config.destination)
    _validate_destination_boundary(destination)
    receipts = _proven_archive_receipts(
        destination,
        installation_id=installation_id,
        expected_uid=expected_uid,
    )
    newest = sorted(
        receipts.values(),
        key=lambda receipt: (receipt.protected_at, receipt.archive_name),
        reverse=True,
    )
    keep = {receipt.archive_name for receipt in newest[: config.retention]}
    newest_complete = next(
        (receipt for receipt in newest if receipt.capture_status == "complete"),
        None,
    )
    if newest_complete is not None:
        keep.add(newest_complete.archive_name)
    delete = set(receipts).difference(keep)
    return BackupRetentionPlan(
        destination=str(destination),
        kept_archives=tuple(sorted(keep)),
        delete_archives=tuple(sorted(delete)),
    )


def latest_protected_backup_receipt(
    destination: Path,
    *,
    installation_id: str,
    expected_uid: int,
) -> BackupArchiveReceipt | None:
    """Read the newest retained archive proof independently of the last run outcome."""

    _validate_destination_boundary(destination)
    try:
        candidates = sorted(destination.glob("*.tar.age.receipt.json"), key=lambda path: path.name)
    except OSError as exc:
        raise BackupRunRefused(
            "The backup destination could not be inspected for server status."
        ) from exc
    receipts = [
        read_backup_archive_receipt(
            path,
            expected_destination=destination,
            expected_installation_id=installation_id,
            expected_uid=expected_uid,
            verify_digest=False,
        )
        for path in candidates
    ]
    return max(
        receipts,
        key=lambda receipt: (receipt.protected_at, receipt.archive_name),
        default=None,
    )


def apply_backup_retention(
    plan: BackupRetentionPlan,
    *,
    installation_id: str,
    expected_uid: int,
) -> tuple[str, ...]:
    """Apply only the exact deletion targets already exposed by ``plan``."""

    destination = Path(plan.destination)
    deleted: list[str] = []
    for archive_name in plan.delete_archives:
        receipt_path = destination / f"{archive_name}.receipt.json"
        receipt = read_backup_archive_receipt(
            receipt_path,
            expected_destination=destination,
            expected_installation_id=installation_id,
            expected_uid=expected_uid,
            verify_digest=True,
        )
        if receipt.archive_name != archive_name:
            raise BackupRunRefused("A retention target changed after its deletion preview.")
        archive_path = destination / archive_name
        try:
            archive_path.unlink()
            _fsync_directory(destination)
            receipt_path.unlink()
            _fsync_directory(destination)
        except OSError as exc:
            raise BackupRunRefused(
                f"Retention could not delete the proven archive {archive_name}. Inspect the "
                "destination and rerun backup."
            ) from exc
        deleted.append(archive_name)
    return tuple(sorted(deleted))


def _proven_archive_receipts(
    destination: Path,
    *,
    installation_id: str,
    expected_uid: int,
) -> dict[str, BackupArchiveReceipt]:
    receipts: dict[str, BackupArchiveReceipt] = {}
    try:
        candidates = sorted(destination.glob("*.tar.age.receipt.json"), key=lambda path: path.name)
    except OSError as exc:
        raise BackupRunRefused(
            "The backup destination could not be inspected for retention."
        ) from exc
    for path in candidates:
        try:
            receipt = read_backup_archive_receipt(
                path,
                expected_destination=destination,
                expected_installation_id=installation_id,
                expected_uid=expected_uid,
                verify_digest=False,
            )
        except (BackupRunRefused, OSError, ValueError):
            continue
        receipts[receipt.archive_name] = receipt
    return receipts


def read_backup_archive_receipt(
    path: Path,
    *,
    expected_destination: Path,
    expected_installation_id: str,
    expected_uid: int,
    verify_digest: bool,
    expected_receipt_sha256: str | None = None,
) -> BackupArchiveReceipt:
    expected_name = path.name.removesuffix(".receipt.json")
    if (
        path.parent != expected_destination
        or _ARCHIVE_NAME.fullmatch(expected_name) is None
        or path.name != f"{expected_name}.receipt.json"
    ):
        raise BackupRunRefused("The backup archive receipt path is invalid.")
    payload = _read_private_file(path, expected_uid=expected_uid, maximum=BACKUP_RECEIPT_MAX_BYTES)
    if expected_receipt_sha256 is not None and (
        _SHA256.fullmatch(expected_receipt_sha256) is None
        or hashlib.sha256(payload).hexdigest() != expected_receipt_sha256
    ):
        raise BackupRunRefused("A backup archive receipt no longer matches its published digest.")
    try:
        receipt = BackupArchiveReceipt.model_validate_json(payload)
    except ValueError as exc:
        raise BackupRunRefused("A backup archive receipt is invalid.") from exc
    if (
        receipt.archive_name != expected_name
        or receipt.destination != str(expected_destination)
        or receipt.installation_id != expected_installation_id
    ):
        raise BackupRunRefused("A backup archive receipt belongs to another destination or server.")
    archive_path = expected_destination / receipt.archive_name
    info = archive_path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != expected_uid
        or stat.S_IMODE(info.st_mode) != _ARCHIVE_MODE
        or info.st_size != receipt.archive_size_bytes
    ):
        raise BackupRunRefused("A protected archive no longer matches its receipt metadata.")
    if verify_digest:
        digest, size = _readback_ciphertext(archive_path, expected_uid=expected_uid)
        if digest != receipt.archive_sha256 or size != receipt.archive_size_bytes:
            raise BackupRunRefused("A protected archive no longer matches its readback digest.")
    return receipt


def backup_status_path(layout: ServerLayout = DEFAULT_SERVER_LAYOUT) -> Path:
    return layout.server_root / _STATUS_NAME


def write_backup_outcome(
    outcome: BackupRunOutcome,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> None:
    path = backup_status_path(layout)
    payload = _model_bytes(outcome)
    _atomic_replace_private(path, payload)


def read_backup_outcome(
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
    *,
    expected_uid: int | None = None,
) -> BackupRunOutcome:
    payload = _read_private_file(
        backup_status_path(layout),
        expected_uid=os.geteuid() if expected_uid is None else expected_uid,
        maximum=BACKUP_RECEIPT_MAX_BYTES,
    )
    try:
        return BackupRunOutcome.model_validate_json(payload)
    except ValueError as exc:
        raise ValueError("backup status is invalid") from exc


@contextmanager
def backup_run_coordination_lock(
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    timeout: float = 0.0,
) -> Iterator[None]:
    """Fence a service-account backup against the root update coordinator."""

    if timeout < 0:
        raise ValueError("backup coordination timeout must be nonnegative")
    owner_uid = os.geteuid() if expected_uid is None else expected_uid
    owner_gid = os.getegid() if expected_gid is None else expected_gid
    path = layout.server_root / _LOCK_NAME
    descriptor = -1
    try:
        flags = os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, _ARCHIVE_MODE)
            if (owner_uid, owner_gid) != (os.geteuid(), os.getegid()):
                os.fchown(descriptor, owner_uid, owner_gid)
            os.fchmod(descriptor, _ARCHIVE_MODE)
            os.fsync(descriptor)
            _fsync_directory(path.parent)
        except FileExistsError:
            descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_uid, metadata.st_gid) != (owner_uid, owner_gid)
            or stat.S_IMODE(metadata.st_mode) != _ARCHIVE_MODE
        ):
            raise BackupRunRefused("The backup operation lock has unsafe ownership or mode.")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise BackupRunRefused(
                        "Another backup run is already active. Wait for it to finish before retrying."
                    ) from exc
                time.sleep(min(0.1, deadline - time.monotonic()))
        yield
    except BackupRunRefused:
        raise
    except OSError as exc:
        raise BackupRunRefused("RCP could not acquire the private backup operation lock.") from exc
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


@contextmanager
def backup_run_lock(layout: ServerLayout = DEFAULT_SERVER_LAYOUT) -> Iterator[None]:
    with backup_run_coordination_lock(layout):
        yield


def _publish_new_json(path: Path, model: BaseModel) -> str:
    payload = _model_bytes(model)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            _ARCHIVE_MODE,
        )
        _write_all(descriptor, payload)
        os.fchmod(descriptor, _ARCHIVE_MODE)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise BackupRunRefused("The backup archive receipt already exists.") from exc
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _atomic_replace_private(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != _ARCHIVE_MODE
        ):
            raise ValueError("existing backup status has unsafe ownership or mode")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, _ARCHIVE_MODE)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _model_bytes(model: BaseModel) -> bytes:
    payload = (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > BACKUP_RECEIPT_MAX_BYTES:
        raise ValueError("backup status exceeds its size bound")
    return payload


def _read_private_file(path: Path, *, expected_uid: int, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != expected_uid
            or stat.S_IMODE(before.st_mode) != _ARCHIVE_MODE
            or before.st_size > maximum
        ):
            raise ValueError("private backup record has unsafe ownership, mode, or size")
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(BACKUP_COPY_BUFFER_BYTES, remaining))
            if not chunk:
                raise ValueError("private backup record is incomplete")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("private backup record changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _validate_destination_boundary(destination: Path) -> None:
    for candidate in (destination, *destination.parents):
        if candidate.is_symlink():
            raise BackupRunRefused(
                "The configured backup destination ancestry now contains a symlink. Disable the "
                "timer, repair the destination, and rerun backup configure."
            )
    try:
        info = destination.lstat()
        descriptor = os.open(
            destination,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise BackupRunRefused(
            "The configured backup destination is missing or unavailable. Restore its access and retry."
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise BackupRunRefused("The configured backup destination boundary is unsafe.")
    finally:
        os.close(descriptor)


def discard_backup_capture_root(
    capture_root: Path,
    *,
    data_dir: Path,
    capture_id: str,
) -> None:
    expected = data_dir.resolve() / "run-stage" / f"backup-{capture_id}"
    try:
        info = capture_root.lstat()
    except OSError as exc:
        raise BackupRunRefused(
            "The protected archive exists, but its private stage is unavailable."
        ) from exc
    if (
        capture_root != expected
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise BackupRunRefused("The protected archive exists, but its private stage is unsafe.")
    try:
        shutil.rmtree(capture_root)
        _fsync_directory(expected.parent)
    except OSError as exc:
        raise BackupRunRefused(
            "The archive is protected, but RCP could not remove its private plaintext stage. "
            "Inspect the exact run-stage directory before retrying."
        ) from exc


def _recipient_fingerprint(recipient: str) -> str:
    return hashlib.sha256(recipient.encode("ascii")).hexdigest()


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short backup write")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BACKUP_ARCHIVE_FORMAT",
    "BACKUP_ARCHIVE_RECEIPT_SCHEMA_VERSION",
    "BACKUP_OUTCOME_SCHEMA_VERSION",
    "BackupArchiveReceipt",
    "BackupRetentionPlan",
    "BackupRunMachine",
    "BackupRunOutcome",
    "BackupRunRefused",
    "LinuxBackupRunMachine",
    "ProtectedBackupArchive",
    "apply_backup_retention",
    "backup_run_coordination_lock",
    "backup_run_lock",
    "backup_status_path",
    "build_archive_manifest",
    "discard_backup_capture_root",
    "latest_protected_backup_receipt",
    "plan_backup_retention",
    "prepare_backup_run_command",
    "protect_backup_archive",
    "read_backup_archive_receipt",
    "read_backup_outcome",
    "require_age_1x",
    "write_backup_outcome",
]
