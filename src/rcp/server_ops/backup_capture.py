"""Online SQLite snapshot and immutable typed inventory for server backups."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp.artifacts import AgentArtifactDescriptor, ArtifactMediaType
from rcp.limits import (
    BACKUP_COPY_BUFFER_BYTES,
    BACKUP_DIAGNOSTIC_MAX_CHARS,
    BACKUP_INVENTORY_MAX_ENTRIES,
    BACKUP_RECEIPT_MAX_BYTES,
)
from rcp.projects import BackupProjectUnavailable, inspect_backup_project_registration
from rcp.server_ops.backup_models import (
    BackupAppDataCapturePlan,
    BackupCheckoutRecoveryDescriptor,
    BackupFileEntry,
    BackupImportedProviderSourceInventory,
    inspect_app_data_capture_plan,
)
from rcp.server_ops.models import redact_server_text
from rcp.server_runtime import ServerMetadata, data_dir_identity
from rcp.sources.imported import ImportedProviderSourceStore
from rcp.storage import AppStore, ProjectRecord, ResultViewRecord

BACKUP_SQLITE_CAPTURE_SCHEMA_VERSION = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_ARTIFACT_ID = re.compile(r"[0-9a-f]{24}")


class BackupCaptureUnavailable(RuntimeError):
    """The service cannot publish an honest immutable database capture."""


class BackupProjectInventoryUnavailable(ValueError):
    """One captured catalog project has no valid typed file inventory."""


def _canonical_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase, hyphenated canonical UUID4")
    return value


def _safe_line(value: str, *, label: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be one bounded nonempty line")
    if redact_server_text(value) != value:
        raise ValueError(f"{label} cannot contain credential-shaped text")
    return value


def _plain_filename(value: str, *, label: str) -> str:
    _safe_line(value, label=label, maximum=255)
    if PurePosixPath(value).name != value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"{label} must be a plain filename")
    return value


def _absolute_path(value: str, *, label: str) -> str:
    _safe_line(value, label=label)
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError(f"{label} must be a normalized absolute non-root path")
    if str(path) != value:
        raise ValueError(f"{label} must be normalized")
    return value


def _safe_project_locator(value: str) -> str | None:
    try:
        return _absolute_path(value, label="project locator")
    except ValueError:
        return None


def _aware_timestamp(value: str, *, label: str) -> str:
    _safe_line(value, label=label, maximum=80)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return value


def _aware_time(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return value


class _StrictCaptureModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


class BackupKeptArtifactReference(_StrictCaptureModel):
    operation_id: str
    artifact_id: str
    source_name: str
    media_type: ArtifactMediaType
    expected_size_bytes: int | None = Field(default=None, ge=1)
    kept_filename: str
    kept_at: str

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="artifact operation identity")

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if _ARTIFACT_ID.fullmatch(value) is None:
            raise ValueError("artifact identity must be 24 lowercase hexadecimal characters")
        return value

    @field_validator("source_name", "kept_filename")
    @classmethod
    def validate_filename(cls, value: str, info) -> str:
        return _plain_filename(value, label=info.field_name.replace("_", " "))

    @field_validator("kept_at")
    @classmethod
    def validate_kept_at(cls, value: str) -> str:
        return _aware_timestamp(value, label="artifact kept time")


class BackupKeptResultViewReference(_StrictCaptureModel):
    view_id: str
    origin_operation_id: str
    latest_operation_id: str
    kept_filename: str
    content_sha256: str
    size_bytes: int = Field(gt=0)
    kept_at: str

    @field_validator("view_id")
    @classmethod
    def validate_view_id(cls, value: str) -> str:
        if _ARTIFACT_ID.fullmatch(value) is None:
            raise ValueError("result-view identity must be 24 lowercase hexadecimal characters")
        return value

    @field_validator("origin_operation_id", "latest_operation_id")
    @classmethod
    def validate_operation_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("kept_filename")
    @classmethod
    def validate_kept_filename(cls, value: str) -> str:
        return _plain_filename(value, label="kept result-view filename")

    @field_validator("content_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("kept result-view digest must be lowercase SHA-256")
        return value

    @field_validator("kept_at")
    @classmethod
    def validate_kept_at(cls, value: str) -> str:
        return _aware_timestamp(value, label="result-view kept time")


class BackupSnapshotProjectInventory(_StrictCaptureModel):
    project_id: str
    home_space_id: str | None
    locator: str | None
    status: Literal["capturable", "uncaptured"]
    recovery: BackupCheckoutRecoveryDescriptor | None = None
    task_operation_ids: tuple[str, ...] = ()
    kept_artifacts: tuple[BackupKeptArtifactReference, ...] = ()
    kept_result_views: tuple[BackupKeptResultViewReference, ...] = ()
    unavailable_reason: str | None = None
    unavailable_at: datetime | None = None

    @field_validator("project_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("home_space_id")
    @classmethod
    def validate_home_space_id(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_uuid4(value, label="home space identity")

    @field_validator("locator")
    @classmethod
    def validate_locator(cls, value: str | None) -> str | None:
        return None if value is None else _absolute_path(value, label="project locator")

    @field_validator("task_operation_ids")
    @classmethod
    def validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for operation_id in value:
            _canonical_uuid4(operation_id, label="task operation identity")
        if tuple(sorted(set(value))) != value:
            raise ValueError("captured task identities must be sorted and unique")
        return value

    @field_validator("unavailable_reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_line(
            value,
            label="project capture diagnostic",
            maximum=BACKUP_DIAGNOSTIC_MAX_CHARS,
        )

    @field_validator("unavailable_at")
    @classmethod
    def validate_unavailable_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_time(value, label="project unavailable time")

    @model_validator(mode="after")
    def validate_inventory(self) -> BackupSnapshotProjectInventory:
        entries = (
            len(self.task_operation_ids) + len(self.kept_artifacts) + len(self.kept_result_views)
        )
        if entries > BACKUP_INVENTORY_MAX_ENTRIES:
            raise ValueError("project backup inventory exceeds its entry bound")
        if self.status == "capturable":
            if (
                self.recovery is None
                or self.locator is None
                or self.home_space_id is None
                or self.recovery.project_id != self.project_id
                or self.recovery.home_space_id != self.home_space_id
                or self.unavailable_reason is not None
                or self.unavailable_at is not None
            ):
                raise ValueError("capturable project inventory requires exact recovery proof")
            _absolute_path(self.locator, label="project locator")
        elif any(
            (
                self.recovery is not None,
                bool(self.task_operation_ids),
                bool(self.kept_artifacts),
                bool(self.kept_result_views),
                self.unavailable_reason is None,
                self.unavailable_at is None,
            )
        ):
            raise ValueError("uncaptured project inventory carries only its failure")
        task_ids = set(self.task_operation_ids)
        artifact_keys = [
            (reference.operation_id, reference.artifact_id) for reference in self.kept_artifacts
        ]
        artifact_filenames = [reference.kept_filename for reference in self.kept_artifacts]
        if len(artifact_keys) != len(set(artifact_keys)) or len(artifact_filenames) != len(
            set(artifact_filenames)
        ):
            raise ValueError("project backup inventory repeats a kept artifact")
        if any(reference.operation_id not in task_ids for reference in self.kept_artifacts):
            raise ValueError("kept artifacts must belong to the captured project task set")
        view_ids = [reference.view_id for reference in self.kept_result_views]
        view_filenames = [reference.kept_filename for reference in self.kept_result_views]
        if len(view_ids) != len(set(view_ids)) or len(view_filenames) != len(set(view_filenames)):
            raise ValueError("project backup inventory repeats a kept result view")
        if any(
            reference.origin_operation_id not in task_ids
            or reference.latest_operation_id not in task_ids
            for reference in self.kept_result_views
        ):
            raise ValueError("kept result views must belong to the captured project task set")
        return self


class BackupSQLiteCaptureReceipt(_StrictCaptureModel):
    schema_version: Literal[BACKUP_SQLITE_CAPTURE_SCHEMA_VERSION] = (
        BACKUP_SQLITE_CAPTURE_SCHEMA_VERSION
    )
    capture_id: str
    captured_at: datetime
    rcp_source_commit: str
    space_id: str
    space_name: str
    snapshot_path: str
    database_schema_sha256: str
    sqlite_snapshot: BackupFileEntry
    app_data_plan: BackupAppDataCapturePlan
    projects: tuple[BackupSnapshotProjectInventory, ...]
    imported_source_inventories: tuple[BackupImportedProviderSourceInventory, ...] = ()
    status: Literal["complete", "partial"]

    @field_validator("capture_id", "space_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        return _aware_time(value, label="backup capture time")

    @field_validator("rcp_source_commit")
    @classmethod
    def validate_source_commit(cls, value: str) -> str:
        if _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("backup source commit must be a full lowercase object id")
        return value

    @field_validator("space_name")
    @classmethod
    def validate_space_name(cls, value: str) -> str:
        return _safe_line(value, label="backup space name", maximum=120)

    @field_validator("snapshot_path")
    @classmethod
    def validate_snapshot_path(cls, value: str) -> str:
        return _absolute_path(value, label="SQLite snapshot path")

    @field_validator("database_schema_sha256")
    @classmethod
    def validate_schema_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("database schema digest must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> BackupSQLiteCaptureReceipt:
        if (
            self.sqlite_snapshot.group != "sqlite_snapshot"
            or self.sqlite_snapshot.archive_path != "database/rcp.sqlite3"
            or self.sqlite_snapshot.source_relative_path != "rcp.sqlite3"
        ):
            raise ValueError("SQLite capture receipt requires its fixed database entry")
        snapshot = PurePosixPath(self.snapshot_path)
        if snapshot.name != "rcp.sqlite3" or snapshot.parent.name != f"backup-{self.capture_id}":
            raise ValueError("SQLite snapshot path is not bound to its capture identity")
        project_ids = [project.project_id for project in self.projects]
        if tuple(sorted(project_ids)) != tuple(project_ids) or len(project_ids) != len(
            set(project_ids)
        ):
            raise ValueError("SQLite capture projects must be sorted and unique")
        if any(
            project.status == "capturable" and project.home_space_id != self.space_id
            for project in self.projects
        ):
            raise ValueError("SQLite capture cannot include another space's project")
        imported_ids = [item.project_id for item in self.imported_source_inventories]
        if imported_ids and (
            tuple(sorted(imported_ids)) != tuple(imported_ids)
            or len(imported_ids) != len(set(imported_ids))
            or set(imported_ids) != set(project_ids)
        ):
            raise ValueError("SQLite capture imported sources must inventory every project")
        task_ids = [
            operation_id for project in self.projects for operation_id in project.task_operation_ids
        ]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("SQLite capture cannot assign one task to multiple projects")
        partial = bool(
            not self.app_data_plan.complete
            or any(project.status == "uncaptured" for project in self.projects)
        )
        if self.status != ("partial" if partial else "complete"):
            raise ValueError("SQLite capture status does not match its inventory")
        return self


@dataclass(frozen=True)
class BackupSQLiteCapturePublication:
    receipt: BackupSQLiteCaptureReceipt
    receipt_path: Path
    receipt_sha256: str


class BackupCaptureCoordinator:
    """Run O2a inside the exact service process that owns the live store."""

    def __init__(
        self,
        store: AppStore,
        data_dir: Path,
        metadata: ServerMetadata,
    ) -> None:
        resolved_data = data_dir.resolve()
        if (
            store.path.resolve() != resolved_data / "rcp.sqlite3"
            or store.space_kind != "team"
            or metadata.owner_kind != "cli"
            or metadata.control_socket is None
            or metadata.data_dir_id != data_dir_identity(resolved_data)
        ):
            raise ValueError("backup capture requires this exact installed team service")
        self.store = store
        self.data_dir = resolved_data
        self.metadata = metadata

    def capture_sqlite(self) -> BackupSQLiteCapturePublication:
        source_commit = self.metadata.running_commit
        if source_commit is None or _FULL_GIT_COMMIT.fullmatch(source_commit) is None:
            raise BackupCaptureUnavailable(
                "The installed service does not expose one immutable source commit."
            )
        capture_id = str(uuid.uuid4())
        captured_at = datetime.now(UTC)
        capture_root = self._create_capture_root(capture_id)
        app_data_plan = inspect_app_data_capture_plan(self.data_dir)
        if app_data_plan.database_path is None:
            raise BackupCaptureUnavailable(
                app_data_plan.database_unavailable_reason
                or "The application database is unavailable."
            )
        snapshot_path = capture_root / "rcp.sqlite3"
        self.store.online_snapshot(snapshot_path)
        snapshot_store = AppStore.open_read_only_snapshot(snapshot_path)
        space_id = snapshot_store.space_id
        space_name = snapshot_store.space_name
        if snapshot_store.space_kind != "team" or space_name is None:
            raise BackupCaptureUnavailable("The SQLite snapshot is not one named team space.")
        records = tuple(sorted(snapshot_store.projects(), key=lambda item: item.project_id))
        projects = tuple(
            self._project_inventory(snapshot_store, record, captured_at=captured_at)
            for record in records
        )
        if len(projects) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise BackupCaptureUnavailable("The project inventory exceeds its entry bound.")
        project_ids = tuple(record.project_id for record in records)
        try:
            if ImportedProviderSourceStore.project_ids(self.data_dir) != tuple(
                project_id
                for project_id in project_ids
                if os.path.lexists(
                    ImportedProviderSourceStore(self.data_dir, project_id).project_root
                )
            ):
                raise BackupCaptureUnavailable(
                    "The imported provider-source owners differ from the project catalog."
                )
            imported_source_inventories = tuple(
                BackupImportedProviderSourceInventory.model_validate(
                    ImportedProviderSourceStore(
                        self.data_dir,
                        project_id,
                    )
                    .inventory()
                    .model_dump()
                )
                for project_id in project_ids
            )
        except (OSError, ValueError) as exc:
            raise BackupCaptureUnavailable(
                "The imported provider-source inventory is invalid or unavailable."
            ) from exc
        database_schema_sha256 = _database_schema_sha256(snapshot_store)
        final_app_data_plan = inspect_app_data_capture_plan(self.data_dir)
        if final_app_data_plan.database_path != app_data_plan.database_path:
            raise BackupCaptureUnavailable(
                "The live application database boundary changed during capture."
            )
        app_data_plan = final_app_data_plan
        snapshot_sha256, snapshot_size = _file_sha256(snapshot_path)
        sqlite_entry = BackupFileEntry(
            archive_path="database/rcp.sqlite3",
            source_relative_path="rcp.sqlite3",
            group="sqlite_snapshot",
            sha256=snapshot_sha256,
            size_bytes=snapshot_size,
        )
        partial = not app_data_plan.complete or any(
            project.status == "uncaptured" for project in projects
        )
        receipt = BackupSQLiteCaptureReceipt(
            capture_id=capture_id,
            captured_at=captured_at,
            rcp_source_commit=source_commit,
            space_id=space_id,
            space_name=space_name,
            snapshot_path=str(snapshot_path),
            database_schema_sha256=database_schema_sha256,
            sqlite_snapshot=sqlite_entry,
            app_data_plan=app_data_plan,
            projects=projects,
            imported_source_inventories=imported_source_inventories,
            status="partial" if partial else "complete",
        )
        receipt_path = capture_root / "sqlite-capture.json"
        receipt_sha256 = write_immutable_backup_receipt(receipt_path, receipt)
        _fsync_directory(capture_root)
        return BackupSQLiteCapturePublication(
            receipt=receipt,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
        )

    def _create_capture_root(self, capture_id: str) -> Path:
        stage_root = self.data_dir / "run-stage"
        try:
            stage_root.mkdir(mode=0o700, exist_ok=True)
            stage = stage_root.lstat()
        except OSError as exc:
            raise BackupCaptureUnavailable("The backup staging boundary is unavailable.") from exc
        if (
            not stat.S_ISDIR(stage.st_mode)
            or stage.st_uid != os.geteuid()
            or stat.S_IMODE(stage.st_mode) & 0o077
        ):
            raise BackupCaptureUnavailable("The backup staging boundary is unsafe.")
        capture_root = stage_root / f"backup-{capture_id}"
        try:
            capture_root.mkdir(mode=0o700)
            capture = capture_root.lstat()
        except OSError as exc:
            raise BackupCaptureUnavailable(
                "The private backup capture could not be created."
            ) from exc
        if (
            not stat.S_ISDIR(capture.st_mode)
            or capture.st_uid != os.geteuid()
            or stat.S_IMODE(capture.st_mode) != 0o700
        ):
            raise BackupCaptureUnavailable("The private backup capture boundary is unsafe.")
        return capture_root

    def _project_inventory(
        self,
        snapshot_store: AppStore,
        record: ProjectRecord,
        *,
        captured_at: datetime,
    ) -> BackupSnapshotProjectInventory:
        try:
            if record.home_space_id != snapshot_store.space_id:
                raise BackupProjectInventoryUnavailable(
                    "The project home differs from the captured team space."
                )
            requests = snapshot_store.completed_project_provisioning_requests(record.project_id)
            registration = inspect_backup_project_registration(
                record,
                data_dir=self.data_dir,
                provisioning_requests=requests,
            )
            tasks = snapshot_store.all_project_agent_tasks(record.project_id)
            task_ids = tuple(sorted(task.operation_id for task in tasks))
            if len(task_ids) != len(set(task_ids)):
                raise BackupProjectInventoryUnavailable(
                    "The project task set repeats an operation identity."
                )
            artifacts = _kept_artifact_references(tasks)
            views = tuple(
                _kept_result_view_reference(view)
                for view in snapshot_store.kept_result_views(record.project_id)
            )
            return BackupSnapshotProjectInventory(
                project_id=record.project_id,
                home_space_id=record.home_space_id,
                locator=record.locator,
                status="capturable",
                recovery=registration.recovery,
                task_operation_ids=task_ids,
                kept_artifacts=artifacts,
                kept_result_views=views,
            )
        except BackupProjectUnavailable as exc:
            reason = str(exc)
        except (
            BackupProjectInventoryUnavailable,
            OSError,
            RuntimeError,
            ValueError,
            sqlite3.Error,
        ):
            reason = "The captured project inventory is invalid or unavailable."
        return BackupSnapshotProjectInventory(
            project_id=record.project_id,
            home_space_id=record.home_space_id,
            locator=_safe_project_locator(record.locator),
            status="uncaptured",
            unavailable_reason=reason,
            unavailable_at=captured_at,
        )


def _kept_artifact_references(tasks) -> tuple[BackupKeptArtifactReference, ...]:
    references: list[BackupKeptArtifactReference] = []
    for task in tasks:
        result = task.result
        if result is None or "artifacts" not in result:
            continue
        raw_artifacts = result["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise BackupProjectInventoryUnavailable("A task artifact list is malformed.")
        for raw in raw_artifacts:
            try:
                descriptor = AgentArtifactDescriptor.model_validate(raw)
            except (TypeError, ValueError) as exc:
                raise BackupProjectInventoryUnavailable(
                    "A task artifact descriptor is malformed."
                ) from exc
            if (descriptor.kept_filename is None) != (descriptor.kept_at is None):
                raise BackupProjectInventoryUnavailable(
                    "A task artifact has an incomplete kept-file binding."
                )
            if descriptor.kept_filename is None or descriptor.kept_at is None:
                continue
            references.append(
                BackupKeptArtifactReference(
                    operation_id=task.operation_id,
                    artifact_id=descriptor.artifact_id,
                    source_name=descriptor.name,
                    media_type=descriptor.media_type,
                    expected_size_bytes=descriptor.size_bytes,
                    kept_filename=descriptor.kept_filename,
                    kept_at=descriptor.kept_at,
                )
            )
    return tuple(
        sorted(
            references,
            key=lambda reference: (
                reference.kept_filename,
                reference.operation_id,
                reference.artifact_id,
            ),
        )
    )


def _kept_result_view_reference(view: ResultViewRecord) -> BackupKeptResultViewReference:
    if view.kept_filename is None or view.kept_at is None:
        raise BackupProjectInventoryUnavailable("A kept result-view binding is incomplete.")
    return BackupKeptResultViewReference(
        view_id=view.view_id,
        origin_operation_id=view.origin_operation_id,
        latest_operation_id=view.latest_operation_id,
        kept_filename=view.kept_filename,
        content_sha256=view.content_sha256,
        size_bytes=view.size_bytes,
        kept_at=view.kept_at,
    )


def _database_schema_sha256(store: AppStore) -> str:
    with store.connection() as connection:
        rows = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name, tbl_name
            """
        ).fetchall()
    payload = [dict(row) for row in rows]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BackupCaptureUnavailable("A backup capture input is not a regular file.")
        while True:
            chunk = os.read(descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        final = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ) or size != final.st_size:
            raise BackupCaptureUnavailable("The SQLite snapshot changed during hashing.")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def write_immutable_backup_receipt(path: Path, receipt: BaseModel) -> str:
    """Publish one strict bounded JSON receipt as a new read-only file."""

    payload = (
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > BACKUP_RECEIPT_MAX_BYTES:
        raise BackupCaptureUnavailable("The backup receipt exceeds its size bound.")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable receipt write")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def read_backup_sqlite_capture_receipt(
    path: Path,
    *,
    expected_sha256: str,
) -> BackupSQLiteCaptureReceipt:
    """Read one immutable O2a receipt through its published digest."""

    payload = read_immutable_backup_receipt(path, expected_sha256=expected_sha256)
    try:
        receipt = BackupSQLiteCaptureReceipt.model_validate_json(payload)
    except ValueError as exc:
        raise BackupCaptureUnavailable("The SQLite capture receipt is invalid.") from exc
    if path.name != "sqlite-capture.json" or path.parent.name != f"backup-{receipt.capture_id}":
        raise BackupCaptureUnavailable(
            "The SQLite capture receipt path is not bound to its capture identity."
        )
    return receipt


def read_immutable_backup_receipt(path: Path, *, expected_sha256: str) -> bytes:
    """Read one stable bounded receipt and verify its caller-published digest."""

    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("the expected backup receipt digest is invalid")
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("the backup receipt path must be absolute and normalized")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > BACKUP_RECEIPT_MAX_BYTES:
            raise BackupCaptureUnavailable("The backup receipt is unsafe or oversized.")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(BACKUP_COPY_BUFFER_BYTES, remaining))
            if not chunk:
                raise BackupCaptureUnavailable("The backup receipt is incomplete.")
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ):
        raise BackupCaptureUnavailable("The backup receipt changed while reading.")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise BackupCaptureUnavailable("The backup receipt digest does not match.")
    return payload


def validate_backup_sqlite_snapshot(receipt: BackupSQLiteCaptureReceipt) -> None:
    """Revalidate the O2a database bytes and identity before project-file capture."""

    snapshot_path = Path(receipt.snapshot_path)
    digest, size = _file_sha256(snapshot_path)
    if digest != receipt.sqlite_snapshot.sha256 or size != receipt.sqlite_snapshot.size_bytes:
        raise BackupCaptureUnavailable("The SQLite snapshot no longer matches its receipt.")
    store = AppStore.open_read_only_snapshot(snapshot_path)
    if (
        store.space_id != receipt.space_id
        or store.space_name != receipt.space_name
        or store.space_kind != "team"
        or _database_schema_sha256(store) != receipt.database_schema_sha256
    ):
        raise BackupCaptureUnavailable(
            "The SQLite snapshot identity no longer matches its receipt."
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BACKUP_SQLITE_CAPTURE_SCHEMA_VERSION",
    "BackupCaptureCoordinator",
    "BackupCaptureUnavailable",
    "BackupKeptArtifactReference",
    "BackupKeptResultViewReference",
    "BackupSQLiteCapturePublication",
    "BackupSQLiteCaptureReceipt",
    "BackupSnapshotProjectInventory",
    "read_immutable_backup_receipt",
    "read_backup_sqlite_capture_receipt",
    "validate_backup_sqlite_snapshot",
    "write_immutable_backup_receipt",
]
