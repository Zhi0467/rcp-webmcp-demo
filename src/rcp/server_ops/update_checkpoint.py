"""Coherent local rollback checkpoints for source-built server updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp.attachments import checkpoint_attachment_sets
from rcp.limits import (
    BACKUP_COPY_BUFFER_BYTES,
    BACKUP_INVENTORY_MAX_ENTRIES,
    BACKUP_RECEIPT_MAX_BYTES,
)
from rcp.runs.shared import checkpoint_local_recovery_stages
from rcp.server_ops.backup_capture import (
    BackupSQLiteCaptureReceipt,
    read_immutable_backup_receipt,
    validate_backup_sqlite_snapshot,
)
from rcp.server_ops.backup_models import (
    BackupImportedProviderSourceCapture,
    BackupProjectCapture,
)
from rcp.server_ops.backup_project_files import (
    BackupProjectFileCaptureReceipt,
)
from rcp.server_ops.rehearsal import VerifiedCandidateReceipt
from rcp.sources.imported import (
    ImportedProviderSourceInventory,
    ImportedProviderSourceStore,
)
from rcp.storage import AppStore
from rcp.storage.models import ProjectTransferUploadRecord
from rcp.transfer.target import target_transfer_archive_path

UPDATE_CHECKPOINT_SCHEMA_VERSION = 1
ROLLBACK_JOURNAL_SCHEMA_VERSION = 1

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_ROOT_NAME = re.compile(r"checkpoint-([0-9a-f]{40})-([0-9a-f]{32})")
_VERIFIED_RECEIPT_NAME = re.compile(r"verified-candidate-([0-9a-f]{40})-([0-9a-f-]{36})\.json")
_DIRECTORY_MODE = 0o700
_RECEIPT_MODE = 0o600
_PAYLOAD_FILE_MODE = 0o400
_MAX_JSON_BYTES = BACKUP_RECEIPT_MAX_BYTES


class UpdateCheckpointRefused(RuntimeError):
    """The final local rollback boundary was incomplete or unsafe."""


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


def _absolute_path(value: str, *, label: str) -> str:
    if (
        not value
        or value != value.strip()
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be one bounded absolute path")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be absolute and normalized")
    return value


def _relative_path(value: str, *, label: str) -> str:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be one bounded relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be normalized and relative")
    return value


class UpdateCheckpointFile(_StrictModel):
    relative_path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    mode: int = Field(ge=0, le=0o777)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _relative_path(value, label="checkpoint file path")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("checkpoint files require lowercase SHA-256")
        return value


class UpdateCheckpointRoot(_StrictModel):
    kind: Literal["app_data", "project_research"]
    identity: str
    live_path: str
    quarantine_path: str
    partial_path: str
    archive_path: str
    directories: tuple[str, ...] = ()
    files: tuple[UpdateCheckpointFile, ...]

    @field_validator("live_path", "quarantine_path", "partial_path")
    @classmethod
    def validate_absolute_path(cls, value: str, info) -> str:
        return _absolute_path(value, label=info.field_name.replace("_", " "))

    @field_validator("archive_path")
    @classmethod
    def validate_archive_path(cls, value: str) -> str:
        return _relative_path(value, label="checkpoint archive root")

    @field_validator("directories")
    @classmethod
    def validate_directories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _relative_path(path, label="checkpoint directory path")
        if tuple(sorted(set(value))) != value:
            raise ValueError("checkpoint directory paths must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_root(self) -> UpdateCheckpointRoot:
        paths = [item.relative_path for item in self.files]
        if tuple(sorted(paths)) != tuple(paths) or len(paths) != len(set(paths)):
            raise ValueError("checkpoint root files must be sorted and unique")
        if len(paths) + len(self.directories) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise ValueError("checkpoint root inventory exceeds its bound")
        live = Path(self.live_path)
        if any(
            candidate == live or candidate in live.parents or live in candidate.parents
            for candidate in (Path(self.quarantine_path), Path(self.partial_path))
        ):
            raise ValueError("checkpoint live and quarantine roots must not overlap")
        if self.kind == "app_data" and self.identity != "app-data":
            raise ValueError("the app-data checkpoint root has one fixed identity")
        if self.kind == "project_research":
            _canonical_uuid4(self.identity, label="checkpoint project identity")
            if live.name != ".research":
                raise ValueError("project rollback roots must be exact .research directories")
        return self


class UpdateCheckpointProject(_StrictModel):
    project_id: str
    capture_status: Literal["captured", "remote_unreachable"]
    restore_kind: Literal["local_research", "remote_excluded"]
    archive_path: str | None
    files: tuple[UpdateCheckpointFile, ...] = ()

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="checkpoint project identity")

    @field_validator("archive_path")
    @classmethod
    def validate_archive_path(cls, value: str | None) -> str | None:
        return None if value is None else _relative_path(value, label="project archive root")

    @model_validator(mode="after")
    def validate_project(self) -> UpdateCheckpointProject:
        paths = [item.relative_path for item in self.files]
        if tuple(sorted(paths)) != tuple(paths) or len(paths) != len(set(paths)):
            raise ValueError("checkpoint project files must be sorted and unique")
        if len(paths) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise ValueError("checkpoint project file inventory exceeds its bound")
        if self.capture_status == "remote_unreachable":
            if (
                self.restore_kind != "remote_excluded"
                or self.archive_path is not None
                or self.files
            ):
                raise ValueError("unreachable remote projects cannot carry local rollback bytes")
        elif self.archive_path is None:
            raise ValueError("captured projects require their exact checkpoint archive root")
        return self


class VerifiedUpdateCheckpoint(_StrictModel):
    schema_version: Literal[1] = UPDATE_CHECKPOINT_SCHEMA_VERSION
    checkpoint_id: str
    installation_id: str
    space_id: str
    capture_id: str
    base_commit: str
    candidate_commit: str
    previous_release_path: str
    candidate_release_path: str
    operation_root: str
    manifest_path: str
    sqlite_receipt_archive_path: str
    sqlite_receipt_sha256: str
    sqlite_snapshot_sha256: str
    project_receipt_archive_path: str
    project_receipt_sha256: str
    candidate_receipt_archive_path: str
    candidate_receipt_sha256: str
    projects: tuple[UpdateCheckpointProject, ...]
    imported_sources: tuple[BackupImportedProviderSourceCapture, ...] = ()
    roots: tuple[UpdateCheckpointRoot, ...]
    status: Literal["verified"] = "verified"
    created_at: datetime

    @field_validator("checkpoint_id", "installation_id", "space_id", "capture_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("base_commit", "candidate_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("update checkpoints require full lowercase Git commits")
        return value

    @field_validator(
        "previous_release_path",
        "candidate_release_path",
        "operation_root",
        "manifest_path",
    )
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        return _absolute_path(value, label=info.field_name.replace("_", " "))

    @field_validator(
        "sqlite_receipt_archive_path",
        "project_receipt_archive_path",
        "candidate_receipt_archive_path",
    )
    @classmethod
    def validate_archive_path(cls, value: str, info) -> str:
        return _relative_path(value, label=info.field_name.replace("_", " "))

    @field_validator(
        "sqlite_receipt_sha256",
        "sqlite_snapshot_sha256",
        "project_receipt_sha256",
        "candidate_receipt_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("update checkpoint digests must be lowercase SHA-256")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("update checkpoint time requires a UTC offset")
        return value

    @model_validator(mode="after")
    def validate_checkpoint(self) -> VerifiedUpdateCheckpoint:
        root = Path(self.operation_root)
        if Path(self.manifest_path) != root / "checkpoint.json":
            raise ValueError("checkpoint manifest path escaped its operation root")
        if self.base_commit == self.candidate_commit:
            raise ValueError("checkpoint candidate must differ from its base release")
        matched = _CHECKPOINT_ROOT_NAME.fullmatch(root.name)
        if (
            matched is None
            or matched.group(1) != self.candidate_commit
            or matched.group(2) != uuid.UUID(self.checkpoint_id).hex
        ):
            raise ValueError("checkpoint operation root and candidate commit disagree")
        project_ids = [project.project_id for project in self.projects]
        if tuple(sorted(project_ids)) != tuple(project_ids) or len(project_ids) != len(
            set(project_ids)
        ):
            raise ValueError("checkpoint projects must be sorted and unique")
        live_paths = [item.live_path for item in self.roots]
        if (
            len(live_paths) != len(set(live_paths))
            or sum(item.kind == "app_data" for item in self.roots) != 1
        ):
            raise ValueError("checkpoint replacement roots must be unique with one app-data root")
        replacement_paths = [
            Path(path)
            for item in self.roots
            for path in (item.live_path, item.quarantine_path, item.partial_path)
        ]
        if len(replacement_paths) != len(set(replacement_paths)):
            raise ValueError("checkpoint live and quarantine paths must be globally unique")
        for index, first in enumerate(replacement_paths):
            for second in replacement_paths[index + 1 :]:
                if first in second.parents or second in first.parents:
                    raise ValueError("checkpoint replacement paths must not overlap")
        if any(
            root == Path(self.operation_root)
            or root in Path(self.operation_root).parents
            or Path(self.operation_root) in root.parents
            for root in replacement_paths
        ):
            raise ValueError("checkpoint storage must not overlap a replacement path")
        local_ids = {
            project.project_id
            for project in self.projects
            if project.restore_kind == "local_research"
        }
        root_ids = {item.identity for item in self.roots if item.kind == "project_research"}
        if local_ids != root_ids:
            raise ValueError("checkpoint local projects and replacement roots disagree")
        imported_ids = [capture.project_id for capture in self.imported_sources]
        if imported_ids and (
            tuple(sorted(imported_ids)) != tuple(imported_ids)
            or len(imported_ids) != len(set(imported_ids))
            or set(imported_ids) != set(project_ids)
        ):
            raise ValueError("checkpoint imported sources must inventory every project")
        return self


class RollbackJournal(_StrictModel):
    schema_version: Literal[1] = ROLLBACK_JOURNAL_SCHEMA_VERSION
    checkpoint_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    previous_release_path: str
    phase: Literal["prepared", "quarantined", "restored", "verified", "complete"]
    quarantine_paths: tuple[str, ...]
    partial_paths: tuple[str, ...]
    updated_at: datetime

    @field_validator("checkpoint_id")
    @classmethod
    def validate_checkpoint_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="rollback checkpoint identity")

    @field_validator("checkpoint_path", "previous_release_path")
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        return _absolute_path(value, label=info.field_name.replace("_", " "))

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("rollback journal requires one checkpoint digest")
        return value

    @field_validator("quarantine_paths", "partial_paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        for path in value:
            _absolute_path(path, label=info.field_name.replace("_", " "))
        if len(value) != len(set(value)):
            raise ValueError("rollback journal paths must be unique")
        return value

    @field_validator("updated_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("rollback journal time requires a UTC offset")
        return value


def _read_sqlite_receipt(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[BackupSQLiteCaptureReceipt, bytes]:
    payload = read_immutable_backup_receipt(path, expected_sha256=expected_sha256)
    try:
        receipt = BackupSQLiteCaptureReceipt.model_validate_json(payload)
    except ValueError as exc:
        raise UpdateCheckpointRefused("The SQLite capture receipt is invalid.") from exc
    if path.name != "sqlite-capture.json" or path.parent.name != f"backup-{receipt.capture_id}":
        raise UpdateCheckpointRefused(
            "The SQLite capture receipt path is not bound to its capture identity."
        )
    return receipt, payload


def _read_project_receipt(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[BackupProjectFileCaptureReceipt, bytes]:
    payload = read_immutable_backup_receipt(path, expected_sha256=expected_sha256)
    try:
        receipt = BackupProjectFileCaptureReceipt.model_validate_json(payload)
    except ValueError as exc:
        raise UpdateCheckpointRefused("The project-file capture receipt is invalid.") from exc
    if path.name != "project-files.json" or path.parent.name != f"backup-{receipt.capture_id}":
        raise UpdateCheckpointRefused(
            "The project-file receipt path is not bound to its capture identity."
        )
    return receipt, payload


def _read_candidate_receipt(
    path: Path,
    *,
    expected_uid: int,
    expected_sha256: str,
) -> tuple[VerifiedCandidateReceipt, bytes]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("the expected verified-candidate receipt digest is invalid")
    payload = _read_private_file(path, expected_uid=expected_uid, expected_mode=_RECEIPT_MODE)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise UpdateCheckpointRefused("The verified-candidate receipt digest changed.")
    try:
        receipt = VerifiedCandidateReceipt.model_validate_json(payload)
    except ValueError as exc:
        raise UpdateCheckpointRefused("The verified-candidate receipt is invalid.") from exc
    matched = _VERIFIED_RECEIPT_NAME.fullmatch(path.name)
    if (
        matched is None
        or matched.group(1) != receipt.candidate_commit
        or matched.group(2) != receipt.capture_id
        or receipt.receipt_path != str(path)
    ):
        raise UpdateCheckpointRefused("The verified-candidate receipt path and commit disagree.")
    return receipt, payload


class UpdateCheckpointCoordinator:
    """Bind one final O2 capture and matching rehearsal receipt into rollback state."""

    def __init__(
        self,
        *,
        data_dir: Path,
        update_root: Path,
        previous_release_path: Path,
        expected_uid: int,
    ) -> None:
        for path, label in (
            (data_dir, "live app-data root"),
            (update_root, "update checkpoint root"),
            (previous_release_path, "previous release path"),
        ):
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{label} must be absolute and normalized")
        self.data_dir = data_dir
        self.update_root = update_root
        self.previous_release_path = previous_release_path
        self.expected_uid = expected_uid

    def create(
        self,
        *,
        sqlite_receipt_path: Path,
        sqlite_receipt_sha256: str,
        project_receipt_path: Path,
        project_receipt_sha256: str,
        candidate_receipt_path: Path,
        candidate_receipt_sha256: str,
    ) -> VerifiedUpdateCheckpoint:
        sqlite_receipt, sqlite_receipt_payload = _read_sqlite_receipt(
            sqlite_receipt_path,
            expected_sha256=sqlite_receipt_sha256,
        )
        project_receipt, project_receipt_payload = _read_project_receipt(
            project_receipt_path,
            expected_sha256=project_receipt_sha256,
        )
        candidate_receipt, candidate_receipt_payload = _read_candidate_receipt(
            candidate_receipt_path,
            expected_uid=self.expected_uid,
            expected_sha256=candidate_receipt_sha256,
        )
        validate_backup_sqlite_snapshot(sqlite_receipt)
        self._validate_receipts(
            sqlite_receipt,
            project_receipt,
            candidate_receipt,
            sqlite_receipt_sha256=sqlite_receipt_sha256,
            project_receipt_sha256=project_receipt_sha256,
        )
        self._validate_live_boundaries(
            sqlite_receipt_path,
            project_receipt_path,
            candidate_receipt_path,
        )

        checkpoint_id = str(uuid.uuid4())
        operation_root = self.update_root / (
            f"checkpoint-{candidate_receipt.candidate_commit}-{uuid.UUID(checkpoint_id).hex}"
        )
        try:
            operation_root.mkdir(mode=_DIRECTORY_MODE)
            (operation_root / "payload").mkdir(mode=_DIRECTORY_MODE)
            (operation_root / "proof").mkdir(mode=_DIRECTORY_MODE)
            _fsync_directory(self.update_root)
        except OSError as exc:
            raise UpdateCheckpointRefused(
                "The private update-checkpoint root could not be created."
            ) from exc

        try:
            proof_digests = self._copy_proofs(
                operation_root,
                sqlite_receipt_payload=sqlite_receipt_payload,
                sqlite_receipt_sha256=sqlite_receipt_sha256,
                project_receipt_payload=project_receipt_payload,
                project_receipt_sha256=project_receipt_sha256,
                candidate_receipt_payload=candidate_receipt_payload,
                candidate_receipt_sha256=candidate_receipt_sha256,
            )
            snapshot_store = AppStore.open_read_only_snapshot(Path(sqlite_receipt.snapshot_path))
            if (
                snapshot_store.space_kind != "team"
                or snapshot_store.space_id != sqlite_receipt.space_id
            ):
                raise UpdateCheckpointRefused(
                    "The final SQLite snapshot does not belong to the captured team space."
                )
            stages = checkpoint_local_recovery_stages(snapshot_store, self.data_dir)
            self._require_empty_transfer_exports()
            app_root_path = operation_root / "payload" / "app-data"
            app_root_path.mkdir(mode=_DIRECTORY_MODE)
            app_files = [
                _copy_declared_file(
                    Path(sqlite_receipt.snapshot_path),
                    app_root_path / "rcp.sqlite3",
                    relative_path="rcp.sqlite3",
                    expected_sha256=sqlite_receipt.sqlite_snapshot.sha256,
                    expected_size=sqlite_receipt.sqlite_snapshot.size_bytes,
                    restore_mode=0o600,
                )
            ]
            app_directories: set[str] = set()
            transfer_directories, transfer_files = self._copy_transfer_inbox(
                snapshot_store,
                app_root_path,
            )
            app_directories.update(transfer_directories)
            app_files.extend(transfer_files)
            bootstrap_root = self.data_dir / "bootstrap-manifests"
            if bootstrap_root.exists():
                directories, files = _snapshot_tree(
                    bootstrap_root,
                    app_root_path / "bootstrap-manifests",
                    relative_prefix=PurePosixPath("bootstrap-manifests"),
                )
                app_directories.update(directories)
                app_files.extend(files)
            project_snapshots_root = self.data_dir / "project-snapshots"
            if project_snapshots_root.exists():
                directories, files = _snapshot_tree(
                    project_snapshots_root,
                    app_root_path / "project-snapshots",
                    relative_prefix=PurePosixPath("project-snapshots"),
                )
                app_directories.update(directories)
                app_files.extend(files)
            source_directories, source_files = self._copy_imported_sources(
                app_root_path,
                project_receipt,
                captured_root=("project-sources" in sqlite_receipt.app_data_plan.captured_entries),
            )
            app_directories.update(source_directories)
            app_files.extend(source_files)
            for stage in stages:
                relative_root = PurePosixPath("run-stage") / stage.root.name
                directories, files = _snapshot_tree(
                    stage.root,
                    app_root_path.joinpath(*relative_root.parts),
                    relative_prefix=relative_root,
                )
                app_directories.update(directories)
                app_files.extend(files)
            with checkpoint_attachment_sets(self.data_dir / "chat-attachments") as attachments:
                for attachment in attachments:
                    relative_root = PurePosixPath("chat-attachments") / attachment.attachment_set_id
                    directories, files = _snapshot_tree(
                        attachment.root,
                        app_root_path.joinpath(*relative_root.parts),
                        relative_prefix=relative_root,
                    )
                    app_directories.update(directories)
                    app_files.extend(files)

            app_root = self._root_model(
                checkpoint_id,
                kind="app_data",
                identity="app-data",
                live_path=self.data_dir,
                archive_path="payload/app-data",
                directories=app_directories,
                files=app_files,
            )
            projects, project_roots = self._copy_projects(
                operation_root,
                project_receipt,
                checkpoint_id=checkpoint_id,
                capture_root=project_receipt_path.parent,
            )
            _set_private_directory_modes(operation_root / "payload")
            checkpoint = VerifiedUpdateCheckpoint(
                checkpoint_id=checkpoint_id,
                installation_id=candidate_receipt.installation_id,
                space_id=sqlite_receipt.space_id,
                capture_id=sqlite_receipt.capture_id,
                base_commit=candidate_receipt.base_running_commit,
                candidate_commit=candidate_receipt.candidate_commit,
                previous_release_path=str(self.previous_release_path),
                candidate_release_path=candidate_receipt.release_path,
                operation_root=str(operation_root),
                manifest_path=str(operation_root / "checkpoint.json"),
                sqlite_receipt_archive_path="proof/sqlite-capture.json",
                sqlite_receipt_sha256=proof_digests[0],
                sqlite_snapshot_sha256=sqlite_receipt.sqlite_snapshot.sha256,
                project_receipt_archive_path="proof/project-files.json",
                project_receipt_sha256=proof_digests[1],
                candidate_receipt_archive_path="proof/verified-candidate.json",
                candidate_receipt_sha256=proof_digests[2],
                projects=projects,
                imported_sources=project_receipt.imported_sources,
                roots=(app_root, *project_roots),
                created_at=datetime.now(UTC),
            )
            _validate_replacement_targets(checkpoint, expected_uid=self.expected_uid)
            _verify_checkpoint_in_temporary_root(checkpoint, expected_uid=self.expected_uid)
            _write_new_model(Path(checkpoint.manifest_path), checkpoint)
            _fsync_tree(operation_root)
            _fsync_directory(self.update_root)
            return read_verified_update_checkpoint(
                Path(checkpoint.manifest_path),
                expected_uid=self.expected_uid,
            )
        except BaseException:
            # A retained operation without checkpoint.json is deliberately unusable and
            # blocks later update admission for operator inspection.
            raise

    def _copy_imported_sources(
        self,
        app_root_path: Path,
        receipt: BackupProjectFileCaptureReceipt,
        *,
        captured_root: bool,
    ) -> tuple[set[str], list[UpdateCheckpointFile]]:
        directories: set[str] = set()
        files: list[UpdateCheckpointFile] = []
        collection = app_root_path / "project-sources"
        if captured_root:
            collection.mkdir(mode=_DIRECTORY_MODE)
            directories.add("project-sources")
        expected_present = tuple(
            sorted(capture.project_id for capture in receipt.imported_sources if capture.present)
        )
        try:
            if expected_present and not captured_root:
                raise UpdateCheckpointRefused(
                    "The final capture omitted the imported provider-source app-data root."
                )
            if ImportedProviderSourceStore.project_ids(self.data_dir) != expected_present:
                raise UpdateCheckpointRefused(
                    "The imported provider-source owners changed after final capture."
                )
            for capture in receipt.imported_sources:
                expected = ImportedProviderSourceInventory.model_validate(
                    capture.inventory.model_dump()
                )
                owner = ImportedProviderSourceStore(self.data_dir, capture.project_id)
                if not capture.present:
                    if os.path.lexists(owner.root):
                        raise UpdateCheckpointRefused(
                            "An imported provider-source root appeared after final capture."
                        )
                    continue
                project_root = collection / capture.project_id
                project_root.mkdir(mode=_DIRECTORY_MODE)
                snapshot = owner.capture_snapshot(
                    project_root / "provider-history",
                    expected_inventory=expected,
                )
                expected_files = {
                    entry.source_relative_path.removeprefix("provider-history/"): (
                        entry.sha256,
                        entry.size_bytes,
                    )
                    for entry in capture.files
                }
                if (
                    not snapshot.present
                    or {
                        item.relative_path: (item.sha256, item.size_bytes)
                        for item in snapshot.files
                    }
                    != expected_files
                ):
                    raise UpdateCheckpointRefused(
                        "The imported provider-source checkpoint differs from final capture."
                    )
                prefix = PurePosixPath("project-sources") / capture.project_id
                directories.update(
                    {
                        prefix.as_posix(),
                        (prefix / "provider-history").as_posix(),
                        *(
                            (prefix / "provider-history" / item.provider).as_posix()
                            for item in expected.files
                        ),
                    }
                )
                files.extend(
                    UpdateCheckpointFile(
                        relative_path=(prefix / "provider-history" / item.relative_path).as_posix(),
                        sha256=item.sha256,
                        size_bytes=item.size_bytes,
                        mode=0o400,
                    )
                    for item in snapshot.files
                )
        except UpdateCheckpointRefused:
            raise
        except (OSError, ValueError) as exc:
            raise UpdateCheckpointRefused(
                "The imported provider-source checkpoint could not be captured safely."
            ) from exc
        return directories, files

    def _validate_receipts(
        self,
        sqlite: BackupSQLiteCaptureReceipt,
        projects: BackupProjectFileCaptureReceipt,
        candidate: VerifiedCandidateReceipt,
        *,
        sqlite_receipt_sha256: str,
        project_receipt_sha256: str,
    ) -> None:
        if (
            sqlite.capture_id != projects.capture_id
            or sqlite.capture_id != candidate.capture_id
            or sqlite.space_id != projects.space_id
            or sqlite.space_id != candidate.space_id
            or sqlite.rcp_source_commit != candidate.base_running_commit
            or projects.rcp_source_commit != candidate.base_running_commit
            or sqlite.sqlite_snapshot.sha256 != candidate.sqlite_snapshot_sha256
            or projects.sqlite_receipt_sha256 != sqlite_receipt_sha256
            or projects.sqlite_snapshot_sha256 != candidate.sqlite_snapshot_sha256
            or project_receipt_sha256 != candidate.project_capture_sha256
            or self.previous_release_path.name != candidate.base_running_commit
            or Path(sqlite.app_data_plan.data_dir) != self.data_dir
            or Path(sqlite.app_data_plan.database_path or "") != self.data_dir / "rcp.sqlite3"
            or Path(sqlite.snapshot_path)
            != self.data_dir / "run-stage" / f"backup-{sqlite.capture_id}" / "rcp.sqlite3"
        ):
            raise UpdateCheckpointRefused(
                "The rollback capture, candidate rehearsal, and previous release do not name "
                "one exact final boundary."
            )
        if sqlite.app_data_plan.deferred_entries or sqlite.app_data_plan.unclassified_entries:
            raise UpdateCheckpointRefused(
                "The final app-data boundary contains deferred or unclassified durable state."
            )
        sqlite_sources = {
            item.project_id: item.model_dump(mode="json")
            for item in sqlite.imported_source_inventories
        }
        captured_sources = {
            item.project_id: item.inventory.model_dump(mode="json")
            for item in projects.imported_sources
        }
        if sqlite_sources != captured_sources:
            raise UpdateCheckpointRefused(
                "The final imported provider-source inventories do not match."
            )
        sqlite_projects = {project.project_id: project for project in sqlite.projects}
        captured_projects = {project.project_id: project for project in projects.projects}
        candidate_projects = {project.project_id: project for project in candidate.projects}
        if not (sqlite_projects.keys() == captured_projects.keys() == candidate_projects.keys()):
            raise UpdateCheckpointRefused("The final project inventories do not match.")
        for project_id, captured in captured_projects.items():
            sqlite_project = sqlite_projects[project_id]
            candidate_project = candidate_projects[project_id]
            if captured.status == "captured":
                if sqlite_project.status != "capturable" or candidate_project.status != "verified":
                    raise UpdateCheckpointRefused(
                        "A locally capturable project lacks complete rehearsal proof."
                    )
            elif not (
                captured.unavailable_kind == "remote_unreachable"
                and sqlite_project.status == "uncaptured"
                and candidate_project.status == "not_replay_verified"
            ):
                raise UpdateCheckpointRefused(
                    "Only one already-unreachable SSH project may remain explicitly excluded."
                )

    def _validate_live_boundaries(
        self,
        sqlite_path: Path,
        project_path: Path,
        candidate_path: Path,
    ) -> None:
        expected_capture_root = self.data_dir / "run-stage" / sqlite_path.parent.name
        if (
            sqlite_path.parent != expected_capture_root
            or project_path.parent != expected_capture_root
            or sqlite_path.name != "sqlite-capture.json"
            or project_path.name != "project-files.json"
            or candidate_path.parent != self.update_root
        ):
            raise UpdateCheckpointRefused(
                "The final O2 capture escaped this server's private run-stage boundary."
            )
        for path, label in (
            (self.update_root, "update checkpoint root"),
            (self.data_dir, "live app-data root"),
            (expected_capture_root, "final capture root"),
        ):
            _require_directory(path, expected_uid=self.expected_uid, label=label)

    def _copy_transfer_inbox(
        self,
        snapshot_store: AppStore,
        app_root_path: Path,
    ) -> tuple[set[str], list[UpdateCheckpointFile]]:
        """Capture only receipt-backed complete target upload archives.

        The live inbox is deliberately excluded from ordinary backup and
        rehearsal.  An update checkpoint is the one local boundary that may
        retain a complete upload, but only when the immutable SQLite snapshot
        contains a typed completion row that binds the request, archive, and
        exact bytes.  Filesystem names are derived through the target owner;
        no path supplied by a database row is trusted.
        """

        uploads = self._read_complete_transfer_uploads(snapshot_store)
        live_root = self.data_dir / "transfer-inbox"
        if not uploads:
            if os.path.lexists(live_root):
                self._require_private_transfer_directory(live_root)
            self._require_empty_transfer_root(
                "transfer-inbox",
                "A transfer inbox entry has no typed completed-upload proof yet. Finish or "
                "remove the transfer before this source update.",
            )
            return set(), []

        self._require_private_transfer_directory(live_root)
        try:
            entries = tuple(live_root.iterdir())
        except OSError as exc:
            raise UpdateCheckpointRefused(
                "The transfer inbox cannot be inventoried safely."
            ) from exc
        expected_names = {f"{upload.request_id}.rcp-transfer" for upload in uploads}
        actual_names = {entry.name for entry in entries}
        if expected_names - actual_names:
            raise UpdateCheckpointRefused("A complete transfer upload is missing its archive file.")
        if actual_names - expected_names:
            raise UpdateCheckpointRefused(
                "The transfer inbox contains an unknown, partial, or untyped entry."
            )

        destination_root = app_root_path / "transfer-inbox"
        destination_root.mkdir(mode=_DIRECTORY_MODE)
        copied: list[UpdateCheckpointFile] = []
        for upload in uploads:
            source = target_transfer_archive_path(self.data_dir, upload.request_id)
            self._require_transfer_archive_file(source)
            copied_file = _copy_declared_file(
                source,
                destination_root / source.name,
                relative_path=f"transfer-inbox/{source.name}",
                expected_sha256=upload.archive_sha256,
                expected_size=upload.archive_size_bytes,
                restore_mode=0o600,
            )
            # The source mode/owner are part of this checkpoint boundary, not
            # merely properties of the copied payload.
            self._require_transfer_archive_file(source)
            copied.append(copied_file)
        return {"transfer-inbox"}, copied

    def _read_complete_transfer_uploads(
        self,
        snapshot_store: AppStore,
    ) -> tuple[ProjectTransferUploadRecord, ...]:
        try:
            stored = snapshot_store.target_project_transfer_uploads()
        except (KeyError, RuntimeError, ValueError, sqlite3.Error) as exc:
            raise UpdateCheckpointRefused(
                "The SQLite snapshot has no readable typed transfer-upload table."
            ) from exc
        if len(stored) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise UpdateCheckpointRefused("The transfer-upload inventory exceeds its bound.")

        uploads: list[ProjectTransferUploadRecord] = []
        for upload in stored:
            if upload.status == "consumed":
                continue
            if upload.status != "complete" or upload.receipt is None:
                raise UpdateCheckpointRefused(
                    "A transfer inbox upload is not at its durable complete boundary."
                )
            try:
                request = snapshot_store.project_transfer_request(upload.request_id)
            except (KeyError, RuntimeError, ValueError) as exc:
                raise UpdateCheckpointRefused(
                    "A complete transfer upload is not bound to a valid target request."
                ) from exc
            if (
                request is None
                or request.side != "target"
                or request.project_id != upload.project_id
                or request.archive_sha256 != upload.archive_sha256
                or request.archive_size_bytes != upload.archive_size_bytes
            ):
                raise UpdateCheckpointRefused(
                    "A complete transfer upload is not bound to its target archive request."
                )
            uploads.append(upload)
        return tuple(uploads)

    def _require_private_transfer_directory(self, path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise UpdateCheckpointRefused("The transfer inbox is unavailable.") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or path.is_symlink()
            or metadata.st_uid != self.expected_uid
            or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
        ):
            raise UpdateCheckpointRefused("The transfer inbox has unsafe ownership, mode, or type.")

    def _require_transfer_archive_file(self, path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise UpdateCheckpointRefused(
                "A complete transfer upload is missing its archive file."
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or metadata.st_uid != self.expected_uid
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise UpdateCheckpointRefused(
                "A complete transfer upload has unsafe ownership, mode, or type."
            )

    def _require_empty_transfer_exports(self) -> None:
        self._require_empty_transfer_root(
            "transfer-exports",
            "A sealed transfer export has no update-checkpoint integration yet. Finish the "
            "transfer before this source update.",
        )

    def _require_empty_transfer_root(self, name: str, diagnostic: str) -> None:
        root = self.data_dir / name
        if not os.path.lexists(root):
            return
        _require_directory(root, expected_uid=self.expected_uid, label=name.replace("-", " "))
        try:
            entries = tuple(root.iterdir())
        except OSError as exc:
            raise UpdateCheckpointRefused(f"The {name} root cannot be inventoried.") from exc
        if entries:
            raise UpdateCheckpointRefused(diagnostic)

    def _copy_proofs(
        self,
        operation_root: Path,
        *,
        sqlite_receipt_payload: bytes,
        sqlite_receipt_sha256: str,
        project_receipt_payload: bytes,
        project_receipt_sha256: str,
        candidate_receipt_payload: bytes,
        candidate_receipt_sha256: str,
    ) -> tuple[str, str, str]:
        values = (
            (sqlite_receipt_payload, "sqlite-capture.json", sqlite_receipt_sha256),
            (project_receipt_payload, "project-files.json", project_receipt_sha256),
            (candidate_receipt_payload, "verified-candidate.json", candidate_receipt_sha256),
        )
        copied: list[str] = []
        for payload, name, expected in values:
            digest = hashlib.sha256(payload).hexdigest()
            if digest != expected:
                raise UpdateCheckpointRefused("A checkpoint proof changed after validation.")
            _write_new_payload(operation_root / "proof" / name, payload)
            copied.append(digest)
        return copied[0], copied[1], copied[2]

    def _copy_projects(
        self,
        operation_root: Path,
        receipt: BackupProjectFileCaptureReceipt,
        *,
        checkpoint_id: str,
        capture_root: Path,
    ) -> tuple[tuple[UpdateCheckpointProject, ...], tuple[UpdateCheckpointRoot, ...]]:
        projects: list[UpdateCheckpointProject] = []
        roots: list[UpdateCheckpointRoot] = []
        for project in receipt.projects:
            if project.status == "uncaptured":
                projects.append(
                    UpdateCheckpointProject(
                        project_id=project.project_id,
                        capture_status="remote_unreachable",
                        restore_kind="remote_excluded",
                        archive_path=None,
                    )
                )
                continue
            archive_root = PurePosixPath("payload/project-files") / project.project_id
            destination_root = operation_root.joinpath(*archive_root.parts)
            destination_root.mkdir(mode=_DIRECTORY_MODE, parents=True)
            copied_files: list[UpdateCheckpointFile] = []
            for entry in project.files:
                source = capture_root.joinpath(*PurePosixPath(entry.archive_path).parts)
                relative = PurePosixPath(entry.source_relative_path)
                copied_files.append(
                    _copy_declared_file(
                        source,
                        destination_root.joinpath(*relative.parts),
                        relative_path=relative.as_posix(),
                        expected_sha256=entry.sha256,
                        expected_size=entry.size_bytes,
                        restore_mode=0o600,
                    )
                )
            restore_kind, research_root = _project_restore_location(project)
            projects.append(
                UpdateCheckpointProject(
                    project_id=project.project_id,
                    capture_status="captured",
                    restore_kind=restore_kind,
                    archive_path=archive_root.as_posix(),
                    files=tuple(sorted(copied_files, key=lambda item: item.relative_path)),
                )
            )
            if research_root is None:
                continue
            research_files = [
                item.model_copy(
                    update={"relative_path": _strip_research_prefix(item.relative_path)}
                )
                for item in copied_files
                if PurePosixPath(item.relative_path).parts[0] == ".research"
            ]
            roots.append(
                self._root_model(
                    checkpoint_id,
                    kind="project_research",
                    identity=project.project_id,
                    live_path=research_root,
                    archive_path=(archive_root / ".research").as_posix(),
                    directories=_parent_directories(research_files),
                    files=research_files,
                )
            )
        return (
            tuple(sorted(projects, key=lambda item: item.project_id)),
            tuple(sorted(roots, key=lambda item: item.identity)),
        )

    def _root_model(
        self,
        checkpoint_id: str,
        *,
        kind: Literal["app_data", "project_research"],
        identity: str,
        live_path: Path,
        archive_path: str,
        directories: set[str] | tuple[str, ...],
        files: list[UpdateCheckpointFile],
    ) -> UpdateCheckpointRoot:
        suffix = uuid.UUID(checkpoint_id).hex
        quarantine = live_path.parent / f".{live_path.name}.rcp-update-{suffix}"
        partial = live_path.parent / f".{live_path.name}.rcp-update-{suffix}-partial"
        return UpdateCheckpointRoot(
            kind=kind,
            identity=identity,
            live_path=str(live_path),
            quarantine_path=str(quarantine),
            partial_path=str(partial),
            archive_path=archive_path,
            directories=tuple(sorted(set(directories))),
            files=tuple(sorted(files, key=lambda item: item.relative_path)),
        )


def _project_restore_location(
    project: BackupProjectCapture,
) -> tuple[Literal["local_research", "remote_excluded"], Path | None]:
    assert project.recovery is not None
    configuration = project.recovery.configuration
    repositories = {item.alias: item for item in project.recovery.repositories}
    machines = {item.alias: item for item in project.recovery.machines}
    state_repository = repositories[configuration.state_repository]
    machine = machines[state_repository.machine_alias]
    if machine.location == "ssh":
        return "remote_excluded", None
    repository_root = Path(state_repository.resolved_path)
    research_root = repository_root / ".research"
    if project.locator != str(research_root / "manifest.toml"):
        raise UpdateCheckpointRefused(
            "A local project's catalog locator differs from its reviewed central checkout."
        )
    return "local_research", research_root


def _validate_replacement_targets(
    checkpoint: VerifiedUpdateCheckpoint,
    *,
    expected_uid: int,
) -> None:
    for root in checkpoint.roots:
        live = Path(root.live_path)
        parent = live.parent
        _require_directory(live, expected_uid=expected_uid, label="live rollback root")
        _require_directory(parent, expected_uid=expected_uid, label="rollback parent")
        if not os.access(parent, os.W_OK | os.X_OK):
            raise UpdateCheckpointRefused("A rollback parent is not writable by the service.")
        if os.path.lexists(root.quarantine_path) or os.path.lexists(root.partial_path):
            raise UpdateCheckpointRefused(
                "A checkpoint-specific quarantine path already exists and needs inspection."
            )


def _strip_research_prefix(value: str) -> str:
    path = PurePosixPath(value)
    if len(path.parts) < 2 or path.parts[0] != ".research":
        raise ValueError("project research rollback file escaped .research")
    return PurePosixPath(*path.parts[1:]).as_posix()


def _parent_directories(files: list[UpdateCheckpointFile]) -> tuple[str, ...]:
    directories: set[str] = set()
    for item in files:
        parent = PurePosixPath(item.relative_path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return tuple(sorted(directories))


def read_verified_update_checkpoint(
    path: Path,
    *,
    expected_uid: int,
    expected_sha256: str | None = None,
) -> VerifiedUpdateCheckpoint:
    return _read_verified_update_checkpoint(
        path,
        expected_uid=expected_uid,
        expected_sha256=expected_sha256,
    )[0]


def _read_verified_update_checkpoint(
    path: Path,
    *,
    expected_uid: int,
    expected_sha256: str | None = None,
) -> tuple[VerifiedUpdateCheckpoint, str]:
    payload = _read_private_file(path, expected_uid=expected_uid, expected_mode=_RECEIPT_MODE)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise UpdateCheckpointRefused("The update checkpoint manifest digest changed.")
    try:
        checkpoint = VerifiedUpdateCheckpoint.model_validate_json(payload)
    except ValueError as exc:
        raise UpdateCheckpointRefused("The update checkpoint manifest is invalid.") from exc
    if checkpoint.manifest_path != str(path):
        raise UpdateCheckpointRefused("The update checkpoint path and payload disagree.")
    operation_root = Path(checkpoint.operation_root)
    try:
        root_metadata = operation_root.lstat()
    except OSError as exc:
        raise UpdateCheckpointRefused(
            "The update checkpoint operation root is unavailable."
        ) from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or operation_root.is_symlink()
        or root_metadata.st_uid != expected_uid
        or stat.S_IMODE(root_metadata.st_mode) != _DIRECTORY_MODE
    ):
        raise UpdateCheckpointRefused("The update checkpoint operation root is unsafe.")
    _verify_checkpoint_payload(checkpoint, expected_uid=expected_uid)
    return checkpoint, digest


def restore_update_checkpoint(
    checkpoint_path: Path,
    *,
    expected_uid: int,
    expected_sha256: str | None = None,
    after_phase: Callable[[str], None] | None = None,
) -> RollbackJournal:
    checkpoint, checkpoint_sha256 = _read_verified_update_checkpoint(
        checkpoint_path,
        expected_uid=expected_uid,
        expected_sha256=expected_sha256,
    )
    journal_path = Path(checkpoint.operation_root) / "rollback-journal.json"
    if journal_path.exists():
        journal = read_rollback_journal(journal_path, expected_uid=expected_uid)
        if (
            journal.checkpoint_id != checkpoint.checkpoint_id
            or journal.checkpoint_path != str(checkpoint_path)
            or journal.checkpoint_sha256 != checkpoint_sha256
            or journal.previous_release_path != checkpoint.previous_release_path
            or journal.quarantine_paths != tuple(root.quarantine_path for root in checkpoint.roots)
            or journal.partial_paths != tuple(root.partial_path for root in checkpoint.roots)
        ):
            raise UpdateCheckpointRefused("The rollback journal names another checkpoint.")
    else:
        journal = RollbackJournal(
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            previous_release_path=checkpoint.previous_release_path,
            phase="prepared",
            quarantine_paths=tuple(root.quarantine_path for root in checkpoint.roots),
            partial_paths=tuple(root.partial_path for root in checkpoint.roots),
            updated_at=datetime.now(UTC),
        )
        _write_journal(journal_path, journal)
        _after_phase(after_phase, "prepared")

    phases = ("prepared", "quarantined", "restored", "verified", "complete")
    phase_index = phases.index(journal.phase)
    if phase_index < 1:
        _quarantine_live_roots(checkpoint, expected_uid=expected_uid)
        journal = _advance_journal(journal_path, journal, "quarantined")
        _after_phase(after_phase, "quarantined")
        phase_index = 1
    if phase_index < 2:
        _restore_live_roots(checkpoint, expected_uid=expected_uid)
        journal = _advance_journal(journal_path, journal, "restored")
        _after_phase(after_phase, "restored")
        phase_index = 2
    if phase_index < 3:
        _verify_live_roots(checkpoint, expected_uid=expected_uid)
        journal = _advance_journal(journal_path, journal, "verified")
        _after_phase(after_phase, "verified")
        phase_index = 3
    if phase_index < 4:
        _verify_live_roots(checkpoint, expected_uid=expected_uid)
        journal = _advance_journal(journal_path, journal, "complete")
        _after_phase(after_phase, "complete")
    else:
        _verify_live_roots(checkpoint, expected_uid=expected_uid)
    return journal


def read_rollback_journal(path: Path, *, expected_uid: int) -> RollbackJournal:
    payload = _read_private_file(path, expected_uid=expected_uid, expected_mode=_RECEIPT_MODE)
    try:
        return RollbackJournal.model_validate_json(payload)
    except ValueError as exc:
        raise UpdateCheckpointRefused("The rollback journal is invalid.") from exc


def unfinished_rollback_journals(update_root: Path, *, expected_uid: int) -> tuple[Path, ...]:
    pending: list[Path] = []
    try:
        operations = tuple(update_root.iterdir())
    except OSError as exc:
        raise UpdateCheckpointRefused("The update checkpoint root cannot be inspected.") from exc
    for operation in operations:
        if _CHECKPOINT_ROOT_NAME.fullmatch(operation.name) is None:
            continue
        journal_path = operation / "rollback-journal.json"
        if not journal_path.exists():
            continue
        if read_rollback_journal(journal_path, expected_uid=expected_uid).phase != "complete":
            pending.append(journal_path)
    return tuple(sorted(pending))


def _advance_journal(path: Path, journal: RollbackJournal, phase: str) -> RollbackJournal:
    advanced = journal.model_copy(update={"phase": phase, "updated_at": datetime.now(UTC)})
    _write_journal(path, advanced)
    return advanced


def _write_journal(path: Path, journal: RollbackJournal) -> None:
    payload = _model_bytes(journal)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".rollback-journal-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, _RECEIPT_MODE)
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


def _quarantine_live_roots(
    checkpoint: VerifiedUpdateCheckpoint,
    *,
    expected_uid: int,
) -> None:
    for root in checkpoint.roots:
        live = Path(root.live_path)
        quarantine = Path(root.quarantine_path)
        partial = Path(root.partial_path)
        if os.path.lexists(partial):
            raise UpdateCheckpointRefused("A prior partial rollback root needs inspection.")
        if os.path.lexists(quarantine):
            _require_directory(
                quarantine,
                expected_uid=expected_uid,
                label="quarantined rollback root",
            )
            if os.path.lexists(live):
                raise UpdateCheckpointRefused(
                    "A rollback root exists beside its quarantine before restoration."
                )
            continue
        if not os.path.lexists(live):
            raise UpdateCheckpointRefused("A live rollback root disappeared before quarantine.")
        _require_directory(live, expected_uid=expected_uid, label="live rollback root")
        os.replace(live, quarantine)
        _fsync_directory(live.parent)


def _restore_live_roots(
    checkpoint: VerifiedUpdateCheckpoint,
    *,
    expected_uid: int,
) -> None:
    for root in checkpoint.roots:
        live = Path(root.live_path)
        if os.path.lexists(live):
            if _root_matches(checkpoint, root, live, expected_uid=expected_uid):
                continue
            partial = Path(root.partial_path)
            if os.path.lexists(partial):
                raise UpdateCheckpointRefused("A partial rollback root already needs inspection.")
            _require_directory(live, expected_uid=expected_uid, label="partial rollback root")
            os.replace(live, partial)
            _fsync_directory(live.parent)
        temporary = live.parent / f".{live.name}.rcp-restore-{uuid.uuid4().hex}"
        try:
            _restore_root(checkpoint, root, temporary)
            if not _root_matches(
                checkpoint,
                root,
                temporary,
                expected_uid=expected_uid,
            ):
                raise UpdateCheckpointRefused("A rebuilt rollback root failed byte verification.")
            os.replace(temporary, live)
            _fsync_directory(live.parent)
        finally:
            if temporary.exists():
                _remove_tree(temporary)


def _verify_live_roots(
    checkpoint: VerifiedUpdateCheckpoint,
    *,
    expected_uid: int,
) -> None:
    for root in checkpoint.roots:
        if not _root_matches(
            checkpoint,
            root,
            Path(root.live_path),
            expected_uid=expected_uid,
        ):
            raise UpdateCheckpointRefused(
                "The restored rollback state differs from its checkpoint."
            )


def _verify_checkpoint_in_temporary_root(
    checkpoint: VerifiedUpdateCheckpoint,
    *,
    expected_uid: int,
) -> None:
    operation_root = Path(checkpoint.operation_root)
    verification_root = Path(tempfile.mkdtemp(prefix="verify-", dir=operation_root))
    try:
        for root in checkpoint.roots:
            destination = verification_root / "roots" / root.kind / root.identity
            destination.parent.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
            _restore_root(checkpoint, root, destination)
            if not _root_matches(
                checkpoint,
                root,
                destination,
                expected_uid=expected_uid,
            ):
                raise UpdateCheckpointRefused(
                    "The checkpoint could not reproduce one replacement root in verification."
                )
        for project in checkpoint.projects:
            if project.archive_path is None:
                continue
            destination = verification_root / "projects" / project.project_id
            destination.mkdir(mode=_DIRECTORY_MODE, parents=True)
            archive = operation_root.joinpath(*PurePosixPath(project.archive_path).parts)
            for item in project.files:
                source = archive.joinpath(*PurePosixPath(item.relative_path).parts)
                target = destination.joinpath(*PurePosixPath(item.relative_path).parts)
                _copy_checkpoint_file(source, target, mode=item.mode)
            _set_private_directory_modes(destination)
            _verify_declared_tree(
                destination,
                project.files,
                allow_directories=True,
                expected_uid=expected_uid,
            )
    finally:
        _remove_tree(verification_root)


def _verify_checkpoint_payload(
    checkpoint: VerifiedUpdateCheckpoint,
    *,
    expected_uid: int,
) -> None:
    operation_root = Path(checkpoint.operation_root)
    for relative, expected in (
        (checkpoint.sqlite_receipt_archive_path, checkpoint.sqlite_receipt_sha256),
        (checkpoint.project_receipt_archive_path, checkpoint.project_receipt_sha256),
        (checkpoint.candidate_receipt_archive_path, checkpoint.candidate_receipt_sha256),
    ):
        path = operation_root.joinpath(*PurePosixPath(relative).parts)
        payload = _read_private_file(
            path,
            expected_uid=expected_uid,
            expected_mode=_PAYLOAD_FILE_MODE,
        )
        if hashlib.sha256(payload).hexdigest() != expected:
            raise UpdateCheckpointRefused("A checkpoint proof receipt changed.")
    for root in checkpoint.roots:
        archive = operation_root.joinpath(*PurePosixPath(root.archive_path).parts)
        _verify_declared_tree(
            archive,
            root.files,
            directories=root.directories,
            expected_uid=expected_uid,
            uniform_file_mode=_PAYLOAD_FILE_MODE,
        )
        if root.kind == "app_data":
            _verify_imported_source_owners(checkpoint, archive)
    for project in checkpoint.projects:
        if project.archive_path is None:
            continue
        archive = operation_root.joinpath(*PurePosixPath(project.archive_path).parts)
        _verify_declared_tree(
            archive,
            project.files,
            allow_directories=True,
            expected_uid=expected_uid,
            uniform_file_mode=_PAYLOAD_FILE_MODE,
        )


def _set_private_directory_modes(root: Path) -> None:
    for current, directory_names, _file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        metadata = current_path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or current_path.is_symlink():
            raise UpdateCheckpointRefused("A rebuilt checkpoint directory is unsafe.")
        current_path.chmod(_DIRECTORY_MODE)
        for name in directory_names:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
                raise UpdateCheckpointRefused("A rebuilt checkpoint directory is unsafe.")


def _restore_root(
    checkpoint: VerifiedUpdateCheckpoint,
    root: UpdateCheckpointRoot,
    destination: Path,
) -> None:
    archive = Path(checkpoint.operation_root).joinpath(*PurePosixPath(root.archive_path).parts)
    destination.mkdir(mode=_DIRECTORY_MODE)
    destination.chmod(_DIRECTORY_MODE)
    for relative in root.directories:
        directory = destination.joinpath(*PurePosixPath(relative).parts)
        directory.mkdir(
            mode=_DIRECTORY_MODE,
            parents=True,
            exist_ok=True,
        )
        directory.chmod(_DIRECTORY_MODE)
    for item in root.files:
        source = archive.joinpath(*PurePosixPath(item.relative_path).parts)
        target = destination.joinpath(*PurePosixPath(item.relative_path).parts)
        _copy_checkpoint_file(source, target, mode=item.mode)
    _fsync_tree(destination)


def _root_matches(
    checkpoint: VerifiedUpdateCheckpoint,
    root: UpdateCheckpointRoot,
    destination: Path,
    *,
    expected_uid: int,
) -> bool:
    try:
        _verify_declared_tree(
            destination,
            root.files,
            directories=root.directories,
            expected_uid=expected_uid,
        )
        if root.kind == "app_data":
            _verify_imported_source_owners(checkpoint, destination)
    except (OSError, UpdateCheckpointRefused):
        return False
    return True


def _verify_imported_source_owners(
    checkpoint: VerifiedUpdateCheckpoint,
    data_dir: Path,
) -> None:
    expected_present = tuple(
        sorted(capture.project_id for capture in checkpoint.imported_sources if capture.present)
    )
    try:
        if ImportedProviderSourceStore.project_ids(data_dir) != expected_present:
            raise ValueError("imported provider-source owner inventory changed")
        for capture in checkpoint.imported_sources:
            expected = ImportedProviderSourceInventory.model_validate(
                capture.inventory.model_dump()
            )
            if ImportedProviderSourceStore(data_dir, capture.project_id).inventory() != expected:
                raise ValueError("imported provider-source inventory changed")
    except (OSError, ValueError) as exc:
        raise UpdateCheckpointRefused(
            "The checkpoint imported provider-source state is invalid."
        ) from exc


def _verify_declared_tree(
    root: Path,
    files: tuple[UpdateCheckpointFile, ...] | list[UpdateCheckpointFile],
    *,
    directories: tuple[str, ...] | None = None,
    allow_directories: bool = False,
    expected_uid: int | None = None,
    uniform_file_mode: int | None = None,
) -> None:
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise UpdateCheckpointRefused("A checkpoint tree root is unavailable or unsafe.") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or root.is_symlink():
        raise UpdateCheckpointRefused("A checkpoint tree root is unavailable or unsafe.")
    if expected_uid is not None and (
        root_metadata.st_uid != expected_uid
        or stat.S_IMODE(root_metadata.st_mode) != _DIRECTORY_MODE
    ):
        raise UpdateCheckpointRefused("A restored tree root has unsafe ownership or mode.")
    expected_files = {item.relative_path: item for item in files}
    expected_directories = set(directories or ())
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
                raise UpdateCheckpointRefused("A checkpoint tree contains an unsafe directory.")
            if expected_uid is not None and (
                metadata.st_uid != expected_uid or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
            ):
                raise UpdateCheckpointRefused(
                    "A restored tree directory has unsafe ownership or mode."
                )
            actual_directories.add(path.relative_to(root).as_posix())
        for name in file_names:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                raise UpdateCheckpointRefused("A checkpoint tree contains an unsafe file.")
            relative = path.relative_to(root).as_posix()
            actual_files.add(relative)
            expected = expected_files.get(relative)
            if expected is None:
                raise UpdateCheckpointRefused("A checkpoint tree contains an unknown file.")
            expected_mode = expected.mode if uniform_file_mode is None else uniform_file_mode
            if expected_uid is not None and (
                metadata.st_uid != expected_uid or stat.S_IMODE(metadata.st_mode) != expected_mode
            ):
                raise UpdateCheckpointRefused("A restored tree file has unsafe ownership or mode.")
            digest, size = _file_sha256(path)
            if digest != expected.sha256 or size != expected.size_bytes:
                raise UpdateCheckpointRefused("A checkpoint tree file changed.")
    if actual_files != set(expected_files):
        raise UpdateCheckpointRefused("A checkpoint tree is missing a declared file.")
    if not allow_directories and actual_directories != expected_directories:
        raise UpdateCheckpointRefused("A checkpoint tree directory inventory changed.")


def _snapshot_tree(
    source: Path,
    destination: Path,
    *,
    relative_prefix: PurePosixPath,
) -> tuple[tuple[str, ...], list[UpdateCheckpointFile]]:
    initial = _tree_inventory(source)
    destination.mkdir(mode=_DIRECTORY_MODE, parents=True)
    for relative in initial[0]:
        destination.joinpath(*PurePosixPath(relative).parts).mkdir(
            mode=_DIRECTORY_MODE,
            parents=True,
            exist_ok=True,
        )
    files: list[UpdateCheckpointFile] = []
    for relative, signature in initial[1]:
        copied = _copy_stable_file(
            source.joinpath(*PurePosixPath(relative).parts),
            destination.joinpath(*PurePosixPath(relative).parts),
            relative_path=(relative_prefix / relative).as_posix(),
            restore_mode=signature[-1],
        )
        files.append(copied)
    if _tree_inventory(source) != initial:
        raise UpdateCheckpointRefused("A recovery-critical tree changed during checkpointing.")
    directories: set[str] = set()
    for path in (relative_prefix, *(relative_prefix / path for path in initial[0])):
        current = path
        while current != PurePosixPath("."):
            directories.add(current.as_posix())
            current = current.parent
    return tuple(sorted(directories)), files


def _tree_inventory(
    root: Path,
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    if not root.is_dir() or root.is_symlink():
        raise UpdateCheckpointRefused("A recovery-critical root is not an ordinary directory.")
    directories: list[str] = []
    files: list[tuple[str, tuple[int, ...]]] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
                raise UpdateCheckpointRefused("A recovery-critical tree contains a link.")
            directories.append(path.relative_to(root).as_posix())
        for name in file_names:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                raise UpdateCheckpointRefused("A recovery-critical tree contains a special file.")
            files.append(
                (
                    path.relative_to(root).as_posix(),
                    (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                        stat.S_IMODE(metadata.st_mode),
                    ),
                )
            )
        if len(directories) + len(files) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise UpdateCheckpointRefused("A recovery-critical tree exceeds its inventory bound.")
    return tuple(directories), tuple(files)


def _copy_declared_file(
    source: Path,
    destination: Path,
    *,
    relative_path: str,
    expected_sha256: str,
    expected_size: int | None,
    restore_mode: int,
) -> UpdateCheckpointFile:
    copied = _copy_stable_file(
        source,
        destination,
        relative_path=relative_path,
        restore_mode=restore_mode,
    )
    if copied.sha256 != expected_sha256 or (
        expected_size is not None and copied.size_bytes != expected_size
    ):
        raise UpdateCheckpointRefused("A declared capture file differs from its receipt.")
    return copied


def _copy_stable_file(
    source: Path,
    destination: Path,
    *,
    relative_path: str,
    restore_mode: int,
) -> UpdateCheckpointFile:
    try:
        initial = source.lstat()
    except OSError as exc:
        raise UpdateCheckpointRefused("A checkpoint source file is unavailable.") from exc
    if not stat.S_ISREG(initial.st_mode) or source.is_symlink():
        raise UpdateCheckpointRefused("A checkpoint source is not an ordinary file.")
    destination.parent.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(source_descriptor)
        if (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino):
            raise UpdateCheckpointRefused("A checkpoint source changed while it was opened.")
        destination_descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            _PAYLOAD_FILE_MODE,
        )
        while True:
            chunk = os.read(source_descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            _write_all(destination_descriptor, chunk)
        os.fchmod(destination_descriptor, _PAYLOAD_FILE_MODE)
        os.fsync(destination_descriptor)
        final = os.fstat(source_descriptor)
        path_final = source.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, name) != getattr(final, name) for name in stable) or any(
            getattr(final, name) != getattr(path_final, name) for name in stable
        ):
            raise UpdateCheckpointRefused("A checkpoint source changed during copying.")
        if size != final.st_size:
            raise UpdateCheckpointRefused("A checkpoint file copy is incomplete.")
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
    _fsync_directory(destination.parent)
    return UpdateCheckpointFile(
        relative_path=relative_path,
        sha256=digest.hexdigest(),
        size_bytes=size,
        mode=restore_mode,
    )


def _copy_checkpoint_file(source: Path, destination: Path, *, mode: int) -> None:
    destination.parent.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor = os.open(
        destination,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        while True:
            chunk = os.read(source_descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            _write_all(destination_descriptor, chunk)
        os.fchmod(destination_descriptor, mode)
        os.fsync(destination_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    _fsync_directory(destination.parent)


def _write_new_model(path: Path, model: BaseModel) -> None:
    payload = _model_bytes(model)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        _RECEIPT_MODE,
    )
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, _RECEIPT_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_new_payload(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        _PAYLOAD_FILE_MODE,
    )
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, _PAYLOAD_FILE_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _model_bytes(model: BaseModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_private_file(path: Path, *, expected_uid: int, expected_mode: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise UpdateCheckpointRefused("A private update receipt is unavailable.") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != expected_uid
            or stat.S_IMODE(metadata.st_mode) != expected_mode
            or metadata.st_size > _MAX_JSON_BYTES
        ):
            raise UpdateCheckpointRefused("A private update receipt has unsafe ownership or mode.")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(BACKUP_COPY_BUFFER_BYTES, remaining))
            if not chunk:
                raise UpdateCheckpointRefused("A private update receipt is incomplete.")
            chunks.append(chunk)
            remaining -= len(chunk)
        final = os.fstat(descriptor)
        path_final = path.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(metadata, name) != getattr(final, name) for name in stable) or any(
            getattr(final, name) != getattr(path_final, name) for name in stable
        ):
            raise UpdateCheckpointRefused("A private update receipt changed while reading.")
        return b"".join(chunks)
    except OSError as exc:
        raise UpdateCheckpointRefused("A private update receipt cannot be read.") from exc
    finally:
        os.close(descriptor)


def _require_directory(path: Path, *, expected_uid: int | None, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise UpdateCheckpointRefused(f"The {label} is unavailable.") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or (expected_uid is not None and metadata.st_uid != expected_uid)
    ):
        raise UpdateCheckpointRefused(f"The {label} is not one safe owned directory.")


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise UpdateCheckpointRefused("A checkpoint file is not regular.")
        while True:
            chunk = os.read(descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        final = os.fstat(descriptor)
        path_final = path.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, name) != getattr(final, name) for name in stable) or any(
            getattr(final, name) != getattr(path_final, name) for name in stable
        ):
            raise UpdateCheckpointRefused("A checkpoint file changed while hashing.")
        if size != final.st_size:
            raise UpdateCheckpointRefused("A checkpoint file hash is incomplete.")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short checkpoint write")
        remaining = remaining[written:]


def _fsync_tree(root: Path) -> None:
    for current, _directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in files:
            descriptor = os.open(
                current_path / name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(current_path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise UpdateCheckpointRefused("Refusing to remove a non-directory checkpoint temp path.")
    for current, directories, files in os.walk(path, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in files:
            child = current_path / name
            if child.is_symlink() or not child.is_file():
                raise UpdateCheckpointRefused("Checkpoint temp cleanup encountered a special file.")
            child.chmod(0o600)
            child.unlink()
        for name in directories:
            child = current_path / name
            if child.is_symlink() or not child.is_dir():
                raise UpdateCheckpointRefused(
                    "Checkpoint temp cleanup encountered a special directory."
                )
            child.chmod(_DIRECTORY_MODE)
            child.rmdir()
    path.chmod(_DIRECTORY_MODE)
    path.rmdir()


def _after_phase(callback: Callable[[str], None] | None, phase: str) -> None:
    if callback is not None:
        callback(phase)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RCP internal update-checkpoint worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("data_dir", type=Path)
    create.add_argument("update_root", type=Path)
    create.add_argument("previous_release", type=Path)
    create.add_argument("sqlite_receipt", type=Path)
    create.add_argument("sqlite_receipt_sha256")
    create.add_argument("project_receipt", type=Path)
    create.add_argument("project_receipt_sha256")
    create.add_argument("candidate_receipt", type=Path)
    create.add_argument("candidate_receipt_sha256")
    restore = subparsers.add_parser("restore")
    restore.add_argument("checkpoint", type=Path)
    restore.add_argument("checkpoint_sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "create":
            checkpoint = UpdateCheckpointCoordinator(
                data_dir=arguments.data_dir,
                update_root=arguments.update_root,
                previous_release_path=arguments.previous_release,
                expected_uid=os.geteuid(),
            ).create(
                sqlite_receipt_path=arguments.sqlite_receipt,
                sqlite_receipt_sha256=arguments.sqlite_receipt_sha256,
                project_receipt_path=arguments.project_receipt,
                project_receipt_sha256=arguments.project_receipt_sha256,
                candidate_receipt_path=arguments.candidate_receipt,
                candidate_receipt_sha256=arguments.candidate_receipt_sha256,
            )
            print(checkpoint.manifest_path)
            return 0
        journal = restore_update_checkpoint(
            arguments.checkpoint,
            expected_uid=os.geteuid(),
            expected_sha256=arguments.checkpoint_sha256,
        )
        print(journal.phase)
        return 0
    except (OSError, UpdateCheckpointRefused, ValueError) as exc:
        print(str(exc) or "Update checkpoint failed safely.", file=sys.stderr)
        return 1


__all__ = [
    "RollbackJournal",
    "UpdateCheckpointCoordinator",
    "UpdateCheckpointRefused",
    "VerifiedUpdateCheckpoint",
    "read_rollback_journal",
    "read_verified_update_checkpoint",
    "restore_update_checkpoint",
    "unfinished_rollback_journals",
]


if __name__ == "__main__":  # pragma: no cover - exercised by the root coordinator
    raise SystemExit(main())
