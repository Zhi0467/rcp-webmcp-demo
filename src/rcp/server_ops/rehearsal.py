"""Rehearse one built server candidate against a path-safe copy of live state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol

import tomlkit
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator

from rcp.api import create_app
from rcp.background import BackgroundAgentTasks, StartupEffectFence
from rcp.config import AGENT_EXECUTION_PROFILES, Manifest, load_manifest
from rcp.history import HistoryManager
from rcp.limits import (
    BACKUP_COPY_BUFFER_BYTES,
    BACKUP_DIAGNOSTIC_MAX_CHARS,
    SERVER_UPDATE_REHEARSAL_TIMEOUT_SECONDS,
)
from rcp.projects import TEAM_PROJECT_DELETE_UNAVAILABLE_REASON
from rcp.server_ops.backup import BackupRunRefused, discard_backup_capture_root
from rcp.server_ops.backup_capture import (
    BackupCaptureUnavailable,
    BackupSQLiteCaptureReceipt,
    read_backup_sqlite_capture_receipt,
)
from rcp.server_ops.backup_models import BackupManifestConfiguration, BackupProjectCapture
from rcp.server_ops.backup_project_files import (
    BackupProjectFileCaptureCoordinator,
    BackupProjectFileCaptureReceipt,
)
from rcp.server_ops.control import ServerControlBackupCaptureResult, ServerControlClient
from rcp.server_ops.models import redact_server_text
from rcp.server_ops.update import BuiltCandidateReceipt
from rcp.server_runtime import data_dir_identity
from rcp.sources.imported import (
    ImportedProviderSourceInventory,
    ImportedProviderSourceStore,
)

REHEARSAL_OVERLAY_SCHEMA_VERSION = 1
CANDIDATE_MIGRATION_RESULT_SCHEMA_VERSION = 1
CANDIDATE_REHEARSAL_RESULT_SCHEMA_VERSION = 1
VERIFIED_CANDIDATE_RECEIPT_SCHEMA_VERSION = 1

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERIFIED_RECEIPT_NAME = re.compile(r"verified-candidate-([0-9a-f]{40})-([0-9a-f-]{36})\.json")
_REHEARSAL_ROOT_NAME = re.compile(r"rehearsal-([0-9a-f]{40})-([0-9a-f]{32})")
_RECEIPT_MODE = 0o600
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_PATH_COLUMN_NAMES = frozenset(
    {"locator", "state_location", "stage_root", "output_path", "log_path", "cwd"}
)


class CandidateRehearsalRefused(RuntimeError):
    """The candidate or copied-state boundary failed safely before cutover."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class StartupRecoveryReadModel(_StrictModel):
    active_operation_ids: tuple[str, ...]
    stopping_experiment_operation_ids: tuple[str, ...]
    report_episode_ids: tuple[str, ...]
    auto_research_recovery_operation_ids: tuple[str, ...]
    active_watcher_ids: tuple[str, ...]


def _absolute_path(value: str, *, label: str) -> str:
    if (
        not value
        or value != value.strip()
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be one bounded absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be absolute and normalized")
    return value


def _canonical_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase canonical UUID4")
    return value


class RehearsalProjectOverlay(_StrictModel):
    project_id: str
    name: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    capture_status: Literal["captured", "remote_unreachable"]
    overlay_locator: str
    original_locator: str
    original_state_location: str
    original_remote: bool
    original_reachable: bool | None
    original_error_sha256: str | None
    expected_card_sha256: str
    expected_graph_sha256: str | None
    expected_revision: int | None = None

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="project identity")

    @field_validator("overlay_locator", "original_locator")
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        return _absolute_path(value, label=info.field_name.replace("_", " "))

    @field_validator(
        "original_error_sha256",
        "expected_card_sha256",
        "expected_graph_sha256",
    )
    @classmethod
    def validate_projection_digest(cls, value: str | None, info) -> str | None:
        if value is None:
            if info.field_name in {"original_error_sha256", "expected_graph_sha256"}:
                return None
            raise ValueError("project projection digest is required")
        if _SHA256.fullmatch(value) is None:
            raise ValueError("project projection digest must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_capture(self) -> RehearsalProjectOverlay:
        if self.capture_status == "captured" and self.expected_revision is None:
            raise ValueError("captured rehearsal projects require an expected revision")
        if (self.capture_status == "captured") != (self.expected_graph_sha256 is not None):
            raise ValueError("only captured rehearsal projects require an expected graph digest")
        if self.capture_status == "remote_unreachable" and (
            not self.original_remote or self.original_reachable is not False
        ):
            raise ValueError(
                "an unavailable rehearsal project must already be a failed remote projection"
            )
        return self


class RehearsalOverlay(_StrictModel):
    schema_version: Literal[1] = REHEARSAL_OVERLAY_SCHEMA_VERSION
    root: str
    data_dir: str
    database_path: str
    capture_id: str
    sqlite_receipt_sha256: str
    sqlite_snapshot_sha256: str
    project_receipt_sha256: str
    space_id: str
    expected_startup_recovery: StartupRecoveryReadModel
    projects: tuple[RehearsalProjectOverlay, ...]
    transfer_inbox_entries: tuple[str, ...]

    @field_validator("root", "data_dir", "database_path")
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        return _absolute_path(value, label=info.field_name.replace("_", " "))

    @field_validator("capture_id", "space_id")
    @classmethod
    def validate_capture_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="capture identity")

    @field_validator(
        "sqlite_receipt_sha256",
        "sqlite_snapshot_sha256",
        "project_receipt_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("rehearsal capture digests must be lowercase SHA-256")
        return value

    @field_validator("transfer_inbox_entries")
    @classmethod
    def validate_transfer_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _absolute_path(path, label="transfer inbox overlay path")
        if tuple(sorted(set(value))) != value:
            raise ValueError("transfer inbox overlay paths must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_boundary(self) -> RehearsalOverlay:
        root = Path(self.root)
        if (
            Path(self.data_dir).parent != root
            or Path(self.database_path) != Path(self.data_dir) / "rcp.sqlite3"
            or any(
                not Path(project.overlay_locator).is_relative_to(root) for project in self.projects
            )
            or any(not Path(path).is_relative_to(root) for path in self.transfer_inbox_entries)
        ):
            raise ValueError("rehearsal overlay paths escaped their request root")
        project_ids = [project.project_id for project in self.projects]
        if tuple(sorted(project_ids)) != tuple(project_ids) or len(project_ids) != len(
            set(project_ids)
        ):
            raise ValueError("rehearsal projects must be sorted and unique")
        return self


class CandidateMigrationResult(_StrictModel):
    schema_version: Literal[1] = CANDIDATE_MIGRATION_RESULT_SCHEMA_VERSION
    status: Literal["migrated", "failed"]
    diagnostic: Annotated[str, StringConstraints(max_length=BACKUP_DIAGNOSTIC_MAX_CHARS)] | None = (
        None
    )

    @model_validator(mode="after")
    def validate_result(self) -> CandidateMigrationResult:
        if (self.status == "failed") != (self.diagnostic is not None):
            raise ValueError("candidate migration result and diagnostic disagree")
        return self


class CandidateProjectVerification(_StrictModel):
    project_id: str
    status: Literal["verified", "not_replay_verified"]
    revision: int | None
    projection_sha256: str

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        return _canonical_uuid4(value, label="project identity")

    @field_validator("projection_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("project projection digest must be lowercase SHA-256")
        return value


class CandidateRehearsalResult(_StrictModel):
    schema_version: Literal[1] = CANDIDATE_REHEARSAL_RESULT_SCHEMA_VERSION
    status: Literal["verified", "failed"]
    space_id: str | None = None
    space_kind: Literal["team"] | None = None
    startup_recovery: StartupRecoveryReadModel | None = None
    projects: tuple[CandidateProjectVerification, ...] = ()
    reads: tuple[str, ...] = ()
    attempted_effects: tuple[str, ...] = ()
    diagnostic: Annotated[str, StringConstraints(max_length=BACKUP_DIAGNOSTIC_MAX_CHARS)] | None = (
        None
    )

    @model_validator(mode="after")
    def validate_result(self) -> CandidateRehearsalResult:
        if self.status == "verified":
            if (
                self.space_id is None
                or self.space_kind != "team"
                or self.startup_recovery is None
                or self.diagnostic is not None
                or self.attempted_effects
            ):
                raise ValueError("verified rehearsal result is incomplete or crossed its fence")
        elif self.diagnostic is None:
            raise ValueError("failed rehearsal result requires a diagnostic")
        project_ids = [project.project_id for project in self.projects]
        if tuple(sorted(project_ids)) != tuple(project_ids) or len(project_ids) != len(
            set(project_ids)
        ):
            raise ValueError("candidate project results must be sorted and unique")
        return self


class VerifiedCandidateReceipt(_StrictModel):
    schema_version: Literal[1] = VERIFIED_CANDIDATE_RECEIPT_SCHEMA_VERSION
    installation_id: str
    candidate_commit: str
    base_current_commit: str
    base_running_commit: str
    base_instance_id: str
    base_process_pid: int
    release_path: str
    built_receipt_path: str
    built_receipt_sha256: str
    receipt_path: str
    web_build_id: str
    capture_id: str
    sqlite_snapshot_sha256: str
    project_capture_sha256: str
    space_id: str
    projects: tuple[CandidateProjectVerification, ...]
    startup_recovery: StartupRecoveryReadModel
    reads: tuple[str, ...]
    verified_at: datetime

    @field_validator("candidate_commit", "base_current_commit", "base_running_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("verified candidate receipts require full Git commits")
        return value

    @field_validator("release_path", "built_receipt_path", "receipt_path")
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        return _absolute_path(value, label=info.field_name.replace("_", " "))

    @field_validator("built_receipt_sha256", "sqlite_snapshot_sha256", "project_capture_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("verified candidate digests must be lowercase SHA-256")
        return value

    @field_validator("capture_id", "space_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("verified_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("verified candidate time requires a UTC offset")
        return value

    @model_validator(mode="after")
    def validate_relationship(self) -> VerifiedCandidateReceipt:
        if self.base_current_commit != self.base_running_commit:
            raise ValueError("verified candidate base must name one running release")
        if self.candidate_commit == self.base_running_commit or self.base_process_pid <= 0:
            raise ValueError("verified candidate must differ from one live positive-pid base")
        return self


class RehearsalCaptureControl(Protocol):
    def capture_backup_sqlite(self) -> ServerControlBackupCaptureResult: ...


class CandidateProcessRunner(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


class CandidateDatabaseMigrator(Protocol):
    def __call__(self, database_path: Path) -> None: ...


def verified_candidate_receipt_path(commit: str, capture_id: str, update_root: Path) -> Path:
    if _FULL_GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("candidate commit must be one full lowercase Git object id")
    _canonical_uuid4(capture_id, label="candidate capture identity")
    return update_root / f"verified-candidate-{commit}-{capture_id}.json"


class CandidateRehearsalCoordinator:
    """Compose the existing online capture with one fenced candidate child."""

    def __init__(
        self,
        *,
        data_dir: Path,
        update_root: Path,
        built_receipt: BuiltCandidateReceipt,
        built_receipt_sha256: str,
        control: RehearsalCaptureControl | None = None,
        runner: CandidateProcessRunner | None = None,
        candidate_python: Path | None = None,
        capture_result: ServerControlBackupCaptureResult | None = None,
        retain_capture: bool = False,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.update_root = update_root.resolve()
        self.built_receipt = built_receipt
        self.built_receipt_sha256 = built_receipt_sha256
        self.control = control
        self.runner = runner or _run_candidate_process
        self.candidate_python = candidate_python or (
            Path(built_receipt.release_path) / ".venv" / "bin" / "python"
        )
        self.capture_result = capture_result
        self.retain_capture = retain_capture

    def run(self) -> VerifiedCandidateReceipt:
        self._validate_boundary()
        operation_root = self.update_root / (
            f"rehearsal-{self.built_receipt.candidate_commit}-{uuid.uuid4().hex}"
        )
        capture_result = self.capture_result
        try:
            operation_root.mkdir(mode=_DIRECTORY_MODE)
            if capture_result is None:
                control = self.control or ServerControlClient.from_data_dir(
                    self.data_dir,
                    expected_server_uid=os.geteuid(),
                )
                capture_result = control.capture_backup_sqlite()
            self._validate_live_capture(capture_result)
            project_publication = BackupProjectFileCaptureCoordinator(self.data_dir).capture(
                Path(capture_result.receipt_path),
                expected_sha256=capture_result.receipt_sha256,
            )
            sqlite_receipt = read_backup_sqlite_capture_receipt(
                Path(capture_result.receipt_path),
                expected_sha256=capture_result.receipt_sha256,
            )
            if (
                sqlite_receipt.rcp_source_commit != self.built_receipt.base_running_commit
                or sqlite_receipt.space_id != capture_result.space_id
            ):
                raise CandidateRehearsalRefused(
                    "The copied state does not belong to the running release and space."
                )
            if (
                sqlite_receipt.app_data_plan.deferred_entries
                or sqlite_receipt.app_data_plan.unclassified_entries
            ):
                raise CandidateRehearsalRefused(
                    "The live app-data capture has deferred or unknown durable entries. "
                    "Classify them before updating."
                )
            overlay = build_rehearsal_overlay(
                operation_root,
                sqlite_receipt=sqlite_receipt,
                sqlite_receipt_sha256=capture_result.receipt_sha256,
                project_receipt=project_publication.receipt,
                project_receipt_sha256=project_publication.receipt_sha256,
                capture_root=Path(capture_result.receipt_path).parent,
                candidate_migrator=self._migrate_candidate,
            )
            overlay_path = operation_root / "overlay.json"
            _write_private_json(overlay_path, overlay)
            result_path = operation_root / "candidate-result.json"
            completed = self.runner(
                (
                    str(self.candidate_python),
                    "-m",
                    "rcp.server_ops.rehearsal",
                    "--candidate-child",
                    str(overlay_path),
                    str(result_path),
                ),
                cwd=Path(self.built_receipt.release_path),
                environment={
                    "LANG": "C.UTF-8",
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                timeout=SERVER_UPDATE_REHEARSAL_TIMEOUT_SECONDS,
            )
            result = _read_candidate_result(result_path)
            if completed.returncode != 0 or result.status != "verified":
                raise CandidateRehearsalRefused(
                    result.diagnostic
                    or "The candidate process failed copied-state startup verification."
                )
            self._validate_candidate_result(overlay, result)
            assert result.space_id is not None and result.startup_recovery is not None
            receipt_path = verified_candidate_receipt_path(
                self.built_receipt.candidate_commit,
                overlay.capture_id,
                self.update_root,
            )
            receipt = VerifiedCandidateReceipt(
                installation_id=self.built_receipt.installation_id,
                candidate_commit=self.built_receipt.candidate_commit,
                base_current_commit=self.built_receipt.base_current_commit,
                base_running_commit=self.built_receipt.base_running_commit,
                base_instance_id=self.built_receipt.base_instance_id,
                base_process_pid=self.built_receipt.base_process_pid,
                release_path=self.built_receipt.release_path,
                built_receipt_path=self.built_receipt.receipt_path,
                built_receipt_sha256=self.built_receipt_sha256,
                receipt_path=str(receipt_path),
                web_build_id=self.built_receipt.web_build_id,
                capture_id=overlay.capture_id,
                sqlite_snapshot_sha256=overlay.sqlite_snapshot_sha256,
                project_capture_sha256=overlay.project_receipt_sha256,
                space_id=result.space_id,
                projects=result.projects,
                startup_recovery=result.startup_recovery,
                reads=result.reads,
                verified_at=datetime.now(UTC),
            )
            _publish_private_json(receipt_path, receipt)
            published = read_verified_candidate_receipt(
                receipt_path,
                expected_uid=os.geteuid(),
            )
            self._validate_existing(published)
            # Publish the durable proof before deleting either evidence root. If
            # publication/readback fails, both the capture and overlay remain.
            # If cleanup itself fails, the receipt plus the remaining exact root
            # make that failure explicit to the next maintenance inspection.
            if not self.retain_capture:
                discard_backup_capture_root(
                    Path(capture_result.receipt_path).parent,
                    data_dir=self.data_dir,
                    capture_id=capture_result.capture_id,
                )
            _discard_operation_root(operation_root, update_root=self.update_root)
            return published
        except CandidateRehearsalRefused:
            raise
        except (BackupCaptureUnavailable, BackupRunRefused, OSError, ValueError) as exc:
            raise CandidateRehearsalRefused(
                "Candidate rehearsal could not prove its copied-state boundary. "
                "The old release is still serving; inspect the retained rehearsal and capture."
            ) from exc

    def _migrate_candidate(self, database_path: Path) -> None:
        operation_root = database_path.parents[2]
        if database_path != operation_root / "overlay" / "data" / "rcp.sqlite3":
            raise CandidateRehearsalRefused(
                "The candidate migration database escaped its rehearsal operation."
            )
        for attempt in (1, 2):
            result_path = operation_root / f"candidate-migration-{attempt}.json"
            completed = self.runner(
                (
                    str(self.candidate_python),
                    "-m",
                    "rcp.server_ops.rehearsal",
                    "--candidate-migrate",
                    str(database_path),
                    str(result_path),
                ),
                cwd=Path(self.built_receipt.release_path),
                environment={
                    "LANG": "C.UTF-8",
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                timeout=SERVER_UPDATE_REHEARSAL_TIMEOUT_SECONDS,
            )
            result = _read_candidate_migration_result(result_path)
            if completed.returncode != 0 or result.status != "migrated":
                raise CandidateRehearsalRefused(
                    result.diagnostic or "The candidate could not migrate the copied database."
                )
            result_path.unlink()

    def _validate_candidate_result(
        self,
        overlay: RehearsalOverlay,
        result: CandidateRehearsalResult,
    ) -> None:
        if (
            result.space_id != overlay.space_id
            or result.startup_recovery != overlay.expected_startup_recovery
            or result.reads != _expected_candidate_reads(overlay.projects)
            or result.attempted_effects
        ):
            raise CandidateRehearsalRefused(
                "The candidate changed copied space, recovery, read, or effect-fence semantics."
            )
        expected = {project.project_id: project for project in overlay.projects}
        observed = {project.project_id: project for project in result.projects}
        if set(observed) != set(expected):
            raise CandidateRehearsalRefused(
                "The candidate result omitted or substituted a captured project."
            )
        for project_id, overlay_project in expected.items():
            candidate_project = observed[project_id]
            if overlay_project.capture_status == "captured":
                if (
                    candidate_project.status != "verified"
                    or candidate_project.revision != overlay_project.expected_revision
                    or candidate_project.projection_sha256 != overlay_project.expected_graph_sha256
                ):
                    raise CandidateRehearsalRefused(
                        f"The candidate changed replay semantics for project {project_id}."
                    )
            elif (
                candidate_project.status != "not_replay_verified"
                or candidate_project.revision is not None
                or candidate_project.projection_sha256 != overlay_project.expected_card_sha256
            ):
                raise CandidateRehearsalRefused(
                    f"The candidate changed unavailable semantics for project {project_id}."
                )

    def _validate_boundary(self) -> None:
        for path, label in (
            (self.data_dir, "live data directory"),
            (self.update_root, "update checkpoint root"),
        ):
            _require_private_directory(path, label=label)
        if (
            _SHA256.fullmatch(self.built_receipt_sha256) is None
            or Path(self.built_receipt.receipt_path).parent != self.update_root
            or Path(self.built_receipt.release_path).name != self.built_receipt.candidate_commit
        ):
            raise CandidateRehearsalRefused(
                "The built-candidate receipt is not bound to this update root and release."
            )
        built_bytes = _read_private_file(
            Path(self.built_receipt.receipt_path),
            expected_uid=os.geteuid(),
            expected_mode=_RECEIPT_MODE,
        )
        try:
            observed = BuiltCandidateReceipt.model_validate_json(built_bytes)
        except ValueError as exc:
            raise CandidateRehearsalRefused(
                "The built-candidate receipt cannot be decoded by the candidate."
            ) from exc
        if (
            observed != self.built_receipt
            or hashlib.sha256(built_bytes).hexdigest() != self.built_receipt_sha256
        ):
            raise CandidateRehearsalRefused(
                "The built-candidate receipt bytes changed before rehearsal."
            )

    def _validate_live_capture(self, result: ServerControlBackupCaptureResult) -> None:
        if (
            result.instance_id != self.built_receipt.base_instance_id
            or result.pid != self.built_receipt.base_process_pid
            or result.data_dir_id != data_dir_identity(self.data_dir)
            or result.receipt_path
            != str(
                self.data_dir / "run-stage" / f"backup-{result.capture_id}" / "sqlite-capture.json"
            )
        ):
            raise CandidateRehearsalRefused(
                "The copied state came from a different process or data boundary."
            )

    def _validate_existing(self, receipt: VerifiedCandidateReceipt) -> None:
        if (
            receipt.installation_id != self.built_receipt.installation_id
            or receipt.candidate_commit != self.built_receipt.candidate_commit
            or receipt.base_current_commit != self.built_receipt.base_current_commit
            or receipt.base_running_commit != self.built_receipt.base_running_commit
            or receipt.base_instance_id != self.built_receipt.base_instance_id
            or receipt.base_process_pid != self.built_receipt.base_process_pid
            or receipt.release_path != self.built_receipt.release_path
            or receipt.built_receipt_path != self.built_receipt.receipt_path
            or receipt.built_receipt_sha256 != self.built_receipt_sha256
            or receipt.web_build_id != self.built_receipt.web_build_id
            or Path(receipt.receipt_path)
            != verified_candidate_receipt_path(
                receipt.candidate_commit,
                receipt.capture_id,
                self.update_root,
            )
        ):
            raise CandidateRehearsalRefused(
                "The existing verified-candidate receipt belongs to another live base or build."
            )


def build_rehearsal_overlay(
    operation_root: Path,
    *,
    sqlite_receipt: BackupSQLiteCaptureReceipt,
    sqlite_receipt_sha256: str,
    project_receipt: BackupProjectFileCaptureReceipt,
    project_receipt_sha256: str,
    capture_root: Path,
    candidate_migrator: CandidateDatabaseMigrator | None = None,
) -> RehearsalOverlay:
    root = operation_root.resolve() / "overlay"
    data_dir = root / "data"
    projects_root = root / "projects"
    absent_root = root / "known-absent"
    for path in (root, data_dir, projects_root, absent_root):
        path.mkdir(mode=_DIRECTORY_MODE)
    database_path = data_dir / "rcp.sqlite3"
    _copy_verified_file(
        Path(sqlite_receipt.snapshot_path),
        database_path,
        expected_sha256=sqlite_receipt.sqlite_snapshot.sha256,
        expected_size=sqlite_receipt.sqlite_snapshot.size_bytes,
    )
    if (
        project_receipt.capture_id != sqlite_receipt.capture_id
        or project_receipt.sqlite_receipt_sha256 != sqlite_receipt_sha256
        or project_receipt.sqlite_snapshot_sha256 != sqlite_receipt.sqlite_snapshot.sha256
    ):
        raise CandidateRehearsalRefused(
            "The project files and SQLite snapshot do not share one capture boundary."
        )

    for capture in project_receipt.imported_sources:
        if not capture.present:
            continue
        source_root = capture_root / "project-sources" / capture.project_id / "provider-history"
        expected = ImportedProviderSourceInventory.model_validate(capture.inventory.model_dump())
        try:
            published = ImportedProviderSourceStore(
                data_dir,
                capture.project_id,
            ).publish_snapshot(source_root, expected)
        except (OSError, ValueError) as exc:
            raise CandidateRehearsalRefused(
                "The imported provider-source snapshot failed rehearsal publication."
            ) from exc
        if published != expected:
            raise CandidateRehearsalRefused(
                "The imported provider-source rehearsal readback differs."
            )

    # The running release owns the coordinator, while the candidate gets only
    # this disposable migration phase. Run it before any schema/path inventory
    # so candidate-added path owners cannot appear after containment validation.
    if candidate_migrator is None:
        from rcp.storage import AppStore

        AppStore(database_path)
    else:
        candidate_migrator(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT project_id, home_space_id, locator, name, state_location, "
            "state_remote, last_opened_at, revision, primary_question, attention_count, "
            "last_refresh_at, reachable, error FROM projects ORDER BY project_id"
        ).fetchall()
        captures = {project.project_id: project for project in project_receipt.projects}
        if set(captures) != {str(row["project_id"]) for row in rows}:
            raise CandidateRehearsalRefused(
                "The project capture does not inventory every copied database project."
            )
        projects: list[RehearsalProjectOverlay] = []
        connection.execute("BEGIN IMMEDIATE")
        for row in rows:
            project_id = str(row["project_id"])
            capture = captures[project_id]
            project_root = projects_root / project_id
            project_root.mkdir(mode=_DIRECTORY_MODE)
            project = _prepare_overlay_project(
                row,
                capture,
                project_root=project_root,
                capture_root=capture_root,
            )
            projects.append(project)
            state_location = (
                row["state_location"]
                if bool(row["state_remote"])
                else str(Path(project.overlay_locator).parent)
            )
            connection.execute(
                "UPDATE projects SET locator = ?, state_location = ? WHERE project_id = ?",
                (project.overlay_locator, state_location, project_id),
            )
        _rebind_local_stage_paths(connection, absent_root)
        transfer_paths = _transfer_inbox_overlay_paths(connection, data_dir)
        _validate_path_column_inventory(connection)
        connection.commit()
        _validate_rebound_paths(connection, root=root, projects=projects)
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    _fsync_file(database_path)
    _fsync_directory(data_dir)
    expected_startup_recovery = _expected_startup_recovery(database_path)
    overlay = RehearsalOverlay(
        root=str(root),
        data_dir=str(data_dir),
        database_path=str(database_path),
        capture_id=sqlite_receipt.capture_id,
        sqlite_receipt_sha256=sqlite_receipt_sha256,
        sqlite_snapshot_sha256=sqlite_receipt.sqlite_snapshot.sha256,
        project_receipt_sha256=project_receipt_sha256,
        space_id=sqlite_receipt.space_id,
        expected_startup_recovery=expected_startup_recovery,
        projects=tuple(projects),
        transfer_inbox_entries=transfer_paths,
    )
    for path in overlay.transfer_inbox_entries:
        if Path(path).exists() or Path(path).is_symlink():
            raise CandidateRehearsalRefused(
                "A copied incoming transfer resolved to an existing rehearsal path."
            )
    return overlay


def _prepare_overlay_project(
    row: sqlite3.Row,
    capture: BackupProjectCapture,
    *,
    project_root: Path,
    capture_root: Path,
) -> RehearsalProjectOverlay:
    if capture.recovery is None:
        raise CandidateRehearsalRefused(
            "A copied project has no typed configuration for path-safe rehearsal."
        )
    if capture.status == "uncaptured":
        if not (
            capture.unavailable_kind == "remote_unreachable"
            and bool(row["state_remote"])
            and row["reachable"] == 0
            and row["error"]
            and _configuration_state_is_remote(capture.recovery.configuration)
        ):
            raise CandidateRehearsalRefused(
                "A project capture failed without an already-unreachable SSH proof."
            )
        capture_status: Literal["captured", "remote_unreachable"] = "remote_unreachable"
        expected_revision = None
        expected_graph_sha256 = None
    else:
        capture_status = "captured"
        if capture.main_head is None:
            raise CandidateRehearsalRefused("A captured project lost its canonical head.")
        expected_revision = capture.main_head.revision

    configuration = capture.recovery.configuration
    repository_roots = {
        repository.alias: project_root / "repositories" / repository.alias
        for repository in configuration.repositories
    }
    for repository_root in repository_roots.values():
        _mkdir_private_parents(repository_root)
    state_root = repository_roots[configuration.state_repository]
    if capture.status == "captured":
        for entry in capture.files:
            source = capture_root.joinpath(*PurePosixPath(entry.archive_path).parts)
            destination = state_root.joinpath(*PurePosixPath(entry.source_relative_path).parts)
            if entry.source_relative_path == ".research/manifest.toml":
                _read_verified_bytes(
                    source,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size_bytes,
                )
            else:
                _copy_verified_file(
                    source,
                    destination,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size_bytes,
                )
    manifest_path = state_root / ".research" / "manifest.toml"
    _mkdir_private_parents(manifest_path.parent)
    _write_private_bytes(
        manifest_path,
        _render_overlay_manifest(configuration, repository_roots, project_root).encode("utf-8"),
    )
    if capture.status == "captured":
        expected_graph = HistoryManager(
            load_manifest(manifest_path),
            expected_space_id=str(row["home_space_id"]),
        ).state()
        if expected_graph.revision != expected_revision:
            raise CandidateRehearsalRefused(
                "The current release could not replay the captured canonical head."
            )
        expected_graph_sha256 = _canonical_sha256(expected_graph.model_dump(mode="json"))
    overlay_state_location = (
        str(row["state_location"]) if bool(row["state_remote"]) else str(manifest_path.parent)
    )
    expected_card = {
        "id": str(row["project_id"]),
        "home_space_id": row["home_space_id"],
        "name": str(row["name"]),
        "locator": str(manifest_path),
        "state_location": overlay_state_location,
        "remote": bool(row["state_remote"]),
        "last_opened_at": row["last_opened_at"],
        "revision": row["revision"],
        "primary_question": row["primary_question"],
        "attention_count": row["attention_count"],
        "last_refresh_at": row["last_refresh_at"],
        "reachable": None if row["reachable"] is None else bool(row["reachable"]),
        "error_sha256": _optional_text_sha256(row["error"]),
        "can_delete": False,
        "delete_unavailable_reason": TEAM_PROJECT_DELETE_UNAVAILABLE_REASON,
    }
    return RehearsalProjectOverlay(
        project_id=str(row["project_id"]),
        name=str(row["name"]),
        capture_status=capture_status,
        overlay_locator=str(manifest_path),
        original_locator=str(row["locator"]),
        original_state_location=str(row["state_location"]),
        original_remote=bool(row["state_remote"]),
        original_reachable=(None if row["reachable"] is None else bool(row["reachable"])),
        original_error_sha256=_optional_text_sha256(row["error"]),
        expected_card_sha256=_canonical_sha256(expected_card),
        expected_graph_sha256=expected_graph_sha256,
        expected_revision=expected_revision,
    )


async def _unused_rehearsal_stream(*_args, **_kwargs) -> AsyncIterator[object]:
    if False:  # pragma: no cover - type-correct effect tripwire
        yield object()


def _expected_startup_recovery(database_path: Path) -> StartupRecoveryReadModel:
    from rcp.storage import AppStore

    tasks = BackgroundAgentTasks(
        AppStore(database_path),
        _unused_rehearsal_stream,
        startup_effect_fence=StartupEffectFence("current-release rehearsal expectation"),
    )
    try:
        return StartupRecoveryReadModel.model_validate(tasks.plan_startup_recovery().as_dict())
    finally:
        tasks.shutdown()


def _configuration_state_is_remote(configuration: BackupManifestConfiguration) -> bool:
    repositories = {repository.alias: repository for repository in configuration.repositories}
    machines = {machine.alias: machine for machine in configuration.machines}
    repository = repositories.get(configuration.state_repository)
    if repository is None:
        return False
    machine = machines.get(repository.machine)
    return bool(machine is not None and machine.host)


def _render_overlay_manifest(
    configuration: BackupManifestConfiguration,
    repository_roots: Mapping[str, Path],
    project_root: Path,
) -> str:
    document = tomlkit.document()
    document.add("name", configuration.name)
    machines = tomlkit.aot()
    for item in configuration.machines:
        machine = tomlkit.table()
        machine.add("alias", item.alias)
        machine.add("host", "")
        machine.add("os_account", "")
        machines.append(machine)
    document.add("machines", machines)
    repositories = tomlkit.aot()
    for item in configuration.repositories:
        repository = tomlkit.table()
        repository.add("alias", item.alias)
        repository.add("machine", item.machine)
        repository.add("path", str(repository_roots[item.alias]))
        repositories.append(repository)
    document.add("repositories", repositories)
    project = tomlkit.table()
    project.add("truth_scope", list(configuration.project_truth_scope))
    document.add("project", project)
    state = tomlkit.table()
    state.add("repository", configuration.state_repository)
    document.add("state", state)
    agent = tomlkit.table()
    agent.add("default_run_truth_scope", list(configuration.default_run_truth_scope))
    agent.add(
        "default_auto_research_invocation_ceiling",
        configuration.default_auto_research_invocation_ceiling,
    )
    defaults = tomlkit.table()
    defaults.add("workflow_ids", list(configuration.skill_defaults.workflow_ids))
    defaults.add("skill_ids", list(configuration.skill_defaults.skill_ids))
    agent.add("skill_defaults", defaults)
    profiles = {profile.profile: profile for profile in configuration.agent_profiles}
    for surface in AGENT_EXECUTION_PROFILES:
        item = profiles[surface]
        profile = tomlkit.table()
        profile.add("provider", item.provider)
        profile.add("runtime", item.runtime)
        profile.add("model", item.model)
        profile.add("reasoning", item.reasoning)
        profile.add("run_on", item.run_on)
        permissions = tomlkit.table()
        for key, value in item.permissions.model_dump(mode="json").items():
            permissions.add(key, value)
        profile.add("permissions", permissions)
        agent.add(surface, profile)
    document.add("agent", agent)
    absent = project_root / "known-absent-provider-history"
    sources = tomlkit.table()
    for name in (
        "claude_roots",
        "codex_roots",
        "remote_claude_roots",
        "remote_codex_roots",
    ):
        sources.add(name, [str(absent / name)])
    document.add("sources", sources)
    content = tomlkit.dumps(document)
    Manifest.model_validate(tomlkit.parse(content).unwrap())
    return content


def _rebind_local_stage_paths(connection: sqlite3.Connection, absent_root: Path) -> None:
    tables = _schema_columns(connection)
    for table, columns in tables.items():
        if "stage_root" not in columns:
            continue
        if "stage_host" not in columns:
            raise CandidateRehearsalRefused(
                f"Path-bearing table {table!r} has no stage-host boundary."
            )
        rows = connection.execute(
            f'SELECT rowid, stage_host, stage_root FROM "{table}" '
            "WHERE stage_root IS NOT NULL AND stage_root != ''"
        ).fetchall()
        for row in rows:
            if row["stage_host"]:
                continue
            rebound = absent_root / "stages" / table / str(row["rowid"])
            connection.execute(
                f'UPDATE "{table}" SET stage_root = ? WHERE rowid = ?',
                (str(rebound), row["rowid"]),
            )
            if "output_path" in columns:
                connection.execute(
                    f'UPDATE "{table}" SET output_path = ? '
                    "WHERE rowid = ? AND output_path IS NOT NULL",
                    (str(rebound / "output"), row["rowid"]),
                )
    if "watchers" in tables:
        rows = connection.execute("SELECT rowid, execution_host FROM watchers").fetchall()
        for row in rows:
            if row["execution_host"]:
                continue
            rebound = absent_root / "watchers" / str(row["rowid"])
            connection.execute(
                "UPDATE watchers SET log_path = ?, cwd = ? WHERE rowid = ?",
                (str(rebound / "log"), str(rebound / "cwd"), row["rowid"]),
            )


def _transfer_inbox_overlay_paths(
    connection: sqlite3.Connection,
    data_dir: Path,
) -> tuple[str, ...]:
    if "project_provisioning_requests" not in _schema_columns(connection):
        return ()
    request_ids = [
        str(row[0])
        for row in connection.execute(
            "SELECT request_id FROM project_provisioning_requests "
            "WHERE kind = 'incoming_transfer' ORDER BY request_id"
        ).fetchall()
    ]
    return tuple(str(data_dir / "transfer-inbox" / request_id) for request_id in request_ids)


def _validate_path_column_inventory(connection: sqlite3.Connection) -> None:
    for table, columns in _schema_columns(connection).items():
        unexpected = {
            column
            for column in columns
            if (
                column in _PATH_COLUMN_NAMES or column.endswith("_path") or column.endswith("_root")
            )
            and column not in _PATH_COLUMN_NAMES
        }
        if unexpected:
            raise CandidateRehearsalRefused(
                f"Copied table {table!r} has unclassified path columns: {sorted(unexpected)}."
            )
        for column in columns.intersection(_PATH_COLUMN_NAMES):
            if column in {"locator", "state_location"} and table != "projects":
                raise CandidateRehearsalRefused(
                    f"Copied table {table!r} unexpectedly owns {column!r}."
                )
            if column in {"log_path", "cwd"} and table != "watchers":
                raise CandidateRehearsalRefused(
                    f"Copied table {table!r} unexpectedly owns {column!r}."
                )
            if column == "output_path" and "stage_root" not in columns:
                raise CandidateRehearsalRefused(
                    f"Copied table {table!r} has output paths without a stage boundary."
                )


def _validate_rebound_paths(
    connection: sqlite3.Connection,
    *,
    root: Path,
    projects: list[RehearsalProjectOverlay],
) -> None:
    expected = {project.project_id: project for project in projects}
    for row in connection.execute(
        "SELECT project_id, locator, state_location, state_remote FROM projects"
    ).fetchall():
        project = expected[str(row["project_id"])]
        if row["locator"] != project.overlay_locator or not Path(row["locator"]).is_relative_to(
            root
        ):
            raise CandidateRehearsalRefused("A copied project locator escaped its overlay.")
        if not bool(row["state_remote"]):
            state = Path(str(row["state_location"]))
            if not state.is_relative_to(root):
                raise CandidateRehearsalRefused(
                    "A copied local canonical-state pointer escaped its overlay."
                )
    for table, columns in _schema_columns(connection).items():
        if "stage_root" in columns:
            for row in connection.execute(
                f'SELECT stage_host, stage_root FROM "{table}" '
                "WHERE stage_root IS NOT NULL AND stage_root != ''"
            ).fetchall():
                if not row["stage_host"] and not Path(str(row["stage_root"])).is_relative_to(root):
                    raise CandidateRehearsalRefused(
                        f"A copied local stage in {table!r} escaped its overlay."
                    )
                if not row["stage_host"] and (
                    Path(str(row["stage_root"])).exists()
                    or Path(str(row["stage_root"])).is_symlink()
                ):
                    raise CandidateRehearsalRefused(
                        f"A copied local stage in {table!r} is not known-absent."
                    )
        if table == "watchers":
            for row in connection.execute(
                "SELECT execution_host, log_path, cwd FROM watchers"
            ).fetchall():
                if not row["execution_host"] and any(
                    not Path(str(row[name])).is_relative_to(root) for name in ("log_path", "cwd")
                ):
                    raise CandidateRehearsalRefused(
                        "A copied local watcher path escaped its overlay."
                    )


def _schema_columns(connection: sqlite3.Connection) -> dict[str, set[str]]:
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    if any('"' in table or "\x00" in table for table in tables):
        raise CandidateRehearsalRefused("The copied database has an unsafe table name.")
    return {
        table: {
            str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        }
        for table in tables
    }


def run_candidate_child(overlay_path: Path, result_path: Path) -> int:
    overlay: RehearsalOverlay | None = None
    fence = StartupEffectFence("candidate update rehearsal")
    lock_context = None
    lock_acquired = False
    try:
        overlay = RehearsalOverlay.model_validate_json(_read_bounded_file(overlay_path))
        if overlay_path.parent != Path(overlay.root).parent:
            raise CandidateRehearsalRefused(
                "The candidate overlay manifest is outside its rehearsal operation."
            )
        from rcp.__main__ import instance_lock
        from rcp.storage import AppStore

        lock_context = instance_lock(Path(overlay.data_dir), timeout=0.0)
        lock_context.__enter__()
        lock_acquired = True
        opened = AppStore(Path(overlay.database_path))
        users = opened.space_users()
        if opened.space_kind != "team" or not users:
            raise CandidateRehearsalRefused(
                "The copied database is not one usable enrolled team space."
            )
        users_by_id = {user.user_id: user for user in users if user.identity_kind == "team_member"}
        project_principals: dict[str, str] = {}
        for project in overlay.projects:
            principal = next(
                (
                    membership.user_id
                    for membership in opened.project_members(project.project_id)
                    if membership.user_id in users_by_id
                ),
                None,
            )
            if principal is None:
                raise CandidateRehearsalRefused(
                    f"Copied project {project.project_id} has no enrolled member."
                )
            project_principals[project.project_id] = principal
        default_user_id = next(iter(users_by_id), None)
        if default_user_id is None:
            raise CandidateRehearsalRefused("The copied team space has no enrolled member.")

        def rehearsal_principal(request, store):
            requested = request.headers.get("x-rcp-rehearsal-user", default_user_id)
            if requested not in users_by_id:
                return None
            return store.space_user(requested)

        app = create_app(
            data_dir=Path(overlay.data_dir),
            trusted_principal_resolver=rehearsal_principal,
            startup_effect_fence=fence,
        )
        reads: list[str] = []
        results: list[CandidateProjectVerification] = []
        with TestClient(app) as client:
            health = client.get("/api/health")
            if health.status_code != 200:
                raise CandidateRehearsalRefused("Candidate health read failed.")
            health_payload = health.json()
            if (
                health_payload.get("space_id") != opened.space_id
                or health_payload.get("space_kind") != "team"
            ):
                raise CandidateRehearsalRefused("Candidate health changed copied space identity.")
            reads.append("/api/health")
            cards: dict[str, dict[str, object]] = {}
            for principal in sorted(users_by_id):
                listing = client.get(
                    "/api/projects",
                    headers={"x-rcp-rehearsal-user": principal},
                )
                if listing.status_code != 200:
                    raise CandidateRehearsalRefused("Candidate project inventory read failed.")
                for raw_card in listing.json():
                    card = dict(raw_card)
                    project_id = str(card["id"])
                    existing = cards.get(project_id)
                    if existing is not None and existing != card:
                        raise CandidateRehearsalRefused(
                            "Candidate project inventory changed between member reads."
                        )
                    cards[project_id] = card
            if set(cards) != {project.project_id for project in overlay.projects}:
                raise CandidateRehearsalRefused(
                    "Candidate project inventory omitted or substituted a project."
                )
            reads.append("/api/projects")
            for project in overlay.projects:
                headers = {"x-rcp-rehearsal-user": project_principals[project.project_id]}
                if project.capture_status == "captured":
                    response = client.get(
                        f"/api/projects/{project.project_id}",
                        headers=headers,
                    )
                    if response.status_code != 200:
                        raise CandidateRehearsalRefused(
                            f"Candidate replay failed for project {project.project_id}."
                        )
                    payload = response.json()
                    graph = payload.get("graph")
                    revision = graph.get("revision") if isinstance(graph, dict) else None
                    if (
                        payload.get("id") != project.project_id
                        or revision != project.expected_revision
                    ):
                        raise CandidateRehearsalRefused(
                            f"Candidate replay changed project {project.project_id}."
                        )
                    tasks = client.get(
                        f"/api/projects/{project.project_id}/tasks",
                        headers=headers,
                    )
                    watchers = client.get(
                        f"/api/projects/{project.project_id}/watchers",
                        headers=headers,
                    )
                    if tasks.status_code != 200 or watchers.status_code != 200:
                        raise CandidateRehearsalRefused(
                            f"Candidate operational reads failed for project {project.project_id}."
                        )
                    reads.extend(
                        (
                            f"/api/projects/{project.project_id}",
                            f"/api/projects/{project.project_id}/tasks",
                            f"/api/projects/{project.project_id}/watchers",
                        )
                    )
                    results.append(
                        CandidateProjectVerification(
                            project_id=project.project_id,
                            status="verified",
                            revision=revision,
                            projection_sha256=_canonical_sha256(graph),
                        )
                    )
                else:
                    card = cards[project.project_id]
                    comparison = _project_card_comparison(card)
                    if _canonical_sha256(comparison) != project.expected_card_sha256:
                        raise CandidateRehearsalRefused(
                            f"Candidate changed unavailable projection {project.project_id}."
                        )
                    results.append(
                        CandidateProjectVerification(
                            project_id=project.project_id,
                            status="not_replay_verified",
                            revision=None,
                            projection_sha256=_canonical_sha256(comparison),
                        )
                    )
        if fence.attempted_effects:
            raise CandidateRehearsalRefused(
                "Candidate startup attempted an effect while its fence was closed."
            )
        startup = app.state.startup_recovery_plan
        result = CandidateRehearsalResult(
            status="verified",
            space_id=opened.space_id,
            space_kind="team",
            startup_recovery=StartupRecoveryReadModel.model_validate(startup),
            projects=tuple(sorted(results, key=lambda item: item.project_id)),
            reads=tuple(reads),
            attempted_effects=fence.attempted_effects,
        )
        _write_private_json(result_path, result)
        return 0
    except BaseException as exc:
        diagnostic = redact_server_text(str(exc)).strip()
        if not diagnostic or len(diagnostic) > BACKUP_DIAGNOSTIC_MAX_CHARS:
            diagnostic = "Candidate copied-state verification failed."
        failed = CandidateRehearsalResult(
            status="failed",
            attempted_effects=fence.attempted_effects,
            diagnostic=diagnostic,
        )
        with suppress(OSError, ValueError):
            _write_private_json(result_path, failed)
        return 1
    finally:
        if lock_context is not None and lock_acquired:
            lock_context.__exit__(None, None, None)


def run_candidate_migration(database_path: Path, result_path: Path) -> int:
    try:
        if database_path.name != "rcp.sqlite3" or database_path.parent.name != "data":
            raise CandidateRehearsalRefused(
                "The candidate migration target is not one rehearsal database."
            )
        from rcp.storage import AppStore

        AppStore(database_path)
        _write_private_json(result_path, CandidateMigrationResult(status="migrated"))
        return 0
    except BaseException as exc:
        diagnostic = redact_server_text(str(exc)).strip()
        if not diagnostic or len(diagnostic) > BACKUP_DIAGNOSTIC_MAX_CHARS:
            diagnostic = "Candidate copied-state migration failed."
        with suppress(OSError, ValueError):
            _write_private_json(
                result_path,
                CandidateMigrationResult(status="failed", diagnostic=diagnostic),
            )
        return 1


def run_rehearsal_orchestrator(
    built_receipt_path: Path,
    data_dir: Path,
    update_root: Path,
    *,
    operation_receipt_path: Path | None = None,
    operation_receipt_sha256: str | None = None,
) -> int:
    """Service-account entrypoint invoked by the narrow root update coordinator."""

    try:
        content = _read_private_file(
            built_receipt_path,
            expected_uid=os.geteuid(),
            expected_mode=_RECEIPT_MODE,
        )
        built = BuiltCandidateReceipt.model_validate_json(content)
        if built.receipt_path != str(built_receipt_path):
            raise CandidateRehearsalRefused(
                "The built-candidate receipt does not name its exact path."
            )
        if (operation_receipt_path is None) != (operation_receipt_sha256 is None):
            raise CandidateRehearsalRefused(
                "Final rehearsal requires both the update receipt and its digest."
            )
        capture_result = None
        retain_capture = False
        if operation_receipt_path is not None:
            from rcp.server_ops.update_cutover import (
                control_capture_from_boundary,
                read_update_operation,
            )

            operation, _digest = read_update_operation(
                operation_receipt_path,
                expected_uid=os.geteuid(),
                expected_sha256=operation_receipt_sha256,
            )
            if operation.state != "maintenance_closed" or operation.capture is None:
                raise CandidateRehearsalRefused(
                    "Final rehearsal requires one closed-admission capture boundary."
                )
            built_sha256 = hashlib.sha256(content).hexdigest()
            if (
                operation.base_instance_id != built.base_instance_id
                or operation.base_process_pid != built.base_process_pid
                or operation.built_receipt_path != built.receipt_path
                or operation.built_receipt_sha256 != built_sha256
            ):
                raise CandidateRehearsalRefused(
                    "The update maintenance receipt differs from its built candidate."
                )
            capture_result = control_capture_from_boundary(operation.capture)
            retain_capture = True
        receipt = CandidateRehearsalCoordinator(
            data_dir=data_dir,
            update_root=update_root,
            built_receipt=built,
            built_receipt_sha256=hashlib.sha256(content).hexdigest(),
            capture_result=capture_result,
            retain_capture=retain_capture,
        ).run()
        print(receipt.receipt_path, flush=True)
        return 0
    except (CandidateRehearsalRefused, OSError, ValueError) as exc:
        diagnostic = redact_server_text(str(exc)).strip()
        print(
            diagnostic or "Candidate copied-state rehearsal failed safely.",
            file=sys.stderr,
        )
        return 1


def read_verified_candidate_receipt(
    path: Path,
    *,
    expected_uid: int,
) -> VerifiedCandidateReceipt:
    payload = _read_private_file(path, expected_uid=expected_uid, expected_mode=_RECEIPT_MODE)
    try:
        receipt = VerifiedCandidateReceipt.model_validate_json(payload)
    except ValueError as exc:
        raise CandidateRehearsalRefused("The verified-candidate receipt is invalid.") from exc
    matched = _VERIFIED_RECEIPT_NAME.fullmatch(path.name)
    if (
        matched is None
        or matched.group(1) != receipt.candidate_commit
        or matched.group(2) != receipt.capture_id
        or receipt.receipt_path != str(path)
    ):
        raise CandidateRehearsalRefused("The verified-candidate receipt path and commit disagree.")
    return receipt


def _read_candidate_result(path: Path) -> CandidateRehearsalResult:
    try:
        return CandidateRehearsalResult.model_validate_json(_read_bounded_file(path))
    except (OSError, ValueError) as exc:
        raise CandidateRehearsalRefused(
            "The candidate did not publish one valid copied-state result."
        ) from exc


def _read_candidate_migration_result(path: Path) -> CandidateMigrationResult:
    try:
        return CandidateMigrationResult.model_validate_json(_read_bounded_file(path))
    except (OSError, ValueError) as exc:
        raise CandidateRehearsalRefused(
            "The candidate did not publish one valid migration result."
        ) from exc


def _run_candidate_process(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CandidateRehearsalRefused(
            "The isolated candidate process could not complete."
        ) from exc


def _copy_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    _mkdir_private_parents(destination.parent)
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor = -1
    try:
        initial = os.fstat(source_descriptor)
        path_initial = source.lstat()
        if (
            not stat.S_ISREG(initial.st_mode)
            or (initial.st_dev, initial.st_ino) != (path_initial.st_dev, path_initial.st_ino)
            or initial.st_size != expected_size
        ):
            raise CandidateRehearsalRefused(
                "A captured rehearsal file has unsafe type, identity, or size."
            )
        destination_descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            _FILE_MODE,
        )
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(source_descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("short rehearsal file write")
                view = view[written:]
            digest.update(chunk)
            size += len(chunk)
        final = os.fstat(source_descriptor)
        path_final = source.lstat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(initial, field) != getattr(final, field) for field in stable_fields)
            or any(getattr(final, field) != getattr(path_final, field) for field in stable_fields)
            or size != expected_size
            or digest.hexdigest() != expected_sha256
        ):
            raise CandidateRehearsalRefused(
                "A captured rehearsal file changed or was copied incompletely."
            )
        os.fchmod(destination_descriptor, _FILE_MODE)
        os.fsync(destination_descriptor)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
    _fsync_directory(destination.parent)


def _read_verified_bytes(
    source: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> bytes:
    try:
        info = source.lstat()
    except OSError as exc:
        raise CandidateRehearsalRefused("A captured rehearsal file is unavailable.") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
        raise CandidateRehearsalRefused("A captured rehearsal file has unsafe type or size.")
    data = source.read_bytes()
    if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise CandidateRehearsalRefused("A captured rehearsal file changed after capture.")
    return data


def _mkdir_private_parents(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=_DIRECTORY_MODE)
    for directory in (path, *path.parents):
        if directory == current.parent:
            break
        if directory.exists() and directory.is_relative_to(current):
            os.chmod(directory, _DIRECTORY_MODE)


def _write_private_json(path: Path, model: BaseModel) -> None:
    _write_private_bytes(path, _model_bytes(model))


def _write_private_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        _FILE_MODE,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short rehearsal write")
            view = view[written:]
        os.fchmod(descriptor, _FILE_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _publish_private_json(path: Path, model: BaseModel) -> None:
    content = _model_bytes(model)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, _RECEIPT_MODE)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise CandidateRehearsalRefused(
            "Another verified-candidate receipt appeared during publication."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _model_bytes(model: BaseModel) -> bytes:
    content = (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(content) > _MAX_RECEIPT_BYTES:
        raise CandidateRehearsalRefused("A rehearsal receipt exceeds its fixed size bound.")
    return content


def _read_private_file(path: Path, *, expected_uid: int, expected_mode: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != expected_uid
            or stat.S_IMODE(info.st_mode) != expected_mode
            or info.st_size > _MAX_RECEIPT_BYTES
        ):
            raise CandidateRehearsalRefused("A rehearsal receipt has unsafe metadata.")
        content = os.read(descriptor, _MAX_RECEIPT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(content) > _MAX_RECEIPT_BYTES or len(content) != info.st_size:
        raise CandidateRehearsalRefused("A rehearsal receipt is oversized or incomplete.")
    return content


def _read_bounded_file(path: Path) -> bytes:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_RECEIPT_BYTES:
            raise ValueError("unsafe file")
        content = path.read_bytes()
    except (OSError, ValueError) as exc:
        raise CandidateRehearsalRefused("A rehearsal handoff file is unavailable.") from exc
    if len(content) > _MAX_RECEIPT_BYTES:
        raise CandidateRehearsalRefused("A rehearsal handoff file is oversized.")
    return content


def _require_private_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CandidateRehearsalRefused(f"The {label} is unavailable.") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise CandidateRehearsalRefused(f"The {label} has unsafe ownership or mode.")


def _discard_operation_root(operation_root: Path, *, update_root: Path) -> None:
    if (
        operation_root.parent != update_root
        or _REHEARSAL_ROOT_NAME.fullmatch(operation_root.name) is None
    ):
        raise CandidateRehearsalRefused("The rehearsal operation root is not canonical.")
    _require_private_directory(operation_root, label="rehearsal operation root")
    shutil.rmtree(operation_root)
    _fsync_directory(update_root)


def _optional_text_sha256(value: object) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _project_card_comparison(card: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": card.get("id"),
        "home_space_id": card.get("home_space_id"),
        "name": card.get("name"),
        "locator": card.get("locator"),
        "state_location": card.get("state_location"),
        "remote": card.get("remote"),
        "last_opened_at": card.get("last_opened_at"),
        "revision": card.get("revision"),
        "primary_question": card.get("primary_question"),
        "attention_count": card.get("attention_count"),
        "last_refresh_at": card.get("last_refresh_at"),
        "reachable": card.get("reachable"),
        "error_sha256": _optional_text_sha256(card.get("error")),
        "can_delete": card.get("can_delete"),
        "delete_unavailable_reason": card.get("delete_unavailable_reason"),
    }


def _expected_candidate_reads(
    projects: tuple[RehearsalProjectOverlay, ...],
) -> tuple[str, ...]:
    reads = ["/api/health", "/api/projects"]
    for project in projects:
        if project.capture_status == "captured":
            reads.extend(
                (
                    f"/api/projects/{project.project_id}",
                    f"/api/projects/{project.project_id}/tasks",
                    f"/api/projects/{project.project_id}/watchers",
                )
            )
    return tuple(reads)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--candidate-child", nargs=2, metavar=("OVERLAY", "RESULT"))
    modes.add_argument(
        "--candidate-migrate",
        nargs=2,
        metavar=("DATABASE", "RESULT"),
    )
    modes.add_argument(
        "--orchestrate",
        nargs=3,
        metavar=("BUILT_RECEIPT", "DATA_DIR", "UPDATE_ROOT"),
    )
    modes.add_argument(
        "--orchestrate-maintenance",
        nargs=5,
        metavar=(
            "BUILT_RECEIPT",
            "DATA_DIR",
            "UPDATE_ROOT",
            "OPERATION_RECEIPT",
            "OPERATION_SHA256",
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.candidate_child is not None:
        overlay, result = (Path(value) for value in arguments.candidate_child)
        return run_candidate_child(overlay, result)
    if arguments.candidate_migrate is not None:
        database, result = (Path(value) for value in arguments.candidate_migrate)
        return run_candidate_migration(database, result)
    if arguments.orchestrate is not None:
        built_receipt, data_dir, update_root = (Path(value) for value in arguments.orchestrate)
        return run_rehearsal_orchestrator(built_receipt, data_dir, update_root)
    built_receipt, data_dir, update_root, operation_receipt, operation_sha256 = (
        arguments.orchestrate_maintenance
    )
    return run_rehearsal_orchestrator(
        Path(built_receipt),
        Path(data_dir),
        Path(update_root),
        operation_receipt_path=Path(operation_receipt),
        operation_receipt_sha256=operation_sha256,
    )


if __name__ == "__main__":  # pragma: no cover - exercised through the candidate subprocess
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "CandidateProjectVerification",
    "CandidateMigrationResult",
    "CandidateRehearsalCoordinator",
    "CandidateRehearsalRefused",
    "CandidateRehearsalResult",
    "RehearsalOverlay",
    "RehearsalProjectOverlay",
    "StartupRecoveryReadModel",
    "VerifiedCandidateReceipt",
    "build_rehearsal_overlay",
    "read_verified_candidate_receipt",
    "run_candidate_child",
    "run_candidate_migration",
    "run_rehearsal_orchestrator",
    "verified_candidate_receipt_path",
]
