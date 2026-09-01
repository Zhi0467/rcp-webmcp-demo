"""Optimistic canonical and referenced project-file capture for server backups."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from rcp.config import load_manifest
from rcp.core.transition_models import GraphHeadRef
from rcp.paper.service import (
    canonical_introduction_backup_source,
    validate_canonical_introduction_backup,
)
from rcp.server_ops.backup_capture import (
    BackupCaptureUnavailable,
    BackupSnapshotProjectInventory,
    BackupSQLiteCaptureReceipt,
    read_backup_sqlite_capture_receipt,
    read_immutable_backup_receipt,
    validate_backup_sqlite_snapshot,
    write_immutable_backup_receipt,
)
from rcp.server_ops.backup_checkout import (
    BackupCheckoutHostUnavailable,
    verify_checkout_identities,
)
from rcp.server_ops.backup_models import (
    BackupFileEntry,
    BackupImportedProviderSourceCapture,
    BackupManifestConfiguration,
    BackupProjectCapture,
)
from rcp.server_ops.backup_project_io import (
    BackupProjectFileUnavailable,
    capture_chat_files,
    discard_failed_project_capture,
    fact_backup_sources,
    fsync_directory,
    fsync_tree,
    stable_copy_entry,
    stable_copy_fact_entry,
    stable_workspace_bytes,
    write_bytes_entry,
)
from rcp.sources.imported import (
    ImportedProviderSourceInventory,
    ImportedProviderSourceStore,
)
from rcp.transport.remote_backup_checkout import CheckoutInspectionError
from rcp.transport.state import LocalStateWorkspace, StateUnavailable, state_workspace_for_probe

BACKUP_PROJECT_FILE_CAPTURE_SCHEMA_VERSION = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_PROJECT_CAPTURE_FAILURE = "The project files were invalid, changing, or unavailable."


def _canonical_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase canonical UUID4")
    return value


class _StrictProjectCaptureModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


class BackupProjectFileCaptureReceipt(_StrictProjectCaptureModel):
    schema_version: Literal[BACKUP_PROJECT_FILE_CAPTURE_SCHEMA_VERSION] = (
        BACKUP_PROJECT_FILE_CAPTURE_SCHEMA_VERSION
    )
    capture_id: str
    captured_at: datetime
    completed_at: datetime
    rcp_source_commit: str
    space_id: str
    sqlite_receipt_sha256: str
    sqlite_snapshot_sha256: str
    sqlite_capture_status: Literal["complete", "partial"]
    projects: tuple[BackupProjectCapture, ...]
    imported_sources: tuple[BackupImportedProviderSourceCapture, ...] = ()
    status: Literal["complete", "partial"]

    @field_validator("capture_id", "space_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("captured_at", "completed_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("backup project-file times require a timezone")
        return value

    @field_validator("rcp_source_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("backup source commit must be one full Git object id")
        return value

    @field_validator("sqlite_receipt_sha256", "sqlite_snapshot_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("backup capture digests must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_capture(self) -> BackupProjectFileCaptureReceipt:
        if self.completed_at < self.captured_at:
            raise ValueError("project-file capture cannot complete before its SQLite snapshot")
        project_ids = [project.project_id for project in self.projects]
        if tuple(sorted(project_ids)) != tuple(project_ids) or len(project_ids) != len(
            set(project_ids)
        ):
            raise ValueError("project-file capture projects must be sorted and unique")
        if any(
            project.status == "captured" and project.home_space_id != self.space_id
            for project in self.projects
        ):
            raise ValueError("project-file capture cannot protect another space's project")
        imported_ids = [capture.project_id for capture in self.imported_sources]
        if imported_ids and (
            tuple(sorted(imported_ids)) != tuple(imported_ids)
            or len(imported_ids) != len(set(imported_ids))
            or set(imported_ids) != set(project_ids)
        ):
            raise ValueError("project-file imported sources must inventory every project")
        partial = self.sqlite_capture_status == "partial" or any(
            project.status == "uncaptured" for project in self.projects
        )
        if self.status != ("partial" if partial else "complete"):
            raise ValueError("project-file capture status does not match its results")
        return self


class BackupProjectFileCapturePublication(_StrictProjectCaptureModel):
    receipt: BackupProjectFileCaptureReceipt
    receipt_path: Path
    receipt_sha256: str

    @field_validator("receipt_sha256")
    @classmethod
    def validate_receipt_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("project-file receipt digest must be lowercase SHA-256")
        return value


class BackupProjectFileCaptureCoordinator:
    """Consume one O2a receipt without consulting later live database state."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()

    def capture(
        self,
        sqlite_receipt_path: Path,
        *,
        expected_sha256: str,
    ) -> BackupProjectFileCapturePublication:
        sqlite_receipt = read_backup_sqlite_capture_receipt(
            sqlite_receipt_path,
            expected_sha256=expected_sha256,
        )
        self._validate_capture_boundary(sqlite_receipt_path, sqlite_receipt)
        validate_backup_sqlite_snapshot(sqlite_receipt)
        capture_root = sqlite_receipt_path.parent
        projects_root = capture_root / "projects"
        try:
            projects_root.mkdir(mode=0o700)
        except OSError as exc:
            raise BackupCaptureUnavailable(
                "The private project-file capture root could not be created."
            ) from exc

        operation_projects = {
            operation_id: project.project_id
            for project in sqlite_receipt.projects
            for operation_id in project.task_operation_ids
        }
        imported_sources = self._capture_imported_sources(capture_root, sqlite_receipt)
        projects = tuple(
            self._capture_or_preserve_failure(
                capture_root,
                inventory,
                operation_projects=operation_projects,
            )
            for inventory in sqlite_receipt.projects
        )
        fsync_directory(projects_root)
        completed_at = datetime.now(UTC)
        partial = sqlite_receipt.status == "partial" or any(
            project.status == "uncaptured" for project in projects
        )
        receipt = BackupProjectFileCaptureReceipt(
            capture_id=sqlite_receipt.capture_id,
            captured_at=sqlite_receipt.captured_at,
            completed_at=completed_at,
            rcp_source_commit=sqlite_receipt.rcp_source_commit,
            space_id=sqlite_receipt.space_id,
            sqlite_receipt_sha256=expected_sha256,
            sqlite_snapshot_sha256=sqlite_receipt.sqlite_snapshot.sha256,
            sqlite_capture_status=sqlite_receipt.status,
            projects=projects,
            imported_sources=imported_sources,
            status="partial" if partial else "complete",
        )
        receipt_path = capture_root / "project-files.json"
        receipt_sha256 = write_immutable_backup_receipt(receipt_path, receipt)
        fsync_directory(capture_root)
        return BackupProjectFileCapturePublication(
            receipt=receipt,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
        )

    def _capture_imported_sources(
        self,
        capture_root: Path,
        sqlite_receipt: BackupSQLiteCaptureReceipt,
    ) -> tuple[BackupImportedProviderSourceCapture, ...]:
        collection_root = capture_root / "project-sources"
        captures: list[BackupImportedProviderSourceCapture] = []
        for recorded in sqlite_receipt.imported_source_inventories:
            expected = ImportedProviderSourceInventory.model_validate(recorded.model_dump())
            store = ImportedProviderSourceStore(self.data_dir, recorded.project_id)
            if os.path.lexists(store.root):
                collection_root.mkdir(mode=0o700, exist_ok=True)
                project_root = collection_root / recorded.project_id
                project_root.mkdir(mode=0o700)
                destination = project_root / "provider-history"
            else:
                destination = collection_root / recorded.project_id / "provider-history"
            snapshot = store.capture_snapshot(
                destination,
                expected_inventory=expected,
            )
            files = tuple(
                BackupFileEntry(
                    archive_path=(
                        f"project-sources/{recorded.project_id}/provider-history/"
                        f"{item.relative_path}"
                    ),
                    source_relative_path=f"provider-history/{item.relative_path}",
                    group="imported_provider_history",
                    sha256=item.sha256,
                    size_bytes=item.size_bytes,
                )
                for item in snapshot.files
            )
            captures.append(
                BackupImportedProviderSourceCapture(
                    project_id=recorded.project_id,
                    inventory=recorded,
                    present=snapshot.present,
                    files=files,
                    total_bytes=sum(item.size_bytes for item in files),
                )
            )
        if collection_root.exists():
            fsync_tree(collection_root)
        return tuple(captures)

    def _validate_capture_boundary(
        self,
        receipt_path: Path,
        receipt: BackupSQLiteCaptureReceipt,
    ) -> None:
        capture_root = self.data_dir / "run-stage" / f"backup-{receipt.capture_id}"
        if (
            receipt_path != capture_root / "sqlite-capture.json"
            or Path(receipt.snapshot_path) != capture_root / "rcp.sqlite3"
            or receipt.app_data_plan.data_dir != str(self.data_dir)
        ):
            raise BackupCaptureUnavailable(
                "The SQLite capture does not belong to this exact data directory."
            )
        try:
            metadata = capture_root.lstat()
        except OSError as exc:
            raise BackupCaptureUnavailable("The private backup capture is unavailable.") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise BackupCaptureUnavailable("The private backup capture boundary is unsafe.")

    def _capture_or_preserve_failure(
        self,
        capture_root: Path,
        inventory: BackupSnapshotProjectInventory,
        *,
        operation_projects: Mapping[str, str],
    ) -> BackupProjectCapture:
        if inventory.status == "uncaptured":
            return BackupProjectCapture(
                project_id=inventory.project_id,
                home_space_id=inventory.home_space_id,
                locator=inventory.locator,
                status="uncaptured",
                unavailable_kind="inventory_failure",
                unavailable_reason=inventory.unavailable_reason,
                unavailable_at=inventory.unavailable_at,
                total_bytes=0,
            )
        project_root = capture_root / "projects" / inventory.project_id
        try:
            project_root.mkdir(mode=0o700)
        except OSError:
            return BackupProjectCapture(
                project_id=inventory.project_id,
                home_space_id=inventory.home_space_id,
                locator=inventory.locator,
                status="uncaptured",
                unavailable_kind="capture_failure",
                unavailable_reason=_PROJECT_CAPTURE_FAILURE,
                unavailable_at=datetime.now(UTC),
                total_bytes=0,
            )
        try:
            return self._capture_project(
                capture_root,
                project_root,
                inventory,
                operation_projects=operation_projects,
            )
        except BackupCheckoutHostUnavailable as exc:
            discard_failed_project_capture(capture_root, project_root)
            if _inventory_state_machine_alias(inventory) == exc.machine_alias:
                return BackupProjectCapture(
                    project_id=inventory.project_id,
                    home_space_id=inventory.home_space_id,
                    locator=inventory.locator,
                    status="uncaptured",
                    recovery=inventory.recovery,
                    unavailable_kind="remote_unreachable",
                    unavailable_reason="The configured SSH canonical state was unreachable.",
                    unavailable_at=datetime.now(UTC),
                    total_bytes=0,
                )
            return BackupProjectCapture(
                project_id=inventory.project_id,
                home_space_id=inventory.home_space_id,
                locator=inventory.locator,
                status="uncaptured",
                unavailable_kind="capture_failure",
                unavailable_reason=_PROJECT_CAPTURE_FAILURE,
                unavailable_at=datetime.now(UTC),
                total_bytes=0,
            )
        except StateUnavailable:
            discard_failed_project_capture(capture_root, project_root)
            if _inventory_state_is_remote(inventory):
                return BackupProjectCapture(
                    project_id=inventory.project_id,
                    home_space_id=inventory.home_space_id,
                    locator=inventory.locator,
                    status="uncaptured",
                    recovery=inventory.recovery,
                    unavailable_kind="remote_unreachable",
                    unavailable_reason="The configured SSH canonical state was unreachable.",
                    unavailable_at=datetime.now(UTC),
                    total_bytes=0,
                )
            return BackupProjectCapture(
                project_id=inventory.project_id,
                home_space_id=inventory.home_space_id,
                locator=inventory.locator,
                status="uncaptured",
                unavailable_kind="capture_failure",
                unavailable_reason=_PROJECT_CAPTURE_FAILURE,
                unavailable_at=datetime.now(UTC),
                total_bytes=0,
            )
        except (
            BackupProjectFileUnavailable,
            CheckoutInspectionError,
            OSError,
            TypeError,
            ValueError,
        ):
            discard_failed_project_capture(capture_root, project_root)
            return BackupProjectCapture(
                project_id=inventory.project_id,
                home_space_id=inventory.home_space_id,
                locator=inventory.locator,
                status="uncaptured",
                unavailable_kind="capture_failure",
                unavailable_reason=_PROJECT_CAPTURE_FAILURE,
                unavailable_at=datetime.now(UTC),
                total_bytes=0,
            )

    def _capture_project(
        self,
        capture_root: Path,
        project_root: Path,
        inventory: BackupSnapshotProjectInventory,
        *,
        operation_projects: Mapping[str, str],
    ) -> BackupProjectCapture:
        recovery = inventory.recovery
        locator = inventory.locator
        if recovery is None or locator is None or inventory.home_space_id is None:
            raise BackupProjectFileUnavailable("The project capture proof is incomplete.")
        manifest = load_manifest(locator)
        if BackupManifestConfiguration.from_manifest(manifest) != recovery.configuration:
            raise BackupProjectFileUnavailable("The project manifest changed after SQLite capture.")
        verify_checkout_identities(recovery)
        workspace = state_workspace_for_probe(manifest, self.data_dir)

        with tempfile.TemporaryDirectory(
            prefix=f".sources-{inventory.project_id}-",
            dir=capture_root,
        ) as temporary:
            export_root = Path(temporary)
            export_root.chmod(0o700)
            source_root = workspace.backup_source_root(export_root)
            plan = LocalStateWorkspace(source_root, str(source_root)).backup_canonical_source_plan()
            if not plan.complete:
                raise BackupProjectFileUnavailable(
                    "The project contains an unclassified durable research root."
                )
            files: list[BackupFileEntry] = []
            for source in (
                *plan.main_files,
                *(item for branch in plan.branches for item in branch.files),
            ):
                relative = PurePosixPath(".research") / source.relative_path
                files.append(
                    stable_copy_entry(
                        source_root / source.relative_path,
                        project_root,
                        inventory.project_id,
                        relative,
                        group="canonical",
                        expected_size=source.observed_size_bytes,
                    )
                )
            files.extend(
                capture_chat_files(
                    source_root,
                    project_root,
                    inventory.project_id,
                    operation_projects=operation_projects,
                )
            )
            introduction = canonical_introduction_backup_source(source_root)
            if introduction is not None:
                paper_entry = stable_copy_entry(
                    introduction,
                    project_root,
                    inventory.project_id,
                    PurePosixPath(".research/paper/introduction.md"),
                    group="paper_introduction",
                )
                validate_canonical_introduction_backup(
                    project_root / paper_entry.source_relative_path
                )
                files.append(paper_entry)
            for fact in fact_backup_sources(source_root):
                files.append(
                    stable_copy_fact_entry(
                        source_root,
                        fact,
                        project_root,
                        inventory.project_id,
                        PurePosixPath(".research/facts") / fact.relative_path,
                    )
                )

        for reference in inventory.kept_artifacts:
            data = stable_workspace_bytes(
                lambda name=reference.kept_filename: workspace.read_kept_artifact(name)
            )
            if (
                reference.expected_size_bytes is not None
                and len(data) != reference.expected_size_bytes
            ):
                raise BackupProjectFileUnavailable("A kept artifact size differs from SQLite.")
            files.append(
                write_bytes_entry(
                    data,
                    project_root,
                    inventory.project_id,
                    PurePosixPath("artifacts") / reference.kept_filename,
                    group="kept_artifact",
                )
            )
        for reference in inventory.kept_result_views:
            data = stable_workspace_bytes(
                lambda name=reference.kept_filename: workspace.read_kept_result_view(name)
            )
            if (
                len(data) != reference.size_bytes
                or hashlib.sha256(data).hexdigest() != reference.content_sha256
            ):
                raise BackupProjectFileUnavailable("A kept result view differs from SQLite.")
            files.append(
                write_bytes_entry(
                    data,
                    project_root,
                    inventory.project_id,
                    PurePosixPath("views") / reference.kept_filename,
                    group="legacy_kept_result_view",
                )
            )

        ordered = tuple(sorted(files, key=lambda item: item.archive_path))
        fsync_tree(project_root)
        return BackupProjectCapture(
            project_id=inventory.project_id,
            home_space_id=inventory.home_space_id,
            locator=locator,
            status="captured",
            main_head=GraphHeadRef(revision=plan.main_observed_revision),
            branch_heads=tuple(branch.head for branch in plan.branches),
            files=ordered,
            recovery=recovery,
            total_bytes=sum(item.size_bytes for item in ordered),
        )


def _inventory_state_is_remote(inventory: BackupSnapshotProjectInventory) -> bool:
    return _inventory_state_machine_alias(inventory) is not None


def _inventory_state_machine_alias(
    inventory: BackupSnapshotProjectInventory,
) -> str | None:
    recovery = inventory.recovery
    if recovery is None:
        return None
    configuration = recovery.configuration
    repositories = {repository.alias: repository for repository in configuration.repositories}
    machines = {machine.alias: machine for machine in configuration.machines}
    repository = repositories.get(configuration.state_repository)
    if repository is None:
        return None
    machine = machines.get(repository.machine)
    return machine.alias if machine is not None and machine.host else None


def read_backup_project_file_capture_receipt(
    path: Path,
    *,
    expected_sha256: str,
) -> BackupProjectFileCaptureReceipt:
    payload = read_immutable_backup_receipt(path, expected_sha256=expected_sha256)
    try:
        receipt = BackupProjectFileCaptureReceipt.model_validate_json(payload)
    except ValueError as exc:
        raise BackupCaptureUnavailable("The project-file capture receipt is invalid.") from exc
    if path.name != "project-files.json" or path.parent.name != f"backup-{receipt.capture_id}":
        raise BackupCaptureUnavailable(
            "The project-file receipt path is not bound to its capture identity."
        )
    return receipt


__all__ = [
    "BACKUP_PROJECT_FILE_CAPTURE_SCHEMA_VERSION",
    "BackupProjectFileCaptureCoordinator",
    "BackupProjectFileCapturePublication",
    "BackupProjectFileCaptureReceipt",
    "BackupProjectFileUnavailable",
    "read_backup_project_file_capture_receipt",
]
