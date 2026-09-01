from __future__ import annotations

import hashlib
import io
import os
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from rcp.projects import ProjectCatalog
from rcp.server_ops.backup import _write_deterministic_archive
from rcp.server_ops.backup_models import (
    BackupArchiveManifest,
    BackupFileEntry,
    BackupImportedProviderSourceCapture,
    BackupImportedProviderSourceInventory,
    BackupProjectCapture,
)
from rcp.server_ops.restore import _extract_verified_archive
from rcp.sources import ImportedProviderSourceStore
from rcp.transfer import TransferArchiveEntry


def _published_source(
    root: Path,
    project_id: str,
    *,
    payload: bytes = b'{"type":"assistant","text":"retained"}\n',
):
    data_dir = root / "data"
    data_dir.mkdir(parents=True, mode=0o700)
    digest = hashlib.sha256(payload).hexdigest()
    capture_root = root / "transfer-capture"
    source = capture_root / "provider-history" / "codex" / digest
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    entry = TransferArchiveEntry(
        archive_path=f"provider-history/codex/{digest}",
        group="provider_history",
        sha256=digest,
        size_bytes=len(payload),
    )
    owner = ImportedProviderSourceStore(data_dir, project_id)
    inventory = owner.publish(capture_root, (entry,))
    return data_dir, owner, inventory, payload


def _backup_source_capture(
    capture_root: Path,
    owner: ImportedProviderSourceStore,
    inventory,
) -> BackupImportedProviderSourceCapture:
    destination = capture_root / "project-sources" / owner.project_id / "provider-history"
    destination.parent.mkdir(parents=True, mode=0o700)
    destination.parent.parent.chmod(0o700)
    snapshot = owner.capture_snapshot(destination, expected_inventory=inventory)
    files = tuple(
        BackupFileEntry(
            archive_path=(
                f"project-sources/{owner.project_id}/provider-history/{item.relative_path}"
            ),
            source_relative_path=f"provider-history/{item.relative_path}",
            group="imported_provider_history",
            sha256=item.sha256,
            size_bytes=item.size_bytes,
        )
        for item in snapshot.files
    )
    return BackupImportedProviderSourceCapture(
        project_id=owner.project_id,
        inventory=BackupImportedProviderSourceInventory.model_validate(inventory.model_dump()),
        present=True,
        files=files,
        total_bytes=sum(item.size_bytes for item in files),
    )


def test_imported_sources_round_trip_through_declared_backup_and_restore_bytes(
    tmp_path: Path,
) -> None:
    project_id = str(uuid.uuid4())
    _data_dir, owner, inventory, payload = _published_source(tmp_path / "source", project_id)
    capture_root = tmp_path / "backup-capture"
    capture_root.mkdir(mode=0o700)
    sqlite_bytes = b"sqlite snapshot"
    (capture_root / "rcp.sqlite3").write_bytes(sqlite_bytes)
    imported = _backup_source_capture(capture_root, owner, inventory)
    space_id = str(uuid.uuid4())
    manifest = BackupArchiveManifest(
        space_id=space_id,
        space_name="Lifecycle lab",
        rcp_source_commit="a" * 40,
        database_schema_sha256="b" * 64,
        captured_at=datetime.now(UTC),
        sqlite_snapshot=BackupFileEntry(
            archive_path="database/rcp.sqlite3",
            source_relative_path="rcp.sqlite3",
            group="sqlite_snapshot",
            sha256=hashlib.sha256(sqlite_bytes).hexdigest(),
            size_bytes=len(sqlite_bytes),
        ),
        encryption_recipient_fingerprint="c" * 64,
        installation_id=str(uuid.uuid4()),
        excluded_app_data_entries=(),
        captured_app_data_entries=("project-sources",),
        uncaptured_app_data_entries=(),
        projects=(
            BackupProjectCapture(
                project_id=project_id,
                home_space_id=space_id,
                locator=None,
                status="uncaptured",
                unavailable_kind="inventory_failure",
                unavailable_reason="The project checkout was unavailable.",
                unavailable_at=datetime.now(UTC),
                total_bytes=0,
            ),
        ),
        imported_sources=(imported,),
        status="partial",
        total_bytes=len(sqlite_bytes) + imported.total_bytes,
    )

    archive_stream = io.BytesIO()
    _write_deterministic_archive(archive_stream, manifest, capture_root)
    archive_stream.seek(0)
    with tarfile.open(fileobj=archive_stream, mode="r:") as archive:
        names = [member.name for member in archive.getmembers()]
    assert names == [
        "manifest.json",
        "database/rcp.sqlite3",
        *sorted(entry.archive_path for entry in imported.files),
    ]

    plaintext = tmp_path / "archive.tar"
    plaintext.write_bytes(archive_stream.getvalue())
    payload_root = tmp_path / "restore-candidate" / "payload"
    payload_root.parent.mkdir(mode=0o700)
    restored_manifest, _digest = _extract_verified_archive(
        plaintext,
        payload_root,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    restored = restored_manifest.imported_sources[0]
    target_data = tmp_path / "target-data"
    target_data.mkdir(mode=0o700)
    target_owner = ImportedProviderSourceStore(target_data, project_id)
    target_inventory = target_owner.publish_snapshot(
        payload_root / "project-sources" / project_id / "provider-history",
        inventory,
    )

    assert target_inventory == inventory
    assert target_owner.inventory() == inventory
    assert (target_owner.root / "codex" / inventory.files[0].sha256).read_bytes() == payload
    assert restored.total_bytes == imported.total_bytes
    assert not any("native" in name or "credentials" in name for name in names)


@pytest.mark.parametrize("corruption", ["symlink", "fifo"])
def test_imported_source_snapshot_rejects_special_files(
    tmp_path: Path,
    corruption: str,
) -> None:
    project_id = str(uuid.uuid4())
    _data_dir, owner, inventory, _payload = _published_source(tmp_path, project_id)
    source = owner.root / "codex" / inventory.files[0].sha256
    source.unlink()
    if corruption == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        source.symlink_to(outside)
    else:
        os.mkfifo(source, 0o400)
    destination = tmp_path / "snapshot-parent"
    destination.mkdir(mode=0o700)

    with pytest.raises((OSError, ValueError)):
        owner.capture_snapshot(destination / "provider-history")

    assert not os.path.lexists(destination / "provider-history")


def test_imported_source_inventory_rejects_an_incomplete_owner_root(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    owner = ImportedProviderSourceStore(data_dir, str(uuid.uuid4()))
    owner.project_root.mkdir(parents=True, mode=0o700)

    with pytest.raises(ValueError, match="lacks its sealed provider history"):
        owner.inventory()


class _CleanupStore:
    space_kind = "team"

    def __init__(self, project_id: str, *, phase: str = "archive_bound") -> None:
        self.project_id = project_id
        self.phase = phase

    def project_aliases(self):
        return {}

    def project_transfer_request(self, request_id: str):
        return SimpleNamespace(
            request_id=request_id,
            side="target",
            phase=self.phase,
            project_id=self.project_id,
        )

    def project_provisioning_request(self, request_id: str):
        return SimpleNamespace(
            request_id=request_id,
            kind="incoming_transfer",
            proposed_project_id=self.project_id,
            status="operator_action_needed",
        )

    def project(self, _project_id: str):
        return None


def test_only_request_bound_pre_activation_cleanup_can_discard_imported_sources(
    tmp_path: Path,
) -> None:
    project_id = str(uuid.uuid4())
    data_dir, owner, inventory, _payload = _published_source(tmp_path, project_id)
    request_id = str(uuid.uuid4())
    store = _CleanupStore(project_id)
    catalog = ProjectCatalog(data_dir, store, object())  # type: ignore[arg-type]

    assert catalog.discard_unactivated_imported_sources(
        request_id,
        expected_inventory=inventory,
    )
    assert not os.path.lexists(owner.project_root)
    with pytest.raises(ValueError, match="Team projects cannot be deleted"):
        catalog.delete(project_id)

    _data_dir, owner, inventory, _payload = _published_source(tmp_path / "activated", project_id)
    activated = ProjectCatalog(
        _data_dir,
        _CleanupStore(project_id, phase="target_activated"),
        object(),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="outside its pending request"):
        activated.discard_unactivated_imported_sources(
            request_id,
            expected_inventory=inventory,
        )
    assert owner.inventory() == inventory
