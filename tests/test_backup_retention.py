from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rcp.server_ops.backup import (
    BackupRunRefused,
    apply_backup_retention,
    plan_backup_retention,
    protect_backup_archive,
)
from rcp.server_ops.config import ServerBackupConfig

from .test_backup_encryption import (
    CAPTURED_AT,
    INSTALLATION_ID,
    _capture,
    _fake_age,
    _installed,
    _manifest,
)


def _protect(
    tmp_path: Path,
    destination: Path,
    age: Path,
    *,
    captured_at: datetime,
    status: str,
):
    capture_id = str(uuid.uuid4())
    capture_root, database = _capture(tmp_path / capture_id, capture_id=capture_id)
    manifest = _manifest(database, captured_at=captured_at)
    if status == "partial":
        manifest = manifest.model_copy(
            update={
                "uncaptured_app_data_entries": ("unknown-root",),
                "status": "partial",
            }
        )
    return protect_backup_archive(
        installed=_installed(destination),
        manifest=manifest,
        capture_root=capture_root,
        age_version="1.1.1",
        age_executable=str(age),
        protected_at=captured_at + timedelta(minutes=1),
    )


def test_retention_keeps_the_window_and_newest_complete_archive(tmp_path: Path) -> None:
    age = _fake_age(tmp_path)
    destination = tmp_path / "backups"
    destination.mkdir()
    old_complete = _protect(
        tmp_path,
        destination,
        age,
        captured_at=CAPTURED_AT,
        status="complete",
    )
    old_partial = _protect(
        tmp_path,
        destination,
        age,
        captured_at=CAPTURED_AT + timedelta(days=1),
        status="partial",
    )
    new_partial = _protect(
        tmp_path,
        destination,
        age,
        captured_at=CAPTURED_AT + timedelta(days=2),
        status="partial",
    )
    newest_partial = _protect(
        tmp_path,
        destination,
        age,
        captured_at=CAPTURED_AT + timedelta(days=3),
        status="partial",
    )
    config = ServerBackupConfig(
        destination=str(destination),
        age_recipient=_installed(destination).backup.age_recipient,
        retention=2,
    )

    plan = plan_backup_retention(
        config,
        installation_id=INSTALLATION_ID,
        expected_uid=os.geteuid(),
    )

    assert set(plan.kept_archives) == {
        old_complete.receipt.archive_name,
        new_partial.receipt.archive_name,
        newest_partial.receipt.archive_name,
    }
    assert plan.delete_archives == (old_partial.receipt.archive_name,)
    assert old_partial.archive_path.exists()

    deleted = apply_backup_retention(
        plan,
        installation_id=INSTALLATION_ID,
        expected_uid=os.geteuid(),
    )

    assert deleted == plan.delete_archives
    assert not old_partial.archive_path.exists()
    assert not old_partial.receipt_path.exists()
    assert old_complete.archive_path.exists()
    assert new_partial.archive_path.exists()
    assert newest_partial.archive_path.exists()


def test_retention_ignores_unproven_files_and_rechecks_before_deletion(tmp_path: Path) -> None:
    age = _fake_age(tmp_path)
    destination = tmp_path / "backups"
    destination.mkdir()
    older = _protect(
        tmp_path,
        destination,
        age,
        captured_at=CAPTURED_AT,
        status="complete",
    )
    _protect(
        tmp_path,
        destination,
        age,
        captured_at=CAPTURED_AT + timedelta(days=1),
        status="complete",
    )
    unknown = destination / "human-notes.tar.age"
    unknown.write_bytes(b"do not delete\n")
    forged = (
        destination
        / "rcp-team-backup-v1-20260829T120000000000Z-9c59550a-9787-466a-9435-1e59f0a9803f.tar.age"
    )
    forged.write_bytes(b"not age\n")
    config = ServerBackupConfig(
        destination=str(destination),
        age_recipient=_installed(destination).backup.age_recipient,
        retention=1,
    )
    plan = plan_backup_retention(
        config,
        installation_id=INSTALLATION_ID,
        expected_uid=os.geteuid(),
    )
    assert plan.delete_archives == (older.receipt.archive_name,)

    with older.archive_path.open("ab") as stream:
        stream.write(b"changed after preview")
    with pytest.raises(BackupRunRefused, match="no longer matches"):
        apply_backup_retention(
            plan,
            installation_id=INSTALLATION_ID,
            expected_uid=os.geteuid(),
        )

    assert older.archive_path.exists()
    assert older.receipt_path.exists()
    assert unknown.read_bytes() == b"do not delete\n"
    assert forged.read_bytes() == b"not age\n"
