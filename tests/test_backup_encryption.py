from __future__ import annotations

import hashlib
import io
import json
import shutil
import stat
import subprocess
import tarfile
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from rcp.__main__ import build_parser
from rcp.server_ops import backup as backup_owner
from rcp.server_ops.backup import (
    BackupArchiveReceipt,
    BackupRunOutcome,
    BackupRunRefused,
    LinuxBackupRunMachine,
    ProtectedBackupArchive,
    latest_protected_backup_receipt,
    prepare_backup_run_command,
    protect_backup_archive,
    read_backup_archive_receipt,
    read_backup_outcome,
    require_age_1x,
    write_backup_outcome,
)
from rcp.server_ops.backup_models import BackupArchiveManifest, BackupFileEntry
from rcp.server_ops.cli import CallerIdentity, run_server_command
from rcp.server_ops.config import (
    ServerBackupConfig,
    ServerSourceConfig,
    create_installed_server_config,
)

AGE_RECIPIENT = "age1qypqxpq9qcrsszg2pvxq6rs0zqg3yyc5z5tpwxqergd3c8g7rusqmwn7f2"
INSTALLATION_ID = "69726714-fee6-427f-8e1b-337350518beb"
SPACE_ID = "70994440-4c57-41b0-a2f6-8878856db969"
CAPTURE_ID = "9c59550a-9787-466a-9435-1e59f0a9803f"
CAPTURED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def _fake_age(tmp_path: Path, *, version: str = "1.1.1") -> Path:
    executable = tmp_path / f"fake-age-{version}"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        f"  echo 'v{version}'\n"
        "  exit 0\n"
        "fi\n"
        "printf 'age-encryption.org/v1\\n'\n"
        "cat\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def _capture(tmp_path: Path, *, capture_id: str = CAPTURE_ID) -> tuple[Path, bytes]:
    capture_root = tmp_path / "run-stage" / f"backup-{capture_id}"
    capture_root.mkdir(parents=True, mode=0o700)
    database = b"deterministic SQLite snapshot bytes\n"
    snapshot = capture_root / "rcp.sqlite3"
    snapshot.write_bytes(database)
    snapshot.chmod(0o400)
    return capture_root, database


def _installed(destination: Path, *, recipient: str = AGE_RECIPIENT):
    installed = create_installed_server_config(
        source=ServerSourceConfig(
            origin="https://github.com/openai/rcp",
            authentication="public",
        ),
        installation_id=INSTALLATION_ID,
    )
    return installed.model_copy(
        update={
            "backup": ServerBackupConfig(
                destination=str(destination),
                age_recipient=recipient,
            )
        }
    )


def _manifest(database: bytes, *, captured_at: datetime = CAPTURED_AT) -> BackupArchiveManifest:
    return BackupArchiveManifest(
        space_id=SPACE_ID,
        space_name="Backup Lab",
        rcp_source_commit="a" * 40,
        database_schema_sha256="b" * 64,
        captured_at=captured_at,
        sqlite_snapshot=BackupFileEntry(
            archive_path="database/rcp.sqlite3",
            source_relative_path="rcp.sqlite3",
            group="sqlite_snapshot",
            sha256=hashlib.sha256(database).hexdigest(),
            size_bytes=len(database),
        ),
        encryption_recipient_fingerprint=hashlib.sha256(AGE_RECIPIENT.encode("ascii")).hexdigest(),
        installation_id=INSTALLATION_ID,
        excluded_app_data_entries=("rcp.lock",),
        uncaptured_app_data_entries=(),
        projects=(),
        status="complete",
        total_bytes=len(database),
    )


def _fake_decrypt(path: Path) -> bytes:
    payload = path.read_bytes()
    header = b"age-encryption.org/v1\n"
    assert payload.startswith(header)
    return payload[len(header) :]


def _archive_receipt(destination: Path) -> BackupArchiveReceipt:
    return BackupArchiveReceipt(
        installation_id=INSTALLATION_ID,
        space_id=SPACE_ID,
        capture_id=CAPTURE_ID,
        destination=str(destination),
        archive_name=(f"rcp-team-backup-v1-20260829T120000000000Z-{CAPTURE_ID}.tar.age"),
        captured_at=CAPTURED_AT,
        protected_at=CAPTURED_AT + timedelta(minutes=1),
        capture_status="complete",
        age_version="1.1.1",
        age_recipient_fingerprint=hashlib.sha256(AGE_RECIPIENT.encode("ascii")).hexdigest(),
        archive_sha256="a" * 64,
        archive_size_bytes=4096,
        manifest_sha256="b" * 64,
        captured_bytes=2048,
        project_count=2,
        protected_project_count=2,
        uncaptured_project_count=0,
    )


def test_backup_streams_one_deterministic_tar_into_an_atomic_age_archive(tmp_path: Path) -> None:
    age = _fake_age(tmp_path)
    capture_root, database = _capture(tmp_path)
    first_destination = tmp_path / "first-backups"
    second_destination = tmp_path / "second-backups"
    first_destination.mkdir()
    second_destination.mkdir()
    manifest = _manifest(database)

    first = protect_backup_archive(
        installed=_installed(first_destination),
        manifest=manifest,
        capture_root=capture_root,
        age_version=require_age_1x(str(age)),
        age_executable=str(age),
        protected_at=CAPTURED_AT + timedelta(minutes=1),
    )
    second = protect_backup_archive(
        installed=_installed(second_destination),
        manifest=manifest,
        capture_root=capture_root,
        age_version=require_age_1x(str(age)),
        age_executable=str(age),
        protected_at=CAPTURED_AT + timedelta(minutes=2),
    )

    plaintext = _fake_decrypt(first.archive_path)
    assert plaintext == _fake_decrypt(second.archive_path)
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:") as archive:
        assert archive.getnames() == ["manifest.json", "database/rcp.sqlite3"]
        assert archive.extractfile("database/rcp.sqlite3").read() == database
        restored_manifest = json.load(archive.extractfile("manifest.json"))
    assert restored_manifest == manifest.model_dump(mode="json")
    assert first.receipt.capture_status == "complete"
    assert first.receipt.captured_bytes == len(database)
    assert first.receipt.readback == "passed"
    assert (
        first.receipt.archive_sha256 == hashlib.sha256(first.archive_path.read_bytes()).hexdigest()
    )
    assert stat.S_IMODE(first.archive_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(first.receipt_path.stat().st_mode) == 0o600
    assert not tuple(first_destination.glob("*.partial"))

    readback = read_backup_archive_receipt(
        first.receipt_path,
        expected_destination=first_destination,
        expected_installation_id=INSTALLATION_ID,
        expected_uid=first.archive_path.stat().st_uid,
        verify_digest=True,
    )
    assert readback == first.receipt


def test_backup_rejects_unsupported_age_and_changed_capture_bytes(tmp_path: Path) -> None:
    unsupported = _fake_age(tmp_path, version="2.0.0")
    with pytest.raises(BackupRunRefused, match="must be >=1.0.0,<2.0.0"):
        require_age_1x(str(unsupported))

    age = _fake_age(tmp_path, version="1.2.3")
    capture_root, database = _capture(tmp_path)
    destination = tmp_path / "backups"
    destination.mkdir()
    snapshot = capture_root / "rcp.sqlite3"
    snapshot.chmod(0o600)
    snapshot.write_bytes(b"changed after manifest\n")
    snapshot.chmod(0o400)
    with pytest.raises(BackupRunRefused, match="changed before archive streaming"):
        protect_backup_archive(
            installed=_installed(destination),
            manifest=_manifest(database),
            capture_root=capture_root,
            age_version=require_age_1x(str(age)),
            age_executable=str(age),
        )
    assert not tuple(destination.glob("*.tar.age"))
    assert not tuple(destination.glob("*.receipt.json"))


def test_last_backup_outcome_is_atomically_replaceable_machine_status(tmp_path: Path) -> None:
    age = _fake_age(tmp_path)
    capture_root, database = _capture(tmp_path)
    destination = tmp_path / "backups"
    destination.mkdir()
    protected = protect_backup_archive(
        installed=_installed(destination),
        manifest=_manifest(database),
        capture_root=capture_root,
        age_version="1.1.1",
        age_executable=str(age),
        protected_at=CAPTURED_AT + timedelta(minutes=1),
    )
    layout = SimpleNamespace(server_root=tmp_path)
    successful = BackupRunOutcome(
        operation_id="cf4d29d0-a1bd-4d38-8620-242adf195bf6",
        installation_id=INSTALLATION_ID,
        destination=str(destination),
        started_at=CAPTURED_AT,
        completed_at=CAPTURED_AT + timedelta(minutes=2),
        status="protected",
        archive=protected.receipt,
        archive_receipt_sha256=protected.receipt_sha256,
    )

    write_backup_outcome(successful, layout)
    assert read_backup_outcome(layout) == successful
    assert stat.S_IMODE((tmp_path / "backup-status.json").stat().st_mode) == 0o600

    failed = BackupRunOutcome(
        operation_id="f0b1cd24-a735-43f8-a184-2f4ba933934b",
        installation_id=INSTALLATION_ID,
        destination=str(destination),
        started_at=CAPTURED_AT + timedelta(days=1),
        completed_at=CAPTURED_AT + timedelta(days=1, minutes=1),
        status="failure",
        failure="The configured destination was unavailable.",
    )
    write_backup_outcome(failed, layout)
    assert read_backup_outcome(layout) == failed
    assert (
        latest_protected_backup_receipt(
            destination,
            installation_id=INSTALLATION_ID,
            expected_uid=destination.stat().st_uid,
        )
        == protected.receipt
    )

    corrupt = destination / (
        "rcp-team-backup-v1-20260830T120000000000Z-"
        "1417a462-8b46-45f8-8882-69a216718258.tar.age.receipt.json"
    )
    corrupt.write_text("not a receipt", encoding="utf-8")
    corrupt.chmod(0o600)
    with pytest.raises(BackupRunRefused, match="archive receipt is invalid"):
        latest_protected_backup_receipt(
            destination,
            installation_id=INSTALLATION_ID,
            expected_uid=destination.stat().st_uid,
        )


def test_backup_run_composes_capture_protection_retention_and_stage_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    server_root = tmp_path / "server"
    destination = tmp_path / "backups"
    capture_root = data_dir / "run-stage" / f"backup-{CAPTURE_ID}"
    capture_root.mkdir(parents=True, mode=0o700)
    server_root.mkdir()
    destination.mkdir()
    installed = _installed(destination)
    sqlite_receipt = object()
    project_publication = object()
    manifest = object()
    receipt = _archive_receipt(destination)
    protected = ProtectedBackupArchive(
        receipt=receipt,
        archive_path=destination / receipt.archive_name,
        receipt_path=destination / f"{receipt.archive_name}.receipt.json",
        receipt_sha256="c" * 64,
    )
    deleted_name = (
        "rcp-team-backup-v1-20260828T120000000000Z-cf4d29d0-a1bd-4d38-8620-242adf195bf6.tar.age"
    )
    events: list[str] = []

    class Control:
        def capture_backup_sqlite(self):
            events.append("sqlite")
            return SimpleNamespace(
                receipt_path=str(capture_root / "sqlite-capture.json"),
                receipt_sha256="d" * 64,
                capture_id=CAPTURE_ID,
            )

    class ProjectCapture:
        def __init__(self, observed_data_dir: Path) -> None:
            assert observed_data_dir == data_dir

        def capture(self, receipt_path: Path, *, expected_sha256: str):
            assert receipt_path == capture_root / "sqlite-capture.json"
            assert expected_sha256 == "d" * 64
            events.append("projects")
            return project_publication

    monkeypatch.setattr(backup_owner, "_load_backup_configuration", lambda _layout: installed)
    monkeypatch.setattr(backup_owner, "require_age_1x", lambda _executable: "1.1.1")
    monkeypatch.setattr(backup_owner, "BackupProjectFileCaptureCoordinator", ProjectCapture)
    monkeypatch.setattr(
        backup_owner,
        "read_backup_sqlite_capture_receipt",
        lambda *_args, **_kwargs: sqlite_receipt,
    )

    def build_manifest(**kwargs):
        assert kwargs == {
            "installed": installed,
            "sqlite_receipt": sqlite_receipt,
            "project_publication": project_publication,
        }
        events.append("manifest")
        return manifest

    def protect(**kwargs):
        assert kwargs["installed"] == installed
        assert kwargs["manifest"] is manifest
        assert kwargs["capture_root"] == capture_root
        assert kwargs["age_version"] == "1.1.1"
        events.append("protect")
        return protected

    monkeypatch.setattr(backup_owner, "build_archive_manifest", build_manifest)
    monkeypatch.setattr(backup_owner, "protect_backup_archive", protect)
    monkeypatch.setattr(
        backup_owner,
        "plan_backup_retention",
        lambda *_args, **_kwargs: events.append("plan") or object(),
    )
    monkeypatch.setattr(
        backup_owner,
        "apply_backup_retention",
        lambda *_args, **_kwargs: events.append("retain") or (deleted_name,),
    )
    times = iter(
        (
            CAPTURED_AT,
            CAPTURED_AT + timedelta(minutes=1),
            CAPTURED_AT + timedelta(minutes=2),
        )
    )
    restore_operations_root = server_root / "restore-operations"
    restore_operations_root.mkdir()
    restore_operations_root.chmod(0o700)
    layout = SimpleNamespace(
        data_dir=data_dir,
        server_root=server_root,
        restore_operations_root=restore_operations_root,
    )

    outcome = LinuxBackupRunMachine(
        layout,
        control=Control(),
        clock=lambda: next(times),
    ).run()

    assert events == ["sqlite", "projects", "manifest", "protect", "plan", "retain"]
    assert outcome.status == "protected"
    assert outcome.archive == receipt
    assert outcome.retention_deleted_archives == (deleted_name,)
    assert read_backup_outcome(layout) == outcome
    assert not capture_root.exists()


def test_backup_run_publishes_a_durable_failure_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server_root = tmp_path / "server"
    destination = tmp_path / "backups"
    server_root.mkdir()
    destination.mkdir()
    installed = _installed(destination)
    restore_operations_root = server_root / "restore-operations"
    restore_operations_root.mkdir()
    restore_operations_root.chmod(0o700)
    layout = SimpleNamespace(
        data_dir=tmp_path / "data",
        server_root=server_root,
        restore_operations_root=restore_operations_root,
    )
    monkeypatch.setattr(backup_owner, "_load_backup_configuration", lambda _layout: installed)
    monkeypatch.setattr(
        backup_owner,
        "require_age_1x",
        lambda _executable: (_ for _ in ()).throw(
            BackupRunRefused("The configured age executable is unavailable.")
        ),
    )

    with pytest.raises(BackupRunRefused, match="age executable is unavailable"):
        LinuxBackupRunMachine(layout, clock=lambda: CAPTURED_AT).run()

    outcome = read_backup_outcome(layout)
    assert outcome.status == "failure"
    assert outcome.failure == "The configured age executable is unavailable."
    assert outcome.archive is None


def test_backup_run_cli_reports_the_exact_protected_outcome(tmp_path: Path) -> None:
    destination = tmp_path / "backups"
    destination.mkdir()
    receipt = _archive_receipt(destination)
    outcome = BackupRunOutcome(
        operation_id="cf4d29d0-a1bd-4d38-8620-242adf195bf6",
        installation_id=INSTALLATION_ID,
        destination=str(destination),
        started_at=CAPTURED_AT,
        completed_at=CAPTURED_AT + timedelta(minutes=2),
        status="protected",
        archive=receipt,
        archive_receipt_sha256="c" * 64,
    )
    machine = SimpleNamespace(run=lambda: outcome)
    args = build_parser().parse_args(("server", "backup", "run", "--machine-readable"))
    output = StringIO()

    exit_code = run_server_command(
        args,
        handler=lambda request, identity: prepare_backup_run_command(
            request,
            identity,
            machine=machine,
        ),
        identity=CallerIdentity(uid=501, username="rcp", host="lab.example"),
        stream=output,
    )

    assert exit_code == 0
    final = json.loads(output.getvalue().splitlines()[-1])["step"]
    assert final["state"] == "succeeded"
    fields = {field["name"]: field["value"] for field in final["fields"]}
    assert fields["backup_status"] == "protected"
    assert fields["archive_sha256"] == "a" * 64
    assert fields["protected_projects"] == 2


@pytest.mark.skipif(
    shutil.which("age") is None or shutil.which("age-keygen") is None,
    reason="real upstream age tools are not installed on this development host",
)
def test_real_age_1x_encrypts_and_decrypts_the_archive(tmp_path: Path) -> None:
    identity = tmp_path / "recovery.agekey"
    subprocess.run(
        ("age-keygen", "-o", str(identity)),
        check=True,
        capture_output=True,
        text=True,
    )
    recipient = subprocess.run(
        ("age-keygen", "-y", str(identity)),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    capture_root, database = _capture(tmp_path)
    destination = tmp_path / "backups"
    destination.mkdir()
    manifest = _manifest(database).model_copy(
        update={
            "encryption_recipient_fingerprint": hashlib.sha256(
                recipient.encode("ascii")
            ).hexdigest()
        }
    )

    protected = protect_backup_archive(
        installed=_installed(destination, recipient=recipient),
        manifest=manifest,
        capture_root=capture_root,
        age_version=require_age_1x(),
    )
    decrypted = subprocess.run(
        ("age", "--decrypt", "--identity", str(identity), str(protected.archive_path)),
        check=True,
        capture_output=True,
    ).stdout

    with tarfile.open(fileobj=io.BytesIO(decrypted), mode="r:") as archive:
        assert archive.extractfile("database/rcp.sqlite3").read() == database
        assert json.load(archive.extractfile("manifest.json"))["space_id"] == SPACE_ID
