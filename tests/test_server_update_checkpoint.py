from __future__ import annotations

import hashlib
import io
import json
import os
import pwd
import sqlite3
import stat
import subprocess
import sys
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import rcp.server_ops.update_checkpoint as update_checkpoint_module
from rcp.attachments import ChatAttachmentStore
from rcp.config import load_manifest
from rcp.core.transition_models import GraphHeadRef
from rcp.runs.shared import checkpoint_local_recovery_stages
from rcp.server_ops.backup_capture import (
    BackupCaptureCoordinator,
    BackupSnapshotProjectInventory,
    write_immutable_backup_receipt,
)
from rcp.server_ops.backup_models import (
    BackupCheckoutRecoveryDescriptor,
    BackupFileEntry,
    BackupImportedProviderSourceCapture,
    BackupImportedProviderSourceInventory,
    BackupManifestConfiguration,
    BackupProjectCapture,
    BackupRecoveryMachine,
    BackupRecoveryRepository,
)
from rcp.server_ops.backup_project_files import (
    BackupProjectFileCaptureCoordinator,
    BackupProjectFileCaptureReceipt,
)
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.rehearsal import (
    CandidateProjectVerification,
    StartupRecoveryReadModel,
    VerifiedCandidateReceipt,
    verified_candidate_receipt_path,
)
from rcp.server_ops.update_checkpoint import (
    UpdateCheckpointCoordinator,
    UpdateCheckpointRefused,
    read_rollback_journal,
    restore_update_checkpoint,
)
from rcp.server_runtime import ServerMetadata
from rcp.sources import ImportedProviderSourceStore
from rcp.storage import AppStore
from rcp.storage.models import (
    ProjectTransferUploadCompleteReceipt,
    ProjectTransferUploadRecord,
)
from rcp.transfer import TransferArchiveEntry
from rcp.transfer.target import target_transfer_archive_path

BASE_COMMIT = "a" * 40
CANDIDATE_COMMIT = "b" * 40
WEB_BUILD_ID = "sha256:" + ("c" * 64)


def _metadata(data_dir: Path) -> ServerMetadata:
    return ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=data_dir / "control.sock",
        running_commit=BASE_COMMIT,
        web_build_id=WEB_BUILD_ID,
    )


def _write_private_model(path: Path, model: VerifiedCandidateReceipt) -> None:
    payload = (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _SQLiteStageStore:
    def __init__(self, database: Path) -> None:
        self.database = database

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _checkpoint_fixture(
    tmp_path: Path,
    *,
    project_sqlite_receipt_sha256: str | None = None,
):
    data_dir = tmp_path / "server" / "data"
    data_dir.mkdir(parents=True, mode=0o700)
    update_root = tmp_path / "server" / "update-checkpoints"
    update_root.mkdir(mode=0o700)
    releases = tmp_path / "server" / "releases"
    previous_release = releases / BASE_COMMIT
    candidate_release = releases / CANDIDATE_COMMIT
    previous_release.mkdir(parents=True, mode=0o700)
    candidate_release.mkdir(mode=0o700)

    store, _bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Test Lab")
    bootstrap_root = data_dir / "bootstrap-manifests"
    bootstrap_root.mkdir(mode=0o700)
    (bootstrap_root / "remote-project.toml").write_bytes(b'name = "Remote fixture"\n')
    project_snapshots_root = data_dir / "project-snapshots"
    project_snapshots_root.mkdir(mode=0o700)
    (project_snapshots_root / "project.json").write_bytes(b'{"graph":{"revision":0}}\n')
    attachment_store = ChatAttachmentStore(data_dir / "chat-attachments")
    attachment = attachment_store.add(
        project_id=str(uuid.uuid4()),
        chat_id=str(uuid.uuid4()),
        client_id=str(uuid.uuid4()),
        filename="notes.txt",
        media_type="text/plain",
        source=io.BytesIO(b"pre-cutover attachment\n"),
    )

    sqlite_publication = BackupCaptureCoordinator(
        store, data_dir, _metadata(data_dir)
    ).capture_sqlite()
    project_publication = BackupProjectFileCaptureCoordinator(data_dir).capture(
        sqlite_publication.receipt_path,
        expected_sha256=sqlite_publication.receipt_sha256,
    )
    project_receipt_sha256 = project_publication.receipt_sha256
    if project_sqlite_receipt_sha256 is not None:
        project_publication.receipt_path.unlink()
        project_receipt_sha256 = write_immutable_backup_receipt(
            project_publication.receipt_path,
            project_publication.receipt.model_copy(
                update={"sqlite_receipt_sha256": project_sqlite_receipt_sha256}
            ),
        )
    candidate_path = verified_candidate_receipt_path(
        CANDIDATE_COMMIT,
        sqlite_publication.receipt.capture_id,
        update_root,
    )
    candidate = VerifiedCandidateReceipt(
        installation_id=str(uuid.uuid4()),
        candidate_commit=CANDIDATE_COMMIT,
        base_current_commit=BASE_COMMIT,
        base_running_commit=BASE_COMMIT,
        base_instance_id=str(uuid.uuid4()),
        base_process_pid=421,
        release_path=str(candidate_release),
        built_receipt_path=str(update_root / f"built-candidate-{CANDIDATE_COMMIT}.json"),
        built_receipt_sha256="d" * 64,
        receipt_path=str(candidate_path),
        web_build_id=WEB_BUILD_ID,
        capture_id=sqlite_publication.receipt.capture_id,
        sqlite_snapshot_sha256=sqlite_publication.receipt.sqlite_snapshot.sha256,
        project_capture_sha256=project_receipt_sha256,
        space_id=store.space_id,
        projects=(),
        startup_recovery=StartupRecoveryReadModel(
            active_operation_ids=(),
            stopping_experiment_operation_ids=(),
            report_episode_ids=(),
            auto_research_recovery_operation_ids=(),
            active_watcher_ids=(),
        ),
        reads=("/api/health", "/api/projects"),
        verified_at=datetime.now(UTC),
    )
    _write_private_model(candidate_path, candidate)
    checkpoint = UpdateCheckpointCoordinator(
        data_dir=data_dir,
        update_root=update_root,
        previous_release_path=previous_release,
        expected_uid=os.geteuid(),
    ).create(
        sqlite_receipt_path=sqlite_publication.receipt_path,
        sqlite_receipt_sha256=sqlite_publication.receipt_sha256,
        project_receipt_path=project_publication.receipt_path,
        project_receipt_sha256=project_receipt_sha256,
        candidate_receipt_path=candidate_path,
        candidate_receipt_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    )
    attachment_file = (
        data_dir
        / "chat-attachments"
        / attachment.attachment_set_id
        / "files"
        / next(
            path.name
            for path in (
                data_dir / "chat-attachments" / attachment.attachment_set_id / "files"
            ).iterdir()
        )
    )
    return checkpoint, data_dir, attachment_file


def _local_project_checkpoint_fixture(tmp_path: Path):
    data_dir = tmp_path / "server" / "data"
    data_dir.mkdir(parents=True, mode=0o700)
    update_root = tmp_path / "server" / "update-checkpoints"
    update_root.mkdir(mode=0o700)
    previous_release = tmp_path / "server" / "releases" / BASE_COMMIT
    candidate_release = tmp_path / "server" / "releases" / CANDIDATE_COMMIT
    previous_release.mkdir(parents=True)
    candidate_release.mkdir()
    store, _bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Local Lab")
    publication = BackupCaptureCoordinator(store, data_dir, _metadata(data_dir)).capture_sqlite()

    project_id = str(uuid.uuid4())
    imported_payload = b'{"type":"assistant","text":"checkpoint source"}\n'
    imported_digest = hashlib.sha256(imported_payload).hexdigest()
    imported_capture_root = tmp_path / "imported-capture"
    imported_source = imported_capture_root / "provider-history" / "codex" / imported_digest
    imported_source.parent.mkdir(parents=True, mode=0o700)
    imported_source.write_bytes(imported_payload)
    imported_owner = ImportedProviderSourceStore(data_dir, project_id)
    imported_inventory = imported_owner.publish(
        imported_capture_root,
        (
            TransferArchiveEntry(
                archive_path=f"provider-history/codex/{imported_digest}",
                group="provider_history",
                sha256=imported_digest,
                size_bytes=len(imported_payload),
            ),
        ),
    )
    central_root = tmp_path / "server" / "projects"
    repository = central_root / project_id / "repositories" / "repo"
    research = repository / ".research"
    research.mkdir(parents=True)
    account = pwd.getpwuid(os.geteuid()).pw_name
    provider_home = tmp_path / "provider-home"
    provider_home.mkdir()
    manifest_path = research / "manifest.toml"
    manifest_path.write_text(
        f'''name = "Local checkpoint project"

[[machines]]
alias = "server"
host = ""
os_account = "{account}"
provider_paths = {{ codex = "{provider_home}" }}

[[repositories]]
alias = "repo"
machine = "server"
path = "{repository}"

[project]
truth_scope = ["repo"]

[state]
repository = "repo"

[agent]
default_run_truth_scope = ["repo"]

[sources]
claude_roots = ["{provider_home / "claude"}"]
codex_roots = ["{provider_home / "codex"}"]

[execution]
run_on = "server"

[paper.coach]
default_provider = "codex"
default_model = ""
default_reasoning = "medium"
''',
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()
    configuration = BackupManifestConfiguration.from_manifest(load_manifest(manifest_path))
    recovery = BackupCheckoutRecoveryDescriptor(
        request_id=str(uuid.uuid4()),
        project_id=project_id,
        home_space_id=store.space_id,
        completed_at=datetime.now(UTC),
        final_review_digest="e" * 64,
        configuration=configuration,
        configuration_sha256=configuration.sha256,
        machines=(
            BackupRecoveryMachine(
                alias="server",
                location="local",
                host="",
                os_account=account,
                resolved_central_root=str(central_root),
            ),
        ),
        repositories=(
            BackupRecoveryRepository(
                alias="repo",
                repository=parse_github_repository_ref("git@github.com:openai/rcp.git"),
                machine_alias="server",
                resolved_path=str(repository),
                git_commit="f" * 40,
                deploy_key_label=f"rcp:{store.space_id}:{project_id}:repo",
                public_key_fingerprint="SHA256:" + ("A" * 43),
            ),
        ),
    )
    captured_at = publication.receipt.captured_at
    inventory = BackupSnapshotProjectInventory(
        project_id=project_id,
        home_space_id=store.space_id,
        locator=str(manifest_path),
        status="capturable",
        recovery=recovery,
    )
    sqlite_receipt = publication.receipt.model_copy(
        update={
            "app_data_plan": publication.receipt.app_data_plan.model_copy(
                update={"captured_entries": ("project-sources",)}
            ),
            "projects": (inventory,),
            "imported_source_inventories": (
                BackupImportedProviderSourceInventory.model_validate(
                    imported_inventory.model_dump()
                ),
            ),
            "status": "complete",
        }
    )
    publication.receipt_path.unlink()
    sqlite_sha256 = write_immutable_backup_receipt(publication.receipt_path, sqlite_receipt)
    project_root = publication.receipt_path.parent / "projects" / project_id
    captured_manifest = project_root / ".research" / "manifest.toml"
    captured_manifest.parent.mkdir(parents=True)
    captured_manifest.write_bytes(original_manifest)
    captured_manifest.chmod(0o400)
    entry = BackupFileEntry(
        archive_path=f"projects/{project_id}/.research/manifest.toml",
        source_relative_path=".research/manifest.toml",
        group="canonical",
        sha256=hashlib.sha256(original_manifest).hexdigest(),
        size_bytes=len(original_manifest),
    )
    project_capture = BackupProjectCapture(
        project_id=project_id,
        home_space_id=store.space_id,
        locator=str(manifest_path),
        status="captured",
        main_head=GraphHeadRef(revision=0),
        files=(entry,),
        recovery=recovery,
        total_bytes=len(original_manifest),
    )
    imported_collection = publication.receipt_path.parent / "project-sources"
    imported_collection.mkdir(mode=0o700)
    imported_project_root = imported_collection / project_id
    imported_project_root.mkdir(mode=0o700)
    imported_snapshot = imported_owner.capture_snapshot(
        imported_project_root / "provider-history",
        expected_inventory=imported_inventory,
    )
    imported_files = tuple(
        BackupFileEntry(
            archive_path=(f"project-sources/{project_id}/provider-history/{item.relative_path}"),
            source_relative_path=f"provider-history/{item.relative_path}",
            group="imported_provider_history",
            sha256=item.sha256,
            size_bytes=item.size_bytes,
        )
        for item in imported_snapshot.files
    )
    imported_capture = BackupImportedProviderSourceCapture(
        project_id=project_id,
        inventory=BackupImportedProviderSourceInventory.model_validate(
            imported_inventory.model_dump()
        ),
        present=True,
        files=imported_files,
        total_bytes=sum(item.size_bytes for item in imported_files),
    )
    project_receipt = BackupProjectFileCaptureReceipt(
        capture_id=sqlite_receipt.capture_id,
        captured_at=captured_at,
        completed_at=datetime.now(UTC),
        rcp_source_commit=BASE_COMMIT,
        space_id=store.space_id,
        sqlite_receipt_sha256=sqlite_sha256,
        sqlite_snapshot_sha256=sqlite_receipt.sqlite_snapshot.sha256,
        sqlite_capture_status="complete",
        projects=(project_capture,),
        imported_sources=(imported_capture,),
        status="complete",
    )
    project_receipt_path = publication.receipt_path.parent / "project-files.json"
    project_sha256 = write_immutable_backup_receipt(project_receipt_path, project_receipt)
    candidate_path = verified_candidate_receipt_path(
        CANDIDATE_COMMIT,
        sqlite_receipt.capture_id,
        update_root,
    )
    candidate = VerifiedCandidateReceipt(
        installation_id=str(uuid.uuid4()),
        candidate_commit=CANDIDATE_COMMIT,
        base_current_commit=BASE_COMMIT,
        base_running_commit=BASE_COMMIT,
        base_instance_id=str(uuid.uuid4()),
        base_process_pid=421,
        release_path=str(candidate_release),
        built_receipt_path=str(update_root / "built.json"),
        built_receipt_sha256="d" * 64,
        receipt_path=str(candidate_path),
        web_build_id=WEB_BUILD_ID,
        capture_id=sqlite_receipt.capture_id,
        sqlite_snapshot_sha256=sqlite_receipt.sqlite_snapshot.sha256,
        project_capture_sha256=project_sha256,
        space_id=store.space_id,
        projects=(
            CandidateProjectVerification(
                project_id=project_id,
                status="verified",
                revision=0,
                projection_sha256="1" * 64,
            ),
        ),
        startup_recovery=StartupRecoveryReadModel(
            active_operation_ids=(),
            stopping_experiment_operation_ids=(),
            report_episode_ids=(),
            auto_research_recovery_operation_ids=(),
            active_watcher_ids=(),
        ),
        reads=("/api/health", "/api/projects"),
        verified_at=datetime.now(UTC),
    )
    _write_private_model(candidate_path, candidate)
    checkpoint = UpdateCheckpointCoordinator(
        data_dir=data_dir,
        update_root=update_root,
        previous_release_path=previous_release,
        expected_uid=os.geteuid(),
    ).create(
        sqlite_receipt_path=publication.receipt_path,
        sqlite_receipt_sha256=sqlite_sha256,
        project_receipt_path=project_receipt_path,
        project_receipt_sha256=project_sha256,
        candidate_receipt_path=candidate_path,
        candidate_receipt_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    )
    return (
        checkpoint,
        research,
        original_manifest,
        imported_owner,
        imported_inventory,
        imported_payload,
    )


def test_local_recovery_stage_inventory_uses_the_task_ledger_and_ignores_remote(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    local = data_dir / "run-stage" / "local-stage"
    local.mkdir(parents=True)
    database = tmp_path / "stages.sqlite3"
    local_operation = str(uuid.uuid4())
    local_episode = str(uuid.uuid4())
    remote_view = uuid.uuid4().hex[:24]
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE graph_runs "
            "(operation_id TEXT, status TEXT, stage_host TEXT, stage_root TEXT)"
        )
        connection.execute(
            "CREATE TABLE result_views (view_id TEXT, stage_host TEXT, stage_root TEXT)"
        )
        connection.execute(
            "CREATE TABLE experiment_episode_state "
            "(episode_id TEXT, stage_host TEXT, stage_root TEXT)"
        )
        connection.execute("CREATE TABLE episodes (episode_id TEXT, status TEXT)")
        connection.execute(
            "INSERT INTO graph_runs VALUES (?, ?, ?, ?)",
            (local_operation, "running", None, str(local)),
        )
        connection.execute(
            "INSERT INTO result_views VALUES (?, ?, ?)",
            (remote_view, "gpu.example", "/srv/rcp/run-stage/remote-stage"),
        )
        connection.execute(
            "INSERT INTO experiment_episode_state VALUES (?, ?, ?)",
            (local_episode, "", str(local)),
        )
        connection.execute(
            "INSERT INTO episodes VALUES (?, ?)",
            (local_episode, "running"),
        )

    inventory = checkpoint_local_recovery_stages(_SQLiteStageStore(database), data_dir)

    assert [item.root for item in inventory] == [local]
    assert inventory[0].owner_refs == (
        f"experiment_episode_state:{local_episode}",
        f"graph_runs:{local_operation}",
    )


def test_local_recovery_stage_inventory_ignores_retention_swept_history(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    missing_task = data_dir / "run-stage" / "old-task"
    missing_view = data_dir / "run-stage" / "old-result-view"
    database = tmp_path / "stages.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE graph_runs "
            "(operation_id TEXT, status TEXT, stage_host TEXT, stage_root TEXT)"
        )
        connection.execute(
            "CREATE TABLE result_views (view_id TEXT, stage_host TEXT, stage_root TEXT)"
        )
        connection.execute(
            "INSERT INTO graph_runs VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), "succeeded", "", str(missing_task)),
        )
        connection.execute(
            "INSERT INTO result_views VALUES (?, ?, ?)",
            (uuid.uuid4().hex[:24], "", str(missing_view)),
        )

    assert checkpoint_local_recovery_stages(_SQLiteStageStore(database), data_dir) == ()


def test_local_recovery_stage_inventory_requires_active_episode_stage(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    episode_id = str(uuid.uuid4())
    missing = data_dir / "run-stage" / "active-episode"
    database = tmp_path / "stages.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE episodes (episode_id TEXT, status TEXT)")
        connection.execute(
            "CREATE TABLE experiment_episode_state "
            "(episode_id TEXT, stage_host TEXT, stage_root TEXT)"
        )
        connection.execute("INSERT INTO episodes VALUES (?, ?)", (episode_id, "running"))
        connection.execute(
            "INSERT INTO experiment_episode_state VALUES (?, ?, ?)",
            (episode_id, "", str(missing)),
        )

    with pytest.raises(ValueError, match="recovery-critical local run stage"):
        checkpoint_local_recovery_stages(_SQLiteStageStore(database), data_dir)


def test_checkpoint_captures_attachment_bytes_and_excludes_runtime_debris(
    tmp_path: Path,
) -> None:
    checkpoint, _data_dir, _attachment_file = _checkpoint_fixture(tmp_path)

    app_root = next(root for root in checkpoint.roots if root.kind == "app_data")
    paths = {item.relative_path for item in app_root.files}
    assert "rcp.sqlite3" in paths
    assert "bootstrap-manifests/remote-project.toml" in paths
    assert "project-snapshots/project.json" in paths
    assert any(path.startswith("chat-attachments/") for path in paths)
    assert not any(path.endswith("rcp-server.json") or path.endswith("rcp.lock") for path in paths)
    assert checkpoint.status == "verified"
    assert Path(checkpoint.manifest_path).stat().st_mode & 0o777 == 0o600


def test_checkpoint_rejects_project_receipt_bound_to_another_sqlite_receipt(
    tmp_path: Path,
) -> None:
    with pytest.raises(UpdateCheckpointRefused, match="one exact final boundary"):
        _checkpoint_fixture(tmp_path, project_sqlite_receipt_sha256="e" * 64)


def test_checkpoint_publication_fsyncs_its_parent_before_and_after_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update_root = tmp_path / "server" / "update-checkpoints"
    observed: list[bool] = []
    original = update_checkpoint_module._fsync_directory

    def track(path: Path) -> None:
        if path == update_root:
            observed.append(bool(list(update_root.glob("checkpoint-*/checkpoint.json"))))
        original(path)

    monkeypatch.setattr(update_checkpoint_module, "_fsync_directory", track)

    _checkpoint_fixture(tmp_path)

    assert observed[0] is False
    assert observed[-1] is True


@pytest.mark.parametrize(
    "crash_phase",
    ["prepared", "quarantined", "restored", "verified", "complete"],
)
def test_rollback_reentry_restores_exact_bytes_after_every_journaled_phase(
    tmp_path: Path,
    crash_phase: str,
) -> None:
    checkpoint, data_dir, attachment_file = _checkpoint_fixture(tmp_path)
    original_attachment = attachment_file.read_bytes()
    attachment_file.write_bytes(b"candidate changed this\n")
    project_snapshot = data_dir / "project-snapshots" / "project.json"
    original_project_snapshot = project_snapshot.read_bytes()
    project_snapshot.write_bytes(b'{"graph":{"revision":1}}\n')
    (data_dir / "candidate-only-root").mkdir()
    (data_dir / "candidate-only-root" / "unknown").write_bytes(b"candidate\n")

    checkpoint_path = Path(checkpoint.manifest_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    child = subprocess.run(
        (
            sys.executable,
            "-c",
            """
import os
import sys
from pathlib import Path
from rcp.server_ops.update_checkpoint import restore_update_checkpoint

def crash_after(phase: str) -> None:
    if phase == sys.argv[3]:
        os._exit(91)

restore_update_checkpoint(
    Path(sys.argv[1]),
    expected_uid=os.geteuid(),
    expected_sha256=sys.argv[2],
    after_phase=crash_after,
)
""",
            str(checkpoint_path),
            checkpoint_sha256,
            crash_phase,
        ),
        check=False,
    )
    assert child.returncode == 91

    journal = restore_update_checkpoint(
        checkpoint_path,
        expected_uid=os.geteuid(),
        expected_sha256=checkpoint_sha256,
    )

    assert journal.phase == "complete"
    assert not (data_dir / "candidate-only-root").exists()
    assert (data_dir / "bootstrap-manifests" / "remote-project.toml").read_bytes() == (
        b'name = "Remote fixture"\n'
    )
    restored_attachment = (
        data_dir
        / "chat-attachments"
        / attachment_file.parent.parent.name
        / "files"
        / attachment_file.name
    )
    assert restored_attachment.read_bytes() == original_attachment
    assert project_snapshot.read_bytes() == original_project_snapshot
    quarantine = Path(
        next(root for root in checkpoint.roots if root.kind == "app_data").quarantine_path
    )
    assert (quarantine / "candidate-only-root" / "unknown").read_bytes() == b"candidate\n"
    persisted = read_rollback_journal(
        Path(checkpoint.operation_root) / "rollback-journal.json",
        expected_uid=os.geteuid(),
    )
    assert persisted.phase == "complete"


def test_rollback_refuses_a_checkpoint_whose_bound_digest_changed(tmp_path: Path) -> None:
    checkpoint, data_dir, _attachment_file = _checkpoint_fixture(tmp_path)
    candidate_only = data_dir / "candidate-only"
    candidate_only.write_text("candidate\n", encoding="utf-8")

    with pytest.raises(UpdateCheckpointRefused, match="manifest digest changed"):
        restore_update_checkpoint(
            Path(checkpoint.manifest_path),
            expected_uid=os.geteuid(),
            expected_sha256="f" * 64,
        )

    assert candidate_only.read_text(encoding="utf-8") == "candidate\n"


def test_rollback_rejects_journal_paths_that_do_not_match_the_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint, _data_dir, _attachment_file = _checkpoint_fixture(tmp_path)

    def crash_after_prepared(phase: str) -> None:
        if phase == "prepared":
            raise RuntimeError("crash after prepared")

    with pytest.raises(RuntimeError, match="crash after prepared"):
        restore_update_checkpoint(
            Path(checkpoint.manifest_path),
            expected_uid=os.geteuid(),
            after_phase=crash_after_prepared,
        )
    journal_path = Path(checkpoint.operation_root) / "rollback-journal.json"
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    payload["quarantine_paths"][0] = str(tmp_path / "wrong-quarantine")
    journal_path.write_text(json.dumps(payload), encoding="utf-8")
    journal_path.chmod(0o600)

    with pytest.raises(UpdateCheckpointRefused, match="another checkpoint"):
        restore_update_checkpoint(
            Path(checkpoint.manifest_path),
            expected_uid=os.geteuid(),
        )


def test_rollback_replaces_mode_only_drift_instead_of_skipping_restore(tmp_path: Path) -> None:
    checkpoint, data_dir, attachment_file = _checkpoint_fixture(tmp_path)
    app_root = next(root for root in checkpoint.roots if root.kind == "app_data")
    relative = attachment_file.relative_to(data_dir).as_posix()
    expected = next(item for item in app_root.files if item.relative_path == relative)
    attachment_file.chmod(0o644)

    restore_update_checkpoint(
        Path(checkpoint.manifest_path),
        expected_uid=os.geteuid(),
    )

    assert stat.S_IMODE(attachment_file.stat().st_mode) == expected.mode
    quarantined = Path(app_root.quarantine_path).joinpath(*PurePosixPath(relative).parts)
    assert stat.S_IMODE(quarantined.stat().st_mode) == 0o644


def test_checkpoint_fails_closed_on_unclassified_transfer_inbox_entry(tmp_path: Path) -> None:
    data_dir = tmp_path / "server" / "data"
    data_dir.mkdir(parents=True, mode=0o700)
    update_root = tmp_path / "server" / "update-checkpoints"
    update_root.mkdir(mode=0o700)
    previous_release = tmp_path / "server" / "releases" / BASE_COMMIT
    candidate_release = tmp_path / "server" / "releases" / CANDIDATE_COMMIT
    previous_release.mkdir(parents=True)
    candidate_release.mkdir()
    store, _bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Test Lab")
    inbox = data_dir / "transfer-inbox"
    inbox.mkdir(mode=0o700)
    inbox.chmod(0o700)
    (inbox / "partial").write_bytes(b"not complete")
    sqlite_publication = BackupCaptureCoordinator(
        store, data_dir, _metadata(data_dir)
    ).capture_sqlite()
    project_publication = BackupProjectFileCaptureCoordinator(data_dir).capture(
        sqlite_publication.receipt_path,
        expected_sha256=sqlite_publication.receipt_sha256,
    )
    candidate_path = verified_candidate_receipt_path(
        CANDIDATE_COMMIT,
        sqlite_publication.receipt.capture_id,
        update_root,
    )
    candidate = VerifiedCandidateReceipt(
        installation_id=str(uuid.uuid4()),
        candidate_commit=CANDIDATE_COMMIT,
        base_current_commit=BASE_COMMIT,
        base_running_commit=BASE_COMMIT,
        base_instance_id=str(uuid.uuid4()),
        base_process_pid=421,
        release_path=str(candidate_release),
        built_receipt_path=str(update_root / "built.json"),
        built_receipt_sha256="d" * 64,
        receipt_path=str(candidate_path),
        web_build_id=WEB_BUILD_ID,
        capture_id=sqlite_publication.receipt.capture_id,
        sqlite_snapshot_sha256=sqlite_publication.receipt.sqlite_snapshot.sha256,
        project_capture_sha256=project_publication.receipt_sha256,
        space_id=store.space_id,
        projects=(),
        startup_recovery=StartupRecoveryReadModel(
            active_operation_ids=(),
            stopping_experiment_operation_ids=(),
            report_episode_ids=(),
            auto_research_recovery_operation_ids=(),
            active_watcher_ids=(),
        ),
        reads=("/api/health",),
        verified_at=datetime.now(UTC),
    )
    _write_private_model(candidate_path, candidate)

    with pytest.raises(UpdateCheckpointRefused, match="typed completed-upload proof"):
        UpdateCheckpointCoordinator(
            data_dir=data_dir,
            update_root=update_root,
            previous_release_path=previous_release,
            expected_uid=os.geteuid(),
        ).create(
            sqlite_receipt_path=sqlite_publication.receipt_path,
            sqlite_receipt_sha256=sqlite_publication.receipt_sha256,
            project_receipt_path=project_publication.receipt_path,
            project_receipt_sha256=project_publication.receipt_sha256,
            candidate_receipt_path=candidate_path,
            candidate_receipt_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        )
    assert not list(update_root.glob("checkpoint-*/checkpoint.json"))


class _TransferUploadSnapshot:
    def __init__(self, row: dict[str, object], request: object) -> None:
        self._row = row
        self._request = request

    def project_transfer_request(self, _request_id: str):
        return self._request

    def target_project_transfer_uploads(self):
        receipt_json = self._row["receipt_json"]
        receipt = (
            None
            if receipt_json is None
            else ProjectTransferUploadCompleteReceipt.model_validate_json(receipt_json)
        )
        return [
            ProjectTransferUploadRecord(
                **{
                    **{key: value for key, value in self._row.items() if key != "receipt_json"},
                    "receipt": receipt,
                },
            )
        ]


def _transfer_upload_capture_fixture(tmp_path: Path, *, status: str = "complete"):
    data_dir = tmp_path / "server" / "data"
    data_dir.mkdir(parents=True, mode=0o700)
    data_dir.chmod(0o700)
    inbox = data_dir / "transfer-inbox"
    inbox.mkdir(mode=0o700)
    inbox.chmod(0o700)
    request_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())
    payload = b"one exact transfer archive\n"
    digest = hashlib.sha256(payload).hexdigest()
    final = target_transfer_archive_path(data_dir, request_id)
    final.write_bytes(payload)
    final.chmod(0o600)
    timestamp = datetime.now(UTC).isoformat()
    receipt = {
        "request_id": request_id,
        "project_id": project_id,
        "archive_sha256": digest,
        "archive_size_bytes": len(payload),
        "lease_boundary_sha256": "a" * 64,
        "completed_at": timestamp,
    }
    row = {
        "request_id": request_id,
        "project_id": project_id,
        "archive_sha256": digest,
        "archive_size_bytes": len(payload),
        "lease_boundary_sha256": "a" * 64,
        "status": status,
        "receipt_json": json.dumps(receipt) if status in {"complete", "consumed"} else None,
        "created_at": timestamp,
        "updated_at": timestamp,
        "invalidated_at": None,
    }
    request = SimpleNamespace(
        side="target",
        project_id=project_id,
        archive_sha256=digest,
        archive_size_bytes=len(payload),
    )
    destination = tmp_path / "checkpoint" / "app-data"
    destination.mkdir(parents=True, mode=0o700)
    destination.chmod(0o700)
    coordinator = UpdateCheckpointCoordinator(
        data_dir=data_dir,
        update_root=tmp_path / "server" / "update-checkpoints",
        previous_release_path=tmp_path / "server" / "releases" / BASE_COMMIT,
        expected_uid=os.geteuid(),
    )
    return coordinator, _TransferUploadSnapshot(row, request), destination, final, payload


def test_checkpoint_captures_only_receipt_backed_complete_transfer_archive(
    tmp_path: Path,
) -> None:
    coordinator, snapshot, destination, final, payload = _transfer_upload_capture_fixture(tmp_path)

    directories, files = coordinator._copy_transfer_inbox(snapshot, destination)  # noqa: SLF001

    assert directories == {"transfer-inbox"}
    assert [item.relative_path for item in files] == [f"transfer-inbox/{final.name}"]
    copied = destination / "transfer-inbox" / final.name
    assert copied.read_bytes() == payload
    assert stat.S_IMODE(copied.stat().st_mode) == 0o400


def test_checkpoint_ignores_consumed_transfer_upload_without_an_inbox_archive(
    tmp_path: Path,
) -> None:
    coordinator, snapshot, destination, final, _payload = _transfer_upload_capture_fixture(
        tmp_path,
        status="consumed",
    )
    final.unlink()

    directories, files = coordinator._copy_transfer_inbox(snapshot, destination)  # noqa: SLF001

    assert directories == set()
    assert files == []


@pytest.mark.parametrize(
    ("status", "mutation", "message"),
    [
        ("active", lambda _path: None, "durable complete boundary"),
        ("consumed", lambda _path: None, "no typed completed-upload proof"),
        ("complete", lambda path: path.unlink(), "missing its archive file"),
        ("complete", lambda path: path.write_bytes(b"wrong"), "differs from its receipt"),
        ("complete", lambda path: path.chmod(0o644), "unsafe ownership, mode, or type"),
    ],
)
def test_checkpoint_refuses_unfinished_or_unsafe_transfer_archive(
    tmp_path: Path,
    status: str,
    mutation: Callable[[Path], None],
    message: str,
) -> None:
    coordinator, snapshot, destination, final, _payload = _transfer_upload_capture_fixture(
        tmp_path,
        status=status,
    )
    mutation(final)

    with pytest.raises(UpdateCheckpointRefused, match=message):
        coordinator._copy_transfer_inbox(snapshot, destination)  # noqa: SLF001


def test_checkpoint_refuses_extra_transfer_partial_beside_complete_archive(
    tmp_path: Path,
) -> None:
    coordinator, snapshot, destination, final, _payload = _transfer_upload_capture_fixture(tmp_path)
    (final.parent / ".unexpected.partial").write_bytes(b"partial")
    (final.parent / ".unexpected.partial").chmod(0o600)

    with pytest.raises(UpdateCheckpointRefused, match="unknown, partial, or untyped"):
        coordinator._copy_transfer_inbox(snapshot, destination)  # noqa: SLF001


def test_checkpoint_manifest_detects_payload_tampering(tmp_path: Path) -> None:
    checkpoint, _data_dir, _attachment_file = _checkpoint_fixture(tmp_path)
    app_root = next(root for root in checkpoint.roots if root.kind == "app_data")
    database = Path(checkpoint.operation_root) / app_root.archive_path / "rcp.sqlite3"
    database.chmod(0o600)
    database.write_bytes(b"tampered")
    database.chmod(0o400)

    with pytest.raises(UpdateCheckpointRefused, match="file changed"):
        restore_update_checkpoint(
            Path(checkpoint.manifest_path),
            expected_uid=os.geteuid(),
        )


def test_rollback_replaces_local_research_root_and_never_overlays_candidate_files(
    tmp_path: Path,
) -> None:
    (
        checkpoint,
        research,
        original_manifest,
        imported_owner,
        imported_inventory,
        imported_payload,
    ) = _local_project_checkpoint_fixture(tmp_path)
    (research / "manifest.toml").write_bytes(b"candidate changed the manifest\n")
    (research / "candidate-only").mkdir()
    (research / "candidate-only" / "unknown.json").write_bytes(b"unknown\n")
    imported_file = (
        imported_owner.root
        / imported_inventory.files[0].provider
        / imported_inventory.files[0].sha256
    )
    imported_file.chmod(0o600)
    imported_file.write_bytes(b"candidate changed imported history\n")

    restore_update_checkpoint(
        Path(checkpoint.manifest_path),
        expected_uid=os.geteuid(),
    )

    assert (research / "manifest.toml").read_bytes() == original_manifest
    assert not (research / "candidate-only").exists()
    assert imported_owner.inventory() == imported_inventory
    assert imported_file.read_bytes() == imported_payload
    project_root = next(root for root in checkpoint.roots if root.kind == "project_research")
    quarantine = Path(project_root.quarantine_path)
    assert (quarantine / "candidate-only" / "unknown.json").read_bytes() == b"unknown\n"
    assert (quarantine / "manifest.toml").read_bytes() == b"candidate changed the manifest\n"


def test_checkpoint_rejects_a_symlinked_imported_source_payload(tmp_path: Path) -> None:
    checkpoint, *_rest = _local_project_checkpoint_fixture(tmp_path)
    app_root = next(root for root in checkpoint.roots if root.kind == "app_data")
    imported = next(
        item
        for item in app_root.files
        if item.relative_path.startswith("project-sources/")
        and not item.relative_path.endswith("manifest.json")
    )
    payload = Path(checkpoint.operation_root).joinpath(
        *PurePosixPath(app_root.archive_path).parts,
        *PurePosixPath(imported.relative_path).parts,
    )
    outside = tmp_path / "outside-imported-history"
    outside.write_bytes(b"outside\n")
    payload.unlink()
    payload.symlink_to(outside)

    with pytest.raises(UpdateCheckpointRefused, match="unsafe file"):
        restore_update_checkpoint(
            Path(checkpoint.manifest_path),
            expected_uid=os.geteuid(),
        )


def test_checkpoint_attachment_digest_is_bound_to_original_bytes(tmp_path: Path) -> None:
    checkpoint, _data_dir, attachment_file = _checkpoint_fixture(tmp_path)
    app_root = next(root for root in checkpoint.roots if root.kind == "app_data")
    entry = next(
        item for item in app_root.files if item.relative_path.endswith(attachment_file.name)
    )
    assert entry.sha256 == hashlib.sha256(b"pre-cutover attachment\n").hexdigest()
