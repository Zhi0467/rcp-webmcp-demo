from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rcp.api import create_app
from rcp.artifacts import AgentArtifactDescriptor
from rcp.config import AGENT_EXECUTION_PROFILES
from rcp.core.models import AuthorizedHuman
from rcp.providers import configured_runtime_id
from rcp.server_ops.backup_capture import (
    BackupCaptureCoordinator,
    BackupCaptureUnavailable,
    read_backup_sqlite_capture_receipt,
)
from rcp.server_ops.control import ServerControlClient
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_runtime import ServerMetadata, published_server_metadata
from rcp.setup import render_prepared_team_manifest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineRecord,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningRepositoryRecord,
    ProjectProvisioningRequestRecord,
    ProjectRecord,
    ResultViewRecord,
)
from rcp.storage.provisioning import project_provisioning_review_digest

SOURCE_COMMIT = "a" * 40
WEB_BUILD_ID = "sha256:" + ("b" * 64)
PUBLIC_KEY_FINGERPRINT = "SHA256:" + ("A" * 43)


def _metadata(data_dir: Path, *, control_socket: Path | None = None) -> ServerMetadata:
    return ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=control_socket or data_dir / "control.sock",
        running_commit=SOURCE_COMMIT,
        web_build_id=WEB_BUILD_ID,
    )


def _register_completed_project(
    store: AppStore,
    data_dir: Path,
    *,
    name: str,
) -> ProjectRecord:
    project_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    checked_at = "2026-08-29T12:00:00+00:00"
    central_root = data_dir.parent / "remote-central"
    repository_path = central_root / project_id / "repositories" / "paper"
    machine = ProjectProvisioningMachineRecord(
        alias="worker",
        location="ssh",
        host="gpu.example",
        os_account="rcp",
        central_root=str(central_root),
        resolved_central_root=str(central_root),
    )
    repository = ProjectProvisioningRepositoryRecord(
        alias="paper",
        repository=parse_github_repository_ref("git@github.com:OpenAI/RCP.git"),
        machine_alias="worker",
        intended_path=str(repository_path),
        resolved_path=str(repository_path),
        checkout_disposition="request_created",
        git_check=ProjectProvisioningGitCheckRecord(
            status="ready",
            commit="c" * 40,
            write_verified=True,
            deploy_key_label=f"rcp:{store.space_id}:{project_id}:paper",
            public_key_fingerprint=PUBLIC_KEY_FINGERPRINT,
            checked_at=checked_at,
        ),
    )
    provider_checks = [
        ProjectProvisioningProviderCheckRecord(
            profile=profile,
            provider="codex",
            runtime_id="codex:exec",
            model="gpt-test",
            reasoning="medium",
            machine_alias="worker",
            status="ready",
            binary_path="/usr/local/bin/codex",
            version="codex-cli 1.2.3",
            resolved_runtime_id=configured_runtime_id("codex", "exec"),
            execution_account="rcp",
            checked_at=checked_at,
        )
        for profile in AGENT_EXECUTION_PROFILES
    ]
    values = {
        "request_id": request_id,
        "kind": "create_team_project",
        "status": "completed",
        "target_space_id": store.space_id,
        "authorized_by": AuthorizedHuman(
            space_id=store.space_id,
            user_id=user_id,
            display_name="Alice",
        ),
        "proposed_project_id": project_id,
        "name": name,
        "state_repository": "paper",
        "project_truth_scope": ["paper"],
        "default_run_truth_scope": ["paper"],
        "default_auto_research_invocation_ceiling": 10,
        "machines": [machine],
        "repositories": [repository],
        "provider_checks": provider_checks,
        "final_review_digest": "0" * 64,
        "revision": 5,
        "created_at": checked_at,
        "updated_at": checked_at,
        "setup_started_at": checked_at,
        "ready_at": checked_at,
        "completed_at": checked_at,
    }
    draft = ProjectProvisioningRequestRecord.model_validate(values)
    request = ProjectProvisioningRequestRecord.model_validate(
        {
            **values,
            "final_review_digest": project_provisioning_review_digest(draft),
        }
    )
    with store.connection() as connection:
        store._insert_project_provisioning_request(connection, request)
    locator = data_dir / "bootstrap-manifests" / f"{project_id}.toml"
    locator.parent.mkdir(exist_ok=True)
    locator.write_text(render_prepared_team_manifest(request), encoding="utf-8")
    record = ProjectRecord(
        project_id=project_id,
        home_space_id=store.space_id,
        locator=str(locator),
        name=name,
        state_location=f"gpu.example:{repository_path}/.research",
        state_remote=True,
        added_at=checked_at,
    )
    return store.upsert_project(record)


def _create_task_with_kept_artifact(store: AppStore, project_id: str) -> str:
    operation_id = str(uuid.uuid4())
    now = store.now()
    descriptor = AgentArtifactDescriptor(
        artifact_id="d" * 24,
        name="figure.png",
        media_type="image/png",
        size_bytes=123,
        kept_filename="kept-figure.png",
        kept_at=now,
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="refresh",
            status="queued",
            request={},
            result={"artifacts": [descriptor.model_dump(mode="json")]},
            created_at=now,
            updated_at=now,
            status_message="queued",
        )
    )
    return operation_id


def _create_kept_view(
    store: AppStore,
    project_id: str,
    *,
    origin_operation_id: str,
    latest_operation_id: str | None = None,
) -> None:
    html = b"<html><body>captured</body></html>"
    now = store.now()
    expires_at = (datetime.fromisoformat(now) + timedelta(days=365)).isoformat()
    store.create_result_view(
        ResultViewRecord(
            view_id=uuid.uuid4().hex[:24],
            project_id=project_id,
            experiment_id="exp/capture",
            chat_id=str(uuid.uuid4()),
            origin_operation_id=origin_operation_id,
            latest_operation_id=latest_operation_id or origin_operation_id,
            provider="codex",
            model="gpt-test",
            reasoning="medium",
            run_on="worker",
            native_session_id="session-test",
            stage_host="",
            stage_root="/tmp/rcp-stage",
            source_name="result.html",
            content_sha256=hashlib.sha256(html).hexdigest(),
            size_bytes=len(html),
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            kept_filename="kept-result.html",
            kept_at=now,
        ),
        html=html,
    )


def test_online_sqlite_snapshot_stays_consistent_while_writers_continue(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    store, _ = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Snapshot lab")
    with store.connection() as connection:
        connection.execute(
            "CREATE TABLE backup_writer_probe (sequence INTEGER PRIMARY KEY, payload TEXT)"
        )

    started = threading.Event()
    require_post_snapshot_write = threading.Event()
    post_snapshot_write = threading.Event()
    stop = threading.Event()
    writer_error: list[BaseException] = []
    committed = 0
    committed_lock = threading.Lock()

    def writer() -> None:
        nonlocal committed
        try:
            connection = sqlite3.connect(store.path, timeout=30.0)
            try:
                while not stop.is_set():
                    connection.execute(
                        "INSERT INTO backup_writer_probe(payload) VALUES (?)",
                        ("writer",),
                    )
                    connection.commit()
                    with committed_lock:
                        committed += 1
                        if committed >= 10:
                            started.set()
                    if require_post_snapshot_write.is_set():
                        post_snapshot_write.set()
                    if stop.wait(0.001):
                        break
            finally:
                connection.close()
        except BaseException as exc:  # pragma: no cover - surfaced below
            writer_error.append(exc)
            started.set()

    thread = threading.Thread(target=writer, name="backup-test-writer")
    thread.start()
    assert started.wait(timeout=5)
    capture_root = tmp_path / "capture"
    capture_root.mkdir(mode=0o700)
    snapshot_path = capture_root / "rcp.sqlite3"
    try:
        store.online_snapshot(snapshot_path)
        require_post_snapshot_write.set()
        assert post_snapshot_write.wait(timeout=5)
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert writer_error == []

    snapshot = AppStore.open_read_only_snapshot(snapshot_path)
    with snapshot.connection() as connection:
        snapshot_count = connection.execute("SELECT COUNT(*) FROM backup_writer_probe").fetchone()[
            0
        ]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO backup_writer_probe(payload) VALUES ('forbidden')")
    with store.connection() as connection:
        live_count = connection.execute("SELECT COUNT(*) FROM backup_writer_probe").fetchone()[0]
    assert 10 <= snapshot_count <= live_count
    assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o400


def test_capture_inventory_is_bound_to_the_copied_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    store, _ = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Capture lab")
    captured = _register_completed_project(store, data_dir, name="Captured project")
    operation_id = _create_task_with_kept_artifact(store, captured.project_id)
    _create_kept_view(store, captured.project_id, origin_operation_id=operation_id)
    original_snapshot = store.online_snapshot
    late_project: ProjectRecord | None = None

    def snapshot_then_register(destination: Path) -> None:
        nonlocal late_project
        original_snapshot(destination)
        late_project = _register_completed_project(store, data_dir, name="Registered later")

    monkeypatch.setattr(store, "online_snapshot", snapshot_then_register)
    publication = BackupCaptureCoordinator(store, data_dir, _metadata(data_dir)).capture_sqlite()

    receipt = read_backup_sqlite_capture_receipt(
        publication.receipt_path,
        expected_sha256=publication.receipt_sha256,
    )
    assert late_project is not None
    assert [project.project_id for project in receipt.projects] == [captured.project_id]
    project = receipt.projects[0]
    assert project.status == "capturable"
    assert project.task_operation_ids == (operation_id,)
    assert [item.kept_filename for item in project.kept_artifacts] == ["kept-figure.png"]
    assert [item.kept_filename for item in project.kept_result_views] == ["kept-result.html"]
    assert receipt.status == "complete"
    assert receipt.sqlite_snapshot.size_bytes == Path(receipt.snapshot_path).stat().st_size
    assert stat.S_IMODE(publication.receipt_path.stat().st_mode) == 0o400
    snapshot_store = AppStore.open_read_only_snapshot(Path(receipt.snapshot_path))
    assert snapshot_store.project(late_project.project_id) is None


def test_malformed_and_cross_project_references_make_only_their_projects_uncaptured(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    store, _ = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Partial lab")
    malformed = _register_completed_project(store, data_dir, name="Malformed artifacts")
    cross_reference = _register_completed_project(store, data_dir, name="Cross reference")
    wrong_home = _register_completed_project(store, data_dir, name="Wrong home")
    healthy = _register_completed_project(store, data_dir, name="Healthy")

    malformed_operation = _create_task_with_kept_artifact(store, malformed.project_id)
    with store.connection() as connection:
        connection.execute(
            "UPDATE graph_runs SET result_json = ? WHERE operation_id = ?",
            (
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "artifact_id": "e" * 24,
                                "name": "figure.png",
                                "media_type": "image/png",
                                "size_bytes": 5,
                                "kept_filename": "../outside.png",
                                "kept_at": store.now(),
                            }
                        ]
                    }
                ),
                malformed_operation,
            ),
        )
    foreign_operation = _create_task_with_kept_artifact(store, healthy.project_id)
    local_operation = _create_task_with_kept_artifact(store, cross_reference.project_id)
    _create_kept_view(
        store,
        cross_reference.project_id,
        origin_operation_id=foreign_operation,
        latest_operation_id=local_operation,
    )
    observed_wrong_home = str(uuid.uuid4())
    with store.connection() as connection:
        connection.execute(
            "UPDATE projects SET home_space_id = ? WHERE project_id = ?",
            (observed_wrong_home, wrong_home.project_id),
        )

    receipt = (
        BackupCaptureCoordinator(store, data_dir, _metadata(data_dir)).capture_sqlite().receipt
    )
    status = {project.project_id: project.status for project in receipt.projects}
    assert status == {
        malformed.project_id: "uncaptured",
        cross_reference.project_id: "uncaptured",
        wrong_home.project_id: "uncaptured",
        healthy.project_id: "capturable",
    }
    assert (
        next(
            project for project in receipt.projects if project.project_id == wrong_home.project_id
        ).home_space_id
        == observed_wrong_home
    )
    assert receipt.status == "partial"


def test_capture_receipt_rejects_tampering(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store, _ = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Receipt lab")
    publication = BackupCaptureCoordinator(store, data_dir, _metadata(data_dir)).capture_sqlite()
    publication.receipt_path.chmod(0o600)
    publication.receipt_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(BackupCaptureUnavailable, match="digest does not match"):
        read_backup_sqlite_capture_receipt(
            publication.receipt_path,
            expected_sha256=publication.receipt_sha256,
        )


def test_capture_receipt_is_bound_to_its_private_capture_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store, _ = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Receipt path lab")
    publication = BackupCaptureCoordinator(store, data_dir, _metadata(data_dir)).capture_sqlite()
    other_root = data_dir / "run-stage" / f"backup-{uuid.uuid4()}"
    other_root.mkdir(mode=0o700)
    copied_receipt = other_root / "sqlite-capture.json"
    copied_receipt.write_bytes(publication.receipt_path.read_bytes())
    copied_receipt.chmod(0o400)

    with pytest.raises(BackupCaptureUnavailable, match="not bound"):
        read_backup_sqlite_capture_receipt(
            copied_receipt,
            expected_sha256=publication.receipt_sha256,
        )


def test_installed_control_socket_publishes_only_the_small_capture_receipt(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Control capture lab")
    with tempfile.TemporaryDirectory(prefix="rcp-backup-control-", dir="/tmp") as root:
        control_root = Path(root)
        os.chown(control_root, os.geteuid(), os.getegid())
        control_root.chmod(0o700)
        metadata = _metadata(data_dir, control_socket=control_root / "control.sock")
        app = create_app(data_dir=data_dir, instance_metadata=metadata)

        with published_server_metadata(data_dir, metadata), TestClient(app):
            result = ServerControlClient.from_data_dir(
                data_dir,
                expected_server_uid=os.geteuid(),
            ).capture_backup_sqlite()

    receipt = read_backup_sqlite_capture_receipt(
        Path(result.receipt_path),
        expected_sha256=result.receipt_sha256,
    )
    assert result.capture_id == receipt.capture_id
    assert result.snapshot_sha256 == receipt.sqlite_snapshot.sha256
    assert result.project_count == 0
    assert result.uncaptured_project_count == 0
    assert result.status == "complete"
