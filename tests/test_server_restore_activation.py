from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rcp.api.app as app_module
from rcp.api import create_app
from rcp.server_ops.backup_capture import _database_schema_sha256
from rcp.server_ops.backup_models import BackupArchiveManifest, BackupFileEntry
from rcp.server_ops.cli import CallerIdentity
from rcp.server_ops.control import (
    ServerControlPeer,
    ServerControlRequest,
    ServerControlRestoreResult,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.restore import (
    LinuxRestoreMachine,
    RestoreConfirmation,
    RestoreMemberRosterEntry,
    RestoreMemberRosterReview,
    RestoreOldAuthorityReview,
    RestoreOperationJournal,
    RestoreRefused,
    _restore_plan,
    read_restore_journal,
    restore_activation_boundary,
    restore_member_roster_boundary,
    restore_old_authority_boundary,
    write_restore_journal,
)
from rcp.server_runtime import ServerMetadata, published_server_metadata
from rcp.storage import AppStore

NOW = datetime(2026, 8, 29, 18, 0, tzinfo=UTC)
COMMIT = "a" * 40


class _StoppedService:
    def fence_stopped_disabled(self) -> None:
        return None


class _ActivationService(_StoppedService):
    def __init__(self, layout: ServerLayout) -> None:
        self.layout = layout
        self.calls: list[str] = []

    def current_release(self) -> Path:
        return self.layout.release_dir(COMMIT)

    def start(self) -> int:
        self.calls.append("start")
        return os.getpid()

    def enable(self) -> int:
        self.calls.append("enable")
        return os.getpid()

    def enable_and_start(self) -> int:
        self.calls.append("enable_and_start")
        return os.getpid()


class _ReviewMachine(LinuxRestoreMachine):
    @contextmanager
    def admission(self):
        yield


def _layout(tmp_path: Path) -> ServerLayout:
    home = tmp_path / "home"
    root = home / "server"
    runtime = Path(tempfile.mkdtemp(prefix="rcp-restore-activation-", dir="/tmp"))
    os.chown(runtime, os.geteuid(), os.getegid())
    runtime.chmod(0o700)
    layout = replace(
        DEFAULT_SERVER_LAYOUT,
        service_home=home,
        server_root=root,
        source_checkout=root / "source",
        releases_root=root / "releases",
        data_dir=root / "data",
        projects_root=root / "projects",
        credentials_root=root / "credentials",
        update_checkpoints_root=root / "update-checkpoints",
        restore_operations_root=root / "restore-operations",
        codex_state_root=home / ".codex",
        claude_state_root=home / ".claude",
        ssh_state_root=home / ".ssh",
        runtime_dir=runtime,
        control_socket=runtime / "control.sock",
    )
    for path in (
        layout.data_dir,
        layout.projects_root,
        layout.update_checkpoints_root,
        layout.restore_operations_root,
        runtime,
    ):
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        path.chmod(0o700)
    return layout


def _activation_ready(tmp_path: Path) -> tuple[ServerLayout, RestoreOperationJournal]:
    layout = _layout(tmp_path)
    store, bootstrap = AppStore.initialize_team_space(
        layout.data_dir / "rcp.sqlite3", "Restored lab"
    )
    member, _token = store.enroll_team_member(bootstrap, "Alice")
    sqlite_path = layout.data_dir / "rcp.sqlite3"
    sqlite_sha256 = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    entry = BackupFileEntry(
        archive_path="database/rcp.sqlite3",
        source_relative_path="rcp.sqlite3",
        group="sqlite_snapshot",
        sha256=sqlite_sha256,
        size_bytes=sqlite_path.stat().st_size,
    )
    manifest = BackupArchiveManifest(
        space_id=store.space_id,
        space_name=store.space_name,
        rcp_source_commit=COMMIT,
        database_schema_sha256=_database_schema_sha256(store),
        captured_at=NOW,
        sqlite_snapshot=entry,
        encryption_recipient_fingerprint="e" * 64,
        installation_id=str(uuid.uuid4()),
        excluded_app_data_entries=(),
        uncaptured_app_data_entries=(),
        projects=(),
        status="complete",
        total_bytes=entry.size_bytes,
    )
    archive = tmp_path / "archive.age"
    archive.write_bytes(b"archive")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    candidate_root = layout.restore_operations_root / f"candidate-{archive_sha256}"
    candidate_root.mkdir(mode=0o700)
    candidate = candidate_root / "restored" / "rcp.sqlite3"
    candidate.parent.mkdir(mode=0o700)
    shutil.copyfile(sqlite_path, candidate)
    authority = store.active_team_member_authority()
    roster = tuple(
        RestoreMemberRosterEntry(
            member_id=item.member_id,
            display_name=item.display_name,
            active_token_ids=item.active_token_ids,
        )
        for item in authority
    )
    assert roster[0].member_id == member.user_id
    journal = RestoreOperationJournal(
        operation_id=str(uuid.uuid4()),
        archive_path=str(archive),
        archive_sha256=archive_sha256,
        archive_size_bytes=archive.stat().st_size,
        manifest_sha256="1" * 64,
        configured_data_dir=str(layout.data_dir),
        candidate_root=str(candidate_root),
        candidate_sqlite_path=str(candidate),
        candidate_sqlite_sha256=sqlite_sha256,
        manifest=manifest,
        confirmation=RestoreConfirmation(
            confirmed_data_dir=str(layout.data_dir),
            confirmed_by="root@lab uid=0",
            confirmed_at=NOW,
        ),
        phase="activation_ready",
        detached_at=NOW,
        restored_sqlite_sha256=sqlite_sha256,
        old_authority_review=RestoreOldAuthorityReview(
            boundary_sha256=restore_old_authority_boundary(manifest),
            disposition="old-machine-destroyed",
            reviewed_by="root@lab uid=0",
            reviewed_at=NOW,
        ),
        member_roster_review=RestoreMemberRosterReview(
            boundary_sha256=restore_member_roster_boundary(NOW, roster),
            members=roster,
            reviewed_by="root@lab uid=0",
            reviewed_at=NOW,
        ),
        updated_at=NOW,
    )
    write_restore_journal(journal, layout, uid=os.geteuid(), gid=os.getegid())
    return layout, journal


def test_replacement_opens_only_after_private_durable_activation(tmp_path: Path) -> None:
    layout, journal = _activation_ready(tmp_path)
    metadata = ServerMetadata.create(
        layout.data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=layout.control_socket,
        running_commit=COMMIT,
        web_build_id=f"sha256:{'b' * 64}",
    )
    try:
        app = create_app(
            data_dir=layout.data_dir,
            instance_metadata=metadata,
            server_layout=layout,
        )

        with TestClient(app) as client:
            assert client.get("/api/health").status_code == 503
            control = app.state.server_control
            result = control.handler(
                ServerControlRequest(
                    request_id=str(uuid.uuid4()),
                    instance_id=metadata.instance_id,
                    operation="restore_activation_commit",
                    selector_id=journal.operation_id,
                    boundary_sha256=restore_activation_boundary(journal),
                ),
                ServerControlPeer(pid=os.getpid(), uid=0, gid=0),
            )
            assert isinstance(result, ServerControlRestoreResult)
            assert result.restore_phase == "complete"
            assert client.get("/api/health").status_code == 200

        completed = read_restore_journal(layout, expected_uid=os.geteuid())
        assert completed.phase == "complete"
        assert completed.activation_readback == result.readback
        assert not any(app.state.startup_recovery_plan.values())
    finally:
        shutil.rmtree(layout.runtime_dir)


def test_uncommitted_replacement_requests_clean_stop_after_bounded_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, _journal = _activation_ready(tmp_path)
    metadata = ServerMetadata.create(
        layout.data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=layout.control_socket,
        running_commit=COMMIT,
        web_build_id=f"sha256:{'b' * 64}",
    )
    stop_requested = threading.Event()
    monkeypatch.setattr(app_module, "SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        app_module,
        "_terminate_uncommitted_restore_process",
        stop_requested.set,
    )
    try:
        app = create_app(
            data_dir=layout.data_dir,
            instance_metadata=metadata,
            server_layout=layout,
        )

        with TestClient(app) as client:
            assert stop_requested.wait(timeout=1.0)
            assert client.get("/api/health").status_code == 503
            assert app.state.startup_effect_runtime_started is False

        assert read_restore_journal(layout, expected_uid=os.geteuid()).phase == "activation_ready"
    finally:
        shutil.rmtree(layout.runtime_dir)


def test_root_restore_starts_disabled_then_enables_after_private_commit(
    tmp_path: Path,
) -> None:
    layout, journal = _activation_ready(tmp_path)
    metadata = ServerMetadata.create(
        layout.data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=layout.control_socket,
        running_commit=COMMIT,
        web_build_id=f"sha256:{'b' * 64}",
    )
    service = _ActivationService(layout)
    machine = LinuxRestoreMachine(
        layout,
        config_loader=lambda _path: None,  # type: ignore[arg-type]
        service_control=service,  # type: ignore[arg-type]
        service_identity=(os.geteuid(), os.getegid()),
        root_identity=(os.geteuid(), os.getegid()),
        clock=lambda: NOW,
    )
    try:
        app = create_app(
            data_dir=layout.data_dir,
            instance_metadata=metadata,
            server_layout=layout,
        )
        assert app.state.server_control is not None
        app.state.server_control.peer_resolver = lambda _connection: ServerControlPeer(
            pid=os.getpid(), uid=0, gid=0
        )

        with published_server_metadata(layout.data_dir, metadata), TestClient(app) as client:
            completed = machine.activate_replacement(journal)
            assert completed.phase == "complete"
            assert client.get("/api/health").status_code == 200

        assert service.calls == ["start", "enable"]
    finally:
        shutil.rmtree(layout.runtime_dir)


def test_restore_reentry_preserves_evolved_database_after_project_publication(
    tmp_path: Path,
) -> None:
    layout, ready = _activation_ready(tmp_path)
    published = ready.model_copy(
        update={
            "phase": "projects_published",
            "old_authority_review": None,
            "member_roster_review": None,
        }
    )
    write_restore_journal(published, layout, uid=os.geteuid(), gid=os.getegid())

    restored_database = layout.data_dir / "rcp.sqlite3"
    restored_database.chmod(0o600)
    store = AppStore(restored_database)
    first = store.active_team_member_authority()[0]
    store.create_team_invitation(first.member_id)
    assert (
        hashlib.sha256(restored_database.read_bytes()).hexdigest()
        != published.candidate_sqlite_sha256
    )

    machine = _ReviewMachine(
        layout,
        config_loader=lambda _path: None,  # type: ignore[arg-type]
        service_control=_StoppedService(),  # type: ignore[arg-type]
        service_identity=(os.geteuid(), os.getegid()),
        root_identity=(os.geteuid(), os.getegid()),
        clock=lambda: NOW,
    )
    assert machine.install_sqlite_candidate(published) == published
    assert machine.verify_offline_candidate(published) == published


def test_restore_reviews_old_authority_then_removes_stale_member_offline(
    tmp_path: Path,
) -> None:
    layout, ready = _activation_ready(tmp_path)
    store = AppStore(layout.data_dir / "rcp.sqlite3")
    first = store.active_team_member_authority()[0]
    _invitation, code = store.create_team_invitation(first.member_id)
    stale, _token = store.enroll_team_member(code, "Stale member")
    published = ready.model_copy(
        update={
            "phase": "projects_published",
            "old_authority_review": None,
            "member_roster_review": None,
        }
    )
    write_restore_journal(published, layout, uid=os.geteuid(), gid=os.getegid())
    machine = _ReviewMachine(
        layout,
        config_loader=lambda _path: None,  # type: ignore[arg-type]
        service_control=_StoppedService(),  # type: ignore[arg-type]
        service_identity=(os.geteuid(), os.getegid()),
        root_identity=(os.geteuid(), os.getegid()),
        clock=lambda: NOW,
    )
    steps = _restore_plan(CallerIdentity(uid=0, username="root", host="lab"), layout.data_dir)
    resume = ("sudo", "rcp", "server", "restore", "/archive")

    authority_pause = machine.review_old_authority(
        published,
        disposition=None,
        confirmed_boundary=None,
        confirmed_by="root@lab uid=0",
        resume_argv=resume,
        step=steps[8],
    )
    assert authority_pause.operator_action is not None
    boundary = restore_old_authority_boundary(published.manifest)
    with pytest.raises(RestoreRefused, match="changed after confirmation"):
        machine.review_old_authority(
            published,
            disposition="old-machine-destroyed",
            confirmed_boundary="0" * 64,
            confirmed_by="root@lab uid=0",
            resume_argv=resume,
            step=steps[8],
        )
    reviewed = machine.review_old_authority(
        published,
        disposition="old-machine-destroyed",
        confirmed_boundary=boundary,
        confirmed_by="root@lab uid=0",
        resume_argv=resume,
        step=steps[8],
    ).journal

    roster_pause = machine.review_member_roster(
        reviewed,
        confirmed_boundary=None,
        stale_member_id=stale.user_id,
        confirmed_by="root@lab uid=0",
        resume_argv=resume,
        step=steps[9],
    )
    assert roster_pause.operator_action is not None
    assert store.space_user(stale.user_id).removed_at is not None  # type: ignore[union-attr]
    members = tuple(
        RestoreMemberRosterEntry(
            member_id=item.member_id,
            display_name=item.display_name,
            active_token_ids=item.active_token_ids,
        )
        for item in store.active_team_member_authority()
    )
    roster_boundary = restore_member_roster_boundary(ready.manifest.captured_at, members)
    completed_review = machine.review_member_roster(
        roster_pause.journal,
        confirmed_boundary=roster_boundary,
        stale_member_id=None,
        confirmed_by="root@lab uid=0",
        resume_argv=resume,
        step=steps[9],
    ).journal
    assert completed_review.phase == "member_roster_reviewed"
    assert tuple(item.member_id for item in completed_review.member_roster_review.members) == (
        first.member_id,
    )
