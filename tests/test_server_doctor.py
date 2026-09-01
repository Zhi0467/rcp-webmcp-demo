from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import tempfile
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rcp.__main__ import build_parser
from rcp.server_ops import backup as backup_owner
from rcp.server_ops import doctor as server_doctor
from rcp.server_ops.backup import BackupArchiveReceipt, BackupRunOutcome
from rcp.server_ops.cli import CallerIdentity, run_server_command
from rcp.server_ops.config import ServerBackupConfig, ServerSourceConfig
from rcp.server_ops.control import SERVER_CONTROL_OPERATIONS, ServerControlMemberSnapshot
from rcp.server_ops.doctor import (
    LinuxServerDoctorMachine,
    ServerDoctorReport,
    _member_removal_problems,
    prepare_doctor_command,
    release_relationship,
)
from rcp.server_ops.layout import ServerLayout, server_service_unit_text
from rcp.server_ops.update_cutover import new_update_operation, publish_update_operation
from rcp.server_runtime import (
    ServerMetadata,
    ServerMetadataError,
    capture_installed_release_identity,
    metadata_path,
    web_build_identity,
)

from .helpers import create_named_app

INSTALLATION_ID = "123e4567-e89b-42d3-a456-426614174000"
SPACE_ID = "123e4567-e89b-42d3-b456-426614174001"
COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
WEB_BUILD_ID = "sha256:" + ("c" * 64)
IDENTITY = CallerIdentity(uid=501, username="rcp", host="lab.example")


class FakeDoctorMachine:
    def __init__(self, report: ServerDoctorReport) -> None:
        self.report = report
        self.calls = 0

    def inspect(self) -> ServerDoctorReport:
        self.calls += 1
        return self.report


def _report(*, problems: tuple[str, ...] = ()) -> ServerDoctorReport:
    return ServerDoctorReport(
        overall_state="problems" if problems else "healthy",
        installation_id=INSTALLATION_ID,
        service_account="rcp",
        data_dir="/home/rcp/rcp-server/data",
        source_root="/home/rcp/rcp-server/source",
        releases_root="/home/rcp/rcp-server/releases",
        configured_origin="https://github.com/openai/rcp.git",
        configured_branch="main",
        source_public_key_fingerprint=None,
        managed_main_head=COMMIT,
        upstream_head=COMMIT,
        candidate_commit=None,
        current_commit=COMMIT,
        running_commit=COMMIT,
        release_state="aligned",
        source_state="aligned",
        current_web_build_id=WEB_BUILD_ID,
        running_web_build_id=WEB_BUILD_ID,
        service_active_state="active",
        service_unit_file_state="enabled",
        service_main_pid=421,
        reload_mode="disabled",
        space_id=SPACE_ID,
        instance_id=INSTALLATION_ID,
        process_pid=421,
        data_dir_id="d" * 64,
        control_socket_status="healthy",
        provider_check_status="available",
        dependencies_ready=True,
        dependency_versions=(
            "git=2.43.0,node=24.1.0,npm=11.0.0,uv=0.8.0,age=1.2.1,ssh=OpenSSH_9.6p1,python=3.12.10"
        ),
        problems=problems,
    )


def _run_doctor(report: ServerDoctorReport, *, machine_readable: bool) -> tuple[int, str, int]:
    machine = FakeDoctorMachine(report)
    argv = ["server", "doctor"]
    if machine_readable:
        argv.append("--machine-readable")
    args = build_parser().parse_args(argv)

    def handler(request, identity):
        return prepare_doctor_command(request, identity, machine=machine)

    output = StringIO()
    exit_code = run_server_command(args, identity=IDENTITY, handler=handler, stream=output)
    return exit_code, output.getvalue(), machine.calls


def test_doctor_renders_one_complete_report_through_both_cli_modes() -> None:
    exit_code, machine_output, calls = _run_doctor(_report(), machine_readable=True)

    assert exit_code == 0
    assert calls == 1
    events = [json.loads(line) for line in machine_output.splitlines()]
    assert [event["event"] for event in events] == ["plan", "step", "step"]
    assert events[-1]["step"]["state"] == "succeeded"
    fields = {item["name"]: item["value"] for item in events[-1]["step"]["fields"]}
    assert len(fields) == 48
    assert fields["overall_state"] == "healthy"
    assert fields["candidate_commit"] == "none"
    assert fields["running_commit"] == COMMIT
    assert fields["provider_check_status"] == "available"
    assert fields["update_operation_state"] == "none"
    assert fields["problems"] == "none"

    interactive_code, interactive, interactive_calls = _run_doctor(
        _report(), machine_readable=False
    )
    assert interactive_code == 0
    assert interactive_calls == 1
    for name, value in fields.items():
        assert f"{name}: {value}" in interactive


def test_doctor_returns_a_complete_failed_report_for_owned_problems() -> None:
    report = _report(problems=("runtime directory has the wrong type, owner, group, or mode",))

    exit_code, output, calls = _run_doctor(report, machine_readable=True)

    assert exit_code == 1
    assert calls == 1
    final = json.loads(output.splitlines()[-1])["step"]
    assert final["state"] == "failed"
    fields = {item["name"]: item["value"] for item in final["fields"]}
    assert fields["overall_state"] == "problems"
    assert fields["problems"] == report.problems[0]


def test_doctor_reports_the_exact_last_protected_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    destination = tmp_path / "backups"
    backup = ServerBackupConfig(
        destination=str(destination),
        age_recipient="age1qypqxpq9qcrsszg2pvxq6rs0zqg3yyc5z5tpwxqergd3c8g7rusqmwn7f2",
    )
    capture_id = "9c59550a-9787-466a-9435-1e59f0a9803f"
    archive = BackupArchiveReceipt(
        installation_id=INSTALLATION_ID,
        space_id=SPACE_ID,
        capture_id=capture_id,
        destination=str(destination),
        archive_name=(f"rcp-team-backup-v1-20260829T120000000000Z-{capture_id}.tar.age"),
        captured_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        protected_at=datetime(2026, 8, 29, 12, 1, tzinfo=UTC),
        capture_status="complete",
        age_version="1.2.1",
        age_recipient_fingerprint="a" * 64,
        archive_sha256="b" * 64,
        archive_size_bytes=4096,
        manifest_sha256="c" * 64,
        captured_bytes=2048,
        project_count=2,
        protected_project_count=2,
        uncaptured_project_count=0,
    )
    outcome = BackupRunOutcome(
        operation_id="cf4d29d0-a1bd-4d38-8620-242adf195bf6",
        installation_id=INSTALLATION_ID,
        destination=str(destination),
        started_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 29, 12, 2, tzinfo=UTC),
        status="protected",
        archive=archive,
        archive_receipt_sha256="d" * 64,
    )
    receipt_calls: list[dict[str, object]] = []
    monkeypatch.setattr(backup_owner, "read_backup_outcome", lambda *_args, **_kwargs: outcome)

    def read_receipt(_path, **kwargs):
        receipt_calls.append(kwargs)
        return archive

    monkeypatch.setattr(backup_owner, "read_backup_archive_receipt", read_receipt)

    def runner(argv: tuple[str, ...], *, cwd: Path | None = None):
        del cwd
        value = "active" if "--property=ActiveState" in argv else "enabled"
        return subprocess.CompletedProcess(argv, 0, value + "\n", "")

    problems: list[str] = []
    summary = LinuxServerDoctorMachine(layout, runner=runner)._inspect_backup(
        SimpleNamespace(installation_id=INSTALLATION_ID, backup=backup),
        service_uid=os.geteuid(),
        add_problem=problems.append,
    )

    assert problems == []
    assert summary.status == "protected"
    assert summary.archive == str(destination / archive.archive_name)
    assert summary.captured_bytes == 2048
    assert summary.protected_projects == 2
    assert summary.uncaptured_projects == 0
    assert receipt_calls[0]["expected_receipt_sha256"] == "d" * 64


def test_doctor_reports_an_unfinished_source_update_as_a_problem(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.update_checkpoints_root.mkdir(parents=True, mode=0o700)
    built = layout.update_checkpoints_root / f"built-candidate-{OTHER_COMMIT}.json"
    preflight = layout.update_checkpoints_root / "preflight.json"
    for path in (built, preflight):
        path.write_text("receipt\n", encoding="utf-8")
        path.chmod(0o600)
    operation = new_update_operation(
        operation_id=str(uuid.uuid4()),
        installation_id=INSTALLATION_ID,
        space_id=SPACE_ID,
        base_commit=COMMIT,
        candidate_commit=OTHER_COMMIT,
        base_instance_id=str(uuid.uuid4()),
        base_process_pid=421,
        built_receipt_path=built,
        built_receipt_sha256="a" * 64,
        preflight_receipt_path=preflight,
        preflight_receipt_sha256="b" * 64,
        update_root=layout.update_checkpoints_root,
    )
    publish_update_operation(
        operation,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    problems: list[str] = []

    summary = LinuxServerDoctorMachine(layout)._inspect_update(
        service_uid=os.geteuid(),
        installation_id=INSTALLATION_ID,
        add_problem=problems.append,
    )

    assert summary.state == "maintenance_closing"
    assert summary.candidate_commit == OTHER_COMMIT
    assert summary.restored_commit is None
    assert problems == ["unfinished source update requires sudo rcp server update re-entry"]


@pytest.mark.parametrize("state", ["committed", "rolled_back"])
def test_doctor_reports_a_selected_release_that_needs_runtime_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: str,
) -> None:
    layout = _layout(tmp_path)
    latest = SimpleNamespace(
        updated_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        operation_id=str(uuid.uuid4()),
        installation_id=INSTALLATION_ID,
        terminal=True,
        state=state,
        candidate_commit=OTHER_COMMIT,
        base_commit=COMMIT,
        failure="candidate verification failed" if state == "rolled_back" else None,
        runtime_failure="deferred runtime restart failed",
    )
    monkeypatch.setattr(
        "rcp.server_ops.update_cutover.update_operation_receipts",
        lambda _root, *, expected_uid: ((Path("/receipt"), latest, "a" * 64),),
    )
    monkeypatch.setattr(
        "rcp.server_ops.update_checkpoint.unfinished_rollback_journals",
        lambda _root, *, expected_uid: (),
    )
    problems: list[str] = []

    summary = LinuxServerDoctorMachine(layout)._inspect_update(
        service_uid=os.geteuid(),
        installation_id=INSTALLATION_ID,
        add_problem=problems.append,
    )

    assert summary.state == state
    assert summary.failure == "deferred runtime restart failed"
    assert problems == [
        "selected source release needs safe runtime restart via sudo rcp server update"
    ]


def test_running_release_identity_and_health_are_exact(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    release = layout.release_dir(COMMIT)
    dist = release / "web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<main>one</main>", encoding="utf-8")
    layout.current_release.parent.mkdir(parents=True)
    layout.current_release.symlink_to(release)

    identity = capture_installed_release_identity(layout, working_dir=release)
    metadata = ServerMetadata.create(
        layout.data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        running_commit=identity.commit,
        web_build_id=identity.web_build_id,
    )
    app = create_named_app(data_dir=layout.data_dir, instance_metadata=metadata)

    with TestClient(app) as client:
        health = client.get("/api/health").json()

    assert identity.commit == COMMIT
    assert identity.web_build_id == web_build_identity(dist)
    assert health["running_commit"] == COMMIT
    assert health["web_build_id"] == identity.web_build_id
    restored = ServerMetadata.from_dict(metadata.as_dict())
    assert restored == metadata
    malformed = metadata.as_dict()
    malformed.pop("web_build_id")
    with pytest.raises(ServerMetadataError, match="unsupported shape"):
        ServerMetadata.from_dict(malformed)


def test_web_build_identity_covers_paths_and_bytes_and_rejects_symlinks(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("one", encoding="utf-8")
    script = assets / "app.js"
    script.write_text("one", encoding="utf-8")
    first = web_build_identity(dist)

    script.write_text("two", encoding="utf-8")
    assert web_build_identity(dist) != first

    script.unlink()
    script.symlink_to(dist / "index.html")
    with pytest.raises(ServerMetadataError, match="non-regular"):
        web_build_identity(dist)


def test_web_build_identity_wraps_traversal_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("ready", encoding="utf-8")

    def fail_traversal(_path: Path, _pattern: str):
        raise OSError("fixture traversal failure")

    monkeypatch.setattr(Path, "rglob", fail_traversal)

    with pytest.raises(ServerMetadataError, match="could not be inspected"):
        web_build_identity(dist)


def test_default_doctor_runner_disables_optional_git_writes(monkeypatch) -> None:
    observed = {}

    def run(argv, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(server_doctor.subprocess, "run", run)

    server_doctor._run_read_only(("git", "status"))

    assert observed["env"]["GIT_OPTIONAL_LOCKS"] == "0"
    assert observed["check"] is False


@pytest.mark.parametrize(
    ("managed", "current", "running", "expected"),
    [
        (COMMIT, COMMIT, COMMIT, "aligned"),
        (OTHER_COMMIT, COMMIT, COMMIT, "candidate_pending"),
        (OTHER_COMMIT, OTHER_COMMIT, COMMIT, "restart_pending"),
        (COMMIT, OTHER_COMMIT, "c" * 40, "inconsistent"),
        (None, COMMIT, COMMIT, "unavailable"),
    ],
)
def test_release_relationship_distinguishes_update_from_corruption(
    managed: str | None,
    current: str | None,
    running: str | None,
    expected: str,
) -> None:
    assert release_relationship(managed, current, running) == expected


def test_doctor_does_not_traverse_an_unsafe_release_directory(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.releases_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    layout.release_dir(COMMIT).symlink_to(outside)
    calls: list[tuple[str, ...]] = []
    problems: list[str] = []

    def runner(argv: tuple[str, ...], *, cwd: Path | None = None):
        del cwd
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, COMMIT + "\n", "")

    result = LinuxServerDoctorMachine(layout, runner=runner)._inspect_release(
        COMMIT,
        label="current",
        service_uid=os.getuid(),
        service_gid=os.getgid(),
        add_problem=problems.append,
    )

    assert result is None
    assert calls == []
    assert problems == ["current release directory has the wrong type, owner, group, or mode"]


def test_doctor_does_not_probe_a_metadata_selected_control_socket(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.data_dir.mkdir(parents=True)
    metadata = ServerMetadata.create(
        layout.data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=tmp_path / "wrong.sock",
        running_commit=COMMIT,
        web_build_id=WEB_BUILD_ID,
    )
    probe_calls = 0
    problems: list[str] = []

    def probe(_metadata: ServerMetadata, _expected_uid: int):
        nonlocal probe_calls
        probe_calls += 1
        raise AssertionError("an untrusted socket must not be probed")

    observed, result, status = LinuxServerDoctorMachine(
        layout,
        metadata_reader=lambda _data_dir: metadata,
        control_probe=probe,
    )._inspect_process(
        service_uid=os.getuid(),
        main_pid=metadata.pid,
        add_problem=problems.append,
    )

    assert observed is metadata
    assert result is None
    assert status == "identity_mismatch"
    assert probe_calls == 0
    assert problems == ["running process metadata names a different control socket"]


def test_doctor_names_each_live_operation_during_member_removal() -> None:
    member_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    episode_id = str(uuid.uuid4())
    snapshot = ServerControlMemberSnapshot(
        member_id=member_id,
        member_display_name="Alice",
        removal_started_at=datetime.now(UTC).isoformat(),
        removed_at=None,
        last_authenticating_member=False,
        project_ids=(),
        orphaned_project_ids=(),
        orphaned_project_labels=(),
        active_task_ids=(task_id,),
        active_episode_ids=(episode_id,),
        active_token_ids=(),
        browser_session_count=0,
        space_invitation_ids=(),
        project_invitation_ids=(),
        boundary_sha256="a" * 64,
    )

    assert _member_removal_problems(snapshot) == (
        f"member removal remains in progress: {member_id}",
        f"member removal has a live task: {task_id}",
        f"member removal has a live episode: {episode_id}",
    )


def test_doctor_rejects_systemd_drop_ins(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    uid = os.getuid()
    gid = os.getgid()
    _prepare_layout(layout, uid=uid, gid=gid)
    problems: list[str] = []
    runner = _HealthyRunner(
        layout=layout,
        commit=COMMIT,
        pid=os.getpid(),
        systemd_overrides={"DropInPaths": "/etc/systemd/system/rcp.service.d/override.conf"},
    )

    _active, _enabled, _pid, reload_mode = LinuxServerDoctorMachine(
        layout,
        runner=runner,
    )._inspect_service(problems.append)

    assert reload_mode == "unknown"
    assert "systemd has not loaded the exact unit without overrides" in problems


def test_linux_doctor_reads_a_healthy_installed_layout_without_mutating_it(
    tmp_path: Path,
) -> None:
    runtime_temp = tempfile.TemporaryDirectory(prefix="rcpd-", dir="/tmp")
    runtime_dir = Path(runtime_temp.name)
    layout = replace(
        _layout(tmp_path),
        runtime_dir=runtime_dir,
        control_socket=runtime_dir / "control.sock",
    )
    uid = os.getuid()
    gid = os.getgid()
    _prepare_layout(layout, uid=uid, gid=gid)
    release = _prepare_release(layout, COMMIT)
    web_identity = web_build_identity(release / "web" / "dist")
    metadata = ServerMetadata.create(
        layout.data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=layout.control_socket,
        running_commit=COMMIT,
        web_build_id=web_identity,
    )
    metadata_path(layout.data_dir).write_text("{}\n", encoding="utf-8")
    os.chmod(metadata_path(layout.data_dir), 0o600)
    for name in ("rcp.sqlite3", "rcp.lock"):
        path = layout.data_dir / name
        path.write_text("fixture\n", encoding="utf-8")
        os.chmod(path, 0o600)
    layout.current_release.symlink_to(release)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(layout.control_socket))
    os.chown(layout.runtime_dir, uid, gid)
    os.chown(layout.control_socket, uid, gid)
    os.chmod(layout.control_socket, 0o600)
    before = _tree_snapshot(tmp_path)

    config = SimpleNamespace(
        installation_id=INSTALLATION_ID,
        service_account="rcp",
        service_unit="rcp.service",
        source=ServerSourceConfig(
            origin="https://github.com/openai/rcp.git",
            authentication="public",
        ),
        paths=SimpleNamespace(model_dump=lambda: layout.recorded_paths()),
        backup=None,
    )

    def config_loader(_path: Path):
        return config

    def metadata_reader(_data_dir: Path) -> ServerMetadata:
        return metadata

    def control_probe(observed: ServerMetadata, expected_uid: int):
        assert observed is metadata
        assert expected_uid == uid
        return SimpleNamespace(
            instance_id=metadata.instance_id,
            pid=metadata.pid,
            data_dir_id=metadata.data_dir_id,
            space_id=SPACE_ID,
            operations=SERVER_CONTROL_OPERATIONS,
        )

    runner = _HealthyRunner(layout=layout, commit=COMMIT, pid=metadata.pid)
    try:
        report = LinuxServerDoctorMachine(
            layout,
            config_loader=config_loader,
            metadata_reader=metadata_reader,
            control_probe=control_probe,
            runner=runner,
            service_identity=(uid, gid),
            root_identity=(uid, gid),
        ).inspect()
    finally:
        listener.close()

    assert report.overall_state == "healthy", report.problems
    assert report.problems == ()
    assert report.current_commit == report.running_commit == report.managed_main_head == COMMIT
    assert report.current_web_build_id == report.running_web_build_id == web_identity
    assert report.control_socket_status == "healthy"
    assert report.dependencies_ready is True
    assert _tree_snapshot(tmp_path) == before
    runtime_temp.cleanup()


class _HealthyRunner:
    def __init__(
        self,
        *,
        layout: ServerLayout,
        commit: str,
        pid: int,
        systemd_overrides: dict[str, str] | None = None,
    ) -> None:
        self.layout = layout
        self.commit = commit
        self.pid = pid
        self.systemd_overrides = systemd_overrides or {}

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        if argv[0] == "systemctl":
            property_name = next(
                value.split("=", 1)[1] for value in argv if value.startswith("--property=")
            )
            values = {
                "ActiveState": "active",
                "UnitFileState": "enabled",
                "MainPID": str(self.pid),
                "NeedDaemonReload": "no",
                "FragmentPath": str(self.layout.systemd_unit),
                "DropInPaths": "",
                **self.systemd_overrides,
            }
            return subprocess.CompletedProcess(argv, 0, values[property_name] + "\n", "")
        if argv[0] == "git":
            if "remote" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, "https://github.com/openai/rcp.git\n", ""
                )
            if "symbolic-ref" in argv:
                return subprocess.CompletedProcess(argv, 0, "main\n", "")
            if "status" in argv:
                return subprocess.CompletedProcess(argv, 0, "", "")
            if "rev-parse" in argv:
                return subprocess.CompletedProcess(argv, 0, self.commit + "\n", "")
            if "merge-base" in argv:
                return subprocess.CompletedProcess(argv, 0, "", "")
        versions = {
            ("git", "--version"): ("git version 2.43.0\n", ""),
            ("node", "--version"): ("v24.1.0\n", ""),
            ("npm", "--version"): ("11.0.0\n", ""),
            ("uv", "--version"): ("uv 0.8.0\n", ""),
            ("age", "--version"): ("1.2.1\n", ""),
            ("ssh", "-V"): ("", "OpenSSH_9.6p1\n"),
        }
        if argv in versions:
            stdout, stderr = versions[argv]
            return subprocess.CompletedProcess(argv, 0, stdout, stderr)
        if argv[-1:] == ("--version",) and Path(argv[0]).name == "python":
            return subprocess.CompletedProcess(argv, 0, "Python 3.12.10\n", "")
        return subprocess.CompletedProcess(argv, 1, "", "")


def _layout(root: Path) -> ServerLayout:
    home = root / "home" / "rcp"
    server = home / "rcp-server"
    return ServerLayout(
        service_account="rcp",
        service_home=home,
        server_root=server,
        source_checkout=server / "source",
        releases_root=server / "releases",
        data_dir=server / "data",
        projects_root=server / "projects",
        credentials_root=server / "credentials",
        update_checkpoints_root=server / "update-checkpoints",
        restore_operations_root=server / "restore-operations",
        codex_state_root=home / ".codex",
        claude_state_root=home / ".claude",
        ssh_state_root=home / ".ssh",
        config_path=root / "etc" / "rcp" / "server.toml",
        current_release=root / "etc" / "rcp" / "current",
        runtime_dir=root / "run" / "rcp",
        control_socket=root / "run" / "rcp" / "control.sock",
        cli_wrapper=root / "usr" / "local" / "bin" / "rcp",
        systemd_unit=root / "etc" / "systemd" / "system" / "rcp.service",
        service_unit_name="rcp.service",
    )


def _prepare_layout(layout: ServerLayout, *, uid: int, gid: int) -> None:
    del uid, gid
    for path in (
        layout.service_home,
        layout.server_root,
        layout.releases_root,
        layout.data_dir,
        layout.projects_root,
        layout.credentials_root,
        layout.update_checkpoints_root,
        layout.restore_operations_root,
        layout.codex_state_root,
        layout.claude_state_root,
        layout.ssh_state_root,
        layout.source_checkout,
        layout.runtime_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    layout.config_path.parent.mkdir(parents=True)
    os.chmod(layout.config_path.parent, 0o750)
    layout.config_path.write_text("fixture\n", encoding="utf-8")
    os.chmod(layout.config_path, 0o640)
    layout.cli_wrapper.parent.mkdir(parents=True)
    layout.cli_wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(layout.cli_wrapper, 0o755)
    layout.systemd_unit.parent.mkdir(parents=True)
    layout.systemd_unit.write_text(server_service_unit_text(), encoding="utf-8")
    os.chmod(layout.systemd_unit, 0o644)


def _prepare_release(layout: ServerLayout, commit: str) -> Path:
    release = layout.release_dir(commit)
    for path in (release / ".venv" / "bin", release / "web" / "dist"):
        path.mkdir(parents=True)
    for path, content in (
        (release / ".venv" / "bin" / "rcp", "#!/bin/sh\n"),
        (release / ".venv" / "bin" / "python", "python\n"),
        (release / "web" / "dist" / "index.html", "<main>ready</main>\n"),
    ):
        path.write_text(content, encoding="utf-8")
    return release


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, int], ...]:
    entries = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        entries.append((str(path.relative_to(root)), stat.S_IFMT(info.st_mode), info.st_mtime_ns))
    return tuple(entries)
