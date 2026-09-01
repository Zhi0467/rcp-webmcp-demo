from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from contextlib import nullcontext
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import rcp.server_ops.restore as restore_code
from rcp.__main__ import build_parser
from rcp.api import create_app
from rcp.core.models import AuthorizedHuman
from rcp.server_ops.backup import (
    BackupRunRefused,
    LinuxBackupRunMachine,
    _write_deterministic_archive,
    backup_run_coordination_lock,
)
from rcp.server_ops.backup_capture import _database_schema_sha256
from rcp.server_ops.backup_models import BackupArchiveManifest, BackupFileEntry
from rcp.server_ops.cli import SERVER_CLI_EXIT_OPERATOR_ACTION, CallerIdentity, run_server_command
from rcp.server_ops.doctor import LinuxServerDoctorMachine
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.restore import (
    SUPPORTED_RESTORE_DATABASE_SCHEMAS,
    LinuxRestoreMachine,
    RestoreConfirmation,
    RestoreRefused,
    detach_restore_database,
    prepare_restore_command,
    read_restore_journal,
    restore_journal_path,
    unfinished_restore_operation,
)
from rcp.server_ops.update import server_update_operation_lock
from rcp.storage import (
    AppStore,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
)

COMMIT = "a" * 40
CAPTURED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


class _StoppedService:
    def __init__(self, release: Path) -> None:
        self.release = release
        self.stop_calls = 0

    def current_release(self) -> Path:
        return self.release

    def fence_stopped_disabled(self) -> None:
        self.stop_calls += 1


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


def _prepare_layout(layout: ServerLayout) -> Path:
    for path in (
        layout.service_home,
        layout.server_root,
        layout.data_dir,
        layout.restore_operations_root,
        layout.releases_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    layout.config_path.parent.mkdir(parents=True)
    os.chmod(layout.config_path.parent, 0o750)
    release = layout.release_dir(COMMIT)
    release.mkdir()
    return release


def _config(layout: ServerLayout):
    return SimpleNamespace(paths=SimpleNamespace(model_dump=lambda: layout.recorded_paths()))


def _provisioning_request(store: AppStore):
    member = store.preprovision_team_member("Alice")
    request = store.create_project_provisioning_request(
        kind="create_team_project",
        authorized_by=AuthorizedHuman(
            space_id=store.space_id,
            user_id=member.user_id,
            display_name=member.display_name,
        ),
        name="Restored project",
        state_repository="paper",
        project_truth_scope=["paper"],
        default_run_truth_scope=["paper"],
        machines=[
            ProjectProvisioningMachineIntent(
                alias="server",
                location="local",
                os_account="rcp",
                central_root=str(DEFAULT_SERVER_LAYOUT.projects_root),
            )
        ],
        repositories=[
            ProjectProvisioningRepositoryIntent(
                alias="paper",
                repository=parse_github_repository_ref("git@github.com:OpenAI/RCP-paper.git"),
                machine_alias="server",
            )
        ],
        provider_checks=[
            ProjectProvisioningProviderIntent(
                profile="seed",
                provider="codex",
                runtime_id="codex:exec",
                model="gpt-test",
                reasoning="medium",
                machine_alias="server",
            )
        ],
    )
    return store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="machine-started",
        phase="provisioning_start",
        expected_revision=request.revision,
        expected_status=request.status,
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )


def _archive(
    root: Path,
    *,
    schema_sha256: str | None = None,
    source_commit: str = COMMIT,
) -> tuple[Path, AppStore, BackupArchiveManifest]:
    root.mkdir(parents=True)
    capture = root / "capture"
    capture.mkdir()
    os.chmod(capture, 0o700)
    store, _bootstrap = AppStore.initialize_team_space(capture / "rcp.sqlite3", "Archive lab")
    _provisioning_request(store)
    sqlite = capture / "rcp.sqlite3"
    sqlite_sha256 = hashlib.sha256(sqlite.read_bytes()).hexdigest()
    manifest = BackupArchiveManifest(
        space_id=store.space_id,
        space_name=store.space_name,
        rcp_source_commit=source_commit,
        database_schema_sha256=schema_sha256 or _database_schema_sha256(store),
        captured_at=CAPTURED_AT,
        sqlite_snapshot=BackupFileEntry(
            archive_path="database/rcp.sqlite3",
            source_relative_path="rcp.sqlite3",
            group="sqlite_snapshot",
            sha256=sqlite_sha256,
            size_bytes=sqlite.stat().st_size,
        ),
        encryption_recipient_fingerprint="b" * 64,
        installation_id="123e4567-e89b-42d3-a456-426614174000",
        excluded_app_data_entries=(),
        uncaptured_app_data_entries=(),
        projects=(),
        status="complete",
        total_bytes=sqlite.stat().st_size,
    )
    archive = root / "archive.tar.age"
    with archive.open("wb") as stream:
        _write_deterministic_archive(stream, manifest, capture)
    return archive, store, manifest


def _machine(
    tmp_path: Path,
    archive: Path,
    *,
    compatible: bool = True,
) -> tuple[LinuxRestoreMachine, ServerLayout, _StoppedService, Path]:
    del archive
    layout = _layout(tmp_path / "server")
    release = _prepare_layout(layout)
    service = _StoppedService(release)
    identity = tmp_path / "identity.txt"
    identity.write_text("AGE-SECRET-KEY-1TEST\n", encoding="utf-8")
    os.chmod(identity, 0o600)

    def decrypt(source: Path, _identity: Path, destination: Path) -> None:
        shutil.copyfile(source, destination)

    machine = LinuxRestoreMachine(
        layout,
        config_loader=lambda _path: _config(layout),
        service_control=service,  # type: ignore[arg-type]
        service_identity=(os.getuid(), os.getgid()),
        root_identity=(os.getuid(), os.getgid()),
        decryptor=decrypt,
        commit_compatible=lambda *_args: compatible,
        detach_worker=lambda database, actor, at: detach_restore_database(
            database,
            confirmed_by=actor,
            detached_at=at,
        ),
        clock=lambda: CAPTURED_AT,
    )
    return machine, layout, service, identity


def _confirmation(layout: ServerLayout) -> RestoreConfirmation:
    return RestoreConfirmation(
        confirmed_data_dir=str(layout.data_dir),
        confirmed_by="root@lab uid=0",
        confirmed_at=CAPTURED_AT,
    )


def test_restore_cli_pauses_with_the_exact_confirmed_destination(tmp_path: Path) -> None:
    class PlanOnlyMachine:
        def configured_data_dir(self) -> Path:
            return tmp_path / "configured-data"

        def __getattr__(self, name: str):
            raise AssertionError(f"restore work began before destination confirmation: {name}")

    arguments = build_parser().parse_args(
        (
            "server",
            "restore",
            "/backups/lab.age",
            "--identity-file",
            "/safe/identity.txt",
        )
    )
    output = StringIO()
    identity = CallerIdentity(uid=0, username="root", host="lab.example")

    exit_code = run_server_command(
        arguments,
        identity=identity,
        input_stream=BytesIO(),
        stream=output,
        handler=lambda request, caller: prepare_restore_command(
            request,
            caller,
            machine=PlanOnlyMachine(),  # type: ignore[arg-type]
            resume_executable=Path("/usr/local/bin/rcp"),
        ),
    )

    assert exit_code == SERVER_CLI_EXIT_OPERATOR_ACTION
    rendered = output.getvalue()
    assert str(tmp_path / "configured-data") in rendered
    assert "--confirm-data-dir" in rendered
    assert "AGE-SECRET-KEY" not in rendered


def test_restore_cli_accepts_confirmed_destination_before_archive_work(
    tmp_path: Path,
) -> None:
    class BoundaryMachine:
        def configured_data_dir(self) -> Path:
            return tmp_path / "configured-data"

        def admission(self):
            return nullcontext()

        def stage_candidate(self, *_args, **_kwargs):
            raise RestoreRefused("The test archive boundary stopped restore as intended.")

    arguments = build_parser().parse_args(
        (
            "server",
            "restore",
            "/backups/lab.age",
            "--identity-file",
            "/safe/identity.txt",
            "--confirm-data-dir",
            str(tmp_path / "configured-data"),
            "--machine-readable",
        )
    )
    output = StringIO()

    exit_code = run_server_command(
        arguments,
        identity=CallerIdentity(uid=0, username="root", host="runnervm76wwg"),
        input_stream=BytesIO(),
        stream=output,
        handler=lambda request, caller: prepare_restore_command(
            request,
            caller,
            machine=BoundaryMachine(),  # type: ignore[arg-type]
            resume_executable=Path("/usr/local/bin/rcp"),
        ),
    )

    assert exit_code == 1
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    states = [
        (event["step"]["number"], event["step"]["state"])
        for event in events
        if event["event"] == "step"
    ]
    assert states == [(1, "running"), (1, "succeeded"), (2, "running"), (2, "failed")]
    assert events[-1]["step"]["message"] == (
        "The test archive boundary stopped restore as intended."
    )


def test_restore_builds_one_detached_offline_sqlite_candidate(tmp_path: Path) -> None:
    archive, source, manifest = _archive(tmp_path / "backup")
    request = source.project_provisioning_requests()[0]
    source_receipts = source.project_provisioning_step_receipts(request.request_id)
    machine, layout, service, identity = _machine(tmp_path, archive)

    candidate = machine.stage_candidate(archive, identity, confirmed_by="root@lab uid=0")

    assert not any(layout.data_dir.iterdir())
    assert not restore_journal_path(layout).exists()
    restored = AppStore.open_read_only_snapshot(candidate.sqlite_path)
    detached = restored.project_provisioning_request(request.request_id)
    assert detached is not None
    assert detached.status == "operator_action_needed"
    assert detached.operator_action is not None
    assert detached.operator_action.phase == "restore_reentry"
    assert detached.machines[0].resolved_central_root is None
    assert detached.repositories[0].resolved_path is None
    assert detached.repositories[0].git_check.status == "pending"
    assert detached.provider_checks[0].status == "pending"
    assert restored.project_provisioning_step_receipts(request.request_id) == source_receipts

    journal = machine.journal_candidate(candidate, _confirmation(layout))
    assert journal.phase == "archive_verified"
    assert journal.manifest == manifest
    assert journal.machine_local_operations == "not_restored"
    journal_text = restore_journal_path(layout).read_text(encoding="utf-8")
    assert str(identity) not in journal_text
    assert "AGE-SECRET-KEY" not in journal_text
    assert service.stop_calls == 1
    assert not any(layout.data_dir.iterdir())

    installed = machine.install_sqlite_candidate(journal)
    verified = machine.verify_offline_candidate(installed)

    assert verified.phase == "sqlite_restored"
    assert service.stop_calls == 2
    assert [path.name for path in layout.data_dir.iterdir()] == ["rcp.sqlite3"]
    target_sha256 = hashlib.sha256((layout.data_dir / "rcp.sqlite3").read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="replacement restore is incomplete"):
        create_app(data_dir=layout.data_dir, server_layout=layout)
    assert (
        hashlib.sha256((layout.data_dir / "rcp.sqlite3").read_bytes()).hexdigest() == target_sha256
    )
    target = AppStore(layout.data_dir / "rcp.sqlite3")
    assert target.space_id == manifest.space_id
    assert target.project_provisioning_request(request.request_id) == detached
    assert unfinished_restore_operation(layout, expected_uid=os.getuid()) == verified
    problems: list[str] = []
    assert LinuxServerDoctorMachine(
        layout,
        service_identity=(os.getuid(), os.getgid()),
    )._inspect_restore(  # noqa: SLF001 - focused owner integration
        service_uid=os.getuid(),
        add_problem=problems.append,
    )
    assert problems == ["unfinished replacement restore requires sudo rcp server restore re-entry"]


def test_unsafe_restore_root_blocks_installed_startup_before_database_mutation(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path / "server")
    _prepare_layout(layout)
    store, _bootstrap = AppStore.initialize_team_space(
        layout.data_dir / "rcp.sqlite3",
        "Unsafe restore lab",
    )
    before = (layout.data_dir / "rcp.sqlite3").read_bytes()
    os.chmod(layout.restore_operations_root, 0o755)

    with pytest.raises(RuntimeError, match="restore state is unsafe"):
        create_app(data_dir=layout.data_dir, server_layout=layout)

    assert (layout.data_dir / "rcp.sqlite3").read_bytes() == before
    assert store.space_kind == "team"


def test_restore_journal_serializes_update_and_backup_machine_owners(tmp_path: Path) -> None:
    archive, _source, _manifest = _archive(tmp_path / "backup")
    machine, layout, _service, identity = _machine(tmp_path, archive)
    candidate = machine.stage_candidate(archive, identity, confirmed_by="root@lab uid=0")

    with (
        server_update_operation_lock(
            layout,
            root_uid=os.getuid(),
            root_gid=os.getgid(),
            service_gid=os.getgid(),
        ),
        pytest.raises(RestoreRefused, match="source update or protected backup"),
    ):
        machine.journal_candidate(candidate, _confirmation(layout))
    with (
        backup_run_coordination_lock(layout),
        pytest.raises(RestoreRefused, match="source update or protected backup"),
    ):
        machine.journal_candidate(candidate, _confirmation(layout))

    journal = machine.journal_candidate(candidate, _confirmation(layout))
    with pytest.raises(BackupRunRefused, match="unfinished replacement restore"):
        LinuxBackupRunMachine(layout).run()
    assert journal.phase == "archive_verified"


def test_restore_reentry_rebuilds_a_missing_candidate_with_identical_bytes(
    tmp_path: Path,
) -> None:
    archive, _source, _manifest = _archive(tmp_path / "backup")
    machine, layout, _service, identity = _machine(tmp_path, archive)
    candidate = machine.stage_candidate(archive, identity, confirmed_by="root@lab uid=0")
    journal = machine.journal_candidate(candidate, _confirmation(layout))
    shutil.rmtree(candidate.root)

    rebuilt = machine.stage_candidate(archive, identity, confirmed_by="root@lab uid=0")

    assert rebuilt.detached_at == journal.detached_at
    assert rebuilt.sqlite_sha256 == journal.candidate_sqlite_sha256
    assert machine.journal_candidate(rebuilt, _confirmation(layout)) == journal


def test_restore_recovers_crash_after_database_publish_before_phase_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, _source, _manifest = _archive(tmp_path / "backup")
    machine, layout, _service, identity = _machine(tmp_path, archive)
    candidate = machine.stage_candidate(archive, identity, confirmed_by="root@lab uid=0")
    journal = machine.journal_candidate(candidate, _confirmation(layout))
    real_write = restore_code.write_restore_journal

    def crash_on_advance(updated, *args, **kwargs):
        if updated.phase == "sqlite_restored":
            raise OSError("simulated power loss")
        return real_write(updated, *args, **kwargs)

    monkeypatch.setattr(restore_code, "write_restore_journal", crash_on_advance)
    with pytest.raises(OSError, match="power loss"):
        machine.install_sqlite_candidate(journal)
    assert (layout.data_dir / "rcp.sqlite3").is_file()
    assert read_restore_journal(layout, expected_uid=os.getuid()).phase == "archive_verified"

    monkeypatch.setattr(restore_code, "write_restore_journal", real_write)
    recovered = machine.install_sqlite_candidate(journal)

    assert recovered.phase == "sqlite_restored"
    machine.verify_offline_candidate(recovered)


@pytest.mark.parametrize(
    ("schema_sha256", "compatible", "match"),
    [
        ("f" * 64, True, "unsupported newer database boundary"),
        (None, False, "code the installed restore release does not support"),
    ],
)
def test_unknown_archive_boundaries_fail_before_target_or_journal_mutation(
    tmp_path: Path,
    schema_sha256: str | None,
    compatible: bool,
    match: str,
) -> None:
    archive, _source, _manifest = _archive(
        tmp_path / "backup",
        schema_sha256=schema_sha256,
        source_commit="c" * 40,
    )
    machine, layout, _service, identity = _machine(tmp_path, archive, compatible=compatible)

    with pytest.raises(RestoreRefused, match=match):
        machine.stage_candidate(archive, identity, confirmed_by="root@lab uid=0")

    assert not any(layout.data_dir.iterdir())
    assert not restore_journal_path(layout).exists()


def test_archive_hash_failure_and_nonempty_target_fail_without_replacement(
    tmp_path: Path,
) -> None:
    archive, _source, _manifest = _archive(tmp_path / "backup")
    machine, layout, _service, identity = _machine(tmp_path, archive)
    corrupted = bytearray(archive.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0xFF
    archive.write_bytes(corrupted)
    with pytest.raises(RestoreRefused):
        machine.stage_candidate(archive, identity, confirmed_by="root@lab uid=0")
    assert not any(layout.data_dir.iterdir())

    archive, _source, _manifest = _archive(tmp_path / "backup-2")
    machine, layout, _service, identity = _machine(tmp_path / "other", archive)
    candidate = machine.stage_candidate(archive, identity, confirmed_by="root@lab uid=0")
    unexpected = layout.data_dir / "keep.txt"
    unexpected.write_text("do not replace\n", encoding="utf-8")
    with pytest.raises(RestoreRefused, match="not fresh and empty"):
        machine.journal_candidate(candidate, _confirmation(layout))
    assert unexpected.read_text(encoding="utf-8") == "do not replace\n"
    assert not restore_journal_path(layout).exists()


def test_provisioning_restore_detachment_is_idempotent_and_keeps_step_receipts(
    tmp_path: Path,
) -> None:
    store, _bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Lease lab")
    running = _provisioning_request(store)
    receipts = store.project_provisioning_step_receipts(running.request_id)

    store.detach_restored_lifecycle(
        diagnostic="Replacement restore invalidated old machine authority.",
        confirmed_by="root@lab uid=0",
        detached_at=CAPTURED_AT.isoformat(),
    )
    first = store.project_provisioning_request(running.request_id)
    assert first is not None
    assert first.status == "operator_action_needed"
    assert first.revision == running.revision + 1
    assert store.project_provisioning_step_receipts(running.request_id) == receipts

    store.detach_restored_lifecycle(
        diagnostic="Replacement restore invalidated old machine authority.",
        confirmed_by="root@lab uid=0",
        detached_at=CAPTURED_AT.isoformat(),
    )

    assert store.project_provisioning_request(running.request_id) == first
    assert store.project_provisioning_step_receipts(running.request_id) == receipts


def test_restore_schema_registry_covers_current_and_immutable_upgrade_boundaries(
    tmp_path: Path,
) -> None:
    current, _bootstrap = AppStore.initialize_team_space(tmp_path / "current.sqlite3", "Current")
    assert _database_schema_sha256(current) in SUPPORTED_RESTORE_DATABASE_SCHEMAS
    for fixture in sorted(
        path for path in Path("tests/fixtures/server_upgrade").iterdir() if path.is_dir()
    ):
        compressed = fixture / "data" / "rcp.sqlite3.gz"
        database = tmp_path / f"{fixture.name}.sqlite3"
        database.write_bytes(gzip.decompress(compressed.read_bytes()))
        assert (
            restore_code._database_schema_sha256(  # noqa: SLF001 - compatibility boundary
                database
            )
            in SUPPORTED_RESTORE_DATABASE_SCHEMAS
        )
