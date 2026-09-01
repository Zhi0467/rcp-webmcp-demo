from __future__ import annotations

import builtins
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
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rcp.server_ops.backup_project_files as project_files_module
import rcp.server_ops.rehearsal as rehearsal_module
from rcp.__main__ import instance_lock
from rcp.api import create_app
from rcp.background import StartupEffectBlocked, StartupEffectFence
from rcp.config import load_manifest
from rcp.core.models import AuthorizedHuman
from rcp.history import HistoryManager
from rcp.server_ops.backup_capture import (
    BackupCaptureCoordinator,
    BackupSnapshotProjectInventory,
    BackupSQLiteCaptureReceipt,
    write_immutable_backup_receipt,
)
from rcp.server_ops.backup_checkout import BackupCheckoutHostUnavailable
from rcp.server_ops.backup_models import (
    BackupAppDataCapturePlan,
    BackupCheckoutRecoveryDescriptor,
    BackupFileEntry,
    BackupManifestConfiguration,
    BackupRecoveryMachine,
    BackupRecoveryRepository,
)
from rcp.server_ops.backup_project_files import BackupProjectFileCaptureReceipt
from rcp.server_ops.control import ServerControlBackupCaptureResult
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.rehearsal import (
    CandidateRehearsalCoordinator,
    CandidateRehearsalRefused,
    RehearsalOverlay,
    StartupRecoveryReadModel,
    build_rehearsal_overlay,
    run_candidate_child,
    run_candidate_migration,
    verified_candidate_receipt_path,
)
from rcp.server_ops.update import BuiltCandidateReceipt
from rcp.server_runtime import ServerMetadata
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.sources import ImportedProviderSourceStore
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
    ProjectRecord,
)
from rcp.transfer import TransferArchiveEntry

BASE_COMMIT = "a" * 40
CANDIDATE_COMMIT = "b" * 40
WEB_BUILD_ID = "sha256:" + ("c" * 64)
FINGERPRINT = "SHA256:" + ("A" * 43)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _initialize_checkout(repository: Path) -> str:
    repository.mkdir(parents=True, mode=0o700)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "RCP Rehearsal Test")
    _git(repository, "config", "user.email", "rehearsal@example.test")
    (repository / "README.md").write_text("candidate rehearsal fixture\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-q", "-m", "fixture")
    _git(repository, "remote", "add", "origin", "git@github.com:openai/rcp.git")
    _git(
        repository,
        "config",
        "remote.origin.pushurl",
        "git@github.com:openai/rcp.git",
    )
    return _git(repository, "rev-parse", "HEAD")


def _write_manifest(
    repository: Path,
    *,
    account: str,
    provider_home: Path,
) -> Path:
    research = repository / ".research"
    research.mkdir(mode=0o700)
    manifest = research / "manifest.toml"
    manifest.write_text(
        f'''name = "Candidate rehearsal project"

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
    return manifest


def _recovery_descriptor(
    manifest_path: Path,
    *,
    project_id: str,
    space_id: str,
    central_root: Path,
    checkout_commit: str,
    account: str,
) -> BackupCheckoutRecoveryDescriptor:
    manifest = load_manifest(manifest_path)
    configuration = BackupManifestConfiguration.from_manifest(manifest)
    return BackupCheckoutRecoveryDescriptor(
        request_id=str(uuid.uuid4()),
        project_id=project_id,
        home_space_id=space_id,
        completed_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        final_review_digest="d" * 64,
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
                resolved_path=str(manifest.repository_map["repo"].path),
                git_commit=checkout_commit,
                deploy_key_label=f"rcp:{space_id}:{project_id}:repo",
                public_key_fingerprint=FINGERPRINT,
            ),
        ),
    )


def _remote_recovery_descriptor(
    manifest_path: Path,
    *,
    project_id: str,
    space_id: str,
    central_root: str,
) -> BackupCheckoutRecoveryDescriptor:
    manifest = load_manifest(manifest_path)
    configuration = BackupManifestConfiguration.from_manifest(manifest)
    return BackupCheckoutRecoveryDescriptor(
        request_id=str(uuid.uuid4()),
        project_id=project_id,
        home_space_id=space_id,
        completed_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        final_review_digest="2" * 64,
        configuration=configuration,
        configuration_sha256=configuration.sha256,
        machines=(
            BackupRecoveryMachine(
                alias="remote",
                location="ssh",
                host="unreachable.example.test",
                os_account="rcp",
                resolved_central_root=central_root,
            ),
        ),
        repositories=(
            BackupRecoveryRepository(
                alias="repo",
                repository=parse_github_repository_ref("git@github.com:openai/rcp.git"),
                machine_alias="remote",
                resolved_path=manifest.repository_map["repo"].path,
                git_commit="3" * 40,
                deploy_key_label=f"rcp:{space_id}:{project_id}:repo",
                public_key_fingerprint=FINGERPRINT,
            ),
        ),
    )


def _write_remote_manifest(root: Path, *, project_id: str) -> Path:
    remote_root = f"/srv/rcp/projects/{project_id}/repositories/repo"
    root.mkdir(parents=True, mode=0o700)
    manifest = root / "manifest.toml"
    manifest.write_text(
        f'''name = "Unavailable remote project"

[[machines]]
alias = "remote"
host = "unreachable.example.test"
os_account = "rcp"

[[repositories]]
alias = "repo"
machine = "remote"
path = "{remote_root}"

[project]
truth_scope = ["repo"]

[state]
repository = "repo"

[agent]
default_run_truth_scope = ["repo"]

[sources]
claude_roots = ["~/.claude/projects"]
codex_roots = ["~/.codex/sessions"]

[execution]
run_on = "remote"

[paper.coach]
default_provider = "codex"
default_model = ""
default_reasoning = "medium"
''',
        encoding="utf-8",
    )
    return manifest


def _write_built_receipt(path: Path, receipt: BuiltCandidateReceipt) -> str:
    content = (
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path.write_bytes(content)
    path.chmod(0o600)
    return hashlib.sha256(content).hexdigest()


class _CapturedControl:
    def __init__(self, result: ServerControlBackupCaptureResult) -> None:
        self.result = result
        self.calls = 0

    def capture_backup_sqlite(self) -> ServerControlBackupCaptureResult:
        self.calls += 1
        return self.result


def test_candidate_rehearsal_replays_a_copy_without_touching_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    update_root = tmp_path / "update"
    release_root = tmp_path / "releases"
    central_root = tmp_path / "central"
    for path in (data_dir, update_root, release_root, central_root):
        path.mkdir(mode=0o700)

    store, bootstrap = AppStore.initialize_team_space(
        data_dir / "rcp.sqlite3",
        "Rehearsal Lab",
    )
    member, _token = store.enroll_team_member(bootstrap, "Alice")
    _invitation, invitation_code = store.create_team_invitation(member.user_id)
    remote_member, _remote_token = store.enroll_team_member(invitation_code, "Bob")
    authorized_by = AuthorizedHuman(
        space_id=store.space_id,
        user_id=member.user_id,
        display_name=member.display_name,
    )

    transfer_sentinels: dict[Path, bytes] = {}
    for label in ("partial", "complete"):
        transfer_machine = ProjectProvisioningMachineIntent(
            alias="transfer",
            location="ssh",
            host="transfer.example.test",
            os_account="rcp",
            central_root="/srv/rcp/projects",
        )
        request = store.create_project_provisioning_request(
            kind="incoming_transfer",
            authorized_by=authorized_by,
            machines=[transfer_machine],
            repositories=[
                ProjectProvisioningRepositoryIntent(
                    alias="repo",
                    repository=parse_github_repository_ref("git@github.com:openai/rcp.git"),
                    machine_alias=transfer_machine.alias,
                )
            ],
            provider_checks=[
                ProjectProvisioningProviderIntent(
                    profile="refresh",
                    provider="codex",
                    runtime_id="codex-exec",
                    model="",
                    reasoning="medium",
                    machine_alias=transfer_machine.alias,
                )
            ],
            source_project_id=str(uuid.uuid4()),
        )
        inbox = data_dir / "transfer-inbox" / request.request_id
        inbox.mkdir(parents=True)
        sentinel = inbox / f"{label}.sentinel"
        payload = f"live {label} transfer\n".encode()
        sentinel.write_bytes(payload)
        transfer_sentinels[sentinel] = payload

    project_id = str(uuid.uuid4())
    account = pwd.getpwuid(os.geteuid()).pw_name
    repository = central_root / project_id / "repositories" / "repo"
    checkout_commit = _initialize_checkout(repository)
    checkout_sentinel = repository / "README.md"
    checkout_sentinel_bytes = checkout_sentinel.read_bytes()
    provider_home = tmp_path / "provider-home"
    provider_home.mkdir(mode=0o700)
    provider_sentinel = provider_home / "authenticated-session"
    provider_sentinel.write_bytes(b"live provider state\n")
    manifest_path = _write_manifest(
        repository,
        account=account,
        provider_home=provider_home,
    )
    manifest = load_manifest(manifest_path)
    HistoryManager(manifest).initialize()
    history = HistoryManager(manifest, expected_space_id=store.space_id)
    identity = history.claim_project_identity("created", project_id=project_id)
    materialized = history.initialize().state
    assert identity.project_id == project_id
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=str(manifest.path),
            name=manifest.name,
            state_location=str(manifest.research_dir),
            state_remote=False,
            added_at=store.now(),
            revision=materialized.revision,
            reachable=True,
        )
    )
    store.seat_project_member(project_id, member.user_id)
    imported_payload = b'{"type":"assistant","text":"rehearsal source"}\n'
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

    remote_project_id = str(uuid.uuid4())
    remote_manifest_path = _write_remote_manifest(
        tmp_path / "remote-bootstrap" / remote_project_id,
        project_id=remote_project_id,
    )
    remote_manifest = load_manifest(remote_manifest_path)
    remote_error = "The configured SSH canonical state is currently unreachable."
    store.upsert_project(
        ProjectRecord(
            project_id=remote_project_id,
            home_space_id=store.space_id,
            locator=str(remote_manifest.path),
            name=remote_manifest.name,
            state_location=f"/srv/rcp/projects/{remote_project_id}/repositories/repo/.research",
            state_remote=True,
            added_at=store.now(),
            revision=7,
            reachable=False,
            error=remote_error,
        )
    )
    store.seat_project_member(remote_project_id, remote_member.user_id)

    run_stage = data_dir / "run-stage"
    run_stage.mkdir(mode=0o700)
    live_stage = run_stage / "live-task"
    live_stage.mkdir(mode=0o700)
    stage_sentinel = live_stage / "provider-output"
    stage_sentinel.write_bytes(b"live task output\n")
    operation_id = str(uuid.uuid4())
    request = RunRequest(
        provider="codex",
        run_truth_scope=["repo"],
        run_on="server",
        mode="work",
        chat_scope="project",
    )
    authority = resolve_dispatch_authority("refresh", request)
    assert authority is not None
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="refresh",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Queued",
            phase="queued",
            last_activity_at=now,
            stage_root=str(live_stage),
            authorized_by=authorized_by,
            dispatch_authority=authority,
        )
    )
    store.mark_agent_task_running(operation_id)

    metadata = ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=data_dir / "control.sock",
        running_commit=BASE_COMMIT,
        web_build_id=WEB_BUILD_ID,
    )
    publication = BackupCaptureCoordinator(store, data_dir, metadata).capture_sqlite()
    recovery = _recovery_descriptor(
        manifest_path,
        project_id=project_id,
        space_id=store.space_id,
        central_root=central_root,
        checkout_commit=checkout_commit,
        account=account,
    )
    remote_recovery = _remote_recovery_descriptor(
        remote_manifest_path,
        project_id=remote_project_id,
        space_id=store.space_id,
        central_root="/srv/rcp/projects",
    )
    inventories = tuple(
        sorted(
            (
                BackupSnapshotProjectInventory(
                    project_id=project_id,
                    home_space_id=store.space_id,
                    locator=str(manifest.path),
                    status="capturable",
                    recovery=recovery,
                    task_operation_ids=(operation_id,),
                ),
                BackupSnapshotProjectInventory(
                    project_id=remote_project_id,
                    home_space_id=store.space_id,
                    locator=str(remote_manifest.path),
                    status="capturable",
                    recovery=remote_recovery,
                ),
            ),
            key=lambda item: item.project_id,
        )
    )
    receipt = publication.receipt.model_copy(update={"projects": inventories, "status": "complete"})
    original_verify = project_files_module.verify_checkout_identities

    def verify_or_report_unreachable(
        descriptor: BackupCheckoutRecoveryDescriptor,
    ) -> None:
        remote_machine = next(
            (machine for machine in descriptor.machines if machine.location == "ssh"),
            None,
        )
        if remote_machine is not None:
            raise BackupCheckoutHostUnavailable(
                "remote host unavailable",
                machine_alias=remote_machine.alias,
            )
        original_verify(descriptor)

    monkeypatch.setattr(
        project_files_module,
        "verify_checkout_identities",
        verify_or_report_unreachable,
    )
    publication.receipt_path.chmod(0o600)
    publication.receipt_path.unlink()
    sqlite_receipt_sha256 = write_immutable_backup_receipt(
        publication.receipt_path,
        receipt,
    )
    control = _CapturedControl(
        ServerControlBackupCaptureResult(
            instance_id=metadata.instance_id,
            pid=metadata.pid,
            data_dir_id=metadata.data_dir_id,
            space_id=store.space_id,
            capture_id=receipt.capture_id,
            receipt_path=str(publication.receipt_path),
            receipt_sha256=sqlite_receipt_sha256,
            snapshot_sha256=receipt.sqlite_snapshot.sha256,
            status="complete",
            project_count=2,
            uncaptured_project_count=0,
        )
    )

    candidate_release = release_root / CANDIDATE_COMMIT
    candidate_release.mkdir(mode=0o700)
    built_path = update_root / f"built-candidate-{CANDIDATE_COMMIT}.json"
    built = BuiltCandidateReceipt(
        installation_id=str(uuid.uuid4()),
        source_origin="https://github.com/openai/rcp.git",
        base_current_commit=BASE_COMMIT,
        base_running_commit=BASE_COMMIT,
        base_instance_id=metadata.instance_id,
        base_process_pid=metadata.pid,
        candidate_commit=CANDIDATE_COMMIT,
        release_path=str(candidate_release),
        receipt_path=str(built_path),
        web_build_id=WEB_BUILD_ID,
        prepared_at=datetime.now(UTC),
    )
    built_sha256 = _write_built_receipt(built_path, built)
    live_release_marker = tmp_path / "current-release"
    live_release_marker.write_text(BASE_COMMIT + "\n", encoding="utf-8")
    candidate_calls: list[tuple[tuple[str, ...], Path, dict[str, str], float]] = []
    forbidden_candidate_accesses: list[str] = []
    original_publish = rehearsal_module._publish_private_json

    def publish_with_evidence_retained(path: Path, model) -> None:
        assert publication.receipt_path.parent.is_dir()
        assert len(list(update_root.glob("rehearsal-*"))) == 1
        original_publish(path, model)

    monkeypatch.setattr(
        rehearsal_module,
        "_publish_private_json",
        publish_with_evidence_retained,
    )

    def run_candidate(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        candidate_calls.append((argv, cwd, environment, timeout))
        if "--candidate-migrate" in argv:
            returncode = run_candidate_migration(Path(argv[-2]), Path(argv[-1]))
        else:
            overlay = RehearsalOverlay.model_validate_json(Path(argv[-2]).read_bytes())
            rehearsed_owner = ImportedProviderSourceStore(
                Path(overlay.data_dir),
                project_id,
            )
            assert rehearsed_owner.inventory() == imported_inventory
            rehearsed_file = (
                rehearsed_owner.root
                / imported_inventory.files[0].provider
                / imported_inventory.files[0].sha256
            )
            assert rehearsed_file.read_bytes() == imported_payload
            blocked_roots = (
                data_dir,
                repository,
                provider_home,
                Path(f"/srv/rcp/projects/{remote_project_id}"),
            )

            def blocked(path) -> bool:
                if isinstance(path, int):
                    return False
                candidate = Path(os.fsdecode(path))
                if not candidate.is_absolute():
                    candidate = (cwd / candidate).resolve()
                return any(
                    candidate == root or candidate.is_relative_to(root) for root in blocked_roots
                )

            def guard(original):
                def checked(path, *args, **kwargs):
                    if blocked(path):
                        forbidden_candidate_accesses.append(os.fsdecode(path))
                        raise AssertionError(f"candidate touched live path {path}")
                    return original(path, *args, **kwargs)

                return checked

            with monkeypatch.context() as access_guard:
                access_guard.setattr(os, "open", guard(os.open))
                access_guard.setattr(os, "stat", guard(os.stat))
                access_guard.setattr(os, "lstat", guard(os.lstat))
                access_guard.setattr(os, "listdir", guard(os.listdir))
                access_guard.setattr(os, "scandir", guard(os.scandir))
                access_guard.setattr(io, "open", guard(io.open))
                access_guard.setattr(builtins, "open", guard(builtins.open))
                returncode = run_candidate_child(Path(argv[-2]), Path(argv[-1]))
        return subprocess.CompletedProcess(argv, returncode, "", "")

    verified = CandidateRehearsalCoordinator(
        data_dir=data_dir,
        update_root=update_root,
        built_receipt=built,
        built_receipt_sha256=built_sha256,
        control=control,
        runner=run_candidate,
        candidate_python=Path(sys.executable),
    ).run()

    assert control.calls == 1
    assert forbidden_candidate_accesses == []
    assert len(candidate_calls) == 3
    assert ["--candidate-migrate" in call[0] for call in candidate_calls] == [
        True,
        True,
        False,
    ]
    assert all(call[1] == candidate_release for call in candidate_calls)
    assert all(
        call[2]
        == {
            "LANG": "C.UTF-8",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for call in candidate_calls
    )
    assert verified.candidate_commit == CANDIDATE_COMMIT
    assert verified.base_running_commit == BASE_COMMIT
    assert verified.space_id == store.space_id
    assert verified.startup_recovery.active_operation_ids == (operation_id,)
    project_results = {project.project_id: project for project in verified.projects}
    assert (
        project_results[project_id].status,
        project_results[project_id].revision,
    ) == ("verified", materialized.revision)
    assert (
        project_results[remote_project_id].status,
        project_results[remote_project_id].revision,
    ) == ("not_replay_verified", None)
    assert set(verified.reads) == {
        "/api/health",
        "/api/projects",
        f"/api/projects/{project_id}",
        f"/api/projects/{project_id}/tasks",
        f"/api/projects/{project_id}/watchers",
    }
    verified_path = verified_candidate_receipt_path(
        CANDIDATE_COMMIT,
        verified.capture_id,
        update_root,
    )
    assert stat.S_IMODE(verified_path.stat().st_mode) == 0o600
    assert not publication.receipt_path.parent.exists()
    assert list(update_root.glob("rehearsal-*")) == []

    live_task = store.agent_task(operation_id)
    assert live_task is not None and live_task.status == "running"
    assert stage_sentinel.read_bytes() == b"live task output\n"
    assert checkout_sentinel.read_bytes() == checkout_sentinel_bytes
    assert provider_sentinel.read_bytes() == b"live provider state\n"
    assert imported_owner.inventory() == imported_inventory
    assert all(path.read_bytes() == payload for path, payload in transfer_sentinels.items())
    assert live_release_marker.read_text(encoding="utf-8") == BASE_COMMIT + "\n"


def test_rehearsal_directories_must_be_private(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    update_root = tmp_path / "update"
    release = tmp_path / "releases" / CANDIDATE_COMMIT
    data_dir.mkdir(mode=0o755)
    update_root.mkdir(mode=0o700)
    release.mkdir(parents=True, mode=0o700)
    built_path = update_root / f"built-candidate-{CANDIDATE_COMMIT}.json"
    built = BuiltCandidateReceipt(
        installation_id=str(uuid.uuid4()),
        source_origin="https://github.com/openai/rcp.git",
        base_current_commit=BASE_COMMIT,
        base_running_commit=BASE_COMMIT,
        base_instance_id=str(uuid.uuid4()),
        base_process_pid=os.getpid(),
        candidate_commit=CANDIDATE_COMMIT,
        release_path=str(release),
        receipt_path=str(built_path),
        web_build_id=WEB_BUILD_ID,
        prepared_at=datetime.now(UTC),
    )
    digest = _write_built_receipt(built_path, built)

    with pytest.raises(RuntimeError, match="unsafe ownership or mode"):
        CandidateRehearsalCoordinator(
            data_dir=data_dir,
            update_root=update_root,
            built_receipt=built,
            built_receipt_sha256=digest,
        ).run()


def test_fenced_startup_only_plans_recovery_and_rejects_effect_entrypoints(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    store, bootstrap = AppStore.initialize_team_space(
        data_dir / "rcp.sqlite3",
        "Fenced Startup Lab",
    )
    member, _token = store.enroll_team_member(bootstrap, "Alice")
    fence = StartupEffectFence("candidate update rehearsal")
    app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, current: current.space_user(member.user_id),
        startup_effect_fence=fence,
    )

    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert app.state.startup_recovery_plan == {
            "active_operation_ids": (),
            "stopping_experiment_operation_ids": (),
            "report_episode_ids": (),
            "auto_research_recovery_operation_ids": (),
            "active_watcher_ids": (),
        }
        with pytest.raises(StartupEffectBlocked, match="blocked startup recovery"):
            app.state.services.background_tasks.recover_at_startup()

    assert fence.attempted_effects == ("startup recovery",)
    with pytest.raises(StartupEffectBlocked, match="cannot open"):
        fence.release()


def test_releasing_the_same_startup_fence_starts_the_deferred_runtime(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    store, bootstrap = AppStore.initialize_team_space(
        data_dir / "rcp.sqlite3",
        "Deferred Startup Lab",
    )
    member, _token = store.enroll_team_member(bootstrap, "Alice")
    fence = StartupEffectFence("candidate cutover verification")
    app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, current: current.space_user(member.user_id),
        startup_effect_fence=fence,
    )

    with TestClient(app):
        assert not app.state.startup_effect_runtime_started
        fence.release()
        assert app.state.startup_effect_runtime_event.wait(timeout=2)
        assert app.state.startup_effect_runtime_started
        assert app.state.startup_effect_release_error is None


def test_candidate_child_refuses_when_overlay_ownership_is_already_held(tmp_path: Path) -> None:
    operation_root = tmp_path / "operation"
    data_dir = operation_root / "overlay" / "data"
    data_dir.mkdir(parents=True, mode=0o700)
    store, bootstrap = AppStore.initialize_team_space(
        data_dir / "rcp.sqlite3",
        "Locked Rehearsal Lab",
    )
    store.enroll_team_member(bootstrap, "Alice")
    overlay = RehearsalOverlay(
        root=str(operation_root / "overlay"),
        data_dir=str(data_dir),
        database_path=str(data_dir / "rcp.sqlite3"),
        capture_id=str(uuid.uuid4()),
        sqlite_receipt_sha256="1" * 64,
        sqlite_snapshot_sha256="2" * 64,
        project_receipt_sha256="3" * 64,
        space_id=store.space_id,
        expected_startup_recovery=StartupRecoveryReadModel(
            active_operation_ids=(),
            stopping_experiment_operation_ids=(),
            report_episode_ids=(),
            auto_research_recovery_operation_ids=(),
            active_watcher_ids=(),
        ),
        projects=(),
        transfer_inbox_entries=(),
    )
    overlay_path = operation_root / "overlay.json"
    result_path = operation_root / "result.json"
    overlay_path.write_text(overlay.model_dump_json(), encoding="utf-8")
    overlay_path.chmod(0o600)

    with instance_lock(data_dir):
        assert run_candidate_child(overlay_path, result_path) == 1

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "Another RCP process" in result["diagnostic"]


def test_candidate_migration_and_reads_cross_the_real_subprocess_boundary(tmp_path: Path) -> None:
    operation_root = tmp_path / "operation"
    data_dir = operation_root / "overlay" / "data"
    data_dir.mkdir(parents=True, mode=0o700)
    store, bootstrap = AppStore.initialize_team_space(
        data_dir / "rcp.sqlite3",
        "Subprocess Rehearsal Lab",
    )
    store.enroll_team_member(bootstrap, "Alice")
    overlay = RehearsalOverlay(
        root=str(operation_root / "overlay"),
        data_dir=str(data_dir),
        database_path=str(data_dir / "rcp.sqlite3"),
        capture_id=str(uuid.uuid4()),
        sqlite_receipt_sha256="1" * 64,
        sqlite_snapshot_sha256="2" * 64,
        project_receipt_sha256="3" * 64,
        space_id=store.space_id,
        expected_startup_recovery=StartupRecoveryReadModel(
            active_operation_ids=(),
            stopping_experiment_operation_ids=(),
            report_episode_ids=(),
            auto_research_recovery_operation_ids=(),
            active_watcher_ids=(),
        ),
        projects=(),
        transfer_inbox_entries=(),
    )
    environment = {
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    migration_result = operation_root / "migration.json"
    migrated = subprocess.run(
        (
            sys.executable,
            "-m",
            "rcp.server_ops.rehearsal",
            "--candidate-migrate",
            str(data_dir / "rcp.sqlite3"),
            str(migration_result),
        ),
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert migrated.returncode == 0, migrated.stderr
    assert json.loads(migration_result.read_text(encoding="utf-8"))["status"] == "migrated"

    overlay_path = operation_root / "overlay.json"
    result_path = operation_root / "candidate-result.json"
    overlay_path.write_text(overlay.model_dump_json(), encoding="utf-8")
    overlay_path.chmod(0o600)
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "rcp.server_ops.rehearsal",
            "--candidate-child",
            str(overlay_path),
            str(result_path),
        ),
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "verified"
    assert result["reads"] == ["/api/health", "/api/projects"]


def test_overlay_refuses_a_new_unclassified_database_path_column(
    tmp_path: Path,
) -> None:
    capture_id = str(uuid.uuid4())
    space_id = str(uuid.uuid4())
    capture_root = tmp_path / f"backup-{capture_id}"
    capture_root.mkdir(mode=0o700)
    snapshot = capture_root / "rcp.sqlite3"
    AppStore(snapshot)
    snapshot_bytes = snapshot.read_bytes()
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    sqlite_receipt_sha256 = "e" * 64
    captured_at = datetime.now(UTC)
    sqlite_receipt = BackupSQLiteCaptureReceipt(
        capture_id=capture_id,
        captured_at=captured_at,
        rcp_source_commit=BASE_COMMIT,
        space_id=space_id,
        space_name="Future schema lab",
        snapshot_path=str(snapshot),
        database_schema_sha256="f" * 64,
        sqlite_snapshot=BackupFileEntry(
            archive_path="database/rcp.sqlite3",
            source_relative_path="rcp.sqlite3",
            group="sqlite_snapshot",
            sha256=snapshot_sha256,
            size_bytes=len(snapshot_bytes),
        ),
        app_data_plan=BackupAppDataCapturePlan(
            data_dir=str(tmp_path / "live-data"),
            database_path=str(tmp_path / "live-data" / "rcp.sqlite3"),
            database_unavailable_reason=None,
            excluded_entries=(),
            deferred_entries=(),
            unclassified_entries=(),
        ),
        projects=(),
        status="complete",
    )
    project_receipt = BackupProjectFileCaptureReceipt(
        capture_id=capture_id,
        captured_at=captured_at,
        completed_at=captured_at,
        rcp_source_commit=BASE_COMMIT,
        space_id=space_id,
        sqlite_receipt_sha256=sqlite_receipt_sha256,
        sqlite_snapshot_sha256=snapshot_sha256,
        sqlite_capture_status="complete",
        projects=(),
        status="complete",
    )
    operation_root = tmp_path / "operation"
    operation_root.mkdir(mode=0o700)

    def candidate_adds_unknown_path(database_path: Path) -> None:
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE future_state (future_root TEXT)")

    with pytest.raises(CandidateRehearsalRefused, match="unclassified path columns"):
        build_rehearsal_overlay(
            operation_root,
            sqlite_receipt=sqlite_receipt,
            sqlite_receipt_sha256=sqlite_receipt_sha256,
            project_receipt=project_receipt,
            project_receipt_sha256="1" * 64,
            capture_root=capture_root,
            candidate_migrator=candidate_adds_unknown_path,
        )
