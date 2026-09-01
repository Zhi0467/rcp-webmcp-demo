from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

import rcp.server_ops.restore as restore_code
from rcp.config import AGENT_EXECUTION_PROFILES, permissions_for
from rcp.core.transition_models import GraphHeadRef
from rcp.projects import rebind_restored_project_registration
from rcp.server_ops.backup_capture import _database_schema_sha256
from rcp.server_ops.backup_models import (
    BackupArchiveManifest,
    BackupCheckoutRecoveryDescriptor,
    BackupFileEntry,
    BackupManifestAgentProfile,
    BackupManifestConfiguration,
    BackupManifestMachine,
    BackupManifestRepository,
    BackupManifestSources,
    BackupProjectCapture,
    BackupRecoveryMachine,
    BackupRecoveryRepository,
)
from rcp.server_ops.cli import CallerIdentity
from rcp.server_ops.git_credentials import DeployKeyMaterial, GitWriteProbe
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.layout import ServerLayout
from rcp.server_ops.models import (
    ExternalAction,
    ExternalServiceTarget,
    NonsecretField,
    ServerCommandExecution,
    ServerPlanEvent,
    ServerStep,
    ServerStepEvent,
)
from rcp.server_ops.project_checkout import ProjectCheckoutResult, RetainedResearchState
from rcp.server_ops.restore import (
    LinuxRestoreMachine,
    RestoreConfirmation,
    RestoreOperationJournal,
    read_restore_journal,
    write_restore_journal,
)
from rcp.skill_registry import SkillDefaults
from rcp.storage import AppStore, ProjectProvisioningMachineIntent, ProjectRecord

CAPTURED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
ARCHIVED_COMMIT = "a" * 40
CURRENT_COMMIT = "c" * 40
ARCHIVED_FINGERPRINT = "SHA256:" + ("A" * 43)
FRESH_FINGERPRINT = "SHA256:" + ("B" * 43)
REPOSITORY = GitHubRepositoryRef(identity="openai/rcp")


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


def _configuration(
    *,
    location: Literal["local", "ssh"],
    repository_path: Path,
) -> BackupManifestConfiguration:
    host = "" if location == "local" else "gpu.example"
    account = "rcp" if location == "local" else "alice"
    return BackupManifestConfiguration(
        name="Restored project",
        machines=(
            BackupManifestMachine(
                alias="server",
                host=host,
                os_account=account,
                provider_paths={},
            ),
        ),
        repositories=(
            BackupManifestRepository(
                alias="paper",
                machine="server",
                path=str(repository_path),
            ),
        ),
        project_truth_scope=("paper",),
        state_repository="paper",
        default_run_truth_scope=("paper",),
        default_auto_research_invocation_ceiling=10,
        skill_defaults=SkillDefaults(),
        agent_profiles=tuple(
            BackupManifestAgentProfile(
                profile=profile,
                provider="codex",
                runtime="exec",
                model="gpt-test",
                reasoning="medium",
                run_on="server",
                permissions=permissions_for(profile),
            )
            for profile in AGENT_EXECUTION_PROFILES
        ),
        sources=BackupManifestSources(
            claude_roots=("~/.claude/projects",),
            codex_roots=("~/.codex/sessions",),
            remote_claude_roots=("~/.claude/projects",),
            remote_codex_roots=("~/.codex/sessions",),
        ),
    )


def _capture(
    *,
    store: AppStore,
    layout: ServerLayout,
    location: Literal["local", "ssh"],
) -> BackupProjectCapture:
    project_id = str(uuid.uuid4())
    central_root = layout.projects_root if location == "local" else layout.server_root / "remote"
    repository_path = central_root / project_id / "repositories" / "paper"
    configuration = _configuration(location=location, repository_path=repository_path)
    recovery = BackupCheckoutRecoveryDescriptor(
        request_id=str(uuid.uuid4()),
        project_id=project_id,
        home_space_id=store.space_id,
        completed_at=CAPTURED_AT,
        final_review_digest="d" * 64,
        configuration=configuration,
        configuration_sha256=configuration.sha256,
        machines=(
            BackupRecoveryMachine(
                alias="server",
                location=location,
                host="" if location == "local" else "gpu.example",
                os_account="rcp" if location == "local" else "alice",
                resolved_central_root=str(central_root),
            ),
        ),
        repositories=(
            BackupRecoveryRepository(
                alias="paper",
                repository=REPOSITORY,
                machine_alias="server",
                resolved_path=str(repository_path),
                git_commit=ARCHIVED_COMMIT,
                deploy_key_label=f"rcp:{store.space_id}:{project_id}:paper",
                public_key_fingerprint=ARCHIVED_FINGERPRINT,
            ),
        ),
    )
    manifest_bytes = b"archived manifest"
    entry = BackupFileEntry(
        archive_path=f"projects/{project_id}/canonical/manifest.toml",
        source_relative_path=".research/manifest.toml",
        group="canonical",
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        size_bytes=len(manifest_bytes),
    )
    old_locator = layout.data_dir / "old" / project_id / "manifest.toml"
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=str(old_locator),
            name=configuration.name,
            state_location=f"old:{project_id}",
            state_remote=location == "ssh",
            added_at=CAPTURED_AT.isoformat(),
            reachable=True,
        )
    )
    return BackupProjectCapture(
        project_id=project_id,
        home_space_id=store.space_id,
        locator=str(old_locator),
        status="captured",
        main_head=GraphHeadRef(revision=0),
        files=(entry,),
        recovery=recovery,
        total_bytes=len(manifest_bytes),
    )


def _journal(
    tmp_path: Path,
    *,
    location: Literal["local", "ssh"],
) -> tuple[ServerLayout, RestoreOperationJournal, BackupProjectCapture]:
    layout = _layout(tmp_path)
    for path in (layout.data_dir, layout.restore_operations_root, layout.projects_root):
        path.mkdir(parents=True, mode=0o700)
        path.chmod(0o700)
    store, _bootstrap = AppStore.initialize_team_space(
        layout.data_dir / "rcp.sqlite3",
        "Restore lab",
    )
    capture = _capture(store=store, layout=layout, location=location)
    sqlite_path = layout.data_dir / "rcp.sqlite3"
    sqlite_sha256 = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    sqlite_entry = BackupFileEntry(
        archive_path="database/rcp.sqlite3",
        source_relative_path="rcp.sqlite3",
        group="sqlite_snapshot",
        sha256=sqlite_sha256,
        size_bytes=sqlite_path.stat().st_size,
    )
    manifest = BackupArchiveManifest(
        space_id=store.space_id,
        space_name=store.space_name,
        rcp_source_commit=ARCHIVED_COMMIT,
        database_schema_sha256=_database_schema_sha256(store),
        captured_at=CAPTURED_AT,
        sqlite_snapshot=sqlite_entry,
        encryption_recipient_fingerprint="e" * 64,
        installation_id=str(uuid.uuid4()),
        excluded_app_data_entries=(),
        uncaptured_app_data_entries=(),
        projects=(capture,),
        status="complete",
        total_bytes=sqlite_entry.size_bytes + capture.total_bytes,
    )
    archive = tmp_path / "archive.tar.age"
    archive.write_bytes(b"archive")
    candidate_root = layout.restore_operations_root / f"candidate-{'f' * 64}"
    candidate = candidate_root / "restored" / "rcp.sqlite3"
    candidate.parent.mkdir(parents=True, mode=0o700)
    shutil.copyfile(sqlite_path, candidate)
    journal = RestoreOperationJournal(
        operation_id=str(uuid.uuid4()),
        archive_path=str(archive),
        archive_sha256="f" * 64,
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
            confirmed_at=CAPTURED_AT,
        ),
        phase="sqlite_restored",
        detached_at=CAPTURED_AT,
        restored_sqlite_sha256=sqlite_sha256,
        updated_at=CAPTURED_AT,
    )
    write_restore_journal(journal, layout, uid=os.getuid(), gid=os.getgid())
    return layout, journal, capture


class _Credentials:
    def __init__(self, *, first_probe: str = "ready") -> None:
        self.first_probe = first_probe
        self.events: list[str] = []

    def preflight_recovery_key(self, *_args, **_kwargs) -> None:
        self.events.append("preflight")

    def prepare_recovery_key(
        self,
        machine: ProjectProvisioningMachineIntent,
        repository: GitHubRepositoryRef,
        *,
        space_id: str,
        project_id: str,
        repository_alias: str,
    ) -> DeployKeyMaterial:
        self.events.append("prepare")
        return DeployKeyMaterial(
            space_id=space_id,
            project_id=project_id,
            repository_alias=repository_alias,
            repository=repository,
            machine_alias=machine.alias,
            location=machine.location,
            host=machine.host,
            os_account=machine.os_account,
            central_root=str(machine.central_root),
            account_home="/home/rcp" if machine.location == "local" else "/home/alice",
            credentials_root="/credentials",
            private_key_path=f"/credentials/{project_id}/{repository_alias}",
            label=f"rcp:{space_id}:{project_id}:{repository_alias}",
            public_key="ssh-ed25519 AAAA fresh-restore-key",
            public_key_fingerprint=FRESH_FINGERPRINT,
            created=len(self.events) == 2,
        )

    def probe_write(self, *_args, **_kwargs) -> GitWriteProbe:
        self.events.append("probe")
        status = self.first_probe
        self.first_probe = "ready"
        return GitWriteProbe(
            status=status,  # type: ignore[arg-type]
            commit=CURRENT_COMMIT if status == "ready" else None,
            temporary_ref=None,
            diagnostic="Grant the fresh key." if status != "ready" else "ready",
        )

    def github_trust_argv(self, *_args) -> tuple[str, ...]:
        return ("ssh", "git@github.com")


class _Checkouts:
    def __init__(self, capture: BackupProjectCapture) -> None:
        assert capture.recovery is not None
        self.recovery = capture.recovery
        self.calls: list[dict[str, object]] = []

    def prepare_recovery(self, machine, _material, **kwargs) -> ProjectCheckoutResult:
        self.calls.append(kwargs)
        repository = self.recovery.repositories[0]
        return ProjectCheckoutResult(
            machine_alias=machine.alias,
            repository_alias=repository.alias,
            central_root=self.recovery.machines[0].resolved_central_root,
            repository_path=repository.resolved_path,
            checkout_disposition="request_created",
            commit=str(kwargs["expected_head"]),
            retained_research=RetainedResearchState(False, False, None, None),
        )


class _RestoreMachine(LinuxRestoreMachine):
    @contextmanager
    def admission(self) -> Iterator[None]:
        yield


def _machine(
    layout: ServerLayout,
    credentials: _Credentials,
    checkouts: _Checkouts,
) -> _RestoreMachine:
    return _RestoreMachine(
        layout,
        config_loader=lambda _path: SimpleNamespace(
            paths=SimpleNamespace(model_dump=layout.recorded_paths)
        ),
        service_control=SimpleNamespace(fence_stopped_disabled=lambda: None),
        service_identity=(os.getuid(), os.getgid()),
        root_identity=(os.getuid(), os.getgid()),
        credential_manager=credentials,  # type: ignore[arg-type]
        checkout_manager=checkouts,  # type: ignore[arg-type]
        clock=lambda: CAPTURED_AT,
    )


@pytest.mark.parametrize("location", ["local", "ssh"])
def test_restore_recovers_and_rebinds_local_or_ssh_checkout(
    tmp_path: Path,
    location: Literal["local", "ssh"],
) -> None:
    layout, journal, capture = _journal(tmp_path, location=location)
    credentials = _Credentials()
    checkouts = _Checkouts(capture)
    machine = _machine(layout, credentials, checkouts)

    outcome = machine.recover_checkouts(
        journal,
        resume_argv=(
            "sudo",
            "/usr/local/bin/rcp",
            "server",
            "restore",
            "/backup.age",
            "--identity-file",
            "/recovery.txt",
            "--confirm-data-dir",
            str(layout.data_dir),
        ),
        step_number=6,
    )

    assert outcome.operator_action is None
    assert outcome.journal.phase == "checkouts_ready"
    assert credentials.events == ["preflight", "prepare", "probe"]
    assert checkouts.calls[0]["retained_provisioning_commit"] == ARCHIVED_COMMIT
    assert checkouts.calls[0]["expected_head"] == CURRENT_COMMIT
    assert checkouts.calls[0]["archived_research"]

    completed = machine.rebind_checkouts(outcome.journal)

    assert completed.phase == "checkouts_reconstructed"
    stored = AppStore(layout.data_dir / "rcp.sqlite3").project(capture.project_id)
    assert stored is not None
    assert stored.reachable is False
    assert stored.error == "Replacement restore publication is pending."
    if location == "local":
        assert stored.locator.endswith("/.research/manifest.toml")
        assert not Path(stored.locator).exists()
    else:
        assert Path(stored.locator).is_file()
        assert stored.state_location.startswith("gpu.example:")


def test_restore_pauses_for_fresh_github_grant_and_resumes_same_key(tmp_path: Path) -> None:
    layout, journal, capture = _journal(tmp_path, location="ssh")
    credentials = _Credentials(first_probe="github_grant_needed")
    checkouts = _Checkouts(capture)
    machine = _machine(layout, credentials, checkouts)
    resume = (
        "sudo",
        "/usr/local/bin/rcp",
        "server",
        "restore",
        "/backup.age",
        "--identity-file",
        "/recovery.txt",
        "--confirm-data-dir",
        str(layout.data_dir),
    )

    paused = machine.recover_checkouts(
        journal,
        resume_argv=resume,
        step_number=6,
    )

    assert paused.operator_action is not None
    assert paused.operator_action.state == "operator_action_needed"
    assert "Allow write access" in paused.operator_action.actions[0].instruction
    durable = read_restore_journal(layout, expected_uid=os.getuid())
    assert durable.repository_recoveries[0].state == "key_ready"
    assert durable.repository_recoveries[0].public_key_fingerprint == FRESH_FINGERPRINT
    assert checkouts.calls == []

    resumed = machine.recover_checkouts(
        durable,
        resume_argv=resume,
        step_number=6,
    )

    assert resumed.operator_action is None
    assert resumed.journal.phase == "checkouts_ready"
    assert credentials.events == ["preflight", "prepare", "probe", "prepare", "probe"]


def test_restore_binds_checkout_operator_action_to_the_published_plan(tmp_path: Path) -> None:
    layout, _journal_value, _capture = _journal(tmp_path, location="ssh")
    planned = restore_code._restore_plan(  # noqa: SLF001 - event-contract regression
        CallerIdentity(uid=0, username="root", host="lab.example"),
        layout.data_dir,
    )[5].model_copy(update={"number": 1})
    action = ServerStep(
        number=planned.number,
        title="Grant the fresh restore deploy key",
        purpose="Grant one repository-scoped GitHub identity.",
        performed_by="human",
        target=ExternalServiceTarget(
            service="github.com",
            resource="OpenAI/RCP-paper",
            destination_url="https://github.com/OpenAI/RCP-paper/settings/keys",
            required_authority_role="repository administrator",
        ),
        phase="restore_github_grant",
        state="operator_action_needed",
        expected_success="The replacement key can read and write the captured commit.",
        message="Add the fresh deploy key, then resume restore.",
        actions=(ExternalAction(instruction="Add the displayed key with write access."),),
        fields=(NonsecretField(name="deploy_key_label", value="rcp:test:key"),),
        resume_argv=(
            "sudo",
            "/usr/local/bin/rcp",
            "server",
            "restore",
            "/backup.age",
            "--identity-file",
            "/recovery.txt",
            "--confirm-data-dir",
            str(layout.data_dir),
        ),
    )

    paused = restore_code._bind_restore_operator_action_to_plan(  # noqa: SLF001
        planned,
        action,
    )

    assert paused.number == planned.number
    assert paused.title == planned.title
    assert paused.target == planned.target
    assert paused.phase == planned.phase
    assert paused.performed_by == "human"
    assert paused.actions == action.actions
    assert paused.resume_argv == action.resume_argv
    ServerCommandExecution(
        events=(
            ServerPlanEvent(
                command="server restore",
                timestamp=CAPTURED_AT,
                steps=(planned,),
            ),
            ServerStepEvent(
                command="server restore",
                timestamp=CAPTURED_AT,
                step=planned.model_copy(
                    update={"state": "running", "message": "Recovering checkouts."}
                ),
            ),
            ServerStepEvent(
                command="server restore",
                timestamp=CAPTURED_AT,
                step=paused,
            ),
        ),
        exit_code=3,
    )


def test_rebind_helper_is_idempotent_but_refuses_conflicting_repository_path(
    tmp_path: Path,
) -> None:
    layout, _journal_value, capture = _journal(tmp_path, location="ssh")
    assert capture.recovery is not None
    repository = capture.recovery.repositories[0]
    store = AppStore(layout.data_dir / "rcp.sqlite3")

    first = rebind_restored_project_registration(
        store,
        capture,
        repository_paths={repository.alias: repository.resolved_path},
        data_dir=layout.data_dir,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    second = rebind_restored_project_registration(
        store,
        capture,
        repository_paths={repository.alias: repository.resolved_path},
        data_dir=layout.data_dir,
        uid=os.getuid(),
        gid=os.getgid(),
    )

    assert first == second
    with pytest.raises(RuntimeError, match="differs from its reviewed recovery path"):
        rebind_restored_project_registration(
            store,
            capture,
            repository_paths={repository.alias: str(tmp_path / "wrong")},
            data_dir=layout.data_dir,
            uid=os.getuid(),
            gid=os.getgid(),
        )
