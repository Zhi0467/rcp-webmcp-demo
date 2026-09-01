from __future__ import annotations

import json
import os
import shutil
import tempfile
from io import StringIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rcp.setup as setup_code
import rcp.storage.models as storage_models
from rcp.__main__ import build_parser
from rcp.agents.launcher import ProviderExecutionAccount, ProviderReadiness
from rcp.api import create_app
from rcp.config import AGENT_EXECUTION_PROFILES, load_manifest, permissions_for
from rcp.core.models import AuthorizedHuman
from rcp.history import HistoryManager
from rcp.providers import ModelChoice, configured_runtime_id
from rcp.server_ops.cli import (
    SERVER_CLI_EXIT_OPERATOR_ACTION,
    CallerIdentity,
    run_server_command,
)
from rcp.server_ops.control import ServerControlClient
from rcp.server_ops.git_credentials import (
    DeployKeyMaterial,
    GitCredentialRefused,
    GitWriteProbe,
)
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.project_checkout import (
    ProjectCheckoutRefused,
    ProjectCheckoutResult,
    RetainedResearchState,
)
from rcp.server_ops.project_provision import (
    ProjectProvisionCoordinator,
    ProjectProvisionRefused,
    prepare_project_provision_command,
)
from rcp.server_ops.provider_readiness import ProviderReadinessCoordinator
from rcp.server_runtime import ServerMetadata, published_server_metadata
from rcp.storage import (
    AppStore,
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
)


class _Launcher:
    def __init__(self, *, authenticated: bool = True) -> None:
        self.authenticated = authenticated

    def execution_account(self, *, host: str = "") -> ProviderExecutionAccount:
        return ProviderExecutionAccount(
            host=host,
            reachable=True,
            os_account="alice" if host else "rcp",
        )

    def readiness(
        self,
        provider: str,
        *,
        host: str = "",
        binary: str | None = None,
        refresh: bool = False,
    ) -> ProviderReadiness:
        assert provider == "codex"
        assert host in {"", "gpu.example"}
        assert binary in {None, "/usr/local/bin/codex"}
        assert refresh is True
        return ProviderReadiness(
            provider="codex",
            installed=True,
            authenticated=self.authenticated,
            version="codex-cli 1.2.3",
            binary_path="/usr/local/bin/codex",
            path_state="resolved",
            models=[ModelChoice(id="gpt-test", label="GPT Test", reasoning=["medium"])],
            reason=(None if self.authenticated else "Codex is not authenticated."),
        )


class _Credentials:
    def __init__(self, *, probe_status: str = "ready") -> None:
        self.probe_status = probe_status
        self.fingerprint_character = "A"
        self.prepare_calls = 0
        self.probe_calls = 0

    def prepare_key(
        self,
        machine,
        repository,
        *,
        space_id: str,
        project_id: str,
        repository_alias: str,
    ) -> DeployKeyMaterial:
        self.prepare_calls += 1
        return DeployKeyMaterial(
            space_id=space_id,
            project_id=project_id,
            repository_alias=repository_alias,
            repository=repository,
            machine_alias=machine.alias,
            location=machine.location,
            host=machine.host,
            os_account=machine.os_account,
            central_root=str(DEFAULT_SERVER_LAYOUT.projects_root),
            account_home=str(DEFAULT_SERVER_LAYOUT.service_home),
            credentials_root=str(DEFAULT_SERVER_LAYOUT.credentials_root),
            private_key_path=str(
                DEFAULT_SERVER_LAYOUT.credentials_root / project_id / f"{repository_alias}.key"
            ),
            label=f"rcp:{space_id}:{project_id}:{repository_alias}",
            public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey rcp-test",
            public_key_fingerprint="SHA256:" + (self.fingerprint_character * 43),
            created=self.prepare_calls == 1,
        )

    def probe_write(self, _machine, _material, *, request_id: str) -> GitWriteProbe:
        self.probe_calls += 1
        assert request_id
        if self.probe_status == "ready":
            return GitWriteProbe(
                status="ready",
                commit="a" * 40,
                temporary_ref=None,
                diagnostic=(
                    "The request-scoped Git write probe passed and its temporary ref is gone."
                ),
            )
        return GitWriteProbe(
            status="github_grant_needed",
            commit="a" * 40,
            temporary_ref=None,
            diagnostic="GitHub has not granted this deploy key write access.",
        )

    def github_trust_argv(self, machine, material) -> tuple[str, ...]:
        return (
            "sudo",
            "-n",
            "-u",
            machine.os_account,
            "-H",
            "ssh",
            "-i",
            material.private_key_path,
            "git@github.com",
        )


class _Checkouts:
    def __init__(self) -> None:
        self.calls = 0

    def prepare(
        self,
        machine,
        material,
        *,
        request_kind: str,
        project_id: str,
        repository_alias: str,
        state_repository: bool,
        expected_commit: str | None,
    ) -> ProjectCheckoutResult:
        self.calls += 1
        assert request_kind == "create_team_project"
        assert isinstance(state_repository, bool)
        assert expected_commit == "a" * 40
        root = material.central_root
        return ProjectCheckoutResult(
            machine_alias=machine.alias,
            repository_alias=repository_alias,
            central_root=root,
            repository_path=(f"{root}/{project_id}/repositories/{repository_alias}"),
            checkout_disposition="request_created",
            commit="a" * 40,
            retained_research=RetainedResearchState(False, False, None, None),
        )


class _RefusingCredentials(_Credentials):
    def prepare_key(self, *_args, **_kwargs):
        self.prepare_calls += 1
        raise GitCredentialRefused("The exact credential path has unsafe writable ancestry.")


class _RetainedCheckout(_Checkouts):
    def prepare(self, machine, material, **_kwargs):
        self.calls += 1
        raise ProjectCheckoutRefused(
            "retained_research",
            "The state repository already contains retained RCP research.",
            central_root=material.central_root,
            repository_path=(
                f"{material.central_root}/{material.project_id}/repositories/"
                f"{material.repository_alias}"
            ),
            checkout_disposition="request_created",
            retained_research=RetainedResearchState(
                True,
                True,
                "123e4567-e89b-42d3-a456-426614174000",
                "123e4567-e89b-42d3-a456-426614174001",
            ),
        )


def _store_and_request(
    tmp_path: Path,
    *,
    repository_aliases: tuple[str, ...] = ("paper",),
    configured: bool = True,
    provider_only_machine: bool = False,
):
    store, _bootstrap = AppStore.initialize_team_space(
        tmp_path / "data" / "rcp.sqlite3",
        "Provisioning lab",
    )
    member = store.preprovision_team_member("Alice")
    authorizer = AuthorizedHuman(
        space_id=store.space_id,
        user_id=member.user_id,
        display_name="Alice",
    )
    machine = ProjectProvisioningMachineIntent(
        alias="server",
        location="local",
        os_account="rcp",
        central_root=str(DEFAULT_SERVER_LAYOUT.projects_root),
    )
    project_configuration = (
        {
            "name": "Shared paper",
            "state_repository": repository_aliases[0],
            "project_truth_scope": list(repository_aliases),
            "default_run_truth_scope": list(repository_aliases),
        }
        if configured
        else {}
    )
    machines = [machine]
    if provider_only_machine:
        machines.append(
            ProjectProvisioningMachineIntent(
                alias="worker",
                location="ssh",
                host="gpu.example",
                os_account="alice",
            )
        )
    request = store.create_project_provisioning_request(
        kind="create_team_project",
        authorized_by=authorizer,
        **project_configuration,
        machines=machines,
        repositories=[
            ProjectProvisioningRepositoryIntent(
                alias=alias,
                repository=parse_github_repository_ref(f"git@github.com:OpenAI/RCP-{index}.git"),
                machine_alias="server",
            )
            for index, alias in enumerate(repository_aliases, start=1)
        ],
        provider_checks=[
            ProjectProvisioningProviderIntent(
                profile="seed",
                provider="codex",
                runtime_id="codex:exec",
                model="gpt-test",
                reasoning="medium",
                machine_alias="worker" if provider_only_machine else "server",
            )
        ],
    )
    return store, request


def _coordinator(
    store: AppStore,
    tmp_path: Path,
    *,
    credentials: _Credentials | None = None,
    checkouts: _Checkouts | None = None,
    launcher: _Launcher | None = None,
) -> tuple[ProjectProvisionCoordinator, _Credentials, _Checkouts]:
    metadata = ServerMetadata.create(
        store.path.parent,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=tmp_path / "control.sock",
    )
    provider = ProviderReadinessCoordinator(
        store,
        launcher or _Launcher(),  # type: ignore[arg-type]
        metadata,
        local_host="lab.example",
    )
    credential_owner = credentials or _Credentials()
    checkout_owner = checkouts or _Checkouts()
    return (
        ProjectProvisionCoordinator(
            store,
            metadata,
            provider,
            credential_manager=credential_owner,  # type: ignore[arg-type]
            checkout_manager=checkout_owner,  # type: ignore[arg-type]
            local_host="lab.example",
        ),
        credential_owner,
        checkout_owner,
    )


def _advance_all(coordinator: ProjectProvisionCoordinator, request_id: str) -> None:
    plan = coordinator.plan(request_id)
    boundary = plan.boundary_sha256
    for target in plan.targets:
        result = coordinator.advance(
            request_id,
            boundary_sha256=boundary,
            target_id=target.target_id,
        )
        assert result.step.state == "succeeded"
        boundary = result.next_boundary_sha256
    assert result.request_status == "ready_for_review"


def test_complete_plan_prepares_every_owner_and_stops_before_project_creation(
    tmp_path: Path,
) -> None:
    store, request = _store_and_request(tmp_path)
    coordinator, credentials, checkouts = _coordinator(store, tmp_path)

    preview = coordinator.plan(request.request_id)

    assert [target.step.phase for target in preview.targets] == [
        "provisioning_start",
        "repository_key",
        "repository_write",
        "repository_checkout",
        "provider_readiness",
        "provisioning_review",
    ]
    assert credentials.prepare_calls == 0
    assert checkouts.calls == 0

    _advance_all(coordinator, request.request_id)

    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None
    assert stored.status == "ready_for_review"
    assert stored.final_review_digest is not None
    assert stored.repositories[0].git_check.status == "ready"
    assert stored.repositories[0].checkout_disposition == "request_created"
    assert stored.provider_checks[0].status == "ready"
    assert store.project(request.proposed_project_id) is None
    assert [
        receipt.phase for receipt in store.project_provisioning_step_receipts(request.request_id)
    ] == [
        "provisioning_start",
        "repository_key",
        "repository_write",
        "repository_checkout",
        "provider_readiness",
        "provisioning_review",
    ]


def test_cli_returns_zero_only_after_ready_for_review_readback(tmp_path: Path) -> None:
    store, request = _store_and_request(tmp_path)
    coordinator, _, _ = _coordinator(store, tmp_path)
    output = StringIO()
    args = build_parser().parse_args(
        (
            "server",
            "project",
            "provision",
            request.request_id,
            "--machine-readable",
        )
    )

    exit_code = run_server_command(
        args,
        identity=CallerIdentity(uid=501, username="rcp", host="lab.example"),
        handler=lambda command, identity: prepare_project_provision_command(
            command,
            identity,
            control=coordinator,
        ),
        stream=output,
    )

    assert exit_code == 0
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert events[-1]["step"]["phase"] == "provisioning_review"
    assert events[-1]["step"]["state"] == "succeeded"
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None and stored.status == "ready_for_review"
    assert store.project(request.proposed_project_id) is None


def test_cli_refuses_zero_exit_when_final_ready_readback_is_missing(tmp_path: Path) -> None:
    store, request = _store_and_request(tmp_path)
    coordinator, _, _ = _coordinator(store, tmp_path)
    output = StringIO()
    args = build_parser().parse_args(
        ("server", "project", "provision", request.request_id, "--machine-readable")
    )

    class _MissingReadyReadback:
        def project_provision_plan(self, *, request_id: str):
            return coordinator.project_provision_plan(request_id=request_id)

        def advance_project_provision(
            self,
            *,
            request_id: str,
            boundary_sha256: str,
            target_id: str,
        ):
            result = coordinator.advance_project_provision(
                request_id=request_id,
                boundary_sha256=boundary_sha256,
                target_id=target_id,
            )
            if result.request_status == "ready_for_review":
                return result.model_copy(update={"request_status": "setup_in_progress"})
            return result

    exit_code = run_server_command(
        args,
        identity=CallerIdentity(uid=501, username="rcp", host="lab.example"),
        handler=lambda command, identity: prepare_project_provision_command(
            command,
            identity,
            control=_MissingReadyReadback(),
        ),
        stream=output,
    )

    assert exit_code != 0
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert events[-1]["step"]["phase"] == "provisioning_review"
    assert events[-1]["step"]["state"] == "failed"
    assert "ready-for-review readback" in events[-1]["step"]["message"]


def test_multiple_repositories_commit_each_checkout_as_its_own_resume_boundary(
    tmp_path: Path,
) -> None:
    store, request = _store_and_request(
        tmp_path,
        repository_aliases=("paper", "code"),
    )
    coordinator, _, _ = _coordinator(store, tmp_path)
    plan = coordinator.plan(request.request_id)
    boundary = plan.boundary_sha256

    for target in plan.targets[:4]:
        result = coordinator.advance(
            request.request_id,
            boundary_sha256=boundary,
            target_id=target.target_id,
        )
        assert result.step.state == "succeeded"
        boundary = result.next_boundary_sha256

    partial = store.project_provisioning_request(request.request_id)
    assert partial is not None
    assert partial.machines[0].resolved_central_root == str(DEFAULT_SERVER_LAYOUT.projects_root)
    assert partial.repositories[0].checkout_disposition == "request_created"
    assert partial.repositories[1].resolved_path is None

    _advance_all(coordinator, request.request_id)
    ready = store.project_provisioning_request(request.request_id)
    assert ready is not None and ready.status == "ready_for_review"
    assert all(repository.resolved_path is not None for repository in ready.repositories)


def test_provider_only_machine_does_not_invent_a_checkout_root(tmp_path: Path) -> None:
    store, request = _store_and_request(tmp_path, provider_only_machine=True)
    coordinator, _, _ = _coordinator(store, tmp_path)

    _advance_all(coordinator, request.request_id)

    ready = store.project_provisioning_request(request.request_id)
    assert ready is not None and ready.status == "ready_for_review"
    machines = {machine.alias: machine for machine in ready.machines}
    assert machines["server"].resolved_central_root == str(DEFAULT_SERVER_LAYOUT.projects_root)
    assert machines["worker"].resolved_central_root is None


def test_legacy_request_without_project_configuration_pauses_before_machine_work(
    tmp_path: Path,
) -> None:
    store, request = _store_and_request(tmp_path, configured=False)
    coordinator, credentials, checkouts = _coordinator(store, tmp_path)
    plan = coordinator.plan(request.request_id)

    assert len(plan.targets) == 1
    result = coordinator.advance(
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    assert result.step.state == "operator_action_needed"
    assert "legacy request" in result.step.message
    assert result.step.resume_argv[-1] == request.request_id
    assert credentials.prepare_calls == 0
    assert checkouts.calls == 0
    paused = store.project_provisioning_request(request.request_id)
    assert paused is not None and paused.status == "operator_action_needed"
    assert store.project(request.proposed_project_id) is None


def test_installed_app_control_dispatches_the_same_durable_project_workflow(
    tmp_path: Path,
) -> None:
    store, request = _store_and_request(tmp_path, configured=False)
    control_root = Path(tempfile.mkdtemp(prefix="rcp-project-provision-", dir="/tmp"))
    os.chown(control_root, os.geteuid(), os.getegid())
    control_root.chmod(0o700)
    metadata = ServerMetadata.create(
        store.path.parent,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=control_root / "control.sock",
    )
    app = create_app(data_dir=store.path.parent, instance_metadata=metadata)

    try:
        with published_server_metadata(store.path.parent, metadata), TestClient(app):
            client = ServerControlClient.from_data_dir(
                store.path.parent,
                expected_server_uid=os.geteuid(),
            )
            plan = client.project_provision_plan(request_id=request.request_id)
            result = client.advance_project_provision(
                request_id=request.request_id,
                boundary_sha256=plan.boundary_sha256,
                target_id=plan.targets[0].target_id,
            )
    finally:
        shutil.rmtree(control_root)

    assert result.step.state == "operator_action_needed"
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None and stored.status == "operator_action_needed"
    assert store.project(request.proposed_project_id) is None


@pytest.mark.parametrize("completed_steps", range(7))
def test_restart_after_every_durable_boundary_resumes_without_duplicate_authority(
    tmp_path: Path,
    completed_steps: int,
) -> None:
    store, request = _store_and_request(tmp_path)
    credentials = _Credentials()
    checkouts = _Checkouts()
    first, _, _ = _coordinator(
        store,
        tmp_path,
        credentials=credentials,
        checkouts=checkouts,
    )
    plan = first.plan(request.request_id)
    boundary = plan.boundary_sha256
    for target in plan.targets[:completed_steps]:
        result = first.advance(
            request.request_id,
            boundary_sha256=boundary,
            target_id=target.target_id,
        )
        assert result.step.state == "succeeded"
        boundary = result.next_boundary_sha256

    resumed, _, _ = _coordinator(
        store,
        tmp_path,
        credentials=credentials,
        checkouts=checkouts,
    )
    _advance_all(resumed, request.request_id)

    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None and stored.status == "ready_for_review"
    assert len(store.project_provisioning_step_receipts(request.request_id)) == 6
    assert store.project(request.proposed_project_id) is None


def test_missing_github_grant_persists_exact_project_resume_then_completes(
    tmp_path: Path,
) -> None:
    store, request = _store_and_request(tmp_path)
    credentials = _Credentials(probe_status="github_grant_needed")
    coordinator, _, _ = _coordinator(store, tmp_path, credentials=credentials)
    output = StringIO()
    args = build_parser().parse_args(
        (
            "server",
            "project",
            "provision",
            request.request_id,
            "--machine-readable",
        )
    )

    exit_code = run_server_command(
        args,
        identity=CallerIdentity(uid=501, username="rcp", host="lab.example"),
        handler=lambda command, identity: prepare_project_provision_command(
            command,
            identity,
            control=coordinator,
        ),
        stream=output,
    )

    assert exit_code == SERVER_CLI_EXIT_OPERATOR_ACTION
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert events[-1]["step"]["phase"] == "repository_write"
    assert events[-1]["step"]["resume_argv"][-4:] == [
        "server",
        "project",
        "provision",
        request.request_id,
    ]
    paused = store.project_provisioning_request(request.request_id)
    assert paused is not None and paused.status == "operator_action_needed"
    assert paused.operator_action is not None
    assert "PRIVATE KEY" not in paused.operator_action.model_dump_json()

    credentials.probe_status = "ready"
    _advance_all(coordinator, request.request_id)
    ready = store.project_provisioning_request(request.request_id)
    assert ready is not None and ready.status == "ready_for_review"


def test_unsafe_credential_path_pauses_with_exact_account_and_project_resume(
    tmp_path: Path,
) -> None:
    store, request = _store_and_request(tmp_path)
    credentials = _RefusingCredentials()
    coordinator, _, _ = _coordinator(store, tmp_path, credentials=credentials)
    plan = coordinator.plan(request.request_id)
    started = coordinator.advance(
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    paused = coordinator.advance(
        request.request_id,
        boundary_sha256=started.next_boundary_sha256,
        target_id=plan.targets[1].target_id,
    )

    assert paused.step.state == "operator_action_needed"
    assert paused.step.performed_by == "human"
    assert paused.step.target.os_account == "rcp"
    assert paused.step.resume_argv[-4:] == (
        "server",
        "project",
        "provision",
        request.request_id,
    )
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None and stored.status == "operator_action_needed"
    assert stored.repositories[0].git_check.status == "operator_action_needed"
    assert store.project(request.proposed_project_id) is None


def test_provider_login_pause_resumes_the_unified_project_command(tmp_path: Path) -> None:
    store, request = _store_and_request(tmp_path)
    launcher = _Launcher(authenticated=False)
    coordinator, _, _ = _coordinator(store, tmp_path, launcher=launcher)
    plan = coordinator.plan(request.request_id)
    boundary = plan.boundary_sha256
    for target in plan.targets[:4]:
        result = coordinator.advance(
            request.request_id,
            boundary_sha256=boundary,
            target_id=target.target_id,
        )
        assert result.step.state == "succeeded"
        boundary = result.next_boundary_sha256

    paused = coordinator.advance(
        request.request_id,
        boundary_sha256=boundary,
        target_id=plan.targets[4].target_id,
    )

    assert paused.step.state == "operator_action_needed"
    assert paused.step.performed_by == "human"
    assert paused.step.resume_argv[-4:] == (
        "server",
        "project",
        "provision",
        request.request_id,
    )
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None and stored.status == "operator_action_needed"
    assert stored.provider_checks[0].status == "operator_action_needed"
    assert stored.provider_checks[0].diagnostic == "Codex is not authenticated."
    assert store.project(request.proposed_project_id) is None


def test_retained_research_checkout_pauses_without_adopting_existing_state(
    tmp_path: Path,
) -> None:
    store, request = _store_and_request(tmp_path)
    checkouts = _RetainedCheckout()
    coordinator, _, _ = _coordinator(store, tmp_path, checkouts=checkouts)
    plan = coordinator.plan(request.request_id)
    boundary = plan.boundary_sha256
    for target in plan.targets[:3]:
        result = coordinator.advance(
            request.request_id,
            boundary_sha256=boundary,
            target_id=target.target_id,
        )
        assert result.step.state == "succeeded"
        boundary = result.next_boundary_sha256

    paused = coordinator.advance(
        request.request_id,
        boundary_sha256=boundary,
        target_id=plan.targets[3].target_id,
    )

    assert paused.step.state == "operator_action_needed"
    assert paused.step.performed_by == "human"
    assert [(field.name, field.value) for field in paused.step.fields] == [
        (
            "repository_path",
            (
                f"{DEFAULT_SERVER_LAYOUT.projects_root}/{request.proposed_project_id}/"
                "repositories/paper"
            ),
        ),
        ("patch_history", True),
    ]
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None and stored.status == "operator_action_needed"
    assert stored.repositories[0].resolved_path is None
    assert stored.repositories[0].checkout_disposition is None
    assert store.project(request.proposed_project_id) is None


def test_key_change_after_write_proof_returns_to_git_action_before_checkout(
    tmp_path: Path,
) -> None:
    store, request = _store_and_request(tmp_path)
    credentials = _Credentials()
    checkouts = _Checkouts()
    coordinator, _, _ = _coordinator(
        store,
        tmp_path,
        credentials=credentials,
        checkouts=checkouts,
    )
    plan = coordinator.plan(request.request_id)
    boundary = plan.boundary_sha256
    for target in plan.targets[:3]:
        result = coordinator.advance(
            request.request_id,
            boundary_sha256=boundary,
            target_id=target.target_id,
        )
        assert result.step.state == "succeeded"
        boundary = result.next_boundary_sha256

    credentials.fingerprint_character = "B"
    paused = coordinator.advance(
        request.request_id,
        boundary_sha256=boundary,
        target_id=plan.targets[3].target_id,
    )

    assert paused.step.state == "operator_action_needed"
    assert paused.step.phase == "repository_checkout"
    assert paused.step.target.kind == "machine"
    assert "Resume this exact request" in paused.step.actions[0].instruction
    assert checkouts.calls == 0
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None and stored.status == "operator_action_needed"
    assert stored.repositories[0].git_check.status == "operator_action_needed"
    assert stored.repositories[0].git_check.public_key_fingerprint == "SHA256:" + ("B" * 43)
    assert store.project_provisioning_step_receipts(request.request_id)[-1].phase == (
        "repository_checkout"
    )


def test_changed_request_refuses_stale_plan_before_machine_effect(tmp_path: Path) -> None:
    store, request = _store_and_request(tmp_path)
    coordinator, credentials, _ = _coordinator(store, tmp_path)
    plan = coordinator.plan(request.request_id)
    first = coordinator.advance(
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    with pytest.raises(ProjectProvisionRefused, match="changed after the plan"):
        coordinator.advance(
            request.request_id,
            boundary_sha256=plan.boundary_sha256,
            target_id=plan.targets[1].target_id,
        )

    assert first.request_status == "setup_in_progress"
    assert credentials.prepare_calls == 0


def _test_server_layout(root: Path) -> ServerLayout:
    service_home = root / "home" / "rcp"
    server_root = service_home / "rcp-server"
    runtime_root = root / "run" / "rcp"
    return ServerLayout(
        service_account="rcp",
        service_home=service_home,
        server_root=server_root,
        source_checkout=server_root / "source",
        releases_root=server_root / "releases",
        data_dir=server_root / "data",
        projects_root=server_root / "projects",
        credentials_root=server_root / "credentials",
        update_checkpoints_root=server_root / "update-checkpoints",
        restore_operations_root=server_root / "restore-operations",
        codex_state_root=service_home / ".codex",
        claude_state_root=service_home / ".claude",
        ssh_state_root=service_home / ".ssh",
        config_path=root / "etc" / "rcp" / "server.toml",
        current_release=root / "etc" / "rcp" / "current",
        runtime_dir=runtime_root,
        control_socket=runtime_root / "control.sock",
        cli_wrapper=root / "usr" / "local" / "bin" / "rcp-server",
        systemd_unit=(root / "etc" / "systemd" / "system" / "research-control-panel.service"),
        service_unit_name="research-control-panel.service",
    )


def _ready_team_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_appendix: bool = False,
):
    layout = _test_server_layout(tmp_path / "installation")
    monkeypatch.setattr(storage_models, "DEFAULT_SERVER_LAYOUT", layout)
    data_dir = tmp_path / "app-data"
    store, _bootstrap = AppStore.initialize_team_space(
        data_dir / "rcp.sqlite3",
        "Final review lab",
    )
    alice = store.preprovision_team_member("Alice")
    bob = store.preprovision_team_member("Bob")
    authorized_by = AuthorizedHuman(
        space_id=store.space_id,
        user_id=alice.user_id,
        display_name="Alice",
    )
    repository_aliases = [
        "paper",
        *(("appendix",) if include_appendix else ()),
    ]
    request = store.create_project_provisioning_request(
        kind="create_team_project",
        authorized_by=authorized_by,
        name="Reviewed team project",
        state_repository="paper",
        project_truth_scope=repository_aliases,
        default_run_truth_scope=repository_aliases,
        machines=[
            ProjectProvisioningMachineIntent(
                alias="server",
                location="local",
                os_account="rcp",
                central_root=str(layout.projects_root),
            )
        ],
        repositories=[
            ProjectProvisioningRepositoryIntent(
                alias=alias,
                repository=parse_github_repository_ref(f"git@github.com:OpenAI/RCP-{alias}.git"),
                machine_alias="server",
            )
            for alias in repository_aliases
        ],
        provider_checks=[
            ProjectProvisioningProviderIntent(
                profile=profile,
                provider="codex",
                runtime_id="codex:exec",
                model="gpt-test",
                reasoning="medium",
                machine_alias="server",
            )
            for profile in AGENT_EXECUTION_PROFILES
        ],
    )
    running = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="finalizer-start",
        phase="provisioning_start",
        expected_revision=request.revision,
        expected_status="waiting_for_server_setup",
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )
    checked_at = store.now()
    machines = [
        running.machines[0].model_copy(update={"resolved_central_root": str(layout.projects_root)})
    ]
    repository_paths = {
        alias: layout.project_repository_dir(request.proposed_project_id, alias)
        for alias in repository_aliases
    }
    for repository_path in repository_paths.values():
        repository_path.mkdir(parents=True)
    repositories = [
        repository.model_copy(
            update={
                "resolved_path": str(repository_paths[repository.alias]),
                "checkout_disposition": "request_created",
                "git_check": ProjectProvisioningGitCheckRecord(
                    status="ready",
                    commit="a" * 40,
                    write_verified=True,
                    deploy_key_label=(
                        f"rcp:{store.space_id}:{request.proposed_project_id}:{repository.alias}"
                    ),
                    public_key_fingerprint="SHA256:" + ("A" * 43),
                    checked_at=checked_at,
                ),
            }
        )
        for repository in running.repositories
    ]
    providers = [
        ProjectProvisioningProviderCheckRecord(
            **check.model_dump(
                mode="json",
                exclude={
                    "status",
                    "binary_path",
                    "version",
                    "resolved_runtime_id",
                    "execution_account",
                    "checked_at",
                    "diagnostic",
                },
            ),
            status="ready",
            binary_path="/usr/local/bin/codex",
            version="codex-cli 1.2.3",
            resolved_runtime_id=configured_runtime_id("codex", "exec"),
            execution_account="rcp",
            checked_at=checked_at,
        )
        for check in running.provider_checks
    ]
    ready = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="finalizer-ready",
        phase="provisioning_review",
        expected_revision=running.revision,
        expected_status="setup_in_progress",
        to_status="ready_for_review",
        machines=machines,
        repositories=repositories,
        provider_checks=providers,
    )
    selected = [bob.user_id]
    app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, opened: opened.space_user(selected[0]),
    )
    return app, ready, alice, bob, repository_paths["paper"]


def _complete_ready_request(client: TestClient, request) -> object:
    return client.post(
        f"/api/project-provisioning/requests/{request.request_id}/complete",
        json={"final_review_digest": request.final_review_digest},
    )


def test_final_review_creates_exact_reserved_project_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, ready, _alice, bob, repository_path = _ready_team_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        completed = _complete_ready_request(client, ready)
        repeated = _complete_ready_request(client, ready)
        projects = client.get("/api/projects")

    assert completed.status_code == 200
    assert repeated.status_code == 200
    assert completed.json() == repeated.json()
    assert completed.json()["status"] == "completed"
    assert completed.json()["proposed_project_id"] == ready.proposed_project_id
    assert projects.status_code == 200
    assert [project["id"] for project in projects.json()] == [ready.proposed_project_id]
    store = app.state.services.store
    project = store.project(ready.proposed_project_id)
    assert project is not None
    assert project.home_space_id == store.space_id
    manifest = load_manifest(repository_path / ".research" / "manifest.toml")
    assert manifest.name == "Reviewed team project"
    assert [machine.model_dump() for machine in manifest.machines] == [
        {
            "alias": "server",
            "host": "",
            "os_account": "rcp",
            "provider_paths": {"codex": "/usr/local/bin/codex"},
        }
    ]
    assert manifest.repository_paths == {"paper": str(repository_path)}
    assert manifest.project.truth_scope == ["paper"]
    assert manifest.state.repository == "paper"
    assert manifest.agent.default_run_truth_scope == ["paper"]
    assert manifest.agent.default_auto_research_invocation_ceiling == 10
    for profile_name in AGENT_EXECUTION_PROFILES:
        profile = manifest.agent_profile(profile_name)
        assert profile.provider == "codex"
        assert profile.runtime == "exec"
        assert profile.model == "gpt-test"
        assert profile.reasoning == "medium"
        assert profile.run_on == "server"
        assert profile.permissions == permissions_for(profile_name)
    members = store.project_members(ready.proposed_project_id)
    assert [(member.user_id, member.seated_by) for member in members] == [
        (bob.user_id, bob.user_id)
    ]
    patches = sorted((repository_path / ".research" / "patches").glob("*.json"))
    assert [path.name for path in patches] == ["000001.json"]
    patch = json.loads(patches[0].read_text(encoding="utf-8"))
    assert patch["kind"] == "identity"
    assert patch["project_identity"] == {
        "project_id": ready.proposed_project_id,
        "home_space_id": store.space_id,
        "action": "created",
    }
    receipts = store.project_provisioning_step_receipts(ready.request_id)
    assert [receipt.phase for receipt in receipts].count("member_finalize") == 1


def test_final_review_refuses_stale_digest_and_new_retained_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, ready, _alice, _bob, repository_path = _ready_team_app(tmp_path, monkeypatch)
    patches = repository_path / ".research" / "patches"
    patches.mkdir(parents=True)
    (patches / "000001.json").write_text("{}\n", encoding="utf-8")

    with TestClient(app) as client:
        stale = client.post(
            f"/api/project-provisioning/requests/{ready.request_id}/complete",
            json={"final_review_digest": "0" * 64},
        )
        retained = _complete_ready_request(client, ready)

    assert stale.status_code == 409
    assert "review changed" in stale.json()["detail"]
    assert retained.status_code == 409
    assert "Patch history appeared" in retained.json()["detail"]
    assert app.state.services.store.project(ready.proposed_project_id) is None
    assert not (repository_path / ".research" / "manifest.toml").exists()


def test_final_review_refuses_a_checkout_path_replaced_by_a_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, ready, _alice, _bob, repository_path = _ready_team_app(tmp_path, monkeypatch)
    moved = tmp_path / "moved-checkout"
    repository_path.rename(moved)
    replacement = tmp_path / "replacement-checkout"
    replacement.mkdir()
    repository_path.symlink_to(replacement, target_is_directory=True)

    with TestClient(app) as client:
        response = _complete_ready_request(client, ready)

    assert response.status_code == 409
    assert "resolves to another path" in response.json()["detail"]
    assert app.state.services.store.project(ready.proposed_project_id) is None
    assert not (replacement / ".research").exists()


def test_final_review_rechecks_every_reviewed_checkout_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, ready, _alice, _bob, canonical_path = _ready_team_app(
        tmp_path,
        monkeypatch,
        include_appendix=True,
    )
    appendix_path = Path(
        next(
            repository.resolved_path
            for repository in ready.repositories
            if repository.alias == "appendix"
        )
    )
    moved = tmp_path / "moved-appendix"
    appendix_path.rename(moved)
    replacement = tmp_path / "replacement-appendix"
    replacement.mkdir()
    appendix_path.symlink_to(replacement, target_is_directory=True)

    with TestClient(app) as client:
        response = _complete_ready_request(client, ready)

    assert response.status_code == 409
    assert "prepared checkout appendix now resolves to another path" in response.json()["detail"]
    assert app.state.services.store.project(ready.proposed_project_id) is None
    assert not (canonical_path / ".research").exists()
    assert not (replacement / ".research").exists()


class _FinalizationBoundaryCrash(RuntimeError):
    pass


@pytest.mark.parametrize(
    "boundary",
    ["manifest", "identity", "catalog_row", "membership", "completion"],
)
def test_final_review_recovers_after_each_durable_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    app, ready, _alice, _bob, repository_path = _ready_team_app(tmp_path, monkeypatch)
    store = app.state.services.store

    with TestClient(app) as client:
        with monkeypatch.context() as crash:
            if boundary == "manifest":
                original_write = setup_code._exclusive_write

                def write_then_crash(path: Path, content: str) -> None:
                    original_write(path, content)
                    raise _FinalizationBoundaryCrash(boundary)

                crash.setattr(setup_code, "_exclusive_write", write_then_crash)
            elif boundary == "identity":
                original_claim = HistoryManager.claim_project_identity

                def claim_then_crash(self, *args, **kwargs):
                    original_claim(self, *args, **kwargs)
                    raise _FinalizationBoundaryCrash(boundary)

                crash.setattr(HistoryManager, "claim_project_identity", claim_then_crash)
            elif boundary == "catalog_row":

                def seat_after_row_crashes(*_args, **_kwargs):
                    raise _FinalizationBoundaryCrash(boundary)

                crash.setattr(store, "seat_project_member", seat_after_row_crashes)
            elif boundary == "membership":
                original_register = app.state.catalog.register_prepared_team_project

                def register_then_crash(*args, **kwargs):
                    original_register(*args, **kwargs)
                    raise _FinalizationBoundaryCrash(boundary)

                crash.setattr(
                    app.state.catalog,
                    "register_prepared_team_project",
                    register_then_crash,
                )
            else:
                original_transition = store.transition_project_provisioning_request

                def transition_then_crash(*args, **kwargs):
                    result = original_transition(*args, **kwargs)
                    if kwargs.get("to_status") == "completed":
                        raise _FinalizationBoundaryCrash(boundary)
                    return result

                crash.setattr(
                    store,
                    "transition_project_provisioning_request",
                    transition_then_crash,
                )

            with pytest.raises(_FinalizationBoundaryCrash, match=boundary):
                _complete_ready_request(client, ready)

        recovered = _complete_ready_request(client, ready)

    assert recovered.status_code == 200
    assert recovered.json()["status"] == "completed"
    assert store.project(ready.proposed_project_id) is not None
    assert [path.name for path in (repository_path / ".research" / "patches").glob("*.json")] == [
        "000001.json"
    ]
    receipts = store.project_provisioning_step_receipts(ready.request_id)
    assert [receipt.phase for receipt in receipts].count("member_finalize") == 1
