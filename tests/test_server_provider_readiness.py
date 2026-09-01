from __future__ import annotations

import json
import os
import pwd
import shutil
import tempfile
import uuid
from io import StringIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rcp.__main__ import build_parser
from rcp.agents.launcher import (
    AgentLauncher,
    ProviderExecutionAccount,
    ProviderReadiness,
)
from rcp.api import create_app
from rcp.core.models import AuthorizedHuman
from rcp.providers import ModelChoice
from rcp.server_ops.cli import (
    SERVER_CLI_EXIT_OPERATOR_ACTION,
    CallerIdentity,
    run_server_command,
)
from rcp.server_ops.control import ServerControlClient, ServerControlError
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.server_ops.models import CommandAction, ExternalAction
from rcp.server_ops.provider_readiness import (
    ProviderReadinessCoordinator,
    ProviderReadinessRefused,
    prepare_provider_check_command,
)
from rcp.server_runtime import ServerMetadata, published_server_metadata
from rcp.storage import (
    AppStore,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
    ProjectRecord,
)


class _FakeLauncher:
    def __init__(
        self,
        *,
        account: ProviderExecutionAccount,
        readiness: ProviderReadiness,
    ) -> None:
        self.account = account
        self.readiness_result = readiness
        self.account_calls: list[str] = []
        self.readiness_calls: list[tuple[str, str, str | None, bool]] = []

    def execution_account(self, *, host: str = "") -> ProviderExecutionAccount:
        self.account_calls.append(host)
        return self.account.model_copy(update={"host": host})

    def readiness(
        self,
        provider: str,
        *,
        host: str = "",
        binary: str | None = None,
        refresh: bool = False,
    ) -> ProviderReadiness:
        self.readiness_calls.append((provider, host, binary, refresh))
        return self.readiness_result.model_copy(deep=True)


def _ready_provider() -> ProviderReadiness:
    return ProviderReadiness(
        provider="codex",
        installed=True,
        authenticated=True,
        version="codex-cli 1.2.3",
        binary_path="/usr/local/bin/codex",
        path_state="resolved",
        models=[ModelChoice(id="gpt-test", label="GPT Test", reasoning=["medium"])],
    )


def _team_store(tmp_path: Path) -> tuple[AppStore, AuthorizedHuman]:
    data_dir = tmp_path / "team"
    store, _bootstrap = AppStore.initialize_team_space(
        data_dir / "rcp.sqlite3",
        "Provider lab",
    )
    member = store.preprovision_team_member("Alice")
    return store, AuthorizedHuman(
        space_id=store.space_id,
        user_id=member.user_id,
        display_name="Alice",
    )


def _request(
    store: AppStore,
    authorizer: AuthorizedHuman,
    *,
    remote: bool = False,
    profile: str = "seed",
    profiles: tuple[str, ...] | None = None,
):
    machine = ProjectProvisioningMachineIntent(
        alias="compute" if remote else "server",
        location="ssh" if remote else "local",
        host="gpu.example" if remote else "",
        os_account="alice" if remote else "rcp",
        central_root=(
            "/home/alice/.local/share/rcp/projects"
            if remote
            else str(DEFAULT_SERVER_LAYOUT.projects_root)
        ),
    )
    return store.create_project_provisioning_request(
        kind="create_team_project",
        authorized_by=authorizer,
        machines=[machine],
        repositories=[
            ProjectProvisioningRepositoryIntent(
                alias="paper",
                repository=parse_github_repository_ref("git@github.com:OpenAI/RCP.git"),
                machine_alias=machine.alias,
            )
        ],
        provider_checks=[
            ProjectProvisioningProviderIntent(
                profile=selected_profile,
                provider="codex",
                runtime_id="codex:exec",
                model="gpt-test",
                reasoning="medium",
                machine_alias=machine.alias,
            )
            for selected_profile in (profiles or (profile,))
        ],
    )


def _metadata(store: AppStore, socket_path: Path) -> ServerMetadata:
    return ServerMetadata.create(
        store.path.parent,
        host="127.0.0.1",
        port=8421,
        owner_kind="cli",
        control_socket=socket_path,
    )


def _coordinator(
    store: AppStore,
    launcher: _FakeLauncher,
    tmp_path: Path,
) -> ProviderReadinessCoordinator:
    return ProviderReadinessCoordinator(
        store,
        launcher,  # type: ignore[arg-type]
        _metadata(store, tmp_path / "control.sock"),
        local_host="lab.example",
    )


def test_request_plan_is_read_only_and_success_persists_exact_nonsecret_proof(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=_ready_provider(),
    )
    coordinator = _coordinator(store, launcher, tmp_path)

    plan = coordinator.plan("request", request.request_id)

    assert launcher.account_calls == []
    assert launcher.readiness_calls == []
    assert plan.selector_id == request.request_id
    assert len(plan.targets) == 1
    assert plan.targets[0].step.target.os_account == "rcp"

    checked = coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    assert checked.step.state == "succeeded"
    assert launcher.account_calls == [""]
    assert launcher.readiness_calls == [("codex", "", None, True)]
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None
    assert stored.status == "setup_in_progress"
    assert stored.revision == 1
    proof = stored.provider_checks[0]
    assert proof.status == "ready"
    assert proof.binary_path == "/usr/local/bin/codex"
    assert proof.version == "codex-cli 1.2.3"
    assert proof.resolved_runtime_id == "codex.exec-json.v1"
    assert proof.execution_account == "rcp"
    assert proof.checked_at is not None
    assert proof.diagnostic is None

    repeated = coordinator.plan("request", request.request_id)
    coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=repeated.boundary_sha256,
        target_id=repeated.targets[0].target_id,
    )
    assert launcher.readiness_calls[-1] == (
        "codex",
        "",
        "/usr/local/bin/codex",
        True,
    )


def test_missing_native_login_pauses_the_initial_request_with_exact_operator_action(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=ProviderReadiness(
            provider="codex",
            installed=True,
            authenticated=False,
            version="codex-cli 1.2.3",
            binary_path="/usr/local/bin/codex",
            reason="Codex is not authenticated on this account.",
        ),
    )
    coordinator = _coordinator(store, launcher, tmp_path)
    plan = coordinator.plan("request", request.request_id)

    checked = coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    assert checked.step.state == "operator_action_needed"
    assert checked.step.actions == (
        CommandAction(argv=("sudo", "-u", "rcp", "-H", "/usr/local/bin/codex", "login")),
        ExternalAction(
            instruction=(
                "Complete Codex's native login directly as OS account rcp; do not paste "
                "provider credentials into RCP."
            )
        ),
    )
    assert checked.step.resume_argv[-4:] == (
        "provider",
        "check",
        "--request",
        request.request_id,
    )
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None
    assert stored.status == "operator_action_needed"
    assert stored.operator_action == checked.step
    assert stored.provider_checks[0].status == "operator_action_needed"
    assert stored.provider_checks[0].binary_path is None


def test_wrong_remote_account_stops_before_provider_probe(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer, remote=True)
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="bob"),
        readiness=_ready_provider(),
    )
    coordinator = _coordinator(store, launcher, tmp_path)
    plan = coordinator.plan("request", request.request_id)

    checked = coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    assert checked.step.state == "operator_action_needed"
    assert "requires OS account alice" in checked.step.message
    assert "reached bob" in checked.step.message
    assert launcher.readiness_calls == []
    assert checked.step.actions[-1] == CommandAction(
        argv=("sudo", "-u", "rcp", "-H", "ssh", "gpu.example", "id -un")
    )


def test_unsupported_saved_model_requires_configuration_not_login(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    readiness = _ready_provider().model_copy(
        update={"models": [ModelChoice(id="other-model", label="Other")]}
    )
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=readiness,
    )
    coordinator = _coordinator(store, launcher, tmp_path)
    plan = coordinator.plan("request", request.request_id)

    checked = coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    assert checked.step.state == "operator_action_needed"
    assert "saved model 'gpt-test'" in checked.step.message
    assert all(isinstance(action, ExternalAction) for action in checked.step.actions)
    assert "setup or settings flow" in checked.step.actions[0].instruction


def test_missing_catalog_cannot_approve_an_explicit_saved_model(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=_ready_provider().model_copy(update={"models": []}),
    )
    coordinator = _coordinator(store, launcher, tmp_path)
    plan = coordinator.plan("request", request.request_id)

    checked = coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    assert checked.step.state == "operator_action_needed"
    assert "did not return a model catalog" in checked.step.message


def test_orchestrator_uses_the_real_provider_version_floor(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer, profile="orchestrator")
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=_ready_provider().model_copy(update={"version": "codex-cli 0.137.9"}),
    )
    coordinator = _coordinator(store, launcher, tmp_path)
    plan = coordinator.plan("request", request.request_id)

    checked = coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=plan.boundary_sha256,
        target_id=plan.targets[0].target_id,
    )

    assert checked.step.state == "operator_action_needed"
    assert "requires 0.138.0 or newer" in checked.step.message


def test_unexpected_launcher_failure_is_not_misreported_as_missing_install(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=_ready_provider(),
    )

    def fail_readiness(*_args, **_kwargs):
        raise RuntimeError("implementation bug")

    launcher.readiness = fail_readiness  # type: ignore[method-assign]
    coordinator = _coordinator(store, launcher, tmp_path)
    plan = coordinator.plan("request", request.request_id)

    with pytest.raises(RuntimeError, match="implementation bug"):
        coordinator.check(
            "request",
            request.request_id,
            boundary_sha256=plan.boundary_sha256,
            target_id=plan.targets[0].target_id,
        )

    stored = store.project_provisioning_request(request.request_id)
    assert stored == request


def test_changed_durable_request_refuses_stale_plan_before_probe(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=_ready_provider(),
    )
    coordinator = _coordinator(store, launcher, tmp_path)
    plan = coordinator.plan("request", request.request_id)
    store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="another-step",
        phase="setup_start",
        expected_revision=0,
        expected_status="waiting_for_server_setup",
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )

    with pytest.raises(ProviderReadinessRefused, match="changed after the plan"):
        coordinator.check(
            "request",
            request.request_id,
            boundary_sha256=plan.boundary_sha256,
            target_id=plan.targets[0].target_id,
        )

    assert launcher.account_calls == []
    assert launcher.readiness_calls == []


def test_success_for_one_profile_preserves_another_profiles_operator_action(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer, profiles=("seed", "refresh"))
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=ProviderReadiness(
            provider="codex",
            installed=True,
            authenticated=False,
            version="codex-cli 1.2.3",
            binary_path="/usr/local/bin/codex",
        ),
    )
    coordinator = _coordinator(store, launcher, tmp_path)
    initial = coordinator.plan("request", request.request_id)
    paused = coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=initial.boundary_sha256,
        target_id=initial.targets[1].target_id,
    )
    assert paused.step.state == "operator_action_needed"

    launcher.readiness_result = _ready_provider()
    resumed = coordinator.plan("request", request.request_id)
    coordinator.check(
        "request",
        request.request_id,
        boundary_sha256=resumed.boundary_sha256,
        target_id=resumed.targets[0].target_id,
    )

    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None
    assert stored.status == "operator_action_needed"
    assert stored.operator_action == paused.step
    assert stored.provider_checks[0].status == "ready"
    assert stored.provider_checks[1].status == "operator_action_needed"


def test_project_selector_resolves_only_the_six_stored_profiles(
    tmp_path: Path,
    manifest,
) -> None:
    store, _authorizer = _team_store(tmp_path)
    project_id = str(uuid.uuid4())
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=str(manifest.path),
            name="Existing team project",
            state_location="local",
            state_remote=False,
            added_at=store.now(),
        )
    )
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=_ready_provider(),
    )
    coordinator = _coordinator(store, launcher, tmp_path)

    plan = coordinator.plan("project", project_id)

    assert [target.step.number for target in plan.targets] == list(range(1, 7))
    assert {target.step.fields for target in plan.targets} == {()}
    assert launcher.account_calls == []
    assert launcher.readiness_calls == []


def test_remote_project_requires_and_uses_its_manifest_account(
    tmp_path: Path,
    manifest,
) -> None:
    store, _authorizer = _team_store(tmp_path)
    original = manifest.path.read_text(encoding="utf-8")
    remote_manifest = tmp_path / "remote-manifest.toml"
    remote_manifest.write_text(
        original.replace(
            'host = ""',
            'host = "gpu.example"\nos_account = "alice"',
            1,
        ),
        encoding="utf-8",
    )
    project_id = str(uuid.uuid4())
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=str(remote_manifest),
            name="Remote team project",
            state_location="gpu.example",
            state_remote=True,
            added_at=store.now(),
        )
    )
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="alice"),
        readiness=_ready_provider(),
    )
    coordinator = _coordinator(store, launcher, tmp_path)

    plan = coordinator.plan("project", project_id)

    assert {target.step.target.host for target in plan.targets} == {"gpu.example"}
    assert {target.step.target.os_account for target in plan.targets} == {"alice"}

    remote_manifest.write_text(
        original.replace('host = ""', 'host = "gpu.example"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ProviderReadinessRefused, match="no recorded remote execution account"):
        coordinator.plan("project", project_id)


def test_cli_control_app_and_storage_share_one_request_bound_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    control_root = Path(tempfile.mkdtemp(prefix="rcp-provider-", dir="/tmp"))
    os.chown(control_root, os.geteuid(), os.getegid())
    control_root.chmod(0o700)
    metadata = _metadata(store, control_root / "control.sock")
    app = create_app(data_dir=store.path.parent, instance_metadata=metadata)
    launcher = app.state.catalog.launcher
    fake = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=_ready_provider(),
    )
    monkeypatch.setattr(launcher, "execution_account", fake.execution_account)
    monkeypatch.setattr(launcher, "readiness", fake.readiness)
    output = StringIO()
    args = build_parser().parse_args(
        (
            "server",
            "provider",
            "check",
            "--request",
            request.request_id,
            "--machine-readable",
        )
    )

    try:
        with published_server_metadata(store.path.parent, metadata), TestClient(app):
            client = ServerControlClient.from_data_dir(
                store.path.parent,
                expected_server_uid=os.geteuid(),
            )
            preview = client.provider_readiness_plan(
                selector_kind="request",
                selector_id=request.request_id,
            )
            assert preview.targets[0].step.state == "pending"
            assert fake.account_calls == []
            exit_code = run_server_command(
                args,
                identity=CallerIdentity(uid=501, username="rcp", host="lab.example"),
                handler=lambda command, identity: prepare_provider_check_command(
                    command,
                    identity,
                    control_factory=lambda _layout: client,
                ),
                stream=output,
            )

            stale_request = _request(store, authorizer)
            stale_plan = client.provider_readiness_plan(
                selector_kind="request",
                selector_id=stale_request.request_id,
            )
            store.transition_project_provisioning_request(
                stale_request.request_id,
                receipt_id="concurrent-step",
                phase="setup_start",
                expected_revision=0,
                expected_status="waiting_for_server_setup",
                to_status="setup_in_progress",
                machines=stale_request.machines,
                repositories=stale_request.repositories,
                provider_checks=stale_request.provider_checks,
            )
            with pytest.raises(ServerControlError, match="changed after the plan") as caught:
                client.check_provider_readiness(
                    selector_kind="request",
                    selector_id=stale_request.request_id,
                    boundary_sha256=stale_plan.boundary_sha256,
                    target_id=stale_plan.targets[0].target_id,
                )
            assert caught.value.code == "operation_refused"
    finally:
        shutil.rmtree(control_root)

    assert exit_code == 0
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [event["event"] for event in events] == ["plan", "step", "step"]
    assert events[-1]["step"]["state"] == "succeeded"
    stored = store.project_provisioning_request(request.request_id)
    assert stored is not None
    assert stored.provider_checks[0].execution_account == "rcp"


def test_cli_returns_operator_action_exit_for_native_login(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=ProviderReadiness(
            provider="codex",
            installed=True,
            authenticated=False,
            version="codex-cli 1.2.3",
            binary_path="/usr/local/bin/codex",
        ),
    )
    coordinator = _coordinator(store, launcher, tmp_path)

    class Control:
        def provider_readiness_plan(self, **kwargs):
            return coordinator.plan(kwargs["selector_kind"], kwargs["selector_id"])

        def check_provider_readiness(self, **kwargs):
            return coordinator.check(
                kwargs["selector_kind"],
                kwargs["selector_id"],
                boundary_sha256=kwargs["boundary_sha256"],
                target_id=kwargs["target_id"],
            )

    args = build_parser().parse_args(
        ("server", "provider", "check", "--request", request.request_id)
    )
    output = StringIO()

    exit_code = run_server_command(
        args,
        identity=CallerIdentity(uid=501, username="rcp", host="lab.example"),
        handler=lambda command, identity: prepare_provider_check_command(
            command,
            identity,
            control_factory=lambda _layout: Control(),
        ),
        stream=output,
    )

    assert exit_code == SERVER_CLI_EXIT_OPERATOR_ACTION
    assert "operator action needed" in output.getvalue().lower()
    assert "/usr/local/bin/codex login" in output.getvalue()


def test_cli_shows_a_safe_durable_boundary_refusal(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _request(store, authorizer)
    launcher = _FakeLauncher(
        account=ProviderExecutionAccount(host="", reachable=True, os_account="rcp"),
        readiness=_ready_provider(),
    )
    coordinator = _coordinator(store, launcher, tmp_path)

    class Control:
        def provider_readiness_plan(self, **kwargs):
            return coordinator.plan(kwargs["selector_kind"], kwargs["selector_id"])

        def check_provider_readiness(self, **_kwargs):
            raise ServerControlError(
                "operation_refused",
                "The provider configuration changed after the plan; rerun the command.",
            )

    args = build_parser().parse_args(
        ("server", "provider", "check", "--request", request.request_id)
    )
    output = StringIO()

    exit_code = run_server_command(
        args,
        identity=CallerIdentity(uid=501, username="rcp", host="lab.example"),
        handler=lambda command, identity: prepare_provider_check_command(
            command,
            identity,
            control_factory=lambda _layout: Control(),
        ),
        stream=output,
    )

    assert exit_code == 1
    assert "provider configuration changed after the plan" in output.getvalue()


def test_local_execution_account_probe_reports_the_process_account() -> None:
    account = AgentLauncher().execution_account()

    assert account.reachable is True
    assert account.os_account == pwd.getpwuid(os.geteuid()).pw_name
    assert account.reason is None
