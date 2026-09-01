from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from rcp.core.models import AuthorizedHuman
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.server_ops.models import (
    CommandAction,
    ExternalAction,
    ExternalServiceTarget,
    NonsecretField,
    ServerStep,
)
from rcp.storage import (
    AppStore,
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
)
from rcp.storage.provisioning import project_provisioning_review_digest


def _team_store(tmp_path: Path) -> tuple[AppStore, AuthorizedHuman]:
    store = AppStore(tmp_path / "team" / "rcp.sqlite3", space_kind="team")
    member = store.preprovision_team_member("Alice")
    return store, AuthorizedHuman(
        space_id=store.space_id,
        user_id=member.user_id,
        display_name="Alice",
    )


def _machine() -> ProjectProvisioningMachineIntent:
    return ProjectProvisioningMachineIntent(
        alias="server",
        location="local",
        os_account="rcp",
        central_root=str(DEFAULT_SERVER_LAYOUT.projects_root),
    )


def _repository() -> ProjectProvisioningRepositoryIntent:
    return ProjectProvisioningRepositoryIntent(
        alias="paper",
        repository=parse_github_repository_ref("git@github.com:OpenAI/RCP.git"),
        machine_alias="server",
    )


def _provider() -> ProjectProvisioningProviderIntent:
    return ProjectProvisioningProviderIntent(
        profile="seed",
        provider="codex",
        runtime_id="codex:exec",
        model="gpt-5.6-luna",
        reasoning="medium",
        machine_alias="server",
    )


def _create(store: AppStore, authorizer: AuthorizedHuman):
    return store.create_project_provisioning_request(
        kind="create_team_project",
        authorized_by=authorizer,
        machines=[_machine()],
        repositories=[_repository()],
        provider_checks=[_provider()],
        name="Shared paper project",
        state_repository="paper",
        project_truth_scope=["paper"],
        default_run_truth_scope=["paper"],
    )


def _create_legacy(store: AppStore, authorizer: AuthorizedHuman):
    return store.create_project_provisioning_request(
        kind="create_team_project",
        authorized_by=authorizer,
        machines=[_machine()],
        repositories=[_repository()],
        provider_checks=[_provider()],
    )


def _operator_action(request_id: str) -> ServerStep:
    return ServerStep(
        number=2,
        title="Grant repository write access",
        purpose="Let the central checkout prove one request-scoped Git write.",
        performed_by="human",
        target=ExternalServiceTarget(
            service="github.com",
            resource="openai/rcp",
            destination_url="https://github.com/openai/rcp/settings/keys",
            required_authority_role="repository administrator",
        ),
        phase="github_grant",
        state="operator_action_needed",
        expected_success="The request-scoped write probe succeeds and cleans up its ref.",
        message="Add this public key and enable Allow write access.",
        actions=(
            ExternalAction(
                instruction=(
                    "Add the displayed public key to openai/rcp and enable Allow write access."
                )
            ),
            CommandAction(
                argv=(
                    "sudo",
                    "-n",
                    "-u",
                    "rcp",
                    "-H",
                    "/usr/local/bin/rcp",
                    "server",
                    "project",
                    "provision",
                    request_id,
                    "--machine-readable",
                )
            ),
        ),
        fields=(
            NonsecretField(name="deploy_key_label", value=f"rcp:space:{request_id}:paper"),
            NonsecretField(name="deploy_public_key", value="ssh-ed25519 AAAATEST public"),
        ),
        resume_argv=(
            "sudo",
            "-n",
            "-u",
            "rcp",
            "-H",
            "/usr/local/bin/rcp",
            "server",
            "project",
            "provision",
            request_id,
            "--machine-readable",
        ),
    )


def test_new_request_reserves_one_canonical_project_namespace_without_creating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, authorizer = _team_store(tmp_path)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: pytest.fail("request creation performed DNS"),
    )

    record = _create(store, authorizer)

    assert uuid.UUID(record.request_id).version == 4
    assert uuid.UUID(record.proposed_project_id).version == 4
    assert record.status == "waiting_for_server_setup"
    assert record.target_space_id == store.space_id
    assert record.authorized_by == authorizer
    assert record.repositories[0].repository.identity == "openai/rcp"
    assert record.repositories[0].intended_path == str(
        DEFAULT_SERVER_LAYOUT.projects_root / record.proposed_project_id / "repositories" / "paper"
    )
    assert record.repositories[0].resolved_path is None
    assert store.project(record.proposed_project_id) is None
    assert not Path(record.repositories[0].intended_path).exists()

    reopened = AppStore(store.path, space_kind="team")
    assert reopened.project_provisioning_request(record.request_id) == record
    assert reopened.project_provisioning_requests() == [record]
    with sqlite3.connect(store.path) as connection:
        raw = connection.execute(
            "SELECT repositories_json FROM project_provisioning_requests"
        ).fetchone()[0]
    assert "openai/rcp" in raw
    assert "OpenAI" not in raw
    assert "git@github.com" not in raw


def test_incoming_transfer_uses_the_existing_project_id_and_rejects_wrong_id_shapes(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)
    project_id = str(uuid.uuid4())

    incoming = store.create_project_provisioning_request(
        kind="incoming_transfer",
        authorized_by=authorizer,
        machines=[_machine()],
        repositories=[_repository()],
        provider_checks=[_provider()],
        source_project_id=project_id,
    )

    assert incoming.proposed_project_id == project_id
    assert store.project(project_id) is None
    with pytest.raises(ValueError, match="requires the source project id"):
        store.create_project_provisioning_request(
            kind="incoming_transfer",
            authorized_by=authorizer,
            machines=[_machine()],
            repositories=[_repository()],
            provider_checks=[_provider()],
        )
    with pytest.raises(ValueError, match="cannot name a source project id"):
        store.create_project_provisioning_request(
            kind="create_team_project",
            authorized_by=authorizer,
            machines=[_machine()],
            repositories=[_repository()],
            provider_checks=[_provider()],
            source_project_id=str(uuid.uuid4()),
        )


def test_request_requires_the_exact_team_space_and_current_human_authorizer(
    tmp_path: Path,
) -> None:
    personal = AppStore(tmp_path / "personal" / "rcp.sqlite3")
    owner = personal.local_owner
    assert owner is not None
    personal_authorizer = AuthorizedHuman(
        space_id=personal.space_id,
        user_id=owner.user_id,
        display_name="Owner",
    )
    with pytest.raises(ValueError, match="exact team space"):
        _create(personal, personal_authorizer)

    team, authorizer = _team_store(tmp_path / "other")
    stranger = authorizer.model_copy(update={"user_id": str(uuid.uuid4())})
    with pytest.raises(ValueError, match="not a current space member"):
        _create(team, stranger)
    wrong_space = authorizer.model_copy(update={"space_id": str(uuid.uuid4())})
    with pytest.raises(ValidationError, match="target space"):
        team.create_project_provisioning_request(
            kind="create_team_project",
            authorized_by=wrong_space,
            machines=[_machine()],
            repositories=[_repository()],
            provider_checks=[_provider()],
        )
    assert team.project_provisioning_requests() == []


def test_duplicate_repositories_machines_and_profiles_fail_before_persistence(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)

    with pytest.raises(ValidationError, match="machine aliases"):
        store.create_project_provisioning_request(
            kind="create_team_project",
            authorized_by=authorizer,
            machines=[_machine(), _machine()],
            repositories=[_repository()],
            provider_checks=[_provider()],
        )
    with pytest.raises(ValidationError, match="repository aliases"):
        store.create_project_provisioning_request(
            kind="create_team_project",
            authorized_by=authorizer,
            machines=[_machine()],
            repositories=[_repository(), _repository()],
            provider_checks=[_provider()],
        )
    with pytest.raises(ValidationError, match="provider profiles"):
        store.create_project_provisioning_request(
            kind="create_team_project",
            authorized_by=authorizer,
            machines=[_machine()],
            repositories=[_repository()],
            provider_checks=[_provider(), _provider()],
        )
    assert store.project_provisioning_requests() == []


def test_guarded_receipted_transitions_resume_and_bind_final_review(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _create(store, authorizer)

    running = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="setup-started",
        phase="setup_start",
        expected_revision=0,
        expected_status="waiting_for_server_setup",
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )
    action = _operator_action(request.request_id)
    paused = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="github-grant-needed",
        phase="github_grant",
        expected_revision=1,
        expected_status="setup_in_progress",
        to_status="operator_action_needed",
        machines=running.machines,
        repositories=running.repositories,
        provider_checks=running.provider_checks,
        retryable_diagnostic="  The write grant is not ready yet.  ",
        operator_action=action,
    )
    assert paused.operator_action == action
    assert paused.retryable_diagnostic == "The write grant is not ready yet."

    replayed = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="github-grant-needed",
        phase="github_grant",
        expected_revision=1,
        expected_status="setup_in_progress",
        to_status="operator_action_needed",
        machines=running.machines,
        repositories=running.repositories,
        provider_checks=running.provider_checks,
        retryable_diagnostic="The write grant is not ready yet.",
        operator_action=action,
    )
    assert replayed == paused
    with pytest.raises(ValueError, match="another step"):
        store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="github-grant-needed",
            phase="github_grant",
            expected_revision=1,
            expected_status="setup_in_progress",
            to_status="operator_action_needed",
            machines=running.machines,
            repositories=running.repositories,
            provider_checks=running.provider_checks,
            retryable_diagnostic="A different meaning.",
            operator_action=action,
        )

    resumed = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="github-grant-resumed",
        phase="github_grant",
        expected_revision=2,
        expected_status="operator_action_needed",
        to_status="setup_in_progress",
        machines=paused.machines,
        repositories=paused.repositories,
        provider_checks=paused.provider_checks,
    )
    checked_at = store.now()
    machines = [
        resumed.machines[0].model_copy(
            update={"resolved_central_root": resumed.machines[0].central_root}
        )
    ]
    resolved_path = resumed.repositories[0].intended_path
    repositories = [
        resumed.repositories[0].model_copy(
            update={
                "resolved_path": resolved_path,
                "checkout_disposition": "request_created",
                "git_check": ProjectProvisioningGitCheckRecord(
                    status="ready",
                    commit="a" * 40,
                    write_verified=True,
                    deploy_key_label=(f"rcp:{store.space_id}:{request.proposed_project_id}:paper"),
                    public_key_fingerprint="SHA256:" + ("A" * 43),
                    checked_at=checked_at,
                ),
            }
        )
    ]
    providers = [
        ProjectProvisioningProviderCheckRecord(
            **_provider().model_dump(mode="json"),
            status="ready",
            checked_at=checked_at,
        )
    ]
    ready = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="preparation-ready",
        phase="final_review",
        expected_revision=3,
        expected_status="setup_in_progress",
        to_status="ready_for_review",
        machines=machines,
        repositories=repositories,
        provider_checks=providers,
    )
    assert ready.final_review_digest == project_provisioning_review_digest(ready)
    assert ready.ready_at is not None
    assert store.project(ready.proposed_project_id) is None

    completed = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="human-review-completed",
        phase="final_review",
        expected_revision=4,
        expected_status="ready_for_review",
        to_status="completed",
        machines=ready.machines,
        repositories=ready.repositories,
        provider_checks=ready.provider_checks,
    )
    assert completed.final_review_digest == ready.final_review_digest
    assert completed.completed_at is not None
    assert store.project(completed.proposed_project_id) is None
    assert [
        receipt.resulting_revision
        for receipt in store.project_provisioning_step_receipts(request.request_id)
    ] == [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="cannot move"):
        store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="illegal-terminal-reentry",
            phase="setup_start",
            expected_revision=5,
            expected_status="completed",
            to_status="setup_in_progress",
            machines=completed.machines,
            repositories=completed.repositories,
            provider_checks=completed.provider_checks,
        )

    with sqlite3.connect(store.path) as connection:
        persisted_repositories = json.loads(
            connection.execute(
                "SELECT repositories_json FROM project_provisioning_requests"
            ).fetchone()[0]
        )
        persisted_repositories[0]["git_check"]["commit"] = "b" * 40
        connection.execute(
            "UPDATE project_provisioning_requests SET repositories_json = ?",
            (json.dumps(persisted_repositories),),
        )
    with pytest.raises(RuntimeError, match="stored project provisioning request is invalid"):
        store.project_provisioning_request(request.request_id)


@pytest.mark.parametrize("terminal_status", ["ready_for_review", "completed"])
def test_pre_configuration_review_digest_remains_readable_after_upgrade(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _create_legacy(store, authorizer)
    running = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="legacy-setup-started",
        phase="setup_start",
        expected_revision=0,
        expected_status="waiting_for_server_setup",
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )
    checked_at = store.now()
    machines = [
        running.machines[0].model_copy(
            update={"resolved_central_root": running.machines[0].central_root}
        )
    ]
    repositories = [
        running.repositories[0].model_copy(
            update={
                "resolved_path": running.repositories[0].intended_path,
                "git_check": ProjectProvisioningGitCheckRecord(
                    status="ready",
                    commit="a" * 40,
                    write_verified=True,
                    deploy_key_label=(f"rcp:{store.space_id}:{request.proposed_project_id}:paper"),
                    public_key_fingerprint="SHA256:" + ("A" * 43),
                    checked_at=checked_at,
                ),
            }
        )
    ]
    providers = [
        ProjectProvisioningProviderCheckRecord(
            **_provider().model_dump(mode="json"),
            status="ready",
            checked_at=checked_at,
        )
    ]
    terminal = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="legacy-ready",
        phase="final_review",
        expected_revision=1,
        expected_status="setup_in_progress",
        to_status="ready_for_review",
        machines=machines,
        repositories=repositories,
        provider_checks=providers,
    )
    if terminal_status == "completed":
        terminal = store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="legacy-completed",
            phase="final_review",
            expected_revision=2,
            expected_status="ready_for_review",
            to_status="completed",
            machines=terminal.machines,
            repositories=terminal.repositories,
            provider_checks=terminal.provider_checks,
        )
    legacy_payload = {
        "request_id": terminal.request_id,
        "kind": terminal.kind,
        "target_space_id": terminal.target_space_id,
        "authorized_by": terminal.authorized_by.model_dump(mode="json"),
        "proposed_project_id": terminal.proposed_project_id,
        "machines": [machine.model_dump(mode="json") for machine in terminal.machines],
        "repositories": [
            repository.model_dump(mode="json", exclude={"checkout_disposition"})
            for repository in terminal.repositories
        ],
        "provider_checks": [
            provider.model_dump(mode="json") for provider in terminal.provider_checks
        ],
    }
    legacy_digest = hashlib.sha256(
        json.dumps(
            legacy_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with sqlite3.connect(store.path) as connection:
        project_config = connection.execute(
            "SELECT project_config_json FROM project_provisioning_requests WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()[0]
        assert project_config is None
        connection.execute(
            "UPDATE project_provisioning_requests SET final_review_digest = ? WHERE request_id = ?",
            (legacy_digest, request.request_id),
        )

    loaded = store.project_provisioning_request(request.request_id)
    assert loaded.status == terminal_status
    assert loaded.final_review_digest == legacy_digest
    assert loaded.configuration_complete is False
    assert loaded.repositories[0].checkout_disposition is None


def test_operator_action_is_bound_to_the_request_and_declared_target(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _create(store, authorizer)
    running = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="setup-started",
        phase="setup_start",
        expected_revision=0,
        expected_status="waiting_for_server_setup",
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )
    action = _operator_action(request.request_id)
    unrelated_target = ExternalServiceTarget(
        service="github.com",
        resource="openai/another-repository",
        destination_url="https://github.com/openai/another-repository/settings/keys",
        required_authority_role="repository administrator",
    )

    with pytest.raises(ValidationError, match="declared GitHub repository"):
        store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="unrelated-target",
            phase="github_grant",
            expected_revision=1,
            expected_status="setup_in_progress",
            to_status="operator_action_needed",
            machines=running.machines,
            repositories=running.repositories,
            provider_checks=running.provider_checks,
            operator_action=action.model_copy(update={"target": unrelated_target}),
        )

    wrong_request_id = str(uuid.uuid4())
    with pytest.raises(ValidationError, match="resume this exact request"):
        store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="unrelated-resume",
            phase="github_grant",
            expected_revision=1,
            expected_status="setup_in_progress",
            to_status="operator_action_needed",
            machines=running.machines,
            repositories=running.repositories,
            provider_checks=running.provider_checks,
            operator_action=action.model_copy(
                update={
                    "resume_argv": (
                        "rcp",
                        "server",
                        "project",
                        "provision",
                        wrong_request_id,
                    )
                }
            ),
        )

    unchanged = store.project_provisioning_request(request.request_id)
    assert unchanged == running
    assert [
        receipt.receipt_id
        for receipt in store.project_provisioning_step_receipts(request.request_id)
    ] == ["setup-started"]


def test_stale_transition_loses_without_writing_a_receipt(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _create(store, authorizer)
    store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="winner",
        phase="setup_start",
        expected_revision=0,
        expected_status="waiting_for_server_setup",
        to_status="setup_in_progress",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
    )

    with pytest.raises(ValueError, match="changed; reload"):
        store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="stale-loser",
            phase="setup_start",
            expected_revision=0,
            expected_status="waiting_for_server_setup",
            to_status="setup_in_progress",
            machines=request.machines,
            repositories=request.repositories,
            provider_checks=request.provider_checks,
        )
    assert [
        receipt.receipt_id
        for receipt in store.project_provisioning_step_receipts(request.request_id)
    ] == ["winner"]


def test_cancellation_requires_an_explicit_disposition_and_remains_inert(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _create(store, authorizer)

    with pytest.raises(ValidationError, match="explicit disposition"):
        store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="cancel-without-disposition",
            phase="cancel",
            expected_revision=0,
            expected_status="waiting_for_server_setup",
            to_status="cancelled",
            machines=request.machines,
            repositories=request.repositories,
            provider_checks=request.provider_checks,
        )
    cancelled = store.transition_project_provisioning_request(
        request.request_id,
        receipt_id="cancelled-cleanly",
        phase="cancel",
        expected_revision=0,
        expected_status="waiting_for_server_setup",
        to_status="cancelled",
        machines=request.machines,
        repositories=request.repositories,
        provider_checks=request.provider_checks,
        cancellation_disposition="nothing_to_remove",
    )

    assert cancelled.status == "cancelled"
    assert cancelled.cancellation_disposition == "nothing_to_remove"
    assert cancelled.cancelled_at is not None
    assert store.project(cancelled.proposed_project_id) is None


def test_secret_shaped_provider_and_path_values_never_enter_a_request(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)

    with pytest.raises(ValidationError, match="credential-shaped"):
        ProjectProvisioningProviderIntent(
            profile="seed",
            provider="codex",
            runtime_id="github_pat_abcdefghijklmnop",
            model="gpt-5.6-luna",
            reasoning="medium",
            machine_alias="server",
        )
    with pytest.raises(ValidationError, match="credential-shaped"):
        ProjectProvisioningMachineIntent(
            alias="remote",
            location="ssh",
            host="cluster",
            os_account="alice",
            central_root="/srv/github_pat_abcdefghijklmnop/projects",
        )
    with pytest.raises(ValidationError, match="credential-shaped"):
        ProjectProvisioningGitCheckRecord(
            deploy_key_label="github_pat_abcdefghijklmnop",
        )
    with pytest.raises(ValidationError, match="safe line"):
        ProjectProvisioningGitCheckRecord(deploy_key_label="rcp:label\nsecond-line")
    assert store.project_provisioning_requests() == []


def test_new_project_configuration_is_complete_safe_and_repository_bound(
    tmp_path: Path,
) -> None:
    store, authorizer = _team_store(tmp_path)
    base = {
        "kind": "create_team_project",
        "authorized_by": authorizer,
        "machines": [_machine()],
        "repositories": [_repository()],
        "provider_checks": [_provider()],
    }

    with pytest.raises(ValidationError, match="credential-shaped"):
        store.create_project_provisioning_request(
            **base,
            name="github_pat_abcdefghijklmnop",
            state_repository="paper",
            project_truth_scope=["paper"],
            default_run_truth_scope=["paper"],
        )
    with pytest.raises(ValidationError, match="must be complete"):
        store.create_project_provisioning_request(
            **base,
            default_auto_research_invocation_ceiling=11,
        )
    with pytest.raises(ValidationError, match="state repository must name"):
        store.create_project_provisioning_request(
            **base,
            name="Shared paper project",
            state_repository="code",
            project_truth_scope=["paper"],
            default_run_truth_scope=["paper"],
        )
    with pytest.raises(ValidationError, match="default run truth scope"):
        store.create_project_provisioning_request(
            **base,
            name="Shared paper project",
            state_repository="paper",
            project_truth_scope=["paper"],
            default_run_truth_scope=["code"],
        )

    assert store.project_provisioning_requests() == []


def test_raw_rows_are_revalidated_instead_of_becoming_authority(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _create(store, authorizer)
    with sqlite3.connect(store.path) as connection:
        repositories = json.loads(
            connection.execute(
                "SELECT repositories_json FROM project_provisioning_requests"
            ).fetchone()[0]
        )
        repositories[0]["repository"]["identity"] = "OpenAI/RCP"
        connection.execute(
            "UPDATE project_provisioning_requests SET repositories_json = ?",
            (json.dumps(repositories),),
        )

    with pytest.raises(RuntimeError, match="stored project provisioning request is invalid"):
        store.project_provisioning_request(request.request_id)


def test_project_configuration_json_cannot_shadow_request_columns(tmp_path: Path) -> None:
    store, authorizer = _team_store(tmp_path)
    request = _create(store, authorizer)
    with sqlite3.connect(store.path) as connection:
        project_config = json.loads(
            connection.execute(
                "SELECT project_config_json FROM project_provisioning_requests"
            ).fetchone()[0]
        )
        project_config["status"] = "completed"
        connection.execute(
            "UPDATE project_provisioning_requests SET project_config_json = ?",
            (json.dumps(project_config),),
        )

    with pytest.raises(RuntimeError, match="stored project provisioning request is invalid"):
        store.project_provisioning_request(request.request_id)
