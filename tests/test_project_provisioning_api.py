from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rcp.api import create_app
from rcp.config import AGENT_EXECUTION_PROFILES
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.server_ops.models import ExternalAction, ExternalServiceTarget, ServerStep
from rcp.storage import (
    AppStore,
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningProviderCheckRecord,
)

from .helpers import create_named_app


def _payload(*, source: str = "https://github.com/OpenAI/RCP.git") -> dict[str, object]:
    return {
        "name": "Shared paper project",
        "state_repository": "paper",
        "project_truth_scope": ["paper"],
        "default_run_truth_scope": ["paper"],
        "default_auto_research_invocation_ceiling": 10,
        "machines": [
            {
                "alias": "server",
                "location": "local",
                "host": "",
                "os_account": "rcp",
            }
        ],
        "repositories": [
            {
                "alias": "paper",
                "source": source,
                "machine_alias": "server",
            }
        ],
        "provider_checks": [
            {
                "profile": profile,
                "provider": "codex",
                "runtime_id": "codex:exec",
                "model": "gpt-5.6-luna",
                "reasoning": "medium",
                "machine_alias": "server",
            }
            for profile in AGENT_EXECUTION_PROFILES
        ],
    }


def _team_app(tmp_path: Path, *names: str):
    data_dir = tmp_path / "team"
    store, _bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Team Lab")
    members = [store.preprovision_team_member(name) for name in names]
    selected = [members[0].user_id]
    app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, opened: opened.space_user(selected[0]),
    )
    return data_dir, members, selected, app


def _operator_action(request_id: str) -> ServerStep:
    message = "Add this public key and enable Allow write access."
    return ServerStep(
        number=1,
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
        message=message,
        actions=(ExternalAction(instruction=message),),
        resume_argv=(
            str(DEFAULT_SERVER_LAYOUT.cli_wrapper),
            "server",
            "project",
            "provision",
            request_id,
        ),
    )


def test_member_creates_restart_reads_and_authorizer_cancels_inert_request(tmp_path) -> None:
    data_dir, (alice, bob), selected, app = _team_app(tmp_path, "Alice", "Bob")
    with TestClient(app) as client:
        created_response = client.post("/api/project-provisioning/requests", json=_payload())

    assert created_response.status_code == 201
    created = created_response.json()
    assert created["kind"] == "create_team_project"
    assert created["status"] == "waiting_for_server_setup"
    assert created["status_label"] == "Waiting for server setup"
    assert created["next_action"] == "Run server setup."
    assert created["can_run_setup"] is True
    assert created["can_review"] is False
    assert created["can_cancel"] is True
    assert created["name"] == "Shared paper project"
    assert created["state_repository"] == "paper"
    assert created["project_truth_scope"] == ["paper"]
    assert created["default_run_truth_scope"] == ["paper"]
    assert created["default_auto_research_invocation_ceiling"] == 10
    assert created["authorized_by"] == {
        "space_id": app.state.space_id,
        "user_id": alice.user_id,
        "display_name": "Alice",
    }
    assert created["machines"] == [
        {
            "alias": "server",
            "location": "local",
            "host": "",
            "os_account": "rcp",
            "intended_central_root": str(DEFAULT_SERVER_LAYOUT.projects_root),
            "resolved_central_root": None,
            "ready": False,
            "status_label": "Waiting for setup",
        }
    ]
    repository = created["repositories"][0]
    assert repository["repository"] == {"identity": "openai/rcp"}
    assert repository["https_clone_url"] == "https://github.com/openai/rcp.git"
    assert repository["ssh_clone_url"] == "git@github.com:openai/rcp.git"
    assert repository["settings_url"] == "https://github.com/openai/rcp/settings/keys"
    assert repository["intended_path"] == str(
        DEFAULT_SERVER_LAYOUT.projects_root
        / created["proposed_project_id"]
        / "repositories"
        / "paper"
    )
    assert repository["checkout_disposition"] is None
    assert created["readiness"] == {
        "machines_ready": 0,
        "machines_total": 1,
        "repositories_ready": 0,
        "repositories_total": 1,
        "providers_ready": 0,
        "providers_total": len(AGENT_EXECUTION_PROFILES),
        "all_ready": False,
    }
    assert created["operator_argv"] == [
        str(DEFAULT_SERVER_LAYOUT.cli_wrapper),
        "server",
        "project",
        "provision",
        created["request_id"],
    ]
    assert app.state.background_tasks.store.project(created["proposed_project_id"]) is None
    assert app.state.catalog.cards() == []

    selected[:] = [bob.user_id]
    restarted = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, opened: opened.space_user(selected[0]),
    )
    with TestClient(restarted) as client:
        listed = client.get("/api/project-provisioning/requests")
        read = client.get(f"/api/project-provisioning/requests/{created['request_id']}")
        refused = client.post(
            f"/api/project-provisioning/requests/{created['request_id']}/cancel",
            json={},
        )
        selected[:] = [alice.user_id]
        cancelled = client.post(
            f"/api/project-provisioning/requests/{created['request_id']}/cancel",
            json={},
        )
        repeated = client.post(
            f"/api/project-provisioning/requests/{created['request_id']}/cancel",
            json={},
        )

    assert listed.status_code == read.status_code == 200
    assert [item["request_id"] for item in listed.json()] == [created["request_id"]]
    assert read.json()["can_cancel"] is False
    assert refused.status_code == 403
    assert cancelled.status_code == repeated.status_code == 200
    assert cancelled.json() == repeated.json()
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["status_label"] == "Cancelled"
    assert cancelled.json()["next_action"] is None
    assert cancelled.json()["can_run_setup"] is False
    assert cancelled.json()["can_cancel"] is False
    assert cancelled.json()["cancellation_disposition"] == "nothing_to_remove"
    assert restarted.state.background_tasks.store.project(created["proposed_project_id"]) is None
    assert restarted.state.catalog.cards() == []


def test_ssh_machine_can_defer_its_default_root_to_exact_account_resolution(tmp_path) -> None:
    _data_dir, (_alice,), _selected, app = _team_app(tmp_path, "Alice")
    payload = _payload()
    payload["machines"] = [
        {
            "alias": "gpu",
            "location": "ssh",
            "host": "alice@gpu-lab",
            "os_account": "alice",
        }
    ]
    repositories = payload["repositories"]
    providers = payload["provider_checks"]
    assert isinstance(repositories, list) and isinstance(repositories[0], dict)
    assert isinstance(providers, list) and isinstance(providers[0], dict)
    repositories[0]["machine_alias"] = "gpu"
    for provider in providers:
        assert isinstance(provider, dict)
        provider["machine_alias"] = "gpu"

    with TestClient(app) as client:
        response = client.post("/api/project-provisioning/requests", json=payload)

    assert response.status_code == 201
    created = response.json()
    assert created["machines"][0]["intended_central_root"] is None
    assert created["machines"][0]["resolved_central_root"] is None
    assert created["repositories"][0]["intended_path"] is None
    assert created["repositories"][0]["resolved_path"] is None


def test_invalid_repository_is_rejected_before_persistence(tmp_path, monkeypatch) -> None:
    _data_dir, _members, _selected, app = _team_app(tmp_path, "Alice")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("invalid repository reached durable request creation")

    monkeypatch.setattr(
        app.state.background_tasks.store,
        "create_project_provisioning_request",
        fail_if_called,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/project-provisioning/requests",
            json=_payload(source="https://example.com/openai/rcp.git"),
        )

    assert response.status_code == 422
    assert app.state.background_tasks.store.project_provisioning_requests() == []


def test_new_team_request_requires_all_profiles_on_their_valid_machines(tmp_path) -> None:
    _data_dir, _members, _selected, app = _team_app(tmp_path, "Alice")
    incomplete = _payload()
    checks = incomplete["provider_checks"]
    assert isinstance(checks, list)
    incomplete["provider_checks"] = checks[:-1]
    misplaced = _payload()
    machines = misplaced["machines"]
    misplaced_checks = misplaced["provider_checks"]
    assert isinstance(machines, list) and isinstance(misplaced_checks, list)
    machines.append(
        {
            "alias": "worker",
            "location": "ssh",
            "host": "alice@gpu-lab",
            "os_account": "alice",
        }
    )
    assert isinstance(misplaced_checks[0], dict)
    misplaced_checks[0]["machine_alias"] = "worker"

    with TestClient(app) as client:
        incomplete_response = client.post(
            "/api/project-provisioning/requests",
            json=incomplete,
        )
        misplaced_response = client.post(
            "/api/project-provisioning/requests",
            json=misplaced,
        )

    assert incomplete_response.status_code == 422
    assert "every agent execution profile" in incomplete_response.text
    assert misplaced_response.status_code == 422
    assert "canonical state machine" in misplaced_response.text
    assert app.state.background_tasks.store.project_provisioning_requests() == []


def test_project_provisioning_requires_a_named_authenticated_team_member(tmp_path) -> None:
    personal = create_named_app(data_dir=tmp_path / "personal")
    with TestClient(personal) as client:
        assert client.post("/api/project-provisioning/requests", json=_payload()).status_code == 404

    data_dir = tmp_path / "team-unnamed"
    store, _bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Team Lab")
    unnamed = store.preprovision_team_member()
    app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, opened: opened.space_user(unnamed.user_id),
    )
    with TestClient(app) as client:
        unnamed_response = client.post("/api/project-provisioning/requests", json=_payload())
    assert unnamed_response.status_code == 428
    assert unnamed_response.json()["detail"]["code"] == "identity_name_required"

    unauthenticated_dir = tmp_path / "team-unauthenticated"
    AppStore.initialize_team_space(unauthenticated_dir / "rcp.sqlite3", "Team Lab")
    unauthenticated = create_app(data_dir=unauthenticated_dir)
    with TestClient(unauthenticated) as client:
        response = client.post("/api/project-provisioning/requests", json=_payload())
    assert response.status_code == 401


def test_started_and_operator_action_requests_publish_backend_controls(tmp_path) -> None:
    _data_dir, _members, _selected, app = _team_app(tmp_path, "Alice")
    with TestClient(app) as client:
        created = client.post("/api/project-provisioning/requests", json=_payload()).json()
        store = app.state.background_tasks.store
        record = store.project_provisioning_request(created["request_id"])
        assert record is not None
        running = store.transition_project_provisioning_request(
            record.request_id,
            receipt_id="setup-started",
            phase="setup_start",
            expected_revision=0,
            expected_status="waiting_for_server_setup",
            to_status="setup_in_progress",
            machines=record.machines,
            repositories=record.repositories,
            provider_checks=record.provider_checks,
        )
        running_response = client.get(f"/api/project-provisioning/requests/{record.request_id}")
        response = client.post(
            f"/api/project-provisioning/requests/{record.request_id}/cancel",
            json={},
        )
        action = _operator_action(record.request_id)
        store.transition_project_provisioning_request(
            record.request_id,
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
        action_response = client.get(f"/api/project-provisioning/requests/{record.request_id}")

    assert running_response.status_code == 200
    running_projection = running_response.json()
    assert running_projection["status"] == "setup_in_progress"
    assert running_projection["status_label"] == "Setup in progress"
    assert running_projection["next_action"] == (
        "Wait for server setup, or resume the same command after an interruption."
    )
    assert running_projection["can_run_setup"] is True
    assert running_projection["can_review"] is False
    assert running_projection["can_cancel"] is False
    assert response.status_code == 409
    assert "cleanup or reuse disposition" in response.json()["detail"]
    assert action_response.status_code == 200
    action_projection = action_response.json()
    assert action_projection["status"] == "operator_action_needed"
    assert action_projection["status_label"] == "Operator action needed"
    assert action_projection["next_action"] == action.message
    assert action_projection["can_run_setup"] is True
    assert action_projection["can_review"] is False
    assert action_projection["can_cancel"] is False
    assert action_projection["diagnostic"] == "The write grant is not ready yet."
    assert action_projection["operator_action"] == action.model_dump(mode="json")


def test_final_review_projection_contains_only_backend_decisions(tmp_path) -> None:
    _data_dir, _members, _selected, app = _team_app(tmp_path, "Alice")
    with TestClient(app) as client:
        created = client.post("/api/project-provisioning/requests", json=_payload()).json()
        store = app.state.background_tasks.store
        request = store.project_provisioning_request(created["request_id"])
        assert request is not None
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
                    "checkout_disposition": "request_created",
                    "git_check": ProjectProvisioningGitCheckRecord(
                        status="ready",
                        commit="a" * 40,
                        write_verified=True,
                        deploy_key_label=(
                            f"rcp:{store.space_id}:{request.proposed_project_id}:paper"
                        ),
                        public_key_fingerprint="SHA256:" + ("A" * 43),
                        checked_at=checked_at,
                    ),
                }
            )
        ]
        providers = [
            ProjectProvisioningProviderCheckRecord(
                **check.model_dump(
                    mode="json",
                    exclude={"status", "checked_at", "diagnostic"},
                ),
                status="ready",
                checked_at=checked_at,
            )
            for check in running.provider_checks
        ]
        ready = store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="preparation-ready",
            phase="final_review",
            expected_revision=1,
            expected_status="setup_in_progress",
            to_status="ready_for_review",
            machines=machines,
            repositories=repositories,
            provider_checks=providers,
        )
        response = client.get(f"/api/project-provisioning/requests/{request.request_id}")
        completed = store.transition_project_provisioning_request(
            request.request_id,
            receipt_id="human-review-completed",
            phase="final_review",
            expected_revision=2,
            expected_status="ready_for_review",
            to_status="completed",
            machines=ready.machines,
            repositories=ready.repositories,
            provider_checks=ready.provider_checks,
        )
        completed_response = client.get(f"/api/project-provisioning/requests/{request.request_id}")

    assert response.status_code == 200
    projected = response.json()
    assert projected["status"] == "ready_for_review"
    assert projected["status_label"] == "Ready for review"
    assert projected["next_action"] == "Review the prepared project."
    assert projected["can_run_setup"] is False
    assert projected["can_review"] is True
    assert projected["can_cancel"] is False
    assert projected["readiness"]["all_ready"] is True
    assert projected["machines"][0]["ready"] is True
    assert projected["repositories"][0]["ready"] is True
    assert projected["repositories"][0]["checkout_disposition"] == "request_created"
    assert projected["provider_checks"][0]["ready"] is True
    assert projected["final_review"] == {
        "digest": ready.final_review_digest,
        "proposed_project_id": ready.proposed_project_id,
        "authorized_by": ready.authorized_by.model_dump(mode="json"),
        "ready_at": ready.ready_at,
    }
    assert app.state.background_tasks.store.project(ready.proposed_project_id) is None
    assert app.state.catalog.cards() == []
    assert completed_response.status_code == 200
    completed_projection = completed_response.json()
    assert completed_projection["status"] == "completed"
    assert completed_projection["status_label"] == "Completed"
    assert completed_projection["next_action"] is None
    assert completed_projection["can_run_setup"] is False
    assert completed_projection["can_review"] is False
    assert completed_projection["can_cancel"] is False
    assert completed_projection["final_review"] == projected["final_review"]
    assert completed_projection["completed_at"] == completed.completed_at
