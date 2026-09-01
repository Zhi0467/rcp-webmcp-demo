from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from rcp.agents import AgentEvent, ProviderReadiness
from rcp.agents.command_protocol import SpawnArguments
from rcp.api.app import _auto_research_worker_request
from rcp.config import write_agent_settings
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.runs.auto_research import AutoResearchCommandContext, AutoResearchRunRequest
from rcp.storage import AgentTaskRecord, EpisodeRecord

from .helpers import create_named_app, wait_for_task
from .test_episode_api import settling_auto_research_stream

_EXECUTION_PROFILES = (
    "seed",
    "refresh",
    "node_chat",
    "project_chat",
    "paper_coach",
    "orchestrator",
)


def _distinct_orchestrator_profile(manifest) -> None:
    profiles = {
        name: manifest.agent_profile(name).model_copy(deep=True) for name in _EXECUTION_PROFILES
    }
    profiles["project_chat"] = profiles["project_chat"].model_copy(
        update={
            "provider": "claude",
            "runtime": "stream-json",
            "model": "chat-only",
            "reasoning": "low",
        }
    )
    profiles["orchestrator"] = profiles["orchestrator"].model_copy(
        update={"provider": "codex", "model": "orchestrator-only", "reasoning": "high"}
    )
    write_agent_settings(
        manifest,
        list(manifest.agent.default_run_truth_scope),
        profiles,
    )


def test_episode_start_uses_the_dedicated_orchestrator_profile(manifest, tmp_path) -> None:
    _distinct_orchestrator_profile(manifest)
    app = create_named_app(
        str(manifest.path),
        data_dir=tmp_path / "data",
    )
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    stage = tmp_path / "auto-stage"
    stage.mkdir()
    app.state.background_tasks.stream = settling_auto_research_stream(stage)

    with TestClient(app) as client:
        response = client.post(
            f"/api/projects/{project_id}/episodes",
            json={
                "mode": "auto_research",
                "invocation_ceiling": 5,
                "starting_instruction": "Investigate the unresolved evidence.",
            },
        )

        assert response.status_code == 202
        payload = response.json()
        episode_id = payload["episode_id"]
        operation_id = payload["root_operation_id"]
        assert payload["mode"] == "auto_research"
        assert payload["budget"] == {
            "invocation_ceiling": 5,
            "invocations_used": 1,
            "invocations_remaining": 4,
            "observed_input_tokens": 0,
            "observed_generated_tokens": 0,
        }
        assert "campaign_id" not in payload
        assert "reports" not in payload

        root = store.agent_task(operation_id)
        assert root is not None
        request = AutoResearchRunRequest.model_validate(root.request)
        assert request.episode_id == episode_id
        assert request.role == "orchestrator"
        assert request.actor_operation_id == operation_id
        assert request.provider == "codex"
        assert request.model == "orchestrator-only"
        assert request.reasoning == "high"
        assert request.run_on == "laptop"
        assert request.run_truth_scope == ["repo-a"]
        assert request.instruction == "Investigate the unresolved evidence."
        assert root.kind == "auto_research"
        assert root.episode_id == episode_id
        assert root.dispatch_authority is not None
        assert root.dispatch_authority.profile == "orchestrator"
        assert root.dispatch_authority.task_contract == "orchestrate"
        assert root.dispatch_authority.scope.episode_id == episode_id
        assert root.dispatch_authority.scope.run_truth_scope == ["repo-a"]

        settled = wait_for_task(store, operation_id, expect="succeeded")
        assert settled.error is None
        meter = store.episode_budget_meter(episode_id)
        assert meter.observed_input_tokens == 0
        assert meter.observed_generated_tokens == 0

        duplicate = client.post(
            f"/api/projects/{project_id}/episodes",
            json={"mode": "auto_research", "invocation_ceiling": 5},
        )
        assert duplicate.status_code == 409
        assert [item.episode_id for item in store.episodes(project_id)] == [episode_id]
        assert [item.operation_id for item in store.episode_tasks(episode_id)] == [operation_id]


def test_episode_profile_resolution_failure_is_pre_mutation(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    service = app.state.catalog.open(project_id)

    def reject_profile(_surface):
        raise ValueError("orchestrator profile is not launchable")

    monkeypatch.setattr(service, "resolve_agent_profile", reject_profile)
    with TestClient(app) as client:
        response = client.post(
            f"/api/projects/{project_id}/episodes",
            json={"mode": "auto_research", "invocation_ceiling": 5},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "orchestrator profile is not launchable"}
    assert store.episodes(project_id) == []
    assert store.agent_tasks(project_id) == []


def test_auto_research_retry_rechecks_remote_target_before_creating_child(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    host = "gpu.example.edu"
    binary = "/home/researcher/.nvm/versions/node/v18.20.8/bin/codex"
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    tasks = app.state.background_tasks
    store = tasks.store
    service = app.state.catalog.open(project_id)
    machine = service.manifest.machine_map["laptop"]
    machine.host = host
    machine.provider_paths["codex"] = binary
    monkeypatch.setattr(service.history, "_reload_manifest", lambda: None)

    async def failing_stream(_project_id, _kind, _request, _execution):
        yield f"data: {AgentEvent(event='error', text='Host unreachable.').model_dump_json()}\n\n"

    tasks.stream = failing_stream
    client = TestClient(app)
    started = client.post(
        f"/api/projects/{project_id}/episodes",
        json={"mode": "auto_research", "invocation_ceiling": 3},
    )
    assert started.status_code == 202
    episode_id = started.json()["episode_id"]
    operation_id = started.json()["root_operation_id"]
    failed = wait_for_task(store, operation_id, expect="failed")
    assert failed.can_retry is True
    before = [task.operation_id for task in store.episode_tasks(episode_id, include_hidden=True)]

    readiness_calls: list[tuple[str, str, str | None, bool]] = []

    def unreachable_readiness(
        provider: str,
        *,
        host: str = "",
        binary: str | None = None,
        refresh: bool = False,
    ) -> ProviderReadiness:
        readiness_calls.append((provider, host, binary, refresh))
        return ProviderReadiness(
            provider=provider,
            installed=False,
            authenticated=False,
            binary_path=binary,
            path_state="unreachable",
            reason=f"{host} is unreachable, so {binary} could not be checked.",
        )

    monkeypatch.setattr(service.launcher, "readiness", unreachable_readiness)
    blocked = client.post(
        f"/api/projects/{project_id}/tasks/{operation_id}/retry",
        json={},
    )

    assert blocked.status_code == 409
    assert blocked.json() == {
        "detail": (
            "Auto-research Retry cannot start: gpu.example.edu is unreachable, "
            "so /home/researcher/.nvm/versions/node/v18.20.8/bin/codex could not be checked. "
            "The current task was left unchanged."
        )
    }
    assert readiness_calls == [
        (
            "codex",
            "gpu.example.edu",
            "/home/researcher/.nvm/versions/node/v18.20.8/bin/codex",
            True,
        )
    ]
    assert [
        task.operation_id for task in store.episode_tasks(episode_id, include_hidden=True)
    ] == before
    assert store.episode_budget_meter(episode_id).invocations_used == 1

    claude_binary = "/opt/claude/bin/claude"
    machine.provider_paths["claude"] = claude_binary
    readiness_calls.clear()
    switched = client.post(
        f"/api/projects/{project_id}/tasks/{operation_id}/retry",
        json={"provider": "claude"},
    )
    assert switched.status_code == 409
    assert readiness_calls == [("claude", host, claude_binary, True)]
    assert [
        task.operation_id for task in store.episode_tasks(episode_id, include_hidden=True)
    ] == before

    stage = tmp_path / "recovered-stage"
    stage.mkdir()
    tasks.stream = settling_auto_research_stream(stage)

    def ready(
        provider: str,
        *,
        host: str = "",
        binary: str | None = None,
        refresh: bool = False,
    ) -> ProviderReadiness:
        readiness_calls.append((provider, host, binary, refresh))
        return ProviderReadiness(
            provider=provider,
            installed=True,
            authenticated=True,
            binary_path=binary,
            path_state="resolved",
        )

    monkeypatch.setattr(service.launcher, "readiness", ready)
    readiness_calls.clear()
    accepted = client.post(
        f"/api/projects/{project_id}/tasks/{operation_id}/retry",
        json={},
    )

    assert accepted.status_code == 202
    retried = wait_for_task(store, accepted.json()["operation_id"], expect="succeeded")
    assert retried.parent_operation_id == operation_id
    assert retried.episode_id == episode_id
    assert store.episode_budget_meter(episode_id).invocations_used == 1


def test_worker_request_uses_the_current_human_node_work_profile(manifest, tmp_path) -> None:
    profiles = {
        name: manifest.agent_profile(name).model_copy(deep=True) for name in _EXECUTION_PROFILES
    }
    profiles["node_chat"] = profiles["node_chat"].model_copy(
        update={
            "provider": "claude",
            "runtime": "stream-json",
            "model": "current-worker",
            "reasoning": "low",
        }
    )
    profiles["orchestrator"] = profiles["orchestrator"].model_copy(
        update={"provider": "codex", "model": "pinned-orchestrator", "reasoning": "high"}
    )
    write_agent_settings(
        manifest,
        list(manifest.agent.default_run_truth_scope),
        profiles,
    )
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    episode_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())
    worker_id = str(uuid.uuid4())
    now = "2026-08-12T00:00:00+00:00"
    request = AutoResearchRunRequest(
        episode_id=episode_id,
        role="orchestrator",
        actor_operation_id=operation_id,
        provider="codex",
        model="pinned-model",
        reasoning="high",
        run_on="canonical-machine",
        run_truth_scope=["repo-a", "repo-b"],
        session_id="orchestrator-session",
        instruction="Root instruction",
        workflow_ids=["workflow-a"],
        skill_ids=["skill-a"],
        invoked_workflow_ids=["workflow-a"],
        invoked_skill_ids=["skill-a"],
        invoked_provider_skill_names=["provider-skill"],
        resolved_provider_skills=[
            {
                "provider": "codex",
                "machine": "canonical-machine",
                "provider_version": "1",
                "inventory_hash": "inventory",
                "name": "provider-skill",
                "label": "Provider skill",
                "description": "Pinned provider package.",
            }
        ],
        resolved_skill_packages=[{"id": "skill-a", "kind": "skill", "version": "1.2.3"}],
    )
    authorizer = AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name="Researcher",
    )
    graph_target = GraphTargetRef(kind="branch", branch_id=episode_id)
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        root_operation_id=operation_id,
        mode="auto_research",
        graph_target=graph_target,
        graph_base_head=GraphHeadRef(revision=0),
        status="running",
        invocation_ceiling=5,
        invocations_used=1,
        authorized_by=authorizer,
        created_at=now,
        updated_at=now,
    )
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        episode_id=episode_id,
        graph_target=graph_target,
        kind="auto_research",
        status="running",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="running",
        authorized_by=authorizer,
    )

    worker = _auto_research_worker_request(
        service,
        AutoResearchCommandContext(episode=episode, task=task, request=request),
        SpawnArguments(seat_node_id="blocker-1", instruction_file="worker-task.md"),
        "Resolve the blocker.",
        worker_id,
    )

    assert (worker.provider, worker.model, worker.reasoning) == (
        "claude",
        "current-worker",
        "low",
    )
    assert worker.run_on == profiles["node_chat"].run_on
    assert worker.run_truth_scope == list(manifest.agent.default_run_truth_scope)
    assert worker.chat_id == worker_id
    assert worker.chat_scope == "node"
    assert worker.node_id == "blocker-1"
    assert worker.message == "Resolve the blocker."
    assert worker.mode == "work"
    assert worker.trigger == "orchestrator"
    assert worker.patch_kind == "work"
    assert worker.session_id is None
    assert worker.watcher_ids == []
    assert worker.model != request.model
