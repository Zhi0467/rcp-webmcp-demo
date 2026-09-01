from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rcp.api.app as api_app_module
from rcp.agents import AgentEvent
from rcp.api.app import create_app as create_raw_app
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import GraphBranchMetadata, Patch
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.storage import (
    AgentTaskRecord,
    AutoResearchChildExperimentRecord,
    AutoResearchStateRecord,
    EpisodeNotRunning,
    EpisodeRecord,
    ProjectRecord,
)

from .helpers import append_fixture_patch, authorized_human, seed_patch, wait_for_task
from .helpers import create_named_app as create_app


def _experiment_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added experiments for the landing-page index.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "exp/launched",
                        "type": "experiment",
                        "title": "Launched loop",
                        "objective": "Exercise the cross-project loop index.",
                        "completion_criteria": ["The indexed loop reaches a conclusion."],
                        "invocation_ceiling": 3,
                    },
                    {
                        "id": "exp/never-run",
                        "type": "experiment",
                        "title": "Never-run experiment",
                        "objective": "Remain absent from the loop index.",
                        "invocation_ceiling": 2,
                    },
                ],
            }
        ],
    )


def _update_experiment_summary(summary: str) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Updated the indexed experiment.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/launched",
                        "changes": {"current_summary": summary},
                    }
                ],
            }
        ],
    )


def _update_primary_question(question: str, *, base_updated_rev: int) -> Patch:
    return Patch(
        kind="approval",
        author="human",
        summary="Updated the primary question.",
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "rq/learning-after-shift",
                        "base_updated_rev": base_updated_rev,
                        "changes": {"question": question},
                    }
                ],
            }
        ],
    )


def _record_loop(
    store,
    project_id: str,
    *,
    episode_id: str,
    operation_id: str,
    created_at: str,
) -> None:
    request = RunRequest(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_id=str(uuid.uuid4()),
        chat_scope="node",
        node_id="exp/launched",
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/launched",
        control_revision=2,
        control_episode_id=episode_id,
        control_invocation=1,
        control_invocation_ceiling=3,
        control_decision_bundle=[],
        control_completion_criteria=["The indexed loop reaches a conclusion."],
    )
    store.create_experiment_episode_with_invocation(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            episode_id=episode_id,
            kind="node_chat",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=created_at,
            updated_at=created_at,
            status_message="Waiting for the loop invocation.",
            phase="queued",
            last_activity_at=created_at,
            authorized_by=authorized_human(store),
        )
    )
    store.complete_agent_task(operation_id, applied_revision=None, result={})


def _record_branch_target_child_experiment(
    app,
    *,
    node_id: str = "exp/launched",
    branch_ops: list[dict[str, object]] | None = None,
) -> tuple[EpisodeRecord, EpisodeRecord]:
    """Persist one real Auto-research branch and an Experiment child that uses it."""

    service = app.state.service
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id
    assert project_id is not None
    authorizer = authorized_human(store)
    now = store.now()
    parent_id = str(uuid.uuid4())
    parent_target = GraphTargetRef(kind="branch", branch_id=parent_id)
    base_head = service.history.head_ref()
    service.history.create_auto_research_branch(
        GraphBranchMetadata(
            branch_id=parent_id,
            episode_id=parent_id,
            project_id=project_id,
            base_head=base_head,
            head=GraphHeadRef(
                target=parent_target,
                revision=base_head.revision,
                transition_id=base_head.transition_id,
            ),
            authorized_by=authorizer,
        )
    )
    root_id = str(uuid.uuid4())
    parent, root = store.create_auto_research_episode_with_root_task(
        EpisodeRecord(
            episode_id=parent_id,
            project_id=project_id,
            mode="auto_research",
            graph_target=parent_target,
            graph_base_head=base_head,
            status="queued",
            invocation_ceiling=2,
            authorized_by=authorizer,
            created_at=now,
            updated_at=now,
        ),
        AutoResearchStateRecord(
            episode_id=parent_id,
            starting_instruction="Run the branch-scoped experiment.",
            created_at=now,
            updated_at=now,
        ),
        AgentTaskRecord(
            operation_id=root_id,
            project_id=project_id,
            episode_id=parent_id,
            graph_target=parent_target,
            kind="auto_research",
            status="queued",
            request={
                "episode_id": parent_id,
                "role": "orchestrator",
                "actor_operation_id": root_id,
                "run_truth_scope": ["repo-a"],
            },
            created_at=now,
            updated_at=now,
            status_message="Queued",
            authorized_by=authorizer,
            dispatch_authority=AgentDispatchAuthority(
                profile="orchestrator",
                task_contract="orchestrate",
                scope=AgentDispatchScope(
                    run_truth_scope=["repo-a"],
                    episode_id=parent_id,
                    patch_kind="work",
                ),
            ),
        ),
    )
    branch_service = service.for_graph_target(
        parent_target,
        expected_episode_id=parent_id,
    )
    if branch_ops:
        branch_service.history.append(
            Patch(
                kind="work",
                author="agent",
                summary="Prepared the branch-scoped Experiment index fixture.",
                source_operation_id=root.operation_id,
                run_truth_scope=["repo-a"],
                repositories_read=[],
                ops=branch_ops,
            )
        )
    branch_head = branch_service.history.head_ref()

    child_id = str(uuid.uuid4())
    child_operation_id = str(uuid.uuid4())
    goal = "Run the bounded branch experiment."
    child_request = RunRequest(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_id=child_id,
        chat_scope="node",
        node_id=node_id,
        message=goal,
        mode="work",
        trigger="orchestrator",
        patch_kind="experiment_loop",
        control_node_id=node_id,
        control_revision=branch_head.revision,
        control_episode_id=child_id,
        control_invocation=1,
        control_invocation_ceiling=3,
        control_decision_bundle=[],
        control_completion_criteria=["The branch-indexed loop reaches a conclusion."],
    )
    child_authority = resolve_dispatch_authority("node_chat", child_request)
    assert child_authority is not None
    child_task = AgentTaskRecord(
        operation_id=child_operation_id,
        project_id=project_id,
        episode_id=child_id,
        graph_target=parent_target,
        kind="node_chat",
        status="queued",
        request=child_request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Queued",
        authorized_by=authorizer,
        dispatch_authority=child_authority,
    )
    store.create_experiment_episode_with_invocation(
        child_task,
        auto_research_route=AutoResearchChildExperimentRecord(
            child_episode_id=child_id,
            auto_research_episode_id=parent_id,
            project_id=project_id,
            control_node_id=node_id,
            state="running",
            request={"goal": goal},
            goal_sha256=hashlib.sha256(goal.encode()).hexdigest(),
            parent_operation_id=root.operation_id,
            created_at=now,
            updated_at=now,
        ),
    )
    child = store.episode(child_id)
    assert child is not None
    return parent, child


def _seed_indexed_project(app) -> tuple[str, str]:
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    old_episode = str(uuid.uuid4())
    current_episode = str(uuid.uuid4())
    store = app.state.background_tasks.store
    _record_loop(
        store,
        project_id,
        episode_id=old_episode,
        operation_id="older-loop",
        created_at="2026-08-08T00:00:00+00:00",
    )
    store.request_episode_stop(old_episode)
    store.mark_episode_stop_skipped(old_episode)
    _record_loop(
        store,
        project_id,
        episode_id=current_episode,
        operation_id="current-loop",
        created_at="2026-08-09T00:00:00+00:00",
    )
    return project_id, current_episode


def _event_frame(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def test_experiment_index_uses_main_cache_and_unbounded_project_runtime(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, current_episode = _seed_indexed_project(app)
    client = TestClient(app)

    cached = client.get(f"/api/projects/{project_id}")
    assert cached.status_code == 200
    append_fixture_patch(app.state.service, _update_experiment_summary("Current graph summary."))

    def refuse_current_state_read():
        raise AssertionError("landing polling must not read current project history")

    monkeypatch.setattr(app.state.service.history, "state", refuse_current_state_read)

    store = app.state.background_tasks.store
    original = store.experiment_control_projection_snapshots
    calls: list[str] = []

    def capture(requested_project_id, *args, **kwargs):
        calls.append(requested_project_id)
        return original(requested_project_id, *args, **kwargs)

    monkeypatch.setattr(store, "experiment_control_projection_snapshots", capture)
    response = client.get("/api/episodes?mode=experiment_loop")

    assert response.status_code == 200
    assert calls == [project_id]
    assert len(response.json()) == 1
    entry = response.json()[0]
    assert set(entry) == {
        "project_id",
        "project_name",
        "project_reachable",
        "graph_target",
        "graph_head",
        "parent_episode_id",
        "node",
        "control",
        "episode",
    }
    assert entry["project_id"] == project_id
    assert entry["project_name"] == manifest.name
    assert entry["project_reachable"] is True
    assert entry["graph_target"] == {"kind": "main", "branch_id": None}
    assert entry["graph_head"] is None
    assert entry["parent_episode_id"] is None
    assert entry["node"]["id"] == "exp/launched"
    assert entry["node"]["current_summary"] == ""
    assert entry["control"]["episode_id"] == current_episode
    assert entry["control"]["invocations_used"] == 1
    assert entry["control"]["invocation_ceiling"] == 3
    assert entry["control"]["operational"]["current_operation_id"] == "current-loop"
    assert entry["control"]["operational"]["current_status"] == "succeeded"


def test_branch_modified_child_experiment_uses_exact_target_across_index_and_stop(
    manifest,
    tmp_path: Path,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    main_episode_id = str(uuid.uuid4())
    store = app.state.background_tasks.store
    _record_loop(
        store,
        project_id,
        episode_id=main_episode_id,
        operation_id="superseded-main-loop",
        created_at="2026-08-17T00:00:00+00:00",
    )
    store.request_episode_stop(main_episode_id)
    store.mark_episode_stop_skipped(main_episode_id)
    parent, child = _record_branch_target_child_experiment(
        app,
        branch_ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/launched",
                        "changes": {
                            "title": "Branch-only loop title",
                            "current_summary": "Visible only on the episode branch.",
                        },
                    }
                ],
            }
        ],
    )
    assert parent.graph_base_head is not None
    assert store.experiment_loop_runtime(project_id, "exp/launched").episode_id == child.episode_id
    assert (
        store.experiment_loop_runtime(
            project_id,
            "exp/launched",
            graph_target=GraphTargetRef(),
        ).episode_id
        is None
    )
    assert (
        store.experiment_loop_runtime(
            project_id,
            "exp/launched",
            graph_target=parent.graph_target,
        ).episode_id
        == child.episode_id
    )
    assert (
        store.active_experiment_control_ids(
            project_id,
            graph_target=GraphTargetRef(),
        )
        == set()
    )
    assert store.active_experiment_control_ids(
        project_id,
        graph_target=parent.graph_target,
    ) == {"exp/launched"}

    client = TestClient(app)
    try:
        cached = client.get(f"/api/projects/{project_id}")
        assert cached.status_code == 200
        assert cached.json()["graph"]["nodes"]["exp/launched"]["current_summary"] == ""
        assert cached.json()["experiment_control"]["exp/launched"]["episode_id"] is None
        assert cached.json()["experiment_control"]["exp/launched"]["episode"] is None

        main_state = service.history.state()
        main_node = main_state.nodes["exp/launched"]
        preview = client.post(
            f"/api/projects/{project_id}/sync/preview",
            json={
                "base_revision": main_state.revision,
                "nodes": [
                    {
                        "node_id": main_node.id,
                        "base_updated_rev": main_node.updated_rev,
                        "changes": {"current_summary": "Previewed on main only."},
                    }
                ],
            },
        )
        assert preview.status_code == 200, preview.text
        preview_control = preview.json()["projection"]["experiment_control"]["exp/launched"]
        assert preview_control["episode_id"] is None
        assert preview_control["episode"] is None

        invalid_main_run = client.post(
            f"/api/projects/{project_id}/experiments/exp%2Flaunched/run",
            json={},
        )
        assert invalid_main_run.status_code == 422

        project_list = client.get(
            f"/api/projects/{project_id}/episodes",
            params={"mode": "experiment_loop"},
        )
        assert project_list.status_code == 200
        project_child = next(
            item for item in project_list.json() if item["episode_id"] == child.episode_id
        )
        assert project_child["episode_id"] == child.episode_id
        assert project_child["graph_target"] == parent.graph_target.model_dump(mode="json")
        assert project_child["graph_base_head"] == parent.graph_base_head.model_dump(mode="json")
        assert project_child["graph_branch"] is None

        experiment_index = client.get("/api/episodes", params={"mode": "experiment_loop"})
        assert experiment_index.status_code == 200
        assert len(experiment_index.json()) == 1
        entry = experiment_index.json()[0]
        indexed_child = entry["episode"]
        assert indexed_child["episode_id"] == child.episode_id
        assert indexed_child["graph_target"] == parent.graph_target.model_dump(mode="json")
        assert indexed_child["graph_base_head"] == parent.graph_base_head.model_dump(mode="json")
        assert indexed_child["graph_branch"] is None
        assert entry["graph_target"] == parent.graph_target.model_dump(mode="json")
        assert entry["graph_head"]["target"] == parent.graph_target.model_dump(mode="json")
        assert entry["graph_head"]["revision"] > parent.graph_base_head.revision
        assert entry["parent_episode_id"] == parent.episode_id
        assert entry["node"]["title"] == "Branch-only loop title"
        assert entry["node"]["current_summary"] == "Visible only on the episode branch."
        assert entry["control"]["episode_id"] == child.episode_id

        ambiguous_stop = client.post(f"/api/projects/{project_id}/experiments/exp%2Flaunched/stop")
        assert ambiguous_stop.status_code == 409
        exact_stop = client.post(
            f"/api/projects/{project_id}/experiments/exp%2Flaunched/stop",
            params={"episode_id": child.episode_id},
        )
        assert exact_stop.status_code == 200
        assert exact_stop.json()["episode_id"] == child.episode_id
        exact_episode_stop = client.post(
            f"/api/projects/{project_id}/episodes/{child.episode_id}/stop"
        )
        assert exact_episode_stop.status_code == 200
        assert exact_episode_stop.json()["episode_id"] == child.episode_id
        assert exact_episode_stop.json()["graph_target"] == parent.graph_target.model_dump(
            mode="json"
        )
    finally:
        client.close()


def test_branch_created_child_experiment_is_indexed_without_entering_main_cache(
    manifest,
    tmp_path: Path,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    parent, child = _record_branch_target_child_experiment(
        app,
        node_id="exp/branch-created",
        branch_ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "exp/branch-created",
                        "type": "experiment",
                        "title": "Created only on the episode branch",
                        "objective": "Prove the cross-project index follows branch truth.",
                        "completion_criteria": ["The branch-only node appears in Runs."],
                        "invocation_ceiling": 3,
                    }
                ],
            }
        ],
    )
    project_id = app.state.default_project_id
    assert project_id is not None

    with TestClient(app) as client:
        cached = client.get(f"/api/projects/{project_id}")
        assert cached.status_code == 200
        assert "exp/branch-created" not in cached.json()["graph"]["nodes"]

        response = client.get("/api/episodes", params={"mode": "experiment_loop"})
        assert response.status_code == 200
        assert len(response.json()) == 1
        entry = response.json()[0]
        assert entry["node"]["id"] == "exp/branch-created"
        assert entry["node"]["title"] == "Created only on the episode branch"
        assert entry["control"]["episode_id"] == child.episode_id
        assert entry["episode"]["episode_id"] == child.episode_id
        assert entry["graph_target"] == parent.graph_target.model_dump(mode="json")
        assert entry["graph_head"]["target"] == parent.graph_target.model_dump(mode="json")
        assert entry["parent_episode_id"] == parent.episode_id
        assert "exp/branch-created" not in service.history.state().nodes


def test_exact_experiment_stop_routes_share_the_named_human_gate(manifest, tmp_path: Path) -> None:
    app = create_raw_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    episode_id = str(uuid.uuid4())

    with TestClient(app) as client:
        responses = [
            client.post(
                f"/api/projects/{project_id}/experiments/exp%2Fbranch-created/stop",
                params={"episode_id": episode_id},
            ),
            client.post(f"/api/projects/{project_id}/episodes/{episode_id}/stop"),
        ]

    for response in responses:
        assert response.status_code == 428
        assert response.json()["detail"]["code"] == "identity_name_required"


def test_terminal_exact_experiment_stop_is_a_conflict_instead_of_a_server_error(
    manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    append_fixture_patch(app.state.service, seed_patch())
    _parent, child = _record_branch_target_child_experiment(
        app,
        node_id="exp/terminal-stop",
        branch_ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "exp/terminal-stop",
                        "type": "experiment",
                        "title": "Terminal stop",
                        "objective": "Reject a stale Stop action without a 500.",
                        "invocation_ceiling": 2,
                    }
                ],
            }
        ],
    )

    def terminal_stop(*_args, **_kwargs):
        raise EpisodeNotRunning("the episode can no longer be stopped before wrap-up")

    monkeypatch.setattr(
        app.state.background_tasks.store,
        "request_experiment_loop_stop",
        terminal_stop,
    )
    with TestClient(app) as client:
        response = client.post(
            f"/api/projects/{child.project_id}/experiments/exp%2Fterminal-stop/stop",
            params={"episode_id": child.episode_id},
        )

    assert response.status_code == 409
    assert "can no longer be stopped" in response.json()["detail"]


def test_experiment_index_keeps_cached_unavailable_project_without_opening_it(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    first_app = create_app(str(manifest.path), data_dir=data_dir)
    project_id, current_episode = _seed_indexed_project(first_app)
    first_client = TestClient(first_app)
    snapshot = first_client.get(f"/api/projects/{project_id}").json()
    snapshot["canonical_state"]["reachable"] = False
    snapshot["canonical_state"]["error"] = "Project host is unavailable."
    first_app.state.catalog.update_summary(project_id, snapshot)
    first_app.state.catalog.write_cached_snapshot(project_id, snapshot)

    record = first_app.state.catalog.store.project(project_id)
    assert record is not None
    first_app.state.catalog.store.upsert_project(
        ProjectRecord(
            project_id="unusable-project",
            locator=str(tmp_path / "missing" / "manifest.toml"),
            name="Unusable project",
            state_location=str(tmp_path / "missing" / ".research"),
            state_remote=False,
            added_at=record.added_at,
        )
    )

    restarted = create_app(data_dir=data_dir)

    def refuse_open(_project_id):
        raise AssertionError("the experiment index must not open inactive projects")

    monkeypatch.setattr(restarted.state.catalog, "_open_service", refuse_open)
    response = TestClient(restarted).get("/api/episodes?mode=experiment_loop")

    assert response.status_code == 200
    assert len(response.json()) == 1
    entry = response.json()[0]
    assert entry["project_id"] == project_id
    assert entry["project_reachable"] is False
    assert entry["control"]["episode_id"] == current_episode
    assert project_id not in restarted.state.catalog._services
    assert "unusable-project" not in restarted.state.catalog._services


def test_graph_capable_background_stream_refreshes_cached_experiment_semantics(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, current_episode = _seed_indexed_project(app)
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    release = threading.Event()
    operation: dict[str, str] = {}

    async def update_graph(service, _launcher, _request, _data_dir, *, execution):
        del execution
        await asyncio.to_thread(release.wait)
        append_fixture_patch(service, _update_experiment_summary("Refreshed after Work."))
        yield _event_frame(AgentEvent(event="answer", text="Updated the experiment."))
        yield _event_frame(AgentEvent(event="done"))

    monkeypatch.setattr(api_app_module, "stream_work_run", update_graph)
    store = app.state.background_tasks.store
    original_commit = app.state.catalog.commit_cached_snapshot
    stream_closed_statuses: list[str] = []
    stream_closed = threading.Event()

    def capture_commit(requested_project_id, snapshot, *, generation, patch_log_head=None):
        record = store.agent_task(operation["id"])
        assert record is not None
        stream_closed_statuses.append(record.status)
        committed = original_commit(
            requested_project_id,
            snapshot,
            generation=generation,
            patch_log_head=patch_log_head,
        )
        stream_closed.set()
        return committed

    monkeypatch.setattr(app.state.catalog, "commit_cached_snapshot", capture_commit)
    request = RunRequest(
        provider="codex",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Update the indexed experiment.",
        mode="work",
        patch_kind="work",
    )
    task = app.state.background_tasks.start(
        project_id,
        "project_chat",
        request,
        authorized_by=authorized_human(app),
    )
    operation["id"] = task.operation_id
    release.set()
    completed = wait_for_task(store, task.operation_id)

    assert completed.status == "succeeded"
    assert stream_closed.wait(timeout=2)
    assert stream_closed_statuses == ["running"]
    cached = client.get(f"/api/projects/{project_id}/cached").json()
    assert cached["graph"]["nodes"]["exp/launched"]["current_summary"] == ("Refreshed after Work.")
    assert (
        cached["experiment_control"]["exp/launched"]["operational"]["current_status"] == "succeeded"
    )
    response = client.get("/api/episodes?mode=experiment_loop")
    assert response.status_code == 200
    assert response.json()[0]["node"]["current_summary"] == "Refreshed after Work."
    assert response.json()[0]["control"]["episode_id"] == current_episode


def test_experiment_index_runtime_projection_failure_fails_the_request(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, current_episode = _seed_indexed_project(app)
    assert TestClient(app).get(f"/api/projects/{project_id}").status_code == 200

    def fail_runtime_projection(_project_id, *_args, **_kwargs):
        raise RuntimeError("runtime projection broke")

    monkeypatch.setattr(
        app.state.background_tasks.store,
        "experiment_control_projection_snapshots",
        fail_runtime_projection,
    )
    response = TestClient(app, raise_server_exceptions=False).get(
        "/api/episodes?mode=experiment_loop"
    )

    assert response.status_code == 500


def test_display_cache_refresh_failure_is_diagnostic_not_task_failure(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, _current_episode = _seed_indexed_project(app)
    assert TestClient(app).get(f"/api/projects/{project_id}").status_code == 200

    async def finish_work(_service, _launcher, _request, _data_dir, *, execution):
        del execution
        yield _event_frame(AgentEvent(event="answer", text="Work completed."))
        yield _event_frame(AgentEvent(event="done"))

    def fail_cache_write(_project_id, _snapshot, *, generation, patch_log_head=None):
        del generation, patch_log_head
        raise OSError("display cache is unavailable")

    monkeypatch.setattr(api_app_module, "stream_work_run", finish_work)
    monkeypatch.setattr(app.state.catalog, "commit_cached_snapshot", fail_cache_write)
    store = app.state.background_tasks.store
    request = RunRequest(
        provider="codex",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Complete graph-capable work.",
        mode="work",
        patch_kind="work",
    )
    task = app.state.background_tasks.start(
        project_id,
        "project_chat",
        request,
        authorized_by=authorized_human(app),
    )
    completed = wait_for_task(store, task.operation_id)

    assert completed.status == "succeeded"
    deadline = time.monotonic() + 2
    failure = None
    while time.monotonic() < deadline and failure is None:
        failure = next(
            (
                item
                for item in store.agent_task_receipts(task.operation_id)
                if item.category == "display_cache_refresh_failed"
            ),
            None,
        )
        time.sleep(0.01)
    assert failure is not None
    assert failure.payload["exception_type"] == "OSError"


def test_versioned_cache_commit_cannot_regress_graph_or_project_summary(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, _current_episode = _seed_indexed_project(app)
    client = TestClient(app)
    older = client.get(f"/api/projects/{project_id}").json()
    question = app.state.service.history.state().nodes["rq/learning-after-shift"]
    append_fixture_patch(
        app.state.service,
        _update_primary_question(
            "What is the newest question?", base_updated_rev=question.updated_rev
        ),
    )
    state = app.state.service.history.materialize(write_outputs=False).state
    newer = app.state.service.project_snapshot(state=state)
    newer["id"] = project_id

    catalog = app.state.catalog
    newer_generation = catalog.reserve_cached_snapshot_generation(project_id)
    older_generation = catalog.reserve_cached_snapshot_generation(project_id)
    entered = threading.Event()
    release = threading.Event()
    original_write = catalog._write_cached_snapshot_locked

    def block_older_write(requested_project_id, snapshot):
        if snapshot["revision"] == older["revision"]:
            entered.set()
            assert release.wait(timeout=2)
        original_write(requested_project_id, snapshot)

    monkeypatch.setattr(catalog, "_write_cached_snapshot_locked", block_older_write)
    results: dict[str, bool] = {}
    newer_thread = threading.Thread(
        target=lambda: results.setdefault(
            "newer",
            catalog.commit_cached_snapshot(
                project_id,
                newer,
                generation=newer_generation,
            ),
        )
    )
    older_thread = threading.Thread(
        target=lambda: results.setdefault(
            "older",
            catalog.commit_cached_snapshot(
                project_id,
                older,
                generation=older_generation,
            ),
        )
    )

    older_thread.start()
    assert entered.wait(timeout=1)
    newer_thread.start()
    release.set()
    newer_thread.join(timeout=2)
    older_thread.join(timeout=2)

    assert not newer_thread.is_alive()
    assert not older_thread.is_alive()
    assert results == {"older": True, "newer": True}
    cached = catalog.cached_snapshot(project_id)
    assert cached is not None
    assert cached["revision"] == newer["revision"]
    assert cached["primary_question"]["question"] == "What is the newest question?"
    record = catalog.store.project(project_id)
    assert record is not None
    assert record.revision == newer["revision"]
    assert record.primary_question == "What is the newest question?"

    regressive_generation = catalog.reserve_cached_snapshot_generation(project_id)
    assert not catalog.commit_cached_snapshot(
        project_id,
        older,
        generation=regressive_generation,
    )


def test_cache_generation_rejects_out_of_order_same_revision_reachability(
    manifest, tmp_path: Path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, _current_episode = _seed_indexed_project(app)
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    catalog = app.state.catalog
    first_read = threading.Event()
    release_first = threading.Event()
    results: dict[str, bool] = {}

    def first_writer() -> None:
        generation = catalog.reserve_cached_snapshot_generation(project_id)
        snapshot = catalog.cached_snapshot(project_id)
        assert snapshot is not None
        first_read.set()
        assert release_first.wait(timeout=2)
        results["first"] = catalog.commit_cached_snapshot(
            project_id,
            snapshot,
            generation=generation,
        )

    thread = threading.Thread(target=first_writer)
    thread.start()
    assert first_read.wait(timeout=1)
    newer_generation = catalog.reserve_cached_snapshot_generation(project_id)
    newer = catalog.cached_snapshot(project_id)
    assert newer is not None
    newer = {**newer, "canonical_state": {**newer["canonical_state"], "reachable": False}}
    assert catalog.commit_cached_snapshot(
        project_id,
        newer,
        generation=newer_generation,
    )
    release_first.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert results == {"first": False}
    cached = catalog.cached_snapshot(project_id)
    assert cached is not None
    assert cached["canonical_state"]["reachable"] is False
    record = app.state.catalog.store.project(project_id)
    assert record is not None
    assert record.reachable is False


def test_experiment_loop_cache_blocks_terminal_runtime_until_graph_is_visible(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, current_episode = _seed_indexed_project(app)
    app.state.background_tasks.store.request_episode_stop(current_episode)
    app.state.background_tasks.store.mark_episode_stop_skipped(current_episode)
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    entered_cache = threading.Event()
    release_cache = threading.Event()

    async def update_graph(service, _launcher, _request, _data_dir, *, execution):
        del execution
        append_fixture_patch(
            service, _update_experiment_summary("Graph visible with terminal task.")
        )
        yield _event_frame(AgentEvent(event="answer", text="Updated the experiment."))
        yield _event_frame(AgentEvent(event="done"))

    monkeypatch.setattr(api_app_module, "stream_experiment_loop_task", update_graph)
    catalog = app.state.catalog
    original_commit = catalog.commit_cached_snapshot

    def block_cache(requested_project_id, snapshot, *, generation, patch_log_head=None):
        if snapshot["graph"]["nodes"]["exp/launched"]["current_summary"]:
            entered_cache.set()
            assert release_cache.wait(timeout=2)
        return original_commit(
            requested_project_id,
            snapshot,
            generation=generation,
            patch_log_head=patch_log_head,
        )

    monkeypatch.setattr(catalog, "commit_cached_snapshot", block_cache)
    episode_id = str(uuid.uuid4())
    request = RunRequest(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_id=str(uuid.uuid4()),
        chat_scope="node",
        node_id="exp/launched",
        message="Continue the experiment loop.",
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/launched",
        control_revision=2,
        control_episode_id=episode_id,
        control_invocation=1,
        control_invocation_ceiling=3,
        control_decision_bundle=[],
        control_completion_criteria=["The indexed loop reaches a conclusion."],
    )
    task = app.state.background_tasks.start(
        project_id,
        "node_chat",
        request,
        authorized_by=authorized_human(app),
    )
    assert entered_cache.wait(timeout=1)

    running = app.state.background_tasks.store.agent_task(task.operation_id)
    assert running is not None
    assert running.status == "running"
    before_release = client.get("/api/episodes?mode=experiment_loop").json()[0]
    assert before_release["node"]["current_summary"] == ""
    assert before_release["control"]["episode_id"] == episode_id
    assert before_release["control"]["operational"]["current_status"] == "running"

    release_cache.set()
    completed = wait_for_task(app.state.background_tasks.store, task.operation_id)
    assert completed.status == "succeeded"
    after_release = client.get("/api/episodes?mode=experiment_loop").json()[0]
    assert after_release["node"]["current_summary"] == ("Graph visible with terminal task.")
    assert after_release["control"]["episode_id"] == episode_id
    assert after_release["control"]["operational"]["current_status"] == "succeeded"


def test_persisted_experiment_request_selects_owner_without_work_fallback(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, current_episode = _seed_indexed_project(app)
    store = app.state.background_tasks.store
    store.request_episode_stop(current_episode)
    store.mark_episode_stop_skipped(current_episode)
    selected: list[str] = []

    async def selected_owner(_service, _launcher, _request, _data_dir, *, execution):
        selected.append(execution.operation_id)
        raise RuntimeError("specialized Experiment owner failed")
        yield  # pragma: no cover - keeps this an async generator

    async def unexpected_work_fallback(*_args, **_kwargs):
        raise AssertionError("Experiment dispatch fell back to ordinary Work")
        yield  # pragma: no cover - keeps this an async generator

    monkeypatch.setattr(api_app_module, "stream_experiment_loop_task", selected_owner)
    monkeypatch.setattr(api_app_module, "stream_work_run", unexpected_work_fallback)
    episode_id = str(uuid.uuid4())
    request = RunRequest(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_id=str(uuid.uuid4()),
        chat_scope="node",
        node_id="exp/launched",
        message="Continue the persisted Experiment loop.",
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/launched",
        control_revision=2,
        control_episode_id=episode_id,
        control_invocation=1,
        control_invocation_ceiling=3,
        control_decision_bundle=[],
        control_completion_criteria=["The indexed loop reaches a conclusion."],
    )

    task = app.state.background_tasks.start(
        project_id,
        "node_chat",
        request,
        authorized_by=authorized_human(store),
    )
    completed = wait_for_task(store, task.operation_id, expect="failed")

    assert selected == [task.operation_id]
    assert completed.error == "specialized Experiment owner failed"


@pytest.mark.parametrize(
    ("terminal_event", "expected_status"),
    [("error", "failed"), ("paused", "paused")],
)
def test_stream_closed_cache_hook_runs_before_error_and_pause_verdicts(
    manifest, tmp_path: Path, monkeypatch, terminal_event: str, expected_status: str
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, _current_episode = _seed_indexed_project(app)
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200

    async def update_then_stop(service, _launcher, _request, _data_dir, *, execution):
        del execution
        append_fixture_patch(service, _update_experiment_summary(f"Graph before {terminal_event}."))
        yield _event_frame(AgentEvent(event=terminal_event, text=f"Task {terminal_event}."))

    monkeypatch.setattr(api_app_module, "stream_work_run", update_then_stop)
    request = RunRequest(
        provider="codex",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Update before stopping.",
        mode="work",
        patch_kind="work",
    )
    task = app.state.background_tasks.start(
        project_id,
        "project_chat",
        request,
        authorized_by=authorized_human(app),
    )
    completed = wait_for_task(app.state.background_tasks.store, task.operation_id)

    assert completed.status == expected_status
    indexed = client.get("/api/episodes?mode=experiment_loop").json()[0]
    assert indexed["node"]["current_summary"] == f"Graph before {terminal_event}."


def test_experiment_index_fails_for_malformed_existing_cache(manifest, tmp_path: Path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, _current_episode = _seed_indexed_project(app)
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    app.state.catalog._cached_snapshot_path(project_id).write_text("{", encoding="utf-8")

    response = client.get("/api/episodes?mode=experiment_loop")

    assert response.status_code == 503


def test_experiment_index_reads_pre_identity_display_cache(manifest, tmp_path: Path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, current_episode = _seed_indexed_project(app)
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200

    store = app.state.background_tasks.store
    with store.connection() as connection:
        connection.execute(
            "UPDATE projects SET home_space_id = NULL WHERE project_id = ?",
            (project_id,),
        )
    cache_path = app.state.catalog._cached_snapshot_path(project_id)
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    del envelope["snapshot"]["home_space_id"]
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    response = client.get("/api/episodes?mode=experiment_loop")

    assert response.status_code == 200
    assert response.json()[0]["control"]["episode_id"] == current_episode
    assert "home_space_id" not in json.loads(cache_path.read_text(encoding="utf-8"))["snapshot"]


def test_experiment_index_fails_when_revisioned_project_cache_is_missing(
    manifest, tmp_path: Path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id, _current_episode = _seed_indexed_project(app)
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    app.state.catalog._cached_snapshot_path(project_id).unlink()

    response = client.get("/api/episodes?mode=experiment_loop")

    assert response.status_code == 503
