from __future__ import annotations

import json
import threading
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from rcp.agents import AgentEvent, AgentProcessControl
from rcp.api.app import create_app
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import (
    BranchMergeProvenance,
    BranchMergeReceipt,
    GraphBranchMetadata,
    Patch,
)
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import BranchMergeAlreadyCommitted
from rcp.runs.auto_research import AutoResearchRunRequest, AutoResearchStartRequest
from rcp.runs.auto_research_admission import (
    reconcile_reserved_auto_research_roots,
    reserve_auto_research,
)
from rcp.runs.branch_merge import branch_merge_id
from rcp.runs.tasks.branch_merge import _apply_receipt_to_execution
from rcp.service import RunRequest, resolve_dispatch_authority
from rcp.setup import ProjectSetupRequest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchStateRecord,
    EpisodeRecord,
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    WatcherContinuation,
)

from .helpers import (
    TASK_SETTLE_TIMEOUT,
    append_fixture_patch,
    authorized_human,
    create_named_app,
    wait_for_task,
)

BranchChange = Literal["none", "evidence", "status", "bookkeeping"]


@dataclass(frozen=True)
class _BranchHarness:
    app: object
    client: TestClient
    project_id: str
    store: AppStore
    service: object
    episode: EpisodeRecord
    root: AgentTaskRecord
    branch: object


def _main_fixture_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Created the branch-merge fixture blocker.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "blk/merge-ready",
                        "type": "blocker",
                        "title": "Merge gate",
                        "description": "The branch may resolve this gate.",
                        "status": "open",
                    }
                ],
            }
        ],
    )


def _branch_patch(operation_id: str, change: BranchChange, *, suffix: str = "") -> Patch:
    if change == "evidence":
        operations: list[dict[str, object]] = [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": f"ev/branch-result{suffix}",
                        "type": "evidence",
                        "title": f"Branch result{suffix}",
                        "observation": "The isolated branch recorded a result.",
                        "origin": "internal_run",
                    }
                ],
            }
        ]
    elif change == "status":
        operations = [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "blk/merge-ready",
                        "changes": {"status": "resolved"},
                    }
                ],
            }
        ]
    elif change == "bookkeeping":
        # The operation advances the branch transition chain, but its only
        # domain value is unchanged. The resulting semantic branch delta is empty.
        operations = [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "blk/merge-ready",
                        "changes": {"title": "Merge gate"},
                    }
                ],
            }
        ]
    else:
        raise ValueError("the no-patch case is represented by not appending")
    return Patch(
        kind="work",
        author="agent",
        summary=f"Recorded the {change} branch fixture.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        source_operation_id=operation_id,
        ops=operations,
    )


def _orchestrator_authority(episode_id: str) -> AgentDispatchAuthority:
    return AgentDispatchAuthority(
        profile="orchestrator",
        task_contract="orchestrate",
        scope=AgentDispatchScope(
            run_truth_scope=["repo-a"],
            episode_id=episode_id,
            patch_kind="work",
        ),
    )


def _create_branch_harness(
    manifest,
    tmp_path: Path,
    *,
    change: BranchChange,
    ended: bool = True,
) -> _BranchHarness:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    service = app.state.catalog.open(project_id)
    store = app.state.background_tasks.store
    authorizer = authorized_human(app)

    append_fixture_patch(service, _main_fixture_patch())
    base_head = service.history.head_ref()
    assert base_head.transition_id is not None
    episode_id = str(uuid.uuid4())
    target = GraphTargetRef(kind="branch", branch_id=episode_id)
    branch = service.history.create_auto_research_branch(
        GraphBranchMetadata(
            branch_id=episode_id,
            episode_id=episode_id,
            project_id=project_id,
            base_head=base_head,
            head=GraphHeadRef(
                target=target,
                revision=base_head.revision,
                transition_id=base_head.transition_id,
            ),
            authorized_by=authorizer,
        )
    )

    now = store.now()
    root_id = str(uuid.uuid4())
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode="auto_research",
        graph_target=target,
        graph_base_head=base_head,
        status="queued",
        invocation_ceiling=4,
        authorized_by=authorizer,
        created_at=now,
        updated_at=now,
    )
    root_request = AutoResearchRunRequest(
        episode_id=episode_id,
        role="orchestrator",
        actor_operation_id=root_id,
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
    )
    root = AgentTaskRecord(
        operation_id=root_id,
        project_id=project_id,
        episode_id=episode_id,
        graph_target=target,
        kind="auto_research",
        status="queued",
        request=root_request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Queued",
        authorized_by=authorizer,
        dispatch_authority=_orchestrator_authority(episode_id),
    )
    episode, root = store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode_id,
            created_at=now,
            updated_at=now,
        ),
        root,
    )
    if change != "none":
        branch.append(
            _branch_patch(root.operation_id, change),
            expected_revision=base_head.revision,
        )
    if ended:
        store.complete_agent_task(root.operation_id, applied_revision=None, result={})
        episode = store.mark_episode_stop_skipped(episode_id)
        stored_root = store.agent_task(root.operation_id)
        assert stored_root is not None
        root = stored_root

    return _BranchHarness(
        app=app,
        client=TestClient(app),
        project_id=project_id,
        store=store,
        service=service,
        episode=episode,
        root=root,
        branch=branch,
    )


def _episode_payload(harness: _BranchHarness) -> dict[str, object]:
    response = harness.client.get(
        f"/api/projects/{harness.project_id}/episodes",
        params={"mode": "auto_research"},
    )
    assert response.status_code == 200, response.text
    return next(
        item for item in response.json() if item["episode_id"] == harness.episode.episode_id
    )


def test_episode_list_reads_all_branches_from_one_snapshot_without_publishing(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="evidence")
    tasks = harness.app.state.background_tasks
    monkeypatch.setattr(tasks, "_spawn_record", lambda record, _request, **_kwargs: record)
    second = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes",
        json={"mode": "auto_research", "invocation_ceiling": 2},
    )
    assert second.status_code == 202, second.text

    workspace = harness.service.history.workspace
    refresh_if_stale = workspace.refresh_if_stale
    materialize = harness.service.history.materialize
    calls = {"refresh": 0, "main_replay": 0}

    def counted_refresh() -> bool:
        calls["refresh"] += 1
        return refresh_if_stale()

    def counted_materialize(*args, **kwargs):
        calls["main_replay"] += 1
        return materialize(*args, **kwargs)

    def forbidden_publish(_paths) -> None:
        raise AssertionError("an episode-list GET must not publish canonical files")

    @contextmanager
    def forbidden_transaction():
        raise AssertionError("an episode-list GET must not take a publication transaction")
        yield

    monkeypatch.setattr(workspace, "refresh_if_stale", counted_refresh)
    monkeypatch.setattr(workspace, "transaction", forbidden_transaction)
    monkeypatch.setattr(workspace, "publish", forbidden_publish)
    monkeypatch.setattr(harness.service.history, "materialize", counted_materialize)

    listed = harness.client.get(
        f"/api/projects/{harness.project_id}/episodes",
        params={"mode": "auto_research"},
    )

    assert listed.status_code == 200, listed.text
    assert len(listed.json()) == 2
    assert calls == {"refresh": 1, "main_replay": 1}


def _current_receipt(harness: _BranchHarness, *, task_id: str) -> BranchMergeReceipt:
    metadata = harness.branch.branch_metadata()
    main_head = harness.service.history.head_ref()
    return BranchMergeReceipt(
        outcome="no_change",
        provenance=BranchMergeProvenance(
            merge_id=branch_merge_id(metadata),
            branch_id=metadata.branch_id,
            episode_id=metadata.episode_id,
            branch_base_head=metadata.base_head,
            branch_head=metadata.head,
            rebased_main_head=main_head,
            merge_task_id=task_id,
        ),
        result_main_head=main_head,
        authorized_by=authorized_human(harness.app),
    )


def _admit_held_merge_task(
    harness: _BranchHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> AgentTaskRecord:
    held: list[AgentTaskRecord] = []
    monkeypatch.setattr(
        harness.app.state.background_tasks,
        "_spawn_record",
        lambda record, _request, **_kwargs: held.append(record) or record,
    )
    response = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes/{harness.episode.episode_id}/merge"
    )
    assert response.status_code == 202, response.text
    assert len(held) == 1
    stored = harness.store.agent_task(held[0].operation_id)
    assert stored is not None
    return stored


@pytest.mark.parametrize(
    ("task_status", "expected_merge_state", "diagnostic"),
    [
        ("paused", "needs_action", "The merge was paused by the provider."),
        ("interrupted", "needs_action", "The merge connection was interrupted."),
        ("failed", "failed", "The merge candidate failed validation."),
    ],
)
def test_episode_projection_distinguishes_merge_failures_from_tasks_needing_action(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_status: Literal["paused", "interrupted", "failed"],
    expected_merge_state: Literal["needs_action", "failed"],
    diagnostic: str,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="evidence")
    task = _admit_held_merge_task(harness, monkeypatch)

    if task_status == "paused":
        harness.store.pause_agent_task(task.operation_id, detail=diagnostic)
    elif task_status == "interrupted":
        harness.store.fail_agent_task(task.operation_id, diagnostic, status="interrupted")
    else:
        harness.store.fail_agent_task(task.operation_id, diagnostic)

    summary = _episode_payload(harness)["graph_branch"]
    assert summary["merge_state"] == expected_merge_state
    assert summary["merge_diagnostic"] == diagnostic
    assert summary["active_merge_task_id"] is None
    assert summary["merge_eligible"] is True


def _candidate_for(change: Literal["evidence", "status"]) -> str:
    if change == "evidence":
        operations: list[dict[str, object]] = [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "ev/branch-result",
                        "type": "evidence",
                        "title": "Branch result",
                        "observation": "The isolated branch recorded a result.",
                        "origin": "internal_run",
                    }
                ],
            }
        ]
    else:
        operations = [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "blk/merge-ready",
                        "changes": {"status": "resolved"},
                    }
                ],
            }
        ]
    return json.dumps(
        {
            "summary": "Carry the isolated branch result onto main.",
            "ops": operations,
            "repositories_read": [],
            "change_summary": ["Merged the branch's semantic result."],
        }
    )


class _PatchWritingLauncher:
    def __init__(self, candidate: str) -> None:
        self.candidate = candidate
        self.calls = 0

    async def stream(
        self,
        _provider: str,
        _prompt: str,
        *,
        cwd: Path,
        session_id: str | None,
        **_kwargs: object,
    ) -> AsyncIterator[AgentEvent]:
        self.calls += 1
        assert len(list(cwd.glob("rcp-command-*.credential.json"))) == 1
        (cwd / "patch.json").write_text(self.candidate, encoding="utf-8")
        yield AgentEvent(event="session", session_id=session_id or "merge-native-session")
        yield AgentEvent(event="provider_exit", text='{"return_code":0}')
        yield AgentEvent(event="done")


def _setup_payload(repository: Path, *, name: str) -> dict[str, object]:
    repository.mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "repositories": [
            {
                "alias": "paper-repo",
                "location": "local",
                "path": str(repository),
                "host": "",
                "default_read": True,
            }
        ],
        "state_repository": "paper-repo",
        "execution": {"location": "local", "host": ""},
        "confirmed": True,
    }


def test_start_episode_projects_the_exact_main_transition_into_its_branch_head(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, _main_fixture_patch())
    exact_main = service.history.head_ref()
    assert exact_main.transition_id is not None

    tasks = app.state.background_tasks
    launch_observations: list[str] = []

    def hold_after_branch(record, _request, **_kwargs):
        episode = tasks.store.episode(record.episode_id)
        assert episode is not None
        service.history.branch(
            episode.episode_id,
            expected_episode_id=episode.episode_id,
            expected_project_id=episode.project_id,
        )
        launch_observations.append(record.operation_id)
        return record

    monkeypatch.setattr(tasks, "_spawn_record", hold_after_branch)
    response = TestClient(app).post(
        f"/api/projects/{app.state.default_project_id}/episodes",
        json={"mode": "auto_research", "invocation_ceiling": 2},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    branch = payload["graph_branch"]
    assert payload["graph_base_head"] == exact_main.model_dump(mode="json")
    assert branch["base_head"] == exact_main.model_dump(mode="json")
    assert branch["head"] == {
        "target": {"kind": "branch", "branch_id": payload["episode_id"]},
        "revision": exact_main.revision,
        "transition_id": exact_main.transition_id,
    }
    assert launch_observations == [payload["root_operation_id"]]

    refused = TestClient(app).post(
        f"/api/projects/{app.state.default_project_id}/episodes",
        json={"mode": "auto_research", "invocation_ceiling": 2},
    )
    assert refused.status_code == 409
    branches = service.manifest.research_dir / "branches"
    assert sorted(path.name for path in branches.iterdir() if not path.name.startswith(".")) == [
        payload["episode_id"]
    ]


def test_start_episode_keeps_a_visible_failed_reservation_when_branch_creation_fails(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    service = app.state.catalog.open(project_id)
    tasks = app.state.background_tasks
    spawned: list[str] = []
    monkeypatch.setattr(
        tasks,
        "_spawn_record",
        lambda record, _request, **_kwargs: spawned.append(record.operation_id),
    )

    def unavailable_branch(_metadata):
        raise ValueError("canonical branch publication unavailable")

    monkeypatch.setattr(service.history, "create_auto_research_branch", unavailable_branch)
    response = TestClient(app).post(
        f"/api/projects/{project_id}/episodes",
        json={"mode": "auto_research", "invocation_ceiling": 2},
    )

    assert response.status_code == 422, response.text
    assert spawned == []
    episodes = tasks.store.episodes(project_id)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.status == "failed"
    assert episode.ending == "failed"
    # The root failed before provider launch, so it never bound a session to report
    # from and the episode carries no report error for work that never ran.
    assert episode.wrapup_state == "not_started"
    assert episode.wrapup_error is None
    assert tasks.store.episode_wrapup(episode.episode_id) is None
    root = tasks.store.agent_task(episode.root_operation_id)
    assert root is not None
    assert root.status == "failed"
    assert "before provider launch" in (root.error or "")
    listed = TestClient(app).get(
        f"/api/projects/{project_id}/episodes",
        params={"mode": "auto_research"},
    )
    assert listed.status_code == 200, listed.text
    summary = listed.json()[0]["graph_branch"]
    assert summary["merge_state"] == "failed"
    assert summary["merge_eligible"] is False


def test_episode_list_projects_a_reservation_while_remote_branch_publication_is_pending(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    service = app.state.catalog.open(project_id)
    tasks = app.state.background_tasks
    entered = threading.Event()
    release = threading.Event()
    real_create = service.history.create_auto_research_branch

    def delayed_create(metadata):
        entered.set()
        assert release.wait(timeout=5)
        return real_create(metadata)

    monkeypatch.setattr(service.history, "create_auto_research_branch", delayed_create)
    monkeypatch.setattr(tasks, "_spawn_record", lambda record, _request, **_kwargs: record)
    start_client = TestClient(app)
    list_client = TestClient(app)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            start_client.post,
            f"/api/projects/{project_id}/episodes",
            json={"mode": "auto_research", "invocation_ceiling": 2},
        )
        assert entered.wait(timeout=5)
        try:
            listed = list_client.get(
                f"/api/projects/{project_id}/episodes",
                params={"mode": "auto_research"},
            )
            assert listed.status_code == 200, listed.text
            reservation = listed.json()[0]
            assert reservation["status"] == "queued"
            assert reservation["graph_branch"]["merge_state"] == "unmerged"
            assert reservation["graph_branch"]["merge_eligible"] is False
            assert "Establishing" in reservation["graph_branch"]["merge_diagnostic"]
        finally:
            release.set()
        started = future.result(timeout=5)

    assert started.status_code == 202, started.text
    assert started.json()["status"] == "running"


def test_restart_reconciles_an_already_published_reserved_branch(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    service = app.state.catalog.open(project_id)
    tasks = app.state.background_tasks
    base_head = service.history.head_ref()
    episode, root, _request = reserve_auto_research(
        tasks,
        project_id,
        AutoResearchStartRequest(
            invocation_ceiling=2,
            provider="codex",
            run_on="laptop",
            run_truth_scope=["repo-a"],
        ),
        authorized_by=authorized_human(app),
        graph_base_head=base_head,
    )

    def ensure(reserved: EpisodeRecord) -> None:
        service.history.create_auto_research_branch(
            GraphBranchMetadata(
                branch_id=reserved.episode_id,
                episode_id=reserved.episode_id,
                project_id=reserved.project_id,
                base_head=reserved.graph_base_head,
                head=GraphHeadRef(
                    target=reserved.graph_target,
                    revision=base_head.revision,
                    transition_id=base_head.transition_id,
                ),
                created_at=reserved.created_at,
                authorized_by=reserved.authorized_by,
            )
        )

    ensure(episode)
    restarted = BackgroundAgentTasks(tasks.store, tasks.stream)
    spawned: list[str] = []

    def held_spawn(record, _request, **_kwargs):
        spawned.append(record.operation_id)
        return record

    monkeypatch.setattr(restarted, "_spawn_record", held_spawn)
    assert reconcile_reserved_auto_research_roots(restarted, ensure) == [root.operation_id]
    assert spawned == [root.operation_id]
    assert tasks.store.episode(episode.episode_id).status == "running"  # type: ignore[union-attr]


def test_episode_projection_and_merge_admission_are_exact_and_recover_by_fresh_dispatch(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="evidence")
    payload = _episode_payload(harness)
    metadata = harness.branch.branch_metadata()
    summary = payload["graph_branch"]
    assert isinstance(summary, dict)
    assert set(summary) == {
        "branch_id",
        "episode_id",
        "base_head",
        "head",
        "merge_eligible",
        "merge_state",
        "latest_successful_merge",
        "active_merge_task_id",
        "merge_diagnostic",
    }
    assert payload["graph_target"] == harness.episode.graph_target.model_dump(mode="json")
    assert payload["graph_base_head"] == metadata.base_head.model_dump(mode="json")
    assert summary == {
        "branch_id": metadata.branch_id,
        "episode_id": metadata.episode_id,
        "base_head": metadata.base_head.model_dump(mode="json"),
        "head": metadata.head.model_dump(mode="json"),
        "merge_eligible": True,
        "merge_state": "unmerged",
        "latest_successful_merge": None,
        "active_merge_task_id": None,
        "merge_diagnostic": None,
    }

    second = harness.client.post(
        "/api/project-setup/create",
        json=_setup_payload(tmp_path / "second-project", name="Second project"),
    )
    assert second.status_code == 200, second.text
    refused_cross_project = harness.client.post(
        f"/api/projects/{second.json()['id']}/episodes/{harness.episode.episode_id}/merge"
    )
    assert refused_cross_project.status_code == 404

    held: list[AgentTaskRecord] = []

    def hold_spawn(record, _request, **_kwargs):
        held.append(record)
        return record

    monkeypatch.setattr(harness.app.state.background_tasks, "_spawn_record", hold_spawn)
    budget_before = harness.store.episode_budget_meter(harness.episode.episode_id)
    admitted = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes/{harness.episode.episode_id}/merge"
    )

    assert admitted.status_code == 202, admitted.text
    assert len(held) == 1
    first = harness.store.agent_task(held[0].operation_id)
    assert first is not None
    assert first.kind == "branch_merge"
    assert first.status == "queued"
    assert first.graph_target == harness.episode.graph_target
    assert first.episode_id == harness.episode.episode_id
    assert first.authorized_by == authorized_human(harness.app)
    assert first.dispatch_authority == _orchestrator_authority(harness.episode.episode_id)
    assert harness.store.episode_budget_meter(harness.episode.episode_id) == budget_before
    admitted_summary = admitted.json()["graph_branch"]
    assert admitted_summary["merge_state"] == "running"
    assert admitted_summary["merge_eligible"] is False
    assert admitted_summary["active_merge_task_id"] == first.operation_id

    concurrent = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes/{harness.episode.episode_id}/merge"
    )
    assert concurrent.status_code == 409
    assert "active" in concurrent.json()["detail"]

    harness.store.fail_agent_task(first.operation_id, "The merge provider exited early.")
    for action in ("resume", "retry"):
        recovery = harness.client.post(
            f"/api/projects/{harness.project_id}/tasks/{first.operation_id}/{action}"
        )
        assert recovery.status_code == 409
        assert "new Merge" in recovery.json()["detail"]

    redispatched = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes/{harness.episode.episode_id}/merge"
    )
    assert redispatched.status_code == 202, redispatched.text
    assert len(held) == 2
    assert held[1].operation_id != first.operation_id
    assert redispatched.json()["graph_branch"]["active_merge_task_id"] == held[1].operation_id
    assert harness.store.episode_budget_meter(harness.episode.episode_id) == budget_before


@pytest.mark.parametrize(
    ("change", "ended"),
    [("evidence", False), ("none", True)],
)
def test_merge_refuses_an_active_or_unchanged_branch(
    manifest,
    tmp_path: Path,
    change: BranchChange,
    ended: bool,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change=change, ended=ended)

    summary = _episode_payload(harness)["graph_branch"]
    assert summary["merge_eligible"] is False
    refused = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes/{harness.episode.episode_id}/merge"
    )

    assert refused.status_code == 409
    assert "not merge eligible" in refused.json()["detail"]
    assert not any(
        task.kind == "branch_merge" for task in harness.store.agent_tasks(harness.project_id)
    )


def test_merge_requires_a_named_member_without_disclosing_nonmember_projects(
    manifest,
    tmp_path: Path,
) -> None:
    unnamed_app = create_app(str(manifest.path), data_dir=tmp_path / "personal")
    unnamed = TestClient(unnamed_app).post(
        f"/api/projects/{unnamed_app.state.default_project_id}/episodes/{uuid.uuid4()}/merge"
    )
    assert unnamed.status_code == 428
    assert unnamed.json()["detail"]["code"] == "identity_name_required"
    assert (
        unnamed_app.state.background_tasks.store.agent_tasks(unnamed_app.state.default_project_id)
        == []
    )

    data_dir = tmp_path / "team"
    initial_store = AppStore(data_dir / "rcp.sqlite3", space_kind="team")
    creator = initial_store.preprovision_team_member("Project creator")
    outsider = initial_store.preprovision_team_member("Outside researcher")
    acting = [creator.user_id]
    team_app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, opened: opened.space_user(acting[0]),
    )
    team_client = TestClient(team_app)
    created = team_app.state.setup.create(
        ProjectSetupRequest.model_validate(
            _setup_payload(tmp_path / "team-project", name="Private project")
        ),
        seat_member=creator.user_id,
    )
    project_id = str(created["id"])

    acting[0] = outsider.user_id
    nonmember = team_client.post(f"/api/projects/{project_id}/episodes/{uuid.uuid4()}/merge")
    unknown = team_client.post(f"/api/projects/{uuid.uuid4()}/episodes/{uuid.uuid4()}/merge")
    assert nonmember.status_code == unknown.status_code == 404
    assert nonmember.json() == unknown.json() == {"detail": "Project not found"}
    assert team_app.state.background_tasks.store.agent_tasks(project_id) == []


def test_no_change_merge_writes_a_receipt_without_launching_a_provider(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="bookkeeping")
    calls: list[object] = []

    async def forbidden_launch(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("a semantic no-change branch must not launch a provider")
        yield  # pragma: no cover

    monkeypatch.setattr(harness.app.state.launcher, "stream", forbidden_launch)
    main_before = harness.service.history.head_ref()
    budget_before = harness.store.episode_budget_meter(harness.episode.episode_id)
    response = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes/{harness.episode.episode_id}/merge"
    )
    assert response.status_code == 202, response.text
    operation_id = response.json()["graph_branch"]["active_merge_task_id"]
    assert isinstance(operation_id, str)

    task = wait_for_task(harness.store, operation_id, expect="succeeded")
    receipts = harness.branch.merge_receipts()
    assert calls == []
    assert task.applied_revision is None
    assert harness.service.history.head_ref() == main_before
    assert len(receipts) == 1
    assert receipts[0].outcome == "no_change"
    assert receipts[0].result_main_head == main_before
    assert harness.store.episode_budget_meter(harness.episode.episode_id) == budget_before
    settled = _episode_payload(harness)["graph_branch"]
    assert settled["merge_state"] == "merged"
    assert settled["merge_eligible"] is False
    assert settled["latest_successful_merge"]["outcome"] == "no_change"


def _create_main_graph_watcher(harness: _BranchHarness) -> str:
    authorizer = authorized_human(harness.app)
    request = RunRequest(
        chat_id=str(uuid.uuid4()),
        chat_scope="project",
        message="Wait for the merge gate.",
        mode="work",
        trigger="human",
        patch_kind="work",
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
    )
    authority = resolve_dispatch_authority("project_chat", request)
    assert authority is not None
    now = harness.store.now()
    origin = AgentTaskRecord(
        operation_id=str(uuid.uuid4()),
        project_id=harness.project_id,
        kind="project_chat",
        status="queued",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="Queued",
        authorized_by=authorizer,
        dispatch_authority=authority,
    )
    origin = harness.store.create_agent_task(origin)
    watcher_id = str(uuid.uuid4())
    harness.store.create_watchers(
        [
            GraphWatcherRecord(
                watcher_id=watcher_id,
                project_id=harness.project_id,
                origin_operation_id=origin.operation_id,
                origin_task_kind="project_chat",
                chat_id=request.chat_id,
                graph_target=GraphTargetRef(),
                condition=NodeStatusGraphCondition(
                    node_id="blk/merge-ready",
                    status_in=["resolved"],
                ),
                armed_revision=harness.service.history.state().revision,
                continuation=WatcherContinuation(
                    provider="codex",
                    model="",
                    reasoning="medium",
                    run_on="laptop",
                    run_truth_scope=["repo-a"],
                    patch_kind="work",
                ),
                created_at=now,
            )
        ]
    )
    return watcher_id


def test_committed_merge_persists_its_receipt_before_success_and_evaluates_main_watchers(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="status")
    launcher = _PatchWritingLauncher(_candidate_for("status"))
    monkeypatch.setattr(harness.app.state.launcher, "stream", launcher.stream)
    # Keep delivery pending after the condition is completed; this test is about
    # which graph the merge boundary evaluates, not the subsequent chat turn.
    monkeypatch.setattr("rcp.api.app.start_watcher_notification", lambda *_args, **_kwargs: None)
    settled = threading.Event()
    after_task_settled = harness.app.state.background_tasks.on_task_settled
    assert after_task_settled is not None

    def observe_task_settlement(*args, **kwargs) -> None:
        try:
            after_task_settled(*args, **kwargs)
        finally:
            settled.set()

    monkeypatch.setattr(
        harness.app.state.background_tasks,
        "on_task_settled",
        observe_task_settlement,
    )
    watcher_id = _create_main_graph_watcher(harness)
    budget_before = harness.store.episode_budget_meter(harness.episode.episode_id)
    response = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes/{harness.episode.episode_id}/merge"
    )
    assert response.status_code == 202, response.text
    operation_id = response.json()["graph_branch"]["active_merge_task_id"]
    assert isinstance(operation_id, str)

    task = wait_for_task(harness.store, operation_id, expect="succeeded")
    assert launcher.calls == 1
    assert task.applied_revision == harness.service.history.state().revision
    assert harness.service.history.state().nodes["blk/merge-ready"].status == "resolved"
    receipts = harness.branch.merge_receipts()
    assert len(receipts) == 1
    assert receipts[0].outcome == "committed"
    assert receipts[0].result_main_head.revision == task.applied_revision
    categories = [item.category for item in harness.store.agent_task_receipts(operation_id)]
    assert categories.index("branch_merge_outcome") < categories.index("operation_completed")
    assert harness.store.episode_budget_meter(harness.episode.episode_id) == budget_before
    assert settled.wait(TASK_SETTLE_TIMEOUT), "branch merge settlement callback did not finish"

    watcher = harness.store.watcher(watcher_id)
    assert isinstance(watcher, GraphWatcherRecord)
    assert watcher.graph_target == GraphTargetRef()
    assert watcher.status == "completed"
    assert watcher.notified is False


def test_graph_projection_reconciles_a_crashed_commit_and_blocks_duplicate_merge(
    manifest,
    tmp_path: Path,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="evidence")
    metadata = harness.branch.branch_metadata()
    merge_id = branch_merge_id(metadata)
    main_head = harness.service.history.head_ref()
    merge_task_id = str(uuid.uuid4())
    provenance = BranchMergeProvenance(
        merge_id=merge_id,
        branch_id=metadata.branch_id,
        episode_id=metadata.episode_id,
        branch_base_head=metadata.base_head,
        branch_head=metadata.head,
        rebased_main_head=main_head,
        merge_task_id=merge_task_id,
    )
    committed = _branch_patch(merge_task_id, "evidence").model_copy(
        update={
            "profile": "orchestrator",
            "task_id": merge_task_id,
            "episode_id": metadata.episode_id,
            "authorized_by": authorized_human(harness.app),
            "branch_merge": provenance,
        }
    )
    appended, _result = append_fixture_patch(harness.service, committed)
    assert appended.transition is not None

    duplicate_candidate = committed.model_copy(
        update={
            "revision": 0,
            "transition": None,
            "admission": "accepted",
            "admission_messages": [],
        }
    )
    with pytest.raises(BranchMergeAlreadyCommitted) as duplicate_commit:
        harness.service.history.append(
            duplicate_candidate,
            expected_revision=appended.revision,
        )
    assert duplicate_commit.value.patch == appended
    assert harness.service.history.head_ref().revision == appended.revision

    receipt_path = (
        harness.service.manifest.research_dir
        / "branches"
        / metadata.branch_id
        / "merges"
        / f"{merge_id}.json"
    )
    assert not receipt_path.exists()

    summary = _episode_payload(harness)["graph_branch"]
    assert not receipt_path.exists(), "a GET must not publish a crash-recovery receipt"
    assert summary["merge_state"] == "merged"
    assert summary["merge_eligible"] is False
    assert summary["latest_successful_merge"]["outcome"] == "committed"
    assert summary["latest_successful_merge"]["result_main_head"] == {
        "target": {"kind": "main", "branch_id": None},
        "revision": appended.revision,
        "transition_id": appended.transition.transition_id,
    }
    duplicate = harness.client.post(
        f"/api/projects/{harness.project_id}/episodes/{harness.episode.episode_id}/merge"
    )
    assert duplicate.status_code == 409
    assert not any(
        task.kind == "branch_merge" for task in harness.store.agent_tasks(harness.project_id)
    )


def test_crash_receipt_reconciliation_accepts_later_main_history(
    manifest,
    tmp_path: Path,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="evidence")
    metadata = harness.branch.branch_metadata()
    merge_task_id = str(uuid.uuid4())
    provenance = BranchMergeProvenance(
        merge_id=branch_merge_id(metadata),
        branch_id=metadata.branch_id,
        episode_id=metadata.episode_id,
        branch_base_head=metadata.base_head,
        branch_head=metadata.head,
        rebased_main_head=harness.service.history.head_ref(),
        merge_task_id=merge_task_id,
    )
    candidate = _branch_patch(merge_task_id, "evidence").model_copy(
        update={
            "profile": "orchestrator",
            "task_id": merge_task_id,
            "episode_id": metadata.episode_id,
            "authorized_by": authorized_human(harness.app),
            "branch_merge": provenance,
        }
    )
    committed, _ = append_fixture_patch(harness.service, candidate)
    assert committed.transition is not None
    receipt = BranchMergeReceipt(
        outcome="committed",
        provenance=provenance,
        result_main_head=GraphHeadRef(
            revision=committed.revision,
            transition_id=committed.transition.transition_id,
        ),
        authorized_by=committed.authorized_by,
        created_at=committed.created_at,
    )

    later, later_result = append_fixture_patch(
        harness.service,
        Patch(
            kind="refresh",
            author="agent",
            summary="Advanced main after the merge receipt was committed.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "ev/later-main-result",
                            "type": "evidence",
                            "title": "Later main result",
                            "observation": "Main legitimately advanced after the branch merge.",
                            "origin": "internal_run",
                        }
                    ],
                }
            ],
        ),
    )
    assert later.revision == committed.revision + 1

    execution = AgentTaskExecution(
        operation_id=merge_task_id,
        store=harness.store,
        control=AgentProcessControl(),
    )
    _apply_receipt_to_execution(harness.service, execution, receipt)

    assert execution.applied_revision == committed.revision
    assert execution.applied_graph_state == later_result.state


def test_historical_receipt_does_not_cover_a_later_branch_head(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="evidence")
    task = _admit_held_merge_task(harness, monkeypatch)
    historical = harness.branch.write_merge_receipt(
        _current_receipt(harness, task_id=task.operation_id)
    )
    harness.store.complete_agent_task(task.operation_id, applied_revision=None, result={})
    previous_head = harness.branch.head_ref()
    harness.branch.append(
        _branch_patch(harness.root.operation_id, "evidence", suffix="-later"),
        expected_revision=previous_head.revision,
    )

    summary = _episode_payload(harness)["graph_branch"]
    assert summary["head"]["revision"] == previous_head.revision + 1
    assert summary["merge_state"] == "unmerged"
    assert summary["merge_eligible"] is True
    assert summary["latest_successful_merge"]["provenance"]["merge_id"] == (
        historical.provenance.merge_id
    )


def test_receipt_reconciliation_rejects_a_filename_content_mismatch(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="evidence")
    task = _admit_held_merge_task(harness, monkeypatch)
    receipt = harness.branch.write_merge_receipt(
        _current_receipt(harness, task_id=task.operation_id)
    )
    merge_id = receipt.provenance.merge_id
    wrong_id = "0" * 64 if merge_id != "0" * 64 else "1" * 64
    merges_dir = (
        harness.service.manifest.research_dir / "branches" / harness.episode.episode_id / "merges"
    )
    (merges_dir / f"{merge_id}.json").replace(merges_dir / f"{wrong_id}.json")

    with pytest.raises(ValueError, match="name disagrees with its content"):
        harness.branch.reconcile_merge_receipt(wrong_id)


def test_no_change_receipt_rechecks_live_project_membership(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _create_branch_harness(manifest, tmp_path, change="evidence")
    task = _admit_held_merge_task(harness, monkeypatch)
    harness.service.history.project_membership_check = lambda _project_id, _user_id: False

    with pytest.raises(ValueError, match="not a member"):
        harness.branch.write_merge_receipt(_current_receipt(harness, task_id=task.operation_id))

    assert harness.branch.merge_receipts() == []
