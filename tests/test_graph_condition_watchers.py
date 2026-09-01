from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from rcp.agents import AgentEvent, AgentProcessControl
from rcp.background import AgentTaskExecution, BackgroundAgentTasks
from rcp.core.materialize import MaterializationResult
from rcp.core.models import (
    AuthorizedHuman,
    Blocker,
    GatedCard,
    GraphState,
    Hypothesis,
    Patch,
    Proposal,
)
from rcp.core.transition_models import GraphTargetRef
from rcp.runs.experiment_loop import persist_experiment_watchers_idempotently
from rcp.runs.shared import _record_patch_applied_receipt
from rcp.runs.watcher_admission import start_watcher_notification
from rcp.service import RunRequest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    ProposalResolvedGraphCondition,
    StoredWatcherRecord,
    WatcherContinuation,
    WatcherRecord,
    WatcherStopRequest,
)
from rcp.transport import StateUnavailable
from rcp.watchers import (
    WatcherBinding,
    WatcherCheckResult,
    WatcherInitialCheckError,
    WatcherPoller,
    WatcherRetryWorker,
    WatchSpec,
    arm_watchers,
    evaluate_graph_watchers,
    graph_condition_result,
    ready_graph_watcher_groups,
)

from .helpers import append_fixture_patch
from .helpers import create_named_app as create_app

_CREATED_AT = "2026-08-12T00:00:00+00:00"


def _blocker(status: str = "open") -> Blocker:
    return Blocker(
        id="blk/foo",
        type="blocker",
        title="Waiting on a canonical result",
        description="The result is not ready yet.",
        status=status,
    )


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        id="hyp/foo",
        type="hypothesis",
        title="The intervention works",
        statement="The intervention improves the measured outcome.",
    )


def _proposal(status: str = "pending") -> Proposal:
    return Proposal(
        id="prop/foo",
        title="Resolve the hypothesis",
        card=GatedCard(),
        ops=[],
        related_node_ids=["hyp/foo"],
        base_rev=1,
        status=status,
        resolved_rev=2 if status != "pending" else None,
    )


def _state(
    *,
    blocker_status: str = "open",
    proposal_status: str = "pending",
    revision: int = 1,
    replay_status: str = "complete",
    include_blocker: bool = True,
    include_hypothesis: bool = True,
) -> GraphState:
    nodes = {}
    if include_blocker:
        blocker = _blocker(blocker_status)
        nodes[blocker.id] = blocker
    if include_hypothesis:
        hypothesis = _hypothesis()
        nodes[hypothesis.id] = hypothesis
    return GraphState(
        revision=revision,
        nodes=nodes,
        proposals={"prop/foo": _proposal(proposal_status)},
        replay_status=replay_status,
    )


def _canonical_fixture_patch(*, blocker_status: str) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Created the graph-condition watcher fixture.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "exp/one",
                        "type": "experiment",
                        "title": "Bounded graph wake",
                        "objective": "Wait for the canonical blocker resolution.",
                        "completion_criteria": ["The blocker is resolved."],
                        "invocation_ceiling": 2,
                    },
                    {
                        "id": "blk/foo",
                        "type": "blocker",
                        "title": "Waiting on a canonical result",
                        "description": "The result is not ready yet.",
                        "status": blocker_status,
                    },
                ],
            }
        ],
    )


def _blocker_status_patch(status: str) -> Patch:
    return Patch(
        kind="work",
        author="agent",
        summary=f"Set the blocker status to {status}.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [{"id": "blk/foo", "changes": {"status": status}}],
            }
        ],
    )


def _test_authorizer(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name or "Test researcher",
    )


def _wait_for_terminal_task(store: AppStore, operation_id: str) -> AgentTaskRecord:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        task = store.agent_task(operation_id)
        assert task is not None
        if task.status in {"succeeded", "failed", "paused", "stopped"}:
            return task
        time.sleep(0.01)
    raise AssertionError(f"watcher wake {operation_id} did not settle")


def _wait_until(predicate, *, detail: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(detail)


def _continuation() -> WatcherContinuation:
    return WatcherContinuation(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo-a"],
        patch_kind="work",
    )


def _binding(*, continuation: WatcherContinuation | None = None) -> WatcherBinding:
    return WatcherBinding(
        project_id="project",
        origin_operation_id="origin",
        origin_task_kind="node_chat",
        chat_id="chat",
        node_id="exp/one",
        continuation=continuation or _continuation(),
    )


def _graph_record(
    watcher_id: str,
    condition: NodeStatusGraphCondition | ProposalResolvedGraphCondition,
    *,
    continuation: WatcherContinuation | None = None,
    status: str = "active",
) -> GraphWatcherRecord:
    return GraphWatcherRecord(
        watcher_id=watcher_id,
        project_id="project",
        origin_operation_id="origin",
        origin_task_kind="node_chat",
        chat_id="chat",
        node_id="exp/one",
        condition=condition,
        armed_revision=1,
        continuation=continuation or _continuation(),
        status=status,
        created_at=_CREATED_AT,
        completed_at=_CREATED_AT if status == "completed" else None,
    )


def _external_record(
    watcher_id: str,
    *,
    continuation: WatcherContinuation | None = None,
    status: str = "active",
) -> WatcherRecord:
    return WatcherRecord(
        watcher_id=watcher_id,
        project_id="project",
        origin_operation_id="origin",
        origin_task_kind="node_chat",
        chat_id="chat",
        node_id="exp/one",
        check_command="test -f /tmp/result.done",
        log_path="/tmp/result.log",
        cwd="/tmp",
        continuation=continuation or _continuation(),
        status=status,
        created_at=_CREATED_AT,
        completed_at=_CREATED_AT if status == "completed" else None,
    )


def _notification_task(
    store: AppStore,
    operation_id: str,
    watcher_ids: list[str],
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id="project",
        kind="node_chat",
        status="queued",
        request={
            "chat_id": "chat",
            "node_id": "exp/one",
            "provider": "codex",
            "model": "gpt-5",
            "reasoning": "medium",
            "run_on": "laptop",
            "run_truth_scope": ["repo-a"],
            "mode": "work",
            "trigger": "watcher",
            "patch_kind": "work",
            "workflow_ids": [],
            "skill_ids": [],
            "invoked_workflow_ids": [],
            "invoked_skill_ids": [],
            "resolved_skill_packages": [],
            "watcher_ids": watcher_ids,
        },
        created_at=now,
        updated_at=now,
        status_message="Queued graph-condition wake.",
    )


def _watcher_ids(groups: list[list[StoredWatcherRecord]]) -> list[list[str]]:
    return [[record.watcher_id for record in group] for group in groups]


def test_watch_json_validation_is_all_or_none_across_external_and_graph(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    spec = WatchSpec(check_command="still-running", log_path="/tmp/run.log", cwd="/tmp")
    valid = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    invalid = NodeStatusGraphCondition(node_id="blk/foo", status_in=["completed"])
    checked: list[str] = []

    def active_check(
        candidate: WatchSpec,
        _host: str,
        _timeout: float,
    ) -> WatcherCheckResult:
        checked.append(candidate.check_command)
        return WatcherCheckResult(state="active", checked_at=_CREATED_AT, exit_code=1)

    with pytest.raises(ValueError, match="invalid statuses"):
        arm_watchers(
            store,
            [spec],
            _binding(),
            graph_conditions=[invalid],
            state=_state(),
            check_runner=active_check,
        )
    assert checked == []
    assert store.watchers("project") == []

    def failed_check(
        _candidate: WatchSpec,
        _host: str,
        _timeout: float,
    ) -> WatcherCheckResult:
        return WatcherCheckResult(
            state="error",
            checked_at=_CREATED_AT,
            exit_code=2,
            error="scheduler unavailable",
        )

    with pytest.raises(WatcherInitialCheckError, match="scheduler unavailable"):
        arm_watchers(
            store,
            [spec],
            _binding(),
            graph_conditions=[valid],
            state=_state(),
            check_runner=failed_check,
        )
    assert store.watchers("project") == []

    armed = arm_watchers(
        store,
        [spec],
        _binding(),
        graph_conditions=[valid],
        state=_state(),
        check_runner=active_check,
    )
    assert {type(record) for record in armed} == {WatcherRecord, GraphWatcherRecord}
    assert len(store.watchers("project")) == 2


def test_supplied_watcher_ids_follow_external_then_graph_order(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    spec = WatchSpec(check_command="still-running", log_path="/tmp/run.log", cwd="/tmp")
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])

    def active_check(
        _candidate: WatchSpec,
        _host: str,
        _timeout: float,
    ) -> WatcherCheckResult:
        return WatcherCheckResult(state="active", checked_at=_CREATED_AT, exit_code=1)

    records = arm_watchers(
        store,
        [spec],
        _binding(),
        graph_conditions=[condition],
        state=_state(),
        watcher_ids=["external-id", "graph-id"],
        check_runner=active_check,
    )

    assert [record.watcher_id for record in records] == ["external-id", "graph-id"]
    assert isinstance(records[0], WatcherRecord)
    assert isinstance(records[1], GraphWatcherRecord)


@pytest.mark.parametrize(
    "watcher_ids",
    [[], ["only-one"], ["same", "same"], ["external", "  "]],
)
def test_invalid_supplied_watcher_ids_fail_before_checks_or_insertion(
    tmp_path,
    watcher_ids: list[str],
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    spec = WatchSpec(check_command="still-running", log_path="/tmp/run.log", cwd="/tmp")
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    checked: list[str] = []

    def active_check(
        candidate: WatchSpec,
        _host: str,
        _timeout: float,
    ) -> WatcherCheckResult:
        checked.append(candidate.check_command)
        return WatcherCheckResult(state="active", checked_at=_CREATED_AT, exit_code=1)

    with pytest.raises(ValueError, match="watcher_ids"):
        arm_watchers(
            store,
            [spec],
            _binding(),
            graph_conditions=[condition],
            state=_state(),
            watcher_ids=watcher_ids,
            check_runner=active_check,
        )

    assert checked == []
    assert store.watchers("project") == []


def test_initially_satisfied_graph_condition_is_immediately_ready(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])

    armed = arm_watchers(
        store,
        [],
        _binding(),
        graph_conditions=[condition],
        state=_state(blocker_status="resolved"),
    )

    assert len(armed) == 1
    assert isinstance(armed[0], GraphWatcherRecord)
    assert armed[0].status == "completed"
    assert armed[0].last_evaluated_at is not None
    assert _watcher_ids(ready_graph_watcher_groups(store, "project")) == [[armed[0].watcher_id]]


def test_experiment_graph_handoff_persists_deterministically_and_idempotently(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    root = _experiment_task(
        store,
        "loop-root",
        episode_id=episode_id,
        invocation=1,
        ceiling=2,
        watcher_ids=[],
    )
    store.create_experiment_episode_with_invocation(root)
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    continuation = _continuation().model_copy(
        update={
            "patch_kind": "experiment_loop",
            "control_node_id": "exp/one",
            "control_revision": 1,
            "control_episode_id": episode_id,
            "control_invocation": 1,
            "control_invocation_ceiling": 2,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
        }
    )
    binding = _binding(continuation=continuation).model_copy(
        update={"origin_operation_id": root.operation_id}
    )
    execution = AgentTaskExecution(
        operation_id=root.operation_id,
        store=store,
        control=AgentProcessControl(),
    )
    condition = ProposalResolvedGraphCondition(node_id="hyp/foo", proposal_resolved=True)

    first = persist_experiment_watchers_idempotently(
        execution,
        [],
        [],
        binding,
        graph_conditions=[condition],
        graph_state=_state(),
        armed_revision=1,
    )
    repeated = persist_experiment_watchers_idempotently(
        execution,
        [],
        [],
        binding,
        graph_conditions=[condition],
        graph_state=_state(),
        armed_revision=1,
    )

    assert len(first) == 1
    assert isinstance(first[0], GraphWatcherRecord)
    assert first[0].watcher_id == repeated[0].watcher_id
    assert first[0].condition == condition
    assert first[0].episode_id == episode_id
    assert len(store.watchers("project")) == 1


def test_experiment_same_patch_resolution_uses_pre_patch_arming_baseline(tmp_path) -> None:
    store = AppStore(tmp_path / "same-patch.sqlite3")
    episode_id = str(uuid.uuid4())
    root = _experiment_task(
        store,
        "same-patch-root",
        episode_id=episode_id,
        invocation=1,
        ceiling=2,
        watcher_ids=[],
    )
    store.create_experiment_episode_with_invocation(root)
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    continuation = _continuation().model_copy(
        update={
            "patch_kind": "experiment_loop",
            "control_node_id": "exp/one",
            "control_revision": 1,
            "control_episode_id": episode_id,
            "control_invocation": 1,
            "control_invocation_ceiling": 2,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
        }
    )
    execution = AgentTaskExecution(
        operation_id=root.operation_id,
        store=store,
        control=AgentProcessControl(),
    )
    condition = ProposalResolvedGraphCondition(node_id="hyp/foo", proposal_resolved=True)

    armed = persist_experiment_watchers_idempotently(
        execution,
        [],
        [],
        _binding(continuation=continuation).model_copy(
            update={"origin_operation_id": root.operation_id}
        ),
        graph_conditions=[condition],
        graph_state=_state(proposal_status="approved", revision=2),
        armed_revision=1,
    )

    assert len(armed) == 1
    stored = armed[0]
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.armed_revision == 1
    assert stored.status == "completed"


def test_status_and_proposal_conditions_use_the_closed_vocabulary() -> None:
    status = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    proposal = ProposalResolvedGraphCondition(node_id="hyp/foo", proposal_resolved=True)

    assert graph_condition_result(status, _state(), armed_revision=1) == "active"
    assert (
        graph_condition_result(
            status,
            _state(blocker_status="resolved"),
            armed_revision=1,
        )
        == "completed"
    )
    assert graph_condition_result(proposal, _state(), armed_revision=1) == "active"
    assert (
        graph_condition_result(
            proposal,
            _state(proposal_status="approved", revision=2),
            armed_revision=1,
        )
        == "completed"
    )


def test_proposal_resolution_must_happen_after_the_watcher_was_armed(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    condition = ProposalResolvedGraphCondition(node_id="hyp/foo", proposal_resolved=True)
    store.create_watchers(
        [_graph_record("proposal-after-arm", condition).model_copy(update={"armed_revision": 2})]
    )
    old_resolved = _proposal("approved").model_copy(
        update={"id": "prop/old", "base_rev": 0, "resolved_rev": 1}
    )
    new_pending = _proposal().model_copy(update={"id": "prop/new", "base_rev": 2})
    armed_state = _state(revision=3).model_copy(
        update={"proposals": {old_resolved.id: old_resolved, new_pending.id: new_pending}}
    )

    assert evaluate_graph_watchers(store, "project", armed_state) == []
    active = store.watcher("proposal-after-arm")
    assert isinstance(active, GraphWatcherRecord)
    assert active.status == "active"

    newly_resolved = new_pending.model_copy(update={"status": "approved", "resolved_rev": 4})
    resolved_state = armed_state.model_copy(
        update={
            "revision": 4,
            "proposals": {old_resolved.id: old_resolved, newly_resolved.id: newly_resolved},
        }
    )

    assert _watcher_ids(evaluate_graph_watchers(store, "project", resolved_state)) == [
        ["proposal-after-arm"]
    ]
    completed = store.watcher("proposal-after-arm")
    assert isinstance(completed, GraphWatcherRecord)
    assert completed.status == "completed"


def test_unrelated_canonical_movement_does_not_fire_a_condition(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    store.create_watchers([_graph_record("graph", condition)])

    groups = evaluate_graph_watchers(store, "project", _state(revision=9))

    assert groups == []
    stored = store.watcher("graph")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "active"
    assert stored.notified is False
    assert stored.last_evaluated_at is not None


def test_human_sync_boundary_claims_a_graph_wake_and_spends_experiment_budget(
    manifest,
    tmp_path,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, _canonical_fixture_patch(blocker_status="open"))
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    episode_id = str(uuid.uuid4())
    root = _experiment_task(
        store,
        "sync-loop-root",
        episode_id=episode_id,
        invocation=1,
        ceiling=2,
        watcher_ids=[],
    ).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": _test_authorizer(store),
        }
    )
    root.request["control_revision"] = service.history.state().revision
    store.create_experiment_episode_with_invocation(root)
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    stage = tmp_path / "loop-stage"
    stage.mkdir()
    store.commit_experiment_episode_turn(
        episode_id=episode_id,
        project_id=project_id,
        control_node_id="exp/one",
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="native-sync-loop",
        stage_host=None,
        stage_root=str(stage),
        chat_id="chat",
        operation_id=root.operation_id,
        invocation=1,
        graph_result="no graph change",
        watcher_ids=[],
        context_baseline={},
    )
    continuation = _continuation().model_copy(
        update={
            "patch_kind": "experiment_loop",
            "control_node_id": "exp/one",
            "control_revision": service.history.state().revision,
            "control_episode_id": episode_id,
            "control_invocation": 1,
            "control_invocation_ceiling": 2,
            "control_decision_bundle": [],
            "control_completion_criteria": ["The blocker is resolved."],
        }
    )
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    graph_record = _graph_record(
        "sync-graph",
        condition,
        continuation=continuation,
    ).model_copy(
        update={
            "project_id": project_id,
            "origin_operation_id": root.operation_id,
        }
    )
    store.create_watchers([graph_record])

    async def settle_wake(_project_id, _kind, _request, _execution):
        yield f"data: {AgentEvent(event='answer', text='Observed the Sync.').model_dump_json()}\n\n"
        yield f"data: {AgentEvent(event='done').model_dump_json()}\n\n"

    app.state.background_tasks.stream = settle_wake
    blocker = service.history.state().nodes["blk/foo"]
    client = TestClient(app)
    try:
        response = client.post(
            f"/api/projects/{project_id}/sync",
            json={
                "base_revision": service.history.state().revision,
                "nodes": [
                    {
                        "node_id": blocker.id,
                        "base_updated_rev": blocker.updated_rev,
                        "changes": {"status": "resolved"},
                    }
                ],
            },
        )

        assert response.status_code == 200, response.text
        stored = store.watcher("sync-graph")
        assert isinstance(stored, GraphWatcherRecord)
        assert stored.status == "completed"
        assert stored.notified is True
        assert stored.notification_operation_id is not None
        wake = _wait_for_terminal_task(store, stored.notification_operation_id)
        assert wake.status == "succeeded"
        assert wake.request["trigger"] == "watcher"
        assert wake.request["control_invocation"] == 2
        assert wake.request["watcher_ids"] == ["sync-graph"]
        runtime = store.experiment_loop_runtime(project_id, "exp/one")
        assert runtime.invocations_used == 2
        assert runtime.invocation_ceiling - runtime.invocations_used == 0
    finally:
        client.close()
        app.state.background_tasks.shutdown()


def test_agent_settlement_evaluates_the_exact_applied_revision_boundary(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, _canonical_fixture_patch(blocker_status="open"))
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    origin = _notification_task(store, "agent-boundary-origin", []).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": _test_authorizer(store),
            "request": {
                **_notification_task(store, "unused", []).request,
                "trigger": "human",
                "watcher_ids": [],
            },
        }
    )
    store.create_agent_task(origin)
    store.complete_agent_task(origin.operation_id, applied_revision=None, result={})
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    store.create_watchers(
        [
            _graph_record("agent-boundary", condition).model_copy(
                update={
                    "project_id": project_id,
                    "origin_operation_id": origin.operation_id,
                }
            )
        ]
    )
    applied, applied_result = append_fixture_patch(
        service,
        _blocker_status_patch("resolved"),
    )
    assert applied.revision is not None
    append_fixture_patch(service, _blocker_status_patch("open"))
    assert service.history.state().nodes["blk/foo"].status == "open"
    deliveries: list[list[str]] = []

    def capture_delivery(
        _tasks,
        _project_id,
        _kind,
        _request,
        watcher_ids,
        **_kwargs,
    ):
        deliveries.append(watcher_ids)
        return None

    monkeypatch.setattr("rcp.api.app.start_watcher_notification", capture_delivery)
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None
    request = RunRequest(
        chat_scope="node",
        chat_id="chat",
        node_id="exp/one",
        message="Resolve the blocker.",
        mode="work",
    )
    callback(
        project_id,
        "node_chat",
        request,
        AgentTaskExecution(
            operation_id=origin.operation_id,
            store=store,
            control=AgentProcessControl(),
            applied_revision=applied.revision,
            applied_graph_state=applied_result.state,
        ),
    )

    assert deliveries == [["agent-boundary"]]
    stored = store.watcher("agent-boundary")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "completed"
    app.state.background_tasks.shutdown()


def test_delayed_historical_callback_cannot_complete_a_newer_watcher(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    historical_true = _state(blocker_status="resolved", revision=2)
    armed_false = _state(blocker_status="open", revision=3)
    store.create_watchers(
        [
            _graph_record(
                "newer-false",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(update={"project_id": project_id, "armed_revision": armed_false.revision})
        ]
    )
    monkeypatch.setattr(
        app.state.service.history,
        "accepted_boundary_states",
        lambda: (
            MaterializationResult(state=armed_false),
            [historical_true, armed_false],
        ),
    )
    monkeypatch.setattr(
        "rcp.api.app.ready_graph_watcher_groups",
        lambda *_args, **_kwargs: [],
    )
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="delayed-r2",
            store=store,
            control=AgentProcessControl(),
            applied_revision=historical_true.revision,
            applied_graph_state=historical_true,
        ),
    )

    stored = store.watcher("newer-false")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "active"
    app.state.background_tasks.shutdown()


def test_historical_absence_cannot_retire_a_target_created_then_armed(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    before_creation = _state(include_blocker=False, revision=1)
    created_and_armed = _state(blocker_status="open", revision=2)
    store.create_watchers(
        [
            _graph_record(
                "created-then-armed",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(
                update={
                    "project_id": project_id,
                    "armed_revision": created_and_armed.revision,
                }
            )
        ]
    )
    monkeypatch.setattr(
        app.state.service.history,
        "accepted_boundary_states",
        lambda: (
            MaterializationResult(state=created_and_armed),
            [before_creation, created_and_armed],
        ),
    )
    monkeypatch.setattr(
        "rcp.api.app.ready_graph_watcher_groups",
        lambda *_args, **_kwargs: [],
    )
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="delayed-r1",
            store=store,
            control=AgentProcessControl(),
            applied_revision=before_creation.revision,
            applied_graph_state=before_creation,
        ),
    )

    stored = store.watcher("created-then-armed")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "active"
    app.state.background_tasks.shutdown()


def test_no_patch_arming_settlement_catches_revision_between_validation_and_insert(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    _initial, initial_result = append_fixture_patch(
        service,
        _canonical_fixture_patch(blocker_status="open"),
    )
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store

    # Validation observed the open R1 state. R2 committed before the durable
    # insert, so its original boundary handling could not see this row.
    append_fixture_patch(service, _blocker_status_patch("resolved"))
    store.create_watchers(
        [
            _graph_record(
                "insert-race",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(
                update={
                    "project_id": project_id,
                    "armed_revision": initial_result.state.revision,
                }
            )
        ]
    )
    monkeypatch.setattr(
        "rcp.api.app.ready_graph_watcher_groups",
        lambda *_args, **_kwargs: [],
    )
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="arming-without-patch",
            store=store,
            control=AgentProcessControl(),
            armed_graph_watchers=True,
        ),
    )

    stored = store.watcher("insert-race")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "completed"
    app.state.background_tasks.shutdown()


def test_degraded_final_replay_applies_no_satisfying_prefix_transition(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    satisfying_prefix = _state(blocker_status="resolved", revision=2)
    degraded_head = _state(
        blocker_status="resolved",
        revision=3,
        replay_status="degraded",
    )
    store.create_watchers(
        [
            _graph_record(
                "degraded-prefix",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(update={"project_id": project_id})
        ]
    )
    replay_calls = 0
    retry_passed = threading.Event()

    def degraded_replay():
        nonlocal replay_calls
        replay_calls += 1
        return (
            MaterializationResult(state=degraded_head),
            [satisfying_prefix],
        )

    def observe_ready_pass(candidate_store, candidate_project_id):
        retry_passed.set()
        return ready_graph_watcher_groups(candidate_store, candidate_project_id)

    monkeypatch.setattr(
        app.state.service.history,
        "accepted_boundary_states",
        degraded_replay,
    )
    monkeypatch.setattr("rcp.api.app.ready_graph_watcher_groups", observe_ready_pass)
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="degraded-r3",
            store=store,
            control=AgentProcessControl(),
            applied_revision=degraded_head.revision,
            applied_graph_state=degraded_head,
        ),
    )

    stored = store.watcher("degraded-prefix")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "active"
    assert stored.last_evaluated_at is None
    assert replay_calls == 1

    # A degraded replay is fail-closed until a real boundary/startup trigger;
    # the periodic worker retries ready delivery only for this project.
    retry_passed.clear()
    app.state.graph_watcher_retry_worker.start()
    try:
        app.state.graph_watcher_retry_worker.signal()
        assert retry_passed.wait(1), "periodic graph retry pass did not run"
        assert replay_calls == 1
    finally:
        app.state.graph_watcher_retry_worker.stop()
        app.state.background_tasks.shutdown()


@pytest.mark.parametrize("refresh_failure", ["exception", "false-return"])
def test_remote_refresh_failure_never_evaluates_the_stale_graph_mirror(
    manifest,
    tmp_path,
    monkeypatch,
    refresh_failure,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, _canonical_fixture_patch(blocker_status="open"))
    armed_revision = service.history.state().revision
    append_fixture_patch(service, _blocker_status_patch("resolved"))
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    store.create_watchers(
        [
            _graph_record(
                "stale-remote",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(update={"project_id": project_id, "armed_revision": armed_revision})
        ]
    )

    def unavailable() -> bool:
        if refresh_failure == "exception":
            raise StateUnavailable("canonical remote is unavailable")
        return False

    monkeypatch.setattr(service.history.workspace, "refresh_if_stale", unavailable)
    monkeypatch.setattr(
        "rcp.api.app.ready_graph_watcher_groups",
        lambda *_args, **_kwargs: [],
    )
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="stale-remote-trigger",
            store=store,
            control=AgentProcessControl(),
            applied_revision=service.history.state().revision,
        ),
    )

    stored = store.watcher("stale-remote")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "active"
    assert stored.last_evaluated_at is None
    app.state.background_tasks.shutdown()


@pytest.mark.parametrize(
    "transient_failure",
    [
        StateUnavailable("canonical remote is temporarily unavailable"),
        sqlite3.OperationalError("database is locked"),
    ],
)
def test_transient_reconciliation_failure_retries_without_a_new_revision(
    manifest,
    tmp_path,
    monkeypatch,
    transient_failure,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, _canonical_fixture_patch(blocker_status="open"))
    armed_revision = service.history.state().revision
    append_fixture_patch(service, _blocker_status_patch("resolved"))
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    authorizer = _test_authorizer(store)
    origin = _notification_task(store, "retry-origin", []).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": authorizer,
            "request": {
                **_notification_task(store, "unused", []).request,
                "trigger": "human",
                "watcher_ids": [],
            },
        }
    )
    store.create_agent_task(origin)
    store.complete_agent_task(origin.operation_id, applied_revision=None, result={})
    store.create_watchers(
        [
            _graph_record(
                "reconcile-retry",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(
                update={
                    "project_id": project_id,
                    "origin_operation_id": origin.operation_id,
                    "armed_revision": armed_revision,
                }
            )
        ]
    )
    original_replay = service.history.accepted_boundary_states
    replay_calls = 0

    def unavailable_once():
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls == 1:
            raise transient_failure
        return original_replay()

    deliveries: list[list[str]] = []

    def claim_delivery(_tasks, _project_id, _kind, _request, watcher_ids, **_kwargs):
        deliveries.append(watcher_ids)
        task = _notification_task(store, "reconciliation-retry-wake", watcher_ids).model_copy(
            update={"project_id": project_id, "authorized_by": authorizer}
        )
        return store.create_watcher_notification_task(task, watcher_ids)

    monkeypatch.setattr(service.history, "accepted_boundary_states", unavailable_once)
    monkeypatch.setattr("rcp.api.app.start_watcher_notification", claim_delivery)
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None
    revision = service.history.state().revision

    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="failed-reconciliation",
            store=store,
            control=AgentProcessControl(),
            applied_revision=revision,
        ),
    )
    first = store.watcher("reconcile-retry")
    assert isinstance(first, GraphWatcherRecord)
    assert first.status == "active"

    app.state.graph_watcher_retry_worker.start()
    try:
        app.state.graph_watcher_retry_worker.signal()
        _wait_until(lambda: bool(deliveries), detail="reconciliation retry did not wake")
        assert replay_calls == 2
        assert deliveries == [["reconcile-retry"]]
        assert service.history.state().revision == revision

        app.state.graph_watcher_retry_worker.signal()
        time.sleep(0.05)
        assert deliveries == [["reconcile-retry"]]
    finally:
        app.state.graph_watcher_retry_worker.stop()
        app.state.background_tasks.shutdown()


def test_due_reconciliation_survives_retry_worker_stop_then_start(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, _canonical_fixture_patch(blocker_status="open"))
    armed_revision = service.history.state().revision
    append_fixture_patch(service, _blocker_status_patch("resolved"))
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    authorizer = _test_authorizer(store)
    origin = _notification_task(store, "generation-retry-origin", []).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": authorizer,
            "request": {
                **_notification_task(store, "unused", []).request,
                "trigger": "human",
                "watcher_ids": [],
            },
        }
    )
    store.create_agent_task(origin)
    store.complete_agent_task(origin.operation_id, applied_revision=None, result={})
    store.create_watchers(
        [
            _graph_record(
                "generation-reconcile-retry",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(
                update={
                    "project_id": project_id,
                    "origin_operation_id": origin.operation_id,
                    "armed_revision": armed_revision,
                }
            )
        ]
    )
    original_replay = service.history.accepted_boundary_states
    replay_calls = 0

    def unavailable_once():
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls == 1:
            raise StateUnavailable("canonical remote is temporarily unavailable")
        return original_replay()

    deliveries: list[list[str]] = []

    def claim_delivery(_tasks, _project_id, _kind, _request, watcher_ids, **_kwargs):
        deliveries.append(watcher_ids)
        task = _notification_task(store, "generation-retry-wake", watcher_ids).model_copy(
            update={"project_id": project_id, "authorized_by": authorizer}
        )
        return store.create_watcher_notification_task(task, watcher_ids)

    monkeypatch.setattr(service.history, "accepted_boundary_states", unavailable_once)
    monkeypatch.setattr("rcp.api.app.start_watcher_notification", claim_delivery)
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None
    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="schedule-generation-retry",
            store=store,
            control=AgentProcessControl(),
            applied_revision=service.history.state().revision,
        ),
    )
    assert replay_calls == 1

    worker = app.state.graph_watcher_retry_worker
    retry_callback = worker.callback
    first_started = threading.Event()
    replacement_started = threading.Event()
    release_first = threading.Event()
    first_finished = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def controlled_retry(generation) -> None:
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            call = call_count
        if call == 1:
            first_started.set()
            release_first.wait()
            retry_callback(generation)
            first_finished.set()
            return
        replacement_started.set()
        first_finished.wait()
        retry_callback(generation)

    worker.callback = controlled_retry
    worker.start()
    try:
        worker.signal()
        assert first_started.wait(1), "first retry generation did not start"
        worker.stop(timeout=0.01)
        worker.start()
        worker.signal()
        assert replacement_started.wait(1), "replacement retry generation did not start"

        # The stale generation selects the due project, then observes its invalid
        # lease. The replacement must still find and process that same due work.
        release_first.set()
        _wait_until(lambda: bool(deliveries), detail="replacement generation lost due retry")
        assert replay_calls == 2
        assert deliveries == [["generation-reconcile-retry"]]
    finally:
        release_first.set()
        worker.stop()
        app.state.background_tasks.shutdown()


def test_ready_only_graph_project_skips_canonical_boundary_replay(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    store.create_watchers(
        [
            _graph_record(
                "ready-fast-path",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
                status="completed",
            ).model_copy(update={"project_id": project_id})
        ]
    )
    replay_calls = 0
    ready_calls = 0

    def unexpected_replay():
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("ready-only retry must not replay canonical history")

    def observe_ready(_store, candidate_project_id):
        nonlocal ready_calls
        assert candidate_project_id == project_id
        ready_calls += 1
        return []

    monkeypatch.setattr(app.state.service.history, "accepted_boundary_states", unexpected_replay)
    monkeypatch.setattr("rcp.api.app.ready_graph_watcher_groups", observe_ready)
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="ready-fast-path",
            store=store,
            control=AgentProcessControl(),
            applied_revision=1,
        ),
    )

    assert replay_calls == 0
    assert ready_calls == 1
    app.state.background_tasks.shutdown()


@pytest.mark.parametrize(
    ("boundaries", "expected_status"),
    [
        (
            [
                _state(blocker_status="resolved", revision=1),
                _state(include_blocker=False, revision=2),
            ],
            "completed",
        ),
        (
            [
                _state(include_blocker=False, revision=1),
                _state(blocker_status="resolved", revision=2),
            ],
            "stopped",
        ),
    ],
    ids=["target-then-removal", "removal-then-recreation"],
)
def test_reversed_task_settlement_uses_canonical_boundary_order(
    manifest,
    tmp_path,
    monkeypatch,
    boundaries,
    expected_status,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    store.create_watchers(
        [
            _graph_record(
                "canonical-order",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(update={"project_id": project_id, "armed_revision": 0})
        ]
    )
    monkeypatch.setattr(
        app.state.service.history,
        "accepted_boundary_states",
        lambda: (MaterializationResult(state=boundaries[-1]), boundaries),
    )
    monkeypatch.setattr(
        "rcp.api.app.ready_graph_watcher_groups",
        lambda *_args, **_kwargs: [],
    )
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    # The later task settles first, but its captured state is only an arrival
    # signal; reconciliation must still visit R1 before R2.
    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="settled-r2-first",
            store=store,
            control=AgentProcessControl(),
            applied_revision=2,
            applied_graph_state=boundaries[-1],
        ),
    )

    stored = store.watcher("canonical-order")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == expected_status
    app.state.background_tasks.shutdown()


def test_repeated_task_settlement_skips_transition_traces_before_durable_cursor(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    store.create_watchers(
        [
            _graph_record(
                "durable-cursor",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ).model_copy(update={"project_id": project_id, "armed_revision": 0})
        ]
    )
    boundaries = [_state(revision=revision) for revision in (1, 2, 3)]
    monkeypatch.setattr(
        app.state.service.history,
        "accepted_boundary_states",
        lambda: (MaterializationResult(state=boundaries[-1]), boundaries),
    )
    traced_revisions: list[int] = []

    def observe_trace(revision: int):
        traced_revisions.append(revision)
        return None

    monkeypatch.setattr(
        app.state.service.history,
        "transition_trace_at_revision",
        observe_trace,
    )
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None
    execution = AgentTaskExecution(
        operation_id="durable-cursor-signal",
        store=store,
        control=AgentProcessControl(),
        applied_revision=3,
        applied_graph_state=boundaries[-1],
    )
    request = RunRequest(
        chat_scope="node",
        chat_id="chat",
        node_id="exp/one",
        mode="work",
    )

    callback(project_id, "node_chat", request, execution)
    callback(project_id, "node_chat", request, execution)

    assert traced_revisions == [1, 2, 3, 3]
    consumed = store.graph_watcher_reconciliation_head(project_id, GraphTargetRef())
    assert consumed is not None and consumed.revision == 3
    assert consumed.transition_id is None
    app.state.background_tasks.shutdown()


def test_task_settlement_retries_durable_ready_deliveries_without_active_conditions(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    retries: list[str] = []

    def capture_ready_retry(_store, candidate_project_id):
        retries.append(candidate_project_id)
        return []

    monkeypatch.setattr("rcp.api.app.ready_graph_watcher_groups", capture_ready_retry)
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    callback(
        project_id,
        "node_chat",
        RunRequest(chat_scope="node", chat_id="chat", node_id="exp/one", mode="work"),
        AgentTaskExecution(
            operation_id="evaluation-failure",
            store=app.state.background_tasks.store,
            control=AgentProcessControl(),
            applied_revision=2,
            applied_graph_state=_state(revision=2),
        ),
    )

    assert retries == [project_id]
    app.state.background_tasks.shutdown()


def test_condition_satisfied_while_rcp_was_down_fires_at_startup(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    condition = ProposalResolvedGraphCondition(node_id="hyp/foo", proposal_resolved=True)
    store.create_watchers([_graph_record("startup-graph", condition)])

    reopened = AppStore(path)
    startup_groups = [
        group
        for project_id in reopened.graph_watcher_project_ids()
        for group in evaluate_graph_watchers(
            reopened,
            project_id,
            _state(proposal_status="approved", revision=2),
        )
    ]

    assert _watcher_ids(startup_groups) == [["startup-graph"]]
    stored = reopened.watcher("startup-graph")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "completed"


def test_app_lifespan_evaluates_conditions_satisfied_before_restart(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "startup-data"
    first = create_app(str(manifest.path), data_dir=data_dir)
    append_fixture_patch(
        first.state.service,
        _canonical_fixture_patch(blocker_status="resolved"),
    )
    project_id = first.state.default_project_id
    assert project_id is not None
    first_store = first.state.background_tasks.store
    origin = _notification_task(first_store, "startup-origin", []).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": _test_authorizer(first_store),
            "request": {
                **_notification_task(first_store, "unused", []).request,
                "trigger": "human",
                "watcher_ids": [],
            },
        }
    )
    first_store.create_agent_task(origin)
    first_store.complete_agent_task(origin.operation_id, applied_revision=None, result={})
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    first_store.create_watchers(
        [
            _graph_record("startup-hook", condition).model_copy(
                update={
                    "project_id": project_id,
                    "origin_operation_id": origin.operation_id,
                }
            ),
            _graph_record("startup-retry", condition, status="completed").model_copy(
                update={
                    "project_id": project_id,
                    "origin_operation_id": origin.operation_id,
                }
            ),
        ]
    )
    first.state.background_tasks.shutdown()

    reopened = create_app(str(manifest.path), data_dir=data_dir)
    deliveries: list[list[str]] = []

    def capture_delivery(
        _tasks,
        _project_id,
        _kind,
        _request,
        watcher_ids,
        **_kwargs,
    ):
        deliveries.append(watcher_ids)
        task = _notification_task(
            reopened.state.background_tasks.store,
            "startup-graph-wake",
            watcher_ids,
        ).model_copy(
            update={
                "project_id": project_id,
                "authorized_by": _test_authorizer(reopened.state.background_tasks.store),
            }
        )
        return reopened.state.background_tasks.store.create_watcher_notification_task(
            task,
            watcher_ids,
        )

    monkeypatch.setattr("rcp.api.app.start_watcher_notification", capture_delivery)
    with TestClient(reopened) as client:
        assert client.get("/api/health").status_code == 200
        time.sleep(0.05)

    assert len(deliveries) == 1
    assert set(deliveries[0]) == {"startup-hook", "startup-retry"}
    for watcher_id in deliveries[0]:
        stored = reopened.state.background_tasks.store.watcher(watcher_id)
        assert isinstance(stored, GraphWatcherRecord)
        assert stored.status == "completed"


def test_discuss_settlement_retries_graph_wake_blocked_by_same_chat_overlap(
    manifest,
    tmp_path,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    authorizer = _test_authorizer(store)
    origin = _notification_task(store, "overlap-origin", []).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": authorizer,
            "request": {
                **_notification_task(store, "unused", []).request,
                "trigger": "human",
                "watcher_ids": [],
            },
        }
    )
    store.create_agent_task(origin)
    store.complete_agent_task(origin.operation_id, applied_revision=None, result={})
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    store.create_watchers(
        [
            _graph_record("overlap-graph", condition, status="completed").model_copy(
                update={
                    "project_id": project_id,
                    "origin_operation_id": origin.operation_id,
                }
            )
        ]
    )
    overlap = _notification_task(store, "active-discuss", []).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": authorizer,
            "request": {
                **_notification_task(store, "unused", []).request,
                "mode": "discuss",
                "trigger": "human",
                "watcher_ids": [],
            },
        }
    )
    store.create_agent_task(overlap)
    store.mark_agent_task_running(overlap.operation_id)
    callback = app.state.background_tasks.on_task_settled
    assert callback is not None

    async def settle_wake(_project_id, _kind, _request, _execution):
        yield f"data: {AgentEvent(event='answer', text='Retried after Discuss.').model_dump_json()}\n\n"
        yield f"data: {AgentEvent(event='done').model_dump_json()}\n\n"

    app.state.background_tasks.stream = settle_wake
    try:
        callback(
            project_id,
            "project_chat",
            RunRequest(chat_scope="project", chat_id="other-chat", mode="discuss"),
            AgentTaskExecution(
                operation_id="other-discuss",
                store=store,
                control=AgentProcessControl(),
            ),
        )
        blocked = store.watcher("overlap-graph")
        assert isinstance(blocked, GraphWatcherRecord)
        assert blocked.notified is False

        store.complete_agent_task(overlap.operation_id, applied_revision=None, result={})
        callback(
            project_id,
            "node_chat",
            RunRequest.model_validate(overlap.request),
            AgentTaskExecution(
                operation_id=overlap.operation_id,
                store=store,
                control=AgentProcessControl(),
            ),
        )

        delivered = store.watcher("overlap-graph")
        assert isinstance(delivered, GraphWatcherRecord)
        assert delivered.notified is True
        assert delivered.notification_operation_id is not None
        wake = _wait_for_terminal_task(store, delivered.notification_operation_id)
        assert wake.status == "succeeded"
        assert wake.request["mode"] == "work"
        assert wake.request["trigger"] == "watcher"
        assert wake.request["watcher_ids"] == ["overlap-graph"]
    finally:
        app.state.background_tasks.shutdown()


def test_degraded_or_halted_replay_never_fires_a_graph_condition(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers(
        [
            _graph_record(
                "resolved-status",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ),
            _graph_record(
                "resolved-proposal",
                ProposalResolvedGraphCondition(node_id="hyp/foo", proposal_resolved=True),
            ),
            _graph_record(
                "apparently-removed",
                NodeStatusGraphCondition(node_id="blk/missing", status_in=["resolved"]),
            ),
        ]
    )

    groups = evaluate_graph_watchers(
        store,
        "project",
        _state(
            blocker_status="resolved",
            proposal_status="approved",
            replay_status="degraded",
        ),
    )

    assert groups == []
    stored = store.watchers("project")
    assert {record.status for record in stored} == {"active"}
    assert all(not record.notified for record in stored)


def test_condition_on_a_removed_node_is_terminally_retired(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    store.create_watchers([_graph_record("removed-target", condition)])

    assert (
        evaluate_graph_watchers(
            store,
            "project",
            _state(include_blocker=False, revision=2),
        )
        == []
    )
    stored = store.watcher("removed-target")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "stopped"
    assert stored.notified is True
    assert stored.stop_reason == "Graph condition target was removed."
    assert store.graph_watcher_project_ids() == []


def test_graph_rows_never_enter_watcher_poller(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    solo = _external_record("external-solo").model_copy(
        update={
            "chat_id": "other-chat",
            "log_path": "/tmp/solo.log",
            "check_command": "test -f /tmp/solo.done",
        }
    )
    store.create_watchers([solo])
    store.create_watchers(
        [
            _external_record("mixed-external"),
            _graph_record(
                "active-graph",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
            ),
            _graph_record(
                "completed-graph",
                ProposalResolvedGraphCondition(node_id="hyp/foo", proposal_resolved=True),
                status="completed",
            ),
        ]
    )
    checked: list[str] = []

    def complete_external(
        spec: WatchSpec,
        _host: str,
        _timeout: float,
    ) -> WatcherCheckResult:
        checked.append(spec.log_path)
        return WatcherCheckResult(state="complete", checked_at=_CREATED_AT, exit_code=0)

    groups = WatcherPoller(
        store,
        check_runner=complete_external,
        clock=lambda: "2100-01-01T00:00:00+00:00",
    ).poll_once()

    assert sorted(checked) == ["/tmp/result.log", "/tmp/solo.log"]
    assert _watcher_ids(groups) == [["external-solo"]]
    graph = store.watcher("active-graph")
    assert isinstance(graph, GraphWatcherRecord)
    assert graph.status == "active"
    assert graph.last_evaluated_at is None
    assert store.watcher("mixed-external").notified is False
    assert store.watcher("completed-graph").notified is False


def test_retry_worker_keeps_the_watcher_poller_nonblocking(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    retry_started = threading.Event()
    release_retry = threading.Event()
    second_poll_finished = threading.Event()

    def blocked_retry(_generation) -> None:
        retry_started.set()
        release_retry.wait()

    worker = WatcherRetryWorker(blocked_retry)
    poller = WatcherPoller(store, on_poll_completed=worker.signal)
    worker.start()
    second_poll = threading.Thread(
        target=lambda: (poller.poll_once(), second_poll_finished.set()),
        name="second-watcher-poll",
    )
    try:
        assert poller.poll_once() == []
        assert retry_started.wait(1), "retry worker did not start"
        second_poll.start()
        assert second_poll_finished.wait(1), "retry callback blocked the poller thread"
    finally:
        release_retry.set()
        second_poll.join(timeout=1)
        worker.stop()


def test_retry_worker_stop_then_start_invalidates_the_slow_prior_generation() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    first_finished = threading.Event()
    stale_delivery = threading.Event()
    replacement_delivery = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def retry(generation) -> None:
        nonlocal calls
        with calls_lock:
            calls += 1
            call = calls
        if call == 1:
            first_started.set()
            release_first.wait()
            generation.run_if_current(stale_delivery.set)
            first_finished.set()
            return
        generation.run_if_current(replacement_delivery.set)

    worker = WatcherRetryWorker(retry)
    worker.start()
    worker.signal()
    assert first_started.wait(1), "first retry generation did not start"

    # Shutdown is bounded even while the callback is slow. A new lifespan gets
    # a replacement generation instead of silently reusing the stopping thread.
    worker.stop(timeout=0.01)
    worker.start()
    worker.signal()
    assert replacement_delivery.wait(1), "replacement retry generation did not run"

    release_first.set()
    assert first_finished.wait(1), "stale retry generation did not finish"
    assert not stale_delivery.is_set()
    worker.stop()


def test_watcher_notification_admission_fence_owns_claim_and_spawn(
    tmp_path,
    monkeypatch,
) -> None:
    store = AppStore(tmp_path / "fenced-claim.sqlite3")
    owner = store.local_owner
    assert owner is not None
    store.rename_space_user(owner.user_id, "Test researcher")
    store.create_watchers([_external_record("fenced-claim", status="completed")])

    async def unused_stream(_project_id, _kind, _request, _execution):
        yield f"data: {AgentEvent(event='done').model_dump_json()}\n\n"

    tasks = BackgroundAgentTasks(store, unused_stream)
    request = RunRequest.model_validate(
        _notification_task(store, "unused", ["fenced-claim"]).request
    )
    fence_active = False
    original_claim = store.create_watcher_notification_task

    def observed_claim(record, watcher_ids, **kwargs):
        assert fence_active, "durable watcher claim escaped the generation fence"
        return original_claim(record, watcher_ids, **kwargs)

    def observed_spawn(record, _request, *, continuation, parent=None):
        assert fence_active, "watcher task spawn escaped the generation fence"
        assert continuation == "fresh"
        assert parent is None
        return record

    def fence(callback) -> bool:
        nonlocal fence_active
        fence_active = True
        try:
            callback()
        finally:
            fence_active = False
        return True

    monkeypatch.setattr(store, "create_watcher_notification_task", observed_claim)
    monkeypatch.setattr(tasks, "_spawn_record", observed_spawn)

    started = start_watcher_notification(
        tasks,
        "project",
        "node_chat",
        request,
        ["fenced-claim"],
        authorized_by=_test_authorizer(store),
        admission_fence=fence,
    )

    assert started is not None
    assert store.watcher("fenced-claim").notified is True
    tasks.shutdown()


def test_stale_retry_generation_cannot_make_the_final_watcher_claim(tmp_path) -> None:
    store = AppStore(tmp_path / "stale-generation-claim.sqlite3")
    owner = store.local_owner
    assert owner is not None
    store.rename_space_user(owner.user_id, "Test researcher")
    store.create_watchers([_external_record("stale-final-claim", status="completed")])

    async def unused_stream(_project_id, _kind, _request, _execution):
        yield f"data: {AgentEvent(event='done').model_dump_json()}\n\n"

    tasks = BackgroundAgentTasks(store, unused_stream)
    request = RunRequest.model_validate(
        _notification_task(store, "unused", ["stale-final-claim"]).request
    )
    retry_started = threading.Event()
    release_retry = threading.Event()
    retry_finished = threading.Event()
    results: list[AgentTaskRecord | None] = []

    def delayed_retry(generation) -> None:
        retry_started.set()
        release_retry.wait()
        results.append(
            start_watcher_notification(
                tasks,
                "project",
                "node_chat",
                request,
                ["stale-final-claim"],
                authorized_by=_test_authorizer(store),
                admission_fence=generation.run_if_current,
            )
        )
        retry_finished.set()

    worker = WatcherRetryWorker(delayed_retry)
    worker.start()
    try:
        worker.signal()
        assert retry_started.wait(1), "retry generation did not reach final admission"
        worker.stop(timeout=0.01)
        worker.start()
        release_retry.set()
        assert retry_finished.wait(1), "stale retry generation did not finish"

        assert results == [None]
        stored = store.watcher("stale-final-claim")
        assert isinstance(stored, WatcherRecord)
        assert stored.status == "completed"
        assert stored.notified is False
    finally:
        release_retry.set()
        worker.stop()
        tasks.shutdown()


def test_periodic_poll_delivers_mixed_group_after_external_completes(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    authorizer = _test_authorizer(store)
    origin = _notification_task(store, "periodic-mixed-origin", []).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": authorizer,
            "request": {
                **_notification_task(store, "unused", []).request,
                "trigger": "human",
                "watcher_ids": [],
            },
        }
    )
    store.create_agent_task(origin)
    store.complete_agent_task(origin.operation_id, applied_revision=None, result={})
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    store.create_watchers(
        [
            _graph_record("periodic-graph", condition, status="completed").model_copy(
                update={"project_id": project_id, "origin_operation_id": origin.operation_id}
            ),
            _external_record("periodic-external").model_copy(
                update={"project_id": project_id, "origin_operation_id": origin.operation_id}
            ),
        ]
    )
    deliveries: list[list[str]] = []

    def claim_delivery(_tasks, _project_id, _kind, _request, watcher_ids, **_kwargs):
        deliveries.append(watcher_ids)
        task = _notification_task(store, "periodic-mixed-wake", watcher_ids).model_copy(
            update={"project_id": project_id, "authorized_by": authorizer}
        )
        return store.create_watcher_notification_task(task, watcher_ids)

    def complete_external(_spec, _host, _timeout):
        return WatcherCheckResult(state="complete", checked_at=_CREATED_AT, exit_code=0)

    monkeypatch.setattr("rcp.api.app.start_watcher_notification", claim_delivery)
    app.state.watcher_poller.check_runner = complete_external
    app.state.watcher_poller.clock = lambda: "2100-01-01T00:00:00+00:00"
    app.state.graph_watcher_retry_worker.start()
    try:
        groups = app.state.watcher_poller.poll_once()
        # `notified` is written after the delivery callback returns, so waiting on
        # the callback alone races the flag this test asserts.
        _wait_until(
            lambda: all(
                store.watcher(watcher_id).notified
                for watcher_id in ("periodic-external", "periodic-graph")
            ),
            detail="mixed graph delivery did not mark both watchers notified",
        )

        assert groups == []
        assert deliveries == [["periodic-external", "periodic-graph"]]
        assert all(
            store.watcher(watcher_id).notified
            for watcher_id in ("periodic-external", "periodic-graph")
        )
    finally:
        app.state.graph_watcher_retry_worker.stop()
        app.state.background_tasks.shutdown()


def test_periodic_poll_retries_pure_graph_delivery_after_transient_failure(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    authorizer = _test_authorizer(store)
    origin = _notification_task(store, "periodic-retry-origin", []).model_copy(
        update={
            "project_id": project_id,
            "authorized_by": authorizer,
            "request": {
                **_notification_task(store, "unused", []).request,
                "trigger": "human",
                "watcher_ids": [],
            },
        }
    )
    store.create_agent_task(origin)
    store.complete_agent_task(origin.operation_id, applied_revision=None, result={})
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    store.create_watchers(
        [
            _graph_record("periodic-retry-graph", condition, status="completed").model_copy(
                update={"project_id": project_id, "origin_operation_id": origin.operation_id}
            )
        ]
    )
    attempts: list[list[str]] = []

    def transient_then_claim(_tasks, _project_id, _kind, _request, watcher_ids, **_kwargs):
        attempts.append(watcher_ids)
        if len(attempts) == 1:
            raise RuntimeError("transient delivery failure")
        task = _notification_task(store, "periodic-retry-wake", watcher_ids).model_copy(
            update={"project_id": project_id, "authorized_by": authorizer}
        )
        return store.create_watcher_notification_task(task, watcher_ids)

    monkeypatch.setattr("rcp.api.app.start_watcher_notification", transient_then_claim)
    app.state.graph_watcher_retry_worker.start()
    try:
        assert app.state.watcher_poller.poll_once() == []
        _wait_until(lambda: len(attempts) == 1, detail="first graph delivery was not attempted")
        first = store.watcher("periodic-retry-graph")
        assert isinstance(first, GraphWatcherRecord)
        assert first.notified is False

        assert app.state.watcher_poller.poll_once() == []
        # Same race: the retry attempt is recorded before `notified` is written.
        _wait_until(
            lambda: len(attempts) == 2 and store.watcher("periodic-retry-graph").notified,
            detail="graph delivery retry did not mark the watcher notified",
        )
        second = store.watcher("periodic-retry-graph")
        assert isinstance(second, GraphWatcherRecord)
        assert second.notified is True

        assert app.state.watcher_poller.poll_once() == []
        assert attempts == [["periodic-retry-graph"], ["periodic-retry-graph"]]
    finally:
        app.state.graph_watcher_retry_worker.stop()
        app.state.background_tasks.shutdown()


def test_external_and_graph_completions_coalesce_into_one_wake(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    store.create_watchers(
        [
            _external_record("external", status="completed"),
            _graph_record("graph", condition),
        ]
    )

    groups = evaluate_graph_watchers(
        store,
        "project",
        _state(blocker_status="resolved", revision=2),
    )

    assert [set(group) for group in _watcher_ids(groups)] == [{"external", "graph"}]
    watcher_ids = _watcher_ids(groups)[0]
    queued = store.create_watcher_notification_task(
        _notification_task(store, "one-wake", watcher_ids),
        watcher_ids,
    )
    assert queued is not None
    assert all(record.notified for record in store.watchers("project"))
    assert (
        evaluate_graph_watchers(
            store,
            "project",
            _state(blocker_status="resolved", revision=2),
        )
        == []
    )


def test_every_graph_wake_spends_one_experiment_budget_unit(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    continuation = _continuation().model_copy(
        update={
            "patch_kind": "experiment_loop",
            "control_node_id": "exp/one",
            "control_revision": 1,
            "control_episode_id": episode_id,
            "control_invocation": 1,
            "control_invocation_ceiling": 2,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
        }
    )
    root = _experiment_task(
        store,
        "loop-root",
        episode_id=episode_id,
        invocation=1,
        ceiling=2,
        watcher_ids=[],
    )
    store.create_experiment_episode_with_invocation(root)
    store.complete_agent_task("loop-root", applied_revision=None, result={})
    store.commit_experiment_episode_turn(
        episode_id=episode_id,
        project_id="project",
        control_node_id="exp/one",
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="native-loop-session",
        stage_host=None,
        stage_root="/tmp/loop-stage",
        chat_id="chat",
        operation_id="loop-root",
        invocation=1,
        graph_result="no graph change",
        watcher_ids=[],
        context_baseline={},
    )
    condition = NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"])
    store.create_watchers([_graph_record("budgeted-graph", condition, continuation=continuation)])
    groups = evaluate_graph_watchers(
        store,
        "project",
        _state(blocker_status="resolved", revision=2),
    )
    assert _watcher_ids(groups) == [["budgeted-graph"]]

    wake = _experiment_task(
        store,
        "graph-wake",
        episode_id=episode_id,
        invocation=2,
        ceiling=2,
        watcher_ids=["budgeted-graph"],
    ).model_copy(
        update={
            "native_session_id": "native-loop-session",
            "stage_root": "/tmp/loop-stage",
        }
    )
    wake.request["session_id"] = "native-loop-session"
    queued = store.create_experiment_watcher_invocation(wake, ["budgeted-graph"])

    assert queued is not None
    runtime = store.experiment_loop_runtime("project", "exp/one")
    assert runtime.invocations_used == 2
    assert runtime.invocation_ceiling - runtime.invocations_used == 0


def test_experiment_agent_cannot_retire_graph_watcher_and_stop_list_is_atomic(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    root = _experiment_task(
        store,
        "stop-list-root",
        episode_id=episode_id,
        invocation=1,
        ceiling=2,
        watcher_ids=[],
    )
    store.create_experiment_episode_with_invocation(root)
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    store.commit_experiment_episode_turn(
        episode_id=episode_id,
        project_id="project",
        control_node_id="exp/one",
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="native-stop-list",
        stage_host=None,
        stage_root="/tmp/stop-list-stage",
        chat_id="chat",
        operation_id=root.operation_id,
        invocation=1,
        graph_result="no graph change",
        watcher_ids=[],
        context_baseline={},
    )
    continuation = _continuation().model_copy(
        update={
            "patch_kind": "experiment_loop",
            "control_node_id": "exp/one",
            "control_revision": 1,
            "control_episode_id": episode_id,
            "control_invocation": 1,
            "control_invocation_ceiling": 2,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
        }
    )
    binding = _binding(continuation=continuation).model_copy(
        update={"origin_operation_id": root.operation_id}
    )
    store.create_watchers(
        [
            _external_record("stoppable-external", continuation=continuation).model_copy(
                update={"origin_operation_id": root.operation_id}
            ),
            _graph_record(
                "protected-graph",
                NodeStatusGraphCondition(node_id="blk/foo", status_in=["resolved"]),
                continuation=continuation,
            ).model_copy(update={"origin_operation_id": root.operation_id}),
        ]
    )

    with pytest.raises(ValueError, match="only external observers: protected-graph"):
        store.persist_experiment_watchers_idempotently(
            [],
            stops=[
                WatcherStopRequest(
                    stop_watcher_id="stoppable-external",
                    reason="External work is no longer needed.",
                ),
                WatcherStopRequest(
                    stop_watcher_id="protected-graph",
                    reason="The canonical condition is no longer needed.",
                ),
            ],
            binding=binding,
        )

    external = store.watcher("stoppable-external")
    graph = store.watcher("protected-graph")
    assert isinstance(external, WatcherRecord)
    assert isinstance(graph, GraphWatcherRecord)
    assert external.status == "active"
    assert external.notified is False
    assert graph.status == "active"
    assert graph.notified is False


def test_pre_start_pause_still_runs_settlement_hook_without_stream_close(tmp_path) -> None:
    store = AppStore(tmp_path / "pre-start-pause.sqlite3")
    record = _notification_task(store, "pre-start-pause", [])
    streamed: list[str] = []
    stream_closed: list[str] = []
    settled: list[tuple[str, str]] = []

    async def stream(_project_id, _kind, _request, _execution):
        streamed.append("called")
        yield f"data: {AgentEvent(event='done').model_dump_json()}\n\n"

    def closed(_project_id, _kind, _request, execution) -> None:
        stream_closed.append(execution.operation_id)

    def after_settlement(_project_id, _kind, _request, execution) -> None:
        task = store.agent_task(execution.operation_id)
        assert task is not None
        settled.append((execution.operation_id, task.status))

    tasks = BackgroundAgentTasks(
        store,
        stream,
        on_stream_closed=closed,
        on_task_settled=after_settlement,
    )
    store.create_agent_task(record)
    control = AgentProcessControl()
    control.pause_requested.set()

    tasks._run(
        record,
        RunRequest.model_validate(record.request),
        control,
        "fresh",
    )

    paused = store.agent_task(record.operation_id)
    assert paused is not None and paused.status == "paused"
    assert settled == [(record.operation_id, "paused")]
    assert streamed == []
    assert stream_closed == []


def test_shutdown_serializes_watcher_delivery_admission_with_worker_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    store = AppStore(tmp_path / "shutdown-admission.sqlite3")
    owner = store.local_owner
    assert owner is not None
    store.rename_space_user(owner.user_id, "Test researcher")
    authorized_by = _test_authorizer(store)
    store.create_watchers([_external_record("pre-close", status="completed")])
    spawn_entered = threading.Event()
    release_spawn = threading.Event()
    shutdown_finished = threading.Event()
    delivered: list[AgentTaskRecord | None] = []
    errors: list[BaseException] = []

    async def wait_for_shutdown(_project_id, _kind, _request, execution):
        while not execution.control.pause_requested.is_set():
            await asyncio.sleep(0.01)
        yield f"data: {AgentEvent(event='paused', text='Server shutdown.').model_dump_json()}\n\n"

    tasks = BackgroundAgentTasks(store, wait_for_shutdown)
    original_spawn = tasks._spawn_record

    def blocked_spawn(record, request, *, continuation, parent=None):
        spawn_entered.set()
        if not release_spawn.wait(2):
            raise RuntimeError("test did not release watcher spawn")
        return original_spawn(
            record,
            request,
            continuation=continuation,
            parent=parent,
        )

    monkeypatch.setattr(tasks, "_spawn_record", blocked_spawn)
    first_request = RunRequest.model_validate(
        _notification_task(store, "unused", ["pre-close"]).request
    )

    def deliver_before_close() -> None:
        try:
            delivered.append(
                start_watcher_notification(
                    tasks,
                    "project",
                    "node_chat",
                    first_request,
                    ["pre-close"],
                    authorized_by=authorized_by,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def shut_down() -> None:
        try:
            tasks.shutdown(timeout=2)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            shutdown_finished.set()

    delivery_thread = threading.Thread(target=deliver_before_close, name="pre-close-delivery")
    shutdown_thread = threading.Thread(target=shut_down, name="watcher-shutdown")
    try:
        delivery_thread.start()
        assert spawn_entered.wait(1), "delivery did not enter its serialized spawn"
        shutdown_thread.start()
        assert not shutdown_finished.wait(0.05), "shutdown bypassed watcher admission"
        release_spawn.set()
        delivery_thread.join(timeout=3)
        shutdown_thread.join(timeout=3)

        assert not delivery_thread.is_alive()
        assert not shutdown_thread.is_alive()
        assert errors == []
        assert len(delivered) == 1 and delivered[0] is not None
        started = delivered[0]
        assert started is not None
        settled = store.agent_task(started.operation_id)
        assert settled is not None
        assert settled.status == "paused"

        store.create_watchers([_external_record("post-close", status="completed")])
        second_request = RunRequest.model_validate(
            _notification_task(store, "unused", ["post-close"]).request
        )
        assert (
            start_watcher_notification(
                tasks,
                "project",
                "node_chat",
                second_request,
                ["post-close"],
                authorized_by=authorized_by,
            )
            is None
        )
        post_close = store.watcher("post-close")
        assert isinstance(post_close, WatcherRecord)
        assert post_close.status == "completed"
        assert post_close.notified is False
    finally:
        release_spawn.set()
        delivery_thread.join(timeout=1)
        shutdown_thread.join(timeout=1)
        tasks.shutdown(timeout=1)


@pytest.mark.parametrize(
    ("terminal_event", "expected_status"),
    [("error", "failed"), ("paused", "paused")],
)
def test_post_settlement_hook_observes_verdict_and_revision_without_replacing_it(
    tmp_path,
    terminal_event,
    expected_status,
) -> None:
    store = AppStore(tmp_path / f"{terminal_event}.sqlite3")
    boundary = _state(blocker_status="resolved", revision=7)
    observed: list[tuple[str, int | None, bool]] = []

    async def stream(_project_id, _kind, _request, execution):
        _record_patch_applied_receipt(execution, boundary)
        yield f"data: {AgentEvent(event=terminal_event, text='provider stopped').model_dump_json()}\n\n"

    def settled(_project_id, _kind, _request, execution) -> None:
        task = store.agent_task(execution.operation_id)
        assert task is not None
        observed.append(
            (
                task.status,
                execution.applied_revision,
                execution.applied_graph_state is boundary,
            )
        )
        raise RuntimeError("observer failure")

    tasks = BackgroundAgentTasks(store, stream, on_task_settled=settled)
    request = RunRequest(
        chat_scope="project",
        chat_id=str(uuid.uuid4()),
        message="Apply a graph change, then stop.",
        mode="work",
    )
    owner = store.local_owner
    assert owner is not None
    owner = store.rename_space_user(owner.user_id, "Test researcher")
    assert owner.display_name is not None
    started = tasks.start(
        "project",
        "project_chat",
        request,
        authorized_by=AuthorizedHuman(
            space_id=store.space_id,
            user_id=owner.user_id,
            display_name=owner.display_name,
        ),
    )
    try:
        terminal = _wait_for_terminal_task(store, started.operation_id)
    finally:
        tasks.shutdown()

    assert terminal.status == expected_status
    assert observed == [(expected_status, 7, True)]
    receipt_categories = {
        receipt.category for receipt in store.agent_task_receipts(started.operation_id)
    }
    assert {"patch_applied", "task_settled_callback_failed"} <= receipt_categories


def _experiment_task(
    store: AppStore,
    operation_id: str,
    *,
    episode_id: str,
    invocation: int,
    ceiling: int,
    watcher_ids: list[str],
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id="project",
        episode_id=episode_id,
        kind="node_chat",
        status="queued",
        request={
            "chat_id": "chat",
            "node_id": "exp/one",
            "provider": "codex",
            "model": "gpt-5",
            "reasoning": "medium",
            "run_on": "laptop",
            "run_truth_scope": ["repo-a"],
            "mode": "work",
            "trigger": "watcher" if watcher_ids else "experiment_run",
            "patch_kind": "experiment_loop",
            "control_node_id": "exp/one",
            "control_revision": 1,
            "control_episode_id": episode_id,
            "control_invocation": invocation,
            "control_invocation_ceiling": ceiling,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
            "workflow_ids": [],
            "skill_ids": [],
            "invoked_workflow_ids": [],
            "invoked_skill_ids": [],
            "resolved_skill_packages": [],
            "watcher_ids": watcher_ids,
        },
        created_at=now,
        updated_at=now,
        status_message="Queued bounded graph wake.",
        authorized_by=_test_authorizer(store),
    )
