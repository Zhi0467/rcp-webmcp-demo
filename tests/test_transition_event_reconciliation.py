from __future__ import annotations

from dataclasses import replace

import pytest

from rcp.core.models import Blocker, GraphState, Patch
from rcp.core.transition_models import (
    GraphHeadRef,
    GraphTargetRef,
    TransitionCauseRef,
    TransitionEvent,
    TransitionInitiatingGroup,
    TransitionTrace,
)
from rcp.runs.transition_event_reconciliation import (
    AcceptedGraphBoundary,
    reconcile_accepted_graph_boundaries,
)
from rcp.storage import (
    AppStore,
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    WatcherContinuation,
)
from rcp.watchers import evaluate_graph_watchers

_CREATED_AT = "2026-08-18T00:00:00+00:00"


def _blocker(status: str = "open") -> Blocker:
    return Blocker(
        id="blk/capacity",
        type="blocker",
        title="Capacity unavailable",
        description="The required capacity is unavailable.",
        status=status,
    )


def _state(revision: int, *, blocker_status: str | None = "open") -> GraphState:
    nodes = {}
    if blocker_status is not None:
        blocker = _blocker(blocker_status)
        nodes[blocker.id] = blocker
    return GraphState(revision=revision, nodes=nodes)


def _continuation() -> WatcherContinuation:
    return WatcherContinuation(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["repo"],
    )


def _watcher(
    watcher_id: str,
    *,
    armed_revision: int = 1,
    status_in: list[str] | None = None,
    graph_target: GraphTargetRef | None = None,
    origin_operation_id: str = "origin",
) -> GraphWatcherRecord:
    return GraphWatcherRecord(
        watcher_id=watcher_id,
        project_id="project",
        origin_operation_id=origin_operation_id,
        origin_task_kind="node_chat",
        chat_id="chat",
        node_id="exp/one",
        graph_target=graph_target or GraphTargetRef(),
        condition=NodeStatusGraphCondition(
            node_id="blk/capacity",
            status_in=status_in or ["resolved"],
        ),
        armed_revision=armed_revision,
        continuation=_continuation(),
        created_at=_CREATED_AT,
        last_evaluated_at=_CREATED_AT,
    )


def _status_event(
    *,
    event_id: str = "a" * 64,
    after: str = "resolved",
) -> TransitionEvent:
    return TransitionEvent(
        event_id=event_id,
        event_type="node_status_changed",
        cause=TransitionCauseRef(kind="action", action_index=0),
        node_id="blk/capacity",
        field="status",
        before="open",
        after=after,
    )


def _accepted_patch(
    *,
    revision: int = 2,
    target: GraphTargetRef | None = None,
    event: TransitionEvent | None = None,
    admission: str = "accepted",
) -> Patch:
    operation = {
        "op": "update_nodes",
        "nodes": [
            {
                "id": "blk/capacity",
                "base_updated_rev": revision - 1,
                "changes": {"status": "resolved"},
            }
        ],
    }
    transition = TransitionTrace(
        transition_id="b" * 64,
        pre_head=GraphHeadRef(
            target=target or GraphTargetRef(),
            revision=revision - 1,
        ),
        ruleset_tag="rcp.lifecycle.v1",
        initiating_groups=[
            TransitionInitiatingGroup(
                group_id="group-1",
                operation_indexes=[0],
                summary="Resolve the capacity blocker.",
            )
        ],
        lifecycle_events=[event or _status_event()],
        expanded_ops_sha256="c" * 64,
    )
    return Patch(
        revision=revision,
        kind="work",
        author="agent",
        producer="agent",
        summary="Resolve the capacity blocker.",
        ops=[operation],
        admission=admission,
        transition=transition,
    )


def test_status_event_completes_removed_target_once(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_watcher("resolved-watcher")])
    final_state = _state(2, blocker_status=None)
    event = _status_event()

    first_groups = evaluate_graph_watchers(
        store,
        "project",
        final_state,
        lifecycle_events=[event],
    )
    first = store.watcher("resolved-watcher")
    assert isinstance(first, GraphWatcherRecord)
    assert first.status == "completed"
    assert first.notified is False
    assert [[record.watcher_id for record in group] for group in first_groups] == [
        ["resolved-watcher"]
    ]

    second_groups = evaluate_graph_watchers(
        store,
        "project",
        final_state,
        lifecycle_events=[event],
    )
    second = store.watcher("resolved-watcher")
    assert isinstance(second, GraphWatcherRecord)
    assert second.completed_at == first.completed_at
    assert [[record.watcher_id for record in group] for group in second_groups] == [
        ["resolved-watcher"]
    ]


def test_same_revision_watcher_does_not_retroactively_consume_event(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_watcher("later-watcher", armed_revision=2)])

    reconcile_accepted_graph_boundaries(
        store,
        "project",
        [
            AcceptedGraphBoundary(
                target=GraphTargetRef(),
                revision=2,
                transition_id="6" * 64,
                state=_state(2, blocker_status=None),
                lifecycle_events=(_status_event(),),
            )
        ],
        current_head=GraphHeadRef(target=GraphTargetRef(), revision=2),
    )

    stored = store.watcher("later-watcher")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "active"


def test_nonmatching_event_preserves_final_removed_semantics(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_watcher("closed-watcher")])

    evaluate_graph_watchers(
        store,
        "project",
        _state(2, blocker_status=None),
        lifecycle_events=[_status_event(after="closed")],
    )

    stored = store.watcher("closed-watcher")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "stopped"
    assert stored.notified is True


def test_target_resolver_routes_only_the_matching_graph_head(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_watcher("branch-watcher")])
    branch = GraphTargetRef(kind="branch", branch_id="episode/one")

    def branch_target(_record: GraphWatcherRecord) -> GraphTargetRef:
        return branch

    for target in (
        GraphTargetRef(),
        GraphTargetRef(kind="branch", branch_id="episode/two"),
    ):
        evaluate_graph_watchers(
            store,
            "project",
            _state(2, blocker_status=None),
            lifecycle_events=[_status_event()],
            graph_target=target,
            watcher_target=branch_target,
        )
        stored = store.watcher("branch-watcher")
        assert isinstance(stored, GraphWatcherRecord)
        assert stored.status == "active"

    evaluate_graph_watchers(
        store,
        "project",
        _state(2, blocker_status=None),
        lifecycle_events=[_status_event()],
        graph_target=branch,
        watcher_target=branch_target,
    )
    stored = store.watcher("branch-watcher")
    assert isinstance(stored, GraphWatcherRecord)
    assert stored.status == "completed"


def test_replay_boundary_reconciles_post_commit_crash_window_idempotently(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_watcher("crash-window-watcher")])
    pre_state = _state(1)
    post_state = _state(2, blocker_status=None)
    patch = _accepted_patch()
    boundary = AcceptedGraphBoundary.from_replay(pre_state, patch, post_state)

    first = reconcile_accepted_graph_boundaries(store, "project", [boundary])
    completed = store.watcher("crash-window-watcher")
    assert isinstance(completed, GraphWatcherRecord)
    assert completed.status == "completed"
    assert [[record.watcher_id for record in group] for group in first] == [
        ["crash-window-watcher"]
    ]

    second = reconcile_accepted_graph_boundaries(store, "project", [boundary])
    repeated = store.watcher("crash-window-watcher")
    assert isinstance(repeated, GraphWatcherRecord)
    assert repeated.completed_at == completed.completed_at
    assert [[record.watcher_id for record in group] for group in second] == [
        ["crash-window-watcher"]
    ]


def test_boundary_construction_rejects_noncanonical_inputs() -> None:
    pre_state = _state(1)
    post_state = _state(2, blocker_status=None)

    with pytest.raises(ValueError, match="rejected Patch"):
        AcceptedGraphBoundary.from_replay(
            pre_state,
            _accepted_patch(admission="rejected"),
            post_state,
        )
    with pytest.raises(ValueError, match="state does not match"):
        AcceptedGraphBoundary.from_replay(pre_state, _accepted_patch(), _state(3))


def test_reconciliation_rejects_target_local_reordering(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    first = AcceptedGraphBoundary.from_replay(
        _state(1),
        _accepted_patch(),
        _state(2, blocker_status="resolved"),
    )

    with pytest.raises(ValueError, match="canonical order"):
        reconcile_accepted_graph_boundaries(store, "project", [first, first])


def test_reconciliation_heads_are_durable_and_target_isolated(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    branch = GraphTargetRef(kind="branch", branch_id="episode/one")
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO graph_runs(
                operation_id, project_id, kind, status, request_json,
                created_at, updated_at, status_message, graph_target_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "branch-origin",
                "project",
                "auto_research",
                "succeeded",
                "{}",
                _CREATED_AT,
                _CREATED_AT,
                "Succeeded",
                branch.model_dump_json(),
            ),
        )
    store.create_watchers([_watcher("main-watcher")])
    store.create_watchers(
        [
            _watcher(
                "branch-watcher",
                graph_target=branch,
                origin_operation_id="branch-origin",
            ),
        ]
    )
    main_boundary = AcceptedGraphBoundary(
        target=GraphTargetRef(),
        revision=2,
        transition_id="1" * 64,
        state=_state(2, blocker_status=None),
        lifecycle_events=(_status_event(event_id="2" * 64),),
    )
    branch_boundary = AcceptedGraphBoundary(
        target=branch,
        revision=2,
        transition_id="3" * 64,
        state=_state(2),
        lifecycle_events=(),
    )

    reconcile_accepted_graph_boundaries(
        store,
        "project",
        [main_boundary],
        current_head=GraphHeadRef(target=GraphTargetRef(), revision=2),
    )
    assert store.watcher("main-watcher").status == "completed"  # type: ignore[union-attr]
    assert store.watcher("branch-watcher").status == "active"  # type: ignore[union-attr]

    reconcile_accepted_graph_boundaries(
        store,
        "project",
        [branch_boundary],
        current_head=GraphHeadRef(target=branch, revision=2),
    )
    reopened = AppStore(path)
    main_head = reopened.graph_watcher_reconciliation_head("project", GraphTargetRef())
    branch_head = reopened.graph_watcher_reconciliation_head("project", branch)
    assert main_head is not None and main_head.revision == 2
    assert main_head.transition_id == "1" * 64
    assert branch_head is not None and branch_head.revision == 2
    assert branch_head.transition_id == "3" * 64
    assert reopened.watcher("branch-watcher").status == "active"  # type: ignore[union-attr]


def test_restart_reconciliation_evaluates_only_boundaries_after_durable_head(
    tmp_path,
    monkeypatch,
) -> None:
    import rcp.runs.transition_event_reconciliation as reconciliation

    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    store.create_watchers([_watcher("restart-watcher", armed_revision=0)])
    boundaries = [
        AcceptedGraphBoundary(
            target=GraphTargetRef(),
            revision=revision,
            transition_id=f"{revision:x}" * 64,
            state=_state(revision),
            lifecycle_events=(),
        )
        for revision in (1, 2, 3)
    ]
    reconcile_accepted_graph_boundaries(
        store,
        "project",
        boundaries[:2],
        current_head=GraphHeadRef(target=GraphTargetRef(), revision=2),
    )

    reopened = AppStore(path)
    evaluated_revisions: list[int] = []
    original = reconciliation.graph_watcher_boundary_result

    def observe(record, state, lifecycle_events):
        evaluated_revisions.append(state.revision)
        return original(record, state, lifecycle_events)

    monkeypatch.setattr(reconciliation, "graph_watcher_boundary_result", observe)
    reconcile_accepted_graph_boundaries(
        reopened,
        "project",
        boundaries,
        current_head=GraphHeadRef(target=GraphTargetRef(), revision=3),
    )

    assert evaluated_revisions == [3]
    head = reopened.graph_watcher_reconciliation_head("project", GraphTargetRef())
    assert head is not None and head.revision == 3

    with pytest.raises(RuntimeError, match="changed identity"):
        reconcile_accepted_graph_boundaries(
            reopened,
            "project",
            [replace(boundaries[-1], transition_id="d" * 64)],
            current_head=GraphHeadRef(target=GraphTargetRef(), revision=3),
        )

    with pytest.raises(ValueError, match="cannot arm behind"):
        reopened.create_watchers([_watcher("stale-arm", armed_revision=2)])


def test_failed_boundary_evaluation_rolls_back_watcher_and_head_for_retry(
    tmp_path,
    monkeypatch,
) -> None:
    import rcp.runs.transition_event_reconciliation as reconciliation

    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers(
        [
            _watcher("first-retry-watcher", armed_revision=0),
            _watcher("second-retry-watcher", armed_revision=0),
        ]
    )
    boundary = AcceptedGraphBoundary(
        target=GraphTargetRef(),
        revision=2,
        transition_id="4" * 64,
        state=_state(2, blocker_status="resolved"),
        lifecycle_events=(_status_event(event_id="5" * 64),),
    )
    original = reconciliation.graph_watcher_boundary_result
    evaluations = 0

    def fail_second(record, state, lifecycle_events):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 2:
            raise OSError("transient SQLite-adjacent failure")
        return original(record, state, lifecycle_events)

    monkeypatch.setattr(reconciliation, "graph_watcher_boundary_result", fail_second)
    with pytest.raises(OSError, match="transient"):
        reconcile_accepted_graph_boundaries(
            store,
            "project",
            [boundary],
            current_head=GraphHeadRef(target=GraphTargetRef(), revision=2),
        )

    assert store.graph_watcher_reconciliation_head("project", GraphTargetRef()) is None
    assert store.watcher("first-retry-watcher").status == "active"  # type: ignore[union-attr]
    assert store.watcher("second-retry-watcher").status == "active"  # type: ignore[union-attr]

    monkeypatch.setattr(reconciliation, "graph_watcher_boundary_result", original)
    reconcile_accepted_graph_boundaries(
        store,
        "project",
        [boundary],
        current_head=GraphHeadRef(target=GraphTargetRef(), revision=2),
    )
    assert store.watcher("first-retry-watcher").status == "completed"  # type: ignore[union-attr]
    assert store.watcher("second-retry-watcher").status == "completed"  # type: ignore[union-attr]
    head = store.graph_watcher_reconciliation_head("project", GraphTargetRef())
    assert head is not None and head.revision == 2


def test_legacy_database_adds_graph_watcher_reconciliation_table(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    with store.connection() as connection:
        connection.execute("DROP TABLE graph_watcher_reconciliation")

    reopened = AppStore(path)
    assert reopened.graph_watcher_reconciliation_head("project", GraphTargetRef()) is None
    with reopened.connection() as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'graph_watcher_reconciliation'"
        ).fetchone()
    assert table is not None
