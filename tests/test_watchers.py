from __future__ import annotations

import shlex
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from rcp.core.models import AuthorizedHuman
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    EpisodeInvocationCeilingReached,
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    WatcherClaimConflict,
    WatcherContinuation,
    WatcherRecord,
    WatcherStopRequest,
    watcher_next_check_at,
)
from rcp.watchers import (
    ExperimentWatchSpec,
    WatcherBinding,
    WatcherCheckResult,
    WatcherInitialCheckError,
    WatcherPoller,
    WatchSpec,
    arm_watchers,
    parse_experiment_watch_json,
    parse_watch_json,
    run_watcher_check,
)


def _continuation() -> WatcherContinuation:
    return WatcherContinuation(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["state"],
        patch_kind="work",
    )


def _binding(origin: str = "origin") -> WatcherBinding:
    return WatcherBinding(
        project_id="project",
        origin_operation_id=origin,
        origin_task_kind="node_chat",
        chat_id="chat",
        node_id="exp-one",
        continuation=_continuation(),
    )


def _identity(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name or "Test researcher",
    )


def _record(
    watcher_id: str,
    *,
    origin: str = "origin",
    status: str = "active",
) -> WatcherRecord:
    return WatcherRecord(
        watcher_id=watcher_id,
        project_id="project",
        origin_operation_id=origin,
        origin_task_kind="node_chat",
        chat_id="chat",
        node_id="exp-one",
        check_command="true",
        log_path=f"/tmp/{watcher_id}.log",
        cwd="/tmp",
        continuation=_continuation(),
        status=status,
        created_at="2026-08-01T00:00:00+00:00",
        completed_at=("2026-08-01T00:01:00+00:00" if status == "completed" else None),
    )


def _task(store: AppStore, operation_id: str, watcher_ids: list[str]) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id="project",
        kind="node_chat",
        status="queued",
        request={
            "chat_id": "chat",
            "node_id": "exp-one",
            "provider": "codex",
            "model": "gpt-5",
            "reasoning": "medium",
            "run_on": "laptop",
            "run_truth_scope": ["state"],
            "mode": "work",
            "trigger": "watcher",
            "patch_kind": "work",
            "control_node_id": None,
            "workflow_ids": [],
            "skill_ids": [],
            "invoked_workflow_ids": [],
            "invoked_skill_ids": [],
            "resolved_skill_packages": [],
            "watcher_ids": watcher_ids,
        },
        created_at=now,
        updated_at=now,
        status_message="Queued watcher wake.",
    )


def _loop_task(
    store: AppStore,
    operation_id: str,
    *,
    episode_id: str,
    invocation: int,
    ceiling: int = 2,
    watcher_ids: list[str] | None = None,
    parent_operation_id: str | None = None,
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
            "node_id": "exp-one",
            "provider": "codex",
            "model": "gpt-5",
            "reasoning": "medium",
            "run_on": "laptop",
            "run_truth_scope": ["state"],
            "mode": "work",
            "trigger": "watcher" if watcher_ids else "experiment_run",
            "patch_kind": "experiment_loop",
            "control_node_id": "exp-one",
            "control_revision": 0,
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
            "watcher_ids": watcher_ids or [],
        },
        created_at=now,
        updated_at=now,
        status_message="Queued loop invocation.",
        parent_operation_id=parent_operation_id,
        attempt=2 if parent_operation_id else 1,
        authorized_by=_identity(store),
    )


def test_watch_json_is_a_nonempty_strict_two_list_object() -> None:
    parsed = parse_watch_json(
        '{"external":[{"check_command":"squeue -h -j 4471 >/dev/null",'
        '"log_path":"/logs/4471.log","cwd":"/work"}],'
        '"graph":[{"node_id":"blk/waiting","status_in":["resolved"]}]}'
    )

    assert parsed.external == [
        WatchSpec(
            check_command="squeue -h -j 4471 >/dev/null",
            log_path="/logs/4471.log",
            cwd="/work",
        )
    ]
    assert [item.model_dump(mode="json") for item in parsed.graph] == [
        {"node_id": "blk/waiting", "status_in": ["resolved"]}
    ]
    for payload in (
        '{"external":[],"graph":[]}',
        "[]",
        '{"check_command":"true","log_path":"/tmp/x","cwd":"/tmp"}',
        '{"external":[{"check_command":"true","log_path":"relative","cwd":"/tmp"}],"graph":[]}',
        '{"external":[{"check_command":"true","log_path":"/tmp/x","cwd":"/tmp",'
        '"host":"bad"}],"graph":[]}',
        '{"external":[],"graph":[{"node_id":"blk/waiting","standing":"accepted"}]}',
    ):
        with pytest.raises((ValidationError, ValueError)):
            parse_watch_json(payload)


def test_experiment_watch_json_accepts_external_maintenance_and_graph_conditions() -> None:
    assert parse_experiment_watch_json('{"external":[],"graph":[]}').is_empty
    handoff = parse_experiment_watch_json(
        '{"external":['
        '{"group":"eval-shards","check_command":"exit 1","log_path":"/tmp/a.log","cwd":"/tmp"},'
        '{"group":"eval-shards","check_command":"exit 1","log_path":"/tmp/b.log","cwd":"/tmp"},'
        '{"stop_watcher_id":"old-watcher","reason":"Cancelled superseded job"}'
        '],"graph":[{"node_id":"hyp/result","proposal_resolved":true}]}'
    )

    assert handoff.observers == [
        ExperimentWatchSpec(
            group="eval-shards", check_command="exit 1", log_path="/tmp/a.log", cwd="/tmp"
        ),
        ExperimentWatchSpec(
            group="eval-shards", check_command="exit 1", log_path="/tmp/b.log", cwd="/tmp"
        ),
    ]
    assert handoff.stops == [
        WatcherStopRequest(stop_watcher_id="old-watcher", reason="Cancelled superseded job")
    ]
    assert [item.model_dump(mode="json") for item in handoff.graph_conditions] == [
        {"node_id": "hyp/result", "proposal_resolved": True}
    ]
    with pytest.raises(ValidationError):
        parse_watch_json(
            '{"external":[{"group":"eval-shards","check_command":"exit 1",'
            '"log_path":"/tmp/a.log","cwd":"/tmp"}],"graph":[]}'
        )
    with pytest.raises(ValueError, match="at least two"):
        parse_experiment_watch_json(
            '{"external":[{"group":"eval-shards","check_command":"exit 1",'
            '"log_path":"/tmp/a.log","cwd":"/tmp"}],"graph":[]}'
        )


def _loop_continuation(episode_id: str, *, invocation: int = 1) -> WatcherContinuation:
    return _continuation().model_copy(
        update={
            "patch_kind": "experiment_loop",
            "control_node_id": "exp-one",
            "control_revision": 0,
            "control_episode_id": episode_id,
            "control_invocation": invocation,
            "control_invocation_ceiling": 3,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
        }
    )


def _bound_episode(store: AppStore, episode_id: str, *, operation_id: str = "loop-root") -> None:
    root = _loop_task(store, operation_id, episode_id=episode_id, invocation=1, ceiling=3)
    store.create_experiment_episode_with_invocation(root)
    store.complete_agent_task(operation_id, applied_revision=None, result={})
    store.commit_experiment_episode_turn(
        episode_id=episode_id,
        project_id="project",
        control_node_id="exp-one",
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="native-loop-session",
        stage_host=None,
        stage_root="/tmp/loop-stage",
        chat_id="chat",
        operation_id=operation_id,
        invocation=1,
        graph_result="no graph change",
        watcher_ids=[],
        context_baseline={},
    )


def _maintenance_task(
    store: AppStore,
    operation_id: str,
    *,
    kind: str = "project_chat",
    mode: str = "work",
    node_id: str | None = None,
    chat_id: str = "maintenance-chat",
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id="project",
        kind=kind,
        status="queued",
        request={
            "chat_id": chat_id,
            "node_id": node_id,
            "provider": "claude",
            "model": "different-model",
            "reasoning": "high",
            "run_on": "different-machine",
            "run_truth_scope": ["different-scope"],
            "mode": mode,
            "trigger": "human",
            "patch_kind": "work",
            "workflow_ids": [],
            "skill_ids": [],
            "invoked_workflow_ids": [],
            "invoked_skill_ids": [],
            "resolved_skill_packages": [],
            "watcher_ids": [],
        },
        created_at=now,
        updated_at=now,
        status_message="Maintaining node observers.",
    )


def test_watcher_admission_is_node_scoped_not_conversation_provider_or_machine(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    store.create_agent_task(_maintenance_task(store, "maintenance"))
    resource = store.experiment_watcher_resources("project")[0]
    binding = WatcherBinding(
        project_id="project",
        origin_operation_id="maintenance",
        origin_task_kind="project_chat",
        chat_id="maintenance-chat",
        node_id="exp-one",
        execution_host=resource.execution_host,
        continuation=resource.continuation,
    )

    assert store.admit_experiment_watcher_maintenance(binding) == resource
    assert resource.wake_chat_id == "chat"
    assert resource.continuation.provider == "codex"
    assert resource.continuation.run_on == "laptop"

    store.create_agent_task(
        _maintenance_task(
            store,
            "other-node",
            kind="node_chat",
            node_id="different-experiment",
        )
    )
    with pytest.raises(ValueError, match="permission denied: node scope"):
        store.admit_experiment_watcher_maintenance(
            binding.model_copy(
                update={
                    "origin_operation_id": "other-node",
                    "origin_task_kind": "node_chat",
                }
            )
        )
    store.complete_agent_task("other-node", applied_revision=None, result={})

    store.create_agent_task(
        _maintenance_task(store, "discuss", kind="node_chat", mode="discuss", node_id="exp-one")
    )
    with pytest.raises(ValueError, match="Work capability is required"):
        store.admit_experiment_watcher_maintenance(
            binding.model_copy(
                update={"origin_operation_id": "discuss", "origin_task_kind": "node_chat"}
            )
        )


def test_cross_chat_maintenance_retires_and_replaces_without_rebinding_episode(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    store.create_agent_task(_maintenance_task(store, "maintenance"))
    old = _record("old", origin="loop-root").model_copy(
        update={"continuation": _loop_continuation(episode_id)}
    )
    store.create_watchers([old])
    resource = store.experiment_watcher_resources("project")[0]
    binding = WatcherBinding(
        project_id="project",
        origin_operation_id="maintenance",
        origin_task_kind="project_chat",
        chat_id="maintenance-chat",
        node_id="exp-one",
        execution_host=resource.execution_host,
        continuation=resource.continuation,
    )
    replacement = _record("replacement", origin="maintenance").model_copy(
        update={
            "origin_task_kind": "project_chat",
            "chat_id": "maintenance-chat",
            "episode_id": episode_id,
            "continuation": resource.continuation,
        }
    )
    before = store.experiment_episode(episode_id)

    stored = store.persist_experiment_watchers_idempotently(
        [replacement],
        stops=[WatcherStopRequest(stop_watcher_id="old", reason="Replaced observer")],
        binding=binding,
        expected_watcher_snapshot_token=resource.watcher_snapshot_token,
    )

    assert [item.watcher_id for item in stored] == ["replacement"]
    assert stored[0].chat_id == "maintenance-chat"
    assert stored[0].origin_task_kind == "project_chat"
    assert stored[0].node_id == "exp-one"
    assert stored[0].episode_id == episode_id
    assert stored[0].execution_host == resource.execution_host
    assert stored[0].continuation.provider == "codex"
    assert store.watcher("old").stop_operation_id == "maintenance"
    assert store.experiment_episode(episode_id) == before


def test_episode_origin_cannot_arm_a_watcher_for_another_episode(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    mismatched = _record("mismatched-episode", origin="loop-root").model_copy(
        update={"continuation": _loop_continuation(str(uuid.uuid4()))}
    )

    with pytest.raises(ValueError, match="cannot change its origin task graph binding"):
        store.create_watchers([mismatched])


def test_concurrent_maintenance_cannot_commit_against_a_stale_watcher_snapshot(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    store.create_agent_task(_maintenance_task(store, "first-maintenance"))
    store.create_agent_task(
        _maintenance_task(store, "second-maintenance", chat_id="second-maintenance-chat")
    )
    resource = store.experiment_watcher_resources("project")[0]

    def binding(operation_id: str, chat_id: str) -> WatcherBinding:
        return WatcherBinding(
            project_id="project",
            origin_operation_id=operation_id,
            origin_task_kind="project_chat",
            chat_id=chat_id,
            node_id="exp-one",
            execution_host=resource.execution_host,
            continuation=resource.continuation,
        )

    def replacement(watcher_id: str, operation_id: str, chat_id: str) -> WatcherRecord:
        return _record(watcher_id, origin=operation_id).model_copy(
            update={
                "origin_task_kind": "project_chat",
                "chat_id": chat_id,
                "episode_id": episode_id,
                "continuation": resource.continuation,
            }
        )

    store.persist_experiment_watchers_idempotently(
        [replacement("first-replacement", "first-maintenance", "maintenance-chat")],
        binding=binding("first-maintenance", "maintenance-chat"),
        expected_watcher_snapshot_token=resource.watcher_snapshot_token,
    )

    with pytest.raises(WatcherClaimConflict, match="changed after it was staged"):
        store.persist_experiment_watchers_idempotently(
            [
                replacement(
                    "second-replacement",
                    "second-maintenance",
                    "second-maintenance-chat",
                )
            ],
            binding=binding("second-maintenance", "second-maintenance-chat"),
            expected_watcher_snapshot_token=resource.watcher_snapshot_token,
        )
    assert store.watcher("second-replacement") is None


def test_observing_a_degraded_watcher_does_not_invalidate_maintenance(tmp_path) -> None:
    """Polling is not a claim.

    The motivating repair targets a degraded observer, and S84 re-checks one every
    few minutes, so a maintenance turn that fingerprinted observation would be
    rejected by the very watcher it exists to fix.
    """

    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    store.create_agent_task(_maintenance_task(store, "maintenance"))
    old = _record("old", origin="loop-root").model_copy(
        update={"continuation": _loop_continuation(episode_id)}
    )
    store.create_watchers([old])
    resource = store.experiment_watcher_resources("project")[0]

    # While the agent inspects the scheduler, the poller observes the broken check.
    store.record_watcher_check(
        "old", status="degraded", exit_code=127, error="squeue: command not found"
    )
    store.record_watcher_check(
        "old", status="degraded", exit_code=127, error="squeue: command not found"
    )

    binding = WatcherBinding(
        project_id="project",
        origin_operation_id="maintenance",
        origin_task_kind="project_chat",
        chat_id="maintenance-chat",
        node_id="exp-one",
        execution_host=resource.execution_host,
        continuation=resource.continuation,
    )
    replacement = _record("replacement", origin="maintenance").model_copy(
        update={
            "origin_task_kind": "project_chat",
            "chat_id": "maintenance-chat",
            "episode_id": episode_id,
            "continuation": resource.continuation,
        }
    )

    stored = store.persist_experiment_watchers_idempotently(
        [replacement],
        stops=[WatcherStopRequest(stop_watcher_id="old", reason="Replaced degraded observer")],
        binding=binding,
        expected_watcher_snapshot_token=resource.watcher_snapshot_token,
    )

    assert [item.watcher_id for item in stored] == ["replacement"]
    assert store.watcher("old").status == "stopped"


def test_a_retirement_another_turn_already_won_is_refused_per_item(tmp_path) -> None:
    """Membership is the fence; a resolved stop is caught by its own compare-and-swap."""

    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    store.create_agent_task(_maintenance_task(store, "maintenance"))
    old = _record("old", origin="loop-root").model_copy(
        update={"continuation": _loop_continuation(episode_id)}
    )
    store.create_watchers([old])
    resource = store.experiment_watcher_resources("project")[0]
    binding = WatcherBinding(
        project_id="project",
        origin_operation_id="maintenance",
        origin_task_kind="project_chat",
        chat_id="maintenance-chat",
        node_id="exp-one",
        execution_host=resource.execution_host,
        continuation=resource.continuation,
    )

    # Another turn retires the same observer first. Retirement keeps the row, so
    # membership is unchanged and the fingerprint still matches.
    store.stop_watchers("project", ["old"])

    with pytest.raises(ValueError, match="already resolved"):
        store.persist_experiment_watchers_idempotently(
            [],
            stops=[WatcherStopRequest(stop_watcher_id="old", reason="Replaced degraded observer")],
            binding=binding,
            expected_watcher_snapshot_token=resource.watcher_snapshot_token,
        )


def test_watcher_admission_fails_closed_after_stop_or_stale_episode(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    store.create_agent_task(_maintenance_task(store, "maintenance"))
    resource = store.experiment_watcher_resources("project")[0]
    binding = WatcherBinding(
        project_id="project",
        origin_operation_id="maintenance",
        origin_task_kind="project_chat",
        chat_id="maintenance-chat",
        node_id="exp-one",
        execution_host=resource.execution_host,
        continuation=resource.continuation,
    )

    stale = binding.model_copy(
        update={
            "continuation": binding.continuation.model_copy(
                update={"control_episode_id": str(uuid.uuid4())}
            )
        }
    )
    with pytest.raises(ValueError, match="stale episode"):
        store.admit_experiment_watcher_maintenance(stale)

    store.request_experiment_loop_stop("project", "exp-one")
    assert store.experiment_watcher_resources("project") == []
    with pytest.raises(ValueError, match="live, unstopped episode"):
        store.admit_experiment_watcher_maintenance(binding)


def test_watcher_episode_owner_migrates_and_backfills_before_indexing(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    store = AppStore(path)
    episode_id = str(uuid.uuid4())
    store.create_watchers(
        [_record("legacy-loop").model_copy(update={"continuation": _loop_continuation(episode_id)})]
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX watchers_episode")
        connection.execute("ALTER TABLE watchers RENAME COLUMN episode_id TO experiment_episode_id")

    reopened = AppStore(path)
    with reopened.connection() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(watchers)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(watchers)")}
        stored_episode_id = connection.execute(
            "SELECT episode_id FROM watchers WHERE watcher_id = 'legacy-loop'"
        ).fetchone()[0]

    assert "episode_id" in columns
    assert "watchers_episode" in indexes
    assert stored_episode_id == episode_id


def test_graph_condition_column_migrates_before_its_index_is_created(tmp_path) -> None:
    path = tmp_path / "legacy-graph-watcher.sqlite3"
    AppStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX watchers_graph_conditions")
        connection.execute("ALTER TABLE watchers DROP COLUMN graph_condition_json")
        connection.execute("ALTER TABLE watchers DROP COLUMN armed_revision")

    reopened = AppStore(path)
    with reopened.connection() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(watchers)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(watchers)")}

    assert "graph_condition_json" in columns
    assert "armed_revision" in columns
    assert "watchers_graph_conditions" in indexes


def test_agent_stop_is_atomic_idempotent_and_scoped_to_the_bound_episode(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    continuation = _loop_continuation(episode_id)
    binding = _binding("loop-root").model_copy(update={"continuation": continuation})
    old = _record("old-observer", origin="loop-root").model_copy(
        update={"continuation": continuation}
    )
    store.create_watchers([old])
    replacement = _record("replacement", origin="loop-root").model_copy(
        update={"continuation": continuation}
    )

    armed = store.persist_experiment_watchers_idempotently(
        [replacement],
        stops=[
            WatcherStopRequest(stop_watcher_id="old-observer", reason="Cancelled superseded job")
        ],
        binding=binding,
    )

    stopped = store.watcher("old-observer")
    assert stopped is not None
    assert stopped.status == "stopped"
    assert stopped.notified is True
    assert stopped.stopped_by == "agent"
    assert stopped.stop_reason == "Cancelled superseded job"
    assert stopped.stopped_at is not None
    assert stopped.stop_operation_id == "loop-root"
    assert [item.watcher_id for item in armed] == ["replacement"]
    assert "old-observer" not in {item.watcher_id for item in store.pollable_watchers()}
    assert all(
        "old-observer" not in {item.watcher_id for item in group}
        for group in store.completed_watcher_groups()
    )

    assert (
        store.persist_experiment_watchers_idempotently(
            [replacement],
            stops=[
                WatcherStopRequest(
                    stop_watcher_id="old-observer", reason="Cancelled superseded job"
                )
            ],
            binding=binding,
        )
        == armed
    )

    with pytest.raises(ValueError, match="unknown staged"):
        store.persist_experiment_watchers_idempotently(
            [
                _record("must-not-arm", origin="loop-root").model_copy(
                    update={"continuation": continuation}
                )
            ],
            stops=[WatcherStopRequest(stop_watcher_id="missing", reason="No longer useful")],
            binding=binding,
        )
    assert store.watcher("must-not-arm") is None


def test_stop_loop_absorbs_the_running_turn_s_own_watcher_retirement(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    continuation = _loop_continuation(episode_id)
    binding = _binding("loop-root").model_copy(update={"continuation": continuation})
    observer = _record("observer", origin="loop-root").model_copy(
        update={"continuation": continuation}
    )
    store.create_watchers([observer])

    store.request_experiment_loop_stop("project", "exp-one")
    stopped_by_loop = store.watcher("observer")
    assert stopped_by_loop is not None and stopped_by_loop.stopped_by == "loop"

    stops = [WatcherStopRequest(stop_watcher_id="observer", reason="Cancelled the job")]
    store.validate_experiment_agent_watcher_stops(binding, stops)
    replacement = _record("replacement", origin="loop-root").model_copy(
        update={"continuation": continuation}
    )
    armed = store.persist_experiment_watchers_idempotently(
        [replacement], stops=stops, binding=binding
    )

    retained = store.watcher("observer")
    assert retained is not None
    assert (retained.status, retained.stopped_by) == ("stopped", "loop")
    assert [item.status for item in armed] == ["stopped"]
    assert store.pollable_watchers() == []


def test_watcher_schedule_persists_backoff_and_resets_after_a_healthy_check(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    created_at = "2026-08-01T00:00:00+00:00"
    armed_due_times = {
        watcher_next_check_at(f"watcher-{index}", created_at, 0) for index in range(20)
    }
    assert len(armed_due_times) > 10
    for due_at in armed_due_times:
        delay = datetime.fromisoformat(due_at) - datetime.fromisoformat(created_at)
        assert 2 * 60 * 0.9 <= delay.total_seconds() <= 2 * 60 * 1.1

    store.create_watchers([_record("scheduled", status="active")])
    record = store.watcher("scheduled")
    assert record is not None and record.next_check_at is not None
    assert store.pollable_watchers(as_of="2026-08-01T00:01:00+00:00") == []

    checked_at = created_at
    expected_minutes = (2, 4, 8, 15, 30, 30)
    for error_count, minutes in enumerate(expected_minutes, start=1):
        record = store.record_watcher_check(
            "scheduled",
            status="degraded",
            exit_code=255,
            error="transport unavailable",
            checked_at=checked_at,
        )
        assert record.consecutive_error_count == error_count
        assert record.next_check_at is not None
        delay = datetime.fromisoformat(record.next_check_at) - datetime.fromisoformat(checked_at)
        assert minutes * 60 * 0.9 <= delay.total_seconds() <= minutes * 60 * 1.1
        checked_at = record.next_check_at

    reopened = AppStore(store.path)
    persisted = reopened.watcher("scheduled")
    assert persisted is not None and persisted.next_check_at == checked_at
    assert persisted.consecutive_error_count == 6
    assert (
        reopened.pollable_watchers(
            as_of=(datetime.fromisoformat(checked_at) - timedelta(seconds=1)).isoformat()
        )
        == []
    )

    healthy = reopened.record_watcher_check(
        "scheduled",
        status="active",
        exit_code=1,
        error=None,
        checked_at=checked_at,
    )
    assert healthy.consecutive_error_count == 0
    assert healthy.next_check_at == watcher_next_check_at("scheduled", checked_at, 0)
    assert healthy.last_error is None


def test_grouped_watchers_wait_for_all_members_then_claim_once(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    continuation = _loop_continuation(episode_id)
    first = _record("shard-a", origin="loop-root", status="completed").model_copy(
        update={
            "continuation": continuation,
            "group_id": "eval-group",
            "group_label": "eval-shards",
        }
    )
    second = _record("shard-b", origin="loop-root").model_copy(
        update={
            "continuation": continuation,
            "group_id": "eval-group",
            "group_label": "eval-shards",
        }
    )
    store.create_watchers([first, second])
    assert store.completed_watcher_groups() == []

    store.record_watcher_check(
        "shard-b",
        status="completed",
        exit_code=0,
        error=None,
        checked_at="2026-08-01T00:02:00+00:00",
    )
    groups = store.completed_watcher_groups()
    assert [[item.watcher_id for item in group] for group in groups] == [["shard-a", "shard-b"]]

    wake = _loop_task(
        store,
        "group-wake",
        episode_id=episode_id,
        invocation=2,
        ceiling=3,
        watcher_ids=["shard-a", "shard-b"],
    )
    wake = wake.model_copy(
        update={
            "request": {**wake.request, "session_id": "native-loop-session"},
            "native_session_id": "native-loop-session",
            "stage_root": "/tmp/loop-stage",
        }
    )
    assert store.create_watcher_notification_task(wake, ["shard-a", "shard-b"]) is not None
    assert all(item.notified for item in store.watchers("project"))
    assert store.completed_watcher_groups() == []


def test_agent_stopped_group_members_neither_block_nor_trigger_delivery(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    continuation = _loop_continuation(str(uuid.uuid4()))

    def group_member(watcher_id: str, group_id: str, status: str) -> WatcherRecord:
        return _record(watcher_id, status=status).model_copy(
            update={
                "continuation": continuation,
                "group_id": group_id,
                "group_label": group_id,
                "notified": status == "stopped",
                "stopped_by": "agent" if status == "stopped" else None,
                "stop_reason": "Cancelled obsolete shard" if status == "stopped" else None,
                "stopped_at": ("2026-08-01T00:01:00+00:00" if status == "stopped" else None),
            }
        )

    store.create_watchers(
        [
            group_member("remaining-complete", "partially-stopped", "completed"),
            group_member("retired-sibling", "partially-stopped", "stopped"),
        ]
    )
    store.create_watchers(
        [
            group_member("retired-a", "fully-stopped", "stopped"),
            group_member("retired-b", "fully-stopped", "stopped"),
        ]
    )

    assert [[item.watcher_id for item in group] for group in store.completed_watcher_groups()] == [
        ["remaining-complete"]
    ]


def test_fifth_group_observation_error_is_ready_but_remains_degraded(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    completed = _record("shard-complete", status="completed").model_copy(
        update={
            "continuation": _loop_continuation(episode_id),
            "group_id": "diagnostic-group",
            "group_label": "eval-shards",
        }
    )
    degraded = _record("shard-unknown", status="degraded").model_copy(
        update={
            "continuation": _loop_continuation(episode_id),
            "group_id": "diagnostic-group",
            "group_label": "eval-shards",
            "consecutive_error_count": 5,
            "last_error": "SSH unavailable",
        }
    )
    store.create_watchers([completed, degraded])

    groups = store.completed_watcher_groups()

    assert [{item.watcher_id for item in group} for group in groups] == [
        {"shard-complete", "shard-unknown"}
    ]
    unknown = store.watcher("shard-unknown")
    assert unknown is not None
    assert unknown.status == "degraded"
    assert unknown.consecutive_error_count == 5


def test_claimed_diagnostic_group_remains_history_not_live_work(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    completed = _record("shard-complete", origin="loop-root", status="completed").model_copy(
        update={
            "continuation": _loop_continuation(episode_id),
            "group_id": "diagnostic-group",
            "group_label": "eval-shards",
        }
    )
    degraded = _record("shard-unknown", origin="loop-root", status="degraded").model_copy(
        update={
            "continuation": _loop_continuation(episode_id),
            "group_id": "diagnostic-group",
            "group_label": "eval-shards",
            "consecutive_error_count": 5,
            "last_error": "SSH unavailable",
        }
    )
    store.create_watchers([completed, degraded])
    wake = _loop_task(
        store,
        "diagnostic-wake",
        episode_id=episode_id,
        invocation=2,
        ceiling=3,
        watcher_ids=["shard-complete", "shard-unknown"],
    )
    wake = wake.model_copy(
        update={
            "request": {**wake.request, "session_id": "native-loop-session"},
            "native_session_id": "native-loop-session",
            "stage_root": "/tmp/loop-stage",
        }
    )

    assert (
        store.create_experiment_watcher_invocation(
            wake,
            ["shard-complete", "shard-unknown"],
        )
        is not None
    )
    store.complete_agent_task("diagnostic-wake", applied_revision=None, result={})

    stored = store.watcher("shard-unknown")
    assert stored is not None
    assert stored.status == "degraded"
    assert stored.notified is True
    assert (
        store.record_watcher_check(
            "shard-unknown",
            status="active",
            exit_code=1,
            error=None,
        )
        == stored
    )
    assert store.pollable_watchers(as_of="2026-08-02T00:00:00+00:00") == []
    assert store.experiment_watcher_ids("project", "exp-one") == []
    runtime = store.experiment_loop_runtime("project", "exp-one")
    assert runtime.detached_work_active is False
    assert runtime.watcher_degraded is False
    assert runtime.active is False
    assert store.completed_watcher_groups() == []
    with pytest.raises(ValueError, match="missing, unready, or already notified"):
        store.create_watcher_notification_task(
            _loop_task(
                store,
                "diagnostic-wake-again",
                episode_id=episode_id,
                invocation=3,
                ceiling=3,
                watcher_ids=["shard-complete", "shard-unknown"],
            ),
            ["shard-complete", "shard-unknown"],
        )


def test_check_runs_from_declared_cwd_and_uses_exit_table(tmp_path) -> None:
    cwd = str(tmp_path)
    common = {"log_path": str(tmp_path / "job.log"), "cwd": cwd}

    complete = run_watcher_check(
        WatchSpec(check_command=f'test "$PWD" = {shlex.quote(cwd)}', **common)
    )
    active = run_watcher_check(WatchSpec(check_command="exit 1", **common))
    error = run_watcher_check(WatchSpec(check_command="echo broken >&2; exit 9", **common))

    assert complete.state == "complete"
    assert complete.exit_code == 0
    assert active.state == "active"
    assert active.exit_code == 1
    assert error.state == "error"
    assert error.exit_code == 9
    assert error.error == "check exited with status 9: broken"


def test_check_has_a_hard_timeout(tmp_path) -> None:
    result = run_watcher_check(
        WatchSpec(
            check_command="sleep 1",
            log_path=str(tmp_path / "job.log"),
            cwd=str(tmp_path),
        ),
        timeout=0.01,
    )

    assert result.state == "error"
    assert result.exit_code is None
    assert result.error == "check timed out after 0.01 seconds"


def test_remote_check_uses_existing_ssh_login_shell(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr("rcp.watchers.subprocess.run", fake_run)

    result = run_watcher_check(
        WatchSpec(check_command="squeue -h -j 4471", log_path="/logs/job", cwd="/work/a b"),
        "gpu.example",
    )

    command = seen["command"]
    assert isinstance(command, list)
    assert command[0] == "ssh"
    assert command[-2] == "gpu.example"
    assert shlex.split(command[-1]) == [
        "bash",
        "-lic",
        "cd '/work/a b' && squeue -h -j 4471",
    ]
    assert seen["kwargs"]["cwd"] is None
    assert result.state == "active"


def test_initial_error_arms_none_then_corrected_list_persists_atomically(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    specs = [
        WatchSpec(check_command="one", log_path="/tmp/one.log", cwd="/tmp"),
        WatchSpec(check_command="two", log_path="/tmp/two.log", cwd="/tmp"),
    ]

    def one_bad(spec: WatchSpec, _host: str, _timeout: float) -> WatcherCheckResult:
        if spec.check_command == "two":
            return WatcherCheckResult(
                state="error",
                checked_at="2026-08-01T00:00:00+00:00",
                exit_code=2,
                error="check exited with status 2",
            )
        return WatcherCheckResult(
            state="active", checked_at="2026-08-01T00:00:00+00:00", exit_code=1
        )

    with pytest.raises(WatcherInitialCheckError, match="watcher 2"):
        arm_watchers(store, specs, _binding(), check_runner=one_bad)
    assert store.watchers("project") == []

    def corrected(spec: WatchSpec, _host: str, _timeout: float) -> WatcherCheckResult:
        return WatcherCheckResult(
            state="complete" if spec.check_command == "two" else "active",
            checked_at="2026-08-01T00:01:00+00:00",
            exit_code=0 if spec.check_command == "two" else 1,
        )

    records = arm_watchers(store, specs, _binding(), check_runner=corrected)

    assert len(records) == 2
    assert {record.status for record in records} == {"active", "completed"}
    reopened = AppStore(store.path)
    assert {record.watcher_id for record in reopened.watchers("project")} == {
        record.watcher_id for record in records
    }


def test_runtime_error_degrades_only_that_watcher_and_later_clears(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("bad"), _record("done")])

    def first(spec: WatchSpec, _host: str, _timeout: float) -> WatcherCheckResult:
        if spec.log_path.endswith("bad.log"):
            return WatcherCheckResult(
                state="error",
                checked_at="2026-08-01T00:01:00+00:00",
                exit_code=255,
                error="ssh unavailable",
            )
        return WatcherCheckResult(
            state="complete", checked_at="2026-08-01T00:01:00+00:00", exit_code=0
        )

    groups = WatcherPoller(store, check_runner=first).poll_once()

    assert store.watcher("bad").status == "degraded"
    assert store.watcher("bad").last_error == "ssh unavailable"
    assert store.watcher("done").status == "completed"
    assert [[item.watcher_id for item in group] for group in groups] == [["done"]]

    def recovered(_spec: WatchSpec, _host: str, _timeout: float) -> WatcherCheckResult:
        return WatcherCheckResult(
            state="active", checked_at="2026-08-01T00:02:00+00:00", exit_code=1
        )

    WatcherPoller(store, check_runner=recovered).poll_once()

    assert store.watcher("bad").status == "active"
    assert store.watcher("bad").last_error is None


def test_manual_check_bypasses_backoff_and_keeps_healthy_scheduling(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    record = _record("manual").model_copy(update={"execution_host": "gpu.example"})
    store.create_watchers([record])
    degraded = store.record_watcher_check(
        "manual",
        status="degraded",
        exit_code=255,
        error="ssh unavailable",
        checked_at="2026-08-01T00:01:00+00:00",
    )
    assert degraded.next_check_at is not None
    assert degraded.next_check_at > "2026-08-01T00:01:30+00:00"
    calls: list[tuple[str, float]] = []

    def recovered(_spec: WatchSpec, host: str, timeout: float) -> WatcherCheckResult:
        calls.append((host, timeout))
        return WatcherCheckResult(
            state="active",
            checked_at="2026-08-01T00:01:30+00:00",
            exit_code=1,
        )

    poller = WatcherPoller(
        store,
        check_runner=recovered,
        timeout=7,
        clock=lambda: "2026-08-01T00:01:31+00:00",
    )
    updated = poller.check_now("project", "manual")

    assert calls == [("gpu.example", 7)]
    assert updated.status == "active"
    assert updated.consecutive_error_count == 0
    assert updated.last_error is None
    assert updated.next_check_at == watcher_next_check_at("manual", "2026-08-01T00:01:30+00:00", 0)
    assert poller.poll_once() == []
    assert calls == [("gpu.example", 7)]


def test_manual_completion_runs_the_ordinary_delivery_callback_once(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("manual-complete", status="degraded")])
    delivered: list[list[str]] = []

    def completed(_spec: WatchSpec, _host: str, _timeout: float) -> WatcherCheckResult:
        return WatcherCheckResult(
            state="complete",
            checked_at="2026-08-01T00:02:00+00:00",
            exit_code=0,
        )

    updated = WatcherPoller(
        store,
        check_runner=completed,
        on_completed=lambda group: delivered.append([item.watcher_id for item in group]),
    ).check_now("project", "manual-complete")

    assert updated.status == "completed"
    assert updated.next_check_at is None
    assert delivered == [["manual-complete"]]


def test_manual_error_advances_backoff_from_the_new_check_time(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("manual-error", status="degraded")])

    def still_broken(_spec: WatchSpec, _host: str, _timeout: float) -> WatcherCheckResult:
        return WatcherCheckResult(
            state="error",
            checked_at="2026-08-01T00:05:00+00:00",
            exit_code=255,
            error="still unavailable",
        )

    updated = WatcherPoller(store, check_runner=still_broken).check_now("project", "manual-error")

    assert updated.status == "degraded"
    assert updated.consecutive_error_count == 2
    assert updated.next_check_at == watcher_next_check_at(
        "manual-error", "2026-08-01T00:05:00+00:00", 2
    )


def test_manual_check_waits_for_a_scheduled_check_of_the_same_watcher(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("serialized", status="degraded")])
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    manual_started = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def blocked_check(_spec: WatchSpec, _host: str, _timeout: float) -> WatcherCheckResult:
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_started.set()
            release_first.wait(timeout=2)
        else:
            second_started.set()
        return WatcherCheckResult(
            state="error",
            checked_at=f"2026-08-01T00:0{call_number}:00+00:00",
            exit_code=255,
            error="still unavailable",
        )

    poller = WatcherPoller(
        store,
        check_runner=blocked_check,
        clock=lambda: "2027-08-01T00:00:00+00:00",
    )
    scheduled = threading.Thread(target=poller.poll_once)

    def check_manually() -> None:
        manual_started.set()
        poller.check_now("project", "serialized")

    manual = threading.Thread(target=check_manually)
    scheduled.start()
    assert first_started.wait(timeout=1)
    manual.start()
    assert manual_started.wait(timeout=1)
    assert not second_started.wait(timeout=0.1)

    release_first.set()
    scheduled.join(timeout=2)
    manual.join(timeout=2)

    assert not scheduled.is_alive()
    assert not manual.is_alive()
    assert second_started.is_set()
    assert calls == 2


def test_manual_check_rejects_missing_graph_and_ineligible_watchers(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers(
        [
            _record("active"),
            _record("already-notified", status="degraded").model_copy(update={"notified": True}),
        ]
    )
    store.create_watchers(
        [
            GraphWatcherRecord(
                watcher_id="graph",
                project_id="project",
                origin_operation_id="origin-graph",
                origin_task_kind="node_chat",
                chat_id="chat",
                node_id="exp-one",
                continuation=_continuation(),
                condition=NodeStatusGraphCondition(
                    node_id="exp-one",
                    status_in=["resolved"],
                ),
                armed_revision=0,
                created_at="2026-08-01T00:00:00+00:00",
            )
        ]
    )
    poller = WatcherPoller(store)

    with pytest.raises(KeyError):
        poller.check_now("project", "missing")
    with pytest.raises(KeyError):
        poller.check_now("wrong-project", "active")
    with pytest.raises(ValueError, match="external watcher"):
        poller.check_now("project", "graph")
    with pytest.raises(ValueError, match="degraded watcher awaiting delivery"):
        poller.check_now("project", "active")
    with pytest.raises(ValueError, match="degraded watcher awaiting delivery"):
        poller.check_now("project", "already-notified")


def test_completed_groups_do_not_merge_different_origin_policies(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("one", status="completed")])
    different_policy = _record("two", status="completed").model_copy(
        update={
            "continuation": _continuation().model_copy(
                update={
                    "patch_kind": "experiment_loop",
                    "control_node_id": "exp-one",
                    "control_episode_id": str(uuid.uuid4()),
                    "control_invocation": 1,
                    "control_invocation_ceiling": 2,
                }
            )
        }
    )
    store.create_watchers([different_policy])

    groups = store.completed_watcher_groups()

    assert {tuple(item.watcher_id for item in group) for group in groups} == {("one",), ("two",)}


def test_completed_groups_merge_compatible_watchers_from_different_work_turns(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("one", origin="work-one", status="completed")])
    store.create_watchers([_record("two", origin="work-two", status="completed")])

    groups = store.completed_watcher_groups()

    assert [[item.watcher_id for item in group] for group in groups] == [["one", "two"]]
    queued = store.create_watcher_notification_task(
        _task(store, "watcher-turn", ["one", "two"]), ["one", "two"]
    )
    assert queued is not None
    assert all(item.notified for item in store.watchers("project"))


def test_queue_and_notified_ledger_are_atomic_and_wait_behind_live_task(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("one", status="completed"), _record("two", status="completed")])
    live = _task(store, "human-turn", [])
    store.create_agent_task(live)

    assert (
        store.create_watcher_notification_task(
            _task(store, "watcher-turn-blocked", ["one", "two"]), ["one", "two"]
        )
        is None
    )
    assert store.agent_task("watcher-turn-blocked") is None
    assert all(not item.notified for item in store.watchers("project"))

    store.fail_agent_task("human-turn", "done")
    queued = store.create_watcher_notification_task(
        _task(store, "watcher-turn", ["one", "two"]), ["one", "two"]
    )

    assert queued is not None
    assert queued.status == "queued"
    reopened = AppStore(store.path)
    assert reopened.agent_task("watcher-turn") is not None
    assert all(item.notified for item in reopened.watchers("project"))
    assert {item.notification_operation_id for item in reopened.watchers("project")} == {
        "watcher-turn"
    }


def test_a_human_release_takes_a_watcher_out_of_the_polling_set(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("watch-live"), _record("watch-degraded")])
    store.record_watcher_check(
        "watch-degraded",
        status="degraded",
        exit_code=255,
        error="ssh: connect to host gpu01 port 22: No route to host",
    )
    assert {record.watcher_id for record in store.pollable_watchers()} == {"watch-live"}

    stopped = store.stop_watchers("project", ["watch-degraded"])

    assert [record.status for record in stopped] == ["stopped"]
    assert [record.watcher_id for record in store.pollable_watchers()] == ["watch-live"]
    # A stopped watcher is already accounted for, so it can never wake a turn.
    assert store.watcher("watch-degraded").notified is True
    assert store.completed_watcher_groups() == []


def test_experiment_watchers_are_found_by_the_loop_that_armed_them(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    bound = _record("watch-bound")
    bound = bound.model_copy(
        update={
            "continuation": bound.continuation.model_copy(
                update={
                    "patch_kind": "experiment_loop",
                    "control_node_id": "exp/one",
                    "control_episode_id": str(uuid.uuid4()),
                    "control_invocation": 1,
                    "control_invocation_ceiling": 2,
                }
            )
        }
    )
    store.create_watchers([bound])
    store.create_watchers([_record("watch-plain")])

    assert store.experiment_watcher_ids("project", "exp/one") == ["watch-bound"]
    assert store.experiment_watcher_ids("project", "exp/other") == []

    store.stop_watchers("project", ["watch-bound"])
    assert store.experiment_watcher_ids("project", "exp/one") == []


def test_loop_root_invocations_are_sequential_and_recovery_preserves_binding(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    first = _loop_task(store, "first", episode_id=episode_id, invocation=1, ceiling=4)
    store.create_experiment_episode_with_invocation(first)
    store.fail_agent_task("first", "provider failed")

    changed = _loop_task(
        store,
        "bad-retry",
        episode_id=episode_id,
        invocation=1,
        ceiling=4,
        parent_operation_id="first",
    )
    changed.request["control_revision"] = 1
    with pytest.raises(ValueError, match="preserve its control binding"):
        store.create_experiment_recovery_task(changed)

    recovery = _loop_task(
        store,
        "retry",
        episode_id=episode_id,
        invocation=1,
        ceiling=4,
        parent_operation_id="first",
    )
    store.create_experiment_recovery_task(recovery)
    runtime = store.experiment_loop_runtime("project", "exp-one")
    assert runtime.invocations_used == 1
    assert runtime.episode_id == episode_id
    store.complete_agent_task("retry", applied_revision=None, result={})

    skipped = _loop_task(store, "third", episode_id=episode_id, invocation=3, ceiling=4)
    skipped.request["trigger"] = "watcher"
    with pytest.raises(ValueError, match="out of sequence; expected 2"):
        store.create_experiment_watcher_invocation(skipped, [])


def test_ceiling_refuses_wake_without_consuming_pending_completion(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    old_episode = str(uuid.uuid4())
    first = _loop_task(store, "first", episode_id=old_episode, invocation=1, ceiling=1)
    store.create_experiment_episode_with_invocation(first)
    store.complete_agent_task("first", applied_revision=None, result={})
    watcher = _record("done", status="completed").model_copy(
        update={
            "continuation": _continuation().model_copy(
                update={
                    "patch_kind": "experiment_loop",
                    "control_node_id": "exp-one",
                    "control_episode_id": old_episode,
                    "control_invocation": 1,
                    "control_invocation_ceiling": 1,
                }
            )
        }
    )
    store.create_watchers([watcher])

    over_budget = _loop_task(
        store,
        "over-budget",
        episode_id=old_episode,
        invocation=2,
        ceiling=1,
        watcher_ids=["done"],
    )
    with pytest.raises(EpisodeInvocationCeilingReached, match="spent"):
        store.create_experiment_watcher_invocation(over_budget, ["done"])
    assert store.watcher("done").notified is False


def test_runtime_distinguishes_detached_work_from_a_pending_completion_at_ceiling(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    first = _loop_task(store, "first", episode_id=episode_id, invocation=1, ceiling=1)
    store.create_experiment_episode_with_invocation(first)
    store.complete_agent_task("first", applied_revision=None, result={})
    watcher = _record("bounded-work").model_copy(
        update={
            "continuation": _continuation().model_copy(
                update={
                    "patch_kind": "experiment_loop",
                    "control_node_id": "exp-one",
                    "control_episode_id": episode_id,
                    "control_invocation": 1,
                    "control_invocation_ceiling": 1,
                }
            )
        }
    )
    store.create_watchers([watcher])

    running = store.experiment_loop_runtime("project", "exp-one")
    assert running.detached_work_active is True
    assert running.watcher_completion_pending is False
    assert running.paused is True

    store.record_watcher_check(
        "bounded-work",
        status="completed",
        exit_code=0,
        error=None,
    )
    completed = store.experiment_loop_runtime("project", "exp-one")
    assert completed.detached_work_active is False
    assert completed.watcher_completion_pending is True
    assert completed.paused is True


def test_new_episode_adopts_remaining_watchers_without_mutating_their_origin(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    old_episode = str(uuid.uuid4())
    watcher = _record("still-running").model_copy(
        update={
            "continuation": _continuation().model_copy(
                update={
                    "patch_kind": "experiment_loop",
                    "control_node_id": "exp-one",
                    "control_revision": 1,
                    "control_episode_id": old_episode,
                    "control_invocation": 1,
                    "control_invocation_ceiling": 1,
                }
            )
        }
    )
    store.create_watchers([watcher])
    new_episode = str(uuid.uuid4())
    root = _loop_task(
        store,
        "reauthorized",
        episode_id=new_episode,
        invocation=1,
        ceiling=3,
    )
    store.create_experiment_episode_with_invocation(root)
    store.complete_agent_task("reauthorized", applied_revision=None, result={})

    runtime = store.experiment_loop_runtime("project", "exp-one")

    assert runtime.episode_id == new_episode
    assert runtime.invocations_used == 1
    assert runtime.active is True
    assert store.watcher("still-running").continuation.control_episode_id == old_episode


def test_exit_receipt_on_recovery_child_requires_a_new_human_episode(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    root = _loop_task(store, "root", episode_id=episode_id, invocation=1, ceiling=3)
    store.create_experiment_episode_with_invocation(root)
    store.fail_agent_task("root", "provider failed")
    child = _loop_task(
        store,
        "repair",
        episode_id=episode_id,
        invocation=1,
        ceiling=3,
        parent_operation_id="root",
    )
    store.create_experiment_recovery_task(child)
    store.complete_agent_task("repair", applied_revision=4, result={})
    store.record_agent_task_receipt(
        "repair",
        "experiment_loop_exit",
        {"episode_id": episode_id, "invocation": 1},
    )
    watcher = _record("pending", status="completed").model_copy(
        update={
            "continuation": _continuation().model_copy(
                update={
                    "patch_kind": "experiment_loop",
                    "control_node_id": "exp-one",
                    "control_revision": 0,
                    "control_episode_id": episode_id,
                    "control_invocation": 1,
                    "control_invocation_ceiling": 3,
                }
            )
        }
    )
    store.create_watchers([watcher])

    runtime = store.experiment_loop_runtime("project", "exp-one")

    assert runtime.episode_exited is True
    assert runtime.active is False
    assert runtime.paused is False


def test_operational_recovery_rejects_siblings_and_successful_tasks(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    root = _loop_task(store, "root", episode_id=episode_id, invocation=1, ceiling=3)
    store.create_experiment_episode_with_invocation(root)
    store.fail_agent_task("root", "failed")
    child = _loop_task(
        store,
        "child",
        episode_id=episode_id,
        invocation=1,
        ceiling=3,
        parent_operation_id="root",
    )
    store.create_experiment_recovery_task(child)
    store.fail_agent_task("child", "failed again")

    sibling = child.model_copy(update={"operation_id": "sibling", "parent_operation_id": "root"})
    with pytest.raises(ValueError, match="already has a recovery child"):
        store.create_experiment_recovery_task(sibling)

    successful_store = AppStore(tmp_path / "successful.sqlite3")
    successful_episode = str(uuid.uuid4())
    successful = _loop_task(
        successful_store,
        "successful",
        episode_id=successful_episode,
        invocation=1,
        ceiling=3,
    )
    successful_store.create_experiment_episode_with_invocation(successful)
    successful_store.complete_agent_task("successful", applied_revision=None, result={})
    invalid_retry = _loop_task(
        successful_store,
        "retry-success",
        episode_id=successful_episode,
        invocation=1,
        ceiling=3,
        parent_operation_id="successful",
    )
    with pytest.raises(ValueError, match="latest unresolved loop task"):
        successful_store.create_experiment_recovery_task(invalid_retry)


def test_patch_only_graph_repair_is_not_treated_as_operational_recovery(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    root = _loop_task(store, "root", episode_id=episode_id, invocation=1)
    store.create_experiment_episode_with_invocation(root)
    store.complete_agent_task(
        "root",
        applied_revision=None,
        result={
            "graph_update": {
                "status": "rejected",
                "repairable": False,
            }
        },
    )
    repair = _loop_task(
        store,
        "repair",
        episode_id=episode_id,
        invocation=1,
        parent_operation_id="root",
    )

    stored = store.create_experiment_recovery_task(repair)

    assert stored.parent_operation_id == "root"
    assert stored.request["control_invocation"] == 1


def test_experiment_groups_coalesce_across_origin_episode_provenance(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    records = []
    for watcher_id, revision, invocation, ceiling in (
        ("old", 1, 1, 2),
        ("new", 9, 3, 5),
    ):
        record = _record(watcher_id, status="completed")
        records.append(
            record.model_copy(
                update={
                    "continuation": record.continuation.model_copy(
                        update={
                            "patch_kind": "experiment_loop",
                            "control_node_id": "exp-one",
                            "control_revision": revision,
                            "control_episode_id": str(uuid.uuid4()),
                            "control_invocation": invocation,
                            "control_invocation_ceiling": ceiling,
                        }
                    )
                }
            )
        )
    for record in records:
        store.create_watchers([record])

    groups = store.completed_watcher_groups()

    assert [[item.watcher_id for item in group] for group in groups] == [["new", "old"]]


def test_notification_claim_rejects_forged_scope_without_consuming_watchers(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("done", status="completed")])
    forged = _task(store, "forged", ["done"])
    forged.request["provider"] = "claude"

    with pytest.raises(ValueError, match="immutable delivery policy"):
        store.create_watcher_notification_task(forged, ["done"])

    assert store.watcher("done").notified is False
    assert store.agent_task("forged") is None


def test_stop_acknowledges_pending_completion_and_conflicts_after_claim(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("pending", status="completed")])

    stopped = store.stop_watchers("project", ["pending"])

    assert stopped[0].status == "stopped"
    assert stopped[0].notified is True
    assert store.completed_watcher_groups() == []

    store.create_watchers([_record("claimed", status="completed")])
    assert store.create_watcher_notification_task(
        _task(store, "delivery", ["claimed"]), ["claimed"]
    )
    with pytest.raises(WatcherClaimConflict, match="already claimed"):
        store.stop_watchers("project", ["claimed"])


def test_legacy_delivery_terminalizes_watchers_and_episode_diagnostic_atomically(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    _bound_episode(store, episode_id)
    with store.connection() as connection:
        connection.execute(
            """
            UPDATE graph_runs
            SET authorized_space_id = NULL, authorized_user_id = NULL,
                authorized_display_name = NULL
            WHERE operation_id = 'loop-root'
            """
        )
    watcher = _record("legacy-loop", origin="loop-root", status="completed").model_copy(
        update={
            "episode_id": episode_id,
            "continuation": _loop_continuation(episode_id),
        }
    )
    store.create_watchers([watcher])

    authorized_by, diagnostic = store.resolve_watcher_delivery_authorizer([watcher.watcher_id])

    terminal = store.watcher(watcher.watcher_id)
    episode = store.experiment_episode(episode_id)
    assert authorized_by is None
    assert diagnostic is not None
    assert "predates durable human attribution" in diagnostic
    assert terminal is not None
    assert terminal.status == "stopped"
    assert terminal.notified is True
    assert terminal.stop_reason == diagnostic
    assert episode is not None
    assert episode.session_diagnostic == diagnostic
    assert store.completed_watcher_groups() == []


def test_authorizer_terminalization_loses_cleanly_to_notification_claim(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_watchers([_record("claimed-first", status="completed")])
    assert store.create_watcher_notification_task(
        _task(store, "delivery-first", ["claimed-first"]),
        ["claimed-first"],
    )

    authorized_by, diagnostic = store.resolve_watcher_delivery_authorizer(["claimed-first"])

    claimed = store.watcher("claimed-first")
    assert authorized_by is None
    assert diagnostic is None
    assert claimed is not None
    assert claimed.status == "completed"
    assert claimed.notified is True
    assert claimed.notification_operation_id == "delivery-first"
    assert claimed.stop_reason is None


def test_poller_isolates_completion_callback_failures_between_groups(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    first = _record("first", status="completed")
    second = _record("second", status="completed").model_copy(
        update={"continuation": _continuation().model_copy(update={"model": "other"})}
    )
    store.create_watchers([first])
    store.create_watchers([second])
    called: list[str] = []

    def callback(group: list[WatcherRecord]) -> None:
        called.append(group[0].watcher_id)
        if group[0].watcher_id == "first":
            raise RuntimeError("one bad group")

    WatcherPoller(store, on_completed=callback).poll_once()

    assert called == ["first", "second"]
