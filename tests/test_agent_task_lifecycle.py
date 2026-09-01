from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from rcp.storage import (
    ACTIVE_AGENT_TASK_STATUSES,
    AgentTaskEventRecord,
    AgentTaskStatus,
    AppStore,
)
from tests.helpers import agent_patch_json, seed_patch
from tests.test_auto_research_children_storage import (
    _auto_parent,
    _project,
    _work_pair,
)

Operation = Literal[
    "mark_running",
    "request_pause",
    "pause",
    "complete",
    "fail",
    "interrupted_fail",
    "bulk_restart_interrupt",
]

_STATUSES: tuple[AgentTaskStatus, ...] = (
    "queued",
    "running",
    "pausing",
    "paused",
    "succeeded",
    "failed",
    "interrupted",
)
_OPERATIONS: tuple[Operation, ...] = (
    "mark_running",
    "request_pause",
    "pause",
    "complete",
    "fail",
    "interrupted_fail",
    "bulk_restart_interrupt",
)
_TERMINAL_STATUSES: tuple[AgentTaskStatus, ...] = (
    "paused",
    "succeeded",
    "failed",
    "interrupted",
)
_PATCH_OUTPUT = agent_patch_json(seed_patch())


@dataclass(frozen=True)
class LifecycleExpectation:
    status: AgentTaskStatus
    events: int
    receipts: int
    patch_retained: bool
    notices: int


def _expected(source_status: AgentTaskStatus, operation: Operation) -> LifecycleExpectation:
    active = source_status in ACTIVE_AGENT_TASK_STATUSES
    if operation == "mark_running":
        return LifecycleExpectation(
            status="running" if source_status == "queued" else source_status,
            events=1,
            receipts=0,
            patch_retained=True,
            notices=0,
        )
    if operation == "request_pause":
        return LifecycleExpectation(
            status="pausing" if source_status in {"queued", "running"} else source_status,
            events=1 if source_status in {"queued", "running"} else 0,
            receipts=0,
            patch_retained=True,
            notices=0,
        )
    if operation == "pause":
        return LifecycleExpectation(
            status="paused" if active else source_status,
            events=1,
            receipts=1 if active else 0,
            patch_retained=True,
            notices=1 if active else 0,
        )
    if operation == "complete":
        return LifecycleExpectation(
            status="succeeded" if active else source_status,
            events=1,
            receipts=1 if active else 0,
            patch_retained=not active,
            notices=1 if active else 0,
        )
    if operation == "fail":
        return LifecycleExpectation(
            status="failed" if active else source_status,
            events=1,
            receipts=1 if active else 0,
            patch_retained=True,
            notices=1 if active else 0,
        )
    if operation == "interrupted_fail":
        return LifecycleExpectation(
            status="interrupted" if active else source_status,
            events=1,
            receipts=1 if active else 0,
            patch_retained=True,
            notices=1 if active else 0,
        )
    return LifecycleExpectation(
        status="interrupted" if active else source_status,
        events=1 if active else 0,
        receipts=1 if active else 0,
        patch_retained=True,
        notices=1 if active else 0,
    )


def _is_single_task_refusal(source_status: AgentTaskStatus, operation: Operation) -> bool:
    return (operation == "mark_running" and source_status != "queued") or (
        operation in {"pause", "complete", "fail", "interrupted_fail"}
        and source_status not in ACTIVE_AGENT_TASK_STATUSES
    )


def _cases() -> list[object]:
    cases: list[object] = []
    for operation in _OPERATIONS:
        for source_status in _STATUSES:
            cases.append(
                pytest.param(
                    source_status,
                    operation,
                    id=f"{operation}-{source_status}",
                )
            )
    return cases


def _notice_count(store: AppStore) -> int:
    with store.connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM auto_research_lifecycle_notices"
        ).fetchone()
    assert row is not None
    return int(row["count"])


def _assert_truthful_refusal_event(
    events: list[AgentTaskEventRecord],
    *,
    before_events: int,
    source_status: AgentTaskStatus,
    operation: Operation,
) -> None:
    appended = events[before_events:]
    assert len(appended) == 1
    event = appended[0]
    assert event.level == "warning"
    message = event.message.lower()
    assert "refused" in message
    assert source_status in message
    if operation == "pause":
        assert event.message == f"Pause refused: this task already {source_status}."


def _prepare_task(tmp_path: Path, source_status: AgentTaskStatus) -> tuple[AppStore, str]:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _project(store)
    episode, root = _auto_parent(store)
    route, task = _work_pair(store, episode, root, worker_id="lifecycle-worker")
    store.create_auto_research_child_work(route, task)
    if source_status != "queued":
        with store.connection() as connection:
            connection.execute(
                "UPDATE graph_runs SET status = ?, phase = ? WHERE operation_id = ?",
                (source_status, source_status, task.operation_id),
            )
    store.record_agent_task_patch_output(task.operation_id, _PATCH_OUTPUT)
    return store, task.operation_id


def _run_operation(store: AppStore, operation: Operation, operation_id: str) -> None:
    if operation == "mark_running":
        store.mark_agent_task_running(operation_id)
    elif operation == "request_pause":
        store.request_agent_task_pause(operation_id)
    elif operation == "pause":
        store.pause_agent_task(operation_id)
    elif operation == "complete":
        store.complete_agent_task(operation_id, applied_revision=None, result={})
    elif operation == "fail":
        store.fail_agent_task(operation_id, "lifecycle test failure")
    elif operation == "interrupted_fail":
        store.fail_agent_task(
            operation_id,
            "lifecycle test interruption",
            status="interrupted",
        )
    else:
        store.interrupt_active_agent_tasks()


@pytest.mark.parametrize(
    "operation",
    (
        "mark_running",
        "update_message",
        "request_pause",
        "pause",
        "complete",
        "fail",
        "interrupted_fail",
    ),
)
def test_missing_single_id_lifecycle_writes_nothing(tmp_path: Path, operation: str) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    operation_id = "missing-operation"

    with pytest.raises(KeyError, match=operation_id):
        if operation == "mark_running":
            store.mark_agent_task_running(operation_id)
        elif operation == "update_message":
            store.update_agent_task_message(operation_id, "progress", event=True)
        elif operation == "request_pause":
            store.request_agent_task_pause(operation_id)
        elif operation == "pause":
            store.pause_agent_task(operation_id)
        elif operation == "complete":
            store.complete_agent_task(operation_id, applied_revision=None, result={})
        elif operation == "fail":
            store.fail_agent_task(operation_id, "missing failure")
        else:
            store.fail_agent_task(operation_id, "missing interruption", status="interrupted")

    assert store.agent_task(operation_id) is None
    assert store.agent_task_events(operation_id) == []
    assert store.agent_task_receipts(operation_id) == []
    assert store.agent_task_patch_output(operation_id) is None
    assert _notice_count(store) == 0


@pytest.mark.parametrize("source_status", _TERMINAL_STATUSES)
def test_terminal_progress_update_is_a_quiet_noop(
    tmp_path: Path,
    source_status: AgentTaskStatus,
) -> None:
    store, operation_id = _prepare_task(tmp_path, source_status)
    before_events = store.agent_task_events(operation_id)
    before_receipts = store.agent_task_receipts(operation_id)
    before_notices = _notice_count(store)

    store.update_agent_task_message(operation_id, "late progress", event=True)

    task = store.agent_task(operation_id)
    assert task is not None
    assert task.status == source_status
    assert store.agent_task_events(operation_id) == before_events
    assert store.agent_task_receipts(operation_id) == before_receipts
    assert _notice_count(store) == before_notices
    assert store.agent_task_patch_output(operation_id) == _PATCH_OUTPUT


@pytest.mark.parametrize(("source_status", "operation"), _cases())
def test_agent_task_lifecycle_matrix(
    tmp_path: Path,
    source_status: AgentTaskStatus,
    operation: Operation,
) -> None:
    store, operation_id = _prepare_task(tmp_path, source_status)
    before_events = len(store.agent_task_events(operation_id))
    before_receipts = len(store.agent_task_receipts(operation_id))
    before_notices = _notice_count(store)

    expected = _expected(source_status, operation)
    if operation == "request_pause" and source_status not in {"queued", "running"}:
        with pytest.raises(ValueError, match="Only a queued or running operation can be paused"):
            _run_operation(store, operation, operation_id)
    else:
        _run_operation(store, operation, operation_id)

    task = store.agent_task(operation_id)
    assert task is not None
    events = store.agent_task_events(operation_id)
    if _is_single_task_refusal(source_status, operation):
        _assert_truthful_refusal_event(
            events,
            before_events=before_events,
            source_status=source_status,
            operation=operation,
        )
    assert {
        "status": task.status,
        "events": len(events) - before_events,
        "receipts": len(store.agent_task_receipts(operation_id)) - before_receipts,
        "patch_retained": store.agent_task_patch_output(operation_id) is not None,
        "notices": _notice_count(store) - before_notices,
    } == {
        "status": expected.status,
        "events": expected.events,
        "receipts": expected.receipts,
        "patch_retained": expected.patch_retained,
        "notices": expected.notices,
    }
