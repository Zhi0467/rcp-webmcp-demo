from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rcp.storage import AgentTaskRecord, AppStore


def _task(
    operation_id: str,
    created_at: datetime,
    *,
    project_id: str = "project",
    parent_operation_id: str | None = None,
) -> AgentTaskRecord:
    timestamp = created_at.isoformat()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        kind="refresh",
        status="succeeded",
        request={},
        created_at=timestamp,
        updated_at=timestamp,
        finished_at=timestamp,
        status_message="done",
        parent_operation_id=parent_operation_id,
    )


def _backdate_operational_rows(store: AppStore, when: datetime) -> None:
    """Age every run row and its trace payloads past all retention windows."""

    timestamp = when.isoformat()
    with store.connection() as connection:
        for table in ("graph_runs", "graph_run_events", "graph_run_receipts"):
            connection.execute(f"UPDATE {table} SET created_at = ?", (timestamp,))


def _chat_attempt(
    store: AppStore,
    operation_id: str,
    *,
    parent: str | None,
    resumed: bool,
    attempt: int,
    graph_revision: int,
    status: str = "paused",
) -> None:
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id="project",
            kind="node_chat",
            status=status,  # type: ignore[arg-type]
            request={},
            created_at=now,
            updated_at=now,
            status_message="fixture",
            parent_operation_id=parent,
            attempt=attempt,
        )
    )
    store.record_agent_task_receipt(
        operation_id,
        "operation_created",
        {
            "kind": "node_chat",
            "attempt": attempt,
            "has_parent": parent is not None,
            "resumed": resumed,
        },
    )
    store.record_agent_task_receipt(
        operation_id,
        "chat_context_assembled",
        {"graph_revision": graph_revision},
    )
    store.record_agent_task_receipt(
        operation_id, "provider_diagnostics", {"detail": "bulk"}, tier="diagnostic"
    )
    store.record_agent_task_event(operation_id, "attempt trace")


def test_operational_prune_never_deletes_run_rows(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = datetime(2026, 7, 29, tzinfo=UTC)
    store.create_agent_task(_task("ancient", now - timedelta(days=4000)))
    for index in range(105):
        created = now - timedelta(days=2000, seconds=index)
        parent = "ancient" if index == 0 else None
        store.create_agent_task(
            _task(f"operation-{index:03d}", created, parent_operation_id=parent)
        )

    result = store.prune_operational_storage(now=now)

    assert "tasks" not in result
    assert store.agent_task("ancient") is not None
    assert store.agent_task("operation-000") is not None
    assert store.agent_task("operation-104") is not None
    with store.connection() as connection:
        count = connection.execute("SELECT COUNT(*) FROM graph_runs").fetchone()[0]
    assert count == 106


def test_operational_prune_ages_out_payloads_but_not_summary_receipts(tmp_path) -> None:
    store = AppStore(tmp_path / "payloads.sqlite3")
    now = datetime(2026, 7, 29, tzinfo=UTC)
    _chat_attempt(store, "old", parent=None, resumed=False, attempt=1, graph_revision=1)
    store.record_agent_task_patch_output("old", '{"kind":"node_chat"}')
    _backdate_operational_rows(store, now - timedelta(days=200))
    with store.connection() as connection:
        connection.execute(
            "UPDATE graph_run_outputs SET created_at = ?",
            ((now - timedelta(days=200)).isoformat(),),
        )

    result = store.prune_operational_storage(now=now)

    assert result["outputs"] == 1
    assert result["events"] == 1
    assert result["receipts"] == 1
    assert store.agent_task_patch_output("old") is None
    categories = {receipt.category for receipt in store.agent_task_receipts("old")}
    assert categories == {
        "operation_admitted",
        "operation_created",
        "chat_context_assembled",
    }


def test_operational_prune_keeps_command_ledger_while_aging_message_events(tmp_path) -> None:
    store = AppStore(tmp_path / "commands.sqlite3")
    now = datetime(2026, 7, 29, tzinfo=UTC)
    store.create_agent_task(_task("old", now - timedelta(days=200)))
    store.record_agent_task_event("old", "ordinary trace message")
    command = store.start_agent_command(
        operation_id="old",
        command_id="command",
        episode_id=None,
        verb="validate",
        idempotency_key=None,
        payload={"request_id": "request", "arguments": {}},
    )
    store.finish_agent_command(
        command.command_id,
        status="ok",
        payload={"result": {"status": "valid"}},
        message="validation completed",
    )
    _backdate_operational_rows(store, now - timedelta(days=200))

    result = store.prune_operational_storage(now=now)

    assert result["events"] == 1
    retained = store.agent_command(command.command_id)
    assert retained is not None
    assert retained.started_at
    assert retained.exited_at is not None
    assert retained.status == "ok"


def test_operational_prune_leaves_active_work_alone(tmp_path) -> None:
    store = AppStore(tmp_path / "active.sqlite3")
    now = datetime(2026, 7, 29, tzinfo=UTC)
    _chat_attempt(
        store, "live", parent=None, resumed=False, attempt=1, graph_revision=1, status="running"
    )
    store.record_agent_task_patch_output("live", '{"kind":"node_chat"}')
    _backdate_operational_rows(store, now - timedelta(days=400))
    with store.connection() as connection:
        connection.execute(
            "UPDATE graph_run_outputs SET created_at = ?",
            ((now - timedelta(days=400)).isoformat(),),
        )

    result = store.prune_operational_storage(now=now)

    assert result["outputs"] == 0
    assert result["events"] == 0
    assert result["receipts"] == 0
    assert store.agent_task_patch_output("live") is not None
    assert len(store.agent_task_events("live")) == 1


def test_operational_prune_bounds_recovery_outputs_and_writing_sessions(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = datetime(2026, 7, 29, tzinfo=UTC)
    failed = _task("failed", now - timedelta(days=8)).model_copy(update={"status": "failed"})
    store.create_agent_task(failed)
    store.record_agent_task_patch_output("failed", '{"kind":"refresh"}')
    with store.connection() as connection:
        connection.execute(
            "UPDATE graph_run_outputs SET created_at = ? WHERE operation_id = 'failed'",
            ((now - timedelta(days=8)).isoformat(),),
        )
    for index in range(55):
        resumed = (now - timedelta(days=200, seconds=index)).isoformat()
        with store.connection() as connection:
            connection.execute(
                """
                INSERT INTO writing_sessions (
                    native_session_id, provider, execution_machine, project_id,
                    model, created_at, last_resumed_at, introduction_hash_examined,
                    graph_revision_examined, research_md_hash_examined
                ) VALUES (?, 'codex', 'laptop', 'project', '', ?, ?, '', 0, '')
                """,
                (f"session-{index:03d}", resumed, resumed),
            )

    result = store.prune_operational_storage(now=now)

    assert result["outputs"] == 1
    assert result["writing_sessions"] == 5
    assert store.agent_task_patch_output("failed") is None
    with store.connection() as connection:
        count = connection.execute("SELECT COUNT(*) FROM writing_sessions").fetchone()[0]
    assert count == 50


def test_success_deletes_patch_recovery_output_immediately(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    record = AgentTaskRecord(
        operation_id="operation",
        project_id="project",
        kind="refresh",
        status="running",
        request={},
        created_at=now,
        updated_at=now,
        status_message="running",
    )
    store.create_agent_task(record)
    store.record_agent_task_patch_output("operation", '{"kind":"refresh"}')

    store.complete_agent_task("operation", applied_revision=1, result={})

    assert store.agent_task_patch_output("operation") is None


def test_recent_failure_outputs_are_not_dropped_by_an_unrelated_count_cap(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = datetime(2026, 7, 29, tzinfo=UTC)
    for index in range(70):
        operation_id = f"failed-{index:03d}"
        record = _task(operation_id, now - timedelta(seconds=index)).model_copy(
            update={"status": "failed"}
        )
        store.create_agent_task(record)
        store.record_agent_task_patch_output(operation_id, '{"kind":"refresh"}')

    assert store.agent_task_patch_output("failed-069") is not None
    assert store.prune_operational_storage(now=now)["outputs"] == 0
