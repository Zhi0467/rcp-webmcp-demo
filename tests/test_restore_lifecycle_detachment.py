from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rcp.api import create_app
from rcp.background import BackgroundAgentTasks
from rcp.config import Manifest
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchMessageRecord,
    AutoResearchStateRecord,
    EpisodeRecord,
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    WatcherContinuation,
    WatcherRecord,
)

RESTORE_DIAGNOSTIC = "This work was stopped because RCP restored an older snapshot."
RESTORE_CONFIRMER = "server operator Alice"
RESTORE_RECORDED_DIAGNOSTIC = f"{RESTORE_DIAGNOSTIC} Restore confirmed by {RESTORE_CONFIRMER}."
PRESERVED_WATCHER_CHECK_AT = "2026-08-29T12:00:00+00:00"


def _authorizer(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    if owner.display_name is None:
        owner = store.rename_space_user(owner.user_id, "Restore owner")
    assert owner.display_name is not None
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )


def _auto_parent(
    store: AppStore,
    *,
    episode_id: str,
    project_id: str,
) -> tuple[EpisodeRecord, AgentTaskRecord]:
    now = store.now()
    graph_target = GraphTargetRef(kind="branch", branch_id=episode_id)
    authorizer = _authorizer(store)
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode="auto_research",
        graph_target=graph_target,
        graph_base_head=GraphHeadRef(revision=0),
        status="queued",
        invocation_ceiling=4,
        authorized_by=authorizer,
        created_at=now,
        updated_at=now,
    )
    operation_id = f"{episode_id}-root"
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        episode_id=episode_id,
        graph_target=graph_target,
        kind="auto_research",
        status="queued",
        request={
            "episode_id": episode_id,
            "role": "orchestrator",
            "actor_operation_id": operation_id,
            "run_truth_scope": ["repo"],
        },
        created_at=now,
        updated_at=now,
        status_message="Queued",
        authorized_by=authorizer,
        dispatch_authority=AgentDispatchAuthority(
            profile="orchestrator",
            task_contract="orchestrate",
            scope=AgentDispatchScope(
                run_truth_scope=["repo"],
                episode_id=episode_id,
                patch_kind="work",
            ),
        ),
    )
    return store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode_id,
            starting_instruction="Continue the bounded plan.",
            created_at=now,
            updated_at=now,
        ),
        task,
    )


def _watcher_continuation() -> WatcherContinuation:
    return WatcherContinuation(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="server",
        run_truth_scope=["repo"],
    )


def _seed_restorable_lifecycle(
    store: AppStore,
    *,
    project_id: str = "project",
) -> tuple[str, str]:
    live_episode_id = str(uuid.uuid4())
    completed_episode_id = str(uuid.uuid4())
    completed_episode, completed_root = _auto_parent(
        store,
        episode_id=completed_episode_id,
        project_id=project_id,
    )
    store.mark_agent_task_running(completed_root.operation_id)
    store.complete_agent_task(
        completed_root.operation_id,
        applied_revision=None,
        result={"messages": ["Preserved answer."]},
    )
    store.record_agent_task_receipt(
        completed_root.operation_id,
        "preserved_result",
        {"kept": True},
    )
    now = store.now()
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE episodes
            SET status = 'completed', ending = 'completed', ending_diagnostic = 'Finished.',
                wrapup_state = 'ready', updated_at = ?, ended_at = ?
            WHERE episode_id = ?
            """,
            (now, now, completed_episode.episode_id),
        )
    live_episode, live_root = _auto_parent(
        store,
        episode_id=live_episode_id,
        project_id=project_id,
    )
    now = store.now()
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO auto_research_recoveries (
                recovery_id, episode_id, operation_id, failure_kind, retry_mode,
                attempts, max_attempts, status, next_attempt_at, diagnostic,
                admitted_operation_id, created_at, updated_at
            ) VALUES ('pending-recovery', ?, ?, 'transient', 'exact', 0, 3, 'pending',
                      ?, 'Retry later.', NULL, ?, ?)
            """,
            (live_episode.episode_id, live_root.operation_id, now, now, now),
        )
        for child_id, state, node_id in (
            ("pending-child", "pending", "exp/pending"),
            ("running-child", "running", "exp/running"),
            ("terminal-child", "terminal", "exp/terminal"),
        ):
            connection.execute(
                """
                INSERT INTO auto_research_child_experiments (
                    child_episode_id, auto_research_episode_id, project_id, control_node_id,
                    state, request_json, parent_operation_id, terminal_diagnostic,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, '{}', ?, ?, ?, ?)
                """,
                (
                    child_id,
                    live_episode.episode_id,
                    project_id,
                    node_id,
                    state,
                    live_root.operation_id,
                    "Already finished." if state == "terminal" else None,
                    now,
                    now,
                ),
            )
        for admission_id, state, child_id in (
            ("accepted-admission", "accepted", "future-worker"),
            ("reflected-admission", "reflected", "existing-worker"),
        ):
            connection.execute(
                """
                INSERT INTO auto_research_child_admissions (
                    admission_id, episode_id, project_id, child_kind, child_id,
                    state, created_at, updated_at
                ) VALUES (?, ?, ?, 'work', ?, ?, ?, ?)
                """,
                (
                    admission_id,
                    live_episode.episode_id,
                    project_id,
                    child_id,
                    state,
                    now,
                    now,
                ),
            )
        connection.execute(
            """
            INSERT INTO auto_research_lifecycle_notices (
                notice_id, episode_id, source_kind, source_id, source_event,
                source_attempt, state, payload_json, created_at
            ) VALUES ('pending-notice', ?, 'worker', 'worker-one', 'completed',
                      1, 'pending', '{"result":"kept"}', ?)
            """,
            (live_episode.episode_id, now),
        )
        connection.execute(
            """
            INSERT INTO auto_research_lifecycle_notices (
                notice_id, episode_id, source_kind, source_id, source_event,
                source_attempt, state, payload_json, created_at,
                acknowledged_at, acknowledged_by
            ) VALUES ('acknowledged-notice', ?, 'worker', 'worker-two', 'completed',
                      1, 'acknowledged', '{"result":"kept"}', ?, ?, 'orchestrator')
            """,
            (live_episode.episode_id, now, now),
        )

    store.record_auto_research_message(
        AutoResearchMessageRecord(
            message_id="preserved-message",
            episode_id=live_episode.episode_id,
            sender_role="human",
            authorized_by=_authorizer(store),
            recipient_task_id=live_root.operation_id,
            body="Preserve this research direction.",
            created_at=now,
        )
    )

    common = {
        "project_id": project_id,
        "origin_operation_id": live_root.operation_id,
        "origin_task_kind": "auto_research",
        "chat_id": "auto-research-chat",
        "episode_id": live_episode.episode_id,
        "graph_target": live_episode.graph_target,
        "continuation": _watcher_continuation(),
        "created_at": now,
    }
    store.create_watchers(
        [
            WatcherRecord(
                watcher_id="active-external",
                check_command="true",
                log_path="/tmp/active-external.log",
                cwd="/tmp",
                last_checked_at=PRESERVED_WATCHER_CHECK_AT,
                last_exit_code=0,
                **common,
            ),
            GraphWatcherRecord(
                watcher_id="completed-graph",
                condition=NodeStatusGraphCondition(node_id="exp/pending", status_in=["done"]),
                armed_revision=0,
                status="completed",
                completed_at=now,
                **common,
            ),
            WatcherRecord(
                watcher_id="delivered-external",
                check_command="true",
                log_path="/tmp/delivered-external.log",
                cwd="/tmp",
                status="completed",
                completed_at=now,
                notified=True,
                notification_operation_id=live_root.operation_id,
                **common,
            ),
            WatcherRecord(
                watcher_id="stopped-external",
                check_command="true",
                log_path="/tmp/stopped-external.log",
                cwd="/tmp",
                status="stopped",
                notified=True,
                stopped_by="human",
                stop_reason="Stopped earlier.",
                stopped_at=now,
                **common,
            ),
        ]
    )
    return live_episode_id, completed_episode_id


def _dump(store: AppStore) -> str:
    with store.connection() as connection:
        return "\n".join(connection.iterdump())


def _operational_rows(store: AppStore) -> dict[str, tuple[tuple[object, ...], ...]]:
    tables = (
        "episodes",
        "episode_wrapups",
        "episode_report_attempts",
        "graph_runs",
        "graph_run_events",
        "graph_run_receipts",
        "graph_run_outputs",
        "watchers",
        "auto_research_recoveries",
        "auto_research_messages",
        "auto_research_child_experiments",
        "auto_research_child_admissions",
        "auto_research_lifecycle_notices",
    )
    with store.connection() as connection:
        return {
            table: tuple(
                tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
            )
            for table in tables
        }


def test_restore_detachment_is_atomic_idempotent_and_leaves_no_startup_work(
    tmp_path: Path,
    manifest: Manifest,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    app = create_app(str(manifest.path), data_dir=data_dir, acceptance_agent=True)
    store = app.state.background_tasks.store
    projects = store.projects()
    assert len(projects) == 1
    project_id = projects[0].project_id
    live_episode_id, completed_episode_id = _seed_restorable_lifecycle(
        store,
        project_id=project_id,
    )
    completed_before = store.episode(completed_episode_id)
    detached_at = store.now()

    store.detach_restored_lifecycle(
        diagnostic=RESTORE_DIAGNOSTIC,
        confirmed_by=RESTORE_CONFIRMER,
        detached_at=detached_at,
    )

    live = store.episode(live_episode_id)
    assert live is not None
    assert live.status == "stopped"
    assert live.ending == "stopped"
    assert live.ending_diagnostic == RESTORE_RECORDED_DIAGNOSTIC
    assert live.wrapup_state == "skipped"
    assert store.episode(completed_episode_id) == completed_before
    tasks = store.agent_tasks(project_id, include_hidden=True)
    assert tasks and all(task.history_only for task in tasks)
    assert store.agent_task(f"{live_episode_id}-root").status == "interrupted"
    completed_task = store.agent_task(f"{completed_episode_id}-root")
    assert completed_task is not None
    assert completed_task.result == {"messages": ["Preserved answer."]}
    assert [
        receipt.category for receipt in store.agent_task_receipts(completed_task.operation_id)
    ].count("preserved_result") == 1
    messages = store.auto_research_messages(live_episode_id)
    assert len(messages) == 1
    assert messages[0].body == "Preserve this research direction."
    assert messages[0].authorized_by == _authorizer(store)

    recovery = store.auto_research_recovery("pending-recovery")
    assert recovery is not None
    assert recovery.status == "blocked"
    assert recovery.next_attempt_at is None
    assert recovery.diagnostic == RESTORE_RECORDED_DIAGNOSTIC
    with store.connection() as connection:
        child_rows = {
            row["child_episode_id"]: (row["state"], row["terminal_diagnostic"])
            for row in connection.execute(
                "SELECT * FROM auto_research_child_experiments ORDER BY child_episode_id"
            )
        }
        assert child_rows == {
            "pending-child": ("cancelled", RESTORE_RECORDED_DIAGNOSTIC),
            "running-child": ("cancelled", RESTORE_RECORDED_DIAGNOSTIC),
            "terminal-child": ("terminal", "Already finished."),
        }
        admission_rows = dict(
            connection.execute(
                "SELECT admission_id, state FROM auto_research_child_admissions"
            ).fetchall()
        )
        assert admission_rows == {
            "accepted-admission": "cancelled",
            "reflected-admission": "reflected",
        }
        notice_rows = {
            row["notice_id"]: (row["state"], row["acknowledged_at"], row["acknowledged_by"])
            for row in connection.execute(
                "SELECT * FROM auto_research_lifecycle_notices ORDER BY notice_id"
            )
        }
        assert notice_rows["pending-notice"] == (
            "acknowledged",
            detached_at,
            RESTORE_CONFIRMER,
        )
        assert notice_rows["acknowledged-notice"][2] == "orchestrator"

    active = store.watcher("active-external")
    completed = store.watcher("completed-graph")
    delivered = store.watcher("delivered-external")
    stopped = store.watcher("stopped-external")
    assert active is not None and active.status == "stopped" and active.notified
    assert active.last_checked_at == PRESERVED_WATCHER_CHECK_AT
    assert active.last_exit_code == 0
    assert completed is not None and completed.status == "stopped" and completed.notified
    assert active.stopped_by == completed.stopped_by == "human"
    assert RESTORE_DIAGNOSTIC in (active.stop_reason or "")
    assert RESTORE_CONFIRMER in (active.stop_reason or "")
    assert delivered is not None and delivered.status == "completed" and delivered.notified
    assert stopped is not None and stopped.stop_reason == "Stopped earlier."

    assert store.due_auto_research_recoveries(as_of="9999-01-01T00:00:00+00:00") == []
    assert store.pending_auto_research_lifecycle_episode_ids() == []
    assert store.graph_watcher_project_ids() == []
    assert store.pollable_watchers(as_of="9999-01-01T00:00:00+00:00") == []
    assert store.completed_watcher_groups() == []
    assert store.episodes_awaiting_report() == []

    async def unused_stream(*_args):
        if False:
            yield ""

    recovery_plan = BackgroundAgentTasks(store, unused_stream).plan_startup_recovery()
    assert recovery_plan.as_dict() == {
        "active_operation_ids": (),
        "stopping_experiment_operation_ids": (),
        "report_episode_ids": (),
        "auto_research_recovery_operation_ids": (),
        "active_watcher_ids": (),
    }

    first_dump = _dump(store)
    store.detach_restored_lifecycle(
        diagnostic=RESTORE_DIAGNOSTIC,
        confirmed_by=RESTORE_CONFIRMER,
        detached_at=detached_at,
    )
    assert _dump(store) == first_dump

    before_startup = _operational_rows(store)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert app.state.startup_recovery_plan is None
        assert app.state.startup_effect_runtime_started
    assert _operational_rows(store) == before_startup


def test_restore_detachment_rolls_back_all_owners_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    live_episode_id, _completed_episode_id = _seed_restorable_lifecycle(store)
    root_id = f"{live_episode_id}-root"
    before = _dump(store)

    def fail_after_earlier_owners(*_args, **_kwargs) -> None:
        raise RuntimeError("injected owner failure")

    monkeypatch.setattr(store, "detach_auto_research_for_restore", fail_after_earlier_owners)
    with pytest.raises(RuntimeError, match="injected owner failure"):
        store.detach_restored_lifecycle(
            diagnostic=RESTORE_DIAGNOSTIC,
            confirmed_by=RESTORE_CONFIRMER,
        )

    assert _dump(store) == before
    root = store.agent_task(root_id)
    assert root is not None and root.status == "queued" and not root.history_only


def test_restore_owner_helpers_require_the_composing_transaction(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    now = store.now()
    with store.connection() as connection:
        with pytest.raises(ValueError, match="active transaction"):
            store.detach_auto_research_for_restore(
                connection,
                diagnostic=RESTORE_DIAGNOSTIC,
                now=now,
            )
        with pytest.raises(ValueError, match="requires a transaction"):
            store.detach_auto_research_children_for_restore(
                connection,
                diagnostic=RESTORE_DIAGNOSTIC,
                confirmed_by=RESTORE_CONFIRMER,
                now=now,
            )
        with pytest.raises(ValueError, match="active transaction"):
            store.detach_watchers_for_restore(
                connection,
                diagnostic=RESTORE_DIAGNOSTIC,
                confirmed_by=RESTORE_CONFIRMER,
                now=now,
            )
