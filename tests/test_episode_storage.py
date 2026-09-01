from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

import pytest

from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.providers import ProviderUsage
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    EpisodeInvocationCeilingReached,
    EpisodeNotRunning,
    EpisodeRecord,
    EpisodeReportConflict,
    EpisodeReportRecord,
    EpisodeWrapupRecord,
)
from rcp.storage.episodes import _legacy_experiment_lifecycle, compact_episode_receipt


def _authorizer(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    if owner.display_name is None:
        owner = store.rename_space_user(owner.user_id, "Episode owner")
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name=owner.display_name,
    )


def _create_legacy_campaign_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE campaigns (
            campaign_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            root_operation_id TEXT,
            status TEXT NOT NULL,
            starting_instruction TEXT,
            invocation_ceiling INTEGER NOT NULL,
            invocations_used INTEGER NOT NULL DEFAULT 0,
            authorized_space_id TEXT,
            authorized_user_id TEXT,
            authorized_display_name TEXT,
            stop_requested_at TEXT,
            ending TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            ended_at TEXT
        );
        CREATE TABLE campaign_invocations (
            campaign_id TEXT NOT NULL,
            operation_id TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE campaign_reports (
            report_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL,
            operation_id TEXT NOT NULL UNIQUE,
            ending TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            html TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )


def _create_legacy_experiment_episode_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE experiment_episodes (
            episode_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            control_node_id TEXT NOT NULL,
            provider TEXT,
            execution_machine TEXT,
            execution_host TEXT NOT NULL DEFAULT '',
            native_session_id TEXT,
            stage_host TEXT,
            stage_root TEXT,
            chat_id TEXT,
            last_turn_operation_id TEXT,
            last_turn_invocation INTEGER,
            last_graph_result TEXT,
            last_watcher_ids_json TEXT NOT NULL DEFAULT '[]',
            context_baseline_json TEXT NOT NULL DEFAULT '{}',
            session_diagnostic TEXT,
            stop_requested_at TEXT,
            stop_settled_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )


def _episode(
    store: AppStore,
    episode_id: str,
    *,
    project_id: str = "project",
    mode: str = "experiment_loop",
    control_node_id: str | None = None,
    ceiling: int = 1,
) -> EpisodeRecord:
    now = store.now()
    if mode == "experiment_loop" and control_node_id is None:
        control_node_id = "experiment-node"
    return EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode=mode,
        control_node_id=control_node_id,
        graph_target=(
            GraphTargetRef(kind="branch", branch_id=episode_id)
            if mode == "auto_research"
            else GraphTargetRef()
        ),
        graph_base_head=(GraphHeadRef(revision=0) if mode == "auto_research" else None),
        status="queued",
        invocation_ceiling=ceiling,
        authorized_by=_authorizer(store),
        created_at=now,
        updated_at=now,
    )


def _operational_task(
    store: AppStore,
    operation_id: str,
    *,
    episode_id: str,
    project_id: str = "project",
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        episode_id=episode_id,
        kind="node_chat",
        status="queued",
        request={},
        created_at=now,
        updated_at=now,
        status_message="Queued",
    )


def _wrapup(
    store: AppStore,
    episode_id: str,
    concluding_operation_id: str,
    allocation_operation_id: str,
    *,
    ending: str = "completed",
    diagnostic: str | None = None,
) -> tuple[EpisodeWrapupRecord, AgentTaskRecord]:
    now = store.now()
    receipt_json, receipt_sha256 = compact_episode_receipt(
        {
            "ending": ending,
            "episode_id": episode_id,
            "source_operation_id": concluding_operation_id,
        }
    )
    wrapup = EpisodeWrapupRecord(
        episode_id=episode_id,
        ending=ending,
        partial=ending != "completed",
        concluding_operation_id=concluding_operation_id,
        allocation_operation_id=allocation_operation_id,
        provider="codex",
        run_on="local",
        execution_host="",
        native_session_id="native-session",
        stage_host=None,
        stage_root="/tmp/episode-stage",
        skill_id="episode-report",
        skill_version="1",
        output_name="episode-report.html",
        output_path="/tmp/episode-stage/episode-report.html",
        receipt_json=receipt_json,
        receipt_sha256=receipt_sha256,
        state="pending",
        diagnostic=diagnostic,
        created_at=now,
        updated_at=now,
    )
    task = AgentTaskRecord(
        operation_id=allocation_operation_id,
        project_id="project",
        episode_id=episode_id,
        kind="episode_report",
        status="queued",
        request={
            "provider": "codex",
            "run_on": "local",
            "execution_host": "",
        },
        created_at=now,
        updated_at=now,
        status_message="Wrapping up visualization and report",
        parent_operation_id=concluding_operation_id,
        native_session_id="native-session",
        stage_root="/tmp/episode-stage",
        visible=False,
    )
    return wrapup, task


def _start_wrapping(
    store: AppStore,
    episode_id: str,
    *,
    ending: str = "completed",
) -> tuple[EpisodeWrapupRecord, AgentTaskRecord]:
    store.create_episode(_episode(store, episode_id))
    store.allocate_episode_invocation(
        episode_id,
        _operational_task(store, f"{episode_id}-operation", episode_id=episode_id),
    )
    wrapup, task = _wrapup(
        store,
        episode_id,
        f"{episode_id}-operation",
        f"{episode_id}-report-allocation",
        ending=ending,
    )
    store.begin_episode_wrapup(episode_id, wrapup, task)
    return wrapup, task


def _insert_legacy_experiment_wrapup(
    store: AppStore,
    episode_id: str,
    concluding_operation_id: str,
    *,
    migration_owned: bool = True,
) -> None:
    now = store.now()
    receipt = {
        "control_node_id": "experiment-node",
        "ending": "completed",
        "episode_id": episode_id,
    }
    if migration_owned:
        receipt["legacy_source"] = "experiment_episode"
    receipt_json, receipt_sha256 = compact_episode_receipt(receipt)
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO episode_wrapups (
                episode_id, ending, partial, concluding_operation_id,
                receipt_json, receipt_sha256, state, created_at, updated_at,
                finished_at
            ) VALUES (?, 'completed', 0, ?, ?, ?, 'legacy_unavailable', ?, ?, ?)
            """,
            (
                episode_id,
                concluding_operation_id,
                receipt_json,
                receipt_sha256,
                now,
                now,
                now,
            ),
        )


def _start_modern_experiment_episode(
    store: AppStore,
    episode_id: str,
    operation_id: str,
    *,
    complete_task: bool,
) -> None:
    task = _operational_task(store, operation_id, episode_id=episode_id).model_copy(
        update={
            "request": {
                "patch_kind": "experiment_loop",
                "control_episode_id": episode_id,
                "control_node_id": "experiment-node",
                "control_revision": 0,
                "control_invocation": 1,
                "control_invocation_ceiling": 10,
                "control_decision_bundle": [],
                "control_completion_criteria": [],
                "trigger": "experiment_run",
            }
        }
    )
    store.create_episode_with_invocation(
        _episode(store, episode_id, ceiling=10),
        task,
    )
    if complete_task:
        store.mark_agent_task_running(operation_id)
        store.complete_agent_task(operation_id, applied_revision=None, result={})


def test_fresh_episode_parents_enforce_mode_specific_live_scope(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "auto-1", mode="auto_research"))

    with pytest.raises(ValueError, match="live parent"):
        store.create_episode(_episode(store, "auto-2", mode="auto_research"))

    first_experiment = _episode(
        store,
        "experiment-1",
        mode="experiment_loop",
        control_node_id="experiment-node-a",
    )
    store.create_episode(first_experiment)
    with pytest.raises(ValueError, match="live parent"):
        store.create_episode(
            _episode(
                store,
                "experiment-2",
                mode="experiment_loop",
                control_node_id="experiment-node-a",
            )
        )
    store.create_episode(
        _episode(
            store,
            "experiment-3",
            mode="experiment_loop",
            control_node_id="experiment-node-b",
        )
    )

    assert {item.episode_id for item in store.episodes("project")} == {
        "auto-1",
        "experiment-1",
        "experiment-3",
    }


def test_episode_and_first_invocation_are_one_exact_atomic_pair(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode = _episode(store, "episode")
    task = _operational_task(store, "episode-operation", episode_id="episode")

    created, invocation, stored_task = store.create_episode_with_invocation(episode, task)

    assert created.status == "running"
    assert created.root_operation_id == task.operation_id
    assert created.invocations_used == 1
    assert invocation.invocation_number == 1
    assert invocation.operation_id == task.operation_id
    assert stored_task.operation_id == task.operation_id
    assert store.create_episode_with_invocation(episode, task)[0] == created
    with pytest.raises(EpisodeReportConflict, match="committed pair"):
        store.create_episode_with_invocation(
            episode,
            _operational_task(store, "different-operation", episode_id="episode"),
        )


def test_auto_research_cannot_bypass_its_atomic_mode_adapter(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode = _episode(store, "episode", mode="auto_research")
    task = _operational_task(store, "episode-operation", episode_id="episode").model_copy(
        update={"kind": "auto_research"}
    )

    with pytest.raises(ValueError, match="mode adapter"):
        store.create_episode_with_invocation(episode, task)

    store.create_episode(episode)
    with pytest.raises(ValueError, match="mode adapter"):
        store.allocate_episode_invocation("episode", task)


def test_operational_ceiling_is_independent_of_hidden_report_attempts(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "episode"))
    store.allocate_episode_invocation(
        "episode",
        _operational_task(store, "episode-operation", episode_id="episode"),
    )

    meter = store.episode_budget_meter("episode")
    assert meter.model_dump() == {
        "invocation_ceiling": 1,
        "invocations_used": 1,
        "invocations_remaining": 0,
        "observed_input_tokens": 0,
        "observed_generated_tokens": 0,
    }
    with pytest.raises(EpisodeInvocationCeilingReached):
        store.allocate_episode_invocation(
            "episode", _operational_task(store, "extra", episode_id="episode")
        )

    wrapup, hidden_task = _wrapup(
        store,
        "episode",
        "episode-operation",
        "episode-report-allocation",
    )
    store.begin_episode_wrapup("episode", wrapup, hidden_task)

    store.record_agent_usage(
        "episode-operation",
        ProviderUsage(
            provider_profile="codex.turn.v1",
            provider_event_type="turn.completed",
            dedupe_key="operational-usage",
            processed_input_tokens=100,
            generated_tokens=10,
        ),
    )
    store.record_agent_usage(
        hidden_task.operation_id,
        ProviderUsage(
            provider_profile="codex.turn.v1",
            provider_event_type="turn.completed",
            dedupe_key="hidden-report-usage",
            processed_input_tokens=900,
            generated_tokens=90,
        ),
    )

    attempt = store.allocate_episode_report_attempt("episode")
    assert attempt.allocation_operation_id == hidden_task.operation_id
    assert store.episode_budget_meter("episode").model_dump() == {
        "invocation_ceiling": 1,
        "invocations_used": 1,
        "invocations_remaining": 0,
        "observed_input_tokens": 100,
        "observed_generated_tokens": 10,
    }
    assert store.agent_task(hidden_task.operation_id) is not None
    assert hidden_task.operation_id not in {
        task.operation_id for task in store.agent_tasks("project")
    }
    assert hidden_task.operation_id in {
        task.operation_id for task in store.agent_tasks("project", include_hidden=True)
    }
    with pytest.raises(ValueError, match="episode wrap-up allocation"):
        store.create_agent_task(hidden_task.model_copy(update={"operation_id": "generic-retry"}))
    stored_wrapup = store.episode_wrapup("episode")
    assert stored_wrapup.state == "running"
    assert stored_wrapup.receipt_sha256 == wrapup.receipt_sha256


def test_report_continuation_may_be_a_same_invocation_recovery_child(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "episode"))
    original = _operational_task(store, "episode-operation", episode_id="episode")
    store.allocate_episode_invocation("episode", original)
    recovery = original.model_copy(
        update={
            "operation_id": "episode-retry",
            "status": "succeeded",
            "parent_operation_id": original.operation_id,
            "native_session_id": "recovered-session",
            "stage_root": "/tmp/recovered-stage",
        }
    )
    with store.connection() as connection:
        store._insert_agent_task(connection, recovery, continuation_cause="retry")
    wrapup, hidden = _wrapup(
        store,
        "episode",
        recovery.operation_id,
        "episode-report-allocation",
    )

    episode, stored, task = store.begin_episode_wrapup("episode", wrapup, hidden)

    assert episode.wrapup_state == "pending"
    assert stored.concluding_operation_id == recovery.operation_id
    assert task.parent_operation_id == recovery.operation_id
    assert store.episode_invocations("episode")[0].operation_id == original.operation_id


def test_pending_hidden_wrapup_is_listed_for_startup_reconciliation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _start_wrapping(store, "episode")

    assert [item.episode_id for item in store.episodes_awaiting_report()] == ["episode"]

    attempt = store.allocate_episode_report_attempt("episode")
    store.finish_episode_report_error(attempt.attempt_id, "terminal report failure")
    assert store.episodes_awaiting_report() == []


def test_stop_is_the_only_report_skipping_terminal_transition(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "stopped"))
    store.allocate_episode_invocation(
        "stopped", _operational_task(store, "stop-operation", episode_id="stopped")
    )

    stopping = store.request_episode_stop("stopped")
    assert stopping.status == "stopping"
    assert stopping.stop_requested_at is not None
    assert store.request_episode_stop("stopped") == stopping

    stopped = store.mark_episode_stop_skipped("stopped", diagnostic="Stopped by the researcher")

    assert stopped.status == "stopped"
    assert stopped.ending == "stopped"
    assert stopped.wrapup_state == "skipped"
    assert stopped.report_attempts_used == 0
    assert store.episode_report("stopped") is None
    assert store.episode_wrapup("stopped").state == "skipped"
    store.create_episode(_episode(store, "replacement"))


def test_stop_cannot_cancel_report_wrapup(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _start_wrapping(store, "episode")

    with pytest.raises(EpisodeNotRunning, match="before wrap-up"):
        store.request_episode_stop("episode")


def test_stop_rejects_a_conflicting_wrapup_before_mutating_live_episode(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    episode_id = str(uuid.uuid4())
    _start_modern_experiment_episode(
        store,
        episode_id,
        "episode-operation",
        complete_task=False,
    )
    _insert_legacy_experiment_wrapup(
        store,
        episode_id,
        "episode-operation",
        migration_owned=False,
    )

    reopened = AppStore(path)
    assert reopened.episode_wrapup(episode_id) is not None

    with pytest.raises(EpisodeNotRunning, match="entered wrap-up"):
        reopened.request_episode_stop(episode_id)

    episode = reopened.episode(episode_id)
    assert episode is not None
    assert episode.status == "running"
    assert episode.stop_requested_at is None


def test_ending_fence_stops_new_work_before_hidden_report_allocation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "episode", ceiling=2))
    store.allocate_episode_invocation(
        "episode",
        _operational_task(store, "episode-operation", episode_id="episode"),
    )

    fenced = store.fence_episode_ending(
        "episode",
        "exhausted",
        diagnostic="The operational ceiling was reached.",
    )

    assert fenced.status == "wrapping_up"
    assert fenced.ending == "exhausted"
    assert fenced.wrapup_state == "not_started"
    assert store.episode_wrapup("episode") is None
    with pytest.raises(EpisodeNotRunning, match="not admitting operational work"):
        store.allocate_episode_invocation(
            "episode",
            _operational_task(store, "late-operation", episode_id="episode"),
        )
    assert (
        store.fence_episode_ending(
            "episode",
            "exhausted",
            diagnostic="The operational ceiling was reached.",
        )
        == fenced
    )
    with pytest.raises(EpisodeReportConflict, match="immutable"):
        store.fence_episode_ending("episode", "failed", diagnostic="different")

    wrapup, task = _wrapup(
        store,
        "episode",
        "episode-operation",
        "episode-report-allocation",
        ending="exhausted",
        diagnostic="The operational ceiling was reached.",
    )
    episode, _stored_wrapup, _stored_task = store.begin_episode_wrapup(
        "episode",
        wrapup,
        task,
    )
    assert episode.wrapup_state == "pending"


def test_three_report_calls_share_one_hidden_allocation_and_final_error_is_terminal(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _start_wrapping(store, "episode", ending="exhausted")

    attempts = []
    for number in range(1, 4):
        attempt = store.allocate_episode_report_attempt("episode")
        attempts.append(attempt)
        episode, failed = store.record_episode_report_attempt_error(
            attempt.attempt_id,
            f"attempt {number} failed",
        )
        assert failed.status == "failed"
        if number < 3:
            assert episode.status == "wrapping_up"
            assert episode.wrapup_state == "pending"

    assert {item.attempt_number for item in attempts} == {1, 2, 3}
    assert len({item.allocation_operation_id for item in attempts}) == 1
    episode = store.episode("episode")
    assert episode is not None
    assert episode.status == "needs_action"
    assert episode.ending == "exhausted"
    assert episode.wrapup_state == "failed"
    assert episode.wrapup_error == "attempt 3 failed"
    assert store.episode_wrapup("episode").state == "failed"
    allocation = store.agent_task(attempts[0].allocation_operation_id)
    assert allocation.status == "failed"
    assert allocation.can_retry is False


def test_successful_report_is_immutable_and_closes_semantic_ending(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _start_wrapping(store, "episode")
    first = store.allocate_episode_report_attempt("episode")
    store.record_episode_report_attempt_error(first.attempt_id, "transient")
    second = store.allocate_episode_report_attempt("episode")
    html = "<html><body><figure>Result</figure></body></html>"
    report = EpisodeReportRecord(
        report_id="report",
        episode_id="episode",
        attempt_id=second.attempt_id,
        allocation_operation_id=second.allocation_operation_id,
        ending="completed",
        sha256=hashlib.sha256(html.encode()).hexdigest(),
        html=html,
        created_at=store.now(),
    )

    episode, stored = store.finish_episode_report_ready(second.attempt_id, report)

    assert episode.status == "completed"
    assert episode.ending == "completed"
    assert episode.wrapup_state == "ready"
    assert stored == report
    assert store.episode_report_by_id("report") == report
    with pytest.raises(EpisodeReportConflict, match="immutable"):
        store.finish_episode_report_ready(
            second.attempt_id,
            report.model_copy(update={"report_id": "different"}),
        )


def test_wrapup_and_current_attempt_are_restart_idempotent(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    wrapup, task = _start_wrapping(store, "episode")
    attempt = store.allocate_episode_report_attempt("episode")

    restarted = AppStore(path)
    assert restarted.episode_wrapup("episode").state == "running"
    assert restarted.current_episode_report_attempt("episode") == attempt
    episode, same_wrapup, same_task = restarted.begin_episode_wrapup("episode", wrapup, task)
    assert episode.wrapup_state == "running"
    assert same_wrapup.state == "running"
    assert same_wrapup.receipt_sha256 == wrapup.receipt_sha256
    assert same_task.operation_id == task.operation_id
    assert restarted.allocate_episode_report_attempt("episode") == attempt


def test_restart_requeues_only_the_same_hidden_report_allocation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _, allocation = _start_wrapping(store, "episode")
    first = store.allocate_episode_report_attempt("episode")
    store.mark_episode_report_attempt_running(first.attempt_id)
    store.interrupt_active_agent_tasks()

    requeued = store.requeue_interrupted_episode_report_allocation("episode")

    assert requeued.operation_id == allocation.operation_id
    assert requeued.status == "queued"
    assert requeued.visible is False
    assert requeued.can_retry is False
    assert store.episode_report_attempt(first.attempt_id).status == "failed"
    assert store.current_episode_report_attempt("episode") is None
    assert store.requeue_interrupted_episode_report_allocation("episode") == requeued
    second = store.allocate_episode_report_attempt("episode")
    assert second.attempt_number == 2
    assert second.allocation_operation_id == allocation.operation_id
    assert (
        len(
            [
                task
                for task in store.agent_tasks("project", include_hidden=True)
                if task.kind == "episode_report"
            ]
        )
        == 1
    )


def test_restart_requeues_the_same_shutdown_paused_hidden_allocation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _, allocation = _start_wrapping(store, "episode")
    first = store.allocate_episode_report_attempt("episode")
    store.mark_episode_report_attempt_running(first.attempt_id)
    store.mark_agent_task_running(allocation.operation_id)
    store.bind_agent_task_write_scope(
        allocation.operation_id,
        project_id="project",
        stage_host="",
        stage_root="/tmp/episode-stage",
        fingerprint="a" * 64,
    )
    store.record_agent_task_receipt(
        allocation.operation_id,
        "operation_dispatch_attempt",
        {"dispatch_attempt_id": "previous-report-dispatch"},
        tier="diagnostic",
    )
    store.record_agent_task_receipt(
        allocation.operation_id,
        "operation_dispatch_started",
        {"dispatch_attempt_id": "previous-report-dispatch"},
        tier="diagnostic",
    )
    store.pause_agent_task(allocation.operation_id, detail="Paused for shutdown")

    requeued = store.requeue_interrupted_episode_report_allocation("episode")

    assert requeued.operation_id == allocation.operation_id
    assert requeued.status == "queued"
    assert requeued.visible is False
    assert requeued.write_scope_fingerprint is None
    assert store.agent_task_dispatch_was_proven_not_started(allocation.operation_id)
    assert any(
        item.category == "operation_dispatch_reset"
        for item in store.agent_task_receipts(allocation.operation_id)
    )
    assert store.episode_report_attempt(first.attempt_id).status == "failed"
    assert store.episode("episode").report_attempts_used == 1
    second = store.allocate_episode_report_attempt("episode")
    assert second.attempt_number == 2
    assert second.allocation_operation_id == allocation.operation_id


def test_restart_during_third_report_attempt_terminalizes_without_stranding(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _, allocation = _start_wrapping(store, "episode", ending="exhausted")
    for number in (1, 2):
        attempt = store.allocate_episode_report_attempt("episode")
        store.mark_episode_report_attempt_running(attempt.attempt_id)
        store.record_episode_report_attempt_error(attempt.attempt_id, f"failure {number}")
    third = store.allocate_episode_report_attempt("episode")
    store.mark_episode_report_attempt_running(third.attempt_id)
    store.interrupt_active_agent_tasks()

    terminal_task = store.requeue_interrupted_episode_report_allocation("episode")

    episode = store.episode("episode")
    assert episode is not None
    assert episode.status == "needs_action"
    assert episode.ending == "exhausted"
    assert episode.wrapup_state == "failed"
    assert "interrupted" in episode.wrapup_error
    assert store.episode_wrapup("episode").state == "failed"
    assert store.episode_report_attempt(third.attempt_id).status == "failed"
    assert terminal_task.operation_id == allocation.operation_id
    assert terminal_task.status == "failed"
    assert terminal_task.visible is False
    store.create_episode(_episode(store, "replacement"))


def test_unlaunchable_wrapup_is_terminal_without_a_hidden_allocation(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "episode"))
    store.allocate_episode_invocation(
        "episode",
        _operational_task(store, "episode-operation", episode_id="episode"),
    )
    now = store.now()
    receipt_json, receipt_sha256 = compact_episode_receipt(
        {
            "ending": "failed",
            "episode_id": "episode",
            "source_operation_id": "episode-operation",
        }
    )
    wrapup = EpisodeWrapupRecord(
        episode_id="episode",
        ending="failed",
        partial=True,
        concluding_operation_id="episode-operation",
        receipt_json=receipt_json,
        receipt_sha256=receipt_sha256,
        state="failed",
        diagnostic="The exact native session is unavailable.",
        created_at=now,
        updated_at=now,
        finished_at=now,
    )

    episode, stored = store.fail_episode_wrapup_unlaunchable("episode", wrapup)

    assert stored == wrapup
    assert episode.status == "failed"
    assert episode.ending == "failed"
    assert episode.wrapup_state == "failed"
    assert episode.wrapup_error == "The exact native session is unavailable."
    assert episode.report_attempts_used == 0
    assert store.episode_report_attempts("episode") == []
    assert all(
        task.kind != "episode_report" for task in store.agent_tasks("project", include_hidden=True)
    )
    assert store.fail_episode_wrapup_unlaunchable("episode", wrapup)[1] == wrapup
    store.create_episode(_episode(store, "replacement"))


def test_ending_without_a_report_terminalizes_and_leaves_no_wrapup(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "episode"))
    store.allocate_episode_invocation(
        "episode",
        _operational_task(store, "episode-operation", episode_id="episode"),
    )

    episode = store.end_episode_without_report(
        "episode",
        ending="failed",
        diagnostic="The turn failed before it started its agent session.",
    )

    assert episode.status == "failed"
    assert episode.ending == "failed"
    assert episode.ended_at is not None
    # No wrap-up began, so there is no report state and no report error to show.
    assert episode.wrapup_state == "not_started"
    assert episode.wrapup_error is None
    assert episode.report_attempts_used == 0
    assert store.episode_wrapup("episode") is None
    assert store.episode_report_attempts("episode") == []
    assert all(
        task.kind != "episode_report" for task in store.agent_tasks("project", include_hidden=True)
    )
    # A terminal status is not a live episode, so the Experiment can start again.
    store.create_episode(_episode(store, "replacement"))


def test_ending_without_a_report_is_idempotent_and_immutable(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "episode"))
    store.allocate_episode_invocation(
        "episode",
        _operational_task(store, "episode-operation", episode_id="episode"),
    )
    first = store.end_episode_without_report("episode", ending="failed", diagnostic="No session.")

    assert (
        store.end_episode_without_report("episode", ending="failed", diagnostic="No session.")
        == first
    )
    with pytest.raises(EpisodeReportConflict):
        store.end_episode_without_report("episode", ending="failed", diagnostic="Something else.")
    with pytest.raises(ValueError, match="Stop settles through its own skip path"):
        store.end_episode_without_report("episode", ending="stopped")


def test_ending_without_a_report_settles_an_episode_already_fenced_into_wrapup(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    store.create_episode(_episode(store, "episode"))
    store.allocate_episode_invocation(
        "episode",
        _operational_task(store, "episode-operation", episode_id="episode"),
    )
    fenced = store.fence_episode_ending("episode", "failed", diagnostic="No session.")
    assert fenced.status == "wrapping_up"

    episode = store.end_episode_without_report("episode", ending="failed", diagnostic="No session.")

    # `wrapping_up` is a live status; leaving an episode parked there is exactly the
    # deadlock this path exists to prevent.
    assert episode.status == "failed"
    assert episode.wrapup_state == "not_started"
    store.create_episode(_episode(store, "replacement"))


def test_allocated_unlaunchable_wrapup_fails_without_fabricating_an_attempt(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    wrapup, allocation = _start_wrapping(store, "episode", ending="exhausted")

    episode, failed_wrapup, failed_task = store.fail_episode_report_allocation_unlaunchable(
        "episode",
        "The frozen provider session cannot be resumed on its saved stage.",
    )

    assert episode.status == "needs_action"
    assert episode.ending == "exhausted"
    assert episode.wrapup_state == "failed"
    assert episode.report_attempts_used == 0
    assert failed_wrapup.ending == wrapup.ending
    assert failed_wrapup.receipt_sha256 == wrapup.receipt_sha256
    assert failed_wrapup.state == "failed"
    assert failed_task.operation_id == allocation.operation_id
    assert failed_task.status == "failed"
    assert failed_task.visible is False
    assert store.episode_report_attempts("episode") == []
    assert (
        store.fail_episode_report_allocation_unlaunchable(
            "episode",
            "The frozen provider session cannot be resumed on its saved stage.",
        )[0]
        == episode
    )
    with pytest.raises(EpisodeReportConflict, match="different diagnostic"):
        store.fail_episode_report_allocation_unlaunchable("episode", "A different error")


def test_allocated_unlaunchable_wrapup_preserves_prior_real_attempt_count(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    _start_wrapping(store, "episode", ending="exhausted")
    first = store.allocate_episode_report_attempt("episode")
    store.mark_episode_report_attempt_running(first.attempt_id)
    store.record_episode_report_attempt_error(first.attempt_id, "provider failure")

    episode, _, failed_task = store.fail_episode_report_allocation_unlaunchable(
        "episode",
        "The saved stage became unavailable before the next provider call.",
    )

    assert episode.status == "needs_action"
    assert episode.wrapup_state == "failed"
    assert episode.report_attempts_used == 1
    assert [attempt.status for attempt in store.episode_report_attempts("episode")] == ["failed"]
    assert failed_task.status == "failed"


def test_campaign_migration_keeps_latest_report_and_is_idempotent(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    authorizer = _authorizer(store)
    now = store.now()
    old_html = "<html>old</html>"
    latest_html = "<html>latest</html>"
    with store.connection() as connection:
        _create_legacy_campaign_tables(connection)
        connection.execute(
            """
            INSERT INTO campaigns (
                campaign_id, project_id, root_operation_id, status,
                invocation_ceiling, invocations_used, authorized_space_id,
                authorized_user_id, authorized_display_name, ending,
                created_at, updated_at, ended_at
            ) VALUES ('campaign', 'project', 'operational', 'succeeded', 1, 1, ?, ?, ?,
                      'completed', ?, ?, ?)
            """,
            (authorizer.space_id, authorizer.user_id, authorizer.display_name, now, now, now),
        )
        for operation_id, kind, created_at in (
            ("operational", "campaign", now),
            ("old-report-task", "campaign", "2026-01-01T00:00:00+00:00"),
            ("latest-report-task", "campaign", "2026-01-02T00:00:00+00:00"),
        ):
            connection.execute(
                """
                INSERT INTO graph_runs (
                    operation_id, project_id, kind, status, request_json,
                    created_at, updated_at, status_message, native_session_id, stage_root
                ) VALUES (?, 'project', ?, 'succeeded', ?, ?, ?, 'done', 'native', '/stage')
                """,
                (
                    operation_id,
                    kind,
                    json.dumps(
                        {"provider": "codex", "run_on": "local", "execution_host": "localhost"}
                    ),
                    created_at,
                    created_at,
                ),
            )
        connection.execute(
            "INSERT INTO campaign_invocations VALUES ('campaign', 'operational', 'orchestrator', ?)",
            (now,),
        )
        for report_id, operation_id, html, created_at in (
            ("old", "old-report-task", old_html, "2026-01-01T00:00:00+00:00"),
            ("latest", "latest-report-task", latest_html, "2026-01-02T00:00:00+00:00"),
        ):
            connection.execute(
                """
                INSERT INTO campaign_reports (
                    report_id, campaign_id, operation_id, ending, sha256, html, created_at
                ) VALUES (?, 'campaign', ?, 'completed', ?, ?, ?)
                """,
                (
                    report_id,
                    operation_id,
                    hashlib.sha256(html.encode()).hexdigest(),
                    html,
                    created_at,
                ),
            )

    migrated = AppStore(path)
    episode = migrated.episode("campaign")
    report = migrated.episode_report("campaign")
    assert episode is not None and report is not None
    assert episode.mode == "auto_research"
    assert episode.status == "completed"
    assert episode.invocation_ceiling == episode.invocations_used == 1
    assert report.report_id == "latest"
    assert report.html == latest_html
    assert migrated.agent_task("latest-report-task").kind == "episode_report"
    assert migrated.agent_task("old-report-task").kind == "episode_report"
    assert migrated.agent_task("latest-report-task").visible is False
    assert migrated.agent_task("old-report-task").visible is False
    assert AppStore(path).episode_report("campaign") == report
    with migrated.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM episode_reports WHERE episode_id = 'campaign'"
            ).fetchone()[0]
            == 1
        )


def test_campaign_migration_removes_each_legacy_report_reservation(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    authorizer = _authorizer(store)
    initial_at = "2026-01-01T00:00:00+00:00"
    report_at = "2026-01-02T00:00:00+00:00"
    reauthorized_at = "2026-01-03T00:00:00+00:00"
    report_html = "<html>first cycle</html>"
    with store.connection() as connection:
        _create_legacy_campaign_tables(connection)
        connection.execute(
            """
            INSERT INTO campaigns (
                campaign_id, project_id, root_operation_id, status,
                invocation_ceiling, invocations_used, authorized_space_id,
                authorized_user_id, authorized_display_name, ending,
                created_at, updated_at, ended_at
            ) VALUES ('completed-cycle', 'completed-project', 'completed-operation',
                      'succeeded', 2, 2, ?, ?, ?, 'completed', ?, ?, ?)
            """,
            (
                authorizer.space_id,
                authorizer.user_id,
                authorizer.display_name,
                initial_at,
                report_at,
                report_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaigns (
                campaign_id, project_id, root_operation_id, status,
                invocation_ceiling, invocations_used, authorized_space_id,
                authorized_user_id, authorized_display_name,
                created_at, updated_at
            ) VALUES ('reauthorized-cycle', 'reauthorized-project', 'initial-operation',
                      'running', 4, 3, ?, ?, ?, ?, ?)
            """,
            (
                authorizer.space_id,
                authorizer.user_id,
                authorizer.display_name,
                initial_at,
                reauthorized_at,
            ),
        )
        tasks = (
            (
                "completed-operation",
                "completed-project",
                "completed-cycle",
                initial_at,
            ),
            ("completed-report", "completed-project", "completed-cycle", report_at),
            ("initial-operation", "reauthorized-project", "reauthorized-cycle", initial_at),
            ("initial-report", "reauthorized-project", "reauthorized-cycle", report_at),
            (
                "reauthorized-operation",
                "reauthorized-project",
                "reauthorized-cycle",
                reauthorized_at,
            ),
        )
        for operation_id, project_id, campaign_id, created_at in tasks:
            connection.execute(
                """
                INSERT INTO graph_runs (
                    operation_id, project_id, kind, status, request_json, episode_id,
                    created_at, updated_at, status_message
                ) VALUES (?, ?, 'campaign', 'succeeded', '{}', ?, ?, ?, 'done')
                """,
                (operation_id, project_id, campaign_id, created_at, created_at),
            )
        for campaign_id, operation_id, role, created_at in (
            ("completed-cycle", "completed-operation", "orchestrator", initial_at),
            ("completed-cycle", "completed-report", "report", report_at),
            ("reauthorized-cycle", "initial-operation", "orchestrator", initial_at),
            ("reauthorized-cycle", "initial-report", "report", report_at),
            (
                "reauthorized-cycle",
                "reauthorized-operation",
                "orchestrator",
                reauthorized_at,
            ),
        ):
            connection.execute(
                "INSERT INTO campaign_invocations VALUES (?, ?, ?, ?)",
                (campaign_id, operation_id, role, created_at),
            )
        for report_id, campaign_id, operation_id, ending in (
            ("completed-cycle-report", "completed-cycle", "completed-report", "completed"),
            ("initial-cycle-report", "reauthorized-cycle", "initial-report", "exhausted"),
        ):
            connection.execute(
                """
                INSERT INTO campaign_reports (
                    report_id, campaign_id, operation_id, ending, sha256, html, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    campaign_id,
                    operation_id,
                    ending,
                    hashlib.sha256(report_html.encode()).hexdigest(),
                    report_html,
                    report_at,
                ),
            )

    migrated = AppStore(path)
    completed = migrated.episode("completed-cycle")
    reauthorized = migrated.episode("reauthorized-cycle")
    assert completed is not None and reauthorized is not None
    assert completed.invocation_ceiling == completed.invocations_used == 1
    assert reauthorized.invocation_ceiling == reauthorized.invocations_used == 2


def test_campaign_migration_does_not_promote_a_legacy_stop_report(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    authorizer = _authorizer(store)
    now = store.now()
    html = "<html>obsolete stop report</html>"
    with store.connection() as connection:
        _create_legacy_campaign_tables(connection)
        connection.execute(
            """
            INSERT INTO campaigns (
                campaign_id, project_id, root_operation_id, status,
                invocation_ceiling, invocations_used, authorized_space_id,
                authorized_user_id, authorized_display_name, stop_requested_at,
                ending, created_at, updated_at, ended_at
            ) VALUES ('stopped-campaign', 'project', 'operational', 'stopped', 1, 1,
                      ?, ?, ?, ?, 'stopped', ?, ?, ?)
            """,
            (
                authorizer.space_id,
                authorizer.user_id,
                authorizer.display_name,
                now,
                now,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO graph_runs (
                operation_id, project_id, kind, status, request_json,
                created_at, updated_at, status_message, native_session_id, stage_root
            ) VALUES ('stop-report-task', 'project', 'campaign', 'succeeded', '{}',
                      ?, ?, 'done', 'native', '/stage')
            """,
            (now, now),
        )
        connection.execute(
            """
            INSERT INTO campaign_reports (
                report_id, campaign_id, operation_id, ending, sha256, html, created_at
            ) VALUES ('stop-report', 'stopped-campaign', 'stop-report-task', 'stopped', ?, ?, ?)
            """,
            (hashlib.sha256(html.encode()).hexdigest(), html, now),
        )

    migrated = AppStore(path)
    episode = migrated.episode("stopped-campaign")
    assert episode is not None
    assert episode.status == "stopped"
    assert episode.ending == "stopped"
    assert episode.wrapup_state == "skipped"
    assert migrated.episode_wrapup("stopped-campaign").state == "skipped"
    assert migrated.episode_report("stopped-campaign") is None
    assert migrated.agent_task("stop-report-task").visible is False


def test_live_legacy_campaign_without_authorizer_is_terminally_unavailable(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE campaigns (
                campaign_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                root_operation_id TEXT,
                status TEXT NOT NULL,
                starting_instruction TEXT,
                invocation_ceiling INTEGER NOT NULL,
                invocations_used INTEGER NOT NULL DEFAULT 0,
                stop_requested_at TEXT,
                ending TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ended_at TEXT
            );
            INSERT INTO campaigns (
                campaign_id, project_id, status, invocation_ceiling, invocations_used,
                created_at, updated_at
            ) VALUES (
                'unauthorized', 'project', 'running', 1, 0,
                '2026-08-14T00:00:00+00:00', '2026-08-14T00:00:00+00:00'
            );
            """
        )

    migrated = AppStore(path)
    episode = migrated.episode("unauthorized")

    assert episode is not None
    assert episode.status == "failed"
    assert episode.ending == "failed"
    assert episode.authorized_by is None
    assert episode.wrapup_state == "legacy_unavailable"
    assert "authorization snapshot" in episode.ending_diagnostic
    assert migrated.episode_wrapup("unauthorized").state == "legacy_unavailable"


def test_experiment_migration_does_not_reclassify_a_modern_live_episode(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    episode_id = str(uuid.uuid4())
    _start_modern_experiment_episode(
        store,
        episode_id,
        "modern-operation",
        complete_task=True,
    )
    before = store.episode(episode_id)
    assert before is not None
    assert before.status == "running"
    assert store.episode_wrapup(episode_id) is None

    reopened = AppStore(path)

    assert reopened.episode(episode_id) == before
    assert reopened.episode_wrapup(episode_id) is None
    assert [invocation.operation_id for invocation in reopened.episode_invocations(episode_id)] == [
        "modern-operation"
    ]


def test_experiment_migration_removes_its_impossible_modern_wrapup(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    episode_id = str(uuid.uuid4())
    _start_modern_experiment_episode(
        store,
        episode_id,
        "modern-operation",
        complete_task=True,
    )
    stopping = store.request_episode_stop(episode_id)
    _insert_legacy_experiment_wrapup(
        store,
        episode_id,
        "modern-operation",
    )
    assert stopping.status == "stopping"
    assert stopping.ending is None
    assert stopping.wrapup_state == "not_started"

    reopened = AppStore(path)

    repaired = reopened.episode(episode_id)
    assert repaired == stopping
    assert reopened.episode_wrapup(episode_id) is None
    settled = reopened.mark_episode_stop_skipped(
        episode_id,
        diagnostic="Stopped by the researcher",
    )
    assert settled.status == "stopped"
    assert settled.ending == "stopped"
    assert settled.wrapup_state == "skipped"


def test_experiment_migration_discovers_state_rows_and_missing_first_turn_rows(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    authorizer = _authorizer(store)
    now = store.now()
    failed_episode_id = str(uuid.uuid4())
    completed_episode_id = str(uuid.uuid4())
    with store.connection() as connection:
        _create_legacy_experiment_episode_table(connection)
        for episode_id, operation_id, status, control_node_id, ceiling in (
            (failed_episode_id, "failed-first", "failed", "control-a", 1),
            (completed_episode_id, "completed-first", "succeeded", "control-b", 3),
        ):
            request = {
                "patch_kind": "experiment_loop",
                "control_episode_id": episode_id,
                "control_node_id": control_node_id,
                "control_invocation": 1,
                "control_invocation_ceiling": ceiling,
            }
            connection.execute(
                """
                INSERT INTO graph_runs (
                    operation_id, project_id, kind, status, request_json,
                    created_at, updated_at, status_message,
                    authorized_space_id, authorized_user_id, authorized_display_name
                ) VALUES (?, 'project', 'project_chat', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    status,
                    json.dumps(request),
                    now,
                    now,
                    status,
                    authorizer.space_id,
                    authorizer.user_id,
                    authorizer.display_name,
                ),
            )
        connection.execute(
            """
            INSERT INTO experiment_episodes (
                episode_id, project_id, control_node_id, created_at, updated_at
            ) VALUES (?, 'project', 'control-b', ?, ?)
            """,
            (completed_episode_id, now, now),
        )

    migrated = AppStore(path)
    failed = migrated.episode(failed_episode_id)
    completed = migrated.episode(completed_episode_id)
    assert failed is not None and completed is not None
    assert failed.mode == "experiment_loop"
    assert failed.control_node_id == "control-a"
    assert failed.status == "running"
    assert failed.ending is None
    assert failed.invocations_used == 1
    assert failed.invocation_ceiling == 1
    assert failed.authorized_by == authorizer
    assert failed.wrapup_state == "not_started"
    assert migrated.episode_wrapup(failed_episode_id) is None
    assert completed.status == "completed"
    assert completed.ending == "completed"
    assert completed.wrapup_state == "legacy_unavailable"
    assert migrated.episode_wrapup(completed_episode_id).state == "legacy_unavailable"


def test_experiment_exit_migration_classifies_only_retained_proof(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    authorizer = _authorizer(store)
    now = store.now()
    proven_id = str(uuid.uuid4())
    unknown_id = str(uuid.uuid4())
    with store.connection() as connection:
        _create_legacy_experiment_episode_table(connection)
        for episode_id, operation_id, control_node_id in (
            (proven_id, "proven-exit", "control-proven"),
            (unknown_id, "unknown-exit", "control-unknown"),
        ):
            request = {
                "patch_kind": "experiment_loop",
                "control_episode_id": episode_id,
                "control_node_id": control_node_id,
                "control_invocation": 1,
                "control_invocation_ceiling": 3,
            }
            connection.execute(
                """
                INSERT INTO graph_runs (
                    operation_id, project_id, kind, status, request_json,
                    created_at, updated_at, status_message,
                    authorized_space_id, authorized_user_id, authorized_display_name
                ) VALUES (?, 'project', 'project_chat', 'succeeded', ?, ?, ?, 'done', ?, ?, ?)
                """,
                (
                    operation_id,
                    json.dumps(request),
                    now,
                    now,
                    authorizer.space_id,
                    authorizer.user_id,
                    authorizer.display_name,
                ),
            )
            connection.execute(
                """
                INSERT INTO experiment_episodes (
                    episode_id, project_id, control_node_id, created_at, updated_at
                ) VALUES (?, 'project', ?, ?, ?)
                """,
                (episode_id, control_node_id, now, now),
            )
            connection.execute(
                """
                INSERT INTO graph_run_receipts (
                    operation_id, created_at, tier, category, payload_json
                ) VALUES (?, ?, 'summary', 'experiment_loop_exit', '{}')
                """,
                (operation_id, now),
            )
        connection.execute(
            """
            INSERT INTO graph_run_outputs (operation_id, created_at, patch_json)
            VALUES ('proven-exit', ?, ?)
            """,
            (
                now,
                json.dumps(
                    {
                        "ops": [
                            {
                                "op": "update_nodes",
                                "nodes": [
                                    {
                                        "id": "control-proven",
                                        "changes": {"status": "completed"},
                                    }
                                ],
                            }
                        ]
                    }
                ),
            ),
        )

    migrated = AppStore(path)
    proven = migrated.episode(proven_id)
    unknown = migrated.episode(unknown_id)
    assert proven is not None and unknown is not None
    assert proven.status == "completed"
    assert proven.ending == "completed"
    assert unknown.status == "needs_action"
    assert unknown.ending is None
    assert unknown.wrapup_state == "legacy_unavailable"
    assert "no retained Patch" in unknown.ending_diagnostic
    assert migrated.episode_wrapup(unknown_id).ending is None


_LEGACY_WRAPUPS_NOT_NULL_DDL = """
CREATE TABLE episode_wrapups (
    episode_id TEXT PRIMARY KEY,
    ending TEXT NOT NULL,
    partial INTEGER NOT NULL,
    concluding_operation_id TEXT,
    allocation_operation_id TEXT UNIQUE,
    provider TEXT,
    run_on TEXT,
    execution_host TEXT,
    native_session_id TEXT,
    stage_host TEXT,
    stage_root TEXT,
    skill_id TEXT,
    skill_version TEXT,
    output_name TEXT,
    output_path TEXT,
    receipt_json TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL,
    state TEXT NOT NULL,
    diagnostic TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
)
"""


def _downgrade_wrapups_to_not_null(path, *, episode_id: str) -> None:
    """Rebuild `episode_wrapups` the way a pre-relaxation store still has it.

    Every test builds a fresh SQLite file and therefore gets the *current*
    schema, which is exactly why the drift this reproduces reached a real
    machine unseen.
    """

    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE episode_wrapups")
    connection.execute(_LEGACY_WRAPUPS_NOT_NULL_DDL)
    connection.execute(
        """
        INSERT INTO episode_wrapups (
            episode_id, ending, partial, receipt_json, receipt_sha256, state,
            created_at, updated_at
        ) VALUES (?, 'completed', 0, '{}', 'digest', 'legacy_unavailable', 'then', 'then')
        """,
        (episode_id,),
    )
    connection.commit()
    connection.close()


def _wrapup_ending_is_not_null(path) -> bool:
    connection = sqlite3.connect(path)
    try:
        return any(
            row[1] == "ending" and row[3]
            for row in connection.execute("PRAGMA table_info(episode_wrapups)")
        )
    finally:
        connection.close()


def test_a_store_predating_the_nullable_ending_is_migrated_on_open(tmp_path) -> None:
    """A legacy Experiment exit can have no ending, and the column must allow it.

    `CREATE TABLE IF NOT EXISTS` never alters a table that already exists, so
    relaxing `ending` in the create path left every existing database refusing
    the row. Opening the real store crashed with
    `NOT NULL constraint failed: episode_wrapups.ending`.
    """

    path = tmp_path / "rcp.sqlite3"
    AppStore(path)
    _downgrade_wrapups_to_not_null(path, episode_id="legacy-episode")
    assert _wrapup_ending_is_not_null(path)

    reopened = AppStore(path)

    assert not _wrapup_ending_is_not_null(path)
    with reopened.connection() as connection:
        retained = connection.execute(
            "SELECT episode_id, ending, state FROM episode_wrapups"
        ).fetchall()
        assert [tuple(row) for row in retained] == [
            ("legacy-episode", "completed", "legacy_unavailable")
        ]
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'episode_wrapups_rebuilt'"
            ).fetchone()[0]
            == 0
        )


def test_the_migrated_column_accepts_a_wrapup_that_has_no_ending(tmp_path) -> None:
    path = tmp_path / "rcp.sqlite3"
    AppStore(path)
    _downgrade_wrapups_to_not_null(path, episode_id="legacy-episode")
    store = AppStore(path)

    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO episode_wrapups (
                episode_id, ending, partial, receipt_json, receipt_sha256, state,
                created_at, updated_at
            ) VALUES ('unclassifiable', NULL, 1, '{}', 'digest', 'legacy_unavailable', 'now', 'now')
            """
        )
        stored = connection.execute(
            "SELECT ending FROM episode_wrapups WHERE episode_id = 'unclassifiable'"
        ).fetchone()
    assert stored[0] is None


def test_an_existing_row_does_not_excuse_the_missing_migration(tmp_path) -> None:
    """`ON CONFLICT DO NOTHING` resolves the named uniqueness conflict only.

    This is why the crash happened on a store whose wrapup rows were all already
    present: the NOT NULL check fires before the conflict is resolved, so the
    insert raises rather than becoming a no-op.
    """

    path = tmp_path / "rcp.sqlite3"
    AppStore(path)
    _downgrade_wrapups_to_not_null(path, episode_id="legacy-episode")

    connection = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError, match="episode_wrapups.ending"):
        connection.execute(
            """
            INSERT INTO episode_wrapups (
                episode_id, ending, partial, receipt_json, receipt_sha256, state,
                created_at, updated_at
            ) VALUES ('legacy-episode', NULL, 1, '{}', 'digest', 'legacy_unavailable', 'n', 'n')
            ON CONFLICT(episode_id) DO NOTHING
            """
        )
    connection.close()


def test_a_migrated_store_matches_a_fresh_one(tmp_path) -> None:
    """Guards the drift itself: a relaxed create path needs a migration beside it."""

    legacy_path = tmp_path / "legacy.sqlite3"
    AppStore(legacy_path)
    _downgrade_wrapups_to_not_null(legacy_path, episode_id="legacy-episode")
    AppStore(legacy_path)
    fresh_path = tmp_path / "fresh.sqlite3"
    AppStore(fresh_path)

    def columns(path):
        connection = sqlite3.connect(path)
        try:
            return [
                (row[1], row[2], row[3], row[5])
                for row in connection.execute("PRAGMA table_info(episode_wrapups)")
            ]
        finally:
            connection.close()

    assert columns(legacy_path) == columns(fresh_path)


def test_an_unclassifiable_legacy_exit_deliberately_has_no_ending() -> None:
    """The shape behind the NULL: do not invent a meaning the data cannot prove.

    A pre-migration Experiment that exited, but whose exit is neither a proven
    completion nor a proven human pause, gets `ending=None` and says so in its
    diagnostic. Recording it as `completed` or `failed` would write a claim about
    somebody's research that nothing supports, so the column carries the absence
    instead.
    """

    status, ending, wrapup_state, diagnostic, _ended_at = _legacy_experiment_lifecycle(
        None,
        used=1,
        ceiling=4,
        recoverable_task=False,
        watcher_active=False,
        exited=True,
        exit_ending=None,
        exit_diagnostic=None,
    )

    assert ending is None
    assert status == "needs_action"
    # Not `not_started`, so this is exactly the combination that writes a wrapup.
    assert wrapup_state == "legacy_unavailable"
    assert diagnostic == (
        "This pre-migration Experiment exit cannot be classified from retained data."
    )


@pytest.mark.parametrize(
    ("exit_ending", "expected"),
    [("completed", "completed"), ("human_pause", "human_pause")],
)
def test_a_provable_legacy_exit_keeps_its_ending(exit_ending: str, expected: str) -> None:
    _status, ending, wrapup_state, _diagnostic, _ended_at = _legacy_experiment_lifecycle(
        None,
        used=1,
        ceiling=4,
        recoverable_task=False,
        watcher_active=False,
        exited=True,
        exit_ending=exit_ending,
        exit_diagnostic=None,
    )

    assert ending == expected
    assert wrapup_state == "legacy_unavailable"
