from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from rcp.core.models import AuthorizedHuman
from rcp.limits import AGENT_TASK_RECEIPT_MAX_BYTES
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    EpisodeWrapupRecord,
    WatcherContinuation,
    WatcherRecord,
)
from rcp.storage.episodes import compact_episode_receipt
from rcp.watchers import WatcherBinding


def _identity(store: AppStore) -> AuthorizedHuman:
    owner = store.local_owner
    assert owner is not None
    return AuthorizedHuman(
        space_id=store.space_id,
        user_id=owner.user_id,
        display_name="Local owner",
    )


def _request(
    episode_id: str,
    *,
    control_node_id: str = "exp-one",
    invocation: int = 1,
    ceiling: int = 2,
    trigger: str = "experiment_run",
    watcher_ids: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    return {
        "provider": "codex",
        "model": "gpt-5",
        "reasoning": "medium",
        "run_on": "laptop",
        "run_truth_scope": ["repo-a"],
        "chat_id": "episode-chat",
        "node_id": control_node_id,
        "message": "Continue the bounded experiment.",
        "mode": "work",
        "trigger": trigger,
        "patch_kind": "experiment_loop",
        "control_node_id": control_node_id,
        "control_revision": 7,
        "control_episode_id": episode_id,
        "control_invocation": invocation,
        "control_invocation_ceiling": ceiling,
        "control_decision_bundle": [],
        "control_completion_criteria": ["The evaluation has finished."],
        "watcher_ids": list(watcher_ids or []),
        "session_id": session_id,
        "workflow_ids": [],
        "skill_ids": [],
        "invoked_workflow_ids": [],
        "invoked_skill_ids": [],
        "resolved_skill_packages": [],
    }


def _task(
    store: AppStore,
    operation_id: str,
    episode_id: str,
    *,
    invocation: int = 1,
    ceiling: int = 2,
    trigger: str = "experiment_run",
    watcher_ids: list[str] | None = None,
    session_id: str | None = None,
    parent_operation_id: str | None = None,
    attempt: int = 1,
    stage_root: str | None = None,
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id="project",
        episode_id=episode_id,
        kind="node_chat",
        status="queued",
        request=_request(
            episode_id,
            invocation=invocation,
            ceiling=ceiling,
            trigger=trigger,
            watcher_ids=watcher_ids,
            session_id=session_id,
        ),
        created_at=now,
        updated_at=now,
        status_message="Queued.",
        attempt=attempt,
        parent_operation_id=parent_operation_id,
        native_session_id=session_id,
        stage_root=stage_root,
        phase="queued",
        last_activity_at=now,
        authorized_by=_identity(store),
    )


def _continuation(episode_id: str) -> WatcherContinuation:
    return WatcherContinuation.model_validate(
        {
            key: value
            for key, value in _request(episode_id).items()
            if key in WatcherContinuation.model_fields
        }
    )


def _admit_root(
    store: AppStore,
    *,
    episode_id: str | None = None,
    operation_id: str = "loop-root",
    ceiling: int = 2,
) -> tuple[str, AgentTaskRecord]:
    identity = episode_id or str(uuid.uuid4())
    task = _task(store, operation_id, identity, ceiling=ceiling)
    return identity, store.create_experiment_episode_with_invocation(task)


def _bind(
    store: AppStore,
    episode_id: str,
    operation_id: str,
    *,
    invocation: int,
    ending_signal: dict[str, object] | None = None,
) -> None:
    store.commit_experiment_episode_turn(
        episode_id=episode_id,
        project_id="project",
        control_node_id="exp-one",
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id="native-session",
        stage_host=None,
        stage_root="/tmp/exact-experiment-stage",
        chat_id="episode-chat",
        operation_id=operation_id,
        invocation=invocation,
        graph_result="no graph change",
        watcher_ids=[],
        context_baseline={"ontology": {"sha256": "abc"}},
        ending_signal=ending_signal,
    )


def test_fresh_schema_and_root_admission_use_generic_parent(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id, task = _admit_root(store)

    with store.connection() as connection:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        child_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(experiment_episode_state)").fetchall()
        }

    assert "experiment_episode_state" in tables
    assert "experiment_episodes" not in tables
    assert (
        not {
            "project_id",
            "control_node_id",
            "status",
            "invocation_ceiling",
            "invocations_used",
            "stop_requested_at",
            "stop_settled_at",
        }
        & child_columns
    )
    episode = store.episode(episode_id)
    state = store.experiment_episode(episode_id)
    assert episode is not None
    assert episode.status == "running"
    assert episode.invocations_used == 1
    assert state is not None and not state.session_bound
    assert task.episode_id == episode_id
    assert [item.operation_id for item in store.episode_invocations(episode_id)] == ["loop-root"]


def test_one_live_parent_per_experiment_survives_a_turn_that_wakes_nothing(
    tmp_path: Path,
) -> None:
    """One Experiment holds one live episode, and Stop loop is what releases it.

    A turn can succeed below the ceiling while arming no observer and taking no
    exit. The runtime then reads inactive, but the parent row is still live and
    admission refuses the next episode, so readiness has to carry that fact.
    """

    store = AppStore(tmp_path / "rcp.sqlite3")
    first_id, first_task = _admit_root(store, operation_id="loop-root", ceiling=10)
    store.complete_agent_task(first_task.operation_id, applied_revision=None, result={})

    stranded = store.episode(first_id)
    assert stranded is not None and stranded.status == "running" and stranded.ending is None
    runtime = store.experiment_loop_runtime("project", "exp-one")
    assert not runtime.active and not runtime.paused and not runtime.task_active
    assert runtime.episode_live

    with pytest.raises(ValueError, match="already has a live episode"):
        store.create_experiment_episode_with_invocation(
            _task(store, "loop-root-2", str(uuid.uuid4()), ceiling=10)
        )

    assert store.request_experiment_loop_stop("project", "exp-one") is not None
    released = store.episode(first_id)
    assert released is not None
    assert released.status == "stopped" and released.ending == "stopped"
    assert not store.experiment_loop_runtime("project", "exp-one").episode_live

    store.create_experiment_episode_with_invocation(
        _task(store, "loop-root-3", str(uuid.uuid4()), ceiling=10)
    )


def test_failed_root_insert_rolls_back_parent_and_child(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id = str(uuid.uuid4())
    task = _task(store, "bad-root", episode_id)
    request = dict(task.request)
    request.pop("control_completion_criteria")

    with pytest.raises(ValueError, match="completion criteria"):
        store.create_experiment_episode_with_invocation(
            task.model_copy(update={"request": request})
        )

    assert store.episode(episode_id) is None
    assert store.experiment_episode(episode_id) is None
    assert store.agent_task("bad-root") is None


def test_recovery_reuses_paid_allocation_and_can_be_exact_ending_task(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id, _ = _admit_root(store)
    store.fail_agent_task("loop-root", "provider failed")
    child = _task(
        store,
        "loop-retry",
        episode_id,
        parent_operation_id="loop-root",
        attempt=2,
        session_id="native-session",
        stage_root="/tmp/exact-experiment-stage",
    )

    stored = store.create_experiment_recovery_task(child)
    store.complete_agent_task(stored.operation_id, applied_revision=None, result={})
    signal = {
        "episode_id": episode_id,
        "ending": "completed",
        "partial": False,
        "receipt": {"semantic_signals": ["experiment_completed"]},
    }
    _bind(store, episode_id, stored.operation_id, invocation=1, ending_signal=signal)

    episode = store.episode(episode_id)
    assert episode is not None and episode.invocations_used == 1
    assert [item.operation_id for item in store.episode_invocations(episode_id)] == ["loop-root"]
    assert store.experiment_episode_ending_signal(episode_id) == ("loop-retry", signal)


def test_binding_and_ending_receipt_commit_or_roll_back_together(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id, _ = _admit_root(store)
    store.complete_agent_task("loop-root", applied_revision=None, result={})
    oversized = {
        "episode_id": episode_id,
        "ending": "completed",
        "partial": False,
        "receipt": {"summary": "x" * AGENT_TASK_RECEIPT_MAX_BYTES},
    }

    with pytest.raises(ValueError, match="storage limit"):
        _bind(store, episode_id, "loop-root", invocation=1, ending_signal=oversized)

    state = store.experiment_episode(episode_id)
    assert state is not None and not state.session_bound
    assert store.experiment_episode_ending_signal(episode_id) is None


def test_compound_handoff_rolls_back_watchers_before_episode_binding(
    tmp_path: Path, monkeypatch
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id, _ = _admit_root(store)
    store.fail_agent_task("loop-root", "provider failed")
    retry = _task(
        store,
        "loop-retry",
        episode_id,
        parent_operation_id="loop-root",
        attempt=2,
        session_id="native-session",
        stage_root="/tmp/exact-experiment-stage",
    )
    store.create_experiment_recovery_task(retry)
    continuation = _continuation(episode_id).model_copy(
        update={
            "patch_kind": "experiment_loop",
            "control_node_id": "exp-one",
            "control_episode_id": episode_id,
            "control_invocation": 1,
        }
    )
    binding = WatcherBinding(
        project_id="project",
        origin_operation_id="loop-root",
        origin_task_kind="node_chat",
        chat_id="episode-chat",
        node_id="exp-one",
        episode_id=episode_id,
        continuation=continuation,
    )
    now = store.now()
    watcher = WatcherRecord(
        watcher_id="loop-watcher",
        project_id="project",
        origin_operation_id="loop-root",
        origin_task_kind="node_chat",
        chat_id="episode-chat",
        node_id="exp-one",
        episode_id=episode_id,
        execution_host="",
        check_command="true",
        log_path="/tmp/loop-watcher.log",
        cwd="/tmp",
        continuation=continuation,
        status="completed",
        created_at=now,
        completed_at=now,
    )

    original_commit = store._commit_experiment_episode_turn

    def fail_after_watcher_insert(connection, **_kwargs: object) -> None:
        assert (
            connection.execute(
                "SELECT 1 FROM watchers WHERE watcher_id = 'loop-watcher'"
            ).fetchone()
            is not None
        )
        raise RuntimeError("simulated episode binding failure")

    monkeypatch.setattr(store, "_commit_experiment_episode_turn", fail_after_watcher_insert)
    with pytest.raises(RuntimeError, match="simulated episode binding failure"):
        store.commit_experiment_episode_handoff(
            [watcher],
            binding=binding,
            operation_id="loop-retry",
            native_session_id="native-session",
            stage_host=None,
            stage_root="/tmp/exact-experiment-stage",
            graph_result="no graph change",
            context_baseline={"ontology": {"sha256": "abc"}},
        )

    assert store.watcher("loop-watcher") is None
    state = store.experiment_episode(episode_id)
    assert state is not None and not state.session_bound

    monkeypatch.setattr(store, "_commit_experiment_episode_turn", original_commit)
    stored_watchers, stored_episode = store.commit_experiment_episode_handoff(
        [watcher],
        binding=binding,
        operation_id="loop-retry",
        native_session_id="native-session",
        stage_host=None,
        stage_root="/tmp/exact-experiment-stage",
        graph_result="no graph change",
        context_baseline={"ontology": {"sha256": "abc"}},
    )

    assert [item.watcher_id for item in stored_watchers] == ["loop-watcher"]
    assert stored_watchers[0].origin_operation_id == "loop-root"
    assert stored_episode.session_bound
    assert stored_episode.last_turn_operation_id == "loop-retry"
    assert stored_episode.last_watcher_ids == ["loop-watcher"]


def test_watcher_claim_spends_once_and_exhausts_only_after_task_settles(
    tmp_path: Path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id, _ = _admit_root(store, ceiling=2)
    store.complete_agent_task("loop-root", applied_revision=None, result={})
    _bind(store, episode_id, "loop-root", invocation=1)
    continuation = _continuation(episode_id)
    now = store.now()
    store.create_watchers(
        [
            WatcherRecord(
                watcher_id="evaluation-done",
                project_id="project",
                origin_operation_id="loop-root",
                origin_task_kind="node_chat",
                chat_id="episode-chat",
                node_id="exp-one",
                episode_id=episode_id,
                execution_host="",
                check_command="true",
                log_path="/tmp/evaluation.log",
                cwd="/tmp",
                continuation=continuation,
                status="completed",
                created_at=now,
                completed_at=now,
            )
        ]
    )
    wake = _task(
        store,
        "watcher-wake",
        episode_id,
        invocation=2,
        ceiling=2,
        trigger="watcher",
        watcher_ids=["evaluation-done"],
        session_id="native-session",
        stage_root="/tmp/exact-experiment-stage",
    )

    assert store.create_experiment_watcher_invocation(wake, ["evaluation-done"]) is not None
    episode = store.episode(episode_id)
    assert episode is not None and episode.invocations_used == 2
    assert episode.ending is None
    assert store.watcher("evaluation-done").notification_operation_id == "watcher-wake"
    assert not store.experiment_loop_runtime("project", "exp-one").paused

    store.complete_agent_task("watcher-wake", applied_revision=None, result={})
    _bind(store, episode_id, "watcher-wake", invocation=2)
    runtime = store.experiment_loop_runtime("project", "exp-one")
    assert runtime.paused
    assert not runtime.task_active
    assert store.episode(episode_id).ending is None


def test_non_stop_wrapup_preserves_unnotified_observers_for_fresh_human_episode(
    tmp_path: Path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    old_episode_id, _ = _admit_root(store, ceiling=1)
    store.complete_agent_task("loop-root", applied_revision=None, result={})
    now = store.now()
    continuation = _continuation(old_episode_id)
    store.create_watchers(
        [
            WatcherRecord(
                watcher_id="pending-completion",
                project_id="project",
                origin_operation_id="loop-root",
                origin_task_kind="node_chat",
                chat_id="episode-chat",
                node_id="exp-one",
                episode_id=old_episode_id,
                execution_host="",
                check_command="true",
                log_path="/tmp/pending-completion.log",
                cwd="/tmp",
                continuation=continuation,
                status="completed",
                created_at=now,
                completed_at=now,
            ),
            WatcherRecord(
                watcher_id="unfinished-observer",
                project_id="project",
                origin_operation_id="loop-root",
                origin_task_kind="node_chat",
                chat_id="episode-chat",
                node_id="exp-one",
                episode_id=old_episode_id,
                execution_host="",
                check_command="true",
                log_path="/tmp/unfinished-observer.log",
                cwd="/tmp",
                continuation=continuation,
                created_at=now,
            ),
        ]
    )
    ending_diagnostic = "The operational invocation ceiling was exhausted."
    store.fence_episode_ending(
        old_episode_id,
        "exhausted",
        diagnostic=ending_diagnostic,
    )

    assert store.settle_experiment_episode_wrapup(old_episode_id)

    pending = store.watcher("pending-completion")
    unfinished = store.watcher("unfinished-observer")
    assert pending is not None and pending.status == "completed" and not pending.notified
    assert unfinished is not None and unfinished.status == "active" and not unfinished.notified

    receipt_json, receipt_sha256 = compact_episode_receipt(
        {"ending": "exhausted", "episode_id": old_episode_id}
    )
    report_error = "The exact report continuation is unavailable."
    terminal_at = store.now()
    store.fail_episode_wrapup_unlaunchable(
        old_episode_id,
        EpisodeWrapupRecord(
            episode_id=old_episode_id,
            ending="exhausted",
            partial=True,
            concluding_operation_id="loop-root",
            receipt_json=receipt_json,
            receipt_sha256=receipt_sha256,
            state="failed",
            diagnostic=report_error,
            created_at=terminal_at,
            updated_at=terminal_at,
            finished_at=terminal_at,
        ),
        ending_diagnostic=ending_diagnostic,
    )
    completed_after_wrapup = store.record_watcher_check(
        "unfinished-observer",
        status="completed",
        exit_code=0,
        error=None,
    )
    assert completed_after_wrapup.status == "completed"
    assert not completed_after_wrapup.notified

    stale_wake = _task(
        store,
        "stale-watcher-wake",
        old_episode_id,
        invocation=2,
        ceiling=1,
        trigger="watcher",
        watcher_ids=["pending-completion", "unfinished-observer"],
        session_id="native-session",
        stage_root="/tmp/exact-experiment-stage",
    )
    assert (
        store.create_experiment_watcher_invocation(
            stale_wake,
            ["pending-completion", "unfinished-observer"],
        )
        is None
    )
    assert store.agent_task("stale-watcher-wake") is None
    assert store.watcher("pending-completion") == pending
    assert store.watcher("unfinished-observer") == completed_after_wrapup

    new_episode_id = str(uuid.uuid4())
    unauthorized = _task(
        store,
        "unauthorized-fresh-root",
        new_episode_id,
        ceiling=1,
        watcher_ids=["pending-completion", "unfinished-observer"],
    ).model_copy(update={"authorized_by": None})
    with pytest.raises(ValueError, match="human authorization"):
        store.create_experiment_episode_with_invocation(
            unauthorized,
            ["pending-completion", "unfinished-observer"],
        )
    assert store.watcher("pending-completion") == pending

    fresh_root = _task(
        store,
        "fresh-root",
        new_episode_id,
        ceiling=1,
        watcher_ids=["pending-completion", "unfinished-observer"],
    )
    store.create_experiment_episode_with_invocation(
        fresh_root,
        ["pending-completion", "unfinished-observer"],
    )

    for watcher_id in ("pending-completion", "unfinished-observer"):
        adopted = store.watcher(watcher_id)
        assert adopted is not None and adopted.status == "completed" and adopted.notified
        assert adopted.notification_operation_id == "fresh-root"
        assert adopted.continuation.control_episode_id == old_episode_id
    new_episode = store.episode(new_episode_id)
    assert new_episode is not None and new_episode.invocations_used == 1


def test_stop_fences_then_skips_report_only_after_quiescence(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    episode_id, _ = _admit_root(store)
    now = store.now()
    store.create_watchers(
        [
            WatcherRecord(
                watcher_id="stop-pending-completion",
                project_id="project",
                origin_operation_id="loop-root",
                origin_task_kind="node_chat",
                chat_id="episode-chat",
                node_id="exp-one",
                episode_id=episode_id,
                execution_host="",
                check_command="true",
                log_path="/tmp/stop-pending-completion.log",
                cwd="/tmp",
                continuation=_continuation(episode_id),
                status="completed",
                created_at=now,
                completed_at=now,
            )
        ]
    )

    stopping = store.request_experiment_loop_stop("project", "exp-one")
    assert stopping is not None and stopping.stop_requested_at is not None
    assert stopping.stop_settled_at is None
    assert store.episode(episode_id).status == "stopping"

    store.complete_agent_task("loop-root", applied_revision=None, result={})
    settled = store.settle_experiment_loop_stop("project", "exp-one")
    assert settled is not None and settled.stop_settled_at is not None
    episode = store.episode(episode_id)
    wrapup = store.episode_wrapup(episode_id)
    assert episode is not None and episode.ending == "stopped"
    assert episode.wrapup_state == "skipped"
    assert wrapup is not None and wrapup.state == "skipped"
    stopped_watcher = store.watcher("stop-pending-completion")
    assert stopped_watcher is not None
    assert stopped_watcher.status == "stopped"
    assert stopped_watcher.notified


def test_legacy_combined_rows_and_missing_state_roots_migrate_one_way(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rcp.sqlite3"
    store = AppStore(path)
    old_episode_id = str(uuid.uuid4())
    missing_state_id = str(uuid.uuid4())
    owner = _identity(store)
    now = store.now()
    with sqlite3.connect(path) as connection:
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
        for index, episode_id in enumerate((old_episode_id, missing_state_id), start=1):
            request = _request(episode_id, control_node_id=f"exp-{index}")
            connection.execute(
                """
                INSERT INTO graph_runs (
                    operation_id, project_id, episode_id, kind, status, request_json,
                    created_at, updated_at, status_message, attempt, estimate_seconds,
                    estimate_samples, phase, authorized_space_id, authorized_user_id,
                    authorized_display_name, visible
                ) VALUES (?, 'project', ?, 'node_chat', 'succeeded', ?, ?, ?, 'done',
                          1, 300, 0, 'complete', ?, ?, ?, 1)
                """,
                (
                    f"legacy-root-{index}",
                    episode_id,
                    json.dumps(request, separators=(",", ":")),
                    now,
                    now,
                    owner.space_id,
                    owner.user_id,
                    owner.display_name,
                ),
            )
        connection.execute(
            """
            INSERT INTO experiment_episodes (
                episode_id, project_id, control_node_id, provider, execution_machine,
                native_session_id, stage_root, chat_id, last_turn_operation_id,
                last_turn_invocation, created_at, updated_at
            ) VALUES (?, 'project', 'exp-1', 'codex', 'laptop', 'legacy-session',
                      '/tmp/legacy-stage', 'episode-chat', 'legacy-root-1', 1, ?, ?)
            """,
            (old_episode_id, now, now),
        )
        connection.commit()

    reopened = AppStore(path)

    assert reopened.experiment_episode(old_episode_id).native_session_id == "legacy-session"
    missing = reopened.experiment_episode(missing_state_id)
    assert missing is not None and not missing.session_bound
    assert reopened.episode(old_episode_id) is not None
    assert reopened.episode(missing_state_id) is not None
    with reopened.connection() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'experiment_episodes'"
            ).fetchone()
            is None
        )
    reopened_again = AppStore(path)
    assert reopened_again.experiment_episode(old_episode_id).native_session_id == "legacy-session"
