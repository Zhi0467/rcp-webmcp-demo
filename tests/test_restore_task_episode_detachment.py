from __future__ import annotations

import hashlib
import uuid

import pytest

from rcp.core.models import AuthorizedHuman
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    EpisodeReportRecord,
    EpisodeWrapupRecord,
    TeamAuthenticationError,
)
from rcp.storage.episodes import compact_episode_receipt

RESTORE_DIAGNOSTIC = "This work was interrupted because RCP restored an older snapshot."


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


def _experiment_task(
    store: AppStore,
    *,
    episode_id: str,
    control_node_id: str,
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=f"{episode_id}-root",
        project_id="project",
        episode_id=episode_id,
        kind="node_chat",
        status="queued",
        request={
            "patch_kind": "experiment_loop",
            "trigger": "experiment_run",
            "control_episode_id": episode_id,
            "control_node_id": control_node_id,
            "node_id": control_node_id,
            "control_invocation": 1,
            "control_invocation_ceiling": 10,
            "control_revision": 0,
            "control_decision_bundle": [],
            "control_completion_criteria": [],
            "watcher_ids": [],
            "provider": "codex",
            "run_on": "laptop",
            "chat_id": f"{episode_id}-chat",
        },
        created_at=now,
        updated_at=now,
        status_message="Queued",
        authorized_by=_authorizer(store),
    )


def _start_experiment(
    store: AppStore,
    *,
    episode_id: str,
    control_node_id: str,
    complete_root: bool,
) -> AgentTaskRecord:
    root = _experiment_task(
        store,
        episode_id=episode_id,
        control_node_id=control_node_id,
    )
    store.create_experiment_episode_with_invocation(root)
    store.checkpoint_agent_task(
        root.operation_id,
        native_session_id=f"{episode_id}-native-session",
        stage_root=f"/tmp/{episode_id}-stage",
    )
    store.commit_experiment_episode_turn(
        episode_id=episode_id,
        project_id="project",
        control_node_id=control_node_id,
        provider="codex",
        execution_machine="laptop",
        execution_host="",
        native_session_id=f"{episode_id}-native-session",
        stage_host=None,
        stage_root=f"/tmp/{episode_id}-stage",
        chat_id=f"{episode_id}-chat",
        operation_id=root.operation_id,
        invocation=1,
        graph_result="applied",
        watcher_ids=[],
        context_baseline={},
    )
    if complete_root:
        store.mark_agent_task_running(root.operation_id)
        store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    stored = store.agent_task(root.operation_id)
    assert stored is not None
    return stored


def _begin_wrapup(
    store: AppStore,
    *,
    episode_id: str,
) -> tuple[EpisodeWrapupRecord, AgentTaskRecord, str]:
    now = store.now()
    allocation_operation_id = f"{episode_id}-report"
    receipt_json, receipt_sha256 = compact_episode_receipt(
        {
            "ending": "completed",
            "episode_id": episode_id,
            "source_operation_id": f"{episode_id}-root",
        }
    )
    wrapup = EpisodeWrapupRecord(
        episode_id=episode_id,
        ending="completed",
        partial=False,
        concluding_operation_id=f"{episode_id}-root",
        allocation_operation_id=allocation_operation_id,
        provider="codex",
        run_on="laptop",
        execution_host="",
        native_session_id=f"{episode_id}-report-session",
        stage_host=None,
        stage_root=f"/tmp/{episode_id}-report-stage",
        skill_id="episode-report",
        skill_version="1",
        output_name="episode-report.html",
        output_path=f"/tmp/{episode_id}-report-stage/episode-report.html",
        receipt_json=receipt_json,
        receipt_sha256=receipt_sha256,
        state="pending",
        created_at=now,
        updated_at=now,
    )
    task = AgentTaskRecord(
        operation_id=allocation_operation_id,
        project_id="project",
        episode_id=episode_id,
        kind="episode_report",
        status="queued",
        request={"provider": "codex", "run_on": "laptop", "execution_host": ""},
        created_at=now,
        updated_at=now,
        status_message="Wrapping up visualization and report",
        parent_operation_id=f"{episode_id}-root",
        native_session_id=f"{episode_id}-report-session",
        stage_root=f"/tmp/{episode_id}-report-stage",
        visible=False,
    )
    store.begin_episode_wrapup(episode_id, wrapup, task)
    attempt = store.allocate_episode_report_attempt(episode_id)
    store.mark_episode_report_attempt_running(attempt.attempt_id)
    return wrapup, task, attempt.attempt_id


def _finish_report(
    store: AppStore,
    *,
    episode_id: str,
    allocation_operation_id: str,
    attempt_id: str,
) -> EpisodeReportRecord:
    html = f"<article><h1>{episode_id}</h1></article>"
    report = EpisodeReportRecord(
        report_id=f"{episode_id}-report-id",
        episode_id=episode_id,
        attempt_id=attempt_id,
        allocation_operation_id=allocation_operation_id,
        ending="completed",
        sha256=hashlib.sha256(html.encode("utf-8")).hexdigest(),
        html=html,
        created_at=store.now(),
    )
    store.finish_episode_report_ready(attempt_id, report)
    return report


def test_restore_helpers_detach_tasks_experiments_and_reports_idempotently(tmp_path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    active_episode_id = str(uuid.uuid4())
    wrapping_episode_id = str(uuid.uuid4())
    completed_episode_id = str(uuid.uuid4())
    active = _start_experiment(
        store,
        episode_id=active_episode_id,
        control_node_id="active-node",
        complete_root=False,
    )
    _start_experiment(
        store,
        episode_id=wrapping_episode_id,
        control_node_id="wrapping-node",
        complete_root=True,
    )
    wrapping_wrapup, wrapping_task, wrapping_attempt_id = _begin_wrapup(
        store,
        episode_id=wrapping_episode_id,
    )
    _start_experiment(
        store,
        episode_id=completed_episode_id,
        control_node_id="completed-node",
        complete_root=True,
    )
    _completed_wrapup, completed_task, completed_attempt_id = _begin_wrapup(
        store,
        episode_id=completed_episode_id,
    )
    completed_report = _finish_report(
        store,
        episode_id=completed_episode_id,
        allocation_operation_id=completed_task.operation_id,
        attempt_id=completed_attempt_id,
    )
    completed_episode_before = store.episode(completed_episode_id)
    completed_wrapup_before = store.episode_wrapup(completed_episode_id)
    terminal = AgentTaskRecord(
        operation_id="terminal-task",
        project_id="project",
        kind="paper_coach",
        status="succeeded",
        request={},
        created_at=store.now(),
        updated_at=store.now(),
        finished_at=store.now(),
        status_message="Completed",
        native_session_id="terminal-native-session",
        result={"messages": ["Preserved answer."]},
        authorized_by=_authorizer(store),
    )
    store.create_agent_task(terminal)
    store.record_agent_task_receipt("terminal-task", "historical_result", {"kept": True})
    terminal_receipts_before = store.agent_task_receipts("terminal-task")
    paused = terminal.model_copy(
        update={
            "operation_id": "paused-task",
            "status": "paused",
            "finished_at": None,
            "status_message": "Paused for review",
            "native_session_id": "paused-native-session",
            "stage_root": "/tmp/paused-stage",
            "result": None,
        }
    )
    store.create_agent_task(paused)
    now = store.now()
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO writing_sessions (
                native_session_id, provider, runtime_id, execution_machine,
                project_id, title, model, reasoning, created_at, last_resumed_at,
                introduction_hash_examined, graph_revision_examined,
                research_md_hash_examined
            ) VALUES (?, 'codex', '', 'laptop', 'project', 'Restore', '', 'medium',
                      ?, ?, '', 0, '')
            """,
            (f"{active.episode_id}-native-session", now, now),
        )

    with store.connection() as connection:
        with pytest.raises(ValueError, match="active transaction"):
            store.detach_agent_tasks_for_restore(
                connection,
                diagnostic=RESTORE_DIAGNOSTIC,
                now=now,
            )
        with pytest.raises(ValueError, match="active transaction"):
            store.detach_episode_reports_for_restore(
                connection,
                diagnostic=RESTORE_DIAGNOSTIC,
                now=now,
            )
        with pytest.raises(ValueError, match="active transaction"):
            store.detach_experiment_episodes_for_restore(
                connection,
                diagnostic=RESTORE_DIAGNOSTIC,
                now=now,
            )

    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        store.detach_agent_tasks_for_restore(
            connection,
            diagnostic=RESTORE_DIAGNOSTIC,
            now=now,
        )
        store.detach_episode_reports_for_restore(
            connection,
            diagnostic=RESTORE_DIAGNOSTIC,
            now=now,
        )
        store.detach_experiment_episodes_for_restore(
            connection,
            diagnostic=RESTORE_DIAGNOSTIC,
            now=now,
        )

    tasks = store.agent_tasks("project", include_hidden=True)
    assert tasks and all(task.history_only for task in tasks)
    assert store.agent_task(active.operation_id).status == "interrupted"
    assert store.agent_task(paused.operation_id).status == "interrupted"
    assert store.agent_task("terminal-task").status == "succeeded"
    assert store.agent_task("terminal-task").result == {"messages": ["Preserved answer."]}
    assert store.agent_task("terminal-task").authorized_by == terminal.authorized_by
    assert store.agent_task_receipts("terminal-task") == terminal_receipts_before
    assert store.agent_task(completed_task.operation_id).status == "succeeded"
    interruption_receipts = [
        receipt
        for receipt in store.agent_task_receipts(active.operation_id)
        if receipt.category == "operation_interrupted"
    ]
    assert len(interruption_receipts) == 1
    assert interruption_receipts[0].payload == {
        "reason": "restore",
        "status": "interrupted",
    }

    for episode_id in (active_episode_id, wrapping_episode_id):
        episode = store.episode(episode_id)
        wrapup = store.episode_wrapup(episode_id)
        assert episode is not None and wrapup is not None
        assert episode.status == "stopped"
        assert episode.ending == "stopped"
        assert episode.wrapup_state == "skipped"
        assert episode.ending_diagnostic == RESTORE_DIAGNOSTIC
        assert wrapup.state == "skipped"
        assert wrapup.ending == "stopped"
        assert wrapup.native_session_id is None
        assert wrapup.stage_host is None
        assert wrapup.stage_root is None
    wrapping_attempt = store.episode_report_attempt(wrapping_attempt_id)
    assert wrapping_attempt is not None
    assert wrapping_attempt.status == "failed"
    assert wrapping_attempt.error == RESTORE_DIAGNOSTIC
    assert (
        wrapping_wrapup.receipt_sha256 != store.episode_wrapup(wrapping_episode_id).receipt_sha256
    )

    assert store.episode(completed_episode_id) == completed_episode_before
    assert store.episode_wrapup(completed_episode_id) == completed_wrapup_before
    assert store.episode_report(completed_episode_id) == completed_report
    assert store.episode_report_attempt(completed_attempt_id).status == "succeeded"
    with store.connection() as connection:
        raw_task = connection.execute(
            "SELECT native_session_id FROM graph_runs WHERE operation_id = 'terminal-task'"
        ).fetchone()
        assert raw_task["native_session_id"] == "terminal-native-session"
        assert connection.execute("SELECT COUNT(*) FROM writing_sessions").fetchone()[0] == 0
        state_rows = connection.execute(
            """
            SELECT native_session_id, stage_host, stage_root
            FROM experiment_episode_state ORDER BY episode_id
            """
        ).fetchall()
        assert state_rows
        assert all(tuple(row) == (None, None, None) for row in state_rows)
        before_counts = {
            "events": connection.execute("SELECT COUNT(*) FROM graph_run_events").fetchone()[0],
            "receipts": connection.execute("SELECT COUNT(*) FROM graph_run_receipts").fetchone()[0],
        }

    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        store.detach_agent_tasks_for_restore(
            connection,
            diagnostic=RESTORE_DIAGNOSTIC,
            now=now,
        )
        store.detach_episode_reports_for_restore(
            connection,
            diagnostic=RESTORE_DIAGNOSTIC,
            now=now,
        )
        store.detach_experiment_episodes_for_restore(
            connection,
            diagnostic=RESTORE_DIAGNOSTIC,
            now=now,
        )
    with store.connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM graph_run_events").fetchone()[0]
            == (before_counts["events"])
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM graph_run_receipts").fetchone()[0]
            == (before_counts["receipts"])
        )


def test_restore_space_auth_detachment_preserves_permanent_member_tokens(tmp_path) -> None:
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    member, member_token = store.enroll_team_member(bootstrap, "Alice")
    browser_session, _member = store.create_team_session(member_token)
    invitation, invitation_code = store.create_team_invitation(member.user_id)
    with store.connection() as connection:
        token_rows_before = connection.execute(
            "SELECT * FROM team_member_tokens ORDER BY token_id"
        ).fetchall()

    now = store.now()
    with store.connection() as connection:
        with pytest.raises(ValueError, match="requires a transaction"):
            store.detach_space_authentication_for_restore(connection, now=now)
        connection.execute("BEGIN IMMEDIATE")
        store.detach_space_authentication_for_restore(connection, now=now)

    assert store.resolve_team_session(browser_session) is None
    with pytest.raises(TeamAuthenticationError) as revoked:
        store.enroll_team_member(invitation_code, "Bob")
    assert revoked.value.code == "enrollment_code_invalid"
    with store.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM team_sessions").fetchone()[0] == 0
        invitation_row = connection.execute(
            "SELECT consumed_at, locked_at, revoked_at FROM team_invitations "
            "WHERE invitation_id = ?",
            (invitation.invitation_id,),
        ).fetchone()
        assert tuple(invitation_row) == (None, None, now)
        bootstrap_row = connection.execute(
            "SELECT consumed_at, locked_at, revoked_at FROM team_bootstrap_codes"
        ).fetchone()
        assert bootstrap_row["consumed_at"] is not None
        assert bootstrap_row["locked_at"] is None
        assert bootstrap_row["revoked_at"] is None
        assert (
            connection.execute("SELECT * FROM team_member_tokens ORDER BY token_id").fetchall()
            == token_rows_before
        )
        connection.execute("BEGIN IMMEDIATE")
        store.detach_space_authentication_for_restore(connection, now=store.now())
        invitation_row_again = connection.execute(
            "SELECT consumed_at, locked_at, revoked_at FROM team_invitations "
            "WHERE invitation_id = ?",
            (invitation.invitation_id,),
        ).fetchone()
        assert tuple(invitation_row_again) == (None, None, now)


def test_restore_revokes_an_unconsumed_bootstrap_code(tmp_path) -> None:
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    now = store.now()
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        store.detach_space_authentication_for_restore(connection, now=now)

    with pytest.raises(TeamAuthenticationError) as revoked:
        store.enroll_team_member(bootstrap, "Alice")
    assert revoked.value.code == "enrollment_code_invalid"
    with store.connection() as connection:
        row = connection.execute(
            "SELECT consumed_at, locked_at, revoked_at FROM team_bootstrap_codes"
        ).fetchone()
    assert tuple(row) == (None, None, now)
