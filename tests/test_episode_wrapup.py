from __future__ import annotations

import json
from pathlib import Path

from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.runs.episodes.wrapup import EpisodeWrapupSpec, begin_episode_report_wrapup
from rcp.storage import AgentTaskRecord, AppStore, EpisodeRecord, ProjectRecord

from .helpers import fabricated_authorizer


def _store_with_episode(tmp_path: Path) -> tuple[AppStore, Path]:
    store = AppStore(tmp_path / "app.sqlite3")
    store.upsert_project(
        ProjectRecord(
            project_id="project",
            locator=str(tmp_path),
            name="Project",
            state_location=str(tmp_path),
            state_remote=False,
            added_at=store.now(),
        )
    )
    now = store.now()
    store.create_episode(
        EpisodeRecord(
            episode_id="episode",
            project_id="project",
            mode="experiment_loop",
            control_node_id="experiment-node",
            status="queued",
            invocation_ceiling=2,
            authorized_by=fabricated_authorizer("Episode owner"),
            created_at=now,
            updated_at=now,
        )
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    operational = AgentTaskRecord(
        operation_id="operation",
        project_id="project",
        episode_id="episode",
        kind="node_chat",
        status="succeeded",
        request={
            "provider": "codex",
            "model": "",
            "reasoning": "medium",
            "run_on": "laptop",
        },
        created_at=now,
        updated_at=now,
        status_message="Complete",
        native_session_id="native-session",
        stage_root=str(stage),
        dispatch_authority=AgentDispatchAuthority(
            profile="ordinary",
            task_contract="scratch_patch",
            scope=AgentDispatchScope(run_truth_scope=[], patch_kind="refresh"),
        ),
    )
    store.allocate_episode_invocation("episode", operational)
    return store, stage


def test_shared_wrapup_admits_one_deterministic_hidden_report(tmp_path: Path) -> None:
    store, stage = _store_with_episode(tmp_path)
    spec = EpisodeWrapupSpec(
        episode_id="episode",
        ending="exhausted",
        partial=True,
        continuation_operation_id="operation",
        receipt={"observations": ["bounded evidence"]},
        diagnostic="The operational invocation ceiling was reached.",
    )

    first = begin_episode_report_wrapup(store, spec)
    second = begin_episode_report_wrapup(store, spec)

    assert first.launchable is True
    assert second.task == first.task
    assert second.request == first.request
    assert first.task is not None and first.request is not None
    assert first.task.visible is False
    assert first.task.kind == "episode_report"
    assert first.task.parent_operation_id == "operation"
    assert first.task.native_session_id == "native-session"
    assert first.task.stage_root == str(stage)
    assert first.task.status_message == "Wrapping up visualization and report"
    assert first.request.session_id == "native-session"
    assert first.wrapup.output_path == str(stage / "episode-report.html")
    receipt = json.loads(first.wrapup.receipt_json)
    assert receipt == {
        "ending": "exhausted",
        "episode_id": "episode",
        "mode": "experiment_loop",
        "observations": ["bounded evidence"],
        "partial": True,
        "diagnostic": "The operational invocation ceiling was reached.",
    }
    assert first.episode.invocations_used == 1
    assert store.agent_tasks("project") == [store.agent_task("operation")]


def test_an_ending_with_no_session_never_enters_report_wrapup(tmp_path: Path) -> None:
    store, _stage = _store_with_episode(tmp_path)
    with store.connection() as connection:
        connection.execute(
            "UPDATE graph_runs SET native_session_id = NULL, stage_root = NULL "
            "WHERE operation_id = 'operation'"
        )

    admission = begin_episode_report_wrapup(
        store,
        EpisodeWrapupSpec(
            episode_id="episode",
            ending="failed",
            partial=True,
            continuation_operation_id="operation",
            receipt={"failure": "provider session unavailable"},
            diagnostic="The operational turn failed.",
        ),
    )

    # There is no session to resume, so no report was ever possible. The episode
    # terminalizes with its own reason and carries no report error.
    assert admission.launchable is False
    assert admission.task is None
    assert admission.wrapup is None
    assert admission.episode.status == "failed"
    assert admission.episode.wrapup_state == "not_started"
    assert admission.episode.wrapup_error is None
    assert admission.episode.report_attempts_used == 0
    assert admission.episode.ending_diagnostic == "The operational turn failed."
    assert admission.episode.ended_at is not None
    assert store.episode_wrapup("episode") is None
    assert store.episode_report("episode") is None


def test_an_ending_with_no_session_settles_the_same_way_twice(tmp_path: Path) -> None:
    store, _stage = _store_with_episode(tmp_path)
    with store.connection() as connection:
        connection.execute(
            "UPDATE graph_runs SET native_session_id = NULL, stage_root = NULL "
            "WHERE operation_id = 'operation'"
        )
    spec = EpisodeWrapupSpec(
        episode_id="episode",
        ending="failed",
        partial=True,
        continuation_operation_id="operation",
        receipt={"failure": "provider session unavailable"},
        diagnostic="The operational turn failed.",
    )

    first = begin_episode_report_wrapup(store, spec)
    second = begin_episode_report_wrapup(store, spec)

    assert first.episode == second.episode
    assert store.episode_wrapup("episode") is None


def test_a_broken_binding_that_is_not_an_absence_still_reports_its_error(
    tmp_path: Path,
) -> None:
    store, _stage = _store_with_episode(tmp_path)
    with store.connection() as connection:
        connection.execute(
            "UPDATE graph_runs SET request_json = ? WHERE operation_id = 'operation'",
            (json.dumps({"provider": "codex", "model": "", "reasoning": "medium"}),),
        )

    admission = begin_episode_report_wrapup(
        store,
        EpisodeWrapupSpec(
            episode_id="episode",
            ending="completed",
            partial=False,
            continuation_operation_id="operation",
            receipt={},
        ),
    )

    assert admission.launchable is False
    assert admission.episode.wrapup_state == "failed"
    assert admission.episode.wrapup_error is not None
    assert "frozen provider profile" in admission.episode.wrapup_error


def test_stop_never_enters_report_wrapup(tmp_path: Path) -> None:
    store, _stage = _store_with_episode(tmp_path)

    try:
        begin_episode_report_wrapup(
            store,
            EpisodeWrapupSpec(
                episode_id="episode",
                ending="stopped",
                partial=True,
                continuation_operation_id="operation",
                receipt={},
            ),
        )
    except ValueError as exc:
        assert "Stop skips report generation" in str(exc)
    else:
        raise AssertionError("Stop must not allocate report work")
