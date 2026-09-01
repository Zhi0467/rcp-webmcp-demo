from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rcp.agents.acceptance import (
    ACCEPTANCE_CAMPAIGN_EXHAUST_MARKER as ACCEPTANCE_EPISODE_EXHAUST_MARKER,
)
from rcp.agents.acceptance import (
    ACCEPTANCE_CAMPAIGN_FAIL_MARKER as ACCEPTANCE_EPISODE_FAIL_MARKER,
)
from rcp.agents.acceptance import (
    ACCEPTANCE_CAMPAIGN_INTERRUPT_ACTIVE_FILE as ACCEPTANCE_EPISODE_INTERRUPT_ACTIVE_FILE,
)
from rcp.agents.acceptance import (
    ACCEPTANCE_CAMPAIGN_SPAWN_THEN_FINISH_MARKER as ACCEPTANCE_EPISODE_FINISH_MARKER,
)
from rcp.agents.acceptance import (
    ACCEPTANCE_CAMPAIGN_SPAWN_THEN_INTERRUPT_MARKER as ACCEPTANCE_EPISODE_INTERRUPT_MARKER,
)
from rcp.agents.acceptance import (
    ACCEPTANCE_CAMPAIGN_STOP_MARKER as ACCEPTANCE_EPISODE_STOP_MARKER,
)
from rcp.core.models import Patch
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.auto_research_admission import (
    start_auto_research_turn,
)
from rcp.storage import EpisodeNotRunning, GraphWatcherRecord, WatcherContinuation

from .helpers import (
    append_fixture_patch,
    create_named_app,
    wait_for_task,
)


def _wait_for_episode(
    client: TestClient,
    project_id: str,
    episode_id: str,
    *,
    status: str,
    ending: str,
    report_ready: bool | None = None,
    timeout: float = 20,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    episode: dict[str, object] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/projects/{project_id}/episodes")
        assert response.status_code == 200, response.text
        episode = next(item for item in response.json() if item["episode_id"] == episode_id)
        report_matches = report_ready is None or bool(episode["report"]) is report_ready
        if episode["status"] == status and episode["ending"] == ending and report_matches:
            return episode
        time.sleep(0.01)
    raise AssertionError(
        f"acceptance episode did not reach {status}/{ending} with report={report_ready}; "
        f"last state: {episode}"
    )


def _wait_for_path(path: Path, *, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.01)
    raise AssertionError(f"acceptance fixture path did not appear: {path}")


def _wait_for_task_stage(store, operation_id: str, *, timeout: float = 10) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = store.agent_task(operation_id)
        if task is not None and task.stage_root is not None:
            return Path(task.stage_root)
        time.sleep(0.01)
    raise AssertionError(f"acceptance task did not persist its stage: {operation_id}")


def _wait_for_child_work(store, episode_id: str, *, timeout: float = 10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        routes = store.auto_research_child_works(episode_id)
        if routes:
            assert len(routes) == 1
            route = routes[0]
            task = store.agent_task(route.current_operation_id)
            if task is not None:
                return route, task
        time.sleep(0.01)
    raise AssertionError("acceptance episode did not admit its ordinary child Work task")


def _add_worker_seat(app, *, node_id: str, title: str, objective: str) -> None:
    project_id = app.state.default_project_id
    assert project_id is not None
    service = app.state.catalog.open(project_id)
    append_fixture_patch(
        service,
        Patch(
            kind="seed",
            author="agent",
            summary=f"Added the {title.lower()} seat.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": node_id,
                            "type": "experiment",
                            "title": title,
                            "objective": objective,
                            "status": "designing",
                        }
                    ],
                }
            ],
        ),
    )


def _start_episode(
    client: TestClient,
    project_id: str,
    *,
    invocation_ceiling: int,
    starting_instruction: str,
) -> dict[str, object]:
    response = client.post(
        f"/api/projects/{project_id}/episodes",
        json={
            "mode": "auto_research",
            "invocation_ceiling": invocation_ceiling,
            "starting_instruction": starting_instruction,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def _hidden_report_task(store, episode_id: str):
    report_tasks = [
        task
        for task in store.episode_tasks(episode_id, include_hidden=True)
        if task.kind == "episode_report"
    ]
    assert len(report_tasks) == 1
    report_task = report_tasks[0]
    assert report_task.visible is False
    assert report_task.episode_id == episode_id
    return report_task


def _assert_corrected_report(store, episode_id: str):
    report_task = _hidden_report_task(store, episode_id)
    attempts = store.episode_report_attempts(episode_id)
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    assert [attempt.status for attempt in attempts] == ["failed", "succeeded"]
    assert attempts[0].error is not None
    launches = [
        receipt
        for receipt in store.agent_task_receipts(report_task.operation_id)
        if receipt.category == "agent_launch"
    ]
    assert [receipt.payload["attempt_number"] for receipt in launches] == [1, 2]
    assert {receipt.payload["continuation_cause"] for receipt in launches} == {"episode_report"}
    assert all(receipt.payload["resumed"] is True for receipt in launches)
    return report_task


def test_acceptance_episode_completes_and_corrects_one_hidden_report(
    manifest,
    tmp_path,
) -> None:
    app = create_named_app(
        str(manifest.path),
        data_dir=tmp_path / "data",
        acceptance_agent=True,
    )
    project_id = app.state.default_project_id
    assert project_id is not None
    _add_worker_seat(
        app,
        node_id="exp/episode-acceptance",
        title="Episode acceptance worker",
        objective="Complete one deterministic bounded worker turn.",
    )
    store = app.state.background_tasks.store

    with TestClient(app) as client:
        started = _start_episode(
            client,
            project_id,
            invocation_ceiling=3,
            starting_instruction=ACCEPTANCE_EPISODE_FINISH_MARKER,
        )
        episode_id = str(started["episode_id"])
        episode = _wait_for_episode(
            client,
            project_id,
            episode_id,
            status="completed",
            ending="completed",
            report_ready=True,
        )

        budget = episode["budget"]
        assert isinstance(budget, dict)
        assert budget["invocation_ceiling"] == 3
        assert budget["invocations_used"] == 3
        assert budget["invocations_remaining"] == 0
        assert episode["wrapup_state"] == "ready"
        assert episode["report"] is not None
        assert all(task["kind"] != "episode_report" for task in episode["tasks"])
        preview = client.get(f"/api/projects/{project_id}/episodes/{episode_id}/report/content")
        assert preview.status_code == 200, preview.text
        assert "Acceptance episode conclusion" in preview.text
        assert "Episode progression" in preview.text

    tasks = store.auto_research_tasks(episode_id)
    roles = [store.auto_research_invocation_role(task.operation_id) for task in tasks]
    assert roles.count("orchestrator") == 2
    assert roles.count("worker") == 0
    child_routes = store.auto_research_child_works(episode_id)
    assert len(child_routes) == 1
    child = store.agent_task(child_routes[0].current_operation_id)
    assert child is not None and child.kind == "node_chat" and child.status == "succeeded"
    assert child.parent_operation_id is None
    assert [item.operation_id for item in store.episode_invocations(episode_id)] == [
        tasks[0].operation_id,
        child.operation_id,
        tasks[1].operation_id,
    ]
    root = next(task for task in tasks if task.parent_operation_id is None)
    report_task = _assert_corrected_report(store, episode_id)
    assert report_task.native_session_id == root.native_session_id
    assert report_task.stage_host == root.stage_host
    assert report_task.stage_root == root.stage_root
    assert store.agent_command_by_key(episode_id, "acceptance-spawn") is not None
    assert store.agent_command_by_key(episode_id, "acceptance-finish-after-worker") is not None
    assert store.episode_report(episode_id) is not None


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_acceptance_episode_restart_retry_reuses_the_successful_spawn(
    manifest,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    app = create_named_app(
        str(manifest.path),
        data_dir=data_dir,
        acceptance_agent=True,
    )
    project_id = app.state.default_project_id
    assert project_id is not None
    _add_worker_seat(
        app,
        node_id="exp/episode-interrupted-spawn",
        title="Episode interrupted spawn",
        objective="Prove a successful worker spawn is never repeated.",
    )
    store = app.state.background_tasks.store

    with TestClient(app) as client:
        started = _start_episode(
            client,
            project_id,
            invocation_ceiling=10,
            starting_instruction=ACCEPTANCE_EPISODE_INTERRUPT_MARKER,
        )
        episode_id = str(started["episode_id"])
        root_operation_id = str(started["root_operation_id"])
        root_stage = _wait_for_task_stage(store, root_operation_id)
        active_path = root_stage / ACCEPTANCE_EPISODE_INTERRUPT_ACTIVE_FILE
        _wait_for_path(active_path)
        worker_route, worker = _wait_for_child_work(store, episode_id)
        wait_for_task(store, worker.operation_id, expect="succeeded")
        root_before_restart = store.agent_task(root_operation_id)
        assert root_before_restart is not None
        assert root_before_restart.status == "running"
        assert root_before_restart.native_session_id is not None

    abandoned = store.agent_task(root_operation_id)
    assert abandoned is not None and abandoned.status == "pausing"
    assert active_path.is_file()

    restarted = create_named_app(
        str(manifest.path),
        data_dir=data_dir,
        acceptance_agent=True,
    )
    restarted_store = restarted.state.background_tasks.store

    with TestClient(restarted) as client:
        # Startup reconciles the abandoned root, not construction, so this is
        # read inside the lifespan rather than immediately after create_app.
        interrupted = restarted_store.agent_task(root_operation_id)
        assert interrupted is not None and interrupted.status == "interrupted"
        retried = client.post(f"/api/projects/{project_id}/tasks/{root_operation_id}/retry")
        assert retried.status_code == 202, retried.text
        retry_operation_id = retried.json()["operation_id"]
        episode = _wait_for_episode(
            client,
            project_id,
            episode_id,
            status="completed",
            ending="completed",
            report_ready=True,
        )

    assert not active_path.exists()
    tasks = restarted_store.auto_research_tasks(episode_id)
    roles = [restarted_store.auto_research_invocation_role(task.operation_id) for task in tasks]
    assert roles.count("orchestrator") == 2
    assert roles.count("worker") == 0
    child_routes = restarted_store.auto_research_child_works(episode_id)
    assert len(child_routes) == 1
    assert child_routes[0].worker_id == worker_route.worker_id
    assert child_routes[0].root_operation_id == worker.operation_id
    assert child_routes[0].current_operation_id == worker.operation_id
    budget = episode["budget"]
    assert isinstance(budget, dict)
    assert budget["invocations_used"] == 2
    assert budget["invocations_remaining"] == 8
    retry = restarted_store.agent_task(retry_operation_id)
    report_task = _assert_corrected_report(restarted_store, episode_id)
    assert retry is not None and retry.status == "succeeded"
    assert retry.parent_operation_id == root_operation_id
    assert retry.native_session_id == interrupted.native_session_id
    assert retry.stage_host == interrupted.stage_host
    assert retry.stage_root == interrupted.stage_root == str(root_stage)
    assert report_task.native_session_id == interrupted.native_session_id
    assert report_task.stage_host == interrupted.stage_host
    assert report_task.stage_root == interrupted.stage_root
    spawn = restarted_store.agent_command_by_key(episode_id, "acceptance-interrupt-spawn")
    assert spawn is not None and spawn.status == "ok"
    assert spawn.operation_id == root_operation_id
    assert spawn.exit_payload is not None
    spawn_result = spawn.exit_payload.get("result")
    assert isinstance(spawn_result, dict)
    assert spawn_result["worker_id"] == worker_route.worker_id
    spawn_events = [
        event
        for event in restarted_store.agent_task_events(root_operation_id)
        if event.command_id == spawn.command_id
    ]
    assert [event.command_phase for event in spawn_events] == ["start", "exit"]
    assert (
        restarted_store.agent_command_by_key(
            episode_id,
            "acceptance-finish-after-interrupt",
        )
        is not None
    )
    assert restarted_store.episode_report(episode_id) is not None


def test_acceptance_episode_exhausts_operational_invocations_then_reports(
    manifest,
    tmp_path,
) -> None:
    app = create_named_app(
        str(manifest.path),
        data_dir=tmp_path / "data",
        acceptance_agent=True,
    )
    project_id = app.state.default_project_id
    assert project_id is not None
    _add_worker_seat(
        app,
        node_id="exp/episode-exhaustion",
        title="Episode exhaustion probe",
        objective="Prove no worker is admitted after the operational ceiling is spent.",
    )
    store = app.state.background_tasks.store

    with TestClient(app) as client:
        started = _start_episode(
            client,
            project_id,
            invocation_ceiling=1,
            starting_instruction=ACCEPTANCE_EPISODE_EXHAUST_MARKER,
        )
        episode_id = str(started["episode_id"])
        episode = _wait_for_episode(
            client,
            project_id,
            episode_id,
            status="needs_action",
            ending="exhausted",
            report_ready=True,
        )

        budget = episode["budget"]
        assert isinstance(budget, dict)
        assert budget["invocation_ceiling"] == 1
        assert budget["invocations_used"] == 1
        assert budget["invocations_remaining"] == 0
        assert episode["can_reauthorize"] is True
        assert episode["report"] is not None
        preview = client.get(f"/api/projects/{project_id}/episodes/{episode_id}/report/content")
        assert preview.status_code == 200, preview.text
        assert "shared invocation pot" in preview.text

    tasks = store.auto_research_tasks(episode_id)
    assert [store.auto_research_invocation_role(task.operation_id) for task in tasks] == [
        "orchestrator"
    ]
    root = tasks[0]
    assert root.status == "succeeded"
    report_task = _assert_corrected_report(store, episode_id)
    assert report_task.native_session_id == root.native_session_id
    assert report_task.stage_root == root.stage_root
    exhaustion_probe = store.agent_command_by_key(episode_id, "acceptance-exhaustion-probe")
    assert exhaustion_probe is not None and exhaustion_probe.status == "invalid"
    assert store.episode_report(episode_id) is not None


def test_acceptance_exhausted_episode_reauthorization_creates_a_fresh_parent(
    manifest,
    tmp_path,
) -> None:
    app = create_named_app(
        str(manifest.path),
        data_dir=tmp_path / "data",
        acceptance_agent=True,
    )
    project_id = app.state.default_project_id
    assert project_id is not None
    _add_worker_seat(
        app,
        node_id="exp/episode-reauthorization",
        title="Episode reauthorization probe",
        objective="Prove reauthorization starts a fresh parent and native session.",
    )
    store = app.state.background_tasks.store

    with TestClient(app) as client:
        started = _start_episode(
            client,
            project_id,
            invocation_ceiling=1,
            starting_instruction=ACCEPTANCE_EPISODE_EXHAUST_MARKER,
        )
        old_episode_id = str(started["episode_id"])
        old_root_operation_id = str(started["root_operation_id"])
        old_episode = _wait_for_episode(
            client,
            project_id,
            old_episode_id,
            status="needs_action",
            ending="exhausted",
            report_ready=True,
        )

        reauthorized = client.post(
            f"/api/projects/{project_id}/episodes/{old_episode_id}/reauthorize",
            json={"invocation_ceiling": 1},
        )
        assert reauthorized.status_code == 202, reauthorized.text
        fresh_episode_id = reauthorized.json()["episode_id"]
        fresh_root_operation_id = reauthorized.json()["root_operation_id"]
        assert fresh_episode_id != old_episode_id
        assert fresh_root_operation_id != old_root_operation_id
        fresh_episode = _wait_for_episode(
            client,
            project_id,
            fresh_episode_id,
            status="needs_action",
            ending="exhausted",
            report_ready=True,
        )

        listed = client.get(f"/api/projects/{project_id}/episodes").json()
        old_after = next(item for item in listed if item["episode_id"] == old_episode_id)
        assert old_after == old_episode

    old_root = store.agent_task(old_root_operation_id)
    fresh_root = store.agent_task(fresh_root_operation_id)
    assert old_root is not None and fresh_root is not None
    assert old_root.parent_operation_id is None
    assert fresh_root.parent_operation_id is None
    assert old_root.episode_id == old_episode_id
    assert fresh_root.episode_id == fresh_episode_id
    assert fresh_root.native_session_id != old_root.native_session_id
    assert fresh_root.stage_root != old_root.stage_root
    assert fresh_episode["starting_instruction"] == ACCEPTANCE_EPISODE_EXHAUST_MARKER
    for episode_id in (old_episode_id, fresh_episode_id):
        meter = store.episode_budget_meter(episode_id)
        assert meter.invocation_ceiling == 1
        assert meter.invocations_used == 1
        assert store.episode_report(episode_id) is not None
        _assert_corrected_report(store, episode_id)


def test_acceptance_episode_stop_is_the_only_ending_without_a_report(
    manifest,
    tmp_path,
) -> None:
    app = create_named_app(
        str(manifest.path),
        data_dir=tmp_path / "data",
        acceptance_agent=True,
    )
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store

    with TestClient(app) as client:
        started = _start_episode(
            client,
            project_id,
            invocation_ceiling=10,
            starting_instruction=ACCEPTANCE_EPISODE_STOP_MARKER,
        )
        episode_id = str(started["episode_id"])
        root_operation_id = str(started["root_operation_id"])
        root_stage = _wait_for_task_stage(store, root_operation_id)
        active_path = root_stage / ".rcp-acceptance-campaign-active"
        release_path = root_stage / ".rcp-acceptance-campaign-release"
        _wait_for_path(active_path)
        root = store.agent_task(root_operation_id)
        assert root is not None and root.status == "running"

        try:
            stopped = client.post(f"/api/projects/{project_id}/episodes/{episode_id}/stop")
            assert stopped.status_code == 200, stopped.text
            assert stopped.json()["status"] == "stopping"
            assert stopped.json()["stop_requested_at"] is not None
            assert stopped.json()["ending"] is None
            assert stopped.json()["report"] is None
            current = store.agent_task(root_operation_id)
            assert current is not None and current.status == "running"
        finally:
            release_path.write_text("release after durable Stop\n", encoding="utf-8")

        episode = _wait_for_episode(
            client,
            project_id,
            episode_id,
            status="stopped",
            ending="stopped",
            report_ready=False,
        )
        budget = episode["budget"]
        assert isinstance(budget, dict)
        assert budget["invocations_used"] == 1
        assert episode["stop_requested_at"] == stopped.json()["stop_requested_at"]
        assert episode["wrapup_state"] == "skipped"
        assert episode["wrapup_error"] is None
        assert episode["report"] is None
        preview = client.get(f"/api/projects/{project_id}/episodes/{episode_id}/report/content")
        assert preview.status_code == 404

    root = store.agent_task(root_operation_id)
    assert root is not None and root.status == "succeeded"
    assert not [
        task
        for task in store.episode_tasks(episode_id, include_hidden=True)
        if task.kind == "episode_report"
    ]
    assert store.episode_report_attempts(episode_id) == []
    assert store.episode_report(episode_id) is None
    assert not active_path.exists()
    assert not release_path.exists()


def test_acceptance_episode_unrecoverable_failure_waits_then_reports_once(
    manifest,
    tmp_path,
) -> None:
    app = create_named_app(
        str(manifest.path),
        data_dir=tmp_path / "data",
        acceptance_agent=True,
    )
    project_id = app.state.default_project_id
    assert project_id is not None
    _add_worker_seat(
        app,
        node_id="exp/episode-terminal-failure",
        title="Episode terminal failure worker",
        objective="Settle admitted work before the partial report is exposed.",
    )
    background = app.state.background_tasks
    store = background.store
    root_release_path: Path | None = None
    worker_release_path: Path | None = None

    with TestClient(app) as client:
        started = _start_episode(
            client,
            project_id,
            invocation_ceiling=10,
            starting_instruction=ACCEPTANCE_EPISODE_FAIL_MARKER,
        )
        episode_id = str(started["episode_id"])
        root_operation_id = str(started["root_operation_id"])
        root_stage = _wait_for_task_stage(store, root_operation_id)
        root_active_path = root_stage / ".rcp-acceptance-campaign-failure-active"
        root_release_path = root_stage / ".rcp-acceptance-campaign-failure-release"
        _wait_for_path(root_active_path)

        worker_route, worker = _wait_for_child_work(store, episode_id)
        worker_stage = _wait_for_task_stage(store, worker.operation_id)
        worker_active_path = worker_stage / ".rcp-acceptance-campaign-worker-active"
        worker_release_path = worker_stage / ".rcp-acceptance-campaign-worker-release"
        _wait_for_path(worker_active_path)
        root = store.agent_task(root_operation_id)
        current_worker = store.agent_task(worker.operation_id)
        assert root is not None and root.status == "running"
        assert current_worker is not None and current_worker.status == "running"

        retained_body = "Retain this human guidance after the partial failure."
        retained = client.post(
            f"/api/projects/{project_id}/episodes/{episode_id}/messages",
            json={"body": retained_body},
        )
        assert retained.status_code == 201, retained.text
        assert retained.json()["delivered_at"] is None

        watcher = store.create_watchers(
            [
                GraphWatcherRecord(
                    watcher_id="acceptance-terminal-failure-watcher",
                    project_id=project_id,
                    origin_operation_id=root_operation_id,
                    origin_task_kind="auto_research",
                    graph_target=root.graph_target,
                    chat_id=root_operation_id,
                    episode_id=episode_id,
                    continuation=WatcherContinuation(
                        provider="codex",
                        run_on="laptop",
                        run_truth_scope=["repo-a"],
                        patch_kind="work",
                    ),
                    condition={
                        "node_id": "exp/episode-terminal-failure",
                        "status_in": ["running"],
                    },
                    armed_revision=1,
                    created_at=store.now(),
                )
            ]
        )[0]
        assert watcher.status == "active"

        try:
            root_release_path.write_text("release terminal failure\n", encoding="utf-8")
            wrapping = _wait_for_episode(
                client,
                project_id,
                episode_id,
                status="wrapping_up",
                ending="failed",
                report_ready=False,
            )
            assert wrapping["wrapup_state"] in {"not_started", "pending"}
            pending_reports = [
                task
                for task in store.episode_tasks(episode_id, include_hidden=True)
                if task.kind == "episode_report"
            ]
            assert len(pending_reports) <= 1
            assert all(task.status == "queued" for task in pending_reports), [
                (task.operation_id, task.status) for task in pending_reports
            ]
            assert store.episode_report_attempts(episode_id) == []
            current_worker = store.agent_task(worker.operation_id)
            assert current_worker is not None and current_worker.status == "running"
            stopped_watcher = store.watcher(watcher.watcher_id)
            assert stopped_watcher is not None
            assert stopped_watcher.status == "stopped"
            assert stopped_watcher.notified is True
            assert stopped_watcher.stopped_by == "loop"

            messages_before = store.auto_research_messages(episode_id)
            rejected_message = client.post(
                f"/api/projects/{project_id}/episodes/{episode_id}/messages",
                json={"body": "This must not become new terminal work."},
            )
            messages_after_terminal_attempt = store.auto_research_messages(episode_id)

            root = store.agent_task(root_operation_id)
            assert root is not None and root.status == "failed"
            denied_request = AutoResearchRunRequest.model_validate(root.request).model_copy(
                update={
                    "instruction": "This continuation must not be admitted.",
                    "session_id": root.native_session_id,
                }
            )
            meter_before = store.episode_budget_meter(episode_id)
            task_ids_before = [task.operation_id for task in store.auto_research_tasks(episode_id)]
            with pytest.raises(EpisodeNotRunning, match="not admitting new work"):
                start_auto_research_turn(
                    background,
                    episode_id,
                    denied_request,
                    parent_operation_id=root_operation_id,
                )
            assert store.episode_budget_meter(episode_id) == meter_before
            assert [
                task.operation_id for task in store.auto_research_tasks(episode_id)
            ] == task_ids_before

            worker_release_path.write_text("settle admitted worker\n", encoding="utf-8")
            episode = _wait_for_episode(
                client,
                project_id,
                episode_id,
                status="failed",
                ending="failed",
                report_ready=True,
            )
        finally:
            if root_active_path.exists():
                root_release_path.write_text("ensure root released\n", encoding="utf-8")
            if worker_active_path.exists():
                worker_release_path.write_text("ensure worker released\n", encoding="utf-8")

        budget = episode["budget"]
        assert isinstance(budget, dict)
        assert budget["invocation_ceiling"] == 10
        assert budget["invocations_used"] == 2
        assert episode["wrapup_state"] == "ready"
        preview = client.get(f"/api/projects/{project_id}/episodes/{episode_id}/report/content")
        assert preview.status_code == 200, preview.text
        assert "partial report" in preview.text

        listed_messages = client.get(f"/api/projects/{project_id}/episodes/{episode_id}/messages")
        assert listed_messages.status_code == 200, listed_messages.text
        listed_message_payload = listed_messages.json()

    tasks = store.auto_research_tasks(episode_id)
    roles = [store.auto_research_invocation_role(task.operation_id) for task in tasks]
    assert roles.count("orchestrator") == 1
    assert roles.count("worker") == 0
    child_routes = store.auto_research_child_works(episode_id)
    assert child_routes == [worker_route]
    settled_worker = store.agent_task(worker_route.current_operation_id)
    assert settled_worker is not None and settled_worker.status == "succeeded"
    root = store.agent_task(root_operation_id)
    assert root is not None and root.status == "failed"
    report_task = _assert_corrected_report(store, episode_id)
    assert report_task.status == "succeeded"
    assert report_task.native_session_id == root.native_session_id
    assert report_task.stage_host == root.stage_host
    assert report_task.stage_root == root.stage_root
    structural = [
        receipt
        for receipt in store.agent_task_receipts(root_operation_id)
        if receipt.category == "auto_research_orchestrator_failure"
    ]
    assert len(structural) == 1
    assert structural[0].payload["classification"] == "structural_unrecoverable"
    assert structural[0].payload["recoverable"] is False
    assert store.episode_report(episode_id) is not None
    assert rejected_message.status_code == 409
    assert messages_after_terminal_attempt == messages_before
    assert [item["body"] for item in listed_message_payload] == [retained_body]
    assert listed_message_payload[0]["delivered_at"] is None
    assert root_release_path is not None and not root_release_path.exists()
    assert worker_release_path is not None and not worker_release_path.exists()
