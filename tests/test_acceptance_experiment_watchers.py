from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from rcp.agents.acceptance import ACCEPTANCE_GENERIC_WATCHER_MARKER
from rcp.core.models import Patch
from rcp.storage import AgentTaskRecord, AppStore, WatcherRecord

from .helpers import TASK_SETTLE_TIMEOUT, append_fixture_patch, wait_for_task_response
from .helpers import create_named_app as create_app

_EXPERIMENT_ID = "exp/acceptance-loop"
_HYPOTHESIS_ID = "hyp/acceptance-sequence"


def _experiment_fixture_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added the hermetic bounded-loop acceptance fixture.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": _EXPERIMENT_ID,
                        "type": "experiment",
                        "title": "Acceptance loop",
                        "objective": "Finish two detached local fixture jobs.",
                        "completion_criteria": ["Both detached fixture jobs finish successfully."],
                        "invocation_ceiling": 1,
                    },
                    {
                        "id": _HYPOTHESIS_ID,
                        "type": "hypothesis",
                        "title": "Watcher delivery preserves the control sequence",
                        "statement": (
                            "A bounded watcher wake can inspect completed work without "
                            "bypassing human graph authority."
                        ),
                        "status": "proposed",
                    },
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": "edge/acceptance-tests",
                        "source": _EXPERIMENT_ID,
                        "target": _HYPOTHESIS_ID,
                        "relation": "tests",
                    }
                ],
            },
        ],
    )


def _wait_for_new_task(
    store: AppStore,
    project_id: str,
    previous_ids: set[str],
    predicate: Callable[[AgentTaskRecord], bool],
    *,
    timeout: float = TASK_SETTLE_TIMEOUT,
) -> AgentTaskRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = [
            task
            for task in store.agent_tasks(project_id)
            if task.operation_id not in previous_ids and predicate(task)
        ]
        if matches:
            assert len(matches) == 1
            return matches[0]
        time.sleep(0.02)
    raise AssertionError("watcher completion did not create its background task")


def _wait_for_watcher_status(
    store: AppStore,
    project_id: str,
    chat_id: str,
    status: str,
    *,
    timeout: float = 8.0,
) -> list[WatcherRecord]:
    """Wait for the lifespan-owned periodic poller to persist one status."""

    deadline = time.monotonic() + timeout
    latest: list[WatcherRecord] = []
    while time.monotonic() < deadline:
        latest = store.watchers(project_id, chat_id=chat_id)
        if len(latest) == 2 and all(record.status == status for record in latest):
            return latest
        time.sleep(0.02)
    raise AssertionError(f"periodic watcher poller did not persist {status}: {latest}")


def _receipt_categories(task: dict[str, object]) -> set[str]:
    return {str(item["category"]) for item in task["debug_receipts"]}


def _watcher_spec(record: WatcherRecord) -> tuple[str, str, str]:
    return record.check_command, record.log_path, record.cwd


def _poll_after_due(app, watchers: list[WatcherRecord]) -> None:
    # This hermetic fixture needs repeated short polls after jobs finish at
    # different instants. Production still reads the durable real clock.
    del watchers
    app.state.watcher_poller.clock = lambda: "2100-01-01T00:00:00+00:00"


def _wait_for_fixture_jobs(watchers: list[WatcherRecord], *, timeout: float = 30.0) -> None:
    """Let every detached fixture job finish before a poller can observe the group.

    Each job is its own process sleeping from its own start, so under load one
    ``.done`` file can land a poll interval ahead of the other. Delivering a wake
    for the watcher that genuinely finished first is correct — coalescing joins
    the watchers already complete at delivery, it does not wait for stragglers —
    so a test about one coalesced wake must settle the jobs before polling starts.
    """

    deadline = time.monotonic() + timeout
    pending = [Path(watcher.log_path).with_suffix(".done") for watcher in watchers]
    while time.monotonic() < deadline:
        if all(path.is_file() for path in pending):
            return
        time.sleep(0.02)
    missing = [str(path) for path in pending if not path.is_file()]
    raise AssertionError(f"detached fixture jobs did not finish within {timeout}s: {missing}")


def _assert_completed_job_artifacts(watchers: list[WatcherRecord]) -> None:
    for watcher in watchers:
        log_path = Path(watcher.log_path)
        assert log_path.read_text(encoding="utf-8").splitlines()[-1].endswith("completed")
        assert log_path.with_suffix(".status").read_text(encoding="utf-8") == "completed\n"
        assert log_path.with_suffix(".done").read_text(encoding="utf-8") == "done\n"


def test_s42_generic_watchers_persist_coalesce_and_never_change_the_graph(
    manifest, tmp_path: Path
) -> None:
    data_dir = tmp_path / "acceptance-data"
    app = create_app(str(manifest.path), data_dir=data_dir, acceptance_agent=True)
    client = TestClient(app)
    project_id = app.state.default_project_id
    assert project_id is not None
    service = app.state.service
    append_fixture_patch(service, _experiment_fixture_patch())
    baseline_patches = service.history.load_patches()
    chat_id = str(uuid.uuid4())

    try:
        health = client.get("/api/health")
        assert health.status_code == 200, health.text
        assert health.json()["agent_mode"] == "acceptance"
        assert app.state.agent_mode == "acceptance"

        started = client.post(
            f"/api/projects/{project_id}/tasks/project_chat",
            json={
                "chat_id": chat_id,
                "message": f"Launch the local fixture. {ACCEPTANCE_GENERIC_WATCHER_MARKER}",
                "mode": "work",
                "run_truth_scope": ["repo-a"],
            },
        )
        assert started.status_code == 202, started.text
        origin = wait_for_task_response(client, project_id, started.json()["operation_id"])
        assert origin["status"] == "succeeded", origin
        assert {"watcher_correction_requested", "watchers_armed"} <= _receipt_categories(origin)

        armed = app.state.background_tasks.store.watchers(project_id, chat_id=chat_id)
        assert len(armed) == 2
        assert {record.status for record in armed} == {"active"}
        assert len({_watcher_spec(record) for record in armed}) == 2
        assert all(record.continuation.patch_kind == "work" for record in armed)
        assert all(record.continuation.control_node_id is None for record in armed)
        assert service.history.load_patches() == baseline_patches
        assert [record.action for record in app.state.launcher.launch_records] == [
            "initial",
            "watch_correction",
        ]
        assert app.state.launcher.launch_records[-1].watcher_count == 2
    finally:
        client.close()
        app.state.background_tasks.shutdown()

    # Both jobs must be finished before the poller starts, or the group coalesces
    # a real subset and this test's single-wake assertion becomes load-dependent.
    _wait_for_fixture_jobs(armed)

    # Reopening the same data directory proves the active watcher ledger is durable.
    reopened = create_app(str(manifest.path), data_dir=data_dir, acceptance_agent=True)
    reopened.state.watcher_poller.interval = 0.05
    _poll_after_due(reopened, armed)
    reopened_store = reopened.state.background_tasks.store
    with TestClient(reopened) as reopened_client:
        persisted = reopened_store.watchers(project_id, chat_id=chat_id)
        assert [_watcher_spec(record) for record in persisted] == [
            _watcher_spec(record) for record in armed
        ]
        assert all(not record.notified for record in persisted)

        before_ids = {task.operation_id for task in reopened_store.agent_tasks(project_id)}
        wake = _wait_for_new_task(
            reopened_store,
            project_id,
            before_ids,
            lambda task: task.request.get("trigger") == "watcher",
        )
        wake_detail = wait_for_task_response(reopened_client, project_id, wake.operation_id)
        assert wake_detail["status"] == "succeeded", wake_detail
        assert wake.request["patch_kind"] == "work"
        assert set(wake.request["watcher_ids"]) == {record.watcher_id for record in persisted}
        assert all(record.log_path in wake.request["message"] for record in persisted)
        assert "watcher_notification" in _receipt_categories(wake_detail)
        assert [record.action for record in reopened.state.launcher.launch_records] == ["wake"]
        _assert_completed_job_artifacts(persisted)

        delivered = reopened_store.watchers(project_id, chat_id=chat_id)
        assert all(record.status == "completed" for record in delivered)
        assert all(record.notified for record in delivered)
        assert {record.notification_operation_id for record in delivered} == {wake.operation_id}
        assert reopened.state.service.history.load_patches() == baseline_patches

        task_count = len(reopened_store.agent_tasks(project_id))
        time.sleep(0.15)
        assert len(reopened_store.agent_tasks(project_id)) == task_count
        assert reopened.state.service.history.load_patches() == baseline_patches


def test_s41_ceiling_pauses_then_human_run_starts_a_new_episode_and_exits(
    manifest, tmp_path: Path
) -> None:
    data_dir = tmp_path / "acceptance-data"
    app = create_app(str(manifest.path), data_dir=data_dir, acceptance_agent=True)
    client = TestClient(app)
    project_id = app.state.default_project_id
    assert project_id is not None
    service = app.state.service
    append_fixture_patch(service, _experiment_fixture_patch())
    baseline_patches = service.history.load_patches()
    chat_id = str(uuid.uuid4())

    try:
        started = client.post(
            f"/api/projects/{project_id}/experiments/exp%2Facceptance-loop/run",
            json={"chat_id": chat_id, "run_truth_scope": ["repo-a"]},
        )
        assert started.status_code == 202, started.text
        initial_record = started.json()
        initial_request = initial_record["request"]
        initial_episode = initial_request["control_episode_id"]
        uuid.UUID(initial_episode)
        assert initial_request["trigger"] == "experiment_run"
        assert initial_request["patch_kind"] == "experiment_loop"
        assert initial_request["control_invocation"] == 1
        assert initial_request["control_invocation_ceiling"] == 1
        assert initial_request["watcher_ids"] == []

        initial = wait_for_task_response(client, project_id, initial_record["operation_id"])
        assert initial["status"] == "succeeded", initial
        assert {"watcher_correction_requested", "watchers_armed"} <= _receipt_categories(initial)
        assert service.history.state().nodes[_EXPERIMENT_ID].attempts == []
        assert service.history.load_patches() == baseline_patches

        armed = app.state.background_tasks.store.watchers(project_id, chat_id=chat_id)
        assert len(armed) == 2
        assert all(record.continuation.patch_kind == "experiment_loop" for record in armed)
        assert all(record.continuation.control_node_id == _EXPERIMENT_ID for record in armed)
        assert all(record.continuation.control_episode_id == initial_episode for record in armed)
        assert all(record.continuation.control_invocation == 1 for record in armed)
        assert [record.action for record in app.state.launcher.launch_records] == [
            "initial",
            "watch_correction",
        ]
        assert app.state.launcher.launch_records[-1].watcher_count == 2
    finally:
        client.close()
        app.state.background_tasks.shutdown()

    # Persist the active watchers across a process-shaped reopen before completion delivery.
    reopened = create_app(str(manifest.path), data_dir=data_dir, acceptance_agent=True)
    reopened.state.watcher_poller.interval = 0.05
    _poll_after_due(reopened, armed)
    reopened_store = reopened.state.background_tasks.store
    with TestClient(reopened) as reopened_client:
        persisted = reopened_store.watchers(project_id, chat_id=chat_id)
        assert len(persisted) == 2
        before_ceiling_ids = {task.operation_id for task in reopened_store.agent_tasks(project_id)}
        pending = _wait_for_watcher_status(
            reopened_store,
            project_id,
            chat_id,
            "completed",
        )
        _assert_completed_job_artifacts(pending)
        assert {
            task.operation_id for task in reopened_store.agent_tasks(project_id)
        } == before_ceiling_ids

        assert all(not record.notified for record in pending)
        control = reopened_client.get(f"/api/projects/{project_id}").json()["experiment_control"][
            _EXPERIMENT_ID
        ]
        assert control["paused"] is False
        assert control["ready"] is True
        assert control["invocations_used"] == 1
        assert control["invocations_remaining"] == 0
        assert reopened.state.service.history.load_patches() == baseline_patches

        reauthorized = reopened_client.post(
            f"/api/projects/{project_id}/experiments/exp%2Facceptance-loop/run",
            json={"chat_id": str(uuid.uuid4()), "run_truth_scope": ["repo-a"]},
        )
        assert reauthorized.status_code == 202, reauthorized.text
        reauthorized_record = reauthorized.json()
        request = reauthorized_record["request"]
        assert request["trigger"] == "experiment_run"
        assert request["patch_kind"] == "experiment_loop"
        assert request["control_invocation"] == 1
        assert request["control_invocation_ceiling"] == 1
        assert request["control_episode_id"] != initial_episode
        assert set(request["watcher_ids"]) == {record.watcher_id for record in pending}

        finished = wait_for_task_response(
            reopened_client,
            project_id,
            reauthorized_record["operation_id"],
        )
        assert finished["status"] == "succeeded", finished
        assert {"watcher_notification", "experiment_loop_exit"} <= _receipt_categories(finished)
        assert any(
            "reauthorized by human Run" in str(event["message"]) for event in finished["events"]
        )
        assert [record.action for record in reopened.state.launcher.launch_records] == ["wake"]

        delivered = reopened_store.watchers(project_id, chat_id=chat_id)
        assert all(record.notified for record in delivered)
        assert {record.notification_operation_id for record in delivered} == {
            reauthorized_record["operation_id"]
        }
        experiment = reopened.state.service.history.state().nodes[_EXPERIMENT_ID]
        assert experiment.status == "completed"
        assert experiment.attempts == []
        state = reopened.state.service.history.state()
        evidence = state.nodes["ev/acceptance-result"]
        assert evidence.type == "evidence"
        assert state.nodes[_HYPOTHESIS_ID].status == "proposed"
        assert {
            (
                edge.source,
                edge.target,
                edge.relation,
            )
            for edge in state.edges.values()
            if edge.id in {"edge/acceptance-produces", "edge/acceptance-supports"}
        } == {
            (_EXPERIMENT_ID, "ev/acceptance-result", "produces"),
            ("ev/acceptance-result", _HYPOTHESIS_ID, "supports"),
        }
        pending_proposals = [
            proposal for proposal in state.proposals.values() if proposal.status == "pending"
        ]
        assert [proposal.id for proposal in pending_proposals] == ["prop/acceptance-result"]
        patches = reopened.state.service.history.load_patches()
        assert len(patches) == len(baseline_patches) + 1
        assert patches[-1].kind == "experiment_loop"
        assert patches[-1].source_operation_id == reauthorized_record["operation_id"]

        task_count = len(reopened_store.agent_tasks(project_id))
        time.sleep(0.15)
        assert len(reopened_store.agent_tasks(project_id)) == task_count
        assert len(reopened.state.service.history.load_patches()) == len(patches)

        approved = reopened_client.post(
            f"/api/projects/{project_id}/sync",
            json={
                "base_revision": state.revision,
                "proposals": [
                    {
                        "proposal_id": "prop/acceptance-result",
                        "decision": "approved",
                    }
                ],
            },
        )
        assert approved.status_code == 200, approved.text
        approved_graph = approved.json()
        assert approved_graph["nodes"][_HYPOTHESIS_ID]["status"] == "supported"
        assert approved_graph["proposals"]["prop/acceptance-result"]["status"] == "approved"
        final_patches = reopened.state.service.history.load_patches()
        assert len(final_patches) == len(patches) + 1
        assert final_patches[-1].kind == "approval"
        assert final_patches[-1].author == "human"
