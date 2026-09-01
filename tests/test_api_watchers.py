from __future__ import annotations

import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from rcp.storage import (
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    WatcherContinuation,
    WatcherRecord,
)
from rcp.watchers import WatcherCheckResult, WatchSpec

from .helpers import create_named_app


def test_degraded_watcher_can_be_checked_now_through_the_api(manifest, tmp_path: Path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    now = "2026-08-12T01:00:00+00:00"
    watcher = WatcherRecord(
        watcher_id="manual-api-check",
        project_id=project_id,
        origin_operation_id="manual-api-origin",
        origin_task_kind="node_chat",
        chat_id="manual-api-chat",
        node_id="exp-one",
        execution_host="gpu.example",
        check_command="squeue -h -j 4471 >/dev/null",
        log_path="/tmp/4471.log",
        cwd="/tmp",
        continuation=WatcherContinuation(
            provider="codex",
            run_on="laptop",
            patch_kind="work",
        ),
        created_at=now,
    )
    store.create_watchers([watcher])
    degraded = store.record_watcher_check(
        watcher.watcher_id,
        status="degraded",
        exit_code=255,
        error="transport unavailable",
        checked_at=now,
    )
    assert degraded.next_check_at is not None and degraded.next_check_at > now
    calls: list[tuple[str, str, float]] = []

    def recovered(spec: WatchSpec, host: str, timeout: float) -> WatcherCheckResult:
        calls.append((spec.check_command, host, timeout))
        return WatcherCheckResult(
            state="active",
            checked_at="2026-08-12T01:00:01+00:00",
            exit_code=1,
        )

    app.state.watcher_poller.check_runner = recovered
    response = TestClient(app).post(
        f"/api/projects/{project_id}/watchers/{watcher.watcher_id}/check"
    )

    assert response.status_code == 200
    assert calls == [
        (
            "squeue -h -j 4471 >/dev/null",
            "gpu.example",
            app.state.watcher_poller.timeout,
        )
    ]
    assert response.json()["status"] == "active"
    assert response.json()["consecutive_error_count"] == 0
    assert response.json()["last_error"] is None


def test_check_watcher_now_rejects_missing_graph_and_ineligible_records(
    manifest, tmp_path: Path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    continuation = WatcherContinuation(provider="codex", run_on="laptop", patch_kind="work")
    active = WatcherRecord(
        watcher_id="active-api-watcher",
        project_id=project_id,
        origin_operation_id="active-api-origin",
        origin_task_kind="node_chat",
        chat_id="active-api-chat",
        check_command="true",
        log_path="/tmp/active-api.log",
        cwd="/tmp",
        continuation=continuation,
        created_at="2026-08-12T01:00:00+00:00",
    )
    graph = GraphWatcherRecord(
        watcher_id="graph-api-watcher",
        project_id=project_id,
        origin_operation_id="graph-api-origin",
        origin_task_kind="node_chat",
        chat_id="graph-api-chat",
        continuation=continuation,
        condition=NodeStatusGraphCondition(node_id="exp-one", status_in=["resolved"]),
        armed_revision=0,
        created_at="2026-08-12T01:00:00+00:00",
    )
    store.create_watchers([active])
    store.create_watchers([graph])
    client = TestClient(app)

    missing_project = client.post(
        f"/api/projects/{uuid.uuid4()}/watchers/{active.watcher_id}/check"
    )
    missing_watcher = client.post(f"/api/projects/{project_id}/watchers/missing-api-watcher/check")
    active_response = client.post(f"/api/projects/{project_id}/watchers/{active.watcher_id}/check")
    graph_response = client.post(f"/api/projects/{project_id}/watchers/{graph.watcher_id}/check")

    assert missing_project.status_code == 404
    assert missing_watcher.status_code == 404
    assert active_response.status_code == 409
    assert active_response.json()["detail"] == (
        "Only a degraded watcher awaiting delivery can be checked now."
    )
    assert graph_response.status_code == 409
    assert graph_response.json()["detail"] == "Only an external watcher can be checked now."


def test_project_watchers_lists_and_stops_an_ordinary_watcher(manifest, tmp_path: Path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    watcher = WatcherRecord(
        watcher_id="ordinary-api-watcher",
        project_id=project_id,
        origin_operation_id="ordinary-api-origin",
        origin_task_kind="node_chat",
        chat_id="ordinary-api-chat",
        node_id="exp-one",
        check_command="true",
        log_path=str(tmp_path / "ordinary.log"),
        cwd=str(tmp_path),
        continuation=WatcherContinuation(
            provider="codex",
            run_on="laptop",
            patch_kind="work",
        ),
        created_at="2026-08-12T01:00:00+00:00",
    )
    store.create_watchers([watcher])
    client = TestClient(app)

    listed = client.get(f"/api/projects/{project_id}/watchers")
    assert listed.status_code == 200
    listed_payload = listed.json()
    assert isinstance(listed_payload, list)
    assert len(listed_payload) == 1
    assert listed_payload[0]["watcher_id"] == watcher.watcher_id
    assert listed_payload[0]["status"] == "active"

    stopped = client.post(f"/api/projects/{project_id}/watchers/{watcher.watcher_id}/stop")
    assert stopped.status_code == 200
    stopped_payload = stopped.json()
    assert isinstance(stopped_payload, dict)
    assert stopped_payload["watcher_id"] == watcher.watcher_id
    assert stopped_payload["status"] == "stopped"
    stored = store.watcher(watcher.watcher_id)
    assert stored is not None
    assert stored.status == "stopped"
