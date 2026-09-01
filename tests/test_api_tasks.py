from __future__ import annotations

import asyncio
import threading
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from rcp.agents import AgentEvent
from rcp.artifacts import AgentArtifactDescriptor

from .helpers import append_fixture_patch, create_named_app, seed_patch


def _event_frame(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def _wait_for_run(
    client: TestClient,
    project_id: str,
    operation_id: str,
) -> dict[str, object]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(f"/api/projects/{project_id}/tasks/{operation_id}")
        assert response.status_code == 200
        record = response.json()
        if record["status"] not in {"queued", "running"}:
            return record
        time.sleep(0.01)
    raise AssertionError("background run did not finish")


@pytest.mark.parametrize("action", ["content", "download"])
def test_remote_artifact_read_does_not_stall_health(
    manifest, tmp_path, monkeypatch, action
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    descriptor = AgentArtifactDescriptor(
        artifact_id="a" * 24,
        name="plot.png",
        media_type="image/png",
    )
    data = b"\x89PNG\r\n\x1a\npreview"
    entered = threading.Event()
    release = threading.Event()

    def blocked_load(*_args):
        entered.set()
        assert release.wait(timeout=3)
        return descriptor, data

    monkeypatch.setattr("rcp.api.tasks._load_agent_artifact", blocked_load)
    project_id = app.state.default_project_id
    url = f"/api/projects/{project_id}/tasks/operation/artifacts/{descriptor.artifact_id}/{action}"

    async def drive_concurrently():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            artifact = asyncio.create_task(client.get(url))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            finally:
                release.set()
            return health, await artifact

    health, artifact = asyncio.run(drive_concurrently())

    assert health.status_code == 200
    assert artifact.status_code == 200


def test_task_list_and_detail_cover_every_agent_kind(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    project_id = app.state.default_project_id

    async def stream(_project_id, kind, _request, _execution):
        yield _event_frame(AgentEvent(event="message", text=f"{kind} answer"))
        if kind != "paper_coach":
            yield _event_frame(AgentEvent(event="message", text='{"applied_revision": 7}'))
        yield _event_frame(AgentEvent(event="done"))

    app.state.background_tasks.stream = stream
    client = TestClient(app)
    requests = {
        "seed": {},
        "refresh": {},
        "node_chat": {
            "node_id": "hyp/replanning-restores-plasticity",
            "chat_id": str(uuid.uuid4()),
            "message": "Explain this node.",
        },
        "project_chat": {
            "chat_id": str(uuid.uuid4()),
            "message": "Explain this project.",
        },
        "paper_coach": {"message": "Review this introduction."},
    }
    operation_ids: dict[str, str] = {}
    for kind, body in requests.items():
        response = client.post(
            f"/api/projects/{project_id}/tasks/{kind}",
            json=body,
        )
        assert response.status_code == 202
        operation_id = response.json()["operation_id"]
        operation_ids[kind] = operation_id
        assert _wait_for_run(client, project_id, operation_id)["status"] == "succeeded"

    listed = client.get(f"/api/projects/{project_id}/tasks")
    assert listed.status_code == 200
    assert {task["kind"] for task in listed.json()} == set(requests)
    for kind, operation_id in operation_ids.items():
        detail = client.get(f"/api/projects/{project_id}/tasks/{operation_id}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["kind"] == kind
        assert payload["result"] == {"messages": [f"{kind} answer"]}
        assert payload["events"]
        assert payload["debug_receipts"]
