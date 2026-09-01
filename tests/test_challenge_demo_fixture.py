from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from challenge.demo_fixture import (
    DEMO_ARTIFACT_NAME,
    DEMO_ARTIFACT_OPERATION_ID,
    DEMO_CHAT_ID,
    DEMO_DISPLAY_NAME,
    seed_demo_records,
)
from rcp.artifacts import descriptor_for, read_local_regular_file
from tests.helpers import create_named_app

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "demo-project"


def test_challenge_seed_is_idempotent_and_uses_ordinary_task_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "demo-project"
    shutil.copytree(FIXTURE, project)
    app = create_named_app(
        str(project / "state-repo" / ".research" / "manifest.toml"),
        data_dir=tmp_path / "data",
    )
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    stage_root = tmp_path / "fixture-stage"

    first = seed_demo_records(store, project_id, stage_root)
    second = seed_demo_records(store, project_id, stage_root)

    assert first == second
    assert store.local_owner is not None
    assert store.local_owner.display_name == DEMO_DISPLAY_NAME
    assert first.operation_id == DEMO_ARTIFACT_OPERATION_ID
    assert first.status == "succeeded"
    assert first.request["provider"] == "rcp-demo"
    assert first.request["session_id"] is None
    assert first.native_session_id is None
    assert first.result is not None
    descriptor = first.result["artifacts"][0]
    directory = stage_root / "turns" / first.operation_id / "artifacts"
    assert first.stage_root == str(stage_root)
    assert (
        descriptor["artifact_id"]
        == descriptor_for(
            first.operation_id,
            descriptor["name"],
        ).artifact_id
    )
    assert read_local_regular_file(directory, descriptor["name"], max_bytes=1_000_000)
    client = TestClient(app)
    content = client.get(
        f"/api/projects/{project_id}/tasks/{first.operation_id}/artifacts/"
        f"{descriptor['artifact_id']}/content"
    )
    assert content.status_code == 200, content.text
    assert "The measurement path is ready" in content.text
    identity = client.get("/api/identity")
    assert identity.status_code == 200, identity.text
    assert identity.json()["user"]["display_name"] == DEMO_DISPLAY_NAME
    transcript = app.state.service.chat_transcript(DEMO_CHAT_ID)
    assert transcript is not None
    assert transcript.node_id == "hyp/search-restores-future-learning"
    assert transcript.messages[-1].operation_id == first.operation_id


def test_challenge_seed_recovers_partial_artifact_and_running_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "demo-project"
    shutil.copytree(FIXTURE, project)
    app = create_named_app(
        str(project / "state-repo" / ".research" / "manifest.toml"),
        data_dir=tmp_path / "data",
    )
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    stage_root = tmp_path / "fixture-stage"
    artifact_directory = stage_root / "turns" / DEMO_ARTIFACT_OPERATION_ID / "artifacts"
    artifact_directory.mkdir(parents=True)
    artifact_path = artifact_directory / DEMO_ARTIFACT_NAME
    artifact_path.write_bytes(b"partial")
    complete = store.complete_agent_task

    def crash_before_completion(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(store, "complete_agent_task", crash_before_completion)
    with pytest.raises(RuntimeError, match="simulated crash"):
        seed_demo_records(store, project_id, stage_root)
    partial = store.agent_task(DEMO_ARTIFACT_OPERATION_ID)
    assert partial is not None
    assert partial.status == "running"
    assert artifact_path.read_bytes() != b"partial"

    monkeypatch.setattr(store, "complete_agent_task", complete)
    recovered = seed_demo_records(store, project_id, stage_root)

    assert recovered.status == "succeeded"
    assert recovered.result is not None
    assert recovered.result["artifacts"][0]["name"] == DEMO_ARTIFACT_NAME
