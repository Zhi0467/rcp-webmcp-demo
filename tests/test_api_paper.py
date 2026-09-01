from __future__ import annotations

from fastapi.testclient import TestClient

from .helpers import create_named_app


def test_paper_endpoints_cover_snapshot_create_save_and_sessions(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id

    initial = client.get(f"/api/projects/{project_id}/paper")
    assert initial.status_code == 200
    assert initial.json()["sync_state"] == "not_created"

    created = client.post(f"/api/projects/{project_id}/paper/create")
    assert created.status_code == 200
    assert created.json()["sync_state"] == "unsynced"

    saved = client.put(
        f"/api/projects/{project_id}/paper",
        json={
            "content": "# API introduction\n",
            "base_hash": created.json()["base_hash"],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["content"] == "# API introduction\n"
    assert saved.json()["sync_state"] == "synced"

    sessions = client.get(f"/api/projects/{project_id}/paper/sessions")
    assert sessions.status_code == 200
    assert sessions.json() == []

    removed_conflict_route = client.post(
        f"/api/projects/{project_id}/paper/conflict",
        json={"strategy": "use_canonical"},
    )
    assert removed_conflict_route.status_code == 405
