from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from rcp.storage import AgentTaskRecord

from .helpers import create_named_app as create_app


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_delete_project_route_refuses_active_task(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    card = app.state.catalog.card(project_id)
    assert card["can_delete"] is True
    assert card["delete_unavailable_reason"] is None
    now = app.state.background_tasks.store.now()
    app.state.background_tasks.store.create_agent_task(
        AgentTaskRecord(
            operation_id="active-delete-test",
            project_id=project_id,
            kind="seed",
            status="queued",
            request={},
            created_at=now,
            updated_at=now,
            status_message="queued",
        )
    )

    response = TestClient(app).delete(f"/api/projects/{project_id}")

    assert response.status_code == 409
    assert response.json()["detail"] == "Pause the active agent task before deleting this project."
    assert app.state.catalog.card(project_id)["id"] == project_id


def test_delete_project_route_disappears_after_restart_without_touching_repository(
    manifest, tmp_path
) -> None:
    data_dir = tmp_path / "data"
    app = create_app(str(manifest.path), data_dir=data_dir)
    project_id = app.state.default_project_id
    repository = Path(manifest.repository_map[manifest.state.repository].path)
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    before = _tree_digest(repository)

    stage = data_dir / "run-stage" / "failed-delete-test"
    stage.mkdir(parents=True)
    (stage / "patch.json").write_text("{}", encoding="utf-8")
    now = app.state.background_tasks.store.now()
    app.state.background_tasks.store.create_agent_task(
        AgentTaskRecord(
            operation_id="failed-delete-test",
            project_id=project_id,
            kind="seed",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
            stage_root=str(stage),
        )
    )
    app.state.catalog.open(project_id).paper.create()

    deleted = client.delete(f"/api/projects/{project_id}")

    assert deleted.status_code == 200
    assert deleted.json()["project_id"] == project_id
    assert all(card["id"] != project_id for card in client.get("/api/projects").json())
    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert client.get(f"/api/projects/{project_id}/cached").status_code == 404
    assert client.get(f"/api/projects/{project_id}/tasks").status_code == 404
    assert not stage.exists()
    assert _tree_digest(repository) == before

    restarted = TestClient(create_app(data_dir=data_dir))
    assert all(card["id"] != project_id for card in restarted.get("/api/projects").json())
    assert restarted.get(f"/api/projects/{project_id}").status_code == 404
