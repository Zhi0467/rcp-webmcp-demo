from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from rcp.projects import TEAM_PROJECT_DELETE_UNAVAILABLE_REASON
from rcp.sources import project_cache_roots
from rcp.storage import AgentTaskRecord

from .test_project_membership import _create_project, _team_app

_EXPECTED_TEAM_DELETE_REASON = (
    "Team projects cannot be deleted here. A server operator must deprovision the "
    "managed checkout and Git deploy keys."
)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_team_card_and_api_refuse_before_entering_catalog_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, _store, _people, _acting = _team_app(tmp_path, members=1)
    project_id = _create_project(client, tmp_path / "managed-checkout")

    assert TEAM_PROJECT_DELETE_UNAVAILABLE_REASON == _EXPECTED_TEAM_DELETE_REASON
    [card] = client.get("/api/projects").json()
    assert card["id"] == project_id
    assert card["can_delete"] is False
    assert card["delete_unavailable_reason"] == _EXPECTED_TEAM_DELETE_REASON

    def unexpected_catalog_delete(_project_id: str):
        raise AssertionError("the team API entered catalog deletion")

    monkeypatch.setattr(app.state.catalog, "delete", unexpected_catalog_delete)

    refused = client.delete(f"/api/projects/{project_id}")

    assert refused.status_code == 409
    assert refused.json()["detail"] == _EXPECTED_TEAM_DELETE_REASON


def test_catalog_refuses_before_touching_any_team_project_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, store, _people, _acting = _team_app(tmp_path, members=1)
    repository = tmp_path / "managed-checkout"
    project_id = _create_project(client, repository)
    catalog = app.state.catalog
    data_dir = store.path.parent

    checkout_marker = repository / "source.txt"
    checkout_marker.write_text("managed checkout\n", encoding="utf-8")
    repository_before = _tree_digest(repository)

    deploy_key = tmp_path / "server-credentials" / project_id / "paper-repo" / "id_ed25519"
    deploy_key.parent.mkdir(parents=True)
    deploy_key.write_text("private key marker\n", encoding="utf-8")

    source_cache, _session_cache = project_cache_roots(data_dir, project_id)
    imported_source = source_cache / "remote" / "imported-source.jsonl"
    imported_source.parent.mkdir(parents=True)
    imported_source.write_text('{"source":"retained"}\n', encoding="utf-8")

    stage = data_dir / "run-stage" / "retained-team-task"
    stage.mkdir(parents=True)
    stage_marker = stage / "patch.json"
    stage_marker.write_text("{}\n", encoding="utf-8")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="retained-team-task",
            project_id=project_id,
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
            stage_root=str(stage),
        )
    )
    display = catalog._cached_snapshot_path(project_id)
    display.parent.mkdir(parents=True, exist_ok=True)
    display.write_text("display snapshot\n", encoding="utf-8")
    paper = catalog._paper_snapshot_path(project_id)
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("paper snapshot\n", encoding="utf-8")
    opened_service = catalog.open(project_id)

    entered_deletion = False

    def unexpected_deletion_preflight(_project_id: str):
        nonlocal entered_deletion
        entered_deletion = True
        raise AssertionError("team deletion reached project cleanup")

    monkeypatch.setattr(store, "project_deletion_stages", unexpected_deletion_preflight)

    with pytest.raises(
        ValueError,
        match=re.escape(_EXPECTED_TEAM_DELETE_REASON),
    ):
        catalog.delete(project_id)

    assert entered_deletion is False
    assert project_id not in catalog._deleting
    assert catalog.open(project_id) is opened_service
    assert store.project(project_id) is not None
    assert len(store.project_members(project_id)) == 1
    assert store.agent_task("retained-team-task") is not None
    assert _tree_digest(repository) == repository_before
    assert deploy_key.read_text(encoding="utf-8") == "private key marker\n"
    assert imported_source.read_text(encoding="utf-8") == '{"source":"retained"}\n'
    assert stage_marker.read_text(encoding="utf-8") == "{}\n"
    assert display.read_text(encoding="utf-8") == "display snapshot\n"
    assert paper.read_text(encoding="utf-8") == "paper snapshot\n"
