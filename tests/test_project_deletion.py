from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from rcp.agents import AgentLauncher
from rcp.projects import ProjectCatalog
from rcp.sources import project_cache_roots
from rcp.storage import AgentTaskRecord, AppStore
from rcp.transport import RemoteRunStage


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_catalog_delete_reclaims_only_app_owned_files_and_persists(manifest, tmp_path) -> None:
    data_dir = tmp_path / "app-data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    record = catalog.register(str(manifest.path), identity_action="adopted")
    repository = Path(manifest.repository_map[manifest.state.repository].path)
    before = _tree_digest(repository)

    stage = data_dir / "run-stage" / "saved-operation"
    stage.mkdir(parents=True)
    (stage / "patch.json").write_text("{}", encoding="utf-8")
    (stage / "patch.json").chmod(0o400)
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="saved-operation",
            project_id=record.project_id,
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
            stage_host="",
            stage_root=str(stage),
        )
    )
    display = catalog._cached_snapshot_path(record.project_id)
    display.parent.mkdir(parents=True)
    display.write_text("display", encoding="utf-8")
    paper = catalog._paper_snapshot_path(record.project_id)
    paper.parent.mkdir(parents=True)
    paper.write_text("draft", encoding="utf-8")
    catalog._services[record.project_id] = object()  # type: ignore[assignment]

    result = catalog.delete(record.project_id)

    assert result.project_id == record.project_id
    assert result.removed_stages == 1
    assert result.removed_display_snapshot is True
    assert result.removed_paper_snapshot is True
    assert not stage.exists()
    assert not display.exists()
    assert not paper.exists()
    assert record.project_id not in catalog._services
    assert _tree_digest(repository) == before

    reopened_store = AppStore(store.path)
    reopened_catalog = ProjectCatalog(data_dir, reopened_store, AgentLauncher())
    assert reopened_store.project(record.project_id) is None
    with pytest.raises(KeyError):
        reopened_catalog.card(record.project_id)


def test_catalog_delete_reports_remote_cleanup_failure_without_forgetting_project(
    manifest, tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "app-data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    record = catalog.register(str(manifest.path), identity_action="adopted")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="saved-remote-operation",
            project_id=record.project_id,
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
            stage_host="research-host",
            stage_root="/tmp/rcp-run.saved-remote-operation",
        )
    )
    monkeypatch.setattr(RemoteRunStage, "close", lambda self: False)

    with pytest.raises(RuntimeError, match="project was not deleted"):
        catalog.delete(record.project_id)

    assert store.project(record.project_id) is not None
    assert store.agent_task("saved-remote-operation") is not None


def test_catalog_delete_rejects_local_stage_outside_app_boundary(manifest, tmp_path) -> None:
    data_dir = tmp_path / "app-data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    record = catalog.register(str(manifest.path), identity_action="adopted")
    outside = tmp_path / "repository-owned"
    outside.mkdir()
    marker = outside / "must-remain"
    marker.write_text("source", encoding="utf-8")
    cache_root, _ = project_cache_roots(data_dir, record.project_id)
    cached = cache_root / "remote" / "saved.jsonl"
    cached.parent.mkdir(parents=True)
    cached.write_text("saved cache", encoding="utf-8")
    display = catalog._cached_snapshot_path(record.project_id)
    display.parent.mkdir(parents=True)
    display.write_text("saved display", encoding="utf-8")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="unsafe-stage",
            project_id=record.project_id,
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
            stage_host="",
            stage_root=str(outside),
        )
    )

    with pytest.raises(ValueError, match="outside the RCP staging boundary"):
        catalog.delete(record.project_id)

    assert marker.read_text(encoding="utf-8") == "source"
    assert cached.read_text(encoding="utf-8") == "saved cache"
    assert display.read_text(encoding="utf-8") == "saved display"
    assert os.path.exists(outside)
    assert store.project(record.project_id) is not None


def test_catalog_delete_validates_snapshot_before_removing_stage_or_cache(
    manifest, tmp_path
) -> None:
    data_dir = tmp_path / "app-data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    record = catalog.register(str(manifest.path), identity_action="adopted")
    stage = data_dir / "run-stage" / "saved-stage"
    stage.mkdir(parents=True)
    stage_marker = stage / "patch.json"
    stage_marker.write_text("saved stage", encoding="utf-8")
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="saved-stage",
            project_id=record.project_id,
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
            stage_root=str(stage),
        )
    )
    cache_root, _ = project_cache_roots(data_dir, record.project_id)
    cached = cache_root / "remote" / "saved.jsonl"
    cached.parent.mkdir(parents=True)
    cached.write_text("saved cache", encoding="utf-8")
    external = tmp_path / "external-display"
    external.write_text("outside", encoding="utf-8")
    display = catalog._cached_snapshot_path(record.project_id)
    display.parent.mkdir(parents=True)
    display.symlink_to(external)

    with pytest.raises(ValueError, match="non-file project display snapshot"):
        catalog.delete(record.project_id)

    assert stage_marker.read_text(encoding="utf-8") == "saved stage"
    assert cached.read_text(encoding="utf-8") == "saved cache"
    assert external.read_text(encoding="utf-8") == "outside"
    assert display.is_symlink()
    assert store.project(record.project_id) is not None


def test_deleted_tagged_project_reregisters_with_canonical_id_and_alias(
    manifest,
    tmp_path,
) -> None:
    data_dir = tmp_path / "app-data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    first = catalog.register(str(manifest.path), identity_action="adopted")
    aliases = store.project_aliases()
    assert len(aliases) == 1
    old_project_id = next(iter(aliases))

    catalog.delete(first.project_id)
    restored = catalog.register(str(manifest.path))

    assert restored.project_id == first.project_id
    assert restored.home_space_id == store.space_id
    assert store.resolve_project_id(old_project_id) == first.project_id
    assert catalog.card(old_project_id)["id"] == first.project_id
