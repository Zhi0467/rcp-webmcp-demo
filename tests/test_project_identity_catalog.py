from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import pytest

from rcp.agents import AgentLauncher
from rcp.attachments import ChatAttachmentStore
from rcp.config import load_manifest
from rcp.history import HistoryManager
from rcp.history.manager import ProjectIdentityConflict
from rcp.projects import ProjectCatalog
from rcp.storage import AppStore, ProjectRecord


def _legacy_record(manifest, store: AppStore, project_id: str) -> ProjectRecord:
    repository = manifest.repository_map[manifest.state.repository]
    machine = manifest.machine_map[repository.machine]
    return store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            locator=str(manifest.path),
            name=manifest.name,
            state_location=(
                f"{machine.host}:{repository.path}/.research"
                if machine.host
                else str(manifest.research_dir)
            ),
            state_remote=bool(machine.host),
            added_at=store.now(),
        )
    )


def _write_legacy_display(path: Path, project_id: str) -> None:
    snapshot = {
        "id": project_id,
        "name": "Legacy",
        "revision": 0,
        "snapshot_freshness": "fresh",
        "last_remote_sync_at": None,
        "state_repository": "repo-a",
        "canonical_state": {"remote": False, "reachable": True, "error": None},
        "run_on": "laptop",
        "project_truth_scope": [],
        "default_run_truth_scope": [],
        "default_campaign_invocation_ceiling": 10,
        "repositories": [],
        "machines": [],
        "primary_question": None,
        "last_refresh_at": None,
        "counts": {},
        "coverage": {},
        "graph": {"revision": 0},
        "paper": {},
        "paper_coach": {},
        "agent_profiles": {},
        "provider_readiness": {},
        "provider_skill_inventories": {},
        "providers": {},
        "cache_metrics": {},
        "validation_messages": [],
    }
    envelope = {
        "schema_version": 2,
        "project_id": project_id,
        "canonical_patch_head": None,
        "snapshot": snapshot,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope), encoding="utf-8")


def test_pre_identity_display_cache_remains_readable_before_legacy_adoption(
    manifest,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    project_id = "legacy-project"
    _legacy_record(manifest, store, project_id)
    cache_path = catalog._cached_snapshot_path(project_id)
    _write_legacy_display(cache_path, project_id)

    status, cached = catalog.cached_snapshot_status(project_id)

    assert status == "valid"
    assert cached is not None
    assert cached["home_space_id"] is None
    assert cached["default_auto_research_invocation_ceiling"] == 10
    assert "default_campaign_invocation_ceiling" not in cached
    assert "home_space_id" not in json.loads(cache_path.read_text(encoding="utf-8"))["snapshot"]


def test_display_cache_from_before_campaign_budget_remains_readable(
    manifest,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    project_id = "pre-campaign-budget-project"
    _legacy_record(manifest, store, project_id)
    cache_path = catalog._cached_snapshot_path(project_id)
    _write_legacy_display(cache_path, project_id)
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    del envelope["snapshot"]["default_campaign_invocation_ceiling"]
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    status, cached = catalog.cached_snapshot_status(project_id)

    assert status == "valid"
    assert cached is not None
    assert cached["default_auto_research_invocation_ceiling"] == 10


def test_registered_legacy_project_auto_adopts_actual_id_once_and_rekeys_caches(
    manifest,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    old_project_id = "actual-stored-id-not-derived"
    _legacy_record(manifest, store, old_project_id)
    display_source = catalog._cached_snapshot_path_for_id(old_project_id)
    paper_source = catalog._paper_snapshot_path_for_id(old_project_id)
    _write_legacy_display(display_source, old_project_id)
    paper_source.parent.mkdir(parents=True, exist_ok=True)
    paper_source.write_text("legacy introduction", encoding="utf-8")

    adopted = catalog.register(str(manifest.path))
    reopened = catalog.register(str(manifest.path))

    assert adopted == reopened
    assert uuid.UUID(adopted.project_id).version == 4
    assert adopted.home_space_id == store.space_id
    assert store.resolve_project_id(old_project_id) == adopted.project_id
    assert store.project_aliases() == {old_project_id: adopted.project_id}
    patches = HistoryManager(load_manifest(manifest.path)).load_patches()
    assert len(patches) == 1
    assert patches[0].project_identity is not None
    assert patches[0].project_identity.action == "adopted"
    assert not display_source.exists()
    assert not paper_source.exists()
    cached = catalog.cached_snapshot(old_project_id)
    assert cached is not None
    assert cached["id"] == adopted.project_id
    assert cached["home_space_id"] == store.space_id
    assert catalog._paper_snapshot_path(adopted.project_id).read_text(encoding="utf-8") == (
        "legacy introduction"
    )
    assert catalog.open(old_project_id) is catalog.open(adopted.project_id)


def test_coherent_cold_open_replays_once_without_reloading_patch_bodies(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    registration_catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    registered = registration_catalog.register(
        str(manifest.path),
        identity_action="adopted",
    )
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    replay = HistoryManager._replay
    load_patches = HistoryManager.load_patches
    calls = {"replay": 0, "load_patches": 0}

    def counted_replay(self, *args, **kwargs):
        calls["replay"] += 1
        return replay(self, *args, **kwargs)

    def counted_load_patches(self):
        calls["load_patches"] += 1
        return load_patches(self)

    monkeypatch.setattr(HistoryManager, "_replay", counted_replay)
    monkeypatch.setattr(HistoryManager, "load_patches", counted_load_patches)

    catalog.open(registered.project_id)

    assert calls == {"replay": 1, "load_patches": 0}


def test_registration_retry_finishes_database_migration_without_second_identity(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    old_project_id = "legacy-retry-id"
    _legacy_record(manifest, store, old_project_id)
    migrate = store.migrate_project_identity
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated catalog migration failure")
        return migrate(*args, **kwargs)

    monkeypatch.setattr(store, "migrate_project_identity", fail_once)

    with pytest.raises(RuntimeError, match="simulated catalog migration failure"):
        catalog.register(str(manifest.path))

    patches_after_failure = HistoryManager(load_manifest(manifest.path)).load_patches()
    assert len(patches_after_failure) == 1
    assert store.project_by_locator(str(manifest.path)).project_id == old_project_id

    recovered = catalog.register(str(manifest.path))

    assert recovered.project_id == patches_after_failure[0].project_identity.project_id
    assert store.resolve_project_id(old_project_id) == recovered.project_id
    assert len(HistoryManager(load_manifest(manifest.path)).load_patches()) == 1


def test_registered_legacy_project_rekeys_unsent_attachment_metadata(
    manifest,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    old_project_id = "legacy-project-with-attachment"
    _legacy_record(manifest, store, old_project_id)
    chat_id = str(uuid.uuid4())
    client_id = str(uuid.uuid4())
    attachment_store = ChatAttachmentStore(data_dir / "chat-attachments")
    uploaded = attachment_store.add(
        project_id=old_project_id,
        chat_id=chat_id,
        client_id=client_id,
        filename="notes.txt",
        media_type="text/plain",
        source=io.BytesIO(b"keep me"),
    )

    adopted = catalog.register(str(manifest.path))

    attachment_store.remove(
        project_id=adopted.project_id,
        chat_id=chat_id,
        client_id=client_id,
        attachment_set_id=uploaded.attachment_set_id,
        attachment_id=uploaded.attachment.attachment_id,
    )
    assert not (data_dir / "chat-attachments" / uploaded.attachment_set_id).exists()


def test_registration_retry_finishes_display_cleanup_after_publish_crash(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    old_project_id = "legacy-display-cleanup-retry"
    _legacy_record(manifest, store, old_project_id)
    source = catalog._cached_snapshot_path_for_id(old_project_id)
    _write_legacy_display(source, old_project_id)
    original_unlink = Path.unlink
    interrupted = False

    def interrupt_legacy_cleanup(path: Path, *args, **kwargs) -> None:
        nonlocal interrupted
        if path == source and not interrupted:
            interrupted = True
            raise OSError("simulated interruption after display publish")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_legacy_cleanup)
    with pytest.raises(OSError, match="simulated interruption"):
        catalog.register(str(manifest.path))

    identity = HistoryManager(load_manifest(manifest.path)).project_identity()
    assert identity is not None
    destination = catalog._cached_snapshot_path_for_id(identity.project_id)
    assert source.exists()
    assert destination.exists()
    assert store.resolve_project_id(old_project_id) == identity.project_id

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = catalog.register(str(manifest.path))

    assert recovered.project_id == identity.project_id
    assert not source.exists()
    assert destination.exists()
    assert catalog.cached_snapshot(recovered.project_id) is not None
    assert len(HistoryManager(load_manifest(manifest.path)).load_patches()) == 1


def test_foreign_tagged_project_refuses_before_catalog_or_cache_creation(
    manifest,
    tmp_path,
) -> None:
    first_store = AppStore(tmp_path / "first" / "rcp.sqlite3")
    first_catalog = ProjectCatalog(tmp_path / "first", first_store, AgentLauncher())
    first_catalog.register(str(manifest.path), identity_action="adopted")
    second_data = tmp_path / "second"
    second_store = AppStore(second_data / "rcp.sqlite3")
    second_catalog = ProjectCatalog(second_data, second_store, AgentLauncher())

    with pytest.raises(ProjectIdentityConflict, match="belongs to space"):
        second_catalog.register(str(manifest.path), identity_action="adopted")

    assert second_store.projects() == []
    assert second_catalog._services == {}
    assert not (second_data / "project-snapshots").exists()
    assert not (second_data / "paper-snapshots").exists()


def test_renamed_and_relocated_project_keeps_canonical_id(manifest, tmp_path) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    original = catalog.register(str(manifest.path), identity_action="adopted")
    repository = Path(manifest.repository_map[manifest.state.repository].path)
    relocated = repository.with_name("repo-a-relocated")
    repository.rename(relocated)
    relocated_manifest = relocated / ".research" / "manifest.toml"
    content = relocated_manifest.read_text(encoding="utf-8")
    content = content.replace('name = "test-paper"', 'name = "Renamed paper"')
    content = content.replace(str(repository), str(relocated))
    relocated_manifest.write_text(content, encoding="utf-8")

    registered = catalog.register(str(relocated_manifest))

    assert registered.project_id == original.project_id
    assert registered.home_space_id == store.space_id
    assert registered.name == "Renamed paper"
    assert registered.locator == str(relocated_manifest)
    assert len(HistoryManager(load_manifest(relocated_manifest)).load_patches()) == 1


def test_cache_destination_conflict_prevents_database_migration_and_overwrite(
    manifest,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    old_project_id = "legacy-cache-conflict"
    _legacy_record(manifest, store, old_project_id)
    identity = HistoryManager(
        manifest,
        expected_space_id=store.space_id,
    ).claim_project_identity("adopted")
    source = catalog._cached_snapshot_path_for_id(old_project_id)
    destination = catalog._cached_snapshot_path_for_id(identity.project_id)
    _write_legacy_display(source, old_project_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("canonical-cache-must-remain", encoding="utf-8")

    with pytest.raises(ValueError, match="destination already exists"):
        catalog.register(str(manifest.path))

    assert source.exists()
    assert destination.read_text(encoding="utf-8") == "canonical-cache-must-remain"
    assert store.project_by_locator(str(manifest.path)).project_id == old_project_id
    assert store.project_aliases() == {}


@pytest.mark.parametrize(
    ("collision", "diagnostic"),
    [
        ("canonical", "registered canonical project"),
        ("foreign_alias", "belongs to canonical project"),
    ],
)
def test_open_refuses_manifest_name_owned_by_another_project(
    manifest,
    tmp_path,
    collision,
    diagnostic,
) -> None:
    data_dir = tmp_path / "data"
    store = AppStore(data_dir / "rcp.sqlite3")
    foreign_project_id = str(uuid.uuid4())
    legacy_name = foreign_project_id if collision == "canonical" else "foreign-project-alias"
    manifest_text = manifest.path.read_text(encoding="utf-8")
    assert 'name = "test-paper"' in manifest_text
    manifest.path.write_text(
        manifest_text.replace('name = "test-paper"', f'name = "{legacy_name}"'),
        encoding="utf-8",
    )
    catalog = ProjectCatalog(data_dir, store, AgentLauncher())
    target = catalog.register(str(manifest.path), identity_action="adopted")
    store.upsert_project(
        ProjectRecord(
            project_id=foreign_project_id,
            home_space_id=store.space_id,
            locator=str(tmp_path / "foreign" / "research.yaml"),
            name="Foreign project",
            state_location=str(tmp_path / "foreign" / ".research"),
            state_remote=False,
            added_at=store.now(),
        )
    )
    if collision == "foreign_alias":
        with store.connection() as connection:
            connection.execute(
                """
                INSERT INTO project_aliases(alias_id, canonical_project_id)
                VALUES (?, ?)
                """,
                (legacy_name, foreign_project_id),
            )

    with pytest.raises(ValueError, match=diagnostic):
        catalog.open(target.project_id)

    assert catalog.loaded_service(target.project_id) is None
    assert store.project(target.project_id) == target
    assert store.project(foreign_project_id) is not None
