from __future__ import annotations

from contextlib import contextmanager

from rcp.paper import INTRODUCTION_TEMPLATE, PaperService
from rcp.storage import AppStore, ProjectRecord
from rcp.transport import StateUnavailable, StateWorkspace


class RecordingWorkspace(StateWorkspace):
    def __init__(self, root) -> None:
        super().__init__(root, "test-host:/canonical/.research")
        self.remote = True
        self.refreshes = 0
        self.transactions = 0
        self.published: list[list[str]] = []

    def refresh(self) -> bool:
        self.refreshes += 1
        return True

    def refresh_if_stale(self, max_age_seconds: float = 2.0) -> bool:
        del max_age_seconds
        return self.refresh()

    @contextmanager
    def transaction(self):
        self.transactions += 1
        yield

    def publish(self, relative_paths) -> None:
        self.published.append([str(path) for path in relative_paths])


class UnavailableWorkspace(RecordingWorkspace):
    def refresh_if_stale(self, max_age_seconds: float = 2.0) -> bool:
        del max_age_seconds
        raise StateUnavailable("offline")

    @contextmanager
    def transaction(self):
        self.transactions += 1
        raise StateUnavailable("offline")
        yield


def test_create_is_local_first_and_next_save_synchronizes(manifest, tmp_path) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    store = AppStore(tmp_path / "app.sqlite3")
    service = PaperService(manifest, store, workspace)

    created = service.create()

    assert created.sync_state == "unsynced"
    assert created.content == INTRODUCTION_TEMPLATE
    assert created.base_hash is None
    assert workspace.refreshes == 0
    assert workspace.transactions == 0
    assert workspace.published == []
    assert not (manifest.research_dir / "paper" / "introduction.md").exists()
    with store.connection() as connection:
        draft = connection.execute(
            "SELECT content, base_hash FROM paper_drafts WHERE project_id = ?",
            (manifest.name,),
        ).fetchone()
    assert draft is not None
    assert draft["content"] == INTRODUCTION_TEMPLATE
    assert draft["base_hash"] is None

    synchronized = service.save(created.content, created.base_hash)

    assert synchronized.sync_state == "synced"
    assert workspace.transactions == 1
    assert workspace.published == [["paper/introduction.md"]]
    assert (manifest.research_dir / "paper" / "introduction.md").read_text(
        encoding="utf-8"
    ) == INTRODUCTION_TEMPLATE


def test_create_preserves_synced_cached_canonical_without_workspace_io(manifest, tmp_path) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    canonical = manifest.research_dir / "paper" / "introduction.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("# Existing introduction\n", encoding="utf-8")
    service = PaperService(manifest, AppStore(tmp_path / "app.sqlite3"), workspace)

    created = service.create()

    assert created.sync_state == "synced"
    assert created.content == "# Existing introduction\n"
    assert created.base_hash is not None
    assert canonical.read_text(encoding="utf-8") == "# Existing introduction\n"
    assert workspace.refreshes == 0
    assert workspace.transactions == 0
    assert workspace.published == []


def test_cached_canonical_without_draft_is_unsynced_while_remote_is_unavailable(
    manifest, tmp_path
) -> None:
    canonical = manifest.research_dir / "paper" / "introduction.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("# Cached introduction\n", encoding="utf-8")
    service = PaperService(
        manifest,
        AppStore(tmp_path / "app.sqlite3"),
        UnavailableWorkspace(manifest.research_dir),
    )

    snapshot = service.snapshot()

    assert snapshot.sync_state == "unsynced"
    assert snapshot.content == "# Cached introduction\n"
    assert snapshot.canonical_available is False


def test_save_keeps_local_draft_when_workspace_is_unavailable(manifest, tmp_path) -> None:
    workspace = UnavailableWorkspace(manifest.research_dir)
    store = AppStore(tmp_path / "app.sqlite3")
    service = PaperService(manifest, store, workspace)
    created = service.create()

    unsynced = service.save("# Recover this local draft\n", created.base_hash)

    assert unsynced.sync_state == "unsynced"
    assert unsynced.content == "# Recover this local draft\n"
    with store.connection() as connection:
        draft = connection.execute(
            "SELECT content FROM paper_drafts WHERE project_id = ?",
            (manifest.name,),
        ).fetchone()
    assert draft is not None
    assert draft["content"] == "# Recover this local draft\n"


def test_template_is_created_once_and_remains_freeform(manifest, tmp_path) -> None:
    service = PaperService(manifest, AppStore(tmp_path / "app.sqlite3"))

    created = service.create()
    assert created.sync_state == "unsynced"
    assert created.content == INTRODUCTION_TEMPLATE
    assert service.create().content == INTRODUCTION_TEMPLATE

    synchronized = service.save(created.content, created.base_hash)
    assert synchronized.sync_state == "synced"

    changed = service.save("# My own structure\n\nNo enforced headings.\n", synchronized.base_hash)
    assert changed.sync_state == "synced"
    assert "What question we study" not in changed.content
    recreated = service.create()
    assert recreated.content == changed.content
    assert recreated.sync_state == "synced"


def test_external_change_preserves_draft_until_later_edit_repins(manifest, tmp_path) -> None:
    store = AppStore(tmp_path / "app.sqlite3")
    service = PaperService(manifest, store)
    created = service.create()
    synchronized = service.save(created.content, created.base_hash)
    canonical = manifest.research_dir / "paper" / "introduction.md"
    canonical.write_text("# Changed elsewhere\n", encoding="utf-8")

    behind = service.save("# Local human draft\n", synchronized.base_hash)
    assert behind.sync_state == "behind"
    assert behind.content == "# Local human draft\n"
    assert behind.incoming_content == "# Changed elsewhere\n"
    assert behind.base_hash == synchronized.base_hash
    assert behind.canonical_hash != synchronized.base_hash
    assert canonical.read_text(encoding="utf-8") == "# Changed elsewhere\n"
    with store.connection() as connection:
        draft = connection.execute(
            "SELECT content, base_hash, ancestor_content FROM paper_drafts WHERE project_id = ?",
            (manifest.name,),
        ).fetchone()
    assert draft is not None
    assert draft["content"] == "# Local human draft\n"
    assert draft["base_hash"] == synchronized.base_hash
    assert draft["ancestor_content"] == INTRODUCTION_TEMPLATE

    unchanged_retry = service.save(behind.content, behind.canonical_hash)
    assert unchanged_retry.sync_state == "behind"
    assert canonical.read_text(encoding="utf-8") == "# Changed elsewhere\n"

    repinned = service.save("# Local human draft\n\nOne later edit.\n", behind.canonical_hash)
    assert repinned.sync_state == "synced"
    assert repinned.incoming_content is None
    assert canonical.read_text(encoding="utf-8") == repinned.content
    with store.connection() as connection:
        draft = connection.execute(
            "SELECT base_hash, ancestor_content FROM paper_drafts WHERE project_id = ?",
            (manifest.name,),
        ).fetchone()
    assert draft is not None
    assert draft["base_hash"] == repinned.canonical_hash
    assert draft["ancestor_content"] == repinned.content


def test_snapshot_discovers_external_change_without_replacing_editor_content(
    manifest, tmp_path
) -> None:
    service = PaperService(manifest, AppStore(tmp_path / "app.sqlite3"))
    created = service.create()
    synchronized = service.save(created.content, created.base_hash)
    canonical = manifest.research_dir / "paper" / "introduction.md"
    canonical.write_text("# Incoming canonical\n", encoding="utf-8")

    behind = service.snapshot()

    assert behind.sync_state == "behind"
    assert behind.content == synchronized.content
    assert behind.incoming_content == "# Incoming canonical\n"


def test_legacy_named_draft_is_copied_to_stable_project_id(manifest, tmp_path) -> None:
    store = AppStore(tmp_path / "app.sqlite3")
    legacy = PaperService(manifest, store)
    created = legacy.create()
    synchronized = legacy.save(created.content, created.base_hash)
    legacy.save("# Legacy draft\n", synchronized.base_hash)
    stable_project_id = "11111111-1111-4111-8111-111111111111"
    store.upsert_project(
        ProjectRecord(
            project_id=stable_project_id,
            home_space_id=store.space_id,
            locator=str(manifest.path),
            name=manifest.name,
            state_location=str(manifest.research_dir),
            state_remote=False,
            added_at=store.now(),
        )
    )

    store.migrate_legacy_project_data(manifest.name, stable_project_id)

    with store.connection() as connection:
        copied = connection.execute(
            "SELECT content, ancestor_content FROM paper_drafts WHERE project_id = ?",
            (stable_project_id,),
        ).fetchone()
    assert copied is not None
    assert copied["content"] == "# Legacy draft\n"
    assert copied["ancestor_content"] == "# Legacy draft\n"
