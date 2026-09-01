from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
from pathlib import Path

import pytest

from rcp.config import MachineConfig
from rcp.history import HistoryManager
from rcp.paper import PaperService
from rcp.service import ProjectService, RunRequest
from rcp.sources import ImportedProviderSourceStore
from rcp.sources import imported as imported_module
from rcp.storage import AppStore
from rcp.transfer import TransferArchiveEntry

PROJECT_ID = "11111111-1111-4111-8111-111111111111"


def _capture(
    tmp_path: Path,
    *,
    provider: str = "codex",
    content: bytes = b'{"cwd":"/project","type":"assistant"}\n',
) -> tuple[Path, tuple[TransferArchiveEntry, ...]]:
    digest = hashlib.sha256(content).hexdigest()
    root = tmp_path / "capture"
    source = root / "provider-history" / provider / digest
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    return root, (
        TransferArchiveEntry(
            archive_path=f"provider-history/{provider}/{digest}",
            group="provider_history",
            sha256=digest,
            size_bytes=len(content),
        ),
    )


def test_imported_provider_sources_publish_exact_read_only_bytes_and_are_idempotent(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capture_root, entries = _capture(tmp_path)
    store = ImportedProviderSourceStore(data_dir, PROJECT_ID)

    published = store.publish(capture_root, entries)
    repeated = store.publish(capture_root, entries)

    assert repeated == published
    assert published.files[0].sha256 == entries[0].sha256
    assert published.payload_size_bytes == entries[0].size_bytes
    imported = store.root / "codex" / entries[0].sha256
    assert imported.read_bytes() == (capture_root / entries[0].archive_path).read_bytes()
    assert stat.S_IMODE(imported.stat().st_mode) == 0o400
    assert store.source_roots() == {"codex": [str(store.root / "codex")]}


@pytest.mark.parametrize("corruption", ["missing", "rewritten", "writable", "symlink"])
def test_imported_provider_source_corruption_fails_instead_of_being_omitted(
    tmp_path: Path,
    corruption: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capture_root, entries = _capture(tmp_path)
    store = ImportedProviderSourceStore(data_dir, PROJECT_ID)
    store.publish(capture_root, entries)
    imported = store.root / "codex" / entries[0].sha256
    if corruption == "missing":
        imported.unlink()
    elif corruption == "rewritten":
        imported.chmod(0o600)
        imported.write_bytes(b"changed")
        imported.chmod(0o400)
    elif corruption == "writable":
        imported.chmod(0o600)
    else:
        outside = tmp_path / "outside.jsonl"
        outside.write_bytes(b"outside")
        imported.unlink()
        imported.symlink_to(outside)

    with pytest.raises((OSError, ValueError)):
        store.source_roots()


def test_imported_provider_source_publication_failure_leaves_no_visible_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capture_root, entries = _capture(tmp_path)
    store = ImportedProviderSourceStore(data_dir, PROJECT_ID)

    def fail_fsync(_root: Path) -> None:
        raise OSError("injected publication fsync failure")

    monkeypatch.setattr(imported_module, "_fsync_tree", fail_fsync)

    with pytest.raises(OSError, match="injected publication fsync failure"):
        store.publish(capture_root, entries)

    assert not os.path.lexists(store.root)
    assert not list(store.project_root.glob(".provider-history-*"))


def test_imported_provider_source_equal_publication_race_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capture_root, entries = _capture(tmp_path)
    store = ImportedProviderSourceStore(data_dir, PROJECT_ID)

    def win_race(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        raise OSError(errno.EEXIST, "injected equal publication race")

    monkeypatch.setattr(imported_module.os, "rename", win_race)

    published = store.publish(capture_root, entries)

    assert published == store.inventory()
    assert not list(store.project_root.glob(".provider-history-*"))


@pytest.mark.parametrize(
    "target",
    ["collection", "project", "root", "provider", "manifest", "history"],
)
def test_imported_provider_source_permission_drift_fails_closed(
    tmp_path: Path,
    target: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capture_root, entries = _capture(tmp_path)
    store = ImportedProviderSourceStore(data_dir, PROJECT_ID)
    store.publish(capture_root, entries)
    paths = {
        "collection": store.project_root.parent,
        "project": store.project_root,
        "root": store.root,
        "provider": store.root / "codex",
        "manifest": store.root / "manifest.json",
        "history": store.root / "codex" / entries[0].sha256,
    }
    paths[target].chmod(0o750 if target not in {"manifest", "history"} else 0o440)

    with pytest.raises(ValueError, match="mode 0700|read-only regular file|manifest"):
        store.inventory()


@pytest.mark.parametrize("target", ["capture", "manifest", "history"])
def test_imported_provider_source_fifo_fails_without_blocking(
    tmp_path: Path,
    target: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capture_root, entries = _capture(tmp_path)
    store = ImportedProviderSourceStore(data_dir, PROJECT_ID)
    if target == "capture":
        path = capture_root / entries[0].archive_path
    else:
        store.publish(capture_root, entries)
        path = (
            store.root / "manifest.json"
            if target == "manifest"
            else store.root / "codex" / entries[0].sha256
        )
    path.unlink()
    os.mkfifo(path, 0o400)

    with pytest.raises(ValueError, match="regular file|manifest"):
        if target == "capture":
            store.publish(capture_root, entries)
        else:
            store.inventory()


def test_remote_seed_inventory_is_available_for_task_staging(
    manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capture_root, entries = _capture(tmp_path)
    ImportedProviderSourceStore(data_dir, PROJECT_ID).publish(capture_root, entries)
    service = ProjectService(
        manifest,
        HistoryManager(manifest),
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
        data_dir=data_dir,
        project_id=PROJECT_ID,
    )

    inventory = service.imported_source_inventory(
        "seed",
        MachineConfig(alias="server", host="rcp.example"),
    )

    assert inventory is not None
    assert inventory.files[0].sha256 == entries[0].sha256


def test_legacy_project_service_has_no_imported_provider_source_owner(
    manifest,
    tmp_path: Path,
) -> None:
    service = ProjectService(
        manifest,
        HistoryManager(manifest),
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
        data_dir=tmp_path / "data",
        project_id="legacy-project",
    )

    assert service.imported_sources is None


def test_seed_context_keeps_imported_sources_separate_from_native_provider_homes(
    manifest,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capture_root, entries = _capture(
        tmp_path,
        content=(
            b'{"type":"session_meta","payload":{"id":"imported",'
            + f'"cwd":"{manifest.repository_map["repo-a"].path}"'.encode()
            + b"}}\n"
        ),
    )
    imported = ImportedProviderSourceStore(data_dir, PROJECT_ID)
    imported.publish(capture_root, entries)
    service = ProjectService(
        manifest,
        HistoryManager(manifest),
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
        data_dir=data_dir,
        project_id=PROJECT_ID,
    )

    context = service.assemble_run(
        RunRequest(run_truth_scope=["repo-a"]),
        surface="refresh",
    )

    assert context.imported_source_roots == {"codex": [str(imported.root / "codex")]}
    assert str(imported.root) not in str(context.source_roots)
    assert context.all_source_roots()["codex"] == [
        *context.source_roots["codex"],
        str(imported.root / "codex"),
    ]
    assert context.prompt_payload()["imported_source_roots"] == context.imported_source_roots
    assert service.index_snapshot().sessions == []
