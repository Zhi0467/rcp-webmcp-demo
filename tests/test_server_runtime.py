from __future__ import annotations

import json
import stat

from fastapi.testclient import TestClient

from rcp.server_runtime import (
    SERVER_METADATA_SCHEMA_VERSION,
    ServerMetadata,
    published_server_metadata,
    read_server_metadata,
    remove_server_metadata,
)
from rcp.sources import indexer

from .helpers import create_named_app as create_app


def test_metadata_is_published_atomically_and_removed_by_its_owner(tmp_path, monkeypatch) -> None:
    metadata = ServerMetadata.create(
        tmp_path,
        host="127.0.0.1",
        port=8421,
        owner_kind="desktop",
    )
    replaced = []

    import rcp.server_runtime as runtime

    real_replace = runtime.os.replace

    def observed_replace(source, destination):
        replaced.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", observed_replace)

    with published_server_metadata(tmp_path, metadata):
        assert read_server_metadata(tmp_path) == metadata
        metadata_file = tmp_path / "rcp-server.json"
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SERVER_METADATA_SCHEMA_VERSION
        assert stat.S_IMODE(metadata_file.stat().st_mode) == 0o600

    assert len(replaced) == 1
    assert replaced[0][0].name.startswith(".rcp-server.json.")
    assert replaced[0][1] == tmp_path / "rcp-server.json"
    assert not (tmp_path / "rcp-server.json").exists()


def test_metadata_cleanup_never_removes_a_replacement(tmp_path) -> None:
    original = ServerMetadata.create(
        tmp_path,
        host="127.0.0.1",
        port=8421,
        owner_kind="desktop",
    )
    replacement = ServerMetadata.create(
        tmp_path,
        host="127.0.0.1",
        port=18421,
        owner_kind="cli",
    )

    with published_server_metadata(tmp_path, original):
        (tmp_path / "rcp-server.json").write_text(
            json.dumps(replacement.as_dict()), encoding="utf-8"
        )

        assert remove_server_metadata(tmp_path, instance_id=original.instance_id) is False
        assert read_server_metadata(tmp_path) == replacement
        assert remove_server_metadata(tmp_path, instance_id=replacement.instance_id) is True


def test_frozen_app_shutdown_cleans_metadata_before_the_outer_server_context_exits(
    tmp_path, monkeypatch
) -> None:
    import rcp.api.app as app_module

    monkeypatch.setattr(app_module.sys, "frozen", True, raising=False)
    data_dir = tmp_path / "data"
    metadata = ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=18421,
        owner_kind="desktop",
    )

    with published_server_metadata(data_dir, metadata):
        app = create_app(data_dir=data_dir, instance_metadata=metadata)
        with TestClient(app):
            assert read_server_metadata(data_dir) == metadata
        assert not (data_dir / "rcp-server.json").exists()


def test_source_app_shutdown_leaves_outer_supervisor_metadata_in_place(tmp_path) -> None:
    data_dir = tmp_path / "data"
    metadata = ServerMetadata.create(
        data_dir,
        host="127.0.0.1",
        port=18421,
        owner_kind="cli",
    )

    with published_server_metadata(data_dir, metadata):
        app = create_app(data_dir=data_dir, instance_metadata=metadata)
        with TestClient(app):
            pass
        assert read_server_metadata(data_dir) == metadata


def test_remote_parser_source_is_loaded_as_a_package_resource(monkeypatch) -> None:
    calls = []

    class Resource:
        def joinpath(self, name):
            calls.append(("joinpath", name))
            return self

        def read_text(self, *, encoding):
            calls.append(("read_text", encoding))
            return "def normalize_record():\n    pass\n"

    indexer._record_parsing_source.cache_clear()
    monkeypatch.setattr(
        indexer.importlib.resources,
        "files",
        lambda package: calls.append(("files", package)) or Resource(),
    )

    program = indexer._remote_program("print('driver')")

    assert program.startswith("def normalize_record()")
    assert program.endswith("print('driver')")
    assert calls == [
        ("files", "rcp.sources"),
        ("joinpath", "record_parsing.py"),
        ("read_text", "utf-8"),
    ]
    indexer._record_parsing_source.cache_clear()
