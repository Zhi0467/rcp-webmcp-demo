from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest

import rcp.runs.tasks.result_views as result_views
from rcp.runs.tasks.result_views import (
    discover_result_view,
    list_local_result_view_files,
    prepare_local_result_view_slot,
    require_result_view_changed,
)
from rcp.transport.run_stage import RemoteRunStage
from rcp.transport.state import StateUnavailable

VIEW_A = "a" * 24
VIEW_B = "b" * 24


def test_local_view_slots_keep_one_conversation_path_across_turns(tmp_path) -> None:
    stage = tmp_path / "rcp-run.chat"
    stage.mkdir()

    first = prepare_local_result_view_slot(stage, VIEW_A, reuse=False)
    first.joinpath("throughput.html").write_text("<h1>linear</h1>", encoding="utf-8")
    old_time = stage.stat().st_mtime - 3600
    os.utime(stage, (old_time, old_time))

    second_turn = prepare_local_result_view_slot(stage, VIEW_A, reuse=True)
    other_view = prepare_local_result_view_slot(stage, VIEW_B, reuse=False)

    assert first == second_turn == stage / "views" / VIEW_A
    assert other_view == stage / "views" / VIEW_B
    assert first != other_view
    assert stage.stat().st_mtime > old_time
    assert not (stage / "turns").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_local_result_view_slot(stage, VIEW_A, reuse=False)


def test_revision_accepts_atomic_replacement_at_same_stable_path(tmp_path) -> None:
    stage = tmp_path / "rcp-run.chat"
    stage.mkdir()
    slot = prepare_local_result_view_slot(stage, VIEW_A, reuse=False)
    target = slot / "throughput.html"
    target.write_text("<h1>linear</h1>", encoding="utf-8")
    before = discover_result_view(stage, None, VIEW_A)
    old_inode = target.stat().st_ino

    replacement = slot / ".replacement"
    replacement.write_text("<h1>log scale</h1>", encoding="utf-8")
    replacement_inode = replacement.stat().st_ino
    os.replace(replacement, target)
    after = discover_result_view(stage, None, VIEW_A, expected_name=before.name)

    assert target.stat().st_ino == replacement_inode
    assert target.stat().st_ino != old_inode
    require_result_view_changed(before, after)
    assert after.name == before.name
    assert after.data == b"<h1>log scale</h1>"


@pytest.mark.parametrize("unsafe_kind", ["symlink", "nested"])
def test_local_view_discovery_rejects_non_regular_entries(tmp_path, unsafe_kind: str) -> None:
    stage = tmp_path / "rcp-run.chat"
    stage.mkdir()
    slot = prepare_local_result_view_slot(stage, VIEW_A, reuse=False)
    target = slot / "view.html"
    if unsafe_kind == "symlink":
        outside = stage / "outside.html"
        outside.write_text("<h1>outside</h1>", encoding="utf-8")
        target.symlink_to(outside)
    else:
        target.mkdir()
        target.joinpath("nested.html").write_text("<h1>nested</h1>", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe entry"):
        list_local_result_view_files(stage, VIEW_A)


def test_local_view_discovery_rejects_oversized_and_non_html_output(tmp_path) -> None:
    stage = tmp_path / "rcp-run.chat"
    stage.mkdir()
    slot = prepare_local_result_view_slot(stage, VIEW_A, reuse=False)
    output = slot / "view.html"
    output.write_bytes(b"<h1>too large</h1>")

    with pytest.raises(ValueError, match="byte limit"):
        discover_result_view(stage, None, VIEW_A, max_bytes=4)

    output.rename(slot / "view.txt")
    with pytest.raises(ValueError, match="descriptively named"):
        discover_result_view(stage, None, VIEW_A)


def test_local_discovery_stops_after_the_second_entry_in_a_wide_slot(tmp_path, monkeypatch) -> None:
    stage = tmp_path / "rcp-run.chat"
    stage.mkdir()
    slot = prepare_local_result_view_slot(stage, VIEW_A, reuse=False)
    for index in range(256):
        slot.joinpath(f"view-{index:03}.html").write_text("<h1>wide</h1>", encoding="utf-8")

    observed = _count_fd_scandir_entries(monkeypatch)

    with pytest.raises(ValueError, match="exactly one"):
        discover_result_view(stage, None, VIEW_A)
    assert observed == [2]


def test_local_view_slot_rejects_symlinked_components(tmp_path) -> None:
    stage = tmp_path / "rcp-run.chat"
    stage.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (stage / "views").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="parent is unsafe"):
        prepare_local_result_view_slot(stage, VIEW_A, reuse=False)

    linked_stage = tmp_path / "linked-stage"
    linked_stage.symlink_to(stage, target_is_directory=True)
    with pytest.raises(StateUnavailable, match="conversation stage"):
        prepare_local_result_view_slot(linked_stage, VIEW_A, reuse=False)


def test_remote_view_operations_use_exact_stable_slot_and_roll_root_mtime(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "rcp-run.remote"
    (root / "workspace").mkdir(parents=True)
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    _run_remote_scripts_locally(stage, monkeypatch)
    old_time = root.stat().st_mtime - 3600
    os.utime(root, (old_time, old_time))

    slot = Path(str(stage.prepare_result_view_slot(VIEW_A, reuse=False)))
    payload = b"<html><body>first</body></html>"
    slot.joinpath("throughput.html").write_bytes(payload)

    assert stage.prepare_result_view_slot(VIEW_A, reuse=True) == PurePosixPath(str(slot))
    assert stage.prepare_result_view_slot(VIEW_B, reuse=False) != PurePosixPath(str(slot))
    assert stage.list_result_view_files(VIEW_A) == [("throughput.html", len(payload))]
    assert stage.read_result_view_bytes(VIEW_A, "throughput.html", max_bytes=1024) == payload
    assert slot.joinpath("throughput.html").read_bytes() == payload
    assert root.stat().st_mtime > old_time
    assert not (root / "workspace" / "turns").exists()


def test_remote_view_operations_distinguish_missing_unsafe_and_unavailable(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "rcp-run.remote"
    (root / "workspace").mkdir(parents=True)
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    _run_remote_scripts_locally(stage, monkeypatch)

    with pytest.raises(FileNotFoundError, match="slot is absent"):
        stage.list_result_view_files(VIEW_A)
    slot = Path(str(stage.prepare_result_view_slot(VIEW_A, reuse=False)))
    with pytest.raises(FileNotFoundError, match="file is absent"):
        stage.read_result_view_bytes(VIEW_A, "missing.html", max_bytes=1024)
    outside = root / "outside.html"
    outside.write_text("<h1>outside</h1>", encoding="utf-8")
    slot.joinpath("linked.html").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe entry"):
        stage.list_result_view_files(VIEW_A)
    with pytest.raises(ValueError, match="file is unsafe"):
        stage.read_result_view_bytes(VIEW_A, "linked.html", max_bytes=1024)
    assert outside.read_text(encoding="utf-8") == "<h1>outside</h1>"

    slot.joinpath("linked.html").unlink()
    slot.joinpath("large.html").write_bytes(b"too large")
    with pytest.raises(ValueError, match="exceeds its byte limit"):
        stage.read_result_view_bytes(VIEW_A, "large.html", max_bytes=4)

    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda _arguments: subprocess.CompletedProcess([], 255, "", "connection lost"),
    )
    with pytest.raises(StateUnavailable, match="connection lost"):
        stage.list_result_view_files(VIEW_A)

    monkeypatch.setattr(
        stage,
        "_ssh_bytes",
        lambda _arguments, *, input_data=None: subprocess.CompletedProcess(
            [], 255, b"", b"connection lost"
        ),
    )
    with pytest.raises(StateUnavailable, match="connection lost"):
        stage.read_result_view_bytes(VIEW_A, "large.html", max_bytes=1024)


def test_remote_view_listing_returns_after_two_wide_entries(tmp_path, monkeypatch) -> None:
    root = tmp_path / "rcp-run.remote"
    (root / "workspace").mkdir(parents=True)
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    _run_remote_scripts_locally(stage, monkeypatch)
    slot = Path(str(stage.prepare_result_view_slot(VIEW_A, reuse=False)))
    for index in range(256):
        slot.joinpath(f"view-{index:03}.html").write_text("<h1>wide</h1>", encoding="utf-8")

    scripts: list[str] = []
    run_remote = stage._ssh

    def capture(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        if len(arguments) >= 3 and arguments[:2] == ["python3", "-c"]:
            scripts.append(arguments[2])
        return run_remote(arguments)

    monkeypatch.setattr(stage, "_ssh", capture)

    assert len(stage.list_result_view_files(VIEW_A)) == 2
    assert len(scripts) == 1
    assert "os.listdir(slot_fd)" not in scripts[0]
    assert "with os.scandir(slot_fd)" in scripts[0]
    assert "if len(result)==2: break" in scripts[0]


def test_remote_view_traversal_rejects_replaced_workspace_or_views(tmp_path, monkeypatch) -> None:
    root = tmp_path / "rcp-run.remote"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "workspace").symlink_to(outside, target_is_directory=True)
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    _run_remote_scripts_locally(stage, monkeypatch)

    with pytest.raises(StateUnavailable, match="workspace"):
        stage.prepare_result_view_slot(VIEW_A, reuse=False)

    (root / "workspace").unlink()
    (root / "workspace").mkdir()
    (root / "workspace" / "views").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="slot is unsafe"):
        stage.prepare_result_view_slot(VIEW_A, reuse=False)


def test_view_id_is_exactly_lowercase_24_hex(tmp_path) -> None:
    stage = tmp_path / "rcp-run.chat"
    stage.mkdir()

    for value in ("a" * 23, "A" * 24, "g" * 24, "../" + "a" * 24):
        with pytest.raises(ValueError, match="24 lowercase hexadecimal"):
            prepare_local_result_view_slot(stage, value, reuse=False)


def _run_remote_scripts_locally(stage: RemoteRunStage, monkeypatch) -> None:
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=False,
        ),
    )
    monkeypatch.setattr(
        stage,
        "_ssh_bytes",
        lambda arguments, *, input_data=None: subprocess.run(
            arguments,
            capture_output=True,
            input=input_data,
            check=False,
        ),
    )


def _count_fd_scandir_entries(monkeypatch) -> list[int]:
    original_scandir = os.scandir
    observed = [0]

    class CountingIterator:
        def __init__(self, iterator) -> None:
            self._iterator = iterator

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self._iterator.close()

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self._iterator)
            observed[0] += 1
            return entry

    def counted(path):
        iterator = original_scandir(path)
        return CountingIterator(iterator) if isinstance(path, int) else iterator

    monkeypatch.setattr(result_views.os, "scandir", counted)
    return observed
