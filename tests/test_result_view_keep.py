from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

from rcp.transport.state import (
    LocalStateWorkspace,
    RunLockLease,
    SSHStateWorkspace,
    StateUnavailable,
    _advisory_lock_holder_arguments,
    _process_advisory_lock,
    _remote_script,
)


def _local_workspace(tmp_path: Path) -> tuple[LocalStateWorkspace, Path]:
    repository = tmp_path / "project"
    root = repository / ".research"
    root.mkdir(parents=True)
    return LocalStateWorkspace(root, str(root)), repository


def _holder_arguments(lock_path: Path) -> list[str]:
    return _advisory_lock_holder_arguments(lock_path, python_executable=sys.executable)


def test_local_keep_uses_repository_views_safe_name_and_no_overwrite(tmp_path) -> None:
    workspace, repository = _local_workspace(tmp_path)
    content = b"<!doctype html><title>pilot</title>"

    first = workspace.keep_result_view(
        source_name="../../Throughput Pilot Readout.HTML",
        project_name="VISTA Follow Up!",
        data=content,
        today=date(2026, 8, 12),
    )
    second = workspace.keep_result_view(
        source_name="Throughput Pilot Readout.html",
        project_name="VISTA Follow Up!",
        data=b"<!doctype html><title>second</title>",
        today=date(2026, 8, 12),
    )

    assert first == "throughput-pilot-readout-vista-follow-up-26-08-12.html"
    assert second == "throughput-pilot-readout-vista-follow-up-26-08-12-2.html"
    assert (repository / "views" / first).read_bytes() == content
    assert (repository / "views" / second).read_bytes().endswith(b"second</title>")
    assert workspace.read_kept_result_view(first) == content
    assert (repository / ".research" / "views").exists() is False


def test_local_keep_rejects_unsafe_views_and_bounded_read_rejects_links(tmp_path) -> None:
    workspace, repository = _local_workspace(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (repository / "views").symlink_to(unrelated, target_is_directory=True)

    with pytest.raises(ValueError, match="views path"):
        workspace.keep_result_view(
            source_name="report.html",
            project_name="project",
            data=b"<html></html>",
            today=date(2026, 8, 12),
        )

    assert list(unrelated.iterdir()) == []
    (repository / "views").unlink()
    name = workspace.keep_result_view(
        source_name="report.html",
        project_name="project",
        data=b"<html>bounded</html>",
        today=date(2026, 8, 12),
    )
    with pytest.raises(ValueError, match="read limit"):
        workspace.read_kept_result_view(name, max_bytes=4)
    with pytest.raises(ValueError, match="safe HTML base name"):
        workspace.read_kept_result_view(f"../{name}")
    with pytest.raises(FileNotFoundError):
        workspace.read_kept_result_view("missing-project-26-08-12.html")

    linked_name = "linked-project-26-08-12.html"
    (repository / "views" / linked_name).symlink_to(repository / "views" / name)
    with pytest.raises(ValueError, match="readable regular file"):
        workspace.read_kept_result_view(linked_name)


def test_local_keep_does_not_follow_a_symlinked_repository_root(tmp_path) -> None:
    repository = tmp_path / "actual-project"
    root = repository / ".research"
    root.mkdir(parents=True)
    linked_repository = tmp_path / "linked-project"
    linked_repository.symlink_to(repository, target_is_directory=True)
    workspace = LocalStateWorkspace(
        linked_repository / ".research",
        str(linked_repository / ".research"),
    )

    with pytest.raises(StateUnavailable, match="Repository root is unavailable"):
        workspace.keep_result_view(
            source_name="report.html",
            project_name="project",
            data=b"<html></html>",
            today=date(2026, 8, 12),
        )

    assert not (repository / "views").exists()


def test_lock_holder_keeps_outside_research_and_preserves_apply(tmp_path) -> None:
    repository = tmp_path / "remote-project"
    root = repository / ".research"
    apply_stage = root / ".publish" / "ordinary"
    keep_stage = root / ".publish" / "view-1-1"
    second_stage = root / ".publish" / "view-1-2"
    (apply_stage / "graph.json").parent.mkdir(parents=True)
    (apply_stage / "graph.json").write_text("graph\n", encoding="utf-8")
    (keep_stage / "content.html").parent.mkdir(parents=True)
    (keep_stage / "content.html").write_text("<html>first</html>", encoding="utf-8")
    (second_stage / "content.html").parent.mkdir(parents=True)
    (second_stage / "content.html").write_text("<html>second</html>", encoding="utf-8")
    base_name = "pilot-vista-26-08-12.html"

    with _process_advisory_lock(
        _holder_arguments(root / ".refresh.lock"),
        str(root / ".refresh.lock"),
    ) as lease:
        ordinary = lease._run_owned_command(
            {
                "op": "apply",
                "root": str(root),
                "stage": str(apply_stage),
                "paths": ["graph.json"],
            }
        )
        first = lease._run_owned_command(
            {
                "op": "keep-view",
                "root": str(root),
                "stage": str(keep_stage),
                "base_name": base_name,
            }
        )
        second = lease._run_owned_command(
            {
                "op": "keep-view",
                "root": str(root),
                "stage": str(second_stage),
                "base_name": base_name,
            }
        )

    assert ordinary == {"ok": True, "commit_status": None}
    assert (root / "graph.json").read_text(encoding="utf-8") == "graph\n"
    assert first == {"ok": True, "name": base_name}
    assert second == {"ok": True, "name": "pilot-vista-26-08-12-2.html"}
    assert (repository / "views" / base_name).read_text(encoding="utf-8") == ("<html>first</html>")
    assert (repository / "views" / second["name"]).read_text(encoding="utf-8") == (
        "<html>second</html>"
    )
    assert (root / "views").exists() is False


def test_lock_holder_rejects_symlink_views_without_touching_target(tmp_path) -> None:
    repository = tmp_path / "remote-project"
    root = repository / ".research"
    stage = root / ".publish" / "view-2-1"
    (stage / "content.html").parent.mkdir(parents=True)
    (stage / "content.html").write_text("<html>unsafe</html>", encoding="utf-8")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (repository / "views").symlink_to(unrelated, target_is_directory=True)

    with _process_advisory_lock(
        _holder_arguments(root / ".refresh.lock"),
        str(root / ".refresh.lock"),
    ) as lease:
        response = lease._run_owned_command(
            {
                "op": "keep-view",
                "root": str(root),
                "stage": str(stage),
                "base_name": "pilot-vista-26-08-12.html",
            }
        )

    assert response["ok"] is False
    assert "views path" in str(response["error"])
    assert list(unrelated.iterdir()) == []


def test_ssh_keep_stages_exact_file_and_uses_refresh_lock(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache" / ".research"
    cache_root.mkdir(parents=True)
    workspace = SSHStateWorkspace(cache_root, "research.example", "/srv/project")
    locks: list[str] = []
    commands: list[dict[str, object]] = []
    uploads: list[tuple[list[str], bytes]] = []

    def fake_ssh(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_rsync(arguments, **_kwargs):
        uploads.append((arguments, Path(arguments[-2]).read_bytes()))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        locks.append(str(path))

        def command(payload):
            commands.append(payload)
            return {"ok": True, "name": payload["base_name"]}

        yield RunLockLease(str(path), command=command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(subprocess, "run", fake_rsync)
    content = b"<!doctype html><title>remote</title>"

    chosen = workspace.keep_result_view(
        source_name="Remote Pilot.html",
        project_name="VISTA",
        data=content,
        today=date(2026, 8, 12),
    )

    assert chosen == "remote-pilot-vista-26-08-12.html"
    assert locks == ["/srv/project/.research/.refresh.lock"]
    assert uploads[0][1] == content
    assert uploads[0][0][-1].endswith(
        "/.research/.publish/" + Path(commands[0]["stage"]).name + "/"
    )
    assert commands == [
        {
            "op": "keep-view",
            "root": "/srv/project/.research",
            "stage": commands[0]["stage"],
            "base_name": chosen,
        }
    ]


def test_remote_read_script_is_bounded_and_never_follows_links(tmp_path) -> None:
    repository = tmp_path / "remote-project"
    views = repository / "views"
    views.mkdir(parents=True)
    name = "pilot-vista-26-08-12.html"
    (views / name).write_bytes(b"<html>kept</html>")

    def run_read(target_name: str, limit: int) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                _remote_script("remote_read_kept_view.py"),
                str(repository),
                target_name,
                str(limit),
            ],
            capture_output=True,
            check=False,
        )

    assert run_read(name, 1024).stdout == b"<html>kept</html>"
    assert run_read(name, 4).returncode == 45
    assert run_read("missing-vista-26-08-12.html", 1024).returncode == 44
    linked = "linked-vista-26-08-12.html"
    (views / linked).symlink_to(views / name)
    assert run_read(linked, 1024).returncode == 46


def test_ssh_read_distinguishes_missing_from_unavailable(tmp_path, monkeypatch) -> None:
    workspace = SSHStateWorkspace(tmp_path / ".research", "research.example", "/srv/project")
    name = "pilot-vista-26-08-12.html"
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, b"<html>kept</html>", b""),
            subprocess.CompletedProcess([], 44, b"", b""),
            subprocess.CompletedProcess([], 255, b"", b"host unavailable"),
        ]
    )
    calls: list[list[str]] = []

    def fake_ssh_bytes(arguments, **_kwargs):
        calls.append(arguments)
        return next(responses)

    monkeypatch.setattr(workspace, "_ssh_bytes", fake_ssh_bytes)

    assert workspace.read_kept_result_view(name) == b"<html>kept</html>"
    with pytest.raises(FileNotFoundError):
        workspace.read_kept_result_view(name)
    with pytest.raises(StateUnavailable, match="host unavailable"):
        workspace.read_kept_result_view(name)
    assert all(call[-3:] == ["/srv/project", name, str(16 * 1024 * 1024)] for call in calls)


def test_keep_does_not_modify_caller_source_file(tmp_path) -> None:
    workspace, _repository = _local_workspace(tmp_path)
    source = tmp_path / "agent-result.html"
    source.write_bytes(b"<html>source</html>")
    before = source.stat()

    workspace.keep_result_view(
        source_name=source.name,
        project_name="project",
        data=source.read_bytes(),
        today=date(2026, 8, 12),
    )

    after = source.stat()
    assert source.read_bytes() == b"<html>source</html>"
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
