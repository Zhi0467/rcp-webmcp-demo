"""The scripts RCP ships to an execution machine, exercised as real subprocesses.

These ran only over SSH while they lived inside string literals, so nothing here
could be checked without a reachable host. They are modules now, so each one can
be run locally exactly the way the remote host runs it — ``python -c <source>``
with the same argv — and its guards can be driven directly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rcp.transport.remote_read_kept_view import MISSING, TOO_LARGE, UNSAFE
from rcp.transport.state import _remote_script


def run_script(name: str, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run a shipped script the way the execution machine runs it."""

    return subprocess.run(
        [sys.executable, "-c", _remote_script(name), *args],
        input=stdin,
        capture_output=True,
        text=True,
    )


def test_every_shipped_script_is_the_module_source() -> None:
    for name in (
        "remote_lock_holder.py",
        "remote_archive_research.py",
        "remote_read_kept_view.py",
        "remote_verify_imported_sources.py",
    ):
        source = _remote_script(name)
        on_disk = (Path(__file__).parent.parent / "src" / "rcp" / "transport" / name).read_text()
        assert source == on_disk
        compile(source, name, "exec")


class TestReadKeptView:
    def test_reads_a_kept_view(self, tmp_path: Path) -> None:
        views = tmp_path / "views"
        views.mkdir()
        (views / "result-demo-26-08-14.html").write_text("<h1>hello</h1>")

        result = run_script(
            "remote_read_kept_view.py",
            str(tmp_path),
            "result-demo-26-08-14.html",
            "1048576",
        )

        assert result.returncode == 0
        assert result.stdout == "<h1>hello</h1>"

    def test_absent_view_reports_missing(self, tmp_path: Path) -> None:
        (tmp_path / "views").mkdir()
        result = run_script(
            "remote_read_kept_view.py", str(tmp_path), "absent-view-26-08-14.html", "1048576"
        )
        assert result.returncode == MISSING

    def test_oversized_view_is_refused(self, tmp_path: Path) -> None:
        views = tmp_path / "views"
        views.mkdir()
        (views / "big-view-26-08-14.html").write_text("x" * 5000)

        result = run_script(
            "remote_read_kept_view.py", str(tmp_path), "big-view-26-08-14.html", "100"
        )

        assert result.returncode == TOO_LARGE

    def test_symlinked_view_is_refused(self, tmp_path: Path) -> None:
        views = tmp_path / "views"
        views.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("private")
        (views / "linked-view-26-08-14.html").symlink_to(secret)

        result = run_script(
            "remote_read_kept_view.py", str(tmp_path), "linked-view-26-08-14.html", "1048576"
        )

        assert result.returncode == UNSAFE
        assert "private" not in result.stdout

    @pytest.mark.parametrize(
        "name",
        ["../escape.html", "Result.html", "no-extension", "a" * 260 + ".html"],
    )
    def test_unsafe_names_are_refused(self, tmp_path: Path, name: str) -> None:
        (tmp_path / "views").mkdir()
        result = run_script("remote_read_kept_view.py", str(tmp_path), name, "1048576")
        assert result.returncode == UNSAFE

    def test_relative_repository_is_refused(self, tmp_path: Path) -> None:
        result = run_script(
            "remote_read_kept_view.py", "relative/path", "result-demo-26-08-14.html", "1048576"
        )
        assert result.returncode == UNSAFE


class TestArchiveResearch:
    branch_id = "11111111-1111-4111-8111-111111111111"
    merge_id = "a" * 64

    def _research(self, tmp_path: Path) -> Path:
        root = tmp_path / ".research"
        (root / "patches").mkdir(parents=True)
        (root / "manifest.toml").write_text("name = 'demo'\n")
        (root / "patches" / "000001.json").write_text("{}")
        return root

    def _branch(self, root: Path) -> Path:
        branch = root / "branches" / self.branch_id
        (branch / "patches").mkdir(parents=True)
        (branch / "merges").mkdir()
        (branch / "branch.json").write_text("{}")
        (branch / "graph.json").write_text('{"derived":true}')
        return branch

    def test_archives_the_directory(self, tmp_path: Path) -> None:
        root = self._research(tmp_path)

        result = run_script("remote_archive_research.py", str(root), "20260814T120000000000Z", "-")

        assert result.returncode == 0
        assert not root.exists()
        archive = Path(result.stdout.strip())
        assert archive.is_dir()
        assert (archive / "manifest.toml").is_file()

    def test_rejects_a_directory_that_is_not_dot_research(self, tmp_path: Path) -> None:
        other = tmp_path / "notresearch"
        other.mkdir()
        result = run_script("remote_archive_research.py", str(other), "20260814T120000000000Z", "-")
        assert result.returncode == 2

    def test_rejects_a_malformed_timestamp(self, tmp_path: Path) -> None:
        root = self._research(tmp_path)
        result = run_script("remote_archive_research.py", str(root), "not-a-timestamp", "-")
        assert result.returncode == 2
        assert root.exists()

    def test_fingerprint_mismatch_leaves_the_directory_alone(self, tmp_path: Path) -> None:
        root = self._research(tmp_path)

        result = run_script(
            "remote_archive_research.py", str(root), "20260814T120000000000Z", "0" * 64
        )

        assert result.returncode == 3
        assert root.is_dir()
        assert "changed since preflight" in result.stderr

    def test_matching_fingerprint_archives(self, tmp_path: Path) -> None:
        root = self._research(tmp_path)
        # Compute the fingerprint with the shipped module itself, the same way the
        # read-only preflight does before offering the archive.
        from rcp.transport.remote_archive_research import retained_history_fingerprint

        fingerprint = retained_history_fingerprint(root)

        result = run_script(
            "remote_archive_research.py", str(root), "20260814T120000000000Z", fingerprint
        )

        assert result.returncode == 0
        assert not root.exists()

    @pytest.mark.parametrize(
        "late_relative",
        [Path("patches/000001.json"), Path("merges") / f"{'a' * 64}.json"],
        ids=["branch-patch", "branch-merge-receipt"],
    )
    def test_late_branch_truth_invalidates_the_fingerprint(
        self,
        tmp_path: Path,
        late_relative: Path,
    ) -> None:
        root = self._research(tmp_path)
        branch = self._branch(root)
        from rcp.transport.remote_archive_research import retained_history_fingerprint

        fingerprint = retained_history_fingerprint(root)
        (branch / late_relative).write_text("late branch truth")

        result = run_script(
            "remote_archive_research.py", str(root), "20260814T120000000000Z", fingerprint
        )

        assert result.returncode == 3
        assert root.is_dir()

    @pytest.mark.parametrize(
        "unsafe_kind",
        [
            "branch-symlink",
            "malformed-branch",
            "patch-parent-symlink",
            "patch-symlink",
            "patch-directory",
            "bad-name",
        ],
    )
    def test_unsafe_branch_history_is_refused(
        self,
        tmp_path: Path,
        unsafe_kind: str,
    ) -> None:
        root = self._research(tmp_path)
        if unsafe_kind in {"branch-symlink", "malformed-branch"}:
            branches = root / "branches"
            branches.mkdir()
            if unsafe_kind == "branch-symlink":
                target = tmp_path / "branch-target"
                target.mkdir()
                (branches / self.branch_id).symlink_to(target, target_is_directory=True)
            else:
                (branches / "NOT-A-CANONICAL-UUID").mkdir()
        else:
            branch = self._branch(root)
            if unsafe_kind == "patch-parent-symlink":
                (branch / "patches").rmdir()
                target = tmp_path / "patches-target"
                target.mkdir()
                (branch / "patches").symlink_to(target, target_is_directory=True)
            elif unsafe_kind == "patch-symlink":
                target = tmp_path / "outside-patch"
                target.write_text("outside")
                (branch / "patches" / "000001.json").symlink_to(target)
            elif unsafe_kind == "patch-directory":
                (branch / "patches" / "000001.json").mkdir()
            else:
                (branch / "merges" / f"{'A' * 64}.json").write_text("{}")

        result = run_script(
            "remote_archive_research.py",
            str(root),
            "20260814T120000000000Z",
            "0" * 64,
        )

        assert result.returncode == 1
        assert root.is_dir()


class TestLockHolder:
    def test_acquires_applies_and_releases(self, tmp_path: Path) -> None:
        root = tmp_path / ".research"
        stage = root / ".publish" / "batch"
        stage.mkdir(parents=True)
        (stage / "graph.json").write_text('{"revision": 4}')
        lock_path = root / ".refresh.lock"

        command = json.dumps(
            {"op": "apply", "root": str(root), "stage": str(stage), "paths": ["graph.json"]}
        )
        result = run_script("remote_lock_holder.py", str(lock_path), stdin=command + "\n")

        assert result.returncode == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "acquired"
        assert json.loads(lines[1]) == {"ok": True, "commit_status": None}
        assert (root / "graph.json").read_text() == '{"revision": 4}'
        assert not stage.exists()

    def test_refuses_a_stage_outside_the_publish_directory(self, tmp_path: Path) -> None:
        root = tmp_path / ".research"
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir(parents=True)
        root.mkdir()
        (elsewhere / "graph.json").write_text("{}")
        lock_path = root / ".refresh.lock"

        command = json.dumps(
            {"op": "apply", "root": str(root), "stage": str(elsewhere), "paths": ["graph.json"]}
        )
        result = run_script("remote_lock_holder.py", str(lock_path), stdin=command + "\n")

        response = json.loads(result.stdout.splitlines()[1])
        assert response["ok"] is False
        assert "invalid canonical root" in response["error"]
        assert not (root / "graph.json").exists()

    def test_refuses_an_absolute_path_escape(self, tmp_path: Path) -> None:
        root = tmp_path / ".research"
        stage = root / ".publish" / "batch"
        stage.mkdir(parents=True)
        lock_path = root / ".refresh.lock"

        command = json.dumps(
            {"op": "apply", "root": str(root), "stage": str(stage), "paths": ["../escaped.json"]}
        )
        result = run_script("remote_lock_holder.py", str(lock_path), stdin=command + "\n")

        response = json.loads(result.stdout.splitlines()[1])
        assert response["ok"] is False
        assert "unsafe relative path" in response["error"]

    def test_exact_restore_is_idempotent_and_refuses_conflicting_remote_bytes(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / ".research"
        stage = root / ".publish" / "restore-1-1"
        stage.mkdir(parents=True)
        content = b"restored fact\n"
        (stage / "content.bin").write_bytes(content)
        lock_path = root / ".refresh.lock"
        command = {
            "op": "restore-exact",
            "root": str(root),
            "stage": str(stage),
            "path": "facts/result.txt",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
            "external": False,
        }

        first = run_script(
            "remote_lock_holder.py", str(lock_path), stdin=json.dumps(command) + "\n"
        )

        assert json.loads(first.stdout.splitlines()[1]) == {"ok": True}
        target = root / "facts" / "result.txt"
        assert target.read_bytes() == content

        stage.mkdir(parents=True)
        (stage / "content.bin").write_bytes(content)
        target.write_text("conflict\n")
        second = run_script(
            "remote_lock_holder.py", str(lock_path), stdin=json.dumps(command) + "\n"
        )

        response = json.loads(second.stdout.splitlines()[1])
        assert response["ok"] is False
        assert "conflicts with existing bytes" in response["error"]
        assert target.read_text() == "conflict\n"

    def test_empty_legacy_lock_directory_is_reclaimed(self, tmp_path: Path) -> None:
        root = tmp_path / ".research"
        stage = root / ".publish" / "batch"
        stage.mkdir(parents=True)
        lock_path = root / ".refresh.lock"
        lock_path.mkdir()

        result = run_script("remote_lock_holder.py", str(lock_path), stdin="")

        assert result.stdout.splitlines()[0] == "acquired"
        assert lock_path.is_file()

    def test_populated_legacy_lock_directory_is_preserved(self, tmp_path: Path) -> None:
        root = tmp_path / ".research"
        root.mkdir(parents=True)
        lock_path = root / ".refresh.lock"
        lock_path.mkdir()
        (lock_path / "owner.json").write_text("{}")

        result = run_script("remote_lock_holder.py", str(lock_path), stdin="")

        assert result.stdout.splitlines()[0] == "legacy-directory"
        assert lock_path.is_dir()
        assert (lock_path / "owner.json").is_file()

    def test_symlinked_lock_path_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / ".research"
        root.mkdir(parents=True)
        target = tmp_path / "elsewhere.lock"
        target.write_text("")
        lock_path = root / ".refresh.lock"
        lock_path.symlink_to(target)

        result = run_script("remote_lock_holder.py", str(lock_path), stdin="")

        assert result.stdout.splitlines()[0] == "unsafe-entry"

    def test_unsupported_command_is_reported_without_dropping_the_lock(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / ".research"
        root.mkdir(parents=True)
        lock_path = root / ".refresh.lock"

        result = run_script(
            "remote_lock_holder.py",
            str(lock_path),
            stdin=json.dumps({"op": "nonsense"}) + "\n" + json.dumps({"op": "nonsense"}) + "\n",
        )

        assert result.returncode == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "acquired"
        assert len(lines) == 3
        for line in lines[1:]:
            assert "unsupported lock-holder command" in json.loads(line)["error"]
