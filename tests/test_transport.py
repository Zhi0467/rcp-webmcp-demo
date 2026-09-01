from __future__ import annotations

import fcntl
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import pytest

from rcp.config import MachineConfig, RepositoryConfig, load_manifest
from rcp.core.models import Patch
from rcp.history import HistoryManager, PatchRejected
from rcp.limits import STATE_LOCK_POLL_INTERVAL_SECONDS
from rcp.paper import PaperService
from rcp.setup import ProjectSetupRequest, render_manifest
from rcp.storage import AppStore
from rcp.transport import (
    BatchPublishFailed,
    LocalStateWorkspace,
    RemoteRunStage,
    SSHStateWorkspace,
    StateUnavailable,
    StateWorkspace,
    prepare_state_workspace,
    repository_access,
)
from rcp.transport.state import (
    _REMOTE_PATCH_LOG_HEAD_SCRIPT,
    RunLockCancelled,
    RunLockLease,
    RunLockOwnershipLost,
    _advisory_lock_holder_arguments,
    _process_advisory_lock,
    _remote_advisory_lock_command,
)

from .helpers import seed_patch

_ARCHIVE_BRANCH_ID = "11111111-1111-4111-8111-111111111111"
_ARCHIVE_MERGE_ID = "a" * 64


def _retained_branch(root: Path) -> Path:
    branch = root / "branches" / _ARCHIVE_BRANCH_ID
    (branch / "patches").mkdir(parents=True)
    (branch / "merges").mkdir()
    (branch / "branch.json").write_text('{"branch_id":"fixture"}\n', encoding="utf-8")
    (branch / "graph.json").write_text('{"derived":true}\n', encoding="utf-8")
    return branch


class RecordingWorkspace(StateWorkspace):
    def __init__(self, root: Path) -> None:
        super().__init__(root, "test-host:/canonical/.research")
        self.remote = True
        self.transactions = 0
        self.refreshes = 0
        self.published: list[list[str]] = []
        self.committed_batches: list[str] = []
        self.committed_patches: list[str] = []

    @contextmanager
    def transaction(self):
        self.transactions += 1
        with super().transaction():
            yield

    def refresh(self):
        self.refreshes += 1
        return super().refresh()

    def publish(self, relative_paths):
        self.published.append([str(Path(item)) for item in relative_paths])

    def publish_committed_batch(self, relative_paths, batch_directory):
        self.committed_batches.append(str(Path(batch_directory)))
        self.publish(relative_paths)

    def publish_committed_patch(self, relative_paths, patch_path):
        self.committed_patches.append(str(Path(patch_path)))
        self.publish(relative_paths)


def _accept_question_patch() -> Patch:
    return Patch(
        kind="approval",
        author="human",
        summary="Accept the research question.",
        ops=[
            {
                "op": "set_standing",
                "node_id": "rq/learning-after-shift",
                "standing": "accepted",
            }
        ],
    )


def test_history_and_paper_publish_only_explicit_canonical_files(manifest, tmp_path) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.initialize()
    history.append(seed_patch())
    paper = PaperService(manifest, AppStore(tmp_path / "app.sqlite3"), workspace)
    created = paper.create()

    published = {path for batch in workspace.published for path in batch}
    assert "patches/000001.json" in published
    assert workspace.committed_patches == ["patches/000001.json"]
    assert "graph.json" in published
    assert "research.md" in published
    assert "manifest.toml" in published
    assert "paper/introduction.md" not in published
    assert created.sync_state == "unsynced"
    assert workspace.transactions == 2

    saved = paper.save(created.content, created.base_hash)
    published = {path for batch in workspace.published for path in batch}
    assert "paper/introduction.md" in published
    assert saved.sync_state == "synced"
    assert workspace.transactions == 3


def test_coherent_remote_initialization_reuses_refreshed_snapshot_without_publish(
    manifest,
) -> None:
    HistoryManager(manifest).initialize()
    workspace = RecordingWorkspace(manifest.research_dir)
    workspace.refresh()

    result = HistoryManager(load_manifest(manifest.path), workspace).initialize()

    assert result.state.revision == 0
    assert workspace.refreshes == 1
    assert workspace.transactions == 0
    assert workspace.published == []


def test_remote_initialization_repairs_and_publishes_mismatched_outputs(manifest) -> None:
    HistoryManager(manifest).initialize()
    (manifest.research_dir / "graph.json").write_text("{}\n", encoding="utf-8")
    workspace = RecordingWorkspace(manifest.research_dir)
    workspace.refresh()

    result = HistoryManager(load_manifest(manifest.path), workspace).initialize()

    assert result.state.revision == 0
    assert workspace.refreshes == 1
    assert workspace.transactions == 1
    assert workspace.published == [
        [
            "graph.json",
            "glossary.json",
            "proposals.json",
            "coverage.json",
            "cursors.json",
            "scope-base.json",
            "research.md",
            "manifest.toml",
        ]
    ]
    graph = json.loads((manifest.research_dir / "graph.json").read_text(encoding="utf-8"))
    assert graph["project_truth_scope"] == ["repo-a", "repo-b"]


def test_shared_snapshot_lock_keeps_refresh_behind_single_patch_publication(
    manifest, monkeypatch
) -> None:
    HistoryManager(manifest).append(seed_patch())
    root = manifest.research_dir
    writer_workspace = StateWorkspace(root, "test-host:/canonical/.research")
    reader_workspace = StateWorkspace(root, "test-host:/canonical/.research")
    writer_workspace.remote = True
    reader_workspace.remote = True
    assert writer_workspace.snapshot_lock is reader_workspace.snapshot_lock

    remote_patches = {"patches/000001.json"}
    publish_entered = threading.Barrier(2)
    publish_release = threading.Barrier(2)
    refresh_probed = threading.Barrier(2)
    refresh_continue = threading.Barrier(2)
    refresh_invoked = threading.Barrier(2)
    reader_lock_probe: list[bool] = []

    def publish_patch(_relative_paths, patch_path) -> None:
        publish_entered.wait(timeout=5)
        publish_release.wait(timeout=5)
        remote_patches.add(Path(patch_path).as_posix())

    def mirror_remote_patches() -> bool:
        for patch_path in (root / "patches").glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"):
            if patch_path.relative_to(root).as_posix() not in remote_patches:
                patch_path.unlink()
        return True

    original_refresh = reader_workspace.refresh

    def refresh_like_chat_read() -> bool:
        acquired = reader_workspace.snapshot_lock.acquire(blocking=False)
        reader_lock_probe.append(acquired)
        if acquired:
            reader_workspace.snapshot_lock.release()
        refresh_probed.wait(timeout=5)
        refresh_continue.wait(timeout=5)
        refresh_invoked.wait(timeout=5)
        return original_refresh()

    monkeypatch.setattr(writer_workspace, "publish_committed_patch", publish_patch)
    monkeypatch.setattr(reader_workspace, "_refresh_snapshot", mirror_remote_patches)
    monkeypatch.setattr(reader_workspace, "refresh", refresh_like_chat_read)
    writer = HistoryManager(load_manifest(manifest.path), writer_workspace)
    reader = HistoryManager(load_manifest(manifest.path), reader_workspace)

    with ThreadPoolExecutor(max_workers=2) as pool:
        append_future = pool.submit(
            writer.append,
            _accept_question_patch(),
            expected_revision=1,
        )
        publish_entered.wait(timeout=5)
        refresh_future = pool.submit(reader_workspace.refresh)
        refresh_probed.wait(timeout=5)
        refresh_continue.wait(timeout=5)
        refresh_invoked.wait(timeout=5)
        publish_release.wait(timeout=5)

        appended, _ = append_future.result(timeout=5)
        assert refresh_future.result(timeout=5) is True

    assert reader_lock_probe == [False]
    assert appended.revision == 2
    assert (root / "patches" / "000002.json").is_file()
    assert reader.materialize(write_outputs=False).state.revision == 2


def test_shared_snapshot_lock_keeps_atomic_transition_out_of_refresh_and_replay(
    manifest, monkeypatch
) -> None:
    HistoryManager(manifest).append(seed_patch())
    root = manifest.research_dir
    reader_workspace = StateWorkspace(root, "test-host:/canonical/.research")
    writer_workspace = StateWorkspace(root, "test-host:/canonical/.research")
    reader_workspace.remote = True
    writer_workspace.remote = True
    reader = HistoryManager(load_manifest(manifest.path), reader_workspace)
    writer = HistoryManager(load_manifest(manifest.path), writer_workspace)

    refresh_entered = threading.Barrier(2)
    refresh_release = threading.Barrier(2)
    writer_probed = threading.Barrier(2)
    writer_continue = threading.Barrier(2)
    replay_entered = threading.Barrier(2)
    replay_release = threading.Barrier(2)
    writer_lock_probe: list[bool] = []
    published_transitions: list[str] = []

    def paused_refresh() -> bool:
        refresh_entered.wait(timeout=5)
        refresh_release.wait(timeout=5)
        return True

    real_replay = reader._replay

    def paused_replay(pending_patch_paths=None):
        replay_entered.wait(timeout=5)
        replay_release.wait(timeout=5)
        return real_replay(pending_patch_paths)

    def append_after_probe():
        acquired = writer_workspace.snapshot_lock.acquire(blocking=False)
        writer_lock_probe.append(acquired)
        if acquired:
            writer_workspace.snapshot_lock.release()
        writer_probed.wait(timeout=5)
        writer_continue.wait(timeout=5)
        return writer.append_batch([_accept_question_patch()], expected_revision=1)

    def record_transition_publish(_relative_paths, patch_path) -> None:
        published_transitions.append(Path(patch_path).as_posix())

    monkeypatch.setattr(reader_workspace, "_refresh_snapshot", paused_refresh)
    monkeypatch.setattr(reader, "_replay", paused_replay)
    monkeypatch.setattr(writer_workspace, "publish_committed_patch", record_transition_publish)

    with ThreadPoolExecutor(max_workers=2) as pool:
        read_future = pool.submit(reader.current_materialization)
        refresh_entered.wait(timeout=5)
        append_future = pool.submit(append_after_probe)
        writer_probed.wait(timeout=5)
        writer_continue.wait(timeout=5)
        refresh_release.wait(timeout=5)
        replay_entered.wait(timeout=5)
        transition_existed_during_replay = (root / "patches" / "000002.json").exists()
        replay_release.wait(timeout=5)

        read_result = read_future.result(timeout=5)
        prepared, appended_result = append_future.result(timeout=5)

    assert writer_lock_probe == [False]
    assert transition_existed_during_replay is False
    assert read_result.state.revision == 1
    assert [patch.revision for patch in prepared] == [2]
    assert appended_result.state.revision == 2
    assert published_transitions == ["patches/000002.json"]


def test_local_graph_run_waits_then_acquires_and_reports_once(tmp_path) -> None:
    workspace = LocalStateWorkspace(tmp_path / ".research", str(tmp_path))
    waiting = threading.Event()
    acquired = threading.Event()
    messages: list[str] = []

    def contend() -> None:
        def on_wait(message: str) -> None:
            messages.append(message)
            waiting.set()

        with workspace.run_lock(on_wait=on_wait) as lease:
            assert isinstance(lease, RunLockLease)
            lease.assert_owned()
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with workspace.run_lock():
            future = pool.submit(contend)
            assert waiting.wait(timeout=5)
            assert acquired.is_set() is False
        future.result(timeout=5)

    assert acquired.is_set() is True
    assert messages == ["Waiting for another graph-writing run to release canonical state."]


def test_local_graph_run_wait_can_be_cancelled(tmp_path) -> None:
    workspace = LocalStateWorkspace(tmp_path / ".research", str(tmp_path))
    waiting = threading.Event()
    cancellation = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as pool, workspace.run_lock():
        future = pool.submit(
            lambda: _enter_run_lock(
                workspace,
                on_wait=lambda _message: waiting.set(),
                cancelled=cancellation.is_set,
            )
        )
        assert waiting.wait(timeout=5)
        cancellation.set()
        with pytest.raises(RunLockCancelled, match="cancelled while waiting"):
            future.result(timeout=5)


def test_local_archive_moves_complete_research_to_unique_timestamped_sibling(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / ".research"
    patch = root / "patches" / "000001.json"
    patch.parent.mkdir(parents=True)
    patch.write_text('{"revision": 1}\n', encoding="utf-8")
    (root / "graph.json").write_text("{}\n", encoding="utf-8")
    timestamp = "20260812T123456123456Z"
    collision = tmp_path / f".research.archive-{timestamp}"
    collision.mkdir()
    (collision / "keep.txt").write_text("older archive\n", encoding="utf-8")
    workspace = LocalStateWorkspace(root, str(root))
    monkeypatch.setattr("rcp.transport.state._archive_timestamp", lambda: timestamp)

    archive_location = workspace.archive_research()

    archive = tmp_path / f".research.archive-{timestamp}-2"
    assert archive_location == str(archive)
    assert root.exists() is False
    assert (archive / "patches" / "000001.json").read_text(encoding="utf-8") == (
        '{"revision": 1}\n'
    )
    assert (archive / "graph.json").read_text(encoding="utf-8") == "{}\n"
    assert (collision / "keep.txt").read_text(encoding="utf-8") == "older archive\n"


def test_local_archive_waits_for_in_flight_append_writer(tmp_path) -> None:
    root = tmp_path / ".research"
    patch = root / "patches" / "000001.json"
    patch.parent.mkdir(parents=True)
    patch.write_text("original patch\n", encoding="utf-8")
    lock_path = root / ".append.lock"
    workspace = LocalStateWorkspace(root, str(root))
    archive_started = threading.Event()

    def archive() -> str:
        archive_started.set()
        return workspace.archive_research()

    with lock_path.open("a+", encoding="utf-8") as writer_lock:
        fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(archive)
            assert archive_started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
            assert patch.read_text(encoding="utf-8") == "original patch\n"
            fcntl.flock(writer_lock.fileno(), fcntl.LOCK_UN)
            archive_location = future.result(timeout=5)

    archive_path = Path(archive_location)
    assert root.exists() is False
    assert (archive_path / "patches" / "000001.json").read_text(encoding="utf-8") == (
        "original patch\n"
    )


def test_local_archive_rechecks_reviewed_history_under_append_lock(tmp_path) -> None:
    root = tmp_path / ".research"
    patch = root / "patches" / "000001.json"
    patch.parent.mkdir(parents=True)
    (root / "manifest.toml").write_text("name = 'reviewed'\n", encoding="utf-8")
    patch.write_text("reviewed patch\n", encoding="utf-8")
    workspace = LocalStateWorkspace(root, str(root))
    reviewed = workspace.retained_history_fingerprint()
    archive_started = threading.Event()

    def archive() -> str:
        archive_started.set()
        return workspace.archive_research(expected_history_fingerprint=reviewed)

    with (root / ".append.lock").open("a+", encoding="utf-8") as writer_lock:
        fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(archive)
            assert archive_started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
            patch.write_text("changed patch\n", encoding="utf-8")
            fcntl.flock(writer_lock.fileno(), fcntl.LOCK_UN)
            with pytest.raises(StateUnavailable, match="changed since you reviewed it"):
                future.result(timeout=5)

    assert patch.read_text(encoding="utf-8") == "changed patch\n"
    assert list(tmp_path.glob(".research.archive-*")) == []


@pytest.mark.parametrize(
    "late_relative",
    [Path("patches/000001.json"), Path("merges") / f"{_ARCHIVE_MERGE_ID}.json"],
    ids=["branch-patch", "branch-merge-receipt"],
)
def test_local_archive_token_detects_late_branch_truth(
    tmp_path: Path,
    late_relative: Path,
) -> None:
    root = tmp_path / ".research"
    root.mkdir()
    (root / "manifest.toml").write_text("name = 'reviewed'\n", encoding="utf-8")
    branch = _retained_branch(root)
    workspace = LocalStateWorkspace(root, str(root))
    reviewed = workspace.retained_history_fingerprint()

    (branch / late_relative).write_text("late branch truth\n", encoding="utf-8")

    with pytest.raises(StateUnavailable, match="changed since you reviewed it"):
        workspace.archive_research(expected_history_fingerprint=reviewed)
    assert root.is_dir()


def test_branch_derived_outputs_do_not_change_the_archive_token(tmp_path: Path) -> None:
    from rcp.transport.remote_archive_research import retained_history_fingerprint

    root = tmp_path / ".research"
    root.mkdir()
    (root / "manifest.toml").write_text("name = 'reviewed'\n", encoding="utf-8")
    branch = _retained_branch(root)
    workspace = LocalStateWorkspace(root, str(root))
    reviewed = workspace.retained_history_fingerprint()
    assert retained_history_fingerprint(root) == reviewed

    (branch / "graph.json").write_text('{"derived":false}\n', encoding="utf-8")
    (branch / "research.md").write_text("rebuilt projection\n", encoding="utf-8")

    assert workspace.retained_history_fingerprint() == reviewed
    assert retained_history_fingerprint(root) == reviewed


@pytest.mark.parametrize(
    "unsafe_kind",
    [
        "branch-symlink",
        "malformed-branch",
        "patch-parent-file",
        "patch-leaf-directory",
        "malformed-receipt",
    ],
)
def test_local_archive_token_refuses_unsafe_branch_history(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    root = tmp_path / ".research"
    root.mkdir()
    (root / "manifest.toml").write_text("name = 'reviewed'\n", encoding="utf-8")
    if unsafe_kind in {"branch-symlink", "malformed-branch"}:
        branches = root / "branches"
        branches.mkdir()
        if unsafe_kind == "branch-symlink":
            target = tmp_path / "branch-target"
            target.mkdir()
            (branches / _ARCHIVE_BRANCH_ID).symlink_to(target, target_is_directory=True)
        else:
            (branches / "NOT-A-CANONICAL-UUID").mkdir()
    else:
        branch = _retained_branch(root)
        if unsafe_kind == "patch-parent-file":
            (branch / "patches").rmdir()
            (branch / "patches").write_text("not a directory\n", encoding="utf-8")
        elif unsafe_kind == "patch-leaf-directory":
            (branch / "patches" / "000001.json").mkdir()
        else:
            (branch / "merges" / f"{'A' * 64}.json").write_text("{}\n", encoding="utf-8")

    workspace = LocalStateWorkspace(root, str(root))
    with pytest.raises(StateUnavailable):
        workspace.retained_history_fingerprint()


def test_local_archive_rename_failure_leaves_complete_original_intact(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / ".research"
    patch = root / "patches" / "000001.json"
    patch.parent.mkdir(parents=True)
    patch.write_text("original patch\n", encoding="utf-8")
    workspace = LocalStateWorkspace(root, str(root))

    def fail_rename(_source, _destination) -> None:
        raise PermissionError("rename denied")

    monkeypatch.setattr("rcp.transport.state.os.rename", fail_rename)

    with pytest.raises(StateUnavailable, match="rename denied"):
        workspace.archive_research()

    assert patch.read_text(encoding="utf-8") == "original patch\n"
    assert list(tmp_path.glob(".research.archive-*")) == []


def _enter_run_lock(workspace, **kwargs) -> None:
    with workspace.run_lock(**kwargs):
        pass


def _local_advisory_lock_arguments(path: Path) -> list[str]:
    return _advisory_lock_holder_arguments(path, python_executable=sys.executable)


def _command_lease(path: str, command) -> RunLockLease:
    return RunLockLease(path, command=command)


@pytest.mark.parametrize("name", [".agent-run.lock", ".refresh.lock"])
def test_process_advisory_lock_waits_then_acquires(name, tmp_path) -> None:
    path = tmp_path / name
    arguments = _local_advisory_lock_arguments(path)
    waiting = threading.Event()
    acquired = threading.Event()
    messages: list[str] = []

    def contend() -> None:
        def on_wait(message: str) -> None:
            messages.append(message)
            waiting.set()

        with _process_advisory_lock(arguments, str(path), on_wait=on_wait):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with _process_advisory_lock(arguments, str(path)):
            future = pool.submit(contend)
            assert waiting.wait(timeout=5)
            assert acquired.is_set() is False
        future.result(timeout=5)

    assert acquired.is_set() is True
    assert messages == ["Waiting for another graph-writing run to release canonical state."]


def test_process_advisory_lock_acquires_when_contention_resolves_within_one_read() -> None:
    """A contention that clears fast delivers both statuses in a single read.

    The holder prints `contended`, blocks, then prints `acquired`. When the wait
    is short, both lines reach the reader together, so a status reader that
    polls the raw descriptor would never see the second one.
    """

    coalescing_holder = 'import sys\nsys.stdout.write("contended\\nacquired\\n")\n' + (
        "sys.stdout.flush()\nfor line in sys.stdin:\n    pass\n"
    )
    acquired = threading.Event()

    def acquire() -> None:
        with _process_advisory_lock(
            [sys.executable, "-c", coalescing_holder], "probe:/state/.agent-run.lock"
        ):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(acquire).result(timeout=10)

    assert acquired.is_set() is True


def test_process_advisory_lock_uses_one_waiter_during_long_contention(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / ".agent-run.lock"
    arguments = _local_advisory_lock_arguments(path)
    waiting = threading.Event()
    acquired = threading.Event()
    real_popen = subprocess.Popen
    started: list[subprocess.Popen[str]] = []

    def counting_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", counting_popen)

    def contend() -> None:
        with _process_advisory_lock(
            arguments,
            str(path),
            on_wait=lambda _message: waiting.set(),
        ):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with _process_advisory_lock(arguments, str(path)):
            baseline = len(started)
            future = pool.submit(contend)
            assert waiting.wait(timeout=5)
            time.sleep(STATE_LOCK_POLL_INTERVAL_SECONDS * 3)
            assert len(started) - baseline == 1
            assert acquired.is_set() is False
        future.result(timeout=5)

    assert acquired.is_set() is True


def test_process_advisory_lock_wait_can_be_cancelled(tmp_path) -> None:
    path = tmp_path / ".agent-run.lock"
    arguments = _local_advisory_lock_arguments(path)
    waiting = threading.Event()
    cancellation = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as pool, _process_advisory_lock(arguments, str(path)):
        future = pool.submit(
            _enter_process_lock,
            arguments,
            path,
            waiting,
            cancellation,
        )
        assert waiting.wait(timeout=5)
        cancellation.set()
        with pytest.raises(RunLockCancelled, match="cancelled while waiting"):
            future.result(timeout=5)


def test_stalled_initial_lock_signal_cancels_promptly(tmp_path, monkeypatch) -> None:
    cancellation = threading.Event()
    started = threading.Event()
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started.set()
        return process

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    arguments = [sys.executable, "-c", "import time; time.sleep(60)"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _enter_stalled_process_lock,
            arguments,
            tmp_path,
            cancellation,
        )
        assert started.wait(timeout=5)
        cancelled_at = time.monotonic()
        cancellation.set()
        with pytest.raises(RunLockCancelled, match="cancelled while waiting"):
            future.result(timeout=2)

    assert time.monotonic() - cancelled_at < 2


def _enter_stalled_process_lock(
    arguments: list[str],
    path: Path,
    cancellation: threading.Event,
) -> None:
    with _process_advisory_lock(
        arguments,
        str(path),
        cancelled=cancellation.is_set,
    ):
        pass


def _enter_process_lock(
    arguments: list[str],
    path: Path,
    waiting: threading.Event,
    cancellation: threading.Event,
) -> None:
    with _process_advisory_lock(
        arguments,
        str(path),
        on_wait=lambda _message: waiting.set(),
        cancelled=cancellation.is_set,
    ):
        pass


def test_process_advisory_lock_holder_death_releases_ownership(tmp_path) -> None:
    path = tmp_path / ".agent-run.lock"
    arguments = _local_advisory_lock_arguments(path)
    holder = subprocess.Popen(
        arguments,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "acquired"

    holder.kill()
    holder.wait(timeout=5)

    with _process_advisory_lock(arguments, str(path)):
        pass


def test_killed_acquired_holder_marks_lease_lost_once(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".agent-run.lock"
    arguments = _local_advisory_lock_arguments(path)
    real_popen = subprocess.Popen
    holders: list[subprocess.Popen[str]] = []
    lost = threading.Event()
    messages: list[str] = []

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        holders.append(process)
        return process

    def on_lost(message: str) -> None:
        messages.append(message)
        lost.set()

    monkeypatch.setattr(subprocess, "Popen", recording_popen)

    with (
        pytest.raises(RunLockOwnershipLost, match="exited unexpectedly"),
        _process_advisory_lock(arguments, str(path), on_lost=on_lost) as lease,
    ):
        holders[-1].kill()
        assert lost.wait(timeout=5)
        with pytest.raises(RunLockOwnershipLost, match="exited unexpectedly"):
            lease.assert_owned()
        lease.assert_owned()

    assert len(messages) == 1


def test_intentional_holder_release_does_not_report_loss(tmp_path) -> None:
    path = tmp_path / ".agent-run.lock"
    messages: list[str] = []

    with _process_advisory_lock(
        _local_advisory_lock_arguments(path),
        str(path),
        on_lost=messages.append,
    ) as lease:
        lease.assert_owned()

    time.sleep(STATE_LOCK_POLL_INTERVAL_SECONDS * 2)
    assert messages == []


def test_owned_holder_command_applies_staged_commit_and_files(tmp_path) -> None:
    root = tmp_path / ".research"
    stage = root / ".publish" / "patch-000001.json"
    patch = Path("patches/000001.json")
    for relative, content in ((patch, "patch\n"), (Path("graph.json"), "graph\n")):
        source = stage / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")

    with _process_advisory_lock(
        _local_advisory_lock_arguments(root / ".refresh.lock"),
        str(root / ".refresh.lock"),
    ) as lease:
        response = lease._run_owned_command(
            {
                "op": "apply",
                "root": str(root),
                "stage": str(stage),
                "paths": [patch.as_posix(), "graph.json"],
                "commit": patch.as_posix(),
                "commit_is_directory": False,
            }
        )

    assert response == {"ok": True, "commit_status": "present"}
    assert (root / patch).read_text(encoding="utf-8") == "patch\n"
    assert (root / "graph.json").read_text(encoding="utf-8") == "graph\n"
    assert stage.exists() is False


@pytest.mark.parametrize("name", [".agent-run.lock", ".refresh.lock"])
def test_process_advisory_lock_ignores_unowned_regular_file(name, tmp_path) -> None:
    path = tmp_path / name
    path.write_text("previous holder\n", encoding="utf-8")

    with _process_advisory_lock(_local_advisory_lock_arguments(path), str(path)):
        assert path.read_text(encoding="utf-8") == "previous holder\n"

    assert path.is_file()


@pytest.mark.parametrize("name", [".agent-run.lock", ".refresh.lock"])
def test_process_advisory_lock_reclaims_an_empty_legacy_directory(name, tmp_path) -> None:
    """A crashed mkdir-era run leaves an empty directory; clearing it is RCP's job."""

    path = tmp_path / name
    path.mkdir()

    with _process_advisory_lock(_local_advisory_lock_arguments(path), str(path)):
        assert path.is_file()

    assert path.is_file()


@pytest.mark.parametrize("name", [".agent-run.lock", ".refresh.lock"])
def test_process_advisory_lock_preserves_a_populated_legacy_directory(name, tmp_path) -> None:
    path = tmp_path / name
    path.mkdir()
    marker = path / "owner.txt"
    marker.write_text("unknown owner\n", encoding="utf-8")

    with (
        pytest.raises(StateUnavailable) as raised,
        _process_advisory_lock(_local_advisory_lock_arguments(path), str(path)),
    ):
        pass

    message = str(raised.value)
    assert str(path) in message
    assert "legacy directory RCP could not reclaim" in message
    assert "RCP preserved it" in message
    assert "remove manually" not in message
    assert marker.read_text(encoding="utf-8") == "unknown owner\n"


def test_process_advisory_lock_preserves_symlink_instead_of_following_it(tmp_path) -> None:
    target = tmp_path / "unrelated-state"
    target.write_text("do not touch\n", encoding="utf-8")
    path = tmp_path / ".agent-run.lock"
    path.symlink_to(target)

    with (
        pytest.raises(StateUnavailable) as raised,
        _process_advisory_lock(_local_advisory_lock_arguments(path), str(path)),
    ):
        pass

    message = str(raised.value)
    assert "not a regular file" in message
    assert "RCP preserved it" in message
    assert path.is_symlink()
    assert target.read_text(encoding="utf-8") == "do not touch\n"


def test_remote_advisory_lock_command_quotes_the_exact_path() -> None:
    lock_path = "/srv/project with spaces/$(touch nope)/.agent-run.lock"

    arguments = _remote_advisory_lock_command("research.example", lock_path)

    assert arguments[-2] == "research.example"
    remote_arguments = shlex.split(arguments[-1])
    assert remote_arguments[0:2] == ["python3", "-c"]
    assert remote_arguments[-1] == lock_path


def test_remote_archive_renames_canonical_tree_then_clears_only_stale_mirror(
    tmp_path, monkeypatch
) -> None:
    remote_repository = tmp_path / "remote" / "project"
    remote_root = remote_repository / ".research"
    remote_patch = remote_root / "patches" / "000001.json"
    remote_patch.parent.mkdir(parents=True)
    remote_patch.write_text("remote patch\n", encoding="utf-8")
    timestamp = "20260812T123456123456Z"
    collision = remote_repository / f".research.archive-{timestamp}"
    collision.mkdir()
    (collision / "keep.txt").write_text("older archive\n", encoding="utf-8")

    cache_parent = tmp_path / "cache"
    mirror = cache_parent / ".research"
    stale_patch = mirror / "patches" / "000001.json"
    stale_patch.parent.mkdir(parents=True)
    stale_patch.write_text("stale mirror patch\n", encoding="utf-8")
    unrelated_cache = cache_parent / "keep.txt"
    unrelated_cache.write_text("unrelated cache\n", encoding="utf-8")
    workspace = SSHStateWorkspace(mirror, "research.example", str(remote_repository))

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        yield RunLockLease(str(path))

    def run_remote_command_locally(arguments, **_kwargs):
        return subprocess.run(
            [sys.executable, *arguments[1:]],
            capture_output=True,
            text=True,
            check=False,
        )

    monkeypatch.setattr("rcp.transport.state._archive_timestamp", lambda: timestamp)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(workspace, "_ssh", run_remote_command_locally)

    archive_location = workspace.archive_research()

    remote_archive = remote_repository / f".research.archive-{timestamp}-2"
    assert archive_location == f"research.example:{remote_archive}"
    assert remote_root.exists() is False
    assert (remote_archive / "patches" / "000001.json").read_text(encoding="utf-8") == (
        "remote patch\n"
    )
    assert (collision / "keep.txt").read_text(encoding="utf-8") == "older archive\n"
    assert mirror.exists() is False
    assert unrelated_cache.read_text(encoding="utf-8") == "unrelated cache\n"


def test_remote_archive_failure_preserves_canonical_tree_and_local_mirror(
    tmp_path, monkeypatch
) -> None:
    remote_repository = tmp_path / "remote" / "project"
    remote_patch = remote_repository / ".research" / "patches" / "000001.json"
    remote_patch.parent.mkdir(parents=True)
    remote_patch.write_text("remote patch\n", encoding="utf-8")
    mirror = tmp_path / "cache" / ".research"
    stale_patch = mirror / "patches" / "000001.json"
    stale_patch.parent.mkdir(parents=True)
    stale_patch.write_text("stale mirror patch\n", encoding="utf-8")
    workspace = SSHStateWorkspace(mirror, "research.example", str(remote_repository))

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        yield RunLockLease(str(path))

    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(
        workspace,
        "_ssh",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 1, "", "rename denied"),
    )

    with pytest.raises(StateUnavailable, match="original directory remains intact"):
        workspace.archive_research()

    assert remote_patch.read_text(encoding="utf-8") == "remote patch\n"
    assert stale_patch.read_text(encoding="utf-8") == "stale mirror patch\n"


def test_remote_archive_rechecks_reviewed_history_while_refresh_lock_is_held(
    tmp_path,
    monkeypatch,
) -> None:
    remote_repository = tmp_path / "remote" / "project"
    remote_root = remote_repository / ".research"
    remote_patch = remote_root / "patches" / "000001.json"
    remote_patch.parent.mkdir(parents=True)
    (remote_root / "manifest.toml").write_text("name = 'remote'\n", encoding="utf-8")
    remote_patch.write_text("reviewed patch\n", encoding="utf-8")
    mirror = tmp_path / "cache" / ".research"
    shutil.copytree(remote_root, mirror)
    workspace = SSHStateWorkspace(mirror, "research.example", str(remote_repository))
    reviewed = workspace.retained_history_fingerprint()
    remote_patch.write_text("changed patch\n", encoding="utf-8")
    lock_held = False

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        nonlocal lock_held
        lock_held = True
        try:
            yield RunLockLease(str(path), owned=lambda: lock_held)
        finally:
            lock_held = False

    def run_remote_command_locally(arguments, **_kwargs):
        assert lock_held is True
        return subprocess.run(
            [sys.executable, *arguments[1:]],
            capture_output=True,
            text=True,
            check=False,
        )

    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(workspace, "_ssh", run_remote_command_locally)

    with pytest.raises(StateUnavailable, match="changed since you reviewed it"):
        workspace.archive_research(expected_history_fingerprint=reviewed)

    assert remote_patch.read_text(encoding="utf-8") == "changed patch\n"
    assert (mirror / "patches" / "000001.json").read_text(encoding="utf-8") == ("reviewed patch\n")
    assert list(remote_repository.glob(".research.archive-*")) == []


@pytest.mark.parametrize(
    "late_relative",
    [Path("patches/000001.json"), Path("merges") / f"{_ARCHIVE_MERGE_ID}.json"],
    ids=["branch-patch", "branch-merge-receipt"],
)
def test_remote_archive_token_detects_late_branch_truth(
    tmp_path: Path,
    monkeypatch,
    late_relative: Path,
) -> None:
    remote_repository = tmp_path / "remote" / "project"
    remote_root = remote_repository / ".research"
    remote_root.mkdir(parents=True)
    (remote_root / "manifest.toml").write_text("name = 'remote'\n", encoding="utf-8")
    remote_branch = _retained_branch(remote_root)
    mirror = tmp_path / "cache" / ".research"
    shutil.copytree(remote_root, mirror)
    workspace = SSHStateWorkspace(mirror, "research.example", str(remote_repository))
    reviewed = workspace.retained_history_fingerprint()
    (remote_branch / late_relative).write_text("late remote branch truth\n", encoding="utf-8")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        yield RunLockLease(str(path))

    def run_remote_command_locally(arguments, **_kwargs):
        return subprocess.run(
            [sys.executable, *arguments[1:]],
            capture_output=True,
            text=True,
            check=False,
        )

    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(workspace, "_ssh", run_remote_command_locally)

    with pytest.raises(StateUnavailable, match="changed since you reviewed it"):
        workspace.archive_research(expected_history_fingerprint=reviewed)
    assert remote_root.is_dir()
    assert mirror.is_dir()


def test_confirmed_absent_remote_discards_stale_mirror_before_fresh_initialization(
    tmp_path,
    monkeypatch,
) -> None:
    request = ProjectSetupRequest.model_validate(
        {
            "name": "same-name",
            "repositories": [
                {
                    "alias": "remote-repo",
                    "location": "ssh",
                    "host": "gpu.example",
                    "path": "/srv/paper",
                    "default_read": True,
                }
            ],
            "state_repository": "remote-repo",
            "execution": {"location": "ssh", "host": "gpu.example"},
        }
    )
    bootstrap_path = tmp_path / "bootstrap.toml"
    bootstrap_path.write_text(render_manifest(request), encoding="utf-8")
    bootstrap = load_manifest(bootstrap_path)
    mirror = tmp_path / "state-cache" / ".research"
    mirror.mkdir(parents=True)
    stale_manifest = mirror / "manifest.toml"
    stale_manifest.write_text(render_manifest(request), encoding="utf-8")
    stale_patch = mirror / "patches" / "000001.json"
    stale_patch.parent.mkdir()
    stale_patch.write_text("archived history must not return\n", encoding="utf-8")
    workspace = SSHStateWorkspace(mirror, "gpu.example", "/srv/paper")
    monkeypatch.setattr(workspace, "refresh", lambda: False)
    monkeypatch.setattr(
        "rcp.transport.state.state_workspace_for_probe",
        lambda _bootstrap, _data_dir: workspace,
    )

    fresh, returned_workspace = prepare_state_workspace(bootstrap, tmp_path / "data")

    assert returned_workspace is workspace
    assert fresh.path == mirror / "manifest.toml"
    assert fresh.name == "same-name"
    assert not stale_patch.exists()
    assert stale_manifest.read_text(encoding="utf-8") == bootstrap_path.read_text(encoding="utf-8")


def test_remote_run_lock_forwards_wait_and_cancellation_hooks(tmp_path, monkeypatch) -> None:
    workspace = SSHStateWorkspace(tmp_path / ".research", "research.example", "/srv/project")
    calls: list[tuple[str, object, object, object]] = []

    def on_wait(_message: str) -> None:
        pass

    def cancelled() -> bool:
        return False

    def on_lost(_message: str) -> None:
        pass

    @contextmanager
    def fake_remote_lock(path, **kwargs):
        calls.append(
            (
                str(path),
                kwargs.get("on_wait"),
                kwargs.get("cancelled"),
                kwargs.get("on_lost"),
            )
        )
        yield RunLockLease(str(path))

    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)

    with workspace.run_lock(on_wait=on_wait, cancelled=cancelled, on_lost=on_lost) as lease:
        lease.assert_owned()

    assert calls == [("/srv/project/.research/.agent-run.lock", on_wait, cancelled, on_lost)]


def test_remote_patch_head_probe_takes_no_lock_and_copies_nothing(tmp_path, monkeypatch) -> None:
    root = tmp_path / ".research"
    direct = root / "patches" / "000002.json"
    batched = root / "patches" / "batch-000003-000004-test" / "000004.json"
    unpublished = root / "patches" / ".batch-000005-000006-test" / "000006.json"
    for path in (direct, batched, unpublished):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not read by the probe\n", encoding="utf-8")
    local_probe = subprocess.run(
        ["sh", "-c", _REMOTE_PATCH_LOG_HEAD_SCRIPT, "rcp-patch-log-head", str(root / "patches")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert local_probe.stdout == "000004.json\n"
    workspace = SSHStateWorkspace(root, "research.example", "/srv/project")
    ssh_calls: list[tuple[list[str], float]] = []

    def fake_ssh(arguments, *, timeout):
        ssh_calls.append((arguments, timeout))
        return subprocess.CompletedProcess(arguments, 0, "000005.json\n", "")

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(
        workspace,
        "_remote_advisory_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("head probe must not take the canonical lock")
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("head probe must not invoke rsync")
        ),
    )

    assert workspace.probe_remote_patch_log_head() == (True, 5)
    assert len(ssh_calls) == 1
    assert ssh_calls[0][0][0:2] == ["sh", "-c"]
    assert ssh_calls[0][0][-1] == "/srv/project/.research/patches"


def test_transient_remote_patch_head_failure_does_not_change_workspace_health(
    tmp_path, monkeypatch
) -> None:
    workspace = SSHStateWorkspace(tmp_path / ".research", "research.example", "/srv/project")
    workspace.reachable = True
    workspace.error = None

    monkeypatch.setattr(
        workspace,
        "_ssh",
        lambda _arguments, **_kwargs: subprocess.CompletedProcess([], 255, "", "timed out"),
    )

    assert workspace.probe_remote_patch_log_head() == (False, None)
    assert workspace.reachable is True
    assert workspace.error is None


def test_remote_refresh_and_transaction_use_one_canonical_lock_and_sync(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / ".research"
    root.mkdir()
    (root / "graph.json").write_text("{}\n", encoding="utf-8")
    workspace = SSHStateWorkspace(root, "research.example", "/srv/project")
    ssh_calls: list[list[str]] = []
    rsync_calls: list[list[str]] = []
    lock_calls: list[str] = []
    commands: list[dict[str, object]] = []

    def fake_ssh(arguments):
        ssh_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_run(arguments, **_kwargs):
        rsync_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        lock_calls.append(str(path))

        def command(payload):
            commands.append(payload)
            return {"ok": True, "commit_status": None}

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert workspace.refresh() is True
    assert ssh_calls == [
        ["test", "-f", "/srv/project/.research/manifest.toml"],
    ]
    assert lock_calls == ["/srv/project/.research/.refresh.lock"]
    assert len([call for call in rsync_calls if "--delete" in call]) == 1

    ssh_calls.clear()
    rsync_calls.clear()
    with workspace.transaction():
        assert workspace.refresh() is True
        workspace.publish(["graph.json"])

    assert ssh_calls.count(["test", "-f", "/srv/project/.research/manifest.toml"]) == 2
    assert lock_calls == [
        "/srv/project/.research/.refresh.lock",
        "/srv/project/.research/.refresh.lock",
    ]
    assert len([call for call in rsync_calls if "--delete" in call]) == 2
    assert len([call for call in rsync_calls if "-aR" in call]) == 1
    assert len(commands) == 1
    assert commands[0]["paths"] == ["graph.json"]
    assert "commit" not in commands[0]
    assert ".publish/files-" in str(commands[0]["stage"])


def test_remote_batch_publication_stages_then_commits_directory_last(tmp_path, monkeypatch) -> None:
    root = tmp_path / ".research"
    batch = Path("patches/batch-000002-000003-test")
    for relative, content in (
        (batch / "000002.json", "{}"),
        (batch / "000003.json", "{}"),
        (Path("graph.json"), "{}"),
        (Path("research.md"), "accepted"),
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    workspace = SSHStateWorkspace(root, "research.example", "/srv/project")
    rsync_calls = []
    commands: list[dict[str, object]] = []

    def fake_ssh(_arguments):
        return subprocess.CompletedProcess([], 0, "", "")

    def fake_run(arguments, **kwargs):
        rsync_calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        def command(payload):
            commands.append(payload)
            return {"ok": True, "commit_status": "present"}

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace.publish_committed_batch(
        [batch / "000002.json", batch / "000003.json", "graph.json", "research.md"],
        batch,
    )

    assert len(rsync_calls) == 1
    assert ".publish/batch-000002-000003-test" in rsync_calls[0][0][-1]
    assert len(commands) == 1
    assert commands[0]["commit"] == batch.as_posix()
    assert commands[0]["commit_is_directory"] is True
    assert workspace.reachable is True


def test_remote_patch_publication_commits_file_before_derived_outputs(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / ".research"
    patch = Path("patches/000001.json")
    for relative in (patch, Path("graph.json")):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    workspace = SSHStateWorkspace(root, "research.example", "/srv/project")
    rsync_calls: list[tuple[list[str], dict]] = []
    commands: list[dict[str, object]] = []

    def fake_ssh(_arguments):
        return subprocess.CompletedProcess([], 0, "", "")

    def fake_run(arguments, **kwargs):
        rsync_calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        def command(payload):
            commands.append(payload)
            return {"ok": True, "commit_status": "present"}

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace.publish_committed_patch([patch, "graph.json"], patch)

    assert len(rsync_calls) == 1
    assert ".publish/patch-000001.json" in rsync_calls[0][0][-1]
    assert commands[0]["commit"] == patch.as_posix()
    assert commands[0]["commit_is_directory"] is False


def test_remote_patch_publish_probes_commit_and_repairs_idempotently(tmp_path, monkeypatch) -> None:
    root = tmp_path / ".research"
    patch = Path("patches/000001.json")
    for relative in (patch, Path("graph.json")):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    workspace = SSHStateWorkspace(root, "research.example", "/srv/project")
    commands: list[dict[str, object]] = []

    def fake_ssh(_arguments):
        return subprocess.CompletedProcess([], 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        def command(payload):
            commands.append(payload)
            if len(commands) == 1:
                return {
                    "ok": False,
                    "commit_status": "present",
                    "error": "derived output failed",
                }
            return {"ok": True, "commit_status": "present"}

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0, "", ""),
    )

    workspace.publish_committed_patch([patch, "graph.json"], patch)

    assert len(commands) == 2
    assert workspace.reachable is True


def test_remote_patch_publish_reports_unknown_when_commit_probe_fails(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / ".research"
    patch = Path("patches/000001.json")
    for relative in (patch, Path("graph.json")):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    workspace = SSHStateWorkspace(root, "research.example", "/srv/project")

    def fake_ssh(arguments):
        if arguments[0:2] == ["test", "-f"]:
            return subprocess.CompletedProcess([], 255, "", "probe disconnected")
        return subprocess.CompletedProcess([], 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        def command(_payload):
            raise RunLockOwnershipLost("apply disconnected")

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0, "", ""),
    )

    with pytest.raises(BatchPublishFailed) as caught:
        workspace.publish_committed_patch([patch, "graph.json"], patch)

    assert caught.value.commit_status == "unknown"
    assert "probe disconnected" in str(caught.value)


def test_holder_death_after_staging_cannot_apply_commit_unfenced(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache" / ".research"
    remote_root = tmp_path / "remote-project" / ".research"
    patch = Path("patches/000001.json")
    for relative in (patch, Path("graph.json")):
        source = cache_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("{}\n", encoding="utf-8")
    workspace = SSHStateWorkspace(cache_root, "research.example", str(remote_root.parent))
    ssh_calls: list[list[str]] = []
    holders: list[subprocess.Popen[str]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        holders.append(process)
        return process

    def fake_ssh(arguments):
        ssh_calls.append(arguments)
        if arguments[0:2] == ["mkdir", "-p"]:
            Path(arguments[-1]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if arguments[0:2] == ["test", "-f"]:
            return subprocess.CompletedProcess(
                arguments,
                0 if Path(arguments[-1]).is_file() else 1,
                "",
                "",
            )
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_rsync(arguments, **kwargs):
        stage = Path(arguments[-1].split(":", 1)[1].rstrip("/"))
        for relative in (patch, Path("graph.json")):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(kwargs["cwd"]) / relative, target)
        holders[-1].kill()
        holders[-1].wait(timeout=5)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    @contextmanager
    def local_remote_lock(path, **_kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with _process_advisory_lock(
            _local_advisory_lock_arguments(Path(path)),
            str(path),
        ) as lease:
            yield lease

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(subprocess, "run", fake_rsync)
    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", local_remote_lock)

    with pytest.raises(BatchPublishFailed) as caught:
        workspace.publish_committed_patch([patch, "graph.json"], patch)

    assert caught.value.commit_status == "absent"
    assert (remote_root / patch).exists() is False
    assert (remote_root / ".publish" / "patch-000001.json" / patch).is_file()
    assert not any(call[0:2] == ["python3", "-c"] for call in ssh_calls)


def test_commit_channel_loss_reconciles_present_commit(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache" / ".research"
    remote_root = tmp_path / "remote-project" / ".research"
    patch = Path("patches/000001.json")
    for relative in (patch, Path("graph.json")):
        source = cache_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("{}\n", encoding="utf-8")
    workspace = SSHStateWorkspace(cache_root, "research.example", str(remote_root.parent))

    def fake_ssh(arguments):
        if arguments[0:2] == ["test", "-f"]:
            return subprocess.CompletedProcess(
                arguments,
                0 if Path(arguments[-1]).is_file() else 1,
                "",
                "",
            )
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_rsync(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        def command(_payload):
            target = remote_root / patch
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("committed\n", encoding="utf-8")
            raise RunLockOwnershipLost("channel died after commit point")

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(subprocess, "run", fake_rsync)

    with pytest.raises(BatchPublishFailed) as caught:
        workspace.publish_committed_patch([patch, "graph.json"], patch)

    assert caught.value.commit_status == "present"
    assert (remote_root / patch).read_text(encoding="utf-8") == "committed\n"


def test_absent_commit_probe_waits_for_old_holder_before_classifying(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache" / ".research"
    remote_root = tmp_path / "remote-project" / ".research"
    patch = Path("patches/000001.json")
    for relative in (patch, Path("graph.json")):
        source = cache_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("{}\n", encoding="utf-8")
    workspace = SSHStateWorkspace(cache_root, "research.example", str(remote_root.parent))
    lock_entries = 0
    probes: list[int] = []

    def fake_ssh(arguments):
        if arguments[0:2] == ["test", "-f"]:
            return_code = 0 if Path(arguments[-1]).is_file() else 1
            probes.append(return_code)
            return subprocess.CompletedProcess(arguments, return_code, "", "")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        nonlocal lock_entries
        lock_entries += 1
        if lock_entries == 2:
            target = remote_root / patch
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("committed by old holder\n", encoding="utf-8")

        def command(_payload):
            raise RunLockOwnershipLost("channel died before the holder reported its outcome")

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0, "", ""),
    )

    with pytest.raises(BatchPublishFailed) as caught:
        workspace.publish_committed_patch([patch, "graph.json"], patch)

    assert probes == [1, 0]
    assert lock_entries == 2
    assert caught.value.commit_status == "present"


def test_ordinary_publish_channel_loss_requires_full_restage(tmp_path, monkeypatch) -> None:
    root = tmp_path / ".research"
    root.mkdir()
    (root / "graph.json").write_text("{}\n", encoding="utf-8")
    workspace = SSHStateWorkspace(root, "research.example", "/srv/project")
    rsync_calls: list[list[str]] = []

    def fake_ssh(arguments):
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_rsync(arguments, **_kwargs):
        rsync_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        def command(_payload):
            raise RunLockOwnershipLost("channel died during ordinary apply")

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(subprocess, "run", fake_rsync)

    with pytest.raises(RunLockOwnershipLost) as caught:
        workspace.publish(["graph.json"])

    assert "prefix may have been applied" in str(caught.value)
    assert "Retry in a new transaction" in str(caught.value)
    assert len(rsync_calls) == 1
    assert ".publish/files-" in rsync_calls[0][-1]
    assert rsync_calls[0][-1] != "research.example:/srv/project/.research/"


def test_failed_remote_single_patch_before_commit_rolls_back_local_mirror(manifest) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())
    graph_before = (manifest.research_dir / "graph.json").read_bytes()

    def fail_before_commit(_relative_paths, _patch_path):
        raise BatchPublishFailed("remote staging failed", commit_status="absent")

    workspace.publish_committed_patch = fail_before_commit

    with pytest.raises(BatchPublishFailed, match="remote staging failed"):
        history.append(_accept_question_patch(), expected_revision=1)

    assert [patch.revision for patch in history.load_patches()] == [1]
    assert (manifest.research_dir / "graph.json").read_bytes() == graph_before
    assert not (manifest.research_dir / "patches" / "000002.json").exists()
    assert not list((manifest.research_dir / "patches").glob(".unconfirmed-000002.json-*"))
    assert workspace.materialization_repair_required is False


def test_confirmed_remote_single_patch_is_success_and_schedules_output_repair(manifest) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())

    def fail_after_commit(_relative_paths, _patch_path):
        raise BatchPublishFailed("derived output repair failed", commit_status="present")

    workspace.publish_committed_patch = fail_after_commit

    appended, result = history.append(_accept_question_patch(), expected_revision=1)

    assert appended.revision == 2
    assert result.state.nodes["rq/learning-after-shift"].standing == "accepted"
    assert [patch.revision for patch in history.load_patches()] == [1, 2]
    assert workspace.materialization_repair_required is True


def test_unknown_remote_single_patch_is_quarantined_from_local_replay(manifest) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())

    def lose_commit_probe(_relative_paths, _patch_path):
        raise BatchPublishFailed("commit probe failed", commit_status="unknown")

    workspace.publish_committed_patch = lose_commit_probe

    with pytest.raises(BatchPublishFailed, match="commit probe failed"):
        history.append(_accept_question_patch(), expected_revision=1)

    assert [patch.revision for patch in history.load_patches()] == [1]
    assert not (manifest.research_dir / "patches" / "000002.json").exists()
    assert list((manifest.research_dir / "patches").glob(".unconfirmed-000002.json-*"))
    assert history.materialize(write_outputs=False).state.revision == 1
    assert workspace.materialization_repair_required is True


def test_rejected_remote_single_patch_remains_in_committed_history(manifest) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())
    rejected = Patch(
        kind="refresh",
        author="agent",
        summary="Invalid gated transition.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "changes": {"status": "supported"},
                    }
                ],
            }
        ],
    )

    with pytest.raises(PatchRejected):
        history.append(rejected)

    assert [patch.revision for patch in history.load_patches()] == [1, 2]
    assert workspace.committed_patches[-1] == "patches/000002.json"


def test_failed_remote_transition_publish_rolls_the_local_mirror_back(manifest) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())
    graph_before = (manifest.research_dir / "graph.json").read_bytes()

    def fail_publish(_relative_paths, _patch_path):
        raise StateUnavailable("remote commit failed")

    workspace.publish_committed_patch = fail_publish

    with pytest.raises(StateUnavailable, match="remote commit failed"):
        history.append_batch(
            [
                Patch(
                    kind="approval",
                    author="human",
                    summary="Accept the research question.",
                    ops=[
                        {
                            "op": "set_standing",
                            "node_id": "rq/learning-after-shift",
                            "standing": "accepted",
                        }
                    ],
                )
            ],
            expected_revision=1,
        )

    assert [patch.revision for patch in history.load_patches()] == [1]
    assert history.state().revision == 1
    assert (manifest.research_dir / "graph.json").read_bytes() == graph_before
    assert not (manifest.research_dir / "patches" / "000002.json").exists()
    assert list((manifest.research_dir / "patches").glob(".unconfirmed-000002.json-*"))


def test_confirmed_remote_transition_commit_is_not_rolled_back(manifest) -> None:
    workspace = RecordingWorkspace(manifest.research_dir)
    history = HistoryManager(manifest, workspace)
    history.append(seed_patch())

    def fail_after_commit(_relative_paths, _patch_path):
        raise BatchPublishFailed("derived output repair failed", commit_status="present")

    workspace.publish_committed_patch = fail_after_commit

    history.append_batch(
        [
            Patch(
                kind="approval",
                author="human",
                summary="Accept the research question.",
                ops=[
                    {
                        "op": "set_standing",
                        "node_id": "rq/learning-after-shift",
                        "standing": "accepted",
                    }
                ],
            )
        ],
        expected_revision=1,
    )

    assert [patch.revision for patch in history.load_patches()] == [1, 2]
    assert workspace.materialization_repair_required is True

    publish = workspace.publish

    def fail_repair(_paths):
        raise StateUnavailable("repair is still blocked")

    workspace.publish = fail_repair
    with pytest.raises(StateUnavailable, match="repair is still blocked"):
        history.state()
    assert workspace.materialization_repair_required is True

    workspace.publish = publish
    assert history.state().nodes["rq/learning-after-shift"].standing == "accepted"
    assert workspace.materialization_repair_required is False


def test_remote_batch_retries_remaining_outputs_after_commit_point(tmp_path, monkeypatch) -> None:
    root = tmp_path / ".research"
    batch = Path("patches/batch-000001-000001-test")
    for relative in (batch / "000001.json", Path("graph.json")):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    workspace = SSHStateWorkspace(root, "research.example", "/srv/project")
    commands: list[dict[str, object]] = []

    def fake_ssh(_arguments):
        return subprocess.CompletedProcess([], 0, "", "")

    @contextmanager
    def fake_remote_lock(path, **_kwargs):
        def command(payload):
            commands.append(payload)
            if len(commands) == 1:
                return {"ok": False, "commit_status": "present", "error": "partial apply"}
            return {"ok": True, "commit_status": "present"}

        yield _command_lease(str(path), command)

    monkeypatch.setattr(workspace, "_ssh", fake_ssh)
    monkeypatch.setattr(workspace, "_remote_advisory_lock", fake_remote_lock)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0, "", ""),
    )

    workspace.publish_committed_batch([batch / "000001.json", "graph.json"], batch)

    assert len(commands) == 2
    assert workspace.reachable is True


def test_local_repository_pointer_has_no_host() -> None:
    access = repository_access(
        RepositoryConfig(alias="rcp", machine="laptop", path="/Users/me/research/RCP"),
        MachineConfig(alias="laptop"),
    )

    assert access.model_dump() == {
        "alias": "rcp",
        "machine": "laptop",
        "host": "",
        "path": "/Users/me/research/RCP",
    }


def test_remote_repository_pointer_keeps_host_and_untouched_path() -> None:
    access = repository_access(
        RepositoryConfig(alias="cot-steering", machine="remote-1", path="/home/research/cot-loop"),
        MachineConfig(alias="remote-1", host="research.example"),
    )

    assert access.host == "research.example"
    assert access.path == "/home/research/cot-loop"


def test_remote_run_stage_probe_rejects_symlink_and_missing_root(tmp_path, monkeypatch) -> None:
    stage = RemoteRunStage("research.example")
    real_run = subprocess.run
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: real_run(arguments, capture_output=True, text=True, check=False),
    )
    remote_root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    try:
        assert stage.directory_exists(str(remote_root)) is True
        remote_root.rmdir()
        target = tmp_path / "replacement"
        target.mkdir()
        remote_root.symlink_to(target, target_is_directory=True)

        assert stage.directory_exists(str(remote_root)) is False
        with pytest.raises(StateUnavailable, match="saved remote staging directory"):
            stage.attach(str(remote_root))

        remote_root.unlink()
        assert stage.directory_exists(str(remote_root)) is False
    finally:
        if remote_root.is_symlink():
            remote_root.unlink()
        elif remote_root.exists():
            remote_root.rmdir()


def test_remote_run_stage_probe_keeps_ssh_failure_transient(monkeypatch) -> None:
    stage = RemoteRunStage("research.example")
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: subprocess.CompletedProcess(arguments, 255, "", "connection lost"),
    )

    assert stage.directory_exists("/tmp/rcp-run.saved-stage") is None


def test_remote_run_inputs_are_published_as_one_bundle(tmp_path, monkeypatch) -> None:
    source_file = tmp_path / "schema.json"
    source_file.write_text('{"type":"object"}\n', encoding="utf-8")
    source_directory = tmp_path / "conversations"
    source_directory.mkdir()
    (source_directory / "session.jsonl").write_text("{}\n", encoding="utf-8")
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath("/tmp/rcp-run.test")
    ssh_calls: list[list[str]] = []
    rsync_calls: list[list[str]] = []

    def fake_ssh(arguments):
        ssh_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_run(arguments, **_kwargs):
        rsync_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(stage, "_ssh", fake_ssh)
    monkeypatch.setattr(subprocess, "run", fake_run)

    schema_path = stage.put_file(source_file, "schema.json")
    conversations_path = stage.put_directory(source_directory, "conversations")
    pending = stage._pending_inputs

    assert schema_path == "/tmp/rcp-run.test/inputs/schema.json"
    assert conversations_path == "/tmp/rcp-run.test/inputs/conversations"
    assert rsync_calls == []
    assert ssh_calls == []

    stage.finalize_inputs()

    assert len(rsync_calls) == 1
    assert rsync_calls[0][0:2] == ["rsync", "-a"]
    assert len(ssh_calls) == 1
    assert ssh_calls[0][0:2] == ["python3", "-c"]
    assert json.loads(ssh_calls[0][5]) == ["conversations", "schema.json"]
    assert ssh_calls[0][6] == "1"
    assert stage._pending_inputs is None
    assert pending is not None
    assert not pending.exists()


def test_remote_directory_input_is_moved_before_being_protected(tmp_path, monkeypatch) -> None:
    """Directory inputs must remain writable until the atomic rename completes."""

    root = tmp_path / "stage"
    (root / "inputs").mkdir(parents=True)
    (root / "workspace").mkdir()
    source_directory = tmp_path / "skill-bundle"
    (source_directory / "skill" / "graph-audit").mkdir(parents=True)
    (source_directory / "skill" / "graph-audit" / "SKILL.md").write_text(
        "# Graph audit\n", encoding="utf-8"
    )
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    real_run = subprocess.run

    def fake_run(arguments, **_kwargs):
        if arguments[0] == "rsync":
            source = Path(arguments[-2].rstrip("/"))
            destination = Path(arguments[-1].split(":", 1)[1].rstrip("/"))
            shutil.copytree(source, destination)
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return real_run(arguments, capture_output=True, text=True, check=False)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: real_run(arguments, capture_output=True, text=True, check=False),
    )

    try:
        stage.put_directory(source_directory, "rcp-skills-turn-1")
        stage.finalize_inputs()

        target = root / "inputs" / "rcp-skills-turn-1"
        assert (target / "skill" / "graph-audit" / "SKILL.md").is_file()
        assert target.stat().st_mode & 0o777 == 0o500
        assert (target / "skill" / "graph-audit" / "SKILL.md").stat().st_mode & 0o777 == 0o400
        assert not any(path.name.startswith(".input-batch-") for path in root.iterdir())
    finally:
        stage.close()


def test_remote_directory_input_reuses_only_matching_immutable_content(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "stage"
    (root / "inputs").mkdir(parents=True)
    (root / "workspace").mkdir()
    source_directory = tmp_path / "skill-bundle"
    source_directory.mkdir()
    (source_directory / "SKILL.md").write_text("# Graph audit\n", encoding="utf-8")
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    real_run = subprocess.run
    rsync_calls = 0

    def fake_run(arguments, **_kwargs):
        nonlocal rsync_calls
        if arguments[0] == "rsync":
            rsync_calls += 1
            source = Path(arguments[-2].rstrip("/"))
            destination = Path(arguments[-1].split(":", 1)[1].rstrip("/"))
            shutil.copytree(source, destination)
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return real_run(arguments, capture_output=True, text=True, check=False)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: real_run(arguments, capture_output=True, text=True, check=False),
    )

    try:
        first = stage.put_directory(source_directory, "rcp-skills-v1-address", reuse=True)
        stage.finalize_inputs()
        second = stage.put_directory(source_directory, "rcp-skills-v1-address", reuse=True)
        stage.finalize_inputs()

        target = root / "inputs" / "rcp-skills-v1-address"
        assert first == second == str(target)
        assert rsync_calls == 1
        assert sorted(path.name for path in (root / "inputs").iterdir()) == [target.name]

        (target / "SKILL.md").chmod(0o600)
        with pytest.raises(ValueError, match="writable"):
            stage.put_directory(source_directory, "rcp-skills-v1-address", reuse=True)
    finally:
        stage.close()


def test_remote_stage_failed_finalize_cleans_local_pending_inputs(tmp_path, monkeypatch) -> None:
    source = tmp_path / "contract.md"
    source.write_text("Run the task.\n", encoding="utf-8")
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath("/tmp/rcp-run.test")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 23, "", "connection lost"
        ),
    )
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: subprocess.CompletedProcess(arguments, 44, "", "incomplete"),
    )
    stage.put_file(source, "contract.md")
    pending = stage._pending_inputs

    with pytest.raises(StateUnavailable, match="connection lost"):
        stage.finalize_inputs()

    assert stage._pending_inputs is None
    assert pending is not None
    assert not pending.exists()


def test_remote_stage_lists_workspace_files(monkeypatch) -> None:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "patch.json").write_text("{}", encoding="utf-8")
    (workspace / "notes.md").write_text("notes", encoding="utf-8")
    (workspace / "linked.json").symlink_to(workspace / "patch.json")
    (workspace / "nested").mkdir()
    (workspace / "nested" / "deep.json").write_text("{}", encoding="utf-8")
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: subprocess.run(arguments, capture_output=True, text=True, check=False),
    )
    try:
        assert stage.list_workspace_files() == ["notes.md", "patch.json"]
    finally:
        shutil.rmtree(root)


def test_remote_stage_workspace_mailbox_round_trip_is_atomic(monkeypatch) -> None:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    workspace = root / "workspace"
    inputs = root / "inputs"
    workspace.mkdir()
    inputs.mkdir()
    immutable_input = inputs / "validator-request.json"
    immutable_input.write_text("input stays immutable", encoding="utf-8")
    immutable_input.chmod(0o400)
    target = workspace / "validator-response-01.json"
    target.write_text("old response", encoding="utf-8")
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    calls: list[tuple[list[str], bytes | None]] = []

    def fake_ssh_bytes(arguments, *, input_data=None):
        calls.append((arguments, input_data))
        return subprocess.run(
            arguments,
            capture_output=True,
            input=input_data,
            check=False,
        )

    monkeypatch.setattr(stage, "_ssh_bytes", fake_ssh_bytes)
    try:
        response = '{"valid":true}\n'
        stage.write_workspace_text("validator-response-01.json", response)

        assert target.is_file()
        assert not target.is_symlink()
        assert stage.read_workspace_text("validator-response-01.json") == response
        assert stage.read_text(stage.workspace / "validator-response-01.json") == response
        assert immutable_input.read_text(encoding="utf-8") == "input stays immutable"
        assert immutable_input.stat().st_mode & 0o777 == 0o400
        assert not any(path.name.startswith(".rcp-write-") for path in workspace.iterdir())
        assert "os.replace" in calls[0][0][2]
        assert calls[0][1] == response.encode("utf-8")

        call_count = len(calls)
        with pytest.raises(ValueError, match="plain base name"):
            stage.write_workspace_text("../inputs/validator-request.json", "not allowed")
        with pytest.raises(ValueError, match="unsupported characters"):
            stage.write_workspace_text("validator response.json", "not allowed")
        with pytest.raises(ValueError, match="direct child"):
            stage.read_text(PurePosixPath(str(immutable_input)))
        assert len(calls) == call_count
    finally:
        immutable_input.chmod(0o600)
        shutil.rmtree(root)


def test_remote_stage_workspace_mailbox_rejects_symlinks(monkeypatch) -> None:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    workspace = root / "workspace"
    workspace.mkdir()
    outside = root / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    linked = workspace / "validator-request.json"
    linked.symlink_to(outside)
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))

    def fake_ssh_bytes(arguments, *, input_data=None):
        return subprocess.run(
            arguments,
            capture_output=True,
            input=input_data,
            check=False,
        )

    monkeypatch.setattr(stage, "_ssh_bytes", fake_ssh_bytes)
    try:
        with pytest.raises(ValueError, match="readable regular file"):
            stage.read_workspace_text("validator-request.json")
        with pytest.raises(ValueError, match="target is not a regular file"):
            stage.write_workspace_text("validator-request.json", "replacement")

        assert linked.is_symlink()
        assert outside.read_text(encoding="utf-8") == "outside"
    finally:
        shutil.rmtree(root)


def test_remote_stage_absent_mailbox_file_is_not_an_unreachable_workspace(monkeypatch) -> None:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    (root / "workspace").mkdir()
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: subprocess.run(arguments, capture_output=True, text=True, check=False),
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
    try:
        assert stage.list_workspace_files() == []
        with pytest.raises(FileNotFoundError, match="is absent"):
            stage.read_workspace_text("validator-request.json")
    finally:
        shutil.rmtree(root)


def test_remote_stage_workspace_operations_fail_closed(monkeypatch) -> None:
    """An unreachable workspace is not an empty one, and a failed delete is not a delete."""

    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath("/tmp/rcp-run.test")
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda _arguments: subprocess.CompletedProcess([], 255, "", "ssh: connect timed out"),
    )
    monkeypatch.setattr(
        stage,
        "_ssh_bytes",
        lambda _arguments, **_kwargs: subprocess.CompletedProcess(
            [], 255, b"", b"ssh: connect timed out"
        ),
    )

    with pytest.raises(StateUnavailable):
        stage.list_workspace_files()
    with pytest.raises(StateUnavailable):
        stage.read_workspace_text("validator-request.json")
    with pytest.raises(StateUnavailable):
        stage.write_workspace_text("validator-response.json", "{}")
    with pytest.raises(StateUnavailable):
        stage.remove_workspace_file("patch.json")


def test_remote_stage_close_removes_read_only_trees_and_verifies_absence(monkeypatch) -> None:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    projection = root / "inputs" / "conversations"
    projection.mkdir(parents=True)
    copied = projection / "conversation-0000.jsonl"
    copied.write_text("large transcript", encoding="utf-8")
    copied.chmod(0o400)
    projection.chmod(0o500)
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: subprocess.run(arguments, capture_output=True, text=True, check=False),
    )

    assert stage.close() is True
    assert not root.exists()
    assert stage.root is None


def test_remote_stage_close_keeps_root_when_deletion_failed(monkeypatch) -> None:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda _arguments: subprocess.CompletedProcess([], 1, "", "still present"),
    )

    try:
        assert stage.close() is False
        assert root.exists()
        assert stage.root == PurePosixPath(str(root))
    finally:
        shutil.rmtree(root)


def test_remote_stage_sweeper_uses_read_only_tree_cleanup(monkeypatch) -> None:
    stage = RemoteRunStage("research.example")
    calls: list[list[str]] = []

    def fake_ssh(arguments):
        calls.append(arguments)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(stage, "_ssh", fake_ssh)

    stage.sweep(retain_days=7)

    assert calls[0][:2] == ["python3", "-c"]
    assert "make_writable" in calls[0][2]
    assert "remove_tree(target)" in calls[0][2]


def test_remote_stage_artifact_operations_are_exact_and_binary(monkeypatch) -> None:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    (root / "workspace").mkdir()
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: subprocess.run(arguments, capture_output=True, text=True, check=False),
    )
    monkeypatch.setattr(
        stage,
        "_ssh_bytes",
        lambda arguments: subprocess.run(arguments, capture_output=True, check=False),
    )
    try:
        directory = Path(str(stage.prepare_artifact_directory("logical-turn", reuse=False)))
        payload = b"\x89PNG\r\n\x1a\n\x00\xffbinary"
        (directory / "plot.png").write_bytes(payload)
        (directory / "linked.png").symlink_to(directory / "plot.png")
        (directory / "nested").mkdir()
        (directory / "nested" / "hidden.png").write_bytes(payload)

        assert stage.list_artifact_files("logical-turn") == [("plot.png", len(payload))]
        assert stage.read_artifact_bytes("logical-turn", "plot.png", max_bytes=1024) == payload
        with pytest.raises(ValueError, match="plain base name"):
            stage.read_artifact_bytes("logical-turn", "../plot.png", max_bytes=1024)
    finally:
        shutil.rmtree(root)


def test_remote_stage_resume_rejects_symlinked_artifact_scope(monkeypatch) -> None:
    root = Path(tempfile.mkdtemp(prefix="rcp-run.", dir="/tmp"))
    workspace = root / "workspace"
    turns = workspace / "turns"
    outside = root / "outside"
    turns.mkdir(parents=True)
    (outside / "artifacts").mkdir(parents=True)
    (turns / "logical-turn").symlink_to(outside, target_is_directory=True)
    stage = RemoteRunStage("research.example")
    stage.root = PurePosixPath(str(root))
    monkeypatch.setattr(
        stage,
        "_ssh",
        lambda arguments: subprocess.run(arguments, capture_output=True, text=True, check=False),
    )
    try:
        with pytest.raises(StateUnavailable, match="saved artifact directory"):
            stage.prepare_artifact_directory("logical-turn", reuse=True)
    finally:
        shutil.rmtree(root)
