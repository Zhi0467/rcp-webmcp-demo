from __future__ import annotations

import signal
import subprocess

import pytest

from rcp import web_assets
from rcp.web_assets import WebBuildError, prepared_web_assets


def test_prepared_web_assets_builds_once_without_watch(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("rcp.web_assets._run_build", lambda: calls.append("build"))

    with prepared_web_assets(watch=False, mode="source"):
        calls.append("serve")

    assert calls == ["build", "serve"]


def test_source_server_starts_from_a_clean_frontend_build(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(web_assets.subprocess, "run", lambda command, check: calls.append(command))

    web_assets._run_build()

    assert calls == [web_assets._npm_command("run", "build:clean")]


def test_prepared_web_assets_stops_watcher_after_server_exits(monkeypatch) -> None:
    watcher = object()
    calls = []
    monkeypatch.setattr("rcp.web_assets._start_build_watcher", lambda: watcher)
    monkeypatch.setattr(
        "rcp.web_assets._wait_for_initial_build",
        lambda process, _stamp: calls.append(("ready", process)),
    )
    monkeypatch.setattr(
        "rcp.web_assets._stop_process_group", lambda process: calls.append(("stop", process))
    )

    with prepared_web_assets(watch=True, mode="source"):
        calls.append(("serve", watcher))

    assert calls == [("ready", watcher), ("serve", watcher), ("stop", watcher)]


def test_prebuilt_assets_never_invoke_npm(tmp_path, monkeypatch) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<main>RCP</main>", encoding="utf-8")
    monkeypatch.setattr("rcp.web_assets.web_dist_path", lambda: dist)
    monkeypatch.setattr("rcp.web_assets._run_build", lambda: pytest.fail("npm build was invoked"))
    monkeypatch.setattr(
        "rcp.web_assets._start_build_watcher",
        lambda: pytest.fail("npm watcher was invoked"),
    )

    with prepared_web_assets(watch=False, mode="prebuilt"):
        pass


def test_prebuilt_assets_fail_before_launch_when_bundle_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("rcp.web_assets.web_dist_path", lambda: tmp_path / "missing")

    with (
        pytest.raises(WebBuildError, match="prebuilt RCP frontend is missing"),
        prepared_web_assets(watch=False, mode="prebuilt"),
    ):
        pass


def test_initial_build_fails_when_watcher_exits(monkeypatch) -> None:
    class Watcher:
        def poll(self) -> int:
            return 7

    monkeypatch.setattr(web_assets.time, "monotonic", lambda: 0)

    with pytest.raises(WebBuildError, match="watcher exited with status 7"):
        web_assets._wait_for_initial_build(Watcher(), None)


def test_initial_build_is_ready_after_index_changes(tmp_path, monkeypatch) -> None:
    stamps = iter([(1, 10), (2, 20)])
    sleeps = []
    monkeypatch.setattr(web_assets, "web_dist_path", lambda: tmp_path)
    monkeypatch.setattr(web_assets, "_file_stamp", lambda _path: next(stamps))
    monkeypatch.setattr(web_assets.time, "monotonic", lambda: 0)
    monkeypatch.setattr(web_assets.time, "sleep", sleeps.append)

    class Watcher:
        def poll(self) -> None:
            return None

    web_assets._wait_for_initial_build(Watcher(), (1, 10))

    assert sleeps == [web_assets.WEB_BUILD_POLL_INTERVAL_SECONDS]


def test_initial_build_times_out_when_index_does_not_change(tmp_path, monkeypatch) -> None:
    times = iter([0, 0, web_assets.WEB_BUILD_TIMEOUT_SECONDS])
    sleeps = []
    monkeypatch.setattr(web_assets, "web_dist_path", lambda: tmp_path)
    monkeypatch.setattr(web_assets, "_file_stamp", lambda _path: (1, 10))
    monkeypatch.setattr(web_assets.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(web_assets.time, "sleep", sleeps.append)

    class Watcher:
        def poll(self) -> None:
            return None

    with pytest.raises(WebBuildError, match="Timed out waiting"):
        web_assets._wait_for_initial_build(Watcher(), (1, 10))

    assert sleeps == [web_assets.WEB_BUILD_POLL_INTERVAL_SECONDS]


def test_stop_process_group_waits_after_sigterm(monkeypatch) -> None:
    signals = []
    waits = []

    class Process:
        pid = 42

        def poll(self) -> None:
            return None

        def wait(self, *, timeout=None) -> None:
            waits.append(timeout)

    monkeypatch.setattr(web_assets.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    web_assets._stop_process_group(Process())

    assert signals == [(42, signal.SIGTERM)]
    assert waits == [web_assets.WEB_BUILD_STOP_TIMEOUT_SECONDS]


def test_stop_process_group_force_kills_after_timeout(monkeypatch) -> None:
    signals = []
    waits = []

    class Process:
        pid = 42

        def poll(self) -> None:
            return None

        def wait(self, *, timeout=None) -> None:
            waits.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="npm watcher", timeout=timeout)

    monkeypatch.setattr(web_assets.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    web_assets._stop_process_group(Process())

    assert signals == [(42, signal.SIGTERM), (42, signal.SIGKILL)]
    assert waits == [web_assets.WEB_BUILD_STOP_TIMEOUT_SECONDS, None]
