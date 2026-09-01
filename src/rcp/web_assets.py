from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from rcp.limits import (
    WEB_BUILD_POLL_INTERVAL_SECONDS,
    WEB_BUILD_STOP_TIMEOUT_SECONDS,
    WEB_BUILD_TIMEOUT_SECONDS,
)

WEB_ASSETS_ENV = "RCP_WEB_DIST_DIR"
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
WebAssetMode = Literal["source", "prebuilt"]


class WebBuildError(RuntimeError):
    pass


def web_dist_path() -> Path:
    """Resolve the one bundle served by both source and packaged backends."""
    override = os.environ.get(WEB_ASSETS_ENV)
    if override:
        return Path(override).expanduser().resolve()

    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root) / "rcp" / "web_dist"

    packaged = Path(__file__).resolve().parent / "web_dist"
    if packaged.exists():
        return packaged
    return WEB_DIR / "dist"


@contextmanager
def prepared_web_assets(*, watch: bool, mode: WebAssetMode = "source") -> Iterator[None]:
    """Build the served web bundle and optionally keep rebuilding it."""
    if mode == "prebuilt":
        index = web_dist_path() / "index.html"
        if not index.is_file():
            raise WebBuildError(f"The prebuilt RCP frontend is missing {index}.")
        yield
        return

    if not watch:
        _run_build()
        yield
        return

    original_stamp = _file_stamp(web_dist_path() / "index.html")
    watcher = _start_build_watcher()
    try:
        _wait_for_initial_build(watcher, original_stamp)
        yield
    finally:
        _stop_process_group(watcher)


def _npm_command(*args: str) -> list[str]:
    return ["npm", "--prefix", str(WEB_DIR), *args]


def _run_build() -> None:
    print("Building the RCP frontend...")
    try:
        subprocess.run(_npm_command("run", "build:clean"), check=True)
    except FileNotFoundError as exc:
        raise WebBuildError("Cannot build the RCP frontend because npm is not installed.") from exc
    except subprocess.CalledProcessError as exc:
        raise WebBuildError("The RCP frontend build failed; see the npm output above.") from exc


def _start_build_watcher() -> subprocess.Popen[bytes]:
    print("Building and watching the RCP frontend...")
    try:
        return subprocess.Popen(
            _npm_command("run", "build:clean", "--", "--watch"),
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise WebBuildError("Cannot build the RCP frontend because npm is not installed.") from exc


def _wait_for_initial_build(
    watcher: subprocess.Popen[bytes], original_stamp: tuple[int, int] | None
) -> None:
    index_path = web_dist_path() / "index.html"
    deadline = time.monotonic() + WEB_BUILD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        return_code = watcher.poll()
        if return_code is not None:
            raise WebBuildError(f"The RCP frontend build watcher exited with status {return_code}.")
        current_stamp = _file_stamp(index_path)
        if current_stamp is not None and current_stamp != original_stamp:
            return
        time.sleep(WEB_BUILD_POLL_INTERVAL_SECONDS)
    raise WebBuildError("Timed out waiting for the initial RCP frontend build.")


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=WEB_BUILD_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
