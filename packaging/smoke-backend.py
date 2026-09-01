#!/usr/bin/env python3
"""Exercise the frozen desktop backend without a developer toolchain on PATH."""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LAUNCH_TIMEOUT_SECONDS = 30.0
HEALTH_TIMEOUT_SECONDS = 30.0
SHUTDOWN_TIMEOUT_SECONDS = 20.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "dist" / "rcp-backend",
    )
    return parser.parse_args()


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _environment(data_dir: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("RCP_")}
    environment["RCP_DATA_DIR"] = str(data_dir)
    # The release may see ordinary macOS utilities, but none of RCP's build or
    # language toolchains. A hidden npm/Python/uv dependency therefore fails.
    environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    for name in (
        "NODE_PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
    ):
        environment.pop(name, None)
    return environment


def _command(backend: Path, port: int) -> list[str]:
    return [
        str(backend),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--reuse-existing",
        "--machine-readable",
        "--owner",
        "desktop",
        "--web-assets",
        "prebuilt",
    ]


def _launch_outcome(process: subprocess.Popen[str]) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        if not selector.select(LAUNCH_TIMEOUT_SECONDS):
            raise RuntimeError("Timed out waiting for the backend launch outcome.")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        raise RuntimeError(
            f"The backend exited before reporting an outcome (status {process.poll()})."
        )
    try:
        outcome = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The backend emitted invalid launch JSON: {line!r}") from exc
    if not isinstance(outcome, dict):
        raise RuntimeError("The backend launch outcome is not an object.")
    return outcome


def _request(url: str) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=5.0) as response:
        return response.read(), response.headers.get_content_type()


def _wait_for_health(base_url: str) -> dict[str, Any]:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            body, _ = _request(f"{base_url}/api/health")
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("status") == "ok":
                return payload
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"The packaged backend never became healthy: {last_error}")


def _verify_web_bundle(base_url: str) -> None:
    index, content_type = _request(f"{base_url}/")
    if content_type != "text/html" or b'id="root"' not in index:
        raise RuntimeError("The packaged project index is not the built React application.")
    asset_paths = re.findall(rb'(?:src|href)="(/assets/[^\"]+)"', index)
    if not asset_paths:
        raise RuntimeError("The packaged project index does not reference built assets.")
    asset, asset_type = _request(f"{base_url}{asset_paths[0].decode('utf-8')}")
    if not asset or asset_type == "text/html":
        raise RuntimeError("A packaged frontend asset could not be loaded.")


def _stderr(process: subprocess.Popen[str]) -> str:
    if process.stderr is None or process.poll() is None:
        return ""
    return process.stderr.read().strip()


def main() -> None:
    backend = _arguments().backend.expanduser().resolve()
    if not backend.is_file() or not os.access(backend, os.X_OK):
        raise SystemExit(f"Packaged backend is not executable: {backend}")

    with tempfile.TemporaryDirectory(prefix="rcp-packaging-smoke-") as temporary:
        data_dir = Path(temporary) / "data"
        environment = _environment(data_dir)
        owner = subprocess.Popen(
            _command(backend, _unused_port()),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            outcome = _launch_outcome(owner)
            if outcome.get("outcome") != "owned" or outcome.get("owned") is not True:
                raise RuntimeError(f"First launch did not own the backend: {outcome}")
            base_url = outcome.get("base_url")
            instance_id = outcome.get("instance_id")
            version = outcome.get("version")
            if not all(isinstance(item, str) and item for item in (base_url, instance_id, version)):
                raise RuntimeError(f"First launch omitted its identity: {outcome}")

            health = _wait_for_health(base_url)
            expected = {
                "version": version,
                "instance_id": instance_id,
                "owner_kind": "desktop",
            }
            if any(health.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"Health identity disagrees with launch identity: {health}")
            metadata_path = data_dir / "rcp-server.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "The packaged backend did not publish ownership metadata."
                ) from exc
            owner_pid = metadata.get("pid")
            if (
                not isinstance(owner_pid, int)
                or isinstance(owner_pid, bool)
                or owner_pid <= 0
                or metadata.get("instance_id") != instance_id
            ):
                raise RuntimeError(f"The packaged backend published invalid ownership: {metadata}")
            if health.get("pid") != owner_pid or health.get("data_dir_id") != metadata.get(
                "data_dir_id"
            ):
                raise RuntimeError(
                    f"Health does not identify the metadata-owning process: {health}"
                )
            projects, project_type = _request(f"{base_url}/api/projects")
            if project_type != "application/json" or json.loads(projects) != []:
                raise RuntimeError("The packaged backend project index API is unavailable.")
            _verify_web_bundle(base_url)

            reused = subprocess.run(
                _command(backend, _unused_port()),
                env=environment,
                capture_output=True,
                text=True,
                timeout=LAUNCH_TIMEOUT_SECONDS,
                check=False,
            )
            if reused.returncode != 0:
                raise RuntimeError(
                    f"Second launch failed ({reused.returncode}): {reused.stderr.strip()}"
                )
            try:
                reused_outcome = json.loads(reused.stdout.strip())
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Second launch emitted invalid JSON: {reused.stdout!r}"
                ) from exc
            if reused_outcome != {**outcome, "outcome": "reused", "owned": False}:
                raise RuntimeError(
                    f"Second launch did not reuse the exact running backend: {reused_outcome}"
                )

            # A PyInstaller one-file executable has a supervising bootloader
            # process. The lock metadata names the Python server process, which
            # is the authority that the desktop Quit path must signal.
            os.kill(owner_pid, signal.SIGTERM)
            try:
                return_code = owner.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("The packaged backend did not shut down gracefully.") from exc
            shutdown_log = _stderr(owner)
            if (data_dir / "rcp-server.json").exists():
                raise RuntimeError("The packaged backend left stale ownership metadata.")
            expected_termination = return_code == 0 or (
                return_code == -signal.SIGTERM and "Application shutdown complete." in shutdown_log
            )
            if not expected_termination:
                raise RuntimeError(
                    f"The packaged backend exited with {return_code}: {shutdown_log}"
                )
        finally:
            if owner.poll() is None:
                os.killpg(owner.pid, signal.SIGKILL)
                owner.wait()

    print(
        json.dumps(
            {
                "backend": str(backend),
                "result": "passed",
                "toolchain_path": "/usr/bin:/bin:/usr/sbin:/sbin",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
