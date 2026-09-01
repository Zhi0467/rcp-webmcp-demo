"""Loopback-only RCP child ownership for the challenge gateway."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from challenge.gateway_state import DemoSession


class GatewayProcessError(RuntimeError):
    """Base class for an isolated child-process failure."""


class GatewayProcessCapacityError(GatewayProcessError):
    """Every active-process slot is doing work and must be preserved."""


class GatewayProcessBusyError(GatewayProcessError):
    """The exact child cannot be stopped without interrupting current work."""


class ChildProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass(frozen=True)
class ProcessLimits:
    """Initial values; W12 replaces them with measured Render limits."""

    max_processes: int = 16
    idle_seconds: float = 5 * 60
    startup_seconds: float = 20
    shutdown_seconds: float = 10


@dataclass
class ManagedChild:
    session_id: str
    process: ChildProcess
    port: int
    log_path: Path
    last_used: float
    in_flight: int = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class ChildLease:
    """Keep one child alive until a proxied response finishes or disconnects."""

    def __init__(self, manager: ChildProcessManager, child: ManagedChild) -> None:
        self._manager = manager
        self.child = child
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager.release(self.child.session_id)


ProcessFactory = Callable[[DemoSession, int, Path], ChildProcess]
HealthReader = Callable[[ManagedChild], dict[str, object]]
SessionPreparer = Callable[[DemoSession], None]


class ChildProcessManager:
    """Own at most one loopback RCP process for each active browser copy."""

    def __init__(
        self,
        *,
        limits: ProcessLimits | None = None,
        prepare_session: SessionPreparer | None = None,
        process_factory: ProcessFactory | None = None,
        health_reader: HealthReader | None = None,
        port_allocator: Callable[[], int] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.limits = limits or ProcessLimits()
        if (
            self.limits.max_processes < 1
            or self.limits.idle_seconds < 0
            or self.limits.startup_seconds <= 0
            or self.limits.shutdown_seconds <= 0
        ):
            raise ValueError("Process limits must be positive, except idle_seconds may be zero.")
        self._prepare_session = prepare_session or (lambda _session: None)
        self._process_factory = process_factory or _launch_rcp_child
        self._health_reader = health_reader or _read_child_health
        self._port_allocator = port_allocator or _allocate_loopback_port
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._children: dict[str, ManagedChild] = {}
        self._lock = threading.RLock()

    def acquire(self, session: DemoSession) -> ChildLease:
        """Return one process lease, starting or restarting the exact child once."""

        with self._lock:
            self._reap_idle_locked()
            child = self._children.get(session.session_id)
            if child is not None and child.process.poll() is not None:
                self._children.pop(session.session_id)
                child = None
            if child is None:
                self._make_process_room_locked()
                child = self._start_child_locked(session)
                self._children[session.session_id] = child
            child.in_flight += 1
            child.last_used = self._monotonic()
            return ChildLease(self, child)

    def release(self, session_id: str) -> None:
        with self._lock:
            child = self._children.get(session_id)
            if child is None:
                return
            if child.in_flight < 1:
                raise RuntimeError("The demo child lease was released more than once.")
            child.in_flight -= 1
            child.last_used = self._monotonic()

    def stop_quiescent(self, session_id: str) -> bool:
        """Stop one exact child only when no request or provider task is active."""

        with self._lock:
            child = self._children.get(session_id)
            if child is None:
                return False
            if child.in_flight:
                raise GatewayProcessBusyError(
                    "Wait for the current demo request to finish before starting over."
                )
            if not self._is_quiescent(child):
                raise GatewayProcessBusyError(
                    "Wait for the current research task to settle before starting over."
                )
            self._stop_child_locked(child)
            self._children.pop(session_id, None)
            return True

    def reap_idle(self) -> list[str]:
        """Stop request-idle children only after RCP reports no active task."""

        with self._lock:
            return self._reap_idle_locked()

    def close_all(self) -> None:
        """Gracefully stop children on gateway shutdown while preserving their data."""

        with self._lock:
            children = list(self._children.values())
            self._children.clear()
            for child in children:
                self._stop_child_locked(child)

    def active_session_ids(self) -> set[str]:
        with self._lock:
            return {
                session_id
                for session_id, child in self._children.items()
                if child.process.poll() is None
            }

    def child_count(self) -> int:
        with self._lock:
            return sum(child.process.poll() is None for child in self._children.values())

    def _make_process_room_locked(self) -> None:
        dead = [
            session_id
            for session_id, child in self._children.items()
            if child.process.poll() is not None
        ]
        for session_id in dead:
            self._children.pop(session_id, None)
        if len(self._children) < self.limits.max_processes:
            return
        candidates = sorted(
            (child for child in self._children.values() if child.in_flight == 0),
            key=lambda child: child.last_used,
        )
        for child in candidates:
            if self._is_quiescent(child):
                self._stop_child_locked(child)
                self._children.pop(child.session_id, None)
                return
        raise GatewayProcessCapacityError(
            "The demo is busy. Existing sessions are preserved; please retry shortly."
        )

    def _reap_idle_locked(self) -> list[str]:
        now = self._monotonic()
        stopped: list[str] = []
        for session_id, child in list(self._children.items()):
            if child.process.poll() is not None:
                self._children.pop(session_id, None)
                continue
            if child.in_flight or now - child.last_used < self.limits.idle_seconds:
                continue
            if not self._is_quiescent(child):
                continue
            self._stop_child_locked(child)
            self._children.pop(session_id, None)
            stopped.append(session_id)
        return stopped

    def _start_child_locked(self, session: DemoSession) -> ManagedChild:
        self._prepare_session(session)
        port = self._port_allocator()
        log_path = session.root / "rcp-child.log"
        process = self._process_factory(session, port, log_path)
        child = ManagedChild(
            session_id=session.session_id,
            process=process,
            port=port,
            log_path=log_path,
            last_used=self._monotonic(),
        )
        deadline = self._monotonic() + self.limits.startup_seconds
        last_error = "RCP did not report readiness."
        while self._monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise GatewayProcessError(
                    f"The isolated RCP process exited during startup ({return_code})."
                )
            try:
                health = self._health_reader(child)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = str(exc)
            else:
                if _health_belongs_to_child(health, child):
                    return child
                last_error = "RCP returned health for a different or invalid process."
            self._sleep(0.05)
        self._terminate_process(process)
        raise GatewayProcessError(f"The isolated RCP process did not become ready: {last_error}")

    def _is_quiescent(self, child: ManagedChild) -> bool:
        if child.process.poll() is not None:
            return True
        try:
            health = self._health_reader(child)
        except (OSError, ValueError, urllib.error.URLError):
            return False
        return _health_belongs_to_child(health, child) and health.get("active_agent_tasks") == 0

    def _stop_child_locked(self, child: ManagedChild) -> None:
        self._terminate_process(child.process)

    def _terminate_process(self, process: ChildProcess) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=self.limits.shutdown_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self.limits.shutdown_seconds)


def _launch_rcp_child(session: DemoSession, port: int, log_path: Path) -> ChildProcess:
    manifest = session.project_root / "state-repo" / ".research" / "manifest.toml"
    if not manifest.is_file():
        raise GatewayProcessError("The isolated demo manifest is missing.")
    if log_path.exists() and log_path.stat().st_size > 2 * 1024 * 1024:
        log_path.replace(log_path.with_suffix(".previous.log"))
    environment = os.environ.copy()
    environment["RCP_DATA_DIR"] = str(session.data_root)
    command = [
        sys.executable,
        "-m",
        "rcp",
        "serve",
        str(manifest),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--web-assets",
        "prebuilt",
        "--force",
        "--machine-readable",
    ]
    with log_path.open("ab") as log:
        return subprocess.Popen(  # noqa: S603 - fixed executable and server-owned paths
            command,
            cwd=session.root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def _read_child_health(child: ManagedChild) -> dict[str, object]:
    request = urllib.request.Request(
        f"{child.base_url}/api/health",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=0.75) as response:  # noqa: S310 - loopback URL
        if response.status != 200:
            raise ValueError(f"RCP health returned HTTP {response.status}.")
        payload = json.loads(response.read(64 * 1024))
    if not isinstance(payload, dict):
        raise ValueError("RCP health did not return an object.")
    return payload


def _health_belongs_to_child(health: dict[str, object], child: ManagedChild) -> bool:
    pid = health.get("pid")
    return (
        health.get("status") == "ok"
        and isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid == child.process.pid
    )


def _allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
        reserved.bind(("127.0.0.1", 0))
        return int(reserved.getsockname()[1])
