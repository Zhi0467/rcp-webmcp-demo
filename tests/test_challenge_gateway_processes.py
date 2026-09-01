from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from challenge.gateway_processes import (
    ChildProcessManager,
    GatewayProcessBusyError,
    GatewayProcessCapacityError,
    GatewayProcessError,
    ProcessLimits,
)
from challenge.gateway_state import DemoSessionRegistry


class FakeProcess:
    def __init__(self, session_id: str, pid: int) -> None:
        self.session_id = session_id
        self.pid = pid
        self.return_code: int | None = None
        self.terminated = 0
        self.killed = 0

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated += 1
        self.return_code = 0

    def kill(self) -> None:
        self.killed += 1
        self.return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.return_code is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.return_code


class Monotonic:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def sessions(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "source.txt").write_text("fixture", encoding="utf-8")
    registry = DemoSessionRegistry(tmp_path / "gateway", fixture)
    return (
        registry.get_or_create(None, client_key="a").session,
        registry.get_or_create(None, client_key="b").session,
    )


def _manager(
    *,
    active_tasks: dict[str, int],
    processes: list[FakeProcess],
    prepared: list[str],
    monotonic: Monotonic,
    max_processes: int = 2,
    idle_seconds: float = 60,
) -> ChildProcessManager:
    def launch(session, _port, _log_path):
        process = FakeProcess(session.session_id, 1_000 + len(processes))
        processes.append(process)
        return process

    def health(child):
        process = child.process
        return {
            "status": "ok",
            "pid": process.pid,
            "active_agent_tasks": active_tasks[process.session_id],
        }

    return ChildProcessManager(
        limits=ProcessLimits(
            max_processes=max_processes,
            idle_seconds=idle_seconds,
            startup_seconds=1,
            shutdown_seconds=1,
        ),
        prepare_session=lambda session: prepared.append(session.session_id),
        process_factory=launch,
        health_reader=health,
        port_allocator=lambda: 43123 + len(processes),
        monotonic=monotonic,
        sleep=lambda _seconds: None,
    )


def test_one_child_is_reused_until_its_last_response_releases(sessions) -> None:
    first, _second = sessions
    active = {first.session_id: 0}
    processes: list[FakeProcess] = []
    prepared: list[str] = []
    manager = _manager(
        active_tasks=active,
        processes=processes,
        prepared=prepared,
        monotonic=Monotonic(),
    )

    first_lease = manager.acquire(first)
    second_lease = manager.acquire(first)

    assert first_lease.child is second_lease.child
    assert first_lease.child.in_flight == 2
    assert len(processes) == 1
    assert prepared == [first.session_id]
    first_lease.release()
    second_lease.release()
    assert first_lease.child.in_flight == 0


def test_capacity_reuses_a_quiescent_slot_but_never_evicts_active_work(sessions) -> None:
    first, second = sessions
    active = {first.session_id: 0, second.session_id: 0}
    processes: list[FakeProcess] = []
    manager = _manager(
        active_tasks=active,
        processes=processes,
        prepared=[],
        monotonic=Monotonic(),
        max_processes=1,
    )
    manager.acquire(first).release()

    second_lease = manager.acquire(second)

    assert processes[0].terminated == 1
    assert second_lease.child.process is processes[1]
    second_lease.release()
    active[second.session_id] = 1
    with pytest.raises(GatewayProcessCapacityError, match="Existing sessions are preserved"):
        manager.acquire(first)
    assert processes[1].terminated == 0


def test_idle_reaper_checks_rcp_task_state_before_stopping(sessions) -> None:
    first, second = sessions
    active = {first.session_id: 0, second.session_id: 1}
    processes: list[FakeProcess] = []
    monotonic = Monotonic()
    manager = _manager(
        active_tasks=active,
        processes=processes,
        prepared=[],
        monotonic=monotonic,
        idle_seconds=10,
    )
    manager.acquire(first).release()
    manager.acquire(second).release()
    monotonic.value = 11

    assert manager.reap_idle() == [first.session_id]
    assert processes[0].terminated == 1
    assert processes[1].terminated == 0
    assert manager.active_session_ids() == {second.session_id}


def test_start_over_stop_refuses_requests_and_provider_work(sessions) -> None:
    first, _second = sessions
    active = {first.session_id: 0}
    processes: list[FakeProcess] = []
    manager = _manager(
        active_tasks=active,
        processes=processes,
        prepared=[],
        monotonic=Monotonic(),
    )
    lease = manager.acquire(first)

    with pytest.raises(GatewayProcessBusyError, match="request"):
        manager.stop_quiescent(first.session_id)
    lease.release()
    active[first.session_id] = 1
    with pytest.raises(GatewayProcessBusyError, match="research task"):
        manager.stop_quiescent(first.session_id)
    active[first.session_id] = 0
    assert manager.stop_quiescent(first.session_id) is True
    assert processes[0].terminated == 1


def test_startup_refuses_health_from_a_different_loopback_process(sessions) -> None:
    first, _second = sessions
    monotonic = Monotonic()
    process = FakeProcess(first.session_id, 1_000)

    def advance(seconds: float) -> None:
        monotonic.value += seconds

    manager = ChildProcessManager(
        limits=ProcessLimits(
            max_processes=1,
            idle_seconds=60,
            startup_seconds=0.1,
            shutdown_seconds=1,
        ),
        process_factory=lambda _session, _port, _log_path: process,
        health_reader=lambda _child: {
            "status": "ok",
            "pid": process.pid + 1,
            "active_agent_tasks": 0,
        },
        port_allocator=lambda: 43123,
        monotonic=monotonic,
        sleep=advance,
    )

    with pytest.raises(GatewayProcessError, match="did not become ready"):
        manager.acquire(first)

    assert process.terminated == 1
