from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import rcp.server_ops.rehearsal as rehearsal_module
from rcp.background import StartupEffectFence
from rcp.server_ops.install import InstalledSystemServiceController
from rcp.server_ops.layout import ServerLayout
from rcp.server_ops.rehearsal import CandidateProjectVerification, StartupRecoveryReadModel
from rcp.server_ops.update_cutover import (
    RuntimeAdmissionGate,
    UpdateAdmissionClosed,
    UpdateCaptureBoundary,
    UpdateCutoverCoordinator,
    UpdateCutoverRefused,
    UpdateServiceCoordinator,
    _live_read_model_digest,
    active_update_operation,
    advance_update_operation,
    new_update_operation,
    publish_update_operation,
    read_update_operation,
    update_operation_needing_recovery,
)
from rcp.server_runtime import data_dir_identity

INSTALLATION_ID = "123e4567-e89b-42d3-a456-426614174000"
SPACE_ID = "123e4567-e89b-42d3-b456-426614174001"
BASE_INSTANCE_ID = "123e4567-e89b-42d3-b456-426614174002"
CAPTURE_ID = "123e4567-e89b-42d3-b456-426614174003"
BASE = "a" * 40
CANDIDATE = "b" * 40
REPAIRED_INSTANCE_ID = "123e4567-e89b-42d3-b456-426614174004"


def test_live_verification_rebuilds_a_never_opened_project_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = str(uuid.uuid4())
    graph = {"revision": 0, "nodes": {}, "edges": []}
    startup = StartupRecoveryReadModel(
        active_operation_ids=(),
        stopping_experiment_operation_ids=(),
        report_episode_ids=(),
        auto_research_recovery_operation_ids=(),
        active_watcher_ids=(),
    )
    final_path = tmp_path / "verified-candidate.json"
    _private_file(final_path, b"final rehearsal receipt\n")
    final = SimpleNamespace(
        startup_recovery=startup,
        projects=(
            CandidateProjectVerification(
                project_id=project_id,
                status="verified",
                revision=0,
                projection_sha256=rehearsal_module._canonical_sha256(graph),
            ),
        ),
        reads=("/api/health", f"/api/projects/{project_id}"),
    )
    monkeypatch.setattr(
        rehearsal_module,
        "read_verified_candidate_receipt",
        lambda _path, *, expected_uid: final,
    )
    opened: list[str] = []
    catalog = SimpleNamespace(
        cards=lambda: [{"id": project_id}],
        cached_snapshot_status=lambda _project_id: ("missing", None),
        open_snapshot=lambda observed_id: (
            opened.append(observed_id),
            {"graph": graph},
        ),
    )
    operational_reads: list[tuple[str, str]] = []
    store = SimpleNamespace(
        agent_tasks=lambda observed_id: operational_reads.append(("tasks", observed_id)),
        watchers=lambda observed_id: operational_reads.append(("watchers", observed_id)),
    )

    digest = _live_read_model_digest(
        SimpleNamespace(
            final_receipt_path=str(final_path),
            final_receipt_sha256=_sha256(final_path),
        ),
        SimpleNamespace(plan_startup_recovery=lambda: SimpleNamespace(as_dict=startup.model_dump)),
        catalog,
        store,
        os.geteuid(),
    )

    assert len(digest) == 64
    assert opened == [project_id]
    assert operational_reads == [("tasks", project_id), ("watchers", project_id)]


def test_runtime_admission_waits_for_entered_mutation_and_reopens() -> None:
    gate = RuntimeAdmissionGate()
    entered = threading.Event()
    release = threading.Event()

    def mutate() -> None:
        with gate.mutation("fixture mutation"):
            entered.set()
            release.wait(2)

    worker = threading.Thread(target=mutate)
    worker.start()
    assert entered.wait(1)
    closed = threading.Event()

    def close() -> None:
        gate.close_and_wait(timeout=2)
        closed.set()

    closer = threading.Thread(target=close)
    closer.start()
    with pytest.raises(UpdateAdmissionClosed, match="maintenance"):
        gate.require_open("new provider task")
    assert not closed.wait(0.05)
    release.set()
    worker.join(1)
    closer.join(1)
    assert closed.is_set()
    gate.reopen()
    gate.require_open("new provider task")


def test_running_service_closes_both_admission_gates_and_can_abort(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    built, preflight = _candidate_inputs(layout)
    operation = new_update_operation(
        operation_id=str(uuid.uuid4()),
        installation_id=INSTALLATION_ID,
        space_id=SPACE_ID,
        base_commit=BASE,
        candidate_commit=CANDIDATE,
        base_instance_id=BASE_INSTANCE_ID,
        base_process_pid=421,
        built_receipt_path=Path(built.receipt_path),
        built_receipt_sha256=_sha256(Path(built.receipt_path)),
        preflight_receipt_path=Path(preflight.receipt_path),
        preflight_receipt_sha256=_sha256(Path(preflight.receipt_path)),
        update_root=layout.update_checkpoints_root,
    )
    digest = publish_update_operation(
        operation,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    background = _BackgroundBoundary()
    runtime_owners: list[str] = []
    http_gate = RuntimeAdmissionGate()
    background_gate = RuntimeAdmissionGate()
    capture_root = layout.data_dir / "run-stage" / f"backup-{CAPTURE_ID}"
    sqlite_receipt = capture_root / "sqlite-capture.json"
    _private_file(sqlite_receipt, b"sqlite receipt\n")
    capture = SimpleNamespace(
        capture_id=CAPTURE_ID,
        instance_id=BASE_INSTANCE_ID,
        pid=421,
        data_dir_id=data_dir_identity(layout.data_dir),
        space_id=SPACE_ID,
        receipt_path=str(sqlite_receipt),
        receipt_sha256=_sha256(sqlite_receipt),
        snapshot_sha256="e" * 64,
        status="complete",
        project_count=0,
        uncaptured_project_count=0,
    )
    coordinator = UpdateServiceCoordinator(
        layout=layout,
        instance_metadata=SimpleNamespace(
            instance_id=BASE_INSTANCE_ID,
            pid=421,
            running_commit=BASE,
            data_dir_id=data_dir_identity(layout.data_dir),
        ),
        space_id=SPACE_ID,
        admission=http_gate,
        background_admission=background_gate,
        background=background,
        capture_sqlite=lambda: capture,
        catalog=SimpleNamespace(),
        store=SimpleNamespace(),
        pause_runtime_owners=lambda timeout: runtime_owners.append(f"pause:{timeout}"),
        resume_runtime_owners=lambda: runtime_owners.append("resume"),
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )

    closed, closed_digest, returned_capture = coordinator.enter_maintenance(
        operation_id=operation.operation_id,
        receipt_sha256=digest,
        timeout=1,
    )

    assert closed.state == "maintenance_closed"
    assert returned_capture is capture
    assert http_gate.closed and background_gate.closed
    assert background.calls == ["close", "idle", "idle"]
    assert runtime_owners == ["pause:1"]
    with pytest.raises(UpdateAdmissionClosed):
        background_gate.require_open("provider dispatch")

    aborted, _digest = coordinator.abort_before_switch(
        operation_id=operation.operation_id,
        receipt_sha256=closed_digest,
        timeout=1,
    )

    assert aborted.state == "aborted_before_switch"
    assert not http_gate.closed and not background_gate.closed
    assert background.calls == ["close", "idle", "idle", "accept"]
    assert runtime_owners == ["pause:1", "resume"]


@pytest.mark.parametrize("fail_candidate", [False, True])
def test_cutover_commits_or_loudly_restores_one_receipt(
    tmp_path: Path,
    fail_candidate: bool,
) -> None:
    layout = _layout(tmp_path)
    built, preflight = _candidate_inputs(layout)
    actions = _CutoverActions(layout, fail_candidate=fail_candidate)
    phases: list[str] = []

    outcome = UpdateCutoverCoordinator(
        layout=layout,
        actions=actions,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        progress=phases.append,
    ).run(built, preflight)

    receipt, digest = read_update_operation(
        outcome.receipt_path,
        expected_uid=os.geteuid(),
        expected_sha256=outcome.receipt_sha256,
    )
    assert digest == outcome.receipt_sha256
    assert (
        active_update_operation(
            layout.update_checkpoints_root,
            expected_uid=os.geteuid(),
        )
        is None
    )
    if not fail_candidate:
        assert outcome.operation_state == receipt.state == "committed"
        assert outcome.running_commit == CANDIDATE
        assert outcome.failure is None
        assert actions.calls == [
            "enter",
            "final_rehearsal",
            "checkpoint",
            "stop",
            f"switch:{BASE}->{CANDIDATE}",
            "start",
            f"control:{CANDIDATE}",
            "verify:candidate_starting",
            "release:candidate_verified",
        ]
        assert phases == [
            "maintenance_closed",
            "checkpoint_ready",
            "candidate_started",
            "candidate_verified",
        ]
    else:
        assert outcome.operation_state == receipt.state == "rolled_back"
        assert outcome.running_commit == BASE
        assert "candidate verification fixture failed" in (outcome.failure or "")
        assert actions.calls[-7:] == [
            "stop",
            "restore",
            f"switch:{CANDIDATE}->{BASE}",
            "start",
            f"control:{BASE}",
            "verify:old_release_starting",
            "release:old_release_verified",
        ]
        assert phases[-1] == "rolled_back"


def test_pre_switch_failure_reopens_old_process_without_switching(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    built, preflight = _candidate_inputs(layout)
    actions = _CutoverActions(layout, fail_final=True)

    with pytest.raises(UpdateCutoverRefused, match="final rehearsal fixture failed"):
        UpdateCutoverCoordinator(
            layout=layout,
            actions=actions,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        ).run(built, preflight)

    operation_path, receipt, _digest = next(
        item for item in _all_receipts(layout) if item[1].candidate_commit == CANDIDATE
    )
    assert operation_path == Path(receipt.receipt_path)
    assert receipt.state == "aborted_before_switch"
    assert actions.current == BASE
    assert not any(call.startswith("switch:") for call in actions.calls)
    assert actions.calls[-2:] == [f"control:{BASE}", "abort:maintenance_closed"]


def test_deferred_startup_failure_restarts_the_committed_candidate_normally(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    built, preflight = _candidate_inputs(layout)
    actions = _CutoverActions(layout, fail_release=True)

    outcome = UpdateCutoverCoordinator(
        layout=layout,
        actions=actions,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    ).run(built, preflight)

    receipt, _digest = read_update_operation(
        outcome.receipt_path,
        expected_uid=os.geteuid(),
        expected_sha256=outcome.receipt_sha256,
    )
    assert outcome.operation_state == receipt.state == "committed"
    assert receipt.failure is None
    assert receipt.runtime_failure is None
    assert receipt.candidate_instance_id == REPAIRED_INSTANCE_ID
    assert actions.calls[-5:] == [
        "stop",
        "start",
        f"control:{CANDIDATE}",
        f"probe:{CANDIDATE}",
        f"normal:{CANDIDATE}",
    ]


def test_failed_committed_restart_stays_loud_and_is_recoverable(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    built, preflight = _candidate_inputs(layout)
    actions = _CutoverActions(layout, fail_release=True, fail_repair=True)
    coordinator = UpdateCutoverCoordinator(
        layout=layout,
        actions=actions,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )

    with pytest.raises(UpdateCutoverRefused, match="kept the service stopped"):
        coordinator.run(built, preflight)

    pending = update_operation_needing_recovery(
        layout.update_checkpoints_root,
        expected_uid=os.geteuid(),
    )
    assert pending is not None
    _path, failed, failed_digest = pending
    assert failed.state == "committed"
    assert "normal restart fixture failed" in (failed.runtime_failure or "")
    assert actions.calls[-1] == "stop"

    actions.fail_repair = False
    recovered, recovered_digest = coordinator.repair_committed(failed, failed_digest)

    assert recovered.state == "committed"
    assert recovered.failure is None
    assert (
        update_operation_needing_recovery(
            layout.update_checkpoints_root,
            expected_uid=os.geteuid(),
        )
        is None
    )
    assert (
        read_update_operation(
            Path(recovered.receipt_path),
            expected_uid=os.geteuid(),
            expected_sha256=recovered_digest,
        )[0]
        == recovered
    )


def test_service_records_the_point_of_no_return_before_deferred_startup(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    actions = _CutoverActions(layout)
    operation, digest = _checkpoint_ready_operation(layout, actions)
    path = Path(operation.receipt_path)
    operation, digest = advance_update_operation(
        path,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        expected_sha256=digest,
        state="candidate_starting",
    )
    candidate_instance = str(uuid.uuid4())
    operation, digest = advance_update_operation(
        path,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        expected_sha256=digest,
        state="candidate_verified",
        update={
            "candidate_instance_id": candidate_instance,
            "candidate_process_pid": 422,
        },
    )
    runtime_started = threading.Event()
    runtime_started.set()
    fence = StartupEffectFence("selected candidate")
    coordinator = UpdateServiceCoordinator(
        layout=layout,
        instance_metadata=SimpleNamespace(
            instance_id=candidate_instance,
            pid=422,
            running_commit=CANDIDATE,
            data_dir_id=data_dir_identity(layout.data_dir),
        ),
        space_id=SPACE_ID,
        admission=RuntimeAdmissionGate(closed=True),
        background_admission=RuntimeAdmissionGate(closed=True),
        background=_BackgroundBoundary(),
        capture_sqlite=lambda: None,
        catalog=SimpleNamespace(),
        store=SimpleNamespace(),
        startup_effect_fence=fence,
        runtime_started=runtime_started,
        runtime_error=lambda: "deferred startup fixture failed",
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )

    with pytest.raises(UpdateCutoverRefused, match="deferred runtime startup"):
        coordinator.release_fence(
            operation_id=operation.operation_id,
            receipt_sha256=digest,
            timeout=1,
        )

    interrupted, _digest = read_update_operation(path, expected_uid=os.geteuid())
    assert interrupted.state == "candidate_reopening"
    assert "deferred startup" in (interrupted.runtime_failure or "")
    assert (
        update_operation_needing_recovery(
            layout.update_checkpoints_root,
            expected_uid=os.geteuid(),
        )
        is not None
    )


def test_failed_rollback_restart_repairs_the_selected_old_release(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    built, preflight = _candidate_inputs(layout)
    actions = _CutoverActions(
        layout,
        fail_candidate=True,
        fail_old_release=True,
    )

    outcome = UpdateCutoverCoordinator(
        layout=layout,
        actions=actions,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    ).run(built, preflight)

    receipt, _digest = read_update_operation(
        outcome.receipt_path,
        expected_uid=os.geteuid(),
        expected_sha256=outcome.receipt_sha256,
    )
    assert outcome.operation_state == receipt.state == "rolled_back"
    assert "candidate verification fixture failed" in (receipt.failure or "")
    assert receipt.runtime_failure is None
    assert receipt.restored_instance_id == REPAIRED_INSTANCE_ID
    assert actions.calls[-5:] == [
        "stop",
        "start",
        f"control:{BASE}",
        f"probe:{BASE}",
        f"normal:{BASE}",
    ]


def test_failed_selected_old_release_remains_recoverable(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    built, preflight = _candidate_inputs(layout)
    actions = _CutoverActions(
        layout,
        fail_candidate=True,
        fail_old_release=True,
        fail_repair=True,
    )
    coordinator = UpdateCutoverCoordinator(
        layout=layout,
        actions=actions,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )

    with pytest.raises(UpdateCutoverRefused, match="kept the service stopped"):
        coordinator.run(built, preflight)

    pending = update_operation_needing_recovery(
        layout.update_checkpoints_root,
        expected_uid=os.geteuid(),
    )
    assert pending is not None
    _path, failed, failed_digest = pending
    assert failed.state == "rolled_back"
    assert failed.failure is not None
    assert failed.runtime_failure is not None
    actions.fail_repair = False

    repaired, _digest = coordinator.repair_selected_release(failed, failed_digest)

    assert repaired.state == "rolled_back"
    assert repaired.failure is not None
    assert repaired.runtime_failure is None


def test_recovery_finishes_a_candidate_switched_before_its_receipt_advanced(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    actions = _CutoverActions(layout)
    operation, digest = _checkpoint_ready_operation(layout, actions)
    actions.current = CANDIDATE
    actions.calls.clear()

    recovered, recovered_digest = UpdateCutoverCoordinator(
        layout=layout,
        actions=actions,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    ).recover(operation, digest)

    assert recovered.state == "committed"
    assert (
        read_update_operation(
            Path(recovered.receipt_path),
            expected_uid=os.geteuid(),
            expected_sha256=recovered_digest,
        )[0]
        == recovered
    )
    assert actions.calls == [
        "stop",
        "start",
        f"control:{CANDIDATE}",
        "verify:candidate_starting",
        "release:candidate_verified",
    ]


def test_candidate_recovery_repairs_a_failure_after_commit(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    actions = _CutoverActions(layout, fail_release=True)
    operation, digest = _checkpoint_ready_operation(layout, actions)
    actions.current = CANDIDATE
    actions.calls.clear()

    recovered, recovered_digest = UpdateCutoverCoordinator(
        layout=layout,
        actions=actions,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    ).recover(operation, digest)

    assert recovered.state == "committed"
    assert recovered.failure is None
    assert (
        read_update_operation(
            Path(recovered.receipt_path),
            expected_uid=os.geteuid(),
            expected_sha256=recovered_digest,
        )[0]
        == recovered
    )
    assert actions.calls[-5:] == [
        "stop",
        "start",
        f"control:{CANDIDATE}",
        f"probe:{CANDIDATE}",
        f"normal:{CANDIDATE}",
    ]


@pytest.mark.parametrize(
    "interrupted_state",
    ["rollback_restoring", "old_release_starting", "old_release_verified", "repair_required"],
)
def test_recovery_resumes_each_durable_rollback_phase(
    tmp_path: Path,
    interrupted_state: str,
) -> None:
    layout = _layout(tmp_path)
    actions = _CutoverActions(layout)
    operation, digest = _checkpoint_ready_operation(layout, actions)
    path = Path(operation.receipt_path)
    operation, digest = advance_update_operation(
        path,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        expected_sha256=digest,
        state="candidate_starting",
    )
    if interrupted_state == "repair_required":
        operation, digest = advance_update_operation(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_sha256=digest,
            state="repair_required",
            update={"failure": "forced candidate failure"},
        )
        actions.current = CANDIDATE
    else:
        operation, digest = advance_update_operation(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_sha256=digest,
            state="rollback_restoring",
            update={"failure": "forced candidate failure"},
        )
        actions.current = CANDIDATE
        if interrupted_state in {"old_release_starting", "old_release_verified"}:
            actions.current = BASE
            operation, digest = advance_update_operation(
                path,
                expected_uid=os.geteuid(),
                expected_gid=os.getegid(),
                expected_sha256=digest,
                state="old_release_starting",
            )
        if interrupted_state == "old_release_verified":
            operation, digest = advance_update_operation(
                path,
                expected_uid=os.geteuid(),
                expected_gid=os.getegid(),
                expected_sha256=digest,
                state="old_release_verified",
                update={
                    "restored_instance_id": str(uuid.uuid4()),
                    "restored_process_pid": 423,
                },
            )
    actions.calls.clear()

    recovered, recovered_digest = UpdateCutoverCoordinator(
        layout=layout,
        actions=actions,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    ).recover(operation, digest)

    assert recovered.state == "rolled_back"
    assert recovered.failure == "forced candidate failure"
    assert (
        read_update_operation(
            path,
            expected_uid=os.geteuid(),
            expected_sha256=recovered_digest,
        )[0]
        == recovered
    )
    assert actions.current == BASE


def test_system_service_seam_proves_stop_pointer_switch_and_start(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    current = layout.current_release
    current.parent.mkdir(parents=True, exist_ok=True)
    current.symlink_to(layout.release_dir(BASE))
    state = {"active": "active", "pid": "41"}
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[1] == "stop":
            state.update(active="inactive", pid="0")
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[1] == "start":
            state.update(active="active", pid="84")
            return subprocess.CompletedProcess(argv, 0, "", "")
        value = state["active"] if "ActiveState" in argv[2] else state["pid"]
        return subprocess.CompletedProcess(argv, 0, value + "\n", "")

    controller = InstalledSystemServiceController(
        layout,
        runner=runner,
        root_identity=(os.geteuid(), os.getegid()),
    )
    controller.stop()
    controller.switch_current(
        expected=layout.release_dir(BASE),
        target=layout.release_dir(CANDIDATE),
    )
    assert controller.start() == 84
    assert controller.current_release() == layout.release_dir(CANDIDATE)
    assert calls[0] == ("systemctl", "stop", layout.service_unit_name)
    assert calls[-3] == ("systemctl", "start", layout.service_unit_name)


def test_system_service_seam_proves_restore_stop_and_disable(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    state = {"active": "active", "pid": "41", "enabled": "enabled"}

    def runner(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        if argv[1:3] == ("disable", "--now"):
            state.update(active="inactive", pid="0", enabled="disabled")
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "ActiveState" in argv[2]:
            value = state["active"]
        elif "MainPID" in argv[2]:
            value = state["pid"]
        else:
            value = state["enabled"]
        return subprocess.CompletedProcess(argv, 0, value + "\n", "")

    controller = InstalledSystemServiceController(
        layout,
        runner=runner,
        root_identity=(os.geteuid(), os.getegid()),
    )

    controller.fence_stopped_disabled()

    assert state == {"active": "inactive", "pid": "0", "enabled": "disabled"}


def test_system_service_enables_an_already_running_restore_without_restart(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    state = {"active": "active", "pid": "41", "enabled": "disabled"}
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[1] == "enable":
            state["enabled"] = "enabled"
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "ActiveState" in argv[2]:
            value = state["active"]
        elif "MainPID" in argv[2]:
            value = state["pid"]
        else:
            value = state["enabled"]
        return subprocess.CompletedProcess(argv, 0, value + "\n", "")

    controller = InstalledSystemServiceController(
        layout,
        runner=runner,
        root_identity=(os.geteuid(), os.getegid()),
    )

    assert controller.enable() == 41
    assert calls[0] == ("systemctl", "enable", layout.service_unit_name)
    assert all("--now" not in call for call in calls)


class _CutoverActions:
    def __init__(
        self,
        layout: ServerLayout,
        *,
        fail_candidate: bool = False,
        fail_final: bool = False,
        fail_release: bool = False,
        fail_repair: bool = False,
        fail_old_release: bool = False,
    ) -> None:
        self.layout = layout
        self.fail_candidate = fail_candidate
        self.fail_final = fail_final
        self.fail_release = fail_release
        self.fail_repair = fail_repair
        self.fail_old_release = fail_old_release
        self.calls: list[str] = []
        self.current = BASE

    def enter_maintenance(self, *, operation_id: str, receipt_sha256: str):
        self.calls.append("enter")
        path, receipt, digest = self._operation(operation_id, receipt_sha256)
        capture_root = self.layout.data_dir / "run-stage" / f"backup-{CAPTURE_ID}"
        _private_dir(capture_root)
        sqlite_receipt = capture_root / "sqlite-capture.json"
        _private_file(sqlite_receipt, b"sqlite receipt\n")
        capture = UpdateCaptureBoundary(
            capture_id=CAPTURE_ID,
            instance_id=BASE_INSTANCE_ID,
            process_pid=421,
            data_dir_id="d" * 64,
            space_id=SPACE_ID,
            sqlite_receipt_path=str(sqlite_receipt),
            sqlite_receipt_sha256=_sha256(sqlite_receipt),
            sqlite_snapshot_sha256="e" * 64,
            status="complete",
            project_count=0,
            uncaptured_project_count=0,
        )
        receipt, digest = advance_update_operation(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_sha256=digest,
            state="maintenance_closed",
            update={"capture": capture},
        )
        return _result(receipt, digest)

    def final_rehearsal(self, operation, receipt_sha256: str):
        del receipt_sha256
        self.calls.append("final_rehearsal")
        if self.fail_final:
            raise RuntimeError("final rehearsal fixture failed")
        assert operation.capture is not None
        project_receipt = Path(operation.capture.sqlite_receipt_path).parent / "project-files.json"
        _private_file(project_receipt, b"project receipt\n")
        final_path = (
            self.layout.update_checkpoints_root
            / f"verified-candidate-{CANDIDATE}-{CAPTURE_ID}.json"
        )
        _private_file(final_path, b"final receipt\n")
        return SimpleNamespace(
            installation_id=INSTALLATION_ID,
            space_id=SPACE_ID,
            candidate_commit=CANDIDATE,
            base_running_commit=BASE,
            base_instance_id=BASE_INSTANCE_ID,
            base_process_pid=421,
            capture_id=CAPTURE_ID,
            sqlite_snapshot_sha256="e" * 64,
            built_receipt_sha256=operation.built_receipt_sha256,
            project_capture_sha256=_sha256(project_receipt),
            receipt_path=str(final_path),
        )

    def create_checkpoint(self, _final, **_boundary):
        self.calls.append("checkpoint")
        root = self.layout.update_checkpoints_root / f"checkpoint-{CANDIDATE}-{uuid.uuid4().hex}"
        _private_dir(root)
        manifest = root / "manifest.json"
        _private_file(manifest, b"checkpoint manifest\n")
        return SimpleNamespace(manifest_path=str(manifest))

    def stop_service(self) -> None:
        self.calls.append("stop")

    def switch_current(self, *, expected: Path, target: Path) -> None:
        assert expected.name == self.current
        self.calls.append(f"switch:{expected.name}->{target.name}")
        self.current = target.name

    def start_service(self) -> int:
        self.calls.append("start")
        return 422

    def control_for_running(self, commit: str):
        assert self.current == commit
        self.calls.append(f"control:{commit}")
        return _Control(self, commit)

    def restore_checkpoint(
        self,
        _checkpoint_path: Path,
        checkpoint_sha256: str,
    ) -> None:
        assert checkpoint_sha256
        self.calls.append("restore")

    def current_release(self) -> Path:
        return self.layout.release_dir(self.current)

    def _operation(self, operation_id: str, digest: str | None = None):
        active = active_update_operation(
            self.layout.update_checkpoints_root,
            expected_uid=os.geteuid(),
        )
        assert active is not None
        path, receipt, observed = active
        assert receipt.operation_id == operation_id
        if digest is not None:
            assert observed == digest
        return path, receipt, observed


class _BackgroundBoundary:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def close_watcher_notifications(self) -> None:
        self.calls.append("close")

    def runtime_is_idle(self) -> bool:
        self.calls.append("idle")
        return True

    def accept_watcher_notifications(self) -> None:
        self.calls.append("accept")


class _Control:
    def __init__(self, actions: _CutoverActions, commit: str) -> None:
        self.actions = actions
        self.commit = commit

    def verify_update_candidate(self, *, operation_id: str, receipt_sha256: str):
        path, receipt, digest = self.actions._operation(operation_id, receipt_sha256)
        self.actions.calls.append(f"verify:{receipt.state}")
        if self.commit == CANDIDATE and self.actions.fail_candidate:
            raise RuntimeError("candidate verification fixture failed")
        state = "candidate_verified" if self.commit == CANDIDATE else "old_release_verified"
        update = (
            {"candidate_instance_id": str(uuid.uuid4()), "candidate_process_pid": 422}
            if self.commit == CANDIDATE
            else {"restored_instance_id": str(uuid.uuid4()), "restored_process_pid": 423}
        )
        receipt, digest = advance_update_operation(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_sha256=digest,
            state=state,
            update=update,
        )
        return _result(receipt, digest)

    def probe(self):
        self.actions.calls.append(f"probe:{self.commit}")
        if self.actions.fail_repair:
            raise RuntimeError("normal restart fixture failed")
        self.actions.calls.append(f"normal:{self.commit}")
        return SimpleNamespace(
            instance_id=REPAIRED_INSTANCE_ID,
            pid=424,
            data_dir_id="d" * 64,
            space_id=SPACE_ID,
        )

    def release_update_fence(self, *, operation_id: str, receipt_sha256: str):
        path, receipt, digest = self.actions._operation(operation_id, receipt_sha256)
        self.actions.calls.append(f"release:{receipt.state}")
        reopening = (
            "candidate_reopening"
            if receipt.state == "candidate_verified"
            else "old_release_reopening"
        )
        terminal = "committed" if receipt.state == "candidate_verified" else "rolled_back"
        receipt, digest = advance_update_operation(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_sha256=digest,
            state=reopening,
        )
        if (self.commit == CANDIDATE and self.actions.fail_release) or (
            self.commit == BASE and self.actions.fail_old_release
        ):
            raise RuntimeError("deferred runtime startup fixture failed")
        receipt, digest = advance_update_operation(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_sha256=digest,
            state=terminal,
        )
        return _result(receipt, digest)

    def abort_update_maintenance(self, *, operation_id: str, receipt_sha256: str):
        path, receipt, digest = self.actions._operation(operation_id, receipt_sha256)
        self.actions.calls.append(f"abort:{receipt.state}")
        receipt, digest = advance_update_operation(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_sha256=digest,
            state="aborted_before_switch",
        )
        return _result(receipt, digest)


def _candidate_inputs(layout: ServerLayout):
    built_path = layout.update_checkpoints_root / f"built-candidate-{CANDIDATE}.json"
    _private_file(built_path, b"built receipt\n")
    preflight_path = (
        layout.update_checkpoints_root
        / f"verified-candidate-{CANDIDATE}-123e4567-e89b-42d3-b456-426614174099.json"
    )
    _private_file(preflight_path, b"preflight receipt\n")
    built = SimpleNamespace(
        installation_id=INSTALLATION_ID,
        base_current_commit=BASE,
        base_running_commit=BASE,
        base_instance_id=BASE_INSTANCE_ID,
        base_process_pid=421,
        candidate_commit=CANDIDATE,
        release_path=str(layout.release_dir(CANDIDATE)),
        receipt_path=str(built_path),
    )
    preflight = SimpleNamespace(
        installation_id=INSTALLATION_ID,
        space_id=SPACE_ID,
        candidate_commit=CANDIDATE,
        base_current_commit=BASE,
        base_running_commit=BASE,
        base_instance_id=BASE_INSTANCE_ID,
        base_process_pid=421,
        built_receipt_path=str(built_path),
        receipt_path=str(preflight_path),
    )
    return built, preflight


def _checkpoint_ready_operation(
    layout: ServerLayout,
    actions: _CutoverActions,
):
    built, preflight = _candidate_inputs(layout)
    operation = new_update_operation(
        operation_id=str(uuid.uuid4()),
        installation_id=INSTALLATION_ID,
        space_id=SPACE_ID,
        base_commit=BASE,
        candidate_commit=CANDIDATE,
        base_instance_id=BASE_INSTANCE_ID,
        base_process_pid=421,
        built_receipt_path=Path(built.receipt_path),
        built_receipt_sha256=_sha256(Path(built.receipt_path)),
        preflight_receipt_path=Path(preflight.receipt_path),
        preflight_receipt_sha256=_sha256(Path(preflight.receipt_path)),
        update_root=layout.update_checkpoints_root,
    )
    digest = publish_update_operation(
        operation,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    result = actions.enter_maintenance(
        operation_id=operation.operation_id,
        receipt_sha256=digest,
    )
    operation, digest = read_update_operation(
        Path(operation.receipt_path),
        expected_uid=os.geteuid(),
        expected_sha256=result.receipt_sha256,
    )
    final = actions.final_rehearsal(operation, digest)
    assert operation.capture is not None
    final_path = Path(final.receipt_path)
    project_path = Path(operation.capture.sqlite_receipt_path).parent / "project-files.json"
    operation, digest = advance_update_operation(
        Path(operation.receipt_path),
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        expected_sha256=digest,
        state="maintenance_closed",
        update={
            "final_receipt_path": str(final_path),
            "final_receipt_sha256": _sha256(final_path),
            "project_receipt_path": str(project_path),
            "project_receipt_sha256": _sha256(project_path),
        },
    )
    checkpoint = actions.create_checkpoint(final)
    checkpoint_path = Path(checkpoint.manifest_path)
    return advance_update_operation(
        Path(operation.receipt_path),
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        expected_sha256=digest,
        state="checkpoint_ready",
        update={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
    )


def _layout(tmp_path: Path) -> ServerLayout:
    home = tmp_path / "home"
    root = home / "server"
    etc = tmp_path / "etc"
    run = tmp_path / "run"
    for path in (
        root / "source",
        root / "releases" / BASE,
        root / "releases" / CANDIDATE,
        root / "data",
        root / "projects",
        root / "credentials",
        root / "update-checkpoints",
        root / "restore-operations",
        home / ".codex",
        home / ".claude",
        home / ".ssh",
        etc,
        run,
    ):
        path.mkdir(parents=True, exist_ok=True)
    (root / "update-checkpoints").chmod(0o700)
    return ServerLayout(
        service_account="rcp",
        service_home=home,
        server_root=root,
        source_checkout=root / "source",
        releases_root=root / "releases",
        data_dir=root / "data",
        projects_root=root / "projects",
        credentials_root=root / "credentials",
        update_checkpoints_root=root / "update-checkpoints",
        restore_operations_root=root / "restore-operations",
        codex_state_root=home / ".codex",
        claude_state_root=home / ".claude",
        ssh_state_root=home / ".ssh",
        config_path=etc / "server.toml",
        current_release=etc / "current",
        runtime_dir=run,
        control_socket=run / "control.sock",
        cli_wrapper=tmp_path / "bin" / "rcp",
        systemd_unit=etc / "rcp.service",
        service_unit_name="rcp.service",
    )


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def _private_file(path: Path, payload: bytes) -> None:
    _private_dir(path.parent)
    path.write_bytes(payload)
    path.chmod(0o600)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(receipt, digest: str):
    return SimpleNamespace(
        operation_id=receipt.operation_id,
        operation_state=receipt.state,
        receipt_sha256=digest,
    )


def _all_receipts(layout: ServerLayout):
    from rcp.server_ops.update_cutover import update_operation_receipts

    return update_operation_receipts(
        layout.update_checkpoints_root,
        expected_uid=os.geteuid(),
    )
