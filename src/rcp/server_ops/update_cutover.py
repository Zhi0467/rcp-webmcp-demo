"""Crash-visible maintenance and cutover state for source-built server updates.

The running service owns the short admission boundary.  The root coordinator
owns only the service stop/start and the atomic ``current`` pointer switch.  A
small service-owned receipt bridges those two processes so either release can
start fenced and a later CLI invocation can resume an interrupted operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_runtime import data_dir_identity

UPDATE_OPERATION_SCHEMA_VERSION = 1
UPDATE_OPERATION_FILE_MODE = 0o600
UPDATE_OPERATION_DIRECTORY_MODE = 0o700
UPDATE_OPERATION_MAX_BYTES = 128 * 1024

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OPERATION_NAME = re.compile(r"update-operation-([0-9a-f]{32})\.json")

UpdateOperationState = Literal[
    "maintenance_closing",
    "maintenance_closed",
    "checkpoint_ready",
    "candidate_starting",
    "candidate_verified",
    "candidate_reopening",
    "rollback_restoring",
    "old_release_starting",
    "old_release_verified",
    "old_release_reopening",
    "repair_required",
    "committed",
    "rolled_back",
    "aborted_before_switch",
]

TERMINAL_UPDATE_STATES: frozenset[UpdateOperationState] = frozenset(
    {"committed", "rolled_back", "aborted_before_switch"}
)
_BASE_RUNNING_STATES: frozenset[UpdateOperationState] = frozenset(
    {
        "maintenance_closing",
        "maintenance_closed",
        "checkpoint_ready",
        "old_release_starting",
        "old_release_verified",
        "old_release_reopening",
        "rolled_back",
        "aborted_before_switch",
    }
)
_CANDIDATE_RUNNING_STATES: frozenset[UpdateOperationState] = frozenset(
    {"candidate_starting", "candidate_verified", "candidate_reopening", "committed"}
)
_ALLOWED_TRANSITIONS: dict[UpdateOperationState, frozenset[UpdateOperationState]] = {
    "maintenance_closing": frozenset(
        {"maintenance_closed", "aborted_before_switch", "repair_required"}
    ),
    "maintenance_closed": frozenset(
        {"checkpoint_ready", "aborted_before_switch", "repair_required"}
    ),
    "checkpoint_ready": frozenset(
        {"candidate_starting", "aborted_before_switch", "repair_required"}
    ),
    "candidate_starting": frozenset(
        {"candidate_verified", "rollback_restoring", "repair_required"}
    ),
    "candidate_verified": frozenset(
        {"candidate_reopening", "rollback_restoring", "repair_required"}
    ),
    "candidate_reopening": frozenset({"committed"}),
    "rollback_restoring": frozenset({"old_release_starting", "repair_required"}),
    "old_release_starting": frozenset({"old_release_verified", "repair_required"}),
    "old_release_verified": frozenset({"old_release_reopening", "repair_required"}),
    "old_release_reopening": frozenset({"rolled_back"}),
    "repair_required": frozenset(
        {"rollback_restoring", "old_release_starting", "old_release_verified"}
    ),
    "committed": frozenset(),
    "rolled_back": frozenset(),
    "aborted_before_switch": frozenset(),
}


class UpdateCutoverRefused(RuntimeError):
    """A cutover boundary was absent, stale, unsafe, or internally inconsistent."""


class UpdateAdmissionClosed(RuntimeError):
    """A mutation or background launch arrived during the update boundary."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


def _canonical_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase canonical UUID4")
    return value


def _absolute_path(value: str, *, label: str) -> str:
    if (
        not value
        or value != value.strip()
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be one bounded absolute path")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be absolute and normalized")
    return value


class UpdateCaptureBoundary(_StrictModel):
    capture_id: str
    instance_id: str
    process_pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    sqlite_receipt_path: str
    sqlite_receipt_sha256: str
    sqlite_snapshot_sha256: str
    status: Literal["complete", "partial"]
    project_count: int = Field(ge=0)
    uncaptured_project_count: int = Field(ge=0)

    @field_validator("capture_id", "instance_id", "space_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("data_dir_id", "sqlite_receipt_sha256", "sqlite_snapshot_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("update capture digests must be lowercase SHA-256")
        return value

    @field_validator("sqlite_receipt_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _absolute_path(value, label="SQLite receipt path")

    @model_validator(mode="after")
    def validate_boundary(self) -> UpdateCaptureBoundary:
        path = Path(self.sqlite_receipt_path)
        if path.name != "sqlite-capture.json" or path.parent.name != f"backup-{self.capture_id}":
            raise ValueError("update capture path is not bound to its capture identity")
        if self.uncaptured_project_count > self.project_count:
            raise ValueError("uncaptured project count exceeds the project inventory")
        if self.status == "complete" and self.uncaptured_project_count:
            raise ValueError("a complete update capture cannot report uncaptured projects")
        return self


class UpdateOperationReceipt(_StrictModel):
    """Durable nonsecret state shared by the old process, root, and candidate."""

    schema_version: Literal[1] = UPDATE_OPERATION_SCHEMA_VERSION
    operation_id: str
    installation_id: str
    space_id: str
    base_commit: str
    candidate_commit: str
    base_instance_id: str
    base_process_pid: int = Field(gt=0)
    built_receipt_path: str
    built_receipt_sha256: str
    preflight_receipt_path: str
    preflight_receipt_sha256: str
    receipt_path: str
    state: UpdateOperationState
    capture: UpdateCaptureBoundary | None = None
    final_receipt_path: str | None = None
    final_receipt_sha256: str | None = None
    project_receipt_path: str | None = None
    project_receipt_sha256: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    candidate_instance_id: str | None = None
    candidate_process_pid: int | None = Field(default=None, gt=0)
    restored_instance_id: str | None = None
    restored_process_pid: int | None = Field(default=None, gt=0)
    failure: str | None = Field(default=None, max_length=240)
    runtime_failure: str | None = Field(default=None, max_length=240)
    started_at: datetime
    updated_at: datetime

    @field_validator(
        "operation_id",
        "installation_id",
        "space_id",
        "base_instance_id",
        "candidate_instance_id",
        "restored_instance_id",
    )
    @classmethod
    def validate_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _canonical_uuid4(value, label=info.field_name.replace("_", " "))

    @field_validator("base_commit", "candidate_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("update operations require full lowercase Git commits")
        return value

    @field_validator(
        "built_receipt_path",
        "preflight_receipt_path",
        "receipt_path",
        "final_receipt_path",
        "project_receipt_path",
        "checkpoint_path",
    )
    @classmethod
    def validate_path(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _absolute_path(value, label=info.field_name.replace("_", " "))

    @field_validator(
        "built_receipt_sha256",
        "preflight_receipt_sha256",
        "final_receipt_sha256",
        "project_receipt_sha256",
        "checkpoint_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("update operation digests must be lowercase SHA-256")
        return value

    @field_validator("failure", "runtime_failure")
    @classmethod
    def validate_failure(cls, value: str | None) -> str | None:
        if value is not None and (
            not value
            or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("update failure must be one bounded nonempty line")
        return value

    @field_validator("started_at", "updated_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("update operation times require a UTC offset")
        return value

    @model_validator(mode="after")
    def validate_relationship(self) -> UpdateOperationReceipt:
        if self.base_commit == self.candidate_commit:
            raise ValueError("an update candidate must differ from its base")
        if self.updated_at < self.started_at:
            raise ValueError("update operation time moved backwards")
        expected = update_operation_receipt_path(
            self.operation_id,
            Path(self.receipt_path).parent,
        )
        if Path(self.receipt_path) != expected:
            raise ValueError("update operation path and identity disagree")
        pairs = (
            (self.final_receipt_path, self.final_receipt_sha256),
            (self.project_receipt_path, self.project_receipt_sha256),
            (self.checkpoint_path, self.checkpoint_sha256),
            (self.candidate_instance_id, self.candidate_process_pid),
            (self.restored_instance_id, self.restored_process_pid),
        )
        if any((first is None) != (second is None) for first, second in pairs):
            raise ValueError("update operation paired fields must be both present or absent")
        checkpoint_states = {
            "checkpoint_ready",
            "candidate_starting",
            "candidate_verified",
            "candidate_reopening",
            "rollback_restoring",
            "old_release_starting",
            "old_release_verified",
            "old_release_reopening",
            "repair_required",
            "committed",
            "rolled_back",
        }
        if self.state in checkpoint_states and any(
            value is None
            for value in (
                self.capture,
                self.final_receipt_path,
                self.final_receipt_sha256,
                self.project_receipt_path,
                self.project_receipt_sha256,
                self.checkpoint_path,
                self.checkpoint_sha256,
            )
        ):
            raise ValueError("post-checkpoint update states require the exact final boundary")
        if self.state == "maintenance_closed" and self.capture is None:
            raise ValueError("closed maintenance requires its exact capture")
        if (
            self.state
            in {
                "candidate_verified",
                "candidate_reopening",
                "committed",
            }
            and self.candidate_instance_id is None
        ):
            raise ValueError("a verified candidate requires its running process identity")
        if (
            self.state
            in {
                "old_release_verified",
                "old_release_reopening",
                "rolled_back",
            }
            and self.restored_instance_id is None
        ):
            raise ValueError("a verified rollback requires its restored process identity")
        if (
            self.state
            in {
                "rollback_restoring",
                "old_release_starting",
                "old_release_verified",
                "old_release_reopening",
                "rolled_back",
                "repair_required",
            }
            and self.failure is None
        ):
            raise ValueError("rollback and repair states require a loud failure diagnostic")
        return self

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_UPDATE_STATES

    def expected_running_commit(self) -> str | None:
        if self.state in _BASE_RUNNING_STATES:
            return self.base_commit
        if self.state in _CANDIDATE_RUNNING_STATES:
            return self.candidate_commit
        return None


def update_operation_receipt_path(operation_id: str, update_root: Path) -> Path:
    _canonical_uuid4(operation_id, label="update operation identity")
    return update_root / f"update-operation-{uuid.UUID(operation_id).hex}.json"


def new_update_operation(
    *,
    operation_id: str,
    installation_id: str,
    space_id: str,
    base_commit: str,
    candidate_commit: str,
    base_instance_id: str,
    base_process_pid: int,
    built_receipt_path: Path,
    built_receipt_sha256: str,
    preflight_receipt_path: Path,
    preflight_receipt_sha256: str,
    update_root: Path,
    now: datetime | None = None,
) -> UpdateOperationReceipt:
    timestamp = now or datetime.now(UTC)
    return UpdateOperationReceipt(
        operation_id=operation_id,
        installation_id=installation_id,
        space_id=space_id,
        base_commit=base_commit,
        candidate_commit=candidate_commit,
        base_instance_id=base_instance_id,
        base_process_pid=base_process_pid,
        built_receipt_path=str(built_receipt_path),
        built_receipt_sha256=built_receipt_sha256,
        preflight_receipt_path=str(preflight_receipt_path),
        preflight_receipt_sha256=preflight_receipt_sha256,
        receipt_path=str(update_operation_receipt_path(operation_id, update_root)),
        state="maintenance_closing",
        started_at=timestamp,
        updated_at=timestamp,
    )


def publish_update_operation(
    receipt: UpdateOperationReceipt,
    *,
    expected_uid: int,
    expected_gid: int,
) -> str:
    path = Path(receipt.receipt_path)
    _require_private_directory(path.parent, expected_uid=expected_uid, expected_gid=expected_gid)
    if os.path.lexists(path):
        observed, digest = read_update_operation(path, expected_uid=expected_uid)
        if observed != receipt:
            raise UpdateCutoverRefused(
                "An existing update operation receipt names another boundary."
            )
        return digest
    payload = _model_bytes(receipt)
    _write_new_private_file(path, payload, uid=expected_uid, gid=expected_gid)
    observed, digest = read_update_operation(path, expected_uid=expected_uid)
    if observed != receipt:
        raise UpdateCutoverRefused("The published update operation changed during readback.")
    return digest


def advance_update_operation(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_sha256: str,
    state: UpdateOperationState,
    update: dict[str, object] | None = None,
    now: datetime | None = None,
) -> tuple[UpdateOperationReceipt, str]:
    current, _digest = read_update_operation(
        path,
        expected_uid=expected_uid,
        expected_sha256=expected_sha256,
    )
    if state != current.state and state not in _ALLOWED_TRANSITIONS[current.state]:
        raise UpdateCutoverRefused(f"Update operation cannot move from {current.state} to {state}.")
    changes = dict(update or {})
    changes.update({"state": state, "updated_at": now or datetime.now(UTC)})
    advanced = current.model_copy(update=changes)
    # Revalidate model_copy output before bytes are published.
    advanced = UpdateOperationReceipt.model_validate(advanced)
    payload = _model_bytes(advanced)
    _replace_private_file(path, payload, uid=expected_uid, gid=expected_gid)
    observed, digest = read_update_operation(path, expected_uid=expected_uid)
    if observed != advanced:
        raise UpdateCutoverRefused("The update operation changed during durable readback.")
    return observed, digest


def read_update_operation(
    path: Path,
    *,
    expected_uid: int,
    expected_sha256: str | None = None,
) -> tuple[UpdateOperationReceipt, str]:
    payload = _read_private_file(path, expected_uid=expected_uid)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise UpdateCutoverRefused("The update operation receipt digest changed.")
    try:
        receipt = UpdateOperationReceipt.model_validate_json(payload)
    except ValueError as exc:
        raise UpdateCutoverRefused("The update operation receipt is invalid.") from exc
    if Path(receipt.receipt_path) != path:
        raise UpdateCutoverRefused("The update operation path and payload disagree.")
    return receipt, digest


def update_operation_receipts(
    update_root: Path,
    *,
    expected_uid: int,
) -> tuple[tuple[Path, UpdateOperationReceipt, str], ...]:
    try:
        entries = tuple(update_root.iterdir())
    except OSError as exc:
        raise UpdateCutoverRefused("The update checkpoint root cannot be inspected.") from exc
    receipts: list[tuple[Path, UpdateOperationReceipt, str]] = []
    for path in entries:
        matched = _OPERATION_NAME.fullmatch(path.name)
        if matched is None:
            continue
        receipt, digest = read_update_operation(path, expected_uid=expected_uid)
        if uuid.UUID(receipt.operation_id).hex != matched.group(1):
            raise UpdateCutoverRefused("An update receipt filename and identity disagree.")
        receipts.append((path, receipt, digest))
    return tuple(sorted(receipts, key=lambda item: item[0].name))


def active_update_operation(
    update_root: Path,
    *,
    expected_uid: int,
) -> tuple[Path, UpdateOperationReceipt, str] | None:
    active = [
        item
        for item in update_operation_receipts(update_root, expected_uid=expected_uid)
        if not item[1].terminal
    ]
    if len(active) > 1:
        raise UpdateCutoverRefused(
            "Multiple unfinished update operations require operator inspection."
        )
    return active[0] if active else None


def update_operation_needing_recovery(
    update_root: Path,
    *,
    expected_uid: int,
) -> tuple[Path, UpdateOperationReceipt, str] | None:
    """Return one nonterminal operation or the latest failed selected release."""

    operations = update_operation_receipts(update_root, expected_uid=expected_uid)
    active = [item for item in operations if not item[1].terminal]
    if len(active) > 1:
        raise UpdateCutoverRefused(
            "Multiple unfinished update operations require operator inspection."
        )
    if active:
        return active[0]
    if not operations:
        return None
    latest = max(
        operations,
        key=lambda item: (item[1].updated_at, item[1].operation_id),
    )
    if latest[1].state in {"committed", "rolled_back"} and latest[1].runtime_failure is not None:
        return latest
    return None


@dataclass(frozen=True)
class UpdateRuntimeBoundary:
    path: Path
    receipt: UpdateOperationReceipt
    receipt_sha256: str


def load_update_runtime_boundary(
    *,
    running_commit: str,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
    expected_uid: int,
) -> UpdateRuntimeBoundary | None:
    active = active_update_operation(layout.update_checkpoints_root, expected_uid=expected_uid)
    if active is None:
        return None
    path, receipt, digest = active
    expected = receipt.expected_running_commit()
    if expected is not None and expected != running_commit:
        raise UpdateCutoverRefused(
            "The unfinished update operation does not name this running release."
        )
    return UpdateRuntimeBoundary(path=path, receipt=receipt, receipt_sha256=digest)


class RuntimeAdmissionGate:
    """Close new mutations while allowing already-entered calls to finish."""

    def __init__(
        self,
        *,
        closed: bool = False,
        reason: str = "Server update maintenance",
    ) -> None:
        if not reason or reason != reason.strip():
            raise ValueError("admission reason must be one nonempty line")
        self._closed = closed
        self._reason = reason
        self._active = 0
        self._condition = threading.Condition()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def require_open(self, effect: str) -> None:
        if not effect or effect != effect.strip():
            raise ValueError("admission effect must be one nonempty line")
        with self._condition:
            if not self._closed:
                return
        raise UpdateAdmissionClosed(f"{self._reason} blocks {effect}.")

    @contextmanager
    def mutation(self, effect: str) -> Iterator[None]:
        if not effect or effect != effect.strip():
            raise ValueError("mutation effect must be one nonempty line")
        with self._condition:
            if self._closed:
                raise UpdateAdmissionClosed(f"{self._reason} blocks {effect}.")
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close_and_wait(
        self,
        *,
        timeout: float,
        additional_idle: Callable[[], bool] = lambda: True,
    ) -> None:
        if timeout <= 0:
            raise ValueError("maintenance admission timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closed = True
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise UpdateCutoverRefused(
                        "Timed out waiting for in-flight server mutations to settle."
                    )
                self._condition.wait(min(remaining, 0.25))
        while not additional_idle():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UpdateCutoverRefused(
                    "Timed out waiting for in-flight provider work to reach a durable boundary."
                )
            time.sleep(min(remaining, 0.05))

    def reopen(self) -> None:
        with self._condition:
            self._closed = False
            self._condition.notify_all()


class BackgroundBoundary(Protocol):
    def close_watcher_notifications(self) -> None: ...

    def runtime_is_idle(self) -> bool: ...

    def accept_watcher_notifications(self) -> None: ...


class CaptureCallable(Protocol):
    def __call__(self): ...


class UpdateControlClient(Protocol):
    def probe(self): ...

    def verify_update_candidate(self, *, operation_id: str, receipt_sha256: str): ...

    def release_update_fence(self, *, operation_id: str, receipt_sha256: str): ...

    def abort_update_maintenance(self, *, operation_id: str, receipt_sha256: str): ...


class RootCutoverActions(Protocol):
    def enter_maintenance(self, *, operation_id: str, receipt_sha256: str): ...

    def final_rehearsal(
        self,
        operation: UpdateOperationReceipt,
        receipt_sha256: str,
    ): ...

    def create_checkpoint(
        self,
        final_receipt,
        *,
        sqlite_receipt_path: Path,
        sqlite_receipt_sha256: str,
        project_receipt_path: Path,
        project_receipt_sha256: str,
    ): ...

    def stop_service(self) -> None: ...

    def switch_current(self, *, expected: Path, target: Path) -> None: ...

    def start_service(self) -> int: ...

    def control_for_running(self, commit: str) -> UpdateControlClient: ...

    def restore_checkpoint(self, checkpoint_path: Path, checkpoint_sha256: str) -> None: ...

    def current_release(self) -> Path: ...


def capture_boundary_from_control(result) -> UpdateCaptureBoundary:
    return UpdateCaptureBoundary(
        capture_id=result.capture_id,
        instance_id=result.instance_id,
        process_pid=result.pid,
        data_dir_id=result.data_dir_id,
        space_id=result.space_id,
        sqlite_receipt_path=result.receipt_path,
        sqlite_receipt_sha256=result.receipt_sha256,
        sqlite_snapshot_sha256=result.snapshot_sha256,
        status=result.status,
        project_count=result.project_count,
        uncaptured_project_count=result.uncaptured_project_count,
    )


class UpdateServiceCoordinator:
    """The running process side of one receipt-bound update maintenance window."""

    def __init__(
        self,
        *,
        layout: ServerLayout,
        instance_metadata,
        space_id: str,
        admission: RuntimeAdmissionGate,
        background_admission: RuntimeAdmissionGate | None = None,
        background: BackgroundBoundary,
        capture_sqlite: CaptureCallable,
        catalog,
        store,
        startup_effect_fence=None,
        runtime_started: threading.Event | None = None,
        runtime_error: Callable[[], str | None] = lambda: None,
        pause_runtime_owners: Callable[[float], None] = lambda _timeout: None,
        resume_runtime_owners: Callable[[], None] = lambda: None,
        expected_uid: int | None = None,
        expected_gid: int | None = None,
    ) -> None:
        self.layout = layout
        self.identity = instance_metadata
        self.space_id = _canonical_uuid4(space_id, label="update space identity")
        self.admission = admission
        self.background_admission = background_admission or admission
        self.background = background
        self.capture_sqlite = capture_sqlite
        self.catalog = catalog
        self.store = store
        self.startup_effect_fence = startup_effect_fence
        self.runtime_started = runtime_started
        self.runtime_error = runtime_error
        self.pause_runtime_owners = pause_runtime_owners
        self.resume_runtime_owners = resume_runtime_owners
        self.expected_uid = os.geteuid() if expected_uid is None else expected_uid
        self.expected_gid = os.getegid() if expected_gid is None else expected_gid

    def enter_maintenance(
        self,
        *,
        operation_id: str,
        receipt_sha256: str,
        timeout: float,
    ) -> tuple[UpdateOperationReceipt, str, object]:
        path = update_operation_receipt_path(operation_id, self.layout.update_checkpoints_root)
        receipt, digest = read_update_operation(
            path,
            expected_uid=self.expected_uid,
            expected_sha256=receipt_sha256,
        )
        self._require_common_identity(receipt)
        if receipt.state == "maintenance_closed":
            if receipt.capture is None:
                raise UpdateCutoverRefused("Closed maintenance lost its capture boundary.")
            return receipt, digest, control_capture_from_boundary(receipt.capture)
        if receipt.state != "maintenance_closing":
            raise UpdateCutoverRefused("This update operation is not waiting to enter maintenance.")
        if (
            receipt.base_instance_id != self.identity.instance_id
            or receipt.base_process_pid != self.identity.pid
            or receipt.base_commit != self.identity.running_commit
        ):
            raise UpdateCutoverRefused(
                "The live process changed after candidate rehearsal; abort and rehearse again."
            )
        self.background.close_watcher_notifications()
        self.admission.close_and_wait(timeout=timeout)
        self.background_admission.close_and_wait(
            timeout=timeout,
            additional_idle=self.background.runtime_is_idle,
        )
        self.pause_runtime_owners(timeout)
        # A watcher/retry owner can pass the check-only launch gate immediately
        # before closure and publish its worker only while that owner is joining.
        # Recheck after every automatic launcher has stopped so the capture cannot
        # race a provider worker that appeared after the first idle observation.
        self.background_admission.close_and_wait(
            timeout=timeout,
            additional_idle=self.background.runtime_is_idle,
        )
        capture_result = self.capture_sqlite()
        boundary = capture_boundary_from_control(capture_result)
        if (
            boundary.instance_id != self.identity.instance_id
            or boundary.process_pid != self.identity.pid
            or boundary.space_id != self.space_id
            or boundary.data_dir_id != self.identity.data_dir_id
        ):
            raise UpdateCutoverRefused(
                "The maintenance capture came from another process or team space."
            )
        receipt, digest = advance_update_operation(
            path,
            expected_uid=self.expected_uid,
            expected_gid=self.expected_gid,
            expected_sha256=digest,
            state="maintenance_closed",
            update={"capture": boundary},
        )
        return receipt, digest, capture_result

    def verify_running_release(
        self,
        *,
        operation_id: str,
        receipt_sha256: str,
    ) -> tuple[UpdateOperationReceipt, str, str]:
        path = update_operation_receipt_path(operation_id, self.layout.update_checkpoints_root)
        receipt, digest = read_update_operation(
            path,
            expected_uid=self.expected_uid,
            expected_sha256=receipt_sha256,
        )
        self._require_common_identity(receipt)
        if self.startup_effect_fence is None or not self.startup_effect_fence.active:
            raise UpdateCutoverRefused(
                "The switched process is not behind the required startup-effect fence."
            )
        if self.startup_effect_fence.attempted_effects:
            raise UpdateCutoverRefused(
                "The switched process attempted an external effect before verification."
            )
        verified_state = receipt.state in {"candidate_verified", "old_release_verified"}
        if receipt.state not in {
            "candidate_starting",
            "candidate_verified",
            "old_release_starting",
            "old_release_verified",
        }:
            raise UpdateCutoverRefused(
                "This update operation is not waiting for fenced-process verification."
            )
        expected_commit = (
            receipt.candidate_commit
            if receipt.state in {"candidate_starting", "candidate_verified"}
            else receipt.base_commit
        )
        if self.identity.running_commit != expected_commit:
            raise UpdateCutoverRefused("The fenced process runs the wrong release commit.")
        verification_sha256 = _live_read_model_digest(
            receipt,
            self.background,
            self.catalog,
            self.store,
            self.expected_uid,
        )
        state: UpdateOperationState = (
            "candidate_verified"
            if receipt.state in {"candidate_starting", "candidate_verified"}
            else "old_release_verified"
        )
        identity_update = (
            {
                "candidate_instance_id": self.identity.instance_id,
                "candidate_process_pid": self.identity.pid,
            }
            if state == "candidate_verified"
            else {
                "restored_instance_id": self.identity.instance_id,
                "restored_process_pid": self.identity.pid,
            }
        )
        current_identity = (
            receipt.candidate_instance_id
            if state == "candidate_verified"
            else receipt.restored_instance_id
        )
        current_pid = (
            receipt.candidate_process_pid
            if state == "candidate_verified"
            else receipt.restored_process_pid
        )
        if not verified_state or (
            current_identity != self.identity.instance_id or current_pid != self.identity.pid
        ):
            receipt, digest = advance_update_operation(
                path,
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state=state,
                update=identity_update,
            )
        return receipt, digest, verification_sha256

    def release_fence(
        self,
        *,
        operation_id: str,
        receipt_sha256: str,
        timeout: float,
    ) -> tuple[UpdateOperationReceipt, str]:
        path = update_operation_receipt_path(operation_id, self.layout.update_checkpoints_root)
        receipt, digest = read_update_operation(
            path,
            expected_uid=self.expected_uid,
            expected_sha256=receipt_sha256,
        )
        self._require_common_identity(receipt)
        if receipt.state in {"committed", "rolled_back"}:
            return receipt, digest
        reopening: UpdateOperationState
        terminal: UpdateOperationState
        if receipt.state in {"candidate_verified", "candidate_reopening"}:
            reopening = "candidate_reopening"
            terminal = "committed"
        elif receipt.state in {"old_release_verified", "old_release_reopening"}:
            reopening = "old_release_reopening"
            terminal = "rolled_back"
        else:
            raise UpdateCutoverRefused(
                "The update fence cannot open before the running release is verified."
            )
        if receipt.state != reopening:
            receipt, digest = advance_update_operation(
                path,
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state=reopening,
            )
        try:
            self._open_runtime(timeout=timeout)
            return advance_update_operation(
                path,
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state=terminal,
                update={"runtime_failure": None},
            )
        except BaseException as exc:
            failure = _safe_failure(exc)
            try:
                current, current_digest = read_update_operation(
                    path,
                    expected_uid=self.expected_uid,
                )
                if current.state in {reopening, terminal}:
                    advance_update_operation(
                        path,
                        expected_uid=self.expected_uid,
                        expected_gid=self.expected_gid,
                        expected_sha256=current_digest,
                        state=current.state,
                        update={"runtime_failure": failure},
                    )
            except BaseException:
                pass
            raise UpdateCutoverRefused(
                "The selected release could not finish deferred runtime startup."
            ) from exc

    def abort_before_switch(
        self,
        *,
        operation_id: str,
        receipt_sha256: str,
        timeout: float,
    ) -> tuple[UpdateOperationReceipt, str]:
        path = update_operation_receipt_path(operation_id, self.layout.update_checkpoints_root)
        receipt, digest = read_update_operation(
            path,
            expected_uid=self.expected_uid,
            expected_sha256=receipt_sha256,
        )
        self._require_common_identity(receipt)
        if receipt.state == "aborted_before_switch":
            return receipt, digest
        if receipt.state not in {
            "maintenance_closing",
            "maintenance_closed",
            "checkpoint_ready",
        }:
            raise UpdateCutoverRefused(
                "Maintenance cannot be aborted after the release switch began."
            )
        if self.identity.running_commit != receipt.base_commit:
            raise UpdateCutoverRefused(
                "The old release is not running, so maintenance cannot reopen safely."
            )
        try:
            self._open_runtime(timeout=timeout)
            return advance_update_operation(
                path,
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state="aborted_before_switch",
                update={"failure": None},
            )
        except BaseException as exc:
            failure = _safe_failure(exc)
            try:
                current, current_digest = read_update_operation(
                    path,
                    expected_uid=self.expected_uid,
                )
                if current.state in {
                    "maintenance_closing",
                    "maintenance_closed",
                    "checkpoint_ready",
                }:
                    advance_update_operation(
                        path,
                        expected_uid=self.expected_uid,
                        expected_gid=self.expected_gid,
                        expected_sha256=current_digest,
                        state=current.state,
                        update={"failure": failure},
                    )
            except BaseException:
                pass
            raise UpdateCutoverRefused(
                "Pre-switch maintenance could not reopen the old release."
            ) from exc

    def _require_common_identity(self, receipt: UpdateOperationReceipt) -> None:
        if receipt.space_id != self.space_id or self.identity.data_dir_id != data_dir_identity(
            self.layout.data_dir
        ):
            raise UpdateCutoverRefused(
                "The update operation belongs to another installed data boundary."
            )

    def _open_runtime(self, *, timeout: float) -> None:
        if self.startup_effect_fence is None:
            self.background_admission.reopen()
            self.admission.reopen()
            self.background.accept_watcher_notifications()
            self.resume_runtime_owners()
            return
        self.background_admission.reopen()
        self.startup_effect_fence.release()
        if self.runtime_started is not None and not self.runtime_started.wait(timeout):
            raise UpdateCutoverRefused(
                "The verified release did not start its deferred runtime before timeout."
            )
        error = self.runtime_error()
        if error:
            raise UpdateCutoverRefused(
                "The verified release opened, but deferred startup reported a failure."
            )
        self.admission.reopen()


@dataclass(frozen=True)
class UpdateCutoverOutcome:
    operation_id: str
    operation_state: Literal["committed", "rolled_back"]
    candidate_commit: str
    running_commit: str
    receipt_path: Path
    receipt_sha256: str
    failure: str | None = None


class UpdateCutoverCoordinator:
    """Root-side exact switch with one service-owned crash-visible receipt."""

    def __init__(
        self,
        *,
        layout: ServerLayout,
        actions: RootCutoverActions,
        expected_uid: int,
        expected_gid: int,
        progress: Callable[[str], None] = lambda _phase: None,
    ) -> None:
        self.layout = layout
        self.actions = actions
        self.expected_uid = expected_uid
        self.expected_gid = expected_gid
        self.progress = progress

    def run(self, built_receipt, preflight_receipt) -> UpdateCutoverOutcome:
        self._validate_preflight(built_receipt, preflight_receipt)
        operation_id = str(uuid.uuid4())
        operation = new_update_operation(
            operation_id=operation_id,
            installation_id=built_receipt.installation_id,
            space_id=preflight_receipt.space_id,
            base_commit=built_receipt.base_running_commit,
            candidate_commit=built_receipt.candidate_commit,
            base_instance_id=built_receipt.base_instance_id,
            base_process_pid=built_receipt.base_process_pid,
            built_receipt_path=Path(built_receipt.receipt_path),
            built_receipt_sha256=file_sha256(
                Path(built_receipt.receipt_path),
                expected_uid=self.expected_uid,
                maximum=16 * 1024,
            ),
            preflight_receipt_path=Path(preflight_receipt.receipt_path),
            preflight_receipt_sha256=file_sha256(
                Path(preflight_receipt.receipt_path),
                expected_uid=self.expected_uid,
                maximum=4 * 1024 * 1024,
            ),
            update_root=self.layout.update_checkpoints_root,
        )
        digest = publish_update_operation(
            operation,
            expected_uid=self.expected_uid,
            expected_gid=self.expected_gid,
        )
        checkpoint_path: Path | None = None
        switched = False
        try:
            entered = self.actions.enter_maintenance(
                operation_id=operation_id,
                receipt_sha256=digest,
            )
            operation, digest = self._read_result(entered)
            if operation.state != "maintenance_closed" or operation.capture is None:
                raise UpdateCutoverRefused(
                    "The running service did not publish one closed maintenance capture."
                )
            self.progress("maintenance_closed")
            final = self.actions.final_rehearsal(operation, digest)
            self._validate_final_boundary(operation, final)
            final_path = Path(final.receipt_path)
            project_path = Path(operation.capture.sqlite_receipt_path).parent / "project-files.json"
            project_sha256 = file_sha256(
                project_path,
                expected_uid=self.expected_uid,
                maximum=4 * 1024 * 1024,
            )
            if project_sha256 != final.project_capture_sha256:
                raise UpdateCutoverRefused(
                    "The final project capture differs from the closed-admission rehearsal."
                )
            operation, digest = advance_update_operation(
                Path(operation.receipt_path),
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state="maintenance_closed",
                update={
                    "final_receipt_path": str(final_path),
                    "final_receipt_sha256": file_sha256(
                        final_path,
                        expected_uid=self.expected_uid,
                        maximum=4 * 1024 * 1024,
                    ),
                    "project_receipt_path": str(project_path),
                    "project_receipt_sha256": project_sha256,
                },
            )
            checkpoint = self.actions.create_checkpoint(
                final,
                sqlite_receipt_path=Path(operation.capture.sqlite_receipt_path),
                sqlite_receipt_sha256=operation.capture.sqlite_receipt_sha256,
                project_receipt_path=project_path,
                project_receipt_sha256=project_sha256,
            )
            checkpoint_path = Path(checkpoint.manifest_path)
            operation, digest = advance_update_operation(
                Path(operation.receipt_path),
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state="checkpoint_ready",
                update={
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": file_sha256(
                        checkpoint_path,
                        expected_uid=self.expected_uid,
                        maximum=4 * 1024 * 1024,
                    ),
                },
            )
            self.progress("checkpoint_ready")
            self.actions.stop_service()
            self.actions.switch_current(
                expected=self.layout.release_dir(operation.base_commit),
                target=self.layout.release_dir(operation.candidate_commit),
            )
            switched = True
            operation, digest = advance_update_operation(
                Path(operation.receipt_path),
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state="candidate_starting",
            )
            self.actions.start_service()
            candidate_control = self.actions.control_for_running(operation.candidate_commit)
            self.progress("candidate_started")
            verified = candidate_control.verify_update_candidate(
                operation_id=operation_id,
                receipt_sha256=digest,
            )
            operation, digest = self._read_result(verified)
            if operation.state != "candidate_verified":
                raise UpdateCutoverRefused(
                    "The switched candidate did not publish its verified read model."
                )
            self.progress("candidate_verified")
            released = candidate_control.release_update_fence(
                operation_id=operation_id,
                receipt_sha256=digest,
            )
            operation, digest = self._read_result(released)
            if operation.state != "committed":
                raise UpdateCutoverRefused(
                    "The verified candidate did not durably commit its update receipt."
                )
            return self._outcome(operation, digest)
        except BaseException as exc:
            failure = _safe_failure(exc)
            if switched:
                current, current_digest = read_update_operation(
                    update_operation_receipt_path(
                        operation_id,
                        self.layout.update_checkpoints_root,
                    ),
                    expected_uid=self.expected_uid,
                )
                if current.state in {"candidate_reopening", "committed"}:
                    failed, failed_digest = advance_update_operation(
                        Path(current.receipt_path),
                        expected_uid=self.expected_uid,
                        expected_gid=self.expected_gid,
                        expected_sha256=current_digest,
                        state=current.state,
                        update={"runtime_failure": failure},
                    )
                    repaired, repaired_digest = self.repair_selected_release(
                        failed,
                        failed_digest,
                    )
                    return self._outcome(repaired, repaired_digest)
                if checkpoint_path is None:
                    raise UpdateCutoverRefused(
                        "The candidate switch failed without a rollback checkpoint; keep the service stopped."
                    ) from exc
                return self._rollback(
                    operation_id=operation_id,
                    checkpoint_path=checkpoint_path,
                    failure=failure,
                )
            self._abort_before_switch(operation_id, failure=failure)
            if isinstance(exc, UpdateCutoverRefused):
                raise
            raise UpdateCutoverRefused(failure) from exc

    def repair_committed(
        self,
        operation: UpdateOperationReceipt,
        receipt_sha256: str,
    ) -> tuple[UpdateOperationReceipt, str]:
        """Compatibility name for repairing one selected committed candidate."""

        if operation.state != "committed":
            raise UpdateCutoverRefused(
                "Committed update repair requires one committed candidate receipt."
            )
        return self.repair_selected_release(operation, receipt_sha256)

    def repair_selected_release(
        self,
        operation: UpdateOperationReceipt,
        receipt_sha256: str,
    ) -> tuple[UpdateOperationReceipt, str]:
        """Finish the chosen release normally after the rollback decision passed."""

        path = Path(operation.receipt_path)
        observed, digest = read_update_operation(
            path,
            expected_uid=self.expected_uid,
            expected_sha256=receipt_sha256,
        )
        selected: Literal["candidate", "base"]
        terminal: UpdateOperationState
        if operation.state in {"candidate_reopening", "committed"}:
            selected = "candidate"
            terminal = "committed"
        elif operation.state in {"old_release_reopening", "rolled_back"}:
            selected = "base"
            terminal = "rolled_back"
        else:
            raise UpdateCutoverRefused(
                "Selected-release repair requires one post-decision update receipt."
            )
        if observed != operation or (
            operation.state == terminal and operation.runtime_failure is None
        ):
            raise UpdateCutoverRefused(
                "Selected-release repair requires one matching failed or interrupted receipt."
            )
        selected_commit = (
            operation.candidate_commit if selected == "candidate" else operation.base_commit
        )
        if self.actions.current_release() != self.layout.release_dir(selected_commit):
            raise UpdateCutoverRefused(
                "Selected-release repair found a different installed current pointer."
            )
        try:
            self.actions.stop_service()
            if operation.state != terminal:
                operation, digest = advance_update_operation(
                    path,
                    expected_uid=self.expected_uid,
                    expected_gid=self.expected_gid,
                    expected_sha256=digest,
                    state=terminal,
                    update={
                        "runtime_failure": operation.runtime_failure
                        or "Deferred runtime startup was interrupted after release selection."
                    },
                )
            self.actions.start_service()
            control = self.actions.control_for_running(selected_commit)
            probe = control.probe()
            if (
                operation.capture is None
                or probe.space_id != operation.space_id
                or probe.data_dir_id != operation.capture.data_dir_id
            ):
                raise UpdateCutoverRefused(
                    "The normally restarted selected release belongs to another installed data boundary."
                )
            identity_update = (
                {
                    "candidate_instance_id": probe.instance_id,
                    "candidate_process_pid": probe.pid,
                }
                if selected == "candidate"
                else {
                    "restored_instance_id": probe.instance_id,
                    "restored_process_pid": probe.pid,
                }
            )
            return advance_update_operation(
                path,
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state=terminal,
                update={
                    **identity_update,
                    "runtime_failure": None,
                },
            )
        except BaseException as exc:
            with suppress(BaseException):
                self.actions.stop_service()
            failure = _safe_failure(exc)
            try:
                current, current_digest = read_update_operation(
                    path,
                    expected_uid=self.expected_uid,
                )
                if current.state in {
                    "candidate_reopening",
                    "committed",
                    "old_release_reopening",
                    "rolled_back",
                }:
                    advance_update_operation(
                        path,
                        expected_uid=self.expected_uid,
                        expected_gid=self.expected_gid,
                        expected_sha256=current_digest,
                        state=current.state,
                        update={"runtime_failure": failure},
                    )
            except BaseException:
                pass
            raise UpdateCutoverRefused(
                "The selected release could not complete a normal runtime restart; RCP kept "
                "the service stopped and did not reverse a completed rollback decision."
            ) from exc

    def _rollback(
        self,
        *,
        operation_id: str,
        checkpoint_path: Path,
        failure: str,
    ) -> UpdateCutoverOutcome:
        path = update_operation_receipt_path(operation_id, self.layout.update_checkpoints_root)
        operation, digest = read_update_operation(path, expected_uid=self.expected_uid)
        try:
            if operation.state in {"candidate_starting", "candidate_verified"}:
                operation, digest = advance_update_operation(
                    path,
                    expected_uid=self.expected_uid,
                    expected_gid=self.expected_gid,
                    expected_sha256=digest,
                    state="rollback_restoring",
                    update={"failure": failure},
                )
            self.actions.stop_service()
            if operation.state in {"rollback_restoring", "repair_required"}:
                if operation.checkpoint_sha256 is None:
                    raise UpdateCutoverRefused(
                        "Rollback lost the digest binding for its verified checkpoint."
                    )
                self.actions.restore_checkpoint(
                    checkpoint_path,
                    operation.checkpoint_sha256,
                )
                current = self.actions.current_release()
                if current == self.layout.release_dir(operation.candidate_commit):
                    self.actions.switch_current(
                        expected=current,
                        target=self.layout.release_dir(operation.base_commit),
                    )
                elif current != self.layout.release_dir(operation.base_commit):
                    raise UpdateCutoverRefused(
                        "Rollback found an unexpected installed current release."
                    )
                operation, digest = advance_update_operation(
                    path,
                    expected_uid=self.expected_uid,
                    expected_gid=self.expected_gid,
                    expected_sha256=digest,
                    state="old_release_starting",
                )
            elif operation.state not in {"old_release_starting", "old_release_verified"}:
                raise UpdateCutoverRefused(
                    "The unfinished update is not at a resumable rollback phase."
                )
            if self.actions.current_release() != self.layout.release_dir(operation.base_commit):
                raise UpdateCutoverRefused(
                    "Rollback cannot start because current does not name the previous release."
                )
            self.actions.start_service()
            old_control = self.actions.control_for_running(operation.base_commit)
            verified = old_control.verify_update_candidate(
                operation_id=operation_id,
                receipt_sha256=digest,
            )
            operation, digest = self._read_result(verified)
            if operation.state != "old_release_verified":
                raise UpdateCutoverRefused(
                    "The restored release did not publish its verified read model."
                )
            released = old_control.release_update_fence(
                operation_id=operation_id,
                receipt_sha256=digest,
            )
            operation, digest = self._read_result(released)
            if operation.state != "rolled_back":
                raise UpdateCutoverRefused("The restored release did not durably finish rollback.")
            self.progress("rolled_back")
            return self._outcome(operation, digest)
        except BaseException as exc:
            try:
                current, current_digest = read_update_operation(
                    path,
                    expected_uid=self.expected_uid,
                )
                if current.state in {"old_release_reopening", "rolled_back"}:
                    failed, failed_digest = advance_update_operation(
                        path,
                        expected_uid=self.expected_uid,
                        expected_gid=self.expected_gid,
                        expected_sha256=current_digest,
                        state=current.state,
                        update={"runtime_failure": _safe_failure(exc)},
                    )
                    repaired, repaired_digest = self.repair_selected_release(
                        failed,
                        failed_digest,
                    )
                    self.progress("rolled_back")
                    return self._outcome(repaired, repaired_digest)
                if current.state != "repair_required":
                    advance_update_operation(
                        path,
                        expected_uid=self.expected_uid,
                        expected_gid=self.expected_gid,
                        expected_sha256=current_digest,
                        state="repair_required",
                        update={"failure": failure},
                    )
            except BaseException:
                pass
            with suppress(BaseException):
                self.actions.stop_service()
            raise UpdateCutoverRefused(
                "Automatic rollback failed; RCP kept the service stopped and recorded repair_required."
            ) from exc

    def recover(
        self,
        operation: UpdateOperationReceipt,
        receipt_sha256: str,
    ) -> tuple[UpdateOperationReceipt, str]:
        """Re-enter one exact unfinished receipt without preparing another target."""

        path = Path(operation.receipt_path)
        observed, digest = read_update_operation(
            path,
            expected_uid=self.expected_uid,
            expected_sha256=receipt_sha256,
        )
        if observed != operation or operation.terminal:
            raise UpdateCutoverRefused(
                "Update recovery requires one matching unfinished operation receipt."
            )
        base = self.layout.release_dir(operation.base_commit)
        candidate = self.layout.release_dir(operation.candidate_commit)
        current = self.actions.current_release()
        if operation.state in {"candidate_reopening", "old_release_reopening"}:
            return self.repair_selected_release(operation, digest)
        if operation.state in {"maintenance_closing", "maintenance_closed"} or (
            operation.state == "checkpoint_ready" and current == base
        ):
            if current != base:
                raise UpdateCutoverRefused(
                    "Pre-switch update recovery found the wrong current release."
                )
            control = self._restart_fenced(operation.base_commit)
            result = control.abort_update_maintenance(
                operation_id=operation.operation_id,
                receipt_sha256=digest,
            )
            recovered, recovered_digest = self._read_result(result)
            if recovered.state != "aborted_before_switch":
                raise UpdateCutoverRefused(
                    "The old release did not durably abort unfinished maintenance."
                )
            return recovered, recovered_digest
        if operation.state == "checkpoint_ready":
            if current != candidate:
                raise UpdateCutoverRefused(
                    "Checkpoint-ready recovery found neither the base nor candidate release."
                )
            operation, digest = advance_update_operation(
                path,
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                expected_sha256=digest,
                state="candidate_starting",
            )
        if operation.state in {"candidate_starting", "candidate_verified"}:
            if self.actions.current_release() != candidate:
                raise UpdateCutoverRefused(
                    "Candidate recovery found a mismatched current release pointer."
                )
            try:
                control = self._restart_fenced(operation.candidate_commit)
                verified = control.verify_update_candidate(
                    operation_id=operation.operation_id,
                    receipt_sha256=digest,
                )
                operation, digest = self._read_result(verified)
                released = control.release_update_fence(
                    operation_id=operation.operation_id,
                    receipt_sha256=digest,
                )
                committed, committed_digest = self._read_result(released)
                if committed.state != "committed":
                    raise UpdateCutoverRefused("Recovered candidate did not durably commit.")
                return committed, committed_digest
            except BaseException as exc:
                current, current_digest = read_update_operation(
                    path,
                    expected_uid=self.expected_uid,
                )
                if current.state in {"candidate_reopening", "committed"}:
                    failed, failed_digest = advance_update_operation(
                        path,
                        expected_uid=self.expected_uid,
                        expected_gid=self.expected_gid,
                        expected_sha256=current_digest,
                        state=current.state,
                        update={"runtime_failure": _safe_failure(exc)},
                    )
                    return self.repair_selected_release(failed, failed_digest)
                if operation.checkpoint_path is None:
                    raise UpdateCutoverRefused(
                        "Candidate recovery has no verified rollback checkpoint."
                    ) from exc
                outcome = self._rollback(
                    operation_id=operation.operation_id,
                    checkpoint_path=Path(operation.checkpoint_path),
                    failure=_safe_failure(exc),
                )
                return read_update_operation(
                    outcome.receipt_path,
                    expected_uid=self.expected_uid,
                    expected_sha256=outcome.receipt_sha256,
                )
        if operation.state in {
            "rollback_restoring",
            "old_release_starting",
            "old_release_verified",
            "repair_required",
        }:
            if operation.checkpoint_path is None:
                raise UpdateCutoverRefused("Rollback recovery lost its verified checkpoint path.")
            outcome = self._rollback(
                operation_id=operation.operation_id,
                checkpoint_path=Path(operation.checkpoint_path),
                failure=operation.failure or "Resuming interrupted automatic rollback.",
            )
            return read_update_operation(
                outcome.receipt_path,
                expected_uid=self.expected_uid,
                expected_sha256=outcome.receipt_sha256,
            )
        raise UpdateCutoverRefused("The unfinished update state has no safe recovery route.")

    def _restart_fenced(self, commit: str) -> UpdateControlClient:
        with suppress(BaseException):
            self.actions.stop_service()
        self.actions.start_service()
        return self.actions.control_for_running(commit)

    def _abort_before_switch(self, operation_id: str, *, failure: str) -> None:
        path = update_operation_receipt_path(operation_id, self.layout.update_checkpoints_root)
        try:
            operation, digest = read_update_operation(path, expected_uid=self.expected_uid)
            if operation.state == "checkpoint_ready":
                with suppress(BaseException):
                    self.actions.start_service()
                control = self.actions.control_for_running(operation.base_commit)
            else:
                control = self.actions.control_for_running(operation.base_commit)
            control.abort_update_maintenance(
                operation_id=operation_id,
                receipt_sha256=digest,
            )
        except BaseException as exc:
            try:
                operation, digest = read_update_operation(path, expected_uid=self.expected_uid)
                if (
                    operation.state not in TERMINAL_UPDATE_STATES
                    and operation.checkpoint_path is not None
                ):
                    advance_update_operation(
                        path,
                        expected_uid=self.expected_uid,
                        expected_gid=self.expected_gid,
                        expected_sha256=digest,
                        state="repair_required",
                        update={"failure": failure},
                    )
            except BaseException:
                pass
            raise UpdateCutoverRefused(
                "Pre-switch maintenance could not reopen; operator repair is required."
            ) from exc

    def _read_result(self, result) -> tuple[UpdateOperationReceipt, str]:
        operation, digest = read_update_operation(
            update_operation_receipt_path(
                result.operation_id,
                self.layout.update_checkpoints_root,
            ),
            expected_uid=self.expected_uid,
            expected_sha256=result.receipt_sha256,
        )
        if operation.state != result.operation_state:
            raise UpdateCutoverRefused("The control result and durable update receipt disagree.")
        return operation, digest

    def _validate_preflight(self, built, preflight) -> None:
        if (
            preflight.installation_id != built.installation_id
            or preflight.candidate_commit != built.candidate_commit
            or preflight.base_running_commit != built.base_running_commit
            or preflight.base_current_commit != built.base_current_commit
            or preflight.base_instance_id != built.base_instance_id
            or preflight.base_process_pid != built.base_process_pid
            or preflight.built_receipt_path != built.receipt_path
            or Path(built.release_path) != self.layout.release_dir(built.candidate_commit)
        ):
            raise UpdateCutoverRefused(
                "The preflight rehearsal differs from its exact built candidate and live base."
            )

    def _validate_final_boundary(self, operation: UpdateOperationReceipt, final) -> None:
        assert operation.capture is not None
        if (
            final.installation_id != operation.installation_id
            or final.space_id != operation.space_id
            or final.candidate_commit != operation.candidate_commit
            or final.base_running_commit != operation.base_commit
            or final.base_instance_id != operation.base_instance_id
            or final.base_process_pid != operation.base_process_pid
            or final.capture_id != operation.capture.capture_id
            or final.sqlite_snapshot_sha256 != operation.capture.sqlite_snapshot_sha256
            or final.built_receipt_sha256 != operation.built_receipt_sha256
        ):
            raise UpdateCutoverRefused(
                "The final rehearsal does not match the closed-admission capture."
            )

    @staticmethod
    def _outcome(operation: UpdateOperationReceipt, digest: str) -> UpdateCutoverOutcome:
        if operation.state not in {"committed", "rolled_back"}:
            raise UpdateCutoverRefused("The update outcome is not terminal.")
        return UpdateCutoverOutcome(
            operation_id=operation.operation_id,
            operation_state=operation.state,
            candidate_commit=operation.candidate_commit,
            running_commit=(
                operation.candidate_commit
                if operation.state == "committed"
                else operation.base_commit
            ),
            receipt_path=Path(operation.receipt_path),
            receipt_sha256=digest,
            failure=operation.runtime_failure or operation.failure,
        )


def _safe_failure(exc: BaseException) -> str:
    from rcp.server_ops.models import redact_server_text

    message = redact_server_text(str(exc)).strip()
    if (
        not message
        or len(message) > 240
        or any(ord(character) < 32 or ord(character) == 127 for character in message)
    ):
        return "The switched candidate failed its bounded verification."
    return message


def _live_read_model_digest(receipt, background, catalog, store, expected_uid: int) -> str:
    from rcp.server_ops.rehearsal import (
        CandidateProjectVerification,
        StartupRecoveryReadModel,
        _canonical_sha256,
        _project_card_comparison,
        read_verified_candidate_receipt,
    )

    if receipt.final_receipt_path is None or receipt.final_receipt_sha256 is None:
        raise UpdateCutoverRefused("The update operation lost its final rehearsal receipt.")
    final = read_verified_candidate_receipt(
        Path(receipt.final_receipt_path),
        expected_uid=expected_uid,
    )
    if (
        file_sha256(
            Path(receipt.final_receipt_path),
            expected_uid=expected_uid,
            maximum=4 * 1024 * 1024,
        )
        != receipt.final_receipt_sha256
    ):
        raise UpdateCutoverRefused("The final rehearsal receipt changed before verification.")
    startup = StartupRecoveryReadModel.model_validate(background.plan_startup_recovery().as_dict())
    if startup != final.startup_recovery:
        raise UpdateCutoverRefused("The switched release changed the startup recovery read model.")
    cards = {str(card["id"]): card for card in catalog.cards()}
    if set(cards) != {project.project_id for project in final.projects}:
        raise UpdateCutoverRefused(
            "The switched release omitted or substituted a registered project."
        )
    projects: list[CandidateProjectVerification] = []
    for expected in final.projects:
        card = cards[expected.project_id]
        if expected.status == "not_replay_verified":
            comparison = _project_card_comparison(card)
            observed = CandidateProjectVerification(
                project_id=expected.project_id,
                status="not_replay_verified",
                revision=None,
                projection_sha256=_canonical_sha256(comparison),
            )
        else:
            status, snapshot = catalog.cached_snapshot_status(expected.project_id)
            graph = snapshot.get("graph") if status == "valid" and snapshot is not None else None
            if not isinstance(graph, dict):
                try:
                    _service, rebuilt = catalog.open_snapshot(expected.project_id)
                    graph = rebuilt["graph"]
                except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
                    raise UpdateCutoverRefused(
                        "The switched release could not reconstruct project projection "
                        f"{expected.project_id}."
                    ) from exc
            if not isinstance(graph, dict):
                raise UpdateCutoverRefused(
                    f"The switched release has no valid project projection for {expected.project_id}."
                )
            revision = graph.get("revision")
            observed = CandidateProjectVerification(
                project_id=expected.project_id,
                status="verified",
                revision=revision if isinstance(revision, int) else None,
                projection_sha256=_canonical_sha256(graph),
            )
        if observed != expected:
            raise UpdateCutoverRefused(
                f"The switched release changed project projection {expected.project_id}."
            )
        # These are the same storage-backed reads as the two operational API routes.
        store.agent_tasks(expected.project_id)
        store.watchers(expected.project_id)
        projects.append(observed)
    read_model = {
        "startup_recovery": startup.model_dump(mode="json"),
        "projects": [item.model_dump(mode="json") for item in projects],
        "reads": list(final.reads),
    }
    return _canonical_sha256(read_model)


def control_capture_from_boundary(boundary: UpdateCaptureBoundary):
    from rcp.server_ops.control import ServerControlBackupCaptureResult

    return ServerControlBackupCaptureResult(
        instance_id=boundary.instance_id,
        pid=boundary.process_pid,
        data_dir_id=boundary.data_dir_id,
        space_id=boundary.space_id,
        capture_id=boundary.capture_id,
        receipt_path=boundary.sqlite_receipt_path,
        receipt_sha256=boundary.sqlite_receipt_sha256,
        snapshot_sha256=boundary.sqlite_snapshot_sha256,
        status=boundary.status,
        project_count=boundary.project_count,
        uncaptured_project_count=boundary.uncaptured_project_count,
    )


def _model_bytes(model: BaseModel) -> bytes:
    payload = (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > UPDATE_OPERATION_MAX_BYTES:
        raise UpdateCutoverRefused("The update operation receipt exceeds its fixed bound.")
    return payload


def _read_private_file(path: Path, *, expected_uid: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise UpdateCutoverRefused("A private update operation receipt is unavailable.") from exc
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_uid != expected_uid
            or stat.S_IMODE(initial.st_mode) != UPDATE_OPERATION_FILE_MODE
            or initial.st_size > UPDATE_OPERATION_MAX_BYTES
        ):
            raise UpdateCutoverRefused(
                "A private update operation receipt has unsafe ownership or mode."
            )
        chunks: list[bytes] = []
        remaining = initial.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise UpdateCutoverRefused("A private update operation receipt is incomplete.")
            chunks.append(chunk)
            remaining -= len(chunk)
        final = os.fstat(descriptor)
        path_final = path.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, name) != getattr(final, name) for name in stable) or any(
            getattr(final, name) != getattr(path_final, name) for name in stable
        ):
            raise UpdateCutoverRefused("A private update operation receipt changed while reading.")
        return b"".join(chunks)
    except OSError as exc:
        raise UpdateCutoverRefused("A private update operation receipt cannot be read.") from exc
    finally:
        os.close(descriptor)


def _write_new_private_file(path: Path, payload: bytes, *, uid: int, gid: int) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, UPDATE_OPERATION_FILE_MODE)
    except OSError as exc:
        raise UpdateCutoverRefused("The update operation receipt could not be created.") from exc
    try:
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, UPDATE_OPERATION_FILE_MODE)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _replace_private_file(path: Path, payload: bytes, *, uid: int, gid: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, UPDATE_OPERATION_FILE_MODE)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise UpdateCutoverRefused("The update operation receipt could not be replaced.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _require_private_directory(path: Path, *, expected_uid: int, expected_gid: int) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise UpdateCutoverRefused("The update checkpoint root is unavailable.") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or (metadata.st_uid, metadata.st_gid) != (expected_uid, expected_gid)
        or stat.S_IMODE(metadata.st_mode) != UPDATE_OPERATION_DIRECTORY_MODE
    ):
        raise UpdateCutoverRefused(
            "The update checkpoint root has unsafe type, ownership, or mode."
        )


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short update receipt write")
        remaining = remaining[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_sha256(path: Path, *, expected_uid: int, maximum: int = UPDATE_OPERATION_MAX_BYTES) -> str:
    """Hash one stable private regular file for a receipt link."""

    if maximum <= 0:
        raise ValueError("file digest bound must be positive")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_uid != expected_uid
            or initial.st_size > maximum
        ):
            raise UpdateCutoverRefused("An update proof has unsafe type, owner, or size.")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum + 1 - size)):
            digest.update(chunk)
            size += len(chunk)
            if size > maximum:
                raise UpdateCutoverRefused("An update proof exceeds its fixed bound.")
        final = os.fstat(descriptor)
        path_final = path.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, name) != getattr(final, name) for name in stable) or any(
            getattr(final, name) != getattr(path_final, name) for name in stable
        ):
            raise UpdateCutoverRefused("An update proof changed while hashing.")
        if size != initial.st_size:
            raise UpdateCutoverRefused("An update proof hash is incomplete.")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


__all__ = [
    "RuntimeAdmissionGate",
    "TERMINAL_UPDATE_STATES",
    "UPDATE_OPERATION_FILE_MODE",
    "UpdateAdmissionClosed",
    "UpdateCaptureBoundary",
    "UpdateCutoverCoordinator",
    "UpdateCutoverOutcome",
    "UpdateCutoverRefused",
    "UpdateOperationReceipt",
    "UpdateOperationState",
    "UpdateRuntimeBoundary",
    "UpdateServiceCoordinator",
    "active_update_operation",
    "advance_update_operation",
    "capture_boundary_from_control",
    "control_capture_from_boundary",
    "file_sha256",
    "load_update_runtime_boundary",
    "new_update_operation",
    "publish_update_operation",
    "read_update_operation",
    "update_operation_receipt_path",
    "update_operation_receipts",
    "update_operation_needing_recovery",
]
