"""Private, kernel-authenticated control transport for one installed team service."""

from __future__ import annotations

import errno
import json
import os
import socket
import stat
import struct
import sys
import threading
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp.limits import (
    MEMBER_REMOVAL_PREVIEW_MAX_ITEMS,
    SERVER_CONTROL_ACCEPT_POLL_INTERVAL_SECONDS,
    SERVER_CONTROL_BACKUP_CAPTURE_TIMEOUT_SECONDS,
    SERVER_CONTROL_IO_TIMEOUT_SECONDS,
    SERVER_CONTROL_PROJECT_PROVISION_TIMEOUT_SECONDS,
    SERVER_CONTROL_PROVIDER_CHECK_TIMEOUT_SECONDS,
    SERVER_CONTROL_STOP_TIMEOUT_SECONDS,
    SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS,
    SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
)
from rcp.server_ops.models import SERVER_CLI_MAX_STEPS, ServerStep, redact_server_text
from rcp.server_ops.update_cutover import TERMINAL_UPDATE_STATES, UpdateOperationState
from rcp.server_runtime import ServerMetadata, read_server_metadata

SERVER_CONTROL_PROTOCOL_VERSION = 9
SERVER_CONTROL_MAX_REQUEST_BYTES = 64 * 1024
SERVER_CONTROL_MAX_RESPONSE_BYTES = 256 * 1024
SERVER_CONTROL_SOCKET_MODE = 0o600
SERVER_CONTROL_RUNTIME_MODE = 0o700
SERVER_CONTROL_MAX_SOCKET_PATH_BYTES = 99

_FRAME_HEADER = struct.Struct("!I")
_HEX_DIGEST = frozenset("0123456789abcdef")

ServerControlOperation = Literal[
    "probe",
    "provider_readiness_plan",
    "provider_readiness_check",
    "project_provision_plan",
    "project_provision_step",
    "project_transfer_upload_plan",
    "project_transfer_upload_complete",
    "project_transfer_activate",
    "member_removal_plan",
    "member_removal_advance",
    "restore_activation_commit",
    "backup_sqlite_capture",
    "update_maintenance_enter",
    "update_candidate_verify",
    "update_fence_release",
    "update_maintenance_abort",
]
SERVER_CONTROL_OPERATIONS: tuple[ServerControlOperation, ...] = (
    "probe",
    "provider_readiness_plan",
    "provider_readiness_check",
    "project_provision_plan",
    "project_provision_step",
    "project_transfer_upload_plan",
    "project_transfer_upload_complete",
    "project_transfer_activate",
    "member_removal_plan",
    "member_removal_advance",
    "restore_activation_commit",
    "backup_sqlite_capture",
    "update_maintenance_enter",
    "update_candidate_verify",
    "update_fence_release",
    "update_maintenance_abort",
)
ServerControlProjectStatus = Literal[
    "waiting_for_server_setup",
    "setup_in_progress",
    "operator_action_needed",
    "ready_for_review",
]


class ServerControlError(RuntimeError):
    """A bounded control request was refused or could not complete."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ServerControlUnavailable(ServerControlError):
    """The private installed-service transport is unavailable."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _canonical_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase, hyphenated canonical UUID4")
    return value


class ServerControlRequest(_StrictModel):
    protocol_version: Literal[SERVER_CONTROL_PROTOCOL_VERSION] = SERVER_CONTROL_PROTOCOL_VERSION
    request_id: str
    instance_id: str
    operation: ServerControlOperation
    selector_kind: Literal["request", "project", "member"] | None = None
    selector_id: str | None = None
    boundary_sha256: str | None = None
    target_id: str | None = None

    @model_validator(mode="after")
    def validate_ids(self) -> ServerControlRequest:
        _canonical_uuid4(self.request_id, label="control request id")
        _canonical_uuid4(self.instance_id, label="control instance id")
        if self.selector_id is not None:
            _canonical_uuid4(self.selector_id, label="control selector id")
        if self.operation in {"probe", "backup_sqlite_capture"}:
            if any(
                value is not None
                for value in (
                    self.selector_kind,
                    self.selector_id,
                    self.boundary_sha256,
                    self.target_id,
                )
            ):
                raise ValueError("selector-free control operations cannot carry selector fields")
        elif self.operation in {
            "update_maintenance_enter",
            "update_candidate_verify",
            "update_fence_release",
            "update_maintenance_abort",
            "restore_activation_commit",
        }:
            if (
                self.selector_kind is not None
                or self.selector_id is None
                or self.boundary_sha256 is None
                or self.target_id is not None
            ):
                raise ValueError(
                    "root control operations require one receipt-bound operation identity"
                )
        elif self.operation in {
            "provider_readiness_plan",
            "project_provision_plan",
            "project_transfer_upload_plan",
            "member_removal_plan",
        }:
            if self.selector_kind is None or self.selector_id is None:
                raise ValueError("control plan requires one selector")
            if self.operation == "project_provision_plan" and self.selector_kind != "request":
                raise ValueError("project provisioning plan requires one request selector")
            if self.operation == "project_transfer_upload_plan" and self.selector_kind != "request":
                raise ValueError("project transfer upload plan requires one request selector")
            if self.operation == "member_removal_plan" and self.selector_kind != "member":
                raise ValueError("member-removal plan requires one member selector")
            if self.boundary_sha256 is not None or self.target_id is not None:
                raise ValueError("control plan cannot carry a step boundary")
        elif self.operation in {
            "project_transfer_upload_complete",
            "project_transfer_activate",
        }:
            if (
                self.selector_kind != "request"
                or self.selector_id is None
                or self.boundary_sha256 is None
                or self.target_id is not None
            ):
                raise ValueError(
                    "project transfer operation requires one confirmed upload boundary"
                )
        elif self.operation == "member_removal_advance":
            if (
                self.selector_kind != "member"
                or self.selector_id is None
                or self.boundary_sha256 is None
                or self.target_id is not None
            ):
                raise ValueError("member-removal advance requires one confirmed member boundary")
        elif any(
            value is None
            for value in (
                self.selector_kind,
                self.selector_id,
                self.boundary_sha256,
                self.target_id,
            )
        ):
            raise ValueError("control step requires its exact plan boundary")
        elif self.operation == "project_provision_step" and self.selector_kind != "request":
            raise ValueError("project provisioning step requires one request selector")
        for value, label in (
            (self.boundary_sha256, "control boundary"),
            (self.target_id, "control target"),
        ):
            if value is not None and (
                len(value) != 64 or any(character not in _HEX_DIGEST for character in value)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        return self


class ServerControlMemberSnapshot(_StrictModel):
    member_id: str
    member_display_name: str | None = Field(default=None, max_length=240)
    removal_started_at: str | None
    removed_at: str | None
    last_authenticating_member: bool
    project_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    orphaned_project_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    orphaned_project_labels: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    active_task_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    active_episode_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    active_token_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    browser_session_count: int = Field(ge=0)
    space_invitation_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    project_invitation_ids: tuple[str, ...] = Field(max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS)
    boundary_sha256: str

    @model_validator(mode="after")
    def validate_snapshot(self) -> ServerControlMemberSnapshot:
        _canonical_uuid4(self.member_id, label="member id")
        for value in (
            self.member_display_name,
            self.removal_started_at,
            self.removed_at,
            *self.project_ids,
            *self.orphaned_project_ids,
            *self.orphaned_project_labels,
            *self.active_task_ids,
            *self.active_episode_ids,
            *self.active_token_ids,
            *self.space_invitation_ids,
            *self.project_invitation_ids,
        ):
            if value is not None and (
                not value
                or len(value) > 2048
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
                or redact_server_text(value) != value
            ):
                raise ValueError("member-removal snapshots require bounded nonsecret text")
        for values in (
            self.project_ids,
            self.orphaned_project_ids,
            self.active_task_ids,
            self.active_episode_ids,
            self.active_token_ids,
            self.space_invitation_ids,
            self.project_invitation_ids,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("member-removal inventories must be sorted and unique")
        if not set(self.orphaned_project_ids).issubset(self.project_ids):
            raise ValueError("member-removal orphaned projects must belong to the target")
        if len(self.orphaned_project_labels) != len(self.orphaned_project_ids):
            raise ValueError("member-removal orphan labels must match their project ids")
        if len(self.boundary_sha256) != 64 or any(
            character not in _HEX_DIGEST for character in self.boundary_sha256
        ):
            raise ValueError("member-removal boundary must be a lowercase SHA-256 digest")
        return self


class ServerControlProbeResult(_StrictModel):
    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    operations: tuple[ServerControlOperation, ...]
    pending_member_removals: tuple[ServerControlMemberSnapshot, ...] = Field(
        default=(), max_length=MEMBER_REMOVAL_PREVIEW_MAX_ITEMS
    )

    @model_validator(mode="after")
    def validate_identity(self) -> ServerControlProbeResult:
        _canonical_uuid4(self.instance_id, label="control instance id")
        _canonical_uuid4(self.space_id, label="space id")
        if len(self.data_dir_id) != 64 or any(
            character not in _HEX_DIGEST for character in self.data_dir_id
        ):
            raise ValueError("data directory identity must be a lowercase SHA-256 digest")
        expected_order = tuple(
            operation for operation in SERVER_CONTROL_OPERATIONS if operation in self.operations
        )
        if (
            not self.operations
            or self.operations[0] != "probe"
            or self.operations != expected_order
        ):
            raise ValueError("control probe operations must be unique and in registry order")
        return self


class ServerControlMemberPlanResult(_StrictModel):
    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    snapshot: ServerControlMemberSnapshot
    step: ServerStep

    @model_validator(mode="after")
    def validate_plan(self) -> ServerControlMemberPlanResult:
        _validate_provider_result_identity(
            self.instance_id,
            self.space_id,
            self.data_dir_id,
            self.snapshot.member_id,
            self.snapshot.boundary_sha256,
        )
        if self.step.state != "pending":
            raise ValueError("member-removal plans require one pending step")
        return self


class ServerControlMemberAdvanceResult(_StrictModel):
    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    confirmed_boundary_sha256: str
    snapshot: ServerControlMemberSnapshot
    step: ServerStep

    @model_validator(mode="after")
    def validate_advance(self) -> ServerControlMemberAdvanceResult:
        _validate_provider_result_identity(
            self.instance_id,
            self.space_id,
            self.data_dir_id,
            self.snapshot.member_id,
            self.confirmed_boundary_sha256,
        )
        if self.step.state not in {
            "succeeded",
            "failed",
            "operator_action_needed",
            "unavailable",
        }:
            raise ValueError("member-removal advance requires one terminal CLI step")
        return self


class ServerControlProviderTarget(_StrictModel):
    target_id: str
    step: ServerStep

    @model_validator(mode="after")
    def validate_target(self) -> ServerControlProviderTarget:
        if len(self.target_id) != 64 or any(
            character not in _HEX_DIGEST for character in self.target_id
        ):
            raise ValueError("provider readiness target must be a lowercase SHA-256 digest")
        if self.step.state != "pending":
            raise ValueError("provider readiness plans require pending steps")
        return self


class ServerControlProviderPlanResult(_StrictModel):
    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    selector_kind: Literal["request", "project"]
    selector_id: str
    boundary_sha256: str
    targets: tuple[ServerControlProviderTarget, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_plan(self) -> ServerControlProviderPlanResult:
        _validate_provider_result_identity(
            self.instance_id,
            self.space_id,
            self.data_dir_id,
            self.selector_id,
            self.boundary_sha256,
        )
        if [target.step.number for target in self.targets] != list(range(1, len(self.targets) + 1)):
            raise ValueError("provider readiness plan steps must be consecutive")
        ids = [target.target_id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("provider readiness plan targets must be unique")
        return self


class ServerControlProviderCheckResult(_StrictModel):
    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    selector_kind: Literal["request", "project"]
    selector_id: str
    target_id: str
    boundary_sha256: str
    next_boundary_sha256: str
    step: ServerStep

    @model_validator(mode="after")
    def validate_check(self) -> ServerControlProviderCheckResult:
        _validate_provider_result_identity(
            self.instance_id,
            self.space_id,
            self.data_dir_id,
            self.selector_id,
            self.boundary_sha256,
        )
        for value, label in (
            (self.target_id, "provider readiness target"),
            (self.next_boundary_sha256, "next provider readiness boundary"),
        ):
            if len(value) != 64 or any(character not in _HEX_DIGEST for character in value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if self.step.state not in {
            "succeeded",
            "failed",
            "operator_action_needed",
            "unavailable",
        }:
            raise ValueError("provider readiness check requires one terminal step")
        return self


class ServerControlProjectTarget(_StrictModel):
    target_id: str
    step: ServerStep

    @model_validator(mode="after")
    def validate_target(self) -> ServerControlProjectTarget:
        if len(self.target_id) != 64 or any(
            character not in _HEX_DIGEST for character in self.target_id
        ):
            raise ValueError("project provisioning target must be a lowercase SHA-256 digest")
        if self.step.state != "pending":
            raise ValueError("project provisioning plans require pending steps")
        return self


class ServerControlProjectPlanResult(_StrictModel):
    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    request_id: str
    request_status: ServerControlProjectStatus
    revision: int = Field(ge=0)
    boundary_sha256: str
    targets: tuple[ServerControlProjectTarget, ...] = Field(
        min_length=1,
        max_length=SERVER_CLI_MAX_STEPS,
    )

    @model_validator(mode="after")
    def validate_plan(self) -> ServerControlProjectPlanResult:
        _validate_provider_result_identity(
            self.instance_id,
            self.space_id,
            self.data_dir_id,
            self.request_id,
            self.boundary_sha256,
        )
        if [target.step.number for target in self.targets] != list(range(1, len(self.targets) + 1)):
            raise ValueError("project provisioning plan steps must be consecutive")
        ids = [target.target_id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("project provisioning plan targets must be unique")
        return self


class ServerControlProjectStepResult(_StrictModel):
    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    request_id: str
    request_status: ServerControlProjectStatus
    revision: int = Field(ge=0)
    target_id: str
    boundary_sha256: str
    next_boundary_sha256: str
    step: ServerStep

    @model_validator(mode="after")
    def validate_step(self) -> ServerControlProjectStepResult:
        _validate_provider_result_identity(
            self.instance_id,
            self.space_id,
            self.data_dir_id,
            self.request_id,
            self.boundary_sha256,
        )
        for value, label in (
            (self.target_id, "project provisioning target"),
            (self.next_boundary_sha256, "next project provisioning boundary"),
        ):
            if len(value) != 64 or any(character not in _HEX_DIGEST for character in value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if self.step.state not in {
            "succeeded",
            "failed",
            "operator_action_needed",
            "unavailable",
        }:
            raise ValueError("project provisioning step requires one terminal step")
        return self


class ServerControlProjectTransferUploadResult(_StrictModel):
    """One request-bound target upload lease or durable completion."""

    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    request_id: str
    project_id: str
    archive_sha256: str
    archive_size_bytes: int = Field(ge=1)
    lease_boundary_sha256: str
    state: Literal["active", "complete", "consumed"]

    @model_validator(mode="after")
    def validate_upload(self) -> ServerControlProjectTransferUploadResult:
        _validate_provider_result_identity(
            self.instance_id,
            self.space_id,
            self.data_dir_id,
            self.request_id,
            self.lease_boundary_sha256,
        )
        _canonical_uuid4(self.project_id, label="project transfer upload project id")
        if len(self.archive_sha256) != 64 or any(
            character not in _HEX_DIGEST for character in self.archive_sha256
        ):
            raise ValueError("project transfer upload archive must be a lowercase SHA-256 digest")
        return self


class ServerControlProjectTransferActivationResult(_StrictModel):
    """Public readback of one committed target activation boundary."""

    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    target_request_id: str
    source_request_id: str
    project_id: str
    archive_sha256: str
    upload_lease_boundary_sha256: str
    archive_manifest_sha256: str
    target_manifest_sha256: str
    publication_sha256: str
    activated_at: str
    state: Literal["activated"] = "activated"

    @model_validator(mode="after")
    def validate_activation(self) -> ServerControlProjectTransferActivationResult:
        _validate_provider_result_identity(
            self.instance_id,
            self.space_id,
            self.data_dir_id,
            self.target_request_id,
            self.upload_lease_boundary_sha256,
        )
        _canonical_uuid4(self.source_request_id, label="source transfer request id")
        _canonical_uuid4(self.project_id, label="project transfer activation project id")
        for value, label in (
            (self.archive_sha256, "project transfer activation archive"),
            (self.archive_manifest_sha256, "project transfer archive manifest"),
            (self.target_manifest_sha256, "project transfer target manifest"),
            (self.publication_sha256, "project transfer import publication"),
        ):
            if len(value) != 64 or any(character not in _HEX_DIGEST for character in value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        try:
            parsed = datetime.fromisoformat(self.activated_at)
        except ValueError as exc:
            raise ValueError("project transfer activation time must be ISO 8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("project transfer activation time must include a timezone")
        return self


class ServerControlBackupCaptureResult(_StrictModel):
    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    capture_id: str
    receipt_path: str
    receipt_sha256: str
    snapshot_sha256: str
    status: Literal["complete", "partial"]
    project_count: int = Field(ge=0)
    uncaptured_project_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_capture(self) -> ServerControlBackupCaptureResult:
        for value, label in (
            (self.instance_id, "control instance id"),
            (self.space_id, "space id"),
            (self.capture_id, "backup capture id"),
        ):
            _canonical_uuid4(value, label=label)
        for value, label in (
            (self.data_dir_id, "data directory identity"),
            (self.receipt_sha256, "backup receipt digest"),
            (self.snapshot_sha256, "SQLite snapshot digest"),
        ):
            if len(value) != 64 or any(character not in _HEX_DIGEST for character in value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        path = Path(self.receipt_path)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path.name != "sqlite-capture.json"
            or path.parent.name != f"backup-{self.capture_id}"
        ):
            raise ValueError("backup receipt path is not bound to its capture identity")
        if self.uncaptured_project_count > self.project_count:
            raise ValueError("uncaptured project count exceeds the captured project inventory")
        if self.status == "complete" and self.uncaptured_project_count:
            raise ValueError("a complete backup capture cannot report uncaptured projects")
        return self


class ServerControlUpdateResult(_StrictModel):
    """Receipt-bound maintenance, verification, or fence-release readback."""

    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    operation_id: str
    operation_state: UpdateOperationState
    receipt_sha256: str
    running_commit: str
    capture: ServerControlBackupCaptureResult | None = None
    verification_sha256: str | None = None

    @model_validator(mode="after")
    def validate_update(self) -> ServerControlUpdateResult:
        for value, label in (
            (self.instance_id, "control instance id"),
            (self.space_id, "space id"),
            (self.operation_id, "update operation id"),
        ):
            _canonical_uuid4(value, label=label)
        for value, label in (
            (self.data_dir_id, "data directory identity"),
            (self.receipt_sha256, "update receipt digest"),
            (self.verification_sha256, "update verification digest"),
        ):
            if value is not None and (
                len(value) != 64 or any(character not in _HEX_DIGEST for character in value)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if len(self.running_commit) != 40 or any(
            character not in _HEX_DIGEST for character in self.running_commit
        ):
            raise ValueError("update control results require one full lowercase Git commit")
        if self.capture is not None and (
            self.capture.instance_id != self.instance_id
            or self.capture.pid != self.pid
            or self.capture.data_dir_id != self.data_dir_id
            or self.capture.space_id != self.space_id
        ):
            raise ValueError("update capture identity differs from its control process")
        if self.operation_state == "maintenance_closed" and self.capture is None:
            raise ValueError("closed maintenance results require their exact capture")
        if (
            self.operation_state in {"candidate_verified", "old_release_verified"}
            and self.verification_sha256 is None
        ):
            raise ValueError("release verification results require a read-model digest")
        if self.operation_state in TERMINAL_UPDATE_STATES and self.capture is not None:
            raise ValueError("terminal update results cannot repeat a capture boundary")
        return self


class ServerControlRestoreResult(_StrictModel):
    """The durable readback that opened one replacement restore."""

    instance_id: str
    pid: int = Field(gt=0)
    data_dir_id: str
    space_id: str
    space_kind: Literal["team"] = "team"
    operation_id: str
    restore_phase: Literal["complete"] = "complete"
    boundary_sha256: str
    readback: object

    @model_validator(mode="after")
    def validate_restore(self) -> ServerControlRestoreResult:
        for value, label in (
            (self.instance_id, "control instance id"),
            (self.space_id, "space id"),
            (self.operation_id, "restore operation id"),
        ):
            _canonical_uuid4(value, label=label)
        for value, label in (
            (self.data_dir_id, "data directory identity"),
            (self.boundary_sha256, "restore activation boundary"),
        ):
            if len(value) != 64 or any(character not in _HEX_DIGEST for character in value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        from rcp.server_ops.restore import RestoreActivationReadback

        readback = (
            self.readback
            if isinstance(self.readback, RestoreActivationReadback)
            else RestoreActivationReadback.model_validate_json(
                json.dumps(
                    self.readback,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        )
        if (
            readback.instance_id != self.instance_id
            or readback.pid != self.pid
            or readback.data_dir_id != self.data_dir_id
            or readback.space_id != self.space_id
        ):
            raise ValueError("restore readback identity differs from its control process")
        object.__setattr__(self, "readback", readback)
        return self


def _validate_provider_result_identity(
    instance_id: str,
    space_id: str,
    data_dir_id: str,
    selector_id: str,
    boundary_sha256: str,
) -> None:
    _canonical_uuid4(instance_id, label="control instance id")
    _canonical_uuid4(space_id, label="space id")
    _canonical_uuid4(selector_id, label="control selector id")
    for value, label in (
        (data_dir_id, "data directory identity"),
        (boundary_sha256, "control boundary"),
    ):
        if len(value) != 64 or any(character not in _HEX_DIGEST for character in value):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")


class ServerControlFailure(_StrictModel):
    code: Literal[
        "invalid_request",
        "oversized_request",
        "operation_failed",
        "operation_refused",
        "unauthorized_peer",
        "wrong_instance",
    ]
    message: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_message(self) -> ServerControlFailure:
        if any(ord(character) < 32 or ord(character) == 127 for character in self.message):
            raise ValueError("control error messages must be one safe line")
        return self


class ServerControlResponse(_StrictModel):
    protocol_version: Literal[SERVER_CONTROL_PROTOCOL_VERSION] = SERVER_CONTROL_PROTOCOL_VERSION
    request_id: str | None
    instance_id: str
    ok: bool
    result: (
        ServerControlProbeResult
        | ServerControlMemberPlanResult
        | ServerControlMemberAdvanceResult
        | ServerControlProviderPlanResult
        | ServerControlProviderCheckResult
        | ServerControlProjectPlanResult
        | ServerControlProjectStepResult
        | ServerControlProjectTransferUploadResult
        | ServerControlProjectTransferActivationResult
        | ServerControlBackupCaptureResult
        | ServerControlUpdateResult
        | ServerControlRestoreResult
        | None
    ) = None
    error: ServerControlFailure | None = None

    @model_validator(mode="after")
    def validate_response(self) -> ServerControlResponse:
        _canonical_uuid4(self.instance_id, label="control instance id")
        if self.request_id is not None:
            _canonical_uuid4(self.request_id, label="control request id")
        if self.ok and (self.result is None or self.error is not None):
            raise ValueError("successful control responses require exactly one result")
        if not self.ok and (self.result is not None or self.error is None):
            raise ValueError("control responses must contain exactly one result or error")
        if self.result is not None and self.request_id is None:
            raise ValueError("successful control responses require a request id")
        return self


@dataclass(frozen=True)
class ServerControlPeer:
    pid: int
    uid: int
    gid: int | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.pid, bool)
            or self.pid <= 0
            or isinstance(self.uid, bool)
            or self.uid < 0
            or (self.gid is not None and (isinstance(self.gid, bool) or self.gid < 0))
        ):
            raise ValueError("control peer credentials must contain valid kernel ids")


ServerControlHandler = Callable[
    [ServerControlRequest, ServerControlPeer],
    ServerControlProbeResult
    | ServerControlMemberPlanResult
    | ServerControlMemberAdvanceResult
    | ServerControlProviderPlanResult
    | ServerControlProviderCheckResult
    | ServerControlProjectPlanResult
    | ServerControlProjectStepResult
    | ServerControlProjectTransferUploadResult
    | ServerControlProjectTransferActivationResult
    | ServerControlBackupCaptureResult
    | ServerControlUpdateResult
    | ServerControlRestoreResult,
]
PeerResolver = Callable[[socket.socket], ServerControlPeer]


def unix_peer_identity(connection: socket.socket) -> ServerControlPeer:
    """Read credentials supplied by the Unix-domain socket implementation."""

    if os.name == "posix" and hasattr(socket, "SO_PEERCRED"):
        raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        pid, uid, gid = struct.unpack("3i", raw)
        return ServerControlPeer(pid=pid, uid=uid, gid=gid)
    if sys.platform == "darwin":
        pid = struct.unpack("i", connection.getsockopt(0, 2, 4))[0]
        credential = connection.getsockopt(0, 1, 8)
        _version, uid = struct.unpack("II", credential[:8])
        return ServerControlPeer(pid=pid, uid=uid, gid=None)
    raise ServerControlUnavailable(
        "peer_credentials_unavailable",
        "This operating system cannot authenticate private control-socket peers.",
    )


class ServerControlClient:
    """One-request client that discovers the socket without opening SQLite."""

    def __init__(
        self,
        metadata: ServerMetadata,
        *,
        expected_server_uid: int,
        peer_resolver: PeerResolver = unix_peer_identity,
    ) -> None:
        if metadata.control_socket is None:
            raise ServerControlUnavailable(
                "control_socket_unavailable",
                "The running RCP process does not publish an installed-service control socket.",
            )
        if isinstance(expected_server_uid, bool) or expected_server_uid < 0:
            raise ValueError("the expected control-server uid must be a nonnegative integer")
        self.metadata = metadata
        self.socket_path = Path(metadata.control_socket)
        self.expected_server_uid = expected_server_uid
        self.peer_resolver = peer_resolver

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        *,
        expected_server_uid: int,
        peer_resolver: PeerResolver = unix_peer_identity,
    ) -> ServerControlClient:
        return cls(
            read_server_metadata(data_dir),
            expected_server_uid=expected_server_uid,
            peer_resolver=peer_resolver,
        )

    def probe(self) -> ServerControlProbeResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="probe",
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlProbeResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong control result.",
            )
        return result

    def member_removal_plan(self, member_id: str) -> ServerControlMemberPlanResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="member_removal_plan",
            selector_kind="member",
            selector_id=member_id,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlMemberPlanResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong member-removal plan.",
            )
        return result

    def advance_member_removal(
        self,
        member_id: str,
        *,
        boundary_sha256: str,
    ) -> ServerControlMemberAdvanceResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="member_removal_advance",
            selector_kind="member",
            selector_id=member_id,
            boundary_sha256=boundary_sha256,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlMemberAdvanceResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong member-removal result.",
            )
        return result

    def provider_readiness_plan(
        self,
        *,
        selector_kind: Literal["request", "project"],
        selector_id: str,
    ) -> ServerControlProviderPlanResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="provider_readiness_plan",
            selector_kind=selector_kind,
            selector_id=selector_id,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlProviderPlanResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong provider plan.",
            )
        return result

    def check_provider_readiness(
        self,
        *,
        selector_kind: Literal["request", "project"],
        selector_id: str,
        boundary_sha256: str,
        target_id: str,
    ) -> ServerControlProviderCheckResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="provider_readiness_check",
            selector_kind=selector_kind,
            selector_id=selector_id,
            boundary_sha256=boundary_sha256,
            target_id=target_id,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlProviderCheckResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong provider check.",
            )
        return result

    def project_provision_plan(
        self,
        *,
        request_id: str,
    ) -> ServerControlProjectPlanResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="project_provision_plan",
            selector_kind="request",
            selector_id=request_id,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlProjectPlanResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong project provisioning plan.",
            )
        return result

    def advance_project_provision(
        self,
        *,
        request_id: str,
        boundary_sha256: str,
        target_id: str,
    ) -> ServerControlProjectStepResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="project_provision_step",
            selector_kind="request",
            selector_id=request_id,
            boundary_sha256=boundary_sha256,
            target_id=target_id,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlProjectStepResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong project provisioning step.",
            )
        return result

    def project_transfer_upload_plan(
        self,
        *,
        request_id: str,
    ) -> ServerControlProjectTransferUploadResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="project_transfer_upload_plan",
            selector_kind="request",
            selector_id=request_id,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlProjectTransferUploadResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong project-transfer upload plan.",
            )
        return result

    def complete_project_transfer_upload(
        self,
        *,
        request_id: str,
        lease_boundary_sha256: str,
    ) -> ServerControlProjectTransferUploadResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="project_transfer_upload_complete",
            selector_kind="request",
            selector_id=request_id,
            boundary_sha256=lease_boundary_sha256,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlProjectTransferUploadResult) or result.state not in {
            "complete",
            "consumed",
        }:
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong project-transfer upload result.",
            )
        return result

    def activate_project_transfer(
        self,
        *,
        request_id: str,
        lease_boundary_sha256: str,
    ) -> ServerControlProjectTransferActivationResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="project_transfer_activate",
            selector_kind="request",
            selector_id=request_id,
            boundary_sha256=lease_boundary_sha256,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlProjectTransferActivationResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong project-transfer activation.",
            )
        return result

    def capture_backup_sqlite(self) -> ServerControlBackupCaptureResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="backup_sqlite_capture",
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlBackupCaptureResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong backup capture result.",
            )
        return result

    def activate_restore(
        self,
        *,
        operation_id: str,
        boundary_sha256: str,
    ) -> ServerControlRestoreResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation="restore_activation_commit",
            selector_id=operation_id,
            boundary_sha256=boundary_sha256,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlRestoreResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong restore activation result.",
            )
        return result

    def enter_update_maintenance(
        self,
        *,
        operation_id: str,
        receipt_sha256: str,
    ) -> ServerControlUpdateResult:
        return self._update_operation(
            "update_maintenance_enter",
            operation_id=operation_id,
            receipt_sha256=receipt_sha256,
        )

    def verify_update_candidate(
        self,
        *,
        operation_id: str,
        receipt_sha256: str,
    ) -> ServerControlUpdateResult:
        return self._update_operation(
            "update_candidate_verify",
            operation_id=operation_id,
            receipt_sha256=receipt_sha256,
        )

    def release_update_fence(
        self,
        *,
        operation_id: str,
        receipt_sha256: str,
    ) -> ServerControlUpdateResult:
        return self._update_operation(
            "update_fence_release",
            operation_id=operation_id,
            receipt_sha256=receipt_sha256,
        )

    def abort_update_maintenance(
        self,
        *,
        operation_id: str,
        receipt_sha256: str,
    ) -> ServerControlUpdateResult:
        return self._update_operation(
            "update_maintenance_abort",
            operation_id=operation_id,
            receipt_sha256=receipt_sha256,
        )

    def _update_operation(
        self,
        operation: Literal[
            "update_maintenance_enter",
            "update_candidate_verify",
            "update_fence_release",
            "update_maintenance_abort",
        ],
        *,
        operation_id: str,
        receipt_sha256: str,
    ) -> ServerControlUpdateResult:
        request = ServerControlRequest(
            request_id=str(uuid.uuid4()),
            instance_id=self.metadata.instance_id,
            operation=operation,
            selector_id=operation_id,
            boundary_sha256=receipt_sha256,
        )
        result = self._exchange(request)
        if not isinstance(result, ServerControlUpdateResult):
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned the wrong update result.",
            )
        return result

    def _exchange(
        self,
        request: ServerControlRequest,
    ) -> (
        ServerControlProbeResult
        | ServerControlMemberPlanResult
        | ServerControlMemberAdvanceResult
        | ServerControlProviderPlanResult
        | ServerControlProviderCheckResult
        | ServerControlProjectPlanResult
        | ServerControlProjectStepResult
        | ServerControlProjectTransferUploadResult
        | ServerControlProjectTransferActivationResult
        | ServerControlBackupCaptureResult
        | ServerControlUpdateResult
        | ServerControlRestoreResult
    ):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        timeout = SERVER_CONTROL_IO_TIMEOUT_SECONDS
        if request.operation == "backup_sqlite_capture":
            timeout = SERVER_CONTROL_BACKUP_CAPTURE_TIMEOUT_SECONDS
        elif request.operation == "update_maintenance_enter":
            timeout = SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS
        elif request.operation in {
            "update_candidate_verify",
            "update_fence_release",
            "update_maintenance_abort",
            "restore_activation_commit",
        }:
            timeout = SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS
        elif request.operation == "provider_readiness_check":
            timeout = SERVER_CONTROL_PROVIDER_CHECK_TIMEOUT_SECONDS
        elif request.operation == "project_provision_step":
            timeout = SERVER_CONTROL_PROJECT_PROVISION_TIMEOUT_SECONDS
        elif request.operation in {
            "project_transfer_upload_complete",
            "project_transfer_activate",
        }:
            # Completion re-hashes the request-bound archive before recording
            # its receipt, so it gets the same bounded long operation window
            # as other archive-sized service work.
            timeout = SERVER_CONTROL_PROJECT_PROVISION_TIMEOUT_SECONDS
        connection.settimeout(timeout)
        try:
            connection.connect(str(self.socket_path))
            peer = self.peer_resolver(connection)
            if peer.uid != self.expected_server_uid:
                raise ServerControlUnavailable(
                    "wrong_server_identity",
                    "The control socket is not owned by the expected RCP service process.",
                )
            _send_model(
                connection,
                request,
                maximum=SERVER_CONTROL_MAX_REQUEST_BYTES,
            )
            response = _receive_response(connection)
        except ServerControlError:
            raise
        except (OSError, TimeoutError) as exc:
            raise ServerControlUnavailable(
                "control_socket_unavailable",
                "The running RCP control socket is unavailable.",
            ) from exc
        finally:
            connection.close()
        if response.instance_id != request.instance_id:
            raise ServerControlError(
                "wrong_instance",
                "The control response came from a different RCP process instance.",
            )
        if response.request_id not in {None, request.request_id}:
            raise ServerControlError(
                "wrong_request",
                "The control response does not match this request.",
            )
        if not response.ok:
            assert response.error is not None
            raise ServerControlError(response.error.code, response.error.message)
        assert response.result is not None
        try:
            result = _validated_control_result(request, response.result)
        except ValueError as exc:
            raise ServerControlError(
                "invalid_response",
                "The running RCP process returned a mismatched control result.",
            ) from exc
        if result.pid != peer.pid:
            raise ServerControlError(
                "wrong_server_identity",
                "The control response does not match the kernel-authenticated server process.",
            )
        return result


class ServerControlServer:
    """Single-process owner of one private Unix-domain control socket."""

    def __init__(
        self,
        socket_path: Path,
        *,
        instance_id: str,
        owner_uid: int,
        owner_gid: int,
        handler: ServerControlHandler,
        peer_resolver: PeerResolver = unix_peer_identity,
    ) -> None:
        _canonical_uuid4(instance_id, label="control instance id")
        if any(isinstance(value, bool) or value < 0 for value in (owner_uid, owner_gid)):
            raise ValueError("control socket owner ids must be nonnegative integers")
        self.socket_path = _validated_socket_path(socket_path)
        self.instance_id = instance_id
        self.owner_uid = owner_uid
        self.owner_gid = owner_gid
        self.handler = handler
        self.peer_resolver = peer_resolver
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the server control socket is already started")
        _validate_runtime_directory(
            self.socket_path.parent,
            owner_uid=self.owner_uid,
            owner_gid=self.owner_gid,
        )
        self._recover_stale_socket()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, SERVER_CONTROL_SOCKET_MODE)
            info = self.socket_path.lstat()
            if (
                not stat.S_ISSOCK(info.st_mode)
                or info.st_uid != self.owner_uid
                or info.st_gid != self.owner_gid
                or stat.S_IMODE(info.st_mode) != SERVER_CONTROL_SOCKET_MODE
            ):
                raise ServerControlUnavailable(
                    "unsafe_control_socket",
                    "The private control socket has unsafe ownership or mode.",
                )
            listener.listen(16)
            listener.settimeout(SERVER_CONTROL_ACCEPT_POLL_INTERVAL_SECONDS)
            self._listener = listener
            self._socket_identity = (info.st_dev, info.st_ino)
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._serve,
                name="rcp-server-control",
                daemon=False,
            )
            self._thread.start()
        except Exception:
            listener.close()
            self._remove_owned_socket()
            self._listener = None
            self._socket_identity = None
            raise

    def stop(self) -> None:
        thread = self._thread
        listener = self._listener
        if thread is None:
            return
        self._stop.set()
        if listener is not None:
            listener.close()
        thread.join(timeout=SERVER_CONTROL_STOP_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise RuntimeError("the private control server did not stop at a durable boundary")
        self._remove_owned_socket()
        self._thread = None
        self._listener = None
        self._socket_identity = None

    def _serve(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError as exc:
                if self._stop.is_set() or exc.errno in {errno.EBADF, errno.EINVAL}:
                    return
                continue
            with connection:
                connection.settimeout(SERVER_CONTROL_IO_TIMEOUT_SECONDS)
                self._serve_one(connection)

    def _serve_one(self, connection: socket.socket) -> None:
        request_id: str | None = None
        try:
            peer = self.peer_resolver(connection)
            if peer.uid not in {0, self.owner_uid}:
                self._send_error(
                    connection,
                    request_id=None,
                    code="unauthorized_peer",
                    message="This operating-system account cannot use the RCP control socket.",
                )
                return
            raw = _receive_json(connection, maximum=SERVER_CONTROL_MAX_REQUEST_BYTES)
            try:
                request = ServerControlRequest.model_validate(raw)
            except Exception:
                self._send_error(
                    connection,
                    request_id=None,
                    code="invalid_request",
                    message="The control request has an unsupported shape.",
                )
                return
            request_id = request.request_id
            if request.instance_id != self.instance_id:
                self._send_error(
                    connection,
                    request_id=request_id,
                    code="wrong_instance",
                    message="The control request names a different RCP process instance.",
                )
                return
            try:
                result = _validated_control_result(request, self.handler(request, peer))
                if result.instance_id != self.instance_id:
                    raise ValueError("control handler returned a different process instance")
            except ServerControlError as exc:
                if exc.code != "operation_refused":
                    self._send_error(
                        connection,
                        request_id=request_id,
                        code="operation_failed",
                        message=(
                            "The named control operation failed inside the running RCP process."
                        ),
                    )
                    return
                self._send_error(
                    connection,
                    request_id=request_id,
                    code="operation_refused",
                    message=_safe_operation_refusal(str(exc)),
                )
                return
            except Exception:
                self._send_error(
                    connection,
                    request_id=request_id,
                    code="operation_failed",
                    message="The named control operation failed inside the running RCP process.",
                )
                return
            _send_model(
                connection,
                ServerControlResponse(
                    request_id=request_id,
                    instance_id=self.instance_id,
                    ok=True,
                    result=result,
                ),
                maximum=SERVER_CONTROL_MAX_RESPONSE_BYTES,
            )
        except ServerControlError as exc:
            code = "oversized_request" if exc.code == "oversized_frame" else "invalid_request"
            self._send_error(
                connection,
                request_id=request_id,
                code=code,
                message=(
                    "The control request exceeds its fixed size limit."
                    if code == "oversized_request"
                    else "The control request is malformed or incomplete."
                ),
            )
        except (OSError, TimeoutError):
            return

    def _send_error(
        self,
        connection: socket.socket,
        *,
        request_id: str | None,
        code: Literal[
            "invalid_request",
            "oversized_request",
            "operation_failed",
            "operation_refused",
            "unauthorized_peer",
            "wrong_instance",
        ],
        message: str,
    ) -> None:
        with suppress(OSError, ServerControlError, TimeoutError):
            _send_model(
                connection,
                ServerControlResponse(
                    request_id=request_id,
                    instance_id=self.instance_id,
                    ok=False,
                    error=ServerControlFailure(code=code, message=message),
                ),
                maximum=SERVER_CONTROL_MAX_RESPONSE_BYTES,
            )

    def _recover_stale_socket(self) -> None:
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if (
            not stat.S_ISSOCK(info.st_mode)
            or info.st_uid != self.owner_uid
            or info.st_gid != self.owner_gid
            or stat.S_IMODE(info.st_mode) != SERVER_CONTROL_SOCKET_MODE
        ):
            raise ServerControlUnavailable(
                "unsafe_control_socket",
                "The existing control-socket path is not a safe stale RCP socket.",
            )
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(SERVER_CONTROL_IO_TIMEOUT_SECONDS)
        try:
            probe.connect(str(self.socket_path))
        except OSError as exc:
            if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                raise ServerControlUnavailable(
                    "control_socket_unavailable",
                    "The existing control-socket path cannot be recovered safely.",
                ) from exc
        else:
            raise ServerControlUnavailable(
                "control_socket_occupied",
                "Another process already owns the installed RCP control socket.",
            )
        finally:
            probe.close()
        try:
            current = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise ServerControlUnavailable(
                "control_socket_changed",
                "The existing control-socket path changed during recovery.",
            )
        self.socket_path.unlink()

    def _remove_owned_socket(self) -> None:
        identity = self._socket_identity
        if identity is None:
            return
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == identity:
            self.socket_path.unlink()


def _validated_control_result(
    request: ServerControlRequest,
    result: ServerControlProbeResult
    | ServerControlMemberPlanResult
    | ServerControlMemberAdvanceResult
    | ServerControlProviderPlanResult
    | ServerControlProviderCheckResult
    | ServerControlProjectPlanResult
    | ServerControlProjectStepResult
    | ServerControlProjectTransferUploadResult
    | ServerControlProjectTransferActivationResult
    | ServerControlBackupCaptureResult
    | ServerControlUpdateResult
    | ServerControlRestoreResult,
) -> (
    ServerControlProbeResult
    | ServerControlMemberPlanResult
    | ServerControlMemberAdvanceResult
    | ServerControlProviderPlanResult
    | ServerControlProviderCheckResult
    | ServerControlProjectPlanResult
    | ServerControlProjectStepResult
    | ServerControlProjectTransferUploadResult
    | ServerControlProjectTransferActivationResult
    | ServerControlBackupCaptureResult
    | ServerControlUpdateResult
    | ServerControlRestoreResult
):
    if request.operation == "probe":
        if not isinstance(result, ServerControlProbeResult):
            raise ValueError("control probe returned another operation's result")
        return ServerControlProbeResult.model_validate(result)
    if request.operation == "provider_readiness_plan":
        if not isinstance(result, ServerControlProviderPlanResult):
            raise ValueError("provider readiness plan returned another operation's result")
        validated = ServerControlProviderPlanResult.model_validate(result)
        if (
            validated.selector_kind != request.selector_kind
            or validated.selector_id != request.selector_id
        ):
            raise ValueError("provider readiness plan returned another selector")
        return validated
    if request.operation == "member_removal_plan":
        if not isinstance(result, ServerControlMemberPlanResult):
            raise ValueError("member-removal plan returned another operation's result")
        validated_member_plan = ServerControlMemberPlanResult.model_validate(result)
        if validated_member_plan.snapshot.member_id != request.selector_id:
            raise ValueError("member-removal plan returned another member")
        return validated_member_plan
    if request.operation == "member_removal_advance":
        if not isinstance(result, ServerControlMemberAdvanceResult):
            raise ValueError("member-removal advance returned another operation's result")
        validated_member_advance = ServerControlMemberAdvanceResult.model_validate(result)
        if (
            validated_member_advance.snapshot.member_id != request.selector_id
            or validated_member_advance.confirmed_boundary_sha256 != request.boundary_sha256
        ):
            raise ValueError("member-removal advance returned another confirmed boundary")
        return validated_member_advance
    if request.operation == "provider_readiness_check":
        if not isinstance(result, ServerControlProviderCheckResult):
            raise ValueError("provider readiness check returned another operation's result")
        validated_provider = ServerControlProviderCheckResult.model_validate(result)
        if (
            validated_provider.selector_kind != request.selector_kind
            or validated_provider.selector_id != request.selector_id
            or validated_provider.boundary_sha256 != request.boundary_sha256
            or validated_provider.target_id != request.target_id
        ):
            raise ValueError("provider readiness check returned another planned target")
        return validated_provider
    if request.operation == "project_provision_plan":
        if not isinstance(result, ServerControlProjectPlanResult):
            raise ValueError("project provisioning plan returned another operation's result")
        validated_plan = ServerControlProjectPlanResult.model_validate(result)
        if validated_plan.request_id != request.selector_id:
            raise ValueError("project provisioning plan returned another request")
        return validated_plan
    if request.operation == "project_transfer_upload_plan":
        if not isinstance(result, ServerControlProjectTransferUploadResult):
            raise ValueError("project transfer upload plan returned another operation's result")
        validated_upload_plan = ServerControlProjectTransferUploadResult.model_validate(result)
        if validated_upload_plan.request_id != request.selector_id:
            raise ValueError("project transfer upload plan returned another request")
        return validated_upload_plan
    if request.operation == "project_transfer_upload_complete":
        if not isinstance(result, ServerControlProjectTransferUploadResult):
            raise ValueError(
                "project transfer upload completion returned another operation's result"
            )
        validated_upload = ServerControlProjectTransferUploadResult.model_validate(result)
        if (
            validated_upload.request_id != request.selector_id
            or validated_upload.lease_boundary_sha256 != request.boundary_sha256
            or validated_upload.state not in {"complete", "consumed"}
        ):
            raise ValueError("project transfer upload returned another request or lease boundary")
        return validated_upload
    if request.operation == "project_transfer_activate":
        if not isinstance(result, ServerControlProjectTransferActivationResult):
            raise ValueError("project transfer activation returned another operation's result")
        validated_activation = ServerControlProjectTransferActivationResult.model_validate(result)
        if (
            validated_activation.target_request_id != request.selector_id
            or validated_activation.upload_lease_boundary_sha256 != request.boundary_sha256
        ):
            raise ValueError("project transfer activation returned another request or lease")
        return validated_activation
    if request.operation == "backup_sqlite_capture":
        if not isinstance(result, ServerControlBackupCaptureResult):
            raise ValueError("backup SQLite capture returned another operation's result")
        return ServerControlBackupCaptureResult.model_validate(result)
    if request.operation == "restore_activation_commit":
        if not isinstance(result, ServerControlRestoreResult):
            raise ValueError("restore activation returned another operation's result")
        validated_restore = ServerControlRestoreResult.model_validate(result)
        if (
            validated_restore.operation_id != request.selector_id
            or validated_restore.boundary_sha256 != request.boundary_sha256
        ):
            raise ValueError("restore activation returned another operation boundary")
        return validated_restore
    if request.operation in {
        "update_maintenance_enter",
        "update_candidate_verify",
        "update_fence_release",
        "update_maintenance_abort",
    }:
        if not isinstance(result, ServerControlUpdateResult):
            raise ValueError("update control returned another operation's result")
        validated_update = ServerControlUpdateResult.model_validate(result)
        if validated_update.operation_id != request.selector_id:
            raise ValueError("update control returned another operation")
        return validated_update
    if not isinstance(result, ServerControlProjectStepResult):
        raise ValueError("project provisioning step returned another operation's result")
    validated_step = ServerControlProjectStepResult.model_validate(result)
    if (
        validated_step.request_id != request.selector_id
        or validated_step.boundary_sha256 != request.boundary_sha256
        or validated_step.target_id != request.target_id
    ):
        raise ValueError("project provisioning step returned another planned target")
    return validated_step


def _safe_operation_refusal(message: str) -> str:
    safe = redact_server_text(message.strip())
    if (
        not safe
        or len(safe) > 240
        or any(ord(character) < 32 or ord(character) == 127 for character in safe)
    ):
        return "The named control operation was refused at its durable boundary."
    return safe


def _validated_socket_path(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("the control socket path must be absolute and normalized")
    if len(os.fsencode(path)) > SERVER_CONTROL_MAX_SOCKET_PATH_BYTES:
        raise ValueError("the control socket path is too long for the supported Unix kernels")
    return path


def _validate_runtime_directory(path: Path, *, owner_uid: int, owner_gid: int) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ServerControlUnavailable(
            "runtime_directory_unavailable",
            "The installed RCP runtime directory is unavailable.",
        ) from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or path.is_symlink()
        or info.st_uid != owner_uid
        or info.st_gid != owner_gid
        or stat.S_IMODE(info.st_mode) != SERVER_CONTROL_RUNTIME_MODE
    ):
        raise ServerControlUnavailable(
            "unsafe_runtime_directory",
            "The installed RCP runtime directory has unsafe ownership or mode.",
        )


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ServerControlError("incomplete_frame", "The control frame is incomplete.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_json(connection: socket.socket, *, maximum: int) -> object:
    body = _receive_body(connection, maximum=maximum)
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServerControlError("invalid_frame", "The control frame is not valid JSON.") from exc


def _receive_body(connection: socket.socket, *, maximum: int) -> bytes:
    header = _receive_exact(connection, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    if size == 0:
        raise ServerControlError("invalid_frame", "The control frame is empty.")
    if size > maximum:
        raise ServerControlError("oversized_frame", "The control frame is too large.")
    return _receive_exact(connection, size)


def _receive_response(connection: socket.socket) -> ServerControlResponse:
    try:
        return ServerControlResponse.model_validate_json(
            _receive_body(connection, maximum=SERVER_CONTROL_MAX_RESPONSE_BYTES)
        )
    except ServerControlError:
        raise
    except Exception as exc:
        raise ServerControlError(
            "invalid_response",
            "The running RCP process returned an invalid control response.",
        ) from exc


def _send_model(connection: socket.socket, model: BaseModel, *, maximum: int) -> None:
    body = model.model_dump_json().encode("utf-8")
    if not body or len(body) > maximum:
        raise ServerControlError("oversized_frame", "The control frame exceeds its size limit.")
    connection.sendall(_FRAME_HEADER.pack(len(body)) + body)


__all__ = [
    "SERVER_CONTROL_OPERATIONS",
    "SERVER_CONTROL_MAX_REQUEST_BYTES",
    "SERVER_CONTROL_MAX_RESPONSE_BYTES",
    "SERVER_CONTROL_PROTOCOL_VERSION",
    "SERVER_CONTROL_RUNTIME_MODE",
    "SERVER_CONTROL_SOCKET_MODE",
    "ServerControlClient",
    "ServerControlBackupCaptureResult",
    "ServerControlError",
    "ServerControlHandler",
    "ServerControlMemberAdvanceResult",
    "ServerControlMemberPlanResult",
    "ServerControlMemberSnapshot",
    "ServerControlPeer",
    "ServerControlProbeResult",
    "ServerControlProjectPlanResult",
    "ServerControlProjectStepResult",
    "ServerControlProjectTransferActivationResult",
    "ServerControlProjectTransferUploadResult",
    "ServerControlProjectTarget",
    "ServerControlProviderCheckResult",
    "ServerControlProviderPlanResult",
    "ServerControlProviderTarget",
    "ServerControlRequest",
    "ServerControlRestoreResult",
    "ServerControlServer",
    "ServerControlUpdateResult",
    "ServerControlUnavailable",
    "unix_peer_identity",
]
