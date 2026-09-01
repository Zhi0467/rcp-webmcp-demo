"""Prepare one exact source-built server update without touching the live release."""

from __future__ import annotations

import fcntl
import hashlib
import os
import pwd
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, BinaryIO, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator

from rcp.limits import (
    SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS,
    SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS,
    SERVER_INSTALL_BUILD_TIMEOUT_SECONDS,
    SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
    SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
    SERVER_UPDATE_CHECKPOINT_TIMEOUT_SECONDS,
    SERVER_UPDATE_REHEARSAL_TIMEOUT_SECONDS,
)
from rcp.server_ops.cli import CallerIdentity, PreparedServerCommand, ServerEventEmitter
from rcp.server_ops.config import InstalledServerConfig, load_installed_server_config
from rcp.server_ops.control import ServerControlClient, ServerControlError
from rcp.server_ops.doctor import (
    LinuxServerDoctorMachine,
    ServerDoctorMachine,
    ServerDoctorReport,
)
from rcp.server_ops.install import (
    InstalledServiceControlRefused,
    InstalledSystemServiceController,
    _run_as_account,
    source_git_environment,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.models import (
    CommandAction,
    ExternalAction,
    MachineTarget,
    NonsecretField,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
)
from rcp.server_runtime import ServerMetadataError, web_build_identity

if TYPE_CHECKING:
    from rcp.server_ops.rehearsal import VerifiedCandidateReceipt
    from rcp.server_ops.update_checkpoint import VerifiedUpdateCheckpoint
    from rcp.server_ops.update_cutover import UpdateCutoverOutcome, UpdateOperationReceipt

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_WEB_BUILD_ID = re.compile(r"sha256:[0-9a-f]{64}")
_RECEIPT_NAME = re.compile(r"built-candidate-([0-9a-f]{40})\.json")
_VERIFIED_RECEIPT_NAME = re.compile(r"verified-candidate-([0-9a-f]{40})-([0-9a-f-]{36})\.json")
_REHEARSAL_ROOT_NAME = re.compile(r"rehearsal-([0-9a-f]{40})-([0-9a-f]{32})")
_UPDATE_OPERATION_NAME = re.compile(r"update-operation-([0-9a-f]{32})\.json")
_CHECKPOINT_ROOT_NAME = re.compile(r"checkpoint-([0-9a-f]{40})-([0-9a-f]{32})")
_UPDATE_LOCK_NAME = ".server-update.lock"
_RECEIPT_SCHEMA_VERSION = 1
_RECEIPT_MODE = 0o600
_LOCK_MODE = 0o600
_DIRECTORY_MODE = 0o700
_CONFIG_DIRECTORY_MODE = 0o750
_MAX_RECEIPT_BYTES = 16 * 1024
_MAX_VERIFIED_RECEIPT_BYTES = 4 * 1024 * 1024


class UpdateRefused(RuntimeError):
    """One safe update refusal whose text may enter the operator event stream."""


class _ReportedUpdateFailure(RuntimeError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class BuiltCandidateReceipt(_StrictModel):
    """Immutable handoff from F6a source/build work to candidate rehearsal."""

    schema_version: Literal[1] = _RECEIPT_SCHEMA_VERSION
    installation_id: str
    source_origin: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    source_branch: Literal["main"] = "main"
    base_current_commit: str
    base_running_commit: str
    base_instance_id: str
    base_process_pid: int
    candidate_commit: str
    release_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    receipt_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    web_build_id: str
    prepared_at: datetime

    @field_validator("source_origin")
    @classmethod
    def validate_source_origin(cls, value: str) -> str:
        if value != value.strip() or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("candidate receipt source origin must be one trimmed line")
        return value

    @field_validator("base_current_commit", "base_running_commit", "candidate_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if _FULL_GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("candidate receipts require full lowercase Git object ids")
        return value

    @field_validator("web_build_id")
    @classmethod
    def validate_web_build(cls, value: str) -> str:
        if _WEB_BUILD_ID.fullmatch(value) is None:
            raise ValueError("candidate receipts require one SHA-256 Web build identity")
        return value

    @field_validator("prepared_at")
    @classmethod
    def validate_prepared_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("candidate receipt time must include a UTC offset")
        return value

    @model_validator(mode="after")
    def validate_relationship(self) -> BuiltCandidateReceipt:
        if self.base_current_commit != self.base_running_commit:
            raise ValueError("candidate receipt base must name one current/running release")
        if self.candidate_commit == self.base_running_commit:
            raise ValueError("candidate receipt must name a different target release")
        if self.base_process_pid <= 0:
            raise ValueError("candidate receipt requires one positive base process id")
        for label, value in (
            ("release", self.release_path),
            ("receipt", self.receipt_path),
        ):
            path = Path(value)
            if not path.is_absolute() or ".." in path.parts or str(path) != value:
                raise ValueError(f"candidate receipt {label} path must be absolute and normalized")
        return self


@dataclass(frozen=True)
class UpdateInspection:
    config: InstalledServerConfig
    managed_head: str
    current_commit: str
    running_commit: str
    instance_id: str
    process_pid: int


@dataclass(frozen=True)
class UpdateTarget:
    inspection: UpdateInspection
    target_commit: str

    @property
    def already_current(self) -> bool:
        return (
            self.inspection.managed_head
            == self.target_commit
            == self.inspection.current_commit
            == self.inspection.running_commit
        )


@dataclass(frozen=True)
class CandidateBuild:
    commit: str
    release_path: Path
    web_build_id: str
    reused_receipt: bool


class ServiceCommandRunner(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        environment: dict[str, str] | None,
        timeout: float,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]: ...


class UpdateMachine(Protocol):
    def admission(self) -> AbstractContextManager[None]: ...

    def inspect(self) -> UpdateInspection: ...

    def status(self) -> UpdateInspection: ...

    def fetch_target(self, inspection: UpdateInspection) -> UpdateTarget: ...

    def fast_forward(self, target: UpdateTarget) -> None: ...

    def prepare_release(self, target: UpdateTarget) -> Path: ...

    def build_candidate(self, target: UpdateTarget, release: Path) -> CandidateBuild: ...

    def finalize_candidate(
        self,
        target: UpdateTarget,
        build: CandidateBuild,
    ) -> BuiltCandidateReceipt: ...

    def rehearse_candidate(
        self,
        target: UpdateTarget,
        built: BuiltCandidateReceipt,
    ) -> VerifiedCandidateReceipt: ...

    def create_rollback_checkpoint(
        self,
        target: UpdateTarget,
        verified: VerifiedCandidateReceipt,
        *,
        sqlite_receipt_path: Path,
        sqlite_receipt_sha256: str,
        project_receipt_path: Path,
        project_receipt_sha256: str,
    ) -> VerifiedUpdateCheckpoint: ...

    def cutover_candidate(
        self,
        target: UpdateTarget,
        built: BuiltCandidateReceipt,
        preflight: VerifiedCandidateReceipt,
        *,
        progress: Callable[[str], None],
    ) -> UpdateCutoverOutcome: ...


def prepare_update_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    machine: UpdateMachine | None = None,
    resume_executable: Path = DEFAULT_SERVER_LAYOUT.cli_wrapper,
) -> PreparedServerCommand:
    if request.command != "server update":
        raise ValueError("prepare_update_command requires one server update request")
    plan = ServerPlanEvent(
        command=request.command,
        timestamp=datetime.now(UTC),
        steps=_update_plan(identity),
    )

    def execute(emitter: ServerEventEmitter, _input_stream: BinaryIO) -> None:
        _execute_update(
            request,
            emitter,
            machine or LinuxUpdateMachine(),
            resume_executable=resume_executable,
        )

    return PreparedServerCommand(plan=plan, execute=execute)


def built_candidate_receipt_path(
    commit: str,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> Path:
    _require_commit(commit)
    return layout.update_checkpoints_root / f"built-candidate-{commit}.json"


def _update_plan(identity: CallerIdentity) -> tuple[ServerStep, ...]:
    root = MachineTarget(host=identity.host, os_account="root")
    service = MachineTarget(host=identity.host, os_account="rcp")
    return (
        ServerStep(
            number=1,
            title="Admit one safe source update",
            purpose=(
                "Serialize update preparation and prove the installed service has one coherent "
                "source, current release, running process, and maintenance boundary."
            ),
            performed_by="system",
            target=root,
            phase="update_admission",
            state="pending",
            expected_success=(
                "No update or restore maintenance conflicts, and current equals the healthy "
                "running release."
            ),
            message="RCP will acquire the update lock and inspect the installed service.",
        ),
        ServerStep(
            number=2,
            title="Fetch and verify the exact main target",
            purpose=(
                "Use only the configured source identity as rcp and refuse dirty, non-main, "
                "ahead, or diverged source state."
            ),
            performed_by="system",
            target=service,
            phase="update_source_fetch",
            state="pending",
            expected_success="The clean managed checkout can fast-forward to one exact origin/main commit.",
            message="RCP will fetch and compare the current and target commits as rcp.",
        ),
        ServerStep(
            number=3,
            title="Confirm the exact target commit",
            purpose="Bind operator approval to the fetched commit so a later main advance cannot be substituted.",
            performed_by="human",
            target=root,
            phase="update_target_confirm",
            state="pending",
            expected_success="The confirmed 40-character commit exactly matches fetched origin/main.",
            message="RCP will request confirmation only when a new target exists.",
        ),
        ServerStep(
            number=4,
            title="Fast-forward the managed main checkout",
            purpose="Advance only by Git fast-forward without reset, force, stash, branch choice, or package fallback.",
            performed_by="system",
            target=service,
            phase="update_source_fast_forward",
            state="pending",
            expected_success="Managed main is clean at the exact confirmed target.",
            message="RCP will fast-forward the managed main checkout as rcp.",
        ),
        ServerStep(
            number=5,
            title="Prepare the immutable candidate release",
            purpose="Create or validate one detached, clean, service-owned worktree for the exact target commit.",
            performed_by="system",
            target=service,
            phase="update_release_prepare",
            state="pending",
            expected_success="The candidate release path contains only the exact confirmed Git commit.",
            message="RCP will create or validate the separate candidate worktree as rcp.",
        ),
        ServerStep(
            number=6,
            title="Build the source candidate",
            purpose="Run npm ci, the production Web build, and uv sync --frozen inside only the candidate release.",
            performed_by="system",
            target=service,
            phase="update_candidate_build",
            state="pending",
            expected_success="The candidate has one exact Web identity and executable Python 3.12 environment.",
            message="RCP will build and validate the isolated candidate as rcp.",
        ),
        ServerStep(
            number=7,
            title="Publish the built-candidate receipt",
            purpose="Verify the old service stayed unchanged and durably bind the exact candidate for rehearsal.",
            performed_by="system",
            target=root,
            phase="update_candidate_receipt",
            state="pending",
            expected_success="One owner-only immutable receipt binds the unchanged base and built candidate.",
            message="RCP will read back identities and publish the candidate handoff.",
        ),
        ServerStep(
            number=8,
            title="Rehearse the candidate against copied server state",
            purpose=(
                "Reuse the online SQLite and project-file capture, rebind every local runtime "
                "path into a private overlay, and run candidate startup and representative "
                "reads with all external effects closed."
            ),
            performed_by="system",
            target=service,
            phase="update_candidate_rehearsal",
            state="pending",
            expected_success=(
                "Every captured project replays or retains one proven pre-existing SSH "
                "unavailable projection, with no provider, watcher, Git, cleanup, or remote effect."
            ),
            message="RCP will capture live state once and run the candidate only on its overlay.",
        ),
        ServerStep(
            number=9,
            title="Close admission and capture the final boundary",
            purpose=(
                "Stop new mutations and launches, wait for active providers, and bind one final "
                "SQLite and project-file capture to this update."
            ),
            performed_by="system",
            target=service,
            phase="update_maintenance_close",
            state="pending",
            expected_success="The old process is idle with normal work closed and one exact capture.",
            message="RCP will enter the short update maintenance window.",
        ),
        ServerStep(
            number=10,
            title="Verify the final rollback checkpoint",
            purpose=(
                "Fresh-rehearse the candidate on the closed boundary and snapshot every owned "
                "local recovery input before any pointer switch."
            ),
            performed_by="system",
            target=service,
            phase="update_rollback_checkpoint",
            state="pending",
            expected_success="One complete verified checkpoint names the exact final rehearsal.",
            message="RCP will prove the exact rollback bytes before switching releases.",
        ),
        ServerStep(
            number=11,
            title="Switch and restart behind the effect fence",
            purpose=(
                "Stop systemd, atomically switch current, and start the candidate while all "
                "deferred runtime effects remain fenced."
            ),
            performed_by="system",
            target=root,
            phase="update_release_cutover",
            state="pending",
            expected_success="systemd runs the exact candidate commit behind the startup fence.",
            message="RCP will perform the bounded root-only release switch.",
        ),
        ServerStep(
            number=12,
            title="Verify the switched candidate",
            purpose=(
                "Repeat startup, recovery, project projection, task, and watcher reads against "
                "the real data while external effects remain closed."
            ),
            performed_by="system",
            target=service,
            phase="update_candidate_readback",
            state="pending",
            expected_success="The running candidate exactly matches the final rehearsal read model.",
            message="RCP will verify the candidate process before committing the update.",
        ),
        ServerStep(
            number=13,
            title="Commit the release and reopen work",
            purpose=(
                "Durably choose the verified release, release the one startup fence exactly "
                "once, and start deferred runtime owners."
            ),
            performed_by="system",
            target=service,
            phase="update_fence_release",
            state="pending",
            expected_success="The verified candidate serves normally, or loud rollback restores the base.",
            message="RCP will commit the candidate or report the exact restored base release.",
        ),
    )


def _execute_update(
    request: ServerCommandRequest,
    emitter: ServerEventEmitter,
    machine: UpdateMachine,
    *,
    resume_executable: Path,
) -> None:
    planned = emitter.events[0]
    if not isinstance(planned, ServerPlanEvent):  # pragma: no cover - emitter owns this
        raise AssertionError("update execution requires its plan")
    steps = planned.steps
    emitter.emit_step(
        steps[0].model_copy(
            update={
                "state": "running",
                "message": "Acquiring the update admission lock and reading installed identities.",
            }
        )
    )
    try:
        with machine.admission():
            try:
                inspection = machine.inspect()
            except UpdateRefused as exc:
                emitter.emit_step(
                    steps[0].model_copy(update={"state": "failed", "message": str(exc)})
                )
                return
            emitter.emit_step(
                steps[0].model_copy(
                    update={
                        "state": "succeeded",
                        "message": "The installed service and update-maintenance boundary are coherent.",
                        "fields": _inspection_fields(inspection),
                    }
                )
            )
            _execute_admitted_update(
                request,
                emitter,
                machine,
                inspection,
                steps=steps,
                resume_executable=resume_executable,
            )
    except UpdateRefused as exc:
        emitter.emit_step(steps[0].model_copy(update={"state": "failed", "message": str(exc)}))
    except _ReportedUpdateFailure:
        return


def _execute_admitted_update(
    request: ServerCommandRequest,
    emitter: ServerEventEmitter,
    machine: UpdateMachine,
    inspection: UpdateInspection,
    *,
    steps: tuple[ServerStep, ...],
    resume_executable: Path,
) -> None:
    target = _run_step(
        emitter,
        steps[1],
        running="Fetching origin/main with the configured source identity and checking fast-forward safety.",
        operation=lambda: machine.fetch_target(inspection),
        succeeded="The managed checkout and fetched target have one safe fast-forward relationship.",
        fields=lambda value: _target_fields(value),
        failure_fields=lambda: _fetch_failure_fields(inspection),
    )
    if request.update_confirmed_commit is None:
        if target.already_current:
            _complete_already_current(emitter, steps[2:], target)
            return
        _emit_confirmation_pause(
            emitter,
            steps[2],
            target,
            resume_executable=resume_executable,
        )
        return
    _run_step(
        emitter,
        steps[2],
        running="Comparing the operator-confirmed commit with the freshly fetched target.",
        operation=lambda: _require_confirmed_target(request.update_confirmed_commit, target),
        succeeded="The operator confirmation names this exact fetched target.",
        fields=lambda _value: (
            NonsecretField(name="confirmed_commit", value=target.target_commit),
        ),
        failure_fields=lambda: _target_fields(target),
    )
    if target.already_current:
        _complete_already_current(emitter, steps[3:], target, confirmation_completed=True)
        return
    _run_step(
        emitter,
        steps[3],
        running="Fast-forwarding managed main to the confirmed commit without rewriting history.",
        operation=lambda: machine.fast_forward(target),
        succeeded="Managed main is clean at the confirmed target commit.",
        fields=lambda _value: _status_fields(target, managed=target.target_commit),
        failure_fields=lambda: _failure_status_fields(machine, target),
    )
    release = _run_step(
        emitter,
        steps[4],
        running="Creating or validating the detached per-commit candidate worktree.",
        operation=lambda: machine.prepare_release(target),
        succeeded="The separate candidate worktree has the exact confirmed Git identity.",
        fields=lambda value: (
            NonsecretField(name="candidate_commit", value=target.target_commit),
            NonsecretField(name="release_path", value=str(value)),
        ),
        failure_fields=lambda: _failure_status_fields(machine, target),
    )
    build = _run_step(
        emitter,
        steps[5],
        running="Running npm ci, the Web build, and uv sync --frozen only in the candidate.",
        operation=lambda: machine.build_candidate(target, release),
        succeeded="The isolated candidate Web and Python source build is ready.",
        fields=lambda value: (
            NonsecretField(name="candidate_commit", value=value.commit),
            NonsecretField(name="web_build_id", value=value.web_build_id),
            NonsecretField(name="reused_receipt", value=value.reused_receipt),
        ),
        failure_fields=lambda: _failure_status_fields(machine, target),
    )
    built = _run_step(
        emitter,
        steps[6],
        running="Rechecking the unchanged live service and publishing the owner-only build receipt.",
        operation=lambda: machine.finalize_candidate(target, build),
        succeeded="The built candidate is durably bound for the separate rehearsal packet.",
        fields=lambda value: (
            NonsecretField(name="update_state", value="candidate_built"),
            NonsecretField(name="candidate_commit", value=value.candidate_commit),
            NonsecretField(name="current_commit", value=value.base_current_commit),
            NonsecretField(name="running_commit", value=value.base_running_commit),
            NonsecretField(name="release_path", value=value.release_path),
            NonsecretField(name="receipt_path", value=value.receipt_path),
        ),
        failure_fields=lambda: _failure_status_fields(machine, target),
    )
    preflight = _run_step(
        emitter,
        steps[7],
        running=(
            "Capturing one consistent live boundary and starting the candidate behind the "
            "closed startup-effect fence."
        ),
        operation=lambda: machine.rehearse_candidate(target, built),
        succeeded=(
            "The candidate migrated, planned recovery, replayed copied projects, and answered "
            "representative reads without crossing the effect fence."
        ),
        fields=lambda value: (
            NonsecretField(name="update_state", value="candidate_verified"),
            NonsecretField(name="candidate_commit", value=value.candidate_commit),
            NonsecretField(name="current_commit", value=value.base_current_commit),
            NonsecretField(name="running_commit", value=value.base_running_commit),
            NonsecretField(
                name="verified_projects",
                value=sum(project.status == "verified" for project in value.projects),
            ),
            NonsecretField(
                name="unavailable_projects",
                value=sum(project.status == "not_replay_verified" for project in value.projects),
            ),
            NonsecretField(name="receipt_path", value=value.receipt_path),
        ),
        failure_fields=lambda: _failure_status_fields(machine, target),
    )
    _run_cutover_steps(
        emitter,
        steps[8:],
        machine,
        target,
        built,
        preflight,
    )


def _run_cutover_steps(
    emitter: ServerEventEmitter,
    steps: tuple[ServerStep, ...],
    machine: UpdateMachine,
    target: UpdateTarget,
    built: BuiltCandidateReceipt,
    preflight: VerifiedCandidateReceipt,
) -> None:
    phases = (
        "maintenance_closed",
        "checkpoint_ready",
        "candidate_started",
        "candidate_verified",
    )
    if len(steps) != len(phases) + 1:
        raise AssertionError("cutover execution requires five exact planned steps")
    index = 0
    emitter.emit_step(
        steps[index].model_copy(
            update={
                "state": "running",
                "message": "Closing normal work and waiting for active provider turns to settle.",
            }
        )
    )

    def progress(phase: str) -> None:
        nonlocal index
        if phase == "rolled_back":
            return
        if index >= len(phases) or phase != phases[index]:
            raise UpdateRefused("The update coordinator reported an out-of-order durable phase.")
        emitter.emit_step(
            steps[index].model_copy(
                update={
                    "state": "succeeded",
                    "message": _cutover_phase_success(phase),
                }
            )
        )
        index += 1
        if index < len(steps):
            emitter.emit_step(
                steps[index].model_copy(
                    update={
                        "state": "running",
                        "message": _cutover_phase_running(
                            phases[index] if index < len(phases) else "committed"
                        ),
                    }
                )
            )

    try:
        outcome = machine.cutover_candidate(
            target,
            built,
            preflight,
            progress=progress,
        )
    except UpdateRefused as exc:
        emitter.emit_step(
            steps[index].model_copy(
                update={
                    "state": "failed",
                    "message": str(exc),
                    "fields": _failure_status_fields(machine, target),
                }
            )
        )
        raise _ReportedUpdateFailure from exc
    if outcome.operation_state == "rolled_back":
        emitter.emit_step(
            steps[index].model_copy(
                update={
                    "state": "failed",
                    "message": (
                        "Candidate verification failed; RCP restored and verified the previous "
                        "release before reopening work."
                    ),
                    "fields": (
                        NonsecretField(name="update_state", value="rolled_back"),
                        NonsecretField(
                            name="failed_candidate_commit", value=outcome.candidate_commit
                        ),
                        NonsecretField(name="restored_commit", value=outcome.running_commit),
                        NonsecretField(name="operation_id", value=outcome.operation_id),
                        NonsecretField(name="receipt_path", value=str(outcome.receipt_path)),
                        NonsecretField(
                            name="failure", value=outcome.failure or "verification failed"
                        ),
                    ),
                }
            )
        )
        raise _ReportedUpdateFailure
    if index != len(phases):
        raise AssertionError("committed cutover did not report every pre-commit phase")
    emitter.emit_step(
        steps[-1].model_copy(
            update={
                "state": "succeeded",
                "message": _cutover_phase_success("committed"),
                "fields": (
                    NonsecretField(name="update_state", value="committed"),
                    NonsecretField(name="candidate_commit", value=outcome.candidate_commit),
                    NonsecretField(name="running_commit", value=outcome.running_commit),
                    NonsecretField(name="operation_id", value=outcome.operation_id),
                    NonsecretField(name="receipt_path", value=str(outcome.receipt_path)),
                ),
            }
        )
    )


def _cutover_phase_running(phase: str) -> str:
    return {
        "maintenance_closed": "Closing normal work and capturing the final live boundary.",
        "checkpoint_ready": "Fresh-rehearsing and verifying the exact rollback checkpoint.",
        "candidate_started": "Stopping systemd and switching to the fenced candidate.",
        "candidate_verified": "Comparing the real candidate with the final rehearsal read model.",
        "committed": "Committing the verified release and starting deferred runtime owners.",
    }[phase]


def _cutover_phase_success(phase: str) -> str:
    return {
        "maintenance_closed": "Normal work is closed at one exact durable capture boundary.",
        "checkpoint_ready": "The final rehearsal and complete rollback checkpoint are verified.",
        "candidate_started": "systemd runs the exact candidate behind the startup-effect fence.",
        "candidate_verified": "The real candidate matches the final rehearsal read model.",
        "committed": "The candidate is committed and deferred runtime startup was released once.",
    }[phase]


_T = TypeVar("_T")


def _run_step(
    emitter: ServerEventEmitter,
    planned: ServerStep,
    *,
    running: str,
    operation: Callable[[], _T],
    succeeded: str,
    fields: Callable[[_T], tuple[NonsecretField, ...]] = lambda _value: (),
    failure_fields: Callable[[], tuple[NonsecretField, ...]] = lambda: (),
) -> _T:
    emitter.emit_step(planned.model_copy(update={"state": "running", "message": running}))
    try:
        value = operation()
    except UpdateRefused as exc:
        emitter.emit_step(
            planned.model_copy(
                update={
                    "state": "failed",
                    "message": str(exc),
                    "fields": failure_fields(),
                }
            )
        )
        raise _ReportedUpdateFailure from exc
    emitter.emit_step(
        planned.model_copy(
            update={
                "state": "succeeded",
                "message": succeeded,
                "fields": fields(value),
            }
        )
    )
    return value


def _emit_confirmation_pause(
    emitter: ServerEventEmitter,
    planned: ServerStep,
    target: UpdateTarget,
    *,
    resume_executable: Path,
) -> None:
    resume = (
        "sudo",
        str(resume_executable),
        "server",
        "update",
        "--confirm-target",
        target.target_commit,
    )
    emitter.emit_step(
        planned.model_copy(
            update={
                "state": "operator_action_needed",
                "message": (
                    "Review the current and fetched target commits, then run the exact command "
                    "shown. RCP will refetch and refuse if main changed before confirmation."
                ),
                "actions": (
                    ExternalAction(
                        instruction=(
                            "Confirm that the shown target commit is the GitHub main revision this "
                            "server should build. No current release or process has changed."
                        )
                    ),
                    CommandAction(argv=resume),
                ),
                "fields": _target_fields(target),
                "resume_argv": resume,
            }
        )
    )


def _complete_already_current(
    emitter: ServerEventEmitter,
    remaining: tuple[ServerStep, ...],
    target: UpdateTarget,
    *,
    confirmation_completed: bool = False,
) -> None:
    messages = (
        "No confirmation is needed because fetched origin/main is already running.",
        "Managed main already names the running commit; no fast-forward was performed.",
        "The current immutable release already names this commit; no candidate was created.",
        "The current source build was left untouched; no candidate build was run.",
        "No candidate receipt is needed because the server already runs the fetched commit.",
        "No candidate rehearsal is needed because the fetched commit is already serving.",
        "No maintenance boundary is needed because the fetched commit is already serving.",
        "No rollback checkpoint is needed because no release switch is pending.",
        "No release switch is needed because current already names the running commit.",
        "No switched-candidate verification is needed for the already-running commit.",
        "No fence release is needed because ordinary runtime is already open.",
    )
    if confirmation_completed:
        messages = messages[1:]
    for index, (planned, message) in enumerate(zip(remaining, messages, strict=True)):
        fields = (
            (
                NonsecretField(name="update_state", value="already_current"),
                NonsecretField(name="current_commit", value=target.inspection.current_commit),
                NonsecretField(name="running_commit", value=target.inspection.running_commit),
                NonsecretField(name="candidate_commit", value="none"),
            )
            if index == len(remaining) - 1
            else _target_fields(target)
        )
        emitter.emit_step(planned.model_copy(update={"state": "running", "message": message}))
        emitter.emit_step(
            planned.model_copy(update={"state": "succeeded", "message": message, "fields": fields})
        )


def _require_confirmed_target(confirmed: str | None, target: UpdateTarget) -> None:
    if confirmed != target.target_commit:
        raise UpdateRefused(
            "The confirmed commit no longer matches fetched origin/main. Review the new target "
            "and rerun without --confirm-target."
        )


def _inspection_fields(inspection: UpdateInspection) -> tuple[NonsecretField, ...]:
    return (
        NonsecretField(name="managed_main_head", value=inspection.managed_head),
        NonsecretField(name="current_commit", value=inspection.current_commit),
        NonsecretField(name="running_commit", value=inspection.running_commit),
    )


def _target_fields(target: UpdateTarget) -> tuple[NonsecretField, ...]:
    return (
        *_inspection_fields(target.inspection),
        NonsecretField(name="target_commit", value=target.target_commit),
    )


def _fetch_failure_fields(inspection: UpdateInspection) -> tuple[NonsecretField, ...]:
    return (
        *_inspection_fields(inspection),
        NonsecretField(name="candidate_commit", value="unavailable"),
    )


def _status_fields(target: UpdateTarget, *, managed: str) -> tuple[NonsecretField, ...]:
    return (
        NonsecretField(name="managed_main_head", value=managed),
        NonsecretField(name="candidate_commit", value=target.target_commit),
        NonsecretField(name="current_commit", value=target.inspection.current_commit),
        NonsecretField(name="running_commit", value=target.inspection.running_commit),
    )


def _failure_status_fields(
    machine: UpdateMachine,
    target: UpdateTarget,
) -> tuple[NonsecretField, ...]:
    try:
        observed = machine.status()
    except UpdateRefused:
        return _status_fields(target, managed="unavailable")
    return (
        NonsecretField(name="managed_main_head", value=observed.managed_head),
        NonsecretField(name="candidate_commit", value=target.target_commit),
        NonsecretField(name="current_commit", value=observed.current_commit),
        NonsecretField(name="running_commit", value=observed.running_commit),
    )


@contextmanager
def server_update_operation_lock(
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
    *,
    root_uid: int,
    root_gid: int,
    service_gid: int,
) -> Iterator[None]:
    """Serialize root restore/update ownership on the existing stable inode."""

    lock_path = layout.config_path.parent / _UPDATE_LOCK_NAME
    descriptor = -1
    try:
        _require_owned_directory(
            lock_path.parent,
            uid=root_uid,
            gid=service_gid,
            mode=_CONFIG_DIRECTORY_MODE,
            label="server configuration directory",
        )
        flags = os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, _LOCK_MODE)
            os.fchown(descriptor, root_uid, root_gid)
            os.fchmod(descriptor, _LOCK_MODE)
            os.fsync(descriptor)
            _fsync_directory(lock_path.parent)
        except FileExistsError:
            descriptor = os.open(lock_path, flags)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_uid, info.st_gid) != (root_uid, root_gid)
            or stat.S_IMODE(info.st_mode) != _LOCK_MODE
        ):
            raise UpdateRefused(
                "The update lock has unexpected type, ownership, or mode. Inspect /etc/rcp "
                "and rerun the same command."
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UpdateRefused(
                "Another server update is running. Wait for it to finish, then rerun the same "
                "command."
            ) from exc
        yield
    except UpdateRefused:
        raise
    except OSError as exc:
        raise UpdateRefused(
            "RCP could not acquire the root-owned update lock or inspect maintenance state. "
            "Inspect the fixed server paths and rerun."
        ) from exc
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(descriptor)


class LinuxUpdateMachine:
    """Root coordinator with every Git/build subprocess fixed to the rcp account."""

    def __init__(
        self,
        layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
        *,
        config_loader: Callable[[Path], InstalledServerConfig] | None = None,
        doctor: ServerDoctorMachine | None = None,
        service_runner: ServiceCommandRunner | None = None,
        service_identity: tuple[int, int] | None = None,
        root_identity: tuple[int, int] | None = None,
        system_service: InstalledSystemServiceController | None = None,
        cutover_control_factory: Callable[[str], object] | None = None,
    ) -> None:
        self.layout = layout
        self._config_loader = config_loader or load_installed_server_config
        try:
            if service_identity is None:
                service = pwd.getpwnam(layout.service_account)
                service_identity = (service.pw_uid, service.pw_gid)
            if root_identity is None:
                root = pwd.getpwnam("root")
                root_identity = (root.pw_uid, root.pw_gid)
        except KeyError as exc:
            raise UpdateRefused(
                "The installed root or rcp operating-system account is missing."
            ) from exc
        self._service_uid, self._service_gid = service_identity
        self._root_uid, self._root_gid = root_identity
        self._system_service = system_service or InstalledSystemServiceController(
            layout,
            root_identity=(self._root_uid, self._root_gid),
        )
        self._cutover_control_factory = cutover_control_factory
        self._service_runner = service_runner or self._run_as_installed_service
        self._doctor = doctor or LinuxServerDoctorMachine(
            layout,
            config_loader=self._config_loader,
            runner=self._doctor_runner,
            service_identity=(self._service_uid, self._service_gid),
            root_identity=(self._root_uid, self._root_gid),
        )

    @contextmanager
    def admission(self) -> Iterator[None]:
        from rcp.server_ops.backup import BackupRunRefused, backup_run_coordination_lock

        try:
            with (
                server_update_operation_lock(
                    self.layout,
                    root_uid=self._root_uid,
                    root_gid=self._root_gid,
                    service_gid=self._service_gid,
                ),
                backup_run_coordination_lock(
                    self.layout,
                    expected_uid=self._service_uid,
                    expected_gid=self._service_gid,
                    timeout=SERVER_CONTROL_UPDATE_MAINTENANCE_TIMEOUT_SECONDS,
                ),
            ):
                self._recover_unfinished_update()
                self._inspect_maintenance_roots()
                yield
        except BackupRunRefused as exc:
            raise UpdateRefused(
                "A protected backup did not reach its durable boundary before update "
                "maintenance timed out."
            ) from exc
        except UpdateRefused:
            raise
        except OSError as exc:
            raise UpdateRefused(
                "RCP could not acquire the root-owned update lock or inspect maintenance state. "
                "Inspect the fixed server paths and rerun."
            ) from exc

    def inspect(self) -> UpdateInspection:
        inspection, report = self._read_status()
        if report.problems:
            raise UpdateRefused(
                f"Server doctor blocks update: {report.problems[0]}. Repair it and rerun."
            )
        if inspection.current_commit != inspection.running_commit:
            raise UpdateRefused(
                "The current pointer and running process differ. Complete or repair the pending "
                "restart before preparing another update."
            )
        if report.release_state not in {"aligned", "candidate_pending"}:
            raise UpdateRefused(
                "The installed release relationship is not safe for candidate preparation."
            )
        if (
            report.installation_id != inspection.config.installation_id
            or report.configured_origin != inspection.config.source.origin
            or report.configured_branch != inspection.config.source.branch
        ):
            raise UpdateRefused("Doctor and installed configuration disagree on source identity.")
        return inspection

    def status(self) -> UpdateInspection:
        inspection, _report = self._read_status()
        return inspection

    def _read_status(self) -> tuple[UpdateInspection, ServerDoctorReport]:
        try:
            config = self._config_loader(self.layout.config_path)
            if config.paths.model_dump() != self.layout.recorded_paths():
                raise ValueError("installed paths differ")
            report = self._doctor.inspect()
        except (OSError, ServerMetadataError, ValueError) as exc:
            raise UpdateRefused(
                "The installed server configuration or doctor readback is invalid. Run "
                "rcp server doctor as rcp and repair its reported problem before updating."
            ) from exc
        values = (
            report.managed_main_head,
            report.current_commit,
            report.running_commit,
            report.instance_id,
            report.process_pid,
        )
        if any(value is None for value in values):
            raise UpdateRefused("Server doctor could not prove every source and process identity.")
        assert report.managed_main_head is not None
        assert report.current_commit is not None
        assert report.running_commit is not None
        assert report.instance_id is not None
        assert report.process_pid is not None
        return (
            UpdateInspection(
                config=config,
                managed_head=report.managed_main_head,
                current_commit=report.current_commit,
                running_commit=report.running_commit,
                instance_id=report.instance_id,
                process_pid=report.process_pid,
            ),
            report,
        )

    def fetch_target(self, inspection: UpdateInspection) -> UpdateTarget:
        source = self.layout.source_checkout
        environment = self._update_git_environment(inspection.config)
        self._validate_source_checkout(
            inspection.config,
            expected_head=inspection.managed_head,
            environment=environment,
        )
        self._run_git(
            source,
            ("fetch", "--prune", "origin", "main"),
            environment=environment,
            timeout=SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
            error="Fetching origin/main failed with the configured source identity. Restore source access and rerun.",
        )
        target = self._git_text(
            source,
            ("rev-parse", "--verify", "refs/remotes/origin/main^{commit}"),
            environment=environment,
        )
        _require_commit(target)
        if inspection.managed_head != target:
            forward = self._git_result(
                source,
                ("merge-base", "--is-ancestor", inspection.managed_head, target),
                environment=environment,
            )
            if forward.returncode != 0:
                ahead = self._git_result(
                    source,
                    ("merge-base", "--is-ancestor", target, inspection.managed_head),
                    environment=environment,
                )
                relationship = "ahead of" if ahead.returncode == 0 else "diverged from"
                raise UpdateRefused(
                    f"Managed main is {relationship} fetched origin/main. RCP will not reset or force-pull it."
                )
        return UpdateTarget(inspection=inspection, target_commit=target)

    def fast_forward(self, target: UpdateTarget) -> None:
        source = self.layout.source_checkout
        environment = self._update_git_environment(target.inspection.config)
        head = self._validate_source_checkout(
            target.inspection.config,
            expected_head=None,
            environment=environment,
        )
        remote = self._git_text(
            source,
            ("rev-parse", "--verify", "refs/remotes/origin/main^{commit}"),
            environment=environment,
        )
        if remote != target.target_commit:
            raise UpdateRefused(
                "The fetched target changed after confirmation. RCP left managed main unchanged; "
                "rerun without --confirm-target."
            )
        if head == target.target_commit:
            return
        if head != target.inspection.managed_head:
            raise UpdateRefused(
                "Managed main changed after update inspection; RCP will not overwrite it."
            )
        self._run_git(
            source,
            ("merge", "--ff-only", "--no-edit", target.target_commit),
            environment=environment,
            timeout=SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
            error="The confirmed fast-forward failed. Inspect the clean managed checkout; RCP did not reset it.",
        )
        observed = self._validate_source_checkout(
            target.inspection.config,
            expected_head=target.target_commit,
            environment=environment,
        )
        if observed != target.target_commit:  # pragma: no cover - validator already compares
            raise UpdateRefused("Managed main did not reach the confirmed target.")

    def prepare_release(self, target: UpdateTarget) -> Path:
        _require_owned_directory(
            self.layout.releases_root,
            uid=self._service_uid,
            gid=self._service_gid,
            mode=_DIRECTORY_MODE,
            label="releases root",
        )
        release = self.layout.release_dir(target.target_commit)
        if release.exists() or release.is_symlink():
            _require_owned_directory(
                release,
                uid=self._service_uid,
                gid=self._service_gid,
                mode=None,
                label="candidate release",
            )
            self._validate_release_git(release, target.target_commit)
            return release
        self._run_service_checked(
            (
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(self.layout.source_checkout),
                "worktree",
                "add",
                "--detach",
                str(release),
                target.target_commit,
            ),
            timeout=SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
            capture_output=False,
            error="Creating the clean per-commit candidate worktree failed. Inspect Git and rerun.",
        )
        _require_owned_directory(
            release,
            uid=self._service_uid,
            gid=self._service_gid,
            mode=None,
            label="candidate release",
        )
        self._validate_release_git(release, target.target_commit)
        return release

    def build_candidate(self, target: UpdateTarget, release: Path) -> CandidateBuild:
        if release != self.layout.release_dir(target.target_commit):
            raise UpdateRefused("The candidate release path does not match the confirmed commit.")
        existing = self._read_receipt_if_present(target.target_commit)
        if existing is not None:
            self._validate_receipt_for_target(existing, target)
            web_identity = self._validate_built_release(release, target.target_commit)
            if web_identity != existing.web_build_id:
                raise UpdateRefused("The built candidate differs from its immutable receipt.")
            return CandidateBuild(
                commit=target.target_commit,
                release_path=release,
                web_build_id=web_identity,
                reused_receipt=True,
            )
        self._validate_release_git(release, target.target_commit)
        for argv, environment, error in (
            (
                ("npm", "--prefix", "web", "ci"),
                None,
                "npm --prefix web ci failed in the candidate. The current service is unchanged.",
            ),
            (
                ("npm", "--prefix", "web", "run", "build"),
                None,
                "The production Web build failed in the candidate. The current service is unchanged.",
            ),
            (
                ("uv", "sync", "--frozen"),
                {"UV_MANAGED_PYTHON": "1", "UV_PYTHON": "3.12"},
                "uv sync --frozen failed in the candidate. The current service is unchanged.",
            ),
        ):
            self._run_service_checked(
                argv,
                cwd=release,
                environment=environment,
                timeout=SERVER_INSTALL_BUILD_TIMEOUT_SECONDS,
                capture_output=False,
                error=error,
            )
        web_identity = self._validate_built_release(release, target.target_commit)
        return CandidateBuild(
            commit=target.target_commit,
            release_path=release,
            web_build_id=web_identity,
            reused_receipt=False,
        )

    def finalize_candidate(
        self,
        target: UpdateTarget,
        build: CandidateBuild,
    ) -> BuiltCandidateReceipt:
        if build.commit != target.target_commit or build.release_path != self.layout.release_dir(
            target.target_commit
        ):
            raise UpdateRefused("Candidate build identity changed before receipt publication.")
        web_identity = self._validate_built_release(build.release_path, build.commit)
        if web_identity != build.web_build_id:
            raise UpdateRefused("Candidate Web bytes changed before receipt publication.")
        readback = self.inspect()
        if (
            readback.managed_head != target.target_commit
            or readback.current_commit != target.inspection.current_commit
            or readback.running_commit != target.inspection.running_commit
            or readback.instance_id != target.inspection.instance_id
            or readback.process_pid != target.inspection.process_pid
            or readback.config.installation_id != target.inspection.config.installation_id
            or readback.config.source != target.inspection.config.source
        ):
            raise UpdateRefused(
                "The managed target or live service changed during candidate preparation. "
                "The current pointer was not switched; inspect doctor before continuing."
            )
        receipt = BuiltCandidateReceipt(
            installation_id=target.inspection.config.installation_id,
            source_origin=target.inspection.config.source.origin,
            base_current_commit=target.inspection.current_commit,
            base_running_commit=target.inspection.running_commit,
            base_instance_id=target.inspection.instance_id,
            base_process_pid=target.inspection.process_pid,
            candidate_commit=target.target_commit,
            release_path=str(build.release_path),
            receipt_path=str(built_candidate_receipt_path(target.target_commit, self.layout)),
            web_build_id=build.web_build_id,
            prepared_at=datetime.now(UTC),
        )
        published = self._publish_receipt(receipt)
        self._validate_receipt_for_target(published, target)
        if published.web_build_id != build.web_build_id:
            raise UpdateRefused(
                "An existing built-candidate receipt names different Web bytes. Preserve it for "
                "diagnosis; RCP will not overwrite it."
            )
        return published

    def rehearse_candidate(
        self,
        target: UpdateTarget,
        built: BuiltCandidateReceipt,
    ) -> VerifiedCandidateReceipt:
        from rcp.server_ops.rehearsal import (
            CandidateRehearsalRefused,
            read_verified_candidate_receipt,
            verified_candidate_receipt_path,
        )

        self._validate_receipt_for_target(built, target)
        built_path = Path(built.receipt_path)
        built_sha256 = _owned_file_sha256(
            built_path,
            uid=self._service_uid,
            gid=self._service_gid,
            mode=_RECEIPT_MODE,
            maximum=_MAX_RECEIPT_BYTES,
            label="built-candidate receipt",
        )
        current_python = (
            self.layout.release_dir(built.base_running_commit) / ".venv" / "bin" / "python"
        )
        completed = self._run_service_checked(
            (
                str(current_python),
                "-m",
                "rcp.server_ops.rehearsal",
                "--orchestrate",
                str(built_path),
                str(self.layout.data_dir),
                str(self.layout.update_checkpoints_root),
            ),
            cwd=self.layout.release_dir(built.base_running_commit),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout=SERVER_UPDATE_REHEARSAL_TIMEOUT_SECONDS,
            error=(
                "Candidate copied-state rehearsal failed. The current release is unchanged; "
                "inspect the retained rehearsal and backup capture before retrying."
            ),
        )
        receipt_lines = completed.stdout.splitlines()
        if len(receipt_lines) != 1:
            raise UpdateRefused(
                "The current release did not report one exact verified-candidate receipt."
            )
        receipt_path = Path(receipt_lines[0])
        if receipt_path.parent != self.layout.update_checkpoints_root:
            raise UpdateRefused(
                "The current release reported a verified-candidate receipt outside its checkpoint root."
            )
        try:
            receipt = read_verified_candidate_receipt(
                receipt_path,
                expected_uid=self._service_uid,
            )
        except CandidateRehearsalRefused as exc:
            raise UpdateRefused(
                "The candidate rehearsal did not publish one valid verified-candidate receipt."
            ) from exc
        if (
            receipt.installation_id != built.installation_id
            or receipt.candidate_commit != built.candidate_commit
            or receipt.base_current_commit != built.base_current_commit
            or receipt.base_running_commit != built.base_running_commit
            or receipt.base_instance_id != built.base_instance_id
            or receipt.base_process_pid != built.base_process_pid
            or receipt.release_path != built.release_path
            or receipt.built_receipt_path != built.receipt_path
            or receipt.built_receipt_sha256 != built_sha256
            or receipt.web_build_id != built.web_build_id
            or receipt_path
            != verified_candidate_receipt_path(
                target.target_commit,
                receipt.capture_id,
                self.layout.update_checkpoints_root,
            )
        ):
            raise UpdateRefused(
                "The verified-candidate receipt differs from its exact build and live base."
            )
        readback = self.inspect()
        if (
            readback.managed_head != target.target_commit
            or readback.current_commit != target.inspection.current_commit
            or readback.running_commit != target.inspection.running_commit
            or readback.instance_id != target.inspection.instance_id
            or readback.process_pid != target.inspection.process_pid
        ):
            raise UpdateRefused(
                "The live service changed during candidate rehearsal. The release pointer was "
                "not switched; inspect server doctor before continuing."
            )
        return receipt

    def create_rollback_checkpoint(
        self,
        target: UpdateTarget,
        verified: VerifiedCandidateReceipt,
        *,
        sqlite_receipt_path: Path,
        sqlite_receipt_sha256: str,
        project_receipt_path: Path,
        project_receipt_sha256: str,
    ) -> VerifiedUpdateCheckpoint:
        """Run F6c as rcp after F6d has closed admission at this exact capture."""

        from rcp.server_ops.update_checkpoint import (
            UpdateCheckpointRefused,
            read_verified_update_checkpoint,
        )

        if (
            verified.candidate_commit != target.target_commit
            or verified.base_running_commit != target.inspection.running_commit
            or verified.base_current_commit != target.inspection.current_commit
            or verified.project_capture_sha256 != project_receipt_sha256
            or Path(verified.receipt_path).parent != self.layout.update_checkpoints_root
        ):
            raise UpdateRefused(
                "The final rollback checkpoint request differs from its verified candidate."
            )
        candidate_receipt_sha256 = _owned_file_sha256(
            Path(verified.receipt_path),
            uid=self._service_uid,
            gid=self._service_gid,
            mode=_RECEIPT_MODE,
            maximum=_MAX_VERIFIED_RECEIPT_BYTES,
            label="verified-candidate receipt",
        )
        current_python = (
            self.layout.release_dir(verified.base_running_commit) / ".venv" / "bin" / "python"
        )
        completed = self._run_service_checked(
            (
                str(current_python),
                "-m",
                "rcp.server_ops.update_checkpoint",
                "create",
                str(self.layout.data_dir),
                str(self.layout.update_checkpoints_root),
                str(self.layout.release_dir(verified.base_running_commit)),
                str(sqlite_receipt_path),
                sqlite_receipt_sha256,
                str(project_receipt_path),
                project_receipt_sha256,
                verified.receipt_path,
                candidate_receipt_sha256,
            ),
            cwd=self.layout.release_dir(verified.base_running_commit),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout=SERVER_UPDATE_CHECKPOINT_TIMEOUT_SECONDS,
            error=(
                "The final rollback checkpoint failed. The current release is unchanged; "
                "inspect the retained checkpoint operation before retrying."
            ),
        )
        lines = completed.stdout.splitlines()
        if len(lines) != 1:
            raise UpdateRefused(
                "The current release did not report one exact rollback checkpoint manifest."
            )
        manifest_path = Path(lines[0])
        if manifest_path.parent.parent != self.layout.update_checkpoints_root:
            raise UpdateRefused(
                "The current release reported a rollback checkpoint outside its private root."
            )
        try:
            checkpoint = read_verified_update_checkpoint(
                manifest_path,
                expected_uid=self._service_uid,
            )
        except UpdateCheckpointRefused as exc:
            raise UpdateRefused(
                "The current release did not publish one valid rollback checkpoint."
            ) from exc
        if (
            checkpoint.installation_id != verified.installation_id
            or checkpoint.space_id != verified.space_id
            or checkpoint.capture_id != verified.capture_id
            or checkpoint.base_commit != verified.base_running_commit
            or checkpoint.candidate_commit != verified.candidate_commit
            or checkpoint.candidate_receipt_sha256 != candidate_receipt_sha256
        ):
            raise UpdateRefused(
                "The rollback checkpoint differs from its exact candidate and capture."
            )
        return checkpoint

    def cutover_candidate(
        self,
        target: UpdateTarget,
        built: BuiltCandidateReceipt,
        preflight: VerifiedCandidateReceipt,
        *,
        progress: Callable[[str], None],
    ) -> UpdateCutoverOutcome:
        from rcp.server_ops.update_cutover import (
            UpdateCutoverCoordinator,
            UpdateCutoverRefused,
        )

        try:
            return UpdateCutoverCoordinator(
                layout=self.layout,
                actions=_LinuxCutoverActions(self, target, built),
                expected_uid=self._service_uid,
                expected_gid=self._service_gid,
                progress=progress,
            ).run(built, preflight)
        except (InstalledServiceControlRefused, ServerControlError, UpdateCutoverRefused) as exc:
            raise UpdateRefused(
                str(exc) or "The server update failed at its safe boundary."
            ) from exc

    def _enter_update_maintenance(self, *, operation_id: str, receipt_sha256: str):
        return self._control_for_running(self._current_pointer_commit()).enter_update_maintenance(
            operation_id=operation_id,
            receipt_sha256=receipt_sha256,
        )

    def _final_maintenance_rehearsal(
        self,
        target: UpdateTarget,
        built: BuiltCandidateReceipt,
        operation: UpdateOperationReceipt,
        receipt_sha256: str,
    ) -> VerifiedCandidateReceipt:
        from rcp.server_ops.rehearsal import (
            CandidateRehearsalRefused,
            read_verified_candidate_receipt,
            verified_candidate_receipt_path,
        )

        current_python = (
            self.layout.release_dir(built.base_running_commit) / ".venv" / "bin" / "python"
        )
        completed = self._run_service_checked(
            (
                str(current_python),
                "-m",
                "rcp.server_ops.rehearsal",
                "--orchestrate-maintenance",
                built.receipt_path,
                str(self.layout.data_dir),
                str(self.layout.update_checkpoints_root),
                operation.receipt_path,
                receipt_sha256,
            ),
            cwd=self.layout.release_dir(built.base_running_commit),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout=SERVER_UPDATE_REHEARSAL_TIMEOUT_SECONDS,
            error=(
                "Final closed-admission rehearsal failed. The old release remains selected; "
                "RCP will reopen work only after the maintenance receipt is resolved."
            ),
        )
        lines = completed.stdout.splitlines()
        if len(lines) != 1:
            raise UpdateRefused(
                "The current release did not report one final verified-candidate receipt."
            )
        receipt_path = Path(lines[0])
        try:
            receipt = read_verified_candidate_receipt(
                receipt_path,
                expected_uid=self._service_uid,
            )
        except CandidateRehearsalRefused as exc:
            raise UpdateRefused(
                "The final maintenance rehearsal did not publish a valid receipt."
            ) from exc
        if (
            operation.capture is None
            or receipt_path
            != verified_candidate_receipt_path(
                target.target_commit,
                operation.capture.capture_id,
                self.layout.update_checkpoints_root,
            )
            or receipt.capture_id != operation.capture.capture_id
            or receipt.built_receipt_path != built.receipt_path
            or receipt.base_instance_id != built.base_instance_id
            or receipt.base_process_pid != built.base_process_pid
            or receipt.candidate_commit != built.candidate_commit
        ):
            raise UpdateRefused(
                "The final verified-candidate receipt differs from its maintenance capture."
            )
        return receipt

    def _control_for_running(self, commit: str):
        if self._cutover_control_factory is not None:
            return self._cutover_control_factory(commit)
        deadline = time.monotonic() + SERVER_CONTROL_UPDATE_VERIFY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                client = ServerControlClient.from_data_dir(
                    self.layout.data_dir,
                    expected_server_uid=self._service_uid,
                )
                probe = client.probe()
                if (
                    client.metadata.running_commit == commit
                    and probe.data_dir_id == client.metadata.data_dir_id
                    and probe.pid == client.metadata.pid
                ):
                    return client
            except (OSError, ServerControlError, ValueError):
                pass
            time.sleep(0.1)
        raise UpdateRefused(
            f"The restarted service did not publish the expected running commit {commit}."
        )

    def _restore_cutover_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        base_commit: str,
    ) -> None:
        current_python = self.layout.release_dir(base_commit) / ".venv" / "bin" / "python"
        completed = self._run_service_checked(
            (
                str(current_python),
                "-m",
                "rcp.server_ops.update_checkpoint",
                "restore",
                str(checkpoint_path),
                checkpoint_sha256,
            ),
            cwd=self.layout.release_dir(base_commit),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout=SERVER_UPDATE_CHECKPOINT_TIMEOUT_SECONDS,
            error=(
                "The rollback checkpoint could not be restored. RCP kept the service stopped "
                "and retained the exact journal for re-entry."
            ),
        )
        if completed.stdout.splitlines() != ["complete"]:
            raise UpdateRefused(
                "The rollback worker did not report one complete replacement journal."
            )

    def _current_pointer_commit(self) -> str:
        return self._system_service.current_release().name

    def _inspect_maintenance_roots(self) -> None:
        from rcp.server_ops.update_checkpoint import (
            UpdateCheckpointRefused,
            read_rollback_journal,
            read_verified_update_checkpoint,
        )
        from rcp.server_ops.update_cutover import (
            UpdateCutoverRefused,
            read_update_operation,
        )

        _require_owned_directory(
            self.layout.restore_operations_root,
            uid=self._service_uid,
            gid=self._service_gid,
            mode=_DIRECTORY_MODE,
            label="restore operations root",
        )
        _require_owned_directory(
            self.layout.update_checkpoints_root,
            uid=self._service_uid,
            gid=self._service_gid,
            mode=_DIRECTORY_MODE,
            label="update checkpoints root",
        )
        try:
            from rcp.server_ops.restore import RestoreRefused, unfinished_restore_operation

            try:
                restore = unfinished_restore_operation(
                    self.layout,
                    expected_uid=self._service_uid,
                )
            except (OSError, RestoreRefused) as exc:
                with suppress(InstalledServiceControlRefused):
                    self._system_service.fence_stopped_disabled()
                raise UpdateRefused(
                    "The unfinished restore recovery state is unsafe. RCP kept the service "
                    "stopped; preserve it and resume server restore before source update."
                ) from exc
            if restore is not None:
                with suppress(InstalledServiceControlRefused):
                    self._system_service.fence_stopped_disabled()
                raise UpdateRefused(
                    "An unfinished restore operation blocks source update. RCP kept the service "
                    "stopped; resume server restore first."
                )
            for entry in self.layout.update_checkpoints_root.iterdir():
                if _RECEIPT_NAME.fullmatch(entry.name) is not None:
                    self._read_receipt(entry)
                    continue
                if _VERIFIED_RECEIPT_NAME.fullmatch(entry.name) is not None:
                    from rcp.server_ops.rehearsal import (
                        CandidateRehearsalRefused,
                        read_verified_candidate_receipt,
                    )

                    try:
                        read_verified_candidate_receipt(entry, expected_uid=self._service_uid)
                    except CandidateRehearsalRefused as exc:
                        raise UpdateRefused(
                            "A verified-candidate receipt is unsafe or invalid. Preserve and "
                            "inspect it before retrying."
                        ) from exc
                    continue
                if _UPDATE_OPERATION_NAME.fullmatch(entry.name) is not None:
                    try:
                        read_update_operation(entry, expected_uid=self._service_uid)
                    except UpdateCutoverRefused as exc:
                        raise UpdateRefused(
                            "An update operation receipt is unsafe or invalid. Preserve and "
                            "inspect it before retrying."
                        ) from exc
                    continue
                if _CHECKPOINT_ROOT_NAME.fullmatch(entry.name) is not None:
                    manifest = entry / "checkpoint.json"
                    try:
                        read_verified_update_checkpoint(
                            manifest,
                            expected_uid=self._service_uid,
                        )
                        journal = entry / "rollback-journal.json"
                        if (
                            journal.exists()
                            and read_rollback_journal(
                                journal,
                                expected_uid=self._service_uid,
                            ).phase
                            != "complete"
                        ):
                            raise UpdateRefused(
                                "An unfinished rollback journal still blocks a new source update."
                            )
                    except UpdateRefused:
                        raise
                    except (OSError, UpdateCheckpointRefused) as exc:
                        raise UpdateRefused(
                            "A retained update checkpoint is incomplete or unsafe. Preserve and "
                            "inspect it before retrying."
                        ) from exc
                    continue
                if _REHEARSAL_ROOT_NAME.fullmatch(entry.name) is not None:
                    raise UpdateRefused(
                        "A retained failed candidate rehearsal blocks another update. Inspect "
                        "and explicitly clean that exact rehearsal before retrying."
                    )
                raise UpdateRefused(
                    "Unfinished update maintenance blocks a new source preparation. Resume or "
                    "repair that exact operation first."
                )
        except OSError as exc:
            raise UpdateRefused(
                "RCP could not inspect update and restore maintenance state."
            ) from exc

    def _recover_unfinished_update(self) -> None:
        from rcp.server_ops.update_checkpoint import (
            UpdateCheckpointRefused,
            unfinished_rollback_journals,
        )
        from rcp.server_ops.update_cutover import (
            UpdateCutoverCoordinator,
            UpdateCutoverRefused,
            update_operation_needing_recovery,
        )

        try:
            pending = update_operation_needing_recovery(
                self.layout.update_checkpoints_root,
                expected_uid=self._service_uid,
            )
            journals = unfinished_rollback_journals(
                self.layout.update_checkpoints_root,
                expected_uid=self._service_uid,
            )
        except (OSError, UpdateCheckpointRefused, UpdateCutoverRefused) as exc:
            with suppress(InstalledServiceControlRefused):
                self._system_service.stop()
            raise UpdateRefused(
                "Update recovery state is unsafe. RCP kept the service stopped; inspect the "
                "private update receipts and journals."
            ) from exc
        if pending is None:
            if journals:
                with suppress(InstalledServiceControlRefused):
                    self._system_service.stop()
                raise UpdateRefused(
                    "An unfinished rollback journal has no active update receipt. RCP kept the "
                    "service stopped for repair."
                )
            return
        _path, operation, digest = pending
        coordinator = UpdateCutoverCoordinator(
            layout=self.layout,
            actions=_LinuxRecoveryActions(self, operation.base_commit),
            expected_uid=self._service_uid,
            expected_gid=self._service_gid,
        )
        try:
            if operation.state in {"committed", "rolled_back"}:
                recovered, _recovered_digest = coordinator.repair_selected_release(
                    operation,
                    digest,
                )
            else:
                recovered, _recovered_digest = coordinator.recover(operation, digest)
        except (InstalledServiceControlRefused, ServerControlError, UpdateCutoverRefused) as exc:
            with suppress(InstalledServiceControlRefused):
                self._system_service.stop()
            raise UpdateRefused(
                "Unfinished source update recovery failed. RCP kept the service stopped; rerun "
                "the same command after inspecting doctor and the durable receipt."
            ) from exc
        raise UpdateRefused(
            f"Recovered unfinished source update as {recovered.state}. Rerun sudo rcp server "
            "update to begin a fresh, fully inspected command."
        )

    def _validate_source_checkout(
        self,
        config: InstalledServerConfig,
        *,
        expected_head: str | None,
        environment: dict[str, str],
    ) -> str:
        source = self.layout.source_checkout
        _require_owned_directory(
            source,
            uid=self._service_uid,
            gid=self._service_gid,
            mode=None,
            label="managed source checkout",
        )
        _require_owned_directory(
            source / ".git",
            uid=self._service_uid,
            gid=self._service_gid,
            mode=None,
            label="managed source Git directory",
        )
        origin = self._git_text(source, ("remote", "get-url", "origin"), environment=environment)
        branch = self._git_text(
            source, ("symbolic-ref", "--short", "HEAD"), environment=environment
        )
        dirty = self._git_text(
            source,
            ("status", "--porcelain", "--untracked-files=all"),
            environment=environment,
        )
        head = self._git_text(
            source,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            environment=environment,
        )
        _require_commit(head)
        if origin != config.source.origin:
            raise UpdateRefused("Managed source origin differs from installed configuration.")
        if branch != config.source.branch:
            raise UpdateRefused("Managed source is not checked out on configured main.")
        if dirty:
            raise UpdateRefused(
                "Managed source has tracked or untracked changes. Preserve and inspect them; "
                "RCP will not reset, clean, or stash."
            )
        if expected_head is not None and head != expected_head:
            raise UpdateRefused("Managed source HEAD changed after doctor inspection.")
        return head

    def _validate_release_git(self, release: Path, commit: str) -> None:
        _require_owned_directory(
            release,
            uid=self._service_uid,
            gid=self._service_gid,
            mode=None,
            label="candidate release",
        )
        _require_safe_descendant_file(
            release / ".git",
            root=release,
            uid=self._service_uid,
            gid=self._service_gid,
            executable=False,
            label="candidate Git worktree link",
        )
        top_level = self._git_text(release, ("rev-parse", "--show-toplevel"))
        common = self._git_text(
            release,
            ("rev-parse", "--path-format=absolute", "--git-common-dir"),
        )
        if top_level != str(release) or common != str(self.layout.source_checkout / ".git"):
            raise UpdateRefused("Candidate release is not the managed source's detached worktree.")
        symbolic_head = self._git_result(release, ("symbolic-ref", "--quiet", "HEAD"))
        if symbolic_head.returncode == 0:
            raise UpdateRefused("Candidate release HEAD is attached to a branch, not detached.")
        if symbolic_head.returncode != 1:
            raise UpdateRefused("Candidate release detached-HEAD state could not be proven.")
        head = self._git_text(release, ("rev-parse", "--verify", "HEAD^{commit}"))
        dirty = self._git_text(release, ("status", "--porcelain", "--untracked-files=all"))
        if head != commit:
            raise UpdateRefused("Candidate release Git identity differs from its directory name.")
        if dirty:
            raise UpdateRefused(
                "Candidate release has tracked or untracked changes. RCP will not clean or replace it."
            )

    def _validate_built_release(self, release: Path, commit: str) -> str:
        self._validate_release_git(release, commit)
        executable = release / ".venv" / "bin" / "rcp"
        python = release / ".venv" / "bin" / "python"
        web_root = release / "web" / "dist"
        _require_safe_descendant_file(
            executable,
            root=release,
            uid=self._service_uid,
            gid=self._service_gid,
            executable=True,
            label="candidate Python entry point",
        )
        _require_safe_descendant_file(
            web_root / "index.html",
            root=release,
            uid=self._service_uid,
            gid=self._service_gid,
            executable=False,
            label="candidate Web entry point",
        )
        version = self._run_service_checked(
            (str(python), "--version"),
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            error="The candidate Python runtime could not execute.",
        )
        if not (version.stdout or version.stderr).startswith("Python 3.12."):
            raise UpdateRefused("The candidate does not use the required Python 3.12 runtime.")
        try:
            return web_build_identity(web_root)
        except ServerMetadataError as exc:
            raise UpdateRefused(
                "The candidate Web bundle is unsafe, incomplete, or too large."
            ) from exc

    def _update_git_environment(self, config: InstalledServerConfig) -> dict[str, str]:
        environment = source_git_environment(config.source, self.layout)
        index = int(environment["GIT_CONFIG_COUNT"])
        environment["GIT_CONFIG_COUNT"] = str(index + 1)
        environment[f"GIT_CONFIG_KEY_{index}"] = "core.hooksPath"
        environment[f"GIT_CONFIG_VALUE_{index}"] = "/dev/null"
        return environment

    def _git_text(
        self,
        root: Path,
        argv: tuple[str, ...],
        *,
        environment: dict[str, str] | None = None,
    ) -> str:
        result = self._git_result(root, argv, environment=environment)
        if result.returncode != 0:
            raise UpdateRefused(
                "A managed Git identity check failed. Inspect the checkout and rerun."
            )
        value = result.stdout.strip()
        if "\n" in value:
            raise UpdateRefused("Git returned an invalid multi-line identity.")
        return value

    def _git_result(
        self,
        root: Path,
        argv: tuple[str, ...],
        *,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        safe_environment = environment or {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        return self._run_service(
            ("git", "-C", str(root), *argv),
            environment=safe_environment,
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            capture_output=True,
        )

    def _run_git(
        self,
        root: Path,
        argv: tuple[str, ...],
        *,
        environment: dict[str, str] | None,
        timeout: float,
        error: str,
    ) -> None:
        self._run_service_checked(
            ("git", "-C", str(root), *argv),
            environment=environment,
            timeout=timeout,
            capture_output=False,
            error=error,
        )

    def _run_service_checked(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
        timeout: float,
        capture_output: bool = True,
        error: str,
    ) -> subprocess.CompletedProcess[str]:
        result = self._run_service(
            argv,
            cwd=cwd,
            environment=environment,
            timeout=timeout,
            capture_output=capture_output,
        )
        if result.returncode != 0:
            raise UpdateRefused(error)
        return result

    def _run_service(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
        timeout: float,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._service_runner(
                argv,
                cwd=cwd,
                environment=environment,
                timeout=timeout,
                capture_output=capture_output,
            )
        except (OSError, subprocess.TimeoutExpired):
            return subprocess.CompletedProcess(argv, 126, "", "")

    def _run_as_installed_service(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        environment: dict[str, str] | None,
        timeout: float,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        account = pwd.getpwnam(self.layout.service_account)
        return _run_as_account(
            account,
            argv,
            cwd=cwd,
            environment=environment,
            timeout=timeout,
            capture_output=capture_output,
        )

    def _doctor_runner(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self._run_service(
            argv,
            cwd=cwd,
            environment={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
            },
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            capture_output=True,
        )

    def _read_receipt_if_present(self, commit: str) -> BuiltCandidateReceipt | None:
        path = built_candidate_receipt_path(commit, self.layout)
        if not path.exists() and not path.is_symlink():
            return None
        return self._read_receipt(path)

    def _read_receipt(self, path: Path) -> BuiltCandidateReceipt:
        descriptor = -1
        try:
            if path.parent != self.layout.update_checkpoints_root:
                raise ValueError("receipt outside update root")
            _require_owned_directory(
                self.layout.update_checkpoints_root,
                uid=self._service_uid,
                gid=self._service_gid,
                mode=_DIRECTORY_MODE,
                label="update checkpoints root",
            )
            flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or (info.st_uid, info.st_gid) != (self._service_uid, self._service_gid)
                or stat.S_IMODE(info.st_mode) != _RECEIPT_MODE
                or info.st_size > _MAX_RECEIPT_BYTES
            ):
                raise ValueError("unsafe receipt")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                content = stream.read(_MAX_RECEIPT_BYTES + 1)
            if len(content) > _MAX_RECEIPT_BYTES:
                raise ValueError("oversized receipt")
            receipt = BuiltCandidateReceipt.model_validate_json(content)
        except (OSError, UnicodeError, ValueError) as exc:
            raise UpdateRefused(
                "A built-candidate receipt is unsafe or invalid. Preserve and inspect it; RCP "
                "will not overwrite it."
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if path != built_candidate_receipt_path(receipt.candidate_commit, self.layout):
            raise UpdateRefused("A built-candidate receipt filename and commit disagree.")
        if receipt.release_path != str(self.layout.release_dir(receipt.candidate_commit)):
            raise UpdateRefused("A built-candidate receipt names a noncanonical release path.")
        if receipt.receipt_path != str(path):
            raise UpdateRefused("A built-candidate receipt names a noncanonical receipt path.")
        return receipt

    def _validate_receipt_for_target(
        self,
        receipt: BuiltCandidateReceipt,
        target: UpdateTarget,
    ) -> None:
        if (
            receipt.installation_id != target.inspection.config.installation_id
            or receipt.source_origin != target.inspection.config.source.origin
            or receipt.source_branch != target.inspection.config.source.branch
            or receipt.base_current_commit != target.inspection.current_commit
            or receipt.base_running_commit != target.inspection.running_commit
            or receipt.base_instance_id != target.inspection.instance_id
            or receipt.base_process_pid != target.inspection.process_pid
            or receipt.candidate_commit != target.target_commit
        ):
            raise UpdateRefused(
                "An existing built-candidate receipt belongs to different source or live base "
                "state. Preserve it for diagnosis; RCP will not overwrite it."
            )

    def _publish_receipt(self, receipt: BuiltCandidateReceipt) -> BuiltCandidateReceipt:
        _require_owned_directory(
            self.layout.update_checkpoints_root,
            uid=self._service_uid,
            gid=self._service_gid,
            mode=_DIRECTORY_MODE,
            label="update checkpoints root",
        )
        path = built_candidate_receipt_path(receipt.candidate_commit, self.layout)
        existing = self._read_receipt_if_present(receipt.candidate_commit)
        if existing is not None:
            return existing
        content = receipt.model_dump_json().encode("utf-8") + b"\n"
        if len(content) > _MAX_RECEIPT_BYTES:
            raise UpdateRefused("The built-candidate receipt exceeds its fixed size bound.")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchown(descriptor, self._service_uid, self._service_gid)
            os.fchmod(descriptor, _RECEIPT_MODE)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                return self._read_receipt(path)
            _fsync_directory(path.parent)
        except UpdateRefused:
            raise
        except OSError as exc:
            raise UpdateRefused(
                "The candidate is built, but its immutable receipt could not be published. "
                "The current service remains unchanged."
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        return self._read_receipt(path)


@dataclass(frozen=True)
class _LinuxCutoverActions:
    machine: LinuxUpdateMachine
    target: UpdateTarget
    built: BuiltCandidateReceipt

    def enter_maintenance(self, *, operation_id: str, receipt_sha256: str):
        return self.machine._enter_update_maintenance(
            operation_id=operation_id,
            receipt_sha256=receipt_sha256,
        )

    def final_rehearsal(
        self,
        operation: UpdateOperationReceipt,
        receipt_sha256: str,
    ) -> VerifiedCandidateReceipt:
        return self.machine._final_maintenance_rehearsal(
            self.target,
            self.built,
            operation,
            receipt_sha256,
        )

    def create_checkpoint(
        self,
        final_receipt: VerifiedCandidateReceipt,
        *,
        sqlite_receipt_path: Path,
        sqlite_receipt_sha256: str,
        project_receipt_path: Path,
        project_receipt_sha256: str,
    ) -> VerifiedUpdateCheckpoint:
        return self.machine.create_rollback_checkpoint(
            self.target,
            final_receipt,
            sqlite_receipt_path=sqlite_receipt_path,
            sqlite_receipt_sha256=sqlite_receipt_sha256,
            project_receipt_path=project_receipt_path,
            project_receipt_sha256=project_receipt_sha256,
        )

    def stop_service(self) -> None:
        self.machine._system_service.stop()

    def switch_current(self, *, expected: Path, target: Path) -> None:
        self.machine._system_service.switch_current(expected=expected, target=target)

    def start_service(self) -> int:
        return self.machine._system_service.start()

    def control_for_running(self, commit: str):
        return self.machine._control_for_running(commit)

    def restore_checkpoint(self, checkpoint_path: Path, checkpoint_sha256: str) -> None:
        self.machine._restore_cutover_checkpoint(
            checkpoint_path,
            checkpoint_sha256,
            self.target.inspection.running_commit,
        )

    def current_release(self) -> Path:
        return self.machine._system_service.current_release()


@dataclass(frozen=True)
class _LinuxRecoveryActions:
    machine: LinuxUpdateMachine
    base_commit: str

    def stop_service(self) -> None:
        self.machine._system_service.stop()

    def switch_current(self, *, expected: Path, target: Path) -> None:
        self.machine._system_service.switch_current(expected=expected, target=target)

    def start_service(self) -> int:
        return self.machine._system_service.start()

    def control_for_running(self, commit: str):
        return self.machine._control_for_running(commit)

    def restore_checkpoint(self, checkpoint_path: Path, checkpoint_sha256: str) -> None:
        self.machine._restore_cutover_checkpoint(
            checkpoint_path,
            checkpoint_sha256,
            self.base_commit,
        )

    def current_release(self) -> Path:
        return self.machine._system_service.current_release()

    def enter_maintenance(self, **_kwargs):  # pragma: no cover - recovery rejects this phase
        raise AssertionError("recovery cannot enter new maintenance")

    def final_rehearsal(self, *_args):  # pragma: no cover - recovery never recaptures
        raise AssertionError("recovery cannot rehearse a new capture")

    def create_checkpoint(self, *_args, **_kwargs):  # pragma: no cover - recovery never snapshots
        raise AssertionError("recovery cannot create a new checkpoint")


def _require_commit(value: str) -> None:
    if _FULL_GIT_COMMIT.fullmatch(value) is None:
        raise UpdateRefused("Git returned a noncanonical commit id; update stopped safely.")


def _reject_symlink_ancestry(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise UpdateRefused(f"Managed update path ancestry contains a symlink at {candidate}.")


def _require_owned_directory(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int | None,
    label: str,
) -> None:
    _reject_symlink_ancestry(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise UpdateRefused(f"The {label} is missing or unreadable.") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_uid, info.st_gid) != (uid, gid)
        or (mode is not None and stat.S_IMODE(info.st_mode) != mode)
    ):
        raise UpdateRefused(f"The {label} has unexpected type, ownership, or mode.")


def _require_safe_descendant_file(
    path: Path,
    *,
    root: Path,
    uid: int,
    gid: int,
    executable: bool,
    label: str,
) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise UpdateRefused(f"The {label} is outside the candidate release.") from exc
    current = root
    for component in relative.parts[:-1]:
        current /= component
        try:
            info = current.lstat()
        except OSError as exc:
            raise UpdateRefused(f"The {label} has missing parent directories.") from exc
        if not stat.S_ISDIR(info.st_mode) or (info.st_uid, info.st_gid) != (uid, gid):
            raise UpdateRefused(f"The {label} has unsafe parent-directory ancestry.")
    try:
        info = path.lstat()
    except OSError as exc:
        raise UpdateRefused(f"The {label} is missing or unreadable.") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or (info.st_uid, info.st_gid) != (uid, gid)
        or (executable and not stat.S_IMODE(info.st_mode) & 0o111)
    ):
        raise UpdateRefused(f"The {label} has unexpected type, ownership, or mode.")


def _owned_file_sha256(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
    maximum: int,
    label: str,
) -> str:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_uid, info.st_gid) != (uid, gid)
            or stat.S_IMODE(info.st_mode) != mode
            or info.st_size > maximum
        ):
            raise ValueError("unsafe file")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise ValueError("unsafe file size")
            chunks.append(chunk)
            remaining -= len(chunk)
        final = os.fstat(descriptor)
        path_final = path.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(info, name) != getattr(final, name) for name in stable) or any(
            getattr(final, name) != getattr(path_final, name) for name in stable
        ):
            raise ValueError("unstable file")
        content = b"".join(chunks)
    except (OSError, ValueError) as exc:
        raise UpdateRefused(f"The {label} has unsafe identity or bytes.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(content).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BuiltCandidateReceipt",
    "CandidateBuild",
    "LinuxUpdateMachine",
    "UpdateInspection",
    "UpdateMachine",
    "UpdateRefused",
    "UpdateTarget",
    "built_candidate_receipt_path",
    "prepare_update_command",
    "server_update_operation_lock",
]
