"""Safe target-side receipt and activation of one project transfer.

Upload remains a byte-only boundary. A separate running-service coordinator
decodes and imports those verified bytes, then crosses the compound catalog,
membership, provisioning, and activation boundary.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol

import tomlkit

from rcp.config import Manifest
from rcp.limits import PROJECT_TRANSFER_COPY_BUFFER_BYTES
from rcp.server_ops.cli import CallerIdentity, PreparedServerCommand, ServerEventEmitter
from rcp.server_ops.control import (
    ServerControlClient,
    ServerControlError,
    ServerControlProjectTransferActivationResult,
    ServerControlProjectTransferUploadResult,
)
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.models import MachineTarget, ServerCommandRequest, ServerPlanEvent, ServerStep
from rcp.server_runtime import ServerMetadata
from rcp.setup import ProjectSetupManager, render_prepared_team_manifest
from rcp.storage import AppStore, ProjectTransferActivationReceipt
from rcp.transfer.archive import TransferArchiveEnvelope
from rcp.transfer.configuration import (
    TransferTargetConfigurationReceipt,
    build_transfer_target_configuration,
)
from rcp.transfer.importer import import_project_transfer
from rcp.transfer.source import discard_transfer_archive_stage, stage_transfer_archive
from rcp.transport.state import state_workspace_for_probe

if TYPE_CHECKING:
    from rcp.projects import ProjectCatalog

_TRANSFER_INBOX_NAME = "transfer-inbox"
_ACTIVATION_STAGE_NAME = "project-transfer-activation"
_ARCHIVE_SUFFIX = ".rcp-transfer"
_PARTIAL_SUFFIX = ".partial"
_ARCHIVE_MODE = 0o600
_DIRECTORY_MODE = 0o700
_PRIVATE_BITS = 0o077
_HEX_DIGITS = frozenset("0123456789abcdef")


class TargetTransferUploadError(ValueError):
    """A target upload cannot safely reach its durable byte boundary."""


class TargetTransferUploadBusy(TargetTransferUploadError):
    """Another process currently owns this request's upload lease."""


class TargetTransferUploadControl(Protocol):
    def project_transfer_upload_plan(
        self,
        *,
        request_id: str,
    ) -> ServerControlProjectTransferUploadResult: ...

    def complete_project_transfer_upload(
        self,
        *,
        request_id: str,
        lease_boundary_sha256: str,
    ) -> ServerControlProjectTransferUploadResult: ...

    def activate_project_transfer(
        self,
        *,
        request_id: str,
        lease_boundary_sha256: str,
    ) -> ServerControlProjectTransferActivationResult: ...


@dataclass(frozen=True)
class TargetTransferUploadReceipt:
    """The exact target file published for one request and digest."""

    request_id: str
    archive_sha256: str
    archive_size_bytes: int
    archive_path: Path
    reused_existing: bool = False

    def __post_init__(self) -> None:
        _canonical_request_id(self.request_id)
        _canonical_digest(self.archive_sha256)
        _positive_size(self.archive_size_bytes)
        if (
            self.archive_path.name != f"{self.request_id}{_ARCHIVE_SUFFIX}"
            or self.archive_path.parent.name != _TRANSFER_INBOX_NAME
        ):
            raise ValueError("target upload receipt names the wrong request archive")


@dataclass
class TargetTransferUploadLease:
    """One held, request-scoped target upload lease.

    The lease is an advisory ``flock`` on a request-derived run-stage file.  The open
    descriptor is intentionally kept on this object: the lease therefore ends
    when the owner closes it or its process dies.  The lock path remains
    outside the transfer inbox, so it cannot be mistaken for archive data.
    """

    request_id: str
    archive_sha256: str
    archive_size_bytes: int
    inbox_directory: Path
    archive_path: Path
    partial_path: Path
    _lock_descriptor: int
    _released: bool = False

    def __post_init__(self) -> None:
        _canonical_request_id(self.request_id)
        _canonical_digest(self.archive_sha256)
        _positive_size(self.archive_size_bytes)
        data_dir = self.inbox_directory.parent
        if self.archive_path != target_transfer_archive_path(data_dir, self.request_id):
            raise ValueError("target upload lease archive escaped its request directory")
        if self.partial_path != target_transfer_partial_path(
            data_dir,
            self.request_id,
            self.archive_sha256,
        ):
            raise ValueError("target upload lease partial escaped its request directory")
        if isinstance(self._lock_descriptor, bool) or self._lock_descriptor < 0:
            raise ValueError("target upload lease descriptor is invalid")

    @property
    def active(self) -> bool:
        return not self._released

    def receive(self, source: BinaryIO) -> TargetTransferUploadReceipt:
        """Stream ``source`` into this lease's exact final file."""

        if self._released:
            raise TargetTransferUploadError("target upload lease is no longer active")
        return _receive_with_lease(self, source)

    def release(self) -> None:
        """Release the OS lease exactly once."""

        if self._released:
            return
        self._released = True
        try:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_descriptor)

    def __enter__(self) -> TargetTransferUploadLease:
        if self._released:
            raise TargetTransferUploadError("target upload lease is no longer active")
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


def target_transfer_archive_path(data_dir: Path, request_id: str) -> Path:
    """Return the only final upload path for ``request_id``."""

    request = _canonical_request_id(request_id)
    return Path(data_dir) / _TRANSFER_INBOX_NAME / f"{request}{_ARCHIVE_SUFFIX}"


def target_transfer_partial_path(
    data_dir: Path,
    request_id: str,
    archive_sha256: str,
) -> Path:
    """Return the request/digest-specific incomplete upload path.

    Including the expected digest in the name is important during recovery:
    retrying a request can remove only the partial it negotiated, never a
    different request or a partial bound to another digest.
    """

    request = _canonical_request_id(request_id)
    digest = _canonical_digest(archive_sha256)
    return Path(data_dir) / _TRANSFER_INBOX_NAME / (f".{request}.{digest}{_PARTIAL_SUFFIX}")


def prepare_transfer_import_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    control: TargetTransferUploadControl | None = None,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> PreparedServerCommand:
    """Prepare the fixed stdin-only upload command without opening SQLite."""

    if request.command != "server project transfer-import" or request.request_id is None:
        raise ValueError("prepare_transfer_import_command requires one transfer request")
    client = control or ServerControlClient.from_data_dir(
        layout.data_dir,
        expected_server_uid=os.geteuid(),
    )
    resolved = client.project_transfer_upload_plan(request_id=request.request_id)
    target = MachineTarget(host=identity.host, os_account=identity.username)
    pending = ServerStep(
        number=1,
        title="Receive the confirmed project transfer",
        purpose="Store exactly the archive bytes already bound to this transfer request.",
        performed_by="system",
        target=target,
        phase="transfer_upload",
        state="pending",
        expected_success="The exact request archive is sealed in the private transfer inbox.",
        message="RCP will verify stdin against the request's expected digest and size.",
    )
    activation_pending = ServerStep(
        number=2,
        title="Activate the transferred project",
        purpose="Decode, validate, import, and register the exact reviewed project.",
        performed_by="system",
        target=target,
        phase="transfer_activation",
        state="pending",
        expected_success="The imported project is registered once in the target team space.",
        message="RCP will cross one compound activation boundary after upload readback.",
    )
    plan = ServerPlanEvent(
        command=request.command,
        timestamp=datetime.now(UTC),
        steps=(pending, activation_pending),
    )

    def execute(emitter: ServerEventEmitter, input_stream: BinaryIO) -> None:
        emitter.emit_step(
            pending.model_copy(
                update={
                    "state": "running",
                    "message": "RCP is receiving the exact request-bound archive from stdin.",
                }
            )
        )
        try:
            if resolved.state == "consumed":
                _consume_stream(
                    input_stream,
                    resolved.archive_sha256,
                    resolved.archive_size_bytes,
                )
            elif resolved.state == "complete":
                replay_target_transfer_archive(
                    layout.data_dir,
                    request.request_id,
                    archive_sha256=resolved.archive_sha256,
                    archive_size_bytes=resolved.archive_size_bytes,
                    source=input_stream,
                )
            else:
                upload_target_transfer_archive(
                    layout.data_dir,
                    request.request_id,
                    archive_sha256=resolved.archive_sha256,
                    archive_size_bytes=resolved.archive_size_bytes,
                    source=input_stream,
                )
            completed = client.complete_project_transfer_upload(
                request_id=request.request_id,
                lease_boundary_sha256=resolved.lease_boundary_sha256,
            )
            if (
                completed.request_id != resolved.request_id
                or completed.project_id != resolved.project_id
                or completed.archive_sha256 != resolved.archive_sha256
                or completed.archive_size_bytes != resolved.archive_size_bytes
                or completed.lease_boundary_sha256 != resolved.lease_boundary_sha256
            ):
                raise TargetTransferUploadError(
                    "the running server completed another transfer upload boundary"
                )
        except (ServerControlError, TargetTransferUploadError, OSError) as exc:
            emitter.emit_step(pending.model_copy(update={"state": "failed", "message": str(exc)}))
            return
        emitter.emit_step(
            pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": "The exact transfer archive is sealed and durably recorded.",
                }
            )
        )
        emitter.emit_step(
            activation_pending.model_copy(
                update={
                    "state": "running",
                    "message": "RCP is activating the exact completed upload.",
                }
            )
        )
        try:
            activated = client.activate_project_transfer(
                request_id=request.request_id,
                lease_boundary_sha256=resolved.lease_boundary_sha256,
            )
            if (
                activated.target_request_id != resolved.request_id
                or activated.project_id != resolved.project_id
                or activated.archive_sha256 != resolved.archive_sha256
                or activated.upload_lease_boundary_sha256 != resolved.lease_boundary_sha256
            ):
                raise TargetTransferUploadError(
                    "the running server activated another project-transfer boundary"
                )
        except (ServerControlError, TargetTransferUploadError, OSError) as exc:
            emitter.emit_step(
                activation_pending.model_copy(update={"state": "failed", "message": str(exc)})
            )
            return
        emitter.emit_step(
            activation_pending.model_copy(
                update={
                    "state": "succeeded",
                    "message": "The transferred project is active in the target team space.",
                }
            )
        )

    return PreparedServerCommand(plan=plan, execute=execute)


class TargetTransferUploadCoordinator:
    """Bind the private control lease to the exact target inbox file."""

    def __init__(self, store: AppStore, data_dir: Path, identity: ServerMetadata) -> None:
        self.store = store
        self.data_dir = Path(data_dir)
        self.identity = identity

    def plan(self, request_id: str) -> ServerControlProjectTransferUploadResult:
        upload = self.store.begin_target_project_transfer_upload(request_id)
        return ServerControlProjectTransferUploadResult(
            instance_id=self.identity.instance_id,
            pid=self.identity.pid,
            data_dir_id=self.identity.data_dir_id,
            space_id=self.store.space_id,
            request_id=upload.request_id,
            project_id=upload.project_id,
            archive_sha256=upload.archive_sha256,
            archive_size_bytes=upload.archive_size_bytes,
            lease_boundary_sha256=upload.lease_boundary_sha256,
            state=upload.status,
        )

    def complete(
        self,
        request_id: str,
        *,
        lease_boundary_sha256: str,
    ) -> ServerControlProjectTransferUploadResult:
        upload = self.store.target_project_transfer_upload(request_id)
        if upload is None:
            raise TargetTransferUploadError("target transfer upload has no durable lease")
        if upload.lease_boundary_sha256 != lease_boundary_sha256:
            raise TargetTransferUploadError("target transfer upload lease boundary changed")
        if upload.status == "consumed":
            completed = self.store.complete_target_project_transfer_upload(
                request_id,
                lease_boundary_sha256=lease_boundary_sha256,
            )
        else:
            with target_transfer_upload_lease(
                self.data_dir,
                request_id,
                archive_sha256=upload.archive_sha256,
                archive_size_bytes=upload.archive_size_bytes,
            ):
                verify_target_transfer_archive(
                    self.data_dir,
                    request_id,
                    expected_sha256=upload.archive_sha256,
                    expected_size_bytes=upload.archive_size_bytes,
                )
                completed = self.store.complete_target_project_transfer_upload(
                    request_id,
                    lease_boundary_sha256=lease_boundary_sha256,
                )
        return ServerControlProjectTransferUploadResult(
            instance_id=self.identity.instance_id,
            pid=self.identity.pid,
            data_dir_id=self.identity.data_dir_id,
            space_id=self.store.space_id,
            request_id=completed.request_id,
            project_id=completed.project_id,
            archive_sha256=completed.archive_sha256,
            archive_size_bytes=completed.archive_size_bytes,
            lease_boundary_sha256=completed.lease_boundary_sha256,
            state=completed.status,
        )

    def uploads_idle(self) -> bool:
        return all(
            upload.status != "active" for upload in self.store.target_project_transfer_uploads()
        )


class TargetTransferActivationCoordinator:
    """Decode, import, and atomically activate one complete target upload."""

    def __init__(
        self,
        store: AppStore,
        catalog: ProjectCatalog,
        setup: ProjectSetupManager,
        data_dir: Path,
    ) -> None:
        self.store = store
        self.catalog = catalog
        self.setup = setup
        self.data_dir = Path(data_dir)

    def activate(
        self,
        request_id: str,
        *,
        lease_boundary_sha256: str,
    ) -> ProjectTransferActivationReceipt:
        """Finish one request or return its exact prior activation receipt."""

        request = _canonical_request_id(request_id)
        upload = self.store.target_project_transfer_upload(request)
        if upload is None:
            raise TargetTransferUploadError("target transfer activation has no durable upload")
        if upload.lease_boundary_sha256 != _canonical_digest(lease_boundary_sha256):
            raise TargetTransferUploadError("target transfer activation lease boundary changed")

        stage_parent = self.data_dir / "run-stage" / _ACTIVATION_STAGE_NAME
        _ensure_private_directory(self.data_dir, label="RCP data")
        _ensure_private_directory(self.data_dir / "run-stage", label="RCP run stage")
        _ensure_private_directory(stage_parent, label="target activation stage root")
        request_stage = stage_parent / request

        with target_transfer_upload_lease(
            self.data_dir,
            request,
            archive_sha256=upload.archive_sha256,
            archive_size_bytes=upload.archive_size_bytes,
        ):
            _discard_activation_stage(request_stage)
            committed = self.store.target_project_transfer_activation(request)
            if committed is not None:
                self._finish_committed_retry(committed, upload)
                return committed

            self._require_ready_boundary(request, upload)
            archive_receipt = verify_target_transfer_archive(
                self.data_dir,
                request,
                expected_sha256=upload.archive_sha256,
                expected_size_bytes=upload.archive_size_bytes,
            )
            request_stage.mkdir(mode=_DIRECTORY_MODE)
            payload_root = request_stage / "payload"
            try:
                readback = stage_transfer_archive(
                    archive_receipt.archive_path,
                    payload_root,
                )
                self._require_envelope_matches_upload(readback.envelope, upload)
                if os.path.lexists(payload_root / "manifest.json"):
                    raise TargetTransferUploadError(
                        "decoded transfer payload retained its reserved archive manifest"
                    )

                transfer = self.store.project_transfer_request(request)
                provisioning = self.store.project_provisioning_request(request)
                if transfer is None or provisioning is None:
                    raise TargetTransferUploadError(
                        "target transfer activation lost its durable requests"
                    )
                if transfer.link_receipt is None:
                    raise TargetTransferUploadError(
                        "target transfer activation lost its link receipt"
                    )

                existing_import = self.store.project_transfer_import(request)
                if existing_import is None:
                    configuration = build_transfer_target_configuration(
                        provisioning,
                        transfer.source_configuration,
                        transfer.link_receipt,
                        readback.manifest,
                        payload_root,
                        retained_research_root=self._capture_retained_history(
                            provisioning,
                            request_stage,
                        ),
                    )
                else:
                    persisted_json = self.store.project_transfer_import_configuration_receipt_json(
                        request
                    )
                    if persisted_json is None:
                        raise TargetTransferUploadError(
                            "target transfer import lost its pre-publication configuration receipt"
                        )
                    persisted = TransferTargetConfigurationReceipt.model_validate_json(
                        persisted_json
                    )
                    configuration = build_transfer_target_configuration(
                        provisioning,
                        transfer.source_configuration,
                        transfer.link_receipt,
                        readback.manifest,
                        payload_root,
                        retained_history=persisted.retained_history,
                    )
                    if configuration.receipt != persisted:
                        raise TargetTransferUploadError(
                            "target transfer import configuration changed before retry"
                        )
                imported = import_project_transfer(
                    self.catalog,
                    archive=readback.manifest,
                    envelope=readback.envelope,
                    archive_root=payload_root,
                    target_configuration=configuration,
                )
                if imported.status != "complete" or imported.publication_sha256 is None:
                    raise TargetTransferUploadError(
                        "target transfer import did not reach its complete boundary"
                    )
                project = self.setup.prepare_incoming_transfer_project(
                    provisioning,
                    target_configuration=configuration,
                )
                assert provisioning.final_review_digest is not None
                committed = self.store.activate_target_project_transfer(
                    request,
                    project=project,
                    expected_provisioning_revision=provisioning.revision,
                    expected_final_review_digest=provisioning.final_review_digest,
                )
                activated_project = self.store.project(committed.project_id)
                if activated_project is None:
                    raise RuntimeError("target activation lost its registered project")
                self.catalog.refresh_after_incoming_transfer_activation(activated_project)
                _discard_verified_target_archive(
                    archive_receipt.archive_path,
                    expected_sha256=upload.archive_sha256,
                    expected_size_bytes=upload.archive_size_bytes,
                )
                return committed
            finally:
                _discard_activation_stage(request_stage)

    def _require_ready_boundary(self, request_id: str, upload) -> None:
        transfer = self.store.project_transfer_request(request_id)
        provisioning = self.store.project_provisioning_request(request_id)
        if transfer is None or provisioning is None:
            raise TargetTransferUploadError("target transfer activation lost its durable requests")
        if (
            transfer.side != "target"
            or transfer.phase != "archive_bound"
            or transfer.request_id != upload.request_id
            or transfer.project_id != upload.project_id
            or transfer.archive_sha256 != upload.archive_sha256
            or transfer.archive_size_bytes != upload.archive_size_bytes
            or transfer.target_space_id != self.store.space_id
            or transfer.link_receipt is None
            or transfer.source_release_receipt is None
            or transfer.target_admission_receipt is None
            or upload.status != "complete"
            or upload.receipt is None
        ):
            raise TargetTransferUploadError(
                "target transfer request is not ready for archive activation"
            )
        if (
            provisioning.kind != "incoming_transfer"
            or provisioning.status != "ready_for_review"
            or provisioning.request_id != request_id
            or provisioning.proposed_project_id != transfer.project_id
            or provisioning.target_space_id != transfer.target_space_id
            or provisioning.final_review_digest is None
        ):
            raise TargetTransferUploadError(
                "target transfer provisioning is not ready for archive activation"
            )

    @staticmethod
    def _require_envelope_matches_upload(envelope: TransferArchiveEnvelope, upload) -> None:
        if (
            envelope.archive_sha256 != upload.archive_sha256
            or envelope.archive_size_bytes != upload.archive_size_bytes
        ):
            raise TargetTransferUploadError(
                "decoded transfer archive differs from its completed upload"
            )

    def _capture_retained_history(self, provisioning, request_stage: Path) -> Path:
        content = render_prepared_team_manifest(provisioning)
        try:
            manifest = Manifest.model_validate(tomlkit.parse(content).unwrap())
        except (ValueError, tomlkit.exceptions.ParseError) as exc:
            raise TargetTransferUploadError(
                "reviewed target manifest is invalid during retained-history capture"
            ) from exc
        state_repository = manifest.repository_map[manifest.state.repository]
        machine = manifest.machine_map[state_repository.machine]
        manifest._path = (
            Path(state_repository.path) / ".research" / "manifest.toml"
            if not machine.host
            else request_stage / "target-bootstrap.toml"
        )
        workspace = state_workspace_for_probe(manifest, self.data_dir)
        retained_stage = request_stage / "retained"
        retained_stage.mkdir(mode=_DIRECTORY_MODE)
        return workspace.backup_source_root(retained_stage)

    def _finish_committed_retry(self, receipt, upload) -> None:
        if upload.status != "consumed":
            raise RuntimeError("target activation receipt lost its consumed upload boundary")
        project = self.store.project(receipt.project_id)
        if project is None:
            raise RuntimeError("target activation receipt lost its registered project")
        self.catalog.refresh_after_incoming_transfer_activation(project)
        archive_path = target_transfer_archive_path(self.data_dir, receipt.target_request_id)
        if os.path.lexists(archive_path):
            _discard_verified_target_archive(
                archive_path,
                expected_sha256=upload.archive_sha256,
                expected_size_bytes=upload.archive_size_bytes,
            )


def acquire_target_transfer_upload_lease(
    data_dir: Path,
    request_id: str,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
) -> TargetTransferUploadLease:
    """Acquire one exclusive request lease without opening SQLite.

    The caller must already have obtained the expected digest and byte count
    from the lock-owning server.  This function only validates those bounds
    and protects the corresponding filesystem paths; it does not decide
    whether the request is authorized to import a project.
    """

    request = _canonical_request_id(request_id)
    digest = _canonical_digest(archive_sha256)
    size = _positive_size(archive_size_bytes)
    data_root = Path(data_dir)
    _require_private_directory(data_root, label="RCP data")
    transfer_root = data_root / _TRANSFER_INBOX_NAME
    _ensure_private_directory(transfer_root, label="transfer inbox")
    run_stage = data_root / "run-stage"
    _ensure_private_directory(run_stage, label="RCP run stage")
    lock_root = run_stage / "transfer-upload-locks"
    _ensure_private_directory(lock_root, label="target upload lock root")
    lock_path = lock_root / f"{request}.lock"

    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            _ARCHIVE_MODE,
        )
    except OSError as exc:
        raise TargetTransferUploadError("the request upload lock cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        _require_private_file_metadata(metadata, label="request upload lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            if isinstance(exc, BlockingIOError) or exc.errno in {
                errno.EACCES,
                errno.EAGAIN,
            }:
                raise TargetTransferUploadBusy(
                    "another upload already owns this transfer request"
                ) from exc
            raise TargetTransferUploadError(
                "the target upload lease could not be acquired"
            ) from exc
        return TargetTransferUploadLease(
            request_id=request,
            archive_sha256=digest,
            archive_size_bytes=size,
            inbox_directory=transfer_root,
            archive_path=transfer_root / f"{request}{_ARCHIVE_SUFFIX}",
            partial_path=transfer_root / f".{request}.{digest}{_PARTIAL_SUFFIX}",
            _lock_descriptor=descriptor,
        )
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def target_transfer_upload_lease(
    data_dir: Path,
    request_id: str,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
) -> Iterator[TargetTransferUploadLease]:
    """Context-manager form of :func:`acquire_target_transfer_upload_lease`."""

    lease = acquire_target_transfer_upload_lease(
        data_dir,
        request_id,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
    )
    try:
        yield lease
    finally:
        lease.release()


def upload_target_transfer_archive(
    data_dir: Path,
    request_id: str,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    source: BinaryIO,
) -> TargetTransferUploadReceipt:
    """Receive stdin-like bytes and atomically publish one exact archive.

    The final file is verified before any incoming bytes are trusted.  If a
    prior exact final already exists, the incoming stream is still consumed
    and checked when supplied so a native relay can finish writing its pipe
    without receiving a premature broken-pipe result.  A failed or interrupted
    write leaves only the exact digest-specific partial for a later retry.
    """

    request = _canonical_request_id(request_id)
    digest = _canonical_digest(archive_sha256)
    size = _positive_size(archive_size_bytes)
    if not hasattr(source, "read"):
        raise TargetTransferUploadError("target upload input must be a binary stream")

    with target_transfer_upload_lease(
        data_dir,
        request,
        archive_sha256=digest,
        archive_size_bytes=size,
    ) as lease:
        if os.path.lexists(lease.archive_path):
            _verify_final_file(lease.archive_path, digest, size)
            _consume_stream(source, digest, size)
            _discard_known_partial(lease.partial_path)
            _fsync_directory(lease.inbox_directory)
            return TargetTransferUploadReceipt(
                request,
                digest,
                size,
                lease.archive_path,
                reused_existing=True,
            )
        return lease.receive(source)


def replay_target_transfer_archive(
    data_dir: Path,
    request_id: str,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    source: BinaryIO,
) -> TargetTransferUploadReceipt:
    """Verify a durable completion and consume the same retry bytes.

    A completed upload is never regenerated. Missing or changed bytes fail
    loudly while the input stream is accepted only when it exactly matches the
    existing archive.
    """

    with target_transfer_upload_lease(
        data_dir,
        request_id,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
    ) as lease:
        _verify_final_file(lease.archive_path, archive_sha256, archive_size_bytes)
        _consume_stream(source, archive_sha256, archive_size_bytes)
        _discard_known_partial(lease.partial_path)
        _fsync_directory(lease.inbox_directory)
        return TargetTransferUploadReceipt(
            request_id,
            archive_sha256,
            archive_size_bytes,
            lease.archive_path,
            reused_existing=True,
        )


def verify_target_transfer_archive(
    data_dir: Path,
    request_id: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> TargetTransferUploadReceipt:
    """Read back one complete target archive without mutating it."""

    request = _canonical_request_id(request_id)
    digest = _canonical_digest(expected_sha256)
    size = _positive_size(expected_size_bytes)
    path = target_transfer_archive_path(data_dir, request)
    _verify_final_file(path, digest, size)
    return TargetTransferUploadReceipt(request, digest, size, path, reused_existing=True)


def discard_target_transfer_partial(
    data_dir: Path,
    request_id: str,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
) -> bool:
    """Discard only one known incomplete partial under its request lease."""

    with target_transfer_upload_lease(
        data_dir,
        request_id,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
    ) as lease:
        present = os.path.lexists(lease.partial_path)
        if present:
            _discard_known_partial(lease.partial_path)
            _fsync_directory(lease.inbox_directory)
        return present


def _discard_activation_stage(path: Path) -> None:
    """Remove only the exact UUID-derived coordinator stage."""

    if not os.path.lexists(path):
        return
    discard_transfer_archive_stage(path)


def _discard_verified_target_archive(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> None:
    """Delete one exact consumed inbox file after immutable readback."""

    _verify_final_file(path, expected_sha256, expected_size_bytes)
    try:
        path.unlink()
    except OSError as exc:
        raise TargetTransferUploadError(
            "consumed target transfer archive could not be removed"
        ) from exc
    _fsync_directory(path.parent)


def _receive_with_lease(
    lease: TargetTransferUploadLease,
    source: BinaryIO,
) -> TargetTransferUploadReceipt:
    _discard_known_partial(lease.partial_path)
    descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        try:
            descriptor = os.open(
                lease.partial_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                _ARCHIVE_MODE,
            )
        except OSError as exc:
            raise TargetTransferUploadError(
                "the request-specific target partial cannot be created safely"
            ) from exc
        _require_private_file_descriptor(descriptor, label="target upload partial")
        while True:
            chunk = source.read(PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if chunk == b"":
                break
            if not isinstance(chunk, bytes):
                raise TargetTransferUploadError("target upload input must yield bytes")
            if len(chunk) > PROJECT_TRANSFER_COPY_BUFFER_BYTES:
                raise TargetTransferUploadError("target upload input exceeded its read bound")
            if size + len(chunk) > lease.archive_size_bytes:
                raise TargetTransferUploadError("target upload exceeds its expected size")
            _write_all(descriptor, chunk)
            digest.update(chunk)
            size += len(chunk)
        if size != lease.archive_size_bytes or digest.hexdigest() != lease.archive_sha256:
            raise TargetTransferUploadError("target upload does not match its expected bytes")
        os.fchmod(descriptor, _ARCHIVE_MODE)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _publish_no_overwrite(lease.partial_path, lease.archive_path)
        _fsync_directory(lease.inbox_directory)
        return TargetTransferUploadReceipt(
            lease.request_id,
            lease.archive_sha256,
            lease.archive_size_bytes,
            lease.archive_path,
        )
    except FileExistsError:
        # A second process cannot race us while this lease is held, but a
        # crash-recovery operator may have published the same final between
        # the initial check and link.  Accept only exact final bytes.
        _verify_final_file(lease.archive_path, lease.archive_sha256, lease.archive_size_bytes)
        _discard_known_partial(lease.partial_path)
        _fsync_directory(lease.inbox_directory)
        return TargetTransferUploadReceipt(
            lease.request_id,
            lease.archive_sha256,
            lease.archive_size_bytes,
            lease.archive_path,
            reused_existing=True,
        )
    except BaseException:
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _publish_no_overwrite(partial: Path, final: Path) -> None:
    """Publish with a hard-link, the POSIX no-overwrite primitive."""

    try:
        os.link(partial, final, follow_symlinks=False)
    except OSError:
        raise
    partial.unlink()


def _verify_final_file(path: Path, expected_digest: str, expected_size: int) -> None:
    descriptor = _open_private_file(path, label="target transfer archive")
    before = os.fstat(descriptor)
    try:
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > expected_size:
                raise TargetTransferUploadError("target transfer archive exceeds its receipt")
            digest.update(chunk)
        if size != expected_size or digest.hexdigest() != expected_digest:
            raise TargetTransferUploadError("target transfer archive differs from its receipt")
        after = os.fstat(descriptor)
        _require_private_file_metadata(after, label="target transfer archive")
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise TargetTransferUploadError("target transfer archive changed during readback")
    except OSError as exc:
        raise TargetTransferUploadError("target transfer archive cannot be read safely") from exc
    finally:
        os.close(descriptor)


def _consume_stream(source: BinaryIO, expected_digest: str, expected_size: int) -> None:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = source.read(PROJECT_TRANSFER_COPY_BUFFER_BYTES)
        if chunk == b"":
            break
        if not isinstance(chunk, bytes):
            raise TargetTransferUploadError("target upload input must yield bytes")
        if len(chunk) > PROJECT_TRANSFER_COPY_BUFFER_BYTES:
            raise TargetTransferUploadError("target upload input exceeded its read bound")
        size += len(chunk)
        if size > expected_size:
            raise TargetTransferUploadError("target upload exceeds its expected size")
        digest.update(chunk)
    if size != expected_size or digest.hexdigest() != expected_digest:
        raise TargetTransferUploadError("retry input does not match the existing receipt")


def _discard_known_partial(path: Path) -> None:
    if not os.path.lexists(path):
        return
    _open_and_close_private_file(path, label="request-specific target partial")
    try:
        path.unlink()
    except OSError as exc:
        raise TargetTransferUploadError(
            "request-specific target partial cannot be removed"
        ) from exc


def _open_and_close_private_file(path: Path, *, label: str) -> None:
    descriptor = _open_private_file(path, label=label)
    os.close(descriptor)


def _open_private_file(path: Path, *, label: str) -> int:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TargetTransferUploadError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise TargetTransferUploadError(f"{label} refuses symlinks")
    _require_private_file_metadata(metadata, label=label)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise TargetTransferUploadError(f"{label} cannot be opened safely") from exc
    opened = os.fstat(descriptor)
    try:
        _require_private_file_metadata(opened, label=label)
        if _stat_fingerprint(metadata) != _stat_fingerprint(opened):
            raise TargetTransferUploadError(f"{label} changed while being opened")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_private_file_descriptor(descriptor: int, *, label: str) -> None:
    metadata = os.fstat(descriptor)
    _require_private_file_metadata(metadata, label=label)


def _require_private_file_metadata(metadata: os.stat_result, *, label: str) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != _ARCHIVE_MODE
    ):
        raise TargetTransferUploadError(f"{label} must be an owned mode-0600 regular file")


def _ensure_private_directory(path: Path, *, label: str) -> None:
    try:
        path.mkdir(mode=_DIRECTORY_MODE)
    except FileExistsError:
        pass
    except OSError as exc:
        raise TargetTransferUploadError(f"{label} cannot be created privately") from exc
    _require_private_directory(path, label=label)


def _require_private_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TargetTransferUploadError(f"{label} is unavailable") from exc
    _require_private_directory_metadata(metadata, label=label)


def _require_private_directory_metadata(metadata: os.stat_result, *, label: str) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & _PRIVATE_BITS
    ):
        raise TargetTransferUploadError(f"{label} must be an owned private directory")


def _canonical_request_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("target transfer request identity must be a canonical UUID4") from exc
    canonical = str(parsed)
    if parsed.version != 4 or canonical != value:
        raise ValueError("target transfer request identity must be a canonical UUID4")
    return value


def _canonical_digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError("target transfer digest must be lowercase SHA-256")
    return value


def _positive_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("target transfer size must be a positive integer")
    return value


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short target upload write")
        remaining = remaining[written:]


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise TargetTransferUploadError("target transfer directory cannot be synced") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise TargetTransferUploadError("target transfer directory cannot be synced") from exc
    finally:
        os.close(descriptor)


def _stat_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        stat.S_IMODE(metadata.st_mode),
    )


__all__ = [
    "TargetTransferActivationCoordinator",
    "TargetTransferUploadBusy",
    "TargetTransferUploadControl",
    "TargetTransferUploadCoordinator",
    "TargetTransferUploadError",
    "TargetTransferUploadLease",
    "TargetTransferUploadReceipt",
    "acquire_target_transfer_upload_lease",
    "discard_target_transfer_partial",
    "prepare_transfer_import_command",
    "replay_target_transfer_archive",
    "target_transfer_archive_path",
    "target_transfer_partial_path",
    "target_transfer_upload_lease",
    "upload_target_transfer_archive",
    "verify_target_transfer_archive",
]
