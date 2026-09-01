"""Source fencing, exact capture, and safe project-transfer archive bytes."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tarfile
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO

from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef
from rcp.limits import (
    PROJECT_TRANSFER_COPY_BUFFER_BYTES,
    PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES,
    PROJECT_TRANSFER_MANIFEST_MAX_BYTES,
)
from rcp.project_transfer import capture_project_transfer_source
from rcp.transfer.archive import (
    TransferArchiveActor,
    TransferArchiveAttribution,
    TransferArchiveEntry,
    TransferArchiveEnvelope,
    TransferArchiveManifest,
    TransferGraphHead,
)
from rcp.transfer.project_files import (
    TRANSFER_OPERATIONAL_RECORDS_PATH,
    capture_project_transfer_files,
    transfer_project_file_payload,
)
from rcp.transfer.provider_history import capture_provider_history
from rcp.transport import LocalStateWorkspace

if TYPE_CHECKING:
    from rcp.projects import ProjectCatalog
    from rcp.server_ops.backup_models import BackupCanonicalSourcePlan
    from rcp.service import ProjectService
    from rcp.storage import (
        AppStore,
        ProjectTransferCleanupAcknowledgment,
        ProjectTransferRequestRecord,
    )
    from rcp.transfer.records import TransferRecordBundle

TRANSFER_ARCHIVE_MANIFEST_PATH = "manifest.json"

_ARCHIVE_MODE = 0o600
_CAPTURE_DIRECTORY_MODE = 0o700
_CAPTURE_FILE_MODE = 0o400
_PRIVATE_MODE_MASK = 0o077
_TAR_BLOCK_SIZE = 512


@dataclass(frozen=True)
class SealedTransferArchive:
    """The path and exact-byte receipt produced by :func:`seal_transfer_archive`."""

    archive_path: Path
    envelope: TransferArchiveEnvelope

    @property
    def manifest_sha256(self) -> str:
        return self.envelope.manifest_sha256

    @property
    def archive_sha256(self) -> str:
        return self.envelope.archive_sha256


@dataclass(frozen=True)
class TransferArchiveReadback:
    """The manifest and exact-byte receipt read from one sealed archive."""

    archive_path: Path
    manifest: TransferArchiveManifest
    envelope: TransferArchiveEnvelope


def source_transfer_export_path(data_dir: Path, request_id: str) -> Path:
    """Return the sole request-derived source recovery archive path."""

    try:
        parsed = uuid.UUID(request_id)
        canonical = str(parsed)
    except (AttributeError, ValueError) as exc:
        raise ValueError("transfer request identity must be a canonical UUID") from exc
    if parsed.version != 4 or canonical != request_id:
        raise ValueError("transfer request identity must be a canonical UUID4")
    return Path(data_dir) / "transfer-exports" / f"{canonical}.rcp-transfer"


def advance_source_project_transfer(
    store: AppStore,
    catalog: ProjectCatalog,
    request_id: str,
) -> ProjectTransferRequestRecord:
    """Settle, fence, capture, seal, and bind one confirmed source transfer."""

    transfer = _source_transfer(store, request_id)
    if transfer.source_release_receipt is None or transfer.target_admission_receipt is None:
        raise ValueError("source transfer requires both human confirmations")
    if transfer.phase == "source_released":
        service = catalog.open_transfer_source(request_id)
        _require_reviewed_source_unchanged(service, transfer)
        attributions = _source_attributions(transfer)
        # This exact projection is the source-of-truth settlement check. It
        # refuses every live task, episode, watcher, report, child, or delivery.
        store.export_project_transfer_records(
            transfer.project_id,
            attributions=attributions,
        )
        _require_reviewed_source_unchanged(service, transfer)
        materialized = service.history.current_materialization()
        identity = service.history.project_identity(materialized)
        if identity is None or identity.project_id != transfer.project_id:
            raise ValueError("source transfer lost its canonical project identity")
        if identity.home_space_id == transfer.source_space_id:
            service.history.transfer_project_home(
                project_id=transfer.project_id,
                previous_home_space_id=transfer.source_space_id,
                new_home_space_id=transfer.target_space_id,
                source_released_by=transfer.source_release_receipt.released_by,
                target_admitted_by=transfer.target_admission_receipt.admitted_by,
            )
        elif identity.home_space_id != transfer.target_space_id:
            raise ValueError("source project moved to an unrelated canonical home")
        fenced_head = service.history.head_ref()
        transfer = store.mark_source_project_transfer_fenced(
            request_id,
            source_head=fenced_head,
        )

    destination = source_transfer_export_path(catalog.data_dir, request_id)
    _prepare_export_directory(destination.parent)
    if transfer.phase == "source_fenced":
        if destination.exists() or destination.is_symlink():
            readback = read_transfer_archive(destination)
        else:
            service = catalog.open_transfer_source(request_id)
            _require_reviewed_source_unchanged(service, transfer)
            attributions = _source_attributions(transfer)
            records = store.export_project_transfer_records(
                transfer.project_id,
                attributions=attributions,
            )
            proof = store.expose_project_transfer_proof(request_id)
            readback = _capture_and_seal_source_archive(
                store=store,
                service=service,
                transfer=transfer,
                records=records,
                attributions=attributions,
                proof=proof,
                destination=destination,
            )
        _require_readback_matches_transfer(readback, transfer)
        transfer = store.bind_project_transfer_archive(
            request_id,
            archive_sha256=readback.envelope.archive_sha256,
            archive_size_bytes=readback.envelope.archive_size_bytes,
            source_fence_head=transfer.source_fence_head,
        )

    if transfer.phase != "archive_bound":
        raise ValueError("source transfer is not ready to expose its sealed archive")
    readback = read_transfer_archive(destination)
    _require_readback_matches_transfer(readback, transfer)
    if (
        transfer.archive_sha256 != readback.envelope.archive_sha256
        or transfer.archive_size_bytes != readback.envelope.archive_size_bytes
    ):
        raise ValueError("sealed source archive differs from its durable receipt")
    return transfer


def complete_source_project_transfer(
    store: AppStore,
    catalog: ProjectCatalog,
    request_id: str,
    *,
    target_activation_proof: bytes,
) -> ProjectTransferCleanupAcknowledgment:
    """Verify target activation, retire the source, and erase only bound recovery bytes."""

    acknowledgment = store.verify_target_project_transfer_activation(
        request_id,
        proof=target_activation_proof,
    )
    transfer = store.acknowledge_project_transfer_cleanup(request_id)
    destination = source_transfer_export_path(catalog.data_dir, request_id)
    if os.path.lexists(destination):
        _require_cleanup_archive_matches_transfer(destination, transfer)
    else:
        retired = store.retired_project(transfer.project_id)
        if (
            transfer.proof_state != "consumed"
            or retired is None
            or retired.retired_transfer_request_id != request_id
        ):
            raise ValueError("source recovery archive disappeared before retirement")
    assert transfer.proof_acknowledgement_sha256 is not None
    transfer = store.consume_project_transfer_proof(
        request_id,
        acknowledgement_sha256=transfer.proof_acknowledgement_sha256,
    )
    store.retire_source_project_transfer(request_id)
    catalog.discard_retired_transfer_source(request_id)
    if os.path.lexists(destination):
        _require_cleanup_archive_matches_transfer(destination, transfer)
        destination.unlink()
        _fsync_directory(destination.parent)
    store.complete_project_transfer_request(request_id)
    return acknowledgment


def stream_source_transfer_archive(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Iterator[bytes]:
    """Yield the verified sealed archive in bounded chunks from one open descriptor."""

    archive_path = Path(archive_path)
    _validate_destination_parent(archive_path.parent)
    descriptor, before = _open_sealed_archive(archive_path)
    try:
        digest, size = _hash_descriptor(descriptor)
        if (digest, size) != (expected_sha256, expected_size_bytes):
            raise ValueError("sealed source archive differs from its durable receipt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            yield chunk
        _require_unchanged(before, os.fstat(descriptor), label="sealed source archive")
    finally:
        os.close(descriptor)


def _source_transfer(store: AppStore, request_id: str) -> ProjectTransferRequestRecord:
    transfer = store.project_transfer_request(request_id)
    if transfer is None or transfer.side != "source":
        raise KeyError(request_id)
    return transfer


def _source_attributions(
    transfer: ProjectTransferRequestRecord,
) -> tuple[TransferArchiveAttribution, ...]:
    release = transfer.source_release_receipt
    admission = transfer.target_admission_receipt
    if release is None or admission is None:
        raise ValueError("source transfer has incomplete human attribution")
    actors = {
        (release.released_by.space_id, release.released_by.user_id): release.released_by,
        (admission.admitted_by.space_id, admission.admitted_by.user_id): admission.admitted_by,
    }
    return tuple(
        sorted(
            (
                TransferArchiveAttribution(
                    archive_actor_id=_archive_actor_id(transfer.request_id, actor),
                    source_actor=TransferArchiveActor.capture(actor),
                )
                for actor in actors.values()
            ),
            key=lambda item: item.archive_actor_id,
        )
    )


def _archive_actor_id(request_id: str, actor: AuthorizedHuman) -> str:
    payload = f"{request_id}\0{actor.space_id}\0{actor.user_id}".encode()
    raw = bytearray(hashlib.sha256(payload).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def _capture_and_seal_source_archive(
    *,
    store: AppStore,
    service: ProjectService,
    transfer: ProjectTransferRequestRecord,
    records: TransferRecordBundle,
    attributions: tuple[TransferArchiveAttribution, ...],
    proof: bytes,
    destination: Path,
) -> TransferArchiveReadback:
    _prepare_export_directory(destination.parent)
    with tempfile.TemporaryDirectory(
        prefix=f".{transfer.request_id}.capture-",
        dir=destination.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        temporary_root.chmod(0o700)
        final_root = temporary_root / "archive"
        final_root.mkdir(mode=0o700)
        project_root = temporary_root / "project"
        workspace = service.history.workspace
        with workspace.transaction():
            project_capture = capture_project_transfer_files(service, records, project_root)
            canonical_entries, canonical_plan = _capture_canonical_history(
                service,
                final_root,
            )
            main_head = service.history.head_ref()
        provider_root = temporary_root / "provider"
        provider_capture = capture_provider_history(service.indexer, provider_root)

        entries = [
            _copy_capture_entry(project_root, final_root, entry)
            for entry in project_capture.entries
        ]
        entries.extend(
            _copy_capture_entry(provider_root, final_root, entry)
            for entry in provider_capture.entries
        )

        operational_payload = transfer_project_file_payload(project_capture)
        entries.append(
            _write_capture_bytes(
                final_root,
                TRANSFER_OPERATIONAL_RECORDS_PATH,
                "operational_records",
                operational_payload,
            )
        )
        entries.extend(canonical_entries)
        entries.append(
            _write_capture_bytes(
                final_root,
                "control/source-release-proof.bin",
                "source_release_proof",
                proof,
            )
        )
        ordered = tuple(sorted(entries, key=lambda item: item.archive_path))
        source_manifest = next(
            entry for entry in ordered if entry.group == "source_manifest_provenance"
        )
        if canonical_plan.main_observed_revision != main_head.revision:
            raise ValueError("source canonical capture moved after its fence")
        branch_heads = tuple(
            TransferGraphHead.capture(branch.head)
            for branch in sorted(canonical_plan.branches, key=lambda item: item.branch_id)
        )
        configuration = transfer.source_configuration
        if (
            configuration is None
            or transfer.linked_request_id is None
            or transfer.target_activation_proof_sha256 is None
        ):
            raise ValueError("source transfer lost its reviewed linked boundary")
        if source_manifest.sha256 != configuration.source_manifest_sha256:
            raise ValueError("source manifest changed after the reviewed transfer boundary")
        manifest = TransferArchiveManifest(
            project_id=transfer.project_id,
            source_space_id=transfer.source_space_id,
            target_space_id=transfer.target_space_id,
            source_request_id=transfer.request_id,
            target_request_id=transfer.linked_request_id,
            source_rcp_version=configuration.source_rcp_version,
            source_schema_generation=configuration.source_schema_generation,
            source_configuration_sha256=transfer.source_configuration_sha256,
            source_manifest_sha256=source_manifest.sha256,
            source_release_proof_sha256=transfer.source_release_proof_sha256,
            target_activation_proof_sha256=transfer.target_activation_proof_sha256,
            main_head=TransferGraphHead.capture(main_head),
            branch_heads=branch_heads,
            attributions=attributions,
            diagnostics=provider_capture.diagnostics,
            entries=ordered,
            payload_size_bytes=sum(entry.size_bytes for entry in ordered),
            created_at=datetime.fromisoformat(store.now()),
        )
        sealed = seal_transfer_archive(
            manifest=manifest,
            capture_root=final_root,
            destination=destination,
        )
    return read_transfer_archive(destination, expected_envelope=sealed.envelope)


def _capture_canonical_history(
    service: ProjectService,
    destination_root: Path,
) -> tuple[list[TransferArchiveEntry], BackupCanonicalSourcePlan]:
    workspace = service.history.workspace
    with tempfile.TemporaryDirectory(prefix="rcp-transfer-canonical-") as temporary:
        export_root = Path(temporary)
        export_root.chmod(0o700)
        source_root = workspace.backup_source_root(export_root)
        plan = LocalStateWorkspace(source_root, str(source_root)).backup_canonical_source_plan()
        if plan.unclassified_roots:
            raise ValueError("source canonical history has unclassified roots")
        files = list(plan.main_files)
        files.extend(item for branch in plan.branches for item in branch.files)
        captured: list[TransferArchiveEntry] = []
        for item in files:
            source = source_root.joinpath(*PurePosixPath(item.relative_path).parts)
            if item.kind == "manifest":
                archive_path = "provenance/manifest.toml"
                group = "source_manifest_provenance"
            else:
                archive_path = f"canonical/{item.relative_path}"
                group = "canonical_history"
            captured.append(
                _copy_regular_file(
                    source,
                    destination_root,
                    archive_path,
                    group,
                    expected_size=item.observed_size_bytes,
                )
            )
        return captured, plan


def _require_reviewed_source_unchanged(
    service: ProjectService,
    transfer: ProjectTransferRequestRecord,
) -> None:
    current, _head = capture_project_transfer_source(service)
    if transfer.source_configuration is None or current != transfer.source_configuration:
        raise ValueError("source configuration changed after the reviewed transfer boundary")


def _copy_capture_entry(
    source_root: Path,
    destination_root: Path,
    entry: TransferArchiveEntry,
) -> TransferArchiveEntry:
    source = source_root.joinpath(*PurePosixPath(entry.archive_path).parts)
    copied = _copy_regular_file(
        source,
        destination_root,
        entry.archive_path,
        entry.group,
        expected_size=entry.size_bytes,
    )
    if copied != entry:
        raise ValueError("captured transfer file differs from its declared entry")
    return copied


def _copy_regular_file(
    source: Path,
    destination_root: Path,
    archive_path: str,
    group: str,
    *,
    expected_size: int,
) -> TransferArchiveEntry:
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination = destination_root.joinpath(*PurePosixPath(archive_path).parts)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        source_metadata = os.fstat(source_descriptor)
        source_path_metadata = source.lstat()
        if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_size != expected_size:
            raise ValueError("source transfer file is unsafe or changed")
        if _stat_fingerprint(source_metadata) != _stat_fingerprint(source_path_metadata):
            raise ValueError("source transfer file is unsafe or changed")
        while True:
            chunk = os.read(source_descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            _write_all(descriptor, chunk)
            digest.update(chunk)
            size += len(chunk)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        final_source_metadata = os.fstat(source_descriptor)
        final_source_path_metadata = source.lstat()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
        os.close(descriptor)
    if (
        size != expected_size
        or _stat_fingerprint(source_metadata) != _stat_fingerprint(final_source_metadata)
        or _stat_fingerprint(final_source_metadata) != _stat_fingerprint(final_source_path_metadata)
    ):
        destination.unlink(missing_ok=True)
        raise ValueError("source transfer file changed during capture")
    return TransferArchiveEntry(
        archive_path=archive_path,
        group=group,
        sha256=digest.hexdigest(),
        size_bytes=size,
    )


def _write_capture_bytes(
    root: Path,
    archive_path: str,
    group: str,
    payload: bytes,
) -> TransferArchiveEntry:
    destination = root.joinpath(*PurePosixPath(archive_path).parts)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return TransferArchiveEntry(
        archive_path=archive_path,
        group=group,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _require_readback_matches_transfer(
    readback: TransferArchiveReadback,
    transfer: ProjectTransferRequestRecord,
) -> None:
    manifest = readback.manifest
    if (
        manifest.project_id != transfer.project_id
        or manifest.source_request_id != transfer.request_id
        or manifest.target_request_id != transfer.linked_request_id
        or manifest.source_space_id != transfer.source_space_id
        or manifest.target_space_id != transfer.target_space_id
        or transfer.source_fence_head is None
        or GraphHeadRef.model_validate_json(manifest.main_head.model_dump_json())
        != transfer.source_fence_head
        or manifest.source_configuration_sha256 != transfer.source_configuration_sha256
        or manifest.source_release_proof_sha256 != transfer.source_release_proof_sha256
        or manifest.target_activation_proof_sha256 != transfer.target_activation_proof_sha256
    ):
        raise ValueError("sealed source archive does not match its transfer receipt")


def _require_cleanup_archive_matches_transfer(
    archive_path: Path,
    transfer: ProjectTransferRequestRecord,
) -> None:
    readback = read_transfer_archive(archive_path)
    _require_readback_matches_transfer(readback, transfer)
    if (
        readback.envelope.archive_sha256 != transfer.archive_sha256
        or readback.envelope.archive_size_bytes != transfer.archive_size_bytes
    ):
        raise ValueError("source cleanup archive differs from its durable receipt")


def _prepare_export_directory(path: Path) -> None:
    try:
        path.mkdir(mode=_CAPTURE_DIRECTORY_MODE, exist_ok=True)
    except OSError as exc:
        raise ValueError("source transfer export directory could not be created privately") from exc
    _validate_destination_parent(path)


def seal_transfer_archive(
    *,
    manifest: TransferArchiveManifest,
    capture_root: Path,
    destination: Path,
) -> SealedTransferArchive:
    """Atomically publish one deterministic mode-0600 transfer archive.

    ``capture_root`` must be a new private capture tree whose regular files
    exactly match ``manifest.entries``.  The destination is never replaced:
    an existing path is a hard failure, so a caller with a durable receipt
    cannot accidentally regenerate the archive from a changed capture.
    """

    manifest = TransferArchiveManifest.model_validate(manifest)
    capture_root = Path(capture_root)
    destination = Path(destination)
    manifest_bytes = manifest.canonical_bytes()
    expected_files = _validate_capture_tree(capture_root, manifest.entries)
    _validate_destination_parent(destination.parent)
    _require_destination_absent(destination)

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            _ARCHIVE_MODE,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            _write_deterministic_archive(
                stream,
                manifest,
                capture_root,
                expected_files,
                manifest_bytes,
            )
            # Recheck the inventory after streaming.  A capture writer adding
            # an undeclared file during the read must not silently leave it
            # outside the source-side receipt.
            if _validate_capture_tree(capture_root, manifest.entries) != expected_files:
                raise ValueError("transfer capture changed during archive sealing")
            stream.flush()
            os.fchmod(descriptor, _ARCHIVE_MODE)
            os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        # link(2) is the no-overwrite atomic publication primitive available
        # for a regular file.  A rename would silently replace a receipt.
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"transfer archive destination already exists: {destination}"
            ) from exc
        temporary.unlink()
        _fsync_directory(destination.parent)
        readback = read_transfer_archive(destination)
        return SealedTransferArchive(destination, readback.envelope)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def read_transfer_archive(
    archive_path: Path,
    *,
    expected_envelope: TransferArchiveEnvelope | None = None,
) -> TransferArchiveReadback:
    """Read and rehash one sealed archive, rejecting any byte or type drift.

    The tar is consumed as a stream after the outer file is checked for exact
    mode-0600 regular-file safety.  Members must be exactly the fixed manifest
    followed by the manifest's sorted entries.  No extraction is performed.
    """

    path = Path(archive_path)
    descriptor, before = _open_sealed_archive(path)
    try:
        archive_sha256, archive_size = _hash_descriptor(descriptor)
        if expected_envelope is not None and (
            archive_sha256 != expected_envelope.archive_sha256
            or archive_size != expected_envelope.archive_size_bytes
        ):
            raise ValueError("sealed transfer archive bytes differ from its receipt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        manifest = _read_tar(descriptor, staging_root=None)
        after = os.fstat(descriptor)
        _require_unchanged(before, after, label="sealed transfer archive")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError("sealed transfer archive is missing, malformed, or unreadable") from exc
    finally:
        os.close(descriptor)

    envelope = expected_envelope or TransferArchiveEnvelope.bind(
        manifest,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size,
    )
    if envelope.archive_codec != manifest.archive_codec:
        raise ValueError("transfer archive receipt names another codec")
    envelope.verify_manifest(manifest)
    return TransferArchiveReadback(path, manifest, envelope)


def stage_transfer_archive(
    archive_path: Path,
    staging_root: Path,
    *,
    expected_envelope: TransferArchiveEnvelope | None = None,
) -> TransferArchiveReadback:
    """Verify and stream one archive into a fresh private staging tree.

    Staging is request-scoped and disposable.  A failed decode removes only
    the newly-created staging root; the sealed source archive is untouched.
    """

    path = Path(archive_path)
    root = Path(staging_root)
    _validate_stage_parent(root.parent)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"transfer staging root already exists: {root}")
    try:
        root.mkdir(mode=_CAPTURE_DIRECTORY_MODE)
    except OSError as exc:
        raise ValueError("transfer staging root could not be created privately") from exc

    descriptor = -1
    completed = False
    try:
        descriptor, before = _open_sealed_archive(path)
        archive_sha256, archive_size = _hash_descriptor(descriptor)
        if expected_envelope is not None and (
            archive_sha256 != expected_envelope.archive_sha256
            or archive_size != expected_envelope.archive_size_bytes
        ):
            raise ValueError("sealed transfer archive bytes differ from its receipt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        manifest = _read_tar(descriptor, staging_root=root)
        after = os.fstat(descriptor)
        _require_unchanged(before, after, label="sealed transfer archive")
        envelope = expected_envelope or TransferArchiveEnvelope.bind(
            manifest,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size,
        )
        if envelope.archive_codec != manifest.archive_codec:
            raise ValueError("transfer archive receipt names another codec")
        envelope.verify_manifest(manifest)
        _fsync_tree(root)
        completed = True
        return TransferArchiveReadback(path, manifest, envelope)
    except (OSError, tarfile.TarError) as exc:
        raise ValueError("transfer archive could not be safely staged") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not completed and (root.exists() or root.is_symlink()):
            _discard_new_stage(root)


def discard_transfer_archive_stage(staging_root: Path) -> None:
    """Remove one exact caller-owned decode tree after validating its parent."""

    root = Path(staging_root)
    _validate_stage_parent(root.parent)
    _discard_new_stage(root)


def _write_deterministic_archive(
    stream: BinaryIO,
    manifest: TransferArchiveManifest,
    capture_root: Path,
    expected_files: dict[str, Path],
    manifest_bytes: bytes,
) -> None:
    with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        _add_bytes_to_tar(archive, TRANSFER_ARCHIVE_MANIFEST_PATH, manifest_bytes)
        for entry in manifest.entries:
            _add_verified_file_to_tar(archive, expected_files[entry.archive_path], entry)


def _add_bytes_to_tar(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))


def _add_verified_file_to_tar(
    archive: tarfile.TarFile,
    source: Path,
    entry: TransferArchiveEntry,
) -> None:
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        path_before = source.lstat()
        _require_private_regular_pair(before, path_before, label=entry.archive_path)
        if before.st_size != entry.size_bytes:
            raise ValueError(f"capture file size differs from its manifest: {entry.archive_path}")
        reader = _DigestingReader(descriptor, expected_size=entry.size_bytes)
        archive.addfile(_tar_info(entry.archive_path, entry.size_bytes), reader)
        after = os.fstat(descriptor)
        path_after = source.lstat()
        _require_private_regular_pair(after, path_after, label=entry.archive_path)
        if (
            reader.size != entry.size_bytes
            or reader.digest.hexdigest() != entry.sha256
            or before.st_size != after.st_size
            or _stat_fingerprint(before) != _stat_fingerprint(after)
            or _stat_fingerprint(after) != _stat_fingerprint(path_after)
        ):
            raise ValueError(f"capture file changed or is corrupt: {entry.archive_path}")
    finally:
        os.close(descriptor)


class _DigestingReader:
    def __init__(self, descriptor: int, *, expected_size: int) -> None:
        self.descriptor = descriptor
        self.expected_size = expected_size
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.expected_size - self.size
        if remaining <= 0:
            return b""
        amount = (
            PROJECT_TRANSFER_COPY_BUFFER_BYTES
            if size < 0
            else min(size, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
        )
        data = os.read(self.descriptor, min(amount, remaining))
        self.digest.update(data)
        self.size += len(data)
        return data


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = _CAPTURE_FILE_MODE
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _validate_capture_tree(
    capture_root: Path,
    entries: tuple[TransferArchiveEntry, ...],
) -> dict[str, Path]:
    _validate_private_directory(capture_root, label="transfer capture")
    expected = {
        entry.archive_path: capture_root.joinpath(*PurePosixPath(entry.archive_path).parts)
        for entry in entries
    }
    if TRANSFER_ARCHIVE_MANIFEST_PATH in expected:
        raise ValueError("transfer manifest path is reserved")
    if len(expected) > PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES:
        raise ValueError("transfer capture exceeds its entry bound")

    actual_files: dict[str, Path] = {}
    actual_directories: set[str] = set()
    _walk_private_tree(capture_root, capture_root, actual_files, actual_directories)
    expected_directories = {
        "/".join(PurePosixPath(path).parts[:index])
        for path in expected
        for index in range(1, len(PurePosixPath(path).parts))
    }
    if set(actual_files) != set(expected):
        missing = sorted(set(expected) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected))
        detail = f"missing={missing[:3]} extra={extra[:3]}"
        raise ValueError(f"transfer capture inventory differs from manifest: {detail}")
    if actual_directories != expected_directories:
        raise ValueError("transfer capture contains an undeclared directory")
    return expected


def _walk_private_tree(
    root: Path,
    current: Path,
    files: dict[str, Path],
    directories: set[str],
) -> None:
    try:
        children = sorted(current.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ValueError("transfer capture could not be enumerated") from exc
    for child in children:
        try:
            metadata = child.lstat()
        except OSError as exc:
            raise ValueError("transfer capture entry could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"transfer capture refuses symlink: {child.name}")
        if stat.S_ISDIR(metadata.st_mode):
            _require_capture_directory_metadata(metadata, label=str(child))
            directories.add(child.relative_to(root).as_posix())
            _walk_private_tree(root, child, files, directories)
        elif stat.S_ISREG(metadata.st_mode):
            _require_private_file_metadata(metadata, label=str(child))
            relative = child.relative_to(root).as_posix()
            if PurePosixPath(relative) in {
                PurePosixPath("."),
                PurePosixPath(""),
            }:
                raise ValueError("transfer capture contains an invalid file name")
            files[relative] = child
        else:
            raise ValueError(f"transfer capture refuses special file: {child.name}")


def _read_tar(descriptor: int, *, staging_root: Path | None) -> TransferArchiveManifest:
    stream = os.fdopen(os.dup(descriptor), "rb")
    try:
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            members = iter(archive)
            first = next(members, None)
            if first is None or first.name != TRANSFER_ARCHIVE_MANIFEST_PATH:
                raise ValueError("transfer archive is missing its reserved manifest")
            _validate_tar_member(first, expected_name=TRANSFER_ARCHIVE_MANIFEST_PATH)
            manifest_bytes = _read_tar_member(archive, first)
            if len(manifest_bytes) > PROJECT_TRANSFER_MANIFEST_MAX_BYTES:
                raise ValueError("transfer archive manifest exceeds its byte bound")
            try:
                manifest = TransferArchiveManifest.model_validate_json(manifest_bytes)
            except ValueError as exc:
                raise ValueError("transfer archive manifest is invalid") from exc
            if manifest.canonical_bytes() != manifest_bytes:
                raise ValueError("transfer archive manifest is not canonical")
            if len(manifest.entries) > PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES:
                raise ValueError("transfer archive exceeds its entry bound")

            for entry in manifest.entries:
                member = next(members, None)
                if member is None or member.name != entry.archive_path:
                    raise ValueError("transfer archive members do not match its manifest")
                _validate_tar_member(member, expected_name=entry.archive_path, entry=entry)
                _stream_tar_member(
                    archive,
                    member,
                    entry,
                    staging_root=staging_root,
                )
            if next(members, None) is not None:
                raise ValueError("transfer archive contains undeclared members")
            return manifest
    finally:
        stream.close()


def _validate_tar_member(
    member: tarfile.TarInfo,
    *,
    expected_name: str,
    entry: TransferArchiveEntry | None = None,
) -> None:
    if (
        member.name != expected_name
        or member.type != tarfile.REGTYPE
        or not member.isfile()
        or member.uid != 0
        or member.gid != 0
        or member.uname != ""
        or member.gname != ""
        or member.mtime != 0
        or stat.S_IMODE(member.mode) != _CAPTURE_FILE_MODE
    ):
        raise ValueError(f"transfer archive member has unsafe metadata: {expected_name}")
    if set(member.pax_headers) - {"path", "size"}:
        raise ValueError(f"transfer archive member has unsupported PAX metadata: {expected_name}")
    if member.pax_headers.get("path", expected_name) != expected_name:
        raise ValueError(f"transfer archive member has a mismatched PAX path: {expected_name}")
    if "size" in member.pax_headers:
        try:
            if int(member.pax_headers["size"]) != member.size:
                raise ValueError(
                    f"transfer archive member has a mismatched PAX size: {expected_name}"
                )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"transfer archive member has an invalid PAX size: {expected_name}"
            ) from exc
    if entry is not None and member.size != entry.size_bytes:
        raise ValueError(f"transfer archive member size differs from its manifest: {expected_name}")


def _read_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if member.size > PROJECT_TRANSFER_MANIFEST_MAX_BYTES:
        raise ValueError("transfer archive manifest exceeds its byte bound")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"transfer archive member cannot be read: {member.name}")
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = source.read(PROJECT_TRANSFER_COPY_BUFFER_BYTES)
        if not chunk:
            break
        size += len(chunk)
        if size > PROJECT_TRANSFER_MANIFEST_MAX_BYTES:
            raise ValueError("transfer archive manifest exceeds its byte bound")
        chunks.append(chunk)
    if size != member.size:
        raise ValueError(f"transfer archive member ended before its declared size: {member.name}")
    return b"".join(chunks)


def _stream_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    entry: TransferArchiveEntry,
    *,
    staging_root: Path | None,
) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"transfer archive member cannot be read: {entry.archive_path}")
    destination: Path | None = None
    temporary: Path | None = None
    descriptor = -1
    if staging_root is not None:
        destination = staging_root.joinpath(*PurePosixPath(entry.archive_path).parts)
        _make_private_stage_directories(staging_root, destination.parent)
        _validate_private_directory(destination.parent, label="transfer staging directory")
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            _CAPTURE_FILE_MODE,
        )

    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = source.read(PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > entry.size_bytes:
                raise ValueError(
                    f"transfer archive member exceeds its manifest size: {entry.archive_path}"
                )
            digest.update(chunk)
            if descriptor >= 0:
                _write_all(descriptor, chunk)
        if size != entry.size_bytes or digest.hexdigest() != entry.sha256:
            raise ValueError(f"transfer archive member is corrupt: {entry.archive_path}")
        if descriptor >= 0:
            os.fchmod(descriptor, _CAPTURE_FILE_MODE)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            assert temporary is not None and destination is not None
            os.link(temporary, destination, follow_symlinks=False)
            temporary.unlink()
            _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_staged_member(root: Path, name: str, payload: bytes) -> None:
    path = root.joinpath(*PurePosixPath(name).parts)
    _make_private_stage_directories(root, path.parent)
    _validate_private_directory(path.parent, label="transfer staging directory")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        _CAPTURE_FILE_MODE,
    )
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, _CAPTURE_FILE_MODE)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _make_private_stage_directories(root: Path, destination: Path) -> None:
    destination.mkdir(mode=_CAPTURE_DIRECTORY_MODE, parents=True, exist_ok=True)
    current = destination
    while True:
        metadata = current.lstat()
        _require_capture_directory_metadata(metadata, label=str(current))
        current.chmod(_CAPTURE_DIRECTORY_MODE)
        if current == root:
            return
        try:
            current = current.parent
            current.relative_to(root)
        except ValueError as exc:
            raise ValueError("transfer staging path escaped its private root") from exc


def _open_sealed_archive(path: Path) -> tuple[int, os.stat_result]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"sealed transfer archive is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("sealed transfer archive refuses symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("sealed transfer archive refuses special file")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != _ARCHIVE_MODE:
        raise ValueError("sealed transfer archive has unsafe ownership or mode")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"sealed transfer archive is unavailable: {path}") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != _ARCHIVE_MODE
        or _stat_fingerprint(opened) != _stat_fingerprint(metadata)
    ):
        os.close(descriptor)
        raise ValueError("sealed transfer archive changed or has unsafe mode")
    return descriptor, metadata


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
        if not chunk:
            return digest.hexdigest(), size
        digest.update(chunk)
        size += len(chunk)


def _require_destination_absent(destination: Path) -> None:
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("transfer archive destination could not be inspected") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("transfer archive destination refuses symlink")
    raise FileExistsError(f"transfer archive destination already exists: {destination}")


def _validate_destination_parent(parent: Path) -> None:
    _validate_private_directory(parent, label="transfer archive destination")


def _validate_stage_parent(parent: Path) -> None:
    _validate_private_directory(parent, label="transfer staging parent")


def _validate_private_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} directory is unavailable") from exc
    _require_private_directory_metadata(metadata, label=label)


def _require_private_directory_metadata(metadata: os.stat_result, *, label: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & _PRIVATE_MODE_MASK:
        raise ValueError(f"{label} must be owned by the current user and private")


def _require_capture_directory_metadata(metadata: os.stat_result, *, label: str) -> None:
    """The private capture root protects descendants; retain owner/type checks below it.

    Existing capture owners create intermediate directories with the platform
    default mode while the root itself is mode 0700.  Descendant permissions
    therefore do not expand access through the private root, but symlinks and
    foreign-owned directories remain unsafe.
    """

    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise ValueError(f"{label} must be an owned directory")


def _require_private_file_metadata(metadata: os.stat_result, *, label: str) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & _PRIVATE_MODE_MASK
    ):
        raise ValueError(f"{label} must be a private regular file")


def _require_private_regular_pair(
    descriptor_metadata: os.stat_result,
    path_metadata: os.stat_result,
    *,
    label: str,
) -> None:
    _require_private_file_metadata(descriptor_metadata, label=label)
    if _stat_fingerprint(descriptor_metadata) != _stat_fingerprint(path_metadata):
        raise ValueError(f"{label} changed while being opened")


def _stat_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        stat.S_IMODE(metadata.st_mode),
    )


def _require_unchanged(before: os.stat_result, after: os.stat_result, *, label: str) -> None:
    if _stat_fingerprint(before) != _stat_fingerprint(after):
        raise ValueError(f"{label} changed during readback")


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short transfer archive write")
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


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in (*directories, root):
        _fsync_directory(directory)


def _discard_new_stage(root: Path) -> None:
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return
    _require_private_directory_metadata(metadata, label="transfer staging root")
    shutil.rmtree(root)


__all__ = [
    "TRANSFER_ARCHIVE_MANIFEST_PATH",
    "SealedTransferArchive",
    "TransferArchiveReadback",
    "advance_source_project_transfer",
    "complete_source_project_transfer",
    "discard_transfer_archive_stage",
    "read_transfer_archive",
    "seal_transfer_archive",
    "source_transfer_export_path",
    "stage_transfer_archive",
    "stream_source_transfer_archive",
]
