"""Validated, resumable target import for one project-transfer archive."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import tomlkit

from rcp.config import Manifest
from rcp.core.transition_models import GraphHeadRef
from rcp.history import HistoryManager
from rcp.limits import PROJECT_TRANSFER_COPY_BUFFER_BYTES
from rcp.paper import PaperService
from rcp.service import ProjectService
from rcp.sources import ImportedProviderSourceInventory
from rcp.storage import ProjectTransferImportRecord
from rcp.transfer.archive import (
    TransferArchiveEntry,
    TransferArchiveEnvelope,
    TransferArchiveManifest,
)
from rcp.transfer.configuration import TransferTargetConfiguration
from rcp.transfer.project_files import (
    TRANSFER_OPERATIONAL_RECORDS_PATH,
    TransferProjectFileCapture,
    parse_transfer_project_file_payload,
)
from rcp.transport import LocalStateWorkspace
from rcp.transport.state import StateWorkspace, state_workspace_for_probe

if TYPE_CHECKING:
    from rcp.projects import ProjectCatalog


_PROJECT_FILE_GROUPS = frozenset(
    {
        "rcp_chat",
        "paper_introduction",
        "fact",
        "kept_artifact",
        "legacy_kept_result_view",
    }
)


@dataclass(frozen=True)
class _TargetOwners:
    manifest: Manifest
    workspace: StateWorkspace
    history: HistoryManager
    paper: PaperService
    service: ProjectService


def import_project_transfer(
    catalog: ProjectCatalog,
    *,
    archive: TransferArchiveManifest,
    envelope: TransferArchiveEnvelope,
    archive_root: Path,
    target_configuration: TransferTargetConfiguration,
) -> ProjectTransferImportRecord:
    """Import one validated archive without registering or activating its project."""

    store = catalog.store
    transfer = store.project_transfer_request(archive.target_request_id)
    provisioning = store.project_provisioning_request(archive.target_request_id)
    _validate_protocol(
        archive,
        envelope,
        target_configuration,
        transfer=transfer,
        provisioning=provisioning,
        target_space_id=store.space_id,
    )
    _validate_archive_root(archive_root, archive.entries)
    operational_entry = _operational_entry(archive.entries)
    operational_payload = _read_entry_bytes(archive_root, operational_entry)
    capture = parse_transfer_project_file_payload(operational_payload)
    _validate_capture(archive, capture)
    kept_html = _kept_result_view_html(archive_root, capture)
    owners = _target_owners(catalog, archive, target_configuration)

    store.begin_project_transfer_import(
        archive.target_request_id,
        archive_manifest_sha256=archive.sha256(),
        target_manifest_sha256=target_configuration.receipt.target_manifest_sha256,
        operational_payload_sha256=operational_entry.sha256,
        target_configuration_receipt=target_configuration.receipt.model_dump(mode="json"),
        capture=capture,
        kept_result_view_html=kept_html,
    )
    published_imported: ImportedProviderSourceInventory | None = None
    try:
        materialization = _publish_canonical(
            owners,
            archive,
            archive_root,
            target_configuration,
        )
        _publish_project_files(owners, archive_root, capture)
        published_imported = _publish_provider_history(owners, archive_root, archive)
        if owners.history.head_ref(materialization) != GraphHeadRef.model_validate(
            archive.main_head.model_dump(mode="json")
        ):
            raise RuntimeError("transfer import lost its canonical readback")
        publication_sha256 = _publication_sha256(
            archive,
            target_configuration,
            operational_payload_sha256=operational_entry.sha256,
            imported_sources=published_imported,
        )
        completed = store.complete_project_transfer_import(
            archive.target_request_id,
            publication_sha256=publication_sha256,
        )
        return completed
    except Exception:
        current_receipt = store.project_transfer_import(archive.target_request_id)
        if (
            published_imported is not None
            and current_receipt is not None
            and current_receipt.status != "complete"
        ):
            catalog.discard_unactivated_imported_sources(
                archive.target_request_id,
                expected_inventory=published_imported,
            )
        raise


def _validate_protocol(
    archive: TransferArchiveManifest,
    envelope: TransferArchiveEnvelope,
    configuration: TransferTargetConfiguration,
    *,
    transfer,
    provisioning,
    target_space_id: str,
) -> None:
    envelope.verify_manifest(archive)
    if transfer is None or provisioning is None:
        raise ValueError("target transfer import requires its linked durable requests")
    if (
        transfer.side != "target"
        or transfer.phase != "archive_bound"
        or transfer.request_id != archive.target_request_id
        or transfer.linked_request_id != archive.source_request_id
        or transfer.project_id != archive.project_id
        or transfer.source_space_id != archive.source_space_id
        or transfer.target_space_id != archive.target_space_id
        or transfer.target_space_id != target_space_id
        or transfer.archive_sha256 != envelope.archive_sha256
        or transfer.archive_size_bytes != envelope.archive_size_bytes
        or transfer.source_fence_head
        != GraphHeadRef.model_validate(archive.main_head.model_dump(mode="json"))
        or transfer.source_configuration_sha256 != archive.source_configuration_sha256
        or transfer.source_configuration.source_manifest_sha256 != archive.source_manifest_sha256
        or transfer.source_release_proof_sha256 != archive.source_release_proof_sha256
        or transfer.target_activation_proof_sha256 != archive.target_activation_proof_sha256
        or transfer.accepted_schema_generation != archive.source_schema_generation
        or transfer.accepted_archive_codec != archive.archive_codec
        or transfer.target_admission_receipt is None
        or transfer.source_release_receipt is None
    ):
        raise ValueError("target transfer request does not bind this archive")
    receipt = configuration.receipt
    if (
        provisioning.kind != "incoming_transfer"
        or provisioning.status != "ready_for_review"
        or provisioning.request_id != archive.target_request_id
        or provisioning.proposed_project_id != archive.project_id
        or provisioning.target_space_id != archive.target_space_id
        or provisioning.final_review_digest != receipt.final_review_sha256
        or receipt.target_request_id != archive.target_request_id
        or receipt.project_id != archive.project_id
        or receipt.target_space_id != archive.target_space_id
        or receipt.archive_manifest_sha256 != archive.sha256()
        or receipt.source_configuration_sha256 != archive.source_configuration_sha256
        or receipt.source_manifest_sha256 != archive.source_manifest_sha256
        or receipt.archive_schema_version != archive.schema_version
        or receipt.archive_codec != archive.archive_codec
        or receipt.main_head != archive.main_head
        or receipt.branch_heads != archive.branch_heads
    ):
        raise ValueError("reviewed target configuration does not bind this import")


def _operational_entry(
    entries: tuple[TransferArchiveEntry, ...],
) -> TransferArchiveEntry:
    operational = [entry for entry in entries if entry.group == "operational_records"]
    if len(operational) != 1 or operational[0].archive_path != TRANSFER_OPERATIONAL_RECORDS_PATH:
        raise ValueError("transfer archive requires one canonical operational payload")
    return operational[0]


def _validate_capture(
    archive: TransferArchiveManifest,
    capture: TransferProjectFileCapture,
) -> None:
    if (
        capture.project_id != archive.project_id
        or capture.records.project_id != archive.project_id
        or capture.records.attributions != archive.attributions
    ):
        raise ValueError("transfer operational records disagree with archive identity")
    expected = tuple(entry for entry in archive.entries if entry.group in _PROJECT_FILE_GROUPS)
    if capture.entries != expected:
        raise ValueError("transfer operational records disagree with project file entries")


def _target_owners(
    catalog: ProjectCatalog,
    archive: TransferArchiveManifest,
    configuration: TransferTargetConfiguration,
) -> _TargetOwners:
    try:
        manifest = Manifest.model_validate(tomlkit.parse(configuration.manifest_content).unwrap())
    except (ValueError, tomlkit.exceptions.ParseError) as exc:
        raise ValueError("reviewed target manifest is invalid") from exc
    state_repository = manifest.repository_map[manifest.state.repository]
    machine = manifest.machine_map[state_repository.machine]
    if machine.host:
        workspace = state_workspace_for_probe(manifest, catalog.data_dir)
    else:
        state_root = Path(state_repository.path) / ".research"
        workspace = LocalStateWorkspace(state_root, str(state_root))
    manifest._path = workspace.root / "manifest.toml"
    history = HistoryManager(
        manifest,
        workspace,
        expected_space_id=archive.target_space_id,
        project_id=archive.project_id,
    )
    paper = PaperService(manifest, catalog.store, workspace, project_id=archive.project_id)
    service = ProjectService(
        manifest,
        history,
        paper,
        data_dir=catalog.data_dir,
        project_id=archive.project_id,
        task_continuation_session=catalog.store.agent_task_continuation_session_id,
    )
    if service.imported_sources is None:
        raise RuntimeError("target transfer import lacks its imported-source owner")
    return _TargetOwners(manifest, workspace, history, paper, service)


def _publish_canonical(
    owners: _TargetOwners,
    archive: TransferArchiveManifest,
    archive_root: Path,
    configuration: TransferTargetConfiguration,
):
    with tempfile.TemporaryDirectory(prefix="rcp-transfer-target-manifest-") as temporary:
        manifest_source = Path(temporary) / "manifest.toml"
        manifest_source.write_text(configuration.manifest_content, encoding="utf-8")
        manifest_bytes = configuration.manifest_content.encode("utf-8")
        sources: dict[str, tuple[Path, str, int]] = {
            "manifest.toml": (
                manifest_source,
                hashlib.sha256(manifest_bytes).hexdigest(),
                len(manifest_bytes),
            )
        }
        for entry in archive.entries:
            if entry.group != "canonical_history":
                continue
            relative = PurePosixPath(entry.archive_path).relative_to("canonical").as_posix()
            sources[relative] = (
                archive_root / entry.archive_path,
                entry.sha256,
                entry.size_bytes,
            )
        result = owners.history.restore_canonical_history(
            sources,
            expected_main_head=GraphHeadRef.model_validate(
                archive.main_head.model_dump(mode="json")
            ),
            expected_branch_heads=tuple(
                GraphHeadRef.model_validate(head.model_dump(mode="json"))
                for head in archive.branch_heads
            ),
        )
    identity = owners.history.project_identity(result)
    if (
        identity is None
        or identity.project_id != archive.project_id
        or identity.home_space_id != archive.target_space_id
    ):
        raise ValueError("imported canonical history does not establish the target project home")
    return result


def _publish_project_files(
    owners: _TargetOwners,
    archive_root: Path,
    capture: TransferProjectFileCapture,
) -> None:
    operation_projects = {task.operation_id: capture.project_id for task in capture.records.tasks}
    with tempfile.TemporaryDirectory(prefix="rcp-transfer-chat-readback-") as temporary:
        chat_root = Path(temporary) / ".research" / "chat"
        for entry in capture.entries:
            source = archive_root / entry.archive_path
            path = PurePosixPath(entry.archive_path)
            if entry.group == "rcp_chat":
                staged_chat = chat_root / path.name
                staged_chat.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, staged_chat)
                owners.service.restore_canonical_chat(
                    staged_chat,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size_bytes,
                    operation_projects=operation_projects,
                )
            elif entry.group == "paper_introduction":
                owners.paper.restore_canonical(
                    source,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size_bytes,
                )
            elif entry.group == "fact":
                owners.workspace.restore_exact_file(
                    Path(*path.parts),
                    source,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size_bytes,
                )
            elif entry.group == "kept_artifact":
                owners.workspace.restore_kept_artifact(
                    path.name,
                    source,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size_bytes,
                )
            elif entry.group == "legacy_kept_result_view":
                owners.workspace.restore_kept_result_view(
                    path.name,
                    source,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size_bytes,
                )
            else:  # pragma: no cover - TransferProjectFileCapture closes this set
                raise RuntimeError("project transfer file capture contains an unknown group")


def _publish_provider_history(
    owners: _TargetOwners,
    archive_root: Path,
    archive: TransferArchiveManifest,
) -> ImportedProviderSourceInventory:
    assert owners.service.imported_sources is not None
    entries = tuple(entry for entry in archive.entries if entry.group == "provider_history")
    if entries:
        return owners.service.imported_sources.publish(archive_root, entries)
    inventory = owners.service.imported_sources.inventory()
    if inventory.files:
        raise ValueError("target imported-source owner already contains another inventory")
    return inventory


def _kept_result_view_html(
    archive_root: Path,
    capture: TransferProjectFileCapture,
) -> dict[str, str]:
    entries = {
        PurePosixPath(entry.archive_path).name: entry
        for entry in capture.entries
        if entry.group == "legacy_kept_result_view"
    }
    result: dict[str, str] = {}
    for view in capture.kept_result_views:
        try:
            result[view.view_id] = _read_entry_bytes(
                archive_root,
                entries[view.kept_filename],
            ).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("kept result-view bytes are not UTF-8 HTML") from exc
    return result


def _publication_sha256(
    archive: TransferArchiveManifest,
    configuration: TransferTargetConfiguration,
    *,
    operational_payload_sha256: str,
    imported_sources: ImportedProviderSourceInventory,
) -> str:
    published_groups = {
        "canonical_history",
        "rcp_chat",
        "paper_introduction",
        "fact",
        "kept_artifact",
        "legacy_kept_result_view",
        "provider_history",
    }
    payload = {
        "project_id": archive.project_id,
        "target_request_id": archive.target_request_id,
        "archive_manifest_sha256": archive.sha256(),
        "target_manifest_sha256": configuration.receipt.target_manifest_sha256,
        "operational_payload_sha256": operational_payload_sha256,
        "entries": [
            entry.model_dump(mode="json")
            for entry in archive.entries
            if entry.group in published_groups
        ],
        "main_head": archive.main_head.model_dump(mode="json"),
        "branch_heads": [head.model_dump(mode="json") for head in archive.branch_heads],
        "imported_source_fingerprint": imported_sources.fingerprint,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_archive_root(
    archive_root: Path,
    entries: tuple[TransferArchiveEntry, ...],
) -> None:
    try:
        root_mode = archive_root.lstat().st_mode
    except OSError as exc:
        raise ValueError("transfer archive staging root is unavailable") from exc
    if not stat.S_ISDIR(root_mode):
        raise ValueError("transfer archive staging root is not a directory")
    expected_files = {entry.archive_path for entry in entries}
    expected_directories = {
        PurePosixPath(*PurePosixPath(path).parts[:index]).as_posix()
        for path in expected_files
        for index in range(1, len(PurePosixPath(path).parts))
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for current, directories, files in os.walk(archive_root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError("transfer archive contains an unsafe directory entry")
            actual_directories.add(path.relative_to(archive_root).as_posix())
        for name in files:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("transfer archive contains an unsafe file entry")
            actual_files.add(path.relative_to(archive_root).as_posix())
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("transfer archive staging tree differs from its manifest")
    for entry in entries:
        _verify_entry(archive_root / entry.archive_path, entry)


def _read_entry_bytes(root: Path, entry: TransferArchiveEntry) -> bytes:
    path = root / entry.archive_path
    _verify_entry(path, entry)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"transfer archive entry is unavailable: {entry.archive_path}") from exc
    try:
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > entry.size_bytes:
                raise ValueError("transfer archive entry grew during readback")
    finally:
        os.close(descriptor)
    if len(payload) != entry.size_bytes or hashlib.sha256(payload).hexdigest() != entry.sha256:
        raise ValueError(f"transfer archive entry changed during readback: {entry.archive_path}")
    return bytes(payload)


def _verify_entry(path: Path, entry: TransferArchiveEntry) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"transfer archive entry is unavailable: {entry.archive_path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != entry.size_bytes:
            raise ValueError(
                f"transfer archive entry has an invalid file type: {entry.archive_path}"
            )
        while True:
            chunk = os.read(descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > entry.size_bytes:
                raise ValueError(f"transfer archive entry exceeds its size: {entry.archive_path}")
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"transfer archive entry is unavailable: {entry.archive_path}") from exc
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        size != entry.size_bytes
        or digest.hexdigest() != entry.sha256
        or any(getattr(before, field) != getattr(after, field) for field in stable)
    ):
        raise ValueError(f"transfer archive entry differs from its manifest: {entry.archive_path}")


__all__ = ["import_project_transfer"]
