"""Reviewed target configuration and replay proof for project transfer."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import uuid
from pathlib import Path, PurePosixPath

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp.config import Manifest
from rcp.core.models import GraphBranchMetadata
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.core.transitions import accepted_transition_head_chain_failure
from rcp.history import HistoryManager
from rcp.limits import (
    PROJECT_TRANSFER_COPY_BUFFER_BYTES,
    PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES,
    PROJECT_TRANSFER_MANIFEST_MAX_BYTES,
)
from rcp.setup import render_prepared_team_manifest
from rcp.storage import (
    ProjectProvisioningRequestRecord,
    ProjectTransferLinkReceipt,
    ProjectTransferSourceConfiguration,
)
from rcp.storage.provisioning import (
    project_provisioning_review_digest,
    project_transfer_source_configuration_sha256,
)
from rcp.transfer.archive import (
    TRANSFER_ARCHIVE_CODEC,
    TRANSFER_ARCHIVE_SCHEMA_VERSION,
    TRANSFER_RESEARCH_CANONICAL_ROOTS,
    TRANSFER_RESEARCH_DELEGATED_ROOTS,
    TRANSFER_RESEARCH_EXCLUDED_ROOTS,
    TRANSFER_RESEARCH_PROVENANCE_ROOTS,
    TransferArchiveEntry,
    TransferArchiveManifest,
    TransferGraphHead,
)
from rcp.transport import LocalStateWorkspace


class _StrictConfigurationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


class RetainedTransferBranch(_StrictConfigurationModel):
    branch_id: str
    revision: int = Field(ge=0)


class RetainedTransferHistory(_StrictConfigurationModel):
    state: str = Field(pattern=r"^(empty|matching)$")
    main_revision: int = Field(ge=0)
    branches: tuple[RetainedTransferBranch, ...]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class TransferTargetConfigurationReceipt(_StrictConfigurationModel):
    target_request_id: str
    project_id: str
    target_space_id: str
    final_review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_schema_version: int = Field(ge=1)
    archive_codec: str
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_manifest_size_bytes: int = Field(gt=0)
    main_head: TransferGraphHead
    branch_heads: tuple[TransferGraphHead, ...]
    retained_history: RetainedTransferHistory


class TransferTargetConfiguration(_StrictConfigurationModel):
    manifest_content: str
    receipt: TransferTargetConfigurationReceipt

    @model_validator(mode="after")
    def manifest_matches_receipt(self) -> TransferTargetConfiguration:
        payload = self.manifest_content.encode("utf-8")
        if (
            len(payload) != self.receipt.target_manifest_size_bytes
            or hashlib.sha256(payload).hexdigest() != self.receipt.target_manifest_sha256
        ):
            raise ValueError("target transfer manifest differs from its receipt")
        return self


def build_transfer_target_configuration(
    provisioning: ProjectProvisioningRequestRecord,
    source_configuration: ProjectTransferSourceConfiguration,
    link_receipt: ProjectTransferLinkReceipt,
    archive: TransferArchiveManifest,
    archive_root: Path,
    *,
    retained_research_root: Path | None = None,
    retained_history: RetainedTransferHistory | None = None,
) -> TransferTargetConfiguration:
    """Build and replay one target manifest without publishing project state."""

    if retained_research_root is not None and retained_history is not None:
        raise ValueError("target configuration has two retained-history sources")

    _validate_protocol_bindings(provisioning, source_configuration, link_receipt, archive)
    entries = {entry.archive_path: entry for entry in archive.entries}
    provenance = entries["provenance/manifest.toml"]
    source_manifest_bytes = _read_bound_file(archive_root, provenance, bounded=True)
    source_manifest = _parse_manifest(source_manifest_bytes, label="source transfer manifest")
    _validate_source_manifest(source_manifest, source_configuration)

    manifest_content = render_prepared_team_manifest(provisioning)
    target_manifest = _parse_manifest(
        manifest_content.encode("utf-8"),
        label="target transfer manifest",
    )
    _validate_target_manifest(target_manifest, provisioning, source_configuration)

    retained = (
        RetainedTransferHistory.model_validate(retained_history)
        if retained_history is not None
        else _inspect_retained_transfer_history(
            retained_research_root,
            archive_root,
            archive,
            source_manifest_bytes=source_manifest_bytes,
        )
    )
    _replay_archive(
        target_manifest,
        manifest_content,
        archive_root,
        archive,
    )
    manifest_bytes = manifest_content.encode("utf-8")
    return TransferTargetConfiguration(
        manifest_content=manifest_content,
        receipt=TransferTargetConfigurationReceipt(
            target_request_id=provisioning.request_id,
            project_id=archive.project_id,
            target_space_id=archive.target_space_id,
            final_review_sha256=provisioning.final_review_digest,
            archive_manifest_sha256=archive.sha256(),
            source_configuration_sha256=archive.source_configuration_sha256,
            source_manifest_sha256=archive.source_manifest_sha256,
            archive_schema_version=archive.schema_version,
            archive_codec=archive.archive_codec,
            target_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            target_manifest_size_bytes=len(manifest_bytes),
            main_head=archive.main_head,
            branch_heads=archive.branch_heads,
            retained_history=retained,
        ),
    )


def _inspect_retained_transfer_history(
    research_root: Path | None,
    archive_root: Path,
    archive: TransferArchiveManifest,
    *,
    source_manifest_bytes: bytes,
) -> RetainedTransferHistory:
    """Prove that retained Git state is empty or one exact archive prefix."""

    empty = RetainedTransferHistory(
        state="empty",
        main_revision=0,
        branches=(),
        fingerprint=hashlib.sha256(b"").hexdigest(),
    )
    if research_root is None or not os.path.lexists(research_root):
        return empty
    _require_directory(research_root, label="retained research root")
    direct = sorted(research_root.iterdir(), key=lambda path: path.name)
    if not direct:
        return empty
    allowed = (
        TRANSFER_RESEARCH_PROVENANCE_ROOTS
        | TRANSFER_RESEARCH_CANONICAL_ROOTS
        | TRANSFER_RESEARCH_DELEGATED_ROOTS
        | TRANSFER_RESEARCH_EXCLUDED_ROOTS
    )
    unknown = [path.name for path in direct if path.name not in allowed]
    if unknown:
        raise ValueError(f"retained research contains an unknown entry: {unknown[0]}")
    _validate_retained_top_level_types(direct)

    manifest_path = research_root / "manifest.toml"
    if not os.path.lexists(manifest_path):
        raise ValueError("retained research has state but no source manifest")
    if _read_regular_file(manifest_path, bounded=True) != source_manifest_bytes:
        raise ValueError("retained source manifest differs from the transfer provenance")

    archive_entries = {
        PurePosixPath(entry.archive_path).relative_to("canonical").as_posix(): entry
        for entry in archive.entries
        if entry.group == "canonical_history"
    }
    observed = _retained_canonical_files(research_root)
    if "scope-base.json" not in observed:
        raise ValueError("retained research is missing its scope provenance")
    if not any(_main_patch_revision(path) == 1 for path in observed):
        raise ValueError("retained research has no canonical project identity")

    bound: list[tuple[str, str]] = []
    for relative, path in sorted(observed.items()):
        entry = archive_entries.get(relative)
        if entry is None:
            raise ValueError(f"retained canonical entry is outside the archive: {relative}")
        digest, size = _file_digest(path, maximum_size=entry.size_bytes)
        is_branch_metadata = (
            re.fullmatch(
                r"branches/[0-9a-f-]+/branch\.json",
                relative,
            )
            is not None
        )
        if not is_branch_metadata and (digest, size) != (entry.sha256, entry.size_bytes):
            raise ValueError(f"retained canonical entry differs from the archive: {relative}")
        archive_path = archive_root / entry.archive_path
        if _file_digest(archive_path, maximum_size=entry.size_bytes) != (
            entry.sha256,
            entry.size_bytes,
        ):
            raise ValueError(f"archive canonical entry differs from its manifest: {relative}")
        bound.append((relative, digest))

    main_revisions = sorted(
        revision
        for relative in observed
        if (revision := _main_patch_revision(relative)) is not None
    )
    if main_revisions != list(range(1, max(main_revisions) + 1)):
        raise ValueError("retained main Patch history is not one contiguous prefix")
    if max(main_revisions) > archive.main_head.revision:
        raise ValueError("retained main history is later than the transfer archive")

    branch_revisions: list[RetainedTransferBranch] = []
    archive_heads = {
        head.target.branch_id: head.revision
        for head in archive.branch_heads
        if head.target.branch_id is not None
    }
    observed_branch_ids = sorted(
        {
            PurePosixPath(relative).parts[1]
            for relative in observed
            if relative.startswith("branches/")
        }
    )
    for branch_id in observed_branch_ids:
        metadata = f"branches/{branch_id}/branch.json"
        if metadata not in observed or branch_id not in archive_heads:
            raise ValueError("retained branch history lacks matching archive metadata")
        try:
            branch_metadata = GraphBranchMetadata.model_validate_json(
                _read_regular_file(observed[metadata], bounded=True)
            )
            archive_metadata = GraphBranchMetadata.model_validate_json(
                _read_bound_file(
                    archive_root,
                    archive_entries[metadata],
                    bounded=True,
                )
            )
        except ValueError as exc:
            raise ValueError("retained branch metadata is invalid") from exc
        if (
            branch_metadata.branch_id != branch_id
            or branch_metadata.project_id != archive.project_id
            or archive_metadata.branch_id != branch_id
            or archive_metadata.project_id != archive.project_id
            or archive_metadata.head.revision != archive_heads[branch_id]
            or branch_metadata.model_copy(update={"head": archive_metadata.head})
            != archive_metadata
            or branch_metadata.head.revision > archive_metadata.head.revision
        ):
            raise ValueError("retained branch metadata differs from the archive")
        patch_paths = sorted(
            (
                int(PurePosixPath(relative).stem),
                observed[relative],
            )
            for relative in observed
            if re.fullmatch(
                rf"branches/{re.escape(branch_id)}/patches/[0-9]{{6}}\.json",
                relative,
            )
        )
        expected_revisions = list(
            range(branch_metadata.base_head.revision + 1, branch_metadata.head.revision + 1)
        )
        if [revision for revision, _path in patch_paths] != expected_revisions:
            raise ValueError("retained branch Patch history does not match its metadata")
        patches = [
            HistoryManager._decode_persisted_patch(_read_regular_file(path, bounded=True))
            for _revision, path in patch_paths
        ]
        failure = accepted_transition_head_chain_failure(
            patches,
            target=GraphTargetRef(kind="branch", branch_id=branch_id),
            initial_transition_id=branch_metadata.base_head.transition_id,
        )
        transition_id = branch_metadata.base_head.transition_id
        for patch in patches:
            if patch.admission == "accepted" and patch.transition is not None:
                transition_id = patch.transition.transition_id
        expected_head = GraphHeadRef(
            target=GraphTargetRef(kind="branch", branch_id=branch_id),
            revision=branch_metadata.head.revision,
            transition_id=transition_id,
        )
        if failure is not None or branch_metadata.head != expected_head:
            raise ValueError("retained branch head does not match its Patch prefix")
        branch_revisions.append(
            RetainedTransferBranch(
                branch_id=branch_id,
                revision=branch_metadata.head.revision,
            )
        )

    fingerprint = hashlib.sha256()
    for relative, digest in bound:
        fingerprint.update(relative.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(digest.encode("ascii"))
        fingerprint.update(b"\n")
    return RetainedTransferHistory(
        state="matching",
        main_revision=max(main_revisions),
        branches=tuple(branch_revisions),
        fingerprint=fingerprint.hexdigest(),
    )


def _validate_protocol_bindings(
    provisioning: ProjectProvisioningRequestRecord,
    source: ProjectTransferSourceConfiguration,
    link: ProjectTransferLinkReceipt,
    archive: TransferArchiveManifest,
) -> None:
    if (
        provisioning.kind != "incoming_transfer"
        or provisioning.status != "ready_for_review"
        or provisioning.final_review_digest != project_provisioning_review_digest(provisioning)
    ):
        raise ValueError("target transfer provisioning is not ready for its reviewed import")
    source_digest = project_transfer_source_configuration_sha256(source)
    if link.source_configuration_sha256 != source_digest:
        raise ValueError("target link does not bind the supplied source configuration")
    if (
        link.target_request_id != provisioning.request_id
        or link.project_id != provisioning.proposed_project_id
        or link.target_space_id != provisioning.target_space_id
    ):
        raise ValueError("target provisioning and transfer link identities differ")
    if (
        archive.schema_version != TRANSFER_ARCHIVE_SCHEMA_VERSION
        or archive.archive_codec != TRANSFER_ARCHIVE_CODEC
        or archive.source_rcp_version != source.source_rcp_version
        or archive.source_schema_generation != source.source_schema_generation
        or archive.source_schema_generation != link.accepted_schema_generation
        or archive.archive_codec != link.accepted_archive_codec
        or link.accepted_archive_codec not in source.supported_archive_codecs
    ):
        raise ValueError("transfer archive uses a different accepted format")
    expected = {
        "project_id": link.project_id,
        "source_space_id": link.source_space_id,
        "target_space_id": link.target_space_id,
        "source_request_id": link.source_request_id,
        "target_request_id": link.target_request_id,
        "source_configuration_sha256": source_digest,
        "source_manifest_sha256": source.source_manifest_sha256,
        "source_release_proof_sha256": link.source_release_proof_sha256,
        "target_activation_proof_sha256": link.target_activation_proof_sha256,
    }
    actual = {key: getattr(archive, key) for key in expected}
    if actual != expected:
        raise ValueError("transfer archive does not match the reviewed source/target link")
    source_repositories = {item.alias: item.repository.identity for item in source.repositories}
    linked_repositories = {
        item.alias: item.repository.identity for item in link.target_repositories
    }
    if linked_repositories != source_repositories:
        raise ValueError("transfer link names different GitHub repositories")


def _validate_source_manifest(
    manifest: Manifest,
    source: ProjectTransferSourceConfiguration,
) -> None:
    repositories = {item.alias: item for item in manifest.repositories}
    provenance = {item.alias: item for item in source.repositories}
    if set(repositories) != set(provenance):
        raise ValueError("source manifest repository aliases differ from transfer provenance")
    if any(repositories[alias].machine != item.machine_alias for alias, item in provenance.items()):
        raise ValueError("source manifest machine aliases differ from transfer provenance")
    if not {item.machine_alias for item in source.repositories}.issubset(manifest.machine_map):
        raise ValueError("source transfer provenance names a missing machine alias")
    if set(manifest.machine_map) != set(source.machine_aliases):
        raise ValueError("source manifest machine aliases differ from transfer provenance")
    if (
        manifest.state.repository != source.state_repository
        or tuple(manifest.project.truth_scope) != source.project_truth_scope
        or tuple(manifest.agent.default_run_truth_scope) != source.default_run_truth_scope
    ):
        raise ValueError("source manifest scope differs from transfer provenance")


def _validate_target_manifest(
    manifest: Manifest,
    provisioning: ProjectProvisioningRequestRecord,
    source: ProjectTransferSourceConfiguration,
) -> None:
    expected_repositories = {item.alias: item for item in provisioning.repositories}
    if set(manifest.repository_map) != set(expected_repositories):
        raise ValueError("target manifest does not preserve every historical repository alias")
    if (
        manifest.state.repository != source.state_repository
        or tuple(manifest.project.truth_scope) != source.project_truth_scope
        or tuple(manifest.agent.default_run_truth_scope) != source.default_run_truth_scope
    ):
        raise ValueError("target manifest changes transferred scope provenance")
    source_repositories = {item.alias: item.repository.identity for item in source.repositories}
    target_repositories = {
        alias: item.repository.identity for alias, item in expected_repositories.items()
    }
    if target_repositories != source_repositories:
        raise ValueError("target manifest review names different GitHub repositories")
    source_repository_machines = {item.alias: item.machine_alias for item in source.repositories}
    target_repository_machines = {
        item.alias: item.machine_alias for item in provisioning.repositories
    }
    if target_repository_machines != source_repository_machines:
        raise ValueError("target manifest renames a historical repository machine alias")
    if not set(source.machine_aliases).issubset(manifest.machine_map):
        raise ValueError("target manifest omits a historical machine alias")


def _replay_archive(
    target_manifest: Manifest,
    manifest_content: str,
    archive_root: Path,
    archive: TransferArchiveManifest,
) -> None:
    with tempfile.TemporaryDirectory(prefix="rcp-transfer-config-") as temporary:
        temporary_root = Path(temporary)
        research_root = temporary_root / ".research"
        research_root.mkdir(mode=0o700)
        manifest_source = temporary_root / "target-manifest.toml"
        manifest_source.write_text(manifest_content, encoding="utf-8")
        manifest_bytes = manifest_content.encode("utf-8")
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
        history = HistoryManager(
            target_manifest,
            LocalStateWorkspace(research_root, str(research_root)),
            expected_space_id=archive.target_space_id,
            project_id=archive.project_id,
        )
        result = history.restore_canonical_history(
            sources,
            expected_main_head=GraphHeadRef.model_validate(archive.main_head.model_dump()),
            expected_branch_heads=tuple(
                GraphHeadRef.model_validate(head.model_dump()) for head in archive.branch_heads
            ),
        )
        identity = history.project_identity(result)
        if (
            identity is None
            or identity.project_id != archive.project_id
            or identity.home_space_id != archive.target_space_id
        ):
            raise ValueError("transferred history does not replay to its target project home")


def _parse_manifest(payload: bytes, *, label: str) -> Manifest:
    if len(payload) > PROJECT_TRANSFER_MANIFEST_MAX_BYTES:
        raise ValueError(f"{label} exceeds its byte bound")
    try:
        document = tomlkit.parse(payload.decode("utf-8")).unwrap()
        return Manifest.model_validate(document)
    except (UnicodeDecodeError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise ValueError(f"{label} is invalid") from exc


def _read_bound_file(root: Path, entry: TransferArchiveEntry, *, bounded: bool) -> bytes:
    path = root / entry.archive_path
    payload = _read_regular_file(path, bounded=bounded)
    if len(payload) != entry.size_bytes or hashlib.sha256(payload).hexdigest() != entry.sha256:
        raise ValueError(f"archive entry differs from its manifest: {entry.archive_path}")
    return payload


def _read_regular_file(path: Path, *, bounded: bool) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        path_before = path.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if not stat.S_ISREG(before.st_mode) or any(
            getattr(before, field) != getattr(path_before, field) for field in stable
        ):
            raise ValueError(f"transfer input is not a regular file: {path.name}")
        if bounded and before.st_size > PROJECT_TRANSFER_MANIFEST_MAX_BYTES:
            raise ValueError(f"transfer input exceeds its byte bound: {path.name}")
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            if bounded and len(payload) + len(chunk) > PROJECT_TRANSFER_MANIFEST_MAX_BYTES:
                raise ValueError(f"transfer input exceeds its byte bound: {path.name}")
            payload.extend(chunk)
        after = os.fstat(descriptor)
        path_after = path.lstat()
        if any(getattr(before, field) != getattr(after, field) for field in stable) or any(
            getattr(after, field) != getattr(path_after, field) for field in stable
        ):
            raise ValueError(f"transfer input changed while it was read: {path.name}")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _file_digest(path: Path, *, maximum_size: int) -> tuple[str, int]:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        path_before = path.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if not stat.S_ISREG(before.st_mode) or any(
            getattr(before, field) != getattr(path_before, field) for field in stable
        ):
            raise ValueError(f"transfer input is not a regular file: {path.name}")
        if before.st_size > maximum_size:
            raise ValueError(f"transfer input exceeds its bound: {path.name}")
        while True:
            chunk = os.read(descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if size > maximum_size:
                raise ValueError(f"transfer input exceeds its bound: {path.name}")
        after = os.fstat(descriptor)
        path_after = path.lstat()
        if any(getattr(before, field) != getattr(after, field) for field in stable) or any(
            getattr(after, field) != getattr(path_after, field) for field in stable
        ):
            raise ValueError(f"transfer input changed while it was read: {path.name}")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _retained_canonical_files(root: Path) -> dict[str, Path]:
    observed: dict[str, Path] = {}
    count = 0
    for name in sorted(TRANSFER_RESEARCH_CANONICAL_ROOTS):
        path = root / name
        if not os.path.lexists(path):
            continue
        if name == "scope-base.json":
            _require_regular(path, label="retained scope provenance")
            observed[name] = path
            count += 1
            continue
        _require_directory(path, label=f"retained {name} root")
        pending = [path]
        while pending:
            directory = pending.pop()
            for child in sorted(directory.iterdir(), key=lambda item: item.name):
                metadata = child.lstat()
                relative = child.relative_to(root).as_posix()
                if stat.S_ISDIR(metadata.st_mode):
                    if not _retained_canonical_directory(relative):
                        raise ValueError(
                            f"retained canonical history contains an unknown directory: {relative}"
                        )
                    pending.append(child)
                elif stat.S_ISREG(metadata.st_mode):
                    if not _retained_canonical_file(relative):
                        raise ValueError(
                            f"retained canonical history contains an unknown file: {relative}"
                        )
                    observed[relative] = child
                    count += 1
                    if count > PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES:
                        raise ValueError("retained canonical history exceeds its entry bound")
                else:
                    raise ValueError("retained canonical history contains an unsafe entry")
    return observed


def _validate_retained_top_level_types(paths: list[Path]) -> None:
    directories = TRANSFER_RESEARCH_DELEGATED_ROOTS | {"branches", "patches", ".publish"}
    for path in paths:
        if path.name in directories:
            _require_directory(path, label=f"retained {path.name} entry")
        else:
            _require_regular(path, label=f"retained {path.name} entry")


def _require_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a regular directory")


def _require_regular(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")


def _main_patch_revision(relative: str) -> int | None:
    path = PurePosixPath(relative)
    if (
        len(path.parts) == 2
        and path.parts[0] == "patches"
        and re.fullmatch(r"[0-9]{6}\.json", path.parts[1])
    ) or (
        len(path.parts) == 3
        and path.parts[0] == "patches"
        and path.parts[1].startswith("batch-")
        and re.fullmatch(r"[0-9]{6}\.json", path.parts[2])
    ):
        return int(path.stem)
    return None


def _retained_canonical_directory(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    if len(parts) == 2 and parts[0] == "patches":
        return parts[1].startswith("batch-")
    if len(parts) == 2 and parts[0] == "branches":
        return _canonical_uuid4(parts[1])
    return (
        len(parts) == 3
        and parts[0] == "branches"
        and _canonical_uuid4(parts[1])
        and parts[2] in {"patches", "merges"}
    )


def _retained_canonical_file(relative: str) -> bool:
    path = PurePosixPath(relative)
    parts = path.parts
    if _main_patch_revision(relative) is not None:
        return True
    if len(parts) == 3 and parts[0] == "branches" and _canonical_uuid4(parts[1]):
        return parts[2] == "branch.json"
    if len(parts) != 4 or parts[0] != "branches" or not _canonical_uuid4(parts[1]):
        return False
    if parts[2] == "patches":
        return re.fullmatch(r"[0-9]{6}\.json", parts[3]) is not None
    if parts[2] == "merges":
        return re.fullmatch(r"[0-9a-f]{64}\.json", parts[3]) is not None
    return False


def _canonical_uuid4(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


__all__ = [
    "RetainedTransferBranch",
    "RetainedTransferHistory",
    "TransferTargetConfiguration",
    "TransferTargetConfigurationReceipt",
    "build_transfer_target_configuration",
]
