from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp.history.manager import canonical_fact_sources, iter_canonical_fact_bytes
from rcp.limits import (
    PROJECT_TRANSFER_COPY_BUFFER_BYTES,
    PROJECT_TRANSFER_STABLE_READ_ATTEMPTS,
)
from rcp.paper.service import (
    canonical_introduction_backup_source,
    validate_canonical_introduction_backup,
)
from rcp.service import (
    ProjectService,
    canonical_chat_backup_sources,
    iter_canonical_chat_transfer,
)
from rcp.transfer.archive import TransferArchiveEntry
from rcp.transfer.records import (
    TransferArtifactReference,
    TransferRecordBundle,
    TransferTaskRecord,
)
from rcp.transport.state import StateUnavailable

_PROJECT_FILE_GROUPS = frozenset(
    {
        "rcp_chat",
        "paper_introduction",
        "fact",
        "kept_artifact",
        "legacy_kept_result_view",
    }
)

TRANSFER_OPERATIONAL_RECORDS_PATH = "records/project.jsonl"


class TransferLegacyKeptResultView(BaseModel):
    """Readable legacy view history without a source execution continuation."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )

    view_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    experiment_id: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    origin_operation_id: str
    latest_operation_id: str
    provider: str = Field(min_length=1)
    model: str
    reasoning: str
    source_name: str = Field(min_length=1, max_length=255)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    created_at: str
    updated_at: str
    expires_at: str
    kept_filename: str = Field(min_length=1, max_length=255)
    kept_at: str

    @field_validator("origin_operation_id", "latest_operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except ValueError as exc:
            raise ValueError("result-view operation identity must be canonical") from exc
        if str(parsed) != value:
            raise ValueError("result-view operation identity must be canonical")
        return value

    @field_validator("source_name", "kept_filename")
    @classmethod
    def validate_plain_filename(cls, value: str) -> str:
        if (
            value != value.strip()
            or "\\" in value
            or PurePosixPath(value).name != value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("result-view filenames must be plain bounded names")
        return value

    @field_validator("created_at", "updated_at", "expires_at", "kept_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("result-view timestamps must be ISO 8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("result-view timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> TransferLegacyKeptResultView:
        created = datetime.fromisoformat(self.created_at)
        if (
            datetime.fromisoformat(self.updated_at) < created
            or datetime.fromisoformat(self.expires_at) < created
            or datetime.fromisoformat(self.kept_at) < created
        ):
            raise ValueError("result-view lifecycle precedes its creation")
        return self


class TransferProjectFileCapture(BaseModel):
    """Exact file entries plus the record bundle bound to captured kept bytes."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )

    project_id: str
    records: TransferRecordBundle
    kept_result_views: tuple[TransferLegacyKeptResultView, ...]
    entries: tuple[TransferArchiveEntry, ...]
    payload_size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_capture(self) -> TransferProjectFileCapture:
        if self.records.project_id != self.project_id:
            raise ValueError("project file capture and record bundle identities differ")
        paths = [entry.archive_path for entry in self.entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("project transfer file paths must be sorted and unique")
        if any(entry.group not in _PROJECT_FILE_GROUPS for entry in self.entries):
            raise ValueError("project file capture contains a file owned by another packet")
        if sum(entry.size_bytes for entry in self.entries) != self.payload_size_bytes:
            raise ValueError("project transfer file byte total does not match its entries")
        artifact_entries = {
            PurePosixPath(entry.archive_path).name: entry
            for entry in self.entries
            if entry.group == "kept_artifact"
        }
        references = [
            artifact
            for task in self.records.tasks
            for artifact in task.artifacts
            if artifact.kept_filename is not None
        ]
        if set(artifact_entries) != {artifact.kept_filename for artifact in references}:
            raise ValueError("captured kept artifacts differ from terminal task references")
        for artifact in references:
            assert artifact.kept_filename is not None
            entry = artifact_entries[artifact.kept_filename]
            if artifact.content_sha256 != entry.sha256:
                raise ValueError("kept artifact record does not match its captured bytes")
        view_ids = [view.view_id for view in self.kept_result_views]
        view_filenames = [view.kept_filename for view in self.kept_result_views]
        if len(view_ids) != len(set(view_ids)) or len(view_filenames) != len(set(view_filenames)):
            raise ValueError("captured kept result views must have unique identities and files")
        task_ids = {task.operation_id for task in self.records.tasks}
        if any(
            view.origin_operation_id not in task_ids or view.latest_operation_id not in task_ids
            for view in self.kept_result_views
        ):
            raise ValueError("captured kept result views must bind to transferred task history")
        view_entries = {
            PurePosixPath(entry.archive_path).name: entry
            for entry in self.entries
            if entry.group == "legacy_kept_result_view"
        }
        if set(view_entries) != set(view_filenames):
            raise ValueError("captured kept result-view bytes differ from their history records")
        for view in self.kept_result_views:
            entry = view_entries[view.kept_filename]
            if (entry.sha256, entry.size_bytes) != (view.content_sha256, view.size_bytes):
                raise ValueError("kept result-view record does not match its captured bytes")
        return self


def transfer_project_file_payload(capture: TransferProjectFileCapture) -> bytes:
    """Encode the one typed operational payload bound by the archive manifest."""

    normalized = TransferProjectFileCapture.model_validate(capture)
    return (
        json.dumps(
            normalized.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def parse_transfer_project_file_payload(payload: bytes) -> TransferProjectFileCapture:
    """Decode one canonical operational payload and reject alternate encodings."""

    try:
        capture = TransferProjectFileCapture.model_validate_json(payload)
    except ValueError as exc:
        raise ValueError("project transfer operational records are invalid") from exc
    if transfer_project_file_payload(capture) != payload:
        raise ValueError("project transfer operational records are not canonical")
    return capture


def capture_project_transfer_files(
    service: ProjectService,
    records: TransferRecordBundle,
    capture_root: Path,
) -> TransferProjectFileCapture:
    """Capture transformed canonical human files and referenced kept bytes once."""

    if service.history.project_id != records.project_id:
        raise ValueError("project service and transfer record identities differ")
    try:
        capture_root.mkdir(mode=0o700)
    except OSError as exc:
        raise ValueError("project transfer capture root must be one new private directory") from exc
    captured: list[TransferArchiveEntry] = []
    try:
        operation_id_map = {task.operation_id: task.operation_id for task in records.tasks}
        workspace = service.history.workspace
        with tempfile.TemporaryDirectory(prefix="rcp-transfer-research-") as temporary:
            export_root = Path(temporary)
            export_root.chmod(0o700)
            source_root = workspace.backup_source_root(export_root)
            for source in canonical_chat_backup_sources(source_root):
                captured.append(
                    _capture_chunks(
                        capture_root,
                        PurePosixPath("chats") / source.path.name,
                        "rcp_chat",
                        iter_canonical_chat_transfer(
                            source,
                            operation_id_map=operation_id_map,
                        ),
                    )
                )
            introduction = canonical_introduction_backup_source(source_root)
            if introduction is not None:
                entry = _capture_regular_file(
                    capture_root,
                    introduction,
                    PurePosixPath("paper/introduction.md"),
                    "paper_introduction",
                )
                validate_canonical_introduction_backup(capture_root / entry.archive_path)
                captured.append(entry)
            for source in canonical_fact_sources(source_root):
                captured.append(
                    _capture_chunks(
                        capture_root,
                        PurePosixPath("facts") / source.relative_path,
                        "fact",
                        iter_canonical_fact_bytes(
                            source_root,
                            source,
                            chunk_size=PROJECT_TRANSFER_COPY_BUFFER_BYTES,
                        ),
                    )
                )

        artifact_digests: dict[str, tuple[str, int]] = {}
        for filename, references in _kept_artifact_references(records).items():
            data = _stable_workspace_bytes(lambda name=filename: workspace.read_kept_artifact(name))
            digest = hashlib.sha256(data).hexdigest()
            for reference in references:
                if reference.size_bytes is not None and reference.size_bytes != len(data):
                    raise ValueError("a kept artifact size differs from its terminal task record")
                if reference.content_sha256 is not None and reference.content_sha256 != digest:
                    raise ValueError("a kept artifact digest differs from its terminal task record")
            captured.append(
                _capture_chunks(
                    capture_root,
                    PurePosixPath("artifacts") / filename,
                    "kept_artifact",
                    (data,),
                )
            )
            artifact_digests[filename] = (digest, len(data))

        task_ids = {task.operation_id for task in records.tasks}
        transferred_views: list[TransferLegacyKeptResultView] = []
        for view in service.paper.store.kept_result_views(records.project_id):
            if (
                view.origin_operation_id not in task_ids
                or view.latest_operation_id not in task_ids
                or view.kept_filename is None
                or view.kept_at is None
            ):
                raise ValueError("a kept result view is not bound to transferred task history")
            transferred = TransferLegacyKeptResultView(
                view_id=view.view_id,
                experiment_id=view.experiment_id,
                chat_id=view.chat_id,
                origin_operation_id=view.origin_operation_id,
                latest_operation_id=view.latest_operation_id,
                provider=view.provider,
                model=view.model,
                reasoning=view.reasoning,
                source_name=view.source_name,
                content_sha256=view.content_sha256,
                size_bytes=view.size_bytes,
                created_at=view.created_at,
                updated_at=view.updated_at,
                expires_at=view.expires_at,
                kept_filename=view.kept_filename,
                kept_at=view.kept_at,
            )
            data = _stable_workspace_bytes(
                lambda name=view.kept_filename: workspace.read_kept_result_view(name)
            )
            if (
                len(data) != view.size_bytes
                or hashlib.sha256(data).hexdigest() != view.content_sha256
            ):
                raise ValueError("a kept result view differs from its stored record")
            captured.append(
                _capture_chunks(
                    capture_root,
                    PurePosixPath("result-views") / view.kept_filename,
                    "legacy_kept_result_view",
                    (data,),
                )
            )
            transferred_views.append(transferred)

        bound_records = _bind_kept_artifact_digests(records, artifact_digests)
        ordered = tuple(sorted(captured, key=lambda entry: entry.archive_path))
        _fsync_tree(capture_root)
        return TransferProjectFileCapture(
            project_id=records.project_id,
            records=bound_records,
            kept_result_views=tuple(transferred_views),
            entries=ordered,
            payload_size_bytes=sum(entry.size_bytes for entry in ordered),
        )
    except BaseException:
        _discard_new_capture_root(capture_root)
        raise


def _kept_artifact_references(
    records: TransferRecordBundle,
) -> dict[str, list[TransferArtifactReference]]:
    references: dict[str, list[TransferArtifactReference]] = {}
    for task in records.tasks:
        for artifact in task.artifacts:
            if artifact.kept_filename is not None:
                references.setdefault(artifact.kept_filename, []).append(artifact)
    return dict(sorted(references.items()))


def _bind_kept_artifact_digests(
    records: TransferRecordBundle,
    digests: dict[str, tuple[str, int]],
) -> TransferRecordBundle:
    tasks: list[TransferTaskRecord] = []
    for task in records.tasks:
        artifacts = tuple(
            artifact.model_copy(
                update={
                    "content_sha256": digests[artifact.kept_filename][0],
                    "size_bytes": artifact.size_bytes
                    if artifact.size_bytes is not None
                    else digests[artifact.kept_filename][1],
                }
            )
            if artifact.kept_filename is not None
            else artifact
            for artifact in task.artifacts
        )
        tasks.append(task.model_copy(update={"artifacts": artifacts}))
    return TransferRecordBundle.model_validate(
        {**records.model_dump(mode="python"), "tasks": tuple(tasks)}
    )


def _stable_workspace_bytes(reader) -> bytes:
    previous: tuple[str, int, bytes] | None = None
    last_error: BaseException | None = None
    for _ in range(PROJECT_TRANSFER_STABLE_READ_ATTEMPTS):
        try:
            data = reader()
            if not isinstance(data, bytes):
                raise ValueError("a kept-file reader returned non-byte content")
        except (OSError, StateUnavailable, ValueError) as exc:
            previous = None
            last_error = exc
            continue
        observed = (hashlib.sha256(data).hexdigest(), len(data), data)
        if previous is not None and previous == observed:
            return data
        previous = observed
    raise ValueError("a referenced kept file did not stabilize for transfer") from last_error


def _capture_regular_file(
    capture_root: Path,
    source: Path,
    archive_path: PurePosixPath,
    group: str,
) -> TransferArchiveEntry:
    return _capture_chunks(
        capture_root,
        archive_path,
        group,
        _regular_file_chunks(source),
    )


def _regular_file_chunks(source: Path) -> Iterator[bytes]:
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        initial = os.fstat(descriptor)
        current = source.lstat()
        if not stat.S_ISREG(initial.st_mode) or (
            initial.st_dev,
            initial.st_ino,
        ) != (current.st_dev, current.st_ino):
            raise ValueError("a project transfer source is not one safe regular file")
        while True:
            chunk = os.read(descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            yield chunk
        final = os.fstat(descriptor)
        path_final = source.lstat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, field) != getattr(final, field) for field in stable_fields) or any(
            getattr(final, field) != getattr(path_final, field) for field in stable_fields
        ):
            raise ValueError("a project file changed during its transfer read")
    finally:
        os.close(descriptor)


def _capture_chunks(
    capture_root: Path,
    archive_path: PurePosixPath,
    group: str,
    chunks: Iterable[bytes],
) -> TransferArchiveEntry:
    entry_path = TransferArchiveEntry(
        archive_path=archive_path.as_posix(),
        group=group,
        sha256="0" * 64,
        size_bytes=0,
    ).archive_path
    destination = capture_root.joinpath(*PurePosixPath(entry_path).parts)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.transfer-{uuid.uuid4().hex}")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise ValueError("project transfer capture requires byte chunks")
            _write_all(descriptor, chunk)
            digest.update(chunk)
            size += len(chunk)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        descriptor = -1
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    return TransferArchiveEntry(
        archive_path=entry_path,
        group=group,
        sha256=digest.hexdigest(),
        size_bytes=size,
    )


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short project transfer file write")
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


def _discard_new_capture_root(capture_root: Path) -> None:
    try:
        metadata = capture_root.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("refusing to discard an unsafe project transfer capture")
    shutil.rmtree(capture_root)


__all__ = [
    "TRANSFER_OPERATIONAL_RECORDS_PATH",
    "TransferLegacyKeptResultView",
    "TransferProjectFileCapture",
    "capture_project_transfer_files",
    "parse_transfer_project_file_payload",
    "transfer_project_file_payload",
]
