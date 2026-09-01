"""Bounded optimistic byte capture for one backup project."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Literal

from rcp.history.manager import (
    CanonicalFactSource,
    canonical_fact_sources,
    iter_canonical_fact_bytes,
)
from rcp.limits import (
    BACKUP_COPY_BUFFER_BYTES,
    BACKUP_INVENTORY_MAX_ENTRIES,
    BACKUP_STABLE_READ_ATTEMPTS,
    CHAT_ARTIFACT_MAX_FILE_BYTES,
)
from rcp.server_ops.backup_models import BackupFileEntry
from rcp.service import (
    CanonicalChatBackupSource,
    canonical_chat_backup_sources,
    iter_canonical_chat_backup_prefix,
)
from rcp.transport.state import StateUnavailable

ProjectFileGroup = Literal[
    "canonical",
    "chat",
    "paper_introduction",
    "fact",
    "kept_artifact",
    "legacy_kept_result_view",
]


class BackupProjectFileUnavailable(RuntimeError):
    """One project could not support an honest optimistic file capture."""


def capture_chat_files(
    source_root: Path,
    project_root: Path,
    project_id: str,
    *,
    operation_projects: Mapping[str, str],
) -> list[BackupFileEntry]:
    entries: list[BackupFileEntry] = []
    for source in canonical_chat_backup_sources(source_root):
        entry = _copy_chat_prefix(
            source,
            project_root,
            project_id,
            operation_projects=operation_projects,
        )
        if entry is not None:
            entries.append(entry)
    return entries


def _copy_chat_prefix(
    source: CanonicalChatBackupSource,
    project_root: Path,
    project_id: str,
    *,
    operation_projects: Mapping[str, str],
) -> BackupFileEntry | None:
    relative = PurePosixPath(".research/chat") / source.path.name
    destination = _destination_path(project_root, relative)
    _prepare_destination_parent(project_root, destination.parent)
    temporary = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        for line in iter_canonical_chat_backup_prefix(
            source,
            project_id=project_id,
            operation_projects=operation_projects,
        ):
            _write_all(descriptor, line)
            digest.update(line)
            size += len(line)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        descriptor = -1
        _unlink_if_present(temporary)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not size:
        _unlink_if_present(temporary)
        return None
    os.replace(temporary, destination)
    fsync_directory(destination.parent)
    return _file_entry(project_id, relative, "chat", digest.hexdigest(), size)


def fact_backup_sources(source_root: Path) -> tuple[CanonicalFactSource, ...]:
    try:
        return canonical_fact_sources(source_root)
    except ValueError as exc:
        raise BackupProjectFileUnavailable(str(exc)) from exc


def stable_copy_fact_entry(
    source_root: Path,
    source: CanonicalFactSource,
    project_root: Path,
    project_id: str,
    relative: PurePosixPath,
) -> BackupFileEntry:
    """Copy one fact through its owner while allowing bounded stable replacement."""

    destination = _destination_path(project_root, relative)
    _prepare_destination_parent(project_root, destination.parent)
    last_error: BaseException | None = None
    for _ in range(BACKUP_STABLE_READ_ATTEMPTS):
        temporary = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
        try:
            current = next(
                (
                    candidate
                    for candidate in canonical_fact_sources(source_root)
                    if candidate.relative_path == source.relative_path
                ),
                None,
            )
            if current is None:
                raise ValueError("The fact disappeared before capture.")
            digest, size = _copy_fact_source(source_root, current, temporary)
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
            return _file_entry(project_id, relative, "fact", digest, size)
        except (OSError, ValueError) as exc:
            last_error = exc
            _unlink_if_present(temporary)
    raise BackupProjectFileUnavailable(
        "A project fact did not stabilize for capture."
    ) from last_error


def _copy_fact_source(
    source_root: Path,
    source: CanonicalFactSource,
    destination: Path,
) -> tuple[str, int]:
    descriptor = os.open(
        destination,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in iter_canonical_fact_bytes(
            source_root,
            source,
            chunk_size=BACKUP_COPY_BUFFER_BYTES,
        ):
            _write_all(descriptor, chunk)
            digest.update(chunk)
            size += len(chunk)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def stable_copy_entry(
    source: Path,
    project_root: Path,
    project_id: str,
    relative: PurePosixPath,
    *,
    group: Literal["canonical", "paper_introduction", "fact"],
    expected_size: int | None = None,
) -> BackupFileEntry:
    destination = _destination_path(project_root, relative)
    _prepare_destination_parent(project_root, destination.parent)
    last_error: BaseException | None = None
    for _ in range(BACKUP_STABLE_READ_ATTEMPTS):
        temporary = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
        try:
            digest, size = _copy_one_stable_file(
                source,
                temporary,
                expected_size=expected_size,
            )
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
            return _file_entry(project_id, relative, group, digest, size)
        except (OSError, ValueError) as exc:
            last_error = exc
            _unlink_if_present(temporary)
    raise BackupProjectFileUnavailable(
        "A project file did not stabilize for capture."
    ) from last_error


def _copy_one_stable_file(
    source: Path,
    destination: Path,
    *,
    expected_size: int | None,
) -> tuple[str, int]:
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor = -1
    try:
        initial = os.fstat(source_descriptor)
        current = source.lstat()
        if (
            not stat.S_ISREG(initial.st_mode)
            or (initial.st_dev, initial.st_ino) != (current.st_dev, current.st_ino)
            or (expected_size is not None and initial.st_size != expected_size)
        ):
            raise ValueError("The project file changed before capture.")
        destination_descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(source_descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            _write_all(destination_descriptor, chunk)
            digest.update(chunk)
            size += len(chunk)
        final = os.fstat(source_descriptor)
        path_final = source.lstat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, field) != getattr(final, field) for field in stable_fields) or any(
            getattr(final, field) != getattr(path_final, field) for field in stable_fields
        ):
            raise ValueError("The project file changed during capture.")
        if size != final.st_size:
            raise ValueError("The project file copy is incomplete.")
        os.fchmod(destination_descriptor, 0o400)
        os.fsync(destination_descriptor)
        return digest.hexdigest(), size
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def stable_workspace_bytes(reader: Callable[[], bytes]) -> bytes:
    previous: tuple[str, int, bytes] | None = None
    last_error: BaseException | None = None
    for _ in range(BACKUP_STABLE_READ_ATTEMPTS):
        try:
            data = reader()
            if not isinstance(data, bytes) or len(data) > CHAT_ARTIFACT_MAX_FILE_BYTES:
                raise ValueError("kept-file bytes exceed their existing product limit")
        except (OSError, StateUnavailable, ValueError) as exc:
            last_error = exc
            previous = None
            continue
        observed = (hashlib.sha256(data).hexdigest(), len(data), data)
        if previous is not None and observed == previous:
            return data
        previous = observed
    raise BackupProjectFileUnavailable("A kept project file did not stabilize.") from last_error


def write_bytes_entry(
    data: bytes,
    project_root: Path,
    project_id: str,
    relative: PurePosixPath,
    *,
    group: Literal["kept_artifact", "legacy_kept_result_view"],
) -> BackupFileEntry:
    destination = _destination_path(project_root, relative)
    _prepare_destination_parent(project_root, destination.parent)
    descriptor = os.open(
        destination,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        _write_all(descriptor, data)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(destination.parent)
    return _file_entry(
        project_id,
        relative,
        group,
        hashlib.sha256(data).hexdigest(),
        len(data),
    )


def _file_entry(
    project_id: str,
    relative: PurePosixPath,
    group: ProjectFileGroup,
    sha256: str,
    size: int,
) -> BackupFileEntry:
    return BackupFileEntry(
        archive_path=(PurePosixPath("projects") / project_id / relative).as_posix(),
        source_relative_path=relative.as_posix(),
        group=group,
        sha256=sha256,
        size_bytes=size,
    )


def _destination_path(project_root: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("project backup destination must be one safe relative path")
    destination = project_root.joinpath(*relative.parts)
    if not destination.is_relative_to(project_root):
        raise ValueError("project backup destination escaped its capture root")
    return destination


def _prepare_destination_parent(project_root: Path, parent: Path) -> None:
    relative = parent.relative_to(project_root)
    current = project_root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            metadata = current.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise ValueError("project backup destination ancestry is unsafe") from None


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short backup file write")
        remaining = remaining[written:]


def discard_failed_project_capture(capture_root: Path, project_root: Path) -> None:
    expected_parent = capture_root / "projects"
    if project_root.parent != expected_parent or not project_root.name:
        raise RuntimeError("refusing to discard an unbound project capture")
    try:
        metadata = project_root.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("refusing to discard an unsafe project capture")
    shutil.rmtree(project_root)


def _unlink_if_present(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    pending = [root]
    directories: list[Path] = []
    observed_entries = 0
    while pending:
        directory = pending.pop()
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("project backup output contains an unsafe directory")
        directories.append(directory)
        entries = list(directory.iterdir())
        observed_entries += len(entries)
        if observed_entries > BACKUP_INVENTORY_MAX_ENTRIES:
            raise ValueError("project backup output exceeds its entry bound")
        for entry in entries:
            item = entry.lstat()
            if stat.S_ISDIR(item.st_mode):
                pending.append(entry)
            elif not stat.S_ISREG(item.st_mode):
                raise ValueError("project backup output contains an unsafe entry")
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)


__all__ = [
    "BackupProjectFileUnavailable",
    "capture_chat_files",
    "discard_failed_project_capture",
    "fact_backup_sources",
    "fsync_directory",
    "fsync_tree",
    "stable_copy_entry",
    "stable_copy_fact_entry",
    "stable_workspace_bytes",
    "write_bytes_entry",
]
