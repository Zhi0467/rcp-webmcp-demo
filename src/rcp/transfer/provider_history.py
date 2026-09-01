"""Best-effort provider-native history capture for one project transfer."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp.limits import (
    PROJECT_TRANSFER_COPY_BUFFER_BYTES,
    PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES,
)
from rcp.sources import ConversationIndexer
from rcp.transfer.archive import TransferArchiveDiagnostic, TransferArchiveEntry


class TransferProviderHistoryCapture(BaseModel):
    """Content-addressed provider originals plus bounded omission accounting."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )

    entries: tuple[TransferArchiveEntry, ...]
    diagnostics: tuple[TransferArchiveDiagnostic, ...]
    selected_files: int = Field(ge=0, le=PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES)
    skipped_files: int = Field(ge=0, le=PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES)
    unreadable_files: int = Field(ge=0, le=PROJECT_TRANSFER_INVENTORY_MAX_ENTRIES)
    payload_size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_capture(self) -> TransferProviderHistoryCapture:
        paths = [entry.archive_path for entry in self.entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("provider-history archive paths must be sorted and unique")
        if any(entry.group != "provider_history" for entry in self.entries):
            raise ValueError("provider-history capture contains another packet's bytes")
        if self.selected_files < len(self.entries):
            raise ValueError("provider-history selected count is smaller than its entries")
        if sum(entry.size_bytes for entry in self.entries) != self.payload_size_bytes:
            raise ValueError("provider-history byte total differs from its entries")
        codes = [diagnostic.code for diagnostic in self.diagnostics]
        if codes != sorted(codes) or len(codes) != len(set(codes)):
            raise ValueError("provider-history diagnostics must be sorted and unique")
        return self


def capture_provider_history(
    indexer: ConversationIndexer,
    capture_root: Path,
) -> TransferProviderHistoryCapture:
    """Capture matched provider originals without turning omissions into authority."""

    try:
        capture_root.mkdir(mode=0o700)
    except OSError as exc:
        raise ValueError("provider-history capture root must be one new private directory") from exc
    try:
        index = indexer.build(cache_remote_sources=False)
        allowed = {repository.alias for repository in indexer.manifest.repositories}
        entries: dict[str, TransferArchiveEntry] = {}
        selected_files = 0
        rewritten = 0
        unreadable = 0
        for session in index.sessions:
            if session.provider == "app_chat" or session.truth_repository not in allowed:
                continue
            with ExitStack() as source_stack:
                try:
                    original = source_stack.enter_context(indexer.original_source(session))
                except (OSError, UnicodeError, ValueError):
                    unreadable += 1
                    continue
                archive_path = (
                    PurePosixPath("provider-history") / session.provider / (original.content_sha256)
                )
                key = archive_path.as_posix()
                created = key not in entries
                if created:
                    entry = _capture_original(
                        capture_root,
                        original.path,
                        archive_path,
                        expected_sha256=original.content_sha256,
                        expected_size=original.size_bytes,
                    )
                else:
                    entry = entries[key]
                copied = capture_root.joinpath(*archive_path.parts)
                try:
                    if (
                        indexer.original_repository_alias(session, copied)
                        != session.truth_repository
                    ):
                        if created:
                            copied.unlink(missing_ok=True)
                        rewritten += 1
                        continue
                except (OSError, UnicodeError, ValueError):
                    if created:
                        copied.unlink(missing_ok=True)
                    unreadable += 1
                    continue
                entries[key] = entry
                selected_files += 1

        skipped = index.unmatched_files + index.malformed_files + rewritten
        unavailable = len(index.source_errors) + unreadable
        ordered = tuple(entries[path] for path in sorted(entries))
        diagnostics = _capture_diagnostics(
            unmatched=index.unmatched_files,
            malformed=index.malformed_files,
            source_errors=len(index.source_errors),
            rewritten=rewritten,
            unreadable=unreadable,
        )
        _fsync_tree(capture_root)
        return TransferProviderHistoryCapture(
            entries=ordered,
            diagnostics=diagnostics,
            selected_files=selected_files,
            skipped_files=skipped,
            unreadable_files=unavailable,
            payload_size_bytes=sum(entry.size_bytes for entry in ordered),
        )
    except BaseException:
        _discard_new_capture_root(capture_root)
        raise


def _capture_diagnostics(
    *,
    unmatched: int,
    malformed: int,
    source_errors: int,
    rewritten: int,
    unreadable: int,
) -> tuple[TransferArchiveDiagnostic, ...]:
    values = (
        ("provider_history_malformed", malformed, "malformed provider conversation file"),
        ("provider_history_rewritten", rewritten, "rewritten provider conversation file"),
        (
            "provider_history_source_unavailable",
            source_errors,
            "configured provider source root that was unavailable",
        ),
        ("provider_history_unmatched", unmatched, "unmatched provider conversation file"),
        ("provider_history_unreadable", unreadable, "unreadable provider conversation file"),
    )
    return tuple(
        TransferArchiveDiagnostic(
            code=code,
            message=f"{count} {label}{'' if count == 1 else 's'} omitted from transfer.",
        )
        for code, count, label in values
        if count
    )


def _capture_original(
    capture_root: Path,
    source: Path,
    archive_path: PurePosixPath,
    *,
    expected_sha256: str,
    expected_size: int,
) -> TransferArchiveEntry:
    entry_path = TransferArchiveEntry(
        archive_path=archive_path.as_posix(),
        group="provider_history",
        sha256=expected_sha256,
        size_bytes=expected_size,
    ).archive_path
    destination = capture_root.joinpath(*PurePosixPath(entry_path).parts)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.transfer-{uuid.uuid4().hex}")
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        source_metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_metadata.st_mode):
            raise ValueError("provider history source is not a regular file")
        destination_descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        while True:
            chunk = os.read(source_descriptor, PROJECT_TRANSFER_COPY_BUFFER_BYTES)
            if not chunk:
                break
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise OSError("short provider-history transfer write")
                remaining = remaining[written:]
            digest.update(chunk)
            size += len(chunk)
        if (size, digest.hexdigest()) != (expected_size, expected_sha256):
            raise ValueError("provider history changed during transfer capture")
        os.fchmod(destination_descriptor, 0o400)
        os.fsync(destination_descriptor)
    except BaseException:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
            destination_descriptor = -1
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    return TransferArchiveEntry(
        archive_path=entry_path,
        group="provider_history",
        sha256=digest.hexdigest(),
        size_bytes=size,
    )


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
        raise RuntimeError("refusing to discard an unsafe provider-history capture")
    shutil.rmtree(capture_root)


__all__ = ["TransferProviderHistoryCapture", "capture_provider_history"]
