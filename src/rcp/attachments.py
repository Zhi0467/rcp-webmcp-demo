from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field

from rcp.limits import (
    CHAT_ATTACHMENT_MAX_COUNT,
    CHAT_ATTACHMENT_MAX_FILE_BYTES,
    CHAT_ATTACHMENT_MAX_TOTAL_BYTES,
    RUN_STAGE_RETENTION_DAYS,
)
from rcp.transport import RemoteRunStage


class ChatAttachmentDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attachment_id: str
    name: str
    media_type: str
    size: int = Field(ge=0, le=CHAT_ATTACHMENT_MAX_FILE_BYTES)
    expires_at: str


class ChatAttachmentUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attachment_set_id: str
    attachment: ChatAttachmentDescriptor


class ClaimedAttachmentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attachment_batch_id: str
    attachments: list[ChatAttachmentDescriptor]


class _StoredAttachment(ChatAttachmentDescriptor):
    sha256: str
    staged_name: str


class _StoredSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attachment_set_id: str
    project_id: str
    chat_id: str
    client_id: str
    created_at: str
    expires_at: str
    claimed_by: str | None = None
    attachments: list[_StoredAttachment] = Field(default_factory=list)


_SOURCE_MEDIA_TYPES = {
    ".c": "text/x-c",
    ".cc": "text/x-c++",
    ".cpp": "text/x-c++",
    ".cs": "text/x-csharp",
    ".css": "text/css",
    ".fish": "text/x-shellscript",
    ".go": "text/x-go",
    ".h": "text/x-c",
    ".hpp": "text/x-c++",
    ".java": "text/x-java-source",
    ".js": "text/javascript",
    ".jsx": "text/jsx",
    ".kt": "text/x-kotlin",
    ".kts": "text/x-kotlin",
    ".lua": "text/x-lua",
    ".mjs": "text/javascript",
    ".mm": "text/x-objective-c++",
    ".php": "text/x-php",
    ".py": "text/x-python",
    ".r": "text/x-r",
    ".rb": "text/x-ruby",
    ".rs": "text/x-rust",
    ".scala": "text/x-scala",
    ".sh": "text/x-shellscript",
    ".sql": "application/sql",
    ".swift": "text/x-swift",
    ".toml": "application/toml",
    ".ts": "text/typescript",
    ".tsx": "text/tsx",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zsh": "text/x-shellscript",
}
_ALLOWED_EXTENSIONS = {
    ".csv": "text/csv",
    ".htm": "text/html",
    ".html": "text/html",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".markdown": "text/markdown",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".webp": "image/webp",
    **_SOURCE_MEDIA_TYPES,
}
_STORE_LOCK = threading.RLock()
_ATTACHMENT_METADATA_MAX_BYTES = 128 * 1024
_ATTACHMENT_SET_MIGRATION_MAX_COUNT = 10_000


@dataclass(frozen=True)
class _AttachmentMetadataRewrite:
    destination: Path
    expected: bytes
    replacement: bytes


@dataclass(frozen=True)
class _AttachmentProjectIdentityMigration:
    rewrites: tuple[_AttachmentMetadataRewrite, ...]


@dataclass(frozen=True)
class AttachmentRecoverySet:
    """One complete temporary attachment set that recovery may still consume."""

    attachment_set_id: str
    project_id: str
    root: Path
    claimed_by: str | None


class ChatAttachmentStore:
    """Bounded temporary ingress for one chat turn's human-provided files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        # Runs reconstruct the lightweight store from data_dir. One process-wide
        # lock keeps ingress, claim, sweep, and staging atomic across those handles.
        self._lock = _STORE_LOCK

    def prepare_project_identity_migration(
        self,
        old_project_id: str,
        canonical_project_id: str,
    ) -> _AttachmentProjectIdentityMigration:
        """Validate and prepare bounded metadata-only project-id rewrites."""

        if old_project_id == canonical_project_id or not self.root.exists():
            return _AttachmentProjectIdentityMigration(rewrites=())
        rewrites: list[_AttachmentMetadataRewrite] = []
        with self._lock:
            candidates: list[Path] = []
            for candidate in self.root.iterdir():
                if candidate.is_symlink() or not candidate.is_dir():
                    continue
                candidates.append(candidate)
                if len(candidates) > _ATTACHMENT_SET_MIGRATION_MAX_COUNT:
                    raise ValueError("Too many attachment sets to migrate safely.")
            for candidate in candidates:
                set_id = _canonical_uuid(candidate.name, "saved attachment set id")
                metadata = candidate / "metadata.json"
                expected = _read_bounded_metadata(metadata)
                try:
                    stored = _StoredSet.model_validate_json(expected)
                except ValueError as exc:
                    raise ValueError("Saved attachment metadata is invalid.") from exc
                if stored.attachment_set_id != set_id:
                    raise ValueError("Saved attachment metadata names a different set.")
                self._verify(stored)
                if stored.project_id != old_project_id:
                    continue
                replacement = _stored_set_bytes(
                    stored.model_copy(update={"project_id": canonical_project_id})
                )
                rewrites.append(
                    _AttachmentMetadataRewrite(
                        destination=metadata,
                        expected=expected,
                        replacement=replacement,
                    )
                )
        return _AttachmentProjectIdentityMigration(rewrites=tuple(rewrites))

    def apply_project_identity_migration(
        self,
        migration: _AttachmentProjectIdentityMigration,
    ) -> None:
        """Apply a prepared migration without weakening chat or client scope."""

        with self._lock:
            for rewrite in migration.rewrites:
                current = _read_bounded_metadata(rewrite.destination)
                if current == rewrite.replacement:
                    continue
                if current != rewrite.expected:
                    raise ValueError("Saved attachment metadata changed during migration.")
                _atomic_write_metadata(rewrite.destination, rewrite.replacement)

    def add(
        self,
        *,
        project_id: str,
        chat_id: str,
        client_id: str,
        filename: str,
        media_type: str | None,
        source: BinaryIO,
        attachment_set_id: str | None = None,
    ) -> ChatAttachmentUpload:
        chat_id = _canonical_uuid(chat_id, "chat_id")
        client_id = _canonical_uuid(client_id, "client_id")
        set_id = (
            _canonical_uuid(attachment_set_id, "attachment_set_id")
            if attachment_set_id
            else str(uuid.uuid4())
        )
        name, normalized_media_type = _validated_name_and_type(filename, media_type)
        with self._lock:
            self.sweep()
            stored = self._load(set_id) if self._metadata_path(set_id).exists() else None
            if stored is None:
                now = datetime.now(UTC)
                stored = _StoredSet(
                    attachment_set_id=set_id,
                    project_id=project_id,
                    chat_id=chat_id,
                    client_id=client_id,
                    created_at=now.isoformat(),
                    expires_at=(now + timedelta(days=RUN_STAGE_RETENTION_DAYS)).isoformat(),
                )
                self._set_path(set_id).mkdir(mode=0o700, parents=True)
                (self._set_path(set_id) / "files").mkdir(mode=0o700)
            self._require_scope(stored, project_id, chat_id, client_id)
            if stored.claimed_by is not None:
                raise ValueError("This attachment set has already been sent.")
            if len(stored.attachments) >= CHAT_ATTACHMENT_MAX_COUNT:
                raise ValueError(
                    f"A chat turn can attach at most {CHAT_ATTACHMENT_MAX_COUNT} files."
                )

            attachment_id = str(uuid.uuid4())
            extension = Path(name).suffix.casefold()
            staged_name = f"{len(stored.attachments):02d}-{attachment_id}{extension}"
            destination = self._set_path(set_id) / "files" / staged_name
            digest = hashlib.sha256()
            size = 0
            descriptor: ChatAttachmentDescriptor | None = None
            try:
                with destination.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        size += len(chunk)
                        if size > CHAT_ATTACHMENT_MAX_FILE_BYTES:
                            raise ValueError(
                                f"Each attachment must be at most "
                                f"{CHAT_ATTACHMENT_MAX_FILE_BYTES // (1024 * 1024)} MiB."
                            )
                        digest.update(chunk)
                        output.write(chunk)
                total = sum(item.size for item in stored.attachments) + size
                if total > CHAT_ATTACHMENT_MAX_TOTAL_BYTES:
                    raise ValueError(
                        f"Attachments for one turn must total at most "
                        f"{CHAT_ATTACHMENT_MAX_TOTAL_BYTES // (1024 * 1024)} MiB."
                    )
                _validate_signature(destination, normalized_media_type)
                descriptor = ChatAttachmentDescriptor(
                    attachment_id=attachment_id,
                    name=name,
                    media_type=normalized_media_type,
                    size=size,
                    expires_at=stored.expires_at,
                )
                stored.attachments.append(
                    _StoredAttachment(
                        **descriptor.model_dump(),
                        sha256=digest.hexdigest(),
                        staged_name=staged_name,
                    )
                )
                self._write(stored)
            except BaseException:
                destination.unlink(missing_ok=True)
                if not stored.attachments:
                    shutil.rmtree(self._set_path(set_id), ignore_errors=True)
                raise
            assert descriptor is not None
            return ChatAttachmentUpload(attachment_set_id=set_id, attachment=descriptor)

    def remove(
        self,
        *,
        project_id: str,
        chat_id: str,
        client_id: str,
        attachment_set_id: str,
        attachment_id: str,
    ) -> None:
        chat_id = _canonical_uuid(chat_id, "chat_id")
        client_id = _canonical_uuid(client_id, "client_id")
        set_id = _canonical_uuid(attachment_set_id, "attachment_set_id")
        attachment_id = _canonical_uuid(attachment_id, "attachment_id")
        with self._lock:
            self.sweep()
            stored = self._load(set_id)
            self._require_scope(stored, project_id, chat_id, client_id)
            if stored.claimed_by is not None:
                raise ValueError("Sent attachments cannot be removed.")
            match = next(
                (item for item in stored.attachments if item.attachment_id == attachment_id), None
            )
            if match is None:
                raise ValueError("Attachment not found.")
            (self._set_path(set_id) / "files" / match.staged_name).unlink()
            stored.attachments.remove(match)
            if stored.attachments:
                self._write(stored)
            else:
                shutil.rmtree(self._set_path(set_id))

    def claim(
        self,
        *,
        project_id: str,
        chat_id: str,
        client_id: str,
        attachment_set_id: str,
        operation_id: str,
    ) -> ClaimedAttachmentBatch:
        chat_id = _canonical_uuid(chat_id, "chat_id")
        client_id = _canonical_uuid(client_id, "client_id")
        set_id = _canonical_uuid(attachment_set_id, "attachment_set_id")
        operation_id = _canonical_uuid(operation_id, "operation_id")
        with self._lock:
            self.sweep()
            stored = self._load(set_id)
            self._require_scope(stored, project_id, chat_id, client_id)
            if not stored.attachments:
                raise ValueError("The attachment set is empty.")
            if stored.claimed_by is not None and stored.claimed_by != operation_id:
                raise ValueError("This attachment set has already been sent.")
            now = datetime.now(UTC)
            stored.claimed_by = operation_id
            stored.expires_at = (now + timedelta(days=RUN_STAGE_RETENTION_DAYS)).isoformat()
            stored.attachments = [
                item.model_copy(update={"expires_at": stored.expires_at})
                for item in stored.attachments
            ]
            self._verify(stored)
            self._write(stored)
            return ClaimedAttachmentBatch(
                attachment_batch_id=set_id,
                attachments=[_public_descriptor(item) for item in stored.attachments],
            )

    def release(self, attachment_set_id: str, operation_id: str) -> None:
        """Return a just-claimed set to ingress if its task record was not created."""

        set_id = _canonical_uuid(attachment_set_id, "attachment_set_id")
        operation_id = _canonical_uuid(operation_id, "operation_id")
        with self._lock:
            stored = self._load(set_id)
            if stored.claimed_by != operation_id:
                raise ValueError("The attachment set is not claimed by this task.")
            stored.claimed_by = None
            self._write(stored)

    def stage(
        self,
        attachment_batch_id: str,
        expected: list[ChatAttachmentDescriptor],
        *,
        local_stage: Path | None,
        remote_stage: RemoteRunStage | None,
    ) -> list[dict[str, object]]:
        if (local_stage is None) == (remote_stage is None):
            raise ValueError("exactly one chat task stage must be selected")
        set_id = _canonical_uuid(attachment_batch_id, "attachment_batch_id")
        with self._lock:
            self.sweep()
            stored = self._load(set_id)
            if stored.claimed_by is None:
                raise ValueError("The attachment set was not claimed by a chat turn.")
            actual = [_public_descriptor(item) for item in stored.attachments]
            if actual != expected:
                raise ValueError("The saved attachment batch metadata does not match this task.")
            self._verify(stored)
            source = self._set_path(set_id) / "files"
            label = f"chat-attachments-v1-{set_id}"
            if remote_stage is not None:
                assert remote_stage.root is not None
                root = Path(str(remote_stage.root / "inputs" / label))
                remote_stage.put_directory(source, label, reuse=True)
            else:
                assert local_stage is not None
                root = local_stage / "inputs" / label
                _stage_local_directory(source, root, stored)
            return [
                {
                    "path": str(root / item.staged_name),
                    "name": item.name,
                    "media_type": item.media_type,
                    "size": item.size,
                }
                for item in stored.attachments
            ]

    def sweep(self) -> None:
        if not self.root.exists():
            return
        now = datetime.now(UTC)
        for candidate in self.root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            try:
                stored = self._load(candidate.name)
                expired = datetime.fromisoformat(stored.expires_at) <= now
            except (OSError, ValueError):
                try:
                    modified = datetime.fromtimestamp(candidate.stat().st_mtime, UTC)
                except OSError:
                    continue
                expired = modified <= now - timedelta(days=RUN_STAGE_RETENTION_DAYS)
            if expired:
                shutil.rmtree(candidate)

    def _verify(self, stored: _StoredSet) -> None:
        if len(stored.attachments) > CHAT_ATTACHMENT_MAX_COUNT:
            raise ValueError("The attachment batch exceeds its file-count limit.")
        total = 0
        expected_names = {item.staged_name for item in stored.attachments}
        files = self._set_path(stored.attachment_set_id) / "files"
        actual_names = {item.name for item in files.iterdir()}
        if actual_names != expected_names:
            raise ValueError("The saved attachment batch is incomplete.")
        for item in stored.attachments:
            path = files / item.staged_name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size != item.size:
                raise ValueError(f"Attachment {item.name!r} changed after upload.")
            total += item.size
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != item.sha256:
                raise ValueError(f"Attachment {item.name!r} changed after upload.")
        if total > CHAT_ATTACHMENT_MAX_TOTAL_BYTES:
            raise ValueError("The attachment batch exceeds its total-size limit.")

    def _require_scope(
        self, stored: _StoredSet, project_id: str, chat_id: str, client_id: str
    ) -> None:
        if (stored.project_id, stored.chat_id, stored.client_id) != (
            project_id,
            chat_id,
            client_id,
        ):
            raise ValueError("The attachment set does not belong to this chat and client.")

    def _set_path(self, set_id: str) -> Path:
        return self.root / set_id

    def _metadata_path(self, set_id: str) -> Path:
        return self._set_path(set_id) / "metadata.json"

    def _load(self, set_id: str) -> _StoredSet:
        try:
            return _StoredSet.model_validate_json(self._metadata_path(set_id).read_text())
        except FileNotFoundError as exc:
            raise ValueError("Attachment set not found or expired.") from exc

    def _write(self, stored: _StoredSet) -> None:
        destination = self._metadata_path(stored.attachment_set_id)
        _atomic_write_metadata(destination, _stored_set_bytes(stored))


@contextmanager
def checkpoint_attachment_sets(root: Path) -> Iterator[tuple[AttachmentRecoverySet, ...]]:
    """Hold attachment ingress still while inventorying exact recoverable set roots."""

    with _STORE_LOCK:
        if not root.exists():
            yield ()
            return
        try:
            root_metadata = root.lstat()
        except OSError as exc:
            raise ValueError("the attachment checkpoint root is unavailable") from exc
        if not stat.S_ISDIR(root_metadata.st_mode) or root.is_symlink():
            raise ValueError("the attachment checkpoint root is not an ordinary directory")

        store = ChatAttachmentStore(root)
        inventory: list[AttachmentRecoverySet] = []
        try:
            candidates = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError("the attachment checkpoint root cannot be listed") from exc
        if len(candidates) > _ATTACHMENT_SET_MIGRATION_MAX_COUNT:
            raise ValueError("too many attachment sets to checkpoint safely")
        for candidate in candidates:
            set_id = _canonical_uuid(candidate.name, "saved attachment set id")
            try:
                candidate_metadata = candidate.lstat()
                metadata = candidate / "metadata.json"
                metadata_status = metadata.lstat()
                files = candidate / "files"
                files_status = files.lstat()
            except OSError as exc:
                raise ValueError("a recovery-critical attachment set is incomplete") from exc
            if (
                not stat.S_ISDIR(candidate_metadata.st_mode)
                or candidate.is_symlink()
                or not stat.S_ISREG(metadata_status.st_mode)
                or metadata.is_symlink()
                or metadata_status.st_size > _ATTACHMENT_METADATA_MAX_BYTES
                or not stat.S_ISDIR(files_status.st_mode)
                or files.is_symlink()
            ):
                raise ValueError("a recovery-critical attachment set has an unsafe shape")
            try:
                children = {item.name for item in candidate.iterdir()}
            except OSError as exc:
                raise ValueError("a recovery-critical attachment set cannot be listed") from exc
            if children != {"metadata.json", "files"}:
                raise ValueError("a recovery-critical attachment set has unknown entries")
            stored = store._load(set_id)
            if stored.attachment_set_id != set_id:
                raise ValueError("saved attachment metadata names a different set")
            store._verify(stored)
            inventory.append(
                AttachmentRecoverySet(
                    attachment_set_id=set_id,
                    project_id=stored.project_id,
                    root=candidate,
                    claimed_by=stored.claimed_by,
                )
            )
        yield tuple(inventory)


def _stored_set_bytes(stored: _StoredSet) -> bytes:
    return json.dumps(stored.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")


def _read_bounded_metadata(path: Path) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError("Saved attachment metadata is unavailable.") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size > _ATTACHMENT_METADATA_MAX_BYTES:
        raise ValueError("Saved attachment metadata is invalid.")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError("Saved attachment metadata is unavailable.") from exc


def _atomic_write_metadata(destination: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".metadata-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_uuid(value: str, label: str) -> str:
    try:
        normalized = str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc
    if normalized != value:
        raise ValueError(f"{label} must be a canonical UUID")
    return normalized


def _public_descriptor(item: _StoredAttachment) -> ChatAttachmentDescriptor:
    return ChatAttachmentDescriptor(
        attachment_id=item.attachment_id,
        name=item.name,
        media_type=item.media_type,
        size=item.size,
        expires_at=item.expires_at,
    )


def _validated_name_and_type(filename: str, supplied: str | None) -> tuple[str, str]:
    if not filename or filename != Path(filename).name or "\x00" in filename:
        raise ValueError("Attachment filename is invalid.")
    if len(filename.encode("utf-8")) > 255:
        raise ValueError("Attachment filename is too long.")
    extension = Path(filename).suffix.casefold()
    media_type = _ALLOWED_EXTENSIONS.get(extension)
    if media_type is None:
        raise ValueError("This file type is not supported for chat attachments.")
    supplied = (supplied or "").split(";", 1)[0].strip().casefold()
    if supplied.startswith(("audio/", "video/")):
        raise ValueError("Audio and video files are not supported as chat attachments.")
    return filename, media_type


def _validate_signature(path: Path, media_type: str) -> None:
    with path.open("rb") as source:
        prefix = source.read(16)
    valid = True
    if media_type == "image/png":
        valid = prefix.startswith(b"\x89PNG\r\n\x1a\n")
    elif media_type == "image/jpeg":
        valid = prefix.startswith(b"\xff\xd8\xff")
    elif media_type == "image/webp":
        valid = prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP"
    elif media_type == "application/pdf":
        valid = prefix.startswith(b"%PDF-")
    if not valid:
        raise ValueError("Attachment bytes do not match the file type.")
    if media_type.startswith("text/") or media_type in {
        "application/json",
        "application/sql",
        "application/toml",
        "application/xml",
        "application/yaml",
        "image/svg+xml",
    }:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text attachments must be valid UTF-8.") from exc
        if "\x00" in text:
            raise ValueError("Text attachments cannot contain NUL bytes.")
        if media_type == "application/json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("JSON attachments must contain valid JSON.") from exc
        if media_type == "image/svg+xml":
            try:
                root = ET.fromstring(text)
            except ET.ParseError as exc:
                raise ValueError("SVG attachments must contain valid XML.") from exc
            if root.tag.rsplit("}", 1)[-1].casefold() != "svg":
                raise ValueError("SVG attachments must have an svg root element.")


def _stage_local_directory(source: Path, target: Path, stored: _StoredSet) -> None:
    if os.path.lexists(target):
        if target.is_symlink() or not target.is_dir():
            raise ValueError("Saved local attachment input is unsafe.")
        _verify_staged_tree(target, stored)
        return
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for item in stored.attachments:
            shutil.copyfile(source / item.staged_name, temporary / item.staged_name)
        for item in temporary.iterdir():
            item.chmod(0o400)
        temporary.chmod(0o500)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.chmod(0o700)
            shutil.rmtree(temporary)
    _verify_staged_tree(target, stored)


def _verify_staged_tree(root: Path, stored: _StoredSet) -> None:
    expected = {item.staged_name: item for item in stored.attachments}
    actual = {item.name: item for item in root.iterdir()}
    if set(actual) != set(expected):
        raise ValueError("Saved local attachment batch is incomplete.")
    for name, path in actual.items():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
            raise ValueError("Saved local attachment input is not immutable.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if info.st_size != expected[name].size or digest != expected[name].sha256:
            raise ValueError("Saved local attachment input changed after staging.")
