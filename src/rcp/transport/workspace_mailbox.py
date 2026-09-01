from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from rcp.transport.run_stage import RemoteRunStage

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
TURN_HANDOFF_FILES = ("patch.json", "watch.json", "messages.json", "lifecycle.json")


@dataclass(frozen=True, slots=True)
class RunStageMailbox:
    """One file mailbox over a local workspace or an existing SSH run stage."""

    workspace: Path
    remote_stage: RemoteRunStage | None = None

    @classmethod
    def for_stage(
        cls,
        *,
        local_stage: Path | None,
        remote_stage: RemoteRunStage | None,
    ) -> RunStageMailbox:
        if (local_stage is None) == (remote_stage is None):
            raise ValueError("exactly one task stage must be selected")
        if remote_stage is not None:
            return cls(workspace=Path(str(remote_stage.workspace)), remote_stage=remote_stage)
        assert local_stage is not None
        return cls(workspace=Path(os.path.abspath(local_stage)))

    def entry_names(self) -> list[str]:
        """List every direct entry so unsafe stale mailbox state stays visible."""

        if self.remote_stage is not None:
            return self.remote_stage.list_workspace_entries()
        descriptor = _open_directory(self.workspace)
        try:
            return sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)

    def read_text(self, name: str, *, max_bytes: int | None = None) -> str:
        """Read one direct regular UTF-8 file, optionally enforcing a byte limit."""

        name = _safe_name(name)
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("mailbox byte limit must not be negative")
        if self.remote_stage is not None:
            return self.remote_stage.read_workspace_text(name, max_bytes=max_bytes)

        directory = _open_directory(self.workspace)
        descriptor: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(name, flags, dir_fd=directory)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"mailbox entry is not a regular file: {name}")
            limit = -1 if max_bytes is None else max_bytes + 1
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                content = stream.read(limit)
            if max_bytes is not None and len(content) > max_bytes:
                raise ValueError(f"mailbox file exceeds {max_bytes} bytes: {name}")
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"mailbox file is not UTF-8 text: {name}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory)

    def write_text(self, name: str, content: str) -> None:
        """Atomically replace one direct regular UTF-8 mailbox file."""

        name = _safe_name(name)
        if self.remote_stage is not None:
            self.remote_stage.write_workspace_text(name, content)
            return
        _atomic_text(self.workspace, name, content, replace=True, mode=0o600)

    def remove(self, name: str, *, missing_ok: bool = True) -> None:
        """Remove one exact direct entry, raising if an unsafe directory occupies it."""

        name = _safe_name(name)
        if self.remote_stage is not None:
            if not missing_ok and name not in self.remote_stage.list_workspace_entries():
                raise FileNotFoundError(f"remote workspace file is absent: {name}")
            self.remote_stage.remove_workspace_file(name)
            return

        directory = _open_directory(self.workspace)
        try:
            try:
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                if missing_ok:
                    return
                raise
            if stat.S_ISDIR(info.st_mode):
                raise ValueError(f"mailbox entry is an unsafe directory: {name}")
            os.unlink(name, dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(directory)

    def remove_if_sha256(self, name: str, expected_sha256: str) -> bool:
        """Remove one direct regular file only if it is still the snapshotted file."""

        name = _safe_name(name)
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("expected mailbox digest must be lowercase SHA-256")
        if self.remote_stage is not None:
            return self.remote_stage.remove_workspace_file_if_sha256(name, expected_sha256)

        directory = _open_directory(self.workspace)
        descriptor: int | None = None
        quarantine = f".rcp-consume-{uuid.uuid4().hex}-{name}"
        quarantined = False
        try:
            try:
                before = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(f"mailbox entry is not a regular file: {name}")
                os.rename(
                    name,
                    quarantine,
                    src_dir_fd=directory,
                    dst_dir_fd=directory,
                )
                quarantined = True
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(quarantine, flags, dir_fd=directory)
            except FileNotFoundError:
                return False
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"mailbox entry is not a regular file: {name}")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                _restore_quarantined_entry(directory, quarantine, name)
                quarantined = False
                return False
            os.unlink(quarantine, dir_fd=directory)
            quarantined = False
            os.fsync(directory)
            return True
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if quarantined:
                with suppress(OSError):
                    _restore_quarantined_entry(directory, quarantine, name)
            os.close(directory)

    def stage_text_input(self, label: str, content: str) -> str:
        """Stage one new immutable source file and return its execution-host path.

        Remote inputs are finalized here so the returned path is immediately usable.
        Any other inputs already queued on the same stage join that atomic input batch.
        """

        label = _safe_name(label)
        if self.remote_stage is not None:
            with tempfile.TemporaryDirectory(prefix="rcp-mailbox-input-") as temporary:
                source = Path(temporary) / label
                source.write_bytes(content.encode("utf-8"))
                source.chmod(0o400)
                remote_path = self.remote_stage.put_file(source, label)
                self.remote_stage.finalize_inputs()
            return remote_path

        descriptor = _open_directory(self.workspace)
        os.close(descriptor)
        inputs = self.workspace / "inputs"
        if os.path.lexists(inputs):
            if inputs.is_symlink() or not inputs.is_dir():
                raise ValueError("task input directory is unsafe")
        else:
            self.workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
            inputs.mkdir(mode=0o700)
        _atomic_text(inputs, label, content, replace=False, mode=0o400)
        return str(inputs / label)


def clear_turn_handoff_files(mailbox: RunStageMailbox) -> None:
    """Fail closed while clearing every reusable per-turn handoff channel."""

    for name in TURN_HANDOFF_FILES:
        mailbox.remove(name)


def _safe_name(value: str) -> str:
    if (
        not value
        or len(value) > 255
        or Path(value).name != value
        or _SAFE_NAME.fullmatch(value) is None
    ):
        raise ValueError("mailbox file name contains unsupported characters")
    return value


def _restore_quarantined_entry(directory: int, quarantine: str, name: str) -> None:
    """Restore a consumed file without ever overwriting newer bytes."""

    try:
        os.link(
            quarantine,
            name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
    except FileExistsError:
        # A writer published a newer file while the snapshot was inspected. Keep
        # the quarantined snapshot as a receipt rather than replacing either file.
        os.fsync(directory)
        return
    os.unlink(quarantine, dir_fd=directory)
    os.fsync(directory)


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise OSError(f"run workspace {path} is unavailable: {exc}") from exc


def _atomic_text(
    directory: Path,
    name: str,
    content: str,
    *,
    replace: bool,
    mode: int,
) -> None:
    encoded = content.encode("utf-8")
    directory_fd = _open_directory(directory)
    temporary = f".rcp-mailbox-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None:
            if not replace:
                raise ValueError(f"immutable task input already exists: {name}")
            if not stat.S_ISREG(current.st_mode):
                raise ValueError(f"mailbox target is not a regular file: {name}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, mode, dir_fd=directory_fd)
        view = memoryview(encoded)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        temporary = ""
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory_fd)
        os.close(directory_fd)
