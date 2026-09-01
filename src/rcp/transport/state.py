from __future__ import annotations

import fcntl
import hashlib
import importlib.resources
import json
import os
import queue
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel

from rcp.config import Manifest, load_manifest
from rcp.core.models import GraphBranchMetadata
from rcp.limits import (
    BACKUP_COPY_BUFFER_BYTES,
    BACKUP_INVENTORY_MAX_ENTRIES,
    BACKUP_REMOTE_EXPORT_TIMEOUT_SECONDS,
    CHAT_ARTIFACT_MAX_FILE_BYTES,
    REMOTE_ARTIFACT_READ_TIMEOUT_SECONDS,
    REMOTE_STATE_HEAD_PROBE_TIMEOUT_SECONDS,
    REMOTE_STATE_RECONCILE_WINDOW_SECONDS,
    STATE_LOCK_ATTEMPT_TIMEOUT_SECONDS,
    STATE_LOCK_HOLDER_STOP_TIMEOUT_SECONDS,
    STATE_LOCK_POLL_INTERVAL_SECONDS,
)
from rcp.server_ops.backup_models import (
    BACKUP_RESEARCH_CANONICAL_ROOTS,
    BACKUP_RESEARCH_DELEGATED_ROOTS,
    BACKUP_RESEARCH_EXCLUSIONS,
    BackupBranchSourcePlan,
    BackupCanonicalSourceFile,
    BackupCanonicalSourcePlan,
)
from rcp.transport.remote_read_kept_view import MISSING as _REMOTE_VIEW_MISSING
from rcp.transport.remote_read_kept_view import TOO_LARGE as _REMOTE_VIEW_TOO_LARGE
from rcp.transport.remote_read_kept_view import UNSAFE as _REMOTE_VIEW_UNSAFE
from rcp.transport.ssh import rsync_ssh_arguments, ssh_arguments

_SNAPSHOT_LOCKS_GUARD = threading.Lock()
_SNAPSHOT_LOCKS: dict[str, threading.RLock] = {}

_LOCK_ACQUIRED = "acquired"
_LOCK_CONTENDED = "contended"
_LOCK_LEGACY_DIRECTORY = "legacy-directory"
_LOCK_UNSAFE_ENTRY = "unsafe-entry"
_KEPT_VIEW_NAME_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,238})\.html")
_KEPT_ARTIFACT_NAME_PATTERN = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,220})\.(?:html?|png|jpe?g|gif|webp|svg)"
)
_ARCHIVE_TIMESTAMP_PATTERN = re.compile(r"[0-9]{8}T[0-9]{12}Z")
_RETAINED_HISTORY_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_RETAINED_BRANCH_PATCH_PATTERN = re.compile(r"[0-9]{6}\.json")
_RETAINED_BRANCH_MERGE_PATTERN = re.compile(r"[a-f0-9]{64}\.json")
_RETAINED_BRANCH_DERIVED_NAMES = frozenset(
    {"graph.json", "research.md", "glossary.json", "proposals.json", "coverage.json"}
)
_RETAINED_HISTORY_CHANGED_MESSAGE = (
    "Retained research changed since you reviewed it. Run the read-only preflight again "
    "before archiving."
)


@lru_cache(maxsize=8)
def _remote_script(name: str) -> str:
    """Load a shipped remote script as a package resource.

    These run on the execution machine through ``python -c``, so RCP sends each
    module's own source rather than a transcribed copy — a literal is invisible to
    ruff and to every test. ``importlib.resources`` keeps this working for source,
    wheel, and frozen builds alike.
    """

    return importlib.resources.files("rcp.transport").joinpath(name).read_text(encoding="utf-8")


_REMOTE_PATCH_LOG_HEAD_SCRIPT = """\
patches=$1
if [ ! -d "$patches" ]; then
    exit 0
fi
find "$patches" -maxdepth 2 -type f -name '[0-9][0-9][0-9][0-9][0-9][0-9].json' \
    ! -path "$patches/.*/*" \
    -exec basename {} \\; | LC_ALL=C sort | tail -n 1
"""


def _snapshot_lock(root: Path) -> threading.RLock:
    key = os.path.normcase(str(root.resolve()))
    with _SNAPSHOT_LOCKS_GUARD:
        return _SNAPSHOT_LOCKS.setdefault(key, threading.RLock())


def _result_view_slug(value: str, fallback: str, *, max_length: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    slug = slug[:max_length].rstrip("-")
    return slug or fallback


def _result_view_base_name(source_name: str, project_name: str, today: date | None) -> str:
    source_base = re.split(r"[/\\]", source_name)[-1]
    source_stem = re.sub(r"(?i)\.html?$", "", source_base)
    source_slug = _result_view_slug(source_stem, "result", max_length=96)
    project_slug = _result_view_slug(project_name, "project", max_length=80)
    current_date = today or date.today()
    name = f"{source_slug}-{project_slug}-{current_date.strftime('%y-%m-%d')}.html"
    if _KEPT_VIEW_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("could not derive a safe repository result-view name")
    return name


def _artifact_base_name(source_name: str, project_name: str, today: date | None) -> str:
    source_base = re.split(r"[/\\]", source_name)[-1]
    suffix = Path(source_base).suffix.casefold()
    if suffix not in {".html", ".htm", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        raise ValueError("unsupported kept artifact type")
    source_slug = _result_view_slug(Path(source_base).stem, "artifact", max_length=80)
    project_slug = _result_view_slug(project_name, "project", max_length=64)
    current_date = today or date.today()
    name = f"{source_slug}-{project_slug}-{current_date.strftime('%y-%m-%d')}{suffix}"
    if _KEPT_ARTIFACT_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("could not derive a safe repository artifact name")
    return name


def _validated_kept_artifact_name(name: str) -> str:
    if _KEPT_ARTIFACT_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("kept artifact name must be a safe supported base name")
    return name


def _validated_kept_view_name(name: str) -> str:
    if _KEPT_VIEW_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("kept result-view name must be a safe HTML base name")
    return name


def _validated_view_bytes(data: bytes) -> bytes:
    if not isinstance(data, bytes):
        raise TypeError("result-view data must be bytes")
    if len(data) > CHAT_ARTIFACT_MAX_FILE_BYTES:
        raise ValueError("result view exceeds the per-file limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("result view is not UTF-8 HTML") from exc
    if "\x00" in text:
        raise ValueError("result view contains NUL bytes")
    return data


def _validated_view_read_limit(max_bytes: int) -> int:
    if not 1 <= max_bytes <= CHAT_ARTIFACT_MAX_FILE_BYTES:
        raise ValueError("result-view read limit is outside the supported range")
    return max_bytes


def _repository_directory_fd(repository: Path) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise StateUnavailable("Safe repository result-view file operations are unavailable.")
    try:
        return os.open(repository, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise StateUnavailable(f"Repository root is unavailable: {exc}") from exc


def _open_views_directory(repository_fd: int, *, create: bool) -> int:
    if create:
        try:
            os.mkdir("views", 0o755, dir_fd=repository_fd)
            os.fsync(repository_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise StateUnavailable(
                f"Could not create the repository views directory: {exc}"
            ) from exc
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open("views", flags, dir_fd=repository_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError("repository views path is not a regular directory") from exc


def _open_artifacts_directory(repository_fd: int, *, create: bool) -> int:
    if create:
        try:
            os.mkdir("artifacts", 0o755, dir_fd=repository_fd)
            os.fsync(repository_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise StateUnavailable(
                f"Could not create the repository artifacts directory: {exc}"
            ) from exc
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open("artifacts", flags, dir_fd=repository_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError("repository artifacts path is not a regular directory") from exc


def _collision_view_name(base_name: str, index: int) -> str:
    if index == 1:
        return base_name
    return f"{base_name[:-5]}-{index}.html"


def _collision_artifact_name(base_name: str, index: int) -> str:
    if index == 1:
        return base_name
    path = Path(base_name)
    return f"{path.stem}-{index}{path.suffix}"


def _is_collision_view_name(name: str, base_name: str) -> bool:
    if name == base_name:
        return True
    prefix = re.escape(base_name[:-5])
    return re.fullmatch(rf"{prefix}-(?:[2-9]|[1-9][0-9]+)\.html", name) is not None


def _is_collision_artifact_name(name: str, base_name: str) -> bool:
    if name == base_name:
        return True
    path = Path(base_name)
    return (
        re.fullmatch(
            rf"{re.escape(path.stem)}-(?:[2-9]|[1-9][0-9]+){re.escape(path.suffix)}",
            name,
        )
        is not None
    )


def _archive_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _archive_research_directory(root: Path, timestamp: str) -> Path:
    if root.name != ".research" or _ARCHIVE_TIMESTAMP_PATTERN.fullmatch(timestamp) is None:
        raise ValueError("invalid canonical research directory or archive timestamp")
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise StateUnavailable(f"Could not archive canonical research at {root}: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise StateUnavailable(
            f"Could not archive canonical research at {root}: path is not a regular directory."
        )

    base_name = f"{root.name}.archive-{timestamp}"
    for index in range(1, 10000):
        name = base_name if index == 1 else f"{base_name}-{index}"
        archive = root.with_name(name)
        if os.path.lexists(archive):
            continue
        try:
            os.rename(root, archive)
        except FileExistsError:
            continue
        except OSError as exc:
            raise StateUnavailable(
                f"Could not archive canonical research at {root}: {exc}"
            ) from exc
        return archive
    raise StateUnavailable(f"Could not choose a unique archive name beside {root}.")


def _retained_history_paths(
    root: Path,
    *,
    skip_quarantine: bool = False,
) -> list[Path]:
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise StateUnavailable(f"Could not inspect retained research at {root}: {exc}") from exc
    if not stat.S_ISDIR(root_mode):
        raise StateUnavailable(f"Retained research path is not a regular directory: {root}")

    def require_regular(path: Path, label: str) -> None:
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise StateUnavailable(f"Could not inspect {label} at {path}: {exc}") from exc
        if not stat.S_ISREG(mode):
            raise StateUnavailable(f"{label.capitalize()} is not a regular file: {path}")

    manifest = root / "manifest.toml"
    require_regular(manifest, "retained manifest")
    paths = [manifest]
    scope_base = root / "scope-base.json"
    if os.path.lexists(scope_base):
        require_regular(scope_base, "retained scope provenance")
        paths.append(scope_base)

    patches = root / "patches"
    if os.path.lexists(patches):
        try:
            patches_mode = patches.lstat().st_mode
        except OSError as exc:
            raise StateUnavailable(
                f"Could not inspect retained patches at {patches}: {exc}"
            ) from exc
        if not stat.S_ISDIR(patches_mode):
            raise StateUnavailable(f"Retained patch path is not a regular directory: {patches}")
        for child in patches.iterdir():
            if re.fullmatch(r"[0-9]{6}\.json", child.name):
                require_regular(child, "retained patch")
                paths.append(child)
            elif child.name.startswith("batch-"):
                try:
                    batch_mode = child.lstat().st_mode
                except OSError as exc:
                    raise StateUnavailable(
                        f"Could not inspect retained patch batch at {child}: {exc}"
                    ) from exc
                if not stat.S_ISDIR(batch_mode):
                    raise StateUnavailable(
                        f"Retained patch batch is not a regular directory: {child}"
                    )
                for patch in child.iterdir():
                    if re.fullmatch(r"[0-9]{6}\.json", patch.name):
                        require_regular(patch, "retained patch")
                        paths.append(patch)
            elif child.name.startswith((".batch-", ".unconfirmed-")):
                continue

    branches = root / "branches"
    if os.path.lexists(branches):
        try:
            branches_mode = branches.lstat().st_mode
        except OSError as exc:
            raise StateUnavailable(
                f"Could not inspect retained graph branches at {branches}: {exc}"
            ) from exc
        if not stat.S_ISDIR(branches_mode):
            raise StateUnavailable(
                f"Retained graph branches path is not a regular directory: {branches}"
            )
        try:
            branch_entries = sorted(branches.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise StateUnavailable(
                f"Could not enumerate retained graph branches at {branches}: {exc}"
            ) from exc
        for branch in branch_entries:
            if skip_quarantine and branch.name.startswith(".unconfirmed-"):
                continue
            try:
                branch_id = uuid.UUID(branch.name)
            except ValueError as exc:
                raise StateUnavailable(
                    f"Retained graph branch has a malformed canonical name: {branch}"
                ) from exc
            if str(branch_id) != branch.name or branch_id.version != 4:
                raise StateUnavailable(
                    f"Retained graph branch has a malformed canonical name: {branch}"
                )
            try:
                branch_mode = branch.lstat().st_mode
            except OSError as exc:
                raise StateUnavailable(
                    f"Could not inspect retained graph branch at {branch}: {exc}"
                ) from exc
            if not stat.S_ISDIR(branch_mode):
                raise StateUnavailable(
                    f"Retained graph branch path is not a regular directory: {branch}"
                )

            try:
                entries = {child.name: child for child in branch.iterdir()}
            except OSError as exc:
                raise StateUnavailable(
                    f"Could not enumerate retained graph branch at {branch}: {exc}"
                ) from exc
            allowed = {
                "branch.json",
                "patches",
                "merges",
                *_RETAINED_BRANCH_DERIVED_NAMES,
            }
            malformed = sorted(entries.keys() - allowed)
            if malformed:
                raise StateUnavailable(
                    "Retained graph branch contains a malformed canonical entry: "
                    f"{entries[malformed[0]]}"
                )

            metadata = branch / "branch.json"
            require_regular(metadata, "retained graph branch metadata")
            paths.append(metadata)
            for name in _RETAINED_BRANCH_DERIVED_NAMES:
                derived = entries.get(name)
                if derived is not None:
                    require_regular(derived, "retained graph branch derived output")

            for directory_name, pattern, label in (
                ("patches", _RETAINED_BRANCH_PATCH_PATTERN, "retained graph branch patch"),
                ("merges", _RETAINED_BRANCH_MERGE_PATTERN, "retained graph branch merge receipt"),
            ):
                directory = branch / directory_name
                try:
                    directory_mode = directory.lstat().st_mode
                except OSError as exc:
                    raise StateUnavailable(
                        f"Could not inspect {label} path at {directory}: {exc}"
                    ) from exc
                if not stat.S_ISDIR(directory_mode):
                    raise StateUnavailable(
                        f"{label.capitalize()} path is not a regular directory: {directory}"
                    )
                try:
                    children = directory.iterdir()
                    for child in children:
                        if (
                            skip_quarantine
                            and directory_name == "patches"
                            and child.name.startswith(".unconfirmed-")
                        ):
                            continue
                        if pattern.fullmatch(child.name) is None:
                            raise StateUnavailable(
                                f"{label.capitalize()} has a malformed canonical name: {child}"
                            )
                        require_regular(child, label)
                        paths.append(child)
                except StateUnavailable:
                    raise
                except OSError as exc:
                    raise StateUnavailable(
                        f"Could not enumerate {label} path at {directory}: {exc}"
                    ) from exc
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _backup_canonical_file_kind(relative: Path) -> str:
    parts = relative.parts
    if relative == Path("manifest.toml"):
        return "manifest"
    if relative == Path("scope-base.json"):
        return "scope_base"
    if parts[0] == "patches":
        return "main_patch"
    if len(parts) == 3 and parts[0] == "branches" and parts[2] == "branch.json":
        return "branch_metadata"
    if len(parts) == 4 and parts[0] == "branches" and parts[2] == "patches":
        return "branch_patch"
    if len(parts) == 4 and parts[0] == "branches" and parts[2] == "merges":
        return "branch_merge_receipt"
    raise StateUnavailable(f"Retained research contains an unclassified canonical file: {relative}")


def _backup_source_file(root: Path, path: Path) -> BackupCanonicalSourceFile:
    relative = path.relative_to(root)
    try:
        mode = path.lstat().st_mode
        size = path.stat(follow_symlinks=False).st_size
    except OSError as exc:
        raise StateUnavailable(f"Could not inspect retained history at {path}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise StateUnavailable(f"Retained history is not a safe regular file: {path}")
    return BackupCanonicalSourceFile(
        relative_path=relative.as_posix(),
        kind=_backup_canonical_file_kind(relative),
        observed_size_bytes=size,
    )


def _canonical_backup_source_plan(root: Path) -> BackupCanonicalSourcePlan:
    """Observe retained inputs and direct-root policy without taking a writer lock."""

    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise StateUnavailable(f"Could not inspect canonical research at {root}: {exc}") from exc
    if not stat.S_ISDIR(root_mode):
        raise StateUnavailable(f"Canonical research is not a regular directory: {root}")
    try:
        direct_entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise StateUnavailable(f"Could not enumerate canonical research at {root}: {exc}") from exc
    if len(direct_entries) > BACKUP_INVENTORY_MAX_ENTRIES:
        raise StateUnavailable("Canonical research direct-root inventory exceeds its entry bound.")

    delegated: list[str] = []
    excluded: list[str] = []
    unclassified: list[str] = []
    excluded_canonical_paths: list[str] = []
    for entry in direct_entries:
        name = entry.name
        if name in BACKUP_RESEARCH_DELEGATED_ROOTS:
            try:
                mode = entry.lstat().st_mode
            except OSError as exc:
                raise StateUnavailable(
                    f"Could not inspect delegated research root at {entry}: {exc}"
                ) from exc
            if not stat.S_ISDIR(mode):
                raise StateUnavailable(
                    f"Delegated research root is not a safe regular directory: {entry}"
                )
            delegated.append(name)
        elif name in BACKUP_RESEARCH_EXCLUSIONS:
            excluded.append(name)
        elif name not in BACKUP_RESEARCH_CANONICAL_ROOTS:
            unclassified.append(name)

    retained_paths = _retained_history_paths(root, skip_quarantine=True)
    if len(retained_paths) > BACKUP_INVENTORY_MAX_ENTRIES:
        raise StateUnavailable("Canonical backup inventory exceeds its entry bound.")

    patches = root / "patches"
    if os.path.lexists(patches):
        try:
            patch_entries = sorted(patches.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise StateUnavailable(
                f"Could not enumerate canonical Patches at {patches}: {exc}"
            ) from exc
        for entry in patch_entries:
            if entry.name.startswith((".batch-", ".unconfirmed-")):
                excluded_canonical_paths.append(entry.relative_to(root).as_posix())

    branches_root = root / "branches"
    if os.path.lexists(branches_root):
        try:
            branch_entries = sorted(branches_root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise StateUnavailable(
                f"Could not enumerate canonical graph branches at {branches_root}: {exc}"
            ) from exc
        for entry in branch_entries:
            if entry.name.startswith(".unconfirmed-"):
                excluded_canonical_paths.append(entry.relative_to(root).as_posix())
                continue
            try:
                entry_mode = entry.lstat().st_mode
            except OSError as exc:
                raise StateUnavailable(
                    f"Could not inspect canonical graph branch at {entry}: {exc}"
                ) from exc
            if not stat.S_ISDIR(entry_mode):
                continue
            branch_patches = entry / "patches"
            if not os.path.lexists(branch_patches):
                continue
            try:
                children = branch_patches.iterdir()
                for child in children:
                    if child.name.startswith(".unconfirmed-"):
                        excluded_canonical_paths.append(child.relative_to(root).as_posix())
            except OSError as exc:
                raise StateUnavailable(
                    f"Could not enumerate graph branch Patches at {branch_patches}: {exc}"
                ) from exc

    files = tuple(_backup_source_file(root, path) for path in retained_paths)
    main_files = tuple(item for item in files if not item.relative_path.startswith("branches/"))
    main_revisions = [
        int(PurePosixPath(item.relative_path).stem)
        for item in main_files
        if item.kind == "main_patch"
    ]
    if len(main_revisions) != len(set(main_revisions)):
        raise StateUnavailable("Canonical main history repeats a Patch revision.")

    branch_files: dict[str, list[BackupCanonicalSourceFile]] = {}
    for item in files:
        parts = PurePosixPath(item.relative_path).parts
        if parts[0] == "branches":
            branch_files.setdefault(parts[1], []).append(item)
    branches: list[BackupBranchSourcePlan] = []
    for branch_id in sorted(branch_files):
        metadata_path = root / "branches" / branch_id / "branch.json"
        try:
            metadata = GraphBranchMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise StateUnavailable(
                f"Could not validate retained graph branch metadata at {metadata_path}."
            ) from exc
        if metadata.branch_id != branch_id:
            raise StateUnavailable(
                f"Retained graph branch metadata names another branch at {metadata_path}."
            )
        revisions = [
            int(PurePosixPath(item.relative_path).stem)
            for item in branch_files[branch_id]
            if item.kind == "branch_patch"
        ]
        if len(revisions) != len(set(revisions)):
            raise StateUnavailable(f"Graph branch {branch_id} repeats a Patch revision.")
        observed_revision = max(revisions, default=metadata.base_head.revision)
        if observed_revision != metadata.head.revision:
            raise StateUnavailable(
                f"Graph branch {branch_id} metadata and retained Patch head disagree."
            )
        ordered_files = tuple(
            sorted(
                branch_files[branch_id],
                key=lambda item: (
                    item.kind != "branch_metadata",
                    item.relative_path,
                ),
            )
        )
        branches.append(
            BackupBranchSourcePlan(
                branch_id=branch_id,
                head=metadata.head,
                files=ordered_files,
            )
        )

    observed_bytes = sum(item.observed_size_bytes for item in files)
    return BackupCanonicalSourcePlan(
        main_observed_revision=max(main_revisions, default=0),
        main_files=main_files,
        branches=tuple(branches),
        delegated_roots=tuple(delegated),
        excluded_roots=tuple(excluded),
        excluded_canonical_paths=tuple(sorted(excluded_canonical_paths)),
        unclassified_roots=tuple(unclassified),
        observed_canonical_bytes=observed_bytes,
    )


def _retained_history_fingerprint(root: Path) -> str:
    digest = hashlib.sha256(b"rcp-retained-history-v2\0")
    for path in _retained_history_paths(root):
        relative = path.relative_to(root).as_posix().encode()
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise StateUnavailable(f"Could not read retained history at {path}: {exc}") from exc
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _require_expected_history_fingerprint(root: Path, expected: str) -> None:
    if _RETAINED_HISTORY_FINGERPRINT_PATTERN.fullmatch(expected) is None:
        raise ValueError("expected retained-history fingerprint is invalid")
    if _retained_history_fingerprint(root) != expected:
        raise StateUnavailable(_RETAINED_HISTORY_CHANGED_MESSAGE)


class StateUnavailable(RuntimeError):
    pass


def _restore_file_matches(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise StateUnavailable(f"Could not inspect restored project file {path}: {exc}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
            return False
        while True:
            chunk = os.read(descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)
    return size == expected_size and digest.hexdigest() == expected_sha256


def _require_restore_source(
    source: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or expected_size < 0:
        raise ValueError("restored project file proof is invalid")
    if not _restore_file_matches(
        source,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    ):
        raise StateUnavailable("An archived project file differs from its manifest proof.")


def _safe_restore_parent(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("restored project path must be one safe relative path")
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError:
        parent_mode = root.parent.lstat().st_mode
        if not stat.S_ISDIR(parent_mode):
            raise StateUnavailable("The restored project root parent is not a directory.") from None
        root.mkdir(mode=0o755)
        root_mode = root.lstat().st_mode
    if not stat.S_ISDIR(root_mode):
        raise StateUnavailable("The restored project root is not a safe directory.")
    current = root
    for part in relative.parent.parts:
        current = current / part
        with suppress(FileExistsError):
            current.mkdir(mode=0o755)
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise StateUnavailable("A restored project directory is unavailable.") from exc
        if not stat.S_ISDIR(mode):
            raise StateUnavailable("A restored project path has an unsafe parent.")
    return current


def _stage_exact_restore_file(
    root: Path,
    relative: Path,
    source: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    parent = _safe_restore_parent(root, relative)
    destination = root / relative
    if os.path.lexists(destination):
        if _restore_file_matches(
            destination,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        ):
            return
        raise StateUnavailable(f"Restored project file conflicts with existing bytes: {relative}")

    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary = parent / f".{relative.name}.restore-{uuid.uuid4().hex}"
    destination_descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(source_descriptor)
        path_before = source.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (path_before.st_dev, path_before.st_ino)
            or before.st_size != expected_size
        ):
            raise StateUnavailable("An archived project file changed before publication.")
        destination_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        while True:
            chunk = os.read(source_descriptor, BACKUP_COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if size > expected_size:
                raise StateUnavailable("An archived project file changed during publication.")
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise OSError("short restored project file write")
                remaining = remaining[written:]
        after = os.fstat(source_descriptor)
        path_after = source.lstat()
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            size != expected_size
            or digest.hexdigest() != expected_sha256
            or any(getattr(before, field) != getattr(after, field) for field in stable)
            or any(getattr(after, field) != getattr(path_after, field) for field in stable)
        ):
            raise StateUnavailable("An archived project file changed during publication.")
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = -1
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if not _restore_file_matches(
                destination,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            ):
                raise StateUnavailable(
                    f"Restored project file raced with conflicting bytes: {relative}"
                ) from None
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


class RunLockCancelled(RuntimeError):
    """Run-lock acquisition stopped because its owning task was cancelled."""


class RunLockOwnershipLost(StateUnavailable):
    """A previously acquired run lock is no longer owned."""


class RunLockLease:
    """Observable ownership for one held canonical-state run lock."""

    def __init__(
        self,
        location: str,
        *,
        on_lost: Callable[[str], None] | None = None,
        owned: Callable[[], bool] | None = None,
        command: Callable[[dict[str, object]], dict[str, object]] | None = None,
    ) -> None:
        self.location = location
        self._on_lost = on_lost
        self._owned = owned
        self._command = command
        self._guard = threading.Lock()
        self._command_guard = threading.Lock()
        self._lost: str | None = None
        self._releasing = False

    def assert_owned(self) -> None:
        if self._owned is not None and not self._owned():
            self._mark_lost(f"Canonical-state lock ownership was lost at {self.location}.")
        with self._guard:
            message = self._lost
            releasing = self._releasing
        if message is not None:
            raise RunLockOwnershipLost(message)
        if releasing:
            raise RunLockOwnershipLost(
                f"Canonical-state lock lease at {self.location} is no longer active."
            )

    def _mark_lost(self, message: str) -> None:
        callback: Callable[[str], None] | None = None
        with self._guard:
            if self._lost is not None or self._releasing:
                return
            self._lost = message
            callback = self._on_lost
        if callback is not None:
            try:
                callback(message)
            except Exception as exc:  # The typed ownership signal must remain authoritative.
                with self._guard:
                    self._lost = f"{message} Ownership-loss callback failed: {str(exc)[:200]}"

    def _begin_release(self) -> None:
        with self._guard:
            self._releasing = True

    def _run_owned_command(self, command: dict[str, object]) -> dict[str, object]:
        with self._command_guard:
            self.assert_owned()
            if self._command is None:
                raise StateUnavailable(
                    f"Canonical-state lock at {self.location} cannot apply remote commands."
                )
            response = self._command(command)
            self.assert_owned()
            return response


class _LegacyLockDirectory(StateUnavailable):
    pass


class _UnsafeLockEntry(StateUnavailable):
    pass


class BatchPublishFailed(StateUnavailable):
    """A remote batch failed with an explicit commit-point observation."""

    def __init__(self, message: str, *, commit_status: Literal["absent", "present", "unknown"]):
        self.commit_status = commit_status
        super().__init__(message)


class StateWorkspaceStatus(BaseModel):
    remote: bool
    reachable: bool
    location: str
    last_synced_at: datetime | None = None
    error: str | None = None


class StateWorkspace:
    def __init__(self, root: Path, location: str) -> None:
        self.root = root
        self.location = location
        self.remote = False
        self.reachable = True
        self.last_synced_at: datetime | None = None
        self.error: str | None = None
        self.materialization_repair_required = False
        self.snapshot_lock = _snapshot_lock(root)

    def refresh(self) -> bool:
        with self.snapshot_lock:
            return self._refresh_snapshot()

    def restore_exact_file(
        self,
        relative_path: Path | str,
        source: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        """Publish one archived project file without replacing different bytes."""

        relative = _validated_relative_path(relative_path)
        with self.transaction():
            _stage_exact_restore_file(
                self.root,
                relative,
                source,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )

    def restore_kept_artifact(
        self,
        name: str,
        source: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        """Restore one referenced kept artifact under its exact durable name."""

        safe_name = _validated_kept_artifact_name(name)
        if not 1 <= expected_size <= CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("restored artifact bytes are outside the supported size range")
        with self.transaction():
            _stage_exact_restore_file(
                self.root.parent,
                Path("artifacts") / safe_name,
                source,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )

    def restore_kept_result_view(
        self,
        name: str,
        source: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        """Restore one referenced legacy result view under its exact durable name."""

        safe_name = _validated_kept_view_name(name)
        if not 1 <= expected_size <= CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("restored result-view bytes are outside the supported size range")
        with self.transaction():
            _stage_exact_restore_file(
                self.root.parent,
                Path("views") / safe_name,
                source,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )

    def retained_history_fingerprint(self) -> str:
        """Fingerprint exactly the canonical inputs used to replay retained history."""

        with self.snapshot_lock:
            return _retained_history_fingerprint(self.root)

    def backup_canonical_source_plan(self) -> BackupCanonicalSourcePlan:
        """Observe backup inputs without refreshing or taking a publication lock.

        A remote workspace's root is only its local mirror. O2b must run this same
        inventory against its retained remote export before claiming a remote head.
        """

        return _canonical_backup_source_plan(self.root)

    def backup_source_root(self, destination: Path) -> Path:
        """Return local live sources; SSH overrides with one private lock-free export."""

        return self.root

    def archive_research(self, *, expected_history_fingerprint: str | None = None) -> str:
        """Atomically move the complete canonical ``.research`` directory aside."""

        with self.snapshot_lock:
            lock_path = self.root / ".append.lock"
            try:
                handle = lock_path.open("a+", encoding="utf-8")
            except OSError as exc:
                raise StateUnavailable(
                    f"Could not lock canonical research for archival at {self.root}: {exc}"
                ) from exc
            with handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    if expected_history_fingerprint is not None:
                        _require_expected_history_fingerprint(
                            self.root,
                            expected_history_fingerprint,
                        )
                    return str(_archive_research_directory(self.root, _archive_timestamp()))
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _refresh_snapshot(self) -> bool:
        self.reachable = True
        return True

    def refresh_if_stale(
        self,
        max_age_seconds: float = REMOTE_STATE_RECONCILE_WINDOW_SECONDS,
    ) -> bool:
        return self.refresh()

    def cached_patch_log_head(self) -> int | None:
        patches = self.root / "patches"
        revisions = [
            int(path.stem)
            for path in (
                *patches.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"),
                *patches.glob("batch-*/[0-9][0-9][0-9][0-9][0-9][0-9].json"),
            )
            if path.is_file()
        ]
        return max(revisions, default=None)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.snapshot_lock:
            yield

    def keep_result_view(
        self,
        *,
        source_name: str,
        project_name: str,
        data: bytes,
        today: date | None = None,
    ) -> str:
        """Create one immutable HTML result view beside, never inside, ``.research``."""

        base_name = _result_view_base_name(source_name, project_name, today)
        content = _validated_view_bytes(data)
        with self.transaction():
            repository_fd = _repository_directory_fd(self.root.parent)
            try:
                views_fd = _open_views_directory(repository_fd, create=True)
                try:
                    for index in range(1, 10000):
                        candidate = _collision_view_name(base_name, index)
                        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                        try:
                            descriptor = os.open(candidate, flags, 0o644, dir_fd=views_fd)
                        except FileExistsError:
                            continue
                        try:
                            remaining = memoryview(content)
                            while remaining:
                                written = os.write(descriptor, remaining)
                                if written <= 0:
                                    raise OSError("short result-view write")
                                remaining = remaining[written:]
                            os.fsync(descriptor)
                        except BaseException:
                            os.close(descriptor)
                            descriptor = -1
                            try:
                                os.unlink(candidate, dir_fd=views_fd)
                                os.fsync(views_fd)
                            except OSError:
                                pass
                            raise
                        finally:
                            if descriptor >= 0:
                                os.close(descriptor)
                        os.fsync(views_fd)
                        return candidate
                    raise StateUnavailable("Too many repository result-view name collisions.")
                finally:
                    os.close(views_fd)
            finally:
                os.close(repository_fd)

    def keep_artifact(
        self,
        *,
        source_name: str,
        project_name: str,
        data: bytes,
        today: date | None = None,
    ) -> str:
        """Keep one live artifact without claiming or replacing existing files."""

        base_name = _artifact_base_name(source_name, project_name, today)
        if not isinstance(data, bytes) or not 1 <= len(data) <= CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("artifact bytes are outside the supported size range")
        with self.transaction():
            repository_fd = _repository_directory_fd(self.root.parent)
            try:
                artifacts_fd = _open_artifacts_directory(repository_fd, create=True)
                try:
                    for index in range(1, 10000):
                        candidate = _collision_artifact_name(base_name, index)
                        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                        try:
                            descriptor = os.open(candidate, flags, 0o644, dir_fd=artifacts_fd)
                        except FileExistsError:
                            continue
                        try:
                            remaining = memoryview(data)
                            while remaining:
                                written = os.write(descriptor, remaining)
                                if written <= 0:
                                    raise OSError("short artifact write")
                                remaining = remaining[written:]
                            os.fsync(descriptor)
                        except BaseException:
                            os.close(descriptor)
                            descriptor = -1
                            try:
                                os.unlink(candidate, dir_fd=artifacts_fd)
                                os.fsync(artifacts_fd)
                            except OSError:
                                pass
                            raise
                        finally:
                            if descriptor >= 0:
                                os.close(descriptor)
                        os.fsync(artifacts_fd)
                        return candidate
                    raise StateUnavailable("Too many repository artifact name collisions.")
                finally:
                    os.close(artifacts_fd)
            finally:
                os.close(repository_fd)

    def read_kept_artifact(
        self,
        name: str,
        *,
        max_bytes: int = CHAT_ARTIFACT_MAX_FILE_BYTES,
    ) -> bytes:
        """Read the current bytes of one live kept artifact without following links."""

        safe_name = _validated_kept_artifact_name(name)
        limit = _validated_view_read_limit(max_bytes)
        with self.snapshot_lock:
            repository_fd = _repository_directory_fd(self.root.parent)
            try:
                artifacts_fd = _open_artifacts_directory(repository_fd, create=False)
                try:
                    descriptor = os.open(
                        safe_name,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=artifacts_fd,
                    )
                    try:
                        metadata = os.fstat(descriptor)
                        if not stat.S_ISREG(metadata.st_mode):
                            raise ValueError("kept artifact is not a regular file")
                        if metadata.st_size > limit:
                            raise ValueError("kept artifact exceeds the read limit")
                        chunks: list[bytes] = []
                        remaining = limit + 1
                        while remaining > 0:
                            chunk = os.read(descriptor, min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            chunks.append(chunk)
                            remaining -= len(chunk)
                        data = b"".join(chunks)
                        if len(data) > limit:
                            raise ValueError("kept artifact exceeds the read limit")
                        return data
                    finally:
                        os.close(descriptor)
                finally:
                    os.close(artifacts_fd)
            finally:
                os.close(repository_fd)

    def replace_kept_artifact(self, name: str, data: bytes) -> None:
        """Atomically update one live kept artifact, accepting intervening external edits."""

        safe_name = _validated_kept_artifact_name(name)
        if not isinstance(data, bytes) or not 1 <= len(data) <= CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("artifact bytes are outside the supported size range")
        temporary_name = f".{safe_name}.rcp-{uuid.uuid4().hex}"
        with self.transaction():
            repository_fd = _repository_directory_fd(self.root.parent)
            try:
                artifacts_fd = _open_artifacts_directory(repository_fd, create=False)
                try:
                    metadata = os.stat(safe_name, dir_fd=artifacts_fd, follow_symlinks=False)
                    if not stat.S_ISREG(metadata.st_mode):
                        raise ValueError("kept artifact is not a regular file")
                    descriptor = os.open(
                        temporary_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o644,
                        dir_fd=artifacts_fd,
                    )
                    try:
                        remaining = memoryview(data)
                        while remaining:
                            written = os.write(descriptor, remaining)
                            if written <= 0:
                                raise OSError("short artifact replacement write")
                            remaining = remaining[written:]
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    os.replace(
                        temporary_name,
                        safe_name,
                        src_dir_fd=artifacts_fd,
                        dst_dir_fd=artifacts_fd,
                    )
                    os.fsync(artifacts_fd)
                except BaseException:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_name, dir_fd=artifacts_fd)
                    raise
                finally:
                    os.close(artifacts_fd)
            finally:
                os.close(repository_fd)

    def read_kept_result_view(
        self,
        name: str,
        *,
        max_bytes: int = CHAT_ARTIFACT_MAX_FILE_BYTES,
    ) -> bytes:
        """Read one immutable repository result view without following links."""

        safe_name = _validated_kept_view_name(name)
        limit = _validated_view_read_limit(max_bytes)
        with self.snapshot_lock:
            repository_fd = _repository_directory_fd(self.root.parent)
            try:
                views_fd = _open_views_directory(repository_fd, create=False)
                try:
                    try:
                        descriptor = os.open(
                            safe_name,
                            os.O_RDONLY | os.O_NOFOLLOW,
                            dir_fd=views_fd,
                        )
                    except FileNotFoundError:
                        raise
                    except OSError as exc:
                        raise ValueError("kept result view is not a readable regular file") from exc
                    try:
                        metadata = os.fstat(descriptor)
                        if not stat.S_ISREG(metadata.st_mode):
                            raise ValueError("kept result view is not a regular file")
                        if metadata.st_size > limit:
                            raise ValueError("kept result view exceeds the read limit")
                        chunks: list[bytes] = []
                        remaining = limit + 1
                        while remaining > 0:
                            chunk = os.read(descriptor, min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            chunks.append(chunk)
                            remaining -= len(chunk)
                        content = b"".join(chunks)
                        if len(content) > limit:
                            raise ValueError("kept result view exceeds the read limit")
                        return content
                    finally:
                        os.close(descriptor)
                finally:
                    os.close(views_fd)
            finally:
                os.close(repository_fd)

    @contextmanager
    def run_lock(
        self,
        *,
        on_wait: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        on_lost: Callable[[str], None] | None = None,
    ) -> Iterator[RunLockLease]:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / ".agent-run.lock"
        with path.open("a+", encoding="utf-8") as handle:
            waiting_reported = False
            while True:
                if cancelled is not None and cancelled():
                    raise RunLockCancelled("Run-lock acquisition was cancelled while waiting.")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not waiting_reported and on_wait is not None:
                        on_wait("Waiting for another graph-writing run to release canonical state.")
                        waiting_reported = True
                    time.sleep(STATE_LOCK_POLL_INTERVAL_SECONDS)
            lease = RunLockLease(str(path), on_lost=on_lost)
            try:
                if cancelled is not None and cancelled():
                    raise RunLockCancelled("Run-lock acquisition was cancelled after acquiring.")
                yield lease
            finally:
                lease._begin_release()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def publish(self, relative_paths: list[Path | str]) -> None:
        del relative_paths

    def publish_committed_batch(
        self,
        relative_paths: list[Path | str],
        batch_directory: Path | str,
    ) -> None:
        """Publish a locally committed patch batch as one visible history unit."""

        del batch_directory
        self.publish(relative_paths)

    def publish_committed_patch(
        self,
        relative_paths: list[Path | str],
        patch_path: Path | str,
    ) -> None:
        """Publish one patch as the visible history commit point."""

        del patch_path
        self.publish(relative_paths)

    def publish_committed_branch_file(
        self,
        relative_paths: list[Path | str],
        commit_path: Path | str,
    ) -> None:
        """Publish immutable branch metadata or a merge receipt as a commit point."""

        _validated_branch_commit_path(commit_path)
        self.publish(relative_paths)

    def require_materialization_repair(self) -> None:
        self.materialization_repair_required = True

    def complete_materialization_repair(self) -> None:
        self.materialization_repair_required = False

    def status(self) -> StateWorkspaceStatus:
        return StateWorkspaceStatus(
            remote=self.remote,
            reachable=self.reachable,
            location=self.location,
            last_synced_at=self.last_synced_at,
            error=self.error,
        )


class LocalStateWorkspace(StateWorkspace):
    pass


def _advisory_lock_holder_arguments(
    lock_path: str | os.PathLike[str],
    *,
    python_executable: str = "python3",
) -> list[str]:
    return [
        python_executable,
        "-c",
        _remote_script("remote_lock_holder.py"),
        os.fspath(lock_path),
    ]


def _remote_advisory_lock_command(host: str, lock_path: str | os.PathLike[str]) -> list[str]:
    command = shlex.join(_advisory_lock_holder_arguments(lock_path))
    return ssh_arguments(host, command)


def _stop_lock_holder(process: subprocess.Popen[str]) -> None:
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    try:
        process.wait(timeout=STATE_LOCK_HOLDER_STOP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=STATE_LOCK_HOLDER_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=STATE_LOCK_HOLDER_STOP_TIMEOUT_SECONDS)


def _terminate_lock_holder(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=STATE_LOCK_HOLDER_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=STATE_LOCK_HOLDER_STOP_TIMEOUT_SECONDS)


def _lock_holder_error(process: subprocess.Popen[str], status: str) -> str:
    _terminate_lock_holder(process)
    stderr = process.stderr.read().strip() if process.stderr is not None else ""
    return stderr[:1000] or f"unexpected holder status {status!r}"


def _raise_lock_cancelled(process: subprocess.Popen[str], *, acquired: bool = False) -> None:
    _terminate_lock_holder(process)
    timing = "after acquiring" if acquired else "while waiting"
    raise RunLockCancelled(f"Run-lock acquisition was cancelled {timing}.")


class _HolderLines:
    """Whole-line reader for a lock holder's stdout.

    Polling the descriptor with ``select`` cannot see bytes already sitting in
    the text wrapper's buffer, so a contention that resolves inside one poll
    interval delivers ``contended`` and ``acquired`` in a single read and the
    second status becomes invisible. A reader thread owns ``readline`` and hands
    complete lines over a queue instead.
    """

    _CLOSED = object()

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process
        self._lines: queue.Queue[str | object] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        stream = self._process.stdout
        try:
            if stream is not None:
                for line in stream:
                    status = line.strip()
                    if status:
                        self._lines.put(status)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(self._CLOSED)

    def next_line(self, timeout: float) -> str | None:
        """Return the next status, ``""`` once the stream ends, ``None`` on timeout."""

        try:
            item = self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is self._CLOSED:
            self._lines.put(self._CLOSED)
            return ""
        assert isinstance(item, str)
        return item


def _wait_for_lock_holder(
    process: subprocess.Popen[str],
    lines: _HolderLines,
    location: str,
    *,
    on_wait: Callable[[str], None] | None,
    cancelled: Callable[[], bool] | None,
) -> None:
    deadline = time.monotonic() + STATE_LOCK_ATTEMPT_TIMEOUT_SECONDS
    contended = False
    waiting_reported = False
    while True:
        if cancelled is not None and cancelled():
            _raise_lock_cancelled(process)
        if not contended and time.monotonic() >= deadline:
            _terminate_lock_holder(process)
            raise StateUnavailable(
                f"Timed out after {STATE_LOCK_ATTEMPT_TIMEOUT_SECONDS:g} seconds while checking "
                f"canonical-state lock ownership at {location}."
            )
        status = lines.next_line(STATE_LOCK_POLL_INTERVAL_SECONDS)
        if status is None:
            continue
        if status == "":
            if cancelled is not None and cancelled():
                _raise_lock_cancelled(process)
            detail = _lock_holder_error(process, "holder exited without a status")
            raise StateUnavailable(
                f"Could not establish canonical-state lock ownership at {location}: {detail}"
            )
        if cancelled is not None and cancelled():
            _raise_lock_cancelled(process, acquired=status == _LOCK_ACQUIRED)
        if status == _LOCK_CONTENDED:
            contended = True
            if not waiting_reported and on_wait is not None:
                on_wait("Waiting for another graph-writing run to release canonical state.")
                waiting_reported = True
            continue
        if status == _LOCK_ACQUIRED:
            return
        _stop_lock_holder(process)
        if status == _LOCK_LEGACY_DIRECTORY:
            raise _LegacyLockDirectory(
                f"Canonical-state lock {location} is a legacy directory RCP could not reclaim. "
                "An empty one is removed automatically, but this directory still holds contents "
                "whose owner RCP cannot identify, so RCP preserved it. Inspect what is inside it, "
                "then use Retry in RCP."
            )
        if status == _LOCK_UNSAFE_ENTRY:
            raise _UnsafeLockEntry(
                f"Canonical-state lock {location} is not a regular file. RCP preserved it "
                "because replacing a directory, symlink, or special file cannot be proved safe. "
                "Inspect the project or deployment that created it, then use Retry in RCP."
            )
        detail = _lock_holder_error(process, status)
        raise StateUnavailable(
            f"Could not establish canonical-state lock ownership at {location}: {detail}"
        )


def _supervise_lock_holder(
    process: subprocess.Popen[str],
    lease: RunLockLease,
    stopped: threading.Event,
) -> None:
    while not stopped.wait(STATE_LOCK_POLL_INTERVAL_SECONDS):
        return_code = process.poll()
        if return_code is not None:
            lease._mark_lost(
                f"Canonical-state lock holder for {lease.location} exited unexpectedly "
                f"with status {return_code}."
            )
            return


def _raise_holder_command_lost(
    process: subprocess.Popen[str],
    lease: RunLockLease,
    message: str,
) -> None:
    lease._mark_lost(message)
    _terminate_lock_holder(process)
    lease.assert_owned()
    raise RunLockOwnershipLost(message)


def _send_lock_holder_command(
    process: subprocess.Popen[str],
    lines: _HolderLines,
    lease: RunLockLease,
    command: dict[str, object],
) -> dict[str, object]:
    if process.stdin is None or process.stdout is None:
        _raise_holder_command_lost(
            process,
            lease,
            f"Canonical-state lock holder channel for {lease.location} is unavailable.",
        )
    try:
        process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        process.stdin.flush()
    except (BrokenPipeError, OSError, ValueError) as exc:
        _raise_holder_command_lost(
            process,
            lease,
            f"Canonical-state lock holder channel for {lease.location} failed before apply: {exc}",
        )
    deadline = time.monotonic() + STATE_LOCK_ATTEMPT_TIMEOUT_SECONDS
    while True:
        if time.monotonic() >= deadline:
            _raise_holder_command_lost(
                process,
                lease,
                f"Canonical-state lock holder at {lease.location} stopped responding during apply.",
            )
        line = lines.next_line(STATE_LOCK_POLL_INTERVAL_SECONDS)
        if line is None:
            continue
        if line == "":
            _raise_holder_command_lost(
                process,
                lease,
                f"Canonical-state lock holder for {lease.location} "
                + ("exited" if process.poll() is not None else "closed")
                + " during apply.",
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            _raise_holder_command_lost(
                process,
                lease,
                f"Canonical-state lock holder for {lease.location} returned an invalid response: "
                f"{exc}",
            )
        if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
            _raise_holder_command_lost(
                process,
                lease,
                f"Canonical-state lock holder for {lease.location} returned an invalid response.",
            )
        return response


@contextmanager
def _process_advisory_lock(
    process_arguments: list[str],
    location: str,
    *,
    on_wait: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    on_lost: Callable[[str], None] | None = None,
) -> Iterator[RunLockLease]:
    if cancelled is not None and cancelled():
        raise RunLockCancelled("Run-lock acquisition was cancelled while waiting.")
    try:
        holder = subprocess.Popen(  # noqa: S603 - argv is constructed without a shell.
            process_arguments,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise StateUnavailable(
            f"Could not start canonical-state lock holder for {location}: {exc}"
        ) from exc
    if holder.stdout is None:
        _terminate_lock_holder(holder)
        raise StateUnavailable(f"Lock holder for {location} did not expose an ownership signal.")
    lines = _HolderLines(holder)
    try:
        _wait_for_lock_holder(
            holder,
            lines,
            location,
            on_wait=on_wait,
            cancelled=cancelled,
        )
    except BaseException:
        _terminate_lock_holder(holder)
        raise
    if cancelled is not None and cancelled():
        _raise_lock_cancelled(holder, acquired=True)
    lease: RunLockLease

    def owned_command(command: dict[str, object]) -> dict[str, object]:
        return _send_lock_holder_command(holder, lines, lease, command)

    lease = RunLockLease(
        location,
        on_lost=on_lost,
        owned=lambda: holder.poll() is None,
        command=owned_command,
    )
    supervisor_stop = threading.Event()
    supervisor = threading.Thread(
        target=_supervise_lock_holder,
        args=(holder, lease, supervisor_stop),
        daemon=True,
    )
    supervisor.start()
    try:
        yield lease
    finally:
        lease._begin_release()
        supervisor_stop.set()
        _stop_lock_holder(holder)
        supervisor.join(timeout=STATE_LOCK_HOLDER_STOP_TIMEOUT_SECONDS)


class SSHStateWorkspace(StateWorkspace):
    def __init__(self, root: Path, host: str, repository_path: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", host):
            raise ValueError("SSH host contains unsupported characters")
        remote_repository = PurePosixPath(repository_path)
        if not remote_repository.is_absolute() or str(remote_repository) == "/":
            raise ValueError("remote state repository must use a specific absolute path")
        super().__init__(root, f"{host}:{remote_repository}/.research")
        self.remote = True
        self.host = host
        self.remote_repository = remote_repository
        self.remote_root = remote_repository / ".research"
        self.lock_dir = self.remote_root / ".refresh.lock"
        self._last_refresh_monotonic = 0.0
        self._publication_lease: RunLockLease | None = None

    def _refresh_snapshot(self) -> bool:
        if not self._remote_manifest_exists():
            return False
        if self._publication_lease is not None:
            self._publication_lease.assert_owned()
            refreshed = self._sync_remote_tree()
            self._publication_lease.assert_owned()
            return refreshed
        with self._remote_advisory_lock(self.lock_dir) as lease:
            lease.assert_owned()
            refreshed = self._sync_remote_tree()
            lease.assert_owned()
            return refreshed

    def archive_research(self, *, expected_history_fingerprint: str | None = None) -> str:
        """Archive remote canonical state, then discard only its stale local mirror."""

        timestamp = _archive_timestamp()
        with self.snapshot_lock, self._remote_advisory_lock(self.lock_dir) as lease:
            lease.assert_owned()
            result = self._ssh(
                [
                    "python3",
                    "-c",
                    _remote_script("remote_archive_research.py"),
                    str(self.remote_root),
                    timestamp,
                    expected_history_fingerprint or "-",
                ]
            )
            lease.assert_owned()
            if result.returncode != 0:
                detail = result.stderr.strip() or "remote archive rename failed"
                if result.returncode == 3:
                    self._mark_reachable()
                    raise StateUnavailable(_RETAINED_HISTORY_CHANGED_MESSAGE)
                if result.returncode in {1, 2}:
                    self._mark_reachable()
                    raise StateUnavailable(
                        f"Could not archive canonical research at {self.location}; "
                        f"the original directory remains intact: {detail}"
                    )
                self._mark_unreachable(detail)
                raise StateUnavailable(
                    f"Could not confirm whether canonical research at {self.location} was "
                    f"archived: {detail}"
                )

            remote_archive = PurePosixPath(result.stdout.strip())
            base_name = f"{self.remote_root.name}.archive-{timestamp}"
            suffix = remote_archive.name.removeprefix(base_name)
            valid_suffix = not suffix or (
                re.fullmatch(r"-(?:[2-9]|[1-9][0-9]+)", suffix) is not None
            )
            if (
                not remote_archive.is_absolute()
                or remote_archive.parent != self.remote_root.parent
                or not remote_archive.name.startswith(base_name)
                or not valid_suffix
            ):
                message = "Remote archive rename returned an invalid archive location."
                self._mark_unreachable(message)
                raise StateUnavailable(message)

            try:
                shutil.rmtree(self.root)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._mark_reachable()
                raise StateUnavailable(
                    f"Canonical research was archived at {self.host}:{remote_archive}, but its "
                    "stale local mirror could not be cleared; fresh initialization must not "
                    f"continue: {exc}"
                ) from exc
            self.last_synced_at = None
            self._mark_reachable()
            return f"{self.host}:{remote_archive}"

    def _remote_manifest_exists(self) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_exists = self._ssh(["test", "-f", str(self.remote_root / "manifest.toml")])
        if manifest_exists.returncode != 0:
            if manifest_exists.returncode == 1:
                self._mark_reachable()
                return False
            self._mark_unreachable(manifest_exists.stderr)
            raise StateUnavailable(self.error or "canonical state is unreachable")
        return True

    def probe_remote_patch_log_head(self) -> tuple[bool, int | None]:
        """Read only the remote patch-log head without changing workspace health."""

        result = self._ssh(
            [
                "sh",
                "-c",
                _REMOTE_PATCH_LOG_HEAD_SCRIPT,
                "rcp-patch-log-head",
                str(self.remote_root / "patches"),
            ],
            timeout=REMOTE_STATE_HEAD_PROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return False, None
        head = result.stdout.strip()
        if not head:
            return True, None
        if not re.fullmatch(r"[0-9]{6}\.json", head):
            return False, None
        return True, int(head[:-5])

    def _sync_remote_tree(self) -> bool:
        remote = f"{self.host}:{shlex.quote(str(self.remote_root))}/"
        result = subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                "--exclude=.refresh.lock",
                "--exclude=.agent-run.lock",
                "--exclude=.publish",
                *rsync_ssh_arguments(),
                remote,
                f"{self.root}/",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            self._mark_unreachable(result.stderr)
            raise StateUnavailable(self.error or "canonical state sync failed")
        self._mark_reachable(synced=True)
        return True

    def backup_source_root(self, destination: Path) -> Path:
        """Export remote research optimistically without refresh/publication locks."""

        direct_root = self._remote_backup_direct_root_inventory()
        if not destination.is_absolute() or ".." in destination.parts:
            raise ValueError("backup source export requires one normalized absolute directory")
        try:
            metadata = destination.lstat()
            entries = list(destination.iterdir())
        except OSError as exc:
            raise StateUnavailable("Backup source export staging is unavailable.") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or entries
        ):
            raise StateUnavailable(
                "Backup source export staging is not one empty private directory."
            )
        if not direct_root:
            if self._remote_backup_direct_root_inventory() != direct_root:
                raise StateUnavailable("The remote backup root changed during its export.")
            self._mark_reachable()
            return destination
        remote = f"{self.host}:{shlex.quote(str(self.remote_root))}/"
        try:
            result = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--delete",
                    "--exclude=/patches/.batch-*/***",
                    "--exclude=/patches/.unconfirmed-*",
                    "--exclude=/branches/.unconfirmed-*/***",
                    "--exclude=/branches/*/graph.json",
                    "--exclude=/branches/*/research.md",
                    "--exclude=/branches/*/glossary.json",
                    "--exclude=/branches/*/proposals.json",
                    "--exclude=/branches/*/coverage.json",
                    "--exclude=/branches/*/patches/.unconfirmed-*",
                    "--include=/manifest.toml",
                    "--include=/scope-base.json",
                    "--include=/patches/***",
                    "--include=/branches/***",
                    "--include=/chat/***",
                    "--include=/facts/***",
                    "--include=/paper/***",
                    "--exclude=.refresh.lock",
                    "--exclude=.agent-run.lock",
                    "--exclude=.append.lock",
                    "--exclude=.chat.lock",
                    "--exclude=.publish",
                    "--exclude=*",
                    *rsync_ssh_arguments(),
                    remote,
                    f"{destination}/",
                ],
                capture_output=True,
                text=True,
                timeout=BACKUP_REMOTE_EXPORT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._mark_unreachable(str(exc))
            raise StateUnavailable("The remote backup source export was unavailable.") from exc
        if (
            len(result.stdout.encode("utf-8", errors="replace")) > 256 * 1024
            or len(result.stderr.encode("utf-8", errors="replace")) > 256 * 1024
        ):
            self._mark_unreachable("remote backup source export returned too much output")
            raise StateUnavailable(self.error or "The remote backup source export failed.")
        if result.returncode:
            self._mark_unreachable(result.stderr)
            raise StateUnavailable(self.error or "The remote backup source export failed.")
        if self._remote_backup_direct_root_inventory() != direct_root:
            raise StateUnavailable("The remote backup root changed during its export.")
        self._mark_reachable()
        return destination

    def _remote_backup_direct_root_inventory(self) -> tuple[tuple[str, str], ...]:
        result = self._ssh(
            [
                "python3",
                "-c",
                _remote_script("remote_backup_inventory.py"),
                str(self.remote_root),
            ],
            timeout=REMOTE_STATE_HEAD_PROBE_TIMEOUT_SECONDS,
        )
        if (
            result.returncode != 0
            or len(result.stdout) > 256 * 1024
            or len(result.stderr) > 256 * 1024
        ):
            self._mark_unreachable(result.stderr.decode(errors="replace"))
            raise StateUnavailable("The remote backup root could not be classified.")
        try:
            document = json.loads(result.stdout)
        except (UnicodeError, json.JSONDecodeError) as exc:
            self._mark_unreachable("remote backup root returned an invalid inventory")
            raise StateUnavailable("The remote backup root inventory is invalid.") from exc
        if not isinstance(document, list) or len(document) > BACKUP_INVENTORY_MAX_ENTRIES:
            raise StateUnavailable("The remote backup root inventory is invalid.")
        known = (
            BACKUP_RESEARCH_CANONICAL_ROOTS
            | BACKUP_RESEARCH_DELEGATED_ROOTS
            | BACKUP_RESEARCH_EXCLUSIONS
        )
        seen: set[str] = set()
        inventory: list[tuple[str, str]] = []
        for entry in document:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"name", "kind"}
                or not isinstance(entry["name"], str)
                or entry["kind"] not in {"directory", "file", "other"}
                or entry["name"] in seen
            ):
                raise StateUnavailable("The remote backup root inventory is invalid.")
            seen.add(entry["name"])
            if entry["name"] not in known:
                raise StateUnavailable(
                    "The remote project contains an unclassified durable research root."
                )
            inventory.append((entry["name"], entry["kind"]))
        return tuple(inventory)

    def refresh_if_stale(
        self,
        max_age_seconds: float = REMOTE_STATE_RECONCILE_WINDOW_SECONDS,
    ) -> bool:
        with self.snapshot_lock:
            if time.monotonic() - self._last_refresh_monotonic < max_age_seconds:
                return self.reachable
            return self._refresh_snapshot()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.snapshot_lock, self._remote_advisory_lock(self.lock_dir) as lease:
            self._publication_lease = lease
            try:
                lease.assert_owned()
                if self._remote_manifest_exists():
                    self._sync_remote_tree()
                lease.assert_owned()
                yield
                lease.assert_owned()
            finally:
                self._publication_lease = None

    @contextmanager
    def _publication_lock(self) -> Iterator[RunLockLease]:
        if self._publication_lease is not None:
            self._publication_lease.assert_owned()
            yield self._publication_lease
            return
        with self._remote_advisory_lock(self.lock_dir) as lease:
            self._publication_lease = lease
            try:
                yield lease
            finally:
                self._publication_lease = None

    @contextmanager
    def run_lock(
        self,
        *,
        on_wait: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        on_lost: Callable[[str], None] | None = None,
    ) -> Iterator[RunLockLease]:
        with self._remote_advisory_lock(
            self.remote_root / ".agent-run.lock",
            on_wait=on_wait,
            cancelled=cancelled,
            on_lost=on_lost,
        ) as lease:
            yield lease

    @contextmanager
    def _remote_advisory_lock(
        self,
        lock_path: PurePosixPath,
        *,
        on_wait: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        on_lost: Callable[[str], None] | None = None,
    ) -> Iterator[RunLockLease]:
        prepared = self._ssh(["mkdir", "-p", str(self.remote_root)])
        if prepared.returncode:
            self._mark_unreachable(prepared.stderr)
            raise StateUnavailable(self.error or "canonical state is unreachable")
        location = f"{self.host}:{lock_path}"

        def ownership_lost(message: str) -> None:
            self._mark_unreachable(message)
            if on_lost is not None:
                on_lost(message)

        try:
            with _process_advisory_lock(
                _remote_advisory_lock_command(self.host, lock_path),
                location,
                on_wait=on_wait,
                cancelled=cancelled,
                on_lost=ownership_lost,
            ) as lease:
                self._mark_reachable()
                yield lease
        except (_LegacyLockDirectory, _UnsafeLockEntry):
            self._mark_reachable()
            raise
        except RunLockCancelled:
            self._mark_reachable()
            raise
        except StateUnavailable as exc:
            self._mark_unreachable(str(exc))
            raise

    def publish(self, relative_paths: list[Path | str]) -> None:
        with self.snapshot_lock, self._publication_lock() as lease:
            self._publish(relative_paths, lease)

    def restore_exact_file(
        self,
        relative_path: Path | str,
        source: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        relative = _validated_relative_path(relative_path)
        with self.snapshot_lock, self._publication_lock() as lease:
            _stage_exact_restore_file(
                self.root,
                relative,
                source,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            self._restore_remote_exact(
                source=self.root / relative,
                remote_relative=relative,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                external=False,
                lease=lease,
            )

    def restore_kept_artifact(
        self,
        name: str,
        source: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        safe_name = _validated_kept_artifact_name(name)
        if not 1 <= expected_size <= CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("restored artifact bytes are outside the supported size range")
        self._restore_remote_repository_file(
            directory="artifacts",
            name=safe_name,
            source=source,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )

    def restore_kept_result_view(
        self,
        name: str,
        source: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        safe_name = _validated_kept_view_name(name)
        if not 1 <= expected_size <= CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("restored result-view bytes are outside the supported size range")
        self._restore_remote_repository_file(
            directory="views",
            name=safe_name,
            source=source,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )

    def _restore_remote_repository_file(
        self,
        *,
        directory: Literal["artifacts", "views"],
        name: str,
        source: Path,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        _require_restore_source(
            source,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        with self.snapshot_lock, self._publication_lock() as lease:
            self._restore_remote_exact(
                source=source,
                remote_relative=Path(directory) / name,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                external=True,
                lease=lease,
            )

    def _restore_remote_exact(
        self,
        *,
        source: Path,
        remote_relative: Path,
        expected_sha256: str,
        expected_size: int,
        external: bool,
        lease: RunLockLease,
    ) -> None:
        stage = self.remote_root / ".publish" / f"restore-{os.getpid()}-{time.time_ns()}"
        prepared = self._ssh(["mkdir", "-p", str(stage)])
        if prepared.returncode:
            self._mark_unreachable(prepared.stderr)
            raise StateUnavailable(self.error or "restored project staging failed")
        destination = f"{self.host}:{shlex.quote(str(stage))}/content.bin"
        try:
            result = subprocess.run(
                ["rsync", "-a", *rsync_ssh_arguments(), str(source), destination],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._mark_unreachable(str(exc))
            raise StateUnavailable(self.error or "restored project staging failed") from exc
        if result.returncode:
            self._mark_unreachable(result.stderr)
            raise StateUnavailable(self.error or "restored project staging failed")
        response = lease._run_owned_command(
            {
                "op": "restore-exact",
                "root": str(self.remote_root),
                "stage": str(stage),
                "path": remote_relative.as_posix(),
                "sha256": expected_sha256,
                "size": expected_size,
                "external": external,
            }
        )
        if not response["ok"]:
            self._mark_reachable()
            raise StateUnavailable(
                str(response.get("error") or "restored project publication failed")
            )
        self._mark_reachable(synced=True)

    def keep_result_view(
        self,
        *,
        source_name: str,
        project_name: str,
        data: bytes,
        today: date | None = None,
    ) -> str:
        base_name = _result_view_base_name(source_name, project_name, today)
        content = _validated_view_bytes(data)
        stage = self.remote_root / ".publish" / f"view-{os.getpid()}-{time.time_ns()}"
        with self.snapshot_lock, self._publication_lock() as lease:
            prepared = self._ssh(["mkdir", "-p", str(stage)])
            if prepared.returncode:
                self._mark_unreachable(prepared.stderr)
                raise StateUnavailable(self.error or "canonical state is unreachable")
            with tempfile.TemporaryDirectory(prefix="rcp-kept-view-") as temporary:
                source = Path(temporary) / "content.html"
                source.write_bytes(content)
                destination = f"{self.host}:{shlex.quote(str(stage))}/"
                try:
                    result = subprocess.run(
                        ["rsync", "-a", *rsync_ssh_arguments(), str(source), destination],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self._mark_unreachable(str(exc))
                    raise StateUnavailable(
                        self.error or "repository result-view staging failed"
                    ) from exc
            if result.returncode:
                self._mark_unreachable(result.stderr)
                raise StateUnavailable(self.error or "repository result-view staging failed")
            try:
                response = lease._run_owned_command(
                    {
                        "op": "keep-view",
                        "root": str(self.remote_root),
                        "stage": str(stage),
                        "base_name": base_name,
                    }
                )
            except RunLockOwnershipLost as exc:
                message = (
                    "Canonical-state ownership was lost while keeping the result view; "
                    "no existing repository view was overwritten."
                )
                self._mark_unreachable(message)
                raise RunLockOwnershipLost(message) from exc
            if not response["ok"]:
                self._mark_reachable()
                message = str(response.get("error") or "repository result-view keep failed")
                raise StateUnavailable(message)
            chosen = response.get("name")
            if (
                not isinstance(chosen, str)
                or _KEPT_VIEW_NAME_PATTERN.fullmatch(chosen) is None
                or not _is_collision_view_name(chosen, base_name)
            ):
                self._mark_unreachable("Remote result-view keep returned an invalid file name.")
                raise StateUnavailable(self.error or "repository result-view keep failed")
            self._mark_reachable(synced=True)
            return chosen

    def read_kept_result_view(
        self,
        name: str,
        *,
        max_bytes: int = CHAT_ARTIFACT_MAX_FILE_BYTES,
    ) -> bytes:
        safe_name = _validated_kept_view_name(name)
        limit = _validated_view_read_limit(max_bytes)
        result = self._ssh_bytes(
            [
                "python3",
                "-c",
                _remote_script("remote_read_kept_view.py"),
                str(self.remote_repository),
                safe_name,
                str(limit),
            ],
            timeout=REMOTE_ARTIFACT_READ_TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            if len(result.stdout) > limit:
                self._mark_reachable()
                raise ValueError("kept result view exceeds the read limit")
            self._mark_reachable()
            return result.stdout
        if result.returncode == _REMOTE_VIEW_MISSING:
            self._mark_reachable()
            raise FileNotFoundError(safe_name)
        if result.returncode == _REMOTE_VIEW_TOO_LARGE:
            self._mark_reachable()
            raise ValueError("kept result view exceeds the read limit")
        if result.returncode == _REMOTE_VIEW_UNSAFE:
            self._mark_reachable()
            raise ValueError("kept result view is not a readable regular file")
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        self._mark_unreachable(detail or "canonical state is unreachable")
        raise StateUnavailable(self.error or "canonical state is unreachable")

    def keep_artifact(
        self,
        *,
        source_name: str,
        project_name: str,
        data: bytes,
        today: date | None = None,
    ) -> str:
        base_name = _artifact_base_name(source_name, project_name, today)
        if not isinstance(data, bytes) or not 1 <= len(data) <= CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("artifact bytes are outside the supported size range")
        stage = self.remote_root / ".publish" / f"artifact-{os.getpid()}-{time.time_ns()}"
        with self.snapshot_lock, self._publication_lock() as lease:
            prepared = self._ssh(["mkdir", "-p", str(stage)])
            if prepared.returncode:
                self._mark_unreachable(prepared.stderr)
                raise StateUnavailable(self.error or "canonical state is unreachable")
            with tempfile.TemporaryDirectory(prefix="rcp-kept-artifact-") as temporary:
                source = Path(temporary) / "content.bin"
                source.write_bytes(data)
                destination = f"{self.host}:{shlex.quote(str(stage))}/"
                try:
                    result = subprocess.run(
                        ["rsync", "-a", *rsync_ssh_arguments(), str(source), destination],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self._mark_unreachable(str(exc))
                    raise StateUnavailable(
                        self.error or "repository artifact staging failed"
                    ) from exc
            if result.returncode:
                self._mark_unreachable(result.stderr)
                raise StateUnavailable(self.error or "repository artifact staging failed")
            response = lease._run_owned_command(
                {
                    "op": "keep-artifact",
                    "root": str(self.remote_root),
                    "stage": str(stage),
                    "base_name": base_name,
                }
            )
            if not response["ok"]:
                self._mark_reachable()
                raise StateUnavailable(str(response.get("error") or "artifact Keep failed"))
            chosen = response.get("name")
            if (
                not isinstance(chosen, str)
                or _KEPT_ARTIFACT_NAME_PATTERN.fullmatch(chosen) is None
                or not _is_collision_artifact_name(chosen, base_name)
            ):
                self._mark_unreachable("Remote artifact Keep returned an invalid file name.")
                raise StateUnavailable(self.error or "artifact Keep failed")
            self._mark_reachable(synced=True)
            return chosen

    def read_kept_artifact(
        self,
        name: str,
        *,
        max_bytes: int = CHAT_ARTIFACT_MAX_FILE_BYTES,
    ) -> bytes:
        safe_name = _validated_kept_artifact_name(name)
        limit = _validated_view_read_limit(max_bytes)
        result = self._ssh_bytes(
            [
                "python3",
                "-c",
                _remote_script("remote_read_kept_view.py"),
                str(self.remote_repository),
                "artifacts",
                safe_name,
                str(limit),
            ],
            timeout=REMOTE_ARTIFACT_READ_TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            if len(result.stdout) > limit:
                self._mark_reachable()
                raise ValueError("kept artifact exceeds the read limit")
            self._mark_reachable()
            return result.stdout
        if result.returncode == _REMOTE_VIEW_MISSING:
            self._mark_reachable()
            raise FileNotFoundError(safe_name)
        if result.returncode == _REMOTE_VIEW_TOO_LARGE:
            self._mark_reachable()
            raise ValueError("kept artifact exceeds the read limit")
        if result.returncode == _REMOTE_VIEW_UNSAFE:
            self._mark_reachable()
            raise ValueError("kept artifact is not a readable regular file")
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        self._mark_unreachable(detail or "canonical state is unreachable")
        raise StateUnavailable(self.error or "canonical state is unreachable")

    def replace_kept_artifact(self, name: str, data: bytes) -> None:
        safe_name = _validated_kept_artifact_name(name)
        if not isinstance(data, bytes) or not 1 <= len(data) <= CHAT_ARTIFACT_MAX_FILE_BYTES:
            raise ValueError("artifact bytes are outside the supported size range")
        stage = self.remote_root / ".publish" / f"artifact-{os.getpid()}-{time.time_ns()}"
        with self.snapshot_lock, self._publication_lock() as lease:
            prepared = self._ssh(["mkdir", "-p", str(stage)])
            if prepared.returncode:
                self._mark_unreachable(prepared.stderr)
                raise StateUnavailable(self.error or "canonical state is unreachable")
            with tempfile.TemporaryDirectory(prefix="rcp-artifact-revision-") as temporary:
                source = Path(temporary) / "content.bin"
                source.write_bytes(data)
                destination = f"{self.host}:{shlex.quote(str(stage))}/"
                try:
                    result = subprocess.run(
                        ["rsync", "-a", *rsync_ssh_arguments(), str(source), destination],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self._mark_unreachable(str(exc))
                    raise StateUnavailable(
                        self.error or "repository artifact staging failed"
                    ) from exc
            if result.returncode:
                self._mark_unreachable(result.stderr)
                raise StateUnavailable(self.error or "repository artifact staging failed")
            response = lease._run_owned_command(
                {
                    "op": "replace-artifact",
                    "root": str(self.remote_root),
                    "stage": str(stage),
                    "name": safe_name,
                }
            )
            if not response["ok"]:
                self._mark_reachable()
                raise StateUnavailable(str(response.get("error") or "artifact update failed"))
            self._mark_reachable(synced=True)

    def _publish(self, relative_paths: list[Path | str], lease: RunLockLease) -> None:
        sources: list[str] = []
        for raw_relative in relative_paths:
            relative = _validated_relative_path(raw_relative)
            source = self.root / relative
            if not source.is_file():
                continue
            sources.append(str(relative))
        if not sources:
            return
        stage = self.remote_root / ".publish" / f"files-{os.getpid()}-{time.time_ns()}"
        prepared = self._ssh(["mkdir", "-p", str(stage)])
        if prepared.returncode:
            self._mark_unreachable(prepared.stderr)
            raise StateUnavailable(self.error or "canonical state is unreachable")
        destination = f"{self.host}:{shlex.quote(str(stage))}/"
        result = subprocess.run(
            ["rsync", "-aR", *rsync_ssh_arguments(), *sources, destination],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            self._mark_unreachable(result.stderr)
            raise StateUnavailable(self.error or "canonical state publish failed")
        try:
            response = lease._run_owned_command(
                {
                    "op": "apply",
                    "root": str(self.remote_root),
                    "stage": str(stage),
                    "paths": sources,
                }
            )
        except RunLockOwnershipLost as exc:
            message = (
                "Canonical-state ownership was lost during ordinary file apply; a prefix may "
                "have been applied while the lock was held. Retry in a new transaction to "
                "restage and idempotently apply the full requested file set."
            )
            self._mark_unreachable(message)
            raise RunLockOwnershipLost(message) from exc
        if not response["ok"]:
            message = str(response.get("error") or "canonical state apply failed")
            self._mark_unreachable(message)
            raise StateUnavailable(self.error or "canonical state apply failed")
        self._mark_reachable(synced=True)

    def publish_committed_batch(
        self,
        relative_paths: list[Path | str],
        batch_directory: Path | str,
    ) -> None:
        """Commit history first, then idempotently publish its derived files."""

        with self.snapshot_lock, self._publication_lock() as lease:
            batch = _validated_batch_directory(batch_directory)
            self._publish_committed_history(
                relative_paths,
                batch,
                commit_is_directory=True,
                lease=lease,
            )

    def publish_committed_patch(
        self,
        relative_paths: list[Path | str],
        patch_path: Path | str,
    ) -> None:
        """Commit one patch first, then idempotently publish its derived files."""

        with self.snapshot_lock, self._publication_lock() as lease:
            patch = _validated_patch_path(patch_path)
            self._publish_committed_history(
                relative_paths,
                patch,
                commit_is_directory=False,
                lease=lease,
            )

    def publish_committed_branch_file(
        self,
        relative_paths: list[Path | str],
        commit_path: Path | str,
    ) -> None:
        """Commit immutable branch metadata or one merge receipt before derived files."""

        with self.snapshot_lock, self._publication_lock() as lease:
            commit = _validated_branch_commit_path(commit_path)
            self._publish_committed_history(
                relative_paths,
                commit,
                commit_is_directory=False,
                lease=lease,
            )

    def _publish_committed_history(
        self,
        relative_paths: list[Path | str],
        commit_path: Path,
        *,
        commit_is_directory: bool,
        lease: RunLockLease,
    ) -> None:
        sources: list[str] = []
        for raw_relative in relative_paths:
            relative = _validated_relative_path(raw_relative)
            source = self.root / relative
            if source.is_file():
                sources.append(str(relative))
        if commit_is_directory:
            commit_prefix = f"{commit_path.as_posix()}/"
            includes_commit = any(path.startswith(commit_prefix) for path in sources)
        else:
            includes_commit = commit_path.as_posix() in sources
        if not includes_commit:
            raise ValueError("committed history publication is missing its patch files")

        if commit_path.parts and commit_path.parts[0] == "branches":
            digest = hashlib.sha256(commit_path.as_posix().encode("utf-8")).hexdigest()[:16]
            stage_name = f"branch-{digest}-{commit_path.name}"
        else:
            stage_name = commit_path.name if commit_is_directory else f"patch-{commit_path.name}"
        stage = self.remote_root / ".publish" / stage_name
        prepared = self._ssh(["mkdir", "-p", str(stage)])
        if prepared.returncode:
            self._mark_unreachable(prepared.stderr)
            raise BatchPublishFailed(
                self.error or "canonical state staging failed",
                commit_status="absent",
            )
        destination = f"{self.host}:{shlex.quote(str(stage))}/"
        result = subprocess.run(
            ["rsync", "-aR", *rsync_ssh_arguments(), *sources, destination],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            self._mark_unreachable(result.stderr)
            raise BatchPublishFailed(
                self.error or "canonical state staging failed",
                commit_status="absent",
            )

        command: dict[str, object] = {
            "op": "apply",
            "root": str(self.remote_root),
            "stage": str(stage),
            "paths": sources,
            "commit": commit_path.as_posix(),
            "commit_is_directory": commit_is_directory,
        }
        try:
            response = lease._run_owned_command(command)
        except RunLockOwnershipLost as exc:
            self._raise_reconciled_commit_failure(commit_path, commit_is_directory, exc)
        if response["ok"] and response.get("commit_status") == "present":
            self._mark_reachable(synced=True)
            return
        status = response.get("commit_status")
        message = str(response.get("error") or "canonical state commit failed")
        if status == "present":
            try:
                repaired = lease._run_owned_command(command)
            except RunLockOwnershipLost as exc:
                self._raise_reconciled_commit_failure(commit_path, commit_is_directory, exc)
            if repaired["ok"] and repaired.get("commit_status") == "present":
                self._mark_reachable(synced=True)
                return
            status = repaired.get("commit_status")
            message = str(repaired.get("error") or message)
        if status not in {"absent", "present"}:
            self._raise_reconciled_commit_failure(
                commit_path,
                commit_is_directory,
                StateUnavailable(message),
            )
        self._mark_unreachable(message)
        raise BatchPublishFailed(
            self.error or "canonical state commit failed",
            commit_status=status,
        )

    def _raise_reconciled_commit_failure(
        self,
        commit_path: Path,
        commit_is_directory: bool,
        error: Exception,
    ) -> None:
        marker_arguments = [
            "test",
            "-d" if commit_is_directory else "-f",
            str(self.remote_root / commit_path),
        ]
        marker = self._ssh(marker_arguments)
        if marker.returncode == 0:
            status: Literal["absent", "present", "unknown"] = "present"
            message = marker.stderr or str(error)
            self._mark_unreachable(message)
            raise BatchPublishFailed(
                self.error or "canonical state commit failed",
                commit_status=status,
            ) from error

        deadline = time.monotonic() + STATE_LOCK_ATTEMPT_TIMEOUT_SECONDS
        try:
            with self._remote_advisory_lock(
                self.lock_dir,
                cancelled=lambda: time.monotonic() >= deadline,
            ):
                marker = self._ssh(marker_arguments)
        except (RunLockCancelled, StateUnavailable) as reacquire_error:
            status = "unknown"
            message = str(reacquire_error) or marker.stderr or str(error)
        else:
            if marker.returncode == 0:
                status = "present"
            elif marker.returncode == 1:
                status = "absent"
            else:
                status = "unknown"
            message = marker.stderr or str(error)
        self._mark_unreachable(message)
        raise BatchPublishFailed(
            self.error or "canonical state commit failed",
            commit_status=status,
        ) from error

    def _ssh(
        self,
        arguments: list[str],
        *,
        timeout: float = 30,
    ) -> subprocess.CompletedProcess[str]:
        command = " ".join(shlex.quote(argument) for argument in arguments)
        try:
            return subprocess.run(
                ssh_arguments(self.host, command),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess([], 255, "", str(exc))

    def _ssh_bytes(
        self,
        arguments: list[str],
        *,
        timeout: float = 30,
    ) -> subprocess.CompletedProcess[bytes]:
        command = " ".join(shlex.quote(argument) for argument in arguments)
        try:
            return subprocess.run(
                ssh_arguments(self.host, command),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess([], 255, b"", str(exc).encode())

    def _mark_reachable(self, *, synced: bool = False) -> None:
        self.reachable = True
        self.error = None
        self._last_refresh_monotonic = time.monotonic()
        if synced:
            self.last_synced_at = datetime.now(UTC)

    def _mark_unreachable(self, message: str) -> None:
        self.reachable = False
        self.error = message.strip() or "canonical state is unreachable"
        self._last_refresh_monotonic = time.monotonic()


def _validated_relative_path(value: Path | str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"state publish path must be relative: {relative}")
    if relative.parts and relative.parts[0] == "branches":
        _validated_branch_relative_path(relative)
    return relative


def _validated_branch_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError("branch paths require a canonical episode UUIDv4") from exc
    if str(parsed) != value or parsed.version != 4:
        raise ValueError("branch paths require a canonical episode UUIDv4")
    return value


def _validated_branch_relative_path(relative: Path) -> None:
    parts = relative.parts
    if len(parts) not in {3, 4}:
        raise ValueError(f"invalid canonical branch path: {relative}")
    _validated_branch_id(parts[1])
    if len(parts) == 3 and parts[2] in {
        "branch.json",
        "graph.json",
        "glossary.json",
        "proposals.json",
        "coverage.json",
        "research.md",
    }:
        return
    if len(parts) == 4:
        if parts[2] == "patches" and re.fullmatch(r"[0-9]{6}\.json", parts[3]):
            return
        if parts[2] == "merges" and re.fullmatch(r"[a-f0-9]{64}\.json", parts[3]):
            return
    raise ValueError(f"invalid canonical branch path: {relative}")


def _validated_branch_commit_path(value: Path | str) -> Path:
    relative = _validated_relative_path(value)
    parts = relative.parts
    if len(parts) == 3 and parts[0] == "branches" and parts[2] == "branch.json":
        _validated_branch_id(parts[1])
        return relative
    if len(parts) == 4 and parts[0] == "branches":
        _validated_branch_id(parts[1])
        if parts[2] == "merges" and re.fullmatch(r"[a-f0-9]{64}\.json", parts[3]):
            return relative
    raise ValueError(f"invalid committed branch path: {relative}")


def _validated_batch_directory(value: Path | str) -> Path:
    relative = _validated_relative_path(value)
    if relative.parent != Path("patches") or not relative.name.startswith("batch-"):
        raise ValueError(f"invalid committed patch batch directory: {relative}")
    return relative


def _validated_patch_path(value: Path | str) -> Path:
    relative = _validated_relative_path(value)
    if relative.parent == Path("patches") and re.fullmatch(r"[0-9]{6}\.json", relative.name):
        return relative
    parts = relative.parts
    if (
        len(parts) == 4
        and parts[0] == "branches"
        and parts[2] == "patches"
        and re.fullmatch(r"[0-9]{6}\.json", parts[3])
    ):
        _validated_branch_id(parts[1])
        return relative
    raise ValueError(f"invalid committed patch path: {relative}")


def prepare_state_workspace(bootstrap: Manifest, data_dir: Path) -> tuple[Manifest, StateWorkspace]:
    state_repository = bootstrap.repository_map[bootstrap.state.repository]
    machine = bootstrap.machine_map[state_repository.machine]
    if not machine.host:
        return bootstrap, LocalStateWorkspace(
            bootstrap.research_dir,
            str(bootstrap.research_dir),
        )

    workspace = state_workspace_for_probe(bootstrap, data_dir)
    assert isinstance(workspace, SSHStateWorkspace)
    cache_root = workspace.root
    try:
        remote_exists = workspace.refresh()
    except StateUnavailable:
        if not (cache_root / "manifest.toml").is_file():
            raise
    else:
        if not remote_exists:
            _discard_absent_remote_snapshot(cache_root)
    cache_manifest = cache_root / "manifest.toml"
    if not cache_manifest.is_file():
        cache_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bootstrap.path, cache_manifest)
    manifest = load_manifest(cache_manifest)
    _validate_remote_identity(bootstrap, manifest, machine.host, state_repository.path)
    return manifest, workspace


def _discard_absent_remote_snapshot(cache_root: Path) -> None:
    """Never let a confirmed-absent remote project inherit an old local mirror."""

    try:
        mode = cache_root.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        raise StateUnavailable(f"Could not inspect stale canonical-state mirror: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise StateUnavailable(
            f"Stale canonical-state mirror is not a regular directory: {cache_root}"
        )
    try:
        shutil.rmtree(cache_root)
    except OSError as exc:
        raise StateUnavailable(
            f"Could not clear stale canonical-state mirror before fresh initialization: {exc}"
        ) from exc


def state_workspace_for_probe(bootstrap: Manifest, data_dir: Path) -> StateWorkspace:
    """Construct the canonical workspace without refreshing or taking its lock."""

    state_repository = bootstrap.repository_map[bootstrap.state.repository]
    machine = bootstrap.machine_map[state_repository.machine]
    if not machine.host:
        return LocalStateWorkspace(
            bootstrap.research_dir,
            str(bootstrap.research_dir),
        )
    cache_key = hashlib.sha256(f"{machine.host}\0{state_repository.path}".encode()).hexdigest()[:16]
    cache_root = data_dir / "state-cache" / cache_key / ".research"
    return SSHStateWorkspace(cache_root, machine.host, state_repository.path)


def _validate_remote_identity(
    bootstrap: Manifest,
    canonical: Manifest,
    expected_host: str,
    expected_path: str,
) -> None:
    canonical_repository = canonical.repository_map[canonical.state.repository]
    canonical_machine = canonical.machine_map[canonical_repository.machine]
    if canonical.name != bootstrap.name:
        raise ValueError("cached canonical manifest belongs to a different project")
    if canonical_machine.host != expected_host or canonical_repository.path != expected_path:
        raise ValueError("canonical manifest changed its own remote state locator")
