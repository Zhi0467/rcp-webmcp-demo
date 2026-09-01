from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rcp.limits import (
    REMOTE_SOURCE_CACHE_MAX_BYTES,
    REMOTE_SOURCE_CACHE_MAX_COUNT,
    REMOTE_SOURCE_CACHE_TTL_SECONDS,
    SESSION_SLICE_CACHE_MAX_BYTES,
    SESSION_SLICE_CACHE_MAX_COUNT,
    SESSION_SLICE_CACHE_TTL_SECONDS,
)


class CacheLimits(BaseModel):
    ttl_seconds: int = Field(gt=0)
    max_count: int = Field(gt=0)
    max_bytes: int = Field(gt=0)


class CacheMetrics(BaseModel):
    root: str
    bytes: int = 0
    count: int = 0
    limits: CacheLimits
    oldest_accessed_at: datetime | None = None
    reclaimable_bytes: int = 0
    reclaimable_count: int = 0


class RebuildableCacheMetrics(BaseModel):
    remote_sources: CacheMetrics
    session_slices: CacheMetrics


REMOTE_SOURCE_CACHE_LIMITS = CacheLimits(
    ttl_seconds=REMOTE_SOURCE_CACHE_TTL_SECONDS,
    max_count=REMOTE_SOURCE_CACHE_MAX_COUNT,
    max_bytes=REMOTE_SOURCE_CACHE_MAX_BYTES,
)
SESSION_SLICE_CACHE_LIMITS = CacheLimits(
    ttl_seconds=SESSION_SLICE_CACHE_TTL_SECONDS,
    max_count=SESSION_SLICE_CACHE_MAX_COUNT,
    max_bytes=SESSION_SLICE_CACHE_MAX_BYTES,
)

_ACCESS_FILE = ".last-access.json"
_Layout = Literal["files", "directories"]
_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[Path, threading.Lock] = {}
_ROOT_PINS: dict[Path, dict[Path, int]] = {}
_PROJECT_CACHE_DIRECTORY = "project-caches"
_SOURCE_CACHE_DIRECTORY = "source-cache"
_SESSION_SLICE_DIRECTORY = "session-slices"
_PROJECT_CACHE_KEY = re.compile(r"[0-9a-f]{64}")


def project_cache_roots(data_dir: Path, project_id: str) -> tuple[Path, Path]:
    """Return safe rebuildable-cache roots owned by one canonical project id."""

    if not project_id.strip():
        raise ValueError("project_id must be non-empty")
    key = hashlib.sha256(project_id.encode()).hexdigest()
    root = data_dir / _PROJECT_CACHE_DIRECTORY / key
    return root / _SOURCE_CACHE_DIRECTORY, root / _SESSION_SLICE_DIRECTORY


def discover_project_cache_roots(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find only cache directories created by :func:`project_cache_roots`."""

    parent = data_dir / _PROJECT_CACHE_DIRECTORY
    if not parent.is_dir() or parent.is_symlink():
        return []
    roots: list[tuple[Path, Path]] = []
    for child in sorted(parent.iterdir(), key=lambda path: path.name):
        if (
            child.is_symlink()
            or not child.is_dir()
            or _PROJECT_CACHE_KEY.fullmatch(child.name) is None
        ):
            continue
        roots.append((child / _SOURCE_CACHE_DIRECTORY, child / _SESSION_SLICE_DIRECTORY))
    return roots


def legacy_shared_cache_roots(data_dir: Path) -> tuple[Path, Path]:
    """Return the exact rebuildable roots used by the former shared layout."""

    return data_dir / _SOURCE_CACHE_DIRECTORY, data_dir / _SESSION_SLICE_DIRECTORY


class RebuildableCache:
    """Bound one rebuildable cache root using explicit access timestamps.

    Provider mtimes are intentionally ignored. A previously untracked entry is
    treated as newly discovered, recorded in the access index, and becomes
    eligible for TTL/LRU eviction only from that point forward.
    """

    def __init__(
        self,
        root: Path,
        limits: CacheLimits,
        *,
        layout: _Layout,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = root
        self.limits = limits
        self.layout = layout
        self._clock = clock or (lambda: datetime.now(UTC))
        resolved_root = root.resolve(strict=False)
        with _LOCKS_GUARD:
            self._lock = _ROOT_LOCKS.setdefault(resolved_root, threading.Lock())
            self._pins = _ROOT_PINS.setdefault(resolved_root, {})

    @contextmanager
    def pin(self, paths: Iterable[Path]) -> Iterator[None]:
        """Keep cache entries reachable for the lifetime of an active consumer."""

        with self.pin_scope() as add:
            for path in paths:
                add(path)
            yield

    @contextmanager
    def pin_scope(self) -> Iterator[Callable[[Path], None]]:
        """Return a mutable pin set that can protect entries before they exist."""

        owned: set[Path] = set()

        def add(path: Path) -> None:
            if not _is_within(path, self.root):
                return
            resolved = path.resolve(strict=False)
            with self._lock:
                if resolved in owned:
                    return
                self._pins[resolved] = self._pins.get(resolved, 0) + 1
                owned.add(resolved)

        try:
            yield add
        finally:
            with self._lock:
                for path in owned:
                    remaining = self._pins.get(path, 0) - 1
                    if remaining > 0:
                        self._pins[path] = remaining
                    else:
                        self._pins.pop(path, None)

    def touch(self, path: Path) -> None:
        """Record access to one cache entry, ignoring paths outside this root."""

        with self._lock:
            entries = self._entries()
            entry = self._entry_for_path(path, entries)
            if entry is None:
                return
            accesses = self._accesses(entries)
            accesses[self._key(entry)] = self._now()
            self._write_accesses(accesses, entries)

    def metrics(self, *, active_paths: Iterable[Path] = ()) -> CacheMetrics:
        with self._lock:
            entries = self._entries()
            accesses = self._accesses(entries)
            active = self._active_entries(active_paths, entries)
            self._write_accesses(accesses, entries)
            return self._metrics(entries, accesses, active)

    def sweep(self, *, active_paths: Iterable[Path] = ()) -> CacheMetrics:
        """Expire by TTL first, then evict deterministic LRU entries to limits."""

        with self._lock:
            entries = self._entries()
            accesses = self._accesses(entries)
            active = self._active_entries(active_paths, entries)
            now = self._now()
            sizes = {entry: _entry_size(entry) for entry in entries}

            expired = [
                entry
                for entry in entries
                if entry not in active
                and now - accesses[self._key(entry)] >= self.limits.ttl_seconds
            ]
            for entry in sorted(expired, key=lambda item: self._key(item)):
                self._remove(entry)
                entries.remove(entry)
                sizes.pop(entry, None)
                accesses.pop(self._key(entry), None)

            count = len(entries)
            byte_count = sum(sizes.values())
            candidates = sorted(
                (entry for entry in entries if entry not in active),
                key=lambda item: (accesses[self._key(item)], self._key(item)),
            )
            while (
                count > self.limits.max_count or byte_count > self.limits.max_bytes
            ) and candidates:
                entry = candidates.pop(0)
                size = sizes.pop(entry)
                self._remove(entry)
                entries.remove(entry)
                accesses.pop(self._key(entry), None)
                count -= 1
                byte_count -= size

            self._prune_empty_directories()
            self._write_accesses(accesses, entries)
            return self._metrics(entries, accesses, active & entries)

    def clear(self, *, active_paths: Iterable[Path] = ()) -> CacheMetrics:
        """Remove only rebuildable entries under this root, preserving active ones."""

        with self._lock:
            entries = self._entries()
            active = self._active_entries(active_paths, entries)
            for entry in sorted(entries - active, key=self._key):
                self._remove(entry)
            self._prune_empty_directories()
            remaining = self._entries()
            accesses = self._accesses(remaining)
            self._write_accesses(accesses, remaining)
            return self._metrics(remaining, accesses, active & remaining)

    def contains(self, path: Path) -> bool:
        return _is_within(path, self.root)

    def _entries(self) -> set[Path]:
        if self.root.is_symlink() or not self.root.is_dir():
            return set()
        if self.layout == "directories":
            return {
                path
                for path in self.root.iterdir()
                if path.is_dir() and not path.is_symlink() and not path.name.startswith(".slice-")
            }
        return {
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.name != _ACCESS_FILE
            and not path.name.startswith(f"{_ACCESS_FILE}.")
        }

    def _accesses(self, entries: set[Path]) -> dict[str, float]:
        if self.root.is_symlink():
            return {}
        raw: object = {}
        with suppress(OSError, json.JSONDecodeError):
            raw = json.loads((self.root / _ACCESS_FILE).read_text(encoding="utf-8"))
        stored = raw if isinstance(raw, dict) else {}
        now = self._now()
        return {
            self._key(entry): (
                float(stored[self._key(entry)])
                if isinstance(stored.get(self._key(entry)), int | float)
                else now
            )
            for entry in entries
        }

    def _write_accesses(self, accesses: dict[str, float], entries: set[Path]) -> None:
        if self.root.is_symlink():
            return
        if not entries and not self.root.exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            self._key(entry): accesses[self._key(entry)]
            for entry in sorted(entries, key=self._key)
            if self._key(entry) in accesses
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=f"{_ACCESS_FILE}.", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.root / _ACCESS_FILE)
        finally:
            temporary.unlink(missing_ok=True)

    def _entry_for_path(self, path: Path, entries: set[Path]) -> Path | None:
        if not _is_within(path, self.root):
            return None
        if self.layout == "files":
            return next((entry for entry in entries if _same_path(entry, path)), None)
        return next(
            (entry for entry in entries if _same_path(entry, path) or _is_within(path, entry)),
            None,
        )

    def _active_entries(self, paths: Iterable[Path], entries: set[Path]) -> set[Path]:
        active: set[Path] = set()
        for path in (*paths, *self._pins):
            entry = self._entry_for_path(path, entries)
            if entry is not None:
                active.add(entry)
        return active

    def _metrics(
        self,
        entries: set[Path],
        accesses: dict[str, float],
        active: set[Path],
    ) -> CacheMetrics:
        sizes = {entry: _entry_size(entry) for entry in entries}
        reclaimable = entries - active
        oldest = min((accesses[self._key(entry)] for entry in entries), default=None)
        return CacheMetrics(
            root=str(self.root),
            bytes=sum(sizes.values()),
            count=len(entries),
            limits=self.limits,
            oldest_accessed_at=datetime.fromtimestamp(oldest, UTC) if oldest is not None else None,
            reclaimable_bytes=sum(sizes[entry] for entry in reclaimable),
            reclaimable_count=len(reclaimable),
        )

    def _remove(self, entry: Path) -> None:
        if not _is_within(entry, self.root) or entry == self.root:
            raise ValueError("cache eviction target is outside its rebuildable root")
        if entry.is_symlink():
            return
        if entry.is_dir():
            for directory, _, _ in os.walk(entry):
                Path(directory).chmod(0o700)
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)

    def _prune_empty_directories(self) -> None:
        if self.layout != "files" or self.root.is_symlink() or not self.root.is_dir():
            return
        directories = sorted(
            (path for path in self.root.rglob("*") if path.is_dir() and not path.is_symlink()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            with suppress(OSError):
                directory.rmdir()

    def _key(self, entry: Path) -> str:
        return entry.relative_to(self.root).as_posix()

    def _now(self) -> float:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.timestamp()


def _entry_size(path: Path) -> int:
    if path.is_file() and not path.is_symlink():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)
