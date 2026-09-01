from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import posixpath
import shlex
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rcp.config import Manifest, RepositoryConfig
from rcp.limits import (
    REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS,
    SOURCE_ORIGINAL_COPY_BUFFER_BYTES,
)
from rcp.providers import PROVIDERS, ProviderId
from rcp.sources.cache import (
    REMOTE_SOURCE_CACHE_LIMITS,
    SESSION_SLICE_CACHE_LIMITS,
    RebuildableCache,
    RebuildableCacheMetrics,
)
from rcp.sources.record_parsing import normalize_path, normalize_record, path_matches_roots
from rcp.transport.ssh import rsync_ssh_arguments, ssh_arguments


class ConversationRecord(BaseModel):
    uuid: str
    timestamp: datetime | None = None
    role: Literal["user", "assistant", "system", "tool", "unknown"] = "unknown"
    text: str = ""
    raw_type: str = ""


class AppChatOrigin(BaseModel):
    machine: str
    host: str = ""
    root: str


class LocalSourceIdentity(BaseModel):
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class OriginalConversationSource:
    """One private byte-exact snapshot of an indexed provider original."""

    path: Path
    content_sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RemoteConversationIndex:
    """Matched remote sessions plus exact best-effort omission counts."""

    sessions: tuple[dict[str, Any], ...]
    unmatched_files: int
    malformed_files: int


class ConversationSession(BaseModel):
    key: str
    provider: ProviderId | Literal["app_chat"]
    source_machine: str
    truth_repository: str
    session_id: str
    cwd: str
    path: str
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    last_uuid: str | None = None
    record_count: int = 0
    thread_source: str | None = None
    parent_session_id: str | None = None
    originator: str | None = None
    source_kind: str | None = None
    remote_source_host: str | None = None
    remote_source_path: str | None = None
    source_path_is_remote: bool = False
    local_source_identity: LocalSourceIdentity | None = None


class ConversationSlice(BaseModel):
    path: str
    record_count: int
    content_sha256: str


class ConversationIndex(BaseModel):
    generated_at: datetime
    sessions: list[ConversationSession] = Field(default_factory=list)
    unmatched_files: int = 0
    malformed_files: int = 0
    source_errors: list[str] = Field(default_factory=list)

    def for_scope(self, aliases: list[str]) -> list[ConversationSession]:
        allowed = set(aliases)
        return [session for session in self.sessions if session.truth_repository in allowed]


class ConversationIndexer:
    def __init__(
        self,
        manifest: Manifest,
        cache_root: Path | None = None,
        *,
        app_chat_origin: AppChatOrigin | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.manifest = manifest
        self.cache_root = cache_root
        state_repository = manifest.repository_map[manifest.state.repository]
        self.app_chat_origin = app_chat_origin or AppChatOrigin(
            machine=state_repository.machine,
            root=str(manifest.research_dir / "chat"),
        )
        source_root = cache_root or manifest.research_dir.parent / ".rcp-source-cache"
        self._remote_source_cache = RebuildableCache(
            source_root,
            REMOTE_SOURCE_CACHE_LIMITS,
            layout="files",
            clock=clock,
        )
        self._session_slice_cache = RebuildableCache(
            self._slice_root(),
            SESSION_SLICE_CACHE_LIMITS,
            layout="directories",
            clock=clock,
        )
        # Session key -> the id a stale cursor was re-resolved to, so the run that
        # read it can say out loud that the source file had been rewritten.
        self.cursor_repairs: dict[str, str] = {}
        self._local_metadata_cache: dict[
            tuple[str, Path], tuple[LocalSourceIdentity, dict[str, Any]]
        ] = {}
        self._local_metadata_repository_paths: tuple[str, ...] = ()

    def build(
        self,
        *,
        execution_machine: str | None = None,
        cache_remote_sources: bool = True,
        active_cache_paths: Iterable[str | Path] = (),
        pin_artifact: Callable[[Path], None] | None = None,
    ) -> ConversationIndex:
        declared_active = tuple(Path(path) for path in active_cache_paths)
        self.sweep_rebuildable_caches(active_paths=declared_active)
        sessions: list[ConversationSession] = []
        unmatched = 0
        malformed = 0
        local_machines = [item.alias for item in self.manifest.machines if not item.host]
        source_machine = (
            local_machines[0] if local_machines else self.manifest.agent_profile("refresh").run_on
        )
        source_errors: list[str] = []

        roots = [
            (profile.id, Path(root).expanduser())
            for profile in PROVIDERS.values()
            for root in profile.session_roots(self.manifest.sources, remote=False)
        ]
        repository_paths = [item.path for item in self.manifest.repositories]
        repository_path_key = tuple(repository_paths)
        if repository_path_key != self._local_metadata_repository_paths:
            self._local_metadata_cache.clear()
            self._local_metadata_repository_paths = repository_path_key
        seen_local_sources: set[tuple[str, Path]] = set()

        for provider, root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.jsonl"):
                if provider == "claude" and "subagents" in path.parts:
                    continue
                cache_key = (provider, path)
                seen_local_sources.add(cache_key)
                try:
                    identity_before = _local_source_identity(path)
                    cached = self._local_metadata_cache.get(cache_key)
                    if cached is not None and cached[0] == identity_before:
                        identity_after, metadata = cached
                    else:
                        metadata = self._inspect(path, provider, repository_paths)
                        identity_after = _local_source_identity(path)
                        if identity_before == identity_after:
                            self._local_metadata_cache[cache_key] = (
                                identity_after,
                                metadata,
                            )
                        else:
                            self._local_metadata_cache.pop(cache_key, None)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    self._local_metadata_cache.pop(cache_key, None)
                    malformed += 1
                    continue
                repository = self._match_repository(metadata["cwd"])
                if repository is None:
                    unmatched += 1
                    continue
                session_id = metadata["session_id"] or path.stem
                sessions.append(
                    ConversationSession(
                        key=f"{repository.alias}/{source_machine}/{provider}/{session_id}",
                        provider=provider,
                        source_machine=source_machine,
                        truth_repository=repository.alias,
                        session_id=session_id,
                        cwd=metadata["cwd"],
                        path=str(path),
                        first_timestamp=metadata["first_timestamp"],
                        last_timestamp=metadata["last_timestamp"],
                        last_uuid=metadata["last_uuid"],
                        record_count=metadata["record_count"],
                        thread_source=metadata.get("thread_source"),
                        parent_session_id=metadata.get("parent_session_id"),
                        originator=metadata.get("originator"),
                        source_kind=metadata.get("source_kind"),
                        local_source_identity=(
                            identity_after if identity_before == identity_after else None
                        ),
                    )
                )

        for cache_key in self._local_metadata_cache.keys() - seen_local_sources:
            del self._local_metadata_cache[cache_key]

        for machine in self.manifest.machines:
            if not machine.host:
                continue
            remote_roots = [
                (profile.id, root)
                for profile in PROVIDERS.values()
                for root in profile.session_roots(self.manifest.sources, remote=True)
            ]
            for provider, root in remote_roots:
                try:
                    remote_index = self._inspect_remote_root(
                        machine.host, root, provider, machine.alias
                    )
                except (OSError, ValueError) as exc:
                    source_errors.append(f"{machine.alias}/{provider}: {exc}")
                    continue
                unmatched += remote_index.unmatched_files
                malformed += remote_index.malformed_files
                matched: list[tuple[dict[str, Any], RepositoryConfig]] = []
                for metadata in remote_index.sessions:
                    repository = self._match_repository(metadata["cwd"])
                    if repository is None:
                        unmatched += 1
                        continue
                    matched.append((metadata, repository))
                original_stays_remote = (
                    execution_machine == machine.alias or not cache_remote_sources
                )
                if original_stays_remote:
                    # Keep the provider original on its own machine. Slice
                    # materialization runs the normalizer there and transfers only
                    # the cursor-bounded derived records.
                    source_paths = {
                        metadata["path"]: Path(metadata["path"]) for metadata, _ in matched
                    }
                else:
                    try:
                        source_paths = self._cache_remote_files(
                            machine.host,
                            machine.alias,
                            provider,
                            [metadata["path"] for metadata, _ in matched],
                            pin_artifact=pin_artifact,
                        )
                    except OSError as exc:
                        source_errors.append(f"{machine.alias}/{provider}: {exc}")
                        continue
                for metadata, repository in matched:
                    session_id = metadata["session_id"] or Path(metadata["path"]).stem
                    sessions.append(
                        ConversationSession(
                            key=(f"{repository.alias}/{machine.alias}/{provider}/{session_id}"),
                            provider=provider,
                            source_machine=machine.alias,
                            truth_repository=repository.alias,
                            session_id=session_id,
                            cwd=metadata["cwd"],
                            path=str(source_paths[metadata["path"]]),
                            first_timestamp=metadata["first_timestamp"],
                            last_timestamp=metadata["last_timestamp"],
                            last_uuid=metadata["last_uuid"],
                            record_count=metadata["record_count"],
                            thread_source=metadata.get("thread_source"),
                            parent_session_id=metadata.get("parent_session_id"),
                            originator=metadata.get("originator"),
                            source_kind=metadata.get("source_kind"),
                            remote_source_host=machine.host,
                            remote_source_path=metadata["path"],
                            source_path_is_remote=original_stays_remote,
                        )
                    )

        chat_root = self.manifest.research_dir / "chat"
        if chat_root.exists():
            state_alias = self.manifest.state.repository
            for path in chat_root.glob("*.jsonl"):
                metadata = self._inspect(path, "app_chat")
                session_id = metadata["session_id"] or path.stem
                remote_source_path = (
                    posixpath.join(self.app_chat_origin.root, path.name)
                    if self.app_chat_origin.host
                    else None
                )
                sessions.append(
                    ConversationSession(
                        key=(f"{state_alias}/{self.app_chat_origin.machine}/app_chat/{session_id}"),
                        provider="app_chat",
                        source_machine=self.app_chat_origin.machine,
                        truth_repository=state_alias,
                        session_id=session_id,
                        cwd=metadata["cwd"] or str(self.manifest.research_dir.parent),
                        path=str(path),
                        first_timestamp=metadata["first_timestamp"],
                        last_timestamp=metadata["last_timestamp"],
                        last_uuid=metadata["last_uuid"],
                        record_count=metadata["record_count"],
                        thread_source=metadata.get("thread_source"),
                        parent_session_id=metadata.get("parent_session_id"),
                        originator=metadata.get("originator"),
                        source_kind=metadata.get("source_kind"),
                        remote_source_host=self.app_chat_origin.host or None,
                        remote_source_path=remote_source_path,
                    )
                )

        sessions.sort(key=lambda item: (item.truth_repository, item.provider, item.session_id))
        self.sweep_rebuildable_caches(active_paths=declared_active)
        return ConversationIndex(
            generated_at=datetime.now(UTC),
            sessions=sessions,
            unmatched_files=unmatched,
            malformed_files=malformed,
            source_errors=source_errors,
        )

    def read_records(
        self,
        session: ConversationSession,
        from_uuid: str | None = None,
        *,
        _repaired: bool = False,
    ) -> Iterator[ConversationRecord]:
        if session.record_count == 0:
            if session.last_uuid is not None:
                raise ValueError(f"Session {session.key!r} has an inconsistent empty source index.")
            if from_uuid is not None:
                raise ValueError(_missing_cursor_message(session, from_uuid))
            return
        if session.last_uuid is None:
            raise ValueError(f"Session {session.key!r} has no indexed terminal record.")
        if (
            from_uuid == session.last_uuid
            and session.local_source_identity is not None
            and _local_source_identity_matches(session)
        ):
            return

        seen_cursor = from_uuid is None
        indexed_records = 0
        with (
            self._source_path(session) as source_path,
            source_path.open(encoding="utf-8") as handle,
        ):
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                indexed_records += 1
                if indexed_records > session.record_count:
                    break
                raw = json.loads(line)
                record = _normalize_record(raw, session.provider, line_number)
                at_terminal = indexed_records == session.record_count
                if at_terminal and record.uuid != session.last_uuid:
                    raise ValueError(
                        f"Conversation source for {session.key!r} changed before its indexed "
                        "terminal record; rebuild the source index and retry."
                    )
                if not seen_cursor:
                    if record.uuid == from_uuid:
                        seen_cursor = True
                else:
                    yield record
                if at_terminal:
                    break

        if indexed_records < session.record_count:
            raise ValueError(
                f"Conversation source for {session.key!r} ended before its indexed terminal "
                "record; rebuild the source index and retry."
            )
        if not seen_cursor:
            repaired = None if _repaired else self._cursor_after_source_rewrite(session, from_uuid)
            if repaired is None:
                raise ValueError(_missing_cursor_message(session, from_uuid))
            # The provider rewrote the file, so every line-derived id moved while the
            # records themselves did not. Resuming from the same record under its new
            # id reads exactly what a stable id would have — nothing is reread.
            self.cursor_repairs[session.key] = repaired
            yield from self.read_records(session, from_uuid=repaired, _repaired=True)

    def _cursor_after_source_rewrite(
        self, session: ConversationSession, cursor: str | None
    ) -> str | None:
        """Re-resolve a line-derived cursor id by the content digest inside it.

        A record with no id of its own gets `line-<n>-<digest>`, so compacting or
        rewriting a session file invalidates every stored cursor even though the
        conversation is unchanged. A digest occurring exactly once still identifies
        its record; anything else is a real gap the human must reseed explicitly.
        """
        if not cursor or not cursor.startswith("line-"):
            return None
        digest = cursor.rpartition("-")[2]
        if not digest:
            return None
        match: str | None = None
        try:
            with (
                self._source_path(session) as source_path,
                source_path.open(encoding="utf-8") as handle,
            ):
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    candidate = _normalize_record(
                        json.loads(line), session.provider, line_number
                    ).uuid
                    if candidate.startswith("line-") and candidate.rpartition("-")[2] == digest:
                        if match is not None:
                            return None
                        match = candidate
        except (OSError, json.JSONDecodeError):
            return None
        return match

    def materialize_slice(
        self,
        session: ConversationSession,
        *,
        from_uuid: str | None = None,
        active_paths: Iterable[str | Path] = (),
        pin_artifact: Callable[[Path], None] | None = None,
    ) -> ConversationSlice:
        root = self._slice_root()
        root.mkdir(parents=True, exist_ok=True)
        declared_active = tuple(Path(path) for path in active_paths)
        if pin_artifact is not None:
            # A cached provider source is part of the task too. Pin it before
            # the first sweep/read; local originals fall outside both cache roots.
            pin_artifact(Path(session.path))
        self._session_slice_cache.sweep(active_paths=declared_active)
        temporary = Path(tempfile.mkdtemp(prefix=".slice-", dir=root))
        temporary_path = temporary / "records.jsonl"
        content_digest = hashlib.sha256()
        record_count = 0
        try:
            remote_slice = (
                session.remote_source_host
                and session.remote_source_path
                and (session.source_path_is_remote or not Path(session.path).is_file())
            )
            if remote_slice:
                remote_records = temporary / "remote-records.jsonl"
                repaired = self._write_remote_slice(session, from_uuid, remote_records)
                if repaired is not None:
                    self.cursor_repairs[session.key] = repaired
                with remote_records.open("rb") as source, temporary_path.open("wb") as handle:
                    for raw_line in source:
                        if not raw_line.strip():
                            continue
                        record = ConversationRecord.model_validate_json(raw_line)
                        line = (record.model_dump_json() + "\n").encode("utf-8")
                        handle.write(line)
                        content_digest.update(line)
                        record_count += 1
                    handle.flush()
                    os.fsync(handle.fileno())
                remote_records.unlink()
            else:
                with temporary_path.open("wb") as handle:
                    for record in self.read_records(session, from_uuid=from_uuid):
                        line = (record.model_dump_json() + "\n").encode("utf-8")
                        handle.write(line)
                        content_digest.update(line)
                        record_count += 1
                    handle.flush()
                    os.fsync(handle.fileno())
            content_sha256 = content_digest.hexdigest()
            identity = json.dumps(
                {
                    "session": session.key,
                    "after": from_uuid,
                    "through": session.last_uuid,
                    "content_sha256": content_sha256,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            slice_id = hashlib.sha256(identity).hexdigest()
            destination = root / slice_id
            final_path = destination / temporary_path.name
            if pin_artifact is not None:
                # Pin the final identity before making it visible so another
                # project's concurrent sweep cannot observe an unpinned entry.
                pin_artifact(final_path)
            temporary_path.chmod(0o444)
            temporary.chmod(0o555)
            try:
                temporary.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
                _remove_temporary_slice(temporary)
            if not final_path.is_file() or _file_sha256(final_path) != content_sha256:
                raise ValueError(f"Immutable session slice cache is corrupt for {session.key!r}.")
            self._session_slice_cache.touch(final_path)
            self._session_slice_cache.sweep(active_paths=(*declared_active, final_path))
            return ConversationSlice(
                path=str(final_path),
                record_count=record_count,
                content_sha256=content_sha256,
            )
        except (OSError, json.JSONDecodeError) as exc:
            _remove_temporary_slice(temporary)
            raise ValueError(
                f"Could not materialize the indexed evidence slice for {session.key!r}: {exc}"
            ) from exc
        except Exception:
            _remove_temporary_slice(temporary)
            raise

    def _slice_root(self) -> Path:
        if self.cache_root is not None:
            return self.cache_root.parent / "session-slices"
        return self.manifest.research_dir.parent / ".rcp-session-slices"

    def session_artifact_root(self) -> Path:
        """Root shared by normalized slices and their bounded routing indexes."""

        return self._slice_root()

    def register_session_artifact(
        self,
        path: str | Path,
        *,
        active_paths: Iterable[str | Path] = (),
    ) -> None:
        artifact = Path(path)
        active = (*active_paths, artifact)
        self._session_slice_cache.touch(artifact)
        self._session_slice_cache.sweep(active_paths=tuple(Path(item) for item in active))

    def cache_metrics(self, *, active_paths: Iterable[str | Path] = ()) -> RebuildableCacheMetrics:
        paths = tuple(Path(path) for path in active_paths)
        return RebuildableCacheMetrics(
            remote_sources=self._remote_source_cache.metrics(active_paths=paths),
            session_slices=self._session_slice_cache.metrics(active_paths=paths),
        )

    def sweep_rebuildable_caches(
        self, *, active_paths: Iterable[str | Path] = ()
    ) -> RebuildableCacheMetrics:
        paths = tuple(Path(path) for path in active_paths)
        return RebuildableCacheMetrics(
            remote_sources=self._remote_source_cache.sweep(active_paths=paths),
            session_slices=self._session_slice_cache.sweep(active_paths=paths),
        )

    def clear_rebuildable_caches(
        self, *, active_paths: Iterable[str | Path] = ()
    ) -> RebuildableCacheMetrics:
        """Clear only remote-log copies and derived session slices."""

        paths = tuple(Path(path) for path in active_paths)
        return RebuildableCacheMetrics(
            remote_sources=self._remote_source_cache.clear(active_paths=paths),
            session_slices=self._session_slice_cache.clear(active_paths=paths),
        )

    @contextmanager
    def pin_rebuildable_paths(
        self,
        paths: Iterable[str | Path],
    ) -> Iterator[None]:
        """Protect files used by one active task from every cache instance."""

        resolved = tuple(Path(path) for path in paths)
        with self._remote_source_cache.pin(resolved), self._session_slice_cache.pin(resolved):
            yield

    @contextmanager
    def pin_rebuildable_scope(self) -> Iterator[Callable[[Path], None]]:
        """Protect task artifacts as their final paths become known."""

        with (
            self._remote_source_cache.pin_scope() as pin_remote,
            self._session_slice_cache.pin_scope() as pin_session,
        ):

            def add(path: Path) -> None:
                pin_remote(path)
                pin_session(path)

            yield add

    @contextmanager
    def original_source(
        self,
        session: ConversationSession,
    ) -> Iterator[OriginalConversationSource]:
        """Snapshot one provider original from its indexed local or SSH account."""

        if session.provider == "app_chat":
            raise ValueError("RCP chats are not provider-native sources.")
        with tempfile.TemporaryDirectory(prefix="rcp-provider-original-") as temporary_root:
            snapshot = Path(temporary_root) / "conversation.jsonl"
            if session.remote_source_host and session.remote_source_path:
                self._fetch_remote_file(
                    session.remote_source_host,
                    session.remote_source_path,
                    snapshot,
                )
            else:
                _copy_local_original(session, snapshot)
            metadata = snapshot.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("provider conversation source is not one regular file")
            yield OriginalConversationSource(
                path=snapshot,
                content_sha256=_file_sha256(snapshot),
                size_bytes=metadata.st_size,
            )

    def original_repository_alias(
        self,
        session: ConversationSession,
        copied_path: Path,
    ) -> str | None:
        """Reinspect copied native bytes through the indexer's matching owner."""

        if session.provider == "app_chat":
            return None
        metadata = self._inspect(
            copied_path,
            session.provider,
            [item.path for item in self.manifest.repositories],
        )
        repository = self._match_repository(metadata["cwd"])
        return None if repository is None else repository.alias

    @contextmanager
    def _source_path(self, session: ConversationSession) -> Iterator[Path]:
        source_path = Path(session.path)
        if not session.source_path_is_remote and source_path.is_file():
            self._remote_source_cache.touch(source_path)
            yield source_path
            return
        if session.remote_source_host and session.remote_source_path:
            with tempfile.TemporaryDirectory(prefix="rcp-remote-source-") as temporary_root:
                temporary = Path(temporary_root) / "conversation.jsonl"
                self._fetch_remote_file(
                    session.remote_source_host,
                    session.remote_source_path,
                    temporary,
                )
                yield temporary
            return
        yield source_path

    @staticmethod
    def _fetch_remote_file(host: str, remote_path: str, destination: Path) -> None:
        try:
            result = subprocess.run(
                [
                    "rsync",
                    "-a",
                    *rsync_ssh_arguments(),
                    f"{host}:{shlex.quote(remote_path)}",
                    str(destination),
                ],
                capture_output=True,
                text=True,
                timeout=REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OSError(
                "remote conversation reconstruction timed out after "
                f"{REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS} seconds"
            ) from exc
        try:
            metadata = destination.lstat()
        except OSError:
            metadata = None
        if result.returncode or metadata is None or not stat.S_ISREG(metadata.st_mode):
            raise OSError(
                result.stderr.strip() or f"remote conversation source is unavailable: {remote_path}"
            )

    @staticmethod
    def _write_remote_slice(
        session: ConversationSession,
        from_uuid: str | None,
        destination: Path,
    ) -> str | None:
        assert session.remote_source_host is not None
        assert session.remote_source_path is not None
        payload = json.dumps(
            {
                "path": session.remote_source_path,
                "provider": session.provider,
                "record_count": session.record_count,
                "last_uuid": session.last_uuid,
                "from_uuid": from_uuid,
                "session_key": session.key,
            }
        )
        command = f"python3 -c {shlex.quote(_REMOTE_SLICE_SCRIPT)} {shlex.quote(payload)}"
        try:
            with destination.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    ssh_arguments(session.remote_source_host, command),
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS,
                    check=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
        except subprocess.TimeoutExpired as exc:
            raise OSError(
                "remote conversation slicing timed out after "
                f"{REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS} seconds"
            ) from exc
        status: dict[str, Any] = {}
        for line in reversed(result.stderr.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                status = candidate
                break
        if result.returncode:
            detail = status.get("error")
            raise OSError(detail or result.stderr.strip() or "remote conversation slicing failed")
        repaired = status.get("cursor_repair")
        return repaired if isinstance(repaired, str) and repaired else None

    def _match_repository(self, cwd: str) -> RepositoryConfig | None:
        candidates = [
            repository
            for repository in self.manifest.repositories
            if path_matches_roots(cwd, [repository.path])
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda repository: len(normalize_path(repository.path)))

    @staticmethod
    def _inspect(
        path: Path, provider: str, repository_paths: list[str] | None = None
    ) -> dict[str, Any]:
        cwd = ""
        session_id = ""
        first_timestamp = None
        last_timestamp = None
        last_uuid = None
        record_count = 0
        thread_source = "user" if provider in {"claude", "app_chat"} else None
        parent_session_id = None
        originator = None
        source_kind = provider if provider in {"claude", "app_chat"} else None
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                record_count += 1
                if provider == "codex" and raw.get("type") == "session_meta":
                    payload = raw.get("payload", {})
                    cwd = payload.get("cwd", cwd)
                    session_id = payload.get("id") or payload.get("session_id") or session_id
                    thread_source = payload.get("thread_source") or thread_source
                    originator = payload.get("originator") or originator
                    source = payload.get("source")
                    if isinstance(source, str):
                        source_kind = source
                    elif isinstance(source, dict):
                        source_kind = "subagent" if "subagent" in source else source_kind
                        subagent = source.get("subagent")
                        spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
                        if isinstance(spawn, dict):
                            parent_session_id = spawn.get("parent_thread_id") or parent_session_id
                else:
                    cwd = raw.get("cwd", cwd)
                    session_id = raw.get("sessionId", session_id)
                if cwd and repository_paths and not path_matches_roots(cwd, repository_paths):
                    break
                record = _normalize_record(raw, provider, line_number)
                last_uuid = record.uuid
                if record.timestamp is not None:
                    first_timestamp = first_timestamp or record.timestamp
                    last_timestamp = record.timestamp
        if provider != "app_chat" and not cwd:
            raise ValueError("conversation has no cwd")
        return {
            "cwd": cwd,
            "session_id": session_id,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "last_uuid": last_uuid,
            "record_count": record_count,
            "thread_source": thread_source,
            "parent_session_id": parent_session_id,
            "originator": originator,
            "source_kind": source_kind,
        }

    def _inspect_remote_root(
        self, host: str, root: str, provider: str, machine_alias: str
    ) -> RemoteConversationIndex:
        repository_paths = [
            item.path for item in self.manifest.repositories if item.machine == machine_alias
        ]
        payload = json.dumps(
            {"root": root, "provider": provider, "repository_paths": repository_paths}
        )
        command = f"python3 -c {shlex.quote(_REMOTE_INDEX_SCRIPT)} {shlex.quote(payload)}"
        try:
            result = subprocess.run(
                ssh_arguments(host, command),
                capture_output=True,
                text=True,
                timeout=REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OSError(
                f"remote {provider} conversation index timed out after "
                f"{REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS} seconds"
            ) from exc
        if result.returncode:
            raise OSError(result.stderr.strip() or f"remote index exited {result.returncode}")
        items: list[dict[str, Any]] = []
        summary: tuple[int, int] | None = None
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            kind = item.pop("kind", None)
            if kind == "summary":
                if summary is not None or set(item) != {"unmatched_files", "malformed_files"}:
                    raise ValueError("remote conversation index returned an invalid summary")
                counts = (item["unmatched_files"], item["malformed_files"])
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in counts
                ):
                    raise ValueError("remote conversation index returned invalid omission counts")
                summary = counts
                continue
            if kind != "session":
                raise ValueError("remote conversation index returned an unknown record")
            item["first_timestamp"] = _parse_datetime(item.get("first_timestamp"))
            item["last_timestamp"] = _parse_datetime(item.get("last_timestamp"))
            items.append(item)
        if summary is None:
            raise ValueError("remote conversation index did not return its omission summary")
        return RemoteConversationIndex(
            sessions=tuple(items),
            unmatched_files=summary[0],
            malformed_files=summary[1],
        )

    def _cache_remote_files(
        self,
        host: str,
        machine_alias: str,
        provider: str,
        remote_paths: list[str],
        *,
        pin_artifact: Callable[[Path], None] | None = None,
    ) -> dict[str, Path]:
        if self.cache_root is None:
            raise OSError("remote conversation cache is not configured")
        directory = self.cache_root / machine_alias / provider
        directory.mkdir(parents=True, exist_ok=True)
        if not remote_paths:
            return {}
        cached = {path: directory / path.lstrip("/") for path in remote_paths}
        if pin_artifact is not None:
            for path in cached.values():
                pin_artifact(path)
        try:
            result = subprocess.run(
                [
                    "rsync",
                    "-aR",
                    *rsync_ssh_arguments(),
                    *(f"{host}:{shlex.quote(path)}" for path in remote_paths),
                    f"{directory}/",
                ],
                capture_output=True,
                text=True,
                timeout=REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OSError(
                f"remote {provider} conversation cache timed out after "
                f"{REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS} seconds"
            ) from exc
        if result.returncode:
            raise OSError(result.stderr.strip() or f"rsync exited {result.returncode}")
        for path in cached.values():
            self._remote_source_cache.touch(path)
        return cached


def _normalize_record(raw: dict[str, Any], provider: str, line_number: int) -> ConversationRecord:
    """Model wrapper over the parser shared with the remote drivers."""

    normalized = normalize_record(raw, provider, line_number)
    return ConversationRecord(
        uuid=normalized["uuid"],
        timestamp=_parse_datetime(normalized["timestamp"]),
        role=normalized["role"],
        text=normalized["text"],
        raw_type=normalized["raw_type"],
    )


def _missing_cursor_message(session: ConversationSession, cursor: str | None) -> str:
    return (
        f"Stored cursor {cursor!r} for session {session.key!r} is missing before the "
        f"indexed terminal record {session.last_uuid!r}; reseed this session explicitly "
        "instead of silently rereading it."
    )


def _local_source_identity(path: Path) -> LocalSourceIdentity:
    stat = path.stat()
    return LocalSourceIdentity(
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _local_source_identity_matches(session: ConversationSession) -> bool:
    expected = session.local_source_identity
    if expected is None or session.source_path_is_remote or session.remote_source_host:
        return False
    try:
        return _local_source_identity(Path(session.path)) == expected
    except OSError:
        return False


def _copy_local_original(session: ConversationSession, destination: Path) -> None:
    expected = session.local_source_identity
    if expected is None:
        raise ValueError("provider conversation changed while it was indexed")
    source = Path(session.path)
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor = -1
    try:
        initial = os.fstat(source_descriptor)
        observed = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
        expected_fields = (expected.device, expected.inode, expected.size, expected.mtime_ns)
        if not stat.S_ISREG(initial.st_mode) or observed != expected_fields:
            raise ValueError("provider conversation changed after it was indexed")
        destination_descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        while True:
            chunk = os.read(source_descriptor, SOURCE_ORIGINAL_COPY_BUFFER_BYTES)
            if not chunk:
                break
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise OSError("short provider conversation snapshot write")
                remaining = remaining[written:]
        final = os.fstat(source_descriptor)
        current = source.lstat()
        if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != expected_fields or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) != expected_fields:
            raise ValueError("provider conversation changed during snapshot")
        os.fchmod(destination_descriptor, 0o400)
        os.fsync(destination_descriptor)
    except BaseException:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
            destination_descriptor = -1
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_temporary_slice(path: Path) -> None:
    if not path.exists():
        return
    path.chmod(0o700)
    for child in path.iterdir():
        child.chmod(0o600)
    shutil.rmtree(path)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _record_parsing_source() -> str:
    """Load the parser as a package resource for source, wheel, and frozen builds."""
    return (
        importlib.resources.files("rcp.sources")
        .joinpath("record_parsing.py")
        .read_text(encoding="utf-8")
    )


def _remote_program(driver: str) -> str:
    """Prepend the shared parser's source to a remote driver."""

    return f"{_record_parsing_source()}\n{driver}"


_REMOTE_INDEX_DRIVER = r"""
import json
import sys
from pathlib import Path

request = json.loads(sys.argv[1])
provider = request["provider"]
roots = request["repository_paths"]
unmatched_files = 0
malformed_files = 0

for path in Path(request["root"]).expanduser().rglob("*.jsonl"):
    if provider == "claude" and "subagents" in path.parts:
        continue
    cwd = ""
    session_id = ""
    first_timestamp = None
    last_timestamp = None
    last_uuid = None
    record_count = 0
    thread_source = "user" if provider == "claude" else None
    parent_session_id = None
    originator = None
    source_kind = provider if provider == "claude" else None
    relevant = False
    sequence_count = 0
    sequence_first_timestamp = None
    sequence_last_timestamp = None
    sequence_last_uuid = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                sequence_count += 1
                record = normalize_record(raw, provider, line_number)
                timestamp = record["timestamp"]
                if timestamp:
                    sequence_first_timestamp = sequence_first_timestamp or timestamp
                    sequence_last_timestamp = timestamp
                sequence_last_uuid = record["uuid"]
                if provider == "codex" and raw.get("type") == "session_meta":
                    inner = raw.get("payload", {})
                    cwd = inner.get("cwd", cwd)
                    session_id = inner.get("id") or inner.get("session_id") or session_id
                    thread_source = inner.get("thread_source") or thread_source
                    originator = inner.get("originator") or originator
                    source = inner.get("source")
                    if isinstance(source, str):
                        source_kind = source
                    elif isinstance(source, dict):
                        source_kind = "subagent" if "subagent" in source else source_kind
                        subagent = source.get("subagent")
                        spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
                        if isinstance(spawn, dict):
                            parent_session_id = (
                                spawn.get("parent_thread_id") or parent_session_id
                            )
                else:
                    cwd = raw.get("cwd", cwd)
                    session_id = raw.get("sessionId", session_id)
                if cwd:
                    relevant = path_matches_roots(cwd, roots)
                    if not relevant:
                        break
                if not relevant:
                    if line_number >= 50:
                        break
                    continue
                # The slice reader counts from the first nonblank record, including
                # provider metadata that precedes the first cwd. Commit that whole
                # prefix and its normalized metadata only after this file is known
                # to belong to the repository.
                record_count = sequence_count
                first_timestamp = sequence_first_timestamp
                last_timestamp = sequence_last_timestamp
                last_uuid = sequence_last_uuid
        if relevant and cwd:
            print(json.dumps({
                "kind": "session",
                "path": str(path),
                "cwd": cwd,
                "session_id": session_id or path.stem,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "last_uuid": last_uuid,
                "record_count": record_count,
                "thread_source": thread_source,
                "parent_session_id": parent_session_id,
                "originator": originator,
                "source_kind": source_kind,
            }))
        elif cwd:
            unmatched_files += 1
        else:
            malformed_files += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        malformed_files += 1

print(json.dumps({
    "kind": "summary",
    "unmatched_files": unmatched_files,
    "malformed_files": malformed_files,
}))
"""


_REMOTE_SLICE_DRIVER = r"""
import json
import sys
from pathlib import Path

request = json.loads(sys.argv[1])
path = Path(request["path"])
provider = request["provider"]
expected_count = int(request["record_count"])
expected_terminal = request.get("last_uuid")
requested_cursor = request.get("from_uuid")
session_key = request["session_key"]


def fail(message):
    print(json.dumps({"error": message}), file=sys.stderr)
    raise SystemExit(1)


if expected_count == 0:
    if expected_terminal is not None:
        fail(f"Session {session_key!r} has an inconsistent empty source index.")
    if requested_cursor is not None:
        fail(
            f"Stored cursor {requested_cursor!r} for session {session_key!r} is missing "
            "from its empty indexed source; reseed this session explicitly."
        )
    print("{}", file=sys.stderr)
    raise SystemExit(0)
if expected_terminal is None:
    fail(f"Session {session_key!r} has no indexed terminal record.")

indexed = 0
cursor_seen = requested_cursor is None
digest_matches = []
cursor_digest = (
    requested_cursor.rpartition("-")[2]
    if isinstance(requested_cursor, str) and requested_cursor.startswith("line-")
    else None
)
try:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            indexed += 1
            if indexed > expected_count:
                break
            record = normalize_record(json.loads(line), provider, line_number)
            record_id = record["uuid"]
            if record_id == requested_cursor:
                cursor_seen = True
            if (
                cursor_digest
                and record_id.startswith("line-")
                and record_id.rpartition("-")[2] == cursor_digest
            ):
                digest_matches.append(record_id)
            if indexed == expected_count:
                if record_id != expected_terminal:
                    fail(
                        f"Conversation source for {session_key!r} changed before its indexed "
                        "terminal record; rebuild the source index and retry."
                    )
                break
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    fail(f"Could not read remote conversation source for {session_key!r}: {exc}")

if indexed < expected_count:
    fail(
        f"Conversation source for {session_key!r} ended before its indexed terminal "
        "record; rebuild the source index and retry."
    )

resolved_cursor = requested_cursor
cursor_repair = None
if not cursor_seen:
    if cursor_digest and len(digest_matches) == 1:
        resolved_cursor = digest_matches[0]
        cursor_repair = resolved_cursor
    else:
        fail(
            f"Stored cursor {requested_cursor!r} for session {session_key!r} is missing before "
            f"the indexed terminal record {expected_terminal!r}; reseed this session explicitly "
            "instead of silently rereading it."
        )

seen = resolved_cursor is None
indexed = 0
try:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            indexed += 1
            if indexed > expected_count:
                break
            record = normalize_record(json.loads(line), provider, line_number)
            if not seen:
                if record["uuid"] == resolved_cursor:
                    seen = True
            else:
                print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            if indexed == expected_count:
                break
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    fail(f"Could not materialize remote conversation slice for {session_key!r}: {exc}")

print(json.dumps({"cursor_repair": cursor_repair}), file=sys.stderr)
"""


_REMOTE_INDEX_SCRIPT = _remote_program(_REMOTE_INDEX_DRIVER)
_REMOTE_SLICE_SCRIPT = _remote_program(_REMOTE_SLICE_DRIVER)
