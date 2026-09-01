from __future__ import annotations

import codecs
import hashlib
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from rcp.config import Manifest
from rcp.providers import ProviderId, legacy_runtime_id, require_runtime_id, runtime_label
from rcp.storage import AppStore
from rcp.transport import LocalStateWorkspace, StateUnavailable, StateWorkspace

INTRODUCTION_TEMPLATE = """# Introduction

## What question we study

## What adjacent questions there are

## Literature review

## High-level methods

## Main results

## Why this deserves publication and communication to the community
"""


def canonical_introduction_backup_source(root: Path) -> Path | None:
    """Return the only canonical Paper file without refreshing or locking it."""

    paper_root = root / "paper"
    try:
        directory = paper_root.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("The canonical Paper directory is unavailable.") from exc
    if not stat.S_ISDIR(directory.st_mode):
        raise ValueError("The canonical Paper path is not a safe directory.")
    try:
        entries = sorted(paper_root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ValueError("The canonical Paper directory cannot be enumerated.") from exc
    if not entries:
        return None
    introduction = paper_root / "introduction.md"
    if entries != [introduction]:
        raise ValueError("The canonical Paper directory contains an unclassified entry.")
    try:
        metadata = introduction.lstat()
    except OSError as exc:
        raise ValueError("The canonical Paper introduction cannot be inspected.") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("The canonical Paper introduction is not a safe regular file.")
    return introduction


def validate_canonical_introduction_backup(path: Path) -> None:
    """Validate one copied introduction as UTF-8 Markdown without loading it whole."""

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("The copied Paper introduction is not a regular file.")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            if "\x00" in decoder.decode(chunk):
                raise ValueError("The copied Paper introduction contains NUL text.")
        if "\x00" in decoder.decode(b"", final=True):
            raise ValueError("The copied Paper introduction contains NUL text.")
    except UnicodeDecodeError as exc:
        raise ValueError("The copied Paper introduction is not UTF-8 Markdown.") from exc
    finally:
        os.close(descriptor)


class PaperSnapshot(BaseModel):
    content: str
    sync_state: Literal["not_created", "synced", "unsynced", "behind"]
    base_hash: str | None = None
    canonical_hash: str | None = None
    incoming_content: str | None = None
    updated_at: datetime | None = None
    canonical_available: bool


class WritingSession(BaseModel):
    provider: ProviderId
    runtime_id: str = ""
    #: The human-facing name for `runtime_id`, filled from the provider registry.
    runtime_label: str = ""
    native_session_id: str
    execution_machine: str
    project_id: str
    title: str | None = None
    model: str
    reasoning: str | None = None
    created_at: datetime
    last_resumed_at: datetime
    introduction_hash_examined: str
    graph_revision_examined: int
    research_md_hash_examined: str

    @model_validator(mode="after")
    def validate_runtime(self) -> WritingSession:
        if not self.runtime_id:
            self.runtime_id = legacy_runtime_id(self.provider)
        require_runtime_id(self.provider, self.runtime_id)
        self.runtime_label = runtime_label(self.provider, self.runtime_id)
        return self


class PaperService:
    def __init__(
        self,
        manifest: Manifest,
        store: AppStore,
        workspace: StateWorkspace | None = None,
        *,
        project_id: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.store = store
        self.workspace = workspace or LocalStateWorkspace(
            manifest.research_dir, str(manifest.research_dir)
        )
        self.project_id = project_id or manifest.name
        self.canonical_path = self.workspace.root / "paper" / "introduction.md"

    def snapshot(self) -> PaperSnapshot:
        draft = self._draft()
        canonical_content, canonical_available = self._read_canonical()
        if draft is None and canonical_content is None:
            return PaperSnapshot(
                content="",
                sync_state="not_created",
                canonical_available=canonical_available,
            )
        if draft is None:
            canonical_hash = _hash(canonical_content)
            return PaperSnapshot(
                content=canonical_content or "",
                sync_state="synced" if canonical_available else "unsynced",
                base_hash=canonical_hash,
                canonical_hash=canonical_hash,
                canonical_available=canonical_available,
            )
        return self._draft_snapshot(draft, canonical_content, canonical_available)

    def create(self) -> PaperSnapshot:
        draft = self._draft()
        if draft is None:
            canonical_content = self._read_cached_canonical()
            content = canonical_content if canonical_content is not None else INTRODUCTION_TEMPLATE
            base_hash = _hash(canonical_content) if canonical_content is not None else None
            self._save_draft(content, base_hash, canonical_content)
            draft = self._draft()
        return self._local_draft_snapshot(draft)

    def save(self, content: str, base_hash: str | None) -> PaperSnapshot:
        draft = self._draft()
        stored_content = draft["content"] if draft is not None else None
        stored_base_hash = draft["base_hash"] if draft is not None else base_hash
        ancestor_content = draft["ancestor_content"] if draft is not None else None
        self._save_draft(content, stored_base_hash, ancestor_content)
        try:
            with self.workspace.transaction():
                canonical_content = self._read_cached_canonical()
                canonical_hash = _hash(canonical_content) if canonical_content is not None else None
                if canonical_hash == _hash(content):
                    self._save_draft(content, canonical_hash, canonical_content)
                    return self.snapshot()
                if canonical_hash != base_hash:
                    return self.snapshot()
                if (
                    draft is not None
                    and stored_base_hash != base_hash
                    and stored_content == content
                ):
                    return self.snapshot()
                self._write_canonical(content)
                new_hash = _hash(content)
                self._save_draft(content, new_hash, content)
                return self.snapshot()
        except StateUnavailable:
            return self.snapshot()

    def restore_canonical(
        self,
        source: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        """Validate and publish the archived introduction without changing its bytes."""

        validate_canonical_introduction_backup(source)
        self.workspace.restore_exact_file(
            Path("paper/introduction.md"),
            source,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )

    def sessions(self) -> list[WritingSession]:
        with self.store.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM writing_sessions WHERE project_id = ? ORDER BY last_resumed_at DESC",
                (self.project_id,),
            ).fetchall()
        return [WritingSession.model_validate(dict(row)) for row in rows]

    def record_session(self, session: WritingSession) -> None:
        data = session.model_dump(mode="json")
        with self.store.connection() as connection:
            connection.execute(
                """
                INSERT INTO writing_sessions (
                    native_session_id, provider, runtime_id, execution_machine, project_id,
                    title, model, reasoning, created_at, last_resumed_at,
                    introduction_hash_examined, graph_revision_examined,
                    research_md_hash_examined
                ) VALUES (
                    :native_session_id, :provider, :runtime_id, :execution_machine, :project_id,
                    :title, :model, :reasoning, :created_at, :last_resumed_at,
                    :introduction_hash_examined, :graph_revision_examined,
                    :research_md_hash_examined
                )
                ON CONFLICT(native_session_id) DO UPDATE SET
                    runtime_id = excluded.runtime_id,
                    title = excluded.title,
                    last_resumed_at = excluded.last_resumed_at,
                    introduction_hash_examined = excluded.introduction_hash_examined,
                    graph_revision_examined = excluded.graph_revision_examined,
                    research_md_hash_examined = excluded.research_md_hash_examined
                """,
                data,
            )

    def _draft(self):
        with self.store.connection() as connection:
            return connection.execute(
                "SELECT * FROM paper_drafts WHERE project_id = ?", (self.project_id,)
            ).fetchone()

    def _save_draft(
        self,
        content: str,
        base_hash: str | None,
        ancestor_content: str | None,
    ) -> None:
        with self.store.connection() as connection:
            connection.execute(
                """
                INSERT INTO paper_drafts(
                    project_id, content, base_hash, ancestor_content, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    content = excluded.content,
                    base_hash = excluded.base_hash,
                    ancestor_content = excluded.ancestor_content,
                    updated_at = excluded.updated_at
                """,
                (self.project_id, content, base_hash, ancestor_content, self.store.now()),
            )

    def _local_draft_snapshot(self, draft) -> PaperSnapshot:
        canonical_content = self._read_cached_canonical()
        return self._draft_snapshot(draft, canonical_content, self.workspace.reachable)

    def _draft_snapshot(
        self,
        draft,
        canonical_content: str | None,
        canonical_available: bool,
    ) -> PaperSnapshot:
        content = draft["content"]
        stored_base_hash = draft["base_hash"]
        canonical_hash = _hash(canonical_content) if canonical_content is not None else None
        incoming_content = None
        if not canonical_available:
            state = "unsynced"
            base_hash = stored_base_hash
        elif canonical_hash == _hash(content):
            state = "synced"
            base_hash = canonical_hash
        elif canonical_hash == stored_base_hash:
            state = "unsynced"
            base_hash = stored_base_hash
        else:
            state = "behind"
            base_hash = stored_base_hash
            incoming_content = canonical_content
        return PaperSnapshot(
            content=content,
            sync_state=state,
            base_hash=base_hash,
            canonical_hash=canonical_hash,
            incoming_content=incoming_content,
            updated_at=datetime.fromisoformat(draft["updated_at"]),
            canonical_available=canonical_available,
        )

    def _read_canonical(self) -> tuple[str | None, bool]:
        available = True
        try:
            self.workspace.refresh_if_stale()
        except StateUnavailable:
            available = False
        return self._read_cached_canonical(), available

    def _read_cached_canonical(self) -> str | None:
        try:
            if not self.canonical_path.exists():
                return None
            return self.canonical_path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_canonical(self, content: str) -> None:
        self.canonical_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.canonical_path.with_name(f".{self.canonical_path.name}.{os.getpid()}.tmp")
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, self.canonical_path)
        self.workspace.publish([Path("paper/introduction.md")])


def _hash(content: str | None) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()
