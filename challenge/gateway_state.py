"""Durable, isolated browser-session state for the challenge gateway."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import sqlite3
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

COOKIE_NAME = "rcp_demo_session"
SESSION_TTL = timedelta(days=7)
_COOKIE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")


class GatewaySessionError(RuntimeError):
    """Base class for a session-state refusal safe to surface at the gateway."""


class GatewayCapacityError(GatewaySessionError):
    """A new isolated copy would exceed the configured session or disk cap."""


class GatewaySessionStorageError(GatewayCapacityError):
    """One existing copy exceeded its per-session storage allowance."""


class GatewayRateLimitError(GatewaySessionError):
    """One client created too many fresh copies in the configured window."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Too many fresh demo sessions. Please retry later.")
        self.retry_after_seconds = max(1, retry_after_seconds)


class GatewaySessionUnavailableError(GatewaySessionError):
    """A mapped copy is missing or otherwise cannot be resumed safely."""


@dataclass(frozen=True)
class GatewayLimits:
    """Initial guardrails; W12 replaces them with measured deployment values."""

    max_sessions: int = 500
    max_total_bytes: int = 32 * 1024 * 1024 * 1024
    max_session_bytes: int = 256 * 1024 * 1024
    max_creations_per_client: int = 20
    creation_window: timedelta = timedelta(hours=1)


@dataclass(frozen=True)
class DemoSession:
    session_id: str
    root: Path
    project_root: Path
    data_root: Path
    stage_root: Path
    created_at: datetime
    expires_at: datetime
    size_bytes: int


@dataclass(frozen=True)
class SessionBinding:
    session: DemoSession
    cookie_value: str
    created: bool


class DemoSessionRegistry:
    """Map opaque cookies to private fixture copies without path-derived input."""

    def __init__(
        self,
        root: Path,
        fixture_root: Path,
        *,
        limits: GatewayLimits | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.fixture_root = fixture_root.resolve()
        self.sessions_root = self.root / "sessions"
        self.database_path = self.root / "gateway.sqlite3"
        self.limits = limits or GatewayLimits()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._validate_configuration()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def get_or_create(
        self,
        cookie_value: str | None,
        *,
        client_key: str,
        protected_session_ids: Collection[str] = (),
        allow_oversized: bool = False,
    ) -> SessionBinding:
        """Resume one copy, or mint a new opaque cookie and isolated copy."""

        now = self._now()
        self.delete_expired(now=now, protected_session_ids=protected_session_ids)
        if cookie_value is not None and _valid_cookie(cookie_value):
            existing = self._resume(cookie_value, now=now)
            if existing is not None:
                if existing.size_bytes > self.limits.max_session_bytes and not allow_oversized:
                    raise GatewaySessionStorageError(
                        "This demo session exceeded its storage allowance. Start over to continue."
                    )
                return SessionBinding(existing, cookie_value, created=False)
        return self._create(client_key=client_key, now=now)

    def resolve(
        self,
        cookie_value: str,
        *,
        protected_session_ids: Collection[str] = (),
    ) -> DemoSession | None:
        """Resolve and renew one valid cookie without silently replacing its copy."""

        if not _valid_cookie(cookie_value):
            return None
        now = self._now()
        self.delete_expired(now=now, protected_session_ids=protected_session_ids)
        return self._resume(cookie_value, now=now)

    def discard_fresh(self, binding: SessionBinding) -> None:
        """Roll back one just-created copy that never reached a browser."""

        if not binding.created or not _valid_cookie(binding.cookie_value):
            raise GatewaySessionUnavailableError("Only a fresh demo session can be discarded.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            deleted = connection.execute(
                "DELETE FROM sessions WHERE token_hash = ? AND session_id = ?",
                (_token_hash(binding.cookie_value), binding.session.session_id),
            )
        if deleted.rowcount != 1:
            raise GatewaySessionUnavailableError("The fresh demo session is no longer registered.")
        _remove_exact_tree(binding.session.root, parent=self.sessions_root)

    def rotate(self, cookie_value: str, *, client_key: str) -> SessionBinding:
        """Replace exactly one current copy after its process has been stopped."""

        if not _valid_cookie(cookie_value):
            raise GatewaySessionUnavailableError("The current demo session is unavailable.")
        now = self._now()
        old_row = self._row_for_cookie(cookie_value)
        if old_row is None:
            raise GatewaySessionUnavailableError("The current demo session is unavailable.")
        old_session = self._session_from_row(old_row)
        self._require_session_directory(old_session)
        new_cookie = _new_cookie()
        new_session_id = str(uuid4())
        new_size = self._clone_fixture(new_session_id)
        new_root = self._session_root(new_session_id)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    "SELECT session_id, size_bytes FROM sessions WHERE token_hash = ?",
                    (_token_hash(cookie_value),),
                ).fetchone()
                if current is None or current[0] != old_session.session_id:
                    raise GatewaySessionUnavailableError(
                        "The current demo session changed before Start over completed."
                    )
                self._admit_creation(
                    connection,
                    client_key=client_key,
                    now=now,
                    new_size=new_size,
                    replacing_size=int(current[1]),
                )
                created_at = _timestamp(now)
                expires_at = _timestamp(now + SESSION_TTL)
                connection.execute(
                    """
                    INSERT INTO sessions(
                        token_hash, session_id, created_at, last_seen_at, expires_at, size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _token_hash(new_cookie),
                        new_session_id,
                        created_at,
                        created_at,
                        expires_at,
                        new_size,
                    ),
                )
                connection.execute(
                    "DELETE FROM sessions WHERE token_hash = ?",
                    (_token_hash(cookie_value),),
                )
                self._record_creation(connection, client_key=client_key, now=now)
        except Exception:
            _remove_exact_tree(new_root, parent=self.sessions_root)
            raise
        _remove_exact_tree(old_session.root, parent=self.sessions_root)
        session = self._session_from_values(
            session_id=new_session_id,
            created_at=now,
            expires_at=now + SESSION_TTL,
            size_bytes=new_size,
        )
        return SessionBinding(session, new_cookie, created=True)

    def delete_expired(
        self,
        *,
        now: datetime | None = None,
        protected_session_ids: Collection[str] = (),
    ) -> list[str]:
        """Delete expired mappings and only their UUID-named private directories."""

        cutoff = now or self._now()
        protected = {_validated_session_id(value) for value in protected_session_ids}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT token_hash, session_id FROM sessions WHERE expires_at <= ?",
                (_timestamp(cutoff),),
            ).fetchall()
            expired = [row for row in rows if row[1] not in protected]
            connection.executemany(
                "DELETE FROM sessions WHERE token_hash = ?",
                ((row[0],) for row in expired),
            )
        deleted: list[str] = []
        for _token, session_id in expired:
            validated = _validated_session_id(session_id)
            _remove_exact_tree(self._session_root(validated), parent=self.sessions_root)
            deleted.append(validated)
        return deleted

    def refresh_usage(self, session_id: str) -> int:
        """Measure one copy and persist its current contribution to future admission."""

        validated = _validated_session_id(session_id)
        root = self._session_root(validated)
        size = _tree_size(root)
        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE sessions SET size_bytes = ? WHERE session_id = ?",
                (size, validated),
            )
        if updated.rowcount != 1:
            raise GatewaySessionUnavailableError("The demo session is no longer registered.")
        if size > self.limits.max_session_bytes:
            raise GatewaySessionStorageError(
                "This demo session exceeded its storage allowance. Start over to continue."
            )
        return size

    def _create(self, *, client_key: str, now: datetime) -> SessionBinding:
        cookie = _new_cookie()
        session_id = str(uuid4())
        size = self._clone_fixture(session_id)
        root = self._session_root(session_id)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._admit_creation(
                    connection,
                    client_key=client_key,
                    now=now,
                    new_size=size,
                )
                created_at = _timestamp(now)
                expires_at = _timestamp(now + SESSION_TTL)
                connection.execute(
                    """
                    INSERT INTO sessions(
                        token_hash, session_id, created_at, last_seen_at, expires_at, size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _token_hash(cookie),
                        session_id,
                        created_at,
                        created_at,
                        expires_at,
                        size,
                    ),
                )
                self._record_creation(connection, client_key=client_key, now=now)
        except Exception:
            _remove_exact_tree(root, parent=self.sessions_root)
            raise
        session = self._session_from_values(
            session_id=session_id,
            created_at=now,
            expires_at=now + SESSION_TTL,
            size_bytes=size,
        )
        return SessionBinding(session, cookie, created=True)

    def _resume(self, cookie_value: str, *, now: datetime) -> DemoSession | None:
        row = self._row_for_cookie(cookie_value)
        if row is None:
            return None
        session = self._session_from_row(row)
        self._require_session_directory(session)
        expires_at = now + SESSION_TTL
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET last_seen_at = ?, expires_at = ? WHERE token_hash = ?",
                (_timestamp(now), _timestamp(expires_at), _token_hash(cookie_value)),
            )
        return self._session_from_values(
            session_id=session.session_id,
            created_at=session.created_at,
            expires_at=expires_at,
            size_bytes=session.size_bytes,
        )

    def _clone_fixture(self, session_id: str) -> int:
        _validated_session_id(session_id)
        if not self.fixture_root.is_dir():
            raise GatewaySessionUnavailableError("The clean demo fixture is unavailable.")
        _reject_symlinks(self.fixture_root)
        temporary = self.sessions_root / f".creating-{session_id}"
        final = self._session_root(session_id)
        if temporary.exists() or final.exists():
            raise GatewaySessionUnavailableError("A demo session directory already exists.")
        temporary.mkdir()
        try:
            shutil.copytree(self.fixture_root, temporary / "project")
            (temporary / "data").mkdir()
            (temporary / "stage").mkdir()
            size = _tree_size(temporary)
            if size > self.limits.max_session_bytes:
                raise GatewayCapacityError("The clean demo fixture exceeds the session cap.")
            os.replace(temporary, final)
            return size
        except Exception:
            _remove_exact_tree(temporary, parent=self.sessions_root)
            raise

    def _admit_creation(
        self,
        connection: sqlite3.Connection,
        *,
        client_key: str,
        now: datetime,
        new_size: int,
        replacing_size: int = 0,
    ) -> None:
        window_start = now - self.limits.creation_window
        connection.execute(
            "DELETE FROM creation_events WHERE created_at <= ?",
            (_timestamp(window_start),),
        )
        client_hash = _client_hash(client_key)
        rows = connection.execute(
            "SELECT created_at FROM creation_events WHERE client_hash = ? ORDER BY created_at",
            (client_hash,),
        ).fetchall()
        if len(rows) >= self.limits.max_creations_per_client:
            oldest = _parse_timestamp(rows[0][0])
            retry_at = oldest + self.limits.creation_window
            raise GatewayRateLimitError(int((retry_at - now).total_seconds()) + 1)
        count, total = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM sessions"
        ).fetchone()
        effective_count = int(count) - (1 if replacing_size else 0)
        effective_total = int(total) - replacing_size
        if effective_count >= self.limits.max_sessions:
            raise GatewayCapacityError(
                "The demo is at capacity. Existing sessions are preserved; please retry later."
            )
        if effective_total + new_size > self.limits.max_total_bytes:
            raise GatewayCapacityError(
                "The demo is at storage capacity. Existing sessions are preserved; please retry later."
            )

    def _record_creation(
        self,
        connection: sqlite3.Connection,
        *,
        client_key: str,
        now: datetime,
    ) -> None:
        connection.execute(
            "INSERT INTO creation_events(client_hash, created_at) VALUES (?, ?)",
            (_client_hash(client_key), _timestamp(now)),
        )

    def _row_for_cookie(self, cookie_value: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT session_id, created_at, expires_at, size_bytes
                FROM sessions WHERE token_hash = ?
                """,
                (_token_hash(cookie_value),),
            ).fetchone()

    def _session_from_row(self, row: sqlite3.Row) -> DemoSession:
        return self._session_from_values(
            session_id=row[0],
            created_at=_parse_timestamp(row[1]),
            expires_at=_parse_timestamp(row[2]),
            size_bytes=int(row[3]),
        )

    def _session_from_values(
        self,
        *,
        session_id: str,
        created_at: datetime,
        expires_at: datetime,
        size_bytes: int,
    ) -> DemoSession:
        validated = _validated_session_id(session_id)
        root = self._session_root(validated)
        return DemoSession(
            session_id=validated,
            root=root,
            project_root=root / "project",
            data_root=root / "data",
            stage_root=root / "stage",
            created_at=created_at,
            expires_at=expires_at,
            size_bytes=size_bytes,
        )

    def _require_session_directory(self, session: DemoSession) -> None:
        if not session.root.is_dir() or session.root.is_symlink():
            raise GatewaySessionUnavailableError(
                "The saved demo copy is missing; use Start over to create a new one."
            )
        for required in (session.project_root, session.data_root, session.stage_root):
            if not required.is_dir() or required.is_symlink():
                raise GatewaySessionUnavailableError(
                    "The saved demo copy is incomplete; use Start over to create a new one."
                )

    def _session_root(self, session_id: str) -> Path:
        return self.sessions_root / _validated_session_id(session_id)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0)
                );
                CREATE INDEX IF NOT EXISTS sessions_expires_at
                    ON sessions(expires_at);
                CREATE TABLE IF NOT EXISTS creation_events (
                    client_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS creation_events_client_created
                    ON creation_events(client_hash, created_at);
                """
            )

    def _validate_configuration(self) -> None:
        if self.root == self.fixture_root or self.root in self.fixture_root.parents:
            raise ValueError("The gateway root cannot contain the clean fixture.")
        if self.fixture_root in self.root.parents:
            raise ValueError("The clean fixture cannot contain the gateway root.")
        if (
            self.limits.max_sessions < 1
            or self.limits.max_total_bytes < 1
            or self.limits.max_session_bytes < 1
            or self.limits.max_creations_per_client < 1
            or self.limits.creation_window <= timedelta(0)
        ):
            raise ValueError("Gateway limits must be positive.")

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("The gateway clock must return a timezone-aware datetime.")
        return value.astimezone(UTC)


def _new_cookie() -> str:
    value = secrets.token_urlsafe(32)
    if not _valid_cookie(value):  # pragma: no cover - token_urlsafe contract guard
        raise RuntimeError("Failed to generate a valid opaque demo-session cookie.")
    return value


def _valid_cookie(value: str) -> bool:
    return bool(_COOKIE_PATTERN.fullmatch(value))


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _client_hash(value: str) -> str:
    if not value:
        raise ValueError("A non-empty client rate-limit key is required.")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validated_session_id(value: str) -> str:
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise GatewaySessionUnavailableError("Invalid stored demo-session identity.") from exc
    canonical = str(parsed)
    if value != canonical:
        raise GatewaySessionUnavailableError("Invalid stored demo-session identity.")
    return canonical


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise GatewaySessionUnavailableError("Invalid stored demo-session timestamp.")
    return parsed.astimezone(UTC)


def _tree_size(root: Path) -> int:
    if not root.is_dir() or root.is_symlink():
        raise GatewaySessionUnavailableError("The demo-session directory is unavailable.")
    size = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise GatewaySessionUnavailableError("Demo-session trees cannot contain symlinks.")
        if path.is_file():
            try:
                size += path.stat().st_size
            except FileNotFoundError:
                # SQLite WAL and shared-memory files may disappear after rglob()
                # observes them and before their size is read.
                continue
    return size


def _reject_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise GatewaySessionUnavailableError("The clean demo fixture cannot contain symlinks.")


def _remove_exact_tree(path: Path, *, parent: Path) -> None:
    resolved_parent = parent.resolve()
    if path.parent.resolve() != resolved_parent:
        raise GatewaySessionUnavailableError("Refused to delete outside the session directory.")
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
