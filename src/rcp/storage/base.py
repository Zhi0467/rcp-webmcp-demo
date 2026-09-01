from __future__ import annotations

import json
import os
import sqlite3
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rcp.limits import BACKUP_SQLITE_BUSY_SLEEP_SECONDS, BACKUP_SQLITE_PAGES_PER_STEP
from rcp.providers import PROVIDER_IDS, legacy_runtime_id
from rcp.storage.auto_research import migrate_legacy_auto_research
from rcp.storage.episodes import migrate_legacy_episodes
from rcp.storage.models import (  # noqa: F401
    _EXPERIMENT_EPISODE_CONTEXT_CANDIDATE_ROLE,
    _EXPERIMENT_EPISODE_PINNED_FIELDS,
    _MISSING_EXPERIMENT_EPISODE_CONTEXT_DIAGNOSTIC,
    _PROJECT_ID_TABLES,
    ACTIVE_AGENT_TASK_STATUSES,
    SPACE_NAME_MAX_LENGTH,
    AgentCommandInvocationRecord,
    AgentTaskContractRecord,
    AgentTaskEventRecord,
    AgentTaskKind,
    AgentTaskReceiptRecord,
    AgentTaskReceiptTier,
    AgentTaskRecord,
    AgentTaskStatus,
    AgentUsageCell,
    AgentUsageCountReason,
    AgentUsageMetric,
    AgentUsageRecord,
    AgentUsageSnapshot,
    ChatSessionContextRecord,
    ExperimentEpisodeRecord,
    ExperimentLoopRuntime,
    ExperimentWatcherResourceRecord,
    GraphCondition,
    GraphWatcherRecord,
    NodeStatusGraphCondition,
    ProjectRecord,
    ProjectStageRecord,
    ProjectTransferActivationReceipt,
    ProjectTransferImportRecord,
    ProjectTransferRestoreReentryReceipt,
    ProposalResolvedGraphCondition,
    ProviderSkillInventoryRecord,
    ResultViewConflict,
    ResultViewRecord,
    SpaceKind,
    SpaceUserKind,
    SpaceUserRecord,
    StoredWatcherRecord,
    TeamAuthenticationError,
    TeamInvitationRecord,
    WatcherClaimConflict,
    WatcherContinuation,
    WatcherDeliveryRecord,
    WatcherRecord,
    WatcherStatus,
    WatcherStopRequest,
    _canonical_space_id,
    _canonical_uuid4,
    _discard_failed_team_initialization,
    _experiment_pinned_value,
    _new_enrollment_code,
    _new_member_token,
    _new_session_token,
    _optional_str,
    _parse_enrollment_code,
    _plain_html_name,
    _required_timestamp,
    _result_view_html_bytes,
    _result_view_is_visible,
    _result_view_reference_time,
    _sha256,
    _stored_space_kind,
    _validated_result_view_html,
    normalize_space_name,
    watcher_next_check_at,
)

if TYPE_CHECKING:
    from rcp.storage import AppStore


class AppStoreBase:
    """Connection ownership, schema initialization, and the clock every mixin shares."""

    def __init__(self, path: Path, *, space_kind: SpaceKind | None = None) -> None:
        if space_kind is not None and space_kind not in ("personal", "team"):
            raise ValueError("space kind must be 'personal' or 'team'")
        self.path = path
        self._read_only_snapshot = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize(space_kind)

    @classmethod
    def initialize_team_space(cls, path: Path, name: str) -> tuple[AppStore, str]:
        store = cls.__new__(cls)
        store.path = path
        store._read_only_snapshot = False
        store.path.parent.mkdir(parents=True, exist_ok=True)
        initial_space_id = str(uuid.uuid4())
        try:
            bootstrap_code = store._initialize(
                "team",
                initial_space_id=initial_space_id,
                initial_space_name=normalize_space_name(name),
                issue_bootstrap=True,
                require_new=True,
            )
        except Exception:
            _discard_failed_team_initialization(path, initial_space_id)
            raise
        if bootstrap_code is None:  # pragma: no cover - guarded by issue_bootstrap
            raise RuntimeError("RCP team bootstrap code was not created.")
        return store, bootstrap_code

    @classmethod
    def open_read_only_snapshot(cls, path: Path) -> AppStore:
        """Open one completed SQLite snapshot without migrations or write authority."""

        if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
            raise ValueError("the SQLite snapshot path must be absolute and normalized")
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ValueError("the SQLite snapshot is unavailable") from exc
        if not stat.S_ISREG(mode):
            raise ValueError("the SQLite snapshot must be a safe regular file")
        store = cls.__new__(cls)
        store.path = path
        store._read_only_snapshot = True
        try:
            with store.connection() as connection:
                result = [row[0] for row in connection.execute("PRAGMA quick_check").fetchall()]
        except sqlite3.Error as exc:
            raise ValueError("the SQLite snapshot could not be validated") from exc
        if result != ["ok"]:
            raise ValueError("the SQLite snapshot failed its integrity check")
        return store

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        if getattr(self, "_read_only_snapshot", False):
            uri = f"{self.path.resolve(strict=True).as_uri()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, timeout=30.0, uri=True)
        else:
            connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def online_snapshot(self, destination: Path) -> None:
        """Copy this live store with SQLite's online backup API and no app-wide lock."""

        if getattr(self, "_read_only_snapshot", False):
            raise ValueError("a read-only SQLite snapshot cannot create another snapshot")
        if (
            not isinstance(destination, Path)
            or not destination.is_absolute()
            or ".." in destination.parts
        ):
            raise ValueError("the SQLite snapshot destination must be absolute and normalized")
        try:
            parent = destination.parent.lstat()
        except OSError as exc:
            raise ValueError("the SQLite snapshot directory is unavailable") from exc
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
            or stat.S_IMODE(parent.st_mode) & 0o077
        ):
            raise ValueError("the SQLite snapshot directory must be private to this account")

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination, flags, 0o600)
        created = os.fstat(descriptor)
        os.close(descriptor)
        try:
            target = sqlite3.connect(destination, timeout=30.0)
            try:
                with self.connection() as source:
                    source.backup(
                        target,
                        pages=BACKUP_SQLITE_PAGES_PER_STEP,
                        sleep=BACKUP_SQLITE_BUSY_SLEEP_SECONDS,
                    )
                target.commit()
                result = [row[0] for row in target.execute("PRAGMA quick_check").fetchall()]
                if result != ["ok"]:
                    raise RuntimeError("the online SQLite snapshot failed its integrity check")
            finally:
                target.close()
            current = destination.lstat()
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
                created.st_dev,
                created.st_ino,
            ):
                raise RuntimeError("the online SQLite snapshot changed during capture")
            read_descriptor = os.open(
                destination,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fchmod(read_descriptor, 0o400)
                os.fsync(read_descriptor)
            finally:
                os.close(read_descriptor)
        except Exception:
            with suppress(FileNotFoundError):
                current = destination.lstat()
                if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                    destination.unlink()
            raise

    def _initialize(
        self,
        requested_space_kind: SpaceKind | None,
        *,
        initial_space_id: str | None = None,
        initial_space_name: str | None = None,
        issue_bootstrap: bool = False,
        require_new: bool = False,
    ) -> str | None:
        bootstrap_code: str | None = None
        recovering_team_initialization = False
        with self.connection() as connection:
            try:
                connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                # A concurrent first opener may be changing the journal mode.
                # Waiting for a write boundary proves that transaction finished
                # before retrying the same required mode change.
                connection.execute("BEGIN IMMEDIATE")
                connection.rollback()
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            identity_table_exists = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'space_identity'"
                ).fetchone()
                is not None
            )
            if require_new and identity_table_exists:
                identity_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(space_identity)")
                }
                users_table_exists_for_recovery = (
                    connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'space_users'"
                    ).fetchone()
                    is not None
                )
                existing_identity = (
                    connection.execute(
                        "SELECT space_kind, space_name FROM space_identity WHERE singleton = 1"
                    ).fetchone()
                    if {"space_kind", "space_name"}.issubset(identity_columns)
                    else None
                )
                existing_user_count = (
                    connection.execute("SELECT COUNT(*) FROM space_users").fetchone()[0]
                    if users_table_exists_for_recovery
                    else -1
                )
                recovering_team_initialization = bool(
                    issue_bootstrap
                    and initial_space_name is not None
                    and existing_identity is not None
                    and existing_identity["space_kind"] == "team"
                    and existing_identity["space_name"] == initial_space_name
                    and existing_user_count == 0
                )
                if not recovering_team_initialization:
                    raise ValueError("This RCP data directory already contains a space.")
            if not identity_table_exists:
                legacy_database = (
                    connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
                    ).fetchone()
                    is not None
                )
                if require_new and legacy_database:
                    raise ValueError("This RCP data directory already contains RCP data.")
                stored_space_kind = (
                    "personal" if legacy_database else requested_space_kind or "personal"
                )
                if requested_space_kind is not None and requested_space_kind != stored_space_kind:
                    raise ValueError(
                        "An existing RCP database migrates to personal; it cannot be opened "
                        f"as {requested_space_kind}."
                    )
                connection.execute(
                    """
                    CREATE TABLE space_identity (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        space_id TEXT NOT NULL UNIQUE,
                        space_kind TEXT NOT NULL CHECK(space_kind IN ('personal', 'team')),
                        space_name TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO space_identity(singleton, space_id, space_kind, space_name)
                    VALUES (1, ?, ?, ?)
                    """,
                    (initial_space_id or str(uuid.uuid4()), stored_space_kind, initial_space_name),
                )
            else:
                identity_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(space_identity)")
                }
                if "space_id" not in identity_columns:
                    raise RuntimeError("RCP space identity schema is invalid.")
                identity = connection.execute(
                    "SELECT space_id FROM space_identity WHERE singleton = 1"
                ).fetchone()
                if identity is None:
                    raise RuntimeError("RCP space identity is unavailable.")
                _canonical_space_id(identity["space_id"])
                if "space_kind" not in identity_columns:
                    connection.execute(
                        """
                        ALTER TABLE space_identity
                        ADD COLUMN space_kind TEXT CHECK(space_kind IN ('personal', 'team'))
                        """
                    )
                    connection.execute(
                        "UPDATE space_identity SET space_kind = 'personal' WHERE singleton = 1"
                    )
                    stored_space_kind = "personal"
                else:
                    identity = connection.execute(
                        "SELECT space_kind FROM space_identity WHERE singleton = 1"
                    ).fetchone()
                    assert identity is not None
                    stored_space_kind = _stored_space_kind(identity["space_kind"])

                if requested_space_kind is not None and requested_space_kind != stored_space_kind:
                    raise ValueError(
                        f"RCP space is {stored_space_kind}; it cannot be opened as "
                        f"{requested_space_kind}."
                    )

            identity = connection.execute(
                "SELECT space_id, space_kind FROM space_identity WHERE singleton = 1"
            ).fetchone()
            if identity is None:
                raise RuntimeError("RCP space identity is unavailable.")
            _canonical_space_id(identity["space_id"])
            stored_space_kind = _stored_space_kind(identity["space_kind"])

            users_table_exists = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'space_users'"
                ).fetchone()
                is not None
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS space_users (
                    user_id TEXT PRIMARY KEY,
                    identity_kind TEXT NOT NULL
                        CHECK(identity_kind IN ('local_owner', 'team_member')),
                    display_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    removal_started_at TEXT,
                    removed_at TEXT,
                    CHECK(removed_at IS NULL OR removal_started_at IS NOT NULL)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS team_bootstrap_codes (
                    code_id TEXT PRIMARY KEY,
                    code_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT,
                    consumed_by TEXT,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_at TEXT,
                    revoked_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS team_invitations (
                    invitation_id TEXT PRIMARY KEY,
                    code_hash TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT,
                    consumed_by TEXT,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_at TEXT,
                    revoked_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS team_member_tokens (
                    token_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS team_sessions (
                    session_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            # S101: projects registered before membership existed have nothing to
            # seed from, so they are backfilled once — below, where ``projects``
            # exists. Whether this table had to be created *is* the guard, so a
            # later start cannot reapply it.
            members_table_existed = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'project_members'"
                ).fetchone()
                is not None
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS project_members (
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    seated_at TEXT NOT NULL,
                    seated_by TEXT,
                    PRIMARY KEY (project_id, user_id)
                )
                """
            )
            # S122. Deliberately not the space-level `team_invitations` table: a
            # project invitation issues no credential, so it carries no code
            # hash, no expiry, and no failed-attempt lockout. It is an
            # authenticated in-product item addressed to an existing member.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS project_invitations (
                    invitation_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    invited_user_id TEXT NOT NULL,
                    invited_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    response TEXT CHECK(response IN ('accepted', 'declined', 'revoked')),
                    responded_at TEXT
                )
                """
            )
            if not users_table_exists and stored_space_kind == "personal":
                now = self.now()
                owner = SpaceUserRecord(
                    user_id=str(uuid.uuid4()),
                    identity_kind="local_owner",
                    created_at=now,
                    updated_at=now,
                )
                connection.execute(
                    """
                    INSERT INTO space_users (
                        user_id, identity_kind, display_name, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        owner.user_id,
                        owner.identity_kind,
                        owner.display_name,
                        owner.created_at,
                        owner.updated_at,
                    ),
                )
            users = self._space_users_from_connection(connection)
            if stored_space_kind == "personal":
                if len(users) != 1 or users[0].identity_kind != "local_owner":
                    raise RuntimeError("A personal RCP space must contain exactly one local owner.")
            elif any(user.identity_kind == "local_owner" for user in users):
                raise RuntimeError("A team RCP space cannot contain a local owner.")

            # S111 stores may already have the earlier trigger that protected
            # only ``space_id``. Replace it atomically so the additive kind is
            # covered as soon as the migration commits.
            connection.execute("DROP TRIGGER IF EXISTS space_identity_immutable")
            connection.execute(
                """
                CREATE TRIGGER space_identity_immutable
                BEFORE UPDATE OF singleton, space_id, space_kind ON space_identity
                BEGIN
                    SELECT RAISE(ABORT, 'space identity is immutable');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS space_user_identity_immutable
                BEFORE UPDATE OF user_id, identity_kind ON space_users
                BEGIN
                    SELECT RAISE(ABORT, 'space user identity is immutable');
                END
                """
            )
            connection.commit()
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS paper_drafts (
                    project_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    base_hash TEXT,
                    updated_at TEXT NOT NULL,
                    cursor_state TEXT
                );
                CREATE TABLE IF NOT EXISTS writing_sessions (
                    native_session_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    runtime_id TEXT NOT NULL DEFAULT '',
                    execution_machine TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    title TEXT,
                    model TEXT NOT NULL,
                    reasoning TEXT,
                    created_at TEXT NOT NULL,
                    last_resumed_at TEXT NOT NULL,
                    introduction_hash_examined TEXT NOT NULL,
                    graph_revision_examined INTEGER NOT NULL,
                    research_md_hash_examined TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS writing_sessions_project
                    ON writing_sessions(project_id, last_resumed_at DESC);
                CREATE TABLE IF NOT EXISTS chat_session_contexts (
                    provider TEXT NOT NULL,
                    execution_machine TEXT NOT NULL,
                    native_session_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    node_id TEXT,
                    protocol_version INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    snapshot_sha256 TEXT NOT NULL,
                    committed_operation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(provider, execution_machine, native_session_id)
                );
                CREATE INDEX IF NOT EXISTS chat_session_contexts_project
                    ON chat_session_contexts(project_id);
                CREATE INDEX IF NOT EXISTS chat_session_contexts_native_session
                    ON chat_session_contexts(native_session_id);
                CREATE TABLE IF NOT EXISTS result_views (
                    view_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    origin_operation_id TEXT NOT NULL,
                    latest_operation_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    run_on TEXT NOT NULL,
                    native_session_id TEXT NOT NULL,
                    stage_host TEXT NOT NULL,
                    stage_root TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                    html TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    kept_filename TEXT,
                    kept_at TEXT,
                    CHECK((kept_filename IS NULL) = (kept_at IS NULL))
                );
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    home_space_id TEXT,
                    locator TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    state_location TEXT NOT NULL,
                    state_remote INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    last_opened_at TEXT,
                    revision INTEGER,
                    primary_question TEXT,
                    attention_count INTEGER NOT NULL DEFAULT 0,
                    last_refresh_at TEXT,
                    reachable INTEGER,
                    error TEXT,
                    retired_at TEXT,
                    retired_transfer_request_id TEXT
                );
                CREATE INDEX IF NOT EXISTS projects_recent
                    ON projects(last_opened_at DESC, added_at DESC);
                CREATE TABLE IF NOT EXISTS project_provisioning_requests (
                    request_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL
                        CHECK(kind IN ('create_team_project', 'incoming_transfer')),
                    status TEXT NOT NULL CHECK(status IN (
                        'waiting_for_server_setup',
                        'setup_in_progress',
                        'operator_action_needed',
                        'ready_for_review',
                        'completed',
                        'cancelled'
                    )),
                    target_space_id TEXT NOT NULL,
                    authorized_by_json TEXT NOT NULL,
                    proposed_project_id TEXT NOT NULL UNIQUE,
                    project_config_json TEXT,
                    machines_json TEXT NOT NULL,
                    repositories_json TEXT NOT NULL,
                    provider_checks_json TEXT NOT NULL,
                    retryable_diagnostic TEXT,
                    operator_action_json TEXT,
                    final_review_digest TEXT,
                    cancellation_disposition TEXT CHECK(
                        cancellation_disposition IS NULL OR cancellation_disposition IN (
                            'nothing_to_remove',
                            'request_owned_state_removed',
                            'prepared_state_preserved',
                            'operator_cleanup_confirmed'
                        )
                    ),
                    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    setup_started_at TEXT,
                    ready_at TEXT,
                    completed_at TEXT,
                    cancelled_at TEXT
                );
                CREATE INDEX IF NOT EXISTS project_provisioning_status
                    ON project_provisioning_requests(status, updated_at DESC, request_id);
                CREATE TABLE IF NOT EXISTS project_provisioning_step_receipts (
                    request_id TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    from_status TEXT NOT NULL,
                    to_status TEXT NOT NULL,
                    transition_sha256 TEXT NOT NULL,
                    resulting_revision INTEGER NOT NULL CHECK(resulting_revision >= 1),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(request_id, receipt_id),
                    FOREIGN KEY(request_id) REFERENCES project_provisioning_requests(request_id)
                );
                CREATE INDEX IF NOT EXISTS project_provisioning_receipts_revision
                    ON project_provisioning_step_receipts(request_id, resulting_revision);
                CREATE TABLE IF NOT EXISTS project_transfer_requests (
                    request_id TEXT PRIMARY KEY,
                    side TEXT NOT NULL CHECK(side IN ('source', 'target')),
                    phase TEXT NOT NULL CHECK(phase IN (
                        'awaiting_link',
                        'linked',
                        'target_admitted',
                        'source_released',
                        'source_fenced',
                        'archive_bound',
                        'target_activated',
                        'cleanup_acknowledged',
                        'completed',
                        'operator_action_needed'
                    )),
                    project_id TEXT NOT NULL,
                    source_space_id TEXT NOT NULL,
                    target_space_id TEXT NOT NULL,
                    linked_request_id TEXT,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(side, project_id)
                );
                CREATE INDEX IF NOT EXISTS project_transfer_phase
                    ON project_transfer_requests(side, phase, updated_at DESC, request_id);
                CREATE TABLE IF NOT EXISTS project_transfer_imports (
                    request_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    archive_manifest_sha256 TEXT NOT NULL,
                    target_manifest_sha256 TEXT NOT NULL,
                    operational_payload_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('database_imported', 'complete')),
                    event_id_map_json TEXT NOT NULL,
                    receipt_id_map_json TEXT NOT NULL,
                    publication_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(request_id) REFERENCES project_transfer_requests(request_id),
                    CHECK(
                        (status = 'database_imported'
                            AND publication_sha256 IS NULL AND completed_at IS NULL)
                        OR (status = 'complete'
                            AND publication_sha256 IS NOT NULL AND completed_at IS NOT NULL)
                    )
                );
                CREATE INDEX IF NOT EXISTS project_transfer_imports_project
                    ON project_transfer_imports(project_id, created_at, request_id);
                CREATE TABLE IF NOT EXISTS project_transfer_import_configurations (
                    request_id TEXT PRIMARY KEY,
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES project_transfer_imports(request_id)
                );
                CREATE TABLE IF NOT EXISTS project_transfer_proofs (
                    request_id TEXT PRIMARY KEY,
                    proof_kind TEXT NOT NULL
                        CHECK(proof_kind IN ('source_release', 'target_activation')),
                    state TEXT NOT NULL
                        CHECK(state IN ('unexposed', 'exposed', 'acknowledged', 'consumed')),
                    commitment_sha256 TEXT NOT NULL,
                    secret BLOB,
                    acknowledgement_sha256 TEXT,
                    exposed_at TEXT,
                    acknowledged_at TEXT,
                    consumed_at TEXT,
                    FOREIGN KEY(request_id) REFERENCES project_transfer_requests(request_id),
                    CHECK(
                        (state = 'unexposed' AND secret IS NOT NULL
                            AND acknowledgement_sha256 IS NULL
                            AND exposed_at IS NULL AND acknowledged_at IS NULL
                            AND consumed_at IS NULL)
                        OR (state = 'exposed' AND secret IS NOT NULL
                            AND acknowledgement_sha256 IS NULL
                            AND exposed_at IS NOT NULL AND acknowledged_at IS NULL
                            AND consumed_at IS NULL)
                        OR (state = 'acknowledged' AND secret IS NOT NULL
                            AND acknowledgement_sha256 IS NOT NULL
                            AND exposed_at IS NOT NULL AND acknowledged_at IS NOT NULL
                            AND consumed_at IS NULL)
                        OR (state = 'consumed' AND secret IS NULL
                            AND acknowledgement_sha256 IS NOT NULL
                            AND exposed_at IS NOT NULL AND acknowledged_at IS NOT NULL
                            AND consumed_at IS NOT NULL)
                    )
                );
                CREATE TABLE IF NOT EXISTS project_transfer_uploads (
                    request_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    archive_sha256 TEXT NOT NULL,
                    archive_size_bytes INTEGER NOT NULL CHECK(archive_size_bytes >= 1),
                    lease_boundary_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('active', 'complete', 'consumed', 'invalidated')),
                    receipt_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    invalidated_at TEXT,
                    FOREIGN KEY(request_id) REFERENCES project_transfer_requests(request_id),
                    CHECK(
                        (status = 'active' AND receipt_json IS NULL AND invalidated_at IS NULL)
                        OR (status = 'complete' AND receipt_json IS NOT NULL
                            AND invalidated_at IS NULL)
                        OR (status = 'consumed' AND receipt_json IS NOT NULL
                            AND invalidated_at IS NULL)
                        OR (status = 'invalidated' AND receipt_json IS NULL
                            AND invalidated_at IS NOT NULL)
                    )
                );
                CREATE TABLE IF NOT EXISTS project_transfer_activations (
                    target_request_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL UNIQUE,
                    receipt_json TEXT NOT NULL,
                    activated_at TEXT NOT NULL,
                    FOREIGN KEY(target_request_id)
                        REFERENCES project_transfer_requests(request_id)
                );
                CREATE TABLE IF NOT EXISTS project_transfer_restore_reentries (
                    target_request_id TEXT NOT NULL,
                    restored_revision INTEGER NOT NULL CHECK(restored_revision >= 0),
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(target_request_id, restored_revision),
                    FOREIGN KEY(target_request_id)
                        REFERENCES project_transfer_requests(request_id)
                );
                CREATE TABLE IF NOT EXISTS project_aliases (
                    alias_id TEXT PRIMARY KEY,
                    canonical_project_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS project_aliases_canonical
                    ON project_aliases(canonical_project_id, alias_id);
                CREATE TABLE IF NOT EXISTS provider_skill_inventories (
                    provider TEXT NOT NULL,
                    host TEXT NOT NULL,
                    configured_binary TEXT NOT NULL,
                    resolved_binary TEXT,
                    provider_version TEXT,
                    command_json TEXT NOT NULL DEFAULT '[]',
                    protocol TEXT,
                    skills_json TEXT NOT NULL DEFAULT '[]',
                    inventory_hash TEXT,
                    status TEXT NOT NULL,
                    diagnostic TEXT,
                    refreshed_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(provider, host, configured_binary)
                );
                CREATE TABLE IF NOT EXISTS graph_runs (
                    operation_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    episode_id TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    status_message TEXT NOT NULL,
                    error TEXT,
                    applied_revision INTEGER,
                    result_json TEXT,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    parent_operation_id TEXT,
                    runtime_id TEXT NOT NULL DEFAULT '',
                    native_session_id TEXT,
                    history_only INTEGER NOT NULL DEFAULT 0,
                    stage_host TEXT,
                    stage_root TEXT,
                    graph_target_json TEXT NOT NULL DEFAULT '{"kind":"main","branch_id":null}',
                    write_scope_fingerprint TEXT,
                    estimate_seconds REAL NOT NULL DEFAULT 300,
                    estimate_samples INTEGER NOT NULL DEFAULT 0,
                    phase TEXT NOT NULL DEFAULT 'queued',
                    last_activity_at TEXT,
                    dispatch_authority_json TEXT,
                    authorized_space_id TEXT,
                    authorized_user_id TEXT,
                    authorized_display_name TEXT,
                    visible INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS graph_runs_project
                    ON graph_runs(project_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS agent_usage (
                    usage_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    task_kind TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT,
                    provider_profile TEXT NOT NULL,
                    provider_event_type TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    counted INTEGER NOT NULL,
                    count_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    processed_input_tokens INTEGER NOT NULL,
                    generated_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    cache_creation_input_tokens INTEGER NOT NULL,
                    cache_write_input_tokens INTEGER NOT NULL,
                    reasoning_output_tokens INTEGER NOT NULL,
                    reported_input_tokens INTEGER,
                    reported_output_tokens INTEGER,
                    reported_total_tokens INTEGER,
                    provider_fields_json TEXT NOT NULL,
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS agent_usage_project
                    ON agent_usage(project_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS graph_run_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    event_kind TEXT NOT NULL DEFAULT 'message',
                    command_id TEXT,
                    episode_id TEXT,
                    command_verb TEXT,
                    command_phase TEXT,
                    idempotency_key TEXT,
                    payload_json TEXT,
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS graph_run_events_operation
                    ON graph_run_events(operation_id, event_id);
                CREATE TABLE IF NOT EXISTS graph_run_receipts (
                    receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    category TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS graph_run_receipts_operation
                    ON graph_run_receipts(operation_id, receipt_id);
                CREATE TABLE IF NOT EXISTS graph_run_outputs (
                    operation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    patch_json TEXT NOT NULL,
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE TABLE IF NOT EXISTS graph_run_contracts (
                    operation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    content TEXT NOT NULL,
                    PRIMARY KEY(operation_id, role),
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE TABLE IF NOT EXISTS watchers (
                    watcher_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    origin_operation_id TEXT NOT NULL,
                    origin_task_kind TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    node_id TEXT,
                    episode_id TEXT,
                    graph_target_json TEXT NOT NULL DEFAULT '{"kind":"main","branch_id":null}',
                    execution_host TEXT NOT NULL,
                    check_command TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    graph_condition_json TEXT,
                    armed_revision INTEGER,
                    continuation_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_checked_at TEXT,
                    last_exit_code INTEGER,
                    last_error TEXT,
                    completed_at TEXT,
                    next_check_at TEXT,
                    consecutive_error_count INTEGER NOT NULL DEFAULT 0,
                    group_id TEXT,
                    group_label TEXT,
                    notified INTEGER NOT NULL DEFAULT 0,
                    notification_operation_id TEXT,
                    stopped_by TEXT,
                    stop_reason TEXT,
                    stopped_at TEXT,
                    stop_operation_id TEXT
                );
                CREATE INDEX IF NOT EXISTS watchers_project
                    ON watchers(project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS watchers_pollable
                    ON watchers(status, created_at);
                CREATE INDEX IF NOT EXISTS watchers_delivery
                    ON watchers(project_id, origin_operation_id, notified, completed_at);
                CREATE TABLE IF NOT EXISTS graph_watcher_reconciliation (
                    project_id TEXT NOT NULL,
                    graph_target_key TEXT NOT NULL,
                    graph_target_json TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    transition_id TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, graph_target_key)
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('auto_research', 'experiment_loop')),
                    control_node_id TEXT,
                    graph_target_json TEXT NOT NULL DEFAULT '{"kind":"main","branch_id":null}',
                    graph_base_head_json TEXT,
                    root_operation_id TEXT,
                    status TEXT NOT NULL,
                    invocation_ceiling INTEGER NOT NULL CHECK(invocation_ceiling >= 1),
                    invocations_used INTEGER NOT NULL DEFAULT 0
                        CHECK(invocations_used >= 0 AND invocations_used <= invocation_ceiling),
                    authorized_space_id TEXT,
                    authorized_user_id TEXT,
                    authorized_display_name TEXT,
                    stop_requested_at TEXT,
                    stop_settled_at TEXT,
                    ending TEXT,
                    ending_diagnostic TEXT,
                    wrapup_state TEXT NOT NULL DEFAULT 'not_started',
                    wrapup_error TEXT,
                    report_attempts_used INTEGER NOT NULL DEFAULT 0
                        CHECK(report_attempts_used >= 0 AND report_attempts_used <= 3),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ended_at TEXT
                );
                CREATE INDEX IF NOT EXISTS episodes_project
                    ON episodes(project_id, created_at DESC, episode_id);
                CREATE TABLE IF NOT EXISTS experiment_episode_state (
                    episode_id TEXT PRIMARY KEY,
                    provider TEXT,
                    execution_machine TEXT,
                    execution_host TEXT NOT NULL DEFAULT '',
                    native_session_id TEXT,
                    stage_host TEXT,
                    stage_root TEXT,
                    chat_id TEXT,
                    last_turn_operation_id TEXT,
                    last_turn_invocation INTEGER,
                    last_graph_result TEXT,
                    last_watcher_ids_json TEXT NOT NULL DEFAULT '[]',
                    context_baseline_json TEXT NOT NULL DEFAULT '{}',
                    session_diagnostic TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
                );
                CREATE TABLE IF NOT EXISTS auto_research_episodes (
                    episode_id TEXT PRIMARY KEY,
                    starting_instruction TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
                );
                CREATE TABLE IF NOT EXISTS auto_research_invocations (
                    episode_id TEXT NOT NULL,
                    operation_id TEXT PRIMARY KEY,
                    allocation_operation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('orchestrator', 'worker')),
                    actor_operation_id TEXT NOT NULL,
                    control_node_id TEXT,
                    handoffs_cleared_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id),
                    FOREIGN KEY(allocation_operation_id) REFERENCES graph_runs(operation_id),
                    FOREIGN KEY(actor_operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_invocations_episode
                    ON auto_research_invocations(episode_id, created_at, operation_id);
                CREATE INDEX IF NOT EXISTS auto_research_invocations_actor
                    ON auto_research_invocations(
                        episode_id, actor_operation_id, created_at, operation_id
                    );
                CREATE TABLE IF NOT EXISTS auto_research_messages (
                    message_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    sender_role TEXT NOT NULL,
                    sender_task_id TEXT,
                    authorized_space_id TEXT,
                    authorized_user_id TEXT,
                    authorized_display_name TEXT,
                    recipient_task_id TEXT NOT NULL,
                    control_node_id TEXT,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    delivery_operation_id TEXT,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_messages_episode
                    ON auto_research_messages(episode_id, created_at, message_id);
                CREATE TABLE IF NOT EXISTS auto_research_recoveries (
                    recovery_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    failure_kind TEXT NOT NULL,
                    retry_mode TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                    max_attempts INTEGER NOT NULL CHECK(max_attempts >= 1),
                    status TEXT NOT NULL,
                    next_attempt_at TEXT,
                    diagnostic TEXT NOT NULL,
                    admitted_operation_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id),
                    FOREIGN KEY(admitted_operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_recoveries_due
                    ON auto_research_recoveries(status, next_attempt_at, created_at);
                CREATE TABLE IF NOT EXISTS auto_research_child_work (
                    worker_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    control_node_id TEXT NOT NULL,
                    root_operation_id TEXT NOT NULL UNIQUE,
                    current_operation_id TEXT NOT NULL UNIQUE,
                    admitted_by_operation_id TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    instruction_sha256 TEXT NOT NULL,
                    stop_requested_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(root_operation_id) REFERENCES graph_runs(operation_id),
                    FOREIGN KEY(current_operation_id) REFERENCES graph_runs(operation_id),
                    FOREIGN KEY(admitted_by_operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_child_work_episode
                    ON auto_research_child_work(episode_id, created_at, worker_id);
                CREATE TABLE IF NOT EXISTS auto_research_child_work_attempts (
                    operation_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    allocation_operation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id),
                    FOREIGN KEY(worker_id) REFERENCES auto_research_child_work(worker_id),
                    FOREIGN KEY(allocation_operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_child_work_attempts_worker
                    ON auto_research_child_work_attempts(worker_id, created_at, operation_id);
                CREATE TABLE IF NOT EXISTS auto_research_child_experiments (
                    child_episode_id TEXT PRIMARY KEY,
                    auto_research_episode_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    control_node_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(
                        state IN ('pending', 'running', 'cancelled', 'terminal')
                    ),
                    replaces_episode_id TEXT,
                    request_json TEXT NOT NULL,
                    goal_sha256 TEXT,
                    parent_operation_id TEXT NOT NULL,
                    terminal_diagnostic TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(auto_research_episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(parent_operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_child_experiments_parent
                    ON auto_research_child_experiments(
                        auto_research_episode_id, created_at, child_episode_id
                    );
                CREATE UNIQUE INDEX IF NOT EXISTS auto_research_pending_experiment_per_node
                    ON auto_research_child_experiments(project_id, control_node_id)
                    WHERE state = 'pending';
                CREATE TABLE IF NOT EXISTS auto_research_experiment_invocations (
                    operation_id TEXT PRIMARY KEY,
                    auto_research_episode_id TEXT NOT NULL,
                    child_episode_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id),
                    FOREIGN KEY(auto_research_episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(child_episode_id)
                        REFERENCES auto_research_child_experiments(child_episode_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_experiment_spend_parent
                    ON auto_research_experiment_invocations(
                        auto_research_episode_id, created_at, operation_id
                    );
                CREATE TABLE IF NOT EXISTS auto_research_child_admissions (
                    admission_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    child_kind TEXT NOT NULL CHECK(child_kind IN ('work', 'experiment')),
                    child_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('accepted', 'reflected', 'cancelled')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(episode_id, child_kind, child_id),
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_child_admissions_pending
                    ON auto_research_child_admissions(episode_id, state, created_at, admission_id);
                CREATE TABLE IF NOT EXISTS auto_research_lifecycle_notices (
                    notice_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_event TEXT NOT NULL,
                    source_attempt INTEGER NOT NULL DEFAULT 1 CHECK(source_attempt >= 1),
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    delivery_operation_id TEXT,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    UNIQUE(episode_id, source_kind, source_id, source_event, source_attempt),
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(delivery_operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_lifecycle_pending
                    ON auto_research_lifecycle_notices(
                        episode_id, acknowledged_at, delivered_at, created_at, notice_id
                    );
                CREATE TABLE IF NOT EXISTS auto_research_inbox_receipts (
                    effect_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('harvest', 'clear')),
                    result_json TEXT NOT NULL,
                    acknowledged_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_inbox_receipts_episode
                    ON auto_research_inbox_receipts(episode_id, created_at, effect_id);
                CREATE TABLE IF NOT EXISTS auto_research_finish_receipts (
                    effect_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    actor_operation_id TEXT NOT NULL,
                    disposition TEXT NOT NULL CHECK(disposition IN ('blocked', 'completed')),
                    blocker_count INTEGER NOT NULL CHECK(blocker_count >= 0),
                    result_json TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_finish_receipts_episode
                    ON auto_research_finish_receipts(episode_id, created_at, effect_id);
                CREATE TABLE IF NOT EXISTS auto_research_apply_results (
                    apply_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    patch_sha256 TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_apply_results_task
                    ON auto_research_apply_results(operation_id, created_at, apply_id);
                CREATE TABLE IF NOT EXISTS auto_research_command_files (
                    command_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('apply', 'instruction', 'goal')),
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS auto_research_command_files_episode
                    ON auto_research_command_files(episode_id, created_at, command_id);
                CREATE TABLE IF NOT EXISTS episode_invocations (
                    episode_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL UNIQUE,
                    invocation_number INTEGER NOT NULL CHECK(invocation_number >= 1),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(episode_id, invocation_number),
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS episode_invocations_episode
                    ON episode_invocations(episode_id, invocation_number);
                CREATE TABLE IF NOT EXISTS episode_report_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL CHECK(attempt_number BETWEEN 1 AND 3),
                    allocation_operation_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    UNIQUE(episode_id, attempt_number),
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(allocation_operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE INDEX IF NOT EXISTS episode_report_attempts_current
                    ON episode_report_attempts(episode_id, status, attempt_number DESC);
                CREATE TABLE IF NOT EXISTS episode_wrapups (
                    episode_id TEXT PRIMARY KEY,
                    ending TEXT,
                    partial INTEGER NOT NULL,
                    concluding_operation_id TEXT,
                    allocation_operation_id TEXT UNIQUE,
                    provider TEXT,
                    run_on TEXT,
                    execution_host TEXT,
                    native_session_id TEXT,
                    stage_host TEXT,
                    stage_root TEXT,
                    skill_id TEXT,
                    skill_version TEXT,
                    output_name TEXT,
                    output_path TEXT,
                    receipt_json TEXT NOT NULL,
                    receipt_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    diagnostic TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(concluding_operation_id) REFERENCES graph_runs(operation_id),
                    FOREIGN KEY(allocation_operation_id) REFERENCES graph_runs(operation_id)
                );
                CREATE TABLE IF NOT EXISTS episode_reports (
                    report_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL UNIQUE,
                    attempt_id TEXT NOT NULL UNIQUE,
                    allocation_operation_id TEXT NOT NULL UNIQUE,
                    ending TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    html TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(attempt_id) REFERENCES episode_report_attempts(attempt_id),
                    FOREIGN KEY(allocation_operation_id) REFERENCES graph_runs(operation_id)
                );
                """
            )
            self._migrate_episode_lineage(connection)
            # Legacy graph_runs tables may still expose campaign_id (or no
            # lineage column at all) until the migration above.  Build the
            # branch-merge index only after episode_id is guaranteed to exist.
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS graph_runs_active_branch_merge "
                "ON graph_runs(episode_id) "
                "WHERE kind = 'branch_merge' "
                "AND status IN ('queued', 'running', 'pausing')"
            )
            has_legacy_campaigns = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'campaigns'"
                ).fetchone()
                is not None
            )
            # Existing v0.2 databases need additive migration before the index
            # can include the new transitional state.
            self._ensure_column(connection, "projects", "home_space_id", "TEXT")
            self._ensure_column(connection, "projects", "retired_at", "TEXT")
            self._ensure_column(
                connection,
                "projects",
                "retired_transfer_request_id",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "project_provisioning_requests",
                "project_config_json",
                "TEXT",
            )
            self._allow_consumed_project_transfer_uploads(connection)
            self._ensure_column(connection, "space_identity", "space_name", "TEXT")
            self._ensure_column(connection, "space_users", "removal_started_at", "TEXT")
            self._ensure_column(connection, "space_users", "removed_at", "TEXT")
            self._ensure_column(connection, "team_bootstrap_codes", "revoked_at", "TEXT")
            self._ensure_column(connection, "team_invitations", "revoked_at", "TEXT")
            self._migrate_project_invitation_revocation(connection)
            self._ensure_column(connection, "paper_drafts", "ancestor_content", "TEXT")
            self._ensure_column(
                connection,
                "result_views",
                "html",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "writing_sessions",
                "runtime_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            for row in connection.execute(
                "SELECT native_session_id, provider FROM writing_sessions WHERE runtime_id = ''"
            ).fetchall():
                try:
                    runtime_id = legacy_runtime_id(row["provider"])
                except ValueError:
                    # Old raw rows may name a provider RCP no longer supports;
                    # preserve them for project deletion and forensic export.
                    continue
                connection.execute(
                    "UPDATE writing_sessions SET runtime_id = ? WHERE native_session_id = ?",
                    (runtime_id, row["native_session_id"]),
                )
            self._ensure_column(connection, "graph_runs", "attempt", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(connection, "graph_runs", "parent_operation_id", "TEXT")
            self._ensure_column(
                connection,
                "graph_runs",
                "runtime_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            # One statement per supported provider rather than a row-at-a-time
            # loop: request_json holds each task's whole payload, and reading
            # every historical one into this process just to name its provider
            # blocks the first open after upgrade. Rows naming a provider RCP no
            # longer supports match nothing and keep an empty runtime, the same
            # way the writing_sessions backfill above preserves them.
            for supported in PROVIDER_IDS:
                connection.execute(
                    """
                    UPDATE graph_runs SET runtime_id = ?
                    WHERE runtime_id = ''
                      AND json_extract(request_json, '$.provider') = ?
                    """,
                    (legacy_runtime_id(supported), supported),
                )
            self._ensure_column(connection, "graph_runs", "native_session_id", "TEXT")
            self._ensure_column(
                connection,
                "graph_runs",
                "history_only",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(connection, "graph_runs", "stage_host", "TEXT")
            self._ensure_column(connection, "graph_runs", "stage_root", "TEXT")
            self._ensure_column(
                connection,
                "graph_runs",
                "graph_target_json",
                'TEXT NOT NULL DEFAULT \'{"kind":"main","branch_id":null}\'',
            )
            self._ensure_column(connection, "graph_runs", "write_scope_fingerprint", "TEXT")
            self._ensure_column(
                connection, "graph_runs", "estimate_seconds", "REAL NOT NULL DEFAULT 300"
            )
            self._ensure_column(
                connection, "graph_runs", "estimate_samples", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(connection, "graph_runs", "phase", "TEXT NOT NULL DEFAULT 'queued'")
            self._ensure_column(connection, "graph_runs", "last_activity_at", "TEXT")
            self._ensure_column(connection, "graph_runs", "result_json", "TEXT")
            self._ensure_column(connection, "graph_runs", "dispatch_authority_json", "TEXT")
            self._ensure_column(connection, "graph_runs", "authorized_space_id", "TEXT")
            self._ensure_column(connection, "graph_runs", "authorized_user_id", "TEXT")
            self._ensure_column(connection, "graph_runs", "authorized_display_name", "TEXT")
            self._ensure_column(
                connection,
                "graph_runs",
                "visible",
                "INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                connection,
                "watchers",
                "graph_target_json",
                'TEXT NOT NULL DEFAULT \'{"kind":"main","branch_id":null}\'',
            )
            self._ensure_column(
                connection,
                "episodes",
                "graph_target_json",
                'TEXT NOT NULL DEFAULT \'{"kind":"main","branch_id":null}\'',
            )
            self._ensure_column(connection, "episodes", "graph_base_head_json", "TEXT")
            self._ensure_column(
                connection,
                "auto_research_child_work",
                "admitted_by_operation_id",
                "TEXT",
            )
            connection.execute(
                """
                UPDATE auto_research_child_work
                SET admitted_by_operation_id = root_operation_id
                WHERE admitted_by_operation_id IS NULL
                """
            )
            if has_legacy_campaigns:
                self._ensure_column(connection, "campaigns", "authorized_space_id", "TEXT")
                self._ensure_column(connection, "campaigns", "authorized_user_id", "TEXT")
                self._ensure_column(connection, "campaigns", "authorized_display_name", "TEXT")
                if (
                    connection.execute(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'campaign_messages'"
                    ).fetchone()
                    is not None
                ):
                    self._ensure_column(
                        connection, "campaign_messages", "authorized_space_id", "TEXT"
                    )
                    self._ensure_column(
                        connection, "campaign_messages", "authorized_user_id", "TEXT"
                    )
                    self._ensure_column(
                        connection,
                        "campaign_messages",
                        "authorized_display_name",
                        "TEXT",
                    )
            self._ensure_column(
                connection,
                "graph_run_events",
                "event_kind",
                "TEXT NOT NULL DEFAULT 'message'",
            )
            self._ensure_column(connection, "graph_run_events", "command_id", "TEXT")
            self._ensure_column(connection, "graph_run_events", "command_verb", "TEXT")
            self._ensure_column(connection, "graph_run_events", "command_phase", "TEXT")
            self._ensure_column(connection, "graph_run_events", "idempotency_key", "TEXT")
            self._ensure_column(connection, "graph_run_events", "payload_json", "TEXT")
            self._ensure_column(connection, "watchers", "next_check_at", "TEXT")
            self._ensure_column(
                connection,
                "watchers",
                "consecutive_error_count",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(connection, "watchers", "group_id", "TEXT")
            self._ensure_column(connection, "watchers", "group_label", "TEXT")
            self._ensure_column(connection, "watchers", "stopped_by", "TEXT")
            self._ensure_column(connection, "watchers", "stop_reason", "TEXT")
            self._ensure_column(connection, "watchers", "stopped_at", "TEXT")
            self._ensure_column(connection, "watchers", "stop_operation_id", "TEXT")
            self._ensure_column(connection, "watchers", "graph_condition_json", "TEXT")
            self._ensure_column(connection, "watchers", "armed_revision", "INTEGER")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS result_views_project_experiment "
                "ON result_views(project_id, experiment_id, updated_at DESC, view_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS result_views_project_chat "
                "ON result_views(project_id, chat_id, updated_at DESC, view_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS result_views_expiry "
                "ON result_views(expires_at, kept_filename)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS team_member_tokens_hash "
                "ON team_member_tokens(token_hash)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS team_member_tokens_active_user "
                "ON team_member_tokens(user_id) WHERE revoked_at IS NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS team_invitations_creator "
                "ON team_invitations(created_by, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS team_sessions_user_expiry "
                "ON team_sessions(user_id, expires_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS project_members_user "
                "ON project_members(user_id, project_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS project_invitations_invitee "
                "ON project_invitations(invited_user_id, created_at DESC, invitation_id)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS project_invitations_pending "
                "ON project_invitations(project_id, invited_user_id) WHERE response IS NULL"
            )
            if not members_table_existed:
                self._backfill_project_members(connection)
            self._relax_episode_wrapup_ending(connection)
            connection.execute("DROP INDEX IF EXISTS graph_runs_campaign")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS graph_runs_episode "
                "ON graph_runs(episode_id, created_at, operation_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS graph_run_events_command "
                "ON graph_run_events(command_id, command_phase, event_id)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS graph_run_events_command_start_id "
                "ON graph_run_events(command_id) "
                "WHERE event_kind = 'command' AND command_phase = 'start'"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS graph_run_events_command_exit_id "
                "ON graph_run_events(command_id) "
                "WHERE event_kind = 'command' AND command_phase = 'exit'"
            )
            connection.execute("DROP INDEX IF EXISTS graph_run_events_campaign_key_start")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS graph_run_events_episode_key_start "
                "ON graph_run_events(episode_id, idempotency_key) "
                "WHERE event_kind = 'command' AND command_phase = 'start' "
                "AND episode_id IS NOT NULL AND idempotency_key IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS watchers_due "
                "ON watchers(status, next_check_at, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS watchers_due_unclaimed "
                "ON watchers(status, notified, next_check_at, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS watchers_group_members "
                "ON watchers(group_id, created_at, watcher_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS watchers_group_delivery_candidates "
                "ON watchers(notified, status, group_id, consecutive_error_count)"
            )
            connection.execute("DROP INDEX IF EXISTS watchers_experiment_episode")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS watchers_episode "
                "ON watchers(project_id, node_id, episode_id, status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS watchers_graph_conditions "
                "ON watchers(project_id, status, notified, graph_condition_json)"
            )
            migrate_legacy_episodes(connection)
            self._migrate_experiment_episode_state(connection)
            if has_legacy_campaigns:
                # The generic parent/report copy must finish before the source
                # tables move under private archive names.
                migrate_legacy_auto_research(connection)
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS episodes_one_live_auto_project "
                "ON episodes(project_id) WHERE mode = 'auto_research' "
                "AND status IN ('queued', 'running', 'stopping', 'wrapping_up')"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS episodes_one_live_experiment_control "
                "ON episodes(project_id, control_node_id) WHERE mode = 'experiment_loop' "
                "AND status IN ('queued', 'running', 'stopping', 'wrapping_up')"
            )
            connection.execute("DROP INDEX IF EXISTS graph_runs_active_project")
            connection.execute("DROP INDEX IF EXISTS agent_tasks_active_project")
            if issue_bootstrap:
                if stored_space_kind != "team" or initial_space_name is None:
                    raise ValueError("A bootstrap code requires a named team space.")
                if recovering_team_initialization:
                    if connection.execute("SELECT 1 FROM space_users LIMIT 1").fetchone():
                        raise ValueError("This RCP data directory already contains a space.")
                    connection.execute("DELETE FROM team_bootstrap_codes")
                bootstrap_code, code_id, code_hash = _new_enrollment_code("bootstrap")
                connection.execute(
                    """
                    INSERT INTO team_bootstrap_codes (code_id, code_hash, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (code_id, code_hash, self.now()),
                )
        return bootstrap_code

    @classmethod
    def _migrate_episode_lineage(cls, connection: sqlite3.Connection) -> None:
        """Replace the three legacy parent columns and canonicalize stored JSON once."""

        connection.execute("DROP INDEX IF EXISTS graph_runs_campaign")
        connection.execute("DROP INDEX IF EXISTS graph_run_events_campaign_key_start")
        connection.execute("DROP INDEX IF EXISTS watchers_experiment_episode")
        cls._replace_lineage_column(connection, "graph_runs", "campaign_id")
        cls._replace_lineage_column(connection, "graph_run_events", "campaign_id")
        cls._replace_lineage_column(connection, "watchers", "experiment_episode_id")

        for table, column in (
            ("graph_runs", "request_json"),
            ("graph_runs", "dispatch_authority_json"),
            ("graph_run_events", "payload_json"),
            ("watchers", "continuation_json"),
            ("watchers", "graph_condition_json"),
        ):
            cls._rewrite_lineage_json_column(connection, table, column)

        connection.execute("UPDATE graph_runs SET kind = 'auto_research' WHERE kind = 'campaign'")
        connection.execute(
            "UPDATE agent_usage SET task_kind = 'auto_research' WHERE task_kind = 'campaign'"
        )
        connection.execute(
            "UPDATE watchers SET origin_task_kind = 'auto_research' "
            "WHERE origin_task_kind = 'campaign'"
        )
        connection.execute(
            """
            UPDATE graph_runs
            SET episode_id = COALESCE(
                CASE
                    WHEN json_type(request_json, '$.episode_id') = 'text'
                    THEN json_extract(request_json, '$.episode_id')
                END,
                CASE
                    WHEN json_extract(request_json, '$.patch_kind') = 'experiment_loop'
                     AND json_type(request_json, '$.control_episode_id') = 'text'
                    THEN json_extract(request_json, '$.control_episode_id')
                END
            )
            WHERE episode_id IS NULL
            """
        )
        connection.execute(
            """
            UPDATE graph_run_events
            SET episode_id = (
                SELECT run.episode_id FROM graph_runs AS run
                WHERE run.operation_id = graph_run_events.operation_id
            )
            WHERE episode_id IS NULL
            """
        )
        cls._backfill_watcher_episode_lineage(connection)

    @staticmethod
    def _migrate_experiment_episode_state(connection: sqlite3.Connection) -> None:
        """Move the legacy combined Experiment parent into its mode-only child."""

        legacy_exists = (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'experiment_episodes'"
            ).fetchone()
            is not None
        )
        if legacy_exists:
            connection.execute(
                """
                INSERT INTO experiment_episode_state (
                    episode_id, provider, execution_machine, execution_host,
                    native_session_id, stage_host, stage_root, chat_id,
                    last_turn_operation_id, last_turn_invocation, last_graph_result,
                    last_watcher_ids_json, context_baseline_json, session_diagnostic,
                    created_at, updated_at
                )
                SELECT legacy.episode_id, legacy.provider, legacy.execution_machine,
                       legacy.execution_host, legacy.native_session_id, legacy.stage_host,
                       legacy.stage_root, legacy.chat_id, legacy.last_turn_operation_id,
                       legacy.last_turn_invocation, legacy.last_graph_result,
                       legacy.last_watcher_ids_json, legacy.context_baseline_json,
                       legacy.session_diagnostic, legacy.created_at, legacy.updated_at
                FROM experiment_episodes AS legacy
                JOIN episodes AS episode ON episode.episode_id = legacy.episode_id
                WHERE episode.mode = 'experiment_loop'
                ON CONFLICT(episode_id) DO NOTHING
                """
            )
            orphan = connection.execute(
                """
                SELECT legacy.episode_id
                FROM experiment_episodes AS legacy
                LEFT JOIN episodes AS episode ON episode.episode_id = legacy.episode_id
                WHERE episode.episode_id IS NULL
                LIMIT 1
                """
            ).fetchone()
            if orphan is not None:
                raise ValueError(
                    "Legacy Experiment episode could not be moved to the generic parent: "
                    f"{orphan['episode_id']}"
                )
            connection.execute("DROP INDEX IF EXISTS experiment_episodes_control")
            connection.execute("DROP TABLE experiment_episodes")
        connection.execute(
            """
            INSERT INTO experiment_episode_state (episode_id, created_at, updated_at)
            SELECT episode_id, created_at, updated_at
            FROM episodes
            WHERE mode = 'experiment_loop'
            ON CONFLICT(episode_id) DO NOTHING
            """
        )

    @staticmethod
    def _replace_lineage_column(
        connection: sqlite3.Connection,
        table: str,
        legacy_name: str,
    ) -> None:
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if legacy_name in columns and "episode_id" in columns:
            conflict = connection.execute(
                f"SELECT 1 FROM {table} WHERE {legacy_name} IS NOT NULL "
                f"AND episode_id IS NOT NULL AND {legacy_name} != episode_id LIMIT 1"
            ).fetchone()
            if conflict is not None:
                raise ValueError(f"{table} has conflicting legacy and canonical episode lineage")
            connection.execute(
                f"UPDATE {table} SET episode_id = COALESCE(episode_id, {legacy_name})"
            )
            connection.execute(f"ALTER TABLE {table} DROP COLUMN {legacy_name}")
            return
        if legacy_name in columns:
            connection.execute(f"ALTER TABLE {table} RENAME COLUMN {legacy_name} TO episode_id")
            return
        if "episode_id" not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN episode_id TEXT")

    @classmethod
    def _rewrite_lineage_json_column(
        cls,
        connection: sqlite3.Connection,
        table: str,
        column: str,
    ) -> None:
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            return
        rows = connection.execute(
            f"SELECT rowid AS lineage_rowid, {column} FROM {table} WHERE {column} IS NOT NULL"
        ).fetchall()
        for row in rows:
            try:
                value = json.loads(row[column])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{table}.{column} contains invalid JSON") from exc
            lineage_keys = cls._lineage_json_keys(value)
            if lineage_keys == {"campaign_id", "episode_id"}:
                raise ValueError(
                    f"{table}.{column} row {row['lineage_rowid']} is ambiguous: "
                    "it contains both campaign_id and episode_id"
                )
            rewritten, changed = cls._rewrite_lineage_json_value(
                value,
                location=f"{table}.{column} row {row['lineage_rowid']}",
            )
            if changed:
                connection.execute(
                    f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                    (
                        json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")),
                        row["lineage_rowid"],
                    ),
                )

    @classmethod
    def _lineage_json_keys(cls, value: object) -> set[str]:
        if isinstance(value, dict):
            keys = {key for key in value if key in {"campaign_id", "episode_id"}}
            for item in value.values():
                keys.update(cls._lineage_json_keys(item))
            return keys
        if isinstance(value, list):
            keys: set[str] = set()
            for item in value:
                keys.update(cls._lineage_json_keys(item))
            return keys
        return set()

    @classmethod
    def _rewrite_lineage_json_value(
        cls,
        value: object,
        *,
        location: str,
    ) -> tuple[object, bool]:
        if isinstance(value, dict):
            if "campaign_id" in value and "episode_id" in value:
                raise ValueError(
                    f"{location} is ambiguous: it contains both campaign_id and episode_id"
                )
            changed = "campaign_id" in value
            rewritten: dict[str, object] = {}
            for key, item in value.items():
                canonical_key = "episode_id" if key == "campaign_id" else key
                canonical_item, item_changed = cls._rewrite_lineage_json_value(
                    item,
                    location=f"{location}.{canonical_key}",
                )
                rewritten[canonical_key] = canonical_item
                changed = changed or item_changed
            return rewritten, changed
        if isinstance(value, list):
            changed = False
            rewritten_items: list[object] = []
            for index, item in enumerate(value):
                canonical_item, item_changed = cls._rewrite_lineage_json_value(
                    item,
                    location=f"{location}[{index}]",
                )
                rewritten_items.append(canonical_item)
                changed = changed or item_changed
            return rewritten_items, changed
        return value, False

    @staticmethod
    def _backfill_watcher_episode_lineage(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT watcher.watcher_id, watcher.episode_id, watcher.continuation_json,
                   origin.episode_id AS origin_episode_id,
                   notification.episode_id AS notification_episode_id
            FROM watchers AS watcher
            LEFT JOIN graph_runs AS origin
              ON origin.operation_id = watcher.origin_operation_id
            LEFT JOIN graph_runs AS notification
              ON notification.operation_id = watcher.notification_operation_id
            WHERE watcher.episode_id IS NULL
            """
        ).fetchall()
        for row in rows:
            continuation = json.loads(row["continuation_json"])
            candidates = {
                candidate
                for candidate in (
                    row["origin_episode_id"],
                    row["notification_episode_id"],
                    (
                        continuation.get("control_episode_id")
                        if continuation.get("patch_kind") == "experiment_loop"
                        else None
                    ),
                )
                if isinstance(candidate, str) and candidate
            }
            if len(candidates) == 1:
                connection.execute(
                    "UPDATE watchers SET episode_id = ? WHERE watcher_id = ?",
                    (next(iter(candidates)), row["watcher_id"]),
                )

    @staticmethod
    def _allow_consumed_project_transfer_uploads(connection: sqlite3.Connection) -> None:
        """Extend the closed upload lifecycle without discarding retained receipts."""

        row = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'project_transfer_uploads'"
        ).fetchone()
        if row is None or "'consumed'" in str(row[0]):
            return
        connection.execute(
            """
            CREATE TABLE project_transfer_uploads_with_consumed (
                request_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                archive_sha256 TEXT NOT NULL,
                archive_size_bytes INTEGER NOT NULL CHECK(archive_size_bytes >= 1),
                lease_boundary_sha256 TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK(status IN ('active', 'complete', 'consumed', 'invalidated')),
                receipt_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                invalidated_at TEXT,
                FOREIGN KEY(request_id) REFERENCES project_transfer_requests(request_id),
                CHECK(
                    (status = 'active' AND receipt_json IS NULL AND invalidated_at IS NULL)
                    OR (status IN ('complete', 'consumed') AND receipt_json IS NOT NULL
                        AND invalidated_at IS NULL)
                    OR (status = 'invalidated' AND receipt_json IS NULL
                        AND invalidated_at IS NOT NULL)
                )
            )
            """
        )
        connection.execute(
            """
            INSERT INTO project_transfer_uploads_with_consumed (
                request_id, project_id, archive_sha256, archive_size_bytes,
                lease_boundary_sha256, status, receipt_json, created_at,
                updated_at, invalidated_at
            )
            SELECT request_id, project_id, archive_sha256, archive_size_bytes,
                   lease_boundary_sha256, status, receipt_json, created_at,
                   updated_at, invalidated_at
            FROM project_transfer_uploads
            """
        )
        connection.execute("DROP TABLE project_transfer_uploads")
        connection.execute(
            "ALTER TABLE project_transfer_uploads_with_consumed RENAME TO project_transfer_uploads"
        )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        name: str,
        definition: str,
    ) -> None:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if name not in columns:
            try:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            except sqlite3.OperationalError:
                columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                if name not in columns:
                    raise

    @staticmethod
    def _migrate_project_invitation_revocation(connection: sqlite3.Connection) -> None:
        """Extend the closed invitation response vocabulary without losing history."""

        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'project_invitations'"
        ).fetchone()
        if row is None or "'revoked'" in str(row[0]):
            return
        connection.execute(
            """
            CREATE TABLE project_invitations_with_revocation (
                invitation_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                invited_user_id TEXT NOT NULL,
                invited_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                response TEXT CHECK(response IN ('accepted', 'declined', 'revoked')),
                responded_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO project_invitations_with_revocation (
                invitation_id, project_id, invited_user_id, invited_by,
                created_at, response, responded_at
            )
            SELECT invitation_id, project_id, invited_user_id, invited_by,
                   created_at, response, responded_at
            FROM project_invitations
            """
        )
        connection.execute("DROP TABLE project_invitations")
        connection.execute(
            "ALTER TABLE project_invitations_with_revocation RENAME TO project_invitations"
        )

    @staticmethod
    def _relax_episode_wrapup_ending(connection: sqlite3.Connection) -> None:
        """Drop the obsolete NOT NULL on ``episode_wrapups.ending``.

        An Experiment whose pre-migration exit cannot be classified from retained
        data legitimately has **no** ending — refusing to invent one is the whole
        point of that branch in ``_legacy_experiment_lifecycle``. The column was
        relaxed to nullable in the create path, but `CREATE TABLE IF NOT EXISTS`
        never alters a table that already exists, so every database created
        before that change still refuses the row and crashes on open.

        SQLite cannot drop a NOT NULL in place, so this rebuilds the table. It is
        guarded on the constraint actually being present.

        These are separate `execute` calls on purpose: `executescript` issues an
        implicit COMMIT first, which would land every earlier migration in this
        open transaction and break the all-or-nothing property of schema setup.
        """

        columns = {row[1]: row for row in connection.execute("PRAGMA table_info(episode_wrapups)")}
        ending = columns.get("ending")
        if ending is None or not ending[3]:
            return
        connection.execute(
            """
            CREATE TABLE episode_wrapups_rebuilt (
                episode_id TEXT PRIMARY KEY,
                ending TEXT,
                partial INTEGER NOT NULL,
                concluding_operation_id TEXT,
                allocation_operation_id TEXT UNIQUE,
                provider TEXT,
                run_on TEXT,
                execution_host TEXT,
                native_session_id TEXT,
                stage_host TEXT,
                stage_root TEXT,
                skill_id TEXT,
                skill_version TEXT,
                output_name TEXT,
                output_path TEXT,
                receipt_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,
                diagnostic TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT,
                FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                FOREIGN KEY(concluding_operation_id) REFERENCES graph_runs(operation_id),
                FOREIGN KEY(allocation_operation_id) REFERENCES graph_runs(operation_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO episode_wrapups_rebuilt SELECT
                episode_id, ending, partial, concluding_operation_id,
                allocation_operation_id, provider, run_on, execution_host,
                native_session_id, stage_host, stage_root, skill_id, skill_version,
                output_name, output_path, receipt_json, receipt_sha256, state,
                diagnostic, created_at, updated_at, finished_at
            FROM episode_wrapups
            """
        )
        connection.execute("DROP TABLE episode_wrapups")
        connection.execute("ALTER TABLE episode_wrapups_rebuilt RENAME TO episode_wrapups")

    def _backfill_project_members(self, connection: sqlite3.Connection) -> None:
        """Seat every current space member on every project registered before S101.

        Nothing records who created those projects, so there is nothing narrower
        to seed from. This is the one place the membership design fails open, and
        it fails open exactly once: failing closed would lock a team out of its
        own projects, and there is no administrator rank to undo that.
        """

        now = self.now()
        connection.execute(
            """
            INSERT OR IGNORE INTO project_members (project_id, user_id, seated_at, seated_by)
            SELECT projects.project_id, space_users.user_id, ?, NULL
            FROM projects CROSS JOIN space_users
            """,
            (now,),
        )

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat()
