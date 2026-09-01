from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from rcp.artifacts import ResultViewDescriptor
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


class ResultViewStoreMixin:
    """Durable result-view records and their bounded HTML bytes."""

    def create_result_view(self, record: ResultViewRecord, *, html: bytes) -> ResultViewRecord:
        """Atomically insert one private result-view binding and its verified HTML."""
        record = ResultViewRecord.model_validate(record)
        stored_html = _validated_result_view_html(record, html)
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO result_views (
                        view_id, project_id, experiment_id, chat_id,
                        origin_operation_id, latest_operation_id,
                        provider, model, reasoning, run_on,
                        native_session_id, stage_host, stage_root, source_name,
                        content_sha256, size_bytes, html, created_at, updated_at, expires_at,
                        kept_filename, kept_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.view_id,
                        record.project_id,
                        record.experiment_id,
                        record.chat_id,
                        record.origin_operation_id,
                        record.latest_operation_id,
                        record.provider,
                        record.model,
                        record.reasoning,
                        record.run_on,
                        record.native_session_id,
                        record.stage_host,
                        record.stage_root,
                        record.source_name,
                        record.content_sha256,
                        record.size_bytes,
                        stored_html,
                        record.created_at,
                        record.updated_at,
                        record.expires_at,
                        record.kept_filename,
                        record.kept_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Result view {record.view_id!r} already exists.") from exc
        return record

    def result_view(
        self,
        view_id: str,
        *,
        include_expired: bool = False,
        as_of: datetime | None = None,
    ) -> ResultViewRecord | None:
        """Return one visible result view, unless diagnostics explicitly include expiry."""
        record = self.result_view_for_diagnostics(view_id)
        if record is None or include_expired or _result_view_is_visible(record, as_of=as_of):
            return record
        return None

    def result_view_for_diagnostics(self, view_id: str) -> ResultViewRecord | None:
        """Return private metadata even after a temporary view expires."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM result_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
        return self._result_view_record(row) if row is not None else None

    def list_result_views(
        self,
        project_id: str,
        *,
        experiment_id: str | None = None,
        chat_id: str | None = None,
        as_of: datetime | None = None,
    ) -> list[ResultViewRecord]:
        """List visible views while retaining kept records past scratch expiry."""
        clauses = ["project_id = ?"]
        values: list[str] = [project_id]
        if experiment_id is not None:
            clauses.append("experiment_id = ?")
            values.append(experiment_id)
        if chat_id is not None:
            clauses.append("chat_id = ?")
            values.append(chat_id)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM result_views
                WHERE {" AND ".join(clauses)}
                ORDER BY updated_at DESC, view_id
                """,
                values,
            ).fetchall()
        records = [self._result_view_record(row) for row in rows]
        return [record for record in records if _result_view_is_visible(record, as_of=as_of)]

    def kept_result_views(self, project_id: str) -> list[ResultViewRecord]:
        """Return every typed kept view for durable project capture."""

        try:
            canonical_project_id = _canonical_uuid4(
                project_id,
                label="project identity",
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM result_views
                WHERE project_id = ? AND kept_filename IS NOT NULL
                ORDER BY kept_filename, view_id
                """,
                (canonical_project_id,),
            ).fetchall()
        return [self._result_view_record(row) for row in rows]

    def result_view_bytes(
        self,
        view_id: str,
        *,
        expected_content_sha256: str,
    ) -> bytes:
        """Return the bounded stored HTML only when it matches its metadata and caller digest."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM result_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
        if row is None:
            raise KeyError(view_id)
        record = self._result_view_record(row)
        if record.content_sha256 != expected_content_sha256:
            raise ResultViewConflict("result view changed before its stored bytes were read")
        return _result_view_html_bytes(record, row["html"])

    def delete_expired_result_views(self, *, as_of: datetime | None = None) -> int:
        """Delete expired unkept views so their stored HTML expires with their metadata."""
        current = _result_view_reference_time(as_of)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._delete_expired_result_views_from_connection(connection, current)

    @staticmethod
    def _delete_expired_result_views_from_connection(
        connection: sqlite3.Connection,
        current: datetime,
    ) -> int:
        deleted = 0
        rows = connection.execute(
            "SELECT view_id, expires_at FROM result_views WHERE kept_filename IS NULL"
        ).fetchall()
        for row in rows:
            if _required_timestamp(row["expires_at"]) > current:
                continue
            deleted += connection.execute(
                """
                DELETE FROM result_views
                WHERE view_id = ? AND expires_at = ? AND kept_filename IS NULL
                """,
                (row["view_id"], row["expires_at"]),
            ).rowcount
        return deleted

    def has_active_result_view_revision(self, record: ResultViewRecord) -> bool:
        """Return whether this view has an active or recoverable revision task."""
        record = ResultViewRecord.model_validate(record)
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM graph_runs AS revision
                WHERE revision.project_id = ? AND revision.kind = 'node_chat'
                  AND json_extract(revision.request_json, '$.chat_id') = ?
                  AND json_extract(revision.request_json, '$.result_view.action') = 'revise'
                  AND json_extract(revision.request_json, '$.result_view.view_id') = ?
                  AND (
                    revision.status IN ('queued', 'running', 'pausing')
                    OR (
                      revision.status IN ('paused', 'interrupted')
                      AND revision.native_session_id IS NOT NULL
                      AND revision.stage_root IS NOT NULL
                      AND NOT EXISTS (
                        SELECT 1 FROM graph_runs AS child
                        WHERE child.parent_operation_id = revision.operation_id
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM graph_run_receipts AS receipt
                        WHERE receipt.operation_id = revision.operation_id
                          AND receipt.category = 'experiment_recovery_abandoned'
                      )
                    )
                  )
                LIMIT 1
                """,
                (record.project_id, record.chat_id, record.view_id),
            ).fetchone()
        return row is not None

    def result_view_descriptor(
        self,
        record: ResultViewRecord,
        *,
        as_of: datetime | None = None,
    ) -> ResultViewDescriptor:
        """Project private storage metadata onto the path-free public contract."""
        record = ResultViewRecord.model_validate(record)
        is_temporary = record.kept_filename is None
        return ResultViewDescriptor(
            view_id=record.view_id,
            chat_id=record.chat_id,
            experiment_id=record.experiment_id,
            name=record.source_name,
            media_type="text/html",
            state="temporary" if is_temporary else "kept",
            created_at=record.created_at,
            updated_at=record.updated_at,
            expires_at=record.expires_at,
            kept_filename=record.kept_filename,
            kept_at=record.kept_at,
            can_revise=is_temporary and _result_view_is_visible(record, as_of=as_of),
        )

    def list_result_view_descriptors(
        self,
        project_id: str,
        *,
        experiment_id: str | None = None,
        chat_id: str | None = None,
        as_of: datetime | None = None,
    ) -> list[ResultViewDescriptor]:
        return [
            self.result_view_descriptor(record, as_of=as_of)
            for record in self.list_result_views(
                project_id,
                experiment_id=experiment_id,
                chat_id=chat_id,
                as_of=as_of,
            )
        ]

    def revise_result_view(
        self,
        view_id: str,
        *,
        expected_content_sha256: str,
        latest_operation_id: str,
        content_sha256: str,
        size_bytes: int,
        html: bytes,
        updated_at: str,
        expires_at: str,
    ) -> ResultViewRecord:
        """CAS one revision onto the same stable view identity."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM result_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
            if row is None:
                raise KeyError(view_id)
            current = self._result_view_record(row)
            if current.kept_filename is not None:
                raise ResultViewConflict("a kept result view cannot be revised")
            if current.content_sha256 != expected_content_sha256:
                raise ResultViewConflict("result view changed before this revision was recorded")
            revised = ResultViewRecord.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "latest_operation_id": latest_operation_id,
                    "content_sha256": content_sha256,
                    "size_bytes": size_bytes,
                    "updated_at": updated_at,
                    "expires_at": expires_at,
                }
            )
            stored_html = _validated_result_view_html(revised, html)
            updated = connection.execute(
                """
                UPDATE result_views
                SET latest_operation_id = ?, content_sha256 = ?, size_bytes = ?, html = ?,
                    updated_at = ?, expires_at = ?
                WHERE view_id = ? AND content_sha256 = ? AND kept_filename IS NULL
                """,
                (
                    revised.latest_operation_id,
                    revised.content_sha256,
                    revised.size_bytes,
                    stored_html,
                    revised.updated_at,
                    revised.expires_at,
                    view_id,
                    expected_content_sha256,
                ),
            ).rowcount
            if updated != 1:
                raise ResultViewConflict("result view changed before this revision was recorded")
        return revised

    def refresh_result_view_expiry(
        self,
        project_id: str,
        chat_id: str,
        *,
        expires_at: str,
        as_of: datetime | None = None,
    ) -> int:
        """Extend active unkept view retention without reviving expired views."""
        requested_expiry = _required_timestamp(expires_at)
        current = as_of or datetime.now(UTC)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("result view refresh time must include a timezone")
        current = current.astimezone(UTC)
        refreshed = 0
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT view_id, expires_at FROM result_views
                WHERE project_id = ? AND chat_id = ? AND kept_filename IS NULL
                """,
                (project_id, chat_id),
            ).fetchall()
            for row in rows:
                current_expiry = _required_timestamp(row["expires_at"])
                if current_expiry <= current or requested_expiry <= current_expiry:
                    continue
                refreshed += connection.execute(
                    """
                    UPDATE result_views SET expires_at = ?
                    WHERE view_id = ? AND expires_at = ? AND kept_filename IS NULL
                    """,
                    (expires_at, row["view_id"], row["expires_at"]),
                ).rowcount
        return refreshed

    def mark_result_view_kept(
        self,
        view_id: str,
        *,
        expected_content_sha256: str,
        kept_filename: str,
        kept_at: str,
    ) -> ResultViewRecord:
        """Remember Keep once, bound to the exact bytes that were copied."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM result_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
            if row is None:
                raise KeyError(view_id)
            current = self._result_view_record(row)
            if current.kept_filename is not None:
                if current.content_sha256 != expected_content_sha256:
                    raise ResultViewConflict("result view changed before Keep was recorded")
                return current
            if current.content_sha256 != expected_content_sha256:
                raise ResultViewConflict("result view changed before Keep was recorded")
            kept = ResultViewRecord.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "kept_filename": kept_filename,
                    "kept_at": kept_at,
                }
            )
            updated = connection.execute(
                """
                UPDATE result_views
                SET kept_filename = ?, kept_at = ?
                WHERE view_id = ? AND kept_filename IS NULL AND content_sha256 = ?
                """,
                (kept.kept_filename, kept.kept_at, view_id, expected_content_sha256),
            ).rowcount
            if updated != 1:
                raise ResultViewConflict("result view changed before Keep was recorded")
        return kept
