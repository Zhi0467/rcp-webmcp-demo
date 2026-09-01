from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pydantic import (
    TypeAdapter,
)

from rcp.core.models import (
    AuthorizedHuman,
)
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.limits import (
    WATCHER_GROUP_DIAGNOSTIC_ERROR_COUNT,
)
from rcp.storage.experiments import ExperimentStoreMixin
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
    AutoResearchActorBusy,
    AutoResearchRole,
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
from rcp.storage.rows import RowMappingMixin

if TYPE_CHECKING:
    from rcp.watchers import WatcherBinding


class WatcherStoreMixin:
    """External and graph watchers, their claims, and notification delivery."""

    @staticmethod
    def detach_watchers_for_restore(
        connection: sqlite3.Connection,
        *,
        diagnostic: str,
        confirmed_by: str,
        now: str,
    ) -> None:
        """Stop every captured watcher that could still check or deliver."""

        if not connection.in_transaction:
            raise ValueError("restored watcher detachment requires an active transaction")
        detail = " ".join(diagnostic.split())[:1500]
        confirmer = " ".join(confirmed_by.split())[:400]
        if not detail or not confirmer:
            raise ValueError("restored watcher detachment requires a reason and confirmer")
        _required_timestamp(now)
        stop_reason = f"{detail} Restore confirmed by {confirmer}."[:2000]
        connection.execute(
            """
            UPDATE watchers
            SET status = 'stopped', notified = 1, next_check_at = NULL,
                stopped_by = 'human', stop_reason = ?, stopped_at = COALESCE(stopped_at, ?)
            WHERE status IN ('active', 'degraded')
               OR (status = 'completed' AND notified = 0)
            """,
            (stop_reason, now),
        )

    def create_watchers(self, records: list[StoredWatcherRecord]) -> list[StoredWatcherRecord]:
        """Insert one validated watch list atomically."""

        records = [self._prepare_watcher_for_insert(record) for record in records]
        self._validate_watch_list(records)
        watcher_ids = [record.watcher_id for record in records]
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for record in records:
                self._insert_watcher(connection, record)
        stored: list[StoredWatcherRecord] = []
        for watcher_id in watcher_ids:
            record = self.watcher(watcher_id)
            assert record is not None
            stored.append(record)
        return stored

    def _validate_and_apply_agent_watcher_stops(
        self,
        connection: sqlite3.Connection,
        binding: WatcherBinding,
        stops: list[WatcherStopRequest],
        episode_row: sqlite3.Row | None,
        *,
        apply: bool = True,
    ) -> None:
        """Retire only staged compatible observers under the arming transaction."""

        continuation = binding.continuation
        episode_id = continuation.control_episode_id
        control_node_id = continuation.control_node_id
        if episode_row is None or not episode_id or not control_node_id:
            raise ValueError("An agent watcher stop requires the current Experiment episode.")
        episode = self._experiment_episode_record(episode_row)
        root_request = self._experiment_episode_root_request(
            connection,
            binding.project_id,
            control_node_id,
            episode_id,
        )
        if root_request is None:
            raise ValueError("An agent watcher stop requires the bound Experiment root task.")
        ids = [item.stop_watcher_id for item in stops]
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT * FROM watchers WHERE watcher_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {str(row["watcher_id"]): self._watcher_record(row) for row in rows}
        for stop in stops:
            record = by_id.get(stop.stop_watcher_id)
            if record is None:
                raise ValueError(
                    f"Watcher stop names an unknown staged watcher: {stop.stop_watcher_id}"
                )
            if isinstance(record, GraphWatcherRecord):
                raise ValueError(
                    "Experiment agent watcher stops may retire only external observers: "
                    f"{stop.stop_watcher_id}"
                )
            if record.status == "stopped":
                if (
                    record.stopped_by == "agent"
                    and record.stop_operation_id == binding.origin_operation_id
                    and record.stop_reason == stop.reason
                ):
                    continue
                # Stop loop retires this episode's watchers while its authorized turn is
                # still running. That turn's retirement is already satisfied, so it
                # finishes normally instead of correcting a race it cannot win.
                if episode.stop_requested_at is not None:
                    continue
                raise ValueError(f"Watcher stop was already resolved: {stop.stop_watcher_id}")
            if record.notified or record.notification_operation_id is not None:
                raise WatcherClaimConflict("A watcher update was already claimed for delivery.")
            if (
                record.project_id != binding.project_id
                or record.node_id != control_node_id
                or not self._experiment_watcher_matches_current(record, root_request, episode)
            ):
                raise ValueError(
                    f"Watcher stop is outside the bound Experiment episode: {stop.stop_watcher_id}"
                )
            if record.status not in {"active", "degraded", "completed"}:
                raise ValueError(f"Watcher cannot be retired: {stop.stop_watcher_id}")

        if not apply:
            return
        timestamp = self.now()
        for stop in stops:
            record = by_id[stop.stop_watcher_id]
            if record.status == "stopped":
                continue
            cursor = connection.execute(
                """
                UPDATE watchers
                SET status = 'stopped', notified = 1, next_check_at = NULL,
                    stopped_by = 'agent', stop_reason = ?, stopped_at = ?, stop_operation_id = ?
                WHERE watcher_id = ? AND status IN ('active', 'degraded', 'completed')
                  AND notified = 0 AND notification_operation_id IS NULL
                """,
                (stop.reason, timestamp, binding.origin_operation_id, stop.stop_watcher_id),
            )
            if cursor.rowcount != 1:
                raise WatcherClaimConflict("A watcher update changed during its retirement claim.")

    def _stop_watcher_for_loop(self, connection: sqlite3.Connection, watcher_id: str) -> None:
        timestamp = self.now()
        connection.execute(
            """
            UPDATE watchers
            SET status = 'stopped', notified = 1, next_check_at = NULL,
                stopped_by = COALESCE(stopped_by, 'loop'),
                stopped_at = COALESCE(stopped_at, ?)
            WHERE watcher_id = ?
            """,
            (timestamp, watcher_id),
        )

    @staticmethod
    def _prepare_watcher_for_insert(record: StoredWatcherRecord) -> StoredWatcherRecord:
        continuation = record.continuation
        if continuation.patch_kind == "experiment_loop":
            episode_id = record.episode_id or continuation.control_episode_id
            if episode_id != record.episode_id:
                record = record.model_copy(update={"episode_id": episode_id})
        if isinstance(record, GraphWatcherRecord):
            return record
        if record.status not in {"active", "degraded"} or record.next_check_at is not None:
            return record
        error_count = record.consecutive_error_count
        if record.status == "degraded" and error_count == 0:
            error_count = 1
        checked_at = record.last_checked_at or record.created_at
        return record.model_copy(
            update={
                "consecutive_error_count": error_count,
                "next_check_at": watcher_next_check_at(
                    record.watcher_id,
                    checked_at,
                    error_count,
                ),
            }
        )

    @staticmethod
    def _validate_watch_list(records: list[StoredWatcherRecord]) -> None:
        if not records:
            raise ValueError("a watch list must contain at least one watcher")
        watcher_ids = [record.watcher_id for record in records]
        if len(watcher_ids) != len(set(watcher_ids)):
            raise ValueError("a watch list cannot repeat a watcher id")
        bindings = {
            (
                record.project_id,
                record.origin_operation_id,
                record.origin_task_kind,
                record.graph_target.key,
                record.chat_id,
                record.node_id,
                record.episode_id,
                record.execution_host,
                record.continuation.model_dump_json(),
            )
            for record in records
        }
        if len(bindings) != 1:
            raise ValueError("one watch list must share one RCP-bound continuation context")
        continuation = records[0].continuation
        if any(
            isinstance(record, GraphWatcherRecord) and record.status == "degraded"
            for record in records
        ):
            raise ValueError("a graph condition cannot have a degraded shell-check state")
        if any(
            isinstance(record, GraphWatcherRecord) and record.armed_revision is None
            for record in records
        ):
            raise ValueError("a new graph condition requires its canonical arming revision")
        grouped = [record for record in records if record.group_id is not None]
        if any(isinstance(record, GraphWatcherRecord) for record in grouped):
            raise ValueError("graph conditions cannot join an external watcher group")
        if any((record.group_id is None) != (record.group_label is None) for record in records):
            raise ValueError("watcher group identity and label must be stored together")
        if grouped and continuation.patch_kind != "experiment_loop":
            raise ValueError("only Experiment-loop watchers may join a watcher group")
        if grouped:
            group_counts: dict[str, int] = {}
            for record in grouped:
                assert record.group_id is not None
                group_counts[record.group_id] = group_counts.get(record.group_id, 0) + 1
            if any(count < 2 for count in group_counts.values()):
                raise ValueError("an Experiment watcher group requires at least two observers")
        if continuation.patch_kind != "experiment_loop":
            return
        if not all(
            (
                continuation.control_node_id,
                continuation.control_episode_id,
                continuation.control_invocation,
                continuation.control_invocation_ceiling,
            )
        ):
            raise ValueError("an experiment-loop watcher must preserve its control binding")
        assert continuation.control_invocation is not None
        assert continuation.control_invocation_ceiling is not None
        if continuation.control_invocation > continuation.control_invocation_ceiling:
            raise ValueError("an experiment-loop watcher invocation exceeds its pinned ceiling")
        if any(record.episode_id != continuation.control_episode_id for record in records):
            raise ValueError("an Experiment watcher must bind explicitly to its control episode")

    @staticmethod
    def _validate_idempotent_watcher(
        existing: StoredWatcherRecord,
        desired: StoredWatcherRecord,
    ) -> None:
        if type(existing) is not type(desired):
            raise ValueError("Experiment-loop watcher identity conflicts with stored state.")
        immutable_fields = [
            "project_id",
            "origin_operation_id",
            "origin_task_kind",
            "graph_target",
            "chat_id",
            "node_id",
            "episode_id",
            "execution_host",
            "continuation",
            "group_id",
            "group_label",
        ]
        if isinstance(existing, WatcherRecord):
            immutable_fields.extend(("check_command", "log_path", "cwd"))
        else:
            immutable_fields.append("condition")
        if any(getattr(existing, field) != getattr(desired, field) for field in immutable_fields):
            raise ValueError("Experiment-loop watcher identity conflicts with stored state.")

    @staticmethod
    def _insert_watcher(connection: sqlite3.Connection, record: StoredWatcherRecord) -> None:
        origin = connection.execute(
            "SELECT project_id, episode_id, graph_target_json FROM graph_runs "
            "WHERE operation_id = ?",
            (record.origin_operation_id,),
        ).fetchone()
        if origin is None:
            if record.graph_target.kind == "branch":
                raise ValueError("a branch watcher requires its durable origin task")
        elif (
            origin["project_id"] != record.project_id
            or (origin["episode_id"] is not None and origin["episode_id"] != record.episode_id)
            or json.loads(origin["graph_target_json"])
            != record.graph_target.model_dump(mode="json")
        ):
            raise ValueError("a watcher cannot change its origin task graph binding")
        stopped_episode = connection.execute(
            """
            SELECT COALESCE(
                       episode.stop_requested_at,
                       episode.ended_at,
                       episode.updated_at
                   ) AS stop_requested_at
            FROM graph_runs AS run
            JOIN episodes AS episode ON episode.episode_id = run.episode_id
            WHERE run.operation_id = ?
              AND (
                  episode.stop_requested_at IS NOT NULL
                  OR episode.ending IS NOT NULL
              )
            """,
            (record.origin_operation_id,),
        ).fetchone()
        if stopped_episode is not None and record.status != "stopped":
            record = record.model_copy(
                update={
                    "status": "stopped",
                    "notified": True,
                    "next_check_at": None,
                    "stopped_by": "loop",
                    "stopped_at": stopped_episode["stop_requested_at"],
                }
            )
        if isinstance(record, GraphWatcherRecord):
            consumed = connection.execute(
                """
                SELECT revision FROM graph_watcher_reconciliation
                WHERE project_id = ? AND graph_target_key = ?
                """,
                (record.project_id, record.graph_target.key),
            ).fetchone()
            if (
                record.status != "stopped"
                and record.armed_revision is not None
                and consumed is not None
                and record.armed_revision < int(consumed["revision"])
            ):
                raise ValueError("a graph watcher cannot arm behind the consumed target boundary")
            # Legacy watcher tables keep these external-only columns NOT NULL.
            # The separate GraphWatcherRecord never exposes the compatibility
            # placeholders; graph_condition_json selects its stored type.
            check_command = ""
            log_path = ""
            cwd = ""
            graph_condition_json = record.condition.model_dump_json()
            armed_revision = record.armed_revision
        else:
            check_command = record.check_command
            log_path = record.log_path
            cwd = record.cwd
            graph_condition_json = None
            armed_revision = None
        connection.execute(
            """
            INSERT INTO watchers (
                watcher_id, project_id, origin_operation_id, origin_task_kind,
                chat_id, node_id, episode_id, graph_target_json, execution_host,
                check_command, log_path, cwd, graph_condition_json, armed_revision,
                continuation_json, status, created_at, last_checked_at,
                last_exit_code, last_error, completed_at, next_check_at,
                consecutive_error_count, group_id, group_label, notified,
                notification_operation_id, stopped_by, stop_reason, stopped_at,
                stop_operation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.watcher_id,
                record.project_id,
                record.origin_operation_id,
                record.origin_task_kind,
                record.chat_id,
                record.node_id,
                record.episode_id,
                record.graph_target.model_dump_json(),
                record.execution_host,
                check_command,
                log_path,
                cwd,
                graph_condition_json,
                armed_revision,
                record.continuation.model_dump_json(),
                record.status,
                record.created_at,
                record.last_checked_at,
                record.last_exit_code,
                record.last_error,
                record.completed_at,
                record.next_check_at,
                record.consecutive_error_count,
                record.group_id,
                record.group_label,
                int(record.notified),
                record.notification_operation_id,
                record.stopped_by,
                record.stop_reason,
                record.stopped_at,
                record.stop_operation_id,
            ),
        )

    def watcher(self, watcher_id: str) -> StoredWatcherRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM watchers WHERE watcher_id = ?", (watcher_id,)
            ).fetchone()
        return self._watcher_record(row) if row is not None else None

    def watchers(
        self,
        project_id: str,
        *,
        chat_id: str | None = None,
    ) -> list[StoredWatcherRecord]:
        query = "SELECT * FROM watchers WHERE project_id = ?"
        parameters: list[object] = [project_id]
        if chat_id is not None:
            query += " AND chat_id = ?"
            parameters.append(chat_id)
        query += " ORDER BY created_at DESC, watcher_id"
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._watcher_record(row) for row in rows]

    def active_graph_watchers(self, project_id: str) -> list[GraphWatcherRecord]:
        """Return graph conditions awaiting a canonical revision boundary."""

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM watchers
                WHERE project_id = ? AND graph_condition_json IS NOT NULL
                  AND status = 'active' AND notified = 0
                ORDER BY created_at, watcher_id
                """,
                (project_id,),
            ).fetchall()
        records = [self._watcher_record(row) for row in rows]
        if any(not isinstance(record, GraphWatcherRecord) for record in records):
            raise RuntimeError("External watcher row appeared in the graph-condition index.")
        return records  # type: ignore[return-value]

    def graph_watcher_project_ids(self) -> list[str]:
        """Return projects needing startup graph evaluation or delivery retry."""

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT project_id FROM watchers
                WHERE graph_condition_json IS NOT NULL
                  AND status IN ('active', 'completed') AND notified = 0
                ORDER BY project_id
                """
            ).fetchall()
        return [str(row["project_id"]) for row in rows]

    def graph_watcher_reconciliation_head(
        self,
        project_id: str,
        graph_target: GraphTargetRef,
    ) -> GraphHeadRef | None:
        """Return the last graph boundary atomically consumed for one target."""

        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT graph_target_json, revision, transition_id
                FROM graph_watcher_reconciliation
                WHERE project_id = ? AND graph_target_key = ?
                """,
                (project_id, graph_target.key),
            ).fetchone()
        if row is None:
            return None
        try:
            stored_target = GraphTargetRef.model_validate_json(row["graph_target_json"])
            head = GraphHeadRef(
                target=stored_target,
                revision=row["revision"],
                transition_id=row["transition_id"],
            )
        except ValueError as exc:
            raise RuntimeError("Stored graph-watcher reconciliation head is invalid.") from exc
        if stored_target != graph_target:
            raise RuntimeError("Stored graph-watcher target key does not match its target.")
        return head

    def initialize_graph_watcher_target_baselines(
        self,
        project_id: str,
        graph_target: GraphTargetRef,
        *,
        armed_revision: int,
        evaluated_at: str | None = None,
    ) -> None:
        """Give every migrated target-local graph watcher one current baseline."""

        timestamp = evaluated_at or self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM watchers
                WHERE project_id = ? AND graph_condition_json IS NOT NULL
                  AND armed_revision IS NULL AND status = 'active' AND notified = 0
                ORDER BY created_at, watcher_id
                """,
                (project_id,),
            ).fetchall()
            ids = [
                record.watcher_id
                for row in rows
                if isinstance((record := self._watcher_record(row)), GraphWatcherRecord)
                and record.graph_target == graph_target
            ]
            if not ids:
                return
            placeholders = ",".join("?" for _ in ids)
            connection.execute(
                f"""
                UPDATE watchers SET armed_revision = ?, last_checked_at = ?
                WHERE watcher_id IN ({placeholders})
                  AND graph_condition_json IS NOT NULL AND armed_revision IS NULL
                  AND status = 'active' AND notified = 0
                """,
                (armed_revision, timestamp, *ids),
            )

    def consume_graph_watcher_boundary(
        self,
        project_id: str,
        head: GraphHeadRef,
        *,
        evaluate: Callable[[GraphWatcherRecord], Literal["active", "completed", "removed"]],
        evaluated_at: str | None = None,
    ) -> bool:
        """Evaluate and checkpoint one accepted target boundary atomically.

        The target-local head is the durable receipt. A repeated or older
        boundary performs no watcher writes; a same-revision identity mismatch
        fails closed. ``transition_id=None`` is the stable identity for an
        accepted historical Patch that predates transition traces. If evaluation
        raises, SQLite rolls back both watcher changes and the head so a restart
        can retry the exact boundary.
        """

        timestamp = evaluated_at or self.now()
        target_json = json.dumps(
            head.target.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor_row = connection.execute(
                """
                SELECT graph_target_json, revision, transition_id
                FROM graph_watcher_reconciliation
                WHERE project_id = ? AND graph_target_key = ?
                """,
                (project_id, head.target.key),
            ).fetchone()
            if cursor_row is not None:
                try:
                    stored_target = GraphTargetRef.model_validate_json(
                        cursor_row["graph_target_json"]
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "Stored graph-watcher reconciliation target is invalid."
                    ) from exc
                if stored_target != head.target:
                    raise RuntimeError("Stored graph-watcher target key does not match its target.")
                stored_revision = int(cursor_row["revision"])
                if stored_revision > head.revision:
                    return False
                if stored_revision == head.revision:
                    if cursor_row["transition_id"] != head.transition_id:
                        raise RuntimeError(
                            "Accepted graph boundary changed identity at a consumed revision."
                        )
                    return False

            rows = connection.execute(
                """
                SELECT * FROM watchers
                WHERE project_id = ? AND graph_condition_json IS NOT NULL
                  AND status = 'active' AND notified = 0
                ORDER BY created_at, watcher_id
                """,
                (project_id,),
            ).fetchall()
            for row in rows:
                record = self._watcher_record(row)
                if not isinstance(record, GraphWatcherRecord):
                    raise RuntimeError(
                        "External watcher row appeared in the graph-condition index."
                    )
                if record.graph_target != head.target or record.armed_revision is None:
                    continue
                if record.armed_revision >= head.revision:
                    continue
                self._record_graph_watcher_result_locked(
                    connection,
                    record,
                    result=evaluate(record),
                    evaluated_at=timestamp,
                )

            if cursor_row is None:
                connection.execute(
                    """
                    INSERT INTO graph_watcher_reconciliation(
                        project_id, graph_target_key, graph_target_json,
                        revision, transition_id, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        head.target.key,
                        target_json,
                        head.revision,
                        head.transition_id,
                        timestamp,
                    ),
                )
            else:
                previous_revision = int(cursor_row["revision"])
                updated = connection.execute(
                    """
                    UPDATE graph_watcher_reconciliation
                    SET graph_target_json = ?, revision = ?, transition_id = ?, updated_at = ?
                    WHERE project_id = ? AND graph_target_key = ? AND revision = ?
                    """,
                    (
                        target_json,
                        head.revision,
                        head.transition_id,
                        timestamp,
                        project_id,
                        head.target.key,
                        previous_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("Graph-watcher reconciliation head changed concurrently.")
        return True

    @staticmethod
    def _record_graph_watcher_result_locked(
        connection: sqlite3.Connection,
        current: GraphWatcherRecord,
        *,
        result: Literal["active", "completed", "removed"],
        evaluated_at: str,
    ) -> None:
        if result == "active":
            connection.execute(
                """
                UPDATE watchers SET last_checked_at = ?
                WHERE watcher_id = ? AND graph_condition_json IS NOT NULL
                  AND status = 'active' AND notified = 0
                """,
                (evaluated_at, current.watcher_id),
            )
        elif result == "completed":
            connection.execute(
                """
                UPDATE watchers
                SET status = 'completed', last_checked_at = ?, completed_at = ?,
                    next_check_at = NULL
                WHERE watcher_id = ? AND graph_condition_json IS NOT NULL
                  AND status = 'active' AND notified = 0
                """,
                (evaluated_at, evaluated_at, current.watcher_id),
            )
        else:
            connection.execute(
                """
                UPDATE watchers
                SET status = 'stopped', notified = 1, last_checked_at = ?,
                    next_check_at = NULL, stopped_by = 'loop',
                    stop_reason = 'Graph condition target was removed.', stopped_at = ?
                WHERE watcher_id = ? AND graph_condition_json IS NOT NULL
                  AND status = 'active' AND notified = 0
                """,
                (evaluated_at, evaluated_at, current.watcher_id),
            )

    def record_graph_watcher_result(
        self,
        watcher_id: str,
        *,
        result: Literal["active", "completed", "removed"],
        evaluated_at: str | None = None,
    ) -> GraphWatcherRecord:
        """Persist one canonical graph evaluation without entering the shell poller."""

        timestamp = evaluated_at or self.now()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM watchers WHERE watcher_id = ?", (watcher_id,)
            ).fetchone()
            if row is None:
                raise KeyError(watcher_id)
            current = self._watcher_record(row)
            if not isinstance(current, GraphWatcherRecord):
                raise ValueError("an external watcher cannot receive a graph evaluation")
            if current.status != "active" or current.notified:
                return current
            self._record_graph_watcher_result_locked(
                connection,
                current,
                result=result,
                evaluated_at=timestamp,
            )
        stored = self.watcher(watcher_id)
        assert isinstance(stored, GraphWatcherRecord)
        return stored

    def initialize_graph_watcher_baseline(
        self,
        watcher_id: str,
        *,
        armed_revision: int,
        evaluated_at: str | None = None,
    ) -> GraphWatcherRecord:
        """Fail closed while giving one pre-baseline graph row a durable boundary."""

        timestamp = evaluated_at or self.now()
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE watchers SET armed_revision = ?, last_checked_at = ?
                WHERE watcher_id = ? AND graph_condition_json IS NOT NULL
                  AND armed_revision IS NULL AND status = 'active' AND notified = 0
                """,
                (armed_revision, timestamp, watcher_id),
            )
        stored = self.watcher(watcher_id)
        assert isinstance(stored, GraphWatcherRecord)
        return stored

    def pollable_watchers(self, *, as_of: str | None = None) -> list[WatcherRecord]:
        """Return only active/degraded observers whose durable due time arrived."""

        now = as_of or self.now()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM watchers
                WHERE status IN ('active', 'degraded')
                  AND notified = 0
                  AND graph_condition_json IS NULL
                  AND (next_check_at IS NULL OR next_check_at <= ?)
                ORDER BY created_at, watcher_id
                """,
                (now,),
            ).fetchall()
            records = [self._watcher_record(row) for row in rows]
            if any(not isinstance(record, WatcherRecord) for record in records):
                raise RuntimeError("Graph conditions cannot enter the external watcher poller.")
            stopping_contexts: dict[
                tuple[str, str, str],
                tuple[dict[str, object], ExperimentEpisodeRecord] | None,
            ] = {}
            return [
                record
                for record in records
                if not self._watcher_suppressed_by_current_stop(
                    connection,
                    record,
                    stopping_contexts,
                )
            ]

    def stop_watchers(self, project_id: str, watcher_ids: list[str]) -> list[StoredWatcherRecord]:
        """Release watchers the human has given up on.

        A stopped watcher leaves the polling set and can never wake a turn. RCP
        never decides this for itself — a check that cannot answer is reported,
        not interpreted.
        """

        ids = list(dict.fromkeys(watcher_ids))
        if not ids:
            raise ValueError("stopping watchers requires at least one watcher id")
        placeholders = ",".join("?" for _ in ids)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""
                SELECT watcher_id, project_id, status, notified, notification_operation_id
                FROM watchers
                WHERE watcher_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
            if {str(row["watcher_id"]) for row in rows} != set(ids) or {
                str(row["project_id"]) for row in rows
            } != {project_id}:
                missing = next(
                    (
                        watcher_id
                        for watcher_id in ids
                        if watcher_id not in {str(row["watcher_id"]) for row in rows}
                    ),
                    ids[0],
                )
                raise KeyError(missing)
            if any(row["notification_operation_id"] is not None for row in rows):
                raise WatcherClaimConflict("A watcher update was already claimed for delivery.")
            invalid = [
                str(row["watcher_id"])
                for row in rows
                if row["status"] not in {"active", "degraded", "completed", "stopped"}
                or (bool(row["notified"]) and row["status"] != "stopped")
            ]
            if invalid:
                raise ValueError(f"Watchers cannot be stopped: {', '.join(sorted(invalid))}.")
            connection.execute(
                f"""
                UPDATE watchers
                SET status = 'stopped', notified = 1, next_check_at = NULL,
                    stopped_by = COALESCE(stopped_by, 'human'),
                    stopped_at = COALESCE(stopped_at, ?)
                WHERE project_id = ? AND watcher_id IN ({placeholders})
                  AND status IN ('active', 'degraded', 'completed')
                  AND notification_operation_id IS NULL
                """,
                (self.now(), project_id, *ids),
            )
        stopped: list[StoredWatcherRecord] = []
        for watcher_id in ids:
            record = self.watcher(watcher_id)
            assert record is not None
            stopped.append(record)
        return stopped

    @classmethod
    def _watcher_suppressed_by_current_stop(
        cls,
        connection: sqlite3.Connection,
        record: StoredWatcherRecord,
        cache: dict[
            tuple[str, str, str],
            tuple[dict[str, object], ExperimentEpisodeRecord] | None,
        ],
    ) -> bool:
        continuation = record.continuation
        control_node_id = continuation.control_node_id
        if continuation.patch_kind != "experiment_loop" or not control_node_id:
            return False
        key = (record.project_id, control_node_id, record.graph_target.key)
        if key not in cache:
            root = connection.execute(
                """
                SELECT request_json FROM graph_runs
                WHERE project_id = ? AND parent_operation_id IS NULL
                  AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
                  AND json_extract(request_json, '$.control_node_id') = ?
                  AND json_extract(graph_target_json, '$.kind') = ?
                  AND json_extract(graph_target_json, '$.branch_id') IS ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (
                    record.project_id,
                    control_node_id,
                    record.graph_target.kind,
                    record.graph_target.branch_id,
                ),
            ).fetchone()
            context = None
            if root is not None:
                root_request = json.loads(root["request_json"])
                episode_id = root_request.get("control_episode_id")
                episode_row = (
                    cls._experiment_episode_row(connection, episode_id)
                    if isinstance(episode_id, str)
                    else None
                )
                if episode_row is not None and episode_row["stop_requested_at"] is not None:
                    context = (root_request, cls._experiment_episode_record(episode_row))
            cache[key] = context
        context = cache[key]
        return context is not None and cls._experiment_watcher_matches_current(
            record,
            context[0],
            context[1],
        )

    def record_watcher_check(
        self,
        watcher_id: str,
        *,
        status: WatcherStatus,
        exit_code: int | None,
        error: str | None,
        checked_at: str | None = None,
    ) -> WatcherRecord:
        if status == "degraded" and not error:
            raise ValueError("a degraded watcher requires a check error")
        if status != "degraded":
            error = None
        timestamp = checked_at or self.now()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM watchers WHERE watcher_id = ?", (watcher_id,)
            ).fetchone()
            if row is None:
                raise KeyError(watcher_id)
            current = self._watcher_record(row)
            if not isinstance(current, WatcherRecord):
                raise ValueError("a graph condition cannot receive a shell check result")
            if current.status not in {"active", "degraded"} or current.notified:
                return current
            consecutive_error_count = (
                current.consecutive_error_count + 1 if status == "degraded" else 0
            )
            next_check_at = (
                watcher_next_check_at(watcher_id, timestamp, consecutive_error_count)
                if status in {"active", "degraded"}
                else None
            )
            cursor = connection.execute(
                """
                UPDATE watchers
                SET status = ?, last_checked_at = ?, last_exit_code = ?, last_error = ?,
                    next_check_at = ?, consecutive_error_count = ?,
                    completed_at = CASE
                        WHEN ? = 'completed' THEN COALESCE(completed_at, ?)
                        ELSE completed_at
                    END
                WHERE watcher_id = ? AND status IN ('active', 'degraded') AND notified = 0
                """,
                (
                    status,
                    timestamp,
                    exit_code,
                    error,
                    next_check_at,
                    consecutive_error_count,
                    status,
                    timestamp,
                    watcher_id,
                ),
            )
            if cursor.rowcount == 0:
                return self._watcher_record(
                    connection.execute(
                        "SELECT * FROM watchers WHERE watcher_id = ?", (watcher_id,)
                    ).fetchone()
                )
        stored = self.watcher(watcher_id)
        assert isinstance(stored, WatcherRecord)
        return stored

    def completed_watcher_groups(self) -> list[list[StoredWatcherRecord]]:
        """Return compatible ready delivery units without splitting Experiment groups."""

        with self.connection() as connection:
            units = self._ready_watcher_delivery_units(connection)
        groups: dict[tuple[object, ...], list[StoredWatcherRecord]] = {}
        for unit in units:
            first = unit[0]
            key = (
                (
                    first.project_id,
                    first.graph_target.key,
                    "experiment_loop",
                    first.node_id,
                    first.execution_host,
                    self._automatic_watcher_delivery_policy(first.continuation),
                )
                if first.continuation.patch_kind == "experiment_loop"
                else (
                    first.project_id,
                    first.graph_target.key,
                    first.origin_task_kind,
                    first.chat_id,
                    first.node_id,
                    first.execution_host,
                    self._automatic_watcher_delivery_policy(first.continuation),
                )
            )
            groups.setdefault(key, []).extend(unit)
        return list(groups.values())

    def _ready_watcher_delivery_units(
        self,
        connection: sqlite3.Connection,
    ) -> list[list[StoredWatcherRecord]]:
        """Build indivisible ready groups plus ordinary completed observer units."""

        ungrouped_rows = connection.execute(
            """
            SELECT * FROM watchers
            WHERE group_id IS NULL AND status = 'completed' AND notified = 0
            ORDER BY completed_at, created_at, watcher_id
            """
        ).fetchall()
        grouped_rows = connection.execute(
            """
            SELECT * FROM watchers
            WHERE group_id IN (
                SELECT DISTINCT group_id FROM watchers
                WHERE group_id IS NOT NULL AND notified = 0
                  AND (
                    status = 'completed'
                    OR (
                        status = 'degraded'
                        AND consecutive_error_count >= ?
                    )
                  )
            )
            ORDER BY completed_at, created_at, watcher_id
            """,
            (WATCHER_GROUP_DIAGNOSTIC_ERROR_COUNT,),
        ).fetchall()
        ungrouped = [self._watcher_record(row) for row in ungrouped_rows]
        grouped_records = [self._watcher_record(row) for row in grouped_rows]
        stopping_contexts: dict[
            tuple[str, str, str], tuple[dict[str, object], ExperimentEpisodeRecord] | None
        ] = {}
        units: list[list[StoredWatcherRecord]] = []
        grouped: dict[str, list[StoredWatcherRecord]] = {}
        for record in ungrouped:
            if self._watcher_suppressed_by_current_stop(connection, record, stopping_contexts):
                continue
            units.append([record])
        for record in grouped_records:
            assert record.group_id is not None
            grouped.setdefault(record.group_id, []).append(record)
        for members in grouped.values():
            ready = self._ready_group_members(members)
            if not ready:
                continue
            if self._watcher_suppressed_by_current_stop(connection, ready[0], stopping_contexts):
                continue
            units.append(ready)
        return units

    def create_watcher_notification_task(
        self,
        record: AgentTaskRecord,
        watcher_ids: list[str],
        *,
        continuation_cause: str = "fresh",
    ) -> AgentTaskRecord | None:
        """Queue a wake and mark its completed watchers notified in one transaction.

        A live task in the same conversation wins its slot. In that case no
        watcher row changes, and the completed group can be retried later.
        """

        ids = list(dict.fromkeys(watcher_ids))
        if not ids or len(ids) != len(watcher_ids):
            raise ValueError("a watcher notification requires unique watcher ids")
        if record.status != "queued":
            raise ValueError("a watcher notification task must be queued")
        requested_ids = record.request.get("watcher_ids")
        if (
            not isinstance(requested_ids, list)
            or any(not isinstance(item, str) for item in requested_ids)
            or len(requested_ids) != len(set(requested_ids))
            or set(requested_ids) != set(ids)
        ):
            raise ValueError("the watcher notification request must name exactly its watcher ids")
        placeholders = ",".join("?" for _ in ids)
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    f"""
                    SELECT *
                    FROM watchers
                    WHERE watcher_id IN ({placeholders})
                        AND status IN ('completed', 'degraded') AND notified = 0
                    """,
                    ids,
                ).fetchall()
                if {str(row["watcher_id"]) for row in rows} != set(ids):
                    raise ValueError("watchers are missing, unready, or already notified")
                watchers = [self._watcher_record(row) for row in rows]
                self._validate_watcher_notification_members(connection, watchers)
                if {item.project_id for item in watchers} != {record.project_id}:
                    raise ValueError("watchers and notification task belong to different projects")
                bindings = {
                    (
                        (
                            "experiment_loop",
                            item.node_id,
                            item.graph_target.key,
                            item.execution_host,
                            self._automatic_watcher_delivery_policy(item.continuation),
                        )
                        if item.continuation.patch_kind == "experiment_loop"
                        else (
                            item.origin_task_kind,
                            item.chat_id,
                            item.node_id,
                            item.execution_host,
                            self._watcher_delivery_policy(item.continuation),
                        )
                    )
                    for item in watchers
                }
                if len(bindings) != 1:
                    raise ValueError("one notification cannot merge incompatible watch lists")
                self._validate_watcher_notification_scope(connection, record, watchers)
                if self._experiment_wake_is_stopped(connection, record):
                    return None
                if self._has_active_chat_overlap(connection, record):
                    return None
                if record.kind == "auto_research":
                    episode_id = record.request.get("episode_id")
                    if not isinstance(episode_id, str) or episode_id != record.episode_id:
                        raise ValueError("Auto-research watcher wake has invalid episode lineage")
                    episode = self._load_auto_research_episode(connection, episode_id)
                    role = TypeAdapter(AutoResearchRole).validate_python(record.request.get("role"))
                    self._insert_paid_auto_research_task(
                        connection,
                        episode,
                        record,
                        role,
                        continuation_cause=continuation_cause,
                    )
                else:
                    self._insert_agent_task(
                        connection,
                        record,
                        continuation_cause=continuation_cause,
                    )
                cursor = connection.execute(
                    f"""
                    UPDATE watchers
                    SET notified = 1, notification_operation_id = ?
                    WHERE watcher_id IN ({placeholders})
                        AND status IN ('completed', 'degraded') AND notified = 0
                    """,
                    [record.operation_id, *ids],
                )
                if cursor.rowcount != len(ids):
                    raise RuntimeError("watcher notification changed during its transaction")
        except AutoResearchActorBusy:
            return None
        except sqlite3.IntegrityError as exc:
            raise ValueError("Could not queue the watcher notification task.") from exc
        stored = self.agent_task(record.operation_id)
        assert stored is not None
        return stored

    def resolve_watcher_delivery_authorizer(
        self,
        watcher_ids: list[str],
    ) -> tuple[AuthorizedHuman | None, str | None]:
        """Resolve one automatic wake's human authority or terminalize it.

        Legacy tasks have no trustworthy authorizer to inherit. Missing tasks,
        partial snapshots, and a delivery unit assembled from different humans
        are equally non-recoverable without a new human action. Consume those
        completed watchers with a durable, UI-visible diagnostic so the poller
        cannot retry an unauthorized wake forever.

        The resolution and terminal transition share the same write transaction
        as the watcher readiness check. A concurrent notification claim or Stop
        therefore wins cleanly instead of producing both a wake and a terminal
        diagnostic.
        """

        ids = list(dict.fromkeys(watcher_ids))
        if not ids or len(ids) != len(watcher_ids):
            raise ValueError("watcher delivery authorization requires unique watcher ids")
        placeholders = ",".join("?" for _ in ids)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""
                SELECT * FROM watchers
                WHERE watcher_id IN ({placeholders})
                  AND status IN ('completed', 'degraded')
                  AND notified = 0 AND notification_operation_id IS NULL
                """,
                ids,
            ).fetchall()
            if {str(row["watcher_id"]) for row in rows} != set(ids):
                return None, None
            watchers = [self._watcher_record(row) for row in rows]
            self._validate_watcher_notification_members(connection, watchers)

            origin_ids = sorted({item.origin_operation_id for item in watchers})
            origin_placeholders = ",".join("?" for _ in origin_ids)
            origin_rows = connection.execute(
                f"""
                SELECT operation_id, authorized_space_id, authorized_user_id,
                       authorized_display_name
                FROM graph_runs
                WHERE operation_id IN ({origin_placeholders})
                """,
                origin_ids,
            ).fetchall()
            by_operation = {str(row["operation_id"]): row for row in origin_rows}

            diagnostic: str | None = None
            if set(by_operation) != set(origin_ids):
                diagnostic = (
                    "Automatic watcher wake stopped: an originating task is unavailable, so "
                    "RCP cannot prove who authorized the wake. Start a new Work turn or "
                    "Experiment Run to continue."
                )
            else:
                try:
                    authorizers = [
                        self._authorized_human_snapshot(by_operation[operation_id])
                        for operation_id in origin_ids
                    ]
                except RuntimeError:
                    diagnostic = (
                        "Automatic watcher wake stopped: an originating task has an invalid "
                        "human authorizer snapshot, so RCP cannot prove who authorized the "
                        "wake. Start a new Work turn or Experiment Run to continue."
                    )
                else:
                    if any(authorizer is None for authorizer in authorizers):
                        diagnostic = (
                            "Automatic watcher wake stopped: an originating task predates "
                            "durable human attribution, so RCP cannot prove who authorized the "
                            "wake. Start a new Work turn or Experiment Run to continue."
                        )
                    else:
                        authorized_by = authorizers[0]
                        assert authorized_by is not None
                        if any(authorizer != authorized_by for authorizer in authorizers[1:]):
                            diagnostic = (
                                "Automatic watcher wake stopped: the originating tasks have "
                                "different human authorizers, so RCP cannot choose one. Start a "
                                "new Work turn or Experiment Run to continue."
                            )
                        else:
                            return authorized_by, None

            assert diagnostic is not None
            timestamp = self.now()
            cursor = connection.execute(
                f"""
                UPDATE watchers
                SET status = 'stopped', notified = 1, next_check_at = NULL,
                    stop_reason = ?, stopped_at = COALESCE(stopped_at, ?)
                WHERE watcher_id IN ({placeholders})
                  AND status IN ('completed', 'degraded')
                  AND notified = 0 AND notification_operation_id IS NULL
                """,
                [diagnostic, timestamp, *ids],
            )
            if cursor.rowcount != len(ids):
                raise RuntimeError(
                    "Watcher delivery changed during its authorizer terminalization."
                )

            episode_ids = sorted(
                {
                    item.episode_id
                    for item in watchers
                    if item.continuation.patch_kind == "experiment_loop"
                    and item.episode_id is not None
                }
            )
            if episode_ids:
                episode_placeholders = ",".join("?" for _ in episode_ids)
                connection.execute(
                    f"""
                    UPDATE experiment_episode_state
                    SET session_diagnostic = ?, updated_at = ?
                    WHERE episode_id IN ({episode_placeholders})
                    """,
                    [diagnostic, timestamp, *episode_ids],
                )
            return None, diagnostic

    @staticmethod
    def _watcher_delivery_policy(continuation: WatcherContinuation) -> str:
        policy = continuation.model_dump(mode="json")
        if continuation.patch_kind == "experiment_loop" and policy.get("model") is None:
            # Legacy Experiment watchers stored the provider-default sentinel
            # as null. It is immutable policy, equivalent to today's "".
            policy["model"] = ""
        for field in (
            "control_revision",
            "control_episode_id",
            "control_invocation",
            "control_invocation_ceiling",
            "control_decision_bundle",
            "control_completion_criteria",
        ):
            policy.pop(field, None)
        return json.dumps(policy, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _automatic_watcher_delivery_policy(continuation: WatcherContinuation) -> str:
        """Policy key for poller-driven delivery; generic Work stays unchanged."""

        if continuation.patch_kind != "experiment_loop":
            return WatcherStoreMixin._watcher_delivery_policy(continuation)
        policy = {
            "patch_kind": continuation.patch_kind,
            "control_node_id": continuation.control_node_id,
        }
        return json.dumps(policy, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _validate_watcher_notification_scope(
        connection: sqlite3.Connection,
        record: AgentTaskRecord,
        watchers: list[StoredWatcherRecord],
    ) -> None:
        first = watchers[0]
        continuation = first.continuation
        request = record.request
        trigger = request.get("trigger")
        if {item.graph_target.key for item in watchers} != {record.graph_target.key}:
            raise ValueError("watcher notification changed its exact graph target")
        auto_research_wake = first.origin_task_kind == "auto_research"
        if auto_research_wake:
            if record.kind != "auto_research" or record.episode_id is None:
                raise ValueError("Auto-research watchers must wake an Auto-research task")
            actor_bindings: set[tuple[object, ...]] = set()
            for watcher in watchers:
                origin = connection.execute(
                    """
                    SELECT run.episode_id, run.request_json, invocation.role
                    FROM graph_runs AS run
                    JOIN auto_research_invocations AS invocation
                      ON invocation.operation_id = run.operation_id
                    WHERE run.operation_id = ?
                    """,
                    (watcher.origin_operation_id,),
                ).fetchone()
                if origin is None or origin["episode_id"] != record.episode_id:
                    raise ValueError("Auto-research watcher origin is outside the episode")
                origin_request = json.loads(origin["request_json"])
                actor_bindings.add(
                    (
                        origin["episode_id"],
                        origin_request.get("actor_operation_id") or watcher.origin_operation_id,
                        origin["role"],
                        origin_request.get("control_node_id"),
                    )
                )
            if len(actor_bindings) != 1:
                raise ValueError("one Auto-research watcher wake cannot merge different actors")
            expected_episode, expected_actor, expected_role, expected_seat = next(
                iter(actor_bindings)
            )
            actual = (
                request.get("episode_id"),
                request.get("actor_operation_id"),
                request.get("role"),
                request.get("control_node_id"),
            )
            if actual != (expected_episode, expected_actor, expected_role, expected_seat):
                raise ValueError("Auto-research watcher wake changed its actor binding")
        elif continuation.patch_kind == "experiment_loop":
            if (
                record.kind != "node_chat"
                or request.get("node_id") != continuation.control_node_id
                or not isinstance(request.get("chat_id"), str)
                or not request.get("chat_id")
            ):
                raise ValueError("Experiment watcher delivery must target its node chat.")
        else:
            expected = {
                "kind": first.origin_task_kind,
                "chat_id": first.chat_id,
                "node_id": first.node_id,
            }
            actual = {
                "kind": record.kind,
                "chat_id": request.get("chat_id"),
                "node_id": request.get("node_id"),
            }
            mismatched = sorted(key for key, value in expected.items() if actual[key] != value)
            if mismatched:
                raise ValueError(
                    f"watcher notification changed immutable scope: {', '.join(mismatched)}"
                )
        request_continuation_data = {
            key: request[key] for key in WatcherContinuation.model_fields if key in request
        }
        for nullable_list in ("workflow_ids", "skill_ids", "resolved_skill_packages"):
            if request_continuation_data.get(nullable_list) is None:
                request_continuation_data[nullable_list] = []
        request_continuation = WatcherContinuation.model_validate(request_continuation_data)
        request_policy = (
            WatcherStoreMixin._automatic_watcher_delivery_policy(request_continuation)
            if continuation.patch_kind == "experiment_loop"
            else WatcherStoreMixin._watcher_delivery_policy(request_continuation)
        )
        continuation_policy = (
            WatcherStoreMixin._automatic_watcher_delivery_policy(continuation)
            if continuation.patch_kind == "experiment_loop"
            else WatcherStoreMixin._watcher_delivery_policy(continuation)
        )
        if request_policy != continuation_policy:
            raise ValueError("watcher notification changed its immutable delivery policy")
        if auto_research_wake:
            graph_wake = all(isinstance(item, GraphWatcherRecord) for item in watchers)
            expected_cause = "graph_condition" if graph_wake else "watcher"
            if request.get("wake_cause") != expected_cause:
                raise ValueError("Auto-research watcher wake changed its continuation cause")
            return
        if continuation.patch_kind != "experiment_loop":
            if trigger != "watcher":
                raise ValueError("a generic watcher notification must use the watcher trigger")
            return
        invocation = request.get("control_invocation")
        episode_id = request.get("control_episode_id")
        if trigger == "watcher":
            if not isinstance(invocation, int) or invocation < 2:
                raise ValueError("an automatic Experiment wake must continue an existing episode")
            newest = connection.execute(
                """
                SELECT kind, request_json, graph_target_json FROM graph_runs
                WHERE project_id = ? AND parent_operation_id IS NULL
                  AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
                  AND json_extract(request_json, '$.control_node_id') = ?
                  AND json_extract(graph_target_json, '$.kind') = ?
                  AND json_extract(graph_target_json, '$.branch_id') IS ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (
                    record.project_id,
                    continuation.control_node_id,
                    record.graph_target.kind,
                    record.graph_target.branch_id,
                ),
            ).fetchone()
            newest_request = json.loads(newest["request_json"]) if newest is not None else None
            if newest_request is None or newest_request.get("control_episode_id") != episode_id:
                raise ValueError("an automatic Experiment wake must use the newest episode")
            episode_row = ExperimentStoreMixin._experiment_episode_row(
                connection,
                episode_id,
            )
            episode = (
                RowMappingMixin._experiment_episode_record(episode_row)
                if episode_row is not None
                else None
            )
            if episode is not None and (
                record.kind != newest["kind"] or request.get("chat_id") != episode.chat_id
            ):
                raise ValueError("Experiment watcher delivery changed its episode wake target.")
            if any(
                not ExperimentStoreMixin._experiment_watcher_matches_current(
                    item, newest_request, episode
                )
                for item in watchers
            ):
                raise ValueError(
                    "completed watchers are incompatible with the current Experiment episode"
                )
            return
        if trigger != "experiment_run" or invocation != 1:
            raise ValueError("a human Experiment watcher claim must start a new episode")
        previous = connection.execute(
            """
            SELECT request_json FROM graph_runs
            WHERE project_id = ? AND parent_operation_id IS NULL
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(request_json, '$.control_node_id') = ?
              AND json_extract(graph_target_json, '$.kind') = ?
              AND json_extract(graph_target_json, '$.branch_id') IS ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (
                record.project_id,
                continuation.control_node_id,
                record.graph_target.kind,
                record.graph_target.branch_id,
            ),
        ).fetchone()
        if previous is None:
            raise ValueError("a human watcher claim requires a prior Experiment episode")
        previous_request = json.loads(previous["request_json"])
        if previous_request.get("control_episode_id") == episode_id:
            raise ValueError("a human watcher claim must authorize a fresh episode")
