from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING

from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.limits import AGENT_TASK_RECEIPT_MAX_BYTES
from rcp.storage.episodes import _LIVE_EPISODE_STATUSES
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
    AutoResearchChildExperimentRecord,
    ChatSessionContextRecord,
    EpisodeBudgetMeter,
    EpisodeInvocationCeilingReached,
    EpisodeNotRunning,
    EpisodeRecord,
    ExperimentControlProjectionSnapshot,
    ExperimentEpisodeProjectionSnapshot,
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

if TYPE_CHECKING:
    from rcp.watchers import WatcherBinding


class ExperimentStoreMixin:
    """Bounded Experiment episodes and their loop runtime projection."""

    @staticmethod
    def _experiment_episode_row(
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT state.*, episode.project_id, episode.control_node_id,
                   episode.graph_target_json,
                   episode.stop_requested_at, episode.stop_settled_at
            FROM experiment_episode_state AS state
            JOIN episodes AS episode ON episode.episode_id = state.episode_id
            WHERE state.episode_id = ? AND episode.mode = 'experiment_loop'
            """,
            (episode_id,),
        ).fetchone()

    def create_experiment_episode_with_invocation(
        self,
        record: AgentTaskRecord,
        watcher_ids: list[str] | None = None,
        *,
        auto_research_route: AutoResearchChildExperimentRecord | None = None,
        auto_research_admission_id: str | None = None,
    ) -> AgentTaskRecord:
        """Atomically create the Experiment parent, mode child, and invocation 1."""

        ids = list(watcher_ids or [])
        parent_episode = (
            self.episode(auto_research_route.auto_research_episode_id)
            if auto_research_route is not None
            else None
        )
        if auto_research_route is not None and (
            parent_episode is None
            or parent_episode.mode != "auto_research"
            or record.graph_target != parent_episode.graph_target
        ):
            raise ValueError("an Auto-research child Experiment changed its parent graph target")
        episode = self._new_experiment_episode(
            record,
            auto_research_route=auto_research_route,
            graph_base_head=(parent_episode.graph_base_head if parent_episode else None),
        )
        self._validate_new_episode(episode)
        self._validate_experiment_watcher_ids(record, ids)
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if (
                    auto_research_route is None
                    and record.request.get("trigger") == "experiment_run"
                ):
                    self._require_project_accepts_new_work(connection, record.project_id)
                if (
                    connection.execute(
                        "SELECT 1 FROM episodes WHERE episode_id = ?", (episode.episode_id,)
                    ).fetchone()
                    is not None
                ):
                    raise ValueError("This Experiment episode id is already in use.")
                if self._live_episode_row(connection, episode) is not None:
                    raise ValueError("This Experiment already has a live episode.")
                watchers = self._ready_experiment_watchers(connection, record, ids)
                if ids:
                    self._validate_watcher_notification_scope(connection, record, watchers)
                started = episode.model_copy(
                    update={
                        "root_operation_id": record.operation_id,
                        "status": "running",
                        "invocations_used": 1,
                        "updated_at": record.created_at,
                    }
                )
                if auto_research_route is not None:
                    if (
                        auto_research_route.child_episode_id != episode.episode_id
                        or auto_research_route.project_id != episode.project_id
                        or auto_research_route.control_node_id != episode.control_node_id
                    ):
                        raise ValueError(
                            "the Auto-research route does not match the Experiment episode"
                        )
                    self._activate_auto_research_child_experiment(
                        connection,
                        auto_research_route,
                        admission_id=auto_research_admission_id,
                    )
                    self._claim_auto_research_experiment_allowance(
                        connection,
                        auto_research_episode_id=auto_research_route.auto_research_episode_id,
                        child_episode_id=episode.episode_id,
                        operation_id=record.operation_id,
                        created_at=record.created_at,
                    )
                self._insert_episode(connection, started)
                connection.execute(
                    """
                    INSERT INTO experiment_episode_state (episode_id, created_at, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (episode.episode_id, episode.created_at, record.created_at),
                )
                self._insert_agent_task(connection, record, continuation_cause="fresh")
                connection.execute(
                    """
                    INSERT INTO episode_invocations (
                        episode_id, operation_id, invocation_number, created_at
                    ) VALUES (?, ?, 1, ?)
                    """,
                    (episode.episode_id, record.operation_id, record.created_at),
                )
                self._claim_experiment_watchers(connection, record.operation_id, ids)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Could not create the Experiment episode.") from exc
        stored = self.agent_task(record.operation_id)
        assert stored is not None
        return stored

    def create_experiment_watcher_invocation(
        self,
        record: AgentTaskRecord,
        watcher_ids: list[str],
        *,
        auto_research_episode_id: str | None = None,
    ) -> AgentTaskRecord | None:
        """Claim one ready watcher unit and its next paid invocation atomically."""

        ids = list(watcher_ids)
        if (
            record.status != "queued"
            or not record.visible
            or record.parent_operation_id is not None
            or record.request.get("patch_kind") != "experiment_loop"
            or record.request.get("trigger") != "watcher"
            or record.request.get("control_episode_id") != record.episode_id
        ):
            raise ValueError("An Experiment watcher wake must be a queued paid root task.")
        self._validate_experiment_watcher_ids(record, ids)
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                watchers = self._ready_experiment_watchers(connection, record, ids)
                episode_id = record.episode_id
                assert episode_id is not None
                episode_row = connection.execute(
                    "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
                ).fetchone()
                if episode_row is None:
                    raise KeyError(episode_id)
                episode = self._episode_record(episode_row)
                if (
                    episode.mode != "experiment_loop"
                    or episode.project_id != record.project_id
                    or episode.control_node_id != record.request.get("control_node_id")
                    or episode.graph_target != record.graph_target
                ):
                    raise ValueError("The watcher wake belongs to another episode.")
                if (
                    episode.status != "running"
                    or episode.ending is not None
                    or episode.stop_requested_at is not None
                    or self._experiment_has_ending_receipt(connection, episode_id)
                ):
                    return None
                if episode.invocations_used >= episode.invocation_ceiling:
                    raise EpisodeInvocationCeilingReached(
                        "the episode has spent its operational invocation ceiling"
                    )
                expected = episode.invocations_used + 1
                if record.request.get("control_invocation") != expected:
                    raise ValueError(
                        f"Experiment-loop invocation is out of sequence; expected {expected}."
                    )
                self._validate_experiment_watcher_wake_scope(
                    connection,
                    record,
                    watchers,
                )
                if self._has_active_chat_overlap(connection, record):
                    return None
                route = connection.execute(
                    """
                    SELECT auto_research_episode_id
                    FROM auto_research_child_experiments
                    WHERE child_episode_id = ? AND state = 'running'
                    """,
                    (episode_id,),
                ).fetchone()
                routed_parent_id = (
                    str(route["auto_research_episode_id"]) if route is not None else None
                )
                if (
                    auto_research_episode_id is not None
                    and routed_parent_id != auto_research_episode_id
                ):
                    raise ValueError(
                        "the Experiment watcher wake does not match its Auto-research route"
                    )
                if routed_parent_id is not None:
                    self._claim_auto_research_experiment_allowance(
                        connection,
                        auto_research_episode_id=routed_parent_id,
                        child_episode_id=episode_id,
                        operation_id=record.operation_id,
                        created_at=record.created_at,
                    )
                self._insert_agent_task(connection, record, continuation_cause="watcher_wake")
                connection.execute(
                    """
                    INSERT INTO episode_invocations (
                        episode_id, operation_id, invocation_number, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (episode_id, record.operation_id, expected, record.created_at),
                )
                connection.execute(
                    """
                    UPDATE episodes
                    SET invocations_used = ?, updated_at = ?
                    WHERE episode_id = ?
                    """,
                    (expected, self.now(), episode_id),
                )
                self._claim_experiment_watchers(connection, record.operation_id, ids)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Could not queue the Experiment watcher wake.") from exc
        stored = self.agent_task(record.operation_id)
        assert stored is not None
        return stored

    def create_experiment_recovery_task(
        self,
        record: AgentTaskRecord,
        *,
        continuation_cause: str = "resume",
    ) -> AgentTaskRecord:
        """Create a recovery/repair child without spending another paid invocation."""

        if (
            record.status != "queued"
            or not record.visible
            or record.parent_operation_id is None
            or record.episode_id is None
        ):
            raise ValueError("An Experiment recovery requires its parent and episode lineage.")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert_experiment_recovery_task(
                connection,
                record,
                continuation_cause=continuation_cause,
            )
        stored = self.agent_task(record.operation_id)
        assert stored is not None
        return stored

    def create_experiment_graph_repair_task(
        self,
        parent_operation_id: str,
        record: AgentTaskRecord,
    ) -> AgentTaskRecord:
        """Atomically claim an Experiment graph repair and admit its child task."""

        if (
            record.status != "queued"
            or not record.visible
            or record.parent_operation_id != parent_operation_id
            or record.episode_id is None
            or record.request.get("patch_kind") != "experiment_loop"
            or record.request.get("message") is not None
        ):
            raise ValueError("An Experiment graph repair requires a queued patch-only child.")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._claim_agent_task_graph_repair(connection, parent_operation_id)
            self._insert_experiment_recovery_task(
                connection,
                record,
                continuation_cause="graph_repair",
            )
            stored = connection.execute(
                "SELECT * FROM graph_runs WHERE operation_id = ?", (record.operation_id,)
            ).fetchone()
            assert stored is not None
        return self._agent_task_record(stored)

    def _insert_experiment_recovery_task(
        self,
        connection: sqlite3.Connection,
        record: AgentTaskRecord,
        *,
        continuation_cause: str,
    ) -> None:
        """Insert one validated Experiment recovery task in an open transaction."""

        episode_id = record.episode_id
        assert episode_id is not None
        episode_row = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if episode_row is None:
            raise KeyError(episode_id)
        episode = self._episode_record(episode_row)
        if (
            episode.mode != "experiment_loop"
            or episode.project_id != record.project_id
            or episode.graph_target != record.graph_target
            or episode.control_node_id != record.request.get("control_node_id")
        ):
            raise ValueError("The recovery task belongs to another episode.")
        route = connection.execute(
            """
            SELECT auto_research_episode_id, state
            FROM auto_research_child_experiments
            WHERE child_episode_id = ?
            """,
            (episode_id,),
        ).fetchone()
        if route is not None:
            if route["state"] != "running":
                raise EpisodeNotRunning(
                    "the routed child Experiment is no longer accepting recovery work"
                )
            parent = self._load_auto_research_episode(
                connection,
                str(route["auto_research_episode_id"]),
            )
            self._validate_auto_research_parent_admission(parent)
        if (
            episode.status not in {"running", "stopping"}
            or episode.ending is not None
            or self._experiment_has_ending_receipt(connection, episode_id)
        ):
            raise EpisodeNotRunning("the episode is not admitting recovery work")
        if not self._experiment_recovery_has_paid_root(
            connection,
            episode_id,
            record.parent_operation_id,
        ):
            raise ValueError("The Experiment recovery has no paid invocation ancestor.")
        if self._has_active_chat_overlap(connection, record):
            raise ValueError("Another task is already active in this conversation.")
        self._insert_agent_task(
            connection,
            record,
            continuation_cause=continuation_cause,
        )

    @staticmethod
    def _new_experiment_episode(
        record: AgentTaskRecord,
        *,
        auto_research_route: AutoResearchChildExperimentRecord | None = None,
        graph_base_head: GraphHeadRef | None = None,
    ) -> EpisodeRecord:
        request = record.request
        episode_id = request.get("control_episode_id")
        control_node_id = request.get("control_node_id")
        ceiling = request.get("control_invocation_ceiling")
        expected_trigger = "orchestrator" if auto_research_route is not None else "experiment_run"
        if (
            record.status != "queued"
            or not record.visible
            or record.kind != "node_chat"
            or record.parent_operation_id is not None
            or request.get("patch_kind") != "experiment_loop"
            or request.get("trigger") != expected_trigger
            or request.get("control_invocation") != 1
            or not isinstance(episode_id, str)
            or record.episode_id != episode_id
            or not isinstance(control_node_id, str)
            or not control_node_id
            or request.get("node_id") != control_node_id
            or not isinstance(ceiling, int)
            or isinstance(ceiling, bool)
            or ceiling < 1
        ):
            raise ValueError("An Experiment Run must be its episode's queued invocation 1.")
        episode = EpisodeRecord(
            episode_id=episode_id,
            project_id=record.project_id,
            mode="experiment_loop",
            control_node_id=control_node_id,
            graph_target=record.graph_target,
            graph_base_head=graph_base_head,
            status="queued",
            invocation_ceiling=ceiling,
            authorized_by=record.authorized_by,
            created_at=record.created_at,
            updated_at=record.created_at,
        )
        ExperimentStoreMixin._validate_new_experiment_episode(episode)
        return episode

    @staticmethod
    def _validate_new_experiment_episode(episode: EpisodeRecord) -> None:
        if episode.authorized_by is None:
            raise ValueError("A new Experiment episode requires its human authorization snapshot.")

    @staticmethod
    def _validate_experiment_watcher_ids(
        record: AgentTaskRecord,
        watcher_ids: list[str],
    ) -> None:
        if len(watcher_ids) != len(set(watcher_ids)):
            raise ValueError("an Experiment watcher claim requires unique watcher ids")
        requested = record.request.get("watcher_ids")
        if (
            not isinstance(requested, list)
            or any(not isinstance(item, str) for item in requested)
            or set(requested) != set(watcher_ids)
            or len(requested) != len(watcher_ids)
        ):
            raise ValueError("the Experiment task must name exactly its watcher ids")

    def _ready_experiment_watchers(
        self,
        connection: sqlite3.Connection,
        record: AgentTaskRecord,
        watcher_ids: list[str],
    ) -> list[StoredWatcherRecord]:
        if not watcher_ids:
            return []
        placeholders = ",".join("?" for _ in watcher_ids)
        rows = connection.execute(
            f"""
            SELECT * FROM watchers
            WHERE watcher_id IN ({placeholders})
              AND status IN ('completed', 'degraded') AND notified = 0
            """,
            watcher_ids,
        ).fetchall()
        if {str(row["watcher_id"]) for row in rows} != set(watcher_ids):
            raise ValueError("watchers are missing, unready, or already notified")
        watchers = [self._watcher_record(row) for row in rows]
        self._validate_watcher_notification_members(connection, watchers)
        if {item.project_id for item in watchers} != {record.project_id}:
            raise ValueError("watchers and Experiment task belong to different projects")
        if {item.graph_target.key for item in watchers} != {record.graph_target.key}:
            raise ValueError("watchers and Experiment task belong to different graph targets")
        bindings = {
            (
                item.node_id,
                item.graph_target.key,
                item.execution_host,
                self._automatic_watcher_delivery_policy(item.continuation),
            )
            for item in watchers
            if item.continuation.patch_kind == "experiment_loop"
        }
        if len(bindings) != 1 or len(watchers) != len(
            [item for item in watchers if item.continuation.patch_kind == "experiment_loop"]
        ):
            raise ValueError("one Experiment claim cannot merge incompatible watch lists")
        return watchers

    def _validate_experiment_watcher_wake_scope(
        self,
        connection: sqlite3.Connection,
        record: AgentTaskRecord,
        watchers: list[StoredWatcherRecord],
    ) -> None:
        if not watchers:
            raise ValueError("an automatic Experiment wake requires its completed watchers")
        request = record.request
        first = watchers[0]
        continuation = first.continuation
        if (
            request.get("trigger") != "watcher"
            or record.kind != "node_chat"
            or request.get("node_id") != continuation.control_node_id
            or request.get("control_episode_id") != record.episode_id
            or not isinstance(request.get("chat_id"), str)
            or not request.get("chat_id")
        ):
            raise ValueError("Experiment watcher delivery changed its episode scope.")
        request_continuation = WatcherContinuation.model_validate(
            {
                key: (
                    []
                    if value is None
                    and key in {"workflow_ids", "skill_ids", "resolved_skill_packages"}
                    else value
                )
                for key in WatcherContinuation.model_fields
                if (value := request.get(key)) is not None
                or key in {"workflow_ids", "skill_ids", "resolved_skill_packages"}
            }
        )
        if self._automatic_watcher_delivery_policy(
            request_continuation
        ) != self._automatic_watcher_delivery_policy(continuation):
            raise ValueError("Experiment watcher delivery changed its immutable policy.")
        state_row = self._experiment_episode_row(connection, record.episode_id or "")
        if state_row is None:
            raise ValueError("The Experiment watcher wake has no episode state.")
        state = self._experiment_episode_record(state_row)
        if state.graph_target != record.graph_target:
            raise ValueError("The Experiment watcher wake changed its exact graph target.")
        newest = connection.execute(
            """
            SELECT kind, request_json FROM graph_runs
            WHERE project_id = ? AND parent_operation_id IS NULL
              AND episode_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT 1
            """,
            (record.project_id, record.episode_id),
        ).fetchone()
        newest_request = json.loads(newest["request_json"]) if newest is not None else None
        if (
            newest_request is None
            or newest_request.get("control_node_id") != continuation.control_node_id
            or newest_request.get("control_episode_id") != record.episode_id
            or newest["kind"] != record.kind
            or request.get("chat_id") != state.chat_id
            or any(
                not self._experiment_watcher_matches_current(item, newest_request, state)
                for item in watchers
            )
        ):
            raise ValueError("completed watchers do not match the current Experiment episode")

    @staticmethod
    def _claim_experiment_watchers(
        connection: sqlite3.Connection,
        operation_id: str,
        watcher_ids: list[str],
    ) -> None:
        if not watcher_ids:
            return
        placeholders = ",".join("?" for _ in watcher_ids)
        cursor = connection.execute(
            f"""
            UPDATE watchers
            SET notified = 1, notification_operation_id = ?
            WHERE watcher_id IN ({placeholders})
              AND status IN ('completed', 'degraded') AND notified = 0
            """,
            [operation_id, *watcher_ids],
        )
        if cursor.rowcount != len(watcher_ids):
            raise RuntimeError("Experiment watcher claim changed during its transaction")

    @staticmethod
    def _experiment_recovery_has_paid_root(
        connection: sqlite3.Connection,
        episode_id: str,
        operation_id: str,
    ) -> bool:
        current_id: str | None = operation_id
        seen: set[str] = set()
        while current_id is not None and current_id not in seen:
            seen.add(current_id)
            if (
                connection.execute(
                    """
                SELECT 1 FROM episode_invocations
                WHERE episode_id = ? AND operation_id = ?
                """,
                    (episode_id, current_id),
                ).fetchone()
                is not None
            ):
                return True
            row = connection.execute(
                "SELECT parent_operation_id FROM graph_runs WHERE operation_id = ?",
                (current_id,),
            ).fetchone()
            current_id = (
                str(row["parent_operation_id"]) if row and row["parent_operation_id"] else None
            )
        return False

    @staticmethod
    def _experiment_has_ending_receipt(
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> bool:
        return (
            connection.execute(
                """
                SELECT 1 FROM graph_run_receipts AS receipt
                JOIN graph_runs AS run ON run.operation_id = receipt.operation_id
                WHERE run.episode_id = ? AND receipt.category = 'experiment_loop_exit'
                LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _validate_experiment_task_insert(
        connection: sqlite3.Connection,
        record: AgentTaskRecord,
    ) -> None:
        request = record.request
        if request.get("patch_kind") != "experiment_loop":
            return

        recovery_binding_keys = (*_EXPERIMENT_EPISODE_PINNED_FIELDS, "control_invocation")
        node_id = request.get("control_node_id")
        control_revision = request.get("control_revision")
        episode_id = request.get("control_episode_id")
        invocation = request.get("control_invocation")
        ceiling = request.get("control_invocation_ceiling")
        decision_bundle = request.get("control_decision_bundle")
        completion_criteria = request.get("control_completion_criteria")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("A bounded experiment-loop task must name its control node.")
        if not isinstance(control_revision, int) or isinstance(control_revision, bool):
            raise ValueError("A bounded experiment-loop task must pin its control revision.")
        if not isinstance(decision_bundle, list):
            raise ValueError("A bounded experiment-loop task must pin its governing decisions.")
        if not isinstance(completion_criteria, list) or any(
            not isinstance(item, str) for item in completion_criteria
        ):
            raise ValueError("A bounded experiment-loop task must pin its completion criteria.")
        if not isinstance(episode_id, str):
            raise ValueError("A bounded experiment-loop task must name a valid episode id.")
        try:
            uuid.UUID(episode_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "A bounded experiment-loop task must name a valid episode id."
            ) from exc
        if record.episode_id != episode_id:
            raise ValueError("A bounded experiment-loop task must use its episode parent lineage.")
        if not isinstance(invocation, int) or isinstance(invocation, bool) or invocation < 1:
            raise ValueError("A bounded experiment-loop task must name its invocation number.")
        if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 1:
            raise ValueError("A bounded experiment-loop task must pin its invocation ceiling.")
        if invocation > ceiling:
            raise ValueError("The experiment-loop invocation exceeds its pinned ceiling.")

        if record.parent_operation_id:
            parent = connection.execute(
                """
                SELECT project_id, kind, status, attempt, request_json, result_json
                FROM graph_runs WHERE operation_id = ?
                """,
                (record.parent_operation_id,),
            ).fetchone()
            if parent is None:
                raise ValueError("An experiment-loop recovery task must have its parent task.")
            if parent["project_id"] != record.project_id or parent["kind"] != record.kind:
                raise ValueError("An experiment-loop recovery task must preserve its task scope.")
            parent_request = json.loads(parent["request_json"])
            if any(
                _experiment_pinned_value(parent_request, key)
                != _experiment_pinned_value(request, key)
                for key in recovery_binding_keys
            ):
                raise ValueError(
                    "An experiment-loop recovery task must preserve its control binding and "
                    "pinned configuration."
                )
            parent_result = json.loads(parent["result_json"]) if parent["result_json"] else None
            graph_update = (
                parent_result.get("graph_update") if isinstance(parent_result, dict) else None
            )
            patch_only_repair = (
                request.get("message") is None
                and parent["status"] == "succeeded"
                and isinstance(graph_update, dict)
                and graph_update.get("status") == "rejected"
                and graph_update.get("repairable") is False
            )
            if not patch_only_repair:
                ExperimentStoreMixin._validate_experiment_recovery_claim(
                    connection,
                    record,
                    parent,
                    parent_request,
                )
            else:
                ExperimentStoreMixin._validate_current_experiment_graph_repair(
                    connection,
                    project_id=record.project_id,
                    control_node_id=node_id,
                    episode_id=episode_id,
                    invocation=invocation,
                    operation_id=record.parent_operation_id,
                )
            return

        trigger = request.get("trigger")
        if trigger not in {"experiment_run", "orchestrator", "watcher"}:
            raise ValueError(
                "A root experiment-loop task must be a Run, orchestrator, or watcher invocation."
            )
        if trigger == "orchestrator":
            route = connection.execute(
                """
                SELECT 1 FROM auto_research_child_experiments
                WHERE child_episode_id = ? AND project_id = ? AND control_node_id = ?
                  AND state = 'running'
                """,
                (episode_id, record.project_id, node_id),
            ).fetchone()
            if route is None:
                raise ValueError(
                    "An orchestrator Experiment start requires its active Auto-research route."
                )
        rows = connection.execute(
            """
            SELECT request_json FROM graph_runs
            WHERE project_id = ? AND parent_operation_id IS NULL
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(request_json, '$.control_node_id') = ?
              AND json_extract(request_json, '$.control_episode_id') = ?
            """,
            (record.project_id, node_id, episode_id),
        ).fetchall()
        prior = [json.loads(row["request_json"]) for row in rows]
        if any(
            _experiment_pinned_value(item, key) != _experiment_pinned_value(request, key)
            for item in prior
            for key in _EXPERIMENT_EPISODE_PINNED_FIELDS
        ):
            raise ValueError("An experiment-loop episode cannot change its pinned configuration.")
        expected = max((int(item["control_invocation"]) for item in prior), default=0) + 1
        if invocation != expected:
            raise ValueError(
                f"Experiment-loop invocation {invocation} is out of sequence; expected {expected}."
            )
        if invocation == 1 and prior:
            raise ValueError("An experiment-loop episode may have only one first invocation.")
        if trigger in {"experiment_run", "orchestrator"} and invocation != 1:
            raise ValueError("A Run or orchestrator start must be experiment-loop invocation 1.")
        if trigger == "watcher" and not prior:
            raise ValueError("An automatic watcher wake requires an existing loop episode.")
        if trigger == "watcher":
            ExperimentStoreMixin._validate_experiment_wake_binding(connection, record)

    @staticmethod
    def _validate_experiment_wake_binding(
        connection: sqlite3.Connection,
        record: AgentTaskRecord,
    ) -> None:
        """Prove the saved native session before an automatic wake spends budget."""

        request = record.request
        episode_id = request.get("control_episode_id")
        session_id = request.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("An automatic Experiment wake requires its episode session id.")
        if record.native_session_id != session_id or not record.stage_root:
            raise ValueError(
                "An automatic Experiment wake requires its exact saved session and stage."
            )
        episode = ExperimentStoreMixin._experiment_episode_row(connection, str(episode_id))
        if episode is None or episode["stop_requested_at"] is not None:
            raise ValueError("The automatic Experiment wake has no active episode binding.")
        binding_task = connection.execute(
            "SELECT request_json FROM graph_runs WHERE operation_id = ?",
            (episode["last_turn_operation_id"],),
        ).fetchone()
        if binding_task is None:
            raise ValueError("The automatic Experiment wake has no active binding task.")
        binding_request = json.loads(binding_task["request_json"])
        expected = {
            "project_id": record.project_id,
            "control_node_id": request.get("control_node_id"),
            "provider": request.get("provider"),
            "execution_machine": request.get("run_on"),
            "native_session_id": session_id,
            "stage_host": record.stage_host or "",
            "stage_root": record.stage_root,
            "chat_id": request.get("chat_id"),
            "model": request.get("model"),
            "reasoning": request.get("reasoning"),
        }
        actual = {
            "project_id": episode["project_id"],
            "control_node_id": episode["control_node_id"],
            "provider": episode["provider"],
            "execution_machine": episode["execution_machine"],
            "native_session_id": episode["native_session_id"],
            "stage_host": episode["stage_host"] or "",
            "stage_root": episode["stage_root"],
            "chat_id": episode["chat_id"],
            "model": binding_request.get("model"),
            "reasoning": binding_request.get("reasoning"),
        }
        mismatched = sorted(key for key, value in expected.items() if actual[key] != value)
        if (episode["execution_host"] or "") != (record.stage_host or ""):
            mismatched.append("execution_host")
        if mismatched:
            raise ValueError(
                "The automatic Experiment wake no longer matches its episode binding: "
                + ", ".join(sorted(set(mismatched)))
            )

    @staticmethod
    def _validate_experiment_recovery_claim(
        connection: sqlite3.Connection,
        record: AgentTaskRecord,
        parent: sqlite3.Row,
        parent_request: dict[str, object],
    ) -> None:
        abandoned = connection.execute(
            """
            SELECT 1 FROM graph_run_receipts
            WHERE operation_id = ? AND category = 'experiment_recovery_abandoned'
            LIMIT 1
            """,
            (record.parent_operation_id,),
        ).fetchone()
        if abandoned is not None:
            raise ValueError("Stop loop already abandoned recovery of this Experiment task.")
        if parent["status"] not in {"paused", "interrupted", "failed"}:
            raise ValueError("Only the latest unresolved loop task can be resumed or retried.")
        if record.attempt != int(parent["attempt"]) + 1:
            raise ValueError("A loop recovery task must advance its provider-attempt lineage.")
        child = connection.execute(
            "SELECT 1 FROM graph_runs WHERE parent_operation_id = ? LIMIT 1",
            (record.parent_operation_id,),
        ).fetchone()
        if child is not None:
            raise ValueError("This loop task already has a recovery child.")
        newest_root = connection.execute(
            """
            SELECT request_json FROM graph_runs
            WHERE project_id = ? AND parent_operation_id IS NULL
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(request_json, '$.control_node_id') = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (record.project_id, parent_request["control_node_id"]),
        ).fetchone()
        if newest_root is None:
            raise ValueError("The loop episode root is no longer available.")
        newest_request = json.loads(newest_root["request_json"])
        if newest_request.get("control_episode_id") != parent_request.get(
            "control_episode_id"
        ) or newest_request.get("control_invocation") != parent_request.get("control_invocation"):
            raise ValueError("Only the newest loop episode and invocation can be recovered.")
        newer_attempt = connection.execute(
            """
            SELECT 1 FROM graph_runs
            WHERE project_id = ?
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(request_json, '$.control_node_id') = ?
              AND json_extract(request_json, '$.control_episode_id') = ?
              AND json_extract(request_json, '$.control_invocation') = ?
              AND attempt > ?
            LIMIT 1
            """,
            (
                record.project_id,
                parent_request["control_node_id"],
                parent_request["control_episode_id"],
                parent_request["control_invocation"],
                parent["attempt"],
            ),
        ).fetchone()
        if newer_attempt is not None:
            raise ValueError("Only the latest unresolved loop task can be recovered.")

    @staticmethod
    def _validate_current_experiment_graph_repair(
        connection: sqlite3.Connection,
        *,
        project_id: str,
        control_node_id: str,
        episode_id: str,
        invocation: int,
        operation_id: str,
    ) -> None:
        """Keep patch-only repair on the newest episode, invocation, and attempt."""

        newest_root = connection.execute(
            """
            SELECT json_extract(request_json, '$.control_episode_id') AS episode_id
            FROM graph_runs
            WHERE project_id = ? AND parent_operation_id IS NULL
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(request_json, '$.control_node_id') = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (project_id, control_node_id),
        ).fetchone()
        if newest_root is None or newest_root["episode_id"] != episode_id:
            raise ValueError("Only the newest Experiment episode can repair its graph update.")
        stopped = connection.execute(
            "SELECT stop_requested_at FROM episodes "
            "WHERE episode_id = ? AND mode = 'experiment_loop'",
            (episode_id,),
        ).fetchone()
        if stopped is not None and stopped["stop_requested_at"] is not None:
            raise ValueError("A stopped Experiment episode cannot repair an old graph update.")
        latest = connection.execute(
            """
            SELECT operation_id,
                   json_extract(request_json, '$.control_invocation') AS invocation
            FROM graph_runs
            WHERE project_id = ?
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(request_json, '$.control_node_id') = ?
              AND json_extract(request_json, '$.control_episode_id') = ?
            ORDER BY CAST(json_extract(request_json, '$.control_invocation') AS INTEGER) DESC,
                     attempt DESC, created_at DESC, rowid DESC
            LIMIT 1
            """,
            (project_id, control_node_id, episode_id),
        ).fetchone()
        if (
            latest is None
            or latest["invocation"] != invocation
            or latest["operation_id"] != operation_id
        ):
            raise ValueError(
                "Only the newest Experiment invocation and task attempt can repair its graph "
                "update."
            )

    def _persist_experiment_watchers_idempotently(
        self,
        connection: sqlite3.Connection,
        records: list[StoredWatcherRecord],
        *,
        stops: list[WatcherStopRequest] | None = None,
        binding: WatcherBinding | None = None,
        expected_watcher_snapshot_token: str | None = None,
    ) -> list[StoredWatcherRecord]:
        """Persist one loop handoff in the caller's transaction.

        Deterministic watcher ids make Retry and crash recovery safe. The same
        ``BEGIN IMMEDIATE`` boundary used by Stop loop ensures either the handoff
        lands first and Stop terminalizes it, or the handoff sees stop intent and
        is born stopped. No pollable row can be created after a persisted stop.
        """

        stop_requests = list(stops or [])
        if not records and not stop_requests:
            return []
        records = [self._prepare_watcher_for_insert(record) for record in records]
        if records:
            self._validate_watch_list(records)
        if binding is None:
            raise ValueError("an Experiment handoff requires its bound watcher context")
        continuation = records[0].continuation if records else binding.continuation
        if continuation.patch_kind != "experiment_loop":
            raise ValueError("idempotent Experiment persistence requires loop watchers")
        episode_id = continuation.control_episode_id
        assert episode_id is not None
        if (
            binding is not None
            and records
            and any(
                (
                    record.project_id != binding.project_id
                    or record.origin_operation_id != binding.origin_operation_id
                    or record.origin_task_kind != binding.origin_task_kind
                    or record.graph_target != binding.graph_target
                    or record.chat_id != binding.chat_id
                    or record.node_id != binding.node_id
                    or record.execution_host != binding.execution_host
                    or record.continuation != binding.continuation
                )
                for record in records
            )
        ):
            raise ValueError("Experiment watcher handoff changed its bound continuation context.")
        stop_ids = [item.stop_watcher_id for item in stop_requests]
        if len(stop_ids) != len(set(stop_ids)):
            raise ValueError("Experiment watcher stop ids must be unique")
        watcher_ids = [record.watcher_id for record in records]
        resource = self._admit_experiment_watcher_maintenance(connection, binding)
        if resource is not None:
            if expected_watcher_snapshot_token is None:
                raise ValueError(
                    "Experiment watcher maintenance requires its staged watcher snapshot."
                )
            if expected_watcher_snapshot_token != resource.watcher_snapshot_token:
                raise WatcherClaimConflict(
                    "Experiment watcher state changed after it was staged; inspect the "
                    "current resource before maintaining it."
                )
        episode = self._experiment_episode_row(connection, episode_id)
        if episode is not None and (
            episode["project_id"] != (records[0].project_id if records else binding.project_id)
            or episode["control_node_id"] != continuation.control_node_id
            or self._experiment_episode_record(episode).graph_target != binding.graph_target
        ):
            raise ValueError("This watcher handoff belongs to a different Experiment episode.")
        if stop_requests:
            assert binding is not None
            self._validate_and_apply_agent_watcher_stops(
                connection,
                binding,
                stop_requests,
                episode,
            )
        stopped = episode is not None and episode["stop_requested_at"] is not None
        existing_rows = []
        if watcher_ids:
            placeholders = ",".join("?" for _ in watcher_ids)
            existing_rows = connection.execute(
                f"SELECT * FROM watchers WHERE watcher_id IN ({placeholders})",
                watcher_ids,
            ).fetchall()
        existing_by_id = {
            str(row["watcher_id"]): self._watcher_record(row) for row in existing_rows
        }
        for desired in records:
            existing = existing_by_id.get(desired.watcher_id)
            if existing is not None:
                self._validate_idempotent_watcher(existing, desired)
                if stopped and (existing.status != "stopped" or not existing.notified):
                    self._stop_watcher_for_loop(connection, desired.watcher_id)
                continue
            persisted = (
                desired.model_copy(
                    update={
                        "status": "stopped",
                        "notified": True,
                        "next_check_at": None,
                        "stopped_by": "loop",
                        "stopped_at": self.now(),
                    }
                )
                if stopped
                else desired
            )
            self._insert_watcher(connection, persisted)
        stored_rows = []
        if watcher_ids:
            placeholders = ",".join("?" for _ in watcher_ids)
            stored_rows = connection.execute(
                f"SELECT * FROM watchers WHERE watcher_id IN ({placeholders})",
                watcher_ids,
            ).fetchall()
        stored_by_id = {str(row["watcher_id"]): self._watcher_record(row) for row in stored_rows}
        return [stored_by_id[watcher_id] for watcher_id in watcher_ids]

    def persist_experiment_watchers_idempotently(
        self,
        records: list[StoredWatcherRecord],
        *,
        stops: list[WatcherStopRequest] | None = None,
        binding: WatcherBinding | None = None,
        expected_watcher_snapshot_token: str | None = None,
    ) -> list[StoredWatcherRecord]:
        """Persist one loop handoff atomically with the episode's graceful stop."""

        if not records and not stops:
            return []
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stored = self._persist_experiment_watchers_idempotently(
                connection,
                records,
                stops=stops,
                binding=binding,
                expected_watcher_snapshot_token=expected_watcher_snapshot_token,
            )
        return stored

    def validate_experiment_agent_watcher_stops(
        self,
        binding: WatcherBinding,
        stops: list[WatcherStopRequest],
    ) -> None:
        """Fail a malformed stop handoff before its Patch can be accepted."""

        if not stops:
            return
        episode_id = binding.continuation.control_episode_id
        with self.connection() as connection:
            self._admit_experiment_watcher_maintenance(connection, binding)
            episode = self._experiment_episode_row(connection, str(episode_id))
            self._validate_and_apply_agent_watcher_stops(
                connection,
                binding,
                stops,
                episode,
                apply=False,
            )

    def experiment_watcher_resources(
        self,
        project_id: str,
        *,
        control_node_ids: set[str] | None = None,
        graph_target: GraphTargetRef | None = None,
    ) -> list[ExperimentWatcherResourceRecord]:
        """Return live Experiment resources visible within one already-resolved scope."""

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT json_extract(request_json, '$.control_node_id') AS control_node_id,
                                graph_target_json
                FROM graph_runs
                WHERE project_id = ? AND parent_operation_id IS NULL
                  AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
                """,
                (project_id,),
            ).fetchall()
            resources: list[ExperimentWatcherResourceRecord] = []
            for row in rows:
                control_node_id = row["control_node_id"]
                if not isinstance(control_node_id, str) or not control_node_id:
                    continue
                if control_node_ids is not None and control_node_id not in control_node_ids:
                    continue
                target = GraphTargetRef.model_validate_json(row["graph_target_json"])
                if graph_target is not None and target != graph_target:
                    continue
                try:
                    resource = self._current_experiment_watcher_resource(
                        connection,
                        project_id,
                        control_node_id,
                        graph_target=target,
                    )
                except ValueError:
                    continue
                resources.append(resource)
        return sorted(resources, key=lambda item: (item.control_node_id, item.graph_target.key))

    def admit_experiment_watcher_maintenance(
        self,
        binding: WatcherBinding,
    ) -> ExperimentWatcherResourceRecord | None:
        """Authorize one node-attached watcher handoff from its durable Work task.

        A loop turn returns ``None`` before its first episode binding exists. A
        conversation maintenance turn always returns the current resource and
        fails closed when durable node, episode, or session identity is absent.
        """

        with self.connection() as connection:
            return self._admit_experiment_watcher_maintenance(connection, binding)

    def _admit_experiment_watcher_maintenance(
        self,
        connection: sqlite3.Connection,
        binding: WatcherBinding,
    ) -> ExperimentWatcherResourceRecord | None:
        task_row = connection.execute(
            "SELECT project_id, kind, request_json, graph_target_json "
            "FROM graph_runs WHERE operation_id = ?",
            (binding.origin_operation_id,),
        ).fetchone()
        if task_row is None:
            raise ValueError("Experiment watcher maintenance permission denied: actor is missing.")
        request = json.loads(task_row["request_json"])
        if task_row["project_id"] != binding.project_id:
            raise ValueError(
                "Experiment watcher maintenance permission denied: project scope does not match."
            )
        actor_graph_target = GraphTargetRef.model_validate_json(task_row["graph_target_json"])
        if actor_graph_target != binding.graph_target:
            raise ValueError(
                "Experiment watcher maintenance permission denied: graph target does not match."
            )
        if request.get("mode") != "work" or task_row["kind"] not in {
            "node_chat",
            "project_chat",
        }:
            raise ValueError(
                "Experiment watcher maintenance permission denied: Work capability is required."
            )
        if (
            request.get("chat_id") != binding.chat_id
            or task_row["kind"] != binding.origin_task_kind
        ):
            raise ValueError(
                "Experiment watcher maintenance permission denied: actor provenance does not match."
            )

        continuation = binding.continuation
        control_node_id = continuation.control_node_id
        episode_id = continuation.control_episode_id
        if continuation.patch_kind != "experiment_loop" or not control_node_id or not episode_id:
            raise ValueError(
                "Experiment watcher maintenance requires an explicit node and episode resource."
            )
        if binding.node_id != control_node_id:
            raise ValueError(
                "Experiment watcher maintenance permission denied: target node does not match."
            )

        actor_patch_kind = request.get("patch_kind")
        if actor_patch_kind == "experiment_loop":
            if (
                request.get("control_node_id") != control_node_id
                or request.get("control_episode_id") != episode_id
            ):
                raise ValueError(
                    "Experiment watcher maintenance permission denied: loop binding does not match."
                )
            episode_row = self._experiment_episode_row(connection, episode_id)
            if episode_row is not None and (
                episode_row["project_id"] != binding.project_id
                or episode_row["control_node_id"] != control_node_id
                or self._experiment_episode_record(episode_row).graph_target != binding.graph_target
            ):
                raise ValueError("Experiment watcher maintenance targets a different episode.")
            return None

        if actor_patch_kind != "work":
            raise ValueError(
                "Experiment watcher maintenance permission denied: captured Patch policy is invalid."
            )
        if task_row["kind"] == "node_chat" and request.get("node_id") != control_node_id:
            raise ValueError(
                "Experiment watcher maintenance permission denied: node scope does not include "
                f"{control_node_id}."
            )
        resource = self._current_experiment_watcher_resource(
            connection,
            binding.project_id,
            control_node_id,
            expected_episode_id=episode_id,
            graph_target=binding.graph_target,
        )
        if binding.execution_host != resource.execution_host:
            raise ValueError("Experiment watcher maintenance must use the episode execution host.")
        if continuation != resource.continuation:
            raise ValueError(
                "Experiment watcher maintenance no longer matches the live episode policy."
            )
        return resource

    def _current_experiment_watcher_resource(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        control_node_id: str,
        *,
        expected_episode_id: str | None = None,
        graph_target: GraphTargetRef | None = None,
    ) -> ExperimentWatcherResourceRecord:
        target = graph_target or GraphTargetRef()
        root_row = connection.execute(
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
            (project_id, control_node_id, target.kind, target.branch_id),
        ).fetchone()
        if root_row is None:
            raise ValueError("Experiment watcher maintenance requires a current live episode.")
        root_request = json.loads(root_row["request_json"])
        episode_id = root_request.get("control_episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(
                "Experiment watcher maintenance cannot prove the current episode identity."
            )
        if expected_episode_id is not None and expected_episode_id != episode_id:
            raise ValueError("Experiment watcher maintenance targets a stale episode.")
        episode_row = self._experiment_episode_row(connection, episode_id)
        if episode_row is None:
            raise ValueError(
                "Experiment watcher maintenance cannot prove the episode session binding."
            )
        episode = self._experiment_episode_record(episode_row)
        if (
            episode.project_id != project_id
            or episode.control_node_id != control_node_id
            or episode.graph_target != target
            or not episode.session_bound
        ):
            raise ValueError(
                "Experiment watcher maintenance cannot prove the episode session binding."
            )
        if episode.stop_requested_at is not None or episode.stop_settled_at is not None:
            raise ValueError("Experiment watcher maintenance requires a live, unstopped episode.")
        exited = connection.execute(
            """
            SELECT 1 FROM graph_run_receipts AS receipt
            JOIN graph_runs AS run ON run.operation_id = receipt.operation_id
            WHERE run.project_id = ?
              AND json_extract(run.request_json, '$.control_episode_id') = ?
              AND receipt.category = 'experiment_loop_exit'
            LIMIT 1
            """,
            (project_id, episode_id),
        ).fetchone()
        if exited is not None:
            raise ValueError("Experiment watcher maintenance requires a live, unexited episode.")
        if not episode.last_turn_operation_id:
            raise ValueError(
                "Experiment watcher maintenance cannot prove the episode's latest turn."
            )
        turn_row = connection.execute(
            "SELECT request_json FROM graph_runs WHERE operation_id = ? AND project_id = ?",
            (episode.last_turn_operation_id, project_id),
        ).fetchone()
        if turn_row is None:
            raise ValueError(
                "Experiment watcher maintenance cannot prove the episode's latest turn."
            )
        turn_request = json.loads(turn_row["request_json"])
        continuation_data = {
            key: turn_request[key]
            for key in WatcherContinuation.model_fields
            if key in turn_request
        }
        for nullable_list in ("workflow_ids", "skill_ids", "resolved_skill_packages"):
            if continuation_data.get(nullable_list) is None:
                continuation_data[nullable_list] = []
        continuation = WatcherContinuation.model_validate(continuation_data)
        if (
            continuation.patch_kind != "experiment_loop"
            or continuation.control_node_id != control_node_id
            or continuation.control_episode_id != episode_id
        ):
            raise ValueError(
                "Experiment watcher maintenance cannot prove the episode continuation policy."
            )
        wake_task_kind = root_row["kind"]
        if wake_task_kind != "node_chat":
            raise ValueError("Experiment watcher maintenance has an invalid wake task binding.")
        if not episode.chat_id:
            # The wake target is derived, never guessed: without the episode's own
            # conversation there is nothing to wake, so fail closed with a diagnostic
            # rather than an AssertionError that -O would strip.
            raise ValueError(
                "Experiment watcher maintenance cannot prove the episode's wake conversation."
            )
        return ExperimentWatcherResourceRecord(
            project_id=project_id,
            control_node_id=control_node_id,
            episode_id=episode_id,
            graph_target=target,
            execution_host=episode.execution_host,
            wake_task_kind=wake_task_kind,
            wake_chat_id=episode.chat_id,
            continuation=continuation,
            watcher_snapshot_token=self._experiment_watcher_snapshot_token(
                connection,
                project_id,
                control_node_id,
                target,
            ),
        )

    @staticmethod
    def _experiment_watcher_snapshot_token(
        connection: sqlite3.Connection,
        project_id: str,
        control_node_id: str,
        graph_target: GraphTargetRef,
    ) -> str:
        """Fingerprint the node's observer membership, and nothing else.

        This defends exactly one gap. Every retirement is already a
        compare-and-swap inside the arming transaction, so a delivery claim, a
        **Stop loop**, or an already-resolved stop is caught per item without a
        fingerprint. Arming is not: new observers are plain inserts, so two
        maintenance turns could each retire the old set and each arm
        replacements, leaving the Experiment double-observed.

        Membership answers that and stays blind to everything RCP merely
        observed. Status and consecutive-error counts deliberately do not appear:
        a degraded observer is re-checked on the S84 backoff, so fingerprinting
        observation would reject the maintenance turn that exists to repair that
        very observer. Retired rows keep their id, so the set only grows and a
        concurrent retirement does not collide with an unrelated repair.
        """

        rows = connection.execute(
            """
            SELECT watcher_id FROM watchers
            WHERE project_id = ? AND node_id = ?
              AND json_extract(graph_target_json, '$.kind') = ?
              AND json_extract(graph_target_json, '$.branch_id') IS ?
              AND json_extract(continuation_json, '$.patch_kind') = 'experiment_loop'
            ORDER BY watcher_id
            """,
            (project_id, control_node_id, graph_target.kind, graph_target.branch_id),
        ).fetchall()
        snapshot = json.dumps(
            [str(row["watcher_id"]) for row in rows],
            separators=(",", ":"),
        )
        return hashlib.sha256(snapshot.encode("utf-8")).hexdigest()

    def experiment_watcher_ids(self, project_id: str, control_node_id: str) -> list[str]:
        """Live watchers armed by a bounded loop on one experiment."""

        return [
            record.watcher_id
            for record in self.watchers(project_id)
            if (
                (record.status in {"active", "degraded"} and not record.notified)
                or (record.status == "completed" and not record.notified)
            )
            and record.continuation.control_node_id == control_node_id
        ]

    def experiment_handoff_has_live_watcher_after_stops(
        self,
        binding: WatcherBinding,
        stop_watcher_ids: list[str],
    ) -> bool:
        """Whether a stop-only handoff leaves another compatible wake source."""

        continuation = binding.continuation
        episode_id = continuation.control_episode_id
        control_node_id = continuation.control_node_id
        if not episode_id or not control_node_id:
            return False
        stopped = set(stop_watcher_ids)
        with self.connection() as connection:
            episode_row = self._experiment_episode_row(connection, episode_id)
            if episode_row is None:
                return False
            episode = self._experiment_episode_record(episode_row)
            root = self._experiment_episode_root_request(
                connection,
                binding.project_id,
                control_node_id,
                episode_id,
            )
            if root is None:
                return False
            rows = connection.execute(
                """
                SELECT * FROM watchers
                WHERE project_id = ? AND status IN ('active', 'degraded', 'completed')
                  AND notified = 0
                """,
                (binding.project_id,),
            ).fetchall()
        return any(
            record.watcher_id not in stopped
            and self._experiment_watcher_matches_current(record, root, episode)
            for record in (self._watcher_record(row) for row in rows)
        )

    def experiment_episode(self, episode_id: str) -> ExperimentEpisodeRecord | None:
        with self.connection() as connection:
            row = self._experiment_episode_row(connection, episode_id)
        return self._experiment_episode_record(row) if row is not None else None

    def experiment_episode_ending_signal(
        self,
        episode_id: str,
    ) -> tuple[str, dict[str, object]] | None:
        """Return the exact accepted continuation and its mode-owned ending receipt."""

        with self.connection() as connection:
            state = self._experiment_episode_row(connection, episode_id)
            operation_id = state["last_turn_operation_id"] if state is not None else None
            if not isinstance(operation_id, str) or not operation_id:
                return None
            receipt = connection.execute(
                """
                SELECT payload_json FROM graph_run_receipts
                WHERE operation_id = ? AND category = 'experiment_loop_exit'
                ORDER BY receipt_id DESC LIMIT 1
                """,
                (operation_id,),
            ).fetchone()
        if receipt is None:
            return None
        signal = json.loads(receipt["payload_json"])
        if not isinstance(signal, dict) or signal.get("episode_id") != episode_id:
            raise ValueError("The Experiment ending receipt has invalid episode lineage.")
        return operation_id, signal

    def settle_experiment_episode_wrapup(self, episode_id: str) -> bool:
        """Quiesce this episode's observers once its non-Stop ending is fenced.

        The generic episode fence prevents any later watcher claim from spending
        another invocation. Unnotified observers keep polling or remain pending
        so a later completion can be claimed by a fresh human Run; only observers
        already consumed by this episode are retired before hidden report work.
        """

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode = connection.execute(
                """
                SELECT * FROM episodes
                WHERE episode_id = ? AND mode = 'experiment_loop'
                """,
                (episode_id,),
            ).fetchone()
            if episode is None:
                raise KeyError(episode_id)
            if episode["ending"] is None or episode["ending"] == "stopped":
                return False
            active = connection.execute(
                """
                SELECT 1 FROM graph_runs
                WHERE episode_id = ? AND visible = 1
                  AND status IN ('queued', 'running', 'pausing')
                LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
            if active is not None:
                return False
            connection.execute(
                """
                UPDATE watchers
                SET status = 'stopped', notified = 1, next_check_at = NULL,
                    stopped_by = COALESCE(stopped_by, 'loop'),
                    stopped_at = COALESCE(stopped_at, ?)
                WHERE episode_id = ?
                  AND status IN ('active', 'degraded', 'completed')
                  AND notified = 1
                """,
                (now, episode_id),
            )
        return True

    def experiment_episode_recovery_context_problem(self, operation_id: str) -> str | None:
        """Explain why this task lineage cannot retain its episode context on recovery."""

        with self.connection() as connection:
            return self._experiment_episode_recovery_context_problem(connection, operation_id)

    @staticmethod
    def _experiment_episode_recovery_context_problem(
        connection: sqlite3.Connection,
        operation_id: str,
    ) -> str | None:
        """Validate the immutable candidate on an Experiment invocation's lineage root."""

        current_id = operation_id
        seen: set[str] = set()
        while True:
            if current_id in seen:
                return (
                    "This Experiment-loop turn cannot be resumed or retried because its task "
                    "lineage contains a cycle. Use Stop loop and press Run to start a fresh "
                    "episode."
                )
            seen.add(current_id)
            row = connection.execute(
                "SELECT parent_operation_id FROM graph_runs WHERE operation_id = ?",
                (current_id,),
            ).fetchone()
            if row is None:
                return (
                    "This Experiment-loop turn cannot be resumed or retried because its task "
                    "lineage is incomplete. Use Stop loop and press Run to start a fresh episode."
                )
            parent_id = row["parent_operation_id"]
            if parent_id is None:
                break
            current_id = str(parent_id)

        contract = connection.execute(
            """
            SELECT content FROM graph_run_contracts
            WHERE operation_id = ? AND role = ?
            """,
            (current_id, _EXPERIMENT_EPISODE_CONTEXT_CANDIDATE_ROLE),
        ).fetchone()
        if contract is None:
            return _MISSING_EXPERIMENT_EPISODE_CONTEXT_DIAGNOSTIC
        try:
            candidate = json.loads(contract["content"])
        except (json.JSONDecodeError, TypeError):
            candidate = None
        if not isinstance(candidate, dict):
            return (
                "This Experiment-loop turn cannot be resumed or retried because its retained "
                "episode context candidate is invalid. Use Stop loop and press Run to start a "
                "fresh episode."
            )
        return None

    def previous_experiment_episode(
        self,
        project_id: str,
        control_node_id: str,
        episode_id: str,
    ) -> ExperimentEpisodeRecord | None:
        """Return the episode immediately before this one for the same Experiment.

        The generic episode parent is the canonical ordering and lifecycle ledger.
        """

        with self.connection() as connection:
            current = connection.execute(
                """
                SELECT graph_target_json FROM episodes
                WHERE episode_id = ? AND project_id = ? AND mode = 'experiment_loop'
                  AND control_node_id = ?
                """,
                (episode_id, project_id, control_node_id),
            ).fetchone()
            if current is None:
                return None
            rows = connection.execute(
                """
                SELECT episode_id FROM episodes
                WHERE project_id = ? AND mode = 'experiment_loop'
                  AND control_node_id = ?
                  AND graph_target_json = ?
                ORDER BY created_at DESC, episode_id DESC
                """,
                (project_id, control_node_id, current["graph_target_json"]),
            ).fetchall()
        ordered = [str(row["episode_id"]) for row in rows]
        if episode_id not in ordered:
            return None
        position = ordered.index(episode_id) + 1
        if position >= len(ordered):
            return None
        return self.experiment_episode(ordered[position])

    def _commit_experiment_episode_turn(
        self,
        connection: sqlite3.Connection,
        *,
        episode_id: str,
        project_id: str,
        control_node_id: str,
        provider: str,
        execution_machine: str,
        execution_host: str,
        native_session_id: str,
        stage_host: str | None,
        stage_root: str,
        chat_id: str,
        operation_id: str,
        invocation: int,
        graph_result: str,
        watcher_ids: list[str],
        context_baseline: dict[str, object],
        ending_signal: dict[str, object] | None = None,
        replace_binding: bool = False,
        replacement_provenance: dict[str, object] | None = None,
    ) -> None:
        """Bind this episode to the session a later automatic wake resumes.

        Only a mechanically successful joint handoff commits, so a wake never
        tries to continue a session that never established one, and the context
        baseline can only move forward with an accepted operational turn. A
        graph-only rejection is retained as that turn's truthful result.
        """

        if not native_session_id or not stage_root:
            raise ValueError("An episode binding requires a native session and its exact stage.")
        if replace_binding and replacement_provenance is None:
            raise ValueError("An episode binding replacement requires its recovery provenance.")
        replacement_payload_json = (
            self._bounded_receipt_payload(replacement_provenance)
            if replacement_provenance is not None
            else None
        )
        ending_payload_json: str | None = None
        if ending_signal is not None:
            ending = ending_signal.get("ending")
            if (
                ending_signal.get("episode_id") != episode_id
                or ending not in {"completed", "human_pause"}
                or not isinstance(ending_signal.get("partial"), bool)
                or ending_signal.get("partial") != (ending == "human_pause")
                or not isinstance(ending_signal.get("receipt"), dict)
            ):
                raise ValueError("The Experiment ending receipt is inconsistent with its episode.")
            try:
                ending_payload_json = json.dumps(
                    ending_signal,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("The Experiment ending receipt is not valid JSON.") from exc
            if len(ending_payload_json.encode("utf-8")) > AGENT_TASK_RECEIPT_MAX_BYTES:
                raise ValueError("The compact Experiment ending receipt exceeds its storage limit.")
        now = self.now()
        existing = self._experiment_episode_row(connection, episode_id)
        if (
            existing is None
            or existing["project_id"] != project_id
            or existing["control_node_id"] != control_node_id
        ):
            raise ValueError("This episode id belongs to a different Experiment.")
        turn = connection.execute(
            """
            SELECT project_id, episode_id, request_json
            FROM graph_runs WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        turn_request = json.loads(turn["request_json"]) if turn is not None else None
        if (
            turn is None
            or turn["project_id"] != project_id
            or turn["episode_id"] != episode_id
            or not isinstance(turn_request, dict)
            or turn_request.get("control_node_id") != control_node_id
            or turn_request.get("control_invocation") != invocation
        ):
            raise ValueError("The accepted handoff does not match its exact Experiment task.")
        if existing["native_session_id"] is not None:
            fixed = {
                "execution_machine": execution_machine,
                "execution_host": execution_host,
                "chat_id": chat_id,
            }
            fixed_conflicts = sorted(
                field for field, value in fixed.items() if (existing[field] or "") != value
            )
            if fixed_conflicts:
                raise ValueError(
                    "An Experiment episode recovery cannot change its pinned identity: "
                    + ", ".join(fixed_conflicts)
                )
            binding = {
                "provider": provider,
                "native_session_id": native_session_id,
                "stage_host": stage_host or "",
                "stage_root": stage_root,
            }
            binding_conflicts = sorted(
                field for field, value in binding.items() if (existing[field] or "") != value
            )
            if binding_conflicts and not replace_binding:
                raise ValueError(
                    "An Experiment episode cannot change its native-session binding: "
                    + ", ".join(binding_conflicts)
                )
        connection.execute(
            """
            UPDATE experiment_episode_state
            SET provider = ?, execution_machine = ?, execution_host = ?,
                native_session_id = ?, stage_host = ?, stage_root = ?, chat_id = ?,
                last_turn_operation_id = ?, last_turn_invocation = ?,
                last_graph_result = ?, last_watcher_ids_json = ?,
                context_baseline_json = ?, session_diagnostic = NULL, updated_at = ?
            WHERE episode_id = ?
            """,
            (
                provider,
                execution_machine,
                execution_host,
                native_session_id,
                stage_host,
                stage_root,
                chat_id,
                operation_id,
                invocation,
                graph_result,
                json.dumps(list(watcher_ids), separators=(",", ":")),
                json.dumps(context_baseline, sort_keys=True, separators=(",", ":")),
                now,
                episode_id,
            ),
        )
        if replace_binding and replacement_payload_json is not None:
            self._insert_agent_task_receipt(
                connection,
                operation_id,
                "experiment_episode_binding_replaced",
                replacement_payload_json,
                tier="summary",
                created_at=now,
            )
        if ending_payload_json is not None:
            committed_ending = connection.execute(
                """
                SELECT payload_json FROM graph_run_receipts
                WHERE operation_id = ? AND category = 'experiment_loop_exit'
                ORDER BY receipt_id DESC LIMIT 1
                """,
                (operation_id,),
            ).fetchone()
            if committed_ending is not None:
                if committed_ending["payload_json"] != ending_payload_json:
                    raise ValueError("The Experiment task already has another ending receipt.")
            else:
                self._insert_agent_task_receipt(
                    connection,
                    operation_id,
                    "experiment_loop_exit",
                    ending_payload_json,
                    tier="summary",
                    created_at=now,
                )

    def commit_experiment_episode_turn(
        self,
        *,
        episode_id: str,
        project_id: str,
        control_node_id: str,
        provider: str,
        execution_machine: str,
        execution_host: str,
        native_session_id: str,
        stage_host: str | None,
        stage_root: str,
        chat_id: str,
        operation_id: str,
        invocation: int,
        graph_result: str,
        watcher_ids: list[str],
        context_baseline: dict[str, object],
        ending_signal: dict[str, object] | None = None,
        replace_binding: bool = False,
        replacement_provenance: dict[str, object] | None = None,
    ) -> ExperimentEpisodeRecord:
        """Bind this episode to the session a later automatic wake resumes."""

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._commit_experiment_episode_turn(
                connection,
                episode_id=episode_id,
                project_id=project_id,
                control_node_id=control_node_id,
                provider=provider,
                execution_machine=execution_machine,
                execution_host=execution_host,
                native_session_id=native_session_id,
                stage_host=stage_host,
                stage_root=stage_root,
                chat_id=chat_id,
                operation_id=operation_id,
                invocation=invocation,
                graph_result=graph_result,
                watcher_ids=watcher_ids,
                context_baseline=context_baseline,
                ending_signal=ending_signal,
                replace_binding=replace_binding,
                replacement_provenance=replacement_provenance,
            )
        stored = self.experiment_episode(episode_id)
        assert stored is not None
        return stored

    def commit_experiment_episode_handoff(
        self,
        records: list[StoredWatcherRecord],
        *,
        binding: WatcherBinding,
        operation_id: str,
        native_session_id: str,
        stage_host: str | None,
        stage_root: str,
        graph_result: str,
        context_baseline: dict[str, object],
        stops: list[WatcherStopRequest] | None = None,
        expected_watcher_snapshot_token: str | None = None,
        ending_signal: dict[str, object] | None = None,
        replace_binding: bool = False,
        replacement_provenance: dict[str, object] | None = None,
    ) -> tuple[list[StoredWatcherRecord], ExperimentEpisodeRecord]:
        """Atomically persist loop watchers and commit their episode handoff.

        ``binding.origin_operation_id`` remains the watcher identity root. The
        separate ``operation_id`` is the current completed task whose session
        and episode turn are being committed.
        """

        continuation = binding.continuation
        episode_id = continuation.control_episode_id
        control_node_id = continuation.control_node_id
        invocation = continuation.control_invocation
        if (
            continuation.patch_kind != "experiment_loop"
            or not episode_id
            or not control_node_id
            or invocation is None
        ):
            raise ValueError("An Experiment handoff requires its complete loop continuation.")
        if binding.episode_id is not None and binding.episode_id != episode_id:
            raise ValueError("An Experiment handoff binding changed its episode identity.")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stored_watchers = self._persist_experiment_watchers_idempotently(
                connection,
                records,
                stops=stops,
                binding=binding,
                expected_watcher_snapshot_token=expected_watcher_snapshot_token,
            )
            self._commit_experiment_episode_turn(
                connection,
                episode_id=episode_id,
                project_id=binding.project_id,
                control_node_id=control_node_id,
                provider=continuation.provider,
                execution_machine=continuation.run_on,
                execution_host=binding.execution_host,
                native_session_id=native_session_id,
                stage_host=stage_host,
                stage_root=stage_root,
                chat_id=binding.chat_id,
                operation_id=operation_id,
                invocation=invocation,
                graph_result=graph_result,
                watcher_ids=[item.watcher_id for item in stored_watchers],
                context_baseline=context_baseline,
                ending_signal=ending_signal,
                replace_binding=replace_binding,
                replacement_provenance=replacement_provenance,
            )
        stored_episode = self.experiment_episode(episode_id)
        assert stored_episode is not None
        return stored_watchers, stored_episode

    def record_experiment_episode_diagnostic(
        self,
        *,
        episode_id: str,
        project_id: str,
        control_node_id: str,
        diagnostic: str | None,
    ) -> None:
        """Persist why an automatic wake could not use this episode's session."""

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode = self._experiment_episode_row(connection, episode_id)
            if (
                episode is None
                or episode["project_id"] != project_id
                or episode["control_node_id"] != control_node_id
            ):
                raise ValueError("This diagnostic belongs to another Experiment episode.")
            connection.execute(
                "UPDATE experiment_episode_state "
                "SET session_diagnostic = ?, updated_at = ? WHERE episode_id = ?",
                (diagnostic, now, episode_id),
            )

    def request_experiment_loop_stop(
        self,
        project_id: str,
        control_node_id: str,
        *,
        episode_id: str | None = None,
        graph_target: GraphTargetRef | None = None,
    ) -> ExperimentEpisodeRecord | None:
        """Persist a durable stop for one resolved episode before a new claim can win.

        The intent is written under the same write lock a watcher claim takes, so
        a claim that committed first becomes the current turn and anything later
        finds the loop already stopped. Omitting exact identity preserves the
        legacy newest-episode selection, while callers that already resolved an
        episode must pass it so a different target cannot win between validation
        and mutation.
        """

        with self.connection() as connection:
            selected = self._experiment_episode_for_stop(
                connection,
                project_id,
                control_node_id,
                episode_id=episode_id,
                graph_target=graph_target,
            )
        if selected is None:
            return None
        selected_episode_id, selected_target = selected
        self.request_episode_stop(selected_episode_id)
        return self.settle_experiment_loop_stop(
            project_id,
            control_node_id,
            episode_id=selected_episode_id,
            graph_target=selected_target,
        )

    def settle_experiment_loop_stop(
        self,
        project_id: str,
        control_node_id: str,
        *,
        episode_id: str | None = None,
        graph_target: GraphTargetRef | None = None,
    ) -> ExperimentEpisodeRecord | None:
        """Reconcile a persisted stop once its authorized turn is no longer live."""

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            selected = self._experiment_episode_for_stop(
                connection,
                project_id,
                control_node_id,
                episode_id=episode_id,
                graph_target=graph_target,
            )
            if selected is None:
                return None
            selected_episode_id, _selected_target = selected
            quiescent = self._settle_experiment_loop_stop(
                connection,
                project_id,
                control_node_id,
                selected_episode_id,
            )
        if quiescent:
            self.mark_episode_stop_skipped(selected_episode_id)
        return self.experiment_episode(selected_episode_id)

    def _settle_experiment_loop_stop(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        control_node_id: str,
        episode_id: str,
    ) -> bool:
        """Terminalize this episode's observers once its authorized turn is resolved.

        "Resolved" is the same predicate the runtime calls `task_active`, not just
        "not running": a turn that paused or failed is still the authorized turn
        the human may Resume, so the loop keeps reading Stopping until it reaches
        a terminal state. A claimed watcher keeps its notification provenance,
        but becomes stopped once the task it woke has finished successfully.
        """

        requested = self._experiment_episode_row(connection, episode_id)
        if (
            requested is None
            or requested["project_id"] != project_id
            or requested["control_node_id"] != control_node_id
            or requested["stop_requested_at"] is None
        ):
            return False
        # A superseded attempt does not count: only the newest attempt of each
        # invocation is the turn the human can still act on, which is exactly what
        # `experiment_loop_runtime` reports as `task_active`.
        unresolved = connection.execute(
            """
            SELECT task.operation_id, task.status FROM graph_runs AS task
            WHERE task.project_id = ?
              AND json_extract(task.request_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(task.request_json, '$.control_node_id') = ?
              AND json_extract(task.request_json, '$.control_episode_id') = ?
              AND task.status IN ('queued', 'running', 'pausing', 'paused', 'failed', 'interrupted')
              AND NOT EXISTS (
                  SELECT 1 FROM graph_runs AS child
                  WHERE child.parent_operation_id = task.operation_id
              )
            """,
            (project_id, control_node_id, episode_id),
        ).fetchall()
        if unresolved:
            diagnostic = requested["session_diagnostic"]
            if not diagnostic:
                diagnostic = next(
                    (
                        problem
                        for row in unresolved
                        if (
                            problem := self._experiment_episode_recovery_context_problem(
                                connection,
                                str(row["operation_id"]),
                            )
                        )
                    ),
                    None,
                )
                if diagnostic:
                    now = self.now()
                    connection.execute(
                        "UPDATE experiment_episode_state "
                        "SET session_diagnostic = ?, updated_at = ? WHERE episode_id = ?",
                        (diagnostic, now, episode_id),
                    )
            abandonable = bool(diagnostic) and all(
                row["status"] in {"paused", "failed", "interrupted"} for row in unresolved
            )
            if not abandonable:
                return False
            now = self.now()
            for row in unresolved:
                already_abandoned = connection.execute(
                    """
                    SELECT 1 FROM graph_run_receipts
                    WHERE operation_id = ? AND category = 'experiment_recovery_abandoned'
                    LIMIT 1
                    """,
                    (row["operation_id"],),
                ).fetchone()
                if already_abandoned is not None:
                    continue
                detail = (
                    "Stop loop abandoned recovery of this terminal task because its saved "
                    "episode session cannot be continued. The task and all history remain "
                    "inspectable."
                )
                self._insert_agent_task_receipt(
                    connection,
                    str(row["operation_id"]),
                    "experiment_recovery_abandoned",
                    self._bounded_receipt_payload({"episode_id": episode_id, "reason": diagnostic}),
                    tier="summary",
                    created_at=now,
                )
                self._insert_agent_task_event(
                    connection,
                    str(row["operation_id"]),
                    detail,
                    level="warning",
                    created_at=now,
                )
        root_request = self._experiment_episode_root_request(
            connection,
            project_id,
            control_node_id,
            episode_id,
        )
        episode = self._experiment_episode_record(requested)
        watcher_rows = connection.execute(
            """
            SELECT * FROM watchers
            WHERE project_id = ?
              AND json_extract(continuation_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(continuation_json, '$.control_node_id') = ?
              AND status IN ('active', 'degraded', 'completed')
            """,
            (project_id, control_node_id),
        ).fetchall()
        watcher_ids = {
            record.watcher_id
            for record in (self._watcher_record(row) for row in watcher_rows)
            if record.episode_id == episode_id
            and root_request is not None
            and self._experiment_watcher_matches_current(record, root_request, episode)
        }
        claimed_rows = connection.execute(
            """
            SELECT watcher_id FROM watchers
            WHERE project_id = ?
              AND notification_operation_id IN (
                  SELECT operation_id FROM graph_runs
                  WHERE project_id = ?
                    AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
                    AND json_extract(request_json, '$.control_node_id') = ?
                    AND json_extract(request_json, '$.control_episode_id') = ?
              )
            """,
            (project_id, project_id, control_node_id, episode_id),
        ).fetchall()
        watcher_ids.update(str(row["watcher_id"]) for row in claimed_rows)
        if watcher_ids:
            placeholders = ",".join("?" for _ in watcher_ids)
            connection.execute(
                f"UPDATE watchers SET status = 'stopped', notified = 1, next_check_at = NULL, "
                "stopped_by = COALESCE(stopped_by, 'loop'), "
                "stopped_at = COALESCE(stopped_at, ?) "
                f"WHERE watcher_id IN ({placeholders})",
                (self.now(), *sorted(watcher_ids)),
            )
        return True

    def settle_ready_experiment_loop_stops(self) -> int:
        """Reconcile every durable stop that no longer has a recoverable turn."""

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT episode_id, project_id, control_node_id, graph_target_json
                FROM episodes
                WHERE mode = 'experiment_loop'
                  AND stop_requested_at IS NOT NULL AND stop_settled_at IS NULL
                ORDER BY created_at, episode_id
                """
            ).fetchall()
        settled = 0
        for row in rows:
            before = self.episode(str(row["episode_id"]))
            self.settle_experiment_loop_stop(
                str(row["project_id"]),
                str(row["control_node_id"]),
                episode_id=str(row["episode_id"]),
                graph_target=GraphTargetRef.model_validate_json(row["graph_target_json"]),
            )
            after = self.episode(str(row["episode_id"]))
            if before is not None and before.stop_settled_at is None and after is not None:
                settled += int(after.stop_settled_at is not None)
        return settled

    def stopping_experiment_recovery_candidates(self) -> list[AgentTaskRecord]:
        """Return each durable Stop's unresolved leaf that may finish in recovery.

        These are already-authorized turns, not new Experiment invocations.  The
        background adapter still proves the exact native session and stage before
        it creates a recovery child; this query only identifies the leaves whose
        Stop would otherwise remain unresolved after a process restart.
        """

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT task.*,
                       EXISTS (
                           SELECT 1 FROM graph_run_receipts AS receipt
                           WHERE receipt.operation_id = task.operation_id
                             AND receipt.category IN (
                                 'experiment_recovery_abandoned',
                                 'auto_research_recovery_abandoned'
                             )
                       ) AS recovery_abandoned
                FROM episodes AS episode
                JOIN experiment_episode_state AS state
                  ON state.episode_id = episode.episode_id
                JOIN graph_runs AS task
                  ON task.episode_id = episode.episode_id
                WHERE episode.mode = 'experiment_loop'
                  AND episode.stop_requested_at IS NOT NULL
                  AND episode.stop_settled_at IS NULL
                  AND episode.ending IS NULL
                  AND state.session_diagnostic IS NULL
                  AND task.visible = 1
                  AND task.status IN ('paused', 'failed', 'interrupted')
                  AND NOT EXISTS (
                      SELECT 1 FROM graph_runs AS child
                      WHERE child.parent_operation_id = task.operation_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM graph_run_receipts AS receipt
                      WHERE receipt.operation_id = task.operation_id
                        AND receipt.category = 'experiment_recovery_abandoned'
                  )
                ORDER BY episode.created_at, episode.episode_id,
                         task.created_at, task.operation_id
                """
            ).fetchall()
        return [self._agent_task_record(row) for row in rows]

    @staticmethod
    def _experiment_episode_for_stop(
        connection: sqlite3.Connection,
        project_id: str,
        control_node_id: str,
        *,
        episode_id: str | None = None,
        graph_target: GraphTargetRef | None = None,
    ) -> tuple[str, GraphTargetRef] | None:
        row = connection.execute(
            """
            SELECT episode_id, graph_target_json FROM episodes
            WHERE project_id = ? AND mode = 'experiment_loop' AND control_node_id = ?
              AND (? IS NULL OR episode_id = ?)
              AND (
                  ? = 0
                  OR (
                      json_extract(graph_target_json, '$.kind') = ?
                      AND json_extract(graph_target_json, '$.branch_id') IS ?
                  )
              )
            ORDER BY created_at DESC, episode_id DESC
            LIMIT 1
            """,
            (
                project_id,
                control_node_id,
                episode_id,
                episode_id,
                int(graph_target is not None),
                graph_target.kind if graph_target is not None else "main",
                graph_target.branch_id if graph_target is not None else None,
            ),
        ).fetchone()
        if row is None or not isinstance(row["episode_id"], str):
            return None
        return row["episode_id"], GraphTargetRef.model_validate_json(row["graph_target_json"])

    def experiment_loop_runtime(
        self,
        project_id: str,
        control_node_id: str,
        *,
        graph_target: GraphTargetRef | None = None,
    ) -> ExperimentLoopRuntime:
        """Project the globally newest episode, optionally only on one target.

        This is the display/cache contract: when a newer same-node episode lives
        on another target it returns an empty runtime instead of reviving older
        operational state. Target-bound operations use
        :meth:`experiment_loop_runtime_for_target` instead.
        """

        return self.experiment_loop_runtimes(
            project_id,
            [control_node_id],
            graph_target=graph_target,
        )[control_node_id]

    def experiment_loop_runtime_for_target(
        self,
        project_id: str,
        control_node_id: str,
        graph_target: GraphTargetRef,
    ) -> ExperimentLoopRuntime:
        """Derive the newest episode on one exact target for operational authority."""

        projected = self._project_experiment_loop_runtimes(
            project_id,
            {control_node_id},
            graph_target=graph_target,
        )
        return projected.get(control_node_id, ExperimentLoopRuntime())

    def experiment_loop_runtimes(
        self,
        project_id: str,
        control_node_ids: Iterable[str],
        *,
        graph_target: GraphTargetRef | None = None,
    ) -> dict[str, ExperimentLoopRuntime]:
        """Derive current runtimes, optionally only when newest belongs to one target."""

        requested = tuple(dict.fromkeys(control_node_ids))
        if not requested:
            return {}
        with self.connection() as connection:
            connection.execute("BEGIN")
            return self._experiment_loop_runtimes_in_connection(
                connection,
                project_id,
                requested,
                graph_target=graph_target,
            )

    def experiment_control_projection_snapshots(
        self,
        project_id: str,
        control_node_ids: Iterable[str] | None = None,
        *,
        graph_target: GraphTargetRef | None = None,
    ) -> dict[str, ExperimentControlProjectionSnapshot]:
        """Read complete Experiment control inputs from one SQLite snapshot."""

        requested = None if control_node_ids is None else tuple(dict.fromkeys(control_node_ids))
        if requested == ():
            return {}
        with self.connection() as connection:
            connection.execute("BEGIN")
            if requested is None:
                runtimes = self._project_experiment_loop_runtimes(
                    project_id,
                    None,
                    _connection=connection,
                )
            else:
                runtimes = self._experiment_loop_runtimes_in_connection(
                    connection,
                    project_id,
                    requested,
                    graph_target=graph_target,
                )
            snapshots: dict[str, ExperimentControlProjectionSnapshot] = {}
            for control_node_id, runtime in runtimes.items():
                episode_snapshot = (
                    self._experiment_episode_projection_snapshot_in_connection(
                        connection,
                        project_id,
                        control_node_id,
                        runtime.episode_id,
                    )
                    if runtime.episode_id is not None
                    else None
                )
                snapshots[control_node_id] = ExperimentControlProjectionSnapshot(
                    runtime=runtime,
                    episode=episode_snapshot,
                    latest_report_episode_id=(
                        self._latest_experiment_report_episode_id_in_connection(
                            connection,
                            project_id,
                            control_node_id,
                            episode_snapshot.episode.graph_target,
                        )
                        if episode_snapshot is not None
                        else None
                    ),
                )
            return snapshots

    @staticmethod
    def _latest_experiment_report_episode_id_in_connection(
        connection: sqlite3.Connection,
        project_id: str,
        control_node_id: str,
        graph_target: GraphTargetRef,
    ) -> str | None:
        row = connection.execute(
            """
            SELECT episodes.episode_id
            FROM episodes
            JOIN episode_reports ON episode_reports.episode_id = episodes.episode_id
            WHERE episodes.project_id = ?
              AND episodes.mode = 'experiment_loop'
              AND episodes.control_node_id = ?
              AND episodes.graph_target_json = ?
            ORDER BY episode_reports.created_at DESC, episodes.episode_id DESC
            LIMIT 1
            """,
            (project_id, control_node_id, graph_target.model_dump_json()),
        ).fetchone()
        return str(row["episode_id"]) if row is not None else None

    def _experiment_loop_runtimes_in_connection(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        requested: tuple[str, ...],
        *,
        graph_target: GraphTargetRef | None,
    ) -> dict[str, ExperimentLoopRuntime]:
        requested_set = set(requested)
        if graph_target is None:
            projected = self._project_experiment_loop_runtimes(
                project_id,
                requested_set,
                _connection=connection,
            )
        else:
            newest_targets = self._newest_experiment_targets(
                project_id,
                requested_set,
                _connection=connection,
            )
            visible = {
                control_node_id
                for control_node_id, target in newest_targets.items()
                if target == graph_target
            }
            projected = (
                self._project_experiment_loop_runtimes(
                    project_id,
                    visible,
                    graph_target=graph_target,
                    _connection=connection,
                )
                if visible
                else {}
            )
        return {
            control_node_id: projected.get(control_node_id, ExperimentLoopRuntime())
            for control_node_id in requested
        }

    def _experiment_episode_projection_snapshot_in_connection(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        control_node_id: str,
        episode_id: str,
    ) -> ExperimentEpisodeProjectionSnapshot:
        episode_row = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if episode_row is None:
            raise ValueError("Experiment runtime identifies a missing durable episode.")
        episode = self._episode_record(episode_row)
        if (
            episode.project_id != project_id
            or episode.mode != "experiment_loop"
            or episode.control_node_id != control_node_id
        ):
            raise ValueError("Experiment runtime does not identify its exact durable episode.")
        task_rows = connection.execute(
            """
            SELECT graph_runs.*,
                   EXISTS (
                       SELECT 1 FROM graph_run_receipts AS receipt
                       WHERE receipt.operation_id = graph_runs.operation_id
                         AND receipt.category IN (
                             'experiment_recovery_abandoned',
                             'auto_research_recovery_abandoned'
                         )
                   ) AS recovery_abandoned
            FROM graph_runs
            WHERE episode_id = ? AND visible = 1 AND kind != 'episode_report'
            ORDER BY created_at, operation_id
            """,
            (episode_id,),
        ).fetchall()
        report_row = connection.execute(
            "SELECT * FROM episode_reports WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        usage = connection.execute(
            """
            SELECT
                COALESCE(SUM(agent_usage.processed_input_tokens), 0) AS input_tokens,
                COALESCE(SUM(agent_usage.generated_tokens), 0) AS generated_tokens
            FROM agent_usage
            JOIN graph_runs ON graph_runs.operation_id = agent_usage.operation_id
            WHERE graph_runs.episode_id = ?
              AND graph_runs.kind != 'episode_report'
              AND agent_usage.counted = 1
            """,
            (episode_id,),
        ).fetchone()
        assert usage is not None
        return ExperimentEpisodeProjectionSnapshot(
            episode=episode,
            tasks=[self._agent_task_record(row) for row in task_rows],
            budget=EpisodeBudgetMeter(
                invocation_ceiling=episode.invocation_ceiling,
                invocations_used=episode.invocations_used,
                invocations_remaining=episode.invocations_remaining,
                observed_input_tokens=int(usage["input_tokens"]),
                observed_generated_tokens=int(usage["generated_tokens"]),
            ),
            report=(self._episode_report_record(report_row) if report_row is not None else None),
        )

    def _newest_experiment_targets(
        self,
        project_id: str,
        requested: set[str] | None,
        *,
        _connection: sqlite3.Connection | None = None,
    ) -> dict[str, GraphTargetRef]:
        if _connection is None:
            with self.connection() as connection:
                return self._newest_experiment_targets(
                    project_id,
                    requested,
                    _connection=connection,
                )
        rows = _connection.execute(
            """
            SELECT control_node_id, graph_target_json
            FROM episodes
            WHERE project_id = ? AND mode = 'experiment_loop'
            ORDER BY created_at DESC, episode_id DESC
            """,
            (project_id,),
        ).fetchall()
        newest: dict[str, GraphTargetRef] = {}
        for row in rows:
            control_node_id = row["control_node_id"]
            if not isinstance(control_node_id, str) or not control_node_id:
                continue
            if requested is not None and control_node_id not in requested:
                continue
            newest.setdefault(
                control_node_id,
                GraphTargetRef.model_validate_json(row["graph_target_json"]),
            )
        return newest

    def project_experiment_loop_runtimes(
        self,
        project_id: str,
    ) -> dict[str, ExperimentLoopRuntime]:
        """Derive every current Experiment runtime without paging episode history."""

        return self._project_experiment_loop_runtimes(project_id, None)

    def _project_experiment_loop_runtimes(
        self,
        project_id: str,
        requested: set[str] | None,
        *,
        graph_target: GraphTargetRef | None = None,
        _connection: sqlite3.Connection | None = None,
    ) -> dict[str, ExperimentLoopRuntime]:
        """Load the generic parents plus mode ledgers and group them in memory."""

        if _connection is None:
            with self.connection() as connection:
                connection.execute("BEGIN")
                return self._project_experiment_loop_runtimes(
                    project_id,
                    requested,
                    graph_target=graph_target,
                    _connection=connection,
                )
        task_rows = _connection.execute(
            """
                SELECT operation_id, parent_operation_id, status, attempt, request_json,
                       created_at, phase, status_message, last_activity_at,
                       rowid AS storage_rowid
                FROM graph_runs
                WHERE project_id = ?
                  AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
                """,
            (project_id,),
        ).fetchall()
        receipt_rows = _connection.execute(
            """
                SELECT receipt.operation_id, receipt.category
                FROM graph_run_receipts AS receipt
                JOIN graph_runs AS task ON task.operation_id = receipt.operation_id
                WHERE task.project_id = ?
                  AND json_extract(task.request_json, '$.patch_kind') = 'experiment_loop'
                  AND receipt.category IN (
                      'experiment_loop_exit', 'experiment_recovery_abandoned'
                  )
                """,
            (project_id,),
        ).fetchall()
        watcher_rows = _connection.execute(
            """
                SELECT * FROM watchers
                WHERE project_id = ?
                  AND json_extract(continuation_json, '$.patch_kind') = 'experiment_loop'
                  AND notified = 0
                  AND status IN ('active', 'degraded', 'completed')
                """,
            (project_id,),
        ).fetchall()
        episode_rows = _connection.execute(
            """
                SELECT state.*, episode.project_id, episode.control_node_id,
                       episode.graph_target_json,
                       episode.stop_requested_at, episode.stop_settled_at
                FROM experiment_episode_state AS state
                JOIN episodes AS episode ON episode.episode_id = state.episode_id
                WHERE episode.project_id = ? AND episode.mode = 'experiment_loop'
                """,
            (project_id,),
        ).fetchall()
        parent_rows = _connection.execute(
            """
                SELECT * FROM episodes
                WHERE project_id = ? AND mode = 'experiment_loop'
                ORDER BY created_at DESC, episode_id DESC
                """,
            (project_id,),
        ).fetchall()

        tasks_by_control: dict[
            str,
            list[tuple[sqlite3.Row, dict[str, object]]],
        ] = {}
        for row in task_rows:
            request = json.loads(row["request_json"])
            control_node_id = request.get("control_node_id")
            if not isinstance(control_node_id, str) or not control_node_id:
                continue
            if requested is not None and control_node_id not in requested:
                continue
            tasks_by_control.setdefault(control_node_id, []).append((row, request))

        watchers_by_control: dict[str, list[StoredWatcherRecord]] = {}
        for row in watcher_rows:
            record = self._watcher_record(row)
            control_node_id = record.continuation.control_node_id
            if not control_node_id:
                continue
            if requested is not None and control_node_id not in requested:
                continue
            watchers_by_control.setdefault(control_node_id, []).append(record)

        receipt_categories: dict[str, set[str]] = {}
        for row in receipt_rows:
            receipt_categories.setdefault(str(row["operation_id"]), set()).add(str(row["category"]))
        episodes = {
            str(row["episode_id"]): self._experiment_episode_record(row) for row in episode_rows
        }
        parents_by_control: dict[str, EpisodeRecord | None] = {}
        for row in parent_rows:
            parent = self._episode_record(row)
            assert parent.control_node_id is not None
            if graph_target is not None and parent.graph_target != graph_target:
                continue
            parents_by_control.setdefault(parent.control_node_id, parent)
        control_node_ids = (
            set(tasks_by_control) | set(watchers_by_control) | set(parents_by_control)
            if requested is None
            else requested
        )
        return {
            control_node_id: self._derive_experiment_loop_runtime(
                tasks_by_control.get(control_node_id, []),
                watchers_by_control.get(control_node_id, []),
                receipt_categories,
                episodes,
                parents_by_control.get(control_node_id),
            )
            for control_node_id in control_node_ids
        }

    @classmethod
    def _derive_experiment_loop_runtime(
        cls,
        task_entries: list[tuple[sqlite3.Row, dict[str, object]]],
        watchers: list[StoredWatcherRecord],
        receipt_categories: dict[str, set[str]],
        episodes: dict[str, ExperimentEpisodeRecord],
        parent: EpisodeRecord | None,
    ) -> ExperimentLoopRuntime:
        """Purely derive one runtime from an already-loaded project ledger."""

        if parent is None:
            return ExperimentLoopRuntime()
        root_entries = [
            entry
            for entry in task_entries
            if entry[0]["parent_operation_id"] is None
            and entry[1].get("control_episode_id") == parent.episode_id
        ]
        if not root_entries:
            raise ValueError("Stored Experiment episode is missing its paid root task.")
        _, root_request = max(
            root_entries,
            key=lambda entry: (entry[0]["created_at"], entry[0]["storage_rowid"]),
        )
        episode_id = root_request.get("control_episode_id")
        if not isinstance(episode_id, str) or episode_id != parent.episode_id:
            raise ValueError("Stored experiment-loop root is missing its episode id.")
        try:
            uuid.UUID(episode_id)
        except ValueError as exc:
            raise ValueError("Stored experiment-loop root has an invalid episode id.") from exc

        episode_entries = [
            entry for entry in task_entries if entry[1].get("control_episode_id") == episode_id
        ]
        episode_entries.sort(
            key=lambda entry: (
                entry[0]["attempt"],
                entry[0]["created_at"],
                entry[0]["storage_rowid"],
            ),
            reverse=True,
        )
        episode = episodes.get(episode_id)
        compatible_watchers = [
            record
            for record in watchers
            if cls._experiment_watcher_matches_current(record, root_request, episode)
        ]
        latest_by_invocation: dict[
            int,
            tuple[sqlite3.Row, dict[str, object]],
        ] = {}
        for row, request in episode_entries:
            invocation = request.get("control_invocation")
            if isinstance(invocation, int) and invocation not in latest_by_invocation:
                latest_by_invocation[invocation] = (row, request)
        ceiling = root_request.get("control_invocation_ceiling")
        if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 1:
            raise ValueError("Stored experiment-loop root is missing its pinned ceiling.")
        if not latest_by_invocation or min(latest_by_invocation) < 1:
            raise ValueError("Stored experiment-loop root is missing its invocation number.")
        invocations_used = parent.invocations_used
        if set(latest_by_invocation) != set(range(1, invocations_used + 1)):
            raise ValueError("Stored experiment-loop root invocations are out of sequence.")
        if ceiling != parent.invocation_ceiling:
            raise ValueError("Stored experiment-loop root changed its parent invocation ceiling.")
        unresolved = any(
            row["status"] in {"queued", "running", "pausing", "paused", "failed", "interrupted"}
            and "experiment_recovery_abandoned"
            not in receipt_categories.get(str(row["operation_id"]), set())
            for row, _request in latest_by_invocation.values()
        )
        detached_work_active = any(
            record.status in {"active", "degraded"} and not record.notified
            for record in compatible_watchers
        )
        watcher_degraded = any(
            record.status == "degraded" and not record.notified for record in compatible_watchers
        )
        watcher_completion_pending = any(
            record.status == "completed" and not record.notified for record in compatible_watchers
        )
        has_watcher = detached_work_active or watcher_completion_pending
        episode_exited = parent.ending is not None or any(
            "experiment_loop_exit" in receipt_categories.get(str(row["operation_id"]), set())
            for row, _request in episode_entries
        )
        at_ceiling = invocations_used >= ceiling
        pins = root_request.get("control_decision_bundle")
        if not isinstance(pins, list):
            raise ValueError("Stored experiment-loop root is missing its pinned decision bundle.")
        control_revision = root_request.get("control_revision")
        if not isinstance(control_revision, int) or isinstance(control_revision, bool):
            raise ValueError("Stored experiment-loop root is missing its control revision.")
        completion_criteria = root_request.get("control_completion_criteria")
        if not isinstance(completion_criteria, list) or any(
            not isinstance(item, str) for item in completion_criteria
        ):
            raise ValueError("Stored experiment-loop root is missing its completion criteria.")
        current_row, current_request = max(
            episode_entries,
            key=lambda entry: (entry[0]["created_at"], entry[0]["storage_rowid"]),
        )
        binding_request = next(
            (
                request
                for row, request in episode_entries
                if episode is not None and row["operation_id"] == episode.last_turn_operation_id
            ),
            root_request,
        )
        current_invocation = current_request.get("control_invocation")
        return ExperimentLoopRuntime(
            episode_id=episode_id,
            invocations_used=invocations_used,
            invocation_ceiling=ceiling,
            control_revision=control_revision,
            task_active=unresolved,
            detached_work_active=detached_work_active,
            watcher_degraded=watcher_degraded,
            watcher_completion_pending=watcher_completion_pending,
            episode_exited=episode_exited,
            episode_live=parent.status in _LIVE_EPISODE_STATUSES,
            active=parent.status in {"running", "stopping"}
            and (
                unresolved
                or (
                    has_watcher
                    and not at_ceiling
                    and not episode_exited
                    and parent.stop_requested_at is None
                )
            ),
            paused=parent.status == "running"
            and at_ceiling
            and not unresolved
            and not episode_exited
            and parent.stop_requested_at is None,
            decision_bundle=pins,
            completion_criteria=completion_criteria,
            stop_requested=parent.stop_requested_at is not None,
            stop_settled=parent.stop_settled_at is not None,
            session_bound=episode is not None and episode.session_bound,
            session_diagnostic=episode.session_diagnostic if episode else None,
            provider=(episode.provider if episode is not None else None)
            or _optional_str(binding_request.get("provider")),
            model=(
                binding_request["model"] if isinstance(binding_request.get("model"), str) else None
            ),
            reasoning=_optional_str(binding_request.get("reasoning")),
            run_on=(episode.execution_machine if episode is not None else None)
            or _optional_str(binding_request.get("run_on")),
            execution_host=episode.execution_host if episode else None,
            run_truth_scope=(
                [str(item) for item in root_request["run_truth_scope"]]
                if isinstance(root_request.get("run_truth_scope"), list)
                else None
            ),
            chat_id=_optional_str(root_request.get("chat_id")),
            current_operation_id=current_row["operation_id"],
            current_status=current_row["status"],
            current_phase=current_row["phase"],
            current_status_message=current_row["status_message"],
            current_last_activity_at=current_row["last_activity_at"],
            current_invocation=(
                current_invocation if isinstance(current_invocation, int) else None
            ),
        )

    @staticmethod
    def _experiment_watcher_matches_current(
        record: StoredWatcherRecord,
        root_request: dict[str, object],
        episode: ExperimentEpisodeRecord | None,
    ) -> bool:
        """Whether this node-owned observer can wake the current episode.

        Conversation, provider, execution-machine alias, and package provenance
        are deliberately absent. The episode owns its session and policy; the
        watcher owns only the node, episode, and check execution host needed to
        answer the operational question.
        """

        continuation = record.continuation
        control_node_id = root_request.get("control_node_id")
        episode_matches = episode is not None and (
            record.project_id == episode.project_id
            and episode.control_node_id == control_node_id
            and record.graph_target == episode.graph_target
            and record.execution_host == episode.execution_host
        )
        return (
            continuation.patch_kind == "experiment_loop"
            and continuation.control_node_id == control_node_id
            and record.node_id == control_node_id
            and record.episode_id is not None
            and episode_matches
        )

    @staticmethod
    def _experiment_episode_root_request(
        connection: sqlite3.Connection,
        project_id: str,
        control_node_id: str,
        episode_id: str,
    ) -> dict[str, object] | None:
        row = connection.execute(
            """
            SELECT request_json FROM graph_runs
            WHERE project_id = ? AND parent_operation_id IS NULL
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_extract(request_json, '$.control_node_id') = ?
              AND json_extract(request_json, '$.control_episode_id') = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (project_id, control_node_id, episode_id),
        ).fetchone()
        return json.loads(row["request_json"]) if row is not None else None

    def experiment_watcher_compatible_with_episode(
        self,
        watcher_id: str,
        episode_id: str,
    ) -> bool:
        """Whether a stopped observer belonged to that episode operationally.

        Watcher origin remains immutable provenance. This derived relation lets
        a fresh post-stop Run stage compatible adopted observers as history even
        when an older invocation or episode originally armed them.
        """

        with self.connection() as connection:
            watcher_row = connection.execute(
                "SELECT * FROM watchers WHERE watcher_id = ?",
                (watcher_id,),
            ).fetchone()
            episode_row = self._experiment_episode_row(connection, episode_id)
            if watcher_row is None or episode_row is None:
                return False
            record = self._watcher_record(watcher_row)
            episode = self._experiment_episode_record(episode_row)
            root_request = self._experiment_episode_root_request(
                connection,
                episode.project_id,
                episode.control_node_id,
                episode_id,
            )
        return root_request is not None and self._experiment_watcher_matches_current(
            record,
            root_request,
            episode,
        )

    def active_experiment_control_ids(
        self,
        project_id: str,
        *,
        graph_target: GraphTargetRef | None = None,
    ) -> set[str]:
        """Return live controls, optionally only when the newest episode owns one target."""

        projected = self._project_experiment_loop_runtimes(
            project_id,
            None,
            graph_target=graph_target,
        )
        if graph_target is not None:
            newest_targets = self._newest_experiment_targets(project_id, None)
            projected = {
                control_node_id: runtime
                for control_node_id, runtime in projected.items()
                if newest_targets.get(control_node_id) == graph_target
            }
        return {control_node_id for control_node_id, runtime in projected.items() if runtime.active}

    def completed_experiment_watcher_group(
        self,
        project_id: str,
        control_node_id: str,
        *,
        graph_target: GraphTargetRef | None = None,
    ) -> list[StoredWatcherRecord] | None:
        """Return the oldest frozen group a human may reauthorize.

        Unlike automatic delivery, human reauthorization preserves the full
        watcher configuration, including model, reasoning, and package pointers.
        """

        target = graph_target or GraphTargetRef()
        with self.connection() as connection:
            units = self._ready_watcher_delivery_units(connection)
        groups: dict[tuple[object, ...], list[StoredWatcherRecord]] = {}
        for unit in units:
            first = unit[0]
            if (
                first.project_id != project_id
                or first.graph_target != target
                or first.continuation.patch_kind != "experiment_loop"
                or first.continuation.control_node_id != control_node_id
            ):
                continue
            key = (
                first.node_id,
                first.graph_target.key,
                first.execution_host,
                self._automatic_watcher_delivery_policy(first.continuation),
            )
            groups.setdefault(key, []).extend(unit)
        return next(iter(groups.values()), None)

    @staticmethod
    def _experiment_wake_is_stopped(
        connection: sqlite3.Connection,
        record: AgentTaskRecord,
    ) -> bool:
        """Refuse an automatic wake whose episode is stopping or already closed.

        The check runs inside the claim's own write transaction, so a claim either
        commits before the ending fence or finds it — there is no window where
        both win. A preserved completion may only enter a fresh episode through
        the human Run admission path.
        """

        request = record.request
        if request.get("patch_kind") != "experiment_loop" or request.get("trigger") != "watcher":
            return False
        episode_id = request.get("control_episode_id")
        if not isinstance(episode_id, str):
            return False
        row = connection.execute(
            "SELECT status, ending, stop_requested_at FROM episodes "
            "WHERE episode_id = ? AND mode = 'experiment_loop'",
            (episode_id,),
        ).fetchone()
        return row is not None and (
            row["status"] != "running"
            or row["ending"] is not None
            or row["stop_requested_at"] is not None
        )
