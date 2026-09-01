"""Durable Auto-research child routing, lifecycle notices, and command snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from typing import Literal

from rcp.limits import (
    AGENT_COMMAND_EVENT_MAX_BYTES,
    AUTO_RESEARCH_LIFECYCLE_MAX_NOTICES,
    AUTO_RESEARCH_MAIL_MAX_MESSAGES,
)
from rcp.providers import classify_terminal_error
from rcp.storage.models import (
    ACTIVE_AGENT_TASK_STATUSES,
    AgentCommandInvocationRecord,
    AgentTaskRecord,
    AutoResearchApplyResultRecord,
    AutoResearchChildAdmissionRecord,
    AutoResearchChildExperimentRecord,
    AutoResearchChildWorkRecord,
    AutoResearchCommandFileRecord,
    AutoResearchExperimentAllowance,
    AutoResearchExperimentAllowanceReached,
    AutoResearchFinishBlocker,
    AutoResearchFinishReceiptRecord,
    AutoResearchInboxReceiptRecord,
    AutoResearchLifecycleNoticeRecord,
    EpisodeInvocationCeilingReached,
    EpisodeNotRunning,
    EpisodeReportConflict,
    _required_timestamp,
)


class AutoResearchInboxClearTooLarge(ValueError):
    """The all-or-nothing Clear snapshot cannot fit its command response."""


class AutoResearchInboxHarvestTooLarge(ValueError):
    """The oldest pending notice cannot fit in a Harvest response."""


class AutoResearchInboxNoticeUnacknowledgeable(ValueError):
    """Neither Harvest nor the all-or-nothing Clear response can fit."""


def auto_research_inbox_projection(
    mode: Literal["harvest", "clear"],
    *,
    notice_ids: list[str],
    notices: list[AutoResearchLifecycleNoticeRecord],
) -> tuple[dict[str, object], str]:
    """Build the one inbox projection the orchestrator sees, with its message.

    The storage layer measures this against the response ceiling before it
    acknowledges anything, and the effect layer emits it. A second copy of the
    shape would let Clear's all-or-nothing bound measure bytes the client never
    receives.
    """

    count = len(notice_ids)
    if mode == "harvest":
        result: dict[str, object] = {
            "action": "harvest",
            "count": count,
            "notices": [
                {
                    "notice_id": notice.notice_id,
                    "source_kind": notice.source_kind,
                    "source_id": notice.source_id,
                    "source_event": notice.source_event,
                    "source_attempt": notice.source_attempt,
                    "payload": notice.payload,
                    "created_at": notice.created_at,
                }
                for notice in notices
            ],
        }
        message = f"Harvested and acknowledged {count} lifecycle notice(s)."
    else:
        result = {
            "action": "clear",
            "count": count,
            "notice_ids": list(notice_ids),
        }
        message = f"Cleared {count} lifecycle notice(s)."
    return result, message


def _auto_research_inbox_effect_fits(
    mode: Literal["harvest", "clear"],
    notices: list[AutoResearchLifecycleNoticeRecord],
) -> bool:
    """Measure the exact successful command projection used by the effect layer."""

    result, message = auto_research_inbox_projection(
        mode,
        notice_ids=[notice.notice_id for notice in notices],
        notices=notices,
    )
    try:
        exit_payload = json.dumps(
            {
                "status": "ok",
                "result": result,
                "diagnostic": message,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        response_payload = (
            json.dumps(
                {
                    "version": 1,
                    "request_id": "0" * 32,
                    "status": "ok",
                    "message": message,
                    "result": result,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        return False
    return max(len(exit_payload), len(response_payload)) <= AGENT_COMMAND_EVENT_MAX_BYTES


def _bounded_auto_research_lifecycle_notice(
    record: AutoResearchLifecycleNoticeRecord,
) -> AutoResearchLifecycleNoticeRecord:
    """Keep every new notice individually representable in a Harvest response."""

    if _auto_research_inbox_effect_fits("harvest", [record]):
        return record
    diagnostic = record.payload.get("diagnostic")
    if isinstance(diagnostic, str):
        payload = {
            **record.payload,
            "diagnostic": diagnostic[:2_000],
            "diagnostic_truncated": True,
        }
        bounded = record.model_copy(update={"payload": payload})
        if _auto_research_inbox_effect_fits("harvest", [bounded]):
            return bounded
    raise ValueError("a lifecycle notice exceeds the durable command response limit")


class AutoResearchChildrenStoreMixin:
    """Storage policy for ordinary children controlled by an Auto-research parent."""

    def record_auto_research_child_admission(
        self,
        record: AutoResearchChildAdmissionRecord,
    ) -> AutoResearchChildAdmissionRecord:
        """Persist a command admission before its child route is reflected."""

        if record.state != "accepted":
            raise ValueError("a new child admission must begin accepted")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert_auto_research_child_admission(connection, record)
        stored = self.auto_research_child_admission(record.admission_id)
        assert stored is not None
        return stored

    def _insert_auto_research_child_admission(
        self,
        connection: sqlite3.Connection,
        record: AutoResearchChildAdmissionRecord,
    ) -> AutoResearchChildAdmissionRecord:
        if record.state != "accepted":
            raise ValueError("a new child admission must begin accepted")
        episode = self._load_auto_research_episode(connection, record.episode_id)
        if episode.project_id != record.project_id:
            raise ValueError("the child admission belongs to another project")
        existing = connection.execute(
            "SELECT * FROM auto_research_child_admissions WHERE admission_id = ?",
            (record.admission_id,),
        ).fetchone()
        if existing is not None:
            stored = self._child_admission_record(existing)
            if stored != record:
                raise ValueError("the child admission identity already names another intent")
            return stored
        self._validate_auto_research_parent_admission(episode)
        connection.execute(
            """
            INSERT INTO auto_research_child_admissions (
                admission_id, episode_id, project_id, child_kind, child_id,
                state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.admission_id,
                record.episode_id,
                record.project_id,
                record.child_kind,
                record.child_id,
                record.state,
                record.created_at,
                record.updated_at,
            ),
        )
        return record

    def auto_research_child_admission(
        self,
        admission_id: str,
    ) -> AutoResearchChildAdmissionRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM auto_research_child_admissions WHERE admission_id = ?",
                (admission_id,),
            ).fetchone()
        return self._child_admission_record(row) if row is not None else None

    def pending_auto_research_child_admissions(
        self,
        episode_id: str | None = None,
    ) -> list[AutoResearchChildAdmissionRecord]:
        with self.connection() as connection:
            if episode_id is None:
                rows = connection.execute(
                    """
                    SELECT * FROM auto_research_child_admissions
                    WHERE state = 'accepted'
                    ORDER BY created_at, admission_id
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM auto_research_child_admissions
                    WHERE episode_id = ? AND state = 'accepted'
                    ORDER BY created_at, admission_id
                    """,
                    (episode_id,),
                ).fetchall()
        return [self._child_admission_record(row) for row in rows]

    def auto_research_child_admission_command(
        self,
        admission_id: str,
    ) -> AgentCommandInvocationRecord | None:
        """Find the one keyed command whose durable start created an admission.

        The admission and command start are inserted in one transaction, but the
        admission deliberately stores only child identity.  Recovery therefore
        matches the deterministic planned child id captured in the immutable
        command-start payload.  More than one match is corruption, not a choice a
        restart reconciler may guess about.
        """

        with self.connection() as connection:
            admission_row = connection.execute(
                "SELECT * FROM auto_research_child_admissions WHERE admission_id = ?",
                (admission_id,),
            ).fetchone()
            if admission_row is None:
                raise KeyError(admission_id)
            admission = self._child_admission_record(admission_row)
            verb = "spawn" if admission.child_kind == "work" else "episode"
            planned_field = (
                "planned_worker_id"
                if admission.child_kind == "work"
                else "planned_episode_effect_id"
            )
            rows = connection.execute(
                """
                SELECT DISTINCT command_id
                FROM graph_run_events
                WHERE event_kind = 'command' AND command_phase = 'start'
                  AND episode_id = ? AND command_verb = ?
                  AND idempotency_key IS NOT NULL
                ORDER BY event_id
                """,
                (admission.episode_id, verb),
            ).fetchall()
            matches: list[AgentCommandInvocationRecord] = []
            for row in rows:
                command = self._agent_command_from_connection(connection, row["command_id"])
                if command is None:
                    continue
                if command.start_payload.get(planned_field) == admission.child_id:
                    matches.append(command)
        if len(matches) > 1:
            raise RuntimeError(
                "an Auto-research child admission matches more than one keyed command"
            )
        return matches[0] if matches else None

    def cancel_auto_research_child_admission(
        self,
        admission_id: str,
    ) -> AutoResearchChildAdmissionRecord:
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM auto_research_child_admissions WHERE admission_id = ?",
                (admission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(admission_id)
            if row["state"] == "reflected":
                raise ValueError("a reflected child admission cannot be cancelled")
            connection.execute(
                """
                UPDATE auto_research_child_admissions
                SET state = 'cancelled', updated_at = ?
                WHERE admission_id = ? AND state = 'accepted'
                """,
                (now, admission_id),
            )
        stored = self.auto_research_child_admission(admission_id)
        assert stored is not None
        return stored

    def create_auto_research_child_work(
        self,
        record: AutoResearchChildWorkRecord,
        task: AgentTaskRecord,
        *,
        admission_id: str | None = None,
    ) -> tuple[AutoResearchChildWorkRecord, AgentTaskRecord]:
        """Atomically spend one B unit, insert ordinary Work, and route it to its parent."""

        if (
            record.root_operation_id != task.operation_id
            or record.current_operation_id != task.operation_id
            or record.episode_id != task.episode_id
            or record.project_id != task.project_id
            or task.kind != "node_chat"
            or task.status != "queued"
            or not task.visible
            or task.parent_operation_id is not None
            or task.request.get("mode") != "work"
            or task.request.get("trigger") != "orchestrator"
            or task.request.get("node_id") != record.control_node_id
        ):
            raise ValueError("an Auto-research child must be an ordinary queued node Work task")
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                episode = self._load_auto_research_episode(connection, record.episode_id)
                self._validate_auto_research_parent_admission(episode)
                if (
                    episode.project_id != record.project_id
                    or task.authorized_by != episode.authorized_by
                ):
                    raise ValueError("the child Work task changed its parent scope or authorizer")
                actor = connection.execute(
                    """
                    SELECT invocation.role
                    FROM auto_research_invocations AS invocation
                    JOIN graph_runs AS run ON run.operation_id = invocation.operation_id
                    WHERE invocation.episode_id = ? AND invocation.operation_id = ?
                      AND run.project_id = ?
                    """,
                    (
                        record.episode_id,
                        record.admitted_by_operation_id,
                        record.project_id,
                    ),
                ).fetchone()
                if actor is None or actor["role"] != "orchestrator":
                    raise ValueError("only the canonical orchestrator may admit ordinary Work")
                if episode.invocations_used >= episode.invocation_ceiling:
                    raise EpisodeInvocationCeilingReached(
                        "the Auto-research operational invocation ceiling is exhausted"
                    )
                if self._has_active_chat_overlap(connection, task):
                    raise ValueError("Another task is already active in this conversation.")
                self._insert_agent_task(connection, task, continuation_cause="fresh")
                connection.execute(
                    """
                    INSERT INTO auto_research_child_work (
                        worker_id, episode_id, project_id, control_node_id,
                        root_operation_id, current_operation_id, admitted_by_operation_id,
                        instruction, instruction_sha256, stop_requested_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.worker_id,
                        record.episode_id,
                        record.project_id,
                        record.control_node_id,
                        record.root_operation_id,
                        record.current_operation_id,
                        record.admitted_by_operation_id,
                        record.instruction,
                        record.instruction_sha256,
                        record.stop_requested_at,
                        record.created_at,
                        record.updated_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO auto_research_child_work_attempts (
                        operation_id, worker_id, allocation_operation_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (task.operation_id, record.worker_id, task.operation_id, task.created_at),
                )
                invocation_number = episode.invocations_used + 1
                connection.execute(
                    """
                    INSERT INTO episode_invocations (
                        episode_id, operation_id, invocation_number, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (record.episode_id, task.operation_id, invocation_number, task.created_at),
                )
                changed = connection.execute(
                    """
                    UPDATE episodes
                    SET invocations_used = ?, updated_at = ?
                    WHERE episode_id = ? AND invocations_used = ? AND status = 'running'
                      AND ending IS NULL AND stop_requested_at IS NULL
                    """,
                    (
                        invocation_number,
                        task.created_at,
                        record.episode_id,
                        episode.invocations_used,
                    ),
                )
                if changed.rowcount != 1:
                    raise ValueError("the episode budget changed during child admission")
                self._reflect_auto_research_child_admission(
                    connection,
                    admission_id=admission_id,
                    episode_id=record.episode_id,
                    child_kind="work",
                    child_id=record.worker_id,
                    updated_at=task.created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Could not create the Auto-research child Work task.") from exc
        stored = self.auto_research_child_work(record.worker_id)
        stored_task = self.agent_task(task.operation_id)
        assert stored is not None and stored_task is not None
        return stored, stored_task

    def create_auto_research_child_work_recovery(
        self,
        worker_id: str,
        task: AgentTaskRecord,
    ) -> tuple[AutoResearchChildWorkRecord, AgentTaskRecord]:
        """Insert one exact saved-session recovery without spending B again."""

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            route_row = connection.execute(
                "SELECT * FROM auto_research_child_work WHERE worker_id = ?",
                (worker_id,),
            ).fetchone()
            if route_row is None:
                raise KeyError(worker_id)
            route = self._child_work_record(route_row)
            episode = self._load_auto_research_episode(connection, route.episode_id)
            self._validate_auto_research_parent_admission(episode)
            current = connection.execute(
                "SELECT * FROM graph_runs WHERE operation_id = ?",
                (route.current_operation_id,),
            ).fetchone()
            if current is None:
                raise RuntimeError("the child Work route lost its current task")
            current_task = self._agent_task_record(current)
            current_attempt = connection.execute(
                """
                SELECT allocation_operation_id
                FROM auto_research_child_work_attempts
                WHERE operation_id = ? AND worker_id = ?
                """,
                (current_task.operation_id, worker_id),
            ).fetchone()
            if current_attempt is None:
                raise RuntimeError("the child Work route lost its current allocation")
            if current_task.status not in {"paused", "interrupted", "failed"}:
                raise ValueError("only a paused, interrupted, or failed child Work task can resume")
            if (
                task.status != "queued"
                or not task.visible
                or task.kind != "node_chat"
                or task.project_id != route.project_id
                or task.episode_id != route.episode_id
                or task.parent_operation_id != current_task.operation_id
                or task.attempt != current_task.attempt + 1
                or task.authorized_by != episode.authorized_by
                or not current_task.native_session_id
                or not current_task.stage_root
                or task.native_session_id != current_task.native_session_id
                or (task.stage_host or "") != (current_task.stage_host or "")
                or task.stage_root != current_task.stage_root
                or task.request.get("session_id") != current_task.native_session_id
                or task.request.get("chat_id") != current_task.request.get("chat_id")
            ):
                raise ValueError(
                    "child Work Resume must preserve its exact saved session and stage"
                )
            if self._has_active_chat_overlap(connection, task):
                raise ValueError("Another task is already active in this conversation.")
            self._insert_agent_task(connection, task, continuation_cause="resume")
            connection.execute(
                """
                INSERT INTO auto_research_child_work_attempts (
                    operation_id, worker_id, allocation_operation_id, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    task.operation_id,
                    worker_id,
                    current_attempt["allocation_operation_id"],
                    task.created_at,
                ),
            )
            changed = connection.execute(
                """
                UPDATE auto_research_child_work
                SET current_operation_id = ?, updated_at = ?
                WHERE worker_id = ? AND current_operation_id = ?
                """,
                (task.operation_id, task.created_at, worker_id, current_task.operation_id),
            )
            if changed.rowcount != 1:
                raise ValueError("the child Work recovery lineage changed during admission")
        stored = self.auto_research_child_work(worker_id)
        stored_task = self.agent_task(task.operation_id)
        assert stored is not None and stored_task is not None
        return stored, stored_task

    def create_auto_research_child_work_message_wake_task(
        self,
        record: AgentTaskRecord,
        *,
        worker_id: str,
        message_ids: list[str],
    ) -> AgentTaskRecord | None:
        """Spend B and bind one exact worker-mail prefix to an ordinary Work continuation."""

        if not worker_id or not message_ids or len(message_ids) != len(set(message_ids)):
            raise ValueError("a child Work message wake needs one worker and unique messages")
        if len(message_ids) > AUTO_RESEARCH_MAIL_MAX_MESSAGES:
            raise ValueError(
                "a child Work message wake may claim at most "
                f"{AUTO_RESEARCH_MAIL_MAX_MESSAGES} messages"
            )
        placeholders = ",".join("?" for _ in message_ids)
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                route_row = connection.execute(
                    "SELECT * FROM auto_research_child_work WHERE worker_id = ?",
                    (worker_id,),
                ).fetchone()
                if route_row is None:
                    raise KeyError(worker_id)
                route = self._child_work_record(route_row)
                episode = self._load_auto_research_episode(connection, route.episode_id)
                self._validate_auto_research_parent_admission(episode)
                if route.stop_requested_at is not None:
                    raise EpisodeNotRunning("the child Work route is stopping")
                messages = connection.execute(
                    f"""
                    SELECT message_id, episode_id, recipient_task_id,
                           delivered_at, delivery_operation_id
                    FROM auto_research_messages
                    WHERE message_id IN ({placeholders})
                    """,
                    message_ids,
                ).fetchall()
                if {str(item["message_id"]) for item in messages} != set(message_ids):
                    raise ValueError("child Work mail delivery names a missing message")
                if any(
                    item["episode_id"] != route.episode_id or item["recipient_task_id"] != worker_id
                    for item in messages
                ):
                    raise ValueError("child Work mail delivery crosses an episode or worker")
                if any(
                    item["delivered_at"] is not None or item["delivery_operation_id"] is not None
                    for item in messages
                ):
                    return None
                pending_prefix = connection.execute(
                    """
                    SELECT message_id FROM auto_research_messages
                    WHERE episode_id = ? AND recipient_task_id = ?
                      AND delivered_at IS NULL AND delivery_operation_id IS NULL
                    ORDER BY created_at, message_id LIMIT ?
                    """,
                    (route.episode_id, worker_id, len(message_ids)),
                ).fetchall()
                if [str(item["message_id"]) for item in pending_prefix] != message_ids:
                    return None
                current_row = connection.execute(
                    "SELECT * FROM graph_runs WHERE operation_id = ?",
                    (route.current_operation_id,),
                ).fetchone()
                if current_row is None:
                    raise RuntimeError("the child Work route lost its current task")
                current = self._agent_task_record(current_row)
                if current.status != "succeeded":
                    return None
                pinned_request_fields = (
                    "provider",
                    "model",
                    "reasoning",
                    "run_on",
                    "run_truth_scope",
                    "chat_scope",
                    "node_id",
                    "chat_id",
                    "mode",
                    "patch_kind",
                )
                if (
                    record.status != "queued"
                    or not record.visible
                    or record.kind != "node_chat"
                    or record.project_id != route.project_id
                    or record.episode_id != route.episode_id
                    or record.parent_operation_id != current.operation_id
                    or record.attempt != current.attempt + 1
                    or record.authorized_by != episode.authorized_by
                    or not current.native_session_id
                    or not current.stage_root
                    or record.native_session_id != current.native_session_id
                    or (record.stage_host or "") != (current.stage_host or "")
                    or record.stage_root != current.stage_root
                    or record.request.get("session_id") != current.native_session_id
                    or record.request.get("trigger") != "orchestrator"
                    or record.request.get("mode") != "work"
                    or record.request.get("patch_kind") != "work"
                    or record.request.get("message") is not None
                    or record.request.get("watcher_ids") not in (None, [])
                    or record.request.get("result_view") is not None
                    or any(
                        record.request.get(field) != current.request.get(field)
                        for field in pinned_request_fields
                    )
                ):
                    raise ValueError(
                        "child Work mail wake must preserve its exact saved Work session and scope"
                    )
                if self._has_active_chat_overlap(connection, record):
                    return None
                if episode.invocations_used >= episode.invocation_ceiling:
                    raise EpisodeInvocationCeilingReached(
                        "the Auto-research operational invocation ceiling is exhausted"
                    )
                self._insert_agent_task(connection, record, continuation_cause="message_wake")
                connection.execute(
                    """
                    INSERT INTO auto_research_child_work_attempts (
                        operation_id, worker_id, allocation_operation_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (record.operation_id, worker_id, record.operation_id, record.created_at),
                )
                invocation_number = episode.invocations_used + 1
                connection.execute(
                    """
                    INSERT INTO episode_invocations (
                        episode_id, operation_id, invocation_number, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        route.episode_id,
                        record.operation_id,
                        invocation_number,
                        record.created_at,
                    ),
                )
                budget_changed = connection.execute(
                    """
                    UPDATE episodes
                    SET invocations_used = ?, updated_at = ?
                    WHERE episode_id = ? AND invocations_used = ? AND status = 'running'
                      AND ending IS NULL AND stop_requested_at IS NULL
                    """,
                    (
                        invocation_number,
                        record.created_at,
                        route.episode_id,
                        episode.invocations_used,
                    ),
                ).rowcount
                route_changed = connection.execute(
                    """
                    UPDATE auto_research_child_work
                    SET current_operation_id = ?, updated_at = ?
                    WHERE worker_id = ? AND current_operation_id = ?
                      AND stop_requested_at IS NULL
                    """,
                    (
                        record.operation_id,
                        record.created_at,
                        worker_id,
                        current.operation_id,
                    ),
                ).rowcount
                if budget_changed != 1 or route_changed != 1:
                    raise ValueError("child Work mail admission changed during its transaction")
                claimed = connection.execute(
                    f"""
                    UPDATE auto_research_messages
                    SET delivered_at = ?, delivery_operation_id = ?
                    WHERE message_id IN ({placeholders})
                      AND delivered_at IS NULL AND delivery_operation_id IS NULL
                    """,
                    (record.created_at, record.operation_id, *message_ids),
                ).rowcount
                if claimed != len(message_ids):
                    raise ValueError("child Work mail changed during its delivery claim")
        except sqlite3.IntegrityError as exc:
            raise ValueError("Could not create the child Work message wake task.") from exc
        stored = self.agent_task(record.operation_id)
        assert stored is not None
        return stored

    def auto_research_child_work(self, worker_id: str) -> AutoResearchChildWorkRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM auto_research_child_work WHERE worker_id = ?",
                (worker_id,),
            ).fetchone()
        return self._child_work_record(row) if row is not None else None

    def auto_research_child_work_for_operation(
        self,
        operation_id: str,
    ) -> AutoResearchChildWorkRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT route.* FROM auto_research_child_work AS route
                JOIN auto_research_child_work_attempts AS attempt
                  ON attempt.worker_id = route.worker_id
                WHERE attempt.operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
        return self._child_work_record(row) if row is not None else None

    def auto_research_child_works(self, episode_id: str) -> list[AutoResearchChildWorkRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM auto_research_child_work
                WHERE episode_id = ? ORDER BY created_at, worker_id
                """,
                (episode_id,),
            ).fetchall()
        return [self._child_work_record(row) for row in rows]

    def request_auto_research_child_work_stop(
        self,
        worker_id: str,
    ) -> AutoResearchChildWorkRecord:
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT current_operation_id FROM auto_research_child_work WHERE worker_id = ?",
                (worker_id,),
            ).fetchone()
            if row is None:
                raise KeyError(worker_id)
            connection.execute(
                """
                UPDATE auto_research_child_work
                SET stop_requested_at = COALESCE(stop_requested_at, ?), updated_at = ?
                WHERE worker_id = ?
                """,
                (now, now, worker_id),
            )
        stored = self.auto_research_child_work(worker_id)
        assert stored is not None
        return stored

    def reserve_auto_research_experiment_replacement(
        self,
        record: AutoResearchChildExperimentRecord,
        *,
        admission_id: str | None = None,
    ) -> AutoResearchChildExperimentRecord:
        """Persist one fresh launch intent while the existing episode stops gracefully."""

        if record.state != "pending" or not record.replaces_episode_id:
            raise ValueError(
                "an Experiment replacement must begin pending and name its predecessor"
            )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_auto_research_experiment_route(connection, record)
            connection.execute(
                """
                INSERT INTO auto_research_child_experiments (
                    child_episode_id, auto_research_episode_id, project_id,
                    control_node_id, state, replaces_episode_id, request_json,
                    goal_sha256, parent_operation_id, terminal_diagnostic,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._child_experiment_values(record),
            )
            self._reflect_auto_research_child_admission(
                connection,
                admission_id=admission_id,
                episode_id=record.auto_research_episode_id,
                child_kind="experiment",
                child_id=record.child_episode_id,
                updated_at=record.updated_at,
            )
        stored = self.auto_research_child_experiment(record.child_episode_id)
        assert stored is not None
        return stored

    def auto_research_child_experiment(
        self,
        child_episode_id: str,
    ) -> AutoResearchChildExperimentRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM auto_research_child_experiments
                WHERE child_episode_id = ?
                """,
                (child_episode_id,),
            ).fetchone()
        return self._child_experiment_record(row) if row is not None else None

    def pending_auto_research_experiment_replacement(
        self,
        project_id: str,
        control_node_id: str,
    ) -> AutoResearchChildExperimentRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM auto_research_child_experiments
                WHERE project_id = ? AND control_node_id = ? AND state = 'pending'
                """,
                (project_id, control_node_id),
            ).fetchone()
        return self._child_experiment_record(row) if row is not None else None

    def auto_research_child_experiments(
        self,
        episode_id: str,
    ) -> list[AutoResearchChildExperimentRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM auto_research_child_experiments
                WHERE auto_research_episode_id = ?
                ORDER BY created_at, child_episode_id
                """,
                (episode_id,),
            ).fetchall()
        return [self._child_experiment_record(row) for row in rows]

    def cancel_auto_research_experiment_replacement(
        self,
        child_episode_id: str,
        *,
        diagnostic: str,
    ) -> AutoResearchChildExperimentRecord:
        return self._settle_auto_research_experiment_replacement(
            child_episode_id,
            source_event="cancelled",
            diagnostic=diagnostic,
        )

    def fail_auto_research_experiment_replacement(
        self,
        child_episode_id: str,
        *,
        diagnostic: str,
    ) -> AutoResearchChildExperimentRecord:
        """Terminalize a replacement that cannot start and notify its orchestrator."""

        return self._settle_auto_research_experiment_replacement(
            child_episode_id,
            source_event="failed",
            diagnostic=diagnostic,
        )

    def terminalize_auto_research_child_experiment(
        self,
        child_episode_id: str,
        *,
        diagnostic: str | None = None,
    ) -> AutoResearchChildExperimentRecord:
        return self._settle_auto_research_child_experiment(
            child_episode_id,
            from_state="running",
            to_state="terminal",
            diagnostic=diagnostic,
        )

    def auto_research_experiment_allowance(
        self,
        episode_id: str,
    ) -> AutoResearchExperimentAllowance:
        with self.connection() as connection:
            return self._auto_research_experiment_allowance(connection, episode_id)

    @staticmethod
    def _auto_research_experiment_allowance(
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> AutoResearchExperimentAllowance:
        row = connection.execute(
            """
            SELECT episode.invocation_ceiling,
                   COUNT(spend.operation_id) AS used
            FROM episodes AS episode
            JOIN auto_research_episodes AS auto ON auto.episode_id = episode.episode_id
            LEFT JOIN auto_research_experiment_invocations AS spend
              ON spend.auto_research_episode_id = episode.episode_id
            WHERE episode.episode_id = ? AND episode.mode = 'auto_research'
            GROUP BY episode.episode_id
            """,
            (episode_id,),
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        total = int(row["invocation_ceiling"]) * 5
        used = int(row["used"])
        return AutoResearchExperimentAllowance(total=total, used=used, remaining=total - used)

    def _claim_auto_research_experiment_allowance(
        self,
        connection: sqlite3.Connection,
        *,
        auto_research_episode_id: str,
        child_episode_id: str,
        operation_id: str,
        created_at: str,
    ) -> AutoResearchExperimentAllowance:
        existing = connection.execute(
            """
            SELECT auto_research_episode_id, child_episode_id
            FROM auto_research_experiment_invocations WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["auto_research_episode_id"] != auto_research_episode_id
                or existing["child_episode_id"] != child_episode_id
            ):
                raise ValueError("the Experiment allocation is already routed to another parent")
            return self._auto_research_experiment_allowance(connection, auto_research_episode_id)
        episode = self._load_auto_research_episode(connection, auto_research_episode_id)
        self._validate_auto_research_parent_admission(episode)
        route = connection.execute(
            """
            SELECT state FROM auto_research_child_experiments
            WHERE child_episode_id = ? AND auto_research_episode_id = ?
            """,
            (child_episode_id, auto_research_episode_id),
        ).fetchone()
        if route is None or route["state"] != "running":
            raise ValueError("the child Experiment invocation has no live parent route")
        allowance = self._auto_research_experiment_allowance(connection, auto_research_episode_id)
        if allowance.remaining == 0:
            raise AutoResearchExperimentAllowanceReached(allowance)
        connection.execute(
            """
            INSERT INTO auto_research_experiment_invocations (
                operation_id, auto_research_episode_id, child_episode_id, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (operation_id, auto_research_episode_id, child_episode_id, created_at),
        )
        return allowance.model_copy(
            update={"used": allowance.used + 1, "remaining": allowance.remaining - 1}
        )

    def _activate_auto_research_child_experiment(
        self,
        connection: sqlite3.Connection,
        record: AutoResearchChildExperimentRecord,
        *,
        admission_id: str | None,
    ) -> None:
        self._validate_auto_research_experiment_route(connection, record)
        existing = connection.execute(
            """
            SELECT * FROM auto_research_child_experiments WHERE child_episode_id = ?
            """,
            (record.child_episode_id,),
        ).fetchone()
        if existing is None:
            if record.state != "running":
                raise ValueError("a direct child Experiment launch must begin running")
            connection.execute(
                """
                INSERT INTO auto_research_child_experiments (
                    child_episode_id, auto_research_episode_id, project_id,
                    control_node_id, state, replaces_episode_id, request_json,
                    goal_sha256, parent_operation_id, terminal_diagnostic,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._child_experiment_values(record),
            )
        else:
            stored = self._child_experiment_record(existing)
            if stored.state != "pending" or not self._same_child_experiment_intent(stored, record):
                raise ValueError("the child Experiment route already names another launch")
            connection.execute(
                """
                UPDATE auto_research_child_experiments
                SET state = 'running', updated_at = ?
                WHERE child_episode_id = ? AND state = 'pending'
                """,
                (record.updated_at, record.child_episode_id),
            )
            assert stored.replaces_episode_id is not None
            self._insert_auto_research_lifecycle_notice(
                connection,
                AutoResearchLifecycleNoticeRecord(
                    notice_id=self._auto_research_notice_id(
                        record.auto_research_episode_id,
                        "experiment_replacement",
                        record.child_episode_id,
                        "advanced",
                        1,
                    ),
                    episode_id=record.auto_research_episode_id,
                    source_kind="experiment_replacement",
                    source_id=record.child_episode_id,
                    source_event="advanced",
                    source_attempt=1,
                    payload={
                        "episode_id": record.child_episode_id,
                        "status": "running",
                        "replaces_episode_id": stored.replaces_episode_id,
                    },
                    created_at=record.updated_at,
                ),
            )
        self._reflect_auto_research_child_admission(
            connection,
            admission_id=admission_id,
            episode_id=record.auto_research_episode_id,
            child_kind="experiment",
            child_id=record.child_episode_id,
            updated_at=record.updated_at,
        )

    def record_auto_research_lifecycle_notice(
        self,
        record: AutoResearchLifecycleNoticeRecord,
    ) -> AutoResearchLifecycleNoticeRecord:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stored = self._insert_auto_research_lifecycle_notice(connection, record)
        return stored

    def _insert_auto_research_lifecycle_notice(
        self,
        connection: sqlite3.Connection,
        record: AutoResearchLifecycleNoticeRecord,
    ) -> AutoResearchLifecycleNoticeRecord:
        if record.state != "pending":
            raise ValueError("a new lifecycle notice must begin pending")
        self._load_auto_research_episode(connection, record.episode_id)
        existing = connection.execute(
            """
            SELECT * FROM auto_research_lifecycle_notices
            WHERE episode_id = ? AND source_kind = ? AND source_id = ?
              AND source_event = ? AND source_attempt = ?
            """,
            (
                record.episode_id,
                record.source_kind,
                record.source_id,
                record.source_event,
                record.source_attempt,
            ),
        ).fetchone()
        if existing is not None:
            stored = self._lifecycle_notice_record(existing)
            if stored.payload == record.payload:
                return stored
            try:
                bounded = _bounded_auto_research_lifecycle_notice(record)
            except ValueError:
                pass
            else:
                if stored.payload == bounded.payload:
                    return stored
            raise ValueError("the lifecycle source event already has different facts")
        record = _bounded_auto_research_lifecycle_notice(record)
        payload_json = json.dumps(record.payload, sort_keys=True, separators=(",", ":"))
        connection.execute(
            """
            INSERT INTO auto_research_lifecycle_notices (
                notice_id, episode_id, source_kind, source_id, source_event,
                source_attempt, state, payload_json, created_at, delivered_at,
                delivery_operation_id, acknowledged_at, acknowledged_by
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL, NULL, NULL)
            """,
            (
                record.notice_id,
                record.episode_id,
                record.source_kind,
                record.source_id,
                record.source_event,
                record.source_attempt,
                payload_json,
                record.created_at,
            ),
        )
        return record

    def _insert_auto_research_task_lifecycle_notice(
        self,
        connection: sqlite3.Connection,
        *,
        operation_id: str,
        status: str,
        created_at: str,
        diagnostic: str | None = None,
    ) -> AutoResearchLifecycleNoticeRecord | None:
        """Emit one routed task transition inside the task's own transaction."""

        task = connection.execute(
            """
            SELECT operation_id, episode_id, attempt, native_session_id, stage_root
            FROM graph_runs WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if task is None:
            raise KeyError(operation_id)
        work = connection.execute(
            """
            SELECT route.episode_id, route.worker_id, route.stop_requested_at
            FROM auto_research_child_work_attempts AS attempt
            JOIN auto_research_child_work AS route ON route.worker_id = attempt.worker_id
            WHERE attempt.operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        source_kind: str
        source_id: str
        parent_episode_id: str
        payload: dict[str, object]
        if work is not None:
            source_kind = "worker"
            source_id = str(work["worker_id"])
            parent_episode_id = str(work["episode_id"])
            payload = {
                "kind": "work",
                "worker_id": source_id,
                "operation_id": operation_id,
                "status": status,
            }
        else:
            child_episode_id = task["episode_id"]
            if not isinstance(child_episode_id, str):
                return None
            experiment = connection.execute(
                """
                SELECT route.auto_research_episode_id, route.child_episode_id
                FROM auto_research_child_experiments AS route
                JOIN episodes AS child ON child.episode_id = route.child_episode_id
                WHERE route.child_episode_id = ? AND route.state = 'running'
                """,
                (child_episode_id,),
            ).fetchone()
            if experiment is None:
                return None
            if status == "succeeded":
                # Successful Experiment turns continue through their own bound
                # watcher/ending path; only attention and terminal state route upward.
                return None
            source_kind = "experiment_task"
            source_id = operation_id
            parent_episode_id = str(experiment["auto_research_episode_id"])
            payload = {
                "kind": "experiment",
                "episode_id": str(experiment["child_episode_id"]),
                "operation_id": operation_id,
                "status": status,
            }
        if diagnostic:
            payload["diagnostic"] = diagnostic
        if status in {"paused", "failed", "interrupted"}:
            unavailable = (
                status != "paused" and classify_terminal_error(diagnostic or "") == "session_limit"
            )
            receipt_rows = connection.execute(
                """
                SELECT category, payload_json FROM graph_run_receipts
                WHERE operation_id = ?
                  AND category IN (
                    'provider_terminal_error', 'continuation_context_unavailable'
                  )
                ORDER BY rowid
                """,
                (operation_id,),
            ).fetchall()
            for receipt in receipt_rows:
                try:
                    receipt_payload = json.loads(receipt["payload_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if (
                    receipt["category"] == "provider_terminal_error"
                    and receipt_payload.get("classification") == "session_limit"
                ) or (
                    receipt["category"] == "continuation_context_unavailable"
                    and receipt_payload.get("retry_required") is True
                ):
                    unavailable = True
            stopped = bool(work is not None and work["stop_requested_at"] is not None)
            resume_available = bool(
                task["native_session_id"] and task["stage_root"] and not unavailable and not stopped
            )
            payload["resume_available"] = resume_available
            if not resume_available:
                payload["replacement_command"] = (
                    "spawn" if source_kind == "worker" else "episode --kick-off-experiment"
                )
        attempt = max(1, int(task["attempt"]))
        notice = AutoResearchLifecycleNoticeRecord(
            notice_id=self._auto_research_notice_id(
                parent_episode_id,
                source_kind,
                source_id,
                status,
                attempt,
            ),
            episode_id=parent_episode_id,
            source_kind=source_kind,
            source_id=source_id,
            source_event=status,
            source_attempt=attempt,
            payload=payload,
            created_at=created_at,
        )
        return self._insert_auto_research_lifecycle_notice(connection, notice)

    def _terminalize_auto_research_child_experiment_with_notice(
        self,
        connection: sqlite3.Connection,
        *,
        child_episode_id: str,
        status: str,
        ending: str,
        diagnostic: str | None,
        created_at: str,
    ) -> AutoResearchLifecycleNoticeRecord | None:
        """Close a routed child and emit its terminal fact in the same transaction."""

        route = connection.execute(
            """
            SELECT * FROM auto_research_child_experiments WHERE child_episode_id = ?
            """,
            (child_episode_id,),
        ).fetchone()
        if route is None:
            return None
        if route["state"] not in {"running", "terminal"}:
            raise ValueError("only a running child Experiment can become terminal")
        if route["state"] == "running":
            connection.execute(
                """
                UPDATE auto_research_child_experiments
                SET state = 'terminal', terminal_diagnostic = ?, updated_at = ?
                WHERE child_episode_id = ? AND state = 'running'
                """,
                (diagnostic, created_at, child_episode_id),
            )
        source_event = {
            "completed": "completed",
            "exhausted": "exhausted",
            "human_pause": "needs_action",
            "failed": "failed",
            "stopped": "stopped",
        }.get(ending, status)
        payload: dict[str, object] = {
            "episode_id": child_episode_id,
            "status": status,
            "ending": ending,
        }
        if diagnostic:
            payload["diagnostic"] = diagnostic
        parent_episode_id = str(route["auto_research_episode_id"])
        notice = AutoResearchLifecycleNoticeRecord(
            notice_id=self._auto_research_notice_id(
                parent_episode_id,
                "experiment_episode",
                child_episode_id,
                source_event,
                1,
            ),
            episode_id=parent_episode_id,
            source_kind="experiment_episode",
            source_id=child_episode_id,
            source_event=source_event,
            source_attempt=1,
            payload=payload,
            created_at=created_at,
        )
        return self._insert_auto_research_lifecycle_notice(connection, notice)

    @staticmethod
    def _auto_research_notice_id(
        episode_id: str,
        source_kind: str,
        source_id: str,
        source_event: str,
        source_attempt: int,
    ) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                ":".join(
                    (
                        "rcp",
                        "auto-research-lifecycle",
                        episode_id,
                        source_kind,
                        source_id,
                        source_event,
                        str(source_attempt),
                    )
                ),
            )
        )

    def pending_auto_research_lifecycle_notices(
        self,
        episode_id: str,
        *,
        limit: int = 50,
    ) -> list[AutoResearchLifecycleNoticeRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM auto_research_lifecycle_notices
                WHERE episode_id = ? AND delivered_at IS NULL AND acknowledged_at IS NULL
                ORDER BY created_at, notice_id LIMIT ?
                """,
                (episode_id, max(1, min(limit, AUTO_RESEARCH_LIFECYCLE_MAX_NOTICES))),
            ).fetchall()
        return [self._lifecycle_notice_record(row) for row in rows]

    def pending_auto_research_lifecycle_episode_ids(
        self,
        episode_id: str | None = None,
    ) -> list[str]:
        """Return wake-eligible parents that have an undelivered lifecycle notice."""

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT episode.episode_id
                FROM episodes AS episode
                JOIN auto_research_lifecycle_notices AS notice
                  ON notice.episode_id = episode.episode_id
                WHERE episode.mode = 'auto_research'
                  AND episode.status = 'running'
                  AND episode.ending IS NULL
                  AND episode.stop_requested_at IS NULL
                  AND notice.delivered_at IS NULL
                  AND notice.acknowledged_at IS NULL
                  AND (? IS NULL OR episode.episode_id = ?)
                ORDER BY episode.episode_id
                """,
                (episode_id, episode_id),
            ).fetchall()
        return [str(row["episode_id"]) for row in rows]

    def claim_auto_research_lifecycle_notices(
        self,
        episode_id: str,
        operation_id: str,
        *,
        limit: int = 50,
    ) -> list[AutoResearchLifecycleNoticeRecord]:
        """Bind one pending prefix to an admitted wake; exact recovery reuses that claim."""

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = self._claim_auto_research_lifecycle_notices(
                connection,
                episode_id=episode_id,
                operation_id=operation_id,
                delivered_at=now,
                limit=limit,
            )
        return rows

    def _claim_auto_research_lifecycle_notices(
        self,
        connection: sqlite3.Connection,
        *,
        episode_id: str,
        operation_id: str,
        delivered_at: str,
        limit: int,
    ) -> list[AutoResearchLifecycleNoticeRecord]:
        rows = connection.execute(
            """
            SELECT * FROM auto_research_lifecycle_notices
            WHERE episode_id = ? AND delivered_at IS NULL AND acknowledged_at IS NULL
            ORDER BY created_at, notice_id LIMIT ?
            """,
            (episode_id, max(1, min(limit, AUTO_RESEARCH_LIFECYCLE_MAX_NOTICES))),
        ).fetchall()
        if not rows:
            return []
        ids = [str(row["notice_id"]) for row in rows]
        placeholders = ",".join("?" for _ in ids)
        connection.execute(
            f"""
            UPDATE auto_research_lifecycle_notices
            SET state = 'delivered', delivered_at = ?, delivery_operation_id = ?
            WHERE notice_id IN ({placeholders})
              AND delivered_at IS NULL AND acknowledged_at IS NULL
            """,
            (delivered_at, operation_id, *ids),
        )
        return [
            notice.model_copy(
                update={
                    "state": "delivered",
                    "delivered_at": delivered_at,
                    "delivery_operation_id": operation_id,
                }
            )
            for notice in (self._lifecycle_notice_record(row) for row in rows)
        ]

    def harvest_auto_research_lifecycle_notices(
        self,
        episode_id: str,
        *,
        acknowledged_by: str,
        limit: int = 50,
    ) -> list[AutoResearchLifecycleNoticeRecord]:
        if not acknowledged_by.strip():
            raise ValueError("lifecycle harvest requires its acknowledging actor")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM auto_research_lifecycle_notices
                WHERE episode_id = ? AND delivered_at IS NULL AND acknowledged_at IS NULL
                ORDER BY created_at, notice_id LIMIT ?
                """,
                (episode_id, max(1, min(limit, AUTO_RESEARCH_LIFECYCLE_MAX_NOTICES))),
            ).fetchall()
            notices = [self._lifecycle_notice_record(row) for row in rows]
            self._acknowledge_lifecycle_rows(
                connection,
                notices,
                acknowledged_at=now,
                acknowledged_by=acknowledged_by,
            )
        return [
            notice.model_copy(
                update={
                    "state": "acknowledged",
                    "acknowledged_at": now,
                    "acknowledged_by": acknowledged_by,
                }
            )
            for notice in notices
        ]

    def clear_auto_research_lifecycle_notices(
        self,
        episode_id: str,
        *,
        acknowledged_by: str,
    ) -> list[str]:
        """Acknowledge exactly the pending snapshot and return no notice bodies."""

        if not acknowledged_by.strip():
            raise ValueError("lifecycle clear requires its acknowledging actor")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM auto_research_lifecycle_notices
                WHERE episode_id = ? AND delivered_at IS NULL AND acknowledged_at IS NULL
                ORDER BY created_at, notice_id
                """,
                (episode_id,),
            ).fetchall()
            notices = [self._lifecycle_notice_record(row) for row in rows]
            self._acknowledge_lifecycle_rows(
                connection,
                notices,
                acknowledged_at=now,
                acknowledged_by=acknowledged_by,
            )
        return [notice.notice_id for notice in notices]

    def process_auto_research_lifecycle_inbox(
        self,
        episode_id: str,
        *,
        effect_id: str,
        mode: Literal["harvest", "clear"],
        acknowledged_by: str,
        limit: int = 50,
    ) -> AutoResearchInboxReceiptRecord:
        """Acknowledge one exact snapshot and durably bind it to a keyed effect."""

        if not effect_id.strip():
            raise ValueError("a lifecycle inbox effect requires its durable id")
        if mode not in {"harvest", "clear"}:
            raise ValueError("a lifecycle inbox effect must harvest or clear")
        if not acknowledged_by.strip():
            raise ValueError("a lifecycle inbox effect requires its acknowledging actor")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM auto_research_inbox_receipts WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()
            if existing is not None:
                stored = self._inbox_receipt_record(existing)
                if (
                    stored.episode_id != episode_id
                    or stored.mode != mode
                    or stored.acknowledged_by != acknowledged_by
                ):
                    raise ValueError("the lifecycle inbox effect id already names another command")
                return stored
            self._load_auto_research_episode(connection, episode_id)
            sql = """
                SELECT * FROM auto_research_lifecycle_notices
                WHERE episode_id = ? AND delivered_at IS NULL AND acknowledged_at IS NULL
                ORDER BY created_at, notice_id
            """
            parameters: tuple[object, ...] = (episode_id,)
            if mode == "harvest":
                sql += " LIMIT ?"
                parameters = (
                    episode_id,
                    max(1, min(limit, AUTO_RESEARCH_LIFECYCLE_MAX_NOTICES)),
                )
            rows = connection.execute(sql, parameters).fetchall()
            pending = [self._lifecycle_notice_record(row) for row in rows]
            clear_fits = None
            if mode == "harvest" and pending:
                acknowledged_first = pending[0].model_copy(
                    update={
                        "state": "acknowledged",
                        "acknowledged_at": now,
                        "acknowledged_by": acknowledged_by,
                    }
                )
                if not _auto_research_inbox_effect_fits("harvest", [acknowledged_first]):
                    clear_rows = connection.execute(
                        """
                        SELECT * FROM auto_research_lifecycle_notices
                        WHERE episode_id = ?
                          AND delivered_at IS NULL AND acknowledged_at IS NULL
                        ORDER BY created_at, notice_id
                        """,
                        (episode_id,),
                    ).fetchall()
                    clear_snapshot = [
                        self._lifecycle_notice_record(row).model_copy(
                            update={
                                "state": "acknowledged",
                                "acknowledged_at": now,
                                "acknowledged_by": acknowledged_by,
                            }
                        )
                        for row in clear_rows
                    ]
                    clear_fits = _auto_research_inbox_effect_fits("clear", clear_snapshot)
            notices = self._bounded_auto_research_inbox_snapshot(
                pending,
                mode=mode,
                acknowledged_at=now,
                acknowledged_by=acknowledged_by,
                clear_fits=clear_fits,
            )
            self._acknowledge_lifecycle_rows(
                connection,
                notices,
                acknowledged_at=now,
                acknowledged_by=acknowledged_by,
            )
            acknowledged = [
                notice.model_copy(
                    update={
                        "state": "acknowledged",
                        "acknowledged_at": now,
                        "acknowledged_by": acknowledged_by,
                    }
                )
                for notice in notices
            ]
            receipt = AutoResearchInboxReceiptRecord(
                effect_id=effect_id,
                episode_id=episode_id,
                mode=mode,
                notice_ids=[notice.notice_id for notice in acknowledged],
                count=len(acknowledged),
                notices=acknowledged if mode == "harvest" else [],
                acknowledged_by=acknowledged_by,
                created_at=now,
            )
            result_json = json.dumps(
                {
                    "notice_ids": receipt.notice_ids,
                    "count": receipt.count,
                    "notices": [notice.model_dump(mode="json") for notice in receipt.notices],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            connection.execute(
                """
                INSERT INTO auto_research_inbox_receipts (
                    effect_id, episode_id, mode, result_json, acknowledged_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    effect_id,
                    episode_id,
                    mode,
                    result_json,
                    acknowledged_by,
                    now,
                ),
            )
        return receipt

    @staticmethod
    def _bounded_auto_research_inbox_snapshot(
        pending: list[AutoResearchLifecycleNoticeRecord],
        *,
        mode: Literal["harvest", "clear"],
        acknowledged_at: str,
        acknowledged_by: str,
        clear_fits: bool | None = None,
    ) -> list[AutoResearchLifecycleNoticeRecord]:
        """Choose the exact snapshot whose command response is durable.

        The response ledger is smaller than the lifecycle handoff.  Selection
        therefore happens before acknowledgment, inside the caller's immediate
        transaction. Harvest takes an ordered bounded prefix. Clear remains
        all-or-nothing and refuses before mutation when its compact full snapshot
        cannot fit.
        """

        acknowledged = [
            notice.model_copy(
                update={
                    "state": "acknowledged",
                    "acknowledged_at": acknowledged_at,
                    "acknowledged_by": acknowledged_by,
                }
            )
            for notice in pending
        ]
        if mode == "clear":
            if _auto_research_inbox_effect_fits(mode, acknowledged):
                return acknowledged
            raise AutoResearchInboxClearTooLarge(
                "Clear would exceed the durable command response limit, so no lifecycle "
                "notices were acknowledged; run inbox --harvest with a new key before "
                "running inbox --clear with another new key."
            )

        selected: list[AutoResearchLifecycleNoticeRecord] = []
        for notice in acknowledged:
            candidate = [*selected, notice]
            if not _auto_research_inbox_effect_fits(mode, candidate):
                break
            selected = candidate
        if acknowledged and not selected:
            if clear_fits is not True:
                raise AutoResearchInboxNoticeUnacknowledgeable(
                    "The oldest lifecycle notice cannot fit in Harvest, and the complete "
                    "Clear response also exceeds the durable command response limit; no "
                    "lifecycle notices were acknowledged."
                )
            raise AutoResearchInboxHarvestTooLarge(
                "Harvest could not acknowledge the oldest lifecycle notice because its body "
                "exceeds the durable command response limit; run inbox --key <new-key> "
                "--clear to acknowledge it without returning the body."
            )
        if not _auto_research_inbox_effect_fits(mode, selected):
            raise RuntimeError("the empty lifecycle inbox response exceeds its durable limit")
        return selected

    def auto_research_lifecycle_notices(
        self,
        episode_id: str,
    ) -> list[AutoResearchLifecycleNoticeRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM auto_research_lifecycle_notices
                WHERE episode_id = ? ORDER BY created_at, notice_id
                """,
                (episode_id,),
            ).fetchall()
        return [self._lifecycle_notice_record(row) for row in rows]

    def auto_research_lifecycle_delivery(
        self,
        operation_id: str,
    ) -> list[AutoResearchLifecycleNoticeRecord]:
        """Return only lifecycle facts durably bound to one wake allocation."""

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM auto_research_lifecycle_notices
                WHERE delivery_operation_id = ? ORDER BY created_at, notice_id
                """,
                (operation_id,),
            ).fetchall()
        return [self._lifecycle_notice_record(row) for row in rows]

    def auto_research_inbox_receipt(
        self,
        effect_id: str,
    ) -> AutoResearchInboxReceiptRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM auto_research_inbox_receipts WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()
        return self._inbox_receipt_record(row) if row is not None else None

    def save_auto_research_apply_result(
        self,
        record: AutoResearchApplyResultRecord,
    ) -> AutoResearchApplyResultRecord:
        result_json = json.dumps(record.result, sort_keys=True, separators=(",", ":"))
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute(
                "SELECT episode_id FROM graph_runs WHERE operation_id = ?",
                (record.operation_id,),
            ).fetchone()
            if task is None:
                raise KeyError(record.operation_id)
            if task["episode_id"] != record.episode_id:
                raise ValueError("the Apply result does not match its task episode")
            self._load_auto_research_episode(connection, record.episode_id)
            existing = connection.execute(
                "SELECT * FROM auto_research_apply_results WHERE apply_id = ?",
                (record.apply_id,),
            ).fetchone()
            if existing is not None:
                stored = self._apply_result_record(existing)
                if (
                    stored.episode_id != record.episode_id
                    or stored.operation_id != record.operation_id
                    or stored.patch_sha256 != record.patch_sha256
                ):
                    raise ValueError("the Apply identity already has another durable result")
                return stored
            connection.execute(
                """
                INSERT INTO auto_research_apply_results (
                    apply_id, episode_id, operation_id, patch_sha256,
                    result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.apply_id,
                    record.episode_id,
                    record.operation_id,
                    record.patch_sha256,
                    result_json,
                    record.created_at,
                ),
            )
        return record

    def auto_research_apply_result(
        self,
        apply_id: str,
    ) -> AutoResearchApplyResultRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM auto_research_apply_results WHERE apply_id = ?",
                (apply_id,),
            ).fetchone()
        return self._apply_result_record(row) if row is not None else None

    def auto_research_apply_results(
        self,
        operation_id: str,
    ) -> list[AutoResearchApplyResultRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM auto_research_apply_results
                WHERE operation_id = ? ORDER BY rowid
                """,
                (operation_id,),
            ).fetchall()
        return [self._apply_result_record(row) for row in rows]

    def auto_research_command_file(
        self,
        command_id: str,
    ) -> AutoResearchCommandFileRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM auto_research_command_files WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        return self._command_file_record(row) if row is not None else None

    @staticmethod
    def _insert_auto_research_command_file(
        connection: sqlite3.Connection,
        record: AutoResearchCommandFileRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO auto_research_command_files (
                command_id, episode_id, operation_id, kind, filename,
                sha256, content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.command_id,
                record.episode_id,
                record.operation_id,
                record.kind,
                record.filename,
                record.sha256,
                record.content,
                record.created_at,
            ),
        )

    def auto_research_finish_blockers(
        self,
        episode_id: str,
    ) -> list[AutoResearchFinishBlocker]:
        """Return every current obligation without mutating or settling any of them."""

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._auto_research_finish_blockers(connection, episode_id)

    def guard_auto_research_finish(
        self,
        episode_id: str,
        *,
        effect_id: str,
        actor_operation_id: str,
        diagnostic: str | None = None,
    ) -> AutoResearchFinishReceiptRecord:
        """Snapshot every blocker or fence completion, atomically and exactly once."""

        if not effect_id.strip():
            raise ValueError("a guarded-Finish effect requires its durable id")
        if not actor_operation_id.strip():
            raise ValueError("a guarded-Finish effect requires its canonical actor")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM auto_research_finish_receipts WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()
            if existing is not None:
                receipt = self._finish_receipt_record(existing)
                if (
                    receipt.episode_id != episode_id
                    or receipt.actor_operation_id != actor_operation_id
                ):
                    raise ValueError("the guarded-Finish effect id already names another command")
                return receipt
            episode = self._load_auto_research_episode(connection, episode_id)
            if episode.ending is not None:
                if episode.ending != "completed" or episode.ending_diagnostic != diagnostic:
                    raise EpisodeReportConflict("the episode ending fence is immutable")
                self._settle_auto_research_watchers_in_connection(
                    connection,
                    episode_id=episode_id,
                    now=now,
                )
                result: dict[str, object] = {
                    "episode_id": episode.episode_id,
                    "status": episode.status,
                    "ending": episode.ending,
                }
                disposition: Literal["blocked", "completed"] = "completed"
                blocker_count = 0
            else:
                if episode.stop_requested_at is not None or episode.status == "stopping":
                    raise EpisodeNotRunning("Stop already fenced this episode")
                if episode.status not in {"queued", "running"}:
                    raise EpisodeNotRunning("the episode can no longer accept an ending fence")
                blockers = self._auto_research_finish_blockers(connection, episode_id)
                if blockers:
                    result = {
                        "episode_id": episode.episode_id,
                        "blockers": [item.model_dump(mode="json") for item in blockers],
                    }
                    disposition = "blocked"
                    blocker_count = len(blockers)
                else:
                    changed = connection.execute(
                        """
                        UPDATE episodes
                        SET status = 'wrapping_up', ending = 'completed',
                            ending_diagnostic = ?, updated_at = ?
                        WHERE episode_id = ? AND ending IS NULL
                          AND stop_requested_at IS NULL AND status IN ('queued', 'running')
                        """,
                        (diagnostic, now, episode_id),
                    ).rowcount
                    if changed != 1:
                        raise EpisodeNotRunning("the episode changed while completion was fenced")
                    self._settle_auto_research_watchers_in_connection(
                        connection,
                        episode_id=episode_id,
                        now=now,
                    )
                    row = connection.execute(
                        "SELECT * FROM episodes WHERE episode_id = ?",
                        (episode_id,),
                    ).fetchone()
                    assert row is not None
                    episode = self._episode_record(row)
                    result = {
                        "episode_id": episode.episode_id,
                        "status": episode.status,
                        "ending": episode.ending,
                    }
                    disposition = "completed"
                    blocker_count = 0
            result_json = json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            receipt = AutoResearchFinishReceiptRecord(
                effect_id=effect_id,
                episode_id=episode_id,
                actor_operation_id=actor_operation_id,
                disposition=disposition,
                blocker_count=blocker_count,
                result=result,
                result_sha256=hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
                created_at=now,
            )
            connection.execute(
                """
                INSERT INTO auto_research_finish_receipts (
                    effect_id, episode_id, actor_operation_id, disposition,
                    blocker_count, result_json, result_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.effect_id,
                    receipt.episode_id,
                    receipt.actor_operation_id,
                    receipt.disposition,
                    receipt.blocker_count,
                    result_json,
                    receipt.result_sha256,
                    receipt.created_at,
                ),
            )
        return receipt

    def auto_research_finish_receipt(
        self,
        effect_id: str,
    ) -> AutoResearchFinishReceiptRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM auto_research_finish_receipts WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()
        return self._finish_receipt_record(row) if row is not None else None

    def _auto_research_finish_blockers(
        self,
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> list[AutoResearchFinishBlocker]:
        self._load_auto_research_episode(connection, episode_id)
        blockers: list[AutoResearchFinishBlocker] = []
        # A worker blocks Finish for exactly as long as it is an active task.
        # Restating that set as a SQL literal is how the copies of it in this
        # repository drifted before.
        active_statuses = sorted(ACTIVE_AGENT_TASK_STATUSES)
        workers = connection.execute(
            f"""
            SELECT route.worker_id, run.status
            FROM auto_research_child_work AS route
            JOIN graph_runs AS run ON run.operation_id = route.current_operation_id
            WHERE route.episode_id = ?
              AND run.status IN ({",".join("?" * len(active_statuses))})
            ORDER BY route.created_at, route.worker_id
            """,
            (episode_id, *active_statuses),
        ).fetchall()
        blockers.extend(
            AutoResearchFinishBlocker(
                kind="spawned_work",
                blocker_id=str(row["worker_id"]),
                state=str(row["status"]),
                action=f"stop --key <key> {row['worker_id']}",
            )
            for row in workers
        )
        experiments = connection.execute(
            """
            SELECT route.child_episode_id, child.status
            FROM auto_research_child_experiments AS route
            JOIN episodes AS child ON child.episode_id = route.child_episode_id
            WHERE route.auto_research_episode_id = ? AND route.state = 'running'
              AND child.status IN ('queued', 'running', 'stopping', 'wrapping_up')
            ORDER BY route.created_at, route.child_episode_id
            """,
            (episode_id,),
        ).fetchall()
        blockers.extend(
            AutoResearchFinishBlocker(
                kind="experiment_episode",
                blocker_id=str(row["child_episode_id"]),
                state=str(row["status"]),
                action=(
                    "wait for report settlement"
                    if row["status"] == "wrapping_up"
                    else f"episode --key <key> --stop {row['child_episode_id']}"
                ),
            )
            for row in experiments
        )
        replacements = connection.execute(
            """
            SELECT child_episode_id, state FROM auto_research_child_experiments
            WHERE auto_research_episode_id = ? AND state = 'pending'
            ORDER BY created_at, child_episode_id
            """,
            (episode_id,),
        ).fetchall()
        blockers.extend(
            AutoResearchFinishBlocker(
                kind="experiment_replacement",
                blocker_id=str(row["child_episode_id"]),
                state=str(row["state"]),
                action=f"episode --key <key> --stop {row['child_episode_id']}",
            )
            for row in replacements
        )
        notices = connection.execute(
            """
            SELECT notice_id, state FROM auto_research_lifecycle_notices
            WHERE episode_id = ? AND delivered_at IS NULL AND acknowledged_at IS NULL
            ORDER BY created_at, notice_id
            """,
            (episode_id,),
        ).fetchall()
        blockers.extend(
            AutoResearchFinishBlocker(
                kind="lifecycle_notice",
                blocker_id=str(row["notice_id"]),
                state=str(row["state"]),
                action="inbox --key <key> --harvest or inbox --key <key> --clear",
            )
            for row in notices
        )
        admissions = connection.execute(
            """
            SELECT admission_id, state FROM auto_research_child_admissions
            WHERE episode_id = ? AND state = 'accepted'
            ORDER BY created_at, admission_id
            """,
            (episode_id,),
        ).fetchall()
        blockers.extend(
            AutoResearchFinishBlocker(
                kind="child_admission",
                blocker_id=str(row["admission_id"]),
                state=str(row["state"]),
                action="wait for child admission reconciliation",
            )
            for row in admissions
        )
        return blockers

    @staticmethod
    def _validate_auto_research_parent_admission(episode: object) -> None:
        if (
            episode.status != "running"
            or episode.ending is not None
            or episode.stop_requested_at is not None
        ):
            raise EpisodeNotRunning("the Auto-research episode is not accepting new work")

    def _validate_auto_research_experiment_route(
        self,
        connection: sqlite3.Connection,
        record: AutoResearchChildExperimentRecord,
    ) -> None:
        episode = self._load_auto_research_episode(connection, record.auto_research_episode_id)
        self._validate_auto_research_parent_admission(episode)
        if episode.project_id != record.project_id:
            raise ValueError("the child Experiment route belongs to another project")
        parent = connection.execute(
            """
            SELECT invocation.role FROM auto_research_invocations AS invocation
            WHERE invocation.episode_id = ? AND invocation.operation_id = ?
            """,
            (record.auto_research_episode_id, record.parent_operation_id),
        ).fetchone()
        if parent is None or parent["role"] != "orchestrator":
            raise ValueError("only the canonical orchestrator may launch an Experiment")

    @staticmethod
    def _reflect_auto_research_child_admission(
        connection: sqlite3.Connection,
        *,
        admission_id: str | None,
        episode_id: str,
        child_kind: str,
        child_id: str,
        updated_at: str,
    ) -> None:
        if admission_id is None:
            return
        row = connection.execute(
            "SELECT * FROM auto_research_child_admissions WHERE admission_id = ?",
            (admission_id,),
        ).fetchone()
        if row is None:
            raise ValueError("the child effect has no durable command admission")
        if (
            row["episode_id"] != episode_id
            or row["child_kind"] != child_kind
            or row["child_id"] != child_id
            or row["state"] not in {"accepted", "reflected"}
        ):
            raise ValueError("the child effect does not match its durable admission")
        connection.execute(
            """
            UPDATE auto_research_child_admissions
            SET state = 'reflected', updated_at = ?
            WHERE admission_id = ? AND state = 'accepted'
            """,
            (updated_at, admission_id),
        )

    def _settle_auto_research_child_experiment(
        self,
        child_episode_id: str,
        *,
        from_state: str,
        to_state: str,
        diagnostic: str | None,
    ) -> AutoResearchChildExperimentRecord:
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state FROM auto_research_child_experiments
                WHERE child_episode_id = ?
                """,
                (child_episode_id,),
            ).fetchone()
            if row is None:
                raise KeyError(child_episode_id)
            if row["state"] == to_state:
                stored = self._child_experiment_record(
                    connection.execute(
                        "SELECT * FROM auto_research_child_experiments WHERE child_episode_id = ?",
                        (child_episode_id,),
                    ).fetchone()
                )
                return stored
            changed = connection.execute(
                """
                UPDATE auto_research_child_experiments
                SET state = ?, terminal_diagnostic = ?, updated_at = ?
                WHERE child_episode_id = ? AND state = ?
                """,
                (to_state, diagnostic, now, child_episode_id, from_state),
            )
            if changed.rowcount != 1:
                raise ValueError("the child Experiment is no longer in the expected state")
        stored = self.auto_research_child_experiment(child_episode_id)
        assert stored is not None
        return stored

    def _settle_auto_research_experiment_replacement(
        self,
        child_episode_id: str,
        *,
        source_event: Literal["cancelled", "failed"],
        diagnostic: str,
    ) -> AutoResearchChildExperimentRecord:
        detail = " ".join(diagnostic.split())[:2000]
        if not detail:
            raise ValueError("replacement settlement requires a diagnostic")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM auto_research_child_experiments WHERE child_episode_id = ?",
                (child_episode_id,),
            ).fetchone()
            if row is None:
                raise KeyError(child_episode_id)
            route = self._child_experiment_record(row)
            if route.state == "pending":
                connection.execute(
                    """
                    UPDATE auto_research_child_experiments
                    SET state = 'cancelled', terminal_diagnostic = ?, updated_at = ?
                    WHERE child_episode_id = ? AND state = 'pending'
                    """,
                    (detail, now, child_episode_id),
                )
            elif route.state != "cancelled" or route.terminal_diagnostic != detail:
                raise ValueError("the Experiment replacement is no longer pending")
            payload: dict[str, object] = {
                "episode_id": child_episode_id,
                "status": source_event,
                "diagnostic": detail,
            }
            if route.replaces_episode_id is not None:
                payload["replaces_episode_id"] = route.replaces_episode_id
            notice = AutoResearchLifecycleNoticeRecord(
                notice_id=self._auto_research_notice_id(
                    route.auto_research_episode_id,
                    "experiment_replacement",
                    child_episode_id,
                    source_event,
                    1,
                ),
                episode_id=route.auto_research_episode_id,
                source_kind="experiment_replacement",
                source_id=child_episode_id,
                source_event=source_event,
                source_attempt=1,
                payload=payload,
                created_at=now,
            )
            prior = connection.execute(
                """
                SELECT source_event FROM auto_research_lifecycle_notices
                WHERE episode_id = ? AND source_kind = 'experiment_replacement'
                  AND source_id = ?
                """,
                (route.auto_research_episode_id, child_episode_id),
            ).fetchone()
            if prior is not None and prior["source_event"] != source_event:
                raise ValueError("the Experiment replacement already has another terminal outcome")
            self._insert_auto_research_lifecycle_notice(connection, notice)
            stored_row = connection.execute(
                "SELECT * FROM auto_research_child_experiments WHERE child_episode_id = ?",
                (child_episode_id,),
            ).fetchone()
            assert stored_row is not None
            stored = self._child_experiment_record(stored_row)
        return stored

    @staticmethod
    def _acknowledge_lifecycle_rows(
        connection: sqlite3.Connection,
        notices: list[AutoResearchLifecycleNoticeRecord],
        *,
        acknowledged_at: str,
        acknowledged_by: str,
    ) -> None:
        if not notices:
            return
        ids = [notice.notice_id for notice in notices]
        placeholders = ",".join("?" for _ in ids)
        connection.execute(
            f"""
            UPDATE auto_research_lifecycle_notices
            SET state = 'acknowledged', acknowledged_at = ?, acknowledged_by = ?
            WHERE notice_id IN ({placeholders}) AND acknowledged_at IS NULL
            """,
            (acknowledged_at, acknowledged_by, *ids),
        )

    @staticmethod
    def _same_child_experiment_intent(
        stored: AutoResearchChildExperimentRecord,
        requested: AutoResearchChildExperimentRecord,
    ) -> bool:
        return stored.model_dump(exclude={"state", "updated_at"}) == requested.model_dump(
            exclude={"state", "updated_at"}
        )

    @staticmethod
    def _child_experiment_values(record: AutoResearchChildExperimentRecord) -> tuple[object, ...]:
        return (
            record.child_episode_id,
            record.auto_research_episode_id,
            record.project_id,
            record.control_node_id,
            record.state,
            record.replaces_episode_id,
            json.dumps(record.request, sort_keys=True, separators=(",", ":")),
            record.goal_sha256,
            record.parent_operation_id,
            record.terminal_diagnostic,
            record.created_at,
            record.updated_at,
        )

    @staticmethod
    def _child_work_record(row: sqlite3.Row) -> AutoResearchChildWorkRecord:
        return AutoResearchChildWorkRecord.model_validate(dict(row))

    @staticmethod
    def _child_experiment_record(row: sqlite3.Row) -> AutoResearchChildExperimentRecord:
        data = dict(row)
        data["request"] = json.loads(data.pop("request_json"))
        return AutoResearchChildExperimentRecord.model_validate(data)

    @staticmethod
    def _child_admission_record(row: sqlite3.Row) -> AutoResearchChildAdmissionRecord:
        return AutoResearchChildAdmissionRecord.model_validate(dict(row))

    @staticmethod
    def _lifecycle_notice_record(row: sqlite3.Row) -> AutoResearchLifecycleNoticeRecord:
        data = dict(row)
        data["payload"] = json.loads(data.pop("payload_json"))
        return AutoResearchLifecycleNoticeRecord.model_validate(data)

    @staticmethod
    def _apply_result_record(row: sqlite3.Row) -> AutoResearchApplyResultRecord:
        data = dict(row)
        data["result"] = json.loads(data.pop("result_json"))
        return AutoResearchApplyResultRecord.model_validate(data)

    @staticmethod
    def _command_file_record(row: sqlite3.Row) -> AutoResearchCommandFileRecord:
        return AutoResearchCommandFileRecord.model_validate(dict(row))

    @staticmethod
    def _inbox_receipt_record(row: sqlite3.Row) -> AutoResearchInboxReceiptRecord:
        result = json.loads(row["result_json"])
        return AutoResearchInboxReceiptRecord.model_validate(
            {
                "effect_id": row["effect_id"],
                "episode_id": row["episode_id"],
                "mode": row["mode"],
                "notice_ids": result["notice_ids"],
                "count": result["count"],
                "notices": result["notices"],
                "acknowledged_by": row["acknowledged_by"],
                "created_at": row["created_at"],
            }
        )

    @staticmethod
    def _finish_receipt_record(row: sqlite3.Row) -> AutoResearchFinishReceiptRecord:
        data = dict(row)
        data["result"] = json.loads(data.pop("result_json"))
        return AutoResearchFinishReceiptRecord.model_validate(data)

    @staticmethod
    def detach_auto_research_children_for_restore(
        connection: sqlite3.Connection,
        *,
        diagnostic: str,
        confirmed_by: str,
        now: str,
    ) -> None:
        """Cancel captured child admissions/routes and consume pending lifecycle wakes."""

        if not connection.in_transaction:
            raise ValueError("restored Auto-research child detachment requires a transaction")
        detail = " ".join(diagnostic.split())[:2000]
        confirmer = " ".join(confirmed_by.split())[:500]
        if not detail or not confirmer:
            raise ValueError("restored child detachment requires a reason and confirmer")
        _required_timestamp(now)
        connection.execute(
            """
            UPDATE auto_research_child_experiments
            SET state = 'cancelled', terminal_diagnostic = ?, updated_at = ?
            WHERE state IN ('pending', 'running')
            """,
            (detail, now),
        )
        connection.execute(
            """
            UPDATE auto_research_child_admissions
            SET state = 'cancelled', updated_at = ?
            WHERE state = 'accepted'
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE auto_research_lifecycle_notices
            SET state = 'acknowledged', acknowledged_at = ?, acknowledged_by = ?
            WHERE delivered_at IS NULL AND acknowledged_at IS NULL
            """,
            (now, confirmer),
        )
