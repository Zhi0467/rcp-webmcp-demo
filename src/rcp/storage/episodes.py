from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

from rcp.core.models import AuthorizedHuman
from rcp.storage.models import (
    AGENT_TASK_PROJECTION_FIELDS,
    AgentTaskRecord,
    EpisodeBudgetMeter,
    EpisodeEnding,
    EpisodeInvocationCeilingReached,
    EpisodeInvocationRecord,
    EpisodeNotRunning,
    EpisodeRecord,
    EpisodeReportAttemptLimitReached,
    EpisodeReportAttemptRecord,
    EpisodeReportConflict,
    EpisodeReportRecord,
    EpisodeWrapupRecord,
    _required_timestamp,
)

_LIVE_EPISODE_STATUSES = ("queued", "running", "stopping", "wrapping_up")
_REPORT_ATTEMPT_LIMIT = 3


class EpisodeStoreMixin:
    """Mode-neutral episode lifecycle, operational budget, and report ledger."""

    def create_episode(self, record: EpisodeRecord) -> EpisodeRecord:
        self._validate_new_episode(record)
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._require_project_accepts_new_work(connection, record.project_id)
                if self._live_episode_row(connection, record) is not None:
                    raise ValueError("This episode mode already has a live parent.")
                self._insert_episode(connection, record)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Could not create the episode.") from exc
        stored = self.episode(record.episode_id)
        assert stored is not None
        return stored

    def create_episode_with_invocation(
        self,
        record: EpisodeRecord,
        root_task: AgentTaskRecord,
    ) -> tuple[EpisodeRecord, EpisodeInvocationRecord, AgentTaskRecord]:
        """Atomically create one parent and spend its first operational invocation."""

        self._validate_new_episode(record)
        if record.mode == "auto_research":
            raise ValueError("Auto-research roots must use the atomic mode adapter admission.")
        if (
            root_task.episode_id != record.episode_id
            or root_task.project_id != record.project_id
            or root_task.graph_target != record.graph_target
            or root_task.kind == "episode_report"
            or root_task.status != "queued"
            or not root_task.visible
            or root_task.parent_operation_id is not None
        ):
            raise ValueError("an episode root task must name its exact visible parent")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (record.episode_id,)
            ).fetchone()
            if existing is not None:
                stored_episode = self._episode_record(existing)
                invocation_row = connection.execute(
                    """
                    SELECT * FROM episode_invocations
                    WHERE episode_id = ? AND invocation_number = 1
                    """,
                    (record.episode_id,),
                ).fetchone()
                task_row = connection.execute(
                    "SELECT * FROM graph_runs WHERE operation_id = ?",
                    (root_task.operation_id,),
                ).fetchone()
                if not self._created_episode_pair_matches(
                    stored_episode,
                    record,
                    invocation_row,
                    task_row,
                    root_task,
                ):
                    raise EpisodeReportConflict(
                        "the episode root creation conflicts with its committed pair"
                    )
            else:
                self._require_project_accepts_new_work(connection, record.project_id)
                if self._live_episode_row(connection, record) is not None:
                    raise ValueError("This episode mode already has a live parent.")
                started = record.model_copy(
                    update={
                        "root_operation_id": root_task.operation_id,
                        "status": "running",
                        "invocations_used": 1,
                        "updated_at": root_task.created_at,
                    }
                )
                self._insert_episode(connection, started)
                self._insert_agent_task(connection, root_task, continuation_cause="fresh")
                connection.execute(
                    """
                    INSERT INTO episode_invocations (
                        episode_id, operation_id, invocation_number, created_at
                    ) VALUES (?, ?, 1, ?)
                    """,
                    (record.episode_id, root_task.operation_id, root_task.created_at),
                )
                stored_episode = started
                invocation_row = connection.execute(
                    """
                    SELECT * FROM episode_invocations
                    WHERE episode_id = ? AND invocation_number = 1
                    """,
                    (record.episode_id,),
                ).fetchone()
        stored_task = self.agent_task(root_task.operation_id)
        assert invocation_row is not None and stored_task is not None
        return (
            stored_episode,
            self._episode_invocation_record(invocation_row),
            stored_task,
        )

    def episode(self, episode_id: str) -> EpisodeRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
        return self._episode_record(row) if row is not None else None

    def episodes(self, project_id: str, *, limit: int = 50) -> list[EpisodeRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM episodes
                WHERE project_id = ?
                ORDER BY created_at DESC, episode_id DESC
                LIMIT ?
                """,
                (project_id, max(1, min(limit, 500))),
            ).fetchall()
        return [self._episode_record(row) for row in rows]

    def episodes_awaiting_report(self) -> list[EpisodeRecord]:
        """Return durable hidden wrap-ups that startup must reconcile."""

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT episode.*
                FROM episodes AS episode
                JOIN episode_wrapups AS wrapup ON wrapup.episode_id = episode.episode_id
                WHERE episode.status = 'wrapping_up'
                  AND episode.wrapup_state IN ('pending', 'running')
                  AND wrapup.state IN ('pending', 'running')
                  AND wrapup.allocation_operation_id IS NOT NULL
                ORDER BY episode.created_at, episode.episode_id
                """
            ).fetchall()
        return [self._episode_record(row) for row in rows]

    @staticmethod
    def detach_episode_reports_for_restore(
        connection: sqlite3.Connection,
        *,
        diagnostic: str,
        now: str,
    ) -> None:
        """Fail captured report calls and remove their native restart bindings."""

        if not connection.in_transaction:
            raise ValueError("restored report detachment requires an active transaction")
        detail = " ".join(diagnostic.split())[:2000]
        if not detail:
            raise ValueError("restored report detachment requires a diagnostic")
        _required_timestamp(now)
        connection.execute(
            """
            UPDATE episode_report_attempts
            SET status = 'failed', error = ?, updated_at = ?, finished_at = COALESCE(finished_at, ?)
            WHERE status IN ('queued', 'running')
            """,
            (detail, now, now),
        )
        connection.execute(
            """
            UPDATE episode_wrapups
            SET native_session_id = NULL, stage_host = NULL, stage_root = NULL,
                updated_at = ?
            WHERE state IN ('pending', 'running')
              AND (native_session_id IS NOT NULL OR stage_host IS NOT NULL OR stage_root IS NOT NULL)
            """,
            (now,),
        )

    def detach_experiment_episodes_for_restore(
        self,
        connection: sqlite3.Connection,
        *,
        diagnostic: str,
        now: str,
    ) -> None:
        """Stop every captured nonterminal Experiment episode without a report retry."""

        if not connection.in_transaction:
            raise ValueError("restored Experiment detachment requires an active transaction")
        detail = " ".join(diagnostic.split())[:2000]
        if not detail:
            raise ValueError("restored Experiment detachment requires a diagnostic")
        _required_timestamp(now)
        connection.execute(
            """
            UPDATE experiment_episode_state
            SET native_session_id = NULL, stage_host = NULL, stage_root = NULL, updated_at = ?
            WHERE native_session_id IS NOT NULL OR stage_host IS NOT NULL OR stage_root IS NOT NULL
            """,
            (now,),
        )
        episodes = connection.execute(
            """
            SELECT * FROM episodes
            WHERE mode = 'experiment_loop'
              AND status IN ('queued', 'running', 'stopping', 'wrapping_up')
            ORDER BY episode_id
            """
        ).fetchall()
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            receipt_json, receipt_sha256 = compact_episode_receipt(
                {
                    "diagnostic": detail,
                    "ending": "stopped",
                    "episode_id": episode_id,
                    "reason": "restore",
                }
            )
            wrapup = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if wrapup is None:
                concluding = connection.execute(
                    """
                    SELECT operation_id FROM episode_invocations
                    WHERE episode_id = ? ORDER BY invocation_number DESC LIMIT 1
                    """,
                    (episode_id,),
                ).fetchone()
                self._insert_episode_wrapup(
                    connection,
                    EpisodeWrapupRecord(
                        episode_id=episode_id,
                        ending="stopped",
                        partial=True,
                        concluding_operation_id=(
                            str(concluding["operation_id"]) if concluding is not None else None
                        ),
                        receipt_json=receipt_json,
                        receipt_sha256=receipt_sha256,
                        state="skipped",
                        diagnostic=detail,
                        created_at=now,
                        updated_at=now,
                        finished_at=now,
                    ),
                )
            else:
                if wrapup["state"] not in {"pending", "running"}:
                    raise EpisodeReportConflict(
                        "a nonterminal restored Experiment has a terminal report wrap-up"
                    )
                connection.execute(
                    """
                    UPDATE episode_wrapups
                    SET ending = 'stopped', partial = 1, native_session_id = NULL,
                        stage_host = NULL, stage_root = NULL, receipt_json = ?,
                        receipt_sha256 = ?, state = 'skipped', diagnostic = ?,
                        updated_at = ?, finished_at = COALESCE(finished_at, ?)
                    WHERE episode_id = ?
                    """,
                    (receipt_json, receipt_sha256, detail, now, now, episode_id),
                )
            connection.execute(
                """
                UPDATE episodes
                SET status = 'stopped', stop_requested_at = COALESCE(stop_requested_at, ?),
                    stop_settled_at = COALESCE(stop_settled_at, ?), ending = 'stopped',
                    ending_diagnostic = ?, wrapup_state = 'skipped', wrapup_error = NULL,
                    updated_at = ?, ended_at = COALESCE(ended_at, ?)
                WHERE episode_id = ?
                """,
                (now, now, detail, now, now, episode_id),
            )

    def episode_budget_meter(self, episode_id: str) -> EpisodeBudgetMeter:
        record = self.episode(episode_id)
        if record is None:
            raise KeyError(episode_id)
        with self.connection() as connection:
            usage = connection.execute(
                """
                SELECT
                    COALESCE(SUM(agent_usage.processed_input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(agent_usage.generated_tokens), 0) AS generated_tokens
                FROM agent_usage
                JOIN graph_runs
                  ON graph_runs.operation_id = agent_usage.operation_id
                WHERE graph_runs.episode_id = ?
                  AND graph_runs.kind != 'episode_report'
                  AND agent_usage.counted = 1
                """,
                (episode_id,),
            ).fetchone()
        return EpisodeBudgetMeter(
            invocation_ceiling=record.invocation_ceiling,
            invocations_used=record.invocations_used,
            invocations_remaining=record.invocations_remaining,
            observed_input_tokens=int(usage["input_tokens"]),
            observed_generated_tokens=int(usage["generated_tokens"]),
        )

    def request_episode_stop(self, episode_id: str) -> EpisodeRecord:
        """Persist the common Stop fence before a mode adapter settles its work."""

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._request_episode_stop_in_connection(connection, episode_id, now=now)
        stopped = self.episode(episode_id)
        assert stopped is not None
        return stopped

    def _request_episode_stop_in_connection(
        self,
        connection: sqlite3.Connection,
        episode_id: str,
        *,
        now: str,
    ) -> EpisodeRecord:
        row = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        episode = self._episode_record(row)
        if episode.ending == "stopped" and episode.wrapup_state == "skipped":
            return episode
        if episode.status not in {"queued", "running", "stopping"}:
            raise EpisodeNotRunning("the episode can no longer be stopped before wrap-up")
        if (
            connection.execute(
                "SELECT 1 FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            is not None
        ):
            raise EpisodeNotRunning("the episode has already entered wrap-up")
        if episode.stop_requested_at is not None:
            return episode
        connection.execute(
            """
            UPDATE episodes
            SET status = 'stopping', stop_requested_at = ?, updated_at = ?
            WHERE episode_id = ?
            """,
            (now, now, episode_id),
        )
        updated = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        assert updated is not None
        return self._episode_record(updated)

    def fence_episode_ending(
        self,
        episode_id: str,
        ending: EpisodeEnding,
        *,
        diagnostic: str | None = None,
    ) -> EpisodeRecord:
        """Fence new operational work as soon as one non-Stop ending is known."""

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._fence_episode_ending_in_connection(
                connection,
                episode_id,
                ending,
                diagnostic=diagnostic,
                now=now,
            )
        fenced = self.episode(episode_id)
        assert fenced is not None
        return fenced

    def _fence_episode_ending_in_connection(
        self,
        connection: sqlite3.Connection,
        episode_id: str,
        ending: EpisodeEnding,
        *,
        diagnostic: str | None,
        now: str,
    ) -> EpisodeRecord:
        if ending == "stopped":
            raise ValueError("Stop uses its dedicated graceful settlement path.")
        self._status_for_ending(ending)
        row = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        episode = self._episode_record(row)
        if episode.ending is not None:
            if episode.ending != ending or episode.ending_diagnostic != diagnostic:
                raise EpisodeReportConflict("the episode ending fence is immutable")
            return episode
        if episode.stop_requested_at is not None or episode.status == "stopping":
            raise EpisodeNotRunning("Stop already fenced this episode")
        if episode.status not in {"queued", "running"}:
            raise EpisodeNotRunning("the episode can no longer accept an ending fence")
        connection.execute(
            """
            UPDATE episodes
            SET status = 'wrapping_up', ending = ?, ending_diagnostic = ?, updated_at = ?
            WHERE episode_id = ?
            """,
            (ending, diagnostic, now, episode_id),
        )
        updated = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        assert updated is not None
        return self._episode_record(updated)

    def allocate_episode_invocation(
        self,
        episode_id: str,
        task: AgentTaskRecord,
        *,
        continuation_cause: str = "fresh",
    ) -> tuple[EpisodeRecord, EpisodeInvocationRecord, AgentTaskRecord]:
        """Atomically spend one operational invocation and create its visible task."""

        if task.kind == "episode_report" or not task.visible:
            raise ValueError("an operational episode invocation requires a visible non-report task")
        if task.episode_id != episode_id:
            raise ValueError("an episode invocation task must name its exact parent episode")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM episode_invocations WHERE operation_id = ?",
                (task.operation_id,),
            ).fetchone()
            if existing is not None:
                if existing["episode_id"] != episode_id:
                    raise ValueError("this operation is already allocated to another episode")
                if (
                    connection.execute(
                        "SELECT 1 FROM graph_runs WHERE operation_id = ?", (task.operation_id,)
                    ).fetchone()
                    is None
                ):
                    raise RuntimeError("the episode invocation lost its task")
                invocation = self._episode_invocation_record(existing)
            else:
                row = connection.execute(
                    "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(episode_id)
                episode = self._episode_record(row)
                if episode.mode == "auto_research":
                    raise ValueError(
                        "Auto-research turns must use the atomic mode adapter admission."
                    )
                if episode.status not in {"queued", "running"}:
                    raise EpisodeNotRunning("the episode is not admitting operational work")
                if episode.invocations_used >= episode.invocation_ceiling:
                    raise EpisodeInvocationCeilingReached(
                        "the episode has spent its operational invocation ceiling"
                    )
                if task.project_id != episode.project_id:
                    raise ValueError("the episode task belongs to a different project")
                self._insert_agent_task(
                    connection,
                    task,
                    continuation_cause=continuation_cause,
                )
                invocation_number = episode.invocations_used + 1
                connection.execute(
                    """
                    INSERT INTO episode_invocations (
                        episode_id, operation_id, invocation_number, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (episode_id, task.operation_id, invocation_number, task.created_at),
                )
                connection.execute(
                    """
                    UPDATE episodes
                    SET root_operation_id = COALESCE(root_operation_id, ?),
                        status = 'running', invocations_used = ?, updated_at = ?
                    WHERE episode_id = ?
                    """,
                    (task.operation_id, invocation_number, self.now(), episode_id),
                )
                invocation = EpisodeInvocationRecord(
                    episode_id=episode_id,
                    operation_id=task.operation_id,
                    invocation_number=invocation_number,
                    created_at=task.created_at,
                )
        stored_episode = self.episode(episode_id)
        stored_task = self.agent_task(task.operation_id)
        assert stored_episode is not None and stored_task is not None
        return stored_episode, invocation, stored_task

    def episode_invocations(self, episode_id: str) -> list[EpisodeInvocationRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM episode_invocations
                WHERE episode_id = ? ORDER BY invocation_number
                """,
                (episode_id,),
            ).fetchall()
        return [self._episode_invocation_record(row) for row in rows]

    def begin_episode_wrapup(
        self,
        episode_id: str,
        wrapup: EpisodeWrapupRecord,
        allocation_task: AgentTaskRecord,
    ) -> tuple[EpisodeRecord, EpisodeWrapupRecord, AgentTaskRecord]:
        """Freeze one non-Stop ending and its single hidden report allocation."""

        self._validate_new_wrapup(episode_id, wrapup, allocation_task)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode_row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if episode_row is None:
                raise KeyError(episode_id)
            episode = self._episode_record(episode_row)
            if episode.status not in {"queued", "running", "wrapping_up"}:
                raise EpisodeNotRunning("the episode cannot begin report generation")
            if allocation_task.project_id != episode.project_id:
                raise ValueError("the report allocation belongs to a different project")
            existing = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if existing is not None:
                stored = self._episode_wrapup_record(existing)
                if not self._wrapup_fence_matches(stored, wrapup):
                    raise EpisodeReportConflict("the episode wrap-up restart fence is immutable")
                operation_id = stored.allocation_operation_id
                assert operation_id is not None
            else:
                if episode.ending is not None and (
                    episode.ending != wrapup.ending
                    or episode.ending_diagnostic != wrapup.diagnostic
                ):
                    raise EpisodeReportConflict(
                        "the episode already has a different semantic ending"
                    )
                if wrapup.concluding_operation_id is not None:
                    concluding = connection.execute(
                        """
                        SELECT 1 FROM graph_runs
                        WHERE episode_id = ? AND operation_id = ?
                          AND visible = 1 AND kind != 'episode_report'
                        """,
                        (episode_id, wrapup.concluding_operation_id),
                    ).fetchone()
                    if concluding is None:
                        raise ValueError(
                            "the report continuation is not a visible task of this episode"
                        )
                self._insert_agent_task(
                    connection,
                    allocation_task,
                    continuation_cause="episode_report",
                )
                self._insert_episode_wrapup(connection, wrapup)
                connection.execute(
                    """
                    UPDATE episodes
                    SET status = 'wrapping_up', ending = ?, ending_diagnostic = ?,
                        wrapup_state = 'pending', wrapup_error = NULL, updated_at = ?
                    WHERE episode_id = ?
                    """,
                    (wrapup.ending, wrapup.diagnostic, self.now(), episode_id),
                )
                stored = wrapup
                operation_id = allocation_task.operation_id
        episode = self.episode(episode_id)
        task = self.agent_task(operation_id)
        assert episode is not None and task is not None
        return episode, stored, task

    def end_episode_without_report(
        self,
        episode_id: str,
        *,
        ending: EpisodeEnding,
        diagnostic: str | None = None,
    ) -> EpisodeRecord:
        """Terminalize one ending that has no episode session to report from.

        An episode whose turn died before it bound a provider session has nothing
        for report generation to resume, so it never enters wrap-up at all. It is
        settled here in one step instead: fencing the ending and then discovering
        the report is impossible would park the episode on the live `wrapping_up`
        status and leave a report error on work that never ran.
        """

        if ending == "stopped":
            raise ValueError("Stop settles through its own skip path.")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if row is None:
                raise KeyError(episode_id)
            episode = self._episode_record(row)
            if (
                connection.execute(
                    "SELECT 1 FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
                ).fetchone()
                is not None
            ):
                raise EpisodeReportConflict("the episode already has a report wrap-up")
            if episode.ending is not None and (
                episode.ending != ending or episode.ending_diagnostic != diagnostic
            ):
                raise EpisodeReportConflict("the episode ending fence is immutable")
            if episode.stop_requested_at is not None or episode.status == "stopping":
                raise EpisodeNotRunning("Stop already fenced this episode")
            final_status = self._status_for_ending(ending)
            if episode.status == final_status:
                return episode
            # `wrapping_up` is admitted because the Experiment path fences the
            # ending before it learns whether a report can be generated at all.
            if episode.status not in {"queued", "running", "wrapping_up"}:
                raise EpisodeNotRunning("the episode can no longer accept an ending fence")
            connection.execute(
                """
                UPDATE episodes
                SET status = ?, ending = ?, ending_diagnostic = ?,
                    wrapup_state = 'not_started', wrapup_error = NULL, updated_at = ?,
                    ended_at = COALESCE(ended_at, ?)
                WHERE episode_id = ?
                """,
                (final_status, ending, diagnostic, now, now, episode_id),
            )
            self._terminalize_auto_research_child_experiment_with_notice(
                connection,
                child_episode_id=episode_id,
                status=final_status,
                ending=ending,
                diagnostic=diagnostic,
                created_at=now,
            )
        stored = self.episode(episode_id)
        assert stored is not None
        return stored

    def fail_episode_wrapup_unlaunchable(
        self,
        episode_id: str,
        wrapup: EpisodeWrapupRecord,
        *,
        ending_diagnostic: str | None = None,
    ) -> tuple[EpisodeRecord, EpisodeWrapupRecord]:
        """Settle a non-Stop ending whose exact report continuation cannot launch."""

        if (
            wrapup.episode_id != episode_id
            or wrapup.ending in {None, "stopped"}
            or wrapup.state != "failed"
            or wrapup.allocation_operation_id is not None
            or not wrapup.diagnostic
            or wrapup.finished_at is None
        ):
            raise ValueError("an unlaunchable wrap-up requires its ending and final diagnostic")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode_row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if episode_row is None:
                raise KeyError(episode_id)
            episode = self._episode_record(episode_row)
            existing = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if existing is not None:
                stored = self._episode_wrapup_record(existing)
                if stored != wrapup:
                    raise EpisodeReportConflict("the episode wrap-up restart fence is immutable")
                return episode, stored
            if episode.status not in {"queued", "running", "wrapping_up"}:
                raise EpisodeNotRunning("the episode has already ended")
            if wrapup.concluding_operation_id is not None:
                concluding = connection.execute(
                    """
                    SELECT 1 FROM graph_runs
                    WHERE episode_id = ? AND operation_id = ?
                      AND visible = 1 AND kind != 'episode_report'
                    """,
                    (episode_id, wrapup.concluding_operation_id),
                ).fetchone()
                if concluding is None:
                    raise ValueError(
                        "the report continuation is not a visible task of this episode"
                    )
            self._insert_episode_wrapup(connection, wrapup)
            stored = wrapup
            connection.execute(
                """
                UPDATE episodes
                SET status = ?, ending = ?, ending_diagnostic = ?,
                    wrapup_state = 'failed', wrapup_error = ?, updated_at = ?,
                    ended_at = COALESCE(ended_at, ?)
                WHERE episode_id = ?
                """,
                (
                    self._status_for_ending(wrapup.ending),
                    wrapup.ending,
                    ending_diagnostic,
                    wrapup.diagnostic,
                    now,
                    now,
                    episode_id,
                ),
            )
            self._terminalize_auto_research_child_experiment_with_notice(
                connection,
                child_episode_id=episode_id,
                status=self._status_for_ending(str(wrapup.ending)),
                ending=str(wrapup.ending),
                diagnostic=wrapup.diagnostic or ending_diagnostic,
                created_at=now,
            )
        episode = self.episode(episode_id)
        assert episode is not None
        return episode, stored

    def episode_wrapup(self, episode_id: str) -> EpisodeWrapupRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
            ).fetchone()
        return self._episode_wrapup_record(row) if row is not None else None

    def mark_episode_stop_skipped(
        self,
        episode_id: str,
        *,
        diagnostic: str | None = None,
    ) -> EpisodeRecord:
        """Settle Stop atomically; it is the only transition that skips a report."""

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if row is None:
                raise KeyError(episode_id)
            episode = self._episode_record(row)
            if episode.ending == "stopped" and episode.wrapup_state == "skipped":
                return episode
            if (
                connection.execute(
                    "SELECT 1 FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
                ).fetchone()
                is not None
            ):
                raise EpisodeNotRunning("report generation has already begun")
            if episode.status not in {"queued", "running", "stopping"}:
                raise EpisodeNotRunning("the episode has already ended")
            receipt_json, receipt_sha256 = compact_episode_receipt(
                {
                    "diagnostic": diagnostic,
                    "ending": "stopped",
                    "episode_id": episode_id,
                }
            )
            concluding = connection.execute(
                """
                SELECT operation_id FROM episode_invocations
                WHERE episode_id = ? ORDER BY invocation_number DESC LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
            wrapup = EpisodeWrapupRecord(
                episode_id=episode_id,
                ending="stopped",
                partial=True,
                concluding_operation_id=(concluding["operation_id"] if concluding else None),
                receipt_json=receipt_json,
                receipt_sha256=receipt_sha256,
                state="skipped",
                diagnostic=diagnostic,
                created_at=now,
                updated_at=now,
                finished_at=now,
            )
            self._insert_episode_wrapup(connection, wrapup)
            connection.execute(
                """
                UPDATE episodes
                SET status = 'stopped', stop_requested_at = COALESCE(stop_requested_at, ?),
                    stop_settled_at = COALESCE(stop_settled_at, ?), ending = 'stopped',
                    ending_diagnostic = ?, wrapup_state = 'skipped', wrapup_error = NULL,
                    updated_at = ?, ended_at = COALESCE(ended_at, ?)
                WHERE episode_id = ?
                """,
                (now, now, diagnostic, now, now, episode_id),
            )
            self._terminalize_auto_research_child_experiment_with_notice(
                connection,
                child_episode_id=episode_id,
                status="stopped",
                ending="stopped",
                diagnostic=diagnostic,
                created_at=now,
            )
        stored = self.episode(episode_id)
        assert stored is not None
        return stored

    def allocate_episode_report_attempt(
        self,
        episode_id: str,
    ) -> EpisodeReportAttemptRecord:
        """Allocate one provider call under the episode's stable hidden task."""

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode_row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if episode_row is None:
                raise KeyError(episode_id)
            episode = self._episode_record(episode_row)
            wrapup_row = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if wrapup_row is None:
                raise EpisodeNotRunning("the episode has no report allocation")
            wrapup = self._episode_wrapup_record(wrapup_row)
            if (
                episode.status != "wrapping_up"
                or episode.wrapup_state not in {"pending", "running"}
                or wrapup.state not in {"pending", "running"}
                or wrapup.allocation_operation_id is None
            ):
                raise EpisodeNotRunning("the episode is not awaiting report generation")
            current = connection.execute(
                """
                SELECT * FROM episode_report_attempts
                WHERE episode_id = ? AND status IN ('queued', 'running')
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
            if current is not None:
                return self._episode_report_attempt_record(current)
            if episode.report_attempts_used >= _REPORT_ATTEMPT_LIMIT:
                raise EpisodeReportAttemptLimitReached(
                    "the episode has spent all three report attempts"
                )
            attempt_number = episode.report_attempts_used + 1
            attempt_id = str(uuid.uuid4())
            now = self.now()
            connection.execute(
                """
                INSERT INTO episode_report_attempts (
                    attempt_id, episode_id, attempt_number, allocation_operation_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    attempt_id,
                    episode_id,
                    attempt_number,
                    wrapup.allocation_operation_id,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE episodes
                SET wrapup_state = 'running', report_attempts_used = ?, updated_at = ?
                WHERE episode_id = ?
                """,
                (attempt_number, now, episode_id),
            )
            connection.execute(
                """
                UPDATE episode_wrapups SET state = 'running', updated_at = ?
                WHERE episode_id = ?
                """,
                (now, episode_id),
            )
        attempt = self.episode_report_attempt(attempt_id)
        assert attempt is not None
        return attempt

    def current_episode_report_attempt(
        self,
        episode_id: str,
    ) -> EpisodeReportAttemptRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM episode_report_attempts
                WHERE episode_id = ? AND status IN ('queued', 'running')
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
        return self._episode_report_attempt_record(row) if row is not None else None

    def requeue_interrupted_episode_report_allocation(
        self,
        episode_id: str,
    ) -> AgentTaskRecord:
        """Requeue only the same hidden allocation interrupted or paused at restart."""

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode_row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            wrapup_row = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if episode_row is None:
                raise KeyError(episode_id)
            if wrapup_row is None or wrapup_row["allocation_operation_id"] is None:
                raise EpisodeNotRunning("the episode has no report allocation to restart")
            episode = self._episode_record(episode_row)
            if episode.status != "wrapping_up" or episode.wrapup_state not in {
                "pending",
                "running",
            }:
                raise EpisodeNotRunning("the episode report allocation cannot restart")
            operation_id = str(wrapup_row["allocation_operation_id"])
            task_row = connection.execute(
                "SELECT * FROM graph_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if (
                task_row is None
                or task_row["episode_id"] != episode_id
                or task_row["kind"] != "episode_report"
                or bool(task_row["visible"])
            ):
                raise EpisodeReportConflict("the hidden report allocation lost its restart fence")
            if task_row["status"] == "queued":
                return self._agent_task_record(task_row)
            if task_row["status"] not in {"interrupted", "paused"}:
                raise EpisodeNotRunning(
                    "only an interrupted or shutdown-paused report allocation may be requeued"
                )
            prior_status = str(task_row["status"])
            diagnostic = (
                "The report provider call was interrupted by an RCP restart."
                if prior_status == "interrupted"
                else "The report provider call was paused during RCP shutdown."
            )
            current = connection.execute(
                """
                SELECT * FROM episode_report_attempts
                WHERE episode_id = ? AND status IN ('queued', 'running')
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
            if int(episode.report_attempts_used) >= _REPORT_ATTEMPT_LIMIT and (
                current is None or current["status"] != "queued"
            ):
                if current is None or current["status"] != "running":
                    raise EpisodeNotRunning("the episode has spent all report attempts")
                connection.execute(
                    """
                    UPDATE episode_report_attempts
                    SET status = 'failed', error = ?, updated_at = ?, finished_at = ?
                    WHERE attempt_id = ?
                    """,
                    (diagnostic, now, now, current["attempt_id"]),
                )
                final_status = self._status_for_ending(str(episode.ending))
                connection.execute(
                    """
                    UPDATE episodes
                    SET status = ?, wrapup_state = 'failed', wrapup_error = ?,
                        updated_at = ?, ended_at = COALESCE(ended_at, ?)
                    WHERE episode_id = ?
                    """,
                    (final_status, diagnostic, now, now, episode_id),
                )
                connection.execute(
                    """
                    UPDATE episode_wrapups
                    SET state = 'failed', diagnostic = ?, updated_at = ?, finished_at = ?
                    WHERE episode_id = ?
                    """,
                    (diagnostic, now, now, episode_id),
                )
                connection.execute(
                    """
                    UPDATE graph_runs
                    SET status = 'failed', status_message = ?, error = ?, updated_at = ?,
                        finished_at = COALESCE(finished_at, ?), phase = 'failed'
                    WHERE operation_id = ? AND status = ?
                    """,
                    (diagnostic, diagnostic, now, now, operation_id, prior_status),
                )
                self._terminalize_auto_research_child_experiment_with_notice(
                    connection,
                    child_episode_id=episode_id,
                    status=final_status,
                    ending=str(episode.ending),
                    diagnostic=diagnostic,
                    created_at=now,
                )
                terminal_task = connection.execute(
                    "SELECT * FROM graph_runs WHERE operation_id = ?", (operation_id,)
                ).fetchone()
                assert terminal_task is not None
                return self._agent_task_record(terminal_task)
            if current is not None and current["status"] == "running":
                connection.execute(
                    """
                    UPDATE episode_report_attempts
                    SET status = 'failed', error = ?, updated_at = ?, finished_at = ?
                    WHERE attempt_id = ?
                    """,
                    (diagnostic, now, now, current["attempt_id"]),
                )
            connection.execute(
                """
                UPDATE graph_runs
                SET status = 'queued', status_message = 'Wrapping up visualization and report',
                    error = NULL, updated_at = ?, started_at = NULL, finished_at = NULL,
                    last_activity_at = NULL, phase = 'queued', write_scope_fingerprint = NULL
                WHERE operation_id = ? AND status = ?
                """,
                (now, operation_id, prior_status),
            )
            self._insert_agent_task_receipt(
                connection,
                operation_id,
                "operation_dispatch_reset",
                self._bounded_receipt_payload(
                    {
                        "status": "queued",
                        "reason": "episode_report_restart",
                        "previous_status": prior_status,
                    }
                ),
                tier="summary",
                created_at=now,
            )
            connection.execute(
                """
                UPDATE episodes
                SET wrapup_state = 'pending', wrapup_error = NULL, updated_at = ?
                WHERE episode_id = ?
                """,
                (now, episode_id),
            )
            connection.execute(
                """
                UPDATE episode_wrapups
                SET state = 'pending', diagnostic = NULL, updated_at = ?, finished_at = NULL
                WHERE episode_id = ?
                """,
                (now, episode_id),
            )
        task = self.agent_task(operation_id)
        assert task is not None
        return task

    def fail_episode_report_allocation_unlaunchable(
        self,
        episode_id: str,
        diagnostic: str,
    ) -> tuple[EpisodeRecord, EpisodeWrapupRecord, AgentTaskRecord]:
        """Fail a fenced allocation before its next provider call is attempted."""

        diagnostic = diagnostic.strip()
        if not diagnostic:
            raise ValueError("an unlaunchable report allocation requires a diagnostic")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode_row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            wrapup_row = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if episode_row is None:
                raise KeyError(episode_id)
            if wrapup_row is None or wrapup_row["allocation_operation_id"] is None:
                raise EpisodeNotRunning("the episode has no report allocation to fail")
            episode = self._episode_record(episode_row)
            operation_id = str(wrapup_row["allocation_operation_id"])
            task_row = connection.execute(
                "SELECT * FROM graph_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if (
                task_row is None
                or task_row["episode_id"] != episode_id
                or task_row["kind"] != "episode_report"
                or bool(task_row["visible"])
            ):
                raise EpisodeReportConflict("the hidden report allocation lost its restart fence")
            attempt_summary = connection.execute(
                """
                SELECT COUNT(*) AS attempt_count,
                       COALESCE(SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END), 0)
                           AS queued_count,
                       COALESCE(SUM(CASE WHEN status IN ('running', 'succeeded')
                                         THEN 1 ELSE 0 END), 0) AS launched_count
                FROM episode_report_attempts WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
            if (
                int(attempt_summary["attempt_count"]) != episode.report_attempts_used
                or int(attempt_summary["queued_count"]) > 1
                or int(attempt_summary["launched_count"]) != 0
            ):
                raise EpisodeNotRunning(
                    "an allocated report can be unlaunchable only between provider calls"
                )
            if (
                episode.wrapup_state == "failed"
                and wrapup_row["state"] == "failed"
                and task_row["status"] == "failed"
            ):
                if episode.wrapup_error != diagnostic or wrapup_row["diagnostic"] != diagnostic:
                    raise EpisodeReportConflict(
                        "the failed report allocation already has a different diagnostic"
                    )
                return (
                    episode,
                    self._episode_wrapup_record(wrapup_row),
                    self._agent_task_record(task_row),
                )
            if (
                episode.status != "wrapping_up"
                or episode.wrapup_state not in {"pending", "running"}
                or wrapup_row["state"] not in {"pending", "running"}
                or task_row["status"]
                not in {"queued", "running", "pausing", "paused", "interrupted"}
            ):
                raise EpisodeNotRunning("the report allocation cannot be failed before launch")
            connection.execute(
                """
                UPDATE episode_report_attempts
                SET status = 'failed', error = ?, updated_at = ?, finished_at = ?
                WHERE episode_id = ? AND status = 'queued'
                """,
                (diagnostic, now, now, episode_id),
            )
            final_status = self._status_for_ending(str(episode.ending))
            connection.execute(
                """
                UPDATE episodes
                SET status = ?, wrapup_state = 'failed', wrapup_error = ?,
                    updated_at = ?, ended_at = COALESCE(ended_at, ?)
                WHERE episode_id = ?
                """,
                (final_status, diagnostic, now, now, episode_id),
            )
            connection.execute(
                """
                UPDATE episode_wrapups
                SET state = 'failed', diagnostic = ?, updated_at = ?, finished_at = ?
                WHERE episode_id = ?
                """,
                (diagnostic, now, now, episode_id),
            )
            connection.execute(
                """
                UPDATE graph_runs
                SET status = 'failed', status_message = ?, error = ?, updated_at = ?,
                    finished_at = COALESCE(finished_at, ?), phase = 'failed'
                WHERE operation_id = ?
                """,
                (diagnostic, diagnostic, now, now, operation_id),
            )
            self._terminalize_auto_research_child_experiment_with_notice(
                connection,
                child_episode_id=episode_id,
                status=final_status,
                ending=str(episode.ending),
                diagnostic=diagnostic,
                created_at=now,
            )
            stored_episode_row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            stored_wrapup_row = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            stored_task_row = connection.execute(
                "SELECT * FROM graph_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            assert (
                stored_episode_row is not None
                and stored_wrapup_row is not None
                and stored_task_row is not None
            )
            return (
                self._episode_record(stored_episode_row),
                self._episode_wrapup_record(stored_wrapup_row),
                self._agent_task_record(stored_task_row),
            )

    def episode_report_attempt(self, attempt_id: str) -> EpisodeReportAttemptRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM episode_report_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
        return self._episode_report_attempt_record(row) if row is not None else None

    def episode_report_attempts(self, episode_id: str) -> list[EpisodeReportAttemptRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM episode_report_attempts
                WHERE episode_id = ? ORDER BY attempt_number
                """,
                (episode_id,),
            ).fetchall()
        return [self._episode_report_attempt_record(row) for row in rows]

    def mark_episode_report_attempt_running(
        self,
        attempt_id: str,
    ) -> EpisodeReportAttemptRecord:
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM episode_report_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            if row["status"] not in {"queued", "running"}:
                raise EpisodeNotRunning("the report attempt has already ended")
            connection.execute(
                "UPDATE episode_report_attempts SET status = 'running', updated_at = ? "
                "WHERE attempt_id = ?",
                (now, attempt_id),
            )
            connection.execute(
                """
                UPDATE graph_runs
                SET status = 'running', started_at = COALESCE(started_at, ?),
                    updated_at = ?, phase = 'running'
                WHERE operation_id = ?
                """,
                (now, now, row["allocation_operation_id"]),
            )
        stored = self.episode_report_attempt(attempt_id)
        assert stored is not None
        return stored

    def record_episode_report_attempt_error(
        self,
        attempt_id: str,
        error: str,
    ) -> tuple[EpisodeRecord, EpisodeReportAttemptRecord]:
        """Record a retryable error; the third error closes wrap-up automatically."""

        return self._finish_episode_report_attempt_error(attempt_id, error, force_final=False)

    def finish_episode_report_error(
        self,
        attempt_id: str,
        error: str,
    ) -> tuple[EpisodeRecord, EpisodeReportAttemptRecord]:
        """Close wrap-up immediately for an unrecoverable report error."""

        return self._finish_episode_report_attempt_error(attempt_id, error, force_final=True)

    def _finish_episode_report_attempt_error(
        self,
        attempt_id: str,
        error: str,
        *,
        force_final: bool,
    ) -> tuple[EpisodeRecord, EpisodeReportAttemptRecord]:
        if not error.strip():
            raise ValueError("a report attempt error must explain the failure")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM episode_report_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            if row["status"] == "succeeded":
                raise EpisodeReportConflict("a successful report attempt cannot fail")
            episode_id = str(row["episode_id"])
            if row["status"] != "failed":
                connection.execute(
                    """
                    UPDATE episode_report_attempts
                    SET status = 'failed', error = ?, updated_at = ?, finished_at = ?
                    WHERE attempt_id = ?
                    """,
                    (error, now, now, attempt_id),
                )
            final = force_final or int(row["attempt_number"]) >= _REPORT_ATTEMPT_LIMIT
            if final:
                episode_row = connection.execute(
                    "SELECT ending FROM episodes WHERE episode_id = ?", (episode_id,)
                ).fetchone()
                if episode_row is None:
                    raise RuntimeError("the report attempt lost its episode")
                final_status = self._status_for_ending(str(episode_row["ending"]))
                connection.execute(
                    """
                    UPDATE episodes
                    SET status = ?, wrapup_state = 'failed', wrapup_error = ?,
                        updated_at = ?, ended_at = COALESCE(ended_at, ?)
                    WHERE episode_id = ?
                    """,
                    (final_status, error, now, now, episode_id),
                )
                connection.execute(
                    """
                    UPDATE episode_wrapups
                    SET state = 'failed', diagnostic = ?, updated_at = ?, finished_at = ?
                    WHERE episode_id = ?
                    """,
                    (error, now, now, episode_id),
                )
                connection.execute(
                    """
                    UPDATE graph_runs
                    SET status = 'failed', status_message = ?, error = ?,
                        updated_at = ?, finished_at = ?, phase = 'failed'
                    WHERE operation_id = ?
                    """,
                    (error, error, now, now, row["allocation_operation_id"]),
                )
                self._terminalize_auto_research_child_experiment_with_notice(
                    connection,
                    child_episode_id=episode_id,
                    status=final_status,
                    ending=str(episode_row["ending"]),
                    diagnostic=error,
                    created_at=now,
                )
            else:
                connection.execute(
                    "UPDATE episodes SET wrapup_state = 'pending', wrapup_error = NULL, "
                    "updated_at = ? WHERE episode_id = ?",
                    (now, episode_id),
                )
                connection.execute(
                    "UPDATE episode_wrapups SET state = 'pending', updated_at = ? "
                    "WHERE episode_id = ?",
                    (now, episode_id),
                )
        episode = self.episode(episode_id)
        attempt = self.episode_report_attempt(attempt_id)
        assert episode is not None and attempt is not None
        return episode, attempt

    def finish_episode_report_ready(
        self,
        attempt_id: str,
        report: EpisodeReportRecord,
    ) -> tuple[EpisodeRecord, EpisodeReportRecord]:
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt_row = connection.execute(
                "SELECT * FROM episode_report_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if attempt_row is None:
                raise KeyError(attempt_id)
            episode_row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (attempt_row["episode_id"],)
            ).fetchone()
            wrapup_row = connection.execute(
                "SELECT * FROM episode_wrapups WHERE episode_id = ?",
                (attempt_row["episode_id"],),
            ).fetchone()
            if episode_row is None or wrapup_row is None:
                raise RuntimeError("the report attempt lost its episode wrap-up")
            episode = self._episode_record(episode_row)
            wrapup = self._episode_wrapup_record(wrapup_row)
            if (
                report.attempt_id != attempt_id
                or report.episode_id != episode.episode_id
                or report.allocation_operation_id != attempt_row["allocation_operation_id"]
                or report.allocation_operation_id != wrapup.allocation_operation_id
                or report.ending != episode.ending
            ):
                raise EpisodeReportConflict("the report does not match its episode attempt")
            if episode.ending == "stopped":
                raise EpisodeReportConflict("a stopped episode cannot publish a report")
            existing = connection.execute(
                "SELECT * FROM episode_reports WHERE episode_id = ?", (episode.episode_id,)
            ).fetchone()
            if existing is not None:
                stored = self._episode_report_record(existing)
                if stored != report:
                    raise EpisodeReportConflict("the episode report is immutable")
                return episode, stored
            if attempt_row["status"] not in {"queued", "running"}:
                raise EpisodeReportConflict("the report attempt has already ended")
            connection.execute(
                """
                INSERT INTO episode_reports (
                    report_id, episode_id, attempt_id, allocation_operation_id, ending,
                    sha256, html, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.episode_id,
                    report.attempt_id,
                    report.allocation_operation_id,
                    report.ending,
                    report.sha256,
                    report.html,
                    report.created_at,
                ),
            )
            connection.execute(
                """
                UPDATE episode_report_attempts
                SET status = 'succeeded', error = NULL, updated_at = ?, finished_at = ?
                WHERE attempt_id = ?
                """,
                (now, now, attempt_id),
            )
            connection.execute(
                """
                UPDATE graph_runs
                SET status = 'succeeded', status_message = 'Report ready', error = NULL,
                    updated_at = ?, finished_at = ?, phase = 'completed'
                WHERE operation_id = ?
                """,
                (now, now, report.allocation_operation_id),
            )
            connection.execute(
                """
                UPDATE episode_wrapups
                SET state = 'ready', diagnostic = NULL, updated_at = ?, finished_at = ?
                WHERE episode_id = ?
                """,
                (now, now, episode.episode_id),
            )
            connection.execute(
                """
                UPDATE episodes
                SET status = ?, wrapup_state = 'ready', wrapup_error = NULL,
                    updated_at = ?, ended_at = COALESCE(ended_at, ?)
                WHERE episode_id = ?
                """,
                (self._status_for_ending(report.ending), now, now, episode.episode_id),
            )
            self._terminalize_auto_research_child_experiment_with_notice(
                connection,
                child_episode_id=episode.episode_id,
                status=self._status_for_ending(report.ending),
                ending=report.ending,
                diagnostic=episode.ending_diagnostic,
                created_at=now,
            )
        stored_episode = self.episode(report.episode_id)
        stored_report = self.episode_report(report.episode_id)
        assert stored_episode is not None and stored_report is not None
        return stored_episode, stored_report

    def episode_report(self, episode_id: str) -> EpisodeReportRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM episode_reports WHERE episode_id = ?", (episode_id,)
            ).fetchone()
        return self._episode_report_record(row) if row is not None else None

    def episode_report_by_id(self, report_id: str) -> EpisodeReportRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM episode_reports WHERE report_id = ?", (report_id,)
            ).fetchone()
        return self._episode_report_record(row) if row is not None else None

    @staticmethod
    def _validate_new_episode(record: EpisodeRecord) -> None:
        if record.authorized_by is None:
            raise ValueError("a new episode requires its human authorization snapshot")
        if record.mode == "auto_research" and record.graph_target.kind != "branch":
            raise ValueError("a new Auto-research episode requires its persistent graph branch")
        if (
            record.status != "queued"
            or record.root_operation_id is not None
            or record.invocations_used != 0
            or record.ending is not None
            or record.wrapup_state != "not_started"
            or record.report_attempts_used != 0
            or record.stop_requested_at is not None
            or record.stop_settled_at is not None
            or record.ended_at is not None
        ):
            raise ValueError("a new episode must begin as an unused queued episode")

    def _created_episode_pair_matches(
        self,
        stored_episode: EpisodeRecord,
        requested_episode: EpisodeRecord,
        invocation_row: sqlite3.Row | None,
        task_row: sqlite3.Row | None,
        requested_task: AgentTaskRecord,
    ) -> bool:
        expected_episode = requested_episode.model_copy(
            update={
                "root_operation_id": requested_task.operation_id,
                "status": "running",
                "invocations_used": 1,
                "updated_at": requested_task.created_at,
            }
        )
        if (
            stored_episode != expected_episode
            or invocation_row is None
            or task_row is None
            or invocation_row["operation_id"] != requested_task.operation_id
            or invocation_row["created_at"] != requested_task.created_at
        ):
            return False
        stored_task = self._agent_task_record(task_row)
        return stored_task.model_dump(
            exclude=AGENT_TASK_PROJECTION_FIELDS
        ) == requested_task.model_dump(exclude=AGENT_TASK_PROJECTION_FIELDS)

    @staticmethod
    def _validate_new_wrapup(
        episode_id: str,
        wrapup: EpisodeWrapupRecord,
        task: AgentTaskRecord,
    ) -> None:
        if wrapup.episode_id != episode_id or wrapup.ending == "stopped":
            raise ValueError("a report wrap-up must name its non-Stop episode ending")
        if wrapup.state != "pending" or wrapup.finished_at is not None:
            raise ValueError("a new report wrap-up must begin pending")
        required = {
            "allocation_operation_id": wrapup.allocation_operation_id,
            "provider": wrapup.provider,
            "run_on": wrapup.run_on,
            "native_session_id": wrapup.native_session_id,
            "stage_root": wrapup.stage_root,
            "skill_id": wrapup.skill_id,
            "skill_version": wrapup.skill_version,
            "output_name": wrapup.output_name,
            "output_path": wrapup.output_path,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if wrapup.execution_host is None:
            missing.append("execution_host")
        if missing:
            raise ValueError("episode wrap-up restart fence is incomplete: " + ", ".join(missing))
        if (
            task.kind != "episode_report"
            or task.visible
            or task.operation_id != wrapup.allocation_operation_id
            or wrapup.concluding_operation_id is None
            or task.parent_operation_id != wrapup.concluding_operation_id
            or task.native_session_id != wrapup.native_session_id
            or (task.stage_host or "") != (wrapup.stage_host or "")
            or task.stage_root != wrapup.stage_root
            or task.episode_id != episode_id
        ):
            raise ValueError("the hidden report task does not match its wrap-up restart fence")
        for field in ("provider", "run_on", "execution_host"):
            if task.request.get(field) != getattr(wrapup, field):
                raise ValueError(f"the hidden report task changed its frozen {field}")

    @staticmethod
    def _insert_episode(connection: sqlite3.Connection, record: EpisodeRecord) -> None:
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, project_id, mode, control_node_id,
                graph_target_json, graph_base_head_json, root_operation_id, status,
                invocation_ceiling, invocations_used, authorized_space_id,
                authorized_user_id, authorized_display_name, stop_requested_at,
                stop_settled_at, ending, ending_diagnostic, wrapup_state,
                wrapup_error, report_attempts_used, created_at, updated_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.episode_id,
                record.project_id,
                record.mode,
                record.control_node_id,
                record.graph_target.model_dump_json(),
                record.graph_base_head.model_dump_json() if record.graph_base_head else None,
                record.root_operation_id,
                record.status,
                record.invocation_ceiling,
                record.invocations_used,
                record.authorized_by.space_id if record.authorized_by else None,
                record.authorized_by.user_id if record.authorized_by else None,
                record.authorized_by.display_name if record.authorized_by else None,
                record.stop_requested_at,
                record.stop_settled_at,
                record.ending,
                record.ending_diagnostic,
                record.wrapup_state,
                record.wrapup_error,
                record.report_attempts_used,
                record.created_at,
                record.updated_at,
                record.ended_at,
            ),
        )

    @staticmethod
    def _insert_episode_wrapup(
        connection: sqlite3.Connection,
        record: EpisodeWrapupRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO episode_wrapups (
                episode_id, ending, partial, concluding_operation_id,
                allocation_operation_id, provider, run_on, execution_host,
                native_session_id, stage_host, stage_root, skill_id, skill_version,
                output_name, output_path, receipt_json, receipt_sha256, state,
                diagnostic, created_at, updated_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.episode_id,
                record.ending,
                int(record.partial),
                record.concluding_operation_id,
                record.allocation_operation_id,
                record.provider,
                record.run_on,
                record.execution_host,
                record.native_session_id,
                record.stage_host,
                record.stage_root,
                record.skill_id,
                record.skill_version,
                record.output_name,
                record.output_path,
                record.receipt_json,
                record.receipt_sha256,
                record.state,
                record.diagnostic,
                record.created_at,
                record.updated_at,
                record.finished_at,
            ),
        )

    @staticmethod
    def _wrapup_fence_matches(
        stored: EpisodeWrapupRecord,
        requested: EpisodeWrapupRecord,
    ) -> bool:
        immutable_fields = (
            "episode_id",
            "ending",
            "partial",
            "concluding_operation_id",
            "allocation_operation_id",
            "provider",
            "run_on",
            "execution_host",
            "native_session_id",
            "stage_host",
            "stage_root",
            "skill_id",
            "skill_version",
            "output_name",
            "output_path",
            "receipt_json",
            "receipt_sha256",
            "created_at",
        )
        return all(
            getattr(stored, field) == getattr(requested, field) for field in immutable_fields
        )

    @staticmethod
    def _live_episode_row(
        connection: sqlite3.Connection,
        record: EpisodeRecord,
    ) -> sqlite3.Row | None:
        placeholders = ",".join("?" for _ in _LIVE_EPISODE_STATUSES)
        if record.mode == "auto_research":
            return connection.execute(
                f"""
                SELECT * FROM episodes
                WHERE project_id = ? AND mode = 'auto_research'
                  AND status IN ({placeholders})
                LIMIT 1
                """,
                (record.project_id, *_LIVE_EPISODE_STATUSES),
            ).fetchone()
        return connection.execute(
            f"""
            SELECT * FROM episodes
            WHERE project_id = ? AND mode = 'experiment_loop' AND control_node_id = ?
              AND status IN ({placeholders})
            LIMIT 1
            """,
            (record.project_id, record.control_node_id, *_LIVE_EPISODE_STATUSES),
        ).fetchone()

    @staticmethod
    def _episode_record(row: sqlite3.Row) -> EpisodeRecord:
        data = dict(row)
        data["graph_target"] = json.loads(
            data.pop("graph_target_json", '{"kind":"main","branch_id":null}')
        )
        graph_base_head_json = data.pop("graph_base_head_json", None)
        data["graph_base_head"] = (
            json.loads(graph_base_head_json) if graph_base_head_json is not None else None
        )
        data["authorized_by"] = _authorized_human_snapshot(data)
        data.pop("authorized_space_id", None)
        data.pop("authorized_user_id", None)
        data.pop("authorized_display_name", None)
        return EpisodeRecord.model_validate(data)

    @staticmethod
    def _episode_invocation_record(row: sqlite3.Row) -> EpisodeInvocationRecord:
        return EpisodeInvocationRecord.model_validate(dict(row))

    @staticmethod
    def _episode_report_attempt_record(row: sqlite3.Row) -> EpisodeReportAttemptRecord:
        return EpisodeReportAttemptRecord.model_validate(dict(row))

    @staticmethod
    def _episode_report_record(row: sqlite3.Row) -> EpisodeReportRecord:
        return EpisodeReportRecord.model_validate(dict(row))

    @staticmethod
    def _episode_wrapup_record(row: sqlite3.Row) -> EpisodeWrapupRecord:
        data = dict(row)
        data["partial"] = bool(data["partial"])
        return EpisodeWrapupRecord.model_validate(data)

    @staticmethod
    def _status_for_ending(ending: str) -> str:
        try:
            return {
                "completed": "completed",
                "exhausted": "needs_action",
                "human_pause": "needs_action",
                "failed": "failed",
                "stopped": "stopped",
            }[ending]
        except KeyError as exc:
            raise ValueError("the episode has no valid semantic ending") from exc


def migrate_legacy_episodes(connection: sqlite3.Connection) -> None:
    """One-way, idempotent copy from legacy parents into the episode ledger."""

    _migrate_campaign_episodes(connection)
    _migrate_experiment_episodes(connection)


def _migrate_campaign_episodes(connection: sqlite3.Connection) -> None:
    if not _legacy_table_exists(connection, "campaigns"):
        return
    has_reports = _legacy_table_exists(connection, "campaign_reports")
    has_invocations = _legacy_table_exists(connection, "campaign_invocations")
    campaigns = connection.execute(
        "SELECT * FROM campaigns ORDER BY created_at, campaign_id"
    ).fetchall()
    for campaign in campaigns:
        campaign_id = str(campaign["campaign_id"])
        reports = (
            connection.execute(
                """
                SELECT report.*, run.request_json, run.native_session_id,
                       run.stage_host, run.stage_root
                FROM campaign_reports AS report
                LEFT JOIN graph_runs AS run ON run.operation_id = report.operation_id
                WHERE report.campaign_id = ? AND report.ending != 'stopped'
                ORDER BY report.created_at DESC, report.rowid DESC
                """,
                (campaign_id,),
            ).fetchall()
            if has_reports
            else []
        )
        report = (
            None
            if campaign["ending"] == "stopped" or campaign["status"] == "stopped"
            else (reports[0] if reports else None)
        )
        invocation_rows = (
            connection.execute(
                """
                SELECT invocation.operation_id, invocation.created_at
                FROM campaign_invocations AS invocation
                JOIN graph_runs AS run ON run.operation_id = invocation.operation_id
                WHERE invocation.campaign_id = ? AND invocation.role != 'report'
                  AND run.attempt = 1
                ORDER BY invocation.created_at, invocation.rowid
                """,
                (campaign_id,),
            ).fetchall()
            if has_invocations
            else []
        )
        operational_ceiling = _legacy_campaign_operational_ceiling(
            connection,
            campaign,
            operational_used=len(invocation_rows),
        )
        ending = _campaign_ending(campaign, report)
        status = _campaign_status(campaign, ending, report is not None)
        wrapup_state = _legacy_wrapup_state(
            status=status, ending=ending, has_report=report is not None
        )
        ending_diagnostic = campaign["error"]
        authorizer = (
            campaign["authorized_space_id"],
            campaign["authorized_user_id"],
            campaign["authorized_display_name"],
        )
        if status in _LIVE_EPISODE_STATUSES and not _complete_authorizer(authorizer):
            status = "failed"
            ending = "failed"
            wrapup_state = "legacy_unavailable"
            ending_diagnostic = (
                "This legacy episode has no recoverable human authorization snapshot."
            )
        attempts_used = 1 if report is not None else 0
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, project_id, mode, control_node_id, root_operation_id, status,
                invocation_ceiling, invocations_used, authorized_space_id,
                authorized_user_id, authorized_display_name, stop_requested_at,
                stop_settled_at, ending, ending_diagnostic, wrapup_state,
                wrapup_error, report_attempts_used, created_at, updated_at, ended_at
            ) VALUES (?, ?, 'auto_research', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO NOTHING
            """,
            (
                campaign_id,
                campaign["project_id"],
                campaign["root_operation_id"],
                status,
                operational_ceiling,
                len(invocation_rows),
                campaign["authorized_space_id"],
                campaign["authorized_user_id"],
                campaign["authorized_display_name"],
                campaign["stop_requested_at"],
                campaign["ended_at"] if ending == "stopped" else None,
                ending,
                ending_diagnostic,
                wrapup_state,
                attempts_used,
                campaign["created_at"],
                campaign["updated_at"],
                campaign["ended_at"] if ending is not None else None,
            ),
        )
        for invocation_number, row in enumerate(invocation_rows, start=1):
            connection.execute(
                """
                INSERT INTO episode_invocations (
                    episode_id, operation_id, invocation_number, created_at
                ) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING
                """,
                (campaign_id, row["operation_id"], invocation_number, row["created_at"]),
            )
        if wrapup_state != "not_started":
            _migrate_campaign_wrapup(
                connection,
                campaign,
                report,
                ending,
                wrapup_state,
                diagnostic=ending_diagnostic,
            )
        report_operations: list[str] = []
        if has_reports:
            report_operations.extend(
                str(row["operation_id"])
                for row in connection.execute(
                    "SELECT operation_id FROM campaign_reports WHERE campaign_id = ?",
                    (campaign_id,),
                ).fetchall()
            )
        if has_invocations:
            report_operations.extend(
                str(row["operation_id"])
                for row in connection.execute(
                    """
                    SELECT operation_id FROM campaign_invocations
                    WHERE campaign_id = ? AND role = 'report'
                    """,
                    (campaign_id,),
                ).fetchall()
            )
        for operation_id in set(report_operations):
            connection.execute(
                "UPDATE graph_runs SET kind = 'episode_report', visible = 0 WHERE operation_id = ?",
                (operation_id,),
            )
            connection.execute(
                "UPDATE agent_usage SET task_kind = 'episode_report' WHERE operation_id = ?",
                (operation_id,),
            )


def _legacy_campaign_operational_ceiling(
    connection: sqlite3.Connection,
    campaign: sqlite3.Row,
    *,
    operational_used: int,
) -> int:
    """Remove the one report reservation carried by each legacy authorization cycle."""

    campaign_id = str(campaign["campaign_id"])
    completed_cycles = (
        int(
            connection.execute(
                "SELECT COUNT(*) FROM campaign_reports WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()[0]
        )
        if _legacy_table_exists(connection, "campaign_reports")
        else 0
    )
    allocated_cycles = (
        int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM campaign_invocations AS invocation
                JOIN graph_runs AS run ON run.operation_id = invocation.operation_id
                WHERE invocation.campaign_id = ? AND invocation.role = 'report'
                  AND run.attempt = 1
                """,
                (campaign_id,),
            ).fetchone()[0]
        )
        if _legacy_table_exists(connection, "campaign_invocations")
        else 0
    )
    represented_cycles = max(completed_cycles, allocated_cycles)
    active_without_report_allocation = (
        represented_cycles > 0
        and str(campaign["status"]) in {"queued", "running", "stopping", "wrapping_up"}
        and allocated_cycles <= completed_cycles
    )
    reserved_cycles = max(1, represented_cycles + int(active_without_report_allocation))
    converted = int(campaign["invocation_ceiling"]) - reserved_cycles
    return max(1, operational_used, converted)


def _migrate_campaign_wrapup(
    connection: sqlite3.Connection,
    campaign: sqlite3.Row,
    report: sqlite3.Row | None,
    ending: str,
    state: str,
    *,
    diagnostic: str | None,
) -> None:
    campaign_id = str(campaign["campaign_id"])
    receipt_json, receipt_sha256 = compact_episode_receipt(
        {"ending": ending, "episode_id": campaign_id, "legacy_source": "campaign"}
    )
    request = _json_object(report["request_json"]) if report is not None else {}
    stage_root = report["stage_root"] if report is not None else None
    output_name = "campaign-report.html" if report is not None else None
    output_path = (
        f"{str(stage_root).rstrip('/')}/{output_name}"
        if stage_root and output_name
        else output_name
    )
    allocation_operation_id = str(report["operation_id"]) if report is not None else None
    wrapup = EpisodeWrapupRecord(
        episode_id=campaign_id,
        ending=ending,
        partial=ending != "completed",
        concluding_operation_id=campaign["root_operation_id"],
        allocation_operation_id=allocation_operation_id,
        provider=request.get("provider") if isinstance(request.get("provider"), str) else None,
        run_on=request.get("run_on") if isinstance(request.get("run_on"), str) else None,
        execution_host=(
            request.get("execution_host")
            if isinstance(request.get("execution_host"), str)
            else None
        ),
        native_session_id=(report["native_session_id"] if report is not None else None),
        stage_host=report["stage_host"] if report is not None else None,
        stage_root=stage_root,
        skill_id="campaign-report" if report is not None else None,
        skill_version="legacy" if report is not None else None,
        output_name=output_name,
        output_path=output_path,
        receipt_json=receipt_json,
        receipt_sha256=receipt_sha256,
        state=state,
        diagnostic=diagnostic,
        created_at=campaign["updated_at"],
        updated_at=campaign["updated_at"],
        finished_at=campaign["ended_at"] or campaign["updated_at"],
    )
    connection.execute("SELECT 1 FROM episodes WHERE episode_id = ?", (campaign_id,)).fetchone()
    _insert_legacy_wrapup(connection, wrapup)
    if report is None:
        return
    report_id = str(report["report_id"])
    attempt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"rcp:legacy-campaign-report:{report_id}"))
    connection.execute(
        """
        INSERT INTO episode_report_attempts (
            attempt_id, episode_id, attempt_number, allocation_operation_id,
            status, created_at, updated_at, finished_at
        ) VALUES (?, ?, 1, ?, 'succeeded', ?, ?, ?) ON CONFLICT DO NOTHING
        """,
        (
            attempt_id,
            campaign_id,
            allocation_operation_id,
            report["created_at"],
            report["created_at"],
            report["created_at"],
        ),
    )
    connection.execute(
        """
        INSERT INTO episode_reports (
            report_id, episode_id, attempt_id, allocation_operation_id, ending,
            sha256, html, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING
        """,
        (
            report_id,
            campaign_id,
            attempt_id,
            allocation_operation_id,
            ending,
            report["sha256"],
            report["html"],
            report["created_at"],
        ),
    )


def _migrate_experiment_episodes(connection: sqlite3.Connection) -> None:
    has_legacy_state = _legacy_table_exists(connection, "experiment_episodes")
    if has_legacy_state:
        ids = connection.execute(
            """
            SELECT episode_id FROM experiment_episodes
            UNION
            SELECT DISTINCT json_extract(request_json, '$.control_episode_id') AS episode_id
            FROM graph_runs
            WHERE parent_operation_id IS NULL
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_type(request_json, '$.control_episode_id') = 'text'
            ORDER BY episode_id
            """
        ).fetchall()
    else:
        ids = connection.execute(
            """
            SELECT DISTINCT json_extract(request_json, '$.control_episode_id') AS episode_id
            FROM graph_runs
            WHERE parent_operation_id IS NULL
              AND json_extract(request_json, '$.patch_kind') = 'experiment_loop'
              AND json_type(request_json, '$.control_episode_id') = 'text'
            ORDER BY episode_id
            """
        ).fetchall()
    for identity in ids:
        episode_id = str(identity["episode_id"])
        existing = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if existing is not None:
            _remove_impossible_legacy_experiment_wrapup(connection, existing)
            continue
        legacy = (
            connection.execute(
                "SELECT * FROM experiment_episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if has_legacy_state
            else None
        )
        task_rows = connection.execute(
            """
            SELECT * FROM graph_runs
            WHERE json_extract(request_json, '$.control_episode_id') = ?
              AND parent_operation_id IS NULL
            ORDER BY created_at, rowid
            """,
            (episode_id,),
        ).fetchall()
        all_task_rows = connection.execute(
            """
            SELECT status FROM graph_runs
            WHERE json_extract(request_json, '$.control_episode_id') = ?
            """,
            (episode_id,),
        ).fetchall()
        if not task_rows and legacy is None:
            continue
        invocations: dict[int, sqlite3.Row] = {}
        ceiling = 1
        control_node_id = str(legacy["control_node_id"]) if legacy is not None else None
        for task in task_rows:
            request = _json_object(task["request_json"])
            invocation = request.get("control_invocation")
            requested_ceiling = request.get("control_invocation_ceiling")
            requested_control = request.get("control_node_id")
            if control_node_id is None and isinstance(requested_control, str) and requested_control:
                control_node_id = requested_control
            if isinstance(invocation, int) and not isinstance(invocation, bool) and invocation >= 1:
                invocations.setdefault(invocation, task)
            if (
                isinstance(requested_ceiling, int)
                and not isinstance(requested_ceiling, bool)
                and requested_ceiling >= 1
            ):
                ceiling = max(ceiling, requested_ceiling)
        if not control_node_id:
            continue
        ordered_invocations = sorted(invocations.items())
        used = min(len(ordered_invocations), ceiling)
        root = ordered_invocations[0][1] if ordered_invocations else None
        authorizer = (
            (
                root["authorized_space_id"],
                root["authorized_user_id"],
                root["authorized_display_name"],
            )
            if root is not None
            else (None, None, None)
        )
        active_task = any(
            task["status"] in {"queued", "running", "pausing", "paused", "failed", "interrupted"}
            for task in all_task_rows
        )
        live_watcher = connection.execute(
            """
            SELECT 1 FROM watchers
            WHERE episode_id = ? AND notified = 0
              AND status IN ('active', 'degraded', 'completed') LIMIT 1
            """,
            (episode_id,),
        ).fetchone()
        exit_receipt = connection.execute(
            """
            SELECT 1 FROM graph_run_receipts AS receipt
            JOIN graph_runs AS run ON run.operation_id = receipt.operation_id
            WHERE json_extract(run.request_json, '$.control_episode_id') = ?
              AND receipt.category = 'experiment_loop_exit' LIMIT 1
            """,
            (episode_id,),
        ).fetchone()
        exit_ending, exit_diagnostic = _legacy_experiment_exit_ending(
            connection,
            episode_id,
            control_node_id,
        )
        status, ending, wrapup_state, diagnostic, ended_at = _legacy_experiment_lifecycle(
            legacy,
            used=used,
            ceiling=ceiling,
            recoverable_task=active_task,
            watcher_active=live_watcher is not None,
            exited=exit_receipt is not None,
            exit_ending=exit_ending,
            exit_diagnostic=exit_diagnostic,
        )
        if status in _LIVE_EPISODE_STATUSES and not _complete_authorizer(authorizer):
            status, ending, wrapup_state, ended_at = (
                "failed",
                "failed",
                "legacy_unavailable",
                (legacy["updated_at"] if legacy is not None else root["updated_at"]),
            )
            diagnostic = "This legacy episode has no recoverable human authorization snapshot."
        created_at = legacy["created_at"] if legacy is not None else root["created_at"]
        updated_at = legacy["updated_at"] if legacy is not None else root["updated_at"]
        if status not in _LIVE_EPISODE_STATUSES and ended_at is None:
            ended_at = updated_at
        project_id = legacy["project_id"] if legacy is not None else root["project_id"]
        stop_requested = legacy["stop_requested_at"] if legacy is not None else None
        stop_settled = legacy["stop_settled_at"] if legacy is not None else None
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, project_id, mode, control_node_id, root_operation_id, status,
                invocation_ceiling, invocations_used, authorized_space_id,
                authorized_user_id, authorized_display_name, stop_requested_at,
                stop_settled_at, ending, ending_diagnostic, wrapup_state,
                wrapup_error, report_attempts_used, created_at, updated_at, ended_at
            ) VALUES (?, ?, 'experiment_loop', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?)
            ON CONFLICT(episode_id) DO NOTHING
            """,
            (
                episode_id,
                project_id,
                control_node_id,
                root["operation_id"] if root is not None else None,
                status,
                ceiling,
                used,
                authorizer[0],
                authorizer[1],
                authorizer[2],
                stop_requested,
                stop_settled,
                ending,
                diagnostic,
                wrapup_state,
                created_at,
                updated_at,
                ended_at,
            ),
        )
        for invocation_number, task in ordered_invocations:
            connection.execute(
                """
                INSERT INTO episode_invocations (
                    episode_id, operation_id, invocation_number, created_at
                ) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING
                """,
                (episode_id, task["operation_id"], invocation_number, task["created_at"]),
            )
        if wrapup_state != "not_started":
            receipt_json, receipt_sha256 = compact_episode_receipt(
                {
                    "control_node_id": control_node_id,
                    "ending": ending,
                    "episode_id": episode_id,
                    "legacy_source": "experiment_episode",
                }
            )
            concluding = ordered_invocations[-1][1]["operation_id"] if ordered_invocations else None
            _insert_legacy_wrapup(
                connection,
                EpisodeWrapupRecord(
                    episode_id=episode_id,
                    ending=ending,
                    partial=ending != "completed",
                    concluding_operation_id=concluding,
                    receipt_json=receipt_json,
                    receipt_sha256=receipt_sha256,
                    state=wrapup_state,
                    diagnostic=diagnostic,
                    created_at=updated_at,
                    updated_at=updated_at,
                    finished_at=ended_at or updated_at,
                ),
            )


def _remove_impossible_legacy_experiment_wrapup(
    connection: sqlite3.Connection,
    episode: sqlite3.Row,
) -> None:
    """Remove only the old migration row that contradicts a modern live parent."""

    if (
        episode["mode"] != "experiment_loop"
        or episode["status"] not in _LIVE_EPISODE_STATUSES
        or episode["ending"] is not None
        or episode["wrapup_state"] != "not_started"
        or episode["report_attempts_used"] != 0
    ):
        return
    wrapup = connection.execute(
        "SELECT * FROM episode_wrapups WHERE episode_id = ?", (episode["episode_id"],)
    ).fetchone()
    if wrapup is None or wrapup["state"] != "legacy_unavailable":
        return
    if any(
        wrapup[field] is not None
        for field in (
            "allocation_operation_id",
            "provider",
            "run_on",
            "execution_host",
            "native_session_id",
            "stage_host",
            "stage_root",
            "skill_id",
            "skill_version",
            "output_name",
            "output_path",
        )
    ):
        return
    if int(wrapup["partial"]) != int(wrapup["ending"] != "completed"):
        return
    receipt_json, receipt_sha256 = compact_episode_receipt(
        {
            "control_node_id": episode["control_node_id"],
            "ending": wrapup["ending"],
            "episode_id": episode["episode_id"],
            "legacy_source": "experiment_episode",
        }
    )
    if wrapup["receipt_json"] != receipt_json or wrapup["receipt_sha256"] != receipt_sha256:
        return
    if (
        connection.execute(
            "SELECT 1 FROM episode_report_attempts WHERE episode_id = ? LIMIT 1",
            (episode["episode_id"],),
        ).fetchone()
        is not None
        or connection.execute(
            "SELECT 1 FROM episode_reports WHERE episode_id = ? LIMIT 1",
            (episode["episode_id"],),
        ).fetchone()
        is not None
    ):
        return
    connection.execute("DELETE FROM episode_wrapups WHERE episode_id = ?", (episode["episode_id"],))


def _campaign_ending(campaign: sqlite3.Row, report: sqlite3.Row | None) -> str | None:
    ending = report["ending"] if report is not None else campaign["ending"]
    if ending in {"completed", "exhausted", "stopped", "failed"}:
        return str(ending)
    return {
        "succeeded": "completed",
        "stopped": "stopped",
        "failed": "failed",
        "needs_action": "failed",
    }.get(str(campaign["status"]))


def _campaign_status(campaign: sqlite3.Row, ending: str | None, has_report: bool) -> str:
    status = str(campaign["status"])
    if ending is None:
        return status
    if status == "wrapping_up" and not has_report:
        return _status_for_ending(ending)
    return _status_for_ending(ending)


def _legacy_wrapup_state(*, status: str, ending: str | None, has_report: bool) -> str:
    if ending == "stopped" or status == "stopped":
        return "skipped"
    if has_report:
        return "ready"
    if ending is not None:
        return "legacy_unavailable"
    return "not_started"


def _legacy_experiment_lifecycle(
    row: sqlite3.Row | None,
    *,
    used: int,
    ceiling: int,
    recoverable_task: bool,
    watcher_active: bool,
    exited: bool,
    exit_ending: str | None,
    exit_diagnostic: str | None,
) -> tuple[str, str | None, str, str | None, str | None]:
    diagnostic = row["session_diagnostic"] if row is not None else None
    if row is not None and row["stop_settled_at"] is not None:
        return "stopped", "stopped", "skipped", diagnostic, row["stop_settled_at"]
    if row is not None and row["stop_requested_at"] is not None:
        if recoverable_task or watcher_active:
            return "stopping", None, "not_started", diagnostic, None
        return "stopped", "stopped", "skipped", diagnostic, row["updated_at"]
    if exited:
        if exit_ending == "completed":
            return (
                "completed",
                "completed",
                "legacy_unavailable",
                diagnostic,
                row["updated_at"] if row is not None else None,
            )
        if exit_ending == "human_pause":
            return (
                "needs_action",
                "human_pause",
                "legacy_unavailable",
                diagnostic,
                row["updated_at"] if row is not None else None,
            )
        return (
            "needs_action",
            None,
            "legacy_unavailable",
            diagnostic
            or exit_diagnostic
            or "This pre-migration Experiment exit cannot be classified from retained data.",
            row["updated_at"] if row is not None else None,
        )
    if recoverable_task:
        return "running", None, "not_started", diagnostic, None
    if watcher_active and used >= ceiling:
        return (
            "needs_action",
            "exhausted",
            "legacy_unavailable",
            diagnostic,
            row["updated_at"] if row is not None else None,
        )
    if watcher_active:
        return "running", None, "not_started", diagnostic, None
    return (
        "completed",
        "completed",
        "legacy_unavailable",
        diagnostic,
        row["updated_at"] if row is not None else None,
    )


def _legacy_experiment_exit_ending(
    connection: sqlite3.Connection,
    episode_id: str,
    control_node_id: str,
) -> tuple[str | None, str | None]:
    """Recover a legacy exit meaning only when retained output proves it."""

    rows = connection.execute(
        """
        SELECT output.patch_json, run.result_json
        FROM graph_runs AS run
        LEFT JOIN graph_run_outputs AS output ON output.operation_id = run.operation_id
        WHERE json_extract(run.request_json, '$.control_episode_id') = ?
        ORDER BY run.created_at DESC, run.rowid DESC
        """,
        (episode_id,),
    ).fetchall()
    for row in rows:
        payload = _json_object(row["patch_json"])
        operations = payload.get("ops")
        if not isinstance(operations, list):
            continue
        if _ops_complete_experiment(operations, control_node_id):
            return "completed", None
        if _ops_pause_for_human(operations, control_node_id):
            return "human_pause", None
    for row in rows:
        result = _json_object(row["result_json"])
        graph_update = result.get("graph_update")
        if isinstance(graph_update, dict):
            proposals = graph_update.get("proposal_ids")
            if isinstance(proposals, list) and proposals:
                return "human_pause", None
    return (
        None,
        "This pre-migration Experiment exit has no retained Patch or result that proves "
        "completion versus a human-authority pause.",
    )


def _ops_complete_experiment(operations: list[object], control_node_id: str) -> bool:
    return any(
        isinstance(operation, dict)
        and operation.get("op") == "update_nodes"
        and any(
            isinstance(update, dict)
            and update.get("id") == control_node_id
            and isinstance(update.get("changes"), dict)
            and update["changes"].get("status") == "completed"
            for update in operation.get("nodes", [])
        )
        for operation in operations
    )


def _ops_pause_for_human(operations: list[object], control_node_id: str) -> bool:
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if operation.get("op") == "create_proposals" and operation.get("proposals"):
            return True
        if operation.get("op") == "update_nodes":
            for update in operation.get("nodes", []):
                if not isinstance(update, dict) or update.get("id") == control_node_id:
                    continue
                changes = update.get("changes")
                if isinstance(changes, dict) and changes.get("status") in {"ready", "revisit"}:
                    return True
    created_blockers = {
        node.get("id")
        for operation in operations
        if isinstance(operation, dict) and operation.get("op") == "create_nodes"
        for node in operation.get("nodes", [])
        if isinstance(node, dict) and node.get("type") == "blocker"
    }
    return any(
        isinstance(operation, dict)
        and operation.get("op") == "create_edges"
        and any(
            isinstance(edge, dict)
            and edge.get("source") == control_node_id
            and edge.get("relation") == "blocked_by"
            and edge.get("target") in created_blockers
            for edge in operation.get("edges", [])
        )
        for operation in operations
    )


def _insert_legacy_wrapup(
    connection: sqlite3.Connection,
    wrapup: EpisodeWrapupRecord,
) -> None:
    connection.execute(
        """
        INSERT INTO episode_wrapups (
            episode_id, ending, partial, concluding_operation_id,
            allocation_operation_id, provider, run_on, execution_host,
            native_session_id, stage_host, stage_root, skill_id, skill_version,
            output_name, output_path, receipt_json, receipt_sha256, state,
            diagnostic, created_at, updated_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(episode_id) DO NOTHING
        """,
        (
            wrapup.episode_id,
            wrapup.ending,
            int(wrapup.partial),
            wrapup.concluding_operation_id,
            wrapup.allocation_operation_id,
            wrapup.provider,
            wrapup.run_on,
            wrapup.execution_host,
            wrapup.native_session_id,
            wrapup.stage_host,
            wrapup.stage_root,
            wrapup.skill_id,
            wrapup.skill_version,
            wrapup.output_name,
            wrapup.output_path,
            wrapup.receipt_json,
            wrapup.receipt_sha256,
            wrapup.state,
            wrapup.diagnostic,
            wrapup.created_at,
            wrapup.updated_at,
            wrapup.finished_at,
        ),
    )


def _status_for_ending(ending: str | None) -> str:
    return {
        "completed": "completed",
        "exhausted": "needs_action",
        "human_pause": "needs_action",
        "failed": "failed",
        "stopped": "stopped",
    }.get(ending, "failed")


def _legacy_table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _authorized_human_snapshot(data: dict[str, object]) -> AuthorizedHuman | None:
    values = (
        data.get("authorized_space_id"),
        data.get("authorized_user_id"),
        data.get("authorized_display_name"),
    )
    if values == (None, None, None):
        return None
    if not _complete_authorizer(values):
        raise RuntimeError("The stored episode authorizer snapshot is incomplete.")
    try:
        return AuthorizedHuman(space_id=values[0], user_id=values[1], display_name=values[2])
    except ValueError as exc:
        raise RuntimeError("The stored episode authorizer snapshot is invalid.") from exc


def _complete_authorizer(values: tuple[object, object, object]) -> bool:
    return all(isinstance(value, str) and value for value in values)


def _json_object(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def compact_episode_receipt(receipt: dict[str, object]) -> tuple[str, str]:
    """Canonicalize the immutable minimal receipt and return it with its digest."""

    receipt_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    return receipt_json, hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()


def report_sha256(html: str) -> str:
    """Return the canonical digest for an immutable report artifact."""

    return hashlib.sha256(html.encode("utf-8")).hexdigest()
