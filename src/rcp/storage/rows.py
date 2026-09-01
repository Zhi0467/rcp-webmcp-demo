from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime

from rcp.core.authority import (
    AgentDispatchAuthority,
)
from rcp.core.models import (
    AuthorizedHuman,
)
from rcp.storage.models import (  # noqa: F401
    _EXPERIMENT_EPISODE_CONTEXT_CANDIDATE_ROLE,
    _EXPERIMENT_EPISODE_PINNED_FIELDS,
    _MISSING_EXPERIMENT_EPISODE_CONTEXT_DIAGNOSTIC,
    _PROJECT_ID_TABLES,
    ACTIVE_AGENT_TASK_STATUSES,
    AGENT_TASK_TRANSITIONS,
    AWAITING_HUMAN_AGENT_TASK_STATUSES,
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
from rcp.storage.request_compat import migrate_stored_task_request


def _agent_task_status_label(status: str, applied_revision: object) -> str:
    """Name the state a human reads, so no surface maps the status itself."""

    if status == "succeeded":
        return (
            f"Completed at revision {applied_revision}"
            if isinstance(applied_revision, int)
            else "Completed"
        )
    return {
        "queued": "Queued",
        "running": "Running in the background",
        "pausing": "Pausing",
        "paused": "Paused at checkpoint",
        "interrupted": "Interrupted",
    }.get(status, "Failed")


class RowMappingMixin:
    """Pure `sqlite3.Row` to record mappers."""

    @staticmethod
    def _space_user_record(row: sqlite3.Row) -> SpaceUserRecord:
        try:
            return SpaceUserRecord.model_validate(dict(row))
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("RCP space user record is invalid.") from exc

    @staticmethod
    def _chat_session_context_record(row: sqlite3.Row) -> ChatSessionContextRecord:
        return ChatSessionContextRecord.model_validate(dict(row))

    @staticmethod
    def _result_view_record(row: sqlite3.Row) -> ResultViewRecord:
        data = dict(row)
        data.pop("html", None)
        return ResultViewRecord.model_validate(data)

    @staticmethod
    def _project_record(row: sqlite3.Row) -> ProjectRecord:
        data = dict(row)
        data["state_remote"] = bool(data["state_remote"])
        if data["reachable"] is not None:
            data["reachable"] = bool(data["reachable"])
        return ProjectRecord.model_validate(data)

    @staticmethod
    def _watcher_record(row: sqlite3.Row) -> StoredWatcherRecord:
        data = dict(row)
        data["continuation"] = json.loads(data.pop("continuation_json"))
        data["graph_target"] = json.loads(
            data.pop("graph_target_json", '{"kind":"main","branch_id":null}')
        )
        data["notified"] = bool(data["notified"])
        graph_condition_json = data.pop("graph_condition_json", None)
        if graph_condition_json is None:
            data.pop("armed_revision", None)
            return WatcherRecord.model_validate(data)
        data.pop("check_command", None)
        data.pop("log_path", None)
        data.pop("cwd", None)
        data["last_evaluated_at"] = data.pop("last_checked_at", None)
        data.pop("last_exit_code", None)
        data.pop("last_error", None)
        data.pop("next_check_at", None)
        data.pop("consecutive_error_count", None)
        data.pop("group_id", None)
        data.pop("group_label", None)
        data["condition"] = json.loads(graph_condition_json)
        return GraphWatcherRecord.model_validate(data)

    @staticmethod
    def _experiment_episode_record(row: sqlite3.Row) -> ExperimentEpisodeRecord:
        data = dict(row)
        data["graph_target"] = json.loads(
            data.pop("graph_target_json", '{"kind":"main","branch_id":null}')
        )
        data["last_watcher_ids"] = json.loads(data.pop("last_watcher_ids_json"))
        data["context_baseline"] = json.loads(data.pop("context_baseline_json"))
        return ExperimentEpisodeRecord.model_validate(data)

    def _agent_task_record(self, row: sqlite3.Row) -> AgentTaskRecord:
        data = dict(row)
        recovery_abandoned = bool(data.pop("recovery_abandoned", False))
        data.pop("campaign_worker_handoffs_cleared_at", None)
        dispatch_json = data.pop("dispatch_authority_json", None)
        data["dispatch_authority"] = (
            AgentDispatchAuthority.model_validate_json(dispatch_json)
            if dispatch_json is not None
            else None
        )
        data["authorized_by"] = self._authorized_human_snapshot(data)
        data.pop("authorized_space_id", None)
        data.pop("authorized_user_id", None)
        data.pop("authorized_display_name", None)
        raw_request = json.loads(data.pop("request_json"))
        data["request"] = (
            migrate_stored_task_request(
                str(data.get("kind", "")),
                raw_request,
                operation_id=str(data.get("operation_id", "")) or None,
                warn=False,
            )
            if isinstance(raw_request, dict)
            else raw_request
        )
        data["graph_target"] = json.loads(
            data.pop("graph_target_json", '{"kind":"main","branch_id":null}')
        )
        result_json = data.pop("result_json", None)
        data["result"] = json.loads(result_json) if result_json else None
        status = data["status"]
        started = self._parse_time(data.get("started_at"))
        finished = self._parse_time(data.get("finished_at"))
        end = finished or datetime.now(UTC)
        elapsed = max(0.0, (end - started).total_seconds()) if started else 0.0
        estimate = max(1.0, float(data.get("estimate_seconds") or 300.0))
        if status == "succeeded":
            progress = 1.0
        elif not started:
            progress = 0.0
        elif elapsed <= estimate:
            progress = 0.85 * elapsed / estimate
        else:
            progress = 0.85 + 0.14 * (1.0 - math.exp(-(elapsed - estimate) / estimate))
        data["elapsed_seconds"] = round(elapsed, 1)
        data["progress"] = round(min(0.99, max(0.0, progress)), 4) if status != "succeeded" else 1.0
        active = status in ACTIVE_AGENT_TASK_STATUSES
        stage_ready = not data.get("stage_host") or bool(data.get("stage_root"))
        visible = bool(data.get("visible", True))
        history_only = bool(data.get("history_only", False))
        data["visible"] = visible
        data["history_only"] = history_only
        data["can_pause"] = (
            visible and not history_only and status in AGENT_TASK_TRANSITIONS["pausing"]
        )
        data["can_resume"] = (
            visible
            and not history_only
            and status in {"paused", "interrupted"}
            and bool(data.get("native_session_id"))
            and stage_ready
            and not recovery_abandoned
        )
        data["can_retry"] = (
            visible
            and not history_only
            and status in {"paused", "interrupted", "failed"}
            and not active
            and not recovery_abandoned
        )
        data["active"] = active
        data["awaiting_human"] = not history_only and status in AWAITING_HUMAN_AGENT_TASK_STATUSES
        data["queued"] = status == "queued"
        data["pausing"] = status == "pausing"
        data["paused"] = status == "paused"
        data["finished"] = status == "succeeded" or status in {"failed", "interrupted"}
        data["failed"] = status == "failed"
        data["settled"] = status == "succeeded"
        data["status_label"] = _agent_task_status_label(status, data.get("applied_revision"))
        if history_only:
            data["native_session_id"] = None
        if data.get("kind") == "branch_merge":
            # A merge retry is a new human dispatch against the then-current
            # main head, never recovery of an old native session or stage.
            data["can_resume"] = False
            data["can_retry"] = False
        return AgentTaskRecord.model_validate(data)

    @staticmethod
    def _agent_task_event_record(row: sqlite3.Row) -> AgentTaskEventRecord:
        data = dict(row)
        payload_json = data.pop("payload_json", None)
        data["payload"] = json.loads(payload_json) if payload_json else None
        return AgentTaskEventRecord.model_validate(data)

    @staticmethod
    def _authorized_human_snapshot(
        row: sqlite3.Row | dict[str, object],
    ) -> AuthorizedHuman | None:
        values = {
            "space_id": row["authorized_space_id"],
            "user_id": row["authorized_user_id"],
            "display_name": row["authorized_display_name"],
        }
        present = {name for name, value in values.items() if value is not None}
        if not present:
            return None
        if len(present) != len(values):
            raise RuntimeError(
                "Agent task authorizer snapshot is partial; refusing to infer identity."
            )
        try:
            return AuthorizedHuman.model_validate(values)
        except ValueError as exc:
            raise RuntimeError("Agent task authorizer snapshot is invalid.") from exc

    @staticmethod
    def _agent_usage_record(row: sqlite3.Row) -> AgentUsageRecord:
        data = dict(row)
        data["counted"] = bool(data["counted"])
        data["provider_fields"] = json.loads(data.pop("provider_fields_json"))
        return AgentUsageRecord.model_validate(data)

    @staticmethod
    def _parse_time(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
