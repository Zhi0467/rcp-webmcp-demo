from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, JsonValue

from rcp.storage.models import ProjectTransferImportRecord
from rcp.transfer.archive import (
    TransferArchiveAttribution,
    TransferGraphHead,
    TransferGraphTarget,
    inspect_transfer_table_inventory,
)
from rcp.transfer.records import (
    TransferArtifactReference,
    TransferAssistantHistory,
    TransferAutoResearchApplyResult,
    TransferAutoResearchChildAdmission,
    TransferAutoResearchChildExperiment,
    TransferAutoResearchChildExperimentRequest,
    TransferAutoResearchChildWork,
    TransferAutoResearchChildWorkAttempt,
    TransferAutoResearchCommand,
    TransferAutoResearchExperimentInvocation,
    TransferAutoResearchFinishReceipt,
    TransferAutoResearchHistory,
    TransferAutoResearchInboxReceipt,
    TransferAutoResearchInvocation,
    TransferAutoResearchLifecycleNotice,
    TransferAutoResearchMessage,
    TransferAutoResearchRecovery,
    TransferEpisodeInvocation,
    TransferEpisodeRecord,
    TransferEpisodeReport,
    TransferEpisodeReportAttempt,
    TransferEpisodeWrapup,
    TransferExperimentEpisodeHistory,
    TransferJsonDocument,
    TransferLocalId,
    TransferPaperDraft,
    TransferRecordBundle,
    TransferTaskContract,
    TransferTaskEvent,
    TransferTaskOutput,
    TransferTaskReceipt,
    TransferTaskRecord,
    TransferTaskUsage,
    TransferWatcherRecord,
    capture_task_request_history,
    validate_transfer_table_policy,
)

if TYPE_CHECKING:
    from rcp.transfer.project_files import TransferLegacyKeptResultView, TransferProjectFileCapture


def _json_value(raw: str) -> JsonValue:
    value: JsonValue = json.loads(raw)
    return value


def _json_object(raw: str | None) -> dict[str, JsonValue]:
    if raw is None:
        return {}
    value = _json_value(raw)
    if not isinstance(value, dict):
        raise ValueError("stored transfer JSON must be an object")
    return value


def _optional_json_document(raw: str | None) -> TransferJsonDocument | None:
    if raw is None:
        return None
    return TransferJsonDocument.capture_sanitized(_json_value(raw))


def _rows_by(
    rows: Iterable[sqlite3.Row],
    key: str,
) -> dict[str, list[sqlite3.Row]]:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


_TRANSFER_IMPORT_INERT_PROVIDER = "history-only"
_TRANSFER_IMPORT_INERT_STAGE_ROOT = "/history-only"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _history_json_value(value: object) -> object:
    """Expand typed JSON documents while retaining only inert history values."""

    if isinstance(value, TransferJsonDocument):
        return value.value()
    if isinstance(value, BaseModel):
        return _history_json_value(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): _history_json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_history_json_value(child) for child in value]
    return value


def _history_json_document(value: TransferJsonDocument) -> str:
    return _canonical_json(value.value())


def _transfer_import_digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _inert_watcher_continuation() -> str:
    return _canonical_json(
        {
            "provider": _TRANSFER_IMPORT_INERT_PROVIDER,
            "model": None,
            "reasoning": None,
            "run_on": _TRANSFER_IMPORT_INERT_PROVIDER,
            "patch_kind": "work",
            "workflow_ids": [],
            "skill_ids": [],
            "invoked_workflow_ids": [],
            "invoked_skill_ids": [],
            "resolved_skill_packages": [],
            "control_decision_bundle": [],
            "control_completion_criteria": [],
        }
    )


def _task_result_history(task: TransferTaskRecord) -> str | None:
    result: dict[str, object] = {}
    if task.assistant.answer is not None:
        result["answer"] = task.assistant.answer
    if task.assistant.trace_messages:
        result["trace_messages"] = list(task.assistant.trace_messages)
    if task.assistant.legacy_unlabelled_lines:
        result["messages"] = list(task.assistant.legacy_unlabelled_lines)
    if task.graph_updates:
        result["graph_updates"] = [item.value() for item in task.graph_updates]
    if task.artifacts:
        result["artifacts"] = [_history_json_value(item) for item in task.artifacts]
    return _canonical_json(result) if result else None


def _task_request_history(task: TransferTaskRecord) -> str:
    return _canonical_json(_history_json_value(task.request))


def _attribution_fields(
    attributions: Mapping[str, TransferArchiveAttribution],
    archive_actor_id: str | None,
) -> tuple[str | None, str | None, str | None]:
    if archive_actor_id is None:
        return None, None, None
    attribution = attributions.get(archive_actor_id)
    if attribution is None:
        raise ValueError("transfer history references an unknown archive attribution")
    actor = attribution.source_actor
    return actor.space_id, actor.user_id, actor.display_name


class ProjectTransferStoreMixin:
    """One read-only, snapshot-consistent projection of finished project history."""

    def export_project_transfer_records(
        self,
        project_id: str,
        *,
        attributions: tuple[TransferArchiveAttribution, ...],
    ) -> TransferRecordBundle:
        with self.connection() as connection:
            connection.execute("BEGIN")
            project = connection.execute(
                "SELECT project_id FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                raise KeyError(project_id)
            inventory = inspect_transfer_table_inventory(connection)
            validate_transfer_table_policy(inventory.project_linked_tables)
            self._validate_transfer_relational_integrity(connection, project_id)
            self._require_finished_transfer_state(connection, project_id)
            tasks = self._transfer_tasks(connection, project_id, attributions)
            watchers = self._transfer_watchers(connection, project_id)
            episodes = self._transfer_episodes(connection, project_id, attributions)
            paper = self._transfer_paper_draft(connection, project_id)
        return TransferRecordBundle(
            project_id=project_id,
            attributions=attributions,
            tasks=tasks,
            watchers=watchers,
            episodes=episodes,
            paper_draft=paper,
        )

    @staticmethod
    def _validate_transfer_relational_integrity(
        connection: sqlite3.Connection,
        project_id: str,
    ) -> None:
        ownership_checks = (
            (
                "SELECT usage.usage_id FROM agent_usage AS usage "
                "LEFT JOIN graph_runs AS run ON run.operation_id = usage.operation_id "
                "WHERE (usage.project_id = ? OR run.project_id = ?) "
                "AND (run.project_id IS NULL OR usage.project_id != run.project_id) LIMIT 1",
                "agent usage",
            ),
            (
                "SELECT child.worker_id FROM auto_research_child_work AS child "
                "LEFT JOIN episodes AS episode ON episode.episode_id = child.episode_id "
                "WHERE (child.project_id = ? OR episode.project_id = ?) "
                "AND (episode.project_id IS NULL OR child.project_id != episode.project_id) "
                "LIMIT 1",
                "Auto-research child Work",
            ),
            (
                "SELECT child.child_episode_id FROM auto_research_child_experiments AS child "
                "LEFT JOIN episodes AS episode "
                "ON episode.episode_id = child.auto_research_episode_id "
                "WHERE (child.project_id = ? OR episode.project_id = ?) "
                "AND (episode.project_id IS NULL OR child.project_id != episode.project_id) "
                "LIMIT 1",
                "Auto-research child Experiment",
            ),
            (
                "SELECT admission.admission_id "
                "FROM auto_research_child_admissions AS admission "
                "LEFT JOIN episodes AS episode ON episode.episode_id = admission.episode_id "
                "WHERE (admission.project_id = ? OR episode.project_id = ?) "
                "AND (episode.project_id IS NULL OR admission.project_id != episode.project_id) "
                "LIMIT 1",
                "Auto-research child admission",
            ),
        )
        for query, label in ownership_checks:
            if connection.execute(query, (project_id, project_id)).fetchone() is not None:
                raise ValueError(f"stored {label} belongs to conflicting projects")

    @staticmethod
    def _require_finished_transfer_state(
        connection: sqlite3.Connection,
        project_id: str,
    ) -> None:
        checks = (
            (
                "SELECT operation_id FROM graph_runs WHERE project_id = ? "
                "AND status NOT IN ('succeeded', 'failed', 'interrupted') LIMIT 1",
                "agent task",
            ),
            (
                "SELECT watcher_id FROM watchers WHERE project_id = ? "
                "AND (status NOT IN ('completed', 'stopped') "
                "OR (status = 'completed' AND notified = 0)) LIMIT 1",
                "watcher",
            ),
            (
                "SELECT episode_id FROM episodes WHERE project_id = ? "
                "AND status NOT IN ('completed', 'stopped', 'failed') LIMIT 1",
                "episode",
            ),
            (
                "SELECT attempt.attempt_id FROM episode_report_attempts AS attempt "
                "JOIN episodes AS episode ON episode.episode_id = attempt.episode_id "
                "WHERE episode.project_id = ? "
                "AND attempt.status NOT IN ('succeeded', 'failed') LIMIT 1",
                "episode report attempt",
            ),
            (
                "SELECT episode_id FROM episodes WHERE project_id = ? "
                "AND wrapup_state IN ('pending', 'running') LIMIT 1",
                "episode wrap-up",
            ),
            (
                "SELECT recovery.recovery_id FROM auto_research_recoveries AS recovery "
                "JOIN episodes AS episode ON episode.episode_id = recovery.episode_id "
                "WHERE episode.project_id = ? AND recovery.status = 'pending' LIMIT 1",
                "Auto-research recovery",
            ),
            (
                "SELECT child.child_episode_id FROM auto_research_child_experiments AS child "
                "WHERE child.project_id = ? AND child.state IN ('pending', 'running') LIMIT 1",
                "Auto-research child Experiment",
            ),
            (
                "SELECT admission.admission_id FROM auto_research_child_admissions AS admission "
                "WHERE admission.project_id = ? AND admission.state = 'accepted' LIMIT 1",
                "Auto-research child admission",
            ),
            (
                "SELECT notice.notice_id FROM auto_research_lifecycle_notices AS notice "
                "JOIN episodes AS episode ON episode.episode_id = notice.episode_id "
                "WHERE episode.project_id = ? AND notice.acknowledged_at IS NULL LIMIT 1",
                "Auto-research lifecycle notice",
            ),
            (
                "SELECT message.message_id FROM auto_research_messages AS message "
                "JOIN episodes AS episode ON episode.episode_id = message.episode_id "
                "WHERE episode.project_id = ? AND message.delivered_at IS NULL LIMIT 1",
                "Auto-research message",
            ),
        )
        for query, label in checks:
            row = connection.execute(query, (project_id,)).fetchone()
            if row is not None:
                raise ValueError(f"project transfer requires every {label} to be settled")

    @classmethod
    def _transfer_tasks(
        cls,
        connection: sqlite3.Connection,
        project_id: str,
        attributions: tuple[TransferArchiveAttribution, ...],
    ) -> tuple[TransferTaskRecord, ...]:
        task_rows = connection.execute(
            "SELECT * FROM graph_runs WHERE project_id = ? ORDER BY created_at, operation_id",
            (project_id,),
        ).fetchall()
        event_rows = connection.execute(
            "SELECT event.* FROM graph_run_events AS event "
            "JOIN graph_runs AS run ON run.operation_id = event.operation_id "
            "WHERE run.project_id = ? ORDER BY event.operation_id, event.event_id",
            (project_id,),
        ).fetchall()
        receipt_rows = connection.execute(
            "SELECT receipt.* FROM graph_run_receipts AS receipt "
            "JOIN graph_runs AS run ON run.operation_id = receipt.operation_id "
            "WHERE run.project_id = ? ORDER BY receipt.operation_id, receipt.receipt_id",
            (project_id,),
        ).fetchall()
        usage_rows = connection.execute(
            "SELECT usage.* FROM agent_usage AS usage "
            "JOIN graph_runs AS run ON run.operation_id = usage.operation_id "
            "WHERE run.project_id = ? ORDER BY usage.operation_id, usage.created_at, usage.usage_id",
            (project_id,),
        ).fetchall()
        contract_rows = connection.execute(
            "SELECT contract.* FROM graph_run_contracts AS contract "
            "JOIN graph_runs AS run ON run.operation_id = contract.operation_id "
            "WHERE run.project_id = ? ORDER BY contract.operation_id, contract.role",
            (project_id,),
        ).fetchall()
        output_rows = connection.execute(
            "SELECT output.* FROM graph_run_outputs AS output "
            "JOIN graph_runs AS run ON run.operation_id = output.operation_id "
            "WHERE run.project_id = ?",
            (project_id,),
        ).fetchall()
        events = _rows_by(event_rows, "operation_id")
        receipts = _rows_by(receipt_rows, "operation_id")
        usages = _rows_by(usage_rows, "operation_id")
        contracts = _rows_by(contract_rows, "operation_id")
        outputs = {str(row["operation_id"]): row for row in output_rows}

        records: list[TransferTaskRecord] = []
        for row in task_rows:
            operation_id = str(row["operation_id"])
            request = _json_object(row["request_json"])
            result = _json_object(row["result_json"])
            output = outputs.get(operation_id)
            records.append(
                TransferTaskRecord(
                    operation_id=operation_id,
                    kind=row["kind"],
                    status=row["status"],
                    request=capture_task_request_history(row["kind"], request),
                    assistant=cls._transfer_assistant_history(result),
                    error=row["error"],
                    applied_revision=row["applied_revision"],
                    graph_updates=cls._transfer_graph_updates(result),
                    attempt=row["attempt"],
                    parent_operation_id=row["parent_operation_id"],
                    episode_id=row["episode_id"],
                    graph_target=TransferGraphTarget.model_validate_json(row["graph_target_json"]),
                    authorized_by_attribution_id=cls._transfer_attribution_id(
                        row,
                        attributions,
                    ),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    status_message=row["status_message"],
                    events=tuple(
                        cls._transfer_task_event(project_id, item) for item in events[operation_id]
                    ),
                    receipts=tuple(
                        cls._transfer_task_receipt(project_id, item)
                        for item in receipts[operation_id]
                    ),
                    usage=tuple(cls._transfer_task_usage(item) for item in usages[operation_id]),
                    contracts=tuple(
                        TransferTaskContract(
                            role=item["role"],
                            content=item["content"],
                            sha256=item["sha256"],
                            created_at=item["created_at"],
                        )
                        for item in contracts[operation_id]
                    ),
                    output=(
                        TransferTaskOutput(
                            created_at=output["created_at"],
                            patch=TransferJsonDocument.capture_sanitized(
                                _json_value(output["patch_json"])
                            ),
                        )
                        if output is not None
                        else None
                    ),
                    artifacts=cls._transfer_artifacts(result),
                    visible=bool(row["visible"]),
                    history_only=True,
                )
            )
        return tuple(records)

    @staticmethod
    def _transfer_assistant_history(result: dict[str, JsonValue]) -> TransferAssistantHistory:
        answer = result.get("answer")
        traces = result.get("trace_messages")
        legacy = result.get("messages")
        return TransferAssistantHistory(
            answer=answer if isinstance(answer, str) else None,
            trace_messages=tuple(item for item in traces if isinstance(item, str))
            if isinstance(traces, list)
            else (),
            legacy_unlabelled_lines=tuple(item for item in legacy if isinstance(item, str))
            if isinstance(legacy, list)
            else (),
        )

    @staticmethod
    def _transfer_graph_updates(
        result: dict[str, JsonValue],
    ) -> tuple[TransferJsonDocument, ...]:
        updates = result.get("graph_updates")
        values = list(updates) if isinstance(updates, list) else []
        latest = result.get("graph_update")
        if latest is not None and not values:
            values.append(latest)
        return tuple(TransferJsonDocument.capture_sanitized(item) for item in values)

    @staticmethod
    def _transfer_artifacts(
        result: dict[str, JsonValue],
    ) -> tuple[TransferArtifactReference, ...]:
        raw = result.get("artifacts")
        if not isinstance(raw, list):
            return ()
        artifacts: list[TransferArtifactReference] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("stored artifact history must be an object")
            artifacts.append(
                TransferArtifactReference(
                    artifact_id=item.get("artifact_id"),
                    source_name=item.get("name"),
                    media_type=item.get("media_type"),
                    size_bytes=item.get("size_bytes"),
                    content_sha256=item.get("content_sha256"),
                    expires_at=item.get("expires_at"),
                    kept_filename=item.get("kept_filename"),
                    kept_at=item.get("kept_at"),
                )
            )
        return tuple(artifacts)

    @staticmethod
    def _transfer_task_event(project_id: str, row: sqlite3.Row) -> TransferTaskEvent:
        return TransferTaskEvent(
            identity=TransferLocalId(
                archive_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"rcp:transfer:{project_id}:graph_run_events:{row['event_id']}",
                    )
                ),
                source_table="graph_run_events",
                source_id=str(row["event_id"]),
            ),
            created_at=row["created_at"],
            level=row["level"],
            message=row["message"],
            event_kind=row["event_kind"],
            command_id=row["command_id"],
            command_verb=row["command_verb"],
            command_phase=row["command_phase"],
            payload=_optional_json_document(row["payload_json"]),
        )

    @staticmethod
    def _transfer_task_receipt(project_id: str, row: sqlite3.Row) -> TransferTaskReceipt:
        return TransferTaskReceipt(
            identity=TransferLocalId(
                archive_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"rcp:transfer:{project_id}:graph_run_receipts:{row['receipt_id']}",
                    )
                ),
                source_table="graph_run_receipts",
                source_id=str(row["receipt_id"]),
            ),
            created_at=row["created_at"],
            tier=row["tier"],
            category=row["category"],
            payload=TransferJsonDocument.capture_sanitized(_json_value(row["payload_json"])),
        )

    @staticmethod
    def _transfer_task_usage(row: sqlite3.Row) -> TransferTaskUsage:
        return TransferTaskUsage(
            usage_id=row["usage_id"],
            provider=row["provider"],
            model=row["model"],
            provider_profile=row["provider_profile"],
            provider_event_type=row["provider_event_type"],
            counted=bool(row["counted"]),
            count_reason=row["count_reason"],
            processed_input_tokens=row["processed_input_tokens"],
            generated_tokens=row["generated_tokens"],
            cached_input_tokens=row["cached_input_tokens"],
            cache_creation_input_tokens=row["cache_creation_input_tokens"],
            cache_write_input_tokens=row["cache_write_input_tokens"],
            reasoning_output_tokens=row["reasoning_output_tokens"],
            reported_input_tokens=row["reported_input_tokens"],
            reported_output_tokens=row["reported_output_tokens"],
            reported_total_tokens=row["reported_total_tokens"],
            provider_fields=TransferJsonDocument.capture_sanitized(
                _json_value(row["provider_fields_json"])
            ),
            created_at=row["created_at"],
        )

    @staticmethod
    def _transfer_attribution_id(
        row: sqlite3.Row,
        attributions: tuple[TransferArchiveAttribution, ...],
    ) -> str | None:
        space_id = row["authorized_space_id"]
        user_id = row["authorized_user_id"]
        display_name = row["authorized_display_name"]
        if (space_id, user_id, display_name) == (None, None, None):
            return None
        if any(value is None for value in (space_id, user_id, display_name)):
            raise ValueError("stored human attribution is incomplete")
        for attribution in attributions:
            actor = attribution.source_actor
            if (actor.space_id, actor.user_id) == (space_id, user_id):
                return attribution.archive_actor_id
        raise ValueError("stored history references an unmapped human attribution")

    @staticmethod
    def _transfer_watchers(
        connection: sqlite3.Connection,
        project_id: str,
    ) -> tuple[TransferWatcherRecord, ...]:
        rows = connection.execute(
            "SELECT * FROM watchers WHERE project_id = ? ORDER BY created_at, watcher_id",
            (project_id,),
        ).fetchall()
        return tuple(
            TransferWatcherRecord(
                watcher_id=row["watcher_id"],
                kind="graph" if row["graph_condition_json"] is not None else "external",
                origin_operation_id=row["origin_operation_id"],
                origin_task_kind=row["origin_task_kind"],
                chat_id=row["chat_id"],
                node_id=row["node_id"],
                episode_id=row["episode_id"],
                graph_target=TransferGraphTarget.model_validate_json(row["graph_target_json"]),
                status=row["status"],
                graph_condition=_optional_json_document(row["graph_condition_json"]),
                last_checked_at=row["last_checked_at"],
                last_exit_code=row["last_exit_code"],
                last_error=row["last_error"],
                consecutive_error_count=row["consecutive_error_count"],
                group_id=row["group_id"],
                group_label=row["group_label"],
                stopped_by=row["stopped_by"],
                stop_reason=row["stop_reason"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                stopped_at=row["stopped_at"],
                stop_operation_id=row["stop_operation_id"],
            )
            for row in rows
        )

    @classmethod
    def _transfer_episodes(
        cls,
        connection: sqlite3.Connection,
        project_id: str,
        attributions: tuple[TransferArchiveAttribution, ...],
    ) -> tuple[TransferEpisodeRecord, ...]:
        rows = connection.execute(
            "SELECT * FROM episodes WHERE project_id = ? ORDER BY created_at, episode_id",
            (project_id,),
        ).fetchall()
        return tuple(cls._transfer_episode(connection, row, attributions) for row in rows)

    @classmethod
    def _transfer_episode(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        attributions: tuple[TransferArchiveAttribution, ...],
    ) -> TransferEpisodeRecord:
        episode_id = str(row["episode_id"])
        invocations = connection.execute(
            "SELECT * FROM episode_invocations WHERE episode_id = ? ORDER BY invocation_number",
            (episode_id,),
        ).fetchall()
        attempts = connection.execute(
            "SELECT * FROM episode_report_attempts WHERE episode_id = ? ORDER BY attempt_number",
            (episode_id,),
        ).fetchall()
        wrapup = connection.execute(
            "SELECT * FROM episode_wrapups WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        report = connection.execute(
            "SELECT * FROM episode_reports WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        cls._validate_episode_report_history(row, attempts, wrapup, report)
        return TransferEpisodeRecord(
            episode_id=episode_id,
            mode=row["mode"],
            control_node_id=row["control_node_id"],
            graph_target=TransferGraphTarget.model_validate_json(row["graph_target_json"]),
            graph_base_head=(
                TransferGraphHead.model_validate_json(row["graph_base_head_json"])
                if row["graph_base_head_json"] is not None
                else None
            ),
            root_operation_id=row["root_operation_id"],
            status=row["status"],
            invocation_ceiling=row["invocation_ceiling"],
            invocations_used=row["invocations_used"],
            authorized_by_attribution_id=cls._transfer_attribution_id(row, attributions),
            ending=row["ending"],
            ending_diagnostic=row["ending_diagnostic"],
            wrapup_state=row["wrapup_state"],
            wrapup_error=row["wrapup_error"],
            report_attempts_used=row["report_attempts_used"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            ended_at=row["ended_at"],
            invocations=tuple(
                TransferEpisodeInvocation(
                    operation_id=item["operation_id"],
                    invocation_number=item["invocation_number"],
                    created_at=item["created_at"],
                )
                for item in invocations
            ),
            report_attempts=tuple(
                TransferEpisodeReportAttempt(
                    attempt_id=item["attempt_id"],
                    attempt_number=item["attempt_number"],
                    allocation_operation_id=item["allocation_operation_id"],
                    status=item["status"],
                    error=item["error"],
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                    finished_at=item["finished_at"],
                )
                for item in attempts
            ),
            wrapup=cls._transfer_episode_wrapup(wrapup),
            report=cls._transfer_episode_report(report),
            experiment=(
                cls._transfer_experiment_history(connection, episode_id)
                if row["mode"] == "experiment_loop"
                else None
            ),
            auto_research=(
                cls._transfer_auto_research_history(connection, episode_id, attributions)
                if row["mode"] == "auto_research"
                else None
            ),
        )

    @staticmethod
    def _validate_episode_report_history(
        episode: sqlite3.Row,
        attempts: list[sqlite3.Row],
        wrapup: sqlite3.Row | None,
        report: sqlite3.Row | None,
    ) -> None:
        wrapup_state = episode["wrapup_state"]
        if wrapup is None:
            if wrapup_state != "not_started" or attempts or report is not None:
                raise ValueError("stored episode report history is incomplete")
            return
        if wrapup["state"] != wrapup_state or wrapup["ending"] != episode["ending"]:
            raise ValueError("stored episode and wrap-up lifecycle disagree")
        if report is None:
            if wrapup_state == "ready" or any(item["status"] == "succeeded" for item in attempts):
                raise ValueError("stored succeeded episode report history is incomplete")
            return
        attempt = next(
            (item for item in attempts if item["attempt_id"] == report["attempt_id"]),
            None,
        )
        if (
            wrapup_state != "ready"
            or attempt is None
            or attempt["status"] != "succeeded"
            or report["ending"] != episode["ending"]
            or report["allocation_operation_id"] != attempt["allocation_operation_id"]
            or report["allocation_operation_id"] != wrapup["allocation_operation_id"]
        ):
            raise ValueError("stored episode report lineage is inconsistent")

    @staticmethod
    def _transfer_episode_wrapup(row: sqlite3.Row | None) -> TransferEpisodeWrapup | None:
        if row is None:
            return None
        if hashlib.sha256(row["receipt_json"].encode()).hexdigest() != row["receipt_sha256"]:
            raise ValueError("stored episode wrap-up receipt does not match its digest")
        return TransferEpisodeWrapup(
            ending=row["ending"],
            partial=bool(row["partial"]),
            concluding_operation_id=row["concluding_operation_id"],
            allocation_operation_id=row["allocation_operation_id"],
            provider=row["provider"],
            skill_id=row["skill_id"],
            skill_version=row["skill_version"],
            receipt=TransferJsonDocument.capture_sanitized(_json_value(row["receipt_json"])),
            state=row["state"],
            diagnostic=row["diagnostic"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
        )

    @staticmethod
    def _transfer_episode_report(row: sqlite3.Row | None) -> TransferEpisodeReport | None:
        if row is None:
            return None
        return TransferEpisodeReport(
            report_id=row["report_id"],
            attempt_id=row["attempt_id"],
            allocation_operation_id=row["allocation_operation_id"],
            ending=row["ending"],
            sha256=row["sha256"],
            html=row["html"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _transfer_experiment_history(
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> TransferExperimentEpisodeHistory:
        row = connection.execute(
            "SELECT * FROM experiment_episode_state WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise ValueError("Experiment episode state is missing")
        watcher_ids = _json_value(row["last_watcher_ids_json"])
        if not isinstance(watcher_ids, list) or any(
            not isinstance(item, str) for item in watcher_ids
        ):
            raise ValueError("stored Experiment watcher history is invalid")
        return TransferExperimentEpisodeHistory(
            provider=row["provider"],
            chat_id=row["chat_id"],
            last_turn_operation_id=row["last_turn_operation_id"],
            last_turn_invocation=row["last_turn_invocation"],
            last_graph_result=row["last_graph_result"],
            last_watcher_ids=tuple(watcher_ids),
            session_diagnostic=row["session_diagnostic"],
        )

    @classmethod
    def _transfer_auto_research_history(
        cls,
        connection: sqlite3.Connection,
        episode_id: str,
        attributions: tuple[TransferArchiveAttribution, ...],
    ) -> TransferAutoResearchHistory:
        metadata = connection.execute(
            "SELECT * FROM auto_research_episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if metadata is None:
            raise ValueError("Auto-research episode metadata is missing")
        invocation_rows = connection.execute(
            "SELECT * FROM auto_research_invocations WHERE episode_id = ? "
            "ORDER BY created_at, operation_id",
            (episode_id,),
        ).fetchall()
        message_rows = connection.execute(
            "SELECT * FROM auto_research_messages WHERE episode_id = ? "
            "ORDER BY created_at, message_id",
            (episode_id,),
        ).fetchall()
        recovery_rows = connection.execute(
            "SELECT * FROM auto_research_recoveries WHERE episode_id = ? "
            "ORDER BY created_at, recovery_id",
            (episode_id,),
        ).fetchall()
        work_rows = connection.execute(
            "SELECT * FROM auto_research_child_work WHERE episode_id = ? "
            "ORDER BY created_at, worker_id",
            (episode_id,),
        ).fetchall()
        experiment_rows = connection.execute(
            "SELECT * FROM auto_research_child_experiments "
            "WHERE auto_research_episode_id = ? ORDER BY created_at, child_episode_id",
            (episode_id,),
        ).fetchall()
        admission_rows = connection.execute(
            "SELECT * FROM auto_research_child_admissions WHERE episode_id = ? "
            "ORDER BY created_at, admission_id",
            (episode_id,),
        ).fetchall()
        notice_rows = connection.execute(
            "SELECT * FROM auto_research_lifecycle_notices WHERE episode_id = ? "
            "ORDER BY created_at, notice_id",
            (episode_id,),
        ).fetchall()
        inbox_rows = connection.execute(
            "SELECT * FROM auto_research_inbox_receipts WHERE episode_id = ? "
            "ORDER BY created_at, effect_id",
            (episode_id,),
        ).fetchall()
        finish_rows = connection.execute(
            "SELECT * FROM auto_research_finish_receipts WHERE episode_id = ? "
            "ORDER BY created_at, effect_id",
            (episode_id,),
        ).fetchall()
        apply_rows = connection.execute(
            "SELECT * FROM auto_research_apply_results WHERE episode_id = ? "
            "ORDER BY created_at, apply_id",
            (episode_id,),
        ).fetchall()
        command_rows = connection.execute(
            "SELECT * FROM auto_research_command_files WHERE episode_id = ? "
            "ORDER BY created_at, command_id",
            (episode_id,),
        ).fetchall()
        return TransferAutoResearchHistory(
            starting_instruction=metadata["starting_instruction"],
            created_at=metadata["created_at"],
            updated_at=metadata["updated_at"],
            invocations=tuple(
                TransferAutoResearchInvocation(
                    operation_id=row["operation_id"],
                    allocation_operation_id=row["allocation_operation_id"],
                    role=row["role"],
                    actor_operation_id=row["actor_operation_id"],
                    control_node_id=row["control_node_id"],
                    created_at=row["created_at"],
                )
                for row in invocation_rows
            ),
            messages=tuple(
                TransferAutoResearchMessage(
                    message_id=row["message_id"],
                    sender_role=row["sender_role"],
                    sender_task_id=row["sender_task_id"],
                    authorized_by_attribution_id=cls._transfer_attribution_id(
                        row,
                        attributions,
                    )
                    if row["authorized_space_id"] is not None
                    else None,
                    recipient_task_id=row["recipient_task_id"],
                    control_node_id=row["control_node_id"],
                    body=row["body"],
                    disposition="delivered",
                    created_at=row["created_at"],
                    delivered_at=row["delivered_at"],
                )
                for row in message_rows
            ),
            recoveries=tuple(
                TransferAutoResearchRecovery(
                    recovery_id=row["recovery_id"],
                    operation_id=row["operation_id"],
                    failure_kind=row["failure_kind"],
                    retry_mode=row["retry_mode"],
                    attempts=row["attempts"],
                    max_attempts=row["max_attempts"],
                    status=row["status"],
                    diagnostic=row["diagnostic"],
                    admitted_operation_id=row["admitted_operation_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in recovery_rows
            ),
            child_work=tuple(cls._transfer_child_work(connection, row) for row in work_rows),
            child_experiments=tuple(
                cls._transfer_child_experiment(connection, row) for row in experiment_rows
            ),
            child_admissions=tuple(
                TransferAutoResearchChildAdmission(
                    admission_id=row["admission_id"],
                    child_kind=row["child_kind"],
                    child_id=row["child_id"],
                    state=row["state"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in admission_rows
            ),
            lifecycle_notices=tuple(
                TransferAutoResearchLifecycleNotice(
                    notice_id=row["notice_id"],
                    source_kind=row["source_kind"],
                    source_id=row["source_id"],
                    source_event=row["source_event"],
                    source_attempt=row["source_attempt"],
                    payload=TransferJsonDocument.capture_sanitized(
                        _json_value(row["payload_json"])
                    ),
                    created_at=row["created_at"],
                    delivered_at=row["delivered_at"],
                    acknowledged_at=row["acknowledged_at"],
                    acknowledged_by=row["acknowledged_by"],
                )
                for row in notice_rows
            ),
            inbox_receipts=tuple(
                TransferAutoResearchInboxReceipt(
                    effect_id=row["effect_id"],
                    mode=row["mode"],
                    result=TransferJsonDocument.capture_sanitized(_json_value(row["result_json"])),
                    acknowledged_by=row["acknowledged_by"],
                    created_at=row["created_at"],
                )
                for row in inbox_rows
            ),
            finish_receipts=tuple(
                TransferAutoResearchFinishReceipt(
                    effect_id=row["effect_id"],
                    actor_operation_id=row["actor_operation_id"],
                    disposition=row["disposition"],
                    blocker_count=row["blocker_count"],
                    result=TransferJsonDocument.capture(_json_value(row["result_json"])),
                    result_sha256=row["result_sha256"],
                    created_at=row["created_at"],
                )
                for row in finish_rows
            ),
            apply_results=tuple(
                TransferAutoResearchApplyResult(
                    apply_id=row["apply_id"],
                    operation_id=row["operation_id"],
                    patch_sha256=row["patch_sha256"],
                    result=TransferJsonDocument.capture_sanitized(_json_value(row["result_json"])),
                    created_at=row["created_at"],
                )
                for row in apply_rows
            ),
            commands=tuple(
                TransferAutoResearchCommand(
                    command_id=row["command_id"],
                    operation_id=row["operation_id"],
                    kind=row["kind"],
                    filename=row["filename"],
                    sha256=row["sha256"],
                    content=row["content"],
                    created_at=row["created_at"],
                )
                for row in command_rows
            ),
        )

    @staticmethod
    def _transfer_child_work(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> TransferAutoResearchChildWork:
        attempts = connection.execute(
            "SELECT * FROM auto_research_child_work_attempts WHERE worker_id = ? "
            "ORDER BY created_at, operation_id",
            (row["worker_id"],),
        ).fetchall()
        return TransferAutoResearchChildWork(
            worker_id=row["worker_id"],
            control_node_id=row["control_node_id"],
            root_operation_id=row["root_operation_id"],
            final_operation_id=row["current_operation_id"],
            admitted_by_operation_id=row["admitted_by_operation_id"],
            instruction=row["instruction"],
            instruction_sha256=row["instruction_sha256"],
            attempts=tuple(
                TransferAutoResearchChildWorkAttempt(
                    operation_id=item["operation_id"],
                    allocation_operation_id=item["allocation_operation_id"],
                    created_at=item["created_at"],
                )
                for item in attempts
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _transfer_child_experiment(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> TransferAutoResearchChildExperiment:
        invocations = connection.execute(
            "SELECT * FROM auto_research_experiment_invocations WHERE child_episode_id = ? "
            "ORDER BY created_at, operation_id",
            (row["child_episode_id"],),
        ).fetchall()
        request = _json_object(row["request_json"])
        return TransferAutoResearchChildExperiment(
            child_episode_id=row["child_episode_id"],
            control_node_id=row["control_node_id"],
            state=row["state"],
            replaces_episode_id=row["replaces_episode_id"],
            request=TransferAutoResearchChildExperimentRequest(
                goal=request.get("goal"),
                invocation_limit=request.get("invocation_limit"),
            ),
            goal_sha256=row["goal_sha256"],
            parent_operation_id=row["parent_operation_id"],
            terminal_diagnostic=row["terminal_diagnostic"],
            invocations=tuple(
                TransferAutoResearchExperimentInvocation(
                    operation_id=item["operation_id"],
                    created_at=item["created_at"],
                )
                for item in invocations
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _transfer_paper_draft(
        connection: sqlite3.Connection,
        project_id: str,
    ) -> TransferPaperDraft | None:
        row = connection.execute(
            "SELECT * FROM paper_drafts WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            return None
        return TransferPaperDraft(
            content=row["content"],
            base_hash=row["base_hash"],
            ancestor_content=row["ancestor_content"],
            cursor_state=row["cursor_state"],
            updated_at=row["updated_at"],
        )

    def begin_project_transfer_import(
        self,
        request_id: str,
        *,
        archive_manifest_sha256: str,
        target_manifest_sha256: str,
        operational_payload_sha256: str,
        target_configuration_receipt: Mapping[str, object],
        capture: TransferProjectFileCapture,
        kept_result_view_html: Mapping[str, str],
    ) -> ProjectTransferImportRecord:
        """Insert one complete inert transfer corpus under one SQLite fence.

        This method deliberately does not register a project.  The importer
        owns file publication and calls ``complete_project_transfer_import``
        only after every target-side readback has succeeded.
        """

        from rcp.transfer.configuration import TransferTargetConfigurationReceipt
        from rcp.transfer.project_files import TransferProjectFileCapture

        canonical_request_id = self._transfer_uuid(request_id, "transfer import request identity")
        archive_digest = _transfer_import_digest(
            archive_manifest_sha256,
            "transfer archive manifest digest",
        )
        target_digest = _transfer_import_digest(
            target_manifest_sha256,
            "transfer target manifest digest",
        )
        payload_digest = _transfer_import_digest(
            operational_payload_sha256,
            "transfer operational payload digest",
        )
        configuration_receipt = TransferTargetConfigurationReceipt.model_validate_json(
            _canonical_json(dict(target_configuration_receipt))
        )
        configuration_json = _canonical_json(configuration_receipt.model_dump(mode="json"))
        normalized_capture = TransferProjectFileCapture.model_validate(capture)
        if canonical_request_id == normalized_capture.records.project_id:
            raise ValueError("transfer import request and project identities must differ")
        if (
            configuration_receipt.target_request_id != canonical_request_id
            or configuration_receipt.project_id != normalized_capture.project_id
            or configuration_receipt.archive_manifest_sha256 != archive_digest
            or configuration_receipt.target_manifest_sha256 != target_digest
        ):
            raise ValueError("target configuration receipt does not bind this import")
        html_by_filename = self._validate_transfer_view_html(
            normalized_capture.kept_result_views,
            kept_result_view_html,
        )
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                target_request = self._project_transfer_request_from_connection(
                    connection,
                    canonical_request_id,
                )
                self._validate_transfer_import_boundary(
                    connection,
                    target_request,
                    normalized_capture.project_id,
                )
                existing_row = connection.execute(
                    "SELECT * FROM project_transfer_imports WHERE request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if existing_row is not None:
                    existing = self._project_transfer_import_record(existing_row)
                    configuration_row = connection.execute(
                        "SELECT * FROM project_transfer_import_configurations WHERE request_id = ?",
                        (canonical_request_id,),
                    ).fetchone()
                    if configuration_row is None:
                        raise ValueError(
                            "project transfer import predates its resumable configuration receipt"
                        )
                    existing_configuration = (
                        self._project_transfer_import_configuration_receipt_json(configuration_row)
                    )
                    expected = (
                        normalized_capture.project_id,
                        archive_digest,
                        target_digest,
                        payload_digest,
                        configuration_json,
                    )
                    observed = (
                        existing.project_id,
                        existing.archive_manifest_sha256,
                        existing.target_manifest_sha256,
                        existing.operational_payload_sha256,
                        existing_configuration,
                    )
                    if observed != expected:
                        raise ValueError(
                            "project transfer import retry does not match its original digests"
                        )
                    return existing

                self._reject_transfer_import_collisions(
                    connection,
                    normalized_capture,
                )
                attributions = {
                    item.archive_actor_id: item for item in normalized_capture.records.attributions
                }
                event_id_map, receipt_id_map = self._insert_transfer_tasks(
                    connection,
                    normalized_capture.records,
                    attributions,
                )
                self._insert_transfer_watchers(connection, normalized_capture.records)
                self._insert_transfer_episodes(
                    connection,
                    normalized_capture.records,
                    attributions,
                )
                self._insert_transfer_views(
                    connection,
                    normalized_capture,
                    html_by_filename,
                )
                self._insert_transfer_paper(connection, normalized_capture.records)
                now = self.now()
                connection.execute(
                    """
                    INSERT INTO project_transfer_imports (
                        request_id, project_id, archive_manifest_sha256,
                        target_manifest_sha256, operational_payload_sha256, status,
                        event_id_map_json, receipt_id_map_json, publication_sha256,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, 'database_imported', ?, ?, NULL, ?, NULL)
                    """,
                    (
                        canonical_request_id,
                        normalized_capture.project_id,
                        archive_digest,
                        target_digest,
                        payload_digest,
                        _canonical_json(event_id_map),
                        _canonical_json(receipt_id_map),
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO project_transfer_import_configurations (
                        request_id, receipt_json, created_at
                    ) VALUES (?, ?, ?)
                    """,
                    (canonical_request_id, configuration_json, now),
                )
                row = connection.execute(
                    "SELECT * FROM project_transfer_imports WHERE request_id = ?",
                    (canonical_request_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("project transfer import receipt disappeared during insert")
                return self._project_transfer_import_record(row)
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "project transfer import conflicts with existing target state"
            ) from exc

    def complete_project_transfer_import(
        self,
        request_id: str,
        *,
        publication_sha256: str,
    ) -> ProjectTransferImportRecord:
        """Mark an imported corpus complete after external publication readback."""

        canonical_request_id = self._transfer_uuid(request_id, "transfer import request identity")
        publication_digest = _transfer_import_digest(
            publication_sha256,
            "transfer publication digest",
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM project_transfer_imports WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if row is None:
                raise KeyError(canonical_request_id)
            current = self._project_transfer_import_record(row)
            if current.status == "complete":
                if current.publication_sha256 != publication_digest:
                    raise ValueError(
                        "project transfer completion already binds another publication"
                    )
                return current
            if current.status != "database_imported":
                raise RuntimeError("project transfer import has an invalid completion state")
            target_request = self._project_transfer_request_from_connection(
                connection,
                canonical_request_id,
            )
            self._validate_transfer_import_boundary(
                connection,
                target_request,
                current.project_id,
            )
            completed_at = self.now()
            changed = connection.execute(
                """
                UPDATE project_transfer_imports
                SET status = 'complete', publication_sha256 = ?, completed_at = ?
                WHERE request_id = ? AND status = 'database_imported'
                """,
                (publication_digest, completed_at, canonical_request_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("project transfer import completion lost its transaction guard")
            updated = connection.execute(
                "SELECT * FROM project_transfer_imports WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
            if updated is None:
                raise RuntimeError("project transfer import receipt disappeared during completion")
            return self._project_transfer_import_record(updated)

    def project_transfer_import(
        self,
        request_id: str,
    ) -> ProjectTransferImportRecord | None:
        canonical_request_id = self._transfer_uuid(request_id, "transfer import request identity")
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM project_transfer_imports WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
        return self._project_transfer_import_record(row) if row is not None else None

    def project_transfer_import_configuration_receipt_json(
        self,
        request_id: str,
    ) -> str | None:
        """Return one validated configuration receipt captured before publication."""

        canonical_request_id = self._transfer_uuid(
            request_id,
            "transfer import request identity",
        )
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM project_transfer_import_configurations WHERE request_id = ?",
                (canonical_request_id,),
            ).fetchone()
        if row is None:
            return None
        return self._project_transfer_import_configuration_receipt_json(row)

    @staticmethod
    def _validate_transfer_view_html(
        views: tuple[TransferLegacyKeptResultView, ...],
        supplied: Mapping[str, str],
    ) -> dict[str, str]:
        if not isinstance(supplied, Mapping):
            raise ValueError("transfer kept result view HTML must be a mapping")
        expected_names = {view.kept_filename for view in views}
        expected_ids = {view.view_id for view in views}
        unknown = set(supplied) - expected_names - expected_ids
        if unknown:
            raise ValueError("transfer kept result view HTML contains an unknown file")
        result: dict[str, str] = {}
        for view in views:
            value = supplied.get(view.kept_filename)
            if value is None:
                value = supplied.get(view.view_id)
            if not isinstance(value, str):
                raise ValueError("transfer kept result view HTML is missing or not text")
            encoded = value.encode("utf-8")
            if (
                len(encoded) != view.size_bytes
                or hashlib.sha256(encoded).hexdigest() != view.content_sha256
            ):
                raise ValueError(
                    "transfer kept result view HTML does not match its captured digest"
                )
            result[view.kept_filename] = value
        return result

    @staticmethod
    def _validate_transfer_import_boundary(
        connection: sqlite3.Connection,
        target_request: object,
        project_id: str,
    ) -> None:
        if (
            getattr(target_request, "side", None) != "target"
            or getattr(target_request, "phase", None) != "archive_bound"
            or getattr(target_request, "project_id", None) != project_id
        ):
            raise ValueError("target transfer must be archive-bound for database import")
        request_id = getattr(target_request, "request_id", None)
        row = connection.execute(
            "SELECT * FROM project_provisioning_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise ValueError("linked incoming provisioning request is missing")
        provisioning = ProjectTransferStoreMixin._provisioning_record_for_import(row)
        if (
            provisioning.kind != "incoming_transfer"
            or provisioning.status != "ready_for_review"
            or provisioning.proposed_project_id != project_id
        ):
            raise ValueError("linked incoming provisioning request is not ready for review")

    @staticmethod
    def _provisioning_record_for_import(row: sqlite3.Row) -> object:
        # ``ProjectProvisioningStoreMixin`` owns this decoder.  Keeping this
        # narrow call here avoids importing its implementation into transfer
        # model code and still validates the persisted review digest.
        from rcp.storage.provisioning import ProjectProvisioningStoreMixin

        return ProjectProvisioningStoreMixin._project_provisioning_record(row)

    @staticmethod
    def _project_transfer_import_record(row: sqlite3.Row) -> ProjectTransferImportRecord:
        try:
            event_map = json.loads(row["event_id_map_json"])
            receipt_map = json.loads(row["receipt_id_map_json"])
            if not isinstance(event_map, dict) or not isinstance(receipt_map, dict):
                raise ValueError("import id maps must be objects")
            return ProjectTransferImportRecord(
                request_id=row["request_id"],
                project_id=row["project_id"],
                archive_manifest_sha256=row["archive_manifest_sha256"],
                target_manifest_sha256=row["target_manifest_sha256"],
                operational_payload_sha256=row["operational_payload_sha256"],
                status=row["status"],
                event_id_map={str(key): int(value) for key, value in event_map.items()},
                receipt_id_map={str(key): int(value) for key, value in receipt_map.items()},
                publication_sha256=row["publication_sha256"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("stored project transfer import receipt is invalid") from exc

    @staticmethod
    def _project_transfer_import_configuration_receipt_json(row: sqlite3.Row) -> str:
        from rcp.transfer.configuration import TransferTargetConfigurationReceipt

        try:
            receipt = TransferTargetConfigurationReceipt.model_validate_json(row["receipt_json"])
            if receipt.target_request_id != row["request_id"]:
                raise ValueError("configuration receipt belongs to another import")
            return _canonical_json(receipt.model_dump(mode="json"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "stored project transfer import configuration receipt is invalid"
            ) from exc

    @staticmethod
    def _transfer_uuid(value: str, label: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"{label} must be a canonical UUID4") from exc
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError(f"{label} must be a canonical UUID4")
        return value

    @staticmethod
    def _reject_transfer_import_collisions(
        connection: sqlite3.Connection,
        capture: TransferProjectFileCapture,
    ) -> None:
        """Reject every target row that could make this import ambiguous."""

        records = capture.records
        project_id = capture.project_id
        project_tables = (
            "projects",
            "paper_drafts",
            "result_views",
            "graph_runs",
            "episodes",
            "agent_usage",
            "watchers",
            "graph_watcher_reconciliation",
            "auto_research_child_work",
            "auto_research_child_experiments",
            "auto_research_child_admissions",
        )
        for table in project_tables:
            if (
                connection.execute(
                    f"SELECT 1 FROM {table} WHERE project_id = ? LIMIT 1",
                    (project_id,),
                ).fetchone()
                is not None
            ):
                raise ValueError(f"target already contains project history in {table}")

        keyed_ids: tuple[tuple[str, str, tuple[str, ...]], ...] = (
            (
                "graph_runs",
                "operation_id",
                tuple(task.operation_id for task in records.tasks),
            ),
            ("episodes", "episode_id", tuple(episode.episode_id for episode in records.episodes)),
            ("watchers", "watcher_id", tuple(watcher.watcher_id for watcher in records.watchers)),
            (
                "agent_usage",
                "usage_id",
                tuple(usage.usage_id for task in records.tasks for usage in task.usage),
            ),
            (
                "episode_report_attempts",
                "attempt_id",
                tuple(
                    attempt.attempt_id
                    for episode in records.episodes
                    for attempt in episode.report_attempts
                ),
            ),
            (
                "episode_reports",
                "report_id",
                tuple(
                    episode.report.report_id
                    for episode in records.episodes
                    if episode.report is not None
                ),
            ),
            (
                "auto_research_messages",
                "message_id",
                tuple(
                    message.message_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for message in episode.auto_research.messages
                ),
            ),
            (
                "auto_research_recoveries",
                "recovery_id",
                tuple(
                    recovery.recovery_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for recovery in episode.auto_research.recoveries
                ),
            ),
            (
                "auto_research_child_work",
                "worker_id",
                tuple(
                    child.worker_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for child in episode.auto_research.child_work
                ),
            ),
            (
                "auto_research_child_experiments",
                "child_episode_id",
                tuple(
                    child.child_episode_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for child in episode.auto_research.child_experiments
                ),
            ),
            (
                "auto_research_child_admissions",
                "admission_id",
                tuple(
                    admission.admission_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for admission in episode.auto_research.child_admissions
                ),
            ),
            (
                "auto_research_lifecycle_notices",
                "notice_id",
                tuple(
                    notice.notice_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for notice in episode.auto_research.lifecycle_notices
                ),
            ),
            (
                "auto_research_inbox_receipts",
                "effect_id",
                tuple(
                    receipt.effect_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for receipt in episode.auto_research.inbox_receipts
                ),
            ),
            (
                "auto_research_finish_receipts",
                "effect_id",
                tuple(
                    receipt.effect_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for receipt in episode.auto_research.finish_receipts
                ),
            ),
            (
                "auto_research_apply_results",
                "apply_id",
                tuple(
                    result.apply_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for result in episode.auto_research.apply_results
                ),
            ),
            (
                "auto_research_command_files",
                "command_id",
                tuple(
                    command.command_id
                    for episode in records.episodes
                    if episode.auto_research is not None
                    for command in episode.auto_research.commands
                ),
            ),
            (
                "result_views",
                "view_id",
                tuple(view.view_id for view in capture.kept_result_views),
            ),
        )
        for table, column, values in keyed_ids:
            if not values:
                continue
            placeholders = ",".join("?" for _ in values)
            row = connection.execute(
                f"SELECT {column} FROM {table} WHERE {column} IN ({placeholders}) LIMIT 1",
                values,
            ).fetchone()
            if row is not None:
                raise ValueError(f"target already contains imported identity in {table}")

        event_archive_ids = [
            event.identity.archive_id for task in records.tasks for event in task.events
        ]
        receipt_archive_ids = [
            receipt.identity.archive_id for task in records.tasks for receipt in task.receipts
        ]
        if len(event_archive_ids) != len(set(event_archive_ids)):
            raise ValueError("transfer import repeats one event archive identity")
        if len(receipt_archive_ids) != len(set(receipt_archive_ids)):
            raise ValueError("transfer import repeats one receipt archive identity")

    @classmethod
    def _insert_transfer_tasks(
        cls,
        connection: sqlite3.Connection,
        records: TransferRecordBundle,
        attributions: Mapping[str, TransferArchiveAttribution],
    ) -> tuple[dict[str, int], dict[str, int]]:
        event_id_map: dict[str, int] = {}
        receipt_id_map: dict[str, int] = {}
        for task in records.tasks:
            space_id, user_id, display_name = _attribution_fields(
                attributions,
                task.authorized_by_attribution_id,
            )
            connection.execute(
                """
                INSERT INTO graph_runs (
                    operation_id, project_id, episode_id, kind, status, request_json,
                    created_at, updated_at, started_at, finished_at, status_message,
                    error, applied_revision, result_json, attempt, parent_operation_id,
                    runtime_id, native_session_id, history_only, stage_host, stage_root,
                    graph_target_json, write_scope_fingerprint, estimate_seconds,
                    estimate_samples, phase, last_activity_at, dispatch_authority_json,
                    authorized_space_id, authorized_user_id, authorized_display_name, visible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', NULL, 1,
                          NULL, NULL, ?, NULL, 300, 0, 'finished', NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    task.operation_id,
                    records.project_id,
                    task.episode_id,
                    task.kind,
                    task.status,
                    _task_request_history(task),
                    task.created_at,
                    task.updated_at,
                    task.started_at,
                    task.finished_at,
                    task.status_message,
                    task.error,
                    task.applied_revision,
                    _task_result_history(task),
                    task.attempt,
                    task.parent_operation_id,
                    _canonical_json(task.graph_target.model_dump(mode="json")),
                    space_id,
                    user_id,
                    display_name,
                    int(task.visible),
                ),
            )
            for event in task.events:
                source_id = cls._source_local_integer(event.identity.source_id, "event")
                cursor = connection.execute(
                    """
                    INSERT INTO graph_run_events (
                        operation_id, created_at, level, message, event_kind,
                        command_id, episode_id, command_verb, command_phase,
                        idempotency_key, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        task.operation_id,
                        event.created_at,
                        event.level,
                        event.message,
                        event.event_kind,
                        event.command_id,
                        task.episode_id,
                        event.command_verb,
                        event.command_phase,
                        _history_json_document(event.payload)
                        if event.payload is not None
                        else None,
                    ),
                )
                target_id = cursor.lastrowid
                if target_id is None:
                    raise RuntimeError("target event id was not allocated")
                if source_id < 1:
                    raise ValueError("transfer event source id must be positive")
                event_id_map[event.identity.archive_id] = int(target_id)
            for receipt in task.receipts:
                source_id = cls._source_local_integer(receipt.identity.source_id, "receipt")
                cursor = connection.execute(
                    """
                    INSERT INTO graph_run_receipts (
                        operation_id, created_at, tier, category, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        task.operation_id,
                        receipt.created_at,
                        receipt.tier,
                        receipt.category,
                        _history_json_document(receipt.payload),
                    ),
                )
                target_id = cursor.lastrowid
                if target_id is None:
                    raise RuntimeError("target receipt id was not allocated")
                if source_id < 1:
                    raise ValueError("transfer receipt source id must be positive")
                receipt_id_map[receipt.identity.archive_id] = int(target_id)
            for usage in task.usage:
                connection.execute(
                    """
                    INSERT INTO agent_usage (
                        usage_id, project_id, operation_id, task_kind, provider, model,
                        provider_profile, provider_event_type, dedupe_key, counted,
                        count_reason, created_at, processed_input_tokens, generated_tokens,
                        cached_input_tokens, cache_creation_input_tokens,
                        cache_write_input_tokens, reasoning_output_tokens,
                        reported_input_tokens, reported_output_tokens, reported_total_tokens,
                        provider_fields_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        usage.usage_id,
                        records.project_id,
                        task.operation_id,
                        task.kind,
                        usage.provider,
                        usage.model,
                        usage.provider_profile,
                        usage.provider_event_type,
                        f"transfer:{usage.usage_id}",
                        int(usage.counted),
                        usage.count_reason,
                        usage.created_at,
                        usage.processed_input_tokens,
                        usage.generated_tokens,
                        usage.cached_input_tokens,
                        usage.cache_creation_input_tokens,
                        usage.cache_write_input_tokens,
                        usage.reasoning_output_tokens,
                        usage.reported_input_tokens,
                        usage.reported_output_tokens,
                        usage.reported_total_tokens,
                        _history_json_document(usage.provider_fields),
                    ),
                )
            for contract in task.contracts:
                connection.execute(
                    """
                    INSERT INTO graph_run_contracts (
                        operation_id, role, created_at, sha256, content
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        task.operation_id,
                        contract.role,
                        contract.created_at,
                        contract.sha256,
                        contract.content,
                    ),
                )
            if task.output is not None:
                connection.execute(
                    """
                    INSERT INTO graph_run_outputs (operation_id, created_at, patch_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        task.operation_id,
                        task.output.created_at,
                        _history_json_document(task.output.patch),
                    ),
                )
        return event_id_map, receipt_id_map

    @staticmethod
    def _source_local_integer(value: str, label: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"transfer {label} source id must be an integer") from exc
        return parsed

    @staticmethod
    def _insert_transfer_watchers(
        connection: sqlite3.Connection,
        records: TransferRecordBundle,
    ) -> None:
        continuation_json = _inert_watcher_continuation()
        for watcher in records.watchers:
            if watcher.kind == "graph":
                graph_condition_json = (
                    _history_json_document(watcher.graph_condition)
                    if watcher.graph_condition is not None
                    else None
                )
            else:
                graph_condition_json = None
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '', '', '', ?, NULL, ?, ?, ?, ?, ?, ?, ?,
                          NULL, ?, ?, ?, 1, NULL, ?, ?, ?, ?)
                """,
                (
                    watcher.watcher_id,
                    records.project_id,
                    watcher.origin_operation_id,
                    watcher.origin_task_kind,
                    watcher.chat_id,
                    watcher.node_id,
                    watcher.episode_id,
                    _canonical_json(watcher.graph_target.model_dump(mode="json")),
                    graph_condition_json,
                    continuation_json,
                    watcher.status,
                    watcher.created_at,
                    watcher.last_checked_at,
                    watcher.last_exit_code,
                    watcher.last_error,
                    watcher.completed_at,
                    watcher.consecutive_error_count,
                    watcher.group_id,
                    watcher.group_label,
                    watcher.stopped_by,
                    watcher.stop_reason,
                    watcher.stopped_at,
                    watcher.stop_operation_id,
                ),
            )

    @classmethod
    def _insert_transfer_episodes(
        cls,
        connection: sqlite3.Connection,
        records: TransferRecordBundle,
        attributions: Mapping[str, TransferArchiveAttribution],
    ) -> None:
        for episode in records.episodes:
            space_id, user_id, display_name = _attribution_fields(
                attributions,
                episode.authorized_by_attribution_id,
            )
            connection.execute(
                """
                INSERT INTO episodes (
                    episode_id, project_id, mode, control_node_id, graph_target_json,
                    graph_base_head_json, root_operation_id, status, invocation_ceiling,
                    invocations_used, authorized_space_id, authorized_user_id,
                    authorized_display_name, stop_requested_at, stop_settled_at,
                    ending, ending_diagnostic, wrapup_state, wrapup_error,
                    report_attempts_used, created_at, updated_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    records.project_id,
                    episode.mode,
                    episode.control_node_id,
                    _canonical_json(episode.graph_target.model_dump(mode="json")),
                    (
                        _canonical_json(episode.graph_base_head.model_dump(mode="json"))
                        if episode.graph_base_head is not None
                        else None
                    ),
                    episode.root_operation_id,
                    episode.status,
                    episode.invocation_ceiling,
                    episode.invocations_used,
                    space_id,
                    user_id,
                    display_name,
                    episode.ending,
                    episode.ending_diagnostic,
                    episode.wrapup_state,
                    episode.wrapup_error,
                    episode.report_attempts_used,
                    episode.created_at,
                    episode.updated_at,
                    episode.ended_at,
                ),
            )
            if episode.experiment is not None:
                experiment = episode.experiment
                connection.execute(
                    """
                    INSERT INTO experiment_episode_state (
                        episode_id, provider, execution_machine, execution_host,
                        native_session_id, stage_host, stage_root, chat_id,
                        last_turn_operation_id, last_turn_invocation, last_graph_result,
                        last_watcher_ids_json, context_baseline_json, session_diagnostic,
                        created_at, updated_at
                    ) VALUES (?, ?, NULL, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, '{}', ?, ?, ?)
                    """,
                    (
                        episode.episode_id,
                        experiment.provider,
                        _TRANSFER_IMPORT_INERT_PROVIDER,
                        experiment.chat_id,
                        experiment.last_turn_operation_id,
                        experiment.last_turn_invocation,
                        experiment.last_graph_result,
                        _canonical_json(list(experiment.last_watcher_ids)),
                        experiment.session_diagnostic,
                        episode.created_at,
                        episode.updated_at,
                    ),
                )
            if episode.auto_research is not None:
                cls._insert_auto_research_history(
                    connection,
                    records.project_id,
                    episode,
                    attributions,
                )
            for invocation in episode.invocations:
                connection.execute(
                    """
                    INSERT INTO episode_invocations (
                        episode_id, operation_id, invocation_number, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        episode.episode_id,
                        invocation.operation_id,
                        invocation.invocation_number,
                        invocation.created_at,
                    ),
                )
            for attempt in episode.report_attempts:
                connection.execute(
                    """
                    INSERT INTO episode_report_attempts (
                        attempt_id, episode_id, attempt_number, allocation_operation_id,
                        status, error, created_at, updated_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt.attempt_id,
                        episode.episode_id,
                        attempt.attempt_number,
                        attempt.allocation_operation_id,
                        attempt.status,
                        attempt.error,
                        attempt.created_at,
                        attempt.updated_at,
                        attempt.finished_at,
                    ),
                )
            if episode.wrapup is not None:
                wrapup = episode.wrapup
                connection.execute(
                    """
                    INSERT INTO episode_wrapups (
                        episode_id, ending, partial, concluding_operation_id,
                        allocation_operation_id, provider, run_on, execution_host,
                        native_session_id, stage_host, stage_root, skill_id, skill_version,
                        output_name, output_path, receipt_json, receipt_sha256, state,
                        diagnostic, created_at, updated_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, NULL, NULL,
                              ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode.episode_id,
                        wrapup.ending,
                        int(wrapup.partial),
                        wrapup.concluding_operation_id,
                        wrapup.allocation_operation_id,
                        wrapup.provider,
                        _TRANSFER_IMPORT_INERT_PROVIDER,
                        _TRANSFER_IMPORT_INERT_PROVIDER,
                        wrapup.skill_id,
                        wrapup.skill_version,
                        _history_json_document(wrapup.receipt),
                        wrapup.receipt.sha256,
                        wrapup.state,
                        wrapup.diagnostic,
                        wrapup.created_at,
                        wrapup.updated_at,
                        wrapup.finished_at,
                    ),
                )
            if episode.report is not None:
                report = episode.report
                connection.execute(
                    """
                    INSERT INTO episode_reports (
                        report_id, episode_id, attempt_id, allocation_operation_id,
                        ending, sha256, html, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report.report_id,
                        episode.episode_id,
                        report.attempt_id,
                        report.allocation_operation_id,
                        report.ending,
                        report.sha256,
                        report.html,
                        report.created_at,
                    ),
                )

    @classmethod
    def _insert_auto_research_history(
        cls,
        connection: sqlite3.Connection,
        project_id: str,
        episode: TransferEpisodeRecord,
        attributions: Mapping[str, TransferArchiveAttribution],
    ) -> None:
        history = episode.auto_research
        assert history is not None
        connection.execute(
            """
            INSERT INTO auto_research_episodes (
                episode_id, starting_instruction, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                history.starting_instruction,
                history.created_at,
                history.updated_at,
            ),
        )
        for invocation in history.invocations:
            connection.execute(
                """
                INSERT INTO auto_research_invocations (
                    episode_id, operation_id, allocation_operation_id, role,
                    actor_operation_id, control_node_id, handoffs_cleared_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    episode.episode_id,
                    invocation.operation_id,
                    invocation.allocation_operation_id,
                    invocation.role,
                    invocation.actor_operation_id,
                    invocation.control_node_id,
                    invocation.created_at,
                ),
            )
        for message in history.messages:
            space_id, user_id, display_name = _attribution_fields(
                attributions,
                message.authorized_by_attribution_id,
            )
            # Human sender attribution is resolved by the outer task/episode
            # import caller before this helper is reached.  A message's
            # archive id is retained in its task lineage; unknown agent
            # senders intentionally receive no human snapshot.
            connection.execute(
                """
                INSERT INTO auto_research_messages (
                    message_id, episode_id, sender_role, sender_task_id,
                    authorized_space_id, authorized_user_id, authorized_display_name,
                    recipient_task_id, control_node_id, body, created_at, delivered_at,
                    delivery_operation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    message.message_id,
                    episode.episode_id,
                    message.sender_role,
                    message.sender_task_id,
                    space_id,
                    user_id,
                    display_name,
                    message.recipient_task_id,
                    message.control_node_id,
                    message.body,
                    message.created_at,
                    message.delivered_at,
                ),
            )
        for recovery in history.recoveries:
            connection.execute(
                """
                INSERT INTO auto_research_recoveries (
                    recovery_id, episode_id, operation_id, failure_kind, retry_mode,
                    attempts, max_attempts, status, next_attempt_at, diagnostic,
                    admitted_operation_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
                """,
                (
                    recovery.recovery_id,
                    episode.episode_id,
                    recovery.operation_id,
                    recovery.failure_kind,
                    recovery.retry_mode,
                    recovery.attempts,
                    recovery.max_attempts,
                    recovery.status,
                    recovery.diagnostic,
                    recovery.admitted_operation_id,
                    recovery.created_at,
                    recovery.updated_at,
                ),
            )
        for child in history.child_work:
            connection.execute(
                """
                INSERT INTO auto_research_child_work (
                    worker_id, episode_id, project_id, control_node_id,
                    root_operation_id, current_operation_id, admitted_by_operation_id,
                    instruction, instruction_sha256, stop_requested_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    child.worker_id,
                    episode.episode_id,
                    project_id,
                    child.control_node_id,
                    child.root_operation_id,
                    child.final_operation_id,
                    child.admitted_by_operation_id,
                    child.instruction,
                    child.instruction_sha256,
                    child.created_at,
                    child.updated_at,
                ),
            )
            for attempt in child.attempts:
                connection.execute(
                    """
                    INSERT INTO auto_research_child_work_attempts (
                        operation_id, worker_id, allocation_operation_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        attempt.operation_id,
                        child.worker_id,
                        attempt.allocation_operation_id,
                        attempt.created_at,
                    ),
                )
        for child in history.child_experiments:
            connection.execute(
                """
                INSERT INTO auto_research_child_experiments (
                    child_episode_id, auto_research_episode_id, project_id,
                    control_node_id, state, replaces_episode_id, request_json,
                    goal_sha256, parent_operation_id, terminal_diagnostic, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    child.child_episode_id,
                    episode.episode_id,
                    project_id,
                    child.control_node_id,
                    child.state,
                    child.replaces_episode_id,
                    _canonical_json(_history_json_value(child.request)),
                    child.goal_sha256,
                    child.parent_operation_id,
                    child.terminal_diagnostic,
                    child.created_at,
                    child.updated_at,
                ),
            )
            for invocation in child.invocations:
                connection.execute(
                    """
                    INSERT INTO auto_research_experiment_invocations (
                        operation_id, auto_research_episode_id, child_episode_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        invocation.operation_id,
                        episode.episode_id,
                        child.child_episode_id,
                        invocation.created_at,
                    ),
                )
        for admission in history.child_admissions:
            connection.execute(
                """
                INSERT INTO auto_research_child_admissions (
                    admission_id, episode_id, project_id, child_kind, child_id,
                    state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    admission.admission_id,
                    episode.episode_id,
                    project_id,
                    admission.child_kind,
                    admission.child_id,
                    admission.state,
                    admission.created_at,
                    admission.updated_at,
                ),
            )
        for notice in history.lifecycle_notices:
            connection.execute(
                """
                INSERT INTO auto_research_lifecycle_notices (
                    notice_id, episode_id, source_kind, source_id, source_event,
                    source_attempt, state, payload_json, created_at, delivered_at,
                    delivery_operation_id, acknowledged_at, acknowledged_by
                ) VALUES (?, ?, ?, ?, ?, ?, 'acknowledged', ?, ?, ?, NULL, ?, ?)
                """,
                (
                    notice.notice_id,
                    episode.episode_id,
                    notice.source_kind,
                    notice.source_id,
                    notice.source_event,
                    notice.source_attempt,
                    _history_json_document(notice.payload),
                    notice.created_at,
                    notice.delivered_at,
                    notice.acknowledged_at,
                    notice.acknowledged_by,
                ),
            )
        for receipt in history.inbox_receipts:
            connection.execute(
                """
                INSERT INTO auto_research_inbox_receipts (
                    effect_id, episode_id, mode, result_json, acknowledged_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.effect_id,
                    episode.episode_id,
                    receipt.mode,
                    _history_json_document(receipt.result),
                    receipt.acknowledged_by,
                    receipt.created_at,
                ),
            )
        for receipt in history.finish_receipts:
            connection.execute(
                """
                INSERT INTO auto_research_finish_receipts (
                    effect_id, episode_id, actor_operation_id, disposition,
                    blocker_count, result_json, result_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.effect_id,
                    episode.episode_id,
                    receipt.actor_operation_id,
                    receipt.disposition,
                    receipt.blocker_count,
                    _history_json_document(receipt.result),
                    receipt.result_sha256,
                    receipt.created_at,
                ),
            )
        for result in history.apply_results:
            connection.execute(
                """
                INSERT INTO auto_research_apply_results (
                    apply_id, episode_id, operation_id, patch_sha256, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.apply_id,
                    episode.episode_id,
                    result.operation_id,
                    result.patch_sha256,
                    _history_json_document(result.result),
                    result.created_at,
                ),
            )
        for command in history.commands:
            connection.execute(
                """
                INSERT INTO auto_research_command_files (
                    command_id, episode_id, operation_id, kind, filename,
                    sha256, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command.command_id,
                    episode.episode_id,
                    command.operation_id,
                    command.kind,
                    command.filename,
                    command.sha256,
                    command.content,
                    command.created_at,
                ),
            )

    @staticmethod
    def _insert_transfer_views(
        connection: sqlite3.Connection,
        capture: TransferProjectFileCapture,
        html_by_filename: Mapping[str, str],
    ) -> None:
        for view in capture.kept_result_views:
            html = html_by_filename[view.kept_filename]
            connection.execute(
                """
                INSERT INTO result_views (
                    view_id, project_id, experiment_id, chat_id, origin_operation_id,
                    latest_operation_id, provider, model, reasoning, run_on,
                    native_session_id, stage_host, stage_root, source_name,
                    content_sha256, size_bytes, html, created_at, updated_at,
                    expires_at, kept_filename, kept_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    view.view_id,
                    capture.project_id,
                    view.experiment_id,
                    view.chat_id,
                    view.origin_operation_id,
                    view.latest_operation_id,
                    view.provider,
                    view.model,
                    view.reasoning,
                    _TRANSFER_IMPORT_INERT_PROVIDER,
                    _TRANSFER_IMPORT_INERT_PROVIDER,
                    _TRANSFER_IMPORT_INERT_PROVIDER,
                    _TRANSFER_IMPORT_INERT_STAGE_ROOT,
                    view.source_name,
                    view.content_sha256,
                    view.size_bytes,
                    html,
                    view.created_at,
                    view.updated_at,
                    view.expires_at,
                    view.kept_filename,
                    view.kept_at,
                ),
            )

    @staticmethod
    def _insert_transfer_paper(
        connection: sqlite3.Connection,
        records: TransferRecordBundle,
    ) -> None:
        if records.paper_draft is None:
            return
        draft = records.paper_draft
        connection.execute(
            """
            INSERT INTO paper_drafts (
                project_id, content, base_hash, updated_at, cursor_state, ancestor_content
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                records.project_id,
                draft.content,
                draft.base_hash,
                draft.updated_at,
                draft.cursor_state,
                draft.ancestor_content,
            ),
        )


__all__ = ["ProjectTransferStoreMixin"]
