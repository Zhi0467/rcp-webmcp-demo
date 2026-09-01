from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import rcp.transfer.records as transfer_records
from rcp.core.models import AuthorizedHuman
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.tasks.episode_report import EpisodeReportRunRequest
from rcp.service import CoachRequest, RunRequest
from rcp.storage import AppStore
from rcp.transfer import (
    TRANSFER_EXCLUDED_PROJECT_TABLES,
    TRANSFER_RECORD_TABLES,
    TransferArchiveActor,
    TransferArchiveAttribution,
    TransferGraphHead,
    TransferGraphTarget,
    inspect_transfer_table_inventory,
    validate_transfer_table_policy,
)
from rcp.transfer.records import (
    TransferArtifactReference,
    TransferAssistantHistory,
    TransferAutoResearchChildAdmission,
    TransferAutoResearchChildExperiment,
    TransferAutoResearchChildExperimentRequest,
    TransferAutoResearchHistory,
    TransferAutoResearchLifecycleNotice,
    TransferAutoResearchRecovery,
    TransferEpisodeInvocation,
    TransferEpisodeRecord,
    TransferEpisodeWrapup,
    TransferJsonDocument,
    TransferLocalId,
    TransferPaperCoachRequestHistory,
    TransferPaperDraft,
    TransferRecordBundle,
    TransferTaskEvent,
    TransferTaskOutput,
    TransferTaskReceipt,
    TransferTaskRecord,
    TransferWatcherRecord,
    capture_task_request_history,
    sanitize_transfer_history_json,
)

PROJECT_ID = "11111111-1111-4111-8111-111111111111"
SOURCE_SPACE_ID = "22222222-2222-4222-8222-222222222222"
SOURCE_USER_ID = "33333333-3333-4333-8333-333333333333"
ARCHIVE_ACTOR_ID = "44444444-4444-4444-8444-444444444444"
EPISODE_ID = "55555555-5555-4555-8555-555555555555"
NOW = "2026-08-31T12:00:00+00:00"


def _json(value: object) -> TransferJsonDocument:
    return TransferJsonDocument.capture(value)  # type: ignore[arg-type]


def _attribution() -> TransferArchiveAttribution:
    return TransferArchiveAttribution(
        archive_actor_id=ARCHIVE_ACTOR_ID,
        source_actor=TransferArchiveActor.capture(
            AuthorizedHuman(
                space_id=SOURCE_SPACE_ID,
                user_id=SOURCE_USER_ID,
                display_name="Z",
            )
        ),
    )


def _task(**changes: object) -> TransferTaskRecord:
    values: dict[str, object] = {
        "operation_id": "task-1",
        "kind": "paper_coach",
        "status": "succeeded",
        "request": TransferPaperCoachRequestHistory(
            message="Review the draft.",
            provider="codex",
            model="gpt-5.6-sol",
            reasoning="high",
        ),
        "assistant": TransferAssistantHistory(
            legacy_unlabelled_lines=("The completed coach answer.",)
        ),
        "attempt": 1,
        "authorized_by_attribution_id": ARCHIVE_ACTOR_ID,
        "created_at": NOW,
        "updated_at": NOW,
        "started_at": NOW,
        "finished_at": NOW,
        "status_message": "Finished",
    }
    values.update(changes)
    return TransferTaskRecord.model_validate(values)


def _experiment_episode(**changes: object) -> TransferEpisodeRecord:
    values: dict[str, object] = {
        "episode_id": EPISODE_ID,
        "mode": "experiment_loop",
        "control_node_id": "experiment-node",
        "status": "stopped",
        "invocation_ceiling": 2,
        "invocations_used": 1,
        "authorized_by_attribution_id": ARCHIVE_ACTOR_ID,
        "ending": "stopped",
        "wrapup_state": "skipped",
        "report_attempts_used": 0,
        "created_at": NOW,
        "updated_at": NOW,
        "ended_at": NOW,
        "invocations": (
            TransferEpisodeInvocation(
                operation_id="task-1",
                invocation_number=1,
                created_at=NOW,
            ),
        ),
        "wrapup": TransferEpisodeWrapup(
            ending="stopped",
            partial=False,
            receipt=_json({"ending": "stopped"}),
            state="skipped",
            created_at=NOW,
            updated_at=NOW,
            finished_at=NOW,
        ),
        "experiment": {"provider": "codex", "chat_id": "chat-1"},
    }
    values.update(changes)
    return TransferEpisodeRecord.model_validate(values)


def test_current_project_tables_have_one_explicit_transfer_disposition(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    with store.connection() as connection:
        inventory = inspect_transfer_table_inventory(connection)
        validate_transfer_table_policy(inventory.project_linked_tables)

        connection.execute(
            "CREATE TABLE future_project_history (project_id TEXT NOT NULL, value TEXT)"
        )
        changed = inspect_transfer_table_inventory(connection)

    assert TRANSFER_RECORD_TABLES.isdisjoint(TRANSFER_EXCLUDED_PROJECT_TABLES)
    assert len(TRANSFER_RECORD_TABLES) == 28
    assert len(TRANSFER_EXCLUDED_PROJECT_TABLES) == 17
    with pytest.raises(ValueError, match="future_project_history"):
        validate_transfer_table_policy(changed.project_linked_tables)


def test_transfer_table_policy_names_history_and_exclusions_deliberately() -> None:
    assert {
        "graph_runs",
        "watchers",
        "episodes",
        "experiment_episode_state",
        "auto_research_messages",
        "paper_drafts",
    } <= TRANSFER_RECORD_TABLES
    assert {
        "writing_sessions",
        "chat_session_contexts",
        "result_views",
        "graph_watcher_reconciliation",
        "project_members",
        "project_invitations",
        "project_transfer_requests",
        "project_transfer_proofs",
        "project_transfer_activations",
        "project_transfer_import_configurations",
        "project_transfer_restore_reentries",
        "project_transfer_uploads",
        "projects",
    } <= TRANSFER_EXCLUDED_PROJECT_TABLES


def test_history_json_is_canonical_immutable_and_rejects_live_bindings() -> None:
    source = {"nested": {"b": 2, "a": [1]}}
    document = _json(source)
    source["nested"]["a"].append(3)  # type: ignore[index, union-attr]

    assert document.canonical_json == '{"nested":{"a":[1],"b":2}}'
    assert document.sha256 == hashlib.sha256(document.canonical_json.encode()).hexdigest()
    first = document.value()
    assert first == {"nested": {"a": [1], "b": 2}}
    assert document.value() is not first

    for field in sorted(transfer_records.TRANSFER_EXECUTABLE_JSON_FIELDS):
        with pytest.raises(ValueError, match="executable field"):
            _json({"safe": [{field: "source binding"}]})

    sanitized = sanitize_transfer_history_json(
        {"message": "kept", "nested": {"session_id": "removed", "count": 1}}
    )
    assert sanitized == {"message": "kept", "nested": {"count": 1}}

    assert {
        "session_id",
        "run_on",
        "can_retry",
        "retryable",
        "attachments",
        "result_view",
    } <= transfer_records.TRANSFER_EXECUTABLE_JSON_FIELDS


def test_real_task_requests_project_to_typed_history_without_resume_bindings() -> None:
    episode_id = "77777777-7777-4777-8777-777777777777"
    raw_requests = (
        (
            "node_chat",
            RunRequest(
                provider="codex",
                model="gpt-5.6-sol",
                reasoning="high",
                run_on="source-gpu",
                node_id="node-1",
                message="Question",
                chat_id="chat-1",
                session_id="native-chat",
                attachment_set_id="set-1",
                attachment_client_id="client-1",
                attachment_batch_id="batch-1",
            ).model_dump(mode="json"),
        ),
        (
            "paper_coach",
            CoachRequest(
                message="Review this.",
                provider="codex",
                model="gpt-5.6-sol",
                reasoning="high",
                run_on="source-gpu",
                session_id="native-paper",
            ).model_dump(mode="json"),
        ),
        (
            "auto_research",
            AutoResearchRunRequest(
                episode_id=episode_id,
                role="orchestrator",
                provider="codex",
                model="gpt-5.6-sol",
                reasoning="high",
                run_on="source-gpu",
                session_id="native-auto",
                wake_cause="watcher",
                watcher_ids=["watcher-1"],
            ).model_dump(mode="json"),
        ),
        (
            "episode_report",
            EpisodeReportRunRequest(
                episode_id=episode_id,
                provider="codex",
                model="gpt-5.6-sol",
                reasoning="high",
                run_on="source-gpu",
                execution_host="source.example",
                session_id="native-report",
            ).model_dump(mode="json"),
        ),
    )

    for kind, request in raw_requests:
        history = capture_task_request_history(kind, request)  # type: ignore[arg-type]
        serialized = history.model_dump(mode="json")
        assert {
            "run_on",
            "session_id",
            "execution_host",
            "watcher_ids",
            "attachments",
            "attachment_set_id",
            "attachment_client_id",
            "attachment_batch_id",
        }.isdisjoint(serialized)


def test_terminal_task_preserves_unlabelled_legacy_answer_without_relabelling() -> None:
    task = _task(
        events=(
            TransferTaskEvent(
                identity=TransferLocalId(
                    archive_id=str(uuid.uuid4()),
                    source_table="graph_run_events",
                    source_id="41",
                ),
                created_at=NOW,
                level="info",
                message="Finished",
            ),
        ),
        receipts=(
            TransferTaskReceipt(
                identity=TransferLocalId(
                    archive_id=str(uuid.uuid4()),
                    source_table="graph_run_receipts",
                    source_id="7",
                ),
                created_at=NOW,
                tier="summary",
                category="answer",
                payload=_json({"accepted": True}),
            ),
        ),
        output=TransferTaskOutput(created_at=NOW, patch=_json({"operations": []})),
        artifacts=(
            TransferArtifactReference(
                artifact_id="a" * 24,
                source_name="figure.png",
                media_type="image/png",
                size_bytes=123,
                content_sha256="b" * 64,
                expires_at=NOW,
                kept_filename="figure.png",
                kept_at=NOW,
            ),
        ),
    )

    assert task.assistant.answer is None
    assert task.assistant.legacy_unlabelled_lines == ("The completed coach answer.",)
    assert task.history_only is True
    assert task.events[0].identity.source_id == "41"
    assert task.artifacts[0].kept_filename == "figure.png"
    with pytest.raises(ValidationError):
        task.assistant = TransferAssistantHistory(answer="changed")  # type: ignore[misc]

    for status in ("queued", "running", "pausing", "paused"):
        with pytest.raises(ValidationError):
            _task(status=status)
    with pytest.raises(ValidationError):
        _task(history_only=False)


def test_terminal_watcher_drops_shell_and_delivery_bindings() -> None:
    watcher = TransferWatcherRecord(
        watcher_id="watcher-1",
        kind="external",
        origin_operation_id="task-1",
        origin_task_kind="node_chat",
        chat_id="chat-1",
        status="completed",
        last_checked_at=NOW,
        last_exit_code=0,
        consecutive_error_count=0,
        created_at=NOW,
        completed_at=NOW,
    )
    assert watcher.status == "completed"
    assert {
        "execution_host",
        "check_command",
        "log_path",
        "cwd",
        "continuation",
        "next_check_at",
        "notification_operation_id",
    }.isdisjoint(TransferWatcherRecord.model_fields)
    with pytest.raises(ValidationError):
        TransferWatcherRecord.model_validate({**watcher.model_dump(), "status": "active"})


def test_artifact_history_supports_unkept_metadata_and_rejects_paths() -> None:
    unkept = TransferArtifactReference(
        artifact_id="a" * 24,
        source_name="preview.png",
        media_type="image/png",
        size_bytes=None,
    )
    assert unkept.content_sha256 is None
    assert unkept.kept_filename is None

    for name in ("../outside.png", "/tmp/outside.png", r"..\outside.png"):
        with pytest.raises(ValidationError, match="direct filename"):
            TransferArtifactReference(
                artifact_id="a" * 24,
                source_name=name,
                media_type="image/png",
            )
    kept = TransferArtifactReference(
        artifact_id="a" * 24,
        source_name="preview.png",
        media_type="image/png",
        kept_filename="preview.png",
        kept_at=NOW,
    )
    assert kept.content_sha256 is None


def test_episode_records_accept_only_settled_mode_history() -> None:
    episode = _experiment_episode()
    assert episode.experiment is not None
    assert episode.experiment.chat_id == "chat-1"
    assert episode.wrapup is not None and episode.wrapup.state == "skipped"

    for status in ("queued", "running", "stopping", "wrapping_up", "needs_action"):
        with pytest.raises(ValidationError):
            _experiment_episode(status=status)

    auto_episode_id = "66666666-6666-4666-8666-666666666666"
    auto = TransferEpisodeRecord(
        episode_id=auto_episode_id,
        mode="auto_research",
        graph_target=TransferGraphTarget(kind="branch", branch_id=auto_episode_id),
        graph_base_head=TransferGraphHead(revision=4, transition_id="c" * 64),
        status="completed",
        invocation_ceiling=5,
        invocations_used=0,
        ending="completed",
        wrapup_state="not_started",
        report_attempts_used=0,
        created_at=NOW,
        updated_at=NOW,
        ended_at=NOW,
        auto_research=TransferAutoResearchHistory(created_at=NOW, updated_at=NOW),
    )
    assert auto.graph_target.branch_id == auto.episode_id

    with pytest.raises(ValidationError):
        TransferAutoResearchRecovery(
            recovery_id="recovery-1",
            operation_id="task-1",
            failure_kind="provider",
            retry_mode="exact",
            attempts=1,
            max_attempts=2,
            status="pending",
            diagnostic="waiting",
            created_at=NOW,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError):
        TransferAutoResearchChildExperiment(
            child_episode_id="child-1",
            control_node_id="node-1",
            state="running",
            request=TransferAutoResearchChildExperimentRequest(),
            parent_operation_id="task-1",
            created_at=NOW,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError):
        TransferAutoResearchChildAdmission(
            admission_id="admission-1",
            child_kind="work",
            child_id="worker-1",
            state="accepted",
            created_at=NOW,
            updated_at=NOW,
        )


def test_lifecycle_notice_requires_completed_delivery_and_acknowledgement() -> None:
    notice = TransferAutoResearchLifecycleNotice(
        notice_id="notice-1",
        source_kind="worker",
        source_id="worker-1",
        source_event="finished",
        source_attempt=1,
        payload=_json({"status": "succeeded"}),
        created_at=NOW,
        delivered_at=NOW,
        acknowledged_at=NOW,
        acknowledged_by="task-1",
    )
    assert notice.acknowledged_at == NOW
    with pytest.raises(ValidationError):
        TransferAutoResearchLifecycleNotice.model_validate(
            {**notice.model_dump(), "source_kind": "work"}
        )
    with pytest.raises(ValidationError):
        TransferAutoResearchLifecycleNotice.model_validate(
            {**notice.model_dump(), "acknowledged_at": None}
        )


def test_bundle_resolves_human_attribution_and_preserves_paper_conflict_state() -> None:
    bundle = TransferRecordBundle(
        project_id=PROJECT_ID,
        attributions=(_attribution(),),
        tasks=(_task(),),
        watchers=(),
        episodes=(_experiment_episode(),),
        paper_draft=TransferPaperDraft(
            content="Current draft",
            base_hash="base-hash",
            ancestor_content="Ancestor draft",
            cursor_state="selection",
            updated_at=NOW,
        ),
    )
    assert bundle.paper_draft is not None
    assert bundle.paper_draft.ancestor_content == "Ancestor draft"

    with pytest.raises(ValidationError, match="unknown archive attribution"):
        TransferRecordBundle.model_validate(
            {
                **bundle.model_dump(),
                "tasks": (
                    {
                        **bundle.tasks[0].model_dump(),
                        "authorized_by_attribution_id": str(uuid.uuid4()),
                    },
                ),
            }
        )
    with pytest.raises(ValidationError, match="repeats one task identity"):
        TransferRecordBundle.model_validate(
            {**bundle.model_dump(), "tasks": (bundle.tasks[0], bundle.tasks[0])}
        )
    with pytest.raises(ValidationError, match="unknown parent task"):
        TransferRecordBundle.model_validate(
            {
                **bundle.model_dump(),
                "tasks": ({**bundle.tasks[0].model_dump(), "parent_operation_id": "missing-task"},),
            }
        )
    local_identity = TransferLocalId(
        archive_id=str(uuid.uuid4()),
        source_table="graph_run_events",
        source_id="1",
    )
    event = TransferTaskEvent(
        identity=local_identity,
        created_at=NOW,
        level="info",
        message="same source-local mapping",
    )
    with pytest.raises(ValidationError, match="archive-local record identity"):
        TransferRecordBundle(
            project_id=PROJECT_ID,
            attributions=(_attribution(),),
            tasks=(
                _task(events=(event,)),
                _task(operation_id="task-2", events=(event,)),
            ),
            watchers=(),
            episodes=(_experiment_episode(),),
        )


def test_record_models_do_not_expose_executable_source_bindings() -> None:
    forbidden = {
        "armed_revision",
        "check_command",
        "continuation",
        "cwd",
        "delivery_operation_id",
        "dispatch_authority",
        "execution_host",
        "execution_machine",
        "handoffs_cleared_at",
        "log_path",
        "native_session_id",
        "next_attempt_at",
        "next_check_at",
        "output_path",
        "runtime_id",
        "stage_host",
        "stage_root",
        "stop_requested_at",
        "watcher_snapshot_token",
        "write_scope_fingerprint",
    }
    model_fields = {
        field
        for name in transfer_records.__dict__
        for model in [getattr(transfer_records, name)]
        if isinstance(model, type) and issubclass(model, BaseModel) and name.startswith("Transfer")
        for field in model.model_fields
    }
    assert forbidden.isdisjoint(model_fields)
