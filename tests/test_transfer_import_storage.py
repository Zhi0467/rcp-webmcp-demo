from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from rcp.transfer.project_files import parse_transfer_project_file_payload
from rcp.transfer.records import (
    TransferEpisodeInvocation,
    TransferEpisodeRecord,
    TransferEpisodeWrapup,
    TransferExperimentEpisodeHistory,
    TransferJsonDocument,
    TransferLocalId,
    TransferTaskContract,
    TransferTaskEvent,
    TransferTaskOutput,
    TransferTaskReceipt,
    TransferTaskUsage,
    TransferWatcherRecord,
)

from .test_transfer_import import _archive_fixture


def _json(value: object) -> TransferJsonDocument:
    return TransferJsonDocument.capture(value)  # type: ignore[arg-type]


def _rich_capture(fixture: dict[str, object]):
    archive_root = fixture["archive_root"]
    assert isinstance(archive_root, Path)
    operational = parse_transfer_project_file_payload(
        (archive_root / "records/project.jsonl").read_bytes()
    )
    task = operational.records.tasks[0]
    now = task.updated_at
    event = TransferTaskEvent(
        identity=TransferLocalId(
            archive_id=str(uuid.uuid4()),
            source_table="graph_run_events",
            source_id="1",
        ),
        created_at=now,
        level="info",
        message="historical event",
        payload=_json({"display": "history"}),
    )
    receipt = TransferTaskReceipt(
        identity=TransferLocalId(
            archive_id=str(uuid.uuid4()),
            source_table="graph_run_receipts",
            source_id="2",
        ),
        created_at=now,
        tier="summary",
        category="history",
        payload=_json({"display": "receipt"}),
    )
    usage = TransferTaskUsage(
        usage_id="usage-rich",
        provider="codex",
        model="gpt-test",
        provider_profile="default",
        provider_event_type="turn",
        counted=True,
        count_reason="counted",
        processed_input_tokens=1,
        generated_tokens=2,
        cached_input_tokens=0,
        cache_creation_input_tokens=0,
        cache_write_input_tokens=0,
        reasoning_output_tokens=0,
        provider_fields=_json({"tokens": 3}),
        created_at=now,
    )
    contract_content = "historical contract"
    contract = TransferTaskContract(
        role="system",
        content=contract_content,
        sha256=hashlib.sha256(contract_content.encode()).hexdigest(),
        created_at=now,
    )
    output = TransferTaskOutput(created_at=now, patch=_json({"status": "none"}))
    task = task.model_copy(
        update={
            "events": (event,),
            "receipts": (receipt,),
            "usage": (usage,),
            "contracts": (contract,),
            "output": output,
            "graph_updates": (_json({"status": "none"}),),
        }
    )
    episode_id = str(uuid.uuid4())
    watcher_id = str(uuid.uuid4())
    episode = TransferEpisodeRecord(
        episode_id=episode_id,
        mode="experiment_loop",
        control_node_id="experiment-node",
        status="stopped",
        invocation_ceiling=1,
        invocations_used=1,
        authorized_by_attribution_id=operational.records.attributions[0].archive_actor_id,
        ending="stopped",
        wrapup_state="skipped",
        report_attempts_used=0,
        created_at=now,
        updated_at=now,
        ended_at=now,
        invocations=(
            TransferEpisodeInvocation(
                operation_id=task.operation_id,
                invocation_number=1,
                created_at=now,
            ),
        ),
        wrapup=TransferEpisodeWrapup(
            ending="stopped",
            partial=False,
            receipt=_json({"ending": "stopped"}),
            state="skipped",
            created_at=now,
            updated_at=now,
            finished_at=now,
        ),
        experiment=TransferExperimentEpisodeHistory(
            provider="codex",
            chat_id="history-chat",
            last_turn_operation_id=task.operation_id,
            last_turn_invocation=1,
            last_watcher_ids=(watcher_id,),
            session_diagnostic="history only",
        ),
    )
    watcher = TransferWatcherRecord(
        watcher_id=watcher_id,
        kind="external",
        origin_operation_id=task.operation_id,
        origin_task_kind="project_chat",
        chat_id="history-chat",
        episode_id=episode_id,
        status="completed",
        created_at=now,
        completed_at=now,
        consecutive_error_count=0,
    )
    records = operational.records.model_copy(
        update={"tasks": (task,), "watchers": (watcher,), "episodes": (episode,)}
    )
    return operational.model_copy(update={"records": records})


def test_storage_import_inserts_full_inert_history_and_receipt(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _archive_fixture(manifest, tmp_path, monkeypatch)
    capture = _rich_capture(fixture)
    archive = fixture["archive"]
    target = fixture["target"]
    configuration = fixture["configuration"]
    assert capture.kept_result_views
    html = {
        view.kept_filename: (
            fixture["archive_root"] / "result-views" / view.kept_filename
        ).read_text(encoding="utf-8")
        for view in capture.kept_result_views
    }
    operational_entry = next(
        item for item in archive.entries if item.archive_path == "records/project.jsonl"
    )
    receipt = target.begin_project_transfer_import(
        archive.target_request_id,
        archive_manifest_sha256=archive.sha256(),
        target_manifest_sha256=configuration.receipt.target_manifest_sha256,
        operational_payload_sha256=operational_entry.sha256,
        target_configuration_receipt=configuration.receipt.model_dump(mode="json"),
        capture=capture,
        kept_result_view_html=html,
    )
    assert receipt.status == "database_imported"
    assert len(receipt.event_id_map) == 1
    assert len(receipt.receipt_id_map) == 1
    assert target.project(capture.project_id) is None
    stored_configuration = target.project_transfer_import_configuration_receipt_json(
        archive.target_request_id
    )
    assert stored_configuration is not None
    assert json.loads(stored_configuration) == configuration.receipt.model_dump(mode="json")
    with target.connection() as connection:
        task_row = connection.execute(
            "SELECT * FROM graph_runs WHERE operation_id = ?",
            (capture.records.tasks[0].operation_id,),
        ).fetchone()
        assert task_row["history_only"] == 1
        assert task_row["native_session_id"] is None
        assert task_row["stage_host"] is None
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM graph_run_events WHERE operation_id = ?",
                (capture.records.tasks[0].operation_id,),
            ).fetchone()[0]
            == 1
        )
        watcher_row = connection.execute(
            "SELECT * FROM watchers WHERE watcher_id = ?",
            (capture.records.watchers[0].watcher_id,),
        ).fetchone()
        assert watcher_row["next_check_at"] is None
        assert watcher_row["execution_host"] == ""
        assert json.loads(watcher_row["continuation_json"])["provider"] == "history-only"
        episode_row = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?",
            (capture.records.episodes[0].episode_id,),
        ).fetchone()
        assert episode_row["stop_requested_at"] is None
        experiment_row = connection.execute(
            "SELECT * FROM experiment_episode_state WHERE episode_id = ?",
            (capture.records.episodes[0].episode_id,),
        ).fetchone()
        assert experiment_row["native_session_id"] is None
        assert experiment_row["execution_host"] == "history-only"
        wrapup_row = connection.execute(
            "SELECT * FROM episode_wrapups WHERE episode_id = ?",
            (capture.records.episodes[0].episode_id,),
        ).fetchone()
        assert wrapup_row["native_session_id"] is None
        assert wrapup_row["execution_host"] == "history-only"
        view_row = connection.execute(
            "SELECT * FROM result_views WHERE view_id = ?",
            (capture.kept_result_views[0].view_id,),
        ).fetchone()
        assert view_row["html"] == next(iter(html.values()))
    imported_view = target.result_view_for_diagnostics(capture.kept_result_views[0].view_id)
    assert imported_view is not None
    descriptor = target.result_view_descriptor(imported_view)
    assert descriptor.state == "kept"
    assert descriptor.can_revise is False

    repeated = target.begin_project_transfer_import(
        archive.target_request_id,
        archive_manifest_sha256=archive.sha256(),
        target_manifest_sha256=configuration.receipt.target_manifest_sha256,
        operational_payload_sha256=operational_entry.sha256,
        target_configuration_receipt=configuration.receipt.model_dump(mode="json"),
        capture=capture,
        kept_result_view_html=html,
    )
    assert repeated == receipt
    complete = target.complete_project_transfer_import(
        archive.target_request_id,
        publication_sha256="b" * 64,
    )
    assert complete.status == "complete"
    assert complete.publication_sha256 == "b" * 64
