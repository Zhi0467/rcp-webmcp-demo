from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from rcp.artifacts import descriptor_for
from rcp.providers import ProviderUsage
from rcp.runs.auto_research import AutoResearchRunRequest
from rcp.runs.tasks.episode_report import EpisodeReportRunRequest
from rcp.service import CoachRequest, RunRequest
from rcp.storage import AgentTaskRecord, AppStore
from rcp.transfer import (
    TransferArchiveActor,
    TransferArchiveAttribution,
)

from .helpers import authorized_human, create_named_app


def _store(
    manifest, tmp_path: Path
) -> tuple[AppStore, str, tuple[TransferArchiveAttribution, ...]]:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    store = app.state.background_tasks.store
    actor = authorized_human(store)
    attribution = TransferArchiveAttribution(
        archive_actor_id=str(uuid.uuid4()),
        source_actor=TransferArchiveActor.capture(actor),
    )
    return store, app.state.default_project_id, (attribution,)


def _finished_task(
    store: AppStore,
    project_id: str,
    *,
    actor,
    operation_id: str | None = None,
) -> AgentTaskRecord:
    now = store.now()
    operation_id = operation_id or str(uuid.uuid4())
    kept = descriptor_for(operation_id, "kept.html", size_bytes=18).model_copy(
        update={"kept_filename": "kept.html", "kept_at": now}
    )
    temporary = descriptor_for(operation_id, "temporary.html")
    request = CoachRequest(
        message="Review the current introduction.",
        provider="codex",
        model="gpt-5.6-sol",
        reasoning="high",
        run_on="source-gpu",
        session_id="native-paper-session",
    )
    task = AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        kind="paper_coach",
        status="succeeded",
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=now,
        status_message="Finished",
        result={
            "messages": ["Historical Paper-coach answer."],
            "artifacts": [kept.model_dump(mode="json"), temporary.model_dump(mode="json")],
            "graph_update": {
                "status": "rejected",
                "repairable": True,
                "validation_messages": ["Keep this inert."],
            },
        },
        native_session_id="native-paper-session",
        stage_host="source-gpu",
        stage_root="/source/stage",
        authorized_by=actor,
    )
    return store.create_agent_task(task)


def test_finished_project_export_is_inert_complete_and_repeatable(manifest, tmp_path: Path) -> None:
    store, project_id, attributions = _store(manifest, tmp_path)
    actor = authorized_human(store)
    task = _finished_task(store, project_id, actor=actor)
    store.record_agent_task_event(task.operation_id, "Provider finished.")
    store.record_agent_task_receipt(
        task.operation_id,
        "native_agent_checkpoint",
        {
            "provider": "codex",
            "run_on": "source-gpu",
            "native_session_id": "native-paper-session",
            "nested": {"stage_root": "/source/stage", "kept": True},
        },
        tier="diagnostic",
    )
    contract = "Exact historical prompt contract."
    store.record_agent_task_contract(
        task.operation_id,
        "prompt",
        contract,
        hashlib.sha256(contract.encode()).hexdigest(),
    )
    store.record_agent_task_patch_output(task.operation_id, '{"operations":[]}')
    store.record_agent_usage(
        task.operation_id,
        ProviderUsage(
            provider_profile="ordinary",
            provider_event_type="turn.completed",
            dedupe_key="usage-1",
            processed_input_tokens=12,
            generated_tokens=3,
            provider_fields={"session_id": "native-paper-session", "safe": "retained"},
        ),
    )
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO paper_drafts (
                project_id, content, base_hash, updated_at, cursor_state, ancestor_content
            ) VALUES (?, 'Current draft', 'base-hash', ?, 'cursor', 'Ancestor draft')
            """,
            (project_id, store.now()),
        )
        connection.execute(
            """
            INSERT INTO writing_sessions (
                native_session_id, provider, runtime_id, execution_machine,
                project_id, title, model, reasoning, created_at, last_resumed_at,
                introduction_hash_examined, graph_revision_examined,
                research_md_hash_examined
            ) VALUES ('excluded-paper-session', 'codex', '', 'source-gpu', ?, 'Paper',
                      '', 'high', ?, ?, '', 0, '')
            """,
            (project_id, store.now(), store.now()),
        )
        connection.execute(
            "UPDATE graph_runs SET authorized_display_name = 'Earlier display name' "
            "WHERE operation_id = ?",
            (task.operation_id,),
        )

    first = store.export_project_transfer_records(project_id, attributions=attributions)
    second = store.export_project_transfer_records(project_id, attributions=attributions)

    assert first.model_dump_json() == second.model_dump_json()
    assert len(first.tasks) == 1
    exported = first.tasks[0]
    assert exported.history_only is True
    assert exported.request.shape == "paper_coach"
    assert "run_on" not in type(exported.request).model_fields
    assert "session_id" not in type(exported.request).model_fields
    assert exported.assistant.answer is None
    assert exported.assistant.legacy_unlabelled_lines == ("Historical Paper-coach answer.",)
    graph_update = exported.graph_updates[0].value()
    assert isinstance(graph_update, dict)
    assert graph_update["status"] == "rejected"
    assert graph_update["validation_messages"] == ["Keep this inert."]
    assert "repairable" not in graph_update
    checkpoint = next(
        receipt for receipt in exported.receipts if receipt.category == "native_agent_checkpoint"
    )
    assert checkpoint.payload.value() == {
        "nested": {"kept": True},
        "provider": "codex",
    }
    assert exported.usage[0].provider_fields.value() == {"safe": "retained"}
    assert {item.source_name for item in exported.artifacts} == {
        "kept.html",
        "temporary.html",
    }
    temporary = next(item for item in exported.artifacts if item.source_name == "temporary.html")
    assert temporary.content_sha256 is None
    assert temporary.kept_filename is None
    assert exported.output is not None
    assert exported.output.patch.value() == {"operations": []}
    assert exported.contracts[0].content == contract
    assert first.paper_draft is not None
    assert first.paper_draft.ancestor_content == "Ancestor draft"

    serialized = first.model_dump_json()
    assert "native-paper-session" not in serialized
    assert "excluded-paper-session" not in serialized
    assert "/source/stage" not in serialized
    assert "source-gpu" not in serialized
    with store.connection() as connection:
        source = connection.execute(
            "SELECT history_only, native_session_id, stage_root FROM graph_runs "
            "WHERE operation_id = ?",
            (task.operation_id,),
        ).fetchone()
    assert source is not None
    assert source["history_only"] == 0
    assert source["native_session_id"] == "native-paper-session"
    assert source["stage_root"] == "/source/stage"


def test_export_refuses_unfinished_task_without_changing_it(manifest, tmp_path: Path) -> None:
    store, project_id, attributions = _store(manifest, tmp_path)
    actor = authorized_human(store)
    now = store.now()
    task = store.create_agent_task(
        AgentTaskRecord(
            operation_id=str(uuid.uuid4()),
            project_id=project_id,
            kind="paper_coach",
            status="queued",
            request=CoachRequest(
                message="Wait",
                provider="codex",
                run_on="source-gpu",
            ).model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Queued",
            authorized_by=actor,
        )
    )

    with pytest.raises(ValueError, match="agent task to be settled"):
        store.export_project_transfer_records(project_id, attributions=attributions)
    assert store.agent_task(task.operation_id).status == "queued"  # type: ignore[union-attr]


def test_export_refuses_completed_watcher_with_pending_delivery(manifest, tmp_path: Path) -> None:
    store, project_id, attributions = _store(manifest, tmp_path)
    actor = authorized_human(store)
    task = _finished_task(store, project_id, actor=actor)
    watcher_id = str(uuid.uuid4())
    now = store.now()
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO watchers (
                watcher_id, project_id, origin_operation_id, origin_task_kind,
                chat_id, graph_target_json, execution_host, check_command,
                log_path, cwd, continuation_json, status, created_at,
                completed_at, notified
            ) VALUES (?, ?, ?, 'project_chat', 'chat-1',
                      '{"kind":"main","branch_id":null}', 'source-gpu', 'true',
                      '/source/log', '/source/cwd', '{}', 'completed', ?, ?, 0)
            """,
            (watcher_id, project_id, task.operation_id, now, now),
        )

    with pytest.raises(ValueError, match="watcher to be settled"):
        store.export_project_transfer_records(project_id, attributions=attributions)

    with store.connection() as connection:
        connection.execute("UPDATE watchers SET notified = 1 WHERE watcher_id = ?", (watcher_id,))
    bundle = store.export_project_transfer_records(project_id, attributions=attributions)
    assert bundle.watchers[0].status == "completed"
    assert "check_command" not in type(bundle.watchers[0]).model_fields
    assert "continuation" not in type(bundle.watchers[0]).model_fields


def test_finished_auto_research_corpus_exports_all_terminal_record_groups(
    manifest,
    tmp_path: Path,
) -> None:
    store, project_id, attributions = _store(manifest, tmp_path)
    actor = authorized_human(store)
    episode_id = str(uuid.uuid4())
    child_episode_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    worker_id = operation_id
    now = store.now()
    graph_target = json.dumps({"kind": "branch", "branch_id": episode_id}, separators=(",", ":"))
    base_head = json.dumps(
        {
            "target": {"kind": "main", "branch_id": None},
            "revision": 0,
            "transition_id": None,
        },
        separators=(",", ":"),
    )
    request = AutoResearchRunRequest(
        episode_id=episode_id,
        role="orchestrator",
        provider="codex",
        model="gpt-5.6-sol",
        reasoning="high",
        run_on="source-gpu",
        session_id="native-auto-session",
    ).model_dump(mode="json")
    instruction = "Investigate the blocker."
    command_content = '{"operations":[]}'
    finish_result = json.dumps(
        {"ending": "completed", "episode_id": episode_id, "status": "completed"},
        sort_keys=True,
        separators=(",", ":"),
    )
    apply_result = json.dumps(
        {
            "delivery_operation_id": operation_id,
            "notification_operation_id": operation_id,
            "status": "applied",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    parent_receipt = '{"ending":"completed"}'
    child_receipt = '{"ending":"stopped"}'
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO graph_runs (
                operation_id, project_id, episode_id, kind, status, request_json,
                created_at, updated_at, started_at, finished_at, status_message,
                runtime_id, native_session_id, stage_host, stage_root,
                graph_target_json, authorized_space_id, authorized_user_id,
                authorized_display_name
            ) VALUES (?, ?, ?, 'auto_research', 'succeeded', ?, ?, ?, ?, ?, 'Finished',
                      'codex.exec-json.v1', 'native-auto-session', 'source-gpu', '/source/stage',
                      ?, ?, ?, ?)
            """,
            (
                operation_id,
                project_id,
                episode_id,
                json.dumps(request, separators=(",", ":")),
                now,
                now,
                now,
                now,
                graph_target,
                actor.space_id,
                actor.user_id,
                actor.display_name,
            ),
        )
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, project_id, mode, graph_target_json, graph_base_head_json,
                root_operation_id, status, invocation_ceiling, invocations_used,
                authorized_space_id, authorized_user_id, authorized_display_name,
                ending, wrapup_state, report_attempts_used, created_at, updated_at, ended_at
            ) VALUES (?, ?, 'auto_research', ?, ?, ?, 'completed', 2, 1, ?, ?, ?,
                      'completed', 'legacy_unavailable', 0, ?, ?, ?)
            """,
            (
                episode_id,
                project_id,
                graph_target,
                base_head,
                operation_id,
                actor.space_id,
                actor.user_id,
                actor.display_name,
                now,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, project_id, mode, control_node_id, graph_target_json,
                status, invocation_ceiling, invocations_used, ending, wrapup_state,
                report_attempts_used, created_at, updated_at, ended_at
            ) VALUES (?, ?, 'experiment_loop', 'node-1',
                      '{"kind":"main","branch_id":null}', 'stopped', 1, 0,
                      'stopped', 'skipped', 0, ?, ?, ?)
            """,
            (child_episode_id, project_id, now, now, now),
        )
        connection.execute(
            """
            INSERT INTO experiment_episode_state (
                episode_id, execution_host, last_watcher_ids_json,
                context_baseline_json, created_at, updated_at
            ) VALUES (?, '', '[]', '{}', ?, ?)
            """,
            (child_episode_id, now, now),
        )
        for wrapup_episode_id, ending, state, receipt_text in (
            (episode_id, "completed", "legacy_unavailable", parent_receipt),
            (child_episode_id, "stopped", "skipped", child_receipt),
        ):
            connection.execute(
                """
                INSERT INTO episode_wrapups (
                    episode_id, ending, partial, receipt_json, receipt_sha256,
                    state, created_at, updated_at, finished_at
                ) VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wrapup_episode_id,
                    ending,
                    receipt_text,
                    hashlib.sha256(receipt_text.encode()).hexdigest(),
                    state,
                    now,
                    now,
                    now,
                ),
            )
        connection.execute(
            "INSERT INTO episode_invocations VALUES (?, ?, 1, ?)",
            (episode_id, operation_id, now),
        )
        connection.execute(
            "INSERT INTO auto_research_episodes VALUES (?, 'Start here.', ?, ?)",
            (episode_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO auto_research_invocations (
                episode_id, operation_id, allocation_operation_id, role,
                actor_operation_id, control_node_id, created_at
            ) VALUES (?, ?, ?, 'orchestrator', ?, NULL, ?)
            """,
            (episode_id, operation_id, operation_id, operation_id, now),
        )
        connection.execute(
            """
            INSERT INTO auto_research_messages (
                message_id, episode_id, sender_role, sender_task_id,
                recipient_task_id, body, created_at, delivered_at, delivery_operation_id
            ) VALUES (?, ?, 'orchestrator', ?, ?, 'Done.', ?, ?, ?)
            """,
            (str(uuid.uuid4()), episode_id, operation_id, operation_id, now, now, operation_id),
        )
        connection.execute(
            """
            INSERT INTO auto_research_recoveries (
                recovery_id, episode_id, operation_id, failure_kind, retry_mode,
                attempts, max_attempts, status, diagnostic, admitted_operation_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'provider', 'exact', 1, 2, 'admitted',
                      'Recovered.', ?, ?, ?)
            """,
            (str(uuid.uuid4()), episode_id, operation_id, operation_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO auto_research_child_work (
                worker_id, episode_id, project_id, control_node_id,
                root_operation_id, current_operation_id, admitted_by_operation_id,
                instruction, instruction_sha256, created_at, updated_at
            ) VALUES (?, ?, ?, 'node-1', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                worker_id,
                episode_id,
                project_id,
                operation_id,
                operation_id,
                operation_id,
                instruction,
                hashlib.sha256(instruction.encode()).hexdigest(),
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO auto_research_child_work_attempts VALUES (?, ?, ?, ?)",
            (operation_id, worker_id, operation_id, now),
        )
        connection.execute(
            """
            INSERT INTO auto_research_child_experiments (
                child_episode_id, auto_research_episode_id, project_id,
                control_node_id, state, request_json, goal_sha256,
                parent_operation_id, terminal_diagnostic, created_at, updated_at
            ) VALUES (?, ?, ?, 'node-1', 'terminal', ?, ?, ?, 'Complete.', ?, ?)
            """,
            (
                child_episode_id,
                episode_id,
                project_id,
                json.dumps({"goal": "Test", "invocation_limit": 1}, separators=(",", ":")),
                hashlib.sha256(b"Test").hexdigest(),
                operation_id,
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO auto_research_experiment_invocations VALUES (?, ?, ?, ?)",
            (operation_id, episode_id, child_episode_id, now),
        )
        for kind, child_id in (("work", worker_id), ("experiment", child_episode_id)):
            connection.execute(
                """
                INSERT INTO auto_research_child_admissions (
                    admission_id, episode_id, project_id, child_kind, child_id,
                    state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'reflected', ?, ?)
                """,
                (str(uuid.uuid4()), episode_id, project_id, kind, child_id, now, now),
            )
        connection.execute(
            """
            INSERT INTO auto_research_lifecycle_notices (
                notice_id, episode_id, source_kind, source_id, source_event,
                source_attempt, state, payload_json, created_at, delivered_at,
                delivery_operation_id, acknowledged_at, acknowledged_by
            ) VALUES (?, ?, 'worker', ?, 'finished', 1, 'acknowledged', '{}',
                      ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                episode_id,
                worker_id,
                now,
                now,
                operation_id,
                now,
                operation_id,
            ),
        )
        connection.execute(
            """
            INSERT INTO auto_research_inbox_receipts (
                effect_id, episode_id, mode, result_json, acknowledged_by, created_at
            ) VALUES (?, ?, 'clear', '{"count":0,"notice_ids":[]}', ?, ?)
            """,
            (str(uuid.uuid4()), episode_id, operation_id, now),
        )
        connection.execute(
            """
            INSERT INTO auto_research_finish_receipts (
                effect_id, episode_id, actor_operation_id, disposition,
                blocker_count, result_json, result_sha256, created_at
            ) VALUES (?, ?, ?, 'completed', 0, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                episode_id,
                operation_id,
                finish_result,
                hashlib.sha256(finish_result.encode()).hexdigest(),
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO auto_research_apply_results (
                apply_id, episode_id, operation_id, patch_sha256, result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), episode_id, operation_id, "a" * 64, apply_result, now),
        )
        connection.execute(
            """
            INSERT INTO auto_research_command_files (
                command_id, episode_id, operation_id, kind, filename,
                sha256, content, created_at
            ) VALUES (?, ?, ?, 'apply', 'patch.json', ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                episode_id,
                operation_id,
                hashlib.sha256(command_content.encode()).hexdigest(),
                command_content,
                now,
            ),
        )

    bundle = store.export_project_transfer_records(project_id, attributions=attributions)
    auto = next(item for item in bundle.episodes if item.episode_id == episode_id)
    assert auto.auto_research is not None
    history = auto.auto_research
    assert len(history.invocations) == 1
    assert len(history.messages) == 1
    assert history.recoveries[0].status == "admitted"
    assert len(history.child_work) == 1
    assert len(history.child_experiments) == 1
    assert len(history.child_admissions) == 2
    assert len(history.lifecycle_notices) == 1
    assert len(history.inbox_receipts) == 1
    assert len(history.finish_receipts) == 1
    assert len(history.apply_results) == 1
    assert history.apply_results[0].result.value() == {"status": "applied"}
    assert len(history.commands) == 1
    assert "native-auto-session" not in bundle.model_dump_json()

    store.record_agent_usage(
        operation_id,
        ProviderUsage(
            provider_profile="ordinary",
            provider_event_type="turn.completed",
            dedupe_key="auto-usage",
            processed_input_tokens=4,
            generated_tokens=2,
        ),
    )
    other_project_id = str(uuid.uuid4())
    ownership_rows = (
        ("agent_usage", "usage_id", store.agent_usage(project_id)[0].usage_id),
        ("auto_research_child_work", "worker_id", worker_id),
        ("auto_research_child_experiments", "child_episode_id", child_episode_id),
        ("auto_research_child_admissions", "child_id", worker_id),
    )
    for table, identity_column, identity in ownership_rows:
        with store.connection() as connection:
            connection.execute(
                f"UPDATE {table} SET project_id = ? WHERE {identity_column} = ?",
                (other_project_id, identity),
            )
        with pytest.raises(ValueError, match="conflicting projects"):
            store.export_project_transfer_records(project_id, attributions=attributions)
        with store.connection() as connection:
            connection.execute(
                f"UPDATE {table} SET project_id = ? WHERE {identity_column} = ?",
                (project_id, identity),
            )

    with store.connection() as connection:
        connection.execute(
            "UPDATE auto_research_recoveries SET status = 'pending', "
            "admitted_operation_id = NULL WHERE episode_id = ?",
            (episode_id,),
        )
    with pytest.raises(ValueError, match="Auto-research recovery to be settled"):
        store.export_project_transfer_records(project_id, attributions=attributions)


def test_finished_experiment_exports_sanitized_state_wrapup_and_report(
    manifest,
    tmp_path: Path,
) -> None:
    store, project_id, attributions = _store(manifest, tmp_path)
    episode_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    report_task_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())
    now = store.now()
    turn_request = RunRequest(
        provider="codex",
        model="gpt-5.6-sol",
        reasoning="high",
        run_on="source-gpu",
        chat_scope="node",
        node_id="node-1",
        chat_id="chat-1",
        session_id="native-experiment-session",
        mode="work",
        patch_kind="experiment_loop",
        control_node_id="node-1",
        control_episode_id=episode_id,
        control_invocation=1,
        control_invocation_ceiling=1,
    ).model_dump(mode="json")
    report_request = EpisodeReportRunRequest(
        episode_id=episode_id,
        provider="codex",
        model="gpt-5.6-sol",
        reasoning="high",
        run_on="source-gpu",
        execution_host="source.example",
        session_id="native-experiment-session",
    ).model_dump(mode="json")
    receipt = json.dumps(
        {
            "accepted_handoff": {
                "delivery_operation_id": report_task_id,
                "watcher_ids": [str(uuid.uuid4())],
            },
            "ending": "completed",
            "native_session_id": "native-experiment-session",
            "output_path": "/source/stage/report.html",
            "summary": "Complete.",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    report_html = "<h1>Completed</h1>"
    with store.connection() as connection:
        for operation_id, kind, request, visible in (
            (turn_id, "node_chat", turn_request, 1),
            (report_task_id, "episode_report", report_request, 0),
        ):
            connection.execute(
                """
                INSERT INTO graph_runs (
                    operation_id, project_id, episode_id, kind, status, request_json,
                    created_at, updated_at, started_at, finished_at, status_message,
                    runtime_id, native_session_id, stage_host, stage_root,
                    graph_target_json, visible
                ) VALUES (?, ?, ?, ?, 'succeeded', ?, ?, ?, ?, ?, 'Finished',
                          'codex.exec-json.v1', 'native-experiment-session', 'source-gpu',
                          '/source/stage', '{"kind":"main","branch_id":null}', ?)
                """,
                (
                    operation_id,
                    project_id,
                    episode_id,
                    kind,
                    json.dumps(request, separators=(",", ":")),
                    now,
                    now,
                    now,
                    now,
                    visible,
                ),
            )
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, project_id, mode, control_node_id, graph_target_json,
                root_operation_id, status, invocation_ceiling, invocations_used,
                ending, wrapup_state, report_attempts_used, created_at, updated_at, ended_at
            ) VALUES (?, ?, 'experiment_loop', 'node-1',
                      '{"kind":"main","branch_id":null}', ?, 'completed', 1, 1,
                      'completed', 'ready', 1, ?, ?, ?)
            """,
            (episode_id, project_id, turn_id, now, now, now),
        )
        connection.execute(
            "INSERT INTO episode_invocations VALUES (?, ?, 1, ?)",
            (episode_id, turn_id, now),
        )
        connection.execute(
            """
            INSERT INTO experiment_episode_state (
                episode_id, provider, execution_machine, execution_host,
                native_session_id, stage_host, stage_root, chat_id,
                last_turn_operation_id, last_turn_invocation, last_graph_result,
                last_watcher_ids_json, context_baseline_json, session_diagnostic,
                created_at, updated_at
            ) VALUES (?, 'codex', 'source-gpu', 'source.example',
                      'native-experiment-session', 'source-gpu', '/source/stage',
                      'chat-1', ?, 1, 'accepted', '[]',
                      '{"session_id":"native-experiment-session"}', 'Historical only.', ?, ?)
            """,
            (episode_id, turn_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO episode_report_attempts (
                attempt_id, episode_id, attempt_number, allocation_operation_id,
                status, created_at, updated_at, finished_at
            ) VALUES (?, ?, 1, ?, 'succeeded', ?, ?, ?)
            """,
            (attempt_id, episode_id, report_task_id, now, now, now),
        )
        connection.execute(
            """
            INSERT INTO episode_wrapups (
                episode_id, ending, partial, concluding_operation_id,
                allocation_operation_id, provider, run_on, execution_host,
                native_session_id, stage_host, stage_root, output_name, output_path,
                receipt_json, receipt_sha256, state, created_at, updated_at, finished_at
            ) VALUES (?, 'completed', 0, ?, ?, 'codex', 'source-gpu', 'source.example',
                      'native-experiment-session', 'source-gpu', '/source/stage',
                      'report.html', '/source/stage/report.html', ?, ?, 'ready', ?, ?, ?)
            """,
            (
                episode_id,
                turn_id,
                report_task_id,
                receipt,
                hashlib.sha256(receipt.encode()).hexdigest(),
                now,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO episode_reports (
                report_id, episode_id, attempt_id, allocation_operation_id,
                ending, sha256, html, created_at
            ) VALUES (?, ?, ?, ?, 'completed', ?, ?, ?)
            """,
            (
                report_id,
                episode_id,
                attempt_id,
                report_task_id,
                hashlib.sha256(report_html.encode()).hexdigest(),
                report_html,
                now,
            ),
        )

    bundle = store.export_project_transfer_records(project_id, attributions=attributions)
    episode = next(item for item in bundle.episodes if item.episode_id == episode_id)
    assert episode.experiment is not None
    assert episode.experiment.provider == "codex"
    assert episode.experiment.chat_id == "chat-1"
    assert episode.wrapup is not None
    assert episode.wrapup.receipt.value() == {
        "ending": "completed",
        "summary": "Complete.",
    }
    assert episode.report is not None
    assert episode.report.html == report_html
    report_task = next(item for item in bundle.tasks if item.operation_id == report_task_id)
    assert report_task.visible is False
    assert report_task.request.shape == "episode_report"
    serialized = bundle.model_dump_json()
    assert "native-experiment-session" not in serialized
    assert "/source/stage" not in serialized
    assert "source.example" not in serialized

    corruptions = (
        (
            "UPDATE episode_reports SET ending = 'failed' WHERE episode_id = ?",
            "UPDATE episode_reports SET ending = 'completed' WHERE episode_id = ?",
        ),
        (
            "UPDATE episode_report_attempts SET status = 'failed' WHERE episode_id = ?",
            "UPDATE episode_report_attempts SET status = 'succeeded' WHERE episode_id = ?",
        ),
        (
            "UPDATE episode_wrapups SET allocation_operation_id = "
            f"'{turn_id}' WHERE episode_id = ?",
            "UPDATE episode_wrapups SET allocation_operation_id = "
            f"'{report_task_id}' WHERE episode_id = ?",
        ),
        (
            "UPDATE episodes SET wrapup_state = 'failed' WHERE episode_id = ?",
            "UPDATE episodes SET wrapup_state = 'ready' WHERE episode_id = ?",
        ),
    )
    for corrupt, restore in corruptions:
        with store.connection() as connection:
            connection.execute(corrupt, (episode_id,))
        with pytest.raises(ValueError, match="report|wrap-up"):
            store.export_project_transfer_records(project_id, attributions=attributions)
        with store.connection() as connection:
            connection.execute(restore, (episode_id,))

    with store.connection() as connection:
        connection.execute(
            "DELETE FROM experiment_episode_state WHERE episode_id = ?",
            (episode_id,),
        )
    with pytest.raises(ValueError, match="Experiment episode state is missing"):
        store.export_project_transfer_records(project_id, attributions=attributions)
