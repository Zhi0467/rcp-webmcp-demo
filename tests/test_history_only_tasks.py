from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rcp.artifacts import descriptor_for
from rcp.service import CoachRequest, RunRequest
from rcp.storage import AgentTaskRecord, AppStore

from .helpers import create_named_app


def _task(
    store: AppStore,
    *,
    operation_id: str,
    project_id: str,
    kind: str,
    status: str,
    request: dict[str, object],
    native_session_id: str | None = None,
    stage_root: str | None = None,
    result: dict[str, object] | None = None,
) -> AgentTaskRecord:
    now = store.now()
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        kind=kind,
        status=status,
        request=request,
        created_at=now,
        updated_at=now,
        status_message=f"Stored {status} task.",
        native_session_id=native_session_id,
        stage_root=stage_root,
        result=result,
    )


def _insert_session_indexes(
    store: AppStore,
    *,
    project_id: str,
    chat_id: str,
    operation_id: str,
    native_session_id: str,
) -> None:
    now = store.now()
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO writing_sessions (
                native_session_id, provider, runtime_id, execution_machine,
                project_id, title, model, reasoning, created_at, last_resumed_at,
                introduction_hash_examined, graph_revision_examined,
                research_md_hash_examined
            ) VALUES (?, 'codex', '', 'laptop', ?, 'History', '', 'medium', ?, ?, '', 0, '')
            """,
            (native_session_id, project_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO chat_session_contexts (
                provider, execution_machine, native_session_id, project_id,
                kind, chat_id, node_id, protocol_version, snapshot_json,
                snapshot_sha256, committed_operation_id, created_at, updated_at
            ) VALUES ('codex', 'laptop', ?, ?, 'project_chat', ?, NULL, 1,
                      '{}', ?, ?, ?, ?)
            """,
            (native_session_id, project_id, chat_id, "a" * 64, operation_id, now, now),
        )


def test_history_only_fence_preserves_history_and_removes_every_continuation(
    manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    store = app.state.background_tasks.store
    service = app.state.service
    project_id = app.state.default_project_id
    chat_id = str(uuid.uuid4())
    chat_operation_id = str(uuid.uuid4())
    interrupted_operation_id = str(uuid.uuid4())
    coach_operation_id = str(uuid.uuid4())
    chat_session_id = str(uuid.uuid4())
    interrupted_session_id = str(uuid.uuid4())
    coach_session_id = str(uuid.uuid4())
    stage_root = tmp_path / "task-stage"
    artifact_directory = stage_root / "turns" / chat_operation_id / "artifacts"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "temporary.html").write_text("<p>temporary</p>", encoding="utf-8")
    kept_filename = service.history.workspace.keep_artifact(
        source_name="kept.html",
        project_name="History project",
        data=b"<p>kept</p>",
    )
    kept = descriptor_for(chat_operation_id, "kept.html").model_copy(
        update={"kept_filename": kept_filename, "kept_at": store.now()}
    )
    temporary = descriptor_for(chat_operation_id, "temporary.html")
    chat_request = RunRequest(
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        chat_scope="project",
        chat_id=chat_id,
        message="Inspect the result.",
        mode="work",
    )
    repairable_result = {
        "messages": ["Historical answer."],
        "artifacts": [kept.model_dump(mode="json"), temporary.model_dump(mode="json")],
        "graph_update": {
            "status": "rejected",
            "repairable": True,
            "validation_messages": ["Needs repair."],
        },
    }
    store.create_agent_task(
        _task(
            store,
            operation_id=chat_operation_id,
            project_id=project_id,
            kind="project_chat",
            status="succeeded",
            request=chat_request.model_dump(mode="json"),
            native_session_id=chat_session_id,
            stage_root=str(stage_root),
            result=repairable_result,
        )
    )
    store.record_agent_task_receipt(
        chat_operation_id,
        "operation_created",
        {
            "kind": "project_chat",
            "attempt": 1,
            "has_parent": False,
            "resumed": False,
        },
    )
    store.create_agent_task(
        _task(
            store,
            operation_id=interrupted_operation_id,
            project_id=project_id,
            kind="project_chat",
            status="interrupted",
            request=chat_request.model_copy(update={"message": "Resume me."}).model_dump(
                mode="json"
            ),
            native_session_id=interrupted_session_id,
            stage_root=str(tmp_path / "interrupted-stage"),
        )
    )
    store.create_agent_task(
        _task(
            store,
            operation_id=coach_operation_id,
            project_id=project_id,
            kind="paper_coach",
            status="failed",
            request=CoachRequest(
                message="Review the introduction.",
                provider="codex",
                model="",
                reasoning="medium",
                run_on="laptop",
            ).model_dump(mode="json"),
            native_session_id=coach_session_id,
            result={"messages": ["Historical paper feedback."]},
        )
    )
    _insert_session_indexes(
        store,
        project_id=project_id,
        chat_id=chat_id,
        operation_id=chat_operation_id,
        native_session_id=chat_session_id,
    )
    with store.connection() as connection:
        connection.execute(
            """
            INSERT INTO writing_sessions (
                native_session_id, provider, runtime_id, execution_machine,
                project_id, title, model, reasoning, created_at, last_resumed_at,
                introduction_hash_examined, graph_revision_examined,
                research_md_hash_examined
            ) VALUES (?, 'codex', '', 'laptop', ?, 'Paper coach', '', 'medium', ?, ?, '', 0, '')
            """,
            (coach_session_id, project_id, store.now(), store.now()),
        )
    now = store.now()
    transcript_path = manifest.research_dir / "chat" / f"project-{chat_id}.jsonl"
    transcript_path.parent.mkdir(exist_ok=True)
    common = {
        "sessionId": chat_id,
        "nativeSessionId": chat_session_id,
        "nodeId": None,
        "chatScope": "project",
        "provider": "codex",
        "model": "",
        "reasoning": "medium",
        "executionMachine": "laptop",
        "timestamp": now,
        "operationId": chat_operation_id,
        "mode": "work",
    }
    transcript_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {
                    **common,
                    "uuid": str(uuid.uuid4()),
                    "type": "user",
                    "role": "user",
                    "text": "Inspect the result.",
                },
                {
                    **common,
                    "uuid": str(uuid.uuid4()),
                    "type": "assistant",
                    "role": "assistant",
                    "text": "Historical answer.",
                },
            )
        ),
        encoding="utf-8",
    )
    original_transcript = transcript_path.read_bytes()
    assert (
        client.get(f"/api/projects/{project_id}/chats/{chat_id}").json()["messages"][0][
            "native_session_id"
        ]
        == chat_session_id
    )

    changed = store.mark_agent_tasks_history_only(
        [chat_operation_id, interrupted_operation_id, coach_operation_id]
    )

    assert changed == 3
    assert (
        store.mark_agent_tasks_history_only(
            [chat_operation_id, interrupted_operation_id, coach_operation_id]
        )
        == 0
    )
    assert transcript_path.read_bytes() == original_transcript
    with store.connection() as connection:
        raw = connection.execute(
            "SELECT history_only, native_session_id FROM graph_runs WHERE operation_id = ?",
            (chat_operation_id,),
        ).fetchone()
        assert raw is not None
        assert tuple(raw) == (1, chat_session_id)
        for table in ("writing_sessions", "chat_session_contexts"):
            assert (
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE native_session_id = ?",
                    (chat_session_id,),
                ).fetchone()[0]
                == 0
            )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM writing_sessions WHERE native_session_id = ?",
                (coach_session_id,),
            ).fetchone()[0]
            == 0
        )

    tasks = {
        task["operation_id"]: task
        for task in client.get(f"/api/projects/{project_id}/tasks").json()
    }
    for operation_id in (chat_operation_id, interrupted_operation_id, coach_operation_id):
        projected = tasks[operation_id]
        assert projected["history_only"] is True
        assert projected["native_session_id"] is None
        assert projected["can_pause"] is False
        assert projected["can_resume"] is False
        assert projected["can_retry"] is False
        assert projected["awaiting_human"] is False
    assert tasks[chat_operation_id]["result"]["messages"] == ["Historical answer."]
    assert tasks[chat_operation_id]["result"]["graph_update"]["repairable"] is False
    assert tasks[coach_operation_id]["status"] == "failed"
    assert tasks[coach_operation_id]["result"]["messages"] == ["Historical paper feedback."]
    assert store.agent_task_continuation_session_id(project_id, chat_operation_id) is None
    assert not store.has_chat_native_session_origin(
        project_id,
        "project_chat",
        chat_id,
        None,
        "codex",
        "laptop",
        chat_session_id,
    )
    with pytest.raises(ValueError, match="no repairable graph update"):
        store.claim_agent_task_graph_repair(chat_operation_id)
    transcript = client.get(f"/api/projects/{project_id}/chats/{chat_id}").json()
    assert [message["native_session_id"] for message in transcript["messages"]] == [None, None]

    control_paths = (
        f"/api/projects/{project_id}/tasks/{chat_operation_id}/pause",
        f"/api/projects/{project_id}/tasks/{interrupted_operation_id}/resume",
        f"/api/projects/{project_id}/tasks/{coach_operation_id}/retry",
        f"/api/projects/{project_id}/tasks/{chat_operation_id}/repair-graph-update",
    )
    task_count = len(store.agent_tasks(project_id, include_hidden=True))
    for path in control_paths:
        response = client.post(path)
        assert response.status_code == 409
        assert "retained as history" in response.json()["detail"]
    assert len(store.agent_tasks(project_id, include_hidden=True)) == task_count

    artifacts = {
        artifact["name"]: artifact for artifact in tasks[chat_operation_id]["result"]["artifacts"]
    }
    assert artifacts["kept.html"] == {
        **kept.model_dump(mode="json"),
        "available": True,
        "unavailable_reason": None,
        "can_open": True,
        "can_download": True,
        "can_keep": False,
        "can_revise": False,
    }
    assert artifacts["temporary.html"] == {
        **temporary.model_dump(mode="json"),
        "available": False,
        "unavailable_reason": "Artifact bytes were not retained with this task history.",
        "can_open": False,
        "can_download": False,
        "can_keep": False,
        "can_revise": False,
    }
    base = f"/api/projects/{project_id}/tasks/{chat_operation_id}/artifacts"
    kept_base = f"{base}/{kept.artifact_id}"
    assert client.get(f"{kept_base}/content").status_code == 200
    assert client.get(f"{kept_base}/download").status_code == 200
    viewer = client.get(f"{kept_base}/viewer")
    assert viewer.status_code == 200
    assert "Add to chat" not in viewer.text
    assert 'id="keep"' not in viewer.text
    assert 'id="box"' not in viewer.text
    assert client.post(f"{kept_base}/keep").status_code == 409

    monkeypatch.setattr(
        "rcp.api.tasks.read_local_regular_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an unavailable history-only stage must not be read")
        ),
    )
    temporary_base = f"{base}/{temporary.artifact_id}"
    assert client.get(f"{temporary_base}/content").status_code == 410
    assert client.get(f"{temporary_base}/download").status_code == 410
    assert client.get(f"{temporary_base}/viewer").status_code == 410
    assert client.post(f"{temporary_base}/keep").status_code == 409
    context_response = client.post(
        f"/api/projects/{project_id}/tasks/project_chat",
        json={
            "chat_id": chat_id,
            "message": "What does this selection mean?",
            "artifact_context": {
                "source": "task",
                "operation_id": chat_operation_id,
                "artifact_id": kept.artifact_id,
                "selections": [{"kind": "text", "text": "kept"}],
            },
        },
    )
    assert context_response.status_code == 422
    assert "native session is unavailable" in context_response.json()["detail"]
    assert len(store.agent_tasks(project_id, include_hidden=True)) == task_count


def test_history_only_transaction_refuses_nonterminal_or_partially_shared_sessions(
    tmp_path: Path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    queued = _task(
        store,
        operation_id="queued",
        project_id="project",
        kind="seed",
        status="queued",
        request={},
    )
    store.create_agent_task(queued)

    with pytest.raises(ValueError, match="Only terminal tasks"):
        store.mark_agent_tasks_history_only([queued.operation_id])
    assert store.agent_task(queued.operation_id).history_only is False

    shared_session = "shared-native-session"
    first = _task(
        store,
        operation_id="first",
        project_id="project",
        kind="seed",
        status="succeeded",
        request={},
        native_session_id=shared_session,
    )
    second = first.model_copy(update={"operation_id": "second"})
    store.create_agent_task(first)
    store.create_agent_task(second)

    with pytest.raises(ValueError, match="shares its native session"):
        store.mark_agent_tasks_history_only([first.operation_id])
    assert store.agent_task(first.operation_id).history_only is False
    assert store.mark_agent_tasks_history_only([first.operation_id, second.operation_id]) == 2
