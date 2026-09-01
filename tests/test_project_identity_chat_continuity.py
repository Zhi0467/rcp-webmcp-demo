from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from rcp.agents import AgentProcessControl
from rcp.background import AgentTaskExecution
from rcp.runs.chat import (
    _chat_stage_name,
    _validated_local_chat_resume_stage,
    _validated_remote_chat_resume_stage,
)
from rcp.service import RunRequest
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    ProjectRecord,
    WatcherContinuation,
    WatcherRecord,
)


def _register_legacy_project(store: AppStore, project_id: str) -> None:
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name="Legacy project",
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at="2026-08-01T00:00:00+00:00",
        )
    )


def _request(chat_id: str, *, trigger: str = "human", watcher_ids: list[str] | None = None):
    return {
        "provider": "codex",
        "model": "gpt-5",
        "reasoning": "medium",
        "run_on": "laptop",
        "run_truth_scope": ["state"],
        "chat_scope": "project",
        "node_id": None,
        "message": "Continue.",
        "chat_id": chat_id,
        "session_id": None,
        "mode": "work",
        "trigger": trigger,
        "patch_kind": "work",
        "workflow_ids": [],
        "skill_ids": [],
        "invoked_workflow_ids": [],
        "invoked_skill_ids": [],
        "resolved_skill_packages": [],
        "watcher_ids": watcher_ids or [],
    }


def _task(
    store: AppStore,
    operation_id: str,
    project_id: str,
    chat_id: str,
    *,
    status: str = "succeeded",
    stage_host: str | None = None,
    stage_root: str | None = None,
    parent_operation_id: str | None = None,
    native_session_id: str | None = None,
    request_session_id: str | None = None,
) -> AgentTaskRecord:
    now = store.now()
    request = _request(chat_id)
    request["session_id"] = request_session_id
    return AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        kind="project_chat",
        status=status,
        request=request,
        created_at=now,
        updated_at=now,
        status_message="Stored chat turn.",
        attempt=2 if parent_operation_id else 1,
        parent_operation_id=parent_operation_id,
        native_session_id=native_session_id,
        stage_host=stage_host,
        stage_root=stage_root,
    )


@pytest.mark.parametrize("remote", [False, True])
def test_adopted_chat_next_turn_reuses_exact_saved_stage(tmp_path: Path, remote: bool) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    legacy_id = "legacy-project"
    canonical_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    stage_name = f"chat-pre-adoption-{chat_id}"
    native_session_id = str(uuid.uuid4())
    if remote:
        stage_host = "worker.example"
        stage_root = f"/tmp/rcp-run.{stage_name}"
    else:
        stage_host = None
        local_stage = tmp_path / "data" / "run-stage" / stage_name
        local_stage.mkdir(parents=True)
        stage_root = str(local_stage)

    _register_legacy_project(store, legacy_id)
    store.create_agent_task(
        _task(
            store,
            "turn-before-adoption",
            legacy_id,
            chat_id,
            stage_host=stage_host,
            stage_root=stage_root,
            native_session_id=native_session_id,
        )
    )
    store.migrate_project_identity(legacy_id, canonical_id, store.space_id)

    next_turn = store.create_agent_task(
        _task(
            store,
            "turn-after-adoption",
            canonical_id,
            chat_id,
            status="queued",
            request_session_id=native_session_id,
        )
    )

    assert next_turn.stage_host == stage_host
    assert next_turn.stage_root == stage_root
    execution = AgentTaskExecution(
        operation_id=next_turn.operation_id,
        store=store,
        control=AgentProcessControl(),
        stage_host=next_turn.stage_host,
        stage_root=next_turn.stage_root,
    )
    request = RunRequest(chat_scope="project", chat_id=chat_id, message="Continue.")
    assert _chat_stage_name(None, request, execution) == stage_name  # type: ignore[arg-type]
    if remote:
        assert (
            _validated_remote_chat_resume_stage(execution, "worker.example", stage_name)
            == stage_root
        )
    else:
        assert _validated_local_chat_resume_stage(execution, Path(stage_root)) == Path(stage_root)


def test_adopted_chat_resume_and_retry_keep_pre_adoption_stage(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    legacy_id = "legacy-project"
    canonical_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    stage = tmp_path / "data" / "run-stage" / f"chat-pre-adoption-{chat_id}"
    stage.mkdir(parents=True)
    _register_legacy_project(store, legacy_id)
    store.create_agent_task(
        _task(
            store,
            "paused-before-adoption",
            legacy_id,
            chat_id,
            status="paused",
            stage_root=str(stage),
        )
    )
    store.migrate_project_identity(legacy_id, canonical_id, store.space_id)

    resumed = store.create_agent_task(
        _task(
            store,
            "resume-after-adoption",
            canonical_id,
            chat_id,
            status="failed",
            stage_root=str(stage),
            parent_operation_id="paused-before-adoption",
        )
    )
    retried = store.create_agent_task(
        _task(
            store,
            "retry-after-adoption",
            canonical_id,
            chat_id,
            status="queued",
            stage_root=str(stage),
            parent_operation_id="resume-after-adoption",
        )
    )

    assert resumed.stage_root == str(stage)
    assert retried.stage_root == str(stage)


def test_adopted_generic_watcher_wake_inherits_conversation_stage(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    legacy_id = "legacy-project"
    canonical_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    stage = tmp_path / "data" / "run-stage" / f"chat-pre-adoption-{chat_id}"
    stage.mkdir(parents=True)
    _register_legacy_project(store, legacy_id)
    store.create_agent_task(
        _task(
            store,
            "work-before-adoption",
            legacy_id,
            chat_id,
            stage_root=str(stage),
        )
    )
    continuation = WatcherContinuation(
        provider="codex",
        model="gpt-5",
        reasoning="medium",
        run_on="laptop",
        run_truth_scope=["state"],
    )
    store.create_watchers(
        [
            WatcherRecord(
                watcher_id="finished-work",
                project_id=legacy_id,
                origin_operation_id="work-before-adoption",
                origin_task_kind="project_chat",
                chat_id=chat_id,
                check_command="true",
                log_path="/tmp/finished-work.log",
                cwd="/tmp",
                continuation=continuation,
                status="completed",
                created_at="2026-08-01T00:00:00+00:00",
                completed_at="2026-08-01T00:01:00+00:00",
            )
        ]
    )
    store.migrate_project_identity(legacy_id, canonical_id, store.space_id)
    now = store.now()
    wake = AgentTaskRecord(
        operation_id="watcher-wake-after-adoption",
        project_id=canonical_id,
        kind="project_chat",
        status="queued",
        request=_request(chat_id, trigger="watcher", watcher_ids=["finished-work"]),
        created_at=now,
        updated_at=now,
        status_message="Queued watcher wake.",
    )

    queued = store.create_watcher_notification_task(wake, ["finished-work"])

    assert queued is not None
    assert queued.stage_host is None
    assert queued.stage_root == str(stage)


def test_chat_stage_binding_rejects_a_conflicting_explicit_stage(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    legacy_id = "legacy-project"
    canonical_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    first = tmp_path / "data" / "run-stage" / "chat-first"
    second = tmp_path / "data" / "run-stage" / "chat-second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    native_session_id = str(uuid.uuid4())
    _register_legacy_project(store, legacy_id)
    store.create_agent_task(
        _task(
            store,
            "first",
            legacy_id,
            chat_id,
            stage_root=str(first),
            native_session_id=native_session_id,
        )
    )
    store.create_agent_task(
        _task(
            store,
            "conflicting-history",
            legacy_id,
            chat_id,
            stage_root=str(second),
            native_session_id=native_session_id,
        )
    )
    store.migrate_project_identity(legacy_id, canonical_id, store.space_id)

    with pytest.raises(ValueError, match="conflicting saved workspace bindings"):
        store.create_agent_task(
            _task(
                store,
                "second",
                canonical_id,
                chat_id,
                request_session_id=native_session_id,
            )
        )

    assert store.agent_task("second") is None
