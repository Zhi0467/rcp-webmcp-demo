from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime, timedelta

from rcp.agents import AgentProcessControl
from rcp.background import AgentTaskExecution
from rcp.limits import RUN_STAGE_RETENTION_DAYS
from rcp.runs.tasks.discuss import _refresh_result_view_retention
from rcp.service import RunRequest
from rcp.storage import AgentTaskRecord, AppStore, ResultViewRecord

_VIEW_HTML = b"<!doctype html><title>Curves</title><p>loss curve</p>"


def _request(chat_id: str) -> RunRequest:
    return RunRequest(
        provider="codex",
        model="",
        reasoning="medium",
        run_on="local",
        chat_scope="node",
        node_id="experiment/pilot",
        message="Discuss the latest curve.",
        chat_id=chat_id,
        mode="discuss",
    )


def _create_execution(
    store: AppStore,
    *,
    project_id: str,
    operation_id: str,
    request: RunRequest,
    stage_root: str,
    now: datetime,
) -> AgentTaskExecution:
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="node_chat",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            status_message="Queued.",
            phase="queued",
            last_activity_at=now.isoformat(),
        )
    )
    return AgentTaskExecution(
        operation_id=operation_id,
        store=store,
        control=AgentProcessControl(),
        stage_root=stage_root,
    )


def _view(
    *,
    view_id: str,
    project_id: str,
    chat_id: str,
    stage_root: str,
    now: datetime,
) -> ResultViewRecord:
    return ResultViewRecord(
        view_id=view_id,
        project_id=project_id,
        experiment_id="experiment/pilot",
        chat_id=chat_id,
        origin_operation_id=f"create-{view_id}",
        latest_operation_id=f"create-{view_id}",
        provider="codex",
        model="",
        reasoning="medium",
        run_on="local",
        native_session_id=str(uuid.uuid4()),
        stage_host="",
        stage_root=stage_root,
        source_name="curves.html",
        content_sha256=hashlib.sha256(_VIEW_HTML).hexdigest(),
        size_bytes=len(_VIEW_HTML),
        created_at=(now - timedelta(minutes=1)).isoformat(),
        updated_at=now.isoformat(),
        expires_at=(now + timedelta(hours=1)).isoformat(),
    )


def test_discuss_turn_touches_every_saved_stage_before_extending_expiry(
    tmp_path,
    monkeypatch,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    project_id = "project-one"
    chat_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    stage = tmp_path / "rcp-run.chat"
    old_stage = tmp_path / "rcp-run.old-chat"
    stage.mkdir()
    old_stage.mkdir()
    old_mtime = stage.stat().st_mtime - 3600
    os.utime(stage, (old_mtime, old_mtime))
    os.utime(old_stage, (old_mtime, old_mtime))
    now = datetime.now(UTC)
    request = _request(chat_id)
    execution = _create_execution(
        store,
        project_id=project_id,
        operation_id=operation_id,
        request=request,
        stage_root=str(stage),
        now=now,
    )
    views = [
        store.create_result_view(
            _view(
                view_id="a" * 24,
                project_id=project_id,
                chat_id=chat_id,
                stage_root=str(stage),
                now=now,
            ),
            html=_VIEW_HTML,
        ),
        store.create_result_view(
            _view(
                view_id="c" * 24,
                project_id=project_id,
                chat_id=chat_id,
                stage_root=str(old_stage),
                now=now,
            ),
            html=_VIEW_HTML,
        ),
    ]
    real_refresh = store.refresh_result_view_expiry

    def refresh_after_both_stages(*args, **kwargs):
        assert stage.stat().st_mtime > old_mtime
        assert old_stage.stat().st_mtime > old_mtime
        return real_refresh(*args, **kwargs)

    monkeypatch.setattr(store, "refresh_result_view_expiry", refresh_after_both_stages)

    _refresh_result_view_retention(
        execution,
        request,
        local_stage=stage,
        remote_stage=None,
    )

    for view in views:
        refreshed = store.result_view_for_diagnostics(view.view_id)
        assert refreshed is not None
        refreshed_expiry = datetime.fromisoformat(refreshed.expires_at)
        assert refreshed_expiry >= now + timedelta(days=RUN_STAGE_RETENTION_DAYS - 1)


def test_discuss_turn_does_not_extend_any_expiry_when_old_stage_is_unavailable(
    tmp_path,
) -> None:
    store = AppStore(tmp_path / "rcp.sqlite3")
    project_id = "project-one"
    chat_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    stage = tmp_path / "rcp-run.chat"
    missing_stage = tmp_path / "rcp-run.missing-chat"
    stage.mkdir()
    old_mtime = stage.stat().st_mtime - 3600
    os.utime(stage, (old_mtime, old_mtime))
    now = datetime.now(UTC)
    request = _request(chat_id)
    execution = _create_execution(
        store,
        project_id=project_id,
        operation_id=operation_id,
        request=request,
        stage_root=str(stage),
        now=now,
    )
    views = [
        store.create_result_view(
            _view(
                view_id="a" * 24,
                project_id=project_id,
                chat_id=chat_id,
                stage_root=str(stage),
                now=now,
            ),
            html=_VIEW_HTML,
        ),
        store.create_result_view(
            _view(
                view_id="c" * 24,
                project_id=project_id,
                chat_id=chat_id,
                stage_root=str(missing_stage),
                now=now,
            ),
            html=_VIEW_HTML,
        ),
    ]

    _refresh_result_view_retention(
        execution,
        request,
        local_stage=stage,
        remote_stage=None,
    )

    assert stage.stat().st_mtime > old_mtime
    for view in views:
        unchanged = store.result_view_for_diagnostics(view.view_id)
        assert unchanged is not None
        assert unchanged.expires_at == view.expires_at
    warnings = [
        event for event in store.agent_task_events(operation_id) if event.level == "warning"
    ]
    assert len(warnings) == 1
    assert "Result-view retention could not be refreshed" in warnings[0].message
