from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rcp.core.models import Patch
from rcp.runs.tasks.result_views import prepare_local_result_view_slot
from rcp.service import ProjectService, RunRequest
from rcp.storage import AgentTaskRecord, AppStore, ResultViewRecord
from rcp.transport import StateUnavailable

from .helpers import append_fixture_patch, create_named_app, seed_patch

_VIEW_ID = "a" * 24
_EXPERIMENT_ID = "exp/result-view"


@dataclass(frozen=True)
class _ResultViewApiFixture:
    client: TestClient
    store: AppStore
    service: ProjectService
    project_id: str
    record: ResultViewRecord
    stage: Path
    source: Path
    content: bytes


def _experiment_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added an Experiment for result-view API tests.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": _EXPERIMENT_ID,
                        "type": "experiment",
                        "title": "Inspect training results",
                        "objective": "Exercise the result-view API contract.",
                    }
                ],
            }
        ],
    )


def _create_view(
    store: AppStore,
    *,
    project_id: str,
    chat_id: str,
    stage: Path,
    view_id: str,
    content: bytes,
    expires_at: datetime,
    source_name: str = "curves.html",
    created_at: datetime | None = None,
    stage_host: str = "",
) -> tuple[ResultViewRecord, Path]:
    stage.mkdir(parents=True, exist_ok=True)
    source = prepare_local_result_view_slot(stage, view_id, reuse=False) / source_name
    source.write_bytes(content)
    created_at = created_at or datetime.now(UTC)
    record = store.create_result_view(
        ResultViewRecord(
            view_id=view_id,
            project_id=project_id,
            experiment_id=_EXPERIMENT_ID,
            chat_id=chat_id,
            origin_operation_id=f"origin-{view_id}",
            latest_operation_id=f"origin-{view_id}",
            provider="codex",
            model="",
            reasoning="high",
            run_on="laptop",
            native_session_id=f"native-{view_id}",
            stage_host=stage_host,
            stage_root=str(stage),
            source_name=source_name,
            content_sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            created_at=created_at.isoformat(),
            updated_at=created_at.isoformat(),
            expires_at=expires_at.isoformat(),
        ),
        html=content,
    )
    return record, source


def _fixture(manifest, tmp_path: Path) -> _ResultViewApiFixture:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_patch())
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id
    stage = tmp_path / "conversation-stage"
    content = b"<!doctype html><title>Loss curves</title><p>loss curve</p>"
    record, source = _create_view(
        store,
        project_id=project_id,
        chat_id=str(uuid.uuid4()),
        stage=stage,
        view_id=_VIEW_ID,
        content=content,
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    return _ResultViewApiFixture(
        client=TestClient(app),
        store=store,
        service=service,
        project_id=project_id,
        record=record,
        stage=stage,
        source=source,
        content=content,
    )


def test_result_view_list_preview_and_head_are_path_free(manifest, tmp_path: Path) -> None:
    fixture = _fixture(manifest, tmp_path)
    base = f"/api/projects/{fixture.project_id}/result-views"

    listed = fixture.client.get(
        base,
        params={
            "experiment_id": fixture.record.experiment_id,
            "chat_id": fixture.record.chat_id,
        },
    )

    assert listed.status_code == 200
    assert listed.json() == [
        fixture.store.result_view_descriptor(fixture.record).model_dump(mode="json")
    ]
    assert fixture.client.get(base, params={"chat_id": "another-chat"}).json() == []
    serialized = json.dumps(listed.json())
    assert str(fixture.stage) not in serialized
    assert fixture.record.content_sha256 not in serialized
    assert fixture.record.native_session_id not in serialized

    preview_url = f"{base}/{fixture.record.view_id}/preview"
    preview = fixture.client.get(preview_url)
    head = fixture.client.head(preview_url)

    assert preview.status_code == head.status_code == 200
    assert "loss curve" in preview.text
    assert "rcp-result-view-gesture" in preview.text
    assert head.content == b""
    for header in (
        "cache-control",
        "content-length",
        "content-security-policy",
        "content-type",
        "x-content-type-options",
    ):
        assert head.headers[header] == preview.headers[header]
    assert "default-src 'none'" in preview.headers["content-security-policy"]
    assert fixture.client.get(f"{base}/{'f' * 24}/preview").status_code == 404


def test_preview_uses_stored_bytes_after_local_stage_changes_or_disappears(
    manifest,
    tmp_path: Path,
) -> None:
    fixture = _fixture(manifest, tmp_path)
    preview_url = (
        f"/api/projects/{fixture.project_id}/result-views/{fixture.record.view_id}/preview"
    )

    fixture.source.write_bytes(b"<!doctype html><p>changed after discovery</p>")
    changed = fixture.client.get(preview_url)

    assert changed.status_code == 200
    assert "loss curve" in changed.text
    assert "changed after discovery" not in changed.text

    fixture.source.unlink()
    deleted = fixture.client.get(preview_url)

    assert deleted.status_code == 200
    assert "loss curve" in deleted.text


def test_preview_rejects_expired_unavailable_and_corrupt_stored_bytes(
    manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(manifest, tmp_path)
    base = f"/api/projects/{fixture.project_id}/result-views"
    expired_at = datetime.now(UTC)
    expired, _ = _create_view(
        fixture.store,
        project_id=fixture.project_id,
        chat_id=fixture.record.chat_id,
        stage=tmp_path / "expired-stage",
        view_id="b" * 24,
        content=b"<html>expired</html>",
        created_at=expired_at - timedelta(seconds=2),
        expires_at=expired_at - timedelta(seconds=1),
    )
    assert fixture.client.get(f"{base}/{expired.view_id}/preview").status_code == 410

    def unavailable_result_view_bytes(*args, **kwargs):
        raise OSError("stored result view unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(fixture.store, "result_view_bytes", unavailable_result_view_bytes)
        unavailable_response = fixture.client.get(f"{base}/{fixture.record.view_id}/preview")
    assert unavailable_response.status_code == 503
    assert "stored result view unavailable" not in unavailable_response.text

    with fixture.store.connection() as connection:
        connection.execute(
            "UPDATE result_views SET html = ? WHERE view_id = ?",
            ("<html>corrupt stored bytes</html>", fixture.record.view_id),
        )
    corrupt = fixture.client.get(f"{base}/{fixture.record.view_id}/preview")
    assert corrupt.status_code == 410
    assert "corrupt stored bytes" not in corrupt.text


def test_keep_is_idempotent_and_kept_preview_never_returns_to_scratch(
    manifest,
    tmp_path: Path,
) -> None:
    fixture = _fixture(manifest, tmp_path)
    base = f"/api/projects/{fixture.project_id}/result-views/{fixture.record.view_id}"
    revision_before = fixture.service.history.state().revision
    fixture.source.write_bytes(b"<!doctype html><p>changed after discovery</p>")
    fixture.source.unlink()

    first = fixture.client.post(f"{base}/keep")
    second = fixture.client.post(f"{base}/keep")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["state"] == "kept"
    kept_filename = first.json()["kept_filename"]
    assert isinstance(kept_filename, str)
    repository_views = fixture.service.manifest.research_dir.parent / "views"
    assert [path.name for path in repository_views.iterdir()] == [kept_filename]
    assert (repository_views / kept_filename).read_bytes() == fixture.content
    assert not (fixture.service.manifest.research_dir / "views").exists()
    assert fixture.service.history.state().revision == revision_before

    preview = fixture.client.get(f"{base}/preview")
    assert preview.status_code == 200
    assert "loss curve" in preview.text

    (repository_views / kept_filename).write_bytes(b" " * len(fixture.content))
    changed = fixture.client.get(f"{base}/preview")
    assert changed.status_code == 200
    assert "loss curve" in changed.text

    (repository_views / kept_filename).unlink()
    deleted = fixture.client.get(f"{base}/preview")
    assert deleted.status_code == 200
    assert "loss curve" in deleted.text


def test_failed_keep_preserves_the_temporary_view_and_hides_storage_paths(
    manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(manifest, tmp_path)
    keep_attempted = False

    def fail_keep(**_kwargs) -> str:
        nonlocal keep_attempted
        keep_attempted = True
        raise StateUnavailable(f"could not publish from {fixture.stage}")

    monkeypatch.setattr(fixture.service.history.workspace, "keep_result_view", fail_keep)
    base = f"/api/projects/{fixture.project_id}/result-views/{fixture.record.view_id}"
    fixture.source.write_bytes(b"<!doctype html><p>changed after discovery</p>")
    fixture.source.unlink()

    failed = fixture.client.post(f"{base}/keep")

    assert failed.status_code == 503
    assert keep_attempted
    assert str(fixture.stage) not in failed.text
    current = fixture.store.result_view_for_diagnostics(fixture.record.view_id)
    assert current is not None and current.kept_filename is None
    preview = fixture.client.get(f"{base}/preview")
    assert preview.status_code == 200
    assert "loss curve" in preview.text


def test_remote_result_view_preview_and_keep_never_read_the_stage(
    manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(manifest, tmp_path)
    remote_content = b"<!doctype html><title>Remote curves</title><p>remote stored curve</p>"
    remote, source = _create_view(
        fixture.store,
        project_id=fixture.project_id,
        chat_id=fixture.record.chat_id,
        stage=tmp_path / "irrelevant-local-stage",
        stage_host="research-host",
        view_id="d" * 24,
        content=remote_content,
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    source.unlink()

    def must_not_open_remote_stage(*_args, **_kwargs):
        raise AssertionError("result-view HTTP must not access the remote stage")

    monkeypatch.setattr("rcp.api.app.RemoteRunStage", must_not_open_remote_stage)
    base = f"/api/projects/{fixture.project_id}/result-views/{remote.view_id}"

    preview = fixture.client.get(f"{base}/preview")
    kept = fixture.client.post(f"{base}/keep")

    assert preview.status_code == 200
    assert "remote stored curve" in preview.text
    assert kept.status_code == 200
    kept_filename = kept.json()["kept_filename"]
    repository_view = fixture.service.manifest.research_dir.parent / "views" / kept_filename
    assert repository_view.read_bytes() == remote_content


def test_new_special_result_view_intents_are_rejected(
    manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(manifest, tmp_path)
    started = False

    def fake_start(*_args, **_kwargs):
        nonlocal started
        started = True
        raise AssertionError("a retired result-view intent must not start a task")

    monkeypatch.setattr(fixture.client.app.state.background_tasks, "start", fake_start)
    admitted = fixture.client.post(
        f"/api/projects/{fixture.project_id}/tasks/node_chat",
        json={
            "provider": "claude",
            "model": "different-model",
            "reasoning": "low",
            "run_on": "laptop",
            "chat_id": fixture.record.chat_id,
            "node_id": fixture.record.experiment_id,
            "message": "Use a log scale, but keep my wording exactly.",
            "mode": "work",
            "result_view": {"action": "revise", "view_id": fixture.record.view_id},
        },
    )

    assert admitted.status_code == 422
    assert "ordinary task artifacts" in admitted.text
    assert not started


@pytest.mark.parametrize("status", ["paused", "interrupted"])
def test_keep_waits_for_a_resumable_revision_then_allows_recovery(
    manifest,
    tmp_path: Path,
    monkeypatch,
    status: str,
) -> None:
    fixture = _fixture(manifest, tmp_path)
    request = RunRequest(
        provider=fixture.record.provider,
        model=fixture.record.model,
        reasoning=fixture.record.reasoning,
        run_on=fixture.record.run_on,
        chat_scope="node",
        node_id=fixture.record.experiment_id,
        message="Use a log scale.",
        chat_id=fixture.record.chat_id,
        session_id=fixture.record.native_session_id,
        mode="work",
        result_view={"action": "revise", "view_id": fixture.record.view_id},
    )
    now = fixture.store.now()
    recoverable = fixture.store.create_agent_task(
        AgentTaskRecord(
            operation_id=f"{status}-result-view-revision",
            project_id=fixture.project_id,
            kind="node_chat",
            status=status,
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message=f"{status.title()}.",
            native_session_id=fixture.record.native_session_id,
            stage_root=fixture.record.stage_root,
        )
    )
    assert recoverable.can_resume
    base = f"/api/projects/{fixture.project_id}"

    blocked = fixture.client.post(f"{base}/result-views/{fixture.record.view_id}/keep")
    assert blocked.status_code == 409
    assert "active result view revision" in blocked.text

    tasks = fixture.client.app.state.background_tasks
    monkeypatch.setattr(tasks, "_spawn_record", lambda record, *_args, **_kwargs: record)
    resumed = fixture.client.post(f"{base}/tasks/{recoverable.operation_id}/resume")

    assert resumed.status_code == 202
    child_id = resumed.json()["operation_id"]
    child = fixture.store.agent_task(child_id)
    assert child is not None
    assert child.parent_operation_id == recoverable.operation_id
    assert child.stage_root == fixture.record.stage_root
    assert child.native_session_id == fixture.record.native_session_id
    fixture.store.complete_agent_task(child_id, applied_revision=None, result={})

    kept = fixture.client.post(f"{base}/result-views/{fixture.record.view_id}/keep")
    assert kept.status_code == 200
    assert kept.json()["state"] == "kept"


def test_result_view_resume_holds_project_admission_before_view_lock(
    manifest,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(manifest, tmp_path)
    request = RunRequest(
        provider=fixture.record.provider,
        model=fixture.record.model,
        reasoning=fixture.record.reasoning,
        run_on=fixture.record.run_on,
        chat_scope="node",
        node_id=fixture.record.experiment_id,
        message="Use a log scale.",
        chat_id=fixture.record.chat_id,
        session_id=fixture.record.native_session_id,
        mode="work",
        result_view={"action": "revise", "view_id": fixture.record.view_id},
    )
    now = fixture.store.now()
    recoverable = fixture.store.create_agent_task(
        AgentTaskRecord(
            operation_id="paused-result-view-lock-order",
            project_id=fixture.project_id,
            kind="node_chat",
            status="paused",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Paused.",
            native_session_id=fixture.record.native_session_id,
            stage_root=fixture.record.stage_root,
        )
    )
    project_lock = fixture.client.app.state.services.experiment_operation_lock(fixture.project_id)
    observed = threading.Event()

    from rcp.api import tasks as tasks_api

    real_admit = tasks_api._admit_result_view_request

    def assert_project_lock_first(*args, **kwargs):
        assert project_lock.locked()
        observed.set()
        return real_admit(*args, **kwargs)

    monkeypatch.setattr(tasks_api, "_admit_result_view_request", assert_project_lock_first)
    monkeypatch.setattr(
        fixture.client.app.state.background_tasks,
        "_spawn_record",
        lambda record, *_args, **_kwargs: record,
    )

    resumed = fixture.client.post(
        f"/api/projects/{fixture.project_id}/tasks/{recoverable.operation_id}/resume"
    )

    assert resumed.status_code == 202, resumed.text
    assert observed.is_set()
