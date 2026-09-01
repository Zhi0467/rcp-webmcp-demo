from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from rcp.agents import AgentEvent
from rcp.background import AgentTaskExecution
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import GraphBranchMetadata
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.history import HistoryManager
from rcp.runs.auto_research import (
    AutoResearchRunRequest,
    auto_research_exhaustion_signal,
    auto_research_wrapup_spec,
)
from rcp.runs.episodes.wrapup import begin_episode_report_wrapup
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchStateRecord,
    EpisodeRecord,
    EpisodeReportRecord,
)

from .helpers import authorized_human, create_named_app, wait_for_task


def _sse(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def settling_auto_research_stream(stage: Path):
    async def stream(
        _project_id: str,
        kind: str,
        request: object,
        execution: AgentTaskExecution,
    ) -> AsyncIterator[str]:
        assert kind == "auto_research"
        assert isinstance(request, AutoResearchRunRequest)
        execution.checkpoint_stage("", str(stage))
        yield _sse(AgentEvent(event="session", session_id=f"session-{execution.operation_id}"))
        yield _sse(AgentEvent(event="done"))

    return stream


def create_terminal_auto_episode(
    store: AppStore,
    history: HistoryManager,
    project_id: str,
    *,
    episode_id: str,
    invocation_ceiling: int = 2,
    starting_instruction: str | None = "Trace the strongest evidence.",
    report_html: str | None = None,
    report_error: str | None = None,
) -> tuple[EpisodeRecord, AgentTaskRecord, EpisodeReportRecord | None]:
    if (report_html is None) == (report_error is None):
        raise ValueError("provide exactly one report result")
    episode_id = str(
        uuid.UUID(bytes=hashlib.sha256(episode_id.encode("utf-8")).digest()[:16], version=4)
    )
    now = store.now()
    operation_id = f"{episode_id}-root"
    authorizer = authorized_human(store)
    graph_base_head = history.head_ref()
    graph_target = GraphTargetRef(kind="branch", branch_id=episode_id)
    history.create_auto_research_branch(
        GraphBranchMetadata(
            branch_id=episode_id,
            episode_id=episode_id,
            project_id=project_id,
            base_head=graph_base_head,
            head=GraphHeadRef(
                target=graph_target,
                revision=graph_base_head.revision,
                transition_id=graph_base_head.transition_id,
            ),
            authorized_by=authorizer,
        )
    )
    run_request = AutoResearchRunRequest(
        episode_id=episode_id,
        role="orchestrator",
        actor_operation_id=operation_id,
        provider="codex",
        model="test-model",
        reasoning="medium",
        run_on="local",
        run_truth_scope=["repo-a"],
        instruction=starting_instruction,
    )
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode="auto_research",
        graph_target=graph_target,
        graph_base_head=graph_base_head,
        status="queued",
        invocation_ceiling=invocation_ceiling,
        authorized_by=authorizer,
        created_at=now,
        updated_at=now,
    )
    root = AgentTaskRecord(
        operation_id=operation_id,
        project_id=project_id,
        episode_id=episode_id,
        graph_target=graph_target,
        kind="auto_research",
        status="queued",
        request=run_request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        status_message="queued",
        authorized_by=authorizer,
        dispatch_authority=AgentDispatchAuthority(
            profile="orchestrator",
            task_contract="orchestrate",
            scope=AgentDispatchScope(
                run_truth_scope=["repo-a"],
                episode_id=episode_id,
                patch_kind="work",
            ),
        ),
    )
    episode, root = store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode_id,
            starting_instruction=starting_instruction,
            created_at=now,
            updated_at=now,
        ),
        root,
    )
    stage_root = f"/tmp/{episode_id}-stage"
    native_session_id = f"{episode_id}-session"
    store.checkpoint_agent_task(
        root.operation_id,
        native_session_id=native_session_id,
        stage_root=stage_root,
    )
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    root = store.agent_task(root.operation_id)
    assert root is not None

    signal = auto_research_exhaustion_signal(store, episode_id)
    admission = begin_episode_report_wrapup(
        store,
        auto_research_wrapup_spec(store, signal),
    )
    assert admission.task is not None
    allocation_operation_id = admission.task.operation_id
    attempt = store.allocate_episode_report_attempt(episode_id)

    report: EpisodeReportRecord | None = None
    if report_html is not None:
        report = EpisodeReportRecord(
            report_id=f"{episode_id}-report",
            episode_id=episode_id,
            attempt_id=attempt.attempt_id,
            allocation_operation_id=allocation_operation_id,
            ending="exhausted",
            sha256=hashlib.sha256(report_html.encode("utf-8")).hexdigest(),
            html=report_html,
            created_at=store.now(),
        )
        store.finish_episode_report_ready(attempt.attempt_id, report)
    else:
        assert report_error is not None
        store.finish_episode_report_error(attempt.attempt_id, report_error)

    stored_episode = store.episode(episode_id)
    assert stored_episode is not None
    return stored_episode, root, report


def test_episode_list_start_and_stop_use_only_the_canonical_surface(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    tasks = app.state.background_tasks
    store = tasks.store
    stage = tmp_path / "auto-stage"
    stage.mkdir()
    tasks.stream = settling_auto_research_stream(stage)

    with TestClient(app) as client:
        assert client.get(f"/api/projects/{project_id}/episodes").json() == []

        started = client.post(
            f"/api/projects/{project_id}/episodes",
            json={
                "mode": "auto_research",
                "invocation_ceiling": 3,
                "starting_instruction": "  Follow the contradictory evidence.  ",
            },
        )
        assert started.status_code == 202
        payload = started.json()
        episode_id = payload["episode_id"]
        operation_id = payload["root_operation_id"]
        wait_for_task(store, operation_id, expect="succeeded")

        listed = client.get(
            f"/api/projects/{project_id}/episodes",
            params={"mode": "auto_research"},
        )
        assert listed.status_code == 200
        assert len(listed.json()) == 1
        current = listed.json()[0]
        assert current["episode_id"] == episode_id
        assert current["graph_target"] == {"kind": "branch", "branch_id": episode_id}
        assert current["graph_base_head"]["target"] == {"kind": "main", "branch_id": None}
        assert current["graph_branch"]["branch_id"] == episode_id
        assert current["graph_branch"]["base_head"] == current["graph_base_head"]
        assert current["tasks"][0]["graph_target"] == current["graph_target"]
        assert current["starting_instruction"] == "Follow the contradictory evidence."
        assert current["budget"]["invocations_used"] == 1
        assert [task["kind"] for task in current["tasks"]] == ["auto_research"]
        assert current["report"] is None
        assert "campaign_id" not in current
        assert "reports" not in current

        legacy = client.post(
            f"/api/projects/{project_id}/campaigns",
            json={"invocation_ceiling": 3},
        )
        assert legacy.status_code in {404, 405}
        legacy_body = client.post(
            f"/api/projects/{project_id}/episodes",
            json={
                "mode": "auto_research",
                "invocation_ceiling": 3,
                "campaign_id": "legacy",
            },
        )
        assert legacy_body.status_code == 422

        stopped = client.post(f"/api/projects/{project_id}/episodes/{episode_id}/stop")
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "stopped"
        assert stopped.json()["ending"] == "stopped"
        assert stopped.json()["wrapup_state"] == "skipped"
        assert stopped.json()["wrapup_error"] is None
        assert stopped.json()["report"] is None
        assert stopped.json()["budget"]["invocations_used"] == 1
        assert all(
            task.kind != "episode_report"
            for task in store.episode_tasks(episode_id, include_hidden=True)
        )


def test_episode_mail_is_durable_when_immediate_delivery_fails(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    tasks = app.state.background_tasks
    store = tasks.store
    stage = tmp_path / "auto-stage"
    stage.mkdir()
    tasks.stream = settling_auto_research_stream(stage)

    def delivery_is_temporarily_unavailable(*_args, **_kwargs):
        raise RuntimeError("delivery transport is unavailable")

    monkeypatch.setattr(
        "rcp.api.episode_routes.deliver_pending_auto_research_mail",
        delivery_is_temporarily_unavailable,
    )

    with TestClient(app) as client:
        started = client.post(
            f"/api/projects/{project_id}/episodes",
            json={"mode": "auto_research", "invocation_ceiling": 3},
        )
        assert started.status_code == 202
        episode_id = started.json()["episode_id"]
        wait_for_task(store, started.json()["root_operation_id"], expect="succeeded")

        sent = client.post(
            f"/api/projects/{project_id}/episodes/{episode_id}/messages",
            json={"body": "  What remains uncertain?  "},
        )
        assert sent.status_code == 201
        assert sent.json()["episode_id"] == episode_id
        assert sent.json()["body"] == "What remains uncertain?"
        assert sent.json()["delivered_at"] is None
        assert sent.json()["delivery_operation_id"] is None

        inbox = client.get(f"/api/projects/{project_id}/episodes/{episode_id}/messages")
        assert inbox.status_code == 200
        assert inbox.json() == [sent.json()]

        blank = client.post(
            f"/api/projects/{project_id}/episodes/{episode_id}/messages",
            json={"body": " \n "},
        )
        assert blank.status_code == 422


def test_episode_report_preview_is_singular_and_sandboxed(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    html = (
        "<article><h1>Episode report</h1>"
        "<figure><svg aria-label='Evidence map'></svg></figure></article>"
    )
    episode, root, report = create_terminal_auto_episode(
        store,
        app.state.catalog.open(project_id).history,
        project_id,
        episode_id="reported-episode",
        report_html=html,
    )
    assert report is not None

    url = f"/api/projects/{project_id}/episodes/{episode.episode_id}/report/content"
    legacy_preview_url = f"/api/projects/{project_id}/episodes/{episode.episode_id}/report/preview"
    viewer_url = f"/api/projects/{project_id}/episodes/{episode.episode_id}/report/viewer"
    with TestClient(app) as client:
        listed = client.get(f"/api/projects/{project_id}/episodes")
        assert listed.status_code == 200
        payload = listed.json()[0]
        assert payload["report"] == {
            "report_id": report.report_id,
            "ending": "exhausted",
            "created_at": report.created_at,
        }
        assert [task["operation_id"] for task in payload["tasks"]] == [root.operation_id]
        assert all(task["kind"] != "episode_report" for task in payload["tasks"])
        assert "reports" not in payload
        assert "report_attempts_used" not in payload

        preview = client.get(url)
        head = client.head(url)
        legacy_preview = client.get(legacy_preview_url)
        viewer = client.get(viewer_url)

    assert preview.status_code == head.status_code == 200
    assert preview.content
    assert head.content == b""
    expected_length = str(len(preview.content))
    assert preview.headers["content-length"] == expected_length
    assert head.headers["content-length"] == expected_length
    for response in (preview, head):
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "default-src 'none'" in response.headers["content-security-policy"]
        assert "frame-src 'self'" in response.headers["content-security-policy"]
    assert 'sandbox="allow-scripts"' in preview.text
    assert "allow-top-navigation" not in preview.text
    assert "rcp-result-view-gesture" not in preview.text
    assert "connect-src &amp;#x27;none&amp;#x27;" in preview.text
    assert viewer.status_code == 200
    assert "rcp-artifact-context" in viewer.text
    assert '"source": "episode_report"' in viewer.text
    assert 'id="keep"' not in viewer.text
    assert legacy_preview.status_code == 200
    assert "rcp-artifact-context" in legacy_preview.text
    assert url in legacy_preview.text
    with TestClient(app) as client:
        assert client.head(legacy_preview_url).content == b""
    assert "Select text in the artifact or draw a box." in viewer.text
