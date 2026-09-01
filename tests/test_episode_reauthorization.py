from __future__ import annotations

from fastapi.testclient import TestClient

from rcp.runs.auto_research import AutoResearchRunRequest

from .helpers import create_named_app, wait_for_task
from .test_episode_api import create_terminal_auto_episode, settling_auto_research_stream


def test_reauthorize_creates_a_fresh_episode_after_a_report_error(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    tasks = app.state.background_tasks
    store = tasks.store
    stage = tmp_path / "fresh-episode-stage"
    stage.mkdir()
    tasks.stream = settling_auto_research_stream(stage)
    original, original_root, _ = create_terminal_auto_episode(
        store,
        app.state.catalog.open(project_id).history,
        project_id,
        episode_id="exhausted-episode",
        invocation_ceiling=2,
        starting_instruction="Resolve the disputed interpretation.",
        report_error="The report output was invalid.",
    )
    assert original.status == "needs_action"
    assert original.ending == "exhausted"
    assert original.wrapup_state == "failed"

    with TestClient(app) as client:
        before = client.get(f"/api/projects/{project_id}/episodes")
        assert before.status_code == 200
        old_payload = before.json()[0]
        assert old_payload["episode_id"] == original.episode_id
        assert old_payload["wrapup_error"] == "The report output was invalid."
        assert old_payload["report"] is None
        assert old_payload["can_reauthorize"] is True
        assert "report_attempts_used" not in old_payload

        legacy_body = client.post(
            f"/api/projects/{project_id}/episodes/{original.episode_id}/reauthorize",
            json={"additional_invocations": 4},
        )
        assert legacy_body.status_code == 422

        response = client.post(
            f"/api/projects/{project_id}/episodes/{original.episode_id}/reauthorize",
            json={"invocation_ceiling": 4},
        )
        assert response.status_code == 202
        fresh_payload = response.json()
        fresh_episode_id = fresh_payload["episode_id"]
        fresh_operation_id = fresh_payload["root_operation_id"]
        assert fresh_episode_id != original.episode_id
        assert fresh_operation_id != original_root.operation_id
        assert fresh_payload["mode"] == "auto_research"
        assert fresh_payload["graph_target"] == {
            "kind": "branch",
            "branch_id": fresh_episode_id,
        }
        assert fresh_payload["graph_base_head"]["target"] == {
            "kind": "main",
            "branch_id": None,
        }
        assert fresh_payload["graph_branch"]["branch_id"] == fresh_episode_id
        assert fresh_payload["tasks"][0]["graph_target"] == fresh_payload["graph_target"]
        assert fresh_payload["starting_instruction"] == "Resolve the disputed interpretation."
        assert fresh_payload["budget"] == {
            "invocation_ceiling": 4,
            "invocations_used": 1,
            "invocations_remaining": 3,
            "observed_input_tokens": 0,
            "observed_generated_tokens": 0,
        }
        assert fresh_payload["ending"] is None
        assert fresh_payload["wrapup_state"] == "not_started"
        assert fresh_payload["wrapup_error"] is None
        assert fresh_payload["report"] is None

        fresh_root = wait_for_task(store, fresh_operation_id, expect="succeeded")
        fresh_request = AutoResearchRunRequest.model_validate(fresh_root.request)
        assert fresh_root.episode_id == fresh_episode_id
        assert fresh_root.parent_operation_id is None
        assert fresh_request.episode_id == fresh_episode_id
        assert fresh_request.actor_operation_id == fresh_operation_id
        assert fresh_request.session_id is None
        assert fresh_root.native_session_id != original_root.native_session_id

        old_after = client.get(f"/api/projects/{project_id}/episodes").json()
        old_after = next(item for item in old_after if item["episode_id"] == original.episode_id)
        assert old_after["status"] == "needs_action"
        assert old_after["ending"] == "exhausted"
        assert old_after["wrapup_state"] == "failed"
        assert old_after["wrapup_error"] == "The report output was invalid."
        assert old_after["budget"]["invocations_used"] == 1

        report_retry = client.post(
            f"/api/projects/{project_id}/episodes/{original.episode_id}/report/retry"
        )
        assert report_retry.status_code in {404, 405}


def test_reauthorization_preflight_failure_leaves_the_old_episode_unchanged(
    manifest,
    tmp_path,
    monkeypatch,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    service = app.state.catalog.open(project_id)
    original, _root, _ = create_terminal_auto_episode(
        store,
        service.history,
        project_id,
        episode_id="exhausted-episode",
        report_error="The report output was invalid.",
    )
    episodes_before = [item.model_dump(mode="json") for item in store.episodes(project_id)]
    tasks_before = [item.model_dump(mode="json") for item in store.agent_tasks(project_id)]

    def reject_profile(_surface):
        raise ValueError("the pinned orchestrator profile is unavailable")

    monkeypatch.setattr(service, "resolve_agent_profile", reject_profile)

    with TestClient(app) as client:
        response = client.post(
            f"/api/projects/{project_id}/episodes/{original.episode_id}/reauthorize",
            json={"invocation_ceiling": 4},
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "the pinned orchestrator profile is unavailable"}
    assert [item.model_dump(mode="json") for item in store.episodes(project_id)] == episodes_before
    assert [item.model_dump(mode="json") for item in store.agent_tasks(project_id)] == tasks_before
