from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from rcp.storage import AppStore, EpisodeRecord, EpisodeReportRecord

from .helpers import authorized_human, create_named_app


def _summary(revision: int, episode_id: str | None) -> dict[str, object]:
    return {
        "from_revision": revision - 1,
        "to_revision": revision,
        "kind": "work",
        "author": "agent",
        "producer": "agent",
        "authorized_by": None,
        "profile": "orchestrator" if episode_id is not None else "ordinary",
        "task_id": f"task-{revision}",
        "episode_id": episode_id,
        "created_at": f"2026-08-12T00:00:0{revision}+00:00",
        "sentences": [f"Recorded revision {revision}."],
    }


def _episode(
    store: AppStore,
    *,
    episode_id: str,
    project_id: str,
    status: str,
    ending: str | None = None,
    wrapup_state: str = "not_started",
) -> EpisodeRecord:
    now = store.now()
    return EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode="auto_research",
        status=status,
        invocation_ceiling=2,
        authorized_by=authorized_human(store),
        ending=ending,
        wrapup_state=wrapup_state,
        created_at=now,
        updated_at=now,
        ended_at=now if status in {"completed", "needs_action", "stopped", "failed"} else None,
    )


def _report(episode_id: str) -> EpisodeReportRecord:
    html = "<article><h1>Episode finding</h1></article>"
    return EpisodeReportRecord(
        report_id=f"{episode_id}-report",
        episode_id=episode_id,
        attempt_id=f"{episode_id}-attempt",
        allocation_operation_id=f"{episode_id}-report-task",
        ending="exhausted",
        sha256=hashlib.sha256(html.encode("utf-8")).hexdigest(),
        html=html,
        created_at="2026-08-12T01:00:00+00:00",
    )


def test_history_episode_decoration_keeps_missing_and_cross_project_ids(
    manifest,
    monkeypatch,
    tmp_path,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    history = app.state.catalog.open(project_id).history
    summaries = [
        _summary(1, None),
        _summary(2, "removed-episode"),
        _summary(3, "foreign-episode"),
    ]
    foreign = _episode(
        store,
        episode_id="foreign-episode",
        project_id="another-project",
        status="running",
    )
    events: list[str] = []

    def project_history(_from_revision: int, _to_revision: int | None):
        events.append("projection")
        return summaries

    def episode_lookup(episode_id: str):
        events.append(f"episode:{episode_id}")
        return foreign if episode_id == foreign.episode_id else None

    def unexpected_report_lookup(_episode_id: str):
        raise AssertionError("absent and cross-project episodes must not load reports")

    monkeypatch.setattr(history, "revision_summaries", project_history)
    monkeypatch.setattr(store, "episode", episode_lookup)
    monkeypatch.setattr(store, "episode_report", unexpected_report_lookup)

    response = TestClient(app).get(f"/api/projects/{project_id}/history/summaries")

    assert response.status_code == 200
    assert events[0] == "projection"
    assert set(events[1:]) == {"episode:removed-episode", "episode:foreign-episode"}
    assert [(item["episode_id"], item["episode"]) for item in response.json()] == [
        (None, None),
        ("removed-episode", None),
        ("foreign-episode", None),
    ]
    assert all("campaign_id" not in item for item in response.json())


def test_history_episode_decoration_maps_lifecycle_and_singular_report(
    manifest,
    monkeypatch,
    tmp_path,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert project_id is not None
    store = app.state.background_tasks.store
    history = app.state.catalog.open(project_id).history
    episodes = {
        "completed": _episode(
            store,
            episode_id="completed",
            project_id=project_id,
            status="completed",
            ending="completed",
            wrapup_state="ready",
        ),
        "exhausted": _episode(
            store,
            episode_id="exhausted",
            project_id=project_id,
            status="needs_action",
            ending="exhausted",
            wrapup_state="ready",
        ),
        "stopped": _episode(
            store,
            episode_id="stopped",
            project_id=project_id,
            status="stopped",
            ending="stopped",
            wrapup_state="skipped",
        ),
        "failed": _episode(
            store,
            episode_id="failed",
            project_id=project_id,
            status="failed",
            ending="failed",
            wrapup_state="failed",
        ),
        "wrapping": _episode(
            store,
            episode_id="wrapping",
            project_id=project_id,
            status="wrapping_up",
            ending="completed",
            wrapup_state="pending",
        ),
    }
    summaries = [_summary(index, episode_id) for index, episode_id in enumerate(episodes, start=1)]
    report = _report("exhausted")

    monkeypatch.setattr(history, "revision_summaries", lambda *_args: summaries)
    monkeypatch.setattr(store, "episode", episodes.get)
    monkeypatch.setattr(
        store,
        "episode_report",
        lambda episode_id: report if episode_id == "exhausted" else None,
    )

    response = TestClient(app).get(f"/api/projects/{project_id}/history/summaries")

    assert response.status_code == 200
    decorated = {item["episode_id"]: item["episode"] for item in response.json()}
    # The row renders a state name, so the decoration carries the name rather than
    # the status enum for a client to capitalize into one.
    assert {
        episode_id: (
            item["mode"],
            item["state_label"],
            item["ending"],
            item["wrapup_state"],
        )
        for episode_id, item in decorated.items()
    } == {
        "completed": ("auto_research", "Completed", "completed", "ready"),
        "exhausted": ("auto_research", "Exhausted", "exhausted", "ready"),
        "stopped": ("auto_research", "Stopped", "stopped", "skipped"),
        "failed": ("auto_research", "Failed", "failed", "failed"),
        "wrapping": ("auto_research", "Completed", "completed", "pending"),
    }
    assert decorated["exhausted"]["report"] == {
        "report_id": "exhausted-report",
        "ending": "exhausted",
        "created_at": "2026-08-12T01:00:00+00:00",
    }
    assert all(
        item["report"] is None
        for episode_id, item in decorated.items()
        if episode_id != "exhausted"
    )
    assert all("reports" not in item for item in decorated.values())
