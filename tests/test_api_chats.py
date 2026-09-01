from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from rcp.transport import StateUnavailable

from .helpers import create_named_app


def test_chat_history_is_paginated_from_full_canonical_transcripts(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    chat_dir = manifest.research_dir / "chat"
    chat_dir.mkdir(exist_ok=True)
    chat_ids: list[str] = []
    long_answer = "full-answer-" + ("x" * 20_000)

    for index in range(23):
        chat_id = str(uuid.uuid4())
        chat_ids.append(chat_id)
        timestamp = f"2026-07-31T{index:02d}:00:00+00:00"
        answer = long_answer if index == 0 else f"answer {index}"
        common = {
            "sessionId": chat_id,
            "nativeSessionId": str(uuid.uuid4()),
            "nodeId": None,
            "chatScope": "project",
            "provider": "codex",
            "model": "provider-default",
            "reasoning": "medium",
            "executionMachine": "laptop",
            "timestamp": timestamp,
        }
        records = [
            {
                **common,
                "uuid": str(uuid.uuid4()),
                "type": "user",
                "role": "user",
                "text": f"question {index}",
            },
            {
                **common,
                "uuid": str(uuid.uuid4()),
                "type": "assistant",
                "role": "assistant",
                "text": answer,
                "appliedRevision": None,
            },
        ]
        (chat_dir / f"project-{chat_id}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    # Neither malformed files nor a valid transcript reached through a symlink
    # can enter the canonical list.
    (chat_dir / "malformed.jsonl").write_text("not json\n", encoding="utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text(
        (chat_dir / f"project-{chat_ids[0]}.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (chat_dir / "outside-link.jsonl").symlink_to(outside)

    service = app.state.service
    original_read = service._read_chat_transcript
    parsed_paths: list[Path] = []

    def counted_read(path: Path):
        parsed_paths.append(path)
        return original_read(path)

    monkeypatch.setattr(service, "_read_chat_transcript", counted_read)

    first = client.get(f"/api/projects/{project_id}/chats?offset=0&limit=7")
    assert first.status_code == 200
    assert first.json()["total"] == 23
    assert first.json()["offset"] == 0
    assert first.json()["limit"] == 7
    assert len(first.json()["items"]) == 7
    assert first.json()["items"][0]["title"] == "question 22"
    parsed_after_first_page = len(parsed_paths)
    assert parsed_after_first_page > len(first.json()["items"])

    discovered: list[str] = []
    for offset in range(0, 28, 7):
        page = client.get(f"/api/projects/{project_id}/chats?offset={offset}&limit=7").json()
        discovered.extend(item["chat_id"] for item in page["items"])
    assert set(discovered) == set(chat_ids)
    assert len(discovered) == 23
    assert len(parsed_paths) == parsed_after_first_page

    parsed_paths.clear()
    transcript = client.get(f"/api/projects/{project_id}/chats/{chat_ids[0]}")
    assert transcript.status_code == 200
    assert [message["role"] for message in transcript.json()["messages"]] == [
        "user",
        "assistant",
    ]
    assert transcript.json()["messages"][-1]["text"] == long_answer
    assert len(transcript.json()["messages"][-1]["text"]) > 16_000
    assert parsed_paths == [chat_dir / f"project-{chat_ids[0]}.jsonl"]

    assert client.get(f"/api/projects/{project_id}/chats/not-a-uuid").status_code == 422
    assert client.get(f"/api/projects/{project_id}/chats/{uuid.uuid4()}").status_code == 404


def test_chat_history_reports_remote_refresh_failure_as_unavailable(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    project_id = app.state.default_project_id
    service.history.workspace.remote = True
    refresh_threads: list[int] = []

    def fail_refresh() -> bool:
        refresh_threads.append(threading.get_ident())
        raise StateUnavailable("remote transcript refresh failed")

    monkeypatch.setattr(service.history.workspace, "refresh", fail_refresh)
    request_thread = threading.get_ident()

    response = TestClient(app).get(f"/api/projects/{project_id}/chats")

    assert response.status_code == 503
    assert response.json()["detail"] == "remote transcript refresh failed"
    assert refresh_threads and refresh_threads[0] != request_thread


def test_non_main_project_route_does_not_build_project_snapshot(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    monkeypatch.setattr(
        app.state.service,
        "project_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("chat history must not serialize the whole project")
        ),
    )

    response = TestClient(app).get(f"/api/projects/{project_id}/chats")

    assert response.status_code == 200
    assert response.json()["items"] == []
