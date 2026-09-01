from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from rcp.agents import AgentProcessControl
from rcp.agents.prompts import _chat_attachment_section
from rcp.artifacts import (
    AgentArtifactDescriptor,
    artifact_viewer_document,
    descriptor_for,
    validate_artifact_bytes,
)
from rcp.background import AgentTaskExecution
from rcp.runs.chat import finalize_artifact_revision
from rcp.service import RunRequest
from rcp.storage import AgentTaskRecord
from rcp.transport import LocalStateWorkspace

from .helpers import create_named_app


def _workspace(tmp_path: Path) -> LocalStateWorkspace:
    research = tmp_path / "repository" / ".research"
    research.mkdir(parents=True)
    return LocalStateWorkspace(research, str(research))


def test_svg_is_an_ordinary_bounded_artifact() -> None:
    data = b'<svg xmlns="http://www.w3.org/2000/svg"><text>result</text></svg>'

    assert validate_artifact_bytes("result.svg", data) == "image/svg+xml"
    descriptor = descriptor_for(
        "01234567-89ab-cdef-0123-456789abcdef", "result.svg", size_bytes=len(data)
    )

    assert descriptor.media_type == "image/svg+xml"
    assert descriptor.size_bytes == len(data)


def test_keep_reuses_human_artifacts_directory_and_reads_external_edits(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    artifacts = workspace.root.parent / "artifacts"
    artifacts.mkdir()
    human_file = artifacts / "notes.html"
    human_file.write_text("human", encoding="utf-8")

    kept = workspace.keep_artifact(
        source_name="curves.html",
        project_name="Pilot",
        data=b"<p>first</p>",
        today=date(2026, 8, 27),
    )

    assert human_file.read_text(encoding="utf-8") == "human"
    assert kept == "curves-pilot-26-08-27.html"
    (artifacts / kept).write_bytes(b"<p>external</p>")
    assert workspace.read_kept_artifact(kept) == b"<p>external</p>"

    workspace.replace_kept_artifact(kept, b"<p>agent revision</p>")
    assert workspace.read_kept_artifact(kept) == b"<p>agent revision</p>"


def test_keep_refuses_unsafe_artifacts_entry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    target = workspace.root.parent / "elsewhere"
    target.mkdir()
    (workspace.root.parent / "artifacts").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="artifacts path is not a regular directory"):
        workspace.keep_artifact(
            source_name="curves.html",
            project_name="Pilot",
            data=b"<p>first</p>",
        )


def test_viewer_assembles_transient_context_without_dispatch_or_mode_change() -> None:
    descriptor = AgentArtifactDescriptor(
        artifact_id="0123456789abcdef01234567",
        name="curves.html",
        media_type="text/html",
        size_bytes=128,
    )

    document, csp = artifact_viewer_document(
        preview_url="/preview",
        keep_url="/keep",
        project_id="project",
        chat_id="chat",
        operation_id="operation",
        descriptor=descriptor,
    )

    assert "rcp-artifact-context" in document
    assert "Added to the originating chat draft." in document
    assert "BroadcastChannel('rcp-artifact-context')" in document
    assert "mode" not in document
    assert "fetch(config.keepUrl" in document
    assert "A prompt can include at most 12 selections." in document
    assert "if(boxWidth<=0||boxHeight<=0)" in document
    assert "connect-src 'self'" in csp
    assert "img-src 'self' data: blob:" in csp


def test_prompt_addresses_comments_without_implying_an_edit() -> None:
    section = _chat_attachment_section(
        [
            {
                "path": "/tmp/curves.html",
                "name": "curves.html",
                "source_artifact_id": "0123456789abcdef01234567",
                "selections": [{"kind": "text", "text": "spike", "comment": "why?"}],
                "revision_output_path": "/tmp/output/curves.html",
            }
        ]
    )

    assert "Address every comment and question" in section
    assert "does not by itself request an edit" in section
    assert "explicitly asks to change the artifact and this is a Work turn" in section
    assert "Never create a second artifact as a revision" in section


def test_box_selection_must_stay_inside_its_normalized_viewport() -> None:
    with pytest.raises(ValueError, match="must stay inside its viewport"):
        RunRequest.model_validate(
            {
                "artifact_context": {
                    "operation_id": "origin",
                    "artifact_id": "0123456789abcdef01234567",
                    "selections": [
                        {
                            "kind": "box",
                            "rect": {"x": 0.75, "y": 0, "width": 0.5, "height": 0.5},
                            "viewport": {"width": 800, "height": 600},
                        }
                    ],
                }
            }
        )


def test_work_revision_replaces_the_same_kept_artifact_without_a_second_card(
    manifest,
    tmp_path: Path,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id
    chat_id = "3a979535-17c3-4fd2-85fc-219de0ee7a75"
    origin_id = "241df76b-d927-496d-a9a1-02ba7537f9ec"
    revision_id = "d70b7937-ed31-44b7-9823-c2af557d3161"
    name = "curves.html"
    first = b"<!doctype html><p>first</p>"
    second = b"<!doctype html><p>second</p>"
    kept_filename = service.history.workspace.keep_artifact(
        source_name=name,
        project_name="Pilot",
        data=first,
        today=date(2026, 8, 27),
    )
    source = descriptor_for(origin_id, name, size_bytes=len(first)).model_copy(
        update={"kept_filename": kept_filename, "kept_at": store.now()}
    )
    origin_request = RunRequest(
        provider="codex",
        model="",
        reasoning="medium",
        run_on="laptop",
        chat_scope="project",
        chat_id=chat_id,
        message="Create the curves.",
        mode="discuss",
    )
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=origin_id,
            project_id=project_id,
            kind="project_chat",
            status="succeeded",
            request=origin_request.model_dump(mode="json"),
            result={"messages": ["Created."], "artifacts": [source.model_dump(mode="json")]},
            created_at=now,
            updated_at=now,
            status_message="Completed.",
            native_session_id="artifact-session",
            stage_root=str(tmp_path / "origin-stage"),
        )
    )
    store.record_agent_task_receipt(
        origin_id,
        "operation_created",
        {"kind": "project_chat", "attempt": 1, "has_parent": False, "resumed": False},
    )
    revision_request = RunRequest.model_validate(
        {
            **origin_request.model_dump(mode="python"),
            "message": "Make the requested change.",
            "mode": "work",
            "session_id": "artifact-session",
            "artifact_context": {
                "source": "task",
                "operation_id": origin_id,
                "artifact_id": source.artifact_id,
                "selections": [
                    {"kind": "text", "text": "first", "comment": "Change this to second."}
                ],
            },
        }
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=revision_id,
            project_id=project_id,
            kind="project_chat",
            status="running",
            request=revision_request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Running.",
            native_session_id="artifact-session",
        )
    )
    execution = AgentTaskExecution(
        operation_id=revision_id,
        store=store,
        control=AgentProcessControl(),
    )
    artifact_directory = tmp_path / "revision-artifacts"
    artifact_directory.mkdir()
    (artifact_directory / name).write_bytes(second)
    replacement = descriptor_for(revision_id, name, size_bytes=len(second))

    remaining = finalize_artifact_revision(
        service,
        revision_request,
        execution,
        artifact_scope_id=revision_id,
        artifact_directory=artifact_directory,
        remote_stage=None,
        artifacts=[replacement],
    )

    assert remaining == []
    assert service.history.workspace.read_kept_artifact(kept_filename) == second
    updated = store.agent_task(origin_id)
    assert updated is not None
    assert updated.result["artifacts"] == [
        source.model_copy(update={"size_bytes": len(second)}).model_dump(mode="json")
    ]
    assert any(
        receipt.category == "artifact_revised" for receipt in store.agent_task_receipts(revision_id)
    )
