from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rcp.artifacts import descriptor_for
from rcp.history.manager import canonical_fact_sources, iter_canonical_fact_bytes
from rcp.service import (
    CoachRequest,
    canonical_chat_backup_sources,
    iter_canonical_chat_transfer,
)
from rcp.storage import AgentTaskRecord, ResultViewRecord
from rcp.transfer import TransferArchiveActor, TransferArchiveAttribution
from rcp.transfer import project_files as project_files_module
from rcp.transfer.project_files import (
    TransferProjectFileCapture,
    capture_project_transfer_files,
)
from rcp.transport import state as state_module
from rcp.transport.state import SSHStateWorkspace

from .helpers import authorized_human, create_named_app


def _finished_project(manifest, tmp_path: Path):
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id
    actor = authorized_human(store)
    operation_id = str(uuid.uuid4())
    now = store.now()
    artifact_bytes = b"<!doctype html><p>kept artifact</p>"
    kept_filename = service.history.workspace.keep_artifact(
        source_name="result.html",
        project_name="Transfer fixture",
        data=artifact_bytes,
    )
    descriptor = descriptor_for(
        operation_id,
        "result.html",
        size_bytes=len(artifact_bytes),
    ).model_copy(update={"kept_filename": kept_filename, "kept_at": now})
    request = CoachRequest(
        message="Review this history.",
        provider="codex",
        run_on="laptop",
        session_id="native-paper-session",
    )
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="paper_coach",
            status="succeeded",
            request=request.model_dump(mode="json"),
            result={
                "messages": ["Complete."],
                "artifacts": [descriptor.model_dump(mode="json")],
            },
            created_at=now,
            updated_at=now,
            started_at=now,
            finished_at=now,
            status_message="Finished",
            authorized_by=actor,
            native_session_id="native-paper-session",
        )
    )
    view_bytes = b"<!doctype html><p>legacy kept view</p>"
    view_filename = service.history.workspace.keep_result_view(
        source_name="legacy.html",
        project_name="Transfer fixture",
        data=view_bytes,
    )
    store.create_result_view(
        ResultViewRecord(
            view_id="a" * 24,
            project_id=project_id,
            experiment_id="experiment-1",
            chat_id="chat-1",
            origin_operation_id=operation_id,
            latest_operation_id=operation_id,
            provider="codex",
            model="gpt-5.6-sol",
            reasoning="high",
            run_on="laptop",
            native_session_id="native-view-session",
            stage_host="laptop",
            stage_root="/source/stage",
            source_name="legacy.html",
            content_sha256=hashlib.sha256(view_bytes).hexdigest(),
            size_bytes=len(view_bytes),
            created_at=now,
            updated_at=now,
            expires_at=now,
            kept_filename=view_filename,
            kept_at=now,
        ),
        html=view_bytes,
    )
    attribution = TransferArchiveAttribution(
        archive_actor_id=str(uuid.uuid4()),
        source_actor=TransferArchiveActor.capture(actor),
    )
    records = store.export_project_transfer_records(
        project_id,
        attributions=(attribution,),
    )
    return service, records, artifact_bytes, kept_filename, view_bytes, view_filename


def _write_canonical_sources(service, operation_id: str) -> tuple[str, str]:
    research = service.history.workspace.root
    chat_id = str(uuid.uuid4())
    unknown_operation_id = str(uuid.uuid4())
    chat = service.chat_path(chat_id, chat_scope="project", node_id=None)
    chat.parent.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    records = (
        {
            "sessionId": chat_id,
            "nativeSessionId": "native-chat-session",
            "nodeId": None,
            "chatScope": "project",
            "provider": "codex",
            "model": "gpt-5.6-sol",
            "reasoning": "high",
            "executionMachine": "source-gpu",
            "timestamp": timestamp,
            "uuid": str(uuid.uuid4()),
            "operationId": operation_id,
            "type": "user",
            "role": "user",
            "text": "Inspect the attached result.",
            "attachments": [
                {
                    "attachment_id": str(uuid.uuid4()),
                    "name": "notes.txt",
                    "media_type": "text/plain",
                    "size": 12,
                    "expires_at": timestamp,
                }
            ],
        },
        {
            "sessionId": chat_id,
            "nativeSessionId": "native-chat-session",
            "nodeId": None,
            "chatScope": "project",
            "provider": "codex",
            "executionMachine": "source-gpu",
            "timestamp": timestamp,
            "uuid": str(uuid.uuid4()),
            "operationId": unknown_operation_id,
            "type": "assistant",
            "role": "assistant",
            "text": "Historical answer.",
            "graphUpdate": {
                "status": "none",
                "change_summary": [],
                "proposal_ids": [],
                "validation_messages": [],
                "correction_rounds": 0,
                "repairable": False,
            },
        },
    )
    chat.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    paper = research / "paper"
    paper.mkdir(exist_ok=True)
    (paper / "introduction.md").write_bytes(b"# Canonical introduction\n")
    facts = research / "facts" / "methods"
    facts.mkdir(parents=True, exist_ok=True)
    (facts / "protocol.bin").write_bytes(b"opaque fact bytes")
    return chat_id, unknown_operation_id


def test_project_file_capture_transforms_human_history_and_binds_kept_bytes(
    manifest,
    tmp_path: Path,
) -> None:
    service, records, artifact, artifact_name, view, view_name = _finished_project(
        manifest,
        tmp_path,
    )
    operation_id = records.tasks[0].operation_id
    chat_id, unknown_operation_id = _write_canonical_sources(service, operation_id)
    capture_root = tmp_path / "capture"

    capture = capture_project_transfer_files(service, records, capture_root)

    assert [entry.archive_path for entry in capture.entries] == [
        f"artifacts/{artifact_name}",
        f"chats/project-{chat_id}.jsonl",
        "facts/methods/protocol.bin",
        "paper/introduction.md",
        f"result-views/{view_name}",
    ]
    assert (capture_root / "paper/introduction.md").read_bytes() == b"# Canonical introduction\n"
    assert (capture_root / "facts/methods/protocol.bin").read_bytes() == b"opaque fact bytes"
    assert (capture_root / "artifacts" / artifact_name).read_bytes() == artifact
    assert (capture_root / "result-views" / view_name).read_bytes() == view
    transferred_chat = [
        json.loads(line)
        for line in (capture_root / f"chats/project-{chat_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert transferred_chat[0]["operationId"] == operation_id
    assert "operationId" not in transferred_chat[1]
    assert unknown_operation_id not in json.dumps(transferred_chat)
    assert "nativeSessionId" not in transferred_chat[0]
    assert "executionMachine" not in transferred_chat[0]
    assert transferred_chat[0]["attachments"][0]["name"] == "notes.txt"
    bound = capture.records.tasks[0].artifacts[0]
    assert bound.content_sha256 == hashlib.sha256(artifact).hexdigest()
    assert records.tasks[0].artifacts[0].content_sha256 is None
    assert len(capture.kept_result_views) == 1
    transferred_view = capture.kept_result_views[0]
    assert transferred_view.view_id == "a" * 24
    assert transferred_view.experiment_id == "experiment-1"
    assert transferred_view.chat_id == "chat-1"
    assert transferred_view.kept_filename == view_name
    assert transferred_view.content_sha256 == hashlib.sha256(view).hexdigest()
    serialized_view = transferred_view.model_dump(mode="json")
    assert {
        "run_on",
        "native_session_id",
        "stage_host",
        "stage_root",
    }.isdisjoint(serialized_view)

    invalid = capture.model_dump(mode="python")
    invalid["kept_result_views"][0]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        TransferProjectFileCapture.model_validate(invalid)


def test_project_file_capture_uses_the_remote_export_and_named_reader_seams(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, records, _artifact, _artifact_name, _view, _view_name = _finished_project(
        manifest,
        tmp_path,
    )
    _write_canonical_sources(service, records.tasks[0].operation_id)
    local = service.history.workspace

    remote = SSHStateWorkspace(
        tmp_path / "remote-cache" / ".research",
        "research.example",
        "/srv/project",
    )
    inventory = json.dumps(
        [
            {
                "name": entry.name,
                "kind": "directory" if entry.is_dir() else "file",
            }
            for entry in sorted(local.root.iterdir(), key=lambda path: path.name)
        ],
        separators=(",", ":"),
    ).encode()
    remote_calls: list[list[str]] = []
    artifact_reads = 0
    view_reads = 0

    def fake_ssh(arguments: list[str], **_kwargs):
        remote_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, inventory, b"")

    def fake_rsync(arguments: list[str], **_kwargs):
        shutil.copytree(local.root, Path(arguments[-1]), dirs_exist_ok=True)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_ssh_bytes(arguments: list[str], **_kwargs):
        nonlocal artifact_reads, view_reads
        if arguments[-3] == "artifacts":
            artifact_reads += 1
            data = local.read_kept_artifact(arguments[-2])
        else:
            view_reads += 1
            data = local.read_kept_result_view(arguments[-2])
        return subprocess.CompletedProcess(arguments, 0, data, b"")

    monkeypatch.setattr(remote, "_ssh", fake_ssh)
    monkeypatch.setattr(remote, "_ssh_bytes", fake_ssh_bytes)
    monkeypatch.setattr(state_module.subprocess, "run", fake_rsync)
    service.history.workspace = remote  # type: ignore[assignment]

    capture = capture_project_transfer_files(service, records, tmp_path / "remote-capture")

    assert capture.entries
    assert len(remote_calls) == 2
    assert artifact_reads == 2
    assert view_reads == 2


def test_project_file_capture_rejects_a_missing_remote_kept_file(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, records, *_unused = _finished_project(manifest, tmp_path)
    _write_canonical_sources(service, records.tasks[0].operation_id)
    local = service.history.workspace
    remote = SSHStateWorkspace(
        tmp_path / "remote-missing-cache" / ".research",
        "research.example",
        "/srv/project",
    )
    inventory = json.dumps(
        [
            {
                "name": entry.name,
                "kind": "directory" if entry.is_dir() else "file",
            }
            for entry in sorted(local.root.iterdir(), key=lambda path: path.name)
        ],
        separators=(",", ":"),
    ).encode()
    monkeypatch.setattr(
        remote,
        "_ssh",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0, inventory, b""),
    )
    monkeypatch.setattr(
        remote,
        "_ssh_bytes",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 44, b"", b""),
    )

    def fake_rsync(arguments: list[str], **_kwargs):
        shutil.copytree(local.root, Path(arguments[-1]), dirs_exist_ok=True)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(state_module.subprocess, "run", fake_rsync)
    service.history.workspace = remote
    capture_root = tmp_path / "remote-missing-capture"

    with pytest.raises(ValueError, match="did not stabilize"):
        capture_project_transfer_files(service, records, capture_root)

    assert not capture_root.exists()


@pytest.mark.parametrize("append_timing", ["before_open", "during_read"])
def test_project_chat_transfer_keeps_the_observed_complete_prefix(
    manifest,
    tmp_path: Path,
    append_timing: str,
) -> None:
    service, records, *_unused = _finished_project(manifest, tmp_path)
    operation_id = records.tasks[0].operation_id
    _write_canonical_sources(service, operation_id)
    source = canonical_chat_backup_sources(service.history.workspace.root)[0]
    appended = {
        "sessionId": source.path.stem.removeprefix("project-"),
        "nodeId": None,
        "chatScope": "project",
        "provider": "codex",
        "timestamp": datetime.now(UTC).isoformat(),
        "uuid": str(uuid.uuid4()),
        "operationId": operation_id,
        "type": "assistant",
        "role": "assistant",
        "text": "After the observed boundary.",
    }

    def append_record() -> None:
        with source.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(appended, separators=(",", ":")) + "\n")

    if append_timing == "before_open":
        append_record()
    transferred = iter_canonical_chat_transfer(
        source,
        operation_id_map={operation_id: operation_id},
    )
    first = next(transferred)
    if append_timing == "during_read":
        append_record()
    captured = first + b"".join(transferred)

    assert b"Historical answer." in captured
    assert b"After the observed boundary." not in captured


def test_project_fact_capture_rejects_a_parent_directory_symlink_swap(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, records, *_unused = _finished_project(manifest, tmp_path)
    _write_canonical_sources(service, records.tasks[0].operation_id)
    research = service.history.workspace.root
    facts = research / "facts"
    outside = tmp_path / "outside-facts"
    outside.mkdir()
    (outside / "secret.txt").write_text("must not transfer", encoding="utf-8")
    original_inventory = project_files_module.canonical_fact_sources

    def inventory_then_swap(root: Path):
        sources = original_inventory(root)
        facts.rename(research / "facts-before-swap")
        facts.symlink_to(outside, target_is_directory=True)
        return sources

    monkeypatch.setattr(project_files_module, "canonical_fact_sources", inventory_then_swap)
    capture_root = tmp_path / "fact-race-capture"

    with pytest.raises(ValueError, match="fact"):
        capture_project_transfer_files(service, records, capture_root)

    assert not capture_root.exists()


def test_canonical_fact_reader_rejects_a_replaced_root(tmp_path: Path) -> None:
    research = tmp_path / ".research"
    facts = research / "facts"
    facts.mkdir(parents=True)
    (facts / "inside.bin").write_bytes(b"inside")
    source = canonical_fact_sources(research)[0]
    facts.rename(research / "old-facts")
    facts.mkdir()
    (facts / "inside.bin").write_bytes(b"outside")

    with pytest.raises(ValueError, match="facts directory changed"):
        b"".join(iter_canonical_fact_bytes(research, source, chunk_size=16))


@pytest.mark.parametrize("failure", ["unknown_paper", "unsafe_fact", "missing_artifact"])
def test_project_file_capture_fails_closed_without_a_partial_root(
    manifest,
    tmp_path: Path,
    failure: str,
) -> None:
    service, records, _artifact, artifact_name, _view, _view_name = _finished_project(
        manifest,
        tmp_path,
    )
    _write_canonical_sources(service, records.tasks[0].operation_id)
    research = service.history.workspace.root
    if failure == "unknown_paper":
        (research / "paper/notes.md").write_text("not canonical", encoding="utf-8")
    elif failure == "unsafe_fact":
        (research / "facts/unsafe-link").symlink_to(tmp_path / "outside")
    else:
        (research.parent / "artifacts" / artifact_name).unlink()
    capture_root = tmp_path / f"failed-{failure}"

    with pytest.raises((FileNotFoundError, ValueError)):
        capture_project_transfer_files(service, records, capture_root)

    assert not capture_root.exists()
