from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import rcp.sources.indexer as source_indexer
from rcp.agents import ContextAssembler
from rcp.config import MachineConfig
from rcp.core.models import GraphState
from rcp.history import HistoryManager
from rcp.paper import PaperService
from rcp.service import ProjectService, RunRequest
from rcp.sources import ConversationIndexer, record_parsing
from rcp.sources.indexer import (
    _REMOTE_INDEX_SCRIPT,
    _REMOTE_SLICE_SCRIPT,
    ConversationRecord,
    _normalize_record,
)
from rcp.storage import AppStore
from rcp.transport import SSHStateWorkspace


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_matching_local_histories_are_fully_normalized(manifest, monkeypatch, provider) -> None:
    root = Path(
        next(
            iter(
                manifest.sources.codex_roots
                if provider == "codex"
                else manifest.sources.claude_roots
            )
        )
    )
    repository = manifest.repository_map["repo-a"].path
    if provider == "codex":
        records = [
            {
                "type": "session_meta",
                "payload": {"id": "matching", "cwd": repository},
            },
            *[
                {
                    "type": "response_item",
                    "payload": {
                        "id": f"record-{index}",
                        "type": "message",
                        "role": "assistant",
                    },
                }
                for index in range(1, 6)
            ],
        ]
    else:
        records = [
            {
                "type": "user",
                "uuid": "record-0",
                "sessionId": "matching",
                "cwd": repository,
                "message": {"content": "question"},
            },
            *[
                {
                    "type": "assistant",
                    "uuid": f"record-{index}",
                    "sessionId": "matching",
                    "message": {"content": "answer"},
                }
                for index in range(1, 6)
            ],
        ]
    source = root / f"matching-{provider}.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    normalized: list[int] = []
    original_normalize = source_indexer._normalize_record

    def track_normalization(raw, inspected_provider, line_number):
        normalized.append(line_number)
        return original_normalize(raw, inspected_provider, line_number)

    monkeypatch.setattr(source_indexer, "_normalize_record", track_normalization)

    session = ConversationIndexer(manifest).build().sessions[0]

    assert normalized == list(range(1, len(records) + 1))
    assert session.record_count == len(records)
    assert session.local_source_identity is not None
    stat = source.stat()
    assert session.local_source_identity.model_dump() == {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def test_second_index_build_reuses_unchanged_matching_and_unmatched_metadata(
    manifest, monkeypatch
) -> None:
    root = Path(next(iter(manifest.sources.codex_roots)))
    matching = root / "matching.jsonl"
    unmatched = root / "unmatched.jsonl"
    matching.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "matching",
                    "cwd": manifest.repository_map["repo-a"].path,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    unmatched.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "unmatched", "cwd": "/outside/project"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    indexer = ConversationIndexer(manifest)
    first = indexer.build()
    original_open = Path.open

    def reject_source_open(path, *args, **kwargs):
        if path in {matching, unmatched}:
            raise AssertionError("unchanged indexed source must not be reopened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_source_open)

    second = indexer.build()

    assert first.model_dump(exclude={"generated_at"}) == second.model_dump(exclude={"generated_at"})
    assert second.unmatched_files == 1
    assert [session.session_id for session in second.sessions] == ["matching"]


def test_changed_source_reparses_only_that_file(manifest, monkeypatch) -> None:
    root = Path(next(iter(manifest.sources.codex_roots)))
    repository = manifest.repository_map["repo-a"].path

    def write_source(path: Path, session_id: str) -> None:
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "session_meta",
                            "payload": {"id": session_id, "cwd": repository},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "id": f"{session_id}-terminal",
                                "type": "message",
                                "role": "assistant",
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    changed = root / "changed.jsonl"
    unchanged = root / "unchanged.jsonl"
    write_source(changed, "changed")
    write_source(unchanged, "unchanged")
    indexer = ConversationIndexer(manifest)
    indexer.build()
    with changed.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "id": "changed-new-terminal",
                        "type": "message",
                        "role": "assistant",
                    },
                }
            )
            + "\n"
        )
    inspected: list[Path] = []
    original_inspect = indexer._inspect

    def track_inspection(path, provider, repository_paths=None):
        inspected.append(path)
        return original_inspect(path, provider, repository_paths)

    monkeypatch.setattr(indexer, "_inspect", track_inspection)

    second = indexer.build()

    assert inspected == [changed]
    by_id = {session.session_id: session for session in second.sessions}
    assert by_id["changed"].record_count == 3
    assert by_id["changed"].last_uuid == "changed-new-terminal"
    assert by_id["unchanged"].record_count == 2


def test_deleted_source_is_evicted_from_metadata_cache(manifest) -> None:
    source = Path(next(iter(manifest.sources.codex_roots))) / "deleted.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "deleted",
                    "cwd": manifest.repository_map["repo-a"].path,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    indexer = ConversationIndexer(manifest)
    assert [session.session_id for session in indexer.build().sessions] == ["deleted"]
    assert len(indexer._local_metadata_cache) == 1

    source.unlink()
    second = indexer.build()

    assert second.sessions == []
    assert indexer._local_metadata_cache == {}


def test_unchanged_terminal_cursor_does_not_reopen_local_source(manifest, monkeypatch) -> None:
    source = Path(next(iter(manifest.sources.codex_roots))) / "terminal.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "terminal",
                    "cwd": manifest.repository_map["repo-a"].path,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    indexer = ConversationIndexer(manifest)
    session = indexer.build().sessions[0]
    original_open = Path.open

    def reject_source_open(path, *args, **kwargs):
        if path == source:
            raise AssertionError("unchanged terminal source must not be reopened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_source_open)

    assert list(indexer.read_records(session, from_uuid=session.last_uuid)) == []


def test_changed_local_source_identity_disables_terminal_cursor_shortcut(
    manifest, monkeypatch
) -> None:
    source = Path(next(iter(manifest.sources.codex_roots))) / "changed-terminal.jsonl"
    records = [
        {
            "type": "session_meta",
            "payload": {
                "id": "changed-terminal",
                "cwd": manifest.repository_map["repo-a"].path,
            },
        },
        {
            "type": "response_item",
            "payload": {
                "id": "terminal",
                "type": "message",
                "role": "assistant",
            },
        },
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    indexer = ConversationIndexer(manifest)
    session = indexer.build().sessions[0]
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    normalized: list[int] = []
    original_normalize = source_indexer._normalize_record

    def track_normalization(raw, inspected_provider, line_number):
        normalized.append(line_number)
        return original_normalize(raw, inspected_provider, line_number)

    monkeypatch.setattr(source_indexer, "_normalize_record", track_normalization)

    assert list(indexer.read_records(session, from_uuid=session.last_uuid)) == []
    assert normalized == [1, 2]


def test_indexes_codex_and_claude_by_embedded_cwd(manifest) -> None:
    repo = manifest.repository_map["repo-a"].path
    codex_path = next(iter(manifest.sources.codex_roots))
    claude_path = next(iter(manifest.sources.claude_roots))
    with open(f"{codex_path}/codex.jsonl", "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "codex-session",
                        "cwd": repo,
                        "timestamp": "2026-07-27T00:00:00Z",
                        "thread_source": "subagent",
                        "originator": "Codex Desktop",
                        "source": {
                            "subagent": {"thread_spawn": {"parent_thread_id": "parent-session"}}
                        },
                    },
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "id": "record-2",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                }
            )
            + "\n"
        )
    with open(f"{claude_path}/claude.jsonl", "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "uuid": "claude-record",
                    "sessionId": "claude-session",
                    "cwd": repo,
                    "timestamp": "2026-07-27T00:00:00Z",
                    "type": "user",
                    "message": {"content": "hello"},
                }
            )
            + "\n"
        )

    indexer = ConversationIndexer(manifest)
    index = indexer.build()

    assert {(session.provider, session.session_id) for session in index.sessions} == {
        ("codex", "codex-session"),
        ("claude", "claude-session"),
    }
    assert {session.truth_repository for session in index.sessions} == {"repo-a"}
    codex = next(session for session in index.sessions if session.provider == "codex")
    claude = next(session for session in index.sessions if session.provider == "claude")
    assert codex.thread_source == "subagent"
    assert codex.parent_session_id == "parent-session"
    assert codex.originator == "Codex Desktop"
    assert codex.source_kind == "subagent"
    assert claude.thread_source == "user"
    assert claude.source_kind == "claude"
    records = list(indexer.read_records(codex, from_uuid="record-2"))
    assert records == []


def test_codex_custom_tool_input_is_readable(manifest) -> None:
    repo = manifest.repository_map["repo-a"].path
    path = Path(next(iter(manifest.sources.codex_roots))) / "artifact-session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "artifact-session",
                            "cwd": repo,
                            "timestamp": "2026-07-27T00:00:00Z",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "id": "artifact-call",
                            "type": "custom_tool_call",
                            "input": "reviewed recap artifact body",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    indexer = ConversationIndexer(manifest)
    session = next(
        item for item in indexer.build().sessions if item.session_id == "artifact-session"
    )

    records = list(indexer.read_records(session))

    assert records[-1].role == "assistant"
    assert records[-1].text == "reviewed recap artifact body"


def test_app_chat_sessions_are_normalized_as_human_led_roots(manifest) -> None:
    chat_root = manifest.research_dir / "chat"
    chat_root.mkdir()
    chat_path = chat_root / "human-chat.jsonl"
    chat_path.write_text(
        json.dumps(
            {
                "uuid": "chat-record",
                "sessionId": "human-chat",
                "cwd": manifest.repository_map["repo-a"].path,
                "timestamp": "2026-07-27T00:00:00Z",
                "type": "user",
                "role": "user",
                "text": "human correction",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    session = next(
        item
        for item in ConversationIndexer(manifest).build().sessions
        if item.provider == "app_chat"
    )

    assert session.thread_source == "user"
    assert session.source_kind == "app_chat"
    assert session.source_machine == "laptop"
    assert session.path == str(chat_path)
    assert session.remote_source_host is None
    assert session.remote_source_path is None


def test_chat_context_ignores_remote_indexed_app_chat(manifest, tmp_path, monkeypatch) -> None:
    manifest.machines.append(MachineConfig(alias="remote-1", host="research.example"))
    state_repository = manifest.repositories[0]
    state_repository.machine = "remote-1"
    state_repository.path = "/remote/project"
    chat_root = manifest.research_dir / "chat"
    chat_root.mkdir()
    chat_path = chat_root / "remote-chat.jsonl"
    chat_path.write_text(
        json.dumps(
            {
                "uuid": "chat-record",
                "sessionId": "remote-chat",
                "cwd": "/remote/project",
                "timestamp": "2026-07-27T00:00:00Z",
                "type": "user",
                "role": "user",
                "text": "remote canonical conversation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workspace = SSHStateWorkspace(
        manifest.research_dir,
        "research.example",
        "/remote/project",
    )
    history = HistoryManager(manifest, workspace)
    service = ProjectService(
        manifest,
        history,
        PaperService(
            manifest,
            AppStore(tmp_path / "app.sqlite3"),
            workspace,
            project_id="project",
        ),
        data_dir=tmp_path / "data",
    )
    monkeypatch.setattr(
        service.indexer,
        "_inspect_remote_root",
        lambda *_args: source_indexer.RemoteConversationIndex(
            sessions=(), unmatched_files=0, malformed_files=0
        ),
    )

    index = service.index_snapshot(execution_machine="remote-1")
    session = next(item for item in index.sessions if item.provider == "app_chat")

    assert session.key == "repo-a/remote-1/app_chat/remote-chat"
    assert session.source_machine == "remote-1"
    assert session.path == str(chat_path)
    assert session.remote_source_host == "research.example"
    assert session.remote_source_path == "/remote/project/.research/chat/remote-chat.jsonl"

    assembler = ContextAssembler(manifest)
    state = GraphState(project_truth_scope=manifest.project.truth_scope)
    on_owner = assembler.chat_context(
        state,
        run_truth_scope=["repo-a"],
    )
    from_laptop = assembler.chat_context(
        state,
        run_truth_scope=["repo-a"],
    )

    assert "conversations" not in on_owner.model_dump()
    assert "conversations" not in from_laptop.model_dump()


def test_chat_context_does_not_include_indexed_transcripts(manifest, tmp_path) -> None:
    """Chat stays independent even when provider sources are indexed."""

    root = Path(next(iter(manifest.sources.codex_roots)))
    for alias in ("repo-a", "repo-b"):
        (root / f"{alias}.jsonl").write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": f"session-{alias}",
                        "cwd": manifest.repository_map[alias].path,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    indexer = ConversationIndexer(manifest, tmp_path / "source-cache")

    index = indexer.build()
    assert index.sessions
    context = ContextAssembler(manifest).chat_context(
        GraphState(project_truth_scope=manifest.project.truth_scope),
        run_truth_scope=["repo-a"],
    )

    assert "conversations" not in context.model_dump()
    assert str(root / "repo-a.jsonl") not in str(context.model_dump())


def test_chat_context_does_not_resolve_remote_transcript_pointers(
    manifest, tmp_path, monkeypatch
) -> None:
    """Remote source indexing cannot add transcript inputs to chat."""

    manifest.machines.append(MachineConfig(alias="remote-1", host="research.example"))
    manifest.repositories[0].machine = "remote-1"
    manifest.repositories[0].path = "/remote/project"
    local_root = Path(next(iter(manifest.sources.codex_roots)))
    (local_root / "repo-b.jsonl").write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "session-repo-b",
                    "cwd": manifest.repository_map["repo-b"].path,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    indexer = ConversationIndexer(manifest, tmp_path / "cache")
    monkeypatch.setattr(
        indexer,
        "_inspect_remote_root",
        lambda _host, _root, provider, _machine: source_indexer.RemoteConversationIndex(
            sessions=(
                {
                    "path": f"/remote/sessions/{provider}.jsonl",
                    "cwd": "/remote/project",
                    "session_id": f"{provider}-session",
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "last_uuid": "last",
                    "record_count": 2,
                },
            ),
            unmatched_files=0,
            malformed_files=0,
        ),
    )

    def cache_remote_files(
        _host: str,
        _machine: str,
        provider: str,
        paths: list[str],
        *,
        pin_artifact=None,
    ) -> dict[str, Path]:
        assert pin_artifact is None
        return {path: tmp_path / "cache" / f"{provider}.jsonl" for path in paths}

    monkeypatch.setattr(
        indexer,
        "_cache_remote_files",
        cache_remote_files,
    )
    index = indexer.build()
    assert index.sessions
    state = GraphState(project_truth_scope=manifest.project.truth_scope)
    context = ContextAssembler(manifest).chat_context(state, run_truth_scope=["repo-a"])

    assert "conversations" not in context.model_dump()


def test_out_of_scope_run_is_rejected_before_index(manifest, tmp_path, monkeypatch) -> None:
    history = HistoryManager(manifest)
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )
    monkeypatch.setattr(
        service.indexer,
        "build",
        lambda **_kwargs: pytest.fail("source indexing must follow scope validation"),
    )

    with pytest.raises(ValueError, match="non-empty subset"):
        service.assemble_run(
            RunRequest(run_truth_scope=["outside-project"]),
            surface="refresh",
        )


def test_remote_index_script_filters_by_embedded_cwd(tmp_path) -> None:
    root = tmp_path / "remote-sessions"
    root.mkdir()
    for name, cwd in (
        ("matching", "/remote/project"),
        ("legacy-subagent", "/remote/project"),
        ("other", "/remote/other"),
    ):
        payload = {
            "id": name,
            "cwd": cwd,
            "timestamp": "2026-07-27T00:00:00Z",
        }
        if name == "matching":
            payload.update(
                {
                    "thread_source": "subagent",
                    "originator": "Codex Desktop",
                    "source": {
                        "subagent": {"thread_spawn": {"parent_thread_id": "parent-session"}}
                    },
                }
            )
        elif name == "legacy-subagent":
            payload.update(
                {
                    "thread_source": "subagent",
                    "source": {"subagent": "legacy"},
                }
            )
        (root / f"{name}.jsonl").write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": payload,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (root / "malformed.jsonl").write_text("{not json}\n", encoding="utf-8")
    payload = json.dumps(
        {
            "root": str(root),
            "provider": "codex",
            "repository_paths": ["/remote/project"],
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", _REMOTE_INDEX_SCRIPT, payload],
        capture_output=True,
        text=True,
        check=True,
    )

    records = [
        record
        for line in result.stdout.splitlines()
        if (record := json.loads(line))["kind"] == "session"
    ]
    by_session = {record["session_id"]: record for record in records}
    assert set(by_session) == {"matching", "legacy-subagent"}
    assert by_session["matching"]["thread_source"] == "subagent"
    assert by_session["matching"]["parent_session_id"] == "parent-session"
    assert by_session["matching"]["originator"] == "Codex Desktop"
    assert by_session["matching"]["source_kind"] == "subagent"
    assert by_session["legacy-subagent"]["source_kind"] == "subagent"
    assert by_session["legacy-subagent"]["parent_session_id"] is None
    summary = next(
        json.loads(line)
        for line in result.stdout.splitlines()
        if json.loads(line)["kind"] == "summary"
    )
    assert summary == {"kind": "summary", "unmatched_files": 1, "malformed_files": 1}


def test_remote_index_counts_claude_records_before_first_cwd(tmp_path) -> None:
    root = tmp_path / "remote-sessions"
    root.mkdir()
    source = root / "claude-session.jsonl"
    records = [
        {
            "type": "queue-operation",
            "uuid": "queued",
            "timestamp": "2026-07-27T00:00:00Z",
        },
        {
            "type": "ai-title",
            "uuid": "title",
            "sessionId": "claude-session",
            "timestamp": "2026-07-27T00:00:01Z",
        },
        {
            "type": "user",
            "uuid": "question",
            "sessionId": "claude-session",
            "cwd": "/remote/project",
            "timestamp": "2026-07-27T00:00:02Z",
            "message": {"content": "question"},
        },
        {
            "type": "assistant",
            "uuid": "terminal",
            "sessionId": "claude-session",
            "timestamp": "2026-07-27T00:00:03Z",
            "message": {"content": "answer"},
        },
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    index_payload = json.dumps(
        {
            "root": str(root),
            "provider": "claude",
            "repository_paths": ["/remote/project"],
        }
    )

    indexed = subprocess.run(
        [sys.executable, "-c", _REMOTE_INDEX_SCRIPT, index_payload],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata, summary = [json.loads(line) for line in indexed.stdout.splitlines()]

    assert metadata["session_id"] == "claude-session"
    assert metadata["first_timestamp"] == "2026-07-27T00:00:00Z"
    assert metadata["last_timestamp"] == "2026-07-27T00:00:03Z"
    assert metadata["record_count"] == len(records)
    assert metadata["last_uuid"] == "terminal"
    assert summary == {"kind": "summary", "unmatched_files": 0, "malformed_files": 0}

    slice_payload = json.dumps(
        {
            "path": str(source),
            "provider": "claude",
            "record_count": metadata["record_count"],
            "last_uuid": metadata["last_uuid"],
            "from_uuid": None,
            "session_key": "repo/remote-1/claude/claude-session",
        }
    )
    sliced = subprocess.run(
        [sys.executable, "-c", _REMOTE_SLICE_SCRIPT, slice_payload],
        capture_output=True,
        text=True,
        check=True,
    )

    assert [json.loads(line)["uuid"] for line in sliced.stdout.splitlines()] == [
        "queued",
        "title",
        "question",
        "terminal",
    ]


def test_remote_execution_keeps_same_machine_sources_out_of_permanent_cache(
    manifest, tmp_path, monkeypatch
) -> None:
    manifest.machines.append(MachineConfig(alias="remote-1", host="research.example"))
    manifest.repositories[0].machine = "remote-1"
    manifest.repositories[0].path = "/remote/project"
    indexer = ConversationIndexer(manifest, tmp_path / "cache")

    def inspect_remote(_host, _root, provider, _machine_alias):
        return source_indexer.RemoteConversationIndex(
            sessions=(
                {
                    "path": f"/remote/sessions/{provider}.jsonl",
                    "cwd": "/remote/project",
                    "session_id": f"{provider}-session",
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "last_uuid": "last",
                    "record_count": 2,
                },
            ),
            unmatched_files=0,
            malformed_files=0,
        )

    monkeypatch.setattr(indexer, "_inspect_remote_root", inspect_remote)
    cache_calls: list[list[str]] = []

    def cache_remote(_host, _machine, provider, paths, *, pin_artifact=None):
        assert pin_artifact is None
        cache_calls.append(paths)
        return {path: tmp_path / "cache" / f"{provider}.jsonl" for path in paths}

    monkeypatch.setattr(indexer, "_cache_remote_files", cache_remote)

    result = indexer.build(execution_machine="remote-1")

    assert {session.path for session in result.sessions} == {
        "/remote/sessions/claude.jsonl",
        "/remote/sessions/codex.jsonl",
    }
    assert all(session.source_path_is_remote for session in result.sessions)
    assert all(session.remote_source_host == "research.example" for session in result.sessions)
    assert cache_calls == []


def test_shared_record_parser_imports_only_the_standard_library() -> None:
    """The remote host has no virtualenv and no `rcp` package.

    `record_parsing.py` is shipped as source text and run with `python3 -c`
    there, so anything it imports must already exist on a bare interpreter.
    """

    allowed = {"__future__", "hashlib", "json", "posixpath", "typing"}
    source_path = Path(record_parsing.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the shipped parser cannot use relative imports"
            assert node.module is not None
            imported.add(node.module.partition(".")[0])

    assert imported <= allowed, f"non-stdlib imports in {source_path}: {sorted(imported - allowed)}"


@pytest.mark.parametrize(
    ("provider", "records"),
    [
        (
            "codex",
            [
                {
                    "type": "response_item",
                    "payload": {
                        "id": "first",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "question"}],
                        "timestamp": "2026-07-27T00:00:00Z",
                    },
                },
                # A payload that is not a dict at all; the parser must not crash.
                {"type": "response_item", "payload": ["unexpected", "shape"]},
                # No id anywhere, so the line-digest fallback id has to run.
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "message": "answer",
                        "timestamp": "2026-07-27T00:00:01Z",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "id": "terminal",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                        "timestamp": "2026-07-27T00:00:02Z",
                    },
                },
            ],
        ),
        (
            "claude",
            [
                {
                    "type": "user",
                    "uuid": "first",
                    "timestamp": "2026-07-27T00:00:00Z",
                    "message": {"content": [{"type": "text", "text": "question"}]},
                },
                # No uuid, so the line-digest fallback id has to run.
                {"type": "assistant", "message": {"content": "plain string content"}},
                {
                    "type": "assistant",
                    "uuid": "terminal",
                    "timestamp": "2026-07-27T00:00:02Z",
                    "message": {"content": [{"type": "text", "text": "answer"}]},
                },
            ],
        ),
    ],
)
def test_remote_slice_program_normalizes_exactly_like_the_local_path(
    tmp_path, provider, records
) -> None:
    """The shipped program and the in-process path share one parser.

    Both used to be hand-maintained copies, which is how the local one lost the
    non-dict `payload` guard the remote one had.
    """

    source = tmp_path / "conversation.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    expected = [
        _normalize_record(record, provider, line_number)
        for line_number, record in enumerate(records, start=1)
    ]
    payload = json.dumps(
        {
            "path": str(source),
            "provider": provider,
            "record_count": len(records),
            "last_uuid": expected[-1].uuid,
            "from_uuid": None,
            "session_key": f"repo/remote/{provider}/session",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", _REMOTE_SLICE_SCRIPT, payload],
        capture_output=True,
        text=True,
        check=True,
    )

    emitted = [ConversationRecord.model_validate_json(line) for line in result.stdout.splitlines()]
    assert emitted == expected
    assert [record.uuid for record in emitted if record.uuid.startswith("line-")]
