from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from rcp.config import MachineConfig
from rcp.providers import PROVIDERS, ProviderProfile
from rcp.sources import ConversationIndexer, OriginalConversationSource
from rcp.sources import indexer as indexer_module
from rcp.transfer import provider_history as provider_history_module
from rcp.transfer.provider_history import capture_provider_history


def _codex_bytes(session_id: str, cwd: str, text: str) -> bytes:
    records = (
        {
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": cwd},
        },
        {
            "type": "response_item",
            "payload": {
                "id": f"{session_id}-answer",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        },
    )
    return b"".join(
        json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records
    )


def _claude_bytes(session_id: str, cwd: str, text: str) -> bytes:
    records = (
        {
            "type": "user",
            "uuid": f"{session_id}-question",
            "sessionId": session_id,
            "cwd": cwd,
            "message": {"content": "question"},
        },
        {
            "type": "assistant",
            "uuid": f"{session_id}-answer",
            "sessionId": session_id,
            "message": {"content": text},
        },
    )
    return b"".join(
        json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records
    )


def test_provider_history_capture_preserves_matching_originals_and_summarizes_omissions(
    manifest,
    tmp_path: Path,
) -> None:
    codex_root = Path(manifest.sources.codex_roots[0])
    claude_root = Path(manifest.sources.claude_roots[0])
    manifest.project.truth_scope = ["repo-a"]
    codex = _codex_bytes("codex-match", manifest.repository_map["repo-a"].path, "codex")
    claude = _claude_bytes("claude-match", manifest.repository_map["repo-b"].path, "claude")
    (codex_root / "matching.jsonl").write_bytes(codex)
    (claude_root / "matching.jsonl").write_bytes(claude)
    (codex_root / "unmatched.jsonl").write_bytes(
        _codex_bytes("outside", "/outside/project", "outside")
    )
    (claude_root / "malformed.jsonl").write_text("{not json}\n", encoding="utf-8")
    chat_root = manifest.research_dir / "chat"
    chat_root.mkdir()
    (chat_root / "project-chat.jsonl").write_text(
        json.dumps(
            {
                "uuid": "chat-record",
                "sessionId": "project-chat",
                "type": "user",
                "role": "user",
                "text": "already captured by T3b-files",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    capture_root = tmp_path / "provider-history"
    capture = capture_provider_history(ConversationIndexer(manifest), capture_root)

    expected = {
        f"provider-history/codex/{hashlib.sha256(codex).hexdigest()}": codex,
        f"provider-history/claude/{hashlib.sha256(claude).hexdigest()}": claude,
    }
    assert {entry.archive_path for entry in capture.entries} == set(expected)
    for path, content in expected.items():
        assert (capture_root / path).read_bytes() == content
    assert capture.selected_files == 2
    assert capture.skipped_files == 2
    assert capture.unreadable_files == 0
    assert [item.code for item in capture.diagnostics] == [
        "provider_history_malformed",
        "provider_history_unmatched",
    ]
    assert all("app_chat" not in entry.archive_path for entry in capture.entries)


def test_provider_history_capture_rechecks_remote_originals_on_the_exact_ssh_account(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = "alice@research.example"
    manifest.machines.append(MachineConfig(alias="remote-1", host=host))
    manifest.repositories[0].machine = "remote-1"
    manifest.repositories[0].path = "/remote/repo-a"
    manifest.sources.codex_roots = []
    manifest.sources.claude_roots = []
    remote_files = {
        "/remote/codex/valid.jsonl": _codex_bytes(
            "remote-codex",
            "/remote/repo-a",
            "remote codex",
        ),
        "/remote/claude/rewritten.jsonl": _claude_bytes(
            "remote-claude",
            "/remote/other-project",
            "rewritten claude",
        ),
    }
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_kwargs):
        calls.append(arguments)
        if arguments[0] == "ssh":
            provider = json.loads(shlex.split(arguments[-1])[-1])["provider"]
            remote_path = next(path for path in remote_files if f"/{provider}/" in path)
            item = {
                "path": remote_path,
                "cwd": "/remote/repo-a",
                "session_id": f"remote-{provider}",
                "first_timestamp": None,
                "last_timestamp": None,
                "last_uuid": f"remote-{provider}-answer",
                "record_count": 2,
                "thread_source": "user",
                "parent_session_id": None,
                "originator": None,
                "source_kind": provider,
            }
            records = (
                {"kind": "session", **item},
                {
                    "kind": "summary",
                    "unmatched_files": int(provider == "codex"),
                    "malformed_files": int(provider == "claude"),
                },
            )
            return subprocess.CompletedProcess(
                arguments,
                0,
                "".join(json.dumps(record) + "\n" for record in records),
                "",
            )
        assert arguments[0] == "rsync"
        remote_path = next(
            path for path in remote_files if any(path in argument for argument in arguments)
        )
        if "-aR" in arguments:
            destination = Path(arguments[-1]) / remote_path.lstrip("/")
        else:
            destination = Path(arguments[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(remote_files[remote_path])
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(indexer_module.subprocess, "run", fake_run)

    capture_root = tmp_path / "remote-provider-history"
    capture = capture_provider_history(
        ConversationIndexer(manifest, tmp_path / "source-cache"),
        capture_root,
    )

    valid = remote_files["/remote/codex/valid.jsonl"]
    assert [entry.archive_path for entry in capture.entries] == [
        f"provider-history/codex/{hashlib.sha256(valid).hexdigest()}"
    ]
    assert capture.selected_files == 1
    assert capture.skipped_files == 3
    assert capture.unreadable_files == 0
    assert [item.code for item in capture.diagnostics] == [
        "provider_history_malformed",
        "provider_history_rewritten",
        "provider_history_unmatched",
    ]
    assert len([call for call in calls if call[0] == "ssh"]) == 2
    assert len([call for call in calls if call[0] == "rsync"]) == 2
    assert all(any(host in argument for argument in call) for call in calls)


def test_provider_history_capture_keeps_going_when_one_selected_file_disappears(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(manifest.sources.codex_roots[0])
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    first.write_bytes(_codex_bytes("first", manifest.repository_map["repo-a"].path, "first"))
    second.write_bytes(_codex_bytes("second", manifest.repository_map["repo-a"].path, "second"))
    indexer = ConversationIndexer(manifest)
    original_build = indexer.build

    def build_then_remove(**kwargs):
        index = original_build(**kwargs)
        second.unlink()
        return index

    monkeypatch.setattr(indexer, "build", build_then_remove)

    capture = capture_provider_history(indexer, tmp_path / "partial-provider-history")

    assert capture.selected_files == 1
    assert capture.skipped_files == 0
    assert capture.unreadable_files == 1
    assert [item.code for item in capture.diagnostics] == ["provider_history_unreadable"]


def test_provider_history_selection_uses_the_provider_root_registry(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixture-provider"
    fixture_root.mkdir()
    source = fixture_root / "fixture.jsonl"
    source.write_text(
        json.dumps(
            {
                "uuid": "fixture-record",
                "sessionId": "fixture-session",
                "cwd": manifest.repository_map["repo-a"].path,
                "type": "assistant",
                "role": "assistant",
                "text": "fixture provider history",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FixtureProvider(ProviderProfile):
        id = "fixture"

        def session_roots(self, _sources: object, *, remote: bool) -> list[str]:
            return [] if remote else [str(fixture_root)]

    monkeypatch.setitem(PROVIDERS, "fixture", FixtureProvider())

    sessions = ConversationIndexer(manifest).build().sessions

    fixture = next(item for item in sessions if item.provider == "fixture")
    assert fixture.path == str(source)
    assert fixture.truth_repository == "repo-a"


def test_provider_history_rewritten_copy_is_omitted_without_partial_bytes(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer = ConversationIndexer(manifest)
    source = tmp_path / "rewritten.jsonl"
    data = _codex_bytes("rewritten", "/outside/project", "outside")
    source.write_bytes(data)
    session_source = Path(manifest.sources.codex_roots[0]) / "indexed.jsonl"
    session_source.write_bytes(
        _codex_bytes("rewritten", manifest.repository_map["repo-a"].path, "inside")
    )
    indexed = indexer.build()
    session = next(item for item in indexed.sessions if item.session_id == "rewritten")
    monkeypatch.setattr(indexer, "build", lambda **_kwargs: indexed)

    @contextmanager
    def rewritten_original(_session):
        yield OriginalConversationSource(
            path=source,
            content_sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    monkeypatch.setattr(indexer, "original_source", rewritten_original)
    capture_root = tmp_path / "rewritten-provider-history"

    capture = capture_provider_history(indexer, capture_root)

    assert capture.entries == ()
    assert capture.selected_files == 0
    assert capture.skipped_files == 1
    assert capture.unreadable_files == 0
    assert not any(path.is_file() for path in capture_root.rglob("*"))
    assert session.truth_repository == "repo-a"


def test_provider_history_counts_distinct_files_with_identical_bytes(
    manifest,
    tmp_path: Path,
) -> None:
    root = Path(manifest.sources.codex_roots[0])
    content = _codex_bytes("duplicate", manifest.repository_map["repo-a"].path, "same")
    (root / "first.jsonl").write_bytes(content)
    (root / "second.jsonl").write_bytes(content)

    capture = capture_provider_history(
        ConversationIndexer(manifest),
        tmp_path / "duplicate-provider-history",
    )

    assert len(capture.entries) == 1
    assert capture.selected_files == 2


def test_provider_history_capture_storage_failure_discards_the_whole_root(
    manifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(manifest.sources.codex_roots[0])
    root.joinpath("selected.jsonl").write_bytes(
        _codex_bytes("selected", manifest.repository_map["repo-a"].path, "selected")
    )
    capture_root = tmp_path / "failed-provider-history"

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("destination fsync failed")

    monkeypatch.setattr(provider_history_module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(OSError, match="destination fsync failed"):
        capture_provider_history(ConversationIndexer(manifest), capture_root)

    assert not capture_root.exists()
