from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from rcp.agents import ContextAssembler
from rcp.config import MachineConfig
from rcp.history import HistoryManager
from rcp.limits import REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS
from rcp.paper import PaperService
from rcp.service import ProjectService, RunRequest
from rcp.sources import preflight_provider_roots
from rcp.storage import AppStore


def test_local_preflight_checks_metadata_without_enumerating(manifest, monkeypatch) -> None:
    def forbid_enumeration(*_args, **_kwargs):
        raise AssertionError("provider roots must not be enumerated")

    monkeypatch.setattr(Path, "iterdir", forbid_enumeration)
    monkeypatch.setattr(os, "scandir", forbid_enumeration)

    roots = ContextAssembler(manifest).source_roots("laptop")
    assert preflight_provider_roots(roots, manifest.machine_map["laptop"]) == []


def test_local_preflight_reports_each_invalid_agent_visible_root(manifest, tmp_path) -> None:
    not_a_directory = tmp_path / "provider-log-file"
    not_a_directory.write_text("must not be read", encoding="utf-8")
    missing = tmp_path / "missing-provider-root"
    manifest.sources.claude_roots = [str(not_a_directory)]
    manifest.sources.codex_roots = [str(missing)]

    roots = ContextAssembler(manifest).source_roots("laptop")
    diagnostics = preflight_provider_roots(roots, manifest.machine_map["laptop"])

    assert diagnostics == [
        f"laptop/claude source root {str(not_a_directory)!r}: is not a directory",
        f"laptop/codex source root {str(missing)!r}: does not exist",
    ]


def test_remote_preflight_uses_one_metadata_only_ssh_probe(manifest, monkeypatch) -> None:
    machine = MachineConfig(alias="remote-1", host="research.example")
    manifest.sources.remote_claude_roots = ["~/.claude/projects"]
    manifest.sources.remote_codex_roots = ["/srv/codex;safely-data"]
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        payload = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps(
                [
                    {**payload[0], "error": None},
                    {**payload[1], "error": "does not exist"},
                ]
            ),
            "",
        )

    monkeypatch.setattr("rcp.sources.preflight.subprocess.run", fake_run)

    roots = {
        "claude": manifest.sources.remote_claude_roots,
        "codex": manifest.sources.remote_codex_roots,
    }
    diagnostics = preflight_provider_roots(roots, machine)

    assert len(calls) == 1
    arguments, kwargs = calls[0]
    assert arguments[0] == "ssh"
    assert arguments[-2] == "research.example"
    assert "/srv/codex;safely-data" not in arguments[-1]
    assert kwargs["timeout"] == REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS
    assert kwargs["check"] is False
    assert diagnostics == [
        "remote-1 (research.example)/codex source root '/srv/codex;safely-data': does not exist"
    ]


def test_remote_preflight_timeout_is_precise_per_root(manifest, monkeypatch) -> None:
    machine = MachineConfig(alias="remote-1", host="research.example")
    manifest.sources.remote_claude_roots = ["~/.claude/projects"]
    manifest.sources.remote_codex_roots = ["~/.codex/sessions"]

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ssh", REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS)

    monkeypatch.setattr("rcp.sources.preflight.subprocess.run", time_out)

    roots = {
        "claude": manifest.sources.remote_claude_roots,
        "codex": manifest.sources.remote_codex_roots,
    }
    diagnostics = preflight_provider_roots(roots, machine)

    assert len(diagnostics) == 2
    assert all("remote metadata probe timed out after 180 seconds" in item for item in diagnostics)
    assert "remote-1 (research.example)/claude source root '~/.claude/projects'" in diagnostics[0]
    assert "remote-1 (research.example)/codex source root '~/.codex/sessions'" in diagnostics[1]


def test_assemble_run_keeps_missing_root_as_non_blocking_source_error(manifest, tmp_path) -> None:
    missing = tmp_path / "missing-claude-root"
    manifest.path.write_text(
        manifest.path.read_text(encoding="utf-8").replace(
            manifest.sources.claude_roots[0], str(missing)
        ),
        encoding="utf-8",
    )
    service = ProjectService(
        manifest,
        HistoryManager(manifest),
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )

    context = service.assemble_run(
        RunRequest(run_truth_scope=["repo-a"]),
        surface="refresh",
    )

    assert context.source_roots["claude"] == [str(missing)]
    assert context.source_errors == [f"laptop/claude source root {str(missing)!r}: does not exist"]
