from __future__ import annotations

import json
import subprocess
from pathlib import Path

from rcp.agents import ProviderReadiness
from rcp.provider_skills import ProviderSkillInventoryManager
from rcp.providers import ProviderSkillProbe, profile_for
from rcp.storage import AppStore


def _ready(provider: str, binary: str, version: str = "provider 1.2.3") -> ProviderReadiness:
    return ProviderReadiness(
        provider=provider,
        installed=True,
        authenticated=True,
        version=version,
        binary_path=binary,
        path_state="resolved",
    )


def _claude_output(*names: str) -> str:
    return json.dumps(
        {
            "type": "system",
            "subtype": "init",
            "skills": list(names),
        }
    )


def test_success_replaces_inventory_and_failure_preserves_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = AppStore(tmp_path / "app.sqlite3")
    manager = ProviderSkillInventoryManager(store)
    outputs = [_claude_output("review", "plugin:triage")]

    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, outputs.pop(0), "")

    monkeypatch.setattr(subprocess, "run", run)
    manager.mark_refreshing("claude", "", "/opt/claude")
    fresh = manager.refresh("claude", "", "/opt/claude", _ready("claude", "/opt/claude"))

    assert fresh.status == "fresh"
    assert [skill.name for skill in fresh.skills] == ["plugin:triage", "review"]
    assert fresh.command[0] == "/opt/claude"
    assert fresh.inventory_hash
    saved = fresh.model_copy(deep=True)

    def fail(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 7, "", "probe unavailable")

    monkeypatch.setattr(subprocess, "run", fail)
    manager.mark_refreshing("claude", "", "/opt/claude")
    stale = manager.refresh("claude", "", "/opt/claude", _ready("claude", "/opt/claude"))

    assert stale.status == "stale"
    assert stale.stale is True
    assert stale.diagnostic == "probe unavailable"
    assert stale.skills == saved.skills
    assert stale.provider_version == saved.provider_version
    assert stale.command == saved.command
    assert stale.inventory_hash == saved.inventory_hash
    references = manager.resolve("claude", "", "/opt/claude", "laptop", ["review", "plugin:triage"])
    assert [reference.name for reference in references] == ["review", "plugin:triage"]
    assert all(reference.stale for reference in references)


def test_first_failure_has_no_native_skills(tmp_path: Path) -> None:
    manager = ProviderSkillInventoryManager(AppStore(tmp_path / "app.sqlite3"))
    manager.mark_refreshing("codex", "gpu.example", "/opt/codex")
    snapshot = manager.refresh(
        "codex",
        "gpu.example",
        "/opt/codex",
        ProviderReadiness(
            provider="codex",
            installed=False,
            authenticated=False,
            path_state="unreachable",
            reason="gpu.example is unreachable",
        ),
    )

    assert snapshot.status == "unavailable"
    assert snapshot.skills == []
    assert snapshot.inventory_hash is None
    assert snapshot.diagnostic == "gpu.example is unreachable"


def test_codex_probe_waits_for_initialize_before_listing_skills(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writes: list[dict[str, object]] = []

    class Stdin:
        def write(self, value: str) -> None:
            writes.append(json.loads(value))

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    class Process:
        stdin = Stdin()
        stdout = iter(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"userAgent": "codex"}}) + "\n",
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {
                            "data": [
                                {
                                    "skills": [
                                        {
                                            "name": "audit",
                                            "description": "Audit the graph",
                                            "enabled": True,
                                            "scope": "user",
                                            "path": "/skills/audit/SKILL.md",
                                            "interface": {"displayName": "Graph audit"},
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                )
                + "\n",
            ]
        )
        stderr = iter([])

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: Process())
    manager = ProviderSkillInventoryManager(AppStore(tmp_path / "app.sqlite3"))
    manager.mark_refreshing("codex", "", "/opt/codex")
    snapshot = manager.refresh("codex", "", "/opt/codex", _ready("codex", "/opt/codex"))

    assert snapshot.status == "fresh"
    assert [skill.name for skill in snapshot.skills] == ["audit"]
    assert [(value.get("method"), value.get("id")) for value in writes] == [
        ("initialize", 1),
        ("initialized", None),
        ("skills/list", 2),
    ]
    assert writes[0]["params"] == {
        "clientInfo": {"name": "rcp", "version": "1"},
        "capabilities": {},
    }
    assert writes[-1]["params"] == {"cwds": ["/"], "forceReload": True}


def test_remote_probe_uses_existing_ssh_login_shell_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def run(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(arguments, 0, _claude_output("research"), "")

    monkeypatch.setattr(subprocess, "run", run)
    manager = ProviderSkillInventoryManager(AppStore(tmp_path / "app.sqlite3"))
    manager.mark_refreshing("claude", "gpu.example", "/remote/claude")
    result = manager.refresh(
        "claude",
        "gpu.example",
        "/remote/claude",
        _ready("claude", "/remote/claude"),
    )

    arguments = captured["arguments"]
    assert arguments[0] == "ssh"
    assert "gpu.example" in arguments
    assert "bash -lic" in arguments[-1]
    assert "/remote/claude" in arguments[-1]
    assert result.status == "fresh"


def test_refresh_command_is_owned_by_provider_profile() -> None:
    codex = profile_for("codex").skill_probe("/opt/codex")
    claude = profile_for("claude").skill_probe("/opt/claude")

    assert codex == ProviderSkillProbe(command=["/opt/codex", "app-server"], protocol="jsonrpc")
    assert claude.command[0] == "/opt/claude"
    assert claude.protocol == "jsonl"
    assert "--no-session-persistence" in claude.command
    assert '{"disableAllHooks":true}' in claude.command
