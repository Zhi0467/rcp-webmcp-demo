import json
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rcp.agents import AgentLauncher, ProviderReadiness
from rcp.providers import PROVIDER_IDS, ClaudeProfile, CodexProfile, profile_for, runtime_label


def _result(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


CATALOG = json.dumps(
    {
        "models": [
            {
                "slug": "gpt-5.6-sol",
                "display_name": "GPT-5.6-Sol",
                "visibility": "list",
                "default_reasoning_level": "low",
                "supported_reasoning_levels": [
                    {"effort": "low"},
                    {"effort": "high"},
                    {"effort": "ultra"},
                ],
            },
            {
                "slug": "gpt-5.5",
                "display_name": "GPT-5.5",
                "visibility": "list",
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
            },
            {
                "slug": "codex-auto-review",
                "display_name": "Codex Auto Review",
                "visibility": "hide",
                "supported_reasoning_levels": [{"effort": "low"}],
            },
        ]
    }
)


def _ready_capability(
    provider: str = "codex",
    *,
    binary: str = "/opt/agents/codex",
    path_state: str = "resolved",
) -> ProviderReadiness:
    return ProviderReadiness(
        provider=provider,
        installed=True,
        authenticated=True,
        binary_path=binary,
        path_state=path_state,
    )


def test_readiness_cache_refresh_and_invalidation_are_exact(monkeypatch) -> None:
    launcher = AgentLauncher()
    calls: list[tuple[str, str, str | None]] = []

    def probe(provider: str, *, host: str, binary: str | None) -> ProviderReadiness:
        calls.append((provider, host, binary))
        return _ready_capability(provider, binary=binary or "/opt/agents/codex")

    monkeypatch.setattr(launcher, "_readiness_uncached", probe)

    first = launcher.readiness("codex", binary="/opt/agents/codex")
    second = launcher.readiness("codex", binary="/opt/agents/codex")
    refreshed = launcher.readiness(
        "codex",
        binary="/opt/agents/codex",
        refresh=True,
    )
    launcher.invalidate_readiness("codex", binary="/opt/agents/codex")
    after_invalidation = launcher.readiness("codex", binary="/opt/agents/codex")

    assert first == second == refreshed == after_invalidation
    assert calls == [
        ("codex", "", "/opt/agents/codex"),
        ("codex", "", "/opt/agents/codex"),
        ("codex", "", "/opt/agents/codex"),
    ]


def test_readiness_coalesces_concurrent_probes(monkeypatch) -> None:
    launcher = AgentLauncher()
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def probe(provider: str, *, host: str, binary: str | None) -> ProviderReadiness:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return _ready_capability(provider, binary=binary or "/opt/agents/codex")

    monkeypatch.setattr(launcher, "_readiness_uncached", probe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            launcher.readiness,
            "codex",
            binary="/opt/agents/codex",
        )
        assert entered.wait(timeout=1)
        second = executor.submit(
            launcher.readiness,
            "codex",
            binary="/opt/agents/codex",
        )
        time.sleep(0.05)
        release.set()

    assert first.result() == second.result()
    assert calls == 1


def test_invalidation_during_probe_does_not_restore_stale_capability(monkeypatch) -> None:
    launcher = AgentLauncher()
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def probe(provider: str, *, host: str, binary: str | None) -> ProviderReadiness:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=2)
        return _ready_capability(provider, binary=binary or "/opt/agents/codex")

    monkeypatch.setattr(launcher, "_readiness_uncached", probe)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            launcher.readiness,
            "codex",
            binary="/opt/agents/codex",
        )
        assert entered.wait(timeout=1)
        launcher.invalidate_readiness("codex", binary="/opt/agents/codex")
        release.set()
        assert first.result().authenticated is True

    launcher.readiness("codex", binary="/opt/agents/codex")
    assert calls == 2


def test_readiness_caches_successful_and_failed_results(monkeypatch) -> None:
    launcher = AgentLauncher()
    calls = 0
    ready = True

    def probe(provider: str, *, host: str, binary: str | None) -> ProviderReadiness:
        nonlocal calls
        calls += 1
        if ready:
            return _ready_capability(
                provider,
                binary=binary or "/opt/agents/codex",
                path_state="resolved" if binary else "unconfigured",
            )
        return ProviderReadiness(
            provider=provider,
            installed=True,
            authenticated=False,
            binary_path=binary,
            path_state="resolved" if binary else "unconfigured",
        )

    monkeypatch.setattr(launcher, "_readiness_uncached", probe)

    launcher.readiness("codex")
    launcher.readiness("codex")
    assert calls == 1

    ready = False
    first_failure = launcher.readiness("claude", binary="/opt/agents/claude")
    second_failure = launcher.readiness("claude", binary="/opt/agents/claude")
    assert first_failure == second_failure
    assert calls == 2

    ready = True
    refreshed = launcher.readiness(
        "claude",
        binary="/opt/agents/claude",
        refresh=True,
    )
    assert refreshed.authenticated is True
    assert calls == 3

    ready = False
    launcher.invalidate_readiness("claude", binary="/opt/agents/claude")
    after_invalidation = launcher.readiness("claude", binary="/opt/agents/claude")
    assert after_invalidation.authenticated is False
    assert calls == 4


def test_codex_models_come_from_the_cli_catalog_with_per_model_efforts() -> None:
    models = CodexProfile().models(_result(CATALOG))

    assert [item.id for item in models] == ["gpt-5.6-sol", "gpt-5.5"], (
        "rows Codex marks `hide` are not offered to a human"
    )
    sol, five_five = models
    assert sol.label == "GPT-5.6-Sol"
    # The efforts genuinely differ per model; a provider-wide list would offer
    # `ultra` on a model that rejects it.
    assert sol.reasoning == ["low", "high", "ultra"]
    assert five_five.reasoning == ["low", "high"]
    assert five_five.default_reasoning == "medium"


@pytest.mark.parametrize(
    "catalog",
    [
        None,
        _result("", returncode=1),
        _result("not json"),
        _result(json.dumps({"models": "wrong shape"})),
        _result(json.dumps([1, 2, 3])),
    ],
)
def test_a_broken_catalog_probe_yields_no_models_rather_than_raising(catalog) -> None:
    # An unusable catalog leaves the UI on the saved manifest values. Raising
    # here would take down the whole readiness snapshot for every machine.
    assert CodexProfile().models(catalog) == []


def test_claude_declares_its_lists_and_says_which_cli_they_came_from() -> None:
    profile = ClaudeProfile()

    assert profile.catalog_command("claude") is None, "Claude Code cannot enumerate its models"
    models = profile.models(None)
    assert [item.id for item in models] == ["opus", "sonnet", "haiku", "fable"]
    # `claude --help` documents these as the accepted values of --effort.
    assert models[0].reasoning == ["low", "medium", "high", "xhigh", "max"]
    assert profile.declared_against, "a hand-maintained list must record its CLI version"


def test_authentication_is_read_the_way_each_cli_reports_it() -> None:
    assert CodexProfile().login_command("/opt/codex") == ["/opt/codex", "login"]
    assert CodexProfile().is_authenticated(_result("Logged in using ChatGPT"))
    assert not CodexProfile().is_authenticated(_result("Not logged in"))
    assert not CodexProfile().is_authenticated(_result("Logged in", returncode=1))

    assert ClaudeProfile().login_command("/opt/claude") == [
        "/opt/claude",
        "auth",
        "login",
    ]
    assert ClaudeProfile().is_authenticated(_result(json.dumps({"loggedIn": True})))
    assert not ClaudeProfile().is_authenticated(_result(json.dumps({"loggedIn": False})))
    assert not ClaudeProfile().is_authenticated(_result("not json"))


def test_the_registry_is_the_only_list_of_providers() -> None:
    assert PROVIDER_IDS == ("codex", "claude")
    for provider in PROVIDER_IDS:
        assert profile_for(provider).id == provider
        assert profile_for(provider).label

    with pytest.raises(ValueError, match="Unknown agent provider"):
        profile_for("gemini")


def test_an_unknown_provider_is_rejected_by_the_schema_layer() -> None:
    from rcp.config import AgentSurfaceConfig, MachineConfig

    AgentSurfaceConfig(provider="claude", run_on="local")
    with pytest.raises(ValueError, match="Unknown agent provider"):
        AgentSurfaceConfig(provider="gemini", run_on="local")
    with pytest.raises(ValueError, match="Unknown agent provider"):
        MachineConfig(alias="local", provider_paths={"gemini": "/opt/gemini"})


def test_agent_profile_runtime_is_provider_owned_and_backward_compatible() -> None:
    from rcp.config import AgentSurfaceConfig

    assert AgentSurfaceConfig(provider="codex", run_on="local").runtime == "exec"
    assert (
        AgentSurfaceConfig(
            provider="codex",
            runtime="codex.app-server-stdio.v1",
            run_on="local",
        ).runtime
        == "app-server"
    )
    assert AgentSurfaceConfig(provider="claude", run_on="local").runtime == "stream-json"
    with pytest.raises(ValueError, match="does not support runtime"):
        AgentSurfaceConfig(provider="claude", runtime="app-server", run_on="local")


def test_readiness_names_the_runtimes_and_the_one_an_omitted_value_means() -> None:
    for provider in PROVIDER_IDS:
        readiness = ProviderReadiness(provider=provider, installed=True, authenticated=True)
        profile = profile_for(provider)
        assert readiness.default_runtime == profile.default_runtime
        assert [choice.id for choice in readiness.runtimes] == [
            choice.id for choice in profile.runtime_choices
        ]
        # A surface reads the default by name rather than taking a position.
        assert readiness.default_runtime in {choice.id for choice in readiness.runtimes}


def test_a_durable_runtime_id_is_named_for_the_surface_that_reports_it() -> None:
    assert runtime_label("codex", "codex.exec-json.v1") == "Codex exec"
    assert runtime_label("codex", "codex.app-server-stdio.v1") == "Codex app server"
    assert runtime_label("claude", "claude.stream-json.v1") == "Claude stream JSON"
    # A record naming a runtime this build no longer offers keeps its stored id.
    assert runtime_label("codex", "codex.retired.v1") == "codex.retired.v1"


def test_machine_provider_paths_are_backward_compatible_and_absolute(manifest) -> None:
    from rcp.config import MachineConfig

    assert manifest.machine_map["laptop"].provider_paths == {}
    configured = MachineConfig(
        alias="remote",
        host="gpu.example",
        os_account="alice",
        provider_paths={"codex": "/opt/codex/bin/codex"},
    )
    assert configured.os_account == "alice"
    assert configured.provider_paths == {"codex": "/opt/codex/bin/codex"}
    with pytest.raises(ValueError, match="must be absolute"):
        MachineConfig(alias="local", provider_paths={"codex": "bin/codex"})
    with pytest.raises(ValueError, match="operating-system account"):
        MachineConfig(alias="remote", host="gpu.example", os_account="alice@example")


def test_failed_version_command_does_not_publish_stderr_as_a_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rcp.agents.launcher import AgentLauncher

    binary = tmp_path / "codex"
    monkeypatch.setattr(shutil, "which", lambda _provider: str(binary))

    def probe(_host: str, command: list[str], **_kwargs):
        if command[-1] == "--version":
            return _result("provider internal error", returncode=1)
        if command[-2:] == ["login", "status"]:
            return _result("Logged in using ChatGPT")
        return _result(json.dumps({"models": []}))

    monkeypatch.setattr(AgentLauncher, "_probe", staticmethod(probe))

    readiness = AgentLauncher().readiness("codex")

    assert readiness.installed is True
    assert readiness.authenticated is True
    assert readiness.version is None


def test_provider_commands_use_the_recorded_binary_as_argv_zero() -> None:
    from rcp.agents.launcher import AgentLauncher

    command = AgentLauncher._command(
        "codex",
        "prompt",
        binary="/Applications/Codex.app/Contents/MacOS/codex",
        cwd=Path("/tmp/project"),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[],
        capability="scratch_patch",
    )

    assert command[0] == "/Applications/Codex.app/Contents/MacOS/codex"


def test_stale_recorded_path_never_falls_back_to_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rcp.agents.launcher import AgentLauncher

    monkeypatch.setattr(
        shutil,
        "which",
        lambda _provider: (_ for _ in ()).throw(AssertionError("PATH fallback is forbidden")),
    )

    readiness = AgentLauncher().readiness("codex", binary="/missing/recorded/codex")

    assert readiness.path_state == "missing"
    assert readiness.binary_path == "/missing/recorded/codex"
    assert "does not exist" in (readiness.reason or "")


def test_local_recorded_path_without_execute_permission_is_denied(tmp_path: Path) -> None:
    from rcp.agents.launcher import AgentLauncher

    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o644)

    readiness = AgentLauncher().readiness("codex", binary=str(binary))

    assert readiness.path_state == "denied"
    assert readiness.binary_path == str(binary)
    assert readiness.installed is False
    assert "not executable" in (readiness.reason or "")


def test_local_recorded_path_that_is_not_a_file_is_denied(tmp_path: Path) -> None:
    from rcp.agents.launcher import AgentLauncher

    readiness = AgentLauncher().readiness("codex", binary=str(tmp_path))

    assert readiness.path_state == "denied"
    assert readiness.binary_path == str(tmp_path)
    assert "not a regular file" in (readiness.reason or "")


def test_remote_readiness_checks_and_uses_the_recorded_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rcp.agents.launcher import AgentLauncher

    calls: list[list[str]] = []

    def probe(_self, _host: str, command: list[str]):
        calls.append(command)
        if command[-2:] == ["login", "status"]:
            return _result("Logged in using ChatGPT")
        return _result(CATALOG if command[-2:] == ["debug", "models"] else "codex test")

    monkeypatch.setattr(AgentLauncher, "_probe", probe)
    readiness = AgentLauncher().readiness(
        "codex",
        host="gpu.example",
        binary="/opt/agent/bin/codex",
    )

    assert readiness.path_state == "resolved"
    assert calls[0][0:2] == ["python3", "-c"]
    assert calls[0][-1] == "/opt/agent/bin/codex"
    assert ["/opt/agent/bin/codex", "login", "status"] in calls
    assert all(command[0] != "codex" for command in calls if command[0] != "python3")


@pytest.mark.parametrize(
    ("returncode", "path_state", "reason"),
    [
        (40, "missing", "does not exist"),
        (41, "denied", "access"),
        (42, "denied", "not a regular file"),
        (43, "denied", "not executable"),
        (44, "denied", "could not be inspected"),
    ],
)
def test_remote_recorded_path_probe_distinguishes_why_it_cannot_launch(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    path_state: str,
    reason: str,
) -> None:
    from rcp.agents.launcher import AgentLauncher

    calls: list[list[str]] = []

    def probe(_self, _host: str, command: list[str]):
        calls.append(command)
        return _result(returncode=returncode)

    monkeypatch.setattr(AgentLauncher, "_probe", probe)
    readiness = AgentLauncher().readiness(
        "codex",
        host="gpu.example",
        binary="/opt/Agent Tools/codex",
    )

    assert readiness.path_state == path_state
    assert readiness.binary_path == "/opt/Agent Tools/codex"
    assert reason.lower() in (readiness.reason or "").lower()
    assert calls[0][0:2] == ["python3", "-c"]
    assert calls[0][-1] == "/opt/Agent Tools/codex"


def test_remote_recorded_path_probe_quotes_the_path_as_one_argument() -> None:
    from rcp.agents.launcher import _REMOTE_PATH_PROBE, AgentLauncher

    binary = "/opt/Agent Tools/$(touch should-not-run)'codex"
    remote_command = AgentLauncher._remote_login_command(
        ["python3", "-c", _REMOTE_PATH_PROBE, binary]
    )
    outer = shlex.split(remote_command)

    assert outer[:2] == ["bash", "-lic"]
    assert shlex.split(outer[2]) == ["python3", "-c", _REMOTE_PATH_PROBE, binary]


def test_remote_recorded_path_probe_reports_missing_and_non_executable(
    tmp_path: Path,
) -> None:
    from rcp.agents.launcher import _REMOTE_PATH_PROBE

    missing = subprocess.run(
        [sys.executable, "-c", _REMOTE_PATH_PROBE, str(tmp_path / "missing")],
        check=False,
    )
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o644)
    denied = subprocess.run(
        [sys.executable, "-c", _REMOTE_PATH_PROBE, str(binary)],
        check=False,
    )

    assert missing.returncode == 40
    assert denied.returncode == 43


def test_an_unreachable_host_is_not_reported_as_a_missing_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rcp.agents.launcher import AgentLauncher

    # ssh exits 255 when it cannot reach the host; `command -v` exits 1 when the
    # binary is merely absent. Conflating them tells the human to go install
    # something on a machine that never answered.
    launcher = AgentLauncher()
    monkeypatch.setattr(AgentLauncher, "_probe", lambda self, host, cmd: _result("", 255))
    assert "unreachable" in (launcher.readiness("codex", host="offline").reason or "")

    monkeypatch.setattr(AgentLauncher, "_probe", lambda self, host, cmd: _result("", 1))
    assert "not installed" in (launcher.readiness("codex", host="online").reason or "")


def test_remote_shell_noise_is_not_reported_as_the_failure_reason() -> None:
    from rcp.agents.launcher import _exit_reason, _meaningful_stderr

    # `bash -lic` emits these on every remote run, successful ones included.
    noise = (
        "bash: cannot set terminal process group (-1): Inappropriate ioctl for device\n"
        "bash: no job control in this shell"
    )
    assert _meaningful_stderr(noise) == ""
    assert _meaningful_stderr(noise + "\nerror: real provider failure") == (
        "error: real provider failure"
    )

    # With the noise gone, a severed connection must say so rather than fall
    # back to shell chatter — ssh exits 255 when the connection drops.
    assert _exit_reason("codex", 255, "gpu0") == (
        "The connection to gpu0 was lost before codex finished."
    )
    assert _exit_reason("codex", 1, "gpu0") == "codex exited 1 on gpu0."
    assert _exit_reason("codex", 1, "") == "codex exited 1."

    # asyncio negates the signal number; a severed remote link surfaces as the
    # killed ssh client, which is what S14's interrupt actually produced.
    assert _exit_reason("codex", -9, "gpu0") == (
        "The connection to gpu0 ended (SIGKILL) before codex finished."
    )
    assert _exit_reason("codex", -9, "") == "codex was stopped by SIGKILL."
