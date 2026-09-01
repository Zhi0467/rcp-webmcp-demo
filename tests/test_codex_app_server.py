from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rcp.agents import AgentLauncher
from rcp.agents.codex_app_server import CodexAppServerRuntime
from rcp.agents.write_scope import ProjectWriteScope
from rcp.providers import ProviderTurnRequest


def _fake_app_server(
    tmp_path: Path,
    *,
    request_approval: bool = False,
    permission_profile: str | None = None,
) -> tuple[Path, Path]:
    executable = tmp_path / "fake-codex"
    capture = tmp_path / "app-server-capture.json"
    executable.write_text(
        f"""#!{sys.executable}
import json
import sys
from pathlib import Path

capture = Path({str(capture)!r})
seen = []

def send(value):
    print(json.dumps(value, separators=(\",\", \":\")), flush=True)

for line in sys.stdin:
    message = json.loads(line)
    seen.append(message)
    method = message.get(\"method\")
    request_id = message.get(\"id\")
    if method == \"initialize\":
        send({{"jsonrpc": \"2.0\", "id": request_id, "result": {{"userAgent": \"fake\"}}}})
    elif method == \"config/read\":
        send({{
            "jsonrpc": \"2.0\",
            "id": request_id,
            "result": {{
                "config": {{
                    "apps": {{"calendar": {{"enabled": True}}}},
                    "mcp_servers": {{"github": {{"enabled": True}}}},
                    "plugins": {{"example": {{"enabled": True}}}},
                    "hooks": {{"Stop": [{{"command": \"ambient\"}}]}},
                }}
            }},
        }})
    elif method in {{\"thread/start\", \"thread/resume\"}}:
        thread_id = message[\"params\"].get(\"threadId\", \"app-thread-1\")
        result = {{
            "thread": {{"id": thread_id}},
            "approvalPolicy": \"never\",
            "sandbox": {{"type": \"readOnly\"}},
        }}
        if {permission_profile!r} is not None:
            result[\"activePermissionProfile\"] = {{"id": {permission_profile!r}}}
        send({{"jsonrpc": \"2.0\", "id": request_id, "result": result}})
    elif method == \"turn/start\":
        send({{
            "jsonrpc": \"2.0\",
            "id": request_id,
            "result": {{"turn": {{"id": \"turn-1\", "status": \"inProgress\"}}}},
        }})
        if {request_approval!r}:
            send({{
                "jsonrpc": \"2.0\",
                "id": 99,
                "method": \"item/commandExecution/requestApproval\",
                "params": {{"threadId": \"app-thread-1\", "turnId": \"turn-1\"}},
            }})
        else:
            send({{
                "jsonrpc": \"2.0\",
                "method": \"item/completed\",
                "params": {{
                    "threadId": \"app-thread-1\",
                    "turnId": \"turn-1\",
                    "item": {{"type": \"agentMessage\", "text": \"APP_SERVER_OK\"}},
                }},
            }})
            send({{
                "jsonrpc": \"2.0\",
                "method": \"thread/tokenUsage/updated\",
                "params": {{
                    "threadId": \"app-thread-1\",
                    "turnId": \"turn-1\",
                    "tokenUsage": {{
                        "last": {{"inputTokens": 7, "outputTokens": 3, "totalTokens": 10}}
                    }},
                }},
            }})
            send({{
                "jsonrpc": \"2.0\",
                "method": \"turn/completed\",
                "params": {{
                    "threadId": \"app-thread-1\",
                    "turn": {{"id": \"turn-1\", "status": \"completed\"}},
                }},
            }})

capture.write_text(json.dumps({{"argv": sys.argv, "messages": seen}}), encoding=\"utf-8\")
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, capture


def _ready(executable: Path):
    return type(
        "Readiness",
        (),
        {
            "installed": True,
            "authenticated": True,
            "path_state": "resolved",
            "binary_path": str(executable),
            "version": "0.149.0",
        },
    )()


def _fake_fallback_codex(tmp_path: Path, *, fail_after_prompt: bool) -> tuple[Path, Path]:
    executable = tmp_path / "fallback-codex"
    attempts = tmp_path / "runtime-attempts.txt"
    executable.write_text(
        f"""#!{sys.executable}
import json
import sys
from pathlib import Path

attempts = Path({str(attempts)!r})
runtime = "app-server" if len(sys.argv) > 1 and sys.argv[1] == "app-server" else "exec"
with attempts.open("a", encoding="utf-8") as stream:
    stream.write(runtime + "\\n")

def send(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)

if runtime == "exec":
    sys.stdin.read()
    thread_id = sys.argv[-2] if "resume" in sys.argv else "exec-thread"
    send({{"type": "thread.started", "thread_id": thread_id}})
    send({{"type": "item.completed", "item": {{"type": "agent_message", "text": "EXEC_OK"}}}})
    send({{"type": "turn.completed", "usage": {{"input_tokens": 1, "output_tokens": 1}}}})
    raise SystemExit(0)

if not {fail_after_prompt!r}:
    raise SystemExit(12)

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        send({{"jsonrpc": "2.0", "id": request_id, "result": {{}}}})
    elif method == "config/read":
        send({{"jsonrpc": "2.0", "id": request_id, "result": {{"config": {{}}}}}})
    elif method == "thread/start":
        send({{
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {{
                "thread": {{"id": "app-thread"}},
                "approvalPolicy": "never",
                "sandbox": {{"type": "readOnly"}},
            }},
        }})
    elif method == "turn/start":
        raise SystemExit(13)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, attempts


@pytest.mark.asyncio
async def test_app_server_runtime_normalizes_one_fresh_local_turn(tmp_path: Path) -> None:
    executable, capture = _fake_app_server(tmp_path)
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="", binary=None: _ready(executable)

    events = [
        event
        async for event in launcher.stream(
            "codex",
            "Reply exactly APP_SERVER_OK",
            cwd=tmp_path,
            capability="paper_readonly",
            runtime_id="codex.app-server-stdio.v1",
        )
    ]

    assert [event.session_id for event in events if event.event == "session"] == ["app-thread-1"]
    assert [event.text for event in events if event.event == "answer"] == ["APP_SERVER_OK"]
    usage = next(event.usage for event in events if event.usage is not None)
    assert usage.provider_profile == "codex.app-server.turn.v1"
    assert usage.processed_input_tokens == 7
    assert events[-1].event == "done", [(event.event, event.text) for event in events]
    transcript = json.loads(capture.read_text(encoding="utf-8"))
    assert transcript["argv"][1:3] == ["app-server", "--stdio"]
    assert all("jsonrpc" not in item for item in transcript["messages"])
    thread_start = next(
        item for item in transcript["messages"] if item.get("method") == "thread/start"
    )
    assert thread_start["params"]["config"]["mcp_servers"]["github"] == {"enabled": False}
    assert thread_start["params"]["config"]["hooks"]["Stop"] == []
    assert thread_start["params"]["config"]["project_doc_max_bytes"] == 0


def _work_scope(stage: Path) -> ProjectWriteScope:
    return ProjectWriteScope.create(
        project_id="project-1",
        execution_machine="local",
        execution_host="",
        capability="work_auto",
        stage_root=str(stage),
        workspace_root=str(stage),
        repositories=[],
        protected_write_paths=[],
    )


@pytest.mark.asyncio
async def test_app_server_work_turn_carries_the_exact_project_permission_profile(
    tmp_path: Path,
) -> None:
    """A Work-like app-server turn is contained by the profile, not by a sandbox."""

    executable, capture = _fake_app_server(tmp_path, permission_profile="rcp_project")
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="", binary=None: _ready(executable)
    stage = tmp_path / "stage"
    stage.mkdir()

    events = [
        event
        async for event in launcher.stream(
            "codex",
            "Reply exactly APP_SERVER_OK",
            cwd=stage,
            write_dirs=[],
            write_scope=_work_scope(stage),
            capability="work_auto",
            runtime_id="codex.app-server-stdio.v1",
        )
    ]

    assert [event.text for event in events if event.event == "answer"] == ["APP_SERVER_OK"]
    assert events[-1].event == "done", [(event.event, event.text) for event in events]
    transcript = json.loads(capture.read_text(encoding="utf-8"))
    assert 'default_permissions="rcp_project"' in transcript["argv"]
    assert any(
        item.startswith("permissions={rcp_project=") and str(stage) in item
        for item in transcript["argv"]
    ), transcript["argv"]
    thread_start = next(
        item for item in transcript["messages"] if item.get("method") == "thread/start"
    )
    assert thread_start["params"]["permissions"] == "rcp_project"
    # The profile is the whole containment; a sandbox would silently narrow it.
    assert "sandbox" not in thread_start["params"]
    turn_start = next(item for item in transcript["messages"] if item.get("method") == "turn/start")
    assert "sandboxPolicy" not in turn_start["params"]


@pytest.mark.asyncio
async def test_app_server_work_turn_stops_when_the_permission_profile_is_not_active(
    tmp_path: Path,
) -> None:
    """Codex answering without RCP's exact profile must not reach the prompt."""

    executable, capture = _fake_app_server(tmp_path, permission_profile="workspace")
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="", binary=None: _ready(executable)
    stage = tmp_path / "stage"
    stage.mkdir()

    events = [
        event
        async for event in launcher.stream(
            "codex",
            "Reply exactly APP_SERVER_OK",
            cwd=stage,
            write_dirs=[],
            write_scope=_work_scope(stage),
            capability="work_auto",
            runtime_id="codex.app-server-stdio.v1",
        )
    ]

    assert all(event.event != "answer" for event in events)
    fallbacks = [json.loads(event.text) for event in events if event.event == "runtime_fallback"]
    assert [item["runtime_id"] for item in fallbacks] == ["codex.app-server-stdio.v1"]
    assert "exact project permission profile" in fallbacks[0]["detail"]
    transcript = json.loads(capture.read_text(encoding="utf-8"))
    assert all(item.get("method") != "turn/start" for item in transcript["messages"])


def test_app_server_read_only_capability_rejects_a_project_write_scope(tmp_path: Path) -> None:
    """A read-only turn carrying write authority is a contradiction, not a narrowing."""

    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(ValueError, match="cannot carry a project write scope"):
        CodexAppServerRuntime().turn(
            ProviderTurnRequest(
                prompt="Read only",
                binary="codex",
                cwd=stage,
                model=None,
                reasoning=None,
                session_id=None,
                read_dirs=[],
                write_dirs=[],
                write_scope=_work_scope(stage),
                capability="paper_readonly",
                provider_version="0.149.0",
            )
        )


@pytest.mark.asyncio
async def test_app_server_runtime_fails_loudly_on_interactive_request(tmp_path: Path) -> None:
    executable, _capture = _fake_app_server(tmp_path, request_approval=True)
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="", binary=None: _ready(executable)

    events = [
        event
        async for event in launcher.stream(
            "codex",
            "Do not ask for approval",
            cwd=tmp_path,
            capability="paper_readonly",
            runtime_id="codex.app-server-stdio.v1",
        )
    ]

    errors = [event.text for event in events if event.event == "error"]
    assert errors == [
        "Codex app-server requested interactive input "
        "(item/commandExecution/requestApproval); RCP stopped the unattended turn."
    ]
    assert all(event.event != "done" for event in events)


@pytest.mark.asyncio
async def test_app_server_runtime_uses_the_existing_ssh_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, _capture = _fake_app_server(tmp_path)
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="", binary=None: _ready(executable)
    remote_commands: list[tuple[str, str]] = []

    def local_ssh(host: str, command: str) -> list[str]:
        remote_commands.append((host, command))
        # macOS has no `setsid`; retain the exact remote command for assertions
        # and remove only that Linux process-group wrapper in this local drive.
        return ["bash", "-c", command.replace("setsid sh -c", "sh -c")]

    monkeypatch.setattr("rcp.agents.launcher.ssh_arguments", local_ssh)
    pid_file = tmp_path / "agent.pid"
    events = [
        event
        async for event in launcher.stream(
            "codex",
            "Reply exactly APP_SERVER_OK",
            cwd=tmp_path,
            host="research-host",
            remote_pid_file=str(pid_file),
            capability="paper_readonly",
            runtime_id="codex.app-server-stdio.v1",
        )
    ]

    assert events[-1].event == "done", [(event.event, event.text) for event in events]
    assert remote_commands and remote_commands[0][0] == "research-host"
    wrapped = remote_commands[0][1]
    assert "app-server --stdio" in wrapped
    assert f"cd {tmp_path}" in wrapped
    assert str(pid_file) in wrapped
    assert pid_file.read_text(encoding="utf-8").strip().isdigit()


@pytest.mark.asyncio
async def test_app_server_falls_back_to_exec_only_before_prompt_delivery(
    tmp_path: Path,
) -> None:
    executable, attempts = _fake_fallback_codex(tmp_path, fail_after_prompt=False)
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="", binary=None: _ready(executable)

    events = [
        event
        async for event in launcher.stream(
            "codex",
            "Reply exactly EXEC_OK",
            cwd=tmp_path,
            session_id="existing-app-thread",
            capability="paper_readonly",
            runtime_id="app-server",
        )
    ]

    assert [event.text for event in events if event.event == "runtime"] == ["codex.exec-json.v1"]
    assert [event.text for event in events if event.event == "answer"] == ["EXEC_OK"]
    assert [event.session_id for event in events if event.event == "session"] == [
        "existing-app-thread"
    ]
    assert all(event.event != "error" for event in events)
    assert attempts.read_text(encoding="utf-8").splitlines() == ["app-server", "exec"]


@pytest.mark.asyncio
async def test_app_server_does_not_fallback_after_prompt_delivery(tmp_path: Path) -> None:
    executable, attempts = _fake_fallback_codex(tmp_path, fail_after_prompt=True)
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="", binary=None: _ready(executable)

    events = [
        event
        async for event in launcher.stream(
            "codex",
            "This prompt must not be duplicated",
            cwd=tmp_path,
            capability="paper_readonly",
            runtime_id="app-server",
        )
    ]

    assert [event.text for event in events if event.event == "runtime"] == [
        "codex.app-server-stdio.v1"
    ]
    assert any(event.event == "error" for event in events)
    assert attempts.read_text(encoding="utf-8").splitlines() == ["app-server"]
