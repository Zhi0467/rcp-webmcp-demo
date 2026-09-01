import asyncio
import json
import shlex
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from rcp.agents import AgentEvent, AgentLauncher, AgentProcessControl, ProviderReadiness
from rcp.agents.command_mailbox import serve_command_mailbox, stage_command_mailbox
from rcp.agents.command_protocol import CommandResponse, staged_command_broker_source
from rcp.agents.write_scope import ProjectWriteScope, WritableRepositoryRoot
from rcp.providers import ProviderRuntimeStep, ProviderTurnRequest, profile_for


def _decoded_step(provider: str, line: str) -> ProviderRuntimeStep:
    """Decode one JSONL line through the runtime the launcher actually runs."""

    profile = profile_for(provider)
    turn = profile.runtime(profile.legacy_runtime_id).turn(
        ProviderTurnRequest(
            prompt="",
            binary=provider,
            cwd=Path("/project/.research"),
            model=None,
            reasoning=None,
            session_id=None,
            read_dirs=[],
            write_dirs=[],
            write_scope=None,
            capability="scratch_patch",
            provider_version=None,
            legacy_command=[provider],
        )
    )
    return turn.receive_line(line)


def _decoded_event(provider: str, line: str) -> AgentEvent:
    step = _decoded_step(provider, line)
    assert len(step.events) == 1
    decoded = step.events[0]
    return AgentEvent(
        event=decoded.event,
        text=decoded.text,
        session_id=decoded.session_id,
        usage=decoded.usage,
    )


def _project_write_scope(
    *,
    capability: Literal["work_auto", "orchestrate"],
    stage: str,
    repository_paths: list[str],
) -> ProjectWriteScope:
    repositories = [
        WritableRepositoryRoot(alias=Path(path).name, machine="local", path=path)
        for path in sorted(repository_paths)
    ]
    return ProjectWriteScope.create(
        project_id="project-1",
        execution_machine="local",
        execution_host="",
        capability=capability,
        stage_root=stage,
        workspace_root=stage,
        repositories=repositories,
        protected_write_paths=[f"{path}/.research" for path in repository_paths],
    )


def test_forced_readiness_refresh_supersedes_inflight_warm_probe(monkeypatch) -> None:
    launcher = AgentLauncher()
    ordinary_entered = threading.Event()
    forced_entered = threading.Event()
    release_ordinary = threading.Event()
    release_forced = threading.Event()
    calls = 0

    def probe(provider: str, *, host: str, binary: str | None) -> ProviderReadiness:
        nonlocal calls
        calls += 1
        if calls == 1:
            ordinary_entered.set()
            assert release_ordinary.wait(timeout=2)
            version = "warm-result"
        else:
            forced_entered.set()
            assert release_forced.wait(timeout=2)
            version = "refreshed-result"
        return ProviderReadiness(
            provider=provider,
            installed=True,
            authenticated=True,
            version=version,
            binary_path=binary,
            path_state="resolved",
        )

    monkeypatch.setattr(launcher, "_readiness_uncached", probe)
    binary = "/opt/agents/codex"
    with ThreadPoolExecutor(max_workers=4) as executor:
        ordinary = executor.submit(launcher.readiness, "codex", binary=binary)
        assert ordinary_entered.wait(timeout=1)
        forced = executor.submit(
            launcher.readiness,
            "codex",
            binary=binary,
            refresh=True,
        )
        assert forced_entered.wait(timeout=1)
        with launcher._readiness_lock:
            forced_probe = launcher._readiness_probes[("codex", "", binary, True)]
        follower_waiting = threading.Event()
        wait_for_forced = forced_probe.completed.wait

        def signal_forced_wait(timeout: float | None = None) -> bool:
            follower_waiting.set()
            return wait_for_forced(timeout)

        monkeypatch.setattr(forced_probe.completed, "wait", signal_forced_wait)
        forced_follower = executor.submit(
            launcher.readiness,
            "codex",
            binary=binary,
            refresh=True,
        )
        ordinary_follower = executor.submit(
            launcher.readiness,
            "codex",
            binary=binary,
        )
        assert follower_waiting.wait(timeout=1)
        release_forced.set()
        assert forced.result().version == "refreshed-result"
        assert forced_follower.result().version == "refreshed-result"
        assert ordinary_follower.result().version == "refreshed-result"
        release_ordinary.set()
        assert ordinary.result().version == "warm-result"

    assert calls == 2
    assert launcher.readiness("codex", binary=binary).version == "refreshed-result"


def test_provider_usage_is_normalized_at_provider_boundaries() -> None:
    codex_line = json.dumps(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1_000,
                "cached_input_tokens": 400,
                "output_tokens": 120,
                "reasoning_output_tokens": 80,
            },
        }
    )
    codex = _decoded_event("codex", codex_line)
    assert codex.usage is not None
    assert codex.usage.provider_profile == "codex.turn.v1"
    assert codex.usage.processed_input_tokens == 1_000
    assert codex.usage.cached_input_tokens == 400
    assert codex.usage.generated_tokens == 120

    claude_line = json.dumps(
        {
            "type": "result",
            "uuid": "query-result",
            "usage": {
                "input_tokens": 600,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 300,
                "output_tokens": 90,
                "output_tokens_details": {"thinking_tokens": 50},
            },
            "result": "Done",
        }
    )
    claude = profile_for("claude").decode_event(json.loads(claude_line), claude_line)
    assert claude.usage is not None
    assert claude.usage.provider_profile == "claude.query.v1"
    assert claude.usage.processed_input_tokens == 1_000
    assert claude.usage.cached_input_tokens == 300
    assert claude.usage.reasoning_output_tokens == 50


@pytest.mark.asyncio
async def test_campaign_broker_wraps_provider_and_preserves_exact_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged = stage_command_mailbox(
        local_stage=tmp_path,
        remote_stage=None,
        episode_id="episode",
        task_id="task",
        turn_id="turn",
        timeout_seconds=2,
    )
    assert staged.invocation_gate is not None
    handled: list[str] = []

    def handler(request, _identity):
        handled.append(request.verb)
        return CommandResponse(request_id=request.request_id, status="ok")

    client = list(staged.client_argv("status"))
    provider_script = (
        "import json,subprocess,sys\n"
        "prompt=sys.stdin.read()\n"
        f"result=subprocess.run({client!r},capture_output=True,text=True,check=False)\n"
        "print(json.dumps({'type':'item','item':{'text':"
        "json.dumps({'prompt':prompt,'code':result.returncode,'output':result.stdout})}}),flush=True)\n"
    )
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="": type(
        "Readiness",
        (),
        {"installed": True, "authenticated": True, "path_state": "resolved"},
    )()
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [sys.executable, "-c", provider_script],
    )
    stop = asyncio.Event()
    server = asyncio.create_task(
        serve_command_mailbox(
            staged=staged,
            handler=handler,
            stop=stop,
            poll_seconds=0.01,
            invocation_gate=staged.invocation_gate,
        )
    )
    try:
        events = [
            event
            async for event in launcher.stream(
                "codex",
                "exact provider prompt",
                cwd=tmp_path,
                capability="orchestrate",
                invocation_gate=staged.invocation_gate,
            )
        ]
    finally:
        stop.set()
        await server
        staged.cleanup()

    payload = json.loads(next(event.text for event in events if event.event == "message"))
    assert payload["prompt"] == "exact provider prompt"
    assert payload["code"] == 0
    assert json.loads(payload["output"])["status"] == "ok"
    assert handled == ["status"]
    assert all(staged.invocation_gate.ready_line not in event.text for event in events)


@pytest.mark.asyncio
async def test_campaign_broker_failure_prevents_provider_prompt_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged = stage_command_mailbox(
        local_stage=tmp_path,
        remote_stage=None,
        episode_id="episode",
        task_id="task",
        turn_id="turn",
    )
    assert staged.invocation_gate is not None
    launched = tmp_path / "provider-launched"
    provider_script = f"from pathlib import Path; Path({str(launched)!r}).write_text('bad')"
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="": type(
        "Readiness",
        (),
        {"installed": True, "authenticated": True, "path_state": "resolved"},
    )()
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [sys.executable, "-c", provider_script],
    )
    occupied = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    occupied.bind(staged.invocation_gate.socket_path)
    try:
        events = [
            event
            async for event in launcher.stream(
                "codex",
                "must not be delivered",
                cwd=tmp_path,
                capability="orchestrate",
                invocation_gate=staged.invocation_gate,
            )
        ]
    finally:
        occupied.close()
        Path(staged.invocation_gate.socket_path).unlink(missing_ok=True)
        staged.cleanup()

    assert not launched.exists()
    assert any(
        event.event == "error" and "socket path is already occupied" in event.text
        for event in events
    )


@pytest.mark.asyncio
async def test_campaign_broker_peer_inspection_failure_prevents_provider_prompt_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged = stage_command_mailbox(
        local_stage=tmp_path,
        remote_stage=None,
        episode_id="episode",
        task_id="task",
        turn_id="turn",
    )
    assert staged.invocation_gate is not None
    unsupported_broker = tmp_path / "unsupported-broker.py"
    unsupported_broker.write_text(
        staged_command_broker_source()
        .replace('if sys.platform.startswith("linux")', "if False")
        .replace('if sys.platform == "darwin"', "if False"),
        encoding="utf-8",
    )
    gate = replace(staged.invocation_gate, broker_path=str(unsupported_broker))
    launched = tmp_path / "provider-launched"
    provider_script = f"from pathlib import Path; Path({str(launched)!r}).write_text('bad')"
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="": type(
        "Readiness",
        (),
        {"installed": True, "authenticated": True, "path_state": "resolved"},
    )()
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [sys.executable, "-c", provider_script],
    )

    try:
        events = [
            event
            async for event in launcher.stream(
                "codex",
                "must not be delivered",
                cwd=tmp_path,
                capability="orchestrate",
                invocation_gate=gate,
            )
        ]
    finally:
        staged.cleanup()

    assert not launched.exists()
    assert any(
        event.event == "error" and "cannot authenticate Unix-socket peers" in event.text
        for event in events
    )


@pytest.mark.asyncio
async def test_stream_drains_oversized_jsonl_provider_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeStdin:
        def __init__(self) -> None:
            self.data = bytearray()

        def write(self, value: bytes) -> None:
            self.data.extend(value)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            self.data = (
                json.dumps({"type": "item", "item": {"text": "x" * 100_000}}) + "\n"
            ).encode()
            self.offset = 0

        async def read(self, size):
            if self.offset == len(self.data):
                return b""
            chunk = self.data[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    class FakeStderr:
        async def read(self, _size):
            return b""

    class FakeProcess:
        stdin = FakeStdin()
        stdout = FakeStdout()
        stderr = FakeStderr()
        returncode = 0

        async def wait(self):
            return self.returncode

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["limit"] = kwargs["limit"]
        captured["command"] = args
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    launcher = AgentLauncher()
    launcher._MAX_EVENT_BYTES = 1024
    launcher.readiness = lambda provider, host="": type(
        "Readiness", (), {"installed": True, "authenticated": True}
    )()

    events = [
        event
        async for event in launcher.stream(
            "codex", "prompt", cwd=Path("/tmp"), capability="scratch_patch"
        )
    ]

    assert captured["limit"] == AgentLauncher._STREAM_LIMIT
    assert "prompt" not in captured["command"]
    assert FakeProcess.stdin.data == b"prompt"
    assert any(
        event.event == "raw" and "Omitted oversized provider event" in event.text
        for event in events
    )
    assert events[-1].event == "done"
    exit_evidence = json.loads(
        next(event.text for event in events if event.event == "provider_exit")
    )
    assert exit_evidence == {
        "event_counts": {"raw": 1},
        "explicit_terminal_event": False,
        "return_code": 0,
    }


@pytest.mark.asyncio
async def test_stream_drains_large_output_while_feeding_large_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider_script = (
        "import json\n"
        "import sys\n"
        'sys.stdout.write(json.dumps({"type": "item", "item": '
        '{"text": "x" * (2 * 1024 * 1024)}}) + "\\n")\n'
        "sys.stdout.flush()\n"
        "prompt = sys.stdin.buffer.read()\n"
        'print(json.dumps({"type": "item", "item": '
        '{"text": f"received={len(prompt)}"}}), flush=True)\n'
    )
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="": type(
        "Readiness", (), {"installed": True, "authenticated": True}
    )()
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [sys.executable, "-c", provider_script],
    )
    prompt = "p" * (2 * 1024 * 1024)

    async def collect_events():
        return [
            event
            async for event in launcher.stream(
                "codex",
                prompt,
                cwd=tmp_path,
                capability="scratch_patch",
            )
        ]

    events = await asyncio.wait_for(collect_events(), timeout=10)

    messages = [event.text for event in events if event.event == "message"]
    assert len(messages[0]) == 2 * 1024 * 1024
    assert messages[-1] == f"received={len(prompt.encode())}"
    assert events[-1].event == "done"
    exit_evidence = json.loads(
        next(event.text for event in events if event.event == "provider_exit")
    )
    assert exit_evidence["event_counts"] == {"message": 2}
    assert exit_evidence["explicit_terminal_event"] is False
    assert exit_evidence["return_code"] == 0


@pytest.mark.asyncio
async def test_stream_records_explicit_terminal_provider_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider_script = (
        'import json\nprint(json.dumps({"type": "result", "result": "Finished."}), flush=True)\n'
    )
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="": type(
        "Readiness", (), {"installed": True, "authenticated": True}
    )()
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [sys.executable, "-c", provider_script],
    )

    events = [
        event
        async for event in launcher.stream(
            "claude", "prompt", cwd=tmp_path, capability="scratch_patch"
        )
    ]

    evidence = json.loads(next(event.text for event in events if event.event == "provider_exit"))
    assert evidence == {
        "event_counts": {"answer": 1},
        "explicit_terminal_event": True,
        "return_code": 0,
    }
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_stream_records_nonzero_provider_exit_before_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="": type(
        "Readiness", (), {"installed": True, "authenticated": True}
    )()
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [
            sys.executable,
            "-c",
            'import sys; print("trace", flush=True); sys.exit(7)',
        ],
    )

    events = [
        event
        async for event in launcher.stream(
            "codex", "prompt", cwd=tmp_path, capability="scratch_patch"
        )
    ]

    exit_index = next(index for index, event in enumerate(events) if event.event == "provider_exit")
    evidence = json.loads(events[exit_index].text)
    assert evidence == {
        "event_counts": {"raw": 1},
        "explicit_terminal_event": False,
        "return_code": 7,
    }
    assert events[exit_index + 1].event == "error"
    assert events[exit_index + 1].text == "codex exited 7."


@pytest.mark.asyncio
async def test_stream_reuses_capability_and_invalidates_it_after_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = AgentLauncher()
    probes = 0

    def probe(provider: str, *, host: str, binary: str | None) -> ProviderReadiness:
        nonlocal probes
        probes += 1
        return ProviderReadiness(
            provider=provider,
            installed=True,
            authenticated=True,
            binary_path=binary,
            path_state="resolved",
        )

    monkeypatch.setattr(launcher, "_readiness_uncached", probe)
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [sys.executable, "-c", "raise SystemExit(7)"],
    )
    binary = "/opt/agents/codex"
    launcher.readiness("codex", binary=binary)

    events = [
        event
        async for event in launcher.stream(
            "codex",
            "prompt",
            cwd=tmp_path,
            binary=binary,
            capability="scratch_patch",
        )
    ]

    assert probes == 1
    assert events[-1].event == "error"
    launcher.readiness("codex", binary=binary)
    assert probes == 2


@pytest.mark.asyncio
async def test_stream_launches_an_authenticated_discovered_path_without_manual_recording(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, **_kwargs: ProviderReadiness(
        provider=provider,
        installed=True,
        authenticated=True,
        binary_path="/home/agent/.local/bin/codex",
        path_state="unconfigured",
    )
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [
            sys.executable,
            "-c",
            'import json; print(json.dumps({"type": "result", "result": "Finished."}), flush=True)',
        ],
    )

    events = [
        event
        async for event in launcher.stream(
            "codex", "prompt", cwd=tmp_path, capability="scratch_patch"
        )
    ]

    assert not any(event.event == "error" for event in events)
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_cold_readiness_does_not_block_stream_event_loop(tmp_path: Path) -> None:
    launcher = AgentLauncher()
    entered = threading.Event()
    release = threading.Event()

    def readiness(provider: str, *, host: str = "") -> ProviderReadiness:
        entered.set()
        assert release.wait(timeout=2)
        return ProviderReadiness(
            provider=provider,
            installed=False,
            authenticated=False,
            reason=f"{host or 'local'} unavailable",
        )

    launcher.readiness = readiness
    stream = launcher.stream(
        "codex", "prompt", cwd=tmp_path, host="slow.example", capability="scratch_patch"
    )
    first_event = asyncio.create_task(anext(stream))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
    finally:
        release.set()

    assert (await first_event).event == "error"
    await stream.aclose()


@pytest.mark.asyncio
async def test_stream_cancellation_during_stdin_drain_reaps_and_detaches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = AgentLauncher()
    launcher.readiness = lambda provider, host="": type(
        "Readiness", (), {"installed": True, "authenticated": True}
    )()
    monkeypatch.setattr(
        launcher,
        "_command",
        lambda *args, **kwargs: [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
    )
    control = AgentProcessControl()
    stream = launcher.stream(
        "codex",
        "p" * (2 * 1024 * 1024),
        cwd=tmp_path,
        control=control,
        capability="scratch_patch",
    )
    runtime_event = await anext(stream)
    assert runtime_event.event == "runtime"
    next_event = asyncio.create_task(anext(stream))
    process = None
    try:
        for _ in range(200):
            process = control._process
            if (
                process is not None
                and process.stdin is not None
                and process.stdin.transport.get_write_buffer_size() > 0
            ):
                break
            await asyncio.sleep(0.01)
        assert process is not None
        assert process.stdin is not None
        assert process.stdin.transport.get_write_buffer_size() > 0

        next_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(next_event, timeout=5)

        assert process.returncode is not None
        assert process.stdin.is_closing()
        assert control._process is None
    finally:
        if not next_event.done():
            next_event.cancel()
            with suppress(asyncio.CancelledError):
                await next_event
        await stream.aclose()
        if process is not None and process.returncode is None:
            await AgentProcessControl._terminate(process)


def test_codex_failure_event_surfaces_provider_error() -> None:
    event = _decoded_event(
        "codex",
        json.dumps(
            {
                "type": "turn.failed",
                "error": {"message": "Invalid response schema."},
            }
        ),
    )

    assert event.event == "error"
    assert event.text == "Invalid response schema."


def test_only_the_final_assistant_message_is_an_answer() -> None:
    """A chat reply must never be confused with a reasoning or tool trace."""

    def codex(payload: dict) -> AgentEvent:
        return _decoded_event("codex", json.dumps(payload))

    reply = codex(
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Because X."}}
    )
    assert (reply.event, reply.text) == ("answer", "Because X.")

    for item_type in ("reasoning", "command_execution", "todo_list"):
        trace = codex({"type": "item.completed", "item": {"type": item_type, "text": "noise"}})
        assert trace.event == "message", item_type

    # A partial message still in flight is not the reply yet.
    partial = codex({"type": "item.started", "item": {"type": "agent_message", "text": "Bec"}})
    assert partial.event == "message"

    claude = _decoded_event("claude", json.dumps({"type": "result", "result": "Because X."}))
    assert (claude.event, claude.text) == ("answer", "Because X.")


def test_claude_session_limit_result_is_a_terminal_error() -> None:
    text = "You've hit your session limit · resets 8:50pm (UTC)"
    event = _decoded_event(
        "claude",
        json.dumps(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "result": text,
            }
        ),
    )

    assert (event.event, event.text) == ("error", text)
    assert _decoded_step(
        "claude", json.dumps({"type": "result", "result": "done"})
    ).explicit_terminal


def test_codex_resume_keeps_the_write_permission_its_surface_was_given() -> None:
    """`codex exec resume` has no --sandbox flag and defaults to read-only.

    A resumed run still owes RCP a patch file, so the mode travels as config.
    """

    command = AgentLauncher._command(
        "codex",
        "reread current pointers",
        cwd=Path("/project/.research"),
        model="gpt-test",
        reasoning="medium",
        session_id="019f0000-0000-7000-8000-000000000000",
        read_dirs=[Path("/project/repo-a")],
        capability="scratch_patch",
    )

    assert command[:4] == ["codex", "exec", "resume", "--json"]
    assert "--sandbox" not in command
    assert 'sandbox_mode="workspace-write"' in command
    assert 'web_search="live"' in command
    assert "sandbox_workspace_write.network_access=true" in command
    # No --cd on resume: the writable root is the process working directory.
    assert "--cd" not in command
    assert "--add-dir" not in command
    assert command[-2:] == ["019f0000-0000-7000-8000-000000000000", "-"]


def test_codex_new_session_writes_only_into_its_scratch_folder() -> None:
    scratch = Path("/data/run-stage/operation")
    command = AgentLauncher._command(
        "codex",
        "write patch.json",
        cwd=scratch,
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/project/repo-a"), Path("/project/repo-b")],
        capability="scratch_patch",
    )

    assert command[:3] == ["codex", "exec", "--json"]
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("--cd") + 1] == str(scratch)
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert 'web_search="live"' in command
    assert "sandbox_workspace_write.network_access=true" in command
    assert 'approval_policy="never"' in command
    assert "--output-schema" not in command
    assert command[-1] == "-"
    # Truth repositories are pointers the agent reads; none of them may become a
    # writable sandbox root.
    assert "--add-dir" not in command
    assert not any("/project/repo-" in argument for argument in command)


def test_codex_discuss_keeps_networked_writes_inside_conversation_scratch() -> None:
    scratch = Path("/data/conversations/chat-1")
    command = AgentLauncher._command(
        "codex",
        "answer the question",
        cwd=scratch,
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/project/repo-a")],
        capability="discuss",
    )

    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("--cd") + 1] == str(scratch)
    assert 'approval_policy="never"' in command
    assert 'web_search="live"' in command
    assert "sandbox_workspace_write.network_access=true" in command
    assert "--add-dir" not in command
    assert not any("/project/repo-a" in argument for argument in command)


def test_codex_new_read_only_session_has_no_workspace_write_config() -> None:
    research_dir = Path("/project/.research")
    command = AgentLauncher._command(
        "codex",
        "review the paper introduction",
        cwd=research_dir,
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/project/repo-a")],
        capability="paper_readonly",
    )

    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--cd") + 1] == str(research_dir)
    assert 'web_search="live"' in command
    assert "sandbox_workspace_write.network_access=true" not in command
    assert 'approval_policy="never"' in command


def test_codex_work_uses_exact_project_permission_profile() -> None:
    stage = "/data/chat-stage"
    scope = _project_write_scope(
        capability="work_auto",
        stage=stage,
        repository_paths=["/project/repo-a"],
    )
    command = AgentLauncher._command(
        "codex",
        "run the experiment",
        cwd=Path(stage),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/data/chat-stage/inputs")],
        write_dirs=[Path("/project/repo-a"), Path("/project/repo-a")],
        write_scope=scope,
        capability="work_auto",
        provider_version="0.147.0",
    )

    assert "--sandbox" not in command
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert 'web_search="live"' in command
    assert not any(item.startswith("sandbox_mode=") for item in command)
    assert not any(item.startswith("approval_policy=") for item in command)
    assert not any(item.startswith("approvals_reviewer=") for item in command)
    assert 'default_permissions="rcp_project"' in command
    assert (
        'permissions={rcp_project={workspace_roots={"/data/chat-stage"=true,'
        '"/project/repo-a"=true},filesystem={":root"="read",":workspace_roots"='
        '{"."="write",".research"="read"}},network={enabled=true}}}'
    ) in command
    assert command[command.index("--cd") + 1] == stage
    assert "--add-dir" not in command
    assert command[-1] == "-"


def test_codex_profile_keeps_staged_canonical_state_readable() -> None:
    """The canonical `.research` RCP stages as an agent input must stay readable.

    `_stage_context_paths` hands every graph agent `<state repo>/.research/graph.json`
    and `research.md`, and the task contract orders it to read both first. Codex reads
    a `deny` policy as no access at all, so denying `.research` revoked those reads and
    the agent could only report a Blocker. Invariant 4 protects canonical state from
    writes, which `read` already does.
    """
    state_repository = "/project/repo-a"
    scope = _project_write_scope(
        capability="work_auto",
        stage="/data/chat-stage",
        repository_paths=[state_repository],
    )
    command = AgentLauncher._command(
        "codex",
        "run the experiment",
        cwd=Path("/data/chat-stage"),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[],
        write_dirs=[Path(state_repository)],
        write_scope=scope,
        capability="work_auto",
        provider_version="0.138.0",
    )
    permission_profile = next(item for item in command if item.startswith("permissions={"))

    # The staged pointer and the policy governing it, checked against each other.
    assert f"{state_repository}/.research" in scope.protected_write_paths
    assert '".research"="read"' in permission_profile
    assert "deny" not in permission_profile


def test_codex_orchestrate_uses_only_its_resolved_project_roots() -> None:
    stage = "/data/campaign-stage"
    scope = _project_write_scope(
        capability="orchestrate",
        stage=stage,
        repository_paths=["/project/repo-b"],
    )
    command = AgentLauncher._command(
        "codex",
        "orchestrate the campaign",
        cwd=Path(stage),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/project/repo-a")],
        write_dirs=[Path("/project/repo-b")],
        write_scope=scope,
        capability="orchestrate",
        provider_version="0.138.0",
    )

    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert "--sandbox" not in command
    assert not any(item.startswith("sandbox_mode=") for item in command)
    assert not any(item.startswith("approval_policy=") for item in command)
    permission_profile = next(item for item in command if item.startswith("permissions={"))
    assert '"/data/campaign-stage"=true' in permission_profile
    assert '"/project/repo-b"=true' in permission_profile
    assert "/project/repo-a" not in permission_profile
    assert '":root"="read"' in permission_profile
    assert '".research"="read"' in permission_profile
    assert "network={enabled=true}" in permission_profile
    assert command[-1] == "-"


def test_codex_graph_only_orchestrate_accepts_no_repository_write_roots() -> None:
    stage = "/data/branch-merge-stage/workspace"
    scope = _project_write_scope(
        capability="orchestrate",
        stage=stage,
        repository_paths=[],
    )
    command = AgentLauncher._command(
        "codex",
        "merge the graph branch",
        cwd=Path(stage),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/data/branch-merge-stage/inputs")],
        write_dirs=[],
        write_scope=scope,
        capability="orchestrate",
        provider_version="0.147.0",
    )

    permission_profile = next(item for item in command if item.startswith("permissions={"))
    assert f"{json.dumps(stage)}=true" in permission_profile
    assert "/project/" not in permission_profile
    assert command[command.index("--cd") + 1] == stage
    assert "--add-dir" not in command


def test_codex_work_resume_keeps_the_exact_project_permission_profile() -> None:
    session_id = "019f0000-0000-7000-8000-000000000002"
    stage = "/data/chat-stage"
    scope = _project_write_scope(
        capability="work_auto",
        stage=stage,
        repository_paths=["/project/repo-a"],
    )
    command = AgentLauncher._command(
        "codex",
        "continue the operational turn",
        cwd=Path(stage),
        model=None,
        reasoning=None,
        session_id=session_id,
        read_dirs=[],
        write_dirs=[Path("/project/repo-a")],
        write_scope=scope,
        capability="work_auto",
        provider_version="0.147.0",
    )

    assert command[:4] == ["codex", "exec", "resume", "--json"]
    assert "--sandbox" not in command
    assert not any(item.startswith("sandbox_mode=") for item in command)
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert 'web_search="live"' in command
    assert not any(item.startswith("approval_policy=") for item in command)
    assert not any(item.startswith("approvals_reviewer=") for item in command)
    assert 'default_permissions="rcp_project"' in command
    permission_profile = next(item for item in command if item.startswith("permissions={"))
    assert '"/data/chat-stage"=true' in permission_profile
    assert '"/project/repo-a"=true' in permission_profile
    assert '".research"="read"' in permission_profile
    assert "--cd" not in command
    assert command[-2:] == [session_id, "-"]


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("capability", ["work_auto", "orchestrate"])
def test_work_like_provider_commands_require_a_resolved_project_scope(
    provider: str,
    capability: Literal["work_auto", "orchestrate"],
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^{capability} launch requires a resolved project write scope$",
    ):
        AgentLauncher._command(
            provider,
            "operate on the project",
            cwd=Path("/data/task-stage"),
            model=None,
            reasoning=None,
            session_id=None,
            read_dirs=[],
            write_dirs=[Path("/project/repo-a")],
            capability=capability,
            provider_version="99.0.0",
        )


@pytest.mark.parametrize(
    ("provider", "provider_version", "required_version"),
    [
        ("codex", "0.137.9", "0.138.0"),
        ("codex", None, "0.138.0"),
        ("claude", "2.1.232", "2.1.233"),
        ("claude", None, "2.1.233"),
    ],
)
def test_work_like_provider_commands_reject_versions_without_scope_enforcement(
    provider: str,
    provider_version: str | None,
    required_version: str,
) -> None:
    scope = _project_write_scope(
        capability="work_auto",
        stage="/data/task-stage",
        repository_paths=["/project/repo-a"],
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"cannot enforce the declared project write roots; RCP requires "
            rf"{required_version} or newer$"
        ),
    ):
        AgentLauncher._command(
            provider,
            "operate on the project",
            cwd=Path("/data/task-stage"),
            model=None,
            reasoning=None,
            session_id=None,
            read_dirs=[],
            write_dirs=[Path("/project/repo-a")],
            write_scope=scope,
            capability="work_auto",
            provider_version=provider_version,
        )


def test_codex_read_only_resume_relies_on_pinned_native_session() -> None:
    session_id = "019f0000-0000-7000-8000-000000000001"
    command = AgentLauncher._command(
        "codex",
        "continue reviewing the introduction",
        cwd=Path("/project/.research"),
        model=None,
        reasoning=None,
        session_id=session_id,
        read_dirs=[],
        capability="paper_readonly",
    )

    assert command[:4] == ["codex", "exec", "resume", "--json"]
    assert "--sandbox" not in command
    assert "--cd" not in command
    assert 'sandbox_mode="read-only"' in command
    assert 'web_search="live"' in command
    assert "sandbox_workspace_write.network_access=true" not in command
    assert command[-2:] == [session_id, "-"]


def test_claude_scratch_patch_preauthorizes_only_native_web_retrieval() -> None:
    command = AgentLauncher._command(
        "claude",
        "write patch.json",
        cwd=Path("/data/run-stage/operation"),
        model="claude-test",
        reasoning="high",
        session_id=None,
        read_dirs=[
            Path("/project/repo-a"),
            Path("/project/repo-a"),
            Path("/sessions"),
            Path("/project/repo-a"),
        ],
        capability="scratch_patch",
    )

    assert command[:2] == ["claude", "--print"]
    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    assert command[command.index("--model") + 1] == "claude-test"
    assert command[command.index("--effort") + 1] == "high"
    assert "--tools" not in command
    allowed_index = command.index("--allowedTools")
    assert command[allowed_index + 1 : allowed_index + 3] == ["WebSearch", "WebFetch"]
    assert "--json-schema" not in command
    # One --add-dir per distinct directory: duplicates used to blow the argv limit.
    add_dirs = [command[index + 1] for index, item in enumerate(command) if item == "--add-dir"]
    assert add_dirs == ["/project/repo-a", "/sessions"]


def test_claude_read_only_command_keeps_plan_permission_mode() -> None:
    command = AgentLauncher._command(
        "claude",
        "review the paper introduction",
        cwd=Path("/project/.research"),
        model=None,
        reasoning=None,
        session_id="paper-session",
        read_dirs=[Path("/project/.research")],
        capability="paper_readonly",
    )

    assert command[command.index("--permission-mode") + 1] == "plan"
    assert "acceptEdits" not in command
    allowed_index = command.index("--allowedTools")
    assert command[allowed_index + 1 : allowed_index + 3] == ["WebSearch", "WebFetch"]
    assert command[command.index("--resume") + 1] == "paper-session"


def test_claude_work_uses_exact_sandbox_and_tool_allowlists() -> None:
    stage = "/data/chat-stage"
    scope = _project_write_scope(
        capability="work_auto",
        stage=stage,
        repository_paths=["/project/repo-a"],
    )
    command = AgentLauncher._command(
        "claude",
        "run the experiment",
        cwd=Path(stage),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/data/chat-stage/inputs")],
        write_dirs=[Path("/project/repo-a"), Path("/project/repo-a")],
        write_scope=scope,
        capability="work_auto",
        provider_version="2.1.233",
    )

    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert "bypassPermissions" not in command
    assert "--allowedTools" not in command
    assert command[command.index("--setting-sources") + 1] == ""
    settings = json.loads(command[command.index("--settings") + 1])
    assert settings == {
        "disableAllHooks": True,
        "permissions": {
            "defaultMode": "dontAsk",
            "disableAutoMode": "disable",
            "disableBypassPermissionsMode": "disable",
            "allow": [
                "Bash",
                "WebSearch",
                "WebFetch",
                "Edit(//data/chat-stage/**)",
                "Write(//data/chat-stage/**)",
                "Edit(//project/repo-a/**)",
                "Write(//project/repo-a/**)",
            ],
            "deny": [
                "Edit(//project/repo-a/.research/**)",
                "Write(//project/repo-a/.research/**)",
            ],
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "allowWrite": ["/data/chat-stage", "/project/repo-a"],
                "denyWrite": ["/project/repo-a/.research"],
            },
            "network": {"allowedDomains": ["*"]},
        },
    }
    assert command[command.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    add_dirs = [command[index + 1] for index, item in enumerate(command) if item == "--add-dir"]
    assert add_dirs == ["/data/chat-stage/inputs", "/project/repo-a"]


def test_claude_orchestrate_allows_only_its_resolved_project_roots() -> None:
    stage = "/data/campaign-stage"
    scope = _project_write_scope(
        capability="orchestrate",
        stage=stage,
        repository_paths=["/project/repo-b"],
    )
    command = AgentLauncher._command(
        "claude",
        "orchestrate the campaign",
        cwd=Path(stage),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/project/repo-a")],
        write_dirs=[Path("/project/repo-b")],
        write_scope=scope,
        capability="orchestrate",
        provider_version="2.1.234",
    )

    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert "bypassPermissions" not in command
    assert "--allowedTools" not in command
    settings = json.loads(command[command.index("--settings") + 1])
    assert settings["sandbox"]["filesystem"] == {
        "allowWrite": ["/data/campaign-stage", "/project/repo-b"],
        "denyWrite": ["/project/repo-b/.research"],
    }
    assert settings["permissions"]["deny"] == [
        "Edit(//project/repo-b/.research/**)",
        "Write(//project/repo-b/.research/**)",
    ]
    assert "Edit(//project/repo-a/**)" not in settings["permissions"]["allow"]
    add_dirs = [command[index + 1] for index, item in enumerate(command) if item == "--add-dir"]
    assert add_dirs == ["/project/repo-a", "/project/repo-b"]


def test_claude_graph_only_orchestrate_accepts_no_repository_write_roots() -> None:
    stage = "/data/branch-merge-stage/workspace"
    inputs = "/data/branch-merge-stage/inputs"
    scope = _project_write_scope(
        capability="orchestrate",
        stage=stage,
        repository_paths=[],
    )
    command = AgentLauncher._command(
        "claude",
        "merge the graph branch",
        cwd=Path(stage),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path(inputs)],
        write_dirs=[],
        write_scope=scope,
        capability="orchestrate",
        provider_version="2.1.234",
    )

    settings = json.loads(command[command.index("--settings") + 1])
    assert settings["sandbox"]["filesystem"] == {
        "allowWrite": [stage],
        "denyWrite": [],
    }
    assert settings["permissions"]["deny"] == []
    add_dirs = [command[index + 1] for index, item in enumerate(command) if item == "--add-dir"]
    assert add_dirs == [inputs]


def test_claude_work_resume_keeps_the_native_session_and_exact_scope() -> None:
    session_id = "claude-experiment-episode-session"
    stage = "/data/chat-stage"
    scope = _project_write_scope(
        capability="work_auto",
        stage=stage,
        repository_paths=["/project/repo-a"],
    )
    command = AgentLauncher._command(
        "claude",
        "continue the bounded experiment turn",
        cwd=Path(stage),
        model=None,
        reasoning=None,
        session_id=session_id,
        read_dirs=[Path("/data/chat-stage/inputs")],
        write_dirs=[Path("/project/repo-a")],
        write_scope=scope,
        capability="work_auto",
        provider_version="2.1.233",
    )

    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert "bypassPermissions" not in command
    assert "--allowedTools" not in command
    settings = json.loads(command[command.index("--settings") + 1])
    assert settings["sandbox"]["filesystem"]["allowWrite"] == [
        "/data/chat-stage",
        "/project/repo-a",
    ]
    assert settings["sandbox"]["filesystem"]["denyWrite"] == ["/project/repo-a/.research"]
    assert command[command.index("--resume") + 1] == session_id


def test_claude_discuss_keeps_scratch_writable_without_auto_mode() -> None:
    command = AgentLauncher._command(
        "claude",
        "answer the question",
        cwd=Path("/data/chat-stage"),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[Path("/data/chat-stage/inputs")],
        capability="discuss",
    )

    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    allowed_index = command.index("--allowedTools")
    assert command[allowed_index + 1 : allowed_index + 3] == ["WebSearch", "WebFetch"]
    assert "auto" not in command


def test_remote_provider_command_uses_login_shell_path() -> None:
    command = AgentLauncher._remote_login_command(["codex", "login", "status"])

    assert shlex.split(command) == ["bash", "-lic", "codex login status"]


def test_remote_provider_launch_keeps_the_recorded_absolute_argv_zero() -> None:
    provider_command = AgentLauncher._command(
        "claude",
        "prompt",
        binary="/opt/agents/claude",
        cwd=Path("/srv/project/.research"),
        model=None,
        reasoning=None,
        session_id=None,
        read_dirs=[],
        capability="scratch_patch",
    )
    command = AgentLauncher._remote_login_command(
        provider_command,
        cwd=Path("/srv/project/.research"),
    )

    assert "/opt/agents/claude" in shlex.split(command)[2]
    assert "cd /srv/project/.research" in shlex.split(command)[2]


@pytest.mark.asyncio
async def test_stream_refuses_a_stale_recorded_path_before_subprocess_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def unexpected_launch(*_args, **_kwargs):
        raise AssertionError("a stale provider path must fail before launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_launch)

    events = [
        event
        async for event in AgentLauncher().stream(
            "codex",
            "prompt",
            cwd=tmp_path,
            binary="/missing/recorded/codex",
            capability="scratch_patch",
        )
    ]

    assert [event.event for event in events] == ["error"]
    assert "does not exist" in events[0].text


@pytest.mark.asyncio
async def test_stream_refuses_a_denied_recorded_path_before_subprocess_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o644)

    async def unexpected_launch(*_args, **_kwargs):
        raise AssertionError("a denied provider path must fail before launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_launch)

    events = [
        event
        async for event in AgentLauncher().stream(
            "codex",
            "prompt",
            cwd=tmp_path,
            binary=str(binary),
            capability="scratch_patch",
        )
    ]

    assert [event.event for event in events] == ["error"]
    assert "not executable" in events[0].text


def test_remote_provider_command_records_a_killable_process_group() -> None:
    command = AgentLauncher._remote_login_command(
        ["codex", "exec", "prompt"],
        pid_file="/tmp/rcp-run.operation/agent.pid",
    )
    outer = shlex.split(command)

    assert outer[:2] == ["bash", "-lic"]
    assert "setsid sh -c" in outer[2]
    assert "agent.pid" in outer[2]
    assert "exec codex exec prompt" in outer[2]


def test_remote_provider_pid_wrapper_changes_directory_before_exec() -> None:
    command = AgentLauncher._remote_login_command(
        ["codex", "exec", "prompt"],
        pid_file="/tmp/rcp-run.operation/agent.pid",
        cwd=Path("/tmp/rcp-run.operation/workspace"),
    )
    child = shlex.split(command)[2]

    assert "cd /tmp/rcp-run.operation/workspace && exec codex exec prompt" in child
    assert "exec cd" not in child


@pytest.mark.asyncio
async def test_process_control_terminates_only_its_process_group() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        start_new_session=True,
    )
    control = AgentProcessControl()
    control.attach(process)

    control.request_pause()
    await asyncio.wait_for(process.wait(), timeout=2)

    assert process.returncode is not None
    assert process.returncode != 0
