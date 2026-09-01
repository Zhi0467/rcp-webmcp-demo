from __future__ import annotations

import json
from pathlib import Path

from rcp.providers import (
    ProviderRuntime,
    ProviderRuntimeStep,
    ProviderStreamEvent,
    ProviderTurn,
    ProviderTurnRequest,
    ProviderUsage,
    _codex_permission_profile,
    _require_project_write_scope,
    _require_provider_version,
)

CODEX_APP_SERVER_RUNTIME_ID = "codex.app-server-stdio.v1"
_INITIALIZE_ID = 1
_CONFIG_READ_ID = 2
_THREAD_ID = 3
_TURN_ID = 4
_HOOK_EVENTS = (
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "SubagentStart",
    "SubagentStop",
    "Stop",
)


class CodexAppServerRuntime(ProviderRuntime):
    """Fresh-stdio Codex app-server transport for one RCP provider turn."""

    id = CODEX_APP_SERVER_RUNTIME_ID

    def turn(self, request: ProviderTurnRequest) -> ProviderTurn:
        return _CodexAppServerTurn(request)


class _CodexAppServerTurn(ProviderTurn):
    close_input_after_initial = False
    requires_protocol_completion = True
    initial_input_delivers_prompt = False

    def __init__(self, request: ProviderTurnRequest) -> None:
        _require_provider_version(
            provider="Codex app-server",
            actual=request.provider_version,
            minimum=(0, 149, 0),
        )
        self._request = request
        self._phase = "initialize"
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self._usage: ProviderUsage | None = None
        self._work_like = request.capability in {"work_auto", "orchestrate"}
        self.command = self._command(request)

    @staticmethod
    def _command(request: ProviderTurnRequest) -> list[str]:
        command = [
            request.binary,
            "app-server",
            "--stdio",
            "--disable",
            "apps",
            "--disable",
            "multi_agent",
            "--disable",
            "plugins",
            "--config",
            'web_search="live"',
        ]
        if request.capability in {"work_auto", "orchestrate"}:
            scope = _require_project_write_scope(
                request.write_scope,
                capability=request.capability,
                write_dirs=request.write_dirs,
            )
            command.extend(
                [
                    "--config",
                    'default_permissions="rcp_project"',
                    "--config",
                    _codex_permission_profile(scope),
                ]
            )
        else:
            if request.write_scope is not None:
                raise ValueError(
                    f"capability {request.capability!r} cannot carry a project write scope"
                )
            sandbox = "read-only" if request.capability == "paper_readonly" else "workspace-write"
            command.extend(
                [
                    "--config",
                    'approval_policy="never"',
                    "--config",
                    f'sandbox_mode="{sandbox}"',
                ]
            )
            if request.capability != "paper_readonly":
                command.extend(["--config", "sandbox_workspace_write.network_access=true"])
        return command

    def initial_input(self) -> bytes:
        return _rpc_bytes(
            {
                "id": _INITIALIZE_ID,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "rcp", "title": "RCP", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                },
            }
        )

    def receive_line(self, line: str) -> ProviderRuntimeStep:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return ProviderRuntimeStep(events=(ProviderStreamEvent(event="raw", text=line),))
        if not isinstance(value, dict):
            return self._protocol_error("Codex app-server emitted a non-object JSON message.")
        if "id" in value and "method" in value:
            return self._unexpected_server_request(value)
        if "id" in value:
            return self._response(value)
        method = value.get("method")
        params = value.get("params")
        if not isinstance(method, str) or not isinstance(params, dict):
            return ProviderRuntimeStep(events=(ProviderStreamEvent(event="raw", text=line),))
        return self._notification(method, params)

    def _response(self, value: dict[str, object]) -> ProviderRuntimeStep:
        request_id = value.get("id")
        if request_id not in {_INITIALIZE_ID, _CONFIG_READ_ID, _THREAD_ID, _TURN_ID}:
            return self._protocol_error(
                f"Codex app-server returned an unknown response id {request_id!r}."
            )
        error = value.get("error")
        if error is not None:
            return self._protocol_error(
                f"Codex app-server request {request_id} failed: {_error_text(error)}"
            )
        result = value.get("result")
        if not isinstance(result, dict):
            return self._protocol_error(
                f"Codex app-server response {request_id} has no object result."
            )
        if request_id == _INITIALIZE_ID:
            if self._phase != "initialize":
                return self._protocol_error("Codex app-server initialized out of order.")
            self._phase = "config"
            return ProviderRuntimeStep(
                outgoing=(
                    _rpc_bytes({"method": "initialized", "params": {}}),
                    _rpc_bytes(
                        {
                            "id": _CONFIG_READ_ID,
                            "method": "config/read",
                            "params": {
                                "cwd": str(self._request.cwd),
                                "includeLayers": False,
                            },
                        }
                    ),
                )
            )
        if request_id == _CONFIG_READ_ID:
            if self._phase != "config":
                return self._protocol_error("Codex app-server config arrived out of order.")
            try:
                config = _containment_config(result.get("config"))
            except ValueError as exc:
                return self._protocol_error(str(exc))
            self._phase = "thread"
            method = "thread/resume" if self._request.session_id else "thread/start"
            return ProviderRuntimeStep(
                outgoing=(
                    _rpc_bytes(
                        {
                            "id": _THREAD_ID,
                            "method": method,
                            "params": self._thread_params(config),
                        }
                    ),
                )
            )
        if request_id == _THREAD_ID:
            if self._phase != "thread":
                return self._protocol_error("Codex app-server thread arrived out of order.")
            thread = result.get("thread")
            if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
                return self._protocol_error("Codex app-server returned no thread identity.")
            thread_id = str(thread["id"])
            if self._request.session_id and thread_id != self._request.session_id:
                return self._protocol_error(
                    "Codex app-server resumed a different thread than RCP requested."
                )
            enforcement_problem = self._enforcement_problem(result)
            if enforcement_problem is not None:
                return self._protocol_error(enforcement_problem)
            self._thread_id = thread_id
            self._phase = "turn"
            return ProviderRuntimeStep(
                outgoing=(
                    _rpc_bytes(
                        {
                            "id": _TURN_ID,
                            "method": "turn/start",
                            "params": self._turn_params(thread_id),
                        }
                    ),
                ),
                events=(ProviderStreamEvent(event="session", session_id=thread_id),),
                delivers_prompt=True,
            )
        if self._phase != "turn":
            return self._protocol_error("Codex app-server turn arrived out of order.")
        turn = result.get("turn")
        if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
            return self._protocol_error("Codex app-server returned no turn identity.")
        self._turn_id = str(turn["id"])
        self._phase = "running"
        return ProviderRuntimeStep()

    def _thread_params(self, config: dict[str, object]) -> dict[str, object]:
        params: dict[str, object] = {
            "approvalPolicy": "never",
            "config": config,
            "cwd": str(self._request.cwd),
            "developerInstructions": "",
            "model": self._request.model,
            "serviceName": "rcp",
            "threadSource": "rcp",
        }
        if self._request.session_id:
            params["threadId"] = self._request.session_id
        else:
            params["ephemeral"] = False
        if self._work_like:
            params["permissions"] = "rcp_project"
        else:
            params["sandbox"] = (
                "read-only" if self._request.capability == "paper_readonly" else "workspace-write"
            )
        return params

    def _turn_params(self, thread_id: str) -> dict[str, object]:
        params: dict[str, object] = {
            "approvalPolicy": "never",
            "cwd": str(self._request.cwd),
            "effort": self._request.reasoning,
            "input": [{"type": "text", "text": self._request.prompt}],
            "model": self._request.model,
            "threadId": thread_id,
        }
        if not self._work_like:
            params["sandboxPolicy"] = _sandbox_policy(
                self._request.cwd,
                read_only=self._request.capability == "paper_readonly",
            )
        return params

    def _enforcement_problem(self, result: dict[str, object]) -> str | None:
        if result.get("approvalPolicy") != "never":
            return "Codex app-server did not apply RCP's noninteractive approval policy."
        if self._work_like:
            profile = result.get("activePermissionProfile")
            if not isinstance(profile, dict) or profile.get("id") != "rcp_project":
                return "Codex app-server did not activate RCP's exact project permission profile."
            return None
        sandbox = result.get("sandbox")
        expected = "readOnly" if self._request.capability == "paper_readonly" else "workspaceWrite"
        if not isinstance(sandbox, dict) or sandbox.get("type") != expected:
            return "Codex app-server did not apply RCP's requested sandbox."
        return None

    def _notification(
        self,
        method: str,
        params: dict[str, object],
    ) -> ProviderRuntimeStep:
        if method == "thread/tokenUsage/updated":
            usage = params.get("tokenUsage")
            turn_id = params.get("turnId")
            # Usage notifications are thread-scoped, so a resumed thread can
            # report another turn. Keep only this turn's, and never let a later
            # payload without a `last` breakdown erase one RCP already has.
            if (
                isinstance(usage, dict)
                and isinstance(turn_id, str)
                and (self._turn_id is None or turn_id == self._turn_id)
            ):
                reported = _usage_event(usage, turn_id)
                if reported is not None:
                    self._usage = reported
            return ProviderRuntimeStep()
        if method == "item/completed":
            item = params.get("item")
            if not isinstance(item, dict):
                return ProviderRuntimeStep()
            item_type = item.get("type")
            text = item.get("text")
            if item_type == "agentMessage" and isinstance(text, str) and text:
                return ProviderRuntimeStep(events=(ProviderStreamEvent(event="answer", text=text),))
            if item_type == "plan" and isinstance(text, str) and text:
                return ProviderRuntimeStep(
                    events=(ProviderStreamEvent(event="message", text=text),)
                )
            return ProviderRuntimeStep()
        if method == "error" and params.get("willRetry") is not True:
            return self._protocol_error(
                _error_text(params.get("error")) or "Codex app-server turn failed."
            )
        if method != "turn/completed":
            return ProviderRuntimeStep()
        turn = params.get("turn")
        if not isinstance(turn, dict):
            return self._protocol_error("Codex app-server completed without a turn result.")
        turn_id = turn.get("id")
        if self._turn_id is not None and turn_id != self._turn_id:
            return self._protocol_error("Codex app-server completed a different turn.")
        status = turn.get("status")
        if status != "completed":
            detail = _error_text(turn.get("error")) or f"Codex turn ended with status {status!r}."
            return self._protocol_error(detail)
        events = (
            (ProviderStreamEvent(event="raw", usage=self._usage),)
            if self._usage is not None
            else ()
        )
        self._phase = "completed"
        return ProviderRuntimeStep(
            events=events,
            complete=True,
            explicit_terminal=True,
        )

    def _unexpected_server_request(
        self,
        value: dict[str, object],
    ) -> ProviderRuntimeStep:
        method = str(value.get("method") or "unknown")
        response = _rpc_bytes(
            {
                "id": value.get("id"),
                "error": {
                    "code": -32601,
                    "message": "RCP provider turns are unattended and cannot answer requests.",
                },
            }
        )
        return ProviderRuntimeStep(
            outgoing=(response,),
            events=(
                ProviderStreamEvent(
                    event="error",
                    text=(
                        "Codex app-server requested interactive input "
                        f"({method}); RCP stopped the unattended turn."
                    ),
                ),
            ),
            complete=True,
            explicit_terminal=True,
        )

    @staticmethod
    def _protocol_error(message: str) -> ProviderRuntimeStep:
        return ProviderRuntimeStep(
            events=(ProviderStreamEvent(event="error", text=message),),
            complete=True,
            explicit_terminal=True,
        )


def _containment_config(value: object) -> dict[str, object]:
    """Neutralize every user-config channel that can run code or add instructions.

    `codex app-server` has no `--ignore-user-config`, so unlike `codex exec` this
    runtime cannot drop the whole file and must name each capability-bearing key.
    Every key below was accepted by the installed app-server; add to this list
    rather than assuming an unnamed key is inert.

    One gap has no config lever at all: `codex exec --ignore-rules` refuses user
    and project execpolicy `.rules` files, and app-server exposes neither a config
    key nor a feature flag for them, so `.rules` still applies here. Recheck when
    the app-server config surface grows.
    """

    if not isinstance(value, dict):
        raise ValueError("Codex app-server could not inspect its effective configuration.")
    override: dict[str, object] = {
        "agents": {"enabled": False},
        "apps": {"_default": {"enabled": False}},
        # Instruction channels. RCP's staged task contract, not ambient AGENTS.md
        # files or user config prose, supplies this turn's instructions.
        "compact_prompt": None,
        "developer_instructions": "",
        "instructions": None,
        "project_doc_max_bytes": 0,
        # Code-execution channels. `notify` runs an external program when a turn
        # ends, and the environment policy injects variables into every command.
        "notify": [],
        "shell_environment_policy": {},
        "hooks": _disabled_hooks(value.get("hooks")),
    }
    for key in ("apps", "mcp_servers", "plugins"):
        configured = value.get(key)
        if configured is None:
            continue
        if not isinstance(configured, dict):
            raise ValueError(f"Codex app-server reported an unsupported {key} configuration shape.")
        disabled = override.setdefault(key, {})
        assert isinstance(disabled, dict)
        for name in configured:
            if not isinstance(name, str):
                raise ValueError(
                    f"Codex app-server reported an unsupported {key} configuration shape."
                )
            # The entry's own shape is irrelevant: it is being replaced, and
            # skipping a malformed one would leave it enabled for the turn.
            disabled[name] = {"enabled": False}
    if value.get("model_instructions_file") is not None:
        override["model_instructions_file"] = None
    return override


def _disabled_hooks(configured: object) -> dict[str, object]:
    """Empty every hook list, including an event name this RCP does not know."""

    hooks: dict[str, object] = {event: [] for event in _HOOK_EVENTS}
    if isinstance(configured, dict):
        for event, entries in configured.items():
            # `hooks.state` is Codex trust bookkeeping, not a hook list.
            if isinstance(event, str) and isinstance(entries, list):
                hooks[event] = []
    return hooks


def _sandbox_policy(cwd: Path, *, read_only: bool) -> dict[str, object]:
    if read_only:
        return {"type": "readOnly", "networkAccess": False}
    return {
        "type": "workspaceWrite",
        "writableRoots": [str(cwd)],
        "networkAccess": True,
    }


def _usage_event(value: dict[str, object], turn_id: str) -> ProviderUsage | None:
    last = value.get("last")
    if not isinstance(last, dict):
        return None
    input_tokens = _usage_int(last.get("inputTokens"))
    output_tokens = _usage_int(last.get("outputTokens"))
    cached_input_tokens = _usage_int(last.get("cachedInputTokens"))
    cache_write_input_tokens = _usage_int(last.get("cacheWriteInputTokens"))
    reasoning_output_tokens = _usage_int(last.get("reasoningOutputTokens"))
    return ProviderUsage(
        provider_profile="codex.app-server.turn.v1",
        provider_event_type="thread/tokenUsage/updated",
        dedupe_key=turn_id,
        processed_input_tokens=input_tokens,
        generated_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        reported_input_tokens=input_tokens,
        reported_output_tokens=output_tokens,
        reported_total_tokens=_usage_int(last.get("totalTokens")),
        provider_fields={str(key): item for key, item in last.items()},
    )


def _usage_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _error_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str) and message:
            return message
        return json.dumps(value, ensure_ascii=False)
    return ""


def _rpc_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
