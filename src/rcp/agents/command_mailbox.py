from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import math
import re
import secrets
import shlex
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import ValidationError

from rcp.agents.command_protocol import (
    CommandCredential,
    CommandRequest,
    CommandResponse,
    command_authentication_payload,
    command_requires_idempotency_key,
    request_identity_is_well_formed,
    staged_command_broker_source,
    staged_command_client_source,
    validate_command_request,
)
from rcp.agents.invocation_broker import ProviderInvocationGate
from rcp.agents.staged_command_client import COMMAND_MAILBOX_MAX_REQUEST_BYTES
from rcp.limits import (
    COMMAND_BROKER_RESPONSE_GRACE_SECONDS,
    COMMAND_MAILBOX_POLL_SECONDS,
    COMMAND_MAILBOX_TIMEOUT_SECONDS,
)
from rcp.transport import RemoteRunStage, RunStageMailbox, StateUnavailable

_MAILBOX_ID = re.compile(r"^[a-f0-9]{32}$")
_REQUEST_FILE = re.compile(
    r"^rcp-command-(?P<mailbox_id>[a-f0-9]{32})-"
    r"(?P<request_id>[a-f0-9]{32})\.request\.json$"
)
_COMMAND_STATE_PREFIXES = ("rcp-command-", ".rcp-command-", ".rcp-mailbox-")


@dataclass(frozen=True, slots=True)
class CommandTurnIdentity:
    """The task turn, and optional episode, an in-memory credential represents."""

    episode_id: str | None
    task_id: str
    turn_id: str

    def __post_init__(self) -> None:
        values = (("task id", self.task_id), ("turn id", self.turn_id))
        if self.episode_id is not None:
            values = (("episode id", self.episode_id), *values)
        for label, value in values:
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"command {label} must be a non-blank exact identifier")


@dataclass(slots=True)
class CommandTurnCredential:
    """A one-shot secret binding that cannot be reactivated after its serve loop."""

    identity: CommandTurnIdentity
    mailbox_id: str
    _token: str = field(repr=False)
    _state: Literal["issued", "active", "expired"] = field(default="issued", repr=False)

    @classmethod
    def issue(cls, identity: CommandTurnIdentity) -> CommandTurnCredential:
        return cls(identity=identity, mailbox_id=uuid.uuid4().hex, _token=secrets.token_hex(32))

    @property
    def token(self) -> str:
        if self._state == "expired":
            raise RuntimeError("command credential has expired")
        return self._token

    @property
    def expired(self) -> bool:
        return self._state == "expired"

    def document(self) -> CommandCredential:
        if self.identity.episode_id is not None:
            raise RuntimeError("episode command authority is broker-only")
        return CommandCredential(mailbox_id=self.mailbox_id, token=self.token)

    def activate(self) -> None:
        if self._state != "issued":
            raise RuntimeError("command credential can serve exactly one turn")
        self._state = "active"

    def accepts(self, request: CommandRequest, document: str) -> bool:
        """Check one request against this turn's binding.

        ``document`` is the request exactly as it was written, because an episode
        broker signs those bytes rather than the model they validate into.
        """

        if self.identity.episode_id is not None:
            expected = hmac.new(
                self._token.encode("ascii"),
                command_authentication_payload(document),
                hashlib.sha256,
            ).hexdigest()
        else:
            expected = self._token
        return (
            self._state == "active"
            and _MAILBOX_ID.fullmatch(request.mailbox_id) is not None
            and secrets.compare_digest(self.mailbox_id, request.mailbox_id)
            and secrets.compare_digest(expected, request.credential)
        )

    def expire(self) -> None:
        self._token = ""
        self._state = "expired"


@dataclass(frozen=True, slots=True)
class StagedCommandMailbox:
    mailbox: RunStageMailbox
    credential: CommandTurnCredential
    client_path: str
    credential_path: str | None
    invocation_gate: ProviderInvocationGate | None = None
    timeout_seconds: float = COMMAND_MAILBOX_TIMEOUT_SECONDS

    @property
    def workspace(self) -> str:
        return str(self.mailbox.workspace)

    def client_argv(self, *arguments: str, timeout_seconds: float | None = None) -> tuple[str, ...]:
        """Build an argv tuple for the separately staged agent-only executable."""

        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("command client timeout must be a positive finite number")
        authority = (
            self.invocation_gate.client_arguments()
            if self.invocation_gate is not None
            else ("--credential", self.credential_path or "")
        )
        return (
            "python3",
            self.client_path,
            *authority,
            "--timeout",
            f"{timeout:g}",
            "--workspace",
            self.workspace,
            *arguments,
        )

    def client_command(self, *arguments: str, timeout_seconds: float | None = None) -> str:
        return shlex.join(self.client_argv(*arguments, timeout_seconds=timeout_seconds))

    def cleanup(self) -> None:
        cleanup_command_mailbox(mailbox=self.mailbox, credential=self.credential)


CommandHandlerResult: TypeAlias = CommandResponse | Awaitable[CommandResponse]
CommandHandler: TypeAlias = Callable[[CommandRequest, CommandTurnIdentity], CommandHandlerResult]


def stage_command_mailbox(
    *,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    episode_id: str | None,
    task_id: str,
    turn_id: str,
    timeout_seconds: float = COMMAND_MAILBOX_TIMEOUT_SECONDS,
) -> StagedCommandMailbox:
    """Clear a reusable stage and issue either broker or validate-only authority."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("command client timeout must be a positive finite number")
    mailbox = RunStageMailbox.for_stage(local_stage=local_stage, remote_stage=remote_stage)
    prepare_command_mailbox(mailbox=mailbox)
    identity = CommandTurnIdentity(episode_id=episode_id, task_id=task_id, turn_id=turn_id)
    credential = CommandTurnCredential.issue(identity)
    credential_path: str | None = None
    invocation_gate: ProviderInvocationGate | None = None
    try:
        source = staged_command_client_source()
        source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        client_path = mailbox.stage_text_input(
            f"rcp-agent-client-{credential.mailbox_id}-{source_digest}.py",
            source,
        )
        if episode_id is None:
            credential_name = f"rcp-command-{credential.mailbox_id}.credential.json"
            mailbox.write_text(
                credential_name,
                credential.document().model_dump_json(indent=2) + "\n",
            )
            credential_path = str(mailbox.workspace / credential_name)
        else:
            broker_source = staged_command_broker_source()
            broker_digest = hashlib.sha256(broker_source.encode("utf-8")).hexdigest()[:16]
            broker_path = mailbox.stage_text_input(
                f"rcp-command-broker-{credential.mailbox_id}-{broker_digest}.py",
                broker_source,
            )
            invocation_gate = ProviderInvocationGate(
                mailbox_id=credential.mailbox_id,
                broker_path=broker_path,
                socket_path=f"/tmp/rcp-command-{credential.mailbox_id}.sock",
                workspace=str(mailbox.workspace),
                response_timeout_seconds=timeout_seconds + COMMAND_BROKER_RESPONSE_GRACE_SECONDS,
                _token=credential.token,
            )
    except BaseException:
        with suppress(BaseException):
            cleanup_command_mailbox(mailbox=mailbox, credential=credential)
        raise
    return StagedCommandMailbox(
        mailbox=mailbox,
        credential=credential,
        client_path=client_path,
        credential_path=credential_path,
        invocation_gate=invocation_gate,
        timeout_seconds=timeout_seconds,
    )


def prepare_command_mailbox(*, mailbox: RunStageMailbox) -> None:
    """Clear every prior command request, response, credential, and interrupted temp file."""

    _clear_command_state(mailbox)


def cleanup_command_mailbox(
    *,
    mailbox: RunStageMailbox,
    credential: CommandTurnCredential | None = None,
) -> None:
    """Expire the credential and fail closed unless all command state is gone."""

    if credential is not None:
        credential.expire()
    _clear_command_state(mailbox)


async def serve_command_mailbox(
    *,
    staged: StagedCommandMailbox,
    handler: CommandHandler,
    stop: asyncio.Event,
    poll_seconds: float = COMMAND_MAILBOX_POLL_SECONDS,
    invocation_gate: ProviderInvocationGate | None = None,
) -> None:
    """Validate and dispatch requests; the injected handler owns every effect and record."""

    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("command mailbox poll interval must be a positive finite number")
    credential = staged.credential
    if credential.identity.episode_id is not None:
        if invocation_gate is None or invocation_gate is not staged.invocation_gate:
            raise ValueError("episode command mailbox requires its exact provider invocation gate")
    elif invocation_gate is not None:
        raise ValueError("validate-only mailbox does not accept a provider invocation gate")
    credential.activate()
    seen: set[str] = set()
    try:
        while not stop.is_set():
            names = await asyncio.to_thread(staged.mailbox.entry_names)
            requests = sorted(
                name
                for name in names
                if _request_identity_from_name(name, credential.mailbox_id) is not None
                and name not in seen
            )
            for name in requests:
                if stop.is_set():
                    return
                seen.add(name)
                identity = _request_identity_from_name(name, credential.mailbox_id)
                assert identity is not None
                response = await _answer_request(
                    name,
                    request_id=identity,
                    staged=staged,
                    handler=handler,
                )
                response_name = name.removesuffix(".request.json") + ".response.json"
                await asyncio.to_thread(
                    staged.mailbox.write_text,
                    response_name,
                    response.model_dump_json(indent=2) + "\n",
                )

            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
    finally:
        credential.expire()


async def _answer_request(
    name: str,
    *,
    request_id: str,
    staged: StagedCommandMailbox,
    handler: CommandHandler,
) -> CommandResponse:
    try:
        content = await asyncio.to_thread(
            staged.mailbox.read_text,
            name,
            max_bytes=COMMAND_MAILBOX_MAX_REQUEST_BYTES,
        )
        request = validate_command_request(content)
        if not request_identity_is_well_formed(request):
            raise ValueError("command request identity is malformed")
        if request.request_id != request_id or request.mailbox_id != staged.credential.mailbox_id:
            raise ValueError("command request identity does not match its file name")
        if not staged.credential.accepts(request, content):
            raise ValueError("command credential is invalid or expired")
        if (
            command_requires_idempotency_key(request.verb)
            and staged.credential.identity.episode_id is None
        ):
            raise ValueError(f"{request.verb} requires an episode-bound credential")
    except (FileNotFoundError, OSError, StateUnavailable) as exc:
        return _error_response(request_id, "unavailable", "Command request unavailable", exc)
    except (UnicodeError, ValueError, ValidationError) as exc:
        return _error_response(request_id, "invalid", "Command request invalid", exc)

    try:
        outcome = handler(request, staged.credential.identity)
        response = await outcome if inspect.isawaitable(outcome) else outcome
        if not isinstance(response, CommandResponse):
            raise TypeError("command handler returned an unsupported response")
        if response.request_id != request_id:
            raise ValueError("command handler returned a mismatched request identity")
        return response
    except Exception as exc:
        return _error_response(request_id, "unavailable", "Command handler unavailable", exc)


def _error_response(
    request_id: str,
    status: Literal["invalid", "unavailable"],
    prefix: str,
    error: BaseException,
) -> CommandResponse:
    detail = " ".join(str(error).split())
    message = f"{prefix}: {detail}" if detail else f"{prefix}."
    return CommandResponse(request_id=request_id, status=status, message=message[:2_000])


def _request_identity_from_name(name: str, mailbox_id: str) -> str | None:
    match = _REQUEST_FILE.fullmatch(name)
    if match is None or not secrets.compare_digest(match.group("mailbox_id"), mailbox_id):
        return None
    return match.group("request_id")


def _clear_command_state(mailbox: RunStageMailbox) -> None:
    for name in mailbox.entry_names():
        if name.startswith(_COMMAND_STATE_PREFIXES):
            mailbox.remove(name, missing_ok=False)
