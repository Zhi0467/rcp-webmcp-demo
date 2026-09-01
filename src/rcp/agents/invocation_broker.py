from __future__ import annotations

import asyncio
import json
import math
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProviderInvocationGate:
    """In-memory authority shared only with one execution-host broker wrapper."""

    mailbox_id: str
    broker_path: str
    socket_path: str
    workspace: str
    response_timeout_seconds: float
    _token: str = field(repr=False)
    _ready_nonce: str = field(default_factory=lambda: secrets.token_hex(16), repr=False)

    @property
    def ready_line(self) -> str:
        return f"RCP_COMMAND_BROKER_READY:{self._ready_nonce}"

    def _broker_argv(self) -> list[str]:
        return [
            "python3",
            self.broker_path,
            "--socket",
            self.socket_path,
            "--mailbox-id",
            self.mailbox_id,
            "--ready-line",
            self.ready_line,
            "--response-timeout",
            f"{self.response_timeout_seconds:g}",
        ]

    def wrap_command(self, command: list[str]) -> list[str]:
        if not command:
            raise ValueError("provider command must not be empty")
        return [*self._broker_argv(), "--", *command]

    def bootstrap(self, prompt: bytes) -> bytes:
        document = json.dumps(
            {"version": 1, "mailbox_id": self.mailbox_id, "token": self._token},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return document + b"\n" + prompt

    def client_arguments(self) -> tuple[str, str, str, str]:
        return ("--broker", self.socket_path, "--mailbox-id", self.mailbox_id)

    @asynccontextmanager
    async def serve_current_session(self, *, timeout_seconds: float = 5.0) -> AsyncIterator[None]:
        """Run a broker without a provider wrapper for the in-process acceptance double."""

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("broker startup timeout must be a positive finite number")
        process = await asyncio.create_subprocess_exec(
            *self._broker_argv(),
            "--standalone",
            cwd=self.workspace,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(self.bootstrap(b""))
        await process.stdin.drain()
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout_seconds)
            if line.decode("utf-8", errors="replace").rstrip() != self.ready_line:
                stderr = b""
                if process.stderr is not None:
                    with suppress(TimeoutError):
                        stderr = await asyncio.wait_for(process.stderr.read(), timeout=0.2)
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(detail or "episode command broker did not become ready")
            yield
        finally:
            process.stdin.close()
            with suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
