from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import threading
import time
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from rcp.agents.launcher import AgentLauncher, ProviderReadiness
from rcp.providers import ProviderSkill, ProviderSkillReference, profile_for
from rcp.storage import AppStore, ProviderSkillInventoryRecord
from rcp.transport.ssh import ssh_arguments


class ProviderSkillInventorySnapshot(BaseModel):
    """The last-known provider-native skills for one execution target."""

    provider: str
    machine: str
    host: str
    configured_binary: str | None = None
    resolved_binary: str | None = None
    provider_version: str | None = None
    command: list[str] = Field(default_factory=list)
    protocol: Literal["jsonrpc", "jsonl"] | None = None
    skills: list[ProviderSkill] = Field(default_factory=list)
    inventory_hash: str | None = None
    status: Literal["refreshing", "fresh", "stale", "unavailable"]
    stale: bool = False
    diagnostic: str | None = None
    refreshed_at: str | None = None


class ProviderSkillInventoryManager:
    """Refresh and resolve durable provider-owned skill inventories."""

    def __init__(self, store: AppStore, *, timeout: float = 30.0) -> None:
        self.store = store
        self.timeout = timeout
        self._lock = threading.Lock()
        self._refreshes: dict[tuple[str, str, str], threading.Event] = {}
        self._refresh_deadlines: dict[tuple[str, str, str], float] = {}

    def mark_refreshing(
        self,
        provider: str,
        host: str,
        configured_binary: str | None,
    ) -> ProviderSkillInventorySnapshot:
        key = self._key(provider, host, configured_binary)
        with self._lock:
            self._refreshes[key] = threading.Event()
            self._refresh_deadlines[key] = time.monotonic() + self.timeout + 5
        self.store.mark_provider_skill_inventory_refreshing(
            provider,
            host,
            configured_binary,
            updated_at=_now(),
        )
        return self.snapshot(provider, host, configured_binary, host or "local")

    def refresh(
        self,
        provider: str,
        host: str,
        configured_binary: str | None,
        readiness: ProviderReadiness,
    ) -> ProviderSkillInventorySnapshot:
        key = self._key(provider, host, configured_binary)
        with self._lock:
            event = self._refreshes.get(key)
        if event is None or event.is_set():
            self.mark_refreshing(provider, host, configured_binary)
            with self._lock:
                event = self._refreshes[key]

        machine = host or "local"
        try:
            if not readiness.installed or not readiness.authenticated or not readiness.binary_path:
                raise ValueError(readiness.reason or "Provider is not ready for skill discovery.")
            if readiness.path_state not in {"resolved", "unconfigured"}:
                raise ValueError(readiness.reason or "Provider executable path is unresolved.")
            if not readiness.version:
                raise ValueError("Provider did not report a version during readiness.")

            profile = profile_for(provider)
            probe = profile.skill_probe(readiness.binary_path)
            payload = self._run_probe(host, probe.command, probe.protocol)
            skills = sorted(
                profile.parse_skills(payload), key=lambda item: (item.name, item.path or "")
            )
            inventory_hash = _inventory_hash(skills)
            refreshed_at = _now()
            self.store.save_provider_skill_inventory_success(
                provider,
                host,
                configured_binary,
                resolved_binary=readiness.binary_path,
                provider_version=readiness.version,
                command=probe.command,
                protocol=probe.protocol,
                skills=skills,
                inventory_hash=inventory_hash,
                refreshed_at=refreshed_at,
            )
        except Exception as exc:
            self.store.save_provider_skill_inventory_failure(
                provider,
                host,
                configured_binary,
                diagnostic=str(exc),
                updated_at=_now(),
            )
        finally:
            event.set()
        return self.snapshot(provider, host, configured_binary, machine)

    def snapshot(
        self,
        provider: str,
        host: str,
        configured_binary: str | None,
        machine: str,
    ) -> ProviderSkillInventorySnapshot:
        record = self.store.provider_skill_inventory(provider, host, configured_binary)
        if record is None:
            return ProviderSkillInventorySnapshot(
                provider=provider,
                machine=machine,
                host=host,
                configured_binary=configured_binary,
                status="unavailable",
            )
        return _snapshot(record, machine=machine, configured_binary=configured_binary)

    def wait(
        self,
        provider: str,
        host: str,
        configured_binary: str | None,
    ) -> bool:
        with self._lock:
            key = self._key(provider, host, configured_binary)
            event = self._refreshes.get(key)
            deadline = self._refresh_deadlines.get(key)
        if event is None:
            return True
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        return event.wait(timeout=remaining)

    def resolve(
        self,
        provider: str,
        host: str,
        configured_binary: str | None,
        machine: str,
        names: list[str],
    ) -> list[ProviderSkillReference]:
        if not names:
            return []
        inventory = self.snapshot(provider, host, configured_binary, machine)
        available = {skill.name: skill for skill in inventory.skills if skill.enabled}
        missing = [name for name in names if name not in available]
        if missing:
            rendered = ", ".join(repr(name) for name in missing)
            raise ValueError(
                f"Provider-native skill selection is unavailable for {provider} on {machine}: "
                f"{rendered}."
            )
        if not inventory.provider_version or not inventory.inventory_hash:
            raise ValueError(
                f"Provider-native skill inventory is unavailable for {provider} on {machine}."
            )
        return [
            ProviderSkillReference(
                provider=provider,
                machine=machine,
                provider_version=inventory.provider_version,
                inventory_hash=inventory.inventory_hash,
                name=name,
                label=available[name].label,
                description=available[name].description,
                stale=inventory.stale,
            )
            for name in names
        ]

    def _run_probe(
        self,
        host: str,
        command: list[str],
        protocol: Literal["jsonrpc", "jsonl"],
    ) -> object:
        arguments = command
        if host:
            arguments = ssh_arguments(host, AgentLauncher._remote_login_command(command))
        if protocol == "jsonl":
            result = subprocess.run(
                arguments,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode:
                detail = result.stderr.strip() or f"skill probe exited {result.returncode}"
                raise ValueError(detail)
            return result.stdout
        return self._run_jsonrpc(arguments)

    def _run_jsonrpc(self, arguments: list[str]) -> object:
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Remote login shells and provider diagnostics can both emit on
            # stderr. Merge the streams so the stdout reader drains everything;
            # leaving a separate unread pipe can deadlock a verbose app-server.
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        messages: queue.Queue[str | None] = queue.Queue()

        def read_stdout() -> None:
            for line in process.stdout:
                messages.put(line)
            messages.put(None)

        reader = threading.Thread(target=read_stdout, name="rcp-codex-skill-probe", daemon=True)
        reader.start()
        deadline = time.monotonic() + self.timeout
        try:
            _write_rpc(
                process.stdin,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "clientInfo": {"name": "rcp", "version": "1"},
                        "capabilities": {},
                    },
                },
            )
            self._rpc_response(messages, request_id=1, deadline=deadline)
            _write_rpc(
                process.stdin,
                {"jsonrpc": "2.0", "method": "initialized", "params": {}},
            )
            _write_rpc(
                process.stdin,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "skills/list",
                    "params": {"cwds": ["/"], "forceReload": True},
                },
            )
            return self._rpc_response(messages, request_id=2, deadline=deadline)
        finally:
            process.stdin.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _rpc_response(
        self,
        messages: queue.Queue[str | None],
        *,
        request_id: int,
        deadline: float,
    ) -> object:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError(f"Timed out waiting for provider response {request_id}.")
            try:
                line = messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise ValueError(f"Timed out waiting for provider response {request_id}.") from exc
            if line is None:
                raise ValueError(f"Provider closed before response {request_id}.")
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict) or value.get("id") != request_id:
                continue
            if "error" in value:
                raise ValueError(f"Provider skill probe failed: {value['error']}")
            if "result" not in value:
                raise ValueError(f"Provider response {request_id} has no result.")
            return value["result"]

    @staticmethod
    def _key(provider: str, host: str, configured_binary: str | None) -> tuple[str, str, str]:
        return provider, host, configured_binary or ""


def _write_rpc(stream: object, value: dict[str, object]) -> None:
    stream.write(json.dumps(value, separators=(",", ":")) + "\n")
    stream.flush()


def _inventory_hash(skills: list[ProviderSkill]) -> str:
    payload = [skill.model_dump(mode="json") for skill in skills]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _snapshot(
    record: ProviderSkillInventoryRecord,
    *,
    machine: str,
    configured_binary: str | None,
) -> ProviderSkillInventorySnapshot:
    return ProviderSkillInventorySnapshot(
        provider=record.provider,
        machine=machine,
        host=record.host,
        configured_binary=configured_binary,
        resolved_binary=record.resolved_binary,
        provider_version=record.provider_version,
        command=record.command,
        protocol=record.protocol,
        skills=record.skills,
        inventory_hash=record.inventory_hash,
        status=record.status,
        stale=record.status == "stale",
        diagnostic=record.diagnostic,
        refreshed_at=record.refreshed_at,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
