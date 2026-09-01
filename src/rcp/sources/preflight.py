from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rcp.config import MachineConfig
from rcp.limits import REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS
from rcp.transport.ssh import ssh_arguments


@dataclass(frozen=True)
class _ProviderRoot:
    provider: str
    agent_path: str
    probe_path: str


_REMOTE_PREFLIGHT_SCRIPT = r"""
import json
import os
import stat
import sys

results = []
for item in json.load(sys.stdin):
    root = item["root"]
    path = os.path.expanduser(root)
    error = None
    try:
        info = os.stat(path)
    except FileNotFoundError:
        error = "does not exist"
    except PermissionError as exc:
        error = f"cannot inspect metadata: {exc}"
    except OSError as exc:
        error = f"cannot inspect metadata: {exc}"
    else:
        if not stat.S_ISDIR(info.st_mode):
            error = "is not a directory"
        elif not os.access(path, os.R_OK | os.X_OK):
            error = "is not readable and traversable"
    results.append({"provider": item["provider"], "root": root, "error": error})
print(json.dumps(results, separators=(",", ":")))
"""


def preflight_provider_roots(
    source_roots: Mapping[str, list[str]], machine: MachineConfig
) -> list[str]:
    """Return best-effort metadata diagnostics for roots on ``machine``.

    This intentionally performs no directory enumeration and reads no provider
    files. Every failure is converted to a diagnostic so source preflight can
    never prevent a Seed or Refresh launch.
    """

    roots = [
        _ProviderRoot(provider=provider, agent_path=path, probe_path=path)
        for provider, paths in source_roots.items()
        for path in paths
    ]
    if not roots:
        return []
    try:
        if machine.host:
            return _preflight_remote(machine, roots)
        return _preflight_local(machine, roots)
    except Exception as exc:  # Defensive: source preflight is never launch authority.
        detail = f"preflight unavailable: {type(exc).__name__}: {exc}"
        return [_diagnostic(machine, root, detail) for root in roots]


def _preflight_local(machine: MachineConfig, roots: list[_ProviderRoot]) -> list[str]:
    diagnostics: list[str] = []
    for root in roots:
        try:
            info = os.stat(root.probe_path)
        except FileNotFoundError:
            diagnostics.append(_diagnostic(machine, root, "does not exist"))
        except PermissionError as exc:
            diagnostics.append(_diagnostic(machine, root, f"cannot inspect metadata: {exc}"))
        except OSError as exc:
            diagnostics.append(_diagnostic(machine, root, f"cannot inspect metadata: {exc}"))
        else:
            if not stat.S_ISDIR(info.st_mode):
                diagnostics.append(_diagnostic(machine, root, "is not a directory"))
            elif not os.access(root.probe_path, os.R_OK | os.X_OK):
                diagnostics.append(_diagnostic(machine, root, "is not readable and traversable"))
    return diagnostics


def _preflight_remote(machine: MachineConfig, roots: list[_ProviderRoot]) -> list[str]:
    payload = json.dumps(
        [{"provider": root.provider, "root": root.probe_path} for root in roots],
        separators=(",", ":"),
    )
    command = shlex.join(["python3", "-c", _REMOTE_PREFLIGHT_SCRIPT])
    try:
        result = subprocess.run(
            ssh_arguments(machine.host, command),
            input=payload,
            capture_output=True,
            text=True,
            timeout=REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        detail = (
            "remote metadata probe timed out after "
            f"{REMOTE_SOURCE_OPERATION_TIMEOUT_SECONDS} seconds"
        )
        return [_diagnostic(machine, root, detail) for root in roots]
    except OSError as exc:
        detail = f"remote metadata probe could not start: {exc}"
        return [_diagnostic(machine, root, detail) for root in roots]

    if result.returncode:
        detail = result.stderr.strip() or f"remote metadata probe exited {result.returncode}"
        return [_diagnostic(machine, root, detail) for root in roots]

    try:
        response: Any = json.loads(result.stdout)
    except json.JSONDecodeError:
        response = None
    expected = [(root.provider, root.probe_path) for root in roots]
    if not isinstance(response, list) or len(response) != len(expected):
        detail = "remote metadata probe returned an invalid response"
        return [_diagnostic(machine, root, detail) for root in roots]

    diagnostics: list[str] = []
    for root, expected_identity, item in zip(roots, expected, response, strict=True):
        if (
            not isinstance(item, dict)
            or (item.get("provider"), item.get("root")) != expected_identity
            or not (item.get("error") is None or isinstance(item.get("error"), str))
        ):
            detail = "remote metadata probe returned an invalid response"
            return [_diagnostic(machine, candidate, detail) for candidate in roots]
        if item["error"]:
            diagnostics.append(_diagnostic(machine, root, item["error"]))
    return diagnostics


def _diagnostic(machine: MachineConfig, root: _ProviderRoot, detail: str) -> str:
    location = f"{machine.alias} ({machine.host})" if machine.host else machine.alias
    return f"{location}/{root.provider} source root {root.agent_path!r}: {detail}"
