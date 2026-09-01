"""Read-only local and SSH checkout verification for backup capture."""

from __future__ import annotations

import importlib.resources
import json
import re
import shlex
import subprocess
from functools import lru_cache

from rcp.limits import SERVER_PROJECT_CHECKOUT_TIMEOUT_SECONDS
from rcp.server_ops.backup_models import BackupCheckoutRecoveryDescriptor
from rcp.transport.remote_backup_checkout import CheckoutInspectionError, inspect_checkout
from rcp.transport.ssh import ssh_arguments

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_REMOTE_CHECKOUT_OUTPUT_MAX_BYTES = 64 * 1024


class BackupCheckoutHostUnavailable(CheckoutInspectionError):
    """The configured SSH route could not reach its checkout host."""

    def __init__(self, message: str, *, machine_alias: str) -> None:
        super().__init__(message)
        self.machine_alias = machine_alias


@lru_cache(maxsize=1)
def _remote_checkout_source() -> str:
    return (
        importlib.resources.files("rcp.transport")
        .joinpath("remote_backup_checkout.py")
        .read_text(encoding="utf-8")
    )


def verify_checkout_identities(recovery: BackupCheckoutRecoveryDescriptor) -> None:
    """Prove every central checkout still matches its recovery descriptor."""

    machines = {machine.alias: machine for machine in recovery.machines}
    for repository in recovery.repositories:
        machine = machines[repository.machine_alias]
        arguments = (
            machine.os_account,
            repository.resolved_path,
            repository.repository.ssh_clone_url,
            repository.git_commit,
        )
        if machine.location == "local":
            payload = inspect_checkout(
                os_account=arguments[0],
                repository_path=arguments[1],
                expected_origin=arguments[2],
                recorded_commit=arguments[3],
            )
        else:
            command = shlex.join(("python3", "-c", _remote_checkout_source(), *arguments))
            try:
                result = subprocess.run(
                    ssh_arguments(machine.host, command),
                    capture_output=True,
                    text=True,
                    timeout=SERVER_PROJECT_CHECKOUT_TIMEOUT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BackupCheckoutHostUnavailable(
                    "The remote checkout identity is unavailable.",
                    machine_alias=machine.alias,
                ) from exc
            if result.returncode == 255:
                raise BackupCheckoutHostUnavailable(
                    "The remote checkout host is unreachable.",
                    machine_alias=machine.alias,
                )
            if (
                result.returncode != 0
                or len(result.stdout.encode("utf-8", errors="replace"))
                > _REMOTE_CHECKOUT_OUTPUT_MAX_BYTES
                or len(result.stderr.encode("utf-8", errors="replace"))
                > _REMOTE_CHECKOUT_OUTPUT_MAX_BYTES
            ):
                raise CheckoutInspectionError("The remote checkout identity could not be verified.")
            try:
                payload = json.loads(result.stdout)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise CheckoutInspectionError("The remote checkout proof is invalid.") from exc
        expected = {
            "account": arguments[0],
            "repository_path": arguments[1],
            "origin": arguments[2],
            "recorded_commit": arguments[3],
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != {*expected, "head"}
            or any(payload[key] != value for key, value in expected.items())
            or not isinstance(payload["head"], str)
            or _FULL_GIT_COMMIT.fullmatch(payload["head"]) is None
        ):
            raise CheckoutInspectionError(
                "The checkout proof does not match its recovery identity."
            )


__all__ = ["BackupCheckoutHostUnavailable", "verify_checkout_identities"]
