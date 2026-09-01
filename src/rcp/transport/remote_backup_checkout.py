"""Read-only checkout identity proof used by local and SSH backup capture."""

from __future__ import annotations

import json
import os
import pwd
import re
import stat
import subprocess
import sys
from pathlib import Path

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_MAX_GIT_OUTPUT_BYTES = 64 * 1024
_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class CheckoutInspectionError(RuntimeError):
    """The checkout no longer matches its captured nonsecret identity."""


def _git_result(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = {
        "HOME": str(Path.home()),
        "USER": pwd.getpwuid(os.geteuid()).pw_name,
        "LOGNAME": pwd.getpwuid(os.geteuid()).pw_name,
        "PATH": _SAFE_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        result = subprocess.run(
            (
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(repository),
                *arguments,
            ),
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckoutInspectionError("The central checkout Git metadata is unavailable.") from exc
    if (
        len(result.stdout.encode("utf-8", errors="replace")) > _MAX_GIT_OUTPUT_BYTES
        or len(result.stderr.encode("utf-8", errors="replace")) > _MAX_GIT_OUTPUT_BYTES
    ):
        raise CheckoutInspectionError("The central checkout returned too much Git output.")
    return result


def _git(repository: Path, *arguments: str) -> str:
    result = _git_result(repository, *arguments)
    if result.returncode != 0:
        raise CheckoutInspectionError("The central checkout Git identity could not be read.")
    return result.stdout.strip()


def _optional_git(repository: Path, *arguments: str) -> str | None:
    result = _git_result(repository, *arguments)
    if result.returncode == 1 and not result.stdout and not result.stderr:
        return None
    if result.returncode != 0:
        raise CheckoutInspectionError("The central checkout Git identity could not be read.")
    return result.stdout.strip()


def inspect_checkout(
    *,
    os_account: str,
    repository_path: str,
    expected_origin: str,
    recorded_commit: str,
) -> dict[str, str]:
    """Verify exact account, path, origin, and retained provisioning commit."""

    if pwd.getpwuid(os.geteuid()).pw_name != os_account:
        raise CheckoutInspectionError("The checkout is not being read as its configured account.")
    repository = Path(repository_path)
    if (
        not repository.is_absolute()
        or repository == Path("/")
        or ".." in repository.parts
        or _FULL_COMMIT.fullmatch(recorded_commit) is None
    ):
        raise CheckoutInspectionError("The captured checkout identity is invalid.")
    try:
        metadata = repository.lstat()
    except OSError as exc:
        raise CheckoutInspectionError("The central checkout is unavailable.") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise CheckoutInspectionError("The central checkout is not a safe directory.")

    top_level = _git(repository, "rev-parse", "--show-toplevel")
    if top_level != str(repository):
        raise CheckoutInspectionError("The configured path is not the exact Git checkout root.")
    remotes = _git(repository, "remote").splitlines()
    if remotes != ["origin"]:
        raise CheckoutInspectionError("The central checkout does not have one exact origin.")
    origin = _git(repository, "config", "--local", "--get-all", "remote.origin.url")
    push_origin = _optional_git(
        repository,
        "config",
        "--local",
        "--get-all",
        "remote.origin.pushurl",
    )
    if origin != expected_origin or push_origin not in {None, expected_origin}:
        raise CheckoutInspectionError("The central checkout origin identity changed.")
    head = _git(repository, "rev-parse", "--verify", "HEAD^{commit}")
    if _FULL_COMMIT.fullmatch(head) is None:
        raise CheckoutInspectionError("The central checkout HEAD is invalid.")
    retained = _git(repository, "rev-parse", "--verify", f"{recorded_commit}^{{commit}}")
    if retained != recorded_commit:
        raise CheckoutInspectionError("The provisioning commit is absent from the checkout.")
    return {
        "account": os_account,
        "repository_path": str(repository),
        "origin": origin,
        "head": head,
        "recorded_commit": retained,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        return 2
    try:
        payload = inspect_checkout(
            os_account=argv[1],
            repository_path=argv[2],
            expected_origin=argv[3],
            recorded_commit=argv[4],
        )
    except CheckoutInspectionError:
        return 3
    sys.stdout.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through SSH source execution
    raise SystemExit(main(sys.argv))
