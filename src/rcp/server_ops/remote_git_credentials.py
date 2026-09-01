"""Account-local deploy-key operations shipped to local or SSH targets.

The server sends this module's own source through ``python3 -c``. It validates
the execution account, home, credential paths, ownership, and modes before it
creates, reads, or removes one exact key pair. Its JSON protocol never includes
private-key bytes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ALIAS = re.compile(r"[a-z][a-z0-9-]{0,47}")
ACCOUNT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")
FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{43}")
PRIVATE_MODE = 0o600
PUBLIC_MODE = 0o644
DIRECTORY_MODE = 0o700
MAX_PUBLIC_KEY_BYTES = 16 * 1024
COMMAND_TIMEOUT_SECONDS = 30.0
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _uuid4(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a lowercase canonical UUID4")
    return value


def _account(expected_account: str) -> tuple[pwd.struct_passwd, Path]:
    if ACCOUNT.fullmatch(expected_account) is None:
        raise ValueError("expected account is invalid")
    account = pwd.getpwuid(os.getuid())
    if account.pw_name != expected_account:
        raise ValueError("credential helper is running as the wrong operating-system account")
    home = Path(account.pw_dir)
    if not home.is_absolute() or home == Path("/") or ".." in home.parts:
        raise ValueError("execution account home is not an absolute normalized non-root path")
    _require_directory(home, uid=account.pw_uid, label="execution account home")
    return account, home


def _require_directory(
    path: Path,
    *,
    uid: int,
    label: str,
    exact_mode: int | None = None,
) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != uid
        or mode & 0o022
        or (exact_mode is not None and mode != exact_mode)
    ):
        raise ValueError(f"{label} has unsafe type, ownership, or mode")


def _ensure_directory(
    path: Path,
    *,
    uid: int,
    label: str,
    exact_mode: int | None = None,
) -> None:
    if not os.path.lexists(path):
        try:
            os.mkdir(path, DIRECTORY_MODE)
        except OSError as exc:
            raise ValueError(f"{label} could not be created") from exc
    _require_directory(path, uid=uid, label=label, exact_mode=exact_mode)


def _credentials_root(
    account: pwd.struct_passwd,
    home: Path,
    location: str,
    configured_root: str,
) -> Path:
    if location == "local":
        root = Path(configured_root)
        if (
            not root.is_absolute()
            or root == Path("/")
            or ".." in root.parts
            or str(root) != configured_root
        ):
            raise ValueError("server credential root is invalid")
        _require_directory(
            root,
            uid=account.pw_uid,
            label="server credential root",
            exact_mode=DIRECTORY_MODE,
        )
        return root
    if location != "ssh" or configured_root != "-":
        raise ValueError("credential location is invalid")

    local = home / ".local"
    share = local / "share"
    rcp = share / "rcp"
    root = rcp / "credentials"
    _ensure_directory(local, uid=account.pw_uid, label="remote .local directory")
    _ensure_directory(share, uid=account.pw_uid, label="remote share directory")
    _ensure_directory(
        rcp,
        uid=account.pw_uid,
        label="remote RCP state directory",
        exact_mode=DIRECTORY_MODE,
    )
    _ensure_directory(
        root,
        uid=account.pw_uid,
        label="remote credential root",
        exact_mode=DIRECTORY_MODE,
    )
    return root


def _absolute_path(value: str, label: str) -> Path:
    if len(value) > 4096 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"{label} is invalid")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} is invalid")
    return path


def _central_root(location: str, home: Path, configured_root: str) -> str:
    if location == "ssh" and configured_root == "-":
        return str(home / ".local" / "share" / "rcp" / "projects")
    if configured_root == "-":
        raise ValueError("server-local central checkout root cannot be defaulted")
    return str(_absolute_path(configured_root, "central checkout root"))


def _key_locations(
    root: Path,
    central_root: str,
    *,
    project_id: str,
    alias: str,
) -> tuple[Path, Path]:
    _uuid4(project_id, "project id")
    if ALIAS.fullmatch(alias) is None:
        raise ValueError("repository alias is invalid")
    checkout = _absolute_path(central_root, "central checkout root")
    repository_checkout = checkout / project_id / "repositories" / alias
    if (
        root == repository_checkout
        or root in repository_checkout.parents
        or repository_checkout in root.parents
    ):
        raise ValueError("central checkout and credential paths overlap")
    private = root / "projects" / project_id / alias / "id_ed25519"
    return private, Path(f"{private}.pub")


def _key_paths(
    root: Path,
    central_root: str,
    *,
    uid: int,
    project_id: str,
    alias: str,
    create: bool,
) -> tuple[Path, Path]:
    private, public = _key_locations(
        root,
        central_root,
        project_id=project_id,
        alias=alias,
    )
    projects = root / "projects"
    project = projects / project_id
    repository = project / alias
    for path, label in (
        (projects, "project credential directory"),
        (project, "project key directory"),
        (repository, "repository key directory"),
    ):
        if create:
            _ensure_directory(
                path,
                uid=uid,
                label=label,
                exact_mode=DIRECTORY_MODE,
            )
        else:
            _require_directory(
                path,
                uid=uid,
                label=label,
                exact_mode=DIRECTORY_MODE,
            )
    return private, public


def _require_file(path: Path, *, uid: int, mode: int, label: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or stat.S_IMODE(info.st_mode) != mode:
        raise ValueError(f"{label} has unsafe type, ownership, or mode")


def _command_environment(home: Path, account: str) -> dict[str, str]:
    return {
        "HOME": str(home),
        "USER": account,
        "LOGNAME": account,
        "PATH": SAFE_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _run(
    argv: tuple[str, ...],
    *,
    home: Path,
    account: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=home,
            env=_command_environment(home, account),
            text=True,
            capture_output=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("credential helper command could not complete") from exc


def _public_material(
    account: pwd.struct_passwd,
    home: Path,
    root: Path,
    private: Path,
    public: Path,
    *,
    created: bool,
) -> dict[str, object]:
    _require_file(private, uid=account.pw_uid, mode=PRIVATE_MODE, label="deploy private key")
    _require_file(public, uid=account.pw_uid, mode=PUBLIC_MODE, label="deploy public key")
    try:
        if public.stat().st_size > MAX_PUBLIC_KEY_BYTES:
            raise ValueError("deploy public key is too large")
        public_key = public.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError("deploy public key could not be read") from exc
    if "\n" in public_key:
        raise ValueError("deploy public key must be one line")
    parts = public_key.split()
    if len(parts) != 3 or parts[0] != "ssh-ed25519":
        raise ValueError("deploy public key has an unexpected format")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except ValueError as exc:
        raise ValueError("deploy public key is not valid base64") from exc
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode(
        "ascii"
    ).rstrip("=")
    if FINGERPRINT.fullmatch(fingerprint) is None:
        raise ValueError("deploy public key fingerprint is invalid")

    derived = _run(
        ("ssh-keygen", "-y", "-f", str(private)),
        home=home,
        account=account.pw_name,
    )
    if derived.returncode != 0 or derived.stdout.strip().split()[:2] != parts[:2]:
        raise ValueError("deploy private and public keys are not one key pair")
    return {
        "account": account.pw_name,
        "home": str(home),
        "credentials_root": str(root),
        "private_key_path": str(private),
        "public_key": public_key,
        "public_key_fingerprint": fingerprint,
        "created": created,
    }


def _prepare_or_inspect(
    operation: str,
    expected_account: str,
    location: str,
    configured_root: str,
    central_root: str,
    space_id: str,
    project_id: str,
    alias: str,
) -> dict[str, object]:
    _uuid4(space_id, "space id")
    account, home = _account(expected_account)
    root = _credentials_root(account, home, location, configured_root)
    resolved_central_root = _central_root(location, home, central_root)
    private, public = _key_paths(
        root,
        resolved_central_root,
        uid=account.pw_uid,
        project_id=project_id,
        alias=alias,
        create=operation == "prepare",
    )
    label = f"rcp:{space_id}:{project_id}:{alias}"
    created = False
    private_exists = os.path.lexists(private)
    public_exists = os.path.lexists(public)
    if operation == "prepare" and not private_exists and not public_exists:
        old_mask = os.umask(0o077)
        try:
            generated = _run(
                (
                    "ssh-keygen",
                    "-q",
                    "-t",
                    "ed25519",
                    "-N",
                    "",
                    "-C",
                    label,
                    "-f",
                    str(private),
                ),
                home=home,
                account=account.pw_name,
            )
        finally:
            os.umask(old_mask)
        if generated.returncode != 0:
            raise ValueError("deploy key generation failed; preserve and inspect the exact path")
        try:
            os.chmod(private, PRIVATE_MODE)
            os.chmod(public, PUBLIC_MODE)
        except OSError as exc:
            raise ValueError("deploy key modes could not be restricted") from exc
        created = True
    elif private_exists != public_exists:
        raise ValueError("deploy key pair is incomplete; preserve and inspect the exact path")
    elif not private_exists:
        raise ValueError("deploy key pair does not exist")
    result = _public_material(
        account,
        home,
        root,
        private,
        public,
        created=created,
    )
    result["label"] = label
    return result


def _recovery_preflight(
    expected_account: str,
    location: str,
    configured_root: str,
    central_root: str,
    space_id: str,
    project_id: str,
    alias: str,
) -> dict[str, object]:
    """Prove the deterministic replacement-key target is empty before journaling it."""

    _uuid4(space_id, "space id")
    account, home = _account(expected_account)
    root = _credentials_root(account, home, location, configured_root)
    resolved_central_root = _central_root(location, home, central_root)
    private, public = _key_locations(
        root,
        resolved_central_root,
        project_id=project_id,
        alias=alias,
    )
    return {
        "account": account.pw_name,
        "home": str(home),
        "credentials_root": str(root),
        "private_key_path": str(private),
        "label": f"rcp:{space_id}:{project_id}:{alias}",
        "absent": not os.path.lexists(private) and not os.path.lexists(public),
    }


def _remove(
    expected_account: str,
    location: str,
    configured_root: str,
    central_root: str,
    space_id: str,
    project_id: str,
    alias: str,
    expected_fingerprint: str,
) -> dict[str, object]:
    if FINGERPRINT.fullmatch(expected_fingerprint) is None:
        raise ValueError("expected public-key fingerprint is invalid")
    _uuid4(space_id, "space id")
    account, home = _account(expected_account)
    root = _credentials_root(account, home, location, configured_root)
    resolved_central_root = _central_root(location, home, central_root)
    private, public = _key_locations(
        root,
        resolved_central_root,
        project_id=project_id,
        alias=alias,
    )
    if not os.path.lexists(private) and not os.path.lexists(public):
        return {"removed": False}
    private, public = _key_paths(
        root,
        resolved_central_root,
        uid=account.pw_uid,
        project_id=project_id,
        alias=alias,
        create=False,
    )
    material = _public_material(
        account,
        home,
        root,
        private,
        public,
        created=False,
    )
    if material["public_key_fingerprint"] != expected_fingerprint:
        raise ValueError("deploy key fingerprint changed; the key was not removed")
    try:
        os.unlink(public)
        os.unlink(private)
        for directory in (private.parent, private.parent.parent, private.parent.parent.parent):
            try:
                os.rmdir(directory)
            except OSError:
                break
    except OSError as exc:
        raise ValueError("the exact deploy key pair could not be removed") from exc
    return {"removed": True}


def _prepare_probe_directory(expected_account: str, request_id: str) -> dict[str, object]:
    _uuid4(request_id, "request id")
    account, _home = _account(expected_account)
    path = Path(tempfile.mkdtemp(prefix=f"rcp-git-probe.{request_id}.", dir="/tmp"))
    os.chmod(path, DIRECTORY_MODE)
    _require_directory(
        path,
        uid=account.pw_uid,
        label="Git write-probe directory",
        exact_mode=DIRECTORY_MODE,
    )
    return {"probe_directory": str(path)}


def _cleanup_probe_directory(
    expected_account: str,
    request_id: str,
    raw_path: str,
) -> dict[str, object]:
    _uuid4(request_id, "request id")
    account, _home = _account(expected_account)
    path = Path(raw_path)
    prefix = f"rcp-git-probe.{request_id}."
    if path.parent != Path("/tmp") or not path.name.startswith(prefix):
        raise ValueError("Git write-probe cleanup path is outside its request boundary")
    _require_directory(
        path,
        uid=account.pw_uid,
        label="Git write-probe directory",
        exact_mode=DIRECTORY_MODE,
    )
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise ValueError("Git write-probe directory could not be removed") from exc
    return {"removed": True}


def main() -> None:
    try:
        operation = sys.argv[1]
        if operation in {"prepare", "inspect"} and len(sys.argv) == 9:
            result = _prepare_or_inspect(operation, *sys.argv[2:])
        elif operation == "recovery-preflight" and len(sys.argv) == 9:
            result = _recovery_preflight(*sys.argv[2:])
        elif operation == "remove" and len(sys.argv) == 10:
            result = _remove(*sys.argv[2:])
        elif operation == "probe-prepare" and len(sys.argv) == 4:
            result = _prepare_probe_directory(*sys.argv[2:])
        elif operation == "probe-cleanup" and len(sys.argv) == 5:
            result = _cleanup_probe_directory(*sys.argv[2:])
        else:
            raise ValueError("credential helper operation is invalid")
    except (IndexError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
