"""Destructive fresh-host restore qualification for one disposable backup."""

from __future__ import annotations

import http.cookies
import json
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path

import pytest

from .test_server_install_live import (
    _COMMAND_TIMEOUT_SECONDS,
    _EVENT_ADAPTER,
    _GITHUB_ED25519_FINGERPRINT,
    _REPOSITORY,
    _clear_deploy_key_receipt,
    _command_actions,
    _create_read_only_deploy_key,
    _delete_deploy_key,
    _fields,
    _github_request,
    _http_json,
    _prepare_bootstrap,
    _read_admin_token,
    _require_explicit_disposable_host,
    _run,
    _run_doctor,
    _run_install,
    _run_pty,
    _terminal_step,
    _wait_for_team_health,
    _workspace,
    _write_deploy_key_receipt,
)

_LIVE_GATE = "RCP_RUN_SERVER_RESTORE_LIVE"
_ARCHIVE_FILE = "RCP_LIVE_RESTORE_ARCHIVE_FILE"
_IDENTITY_FILE = "RCP_LIVE_RESTORE_IDENTITY_FILE"
_METADATA_FILE = "RCP_LIVE_RESTORE_METADATA_FILE"
_DEPLOY_KEY_RECEIPT_FILE = "RCP_LIVE_RESTORE_DEPLOY_KEY_RECEIPT_FILE"
_MAX_METADATA_BYTES = 16 * 1024
_GITHUB_GRANT_PROPAGATION_TIMEOUT_SECONDS = 60
_RESTORE_KEY_LABEL = re.compile(r"rcp:[0-9a-f-]{36}:[0-9a-f-]{36}:[a-z][a-z0-9_-]{0,63}")

pytestmark = pytest.mark.skipif(
    os.environ.get(_LIVE_GATE) != "1",
    reason="destructive disposable-host server-restore qualification is disabled",
)


def test_protected_backup_restores_on_a_fresh_disposable_ubuntu() -> None:
    """Install a fresh stopped service, reconstruct Git, review, and activate."""

    _require_explicit_disposable_host()
    workspace = _workspace()
    token = _read_admin_token()
    metadata = _read_restore_metadata()
    source_key_id: int | None = None
    project_key_id: int | None = None
    restore_root = Path("/tmp/rcp-server-restore-live")
    bootstrap_parent = Path(tempfile.mkdtemp(prefix="rcp-server-restore-bootstrap-"))
    bootstrap = bootstrap_parent / "rcp-bootstrap"

    try:
        _prepare_bootstrap(workspace, bootstrap)
        executable = bootstrap / ".venv" / "bin" / "rcp"
        first_code, first_events = _run_install(executable, cwd=bootstrap)
        assert first_code == 3
        source_pause = _terminal_step(first_events, "source_grant")
        source_fields = _fields(source_pause)
        source_label = str(source_fields["deploy_key_label"])
        _write_deploy_key_receipt(source_label)
        source_key_id = _create_read_only_deploy_key(
            token,
            title=source_label,
            public_key=str(source_fields["deploy_public_key"]),
        )
        source_trust = _command_actions(source_pause)
        assert len(source_trust) == 1
        trust_code, trust_output = _run_pty(source_trust[0], answer_host_key=True)
        assert trust_code == 1
        assert _GITHUB_ED25519_FINGERPRINT in trust_output

        second_code, second_events = _run_install(executable, cwd=bootstrap)
        assert second_code == 3
        assert _terminal_step(second_events, "team_space_init")["state"] == (
            "operator_action_needed"
        )
        shutil.rmtree(bootstrap_parent)

        archive, identity = _install_restore_inputs(restore_root)
        first_restore_code, first_restore_events = _run_restore(
            (
                "server",
                "restore",
                str(archive),
                "--identity-file",
                str(identity),
                "--machine-readable",
            )
        )
        assert first_restore_code == 3
        destination = _terminal_step(first_restore_events, "restore_destination_confirm")
        destination_resume = _single_resume(destination)

        grant_code, grant_events = _run_restore_action(destination_resume)
        assert grant_code == 3
        grant = _terminal_step(grant_events, "restore_checkouts")
        grant_fields = _fields(grant)
        project_label = str(grant_fields["deploy_key_label"])
        _write_restore_key_receipt(project_label)
        created = _github_request(
            token,
            method="POST",
            path=f"/repos/{_REPOSITORY}/keys",
            body={
                "title": project_label,
                "key": str(grant_fields["deploy_public_key"]),
                "read_only": False,
            },
        )
        if not isinstance(created.get("id"), int) or created.get("read_only") is not False:
            pytest.fail("GitHub did not confirm the fresh restore key has write access")
        project_key_id = int(created["id"])
        grant_trust = _command_actions(grant)
        assert len(grant_trust) == 1
        trust_code, trust_output = _run_pty(grant_trust[0], answer_host_key=True)
        assert trust_code == 1
        assert "successfully authenticated" in trust_output

        authority_code, authority_events = _resume_after_github_grant(grant)
        assert authority_code == 3
        authority = _terminal_step(authority_events, "restore_old_authority_review")
        destroyed = _command_containing(authority, "old-machine-destroyed")

        roster_code, roster_events = _run_restore_action(destroyed)
        assert roster_code == 3
        roster = _terminal_step(roster_events, "restore_member_roster_review")
        remove_stale = _command_ending_with(roster, str(metadata["stale_member_id"]))

        changed_roster_code, changed_roster_events = _run_restore_action(remove_stale)
        assert changed_roster_code == 3
        changed_roster = _terminal_step(
            changed_roster_events,
            "restore_member_roster_review",
        )
        confirm_roster = _command_containing(changed_roster, "--confirm-member-roster")

        final_code, final_events = _run_restore_action(confirm_roster)
        assert final_code == 0
        activated = _terminal_step(final_events, "restore_replacement_activation")
        assert activated["state"] == "succeeded"
        assert _fields(activated)["restore_phase"] == "complete"

        health = _wait_for_team_health()
        assert health["status"] == "ok"
        surviving_cookie = _exchange_token(str(metadata["surviving_member_token"]))
        projects, _headers = _http_json("GET", "/api/projects", cookie=surviving_cookie)
        if (
            not isinstance(projects, list)
            or not all(isinstance(item, dict) for item in projects)
            or [item.get("id") for item in projects] != [metadata["project_id"]]
        ):
            pytest.fail("the restored member did not read back the exact captured project")
        assert _token_exchange_status(str(metadata["stale_member_token"])) == 401
        doctor = _run_doctor()
        assert doctor["overall_state"] == "healthy"
        assert doctor["problems"] == "none"
    finally:
        if project_key_id is not None:
            _delete_deploy_key(token, project_key_id)
            _clear_restore_key_receipt()
        if source_key_id is not None:
            _delete_deploy_key(token, source_key_id)
            _clear_deploy_key_receipt()
        if bootstrap_parent.exists():
            shutil.rmtree(bootstrap_parent)


def _read_restore_metadata() -> dict[str, str]:
    path = _required_input(_METADATA_FILE)
    if path.stat().st_size > _MAX_METADATA_BYTES:
        pytest.fail("the restore metadata exceeds its fixed bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pytest.fail("the restore metadata is unavailable or invalid")
    expected = {
        "project_id",
        "surviving_member_id",
        "surviving_member_token",
        "stale_member_id",
        "stale_member_token",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or not all(isinstance(item, str) and item for item in value.values())
    ):
        pytest.fail("the restore metadata has an unsupported shape")
    return value


def _required_input(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        pytest.fail(f"{name} must name one downloaded live-test input")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        pytest.fail(f"{name} must name one absolute regular non-symlink file")
    return path


def _install_restore_inputs(root: Path) -> tuple[Path, Path]:
    archive_source = _required_input(_ARCHIVE_FILE)
    identity_source = _required_input(_IDENTITY_FILE)
    _run_checked_root(
        ("install", "--directory", "--owner=root", "--group=root", "--mode=0700", str(root))
    )
    archive = root / "backup.tar.age"
    identity = root / "recovery.agekey"
    for source, target in ((archive_source, archive), (identity_source, identity)):
        _run_checked_root(
            (
                "install",
                "--owner=root",
                "--group=root",
                "--mode=0600",
                str(source),
                str(target),
            )
        )
    return archive, identity


def _run_checked_root(argv: tuple[str, ...]) -> None:
    result = _run(("sudo", "-n", *argv), timeout=_COMMAND_TIMEOUT_SECONDS)
    if result.returncode != 0:
        pytest.fail(f"root live-test setup failed: {result.stderr[-4096:]!r}")


def _run_restore(argv: tuple[str, ...]) -> tuple[int, list[dict[str, object]]]:
    return _run_restore_process(("sudo", "-n", "/usr/local/bin/rcp", *argv))


def _run_restore_action(
    argv: tuple[str, ...],
) -> tuple[int, list[dict[str, object]]]:
    if argv[:2] != ("sudo", "/usr/local/bin/rcp"):
        pytest.fail("restore returned an unsupported resume command")
    return _run_restore_process((*argv, "--machine-readable"))


def _run_restore_process(
    argv: tuple[str, ...],
) -> tuple[int, list[dict[str, object]]]:
    result = _run(argv, timeout=_COMMAND_TIMEOUT_SECONDS)
    if result.returncode not in {0, 3}:
        pytest.fail(
            f"server restore returned {result.returncode}; stdout tail={result.stdout[-4096:]!r}; "
            f"stderr tail={result.stderr[-4096:]!r}"
        )
    events = []
    for line in result.stdout.splitlines():
        try:
            event = _EVENT_ADAPTER.validate_json(line)
        except Exception:
            pytest.fail("server restore mixed non-JSON output into its event stream")
        events.append(event.model_dump(mode="json"))
    if not events or events[0]["event"] != "plan":
        pytest.fail("server restore did not emit its plan first")
    return result.returncode, events


def _single_resume(step: dict[str, object]) -> tuple[str, ...]:
    raw = step.get("resume_argv")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        pytest.fail("restore pause did not return one exact resume command")
    return tuple(raw)


def _resume_after_github_grant(
    grant: dict[str, object],
) -> tuple[int, list[dict[str, object]]]:
    resume = _single_resume(grant)
    expected_fields = _fields(grant)
    deadline = time.monotonic() + _GITHUB_GRANT_PROPAGATION_TIMEOUT_SECONDS
    while True:
        code, events = _run_restore_action(resume)
        terminal = events[-1].get("step")
        if not isinstance(terminal, dict):
            pytest.fail("restore did not end with one terminal step")
        if terminal.get("phase") != "restore_checkouts":
            return code, events
        if (
            code != 3
            or terminal.get("state") != "operator_action_needed"
            or _fields(terminal) != expected_fields
            or _single_resume(terminal) != resume
        ):
            pytest.fail("restore changed its GitHub grant pause while the grant propagated")
        if time.monotonic() >= deadline:
            pytest.fail("GitHub did not activate the confirmed write deploy key within one minute")
        time.sleep(2)


def _command_containing(step: dict[str, object], value: str) -> tuple[str, ...]:
    matches = [command for command in _command_actions(step) if value in command]
    if len(matches) != 1:
        pytest.fail(f"restore pause did not return one command containing {value!r}")
    return matches[0]


def _command_ending_with(step: dict[str, object], value: str) -> tuple[str, ...]:
    matches = [command for command in _command_actions(step) if command[-1:] == (value,)]
    if len(matches) != 1:
        pytest.fail(f"restore pause did not return one command ending with {value!r}")
    return matches[0]


def _exchange_token(token: str) -> str:
    _identity, headers = _http_json(
        "POST",
        "/api/team/session/exchange",
        {"token": token},
    )
    cookies = http.cookies.SimpleCookie()
    cookies.load(headers.get("Set-Cookie", ""))
    if len(cookies) != 1:
        pytest.fail("restored token exchange did not return one browser session")
    morsel = next(iter(cookies.values()))
    return f"{morsel.key}={morsel.value}"


def _token_exchange_status(token: str) -> int:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        "http://127.0.0.1:8421/api/team/session/exchange",
        method="POST",
        data=json.dumps({"token": token}).encode(),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read(2 * 1024 * 1024 + 1)
            return response.status
    except urllib.error.HTTPError as exc:
        exc.read(2 * 1024 * 1024 + 1)
        return exc.code
    except urllib.error.URLError:
        pytest.fail("the restored team API was unreachable")


def _restore_key_receipt_path() -> Path:
    raw = os.environ.get(_DEPLOY_KEY_RECEIPT_FILE)
    if not raw:
        pytest.fail(f"{_DEPLOY_KEY_RECEIPT_FILE} must name one cleanup receipt")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.parent.is_dir():
        pytest.fail(f"{_DEPLOY_KEY_RECEIPT_FILE} must be an absolute new path")
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o022:
        pytest.fail("the restore deploy-key receipt parent is unsafe")
    return path


def _write_restore_key_receipt(label: str) -> None:
    if _RESTORE_KEY_LABEL.fullmatch(label) is None:
        pytest.fail("the restore deploy-key label is unsafe for cleanup")
    path = _restore_key_receipt_path()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, f"{label}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clear_restore_key_receipt() -> None:
    path = _restore_key_receipt_path()
    if path.exists():
        path.unlink()
