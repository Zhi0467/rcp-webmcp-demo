"""Destructive, explicitly gated qualification of source install and rollback.

This test is intentionally absent from ordinary CI execution. It owns an entire
disposable Ubuntu host, installs system state, and temporarily adds one read-only
deploy key to the private source repository. See ``docs/server.md``.
"""

from __future__ import annotations

import contextlib
import http.cookies
import json
import os
import pty
import pwd
import re
import select
import selectors
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import pytest
from pydantic import TypeAdapter

from rcp.server_ops.models import ServerCommandEvent
from rcp.server_ops.update import BuiltCandidateReceipt

_LIVE_GATE = "RCP_RUN_SERVER_INSTALL_LIVE"
_DISPOSABLE_CONFIRMATION = "RCP_SERVER_INSTALL_LIVE_DISPOSABLE"
_EXPECTED_DISPOSABLE_CONFIRMATION = "I_UNDERSTAND_THIS_MUTATES_THE_HOST"
_TOKEN_FILE = "RCP_LIVE_GITHUB_ADMIN_TOKEN_FILE"
_DEPLOY_KEY_RECEIPT_FILE = "RCP_LIVE_DEPLOY_KEY_RECEIPT_FILE"
_BACKUP_ARCHIVE_RECEIPT_FILE = "RCP_LIVE_BACKUP_ARCHIVE_RECEIPT_FILE"
_BACKUP_IDENTITY_FILE = "RCP_LIVE_BACKUP_IDENTITY_FILE"
_BACKUP_METADATA_FILE = "RCP_LIVE_BACKUP_METADATA_FILE"
_REPOSITORY = "example/research-control-panel"
_TEAM_NAME = "RCP live install qualification"
_GITHUB_ED25519_FINGERPRINT = "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU"
_BOOTSTRAP_CODE = re.compile(r"rcp_bootstrap_[A-Za-z0-9_-]{16}\.[A-Za-z0-9_-]{43}")
_REQUEST_ID = "00000000-0000-4000-8000-000000000000"
_COMMAND_TIMEOUT_SECONDS = 45 * 60
_PTY_TIMEOUT_SECONDS = 60
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024
_MAX_COMMAND_OUTPUT_BYTES = 32 * 1024 * 1024
_EVENT_ADAPTER = TypeAdapter(ServerCommandEvent)
_CANDIDATE_APP_MARKER = "candidate-only-f6d-live.txt"
_CANDIDATE_RESEARCH_MARKER = "candidate-only-f6d-live.txt"
_ROLLBACK_JOURNAL_PHASES = ("prepared", "quarantined", "restored", "verified", "complete")

_LIVE_TEST_ONLY = pytest.mark.skipif(
    os.environ.get(_LIVE_GATE) != "1",
    reason="destructive disposable-host server-install qualification is disabled",
)


@_LIVE_TEST_ONLY
def test_source_server_install_on_disposable_ubuntu() -> None:
    """Drive the documented install, removal, SSH, and service readback."""

    _require_explicit_disposable_host()
    workspace = _workspace()
    token = _read_admin_token()
    bootstrap_parent = Path(tempfile.mkdtemp(prefix="rcp-server-install-live-"))
    bootstrap = bootstrap_parent / "rcp-bootstrap"
    deploy_key_id: int | None = None

    try:
        _prepare_bootstrap(workspace, bootstrap)
        executable = bootstrap / ".venv" / "bin" / "rcp"

        first_code, first_events = _run_install(executable, cwd=bootstrap)
        assert first_code == 3
        source_pause = _terminal_step(first_events, "source_grant")
        assert source_pause["state"] == "operator_action_needed"
        source_fields = _fields(source_pause)
        assert source_fields["deploy_key_label"].startswith("rcp-source:")
        assert source_fields["public_key_fingerprint"].startswith("SHA256:")

        _write_deploy_key_receipt(str(source_fields["deploy_key_label"]))
        deploy_key_id = _create_read_only_deploy_key(
            token,
            title=str(source_fields["deploy_key_label"]),
            public_key=str(source_fields["deploy_public_key"]),
        )
        trust_argv = _command_actions(source_pause)[0]
        trust_code, trust_output = _run_pty(trust_argv, answer_host_key=True)
        assert trust_code == 1, (
            "GitHub's successful no-shell SSH probe must exit 1; "
            f"output tail={trust_output[-4096:]!r}"
        )
        assert _GITHUB_ED25519_FINGERPRINT in trust_output
        assert "successfully authenticated" in trust_output.lower()

        second_code, second_events = _run_install(executable, cwd=bootstrap)
        assert second_code == 3
        init_pause = _terminal_step(second_events, "team_space_init")
        assert init_pause["state"] == "operator_action_needed"
        init_commands = _command_actions(init_pause)
        assert len(init_commands) == 1

        shutil.rmtree(bootstrap_parent)
        assert not bootstrap_parent.exists()

        init_code, init_output = _run_pty(init_commands[0])
        assert init_code == 0, "interactive team initialization failed"
        bootstrap_codes = _BOOTSTRAP_CODE.findall(init_output)
        if len(bootstrap_codes) != 1:
            pytest.fail("interactive team initialization did not show exactly one bootstrap code")
        bootstrap_code = bootstrap_codes[0]

        final_code, final_events = _run_install(Path("/usr/local/bin/rcp"), cwd=Path("/tmp"))
        assert final_code == 0
        final_step = _terminal_step(final_events, "service_activate")
        assert final_step["state"] == "succeeded"
        assert _fields(final_step) == {
            "status": "ok",
            "space_kind": "team",
            "space_name": _TEAM_NAME,
        }
        health = json.loads(
            _run_checked(("curl", "--fail", "--silent", "http://127.0.0.1:8421/api/health")).stdout
        )
        assert health["status"] == "ok"
        assert health["space_kind"] == "team"
        assert health["space_name"] == _TEAM_NAME

        _assert_installed_ownership_and_modes()
        _assert_service_process_and_listener()
        first_control = _probe_private_control_socket()
        assert first_control["space_id"] == health["space_id"]
        first_doctor = _run_doctor()
        assert first_doctor["overall_state"] == "healthy"
        assert first_doctor["managed_main_head"] == os.environ.get("GITHUB_SHA")
        assert first_doctor["upstream_head"] == os.environ.get("GITHUB_SHA")
        assert first_doctor["candidate_commit"] == "none"
        assert first_doctor["current_commit"] == os.environ.get("GITHUB_SHA")
        assert first_doctor["running_commit"] == os.environ.get("GITHUB_SHA")
        assert first_doctor["release_state"] == "aligned"
        assert first_doctor["source_state"] == "aligned"
        assert first_doctor["current_web_build_id"] == first_doctor["running_web_build_id"]
        assert str(first_doctor["running_web_build_id"]).startswith("sha256:")
        assert first_doctor["service_active_state"] == "active"
        assert first_doctor["service_unit_file_state"] == "enabled"
        assert first_doctor["reload_mode"] == "disabled"
        assert first_doctor["space_id"] == health["space_id"]
        assert first_doctor["instance_id"] == first_control["instance_id"]
        assert first_doctor["process_pid"] == first_control["pid"]
        assert first_doctor["service_main_pid"] == first_control["pid"]
        assert first_doctor["data_dir_id"] == first_control["data_dir_id"]
        assert first_doctor["control_socket_status"] == "healthy"
        assert first_doctor["dependencies_ready"] is True
        assert first_doctor["problems"] == "none"
        assert health["running_commit"] == first_doctor["running_commit"]
        assert health["web_build_id"] == first_doctor["running_web_build_id"]
        _assert_password_refused_and_public_key_accepted()
        _assert_narrow_operator_rule()
        member_id, session_cookie, member_token = _enroll_live_member(bootstrap_code)

        journal = _run_checked(
            ("sudo", "-n", "journalctl", "--unit=rcp.service", "--no-pager", "--output=cat")
        ).stdout
        if "rcp_bootstrap_" in journal:
            pytest.fail("the one-time team bootstrap code entered the service journal")

        _delete_deploy_key(token, deploy_key_id)
        deploy_key_id = None
        _clear_deploy_key_receipt()
        _run_checked(("sudo", "-n", "systemctl", "restart", "rcp.service"))
        restarted = json.loads(
            _run_checked(
                (
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "30",
                    "--retry-connrefused",
                    "--retry-delay",
                    "1",
                    "--retry-max-time",
                    "15",
                    "--max-time",
                    "20",
                    "http://127.0.0.1:8421/api/health",
                ),
                timeout=30,
            ).stdout
        )
        assert restarted["status"] == "ok"
        assert restarted["space_kind"] == "team"
        restarted_control = _probe_private_control_socket()
        assert restarted_control["space_id"] == first_control["space_id"]
        assert restarted_control["instance_id"] != first_control["instance_id"]
        restarted_doctor = _run_doctor()
        assert restarted_doctor["overall_state"] == "healthy"
        assert restarted_doctor["running_commit"] == first_doctor["running_commit"]
        assert restarted_doctor["running_web_build_id"] == first_doctor["running_web_build_id"]
        assert restarted_doctor["space_id"] == first_doctor["space_id"]
        assert restarted_doctor["instance_id"] == restarted_control["instance_id"]
        assert restarted_doctor["process_pid"] == restarted_control["pid"]

        project = _create_live_update_project(member_id)
        before_projects = _authenticated_get_json("/api/projects", session_cookie)
        assert [item["id"] for item in before_projects] == [project["project_id"]]
        _drive_live_candidate_rollback(
            workspace=workspace,
            project=project,
            base_projects=before_projects,
            session_cookie=session_cookie,
        )
        second_member_id, second_session_cookie, second_member_token = _enroll_invited_member(
            session_cookie
        )
        archive_path = _drive_live_backup()
        _write_backup_archive_receipt(archive_path)
        _write_backup_metadata(
            project_id=project["project_id"],
            surviving_member_id=member_id,
            surviving_member_token=member_token,
            stale_member_id=second_member_id,
            stale_member_token=second_member_token,
        )
        _remove_live_member(
            second_member_id,
            removed_session_cookie=second_session_cookie,
            surviving_session_cookie=session_cookie,
            expected_projects=before_projects,
        )
    finally:
        if deploy_key_id is not None:
            _delete_deploy_key(token, deploy_key_id)
            _clear_deploy_key_receipt()
        if bootstrap_parent.exists():
            shutil.rmtree(bootstrap_parent)


def _enroll_live_member(
    bootstrap_code: str,
    *,
    display_name: str = "Live update operator",
) -> tuple[str, str, str]:
    enrolled, _headers = _http_json(
        "POST",
        "/api/team/enroll",
        {"code": bootstrap_code, "display_name": display_name},
    )
    identity = enrolled.get("identity")
    token = enrolled.get("token")
    if not isinstance(identity, dict) or not isinstance(token, str):
        pytest.fail("team enrollment returned an unsupported response")
    user = identity.get("user")
    if not isinstance(user, dict) or not isinstance(user.get("user_id"), str):
        pytest.fail("team enrollment did not return one member identity")
    _session, headers = _http_json(
        "POST",
        "/api/team/session/exchange",
        {"token": token},
    )
    cookies = http.cookies.SimpleCookie()
    cookies.load(headers.get("Set-Cookie", ""))
    if len(cookies) != 1:
        pytest.fail("team session exchange did not return one cookie")
    morsel = next(iter(cookies.values()))
    return str(user["user_id"]), f"{morsel.key}={morsel.value}", token


def _enroll_invited_member(inviting_cookie: str) -> tuple[str, str, str]:
    invitation, _headers = _http_json(
        "POST",
        "/api/team/invitations",
        {},
        cookie=inviting_cookie,
    )
    if not isinstance(invitation, dict) or not isinstance(invitation.get("code"), str):
        pytest.fail("team invitation did not return one enrollment code")
    return _enroll_live_member(str(invitation["code"]), display_name="Live removal member")


def _drive_live_backup() -> Path:
    identity_path = _protected_live_output_path(_BACKUP_IDENTITY_FILE)
    if identity_path.exists():
        pytest.fail("the live backup identity output already exists")
    _run_checked(("age-keygen", "-o", str(identity_path)))
    identity_path.chmod(0o600)
    recipient = _run_checked(("age-keygen", "-y", str(identity_path))).stdout.strip()
    if not recipient.startswith("age1"):
        pytest.fail("age-keygen did not return one X25519 recovery recipient")

    destination = Path("/tmp/rcp-server-install-live-backups")
    _run_checked(
        (
            "sudo",
            "-n",
            "install",
            "--directory",
            "--owner=rcp",
            "--group=rcp",
            "--mode=0700",
            str(destination),
        )
    )
    configure_code, configure_events = _run_installed_server_command(
        (
            "server",
            "backup",
            "configure",
            "--destination",
            str(destination),
            "--recipient",
            recipient,
            "--schedule",
            "02:00",
            "--retention",
            "2",
            "--confirm",
            "--machine-readable",
        ),
        as_root=True,
    )
    assert configure_code == 0
    assert _terminal_step(configure_events, "backup_configuration_readback")["state"] == (
        "succeeded"
    )

    backup_code, backup_events = _run_installed_server_command(
        ("server", "backup", "run", "--machine-readable"),
    )
    assert backup_code == 0
    backup_step = _terminal_step(backup_events, "backup_run")
    assert backup_step["state"] == "succeeded"
    backup_fields = _fields(backup_step)
    assert backup_fields["backup_status"] == "protected"
    assert backup_fields["uncaptured_projects"] == 0
    archive_path = Path(str(backup_fields["archive_path"]))
    if archive_path.parent != destination or not _root_path_exists_or_is_symlink(archive_path):
        pytest.fail("the protected backup archive was not published in the reviewed destination")
    return archive_path


def _remove_live_member(
    member_id: str,
    *,
    removed_session_cookie: str,
    surviving_session_cookie: str,
    expected_projects: list[dict[str, object]],
) -> None:
    preview_code, preview_events = _run_installed_server_command(
        ("server", "member", "remove", member_id, "--machine-readable"),
    )
    assert preview_code == 3
    preview = _terminal_step(preview_events, "member_removal")
    assert preview["state"] == "operator_action_needed"
    commands = _command_actions(preview)
    if len(commands) != 1:
        pytest.fail("member-removal preview did not return one exact confirmation command")
    confirmation = commands[0]
    if confirmation[:5] != ("rcp", "server", "member", "remove", member_id):
        pytest.fail("member-removal confirmation changed the reviewed member")

    confirmed_code, confirmed_events = _run_installed_server_command(
        (*confirmation[1:], "--machine-readable"),
    )
    assert confirmed_code == 0
    final = _terminal_step(confirmed_events, "member_removal")
    assert final["state"] == "succeeded"
    assert _fields(final)["removal_state"] == "removed"

    assert _http_status("GET", "/api/identity", cookie=removed_session_cookie) == 401
    assert _authenticated_get_json("/api/projects", surviving_session_cookie) == expected_projects
    doctor = _run_doctor()
    assert doctor["overall_state"] == "healthy"


def _http_json(
    method: str,
    path: str,
    body: dict[str, object] | None = None,
    *,
    cookie: str | None = None,
) -> tuple[dict[str, object] | list[object], object]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if cookie is not None:
        headers["Cookie"] = cookie
    request = urllib.request.Request(
        f"http://127.0.0.1:8421{path}",
        method=method,
        data=payload,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read(2 * 1024 * 1024 + 1)
            response_headers = response.headers
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        pytest.fail(f"live team API request {path} failed: {type(exc).__name__}")
    if len(content) > 2 * 1024 * 1024:
        pytest.fail("live team API response exceeded its fixed bound")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        pytest.fail("live team API returned invalid JSON")
    if not isinstance(value, (dict, list)):
        pytest.fail("live team API returned an unsupported JSON value")
    return value, response_headers


def _http_status(method: str, path: str, *, cookie: str | None = None) -> int:
    headers = {"Accept": "application/json"}
    if cookie is not None:
        headers["Cookie"] = cookie
    request = urllib.request.Request(
        f"http://127.0.0.1:8421{path}",
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read(2 * 1024 * 1024 + 1)
            return response.status
    except urllib.error.HTTPError as exc:
        exc.read(2 * 1024 * 1024 + 1)
        return exc.code
    except urllib.error.URLError:
        pytest.fail(f"live team API request {path} was unreachable")


def _authenticated_get_json(path: str, cookie: str) -> list[dict[str, object]]:
    value, _headers = _http_json("GET", path, cookie=cookie)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        pytest.fail("authenticated live read did not return an object list")
    return value


def _create_live_update_project(member_id: str) -> dict[str, str]:
    script = r"""
import json
import subprocess
import sys
from pathlib import Path

from rcp.api import create_app
from rcp.config import AGENT_EXECUTION_PROFILES
from rcp.core.models import AuthorizedHuman
from rcp.providers import configured_runtime_id
from rcp.projects import inspect_backup_project_registration
from rcp.server_ops.backup_checkout import verify_checkout_identities
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.storage import (
    ProjectProvisioningGitCheckRecord,
    ProjectProvisioningMachineIntent,
    ProjectProvisioningProviderCheckRecord,
    ProjectProvisioningProviderIntent,
    ProjectProvisioningRepositoryIntent,
)

member_id = sys.argv[1]
app = create_app(data_dir=DEFAULT_SERVER_LAYOUT.data_dir)
store = app.state.services.store
member = store.space_user(member_id)
if member is None:
    raise RuntimeError("live update member disappeared")
authorized = AuthorizedHuman(
    space_id=store.space_id,
    user_id=member.user_id,
    display_name=member.display_name,
)
repository_ref = parse_github_repository_ref("git@github.com:example/research-control-panel.git")
request = store.create_project_provisioning_request(
    kind="create_team_project",
    authorized_by=authorized,
    name="Live rollback project",
    state_repository="paper",
    project_truth_scope=["paper"],
    default_run_truth_scope=["paper"],
    machines=[
        ProjectProvisioningMachineIntent(
            alias="server",
            location="local",
            os_account="rcp",
            central_root=str(DEFAULT_SERVER_LAYOUT.projects_root),
        )
    ],
    repositories=[
        ProjectProvisioningRepositoryIntent(
            alias="paper",
            repository=repository_ref,
            machine_alias="server",
        )
    ],
    provider_checks=[
        ProjectProvisioningProviderIntent(
            profile=profile,
            provider="codex",
            runtime_id="codex:exec",
            model="gpt-test",
            reasoning="medium",
            machine_alias="server",
        )
        for profile in AGENT_EXECUTION_PROFILES
    ],
)
running = store.transition_project_provisioning_request(
    request.request_id,
    receipt_id="live-update-start",
    phase="provisioning_start",
    expected_revision=request.revision,
    expected_status="waiting_for_server_setup",
    to_status="setup_in_progress",
    machines=request.machines,
    repositories=request.repositories,
    provider_checks=request.provider_checks,
)
repository = DEFAULT_SERVER_LAYOUT.project_repository_dir(request.proposed_project_id, "paper")
subprocess.run(
    (
        "git",
        "clone",
        "--no-hardlinks",
        str(DEFAULT_SERVER_LAYOUT.source_checkout),
        str(repository),
    ),
    check=True,
)
subprocess.run(
    (
        "git",
        "-C",
        str(repository),
        "remote",
        "set-url",
        "origin",
        repository_ref.ssh_clone_url,
    ),
    check=True,
)
commit = subprocess.run(
    ("git", "-C", str(repository), "rev-parse", "HEAD"),
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
checked_at = store.now()
machines = [
    running.machines[0].model_copy(
        update={"resolved_central_root": str(DEFAULT_SERVER_LAYOUT.projects_root)}
    )
]
repositories = [
    running.repositories[0].model_copy(
        update={
            "resolved_path": str(repository),
            "checkout_disposition": "request_created",
            "git_check": ProjectProvisioningGitCheckRecord(
                status="ready",
                commit=commit,
                write_verified=True,
                deploy_key_label=(
                    f"rcp:{store.space_id}:{request.proposed_project_id}:paper"
                ),
                public_key_fingerprint="SHA256:" + ("A" * 43),
                checked_at=checked_at,
            ),
        }
    )
]
providers = [
    ProjectProvisioningProviderCheckRecord(
        **check.model_dump(
            mode="json",
            exclude={
                "status",
                "binary_path",
                "version",
                "resolved_runtime_id",
                "execution_account",
                "checked_at",
                "diagnostic",
            },
        ),
        status="ready",
        binary_path="/bin/true",
        version="live-fixture",
        resolved_runtime_id=configured_runtime_id("codex", "exec"),
        execution_account="rcp",
        checked_at=checked_at,
    )
    for check in running.provider_checks
]
ready = store.transition_project_provisioning_request(
    request.request_id,
    receipt_id="live-update-ready",
    phase="provisioning_review",
    expected_revision=running.revision,
    expected_status="setup_in_progress",
    to_status="ready_for_review",
    machines=machines,
    repositories=repositories,
    provider_checks=providers,
)
card = app.state.setup.create_prepared_team_project(ready, seat_member=member.user_id)
completed = store.transition_project_provisioning_request(
    request.request_id,
    receipt_id=f"live-update-complete:{member.user_id}",
    phase="member_finalize",
    expected_revision=ready.revision,
    expected_status="ready_for_review",
    to_status="completed",
    machines=ready.machines,
    repositories=ready.repositories,
    provider_checks=ready.provider_checks,
)
if card["id"] != completed.proposed_project_id:
    raise RuntimeError("live project finalization changed its reserved identity")
record = store.project(completed.proposed_project_id)
if record is None:
    raise RuntimeError("live project finalization lost its catalog record")
registration = inspect_backup_project_registration(
    record,
    data_dir=DEFAULT_SERVER_LAYOUT.data_dir,
    provisioning_requests=store.completed_project_provisioning_requests(
        completed.proposed_project_id
    ),
)
verify_checkout_identities(registration.recovery)
canonical_plan = registration.workspace.backup_canonical_source_plan()
if not canonical_plan.complete:
    raise RuntimeError("live project canonical backup plan is incomplete")
print(json.dumps({
    "project_id": completed.proposed_project_id,
    "repository": str(repository),
    "research": str(repository / ".research"),
}, sort_keys=True))
"""
    _run_checked(("sudo", "-n", "systemctl", "stop", "rcp.service"))
    result = _run(
        (
            "sudo",
            "-n",
            "-u",
            "rcp",
            "-H",
            "/etc/rcp/current/.venv/bin/python",
            "-c",
            script,
            member_id,
        ),
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    _run_checked(("sudo", "-n", "systemctl", "start", "rcp.service"))
    _wait_for_team_health()
    if result.returncode != 0:
        pytest.fail(
            "live update project setup failed; "
            f"stdout tail={result.stdout[-4096:]!r}; stderr tail={result.stderr[-4096:]!r}"
        )
    try:
        project = json.loads(result.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        pytest.fail("live update project setup returned invalid JSON")
    if set(project) != {"project_id", "repository", "research"} or not all(
        isinstance(value, str) for value in project.values()
    ):
        pytest.fail("live update project setup returned an unsupported shape")
    return project


def _drive_live_candidate_rollback(
    *,
    workspace: Path,
    project: dict[str, str],
    base_projects: list[dict[str, object]],
    session_cookie: str,
) -> None:
    candidate_commit, release = _build_failing_candidate(workspace, Path(project["research"]))
    receipt_path, preflight_path, base_control, base_doctor = _prepare_live_cutover(
        candidate_commit,
        release,
    )
    outcome = _run_cutover(receipt_path, preflight_path, candidate_commit)
    if outcome.get("operation_state") != "rolled_back":
        pytest.fail(f"forced live candidate was not rolled back: {outcome!r}")
    assert outcome["candidate_commit"] == candidate_commit
    assert outcome["running_commit"] == base_doctor["running_commit"]
    assert "forced live candidate readback failure" in str(outcome.get("failure"))

    _assert_live_rollback(
        outcome=outcome,
        candidate_commit=candidate_commit,
        base_commit=str(base_doctor["running_commit"]),
        project=project,
        base_projects=base_projects,
        session_cookie=session_cookie,
    )
    for phase in _ROLLBACK_JOURNAL_PHASES:
        _drive_live_root_death_during_rollback(
            phase=phase,
            candidate_commit=candidate_commit,
            release=release,
            project=project,
            base_projects=base_projects,
            session_cookie=session_cookie,
        )


def _prepare_live_cutover(
    candidate_commit: str,
    release: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    base_control = _probe_private_control_socket()
    base_doctor = _run_doctor()
    receipt_path = Path(
        f"/home/rcp/rcp-server/update-checkpoints/built-candidate-{candidate_commit}.json"
    )
    receipt = BuiltCandidateReceipt(
        installation_id=str(base_doctor["installation_id"]),
        source_origin=str(base_doctor["configured_origin"]),
        base_current_commit=str(base_doctor["current_commit"]),
        base_running_commit=str(base_doctor["running_commit"]),
        base_instance_id=str(base_control["instance_id"]),
        base_process_pid=int(base_control["pid"]),
        candidate_commit=candidate_commit,
        release_path=str(release),
        receipt_path=str(receipt_path),
        web_build_id=_candidate_web_build_identity(release),
        prepared_at=datetime.now(UTC),
    )
    descriptor, receipt_name = tempfile.mkstemp(prefix="rcp-live-built-candidate-")
    os.close(descriptor)
    temporary_receipt = Path(receipt_name)
    try:
        temporary_receipt.write_text(receipt.model_dump_json() + "\n", encoding="utf-8")
        temporary_receipt.chmod(0o600)
        # This synthetic candidate is reused across destructive failure drives,
        # while each receipt remains bound to the newly restarted base process.
        _run_checked(("sudo", "-n", "rm", "-f", "--", str(receipt_path)))
        _run_checked(
            (
                "sudo",
                "-n",
                "install",
                "--owner=rcp",
                "--group=rcp",
                "--mode=0600",
                str(temporary_receipt),
                str(receipt_path),
            )
        )
    finally:
        temporary_receipt.unlink(missing_ok=True)

    rehearsal = _run(
        (
            "sudo",
            "-n",
            "-u",
            "rcp",
            "-H",
            "/etc/rcp/current/.venv/bin/python",
            "-m",
            "rcp.server_ops.rehearsal",
            "--orchestrate",
            str(receipt_path),
            "/home/rcp/rcp-server/data",
            "/home/rcp/rcp-server/update-checkpoints",
        ),
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    if rehearsal.returncode != 0:
        pytest.fail(
            "synthetic candidate rehearsal failed; "
            f"stdout tail={rehearsal.stdout[-4096:]!r}; "
            f"stderr tail={rehearsal.stderr[-4096:]!r}"
        )
    rehearsal_lines = rehearsal.stdout.splitlines()
    if len(rehearsal_lines) != 1:
        pytest.fail("synthetic candidate rehearsal did not return one receipt path")
    preflight_path = Path(rehearsal_lines[0])
    return receipt_path, preflight_path, base_control, base_doctor


def _candidate_web_build_identity(release: Path) -> str:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from rcp.server_runtime import web_build_identity\n"
        "print(web_build_identity(Path(sys.argv[1])))\n"
    )
    return _run_checked(
        (
            "sudo",
            "-n",
            "-u",
            "rcp",
            "-H",
            str(release / ".venv" / "bin" / "python"),
            "-c",
            script,
            str(release / "web" / "dist"),
        )
    ).stdout.strip()


def _assert_live_rollback(
    *,
    outcome: dict[str, object],
    candidate_commit: str,
    base_commit: str,
    project: dict[str, str],
    base_projects: list[dict[str, object]],
    session_cookie: str,
) -> None:
    _wait_for_team_health()
    after_projects = _authenticated_get_json("/api/projects", session_cookie)
    assert after_projects == base_projects
    after_doctor = _run_doctor()
    assert after_doctor["overall_state"] == "healthy"
    assert after_doctor["current_commit"] == base_commit
    assert after_doctor["running_commit"] == base_commit
    assert after_doctor["update_operation_state"] == "rolled_back"
    assert after_doctor["update_candidate_commit"] == candidate_commit
    assert after_doctor["update_restored_commit"] == base_commit

    operation = _read_root_json(Path(str(outcome["receipt_path"])))
    checkpoint = _read_root_json(Path(str(operation["checkpoint_path"])))
    roots = checkpoint.get("roots")
    if not isinstance(roots, list):
        pytest.fail("live rollback checkpoint has no typed replacement roots")
    app_root = next(
        (item for item in roots if isinstance(item, dict) and item.get("kind") == "app_data"),
        None,
    )
    project_root = next(
        (
            item
            for item in roots
            if isinstance(item, dict)
            and item.get("kind") == "project_research"
            and item.get("identity") == project["project_id"]
        ),
        None,
    )
    if app_root is None or project_root is None:
        pytest.fail("live rollback checkpoint omitted an expected replacement root")
    live_app_marker = Path(str(app_root["live_path"])) / _CANDIDATE_APP_MARKER
    live_research_marker = Path(str(project_root["live_path"])) / _CANDIDATE_RESEARCH_MARKER
    quarantined_app_marker = Path(str(app_root["quarantine_path"])) / _CANDIDATE_APP_MARKER
    quarantined_research_marker = (
        Path(str(project_root["quarantine_path"])) / _CANDIDATE_RESEARCH_MARKER
    )
    assert not _root_path_exists_or_is_symlink(live_app_marker)
    assert not _root_path_exists_or_is_symlink(live_research_marker)
    assert _root_path_exists_or_is_symlink(quarantined_app_marker)
    assert _root_path_exists_or_is_symlink(quarantined_research_marker)
    assert _run_checked(("sudo", "-n", "cat", "--", str(quarantined_app_marker))).stdout == (
        "candidate app data\n"
    )
    assert (
        _run_checked(("sudo", "-n", "cat", "--", str(quarantined_research_marker))).stdout
        == "candidate project data\n"
    )


def _drive_live_root_death_during_rollback(
    *,
    phase: str,
    candidate_commit: str,
    release: Path,
    project: dict[str, str],
    base_projects: list[dict[str, object]],
    session_cookie: str,
) -> None:
    receipt_path, preflight_path, _base_control, base_doctor = _prepare_live_cutover(
        candidate_commit,
        release,
    )
    marker_descriptor, marker_name = tempfile.mkstemp(prefix=f"rcp-live-{phase}-")
    os.close(marker_descriptor)
    marker_path = Path(marker_name)
    # Linux's protected_regular hardening prevents the root coordinator from
    # truncating a different user's file in /tmp. Root owns the marker while
    # the runner retains read access for synchronization.
    _run_checked(("sudo", "-n", "chown", "root:root", str(marker_path)))
    _run_checked(("sudo", "-n", "chmod", "0644", str(marker_path)))
    process = _start_crashing_cutover(
        receipt_path,
        preflight_path,
        candidate_commit,
        phase=phase,
        marker_path=marker_path,
    )
    try:
        journal_path = _wait_for_cutover_crash_marker(marker_path, process)
        _wait_for_rollback_phase(journal_path, phase, process)
        _kill_privileged_process_group(process)
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode is None or process.returncode == 0:
            pytest.fail(f"root coordinator was not killed at rollback phase {phase}")
        if len(stdout) + len(stderr) > _MAX_COMMAND_OUTPUT_BYTES:
            pytest.fail("killed root coordinator exceeded its output bound")

        operation = _read_active_update_operation()
        assert operation["state"] == "rollback_restoring"
        assert operation["candidate_commit"] == candidate_commit
        assert operation["base_commit"] == base_doctor["running_commit"]

        # Depending on the killed phase, the replacement roots may be absent or
        # already restored. A direct systemd start may therefore fail closed or
        # publish only the update-fenced control plane; it must never serve HTTP.
        _run(
            ("sudo", "-n", "systemctl", "start", "rcp.service"),
            timeout=_PTY_TIMEOUT_SECONDS,
        )
        active = (
            _run(
                ("sudo", "-n", "systemctl", "is-active", "--quiet", "rcp.service"),
                timeout=_PTY_TIMEOUT_SECONDS,
            ).returncode
            == 0
        )
        if active:
            _wait_for_private_control_socket_or_service_stop()
        response = _run(
            (
                "curl",
                "--silent",
                "--output",
                "/dev/null",
                "--write-out",
                "%{http_code}",
                "http://127.0.0.1:8421/api/health",
            ),
            timeout=_PTY_TIMEOUT_SECONDS,
        )
        assert response.stdout != "200"
        assert _read_active_update_operation()["state"] == "rollback_restoring"
        _run_checked(("sudo", "-n", "systemctl", "stop", "rcp.service"))

        resumed = _run(
            (
                "sudo",
                "-n",
                "/usr/local/bin/rcp",
                "server",
                "update",
                "--machine-readable",
            ),
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
        if resumed.returncode != 1:
            pytest.fail(
                f"rollback recovery returned an unexpected exit after {phase}; "
                f"stdout tail={resumed.stdout[-4096:]!r}; "
                f"stderr tail={resumed.stderr[-4096:]!r}"
            )
        recovery_events = []
        for line in resumed.stdout.splitlines():
            try:
                recovery_events.append(_EVENT_ADAPTER.validate_json(line))
            except Exception:
                pytest.fail("rollback recovery mixed non-JSON output into its event stream")
        if len(recovery_events) != 3:
            pytest.fail("rollback recovery did not return one plan and admission transition")
        assert recovery_events[0].event == "plan"
        running_step = recovery_events[1].step
        assert running_step.phase == "update_admission"
        assert running_step.state == "running"
        recovery_step = recovery_events[-1].step
        assert recovery_step.phase == "update_admission"
        assert recovery_step.state == "failed"
        assert recovery_step.message == (
            "Recovered unfinished source update as rolled_back. Rerun sudo rcp server update "
            "to begin a fresh, fully inspected command."
        )
        recovered = _read_root_json(Path(str(operation["receipt_path"])))
        assert recovered["state"] == "rolled_back"
        assert _read_root_json(journal_path)["phase"] == "complete"
        checkpoint = _read_root_json(Path(str(recovered["checkpoint_path"])))
        roots = checkpoint.get("roots")
        if not isinstance(roots, list):
            pytest.fail("recovered rollback checkpoint lost its replacement roots")
        for root in roots:
            if not isinstance(root, dict):
                pytest.fail("recovered rollback checkpoint has an invalid root")
            assert not _root_path_exists_or_is_symlink(Path(str(root["partial_path"])))

        _assert_live_rollback(
            outcome={
                "receipt_path": operation["receipt_path"],
            },
            candidate_commit=candidate_commit,
            base_commit=str(base_doctor["running_commit"]),
            project=project,
            base_projects=base_projects,
            session_cookie=session_cookie,
        )
    finally:
        if process.poll() is None:
            _kill_privileged_process_group(process)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.communicate(timeout=10)
        _run(
            ("sudo", "-n", "rm", "-f", "--", str(marker_path)),
            timeout=_PTY_TIMEOUT_SECONDS,
        )


def _build_failing_candidate(workspace: Path, research: Path) -> tuple[str, Path]:
    source_parent = Path(tempfile.mkdtemp(prefix="rcp-live-update-candidate-"))
    source_parent.chmod(0o755)
    source = source_parent / "source"
    try:
        _run_checked(("git", "clone", "--no-hardlinks", str(workspace), str(source)))
        target = source / "src" / "rcp" / "server_ops" / "update_cutover.py"
        content = target.read_text(encoding="utf-8")
        needle = (
            "def _live_read_model_digest(receipt, background, catalog, store, "
            "expected_uid: int) -> str:\n"
        )
        injection = (
            f"    Path('/home/rcp/rcp-server/data/{_CANDIDATE_APP_MARKER}').write_text(\n"
            "        'candidate app data\\n', encoding='utf-8'\n"
            "    )\n"
            f"    Path({str(research / _CANDIDATE_RESEARCH_MARKER)!r}).write_text(\n"
            "        'candidate project data\\n', encoding='utf-8'\n"
            "    )\n"
            "    raise UpdateCutoverRefused('forced live candidate readback failure')\n"
        )
        if content.count(needle) != 1:
            pytest.fail("live candidate injection lost its exact verification owner")
        target.write_text(content.replace(needle, needle + injection), encoding="utf-8")
        _run_checked(("git", "-C", str(source), "config", "user.name", "RCP Live Test"))
        _run_checked(
            (
                "git",
                "-C",
                str(source),
                "config",
                "user.email",
                "rcp-live@example.invalid",
            )
        )
        _run_checked(("git", "-C", str(source), "add", str(target.relative_to(source))))
        _run_checked(("git", "-C", str(source), "commit", "-m", "Force live cutover rollback"))
        candidate_commit = _run_checked(
            ("git", "-C", str(source), "rev-parse", "HEAD")
        ).stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", candidate_commit) is None:
            pytest.fail("synthetic candidate commit is not canonical")
        release = Path(f"/home/rcp/rcp-server/releases/{candidate_commit}")
        _run_checked(
            (
                "sudo",
                "-n",
                "-u",
                "rcp",
                "-H",
                "git",
                "clone",
                "--no-checkout",
                "--no-hardlinks",
                str(source),
                str(release),
            )
        )
        _run_checked(
            (
                "sudo",
                "-n",
                "-u",
                "rcp",
                "-H",
                "git",
                "-C",
                str(release),
                "checkout",
                "--detach",
                candidate_commit,
            )
        )
        _run_checked(
            (
                "sudo",
                "-n",
                "-u",
                "rcp",
                "-H",
                "/usr/local/bin/npm",
                "--prefix",
                str(release / "web"),
                "ci",
            )
        )
        _run_checked(
            (
                "sudo",
                "-n",
                "-u",
                "rcp",
                "-H",
                "/usr/local/bin/npm",
                "--prefix",
                str(release / "web"),
                "run",
                "build",
            )
        )
        _run_checked(
            (
                "sudo",
                "-n",
                "-u",
                "rcp",
                "-H",
                "/usr/bin/env",
                "UV_MANAGED_PYTHON=1",
                "UV_PYTHON=3.12",
                "/usr/local/bin/uv",
                "--directory",
                str(release),
                "sync",
                "--frozen",
            )
        )
        return candidate_commit, release
    finally:
        shutil.rmtree(source_parent)


_CUTOVER_DRIVER = r"""
import json
import os
import pwd
import signal
import sys
from pathlib import Path

from rcp.server_ops.rehearsal import read_verified_candidate_receipt
from rcp.server_ops.update import BuiltCandidateReceipt, LinuxUpdateMachine, UpdateTarget
from rcp.server_ops.update_checkpoint import restore_update_checkpoint

built_path = Path(sys.argv[1])
preflight_path = Path(sys.argv[2])
candidate_commit = sys.argv[3]
crash_phase = sys.argv[4] if len(sys.argv) > 4 else None
marker_path = Path(sys.argv[5]) if len(sys.argv) > 5 else None
service_uid = pwd.getpwnam("rcp").pw_uid
built = BuiltCandidateReceipt.model_validate_json(built_path.read_bytes())
preflight = read_verified_candidate_receipt(preflight_path, expected_uid=service_uid)
machine = LinuxUpdateMachine()
if crash_phase is not None:
    if crash_phase not in {"prepared", "quarantined", "restored", "verified", "complete"}:
        raise RuntimeError("unsupported rollback crash phase")
    if marker_path is None or not marker_path.is_absolute():
        raise RuntimeError("rollback crash marker must be absolute")

    def crash_during_restore(checkpoint_path, checkpoint_sha256, _base_commit):
        marker_path.write_text(
            json.dumps({
                "checkpoint_path": str(checkpoint_path),
                "journal_path": str(checkpoint_path.parent / "rollback-journal.json"),
            }, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        child = os.fork()
        if child == 0:
            try:
                account = pwd.getpwnam("rcp")
                os.setgroups([])
                os.setgid(account.pw_gid)
                os.setuid(account.pw_uid)

                def stop_at_phase(observed):
                    if observed == crash_phase:
                        os.kill(os.getpid(), signal.SIGSTOP)

                restore_update_checkpoint(
                    checkpoint_path,
                    expected_uid=account.pw_uid,
                    expected_sha256=checkpoint_sha256,
                    after_phase=stop_at_phase,
                )
            except BaseException:
                os._exit(70)
            os._exit(0)
        _pid, status = os.waitpid(child, 0)
        if status != 0:
            raise RuntimeError("rollback crash worker exited unexpectedly")

    machine._restore_cutover_checkpoint = crash_during_restore
with machine.admission():
    target = UpdateTarget(inspection=machine.inspect(), target_commit=candidate_commit)
    outcome = machine.cutover_candidate(
        target,
        built,
        preflight,
        progress=lambda _phase: None,
    )
print(json.dumps({
    "operation_id": outcome.operation_id,
    "operation_state": outcome.operation_state,
    "candidate_commit": outcome.candidate_commit,
    "running_commit": outcome.running_commit,
    "receipt_path": str(outcome.receipt_path),
    "receipt_sha256": outcome.receipt_sha256,
    "failure": outcome.failure,
}, sort_keys=True))
"""


def _run_cutover(
    built_receipt: Path,
    preflight_receipt: Path,
    candidate_commit: str,
) -> dict[str, object]:
    result = _run(
        (
            "sudo",
            "-n",
            "/etc/rcp/current/.venv/bin/python",
            "-c",
            _CUTOVER_DRIVER,
            str(built_receipt),
            str(preflight_receipt),
            candidate_commit,
        ),
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        pytest.fail(
            "live source cutover failed outside its automatic rollback path; "
            f"stdout tail={result.stdout[-4096:]!r}; stderr tail={result.stderr[-4096:]!r}"
        )
    try:
        value = json.loads(result.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        pytest.fail("live source cutover returned invalid JSON")
    if not isinstance(value, dict):
        pytest.fail("live source cutover returned an unsupported shape")
    return value


def _start_crashing_cutover(
    built_receipt: Path,
    preflight_receipt: Path,
    candidate_commit: str,
    *,
    phase: str,
    marker_path: Path,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        (
            "sudo",
            "-n",
            "/etc/rcp/current/.venv/bin/python",
            "-c",
            _CUTOVER_DRIVER,
            str(built_receipt),
            str(preflight_receipt),
            candidate_commit,
            phase,
            str(marker_path),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _wait_for_cutover_crash_marker(
    marker_path: Path,
    process: subprocess.Popen[str],
) -> Path:
    deadline = time.monotonic() + _COMMAND_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail(
                "root coordinator exited before publishing its rollback crash marker; "
                f"stdout tail={stdout[-4096:]!r}; stderr tail={stderr[-4096:]!r}"
            )
        if marker_path.stat().st_size:
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(0.05)
                continue
            if not isinstance(marker, dict) or not isinstance(marker.get("journal_path"), str):
                pytest.fail("rollback crash marker has an unsupported shape")
            journal_path = Path(marker["journal_path"])
            if not journal_path.is_absolute():
                pytest.fail("rollback crash marker named a relative journal")
            return journal_path
        time.sleep(0.05)
    _kill_privileged_process_group(process)
    pytest.fail("root coordinator did not reach rollback restoration before timeout")


def _wait_for_rollback_phase(
    journal_path: Path,
    phase: str,
    process: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + _PTY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail(
                f"root coordinator exited before rollback phase {phase}; "
                f"stdout tail={stdout[-4096:]!r}; stderr tail={stderr[-4096:]!r}"
            )
        result = _run(
            ("sudo", "-n", "cat", "--", str(journal_path)),
            timeout=_PTY_TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            try:
                journal = json.loads(result.stdout)
            except json.JSONDecodeError:
                time.sleep(0.05)
                continue
            if isinstance(journal, dict) and journal.get("phase") == phase:
                return
        time.sleep(0.05)
    _kill_privileged_process_group(process)
    pytest.fail(f"root coordinator did not stop at rollback phase {phase}")


def _read_active_update_operation() -> dict[str, object]:
    script = r"""
import json
import os
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT
from rcp.server_ops.update_cutover import active_update_operation

active = active_update_operation(
    DEFAULT_SERVER_LAYOUT.update_checkpoints_root,
    expected_uid=os.geteuid(),
)
if active is None:
    raise RuntimeError("no active update operation")
print(active[1].model_dump_json())
"""
    output = _run_checked(
        (
            "sudo",
            "-n",
            "-u",
            "rcp",
            "-H",
            "/etc/rcp/current/.venv/bin/python",
            "-c",
            script,
        ),
        timeout=_PTY_TIMEOUT_SECONDS,
    ).stdout
    try:
        operation = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail("active update operation returned invalid JSON")
    if not isinstance(operation, dict):
        pytest.fail("active update operation has an unsupported shape")
    return operation


def _wait_for_private_control_socket_or_service_stop() -> dict[str, object] | None:
    deadline = time.monotonic() + _PTY_TIMEOUT_SECONDS
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return _probe_private_control_socket()
        except (FileNotFoundError, ConnectionRefusedError, pytest.fail.Exception) as exc:
            last_error = exc
            service = _run(
                ("sudo", "-n", "systemctl", "is-active", "--quiet", "rcp.service"),
                timeout=_PTY_TIMEOUT_SECONDS,
            )
            if service.returncode == 3:
                return None
            if service.returncode != 0:
                pytest.fail("could not determine whether the fenced service remained active")
            time.sleep(0.1)
    pytest.fail(f"fenced service did not publish its control socket: {last_error}")


def _read_root_json(path: Path) -> dict[str, object]:
    output = _run_checked(("sudo", "-n", "cat", "--", str(path))).stdout
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail(f"root-owned live receipt {path.name} is invalid")
    if not isinstance(value, dict):
        pytest.fail(f"root-owned live receipt {path.name} has an unsupported shape")
    return value


def _wait_for_team_health() -> dict[str, object]:
    output = _run_checked(
        (
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "30",
            "--retry-connrefused",
            "--retry-delay",
            "1",
            "--retry-max-time",
            "30",
            "--max-time",
            "35",
            "http://127.0.0.1:8421/api/health",
        ),
        timeout=45,
    ).stdout
    try:
        health = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail("restarted live service returned invalid health JSON")
    if not isinstance(health, dict) or health.get("space_kind") != "team":
        pytest.fail("restarted live service did not return team health")
    return health


def _require_explicit_disposable_host() -> None:
    if os.environ.get(_DISPOSABLE_CONFIRMATION) != _EXPECTED_DISPOSABLE_CONFIRMATION:
        pytest.fail(
            f"set {_DISPOSABLE_CONFIRMATION}={_EXPECTED_DISPOSABLE_CONFIRMATION} only on "
            "an entire disposable host"
        )
    if os.geteuid() == 0:
        pytest.fail("run pytest as the ordinary disposable-host operator, not root")
    if os.uname().machine != "x86_64":
        pytest.fail("the live installer qualification requires x86-64")
    release = _os_release().get("VERSION_ID")
    if _os_release().get("ID") != "ubuntu" or release not in {"22.04", "24.04"}:
        pytest.fail("the live installer qualification requires Ubuntu 22.04 or 24.04")
    if os.environ.get("GITHUB_REPOSITORY") != _REPOSITORY:
        pytest.fail(f"the live source grant is fixed to {_REPOSITORY}")
    _run_checked(("sudo", "-n", "true"))
    for path in (
        Path("/etc/rcp"),
        Path("/etc/systemd/system/rcp.service"),
        Path("/etc/systemd/system/multi-user.target.wants/rcp.service"),
        Path("/etc/sudoers.d/rcp-project-provision"),
        Path("/etc/sudoers.d/rcp-project-provision-live-test"),
        Path("/home/rcp"),
        Path("/lib/systemd/system/rcp.service"),
        Path("/run/rcp"),
        Path("/usr/lib/systemd/system/rcp.service"),
        Path("/usr/local/bin/rcp"),
    ):
        if _root_path_exists_or_is_symlink(path):
            pytest.fail(f"disposable host is not clean: {path} already exists")
    for account in ("rcp", "rcp-live-operator"):
        try:
            pwd.getpwnam(account)
        except KeyError:
            pass
        else:
            pytest.fail(f"disposable host is not clean: the {account} account already exists")
    unit_state = _run_checked(
        ("sudo", "-n", "systemctl", "show", "--property=LoadState", "--value", "rcp.service")
    ).stdout.strip()
    if unit_state != "not-found":
        pytest.fail(f"disposable host is not clean: rcp.service load state is {unit_state!r}")
    listeners = _run_checked(("sudo", "-n", "ss", "--tcp", "--listening", "--numeric")).stdout
    if any(re.search(r":8421\s", line) for line in listeners.splitlines()):
        pytest.fail("disposable host is not clean: TCP port 8421 already has a listener")
    processes = _run_checked(("ps", "-eo", "args=")).stdout
    if any(_looks_like_rcp_server(line) for line in processes.splitlines()):
        pytest.fail("disposable host is not clean: an RCP server process is already running")


def _root_path_exists_or_is_symlink(path: Path) -> bool:
    for predicate in ("-e", "-L"):
        result = _run(
            ("sudo", "-n", "test", predicate, str(path)),
            timeout=_PTY_TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return True
        if result.returncode != 1:
            pytest.fail(f"could not inspect root-owned path {path}")
    return False


@pytest.mark.parametrize(
    ("return_codes", "expected"),
    [
        ((0,), True),
        ((1, 0), True),
        ((1, 1), False),
    ],
)
def test_root_path_probe_uses_sudo(
    monkeypatch: pytest.MonkeyPatch,
    return_codes: tuple[int, ...],
    expected: bool,
) -> None:
    calls: list[tuple[str, ...]] = []
    results = iter(return_codes)

    def fake_run(
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, environment, timeout
        calls.append(argv)
        return subprocess.CompletedProcess(argv, next(results), "", "")

    monkeypatch.setattr(sys.modules[__name__], "_run", fake_run)
    path = Path("/etc/sudoers.d/rcp-project-provision")

    assert _root_path_exists_or_is_symlink(path) is expected
    assert calls == [
        ("sudo", "-n", "test", predicate, str(path))
        for predicate in ("-e", "-L")[: len(return_codes)]
    ]


def test_root_path_probe_fails_on_probe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, environment, timeout
        return subprocess.CompletedProcess(argv, 2, "", "permission failure")

    monkeypatch.setattr(sys.modules[__name__], "_run", fake_run)

    with pytest.raises(pytest.fail.Exception, match="could not inspect root-owned path"):
        _root_path_exists_or_is_symlink(Path("/etc/sudoers.d/rcp-project-provision"))


def test_private_control_socket_waits_for_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {"instance_id": "ready"}
    attempts = 0

    def probe() -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("control socket is not published yet")
        return expected

    monkeypatch.setattr(sys.modules[__name__], "_probe_private_control_socket", probe)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess((), 0, "", ""),
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    assert _wait_for_private_control_socket_or_service_stop() == expected
    assert attempts == 2


def test_private_control_socket_wait_accepts_service_stopping_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def absent() -> dict[str, object]:
        raise FileNotFoundError("socket absent")

    monkeypatch.setattr(
        sys.modules[__name__],
        "_probe_private_control_socket",
        absent,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess((), 3, "", ""),
    )

    assert _wait_for_private_control_socket_or_service_stop() is None


def _looks_like_rcp_server(command: str) -> bool:
    return bool(
        re.search(r"(?:^|\s)(?:\S*/)?rcp\s+serve(?:\s|$)", command)
        or re.search(r"(?:^|\s)python\S*\s+-m\s+rcp\s+serve(?:\s|$)", command)
        or "/usr/local/bin/rcp serve" in command
    )


def _workspace() -> Path:
    raw = os.environ.get("GITHUB_WORKSPACE")
    if not raw:
        pytest.fail("GITHUB_WORKSPACE is required for the guarded live drive")
    workspace = Path(raw)
    if not workspace.is_absolute() or not (workspace / ".git").exists():
        pytest.fail("GITHUB_WORKSPACE must be an absolute Git checkout")
    status = _run_checked(("git", "-C", str(workspace), "status", "--porcelain")).stdout
    if status:
        pytest.fail("the live bootstrap source must be a clean checkout")
    head = _run_checked(("git", "-C", str(workspace), "rev-parse", "HEAD")).stdout.strip()
    expected = os.environ.get("GITHUB_SHA")
    if expected and head != expected:
        pytest.fail("GITHUB_WORKSPACE HEAD differs from GITHUB_SHA")
    return workspace


def _read_admin_token() -> str:
    raw = os.environ.get(_TOKEN_FILE)
    if not raw:
        pytest.fail(f"{_TOKEN_FILE} must name a protected fine-grained token file")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        pytest.fail(f"{_TOKEN_FILE} must be an absolute regular non-symlink file")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        pytest.fail(f"{_TOKEN_FILE} must be owned by the caller and inaccessible to group/other")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 20 or any(ord(character) < 33 or ord(character) == 127 for character in token):
        pytest.fail(f"{_TOKEN_FILE} does not contain one plausible token")
    return token


def _deploy_key_receipt_path() -> Path:
    raw = os.environ.get(_DEPLOY_KEY_RECEIPT_FILE)
    if not raw:
        pytest.fail(f"{_DEPLOY_KEY_RECEIPT_FILE} must name a protected cleanup receipt")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.parent.is_dir():
        pytest.fail(f"{_DEPLOY_KEY_RECEIPT_FILE} must be an absolute new regular-file path")
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o022:
        pytest.fail(
            f"{_DEPLOY_KEY_RECEIPT_FILE} parent must be caller-owned and not writable by others"
        )
    return path


def _write_deploy_key_receipt(label: str) -> None:
    if re.fullmatch(r"rcp-source:[0-9a-f-]{36}", label) is None:
        pytest.fail("the generated source-key label is not safe for cleanup")
    path = _deploy_key_receipt_path()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError:
        pytest.fail("the deploy-key cleanup receipt already exists or cannot be created safely")
    try:
        os.write(descriptor, f"{label}\n".encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clear_deploy_key_receipt() -> None:
    path = _deploy_key_receipt_path()
    if path.exists():
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            pytest.fail("the deploy-key cleanup receipt changed ownership or mode")
        path.unlink()


def _protected_live_output_path(environment_name: str) -> Path:
    raw = os.environ.get(environment_name)
    if not raw:
        pytest.fail(f"{environment_name} must name one protected live-test output")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.parent.is_dir():
        pytest.fail(f"{environment_name} must be an absolute non-symlink path")
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o022:
        pytest.fail(f"{environment_name} parent must be caller-owned and not writable by others")
    return path


def _write_backup_archive_receipt(archive_path: Path) -> None:
    receipt_path = _protected_live_output_path(_BACKUP_ARCHIVE_RECEIPT_FILE)
    if receipt_path.exists():
        pytest.fail("the backup archive cleanup receipt already exists")
    descriptor = os.open(
        receipt_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, f"{archive_path}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_backup_metadata(
    *,
    project_id: str,
    surviving_member_id: str,
    surviving_member_token: str,
    stale_member_id: str,
    stale_member_token: str,
) -> None:
    path = _protected_live_output_path(_BACKUP_METADATA_FILE)
    if path.exists():
        pytest.fail("the live backup metadata output already exists")
    payload = json.dumps(
        {
            "project_id": project_id,
            "surviving_member_id": surviving_member_id,
            "surviving_member_token": surviving_member_token,
            "stale_member_id": stale_member_id,
            "stale_member_token": stale_member_token,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_bootstrap(workspace: Path, bootstrap: Path) -> None:
    _run_checked(("git", "clone", "--no-hardlinks", str(workspace), str(bootstrap)))
    _run_checked(
        (
            "git",
            "-C",
            str(bootstrap),
            "remote",
            "set-url",
            "origin",
            f"https://github.com/{_REPOSITORY}.git",
        )
    )
    _run_checked(("npm", "--prefix", "web", "ci"), cwd=bootstrap)
    _run_checked(("npm", "--prefix", "web", "run", "build"), cwd=bootstrap)
    environment = os.environ.copy()
    environment.update({"UV_MANAGED_PYTHON": "1", "UV_PYTHON": "3.12"})
    _run_checked(("uv", "sync", "--frozen"), cwd=bootstrap, environment=environment)
    if not (bootstrap / ".venv" / "bin" / "rcp").is_file():
        pytest.fail("the documented bootstrap build did not create .venv/bin/rcp")


def _run_install(
    executable: Path,
    *,
    cwd: Path,
) -> tuple[int, list[dict[str, object]]]:
    result = _run(
        (
            "sudo",
            "-n",
            "/usr/bin/env",
            "PYTHONDONTWRITEBYTECODE=1",
            str(executable),
            "server",
            "install",
            "--team-name",
            _TEAM_NAME,
            "--machine-readable",
        ),
        cwd=cwd,
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    if result.returncode not in {0, 3}:
        pytest.fail(
            f"server install returned unexpected exit status {result.returncode}; "
            f"stdout tail={result.stdout[-4096:]!r}; stderr tail={result.stderr[-4096:]!r}"
        )
    lines = result.stdout.splitlines()
    if not lines:
        pytest.fail("server install emitted no machine-readable events")
    events: list[dict[str, object]] = []
    for line in lines:
        try:
            event = _EVENT_ADAPTER.validate_json(line)
        except Exception:
            pytest.fail("server install mixed non-JSON output into its machine-readable stream")
        events.append(event.model_dump(mode="json"))
    assert events[0]["event"] == "plan"
    assert events[0]["command"] == "server install"
    return result.returncode, events


def _run_installed_server_command(
    argv: tuple[str, ...],
    *,
    as_root: bool = False,
) -> tuple[int, list[dict[str, object]]]:
    prefix = ("sudo", "-n") if as_root else ("sudo", "-n", "-u", "rcp", "-H")
    result = _run(
        (*prefix, "/usr/local/bin/rcp", *argv),
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    if result.returncode not in {0, 3}:
        pytest.fail(
            f"installed server command returned unexpected exit status {result.returncode}; "
            f"stdout tail={result.stdout[-4096:]!r}; stderr tail={result.stderr[-4096:]!r}"
        )
    events: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        try:
            event = _EVENT_ADAPTER.validate_json(line)
        except Exception:
            pytest.fail("installed server command mixed non-JSON output into its event stream")
        events.append(event.model_dump(mode="json"))
    if not events or events[0]["event"] != "plan":
        pytest.fail("installed server command did not emit its plan first")
    return result.returncode, events


def _run_doctor() -> dict[str, object]:
    result = _run(
        (
            "sudo",
            "-n",
            "-u",
            "rcp",
            "-H",
            "/usr/local/bin/rcp",
            "server",
            "doctor",
            "--machine-readable",
        ),
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        pytest.fail(
            f"server doctor returned {result.returncode}; stdout tail={result.stdout[-4096:]!r}; "
            f"stderr tail={result.stderr[-4096:]!r}"
        )
    events = []
    for line in result.stdout.splitlines():
        try:
            event = _EVENT_ADAPTER.validate_json(line)
        except Exception:
            pytest.fail("server doctor mixed non-JSON output into its machine-readable stream")
        events.append(event.model_dump(mode="json"))
    if [event["event"] for event in events] != ["plan", "step", "step"]:
        pytest.fail("server doctor did not emit one complete plan and report")
    final = events[-1]["step"]
    if not isinstance(final, dict) or final.get("state") != "succeeded":
        pytest.fail("server doctor did not publish one successful final report")
    return _fields(final)


def _terminal_step(events: list[dict[str, object]], phase: str) -> dict[str, object]:
    final = events[-1]
    assert final["event"] == "step"
    step = final["step"]
    assert isinstance(step, dict)
    fields = step.get("fields")
    field_names = (
        sorted(
            str(item.get("name"))
            for item in fields
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        )
        if isinstance(fields, list)
        else []
    )
    assert step["phase"] == phase, {
        "expected_phase": phase,
        "actual_phase": step.get("phase"),
        "state": step.get("state"),
        "message": step.get("message"),
        "field_names": field_names,
    }
    return step


def _fields(step: dict[str, object]) -> dict[str, object]:
    fields = step["fields"]
    assert isinstance(fields, list)
    return {str(item["name"]): item["value"] for item in fields if isinstance(item, dict)}


def _command_actions(step: dict[str, object]) -> list[tuple[str, ...]]:
    actions = step["actions"]
    assert isinstance(actions, list)
    commands = []
    for action in actions:
        if isinstance(action, dict) and action.get("kind") == "command":
            argv = action.get("argv")
            assert isinstance(argv, list) and all(isinstance(value, str) for value in argv)
            commands.append(tuple(argv))
    return commands


def _create_read_only_deploy_key(token: str, *, title: str, public_key: str) -> int:
    response = _github_request(
        token,
        method="POST",
        path=f"/repos/{_REPOSITORY}/keys",
        body={"title": title, "key": public_key, "read_only": True},
    )
    key_id = response.get("id")
    if not isinstance(key_id, int) or response.get("read_only") is not True:
        pytest.fail("GitHub did not confirm one read-only deploy key")
    return key_id


def _delete_deploy_key(token: str, key_id: int) -> None:
    _github_request(
        token,
        method="DELETE",
        path=f"/repos/{_REPOSITORY}/keys/{key_id}",
        body=None,
    )


def _github_request(
    token: str,
    *,
    method: str,
    path: str,
    body: dict[str, object] | None,
) -> dict[str, object]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        method=method,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "rcp-server-install-live-test",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read(64 * 1024 + 1)
            status_code = response.status
    except urllib.error.HTTPError as exc:
        pytest.fail(f"GitHub deploy-key API returned HTTP {exc.code}")
    except urllib.error.URLError:
        pytest.fail("GitHub deploy-key API was unreachable")
    if len(content) > 64 * 1024:
        pytest.fail("GitHub deploy-key API response exceeded the live-test bound")
    if method == "DELETE":
        if status_code != 204:
            pytest.fail(f"GitHub deploy-key deletion returned HTTP {status_code}")
        return {}
    if status_code != 201:
        pytest.fail(f"GitHub deploy-key creation returned HTTP {status_code}")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        pytest.fail("GitHub deploy-key API returned invalid JSON")
    if not isinstance(value, dict):
        pytest.fail("GitHub deploy-key API returned a non-object")
    return value


def _run_pty(
    argv: tuple[str, ...],
    *,
    answer_host_key: bool = False,
) -> tuple[int, str]:
    pid, master = pty.fork()
    if pid == 0:
        try:
            os.execvp(argv[0], argv)
        except OSError:
            os._exit(127)
    output = bytearray()
    answered = False
    child_status: int | None = None
    deadline = time.monotonic() + _PTY_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.2)
            chunk = b""
            if readable:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    chunk = b""
                output.extend(chunk)
                if len(output) > _MAX_OUTPUT_BYTES:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pid, signal.SIGKILL)
                    pytest.fail("interactive live-test output exceeded its bound")
                text = output.decode("utf-8", errors="replace")
                if answer_host_key and not answered and "Are you sure" in text:
                    if _GITHUB_ED25519_FINGERPRINT not in text:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(pid, signal.SIGKILL)
                        pytest.fail("GitHub host prompt did not show the published fingerprint")
                    os.write(master, b"yes\n")
                    answered = True
            if child_status is None:
                waited_pid, status = os.waitpid(pid, os.WNOHANG)
                if waited_pid == pid:
                    child_status = status
            if child_status is not None and (not readable or not chunk):
                break
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
            pytest.fail("interactive live-test command timed out")
        if child_status is None:
            _, child_status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(child_status), output.decode("utf-8", errors="replace")
    finally:
        if child_status is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
        os.close(master)


def _assert_installed_ownership_and_modes() -> None:
    account = pwd.getpwnam("rcp")
    assert account.pw_dir == "/home/rcp"
    assert account.pw_shell == "/bin/bash"
    shadow = _run_checked(("sudo", "-n", "getent", "shadow", "rcp")).stdout.split(":")
    assert shadow[1] == "*NP*"

    for path in (
        Path("/home/rcp"),
        Path("/home/rcp/rcp-server"),
        Path("/home/rcp/rcp-server/data"),
        Path("/home/rcp/rcp-server/credentials"),
        Path("/home/rcp/.ssh"),
    ):
        _assert_path(
            path,
            uid=account.pw_uid,
            gid=account.pw_gid,
            mode=0o700,
            kind="directory",
        )
    _assert_path(
        Path("/home/rcp/rcp-server/credentials/source_ed25519"),
        uid=account.pw_uid,
        gid=account.pw_gid,
        mode=0o600,
        kind="file",
    )
    _assert_path(
        Path("/home/rcp/rcp-server/credentials/source_ed25519.pub"),
        uid=account.pw_uid,
        gid=account.pw_gid,
        mode=0o644,
        kind="file",
    )
    _assert_path(
        Path("/etc/rcp/server.toml"),
        uid=0,
        gid=account.pw_gid,
        mode=0o640,
        kind="file",
    )
    _assert_path(
        Path("/usr/local/bin/rcp"),
        uid=0,
        gid=0,
        mode=0o755,
        kind="file",
    )
    _assert_path(
        Path("/etc/systemd/system/rcp.service"),
        uid=0,
        gid=0,
        mode=0o644,
        kind="file",
    )
    _assert_path(
        Path("/run/rcp"),
        uid=account.pw_uid,
        gid=account.pw_gid,
        mode=0o700,
        kind="directory",
    )
    _assert_path(
        Path("/run/rcp/control.sock"),
        uid=account.pw_uid,
        gid=account.pw_gid,
        mode=0o600,
        kind="socket",
    )
    current = Path("/etc/rcp/current")
    current_uid, current_gid, _current_mode, current_raw_mode = _root_stat(current)
    assert stat.S_ISLNK(current_raw_mode)
    assert (current_uid, current_gid) == (0, 0)
    target_text = _run_checked(("sudo", "-n", "readlink", "--", str(current))).stdout.strip()
    assert target_text and "\n" not in target_text
    target = Path(target_text)
    assert target.is_absolute()
    assert target.parent == Path("/home/rcp/rcp-server/releases")
    assert re.fullmatch(r"[0-9a-f]{40}", target.name)
    target_uid, _target_gid, _target_mode, target_raw_mode = _root_stat(target)
    assert stat.S_ISDIR(target_raw_mode)
    assert target_uid == account.pw_uid


def _assert_path(path: Path, *, uid: int, gid: int, mode: int, kind: str) -> None:
    actual_uid, actual_gid, actual_mode, raw_mode = _root_stat(path)
    assert (actual_uid, actual_gid) == (uid, gid)
    assert actual_mode == mode
    if kind == "directory":
        assert stat.S_ISDIR(raw_mode)
    elif kind == "file":
        assert stat.S_ISREG(raw_mode)
    else:
        assert kind == "socket"
        assert stat.S_ISSOCK(raw_mode)


def _probe_private_control_socket() -> dict[str, object]:
    unauthorized = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(PermissionError):
            unauthorized.connect("/run/rcp/control.sock")
    finally:
        unauthorized.close()

    script = (
        "import pwd\n"
        "from pathlib import Path\n"
        "from rcp.server_ops.control import ServerControlClient\n"
        "uid = pwd.getpwnam('rcp').pw_uid\n"
        "result = ServerControlClient.from_data_dir(\n"
        "    Path('/home/rcp/rcp-server/data'), expected_server_uid=uid\n"
        ").probe()\n"
        "print(result.model_dump_json())\n"
    )
    output = _run_checked(
        (
            "sudo",
            "-n",
            "-u",
            "rcp",
            "-H",
            "/etc/rcp/current/.venv/bin/python",
            "-c",
            script,
        ),
        timeout=_PTY_TIMEOUT_SECONDS,
    ).stdout
    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail("the installed control probe returned invalid JSON")
    if set(result) != {
        "instance_id",
        "pid",
        "data_dir_id",
        "space_id",
        "space_kind",
        "operations",
        "pending_member_removals",
    }:
        pytest.fail("the installed control probe returned an unsupported shape")
    assert result["space_kind"] == "team"
    assert "update_candidate_verify" in result["operations"]
    assert result["pending_member_removals"] == []
    assert result["pid"] == int(
        _run_checked(
            (
                "sudo",
                "-n",
                "systemctl",
                "show",
                "--property=MainPID",
                "--value",
                "rcp.service",
            )
        ).stdout
    )
    return result


def _root_stat(path: Path) -> tuple[int, int, int, int]:
    output = _run_checked(
        (
            "sudo",
            "-n",
            "stat",
            "--format=%u:%g:%a:%f",
            "--",
            str(path),
        ),
        timeout=_PTY_TIMEOUT_SECONDS,
    ).stdout.strip()
    fields = output.split(":")
    if len(fields) != 4 or "\n" in output:
        pytest.fail(f"root stat returned an invalid result for {path}")
    try:
        uid, gid = (int(value) for value in fields[:2])
        mode = int(fields[2], 8)
        raw_mode = int(fields[3], 16)
    except ValueError:
        pytest.fail(f"root stat returned an invalid result for {path}")
    return uid, gid, mode, raw_mode


def _assert_service_process_and_listener() -> None:
    account = pwd.getpwnam("rcp")
    main_pid = int(
        _run_checked(
            (
                "sudo",
                "-n",
                "systemctl",
                "show",
                "--property=MainPID",
                "--value",
                "rcp.service",
            )
        ).stdout
    )
    assert main_pid > 1
    assert Path(f"/proc/{main_pid}").stat().st_uid == account.pw_uid
    listeners = _run_checked(("sudo", "-n", "ss", "--tcp", "--listening", "--numeric")).stdout
    matching = [line for line in listeners.splitlines() if re.search(r":8421\s", line)]
    assert matching
    assert all("127.0.0.1:8421" in line for line in matching)


def _assert_password_refused_and_public_key_accepted() -> None:
    _run_checked(("sudo", "-n", "systemctl", "start", "ssh.service"))
    common = (
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    )
    password = _run(
        (
            "ssh",
            *common,
            "-o",
            "BatchMode=yes",
            "-o",
            "PubkeyAuthentication=no",
            "-o",
            "PreferredAuthentications=password",
            "rcp@127.0.0.1",
            "true",
        ),
        timeout=30,
    )
    assert password.returncode == 255

    key_root = Path(tempfile.mkdtemp(prefix="rcp-live-client-key-"))
    key = key_root / "id_ed25519"
    try:
        _run_checked(("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)))
        _run_checked(
            (
                "sudo",
                "-n",
                "install",
                "--owner=rcp",
                "--group=rcp",
                "--mode=0600",
                str(key.with_suffix(".pub")),
                "/home/rcp/.ssh/authorized_keys",
            )
        )
        public_key = _run_checked(
            (
                "ssh",
                *common,
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-i",
                str(key),
                "rcp@127.0.0.1",
                "id",
                "-un",
            ),
            timeout=30,
        )
        assert public_key.stdout.strip() == "rcp"
        assert (
            _run_checked(("sudo", "-n", "systemctl", "is-active", "rcp.service")).stdout.strip()
            == "active"
        )
    finally:
        _run_checked(("sudo", "-n", "rm", "-f", "--", "/home/rcp/.ssh/authorized_keys"))
        shutil.rmtree(key_root)
    assert not _root_path_exists_or_is_symlink(Path("/home/rcp/.ssh/authorized_keys"))


def _assert_narrow_operator_rule() -> None:
    operator = "rcp-live-operator"
    target = Path("/etc/sudoers.d/rcp-project-provision-live-test")
    operator_created = False
    descriptor, rule_name = tempfile.mkstemp(prefix="rcp-live-sudoers-")
    os.close(descriptor)
    rule_source = Path(rule_name)
    try:
        _run_checked(
            (
                "sudo",
                "-n",
                "useradd",
                "--create-home",
                "--shell",
                "/bin/bash",
                "--user-group",
                operator,
            )
        )
        operator_created = True
        rule_source.write_text(
            f"{operator} ALL=(rcp) NOPASSWD: /usr/local/bin/rcp server project provision * "
            "--machine-readable\n",
            encoding="utf-8",
        )
        os.chmod(rule_source, 0o600)
        _run_checked(
            (
                "sudo",
                "-n",
                "install",
                "--owner=root",
                "--group=root",
                "--mode=0440",
                str(rule_source),
                str(target),
            )
        )
        _run_checked(("sudo", "-n", "visudo", "--check", "--file", str(target)))
        allowed = _run(
            (
                "sudo",
                "-n",
                "-u",
                operator,
                "-H",
                "sudo",
                "-n",
                "-u",
                "rcp",
                "-H",
                "/usr/local/bin/rcp",
                "server",
                "project",
                "provision",
                _REQUEST_ID,
                "--machine-readable",
            ),
            timeout=30,
        )
        assert allowed.returncode == 1
        allowed_events = [json.loads(line) for line in allowed.stdout.splitlines()]
        assert [event["event"] for event in allowed_events] == ["plan", "step"]
        assert allowed_events[-1]["step"]["phase"] == "operation_prepare"
        assert allowed_events[-1]["step"]["state"] == "failed"
        assert allowed.stderr == ""
        denied = _run(
            (
                "sudo",
                "-n",
                "-u",
                operator,
                "-H",
                "sudo",
                "-n",
                "-u",
                "rcp",
                "-H",
                "/usr/bin/id",
            ),
            timeout=30,
        )
        assert denied.returncode != 0
    finally:
        _run_checked(("sudo", "-n", "rm", "-f", "--", str(target)))
        if operator_created:
            _run_checked(("sudo", "-n", "userdel", "--remove", operator))
        rule_source.unlink(missing_ok=True)
    assert not _root_path_exists_or_is_symlink(target)
    with pytest.raises(KeyError):
        pwd.getpwnam(operator)


def _run_checked(
    argv: tuple[str, ...],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    timeout: float = _COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    result = _run(argv, cwd=cwd, environment=environment, timeout=timeout)
    if result.returncode != 0:
        pytest.fail(f"live-test command {argv[0]!r} returned {result.returncode}")
    return result


def _run(
    argv: tuple[str, ...],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    streams: dict[int, tuple[BinaryIO, bytearray]] = {
        process.stdout.fileno(): (process.stdout, stdout_buffer),
        process.stderr.fileno(): (process.stderr, stderr_buffer),
    }
    selector = selectors.DefaultSelector()
    for stream, _ in streams.values():
        selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                pytest.fail(f"live-test command {argv[0]!r} timed out")
            for key, _ in selector.select(timeout=min(remaining, 0.5)):
                stream = key.fileobj
                chunk = os.read(stream.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                streams[stream.fileno()][1].extend(chunk)
                if sum(len(output) for _, output in streams.values()) > _MAX_COMMAND_OUTPUT_BYTES:
                    _kill_process_group(process)
                    pytest.fail(f"live-test command {argv[0]!r} exceeded its output bound")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process_group(process)
            pytest.fail(f"live-test command {argv[0]!r} timed out")
        return_code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        pytest.fail(f"live-test command {argv[0]!r} timed out")
    finally:
        selector.close()
        for stream, _ in streams.values():
            stream.close()
    stdout = stdout_buffer.decode("utf-8", errors="replace")
    stderr = stderr_buffer.decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(argv, return_code, stdout, stderr)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def _kill_privileged_process_group(process: subprocess.Popen[str]) -> None:
    """Kill the root coordinator and its service-account rollback worker."""

    result = _run(
        (
            "sudo",
            "-n",
            "/usr/bin/kill",
            "-KILL",
            "--",
            f"-{process.pid}",
        ),
        timeout=_PTY_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        pytest.fail("could not kill the privileged live rollback process group")
    process.wait(timeout=10)


def test_bounded_command_runner_keeps_separate_output() -> None:
    result = _run(
        (
            sys.executable,
            "-c",
            "import sys; print('ordinary output'); print('diagnostic', file=sys.stderr)",
        ),
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout == "ordinary output\n"
    assert result.stderr == "diagnostic\n"


def test_bounded_command_runner_stops_excess_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_MAX_COMMAND_OUTPUT_BYTES", 128)

    with pytest.raises(pytest.fail.Exception, match="exceeded its output bound"):
        _run((sys.executable, "-c", "print('x' * 4096)"), timeout=5)


def test_privileged_process_group_kill_uses_sudo(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []
    waits: list[float] = []

    def fake_run(
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, environment, timeout
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    class Process:
        pid = 1234

        def wait(self, timeout: float) -> int:
            waits.append(timeout)
            return -signal.SIGKILL

    monkeypatch.setattr(sys.modules[__name__], "_run", fake_run)

    _kill_privileged_process_group(Process())  # type: ignore[arg-type]

    assert commands == [("sudo", "-n", "/usr/bin/kill", "-KILL", "--", "-1234")]
    assert waits == [10]


def test_pty_runner_supplies_controlling_terminal_for_host_confirmation() -> None:
    script = (
        "import os\n"
        "terminal = os.open('/dev/tty', os.O_RDWR)\n"
        f"os.write(terminal, b'{_GITHUB_ED25519_FINGERPRINT} Are you sure? ')\n"
        "answer = os.read(terminal, 128).decode().strip()\n"
        "os.close(terminal)\n"
        "print(f'answer={answer}')\n"
        "raise SystemExit(0 if answer == 'yes' else 2)\n"
    )

    return_code, output = _run_pty(
        (sys.executable, "-c", script),
        answer_host_key=True,
    )

    assert return_code == 0, output
    assert "answer=yes" in output


def test_root_stat_uses_noninteractive_sudo_and_parses_gnu_stat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run_checked(
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
        timeout: float = _COMMAND_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, environment, timeout
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "123:456:700:41c0\n", "")

    monkeypatch.setattr(sys.modules[__name__], "_run_checked", fake_run_checked)
    path = Path("/home/rcp/rcp-server")

    assert _root_stat(path) == (123, 456, 0o700, 0o40700)
    assert calls == [
        (
            "sudo",
            "-n",
            "stat",
            "--format=%u:%g:%a:%f",
            "--",
            str(path),
        )
    ]


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values
