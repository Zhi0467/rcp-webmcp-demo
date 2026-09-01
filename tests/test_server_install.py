from __future__ import annotations

import json
import os
import subprocess
import uuid
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from rcp.__main__ import build_parser
from rcp.server_ops import backup_config as backup_owner
from rcp.server_ops import install as server_install
from rcp.server_ops.cli import (
    SERVER_CLI_EXIT_FAILED,
    SERVER_CLI_EXIT_OPERATOR_ACTION,
    CallerIdentity,
    run_server_command,
)
from rcp.server_ops.config import ServerSourceConfig, create_installed_server_config
from rcp.server_ops.install import (
    GitHubRepository,
    HostFacts,
    InstallRefused,
    ManagedCheckout,
    ServiceHealth,
    ServiceInstallState,
    SourceAccess,
    discover_bootstrap_repository,
    normalize_github_repository,
    prepare_install_command,
)
from rcp.server_ops.layout import ServerLayout
from rcp.server_ops.update_cutover import new_update_operation, publish_update_operation

INSTALLATION_ID = "123e4567-e89b-42d3-a456-426614174000"
COMMIT = "a" * 40
REPOSITORY = GitHubRepository(
    slug="openai/rcp",
    https_origin="https://github.com/openai/rcp.git",
    ssh_origin="git@github.com:openai/rcp.git",
    deploy_keys_url="https://github.com/openai/rcp/settings/keys",
)
ROOT_IDENTITY = CallerIdentity(uid=0, username="root", host="lab.example")
BOOTSTRAP_EXECUTABLE = Path("/srv/bootstrap/.venv/bin/rcp")


class FakeInstallMachine:
    def __init__(
        self,
        *,
        access: SourceAccess,
        service_state: ServiceInstallState,
        fail_at: str | None = None,
    ) -> None:
        self.access = access
        self.service_state = service_state
        self.fail_at = fail_at
        self.calls: list[str] = []

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise InstallRefused(f"Focused refusal in {name}; correct it and rerun install.")

    def validate_host(self) -> HostFacts:
        self._call("validate_host")
        return HostFacts(ubuntu_release="24.04")

    def converge_account_and_layout(self) -> None:
        self._call("converge_account_and_layout")

    def prepare_source_access(self, repository: GitHubRepository) -> SourceAccess:
        self._call("prepare_source_access")
        assert repository == REPOSITORY
        return self.access

    def converge_source_checkout(self, access: SourceAccess) -> ManagedCheckout:
        self._call("converge_source_checkout")
        assert access == self.access
        return ManagedCheckout(commit=COMMIT, is_current_release=False)

    def build_release(self, checkout: ManagedCheckout) -> Path:
        self._call("build_release")
        assert checkout.commit == COMMIT
        return Path(f"/home/rcp/rcp-server/releases/{COMMIT}")

    def install_service(
        self,
        checkout: ManagedCheckout,
        release: Path,
    ) -> ServiceInstallState:
        self._call("install_service")
        assert checkout.commit == COMMIT
        assert release.name == COMMIT
        return self.service_state

    def activate_and_verify(self) -> ServiceHealth:
        self._call("activate_and_verify")
        return ServiceHealth(status="ok", space_kind="team", space_name="Systems Lab")


def _source_access(*, private: bool, grant_needed: bool = False) -> SourceAccess:
    if private:
        source = ServerSourceConfig(
            origin=REPOSITORY.ssh_origin,
            authentication="deploy_key",
            public_key_fingerprint="SHA256:" + ("A" * 43),
        )
    else:
        source = ServerSourceConfig(
            origin=REPOSITORY.https_origin,
            authentication="public",
        )
    config = create_installed_server_config(source=source, installation_id=INSTALLATION_ID)
    return SourceAccess(
        config=config,
        repository=REPOSITORY,
        grant_needed=grant_needed,
        deploy_key_label=f"rcp-source:{INSTALLATION_ID}" if private else None,
        public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKey rcp-source" if private else None,
    )


def _run_install(
    machine: FakeInstallMachine,
    *,
    machine_readable: bool = True,
) -> tuple[int, str]:
    argv = ["server", "install", "--team-name", "Systems Lab"]
    if machine_readable:
        argv.append("--machine-readable")
    args = build_parser().parse_args(argv)
    output = StringIO()

    def handler(request, identity):
        return prepare_install_command(
            request,
            identity,
            machine=machine,
            repository=REPOSITORY,
            resume_executable=BOOTSTRAP_EXECUTABLE,
        )

    exit_code = run_server_command(
        args,
        handler=handler,
        identity=ROOT_IDENTITY,
        stream=output,
    )
    return exit_code, output.getvalue()


def _events(output: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.splitlines()]


def test_public_fresh_install_prints_exact_init_and_resume_contract() -> None:
    machine = FakeInstallMachine(
        access=_source_access(private=False),
        service_state=ServiceInstallState(
            data_state="fresh",
            service_state="stopped_disabled",
        ),
    )

    exit_code, output = _run_install(machine)
    events = _events(output)

    assert exit_code == SERVER_CLI_EXIT_OPERATOR_ACTION
    assert len(events[0]["steps"]) == 9
    assert [step["number"] for step in events[0]["steps"]] == list(range(1, 10))
    assert machine.calls == [
        "validate_host",
        "converge_account_and_layout",
        "prepare_source_access",
        "converge_source_checkout",
        "build_release",
        "install_service",
    ]
    paused = events[-1]["step"]
    assert paused["phase"] == "team_space_init"
    assert paused["state"] == "operator_action_needed"
    assert paused["actions"][0]["argv"] == [
        "sudo",
        "-u",
        "rcp",
        "-H",
        "/usr/local/bin/rcp",
        "space",
        "init",
        "--team",
        "--name",
        "Systems Lab",
    ]
    assert len(paused["actions"]) == 2
    assert paused["resume_argv"] == [
        "sudo",
        "/usr/local/bin/rcp",
        "server",
        "install",
        "--team-name",
        "Systems Lab",
    ]
    assert all("systemctl" not in action.get("argv", []) for action in paused["actions"])


def test_initialized_install_converges_through_systemd_and_health() -> None:
    machine = FakeInstallMachine(
        access=_source_access(private=False),
        service_state=ServiceInstallState(data_state="initialized", service_state="active"),
    )

    exit_code, output = _run_install(machine)
    events = _events(output)

    assert exit_code == 0
    assert machine.calls[-1] == "activate_and_verify"
    assert events[-1]["step"]["phase"] == "service_activate"
    assert events[-1]["step"]["state"] == "succeeded"
    assert {field["name"]: field["value"] for field in events[-1]["step"]["fields"]} == {
        "status": "ok",
        "space_kind": "team",
        "space_name": "Systems Lab",
    }


def test_private_source_stops_before_checkout_with_complete_read_only_grant_steps() -> None:
    machine = FakeInstallMachine(
        access=_source_access(private=True, grant_needed=True),
        service_state=ServiceInstallState(
            data_state="fresh",
            service_state="stopped_disabled",
        ),
    )

    exit_code, output = _run_install(machine)
    paused = _events(output)[-1]["step"]

    assert exit_code == SERVER_CLI_EXIT_OPERATOR_ACTION
    assert machine.calls == [
        "validate_host",
        "converge_account_and_layout",
        "prepare_source_access",
    ]
    assert paused["phase"] == "source_grant"
    assert paused["target"]["destination_url"] == REPOSITORY.deploy_keys_url
    assert "leave Allow write access unchecked" in paused["actions"][0]["instruction"]
    assert paused["actions"][1]["argv"] == [
        "sudo",
        "-u",
        "rcp",
        "-H",
        "ssh",
        "-F",
        "/dev/null",
        "-i",
        "/home/rcp/rcp-server/credentials/source_ed25519",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=ask",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "UserKnownHostsFile=/home/rcp/.ssh/known_hosts",
        "-T",
        "git@github.com",
    ]
    assert paused["resume_argv"] == [
        "sudo",
        str(BOOTSTRAP_EXECUTABLE),
        "server",
        "install",
        "--team-name",
        "Systems Lab",
    ]
    details = {field["name"]: field["value"] for field in paused["fields"]}
    assert details["deploy_key_label"] == f"rcp-source:{INSTALLATION_ID}"
    assert details["deploy_public_key"].startswith("ssh-ed25519 ")


def test_known_refusal_is_exact_and_stops_before_later_steps() -> None:
    machine = FakeInstallMachine(
        access=_source_access(private=False),
        service_state=ServiceInstallState(data_state="initialized", service_state="active"),
        fail_at="build_release",
    )

    exit_code, output = _run_install(machine)
    events = _events(output)

    assert exit_code == SERVER_CLI_EXIT_FAILED
    assert machine.calls[-1] == "build_release"
    assert events[-1]["step"]["phase"] == "release_build"
    assert events[-1]["step"]["state"] == "failed"
    assert events[-1]["step"]["message"] == (
        "Focused refusal in build_release; correct it and rerun install."
    )


def test_preparation_is_side_effect_free_and_one_plan_serves_both_renderers() -> None:
    machine = FakeInstallMachine(
        access=_source_access(private=False),
        service_state=ServiceInstallState(
            data_state="fresh",
            service_state="stopped_disabled",
        ),
    )
    request = server_install.ServerCommandRequest(
        command="server install",
        team_name="Systems Lab",
    )

    prepared = prepare_install_command(
        request,
        ROOT_IDENTITY,
        machine=machine,
        repository=REPOSITORY,
        resume_executable=BOOTSTRAP_EXECUTABLE,
    )

    assert machine.calls == []
    assert [step.phase for step in prepared.plan.steps] == [
        "host_preflight",
        "account_layout",
        "source_access_prepare",
        "source_grant",
        "source_checkout",
        "release_build",
        "service_install",
        "team_space_init",
        "service_activate",
    ]
    interactive_machine = FakeInstallMachine(
        access=_source_access(private=True, grant_needed=True),
        service_state=ServiceInstallState(
            data_state="fresh",
            service_state="stopped_disabled",
        ),
    )
    _, interactive = _run_install(interactive_machine, machine_readable=False)
    assert "Performed by: human operator" in interactive
    assert "Allow write access unchecked" in interactive
    assert "Success: rcp can read origin/main" in interactive
    assert (
        "sudo /srv/bootstrap/.venv/bin/rcp server install --team-name 'Systems Lab'" in interactive
    )


@pytest.mark.parametrize(
    "origin",
    [
        "https://github.com/openai/rcp",
        "https://github.com/openai/rcp.git",
        "git@github.com:openai/rcp",
        "git@github.com:openai/rcp.git",
    ],
)
def test_github_origin_normalizes_to_credential_free_public_and_private_forms(origin) -> None:
    assert normalize_github_repository(origin) == REPOSITORY


@pytest.mark.parametrize(
    "origin",
    [
        "https://token@github.com/openai/rcp.git",
        "https://github.com/openai/rcp.git?token=value",
        "ssh://git@github.com/openai/rcp.git",
        "/srv/local/rcp",
        "https://gitlab.com/openai/rcp.git",
    ],
)
def test_github_origin_rejects_credentials_queries_other_hosts_and_local_paths(origin) -> None:
    with pytest.raises(ValueError, match="GitHub|github.com"):
        normalize_github_repository(origin)


def test_bootstrap_discovery_reads_origin_without_adopting_checkout(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    subprocess.run(
        ("git", "-C", str(tmp_path), "remote", "add", "origin", REPOSITORY.ssh_origin),
        check=True,
    )

    assert discover_bootstrap_repository(tmp_path) == REPOSITORY
    assert not (tmp_path / ".venv").exists()


def test_service_account_commands_clear_invoking_credentials_and_use_fixed_home(
    monkeypatch,
) -> None:
    captured: list[tuple[tuple[str, ...], dict[str, object]]] = []
    account = SimpleNamespace(pw_name="rcp", pw_dir="/home/rcp")

    def fake_run(argv, **kwargs):
        captured.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setenv("SSH_AUTH_SOCK", "/operator/agent.sock")
    monkeypatch.setenv("GH_TOKEN", "operator-secret")
    monkeypatch.setattr(server_install, "_run_process", fake_run)

    server_install._run_as_account(account, ("git", "--version"), timeout=1)
    explicit_cwd = Path("/srv/rcp/source")
    server_install._run_as_account(
        account,
        ("git", "status"),
        cwd=explicit_cwd,
        timeout=1,
    )

    argv, default_kwargs = captured[0]
    assert argv[:7] == (
        "runuser",
        "--user",
        "rcp",
        "--",
        "env",
        "-i",
        "HOME=/home/rcp",
    )
    assert "GIT_TERMINAL_PROMPT=0" in argv
    assert all("SSH_AUTH_SOCK" not in token and "GH_TOKEN" not in token for token in argv)
    assert default_kwargs["cwd"] == Path("/home/rcp")
    assert captured[1][1]["cwd"] == explicit_cwd


@pytest.mark.parametrize(
    ("returncode", "diagnostic", "expected"),
    [
        (0, "", "ready"),
        (128, "fatal: could not read Username for 'https://github.com'", "grant_needed"),
        (128, "ssh: Could not resolve host github.com", "unavailable"),
        (128, "Host key verification failed", "grant_needed"),
    ],
)
def test_source_probe_separates_grant_work_from_network_failure(
    monkeypatch,
    returncode,
    diagnostic,
    expected,
) -> None:
    machine = server_install.LinuxInstallMachine()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(("git",), returncode, "", diagnostic)

    monkeypatch.setattr(machine, "_run_as_service", fake_run)

    assert machine._probe_source(REPOSITORY.https_origin, source=None) == expected


def test_source_probe_refuses_missing_main_and_unrecognized_failure(monkeypatch) -> None:
    machine = server_install.LinuxInstallMachine()

    monkeypatch.setattr(
        machine,
        "_run_as_service",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(("git",), 2, "", ""),
    )
    with pytest.raises(InstallRefused, match="no readable main"):
        machine._probe_source(REPOSITORY.https_origin, source=None)

    monkeypatch.setattr(
        machine,
        "_run_as_service",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(("git",), 7, "", "odd failure"),
    )
    with pytest.raises(InstallRefused, match="without a recognized"):
        machine._probe_source(REPOSITORY.https_origin, source=None)


def test_private_source_environment_uses_only_fixed_key_and_strict_host_checking() -> None:
    machine = server_install.LinuxInstallMachine()
    source = ServerSourceConfig(
        origin=REPOSITORY.ssh_origin,
        authentication="deploy_key",
        public_key_fingerprint="SHA256:" + ("A" * 43),
    )

    environment = machine._source_environment(source)

    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_KEY_0"] == "credential.helper"
    assert environment["GIT_CONFIG_VALUE_0"] == ""
    assert environment["GIT_ASKPASS"] == "/bin/false"
    assert environment["SSH_ASKPASS"] == "/bin/false"
    assert environment["GIT_SSH_COMMAND"] == (
        "ssh -F /dev/null -i /home/rcp/rcp-server/credentials/source_ed25519 "
        "-o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes "
        "-o GlobalKnownHostsFile=/dev/null "
        "-o UserKnownHostsFile=/home/rcp/.ssh/known_hosts"
    )
    assert "SSH_AUTH_SOCK" not in environment


def test_existing_source_key_must_match_recorded_public_key(monkeypatch, tmp_path: Path) -> None:
    layout = _temporary_layout(tmp_path)
    layout.credentials_root.mkdir(parents=True)
    private = layout.credentials_root / "source_ed25519"
    public = layout.credentials_root / "source_ed25519.pub"
    private.write_text("private", encoding="utf-8")
    private.chmod(0o600)
    public.write_text(
        "ssh-ed25519 cHVibGljLWtleQ== label\n",
        encoding="utf-8",
    )
    public.chmod(0o644)
    _public_key, fingerprint = server_install._read_public_key(public)
    config = create_installed_server_config(
        source=ServerSourceConfig(
            origin=REPOSITORY.ssh_origin,
            authentication="deploy_key",
            public_key_fingerprint=fingerprint,
        ),
        installation_id=INSTALLATION_ID,
    )
    machine = server_install.LinuxInstallMachine(layout)
    machine._service_uid = os.getuid()
    machine._service_gid = os.getgid()
    monkeypatch.setattr(
        machine,
        "_run_as_service",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ("ssh-keygen",),
            0,
            "ssh-ed25519 ZGlmZmVyZW50\n",
            "",
        ),
    )

    with pytest.raises(InstallRefused, match="not one key pair"):
        machine._validate_source_key_pair(config, private, public)


def test_data_state_refuses_unknown_content_without_opening_sqlite(tmp_path: Path) -> None:
    layout = _temporary_layout(tmp_path)
    layout.data_dir.mkdir(parents=True)
    machine = server_install.LinuxInstallMachine(layout)
    machine._service_uid = os.getuid()
    machine._service_gid = os.getgid()

    assert machine._data_state() == "fresh"
    unknown = layout.data_dir / "unknown.bin"
    unknown.write_bytes(b"unknown")
    with pytest.raises(InstallRefused, match="files but no initialized"):
        machine._data_state()
    unknown.unlink()
    database = layout.data_dir / "rcp.sqlite3"
    database.write_bytes(b"not opened by install")
    database.chmod(0o600)
    assert machine._data_state() == "initialized"


def test_wrapper_is_fixed_to_configured_data_and_current_release() -> None:
    wrapper = server_install._wrapper_text(server_install.DEFAULT_SERVER_LAYOUT)

    assert "PYTHONDONTWRITEBYTECODE=1" in wrapper
    assert "RCP_DATA_DIR=/home/rcp/rcp-server/data" in wrapper
    assert 'exec /etc/rcp/current/.venv/bin/rcp "$@"' in wrapper
    assert "server.toml" not in wrapper


def test_supported_host_preflight_checks_exact_versions_without_installing_tools(
    monkeypatch,
) -> None:
    original_is_dir = Path.is_dir

    monkeypatch.setattr(
        server_install,
        "_read_os_release",
        lambda _path: {"ID": "ubuntu", "VERSION_ID": "24.04"},
    )
    monkeypatch.setattr(server_install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: True if path == Path("/run/systemd/system") else original_is_dir(path),
    )
    monkeypatch.setattr(server_install.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_require(argv, _error, **_kwargs):
        stdout = ""
        if argv[0] == "node":
            stdout = "v24.4.0\n"
        elif argv[0] == "age":
            stdout = "1.2.1\n"
        elif argv[:3] == ("systemctl", "show", "--property=Version"):
            stdout = "249\n"
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(server_install, "_require_command", fake_require)

    assert server_install.LinuxInstallMachine().validate_host() == HostFacts(ubuntu_release="24.04")


def test_host_preflight_refuses_when_systemd_manager_is_not_reachable(monkeypatch) -> None:
    original_is_dir = Path.is_dir
    monkeypatch.setattr(
        server_install,
        "_read_os_release",
        lambda _path: {"ID": "ubuntu", "VERSION_ID": "22.04"},
    )
    monkeypatch.setattr(server_install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: True if path == Path("/run/systemd/system") else original_is_dir(path),
    )
    monkeypatch.setattr(server_install.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_require(argv, error, **_kwargs):
        if argv[:3] == ("systemctl", "show", "--property=Version"):
            raise InstallRefused(error)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(server_install, "_require_command", fake_require)

    with pytest.raises(InstallRefused, match="PID 1"):
        server_install.LinuxInstallMachine().validate_host()


@pytest.mark.parametrize(
    ("release", "architecture", "message"),
    [
        ("20.04", "x86_64", "Ubuntu 22.04 or 24.04"),
        ("24.04", "aarch64", "x86-64"),
    ],
)
def test_host_preflight_refuses_unsupported_release_or_architecture(
    monkeypatch,
    release,
    architecture,
    message,
) -> None:
    monkeypatch.setattr(
        server_install,
        "_read_os_release",
        lambda _path: {"ID": "ubuntu", "VERSION_ID": release},
    )
    monkeypatch.setattr(server_install.platform, "machine", lambda: architecture)

    with pytest.raises(InstallRefused, match=message):
        server_install.LinuxInstallMachine().validate_host()


def test_account_layout_convergence_uses_only_fixed_owners_and_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layout = _temporary_layout(tmp_path)
    machine = server_install.LinuxInstallMachine(layout)
    account = SimpleNamespace(pw_uid=701, pw_gid=702)
    directories: list[tuple[Path, int, int, int]] = []
    tooling_checks: list[bool] = []
    monkeypatch.setattr(machine, "_converge_account", lambda: account)
    monkeypatch.setattr(
        server_install,
        "_converge_directory",
        lambda path, *, uid, gid, mode: directories.append((path, uid, gid, mode)),
    )
    monkeypatch.setattr(machine, "_validate_service_tooling", lambda: tooling_checks.append(True))

    machine.converge_account_and_layout()

    assert (layout.service_home, 701, 702, 0o700) in directories
    assert (layout.data_dir, 701, 702, 0o700) in directories
    assert (layout.credentials_root, 701, 702, 0o700) in directories
    assert (layout.config_path.parent, 0, 702, 0o750) in directories
    assert tooling_checks == [True]


def test_existing_service_account_must_be_unprivileged_and_have_no_sudo_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layout = _temporary_layout(tmp_path)
    machine = server_install.LinuxInstallMachine(layout)

    def account(*, uid: int = 701, gid: int = 702):
        return SimpleNamespace(
            pw_name="rcp",
            pw_uid=uid,
            pw_gid=gid,
            pw_dir=str(layout.service_home),
            pw_shell="/bin/bash",
        )

    current = account()
    monkeypatch.setattr(server_install.pwd, "getpwnam", lambda _name: current)
    monkeypatch.setattr(
        server_install.grp,
        "getgrgid",
        lambda gid: SimpleNamespace(gr_name="rcp", gr_gid=gid),
    )
    monkeypatch.setattr(server_install.grp, "getgrall", lambda: [])
    monkeypatch.setattr(
        server_install,
        "_require_command",
        lambda argv, _error, **_kwargs: subprocess.CompletedProcess(
            argv,
            0,
            "rcp:*NP*:20000:0:99999:7:::\n",
            "",
        ),
    )

    current = account(uid=0)
    with pytest.raises(InstallRefused, match="root user or group"):
        machine._converge_account()

    current = account(gid=0)
    with pytest.raises(InstallRefused, match="root user or group"):
        machine._converge_account()

    current = account()
    sudo_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def privileged_policy(argv, **kwargs):
        sudo_calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            "User rcp may run /bin/bash on lab.\n",
            "",
        )

    monkeypatch.setattr(server_install, "_run_process", privileged_policy)
    with pytest.raises(InstallRefused, match="has sudo authority"):
        machine._converge_account()
    assert sudo_calls == [
        (
            ("sudo", "-n", "-U", "rcp", "-l"),
            {
                "environment": {"LANG": "C", "LC_ALL": "C"},
                "timeout": server_install.SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            },
        )
    ]

    monkeypatch.setattr(
        server_install,
        "_run_process",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 1, "", "sudo policy error"),
    )
    with pytest.raises(InstallRefused, match="could not prove"):
        machine._converge_account()

    monkeypatch.setattr(
        server_install,
        "_run_process",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            0,
            "",
            "User rcp is not allowed to run sudo on lab.\n",
        ),
    )
    assert machine._converge_account() == current

    monkeypatch.setattr(
        server_install,
        "_run_process",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            1,
            "User rcp is not allowed to run sudo on lab.\n",
            "",
        ),
    )
    assert machine._converge_account() == current


def test_new_service_account_uses_stateful_timeout_and_reports_expiry(
    monkeypatch,
) -> None:
    calls: list[tuple[tuple[str, ...], float]] = []

    def missing_account(_name: str):
        raise KeyError

    def timed_out_useradd(
        argv: tuple[str, ...],
        *,
        timeout: float,
        **_kwargs,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((argv, timeout))
        return subprocess.CompletedProcess(argv, 126, "", "command timed out")

    monkeypatch.setattr(server_install.pwd, "getpwnam", missing_account)
    monkeypatch.setattr(server_install, "_run_process", timed_out_useradd)

    with pytest.raises(InstallRefused, match="did not finish within five minutes"):
        server_install.LinuxInstallMachine()._converge_account()

    assert calls == [
        (
            (
                "useradd",
                "--create-home",
                "--home-dir",
                "/home/rcp",
                "--shell",
                "/bin/bash",
                "--user-group",
                "--password",
                "*NP*",
                "rcp",
            ),
            server_install.SERVER_INSTALL_ACCOUNT_TIMEOUT_SECONDS,
        )
    ]


def test_root_process_drops_inherited_sudo_identity(monkeypatch) -> None:
    captured_environment: dict[str, str] = {}

    def fake_run(argv, **kwargs):
        captured_environment.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setenv("SUDO_COMMAND", "/usr/bin/env rcp server install")
    monkeypatch.setenv("SUDO_GID", "123")
    monkeypatch.setenv("SUDO_UID", "1001")
    monkeypatch.setenv("SUDO_USER", "runner")
    monkeypatch.setenv("RCP_ENVIRONMENT_SENTINEL", "preserved")
    monkeypatch.setattr(server_install.subprocess, "run", fake_run)

    server_install._run_process(("true",), timeout=1)

    assert captured_environment["RCP_ENVIRONMENT_SENTINEL"] == "preserved"
    assert (
        not {
            "SUDO_COMMAND",
            "SUDO_GID",
            "SUDO_UID",
            "SUDO_USER",
        }
        & captured_environment.keys()
    )


def test_service_tooling_installs_and_rechecks_managed_python_for_fresh_account(
    monkeypatch,
) -> None:
    account = SimpleNamespace(pw_name="rcp", pw_dir="/home/rcp")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    find_count = 0

    def fake_run(_account, argv, **kwargs):
        nonlocal find_count
        calls.append((argv, kwargs))
        if argv[:3] == ("uv", "python", "find"):
            find_count += 1
            if find_count == 1:
                return subprocess.CompletedProcess(argv, 2, "", "not installed")
            return subprocess.CompletedProcess(
                argv,
                0,
                "/home/rcp/.local/share/uv/python/cpython-3.12/bin/python3.12\n",
                "",
            )
        if argv[0].endswith("python3.12"):
            return subprocess.CompletedProcess(argv, 0, "Python 3.12.10\n", "")
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")

    monkeypatch.setattr(server_install.pwd, "getpwnam", lambda _name: account)
    monkeypatch.setattr(server_install, "_run_as_account", fake_run)

    server_install.LinuxInstallMachine()._validate_service_tooling()

    assert (
        "uv",
        "python",
        "find",
        "--managed-python",
        "--no-python-downloads",
        "3.12",
    ) in [argv for argv, _kwargs in calls]
    install_call = next(
        (argv, kwargs) for argv, kwargs in calls if argv[:3] == ("uv", "python", "install")
    )
    assert install_call[0] == (
        "uv",
        "python",
        "install",
        "--managed-python",
        "--no-progress",
        "3.12",
    )
    assert install_call[1]["capture_output"] is False


@pytest.mark.parametrize("private", [False, True])
def test_new_source_access_records_public_or_dedicated_key_mode(
    monkeypatch,
    tmp_path: Path,
    private: bool,
) -> None:
    layout = _temporary_layout(tmp_path)
    layout.credentials_root.mkdir(parents=True)
    machine = server_install.LinuxInstallMachine(layout)
    machine._service_uid = os.getuid()
    machine._service_gid = os.getgid()
    written = []
    created_keys = []
    monkeypatch.setattr(
        machine,
        "_probe_source",
        lambda *_args, **_kwargs: "grant_needed" if private else "ready",
    )
    monkeypatch.setattr(
        machine,
        "_create_source_key_pair",
        lambda private_path, public_path, *, label: created_keys.append(
            (private_path, public_path, label)
        ),
    )
    monkeypatch.setattr(
        server_install,
        "_read_public_key",
        lambda _path: (
            "ssh-ed25519 cHVibGljLWtleQ== source",
            "SHA256:" + ("A" * 43),
        ),
    )
    monkeypatch.setattr(
        server_install,
        "write_installed_server_config",
        lambda config, path: written.append((config, path)),
    )

    access = machine.prepare_source_access(REPOSITORY)

    assert len(written) == 1
    assert written[0][1] == layout.config_path
    if private:
        assert access.grant_needed is True
        assert access.config.source.authentication == "deploy_key"
        assert access.config.source.origin == REPOSITORY.ssh_origin
        assert len(created_keys) == 1
        assert created_keys[0][2] == f"rcp-source:{access.config.installation_id}"
    else:
        assert access.grant_needed is False
        assert access.config.source.authentication == "public"
        assert access.config.source.origin == REPOSITORY.https_origin
        assert created_keys == []


def test_managed_checkout_fetches_clean_main_but_refuses_install_owned_version_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layout = _temporary_layout(tmp_path)
    layout.source_checkout.mkdir(parents=True)
    (layout.source_checkout / ".git").mkdir()
    layout.restore_operations_root.mkdir(parents=True)
    layout.restore_operations_root.chmod(0o700)
    layout.update_checkpoints_root.mkdir(parents=True, mode=0o700)
    machine = server_install.LinuxInstallMachine(layout)
    machine._service_uid = os.getuid()
    machine._service_gid = os.getgid()
    access = _source_access(private=False)
    git_commands: list[tuple[str, ...]] = []

    def git_text(_root, argv, **_kwargs):
        if argv == ("remote", "get-url", "origin"):
            return REPOSITORY.https_origin
        if argv[0] == "status":
            return ""
        if argv == ("rev-parse", "origin/main"):
            return COMMIT
        if argv == ("rev-parse", "HEAD"):
            return COMMIT
        raise AssertionError(argv)

    monkeypatch.setattr(machine, "_git_text", git_text)
    monkeypatch.setattr(
        machine,
        "_run_git",
        lambda _root, argv, **_kwargs: git_commands.append(argv),
    )
    monkeypatch.setattr(machine, "_current_release_commit", lambda: None)

    assert machine.converge_source_checkout(access) == ManagedCheckout(
        commit=COMMIT,
        is_current_release=False,
    )
    assert git_commands == [
        ("fetch", "--prune", "origin", "main"),
        ("checkout", "--force", "-B", "main", "origin/main"),
    ]

    monkeypatch.setattr(machine, "_current_release_commit", lambda: COMMIT)

    def advanced_git_text(_root, argv, **_kwargs):
        if argv == ("remote", "get-url", "origin"):
            return REPOSITORY.https_origin
        if argv[0] == "status":
            return ""
        if argv == ("rev-parse", "origin/main"):
            return "b" * 40
        if argv == ("rev-parse", "HEAD"):
            return COMMIT
        raise AssertionError(argv)

    monkeypatch.setattr(machine, "_git_text", advanced_git_text)
    with pytest.raises(InstallRefused, match="Version changes belong"):
        machine.converge_source_checkout(access)


def test_install_routes_an_unfinished_update_before_touching_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layout = _temporary_layout(tmp_path)
    layout.restore_operations_root.mkdir(parents=True)
    layout.restore_operations_root.chmod(0o700)
    layout.update_checkpoints_root.mkdir(parents=True, mode=0o700)
    built = layout.update_checkpoints_root / f"built-candidate-{'b' * 40}.json"
    preflight = layout.update_checkpoints_root / "preflight.json"
    for path in (built, preflight):
        path.write_text("receipt\n", encoding="utf-8")
        path.chmod(0o600)
    operation = new_update_operation(
        operation_id=str(uuid.uuid4()),
        installation_id=INSTALLATION_ID,
        space_id=str(uuid.uuid4()),
        base_commit=COMMIT,
        candidate_commit="b" * 40,
        base_instance_id=str(uuid.uuid4()),
        base_process_pid=421,
        built_receipt_path=built,
        built_receipt_sha256="a" * 64,
        preflight_receipt_path=preflight,
        preflight_receipt_sha256="b" * 64,
        update_root=layout.update_checkpoints_root,
    )
    publish_update_operation(
        operation,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    machine = server_install.LinuxInstallMachine(layout)
    machine._service_uid = os.geteuid()
    machine._service_gid = os.getegid()
    stopped: list[str] = []
    monkeypatch.setattr(
        server_install.InstalledSystemServiceController,
        "stop",
        lambda _self: stopped.append("stopped"),
    )

    with pytest.raises(InstallRefused, match="sudo rcp server update"):
        machine.converge_source_checkout(_source_access(private=False))

    assert stopped == ["stopped"]


def test_release_build_runs_exact_managed_commands_as_service_account(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layout = _temporary_layout(tmp_path)
    layout.source_checkout.mkdir(parents=True)
    layout.releases_root.mkdir(parents=True)
    machine = server_install.LinuxInstallMachine(layout)
    machine._service_uid = os.getuid()
    machine._service_gid = os.getgid()
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[:4] == ("git", "-C", str(layout.source_checkout), "worktree"):
            layout.release_dir(COMMIT).mkdir()
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(machine, "_run_as_service", fake_run)
    monkeypatch.setattr(machine, "_validate_release_artifacts", lambda _release: None)

    release = machine.build_release(ManagedCheckout(commit=COMMIT, is_current_release=False))

    assert release == layout.release_dir(COMMIT)
    assert [call[0] for call in calls] == [
        (
            "git",
            "-C",
            str(layout.source_checkout),
            "worktree",
            "add",
            "--detach",
            str(release),
            COMMIT,
        ),
        ("npm", "--prefix", "web", "ci"),
        ("npm", "--prefix", "web", "run", "build"),
        ("uv", "sync", "--frozen"),
    ]
    assert calls[-1][1]["environment"] == {
        "UV_MANAGED_PYTHON": "1",
        "UV_PYTHON": "3.12",
    }
    assert all(call[1].get("cwd") == release for call in calls[1:])


def test_service_install_keeps_fresh_data_stopped_and_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layout = _temporary_layout(tmp_path)
    release = layout.release_dir(COMMIT)
    layout.restore_operations_root.mkdir(parents=True)
    layout.restore_operations_root.chmod(0o700)
    layout.update_checkpoints_root.mkdir(parents=True)
    machine = server_install.LinuxInstallMachine(layout)
    machine._service_uid = os.getuid()
    machine._service_gid = os.getgid()
    files = []
    current = []
    commands = []
    monkeypatch.setattr(
        backup_owner,
        "backup_configuration_lock",
        lambda _layout: nullcontext(),
    )
    monkeypatch.setattr(
        backup_owner,
        "recover_pending_backup_configuration",
        lambda _layout: None,
    )
    monkeypatch.setattr(
        server_install,
        "_install_root_file",
        lambda path, content, *, mode, replace_existing=False: files.append(
            (path, content, mode, replace_existing)
        ),
    )
    monkeypatch.setattr(
        server_install,
        "load_installed_server_config",
        lambda _path: create_installed_server_config(
            source=ServerSourceConfig(
                origin="https://github.com/example/research-control-panel.git",
                authentication="public",
            ),
            installation_id=INSTALLATION_ID,
        ),
    )
    monkeypatch.setattr(
        server_install,
        "_converge_current_release",
        lambda actual_layout, actual_release: current.append((actual_layout, actual_release)),
    )
    monkeypatch.setattr(
        server_install,
        "_require_command",
        lambda argv, _error, **_kwargs: subprocess.CompletedProcess(
            argv,
            0,
            (
                "inactive\n"
                if "--property=ActiveState" in argv
                else "disabled\n"
                if "--property=UnitFileState" in argv
                else "not-found\n"
                if "--property=LoadState" in argv
                else ""
            ),
            "",
        ),
    )
    original_require = server_install._require_command

    def recording_require(argv, error, **kwargs):
        commands.append(argv)
        return original_require(argv, error, **kwargs)

    monkeypatch.setattr(server_install, "_require_command", recording_require)
    monkeypatch.setattr(machine, "_data_state", lambda: "fresh")

    result = machine.install_service(
        ManagedCheckout(commit=COMMIT, is_current_release=False),
        release,
    )

    assert result == ServiceInstallState(
        data_state="fresh",
        service_state="stopped_disabled",
    )
    assert [item[0] for item in files] == [
        layout.cli_wrapper,
        layout.systemd_unit,
        layout.systemd_unit.parent / "rcp-backup.service",
        layout.systemd_unit.parent / "rcp-backup.timer",
    ]
    assert files[-1][3] is True
    assert current == [(layout, release)]
    assert ("systemctl", "disable", "--now", "rcp.service") in commands
    assert ("systemctl", "disable", "--now", "rcp-backup.timer") in commands


def test_backup_timer_is_fenced_before_loaded_unit_changes(monkeypatch) -> None:
    fences: list[str] = []
    monkeypatch.setattr(
        server_install,
        "_fence_service_stopped_disabled",
        lambda unit: fences.append(unit),
    )
    monkeypatch.setattr(
        server_install,
        "_read_systemd_property",
        lambda _unit, _property: "not-found",
    )

    server_install.fence_backup_timer_before_unit_change()
    assert fences == []

    monkeypatch.setattr(
        server_install,
        "_read_systemd_property",
        lambda _unit, _property: "loaded",
    )
    server_install.fence_backup_timer_before_unit_change()
    assert fences == ["rcp-backup.timer"]


def test_activation_reads_team_health_and_stops_a_wrong_space(monkeypatch) -> None:
    machine = server_install.LinuxInstallMachine()
    monkeypatch.setattr(machine, "_require_no_unfinished_update", lambda: None)
    monkeypatch.setattr(machine, "_require_no_unfinished_restore", lambda: None)
    commands = []
    fenced = False

    def fake_require(argv, _error, **_kwargs):
        nonlocal fenced
        commands.append(argv)
        if argv[:3] == ("systemctl", "disable", "--now"):
            fenced = True
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "--property=UnitFileState" in argv:
            output = "disabled\n" if fenced else "enabled\n"
        elif "--property=ActiveState" in argv:
            output = "inactive\n" if fenced else "active\n"
        else:
            output = ""
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(server_install, "_require_command", fake_require)
    monkeypatch.setattr(
        server_install,
        "_read_team_health",
        lambda: {"status": "ok", "space_kind": "team", "space_name": "Systems Lab"},
    )
    assert machine.activate_and_verify() == ServiceHealth(
        status="ok",
        space_kind="team",
        space_name="Systems Lab",
    )

    monkeypatch.setattr(
        server_install,
        "_read_team_health",
        lambda: {"status": "ok", "space_kind": "personal", "space_name": "Wrong"},
    )
    with pytest.raises(InstallRefused, match="stopped and disabled"):
        machine.activate_and_verify()
    assert ("systemctl", "disable", "--now", "rcp.service") in commands


def test_service_fence_fails_closed_when_stop_or_readback_fails(monkeypatch) -> None:
    def failed_stop(argv, error, **_kwargs):
        if argv[:3] == ("systemctl", "disable", "--now"):
            raise InstallRefused(error)
        return subprocess.CompletedProcess(argv, 0, "inactive\n", "")

    monkeypatch.setattr(server_install, "_require_command", failed_stop)
    with pytest.raises(InstallRefused, match="could not stop and disable"):
        server_install._fence_service_stopped_disabled("rcp.service")

    def wrong_readback(argv, _error, **_kwargs):
        output = "active\n" if "--property=ActiveState" in argv else "disabled\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(server_install, "_require_command", wrong_readback)
    with pytest.raises(InstallRefused, match="could not prove"):
        server_install._fence_service_stopped_disabled("rcp.service")


def test_health_readback_uses_direct_loopback_http_and_rejects_redirects(monkeypatch) -> None:
    connections = []
    requests = []

    class FakeResponse:
        status = 302

        def read(self, _size):
            return b'{"status":"ok","space_kind":"team","space_name":"Wrong"}'

    class FakeConnection:
        def __init__(self, host, port, *, timeout):
            connections.append((host, port, timeout))

        def request(self, method, path, *, headers):
            requests.append((method, path, headers))

        def getresponse(self):
            return FakeResponse()

        def close(self):
            return None

    clock = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(server_install.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(server_install.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(server_install.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(server_install, "SERVER_INSTALL_HEALTH_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")

    assert server_install._read_team_health() is None
    assert connections == [
        ("127.0.0.1", 8421, server_install.SERVER_HEALTH_REQUEST_TIMEOUT_SECONDS)
    ]
    assert requests == [
        (
            "GET",
            "/api/health",
            {"Accept": "application/json", "Host": "127.0.0.1:8421"},
        )
    ]


def _temporary_layout(tmp_path: Path) -> ServerLayout:
    home = tmp_path / "home" / "rcp"
    root = home / "rcp-server"
    return ServerLayout(
        service_account="rcp",
        service_home=home,
        server_root=root,
        source_checkout=root / "source",
        releases_root=root / "releases",
        data_dir=root / "data",
        projects_root=root / "projects",
        credentials_root=root / "credentials",
        update_checkpoints_root=root / "update-checkpoints",
        restore_operations_root=root / "restore-operations",
        codex_state_root=home / ".codex",
        claude_state_root=home / ".claude",
        ssh_state_root=home / ".ssh",
        config_path=tmp_path / "etc" / "rcp" / "server.toml",
        current_release=tmp_path / "etc" / "rcp" / "current",
        runtime_dir=tmp_path / "run" / "rcp",
        control_socket=tmp_path / "run" / "rcp" / "control.sock",
        cli_wrapper=tmp_path / "usr" / "local" / "bin" / "rcp",
        systemd_unit=tmp_path / "etc" / "systemd" / "system" / "rcp.service",
        service_unit_name="rcp.service",
    )
