from __future__ import annotations

import json
import os
import pwd
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from rcp.server_ops import git_credentials as server_git_credentials
from rcp.server_ops.git_credentials import (
    DeployKeyMaterial,
    GitCredentialManager,
    GitCredentialRefused,
    GitWriteProbe,
    _parse_remote_refs,
    _run_process,
    cleanup_ref_operator_step,
    deploy_key_operator_step,
    empty_repository_operator_step,
)
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.layout import ServerLayout, remote_project_deploy_key_path
from rcp.storage import ProjectProvisioningMachineIntent

SPACE_ID = "7eb4ea9d-cccf-42fd-abfe-09f71f4b8cd2"
PROJECT_ID = "2ad064a6-f015-4703-a223-1d64cde75cc8"
REQUEST_ID = "a29ddba0-a0a7-46be-ab7a-7a6d77644ea5"
ALIAS = "paper"
COMMIT = "a" * 40
FINGERPRINT = "SHA256:" + "A" * 43
PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA "
    f"rcp:{SPACE_ID}:{PROJECT_ID}:{ALIAS}"
)
REPOSITORY = GitHubRepositoryRef(identity="zhi0467/rcp")
HELPER = Path(__file__).parents[1] / "src" / "rcp" / "server_ops" / "remote_git_credentials.py"


def _layout(tmp_path: Path) -> ServerLayout:
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


def _local_machine(layout: ServerLayout) -> ProjectProvisioningMachineIntent:
    return ProjectProvisioningMachineIntent.model_construct(
        alias="server",
        location="local",
        host="",
        os_account="rcp",
        central_root=str(layout.projects_root),
    )


def _ssh_machine() -> ProjectProvisioningMachineIntent:
    return ProjectProvisioningMachineIntent(
        alias="gpu",
        location="ssh",
        host="gpu-lab",
        os_account="alice",
        central_root="/srv/lab/projects",
    )


def _receipt(
    layout: ServerLayout,
    machine: ProjectProvisioningMachineIntent,
    *,
    home: str | None = None,
    created: bool = True,
) -> dict[str, object]:
    resolved_home = (
        str(layout.service_home) if machine.location == "local" else home or "/srv/alice"
    )
    if machine.location == "local":
        root = str(layout.credentials_root)
        private = str(layout.project_deploy_key_path(PROJECT_ID, ALIAS))
    else:
        private = str(remote_project_deploy_key_path(resolved_home, PROJECT_ID, ALIAS))
        root = str(Path(private).parents[3])
    return {
        "account": machine.os_account,
        "home": resolved_home,
        "credentials_root": root,
        "private_key_path": private,
        "label": f"rcp:{SPACE_ID}:{PROJECT_ID}:{ALIAS}",
        "public_key": PUBLIC_KEY,
        "public_key_fingerprint": FINGERPRINT,
        "created": created,
    }


def _material(layout: ServerLayout) -> DeployKeyMaterial:
    return DeployKeyMaterial(
        space_id=SPACE_ID,
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        repository=REPOSITORY,
        machine_alias="server",
        location="local",
        host="",
        os_account="rcp",
        central_root=str(layout.projects_root),
        account_home=str(layout.service_home),
        credentials_root=str(layout.credentials_root),
        private_key_path=str(layout.project_deploy_key_path(PROJECT_ID, ALIAS)),
        label=f"rcp:{SPACE_ID}:{PROJECT_ID}:{ALIAS}",
        public_key=PUBLIC_KEY,
        public_key_fingerprint=FINGERPRINT,
        created=True,
    )


def _helper(
    operation: str,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(HELPER), operation, *arguments),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _helper_prepare(root: Path) -> subprocess.CompletedProcess[str]:
    account = pwd.getpwuid(os.getuid()).pw_name
    return _helper(
        "prepare",
        account,
        "local",
        str(root),
        str(root.parent / "projects"),
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
    )


def test_shipped_helper_creates_reuses_and_removes_only_one_exact_key(
    tmp_path: Path,
) -> None:
    root = tmp_path / "credentials"
    root.mkdir(mode=0o700)
    account = pwd.getpwuid(os.getuid()).pw_name

    first = _helper_prepare(root)

    assert first.returncode == 0, first.stderr
    receipt = json.loads(first.stdout)
    private = Path(receipt["private_key_path"])
    public = Path(f"{private}.pub")
    private_text = private.read_text(encoding="utf-8")
    assert receipt["created"] is True
    assert receipt["label"] == f"rcp:{SPACE_ID}:{PROJECT_ID}:{ALIAS}"
    assert receipt["public_key"].endswith(receipt["label"])
    assert receipt["public_key_fingerprint"].startswith("SHA256:")
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert stat.S_IMODE(public.stat().st_mode) == 0o644
    assert "OPENSSH PRIVATE KEY" not in first.stdout
    assert private_text not in first.stdout

    second = _helper_prepare(root)
    assert second.returncode == 0, second.stderr
    repeated = json.loads(second.stdout)
    assert repeated == {**receipt, "created": False}

    inspected = _helper(
        "inspect",
        account,
        "local",
        str(root),
        str(root.parent / "projects"),
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout) == repeated

    project_directory = private.parent.parent
    sentinel = project_directory / "keep-me"
    sentinel.write_text("unrelated", encoding="utf-8")
    refused = _helper(
        "remove",
        account,
        "local",
        str(root),
        str(root.parent / "projects"),
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
        "SHA256:" + "B" * 43,
    )
    assert refused.returncode == 2
    assert private.exists() and public.exists() and sentinel.exists()

    removed = _helper(
        "remove",
        account,
        "local",
        str(root),
        str(root.parent / "projects"),
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
        str(receipt["public_key_fingerprint"]),
    )
    assert removed.returncode == 0, removed.stderr
    assert json.loads(removed.stdout) == {"removed": True}
    assert not private.exists() and not public.exists()
    assert sentinel.read_text(encoding="utf-8") == "unrelated"

    repeated_remove = _helper(
        "remove",
        account,
        "local",
        str(root),
        str(root.parent / "projects"),
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
        str(receipt["public_key_fingerprint"]),
    )
    assert repeated_remove.returncode == 0
    assert json.loads(repeated_remove.stdout) == {"removed": False}


def test_recovery_preflight_proves_fresh_key_path_before_generation(tmp_path: Path) -> None:
    root = tmp_path / "credentials"
    root.mkdir(mode=0o700)
    account = pwd.getpwuid(os.getuid()).pw_name
    arguments = (
        account,
        "local",
        str(root),
        str(root.parent / "projects"),
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
    )

    absent = _helper("recovery-preflight", *arguments)

    assert absent.returncode == 0, absent.stderr
    assert json.loads(absent.stdout)["absent"] is True

    created = _helper_prepare(root)
    assert created.returncode == 0, created.stderr
    present = _helper("recovery-preflight", *arguments)

    assert present.returncode == 0, present.stderr
    assert json.loads(present.stdout)["absent"] is False


def test_shipped_helper_refuses_unsafe_root_mode_and_incomplete_pair(tmp_path: Path) -> None:
    root = tmp_path / "credentials"
    root.mkdir(mode=0o700)
    root.chmod(0o755)

    unsafe = _helper_prepare(root)
    assert unsafe.returncode == 2
    assert "unsafe type, ownership, or mode" in unsafe.stderr
    assert list(root.iterdir()) == []

    root.chmod(0o700)
    created = _helper_prepare(root)
    assert created.returncode == 0, created.stderr
    private = Path(json.loads(created.stdout)["private_key_path"])
    public = Path(f"{private}.pub")
    public.unlink()

    incomplete = _helper_prepare(root)
    assert incomplete.returncode == 2
    assert "pair is incomplete" in incomplete.stderr
    assert private.exists()


def test_shipped_helper_refuses_a_checkout_inside_the_credential_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "credentials"
    root.mkdir(mode=0o700)
    account = pwd.getpwuid(os.getuid()).pw_name

    refused = _helper(
        "prepare",
        account,
        "local",
        str(root),
        str(root),
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
    )

    assert refused.returncode == 2
    assert "checkout and credential paths overlap" in refused.stderr
    assert list(root.iterdir()) == []


def test_shipped_helper_creates_and_removes_only_its_request_probe_directory() -> None:
    account = pwd.getpwuid(os.getuid()).pw_name
    prepared = _helper("probe-prepare", account, REQUEST_ID)
    assert prepared.returncode == 0, prepared.stderr
    path = Path(json.loads(prepared.stdout)["probe_directory"])
    try:
        assert path.parent == Path("/tmp")
        assert path.name.startswith(f"rcp-git-probe.{REQUEST_ID}.")
        assert stat.S_IMODE(path.stat().st_mode) == 0o700

        wrong_request = _helper("probe-cleanup", account, SPACE_ID, str(path))
        assert wrong_request.returncode == 2
        assert path.exists()

        cleaned = _helper("probe-cleanup", account, REQUEST_ID, str(path))
        assert cleaned.returncode == 0, cleaned.stderr
        assert json.loads(cleaned.stdout) == {"removed": True}
        assert not path.exists()
    finally:
        if path.exists():
            path.rmdir()


class QueueRunner:
    def __init__(self, *results: subprocess.CompletedProcess[str]) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, timeout))
        if not self.results:
            raise AssertionError(f"unexpected command: {argv!r}")
        result = self.results.pop(0)
        return subprocess.CompletedProcess(argv, result.returncode, result.stdout, result.stderr)


def _result(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess((), returncode, stdout, stderr)


def test_default_runner_stops_output_before_it_can_exceed_the_bound() -> None:
    result = _run_process(
        (
            sys.executable,
            "-c",
            "import os; os.write(1, b'x' * (512 * 1024))",
        ),
        timeout=5,
    )

    assert result.returncode == 126
    assert result.stdout == ""
    assert result.stderr == "output exceeded the bound"


def test_default_runner_returns_bounded_stdout_stderr_and_exit_status() -> None:
    result = _run_process(
        (
            sys.executable,
            "-c",
            "import sys; print('ready'); print('note', file=sys.stderr); raise SystemExit(7)",
        ),
        timeout=5,
    )

    assert result.returncode == 7
    assert result.stdout == "ready\n"
    assert result.stderr == "note\n"


def test_default_manager_runner_starts_from_the_service_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    layout.service_home.mkdir(parents=True)
    calls: list[tuple[tuple[str, ...], Path | None, float]] = []

    def fake_run(
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((argv, cwd, timeout))
        return _result(stdout=json.dumps(_receipt(layout, _local_machine(layout))))

    monkeypatch.setattr(server_git_credentials, "_run_process", fake_run)

    GitCredentialManager(layout).prepare_key(
        _local_machine(layout),
        REPOSITORY,
        space_id=SPACE_ID,
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
    )

    assert len(calls) == 1
    assert calls[0][1] == layout.service_home


def test_default_runner_keeps_the_timeout_after_output_pipes_close() -> None:
    result = _run_process(
        (
            sys.executable,
            "-c",
            "import os, time; os.close(1); os.close(2); time.sleep(10)",
        ),
        timeout=0.1,
    )

    assert result.returncode == 126
    assert result.stdout == ""
    assert result.stderr == "command timed out"


def test_manager_ships_one_helper_through_strict_local_and_ssh_account_boundaries(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    local = _local_machine(layout)
    local_runner = QueueRunner(_result(stdout=json.dumps(_receipt(layout, local))))
    local_manager = GitCredentialManager(layout, runner=local_runner)

    material = local_manager.prepare_key(
        local,
        REPOSITORY,
        space_id=SPACE_ID,
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
    )

    assert material.private_key_path == str(layout.project_deploy_key_path(PROJECT_ID, ALIAS))
    local_argv = local_runner.calls[0][0]
    assert local_argv[:4] == ("runuser", "--user", "rcp", "--")
    assert local_argv[12:14] == ("python3", "-c")
    assert local_argv[14] == HELPER.read_text(encoding="utf-8")
    assert local_argv[-8:] == (
        "prepare",
        "rcp",
        "local",
        str(layout.credentials_root),
        str(layout.projects_root),
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
    )

    remote = _ssh_machine()
    remote_runner = QueueRunner(
        _result(stdout=json.dumps(_receipt(layout, remote, home="/srv/accounts/alice")))
    )
    remote_manager = GitCredentialManager(layout, runner=remote_runner)
    remote_material = remote_manager.prepare_key(
        remote,
        REPOSITORY,
        space_id=SPACE_ID,
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
    )

    assert remote_material.private_key_path == (
        f"/srv/accounts/alice/.local/share/rcp/credentials/projects/{PROJECT_ID}/{ALIAS}/id_ed25519"
    )
    remote_argv = remote_runner.calls[0][0]
    assert remote_argv[:4] == ("runuser", "--user", "rcp", "--")
    assert "StrictHostKeyChecking=yes" in remote_argv
    assert remote_argv[-2] == "gpu-lab"
    remote_inner = shlex.split(remote_argv[-1])
    assert remote_inner[:2] == ["python3", "-c"]
    assert remote_inner[2] == HELPER.read_text(encoding="utf-8")
    assert remote_inner[-8:] == [
        "prepare",
        "alice",
        "ssh",
        "-",
        "/srv/lab/projects",
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
    ]

    default_remote = remote.model_copy(update={"central_root": None})
    default_runner = QueueRunner(
        _result(stdout=json.dumps(_receipt(layout, default_remote, home="/srv/accounts/alice")))
    )
    default_material = GitCredentialManager(layout, runner=default_runner).prepare_key(
        default_remote,
        REPOSITORY,
        space_id=SPACE_ID,
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
    )
    default_inner = shlex.split(default_runner.calls[0][0][-1])
    assert default_inner[-8:] == [
        "prepare",
        "alice",
        "ssh",
        "-",
        "-",
        SPACE_ID,
        PROJECT_ID,
        ALIAS,
    ]
    assert default_material.central_root == "/srv/accounts/alice/.local/share/rcp/projects"


def test_manager_rejects_a_local_machine_for_another_project_root(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    machine = ProjectProvisioningMachineIntent.model_construct(
        alias="server",
        location="local",
        host="",
        os_account="rcp",
        central_root=str(layout.projects_root / "other"),
    )
    runner = QueueRunner(_result(stdout="{}"))
    manager = GitCredentialManager(layout, runner=runner)

    with pytest.raises(GitCredentialRefused, match="account and project root"):
        manager.prepare_key(
            machine,
            REPOSITORY,
            space_id=SPACE_ID,
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
        )

    assert runner.calls == []


def test_manager_rejects_a_helper_receipt_for_another_home_or_path(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    machine = _local_machine(layout)
    wrong_home = _receipt(layout, machine)
    wrong_home["home"] = "/home/not-rcp"
    manager = GitCredentialManager(
        layout,
        runner=QueueRunner(_result(stdout=json.dumps(wrong_home))),
    )

    with pytest.raises(GitCredentialRefused, match="wrong local service home"):
        manager.prepare_key(
            machine,
            REPOSITORY,
            space_id=SPACE_ID,
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
        )


@dataclass(frozen=True)
class GitResult:
    expected: tuple[str, ...]
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class GitScript:
    def __init__(self, *results: GitResult) -> None:
        self.results = list(results)
        self.git_calls: list[tuple[str, ...]] = []
        self.outer_calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        self.outer_calls.append(argv)
        git_index = argv.index("git")
        git_argv = argv[git_index:]
        self.git_calls.append(git_argv)
        if not self.results:
            raise AssertionError(f"unexpected Git command: {git_argv!r}")
        result = self.results.pop(0)
        assert git_argv == result.expected
        return subprocess.CompletedProcess(argv, result.returncode, result.stdout, result.stderr)


def _probe_script(
    layout: ServerLayout,
    *,
    cleanup_returncode: int = 0,
    cleanup_readback: str = "",
) -> tuple[GitScript, str, str]:
    origin = REPOSITORY.ssh_clone_url
    probe_directory = "/tmp/rcp-git-probe.test"
    git_dir = f"{probe_directory}/repository.git"
    temporary_ref = f"refs/heads/rcp-provisioning-{REQUEST_ID}"
    return (
        GitScript(
            GitResult(
                ("git", "ls-remote", origin, "HEAD"),
                stdout=f"{COMMIT}\tHEAD\n",
            ),
            GitResult(("git", "ls-remote", origin, temporary_ref)),
            GitResult(("git", "init", "--quiet", "--bare", "--template=", git_dir)),
            GitResult(
                (
                    "git",
                    f"--git-dir={git_dir}",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "--depth=1",
                    origin,
                    "HEAD",
                )
            ),
            GitResult(
                ("git", f"--git-dir={git_dir}", "rev-parse", "FETCH_HEAD"),
                stdout=f"{COMMIT}\n",
            ),
            GitResult(
                (
                    "git",
                    f"--git-dir={git_dir}",
                    "push",
                    "--porcelain",
                    f"--force-with-lease={temporary_ref}:",
                    origin,
                    f"{COMMIT}:{temporary_ref}",
                )
            ),
            GitResult(
                ("git", "ls-remote", origin, temporary_ref),
                stdout=f"{COMMIT}\t{temporary_ref}\n",
            ),
            GitResult(
                (
                    "git",
                    f"--git-dir={git_dir}",
                    "push",
                    "--porcelain",
                    f"--force-with-lease={temporary_ref}:{COMMIT}",
                    origin,
                    f":{temporary_ref}",
                ),
                returncode=cleanup_returncode,
            ),
            GitResult(
                ("git", "ls-remote", origin, temporary_ref),
                stdout=cleanup_readback,
            ),
        ),
        probe_directory,
        temporary_ref,
    )


def test_write_probe_pushes_reads_back_and_removes_one_request_scoped_ref(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    script, probe_directory, temporary_ref = _probe_script(layout)
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory=probe_directory,
    )

    assert probe == GitWriteProbe(
        status="ready",
        commit=COMMIT,
        temporary_ref=None,
        diagnostic="The request-scoped Git write probe passed and its temporary ref is gone.",
    )
    assert not script.results
    assert any(temporary_ref in argument for call in script.git_calls for argument in call)
    assert script.git_calls[0] == (
        "git",
        "ls-remote",
        REPOSITORY.ssh_clone_url,
        "HEAD",
    )
    outer = script.outer_calls[0]
    assert outer[:4] == ("runuser", "--user", "rcp", "--")
    assert "GIT_CONFIG_GLOBAL=/dev/null" in outer
    assert "GIT_CONFIG_NOSYSTEM=1" in outer
    assert "GIT_TERMINAL_PROMPT=0" in outer
    ssh_environment = next(value for value in outer if value.startswith("GIT_SSH_COMMAND="))
    assert "-F /dev/null" in ssh_environment
    assert "IdentitiesOnly=yes" in ssh_environment
    assert "BatchMode=yes" in ssh_environment
    assert "StrictHostKeyChecking=yes" in ssh_environment
    assert _material(layout).private_key_path in ssh_environment


def test_write_probe_keeps_the_exact_ref_visible_when_cleanup_cannot_be_proven(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    temporary_ref = f"refs/heads/rcp-provisioning-{REQUEST_ID}"
    script, probe_directory, _ = _probe_script(
        layout,
        cleanup_returncode=1,
        cleanup_readback=f"{COMMIT}\t{temporary_ref}\n",
    )
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory=probe_directory,
    )

    assert probe.status == "cleanup_failed"
    assert probe.temporary_ref == temporary_ref
    assert "Remove that exact ref" in probe.diagnostic


def test_invalid_post_push_readback_still_removes_the_exact_owned_ref(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    script, probe_directory, temporary_ref = _probe_script(layout)
    readback = script.results[6]
    script.results[6] = GitResult(
        readback.expected,
        stdout=f"{COMMIT}\t{temporary_ref}\n{COMMIT}\t{temporary_ref}\n",
    )
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory=probe_directory,
    )

    assert probe.status == "failed"
    assert probe.temporary_ref is None
    assert "invalid write-probe ref record" in probe.diagnostic
    assert not script.results
    assert any(f":{temporary_ref}" in call for call in script.git_calls)


def test_invalid_cleanup_readback_keeps_the_exact_ref_visible(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    script, probe_directory, temporary_ref = _probe_script(
        layout,
        cleanup_readback="not-a-git-ref-record\n",
    )
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory=probe_directory,
    )

    assert probe.status == "cleanup_failed"
    assert probe.temporary_ref == temporary_ref
    assert not script.results


def test_write_probe_reports_a_read_only_deploy_key_as_needing_the_write_grant(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    script, probe_directory, temporary_ref = _probe_script(layout)
    push = script.results[5]
    script.results = [
        *script.results[:5],
        GitResult(
            push.expected,
            returncode=128,
            stderr="ERROR: Write access to repository not granted.",
        ),
    ]
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory=probe_directory,
    )

    assert probe.status == "github_grant_needed"
    assert probe.commit == COMMIT
    assert probe.temporary_ref is None
    assert not script.results
    assert all(argument != f":{temporary_ref}" for call in script.git_calls for argument in call)


def test_ambiguous_push_failure_never_blindly_deletes_the_request_ref(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    script, probe_directory, temporary_ref = _probe_script(layout)
    push = script.results[5]
    observed = script.results[6]
    script.results = [
        *script.results[:5],
        GitResult(push.expected, returncode=128, stderr="connection closed by remote host"),
        GitResult(observed.expected, returncode=128, stderr="network is unreachable"),
    ]
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory=probe_directory,
    )

    assert probe.status == "cleanup_failed"
    assert probe.temporary_ref == temporary_ref
    assert not script.results
    assert all(argument != f":{temporary_ref}" for call in script.git_calls for argument in call)


def test_atomic_create_lease_rejection_never_deletes_another_writers_ref(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    script, probe_directory, temporary_ref = _probe_script(layout)
    push = script.results[5]
    script.results = [
        *script.results[:5],
        GitResult(
            push.expected,
            returncode=1,
            stderr="! [rejected] probe -> probe (stale info)",
        ),
    ]
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory=probe_directory,
    )

    assert probe.status == "temporary_ref_conflict"
    assert probe.temporary_ref == temporary_ref
    assert not script.results
    assert all(argument != f":{temporary_ref}" for call in script.git_calls for argument in call)


def test_ambiguous_failed_push_leaves_observed_commit_for_operator_cleanup(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    script, probe_directory, temporary_ref = _probe_script(layout)
    push = script.results[5]
    observed = script.results[6]
    script.results = [
        *script.results[:5],
        GitResult(push.expected, returncode=128, stderr="connection closed by remote host"),
        GitResult(observed.expected, stdout=f"{COMMIT}\t{temporary_ref}\n"),
    ]
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory=probe_directory,
    )

    assert probe.status == "cleanup_failed"
    assert probe.temporary_ref == temporary_ref
    assert not script.results
    assert all(argument != f":{temporary_ref}" for call in script.git_calls for argument in call)


@pytest.mark.parametrize(
    ("result", "expected_status"),
    (
        (_result(128, stderr="Host key verification failed"), "github_host_trust_needed"),
        (_result(128, stderr="Permission denied (publickey)"), "github_grant_needed"),
        (_result(128, stderr="Could not resolve hostname github.com"), "unavailable"),
        (_result(0, stdout=""), "empty_repository"),
    ),
)
def test_write_probe_reports_host_grant_network_and_empty_repository_states(
    tmp_path: Path,
    result: subprocess.CompletedProcess[str],
    expected_status: str,
) -> None:
    layout = _layout(tmp_path)
    origin = REPOSITORY.ssh_clone_url
    script = GitScript(
        GitResult(
            ("git", "ls-remote", origin, "HEAD"),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    )
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory="/tmp/rcp-git-probe.test",
    )

    assert probe.status == expected_status


def test_write_probe_names_the_exact_unrecognized_failure_stage(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    origin = REPOSITORY.ssh_clone_url
    script = GitScript(
        GitResult(
            ("git", "ls-remote", origin, "HEAD"),
            returncode=128,
            stderr="an intentionally unclassified Git failure",
        )
    )

    probe = GitCredentialManager(layout, runner=script)._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory="/tmp/rcp-git-probe.test",
    )

    assert probe.status == "failed"
    assert "during advertised HEAD lookup" in probe.diagnostic


def test_write_probe_never_deletes_a_preexisting_request_ref(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    origin = REPOSITORY.ssh_clone_url
    temporary_ref = f"refs/heads/rcp-provisioning-{REQUEST_ID}"
    script = GitScript(
        GitResult(
            ("git", "ls-remote", origin, "HEAD"),
            stdout=f"{COMMIT}\tHEAD\n",
        ),
        GitResult(
            ("git", "ls-remote", origin, temporary_ref),
            stdout=f"{COMMIT}\t{temporary_ref}\n",
        ),
    )
    manager = GitCredentialManager(layout, runner=script)

    probe = manager._probe_in_directory(
        _local_machine(layout),
        _material(layout),
        request_id=REQUEST_ID,
        probe_directory="/tmp/rcp-git-probe.test",
    )

    assert probe.status == "temporary_ref_conflict"
    assert probe.temporary_ref == temporary_ref
    assert len(script.git_calls) == 2


def test_probe_directory_cleanup_runs_even_when_ref_parsing_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    manager = GitCredentialManager(layout, runner=QueueRunner())
    material = _material(layout)
    helper_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(manager, "inspect_key", lambda _machine, _material: material)

    def fake_helper(
        _machine: ProjectProvisioningMachineIntent,
        arguments: tuple[str, ...],
    ) -> dict[str, object]:
        helper_calls.append(arguments)
        if arguments[0] == "probe-prepare":
            return {"probe_directory": "/tmp/rcp-git-probe.test"}
        return {"removed": True}

    monkeypatch.setattr(manager, "_helper", fake_helper)
    monkeypatch.setattr(
        manager,
        "_probe_in_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(GitCredentialRefused("invalid Git ref")),
    )

    with pytest.raises(GitCredentialRefused, match="invalid Git ref"):
        manager.probe_write(_local_machine(layout), material, request_id=REQUEST_ID)

    assert [call[0] for call in helper_calls] == ["probe-prepare", "probe-cleanup"]


def test_probe_directory_cleanup_failure_does_not_hide_the_original_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    manager = GitCredentialManager(layout, runner=QueueRunner())
    material = _material(layout)
    probe_directory = "/tmp/rcp-git-probe.test"
    monkeypatch.setattr(manager, "inspect_key", lambda _machine, _material: material)

    def fake_helper(
        _machine: ProjectProvisioningMachineIntent,
        arguments: tuple[str, ...],
    ) -> dict[str, object]:
        if arguments[0] == "probe-prepare":
            return {"probe_directory": probe_directory}
        raise GitCredentialRefused("cleanup refused")

    monkeypatch.setattr(manager, "_helper", fake_helper)
    monkeypatch.setattr(
        manager,
        "_probe_in_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(GitCredentialRefused("invalid Git ref")),
    )

    with pytest.raises(GitCredentialRefused, match="local directory") as caught:
        manager.probe_write(_local_machine(layout), material, request_id=REQUEST_ID)

    assert probe_directory in str(caught.value)
    assert isinstance(caught.value.__cause__, GitCredentialRefused)
    assert str(caught.value.__cause__) == "invalid Git ref"


def test_ref_parser_rejects_duplicate_records() -> None:
    with pytest.raises(GitCredentialRefused, match="duplicate ref"):
        _parse_remote_refs(f"{COMMIT}\tHEAD\n{COMMIT}\tHEAD\n")


def test_operator_steps_publish_only_exact_public_actions_and_resume_contract(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    manager = GitCredentialManager(layout, runner=QueueRunner())
    machine = _local_machine(layout)
    material = _material(layout)
    resume = (
        "rcp",
        "server",
        "project",
        "provision",
        REQUEST_ID,
        "--machine-readable",
    )

    grant = deploy_key_operator_step(
        manager,
        machine,
        material,
        number=2,
        request_id=REQUEST_ID,
        resume_argv=resume,
    )
    fields = {field.name: field.value for field in grant.fields}
    assert grant.target.destination_url == REPOSITORY.settings_url
    assert grant.target.required_authority_role == "repository administrator"
    assert fields == {
        "deploy_key_label": material.label,
        "deploy_public_key": material.public_key,
        "public_key_fingerprint": material.public_key_fingerprint,
    }
    assert "Allow write access" in grant.actions[0].instruction
    assert grant.resume_argv == resume
    serialized = grant.model_dump_json()
    assert "OPENSSH PRIVATE KEY" not in serialized
    assert material.private_key_path in serialized

    empty = empty_repository_operator_step(
        material,
        number=3,
        request_id=REQUEST_ID,
        resume_argv=resume,
    )
    assert "first real commit" in empty.actions[0].instruction
    assert "will not create a repository" in empty.actions[0].instruction

    temporary_ref = f"refs/heads/rcp-provisioning-{REQUEST_ID}"
    cleanup = cleanup_ref_operator_step(
        material,
        GitWriteProbe(
            status="cleanup_failed",
            commit=COMMIT,
            temporary_ref=temporary_ref,
            diagnostic="Remove the exact ref.",
        ),
        number=4,
        request_id=REQUEST_ID,
        resume_argv=resume,
    )
    assert cleanup.fields[0].value == temporary_ref
    assert temporary_ref in cleanup.actions[0].instruction

    with pytest.raises(ValueError, match="exact provisioning request"):
        deploy_key_operator_step(
            manager,
            machine,
            material,
            number=2,
            request_id=REQUEST_ID,
            resume_argv=("rcp", "server", "project", "provision", SPACE_ID),
        )


def test_layout_uses_one_canonical_key_path_for_local_and_remote_accounts(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    relative = Path("projects") / PROJECT_ID / ALIAS / "id_ed25519"

    assert layout.project_deploy_key_path(PROJECT_ID, ALIAS) == layout.credentials_root / relative
    assert remote_project_deploy_key_path("/srv/alice", PROJECT_ID, ALIAS) == (
        Path("/srv/alice/.local/share/rcp/credentials") / relative
    )
    for invalid_alias in ("Paper", "paper_repo", "paper/repo", "a" * 49):
        with pytest.raises(ValueError, match="canonical provisioning alias"):
            layout.project_deploy_key_path(PROJECT_ID, invalid_alias)
