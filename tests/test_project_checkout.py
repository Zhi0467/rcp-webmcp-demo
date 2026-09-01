from __future__ import annotations

import hashlib
import json
import os
import pwd
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rcp.server_ops import remote_project_checkout
from rcp.server_ops.git_credentials import DeployKeyMaterial, _run_process
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.layout import ServerLayout
from rcp.server_ops.project_checkout import (
    ProjectCheckoutManager,
    ProjectCheckoutRefused,
    retained_research_operator_step,
)
from rcp.storage import ProjectProvisioningMachineIntent

SPACE_ID = "7eb4ea9d-cccf-42fd-abfe-09f71f4b8cd2"
PROJECT_ID = "2ad064a6-f015-4703-a223-1d64cde75cc8"
REQUEST_ID = "a29ddba0-a0a7-46be-ab7a-7a6d77644ea5"
ALIAS = "paper"
REPOSITORY = GitHubRepositoryRef(identity="zhi0467/rcp-checkout-live-test")
HELPER = Path(__file__).parents[1] / "src" / "rcp" / "server_ops" / "remote_project_checkout.py"


def _layout(tmp_path: Path) -> ServerLayout:
    account = pwd.getpwuid(os.getuid())
    root = tmp_path / "rcp-server"
    return ServerLayout(
        service_account=account.pw_name,
        service_home=Path(account.pw_dir),
        server_root=root,
        source_checkout=root / "source",
        releases_root=root / "releases",
        data_dir=root / "data",
        projects_root=root / "projects",
        credentials_root=root / "credentials",
        update_checkpoints_root=root / "update-checkpoints",
        restore_operations_root=root / "restore-operations",
        codex_state_root=Path(account.pw_dir) / ".codex",
        claude_state_root=Path(account.pw_dir) / ".claude",
        ssh_state_root=Path(account.pw_dir) / ".ssh",
        config_path=tmp_path / "etc" / "rcp" / "server.toml",
        current_release=tmp_path / "etc" / "rcp" / "current",
        runtime_dir=tmp_path / "run" / "rcp",
        control_socket=tmp_path / "run" / "rcp" / "control.sock",
        cli_wrapper=tmp_path / "usr" / "local" / "bin" / "rcp",
        systemd_unit=tmp_path / "etc" / "systemd" / "system" / "rcp.service",
        service_unit_name="rcp.service",
    )


def _machine(layout: ServerLayout) -> ProjectProvisioningMachineIntent:
    return ProjectProvisioningMachineIntent.model_construct(
        alias="server",
        location="local",
        host="",
        os_account=layout.service_account,
        central_root=str(layout.projects_root),
    )


def _material(layout: ServerLayout) -> DeployKeyMaterial:
    return DeployKeyMaterial(
        space_id=SPACE_ID,
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        repository=REPOSITORY,
        machine_alias="server",
        location="local",
        host="",
        os_account=layout.service_account,
        central_root=str(layout.projects_root),
        account_home=str(layout.service_home),
        credentials_root=str(layout.credentials_root),
        private_key_path=str(layout.project_deploy_key_path(PROJECT_ID, ALIAS)),
        label=f"rcp:{SPACE_ID}:{PROJECT_ID}:{ALIAS}",
        public_key="ssh-ed25519 AAAA fixture",
        public_key_fingerprint="SHA256:" + ("A" * 43),
        created=False,
    )


class _CurrentAccountRunner:
    def __init__(self, account: str) -> None:
        self.account = account
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        expected = ("runuser", "--user", self.account, "--")
        assert argv[:4] == expected
        self.calls.append(argv)
        return _run_process(tuple(argv[4:]), timeout=timeout)


class _StaticCredentialManager:
    def inspect_key(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
    ) -> DeployKeyMaterial:
        assert machine.alias == material.machine_alias
        return material


class _LocalOriginCheckoutManager(ProjectCheckoutManager):
    """Exercise production checkout logic while replacing only GitHub I/O with a bare repo."""

    def __init__(self, *args: object, origin: Path, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.origin = origin

    def _clone(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
        repository_path: str,
    ) -> None:
        cloned = self._git(
            machine,
            material,
            (
                "git",
                "clone",
                "--no-tags",
                "--no-recurse-submodules",
                "--origin",
                "origin",
                "--template=",
                "--config",
                "core.hooksPath=/dev/null",
                "--",
                str(self.origin),
                repository_path,
            ),
        )
        assert cloned.returncode == 0, cloned.stderr
        configured = self._git_at(
            machine,
            material,
            repository_path,
            ("remote", "set-url", "origin", material.repository.ssh_clone_url),
        )
        assert configured.returncode == 0, configured.stderr

    def _git(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
        argv: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        mapped = list(argv)
        if len(mapped) >= 2 and mapped[-2:] == ["origin", "HEAD"]:
            mapped[-2] = str(self.origin)
        elif len(mapped) >= 1 and mapped[-1] == "origin" and "fetch" in mapped:
            mapped[-1] = str(self.origin)
        return super()._git(machine, material, tuple(mapped))


class _InheritedModeCheckoutManager(_LocalOriginCheckoutManager):
    """Model a host default ACL widening modes after Git creates its metadata."""

    def _clone(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
        repository_path: str,
    ) -> None:
        super()._clone(machine, material, repository_path)
        git_directory = Path(repository_path) / ".git"
        git_directory.chmod(0o770)
        (git_directory / "config").chmod(0o660)


def _helper(operation: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(HELPER), operation, *arguments),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _prepare_helper(root: Path) -> subprocess.CompletedProcess[str]:
    account = pwd.getpwuid(os.getuid())
    return _helper(
        "prepare",
        account.pw_name,
        account.pw_dir,
        "local",
        str(root),
        PROJECT_ID,
        ALIAS,
    )


def _git_command(*argv: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ("git", *argv),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _origin(tmp_path: Path, *, retained: bool = False) -> tuple[Path, str]:
    source = tmp_path / "source"
    bare = tmp_path / "origin.git"
    source.mkdir()
    _git_command("init", "--quiet", cwd=source)
    _git_command("config", "user.name", "RCP test", cwd=source)
    _git_command("config", "user.email", "rcp@example.invalid", cwd=source)
    (source / "README.md").write_text("central checkout fixture\n", encoding="utf-8")
    if retained:
        patches = source / ".research" / "patches"
        patches.mkdir(parents=True)
        (patches / "000001.json").write_text(
            json.dumps(
                {
                    "kind": "identity",
                    "project_identity": {
                        "project_id": "651f8a95-c12d-46ef-9ac2-df13e9c96ee2",
                        "home_space_id": "2f8dfa3b-d91e-4d5e-a622-6e35395bdfe7",
                    },
                }
            ),
            encoding="utf-8",
        )
    _git_command("add", ".", cwd=source)
    _git_command("commit", "--quiet", "-m", "fixture", cwd=source)
    commit = _git_command("rev-parse", "HEAD", cwd=source)
    _git_command("init", "--quiet", "--bare", str(bare))
    _git_command("remote", "add", "origin", str(bare), cwd=source)
    _git_command("push", "--quiet", "origin", "HEAD:refs/heads/main", cwd=source)
    _git_command("symbolic-ref", "HEAD", "refs/heads/main", cwd=bare)
    return bare, commit


def _manager(
    tmp_path: Path, origin: Path
) -> tuple[
    _LocalOriginCheckoutManager,
    ServerLayout,
    ProjectProvisioningMachineIntent,
    DeployKeyMaterial,
    _CurrentAccountRunner,
]:
    layout = _layout(tmp_path)
    layout.projects_root.mkdir(parents=True, mode=0o700)
    layout.projects_root.chmod(0o700)
    runner = _CurrentAccountRunner(layout.service_account)
    machine = _machine(layout)
    material = _material(layout)
    manager = _LocalOriginCheckoutManager(
        layout,
        runner=runner,
        credential_manager=_StaticCredentialManager(),  # type: ignore[arg-type]
        origin=origin,
    )
    return manager, layout, machine, material, runner


def test_shipped_helper_creates_and_reuses_only_the_exact_checkout(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    root.mkdir(mode=0o700)

    first = _prepare_helper(root)
    assert first.returncode == 0, first.stderr
    receipt = json.loads(first.stdout)
    checkout = root / PROJECT_ID / "repositories" / ALIAS
    assert receipt == {
        "account": pwd.getpwuid(os.getuid()).pw_name,
        "home": pwd.getpwuid(os.getuid()).pw_dir,
        "central_root": str(root),
        "repository_path": str(checkout),
        "disposition": "request_created",
        "empty": True,
    }
    assert checkout.is_dir()
    # Every directory the helper creates is private to the account it runs as.
    for created in (root / PROJECT_ID, root / PROJECT_ID / "repositories", checkout):
        assert stat.S_IMODE(created.stat().st_mode) == 0o700, created

    sentinel = checkout / "keep-me"
    sentinel.write_text("preserve", encoding="utf-8")
    second = _prepare_helper(root)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout) == {
        **receipt,
        "disposition": "reused_existing",
        "empty": False,
    }
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("unsafe", ["symlink", "mode"])
def test_shipped_helper_refuses_unsafe_central_root(tmp_path: Path, unsafe: str) -> None:
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    root = tmp_path / "projects"
    if unsafe == "symlink":
        root.symlink_to(actual, target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        root.chmod(0o777)

    refused = _prepare_helper(root)

    assert refused.returncode == 2
    assert not (actual / PROJECT_ID).exists()


def test_manager_refuses_a_symlinked_git_directory_before_running_git(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path)
    manager, layout, machine, material, _runner = _manager(tmp_path, origin)
    actual = tmp_path / "actual-checkout"
    _git_command("clone", "--quiet", str(origin), str(actual))
    checkout = layout.projects_root / PROJECT_ID / "repositories" / ALIAS
    checkout.mkdir(parents=True)
    (checkout / "README.md").write_text("do not follow .git\n", encoding="utf-8")
    (checkout / ".git").symlink_to(actual / ".git", target_is_directory=True)

    with pytest.raises(ProjectCheckoutRefused, match="checkout helper refused") as error:
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )

    assert error.value.checkout_disposition == "reused_existing"
    assert "git-directory operation" in str(error.value)
    assert (checkout / ".git").is_symlink()


def test_shipped_helper_resolves_default_remote_root_from_verified_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "remote-home"
    home.mkdir(mode=0o700)
    account = pwd.getpwuid(os.getuid()).pw_name
    monkeypatch.setattr(
        remote_project_checkout.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_name=account, pw_dir=str(home), pw_uid=os.getuid()),
    )

    receipt = remote_project_checkout._prepare(
        account,
        str(home),
        "ssh",
        "-",
        PROJECT_ID,
        ALIAS,
    )

    expected_root = home / ".local" / "share" / "rcp" / "projects"
    assert receipt["central_root"] == str(expected_root)
    assert receipt["repository_path"] == str(expected_root / PROJECT_ID / "repositories" / ALIAS)


def test_retained_patch_scan_has_one_cumulative_entry_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patches = tmp_path / "patches"
    for batch in ("batch-a", "batch-b"):
        directory = patches / batch
        directory.mkdir(parents=True)
        (directory / "000001.json").write_text("{}", encoding="utf-8")
        (directory / "000002.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(remote_project_checkout, "MAX_RESEARCH_ENTRIES", 4)
    descriptor = os.open(patches, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert remote_project_checkout._patch_names(descriptor) == ["too-many"]
    finally:
        os.close(descriptor)


def test_manager_clones_verifies_and_recovers_without_renaming(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path)
    manager, layout, machine, material, runner = _manager(tmp_path, origin)

    first = manager.prepare(
        machine,
        material,
        request_kind="create_team_project",
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        state_repository=True,
        expected_commit=commit,
    )
    second = manager.prepare(
        machine,
        material,
        request_kind="create_team_project",
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        state_repository=True,
        expected_commit=commit,
    )

    expected_path = layout.projects_root / PROJECT_ID / "repositories" / ALIAS
    assert first.repository_path == str(expected_path)
    assert first.checkout_disposition == "request_created"
    assert first.commit == commit
    assert first.retained_research.retained is False
    assert second.checkout_disposition == "reused_existing"
    assert _git_command("config", "--local", "--get", "remote.origin.url", cwd=expected_path) == (
        REPOSITORY.ssh_clone_url
    )
    assert _git_command("config", "--local", "--get", "core.hooksPath", cwd=expected_path) == (
        "/dev/null"
    )
    assert runner.calls
    assert all(
        call[:4] == ("runuser", "--user", layout.service_account, "--") for call in runner.calls
    )
    git_calls = [call for call in runner.calls if "GIT_CONFIG_GLOBAL=/dev/null" in call]
    assert git_calls
    assert all("GIT_TERMINAL_PROMPT=0" in call for call in git_calls)
    assert all(
        not any(token in {"reset", "clean", "stash"} for token in call) for call in git_calls
    )


def test_manager_clones_with_private_modes_under_operator_umask(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path)
    manager, _layout_value, machine, material, _runner = _manager(tmp_path, origin)

    previous_umask = os.umask(0o002)
    try:
        prepared = manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )
    finally:
        os.umask(previous_umask)

    git_directory = Path(prepared.repository_path) / ".git"
    assert stat.S_IMODE(git_directory.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE((git_directory / "config").stat().st_mode) & 0o077 == 0


def test_manager_seals_modes_inherited_by_a_new_clone(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path)
    manager, layout, machine, material, runner = _manager(tmp_path, origin)
    manager = _InheritedModeCheckoutManager(
        layout,
        runner=runner,
        credential_manager=_StaticCredentialManager(),  # type: ignore[arg-type]
        origin=origin,
    )

    prepared = manager.prepare(
        machine,
        material,
        request_kind="create_team_project",
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        state_repository=False,
        expected_commit=commit,
    )

    git_directory = Path(prepared.repository_path) / ".git"
    assert stat.S_IMODE(git_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((git_directory / "config").stat().st_mode) == 0o600


def test_manager_refuses_wrong_origin_without_rewriting_it(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path)
    manager, layout, machine, material, _runner = _manager(tmp_path, origin)
    checkout = layout.projects_root / PROJECT_ID / "repositories" / ALIAS
    checkout.parent.mkdir(parents=True)
    _git_command("clone", "--quiet", str(origin), str(checkout))
    wrong = "git@github.com:someone/else.git"
    _git_command("remote", "set-url", "origin", wrong, cwd=checkout)

    with pytest.raises(ProjectCheckoutRefused, match="canonical GitHub repository") as error:
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )

    assert error.value.kind == "checkout_conflict"
    assert error.value.checkout_disposition == "reused_existing"
    assert _git_command("config", "--local", "--get", "remote.origin.url", cwd=checkout) == wrong


def test_manager_refuses_local_git_execution_and_url_overrides(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path)
    manager, _layout_value, machine, material, _runner = _manager(tmp_path, origin)
    prepared = manager.prepare(
        machine,
        material,
        request_kind="create_team_project",
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        state_repository=False,
        expected_commit=commit,
    )
    checkout = Path(prepared.repository_path)
    _git_command(
        "config",
        "url.file:///tmp/not-the-reviewed-origin.insteadOf",
        REPOSITORY.ssh_clone_url,
        cwd=checkout,
    )

    with pytest.raises(ProjectCheckoutRefused, match="unsafe local Git"):
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )


def test_manager_refuses_an_origin_fetch_mapping_that_can_rewrite_local_branches(
    tmp_path: Path,
) -> None:
    origin, commit = _origin(tmp_path)
    manager, layout, machine, material, _runner = _manager(tmp_path, origin)
    checkout = layout.projects_root / PROJECT_ID / "repositories" / ALIAS
    checkout.parent.mkdir(parents=True)
    _git_command("clone", "--quiet", str(origin), str(checkout))
    _git_command("remote", "set-url", "origin", REPOSITORY.ssh_clone_url, cwd=checkout)
    unsafe_refspec = "+refs/heads/*:refs/heads/*"
    _git_command("config", "remote.origin.fetch", unsafe_refspec, cwd=checkout)

    with pytest.raises(ProjectCheckoutRefused, match="unsafe origin fetch mapping"):
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )

    assert _git_command("config", "--local", "--get", "remote.origin.fetch", cwd=checkout) == (
        unsafe_refspec
    )


def test_manager_preserves_dirty_and_divergent_existing_checkout(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path)
    manager, layout, machine, material, _runner = _manager(tmp_path, origin)
    prepared = manager.prepare(
        machine,
        material,
        request_kind="create_team_project",
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        state_repository=False,
        expected_commit=commit,
    )
    checkout = Path(prepared.repository_path)
    dirty = checkout / "untracked.txt"
    dirty.write_text("do not remove", encoding="utf-8")

    with pytest.raises(ProjectCheckoutRefused, match="uncommitted or untracked"):
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )
    assert dirty.read_text(encoding="utf-8") == "do not remove"

    dirty.unlink()
    _git_command("config", "user.name", "RCP test", cwd=checkout)
    _git_command("config", "user.email", "rcp@example.invalid", cwd=checkout)
    (checkout / "local-only.txt").write_text("preserve commit\n", encoding="utf-8")
    _git_command("add", "local-only.txt", cwd=checkout)
    _git_command("commit", "--quiet", "-m", "local only", cwd=checkout)
    local_commit = _git_command("rev-parse", "HEAD", cwd=checkout)

    with pytest.raises(ProjectCheckoutRefused, match="differ"):
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )
    assert _git_command("rev-parse", "HEAD", cwd=checkout) == local_commit


def test_manager_does_not_rewrite_hooks_before_refusing_existing_dirty_work(
    tmp_path: Path,
) -> None:
    origin, commit = _origin(tmp_path)
    manager, layout, machine, material, _runner = _manager(tmp_path, origin)
    checkout = layout.projects_root / PROJECT_ID / "repositories" / ALIAS
    checkout.parent.mkdir(parents=True)
    _git_command("clone", "--quiet", str(origin), str(checkout))
    _git_command("remote", "set-url", "origin", REPOSITORY.ssh_clone_url, cwd=checkout)
    dirty = checkout / "preserve.txt"
    dirty.write_text("do not rewrite config\n", encoding="utf-8")

    with pytest.raises(ProjectCheckoutRefused, match="uncommitted or untracked"):
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )

    hooks = subprocess.run(
        ("git", "config", "--local", "--get-all", "core.hooksPath"),
        cwd=checkout,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert hooks.returncode == 1
    assert hooks.stdout == ""
    assert dirty.read_text(encoding="utf-8") == "do not rewrite config\n"


def test_manager_refuses_existing_hook_path_without_rewriting_it(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path)
    manager, layout, machine, material, _runner = _manager(tmp_path, origin)
    checkout = layout.projects_root / PROJECT_ID / "repositories" / ALIAS
    checkout.parent.mkdir(parents=True)
    _git_command("clone", "--quiet", str(origin), str(checkout))
    _git_command("remote", "set-url", "origin", REPOSITORY.ssh_clone_url, cwd=checkout)
    _git_command("config", "core.hooksPath", "/tmp/operator-hooks", cwd=checkout)

    with pytest.raises(ProjectCheckoutRefused, match="unsafe repository hook path"):
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=False,
            expected_commit=commit,
        )

    assert _git_command("config", "--local", "--get", "core.hooksPath", cwd=checkout) == (
        "/tmp/operator-hooks"
    )


def test_direct_creation_stops_on_retained_research_but_transfer_reports_it(
    tmp_path: Path,
) -> None:
    origin, commit = _origin(tmp_path, retained=True)
    manager, _layout_value, machine, material, _runner = _manager(tmp_path, origin)

    with pytest.raises(ProjectCheckoutRefused, match="Move to team space") as error:
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=True,
            expected_commit=commit,
        )

    refusal = error.value
    assert refusal.kind == "retained_research"
    assert refusal.checkout_disposition == "request_created"
    assert refusal.retained_research is not None
    assert refusal.retained_research.patch_history is True
    assert refusal.retained_research.project_id == "651f8a95-c12d-46ef-9ac2-df13e9c96ee2"
    step = retained_research_operator_step(
        machine,
        refusal,
        number=4,
        request_id=REQUEST_ID,
        resume_argv=("rcp", "server", "project", "provision", REQUEST_ID),
        local_host="team-server",
    )
    assert step.state == "operator_action_needed"
    assert step.target.kind == "machine"
    assert step.fields[0].value == refusal.repository_path
    assert "Move to team space" in step.actions[0].instruction

    transferred = manager.prepare(
        machine,
        material,
        request_kind="incoming_transfer",
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        state_repository=True,
        expected_commit=commit,
    )
    assert transferred.checkout_disposition == "reused_existing"
    assert transferred.retained_research.patch_history is True


def test_reused_personal_research_is_refused_before_git_config_changes(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path, retained=True)
    manager, layout, machine, material, _runner = _manager(tmp_path, origin)
    checkout = layout.projects_root / PROJECT_ID / "repositories" / ALIAS
    checkout.parent.mkdir(parents=True)
    _git_command("clone", "--quiet", str(origin), str(checkout))
    _git_command("remote", "set-url", "origin", REPOSITORY.ssh_clone_url, cwd=checkout)

    with pytest.raises(ProjectCheckoutRefused, match="Move to team space") as error:
        manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=True,
            expected_commit=commit,
        )

    hooks = subprocess.run(
        ("git", "config", "--local", "--get-all", "core.hooksPath"),
        cwd=checkout,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert error.value.checkout_disposition == "reused_existing"
    assert hooks.returncode == 1
    assert hooks.stdout == ""


def test_recovery_accepts_only_byte_identical_archived_research(tmp_path: Path) -> None:
    origin, commit = _origin(tmp_path, retained=True)
    manager, _layout_value, machine, material, _runner = _manager(tmp_path, origin)
    retained = json.dumps(
        {
            "kind": "identity",
            "project_identity": {
                "project_id": "651f8a95-c12d-46ef-9ac2-df13e9c96ee2",
                "home_space_id": "2f8dfa3b-d91e-4d5e-a622-6e35395bdfe7",
            },
        }
    ).encode()
    archived = {
        ".research/patches/000001.json": (
            hashlib.sha256(retained).hexdigest(),
            len(retained),
        )
    }

    recovered = manager.prepare_recovery(
        machine,
        material,
        project_id=PROJECT_ID,
        repository_alias=ALIAS,
        state_repository=True,
        expected_head=commit,
        retained_provisioning_commit=commit,
        archived_research=archived,
    )

    assert recovered.commit == commit
    with pytest.raises(ProjectCheckoutRefused, match="newer, unknown, or different"):
        manager.prepare_recovery(
            machine,
            material,
            project_id=PROJECT_ID,
            repository_alias=ALIAS,
            state_repository=True,
            expected_head=commit,
            retained_provisioning_commit=commit,
            archived_research={".research/patches/000001.json": ("f" * 64, len(retained))},
        )


def _recovery_policy(*, offset: int = 0, page_size: int = 8) -> str:
    """The recovery policy the manager ships with every remote inspection."""

    return json.dumps(
        {
            "durable_roots": [
                "branches",
                "chat",
                "facts",
                "manifest.toml",
                "paper",
                "patches",
                "scope-base.json",
            ],
            "excluded_direct": [
                ".agent-run.lock",
                ".append.lock",
                ".chat.lock",
                ".publish",
                ".refresh.lock",
                "coverage.json",
                "cursors.json",
                "glossary.json",
                "graph.json",
                "proposals.json",
                "research.md",
            ],
            "excluded_names": [
                "coverage.json",
                "cursors.json",
                "glossary.json",
                "graph.json",
                "proposals.json",
                "research.md",
            ],
            "excluded_prefixes": [".batch-", ".unconfirmed-"],
            "offset": offset,
            "page_size": page_size,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def test_recovery_research_helper_refuses_unclassified_or_symlinked_input(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    research = repository / ".research"
    research.mkdir(parents=True)
    account = pwd.getpwuid(os.getuid())
    policy = _recovery_policy()
    (research / "future-state").mkdir()

    refused = _helper(
        "recovery-research",
        account.pw_name,
        account.pw_dir,
        str(repository),
        policy,
    )

    assert refused.returncode == 2
    (research / "future-state").rmdir()
    outside = tmp_path / "outside"
    outside.write_text("do not read", encoding="utf-8")
    (research / "manifest.toml").symlink_to(outside)

    symlinked = _helper(
        "recovery-research",
        account.pw_name,
        account.pw_dir,
        str(repository),
        policy,
    )

    assert symlinked.returncode == 2


def _account_arguments() -> tuple[str, str]:
    account = pwd.getpwuid(os.getuid())
    return account.pw_name, account.pw_dir


def _checkout_with_git(tmp_path: Path) -> Path:
    """One account-owned checkout carrying a safe `.git/config`."""

    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)
    git = repository / ".git"
    git.mkdir(mode=0o700)
    config = git / "config"
    config.write_text("[core]\n\trepositoryformatversion = 0\n", encoding="utf-8")
    config.chmod(0o600)
    return repository


def test_shipped_helper_proves_the_central_git_directory_before_any_git_runs(
    tmp_path: Path,
) -> None:
    repository = _checkout_with_git(tmp_path)

    proved = _helper("git-directory", *_account_arguments(), str(repository))

    assert proved.returncode == 0, proved.stderr
    assert json.loads(proved.stdout) == {
        "repository_path": str(repository),
        "safe": True,
    }


def test_shipped_helper_seals_owned_new_git_metadata_before_verification(
    tmp_path: Path,
) -> None:
    repository = _checkout_with_git(tmp_path)
    git = repository / ".git"
    config = git / "config"
    git.chmod(0o770)
    config.chmod(0o660)

    sealed = _helper("seal-git-directory", *_account_arguments(), str(repository))
    proved = _helper("git-directory", *_account_arguments(), str(repository))

    assert sealed.returncode == 0, sealed.stderr
    assert json.loads(sealed.stdout) == {
        "repository_path": str(repository),
        "sealed": True,
    }
    assert proved.returncode == 0, proved.stderr
    assert stat.S_IMODE(git.stat().st_mode) == 0o700
    assert stat.S_IMODE(config.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("mode", "size"),
    [
        pytest.param(0o701, None, id="executable-but-not-writable"),
        pytest.param(0o604, None, id="readable-by-others"),
        pytest.param(
            0o600, remote_project_checkout.MAX_GIT_CONFIG_BYTES, id="exactly-at-the-size-bound"
        ),
    ],
)
def test_shipped_helper_accepts_a_git_config_only_a_stranger_cannot_rewrite(
    tmp_path: Path,
    mode: int,
    size: int | None,
) -> None:
    """The config gate is about who can write, and its size bound is inclusive.

    Later `git` commands obey this file, so the check exists to stop another
    account changing it. Rejecting a config that is merely readable, merely
    executable, or exactly at the bound would refuse ordinary hosts instead.
    """

    repository = _checkout_with_git(tmp_path)
    config = repository / ".git" / "config"
    if size is not None:
        config.write_text("#" * size, encoding="utf-8")
    config.chmod(mode)

    proved = _helper("git-directory", *_account_arguments(), str(repository))

    assert proved.returncode == 0, proved.stderr
    assert json.loads(proved.stdout)["repository_path"] == str(repository)


def test_shipped_helper_refuses_a_git_config_one_byte_over_its_bound(
    tmp_path: Path,
) -> None:
    repository = _checkout_with_git(tmp_path)
    (repository / ".git" / "config").write_text(
        "#" * (remote_project_checkout.MAX_GIT_CONFIG_BYTES + 1), encoding="utf-8"
    )

    refused = _helper("git-directory", *_account_arguments(), str(repository))

    assert refused.returncode == 2
    assert refused.stdout == ""


def _remove_git(repository: Path) -> None:
    for child in sorted((repository / ".git").iterdir()):
        child.unlink()
    (repository / ".git").rmdir()


def _symlink_git(repository: Path) -> None:
    _remove_git(repository)
    elsewhere = repository.parent / "attacker-git"
    elsewhere.mkdir(mode=0o700)
    (elsewhere / "config").write_text("[core]\n", encoding="utf-8")
    (repository / ".git").symlink_to(elsewhere, target_is_directory=True)


def _group_writable_git(repository: Path) -> None:
    (repository / ".git").chmod(0o770)


def _shared_config(repository: Path) -> None:
    (repository / ".git" / "config").chmod(0o666)


def _symlink_config(repository: Path) -> None:
    elsewhere = repository.parent / "attacker-config"
    elsewhere.write_text("[core]\n", encoding="utf-8")
    (repository / ".git" / "config").unlink()
    (repository / ".git" / "config").symlink_to(elsewhere)


@pytest.mark.parametrize(
    "break_checkout",
    [
        pytest.param(_remove_git, id="missing-git-directory"),
        pytest.param(_symlink_git, id="symlinked-git-directory"),
        pytest.param(_group_writable_git, id="group-writable-git-directory"),
        pytest.param(_shared_config, id="world-writable-config"),
        pytest.param(_symlink_config, id="symlinked-config"),
    ],
)
def test_shipped_helper_refuses_an_unsafe_git_directory_or_config(
    tmp_path: Path,
    break_checkout: object,
) -> None:
    """The helper runs on the remote host, so it trusts nothing it did not open itself.

    Every case here is a way another account could redirect or rewrite the Git
    configuration that later Git commands obey.
    """

    repository = _checkout_with_git(tmp_path)
    break_checkout(repository)  # type: ignore[operator]

    refused = _helper("git-directory", *_account_arguments(), str(repository))

    assert refused.returncode == 2
    assert refused.stdout == ""


def test_shipped_helper_refuses_a_relative_path_or_an_account_it_is_not_running_as(
    tmp_path: Path,
) -> None:
    repository = _checkout_with_git(tmp_path)
    account, home = _account_arguments()

    assert _helper("git-directory", account, home, "checkout").returncode == 2
    assert _helper("git-directory", "rcp-not-this-account", home, str(repository)).returncode == 2


def _research(repository: Path) -> Path:
    research = repository / ".research"
    research.mkdir(mode=0o700)
    return research


def _patch(research: Path, document: object) -> None:
    patches = research / "patches"
    patches.mkdir(mode=0o700, exist_ok=True)
    payload = document if isinstance(document, str) else json.dumps(document)
    (patches / "000001.json").write_text(payload, encoding="utf-8")


def test_shipped_helper_reports_no_retained_research_when_there_is_none(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)

    reported = _helper("retained", *_account_arguments(), str(repository))

    assert reported.returncode == 0, reported.stderr
    assert json.loads(reported.stdout) == {
        "retained": False,
        "patch_history": False,
        "project_id": None,
        "home_space_id": None,
    }


def test_shipped_helper_treats_an_empty_research_directory_as_nothing_retained(
    tmp_path: Path,
) -> None:
    """An empty `.research` is not retained research.

    The directory alone does not make a project. Reporting it as retained would
    refuse a checkout that holds no history to lose.
    """

    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)
    _research(repository)

    reported = _helper("retained", *_account_arguments(), str(repository))

    assert reported.returncode == 0, reported.stderr
    assert json.loads(reported.stdout)["retained"] is False


def test_shipped_helper_reports_retained_research_without_patch_history(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)
    (_research(repository) / "graph.json").write_text("{}", encoding="utf-8")

    reported = _helper("retained", *_account_arguments(), str(repository))

    assert reported.returncode == 0, reported.stderr
    assert json.loads(reported.stdout) == {
        "retained": True,
        "patch_history": False,
        "project_id": None,
        "home_space_id": None,
    }


def test_shipped_helper_reads_the_recorded_identity_from_the_first_patch(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)
    _patch(
        _research(repository),
        {"project_identity": {"project_id": PROJECT_ID, "home_space_id": SPACE_ID}},
    )

    reported = _helper("retained", *_account_arguments(), str(repository))

    assert reported.returncode == 0, reported.stderr
    assert json.loads(reported.stdout) == {
        "retained": True,
        "patch_history": True,
        "project_id": PROJECT_ID,
        "home_space_id": SPACE_ID,
    }


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("this is not json", id="unparsable"),
        pytest.param({"project_identity": {}}, id="no-identity-fields"),
        pytest.param(
            {"project_identity": {"project_id": "not-a-uuid", "home_space_id": SPACE_ID}},
            id="project-id-not-uuid4",
        ),
        pytest.param(
            {"project_identity": {"project_id": PROJECT_ID, "home_space_id": 7}},
            id="home-space-id-not-a-string",
        ),
        pytest.param(["project_identity"], id="patch-is-not-an-object"),
    ],
)
def test_shipped_helper_still_reports_retained_history_when_identity_is_unreadable(
    tmp_path: Path,
    document: object,
) -> None:
    """An unreadable identity must never downgrade the retained-history answer.

    The identity is a convenience for the operator prompt. The refusal that
    protects existing research depends on patch_history, so a malformed or
    hostile patch must not clear it.
    """

    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)
    _patch(_research(repository), document)

    reported = _helper("retained", *_account_arguments(), str(repository))

    assert reported.returncode == 0, reported.stderr
    assert json.loads(reported.stdout) == {
        "retained": True,
        "patch_history": True,
        "project_id": None,
        "home_space_id": None,
    }


def test_shipped_helper_refuses_symlinked_retained_research(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)
    elsewhere = tmp_path / "attacker-research"
    elsewhere.mkdir(mode=0o700)
    (repository / ".research").symlink_to(elsewhere, target_is_directory=True)

    refused = _helper("retained", *_account_arguments(), str(repository))

    assert refused.returncode == 2
    assert refused.stdout == ""


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(("nonsense",), id="unknown-operation"),
        pytest.param(("retained",), id="retained-missing-arguments"),
        pytest.param(("git-directory", "one", "two"), id="git-directory-too-few"),
        pytest.param(("prepare", "a", "b", "c"), id="prepare-too-few"),
        pytest.param(
            ("recovery-research", "one", "two", "three"),
            id="recovery-research-too-few",
        ),
    ],
)
def test_shipped_helper_rejects_an_unknown_operation_or_wrong_argument_count(
    argv: tuple[str, ...],
) -> None:
    refused = _helper(*argv)

    assert refused.returncode == 2
    assert refused.stdout == ""


def _recovery_tree(tmp_path: Path) -> Path:
    """One checkout whose `.research` mixes durable, excluded, and nested entries."""

    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)
    research = repository / ".research"
    research.mkdir(mode=0o700)
    (research / "manifest.toml").write_text("x\n", encoding="utf-8")
    (research / "scope-base.json").write_text("{}", encoding="utf-8")
    (research / "graph.json").write_text("rebuildable", encoding="utf-8")
    patches = research / "patches"
    patches.mkdir(mode=0o700)
    (patches / "000001.json").write_text("{}", encoding="utf-8")
    chat = research / "chat"
    chat.mkdir(mode=0o700)
    (chat / "a.json").write_text("a", encoding="utf-8")
    (chat / "graph.json").write_text("rebuildable", encoding="utf-8")
    (chat / ".batch-scratch").write_text("scratch", encoding="utf-8")
    return repository


def test_recovery_research_reports_an_empty_inventory_when_nothing_is_retained(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)

    reported = _helper(
        "recovery-research",
        *_account_arguments(),
        str(repository),
        _recovery_policy(),
    )

    assert reported.returncode == 0, reported.stderr
    assert json.loads(reported.stdout) == {
        "research_present": False,
        "inventory_sha256": hashlib.sha256(b"[]").hexdigest(),
        "total_files": 0,
        "next_offset": None,
        "files": [],
    }


def test_recovery_research_inventories_durable_bytes_and_skips_rebuildable_ones(
    tmp_path: Path,
) -> None:
    """Only durable bytes are inventoried, at every depth.

    `graph.json` is rebuildable both as a direct root and inside a durable
    subtree, and a `.batch-` scratch entry is never durable.
    """

    repository = _recovery_tree(tmp_path)

    reported = _helper(
        "recovery-research",
        *_account_arguments(),
        str(repository),
        _recovery_policy(),
    )

    assert reported.returncode == 0, reported.stderr
    payload = json.loads(reported.stdout)
    assert payload["research_present"] is True
    assert payload["total_files"] == 4
    assert [entry["path"] for entry in payload["files"]] == [
        ".research/chat/a.json",
        ".research/manifest.toml",
        ".research/patches/000001.json",
        ".research/scope-base.json",
    ]
    single_byte = next(e for e in payload["files"] if e["path"] == ".research/chat/a.json")
    assert single_byte["size_bytes"] == 1
    assert single_byte["sha256"] == hashlib.sha256(b"a").hexdigest()


def test_recovery_research_pages_without_changing_the_whole_inventory_digest(
    tmp_path: Path,
) -> None:
    """The digest covers the inventory, not the page.

    A caller reassembling pages needs one digest to prove the tree did not change
    underneath it between requests.
    """

    repository = _recovery_tree(tmp_path)
    account = _account_arguments()

    first = _helper(
        "recovery-research", *account, str(repository), _recovery_policy(offset=0, page_size=3)
    )
    second = _helper(
        "recovery-research", *account, str(repository), _recovery_policy(offset=3, page_size=3)
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    head, tail = json.loads(first.stdout), json.loads(second.stdout)

    assert head["next_offset"] == 3
    assert tail["next_offset"] is None
    assert head["inventory_sha256"] == tail["inventory_sha256"]
    assert len(head["files"]) == 3
    assert len(tail["files"]) == 1
    assert [entry["path"] for entry in head["files"] + tail["files"]] == [
        ".research/chat/a.json",
        ".research/manifest.toml",
        ".research/patches/000001.json",
        ".research/scope-base.json",
    ]


def _symlink_inside_durable_root(repository: Path) -> None:
    (repository / ".research" / "chat" / "escape.json").symlink_to(
        repository.parent / "outside.json"
    )


def _group_writable_research_file(repository: Path) -> None:
    (repository / ".research" / "chat" / "a.json").chmod(0o666)


def _group_writable_research_directory(repository: Path) -> None:
    (repository / ".research" / "chat").chmod(0o770)


@pytest.mark.parametrize(
    "break_tree",
    [
        pytest.param(_symlink_inside_durable_root, id="symlink-inside-durable-root"),
        pytest.param(_group_writable_research_file, id="group-writable-file"),
        pytest.param(_group_writable_research_directory, id="group-writable-directory"),
    ],
)
def test_recovery_research_refuses_bytes_another_account_could_have_changed(
    tmp_path: Path,
    break_tree: object,
) -> None:
    repository = _recovery_tree(tmp_path)
    (tmp_path / "outside.json").write_text("attacker", encoding="utf-8")
    break_tree(repository)  # type: ignore[operator]

    refused = _helper(
        "recovery-research",
        *_account_arguments(),
        str(repository),
        _recovery_policy(),
    )

    assert refused.returncode == 2
    assert refused.stdout == ""


@pytest.mark.parametrize("page_size", [1, 16])
def test_recovery_research_accepts_both_ends_of_its_page_bound(
    tmp_path: Path,
    page_size: int,
) -> None:
    """Both ends of the documented page range are usable.

    A caller that walks the inventory one file at a time, and one that asks for
    the largest page, are the two callers the bound exists for.
    """

    repository = _recovery_tree(tmp_path)

    paged = _helper(
        "recovery-research",
        *_account_arguments(),
        str(repository),
        _recovery_policy(page_size=page_size),
    )

    assert paged.returncode == 0, paged.stderr
    assert len(json.loads(paged.stdout)["files"]) == min(page_size, 4)


def test_recovery_research_refuses_a_page_larger_than_its_bound(tmp_path: Path) -> None:
    repository = _recovery_tree(tmp_path)

    refused = _helper(
        "recovery-research",
        *_account_arguments(),
        str(repository),
        _recovery_policy(page_size=17),
    )

    assert refused.returncode == 2
    assert refused.stdout == ""


def test_recovery_research_ends_on_an_empty_page_rather_than_a_refusal(
    tmp_path: Path,
) -> None:
    """Reading exactly to the end is a finished walk, not an error.

    A caller paging to the end lands on offset == total. Refusing there would
    make a completed inventory read indistinguishable from a corrupted one.
    """

    repository = _recovery_tree(tmp_path)

    ended = _helper(
        "recovery-research",
        *_account_arguments(),
        str(repository),
        _recovery_policy(offset=4, page_size=2),
    )

    assert ended.returncode == 0, ended.stderr
    finished = json.loads(ended.stdout)
    assert finished["total_files"] == 4
    assert finished["files"] == []
    assert finished["next_offset"] is None


def test_recovery_research_refuses_a_page_outside_its_inventory(tmp_path: Path) -> None:
    repository = _recovery_tree(tmp_path)

    refused = _helper(
        "recovery-research",
        *_account_arguments(),
        str(repository),
        _recovery_policy(offset=99, page_size=2),
    )

    assert refused.returncode == 2
    assert refused.stdout == ""


_ABSENT = object()


def _policy_variant(**overrides: object) -> str:
    policy = json.loads(_recovery_policy())
    for key, value in overrides.items():
        if value is _ABSENT:
            policy.pop(key)
        else:
            policy[key] = value
    return json.dumps(policy, separators=(",", ":"), sort_keys=True)


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param("{ not json", id="unparsable"),
        pytest.param(_policy_variant(offset=_ABSENT), id="missing-key"),
        pytest.param(_policy_variant(surprise=1), id="unknown-key"),
        pytest.param(_policy_variant(durable_roots="patches"), id="root-set-is-not-a-list"),
        pytest.param(_policy_variant(durable_roots=[7]), id="root-is-not-a-string"),
        pytest.param(
            _policy_variant(
                durable_roots=[
                    "",
                    "branches",
                    "chat",
                    "facts",
                    "manifest.toml",
                    "paper",
                    "patches",
                    "scope-base.json",
                ]
            ),
            id="root-is-an-empty-name",
        ),
        pytest.param(_policy_variant(excluded_prefixes=[None]), id="prefix-is-not-a-string"),
        pytest.param(_policy_variant(offset=-1), id="negative-offset"),
        pytest.param(_policy_variant(page_size=0), id="empty-page"),
        pytest.param(_policy_variant(offset="0"), id="offset-is-not-an-integer"),
    ],
)
def test_recovery_research_refuses_a_policy_it_cannot_read_exactly(
    tmp_path: Path,
    policy: str,
) -> None:
    """The policy decides what counts as durable, so a vague one is not usable.

    Accepting a partial policy would silently change which bytes are recovered.
    """

    repository = _recovery_tree(tmp_path)

    refused = _helper("recovery-research", *_account_arguments(), str(repository), policy)

    assert refused.returncode == 2
    assert refused.stdout == ""


def test_shipped_helper_reads_identity_from_a_batched_patch_directory(
    tmp_path: Path,
) -> None:
    """Patches may live in `batch-` directories, and the first patch still wins.

    Ordering is by patch filename, not by directory, so a batched `000001.json`
    precedes a later loose `000002.json`.
    """

    repository = tmp_path / "checkout"
    repository.mkdir(mode=0o700)
    research = _research(repository)
    patches = research / "patches"
    patches.mkdir(mode=0o700)
    batch = patches / "batch-0001"
    batch.mkdir(mode=0o700)
    (batch / "000001.json").write_text(
        json.dumps({"project_identity": {"project_id": PROJECT_ID, "home_space_id": SPACE_ID}}),
        encoding="utf-8",
    )
    (patches / "000002.json").write_text(
        json.dumps({"project_identity": {"project_id": REQUEST_ID, "home_space_id": SPACE_ID}}),
        encoding="utf-8",
    )

    reported = _helper("retained", *_account_arguments(), str(repository))

    assert reported.returncode == 0, reported.stderr
    assert json.loads(reported.stdout) == {
        "retained": True,
        "patch_history": True,
        "project_id": PROJECT_ID,
        "home_space_id": SPACE_ID,
    }
