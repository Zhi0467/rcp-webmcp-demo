"""Explicitly gated local and SSH qualification for central checkout preparation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pwd
from pathlib import Path

import pytest

from rcp.server_ops.git_credentials import (
    DeployKeyMaterial,
    GitCredentialManager,
    target_account_argv,
)
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.project_checkout import ProjectCheckoutManager
from rcp.storage import ProjectProvisioningMachineIntent

from .test_git_credentials_live import (
    _GITHUB_ED25519_FINGERPRINT,
    _CurrentAccountRunner,
    _github_optional_request,
    _github_request,
    _live_layout,
    _matching_deploy_key_ids,
    _require_published_github_host_key,
)

_LIVE_GATE = "RCP_RUN_PROJECT_CHECKOUT_LIVE"
_DISPOSABLE_CONFIRMATION = "RCP_PROJECT_CHECKOUT_LIVE_DISPOSABLE"
_EXPECTED_CONFIRMATION = "I_UNDERSTAND_THIS_ADDS_AND_REMOVES_DEPLOY_KEYS_AND_CHECKOUTS"
_TOKEN = "RCP_LIVE_GITHUB_ADMIN_TOKEN"
_REPOSITORY = "RCP_LIVE_GITHUB_REPOSITORY"
_SSH_HOST = "RCP_LIVE_PROJECT_CHECKOUT_SSH_HOST"
_SSH_ACCOUNT = "RCP_LIVE_PROJECT_CHECKOUT_SSH_ACCOUNT"

_LOCAL_SPACE_ID = "a394f820-e0a0-43cb-8d62-21ee382bf26a"
_LOCAL_PROJECT_ID = "e13fd21c-3f2c-4f9e-8a12-72d18a2a3b40"
_LOCAL_REQUEST_ID = "42e8b19c-41ea-48be-8744-89ef67f32f0e"
_REMOTE_SPACE_ID = "b6d15e5d-5784-4467-a17f-485f52c50954"
_REMOTE_PROJECT_ID = "b0ea4e9d-ec27-41d5-9b61-e1f8c3f57e20"
_MAX_REMOTE_OUTPUT_BYTES = 64 * 1024

_LIVE_ONLY = pytest.mark.skipif(
    os.environ.get(_LIVE_GATE) != "1",
    reason="destructive disposable-repository checkout qualification is disabled",
)


@_LIVE_ONLY
def test_local_central_checkout_with_repository_scoped_key(tmp_path: Path) -> None:
    token, repository = _live_inputs()
    account = pwd.getpwuid(os.getuid())
    _require_published_github_host_key(Path(account.pw_dir) / ".ssh" / "known_hosts")
    layout = _live_layout(tmp_path, account.pw_name, Path(account.pw_dir))
    layout.credentials_root.mkdir(parents=True, mode=0o700)
    layout.credentials_root.chmod(0o700)
    layout.projects_root.mkdir(parents=True, mode=0o700)
    layout.projects_root.chmod(0o700)
    machine = ProjectProvisioningMachineIntent.model_construct(
        alias="checkout-local",
        location="local",
        host="",
        os_account=account.pw_name,
        central_root=str(layout.projects_root),
    )
    runner = _CurrentAccountRunner(account.pw_name)

    _drive_checkout(
        token,
        repository,
        layout=layout,
        machine=machine,
        runner=runner,
        space_id=_LOCAL_SPACE_ID,
        project_id=_LOCAL_PROJECT_ID,
        request_id=_LOCAL_REQUEST_ID,
        alias="checkout-local",
    )


@_LIVE_ONLY
def test_ssh_central_checkout_with_remote_account_key(tmp_path: Path) -> None:
    token, repository = _live_inputs()
    host = os.environ.get(_SSH_HOST, "")
    remote_account = os.environ.get(_SSH_ACCOUNT, "")
    if not host or not remote_account:
        pytest.fail(f"set {_SSH_HOST} and {_SSH_ACCOUNT} for the reachable SSH qualification")
    local_account = pwd.getpwuid(os.getuid())
    layout = _live_layout(tmp_path, local_account.pw_name, Path(local_account.pw_dir))
    runner = _CurrentAccountRunner(local_account.pw_name)
    remote_root = f"/tmp/rcp-project-checkout-live-{_REMOTE_PROJECT_ID}"
    machine = ProjectProvisioningMachineIntent(
        alias="checkout-ssh",
        location="ssh",
        host=host,
        os_account=remote_account,
        central_root=remote_root,
    )
    owns_root = False
    try:
        _remote_test_root(runner, layout, machine, operation="create")
        owns_root = True
        _drive_remote_checkout(
            token,
            repository,
            layout=layout,
            machine=machine,
            runner=runner,
            space_id=_REMOTE_SPACE_ID,
            project_id=_REMOTE_PROJECT_ID,
            alias="checkout-ssh",
        )
    finally:
        if owns_root:
            _remote_test_root(runner, layout, machine, operation="remove")


class _StaticCredentialManager:
    def __init__(self, expected: DeployKeyMaterial) -> None:
        self.expected = expected

    def inspect_key(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
    ) -> DeployKeyMaterial:
        assert machine.alias == material.machine_alias
        assert material == self.expected
        return material


def _live_inputs() -> tuple[str, GitHubRepositoryRef]:
    if os.environ.get(_DISPOSABLE_CONFIRMATION) != _EXPECTED_CONFIRMATION:
        pytest.fail(
            f"set {_DISPOSABLE_CONFIRMATION}={_EXPECTED_CONFIRMATION} only for the disposable "
            "checkout repository"
        )
    token = os.environ.get(_TOKEN, "")
    if not token or "\n" in token or "\r" in token:
        pytest.fail(f"{_TOKEN} must contain one GitHub repository-administration token")
    try:
        repository = GitHubRepositoryRef(identity=os.environ.get(_REPOSITORY, ""))
    except ValueError as exc:
        pytest.fail(f"{_REPOSITORY} must contain one canonical owner/repository identity: {exc}")
    if "disposable" not in repository.repository and "live-test" not in repository.repository:
        pytest.fail(f"{_REPOSITORY} must visibly name a disposable or live-test repository")
    return token, repository


def _drive_checkout(
    token: str,
    repository: GitHubRepositoryRef,
    *,
    layout,
    machine: ProjectProvisioningMachineIntent,
    runner: _CurrentAccountRunner,
    space_id: str,
    project_id: str,
    request_id: str,
    alias: str,
) -> None:
    credentials = GitCredentialManager(layout, runner=runner)
    checkouts = ProjectCheckoutManager(
        layout,
        runner=runner,
        credential_manager=credentials,
    )
    label = f"rcp:{space_id}:{project_id}:{alias}"
    temporary_branch = f"rcp-provisioning-{request_id}"
    key_id: int | None = None
    material = None
    owns_ref_boundary = False
    existing_keys = _github_request(
        token,
        method="GET",
        path=f"/repos/{repository.identity}/keys?per_page=100",
    )
    assert isinstance(existing_keys, list)
    if any(isinstance(item, dict) and item.get("title") == label for item in existing_keys):
        pytest.fail("the disposable repository already has this checkout qualification key")
    if (
        _github_optional_request(
            token,
            path=f"/repos/{repository.identity}/git/ref/heads/{temporary_branch}",
        )
        is not None
    ):
        pytest.fail("the disposable repository already has this checkout qualification ref")
    owns_ref_boundary = True
    try:
        material = credentials.prepare_key(
            machine,
            repository,
            space_id=space_id,
            project_id=project_id,
            repository_alias=alias,
        )
        if machine.location == "ssh":
            _require_remote_published_github_host_key(
                runner, layout, machine, material.account_home
            )
        created = _github_request(
            token,
            method="POST",
            path=f"/repos/{repository.identity}/keys",
            body={"title": label, "key": material.public_key, "read_only": False},
        )
        if not isinstance(created, dict) or not isinstance(created.get("id"), int):
            pytest.fail("GitHub did not return the created checkout deploy key")
        if created.get("read_only") is not False:
            pytest.fail("GitHub did not confirm write access for the checkout deploy key")
        key_id = created["id"]
        probe = credentials.probe_write(machine, material, request_id=request_id)
        assert probe.ready, probe.diagnostic
        assert probe.commit is not None

        first = checkouts.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=project_id,
            repository_alias=alias,
            state_repository=True,
            expected_commit=probe.commit,
        )
        repeated = checkouts.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=project_id,
            repository_alias=alias,
            state_repository=True,
            expected_commit=probe.commit,
        )
        assert first.checkout_disposition == "request_created"
        assert repeated.checkout_disposition == "reused_existing"
        assert first.repository_path == repeated.repository_path
        assert first.commit == repeated.commit == probe.commit
        assert first.retained_research.retained is False
    finally:
        try:
            try:
                if owns_ref_boundary:
                    lingering = _github_optional_request(
                        token,
                        path=f"/repos/{repository.identity}/git/ref/heads/{temporary_branch}",
                    )
                    if lingering is not None:
                        _github_request(
                            token,
                            method="DELETE",
                            path=f"/repos/{repository.identity}/git/refs/heads/{temporary_branch}",
                        )
            finally:
                cleanup_ids = (
                    [key_id]
                    if key_id is not None
                    else _matching_deploy_key_ids(
                        token,
                        repository,
                        label=label,
                        public_key=material.public_key if material is not None else "",
                    )
                )
                for cleanup_id in cleanup_ids:
                    _github_request(
                        token,
                        method="DELETE",
                        path=f"/repos/{repository.identity}/keys/{cleanup_id}",
                    )
        finally:
            if material is not None:
                assert credentials.remove_key(machine, material) is True
    final_keys = _github_request(
        token,
        method="GET",
        path=f"/repos/{repository.identity}/keys?per_page=100",
    )
    assert isinstance(final_keys, list)
    assert all(not isinstance(item, dict) or item.get("title") != label for item in final_keys)
    assert (
        _github_optional_request(
            token,
            path=f"/repos/{repository.identity}/git/ref/heads/{temporary_branch}",
        )
        is None
    )


_REMOTE_KEY_SCRIPT = r"""
import base64
import hashlib
import json
import os
import pwd
import re
import stat
import subprocess
import sys

operation, expected_account, root, label, expected_fingerprint = sys.argv[1:]
account = pwd.getpwuid(os.getuid())
if account.pw_name != expected_account:
    raise SystemExit(2)
if re.fullmatch(r"/tmp/rcp-project-checkout-live-key-[0-9a-f-]{36}", root) is None:
    raise SystemExit(2)
private = os.path.join(root, "id_ed25519")
public = private + ".pub"
if operation == "create":
    if os.path.lexists(root):
        raise SystemExit(2)
    os.mkdir(root, 0o700)
    generated = subprocess.run(
        ("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", label, "-f", private),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if generated.returncode != 0:
        raise SystemExit(2)
    os.chmod(private, 0o600)
    os.chmod(public, 0o644)
elif operation != "remove":
    raise SystemExit(2)
root_info = os.lstat(root)
private_info = os.lstat(private)
public_info = os.lstat(public)
if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.getuid():
    raise SystemExit(2)
if stat.S_IMODE(root_info.st_mode) != 0o700:
    raise SystemExit(2)
if not stat.S_ISREG(private_info.st_mode) or stat.S_IMODE(private_info.st_mode) != 0o600:
    raise SystemExit(2)
if not stat.S_ISREG(public_info.st_mode) or stat.S_IMODE(public_info.st_mode) != 0o644:
    raise SystemExit(2)
public_key = open(public, encoding="utf-8").read().strip()
parts = public_key.split()
if len(parts) != 3 or parts[0] != "ssh-ed25519" or parts[2] != label:
    raise SystemExit(2)
blob = base64.b64decode(parts[1], validate=True)
fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
if operation == "remove":
    if fingerprint != expected_fingerprint or sorted(os.listdir(root)) != ["id_ed25519", "id_ed25519.pub"]:
        raise SystemExit(2)
    os.unlink(public)
    os.unlink(private)
    os.rmdir(root)
    print(json.dumps({"removed": True}))
else:
    print(json.dumps({
        "account": account.pw_name,
        "home": account.pw_dir,
        "private_key_path": private,
        "public_key": public_key,
        "public_key_fingerprint": fingerprint,
    }, sort_keys=True, separators=(",", ":")))
""".strip()


def _remote_test_key(
    runner: _CurrentAccountRunner,
    layout,
    machine: ProjectProvisioningMachineIntent,
    *,
    repository: GitHubRepositoryRef,
    space_id: str,
    project_id: str,
    alias: str,
) -> DeployKeyMaterial:
    assert machine.central_root is not None
    key_root = f"/tmp/rcp-project-checkout-live-key-{project_id}"
    label = f"rcp:{space_id}:{project_id}:{alias}"
    result = runner(
        target_account_argv(
            layout,
            machine,
            (
                "python3",
                "-c",
                _REMOTE_KEY_SCRIPT,
                "create",
                machine.os_account,
                key_root,
                label,
                "-",
            ),
        ),
        timeout=30,
    )
    if result.returncode != 0 or len(result.stdout.encode("utf-8")) > _MAX_REMOTE_OUTPUT_BYTES:
        pytest.fail("the disposable remote checkout key could not be created safely")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        pytest.fail("the disposable remote checkout key returned invalid JSON")
    if not isinstance(payload, dict) or set(payload) != {
        "account",
        "home",
        "private_key_path",
        "public_key",
        "public_key_fingerprint",
    }:
        pytest.fail("the disposable remote checkout key returned an invalid receipt")
    if (
        payload["account"] != machine.os_account
        or payload["private_key_path"] != f"{key_root}/id_ed25519"
        or not isinstance(payload["home"], str)
        or not isinstance(payload["public_key"], str)
        or not isinstance(payload["public_key_fingerprint"], str)
        or not payload["public_key"].endswith(f" {label}")
        or not payload["public_key_fingerprint"].startswith("SHA256:")
    ):
        pytest.fail("the disposable remote checkout key receipt did not match its exact target")
    return DeployKeyMaterial(
        space_id=space_id,
        project_id=project_id,
        repository_alias=alias,
        repository=repository,
        machine_alias=machine.alias,
        location=machine.location,
        host=machine.host,
        os_account=machine.os_account,
        central_root=machine.central_root,
        account_home=payload["home"],
        credentials_root=key_root,
        private_key_path=payload["private_key_path"],
        label=label,
        public_key=payload["public_key"],
        public_key_fingerprint=payload["public_key_fingerprint"],
        created=True,
    )


def _remote_test_key_remove(
    runner: _CurrentAccountRunner,
    layout,
    machine: ProjectProvisioningMachineIntent,
    material: DeployKeyMaterial,
) -> None:
    result = runner(
        target_account_argv(
            layout,
            machine,
            (
                "python3",
                "-c",
                _REMOTE_KEY_SCRIPT,
                "remove",
                machine.os_account,
                material.credentials_root,
                material.label,
                material.public_key_fingerprint,
            ),
        ),
        timeout=30,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"removed": True}


def _drive_remote_checkout(
    token: str,
    repository: GitHubRepositoryRef,
    *,
    layout,
    machine: ProjectProvisioningMachineIntent,
    runner: _CurrentAccountRunner,
    space_id: str,
    project_id: str,
    alias: str,
) -> None:
    label = f"rcp:{space_id}:{project_id}:{alias}"
    existing_keys = _github_request(
        token,
        method="GET",
        path=f"/repos/{repository.identity}/keys?per_page=100",
    )
    assert isinstance(existing_keys, list)
    if any(isinstance(item, dict) and item.get("title") == label for item in existing_keys):
        pytest.fail("the disposable repository already has this remote checkout key")
    material = _remote_test_key(
        runner,
        layout,
        machine,
        repository=repository,
        space_id=space_id,
        project_id=project_id,
        alias=alias,
    )
    key_id: int | None = None
    try:
        _require_remote_published_github_host_key(
            runner,
            layout,
            machine,
            material.account_home,
        )
        created = _github_request(
            token,
            method="POST",
            path=f"/repos/{repository.identity}/keys",
            body={"title": label, "key": material.public_key, "read_only": False},
        )
        if not isinstance(created, dict) or not isinstance(created.get("id"), int):
            pytest.fail("GitHub did not return the remote checkout deploy key")
        if created.get("read_only") is not False:
            pytest.fail("GitHub did not confirm write capability for the remote checkout key")
        key_id = created["id"]
        manager = ProjectCheckoutManager(
            layout,
            runner=runner,
            credential_manager=_StaticCredentialManager(material),  # type: ignore[arg-type]
        )
        first = manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=project_id,
            repository_alias=alias,
            state_repository=True,
        )
        repeated = manager.prepare(
            machine,
            material,
            request_kind="create_team_project",
            project_id=project_id,
            repository_alias=alias,
            state_repository=True,
        )
        assert first.checkout_disposition == "request_created"
        assert repeated.checkout_disposition == "reused_existing"
        assert first.repository_path == repeated.repository_path
        assert first.commit == repeated.commit
        assert first.retained_research.retained is False
    finally:
        try:
            cleanup_ids = (
                [key_id]
                if key_id is not None
                else _matching_deploy_key_ids(
                    token,
                    repository,
                    label=label,
                    public_key=material.public_key,
                )
            )
            for cleanup_id in cleanup_ids:
                _github_request(
                    token,
                    method="DELETE",
                    path=f"/repos/{repository.identity}/keys/{cleanup_id}",
                )
        finally:
            _remote_test_key_remove(runner, layout, machine, material)
    final_keys = _github_request(
        token,
        method="GET",
        path=f"/repos/{repository.identity}/keys?per_page=100",
    )
    assert isinstance(final_keys, list)
    assert all(not isinstance(item, dict) or item.get("title") != label for item in final_keys)


def _require_remote_published_github_host_key(
    runner: _CurrentAccountRunner,
    layout,
    machine: ProjectProvisioningMachineIntent,
    account_home: str,
) -> None:
    command = (
        "ssh-keygen",
        "-F",
        "github.com",
        "-f",
        str(Path(account_home) / ".ssh" / "known_hosts"),
    )
    result = runner(target_account_argv(layout, machine, command), timeout=30)
    if result.returncode != 0 or len(result.stdout.encode("utf-8")) > _MAX_REMOTE_OUTPUT_BYTES:
        pytest.fail("the remote checkout account has no bounded GitHub known-host record")
    fingerprints: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[1] != "ssh-ed25519":
            continue
        try:
            blob = base64.b64decode(parts[2], validate=True)
        except ValueError:
            continue
        fingerprints.add(
            "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii").rstrip("=")
        )
    if _GITHUB_ED25519_FINGERPRINT not in fingerprints:
        pytest.fail("the remote checkout account lacks GitHub's published Ed25519 host key")


_REMOTE_ROOT_SCRIPT = r"""
import os
import pwd
import re
import shutil
import stat
import sys

operation, expected_account, root, project_id = sys.argv[1:]
account = pwd.getpwuid(os.getuid())
if account.pw_name != expected_account:
    raise SystemExit(2)
if re.fullmatch(r"/tmp/rcp-project-checkout-live-[0-9a-f-]{36}", root) is None:
    raise SystemExit(2)
if not root.endswith(project_id):
    raise SystemExit(2)
if operation == "create":
    if os.path.lexists(root):
        raise SystemExit(2)
    os.mkdir(root, 0o700)
elif operation == "remove":
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise SystemExit(2)
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise SystemExit(2)
    root_entries = sorted(os.listdir(root))
    if root_entries not in ([], [project_id]):
        raise SystemExit(2)
    if root_entries:
        project = os.path.join(root, project_id)
        if sorted(os.listdir(project)) != ["repositories"]:
            raise SystemExit(2)
    shutil.rmtree(root)
else:
    raise SystemExit(2)
""".strip()


def _remote_test_root(
    runner: _CurrentAccountRunner,
    layout,
    machine: ProjectProvisioningMachineIntent,
    *,
    operation: str,
) -> None:
    assert machine.central_root is not None
    result = runner(
        target_account_argv(
            layout,
            machine,
            (
                "python3",
                "-c",
                _REMOTE_ROOT_SCRIPT,
                operation,
                machine.os_account,
                machine.central_root,
                _REMOTE_PROJECT_ID,
            ),
        ),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
