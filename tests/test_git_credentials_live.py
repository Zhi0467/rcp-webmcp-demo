"""Explicitly gated GitHub qualification for repository-scoped write keys."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pwd
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from rcp.server_ops.git_credentials import GitCredentialManager, _run_process
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.layout import ServerLayout
from rcp.storage import ProjectProvisioningMachineIntent

_LIVE_GATE = "RCP_RUN_GIT_CREDENTIALS_LIVE"
_DISPOSABLE_CONFIRMATION = "RCP_GIT_CREDENTIALS_LIVE_DISPOSABLE"
_EXPECTED_CONFIRMATION = "I_UNDERSTAND_THIS_ADDS_AND_REMOVES_ONE_DEPLOY_KEY_AND_REF"
_TOKEN = "RCP_LIVE_GITHUB_ADMIN_TOKEN"
_REPOSITORY = "RCP_LIVE_GITHUB_REPOSITORY"
_SPACE_ID = "d576f732-eeb9-4392-a8c8-33fe6ca0cba4"
_PROJECT_ID = "68bf8a35-7bd1-4eea-b01e-2dc7539cf954"
_REQUEST_ID = "3db7b74c-62b8-4c59-95fe-6479391e4687"
_ALIAS = "live"
_GITHUB_ED25519_FINGERPRINT = "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU"
_MAX_API_BYTES = 64 * 1024

_LIVE_ONLY = pytest.mark.skipif(
    os.environ.get(_LIVE_GATE) != "1",
    reason="destructive disposable-repository Git credential qualification is disabled",
)


@_LIVE_ONLY
def test_repository_scoped_write_key_on_disposable_github_repository(
    tmp_path: Path,
) -> None:
    """Add one write key, prove push/readback/delete, then revoke it exactly."""

    if os.environ.get(_DISPOSABLE_CONFIRMATION) != _EXPECTED_CONFIRMATION:
        pytest.fail(
            f"set {_DISPOSABLE_CONFIRMATION}={_EXPECTED_CONFIRMATION} only for one disposable "
            "GitHub repository"
        )
    token = os.environ.get(_TOKEN, "")
    if not token or "\n" in token or "\r" in token:
        pytest.fail(f"{_TOKEN} must contain one GitHub repository-administration token")
    try:
        repository = GitHubRepositoryRef(identity=os.environ.get(_REPOSITORY, ""))
    except ValueError as exc:
        pytest.fail(f"{_REPOSITORY} must contain one canonical owner/repository identity: {exc}")
    if "disposable" not in repository.repository and "live-test" not in repository.repository:
        pytest.fail(
            f"{_REPOSITORY} must visibly name a disposable or live-test repository; "
            "the qualification refuses ordinary repositories"
        )

    account = pwd.getpwuid(os.getuid())
    known_hosts = Path(account.pw_dir) / ".ssh" / "known_hosts"
    _require_published_github_host_key(known_hosts)
    layout = _live_layout(tmp_path, account.pw_name, Path(account.pw_dir))
    layout.credentials_root.mkdir(parents=True, mode=0o700)
    layout.credentials_root.chmod(0o700)
    machine = ProjectProvisioningMachineIntent.model_construct(
        alias="live",
        location="local",
        host="",
        os_account=account.pw_name,
        central_root=str(layout.projects_root),
    )
    manager = GitCredentialManager(
        layout,
        runner=_CurrentAccountRunner(account.pw_name),
    )
    label = f"rcp:{_SPACE_ID}:{_PROJECT_ID}:{_ALIAS}"
    temporary_branch = f"rcp-provisioning-{_REQUEST_ID}"
    deploy_key_id: int | None = None
    key_material = None
    owns_ref_boundary = False

    existing_keys = _github_request(
        token,
        method="GET",
        path=f"/repos/{repository.identity}/keys?per_page=100",
    )
    if not isinstance(existing_keys, list):
        pytest.fail("GitHub did not return the repository deploy-key inventory")
    if any(isinstance(item, dict) and item.get("title") == label for item in existing_keys):
        pytest.fail("the disposable repository already has this qualification's deploy-key label")
    existing_ref = _github_optional_request(
        token,
        path=f"/repos/{repository.identity}/git/ref/heads/{temporary_branch}",
    )
    if existing_ref is not None:
        pytest.fail("the disposable repository already has this qualification's temporary ref")
    owns_ref_boundary = True

    try:
        key_material = manager.prepare_key(
            machine,
            repository,
            space_id=_SPACE_ID,
            project_id=_PROJECT_ID,
            repository_alias=_ALIAS,
        )
        created = _github_request(
            token,
            method="POST",
            path=f"/repos/{repository.identity}/keys",
            body={"title": label, "key": key_material.public_key, "read_only": False},
        )
        if not isinstance(created, dict):
            pytest.fail("GitHub did not return the created deploy key")
        observed_id = created.get("id")
        if not isinstance(observed_id, int) or created.get("read_only") is not False:
            pytest.fail("GitHub did not confirm one repository-scoped write deploy key")
        deploy_key_id = observed_id

        probe = manager.probe_write(machine, key_material, request_id=_REQUEST_ID)

        assert probe.ready, probe.diagnostic
        assert probe.temporary_ref is None
        assert (
            _github_optional_request(
                token,
                path=f"/repos/{repository.identity}/git/ref/heads/{temporary_branch}",
            )
            is None
        )
    finally:
        try:
            try:
                if owns_ref_boundary:
                    lingering_ref = _github_optional_request(
                        token,
                        path=f"/repos/{repository.identity}/git/ref/heads/{temporary_branch}",
                    )
                    if lingering_ref is not None:
                        _github_request(
                            token,
                            method="DELETE",
                            path=(
                                f"/repos/{repository.identity}/git/refs/heads/{temporary_branch}"
                            ),
                        )
            finally:
                cleanup_key_ids = (
                    [deploy_key_id]
                    if deploy_key_id is not None
                    else _matching_deploy_key_ids(
                        token,
                        repository,
                        label=label,
                        public_key=key_material.public_key if key_material is not None else "",
                    )
                )
                for cleanup_key_id in cleanup_key_ids:
                    _github_request(
                        token,
                        method="DELETE",
                        path=f"/repos/{repository.identity}/keys/{cleanup_key_id}",
                    )
        finally:
            if key_material is not None:
                assert manager.remove_key(machine, key_material) is True

    final_keys = _github_request(
        token,
        method="GET",
        path=f"/repos/{repository.identity}/keys?per_page=100",
    )
    assert isinstance(final_keys, list)
    assert all(not isinstance(item, dict) or item.get("title") != label for item in final_keys)


class _CurrentAccountRunner:
    """Execute the production inner command without Linux runuser on the dev host."""

    def __init__(self, account: str) -> None:
        self.account = account

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        expected = ("runuser", "--user", self.account, "--")
        if argv[:4] != expected:
            raise AssertionError("the live drive escaped the exact account boundary")
        return _run_process(tuple(argv[4:]), timeout=timeout)


def _live_layout(tmp_path: Path, account: str, home: Path) -> ServerLayout:
    root = tmp_path / "rcp-server"
    return ServerLayout(
        service_account=account,
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


def _require_published_github_host_key(known_hosts: Path) -> None:
    try:
        lines = known_hosts.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        pytest.fail(f"the checkout account has no readable GitHub known-hosts file: {exc}")
    fingerprints: set[str] = set()
    for line in lines:
        parts = line.split()
        if len(parts) < 3 or parts[1] != "ssh-ed25519":
            continue
        hosts = parts[0].split(",")
        if "github.com" not in hosts:
            continue
        try:
            key_blob = base64.b64decode(parts[2], validate=True)
        except ValueError:
            continue
        fingerprints.add(
            "SHA256:"
            + base64.b64encode(hashlib.sha256(key_blob).digest()).decode("ascii").rstrip("=")
        )
    if _GITHUB_ED25519_FINGERPRINT not in fingerprints:
        pytest.fail(
            "the checkout account has not explicitly trusted GitHub's published Ed25519 host key"
        )


def _matching_deploy_key_ids(
    token: str,
    repository: GitHubRepositoryRef,
    *,
    label: str,
    public_key: str,
) -> list[int]:
    if not public_key:
        return []
    inventory = _github_request(
        token,
        method="GET",
        path=f"/repos/{repository.identity}/keys?per_page=100",
    )
    if not isinstance(inventory, list):
        pytest.fail("GitHub did not return the deploy-key inventory needed for cleanup")
    expected_identity = public_key.split()[:2]
    matches: list[int] = []
    for item in inventory:
        if not isinstance(item, dict) or item.get("title") != label:
            continue
        observed_key = item.get("key")
        observed_id = item.get("id")
        if not isinstance(observed_key, str) or observed_key.split()[:2] != expected_identity:
            continue
        if not isinstance(observed_id, int):
            pytest.fail("GitHub returned an invalid exact deploy-key cleanup identity")
        matches.append(observed_id)
    return matches


def test_ambiguous_creation_cleanup_matches_only_the_exact_public_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = f"rcp:{_SPACE_ID}:{_PROJECT_ID}:{_ALIAS}"
    public_key = "ssh-ed25519 AAAAexact rcp-test"
    monkeypatch.setitem(
        globals(),
        "_github_request",
        lambda *_args, **_kwargs: [
            {"id": 1, "title": "another-label", "key": public_key},
            {"id": 2, "title": label, "key": "ssh-ed25519 AAAAother rcp-test"},
            {"id": 3, "title": label, "key": "ssh-ed25519 AAAAexact github-normalized"},
        ],
    )

    assert _matching_deploy_key_ids(
        "redacted-token",
        GitHubRepositoryRef(identity="owner/disposable-repository"),
        label=label,
        public_key=public_key,
    ) == [3]


def _github_optional_request(token: str, *, path: str) -> object | None:
    try:
        return _github_request(token, method="GET", path=path)
    except _GitHubNotFound:
        return None


class _GitHubNotFound(RuntimeError):
    pass


def _github_request(
    token: str,
    *,
    method: str,
    path: str,
    body: dict[str, object] | None = None,
) -> object:
    safe_path = urllib.parse.quote(path, safe="/?=&")
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.github.com{safe_path}",
        method=method,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "rcp-git-credentials-live-test",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read(_MAX_API_BYTES + 1)
            status = response.status
    except urllib.error.HTTPError as exc:
        if method == "GET" and exc.code == 404:
            raise _GitHubNotFound from None
        pytest.fail(f"GitHub repository API returned HTTP {exc.code} for {method}")
    except urllib.error.URLError:
        pytest.fail("GitHub repository API was unreachable")
    if len(content) > _MAX_API_BYTES:
        pytest.fail("GitHub repository API response exceeded the live-test bound")
    if method == "DELETE":
        if status != 204:
            pytest.fail("GitHub did not confirm exact cleanup")
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pytest.fail("GitHub repository API returned invalid JSON")
