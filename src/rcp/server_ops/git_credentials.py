"""Repository-scoped GitHub deploy keys for team-owned checkouts."""

from __future__ import annotations

import contextlib
import importlib.resources
import json
import os
import re
import selectors
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Literal, Protocol

from rcp.limits import (
    SERVER_GIT_CREDENTIAL_TIMEOUT_SECONDS,
    SERVER_GIT_PROBE_TIMEOUT_SECONDS,
)
from rcp.server_ops.github import GitHubRepositoryRef
from rcp.server_ops.layout import (
    DEFAULT_SERVER_LAYOUT,
    ServerLayout,
    remote_project_deploy_key_path,
    remote_projects_root,
)
from rcp.server_ops.models import (
    CommandAction,
    ExternalAction,
    ExternalServiceTarget,
    NonsecretField,
    ServerStep,
)
from rcp.storage import ProjectProvisioningMachineIntent
from rcp.transport.ssh import SSH_OPTIONS

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{43}")
_MAX_HELPER_OUTPUT_BYTES = 64 * 1024
_MAX_GIT_OUTPUT_BYTES = 256 * 1024
_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_GITHUB_FINGERPRINTS_URL = (
    "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/"
    "githubs-ssh-key-fingerprints"
)

GitWriteProbeStatus = Literal[
    "ready",
    "github_host_trust_needed",
    "github_grant_needed",
    "empty_repository",
    "cleanup_failed",
    "temporary_ref_conflict",
    "unavailable",
    "failed",
]

_HOST_TRUST_MARKERS = (
    "host key verification failed",
    "remote host identification has changed",
    "no ed25519 host key is known",
    "no rsa host key is known",
)
_AUTH_MARKERS = (
    "permission denied (publickey)",
    "repository not found",
    "could not read from remote repository",
    "denied to deploy key",
    "write access to repository not granted",
)
_NETWORK_MARKERS = (
    "could not resolve hostname",
    "connection timed out",
    "connection refused",
    "network is unreachable",
    "connection closed by remote host",
)
_LEASE_REJECTION_MARKERS = (
    "stale info",
    "fetch first",
)


class GitCredentialRefused(RuntimeError):
    """A deploy-key effect could not be proven safe or complete."""


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class DeployKeyMaterial:
    space_id: str
    project_id: str
    repository_alias: str
    repository: GitHubRepositoryRef
    machine_alias: str
    location: Literal["local", "ssh"]
    host: str
    os_account: str
    central_root: str
    account_home: str
    credentials_root: str
    private_key_path: str
    label: str
    public_key: str
    public_key_fingerprint: str
    created: bool


@dataclass(frozen=True)
class GitWriteProbe:
    status: GitWriteProbeStatus
    commit: str | None
    temporary_ref: str | None
    diagnostic: str

    @property
    def ready(self) -> bool:
        return self.status == "ready"


class GitCredentialManager:
    """Run every key and Git operation on the checkout's exact OS account."""

    def __init__(
        self,
        layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.layout = layout
        self._runner = runner or partial(_run_process, cwd=self.layout.service_home)

    def prepare_key(
        self,
        machine: ProjectProvisioningMachineIntent,
        repository: GitHubRepositoryRef,
        *,
        space_id: str,
        project_id: str,
        repository_alias: str,
    ) -> DeployKeyMaterial:
        payload = self._helper(
            machine,
            (
                "prepare",
                machine.os_account,
                machine.location,
                str(self.layout.credentials_root) if machine.location == "local" else "-",
                machine.central_root or "-",
                space_id,
                project_id,
                repository_alias,
            ),
        )
        return self._material(
            payload,
            machine,
            repository,
            space_id=space_id,
            project_id=project_id,
            repository_alias=repository_alias,
        )

    def preflight_recovery_key(
        self,
        machine: ProjectProvisioningMachineIntent,
        *,
        space_id: str,
        project_id: str,
        repository_alias: str,
    ) -> None:
        """Prove a replacement restore will not adopt an old deterministic key."""

        payload = self._helper(
            machine,
            (
                "recovery-preflight",
                machine.os_account,
                machine.location,
                str(self.layout.credentials_root) if machine.location == "local" else "-",
                machine.central_root or "-",
                space_id,
                project_id,
                repository_alias,
            ),
        )
        expected_keys = {
            "account",
            "home",
            "credentials_root",
            "private_key_path",
            "label",
            "absent",
        }
        if set(payload) != expected_keys or not isinstance(payload["absent"], bool):
            raise GitCredentialRefused(
                "The recovery deploy-key preflight returned an invalid receipt."
            )
        if not all(isinstance(payload[name], str) for name in expected_keys if name != "absent"):
            raise GitCredentialRefused("The recovery deploy-key preflight returned invalid fields.")
        home = str(payload["home"])
        expected_root, expected_private = self._expected_key_paths(
            machine,
            home=home,
            project_id=project_id,
            repository_alias=repository_alias,
        )
        label = f"rcp:{space_id}:{project_id}:{repository_alias}"
        if (
            payload["account"] != machine.os_account
            or payload["credentials_root"] != expected_root
            or payload["private_key_path"] != expected_private
            or payload["label"] != label
        ):
            raise GitCredentialRefused(
                "The recovery deploy-key preflight names another execution target."
            )
        if not payload["absent"]:
            raise GitCredentialRefused(
                "A deterministic deploy-key path already exists before this restore recorded "
                "key generation. Preserve and inspect it; RCP will not adopt it as a fresh key."
            )

    def prepare_recovery_key(
        self,
        machine: ProjectProvisioningMachineIntent,
        repository: GitHubRepositoryRef,
        *,
        space_id: str,
        project_id: str,
        repository_alias: str,
    ) -> DeployKeyMaterial:
        """Create or re-read the key after restore recorded key generation."""

        return self.prepare_key(
            machine,
            repository,
            space_id=space_id,
            project_id=project_id,
            repository_alias=repository_alias,
        )

    def inspect_key(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
    ) -> DeployKeyMaterial:
        self._require_material_target(machine, material)
        payload = self._helper(
            machine,
            (
                "inspect",
                machine.os_account,
                machine.location,
                str(self.layout.credentials_root) if machine.location == "local" else "-",
                machine.central_root or "-",
                material.space_id,
                material.project_id,
                material.repository_alias,
            ),
        )
        observed = self._material(
            payload,
            machine,
            material.repository,
            space_id=material.space_id,
            project_id=material.project_id,
            repository_alias=material.repository_alias,
        )
        if (
            observed.private_key_path != material.private_key_path
            or observed.public_key != material.public_key
            or observed.public_key_fingerprint != material.public_key_fingerprint
            or observed.label != material.label
        ):
            raise GitCredentialRefused(
                "The prepared deploy key changed. Preserve the exact path and inspect it."
            )
        return observed

    def remove_key(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
    ) -> bool:
        self._require_material_target(machine, material)
        payload = self._helper(
            machine,
            (
                "remove",
                machine.os_account,
                machine.location,
                str(self.layout.credentials_root) if machine.location == "local" else "-",
                machine.central_root or "-",
                material.space_id,
                material.project_id,
                material.repository_alias,
                material.public_key_fingerprint,
            ),
        )
        if set(payload) != {"removed"} or not isinstance(payload["removed"], bool):
            raise GitCredentialRefused("The deploy-key cleanup returned an invalid receipt.")
        return payload["removed"]

    def probe_write(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
        *,
        request_id: str,
    ) -> GitWriteProbe:
        material = self.inspect_key(machine, material)
        prepared = self._helper(
            machine,
            ("probe-prepare", machine.os_account, request_id),
        )
        if set(prepared) != {"probe_directory"} or not isinstance(prepared["probe_directory"], str):
            raise GitCredentialRefused("The Git write-probe directory receipt is invalid.")
        probe_directory = prepared["probe_directory"]
        probe: GitWriteProbe | None = None
        error: Exception | None = None
        try:
            probe = self._probe_in_directory(
                machine,
                material,
                request_id=request_id,
                probe_directory=probe_directory,
            )
        except Exception as exc:
            error = exc
        cleanup_failed = False
        try:
            cleaned = self._helper(
                machine,
                ("probe-cleanup", machine.os_account, request_id, probe_directory),
            )
        except GitCredentialRefused:
            cleaned = None
            cleanup_failed = True
        if cleanup_failed:
            if probe is not None and probe.status == "cleanup_failed":
                return GitWriteProbe(
                    status=probe.status,
                    commit=probe.commit,
                    temporary_ref=probe.temporary_ref,
                    diagnostic=(
                        f"{probe.diagnostic} The request-owned local Git probe directory "
                        f"{probe_directory!r} also could not be removed."
                    ),
                )
            if error is not None:
                raise GitCredentialRefused(
                    "The Git write probe failed and its request-owned local directory "
                    f"{probe_directory!r} could not be removed."
                ) from error
            return GitWriteProbe(
                status="failed",
                commit=probe.commit if probe is not None else None,
                temporary_ref=probe.temporary_ref if probe is not None else None,
                diagnostic=(
                    "The request-owned local Git probe directory could not be removed. "
                    "Inspect that exact directory before resuming."
                ),
            )
        if cleaned != {"removed": True}:
            raise GitCredentialRefused("The Git write-probe cleanup receipt is invalid.")
        if error is not None:
            raise error
        assert probe is not None
        return probe

    def github_trust_argv(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
    ) -> tuple[str, ...]:
        self._require_material_target(machine, material)
        inner = (
            "ssh",
            "-F",
            "/dev/null",
            "-i",
            material.private_key_path,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=ask",
            "-o",
            "GlobalKnownHostsFile=/etc/ssh/ssh_known_hosts",
            "-o",
            f"UserKnownHostsFile={Path(material.account_home) / '.ssh' / 'known_hosts'}",
            "-T",
            "git@github.com",
        )
        if machine.location == "local":
            return ("sudo", "-n", "-u", self.layout.service_account, "-H", *inner)
        return (
            "sudo",
            "-n",
            "-u",
            self.layout.service_account,
            "-H",
            *_strict_ssh_arguments(machine.host, shlex.join(inner)),
        )

    def _probe_in_directory(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
        *,
        request_id: str,
        probe_directory: str,
    ) -> GitWriteProbe:
        origin = material.repository.ssh_clone_url
        advertised = self._git(
            machine,
            material,
            ("git", "ls-remote", origin, "HEAD"),
        )
        if advertised.returncode != 0:
            return _probe_failure(
                advertised,
                operation="read",
                stage="advertised HEAD lookup",
            )
        refs = _parse_remote_refs(advertised.stdout)
        if not refs:
            return GitWriteProbe(
                status="empty_repository",
                commit=None,
                temporary_ref=None,
                diagnostic=(
                    "The GitHub repository has no commit. Push the local code through the "
                    "ordinary human Git workflow, then resume the same provisioning request."
                ),
            )
        source_ref, advertised_commit = _preferred_source_ref(refs)
        temporary_ref = f"refs/heads/rcp-provisioning-{request_id}"
        existing = self._git(
            machine,
            material,
            ("git", "ls-remote", origin, temporary_ref),
        )
        if existing.returncode != 0:
            return _probe_failure(
                existing,
                operation="read",
                stage="temporary-ref absence check",
            )
        if _parse_remote_refs(existing.stdout):
            return GitWriteProbe(
                status="temporary_ref_conflict",
                commit=advertised_commit,
                temporary_ref=temporary_ref,
                diagnostic=(
                    "The request-scoped Git write-probe ref already exists. RCP did not alter or "
                    "delete it; inspect that exact ref before resuming."
                ),
            )

        git_dir = str(Path(probe_directory) / "repository.git")
        initialized = self._git(
            machine,
            material,
            ("git", "init", "--quiet", "--bare", "--template=", git_dir),
        )
        if initialized.returncode != 0:
            return _probe_failure(
                initialized,
                operation="local",
                stage="local bare-repository initialization",
            )
        fetched = self._git(
            machine,
            material,
            (
                "git",
                f"--git-dir={git_dir}",
                "fetch",
                "--quiet",
                "--no-tags",
                "--depth=1",
                origin,
                source_ref,
            ),
        )
        if fetched.returncode != 0:
            return _probe_failure(
                fetched,
                operation="read",
                stage="advertised HEAD fetch",
            )
        resolved = self._git(
            machine,
            material,
            ("git", f"--git-dir={git_dir}", "rev-parse", "FETCH_HEAD"),
        )
        commit = resolved.stdout.strip()
        if resolved.returncode != 0 or _FULL_COMMIT.fullmatch(commit) is None:
            return GitWriteProbe(
                status="failed",
                commit=None,
                temporary_ref=None,
                diagnostic="The fetched Git probe commit could not be read back exactly.",
            )
        if commit != advertised_commit:
            return GitWriteProbe(
                status="failed",
                commit=commit,
                temporary_ref=None,
                diagnostic="The advertised GitHub commit changed during the write probe.",
            )

        pushed = self._git(
            machine,
            material,
            (
                "git",
                f"--git-dir={git_dir}",
                "push",
                "--porcelain",
                f"--force-with-lease={temporary_ref}:",
                origin,
                f"{commit}:{temporary_ref}",
            ),
        )
        if pushed.returncode != 0:
            return self._failed_push_probe(
                machine,
                material,
                origin=origin,
                commit=commit,
                temporary_ref=temporary_ref,
                pushed=pushed,
            )
        readback = self._git(
            machine,
            material,
            ("git", "ls-remote", origin, temporary_ref),
        )
        readback_invalid = False
        try:
            readback_refs = _parse_remote_refs(readback.stdout) if readback.returncode == 0 else {}
        except GitCredentialRefused:
            readback_refs = {}
            readback_invalid = True
        if readback_refs.get(temporary_ref) != commit:
            cleanup = self._cleanup_remote_ref(
                machine,
                material,
                origin=origin,
                git_dir=git_dir,
                commit=commit,
                temporary_ref=temporary_ref,
            )
            return cleanup or GitWriteProbe(
                status="failed",
                commit=commit,
                temporary_ref=None,
                diagnostic=(
                    "GitHub returned an invalid write-probe ref record; RCP removed the exact "
                    "request-scoped ref."
                    if readback_invalid
                    else "GitHub did not read back the request-scoped write probe exactly."
                ),
            )
        cleanup = self._cleanup_remote_ref(
            machine,
            material,
            origin=origin,
            git_dir=git_dir,
            commit=commit,
            temporary_ref=temporary_ref,
        )
        if cleanup is not None:
            return cleanup
        return GitWriteProbe(
            status="ready",
            commit=commit,
            temporary_ref=None,
            diagnostic="The request-scoped Git write probe passed and its temporary ref is gone.",
        )

    def _failed_push_probe(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
        *,
        origin: str,
        commit: str,
        temporary_ref: str,
        pushed: subprocess.CompletedProcess[str],
    ) -> GitWriteProbe:
        failure = _probe_failure(
            pushed,
            operation="write",
            stage="temporary-ref push",
        )
        if failure.status in {"github_host_trust_needed", "github_grant_needed"}:
            return GitWriteProbe(
                status=failure.status,
                commit=commit,
                temporary_ref=None,
                diagnostic=failure.diagnostic,
            )
        push_diagnostic = f"{pushed.stdout}\n{pushed.stderr}".lower()
        if any(marker in push_diagnostic for marker in _LEASE_REJECTION_MARKERS):
            return GitWriteProbe(
                status="temporary_ref_conflict",
                commit=commit,
                temporary_ref=temporary_ref,
                diagnostic=(
                    "The request-scoped Git write-probe ref was created by another writer. "
                    "RCP left it untouched."
                ),
            )
        observed = self._git(
            machine,
            material,
            ("git", "ls-remote", origin, temporary_ref),
        )
        if observed.returncode != 0:
            return GitWriteProbe(
                status="cleanup_failed",
                commit=commit,
                temporary_ref=temporary_ref,
                diagnostic=(
                    "The failed Git push may have created its request-scoped ref, and RCP could "
                    "not prove the ref absent. Remove that exact ref, then resume the same "
                    "provisioning request."
                ),
            )
        try:
            refs = _parse_remote_refs(observed.stdout)
        except GitCredentialRefused:
            return GitWriteProbe(
                status="cleanup_failed",
                commit=commit,
                temporary_ref=temporary_ref,
                diagnostic=(
                    "The failed Git push may have created its request-scoped ref, and GitHub "
                    "returned an invalid ref record. RCP left the ref untouched; inspect and "
                    "remove only that exact ref before resuming."
                ),
            )
        if refs:
            return GitWriteProbe(
                status=(
                    "cleanup_failed"
                    if refs.get(temporary_ref) == commit
                    else "temporary_ref_conflict"
                ),
                commit=commit,
                temporary_ref=temporary_ref,
                diagnostic=(
                    "The failed Git push left a request-scoped ref whose ownership is ambiguous. "
                    "RCP left it untouched; inspect and remove only that exact ref before "
                    "resuming."
                ),
            )
        return GitWriteProbe(
            status=failure.status,
            commit=commit,
            temporary_ref=None,
            diagnostic=failure.diagnostic,
        )

    def _cleanup_remote_ref(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
        *,
        origin: str,
        git_dir: str,
        commit: str,
        temporary_ref: str,
    ) -> GitWriteProbe | None:
        removed = self._git(
            machine,
            material,
            (
                "git",
                f"--git-dir={git_dir}",
                "push",
                "--porcelain",
                f"--force-with-lease={temporary_ref}:{commit}",
                origin,
                f":{temporary_ref}",
            ),
        )
        verification = self._git(
            machine,
            material,
            ("git", "ls-remote", origin, temporary_ref),
        )
        try:
            remaining_refs = (
                _parse_remote_refs(verification.stdout) if verification.returncode == 0 else {}
            )
        except GitCredentialRefused:
            remaining_refs = {temporary_ref: commit}
        if removed.returncode != 0 or verification.returncode != 0 or remaining_refs:
            return GitWriteProbe(
                status="cleanup_failed",
                commit=commit,
                temporary_ref=temporary_ref,
                diagnostic=(
                    "The request-scoped Git write ref could not be proven removed. Remove that "
                    "exact ref, then resume the same provisioning request."
                ),
            )
        return None

    def _git(
        self,
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
        argv: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        command = (
            "env",
            "-i",
            f"HOME={material.account_home}",
            f"USER={material.os_account}",
            f"LOGNAME={material.os_account}",
            f"PATH={_SAFE_PATH}",
            "LANG=C.UTF-8",
            "LC_ALL=C.UTF-8",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "GIT_CONFIG_NOSYSTEM=1",
            "GIT_TERMINAL_PROMPT=0",
            "GIT_SSH_VARIANT=ssh",
            f"GIT_SSH_COMMAND={_git_ssh_command(material)}",
            *argv,
        )
        return self._target_result(
            machine,
            command,
            timeout=SERVER_GIT_PROBE_TIMEOUT_SECONDS,
        )

    def _helper(
        self,
        machine: ProjectProvisioningMachineIntent,
        arguments: tuple[str, ...],
    ) -> dict[str, object]:
        result = self._target_result(
            machine,
            ("python3", "-c", _remote_helper_source(), *arguments),
            timeout=SERVER_GIT_CREDENTIAL_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise GitCredentialRefused(
                "The deploy-key helper refused the account, path, ownership, or mode. "
                "Inspect the named execution account and exact credential path."
            )
        if len(result.stdout.encode("utf-8")) > _MAX_HELPER_OUTPUT_BYTES:
            raise GitCredentialRefused("The deploy-key helper returned too much output.")
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise GitCredentialRefused("The deploy-key helper returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise GitCredentialRefused("The deploy-key helper returned an invalid receipt.")
        return payload

    def _target_result(
        self,
        machine: ProjectProvisioningMachineIntent,
        command: tuple[str, ...],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        argv = target_account_argv(self.layout, machine, command)
        try:
            result = self._runner(argv, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return subprocess.CompletedProcess(argv, 126, "", "")
        if (
            len(result.stdout.encode("utf-8", errors="replace")) > _MAX_GIT_OUTPUT_BYTES
            or len(result.stderr.encode("utf-8", errors="replace")) > _MAX_GIT_OUTPUT_BYTES
        ):
            return subprocess.CompletedProcess(argv, 126, "", "output exceeded the bound")
        return result

    def _material(
        self,
        payload: dict[str, object],
        machine: ProjectProvisioningMachineIntent,
        repository: GitHubRepositoryRef,
        *,
        space_id: str,
        project_id: str,
        repository_alias: str,
    ) -> DeployKeyMaterial:
        expected_keys = {
            "account",
            "home",
            "credentials_root",
            "private_key_path",
            "label",
            "public_key",
            "public_key_fingerprint",
            "created",
        }
        if set(payload) != expected_keys:
            raise GitCredentialRefused("The deploy-key helper receipt has an unexpected shape.")
        string_fields = expected_keys - {"created"}
        if not all(isinstance(payload[name], str) for name in string_fields) or not isinstance(
            payload["created"], bool
        ):
            raise GitCredentialRefused("The deploy-key helper receipt has invalid field types.")
        label = f"rcp:{space_id}:{project_id}:{repository_alias}"
        home = str(payload["home"])
        expected_root, expected_private = self._expected_key_paths(
            machine,
            home=home,
            project_id=project_id,
            repository_alias=repository_alias,
        )
        public_key = str(payload["public_key"])
        fingerprint = str(payload["public_key_fingerprint"])
        if (
            payload["account"] != machine.os_account
            or payload["credentials_root"] != expected_root
            or payload["private_key_path"] != expected_private
            or payload["label"] != label
            or not public_key.startswith("ssh-ed25519 ")
            or not public_key.endswith(f" {label}")
            or _FINGERPRINT.fullmatch(fingerprint) is None
        ):
            raise GitCredentialRefused(
                "The deploy-key helper receipt does not match the reviewed machine and project."
            )
        return DeployKeyMaterial(
            space_id=space_id,
            project_id=project_id,
            repository_alias=repository_alias,
            repository=repository,
            machine_alias=machine.alias,
            location=machine.location,
            host=machine.host,
            os_account=machine.os_account,
            central_root=(
                machine.central_root
                if machine.central_root is not None
                else str(remote_projects_root(home))
            ),
            account_home=home,
            credentials_root=expected_root,
            private_key_path=expected_private,
            label=label,
            public_key=public_key,
            public_key_fingerprint=fingerprint,
            created=bool(payload["created"]),
        )

    def _expected_key_paths(
        self,
        machine: ProjectProvisioningMachineIntent,
        *,
        home: str,
        project_id: str,
        repository_alias: str,
    ) -> tuple[str, str]:
        if machine.location == "local":
            if home != str(self.layout.service_home):
                raise GitCredentialRefused(
                    "The deploy-key helper receipt names the wrong local service home."
                )
            return (
                str(self.layout.credentials_root),
                str(self.layout.project_deploy_key_path(project_id, repository_alias)),
            )
        private = str(remote_project_deploy_key_path(home, project_id, repository_alias))
        return str(Path(private).parents[3]), private

    @staticmethod
    def _require_material_target(
        machine: ProjectProvisioningMachineIntent,
        material: DeployKeyMaterial,
    ) -> None:
        if (
            material.machine_alias != machine.alias
            or material.location != machine.location
            or material.host != machine.host
            or material.os_account != machine.os_account
            or material.central_root
            != (
                machine.central_root
                if machine.central_root is not None
                else str(remote_projects_root(material.account_home))
            )
        ):
            raise GitCredentialRefused("The deploy key belongs to another execution target.")


def deploy_key_operator_step(
    manager: GitCredentialManager,
    machine: ProjectProvisioningMachineIntent,
    material: DeployKeyMaterial,
    *,
    number: int,
    request_id: str,
    resume_argv: tuple[str, ...],
) -> ServerStep:
    _require_resume_request(resume_argv, request_id)
    instruction = (
        f"Open {material.repository.settings_url}; add the displayed public key with title "
        f"{material.label!r}, and enable Allow write access."
    )
    return ServerStep(
        number=number,
        title="Grant repository write access",
        purpose="Give one central checkout its repository-scoped GitHub write identity.",
        performed_by="human",
        target=ExternalServiceTarget(
            service="github.com",
            resource=material.repository.identity,
            destination_url=material.repository.settings_url,
            required_authority_role="repository administrator",
        ),
        phase="github_grant",
        state="operator_action_needed",
        expected_success=(
            "The request-scoped Git push is read back exactly and its temporary ref is removed."
        ),
        message=(
            "GitHub has not yet proven read and write access for this repository-scoped deploy "
            "key. Complete the displayed grant and host-trust steps, then resume."
        ),
        actions=(
            ExternalAction(instruction=instruction),
            CommandAction(argv=manager.github_trust_argv(machine, material)),
            ExternalAction(
                instruction=(
                    "Before accepting GitHub's host key, compare its fingerprint with "
                    f"{_GITHUB_FINGERPRINTS_URL}. A successful no-shell authentication may exit "
                    "with status 1."
                )
            ),
        ),
        fields=(
            NonsecretField(name="deploy_key_label", value=material.label),
            NonsecretField(name="deploy_public_key", value=material.public_key),
            NonsecretField(
                name="public_key_fingerprint",
                value=material.public_key_fingerprint,
            ),
        ),
        resume_argv=resume_argv,
    )


def restore_deploy_key_operator_step(
    manager: GitCredentialManager,
    machine: ProjectProvisioningMachineIntent,
    material: DeployKeyMaterial,
    *,
    number: int,
    resume_argv: tuple[str, ...],
) -> ServerStep:
    """Render the fresh-key grant required by replacement restore."""

    _require_restore_resume(resume_argv)
    instruction = (
        f"Open {material.repository.settings_url}; replace any stale RCP deploy key for "
        f"{material.label!r} with the displayed fresh public key, and enable Allow write access."
    )
    return ServerStep(
        number=number,
        title="Grant the fresh restore deploy key",
        purpose=(
            "Give the reconstructed central checkout a new repository-scoped GitHub identity."
        ),
        performed_by="human",
        target=ExternalServiceTarget(
            service="github.com",
            resource=material.repository.identity,
            destination_url=material.repository.settings_url,
            required_authority_role="repository administrator",
        ),
        phase="restore_github_grant",
        state="operator_action_needed",
        expected_success=(
            "The restore-owned Git push is read back exactly and its temporary ref is removed."
        ),
        message=(
            "GitHub has not yet proven read and write access for this fresh replacement key. "
            "Complete the displayed grant and host-trust steps, then resume restore."
        ),
        actions=(
            ExternalAction(instruction=instruction),
            CommandAction(argv=manager.github_trust_argv(machine, material)),
            ExternalAction(
                instruction=(
                    "Before accepting GitHub's host key, compare its fingerprint with "
                    f"{_GITHUB_FINGERPRINTS_URL}. A successful no-shell authentication may exit "
                    "with status 1."
                )
            ),
        ),
        fields=(
            NonsecretField(name="deploy_key_label", value=material.label),
            NonsecretField(name="deploy_public_key", value=material.public_key),
            NonsecretField(
                name="public_key_fingerprint",
                value=material.public_key_fingerprint,
            ),
        ),
        resume_argv=resume_argv,
    )


def empty_repository_operator_step(
    material: DeployKeyMaterial,
    *,
    number: int,
    request_id: str,
    resume_argv: tuple[str, ...],
) -> ServerStep:
    _require_resume_request(resume_argv, request_id)
    repository_url = f"https://github.com/{material.repository.identity}"
    return ServerStep(
        number=number,
        title="Push the repository's first commit",
        purpose="Give the central checkout one real human-authored Git commit to clone.",
        performed_by="human",
        target=ExternalServiceTarget(
            service="github.com",
            resource=material.repository.identity,
            destination_url=repository_url,
            required_authority_role="repository contributor",
        ),
        phase="github_initial_commit",
        state="operator_action_needed",
        expected_success="GitHub advertises one existing commit for the provisioning write probe.",
        message=(
            "This repository is empty. Push the local code through the ordinary human Git "
            "workflow, then resume the same provisioning request."
        ),
        actions=(
            ExternalAction(
                instruction=(
                    f"Push the intended codebase to {repository_url} with its first real commit. "
                    "RCP will not create a repository or invent an initialization commit."
                )
            ),
        ),
        fields=(NonsecretField(name="repository", value=material.repository.identity),),
        resume_argv=resume_argv,
    )


def cleanup_ref_operator_step(
    material: DeployKeyMaterial,
    probe: GitWriteProbe,
    *,
    number: int,
    request_id: str,
    resume_argv: tuple[str, ...],
) -> ServerStep:
    _require_resume_request(resume_argv, request_id)
    if probe.status != "cleanup_failed" or probe.temporary_ref is None:
        raise ValueError("cleanup action requires one failed request-scoped ref cleanup")
    branches_url = f"https://github.com/{material.repository.identity}/branches"
    return ServerStep(
        number=number,
        title="Remove the request-scoped Git probe ref",
        purpose="Finish the write probe without leaving a temporary GitHub ref.",
        performed_by="human",
        target=ExternalServiceTarget(
            service="github.com",
            resource=material.repository.identity,
            destination_url=branches_url,
            required_authority_role="repository administrator",
        ),
        phase="github_probe_cleanup",
        state="operator_action_needed",
        expected_success="The exact request-scoped ref is absent from GitHub.",
        message=probe.diagnostic,
        actions=(
            ExternalAction(
                instruction=(
                    f"Open {branches_url}; remove only {probe.temporary_ref!r}, then resume the "
                    "same provisioning request."
                )
            ),
        ),
        fields=(NonsecretField(name="temporary_ref", value=probe.temporary_ref),),
        resume_argv=resume_argv,
    )


def _validate_machine(machine: ProjectProvisioningMachineIntent, layout: ServerLayout) -> None:
    if machine.location == "local":
        if (
            machine.os_account != layout.service_account
            or machine.host
            or machine.central_root != str(layout.projects_root)
        ):
            raise GitCredentialRefused(
                "The server-local deploy key must use the installed rcp account and project root."
            )
    elif not machine.host:
        raise GitCredentialRefused("An SSH deploy-key target requires one configured host.")


def _require_resume_request(resume_argv: tuple[str, ...], request_id: str) -> None:
    try:
        parsed = uuid.UUID(request_id)
    except (AttributeError, ValueError) as exc:
        raise ValueError("request id must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != request_id:
        raise ValueError("request id must be a lowercase canonical UUID4")
    meaningful = resume_argv[:-1] if resume_argv[-1:] == ("--machine-readable",) else resume_argv
    if meaningful[-4:] != ("server", "project", "provision", request_id):
        raise ValueError("resume argv must name this exact provisioning request")


def _require_restore_resume(resume_argv: tuple[str, ...]) -> None:
    try:
        marker = resume_argv.index("restore")
    except ValueError as exc:
        raise ValueError("restore resume argv must name server restore") from exc
    if (
        marker == 0
        or resume_argv[marker - 1] != "server"
        or marker + 1 >= len(resume_argv)
        or "--identity-file" not in resume_argv[marker + 1 :]
        or "--confirm-data-dir" not in resume_argv[marker + 1 :]
    ):
        raise ValueError("restore resume argv must carry its archive, identity file, and target")


def _runuser_argv(layout: ServerLayout, command: tuple[str, ...]) -> tuple[str, ...]:
    return (
        "runuser",
        "--user",
        layout.service_account,
        "--",
        "env",
        "-i",
        f"HOME={layout.service_home}",
        f"USER={layout.service_account}",
        f"LOGNAME={layout.service_account}",
        f"PATH={_SAFE_PATH}",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        *command,
    )


def _strict_ssh_arguments(host: str, command: str) -> list[str]:
    return [
        "ssh",
        *SSH_OPTIONS,
        "-o",
        "StrictHostKeyChecking=yes",
        host,
        command,
    ]


def _git_ssh_command(material: DeployKeyMaterial) -> str:
    return shlex.join(
        (
            "ssh",
            "-F",
            "/dev/null",
            "-i",
            material.private_key_path,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "GlobalKnownHostsFile=/etc/ssh/ssh_known_hosts",
            "-o",
            f"UserKnownHostsFile={Path(material.account_home) / '.ssh' / 'known_hosts'}",
        )
    )


def deploy_key_ssh_command(material: DeployKeyMaterial) -> str:
    """Return the exact noninteractive SSH command bound to one prepared key."""

    return _git_ssh_command(material)


def target_account_argv(
    layout: ServerLayout,
    machine: ProjectProvisioningMachineIntent,
    command: tuple[str, ...],
) -> tuple[str, ...]:
    """Wrap one command for the reviewed local or SSH execution account."""

    _validate_machine(machine, layout)
    if machine.location == "local":
        return _runuser_argv(layout, command)
    return _runuser_argv(
        layout,
        tuple(_strict_ssh_arguments(machine.host, shlex.join(command))),
    )


def _parse_remote_refs(output: str) -> dict[str, str]:
    if len(output.encode("utf-8", errors="replace")) > _MAX_GIT_OUTPUT_BYTES:
        raise GitCredentialRefused("Git returned too much ref output.")
    refs: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or _FULL_COMMIT.fullmatch(parts[0]) is None:
            raise GitCredentialRefused("GitHub returned an invalid ref record.")
        ref = parts[1]
        if ref != "HEAD" and not ref.startswith("refs/heads/"):
            raise GitCredentialRefused("GitHub returned an unexpected ref namespace.")
        if ref in refs:
            raise GitCredentialRefused("GitHub returned a duplicate ref record.")
        refs[ref] = parts[0]
    return refs


def _preferred_source_ref(refs: dict[str, str]) -> tuple[str, str]:
    if set(refs) != {"HEAD"}:
        raise GitCredentialRefused("GitHub did not advertise exactly one HEAD commit.")
    return "HEAD", refs["HEAD"]


def _probe_failure(
    result: subprocess.CompletedProcess[str],
    *,
    operation: Literal["read", "write", "local"],
    stage: str,
) -> GitWriteProbe:
    diagnostic = f"{result.stdout}\n{result.stderr}".lower()
    if any(marker in diagnostic for marker in _HOST_TRUST_MARKERS):
        return GitWriteProbe(
            status="github_host_trust_needed",
            commit=None,
            temporary_ref=None,
            diagnostic=(
                "GitHub host trust is not established for the checkout account. Compare the "
                "published fingerprint, accept it explicitly, and resume."
            ),
        )
    if any(marker in diagnostic for marker in _NETWORK_MARKERS) or result.returncode == 255:
        return GitWriteProbe(
            status="unavailable",
            commit=None,
            temporary_ref=None,
            diagnostic="The target account could not reach GitHub. Correct connectivity and resume.",
        )
    if operation in {"read", "write"} and any(marker in diagnostic for marker in _AUTH_MARKERS):
        return GitWriteProbe(
            status="github_grant_needed",
            commit=None,
            temporary_ref=None,
            diagnostic=(
                "GitHub has not granted this repository-scoped key the required read and write "
                "access. Add or correct the deploy key grant, then resume."
            ),
        )
    return GitWriteProbe(
        status="failed",
        commit=None,
        temporary_ref=None,
        diagnostic=(
            f"The Git write probe failed during {stage} without a recognized host, network, "
            "or grant diagnosis. Inspect the target account and resume only after correcting it."
        ),
    )


@lru_cache(maxsize=1)
def _remote_helper_source() -> str:
    return (
        importlib.resources.files("rcp.server_ops")
        .joinpath("remote_git_credentials.py")
        .read_text(encoding="utf-8")
    )


def _run_process(
    argv: tuple[str, ...],
    *,
    cwd: Path | None = None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return subprocess.CompletedProcess(argv, 126, "", "")

    assert process.stdout is not None
    assert process.stderr is not None
    streams = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        streams.register(process.stdout, selectors.EVENT_READ, "stdout")
        streams.register(process.stderr, selectors.EVENT_READ, "stderr")
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                return subprocess.CompletedProcess(argv, 126, "", "command timed out")
            events = streams.select(remaining)
            if not events:
                continue
            for key, _mask in events:
                try:
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                except OSError:
                    _terminate_process(process)
                    return subprocess.CompletedProcess(argv, 126, "", "command output failed")
                if not chunk:
                    streams.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                buffer = buffers[key.data]
                capacity = _MAX_GIT_OUTPUT_BYTES + 1 - len(buffer)
                if capacity > 0:
                    buffer.extend(chunk[:capacity])
                if len(chunk) > capacity or len(buffer) > _MAX_GIT_OUTPUT_BYTES:
                    _terminate_process(process)
                    return subprocess.CompletedProcess(
                        argv,
                        126,
                        "",
                        "output exceeded the bound",
                    )
        try:
            returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            return subprocess.CompletedProcess(argv, 126, "", "command timed out")
        except OSError:
            _terminate_process(process)
            return subprocess.CompletedProcess(argv, 126, "", "command wait failed")
        return subprocess.CompletedProcess(
            argv,
            returncode,
            buffers["stdout"].decode("utf-8", errors="replace"),
            buffers["stderr"].decode("utf-8", errors="replace"),
        )
    finally:
        streams.close()
        process.stdout.close()
        process.stderr.close()


def run_bounded_process(
    argv: tuple[str, ...],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run one subprocess with the server Git output and timeout bounds."""

    return _run_process(argv, timeout=timeout)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        process.kill()
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=1)


__all__ = [
    "CommandRunner",
    "DeployKeyMaterial",
    "GitCredentialManager",
    "GitCredentialRefused",
    "GitWriteProbe",
    "cleanup_ref_operator_step",
    "deploy_key_ssh_command",
    "deploy_key_operator_step",
    "empty_repository_operator_step",
    "restore_deploy_key_operator_step",
    "run_bounded_process",
    "target_account_argv",
]
