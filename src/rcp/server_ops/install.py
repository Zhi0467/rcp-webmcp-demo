"""Concrete, idempotent installation of one source-built RCP team server."""

from __future__ import annotations

import base64
import grp
import hashlib
import http.client
import json
import os
import platform
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, TypeVar

from rcp.limits import (
    SERVER_HEALTH_REQUEST_TIMEOUT_SECONDS,
    SERVER_INSTALL_ACCOUNT_TIMEOUT_SECONDS,
    SERVER_INSTALL_BUILD_TIMEOUT_SECONDS,
    SERVER_INSTALL_HEALTH_POLL_INTERVAL_SECONDS,
    SERVER_INSTALL_HEALTH_RESPONSE_MAX_BYTES,
    SERVER_INSTALL_HEALTH_TIMEOUT_SECONDS,
    SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
    SERVER_INSTALL_SERVICE_TIMEOUT_SECONDS,
    SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
)
from rcp.server_ops.cli import (
    CallerIdentity,
    PreparedServerCommand,
    ServerEventEmitter,
)
from rcp.server_ops.config import (
    InstalledServerConfig,
    ServerSourceConfig,
    create_installed_server_config,
    load_installed_server_config,
    write_installed_server_config,
)
from rcp.server_ops.github import parse_github_repository_ref
from rcp.server_ops.layout import (
    DEFAULT_SERVER_LAYOUT,
    ServerLayout,
    server_service_unit_text,
)
from rcp.server_ops.models import (
    CommandAction,
    ExternalAction,
    ExternalServiceTarget,
    MachineTarget,
    NonsecretField,
    ServerCommandRequest,
    ServerPlanEvent,
    ServerStep,
)

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_OPENSSH_FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{20,64}={0,2}")
_VERSION = re.compile(r"(?:^|\s|v)(?P<major>\d+)\.(?P<minor>\d+)(?:\.\d+)?")
_AUTH_FAILURE_MARKERS = (
    "authentication failed",
    "could not read username",
    "permission denied (publickey)",
    "repository not found",
    "could not read from remote repository",
    "host key verification failed",
)
_NETWORK_FAILURE_MARKERS = (
    "could not resolve host",
    "failed to connect",
    "connection timed out",
    "network is unreachable",
    "connection refused",
    "temporary failure in name resolution",
    "command timed out",
)
_SOURCE_PRIVATE_KEY = "source_ed25519"
_SOURCE_PUBLIC_KEY = "source_ed25519.pub"
_WRAPPER_MODE = 0o755
_UNIT_MODE = 0o644
_SERVICE_DIRECTORY_MODE = 0o700
_CONFIG_DIRECTORY_MODE = 0o750
_PRIVATE_KEY_MODE = 0o600
_PUBLIC_KEY_MODE = 0o644
_PUBLIC_KEY_MAX_BYTES = 4096


class InstallRefused(RuntimeError):
    """A known, safe refusal whose message may be shown to the operator."""


class _ReportedInstallFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class GitHubRepository:
    slug: str
    https_origin: str
    ssh_origin: str
    deploy_keys_url: str


@dataclass(frozen=True)
class HostFacts:
    ubuntu_release: Literal["22.04", "24.04"]
    architecture: Literal["x86_64"] = "x86_64"
    node_major: Literal[24] = 24


@dataclass(frozen=True)
class SourceAccess:
    config: InstalledServerConfig
    repository: GitHubRepository
    grant_needed: bool
    deploy_key_label: str | None = None
    public_key: str | None = None


@dataclass(frozen=True)
class ManagedCheckout:
    commit: str
    is_current_release: bool


@dataclass(frozen=True)
class ServiceInstallState:
    data_state: Literal["fresh", "initialized"]
    service_state: str


@dataclass(frozen=True)
class ServiceHealth:
    status: Literal["ok"]
    space_kind: Literal["team"]
    space_name: str


class InstallMachine(Protocol):
    def validate_host(self) -> HostFacts: ...

    def converge_account_and_layout(self) -> None: ...

    def prepare_source_access(self, repository: GitHubRepository) -> SourceAccess: ...

    def converge_source_checkout(self, access: SourceAccess) -> ManagedCheckout: ...

    def build_release(self, checkout: ManagedCheckout) -> Path: ...

    def install_service(
        self,
        checkout: ManagedCheckout,
        release: Path,
    ) -> ServiceInstallState: ...

    def activate_and_verify(self) -> ServiceHealth: ...


def prepare_install_command(
    request: ServerCommandRequest,
    identity: CallerIdentity,
    *,
    machine: InstallMachine | None = None,
    repository: GitHubRepository | None = None,
    bootstrap_root: Path | None = None,
    resume_executable: Path | None = None,
) -> PreparedServerCommand:
    """Prepare the complete plan without beginning installation work."""

    if request.command != "server install" or request.team_name is None:
        raise ValueError("prepare_install_command requires one server install request")
    resolved_repository = repository or discover_bootstrap_repository(bootstrap_root)
    resolved_executable = _absolute_invoked_executable(resume_executable)
    resolved_machine = machine or LinuxInstallMachine()
    plan = ServerPlanEvent(
        command=request.command,
        timestamp=datetime.now(UTC),
        steps=_install_plan(identity, resolved_repository),
    )

    def execute(emitter: ServerEventEmitter, _input_stream: BinaryIO) -> None:
        _execute_install(
            request,
            emitter,
            resolved_machine,
            resolved_repository,
            bootstrap_executable=resolved_executable,
        )

    return PreparedServerCommand(plan=plan, execute=execute)


def normalize_github_repository(origin: str) -> GitHubRepository:
    """Return the one credential-free repository identity accepted by install."""

    try:
        reference = parse_github_repository_ref(origin)
    except ValueError as exc:
        raise ValueError(
            "the bootstrap origin must be an HTTPS or SSH github.com repository"
        ) from exc
    return GitHubRepository(
        slug=reference.identity,
        https_origin=reference.https_clone_url,
        ssh_origin=reference.ssh_clone_url,
        deploy_keys_url=reference.settings_url,
    )


def discover_bootstrap_repository(bootstrap_root: Path | None = None) -> GitHubRepository:
    """Read origin from the checkout supplying this executable without adopting it."""

    candidates = (
        (bootstrap_root,) if bootstrap_root is not None else (Path.cwd(), Path(__file__).parents[3])
    )
    failures = 0
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        result = _run_process(
            (
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "remote",
                "get-url",
                "origin",
            ),
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            failures += 1
            continue
        try:
            return normalize_github_repository(result.stdout.strip())
        except ValueError:
            failures += 1
    if failures:
        raise ValueError(
            "server install must run from an RCP GitHub checkout with a supported origin"
        )
    raise ValueError("server install could not locate its bootstrap checkout")


def _absolute_invoked_executable(value: Path | None) -> Path:
    executable = value if value is not None else Path(sys.argv[0])
    if value is None and not executable.is_absolute():
        located = shutil.which(str(executable))
        if located is not None:
            executable = Path(located)
    resolved = executable.expanduser().resolve()
    if not resolved.is_absolute():  # pragma: no cover - Path.resolve owns this
        raise ValueError("server install requires an absolute executable path")
    if value is None and (not resolved.is_file() or not os.access(resolved, os.X_OK)):
        raise ValueError(
            "server install must be invoked through an executable RCP CLI so its resume command "
            "is exact"
        )
    return resolved


def _install_plan(
    identity: CallerIdentity,
    repository: GitHubRepository,
) -> tuple[ServerStep, ...]:
    root = MachineTarget(host=identity.host, os_account="root")
    service = MachineTarget(host=identity.host, os_account="rcp")
    github = ExternalServiceTarget(
        service="github.com",
        resource=repository.slug,
        destination_url=repository.deploy_keys_url,
        required_authority_role="repository administrator",
    )
    return (
        ServerStep(
            number=1,
            title="Validate the supported Ubuntu host",
            purpose="Prove the fixed host and tool prerequisites before changing machine state.",
            performed_by="system",
            target=root,
            phase="host_preflight",
            state="pending",
            expected_success=(
                "Ubuntu 22.04 or 24.04 x86-64 has systemd, Git, uv, Node.js 24/npm, "
                "SSH, and age 1.x."
            ),
            message="RCP will validate the supported operating system and required tools.",
        ),
        ServerStep(
            number=2,
            title="Converge the dedicated service account and paths",
            purpose=(
                "Create or validate only the rcp identity and the fixed, conservatively owned "
                "server layout."
            ),
            performed_by="system",
            target=root,
            phase="account_layout",
            state="pending",
            expected_success=(
                "rcp has /home/rcp, /bin/bash, *NP*, no supplemental or general sudo "
                "authority, the fixed owned layout, and an executable uv-managed Python 3.12."
            ),
            message="RCP will converge the dedicated account and fixed directories.",
        ),
        ServerStep(
            number=3,
            title="Prepare isolated source access",
            purpose=(
                "Prove public access or create one dedicated read-only source key without "
                "borrowing the operator's GitHub credential."
            ),
            performed_by="system",
            target=service,
            phase="source_access_prepare",
            state="pending",
            expected_success=(
                "The immutable installation id and credential-free source configuration are "
                "recorded; only a private source has a dedicated key."
            ),
            message="RCP will prepare source access as the rcp account.",
        ),
        ServerStep(
            number=4,
            title="Confirm the GitHub source grant",
            purpose=(
                "Skip external work for a public repository or request the exact read-only "
                "deploy-key grant for a private repository."
            ),
            performed_by="human",
            target=github,
            phase="source_grant",
            state="pending",
            expected_success=(
                "rcp can read origin/main with no operator credential and any deploy key has "
                "Allow write access disabled."
            ),
            message="RCP will determine whether the repository needs a human source grant.",
        ),
        ServerStep(
            number=5,
            title="Converge the managed main checkout",
            purpose=(
                "Fetch through the recorded source mode into the separate RCP-owned checkout "
                "without adopting the bootstrap checkout."
            ),
            performed_by="system",
            target=service,
            phase="source_checkout",
            state="pending",
            expected_success=(
                "The managed checkout is clean at the exact origin/main commit, or install "
                "refuses a version change owned by server update."
            ),
            message="RCP will converge the separate managed source checkout.",
        ),
        ServerStep(
            number=6,
            title="Build the immutable release",
            purpose=(
                "Create the exact per-commit worktree and run npm ci, the Web build, and "
                "uv sync --frozen as rcp."
            ),
            performed_by="system",
            target=service,
            phase="release_build",
            state="pending",
            expected_success="The exact commit has a clean Web build and Python 3.12 environment.",
            message="RCP will build the source commit as the rcp account.",
        ),
        ServerStep(
            number=7,
            title="Install the stable wrapper and systemd unit",
            purpose=(
                "Install root-owned integration without reload mode and keep a fresh service "
                "stopped and disabled."
            ),
            performed_by="system",
            target=root,
            phase="service_install",
            state="pending",
            expected_success=(
                "The wrapper, current release, and non-reloading loopback unit are exact; a "
                "fresh data directory remains stopped and disabled."
            ),
            message="RCP will install and validate the fixed service integration.",
        ),
        ServerStep(
            number=8,
            title="Initialize the team space in the operator terminal",
            purpose=(
                "Keep the one-time bootstrap code out of service logs by running the existing "
                "interactive team initialization command as rcp."
            ),
            performed_by="human",
            target=service,
            phase="team_space_init",
            state="pending",
            expected_success=(
                "The terminal reports the initialized team space and shows its one-time "
                "bootstrap code exactly once."
            ),
            message="RCP will check whether the owned data directory is initialized.",
        ),
        ServerStep(
            number=9,
            title="Activate and read back the team service",
            purpose=(
                "Converge systemd only after initialization and verify the fixed loopback HTTP "
                "health response."
            ),
            performed_by="system",
            target=root,
            phase="service_activate",
            state="pending",
            expected_success=(
                "rcp.service is enabled and active, and 127.0.0.1:8421 reports status ok for a "
                "team space."
            ),
            message="RCP will activate systemd and verify loopback health.",
        ),
    )


def _execute_install(
    request: ServerCommandRequest,
    emitter: ServerEventEmitter,
    machine: InstallMachine,
    repository: GitHubRepository,
    *,
    bootstrap_executable: Path,
) -> None:
    planned = emitter.events[0]
    if not isinstance(planned, ServerPlanEvent):  # pragma: no cover - emitter owns this
        raise AssertionError("install execution requires its plan")
    steps = planned.steps
    try:
        facts = _run_step(
            emitter,
            steps[0],
            running="Validating Ubuntu, architecture, systemd, and required tool versions.",
            operation=machine.validate_host,
            succeeded="The host and every required tool match the supported installation contract.",
            fields=lambda value: (
                NonsecretField(name="ubuntu_release", value=value.ubuntu_release),
                NonsecretField(name="architecture", value=value.architecture),
                NonsecretField(name="node_major", value=value.node_major),
            ),
        )
        _run_step(
            emitter,
            steps[1],
            running="Creating or validating the dedicated rcp account and fixed path ownership.",
            operation=machine.converge_account_and_layout,
            succeeded=(
                "The dedicated rcp account and fixed server directories are present with the "
                "required ownership and modes."
            ),
        )
        access = _run_step(
            emitter,
            steps[2],
            running=f"Preparing credential-isolated read access for {repository.slug} as rcp.",
            operation=lambda: machine.prepare_source_access(repository),
            succeeded="The installed-server source identity and access mode are recorded.",
            fields=lambda value: (
                NonsecretField(name="installation_id", value=value.config.installation_id),
                NonsecretField(name="source_repository", value=value.repository.slug),
                NonsecretField(
                    name="source_authentication", value=value.config.source.authentication
                ),
            ),
        )
        if access.grant_needed:
            _emit_source_grant_pause(
                emitter,
                steps[3],
                access,
                bootstrap_executable=bootstrap_executable,
                team_name=request.team_name,
            )
            return
        _complete_no_action_step(
            emitter,
            steps[3],
            running="Checking that the recorded source mode can read origin/main as rcp.",
            succeeded="No operator action is needed; isolated read access to origin/main passed.",
            fields=(
                NonsecretField(
                    name="source_authentication",
                    value=access.config.source.authentication,
                ),
            ),
        )
        checkout = _run_step(
            emitter,
            steps[4],
            running="Fetching and validating the separate managed main checkout as rcp.",
            operation=lambda: machine.converge_source_checkout(access),
            succeeded="The managed checkout is clean at the exact install commit.",
            fields=lambda value: (NonsecretField(name="release_commit", value=value.commit),),
        )
        release = _run_step(
            emitter,
            steps[5],
            running=(
                "Creating the per-commit release and running npm ci, the Web build, and "
                "uv sync --frozen as rcp."
            ),
            operation=lambda: machine.build_release(checkout),
            succeeded="The immutable per-commit Web and Python release is ready.",
            fields=lambda value: (NonsecretField(name="release_path", value=str(value)),),
        )
        service_state = _run_step(
            emitter,
            steps[6],
            running="Installing or validating the wrapper, current pointer, and systemd unit.",
            operation=lambda: machine.install_service(checkout, release),
            succeeded="The fixed service integration is installed in its safe pre-activation state.",
            fields=lambda value: (
                NonsecretField(name="data_state", value=value.data_state),
                NonsecretField(name="service_state", value=value.service_state),
            ),
        )
        if service_state.data_state == "fresh":
            _emit_team_init_pause(emitter, steps[7], request.team_name)
            return
        _complete_no_action_step(
            emitter,
            steps[7],
            running="Confirming that the owned team data directory was initialized previously.",
            succeeded="The owned data directory already contains the initialized database.",
            fields=(NonsecretField(name="data_state", value="initialized"),),
        )
        _run_step(
            emitter,
            steps[8],
            running="Enabling and starting rcp.service, then reading loopback HTTP health.",
            operation=machine.activate_and_verify,
            succeeded="The source-built team service is enabled, active, and healthy on loopback.",
            fields=lambda value: (
                NonsecretField(name="status", value=value.status),
                NonsecretField(name="space_kind", value=value.space_kind),
                NonsecretField(name="space_name", value=value.space_name),
            ),
        )
        _ = facts
    except _ReportedInstallFailure:
        return


_T = TypeVar("_T")


def _run_step(
    emitter: ServerEventEmitter,
    planned: ServerStep,
    *,
    running: str,
    operation,
    succeeded: str,
    fields=lambda _value: (),
) -> _T:
    emitter.emit_step(planned.model_copy(update={"state": "running", "message": running}))
    try:
        value = operation()
    except InstallRefused as exc:
        emitter.emit_step(planned.model_copy(update={"state": "failed", "message": str(exc)}))
        raise _ReportedInstallFailure from exc
    emitter.emit_step(
        planned.model_copy(
            update={
                "state": "succeeded",
                "message": succeeded,
                "fields": tuple(fields(value)),
            }
        )
    )
    return value


def _complete_no_action_step(
    emitter: ServerEventEmitter,
    planned: ServerStep,
    *,
    running: str,
    succeeded: str,
    fields: tuple[NonsecretField, ...] = (),
) -> None:
    emitter.emit_step(planned.model_copy(update={"state": "running", "message": running}))
    emitter.emit_step(
        planned.model_copy(update={"state": "succeeded", "message": succeeded, "fields": fields})
    )


def _emit_source_grant_pause(
    emitter: ServerEventEmitter,
    planned: ServerStep,
    access: SourceAccess,
    *,
    bootstrap_executable: Path,
    team_name: str,
) -> None:
    if access.deploy_key_label is None or access.public_key is None:
        raise RuntimeError("private source access did not provide its public grant material")
    key_path = DEFAULT_SERVER_LAYOUT.credentials_root / _SOURCE_PRIVATE_KEY
    trust_command = (
        "sudo",
        "-u",
        "rcp",
        "-H",
        "ssh",
        "-F",
        "/dev/null",
        "-i",
        str(key_path),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=ask",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        f"UserKnownHostsFile={DEFAULT_SERVER_LAYOUT.ssh_state_root / 'known_hosts'}",
        "-T",
        "git@github.com",
    )
    resume = (
        "sudo",
        str(bootstrap_executable),
        "server",
        "install",
        "--team-name",
        team_name,
    )
    emitter.emit_step(
        planned.model_copy(
            update={
                "state": "operator_action_needed",
                "message": (
                    "This source is not publicly readable. Add the shown key as a read-only "
                    "deploy key, confirm GitHub host trust as rcp, then rerun the exact command."
                ),
                "actions": (
                    ExternalAction(
                        instruction=(
                            f"Open {access.repository.deploy_keys_url}; add the shown public key "
                            f"with title {access.deploy_key_label!r}; leave Allow write access "
                            "unchecked."
                        )
                    ),
                    CommandAction(argv=trust_command),
                    ExternalAction(
                        instruction=(
                            "Accept the GitHub host key only after comparing its fingerprint with "
                            "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/"
                            "githubs-ssh-key-fingerprints. GitHub then says the key authenticated "
                            "successfully; ssh exit status 1 is expected."
                        )
                    ),
                ),
                "fields": (
                    NonsecretField(name="deploy_key_label", value=access.deploy_key_label),
                    NonsecretField(name="deploy_public_key", value=access.public_key),
                    NonsecretField(
                        name="public_key_fingerprint",
                        value=access.config.source.public_key_fingerprint or "missing",
                    ),
                ),
                "resume_argv": resume,
            }
        )
    )


def _emit_team_init_pause(
    emitter: ServerEventEmitter,
    planned: ServerStep,
    team_name: str,
) -> None:
    wrapper = str(DEFAULT_SERVER_LAYOUT.cli_wrapper)
    init = ("sudo", "-u", "rcp", "-H", wrapper, "space", "init", "--team", "--name", team_name)
    resume = ("sudo", wrapper, "server", "install", "--team-name", team_name)
    emitter.emit_step(
        planned.model_copy(
            update={
                "state": "operator_action_needed",
                "message": (
                    "The release is installed and the fresh service is stopped and disabled. "
                    "Initialize in this terminal and retain the one-time bootstrap code securely, "
                    "then rerun install so RCP activates and verifies the service."
                ),
                "actions": (
                    CommandAction(argv=init),
                    ExternalAction(
                        instruction=(
                            "Success signal: the terminal names the initialized team space and "
                            "shows one one-time bootstrap code. Do not paste that code into RCP "
                            "logs or command arguments."
                        )
                    ),
                ),
                "fields": (NonsecretField(name="team_name", value=team_name),),
                "resume_argv": resume,
            }
        )
    )


class LinuxInstallMachine:
    """One concrete Ubuntu implementation behind the structured operation contract."""

    def __init__(self, layout: ServerLayout = DEFAULT_SERVER_LAYOUT) -> None:
        self.layout = layout
        self._service_uid: int | None = None
        self._service_gid: int | None = None

    def validate_host(self) -> HostFacts:
        os_release = _read_os_release(Path("/etc/os-release"))
        release = os_release.get("VERSION_ID")
        if os_release.get("ID") != "ubuntu" or release not in {"22.04", "24.04"}:
            raise InstallRefused(
                "Install supports only Ubuntu 22.04 or 24.04; no machine state was changed."
            )
        if platform.machine() != "x86_64":
            raise InstallRefused(
                "Install supports only x86-64 hosts; no machine state was changed."
            )
        if not Path("/run/systemd/system").is_dir():
            raise InstallRefused(
                "systemd is not the running service manager. Boot this Ubuntu host with systemd "
                "and rerun the same install command."
            )
        for command in (
            "age",
            "curl",
            "git",
            "node",
            "npm",
            "runuser",
            "ssh",
            "ssh-keygen",
            "sudo",
            "systemctl",
            "useradd",
            "uv",
        ):
            if shutil.which(command) is None:
                raise InstallRefused(
                    f"Required tool {command} is missing. Install it for all users, then rerun "
                    "the same server install command."
                )
        _require_command(("systemctl", "--version"), "systemd could not be executed")
        manager = _require_command(
            ("systemctl", "show", "--property=Version", "--value"),
            "The running systemd manager could not be reached. Boot this Ubuntu host with "
            "systemd as PID 1 and rerun the same install command.",
        )
        if not manager.stdout.strip():
            raise InstallRefused(
                "The running systemd manager returned no version. Boot this Ubuntu host with "
                "systemd as PID 1 and rerun the same install command."
            )
        node = _require_command(("node", "--version"), "Node.js could not be executed").stdout
        node_major = _leading_major(node)
        if node_major != 24:
            raise InstallRefused(
                f"Node.js 24 is required, but the machine reports major {node_major}. Install "
                "Node.js 24 for all users and rerun the same command."
            )
        age = _require_command(("age", "--version"), "age could not be executed")
        age_version = _major_minor(f"{age.stdout} {age.stderr}")
        if age_version[0] != 1:
            raise InstallRefused(
                "age >=1.0.0,<2.0.0 is required. Install age 1.x for all users and rerun the "
                "same command."
            )
        return HostFacts(ubuntu_release=release)

    def converge_account_and_layout(self) -> None:
        account = self._converge_account()
        self._service_uid = account.pw_uid
        self._service_gid = account.pw_gid
        _converge_directory(
            self.layout.service_home,
            uid=account.pw_uid,
            gid=account.pw_gid,
            mode=_SERVICE_DIRECTORY_MODE,
        )
        for path in (
            self.layout.server_root,
            self.layout.source_checkout.parent,
            self.layout.releases_root,
            self.layout.data_dir,
            self.layout.projects_root,
            self.layout.credentials_root,
            self.layout.update_checkpoints_root,
            self.layout.restore_operations_root,
            self.layout.codex_state_root,
            self.layout.claude_state_root,
            self.layout.ssh_state_root,
        ):
            _converge_directory(
                path,
                uid=account.pw_uid,
                gid=account.pw_gid,
                mode=_SERVICE_DIRECTORY_MODE,
            )
        _converge_directory(
            self.layout.config_path.parent,
            uid=0,
            gid=account.pw_gid,
            mode=_CONFIG_DIRECTORY_MODE,
        )
        self._validate_service_tooling()

    def prepare_source_access(self, repository: GitHubRepository) -> SourceAccess:
        self._require_service_identity()
        private_path = self.layout.credentials_root / _SOURCE_PRIVATE_KEY
        public_path = self.layout.credentials_root / _SOURCE_PUBLIC_KEY
        if self.layout.config_path.exists() or self.layout.config_path.is_symlink():
            config = load_installed_server_config(self.layout.config_path)
            configured_repository = normalize_github_repository(config.source.origin)
            if configured_repository.slug != repository.slug:
                raise InstallRefused(
                    "The installed source repository differs from this executable's GitHub "
                    "repository. Use the matching checkout or restore the recorded source."
                )
            if config.source.authentication == "public":
                if (
                    private_path.exists()
                    or private_path.is_symlink()
                    or public_path.exists()
                    or public_path.is_symlink()
                ):
                    raise InstallRefused(
                        "A public source configuration has an unexpected source-key file. Remove "
                        "nothing automatically; inspect the credential path and rerun."
                    )
                probe = self._probe_source(repository.https_origin, source=None)
                if probe != "ready":
                    raise InstallRefused(
                        "The recorded public source is not readable from this host. Restore "
                        "network access or repository visibility, then rerun."
                    )
                return SourceAccess(config=config, repository=repository, grant_needed=False)
            public_key = self._validate_source_key_pair(config, private_path, public_path)
            probe = self._probe_source(repository.ssh_origin, source=config.source)
            if probe == "unavailable":
                raise InstallRefused(
                    "GitHub source access is unreachable from the rcp account. Restore DNS and "
                    "network access, then rerun without changing the recorded key."
                )
            return SourceAccess(
                config=config,
                repository=repository,
                grant_needed=probe != "ready",
                deploy_key_label=f"rcp-source:{config.installation_id}",
                public_key=public_key,
            )

        public_probe = self._probe_source(repository.https_origin, source=None)
        if public_probe == "ready":
            config = create_installed_server_config(
                source=ServerSourceConfig(
                    origin=repository.https_origin,
                    authentication="public",
                )
            )
            write_installed_server_config(config, self.layout.config_path)
            return SourceAccess(config=config, repository=repository, grant_needed=False)
        if public_probe == "unavailable":
            raise InstallRefused(
                "GitHub is unreachable from the rcp account. Restore DNS and network access, "
                "then rerun before creating a source credential."
            )
        installation_id = str(uuid.uuid4())
        label = f"rcp-source:{installation_id}"
        self._create_source_key_pair(private_path, public_path, label=label)
        public_key, fingerprint = _read_public_key(public_path)
        config = create_installed_server_config(
            installation_id=installation_id,
            source=ServerSourceConfig(
                origin=repository.ssh_origin,
                authentication="deploy_key",
                public_key_fingerprint=fingerprint,
            ),
        )
        write_installed_server_config(config, self.layout.config_path)
        return SourceAccess(
            config=config,
            repository=repository,
            grant_needed=True,
            deploy_key_label=label,
            public_key=public_key,
        )

    def converge_source_checkout(self, access: SourceAccess) -> ManagedCheckout:
        self._require_service_identity()
        self._require_no_unfinished_update()
        self._require_no_unfinished_restore()
        source = self.layout.source_checkout
        environment = self._source_environment(access.config.source)
        if not source.exists():
            if source.is_symlink():
                raise InstallRefused(
                    "The managed source path is a symlink; install will not replace it."
                )
            self._run_as_service(
                (
                    "git",
                    "clone",
                    "--branch",
                    "main",
                    "--single-branch",
                    access.config.source.origin,
                    str(source),
                ),
                environment=environment,
                timeout=SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
                capture_output=False,
                error=(
                    "The managed source clone failed. Confirm the recorded GitHub source grant "
                    "and network access, then rerun."
                ),
            )
        _require_owned_directory(source, uid=self._service_uid_value, gid=self._service_gid_value)
        git_dir = source / ".git"
        if git_dir.is_symlink() or not git_dir.is_dir():
            raise InstallRefused(
                "The managed source path is not the RCP-owned Git checkout; install will not "
                "replace or adopt it."
            )
        origin = self._git_text(source, ("remote", "get-url", "origin"), environment=environment)
        if origin != access.config.source.origin:
            raise InstallRefused(
                "The managed checkout origin differs from the installed source configuration; "
                "install will not rewrite it."
            )
        if self._git_text(
            source,
            ("status", "--porcelain", "--untracked-files=all"),
            environment=environment,
        ):
            raise InstallRefused(
                "The managed source checkout has local changes. Preserve or inspect them; install "
                "will not reset or clean this checkout."
            )
        self._run_git(
            source,
            ("fetch", "--prune", "origin", "main"),
            environment=environment,
            timeout=SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
            error="Fetching origin/main failed. Restore source access and rerun install.",
        )
        upstream = self._git_text(source, ("rev-parse", "origin/main"), environment=environment)
        _require_full_commit(upstream)
        current = self._current_release_commit()
        if current is not None:
            head = self._git_text(source, ("rev-parse", "HEAD"), environment=environment)
            if head != current:
                raise InstallRefused(
                    "The managed source HEAD does not match the current installed release. "
                    "Use server update or restore the managed checkout; install will not reset it."
                )
            if upstream != current:
                raise InstallRefused(
                    "GitHub origin/main differs from the installed commit. Version changes belong "
                    "to rcp server update; install will not switch releases."
                )
            return ManagedCheckout(commit=current, is_current_release=True)
        self._run_git(
            source,
            ("checkout", "--force", "-B", "main", "origin/main"),
            environment=environment,
            timeout=SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
            error="The clean managed checkout could not select origin/main. Inspect it and rerun.",
        )
        return ManagedCheckout(commit=upstream, is_current_release=False)

    def build_release(self, checkout: ManagedCheckout) -> Path:
        self._require_service_identity()
        release = self.layout.release_dir(checkout.commit)
        if release.exists() or release.is_symlink():
            _require_owned_directory(
                release,
                uid=self._service_uid_value,
                gid=self._service_gid_value,
            )
            head = self._git_text(release, ("rev-parse", "HEAD"))
            if head != checkout.commit:
                raise InstallRefused(
                    "The per-commit release path contains a different Git commit; install will "
                    "not replace it."
                )
            if self._git_text(release, ("status", "--porcelain", "--untracked-files=all")):
                raise InstallRefused(
                    "The per-commit release worktree has tracked or untracked changes; install "
                    "will not clean it."
                )
        else:
            self._run_as_service(
                (
                    "git",
                    "-C",
                    str(self.layout.source_checkout),
                    "worktree",
                    "add",
                    "--detach",
                    str(release),
                    checkout.commit,
                ),
                timeout=SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
                capture_output=False,
                error="Creating the clean per-commit release worktree failed. Inspect Git and rerun.",
            )
        if checkout.is_current_release:
            self._validate_release_artifacts(release)
            return release
        self._run_as_service(
            ("npm", "--prefix", "web", "ci"),
            cwd=release,
            timeout=SERVER_INSTALL_BUILD_TIMEOUT_SECONDS,
            capture_output=False,
            error="npm --prefix web ci failed in the managed release. Fix the source and rerun.",
        )
        self._run_as_service(
            ("npm", "--prefix", "web", "run", "build"),
            cwd=release,
            timeout=SERVER_INSTALL_BUILD_TIMEOUT_SECONDS,
            capture_output=False,
            error="npm --prefix web run build failed in the managed release. Fix the source and rerun.",
        )
        self._run_as_service(
            ("uv", "sync", "--frozen"),
            cwd=release,
            environment={"UV_MANAGED_PYTHON": "1", "UV_PYTHON": "3.12"},
            timeout=SERVER_INSTALL_BUILD_TIMEOUT_SECONDS,
            capture_output=False,
            error="uv sync --frozen failed in the managed release. Fix the lock or runtime and rerun.",
        )
        self._validate_release_artifacts(release)
        return release

    def install_service(
        self,
        checkout: ManagedCheckout,
        release: Path,
    ) -> ServiceInstallState:
        self._require_service_identity()
        self._require_no_unfinished_update()
        self._require_no_unfinished_restore()
        if release != self.layout.release_dir(checkout.commit):
            raise InstallRefused("The built release path does not match its exact commit.")
        _install_root_file(
            self.layout.cli_wrapper,
            _wrapper_text(self.layout),
            mode=_WRAPPER_MODE,
        )
        _install_root_file(
            self.layout.systemd_unit,
            server_service_unit_text(),
            mode=_UNIT_MODE,
        )
        from rcp.server_ops.backup_config import (
            backup_configuration_lock,
            backup_service_unit_text,
            recover_pending_backup_configuration,
            render_backup_timer_unit,
        )

        with backup_configuration_lock(self.layout):
            recover_pending_backup_configuration(self.layout)
            installed_config = load_installed_server_config(self.layout.config_path)
            backup_schedule = (
                installed_config.backup.schedule if installed_config.backup is not None else None
            )
            fence_backup_timer_before_unit_change()
            install_backup_unit_files(
                service_content=backup_service_unit_text(),
                timer_content=render_backup_timer_unit(backup_schedule),
                layout=self.layout,
            )
            _converge_current_release(self.layout, release)
            _require_command(
                ("systemctl", "daemon-reload"),
                "systemd could not reload the installed unit. Run systemctl daemon-reload, "
                "inspect the unit, and rerun install.",
                timeout=SERVER_INSTALL_SERVICE_TIMEOUT_SECONDS,
            )
            _fence_service_stopped_disabled("rcp-backup.timer")
        data_state = self._data_state()
        if data_state == "fresh":
            _fence_service_stopped_disabled(self.layout.service_unit_name)
            return ServiceInstallState(data_state="fresh", service_state="stopped_disabled")
        active = _read_systemd_property(self.layout.service_unit_name, "ActiveState")
        state = "active" if active == "active" else "initialized_stopped"
        return ServiceInstallState(data_state="initialized", service_state=state)

    def _require_no_unfinished_update(self) -> None:
        from rcp.server_ops.update_checkpoint import (
            UpdateCheckpointRefused,
            unfinished_rollback_journals,
        )
        from rcp.server_ops.update_cutover import (
            UpdateCutoverRefused,
            update_operation_needing_recovery,
        )

        self._require_service_identity()
        assert self._service_uid is not None
        try:
            pending = update_operation_needing_recovery(
                self.layout.update_checkpoints_root,
                expected_uid=self._service_uid,
            )
            journals = unfinished_rollback_journals(
                self.layout.update_checkpoints_root,
                expected_uid=self._service_uid,
            )
        except (OSError, UpdateCheckpointRefused, UpdateCutoverRefused) as exc:
            with suppress(InstalledServiceControlRefused):
                InstalledSystemServiceController(self.layout).stop()
            raise InstallRefused(
                "Update recovery state is unsafe. RCP kept the service stopped; run sudo rcp "
                "server update to resume the exact durable operation."
            ) from exc
        if pending is None and not journals:
            return
        with suppress(InstalledServiceControlRefused):
            InstalledSystemServiceController(self.layout).stop()
        raise InstallRefused(
            "An unfinished source update or rollback blocks install. RCP kept the service "
            "stopped; run sudo rcp server update to resume it."
        )

    def _require_no_unfinished_restore(self) -> None:
        from rcp.server_ops.restore import RestoreRefused, unfinished_restore_operation

        self._require_service_identity()
        assert self._service_uid is not None
        try:
            pending = unfinished_restore_operation(
                self.layout,
                expected_uid=self._service_uid,
            )
        except (OSError, RestoreRefused) as exc:
            with suppress(InstalledServiceControlRefused):
                InstalledSystemServiceController(self.layout).fence_stopped_disabled()
            raise InstallRefused(
                "Restore recovery state is unsafe. RCP kept the service stopped; preserve the "
                "restore operations root and rerun sudo rcp server restore."
            ) from exc
        if pending is None:
            return
        with suppress(InstalledServiceControlRefused):
            InstalledSystemServiceController(self.layout).fence_stopped_disabled()
        raise InstallRefused(
            "An unfinished replacement restore blocks install. RCP kept the service stopped; "
            "re-enter sudo rcp server restore with its exact archive and identity."
        )

    def activate_and_verify(self) -> ServiceHealth:
        self._require_no_unfinished_update()
        self._require_no_unfinished_restore()
        _require_command(
            ("systemctl", "enable", "--now", self.layout.service_unit_name),
            "systemd could not enable and start rcp.service. Run systemctl status --no-pager "
            "rcp.service, correct the reported machine issue, and rerun install.",
            timeout=SERVER_INSTALL_SERVICE_TIMEOUT_SECONDS,
        )
        enabled = _read_systemd_property(self.layout.service_unit_name, "UnitFileState")
        active = _read_systemd_property(self.layout.service_unit_name, "ActiveState")
        if enabled != "enabled" or active != "active":
            raise InstallRefused(
                "rcp.service is not both enabled and active. Run systemctl status --no-pager "
                "rcp.service, correct the reported issue, and rerun install."
            )
        health = _read_team_health()
        if health is None:
            raise InstallRefused(
                "The enabled service did not return valid loopback health within 15 seconds. Run "
                "systemctl status --no-pager rcp.service and curl --fail --silent "
                "http://127.0.0.1:8421/api/health, then rerun install."
            )
        if health.get("status") != "ok" or health.get("space_kind") != "team":
            _fence_service_stopped_disabled(self.layout.service_unit_name)
            raise InstallRefused(
                "Loopback health did not identify an RCP team space, so RCP stopped and disabled "
                "the service. Restore the intended owned team data before rerunning."
            )
        space_name = health.get("space_name")
        if not isinstance(space_name, str) or not space_name.strip():
            raise InstallRefused(
                "Loopback health omitted the team-space name. Inspect the service and rerun install."
            )
        if self.layout.config_path.exists() or self.layout.config_path.is_symlink():
            from rcp.server_ops.backup_config import (
                BackupConfigurationRefused,
                activate_configured_backup_timer,
            )

            try:
                activate_configured_backup_timer(self.layout)
            except BackupConfigurationRefused as exc:
                raise InstallRefused(str(exc)) from exc
        return ServiceHealth(status="ok", space_kind="team", space_name=space_name)

    def _converge_account(self) -> pwd.struct_passwd:
        try:
            account = pwd.getpwnam(self.layout.service_account)
        except KeyError:
            account_creation = _run_process(
                (
                    "useradd",
                    "--create-home",
                    "--home-dir",
                    str(self.layout.service_home),
                    "--shell",
                    "/bin/bash",
                    "--user-group",
                    "--password",
                    "*NP*",
                    self.layout.service_account,
                ),
                timeout=SERVER_INSTALL_ACCOUNT_TIMEOUT_SECONDS,
            )
            if account_creation.returncode != 0:
                if account_creation.stderr == "command timed out":
                    raise InstallRefused(
                        "Creating the dedicated rcp account did not finish within five minutes. "
                        "Inspect useradd, NSS, and home-directory policy, then rerun."
                    ) from None
                raise InstallRefused(
                    "Creating the dedicated rcp account failed. Inspect useradd policy and rerun."
                ) from None
            try:
                account = pwd.getpwnam(self.layout.service_account)
            except KeyError as exc:  # pragma: no cover - broken NSS after successful useradd
                raise InstallRefused(
                    "useradd returned success but the rcp account is unavailable."
                ) from exc
        if account.pw_uid == 0 or account.pw_gid == 0:
            raise InstallRefused(
                "The rcp account has root user or group identity. Replace it with a dedicated "
                "unprivileged rcp account, then rerun install."
            )
        if account.pw_dir != str(self.layout.service_home) or account.pw_shell != "/bin/bash":
            raise InstallRefused(
                "The existing rcp account does not use exact home /home/rcp and shell /bin/bash. "
                "Install will not rewrite the account."
            )
        try:
            primary_group = grp.getgrgid(account.pw_gid)
        except KeyError as exc:
            raise InstallRefused("The rcp account has no resolvable primary group.") from exc
        if primary_group.gr_name != self.layout.service_account:
            raise InstallRefused(
                "The existing rcp account does not use its dedicated rcp primary group. Install "
                "will not rewrite the account."
            )
        shadow = _require_command(
            ("getent", "shadow", self.layout.service_account),
            "The rcp shadow entry could not be read. Run install as root and inspect NSS.",
        ).stdout.strip()
        parts = shadow.split(":")
        if len(parts) < 2 or parts[0] != self.layout.service_account or parts[1] != "*NP*":
            raise InstallRefused(
                "The rcp account must have exact unusable non-locking shadow value *NP*. Install "
                "will not change an existing password state."
            )
        supplemental = sorted(
            group.gr_name
            for group in grp.getgrall()
            if group.gr_gid != account.pw_gid and self.layout.service_account in group.gr_mem
        )
        if supplemental:
            raise InstallRefused(
                "The rcp account has supplemental groups. Remove all supplemental memberships "
                "after reviewing them, then rerun install."
            )
        sudo_policy = _run_process(
            ("sudo", "-n", "-U", account.pw_name, "-l"),
            environment={"LANG": "C", "LC_ALL": "C"},
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
        )
        diagnostic = f"{sudo_policy.stdout}\n{sudo_policy.stderr}".lower()
        if "not allowed to run sudo" in diagnostic:
            return account
        if sudo_policy.returncode == 0:
            raise InstallRefused(
                "The rcp account has sudo authority. Remove every sudo grant for rcp and "
                "rerun install."
            )
        raise InstallRefused(
            "RCP could not prove that the rcp account has no sudo authority. Run "
            "sudo -U rcp -l as root, correct the sudo or NSS error, and rerun install."
        )

    def _validate_service_tooling(self) -> None:
        account = pwd.getpwnam(self.layout.service_account)
        checks = (
            (("git", "--version"), "Git is not executable as rcp."),
            (("node", "--version"), "Node.js is not executable as rcp."),
            (("npm", "--version"), "npm is not executable as rcp."),
            (("ssh", "-V"), "SSH is not executable as rcp."),
            (("uv", "--version"), "uv is not executable as rcp."),
            (("age", "--version"), "age is not executable as rcp."),
        )
        for argv, message in checks:
            result = _run_as_account(
                account,
                argv,
                timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise InstallRefused(
                    f"{message} Install the tool for all users and rerun the same command."
                )
        python = _run_as_account(
            account,
            (
                "uv",
                "python",
                "find",
                "--managed-python",
                "--no-python-downloads",
                "3.12",
            ),
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
        )
        if python.returncode != 0:
            installed = _run_as_account(
                account,
                ("uv", "python", "install", "--managed-python", "--no-progress", "3.12"),
                timeout=SERVER_INSTALL_SOURCE_TIMEOUT_SECONDS,
                capture_output=False,
            )
            if installed.returncode != 0:
                raise InstallRefused(
                    "uv could not install the managed Python 3.12 runtime for rcp. Run the same "
                    "uv python install command as rcp from /home/rcp, correct the reported host "
                    "or network issue, and rerun install."
                )
            python = _run_as_account(
                account,
                (
                    "uv",
                    "python",
                    "find",
                    "--managed-python",
                    "--no-python-downloads",
                    "3.12",
                ),
                timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            )
        if python.returncode != 0:
            raise InstallRefused(
                "uv installed Python 3.12 but could not find it again as rcp. Inspect "
                "/home/rcp ownership and rerun the same command."
            )
        python_path = Path(python.stdout.strip())
        if not python_path.is_absolute():
            raise InstallRefused("uv returned a non-absolute Python 3.12 runtime path for rcp.")
        version = _run_as_account(
            account,
            (str(python_path), "--version"),
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
        )
        if version.returncode != 0 or not (version.stdout or version.stderr).startswith(
            "Python 3.12."
        ):
            raise InstallRefused(
                "The uv-selected service runtime is not Python 3.12. Install the correct runtime "
                "for rcp and rerun."
            )

    def _probe_source(
        self,
        origin: str,
        *,
        source: ServerSourceConfig | None,
    ) -> Literal["ready", "grant_needed", "unavailable"]:
        result = self._run_as_service(
            ("git", "ls-remote", "--exit-code", origin, "refs/heads/main"),
            environment=self._source_environment(source),
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode == 0:
            return "ready"
        diagnostic = f"{result.stdout}\n{result.stderr}".lower()
        if any(marker in diagnostic for marker in _NETWORK_FAILURE_MARKERS):
            return "unavailable"
        if result.returncode == 2:
            raise InstallRefused(
                "The configured GitHub repository has no readable main branch. Create or restore "
                "origin/main, then rerun install."
            )
        if any(marker in diagnostic for marker in _AUTH_FAILURE_MARKERS):
            return "grant_needed"
        raise InstallRefused(
            "The GitHub source probe failed without a recognized authentication or network "
            "diagnostic. Run the same git ls-remote as rcp, correct the host issue, and rerun."
        )

    def _create_source_key_pair(self, private: Path, public: Path, *, label: str) -> None:
        if private.exists() or private.is_symlink() or public.exists() or public.is_symlink():
            raise InstallRefused(
                "A source key exists without installed source configuration. Preserve and inspect "
                "it; install will not adopt or replace it."
            )
        self._run_as_service(
            ("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", label, "-f", str(private)),
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            capture_output=False,
            error="Creating the dedicated source key failed. Inspect the credential path and rerun.",
        )
        os.chmod(private, _PRIVATE_KEY_MODE)
        os.chmod(public, _PUBLIC_KEY_MODE)

    def _validate_source_key_pair(
        self,
        config: InstalledServerConfig,
        private: Path,
        public: Path,
    ) -> str:
        _require_owned_file(
            private,
            uid=self._service_uid_value,
            gid=self._service_gid_value,
            mode=_PRIVATE_KEY_MODE,
            label="source private key",
        )
        _require_owned_file(
            public,
            uid=self._service_uid_value,
            gid=self._service_gid_value,
            mode=_PUBLIC_KEY_MODE,
            label="source public key",
        )
        public_key, fingerprint = _read_public_key(public)
        if fingerprint != config.source.public_key_fingerprint:
            raise InstallRefused(
                "The source public key fingerprint differs from the installed configuration. "
                "Install will not replace either side."
            )
        derived = self._run_as_service(
            ("ssh-keygen", "-y", "-f", str(private)),
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            error="The source private key could not derive its public key. Install will not replace it.",
        ).stdout.strip()
        if derived.split()[:2] != public_key.split()[:2]:
            raise InstallRefused(
                "The installed source private and public keys are not one key pair. Install will "
                "not replace either file."
            )
        return public_key

    def _source_environment(self, source: ServerSourceConfig | None) -> dict[str, str]:
        return source_git_environment(source, self.layout)

    def _run_as_service(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
        timeout: float,
        error: str | None = None,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        account = pwd.getpwnam(self.layout.service_account)
        result = _run_as_account(
            account,
            argv,
            cwd=cwd,
            environment=environment,
            timeout=timeout,
            capture_output=capture_output,
        )
        if check and result.returncode != 0:
            raise InstallRefused(error or "A managed command failed; inspect the host and rerun.")
        return result

    def _run_git(
        self,
        root: Path,
        argv: tuple[str, ...],
        *,
        environment: dict[str, str] | None = None,
        timeout: float = SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
        error: str,
    ) -> None:
        self._run_as_service(
            ("git", "-C", str(root), *argv),
            environment=environment,
            timeout=timeout,
            capture_output=False,
            error=error,
        )

    def _git_text(
        self,
        root: Path,
        argv: tuple[str, ...],
        *,
        environment: dict[str, str] | None = None,
    ) -> str:
        result = self._run_as_service(
            ("git", "-C", str(root), *argv),
            environment=environment,
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            error="The managed Git checkout could not be validated. Inspect it and rerun.",
        )
        return result.stdout.strip()

    def _current_release_commit(self) -> str | None:
        current = self.layout.current_release
        if not current.exists() and not current.is_symlink():
            return None
        if not current.is_symlink():
            raise InstallRefused(
                "The current release path is not a symlink; install will not replace it."
            )
        info = current.lstat()
        if info.st_uid != 0 or info.st_gid != 0:
            raise InstallRefused("The current release symlink is not owned by root.")
        target = Path(os.readlink(current))
        if not target.is_absolute() or target.parent != self.layout.releases_root:
            raise InstallRefused(
                "The current release symlink does not target the fixed releases root."
            )
        commit = target.name
        _require_full_commit(commit)
        if target != self.layout.release_dir(commit):
            raise InstallRefused("The current release symlink target is not canonical.")
        if target.is_symlink() or not target.is_dir():
            raise InstallRefused(
                "The current release symlink target is missing or not a directory. Restore the "
                "known release; install will not reconstruct active state."
            )
        return commit

    def _validate_release_artifacts(self, release: Path) -> None:
        executable = release / ".venv" / "bin" / "rcp"
        python = release / ".venv" / "bin" / "python"
        web_index = release / "web" / "dist" / "index.html"
        for path, label in ((executable, "Python entry point"), (web_index, "Web build")):
            if path.is_symlink() or not path.is_file():
                raise InstallRefused(
                    f"The current release is missing its {label}. Version repair belongs to "
                    "server update; install will not mutate the active release."
                )
            info = path.stat()
            if (info.st_uid, info.st_gid) != (self._service_uid_value, self._service_gid_value):
                raise InstallRefused(
                    f"The current release {label} has unexpected ownership. Install will not "
                    "adopt or replace active artifacts."
                )
        if not python.exists() or not python.is_file():
            raise InstallRefused(
                "The current release is missing its Python runtime. Version repair belongs to "
                "server update; install will not mutate the active release."
            )
        version = self._run_as_service(
            (str(python), "--version"),
            timeout=SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
            error="The current release Python runtime could not execute.",
        )
        if not (version.stdout or version.stderr).startswith("Python 3.12."):
            raise InstallRefused(
                "The current release does not use the required Python 3.12 runtime."
            )

    def _data_state(self) -> Literal["fresh", "initialized"]:
        _require_owned_directory(
            self.layout.data_dir,
            uid=self._service_uid_value,
            gid=self._service_gid_value,
        )
        database = self.layout.data_dir / "rcp.sqlite3"
        if database.exists() or database.is_symlink():
            if database.is_symlink() or not database.is_file():
                raise InstallRefused("The RCP database path is not a regular owned file.")
            info = database.stat()
            if (info.st_uid, info.st_gid) != (self._service_uid_value, self._service_gid_value):
                raise InstallRefused("The RCP database is not owned by the dedicated rcp account.")
            if stat.S_IMODE(info.st_mode) & 0o077:
                raise InstallRefused(
                    "The RCP database is readable or writable outside the rcp account."
                )
            return "initialized"
        if any(self.layout.data_dir.iterdir()):
            raise InstallRefused(
                "The data directory has files but no initialized RCP database. Install will not "
                "remove or adopt unknown data."
            )
        return "fresh"

    def _require_service_identity(self) -> None:
        if self._service_uid is None or self._service_gid is None:
            account = pwd.getpwnam(self.layout.service_account)
            self._service_uid = account.pw_uid
            self._service_gid = account.pw_gid

    @property
    def _service_uid_value(self) -> int:
        self._require_service_identity()
        assert self._service_uid is not None
        return self._service_uid

    @property
    def _service_gid_value(self) -> int:
        self._require_service_identity()
        assert self._service_gid is not None
        return self._service_gid


def source_git_environment(
    source: ServerSourceConfig | None,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> dict[str, str]:
    """Return the one credential-isolated Git environment shared by install and update."""

    environment = {
        "GIT_ASKPASS": "/bin/false",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_KEY_1": "core.askPass",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_VALUE_1": "/bin/false",
        "GIT_TERMINAL_PROMPT": "0",
        "SSH_ASKPASS": "/bin/false",
    }
    if source is None or source.authentication == "public":
        return environment
    private = layout.credentials_root / _SOURCE_PRIVATE_KEY
    command = (
        f"ssh -F /dev/null -i {private} -o IdentitiesOnly=yes -o BatchMode=yes "
        f"-o StrictHostKeyChecking=yes -o GlobalKnownHostsFile=/dev/null "
        f"-o UserKnownHostsFile={layout.ssh_state_root / 'known_hosts'}"
    )
    return {**environment, "GIT_SSH_COMMAND": command}


def _read_os_release(path: Path) -> dict[str, str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstallRefused(
            "Ubuntu release metadata could not be read from /etc/os-release."
        ) from exc
    values: dict[str, str] = {}
    for line in content.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if value.startswith(('"', "'")) and value.endswith(value[:1]):
            value = value[1:-1]
        values[name] = value
    return values


def _leading_major(value: str) -> int:
    match = re.search(r"v?(\d+)", value.strip())
    if match is None:
        raise InstallRefused("A required tool returned an unrecognized version.")
    return int(match.group(1))


def _major_minor(value: str) -> tuple[int, int]:
    match = _VERSION.search(value)
    if match is None:
        raise InstallRefused("A required tool returned an unrecognized semantic version.")
    return int(match.group("major")), int(match.group("minor"))


def _require_full_commit(value: str) -> None:
    if _FULL_GIT_COMMIT.fullmatch(value) is None:
        raise InstallRefused("Git returned a non-canonical commit id; install stopped safely.")


def _run_process(
    argv: tuple[str, ...],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    timeout: float,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_environment = os.environ.copy()
    for name in ("SUDO_COMMAND", "SUDO_GID", "SUDO_UID", "SUDO_USER"):
        merged_environment.pop(name, None)
    if environment:
        merged_environment.update(environment)
    try:
        output = (
            {"capture_output": True}
            if capture_output
            else {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
        )
        return subprocess.run(
            argv,
            cwd=cwd,
            env=merged_environment,
            text=True,
            timeout=timeout,
            check=False,
            **output,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 126, "", "command timed out")
    except OSError:
        return subprocess.CompletedProcess(argv, 126, "", "command could not be executed")


def _require_command(
    argv: tuple[str, ...],
    error: str,
    *,
    timeout: float = SERVER_INSTALL_PROBE_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    result = _run_process(argv, timeout=timeout)
    if result.returncode != 0:
        raise InstallRefused(error)
    return result


def _read_systemd_property(unit: str, property_name: str) -> str:
    result = _require_command(
        ("systemctl", "show", f"--property={property_name}", "--value", unit),
        f"systemd could not read {property_name} for {unit}. Inspect the unit and rerun install.",
        timeout=SERVER_INSTALL_SERVICE_TIMEOUT_SECONDS,
    )
    value = result.stdout.strip()
    if not value or "\n" in value:
        raise InstallRefused(
            f"systemd returned an invalid {property_name} for {unit}. Inspect the unit and "
            "rerun install."
        )
    return value


def _fence_service_stopped_disabled(unit: str) -> None:
    _require_command(
        ("systemctl", "disable", "--now", unit),
        f"systemd could not stop and disable {unit}. Run systemctl disable --now {unit}, "
        "inspect the failure, and rerun install.",
        timeout=SERVER_INSTALL_SERVICE_TIMEOUT_SECONDS,
    )
    active = _read_systemd_property(unit, "ActiveState")
    enabled = _read_systemd_property(unit, "UnitFileState")
    if active != "inactive" or enabled != "disabled":
        raise InstallRefused(
            f"RCP could not prove {unit} is stopped and disabled: ActiveState={active}, "
            f"UnitFileState={enabled}. Run systemctl disable --now {unit}, verify both states, "
            "and rerun install."
        )


def _run_as_account(
    account: pwd.struct_passwd,
    argv: tuple[str, ...],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    timeout: float,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    explicit_environment = {
        "HOME": account.pw_dir,
        "USER": account.pw_name,
        "LOGNAME": account.pw_name,
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if environment:
        explicit_environment.update(environment)
    env_argv = tuple(f"{name}={value}" for name, value in explicit_environment.items())
    return _run_process(
        ("runuser", "--user", account.pw_name, "--", "env", "-i", *env_argv, *argv),
        cwd=cwd or Path(account.pw_dir),
        timeout=timeout,
        capture_output=capture_output,
    )


def _converge_directory(path: Path, *, uid: int, gid: int, mode: int) -> None:
    _reject_symlink_ancestry(path.parent)
    if path.is_symlink():
        raise InstallRefused(f"Managed directory {path} is a symlink; install will not follow it.")
    if not path.exists():
        try:
            path.mkdir()
            os.chown(path, uid, gid)
            os.chmod(path, mode)
        except OSError as exc:
            raise InstallRefused(f"Managed directory {path} could not be created safely.") from exc
        return
    _require_owned_directory(path, uid=uid, gid=gid)
    if stat.S_IMODE(path.stat().st_mode) != mode:
        try:
            os.chmod(path, mode)
        except OSError as exc:
            raise InstallRefused(
                f"Managed directory {path} could not be set to its exact mode."
            ) from exc


def _require_owned_directory(path: Path, *, uid: int, gid: int) -> None:
    _reject_symlink_ancestry(path.parent)
    if path.is_symlink() or not path.is_dir():
        raise InstallRefused(f"Managed path {path} is not a regular directory.")
    info = path.stat()
    if (info.st_uid, info.st_gid) != (uid, gid):
        raise InstallRefused(f"Managed directory {path} has unexpected ownership.")


def _require_owned_file(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
    label: str,
) -> None:
    _reject_symlink_ancestry(path.parent)
    if path.is_symlink() or not path.is_file():
        raise InstallRefused(f"The {label} is not a regular file.")
    info = path.stat()
    if (info.st_uid, info.st_gid) != (uid, gid) or stat.S_IMODE(info.st_mode) != mode:
        raise InstallRefused(f"The {label} has unexpected ownership or mode.")


def _reject_symlink_ancestry(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise InstallRefused(f"Managed path ancestry contains a symlink at {candidate}.")


def _read_public_key(path: Path) -> tuple[str, str]:
    try:
        if path.stat().st_size > _PUBLIC_KEY_MAX_BYTES:
            raise ValueError
        public_key = path.read_text(encoding="utf-8").strip()
        parts = public_key.split()
        if len(parts) < 2 or parts[0] != "ssh-ed25519":
            raise ValueError
        blob = base64.b64decode(parts[1], validate=True)
    except (OSError, UnicodeError, ValueError) as exc:
        raise InstallRefused("The source public key is not a valid OpenSSH Ed25519 key.") from exc
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode(
        "ascii"
    ).rstrip("=")
    if _OPENSSH_FINGERPRINT.fullmatch(fingerprint) is None:
        raise InstallRefused("The source public-key fingerprint is not canonical.")
    return public_key, fingerprint


def _wrapper_text(layout: ServerLayout) -> str:
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        "umask 077\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        f"export RCP_DATA_DIR={layout.data_dir}\n"
        f'exec {layout.current_release}/.venv/bin/rcp "$@"\n'
    )


def _install_root_file(
    path: Path,
    content: str,
    *,
    mode: int,
    replace_existing: bool = False,
) -> None:
    _reject_symlink_ancestry(path.parent)
    if path.is_symlink():
        raise InstallRefused(f"Root-managed file {path} is a symlink; install will not replace it.")
    if path.exists():
        if not path.is_file():
            raise InstallRefused(f"Root-managed path {path} is not a regular file.")
        info = path.stat()
        if (info.st_uid, info.st_gid) != (0, 0):
            raise InstallRefused(f"Root-managed file {path} has unexpected ownership.")
        encoded_size = len(content.encode("utf-8"))
        differs = (
            stat.S_IMODE(info.st_mode) != mode
            or info.st_size != encoded_size
            or path.read_text(encoding="utf-8") != content
        )
        if not differs:
            return
        if not replace_existing:
            raise InstallRefused(
                f"Root-managed file {path} differs from this release. Use server update or "
                "restore the known file; install will not overwrite it."
            )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise InstallRefused(
            f"Root-managed file {path} could not be installed atomically."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def install_backup_unit_files(
    *,
    service_content: str,
    timer_content: str,
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
) -> None:
    unit_root = layout.systemd_unit.parent
    _install_root_file(
        unit_root / "rcp-backup.service",
        service_content,
        mode=_UNIT_MODE,
    )
    _install_root_file(
        unit_root / "rcp-backup.timer",
        timer_content,
        mode=_UNIT_MODE,
        replace_existing=True,
    )


def fence_backup_timer_before_unit_change() -> None:
    load_state = _read_systemd_property("rcp-backup.timer", "LoadState")
    if load_state == "not-found":
        return
    _fence_service_stopped_disabled("rcp-backup.timer")


def reload_and_disable_backup_timer() -> None:
    _require_command(
        ("systemctl", "daemon-reload"),
        "systemd could not reload the backup units. Run systemctl daemon-reload, inspect "
        "rcp-backup.service and rcp-backup.timer, then rerun backup configure.",
        timeout=SERVER_INSTALL_SERVICE_TIMEOUT_SECONDS,
    )
    _fence_service_stopped_disabled("rcp-backup.timer")


def run_backup_service_once() -> None:
    """Run one first backup while the timer remains fenced off."""

    _require_command(
        ("systemctl", "start", "rcp-backup.service"),
        "The first protected backup failed. The timer remains disabled; inspect systemctl "
        "status --no-pager rcp-backup.service and backup-status.json, then rerun backup configure.",
        timeout=SERVER_INSTALL_BUILD_TIMEOUT_SECONDS,
    )
    result = _read_systemd_property("rcp-backup.service", "Result")
    if result != "success":
        raise InstallRefused(
            "The first protected backup did not report a successful systemd result. The timer "
            "remains disabled; inspect rcp-backup.service and rerun backup configure."
        )


def enable_backup_timer() -> None:
    """Enable the rendered timer and prove the loaded systemd state."""

    _require_command(
        ("systemctl", "enable", "--now", "rcp-backup.timer"),
        "systemd could not enable the verified backup timer. Inspect rcp-backup.timer and rerun "
        "backup configure.",
        timeout=SERVER_INSTALL_SERVICE_TIMEOUT_SECONDS,
    )
    active, enabled = read_systemd_unit_state("rcp-backup.timer")
    if active != "active" or enabled != "enabled":
        raise InstallRefused(
            "The verified backup timer is not both active and enabled. Disable it, inspect the "
            "unit, and rerun backup configure."
        )


def read_systemd_unit_state(unit: str) -> tuple[str, str]:
    return _read_systemd_property(unit, "ActiveState"), _read_systemd_property(
        unit, "UnitFileState"
    )


class InstalledServiceControlRefused(RuntimeError):
    """The narrow installed-service stop/start or pointer fence failed."""


class InstalledSystemServiceController:
    """Root-only stop/switch/start seam shared by install recovery and update."""

    def __init__(
        self,
        layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
        *,
        runner=None,
        root_identity: tuple[int, int] = (0, 0),
    ) -> None:
        self.layout = layout
        self.runner = runner or (
            lambda argv: _run_process(
                argv,
                timeout=SERVER_INSTALL_SERVICE_TIMEOUT_SECONDS,
            )
        )
        self.root_uid, self.root_gid = root_identity

    def current_release(self) -> Path:
        current = self.layout.current_release
        try:
            metadata = current.lstat()
            target = Path(os.readlink(current))
        except OSError as exc:
            raise InstalledServiceControlRefused(
                "The installed current release pointer is unavailable."
            ) from exc
        if (
            not stat.S_ISLNK(metadata.st_mode)
            or (metadata.st_uid, metadata.st_gid) != (self.root_uid, self.root_gid)
            or not target.is_absolute()
            or target.parent != self.layout.releases_root
        ):
            raise InstalledServiceControlRefused(
                "The installed current release pointer is unsafe or unowned."
            )
        try:
            if self.layout.release_dir(target.name) != target:
                raise ValueError
        except ValueError as exc:
            raise InstalledServiceControlRefused(
                "The installed current pointer does not name one exact release commit."
            ) from exc
        return target

    def stop(self) -> None:
        self._command(("systemctl", "stop", self.layout.service_unit_name))
        if self._property("ActiveState") != "inactive" or self._property("MainPID") != "0":
            raise InstalledServiceControlRefused(
                "systemd did not prove the RCP service stopped with no main process."
            )

    def fence_stopped_disabled(self) -> None:
        self._command(("systemctl", "disable", "--now", self.layout.service_unit_name))
        if (
            self._property("ActiveState") != "inactive"
            or self._property("MainPID") != "0"
            or self._property("UnitFileState") != "disabled"
        ):
            raise InstalledServiceControlRefused(
                "systemd did not prove the RCP service stopped and disabled with no main process."
            )

    def start(self) -> int:
        self._command(("systemctl", "start", self.layout.service_unit_name))
        active = self._property("ActiveState")
        main_pid = self._property("MainPID")
        try:
            pid = int(main_pid)
        except ValueError as exc:
            raise InstalledServiceControlRefused(
                "systemd returned an invalid RCP main process identity."
            ) from exc
        if active != "active" or pid <= 0:
            raise InstalledServiceControlRefused(
                "systemd did not prove the RCP service started with one main process."
            )
        return pid

    def enable_and_start(self) -> int:
        self._command(("systemctl", "enable", "--now", self.layout.service_unit_name))
        if self._property("UnitFileState") != "enabled":
            raise InstalledServiceControlRefused(
                "systemd did not prove the RCP service enabled for replacement activation."
            )
        active = self._property("ActiveState")
        main_pid = self._property("MainPID")
        try:
            pid = int(main_pid)
        except ValueError as exc:
            raise InstalledServiceControlRefused(
                "systemd returned an invalid RCP replacement process identity."
            ) from exc
        if active != "active" or pid <= 0:
            raise InstalledServiceControlRefused(
                "systemd did not prove the RCP replacement service active."
            )
        return pid

    def enable(self) -> int:
        """Enable an already-running replacement without restarting it."""

        self._command(("systemctl", "enable", self.layout.service_unit_name))
        if self._property("UnitFileState") != "enabled":
            raise InstalledServiceControlRefused(
                "systemd did not prove the active RCP replacement enabled."
            )
        active = self._property("ActiveState")
        main_pid = self._property("MainPID")
        try:
            pid = int(main_pid)
        except ValueError as exc:
            raise InstalledServiceControlRefused(
                "systemd returned an invalid active RCP process identity."
            ) from exc
        if active != "active" or pid <= 0:
            raise InstalledServiceControlRefused(
                "systemd did not prove the enabled RCP replacement remained active."
            )
        return pid

    def switch_current(self, *, expected: Path, target: Path) -> None:
        expected = expected.resolve(strict=False)
        target = target.resolve(strict=False)
        if self.current_release() != expected:
            raise InstalledServiceControlRefused(
                "The installed current release changed before the atomic switch."
            )
        for release in (expected, target):
            try:
                if (
                    self.layout.release_dir(release.name) != release
                    or not stat.S_ISDIR(release.lstat().st_mode)
                    or release.is_symlink()
                ):
                    raise ValueError
            except (OSError, ValueError) as exc:
                raise InstalledServiceControlRefused(
                    "A release pointer target is missing, unsafe, or outside the release root."
                ) from exc
        current = self.layout.current_release
        temporary = current.parent / f".{current.name}.update-{uuid.uuid4().hex}"
        try:
            os.symlink(target, temporary)
            os.lchown(temporary, self.root_uid, self.root_gid)
            os.replace(temporary, current)
            _fsync_directory(current.parent)
        except OSError as exc:
            raise InstalledServiceControlRefused(
                "The installed current release could not be switched atomically."
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        if self.current_release() != target:
            raise InstalledServiceControlRefused(
                "The installed current release switch did not survive readback."
            )

    def _property(self, name: str) -> str:
        completed = self.runner(
            (
                "systemctl",
                "show",
                f"--property={name}",
                "--value",
                self.layout.service_unit_name,
            )
        )
        value = completed.stdout.strip()
        if completed.returncode != 0 or not value or "\n" in value:
            raise InstalledServiceControlRefused(f"systemd could not read the RCP service {name}.")
        return value

    def _command(self, argv: tuple[str, ...]) -> None:
        completed = self.runner(argv)
        if completed.returncode != 0:
            raise InstalledServiceControlRefused(
                "systemd refused the bounded RCP service lifecycle command."
            )


def _converge_current_release(layout: ServerLayout, release: Path) -> None:
    current = layout.current_release
    _reject_symlink_ancestry(current.parent)
    if current.exists() or current.is_symlink():
        if not current.is_symlink():
            raise InstallRefused(
                "The current release path is not a symlink; install will not replace it."
            )
        info = current.lstat()
        if (info.st_uid, info.st_gid) != (0, 0) or Path(os.readlink(current)) != release:
            raise InstallRefused(
                "The current release pointer names another or unowned release. Version changes "
                "belong to server update."
            )
        return
    temporary = current.parent / f".{current.name}.{uuid.uuid4().hex}"
    try:
        os.symlink(release, temporary)
        os.replace(temporary, current)
        _fsync_directory(current.parent)
    except OSError as exc:
        raise InstallRefused(
            "The current release pointer could not be installed atomically."
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_team_health() -> dict[str, object] | None:
    deadline = time.monotonic() + SERVER_INSTALL_HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            8421,
            timeout=SERVER_HEALTH_REQUEST_TIMEOUT_SECONDS,
        )
        try:
            connection.request(
                "GET",
                "/api/health",
                headers={"Accept": "application/json", "Host": "127.0.0.1:8421"},
            )
            response = connection.getresponse()
            if response.status != 200:
                time.sleep(SERVER_INSTALL_HEALTH_POLL_INTERVAL_SECONDS)
                continue
            body = response.read(SERVER_INSTALL_HEALTH_RESPONSE_MAX_BYTES + 1)
            if len(body) > SERVER_INSTALL_HEALTH_RESPONSE_MAX_BYTES:
                return None
            value = json.loads(body.decode("utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeError, ValueError, http.client.HTTPException):
            time.sleep(SERVER_INSTALL_HEALTH_POLL_INTERVAL_SECONDS)
        finally:
            connection.close()
    return None


__all__ = [
    "GitHubRepository",
    "HostFacts",
    "InstallMachine",
    "InstallRefused",
    "InstalledServiceControlRefused",
    "InstalledSystemServiceController",
    "LinuxInstallMachine",
    "ManagedCheckout",
    "ServiceHealth",
    "ServiceInstallState",
    "SourceAccess",
    "discover_bootstrap_repository",
    "enable_backup_timer",
    "fence_backup_timer_before_unit_change",
    "install_backup_unit_files",
    "normalize_github_repository",
    "prepare_install_command",
    "read_systemd_unit_state",
    "reload_and_disable_backup_timer",
    "run_backup_service_once",
    "source_git_environment",
]
