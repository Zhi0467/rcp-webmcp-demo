from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp.agents import AgentLauncher, ProviderReadiness
from rcp.config import (
    AGENT_EXECUTION_PROFILES,
    GRAPH_AGENT_EXECUTION_PROFILES,
    AgentExecutionProfile,
    Manifest,
    load_manifest,
    permissions_for,
)
from rcp.core.materialize import MaterializationResult
from rcp.core.models import ProjectIdentity
from rcp.history import HistoryManager, ProjectIdentityConflict
from rcp.projects import ProjectCatalog
from rcp.providers import (
    DEFAULT_PROVIDER,
    PROVIDER_IDS,
    ProviderId,
    configured_runtime,
    configured_runtime_id,
)
from rcp.storage import ProjectProvisioningRequestRecord, ProjectRecord
from rcp.storage.provisioning import project_provisioning_review_digest
from rcp.transport import StateWorkspace
from rcp.transport.ssh import ssh_arguments
from rcp.transport.state import SSHStateWorkspace, state_workspace_for_probe

if TYPE_CHECKING:
    from rcp.transfer.configuration import TransferTargetConfiguration


class _StrictSetupModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SetupRepository(_StrictSetupModel):
    alias: str
    location: Literal["local", "ssh"]
    path: str
    host: str = ""
    default_read: bool = True

    @model_validator(mode="after")
    def validate_location(self) -> SetupRepository:
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", self.alias):
            raise ValueError(
                "repository aliases must start with a letter and use lowercase letters, "
                "numbers, or hyphens"
            )
        if self.location == "local":
            self.host = ""
            path = Path(self.path).expanduser()
            if not path.is_absolute() or path == Path("/"):
                raise ValueError(f"local repository {self.alias} needs a specific absolute path")
            self.path = str(path.resolve())
        else:
            if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", self.host):
                raise ValueError(f"remote repository {self.alias} needs a valid SSH host")
            path = PurePosixPath(self.path)
            if not path.is_absolute() or str(path) == "/":
                raise ValueError(f"remote repository {self.alias} needs a specific absolute path")
            self.path = str(path)
        return self


class SetupExecution(_StrictSetupModel):
    location: Literal["local", "ssh"] = "local"
    host: str = ""

    @model_validator(mode="after")
    def validate_host(self) -> SetupExecution:
        if self.location == "local":
            self.host = ""
        elif not re.fullmatch(r"[A-Za-z0-9_.@:-]+", self.host):
            raise ValueError("remote execution needs a valid SSH host")
        return self


class SetupAgentProfile(_StrictSetupModel):
    provider: ProviderId = DEFAULT_PROVIDER
    runtime: str = ""
    model: str = ""
    reasoning: str = "medium"
    location: Literal["local", "ssh"] = "local"
    host: str = ""

    @model_validator(mode="after")
    def validate_provider_runtime(self) -> SetupAgentProfile:
        self.runtime = configured_runtime(self.provider, self.runtime)
        return self

    @model_validator(mode="after")
    def validate_host(self) -> SetupAgentProfile:
        if self.location == "local":
            self.host = ""
        elif not re.fullmatch(r"[A-Za-z0-9_.@:-]+", self.host):
            raise ValueError("remote agent execution needs a valid SSH host")
        return self


class SetupAgents(_StrictSetupModel):
    seed: SetupAgentProfile = Field(default_factory=SetupAgentProfile)
    refresh: SetupAgentProfile = Field(default_factory=SetupAgentProfile)
    node_chat: SetupAgentProfile = Field(default_factory=SetupAgentProfile)
    project_chat: SetupAgentProfile = Field(default_factory=SetupAgentProfile)
    paper_coach: SetupAgentProfile = Field(
        default_factory=lambda: SetupAgentProfile(model="gpt-5.6-luna")
    )
    orchestrator: SetupAgentProfile | None = None

    @model_validator(mode="after")
    def default_orchestrator_to_refresh(self) -> SetupAgents:
        if self.orchestrator is None:
            self.orchestrator = self.refresh.model_copy(deep=True)
        return self

    def profile(self, surface: AgentExecutionProfile) -> SetupAgentProfile:
        profile = getattr(self, surface)
        assert profile is not None
        return profile


ExistingResearchAction = Literal[
    "open_existing",
    "open_degraded_read_only",
    "archive_and_create",
]


class ProjectSetupRequest(_StrictSetupModel):
    name: str = Field(min_length=1, max_length=120)
    repositories: list[SetupRepository] = Field(min_length=1)
    state_repository: str
    default_auto_research_invocation_ceiling: int = Field(
        default=10,
        ge=1,
        description="Operational invocations per newly authorized episode.",
    )
    execution: SetupExecution = Field(default_factory=SetupExecution)
    agents: SetupAgents | None = None
    confirmed: bool = False
    existing_research_action: ExistingResearchAction | None = None
    existing_research_token: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_project(self) -> ProjectSetupRequest:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("project name cannot be blank")
        aliases = [repository.alias for repository in self.repositories]
        if len(aliases) != len(set(aliases)):
            raise ValueError("repository aliases must be unique")
        if self.state_repository not in aliases:
            raise ValueError("canonical state must name one of the project repositories")
        if not any(repository.default_read for repository in self.repositories):
            raise ValueError("select at least one repository for default agent reads")
        if (
            self.existing_research_action == "archive_and_create"
            and self.existing_research_token is None
        ):
            raise ValueError(
                "archiving retained research requires the token from the reviewed preflight"
            )
        locations = [
            (repository.location, repository.host, repository.path)
            for repository in self.repositories
        ]
        if len(locations) != len(set(locations)):
            raise ValueError("the same repository path cannot be added twice")
        remote_hosts = {
            repository.host for repository in self.repositories if repository.location == "ssh"
        }
        if self.agents is None:
            canonical = next(
                repository
                for repository in self.repositories
                if repository.alias == self.state_repository
            )
            graph_profile = SetupAgentProfile(
                location=canonical.location,
                host=canonical.host,
            )
            self.agents = SetupAgents(
                seed=graph_profile.model_copy(),
                refresh=graph_profile.model_copy(),
                node_chat=graph_profile.model_copy(),
                project_chat=graph_profile.model_copy(),
                orchestrator=graph_profile.model_copy(),
            )
        canonical = next(
            repository
            for repository in self.repositories
            if repository.alias == self.state_repository
        )
        for surface in GRAPH_AGENT_EXECUTION_PROFILES:
            profile = self.agents.profile(surface)
            if (profile.location, profile.host) != (canonical.location, canonical.host):
                target = canonical.host or "this machine"
                raise ValueError(
                    f"{surface.replace('_', ' ')} must run beside canonical state on {target}"
                )
        for surface in _SETUP_AGENT_EXECUTION_PROFILES:
            profile = self.agents.profile(surface)
            if profile.location == "ssh" and profile.host not in remote_hosts:
                raise ValueError(
                    f"{surface.replace('_', ' ')} must run on a host that owns a project repository"
                )
        return self


class SetupCheck(_StrictSetupModel):
    label: str
    status: Literal["pass", "warn", "fail"]
    detail: str


class ExistingReplayFailure(_StrictSetupModel):
    revision: int | None
    code: str
    message: str


class ExistingResearchPreview(_StrictSetupModel):
    project_name: str
    canonical_location: str
    retained_revision_count: int
    replay_status: Literal["complete", "degraded"]
    coherent_revision: int
    archive_token: str
    replay_failure: ExistingReplayFailure | None = None


SetupAvailableAction = Literal[
    "create",
    "open_existing",
    "open_degraded_read_only",
    "archive_and_create",
]


class SetupPreview(_StrictSetupModel):
    checks: list[SetupCheck]
    can_create: bool
    action: Literal["create", "connect"]
    canonical_location: str
    existing_project_name: str | None = None
    existing_research: ExistingResearchPreview | None = None
    available_actions: list[SetupAvailableAction]
    manifest_preview: str
    remote_write: bool
    providers: dict[str, ProviderReadiness]
    agent_readiness: dict[str, ProviderReadiness]


@dataclass(frozen=True)
class _RetainedResearch:
    manifest: Manifest
    workspace: StateWorkspace
    materialization: MaterializationResult
    preview: ExistingResearchPreview


_SETUP_AGENT_EXECUTION_PROFILES: tuple[AgentExecutionProfile, ...] = (
    "seed",
    "refresh",
    "node_chat",
    "project_chat",
    "paper_coach",
    "orchestrator",
)


class ProjectSetupManager:
    def __init__(
        self,
        data_dir: Path,
        catalog: ProjectCatalog,
        launcher: AgentLauncher,
    ) -> None:
        self.data_dir = data_dir
        self.catalog = catalog
        self.launcher = launcher

    def preflight(self, request: ProjectSetupRequest) -> SetupPreview:
        checks = [self._check_repository(repository) for repository in request.repositories]
        canonical = self._repository(request, request.state_repository)
        existing_content = self._read_existing_manifest(canonical)
        existing_name: str | None = None
        existing_research: ExistingResearchPreview | None = None
        available_actions: list[SetupAvailableAction] = ["create"]
        action: Literal["create", "connect"] = "create"
        if existing_content is not None:
            action = "connect"
            available_actions = []
            try:
                existing_name = self._validate_existing_manifest(canonical, existing_content)
                retained = self._inspect_retained_research(canonical, existing_content)
                existing_research = retained.preview
                identity = HistoryManager(
                    retained.manifest,
                    retained.workspace,
                ).project_identity(retained.materialization)
                if identity is not None and identity.home_space_id != self.catalog.store.space_id:
                    raise ProjectIdentityConflict(
                        f"Project {identity.project_id} belongs to space "
                        f"{identity.home_space_id}; this space is "
                        f"{self.catalog.store.space_id}. Registration is refused."
                    )
                if existing_research.replay_status == "complete":
                    available_actions.extend(["open_existing", "archive_and_create"])
                else:
                    available_actions.extend(["open_degraded_read_only", "archive_and_create"])
                try:
                    self.catalog.require_archive_available(existing_research.canonical_location)
                except ValueError as exc:
                    available_actions.remove("archive_and_create")
                    checks.append(
                        SetupCheck(
                            label="Archive existing research",
                            status="warn",
                            detail=str(exc),
                        )
                    )
                checks.append(
                    SetupCheck(
                        label="Canonical manifest",
                        status=(
                            "pass" if existing_research.replay_status == "complete" else "warn"
                        ),
                        detail=(
                            f"Found existing RCP project “{existing_name}” with "
                            f"{existing_research.retained_revision_count} retained revisions. "
                            + (
                                "Its complete history can be opened without overwriting its "
                                "configuration. If it has no durable identity yet, opening it "
                                "makes this active RCP space its sole writable home."
                                if existing_research.replay_status == "complete"
                                else _replay_failure_detail(existing_research)
                            )
                        ),
                    )
                )
            except (OSError, ValueError) as exc:
                checks.append(
                    SetupCheck(
                        label="Canonical manifest",
                        status="fail",
                        detail=str(exc),
                    )
                )
        else:
            checks.append(self._check_canonical_writable(canonical))

        assert request.agents is not None
        readiness_cache: dict[tuple[str, str], ProviderReadiness] = {}
        machine_hosts = {
            repository.host if repository.location == "ssh" else ""
            for repository in request.repositories
        }
        machine_hosts.update(
            request.agents.profile(surface).host
            if request.agents.profile(surface).location == "ssh"
            else ""
            for surface in _SETUP_AGENT_EXECUTION_PROFILES
        )
        for host in sorted(machine_hosts):
            for provider in PROVIDER_IDS:
                readiness_cache[(provider, host)] = self.launcher.readiness(
                    provider,
                    host=host,
                )
        agent_readiness: dict[str, ProviderReadiness] = {}
        for surface in _SETUP_AGENT_EXECUTION_PROFILES:
            profile = request.agents.profile(surface)
            host = profile.host if profile.location == "ssh" else ""
            key = (profile.provider, host)
            if key not in readiness_cache:
                readiness_cache[key] = self.launcher.readiness(profile.provider, host=host)
            readiness = readiness_cache[key]
            agent_readiness[surface] = readiness
            checks.append(
                SetupCheck(
                    label=f"{surface.replace('_', ' ').title()} agent",
                    status="pass" if readiness.authenticated else "warn",
                    detail=(
                        f"{readiness.version or profile.provider.title()} is installed and "
                        f"authenticated on {host or 'this machine'}."
                        if readiness.authenticated
                        else readiness.reason or f"{profile.provider.title()} is unavailable."
                    ),
                )
            )
        if request.agents.paper_coach.location == "ssh":
            checks.append(
                SetupCheck(
                    label="Paper coach session resume",
                    status="warn",
                    detail=(
                        "The paper coach can be configured remotely, but v1 native-session "
                        "resume requires a local execution machine."
                    ),
                )
            )
        execution_host = request.execution.host if request.execution.location == "ssh" else ""
        providers = {
            provider: readiness_cache.get((provider, execution_host))
            or self.launcher.readiness(provider, host=execution_host)
            for provider in PROVIDER_IDS
        }

        provider_paths = {
            host: {
                provider: readiness.binary_path
                for provider in PROVIDER_IDS
                if (readiness := readiness_cache[(provider, host)]).installed
                and readiness.binary_path
            }
            for host in machine_hosts
        }
        manifest = render_manifest(request, provider_paths)
        selected_existing_action = request.existing_research_action
        action_ready = (
            existing_content is None
            and selected_existing_action is None
            or selected_existing_action in available_actions
        )
        return SetupPreview(
            checks=checks,
            can_create=(not any(check.status == "fail" for check in checks) and action_ready),
            action=action,
            canonical_location=_canonical_location(canonical),
            existing_project_name=existing_name,
            existing_research=existing_research,
            available_actions=available_actions,
            manifest_preview=manifest,
            remote_write=canonical.location == "ssh",
            providers=providers,
            agent_readiness=agent_readiness,
        )

    def create(
        self,
        request: ProjectSetupRequest,
        *,
        seat_member: str | None = None,
    ) -> dict[str, object]:
        if not request.confirmed:
            raise ValueError("project creation requires final human confirmation")
        preview = self.preflight(request)
        if not preview.can_create:
            failures = [check.detail for check in preview.checks if check.status == "fail"]
            if (
                request.existing_research_action == "archive_and_create"
                and "archive_and_create" not in preview.available_actions
            ):
                failures.extend(
                    check.detail
                    for check in preview.checks
                    if check.label == "Archive existing research"
                )
            if not failures and preview.existing_research is not None:
                failures = [
                    "Choose one of the available existing-research actions before continuing."
                ]
            raise ValueError("; ".join(failures))

        canonical = self._repository(request, request.state_repository)
        existing_content = self._read_existing_manifest(canonical)
        selected_action = request.existing_research_action
        if existing_content is None and selected_action is not None:
            raise ValueError("No retained RCP research exists for the selected action.")

        if existing_content is not None and selected_action == "archive_and_create":
            self.catalog.require_archive_available(preview.canonical_location)
            retained = self._inspect_retained_research(canonical, existing_content)
            retained.workspace.archive_research(
                expected_history_fingerprint=request.existing_research_token,
            )
            existing_content = None

        if canonical.location == "local":
            manifest_path = Path(canonical.path) / ".research" / "manifest.toml"
            if existing_content is None:
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                _exclusive_write(manifest_path, preview.manifest_preview)
            locator = str(manifest_path)
        else:
            content = existing_content or preview.manifest_preview
            locator = str(self._write_bootstrap(canonical, content))

        if selected_action == "open_degraded_read_only":
            assert existing_content is not None
            retained = self._inspect_retained_research(canonical, existing_content)
            record = self.catalog.register_degraded_read_only(
                locator,
                materialization=retained.materialization,
                seat_member=seat_member,
            )
            return self.catalog.card(record.project_id)

        record = None
        try:
            record = self.catalog.register(
                locator,
                identity_action=(
                    "created"
                    if preview.action == "create" or selected_action == "archive_and_create"
                    else "adopted"
                ),
                seat_member=seat_member,
            )
            _, snapshot = self.catalog.open_snapshot(record.project_id)
            self.catalog.update_summary(record.project_id, snapshot)
            return self.catalog.card(record.project_id)
        except Exception:
            if selected_action == "archive_and_create":
                registered = record or self.catalog.store.project_by_locator(locator)
                if registered is not None:
                    with suppress(KeyError):
                        self.catalog.delete(registered.project_id)
            raise

    def create_prepared_team_project(
        self,
        request: ProjectProvisioningRequestRecord,
        *,
        seat_member: str,
    ) -> dict[str, object]:
        """Finalize one reviewed request without repeating machine preparation."""

        if self.catalog.store.space_kind != "team":
            raise ValueError("prepared team-project creation requires a team space")
        if request.kind != "create_team_project" or request.status != "ready_for_review":
            raise ValueError("only a ready new-team request can create a project")
        if request.target_space_id != self.catalog.store.space_id:
            raise ValueError("the prepared project targets another RCP space")

        manifest_content = render_prepared_team_manifest(request)
        prepared_repositories = self._prepared_repositories(request)
        for repository in prepared_repositories.values():
            self._require_prepared_checkout_path(repository)
        assert request.state_repository is not None
        canonical = prepared_repositories[request.state_repository]
        self._require_prepared_state_path(canonical)
        existing_content = self._read_existing_manifest(canonical)
        if existing_content is None:
            if self._prepared_patch_history_exists(canonical, manifest_content):
                raise ValueError(
                    "Canonical Patch history appeared after server preparation. Use Move to "
                    "team space for retained personal research, or clean or choose a different "
                    "repository outside this request before reviewing again."
                )
            if canonical.location == "local":
                manifest_path = Path(canonical.path) / ".research" / "manifest.toml"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                _exclusive_write(manifest_path, manifest_content)
                locator = str(manifest_path)
            else:
                locator = str(self._write_bootstrap(canonical, manifest_content))
        else:
            self._require_prepared_manifest(existing_content, manifest_content)
            retained = self._inspect_retained_research(canonical, existing_content)
            self._require_resumable_prepared_identity(retained, request)
            locator = (
                str(Path(canonical.path) / ".research" / "manifest.toml")
                if canonical.location == "local"
                else str(self._write_bootstrap(canonical, manifest_content))
            )

        record = self.catalog.register_prepared_team_project(
            locator,
            project_id=request.proposed_project_id,
            seat_member=seat_member,
        )
        _, snapshot = self.catalog.open_snapshot(record.project_id)
        self.catalog.update_summary(record.project_id, snapshot)
        return self.catalog.card(record.project_id)

    def prepare_incoming_transfer_project(
        self,
        request: ProjectProvisioningRequestRecord,
        *,
        target_configuration: TransferTargetConfiguration,
    ) -> ProjectRecord:
        """Revalidate one published import without making it catalog-visible."""

        if self.catalog.store.space_kind != "team":
            raise ValueError("incoming transfer finalization requires a team space")
        if request.kind != "incoming_transfer" or request.status != "ready_for_review":
            raise ValueError("only a ready incoming-transfer request can be finalized")
        if request.target_space_id != self.catalog.store.space_id:
            raise ValueError("the incoming transfer targets another RCP space")
        if (
            request.final_review_digest is None
            or project_provisioning_review_digest(request) != request.final_review_digest
        ):
            raise ValueError("the incoming transfer final review is stale")

        receipt = target_configuration.receipt
        if (
            receipt.target_request_id != request.request_id
            or receipt.project_id != request.proposed_project_id
            or receipt.target_space_id != request.target_space_id
            or receipt.final_review_sha256 != request.final_review_digest
        ):
            raise ValueError("the reviewed target configuration does not bind this request")
        expected_manifest = render_prepared_team_manifest(request)
        if target_configuration.manifest_content != expected_manifest:
            raise ValueError("the target manifest differs from the reviewed provisioning request")

        imported = self.catalog.store.project_transfer_import(request.request_id)
        if (
            imported is None
            or imported.status != "complete"
            or imported.project_id != request.proposed_project_id
            or imported.archive_manifest_sha256 != receipt.archive_manifest_sha256
            or imported.target_manifest_sha256 != receipt.target_manifest_sha256
        ):
            raise ValueError("the incoming transfer import is not complete for this review")

        prepared_repositories = self._prepared_repositories(request)
        for repository in prepared_repositories.values():
            self._require_prepared_checkout_path(repository)
        assert request.state_repository is not None
        canonical = prepared_repositories[request.state_repository]
        self._require_prepared_state_path(canonical)
        actual_manifest = self._read_existing_manifest(canonical)
        if actual_manifest is None:
            raise ValueError("the imported canonical manifest is missing")
        if actual_manifest != expected_manifest:
            raise ValueError("the imported canonical manifest differs from the reviewed manifest")
        self._require_prepared_manifest(actual_manifest, expected_manifest)
        locator = (
            str(Path(canonical.path) / ".research" / "manifest.toml")
            if canonical.location == "local"
            else str(self._write_bootstrap(canonical, expected_manifest))
        )
        return self.catalog.prepare_incoming_transfer_registration(
            locator,
            project_id=request.proposed_project_id,
            home_space_id=request.target_space_id,
            expected_manifest_content=expected_manifest,
        )

    @staticmethod
    def _prepared_repositories(
        request: ProjectProvisioningRequestRecord,
    ) -> dict[str, SetupRepository]:
        machine_map = {machine.alias: machine for machine in request.machines}
        prepared: dict[str, SetupRepository] = {}
        for repository in request.repositories:
            if repository.resolved_path is None:
                raise ValueError(f"the prepared checkout path for {repository.alias} is missing")
            machine = machine_map[repository.machine_alias]
            item = SetupRepository(
                alias=repository.alias,
                location=machine.location,
                host=machine.host,
                path=repository.resolved_path,
                default_read=repository.alias in request.default_run_truth_scope,
            )
            if item.path != repository.resolved_path:
                raise ValueError(
                    f"the prepared checkout {repository.alias} now resolves to another path"
                )
            prepared[repository.alias] = item
        return prepared

    @staticmethod
    def _require_prepared_checkout_path(repository: SetupRepository) -> None:
        check = ProjectSetupManager._check_repository(repository)
        if check.status == "fail":
            raise ValueError(check.detail)
        if repository.location == "local":
            if Path(repository.path).is_symlink():
                raise ValueError(f"the prepared checkout {repository.alias} became a symbolic link")
            return
        symlink = _ssh(repository.host, ["test", "-L", repository.path])
        if symlink.returncode == 0:
            raise ValueError(f"the prepared checkout {repository.alias} became a symbolic link")
        if symlink.returncode != 1:
            raise ValueError(f"could not recheck the prepared checkout {repository.alias}")

    @staticmethod
    def _require_prepared_state_path(canonical: SetupRepository) -> None:
        if canonical.location == "local":
            research = Path(canonical.path) / ".research"
            manifest = research / "manifest.toml"
            patches = research / "patches"
            for label, path, directory in (
                ("canonical state", research, True),
                ("canonical manifest", manifest, False),
                ("canonical Patch directory", patches, True),
            ):
                if path.is_symlink():
                    raise ValueError(f"the prepared {label} became a symbolic link")
                if path.exists() and (not path.is_dir() if directory else not path.is_file()):
                    raise ValueError(f"the prepared {label} has an invalid file type")
            return

        research = str(PurePosixPath(canonical.path) / ".research")
        for label, path, kind in (
            ("canonical state", research, "d"),
            ("canonical manifest", f"{research}/manifest.toml", "f"),
            ("canonical Patch directory", f"{research}/patches", "d"),
        ):
            symlink = _ssh(canonical.host, ["test", "-L", path])
            if symlink.returncode == 0:
                raise ValueError(f"the prepared {label} became a symbolic link")
            if symlink.returncode not in {0, 1}:
                raise ValueError(f"could not recheck the prepared {label}")
            exists = _ssh(canonical.host, ["test", "-e", path])
            if exists.returncode == 1:
                continue
            if exists.returncode != 0:
                raise ValueError(f"could not recheck the prepared {label}")
            expected = _ssh(canonical.host, ["test", f"-{kind}", path])
            if expected.returncode != 0:
                raise ValueError(f"the prepared {label} has an invalid file type")

    def _prepared_patch_history_exists(
        self,
        canonical: SetupRepository,
        manifest_content: str,
    ) -> bool:
        if canonical.location == "local":
            return bool(_retained_patch_paths(Path(canonical.path) / ".research"))
        bootstrap = load_manifest(self._write_bootstrap(canonical, manifest_content))
        workspace = state_workspace_for_probe(bootstrap, self.data_dir)
        assert isinstance(workspace, SSHStateWorkspace)
        reachable, head = workspace.probe_remote_patch_log_head()
        if not reachable:
            raise ValueError(
                f"Could not recheck canonical Patch history at {_canonical_location(canonical)}."
            )
        return head is not None

    @staticmethod
    def _require_prepared_manifest(actual_content: str, expected_content: str) -> None:
        try:
            actual = Manifest.model_validate(tomlkit.parse(actual_content).unwrap())
            expected = Manifest.model_validate(tomlkit.parse(expected_content).unwrap())
        except (ValueError, tomlkit.exceptions.ParseError) as exc:
            raise ValueError(f"The prepared canonical manifest is invalid: {exc}") from exc
        if actual.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise ValueError(
                "The canonical manifest changed after server preparation; review a new "
                "provisioning request instead of adopting or overwriting it."
            )

    @staticmethod
    def _require_resumable_prepared_identity(
        retained: _RetainedResearch,
        request: ProjectProvisioningRequestRecord,
    ) -> None:
        patches = retained.materialization.patches
        if not patches:
            return
        expected_identity = ProjectIdentity(
            project_id=request.proposed_project_id,
            home_space_id=request.target_space_id,
            action="created",
        )
        patch = patches[0]
        if not (
            len(patches) == 1
            and retained.preview.retained_revision_count == 1
            and retained.preview.replay_status == "complete"
            and patch.revision == 1
            and patch.kind == "identity"
            and patch.admission == "accepted"
            and patch.author is None
            and patch.producer == "system"
            and patch.ops == []
            and patch.project_identity == expected_identity
        ):
            raise ValueError(
                "Canonical identity or Patch history appeared after server preparation. Use "
                "Move to team space for retained personal research, or clean or choose a "
                "different repository outside this request before reviewing again."
            )

    @staticmethod
    def _repository(request: ProjectSetupRequest, alias: str) -> SetupRepository:
        return next(repository for repository in request.repositories if repository.alias == alias)

    @staticmethod
    def _check_repository(repository: SetupRepository) -> SetupCheck:
        if repository.location == "local":
            path = Path(repository.path)
            if not path.is_dir():
                return SetupCheck(
                    label=repository.alias,
                    status="fail",
                    detail=f"Local directory does not exist: {path}",
                )
            return SetupCheck(
                label=repository.alias,
                status="pass",
                detail=f"Local repository is reachable at {path}.",
            )
        result = _ssh(repository.host, ["test", "-d", repository.path])
        if result.returncode:
            detail = result.stderr.strip() or f"Remote directory does not exist: {repository.path}"
            return SetupCheck(label=repository.alias, status="fail", detail=detail)
        return SetupCheck(
            label=repository.alias,
            status="pass",
            detail=f"SSH reached {repository.host}:{repository.path}.",
        )

    @staticmethod
    def _read_existing_manifest(repository: SetupRepository) -> str | None:
        if repository.location == "local":
            path = Path(repository.path) / ".research" / "manifest.toml"
            return path.read_text(encoding="utf-8") if path.is_file() else None
        path = str(PurePosixPath(repository.path) / ".research" / "manifest.toml")
        result = _ssh(repository.host, ["cat", path])
        if result.returncode == 0:
            return result.stdout
        exists = _ssh(repository.host, ["test", "-f", path])
        if exists.returncode == 1:
            return None
        raise ValueError(result.stderr.strip() or "Could not read the remote manifest")

    @staticmethod
    def _validate_existing_manifest(repository: SetupRepository, content: str) -> str:
        try:
            if repository.location == "local":
                manifest = load_manifest(Path(repository.path) / ".research" / "manifest.toml")
            else:
                data = tomlkit.parse(content).unwrap()
                manifest = Manifest.model_validate(data)
        except (ValueError, tomlkit.exceptions.ParseError) as exc:
            raise ValueError(f"Existing manifest is invalid: {exc}") from exc
        state = manifest.repository_map[manifest.state.repository]
        machine = manifest.machine_map[state.machine]
        expected_host = repository.host if repository.location == "ssh" else ""
        expected_path = str(
            PurePosixPath(repository.path)
            if repository.location == "ssh"
            else Path(repository.path).resolve()
        )
        actual_path = str(
            PurePosixPath(state.path) if expected_host else Path(state.path).expanduser().resolve()
        )
        if machine.host != expected_host or actual_path != expected_path:
            raise ValueError(
                "Existing manifest points to a different canonical repository; "
                "RCP will not relabel or overwrite it."
            )
        return manifest.name

    def _inspect_retained_research(
        self,
        repository: SetupRepository,
        content: str,
    ) -> _RetainedResearch:
        if repository.location == "local":
            manifest = load_manifest(Path(repository.path) / ".research" / "manifest.toml")
        else:
            bootstrap = self._write_bootstrap(repository, content)
            manifest = load_manifest(bootstrap)

        workspace = state_workspace_for_probe(manifest, self.data_dir)
        if workspace.remote and not workspace.refresh():
            raise ValueError(
                f"Retained research is unavailable at {_canonical_location(repository)}."
            )
        fingerprint_before = workspace.retained_history_fingerprint()
        manifest = load_manifest(workspace.root / "manifest.toml")
        history = HistoryManager(manifest, workspace)
        patch_paths = _retained_patch_paths(history.root)
        materialization = history.materialize(write_outputs=False)
        replay_failure = materialization.state.replay_failure
        failure = (
            ExistingReplayFailure(
                revision=replay_failure.revision,
                code=replay_failure.code,
                message=replay_failure.message,
            )
            if replay_failure is not None
            else None
        )

        status: Literal["complete", "degraded"] = "complete" if failure is None else "degraded"
        if materialization.state.replay_status == "degraded":
            status = "degraded"
        fingerprint_after = workspace.retained_history_fingerprint()
        if fingerprint_after != fingerprint_before:
            raise ValueError(
                "Retained research changed during read-only preflight. Review it again before "
                "choosing an action."
            )
        preview = ExistingResearchPreview(
            project_name=manifest.name,
            canonical_location=_canonical_location(repository),
            retained_revision_count=len(patch_paths),
            replay_status=status,
            coherent_revision=materialization.state.revision,
            archive_token=fingerprint_after,
            replay_failure=failure,
        )
        return _RetainedResearch(
            manifest=manifest,
            workspace=workspace,
            materialization=materialization,
            preview=preview,
        )

    @staticmethod
    def _check_canonical_writable(repository: SetupRepository) -> SetupCheck:
        if repository.location == "local":
            research_dir = Path(repository.path) / ".research"
            if research_dir.exists() and not research_dir.is_dir():
                return SetupCheck(
                    label="Canonical state write",
                    status="fail",
                    detail=f"Canonical state path is not a directory: {research_dir}",
                )
            target = research_dir if research_dir.is_dir() else Path(repository.path)
            writable = os.access(target, os.W_OK)
        else:
            research_dir = str(PurePosixPath(repository.path) / ".research")
            exists = _ssh(repository.host, ["test", "-e", research_dir])
            if exists.returncode == 0:
                directory = _ssh(repository.host, ["test", "-d", research_dir])
                if directory.returncode:
                    return SetupCheck(
                        label="Canonical state write",
                        status="fail",
                        detail=f"Canonical state path is not a directory: {_canonical_location(repository)}",
                    )
                target = research_dir
            else:
                target = repository.path
            result = _ssh(repository.host, ["test", "-w", target])
            writable = result.returncode == 0
        return SetupCheck(
            label="Canonical state write",
            status="pass" if writable else "fail",
            detail=(
                f"RCP can create .research/ at {_canonical_location(repository)}."
                if writable
                else f"RCP cannot write to {_canonical_location(repository)}."
            ),
        )

    def _write_bootstrap(self, canonical: SetupRepository, content: str) -> Path:
        digest = hashlib.sha256(f"{canonical.host}\0{canonical.path}".encode()).hexdigest()[:16]
        path = self.data_dir / "bootstrap-manifests" / f"{digest}.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)
        return path


def render_prepared_team_manifest(request: ProjectProvisioningRequestRecord) -> str:
    """Render the exact reviewed paths and execution profiles of a ready request."""

    if not request.configuration_complete:
        raise ValueError("the prepared project configuration is incomplete")
    if request.final_review_digest is None:
        raise ValueError("the prepared project has no final-review digest")
    machine_map = {machine.alias: machine for machine in request.machines}
    profile_map = {check.profile: check for check in request.provider_checks}
    missing_profiles = set(AGENT_EXECUTION_PROFILES) - set(profile_map)
    extra_profiles = set(profile_map) - set(AGENT_EXECUTION_PROFILES)
    if missing_profiles or extra_profiles:
        raise ValueError(
            "the prepared project must review every agent execution profile; "
            f"missing={sorted(missing_profiles)}, extra={sorted(extra_profiles)}"
        )

    provider_paths: dict[str, dict[str, str]] = {}
    for check in request.provider_checks:
        proof = (
            check.binary_path,
            check.version,
            check.resolved_runtime_id,
            check.execution_account,
        )
        if check.status != "ready" or any(value is None for value in proof):
            raise ValueError(f"the prepared {check.profile} provider proof is incomplete")
        machine = machine_map[check.machine_alias]
        if check.execution_account != machine.os_account:
            raise ValueError(f"the prepared {check.profile} execution account changed")
        configured = check.runtime_id.removeprefix(f"{check.provider}:")
        public_runtime = configured_runtime(check.provider, configured)
        if check.resolved_runtime_id != configured_runtime_id(check.provider, public_runtime):
            raise ValueError(f"the prepared {check.profile} runtime proof changed")
        assert check.binary_path is not None
        paths = provider_paths.setdefault(machine.alias, {})
        prior = paths.get(check.provider)
        if prior is not None and prior != check.binary_path:
            raise ValueError(
                f"the prepared {check.provider} executable differs across profiles on "
                f"{machine.alias}"
            )
        paths[check.provider] = check.binary_path

    document = tomlkit.document()
    assert request.name is not None
    assert request.state_repository is not None
    document.add("name", request.name)

    machines = tomlkit.aot()
    for item in request.machines:
        machine = tomlkit.table()
        machine.add("alias", item.alias)
        machine.add("host", item.host)
        machine.add("os_account", item.os_account)
        _add_provider_paths(machine, provider_paths.get(item.alias, {}))
        machines.append(machine)
    document.add("machines", machines)

    repositories = tomlkit.aot()
    for item in request.repositories:
        if (
            item.resolved_path is None
            or item.checkout_disposition is None
            or item.git_check.status != "ready"
        ):
            raise ValueError(f"the prepared repository {item.alias} is incomplete")
        repository = tomlkit.table()
        repository.add("alias", item.alias)
        repository.add("machine", item.machine_alias)
        repository.add("path", item.resolved_path)
        repositories.append(repository)
    document.add("repositories", repositories)

    project = tomlkit.table()
    project.add("truth_scope", request.project_truth_scope)
    document.add("project", project)

    state = tomlkit.table()
    state.add("repository", request.state_repository)
    document.add("state", state)

    agent = tomlkit.table()
    agent.add("default_run_truth_scope", request.default_run_truth_scope)
    agent.add(
        "default_auto_research_invocation_ceiling",
        request.default_auto_research_invocation_ceiling,
    )
    for surface in AGENT_EXECUTION_PROFILES:
        check = profile_map[surface]
        configured = check.runtime_id.removeprefix(f"{check.provider}:")
        profile = tomlkit.table()
        profile.add("provider", check.provider)
        profile.add("runtime", configured_runtime(check.provider, configured))
        profile.add("model", check.model)
        profile.add("reasoning", check.reasoning)
        profile.add("run_on", check.machine_alias)
        permissions = tomlkit.table()
        for key, value in permissions_for(surface).model_dump(mode="json").items():
            permissions.add(key, value)
        profile.add("permissions", permissions)
        agent.add(surface, profile)
    document.add("agent", agent)

    sources = tomlkit.table()
    sources.add("claude_roots", ["~/.claude/projects"])
    sources.add("codex_roots", ["~/.codex/sessions"])
    sources.add("remote_claude_roots", ["~/.claude/projects"])
    sources.add("remote_codex_roots", ["~/.codex/sessions"])
    document.add("sources", sources)

    content = tomlkit.dumps(document)
    Manifest.model_validate(tomlkit.parse(content).unwrap())
    return content


def render_manifest(
    request: ProjectSetupRequest,
    provider_paths: dict[str, dict[str, str]] | None = None,
) -> str:
    assert request.agents is not None
    document = tomlkit.document()
    document.add("name", request.name)

    host_aliases: dict[str, str] = {}
    needs_local = any(repository.location == "local" for repository in request.repositories) or any(
        request.agents.profile(surface).location == "local"
        for surface in _SETUP_AGENT_EXECUTION_PROFILES
    )
    machines = tomlkit.aot()
    if needs_local:
        host_aliases[""] = "laptop"
        machine = tomlkit.table()
        machine.add("alias", "laptop")
        machine.add("host", "")
        _add_provider_paths(machine, (provider_paths or {}).get("", {}))
        machines.append(machine)
    remote_hosts = sorted(
        {repository.host for repository in request.repositories if repository.location == "ssh"}
    )
    for index, host in enumerate(remote_hosts, start=1):
        alias = f"remote-{index}"
        host_aliases[host] = alias
        machine = tomlkit.table()
        machine.add("alias", alias)
        machine.add("host", host)
        _add_provider_paths(machine, (provider_paths or {}).get(host, {}))
        machines.append(machine)
    document.add("machines", machines)

    repositories = tomlkit.aot()
    for item in request.repositories:
        repository = tomlkit.table()
        repository.add("alias", item.alias)
        repository.add("machine", host_aliases[item.host])
        repository.add("path", item.path)
        repositories.append(repository)
    document.add("repositories", repositories)

    project = tomlkit.table()
    project.add("truth_scope", [repository.alias for repository in request.repositories])
    document.add("project", project)

    state = tomlkit.table()
    state.add("repository", request.state_repository)
    document.add("state", state)

    agent = tomlkit.table()
    agent.add(
        "default_run_truth_scope",
        [repository.alias for repository in request.repositories if repository.default_read],
    )
    agent.add(
        "default_auto_research_invocation_ceiling",
        request.default_auto_research_invocation_ceiling,
    )
    for surface in _SETUP_AGENT_EXECUTION_PROFILES:
        setup_profile = request.agents.profile(surface)
        profile = tomlkit.table()
        profile.add("provider", setup_profile.provider)
        profile.add("runtime", setup_profile.runtime)
        profile.add("model", setup_profile.model)
        profile.add("reasoning", setup_profile.reasoning)
        profile.add("run_on", host_aliases[setup_profile.host])
        permissions = tomlkit.table()
        for key, value in permissions_for(surface).model_dump(mode="json").items():
            permissions.add(key, value)
        profile.add("permissions", permissions)
        agent.add(surface, profile)
    document.add("agent", agent)

    sources = tomlkit.table()
    sources.add("claude_roots", ["~/.claude/projects"])
    sources.add("codex_roots", ["~/.codex/sessions"])
    sources.add("remote_claude_roots", ["~/.claude/projects"])
    sources.add("remote_codex_roots", ["~/.codex/sessions"])
    document.add("sources", sources)

    return tomlkit.dumps(document)


def _add_provider_paths(machine: tomlkit.items.Table, paths: dict[str, str]) -> None:
    if not paths:
        return
    provider_paths = tomlkit.inline_table()
    for provider in PROVIDER_IDS:
        path = paths.get(provider)
        if path:
            provider_paths.append(provider, path)
    if provider_paths:
        machine.add("provider_paths", provider_paths)


def _canonical_location(repository: SetupRepository) -> str:
    return (
        f"{repository.host}:{repository.path}/.research"
        if repository.location == "ssh"
        else str(Path(repository.path) / ".research")
    )


def _retained_patch_paths(root: Path) -> list[Path]:
    patches = root / "patches"
    paths = [
        *patches.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"),
        *patches.glob("batch-*/[0-9][0-9][0-9][0-9][0-9][0-9].json"),
    ]
    return sorted(paths, key=lambda path: int(path.stem))


def _replay_failure_detail(existing: ExistingResearchPreview) -> str:
    failure = existing.replay_failure
    if failure is None:
        return (
            f"Replay is degraded after coherent revision {existing.coherent_revision}; "
            "ordinary resume is unavailable."
        )
    revision = f"revision {failure.revision}" if failure.revision is not None else "history"
    return (
        f"Replay stops at {revision} ({failure.code}): {failure.message} "
        f"The last coherent revision is {existing.coherent_revision}; ordinary resume is "
        "unavailable."
    )


def _ssh(host: str, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    command = shlex.join(arguments)
    try:
        return subprocess.run(
            ssh_arguments(host, command),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess([], 255, "", str(exc))


def _exclusive_write(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError(
            f"A manifest appeared at {path} after preflight; review it before connecting."
        ) from exc
