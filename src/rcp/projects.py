from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import tomlkit
from pydantic import BaseModel, TypeAdapter

from rcp.agents import AgentLauncher
from rcp.agents.write_scope import RegisteredRepositoryRoot, registered_repository_roots
from rcp.attachments import ChatAttachmentStore
from rcp.config import (
    AGENT_EXECUTION_PROFILES,
    DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING,
    Manifest,
    load_manifest,
)
from rcp.core.attention import project_graph_attention
from rcp.core.materialize import MaterializationResult
from rcp.core.models import Experiment, GraphState
from rcp.core.transition_models import GraphAttentionProjection, GraphTargetRef
from rcp.core.transitions import ProjectTransitionProjection
from rcp.history import HistoryManager, ProjectIdentityConflict, ReplayHalted
from rcp.limits import (
    PROJECT_DISPLAY_SNAPSHOT_MAX_BYTES,
    REMOTE_STATE_HEAD_PROBE_INTERVAL_SECONDS,
)
from rcp.paper import PaperService, PaperSnapshot
from rcp.provider_skills import ProviderSkillInventoryManager
from rcp.providers import PROVIDER_IDS, ProviderId, configured_runtime
from rcp.runs.task_policy import task_graph_capable
from rcp.server_ops.backup_models import (
    BackupCheckoutRecoveryDescriptor,
    BackupManifestConfiguration,
    BackupProjectCapture,
    BackupRecoveryMachine,
    BackupRecoveryRepository,
)
from rcp.service import ProjectService, ProjectSettingsRequest, _ProjectSnapshotDraft
from rcp.sources import (
    ImportedProviderSourceInventory,
    ImportedProviderSourceStore,
    project_cache_roots,
)
from rcp.storage import (
    AgentTaskKind,
    AppStore,
    EpisodeRecord,
    ExperimentEpisodeProjectionSnapshot,
    ExperimentLoopRuntime,
    ProjectProvisioningRequestRecord,
    ProjectRecord,
    ProjectStageRecord,
)
from rcp.transport import (
    LocalStateWorkspace,
    RemoteRunStage,
    StateUnavailable,
    StateWorkspace,
    prepare_state_workspace,
)
from rcp.transport.state import SSHStateWorkspace, state_workspace_for_probe

if TYPE_CHECKING:
    from rcp.background import AgentTaskExecution, AgentTaskRequest

_DISPLAY_SNAPSHOT_SCHEMA_VERSION = 3
_DISPLAY_SNAPSHOT_ENVELOPE_ADAPTER = TypeAdapter(dict[str, object])
_PATCH_LOG_HEAD_UNSET = object()
_DISPLAY_SNAPSHOT_FIELDS = {
    "id",
    "home_space_id",
    "name",
    "revision",
    "snapshot_freshness",
    "last_remote_sync_at",
    "state_repository",
    "canonical_state",
    "run_on",
    "project_truth_scope",
    "default_run_truth_scope",
    "default_auto_research_invocation_ceiling",
    "repositories",
    "machines",
    "primary_question",
    "last_refresh_at",
    "attention",
    "counts",
    "coverage",
    "graph",
    "paper",
    "paper_coach",
    "agent_profiles",
    "provider_readiness",
    "provider_skill_inventories",
    "providers",
    "cache_metrics",
    "validation_messages",
}

ProjectSnapshot = dict[str, object] | _ProjectSnapshotDraft


class BackupProjectUnavailable(ValueError):
    """A registered project cannot support an honest backup claim."""


class RestoredProjectRebindRefused(RuntimeError):
    """A stopped restored catalog row could not bind to replacement checkouts."""


class RestoredProjectPublicationRefused(RuntimeError):
    """A restored project could not be replayed and exposed from its replacement checkout."""


@dataclass(frozen=True)
class BackupProjectRegistration:
    """Read-only live locator plus its validated checkout reconstruction proof."""

    record: ProjectRecord
    manifest: Manifest
    workspace: StateWorkspace
    recovery: BackupCheckoutRecoveryDescriptor


@dataclass(frozen=True)
class RestoredProjectOwners:
    manifest: Manifest
    workspace: StateWorkspace
    history: HistoryManager
    paper: PaperService
    service: ProjectService


def inspect_backup_project_registration(
    record: ProjectRecord,
    *,
    data_dir: Path,
    provisioning_requests: Iterable[ProjectProvisioningRequestRecord],
) -> BackupProjectRegistration:
    """Bind one catalog row to its completed provisioning record and manifest.

    The caller supplies records from the database snapshot it intends to archive.
    This helper reads configuration and constructs a workspace but never refreshes,
    opens canonical history, or mutates the catalog.
    """

    if record.home_space_id is None:
        raise BackupProjectUnavailable("The project has no durable home-space identity.")
    matches = [
        request
        for request in provisioning_requests
        if request.status == "completed"
        and request.proposed_project_id == record.project_id
        and request.target_space_id == record.home_space_id
    ]
    if len(matches) != 1:
        raise BackupProjectUnavailable(
            "The project does not have exactly one completed provisioning record."
        )
    request = matches[0]
    if request.completed_at is None or request.final_review_digest is None:
        raise BackupProjectUnavailable("The completed provisioning proof is incomplete.")
    from rcp.storage.provisioning import project_provisioning_review_digest

    if project_provisioning_review_digest(request) != request.final_review_digest:
        raise BackupProjectUnavailable("The completed provisioning review digest is stale.")

    try:
        manifest = load_manifest(record.locator)
    except (OSError, ValueError) as exc:
        raise BackupProjectUnavailable("The registered project manifest is unavailable.") from exc
    if manifest.name != record.name or str(manifest.path) != record.locator:
        raise BackupProjectUnavailable("The project catalog and registered manifest disagree.")

    # The P6b renderer is the concrete owner of the reviewed team manifest. Reuse
    # it here so backup cannot grow a second interpretation of that configuration.
    from rcp.setup import render_prepared_team_manifest

    try:
        actual_document = tomlkit.parse(Path(record.locator).read_text(encoding="utf-8")).unwrap()
        recorded_manifest = Manifest.model_validate(actual_document)
        expected_document = tomlkit.parse(render_prepared_team_manifest(request)).unwrap()
        expected_manifest = Manifest.model_validate(expected_document)
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise BackupProjectUnavailable(
            "The completed provisioning record cannot reproduce its reviewed manifest."
        ) from exc
    if recorded_manifest.model_dump(mode="json") != expected_manifest.model_dump(mode="json"):
        raise BackupProjectUnavailable(
            "The canonical manifest changed after the completed provisioning proof."
        )

    state_repository = manifest.repository_map[manifest.state.repository]
    state_machine = manifest.machine_map[state_repository.machine]
    expected_state_location = (
        f"{state_machine.host}:{state_repository.path}/.research"
        if state_machine.host
        else str(manifest.research_dir)
    )
    if record.state_location != expected_state_location or record.state_remote is not bool(
        state_machine.host
    ):
        raise BackupProjectUnavailable("The project catalog and canonical state location disagree.")

    configuration = BackupManifestConfiguration.from_manifest(manifest)
    try:
        completed_at = datetime.fromisoformat(request.completed_at)
        recovery = BackupCheckoutRecoveryDescriptor(
            request_id=request.request_id,
            project_id=record.project_id,
            home_space_id=record.home_space_id,
            completed_at=completed_at,
            final_review_digest=request.final_review_digest,
            configuration=configuration,
            configuration_sha256=configuration.sha256,
            machines=tuple(
                BackupRecoveryMachine(
                    alias=machine.alias,
                    location=machine.location,
                    host=machine.host,
                    os_account=machine.os_account,
                    resolved_central_root=machine.resolved_central_root,
                )
                for machine in request.machines
                if machine.resolved_central_root is not None
            ),
            repositories=tuple(
                BackupRecoveryRepository(
                    alias=repository.alias,
                    repository=repository.repository,
                    machine_alias=repository.machine_alias,
                    resolved_path=repository.resolved_path,
                    git_commit=repository.git_check.commit,
                    deploy_key_label=repository.git_check.deploy_key_label,
                    public_key_fingerprint=repository.git_check.public_key_fingerprint,
                )
                for repository in request.repositories
                if repository.resolved_path is not None
                and repository.git_check.commit is not None
                and repository.git_check.deploy_key_label is not None
                and repository.git_check.public_key_fingerprint is not None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise BackupProjectUnavailable(
            "The completed provisioning record cannot reconstruct every checkout."
        ) from exc
    if len(recovery.machines) != len(request.machines) or len(recovery.repositories) != len(
        request.repositories
    ):
        raise BackupProjectUnavailable(
            "The completed provisioning record is missing a checkout recovery field."
        )
    return BackupProjectRegistration(
        record=record,
        manifest=manifest,
        workspace=state_workspace_for_probe(manifest, data_dir),
        recovery=recovery,
    )


def rebind_restored_project_registration(
    store: AppStore,
    capture: BackupProjectCapture,
    *,
    repository_paths: dict[str, str],
    data_dir: Path,
    uid: int,
    gid: int,
) -> ProjectRecord:
    """Atomically point one stopped restored row at its verified replacement checkout."""

    recovery = capture.recovery
    if capture.status != "captured" or recovery is None:
        raise RestoredProjectRebindRefused("Only one captured project can be rebound.")
    record = store.project(capture.project_id)
    if (
        record is None
        or record.home_space_id != capture.home_space_id
        or recovery.project_id != capture.project_id
        or recovery.home_space_id != capture.home_space_id
        or record.name != recovery.configuration.name
    ):
        raise RestoredProjectRebindRefused(
            "The restored catalog identity differs from its recovery descriptor."
        )
    configured_aliases = {item.alias for item in recovery.configuration.repositories}
    recovered = {item.alias: item for item in recovery.repositories}
    if set(repository_paths) != configured_aliases or set(recovered) != configured_aliases:
        raise RestoredProjectRebindRefused(
            "The replacement checkout set differs from the recovery descriptor."
        )
    if any(repository_paths[alias] != recovered[alias].resolved_path for alias in recovered):
        raise RestoredProjectRebindRefused(
            "A replacement checkout path differs from its reviewed recovery path."
        )
    content = _render_restored_manifest(recovery.configuration, repository_paths)
    repositories = {item.alias: item for item in recovery.configuration.repositories}
    machines = {item.alias: item for item in recovery.configuration.machines}
    state_repository = repositories[recovery.configuration.state_repository]
    state_machine = machines[state_repository.machine]
    state_path = repository_paths[state_repository.alias]
    state_remote = bool(state_machine.host)
    if state_remote:
        locator = _write_restored_bootstrap_manifest(
            data_dir,
            host=state_machine.host,
            repository_path=state_path,
            content=content,
            uid=uid,
            gid=gid,
        )
        state_location = f"{state_machine.host}:{state_path}/.research"
    else:
        locator = Path(state_path) / ".research" / "manifest.toml"
        state_location = str(Path(state_path) / ".research")
    try:
        stored = store.rebind_project_registration_for_restore(
            record.project_id,
            home_space_id=recovery.home_space_id,
            name=recovery.configuration.name,
            locator=str(locator),
            state_location=state_location,
            state_remote=state_remote,
        )
    except (KeyError, RuntimeError, ValueError, sqlite3.Error) as exc:
        raise RestoredProjectRebindRefused(
            "The restored catalog rejected its replacement checkout binding."
        ) from exc
    if (
        stored.locator != str(locator)
        or stored.state_location != state_location
        or stored.state_remote is not state_remote
        or stored.reachable is not False
    ):
        raise RestoredProjectRebindRefused(
            "The restored catalog did not read back its replacement checkout binding."
        )
    return stored


def restored_project_owners(
    store: AppStore,
    capture: BackupProjectCapture,
    *,
    archived_manifest: Path,
    data_dir: Path,
    local_home: Path,
) -> RestoredProjectOwners:
    """Construct the concrete canonical owners before a restored project is visible."""

    recovery = capture.recovery
    record = store.project(capture.project_id)
    if capture.status != "captured" or recovery is None or record is None:
        raise RestoredProjectPublicationRefused("Only one captured restored project can publish.")
    try:
        manifest = load_manifest(archived_manifest, local_home=local_home)
    except (OSError, ValueError) as exc:
        raise RestoredProjectPublicationRefused(
            "The archived canonical manifest is unavailable or invalid."
        ) from exc
    if (
        BackupManifestConfiguration.from_manifest(manifest) != recovery.configuration
        or record.home_space_id != capture.home_space_id
        or record.home_space_id != store.space_id
        or not (
            (
                record.reachable is False
                and record.error == "Replacement restore publication is pending."
            )
            or (record.reachable is True and record.error is None)
        )
    ):
        raise RestoredProjectPublicationRefused(
            "The archived manifest, restored catalog, and recovery descriptor disagree."
        )
    if record.state_remote:
        try:
            bootstrap = load_manifest(record.locator)
            workspace = state_workspace_for_probe(bootstrap, data_dir)
            workspace.refresh()
        except (OSError, StateUnavailable, ValueError) as exc:
            raise RestoredProjectPublicationRefused(
                "The restored remote canonical checkout is unavailable."
            ) from exc
    else:
        workspace = LocalStateWorkspace(Path(record.state_location), record.state_location)
    history = HistoryManager(
        manifest,
        workspace,
        expected_space_id=store.space_id,
        project_id=record.project_id,
        require_attribution=True,
        agent_authority_resolver=store.agent_task_authority,
        project_membership_check=store.is_project_member,
    )
    paper = PaperService(manifest, store, workspace, project_id=record.project_id)
    service = ProjectService(
        manifest,
        history,
        paper,
        data_dir=data_dir,
        project_id=record.project_id,
        task_continuation_session=store.agent_task_continuation_session_id,
    )
    return RestoredProjectOwners(manifest, workspace, history, paper, service)


def complete_restored_project_publication(
    store: AppStore,
    capture: BackupProjectCapture,
    owners: RestoredProjectOwners,
    materialization: MaterializationResult,
) -> ProjectRecord:
    """Make one replay-proven restored project readable in the stopped catalog."""

    if capture.main_head is None or owners.history.head_ref(materialization) != capture.main_head:
        raise RestoredProjectPublicationRefused(
            "The restored project cannot be exposed at a different canonical head."
        )
    identity = owners.history.project_identity(materialization)
    if (
        identity is None
        or identity.project_id != capture.project_id
        or identity.home_space_id != store.space_id
    ):
        raise RestoredProjectPublicationRefused(
            "The restored canonical identity differs from its catalog home."
        )
    owners.service.chat_summaries(limit=1)
    owners.paper.snapshot()
    record = store.project(capture.project_id)
    if record is None:
        raise RestoredProjectPublicationRefused(
            "The restored project disappeared before catalog publication."
        )
    try:
        stored = store.complete_project_publication_for_restore(
            capture.project_id,
            expected_locator=record.locator,
            revision=materialization.state.revision,
        )
    except (KeyError, RuntimeError, ValueError, sqlite3.Error) as exc:
        raise RestoredProjectPublicationRefused(
            "The restored catalog refused the replay-proven project."
        ) from exc
    if stored.reachable is not True or stored.error is not None:
        raise RestoredProjectPublicationRefused(
            "The restored project availability did not read back from SQLite."
        )
    return stored


def _render_restored_manifest(
    configuration: BackupManifestConfiguration,
    repository_paths: dict[str, str],
) -> str:
    document = tomlkit.document()
    document.add("name", configuration.name)
    machines = tomlkit.aot()
    for item in configuration.machines:
        machine = tomlkit.table()
        machine.add("alias", item.alias)
        machine.add("host", item.host)
        machine.add("os_account", item.os_account)
        if item.provider_paths:
            paths = tomlkit.inline_table()
            for provider in PROVIDER_IDS:
                if provider in item.provider_paths:
                    paths.append(provider, item.provider_paths[provider])
            machine.add("provider_paths", paths)
        machines.append(machine)
    document.add("machines", machines)
    repositories = tomlkit.aot()
    for item in configuration.repositories:
        repository = tomlkit.table()
        repository.add("alias", item.alias)
        repository.add("machine", item.machine)
        repository.add("path", repository_paths[item.alias])
        repositories.append(repository)
    document.add("repositories", repositories)
    project = tomlkit.table()
    project.add("truth_scope", list(configuration.project_truth_scope))
    document.add("project", project)
    state = tomlkit.table()
    state.add("repository", configuration.state_repository)
    document.add("state", state)
    agent = tomlkit.table()
    agent.add("default_run_truth_scope", list(configuration.default_run_truth_scope))
    agent.add(
        "default_auto_research_invocation_ceiling",
        configuration.default_auto_research_invocation_ceiling,
    )
    defaults = tomlkit.table()
    defaults.add("workflow_ids", list(configuration.skill_defaults.workflow_ids))
    defaults.add("skill_ids", list(configuration.skill_defaults.skill_ids))
    agent.add("skill_defaults", defaults)
    profiles = {item.profile: item for item in configuration.agent_profiles}
    for surface in AGENT_EXECUTION_PROFILES:
        item = profiles[surface]
        profile = tomlkit.table()
        profile.add("provider", item.provider)
        profile.add("runtime", item.runtime)
        profile.add("model", item.model)
        profile.add("reasoning", item.reasoning)
        profile.add("run_on", item.run_on)
        permissions = tomlkit.table()
        for key, value in item.permissions.model_dump(mode="json").items():
            permissions.add(key, value)
        profile.add("permissions", permissions)
        agent.add(surface, profile)
    document.add("agent", agent)
    sources = tomlkit.table()
    for name in (
        "claude_roots",
        "codex_roots",
        "remote_claude_roots",
        "remote_codex_roots",
    ):
        sources.add(name, list(getattr(configuration.sources, name)))
    document.add("sources", sources)
    content = tomlkit.dumps(document)
    manifest = Manifest.model_validate(tomlkit.parse(content).unwrap())
    if BackupManifestConfiguration.from_manifest(manifest) != configuration:
        raise RestoredProjectRebindRefused(
            "The recovery descriptor did not reproduce its exact project manifest."
        )
    return content


def _write_restored_bootstrap_manifest(
    data_dir: Path,
    *,
    host: str,
    repository_path: str,
    content: str,
    uid: int,
    gid: int,
) -> Path:
    root = data_dir / "bootstrap-manifests"
    if not root.exists():
        root.mkdir(mode=0o700)
        os.chown(root, uid, gid)
        _fsync_directory(root.parent)
    info = root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_uid, info.st_gid) != (uid, gid)
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise RestoredProjectRebindRefused(
            "The restored bootstrap-manifest root has unsafe ownership or mode."
        )
    digest = hashlib.sha256(f"{host}\0{repository_path}".encode()).hexdigest()[:16]
    path = root / f"{digest}.toml"
    payload = content.encode("utf-8")
    if path.exists() or path.is_symlink():
        if not _restored_bootstrap_matches(path, payload, uid=uid, gid=gid):
            raise RestoredProjectRebindRefused(
                "The restored bootstrap manifest conflicts with existing machine state."
            )
        return path
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=root)
    temporary = Path(temporary_name)
    try:
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(root)
    except FileExistsError:
        if not _restored_bootstrap_matches(path, payload, uid=uid, gid=gid):
            raise RestoredProjectRebindRefused(
                "The restored bootstrap manifest raced with conflicting machine state."
            ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return path


def _restored_bootstrap_matches(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_uid, info.st_gid) != (uid, gid)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != len(payload)
        ):
            return False
        chunks: list[bytes] = []
        remaining = len(payload) + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) == payload
    finally:
        os.close(descriptor)


def _snapshot_payload(snapshot: ProjectSnapshot) -> dict[str, object]:
    if isinstance(snapshot, _ProjectSnapshotDraft):
        return snapshot._as_dict()
    if isinstance(snapshot, dict):
        return snapshot
    raise TypeError("project snapshot must be an internal draft or dictionary")


class ProjectDeletionResult(BaseModel):
    project_id: str
    database_records: dict[str, int]
    removed_stages: int
    removed_display_snapshot: bool
    removed_paper_snapshot: bool


TEAM_PROJECT_DELETE_UNAVAILABLE_REASON = (
    "Team projects cannot be deleted here. A server operator must deprovision the "
    "managed checkout and Git deploy keys."
)


EpisodeSerializer = Callable[
    [str, EpisodeRecord, ExperimentEpisodeProjectionSnapshot | None],
    dict[str, object],
]
ExperimentControlProjector = Callable[
    [GraphState, str, ExperimentLoopRuntime, dict[str, object] | None, str | None],
    dict[str, object],
]


class ProjectCatalog:
    def __init__(
        self,
        data_dir: Path,
        store: AppStore,
        launcher: AgentLauncher,
        provider_skills: ProviderSkillInventoryManager | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.store = store
        self.launcher = launcher
        self.provider_skills = provider_skills
        self._services: dict[str, ProjectService] = {}
        self._services_lock = threading.Lock()
        self._opening: dict[str, Future[tuple[ProjectService, GraphState]]] = {}
        self._deleting: set[str] = set()
        self._snapshot_locks: dict[str, threading.Lock] = {}
        self._snapshot_generations: dict[str, int] = {}
        self._committed_snapshot_generations: dict[str, int] = {}
        self._cached_snapshot_patch_heads: dict[str, int | None] = {}
        self._candidate_snapshot_patch_heads: dict[str, int | None] = {}
        self._registration_lock = threading.Lock()
        self._project_aliases = self.store.project_aliases()

    def register(
        self,
        locator: str,
        *,
        identity_action: Literal["created", "adopted"] | None = None,
        seat_member: str | None = None,
    ) -> ProjectRecord:
        return self._register(
            locator,
            identity_action=identity_action,
            seat_member=seat_member,
            prepared_project_id=None,
            seated_by=None,
        )

    def register_prepared_team_project(
        self,
        locator: str,
        *,
        project_id: str,
        seat_member: str,
    ) -> ProjectRecord:
        """Register the exact identity reserved by a reviewed team request."""

        if self.store.space_kind != "team":
            raise ValueError("prepared team-project registration requires a team space")
        return self._register(
            locator,
            identity_action="created",
            seat_member=seat_member,
            prepared_project_id=project_id,
            seated_by=seat_member,
        )

    def prepare_incoming_transfer_registration(
        self,
        locator: str,
        *,
        project_id: str,
        home_space_id: str,
        expected_manifest_content: str,
    ) -> ProjectRecord:
        """Build a transfer registration receipt without publishing catalog state."""

        if self.store.space_kind != "team":
            raise ValueError("incoming transfer registration requires a team space")
        if home_space_id != self.store.space_id:
            raise ValueError("the incoming transfer belongs to another RCP space")
        with self._registration_lock:
            bootstrap = load_manifest(locator)
            canonical_locator = str(bootstrap.path)
            existing_at_locator = self.store.project_by_locator(canonical_locator)
            existing_for_identity = self.store.project(project_id)
            if existing_at_locator is not None or existing_for_identity is not None:
                raise ProjectIdentityConflict(
                    "The incoming transfer project is already visible in the catalog."
                )
            manifest, workspace = prepare_state_workspace(bootstrap, self.data_dir)
            try:
                actual_manifest_content = manifest.path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError("the imported canonical manifest is unavailable") from exc
            if actual_manifest_content != expected_manifest_content:
                raise ValueError("the imported canonical manifest changed after review")
            history = self._history_for_manifest(manifest, workspace)
            materialization = history.materialize(write_outputs=False)
            if materialization.state.replay_status != "complete":
                raise ValueError("the imported canonical Patch history does not replay completely")
            identity = history.project_identity(materialization)
            if (
                identity is None
                or identity.project_id != project_id
                or identity.home_space_id != home_space_id
            ):
                raise ProjectIdentityConflict(
                    "The imported canonical project identity differs from its reviewed target."
                )
            return self._record_for_identity(
                bootstrap,
                None,
                project_id=project_id,
                home_space_id=home_space_id,
            )

    def refresh_after_incoming_transfer_activation(self, record: ProjectRecord) -> None:
        """Refresh process-local catalog state after the storage compound commit."""

        stored = self.store.project(record.project_id)
        stable_fields = (
            "project_id",
            "home_space_id",
            "locator",
            "name",
            "state_location",
            "state_remote",
            "added_at",
        )
        if stored is None or any(
            getattr(stored, field) != getattr(record, field) for field in stable_fields
        ):
            raise RuntimeError("incoming transfer activation catalog readback differs")
        self._refresh_project_aliases()

    def _register(
        self,
        locator: str,
        *,
        identity_action: Literal["created", "adopted"] | None,
        seat_member: str | None,
        prepared_project_id: str | None,
        seated_by: str | None,
    ) -> ProjectRecord:
        """Register one canonical project after its durable nameplate is settled.

        ``seat_member`` is the acting person, who becomes the project's first
        member. A team space has no other way to know who that is, so its routes
        always supply it; a personal space has exactly one possible member and
        resolves them here.
        """

        with self._registration_lock:
            bootstrap = load_manifest(locator)
            canonical_locator = str(bootstrap.path)
            existing = self.store.project_by_locator(canonical_locator)
            if prepared_project_id is not None:
                if existing is not None and existing.project_id != prepared_project_id:
                    raise ProjectIdentityConflict(
                        "The prepared canonical location is registered as another project."
                    )
                reserved = self.store.project(prepared_project_id)
                if reserved is not None and (
                    reserved.locator != canonical_locator
                    or reserved.home_space_id != self.store.space_id
                ):
                    raise ProjectIdentityConflict(
                        "The prepared project identity is registered at another canonical home."
                    )
            manifest, workspace = prepare_state_workspace(bootstrap, self.data_dir)
            history = self._history_for_manifest(manifest, workspace)
            identity = history.project_identity()
            if identity is None:
                if existing is not None:
                    claim_action: Literal["created", "adopted"] = "adopted"
                elif identity_action is not None:
                    claim_action = identity_action
                else:
                    raise ValueError(
                        "This existing project has no durable identity. Connect it through "
                        "project setup and confirm that this space becomes its sole writable home."
                    )
                identity = history.claim_project_identity(
                    claim_action,
                    project_id=prepared_project_id,
                )
            else:
                # The idempotent claim path also enforces the expected home-space boundary.
                identity = history.claim_project_identity(
                    "created" if prepared_project_id is not None else identity.action,
                    project_id=prepared_project_id,
                )

            if existing is None:
                existing = self.store.project(identity.project_id)

            if existing is not None:
                old_project_id = existing.project_id
            elif identity.action == "adopted":
                old_project_id = _project_id(manifest)
            else:
                old_project_id = identity.project_id

            record = self._record_for_identity(
                bootstrap,
                existing,
                project_id=old_project_id,
                home_space_id=(
                    self.store.space_id if old_project_id == identity.project_id else None
                ),
            )
            if old_project_id == identity.project_id:
                stored = self.store.upsert_project(record)
            else:
                resolved_old = self.store.resolve_project_id(old_project_id)
                if resolved_old != old_project_id:
                    if resolved_old != identity.project_id:
                        raise ValueError(
                            f"Legacy project alias {old_project_id!r} already belongs to "
                            f"{resolved_old!r}."
                        )
                    self._refresh_project_aliases()
                    canonical_record = self._record_for_identity(
                        bootstrap,
                        self.store.project(identity.project_id),
                        project_id=identity.project_id,
                        home_space_id=self.store.space_id,
                    )
                    stored = self.store.upsert_project(canonical_record)
                else:
                    migration = self._prepare_app_file_migration(
                        old_project_id,
                        identity.project_id,
                    )
                    attachment_store = ChatAttachmentStore(self.data_dir / "chat-attachments")
                    attachment_migration = attachment_store.prepare_project_identity_migration(
                        old_project_id,
                        identity.project_id,
                    )
                    self.store.upsert_project(record)
                    stored = self.store.migrate_project_identity(
                        old_project_id,
                        identity.project_id,
                        self.store.space_id,
                    )
                    self._refresh_project_aliases()
                    self._apply_app_file_migration(migration)
                    attachment_store.apply_project_identity_migration(attachment_migration)
                    self._migrate_runtime_keys(old_project_id, identity.project_id)
                stored = self.store.upsert_project(
                    self._record_for_identity(
                        bootstrap,
                        stored,
                        project_id=identity.project_id,
                        home_space_id=self.store.space_id,
                    )
                )
            self._finish_alias_file_migrations(identity.project_id)
            self._seat_first_member(stored.project_id, seat_member, seated_by=seated_by)
            return stored

    def register_degraded_read_only(
        self,
        locator: str,
        *,
        materialization: MaterializationResult,
        seat_member: str | None = None,
    ) -> ProjectRecord:
        """Catalog the last coherent state without claiming or repairing canonical history."""

        with self._registration_lock:
            bootstrap = load_manifest(locator)
            canonical_locator = str(bootstrap.path)
            existing = self.store.project_by_locator(canonical_locator)
            manifest, workspace = prepare_state_workspace(bootstrap, self.data_dir)
            history = self._history_for_manifest(manifest, workspace)
            if materialization.state.replay_status != "degraded":
                raise ValueError(
                    "The retained research now replays completely; open it normally instead."
                )
            identity = history.project_identity(materialization)
            if identity is not None and identity.home_space_id != self.store.space_id:
                raise ProjectIdentityConflict(
                    f"Project {identity.project_id} belongs to space {identity.home_space_id}; "
                    f"this space is {self.store.space_id}. Registration is refused."
                )
            project_id = identity.project_id if identity is not None else _project_id(manifest)
            if existing is not None and existing.project_id != project_id:
                raise ValueError(
                    "This canonical location is already registered as a different project."
                )
            record = self._record_for_identity(
                manifest,
                existing,
                project_id=project_id,
                home_space_id=None,
            ).model_copy(
                update={
                    "revision": materialization.state.revision,
                    "reachable": workspace.reachable,
                    "error": str(ReplayHalted(materialization.state)),
                }
            )
            self.store.upsert_project(record)
            paper = PaperService(
                manifest,
                self.store,
                workspace,
                project_id=project_id,
            )
            service = ProjectService(
                manifest,
                history,
                paper,
                self.launcher,
                data_dir=self.data_dir,
                provider_skills=self.provider_skills,
                project_id=project_id,
                repository_inventory=self.repository_ownership_inventory,
                task_continuation_session=self.store.agent_task_continuation_session_id,
            )
            snapshot = _snapshot_payload(service.project_snapshot(state=materialization.state))
            self._stamp_snapshot_identity(snapshot, project_id)
            self.mark_snapshot_fresh(snapshot)
            self.write_cached_snapshot(project_id, snapshot)
            self._seat_first_member(project_id, seat_member)
            return self.update_summary(project_id, snapshot)

    def _seat_first_member(
        self,
        project_id: str,
        seat_member: str | None,
        *,
        seated_by: str | None = None,
    ) -> None:
        """Seat the acting person on a project that has no members yet.

        Registration is idempotent — reopening an existing project comes back
        through here — so this only ever seats the *first* member. Later members
        arrive through invitations, and a project someone left must not silently
        readmit them by being reopened.

        With nobody acting — `rcp open` from a console, or server startup — a
        personal space resolves its one possible member, and a team space seats
        everyone. Seating nobody would be worse than seating everyone: the
        project would be invisible to every member, and nobody could invite
        themselves to it, so it could never be recovered.
        """

        if self.store.project_members(project_id):
            return
        if seat_member is not None:
            self.store.seat_project_member(project_id, seat_member, seated_by=seated_by)
            return
        owner = self.store.local_owner
        for user in [owner] if owner is not None else self.store.space_users():
            self.store.seat_project_member(project_id, user.user_id)

    def require_archive_available(self, canonical_location: str) -> None:
        """Refuse to archive canonical state still owned by this catalog."""

        record = next(
            (
                item
                for item in self.store.projects()
                if item.state_location.rstrip("/") == canonical_location.rstrip("/")
            ),
            None,
        )
        if record is None:
            return
        if self.store.has_active_agent_task(record.project_id):
            raise ValueError(
                f"Pause the active agent task for {record.name!r} before archiving its "
                "canonical research."
            )
        with self._services_lock:
            opened = record.project_id in self._services or record.project_id in self._opening
        qualifier = "open and " if opened else ""
        raise ValueError(
            f"Project {record.name!r} is {qualifier}registered in this RCP catalog. "
            "Remove that registration before archiving its canonical research."
        )

    def _record_for_identity(
        self,
        manifest: Manifest,
        existing: ProjectRecord | None,
        *,
        project_id: str,
        home_space_id: str | None,
    ) -> ProjectRecord:
        state_repository = manifest.repository_map[manifest.state.repository]
        state_machine = manifest.machine_map[state_repository.machine]
        state_location = (
            f"{state_machine.host}:{state_repository.path}/.research"
            if state_machine.host
            else str(manifest.research_dir)
        )
        return ProjectRecord(
            project_id=project_id,
            home_space_id=home_space_id,
            locator=str(manifest.path),
            name=manifest.name,
            state_location=state_location,
            state_remote=bool(state_machine.host),
            added_at=existing.added_at if existing else self.store.now(),
            last_opened_at=existing.last_opened_at if existing else None,
            revision=existing.revision if existing else None,
            primary_question=existing.primary_question if existing else None,
            attention_count=existing.attention_count if existing else 0,
            last_refresh_at=existing.last_refresh_at if existing else None,
            reachable=existing.reachable if existing else None,
            error=existing.error if existing else None,
        )

    def resolve_project_id(self, project_id: str) -> str:
        """Resolve a project URL without opening SQLite on the request path."""

        return self._project_aliases.get(project_id, project_id)

    def _canonical_project_id(self, project_id: str) -> str:
        return self.resolve_project_id(project_id)

    def _refresh_project_aliases(self) -> None:
        # Replace the snapshot as one object so concurrent request reads never
        # observe a partially refreshed mapping.
        self._project_aliases = self.store.project_aliases()

    def _history_for_manifest(
        self,
        manifest: Manifest,
        workspace: StateWorkspace,
        *,
        project_id: str | None = None,
    ) -> HistoryManager:
        return HistoryManager(
            manifest,
            workspace,
            expected_space_id=self.store.space_id,
            project_id=project_id,
            require_attribution=True,
            agent_authority_resolver=(
                self.store.agent_task_authority if project_id is not None else None
            ),
            project_membership_check=(
                self.store.is_project_member if project_id is not None else None
            ),
        )

    def _ensure_registered_identity(self, project_id: str) -> str:
        project_id = self._canonical_project_id(project_id)
        record = self.store.project(project_id)
        if record is None:
            raise KeyError(project_id)
        if record.home_space_id is not None:
            if record.home_space_id != self.store.space_id:
                raise ProjectIdentityConflict(
                    f"Project {project_id} belongs to space {record.home_space_id}; "
                    f"this space is {self.store.space_id}. Canonical writes are refused."
                )
            return record.project_id
        bootstrap = load_manifest(record.locator)
        manifest, workspace = prepare_state_workspace(bootstrap, self.data_dir)
        history = self._history_for_manifest(manifest, workspace, project_id=record.project_id)
        materialization = history.materialize(write_outputs=False)
        if materialization.state.replay_status == "degraded":
            identity = history.project_identity(materialization)
            if identity is not None:
                if identity.home_space_id != self.store.space_id:
                    raise ProjectIdentityConflict(
                        f"Project {identity.project_id} belongs to space "
                        f"{identity.home_space_id}; this space is {self.store.space_id}. "
                        "Registration is refused."
                    )
                if identity.project_id != record.project_id:
                    raise ProjectIdentityConflict(
                        "The read-only catalog record does not match canonical project identity."
                    )
            return record.project_id
        return self.register(record.locator).project_id

    def _stamp_snapshot_identity(
        self,
        snapshot: dict[str, object],
        project_id: str,
    ) -> None:
        project_id = self._canonical_project_id(project_id)
        record = self.store.project(project_id)
        if record is None:
            raise KeyError(project_id)
        graph = snapshot.get("graph")
        degraded = isinstance(graph, dict) and graph.get("replay_status") == "degraded"
        if record.home_space_id is None and not degraded:
            raise KeyError(project_id)
        snapshot["id"] = project_id
        snapshot["home_space_id"] = record.home_space_id

    def _finish_alias_file_migrations(self, canonical_project_id: str) -> None:
        for alias_id, destination in self._project_aliases.items():
            if destination != canonical_project_id:
                continue
            migration = self._prepare_app_file_migration(alias_id, canonical_project_id)
            attachment_store = ChatAttachmentStore(self.data_dir / "chat-attachments")
            attachment_migration = attachment_store.prepare_project_identity_migration(
                alias_id,
                canonical_project_id,
            )
            self._apply_app_file_migration(migration)
            attachment_store.apply_project_identity_migration(attachment_migration)
            self._migrate_runtime_keys(alias_id, canonical_project_id)

    def _prepare_app_file_migration(
        self,
        old_project_id: str,
        canonical_project_id: str,
    ) -> list[tuple[Literal["display", "paper"], Path, Path, bytes | None]]:
        migrations: list[tuple[Literal["display", "paper"], Path, Path, bytes | None]] = []
        display_source = self._cached_snapshot_path_for_id(old_project_id)
        display_target = self._cached_snapshot_path_for_id(canonical_project_id)
        if display_source != display_target and display_source.exists():
            _require_regular_app_file(display_source, "legacy display snapshot")
            try:
                envelope = json.loads(display_source.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Legacy project display snapshot is invalid.") from exc
            if not isinstance(envelope, dict) or not isinstance(envelope.get("snapshot"), dict):
                raise ValueError("Legacy project display snapshot is invalid.")
            if envelope.get("project_id") != old_project_id:
                raise ValueError("Legacy project display snapshot names a different project.")
            snapshot = envelope["snapshot"]
            assert isinstance(snapshot, dict)
            if snapshot.get("id") != old_project_id:
                raise ValueError("Legacy project display snapshot names a different project.")
            envelope["project_id"] = canonical_project_id
            snapshot["id"] = canonical_project_id
            snapshot["home_space_id"] = self.store.space_id
            content = (
                json.dumps(envelope, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
            ).encode()
            if len(content) > PROJECT_DISPLAY_SNAPSHOT_MAX_BYTES:
                raise ValueError("Migrated project display snapshot exceeds its size limit.")
            if display_target.exists():
                _require_regular_app_file(display_target, "project display snapshot destination")
                if display_target.read_bytes() != content:
                    raise ValueError(
                        "Project display snapshot migration destination already exists; "
                        "nothing was overwritten."
                    )
            migrations.append(("display", display_source, display_target, content))

        paper_source = self._paper_snapshot_path_for_id(old_project_id)
        paper_target = self._paper_snapshot_path_for_id(canonical_project_id)
        if paper_source != paper_target and paper_source.exists():
            _require_regular_app_file(paper_source, "legacy paper snapshot")
            if paper_target.exists():
                raise ValueError(
                    "Project paper snapshot migration destination already exists; "
                    "nothing was overwritten."
                )
            migrations.append(("paper", paper_source, paper_target, None))
        return migrations

    def _apply_app_file_migration(
        self,
        migrations: list[tuple[Literal["display", "paper"], Path, Path, bytes | None]],
    ) -> None:
        for kind, source, target, content in migrations:
            target.parent.mkdir(parents=True, exist_ok=True)
            if kind == "paper":
                os.replace(source, target)
                _fsync_directory(target.parent)
                continue
            assert content is not None
            if target.exists():
                _require_regular_app_file(target, "project display snapshot destination")
                if target.read_bytes() != content:
                    raise ValueError(
                        "Project display snapshot migration destination already exists; "
                        "nothing was overwritten."
                    )
                source.unlink(missing_ok=True)
                _fsync_directory(source.parent)
                continue
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                source.unlink()
                _fsync_directory(target.parent)
                if source.parent != target.parent:
                    _fsync_directory(source.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def _migrate_runtime_keys(self, old_project_id: str, canonical_project_id: str) -> None:
        if old_project_id == canonical_project_id:
            return
        with self._services_lock:
            old_service = self._services.pop(old_project_id, None)
            current_service = self._services.get(canonical_project_id)
            if old_service is not None:
                if current_service is not None and current_service is not old_service:
                    raise RuntimeError("Project identity migration found duplicate open services.")
                self._services[canonical_project_id] = old_service
            old_opening = self._opening.pop(old_project_id, None)
            if old_opening is not None:
                if canonical_project_id in self._opening:
                    raise RuntimeError("Project identity migration found duplicate open attempts.")
                self._opening[canonical_project_id] = old_opening
            if old_project_id in self._deleting:
                self._deleting.remove(old_project_id)
                self._deleting.add(canonical_project_id)
            for mapping in (
                self._snapshot_locks,
                self._snapshot_generations,
                self._committed_snapshot_generations,
                self._cached_snapshot_patch_heads,
                self._candidate_snapshot_patch_heads,
            ):
                if old_project_id not in mapping:
                    continue
                old_value = mapping.pop(old_project_id)
                if canonical_project_id in mapping and mapping[canonical_project_id] != old_value:
                    raise RuntimeError(
                        "Project identity migration found conflicting in-memory cache state."
                    )
                mapping[canonical_project_id] = old_value

    def cards(self) -> list[dict[str, object]]:
        can_delete = self.store.space_kind == "personal"
        return [self._card(record, can_delete=can_delete) for record in self.store.projects()]

    def card(self, project_id: str) -> dict[str, object]:
        project_id = self._canonical_project_id(project_id)
        record = self.store.project(project_id)
        if record is None:
            raise KeyError(project_id)
        return self._card(record, can_delete=self.store.space_kind == "personal")

    def state_host(self, project_id: str) -> str:
        """Read the registered state host without opening canonical history."""

        project_id = self._canonical_project_id(project_id)
        record = self.store.project(project_id)
        if record is None:
            raise KeyError(project_id)
        manifest = load_manifest(record.locator)
        repository = manifest.repository_map[manifest.state.repository]
        return manifest.machine_map[repository.machine].host

    def open(self, project_id: str) -> ProjectService:
        service, _ = self._service_or_open(project_id)
        return service

    def open_transfer_source(self, request_id: str) -> ProjectService:
        """Open the exact source for export, including after its home Patch committed."""

        transfer = self.store.project_transfer_request(request_id)
        if transfer is None or transfer.side != "source":
            raise KeyError(request_id)
        if transfer.phase not in {"source_released", "source_fenced", "archive_bound"}:
            raise ValueError("source transfer is not at an exportable boundary")
        loaded = self.loaded_service(transfer.project_id)
        if loaded is not None:
            return loaded
        if transfer.phase == "source_released":
            try:
                return self.open(transfer.project_id)
            except ProjectIdentityConflict:
                # The home Patch may have committed just before the process
                # stopped, while the SQLite fence receipt is still one step
                # behind. Reopen against the already-bound target home below.
                pass

        record = self.store.project(transfer.project_id)
        if record is None:
            raise KeyError(transfer.project_id)
        bootstrap = load_manifest(record.locator)
        manifest, workspace = prepare_state_workspace(bootstrap, self.data_dir)
        history = HistoryManager(
            manifest,
            workspace,
            expected_space_id=transfer.target_space_id,
            project_id=transfer.project_id,
            require_attribution=True,
            agent_authority_resolver=self.store.agent_task_authority,
            project_membership_check=self.store.is_project_member,
        )
        try:
            initialized = history.initialize()
        except ReplayHalted:
            initialized = history.materialize(write_outputs=False)
        identity = history.project_identity(initialized)
        if (
            identity is None
            or identity.project_id != transfer.project_id
            or identity.home_space_id != transfer.target_space_id
        ):
            raise ProjectIdentityConflict(
                "Departed source history does not match its bound team-space transfer."
            )
        paper = PaperService(
            history.manifest, self.store, workspace, project_id=transfer.project_id
        )
        return ProjectService(
            history.manifest,
            history,
            paper,
            self.launcher,
            data_dir=self.data_dir,
            provider_skills=self.provider_skills,
            project_id=transfer.project_id,
            repository_inventory=self.repository_ownership_inventory,
            task_continuation_session=self.store.agent_task_continuation_session_id,
        )

    def discard_retired_transfer_source(self, request_id: str) -> None:
        """Drop only runtime visibility after the durable source retirement receipt."""

        transfer = self.store.project_transfer_request(request_id)
        if transfer is None or transfer.side != "source":
            raise KeyError(request_id)
        retired = self.store.retired_project(transfer.project_id)
        if (
            retired is None
            or retired.retired_transfer_request_id != transfer.request_id
            or transfer.phase not in {"cleanup_acknowledged", "completed"}
        ):
            raise ValueError("source transfer has no matching durable retirement receipt")
        with self._services_lock:
            opening = self._opening.get(transfer.project_id)
        if opening is not None:
            with suppress(Exception):
                opening.result()
        with self._services_lock:
            self._services.pop(transfer.project_id, None)

    def open_snapshot(self, project_id: str) -> tuple[ProjectService, _ProjectSnapshotDraft]:
        project_id = self._canonical_project_id(project_id)
        service, initialized_state = self._service_or_open(project_id)
        project_id = self._canonical_project_id(project_id)
        snapshot = _snapshot_payload(service.project_snapshot(state=initialized_state))
        self._stamp_snapshot_identity(snapshot, project_id)
        self.mark_snapshot_fresh(snapshot)
        return service, _ProjectSnapshotDraft(snapshot)

    def reconcile_snapshot(
        self,
        project_id: str,
    ) -> tuple[ProjectService, _ProjectSnapshotDraft]:
        """Refresh canonical state and build one fresh display-snapshot candidate."""

        project_id = self._canonical_project_id(project_id)
        service, initialized_state = self._service_or_open(project_id)
        project_id = self._canonical_project_id(project_id)
        if initialized_state is None:
            refreshed = service.history.workspace.refresh()
            if service.history.workspace.remote and not refreshed:
                raise StateUnavailable("Remote canonical state has no readable manifest.")
            initialized_state = service.history.materialize(write_outputs=False).state
        snapshot = _snapshot_payload(service.project_snapshot(state=initialized_state))
        self._stamp_snapshot_identity(snapshot, project_id)
        self.mark_snapshot_fresh(snapshot)
        return service, _ProjectSnapshotDraft(snapshot)

    def probe_remote_patch_log_head(
        self,
        project_id: str,
    ) -> Literal["moved", "unchanged", "unavailable"]:
        """Compare canonical and cached patch heads without opening the project."""

        project_id = self._canonical_project_id(project_id)
        with self._services_lock:
            if project_id in self._deleting:
                raise KeyError(project_id)
            service = self._services.get(project_id)
        if service is not None:
            workspace = service.history.workspace
        else:
            record = self.store.project(project_id)
            if record is None:
                raise KeyError(project_id)
            workspace = state_workspace_for_probe(load_manifest(record.locator), self.data_dir)
        if isinstance(workspace, SSHStateWorkspace):
            available, canonical_head = workspace.probe_remote_patch_log_head()
            if not available:
                return "unavailable"
        else:
            canonical_head = workspace.cached_patch_log_head()
        snapshot = self.cached_snapshot(project_id)
        if snapshot is None:
            raise KeyError(project_id)
        return (
            "moved"
            if canonical_head != self._cached_snapshot_patch_heads.get(project_id)
            else "unchanged"
        )

    @staticmethod
    def mark_snapshot_fresh(snapshot: dict[str, object]) -> None:
        canonical = snapshot.get("canonical_state")
        unreachable_remote = (
            isinstance(canonical, dict)
            and canonical.get("remote") is True
            and canonical.get("reachable") is False
        )
        _ensure_snapshot_freshness(
            snapshot,
            freshness="stale" if unreachable_remote else "fresh",
        )

    def _service_or_open(
        self,
        project_id: str,
    ) -> tuple[ProjectService, GraphState | None]:
        """Open once per project while leaving snapshot work outside the lock."""

        project_id = self._ensure_registered_identity(project_id)
        with self._services_lock:
            if project_id in self._deleting:
                raise KeyError(project_id)
            cached = self._services.get(project_id)
            if cached is not None:
                return cached, None
            opening = self._opening.get(project_id)
            owner = opening is None
            if opening is None:
                opening = Future()
                self._opening[project_id] = opening

        if not owner:
            return opening.result()

        try:
            result = self._open_service(project_id)
        except BaseException as exc:
            with self._services_lock:
                self._opening.pop(project_id, None)
            opening.set_exception(exc)
            raise

        with self._services_lock:
            deleting = project_id in self._deleting
            if not deleting:
                self._services[project_id] = result[0]
            self._opening.pop(project_id, None)
            if not deleting:
                opening.set_result(result)
        if deleting:
            error = KeyError(project_id)
            opening.set_exception(error)
            raise error
        return result

    def readiness_snapshot(
        self,
        project_id: str,
        *,
        refresh: bool = False,
    ) -> dict[str, object]:
        project_id = self._canonical_project_id(project_id)
        with self._services_lock:
            cached = self._services.get(project_id)
        if cached is not None:
            return cached.readiness_snapshot(refresh=refresh)
        record = self.store.project(project_id)
        if record is None:
            raise KeyError(project_id)
        manifest = load_manifest(record.locator)
        snapshot = ProjectService.readiness_for(manifest, self.launcher, refresh=refresh)
        ProjectService.wait_for_provider_skill_inventories_for(manifest, self.provider_skills)
        snapshot["provider_skill_inventories"] = ProjectService.provider_skill_inventories_for(
            manifest,
            self.provider_skills,
        )
        return snapshot

    def provider_targets(self) -> list[tuple[ProviderId, str, str | None]]:
        """Unique configured provider capabilities known to this app process."""

        targets: set[tuple[ProviderId, str, str | None]] = set()
        for record in self.store.projects():
            try:
                manifest = load_manifest(record.locator)
            except (FileNotFoundError, OSError, ValueError):
                continue
            for machine in manifest.machines:
                for provider in PROVIDER_IDS:
                    targets.add((provider, machine.host, machine.provider_paths.get(provider)))
        return sorted(targets, key=lambda item: (item[1], item[0], item[2] or ""))

    def repository_ownership_inventory(self) -> list[RegisteredRepositoryRoot]:
        """Load every registered repository ownership boundary from its manifest."""

        roots: list[RegisteredRepositoryRoot] = []
        for record in self.store.projects():
            try:
                manifest = load_manifest(record.locator)
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise ValueError(
                    "Cannot establish the repository ownership inventory because registered "
                    f"project {record.project_id!r} is unavailable: {exc}"
                ) from exc
            roots.extend(registered_repository_roots(manifest, project_id=record.project_id))
        return sorted(
            roots,
            key=lambda item: (
                item.execution_host,
                item.project_id,
                item.alias,
                item.machine,
                item.path,
            ),
        )

    def delete(self, project_id: str) -> ProjectDeletionResult:
        """Forget one RCP registration without touching any research source."""
        if self.store.space_kind == "team":
            raise ValueError(TEAM_PROJECT_DELETE_UNAVAILABLE_REASON)
        project_id = self._canonical_project_id(project_id)
        with self._services_lock:
            if self.store.project(project_id) is None or project_id in self._deleting:
                raise KeyError(project_id)
            self._deleting.add(project_id)
            self._services.pop(project_id, None)
            opening = self._opening.get(project_id)
        try:
            if opening is not None:
                with suppress(Exception):
                    opening.result()
            with self._snapshot_lock(project_id):
                stages = self.store.project_deletion_stages(project_id)
                display_snapshot = self._cached_snapshot_path(project_id)
                paper_snapshot = self._paper_snapshot_path(project_id)
                project_cache_root = _validate_project_cache_roots(self.data_dir, project_id)
                for stage in stages:
                    self._validate_stage_target(stage)
                _validate_optional_regular_app_file(display_snapshot, "project display snapshot")
                _validate_optional_regular_app_file(paper_snapshot, "project paper snapshot")

                for stage in stages:
                    self._remove_stage(stage)

                removed_display = _unlink_regular_app_file(display_snapshot)
                removed_paper = _unlink_regular_app_file(paper_snapshot)
                _remove_validated_project_cache_root(project_cache_root)
                self._snapshot_generations.pop(project_id, None)
                self._committed_snapshot_generations.pop(project_id, None)
                self._cached_snapshot_patch_heads.pop(project_id, None)
                self._candidate_snapshot_patch_heads.pop(project_id, None)
                database_records = self.store.delete_project_records(project_id)
            return ProjectDeletionResult(
                project_id=project_id,
                database_records=database_records,
                removed_stages=len(stages),
                removed_display_snapshot=removed_display,
                removed_paper_snapshot=removed_paper,
            )
        finally:
            with self._services_lock:
                self._deleting.discard(project_id)

    def discard_unactivated_imported_sources(
        self,
        request_id: str,
        *,
        expected_inventory: ImportedProviderSourceInventory,
    ) -> bool:
        """Clean one failed incoming transfer without becoming team deprovisioning."""

        if self.store.space_kind != "team":
            raise ValueError("incoming transfer cleanup requires a team space")
        transfer = self.store.project_transfer_request(request_id)
        provisioning = self.store.project_provisioning_request(request_id)
        if (
            transfer is None
            or transfer.side != "target"
            or transfer.project_id != expected_inventory.project_id
            or transfer.phase in {"target_activated", "cleanup_acknowledged", "completed"}
            or provisioning is None
            or provisioning.kind != "incoming_transfer"
            or provisioning.proposed_project_id != expected_inventory.project_id
            or provisioning.status == "completed"
        ):
            raise ValueError("imported provider-source cleanup is outside its pending request")
        project_id = expected_inventory.project_id
        with self._snapshot_lock(project_id):
            if self.store.project(project_id) is not None:
                raise ValueError(
                    "imported provider sources cannot be discarded after project registration"
                )
            return ImportedProviderSourceStore(self.data_dir, project_id).discard(
                expected_inventory=expected_inventory
            )

    def _snapshot_lock(self, project_id: str) -> threading.Lock:
        project_id = self._canonical_project_id(project_id)
        with self._services_lock:
            return self._snapshot_locks.setdefault(project_id, threading.Lock())

    def _is_deleting(self, project_id: str) -> bool:
        project_id = self._canonical_project_id(project_id)
        with self._services_lock:
            return project_id in self._deleting

    def _remove_stage(self, stage: ProjectStageRecord) -> None:
        if stage.host:
            remote = RemoteRunStage(stage.host).attach_artifact_source(stage.root)
            if not remote.close():
                raise RuntimeError(
                    f"Could not remove saved run stage {stage.root!r} on {stage.host!r}; "
                    "the project was not deleted."
                )
            return

        boundary = (self.data_dir / "run-stage").resolve()
        target = Path(stage.root)
        if not target.is_absolute() or target.parent.resolve() != boundary:
            raise ValueError("Saved local run stage is outside the RCP staging boundary")
        _remove_local_stage(target)

    def _validate_stage_target(self, stage: ProjectStageRecord) -> None:
        if stage.host:
            RemoteRunStage(stage.host).attach_artifact_source(stage.root)
            return

        boundary = self.data_dir / "run-stage"
        if os.path.lexists(boundary) and not stat.S_ISDIR(boundary.lstat().st_mode):
            raise ValueError("RCP's local staging boundary is unsafe")
        target = Path(stage.root)
        if not target.is_absolute() or target.parent.resolve() != boundary.resolve():
            raise ValueError("Saved local run stage is outside the RCP staging boundary")

    def write_cached_snapshot(
        self,
        project_id: str,
        snapshot: ProjectSnapshot,
    ) -> None:
        project_id = self._canonical_project_id(project_id)
        snapshot = _snapshot_payload(snapshot)
        self._stamp_snapshot_identity(snapshot, project_id)
        _ensure_snapshot_freshness(snapshot)
        with self._snapshot_lock(project_id):
            if self._is_deleting(project_id) or self.store.project(project_id) is None:
                raise KeyError(project_id)
            self._candidate_snapshot_patch_heads[project_id] = _display_patch_log_head(snapshot)
            try:
                self._write_cached_snapshot_locked(project_id, snapshot)
            finally:
                self._candidate_snapshot_patch_heads.pop(project_id, None)

    def commit_cached_snapshot(
        self,
        project_id: str,
        snapshot: ProjectSnapshot,
        *,
        generation: int,
        patch_log_head: int | None | object = _PATCH_LOG_HEAD_UNSET,
    ) -> bool:
        """Commit a display snapshot unless a newer project view already won."""

        project_id = self._canonical_project_id(project_id)
        snapshot = _snapshot_payload(snapshot)
        self._stamp_snapshot_identity(snapshot, project_id)
        _ensure_snapshot_freshness(snapshot)
        with self._snapshot_lock(project_id):
            if self._is_deleting(project_id):
                raise KeyError(project_id)
            record = self.store.project(project_id)
            if record is None:
                raise KeyError(project_id)
            if not _valid_display_snapshot(project_id, snapshot):
                raise ValueError("Project display snapshot is invalid")
            if generation < 1 or generation > self._snapshot_generations.get(project_id, 0):
                raise ValueError("Project display snapshot generation is invalid")
            cached = self._cached_snapshot_locked(project_id)
            persisted_revisions = [
                revision
                for revision in (
                    record.revision,
                    int(cached["revision"]) if cached is not None else None,
                )
                if revision is not None
            ]
            candidate_revision = int(snapshot["revision"])
            if patch_log_head is _PATCH_LOG_HEAD_UNSET:
                patch_log_head = (
                    self._cached_snapshot_patch_heads.get(project_id)
                    if cached is not None and int(cached["revision"]) == candidate_revision
                    else _display_patch_log_head(snapshot)
                )
            if not _valid_patch_log_head(patch_log_head):
                raise ValueError("Project display snapshot patch-log head is invalid")
            if persisted_revisions:
                persisted_revision = max(persisted_revisions)
                if candidate_revision < persisted_revision:
                    return False
                if (
                    candidate_revision == persisted_revision
                    and generation < self._committed_snapshot_generations.get(project_id, 0)
                ):
                    return False
            assert patch_log_head is None or isinstance(patch_log_head, int)
            self._candidate_snapshot_patch_heads[project_id] = patch_log_head
            try:
                self._write_cached_snapshot_locked(project_id, snapshot)
            finally:
                self._candidate_snapshot_patch_heads.pop(project_id, None)
            self._committed_snapshot_generations[project_id] = max(
                generation,
                self._committed_snapshot_generations.get(project_id, 0),
            )
            self.update_summary(project_id, snapshot)
            return True

    def update_cached_snapshot_freshness(
        self,
        project_id: str,
        freshness: Literal["fresh", "reconciling", "stale"],
    ) -> bool:
        """Version one freshness-only cache update through the normal guards."""

        project_id = self._canonical_project_id(project_id)
        current = self.cached_snapshot(project_id)
        if current is None:
            return False
        if current.get("snapshot_freshness") == freshness:
            return True
        generation = self.reserve_cached_snapshot_generation(project_id)
        snapshot = self.cached_snapshot(project_id)
        if snapshot is None:
            return False
        snapshot["snapshot_freshness"] = freshness
        return self.commit_cached_snapshot(project_id, snapshot, generation=generation)

    def reserve_cached_snapshot_generation(self, project_id: str) -> int:
        """Reserve construction order for one future display snapshot candidate."""

        project_id = self._canonical_project_id(project_id)
        with self._snapshot_lock(project_id):
            if self._is_deleting(project_id) or self.store.project(project_id) is None:
                raise KeyError(project_id)
            generation = self._snapshot_generations.get(project_id, 0) + 1
            self._snapshot_generations[project_id] = generation
            return generation

    def _write_cached_snapshot_locked(
        self,
        project_id: str,
        snapshot: dict[str, object],
    ) -> None:
        project_id = self._canonical_project_id(project_id)
        if not _valid_display_snapshot(project_id, snapshot):
            raise ValueError("Project display snapshot is invalid")
        patch_log_head = self._candidate_snapshot_patch_heads.get(
            project_id,
            _display_patch_log_head(snapshot),
        )
        envelope = {
            "schema_version": _DISPLAY_SNAPSHOT_SCHEMA_VERSION,
            "project_id": project_id,
            "canonical_patch_head": patch_log_head,
            "snapshot": snapshot,
        }
        content = (
            json.dumps(
                _DISPLAY_SNAPSHOT_ENVELOPE_ADAPTER.dump_python(envelope, mode="json"),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
        if len(content) > PROJECT_DISPLAY_SNAPSHOT_MAX_BYTES:
            raise ValueError("Project display snapshot exceeds its size limit")

        target = self._cached_snapshot_path(project_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
            self._cached_snapshot_patch_heads[project_id] = patch_log_head
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def cached_snapshot(self, project_id: str) -> dict[str, object] | None:
        project_id = self._canonical_project_id(project_id)
        _status, snapshot = self.cached_snapshot_status(project_id)
        return snapshot

    def cached_snapshot_status(
        self,
        project_id: str,
    ) -> tuple[Literal["missing", "invalid", "valid"], dict[str, object] | None]:
        project_id = self._canonical_project_id(project_id)
        with self._snapshot_lock(project_id):
            if self._is_deleting(project_id) or self.store.project(project_id) is None:
                return "missing", None
            return self._cached_snapshot_status_locked(project_id)

    def _cached_snapshot_locked(self, project_id: str) -> dict[str, object] | None:
        project_id = self._canonical_project_id(project_id)
        _status, snapshot = self._cached_snapshot_status_locked(project_id)
        return snapshot

    def _cached_snapshot_status_locked(
        self,
        project_id: str,
    ) -> tuple[Literal["missing", "invalid", "valid"], dict[str, object] | None]:
        project_id = self._canonical_project_id(project_id)
        self._cached_snapshot_patch_heads.pop(project_id, None)
        path = self._cached_snapshot_path(project_id)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return "missing", None
        except OSError:
            return "invalid", None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > PROJECT_DISPLAY_SNAPSHOT_MAX_BYTES
        ):
            return "invalid", None
        try:
            content = path.read_bytes()
        except OSError:
            return "invalid", None
        if not content or len(content) > PROJECT_DISPLAY_SNAPSHOT_MAX_BYTES:
            return "invalid", None
        try:
            envelope = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            return "invalid", None
        if not isinstance(envelope, dict):
            return "invalid", None
        schema_version = envelope.get("schema_version")
        if schema_version == 1 and set(envelope) == {
            "schema_version",
            "project_id",
            "snapshot",
        }:
            patch_log_head = None
        elif schema_version in {2, _DISPLAY_SNAPSHOT_SCHEMA_VERSION} and set(envelope) == {
            "schema_version",
            "project_id",
            "canonical_patch_head",
            "snapshot",
        }:
            patch_log_head = envelope["canonical_patch_head"]
            if not _valid_patch_log_head(patch_log_head):
                return "invalid", None
        else:
            return "invalid", None
        if envelope["project_id"] != project_id:
            return "invalid", None
        snapshot = envelope["snapshot"]
        if not isinstance(snapshot, dict):
            return "invalid", None
        if not _migrate_legacy_display_snapshot_settings(snapshot):
            return "invalid", None
        # Pre-identity display caches did not carry the catalog's home-space field.
        allow_pre_identity = False
        if "home_space_id" not in snapshot:
            record = self.store.project(project_id)
            if record is None:
                return "invalid", None
            snapshot["home_space_id"] = record.home_space_id
            allow_pre_identity = record.home_space_id is None
        _ensure_snapshot_freshness(snapshot)
        if "attention" not in snapshot:
            graph_payload = snapshot.get("graph")
            if not isinstance(graph_payload, dict):
                return "invalid", None
            try:
                graph = GraphState.model_validate(graph_payload)
                attention = project_graph_attention(graph)
                snapshot["attention"] = attention.model_dump(mode="json")
                counts = snapshot.get("counts")
                if not isinstance(counts, dict):
                    return "invalid", None
                counts.update(
                    {
                        "pending_proposals": len(attention.pending_proposal_ids),
                        "decisions_awaiting_choice": len(attention.decisions_awaiting_choice_ids),
                        "open_blockers": len(attention.open_blocker_ids),
                    }
                )
            except (TypeError, ValueError):
                return "invalid", None
        if not _valid_display_snapshot(
            project_id,
            snapshot,
            allow_pre_identity=allow_pre_identity,
        ):
            return "invalid", None
        if schema_version == 1:
            patch_log_head = _display_patch_log_head(snapshot)
        assert patch_log_head is None or isinstance(patch_log_head, int)
        self._cached_snapshot_patch_heads[project_id] = patch_log_head
        return "valid", snapshot

    def loaded_service(self, project_id: str) -> ProjectService | None:
        """Return an already-open service without opening or refreshing it."""

        project_id = self._canonical_project_id(project_id)
        with self._services_lock:
            return self._services.get(project_id)

    def _cached_snapshot_path(self, project_id: str) -> Path:
        project_id = self._canonical_project_id(project_id)
        return self._cached_snapshot_path_for_id(project_id)

    def _cached_snapshot_path_for_id(self, project_id: str) -> Path:
        digest = hashlib.sha256(project_id.encode()).hexdigest()
        return self.data_dir / "project-snapshots" / f"{digest}.json"

    def _paper_snapshot_path(self, project_id: str) -> Path:
        project_id = self._canonical_project_id(project_id)
        return self._paper_snapshot_path_for_id(project_id)

    def _paper_snapshot_path_for_id(self, project_id: str) -> Path:
        safe_project_id = re.sub(r"[^A-Za-z0-9._-]+", "_", project_id).strip("._")
        return (
            self.data_dir
            / "paper-snapshots"
            / (f"{(safe_project_id or 'project')[:80]}-introduction.md")
        )

    def _open_service(self, project_id: str) -> tuple[ProjectService, GraphState]:
        project_id = self._canonical_project_id(project_id)
        record = self.store.project(project_id)
        if record is None:
            raise KeyError(project_id)
        bootstrap = load_manifest(record.locator)
        manifest, workspace = prepare_state_workspace(bootstrap, self.data_dir)
        history = self._history_for_manifest(manifest, workspace, project_id=project_id)
        try:
            initialized = history.initialize()
        except ReplayHalted:
            initialized = history.materialize(write_outputs=False)
        identity = history.project_identity(initialized)
        if identity is None and initialized.state.replay_status != "degraded":
            raise RuntimeError(
                "Registered project identity disappeared before the project could open."
            )
        if identity is not None and identity.home_space_id != self.store.space_id:
            raise ProjectIdentityConflict(
                f"Project {identity.project_id} belongs to space {identity.home_space_id}; "
                f"this space is {self.store.space_id}. Registration is refused."
            )
        if identity is not None and identity.project_id != project_id:
            raise RuntimeError(
                f"Registered project id {project_id!r} does not match canonical history "
                f"{identity.project_id!r}."
            )
        initialized_state = initialized.state
        if initialized_state.replay_status != "degraded":
            self.store.migrate_legacy_project_data(history.manifest.name, project_id)
        paper = PaperService(
            history.manifest,
            self.store,
            workspace,
            project_id=project_id,
        )
        service = ProjectService(
            history.manifest,
            history,
            paper,
            self.launcher,
            data_dir=self.data_dir,
            provider_skills=self.provider_skills,
            project_id=project_id,
            repository_inventory=self.repository_ownership_inventory,
            task_continuation_session=self.store.agent_task_continuation_session_id,
        )
        return service, initialized_state

    def update_summary(
        self,
        project_id: str,
        snapshot: ProjectSnapshot,
    ) -> ProjectRecord:
        project_id = self._canonical_project_id(project_id)
        snapshot = _snapshot_payload(snapshot)
        with self._services_lock:
            if project_id in self._deleting or self.store.project(project_id) is None:
                raise KeyError(project_id)
        primary = snapshot.get("primary_question")
        primary_question = None
        if isinstance(primary, dict):
            primary_question = str(primary.get("question") or primary.get("title") or "") or None
        counts = snapshot["counts"]
        assert isinstance(counts, dict)
        canonical = snapshot["canonical_state"]
        assert isinstance(canonical, dict)
        last_refresh = snapshot.get("last_refresh_at")
        return self.store.update_project_summary(
            project_id,
            revision=int(snapshot["revision"]),
            primary_question=primary_question,
            attention_count=sum(
                int(counts[key])
                for key in (
                    "pending_proposals",
                    "decisions_awaiting_choice",
                    "open_blockers",
                )
            ),
            last_refresh_at=_timestamp(last_refresh),
            reachable=bool(canonical["reachable"]),
            error=str(canonical["error"]) if canonical.get("error") else None,
        )

    def update_settings(
        self,
        project_id: str,
        request: ProjectSettingsRequest,
    ) -> _ProjectSnapshotDraft:
        project_id = self._canonical_project_id(project_id)
        generation = self.reserve_cached_snapshot_generation(project_id)
        service = self.open(project_id)
        project_id = self._canonical_project_id(project_id)
        service.update_settings(request)
        self._persist_bootstrap_locator(project_id, service)
        snapshot = _snapshot_payload(service.project_snapshot())
        self._stamp_snapshot_identity(snapshot, project_id)
        self.mark_snapshot_fresh(snapshot)
        self.commit_cached_snapshot(
            project_id,
            snapshot,
            generation=generation,
            patch_log_head=service.history.workspace.cached_patch_log_head(),
        )
        return _ProjectSnapshotDraft(snapshot)

    def resolve_provider_path(
        self,
        project_id: str,
        machine_alias: str,
        provider: ProviderId,
    ) -> dict[str, object]:
        project_id = self._canonical_project_id(project_id)
        generation = self.reserve_cached_snapshot_generation(project_id)
        service = self.open(project_id)
        project_id = self._canonical_project_id(project_id)
        readiness = service.resolve_provider_path(machine_alias, provider)
        self._persist_bootstrap_locator(project_id, service)
        snapshot = _snapshot_payload(service.project_snapshot())
        self._stamp_snapshot_identity(snapshot, project_id)
        self.mark_snapshot_fresh(snapshot)
        self.commit_cached_snapshot(
            project_id,
            snapshot,
            generation=generation,
            patch_log_head=service.history.workspace.cached_patch_log_head(),
        )
        return {
            "machine": machine_alias,
            "provider": provider,
            "binary_path": readiness.binary_path,
            "readiness": readiness.model_dump(mode="json"),
            "project": _ProjectSnapshotDraft(snapshot),
        }

    def _persist_bootstrap_locator(
        self,
        project_id: str,
        service: ProjectService,
    ) -> None:
        project_id = self._canonical_project_id(project_id)
        record = self.store.project(project_id)
        assert record is not None
        locator = Path(record.locator)
        if locator.resolve() != service.manifest.path.resolve():
            temp = locator.with_name(f".{locator.name}.{os.getpid()}.tmp")
            temp.write_text(service.manifest.path.read_text(encoding="utf-8"), encoding="utf-8")
            os.replace(temp, locator)

    @staticmethod
    def _card(record: ProjectRecord, *, can_delete: bool) -> dict[str, object]:
        return {
            "id": record.project_id,
            "home_space_id": record.home_space_id,
            "name": record.name,
            "locator": record.locator,
            "state_location": record.state_location,
            "remote": record.state_remote,
            "last_opened_at": record.last_opened_at,
            "revision": record.revision,
            "primary_question": record.primary_question,
            "attention_count": record.attention_count,
            "last_refresh_at": record.last_refresh_at,
            "reachable": record.reachable,
            "error": record.error,
            "can_delete": can_delete,
            "delete_unavailable_reason": (
                None if can_delete else TEAM_PROJECT_DELETE_UNAVAILABLE_REASON
            ),
        }


class ProjectDisplayCache:
    """Keeps project display snapshots current with graph and episode state."""

    def __init__(
        self,
        store: AppStore,
        catalog: ProjectCatalog,
        *,
        serialize_episode: EpisodeSerializer,
        project_experiment_control: ExperimentControlProjector,
        logger: logging.Logger,
    ) -> None:
        self._store = store
        self._catalog = catalog
        self._serialize_episode = serialize_episode
        self._project_experiment_control = project_experiment_control
        self._logger = logger
        self._reconciliation_tasks: dict[str, asyncio.Task[None]] = {}
        self._probe_started_at: dict[str, float] = {}

    @property
    def reconciliation_tasks(self) -> dict[str, asyncio.Task[None]]:
        return self._reconciliation_tasks

    def complete_snapshot(
        self,
        project_id: str,
        snapshot: ProjectSnapshot,
        *,
        fresh: bool = False,
    ) -> dict[str, object]:
        """Complete one internal draft or saved snapshot for public display."""

        payload = _snapshot_payload(snapshot)
        self._catalog._stamp_snapshot_identity(payload, project_id)
        if fresh:
            self._catalog.mark_snapshot_fresh(payload)
        self._complete_live_control(project_id, payload)
        return payload

    def open_snapshot(self, project_id: str) -> tuple[ProjectService, dict[str, object]]:
        service, draft = self._catalog.open_snapshot(project_id)
        return service, self.complete_snapshot(project_id, draft)

    def cached_project_snapshot(self, project_id: str) -> dict[str, object] | None:
        snapshot = self._catalog.cached_snapshot(project_id)
        if snapshot is None:
            return None
        return self.complete_snapshot(project_id, snapshot)

    def reconcile_snapshot(self, project_id: str) -> tuple[ProjectService, dict[str, object]]:
        service, draft = self._catalog.reconcile_snapshot(project_id)
        return service, self.complete_snapshot(project_id, draft)

    def update_settings(
        self,
        project_id: str,
        request: ProjectSettingsRequest,
    ) -> dict[str, object]:
        return self.complete_snapshot(
            project_id,
            self._catalog.update_settings(project_id, request),
        )

    def resolve_provider_path(
        self,
        project_id: str,
        machine_alias: str,
        provider: ProviderId,
    ) -> dict[str, object]:
        result = self._catalog.resolve_provider_path(project_id, machine_alias, provider)
        result["project"] = self.complete_snapshot(project_id, result["project"])
        return result

    def complete_transition_control(
        self,
        project_id: str,
        snapshot: dict[str, object],
    ) -> dict[str, object]:
        payload = dict(snapshot)
        self._complete_live_control(project_id, payload)
        return payload

    def transition_payload(
        self,
        project_id: str,
        projection: ProjectTransitionProjection,
        *,
        reconcile_operational: bool,
    ) -> dict[str, object]:
        """Combine one graph/head projection with matching run controls."""

        payload = projection.model_dump(mode="json")
        control_snapshot: dict[str, object] = {"graph": payload["graph"]}
        if reconcile_operational:
            control_snapshot = self.complete_transition_control(
                project_id,
                control_snapshot,
            )
        else:
            state = projection.graph
            experiment_ids = [
                node.id for node in state.nodes.values() if isinstance(node, Experiment)
            ]
            read_models = self._store.experiment_control_projection_snapshots(
                project_id,
                experiment_ids,
                graph_target=GraphTargetRef(),
            )
            controls: dict[str, object] = {}
            for experiment_id in experiment_ids:
                read_model = read_models[experiment_id]
                runtime = read_model.runtime
                episode_snapshot = read_model.episode
                serialized_episode = (
                    self._serialize_episode(
                        project_id,
                        episode_snapshot.episode,
                        episode_snapshot,
                    )
                    if episode_snapshot is not None
                    else None
                )
                control = self._project_experiment_control(
                    state,
                    experiment_id,
                    runtime,
                    serialized_episode,
                    read_model.latest_report_episode_id,
                )
                controls[experiment_id] = control
            control_snapshot["experiment_control"] = controls
        payload["experiment_control"] = control_snapshot["experiment_control"]
        return payload

    def refresh_cached_project_after_stream(
        self,
        project_id: str,
        kind: AgentTaskKind,
        request: AgentTaskRequest,
        execution: AgentTaskExecution,
    ) -> None:
        if not task_graph_capable(kind, request):
            return
        try:
            service = self._catalog.loaded_service(project_id)
            if service is None:
                raise RuntimeError("The closed stream's project service is no longer loaded.")
            generation = self._catalog.reserve_cached_snapshot_generation(project_id)
            cache_status, cached = self._catalog.cached_snapshot_status(project_id)
            if cache_status == "missing":
                record = self._store.project(project_id)
                if record is None or record.revision is None:
                    return
                raise ValueError("The expected project display snapshot is missing.")
            if cached is None:
                raise ValueError("The existing project display snapshot is invalid.")
            state = service.history.materialize(write_outputs=False).state
            paper = PaperSnapshot.model_validate(cached["paper"])
            snapshot = self.complete_snapshot(
                project_id,
                service.project_snapshot(state=state, paper=paper),
                fresh=True,
            )
            self._catalog.commit_cached_snapshot(
                project_id,
                snapshot,
                generation=generation,
                patch_log_head=service.history.workspace.cached_patch_log_head(),
            )
        except Exception as exc:
            self._logger.warning(
                "Could not refresh display snapshot after task %s for project %s: %s",
                execution.operation_id,
                project_id,
                exc,
            )
            try:
                self._store.record_agent_task_receipt(
                    execution.operation_id,
                    "display_cache_refresh_failed",
                    {"exception_type": type(exc).__name__, "detail": str(exc)},
                    tier="diagnostic",
                )
            except Exception as receipt_exc:
                self._logger.warning(
                    "Could not record display cache refresh failure for task %s: %s",
                    execution.operation_id,
                    receipt_exc,
                )

    def _complete_live_control(
        self,
        project_id: str,
        snapshot: dict[str, object],
    ) -> None:
        """Add current operational Experiment state to one display snapshot.

        ``ProjectService`` has no task store, so graph-only snapshots deliberately
        omit this field. The display boundary is the only place that restores it.
        """

        state = GraphState.model_validate(snapshot["graph"])
        experiment_ids = [node.id for node in state.nodes.values() if node.type == "experiment"]
        initial_read_models = self._store.experiment_control_projection_snapshots(
            project_id,
            experiment_ids,
            graph_target=GraphTargetRef(),
        )
        settle_ids = [
            experiment_id
            for experiment_id, read_model in initial_read_models.items()
            if read_model.runtime.stop_requested
            and not read_model.runtime.stop_settled
            and not read_model.runtime.task_active
        ]
        for experiment_id in settle_ids:
            episode_snapshot = initial_read_models[experiment_id].episode
            if episode_snapshot is not None:
                episode = episode_snapshot.episode
                self._store.settle_experiment_loop_stop(
                    project_id,
                    experiment_id,
                    episode_id=episode.episode_id,
                    graph_target=episode.graph_target,
                )
        read_models = (
            self._store.experiment_control_projection_snapshots(
                project_id,
                experiment_ids,
                graph_target=GraphTargetRef(),
            )
            if settle_ids
            else initial_read_models
        )
        controls: dict[str, object] = {}
        for experiment_id in experiment_ids:
            read_model = read_models[experiment_id]
            runtime = read_model.runtime
            episode_snapshot = read_model.episode
            serialized_episode = (
                self._serialize_episode(
                    project_id,
                    episode_snapshot.episode,
                    episode_snapshot,
                )
                if episode_snapshot is not None
                else None
            )
            control = self._project_experiment_control(
                state,
                experiment_id,
                runtime,
                serialized_episode,
                read_model.latest_report_episode_id,
            )
            controls[experiment_id] = control
        snapshot["experiment_control"] = controls

    async def reconcile_cached_project(self, project_id: str) -> None:
        try:
            head_status = await asyncio.to_thread(
                self._catalog.probe_remote_patch_log_head,
                project_id,
            )
            if head_status == "unavailable":
                await asyncio.to_thread(
                    self._catalog.update_cached_snapshot_freshness,
                    project_id,
                    "stale",
                )
                return
            if head_status == "unchanged":
                await asyncio.to_thread(
                    self._catalog.update_cached_snapshot_freshness,
                    project_id,
                    "fresh",
                )
                return

            await asyncio.to_thread(
                self._catalog.update_cached_snapshot_freshness,
                project_id,
                "reconciling",
            )
            generation = await asyncio.to_thread(
                self._catalog.reserve_cached_snapshot_generation,
                project_id,
            )
            service, snapshot = await asyncio.to_thread(
                self.reconcile_snapshot,
                project_id,
            )
            await asyncio.to_thread(
                self._catalog.commit_cached_snapshot,
                project_id,
                snapshot,
                generation=generation,
                patch_log_head=service.history.workspace.cached_patch_log_head(),
            )
        except KeyError:
            return
        except Exception as exc:
            self._logger.warning(
                "Could not reconcile display snapshot for %s: %s",
                project_id,
                exc,
            )
            with suppress(KeyError, OSError, TypeError, ValueError):
                await asyncio.to_thread(
                    self._catalog.update_cached_snapshot_freshness,
                    project_id,
                    "stale",
                )

    def schedule_project_reconciliation(self, project_id: str) -> None:
        task = self._reconciliation_tasks.get(project_id)
        if task is not None and not task.done():
            return
        now = time.monotonic()
        last_started = self._probe_started_at.get(project_id)
        if (
            last_started is not None
            and now - last_started < REMOTE_STATE_HEAD_PROBE_INTERVAL_SECONDS
        ):
            return
        self._probe_started_at[project_id] = now
        task = asyncio.create_task(self.reconcile_cached_project(project_id))
        self._reconciliation_tasks[project_id] = task

        def forget(completed: asyncio.Task[None]) -> None:
            if self._reconciliation_tasks.get(project_id) is completed:
                self._reconciliation_tasks.pop(project_id, None)

        task.add_done_callback(forget)


def _project_id(manifest: Manifest) -> str:
    repository = manifest.repository_map[manifest.state.repository]
    machine = manifest.machine_map[repository.machine]
    identity = f"{manifest.name}\0{machine.host}\0{repository.path}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", manifest.name.lower()).strip("-") or "project"
    return f"{slug[:42]}-{digest}"


def _timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _remove_local_stage(target: Path) -> None:
    if not os.path.lexists(target):
        return
    if target.is_symlink() or not target.is_dir():
        target.unlink()
    else:
        _make_tree_writable(target)
        shutil.rmtree(target)
    if os.path.lexists(target):
        raise OSError(f"Local cleanup left {target} behind")


def _make_tree_writable(target: Path) -> None:
    if target.is_symlink():
        return
    target.chmod(0o700 if target.is_dir() else 0o600)
    if target.is_dir():
        for child in target.iterdir():
            _make_tree_writable(child)


def _unlink_regular_app_file(target: Path) -> bool:
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Refusing to remove non-file app snapshot: {target}")
    target.unlink()
    return True


def _validate_project_cache_roots(data_dir: Path, project_id: str) -> Path | None:
    source_root, slice_root = project_cache_roots(data_dir, project_id)
    cache_parent = data_dir / "project-caches"
    project_root = source_root.parent
    expected = cache_parent / hashlib.sha256(project_id.encode()).hexdigest()
    if project_root != expected or slice_root.parent != expected:
        raise ValueError("Project cache roots do not match the canonical project cache boundary.")

    _validate_optional_directory(cache_parent, "project cache parent")
    if not os.path.lexists(project_root):
        return None
    _validate_optional_directory(project_root, "project cache boundary")
    allowed_children = {source_root, slice_root}
    try:
        unexpected = [child for child in project_root.iterdir() if child not in allowed_children]
    except OSError as exc:
        raise ValueError(f"Could not inspect project cache boundary: {project_root}") from exc
    if unexpected:
        raise ValueError(f"Refusing to clear unknown project cache entry: {unexpected[0]}")
    _validate_cache_tree(source_root, "remote-source cache")
    _validate_cache_tree(slice_root, "session-slice cache")
    return project_root


def _validate_optional_directory(target: Path, label: str) -> None:
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"Could not inspect {label}: {target}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Refusing to clear unsafe {label}: {target}")


def _validate_cache_tree(root: Path, label: str) -> None:
    _validate_optional_directory(root, f"{label} root")
    if not os.path.lexists(root):
        return
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = list(directory.iterdir())
        except OSError as exc:
            raise ValueError(f"Could not inspect {label}: {directory}") from exc
        for child in children:
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise ValueError(f"Could not inspect {label} entry: {child}") from exc
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
            elif not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"Refusing to clear unsafe {label} entry: {child}")


def _remove_validated_project_cache_root(project_root: Path | None) -> None:
    if project_root is None:
        return
    _make_tree_writable(project_root)
    shutil.rmtree(project_root)


def _validate_optional_regular_app_file(target: Path, label: str) -> None:
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"Could not inspect {label}: {target}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Refusing to remove non-file {label}: {target}")


def _require_regular_app_file(target: Path, label: str) -> None:
    try:
        metadata = target.lstat()
    except OSError as exc:
        raise ValueError(f"Could not inspect {label}: {target}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Refusing to migrate non-file {label}: {target}")


def _valid_display_snapshot(
    project_id: str,
    snapshot: dict[str, object],
    *,
    allow_pre_identity: bool = False,
) -> bool:
    if not _DISPLAY_SNAPSHOT_FIELDS.issubset(snapshot):
        return False
    if "default_campaign_invocation_ceiling" in snapshot:
        return False
    if snapshot.get("id") != project_id or not isinstance(snapshot.get("name"), str):
        return False
    home_space_id = snapshot.get("home_space_id")
    if home_space_id is None:
        graph = snapshot.get("graph")
        if not allow_pre_identity and (
            not isinstance(graph, dict) or graph.get("replay_status") != "degraded"
        ):
            return False
    else:
        if not isinstance(home_space_id, str):
            return False
        try:
            parsed_home = uuid.UUID(home_space_id)
        except ValueError:
            return False
        if str(parsed_home) != home_space_id or parsed_home.version != 4:
            return False
    revision = snapshot.get("revision")
    if type(revision) is not int or revision < 0:
        return False
    auto_research_ceiling = snapshot.get("default_auto_research_invocation_ceiling")
    if type(auto_research_ceiling) is not int or auto_research_ceiling < 1:
        return False
    if snapshot.get("snapshot_freshness") not in {"fresh", "reconciling", "stale"}:
        return False
    last_remote_sync_at = snapshot.get("last_remote_sync_at")
    if last_remote_sync_at is not None and not isinstance(last_remote_sync_at, str):
        return False
    if not all(
        isinstance(snapshot.get(key), dict)
        for key in (
            "canonical_state",
            "attention",
            "counts",
            "coverage",
            "graph",
            "paper",
            "paper_coach",
            "agent_profiles",
            "provider_readiness",
            "provider_skill_inventories",
            "providers",
            "cache_metrics",
        )
    ):
        return False
    graph_payload = snapshot["graph"]
    assert isinstance(graph_payload, dict)
    try:
        graph_state = GraphState.model_validate(graph_payload)
        attention = GraphAttentionProjection.model_validate(snapshot["attention"])
        expected_attention = project_graph_attention(graph_state)
    except (TypeError, ValueError):
        return False
    if attention != expected_attention:
        return False
    counts = snapshot["counts"]
    assert isinstance(counts, dict)
    expected_counts = {
        "pending_proposals": len(attention.pending_proposal_ids),
        "decisions_awaiting_choice": len(attention.decisions_awaiting_choice_ids),
        "open_blockers": len(attention.open_blocker_ids),
    }
    if any(
        type(counts.get(key)) is not int or counts[key] != value
        for key, value in expected_counts.items()
    ):
        return False
    if not all(
        isinstance(snapshot.get(key), list)
        for key in (
            "project_truth_scope",
            "default_run_truth_scope",
            "repositories",
            "machines",
            "validation_messages",
        )
    ):
        return False
    graph_revision = graph_payload.get("revision")
    return type(graph_revision) is int and graph_revision == revision


def _migrate_legacy_display_snapshot_settings(snapshot: dict[str, object]) -> bool:
    if not _migrate_legacy_display_snapshot_runtimes(snapshot):
        return False
    legacy_key = "default_campaign_invocation_ceiling"
    current_key = "default_auto_research_invocation_ceiling"
    if legacy_key not in snapshot:
        snapshot.setdefault(current_key, DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING)
        return True
    if current_key in snapshot and snapshot[current_key] != snapshot[legacy_key]:
        return False
    snapshot.setdefault(current_key, snapshot[legacy_key])
    del snapshot[legacy_key]
    return True


def _migrate_legacy_display_snapshot_runtimes(snapshot: dict[str, object]) -> bool:
    """Name the runtime on profiles cached before runtime selection existed.

    `agent_profiles` is part of the cached payload, so the first read after this
    upgrade would otherwise hand the settings form a profile with no runtime at
    all. The manifest resolves an omitted value to the provider default; this
    resolves it the same way rather than leaving the field absent.
    """

    profiles = snapshot.get("agent_profiles")
    if not isinstance(profiles, dict):
        return True
    for profile in profiles.values():
        if not isinstance(profile, dict):
            return False
        provider = profile.get("provider")
        if not isinstance(provider, str):
            return False
        runtime = profile.get("runtime")
        if runtime is not None and not isinstance(runtime, str):
            return False
        try:
            profile["runtime"] = configured_runtime(provider, runtime)
        except ValueError:
            # A retired provider or runtime cannot be named. Re-deriving the
            # snapshot from the manifest is cheaper than guessing.
            return False
    return True


def _ensure_snapshot_freshness(
    snapshot: dict[str, object],
    *,
    freshness: Literal["fresh", "reconciling", "stale"] | None = None,
) -> None:
    canonical = snapshot.get("canonical_state")
    remote = isinstance(canonical, dict) and canonical.get("remote") is True
    if freshness is not None:
        snapshot["snapshot_freshness"] = freshness
    elif snapshot.get("snapshot_freshness") not in {"fresh", "reconciling", "stale"}:
        snapshot["snapshot_freshness"] = "stale" if remote else "fresh"
    if "last_remote_sync_at" not in snapshot:
        last_synced_at = canonical.get("last_synced_at") if isinstance(canonical, dict) else None
        snapshot["last_remote_sync_at"] = str(last_synced_at) if remote and last_synced_at else None


def _display_patch_log_head(snapshot: dict[str, object]) -> int | None:
    revisions = [snapshot.get("revision")]
    graph = snapshot.get("graph")
    replay_failure = graph.get("replay_failure") if isinstance(graph, dict) else None
    if isinstance(replay_failure, dict):
        revisions.append(replay_failure.get("revision"))
    return max(
        (revision for revision in revisions if type(revision) is int and revision > 0),
        default=None,
    )


def _valid_patch_log_head(value: object) -> bool:
    return value is None or (type(value) is int and value > 0)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
