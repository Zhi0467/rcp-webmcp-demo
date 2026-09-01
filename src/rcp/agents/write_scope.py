from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp.agents.context import RepositoryPointer
from rcp.config import Manifest, RepositoryConfig
from rcp.providers import AgentCapability
from rcp.transport.run_stage import RemoteRunStage


class WritableRepositoryRoot(BaseModel):
    """One exact project repository admitted on the execution machine."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    alias: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    path: str = Field(min_length=1)


class RegisteredRepositoryRoot(BaseModel):
    """One repository root owned by a project in the application catalog."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    project_id: str = Field(min_length=1)
    alias: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    execution_host: str
    path: str = Field(min_length=1)


class ProjectWriteScope(BaseModel):
    """Provider-neutral, canonical filesystem scope for one Work-like launch."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_generation: Literal[1] = 1
    project_id: str = Field(min_length=1)
    execution_machine: str = Field(min_length=1)
    execution_host: str
    capability: Literal["work_auto", "orchestrate"]
    stage_root: str = Field(min_length=1)
    workspace_root: str = Field(min_length=1)
    repositories: list[WritableRepositoryRoot] = Field(default_factory=list)
    protected_write_paths: list[str] = Field(default_factory=list)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_canonical_scope(self) -> ProjectWriteScope:
        for value in (self.stage_root, self.workspace_root):
            if not PurePosixPath(value).is_absolute():
                raise ValueError("write-scope stage paths must be absolute")
        stage = PurePosixPath(self.stage_root)
        workspace = PurePosixPath(self.workspace_root)
        if workspace != stage and stage not in workspace.parents:
            raise ValueError("write-scope workspace must be inside its exact task stage")
        repository_keys = [(item.alias, item.machine, item.path) for item in self.repositories]
        if repository_keys != sorted(set(repository_keys)):
            raise ValueError("write-scope repositories must be sorted and unique")
        paths = [item.path for item in self.repositories]
        if any(not PurePosixPath(item).is_absolute() for item in paths):
            raise ValueError("write-scope repository roots must be absolute")
        if len(paths) != len(set(paths)):
            raise ValueError("write-scope repository roots must be exact and unique")
        if self.protected_write_paths != sorted(set(self.protected_write_paths)):
            raise ValueError("protected write paths must be sorted and unique")
        if any(not PurePosixPath(item).is_absolute() for item in self.protected_write_paths):
            raise ValueError("protected write paths must be absolute")
        expected = _scope_fingerprint(self._fingerprint_payload())
        if self.fingerprint != expected:
            raise ValueError("write-scope fingerprint does not match its canonical roots")
        return self

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        execution_machine: str,
        execution_host: str,
        capability: Literal["work_auto", "orchestrate"],
        stage_root: str,
        workspace_root: str,
        repositories: list[WritableRepositoryRoot],
        protected_write_paths: list[str],
    ) -> ProjectWriteScope:
        payload: dict[str, object] = {
            "schema_generation": 1,
            "project_id": project_id,
            "execution_machine": execution_machine,
            "execution_host": execution_host,
            "capability": capability,
            "stage_root": stage_root,
            "workspace_root": workspace_root,
            "repositories": [
                item.model_dump(mode="json")
                for item in sorted(
                    repositories,
                    key=lambda item: (item.alias, item.machine, item.path),
                )
            ],
            "protected_write_paths": sorted(set(protected_write_paths)),
        }
        return cls.model_validate({**payload, "fingerprint": _scope_fingerprint(payload)})

    @property
    def repository_roots(self) -> list[str]:
        return [item.path for item in self.repositories]

    @property
    def writable_roots(self) -> list[str]:
        return list(dict.fromkeys([self.workspace_root, *self.repository_roots]))

    def _fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"fingerprint"})


def resolve_project_write_scope(
    *,
    manifest: Manifest,
    project_id: str,
    execution_machine: str,
    capability: AgentCapability,
    stage_root: str,
    workspace_root: str,
    admitted_aliases: list[str],
    repository_pointers: list[RepositoryPointer],
    remote_stage: RemoteRunStage | None,
    app_data_dir: Path | None,
    repository_inventory: list[RegisteredRepositoryRoot],
) -> ProjectWriteScope:
    """Resolve and verify one exact Work-like scope on its execution machine."""

    if capability not in {"work_auto", "orchestrate"}:
        raise ValueError(f"capability {capability!r} has no project write scope")
    if not project_id:
        raise ValueError("project write scope requires a durable project id")
    machine = manifest.machine_map.get(execution_machine)
    if machine is None:
        raise ValueError(f"unknown write-scope execution machine: {execution_machine}")
    if bool(machine.host) != (remote_stage is not None):
        raise ValueError("write-scope execution host does not match its task stage")
    if remote_stage is not None and remote_stage.host != machine.host:
        raise ValueError("write-scope remote stage belongs to a different execution host")

    aliases = sorted(set(admitted_aliases))
    if aliases != sorted(admitted_aliases):
        raise ValueError("write-scope repository aliases must be sorted and unique")
    project_aliases = set(manifest.project.truth_scope)
    if not set(aliases).issubset(project_aliases):
        unknown = sorted(set(aliases) - project_aliases)
        raise ValueError(f"write scope names repositories outside this project: {unknown}")

    pointers: dict[str, RepositoryPointer] = {}
    for pointer in repository_pointers:
        if pointer.alias in pointers:
            raise ValueError(f"duplicate repository pointer in write scope: {pointer.alias}")
        pointers[pointer.alias] = pointer

    eligible = [
        manifest.repository_map[alias]
        for alias in aliases
        if manifest.repository_map[alias].machine == execution_machine
    ]
    declared_paths: list[str] = [stage_root, workspace_root]
    for repository in eligible:
        pointer = pointers.get(repository.alias)
        if pointer is None:
            raise ValueError(f"write scope is missing repository pointer {repository.alias!r}")
        # Only the machine is compared. A pointer's host says how the *agent*
        # reaches the repository -- `_stage_context_paths` blanks it for a
        # repository that lives on the execution machine, because the agent
        # opens it as a local path there -- while `machine.host` says where the
        # machine is. Comparing them refused every remote run and passed every
        # local one by comparing a value to itself. The pointer's path is what
        # this scope must agree with, and `registered_root != pointer_root`
        # below checks it against the execution host's own filesystem.
        if pointer.machine != repository.machine:
            raise ValueError(
                f"repository {repository.alias!r} does not match its project execution machine"
            )
        declared_paths.extend([repository.path, pointer.path])

    canonical, account_home = _canonical_directories(
        declared_paths,
        remote_stage=remote_stage,
        require_writable=True,
    )
    canonical_stage = canonical[stage_root]
    canonical_workspace = canonical[workspace_root]
    if remote_stage is not None:
        assert remote_stage.root is not None
        if canonical_stage != str(remote_stage.root) or canonical_workspace != str(
            remote_stage.workspace
        ):
            raise ValueError("remote write scope does not match its exact RCP task stage")

    execution_inventory = [
        item for item in repository_inventory if item.execution_host == machine.host
    ]
    inventory_paths = [item.path for item in execution_inventory]
    canonical_inventory, _inventory_home = _canonical_directories(
        inventory_paths,
        remote_stage=remote_stage,
        require_writable=False,
    )

    repository_roots: list[WritableRepositoryRoot] = []
    for repository in eligible:
        pointer = pointers[repository.alias]
        registered_root = canonical[repository.path]
        pointer_root = canonical[pointer.path]
        if registered_root != pointer_root:
            raise ValueError(
                f"repository {repository.alias!r} no longer matches its registered project root"
            )
        ownership = [
            item
            for item in execution_inventory
            if item.project_id == project_id
            and item.alias == repository.alias
            and item.machine == repository.machine
            and item.path == repository.path
        ]
        if len(ownership) != 1:
            raise ValueError(
                f"repository {repository.alias!r} is missing from the canonical project inventory"
            )
        if canonical_inventory[ownership[0].path] != registered_root:
            raise ValueError(
                f"repository {repository.alias!r} changed while its ownership was verified"
            )
        _reject_broad_repository_root(
            registered_root,
            account_home=account_home,
            app_data_dir=app_data_dir if remote_stage is None else None,
        )
        _reject_repository_ownership_overlap(
            repository=repository,
            project_id=project_id,
            admitted_owners={
                (project_id, item.alias, item.machine, item.path) for item in eligible
            },
            registered_root=registered_root,
            inventory=execution_inventory,
            canonical_inventory=canonical_inventory,
        )
        repository_roots.append(
            WritableRepositoryRoot(
                alias=repository.alias,
                machine=repository.machine,
                path=registered_root,
            )
        )

    state_repository = manifest.repository_map[manifest.state.repository]
    state_research_declared = str(PurePosixPath(state_repository.path) / ".research")
    protected = [str(PurePosixPath(item.path) / ".research") for item in repository_roots]
    try:
        state_canonical, _unused_home = _canonical_directories(
            [state_research_declared],
            remote_stage=remote_stage,
            require_writable=False,
        )
    except (OSError, ValueError):
        # The canonical state directory may be temporarily unavailable during a
        # degraded read. Its lexical path still remains an explicit write deny.
        protected.append(state_research_declared)
    else:
        protected.extend([state_research_declared, state_canonical[state_research_declared]])

    return ProjectWriteScope.create(
        project_id=project_id,
        execution_machine=execution_machine,
        execution_host=machine.host,
        capability=capability,
        stage_root=canonical_stage,
        workspace_root=canonical_workspace,
        repositories=repository_roots,
        protected_write_paths=protected,
    )


def registered_repository_roots(
    manifest: Manifest,
    *,
    project_id: str,
) -> list[RegisteredRepositoryRoot]:
    """Project-owned repository roots in deterministic catalog-inventory form."""

    roots = [
        RegisteredRepositoryRoot(
            project_id=project_id,
            alias=repository.alias,
            machine=repository.machine,
            execution_host=manifest.machine_map[repository.machine].host,
            path=repository.path,
        )
        for repository in manifest.repositories
    ]
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


def _canonical_directories(
    paths: list[str],
    *,
    remote_stage: RemoteRunStage | None,
    require_writable: bool,
) -> tuple[dict[str, str], str]:
    declared = list(dict.fromkeys(paths))
    if remote_stage is not None:
        return remote_stage.canonical_directories(declared, require_writable=require_writable)
    canonical: dict[str, str] = {}
    for raw in declared:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise ValueError(f"project repository root must be absolute: {raw}")
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError(f"project repository root is unavailable: {raw}") from exc
        if not resolved.is_dir():
            raise ValueError(f"project repository root is not a directory: {raw}")
        if require_writable and not os.access(resolved, os.W_OK):
            raise ValueError(f"project repository root is not writable: {raw}")
        canonical[raw] = str(resolved)
    return canonical, str(Path.home().resolve())


def _reject_broad_repository_root(
    root: str,
    *,
    account_home: str,
    app_data_dir: Path | None,
) -> None:
    candidate = PurePosixPath(root)
    broad_temporary_roots = {
        PurePosixPath("/tmp"),
        PurePosixPath("/private/tmp"),
        PurePosixPath(str(Path(tempfile.gettempdir()).resolve())),
    }
    if candidate == PurePosixPath("/"):
        raise ValueError("the filesystem root cannot be a project repository write root")
    if candidate == PurePosixPath(account_home):
        raise ValueError("the execution account home cannot be a project repository write root")
    if candidate in broad_temporary_roots:
        raise ValueError("a broad temporary directory cannot be a project repository write root")
    if app_data_dir is None:
        return
    data_root = app_data_dir.expanduser().resolve()
    local_root = Path(root)
    if (
        local_root == data_root
        or data_root in local_root.parents
        or local_root in data_root.parents
    ):
        raise ValueError("the RCP application data directory cannot be a repository write root")


def _reject_repository_ownership_overlap(
    *,
    repository: RepositoryConfig,
    project_id: str,
    admitted_owners: set[tuple[str, str, str, str]],
    registered_root: str,
    inventory: list[RegisteredRepositoryRoot],
    canonical_inventory: dict[str, str],
) -> None:
    alias = repository.alias
    declared_root = repository.path
    for owner in inventory:
        is_admitted_owner = (
            owner.project_id,
            owner.alias,
            owner.machine,
            owner.path,
        ) in admitted_owners
        if is_admitted_owner:
            continue
        if not (
            _paths_overlap(declared_root, owner.path)
            or _paths_overlap(registered_root, canonical_inventory[owner.path])
        ):
            continue
        relation = (
            "another project" if owner.project_id != project_id else "an unadmitted repository"
        )
        raise ValueError(
            f"repository {alias!r} overlaps {relation} on this execution host: "
            f"{owner.project_id}/{owner.alias}"
        )


def _paths_overlap(left: str, right: str) -> bool:
    left_path = PurePosixPath(os.path.normpath(left))
    right_path = PurePosixPath(os.path.normpath(right))
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def _scope_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
