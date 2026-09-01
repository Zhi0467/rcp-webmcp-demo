from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from rcp.providers import (
    DEFAULT_PROVIDER,
    AgentCapability,
    ProviderId,
    configured_runtime,
)
from rcp.skill_registry import SkillDefaults

DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING = 10


class MachineConfig(BaseModel):
    alias: str
    host: str = ""
    os_account: str = ""
    provider_paths: dict[ProviderId, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider_paths(self) -> MachineConfig:
        normalized: dict[ProviderId, str] = {}
        for provider, raw_path in self.provider_paths.items():
            path = raw_path.strip()
            if not path:
                continue
            absolute = PurePosixPath(path).is_absolute() if self.host else Path(path).is_absolute()
            if not absolute:
                target = self.host or "the local machine"
                raise ValueError(f"provider path for {provider!r} on {target} must be absolute")
            normalized[provider] = path
        self.provider_paths = normalized
        return self

    @model_validator(mode="after")
    def validate_os_account(self) -> MachineConfig:
        if (
            self.os_account
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,127}", self.os_account) is None
        ):
            raise ValueError("machine operating-system account is invalid")
        return self


class RepositoryConfig(BaseModel):
    alias: str
    machine: str
    path: str


class ProjectConfig(BaseModel):
    truth_scope: list[str]


class StateConfig(BaseModel):
    repository: str


AgentSurface = Literal["seed", "refresh", "node_chat", "project_chat", "paper_coach"]
AgentExecutionProfile = AgentSurface | Literal["orchestrator"]

GRAPH_AGENT_SURFACES: tuple[AgentSurface, ...] = (
    "seed",
    "refresh",
    "node_chat",
    "project_chat",
)
GRAPH_AGENT_EXECUTION_PROFILES: tuple[AgentExecutionProfile, ...] = (
    *GRAPH_AGENT_SURFACES,
    "orchestrator",
)


class AgentPermissions(BaseModel):
    read_graph: bool
    read_research_md: bool
    read_introduction: bool
    read_repositories: Literal["none", "run_scope", "project_scope"]
    read_conversations: Literal["none", "run_scope"]
    write_graph_patch: bool
    write_project_files: bool = False
    write_paper: bool = False


class AgentSurfaceConfig(BaseModel):
    provider: ProviderId = DEFAULT_PROVIDER
    runtime: str = ""
    model: str = ""
    reasoning: str = "medium"
    run_on: str
    permissions: AgentPermissions | None = None

    @model_validator(mode="after")
    def validate_provider_runtime(self) -> AgentSurfaceConfig:
        self.runtime = configured_runtime(self.provider, self.runtime)
        return self


class AgentConfig(BaseModel):
    default_run_truth_scope: list[str]
    default_auto_research_invocation_ceiling: int = Field(
        default=DEFAULT_AUTO_RESEARCH_INVOCATION_CEILING,
        ge=1,
        description="Operational invocations per newly authorized episode.",
    )
    skill_defaults: SkillDefaults = Field(default_factory=SkillDefaults)
    seed: AgentSurfaceConfig | None = None
    refresh: AgentSurfaceConfig | None = None
    node_chat: AgentSurfaceConfig | None = None
    project_chat: AgentSurfaceConfig | None = None
    paper_coach: AgentSurfaceConfig | None = None
    orchestrator: AgentSurfaceConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_auto_research_ceiling(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        legacy_key = "default_campaign_invocation_ceiling"
        current_key = "default_auto_research_invocation_ceiling"
        if legacy_key not in migrated:
            return migrated
        if current_key in migrated and migrated[current_key] != migrated[legacy_key]:
            raise ValueError(f"agent.{legacy_key} conflicts with agent.{current_key}")
        migrated.setdefault(current_key, migrated[legacy_key])
        del migrated[legacy_key]
        return migrated


class SourcesConfig(BaseModel):
    claude_roots: list[str] = Field(default_factory=lambda: ["~/.claude/projects"])
    codex_roots: list[str] = Field(default_factory=lambda: ["~/.codex/sessions"])
    remote_claude_roots: list[str] = Field(default_factory=lambda: ["~/.claude/projects"])
    remote_codex_roots: list[str] = Field(default_factory=lambda: ["~/.codex/sessions"])


class ExecutionConfig(BaseModel):
    run_on: str


class PaperCoachConfig(BaseModel):
    default_provider: ProviderId = DEFAULT_PROVIDER
    default_model: str = ""
    default_reasoning: str = "medium"


class Manifest(BaseModel):
    name: str
    machines: list[MachineConfig]
    repositories: list[RepositoryConfig]
    project: ProjectConfig
    state: StateConfig
    agent: AgentConfig
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    execution: ExecutionConfig | None = None
    paper: dict[str, Any] = Field(default_factory=dict)
    _path: Path = PrivateAttr()

    @model_validator(mode="after")
    def validate_references(self) -> Manifest:
        machine_aliases = [item.alias for item in self.machines]
        repository_aliases = [item.alias for item in self.repositories]
        if len(machine_aliases) != len(set(machine_aliases)):
            raise ValueError("machine aliases must be unique")
        if len(repository_aliases) != len(set(repository_aliases)):
            raise ValueError("repository aliases must be unique")
        machines = set(machine_aliases)
        repositories = set(repository_aliases)
        missing_machines = {item.machine for item in self.repositories} - machines
        if missing_machines:
            raise ValueError(f"repositories use unknown machines: {sorted(missing_machines)}")
        unknown_scope = set(self.project.truth_scope) - repositories
        if unknown_scope:
            raise ValueError(
                f"project truth scope uses unknown repositories: {sorted(unknown_scope)}"
            )
        if self.state.repository not in repositories:
            raise ValueError("state.repository must name a registered repository")
        if self.state.repository not in self.project.truth_scope:
            raise ValueError("state.repository must remain in project.truth_scope in v1")
        default_outside = set(self.agent.default_run_truth_scope) - set(self.project.truth_scope)
        if default_outside or not self.agent.default_run_truth_scope:
            raise ValueError("agent.default_run_truth_scope must be a non-empty project subset")
        legacy_execution = self.execution
        if legacy_execution is None and any(
            getattr(self.agent, surface) is None for surface in _AGENT_SURFACES
        ):
            raise ValueError(
                "agent profiles must all be configured when legacy [execution] is absent"
            )
        legacy_coach = PaperCoachConfig.model_validate(self.paper.get("coach", {}))
        state_machine = self.repository_map[self.state.repository].machine
        for surface in _AGENT_SURFACES:
            profile = getattr(self.agent, surface)
            if profile is None:
                assert legacy_execution is not None
                profile = AgentSurfaceConfig(
                    provider=(
                        legacy_coach.default_provider if surface == "paper_coach" else "codex"
                    ),
                    model=legacy_coach.default_model if surface == "paper_coach" else "",
                    reasoning=(
                        legacy_coach.default_reasoning if surface == "paper_coach" else "medium"
                    ),
                    run_on=legacy_execution.run_on,
                )
                setattr(self.agent, surface, profile)
        if self.agent.orchestrator is None:
            refresh = self.agent.refresh
            assert refresh is not None
            self.agent.orchestrator = refresh.model_copy(
                deep=True,
                update={"permissions": permissions_for("orchestrate")},
            )
        for surface in AGENT_EXECUTION_PROFILES:
            profile = getattr(self.agent, surface)
            assert profile is not None
            if profile.run_on not in machines:
                raise ValueError(f"agent.{surface}.run_on must name a registered machine")
            if surface in GRAPH_AGENT_EXECUTION_PROFILES and profile.run_on != state_machine:
                raise ValueError(
                    f"agent.{surface}.run_on must be the canonical state machine "
                    f"{state_machine!r}; graph-writing agents run beside .research/"
                )
            expected = permissions_for(surface)
            if profile.permissions is None:
                profile.permissions = expected
            elif surface in {
                "node_chat",
                "project_chat",
            } and profile.permissions == permissions_for("scratch_patch"):
                # Before conversation modes, chat surfaces carried the same
                # patch permission as ingestion. Accept that one exact legacy
                # contract, but normalize it to Discuss in memory so a saved
                # manifest cannot silently authorize Work.
                profile.permissions = expected
            elif profile.permissions != expected:
                raise ValueError(
                    f"agent.{surface}.permissions cannot widen or narrow the "
                    f"{surface.replace('_', ' ')} safety contract"
                )
        return self

    @property
    def path(self) -> Path:
        return self._path

    @property
    def research_dir(self) -> Path:
        return self._path.parent

    @property
    def repository_map(self) -> dict[str, RepositoryConfig]:
        return {item.alias: item for item in self.repositories}

    @property
    def machine_map(self) -> dict[str, MachineConfig]:
        return {item.alias: item for item in self.machines}

    @property
    def repository_paths(self) -> dict[str, str]:
        """Registered alias to its path, for reading a declared path back to its alias."""

        return {item.alias: item.path for item in self.repositories}

    @property
    def coach(self) -> PaperCoachConfig:
        profile = self.agent_profile("paper_coach")
        return PaperCoachConfig(
            default_provider=profile.provider,
            default_model=profile.model,
            default_reasoning=profile.reasoning,
        )

    def agent_profile(self, surface: AgentExecutionProfile) -> AgentSurfaceConfig:
        profile = getattr(self.agent, surface)
        assert profile is not None
        return profile


_AGENT_SURFACES: tuple[AgentSurface, ...] = (
    "seed",
    "refresh",
    "node_chat",
    "project_chat",
    "paper_coach",
)
AGENT_EXECUTION_PROFILES: tuple[AgentExecutionProfile, ...] = (
    *_AGENT_SURFACES,
    "orchestrator",
)


def permissions_for(target: AgentExecutionProfile | AgentCapability) -> AgentPermissions:
    """Return the immutable authority envelope for one launch capability."""

    capability: AgentCapability
    if target in {"seed", "refresh"}:
        capability = "scratch_patch"
    elif target in {"node_chat", "project_chat"}:
        capability = "discuss"
    elif target == "paper_coach":
        capability = "paper_readonly"
    elif target == "discuss":
        capability = "discuss"
    elif target == "work_auto":
        capability = "work_auto"
    elif target in {"orchestrator", "orchestrate"}:
        capability = "orchestrate"
    elif target == "scratch_patch":
        capability = "scratch_patch"
    elif target == "paper_readonly":
        capability = "paper_readonly"
    else:
        raise ValueError(f"Unknown agent surface or capability: {target!r}")

    if capability == "paper_readonly":
        return AgentPermissions(
            read_graph=True,
            read_research_md=True,
            read_introduction=True,
            read_repositories="project_scope",
            read_conversations="none",
            write_graph_patch=False,
            write_project_files=False,
            write_paper=False,
        )
    if capability == "discuss":
        return AgentPermissions(
            read_graph=True,
            read_research_md=True,
            read_introduction=True,
            read_repositories="run_scope",
            read_conversations="run_scope",
            write_graph_patch=False,
            write_project_files=False,
            write_paper=False,
        )
    if capability in {"work_auto", "orchestrate"}:
        return AgentPermissions(
            read_graph=True,
            read_research_md=True,
            read_introduction=True,
            read_repositories="run_scope",
            read_conversations="run_scope",
            write_graph_patch=True,
            write_project_files=True,
            write_paper=False,
        )
    if capability == "scratch_patch":
        return AgentPermissions(
            read_graph=True,
            read_research_md=True,
            read_introduction=True,
            read_repositories="run_scope",
            read_conversations="run_scope",
            write_graph_patch=True,
            write_project_files=False,
            write_paper=False,
        )
    raise ValueError(f"Unknown agent capability: {capability!r}")


def resolve_manifest_path(
    value: str | os.PathLike[str],
    *,
    local_home: Path | None = None,
) -> Path:
    path = _expand_local_user_path(str(value), local_home=local_home).resolve()
    if path.is_file():
        return path
    direct = path / "manifest.toml"
    if direct.is_file():
        return direct
    nested = path / ".research" / "manifest.toml"
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"No manifest.toml found at {path}")


def load_manifest(
    value: str | os.PathLike[str],
    *,
    local_home: Path | None = None,
) -> Manifest:
    path = resolve_manifest_path(value, local_home=local_home)
    data = tomlkit.parse(path.read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(data.unwrap())
    manifest._path = path
    project_root = path.parent.parent
    for repository in manifest.repositories:
        repository_path = _expand_local_user_path(repository.path, local_home=local_home)
        if not repository_path.is_absolute() and not manifest.machine_map[repository.machine].host:
            repository.path = str((project_root / repository_path).resolve())
    manifest.sources.claude_roots = [
        _resolve_local_source_root(project_root, root, local_home=local_home)
        for root in manifest.sources.claude_roots
    ]
    manifest.sources.codex_roots = [
        _resolve_local_source_root(project_root, root, local_home=local_home)
        for root in manifest.sources.codex_roots
    ]
    return manifest


def write_project_scope(
    manifest: Manifest,
    truth_scope: list[str],
    repository_descriptor: dict[str, str] | None = None,
) -> Manifest:
    content = _project_scope_content(manifest, truth_scope, repository_descriptor)
    _atomic_write(manifest.path, content)
    return load_manifest(manifest.path)


def validate_project_scope_update(
    manifest: Manifest,
    truth_scope: list[str],
    repository_descriptor: dict[str, str] | None = None,
) -> None:
    _project_scope_content(manifest, truth_scope, repository_descriptor)


def _project_scope_content(
    manifest: Manifest,
    truth_scope: list[str],
    repository_descriptor: dict[str, str] | None,
) -> str:
    if repository_descriptor and repository_descriptor.get("machine") not in manifest.machine_map:
        raise ValueError(
            f"repository uses unknown machine: {repository_descriptor.get('machine')!r}"
        )
    document = tomlkit.parse(manifest.path.read_text(encoding="utf-8"))
    if repository_descriptor:
        aliases = {item.get("alias") for item in document.get("repositories", [])}
        if repository_descriptor["alias"] not in aliases:
            repository = tomlkit.table()
            repository.add("alias", repository_descriptor["alias"])
            repository.add("machine", repository_descriptor["machine"])
            repository.add("path", repository_descriptor["path"])
            repositories = document.get("repositories")
            if repositories is None:
                repositories = tomlkit.aot()
                document.add("repositories", repositories)
            repositories.append(repository)
    document["project"]["truth_scope"] = list(truth_scope)
    content = tomlkit.dumps(document)
    Manifest.model_validate(tomlkit.parse(content).unwrap())
    return content


def write_agent_settings(
    manifest: Manifest,
    default_run_truth_scope: list[str],
    profiles: dict[AgentExecutionProfile, AgentSurfaceConfig],
    provider_path_updates: dict[str, dict[ProviderId, str]] | None = None,
    skill_defaults: SkillDefaults | None = None,
    default_auto_research_invocation_ceiling: int | None = None,
) -> Manifest:
    document = tomlkit.parse(manifest.path.read_text(encoding="utf-8"))
    agent = document.get("agent")
    if agent is None:
        agent = tomlkit.table()
        document.add("agent", agent)
    agent["default_run_truth_scope"] = list(default_run_truth_scope)
    agent.pop("default_campaign_invocation_ceiling", None)
    agent["default_auto_research_invocation_ceiling"] = (
        manifest.agent.default_auto_research_invocation_ceiling
        if default_auto_research_invocation_ceiling is None
        else default_auto_research_invocation_ceiling
    )
    selected_defaults = skill_defaults or manifest.agent.skill_defaults
    defaults_table = tomlkit.table()
    defaults_table.add("workflow_ids", list(selected_defaults.workflow_ids))
    defaults_table.add("skill_ids", list(selected_defaults.skill_ids))
    agent["skill_defaults"] = defaults_table

    for surface in AGENT_EXECUTION_PROFILES:
        # A settings write is a merge: a client that predates a surface omits it,
        # and the omitted surface keeps what the manifest already holds rather
        # than failing the whole save.
        profile = profiles.get(surface) or getattr(manifest.agent, surface)
        if profile is None:
            raise ValueError(f"agent.{surface} has no configuration to write")
        table = tomlkit.table()
        table.add("provider", profile.provider)
        table.add("runtime", profile.runtime)
        table.add("model", profile.model)
        table.add("reasoning", profile.reasoning)
        table.add("run_on", profile.run_on)
        permission_table = tomlkit.table()
        for key, value in permissions_for(surface).model_dump(mode="json").items():
            permission_table.add(key, value)
        table.add("permissions", permission_table)
        agent[surface] = table

    if "execution" in document:
        del document["execution"]
    paper = document.get("paper")
    if paper is not None and "coach" in paper:
        del paper["coach"]
        if not paper:
            del document["paper"]

    _apply_machine_provider_path_updates(document, provider_path_updates or {})

    content = tomlkit.dumps(document)
    Manifest.model_validate(tomlkit.parse(content).unwrap())
    _atomic_write(manifest.path, content)
    return load_manifest(manifest.path)


def write_machine_provider_paths(
    manifest: Manifest,
    provider_path_updates: dict[str, dict[ProviderId, str]],
) -> Manifest:
    document = tomlkit.parse(manifest.path.read_text(encoding="utf-8"))
    _apply_machine_provider_path_updates(document, provider_path_updates)
    content = tomlkit.dumps(document)
    Manifest.model_validate(tomlkit.parse(content).unwrap())
    _atomic_write(manifest.path, content)
    return load_manifest(manifest.path)


def _apply_machine_provider_path_updates(
    document: tomlkit.TOMLDocument,
    updates: dict[str, dict[ProviderId, str]],
) -> None:
    if not updates:
        return
    machines = document.get("machines", [])
    machine_tables = {str(machine.get("alias")): machine for machine in machines}
    unknown = set(updates) - set(machine_tables)
    if unknown:
        raise ValueError(f"provider paths use unknown machines: {sorted(unknown)}")
    for alias, provider_updates in updates.items():
        machine = machine_tables[alias]
        merged = dict(machine.get("provider_paths", {}))
        for provider, raw_path in provider_updates.items():
            path = raw_path.strip()
            if path:
                merged[provider] = path
            else:
                merged.pop(provider, None)
        if merged:
            paths = tomlkit.inline_table()
            for provider, path in merged.items():
                paths.append(provider, path)
            machine["provider_paths"] = paths
        elif "provider_paths" in machine:
            del machine["provider_paths"]


def _atomic_write(path: Path, content: str) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def _expand_local_user_path(value: str, *, local_home: Path | None) -> Path:
    if local_home is not None and (value == "~" or value.startswith("~/")):
        if not local_home.is_absolute():
            raise ValueError("local manifest home must be absolute")
        return local_home if value == "~" else local_home / value[2:]
    return Path(value).expanduser()


def _resolve_local_source_root(
    project_root: Path,
    value: str,
    *,
    local_home: Path | None,
) -> str:
    path = _expand_local_user_path(value, local_home=local_home)
    if path.is_absolute() or value.startswith("~"):
        return str(path)
    return str((project_root / path).resolve())
