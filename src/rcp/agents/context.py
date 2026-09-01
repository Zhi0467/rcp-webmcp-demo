from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rcp.config import Manifest
from rcp.core.models import GraphState, Patch
from rcp.core.operations import SetCoverageOperation
from rcp.history.delta import RefreshDelta
from rcp.providers import PROVIDERS
from rcp.transport import RepositoryAccess


class RepositoryPointer(BaseModel):
    alias: str
    machine: str
    host: str = ""
    path: str


class RunContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    run_truth_scope: list[str]
    repositories: list[RepositoryPointer]
    ingestion_watermark: datetime | None = None
    refresh_delta: RefreshDelta | None = None
    graph_revision: int = Field(ge=0)
    graph_path: str
    research_md_path: str
    introduction_path: str | None
    glossary_path: str
    coverage_path: str
    facts_dir: str
    state_repository: str
    ontology_extensions: bool = False
    source_errors: list[str]
    # Native provider homes remain separate from immutable project-owned imports.
    source_roots: dict[str, list[str]] = Field(default_factory=dict)
    imported_source_roots: dict[str, list[str]] = Field(default_factory=dict)
    imported_source_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    def prompt_payload(self) -> dict[str, Any]:
        """Return the direct-source representation shown to an ingest agent."""

        return self.model_dump(mode="json")

    def all_source_roots(self) -> dict[str, list[str]]:
        combined = {provider: list(paths) for provider, paths in self.source_roots.items()}
        for provider, paths in self.imported_source_roots.items():
            values = combined.setdefault(provider, [])
            values.extend(path for path in paths if path not in values)
        return combined


class ChatRelation(BaseModel):
    relation: str
    direction: Literal["outgoing", "incoming"]
    other_node_id: str
    other_node_type: str
    other_node_title: str
    explanation: str = ""


class ChatContext(BaseModel):
    """Graph and exact repository context for one conversation turn."""

    project_name: str
    run_truth_scope: list[str]
    repositories: list[RepositoryPointer]
    graph_path: str
    research_md_path: str
    introduction_path: str | None
    glossary_path: str
    coverage_path: str
    facts_dir: str
    state_repository: str
    ontology_extensions: bool
    graph_revision: int
    node: dict[str, Any] | None
    relations: list[ChatRelation]


class ContextAssembler:
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

    def assemble(
        self,
        state: GraphState,
        run_truth_scope: list[str] | None = None,
        repository_access: dict[str, RepositoryAccess] | None = None,
        refresh_delta: RefreshDelta | None = None,
        source_roots: dict[str, list[str]] | None = None,
        imported_source_roots: dict[str, list[str]] | None = None,
        imported_source_fingerprint: str | None = None,
        source_errors: list[str] | None = None,
    ) -> RunContext:
        selected = run_truth_scope or self.manifest.agent.default_run_truth_scope
        selected_set = set(selected)
        project_scope = set(state.project_truth_scope or self.manifest.project.truth_scope)
        if not selected_set or not selected_set.issubset(project_scope):
            raise ValueError("run truth scope must be a non-empty subset of project truth scope")

        repositories = []
        for item in self.manifest.repositories:
            if item.alias not in selected_set:
                continue
            access = (repository_access or {}).get(item.alias)
            repositories.append(
                RepositoryPointer(
                    alias=item.alias,
                    machine=item.machine,
                    host=access.host if access else self.manifest.machine_map[item.machine].host,
                    path=item.path,
                )
            )
        root = self.manifest.research_dir
        introduction = root / "paper" / "introduction.md"
        return RunContext(
            project_name=self.manifest.name,
            run_truth_scope=selected,
            repositories=repositories,
            ingestion_watermark=state.last_refresh_at,
            refresh_delta=refresh_delta,
            graph_revision=state.revision,
            graph_path=str(root / "graph.json"),
            research_md_path=str(root / "research.md"),
            introduction_path=str(introduction) if introduction.exists() else None,
            glossary_path=str(root / "glossary.json"),
            coverage_path=str(root / "coverage.json"),
            facts_dir=str(root / "facts"),
            state_repository=self.manifest.state.repository,
            ontology_extensions=_has_ontology_extensions(state),
            source_errors=source_errors or [],
            source_roots=source_roots or {},
            imported_source_roots=imported_source_roots or {},
            imported_source_fingerprint=imported_source_fingerprint,
        )

    def chat_context(
        self,
        state: GraphState,
        *,
        node_id: str | None = None,
        run_truth_scope: list[str] | None = None,
        repository_access: dict[str, RepositoryAccess] | None = None,
    ) -> ChatContext:
        selected = run_truth_scope or self.manifest.agent.default_run_truth_scope
        selected_set = set(selected)
        project_scope = set(state.project_truth_scope or self.manifest.project.truth_scope)
        if not selected_set or not selected_set.issubset(project_scope):
            raise ValueError("run truth scope must be a non-empty subset of project truth scope")
        if node_id is not None and node_id not in state.nodes:
            raise ValueError(f"unknown node: {node_id}")

        repositories = [
            RepositoryPointer(
                alias=item.alias,
                machine=item.machine,
                host=(
                    (repository_access or {}).get(item.alias).host
                    if (repository_access or {}).get(item.alias)
                    else self.manifest.machine_map[item.machine].host
                ),
                path=item.path,
            )
            for item in self.manifest.repositories
            if item.alias in selected_set
        ]
        root = self.manifest.research_dir
        introduction = root / "paper" / "introduction.md"
        return ChatContext(
            project_name=self.manifest.name,
            run_truth_scope=selected,
            repositories=repositories,
            graph_path=str(root / "graph.json"),
            research_md_path=str(root / "research.md"),
            introduction_path=str(introduction) if introduction.exists() else None,
            glossary_path=str(root / "glossary.json"),
            coverage_path=str(root / "coverage.json"),
            facts_dir=str(root / "facts"),
            state_repository=self.manifest.state.repository,
            ontology_extensions=_has_ontology_extensions(state),
            graph_revision=state.revision,
            node=state.nodes[node_id].model_dump(mode="json") if node_id else None,
            relations=_one_hop_relations(state, node_id) if node_id else [],
        )

    def source_roots(self, execution_machine: str | None) -> dict[str, list[str]]:
        """Name every configured provider source root on the execution machine."""
        machine = self.manifest.machine_map.get(execution_machine or "")
        remote = bool(machine and machine.host)

        def display(values: list[str]) -> list[str]:
            return [str(Path(value).expanduser()) if not remote else value for value in values]

        return {
            profile.id: display(profile.session_roots(self.manifest.sources, remote=remote))
            for profile in sorted(PROVIDERS.values(), key=lambda item: item.id)
        }

    def paper_pointers(
        self,
        introduction_override: Path | None = None,
        repository_access: dict[str, RepositoryAccess] | None = None,
    ) -> dict[str, object]:
        root = self.manifest.research_dir
        introduction = introduction_override or root / "paper" / "introduction.md"
        return {
            "introduction": str(introduction),
            "graph": str(root / "graph.json"),
            "research_md": str(root / "research.md"),
            "truth_repositories": [
                {
                    "alias": item.alias,
                    "machine": item.machine,
                    "host": (
                        repository_access[item.alias].host
                        if repository_access and item.alias in repository_access
                        else self.manifest.machine_map[item.machine].host
                    ),
                    "path": item.path,
                }
                for item in self.manifest.repositories
                if item.alias in self.manifest.project.truth_scope
            ],
        }


def validate_work_patch(patch: Patch) -> None:
    """Work may reflect graph changes, but it never advances ingest state."""

    if patch.processed_cursors:
        raise ValueError(
            "A Work patch must not claim processed_cursors; only seed and refresh read "
            "conversations forward from a cursor."
        )
    if any(isinstance(operation, SetCoverageOperation) for operation in patch.ops):
        raise ValueError(
            "A Work patch must not set coverage; only seed and refresh move the coverage boundary."
        )


def _has_ontology_extensions(state: GraphState) -> bool:
    """Extension authoring rules are dead weight where no extension was ever defined."""

    ontology = state.ontology
    return bool(ontology.types or ontology.fields or ontology.relations)


def _one_hop_relations(state: GraphState, node_id: str) -> list[ChatRelation]:
    relations = []
    for edge in state.edges.values():
        if edge.source == node_id:
            other_id, direction = edge.target, "outgoing"
        elif edge.target == node_id:
            other_id, direction = edge.source, "incoming"
        else:
            continue
        other = state.nodes.get(other_id)
        if other is None:
            continue
        relations.append(
            ChatRelation(
                relation=edge.relation,
                direction=direction,
                other_node_id=other_id,
                other_node_type=other.type,
                other_node_title=other.title,
                explanation=edge.explanation,
            )
        )
    return sorted(relations, key=lambda item: (item.direction, item.relation, item.other_node_id))
