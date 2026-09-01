"""Graph-only semantic rebase and merge for one Auto-research branch.

This module deliberately owns no task allocation, branch history, API route, or
merge-receipt persistence.  It consumes exact branch/main snapshots, runs one
fresh dedicated agent in a stage-only write scope, and uses HistoryManager's
single-transition validation/append boundary through a small structural port.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import aclosing, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from rcp.agents import AgentEvent, AgentLauncher, agent_output_schema, parse_agent_patch_json
from rcp.agents.branch_merge_prompt import (
    branch_merge_correction_contract,
    branch_merge_rebase_contract,
    branch_merge_task_contract,
)
from rcp.agents.schema import OrchestratorAgentPatch, prepare_agent_patch
from rcp.agents.write_scope import ProjectWriteScope
from rcp.core.materialize import apply_valid_patch
from rcp.core.models import (
    Ambiguity,
    AuthorizedHuman,
    BranchMergeProvenance,
    BranchMergeReceipt,
    Edge,
    GlossaryTerm,
    GraphBranchMetadata,
    GraphState,
    Patch,
    ProjectNode,
    Proposal,
)
from rcp.core.operations import SetStandingOperation, UpdateNodesOperation
from rcp.core.transition_models import (
    GraphHeadRef,
    TransitionConflictDetail,
    TransitionTrace,
)
from rcp.core.transitions import (
    TRANSITION_RULESET_TAG,
    CommittedTransition,
    PreparedTransition,
    project_transition_projection,
    transition_trigger_manifest,
)
from rcp.history import (
    BranchMergeAlreadyCommitted,
    BranchMergeAlreadyResolved,
    PatchRejected,
    RevisionConflict,
)
from rcp.limits import PATCH_CORRECTION_MAX_ROUNDS
from rcp.runs.shared import (
    AgentOutputProblem,
    _collect_patch_text,
    _existing_patch_digest,
    _ProviderOutcome,
    _record_agent_launch_receipt,
    _sse,
    _stage_json_task_input,
    _stage_task_contract,
    _stream_agent_events,
    _task_token,
)
from rcp.service import RunRequest
from rcp.transport import RemoteRunStage, StateUnavailable

MAX_BRANCH_MERGE_REBASE_ROUNDS = 3
_SEMANTIC_NODE_BOOKKEEPING = frozenset({"created_rev", "updated_rev"})
_SEMANTIC_EDGE_BOOKKEEPING = frozenset({"created_rev"})
_SEMANTIC_PROPOSAL_BOOKKEEPING = frozenset(
    {
        "base_rev",
        "raised_rev",
        "resolved_rev",
        "created_by_operation_id",
        "resolved_by_operation_id",
    }
)
_SEMANTIC_AMBIGUITY_BOOKKEEPING = frozenset({"raised_rev"})
_SEMANTIC_GLOSSARY_BOOKKEEPING = frozenset({"updated_rev"})


class _StrictMergeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BranchPatchSummary(_StrictMergeModel):
    revision: int = Field(ge=0)
    transition_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    summary: str
    change_summary: list[str] = Field(default_factory=list)
    task_id: str | None = None
    profile: Literal["ordinary", "orchestrator"] | None = None


class BranchMergeEligibility(_StrictMergeModel):
    """Caller-supplied durable proof that no branch writer may race the merge."""

    branch_head: GraphHeadRef
    episode_ending: Literal["completed", "exhausted", "stopped", "failed", "human_pause"]
    active_branch_writer_task_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_quiescence(self) -> BranchMergeEligibility:
        if self.branch_head.target.kind != "branch":
            raise ValueError("merge eligibility must name one exact branch head")
        if self.active_branch_writer_task_ids:
            raise ValueError("a graph branch with an active writer is not merge eligible")
        return self


class NodeSemanticDelta(_StrictMergeModel):
    change: Literal["created", "updated", "removed"]
    node_id: str
    before: ProjectNode | None = None
    after: ProjectNode | None = None

    @model_validator(mode="after")
    def require_shape(self) -> NodeSemanticDelta:
        _require_change_shape(self.change, self.before, self.after, self.node_id)
        return self


class EdgeSemanticDelta(_StrictMergeModel):
    change: Literal["created", "updated", "removed"]
    edge_id: str
    before: Edge | None = None
    after: Edge | None = None

    @model_validator(mode="after")
    def require_shape(self) -> EdgeSemanticDelta:
        _require_change_shape(self.change, self.before, self.after, self.edge_id)
        return self


class ProposalSemanticDelta(_StrictMergeModel):
    change: Literal["created", "updated", "removed"]
    proposal_id: str
    before: Proposal | None = None
    after: Proposal | None = None

    @model_validator(mode="after")
    def require_shape(self) -> ProposalSemanticDelta:
        _require_change_shape(self.change, self.before, self.after, self.proposal_id)
        return self


class AmbiguitySemanticDelta(_StrictMergeModel):
    change: Literal["created", "updated", "removed"]
    ambiguity_id: str
    before: Ambiguity | None = None
    after: Ambiguity | None = None

    @model_validator(mode="after")
    def require_shape(self) -> AmbiguitySemanticDelta:
        _require_change_shape(self.change, self.before, self.after, self.ambiguity_id)
        return self


class GlossarySemanticDelta(_StrictMergeModel):
    change: Literal["created", "updated", "removed"]
    term: str
    before: GlossaryTerm | None = None
    after: GlossaryTerm | None = None

    @model_validator(mode="after")
    def require_shape(self) -> GlossarySemanticDelta:
        _require_change_shape(self.change, self.before, self.after, self.term, id_field="term")
        return self


class GlobalSemanticDelta(_StrictMergeModel):
    field: Literal["project_truth_scope", "ontology", "coverage"]
    before: JsonValue
    after: JsonValue

    @model_validator(mode="after")
    def require_actual_change(self) -> GlobalSemanticDelta:
        if self.before == self.after:
            raise ValueError("a global semantic delta must change its field")
        return self


class GraphSemanticDelta(_StrictMergeModel):
    base_head: GraphHeadRef
    branch_head: GraphHeadRef
    nodes: list[NodeSemanticDelta] = Field(default_factory=list)
    edges: list[EdgeSemanticDelta] = Field(default_factory=list)
    proposals: list[ProposalSemanticDelta] = Field(default_factory=list)
    ambiguities: list[AmbiguitySemanticDelta] = Field(default_factory=list)
    glossary: list[GlossarySemanticDelta] = Field(default_factory=list)
    globals: list[GlobalSemanticDelta] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.nodes, self.edges, self.proposals, self.ambiguities, self.glossary, self.globals)
        )


class BranchMergeConflict(_StrictMergeModel):
    code: Literal[
        "both_changed",
        "branch_created_main_created",
        "branch_removed_main_changed",
        "branch_changed_main_removed",
    ]
    collection: Literal[
        "nodes",
        "edges",
        "proposals",
        "ambiguities",
        "glossary",
        "project_truth_scope",
        "ontology",
        "coverage",
    ]
    entity_id: str | None = None
    field_path: str
    base: JsonValue
    branch: JsonValue
    main: JsonValue
    message: str


class BranchMergeTransitionContract(_StrictMergeModel):
    ruleset_tag: str
    trigger_manifest: dict[str, JsonValue]
    transition_trace_schema: dict[str, JsonValue]
    orchestrator_patch_schema: dict[str, JsonValue]

    @classmethod
    def current(cls) -> BranchMergeTransitionContract:
        return cls(
            ruleset_tag=TRANSITION_RULESET_TAG,
            trigger_manifest=transition_trigger_manifest().model_dump(mode="json"),
            transition_trace_schema=TransitionTrace.model_json_schema(),
            orchestrator_patch_schema=agent_output_schema(profile="orchestrator"),
        )


class BranchMergeContext(_StrictMergeModel):
    """One closed, content-addressed base/branch/main merge input."""

    schema_generation: Literal[1] = 1
    context_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    merge_task_id: str = Field(min_length=1)
    authorized_by: AuthorizedHuman
    metadata: GraphBranchMetadata
    eligibility: BranchMergeEligibility
    base_graph: GraphState
    branch_graph: GraphState
    main_head: GraphHeadRef
    main_graph: GraphState
    run_truth_scope: list[str] = Field(min_length=1)
    semantic_delta: GraphSemanticDelta
    branch_patch_summaries: list[BranchPatchSummary] = Field(default_factory=list)
    deterministic_conflicts: list[BranchMergeConflict] = Field(default_factory=list)
    transition_contract: BranchMergeTransitionContract

    @model_validator(mode="after")
    def require_exact_snapshots(self) -> BranchMergeContext:
        if self.run_truth_scope != sorted(set(self.run_truth_scope)):
            raise ValueError("branch merge run truth scope must be sorted and unique")
        if self.metadata.base_head.revision != self.base_graph.revision:
            raise ValueError("branch merge base graph does not match the exact base head")
        if self.metadata.head.revision != self.branch_graph.revision:
            raise ValueError("branch merge graph does not match the exact branch head")
        if (
            self.main_head.target.kind != "main"
            or self.main_head.revision != self.main_graph.revision
        ):
            raise ValueError("branch merge main graph does not match one exact main head")
        if self.eligibility.branch_head != self.metadata.head:
            raise ValueError("merge eligibility was proved for a different branch head")
        if self.semantic_delta.base_head != self.metadata.base_head:
            raise ValueError("semantic delta names a different immutable branch base")
        if self.semantic_delta.branch_head != self.metadata.head:
            raise ValueError("semantic delta names a different branch head")
        expected_delta = build_semantic_delta(
            self.base_graph,
            self.branch_graph,
            base_head=self.metadata.base_head,
            branch_head=self.metadata.head,
        )
        if self.semantic_delta != expected_delta:
            raise ValueError("branch merge semantic delta does not match its graph snapshots")
        expected_conflicts = detect_branch_merge_conflicts(
            self.base_graph,
            self.branch_graph,
            self.main_graph,
        )
        if self.deterministic_conflicts != expected_conflicts:
            raise ValueError("branch merge conflicts do not match its graph snapshots")
        revisions = [item.revision for item in self.branch_patch_summaries]
        if revisions != sorted(set(revisions)):
            raise ValueError("branch Patch summaries must be ordered by unique revision")
        if any(
            revision <= self.metadata.base_head.revision or revision > self.metadata.head.revision
            for revision in revisions
        ):
            raise ValueError("branch Patch summary lies outside the exact branch lineage")
        if self.transition_contract != BranchMergeTransitionContract.current():
            raise ValueError("branch merge context uses a stale transition-manager contract")
        expected_id = _context_id(self.model_dump(mode="json", exclude={"context_id"}))
        if self.context_id != expected_id:
            raise ValueError("branch merge context id does not match its canonical content")
        return self

    @classmethod
    def create(
        cls,
        *,
        merge_task_id: str,
        authorized_by: AuthorizedHuman,
        metadata: GraphBranchMetadata,
        eligibility: BranchMergeEligibility,
        base_graph: GraphState,
        branch_graph: GraphState,
        main_head: GraphHeadRef,
        main_graph: GraphState,
        run_truth_scope: list[str],
        branch_patch_summaries: list[BranchPatchSummary] | None = None,
    ) -> BranchMergeContext:
        if run_truth_scope != sorted(set(run_truth_scope)) or not run_truth_scope:
            raise ValueError("branch merge run truth scope must be non-empty, sorted, and unique")
        delta = build_semantic_delta(
            base_graph,
            branch_graph,
            base_head=metadata.base_head,
            branch_head=metadata.head,
        )
        payload: dict[str, object] = {
            "schema_generation": 1,
            "merge_task_id": merge_task_id,
            "authorized_by": authorized_by,
            "metadata": metadata,
            "eligibility": eligibility,
            "base_graph": base_graph,
            "branch_graph": branch_graph,
            "main_head": main_head,
            "main_graph": main_graph,
            "run_truth_scope": list(run_truth_scope),
            "semantic_delta": delta,
            "branch_patch_summaries": list(branch_patch_summaries or ()),
            "deterministic_conflicts": detect_branch_merge_conflicts(
                base_graph,
                branch_graph,
                main_graph,
            ),
            "transition_contract": BranchMergeTransitionContract.current(),
        }
        encoded = _jsonable(payload)
        return cls.model_validate({**payload, "context_id": _context_id(encoded)})


class BranchMergeCandidateProblem(ValueError):
    def __init__(self, message: str, *, conflicts: list[TransitionConflictDetail] | None = None):
        self.message = " ".join(message.split())
        self.conflicts = list(conflicts or ())
        super().__init__(self.message)


class BranchMergeSemanticConflict(BranchMergeCandidateProblem):
    """A correctable candidate that cannot be prepared against current main."""


class BranchMergeSourceChanged(ValueError):
    """The supposedly quiescent source branch changed during its merge task."""


class _BranchMergeHistory(Protocol):
    def validate_candidate(
        self, patch: Patch, **kwargs: object
    ) -> tuple[Patch, Any, GraphState]: ...

    def append(self, patch: Patch, **kwargs: object) -> tuple[Patch, Any]: ...


@dataclass
class BranchMergeRunOutcome:
    status: Literal[
        "pending",
        "noop",
        "committed",
        "rejected",
        "paused",
        "retryable",
    ] = "pending"
    merge_id: str | None = None
    native_session_id: str | None = None
    source_branch_head: GraphHeadRef | None = None
    rebased_main_head: GraphHeadRef | None = None
    result_main_head: GraphHeadRef | None = None
    prepared: PreparedTransition | None = None
    committed: CommittedTransition | None = None
    receipt: BranchMergeReceipt | None = None
    correction_rounds: int = 0
    rebase_rounds: int = 0
    diagnostic: str | None = None


@dataclass(frozen=True)
class BranchMergeStage:
    local_stage: Path | None
    remote_stage: RemoteRunStage | None
    workspace: Path

    def __post_init__(self) -> None:
        if (self.local_stage is None) == (self.remote_stage is None):
            raise ValueError("exactly one local or remote branch merge stage is required")
        if self.remote_stage is not None:
            if self.remote_stage.root is None:
                raise ValueError("remote branch merge stage is not open")
            if str(self.workspace) != str(self.remote_stage.workspace):
                raise ValueError("branch merge workspace does not match its remote stage")
            return
        assert self.local_stage is not None
        if self.workspace.resolve() != (self.local_stage / "workspace").resolve():
            raise ValueError("branch merge workspace must be the exact local stage workspace")


def build_semantic_delta(
    base: GraphState,
    branch: GraphState,
    *,
    base_head: GraphHeadRef,
    branch_head: GraphHeadRef,
) -> GraphSemanticDelta:
    """Build the typed net semantic change, excluding RCP revision bookkeeping."""

    if base.revision != base_head.revision or branch.revision != branch_head.revision:
        raise ValueError("semantic delta snapshots do not match their exact heads")
    if base_head.target.kind != "main" or branch_head.target.kind != "branch":
        raise ValueError("semantic delta requires a main base and branch head")
    globals_: list[GlobalSemanticDelta] = []
    if base.project_truth_scope != branch.project_truth_scope:
        globals_.append(
            GlobalSemanticDelta(
                field="project_truth_scope",
                before=base.project_truth_scope,
                after=branch.project_truth_scope,
            )
        )
    if base.ontology != branch.ontology:
        globals_.append(
            GlobalSemanticDelta(
                field="ontology",
                before=base.ontology.model_dump(mode="json"),
                after=branch.ontology.model_dump(mode="json"),
            )
        )
    if base.coverage != branch.coverage:
        globals_.append(
            GlobalSemanticDelta(
                field="coverage",
                before=base.coverage.model_dump(mode="json"),
                after=branch.coverage.model_dump(mode="json"),
            )
        )
    return GraphSemanticDelta(
        base_head=base_head,
        branch_head=branch_head,
        nodes=_typed_collection_delta(
            base.nodes,
            branch.nodes,
            model=NodeSemanticDelta,
            id_field="node_id",
            bookkeeping=_SEMANTIC_NODE_BOOKKEEPING,
        ),
        edges=_typed_collection_delta(
            base.edges,
            branch.edges,
            model=EdgeSemanticDelta,
            id_field="edge_id",
            bookkeeping=_SEMANTIC_EDGE_BOOKKEEPING,
        ),
        proposals=_typed_collection_delta(
            base.proposals,
            branch.proposals,
            model=ProposalSemanticDelta,
            id_field="proposal_id",
            bookkeeping=_SEMANTIC_PROPOSAL_BOOKKEEPING,
        ),
        ambiguities=_typed_collection_delta(
            base.ambiguities,
            branch.ambiguities,
            model=AmbiguitySemanticDelta,
            id_field="ambiguity_id",
            bookkeeping=_SEMANTIC_AMBIGUITY_BOOKKEEPING,
        ),
        glossary=_typed_collection_delta(
            base.glossary,
            branch.glossary,
            model=GlossarySemanticDelta,
            id_field="term",
            bookkeeping=_SEMANTIC_GLOSSARY_BOOKKEEPING,
        ),
        globals=globals_,
    )


def detect_branch_merge_conflicts(
    base: GraphState,
    branch: GraphState,
    main: GraphState,
) -> list[BranchMergeConflict]:
    """Return deterministic field conflicts without choosing either side."""

    conflicts: list[BranchMergeConflict] = []
    for collection, base_values, branch_values, main_values, bookkeeping in (
        ("nodes", base.nodes, branch.nodes, main.nodes, _SEMANTIC_NODE_BOOKKEEPING),
        ("edges", base.edges, branch.edges, main.edges, _SEMANTIC_EDGE_BOOKKEEPING),
        (
            "proposals",
            base.proposals,
            branch.proposals,
            main.proposals,
            _SEMANTIC_PROPOSAL_BOOKKEEPING,
        ),
        (
            "ambiguities",
            base.ambiguities,
            branch.ambiguities,
            main.ambiguities,
            _SEMANTIC_AMBIGUITY_BOOKKEEPING,
        ),
        (
            "glossary",
            base.glossary,
            branch.glossary,
            main.glossary,
            _SEMANTIC_GLOSSARY_BOOKKEEPING,
        ),
    ):
        conflicts.extend(
            _collection_conflicts(
                collection,
                base_values,
                branch_values,
                main_values,
                bookkeeping,
            )
        )
    for field_name, base_value, branch_value, main_value in (
        (
            "project_truth_scope",
            base.project_truth_scope,
            branch.project_truth_scope,
            main.project_truth_scope,
        ),
        ("ontology", base.ontology, branch.ontology, main.ontology),
        ("coverage", base.coverage, branch.coverage, main.coverage),
    ):
        base_json = _semantic_document(base_value, frozenset())
        branch_json = _semantic_document(branch_value, frozenset())
        main_json = _semantic_document(main_value, frozenset())
        for path, base_field, branch_field, main_field in _conflicting_fields(
            base_json,
            branch_json,
            main_json,
            prefix=field_name,
        ):
            conflicts.append(
                BranchMergeConflict(
                    code="both_changed",
                    collection=field_name,
                    field_path=path,
                    base=base_field,
                    branch=branch_field,
                    main=main_field,
                    message=(f"Branch and main changed {path} differently from the base."),
                )
            )
    return conflicts


def semantic_delta_is_subsumed(delta: GraphSemanticDelta, main: GraphState) -> bool:
    """Return whether main already contains every semantic change made by the branch.

    Main may also contain compatible changes to fields the branch did not touch. Revision and
    operation-id bookkeeping is excluded by the same rules used to build the semantic delta.
    """

    collection_checks = (
        (delta.nodes, main.nodes, "node_id", _SEMANTIC_NODE_BOOKKEEPING),
        (delta.edges, main.edges, "edge_id", _SEMANTIC_EDGE_BOOKKEEPING),
        (
            delta.proposals,
            main.proposals,
            "proposal_id",
            _SEMANTIC_PROPOSAL_BOOKKEEPING,
        ),
        (
            delta.ambiguities,
            main.ambiguities,
            "ambiguity_id",
            _SEMANTIC_AMBIGUITY_BOOKKEEPING,
        ),
        (delta.glossary, main.glossary, "term", _SEMANTIC_GLOSSARY_BOOKKEEPING),
    )
    for changes, values, identity_field, bookkeeping in collection_checks:
        for change in changes:
            identity = getattr(change, identity_field)
            if not _semantic_change_is_subsumed(
                change.change,
                change.before,
                change.after,
                values.get(identity),
                bookkeeping,
            ):
                return False

    main_globals: dict[str, object] = {
        "project_truth_scope": main.project_truth_scope,
        "ontology": main.ontology,
        "coverage": main.coverage,
    }
    return all(
        _branch_changes_are_present(
            change.before,
            change.after,
            _semantic_document(main_globals[change.field], frozenset()),
        )
        for change in delta.globals
    )


SemanticWritePath = tuple[str, ...]


def validate_branch_merge_candidate_conformance(
    context: BranchMergeContext,
    current_main: GraphState,
    prepared: Patch,
    graph: GraphState,
) -> None:
    """Prove initiating actions stay inside and carry the branch semantic delta."""

    if prepared.transition is None:
        raise BranchMergeSemanticConflict("Prepared branch merge has no transition trace.")
    allowed = _graph_semantic_write_paths(context.base_graph, context.branch_graph)
    conflicts = _branch_conflict_paths(context.deterministic_conflicts)
    mandatory = {
        path for path in allowed if not any(_semantic_path_covers(item, path) for item in conflicts)
    }
    timeline = current_main
    initiating_indexes = sorted(
        index
        for group in prepared.transition.initiating_groups
        for index in group.operation_indexes
    )
    for operation_index in initiating_indexes:
        operation = prepared.ops[operation_index]
        undeclared = sorted(
            path
            for path in _declared_bookkeeping_sensitive_write_paths(operation)
            if not any(_semantic_path_authorizes_declared_write(item, path) for item in allowed)
        )
        if undeclared:
            rendered = ", ".join(_render_semantic_path(path) for path in undeclared[:8])
            raise BranchMergeSemanticConflict(
                "Branch merge candidate declares writes outside the source branch delta: "
                + rendered
            )
        one_action = prepared.model_copy(
            update={
                "ops": [operation],
                "transition": None,
            }
        )
        try:
            next_state = apply_valid_patch(timeline, one_action)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise BranchMergeSemanticConflict(
                f"Prepared branch merge action {operation_index} could not be replayed: {exc}"
            ) from exc
        unexpected = sorted(
            path
            for path in _graph_semantic_write_paths(timeline, next_state)
            if not any(_semantic_path_covers(item, path) for item in allowed)
        )
        if unexpected:
            rendered = ", ".join(_render_semantic_path(path) for path in unexpected[:8])
            raise BranchMergeSemanticConflict(
                "Branch merge candidate writes outside the source branch delta: " + rendered
            )
        timeline = next_state

    branch_document = _graph_semantic_document(context.branch_graph)
    result_document = _graph_semantic_document(graph)
    missing = sorted(
        path
        for path in mandatory
        if _semantic_path_value(branch_document, path)
        != _semantic_path_value(result_document, path)
    )
    if missing:
        rendered = ", ".join(_render_semantic_path(path) for path in missing[:8])
        raise BranchMergeSemanticConflict(
            "Branch merge candidate omits non-conflicting source changes: " + rendered
        )


def branch_merge_can_resolve_without_patch(context: BranchMergeContext) -> bool:
    """Whether choosing current main on every conflict still carries all required changes."""

    allowed = _graph_semantic_write_paths(context.base_graph, context.branch_graph)
    conflicts = _branch_conflict_paths(context.deterministic_conflicts)
    mandatory = {
        path for path in allowed if not any(_semantic_path_covers(item, path) for item in conflicts)
    }
    branch_document = _graph_semantic_document(context.branch_graph)
    main_document = _graph_semantic_document(context.main_graph)
    return all(
        _semantic_path_value(branch_document, path) == _semantic_path_value(main_document, path)
        for path in mandatory
    )


def branch_merge_id(metadata: GraphBranchMetadata) -> str:
    """Hash only immutable branch lineage, so a moving main keeps one merge identity."""

    return _canonical_sha256(
        {
            "schema_generation": 1,
            "kind": "auto_research_graph_branch_merge",
            "branch_id": metadata.branch_id,
            "episode_id": metadata.episode_id,
            "project_id": metadata.project_id,
            "branch_kind": metadata.kind,
            "branch_base_head": metadata.base_head.model_dump(mode="json"),
            "branch_head": metadata.head.model_dump(mode="json"),
        }
    )


def branch_merge_provenance(context: BranchMergeContext) -> BranchMergeProvenance:
    return BranchMergeProvenance(
        merge_id=branch_merge_id(context.metadata),
        branch_id=context.metadata.branch_id,
        episode_id=context.metadata.episode_id,
        branch_base_head=context.metadata.base_head,
        branch_head=context.metadata.head,
        rebased_main_head=context.main_head,
        merge_task_id=context.merge_task_id,
    )


def parse_branch_merge_candidate(value: str, context: BranchMergeContext) -> Patch:
    """Parse semantic-only output and stamp all non-agent merge provenance."""

    try:
        draft = parse_agent_patch_json(value, profile="orchestrator")
    except ValueError as exc:
        raise BranchMergeCandidateProblem(str(exc)) from exc
    if not isinstance(draft, OrchestratorAgentPatch):  # pragma: no cover - profile is fixed above
        raise BranchMergeCandidateProblem("branch merge output did not use orchestrator schema")
    if draft.repositories_read:
        raise BranchMergeCandidateProblem(
            "A graph-only branch merge must declare repositories_read as an empty list."
        )
    patch = prepare_agent_patch(
        draft,
        kind="work",
        run_truth_scope=context.run_truth_scope,
        source_operation_id=context.merge_task_id,
        profile="orchestrator",
    )
    return patch.model_copy(
        update={
            "revision": context.main_head.revision + 1,
            "authorized_by": context.authorized_by,
            "profile": "orchestrator",
            "task_id": context.merge_task_id,
            "episode_id": context.metadata.episode_id,
            "branch_merge": branch_merge_provenance(context),
        }
    )


def prepare_branch_merge_with_history(
    history: _BranchMergeHistory,
    candidate: Patch,
    *,
    expected_main_head: GraphHeadRef,
    context: BranchMergeContext,
) -> PreparedTransition:
    """Run HistoryManager's authoritative validation/transition preparation without writing."""

    if expected_main_head.target.kind != "main":
        raise ValueError("branch merge candidates can prepare only against main")
    try:
        prepared, report, current = history.validate_candidate(candidate)
    except PatchRejected as exc:
        raise BranchMergeSemanticConflict(str(exc)) from exc
    if current.revision != expected_main_head.revision:
        raise RevisionConflict(
            f"main moved from revision {expected_main_head.revision} to {current.revision}"
        )
    if getattr(report, "rejected", False):
        diagnostic, conflicts = _validation_diagnostic(report)
        raise BranchMergeSemanticConflict(diagnostic, conflicts=conflicts)
    _require_prepared_provenance(prepared, candidate, expected_main_head)
    assert prepared.transition is not None
    try:
        graph = apply_valid_patch(current, prepared)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise BranchMergeSemanticConflict(
            f"Prepared branch merge operations could not be applied atomically: {exc}"
        ) from exc
    validate_branch_merge_candidate_conformance(context, current, prepared, graph)
    return PreparedTransition(
        patch=prepared,
        projection=project_transition_projection(
            graph,
            prepared.transition,
            canonical=False,
        ),
    )


def commit_branch_merge_with_history(
    history: _BranchMergeHistory,
    candidate: Patch,
    *,
    expected_main_head: GraphHeadRef,
) -> CommittedTransition:
    """Append one prepared semantic merge at an exact main revision or append nothing."""

    if expected_main_head.target.kind != "main":
        raise ValueError("branch merge candidates can commit only to main")
    try:
        committed, result = history.append(
            candidate,
            discard_on_reject=True,
            expected_revision=expected_main_head.revision,
        )
    except PatchRejected as exc:
        raise BranchMergeSemanticConflict(str(exc)) from exc
    _require_prepared_provenance(committed, candidate, expected_main_head)
    assert committed.transition is not None
    state = result.state
    if state.revision != expected_main_head.revision + 1:
        raise RuntimeError("branch merge append did not produce exactly one main revision")
    return CommittedTransition(
        patch=committed,
        projection=project_transition_projection(
            state,
            committed.transition,
            canonical=True,
        ),
    )


def branch_merge_receipt_from_committed_patch(patch: Patch) -> BranchMergeReceipt:
    """Project one accepted idempotency winner into its durable branch receipt."""

    if (
        patch.admission != "accepted"
        or patch.branch_merge is None
        or patch.transition is None
        or patch.authorized_by is None
    ):
        raise ValueError("committed main branch merge lacks exact receipt provenance")
    return BranchMergeReceipt(
        outcome="committed",
        provenance=patch.branch_merge,
        result_main_head=GraphHeadRef(
            revision=patch.revision,
            transition_id=patch.transition.transition_id,
        ),
        authorized_by=patch.authorized_by,
        created_at=patch.created_at,
    )


def require_graph_only_merge_scope(
    scope: ProjectWriteScope,
    *,
    context: BranchMergeContext,
    stage: BranchMergeStage,
) -> None:
    """Fail closed unless the provider can write only its exact scratch workspace."""

    if scope.capability != "orchestrate":
        raise ValueError("branch merge requires orchestrate capability")
    if scope.project_id != context.metadata.project_id:
        raise ValueError("branch merge write scope belongs to a different project")
    if scope.repositories or scope.repository_roots:
        raise ValueError("branch merge agents receive no repository write roots")
    if scope.workspace_root != str(stage.workspace):
        raise ValueError("branch merge write scope does not name its exact workspace")
    if scope.writable_roots != [str(stage.workspace)]:
        raise ValueError("branch merge writable roots must contain only its scratch workspace")
    if stage.remote_stage is not None:
        assert stage.remote_stage.root is not None
        if scope.stage_root != str(stage.remote_stage.root):
            raise ValueError("branch merge scope does not match its remote stage")
        if scope.execution_host != stage.remote_stage.host:
            raise ValueError("branch merge scope does not match its execution host")
    else:
        assert stage.local_stage is not None
        if Path(scope.stage_root).resolve() != stage.local_stage.resolve():
            raise ValueError("branch merge scope does not match its local stage")
    if not any(PurePosixPath(path).name == ".research" for path in scope.protected_write_paths):
        raise ValueError("branch merge scope must explicitly protect canonical .research")


def classify_refreshed_context(
    previous: BranchMergeContext,
    current: BranchMergeContext,
) -> Literal["unchanged", "main_moved"]:
    """Permit only current-main movement; never retarget a task to another branch head."""

    stable = (
        previous.merge_task_id == current.merge_task_id
        and previous.authorized_by == current.authorized_by
        and previous.metadata == current.metadata
        and previous.eligibility == current.eligibility
        and previous.base_graph == current.base_graph
        and previous.branch_graph == current.branch_graph
        and previous.run_truth_scope == current.run_truth_scope
        and previous.branch_patch_summaries == current.branch_patch_summaries
        and previous.transition_contract == current.transition_contract
    )
    if not stable:
        raise BranchMergeSourceChanged(
            "The source branch or its immutable base changed while the merge task was running."
        )
    if previous.main_head == current.main_head:
        if previous.context_id != current.context_id:
            raise BranchMergeSourceChanged(
                "Merge context changed without a corresponding main or branch head change."
            )
        return "unchanged"
    return "main_moved"


async def stream_branch_merge_run(
    request: RunRequest,
    launcher: AgentLauncher,
    *,
    load_context: Callable[[], BranchMergeContext],
    main_history: _BranchMergeHistory,
    stage: BranchMergeStage,
    write_scope: ProjectWriteScope,
    validator_command: str,
    outcome: BranchMergeRunOutcome,
    execution: Any | None = None,
    binary: str | None = None,
    max_main_rebases: int = MAX_BRANCH_MERGE_REBASE_ROUNDS,
) -> AsyncIterator[str]:
    """Run, correct, rebase, and atomically commit one graph-only branch merge.

    ``load_context`` must resolve the immutable base/branch snapshots and the
    live main head together. ``main_history`` must be the main-target history
    whose ``validate_candidate`` and ``append`` methods use the canonical
    transition manager and append lock.
    """

    if request.provider is None or request.run_on is None:
        raise ValueError("branch merge request must have a pinned provider and execution machine")
    if not validator_command.strip():
        raise ValueError("branch merge requires a staged live validator command")
    if max_main_rebases < 1:
        raise ValueError("branch merge main rebase bound must be positive")

    context = load_context()
    require_graph_only_merge_scope(write_scope, context=context, stage=stage)
    outcome.merge_id = branch_merge_id(context.metadata)
    outcome.source_branch_head = context.metadata.head
    outcome.rebased_main_head = context.main_head

    if context.semantic_delta.is_empty or semantic_delta_is_subsumed(
        context.semantic_delta,
        context.main_graph,
    ):
        provenance = branch_merge_provenance(context)
        outcome.receipt = BranchMergeReceipt(
            outcome="no_change",
            provenance=provenance,
            result_main_head=context.main_head,
            authorized_by=context.authorized_by,
        )
        outcome.status = "noop"
        yield _sse(
            AgentEvent(
                event="answer",
                text=(
                    "The branch head has no net semantic graph change to merge."
                    if context.semantic_delta.is_empty
                    else "Main already contains this branch head's semantic graph change."
                ),
            )
        )
        yield _sse(AgentEvent(event="done"))
        return

    token = _task_token(execution)
    original_contract_path: str | None = None
    session_id: str | None = None
    candidate_text: str | None = None
    candidate: Patch | None = None
    candidate_problem: BranchMergeCandidateProblem | None = None
    first_turn = True

    while True:
        if first_turn:
            _clear_patch_candidates(stage)
            context_path = _stage_merge_context(stage, token, context, round_number=0)
            patch_path = _patch_path(stage)
            contract = branch_merge_task_contract(
                context_path=context_path,
                context_id=context.context_id,
                patch_path=patch_path,
                validator_command=validator_command,
            )
            original_contract_path, prompt = _stage_task_contract(
                stage.local_stage,
                stage.remote_stage,
                f"task-{token}-branch-merge.md",
                contract,
                execution=execution,
                role="branch_merge",
            )
            _record_merge_launch(
                execution,
                request,
                prompt=prompt,
                contract_path=original_contract_path,
                write_scope=write_scope,
                context=context,
                continuation="initial",
                round_number=0,
            )
            provider, events = _provider_turn(
                launcher,
                request,
                prompt,
                stage=stage,
                write_scope=write_scope,
                execution=execution,
                binary=binary,
                session_id=None,
                required_session_id=None,
            )
            async with aclosing(events) as frames:
                async for frame in frames:
                    yield frame
            session_id = provider.session_id
            outcome.native_session_id = session_id
            if provider.paused:
                outcome.status = "paused"
                return
            if provider.failed or not provider.completed:
                outcome.status = "rejected"
                outcome.diagnostic = "The branch merge provider did not complete its first turn."
                return
            try:
                candidate_text = _read_candidate_text(stage)
                candidate = parse_branch_merge_candidate(candidate_text, context)
                candidate_problem = None
            except (
                AgentOutputProblem,
                BranchMergeCandidateProblem,
                OSError,
                StateUnavailable,
            ) as exc:
                candidate_problem = BranchMergeCandidateProblem(str(exc))
                candidate = None
            first_turn = False

        fresh = load_context()
        try:
            refresh = classify_refreshed_context(context, fresh)
        except BranchMergeSourceChanged as exc:
            outcome.status = "rejected"
            outcome.diagnostic = str(exc)
            yield _sse(AgentEvent(event="error", text=str(exc)))
            return
        if refresh == "main_moved":
            if semantic_delta_is_subsumed(fresh.semantic_delta, fresh.main_graph):
                context = fresh
                outcome.rebased_main_head = context.main_head
                outcome.receipt = BranchMergeReceipt(
                    outcome="no_change",
                    provenance=branch_merge_provenance(context),
                    result_main_head=context.main_head,
                    authorized_by=context.authorized_by,
                )
                outcome.status = "noop"
                yield _sse(
                    AgentEvent(
                        event="answer",
                        text="Main already contains this branch head's semantic graph change.",
                    )
                )
                yield _sse(AgentEvent(event="done"))
                return
            if (
                candidate is not None
                and not candidate.ops
                and branch_merge_can_resolve_without_patch(fresh)
            ):
                context = fresh
                outcome.rebased_main_head = context.main_head
                outcome.receipt = BranchMergeReceipt(
                    outcome="no_change",
                    provenance=branch_merge_provenance(context),
                    result_main_head=context.main_head,
                    authorized_by=context.authorized_by,
                )
                outcome.status = "noop"
                yield _sse(
                    AgentEvent(
                        event="answer",
                        text="The merge resolved conflicting branch changes to current main.",
                    )
                )
                yield _sse(AgentEvent(event="done"))
                return
            if session_id is None or outcome.rebase_rounds >= max_main_rebases:
                outcome.status = "retryable"
                outcome.diagnostic = (
                    "Main advanced while the merge was running; retry the merge task against "
                    "the new main head."
                )
                yield _sse(AgentEvent(event="error", text=outcome.diagnostic))
                return
            previous_context = context
            context = fresh
            outcome.rebase_rounds += 1
            outcome.rebased_main_head = context.main_head
            context_path = _stage_merge_context(
                stage,
                token,
                context,
                round_number=outcome.rebase_rounds,
            )
            _clear_patch_candidates(stage)
            assert original_contract_path is not None
            contract = branch_merge_rebase_contract(
                original_contract_path=original_contract_path,
                previous_context_id=previous_context.context_id,
                context_path=context_path,
                context_id=context.context_id,
                patch_path=_patch_path(stage),
                validator_command=validator_command,
            )
            contract_path, prompt = _stage_task_contract(
                stage.local_stage,
                stage.remote_stage,
                f"task-{token}-branch-merge-rebase-{outcome.rebase_rounds}.md",
                contract,
                execution=execution,
                role=f"branch_merge_rebase_{outcome.rebase_rounds}",
            )
            _record_merge_launch(
                execution,
                request,
                prompt=prompt,
                contract_path=contract_path,
                write_scope=write_scope,
                context=context,
                continuation="main_rebase",
                round_number=outcome.rebase_rounds,
            )
            provider, events = _provider_turn(
                launcher,
                request,
                prompt,
                stage=stage,
                write_scope=write_scope,
                execution=execution,
                binary=binary,
                session_id=session_id,
                required_session_id=session_id,
            )
            async with aclosing(events) as frames:
                async for frame in frames:
                    yield frame
            if provider.session_id:
                session_id = provider.session_id
                outcome.native_session_id = session_id
            if provider.paused:
                outcome.status = "paused"
                return
            if provider.failed or not provider.completed:
                outcome.status = "retryable"
                outcome.diagnostic = "The branch merge rebase turn did not complete."
                return
            try:
                candidate_text = _read_candidate_text(stage)
                candidate = parse_branch_merge_candidate(candidate_text, context)
                candidate_problem = None
            except (
                AgentOutputProblem,
                BranchMergeCandidateProblem,
                OSError,
                StateUnavailable,
            ) as exc:
                candidate_problem = BranchMergeCandidateProblem(str(exc))
                candidate = None
            continue

        context = fresh
        if candidate is not None and not candidate.ops:
            if branch_merge_can_resolve_without_patch(context):
                outcome.receipt = BranchMergeReceipt(
                    outcome="no_change",
                    provenance=branch_merge_provenance(context),
                    result_main_head=context.main_head,
                    authorized_by=context.authorized_by,
                )
                outcome.status = "noop"
                yield _sse(
                    AgentEvent(
                        event="answer",
                        text="The merge resolved conflicting branch changes to current main.",
                    )
                )
                yield _sse(AgentEvent(event="done"))
                return
            candidate_problem = BranchMergeCandidateProblem(
                "An empty merge candidate omits non-conflicting source branch changes."
            )
            candidate = None
        if candidate is not None:
            try:
                outcome.prepared = prepare_branch_merge_with_history(
                    main_history,
                    candidate,
                    expected_main_head=context.main_head,
                    context=context,
                )
            except RevisionConflict:
                if classify_refreshed_context(context, load_context()) == "unchanged":
                    outcome.status = "retryable"
                    outcome.diagnostic = (
                        "Main admission reported movement before the refreshed main head became "
                        "available; retry the merge task."
                    )
                    yield _sse(AgentEvent(event="error", text=outcome.diagnostic))
                    return
                continue
            except BranchMergeSemanticConflict as exc:
                candidate_problem = exc
            else:
                try:
                    outcome.committed = commit_branch_merge_with_history(
                        main_history,
                        candidate,
                        expected_main_head=context.main_head,
                    )
                except BranchMergeAlreadyCommitted as exc:
                    receipt = branch_merge_receipt_from_committed_patch(exc.patch)
                    outcome.prepared = None
                    outcome.status = "committed"
                    outcome.result_main_head = receipt.result_main_head
                    outcome.rebased_main_head = receipt.provenance.rebased_main_head
                    outcome.receipt = receipt
                    yield _sse(
                        AgentEvent(
                            event="answer",
                            text=(
                                "This graph branch head was already merged into main revision "
                                f"{receipt.result_main_head.revision}."
                            ),
                        )
                    )
                    yield _sse(AgentEvent(event="done"))
                    return
                except BranchMergeAlreadyResolved as exc:
                    receipt = exc.receipt
                    outcome.prepared = None
                    outcome.status = "noop"
                    outcome.result_main_head = receipt.result_main_head
                    outcome.rebased_main_head = receipt.provenance.rebased_main_head
                    outcome.receipt = receipt
                    yield _sse(
                        AgentEvent(
                            event="answer",
                            text="This graph branch head was already resolved without a main Patch.",
                        )
                    )
                    yield _sse(AgentEvent(event="done"))
                    return
                except RevisionConflict:
                    outcome.prepared = None
                    if classify_refreshed_context(context, load_context()) == "unchanged":
                        outcome.status = "retryable"
                        outcome.diagnostic = (
                            "Main moved during atomic append before the refreshed head became "
                            "available; retry the merge task."
                        )
                        yield _sse(AgentEvent(event="error", text=outcome.diagnostic))
                        return
                    continue
                except BranchMergeSemanticConflict as exc:
                    outcome.prepared = None
                    candidate_problem = exc
                else:
                    committed = outcome.committed
                    assert committed is not None
                    outcome.status = "committed"
                    outcome.result_main_head = committed.projection.head
                    outcome.receipt = branch_merge_receipt_from_committed_patch(committed.patch)
                    yield _sse(
                        AgentEvent(
                            event="answer",
                            text=(
                                "Merged graph branch "
                                f"{context.metadata.branch_id} into main revision "
                                f"{committed.projection.head.revision}."
                            ),
                        )
                    )
                    yield _sse(AgentEvent(event="done"))
                    return

        assert candidate_problem is not None
        if session_id is None or outcome.correction_rounds >= PATCH_CORRECTION_MAX_ROUNDS:
            outcome.status = "rejected"
            outcome.diagnostic = candidate_problem.message
            yield _sse(AgentEvent(event="error", text=candidate_problem.message))
            return

        outcome.correction_rounds += 1
        diagnostics_path = _stage_json_task_input(
            stage.local_stage,
            stage.remote_stage,
            f"task-{token}-branch-merge-correction-{outcome.correction_rounds}.json",
            {
                "kind": "branch_merge_patch",
                "context_id": context.context_id,
                "problem": candidate_problem.message,
                "conflicts": [item.model_dump(mode="json") for item in candidate_problem.conflicts],
            },
        )
        assert original_contract_path is not None
        context_path = _stage_merge_context_reference(stage, token, context, outcome.rebase_rounds)
        contract = branch_merge_correction_contract(
            original_contract_path=original_contract_path,
            context_path=context_path,
            context_id=context.context_id,
            patch_path=_patch_path(stage),
            diagnostics_path=diagnostics_path,
            validator_command=validator_command,
        )
        contract_path, prompt = _stage_task_contract(
            stage.local_stage,
            stage.remote_stage,
            f"task-{token}-branch-merge-correction-{outcome.correction_rounds}.md",
            contract,
            execution=execution,
            role=f"branch_merge_correction_{outcome.correction_rounds}",
        )
        pre_launch_digest = _existing_patch_digest(stage.workspace, stage.remote_stage)
        _record_merge_launch(
            execution,
            request,
            prompt=prompt,
            contract_path=contract_path,
            write_scope=write_scope,
            context=context,
            continuation="patch_correction",
            round_number=outcome.correction_rounds,
        )
        provider, events = _provider_turn(
            launcher,
            request,
            prompt,
            stage=stage,
            write_scope=write_scope,
            execution=execution,
            binary=binary,
            session_id=session_id,
            required_session_id=session_id,
        )
        async with aclosing(events) as frames:
            async for frame in frames:
                yield frame
        if provider.session_id:
            session_id = provider.session_id
            outcome.native_session_id = session_id
        if provider.paused:
            outcome.status = "paused"
            return
        if provider.failed or not provider.completed:
            outcome.status = "rejected"
            outcome.diagnostic = "The branch merge correction turn did not complete."
            return
        try:
            candidate_text = _read_candidate_text(stage)
            digest = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
            if pre_launch_digest is not None and digest == pre_launch_digest:
                raise BranchMergeCandidateProblem(
                    "The correction left patch.json byte-identical; rewrite the candidate."
                )
            candidate = parse_branch_merge_candidate(candidate_text, context)
            candidate_problem = None
        except (AgentOutputProblem, BranchMergeCandidateProblem, OSError, StateUnavailable) as exc:
            candidate_problem = BranchMergeCandidateProblem(str(exc))
            candidate = None


def _require_change_shape(
    change: str,
    before: BaseModel | None,
    after: BaseModel | None,
    identity: str,
    *,
    id_field: str = "id",
) -> None:
    if change == "created" and (before is not None or after is None):
        raise ValueError("created semantic deltas require only an after value")
    if change == "updated" and (before is None or after is None):
        raise ValueError("updated semantic deltas require before and after values")
    if change == "removed" and (before is None or after is not None):
        raise ValueError("removed semantic deltas require only a before value")
    for value in (before, after):
        if value is not None and getattr(value, id_field) != identity:
            raise ValueError("semantic delta identity does not match its payload")


def _typed_collection_delta(
    before: Mapping[str, BaseModel],
    after: Mapping[str, BaseModel],
    *,
    model: type[BaseModel],
    id_field: str,
    bookkeeping: frozenset[str],
) -> list[Any]:
    result: list[Any] = []
    for identity in sorted(set(before) | set(after)):
        old = before.get(identity)
        new = after.get(identity)
        if old is None:
            result.append(
                model.model_validate({"change": "created", id_field: identity, "after": new})
            )
        elif new is None:
            result.append(
                model.model_validate({"change": "removed", id_field: identity, "before": old})
            )
        elif _semantic_document(old, bookkeeping) != _semantic_document(new, bookkeeping):
            result.append(
                model.model_validate(
                    {"change": "updated", id_field: identity, "before": old, "after": new}
                )
            )
    return result


def _graph_semantic_document(state: GraphState) -> dict[str, JsonValue]:
    return {
        "nodes": {
            identity: _semantic_document(value, _SEMANTIC_NODE_BOOKKEEPING)
            for identity, value in state.nodes.items()
        },
        "edges": {
            identity: _semantic_document(value, _SEMANTIC_EDGE_BOOKKEEPING)
            for identity, value in state.edges.items()
        },
        "proposals": {
            identity: _semantic_document(value, _SEMANTIC_PROPOSAL_BOOKKEEPING)
            for identity, value in state.proposals.items()
        },
        "ambiguities": {
            identity: _semantic_document(value, _SEMANTIC_AMBIGUITY_BOOKKEEPING)
            for identity, value in state.ambiguities.items()
        },
        "glossary": {
            identity: _semantic_document(value, _SEMANTIC_GLOSSARY_BOOKKEEPING)
            for identity, value in state.glossary.items()
        },
        "project_truth_scope": _semantic_document(state.project_truth_scope, frozenset()),
        "ontology": _semantic_document(state.ontology, frozenset()),
        "coverage": _semantic_document(state.coverage, frozenset()),
    }


def _graph_semantic_write_paths(
    before: GraphState,
    after: GraphState,
) -> set[SemanticWritePath]:
    old = _graph_semantic_document(before)
    new = _graph_semantic_document(after)
    paths: set[SemanticWritePath] = set()
    for collection in ("nodes", "edges", "proposals", "ambiguities", "glossary"):
        old_values = old[collection]
        new_values = new[collection]
        assert isinstance(old_values, dict) and isinstance(new_values, dict)
        for identity in set(old_values) | set(new_values):
            if identity not in old_values or identity not in new_values:
                paths.add((collection, identity, "$"))
                continue
            paths.update(
                _document_write_paths(
                    old_values[identity],
                    new_values[identity],
                    prefix=(collection, identity),
                )
            )
    for field in ("project_truth_scope", "ontology", "coverage"):
        old_value = old[field]
        new_value = new[field]
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            paths.update(_document_write_paths(old_value, new_value, prefix=(field,)))
        elif old_value != new_value:
            paths.add((field, "$"))
    return paths


def _document_write_paths(
    before: object,
    after: object,
    *,
    prefix: SemanticWritePath,
) -> set[SemanticWritePath]:
    if before == after:
        return set()
    if isinstance(before, dict) and isinstance(after, dict):
        paths: set[SemanticWritePath] = set()
        for key in set(before) | set(after):
            if key not in before or key not in after:
                value = after[key] if key in after else before[key]
                suffix = ("$",) if isinstance(value, dict) else ()
                paths.add((*prefix, str(key), *suffix))
                continue
            paths.update(
                _document_write_paths(
                    before[key],
                    after[key],
                    prefix=(*prefix, str(key)),
                )
            )
        return paths
    return {prefix}


def _branch_conflict_paths(
    conflicts: list[BranchMergeConflict],
) -> set[SemanticWritePath]:
    paths: set[SemanticWritePath] = set()
    globals_ = {"project_truth_scope", "ontology", "coverage"}
    for conflict in conflicts:
        parts = tuple(part for part in conflict.field_path.split(".") if part)
        if conflict.collection in globals_:
            if parts and parts[0] == conflict.collection:
                parts = parts[1:]
            path = (conflict.collection, *(parts or ("$",)))
        else:
            assert conflict.entity_id is not None
            path = (conflict.collection, conflict.entity_id, *(parts or ("$",)))
        if path[-1] != "$" and any(
            isinstance(value, dict) for value in (conflict.base, conflict.branch, conflict.main)
        ):
            path = (*path, "$")
        paths.add(path)
    return paths


def _declared_bookkeeping_sensitive_write_paths(operation: object) -> set[SemanticWritePath]:
    """Name declared writes whose no-op application still mutates node bookkeeping."""

    if isinstance(operation, UpdateNodesOperation):
        return {
            ("nodes", update.id, field) for update in operation.nodes for field in update.changes
        }
    if isinstance(operation, SetStandingOperation):
        return {("nodes", operation.node_id, "standing")}
    return set()


def _semantic_path_authorizes_declared_write(
    allowed: SemanticWritePath,
    declared: SemanticWritePath,
) -> bool:
    if _semantic_path_covers(allowed, declared):
        return True
    allowed_parts = allowed[:-1] if allowed and allowed[-1] == "$" else allowed
    return allowed_parts[: len(declared)] == declared


def _semantic_path_covers(rule: SemanticWritePath, path: SemanticWritePath) -> bool:
    if rule and rule[-1] == "$":
        return path[: len(rule) - 1] == rule[:-1]
    return rule == path


def _semantic_path_value(document: dict[str, JsonValue], path: SemanticWritePath) -> object:
    parts = path[:-1] if path and path[-1] == "$" else path
    value: object = document
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _render_semantic_path(path: SemanticWritePath) -> str:
    return "/".join(path[:-1] if path and path[-1] == "$" else path)


def _semantic_change_is_subsumed(
    change: str,
    before: BaseModel | None,
    after: BaseModel | None,
    main: BaseModel | None,
    bookkeeping: frozenset[str],
) -> bool:
    before_document = _optional_semantic_document(before, bookkeeping)
    after_document = _optional_semantic_document(after, bookkeeping)
    main_document = _optional_semantic_document(main, bookkeeping)
    if change == "created":
        return main_document == after_document
    if change == "removed":
        return main_document is None
    return _branch_changes_are_present(before_document, after_document, main_document)


_MISSING = object()


def _branch_changes_are_present(base: object, branch: object, main: object) -> bool:
    if base == branch:
        return True
    if branch == main:
        return True
    if isinstance(base, dict) and isinstance(branch, dict) and isinstance(main, dict):
        for key in set(base) | set(branch):
            base_value = base.get(key, _MISSING)
            branch_value = branch.get(key, _MISSING)
            if base_value == branch_value:
                continue
            if not _branch_changes_are_present(
                base_value,
                branch_value,
                main.get(key, _MISSING),
            ):
                return False
        return True
    return False


def _collection_conflicts(
    collection: str,
    base: Mapping[str, BaseModel],
    branch: Mapping[str, BaseModel],
    main: Mapping[str, BaseModel],
    bookkeeping: frozenset[str],
) -> list[BranchMergeConflict]:
    conflicts: list[BranchMergeConflict] = []
    for identity in sorted(set(base) | set(branch)):
        base_value = _optional_semantic_document(base.get(identity), bookkeeping)
        branch_value = _optional_semantic_document(branch.get(identity), bookkeeping)
        main_value = _optional_semantic_document(main.get(identity), bookkeeping)
        if base_value == branch_value:
            continue
        if base_value is None:
            if main_value is not None and main_value != branch_value:
                conflicts.append(
                    _conflict(
                        "branch_created_main_created",
                        collection,
                        identity,
                        "$",
                        base_value,
                        branch_value,
                        main_value,
                    )
                )
            continue
        if branch_value is None:
            if main_value not in (base_value, None):
                conflicts.append(
                    _conflict(
                        "branch_removed_main_changed",
                        collection,
                        identity,
                        "$",
                        base_value,
                        branch_value,
                        main_value,
                    )
                )
            continue
        if main_value is None:
            conflicts.append(
                _conflict(
                    "branch_changed_main_removed",
                    collection,
                    identity,
                    "$",
                    base_value,
                    branch_value,
                    main_value,
                )
            )
            continue
        for field_path, base_field, branch_field, main_field in _conflicting_fields(
            base_value,
            branch_value,
            main_value,
        ):
            conflicts.append(
                _conflict(
                    "both_changed",
                    collection,
                    identity,
                    field_path,
                    base_field,
                    branch_field,
                    main_field,
                )
            )
    return conflicts


def _conflicting_fields(
    base: JsonValue,
    branch: JsonValue,
    main: JsonValue,
    *,
    prefix: str = "",
) -> list[tuple[str, JsonValue, JsonValue, JsonValue]]:
    if base in (branch, main) or branch == main:
        return []
    if isinstance(base, dict) and isinstance(branch, dict) and isinstance(main, dict):
        conflicts: list[tuple[str, JsonValue, JsonValue, JsonValue]] = []
        for key in sorted(set(base) | set(branch) | set(main)):
            path = f"{prefix}.{key}" if prefix else key
            conflicts.extend(
                _conflicting_fields(
                    base.get(key),
                    branch.get(key),
                    main.get(key),
                    prefix=path,
                )
            )
        return conflicts
    return [(prefix or "$", base, branch, main)]


def _conflict(
    code: str,
    collection: str,
    identity: str,
    field_path: str,
    base: JsonValue,
    branch: JsonValue,
    main: JsonValue,
) -> BranchMergeConflict:
    return BranchMergeConflict.model_validate(
        {
            "code": code,
            "collection": collection,
            "entity_id": identity,
            "field_path": field_path,
            "base": base,
            "branch": branch,
            "main": main,
            "message": (
                f"Branch and main changed {collection}/{identity} at {field_path} "
                "differently from the base."
            ),
        }
    )


def _semantic_document(value: object, bookkeeping: frozenset[str]) -> JsonValue:
    if isinstance(value, BaseModel):
        document: JsonValue = value.model_dump(mode="json", exclude=bookkeeping)
    else:
        document = _jsonable(value)
    return document


def _optional_semantic_document(
    value: BaseModel | None,
    bookkeeping: frozenset[str],
) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    document = _semantic_document(value, bookkeeping)
    if not isinstance(document, dict):  # pragma: no cover - every collection item is a model
        raise TypeError("semantic graph collection item is not an object")
    return document


def _jsonable(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _context_id(payload: object) -> str:
    return _canonical_sha256({"kind": "branch_merge_context", "payload": payload})


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validation_diagnostic(
    report: object,
) -> tuple[str, list[TransitionConflictDetail]]:
    messages = getattr(report, "messages", ())
    rendered: list[str] = []
    conflicts: list[TransitionConflictDetail] = []
    for item in messages:
        message = getattr(item, "message", None)
        if isinstance(message, str) and message.strip():
            rendered.append(" ".join(message.split()))
        if getattr(item, "code", None) == "transition-conflict":
            with suppress(ValueError):
                conflicts.append(
                    TransitionConflictDetail(
                        operation_index=getattr(item, "operation_index", None),
                        rule_id=getattr(item, "rule_id", None),
                        cause_chain=getattr(item, "cause_chain", ()),
                        affected_ids=[
                            *getattr(item, "related_node_ids", ()),
                            *getattr(item, "related_edge_ids", ()),
                        ],
                        invariant=getattr(item, "failed_invariant", None)
                        or "graph transition preparation",
                        message=message or "Graph transition preparation failed.",
                    )
                )
    if not rendered:
        return (
            "Branch merge candidate was rejected by canonical graph validation.",
            conflicts,
        )
    return " ".join(dict.fromkeys(rendered))[:4_000], conflicts


def _require_prepared_provenance(
    prepared: Patch,
    candidate: Patch,
    expected_main_head: GraphHeadRef,
) -> None:
    if prepared.branch_merge != candidate.branch_merge:
        raise RuntimeError("branch merge preparation changed its exact provenance")
    if prepared.authorized_by != candidate.authorized_by:
        raise RuntimeError("branch merge preparation changed its authorizer snapshot")
    if prepared.profile != "orchestrator" or prepared.task_id != candidate.task_id:
        raise RuntimeError("branch merge preparation changed its task attribution")
    if prepared.transition is None:
        raise RuntimeError("branch merge was prepared without a transition trace")
    if prepared.transition.pre_head != expected_main_head:
        raise RevisionConflict("branch merge preparation used a different main head")
    if prepared.revision != expected_main_head.revision + 1:
        raise RuntimeError("branch merge preparation did not target exactly one next revision")


def _patch_path(stage: BranchMergeStage) -> str:
    if stage.remote_stage is not None:
        return str(stage.remote_stage.workspace / "patch.json")
    return str(stage.workspace / "patch.json")


def _stage_merge_context(
    stage: BranchMergeStage,
    token: str,
    context: BranchMergeContext,
    *,
    round_number: int,
) -> str:
    return _stage_json_task_input(
        stage.local_stage,
        stage.remote_stage,
        f"task-{token}-branch-merge-context-{round_number}-{context.context_id[:16]}.json",
        context.model_dump(mode="json"),
    )


def _stage_merge_context_reference(
    stage: BranchMergeStage,
    token: str,
    context: BranchMergeContext,
    round_number: int,
) -> str:
    label = f"task-{token}-branch-merge-context-{round_number}-{context.context_id[:16]}.json"
    if stage.remote_stage is not None:
        assert stage.remote_stage.root is not None
        return str(stage.remote_stage.root / "inputs" / label)
    assert stage.local_stage is not None
    target = stage.local_stage / "inputs" / label
    if not target.is_file():
        raise ValueError("branch merge correction lost its immutable merge context")
    return str(target)


def _clear_patch_candidates(stage: BranchMergeStage) -> None:
    if stage.remote_stage is not None:
        names = stage.remote_stage.list_workspace_entries()
        for name in names:
            if not _is_agent_json_output(name):
                continue
            stage.remote_stage.remove_workspace_file(name)
        return
    for item in stage.workspace.iterdir():
        if not _is_agent_json_output(item.name):
            continue
        if item.is_symlink() or item.is_file():
            item.unlink()
            continue
        raise ValueError(f"branch merge workspace JSON output is not a regular file: {item.name}")


def _is_agent_json_output(name: str) -> bool:
    """Keep RCP-owned command credentials while clearing inherited agent output."""

    folded = name.casefold()
    return folded.endswith(".json") and not folded.startswith(
        ("rcp-command-", ".rcp-command-", ".rcp-mailbox-")
    )


def _read_candidate_text(stage: BranchMergeStage) -> str:
    text, _name = _collect_patch_text(stage.workspace, stage.remote_stage)
    return text


def _provider_turn(
    launcher: AgentLauncher,
    request: RunRequest,
    prompt: str,
    *,
    stage: BranchMergeStage,
    write_scope: ProjectWriteScope,
    execution: Any | None,
    binary: str | None,
    session_id: str | None,
    required_session_id: str | None,
) -> tuple[_ProviderOutcome, AsyncIterator[str]]:
    provider_outcome = _ProviderOutcome(session_id=session_id)
    inputs = (
        Path(str(stage.remote_stage.root / "inputs"))
        if stage.remote_stage is not None and stage.remote_stage.root is not None
        else stage.local_stage / "inputs"  # type: ignore[operator]
    )
    stream = _stream_agent_events(
        launcher,
        request,
        prompt,
        workspace=stage.workspace,
        session_id=session_id,
        read_dirs=[inputs],
        # The provider API treats write_dirs as admitted repository roots.
        # The scratch workspace is already the cwd/workspace root carried by
        # ProjectWriteScope, while a graph-only merge admits no repositories.
        write_dirs=[],
        write_scope=write_scope,
        execution_host=write_scope.execution_host,
        execution=execution,
        remote_stage=stage.remote_stage,
        capability="orchestrate",
        outcome=provider_outcome,
        binary=binary,
        required_session_id=required_session_id,
    )
    return provider_outcome, stream


def _record_merge_launch(
    execution: Any | None,
    request: RunRequest,
    *,
    prompt: str,
    contract_path: str,
    write_scope: ProjectWriteScope,
    context: BranchMergeContext,
    continuation: str,
    round_number: int,
) -> None:
    _record_agent_launch_receipt(
        execution,
        request,
        prompt=prompt,
        contract_path=contract_path,
        remote=bool(write_scope.execution_host),
        resumed=continuation != "initial",
        write_scope=write_scope,
        continuation=continuation,
        extra={
            "surface": "branch_merge",
            "mode": "branch_merge",
            "capability": "orchestrate",
            "network_access": True,
            "launch_kind": continuation,
            "round": round_number,
            "merge_context_id": context.context_id,
            "merge_id": branch_merge_id(context.metadata),
            "branch_id": context.metadata.branch_id,
            "branch_head": context.metadata.head.model_dump(mode="json"),
            "main_head": context.main_head.model_dump(mode="json"),
            "write_directory_count": 1,
            "repository_write_root_count": 0,
        },
    )
