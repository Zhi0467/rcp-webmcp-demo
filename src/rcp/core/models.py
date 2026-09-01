from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

DISPLAY_NAME_MAX_LENGTH = 120
EVIDENCE_ASSESSMENT_SCOPE_MAX_LENGTH = 500
EVIDENCE_ASSESSMENT_QUALIFICATION_MAX_LENGTH = 300
EVIDENCE_ASSESSMENT_MAX_QUALIFICATIONS = 12


def normalize_display_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("display name must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("display name must not be blank")
    if len(normalized) > DISPLAY_NAME_MAX_LENGTH:
        raise ValueError(f"display name must be at most {DISPLAY_NAME_MAX_LENGTH} characters")
    if any(unicodedata.category(character) in {"Cc", "Zl", "Zp"} for character in normalized):
        raise ValueError("display name must be a single line without control characters")
    return normalized


def utc_now() -> datetime:
    return datetime.now(UTC)


class Standing(StrEnum):
    ASSERTED = "asserted"
    ACCEPTED = "accepted"
    CONTESTED = "contested"


class SourceRef(BaseModel):
    machine: str
    truth_repository: str
    source: Literal["claude", "codex", "app_chat"]
    session_id: str
    record_uuid: str
    timestamp: datetime
    excerpt: str = Field(max_length=800)


class ExperimentDecisionPin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    decision_revision: int = Field(ge=0)
    selected_option: str = Field(min_length=1)


class ExperimentAttemptDebug(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mechanical_fault: str = Field(min_length=1)
    change: str = Field(min_length=1)
    predicted_effect: str = Field(min_length=1)


ACTIVE_EXPERIMENT_ATTEMPT_STATUSES = frozenset({"planned", "submitted", "running"})


class ExperimentAttempt(BaseModel):
    id: str
    sequence: int = Field(ge=1)
    purpose: str
    attempt_kind: Literal["external_run", "proposal_only"] = "external_run"
    decision_bundle: list[ExperimentDecisionPin] = Field(default_factory=list)
    debug: ExperimentAttemptDebug | None = None
    configuration: str = ""
    status: Literal[
        "planned", "submitted", "running", "failed", "completed", "cancelled", "superseded"
    ] = "planned"
    job_refs: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    outcome: str | None = None
    failure_reason: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class BaseNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    extension_type: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    )
    extension_fields: dict[str, str | float | bool | list[str]] = Field(default_factory=dict)
    standing: Standing = Standing.ASSERTED
    created_rev: int = 0
    updated_rev: int = 0
    source_refs: list[SourceRef] = Field(default_factory=list)


class ResearchQuestion(BaseNode):
    type: Literal["research_question"]
    question: str
    motivation: str = ""
    scope: str = ""
    status: Literal["open", "answered", "abandoned", "superseded"] = "open"


class Hypothesis(BaseNode):
    type: Literal["hypothesis"]
    statement: str
    rationale: str = ""
    predictions: list[str] = Field(default_factory=list)
    scope: str = ""
    status: Literal["proposed", "active", "supported", "weakened", "rejected", "superseded"] = (
        "proposed"
    )


class Decision(BaseNode):
    type: Literal["decision"]
    question: str
    options: list[str] = Field(default_factory=list)
    selected_option: str | None = None
    rationale: str | None = None
    consequences: list[str] = Field(default_factory=list)
    status: Literal["open", "ready", "decided", "revisit", "superseded"] = "open"


class Experiment(BaseNode):
    type: Literal["experiment"]
    objective: str
    design: str = ""
    expected_outcomes: list[str] = Field(default_factory=list)
    interpretation_rules: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    invocation_ceiling: int = Field(default=5, ge=1)
    status: Literal[
        "proposed",
        "designing",
        "implementing",
        "debugging",
        "running",
        "analyzing",
        "completed",
        "unspecified",
        "abandoned",
        "superseded",
    ] = "proposed"
    attempts: list[ExperimentAttempt] = Field(default_factory=list)
    current_summary: str = ""
    next_action: str | None = None
    current_summary_stale: bool = False
    next_action_stale: bool = False

    @model_validator(mode="before")
    @classmethod
    def migrate_attempt_ceiling(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "attempt_ceiling" not in value:
            return value
        migrated = dict(value)
        migrated["invocation_ceiling"] = migrated.pop("attempt_ceiling")
        return migrated


# The human's own verdict that an Experiment is finished with, which outranks how
# any one bounded episode inside it happened to end.
CLOSED_EXPERIMENT_STATUSES = frozenset({"completed", "abandoned", "superseded"})


class Evidence(BaseNode):
    type: Literal["evidence"]
    observation: str
    interpretation: str = ""
    role: Literal["result", "diagnostic"] = "result"
    legacy_strength: Literal["diagnostic", "preliminary", "supporting", "confirmatory"] | None = (
        None
    )
    validity: Literal["valid", "qualified", "invalid", "superseded"] = "valid"
    origin: Literal[
        "internal_run", "external_publication", "external_instance", "analytic", "unknown"
    ] = "unknown"
    artifact_refs: list[str] = Field(default_factory=list)


class Blocker(BaseNode):
    type: Literal["blocker"]
    description: str
    blocker_type: Literal[
        "scientific", "design", "data", "implementation", "infrastructure", "unknown"
    ] = "unknown"
    status: Literal["open", "resolved", "superseded"] = "open"
    resolution_condition: str = ""
    recommended_action: str | None = None


ProjectNode = Annotated[
    ResearchQuestion | Hypothesis | Decision | Experiment | Evidence | Blocker,
    Field(discriminator="type"),
]


HUMAN_EDITABLE_NODE_FIELDS: dict[str, frozenset[str]] = {
    "research_question": frozenset({"title", "question", "motivation", "scope"}),
    "hypothesis": frozenset({"title", "statement", "rationale", "predictions", "scope"}),
    "decision": frozenset({"title", "question", "options", "rationale", "consequences", "status"}),
    "experiment": frozenset(
        {
            "title",
            "objective",
            "design",
            "expected_outcomes",
            "interpretation_rules",
            "completion_criteria",
            # Replay compatibility for approval patches written before the
            # invocation-budget rename. New output uses invocation_ceiling.
            "attempt_ceiling",
            "invocation_ceiling",
            "current_summary",
            "next_action",
        }
    ),
    "evidence": frozenset({"title", "observation", "interpretation"}),
    "blocker": frozenset(
        {"title", "description", "status", "resolution_condition", "recommended_action"}
    ),
}


NodeType = Literal[
    "research_question", "hypothesis", "decision", "experiment", "evidence", "blocker"
]
BaseRelation = Literal[
    "has_subquestion",
    "has_hypothesis",
    "has_decision",
    "tests",
    "governed_by",
    "produces",
    "informs",
    "addresses",
    "blocked_by",
    "supports",
    "weakens",
    "contradicts",
    "refutes",
    "inconclusive",
    "requires_decision",
    "supersedes",
    "duplicate_of",
]
RelationLayer = Literal["epistemic", "action", "seam", "meta"]
ALL_NODE_TYPES: frozenset[str] = frozenset(
    {"research_question", "hypothesis", "decision", "experiment", "evidence", "blocker"}
)


class OntologyTypeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    definition: str = Field(min_length=1)
    base_type: NodeType
    layer: Literal["epistemic", "action"]
    deprecated: bool = False


class OntologyFieldDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_type: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    name: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    definition: str = Field(min_length=1)
    kind: Literal["text", "number", "boolean", "text_list"]
    required: bool = False
    agent_writable: bool = True
    deprecated: bool = False


class OntologyRelationDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    definition: str = Field(min_length=1)
    source_types: list[str] = Field(min_length=1)
    target_types: list[str] = Field(min_length=1)
    layer: Literal["epistemic", "action"]
    deprecated: bool = False


class OntologyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    types: list[OntologyTypeDefinition] = Field(default_factory=list)
    fields: list[OntologyFieldDefinition] = Field(default_factory=list)
    relations: list[OntologyRelationDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_names(self) -> OntologyState:
        for label, names in (
            ("type", [item.name for item in self.types]),
            ("field", [f"{item.owner_type}.{item.name}" for item in self.fields]),
            ("relation", [item.name for item in self.relations]),
        ):
            if len(names) != len(set(names)):
                raise ValueError(f"ontology contains duplicate {label} definitions")
        return self


@dataclass(frozen=True)
class RelationSpec:
    source_types: frozenset[str]
    target_types: frozenset[str]
    layer: RelationLayer
    same_type: bool = False


RELATION_SPEC: dict[BaseRelation, RelationSpec] = {
    "has_subquestion": RelationSpec(
        frozenset({"research_question"}), frozenset({"research_question"}), "epistemic"
    ),
    "has_hypothesis": RelationSpec(
        frozenset({"research_question"}), frozenset({"hypothesis"}), "epistemic"
    ),
    "supports": RelationSpec(frozenset({"evidence"}), frozenset({"hypothesis"}), "epistemic"),
    "weakens": RelationSpec(frozenset({"evidence"}), frozenset({"hypothesis"}), "epistemic"),
    "refutes": RelationSpec(frozenset({"evidence"}), frozenset({"hypothesis"}), "epistemic"),
    "inconclusive": RelationSpec(frozenset({"evidence"}), frozenset({"hypothesis"}), "epistemic"),
    "contradicts": RelationSpec(
        frozenset({"evidence", "hypothesis"}), frozenset({"hypothesis"}), "epistemic"
    ),
    "tests": RelationSpec(frozenset({"experiment"}), frozenset({"hypothesis"}), "seam"),
    "produces": RelationSpec(frozenset({"experiment"}), frozenset({"evidence"}), "seam"),
    "informs": RelationSpec(frozenset({"evidence"}), frozenset({"decision"}), "action"),
    "addresses": RelationSpec(frozenset({"evidence"}), frozenset({"blocker"}), "action"),
    "has_decision": RelationSpec(
        frozenset({"research_question"}), frozenset({"decision"}), "action"
    ),
    "governed_by": RelationSpec(frozenset({"experiment"}), frozenset({"decision"}), "action"),
    "blocked_by": RelationSpec(
        frozenset({"experiment", "decision", "research_question"}),
        frozenset({"blocker"}),
        "action",
    ),
    "requires_decision": RelationSpec(frozenset({"blocker"}), frozenset({"decision"}), "action"),
    "supersedes": RelationSpec(ALL_NODE_TYPES, ALL_NODE_TYPES, "meta", same_type=True),
    "duplicate_of": RelationSpec(ALL_NODE_TYPES, ALL_NODE_TYPES, "meta", same_type=True),
}


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: Literal["direct", "indirect", "contextual"]
    weight: Literal["limited", "moderate", "strong"]
    scope: str | None = Field(default=None, max_length=EVIDENCE_ASSESSMENT_SCOPE_MAX_LENGTH)
    qualifications: list[
        Annotated[
            str,
            Field(min_length=1, max_length=EVIDENCE_ASSESSMENT_QUALIFICATION_MAX_LENGTH),
        ]
    ] = Field(default_factory=list, max_length=EVIDENCE_ASSESSMENT_MAX_QUALIFICATIONS)

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_scope(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("assessment scope must not be blank")
        return normalized

    @field_validator("qualifications", mode="before")
    @classmethod
    def normalize_qualifications(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized: list[Any] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                normalized.append(item)
                continue
            qualification = item.strip()
            if not qualification:
                raise ValueError("assessment qualifications must not be blank")
            if qualification in seen:
                raise ValueError("assessment qualifications must not contain duplicates")
            seen.add(qualification)
            normalized.append(qualification)
        return normalized


class Edge(BaseModel):
    # Layer is backend-owned; a supplied one is always discarded. What lands here
    # is the relation's *declared* layer. Materialization then narrows it to the
    # edge's real layer via `ontology.edge_layer`, which needs the endpoint types
    # and so cannot run inside this validator.
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    relation: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    layer: RelationLayer
    explanation: str = ""
    assessment: EvidenceAssessment | None = None
    created_rev: int = 0

    @model_validator(mode="before")
    @classmethod
    def derive_base_relation_layer(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        relation = value.get("relation")
        spec = RELATION_SPEC.get(relation) if isinstance(relation, str) else None
        if spec is None:
            return value
        return {**value, "layer": spec.layer}


class GatedCard(BaseModel):
    situation_cold: str = ""
    why_human_now: str = ""
    consequences: str = ""
    decision_needed: str = ""


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    card: GatedCard
    ops: list[ProposalOperation]
    related_node_ids: list[str] = Field(default_factory=list)
    related_edge_ids: list[str] = Field(default_factory=list)
    related_config_keys: list[str] = Field(default_factory=list)
    base_rev: int = 0
    status: Literal["pending", "approved", "rejected", "withdrawn"] = "pending"
    created_by: Literal["agent", "human"] = "agent"
    created_by_operation_id: str | None = None
    raised_rev: int = 0
    resolved_rev: int | None = None
    resolved_by: Literal["agent", "human"] | None = None
    resolved_by_operation_id: str | None = None
    resolution_reason: str | None = None
    rejection_reason: str | None = None

    @field_serializer("ops")
    def serialize_ops(
        self,
        operations: list[ProposalOperation] | list[dict[str, Any]],
        info: SerializationInfo,
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for operation in operations:
            item = (
                operation.model_dump(mode=info.mode, exclude_unset=True)
                if isinstance(operation, BaseModel)
                else dict(operation)
            )
            if str(item.get("intent", "")).startswith("legacy_"):
                item.pop("intent")
            serialized.append(item)
        return serialized


class Ambiguity(BaseModel):
    id: str
    question: str
    why_it_matters: str
    candidates: list[str] = Field(default_factory=list)
    related_node_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    status: Literal["open", "resolved", "dismissed"] = "open"
    raised_rev: int = 0


class GlossaryTerm(BaseModel):
    term: str
    plain_definition: str
    where_defined: str | None = None
    updated_rev: int = 0


class CoverageBoundary(BaseModel):
    repositories_seen: list[str] = Field(default_factory=list)
    repositories_never_seen: list[str] = Field(default_factory=list)
    sessions_read: list[str] = Field(default_factory=list)
    sessions_skipped: list[str] = Field(default_factory=list)
    earliest_timestamp: datetime | None = None
    note: str = "No seed has completed."


class ValidationMessage(BaseModel):
    level: Literal["flag", "reject"]
    code: str
    message: str
    patch_revision: int | None = None
    related_node_ids: list[str] = Field(default_factory=list)
    related_edge_ids: list[str] = Field(default_factory=list)
    operation_index: int | None = Field(default=None, ge=0)
    rule_id: str | None = None
    cause_chain: list[dict[str, Any]] = Field(default_factory=list)
    failed_invariant: str | None = None


class BeliefTransition(BaseModel):
    hypothesis_id: str
    from_status: str
    to_status: str
    revision: int
    cause: dict[str, Any]


class ReplayFailure(BaseModel):
    revision: int
    created_at: datetime
    code: str
    message: str


class GraphState(BaseModel):
    revision: int = 0
    project_truth_scope: list[str] = Field(default_factory=list)
    config_revisions: dict[str, int] = Field(default_factory=dict)
    nodes: dict[str, ProjectNode] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)
    proposals: dict[str, Proposal] = Field(default_factory=dict)
    ambiguities: dict[str, Ambiguity] = Field(default_factory=dict)
    glossary: dict[str, GlossaryTerm] = Field(default_factory=dict)
    ontology: OntologyState = Field(default_factory=OntologyState)
    coverage: CoverageBoundary = Field(default_factory=CoverageBoundary)
    validation_messages: list[ValidationMessage] = Field(default_factory=list)
    belief_transitions: list[BeliefTransition] = Field(default_factory=list)
    replay_status: Literal["complete", "degraded"] = "complete"
    replay_failure: ReplayFailure | None = None
    last_refresh_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def adapt_persisted_snapshot(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        # Local import keeps the domain-model/operation dependency acyclic.
        from rcp.core.operations import adapt_persisted_graph_state_document

        return adapt_persisted_graph_state_document(value)


def _canonical_uuid4(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError("must be a canonical UUIDv4") from exc
    if str(parsed) != value or parsed.version != 4:
        raise ValueError("must be a canonical UUIDv4")
    return value


class ProjectIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    home_space_id: str
    action: Literal["created", "adopted"]

    _validate_uuid4 = field_validator("project_id", "home_space_id")(_canonical_uuid4)


class AuthorizedHuman(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space_id: str
    user_id: str
    display_name: str = Field(min_length=1, max_length=DISPLAY_NAME_MAX_LENGTH)

    _validate_uuid4 = field_validator("space_id", "user_id")(_canonical_uuid4)
    _normalize_display_name = field_validator("display_name", mode="before")(normalize_display_name)


class ProjectHomeTransfer(BaseModel):
    """One ordered, human-authorized change to a project's writable home."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    previous_home_space_id: str
    new_home_space_id: str
    source_released_by: AuthorizedHuman
    target_admitted_by: AuthorizedHuman

    _validate_uuid4 = field_validator(
        "project_id",
        "previous_home_space_id",
        "new_home_space_id",
    )(_canonical_uuid4)

    @model_validator(mode="after")
    def authority_matches_both_space_scoped_boundaries(self) -> ProjectHomeTransfer:
        if self.previous_home_space_id == self.new_home_space_id:
            raise ValueError("a project home transfer must change spaces")
        if self.source_released_by.space_id != self.previous_home_space_id:
            raise ValueError("the source-release actor must belong to the previous home space")
        if self.target_admitted_by.space_id != self.new_home_space_id:
            raise ValueError("the target-admission actor must belong to the new home space")
        return self


class GraphBranchMetadata(BaseModel):
    """Canonical identity and current head for one episode-owned graph branch."""

    model_config = ConfigDict(extra="forbid")

    schema_generation: Literal[1] = 1
    branch_id: str
    episode_id: str
    project_id: str = Field(min_length=1)
    kind: Literal["auto_research"] = "auto_research"
    base_head: GraphHeadRef
    head: GraphHeadRef
    created_at: datetime = Field(default_factory=utc_now)
    authorized_by: AuthorizedHuman

    _validate_uuid4 = field_validator("branch_id", "episode_id")(_canonical_uuid4)

    @model_validator(mode="after")
    def identity_and_heads_are_coherent(self) -> GraphBranchMetadata:
        if self.branch_id != self.episode_id:
            raise ValueError("an Auto-research graph branch must use its episode UUID")
        if self.base_head.target.kind != "main":
            raise ValueError("a graph branch base must name a main head")
        if self.head.target.kind != "branch" or self.head.target.branch_id != self.branch_id:
            raise ValueError("a graph branch head must name its exact branch")
        if self.head.revision < self.base_head.revision:
            raise ValueError("a graph branch head cannot precede its immutable main base")
        if (
            self.head.revision == self.base_head.revision
            and self.head.transition_id != self.base_head.transition_id
        ):
            raise ValueError(
                "a new graph branch head must preserve its exact main transition identity"
            )
        return self


class BranchMergeProvenance(BaseModel):
    """Strict source/rebase identity stamped on one main-target merge Patch."""

    model_config = ConfigDict(extra="forbid")

    schema_generation: Literal[1] = 1
    merge_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    branch_id: str
    episode_id: str
    branch_base_head: GraphHeadRef
    branch_head: GraphHeadRef
    rebased_main_head: GraphHeadRef
    merge_task_id: str = Field(min_length=1)

    _validate_uuid4 = field_validator("branch_id", "episode_id")(_canonical_uuid4)

    @model_validator(mode="after")
    def source_and_rebase_heads_are_coherent(self) -> BranchMergeProvenance:
        if self.branch_id != self.episode_id:
            raise ValueError("branch merge provenance must use the episode branch UUID")
        if self.branch_base_head.target.kind != "main":
            raise ValueError("branch merge provenance requires a main branch base")
        if (
            self.branch_head.target.kind != "branch"
            or self.branch_head.target.branch_id != self.branch_id
        ):
            raise ValueError("branch merge provenance names a different branch head")
        if self.branch_head.revision < self.branch_base_head.revision:
            raise ValueError("branch merge provenance has a head before its base")
        if self.rebased_main_head.target.kind != "main":
            raise ValueError("branch merge provenance must rebase onto main")
        if self.rebased_main_head.revision < self.branch_base_head.revision:
            raise ValueError("branch merge provenance cannot rebase before its branch base")
        return self


class BranchMergeReceipt(BaseModel):
    """Append-only acknowledgement that one branch head reached main exactly once."""

    model_config = ConfigDict(extra="forbid")

    schema_generation: Literal[1] = 1
    outcome: Literal["committed", "no_change"]
    provenance: BranchMergeProvenance
    result_main_head: GraphHeadRef
    authorized_by: AuthorizedHuman
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def committed_head_is_coherent(self) -> BranchMergeReceipt:
        if self.result_main_head.target.kind != "main":
            raise ValueError("a branch merge receipt must name a resulting main head")
        if self.outcome == "committed":
            if self.result_main_head.transition_id is None:
                raise ValueError("a committed branch merge requires its transition identity")
            if self.result_main_head.revision != self.provenance.rebased_main_head.revision + 1:
                raise ValueError("a committed branch merge must advance main exactly once")
        elif self.result_main_head != self.provenance.rebased_main_head:
            raise ValueError("a no-change branch merge must retain the exact rebased main head")
        return self


class GraphBranchSummary(BaseModel):
    """Strict episode/API projection; the browser never reconstructs branch state."""

    model_config = ConfigDict(extra="forbid")

    branch_id: str
    episode_id: str
    base_head: GraphHeadRef
    head: GraphHeadRef
    merge_eligible: bool
    merge_state: Literal["unmerged", "running", "merged", "needs_action", "failed"]
    latest_successful_merge: BranchMergeReceipt | None = None
    active_merge_task_id: str | None = Field(default=None, min_length=1)
    merge_diagnostic: str | None = None

    _validate_uuid4 = field_validator("branch_id", "episode_id")(_canonical_uuid4)

    @model_validator(mode="after")
    def merge_projection_is_coherent(self) -> GraphBranchSummary:
        if self.branch_id != self.episode_id:
            raise ValueError("branch summary identity does not match its episode")
        if self.base_head.target.kind != "main":
            raise ValueError("branch summary base must name main")
        if self.head.target.kind != "branch" or self.head.target.branch_id != self.branch_id:
            raise ValueError("branch summary head must name its exact branch")
        if (self.merge_state == "running") != (self.active_merge_task_id is not None):
            raise ValueError("only a running branch merge may name an active merge task")
        if self.merge_state == "merged" and self.latest_successful_merge is None:
            raise ValueError("a merged branch summary requires its successful receipt")
        if self.merge_eligible and self.merge_state == "running":
            raise ValueError("a branch with an active merge task is not merge eligible")
        return self


class Patch(BaseModel):
    schema_generation: Literal[1, 2] = 2
    revision: int = 0
    kind: Literal["seed", "refresh", "chat", "work", "experiment_loop", "approval", "identity"]
    # ``author`` retains its historical human/agent role semantics. Identity
    # revisions have no such author; their separate producer is RCP itself.
    author: Literal["agent", "human"] | None
    producer: Literal["agent", "human", "system"]
    created_at: datetime = Field(default_factory=utc_now)
    summary: str
    ops: list[GraphOperation]
    run_truth_scope: list[str] = Field(default_factory=list)
    repositories_read: list[str] = Field(default_factory=list)
    processed_cursors: dict[str, str] = Field(default_factory=dict)
    change_summary: list[str] = Field(default_factory=list)
    source_operation_id: str | None = None
    # One task may perform several in-turn effects. RCP stamps this separate
    # identity when an effect needs crash-safe canonical deduplication while
    # ``source_operation_id`` remains the direct authorized task.
    source_effect_id: str | None = Field(default=None, min_length=1)
    source_effect_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    # Which human authority action produced this patch, when its operations
    # alone cannot say. A direct Decision choice and an ordinary node edit are
    # both one `update_nodes` on one node, but they carry different authority
    # and are validated by different rules, so the producer names the action
    # rather than leaving the validator to infer it from shape.
    human_action: Literal["decision_choice"] | None = None
    # The same explicit action boundary for an elevated agent producer. RCP
    # supplies the profile; episode lineage remains in operational storage.
    agent_action: Literal["decision_choice"] | None = None
    admission: Literal["accepted", "rejected"] = "accepted"
    admission_messages: list[ValidationMessage] = Field(default_factory=list)
    # RCP stamps these after reading an experiment-loop deliverable. They are
    # persisted so canonical replay enforces the same control boundary.
    experiment_control_node_id: str | None = None
    experiment_decision_bundle: list[ExperimentDecisionPin] = Field(default_factory=list)
    project_identity: ProjectIdentity | None = None
    project_home_transfer: ProjectHomeTransfer | None = None
    authorized_by: AuthorizedHuman | None = None
    profile: Literal["ordinary", "orchestrator"] | None = None
    task_id: str | None = Field(default=None, min_length=1)
    episode_id: str | None = Field(default=None, min_length=1)
    branch_merge: BranchMergeProvenance | None = None
    transition: TransitionTrace | None = None

    @field_serializer("ops")
    def serialize_ops(
        self,
        operations: list[GraphOperation] | list[dict[str, Any]],
        info: SerializationInfo,
    ) -> list[dict[str, Any]]:
        return [
            operation.model_dump(mode=info.mode, exclude_unset=True)
            if isinstance(operation, BaseModel)
            else operation
            for operation in operations
        ]

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_producer(cls, value: Any) -> Any:
        """Reject live legacy lineage and fill producer on older lineage-free payloads."""

        if not isinstance(value, dict):
            return value
        if "campaign_id" in value:
            raise ValueError("campaign_id is not accepted; use episode_id")
        if "producer" in value or "author" not in value:
            return value
        migrated = dict(value)
        migrated["producer"] = migrated["author"]
        return migrated

    @model_validator(mode="after")
    def reject_legacy_proposal_intents_in_current_generation(self) -> Patch:
        if self.schema_generation == 1:
            return self
        for operation in self.ops:
            if operation.op != "create_proposals":
                continue
            for proposal in operation.proposals:
                if any(item.intent.startswith("legacy_") for item in proposal.ops):
                    raise ValueError(
                        "legacy Proposal operation shapes are accepted only through persisted "
                        "schema-generation 1 decoding"
                    )
        return self

    @model_validator(mode="after")
    def branch_merge_attribution_is_complete(self) -> Patch:
        provenance = self.branch_merge
        if provenance is None:
            return self
        if (
            self.kind != "work"
            or self.author != "agent"
            or self.producer != "agent"
            or self.profile != "orchestrator"
            or self.authorized_by is None
        ):
            raise ValueError("a branch merge must be an attributed orchestrator Work Patch")
        if self.task_id != provenance.merge_task_id:
            raise ValueError("branch merge provenance does not match its direct task")
        if self.episode_id != provenance.episode_id:
            raise ValueError("branch merge provenance does not match its episode")
        if self.transition is not None and self.transition.pre_head != provenance.rebased_main_head:
            raise ValueError("branch merge transition does not start from its rebased main head")
        return self


# The operation contract depends on the domain payloads above, while Patch and
# Proposal expose the operation unions. Importing after the models are declared
# keeps that dependency one-way at runtime and lets Pydantic resolve the two
# forward references explicitly.
from rcp.core.operations import GraphOperation, ProposalOperation  # noqa: E402
from rcp.core.transition_models import GraphHeadRef, TransitionTrace  # noqa: E402

Proposal.model_rebuild(_types_namespace={"ProposalOperation": ProposalOperation})
Patch.model_rebuild(
    _types_namespace={
        "GraphOperation": GraphOperation,
        "TransitionTrace": TransitionTrace,
        "GraphHeadRef": GraphHeadRef,
    }
)
GraphBranchMetadata.model_rebuild(_types_namespace={"GraphHeadRef": GraphHeadRef})
BranchMergeProvenance.model_rebuild(_types_namespace={"GraphHeadRef": GraphHeadRef})
BranchMergeReceipt.model_rebuild(_types_namespace={"GraphHeadRef": GraphHeadRef})
GraphBranchSummary.model_rebuild(_types_namespace={"GraphHeadRef": GraphHeadRef})
GraphState.model_rebuild(
    _types_namespace={"EvidenceAssessment": EvidenceAssessment, "Proposal": Proposal}
)
