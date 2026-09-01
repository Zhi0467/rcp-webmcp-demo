from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from rcp.core.authority import HYPOTHESIS_PROPOSAL_FIELDS, AgentProfile
from rcp.core.models import (
    EvidenceAssessment,
    ExperimentAttempt,
    ExperimentAttemptDebug,
    ExperimentDecisionPin,
    GatedCard,
    Patch,
    SourceRef,
)
from rcp.core.operations import GraphOperation, operation_dict

_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_NODE_ID = rf"[a-z][a-z0-9]*(?:_[a-z0-9]+)*/{_SLUG}"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


_DATETIME_ADAPTER = TypeAdapter(datetime)


def _wire_datetime(value: Any) -> Any:
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return _DATETIME_ADAPTER.validate_python(value)
    return value


class AgentSourceRef(SourceRef):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("timestamp", mode="before")
    @classmethod
    def accept_json_datetime_wire_form(cls, value: Any) -> Any:
        return _wire_datetime(value)


class AgentExperimentDecisionPin(ExperimentDecisionPin):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentExperimentAttemptDebug(ExperimentAttemptDebug):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentExperimentAttempt(ExperimentAttempt):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision_bundle: list[AgentExperimentDecisionPin] = Field(default_factory=list)
    debug: AgentExperimentAttemptDebug | None = None
    source_refs: list[AgentSourceRef] = Field(default_factory=list)

    @field_validator("started_at", "finished_at", mode="before")
    @classmethod
    def accept_json_datetime_wire_form(cls, value: Any) -> Any:
        return _wire_datetime(value)


class AgentGatedCard(GatedCard):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentEvidenceAssessment(EvidenceAssessment):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentNode(_StrictModel):
    id: str = Field(pattern=rf"^{_NODE_ID}$")
    title: str
    extension_type: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    )
    extension_fields: dict[str, str | float | bool | list[str]] = Field(default_factory=dict)
    source_refs: list[AgentSourceRef] = Field(default_factory=list)


class AgentResearchQuestion(AgentNode):
    type: Literal["research_question"]
    question: str
    motivation: str = ""
    scope: str = ""
    status: Literal["open", "answered", "abandoned", "superseded"] = "open"


class AgentHypothesis(AgentNode):
    type: Literal["hypothesis"]
    statement: str
    rationale: str = ""
    predictions: list[str] = Field(default_factory=list)
    scope: str = ""
    status: Literal["proposed"] = "proposed"


class AgentDecision(AgentNode):
    type: Literal["decision"]
    question: str
    options: list[str] = Field(default_factory=list)
    selected_option: None = None
    rationale: str | None = None
    consequences: list[str] = Field(default_factory=list)
    status: Literal["open", "ready"] = "open"


class AgentExperiment(AgentNode):
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
        "abandoned",
        "superseded",
    ] = "proposed"
    attempts: list[AgentExperimentAttempt] = Field(default_factory=list)
    current_summary: str = ""
    next_action: str | None = None


class AgentEvidence(AgentNode):
    type: Literal["evidence"]
    observation: str
    interpretation: str = ""
    role: Literal["result", "diagnostic"] = "result"
    validity: Literal["valid", "qualified", "invalid", "superseded"] = "valid"
    origin: Literal[
        "internal_run", "external_publication", "external_instance", "analytic", "unknown"
    ]
    artifact_refs: list[str] = Field(default_factory=list)


class AgentBlocker(AgentNode):
    type: Literal["blocker"]
    description: str
    blocker_type: Literal[
        "scientific", "design", "data", "implementation", "infrastructure", "unknown"
    ] = "unknown"
    status: Literal["open", "resolved", "superseded"] = "open"
    resolution_condition: str = ""
    recommended_action: str | None = None


AgentProjectNode = Annotated[
    AgentResearchQuestion
    | AgentHypothesis
    | AgentDecision
    | AgentExperiment
    | AgentEvidence
    | AgentBlocker,
    Field(discriminator="type"),
]


class NewEdge(_StrictModel):
    id: str | None = None
    source: str
    target: str
    relation: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    explanation: str = ""
    assessment: AgentEvidenceAssessment | None = None


class EvidenceEdgeCause(_StrictModel):
    kind: Literal["evidence_edge"]
    ref_id: str


class NodeUpdate(_StrictModel):
    id: str
    changes: dict[str, Any]
    cause: EvidenceEdgeCause | None = None


class SupersedeNode(_StrictModel):
    id: str
    superseded_by: str | None = None
    explanation: str = ""
    cause: EvidenceEdgeCause | None = None


class NodeMerge(_StrictModel):
    duplicate: str
    canonical: str
    explanation: str = ""
    cause: EvidenceEdgeCause | None = None


class CreateNodesOperation(_StrictModel):
    op: Literal["create_nodes"]
    nodes: list[AgentProjectNode] = Field(min_length=1)


class UpdateNodesOperation(_StrictModel):
    op: Literal["update_nodes"]
    nodes: list[NodeUpdate] = Field(min_length=1)


class CreateEdgesOperation(_StrictModel):
    op: Literal["create_edges"]
    edges: list[NewEdge] = Field(min_length=1)


class RemoveEdgesOperation(_StrictModel):
    op: Literal["remove_edges"]
    edge_ids: list[str] = Field(min_length=1)


class RemoveNodesOperation(_StrictModel):
    op: Literal["remove_nodes"]
    node_ids: list[str] = Field(min_length=1)


class SupersedeNodesOperation(_StrictModel):
    op: Literal["supersede_nodes"]
    nodes: list[SupersedeNode] = Field(min_length=1)


class MergeNodesOperation(_StrictModel):
    op: Literal["merge_nodes"]
    merges: list[NodeMerge] = Field(min_length=1)


class ProposalNodeUpdate(NodeUpdate):
    cause: EvidenceEdgeCause

    @model_validator(mode="after")
    def validate_agent_authority_shape(self) -> ProposalNodeUpdate:
        fields = frozenset(self.changes)
        if fields != HYPOTHESIS_PROPOSAL_FIELDS:
            raise ValueError("An agent Proposal may change only Hypothesis status.")
        return self


class ProposalContentNodeUpdate(_StrictModel):
    id: str
    changes: dict[str, Any]

    @model_validator(mode="after")
    def validate_content_change(self) -> ProposalContentNodeUpdate:
        if not self.changes:
            raise ValueError("A content change must change at least one field.")
        return self


class ProposalUpdateNodesOperation(_StrictModel):
    op: Literal["update_nodes"]
    intent: Literal["status_change"]
    nodes: list[ProposalNodeUpdate] = Field(min_length=1, max_length=1)


class ProposalContentChangeOperation(_StrictModel):
    op: Literal["update_nodes"]
    intent: Literal["content_change"]
    nodes: list[ProposalContentNodeUpdate] = Field(min_length=1, max_length=1)


class ProposalRemovalOperation(_StrictModel):
    op: Literal["remove_nodes"]
    intent: Literal["removal"]
    node_ids: list[str] = Field(min_length=1, max_length=1)


class ProposalSupersedeNode(_StrictModel):
    id: str
    superseded_by: str
    explanation: str = ""


class ProposalSupersedeOperation(_StrictModel):
    op: Literal["supersede_nodes"]
    intent: Literal["supersede"]
    nodes: list[ProposalSupersedeNode] = Field(min_length=1, max_length=1)


class ProposalNodeMerge(_StrictModel):
    duplicate: str
    canonical: str
    explanation: str = ""


class ProposalMergeOperation(_StrictModel):
    op: Literal["merge_nodes"]
    intent: Literal["merge"]
    merges: list[ProposalNodeMerge] = Field(min_length=1, max_length=1)


class ProposalCreateProtectedRelationOperation(_StrictModel):
    op: Literal["create_edges"]
    intent: Literal["protected_relation_change"]
    edges: list[NewEdge] = Field(min_length=1, max_length=1)


class ProposalRemoveProtectedRelationOperation(_StrictModel):
    op: Literal["remove_edges"]
    intent: Literal["protected_relation_change"]
    edge_ids: list[str] = Field(min_length=1, max_length=1)


AgentProposalOperation = (
    ProposalUpdateNodesOperation
    | ProposalContentChangeOperation
    | ProposalRemovalOperation
    | ProposalSupersedeOperation
    | ProposalMergeOperation
    | ProposalCreateProtectedRelationOperation
    | ProposalRemoveProtectedRelationOperation
)


class AgentProposal(_StrictModel):
    id: str = Field(pattern=rf"^prop/{_SLUG}$")
    title: str
    card: AgentGatedCard
    ops: list[AgentProposalOperation] = Field(min_length=1, max_length=1)


class CreateProposalsOperation(_StrictModel):
    op: Literal["create_proposals"]
    proposals: list[AgentProposal] = Field(min_length=1)


class AgentProposalWithdrawal(_StrictModel):
    id: str = Field(pattern=rf"^prop/{_SLUG}$")
    reason: str = ""


class WithdrawProposalsOperation(_StrictModel):
    op: Literal["withdraw_proposals"]
    proposals: list[AgentProposalWithdrawal] = Field(min_length=1)


class SetStandingOperation(_StrictModel):
    op: Literal["set_standing"]
    node_id: str
    standing: Literal["asserted", "accepted", "contested"]


AgentOperation = Annotated[
    CreateNodesOperation
    | UpdateNodesOperation
    | CreateEdgesOperation
    | RemoveEdgesOperation
    | RemoveNodesOperation
    | SupersedeNodesOperation
    | MergeNodesOperation
    | CreateProposalsOperation
    | WithdrawProposalsOperation,
    Field(discriminator="op"),
]


class AgentPatch(_StrictModel):
    summary: str
    ops: list[AgentOperation]
    repositories_read: list[str] = Field(
        default_factory=list,
        description=(
            "Repositories this run read, each named by its manifest alias or by the "
            "path the task contract gave for it."
        ),
    )
    change_summary: list[str] = Field(default_factory=list)


OrchestratorAgentOperation = Annotated[
    CreateNodesOperation
    | UpdateNodesOperation
    | CreateEdgesOperation
    | RemoveEdgesOperation
    | RemoveNodesOperation
    | SupersedeNodesOperation
    | MergeNodesOperation
    | CreateProposalsOperation
    | WithdrawProposalsOperation
    | SetStandingOperation,
    Field(discriminator="op"),
]


class OrchestratorAgentPatch(_StrictModel):
    summary: str
    ops: list[OrchestratorAgentOperation]
    repositories_read: list[str] = Field(
        default_factory=list,
        description=(
            "Repositories this run read, each named by its manifest alias or by the "
            "path the task contract gave for it."
        ),
    )
    change_summary: list[str] = Field(default_factory=list)
    agent_action: Literal["decision_choice"] | None = None

    @model_validator(mode="after")
    def require_explicit_decision_action(self) -> OrchestratorAgentPatch:
        has_outcome = any(
            isinstance(operation, UpdateNodesOperation)
            and any(
                update.changes.get("status") == "decided"
                or update.changes.get("selected_option") is not None
                for update in operation.nodes
            )
            for operation in self.ops
        )
        if has_outcome and self.agent_action != "decision_choice":
            raise ValueError(
                "An orchestrator Decision outcome requires agent_action='decision_choice'."
            )
        if self.agent_action == "decision_choice" and not has_outcome:
            raise ValueError("agent_action='decision_choice' requires a Decision outcome update.")
        return self


def agent_output_schema(*, profile: AgentProfile = "ordinary") -> dict[str, object]:
    return _agent_patch_model(profile).model_json_schema()


def parse_agent_patch_json(
    value: str,
    *,
    profile: AgentProfile = "ordinary",
) -> AgentPatch | OrchestratorAgentPatch:
    """Parse one semantic deliverable while preserving actionable schema diagnostics."""

    try:
        return _agent_patch_model(profile).model_validate_json(value)
    except ValidationError as exc:
        raise _agent_patch_shape_error(exc) from exc


def validate_agent_patch_shape(
    patch: AgentPatch | OrchestratorAgentPatch | Patch,
    *,
    profile: AgentProfile | None = None,
) -> None:
    resolved_profile = _agent_patch_profile(patch, profile)
    model = _agent_patch_model(resolved_profile)
    value: AgentPatch | OrchestratorAgentPatch | dict[str, Any]
    if isinstance(patch, (AgentPatch, OrchestratorAgentPatch)):
        value = patch
    else:
        value = {
            "summary": patch.summary,
            "ops": _strip_rcp_bookkeeping(patch.ops),
            "repositories_read": patch.repositories_read,
            "change_summary": patch.change_summary,
        }
        if resolved_profile == "orchestrator":
            value["agent_action"] = patch.agent_action
    try:
        model.model_validate(value)
    except ValidationError as exc:
        raise _agent_patch_shape_error(exc) from exc


def _agent_patch_shape_error(exc: ValidationError) -> ValueError:
    details: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        detail = f"{location}: {error['msg']}" if location else error["msg"]
        if detail not in details:
            details.append(detail)
        if len(details) == 8:
            break
    suffix = "" if len(exc.errors()) <= len(details) else " Additional shape errors omitted."
    return ValueError(
        "Agent patch does not match the graph operation schema: " + "; ".join(details) + suffix
    )


def _repositories_read_aliases(
    declared: list[str],
    repository_paths: dict[str, str] | None,
) -> list[str]:
    """Name each repository the agent read by its manifest alias.

    A task contract shows repositories as paths, while run truth scope is a list
    of manifest aliases, so an honest declaration written either way must mean the
    same repository. A path at or under a registered root reads back to that root's
    alias; anything naming no registered repository is preserved so scope
    validation still reports it.
    """

    if not repository_paths:
        return list(declared)
    roots = sorted(
        ((PurePosixPath(path), alias) for alias, path in repository_paths.items()),
        key=lambda item: len(item[0].parts),
        reverse=True,
    )
    resolved = []
    for value in declared:
        if value in repository_paths:
            resolved.append(value)
            continue
        candidate = PurePosixPath(value)
        resolved.append(
            next(
                (alias for root, alias in roots if candidate.is_relative_to(root)),
                value,
            )
        )
    return list(dict.fromkeys(resolved))


def prepare_agent_patch(
    draft: AgentPatch | OrchestratorAgentPatch,
    *,
    kind: Literal["seed", "refresh", "work", "experiment_loop"],
    run_truth_scope: list[str],
    repository_paths: dict[str, str] | None = None,
    source_operation_id: str | None = None,
    source_effect_id: str | None = None,
    source_effect_sha256: str | None = None,
    profile: AgentProfile | None = None,
) -> Patch:
    """Wrap one semantic agent deliverable in RCP-owned canonical metadata."""

    resolved_profile = _agent_patch_profile(draft, profile)
    normalized = _agent_patch_model(resolved_profile).model_validate(
        draft.model_dump(mode="python", exclude_none=True, exclude_unset=True)
    )
    payload = normalized.model_dump(mode="python", exclude_none=True, exclude_unset=True)
    operations = payload["ops"]
    for operation in operations:
        if operation.get("op") != "create_proposals":
            continue
        for proposal in operation.get("proposals", []):
            proposal.update(
                {
                    "related_node_ids": [],
                    "related_edge_ids": [],
                    "related_config_keys": [],
                    "base_rev": 0,
                    "status": "pending",
                    "created_by": "agent",
                    "created_by_operation_id": source_operation_id,
                    "raised_rev": 0,
                    "resolved_rev": None,
                    "resolved_by": None,
                    "resolved_by_operation_id": None,
                    "resolution_reason": None,
                    "rejection_reason": None,
                }
            )
    return Patch(
        kind=kind,
        author="agent",
        summary=draft.summary,
        ops=operations,
        run_truth_scope=list(run_truth_scope),
        repositories_read=_repositories_read_aliases(
            list(draft.repositories_read), repository_paths
        ),
        change_summary=list(draft.change_summary),
        processed_cursors={},
        source_operation_id=source_operation_id,
        source_effect_id=source_effect_id,
        source_effect_sha256=source_effect_sha256,
        agent_action=(
            normalized.agent_action if isinstance(normalized, OrchestratorAgentPatch) else None
        ),
    )


def _agent_patch_model(
    profile: AgentProfile,
) -> type[AgentPatch] | type[OrchestratorAgentPatch]:
    return OrchestratorAgentPatch if profile == "orchestrator" else AgentPatch


def _agent_patch_profile(
    patch: AgentPatch | OrchestratorAgentPatch | Patch,
    profile: AgentProfile | None,
) -> AgentProfile:
    if profile is not None:
        return profile
    if isinstance(patch, OrchestratorAgentPatch):
        return "orchestrator"
    if isinstance(patch, Patch) and patch.profile == "orchestrator":
        return "orchestrator"
    return "ordinary"


def _strip_rcp_bookkeeping(operations: list[GraphOperation]) -> list[dict[str, Any]]:
    """Project typed core operations onto the deliberately narrower agent schema."""

    stripped: list[dict[str, Any]] = []
    for operation in operations:
        item = operation_dict(operation)
        name = item.get("op")
        if name == "create_nodes":
            item["nodes"] = [
                {
                    key: value
                    for key, value in node.items()
                    if key not in {"standing", "created_rev", "updated_rev"}
                }
                for node in item.get("nodes", [])
            ]
        elif name == "create_edges":
            item["edges"] = [
                {key: value for key, value in edge.items() if key not in {"layer", "created_rev"}}
                for edge in item.get("edges", [])
            ]
        elif name == "create_ambiguities":
            item["ambiguities"] = [
                {key: value for key, value in ambiguity.items() if key != "raised_rev"}
                for ambiguity in item.get("ambiguities", [])
            ]
        elif name == "create_proposals":
            item["proposals"] = [
                {
                    key: value
                    for key, value in proposal.items()
                    if key
                    not in {
                        "related_node_ids",
                        "related_edge_ids",
                        "related_config_keys",
                        "base_rev",
                        "status",
                        "created_by",
                        "created_by_operation_id",
                        "raised_rev",
                        "resolved_rev",
                        "resolved_by",
                        "resolved_by_operation_id",
                        "resolution_reason",
                        "rejection_reason",
                    }
                }
                for proposal in item.get("proposals", [])
            ]
        elif name == "upsert_glossary":
            item["terms"] = [
                {key: value for key, value in term.items() if key != "updated_rev"}
                for term in item.get("terms", [])
            ]
        stripped.append(item)
    return stripped
