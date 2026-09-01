"""Strict persisted graph-operation contracts and compatibility decoding."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from rcp.core.models import (
    EvidenceAssessment,
    OntologyState,
    ProjectNode,
    Proposal,
    Standing,
)


class _StrictPayload(BaseModel):
    """Strict typed payload shared by every graph-operation surface."""

    model_config = ConfigDict(extra="forbid", strict=True)


_DATETIME_ADAPTER = TypeAdapter(datetime)
_PROJECT_NODE_ADAPTER = TypeAdapter(ProjectNode)
_EVIDENCE_ASSESSMENT_ADAPTER = TypeAdapter(EvidenceAssessment)
_ONTOLOGY_STATE_ADAPTER = TypeAdapter(OntologyState)


def _wire_datetime(value: Any) -> Any:
    """Parse the string representation JSON uses without accepting numeric coercion."""

    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return _DATETIME_ADAPTER.validate_python(value)
    return value


def strict_project_node(value: Any) -> ProjectNode:
    """Strictly validate nested node payloads while retaining persisted JSON forms."""

    if isinstance(value, BaseModel):
        return _PROJECT_NODE_ADAPTER.validate_python(value, strict=True)
    if not isinstance(value, dict):
        return _PROJECT_NODE_ADAPTER.validate_python(value, strict=True)
    document = deepcopy(value)
    standing = document.get("standing")
    if isinstance(standing, str):
        document["standing"] = Standing(standing)
    _adapt_source_ref_datetimes(document.get("source_refs"))
    if document.get("type") == "experiment":
        attempts = document.get("attempts")
        if isinstance(attempts, list):
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                for field in ("started_at", "finished_at"):
                    if field in attempt:
                        attempt[field] = _wire_datetime(attempt[field])
                _adapt_source_ref_datetimes(attempt.get("source_refs"))
    return _PROJECT_NODE_ADAPTER.validate_python(document, strict=True)


def _adapt_source_ref_datetimes(value: Any) -> None:
    if not isinstance(value, list):
        return
    for source_ref in value:
        if isinstance(source_ref, dict) and "timestamp" in source_ref:
            source_ref["timestamp"] = _wire_datetime(source_ref["timestamp"])


class EvidenceEdgeCause(_StrictPayload):
    kind: Literal["evidence_edge"]
    ref_id: str


class DecisionCause(_StrictPayload):
    kind: Literal["decision"]
    ref_id: str


class ProposalResolutionCause(_StrictPayload):
    kind: Literal["proposal_resolution"]
    ref_id: str


class HumanEditCause(_StrictPayload):
    kind: Literal["human_edit"]


BeliefCause: TypeAlias = Annotated[
    EvidenceEdgeCause | DecisionCause | ProposalResolutionCause | HumanEditCause,
    Field(discriminator="kind"),
]


class NodeUpdate(_StrictPayload):
    id: str
    changes: dict[str, Any]
    cause: BeliefCause | None = None
    base_updated_rev: int | None = Field(default=None, ge=0)


class SupersedeNode(_StrictPayload):
    id: str
    superseded_by: str | None = None
    explanation: str = ""
    cause: BeliefCause | None = None


class NodeMerge(_StrictPayload):
    duplicate: str
    canonical: str
    explanation: str = ""
    cause: BeliefCause | None = None


class NewEdge(_StrictPayload):
    id: str | None = None
    source: str
    target: str
    relation: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    explanation: str = ""
    assessment: EvidenceAssessment | None = None

    @field_validator("assessment", mode="before")
    @classmethod
    def validate_assessment_strictly(cls, value: Any) -> Any:
        if value is None:
            return None
        return _EVIDENCE_ASSESSMENT_ADAPTER.validate_python(value, strict=True)


class NewAmbiguity(_StrictPayload):
    id: str
    question: str
    why_it_matters: str
    candidates: list[str] = Field(default_factory=list)
    related_node_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    status: Literal["open", "resolved", "dismissed"] = "open"


class AmbiguityResolution(_StrictPayload):
    id: str
    status: Literal["resolved", "dismissed"]


class ProposalResolution(_StrictPayload):
    id: str
    status: Literal["approved", "rejected", "withdrawn"]
    reason: str | None = None


class ProposalWithdrawal(_StrictPayload):
    id: str
    reason: str = ""


class NewGlossaryTerm(_StrictPayload):
    term: str
    plain_definition: str
    where_defined: str | None = None


class CoverageUpdate(_StrictPayload):
    repositories_seen: list[str] = Field(default_factory=list)
    repositories_never_seen: list[str] = Field(default_factory=list)
    sessions_read: list[str] = Field(default_factory=list)
    sessions_skipped: list[str] = Field(default_factory=list)
    earliest_timestamp: datetime | None = None
    note: str = "No seed has completed."

    @field_validator("earliest_timestamp", mode="before")
    @classmethod
    def accept_persisted_datetime_wire_form(cls, value: Any) -> Any:
        return _wire_datetime(value)


class RepositoryDescriptor(_StrictPayload):
    alias: str
    machine: str
    path: str


class CreateNodesOperation(_StrictPayload):
    op: Literal["create_nodes"]
    nodes: list[ProjectNode]

    @field_validator("nodes", mode="before")
    @classmethod
    def validate_nodes_strictly(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [strict_project_node(node) for node in value]


class UpdateNodesOperation(_StrictPayload):
    op: Literal["update_nodes"]
    nodes: list[NodeUpdate]


class CreateEdgesOperation(_StrictPayload):
    op: Literal["create_edges"]
    edges: list[NewEdge]


class RemoveEdgesOperation(_StrictPayload):
    op: Literal["remove_edges"]
    edge_ids: list[str]


class RemoveNodesOperation(_StrictPayload):
    op: Literal["remove_nodes"]
    node_ids: list[str]


class SupersedeNodesOperation(_StrictPayload):
    op: Literal["supersede_nodes"]
    nodes: list[SupersedeNode]


class MergeNodesOperation(_StrictPayload):
    op: Literal["merge_nodes"]
    merges: list[NodeMerge]


class CreateAmbiguitiesOperation(_StrictPayload):
    op: Literal["create_ambiguities"]
    ambiguities: list[NewAmbiguity]


class ResolveAmbiguitiesOperation(_StrictPayload):
    op: Literal["resolve_ambiguities"]
    resolutions: list[AmbiguityResolution]


class ProposalContentChangeOperation(_StrictPayload):
    op: Literal["update_nodes"]
    intent: Literal["content_change"]
    nodes: list[NodeUpdate]


class ProposalStatusChangeOperation(_StrictPayload):
    op: Literal["update_nodes"]
    intent: Literal["status_change"]
    nodes: list[NodeUpdate]


class ProposalRemovalOperation(_StrictPayload):
    op: Literal["remove_nodes"]
    intent: Literal["removal"]
    node_ids: list[str]


class ProposalSupersedeOperation(_StrictPayload):
    op: Literal["supersede_nodes"]
    intent: Literal["supersede"]
    nodes: list[SupersedeNode]


class ProposalMergeOperation(_StrictPayload):
    op: Literal["merge_nodes"]
    intent: Literal["merge"]
    merges: list[NodeMerge]


class ProposalProtectedRelationOperation(_StrictPayload):
    op: Literal["create_edges", "remove_edges"]
    intent: Literal["protected_relation_change"]
    edges: list[NewEdge] | None = None
    edge_ids: list[str] | None = None

    @model_validator(mode="after")
    def require_matching_payload(self) -> ProposalProtectedRelationOperation:
        if self.op == "create_edges" and (self.edges is None or self.edge_ids is not None):
            raise ValueError("create_edges requires edges and forbids edge_ids")
        if self.op == "remove_edges" and (self.edge_ids is None or self.edges is not None):
            raise ValueError("remove_edges requires edge_ids and forbids edges")
        return self


class LegacyProposalContentChangeOperation(ProposalContentChangeOperation):
    intent: Literal["legacy_content_change"]


class LegacyProposalStatusChangeOperation(ProposalStatusChangeOperation):
    intent: Literal["legacy_status_change"]


class LegacyProposalRemovalOperation(ProposalRemovalOperation):
    intent: Literal["legacy_removal"]


class LegacyProposalSupersedeOperation(ProposalSupersedeOperation):
    intent: Literal["legacy_supersede"]


class LegacyProposalMergeOperation(ProposalMergeOperation):
    intent: Literal["legacy_merge"]


class LegacyProposalProtectedRelationOperation(ProposalProtectedRelationOperation):
    intent: Literal["legacy_protected_relation_change"]


class LegacyProposalSetProjectTruthScopeOperation(_StrictPayload):
    op: Literal["set_project_truth_scope"]
    intent: Literal["legacy_project_truth_scope_change"]
    truth_scope: list[str]
    repository: RepositoryDescriptor | None = None


class LegacyProposalSetOntologyOperation(_StrictPayload):
    op: Literal["set_ontology"]
    intent: Literal["legacy_ontology_change"]
    ontology: OntologyState

    @field_validator("ontology", mode="before")
    @classmethod
    def validate_ontology_strictly(cls, value: Any) -> Any:
        return _ONTOLOGY_STATE_ADAPTER.validate_python(value, strict=True)


class LegacyProposalCreateNodesOperation(_StrictPayload):
    op: Literal["create_nodes"]
    intent: Literal["legacy_create_nodes"]
    nodes: list[ProjectNode]

    @field_validator("nodes", mode="before")
    @classmethod
    def validate_nodes_strictly(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [strict_project_node(node) for node in value]


class LegacyProposalCreateAmbiguitiesOperation(_StrictPayload):
    op: Literal["create_ambiguities"]
    intent: Literal["legacy_create_ambiguities"]
    ambiguities: list[NewAmbiguity]


class LegacyProposalResolveAmbiguitiesOperation(_StrictPayload):
    op: Literal["resolve_ambiguities"]
    intent: Literal["legacy_resolve_ambiguities"]
    resolutions: list[AmbiguityResolution]


class LegacyProposalUpsertGlossaryOperation(_StrictPayload):
    op: Literal["upsert_glossary"]
    intent: Literal["legacy_upsert_glossary"]
    terms: list[NewGlossaryTerm]


class LegacyProposalSetCoverageOperation(_StrictPayload):
    op: Literal["set_coverage"]
    intent: Literal["legacy_set_coverage"]
    coverage: CoverageUpdate


ProposalOperation: TypeAlias = Annotated[
    ProposalContentChangeOperation
    | ProposalStatusChangeOperation
    | ProposalRemovalOperation
    | ProposalSupersedeOperation
    | ProposalMergeOperation
    | ProposalProtectedRelationOperation
    | LegacyProposalContentChangeOperation
    | LegacyProposalStatusChangeOperation
    | LegacyProposalRemovalOperation
    | LegacyProposalSupersedeOperation
    | LegacyProposalMergeOperation
    | LegacyProposalProtectedRelationOperation
    | LegacyProposalSetProjectTruthScopeOperation
    | LegacyProposalSetOntologyOperation
    | LegacyProposalCreateNodesOperation
    | LegacyProposalCreateAmbiguitiesOperation
    | LegacyProposalResolveAmbiguitiesOperation
    | LegacyProposalUpsertGlossaryOperation
    | LegacyProposalSetCoverageOperation,
    Field(discriminator="intent"),
]


class CreateProposalsOperation(_StrictPayload):
    op: Literal["create_proposals"]
    proposals: list[Proposal]

    @model_validator(mode="before")
    @classmethod
    def identify_nested_proposal_errors(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not isinstance(value.get("proposals"), list):
            return value
        proposals: list[Proposal] = []
        for index, raw in enumerate(value["proposals"]):
            try:
                proposals.append(Proposal.model_validate(raw, strict=True))
            except ValidationError as exc:
                proposal_id = raw.get("id") if isinstance(raw, dict) else None
                error = exc.errors(include_url=False)[0]
                location = ".".join(str(part) for part in error["loc"])
                raise ValueError(
                    f"Proposal {proposal_id or f'at index {index}'!r} is malformed at "
                    f"{location or 'proposal'}: {error['msg']}"
                ) from exc
        return {**value, "proposals": proposals}


class ResolveProposalsOperation(_StrictPayload):
    op: Literal["resolve_proposals"]
    resolutions: list[ProposalResolution]


class WithdrawProposalsOperation(_StrictPayload):
    op: Literal["withdraw_proposals"]
    proposals: list[ProposalWithdrawal]


class UpsertGlossaryOperation(_StrictPayload):
    op: Literal["upsert_glossary"]
    terms: list[NewGlossaryTerm]


class SetCoverageOperation(_StrictPayload):
    op: Literal["set_coverage"]
    coverage: CoverageUpdate


class SetStandingOperation(_StrictPayload):
    op: Literal["set_standing"]
    node_id: str
    standing: Literal["asserted", "accepted", "contested"]


class SetProjectTruthScopeOperation(_StrictPayload):
    op: Literal["set_project_truth_scope"]
    truth_scope: list[str]
    repository: RepositoryDescriptor | None = None


class SetOntologyOperation(_StrictPayload):
    op: Literal["set_ontology"]
    ontology: OntologyState

    @field_validator("ontology", mode="before")
    @classmethod
    def validate_ontology_strictly(cls, value: Any) -> Any:
        return _ONTOLOGY_STATE_ADAPTER.validate_python(value, strict=True)


GraphOperation: TypeAlias = Annotated[
    CreateNodesOperation
    | UpdateNodesOperation
    | CreateEdgesOperation
    | RemoveEdgesOperation
    | RemoveNodesOperation
    | SupersedeNodesOperation
    | MergeNodesOperation
    | CreateAmbiguitiesOperation
    | ResolveAmbiguitiesOperation
    | CreateProposalsOperation
    | ResolveProposalsOperation
    | WithdrawProposalsOperation
    | UpsertGlossaryOperation
    | SetCoverageOperation
    | SetStandingOperation
    | SetProjectTruthScopeOperation
    | SetOntologyOperation,
    Field(discriminator="op"),
]

GRAPH_OPERATION_ADAPTER = TypeAdapter(GraphOperation)


def graph_operation_from_proposal(operation: ProposalOperation) -> GraphOperation:
    """Drop Proposal-only intent at the explicit semantic replay boundary."""

    document = operation.model_dump(mode="python", exclude_unset=True)
    document.pop("intent")
    return GRAPH_OPERATION_ADAPTER.validate_python(document)


def graph_operations_from_proposal(
    operations: list[ProposalOperation],
) -> list[GraphOperation]:
    return [graph_operation_from_proposal(operation) for operation in operations]


def adapt_persisted_patch_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a current in-memory document without mutating persisted bytes.

    Historical Proposals predate the explicit intent discriminator. Replay used
    their operation shape directly, so the adapter supplies only the current
    discriminator and leaves every semantic payload untouched.
    """

    adapted = deepcopy(document)
    if "campaign_id" in adapted and "episode_id" in adapted:
        raise ValueError("persisted Patch cannot contain both campaign_id and episode_id")
    if "campaign_id" in adapted:
        adapted["episode_id"] = adapted.pop("campaign_id")
    marker_missing = "schema_generation" not in adapted
    legacy_generation = marker_missing or adapted.get("schema_generation") == 1
    if marker_missing:
        adapted["schema_generation"] = 1
    # Older RCP releases deliberately retained rejected candidates, including
    # operations that never belonged to the graph vocabulary. Replay has always
    # skipped their semantics. The strict typed decoder therefore projects only
    # their rejection receipt and chronology, rather than letting an already-
    # rejected malformed payload halt later accepted history.
    if adapted.get("admission") == "rejected":
        adapted["ops"] = []
        return adapted
    operations = adapted.get("ops")
    if not isinstance(operations, list):
        return adapted
    if not legacy_generation:
        for operation in operations:
            if not isinstance(operation, dict) or operation.get("op") != "create_nodes":
                continue
            nodes = operation.get("nodes")
            if not isinstance(nodes, list):
                continue
            for node in nodes:
                if (
                    isinstance(node, dict)
                    and node.get("type") == "evidence"
                    and "legacy_strength" in node
                ):
                    raise ValueError(
                        "schema-generation 2 patches cannot create Evidence with "
                        "legacy_strength compatibility metadata"
                    )
    for operation in operations:
        if legacy_generation and adapted.get("kind") == "approval":
            _adapt_legacy_approval_operation(operation)
        _adapt_legacy_node_operation(operation, legacy_generation=legacy_generation)
        if legacy_generation:
            _adapt_legacy_ambiguity_operation(operation)
        if not isinstance(operation, dict) or operation.get("op") != "create_proposals":
            continue
        proposals = operation.get("proposals")
        if not isinstance(proposals, list):
            continue
        for proposal in proposals:
            _adapt_legacy_proposal_document(
                proposal,
                legacy_generation=legacy_generation,
            )
    return adapted


def _adapt_legacy_approval_operation(operation: Any) -> None:
    """Drop the Proposal-only intent once an old approval made the op semantic."""

    if not isinstance(operation, dict):
        return
    recognized = {
        ("update_nodes", "content_change"),
        ("update_nodes", "status_change"),
        ("remove_nodes", "removal"),
        ("supersede_nodes", "supersede"),
        ("merge_nodes", "merge"),
        ("create_edges", "protected_relation_change"),
        ("remove_edges", "protected_relation_change"),
    }
    if (operation.get("op"), operation.get("intent")) in recognized:
        operation.pop("intent")


def adapt_persisted_graph_state_document(document: dict[str, Any]) -> dict[str, Any]:
    """Adapt a materialized/cached graph snapshot without changing its source bytes."""

    adapted = deepcopy(document)
    nodes = adapted.get("nodes")
    if isinstance(nodes, dict):
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            if node.get("type") == "evidence":
                _adapt_legacy_evidence_record(node, assume_default_strength=True)
            elif node.get("type") == "experiment":
                _adapt_legacy_experiment_record(node)
    proposals = adapted.get("proposals")
    if isinstance(proposals, dict):
        for proposal in proposals.values():
            _adapt_legacy_proposal_document(proposal, legacy_generation=True)
    return adapted


def _adapt_legacy_proposal_document(proposal: Any, *, legacy_generation: bool) -> None:
    if not isinstance(proposal, dict):
        return
    nested = proposal.get("ops")
    if not isinstance(nested, list):
        return
    for proposal_operation in nested:
        _adapt_legacy_node_operation(proposal_operation, legacy_generation=legacy_generation)
        if (
            not legacy_generation
            or not isinstance(proposal_operation, dict)
            or "intent" in proposal_operation
        ):
            continue
        intent = _legacy_proposal_intent(proposal_operation)
        if intent is not None:
            proposal_operation["intent"] = intent


def _adapt_legacy_node_operation(operation: Any, *, legacy_generation: bool) -> None:
    if not isinstance(operation, dict):
        return
    if operation.get("op") == "create_nodes":
        nodes = operation.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if legacy_generation and isinstance(node, dict) and node.get("type") == "evidence":
                    _adapt_legacy_evidence_record(node, assume_default_strength=legacy_generation)
                if (
                    legacy_generation
                    and isinstance(node, dict)
                    and node.get("type") == "experiment"
                ):
                    _adapt_legacy_experiment_record(node)
    elif operation.get("op") == "update_nodes":
        updates = operation.get("nodes")
        if isinstance(updates, list):
            for update in updates:
                if not isinstance(update, dict):
                    continue
                changes = update.get("changes")
                if legacy_generation and isinstance(changes, dict) and "strength" in changes:
                    _adapt_legacy_evidence_record(changes, assume_default_strength=False)
                if legacy_generation and isinstance(changes, dict):
                    _adapt_legacy_experiment_record(changes)


def _adapt_legacy_ambiguity_operation(operation: dict[str, Any]) -> None:
    """Drop the derived revision older releases wrote into a persisted ambiguity.

    `raised_rev` is materialized from the revision applying the operation, so the
    stored value was always inert. The current operation payload forbids it, which
    made a Patch RCP itself wrote unreadable and halted replay at it.
    """

    if not isinstance(operation, dict) or operation.get("op") != "create_ambiguities":
        return
    ambiguities = operation.get("ambiguities")
    if not isinstance(ambiguities, list):
        return
    for ambiguity in ambiguities:
        if isinstance(ambiguity, dict):
            ambiguity.pop("raised_rev", None)


def _adapt_legacy_evidence_record(record: dict[str, Any], *, assume_default_strength: bool) -> None:
    strength = record.pop("strength", None)
    if strength is None and assume_default_strength and "role" not in record:
        strength = "preliminary"
    if strength is None:
        return
    record["role"] = "diagnostic" if strength == "diagnostic" else "result"
    record["legacy_strength"] = strength


def _adapt_legacy_experiment_record(record: dict[str, Any]) -> None:
    if record.get("status") == "blocked":
        record["status"] = "unspecified"
        if record.get("current_summary"):
            record["current_summary_stale"] = True
        if record.get("next_action"):
            record["next_action_stale"] = True


def _legacy_proposal_intent(operation: dict[str, Any]) -> str | None:
    name = operation.get("op")
    if name == "update_nodes":
        updates = operation.get("nodes")
        if isinstance(updates, list) and len(updates) == 1 and isinstance(updates[0], dict):
            changes = updates[0].get("changes")
            cause = updates[0].get("cause")
            if (
                isinstance(changes, dict)
                and set(changes) == {"status"}
                and isinstance(cause, dict)
                and cause.get("kind") == "evidence_edge"
            ):
                return "legacy_status_change"
        return "legacy_content_change"
    return {
        "create_nodes": "legacy_create_nodes",
        "create_ambiguities": "legacy_create_ambiguities",
        "resolve_ambiguities": "legacy_resolve_ambiguities",
        "upsert_glossary": "legacy_upsert_glossary",
        "set_coverage": "legacy_set_coverage",
        "remove_nodes": "legacy_removal",
        "supersede_nodes": "legacy_supersede",
        "merge_nodes": "legacy_merge",
        "create_edges": "legacy_protected_relation_change",
        "remove_edges": "legacy_protected_relation_change",
        "set_project_truth_scope": "legacy_project_truth_scope_change",
        "set_ontology": "legacy_ontology_change",
    }.get(name)


def operation_dict(operation: GraphOperation | ProposalOperation) -> dict[str, Any]:
    """Serialize one typed operation at a deliberate compatibility boundary."""

    return operation.model_dump(mode="python", exclude_unset=True)
