from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter

from rcp.core.models import (
    Ambiguity,
    BeliefTransition,
    CoverageBoundary,
    Edge,
    GlossaryTerm,
    GraphState,
    Hypothesis,
    Patch,
    ProjectNode,
    Proposal,
    ReplayFailure,
    Standing,
)
from rcp.core.ontology import custom_relation, edge_layer
from rcp.core.operations import (
    BeliefCause,
    CreateAmbiguitiesOperation,
    CreateEdgesOperation,
    CreateNodesOperation,
    CreateProposalsOperation,
    GraphOperation,
    MergeNodesOperation,
    RemoveEdgesOperation,
    RemoveNodesOperation,
    ResolveAmbiguitiesOperation,
    ResolveProposalsOperation,
    SetCoverageOperation,
    SetOntologyOperation,
    SetProjectTruthScopeOperation,
    SetStandingOperation,
    SupersedeNodesOperation,
    UpdateNodesOperation,
    UpsertGlossaryOperation,
    WithdrawProposalsOperation,
    strict_project_node,
)
from rcp.core.validation import (
    IMMUTABLE_NODE_UPDATE_FIELDS,
    ValidationReport,
    proposal_dependencies,
    validate_patch,
)

NODE_ADAPTER = TypeAdapter(ProjectNode)
AcceptedPatchObserver = Callable[[GraphState, Patch, GraphState], None]


@dataclass
class MaterializationResult:
    state: GraphState
    reports: dict[int, ValidationReport] = field(default_factory=dict)
    repository_descriptors: list[dict[str, str]] = field(default_factory=list)
    processed_cursors: dict[str, str] = field(default_factory=dict)
    patches: list[Patch] = field(default_factory=list)


def materialize_patches(
    patches: Iterable[Patch],
    initial_truth_scope: Iterable[str],
    repository_aliases: Iterable[str] | None = None,
    machine_aliases: Iterable[str] | None = None,
    default_run_truth_scope: Iterable[str] | None = None,
    state_repository: str | None = None,
    accepted_patch_observer: AcceptedPatchObserver | None = None,
) -> MaterializationResult:
    """Replay patches, optionally observing successful applications through a read-only callback."""

    replayed_patches = list(patches)
    initial_scope = list(initial_truth_scope)
    state = GraphState(project_truth_scope=initial_scope)
    state.coverage = state.coverage.model_copy(
        update={"repositories_never_seen": sorted(initial_scope)}
    )
    reports: dict[int, ValidationReport] = {}
    descriptors: list[dict[str, str]] = []
    processed_cursors: dict[str, str] = {}

    for patch in replayed_patches:
        if patch.admission == "rejected":
            report = ValidationReport()
            report.messages.extend(patch.admission_messages)
            reports[patch.revision] = report
            state = state.model_copy(
                update={
                    "revision": max(state.revision, patch.revision),
                    "validation_messages": [
                        *state.validation_messages,
                        *patch.admission_messages,
                    ],
                }
            )
            continue

        report = validate_patch(
            state,
            patch,
            state.project_truth_scope,
            repository_aliases=repository_aliases,
            machine_aliases=machine_aliases,
            default_run_truth_scope=default_run_truth_scope,
            state_repository=state_repository,
            mode="replay",
        )
        report.messages.extend(patch.admission_messages)
        reports[patch.revision] = report
        if report.rejected:
            failure = next(item for item in report.messages if item.level == "reject")
            state = state.model_copy(
                update={
                    "replay_status": "degraded",
                    "replay_failure": ReplayFailure(
                        revision=patch.revision,
                        created_at=patch.created_at,
                        code=failure.code,
                        message=failure.message,
                    ),
                }
            )
            break
        previous_state = state
        candidate = _fork_state(previous_state)
        candidate_descriptors: list[dict[str, str]] = []
        try:
            _apply_patch(candidate, patch, candidate_descriptors)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            report.reject(
                "malformed-operation",
                f"Patch operations could not be applied atomically: {exc}.",
                patch.revision,
            )
            state = state.model_copy(
                update={
                    "replay_status": "degraded",
                    "replay_failure": ReplayFailure(
                        revision=patch.revision,
                        created_at=patch.created_at,
                        code="malformed-operation",
                        message=f"Patch operations could not be applied atomically: {exc}.",
                    ),
                }
            )
            break
        state = candidate
        descriptors.extend(candidate_descriptors)
        processed_cursors.update(patch.processed_cursors)
        state = state.model_copy(
            update={
                "validation_messages": [
                    *state.validation_messages,
                    *patch.admission_messages,
                ]
            }
        )
        if accepted_patch_observer is not None:
            accepted_patch_observer(previous_state, patch, state)

    return MaterializationResult(
        state=state,
        reports=reports,
        repository_descriptors=descriptors,
        processed_cursors=processed_cursors,
        patches=replayed_patches,
    )


def apply_valid_patch(state: GraphState, patch: Patch) -> GraphState:
    updated = _fork_state(state)
    _apply_patch(updated, patch, [])
    return updated


def apply_valid_operation(
    state: GraphState,
    patch: Patch,
    operation: GraphOperation,
) -> GraphState:
    """Stage one already-validated operation without advancing the graph revision."""

    updated = _fork_state(state)
    _apply_patch(updated, patch.model_copy(update={"ops": [operation]}), [])
    updated.revision = state.revision
    return updated


def apply_transition_generated_operation(
    state: GraphState,
    patch: Patch,
    operation: GraphOperation,
) -> GraphState:
    """Stage one manager-owned system-field action during transition closure."""

    updated = _fork_state(state)
    _apply_patch(
        updated,
        patch.model_copy(update={"ops": [operation]}),
        [],
        system_field_operation_indexes={0},
    )
    updated.revision = state.revision
    return updated


def prepare_patch_bookkeeping(state: GraphState, patch: Patch) -> Patch:
    """Replace RCP-owned Proposal metadata using the graph being appended to."""

    operations: list[GraphOperation] = []
    for operation in patch.ops:
        if not isinstance(operation, CreateProposalsOperation):
            operations.append(operation)
            continue
        proposals: list[Proposal] = []
        for proposal in operation.proposals:
            related_node_ids, related_edge_ids, related_config_keys = proposal_dependencies(
                state, proposal.ops
            )
            proposals.append(
                proposal.model_copy(
                    update={
                        "related_node_ids": related_node_ids,
                        "related_edge_ids": related_edge_ids,
                        "related_config_keys": related_config_keys,
                        "base_rev": state.revision,
                        "status": "pending",
                        "created_by": "human" if patch.author == "human" else "agent",
                        "created_by_operation_id": patch.source_operation_id,
                        "raised_rev": 0,
                        "resolved_rev": None,
                        "resolved_by": None,
                        "resolved_by_operation_id": None,
                        "resolution_reason": None,
                        "rejection_reason": None,
                    }
                )
            )
        operations.append(operation.model_copy(update={"proposals": proposals}))
    return patch.model_copy(update={"ops": operations})


def finalize_patch_bookkeeping(patch: Patch, staged_state: GraphState) -> Patch:
    """Persist Proposal dependencies exactly as staged operations observed them."""

    operations: list[GraphOperation] = []
    for operation in patch.ops:
        if not isinstance(operation, CreateProposalsOperation):
            operations.append(operation)
            continue
        proposals: list[Proposal] = []
        for proposal in operation.proposals:
            staged = staged_state.proposals.get(proposal.id)
            if staged is not None:
                proposal = proposal.model_copy(
                    update={
                        "related_node_ids": list(staged.related_node_ids),
                        "related_edge_ids": list(staged.related_edge_ids),
                        "related_config_keys": list(staged.related_config_keys),
                    }
                )
            proposals.append(proposal)
        operations.append(operation.model_copy(update={"proposals": proposals}))
    return patch.model_copy(update={"ops": operations})


def _fork_state(state: GraphState) -> GraphState:
    """Fork a state so a failed apply cannot touch the caller's copy.

    Only the mutable containers are copied; the nodes, edges, proposals,
    ambiguities, glossary terms, and coverage inside them are shared. That is
    safe because ``_apply_patch`` never mutates one of those objects in place —
    every change replaces a container slot or the whole attribute — so a patch
    that raises part-way leaves the caller's containers untouched.

    Deep-copying instead made replay quadratic in graph size: it dominated
    materialization at 98% of total time, and a 800-patch log took 15s to open.
    """
    return state.model_copy(
        update={
            "nodes": dict(state.nodes),
            "edges": dict(state.edges),
            "proposals": dict(state.proposals),
            "ambiguities": dict(state.ambiguities),
            "glossary": dict(state.glossary),
            "config_revisions": dict(state.config_revisions),
            "project_truth_scope": list(state.project_truth_scope),
            "validation_messages": list(state.validation_messages),
            "belief_transitions": list(state.belief_transitions),
        }
    )


def _apply_patch(
    state: GraphState,
    patch: Patch,
    repository_descriptors: list[dict[str, str]],
    *,
    system_field_operation_indexes: set[int] | None = None,
) -> None:
    revision = patch.revision
    created_edge_ids: list[str] = []
    generated_operation_indexes = set(system_field_operation_indexes or ())
    generated_operation_indexes.update(
        {item.operation_index for item in patch.transition.generated_actions}
        if patch.transition is not None
        else set()
    )
    for operation_index, op in enumerate(patch.ops):
        if isinstance(op, CreateNodesOperation):
            for raw in op.nodes:
                data = raw.model_dump(mode="python", exclude_unset=True)
                data["created_rev"] = revision
                data["updated_rev"] = revision
                data["standing"] = data.get("standing", "asserted")
                node = NODE_ADAPTER.validate_python(data)
                state.nodes[node.id] = node
        elif isinstance(op, UpdateNodesOperation):
            for update in op.nodes:
                node = state.nodes[update.id]
                changes = update.changes
                immutable_fields = IMMUTABLE_NODE_UPDATE_FIELDS
                if patch.schema_generation == 1:
                    immutable_fields = immutable_fields - {
                        "legacy_strength",
                        "current_summary_stale",
                        "next_action_stale",
                    }
                elif operation_index in generated_operation_indexes:
                    immutable_fields = immutable_fields - {
                        "current_summary_stale",
                        "next_action_stale",
                    }
                immutable = sorted(set(changes) & immutable_fields)
                if immutable:
                    raise ValueError(
                        f"node updates cannot change system fields: {', '.join(immutable)}"
                    )
                data = node.model_dump(mode="python")
                data.update(changes)
                data["updated_rev"] = revision
                data["standing"] = node.standing if patch.kind == "approval" else "asserted"
                updated = (
                    NODE_ADAPTER.validate_python(data)
                    if patch.schema_generation == 1
                    else strict_project_node(data)
                )
                _record_belief_transition(
                    state,
                    node,
                    updated,
                    revision,
                    update.cause,
                )
                state.nodes[node.id] = updated
        elif isinstance(op, CreateEdgesOperation):
            for raw in op.edges:
                data = raw.model_dump(mode="python", exclude_unset=True)
                data.setdefault("id", f"{data['source']}::{data['relation']}::{data['target']}")
                if relation := custom_relation(state.ontology, raw.relation):
                    data["layer"] = relation.layer
                data["created_rev"] = revision
                edge = Edge.model_validate(data)
                derived = edge_layer(state, edge.source, edge.target, edge.layer)
                if derived != edge.layer:
                    edge = edge.model_copy(update={"layer": derived})
                state.edges[edge.id] = edge
                created_edge_ids.append(edge.id)
        elif isinstance(op, RemoveEdgesOperation):
            for edge_id in op.edge_ids:
                state.edges.pop(edge_id, None)
        elif isinstance(op, RemoveNodesOperation):
            node_ids = set(op.node_ids)
            state.nodes = {
                node_id: node for node_id, node in state.nodes.items() if node_id not in node_ids
            }
            state.edges = {
                edge_id: edge
                for edge_id, edge in state.edges.items()
                if edge.source not in node_ids and edge.target not in node_ids
            }
        elif isinstance(op, SupersedeNodesOperation):
            for item in op.nodes:
                previous = state.nodes[item.id]
                _set_node_status(
                    state,
                    item.id,
                    "superseded",
                    revision,
                    preserve_standing=patch.kind == "approval",
                )
                _record_belief_transition(
                    state,
                    previous,
                    state.nodes[item.id],
                    revision,
                    item.cause,
                )
                target = item.superseded_by
                if target:
                    edge = Edge(
                        id=f"{item.id}::supersedes::{target}",
                        source=item.id,
                        target=target,
                        relation="supersedes",
                        explanation=item.explanation,
                        created_rev=revision,
                    )
                    state.edges[edge.id] = edge
        elif isinstance(op, MergeNodesOperation):
            for item in op.merges:
                duplicate = item.duplicate
                canonical = item.canonical
                previous = state.nodes[duplicate]
                _set_node_status(
                    state,
                    duplicate,
                    "superseded",
                    revision,
                    preserve_standing=patch.kind == "approval",
                )
                _record_belief_transition(
                    state,
                    previous,
                    state.nodes[duplicate],
                    revision,
                    item.cause,
                )
                edge = Edge(
                    id=f"{duplicate}::duplicate_of::{canonical}",
                    source=duplicate,
                    target=canonical,
                    relation="duplicate_of",
                    explanation=item.explanation,
                    created_rev=revision,
                )
                state.edges[edge.id] = edge
        elif isinstance(op, CreateAmbiguitiesOperation):
            for raw in op.ambiguities:
                data = raw.model_dump(mode="python", exclude_unset=True)
                data["raised_rev"] = revision
                ambiguity = Ambiguity.model_validate(data)
                state.ambiguities[ambiguity.id] = ambiguity
        elif isinstance(op, ResolveAmbiguitiesOperation):
            for resolution in op.resolutions:
                ambiguity = state.ambiguities[resolution.id]
                state.ambiguities[ambiguity.id] = ambiguity.model_copy(
                    update={"status": resolution.status}
                )
        elif isinstance(op, CreateProposalsOperation):
            for raw in op.proposals:
                related_node_ids, related_edge_ids, related_config_keys = proposal_dependencies(
                    state, raw.ops
                )
                proposal = raw.model_copy(
                    update={
                        "base_rev": state.revision,
                        "related_node_ids": related_node_ids,
                        "related_edge_ids": related_edge_ids,
                        "related_config_keys": related_config_keys,
                        "created_by": raw.created_by
                        if "created_by" in raw.model_fields_set
                        else ("human" if patch.author == "human" else "agent"),
                        "created_by_operation_id": raw.created_by_operation_id
                        if "created_by_operation_id" in raw.model_fields_set
                        else patch.source_operation_id,
                        "raised_rev": revision,
                    }
                )
                state.proposals[proposal.id] = proposal
        elif isinstance(op, ResolveProposalsOperation):
            for resolution in op.resolutions:
                proposal = state.proposals[resolution.id]
                state.proposals[proposal.id] = proposal.model_copy(
                    update={
                        "status": resolution.status,
                        "resolved_rev": revision,
                        "resolved_by": "human" if patch.author == "human" else "agent",
                        "resolved_by_operation_id": patch.source_operation_id,
                        "resolution_reason": resolution.reason,
                        "rejection_reason": resolution.reason,
                    }
                )
        elif isinstance(op, WithdrawProposalsOperation):
            for withdrawal in op.proposals:
                proposal = state.proposals[withdrawal.id]
                state.proposals[proposal.id] = proposal.model_copy(
                    update={
                        "status": "withdrawn",
                        "resolved_rev": revision,
                        "resolved_by": "agent",
                        "resolved_by_operation_id": patch.source_operation_id,
                        "resolution_reason": withdrawal.reason,
                    }
                )
        elif isinstance(op, UpsertGlossaryOperation):
            for raw in op.terms:
                data = raw.model_dump(mode="python", exclude_unset=True)
                data["updated_rev"] = revision
                term = GlossaryTerm.model_validate(data)
                state.glossary[term.term] = term
        elif isinstance(op, SetCoverageOperation):
            previous = state.coverage
            data = previous.model_dump(mode="python")
            data.update(op.coverage.model_dump(mode="python", exclude_unset=True))
            data["repositories_seen"] = sorted(set(data.get("repositories_seen", [])))
            data["repositories_never_seen"] = sorted(set(data.get("repositories_never_seen", [])))
            data["sessions_read"] = sorted(set(data.get("sessions_read", [])))
            data["sessions_skipped"] = sorted(set(data.get("sessions_skipped", [])))
            state.coverage = CoverageBoundary.model_validate(data)
        elif isinstance(op, SetStandingOperation):
            node = state.nodes[op.node_id]
            state.nodes[node.id] = node.model_copy(
                update={"standing": Standing(op.standing), "updated_rev": revision}
            )
        elif isinstance(op, SetProjectTruthScopeOperation):
            new_scope = set(op.truth_scope)
            state.project_truth_scope = sorted(new_scope)
            state.config_revisions["project_truth_scope"] = revision
            seen = set(state.coverage.repositories_seen)
            never_seen = set(state.coverage.repositories_never_seen)
            never_seen.update(new_scope - seen)
            never_seen.intersection_update(new_scope)
            state.coverage = state.coverage.model_copy(
                update={"repositories_never_seen": sorted(never_seen)}
            )
            if op.repository is not None:
                repository_descriptors.append(
                    op.repository.model_dump(mode="python", exclude_unset=True)
                )
        elif isinstance(op, SetOntologyOperation):
            state.ontology = op.ontology
            state.config_revisions["ontology"] = revision

    # A legal edge may forward-reference a node created later in this patch.
    # Its first pass necessarily uses the relation's declared fallback layer;
    # derive it once more from the completed staged graph without reordering ops.
    for edge_id in dict.fromkeys(created_edge_ids):
        edge = state.edges.get(edge_id)
        if edge is None:
            continue
        derived = edge_layer(state, edge.source, edge.target, edge.layer)
        if derived != edge.layer:
            state.edges[edge_id] = edge.model_copy(update={"layer": derived})

    state.revision = max(state.revision, revision)
    if patch.kind in {"seed", "refresh"}:
        state.last_refresh_at = patch.created_at


def _set_node_status(
    state: GraphState,
    node_id: str,
    status: str,
    revision: int,
    *,
    preserve_standing: bool,
) -> None:
    node = state.nodes[node_id]
    data: dict[str, Any] = node.model_dump(mode="python")
    data["status"] = status
    data["updated_rev"] = revision
    data["standing"] = node.standing if preserve_standing else "asserted"
    state.nodes[node_id] = NODE_ADAPTER.validate_python(data)


def _record_belief_transition(
    state: GraphState,
    previous: ProjectNode,
    updated: ProjectNode,
    revision: int,
    cause: BeliefCause | None,
) -> None:
    if (
        not isinstance(previous, Hypothesis)
        or not isinstance(updated, Hypothesis)
        or previous.status == updated.status
        or cause is None
    ):
        return
    state.belief_transitions.append(
        BeliefTransition(
            hypothesis_id=previous.id,
            from_status=previous.status,
            to_status=updated.status,
            revision=revision,
            cause=cause.model_dump(mode="python", exclude_unset=True),
        )
    )
