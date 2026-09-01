"""One rule function per operation name.

Each ``validate_*`` checks a single operation and returns the oldest source
reference it cited (or ``None``); each ``depends_*`` reports the existing graph
and project-config objects that operation would touch. The registry pairs them
up — see :mod:`rcp.core.validation.registry`.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from rcp.core.authority import DECIDE_DECISION, QUEUE_DECISION, permits
from rcp.core.models import (
    ACTIVE_EXPERIMENT_ATTEMPT_STATUSES,
    RELATION_SPEC,
    Decision,
    Edge,
    Evidence,
    Experiment,
    GraphState,
    Hypothesis,
    Standing,
)
from rcp.core.ontology import (
    custom_relation,
    edge_matches_relation,
    validate_ontology_change,
    validate_ontology_structure,
)
from rcp.core.operations import (
    BeliefCause,
    CreateAmbiguitiesOperation,
    CreateEdgesOperation,
    CreateNodesOperation,
    CreateProposalsOperation,
    DecisionCause,
    EvidenceEdgeCause,
    HumanEditCause,
    MergeNodesOperation,
    ProposalContentChangeOperation,
    ProposalMergeOperation,
    ProposalProtectedRelationOperation,
    ProposalRemovalOperation,
    ProposalResolutionCause,
    ProposalStatusChangeOperation,
    ProposalSupersedeOperation,
    RemoveEdgesOperation,
    RemoveNodesOperation,
    ResolveAmbiguitiesOperation,
    ResolveProposalsOperation,
    SetOntologyOperation,
    SetProjectTruthScopeOperation,
    SetStandingOperation,
    SupersedeNodesOperation,
    UpdateNodesOperation,
    WithdrawProposalsOperation,
    strict_project_node,
)
from rcp.core.validation.constants import (
    IMMUTABLE_NODE_UPDATE_FIELDS,
    LEGACY_COMPATIBILITY_UPDATE_FIELDS,
    NODE_ADAPTER,
)
from rcp.core.validation.context import OpContext
from rcp.core.validation.nodes import (
    older,
    oldest_source_ref,
    requires_proposal,
    validate_extension_update,
    validate_new_node,
    validate_new_node_authoring,
    validate_updated_node_authoring,
)
from rcp.core.validation.proposals import decision_transition_error, validate_proposal

EVIDENCE_HYPOTHESIS_RELATIONS = frozenset(
    {"supports", "weakens", "refutes", "inconclusive", "contradicts"}
)


def validate_create_nodes(op: CreateNodesOperation, ctx: OpContext) -> Any:
    oldest = None
    for node in op.nodes:
        node_id = node.id
        if ctx.mode == "admission" and node_id in ctx.initial_state.nodes:
            ctx.report.reject(
                "initial-node-id-replacement",
                f"Node {node_id!r} existed before this Patch and cannot be recreated; use "
                "update_nodes or a protected Proposal.",
                ctx.revision,
                related_node_ids=[node_id],
            )
        raw = node.model_dump(mode="python", exclude_unset=True)
        validate_new_node(ctx.state, ctx.patch, raw, ctx.report)
        oldest = older(oldest, oldest_source_ref(raw, ctx.patch, ctx.report))
    return oldest


def author_create_nodes(op: CreateNodesOperation, ctx: OpContext) -> Any:
    for node in op.nodes:
        if isinstance(node, Experiment) and node.status == "unspecified":
            ctx.report.reject(
                "live-legacy-experiment-phase",
                f"New Experiment {node.id!r} cannot author compatibility-only phase 'unspecified'.",
                ctx.revision,
                related_node_ids=[node.id],
            )
        if isinstance(node, Evidence) and "legacy_strength" in node.model_fields_set:
            ctx.report.reject(
                "live-legacy-evidence-strength",
                f"New Evidence {node.id!r} cannot set compatibility-only legacy_strength.",
                ctx.revision,
                related_node_ids=[node.id],
            )
        raw = node.model_dump(mode="python", exclude_unset=True)
        validate_new_node_authoring(ctx.state, ctx.patch, raw, ctx.report)
    return None


def validate_update_nodes(op: UpdateNodesOperation, ctx: OpContext) -> Any:
    oldest = None
    for update in op.nodes:
        node_id = update.id
        node = ctx.state.nodes.get(node_id)
        if node is None:
            ctx.report.reject(
                "unknown-node", f"Cannot update missing node {node_id!r}.", ctx.revision
            )
            continue
        changes = update.changes
        live_legacy_phase = (
            ctx.mode == "admission"
            and isinstance(node, Experiment)
            and changes.get("status") == "unspecified"
        )
        if live_legacy_phase:
            ctx.report.reject(
                "live-legacy-experiment-phase",
                f"Update to Experiment {node.id!r} cannot author compatibility-only phase "
                "'unspecified'.",
                ctx.revision,
                related_node_ids=[node.id],
            )
        live_legacy_strength = (
            ctx.mode == "admission" and isinstance(node, Evidence) and "legacy_strength" in changes
        )
        if live_legacy_strength:
            ctx.report.reject(
                "live-legacy-evidence-strength",
                f"Update to Evidence {node.id!r} cannot set compatibility-only legacy_strength.",
                ctx.revision,
                related_node_ids=[node.id],
            )
        immutable_fields = set(changes) & IMMUTABLE_NODE_UPDATE_FIELDS
        if ctx.mode == "replay" and ctx.patch.schema_generation == 1:
            immutable_fields -= LEGACY_COMPATIBILITY_UPDATE_FIELDS
        elif live_legacy_strength:
            immutable_fields.discard("legacy_strength")
        immutable = sorted(immutable_fields)
        if immutable:
            ctx.report.reject(
                "immutable-node-field",
                f"Update to {node_id} cannot change system fields: {', '.join(immutable)}.",
                ctx.revision,
            )
            continue
        if live_legacy_strength or live_legacy_phase:
            continue
        candidate = node.model_dump(mode="python")
        candidate.update(changes)
        try:
            if ctx.mode == "replay" and ctx.patch.schema_generation == 1:
                NODE_ADAPTER.validate_python(candidate)
            else:
                strict_project_node(candidate)
        except ValidationError as exc:
            ctx.report.reject(
                "invalid-node-update",
                f"Update to {node_id} is invalid: {exc.errors()[0]['msg']}.",
                ctx.revision,
            )
        validate_extension_update(
            ctx.state,
            ctx.patch,
            node,
            changes,
            ctx.report,
            authoring=False,
        )
        is_control_update = (
            ctx.patch.kind == "experiment_loop"
            and node_id == ctx.experiment_control_node_id
            and set(changes) <= {"attempts", "status"}
        )
        if not is_control_update and requires_proposal(node, changes):
            if isinstance(node, Decision):
                if ctx.mode == "admission" and not permits(ctx.patch, DECIDE_DECISION):
                    message = (
                        f"Update to {node_id} uses decide_decision; only a human may write "
                        "selected_option or status decided."
                        if ctx.patch.profile != "orchestrator"
                        else f"Update to {node_id} is not permitted to use {DECIDE_DECISION}."
                    )
                    ctx.report.reject(
                        "decision-action-refused",
                        message,
                        ctx.revision,
                        related_node_ids=[node.id],
                    )
            elif ctx.patch.kind != "approval":
                ctx.report.reject(
                    "gated-transition",
                    f"Update to {node_id} requires a Proposal and human approval.",
                    ctx.revision,
                )
        elif (
            isinstance(node, Decision)
            and ctx.mode == "admission"
            and changes.get("status") in {"open", "ready", "revisit"}
            and not permits(ctx.patch, QUEUE_DECISION)
        ):
            ctx.report.reject(
                "decision-action-refused",
                f"Update to {node_id} is not permitted to use {QUEUE_DECISION}.",
                ctx.revision,
                related_node_ids=[node.id],
            )
        if (
            ctx.mode == "admission"
            and isinstance(node, Decision)
            and (error := decision_transition_error(node, changes))
        ):
            ctx.report.reject(
                "incoherent-decision-transition",
                error,
                ctx.revision,
                related_node_ids=[node.id],
            )
        oldest = older(
            oldest,
            oldest_source_ref(
                {"source_refs": changes.get("source_refs", [])}, ctx.patch, ctx.report
            ),
        )
    return oldest


def author_update_nodes(op: UpdateNodesOperation, ctx: OpContext) -> Any:
    for update in op.nodes:
        node = ctx.state.nodes.get(update.id)
        changes = update.changes
        if node is None:
            continue
        is_direct_human_edit = (
            ctx.patch.kind == "approval"
            and ctx.reference_patch is None
            and not any(
                isinstance(operation, ResolveProposalsOperation) for operation in ctx.patch.ops
            )
        )
        if not is_direct_human_edit:
            validate_updated_node_authoring(node, changes, ctx.report, ctx.revision)
        validate_extension_update(
            ctx.state,
            ctx.patch,
            node,
            changes,
            ctx.report,
            authoring=True,
        )
        if (
            isinstance(node, Hypothesis)
            and "status" in changes
            and changes["status"] != node.status
        ):
            _validate_belief_cause(ctx, node.id, update.cause)
    return None


def depends_update_nodes(
    op: UpdateNodesOperation | ProposalContentChangeOperation | ProposalStatusChangeOperation,
    state: GraphState,
) -> tuple[list[Any], list[str]]:
    updates = op.nodes
    config_keys = ["ontology" for update in updates if "extension_fields" in update.changes]
    return [update.id for update in updates], config_keys


def validate_create_edges(op: CreateEdgesOperation, ctx: OpContext) -> Any:
    created_nodes = [
        node.model_dump(mode="python", exclude_unset=True)
        for patch_op in ctx.patch.ops
        if isinstance(patch_op, CreateNodesOperation)
        for node in patch_op.nodes
    ]
    seen_edge_ids = set(ctx.state.edges)
    for edge in op.edges:
        source_id = edge.source
        target_id = edge.target
        if source_id not in ctx.state.nodes and not _created_node_id(ctx, source_id):
            ctx.report.reject(
                "unknown-edge-source",
                f"Unknown edge source {source_id!r}.",
                ctx.revision,
            )
        if target_id not in ctx.state.nodes and not _created_node_id(ctx, target_id):
            ctx.report.reject(
                "unknown-edge-target",
                f"Unknown edge target {target_id!r}.",
                ctx.revision,
            )
        data = edge.model_dump(mode="python", exclude_unset=True)
        relation = edge.relation
        source_type = _node_type(ctx, source_id)
        target_type = _node_type(ctx, target_id)
        assessment_applies = (
            relation in EVIDENCE_HYPOTHESIS_RELATIONS
            and source_type == "evidence"
            and target_type == "hypothesis"
        )
        edge_id = edge.id or f"{source_id}::{relation}::{target_id}"
        evidence_relation_endpoints_apply = target_type == "hypothesis" and (
            source_type == "evidence" or (relation == "contradicts" and source_type == "hypothesis")
        )
        if (
            ctx.mode == "admission"
            and relation in EVIDENCE_HYPOTHESIS_RELATIONS
            and source_type is not None
            and target_type is not None
            and not evidence_relation_endpoints_apply
        ):
            ctx.report.reject(
                "invalid-evidence-relation-endpoints",
                f"Relation {relation!r} requires Evidence -> Hypothesis"
                + (" or Hypothesis -> Hypothesis" if relation == "contradicts" else "")
                + f" endpoints, not {source_type} -> {target_type}.",
                ctx.revision,
                related_node_ids=[source_id, target_id],
                related_edge_ids=[edge_id],
            )
        if edge.assessment is not None and not assessment_applies:
            ctx.report.reject(
                "inapplicable-evidence-assessment",
                f"Edge {edge_id!r} may carry an assessment only when Evidence bears on a "
                "Hypothesis through supports, weakens, refutes, inconclusive, or contradicts.",
                ctx.revision,
                related_node_ids=[source_id, target_id],
                related_edge_ids=[edge_id],
            )
        elif ctx.mode == "admission" and assessment_applies and edge.assessment is None:
            ctx.report.reject(
                "missing-evidence-assessment",
                f"New Evidence-to-Hypothesis edge {edge_id!r} requires a claim-relative "
                "assessment.",
                ctx.revision,
                related_node_ids=[source_id, target_id],
                related_edge_ids=[edge_id],
            )
        custom = custom_relation(ctx.state.ontology, relation)
        if relation not in RELATION_SPEC:
            if custom is None:
                ctx.report.reject(
                    "invalid-edge",
                    f"Edge {data.get('id')!r} uses unknown relation {relation!r}.",
                    ctx.revision,
                    related_node_ids=[source_id, target_id],
                )
                continue
            else:
                data["layer"] = custom.layer
                if not edge_matches_relation(
                    ctx.state,
                    source_id,
                    target_id,
                    custom,
                    created_nodes=created_nodes,
                ):
                    ctx.report.reject(
                        "custom-relation-type-mismatch",
                        f"Relation {custom.name!r} does not allow the semantic endpoint types "
                        f"for {source_id!r} -> {target_id!r}.",
                        ctx.revision,
                        related_node_ids=[source_id, target_id],
                    )
        if "id" not in data and source_id is not None and target_id is not None:
            data["id"] = f"{source_id}::{data.get('relation')}::{target_id}"
        edge_id = data.get("id")
        if ctx.mode == "admission" and isinstance(edge_id, str):
            if edge_id in seen_edge_ids:
                ctx.report.reject(
                    "duplicate-edge-id",
                    f"Edge {edge_id!r} already exists; remove it before creating a replacement.",
                    ctx.revision,
                    related_edge_ids=[edge_id],
                )
                continue
            seen_edge_ids.add(edge_id)
        try:
            Edge.model_validate(data)
        except ValidationError as exc:
            ctx.report.reject(
                "invalid-edge",
                f"Edge {data.get('id')!r} is invalid: {exc.errors()[0]['msg']}.",
                ctx.revision,
                related_node_ids=[source_id, target_id],
                related_edge_ids=[data["id"]] if isinstance(data.get("id"), str) else [],
            )
    return None


def author_create_edges(op: CreateEdgesOperation, ctx: OpContext) -> Any:
    for edge in op.edges:
        relation = edge.relation
        custom = custom_relation(ctx.state.ontology, relation)
        if custom is not None and custom.deprecated:
            ctx.report.reject(
                "deprecated-custom-relation",
                f"Custom relation {custom.name!r} is deprecated and cannot author new edges.",
                ctx.revision,
            )
        spec = RELATION_SPEC.get(relation)
        if spec is None:
            continue
        source_id = edge.source
        target_id = edge.target
        source_type = _node_type(ctx, source_id)
        target_type = _node_type(ctx, target_id)
        if source_type is None or target_type is None:
            continue
        type_mismatch = source_type not in spec.source_types or target_type not in spec.target_types
        same_type_mismatch = spec.same_type and source_type != target_type
        if not type_mismatch and not same_type_mismatch:
            continue
        edge_id = edge.id or f"{source_id}::{relation}::{target_id}"
        allowed_sources = ", ".join(sorted(spec.source_types))
        allowed_targets = ", ".join(sorted(spec.target_types))
        same_type = "; source and target must have the same type" if spec.same_type else ""
        ctx.report.flag(
            "relation-type-mismatch",
            f"Edge {edge_id!r} uses {relation!r} from {source_type} to {target_type}; "
            f"allowed source types are [{allowed_sources}] and target types are "
            f"[{allowed_targets}]{same_type}.",
            ctx.revision,
            related_node_ids=[source_id, target_id],
            related_edge_ids=[edge_id],
        )
    return None


def depends_create_edges(
    op: CreateEdgesOperation | ProposalProtectedRelationOperation,
    state: GraphState,
) -> tuple[list[Any], list[str]]:
    candidates: list[Any] = []
    for edge in op.edges or []:
        candidates.append(edge.source)
        candidates.append(edge.target)
    return candidates, []


def validate_remove_edges(op: RemoveEdgesOperation, ctx: OpContext) -> Any:
    edge_ids = op.edge_ids
    if not edge_ids or any(not edge_id for edge_id in edge_ids):
        ctx.report.reject(
            "invalid-remove-edges-operation",
            "A remove_edges operation requires at least one non-empty edge id.",
            ctx.revision,
        )
        return None
    if len(edge_ids) != len(set(edge_ids)):
        ctx.report.reject(
            "invalid-remove-edges-operation",
            "A remove_edges operation cannot name the same edge more than once.",
            ctx.revision,
        )
        return None
    return None


def depends_remove_edges(
    op: RemoveEdgesOperation | ProposalProtectedRelationOperation,
    state: GraphState,
) -> tuple[list[Any], list[str]]:
    candidates: list[Any] = []
    for edge_id in op.edge_ids or []:
        edge = state.edges.get(edge_id)
        if edge is not None:
            candidates.append(edge.source)
            candidates.append(edge.target)
    return candidates, []


def validate_remove_nodes(op: RemoveNodesOperation, ctx: OpContext) -> Any:
    node_ids = op.node_ids
    if not node_ids or any(not node_id for node_id in node_ids):
        ctx.report.reject(
            "invalid-remove-nodes-operation",
            "A remove_nodes operation requires at least one non-empty node id.",
            ctx.revision,
        )
        return None
    if len(node_ids) != len(set(node_ids)):
        ctx.report.reject(
            "invalid-remove-nodes-operation",
            "A remove_nodes operation cannot name the same node more than once.",
            ctx.revision,
        )
        return None

    is_proposal_approval = ctx.patch.kind == "approval" and (
        ctx.reference_patch is not None
        or any(isinstance(operation, ResolveProposalsOperation) for operation in ctx.patch.ops)
    )
    for node_id in node_ids:
        node = ctx.state.nodes.get(node_id)
        if node is None:
            ctx.report.reject(
                "unknown-node",
                f"Cannot remove missing node {node_id!r}.",
                ctx.revision,
                related_node_ids=[node_id],
            )
            continue

        initial_node = ctx.initial_state.nodes.get(node_id, node)
        if initial_node.standing == Standing.ACCEPTED and not is_proposal_approval:
            ctx.report.reject(
                "accepted-node-removal",
                f"Cannot remove accepted node {node_id!r}; clear or contest it in an earlier "
                "human Sync first.",
                ctx.revision,
                related_node_ids=[node_id],
            )
        experiment_versions = (
            candidate for candidate in (initial_node, node) if isinstance(candidate, Experiment)
        )
        if any(
            attempt.status in ACTIVE_EXPERIMENT_ATTEMPT_STATUSES
            for experiment in experiment_versions
            for attempt in experiment.attempts
        ):
            ctx.report.reject(
                "active-experiment-removal",
                f"Cannot remove Experiment {node_id!r} while its bounded loop has an active "
                "attempt.",
                ctx.revision,
                related_node_ids=[node_id],
            )
    return None


def depends_remove_nodes(
    op: RemoveNodesOperation | ProposalRemovalOperation,
    state: GraphState,
) -> tuple[list[Any], list[str]]:
    return list(op.node_ids), []


def validate_supersede_nodes(op: SupersedeNodesOperation, ctx: OpContext) -> Any:
    for item in op.nodes:
        node_id = item.id
        node = ctx.state.nodes.get(node_id)
        if node is None:
            ctx.report.reject(
                "unknown-node",
                f"Cannot supersede missing node {node_id!r}.",
                ctx.revision,
            )
        target_id = item.superseded_by
        if (
            target_id
            and target_id not in ctx.state.nodes
            and not _created_node_id(ctx, target_id)
            and _node_type(ctx, target_id) is None
        ):
            ctx.report.reject(
                "unknown-node",
                f"Cannot supersede {node_id!r} with missing node {target_id!r}.",
                ctx.revision,
            )
        if ctx.mode == "admission":
            _validate_generated_relation_endpoints(
                ctx,
                node_id,
                target_id,
                relation="supersedes",
            )
    return None


def depends_supersede_nodes(
    op: SupersedeNodesOperation | ProposalSupersedeOperation,
    state: GraphState,
) -> tuple[list[Any], list[str]]:
    candidates: list[Any] = []
    for item in op.nodes:
        candidates.append(item.id)
        candidates.append(item.superseded_by)
    return candidates, []


def validate_merge_nodes(op: MergeNodesOperation, ctx: OpContext) -> Any:
    for item in op.merges:
        duplicate_id = item.duplicate
        canonical_id = item.canonical
        duplicate = ctx.state.nodes.get(duplicate_id)
        canonical = ctx.state.nodes.get(canonical_id)
        if duplicate is None:
            ctx.report.reject(
                "unknown-node",
                f"Cannot merge missing duplicate node {duplicate_id!r}.",
                ctx.revision,
            )
        if canonical is None and _node_type(ctx, canonical_id) is None:
            ctx.report.reject(
                "unknown-node",
                f"Cannot merge into missing canonical node {canonical_id!r}.",
                ctx.revision,
            )
        if ctx.mode == "admission":
            _validate_generated_relation_endpoints(
                ctx,
                duplicate_id,
                canonical_id,
                relation="duplicate_of",
            )
    return None


def _validate_generated_relation_endpoints(
    ctx: OpContext,
    source_id: Any,
    target_id: Any,
    *,
    relation: str,
) -> None:
    if not isinstance(source_id, str) or not isinstance(target_id, str):
        return
    if source_id == target_id:
        ctx.report.reject(
            "invalid-generated-relation",
            f"Relation {relation!r} requires distinct source and target nodes.",
            ctx.revision,
            related_node_ids=[source_id],
        )
        return
    edge_id = f"{source_id}::{relation}::{target_id}"
    if ctx.mode == "admission" and edge_id in ctx.state.edges:
        ctx.report.reject(
            "duplicate-edge-id",
            f"Edge {edge_id!r} already exists; remove it before creating a replacement.",
            ctx.revision,
            related_edge_ids=[edge_id],
        )
        return
    source_type = _node_type(ctx, source_id)
    target_type = _node_type(ctx, target_id)
    if source_type is not None and target_type is not None and source_type != target_type:
        ctx.report.reject(
            "generated-relation-type-mismatch",
            f"Relation {relation!r} requires endpoints of the same type, not "
            f"{source_type!r} and {target_type!r}.",
            ctx.revision,
            related_node_ids=[source_id, target_id],
        )


def depends_merge_nodes(
    op: MergeNodesOperation | ProposalMergeOperation,
    state: GraphState,
) -> tuple[list[Any], list[str]]:
    candidates: list[Any] = []
    for item in op.merges:
        candidates.append(item.duplicate)
        candidates.append(item.canonical)
    return candidates, []


def depends_create_ambiguities(
    op: CreateAmbiguitiesOperation, state: GraphState
) -> tuple[list[Any], list[str]]:
    candidates: list[Any] = []
    for ambiguity in op.ambiguities:
        candidates.extend(ambiguity.related_node_ids)
    return candidates, []


def validate_resolve_ambiguities(op: ResolveAmbiguitiesOperation, ctx: OpContext) -> Any:
    for resolution in op.resolutions:
        ambiguity_id = resolution.id
        if ambiguity_id not in ctx.state.ambiguities:
            ctx.report.reject(
                "unknown-ambiguity",
                f"Cannot resolve missing ambiguity {ambiguity_id!r}.",
                ctx.revision,
            )
        if resolution.status not in {"resolved", "dismissed"}:
            ctx.report.reject(
                "invalid-ambiguity-resolution",
                "Ambiguities may only be resolved or dismissed.",
                ctx.revision,
            )
    return None


def depends_resolve_ambiguities(
    op: ResolveAmbiguitiesOperation, state: GraphState
) -> tuple[list[Any], list[str]]:
    candidates: list[Any] = []
    for resolution in op.resolutions:
        ambiguity = state.ambiguities.get(resolution.id)
        if ambiguity is not None:
            candidates.extend(ambiguity.related_node_ids)
    return candidates, []


def validate_create_proposals(op: CreateProposalsOperation, ctx: OpContext) -> Any:
    seen_ids: set[str] = set()
    for proposal in op.proposals:
        proposal_id = proposal.id
        if ctx.mode == "admission" and proposal_id in seen_ids:
            ctx.report.reject(
                "duplicate-proposal-id",
                f"Proposal {proposal_id!r} appears more than once in one create_proposals "
                "operation.",
                ctx.revision,
            )
        seen_ids.add(proposal_id)
        validate_proposal(
            proposal,
            ctx.state,
            ctx.report,
            ctx.revision,
            project_truth_scope=ctx.project_truth_scope,
            repository_aliases=ctx.repositories,
            machine_aliases=ctx.machines,
            default_run_truth_scope=ctx.default_run_truth_scope,
            state_repository=ctx.state_repository,
            context_patch=ctx.patch,
        )
    return None


def author_create_proposals(op: CreateProposalsOperation, ctx: OpContext) -> Any:
    for proposal in op.proposals:
        validate_proposal(
            proposal,
            ctx.state,
            ctx.report,
            ctx.revision,
            project_truth_scope=ctx.project_truth_scope,
            repository_aliases=ctx.repositories,
            machine_aliases=ctx.machines,
            default_run_truth_scope=ctx.default_run_truth_scope,
            state_repository=ctx.state_repository,
            validation_mode="admission",
            include_card_flags=True,
            context_patch=ctx.patch,
        )
    return None


def validate_resolve_proposals(op: ResolveProposalsOperation, ctx: OpContext) -> Any:
    if ctx.mode == "admission" and ctx.patch.kind != "approval":
        ctx.report.reject(
            "agent-resolved-proposal",
            "Only a human approval patch may resolve or withdraw a Proposal.",
            ctx.revision,
        )
    for resolution in op.resolutions:
        proposal_id = resolution.id
        proposal = ctx.state.proposals.get(proposal_id)
        if proposal is None:
            ctx.report.reject(
                "unknown-proposal",
                f"Cannot resolve missing proposal {proposal_id!r}.",
                ctx.revision,
            )
        elif proposal.status != "pending":
            ctx.report.reject(
                "proposal-not-pending",
                f"Proposal {proposal_id!r} is not pending.",
                ctx.revision,
            )
    return None


def validate_withdraw_proposals(op: WithdrawProposalsOperation, ctx: OpContext) -> Any:
    if ctx.patch.author != "agent" or ctx.patch.kind == "approval":
        ctx.report.reject(
            "human-only-proposal-withdrawal",
            "Only an agent patch may withdraw a Proposal.",
            ctx.revision,
        )
    for withdrawal in op.proposals:
        proposal_id = withdrawal.id
        proposal = ctx.state.proposals.get(proposal_id)
        if proposal is None:
            ctx.report.reject(
                "unknown-proposal",
                f"Cannot withdraw missing proposal {proposal_id!r}.",
                ctx.revision,
            )
        elif proposal.status != "pending":
            ctx.report.reject(
                "proposal-not-pending",
                f"Proposal {proposal_id!r} is not pending.",
                ctx.revision,
            )
    return None


def validate_set_standing(op: SetStandingOperation, ctx: OpContext) -> Any:
    if ctx.mode == "admission" and not permits(ctx.patch, "set_standing"):
        ctx.report.reject(
            "agent-set-standing",
            "This Patch producer may not set standing.",
            ctx.revision,
        )
    node_id = op.node_id
    if node_id not in ctx.state.nodes:
        ctx.report.reject("unknown-node", f"Cannot review missing node {node_id!r}.", ctx.revision)
    if op.standing not in {"asserted", "accepted", "contested"}:
        ctx.report.reject(
            "invalid-standing",
            "Standing may be set to asserted, accepted, or contested.",
            ctx.revision,
        )
    return None


def validate_set_project_truth_scope(op: SetProjectTruthScopeOperation, ctx: OpContext) -> Any:
    if ctx.patch.kind != "approval":
        ctx.report.reject(
            "agent-set-project-scope",
            "Project truth-scope membership requires human approval.",
            ctx.revision,
        )
    proposed = set(op.truth_scope)
    descriptor = op.repository
    if descriptor is not None:
        alias = descriptor.alias
        machine = descriptor.machine
        if not alias or not machine or not descriptor.path:
            ctx.report.reject(
                "incomplete-repository",
                "A new repository descriptor needs alias, machine, and path.",
                ctx.revision,
            )
        else:
            if ctx.machines is not None and machine not in ctx.machines:
                ctx.report.reject(
                    "unknown-repository-machine",
                    f"Repository {alias!r} uses unknown machine {machine!r}.",
                    ctx.revision,
                )
            if alias not in ctx.repositories:
                ctx.repositories.add(alias)
    unknown = proposed - ctx.repositories
    if unknown:
        ctx.report.reject(
            "unknown-project-repository",
            f"Project truth scope names unknown repositories: {sorted(unknown)}.",
            ctx.revision,
        )
    if ctx.state_repository and ctx.state_repository not in proposed:
        ctx.report.reject(
            "remove-state-repository",
            "The canonical state repository must remain in project truth scope in v1.",
            ctx.revision,
        )
    removed_defaults = ctx.default_run_truth_scope - proposed
    if removed_defaults:
        ctx.report.reject(
            "remove-default-run-repository",
            "Project truth scope must retain every repository in the default run scope: "
            f"{sorted(removed_defaults)}.",
            ctx.revision,
        )
    return None


def depends_set_project_truth_scope(
    op: SetProjectTruthScopeOperation, state: GraphState
) -> tuple[list[Any], list[str]]:
    return [], ["project_truth_scope"]


def validate_set_ontology(op: SetOntologyOperation, ctx: OpContext) -> Any:
    ontology = op.ontology
    validate_ontology_structure(ontology, ctx.report, ctx.revision)
    if ctx.patch.kind != "approval":
        ctx.report.reject(
            "agent-set-ontology",
            "Only the human Settings and Sync paths may change ontology.",
            ctx.revision,
        )
    if ctx.mode == "admission":
        validate_ontology_change(ctx.state, ontology, ctx.report, ctx.revision)
    return None


def depends_set_ontology(
    op: SetOntologyOperation, state: GraphState
) -> tuple[list[Any], list[str]]:
    return [], ["ontology"]


def _validate_belief_cause(
    ctx: OpContext,
    hypothesis_id: str,
    cause: BeliefCause | None,
) -> None:
    related = [hypothesis_id]
    if cause is None:
        ctx.report.reject(
            "missing-belief-cause",
            f"Changing Hypothesis {hypothesis_id!r} status requires a cause object.",
            ctx.revision,
            related_node_ids=related,
        )
        return
    if isinstance(cause, HumanEditCause):
        if ctx.patch.kind != "approval" or ctx.patch.author != "human":
            ctx.report.reject(
                "invalid-belief-cause",
                f"human_edit cause for {hypothesis_id!r} is legal only on human approval.",
                ctx.revision,
                related_node_ids=related,
            )
        return
    if isinstance(cause, DecisionCause):
        ref_id = cause.ref_id
        node_type = _node_type(ctx, ref_id)
        if node_type != "decision":
            ctx.report.reject(
                "invalid-belief-cause",
                f"Decision cause {ref_id!r} for {hypothesis_id!r} does not name a Decision.",
                ctx.revision,
                related_node_ids=[hypothesis_id, ref_id],
            )
        return
    if isinstance(cause, ProposalResolutionCause):
        ref_id = cause.ref_id
        resolved = {
            item.id
            for operation in ctx.patch.ops
            if isinstance(operation, ResolveProposalsOperation)
            for item in operation.resolutions
        }
        if ref_id not in resolved:
            ctx.report.reject(
                "invalid-belief-cause",
                f"Proposal-resolution cause {ref_id!r} for {hypothesis_id!r} is not resolved in this patch.",
                ctx.revision,
                related_node_ids=related,
            )
        return

    if not isinstance(cause, EvidenceEdgeCause):
        ctx.report.reject(
            "invalid-belief-cause",
            f"Hypothesis {hypothesis_id!r} has unknown belief cause kind {cause.kind!r}.",
            ctx.revision,
            related_node_ids=related,
        )
        return
    ref_id = cause.ref_id
    edge = _edge_in_context(ctx, ref_id)
    if (
        edge is None
        or edge.target != hypothesis_id
        or edge.relation not in {"supports", "weakens", "refutes", "inconclusive", "contradicts"}
        or _node_type(ctx, edge.source) != "evidence"
    ):
        ctx.report.reject(
            "invalid-belief-cause",
            f"Evidence-edge cause {ref_id!r} must be an evidence relation targeting {hypothesis_id!r}.",
            ctx.revision,
            related_node_ids=related,
            related_edge_ids=[ref_id] if isinstance(ref_id, str) else [],
        )


def _node_type(ctx: OpContext, node_id: Any) -> str | None:
    existing = ctx.state.nodes.get(node_id)
    if existing is not None:
        return existing.type
    for patch in (ctx.patch, ctx.reference_patch):
        if patch is None:
            continue
        for operation in patch.ops:
            if not isinstance(operation, CreateNodesOperation):
                continue
            for node in operation.nodes:
                if node.id == node_id:
                    return node.type
    return None


def _edge_in_context(ctx: OpContext, edge_id: Any) -> Edge | None:
    existing = ctx.state.edges.get(edge_id)
    if existing is not None:
        return existing
    for patch in (ctx.patch, ctx.reference_patch):
        if patch is None:
            continue
        for operation in patch.ops:
            if not isinstance(operation, CreateEdgesOperation):
                continue
            for edge in operation.edges:
                candidate_id = edge.id or (f"{edge.source}::{edge.relation}::{edge.target}")
                if candidate_id != edge_id:
                    continue
                data = edge.model_dump(mode="python", exclude_unset=True)
                data["id"] = candidate_id
                try:
                    return Edge.model_validate(data)
                except ValidationError:
                    return None
    return None


def _created_node_id(ctx: OpContext, node_id: Any) -> bool:
    return any(
        node.id == node_id
        for operation in ctx.patch.ops
        if isinstance(operation, CreateNodesOperation)
        for node in operation.nodes
    )
