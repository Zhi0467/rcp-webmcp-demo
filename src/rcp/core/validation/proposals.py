from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import ValidationError

from rcp.core.authority import (
    CONTENT_CHANGE_INTENT,
    HYPOTHESIS_PROPOSAL_FIELDS,
    MERGE_INTENT,
    PROPOSAL_INTENTS,
    PROTECTED_EPISTEMIC_RELATIONS,
    PROTECTED_RELATION_CHANGE_INTENT,
    REMOVAL_INTENT,
    STATUS_CHANGE_INTENT,
    SUPERSEDE_INTENT,
    created_edge_ids,
    created_node_ids,
)
from rcp.core.models import (
    Decision,
    GraphState,
    Hypothesis,
    Patch,
    Proposal,
    ResearchQuestion,
)
from rcp.core.operations import (
    CreateNodesOperation,
    DecisionCause,
    EvidenceEdgeCause,
    GraphOperation,
    LegacyProposalMergeOperation,
    LegacyProposalProtectedRelationOperation,
    LegacyProposalRemovalOperation,
    LegacyProposalSupersedeOperation,
    NodeUpdate,
    ProposalContentChangeOperation,
    ProposalMergeOperation,
    ProposalOperation,
    ProposalProtectedRelationOperation,
    ProposalRemovalOperation,
    ProposalStatusChangeOperation,
    ProposalSupersedeOperation,
    UpdateNodesOperation,
    graph_operations_from_proposal,
)
from rcp.core.validation.constants import IDENTIFIER_RE
from rcp.core.validation.nodes import oldest_source_ref
from rcp.core.validation.report import ValidationReport


def decision_transition_error(decision: Decision, changes: dict[str, Any]) -> str | None:
    """Return the authority-coherence error for a proposed Decision transition."""

    selected_option = changes.get("selected_option", decision.selected_option)
    status = changes.get("status", decision.status)
    options = changes.get("options", decision.options)
    if (
        changes.get("selected_option") is not None
        and isinstance(options, list)
        and changes["selected_option"] not in options
    ):
        return f"Decision {decision.id} can select only an option listed in its resulting options."
    if status == "decided" and (
        selected_option is None or not isinstance(options, list) or selected_option not in options
    ):
        return (
            f"Decision {decision.id} can be decided only with a selected option listed in its "
            "resulting options."
        )
    if status == "revisit" and decision.status != "revisit" and decision.selected_option is None:
        return f"Decision {decision.id} can be revisited only after it has a prior decision."
    if (
        decision.status in {"open", "ready", "revisit"}
        and changes.get("selected_option") is not None
        and status != "decided"
    ):
        return f"Decision {decision.id} must become decided when an option is selected."
    return None


def normalized_decision_proposal_ops(state: GraphState, proposal: Proposal) -> list[GraphOperation]:
    """Add the one implied field accepted for legacy Decision Proposal approval."""

    operations = graph_operations_from_proposal(proposal.ops)
    normalized: list[GraphOperation] = []
    for operation in operations:
        if not isinstance(operation, UpdateNodesOperation):
            normalized.append(operation)
            continue
        updates: list[NodeUpdate] = []
        for update in operation.nodes:
            node = state.nodes.get(update.id)
            changes = dict(update.changes)
            if (
                isinstance(node, Decision)
                and changes.get("selected_option") is not None
                and "status" not in changes
                and node.status != "decided"
            ):
                changes["status"] = "decided"
                update = update.model_copy(update={"changes": changes})
            updates.append(update)
        normalized.append(operation.model_copy(update={"nodes": updates}))
    return normalized


def proposal_updates_node(proposal: Proposal, node_id: str) -> bool:
    """Whether any semantic node update in a Proposal targets ``node_id``."""

    return any(
        update.id == node_id
        for operation in proposal.ops
        if isinstance(operation, (ProposalContentChangeOperation, ProposalStatusChangeOperation))
        for update in operation.nodes
    )


def proposal_is_stale(state: GraphState, proposal: Proposal) -> bool:
    """Whether the state a pending Proposal depends on has moved or disappeared."""

    dependency_revision = proposal.raised_rev or proposal.base_rev
    if any(
        node_id not in state.nodes or state.nodes[node_id].updated_rev > dependency_revision
        for node_id in proposal.related_node_ids
    ):
        return True
    if any(
        state.config_revisions.get(key, 0) > dependency_revision
        for key in proposal.related_config_keys
    ):
        return True

    referenced_edge_ids, decision_ids = _proposal_reference_dependencies(proposal)
    edge_ids = set(proposal.related_edge_ids) | referenced_edge_ids
    reference_revision = proposal.raised_rev or proposal.base_rev
    if any(
        edge_id not in state.edges or state.edges[edge_id].created_rev > reference_revision
        for edge_id in edge_ids
    ):
        return True
    if any(edge_id in state.edges for edge_id in _proposal_created_edge_ids(proposal)):
        return True
    removal_targets = _proposal_removal_targets(proposal)
    if removal_targets:
        current_incident_edge_ids = {
            edge.id
            for edge in state.edges.values()
            if edge.source in removal_targets or edge.target in removal_targets
        }
        if current_incident_edge_ids != set(proposal.related_edge_ids):
            return True
    return any(
        not isinstance(state.nodes.get(decision_id), Decision)
        or state.nodes[decision_id].updated_rev > reference_revision
        for decision_id in decision_ids
    )


def _proposal_removal_targets(proposal: Proposal) -> set[str]:
    return {
        node_id
        for operation in proposal.ops
        if isinstance(operation, ProposalRemovalOperation)
        and not isinstance(operation, LegacyProposalRemovalOperation)
        for node_id in operation.node_ids
    }


def _proposal_reference_dependencies(proposal: Proposal) -> tuple[set[str], set[str]]:
    edge_ids: set[str] = set()
    decision_ids: set[str] = set()
    for op in proposal.ops:
        if isinstance(op, ProposalProtectedRelationOperation) and op.op == "remove_edges":
            edge_ids.update(op.edge_ids or [])
        if not isinstance(op, (ProposalContentChangeOperation, ProposalStatusChangeOperation)):
            continue
        for update in op.nodes:
            cause = update.cause
            if isinstance(cause, EvidenceEdgeCause):
                edge_ids.add(cause.ref_id)
            elif isinstance(cause, DecisionCause):
                decision_ids.add(cause.ref_id)
    return edge_ids, decision_ids


def _proposal_created_edge_ids(proposal: Proposal) -> set[str]:
    """Return edge IDs whose absence is part of the judged semantic effect."""

    edge_ids: set[str] = set()
    for operation in proposal.ops:
        if (
            isinstance(operation, ProposalProtectedRelationOperation)
            and not isinstance(operation, LegacyProposalProtectedRelationOperation)
            and operation.op == "create_edges"
        ):
            for raw in operation.edges or []:
                edge_ids.add(raw.id or f"{raw.source}::{raw.relation}::{raw.target}")
        elif isinstance(operation, ProposalSupersedeOperation) and not isinstance(
            operation, LegacyProposalSupersedeOperation
        ):
            edge_ids.update(
                f"{item.id}::supersedes::{item.superseded_by}"
                for item in operation.nodes
                if item.superseded_by is not None
            )
        elif isinstance(operation, ProposalMergeOperation) and not isinstance(
            operation, LegacyProposalMergeOperation
        ):
            edge_ids.update(
                f"{item.duplicate}::duplicate_of::{item.canonical}" for item in operation.merges
            )
    return edge_ids


def validate_proposal(
    proposal: Proposal | dict[str, Any],
    state: GraphState,
    report: ValidationReport,
    revision: int | None,
    *,
    project_truth_scope: Iterable[str],
    repository_aliases: Iterable[str],
    machine_aliases: Iterable[str] | None,
    default_run_truth_scope: Iterable[str],
    state_repository: str | None,
    validation_mode: Literal["admission", "replay"] = "replay",
    include_card_flags: bool = False,
    context_patch: Patch | None = None,
) -> None:
    if not isinstance(proposal, Proposal):
        try:
            proposal = Proposal.model_validate(proposal)
        except ValidationError as exc:
            report.reject(
                "invalid-proposal",
                f"Proposal is malformed: {exc.errors()[0]['msg']}.",
                revision,
            )
            return
    if proposal.id in state.proposals:
        report.reject(
            "duplicate-proposal-id", f"Proposal {proposal.id!r} already exists.", revision
        )
        return
    if validation_mode == "admission" and proposal.base_rev != state.revision:
        report.reject(
            "proposal-base-revision",
            f"Proposal {proposal.id} must use the current graph revision {state.revision}.",
            revision,
        )
    if include_card_flags:
        missing = [
            name
            for name, value in proposal.card.model_dump().items()
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            report.flag(
                "incomplete-gated-card",
                f"Proposal {proposal.id} is missing readable card fields: {', '.join(missing)}.",
                revision,
            )
        card_text = " ".join(proposal.card.model_dump().values())
        unresolved = sorted(
            token for token in set(IDENTIFIER_RE.findall(card_text)) if token not in state.glossary
        )
        if unresolved:
            report.flag(
                "missing-glossary-term",
                f"Proposal {proposal.id} uses unexplained identifiers: {', '.join(unresolved)}.",
                revision,
            )
    if validation_mode == "admission" and (
        context_patch is None or context_patch.author == "agent"
    ):
        _validate_agent_proposal_boundary(
            proposal,
            state,
            report,
            revision,
            context_patch=context_patch,
        )
    _validate_proposal_ops(
        proposal,
        state,
        report,
        revision,
        project_truth_scope=project_truth_scope,
        repository_aliases=repository_aliases,
        machine_aliases=machine_aliases,
        default_run_truth_scope=default_run_truth_scope,
        state_repository=state_repository,
        validation_mode=validation_mode,
        context_patch=context_patch,
    )


def _validate_agent_proposal_boundary(
    proposal: Proposal,
    state: GraphState,
    report: ValidationReport,
    revision: int | None,
    *,
    context_patch: Patch | None,
) -> None:
    def refuse(message: str) -> None:
        report.reject("invalid-agent-proposal-shape", message, revision)

    if len(proposal.ops) != 1:
        refuse(f"Proposal {proposal.id} must declare exactly one protected-change intent.")
        return
    operation = proposal.ops[0]
    intent = operation.intent
    if intent not in PROPOSAL_INTENTS:
        refuse(
            f"Proposal {proposal.id} must declare one of these intents: "
            f"{', '.join(sorted(PROPOSAL_INTENTS))}."
        )
        return
    validators = {
        CONTENT_CHANGE_INTENT: _validate_content_change_intent,
        REMOVAL_INTENT: _validate_removal_intent,
        SUPERSEDE_INTENT: _validate_supersede_intent,
        MERGE_INTENT: _validate_merge_intent,
        PROTECTED_RELATION_CHANGE_INTENT: _validate_protected_relation_change_intent,
        STATUS_CHANGE_INTENT: _validate_status_change_intent,
    }
    error = validators[intent](state, context_patch, operation)
    if error is not None:
        refuse(f"Proposal {proposal.id} declares {intent!r}, but {error}")


def _validate_content_change_intent(
    state: GraphState,
    context_patch: Patch | None,
    operation: ProposalOperation,
) -> str | None:
    update, error = _one_update(operation)
    if error is not None:
        return error
    assert update is not None
    if update.cause is not None or update.base_updated_rev is not None:
        return "a content change requires exactly id and changes, with no cause."
    node = _existing_protected_node(state, context_patch, update.id)
    if node is None:
        return "a content change must target one existing ResearchQuestion or Hypothesis."
    changes = update.changes
    if not changes:
        return "a content change must contain at least one changed field."
    if "status" in changes and isinstance(node, Hypothesis):
        return "Hypothesis status belongs in a status_change intent."
    if all(getattr(node, field, object()) == value for field, value in changes.items()):
        return "a content change must actually change the target node."
    return None


def _validate_status_change_intent(
    state: GraphState,
    context_patch: Patch | None,
    operation: ProposalOperation,
) -> str | None:
    update, error = _one_update(operation)
    if error is not None:
        return error
    assert update is not None
    node = state.nodes.get(update.id)
    if isinstance(node, Decision):
        return (
            f"Decision {node.id!r} cannot be proposed by an agent; its outcome requires the "
            "decide_decision action."
        )
    if not isinstance(node, Hypothesis):
        return "a status change must target one Hypothesis in the staged graph."
    if update.cause is None or update.base_updated_rev is not None:
        return "a status change requires exactly id, changes, and cause."
    changes = update.changes
    if set(changes) != HYPOTHESIS_PROPOSAL_FIELDS:
        return "a status change may change only Hypothesis status."
    if changes["status"] == node.status:
        return "a status change must actually change the Hypothesis status."
    cause = update.cause
    if not isinstance(cause, EvidenceEdgeCause) or not cause.ref_id:
        return "a status change requires an evidence_edge cause naming an epistemic edge."
    return None


def _validate_removal_intent(
    state: GraphState,
    context_patch: Patch | None,
    operation: ProposalOperation,
) -> str | None:
    if not isinstance(operation, ProposalRemovalOperation):
        return "removal requires exactly one remove_nodes operation."
    node_ids = operation.node_ids
    if len(node_ids) != 1:
        return "removal must name exactly one node."
    if _existing_protected_node(state, context_patch, node_ids[0]) is None:
        return "removal must target one existing ResearchQuestion or Hypothesis."
    return None


def _validate_supersede_intent(
    state: GraphState,
    context_patch: Patch | None,
    operation: ProposalOperation,
) -> str | None:
    if not isinstance(operation, ProposalSupersedeOperation):
        return "supersede requires exactly one supersede_nodes operation."
    items = operation.nodes
    if len(items) != 1:
        return "supersede must name exactly one predecessor and successor."
    item = items[0]
    if item.cause is not None:
        return "supersede accepts only id, superseded_by, and optional explanation."
    node_id = item.id
    successor_id = item.superseded_by
    predecessor = _existing_protected_node(state, context_patch, node_id)
    if predecessor is None:
        return "supersede must retire one existing ResearchQuestion or Hypothesis."
    if not isinstance(successor_id, str) or not successor_id or successor_id == node_id:
        return "supersede must name a distinct successor node."
    successor_type = _protected_node_type(state, context_patch, successor_id)
    if successor_type is None:
        return "supersede must name a ResearchQuestion or Hypothesis successor."
    if successor_type != predecessor.type:
        return "supersede predecessor and successor must be the same protected belief type."
    return None


def _validate_merge_intent(
    state: GraphState,
    context_patch: Patch | None,
    operation: ProposalOperation,
) -> str | None:
    if not isinstance(operation, ProposalMergeOperation):
        return "merge requires exactly one merge_nodes operation."
    items = operation.merges
    if len(items) != 1:
        return "merge must name exactly one duplicate and canonical node."
    item = items[0]
    if item.cause is not None:
        return "merge accepts only duplicate, canonical, and optional explanation."
    duplicate_id = item.duplicate
    canonical_id = item.canonical
    duplicate = _existing_protected_node(state, context_patch, duplicate_id)
    if duplicate is None:
        return "merge must fold one existing ResearchQuestion or Hypothesis."
    if not isinstance(canonical_id, str) or not canonical_id or canonical_id == duplicate_id:
        return "merge must name a distinct canonical node."
    canonical_type = _protected_node_type(state, context_patch, canonical_id)
    if canonical_type is None:
        return "merge must name a ResearchQuestion or Hypothesis canonical node."
    if canonical_type != duplicate.type:
        return "merge duplicate and canonical must be the same protected belief type."
    return None


def _validate_protected_relation_change_intent(
    state: GraphState,
    context_patch: Patch | None,
    operation: ProposalOperation,
) -> str | None:
    if not isinstance(operation, ProposalProtectedRelationOperation):
        return "a protected relation change requires one create_edges or remove_edges operation."
    name = operation.op
    if name == "create_edges":
        edges = operation.edges or []
        if len(edges) != 1:
            return "a protected relation change must create exactly one edge."
        edge = edges[0]
        endpoints = (edge.source, edge.target)
        if edge.relation not in PROTECTED_EPISTEMIC_RELATIONS:
            return "a protected relation change must use a protected relation."
        if edge.relation in {"supersedes", "duplicate_of"}:
            return "supersedes and duplicate_of must use their dedicated supersede or merge intent."
        if context_patch is not None and any(
            node_id in created_node_ids(context_patch) for node_id in endpoints
        ):
            return "connecting a node created in the outer Patch is direct, not a Proposal."
        if not any(
            _existing_protected_node(state, context_patch, node_id) for node_id in endpoints
        ):
            return "a protected relation change must touch an existing belief node."
        return None
    if name == "remove_edges":
        edge_ids = operation.edge_ids or []
        if len(edge_ids) != 1:
            return "a protected relation change must remove exactly one edge."
        edge_id = edge_ids[0]
        edge = state.edges.get(edge_id) if isinstance(edge_id, str) else None
        if edge is None or (
            context_patch is not None and edge_id in created_edge_ids(context_patch)
        ):
            return "a protected relation removal must target one existing edge."
        if edge.relation not in PROTECTED_EPISTEMIC_RELATIONS:
            return "a protected relation change must use a protected relation."
        if not any(
            _existing_protected_node(state, context_patch, node_id)
            for node_id in (edge.source, edge.target)
        ):
            return "a protected relation change must touch an existing belief node."
        return None
    return "a protected relation change requires one create_edges or remove_edges operation."


def _one_update(operation: ProposalOperation) -> tuple[NodeUpdate | None, str | None]:
    if not isinstance(operation, (ProposalContentChangeOperation, ProposalStatusChangeOperation)):
        return None, "the declared intent requires exactly one update_nodes operation."
    updates = operation.nodes
    if len(updates) != 1:
        return None, "the declared intent must update exactly one node."
    return updates[0], None


def _existing_protected_node(
    state: GraphState,
    context_patch: Patch | None,
    node_id: Any,
) -> ResearchQuestion | Hypothesis | None:
    if not isinstance(node_id, str):
        return None
    node = state.nodes.get(node_id)
    if (
        context_patch is not None
        and node_id in created_node_ids(context_patch)
        and node is not None
        and node.created_rev == context_patch.revision
    ):
        return None
    return node if isinstance(node, (ResearchQuestion, Hypothesis)) else None


def _protected_node_type(
    state: GraphState,
    context_patch: Patch | None,
    node_id: Any,
) -> str | None:
    node = state.nodes.get(node_id) if isinstance(node_id, str) else None
    if isinstance(node, (ResearchQuestion, Hypothesis)):
        return node.type
    if context_patch is None:
        return None
    for operation in context_patch.ops:
        if not isinstance(operation, CreateNodesOperation):
            continue
        for raw in operation.nodes:
            if raw.id != node_id:
                continue
            node_type = raw.type
            return node_type if node_type in {"research_question", "hypothesis"} else None
    return None


def _validate_proposal_ops(
    proposal: Proposal,
    state: GraphState,
    report: ValidationReport,
    revision: int | None,
    *,
    project_truth_scope: Iterable[str],
    repository_aliases: Iterable[str],
    machine_aliases: Iterable[str] | None,
    default_run_truth_scope: Iterable[str],
    state_repository: str | None,
    validation_mode: Literal["admission", "replay"],
    context_patch: Patch | None,
) -> None:
    # Imported lazily because the operation registry reaches this module.
    from rcp.core.validation.context import OpContext
    from rcp.core.validation.patch import _validate_operations

    if validation_mode == "admission" and context_patch is not None:
        for operation in proposal.ops:
            if not isinstance(operation, ProposalContentChangeOperation):
                continue
            for update in operation.nodes:
                if update.changes.get("source_refs"):
                    oldest_source_ref(
                        {"source_refs": update.changes["source_refs"]}, context_patch, report
                    )

    synthetic_state = state.model_copy(
        update={
            "nodes": dict(state.nodes),
            "edges": dict(state.edges),
            "proposals": dict(state.proposals),
            "ambiguities": dict(state.ambiguities),
            "glossary": dict(state.glossary),
            "config_revisions": dict(state.config_revisions),
        }
    )
    synthetic_patch = Patch(
        revision=revision or state.revision + 1,
        kind="approval",
        author="human",
        summary=f"Validate replay operations for {proposal.id}.",
        ops=graph_operations_from_proposal(proposal.ops),
    )
    replay_report = ValidationReport()
    replay_context = OpContext(
        state=synthetic_state,
        initial_state=synthetic_state,
        patch=synthetic_patch,
        report=replay_report,
        revision=revision,
        project_truth_scope=set(project_truth_scope),
        repositories=set(repository_aliases),
        machines=set(machine_aliases) if machine_aliases is not None else None,
        default_run_truth_scope=set(default_run_truth_scope),
        state_repository=state_repository,
        mode=validation_mode,
        reference_patch=context_patch,
    )
    _validate_operations(replay_context)
    errors = [message.message for message in replay_report.messages if message.level == "reject"]
    if errors:
        report.reject(
            "invalid-proposal-ops",
            f"Proposal {proposal.id} contains invalid replay operations: {'; '.join(errors)}",
            revision,
        )
        return
