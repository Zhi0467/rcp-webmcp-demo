from __future__ import annotations

from typing import Any, Literal

from rcp.core.authority import DECIDE_DECISION, permits
from rcp.core.models import (
    ACTIVE_EXPERIMENT_ATTEMPT_STATUSES,
    HUMAN_EDITABLE_NODE_FIELDS,
    Decision,
    GraphState,
    Patch,
)
from rcp.core.operations import (
    CreateNodesOperation,
    GraphOperation,
    RemoveNodesOperation,
    ResolveAmbiguitiesOperation,
    ResolveProposalsOperation,
    SetOntologyOperation,
    SetProjectTruthScopeOperation,
    SetStandingOperation,
    UpdateNodesOperation,
    graph_operations_from_proposal,
)
from rcp.core.validation.proposals import (
    decision_transition_error,
    normalized_decision_proposal_ops,
    proposal_is_stale,
    proposal_updates_node,
)
from rcp.core.validation.report import ValidationReport


def validate_approval_shape(
    state: GraphState,
    patch: Patch,
    report: ValidationReport,
    *,
    mode: Literal["admission", "replay"],
) -> None:
    revision = patch.revision or None
    resolution_ops = [op for op in patch.ops if isinstance(op, ResolveProposalsOperation)]
    approved_resolution = any(
        resolution.status == "approved"
        for operation in resolution_ops
        for resolution in operation.resolutions
    )
    if patch.human_action == "decision_choice" and not approved_resolution:
        _validate_direct_decision_choice(state, patch, resolution_ops, report, revision)
        return
    if not resolution_ops:
        names = [op.op for op in patch.ops]
        standalone_types = (
            SetStandingOperation,
            RemoveNodesOperation,
            SetOntologyOperation,
            SetProjectTruthScopeOperation,
        )
        if len(patch.ops) == 1 and isinstance(patch.ops[0], standalone_types):
            return
        if (
            mode == "replay"
            and len(patch.ops) == 1
            and isinstance(patch.ops[0], ResolveAmbiguitiesOperation)
        ):
            return
        if len(patch.ops) == 1 and isinstance(patch.ops[0], CreateNodesOperation):
            nodes = patch.ops[0].nodes
            if len(nodes) != 1:
                report.reject(
                    "invalid-direct-node-create",
                    "A confirmed staged New node patch must create exactly one node.",
                    revision,
                )
                return
            node = nodes[0]
            if not isinstance(node.extension_type, str):
                report.reject(
                    "invalid-direct-node-create",
                    "The standalone New-node path creates exactly one custom ontology node.",
                    revision,
                )
            if node.standing != "asserted" or node.source_refs:
                report.reject(
                    "invalid-direct-node-create",
                    "A human-created custom node starts asserted and cannot claim source records.",
                    revision,
                )
            return
        if names and set(names) <= {"update_nodes", "set_standing"}:
            update_ops = [op for op in patch.ops if isinstance(op, UpdateNodesOperation)]
            standing_ops = [op for op in patch.ops if isinstance(op, SetStandingOperation)]
            if len(update_ops) != 1 or len(standing_ops) > 1:
                report.reject(
                    "invalid-standalone-review",
                    "One staged node patch may edit and review exactly one node.",
                    revision,
                )
                return
            _validate_direct_node_edit(
                state,
                patch,
                update_ops[0],
                report,
                revision,
                mode=mode,
            )
            edits = update_ops[0].nodes
            if standing_ops and edits and standing_ops[0].node_id != edits[0].id:
                report.reject(
                    "invalid-standalone-review",
                    "A staged node edit and review must target the same node.",
                    revision,
                )
            return
        report.reject(
            "invalid-standalone-review",
            "A standalone human patch must review one node or edit one node directly.",
            revision,
        )
        return

    if len(resolution_ops) != 1:
        report.reject(
            "invalid-proposal-resolution", "Resolve one proposal per approval patch.", revision
        )
        return
    resolutions = resolution_ops[0].resolutions
    if len(resolutions) != 1:
        report.reject(
            "invalid-proposal-resolution", "Resolve one proposal per approval patch.", revision
        )
        return
    resolution = resolutions[0]
    proposal = state.proposals.get(resolution.id)
    if proposal is None or proposal.status != "pending":
        report.reject("proposal-not-pending", "The referenced proposal is not pending.", revision)
        return
    status = resolution.status
    is_stale = proposal_is_stale(state, proposal)
    if is_stale and status != "withdrawn":
        report.reject(
            "stale-proposal",
            "A node, project setting, or semantic cause changed after this proposal was written.",
            revision,
        )
    semantic_ops = [
        op
        for op in patch.ops
        if not isinstance(op, (ResolveProposalsOperation, SetStandingOperation))
    ]
    if status == "approved":
        normalized_ops = normalized_decision_proposal_ops(state, proposal)
        proposal_semantic_ops = graph_operations_from_proposal(proposal.ops)
        is_verbatim = semantic_ops == proposal_semantic_ops
        requires_normalization = normalized_ops != proposal_semantic_ops
        if not is_verbatim and semantic_ops != normalized_ops:
            report.reject(
                "proposal-replay-mismatch",
                "Approval must replay the proposal's stored operations, allowing only the "
                "implied decided status for a legacy Decision selection.",
                revision,
            )
        elif is_verbatim and requires_normalization and mode == "admission":
            report.reject(
                "unnormalized-decision-approval",
                "A legacy Decision selection approval must add the implied decided status.",
                revision,
            )
        elif not (is_verbatim and requires_normalization):
            _validate_approved_decision_result(state, semantic_ops, report, revision)
        writes_decision_outcome = _writes_decision_outcome(state, semantic_ops)
        if mode == "admission" and writes_decision_outcome:
            if patch.human_action != "decision_choice":
                report.reject(
                    "unnamed-decision-action",
                    f"Approving a legacy Decision Proposal that writes selected_option or status "
                    f"decided must name the {DECIDE_DECISION} action.",
                    revision,
                )
            elif not permits(patch, DECIDE_DECISION):
                report.reject(
                    "decision-action-refused",
                    f"This actor is not permitted to {DECIDE_DECISION} through a legacy "
                    "Decision Proposal approval.",
                    revision,
                )
        elif patch.human_action == "decision_choice" and not writes_decision_outcome:
            report.reject(
                "invalid-direct-decision-choice",
                "A decision_choice Proposal approval must replay a legacy Decision outcome.",
                revision,
            )
    if status in {"rejected", "withdrawn"} and semantic_ops:
        report.reject(
            "rejected-proposal-has-ops",
            "Rejected or withdrawn proposals cannot apply semantic operations.",
            revision,
        )
    if status == "withdrawn" and not is_stale:
        report.reject(
            "invalid-stale-withdrawal",
            "The human UI may withdraw a proposal here only when its base state is stale.",
            revision,
        )
    elif status not in {"approved", "rejected", "withdrawn"}:
        report.reject(
            "invalid-human-resolution",
            "The human UI may approve or reject a pending proposal.",
            revision,
        )


def _validate_direct_decision_choice(
    state: GraphState,
    patch: Patch,
    resolution_ops: list[ResolveProposalsOperation],
    report: ValidationReport,
    revision: int | None,
) -> None:
    def refuse(message: str, *, node_id: str | None = None) -> None:
        report.reject(
            "invalid-direct-decision-choice",
            message,
            revision,
            related_node_ids=[node_id] if node_id else [],
        )

    if not permits(patch, DECIDE_DECISION):
        refuse(f"Only an actor permitted to {DECIDE_DECISION} may choose a Decision option.")
        return

    names = [op.op for op in patch.ops]
    if not names or set(names) - {"update_nodes", "resolve_proposals", "set_standing"}:
        refuse(
            "A direct Decision choice may only update that Decision, withdraw its Proposals, and review it."
        )
        return
    update_ops = [op for op in patch.ops if isinstance(op, UpdateNodesOperation)]
    standing_ops = [op for op in patch.ops if isinstance(op, SetStandingOperation)]
    if len(update_ops) != 1 or len(standing_ops) > 1:
        refuse("A direct Decision choice must update and optionally review exactly one Decision.")
        return
    operation = update_ops[0]
    updates = operation.nodes
    if len(updates) != 1:
        refuse("A direct Decision choice must update exactly one existing Decision.")
        return
    update = updates[0]
    if update.base_updated_rev is None or update.cause is not None:
        refuse("A direct Decision choice requires exactly id, base_updated_rev, and changes.")
        return
    node = state.nodes.get(update.id)
    if not isinstance(node, Decision):
        refuse(f"Cannot choose an option on non-Decision node {update.id!r}.")
        return
    if node.status == "superseded":
        refuse(f"Decision {node.id} is superseded and cannot be decided again.", node_id=node.id)
    if update.base_updated_rev != node.updated_rev:
        refuse(
            f"{node.id} changed after this choice was staged; reload before saving.",
            node_id=node.id,
        )
    changes = update.changes
    if not changes:
        refuse(
            f"A direct choice for {node.id} must include selected_option and status.",
            node_id=node.id,
        )
        return
    allowed = HUMAN_EDITABLE_NODE_FIELDS[node.type] | {
        "extension_fields",
        "selected_option",
        "status",
    }
    disallowed = sorted(set(changes) - allowed)
    if disallowed:
        refuse(
            f"Direct choice on {node.id} cannot change: {', '.join(disallowed)}.",
            node_id=node.id,
        )
    # A choice that repeats the Decision's existing option carries no
    # `selected_option` change, because only the status moves. Resolve the
    # effective choice against the node so repairing a selected-but-open
    # Decision is expressible.
    selected_option = changes.get("selected_option", node.selected_option)
    options = changes.get("options", node.options)
    if changes.get("status") != "decided":
        refuse(f"Direct choice on {node.id} must set status exactly to decided.", node_id=node.id)
    if (
        not isinstance(selected_option, str)
        or not selected_option.strip()
        or not isinstance(options, list)
        or selected_option not in options
    ):
        refuse(
            f"Direct choice on {node.id} must select one non-empty option from its current options.",
            node_id=node.id,
        )
    if standing_ops and standing_ops[0].node_id != node.id:
        refuse(
            "A direct Decision choice and its staged judgment must target the same Decision.",
            node_id=node.id,
        )

    seen_proposals: set[str] = set()
    for operation in resolution_ops:
        resolutions = operation.resolutions
        if not resolutions:
            refuse(f"Proposal withdrawals for {node.id} cannot be empty.", node_id=node.id)
            continue
        for resolution in resolutions:
            proposal_id = resolution.id
            proposal = state.proposals.get(proposal_id)
            if proposal_id in seen_proposals:
                refuse(
                    f"Proposal {proposal_id!r} is withdrawn more than once for {node.id}.",
                    node_id=node.id,
                )
            elif proposal is None or proposal.status != "pending":
                refuse(
                    f"Proposal {proposal_id!r} for {node.id} is not pending.",
                    node_id=node.id,
                )
            elif not proposal_updates_node(proposal, node.id):
                refuse(
                    f"Proposal {proposal_id!r} does not target Decision {node.id}.",
                    node_id=node.id,
                )
            seen_proposals.add(proposal_id)
            if resolution.status != "withdrawn":
                refuse(
                    f"Direct choice on {node.id} may only withdraw superseded Proposals.",
                    node_id=node.id,
                )
            if not isinstance(resolution.reason, str) or not resolution.reason.strip():
                refuse(
                    f"Withdrawal of Proposal {proposal_id!r} for {node.id} requires a reason.",
                    node_id=node.id,
                )
    expected_proposals = {
        proposal.id
        for proposal in state.proposals.values()
        if proposal.status == "pending" and proposal_updates_node(proposal, node.id)
    }
    if seen_proposals != expected_proposals:
        missing = sorted(expected_proposals - seen_proposals)
        unexpected = sorted(seen_proposals - expected_proposals)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        refuse(
            f"Direct choice on {node.id} must withdraw every pending Proposal targeting it "
            f"({'; '.join(details)}).",
            node_id=node.id,
        )


def _validate_approved_decision_result(
    state: GraphState,
    semantic_ops: list[GraphOperation],
    report: ValidationReport,
    revision: int | None,
) -> None:
    for operation in semantic_ops:
        if not isinstance(operation, UpdateNodesOperation):
            continue
        for update in operation.nodes:
            node = state.nodes.get(update.id)
            changes = update.changes
            if isinstance(node, Decision) and (error := decision_transition_error(node, changes)):
                report.reject(
                    "incoherent-decision-approval",
                    error,
                    revision,
                    related_node_ids=[node.id],
                )


def _writes_decision_outcome(state: GraphState, operations: list[GraphOperation]) -> bool:
    return any(
        isinstance(state.nodes.get(update.id), Decision)
        and ("selected_option" in update.changes or update.changes.get("status") == "decided")
        for operation in operations
        if isinstance(operation, UpdateNodesOperation)
        for update in operation.nodes
    )


def _validate_attempt_release(
    node: Any,
    raw_attempts: Any,
    report: ValidationReport,
    revision: int | None,
) -> None:
    """The one non-prose direct edit: releasing an attempt nobody can close.

    A watcher that can no longer answer leaves its attempt open forever, and an
    open attempt keeps the experiment un-runnable. So the human may move an open
    attempt to `cancelled` — and nothing else. A finished attempt is a record.
    """

    def refuse(message: str) -> None:
        report.reject("invalid-attempt-release", message, revision, related_node_ids=[node.id])

    previous = getattr(node, "attempts", None)
    if previous is None:
        refuse(f"{node.id} has no attempts to release.")
        return
    if not isinstance(raw_attempts, list) or len(raw_attempts) != len(previous):
        refuse(f"Releasing an attempt on {node.id} cannot add or remove attempt records.")
        return
    released = 0
    for before, after in zip(previous, raw_attempts, strict=True):
        if not isinstance(after, dict):
            refuse(f"Releasing an attempt on {node.id} requires whole attempt records.")
            return
        unchanged = before.model_dump(mode="json")
        if after == unchanged:
            continue
        expected = {
            **unchanged,
            "status": "cancelled",
            "finished_at": after.get("finished_at"),
            "failure_reason": after.get("failure_reason"),
        }
        if (
            before.status not in ACTIVE_EXPERIMENT_ATTEMPT_STATUSES
            or after != expected
            or after.get("finished_at") is None
        ):
            refuse(
                f"A human release on {node.id} may only move an open attempt to cancelled with "
                "its finish time."
            )
            return
        released += 1
    if released == 0:
        refuse(f"No open attempt on {node.id} was released.")


def _validate_direct_node_edit(
    state: GraphState,
    patch: Patch,
    operation: UpdateNodesOperation,
    report: ValidationReport,
    revision: int | None,
    *,
    mode: Literal["admission", "replay"],
) -> None:
    updates = operation.nodes
    if len(updates) != 1:
        report.reject(
            "invalid-direct-node-edit",
            "A direct node edit must update exactly one existing node.",
            revision,
        )
        return
    update = updates[0]
    if update.base_updated_rev is None or update.cause is not None:
        report.reject(
            "invalid-direct-node-edit",
            "A direct node edit requires exactly id, base_updated_rev, and changes.",
            revision,
        )
    node = state.nodes.get(update.id)
    if node is None:
        report.reject(
            "unknown-node",
            f"Cannot update missing node {update.id!r}.",
            revision,
        )
        return
    base_updated_rev = update.base_updated_rev
    if base_updated_rev != node.updated_rev:
        report.reject(
            "stale-node-edit",
            f"{node.id} changed after this editor opened; reload it before saving.",
            revision,
        )
    changes = update.changes
    if not changes:
        report.reject(
            "empty-node-edit",
            "A direct node edit must change at least one editable field.",
            revision,
        )
        return
    if (
        mode == "admission"
        and isinstance(node, Decision)
        and ("selected_option" in changes or changes.get("status") == "decided")
    ):
        action = DECIDE_DECISION
        permitted = permits(patch, action)
        report.reject(
            "unnamed-decision-action",
            f"An ordinary node edit may queue Decision {node.id}, but only a patch naming "
            f"human_action='decision_choice' may {action}."
            if permitted
            else f"This actor is not permitted to {action} on Decision {node.id}.",
            revision,
            related_node_ids=[node.id],
        )
    if set(changes) == {"attempts"}:
        _validate_attempt_release(node, changes["attempts"], report, revision)
        return
    allowed = HUMAN_EDITABLE_NODE_FIELDS[node.type] | {"extension_fields"}
    disallowed = sorted(set(changes) - allowed)
    if disallowed:
        report.reject(
            "non-prose-node-edit",
            f"Direct edits to {node.id} cannot change: {', '.join(disallowed)}.",
            revision,
        )
    if all(getattr(node, field, object()) == value for field, value in changes.items()):
        report.reject(
            "empty-node-edit",
            "The submitted node edit is unchanged.",
            revision,
        )
