from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import ValidationError

from rcp.core.authority import EVIDENCE_RELATIONS
from rcp.core.models import (
    RELATION_SPEC,
    Decision,
    Experiment,
    ExperimentAttempt,
    ExperimentDecisionPin,
    GraphState,
    Patch,
    Proposal,
)
from rcp.core.ontology import custom_relation
from rcp.core.operations import (
    CreateEdgesOperation,
    CreateNodesOperation,
    CreateProposalsOperation,
    EvidenceEdgeCause,
    GraphOperation,
    ProposalContentChangeOperation,
    ProposalOperation,
    ProposalStatusChangeOperation,
    UpdateNodesOperation,
)
from rcp.core.validation.constants import LEGACY_COMPATIBILITY_UPDATE_FIELDS
from rcp.core.validation.report import ValidationReport

_ATTEMPT_CLOSE_FIELDS = frozenset(
    {"status", "source_refs", "outcome", "failure_reason", "finished_at"}
)
_TERMINAL_ATTEMPT_STATUSES = frozenset({"failed", "completed", "cancelled", "superseded"})


def validate_experiment_loop_authority(
    state: GraphState,
    patch: Patch,
    report: ValidationReport,
    *,
    control_node_id: str | None,
    decision_bundle: Iterable[ExperimentDecisionPin],
    mode: Literal["admission", "replay"] = "admission",
) -> None:
    revision = patch.revision or None
    experiment = state.nodes.get(control_node_id) if control_node_id else None
    if not isinstance(experiment, Experiment):
        report.reject(
            "experiment-loop-control-node",
            "An experiment-loop patch requires an RCP-bound Experiment node.",
            revision,
            related_node_ids=[control_node_id] if control_node_id else [],
        )
        return

    pinned = list(decision_bundle)
    pinned_ids = [item.decision_id for item in pinned]
    if len(pinned_ids) != len(set(pinned_ids)):
        report.reject(
            "experiment-loop-decision-bundle",
            "The RCP-bound governing decision bundle contains duplicate decisions.",
            revision,
            related_node_ids=[experiment.id, *pinned_ids],
        )

    has_proposals = any(
        isinstance(op, CreateProposalsOperation) and bool(op.proposals) for op in patch.ops
    )
    created_types = _created_node_types(patch.ops)
    for op in patch.ops:
        if isinstance(op, UpdateNodesOperation):
            _validate_updates(
                state,
                op,
                experiment,
                pinned,
                has_proposals,
                report,
                revision,
                allowed_experiment_fields=_allowed_experiment_fields(patch, mode),
            )
        elif isinstance(op, CreateNodesOperation):
            _validate_created_nodes(op, experiment.id, report, revision)
        elif isinstance(op, CreateEdgesOperation):
            _validate_created_edges(state, op, experiment.id, created_types, report, revision)
        elif isinstance(op, CreateProposalsOperation):
            _validate_proposals(
                op,
                experiment.id,
                _tested_hypothesis_ids(state, experiment.id),
                _grounding_edge_ids(state, patch.ops, created_types),
                report,
                revision,
            )
        else:
            report.reject(
                "experiment-loop-operation",
                f"Experiment loop {experiment.id} cannot use operation {op.op!r}.",
                revision,
                related_node_ids=[experiment.id],
            )


def _allowed_experiment_fields(
    patch: Patch,
    mode: Literal["admission", "replay"],
) -> frozenset[str]:
    """The fields an Experiment-loop Patch may change on its own control node.

    Loading a pre-generation-2 Patch retires `blocked` in memory and marks the
    guidance it invalidates, so replay sees system fields the original write never
    contained. Refusing them there halts canonical history over RCP's own
    migration; a live write still may not set them.
    """

    allowed = frozenset({"attempts", "status", "current_summary", "next_action"})
    if mode == "replay" and patch.schema_generation == 1:
        return allowed | LEGACY_COMPATIBILITY_UPDATE_FIELDS
    return allowed


def _validate_updates(
    state: GraphState,
    op: UpdateNodesOperation,
    experiment: Experiment,
    pinned: list[ExperimentDecisionPin],
    has_proposals: bool,
    report: ValidationReport,
    revision: int | None,
    *,
    allowed_experiment_fields: frozenset[str],
) -> None:
    pinned_ids = {item.decision_id for item in pinned}
    for update in op.nodes:
        node_id = update.id
        changes = update.changes
        if node_id in pinned_ids:
            if set(changes) != {"status"} or changes.get("status") not in {
                "open",
                "ready",
                "revisit",
            }:
                report.reject(
                    "experiment-loop-decision-action",
                    f"Experiment loop {experiment.id} may only queue pinned Decision {node_id!r} "
                    "as open, ready, or revisit; it may never decide it.",
                    revision,
                    related_node_ids=[experiment.id, node_id],
                )
            elif not isinstance(state.nodes.get(node_id), Decision):
                report.reject(
                    "experiment-loop-foreign-update",
                    f"Experiment loop {experiment.id} cannot queue missing or non-Decision "
                    f"node {node_id!r}.",
                    revision,
                    related_node_ids=[experiment.id, node_id],
                )
            continue
        if node_id != experiment.id:
            report.reject(
                "experiment-loop-foreign-update",
                f"Experiment loop {experiment.id} cannot update node {node_id!r}.",
                revision,
                related_node_ids=[
                    item for item in (experiment.id, node_id) if isinstance(item, str)
                ],
            )
            continue
        forbidden = sorted(set(changes) - allowed_experiment_fields)
        if forbidden:
            report.reject(
                "experiment-loop-experiment-field",
                f"Experiment loop {experiment.id} may update only status, attempts, "
                f"current_summary, and next_action; refused: {', '.join(forbidden)}.",
                revision,
                related_node_ids=[experiment.id],
            )
        if "attempts" in changes:
            _validate_attempts(
                experiment,
                changes["attempts"],
                pinned,
                has_proposals,
                report,
                revision,
            )


def _validate_attempts(
    experiment: Experiment,
    raw_attempts: Any,
    pinned: list[ExperimentDecisionPin],
    has_proposals: bool,
    report: ValidationReport,
    revision: int | None,
) -> None:
    if not isinstance(raw_attempts, list):
        return
    try:
        attempts = [ExperimentAttempt.model_validate(item) for item in raw_attempts]
    except ValidationError:
        return

    previous = experiment.attempts
    if len(attempts) < len(previous):
        report.reject(
            "experiment-loop-attempt-removal",
            f"Experiment loop {experiment.id} cannot remove attempt records.",
            revision,
            related_node_ids=[experiment.id],
        )
        return

    for before, after in zip(previous, attempts, strict=False):
        before_fixed = before.model_dump(mode="python", exclude=_ATTEMPT_CLOSE_FIELDS)
        after_fixed = after.model_dump(mode="python", exclude=_ATTEMPT_CLOSE_FIELDS)
        if before_fixed != after_fixed:
            report.reject(
                "experiment-loop-attempt-mutation",
                f"Experiment loop {experiment.id} cannot rewrite attempt {before.id!r} or "
                "its pinned decision bundle.",
                revision,
                related_node_ids=[experiment.id],
            )
            continue
        if before != after and (
            before.status in _TERMINAL_ATTEMPT_STATUSES
            or after.status not in _TERMINAL_ATTEMPT_STATUSES
        ):
            report.reject(
                "experiment-loop-attempt-close",
                f"Experiment loop {experiment.id} may only close a nonterminal attempt.",
                revision,
                related_node_ids=[experiment.id],
            )

    expected_bundle = [item.model_dump(mode="python") for item in pinned]
    appended = attempts[len(previous) :]
    for attempt in appended:
        actual_bundle = [item.model_dump(mode="python") for item in attempt.decision_bundle]
        if actual_bundle != expected_bundle:
            report.reject(
                "experiment-loop-pinned-bundle",
                f"New attempt {attempt.id!r} must copy the RCP-pinned governing decisions.",
                revision,
                related_node_ids=[experiment.id, *[item.decision_id for item in pinned]],
            )
        if attempt.attempt_kind == "proposal_only" and attempt.job_refs:
            report.reject(
                "experiment-loop-proposal-job",
                f"Proposal-only attempt {attempt.id!r} cannot name an external job.",
                revision,
                related_node_ids=[experiment.id],
            )
        if (
            attempt.attempt_kind == "proposal_only"
            and attempt.status not in _TERMINAL_ATTEMPT_STATUSES
        ):
            report.reject(
                "experiment-loop-proposal-status",
                f"Proposal-only attempt {attempt.id!r} must be terminal in the turn that creates it.",
                revision,
                related_node_ids=[experiment.id],
            )

    if any(attempt.attempt_kind == "proposal_only" for attempt in appended) and not has_proposals:
        report.reject(
            "experiment-loop-proposal-attempt",
            f"Proposal-only attempt in experiment {experiment.id} requires a proposal.",
            revision,
            related_node_ids=[experiment.id],
        )

    ids = [attempt.id for attempt in attempts]
    if len(ids) != len(set(ids)):
        report.reject(
            "experiment-loop-duplicate-attempt",
            f"Experiment {experiment.id} cannot contain duplicate attempt ids.",
            revision,
            related_node_ids=[experiment.id],
        )


def _validate_created_nodes(
    op: CreateNodesOperation,
    experiment_id: str,
    report: ValidationReport,
    revision: int | None,
) -> None:
    for node in op.nodes:
        if node.type not in {"evidence", "blocker"}:
            report.reject(
                "experiment-loop-created-node",
                f"Experiment loop {experiment_id} may create only Evidence or Blocker nodes.",
                revision,
                related_node_ids=[experiment_id],
            )


def _created_node_types(ops: Iterable[GraphOperation]) -> dict[str, str]:
    created: dict[str, str] = {}
    for op in ops:
        if not isinstance(op, CreateNodesOperation):
            continue
        for node in op.nodes:
            created[node.id] = node.type
    return created


# The loop may attach its own output to its own experiment, and nothing else.
# `produces` is provenance and `blocked_by` is self-blocking, so neither widens
# its authority; both targets must be nodes this same patch created.
_SELF_ATTACHMENT_RELATIONS = {"produces": "evidence", "blocked_by": "blocker"}
_EVIDENCE_HANDOFF_RELATIONS = {"informs": "decision", "addresses": "blocker"}


def _validate_created_edges(
    state: GraphState,
    op: CreateEdgesOperation,
    experiment_id: str,
    created_types: dict[str, str],
    report: ValidationReport,
    revision: int | None,
) -> None:
    for edge in op.edges:
        relation_name = edge.relation
        if relation_name in _SELF_ATTACHMENT_RELATIONS:
            expected_type = _SELF_ATTACHMENT_RELATIONS[relation_name]
            target = edge.target
            if edge.source == experiment_id and created_types.get(target) == expected_type:
                continue
            report.reject(
                "experiment-loop-self-attachment",
                f"Experiment loop {experiment_id} may use {relation_name!r} only from its own "
                f"experiment to a {expected_type} node this patch creates.",
                revision,
                related_node_ids=[
                    experiment_id,
                    target,
                ],
            )
            continue
        if relation_name in _EVIDENCE_HANDOFF_RELATIONS:
            source = edge.source
            target = edge.target
            target_node = state.nodes.get(target)
            target_type = target_node.type if target_node is not None else created_types.get(target)
            expected_target_type = _EVIDENCE_HANDOFF_RELATIONS[relation_name]
            if created_types.get(source) == "evidence" and target_type == expected_target_type:
                continue
            report.reject(
                "experiment-loop-evidence-handoff",
                f"Experiment loop {experiment_id} may use {relation_name!r} only from Evidence "
                f"this patch creates to a {expected_target_type} node.",
                revision,
                related_node_ids=[
                    experiment_id,
                    source,
                    target,
                ],
            )
            continue
        base = RELATION_SPEC.get(relation_name)
        relation = custom_relation(state.ontology, relation_name)
        layer = base.layer if base is not None else relation.layer if relation is not None else None
        if layer != "epistemic":
            report.reject(
                "experiment-loop-edge-layer",
                f"Experiment loop {experiment_id} may assert only epistemic edges, attach its "
                "own evidence and blockers to its experiment, or hand its new Evidence to a "
                "Decision or Blocker.",
                revision,
                related_node_ids=[experiment_id],
            )


def _tested_hypothesis_ids(state: GraphState, experiment_id: str) -> set[str]:
    return {
        edge.target
        for edge in state.edges.values()
        if edge.source == experiment_id and edge.relation == "tests"
    }


def _grounding_edge_ids(
    state: GraphState,
    ops: Iterable[GraphOperation],
    created_types: dict[str, str],
) -> dict[str, set[str]]:
    """Same-patch Evidence -> Hypothesis edge ids grouped by target.

    A belief proposal must rest on evidence the same turn recorded, so the human
    is never asked to move a belief the patch supplies no reason for.
    """

    grounded: dict[str, set[str]] = {}
    for op in ops:
        if not isinstance(op, CreateEdgesOperation):
            continue
        for edge in op.edges:
            target = edge.target
            source = edge.source
            existing_source = state.nodes.get(source)
            source_type = (
                existing_source.type if existing_source is not None else created_types.get(source)
            )
            if edge.relation in EVIDENCE_RELATIONS and source_type == "evidence":
                edge_id = edge.id or f"{source}::{edge.relation}::{target}"
                grounded.setdefault(target, set()).add(edge_id)
    return grounded


def _validate_proposals(
    op: CreateProposalsOperation,
    experiment_id: str,
    tested_hypothesis_ids: set[str],
    grounding_edge_ids: dict[str, set[str]],
    report: ValidationReport,
    revision: int | None,
) -> None:
    """Admit the one belief Proposal shape a loop may raise, and nothing else.

    A belief Proposal asks the human to accept the belief change its own evidence
    implies. The loop may queue a pinned Decision directly, but never propose or
    apply a Decision outcome.
    """

    for proposal in op.proposals:
        target_ids = _proposal_update_targets(proposal.ops)

        if target_ids and target_ids <= tested_hypothesis_ids:
            _validate_belief_proposal(
                proposal,
                experiment_id,
                target_ids,
                grounding_edge_ids,
                report,
                revision,
            )
            continue

        report.reject(
            "experiment-loop-proposal-operations",
            f"Experiment loop {experiment_id} proposals may update only one tested Hypothesis.",
            revision,
            related_node_ids=[experiment_id, *sorted(target_ids)],
        )


def _proposal_update_targets(replay_ops: list[ProposalOperation]) -> set[str]:
    """Node ids a proposal's replay would update, or empty when it is malformed."""

    if not replay_ops:
        return set()
    targets: set[str] = set()
    for replay_op in replay_ops:
        if not isinstance(
            replay_op, (ProposalContentChangeOperation, ProposalStatusChangeOperation)
        ):
            return set()
        if not replay_op.nodes:
            return set()
        targets.update(item.id for item in replay_op.nodes)
    return targets


def _validate_belief_proposal(
    proposal: Proposal,
    experiment_id: str,
    target_ids: set[str],
    grounding_edge_ids: dict[str, set[str]],
    report: ValidationReport,
    revision: int | None,
) -> None:
    def refuse(code: str, message: str) -> None:
        report.reject(
            code,
            message,
            revision,
            related_node_ids=[experiment_id, *sorted(target_ids)],
        )

    if len(target_ids) != 1:
        refuse(
            "experiment-loop-belief-proposal-scope",
            f"Experiment loop {experiment_id} may propose one belief change at a time.",
        )
        return
    hypothesis_id = next(iter(target_ids))
    if hypothesis_id not in grounding_edge_ids:
        refuse(
            "experiment-loop-belief-grounding",
            f"A belief proposal from {experiment_id} must rest on an evidence edge this patch "
            f"asserts into {hypothesis_id}.",
        )

    updates = [
        item
        for replay_op in proposal.ops
        if isinstance(replay_op, (ProposalContentChangeOperation, ProposalStatusChangeOperation))
        for item in replay_op.nodes
    ]
    for update in updates:
        changes = update.changes
        if set(changes) != {"status"}:
            refuse(
                "experiment-loop-belief-proposal-operations",
                f"A belief proposal from {experiment_id} may change only {hypothesis_id}'s status.",
            )
            continue
        cause = update.cause
        if not isinstance(cause, EvidenceEdgeCause) or cause.ref_id not in grounding_edge_ids.get(
            hypothesis_id, set()
        ):
            refuse(
                "experiment-loop-belief-cause",
                f"A belief proposal from {experiment_id} must name one of this patch's "
                f"Evidence edges into {hypothesis_id} as its cause.",
            )
