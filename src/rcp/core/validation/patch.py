"""Patch-level validation: the rules that judge a patch as a whole.

Operation-level rules live in :mod:`rcp.core.validation.ops` and are reached
only through :data:`rcp.core.validation.registry.OP_RULES`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from rcp.core.authority import operation_actions, permits
from rcp.core.models import Decision, ExperimentDecisionPin, GraphState, Patch
from rcp.core.operations import (
    CreateEdgesOperation,
    CreateNodesOperation,
    CreateProposalsOperation,
    GraphOperation,
    MergeNodesOperation,
    RemoveEdgesOperation,
    RemoveNodesOperation,
    SetOntologyOperation,
    SetProjectTruthScopeOperation,
    SetStandingOperation,
    SupersedeNodesOperation,
    UpdateNodesOperation,
)
from rcp.core.validation.approval import validate_approval_shape
from rcp.core.validation.context import OpContext
from rcp.core.validation.experiment_loop import validate_experiment_loop_authority
from rcp.core.validation.nodes import older
from rcp.core.validation.proposals import proposal_is_stale
from rcp.core.validation.registry import OP_RULES
from rcp.core.validation.report import ValidationReport


def validate_patch(
    state: GraphState,
    patch: Patch,
    project_truth_scope: Iterable[str],
    repository_aliases: Iterable[str] | None = None,
    machine_aliases: Iterable[str] | None = None,
    default_run_truth_scope: Iterable[str] | None = None,
    state_repository: str | None = None,
    mode: Literal["admission", "replay"] = "admission",
    *,
    experiment_control_node_id: str | None = None,
    experiment_decision_bundle: Iterable[ExperimentDecisionPin] | None = None,
) -> ValidationReport:
    if patch.transition is not None:
        return _validate_transition_patch(
            state,
            patch,
            project_truth_scope,
            repository_aliases=repository_aliases,
            machine_aliases=machine_aliases,
            default_run_truth_scope=default_run_truth_scope,
            state_repository=state_repository,
            mode=mode,
        )
    try:
        return _validate_patch(
            state,
            patch,
            project_truth_scope,
            repository_aliases=repository_aliases,
            machine_aliases=machine_aliases,
            default_run_truth_scope=default_run_truth_scope,
            state_repository=state_repository,
            mode=mode,
            experiment_control_node_id=experiment_control_node_id,
            experiment_decision_bundle=experiment_decision_bundle,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        report = ValidationReport()
        report.reject(
            "malformed-operation",
            f"Patch operations are malformed: {exc}.",
            patch.revision or None,
        )
        return report


def _validate_transition_patch(
    state: GraphState,
    patch: Patch,
    project_truth_scope: Iterable[str],
    *,
    repository_aliases: Iterable[str] | None,
    machine_aliases: Iterable[str] | None,
    default_run_truth_scope: Iterable[str] | None,
    state_repository: str | None,
    mode: Literal["admission", "replay"],
) -> ValidationReport:
    report = ValidationReport()
    if mode != "replay":
        report.reject(
            "reserved-transition-trace",
            "A committed transition trace is backend-owned and cannot be supplied for admission.",
            patch.revision or None,
        )
        return report
    try:
        from rcp.core.materialize import apply_valid_patch
        from rcp.core.transitions import validate_transition_trace

        source_patches = validate_transition_trace(state, patch)
        staged = state
        for source_patch in source_patches:
            validation_state = staged.model_copy(update={"revision": state.revision})
            source_report = validate_patch(
                validation_state,
                source_patch,
                validation_state.project_truth_scope,
                repository_aliases=repository_aliases,
                machine_aliases=machine_aliases,
                default_run_truth_scope=default_run_truth_scope,
                state_repository=state_repository,
                mode="replay",
            )
            report.messages.extend(source_report.messages)
            if source_report.rejected:
                break
            staged = apply_valid_patch(validation_state, source_patch).model_copy(
                update={"revision": state.revision}
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        report.reject(
            "transition-trace-invalid",
            f"Committed transition provenance is invalid: {exc}.",
            patch.revision or None,
        )
    return report


def _validate_patch(
    state: GraphState,
    patch: Patch,
    project_truth_scope: Iterable[str],
    repository_aliases: Iterable[str] | None = None,
    machine_aliases: Iterable[str] | None = None,
    default_run_truth_scope: Iterable[str] | None = None,
    state_repository: str | None = None,
    mode: Literal["admission", "replay"] = "admission",
    *,
    experiment_control_node_id: str | None = None,
    experiment_decision_bundle: Iterable[ExperimentDecisionPin] | None = None,
) -> ValidationReport:
    report = ValidationReport()
    scope = set(project_truth_scope)
    control_node_id = experiment_control_node_id or patch.experiment_control_node_id
    decision_bundle = tuple(
        experiment_decision_bundle
        if experiment_decision_bundle is not None
        else patch.experiment_decision_bundle
    )
    ctx = OpContext(
        state=state,
        initial_state=state,
        patch=patch,
        report=report,
        revision=patch.revision or None,
        project_truth_scope=scope,
        repositories=set(repository_aliases or scope),
        machines=set(machine_aliases) if machine_aliases is not None else None,
        default_run_truth_scope=set(default_run_truth_scope or ()),
        state_repository=state_repository,
        mode=mode,
        experiment_control_node_id=control_node_id,
    )

    if mode == "admission" and patch.schema_generation != 2:
        report.reject(
            "legacy-schema-write",
            "New graph writes must use the current schema generation.",
            ctx.revision,
        )

    if patch.revision and patch.revision != state.revision + 1:
        report.reject(
            "non-monotonic-revision",
            f"Patch revision {patch.revision} must follow graph revision {state.revision}.",
            ctx.revision,
        )
    _validate_authorship(ctx)
    _validate_identity_shape(ctx)
    _validate_attribution_shape(ctx)
    _validate_declared_agent_action(ctx)
    _validate_declared_scope(ctx)

    if patch.kind == "experiment_loop":
        validate_experiment_loop_authority(
            state,
            patch,
            report,
            control_node_id=control_node_id,
            decision_bundle=decision_bundle,
            mode=mode,
        )
    elif control_node_id or decision_bundle:
        report.reject(
            "unexpected-experiment-control",
            "Experiment control metadata is legal only on experiment-loop patches.",
            ctx.revision,
        )

    op_names = [op.op for op in patch.ops]
    if any(name.startswith("delete") for name in op_names):
        report.reject("delete-forbidden", "Graph objects are never deleted.", ctx.revision)

    if patch.kind == "approval":
        validate_approval_shape(state, patch, report, mode=mode)

    oldest_ref = _validate_operations(ctx)
    _validate_queued_decision_options(ctx)
    _validate_created_proposal_liveness(ctx)

    if (
        mode == "admission"
        and oldest_ref is not None
        and state.coverage.earliest_timestamp is not None
        and oldest_ref < state.coverage.earliest_timestamp
        and "set_coverage" not in op_names
    ):
        report.flag(
            "coverage-mismatch",
            "This patch cites history older than the graph's coverage boundary without updating coverage.",
            ctx.revision,
        )

    return report


def _validate_created_proposal_liveness(ctx: OpContext) -> None:
    proposal_positions = {
        proposal.id: index
        for index, operation in enumerate(ctx.patch.ops)
        if isinstance(operation, CreateProposalsOperation)
        for proposal in operation.proposals
    }
    for proposal_id, position in sorted(proposal_positions.items()):
        proposal = ctx.state.proposals.get(proposal_id)
        moved_nodes: set[str] = set()
        moved_edges: set[str] = set()
        moved_config: set[str] = set()
        if ctx.mode == "admission" and proposal is not None:
            moved_nodes, moved_edges, moved_config = _later_dependency_mutations(
                ctx,
                position,
                proposal.related_node_ids,
                proposal.related_edge_ids,
                proposal.related_config_keys,
            )
        if moved_nodes or moved_edges or moved_config:
            moved = sorted(moved_nodes | moved_edges | moved_config)
            ctx.report.reject(
                "stale-created-proposal",
                f"Proposal {proposal_id!r} is already stale because a snapshotted dependency "
                f"moved later in its outer patch: {', '.join(moved)}.",
                ctx.revision,
                related_node_ids=sorted(moved_nodes),
                related_edge_ids=sorted(moved_edges),
            )
            continue
        if proposal is not None and proposal_is_stale(ctx.state, proposal):
            ctx.report.reject(
                "stale-created-proposal",
                f"Proposal {proposal_id!r} is already stale after applying its outer patch.",
                ctx.revision,
                related_node_ids=list(proposal.related_node_ids),
            )


def _later_dependency_mutations(
    ctx: OpContext,
    position: int,
    related_node_ids: Iterable[str],
    related_edge_ids: Iterable[str],
    related_config_keys: Iterable[str],
) -> tuple[set[str], set[str], set[str]]:
    related_nodes = set(related_node_ids)
    related_edges = set(related_edge_ids)
    related_config = set(related_config_keys)
    present_nodes = set(ctx.initial_state.nodes)
    present_edges = {
        edge.id: (edge.source, edge.target) for edge in ctx.initial_state.edges.values()
    }
    for operation in ctx.patch.ops[:position]:
        _update_resource_presence(present_nodes, present_edges, operation)

    later = ctx.patch.ops[position + 1 :]
    created_nodes = {
        raw.id
        for operation in later
        if isinstance(operation, CreateNodesOperation)
        for raw in operation.nodes
    }
    changed_nodes = {
        raw.id
        for operation in later
        if isinstance(operation, (UpdateNodesOperation, SupersedeNodesOperation))
        for raw in operation.nodes
    }
    changed_nodes.update(
        raw.duplicate
        for operation in later
        if isinstance(operation, MergeNodesOperation)
        for raw in operation.merges
    )
    changed_nodes.update(
        node_id
        for operation in later
        if isinstance(operation, RemoveNodesOperation)
        for node_id in operation.node_ids
    )
    changed_nodes.update(
        operation.node_id for operation in later if isinstance(operation, SetStandingOperation)
    )
    created_edges = {edge_id for operation in later for edge_id in _created_edges(operation)}
    removed_edges = {
        edge_id
        for operation in later
        if isinstance(operation, RemoveEdgesOperation)
        for edge_id in operation.edge_ids
    }
    moved_nodes = related_nodes & (changed_nodes | (created_nodes & present_nodes))
    moved_edges = related_edges & (removed_edges | (created_edges & set(present_edges)))
    moved_config: set[str] = set()
    if "ontology" in related_config and any(isinstance(op, SetOntologyOperation) for op in later):
        moved_config.add("ontology")
    if "project_truth_scope" in related_config and any(
        isinstance(op, SetProjectTruthScopeOperation) for op in later
    ):
        moved_config.add("project_truth_scope")
    return moved_nodes, moved_edges, moved_config


def _update_resource_presence(
    nodes: set[str], edges: dict[str, tuple[str, str]], operation: GraphOperation
) -> None:
    if isinstance(operation, CreateNodesOperation):
        nodes.update(raw.id for raw in operation.nodes)
    elif isinstance(operation, RemoveNodesOperation):
        removed = set(operation.node_ids)
        nodes.difference_update(removed)
        for edge_id, endpoints in list(edges.items()):
            if any(node_id in removed for node_id in endpoints):
                edges.pop(edge_id)
    elif isinstance(operation, RemoveEdgesOperation):
        for edge_id in operation.edge_ids:
            edges.pop(edge_id, None)
    edges.update(_created_edges(operation))


def _created_edges(operation: GraphOperation) -> dict[str, tuple[str, str]]:
    created: dict[str, tuple[str, str]] = {}
    if isinstance(operation, CreateEdgesOperation):
        values = [(raw.id, raw.source, raw.relation, raw.target) for raw in operation.edges]
    elif isinstance(operation, SupersedeNodesOperation):
        values = [
            (None, raw.id, "supersedes", raw.superseded_by)
            for raw in operation.nodes
            if raw.superseded_by is not None
        ]
    elif isinstance(operation, MergeNodesOperation):
        values = [(None, raw.duplicate, "duplicate_of", raw.canonical) for raw in operation.merges]
    else:
        values = []
    for edge_id, source, relation, target in values:
        edge_id = edge_id or f"{source}::{relation}::{target}"
        created[edge_id] = (source, target)
    return created


def _validate_queued_decision_options(ctx: OpContext) -> None:
    """Check queued ballots after the Patch's written-order staging has finished."""

    if ctx.mode != "admission":
        return
    touched_ids = {
        raw.id
        for operation in ctx.patch.ops
        if isinstance(
            operation, (CreateNodesOperation, UpdateNodesOperation, SupersedeNodesOperation)
        )
        for raw in operation.nodes
    }
    touched_ids.update(
        raw.duplicate
        for operation in ctx.patch.ops
        if isinstance(operation, MergeNodesOperation)
        for raw in operation.merges
    )
    touched_ids.update(
        operation.node_id
        for operation in ctx.patch.ops
        if isinstance(operation, SetStandingOperation)
    )
    for node_id in sorted(touched_ids):
        node = ctx.state.nodes.get(node_id)
        if (
            isinstance(node, Decision)
            and node.status in {"ready", "revisit"}
            and len(set(node.options)) < 2
        ):
            ctx.report.reject(
                "incomplete-decision-ballot",
                f"Decision {node.id} must have at least two distinct options before it can be "
                f"queued as {node.status}.",
                ctx.revision,
                related_node_ids=[node.id],
            )


def _validate_authorship(ctx: OpContext) -> None:
    if ctx.patch.kind == "identity":
        if ctx.patch.producer != "system":
            ctx.report.reject(
                "wrong-producer",
                "Identity patches must be produced by RCP's system producer.",
                ctx.revision,
            )
        if ctx.patch.author is not None:
            ctx.report.reject(
                "wrong-author",
                "Identity patches have no human or agent author.",
                ctx.revision,
            )
        return

    expected_author = "human" if ctx.patch.kind == "approval" else "agent"
    if ctx.patch.author != expected_author:
        ctx.report.reject(
            "wrong-author",
            f"{ctx.patch.kind} patches must be authored by {expected_author}.",
            ctx.revision,
        )
    if ctx.patch.producer == "system":
        ctx.report.reject(
            "system-producer-forbidden",
            "The system producer is reserved for identity patches.",
            ctx.revision,
        )
    elif ctx.patch.producer != ctx.patch.author:
        ctx.report.reject(
            "producer-author-mismatch",
            "Human and agent patches must retain the same producer and legacy author role.",
            ctx.revision,
        )


def _validate_identity_shape(ctx: OpContext) -> None:
    patch = ctx.patch
    if patch.kind != "identity":
        if patch.project_identity is not None:
            ctx.report.reject(
                "unexpected-project-identity",
                "Project identity is legal only on identity patches.",
                ctx.revision,
            )
        if patch.project_home_transfer is not None:
            ctx.report.reject(
                "unexpected-project-home-transfer",
                "Project home transfer is legal only on identity patches.",
                ctx.revision,
            )
        return

    if patch.project_identity is None and patch.project_home_transfer is None:
        ctx.report.reject(
            "missing-project-identity",
            "Identity patches require exactly one project identity payload.",
            ctx.revision,
        )
    elif patch.project_identity is not None and patch.project_home_transfer is not None:
        ctx.report.reject(
            "conflicting-identity-payloads",
            "Identity patches cannot combine a project identity and home transfer.",
            ctx.revision,
        )
    if patch.ops:
        ctx.report.reject(
            "identity-has-operations",
            "Identity patches cannot carry graph operations.",
            ctx.revision,
        )
    if patch.run_truth_scope or patch.repositories_read:
        ctx.report.reject(
            "identity-has-run-scope",
            "Identity patches cannot carry raw repository scope.",
            ctx.revision,
        )
    if patch.processed_cursors:
        ctx.report.reject(
            "identity-has-cursors",
            "Identity patches cannot carry coverage cursors.",
            ctx.revision,
        )
    if (
        patch.source_operation_id is not None
        or patch.source_effect_id is not None
        or patch.source_effect_sha256 is not None
    ):
        ctx.report.reject(
            "identity-has-operation-id",
            "Identity patches cannot carry an operation or effect id.",
            ctx.revision,
        )
    if patch.human_action is not None:
        ctx.report.reject(
            "identity-has-human-action",
            "Identity patches cannot carry a human authority action.",
            ctx.revision,
        )
    if patch.agent_action is not None:
        ctx.report.reject(
            "identity-has-agent-action",
            "Identity patches cannot carry an agent authority action.",
            ctx.revision,
        )
    if patch.experiment_control_node_id is not None or patch.experiment_decision_bundle:
        ctx.report.reject(
            "identity-has-experiment-control",
            "Identity patches cannot carry experiment control metadata.",
            ctx.revision,
        )
    if patch.authorized_by is not None or patch.profile is not None or patch.task_id is not None:
        ctx.report.reject(
            "identity-has-attribution",
            "Identity patches cannot carry human or task attribution.",
            ctx.revision,
        )


def _validate_attribution_shape(ctx: OpContext) -> None:
    patch = ctx.patch
    if patch.kind == "identity":
        return
    if (patch.source_effect_id is None) != (patch.source_effect_sha256 is None):
        ctx.report.reject(
            "invalid-source-effect",
            "A source effect id and its exact Patch digest must be present together.",
            ctx.revision,
        )
    if patch.source_effect_id is not None and (
        patch.author != "agent" or not patch.source_operation_id
    ):
        ctx.report.reject(
            "invalid-source-effect",
            "A source effect id requires one direct agent task source operation.",
            ctx.revision,
        )
    has_attribution = (
        patch.authorized_by is not None or patch.profile is not None or patch.task_id is not None
    )
    if not has_attribution:
        return

    if patch.kind == "approval":
        if patch.authorized_by is None or patch.profile is not None or patch.task_id is not None:
            ctx.report.reject(
                "invalid-human-attribution",
                "An attributed human approval requires authorized_by and cannot carry an "
                "agent profile or task id.",
                ctx.revision,
            )
        return

    if (
        patch.authorized_by is None
        or patch.profile not in {"ordinary", "orchestrator"}
        or not patch.task_id
    ):
        ctx.report.reject(
            "invalid-agent-attribution",
            "An attributed agent patch requires authorized_by, one known agent profile, and a "
            "non-empty direct task id.",
            ctx.revision,
        )


def _validate_declared_agent_action(ctx: OpContext) -> None:
    patch = ctx.patch
    has_decision_outcome = _has_declared_decision_outcome(ctx)
    if patch.human_action is not None and patch.agent_action is not None:
        ctx.report.reject(
            "conflicting-authority-actions",
            "A Patch cannot declare both a human and an agent authority action.",
            ctx.revision,
        )
        return
    if patch.human_action is not None and (patch.kind != "approval" or patch.author != "human"):
        ctx.report.reject(
            "invalid-human-action",
            "Only a human approval Patch may declare a human authority action.",
            ctx.revision,
        )
    if (
        patch.author == "agent"
        and patch.profile == "orchestrator"
        and has_decision_outcome
        and patch.agent_action != "decision_choice"
    ):
        ctx.report.reject(
            "missing-decision-action",
            "An orchestrator Decision outcome requires agent_action='decision_choice'.",
            ctx.revision,
        )
    if patch.agent_action is None:
        return
    if patch.author != "agent" or patch.profile != "orchestrator":
        ctx.report.reject(
            "invalid-agent-action",
            "Only the orchestrator profile may declare an agent authority action.",
            ctx.revision,
        )
        return
    if not has_decision_outcome:
        ctx.report.reject(
            "unused-agent-action",
            "agent_action='decision_choice' must name a real Decision outcome in this Patch.",
            ctx.revision,
        )


def _has_declared_decision_outcome(ctx: OpContext) -> bool:
    decision_ids = {
        node.id for node in ctx.initial_state.nodes.values() if isinstance(node, Decision)
    }
    decision_ids.update(
        raw.id
        for operation in ctx.patch.ops
        if isinstance(operation, CreateNodesOperation)
        for raw in operation.nodes
        if raw.type == "decision"
    )
    return any(
        update.id in decision_ids
        and (
            update.changes.get("status") == "decided"
            or update.changes.get("selected_option") is not None
        )
        for operation in ctx.patch.ops
        if isinstance(operation, UpdateNodesOperation)
        for update in operation.nodes
    )


def _validate_declared_scope(ctx: OpContext) -> None:
    patch = ctx.patch
    if patch.kind == "identity":
        return
    if patch.kind == "approval":
        if patch.run_truth_scope or patch.repositories_read:
            ctx.report.reject(
                "approval-has-run-scope",
                "Human approval patches cannot carry raw repository scope.",
                ctx.revision,
            )
        return

    run_scope = set(patch.run_truth_scope)
    if not run_scope:
        ctx.report.reject(
            "empty-run-scope", "Agent patches require a non-empty run truth scope.", ctx.revision
        )
    outside = run_scope - ctx.project_truth_scope
    if outside:
        ctx.report.reject(
            "run-scope-outside-project",
            f"Run scope contains repositories outside project truth scope: {sorted(outside)}.",
            ctx.revision,
        )
    read_outside = set(patch.repositories_read) - run_scope
    if read_outside:
        ctx.report.reject(
            "read-outside-run-scope",
            f"Patch read repositories outside its run scope: {sorted(read_outside)}.",
            ctx.revision,
        )


def _validate_operations(ctx: OpContext):
    """Run each operation's rule, returning the oldest source reference cited."""
    oldest_ref = None
    for op in ctx.patch.ops:
        name = op.op
        rule = OP_RULES.get(name)
        if rule is None:  # pragma: no cover - the typed union and registry are kept exhaustive
            raise ValueError(f"typed operation {name!r} is missing from the operation registry")
        if ctx.mode == "admission" and rule.legacy_only:
            ctx.report.reject(
                "legacy-only-operation",
                f"Operation {name!r} is retained for historical replay and cannot be admitted "
                "in a new patch.",
                ctx.revision,
            )
            continue
        rejects_before = sum(message.level == "reject" for message in ctx.report.messages)
        _validate_operation_authority(ctx, op)
        if rule.structural_validate is not None:
            oldest_ref = older(oldest_ref, rule.structural_validate(op, ctx))
        if ctx.mode == "admission" and rule.authoring_validate is not None:
            oldest_ref = older(oldest_ref, rule.authoring_validate(op, ctx))
        rejects_after = sum(message.level == "reject" for message in ctx.report.messages)
        if rejects_after != rejects_before:
            continue
        try:
            # Imported lazily because materialization imports this validator.
            from rcp.core.materialize import apply_valid_operation

            ctx.state = apply_valid_operation(ctx.state, ctx.patch, op)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            ctx.report.reject(
                "malformed-operation",
                f"Operation {name!r} could not be staged: {exc}.",
                ctx.revision,
            )
    return oldest_ref


def _validate_operation_authority(ctx: OpContext, operation: GraphOperation) -> None:
    """Check live producer permission once, before an operation is staged."""

    if ctx.mode != "admission":
        return
    try:
        actions = operation_actions(ctx.initial_state, ctx.patch, operation)
    except ValueError:
        # The operation registry reports missing and unknown operation names.
        return
    for action in sorted(actions):
        if not permits(ctx.patch, action):
            ctx.report.reject(
                "graph-action-refused",
                f"Action {action!r} is not permitted for this Patch producer.",
                ctx.revision,
            )
