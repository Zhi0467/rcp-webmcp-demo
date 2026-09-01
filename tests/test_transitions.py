from __future__ import annotations

import pytest
from pydantic import ValidationError

from rcp.core.materialize import materialize_patches
from rcp.core.models import Blocker, Edge, Evidence, Experiment, GraphState, Hypothesis, Patch
from rcp.core.operations import UpdateNodesOperation
from rcp.core.transition_models import GraphHeadRef, TransitionCauseRef
from rcp.core.transitions import (
    GUIDANCE_RULE_ID,
    TRANSITION_RULESET_TAG,
    GraphTransitionManager,
    TransitionConflict,
    _sha256,
    _source_patch_for_group,
    _transition_id,
    transition_trigger_manifest,
    validate_transition_trace,
)


def _gated_state(*, experiments: int = 1) -> GraphState:
    nodes = {
        "blk/capacity": Blocker(
            id="blk/capacity",
            type="blocker",
            title="Capacity",
            description="The required capacity is unavailable.",
            status="open",
        )
    }
    edges: dict[str, Edge] = {}
    for index in range(experiments):
        experiment_id = f"exp/{index}"
        nodes[experiment_id] = Experiment(
            id=experiment_id,
            type="experiment",
            title=f"Experiment {index}",
            objective="Test the transition manager.",
            current_summary="Waiting for capacity.",
            next_action="Resolve the capacity blocker.",
        )
        edge = Edge(
            id=f"edge/blocked/{index}",
            source=experiment_id,
            target="blk/capacity",
            relation="blocked_by",
            layer="action",
        )
        edges[edge.id] = edge
    return GraphState(nodes=nodes, edges=edges)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (GraphHeadRef, {"revision": "1"}),
        (TransitionCauseRef, {"kind": "action", "action_index": "0"}),
    ],
)
def test_transition_provenance_rejects_scalar_coercion(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def _resolve_patch(
    *,
    revision: int = 1,
    extra_changes: dict[str, object] | None = None,
) -> Patch:
    operations: list[dict[str, object]] = [
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "blk/capacity",
                    "base_updated_rev": revision - 1,
                    "changes": {"status": "resolved"},
                }
            ],
        }
    ]
    if extra_changes:
        operations.append(
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/0",
                        "base_updated_rev": revision - 1,
                        "changes": extra_changes,
                    }
                ],
            }
        )
    return Patch(
        revision=revision,
        kind="approval",
        author="human",
        producer="human",
        summary="Resolve the capacity blocker.",
        ops=operations,
    )


def _setup_patch() -> Patch:
    state = _gated_state()
    return Patch(
        schema_generation=1,
        revision=1,
        kind="seed",
        author="agent",
        producer="agent",
        summary="Historical setup for replay proof.",
        run_truth_scope=["repo"],
        repositories_read=["repo"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [node.model_dump(mode="json") for node in state.nodes.values()],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "relation": edge.relation,
                    }
                    for edge in state.edges.values()
                ],
            },
        ],
    )


def test_blocker_resolution_reaches_deterministic_transition_closure() -> None:
    state = _gated_state()
    first = GraphTransitionManager().prepare_validated(state, [_resolve_patch()])
    second = GraphTransitionManager().prepare_validated(state, [_resolve_patch()])

    assert first.patch.transition == second.patch.transition
    assert first.patch.ops == second.patch.ops
    assert first.patch.transition is not None
    assert first.patch.transition.ruleset_tag == TRANSITION_RULESET_TAG
    assert [item.rule_id for item in first.patch.transition.generated_actions] == [GUIDANCE_RULE_ID]
    blocker = first.projection.graph.nodes["blk/capacity"]
    assert isinstance(blocker, Blocker)
    assert blocker.status == "resolved"
    assert "edge/blocked/0" in first.projection.graph.edges
    experiment = first.projection.graph.nodes["exp/0"]
    assert isinstance(experiment, Experiment)
    assert experiment.current_summary_stale is True
    assert experiment.next_action_stale is True
    assert first.projection.experiment_control["exp/0"].ready is True
    assert first.projection.guidance_validity["exp/0"].current_summary.status == "stale"
    assert [event.event_type for event in first.patch.transition.lifecycle_events] == [
        "node_status_changed",
        "guidance_invalidated",
        "guidance_invalidated",
    ]


def test_same_transition_guidance_edit_remains_stale() -> None:
    prepared = GraphTransitionManager().prepare_validated(
        _gated_state(),
        [_resolve_patch(extra_changes={"current_summary": "Capacity is restored."})],
    )

    experiment = prepared.projection.graph.nodes["exp/0"]
    assert isinstance(experiment, Experiment)
    assert experiment.current_summary == "Capacity is restored."
    assert experiment.current_summary_stale is True
    assert experiment.next_action_stale is True


def test_fresh_experiment_guidance_starts_current() -> None:
    experiment = Experiment(
        id="exp/new",
        type="experiment",
        title="New experiment",
        objective="Establish initial guidance.",
        current_summary="The experiment is ready to begin.",
        next_action="Run the first attempt.",
    )
    patch = Patch(
        revision=1,
        kind="seed",
        author="agent",
        producer="agent",
        summary="Create the experiment with authored guidance.",
        ops=[{"op": "create_nodes", "nodes": [experiment.model_dump(mode="json")]}],
    )

    prepared = GraphTransitionManager().prepare_validated(GraphState(), [patch])

    created = prepared.projection.graph.nodes[experiment.id]
    assert isinstance(created, Experiment)
    assert created.current_summary_stale is False
    assert created.next_action_stale is False
    assert prepared.patch.transition is not None
    assert prepared.patch.transition.generated_actions == []


def test_fresh_experiment_guidance_is_invalidated_by_later_dependency_action() -> None:
    experiment = Experiment(
        id="exp/new",
        type="experiment",
        title="New experiment",
        objective="Establish initial guidance.",
        current_summary="No blockers are known.",
        next_action="Run the first attempt.",
    )
    blocker = Blocker(
        id="blk/new",
        type="blocker",
        title="New blocker",
        description="A blocker discovered while creating the experiment.",
    )
    patch = Patch(
        revision=1,
        kind="seed",
        author="agent",
        producer="agent",
        summary="Create the experiment, then add its blocker.",
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    experiment.model_dump(mode="json"),
                    blocker.model_dump(mode="json"),
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": experiment.id,
                        "target": blocker.id,
                        "relation": "blocked_by",
                    }
                ],
            },
        ],
    )

    prepared = GraphTransitionManager().prepare_validated(GraphState(), [patch])

    created = prepared.projection.graph.nodes[experiment.id]
    assert isinstance(created, Experiment)
    assert created.current_summary_stale is True
    assert created.next_action_stale is True
    assert prepared.patch.transition is not None
    assert {
        (event.field, event.before, event.after, event.cause.action_index)
        for event in prepared.patch.transition.lifecycle_events
    } == {
        ("current_summary_stale", False, True, 2),
        ("next_action_stale", False, True, 2),
    }


@pytest.mark.parametrize("change", ["create", "remove"])
def test_tests_relation_identity_invalidates_guidance_without_evidence(change: str) -> None:
    experiment = Experiment(
        id="exp/direct-test",
        type="experiment",
        title="Direct test",
        objective="Test a hypothesis without existing Evidence.",
        current_summary="No direct hypothesis is currently selected.",
        next_action="Choose the direct hypothesis under test.",
    )
    hypothesis = Hypothesis(
        id="hyp/direct-test",
        type="hypothesis",
        title="Direct hypothesis",
        statement="The intervention changes the measured result.",
    )
    tests_edge = Edge(
        id="edge/direct-test",
        source=experiment.id,
        target=hypothesis.id,
        relation="tests",
        layer="seam",
    )
    state = GraphState(
        nodes={experiment.id: experiment, hypothesis.id: hypothesis},
        edges={tests_edge.id: tests_edge} if change == "remove" else {},
    )
    operation: dict[str, object]
    if change == "create":
        operation = {
            "op": "create_edges",
            "edges": [
                {
                    "id": tests_edge.id,
                    "source": tests_edge.source,
                    "target": tests_edge.target,
                    "relation": tests_edge.relation,
                }
            ],
        }
    else:
        operation = {"op": "remove_edges", "edge_ids": [tests_edge.id]}
    patch = Patch(
        revision=1,
        kind="approval",
        author="human",
        producer="human",
        summary=f"{change.title()} the direct tests relation.",
        ops=[operation],
    )

    prepared = GraphTransitionManager().prepare_validated(state, [patch])

    projected = prepared.projection.graph.nodes[experiment.id]
    assert isinstance(projected, Experiment)
    assert projected.current_summary_stale is True
    assert projected.next_action_stale is True
    assert not any(isinstance(node, Evidence) for node in state.nodes.values())
    assert prepared.patch.transition is not None
    assert [item.cause.action_index for item in prepared.patch.transition.generated_actions] == [0]


def test_later_explicit_guidance_edit_refreshes_only_that_field() -> None:
    first = GraphTransitionManager().prepare_validated(_gated_state(), [_resolve_patch()])
    second_patch = Patch(
        revision=2,
        kind="approval",
        author="human",
        producer="human",
        summary="Affirm current summary against the final gate state.",
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/0",
                        "base_updated_rev": 1,
                        "changes": {"current_summary": "Capacity is restored."},
                    }
                ],
            }
        ],
    )

    second = GraphTransitionManager().prepare_validated(first.projection.graph, [second_patch])

    experiment = second.projection.graph.nodes["exp/0"]
    assert isinstance(experiment, Experiment)
    assert experiment.current_summary_stale is False
    assert experiment.next_action_stale is True
    assert second.projection.guidance_validity["exp/0"].current_summary.status == "current"


def test_distinct_guidance_refreshes_keep_their_exact_initiating_causes() -> None:
    state = GraphState(
        nodes={
            "exp/guidance-causes": Experiment(
                id="exp/guidance-causes",
                type="experiment",
                title="Guidance causes",
                objective="Keep generated refresh attribution field-specific.",
                current_summary="Old summary.",
                current_summary_stale=True,
                next_action="Old next action.",
                next_action_stale=True,
            )
        }
    )
    patch = Patch(
        revision=1,
        kind="approval",
        author="human",
        producer="human",
        summary="Refresh both guidance fields independently.",
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/guidance-causes",
                        "changes": {"current_summary": "Fresh summary."},
                    }
                ],
            },
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "exp/guidance-causes",
                        "changes": {"next_action": "Fresh next action."},
                    }
                ],
            },
        ],
    )

    prepared = GraphTransitionManager().prepare_validated(state, [patch])

    assert prepared.patch.transition is not None
    generated = prepared.patch.transition.generated_actions
    assert [item.cause.action_index for item in generated] == [0, 1]
    generated_operations = [prepared.patch.ops[item.operation_index] for item in generated]
    assert all(isinstance(operation, UpdateNodesOperation) for operation in generated_operations)
    assert [sorted(operation.nodes[0].changes) for operation in generated_operations] == [
        ["current_summary_stale"],
        ["next_action_stale"],
    ]
    forged_generated = [
        generated[0],
        generated[1].model_copy(
            update={"cause": TransitionCauseRef(kind="action", action_index=0)}
        ),
    ]
    forged_trace = prepared.patch.transition.model_copy(
        update={"generated_actions": forged_generated}
    )
    with pytest.raises(ValueError, match="transition id"):
        validate_transition_trace(
            state,
            prepared.patch.model_copy(update={"transition": forged_trace}),
        )


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "supersede_nodes",
            "nodes": [{"id": "hyp/old", "superseded_by": "hyp/current"}],
        },
        {
            "op": "merge_nodes",
            "merges": [{"duplicate": "hyp/old", "canonical": "hyp/current"}],
        },
    ],
)
def test_supersede_and_merge_emit_status_events(operation: dict[str, object]) -> None:
    state = GraphState(
        nodes={
            "hyp/old": Hypothesis(
                id="hyp/old",
                type="hypothesis",
                title="Old hypothesis",
                statement="The older account.",
            ),
            "hyp/current": Hypothesis(
                id="hyp/current",
                type="hypothesis",
                title="Current hypothesis",
                statement="The current account.",
            ),
        }
    )
    patch = Patch(
        revision=1,
        kind="approval",
        author="human",
        producer="human",
        summary="Retire the old hypothesis.",
        ops=[operation],
    )

    prepared = GraphTransitionManager().prepare_validated(state, [patch])

    assert prepared.patch.transition is not None
    event = prepared.patch.transition.lifecycle_events[0]
    assert event.event_type == "node_status_changed"
    assert event.node_id == "hyp/old"
    assert event.before == "proposed"
    assert event.after == "superseded"
    assert event.cause.action_index == 0


def test_status_event_ignores_earlier_same_value_write_for_attribution() -> None:
    state = GraphState(
        nodes={
            "blk/one": Blocker(
                id="blk/one",
                type="blocker",
                title="Blocker",
                description="A temporary blocker.",
            )
        }
    )
    patch = Patch(
        revision=1,
        kind="approval",
        author="human",
        producer="human",
        summary="Resolve the blocker after an idempotent write.",
        ops=[
            {
                "op": "update_nodes",
                "nodes": [{"id": "blk/one", "changes": {"status": "open"}}],
            },
            {
                "op": "update_nodes",
                "nodes": [{"id": "blk/one", "changes": {"status": "resolved"}}],
            },
        ],
    )

    prepared = GraphTransitionManager().prepare_validated(state, [patch])

    assert prepared.patch.transition is not None
    assert len(prepared.patch.transition.lifecycle_events) == 1
    event = prepared.patch.transition.lifecycle_events[0]
    assert event.before == "open"
    assert event.after == "resolved"
    assert event.cause.action_index == 1


def test_status_event_survives_later_node_removal() -> None:
    state = GraphState(
        nodes={
            "blk/one": Blocker(
                id="blk/one",
                type="blocker",
                title="Blocker",
                description="A temporary blocker.",
            )
        }
    )
    patch = Patch(
        revision=1,
        kind="approval",
        author="human",
        producer="human",
        summary="Resolve and remove the blocker.",
        ops=[
            {
                "op": "update_nodes",
                "nodes": [{"id": "blk/one", "changes": {"status": "resolved"}}],
            },
            {"op": "remove_nodes", "node_ids": ["blk/one"]},
        ],
    )

    prepared = GraphTransitionManager().prepare_validated(state, [patch])

    assert "blk/one" not in prepared.projection.graph.nodes
    assert prepared.patch.transition is not None
    assert len(prepared.patch.transition.lifecycle_events) == 1
    event = prepared.patch.transition.lifecycle_events[0]
    assert event.node_id == "blk/one"
    assert event.before == "open"
    assert event.after == "resolved"
    assert event.cause.action_index == 0


def test_recorded_expanded_actions_replay_without_loading_rule_registry(monkeypatch) -> None:
    setup = _setup_patch()
    base = materialize_patches(
        [setup],
        initial_truth_scope=["repo"],
        repository_aliases=["repo"],
    )
    assert base.state.replay_status == "complete"
    prepared = GraphTransitionManager().prepare_validated(
        base.state,
        [_resolve_patch(revision=2)],
    )
    monkeypatch.setattr("rcp.core.transitions.RULE_REGISTRY", ())

    replay = materialize_patches(
        [setup, prepared.patch],
        initial_truth_scope=["repo"],
        repository_aliases=["repo"],
    )

    assert replay.state.replay_status == "complete"
    experiment = replay.state.nodes["exp/0"]
    assert isinstance(experiment, Experiment)
    assert experiment.current_summary_stale is True
    assert replay.state == prepared.projection.graph


def test_tampered_expanded_operation_fails_closed() -> None:
    setup = _setup_patch()
    base = materialize_patches(
        [setup],
        initial_truth_scope=["repo"],
        repository_aliases=["repo"],
    )
    prepared = GraphTransitionManager().prepare_validated(
        base.state,
        [_resolve_patch(revision=2)],
    )
    tampered = prepared.patch.model_copy(update={"ops": prepared.patch.ops[:-1]})

    replay = materialize_patches(
        [setup, tampered],
        initial_truth_scope=["repo"],
        repository_aliases=["repo"],
    )

    assert replay.state.replay_status == "degraded"
    assert replay.state.replay_failure is not None
    assert replay.state.replay_failure.code == "transition-trace-invalid"


def _transition_without_generated_actions() -> tuple[GraphState, Patch]:
    state = GraphState()
    patch = Patch(
        revision=1,
        kind="seed",
        author="agent",
        producer="agent",
        summary="Create one question without triggering lifecycle rules.",
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/transition-identity",
                        "type": "research_question",
                        "title": "Transition identity",
                        "question": "Does the ruleset tag participate in identity?",
                    }
                ],
            }
        ],
    )
    return state, GraphTransitionManager().prepare_validated(state, [patch]).patch


def test_transition_identity_rejects_a_forged_ruleset_tag() -> None:
    state, patch = _transition_without_generated_actions()
    assert patch.transition is not None
    forged_trace = patch.transition.model_copy(update={"ruleset_tag": "rcp.lifecycle.forged"})

    with pytest.raises(ValueError, match="transition id"):
        validate_transition_trace(state, patch.model_copy(update={"transition": forged_trace}))


def test_transition_identity_accepts_a_recomputed_historical_ruleset_tag() -> None:
    state, patch = _transition_without_generated_actions()
    assert patch.transition is not None
    ruleset_tag = "rcp.lifecycle.historical"
    source_patches = [
        _source_patch_for_group(patch, group) for group in patch.transition.initiating_groups
    ]
    source_actions = [
        (source_patch, operation)
        for source_patch in source_patches
        for operation in source_patch.ops
    ]
    transition_id = _transition_id(
        patch.transition.pre_head,
        source_patches,
        patch.transition.initiating_groups,
        source_actions,
        ruleset_tag=ruleset_tag,
    )
    historical_trace = patch.transition.model_copy(
        update={"ruleset_tag": ruleset_tag, "transition_id": transition_id}
    )

    recovered = validate_transition_trace(
        state,
        patch.model_copy(update={"transition": historical_trace}),
    )

    assert recovered == source_patches


def test_transition_trace_rejects_missing_or_forged_lifecycle_events() -> None:
    state = _gated_state()
    patch = GraphTransitionManager().prepare_validated(state, [_resolve_patch()]).patch
    assert patch.transition is not None
    event = patch.transition.lifecycle_events[0]
    forged_payload = {
        "ordinal": 0,
        "event_type": event.event_type,
        "cause": event.cause.action_index,
        "node_id": event.node_id,
        "field": event.field,
        "before": event.before,
        "after": event.before,
    }
    forged = event.model_copy(
        update={
            "after": event.before,
            "event_id": _sha256(
                {"transition_id": patch.transition.transition_id, **forged_payload}
            ),
        }
    )

    for lifecycle_events in ([], [forged]):
        trace = patch.transition.model_copy(update={"lifecycle_events": lifecycle_events})
        with pytest.raises(ValueError, match="lifecycle events"):
            validate_transition_trace(state, patch.model_copy(update={"transition": trace}))


def test_rule_firing_guard_rejects_before_any_candidate_commits() -> None:
    with pytest.raises(TransitionConflict, match="exceeded 1 generated actions"):
        GraphTransitionManager(max_rule_firings=1).prepare_validated(
            _gated_state(experiments=2),
            [_resolve_patch()],
        )


def test_trigger_manifest_is_backend_versioned_and_conservative() -> None:
    manifest = transition_trigger_manifest()

    assert manifest.ruleset_tag == TRANSITION_RULESET_TAG
    assert {trigger.operation for trigger in manifest.triggers} >= {
        "update_nodes",
        "create_edges",
        "remove_edges",
        "create_proposals",
        "resolve_proposals",
    }
