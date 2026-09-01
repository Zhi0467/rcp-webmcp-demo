from __future__ import annotations

from rcp.control import (
    active_flow_projection,
    derive_experiment_control_state,
    experiment_control_dependencies,
    experiment_graph_control,
)
from rcp.core.models import Blocker, Decision, Edge, Experiment, GraphState

EXPERIMENT_ID = "exp/transition-control"
DECISION_ID = "dec/transition-control"
BLOCKER_ID = "blk/transition-control"


def _state(*, phase: str = "unspecified", blocker_status: str = "resolved") -> GraphState:
    experiment = Experiment(
        id=EXPERIMENT_ID,
        type="experiment",
        title="Transition control",
        objective="Exercise graph-derived control.",
        status=phase,
    )
    decision = Decision(
        id=DECISION_ID,
        type="decision",
        title="Execution shape",
        question="Which execution shape?",
        options=["small", "large"],
        selected_option="small",
        status="decided",
    )
    blocker = Blocker(
        id=BLOCKER_ID,
        type="blocker",
        title="Capacity",
        description="Wait for capacity.",
        status=blocker_status,
    )
    return GraphState(
        revision=4,
        nodes={node.id: node for node in (experiment, decision, blocker)},
        edges={
            "edge/governed": Edge(
                id="edge/governed",
                source=EXPERIMENT_ID,
                target=DECISION_ID,
                relation="governed_by",
                layer="action",
            ),
            "edge/blocked": Edge(
                id="edge/blocked",
                source=EXPERIMENT_ID,
                target=BLOCKER_ID,
                relation="blocked_by",
                layer="action",
            ),
        },
    )


def test_intrinsic_phase_never_supplies_experiment_gate_truth() -> None:
    unspecified = _state(phase="unspecified", blocker_status="resolved")
    running = _state(phase="running", blocker_status="resolved")

    assert experiment_graph_control(unspecified, EXPERIMENT_ID).ready
    assert experiment_graph_control(running, EXPERIMENT_ID).ready
    assert derive_experiment_control_state(unspecified, EXPERIMENT_ID).ready

    gated = _state(phase="unspecified", blocker_status="open")
    graph_control = experiment_graph_control(gated, EXPERIMENT_ID)
    assert not graph_control.ready
    assert graph_control.reasons == [f"Blocker {BLOCKER_ID} is open."]
    assert not derive_experiment_control_state(gated, EXPERIMENT_ID).ready


def test_control_dependencies_include_a_ready_governing_choice_change() -> None:
    state = _state()
    before = experiment_control_dependencies(state, EXPERIMENT_ID)
    decision = state.nodes[DECISION_ID]
    assert isinstance(decision, Decision)
    changed = state.model_copy(
        update={
            "nodes": {
                **state.nodes,
                DECISION_ID: decision.model_copy(update={"selected_option": "large"}),
            }
        }
    )
    after = experiment_control_dependencies(changed, EXPERIMENT_ID)

    assert not before.gate_reasons
    assert not after.gate_reasons
    assert before != after
    assert before.governing_decisions[0].selected_option == "small"
    assert after.governing_decisions[0].selected_option == "large"


def test_control_dependencies_include_a_non_gating_blocker_relation() -> None:
    state = _state(blocker_status="resolved")
    before = experiment_control_dependencies(state, EXPERIMENT_ID)
    changed = state.model_copy(
        update={
            "edges": {
                edge_id: edge for edge_id, edge in state.edges.items() if edge_id != "edge/blocked"
            }
        }
    )
    after = experiment_control_dependencies(changed, EXPERIMENT_ID)

    assert not before.gate_reasons
    assert not after.gate_reasons
    assert [item.blocker_id for item in before.blockers] == [BLOCKER_ID]
    assert not after.blockers


def test_active_flow_hides_resolved_blocker_without_changing_canonical_state() -> None:
    state = _state(blocker_status="resolved")

    active = active_flow_projection(state)
    assert BLOCKER_ID not in active.node_ids
    assert "edge/blocked" not in active.edge_ids
    assert BLOCKER_ID in state.nodes
    assert "edge/blocked" in state.edges

    with_history = active_flow_projection(state, include_resolved_blockers=True)
    assert BLOCKER_ID in with_history.node_ids
    assert "edge/blocked" in with_history.edge_ids
