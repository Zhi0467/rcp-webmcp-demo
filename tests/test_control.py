from __future__ import annotations

from copy import deepcopy

from rcp.control import (
    ExperimentOperationalState,
    derive_experiment_control_state,
    governing_decision_bundle,
)
from rcp.core.materialize import apply_valid_patch
from rcp.core.models import (
    Blocker,
    Decision,
    Edge,
    Experiment,
    ExperimentAttempt,
    ExperimentDecisionPin,
    GatedCard,
    GraphState,
    Hypothesis,
    Patch,
    Proposal,
)
from rcp.core.operations import adapt_persisted_patch_document
from rcp.core.validation import validate_patch

EXPERIMENT_ID = "exp/control-loop"
DECISION_ID = "dec/resource-shape"
PIN = ExperimentDecisionPin(
    decision_id=DECISION_ID,
    decision_revision=3,
    selected_option="4xA100",
)


def _state(
    *,
    attempts: list[ExperimentAttempt] | None = None,
    ceiling: int = 5,
    status: str = "proposed",
) -> GraphState:
    experiment = Experiment(
        id=EXPERIMENT_ID,
        type="experiment",
        title="Control loop",
        objective="Test the loop.",
        invocation_ceiling=ceiling,
        attempts=attempts or [],
        status=status,
    )
    decision = Decision(
        id=DECISION_ID,
        type="decision",
        title="Resource shape",
        question="Which resource shape?",
        options=["4xA100", "8xA100"],
        selected_option="4xA100",
        status="decided",
        updated_rev=3,
    )
    blocker = Blocker(
        id="blk/capacity",
        type="blocker",
        title="Capacity",
        description="Wait for capacity.",
        status="resolved",
    )
    hypothesis = Hypothesis(
        id="hyp/target",
        type="hypothesis",
        title="Target",
        statement="The intervention helps.",
    )
    return GraphState(
        revision=3,
        project_truth_scope=["repo"],
        nodes={node.id: node for node in (experiment, decision, blocker, hypothesis)},
        edges={
            "governed": Edge(
                id="governed",
                source=EXPERIMENT_ID,
                target=DECISION_ID,
                relation="governed_by",
                layer="action",
            ),
            "blocked": Edge(
                id="blocked",
                source=EXPERIMENT_ID,
                target=blocker.id,
                relation="blocked_by",
                layer="action",
            ),
            "tests": Edge(
                id="tests",
                source=EXPERIMENT_ID,
                target=hypothesis.id,
                relation="tests",
                layer="seam",
            ),
        },
    )


def _attempt(
    *,
    attempt_id: str = "attempt-1",
    status: str = "running",
    selected_option: str = "4xA100",
    attempt_kind: str = "external_run",
) -> ExperimentAttempt:
    return ExperimentAttempt.model_validate(
        {
            "id": attempt_id,
            "sequence": 1,
            "purpose": "Run the configured experiment.",
            "attempt_kind": attempt_kind,
            "decision_bundle": [
                {
                    "decision_id": DECISION_ID,
                    "decision_revision": 3,
                    "selected_option": selected_option,
                }
            ],
            "status": status,
            "job_refs": ["4471"],
        }
    )


def _patch(ops: list[dict], *, stamp: bool = True) -> Patch:
    return Patch(
        revision=4,
        kind="experiment_loop",
        author="agent",
        summary="Reflect the bounded experiment loop.",
        ops=ops,
        run_truth_scope=["repo"],
        repositories_read=[],
        experiment_control_node_id=EXPERIMENT_ID if stamp else None,
        experiment_decision_bundle=[PIN] if stamp else [],
    )


def _codes(report) -> set[str]:
    return {message.code for message in report.messages if message.level == "reject"}


def test_readiness_is_derived_from_decisions_proposals_blockers_and_ceiling() -> None:
    state = _state()
    ready = derive_experiment_control_state(state, EXPERIMENT_ID)
    assert ready.ready
    assert ready.reasons == []
    assert ready.governing_decisions == [PIN]

    decision = state.nodes[DECISION_ID]
    state.nodes[DECISION_ID] = decision.model_copy(
        update={"status": "open", "selected_option": None}
    )
    state.proposals["proposal/resource"] = Proposal(
        id="proposal/resource",
        title="Change resources",
        card=GatedCard(),
        ops=[
            {
                "op": "update_nodes",
                "intent": "content_change",
                "nodes": [{"id": DECISION_ID, "changes": {"selected_option": "8xA100"}}],
            }
        ],
        related_node_ids=[DECISION_ID],
        base_rev=3,
    )
    blocker = state.nodes["blk/capacity"]
    state.nodes[blocker.id] = blocker.model_copy(update={"status": "open"})
    experiment = state.nodes[EXPERIMENT_ID]
    state.nodes[EXPERIMENT_ID] = experiment.model_copy(
        update={"invocation_ceiling": 1, "attempts": [_attempt(status="completed")]}
    )

    gated = derive_experiment_control_state(state, EXPERIMENT_ID)
    assert not gated.ready
    assert gated.reasons == [
        f"Decision {DECISION_ID} is not decided with a selected option.",
        f"Decision {DECISION_ID} has a pending proposal.",
        "Blocker blk/capacity is open.",
    ]
    assert gated.invocations_used == 0
    assert gated.invocation_ceiling == 1


def test_closed_experiment_blocks_only_a_fresh_episode_until_the_human_reopens_it() -> None:
    state = _state(status="completed")

    closed = derive_experiment_control_state(state, EXPERIMENT_ID)

    assert not closed.ready
    assert closed.reasons == [
        "This Experiment is completed. Edit its status before starting a new episode."
    ]
    assert closed.graph_reasons == []

    experiment = state.nodes[EXPERIMENT_ID]
    state.nodes[EXPERIMENT_ID] = experiment.model_copy(update={"status": "running"})

    reopened = derive_experiment_control_state(state, EXPERIMENT_ID)
    assert reopened.ready
    assert reopened.reasons == []


def test_an_open_episode_gates_readiness_and_publishes_the_graph_reasons_apart() -> None:
    """Readiness carries what admission enforces, and names its own two kinds.

    `create_experiment_episode_with_invocation` refuses a second live parent for
    one Experiment. A turn that settles below the ceiling without arming an
    observer leaves that parent live with nothing to wake it, so the loop reads
    as inactive while a new episode is still impossible. Readers must not have to
    tell that operational reason from a graph gate by matching its prose.
    """

    state = _state()
    open_episode = ExperimentOperationalState(episode_live=True)

    gated = derive_experiment_control_state(state, EXPERIMENT_ID, operational=open_episode)
    assert not gated.ready
    assert not gated.active
    assert gated.reasons == ["A previous episode is still open on this Experiment."]
    assert gated.graph_reasons == []

    blocker = state.nodes["blk/capacity"]
    state.nodes[blocker.id] = blocker.model_copy(update={"status": "open"})
    both = derive_experiment_control_state(state, EXPERIMENT_ID, operational=open_episode)
    assert both.graph_reasons == ["Blocker blk/capacity is open."]
    assert both.reasons == [
        "Blocker blk/capacity is open.",
        "A previous episode is still open on this Experiment.",
    ]

    # A narrower operational reason says more, so the open parent stays quiet
    # rather than restating it.
    active = derive_experiment_control_state(
        state,
        EXPERIMENT_ID,
        [EXPERIMENT_ID],
        operational=open_episode,
    )
    assert active.reasons == [
        "Blocker blk/capacity is open.",
        "An experiment loop is already active.",
    ]


def _belief_patch_ops(
    *,
    relation: str = "weakens",
    changes: dict[str, object] | None = None,
    cause: dict[str, object] | None = None,
    target: str = "hyp/target",
) -> list[dict[str, object]]:
    return [
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "ev/result",
                    "type": "evidence",
                    "title": "Result",
                    "observation": "Val perplexity rose by 1.5.",
                    "origin": "internal_run",
                }
            ],
        },
        {
            "op": "create_edges",
            "edges": [
                {"source": EXPERIMENT_ID, "target": "ev/result", "relation": "produces"},
                {
                    "source": "ev/result",
                    "target": "hyp/target",
                    "relation": relation,
                    "assessment": {
                        "relevance": "direct",
                        "weight": "moderate",
                        "qualifications": [],
                    },
                },
            ],
        },
        {
            "op": "create_proposals",
            "proposals": [
                {
                    "id": "proposal/belief",
                    "title": "Weaken the target hypothesis",
                    "card": {"decision_needed": "Accept this belief change?"},
                    "related_node_ids": [target],
                    "base_rev": 3,
                    "ops": [
                        {
                            "op": "update_nodes",
                            "intent": "status_change",
                            "nodes": [
                                {
                                    "id": target,
                                    "changes": changes or {"status": "weakened"},
                                    "cause": (
                                        cause
                                        if cause is not None
                                        else {
                                            "kind": "evidence_edge",
                                            "ref_id": "ev/result::weakens::hyp/target",
                                        }
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    ]


def test_a_loop_proposes_the_belief_change_its_own_evidence_implies() -> None:
    state = _state()
    assert not validate_patch(state, _patch(_belief_patch_ops()), ["repo"]).rejected

    # The status move is the only thing the human is being asked to accept.
    wider = _belief_patch_ops(changes={"status": "rejected", "scope": "Narrower."})
    assert "experiment-loop-belief-proposal-operations" in _codes(
        validate_patch(state, _patch(wider), ["repo"])
    )

    # Approval authorizes the move, while the scientific cause remains the
    # same-patch evidence edge rather than the Proposal itself.
    approval_as_cause = _belief_patch_ops(
        cause={"kind": "proposal_resolution", "ref_id": "proposal/belief"}
    )
    assert "experiment-loop-belief-cause" in _codes(
        validate_patch(state, _patch(approval_as_cause), ["repo"])
    )

    borrowed = _belief_patch_ops(
        cause={"kind": "evidence_edge", "ref_id": "ev/other::weakens::hyp/target"}
    )
    assert "experiment-loop-belief-cause" in _codes(
        validate_patch(state, _patch(borrowed), ["repo"])
    )

    # And the same patch must actually point evidence at that hypothesis.
    ungrounded = deepcopy(_belief_patch_ops())
    ungrounded[1]["edges"] = [
        {"source": EXPERIMENT_ID, "target": "ev/result", "relation": "produces"}
    ]
    assert "experiment-loop-belief-grounding" in _codes(
        validate_patch(state, _patch(ungrounded), ["repo"])
    )

    # A hypothesis this experiment does not test is neither a belief target nor a
    # governing decision, so its semantic update target is refused.
    foreign = _belief_patch_ops(target="hyp/unrelated")
    assert "experiment-loop-proposal-operations" in _codes(
        validate_patch(state, _patch(foreign), ["repo"])
    )


def test_a_proposal_that_only_references_a_decision_does_not_gate_the_run() -> None:
    state = _state()
    state.proposals["proposal/elsewhere"] = Proposal(
        id="proposal/elsewhere",
        title="Split a hypothesis",
        card=GatedCard(),
        ops=[
            {
                "op": "update_nodes",
                "intent": "content_change",
                "nodes": [{"id": "hyp/target", "changes": {"scope": "Narrower."}}],
            }
        ],
        # A seed may name a decision it merely read while raising this.
        related_node_ids=[DECISION_ID, "hyp/target"],
        base_rev=3,
    )
    assert derive_experiment_control_state(state, EXPERIMENT_ID).ready


def test_decision_drift_reports_a_moved_or_contested_pin_without_gating() -> None:
    state = _state(attempts=[_attempt(status="completed")])
    assert derive_experiment_control_state(state, EXPERIMENT_ID).decision_drift == []

    decision = state.nodes[DECISION_ID]
    state.nodes[DECISION_ID] = decision.model_copy(
        update={"selected_option": "8xA100", "updated_rev": 7}
    )
    moved = derive_experiment_control_state(
        state,
        EXPERIMENT_ID,
        episode_id="episode-1",
        invocations_used=1,
        invocation_ceiling=5,
        episode_decision_bundle=[PIN],
    )
    assert moved.ready
    assert [item.decision_id for item in moved.decision_drift] == [DECISION_ID]
    assert moved.decision_drift[0].pinned_option == "4xA100"
    assert moved.decision_drift[0].current_option == "8xA100"


def test_active_loop_marker_uses_control_runtime_not_semantic_attempts() -> None:
    state = _state()
    operation_active = derive_experiment_control_state(
        state, EXPERIMENT_ID, active_control_node_ids=[EXPERIMENT_ID]
    )
    assert operation_active.active
    assert operation_active.reasons == ["An experiment loop is already active."]

    state = _state(attempts=[_attempt()])
    attempt_active = derive_experiment_control_state(state, EXPERIMENT_ID)
    assert not attempt_active.active
    assert attempt_active.ready


def test_run_pins_the_governing_decision_bundle_in_stable_order() -> None:
    state = _state()
    second = Decision(
        id="dec/analysis",
        type="decision",
        title="Analysis",
        question="Which analysis?",
        selected_option="paired",
        status="decided",
        updated_rev=2,
    )
    state.nodes[second.id] = second
    state.edges["second"] = Edge(
        id="second",
        source=EXPERIMENT_ID,
        target=second.id,
        relation="governed_by",
        layer="action",
    )
    assert [item.decision_id for item in governing_decision_bundle(state, EXPERIMENT_ID)] == [
        "dec/analysis",
        DECISION_ID,
    ]


def test_loop_patch_can_append_and_close_its_own_attempt() -> None:
    state = _state()
    append = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": EXPERIMENT_ID,
                        "changes": {"status": "running", "attempts": [_attempt()]},
                    }
                ],
            }
        ]
    )
    assert not validate_patch(state, append, ["repo"]).rejected

    state = _state(attempts=[_attempt()])
    closed = _attempt(status="completed").model_copy(update={"outcome": "Finished."})
    close = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [{"id": EXPERIMENT_ID, "changes": {"attempts": [closed]}}],
            }
        ]
    )
    assert not validate_patch(state, close, ["repo"]).rejected

    # Lowering the human-owned ceiling stops new attempts; it must not prevent
    # the watcher turn from closing work that was already launched.
    first = _attempt(status="completed")
    second = _attempt(attempt_id="attempt-2").model_copy(update={"sequence": 2})
    state = _state(attempts=[first, second], ceiling=1)
    closed_second = second.model_copy(update={"status": "completed", "outcome": "Finished."})
    close_after_lowering = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": EXPERIMENT_ID,
                        "changes": {"attempts": [first, closed_second]},
                    }
                ],
            }
        ]
    )
    assert not validate_patch(state, close_after_lowering, ["repo"]).rejected


def test_loop_patch_may_append_multiple_semantically_meaningful_attempts() -> None:
    state = _state()
    first = _attempt(status="completed")
    second = _attempt(attempt_id="attempt-2").model_copy(update={"sequence": 2})
    patch = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": EXPERIMENT_ID,
                        "changes": {"attempts": [first, second]},
                    }
                ],
            }
        ]
    )

    assert "experiment-loop-multiple-attempts" not in _codes(validate_patch(state, patch, ["repo"]))


def test_loop_patch_may_refresh_summary_and_next_action_on_its_experiment() -> None:
    running = _attempt()
    state = _state(attempts=[running])
    closed = running.model_copy(update={"status": "completed", "outcome": "Finished."})
    patch = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": EXPERIMENT_ID,
                        "changes": {
                            "attempts": [closed],
                            "current_summary": "The configured run finished successfully.",
                            "next_action": None,
                        },
                    }
                ],
            }
        ]
    )

    report = validate_patch(state, patch, ["repo"])
    assert not report.rejected
    updated = apply_valid_patch(state, patch).nodes[EXPERIMENT_ID]
    assert isinstance(updated, Experiment)
    assert updated.attempts == [closed]
    assert updated.current_summary == "The configured run finished successfully."
    assert updated.next_action is None


def test_replaying_a_legacy_loop_patch_accepts_the_fields_its_own_migration_added() -> None:
    """Loading a pre-generation-2 Patch retires `blocked` and marks guidance stale.

    Those two system fields are added in memory by `adapt_persisted_patch_document`,
    never by the original write. Refusing them halted canonical history at the
    first legacy Experiment-loop Patch and left the whole graph read-only.
    """

    document = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": EXPERIMENT_ID,
                        "changes": {
                            "status": "blocked",
                            "current_summary": "The pinned server crashed.",
                            "next_action": "Reproduce under Slurm.",
                        },
                    }
                ],
            }
        ]
    ).model_dump(mode="json")
    document.pop("schema_generation", None)

    adapted = Patch.model_validate(adapt_persisted_patch_document(document))
    changes = adapted.ops[0].nodes[0].changes
    assert adapted.schema_generation == 1
    assert changes["status"] == "unspecified"
    assert changes["current_summary_stale"] is True
    assert changes["next_action_stale"] is True

    state = _state()
    assert not _codes(validate_patch(state, adapted, ["repo"], mode="replay"))
    # A live write still may not set them.
    assert "experiment-loop-experiment-field" in _codes(
        validate_patch(state, adapted, ["repo"], mode="admission")
    )


def test_loop_summary_authority_does_not_allow_other_fields_or_foreign_nodes() -> None:
    patch = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {"id": EXPERIMENT_ID, "changes": {"objective": "Broadened authority."}},
                    {"id": "hyp/target", "changes": {"current_summary": "Foreign prose."}},
                ],
            }
        ]
    )

    codes = _codes(validate_patch(_state(), patch, ["repo"]))
    assert "experiment-loop-experiment-field" in codes
    assert "experiment-loop-foreign-update" in codes


def test_loop_patch_cannot_edit_the_pinned_bundle_or_other_graph_authority() -> None:
    state = _state(attempts=[_attempt()])
    rewritten = _attempt(status="completed", selected_option="8xA100")
    operations = [
        {
            "op": "update_nodes",
            "nodes": [
                {"id": EXPERIMENT_ID, "changes": {"attempts": [rewritten]}},
                {"id": "blk/capacity", "changes": {"status": "open"}},
            ],
        },
        {"op": "set_standing", "node_id": EXPERIMENT_ID, "standing": "accepted"},
    ]
    codes = _codes(validate_patch(state, _patch(operations), ["repo"]))
    assert "experiment-loop-attempt-mutation" in codes
    assert "experiment-loop-foreign-update" in codes
    assert "experiment-loop-operation" in codes


def test_loop_patch_can_queue_a_pinned_decision_but_cannot_decide_it() -> None:
    state = _state()
    for status in ("open", "ready", "revisit"):
        patch = _patch(
            [
                {
                    "op": "update_nodes",
                    "nodes": [{"id": DECISION_ID, "changes": {"status": status}}],
                }
            ]
        )
        assert not validate_patch(state, patch, ["repo"]).rejected

    decide = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": DECISION_ID,
                        "changes": {"status": "decided", "selected_option": "8xA100"},
                    }
                ],
            }
        ]
    )
    codes = _codes(validate_patch(state, decide, ["repo"]))
    assert "experiment-loop-decision-action" in codes
    assert "decision-action-refused" in codes


def test_loop_patch_cannot_mutate_completion_criteria_but_attempts_do_not_spend_budget() -> None:
    state = _state(attempts=[_attempt(status="completed")], ceiling=1)
    second = _attempt(attempt_id="attempt-2", status="running").model_copy(update={"sequence": 2})
    patch = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": EXPERIMENT_ID,
                        "changes": {
                            "completion_criteria": ["Never agent editable."],
                            "attempts": [
                                *_state(attempts=[_attempt(status="completed")])
                                .nodes[EXPERIMENT_ID]
                                .attempts,
                                second,
                            ],
                        },
                    }
                ],
            }
        ]
    )
    codes = _codes(validate_patch(state, patch, ["repo"]))
    assert "experiment-loop-experiment-field" in codes
    assert "experiment-loop-attempt-ceiling" not in codes


def test_loop_patch_may_create_evidence_blockers_and_scoped_evidence_edges() -> None:
    state = _state()
    evidence = {
        "id": "ev/result",
        "type": "evidence",
        "title": "Result",
        "observation": "The run completed.",
        "origin": "internal_run",
    }
    valid_ops = [
        {"op": "create_nodes", "nodes": [evidence]},
        {
            "op": "create_edges",
            "edges": [
                {
                    "source": evidence["id"],
                    "target": "hyp/target",
                    "relation": "supports",
                    "assessment": {
                        "relevance": "direct",
                        "weight": "moderate",
                        "qualifications": [],
                    },
                }
            ],
        },
    ]
    valid = _patch(valid_ops)
    assert not validate_patch(state, valid, ["repo"]).rejected

    invalid = deepcopy(valid_ops)
    invalid[0]["nodes"] = [
        {
            "id": "hyp/new",
            "type": "hypothesis",
            "title": "New",
            "statement": "New hypothesis.",
        }
    ]
    invalid[1]["edges"] = [
        {
            "source": EXPERIMENT_ID,
            "target": "hyp/target",
            "relation": "tests",
        }
    ]
    codes = _codes(validate_patch(state, _patch(invalid), ["repo"]))
    assert "experiment-loop-created-node" in codes
    assert "experiment-loop-edge-layer" in codes


def test_loop_may_hand_new_evidence_to_existing_decisions_and_blockers() -> None:
    state = _state()
    evidence = {
        "id": "ev/result",
        "type": "evidence",
        "title": "Result",
        "observation": "The run completed.",
        "origin": "internal_run",
    }
    handoff_ops = [
        {"op": "create_nodes", "nodes": [evidence]},
        {
            "op": "create_edges",
            "edges": [
                {"source": EXPERIMENT_ID, "target": evidence["id"], "relation": "produces"},
                {"source": evidence["id"], "target": DECISION_ID, "relation": "informs"},
                {
                    "source": evidence["id"],
                    "target": "blk/capacity",
                    "relation": "addresses",
                },
            ],
        },
    ]
    handoffs = _patch(handoff_ops)
    assert not validate_patch(state, handoffs, ["repo"]).rejected

    wrong_source = deepcopy(handoff_ops)
    wrong_source[1]["edges"][1] = {
        "source": EXPERIMENT_ID,
        "target": DECISION_ID,
        "relation": "informs",
    }
    assert "experiment-loop-evidence-handoff" in _codes(
        validate_patch(state, _patch(wrong_source), ["repo"])
    )

    wrong_target = deepcopy(handoff_ops)
    wrong_target[1]["edges"][2] = {
        "source": evidence["id"],
        "target": "hyp/target",
        "relation": "addresses",
    }
    assert "experiment-loop-evidence-handoff" in _codes(
        validate_patch(state, _patch(wrong_target), ["repo"])
    )


def test_loop_attaches_its_own_evidence_and_blockers_to_its_experiment() -> None:
    state = _state()
    evidence = {
        "id": "ev/result",
        "type": "evidence",
        "title": "Result",
        "observation": "The run completed.",
        "origin": "internal_run",
    }
    blocker = {
        "id": "blk/exhausted",
        "type": "blocker",
        "title": "Exhausted",
        "description": "The attempt ceiling was reached.",
    }
    attached_ops = [
        {"op": "create_nodes", "nodes": [evidence, blocker]},
        {
            "op": "create_edges",
            "edges": [
                {"source": EXPERIMENT_ID, "target": "ev/result", "relation": "produces"},
                {"source": EXPERIMENT_ID, "target": "blk/exhausted", "relation": "blocked_by"},
            ],
        },
    ]
    attached = _patch(attached_ops)
    assert not validate_patch(state, attached, ["repo"]).rejected

    # Attaching a node the patch did not create would let the loop claim
    # someone else's evidence or block itself on an unrelated blocker.
    foreign = deepcopy(attached_ops)
    foreign[1]["edges"] = [
        {"source": EXPERIMENT_ID, "target": "blk/capacity", "relation": "blocked_by"}
    ]
    assert "experiment-loop-self-attachment" in _codes(
        validate_patch(state, _patch(foreign), ["repo"])
    )

    # And it may only attach to its own experiment.
    foreign_source = deepcopy(attached_ops)
    foreign_source[1]["edges"] = [
        {"source": "exp/other", "target": "ev/result", "relation": "produces"}
    ]
    assert "experiment-loop-self-attachment" in _codes(
        validate_patch(state, _patch(foreign_source), ["repo"])
    )


def test_proposal_only_iteration_is_typed_and_scoped_to_a_tested_hypothesis() -> None:
    state = _state()
    proposal = {
        "id": "prop/activate-target",
        "title": "Activate the tested hypothesis",
        "card": {
            "situation_cold": "The new result supports the tested hypothesis.",
            "why_human_now": "Only the human controls the belief transition.",
            "consequences": "The hypothesis becomes active.",
            "decision_needed": "Approve or reject this belief transition.",
        },
        "ops": [
            {
                "op": "update_nodes",
                "intent": "status_change",
                "nodes": [
                    {
                        "id": "hyp/target",
                        "changes": {"status": "active"},
                        "cause": {
                            "kind": "evidence_edge",
                            "ref_id": "ev/proposal::supports::hyp/target",
                        },
                    }
                ],
            }
        ],
        "related_node_ids": ["hyp/target"],
        "base_rev": 3,
    }
    evidence_ops = [
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "ev/proposal",
                    "type": "evidence",
                    "title": "Proposal evidence",
                    "observation": "The experiment supported the target.",
                    "origin": "internal_run",
                }
            ],
        },
        {
            "op": "create_edges",
            "edges": [
                {
                    "source": "ev/proposal",
                    "target": "hyp/target",
                    "relation": "supports",
                    "assessment": {
                        "relevance": "direct",
                        "weight": "moderate",
                        "qualifications": [],
                    },
                }
            ],
        },
    ]
    proposal_attempt = _attempt(attempt_kind="proposal_only", status="completed").model_copy(
        update={"job_refs": []}
    )
    valid = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [{"id": EXPERIMENT_ID, "changes": {"attempts": [proposal_attempt]}}],
            },
            *evidence_ops,
            {"op": "create_proposals", "proposals": [proposal]},
        ]
    )
    assert not validate_patch(state, valid, ["repo"]).rejected

    running_state = _state(attempts=[_attempt()])
    closed_attempt = _attempt(status="completed").model_copy(update={"outcome": "Needs change."})
    close_and_propose = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [{"id": EXPERIMENT_ID, "changes": {"attempts": [closed_attempt]}}],
            },
            *evidence_ops,
            {"op": "create_proposals", "proposals": [proposal]},
        ]
    )
    assert not validate_patch(running_state, close_and_propose, ["repo"]).rejected

    # Proposal dependencies are RCP bookkeeping, not an agent declaration or
    # an experiment-loop authority input. The semantic target controls scope.
    stale_bookkeeping = deepcopy(proposal)
    stale_bookkeeping["related_node_ids"] = [DECISION_ID]
    codes = _codes(
        validate_patch(
            state,
            _patch([*evidence_ops, {"op": "create_proposals", "proposals": [stale_bookkeeping]}]),
            ["repo"],
        )
    )
    assert "experiment-loop-proposal-scope" not in codes
    assert "experiment-loop-proposal-operations" not in codes

    # A target that is not a Hypothesis this experiment tests is refused.
    hidden_foreign_update = deepcopy(proposal)
    hidden_foreign_update["ops"] = [
        {
            "op": "update_nodes",
            "intent": "content_change",
            "nodes": [{"id": "blk/capacity", "changes": {"status": "resolved"}}],
        }
    ]
    hidden_foreign_update["related_node_ids"] = [DECISION_ID]
    codes = _codes(
        validate_patch(
            state,
            _patch(
                [
                    {
                        "op": "update_nodes",
                        "nodes": [
                            {
                                "id": EXPERIMENT_ID,
                                "changes": {"attempts": [proposal_attempt]},
                            }
                        ],
                    },
                    {"op": "create_proposals", "proposals": [hidden_foreign_update]},
                ]
            ),
            ["repo"],
        )
    )
    assert "experiment-loop-proposal-operations" in codes


def test_proposal_only_attempt_requires_a_proposal_in_the_same_patch() -> None:
    attempt = _attempt(attempt_kind="proposal_only", status="completed").model_copy(
        update={"job_refs": []}
    )
    patch = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [{"id": EXPERIMENT_ID, "changes": {"attempts": [attempt]}}],
            }
        ]
    )
    assert "experiment-loop-proposal-attempt" in _codes(validate_patch(_state(), patch, ["repo"]))

    launched_proposal_attempt = _attempt(attempt_kind="proposal_only", status="completed")
    launched_ops = [
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": EXPERIMENT_ID,
                    "changes": {"attempts": [launched_proposal_attempt]},
                }
            ],
        },
        {
            "op": "create_proposals",
            "proposals": [
                {
                    "id": "prop/with-job",
                    "title": "Change resources",
                    "card": {},
                    "ops": [
                        {
                            "op": "update_nodes",
                            "intent": "content_change",
                            "nodes": [{"id": DECISION_ID, "changes": {"status": "revisit"}}],
                        }
                    ],
                    "related_node_ids": [DECISION_ID],
                    "base_rev": 3,
                }
            ],
        },
    ]
    launched = _patch(launched_ops)
    assert "experiment-loop-proposal-job" in _codes(validate_patch(_state(), launched, ["repo"]))

    nonterminal_attempt = attempt.model_copy(update={"status": "running"})
    nonterminal_ops = deepcopy(launched_ops)
    nonterminal_ops[0]["nodes"][0]["changes"]["attempts"] = [nonterminal_attempt]
    nonterminal = _patch(nonterminal_ops)
    assert "experiment-loop-proposal-status" in _codes(
        validate_patch(_state(), nonterminal, ["repo"])
    )


def test_loop_patch_requires_persisted_rcp_control_binding() -> None:
    report = validate_patch(_state(), _patch([], stamp=False), ["repo"])
    assert "experiment-loop-control-node" in _codes(report)


def test_old_attempts_load_with_backward_compatible_control_defaults() -> None:
    attempt = ExperimentAttempt(id="old", sequence=1, purpose="Legacy attempt")
    assert attempt.attempt_kind == "external_run"
    assert attempt.decision_bundle == []
    assert attempt.debug is None


def test_experiment_loop_may_queue_its_pinned_decision_but_never_decide_it() -> None:
    # A queue and a decision are the same operation shape on the same node: one
    # update_nodes changing only "status". Only the requested status separates
    # the move the loop may make from the one reserved for the human, so the
    # rule has to be read off the status rather than off the shape.
    state = _state()

    for queued in ("open", "ready", "revisit"):
        allowed = _patch(
            [{"op": "update_nodes", "nodes": [{"id": DECISION_ID, "changes": {"status": queued}}]}]
        )
        assert "experiment-loop-decision-action" not in _codes(
            validate_patch(state, allowed, ["repo"])
        )

    decided = _patch(
        [{"op": "update_nodes", "nodes": [{"id": DECISION_ID, "changes": {"status": "decided"}}]}]
    )
    assert "experiment-loop-decision-action" in _codes(validate_patch(state, decided, ["repo"]))


def test_experiment_loop_cannot_rewrite_an_attempt_it_already_closed() -> None:
    # A finished attempt is a record. Reopening one is caught elsewhere; this is
    # the subtler move of rewriting a closed attempt into a different closed
    # state, which leaves every fixed field untouched and so reaches the
    # close check rather than the mutation check.
    closed = _attempt(status="completed")
    state = _state(attempts=[closed])

    rewritten = closed.model_dump(mode="json") | {
        "status": "failed",
        "failure_reason": "Rewriting history after the fact.",
    }
    patch = _patch(
        [
            {
                "op": "update_nodes",
                "nodes": [{"id": EXPERIMENT_ID, "changes": {"attempts": [rewritten]}}],
            }
        ]
    )

    codes = _codes(validate_patch(state, patch, ["repo"]))
    assert "experiment-loop-attempt-close" in codes
    assert "experiment-loop-attempt-mutation" not in codes
