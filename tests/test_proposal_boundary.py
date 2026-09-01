from __future__ import annotations

import pytest
from pydantic import ValidationError

from rcp.core.materialize import apply_valid_patch
from rcp.core.models import Patch
from rcp.core.operations import adapt_persisted_patch_document, graph_operations_from_proposal
from rcp.core.validation import validate_patch
from rcp.core.validation.proposals import normalized_decision_proposal_ops, proposal_is_stale
from rcp.history import HistoryManager
from tests.helpers import seed_patch


def _agent_patch(*ops: dict) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Exercised the minimal Proposal boundary.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=list(ops),
    )


def _legacy_patch(*ops: dict, revision: int = 0) -> Patch:
    """Decode intentionally old persisted operation documents through the adapter."""

    return Patch.model_validate(
        adapt_persisted_patch_document(
            {
                "revision": revision,
                "kind": "refresh",
                "author": "agent",
                "summary": "Historical Proposal fixture.",
                "run_truth_scope": ["repo-a"],
                "repositories_read": ["repo-a"],
                "ops": list(ops),
            }
        )
    )


def _proposal(*, proposal_id: str, node_id: str, changes: dict, cause: dict | None = None) -> dict:
    update = {"id": node_id, "changes": changes}
    if cause is not None:
        update["cause"] = cause
    return _intent_proposal(
        proposal_id=proposal_id,
        operation={"op": "update_nodes", "intent": "status_change", "nodes": [update]},
        base_rev=2,
    )


def _intent_proposal(*, proposal_id: str, operation: dict, base_rev: int) -> dict:
    return {
        "op": "create_proposals",
        "proposals": [
            {
                "id": proposal_id,
                "title": "Review the semantic transition",
                "card": {
                    "situation_cold": "The research state now supports a semantic transition.",
                    "why_human_now": "Only the human controls this transition.",
                    "consequences": "The selected research state will change.",
                    "decision_needed": "Approve or reject the transition.",
                },
                "ops": [operation],
                "base_rev": base_rev,
            }
        ],
    }


def _state_with_decision(manifest, *, governed: bool = True):
    history = HistoryManager(manifest)
    history.append(seed_patch())
    edges = [
        {
            "source": "exp/evaluation",
            "target": "hyp/replanning-restores-plasticity",
            "relation": "tests",
        },
        {
            "id": "edge/evaluation-support",
            "source": "ev/evaluation-result",
            "target": "hyp/replanning-restores-plasticity",
            "relation": "supports",
            "assessment": {
                "relevance": "direct",
                "weight": "moderate",
                "scope": "Matched evaluation conditions.",
            },
        },
        {
            "source": "exp/evaluation",
            "target": "ev/evaluation-result",
            "relation": "produces",
        },
    ]
    if governed:
        edges.append(
            {
                "source": "exp/evaluation",
                "target": "dec/evaluation-rule",
                "relation": "governed_by",
            }
        )
    history.append(
        _agent_patch(
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "dec/evaluation-rule",
                        "type": "decision",
                        "title": "Evaluation rule",
                        "question": "Which evaluation rule should govern the experiment?",
                        "options": ["matched", "shifted"],
                    },
                    {
                        "id": "exp/evaluation",
                        "type": "experiment",
                        "title": "Evaluation",
                        "objective": "Evaluate the intervention under the chosen rule.",
                    },
                    {
                        "id": "ev/evaluation-result",
                        "type": "evidence",
                        "title": "Evaluation result",
                        "observation": "The matched evaluation improved.",
                        "origin": "internal_run",
                    },
                ],
            },
            {"op": "create_edges", "edges": edges},
        )
    )
    return history.state()


def _state_with_second_hypothesis(manifest):
    state = _state_with_decision(manifest)
    addition = _agent_patch(
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "hyp/alternative-mechanism",
                    "type": "hypothesis",
                    "title": "Alternative mechanism",
                    "statement": "A distinct mechanism explains the same result.",
                }
            ],
        }
    ).model_copy(update={"revision": state.revision + 1})
    report = validate_patch(state, addition, ["repo-a", "repo-b"])
    assert not report.rejected
    return apply_valid_patch(state, addition)


def test_ordinary_agent_cannot_edit_an_existing_research_question_directly(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Accepted the question.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": "rq/learning-after-shift",
                    "standing": "accepted",
                }
            ],
        )
    )

    patch = _agent_patch(
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "rq/learning-after-shift",
                    "changes": {"motivation": "Repeated shifts make this question urgent."},
                }
            ],
        }
    )
    report = validate_patch(history.state(), patch, ["repo-a"])

    node = history.state().nodes["rq/learning-after-shift"]
    assert report.rejected
    assert any(
        message.code == "graph-action-refused" and "update_protected_epistemic" in message.message
        for message in report.messages
    )
    assert node.motivation == "Persistent agents encounter repeated changes."
    assert node.standing == "accepted"


def test_agent_cannot_decide_directly_or_by_proposal(manifest) -> None:
    state = _state_with_decision(manifest)
    direct_report = validate_patch(
        state,
        _agent_patch(
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "dec/evaluation-rule",
                        "changes": {"status": "decided", "selected_option": "matched"},
                    }
                ],
            }
        ),
        ["repo-a", "repo-b"],
    )
    proposal_report = validate_patch(
        state,
        _agent_patch(
            _proposal(
                proposal_id="prop/select-evaluation",
                node_id="dec/evaluation-rule",
                changes={"status": "decided", "selected_option": "matched"},
            )
        ),
        ["repo-a", "repo-b"],
    )

    assert direct_report.rejected
    assert proposal_report.rejected
    assert any("decide_decision" in message.message for message in direct_report.messages)
    assert any("decide_decision" in message.message for message in proposal_report.messages)


def test_belief_transition_still_requires_an_exact_proposal(manifest) -> None:
    state = _state_with_decision(manifest)
    direct_update = {
        "id": "hyp/replanning-restores-plasticity",
        "changes": {"status": "active"},
        "cause": {"kind": "evidence_edge", "ref_id": "edge/evaluation-support"},
    }
    proposal = _proposal(
        proposal_id="prop/activate-hypothesis",
        node_id="hyp/replanning-restores-plasticity",
        changes={"status": "active"},
        cause={"kind": "evidence_edge", "ref_id": "edge/evaluation-support"},
    )

    direct_report = validate_patch(
        state,
        _agent_patch({"op": "update_nodes", "nodes": [direct_update]}),
        ["repo-a", "repo-b"],
    )
    proposal_report = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])

    assert direct_report.rejected
    assert not proposal_report.rejected


def test_every_declared_protected_intent_is_admitted_as_one_human_question(manifest) -> None:
    state = _state_with_second_hypothesis(manifest)
    edge_id = "rq/learning-after-shift::has_hypothesis::hyp/replanning-restores-plasticity"
    operations = {
        "content": {
            "op": "update_nodes",
            "intent": "content_change",
            "nodes": [
                {
                    "id": "rq/learning-after-shift",
                    "changes": {"question": "Can adaptation remain plastic after repeated shifts?"},
                }
            ],
        },
        "removal": {
            "op": "remove_nodes",
            "intent": "removal",
            "node_ids": ["hyp/replanning-restores-plasticity"],
        },
        "supersede": {
            "op": "supersede_nodes",
            "intent": "supersede",
            "nodes": [
                {
                    "id": "hyp/replanning-restores-plasticity",
                    "superseded_by": "hyp/alternative-mechanism",
                    "explanation": "The alternative states the mechanism more precisely.",
                }
            ],
        },
        "merge": {
            "op": "merge_nodes",
            "intent": "merge",
            "merges": [
                {
                    "duplicate": "hyp/replanning-restores-plasticity",
                    "canonical": "hyp/alternative-mechanism",
                    "explanation": "Both records state the same mechanism.",
                }
            ],
        },
        "protected-create": {
            "op": "create_edges",
            "intent": "protected_relation_change",
            "edges": [
                {
                    "source": "rq/learning-after-shift",
                    "target": "hyp/alternative-mechanism",
                    "relation": "has_hypothesis",
                }
            ],
        },
        "protected-remove": {
            "op": "remove_edges",
            "intent": "protected_relation_change",
            "edge_ids": [edge_id],
        },
        "status": {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/replanning-restores-plasticity",
                    "changes": {"status": "active"},
                    "cause": {"kind": "evidence_edge", "ref_id": "edge/evaluation-support"},
                }
            ],
        },
    }

    for suffix, operation in operations.items():
        patch = _agent_patch(
            _intent_proposal(
                proposal_id=f"prop/{suffix}",
                operation=operation,
                base_rev=state.revision,
            )
        )
        report = validate_patch(state, patch, ["repo-a", "repo-b"])

        assert not report.rejected, [message.message for message in report.messages]


def test_research_question_lifecycle_uses_content_change_without_evidence(manifest) -> None:
    state = _state_with_decision(manifest)
    proposal = _intent_proposal(
        proposal_id="prop/answer-question",
        operation={
            "op": "update_nodes",
            "intent": "content_change",
            "nodes": [
                {
                    "id": "rq/learning-after-shift",
                    "changes": {"status": "answered"},
                }
            ],
        },
        base_rev=state.revision,
    )

    report = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])

    assert not report.rejected


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "rq/learning-after-shift",
                    "changes": {"question": "Can plasticity persist after repeated shifts?"},
                }
            ],
        },
        {
            "op": "remove_nodes",
            "intent": "removal",
            "node_ids": [
                "rq/learning-after-shift",
                "hyp/replanning-restores-plasticity",
            ],
        },
        {
            "op": "merge_nodes",
            "intent": "merge",
            "merges": [
                {
                    "duplicate": "hyp/replanning-restores-plasticity",
                    "canonical": "hyp/alternative-mechanism",
                },
                {
                    "duplicate": "hyp/alternative-mechanism",
                    "canonical": "hyp/replanning-restores-plasticity",
                },
            ],
        },
        {
            "op": "create_edges",
            "intent": "protected_relation_change",
            "edges": [
                {
                    "source": "ev/evaluation-result",
                    "target": "hyp/replanning-restores-plasticity",
                    "relation": "supports",
                }
            ],
        },
    ],
)
def test_agent_proposal_rejects_missing_mismatched_or_bundled_intent(manifest, operation) -> None:
    state = _state_with_second_hypothesis(manifest)
    proposal = _intent_proposal(
        proposal_id="prop/invalid-intent",
        operation=operation,
        base_rev=state.revision,
    )

    if "intent" not in operation:
        with pytest.raises(ValidationError, match="intent"):
            _agent_patch(proposal)
        return

    report = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])

    assert report.rejected
    assert any(message.code == "invalid-agent-proposal-shape" for message in report.messages)


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "rq/learning-after-shift",
                    "changes": {"question": "Can plasticity persist after repeated shifts?"},
                }
            ],
        },
        {
            "op": "remove_nodes",
            "node_ids": ["hyp/replanning-restores-plasticity"],
        },
        {
            "op": "supersede_nodes",
            "nodes": [
                {
                    "id": "hyp/replanning-restores-plasticity",
                    "superseded_by": "hyp/alternative-mechanism",
                }
            ],
        },
        {
            "op": "merge_nodes",
            "merges": [
                {
                    "duplicate": "hyp/replanning-restores-plasticity",
                    "canonical": "hyp/alternative-mechanism",
                }
            ],
        },
        {
            "op": "create_edges",
            "edges": [
                {
                    "source": "rq/learning-after-shift",
                    "target": "hyp/alternative-mechanism",
                    "relation": "has_hypothesis",
                }
            ],
        },
        {
            "op": "remove_edges",
            "edge_ids": [
                "rq/learning-after-shift::has_hypothesis::hyp/replanning-restores-plasticity"
            ],
        },
        {
            "op": "set_standing",
            "node_id": "hyp/replanning-restores-plasticity",
            "standing": "accepted",
        },
    ],
)
def test_direct_agent_changes_to_existing_beliefs_are_refused_at_apply(manifest, operation) -> None:
    state = _state_with_second_hypothesis(manifest)

    report = validate_patch(state, _agent_patch(operation), ["repo-a", "repo-b"])

    assert report.rejected
    assert any(message.code == "graph-action-refused" for message in report.messages)


def test_attaching_evidence_to_an_existing_hypothesis_stays_direct(manifest) -> None:
    state = _state_with_decision(manifest)
    patch = _agent_patch(
        {
            "op": "create_edges",
            "edges": [
                {
                    "source": "ev/evaluation-result",
                    "target": "hyp/replanning-restores-plasticity",
                    "relation": "weakens",
                    "explanation": "The alternative evaluation narrows the claim.",
                    "assessment": {
                        "relevance": "direct",
                        "weight": "limited",
                        "scope": "The alternative evaluation condition.",
                    },
                }
            ],
        }
    )

    report = validate_patch(state, patch, ["repo-a", "repo-b"])

    assert not report.rejected


def test_replay_reads_a_historical_ambiguity_carrying_its_derived_revision() -> None:
    """`raised_rev` was persisted by older releases and is forbidden on the payload now.

    Materialization always overwrites it with the applying revision, so the stored
    value was inert -- but the strict payload made a Patch RCP itself wrote
    unreadable, and replay halted on it at the project's second revision.
    """

    raw = {
        "revision": 2,
        "kind": "refresh",
        "author": "agent",
        "summary": "Record the open questions.",
        "run_truth_scope": ["repo-a"],
        "repositories_read": ["repo-a"],
        "ops": [
            {
                "op": "create_ambiguities",
                "ambiguities": [
                    {
                        "id": "amb/scope",
                        "question": "Which corpus counts?",
                        "why_it_matters": "It changes the denominator.",
                        "raised_rev": 0,
                    }
                ],
            }
        ],
    }

    patch = Patch.model_validate(adapt_persisted_patch_document(raw))

    assert patch.schema_generation == 1
    assert patch.ops[0].ambiguities[0].id == "amb/scope"
    assert not hasattr(patch.ops[0].ambiguities[0], "raised_rev")


def test_replay_accepts_historical_proposals_without_declared_intent(manifest) -> None:
    state = _state_with_decision(manifest)
    current = _agent_patch(
        _proposal(
            proposal_id="prop/legacy-status",
            node_id="hyp/replanning-restores-plasticity",
            changes={"status": "active"},
            cause={"kind": "evidence_edge", "ref_id": "edge/evaluation-support"},
        )
    )
    raw = current.model_dump(mode="python", exclude_unset=True)
    raw.pop("schema_generation", None)
    del raw["ops"][0]["proposals"][0]["ops"][0]["intent"]
    raw["revision"] = state.revision + 1
    patch = Patch.model_validate(adapt_persisted_patch_document(raw))

    report = validate_patch(state, patch, ["repo-a", "repo-b"], mode="replay")

    assert not report.rejected


def test_replay_does_not_add_expected_absence_to_legacy_relation_proposals(manifest) -> None:
    state = _state_with_decision(manifest)
    edge_id = "rq/learning-after-shift::has_hypothesis::hyp/replanning-restores-plasticity"
    proposal_patch = _legacy_patch(
        {
            "op": "create_proposals",
            "proposals": [
                {
                    "id": "prop/legacy-create-relation",
                    "title": "Legacy relation proposal",
                    "card": {"decision_needed": "Approve the historical relation."},
                    "ops": [
                        {
                            "op": "create_edges",
                            "edges": [
                                {
                                    "id": edge_id,
                                    "source": "rq/learning-after-shift",
                                    "target": "hyp/replanning-restores-plasticity",
                                    "relation": "has_hypothesis",
                                }
                            ],
                        }
                    ],
                    "base_rev": state.revision,
                    "raised_rev": state.revision,
                }
            ],
        }
    )
    proposal = proposal_patch.ops[0].proposals[0]  # type: ignore[union-attr]

    assert not proposal_is_stale(state, proposal)


def test_same_patch_edge_recreation_stales_a_new_proposal_but_remains_replayable(manifest) -> None:
    state = _state_with_decision(manifest)
    edge_id = "edge/evaluation-support"
    patch = _agent_patch(
        _proposal(
            proposal_id="prop/status-before-recreate",
            node_id="hyp/replanning-restores-plasticity",
            changes={"status": "active"},
            cause={"kind": "evidence_edge", "ref_id": edge_id},
        ),
        {"op": "remove_edges", "edge_ids": [edge_id]},
        {
            "op": "create_edges",
            "edges": [
                {
                    "id": edge_id,
                    "source": "ev/evaluation-result",
                    "target": "hyp/replanning-restores-plasticity",
                    "relation": "supports",
                }
            ],
        },
    )

    admission = validate_patch(state, patch, ["repo-a", "repo-b"])
    historical = validate_patch(
        state,
        patch.model_copy(update={"revision": state.revision + 1}),
        ["repo-a", "repo-b"],
        mode="replay",
    )

    assert admission.rejected
    assert any(message.code == "stale-created-proposal" for message in admission.messages)
    assert not historical.rejected


def test_replay_does_not_recheck_protected_action_permission(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    state = history.state()
    patch = _agent_patch(
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "rq/learning-after-shift",
                    "changes": {"motivation": "Historical agent wording remains replayable."},
                }
            ],
        }
    ).model_copy(update={"revision": state.revision + 1})

    admission = validate_patch(state, patch, ["repo-a", "repo-b"])
    replay = validate_patch(state, patch, ["repo-a", "repo-b"], mode="replay")

    assert admission.rejected
    assert not replay.rejected


@pytest.mark.parametrize("recreated", [False, True])
def test_protected_relation_removal_stales_when_the_named_edge_moves(
    manifest, recreated: bool
) -> None:
    state = _state_with_second_hypothesis(manifest)
    edge_id = "rq/learning-after-shift::has_hypothesis::hyp/replanning-restores-plasticity"
    patch = _agent_patch(
        _intent_proposal(
            proposal_id="prop/remove-protected-relation",
            operation={
                "op": "remove_edges",
                "intent": "protected_relation_change",
                "edge_ids": [edge_id],
            },
            base_rev=state.revision,
        )
    ).model_copy(update={"revision": state.revision + 1})
    assert not validate_patch(state, patch, ["repo-a", "repo-b"]).rejected
    proposed = apply_valid_patch(state, patch)
    edges = dict(proposed.edges)
    previous = edges.pop(edge_id)
    if recreated:
        edges[edge_id] = previous.model_copy(update={"created_rev": proposed.revision + 1})
    moved = proposed.model_copy(update={"revision": proposed.revision + 1, "edges": edges})

    assert proposal_is_stale(moved, proposed.proposals["prop/remove-protected-relation"])


def test_removal_proposal_snapshots_exact_incident_edges_and_stales_on_set_change(
    manifest,
) -> None:
    state = _state_with_second_hypothesis(manifest)
    target_id = "hyp/replanning-restores-plasticity"
    patch = _agent_patch(
        _intent_proposal(
            proposal_id="prop/remove-replanning",
            operation={
                "op": "remove_nodes",
                "intent": "removal",
                "node_ids": [target_id],
            },
            base_rev=state.revision,
        )
    ).model_copy(update={"revision": state.revision + 1})
    assert not validate_patch(state, patch, ["repo-a", "repo-b"]).rejected
    proposed = apply_valid_patch(state, patch)
    proposal = proposed.proposals["prop/remove-replanning"]

    assert proposal.related_edge_ids == sorted(
        edge.id
        for edge in proposed.edges.values()
        if edge.source == target_id or edge.target == target_id
    )

    unrelated = _agent_patch(
        {
            "op": "create_edges",
            "edges": [
                {
                    "id": "edge/unrelated-informs",
                    "source": "ev/evaluation-result",
                    "target": "dec/evaluation-rule",
                    "relation": "informs",
                }
            ],
        }
    ).model_copy(update={"revision": proposed.revision + 1})
    with_unrelated = apply_valid_patch(proposed, unrelated)
    assert not proposal_is_stale(with_unrelated, proposal)

    incident = _agent_patch(
        {
            "op": "create_edges",
            "edges": [
                {
                    "id": "edge/new-incident-evidence",
                    "source": "ev/evaluation-result",
                    "target": target_id,
                    "relation": "weakens",
                }
            ],
        }
    ).model_copy(update={"revision": proposed.revision + 1})
    with_incident = apply_valid_patch(proposed, incident)
    assert proposal_is_stale(with_incident, proposal)


@pytest.mark.parametrize("recreated", [False, True])
def test_removal_proposal_stales_when_a_snapshotted_incident_edge_is_removed_or_recreated(
    manifest,
    recreated: bool,
) -> None:
    state = _state_with_second_hypothesis(manifest)
    target_id = "hyp/replanning-restores-plasticity"
    patch = _agent_patch(
        _intent_proposal(
            proposal_id="prop/remove-replanning",
            operation={
                "op": "remove_nodes",
                "intent": "removal",
                "node_ids": [target_id],
            },
            base_rev=state.revision,
        )
    ).model_copy(update={"revision": state.revision + 1})
    proposed = apply_valid_patch(state, patch)
    proposal = proposed.proposals["prop/remove-replanning"]
    edge_id = proposal.related_edge_ids[0]
    edges = dict(proposed.edges)
    previous = edges.pop(edge_id)
    if recreated:
        edges[edge_id] = previous.model_copy(update={"created_rev": proposed.revision + 1})
    moved = proposed.model_copy(update={"revision": proposed.revision + 1, "edges": edges})

    assert proposal_is_stale(moved, proposal)


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "supersede_nodes",
            "intent": "supersede",
            "nodes": [
                {
                    "id": "hyp/replanning-restores-plasticity",
                    "superseded_by": "rq/learning-after-shift",
                }
            ],
        },
        {
            "op": "merge_nodes",
            "intent": "merge",
            "merges": [
                {
                    "duplicate": "hyp/replanning-restores-plasticity",
                    "canonical": "rq/learning-after-shift",
                }
            ],
        },
    ],
)
def test_supersede_and_merge_intents_require_matching_protected_belief_types(
    manifest,
    operation,
) -> None:
    state = _state_with_second_hypothesis(manifest)
    proposal = _intent_proposal(
        proposal_id="prop/cross-type-belief",
        operation=operation,
        base_rev=state.revision,
    )

    report = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])

    assert report.rejected
    assert any(
        message.code == "invalid-agent-proposal-shape"
        and "same protected belief type" in message.message
        for message in report.messages
    )


@pytest.mark.parametrize(
    ("relation", "target_id"),
    [
        ("supersedes", "rq/learning-after-shift"),
        ("duplicate_of", "dec/evaluation-rule"),
    ],
)
def test_protected_relation_intent_cannot_bypass_lifecycle_intent_rules(
    manifest,
    relation: str,
    target_id: str,
) -> None:
    state = _state_with_decision(manifest)
    proposal = _intent_proposal(
        proposal_id=f"prop/bypass-{relation}",
        operation={
            "op": "create_edges",
            "intent": "protected_relation_change",
            "edges": [
                {
                    "source": "hyp/replanning-restores-plasticity",
                    "target": target_id,
                    "relation": relation,
                }
            ],
        },
        base_rev=state.revision,
    )

    report = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])

    assert report.rejected
    assert any(
        message.code == "invalid-agent-proposal-shape"
        and "dedicated supersede or merge intent" in message.message
        for message in report.messages
    )


@pytest.mark.parametrize("intent", ["supersede", "merge"])
def test_lifecycle_proposal_may_forward_reference_its_new_same_type_target(
    manifest,
    intent: str,
) -> None:
    state = _state_with_decision(manifest)
    target_id = "hyp/forward-lifecycle-target"
    operation = (
        {
            "op": "supersede_nodes",
            "intent": "supersede",
            "nodes": [
                {
                    "id": "hyp/replanning-restores-plasticity",
                    "superseded_by": target_id,
                }
            ],
        }
        if intent == "supersede"
        else {
            "op": "merge_nodes",
            "intent": "merge",
            "merges": [
                {
                    "duplicate": "hyp/replanning-restores-plasticity",
                    "canonical": target_id,
                }
            ],
        }
    )
    patch = _agent_patch(
        _intent_proposal(
            proposal_id=f"prop/forward-{intent}",
            operation=operation,
            base_rev=state.revision,
        ),
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": target_id,
                    "type": "hypothesis",
                    "title": "Forward lifecycle target",
                    "statement": "This newly asserted belief is the proposed lifecycle target.",
                }
            ],
        },
    ).model_copy(update={"revision": state.revision + 1})

    report = validate_patch(state, patch, ["repo-a", "repo-b"])

    assert not report.rejected, [message.message for message in report.messages]
    staged = apply_valid_patch(state, patch)
    assert target_id in staged.proposals[f"prop/forward-{intent}"].related_node_ids


@pytest.mark.parametrize("intent", ["supersede", "merge"])
def test_later_update_of_lifecycle_target_stales_new_proposal_but_replay_is_tolerant(
    manifest,
    intent: str,
) -> None:
    state = _state_with_second_hypothesis(manifest)
    operation = (
        {
            "op": "supersede_nodes",
            "intent": "supersede",
            "nodes": [
                {
                    "id": "hyp/replanning-restores-plasticity",
                    "superseded_by": "hyp/alternative-mechanism",
                }
            ],
        }
        if intent == "supersede"
        else {
            "op": "merge_nodes",
            "intent": "merge",
            "merges": [
                {
                    "duplicate": "hyp/replanning-restores-plasticity",
                    "canonical": "hyp/alternative-mechanism",
                }
            ],
        }
    )
    patch = _agent_patch(
        _intent_proposal(
            proposal_id=f"prop/moving-{intent}-target",
            operation=operation,
            base_rev=state.revision,
        ),
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "hyp/alternative-mechanism",
                    "changes": {"statement": "A later operation replaced the judged wording."},
                }
            ],
        },
    ).model_copy(update={"revision": state.revision + 1})

    admission = validate_patch(state, patch, ["repo-a", "repo-b"])
    replay = validate_patch(state, patch, ["repo-a", "repo-b"], mode="replay")

    assert any(message.code == "stale-created-proposal" for message in admission.messages)
    assert not replay.rejected, [message.message for message in replay.messages]


def test_later_node_removal_and_recreation_stales_new_proposal_but_replay_is_tolerant(
    manifest,
) -> None:
    state = _state_with_decision(manifest)
    node_id = "rq/learning-after-shift"
    patch = _agent_patch(
        _intent_proposal(
            proposal_id="prop/replaced-question",
            operation={
                "op": "update_nodes",
                "intent": "content_change",
                "nodes": [
                    {
                        "id": node_id,
                        "changes": {"question": "Does the replacement retain plasticity?"},
                    }
                ],
            },
            base_rev=state.revision,
        ),
        {"op": "remove_nodes", "node_ids": [node_id]},
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": node_id,
                    "type": "research_question",
                    "title": "Replacement learning question",
                    "question": "Can a replacement question preserve the original identity?",
                }
            ],
        },
    ).model_copy(update={"revision": state.revision + 1})

    admission = validate_patch(state, patch, ["repo-a", "repo-b"])
    replay = validate_patch(state, patch, ["repo-a", "repo-b"], mode="replay")

    assert any(message.code == "stale-created-proposal" for message in admission.messages)
    assert not replay.rejected, [message.message for message in replay.messages]


@pytest.mark.parametrize(
    ("run_scope", "repositories_read", "expected_code"),
    [
        (["repo-a"], ["repo-a"], "source-outside-run-scope"),
        (["repo-a", "repo-b"], ["repo-a"], "unread-source-repository"),
    ],
)
def test_content_proposal_source_refs_retain_originating_agent_scope(
    manifest,
    run_scope: list[str],
    repositories_read: list[str],
    expected_code: str,
) -> None:
    state = _state_with_decision(manifest)
    proposal = _intent_proposal(
        proposal_id=f"prop/source-scope-{expected_code}",
        operation={
            "op": "update_nodes",
            "intent": "content_change",
            "nodes": [
                {
                    "id": "rq/learning-after-shift",
                    "changes": {
                        "source_refs": [
                            {
                                "machine": "laptop",
                                "truth_repository": "repo-b",
                                "source": "codex",
                                "session_id": "session-source-scope",
                                "record_uuid": "record-source-scope",
                                "timestamp": "2026-08-12T00:00:00Z",
                                "excerpt": "This source belongs to repo-b.",
                            }
                        ]
                    },
                }
            ],
        },
        base_rev=state.revision,
    )
    patch = _agent_patch(proposal).model_copy(
        update={"run_truth_scope": run_scope, "repositories_read": repositories_read}
    )

    report = validate_patch(state, patch, ["repo-a", "repo-b"])

    assert report.rejected
    assert any(message.code == expected_code for message in report.messages)


def test_duplicate_proposal_ids_in_one_create_operation_are_rejected(manifest) -> None:
    state = _state_with_decision(manifest)
    operation = _proposal(
        proposal_id="prop/duplicate-id",
        node_id="hyp/replanning-restores-plasticity",
        changes={"status": "active"},
        cause={"kind": "evidence_edge", "ref_id": "edge/evaluation-support"},
    )
    duplicate = dict(operation["proposals"][0])
    operation["proposals"].append(duplicate)

    report = validate_patch(state, _agent_patch(operation), ["repo-a", "repo-b"])

    assert report.rejected
    assert any(message.code == "duplicate-proposal-id" for message in report.messages)


def test_legacy_duplicate_proposal_ids_remain_replayable(manifest) -> None:
    state = _state_with_decision(manifest)
    operation = _proposal(
        proposal_id="prop/duplicate-id",
        node_id="hyp/replanning-restores-plasticity",
        changes={"status": "active"},
        cause={"kind": "evidence_edge", "ref_id": "edge/evaluation-support"},
    )
    operation["proposals"].append(dict(operation["proposals"][0]))
    historical = _agent_patch(operation).model_copy(update={"revision": state.revision + 1})

    report = validate_patch(
        state,
        historical,
        ["repo-a", "repo-b"],
        mode="replay",
    )

    assert not report.rejected


def test_new_decision_proposal_is_refused_regardless_of_governing_edge(manifest) -> None:
    state = _state_with_decision(manifest, governed=False)
    proposal = _proposal(
        proposal_id="prop/select-evaluation",
        node_id="dec/evaluation-rule",
        changes={"status": "decided", "selected_option": "matched"},
    )

    ungoverned = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])
    historical = validate_patch(
        state,
        _agent_patch(proposal).model_copy(update={"revision": 3}),
        ["repo-a", "repo-b"],
        mode="replay",
    )
    same_patch_governed = validate_patch(
        state,
        _agent_patch(
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "exp/evaluation",
                        "target": "dec/evaluation-rule",
                        "relation": "governed_by",
                    }
                ],
            },
            proposal,
        ),
        ["repo-a", "repo-b"],
    )
    same_patch_experiment = validate_patch(
        state,
        _agent_patch(
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "exp/same-patch",
                        "target": "dec/evaluation-rule",
                        "relation": "governed_by",
                    }
                ],
            },
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "exp/same-patch",
                        "type": "experiment",
                        "title": "Same-patch experiment",
                        "objective": "Use the proposed evaluation rule.",
                    }
                ],
            },
            proposal,
        ),
        ["repo-a", "repo-b"],
    )

    assert ungoverned.rejected
    assert any(message.code == "invalid-agent-proposal-shape" for message in ungoverned.messages)
    assert not historical.rejected
    assert same_patch_governed.rejected
    assert same_patch_experiment.rejected


@pytest.mark.parametrize(
    "changes",
    [
        {"selected_option": "matched"},
        {"selected_option": "not-listed", "status": "revisit"},
        {"status": "decided"},
    ],
)
def test_every_new_decision_proposal_shape_is_refused(manifest, changes) -> None:
    state = _state_with_decision(manifest)
    proposal = _proposal(
        proposal_id="prop/incoherent-evaluation",
        node_id="dec/evaluation-rule",
        changes=changes,
    )

    report = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])

    assert report.rejected
    assert any(
        message.code == "invalid-agent-proposal-shape" and "dec/evaluation-rule" in message.message
        for message in report.messages
    )


def test_new_decision_proposal_cannot_mark_a_retained_selection_decided(manifest) -> None:
    state = _state_with_decision(manifest)
    decision = state.nodes["dec/evaluation-rule"]
    state = state.model_copy(
        update={
            "nodes": {
                **state.nodes,
                decision.id: decision.model_copy(update={"selected_option": "matched"}),
            }
        }
    )
    proposal = _proposal(
        proposal_id="prop/confirm-evaluation",
        node_id=decision.id,
        changes={"status": "decided"},
    )

    report = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])

    assert report.rejected
    assert any("decide_decision" in message.message for message in report.messages)


def test_legacy_decision_selection_approval_adds_implied_decided_status(manifest) -> None:
    state = _state_with_decision(manifest)
    proposal_patch = _legacy_patch(
        {
            "op": "create_proposals",
            "proposals": [
                {
                    "id": "prop/legacy-selection",
                    "title": "Choose matched evaluation",
                    "card": {"decision_needed": "Choose matched evaluation?"},
                    "ops": [
                        {
                            "op": "update_nodes",
                            "nodes": [
                                {
                                    "id": "dec/evaluation-rule",
                                    "changes": {"selected_option": "matched"},
                                }
                            ],
                        }
                    ],
                    "related_node_ids": ["dec/evaluation-rule"],
                    "base_rev": state.revision,
                    "raised_rev": state.revision,
                }
            ],
        }
    )
    proposal = proposal_patch.ops[0].proposals[0]  # type: ignore[union-attr]
    state = state.model_copy(update={"proposals": {proposal.id: proposal}})
    verbatim_approval = Patch(
        revision=state.revision + 1,
        kind="approval",
        author="human",
        summary="Approved the legacy selection without normalization.",
        ops=[
            *graph_operations_from_proposal(proposal.ops),
            {
                "op": "resolve_proposals",
                "resolutions": [{"id": proposal.id, "status": "approved"}],
            },
        ],
    )
    semantic_ops = normalized_decision_proposal_ops(state, proposal)
    approval = Patch(
        revision=state.revision + 1,
        kind="approval",
        author="human",
        human_action="decision_choice",
        summary="Approved the legacy selection.",
        ops=[
            *semantic_ops,
            {
                "op": "resolve_proposals",
                "resolutions": [{"id": proposal.id, "status": "approved"}],
            },
        ],
    )

    admission = validate_patch(state, verbatim_approval, ["repo-a", "repo-b"])
    replay = validate_patch(
        state,
        verbatim_approval,
        ["repo-a", "repo-b"],
        mode="replay",
    )
    report = validate_patch(state, approval, ["repo-a", "repo-b"])
    unnamed = validate_patch(
        state,
        approval.model_copy(update={"human_action": None}),
        ["repo-a", "repo-b"],
    )

    assert admission.rejected
    assert any(message.code == "unnormalized-decision-approval" for message in admission.messages)
    assert not replay.rejected
    assert semantic_ops[0].nodes[0].changes == {
        "selected_option": "matched",
        "status": "decided",
    }
    assert not report.rejected
    assert unnamed.rejected
    assert any(message.code == "unnamed-decision-action" for message in unnamed.messages)
    updated = apply_valid_patch(state, approval)
    assert updated.nodes["dec/evaluation-rule"].selected_option == "matched"
    assert updated.nodes["dec/evaluation-rule"].status == "decided"


def test_legacy_decided_proposal_without_a_listed_selection_is_refused(manifest) -> None:
    state = _state_with_decision(manifest)
    proposal_patch = _legacy_patch(
        {
            "op": "create_proposals",
            "proposals": [
                {
                    "id": "prop/legacy-missing-selection",
                    "title": "Mark the evaluation decided",
                    "card": {"decision_needed": "Mark it decided?"},
                    "ops": [
                        {
                            "op": "update_nodes",
                            "nodes": [
                                {"id": "dec/evaluation-rule", "changes": {"status": "decided"}}
                            ],
                        }
                    ],
                    "related_node_ids": ["dec/evaluation-rule"],
                    "base_rev": state.revision,
                    "raised_rev": state.revision,
                }
            ],
        }
    )
    proposal = proposal_patch.ops[0].proposals[0]  # type: ignore[union-attr]
    state = state.model_copy(update={"proposals": {proposal.id: proposal}})
    approval = Patch(
        revision=state.revision + 1,
        kind="approval",
        author="human",
        human_action="decision_choice",
        summary="Approved an incoherent legacy proposal.",
        ops=[
            *graph_operations_from_proposal(proposal.ops),
            {
                "op": "resolve_proposals",
                "resolutions": [{"id": proposal.id, "status": "approved"}],
            },
        ],
    )

    report = validate_patch(state, approval, ["repo-a", "repo-b"])

    assert report.rejected
    assert any(message.code == "incoherent-decision-approval" for message in report.messages)


@pytest.mark.parametrize(
    "cause",
    [
        None,
        {"kind": "decision", "ref_id": "dec/evaluation-rule"},
        {"kind": "proposal_resolution", "ref_id": "prop/activate-hypothesis"},
    ],
)
def test_hypothesis_proposal_requires_an_evidence_edge_cause(manifest, cause) -> None:
    state = _state_with_decision(manifest)
    proposal = _proposal(
        proposal_id="prop/activate-hypothesis",
        node_id="hyp/replanning-restores-plasticity",
        changes={"status": "active"},
        cause=cause,
    )

    report = validate_patch(state, _agent_patch(proposal), ["repo-a", "repo-b"])

    assert report.rejected
    assert any(message.code == "invalid-agent-proposal-shape" for message in report.messages)


def test_agent_proposal_rejects_a_third_shape(manifest) -> None:
    state = _state_with_decision(manifest)
    edge_proposal = {
        "op": "create_proposals",
        "proposals": [
            {
                "id": "prop/add-edge",
                "title": "Add an ordinary edge",
                "card": {"decision_needed": "Approve the edge?"},
                "ops": [
                    {
                        "op": "create_edges",
                        "intent": "protected_relation_change",
                        "edges": [
                            {
                                "source": "rq/learning-after-shift",
                                "target": "dec/evaluation-rule",
                                "relation": "has_decision",
                            }
                        ],
                    }
                ],
                "related_node_ids": ["rq/learning-after-shift", "dec/evaluation-rule"],
                "base_rev": 2,
            }
        ],
    }

    report = validate_patch(
        state,
        _agent_patch(edge_proposal),
        ["repo-a", "repo-b"],
    )

    assert report.rejected
