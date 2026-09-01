from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rcp.core.materialize import apply_valid_operation, materialize_patches
from rcp.core.models import (
    RELATION_SPEC,
    BaseRelation,
    Decision,
    Edge,
    Evidence,
    GatedCard,
    GraphState,
    Hypothesis,
    Patch,
    Proposal,
    Standing,
    ValidationMessage,
)
from rcp.core.operations import graph_operation_from_proposal, graph_operations_from_proposal
from rcp.core.research_md import render_research_md
from rcp.core.validation import proposal_dependencies, validate_patch
from rcp.core.validation.proposals import proposal_is_stale
from tests.helpers import proposal_operation


def _patch(revision: int, ops: list[dict[str, object]], **changes: object) -> Patch:
    values: dict[str, object] = {
        "revision": revision,
        "kind": "refresh",
        "author": "agent",
        "summary": f"Revision {revision}",
        "run_truth_scope": ["repo"],
        "repositories_read": ["repo"],
        "ops": ops,
    }
    values.update(changes)
    return Patch.model_validate(values)


def _hypothesis(status: str = "proposed") -> Hypothesis:
    return Hypothesis.model_validate(
        {
            "id": "hyp/main",
            "type": "hypothesis",
            "title": "Main hypothesis",
            "statement": "The intervention changes the outcome.",
            "status": status,
            "created_rev": 1,
            "updated_rev": 1,
        }
    )


def _source(excerpt: str) -> dict[str, object]:
    return {
        "machine": "local",
        "truth_repository": "repo",
        "source": "codex",
        "session_id": "session",
        "record_uuid": "record",
        "timestamp": datetime(2026, 7, 30, tzinfo=UTC),
        "excerpt": excerpt,
    }


def _codes(report) -> set[str]:
    return {message.code for message in report.messages}


def _approval_state(ops: list[dict[str, object]]) -> GraphState:
    state = GraphState(
        revision=3,
        project_truth_scope=["repo"],
        nodes={"hyp/main": _hypothesis()},
    )
    proposal = Proposal(
        id="prop/change-belief",
        title="Change the belief",
        card=GatedCard(
            situation_cold="The current state is stale.",
            why_human_now="The belief controls later work.",
            consequences="The graph records a new belief.",
            decision_needed="Approve the belief update.",
        ),
        ops=[proposal_operation(operation) for operation in ops],
        related_node_ids=["hyp/main"],
        base_rev=3,
        raised_rev=3,
    )
    state.proposals[proposal.id] = proposal
    return state


def _approval_patch(ops: list[dict[str, object]]) -> Patch:
    proposal_ops = [proposal_operation(operation) for operation in ops]
    return _patch(
        4,
        [
            *graph_operations_from_proposal(proposal_ops),
            {
                "op": "resolve_proposals",
                "resolutions": [{"id": "prop/change-belief", "status": "approved"}],
            },
        ],
        kind="approval",
        author="human",
        run_truth_scope=[],
        repositories_read=[],
    )


def test_confidence_is_absent_and_rejected() -> None:
    assert "confidence" not in Hypothesis.model_fields
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(
            {
                "id": "hyp/main",
                "type": "hypothesis",
                "title": "Main",
                "statement": "Statement",
                "confidence": "medium",
            }
        )


def test_legacy_evidence_defaults_unknown_on_replay_but_origin_is_explicit_on_admission() -> None:
    raw = {
        "id": "ev/legacy",
        "type": "evidence",
        "title": "Legacy evidence",
        "observation": "The result changed.",
    }
    patch = _patch(1, [{"op": "create_nodes", "nodes": [raw]}])

    admission = validate_patch(GraphState(), patch, ["repo"])
    assert "missing-evidence-origin" in _codes(admission)

    replay = validate_patch(GraphState(), patch, ["repo"], mode="replay")
    assert not replay.rejected
    result = materialize_patches([patch], ["repo"])
    assert result.state.nodes["ev/legacy"].origin == "unknown"

    explicit = raw | {"origin": "external_publication"}
    report = validate_patch(
        GraphState(), _patch(1, [{"op": "create_nodes", "nodes": [explicit]}]), ["repo"]
    )
    assert "missing-evidence-origin" not in _codes(report)


@pytest.mark.parametrize(
    ("scope", "excerpt", "accepted"),
    [
        ("Up to 10B parameters", "The result holds up to 10B parameters.", True),
        ("  UP TO 10B   parameters ", "The result holds up to 10B parameters.", True),
        ("Up to 70B parameters", "The result holds up to 10B parameters.", False),
    ],
)
def test_hypothesis_scope_requires_deterministic_source_grounding(
    scope: str, excerpt: str, accepted: bool
) -> None:
    node = {
        "id": "hyp/scoped",
        "type": "hypothesis",
        "title": "Scoped hypothesis",
        "statement": "Scaling changes adaptation.",
        "scope": scope,
        "source_refs": [_source(excerpt)],
    }
    report = validate_patch(
        GraphState(), _patch(1, [{"op": "create_nodes", "nodes": [node]}]), ["repo"]
    )
    assert ("ungrounded-hypothesis-scope" not in _codes(report)) is accepted


def test_direct_human_scope_edit_does_not_require_source_grounding() -> None:
    state = GraphState(revision=1, project_truth_scope=["repo"], nodes={"hyp/main": _hypothesis()})
    patch = _patch(
        2,
        [
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "hyp/main",
                        "base_updated_rev": 1,
                        "changes": {"scope": "Human-authored boundary conditions."},
                    }
                ],
            }
        ],
        kind="approval",
        author="human",
        run_truth_scope=[],
        repositories_read=[],
    )
    report = validate_patch(state, patch, ["repo"])
    assert not report.rejected
    assert "ungrounded-hypothesis-scope" not in _codes(report)


def test_all_belief_cause_kinds_validate_and_referents_are_checked() -> None:
    evidence_ops = [
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/main",
                    "changes": {"status": "supported"},
                    "cause": {"kind": "evidence_edge", "ref_id": "edge/support"},
                }
            ],
        }
    ]
    evidence_state = _approval_state(evidence_ops)
    evidence_state.nodes["ev/result"] = Evidence(
        id="ev/result",
        type="evidence",
        title="Result",
        observation="The intervention helped.",
        origin="internal_run",
    )
    evidence_state.edges["edge/support"] = Edge(
        id="edge/support",
        source="ev/result",
        target="hyp/main",
        relation="supports",
    )
    assert "invalid-belief-cause" not in _codes(
        validate_patch(evidence_state, _approval_patch(evidence_ops), ["repo"])
    )

    decision_ops = [
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/main",
                    "changes": {"status": "rejected"},
                    "cause": {"kind": "decision", "ref_id": "dec/method"},
                }
            ],
        }
    ]
    decision_state = _approval_state(decision_ops)
    decision_state.nodes["dec/method"] = Decision(
        id="dec/method",
        type="decision",
        title="Method decision",
        question="Which method?",
    )
    assert "invalid-belief-cause" not in _codes(
        validate_patch(decision_state, _approval_patch(decision_ops), ["repo"])
    )

    for cause in (
        {"kind": "proposal_resolution", "ref_id": "prop/change-belief"},
        {"kind": "human_edit"},
    ):
        ops = [
            {
                "op": "update_nodes",
                "intent": "status_change",
                "nodes": [{"id": "hyp/main", "changes": {"status": "active"}, "cause": cause}],
            }
        ]
        assert "invalid-belief-cause" not in _codes(
            validate_patch(_approval_state(ops), _approval_patch(ops), ["repo"])
        )

    invalid = [
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/main",
                    "changes": {"status": "active"},
                    "cause": {"kind": "unknown"},
                }
            ],
        }
    ]
    with pytest.raises(ValidationError, match="cause"):
        _approval_state(invalid)


def test_same_patch_evidence_edge_can_cause_belief_change() -> None:
    proposal_ops: list[dict[str, object]] = [
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/main",
                    "changes": {"status": "supported"},
                    "cause": {"kind": "evidence_edge", "ref_id": "edge/support"},
                }
            ],
        },
    ]
    state = GraphState(revision=1, project_truth_scope=["repo"], nodes={"hyp/main": _hypothesis()})
    patch = _patch(
        2,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "ev/result",
                        "type": "evidence",
                        "title": "Result",
                        "observation": "The intervention helped.",
                        "origin": "internal_run",
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": "edge/support",
                        "source": "ev/result",
                        "target": "hyp/main",
                        "relation": "supports",
                        "assessment": {
                            "relevance": "direct",
                            "weight": "moderate",
                            "scope": "The intervention result.",
                        },
                    }
                ],
            },
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/same-patch",
                        "title": "Apply same-patch evidence",
                        "card": {
                            "situation_cold": "New evidence is available.",
                            "why_human_now": "It changes the belief.",
                            "consequences": "The hypothesis becomes supported.",
                            "decision_needed": "Approve this update.",
                        },
                        "ops": proposal_ops,
                        "related_node_ids": ["hyp/main"],
                        "base_rev": 1,
                    }
                ],
            },
        ],
    )
    report = validate_patch(state, patch, ["repo"])
    assert not report.rejected


def test_existing_decision_belief_cause_is_replay_only() -> None:
    proposal_ops: list[dict[str, object]] = [
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/main",
                    "changes": {"status": "active"},
                    "cause": {"kind": "decision", "ref_id": "dec/method"},
                }
            ],
        },
    ]
    decision = Decision(
        id="dec/method",
        type="decision",
        title="Method",
        question="Which method?",
    )
    state = GraphState(
        revision=1,
        project_truth_scope=["repo"],
        nodes={"hyp/main": _hypothesis(), decision.id: decision},
    )
    patch = _patch(
        2,
        [
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/same-patch-decision",
                        "title": "Apply a method decision",
                        "card": {
                            "situation_cold": "A method must be selected.",
                            "why_human_now": "It changes the belief.",
                            "consequences": "The hypothesis becomes active.",
                            "decision_needed": "Approve this update.",
                        },
                        "ops": proposal_ops,
                        "related_node_ids": ["hyp/main"],
                        "base_rev": 1,
                    }
                ],
            }
        ],
    )
    admission = validate_patch(state, patch, ["repo"])
    replay = validate_patch(state, patch, ["repo"], mode="replay")

    assert admission.rejected
    assert any(message.code == "invalid-agent-proposal-shape" for message in admission.messages)
    assert not replay.rejected


@pytest.mark.parametrize(
    ("operation", "expected_code"),
    [
        (
            {"op": "update_nodes", "nodes": [{"id": "hyp/main", "changes": {"status": "active"}}]},
            "missing-belief-cause",
        ),
        (
            {"op": "supersede_nodes", "nodes": [{"id": "hyp/main"}]},
            "graph-action-refused",
        ),
        (
            {
                "op": "merge_nodes",
                "merges": [{"duplicate": "hyp/main", "canonical": "hyp/other"}],
            },
            "graph-action-refused",
        ),
    ],
)
def test_every_belief_changing_operation_requires_human_authority(
    operation: dict[str, object],
    expected_code: str,
) -> None:
    state = GraphState(
        project_truth_scope=["repo"],
        nodes={
            "hyp/main": _hypothesis(),
            "hyp/other": _hypothesis().model_copy(update={"id": "hyp/other"}),
        },
    )
    report = validate_patch(state, _patch(1, [operation]), ["repo"])
    assert expected_code in _codes(report)
    assert expected_code not in _codes(
        validate_patch(state, _patch(1, [operation]), ["repo"], mode="replay")
    )


def test_relation_spec_covers_every_relation_flags_mismatches_and_serializes_layer() -> None:
    assert set(RELATION_SPEC) == set(BaseRelation.__args__)
    assert {spec.layer for spec in RELATION_SPEC.values()} == {
        "epistemic",
        "action",
        "seam",
        "meta",
    }
    assert RELATION_SPEC["supersedes"].same_type
    assert RELATION_SPEC["duplicate_of"].same_type
    assert RELATION_SPEC["informs"].source_types == frozenset({"evidence"})
    assert RELATION_SPEC["informs"].target_types == frozenset({"decision"})
    assert RELATION_SPEC["addresses"].source_types == frozenset({"evidence"})
    assert RELATION_SPEC["addresses"].target_types == frozenset({"blocker"})

    patch = _patch(
        1,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/question",
                        "type": "research_question",
                        "title": "Question",
                        "question": "What changes?",
                    },
                    {
                        "id": "hyp/answer",
                        "type": "hypothesis",
                        "title": "Answer",
                        "statement": "Something changes.",
                    },
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": "edge/wrong",
                        "source": "rq/question",
                        "target": "hyp/answer",
                        "relation": "supports",
                    }
                ],
            },
        ],
    )
    message = next(
        item
        for item in validate_patch(GraphState(), patch, ["repo"]).messages
        if item.code == "relation-type-mismatch"
    )
    assert message.related_node_ids == ["rq/question", "hyp/answer"]
    assert message.related_edge_ids == ["edge/wrong"]
    assert "research_question" in message.message

    edge = Edge(id="edge/good", source="ev/a", target="hyp/a", relation="supports")
    dumped = GraphState(edges={edge.id: edge}).model_dump(mode="json")
    assert dumped["edges"][edge.id]["layer"] == "epistemic"
    round_tripped = GraphState.model_validate(dumped)
    assert round_tripped.edges[edge.id].layer == "epistemic"
    overridden = Edge.model_validate(edge.model_dump() | {"layer": "action"})
    assert overridden.layer == "epistemic"


def test_authoring_rules_are_skipped_on_replay() -> None:
    patch = _patch(
        1,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "hyp/scoped",
                        "type": "hypothesis",
                        "title": "Scoped",
                        "statement": "A claim.",
                        "scope": "Not present in any source",
                    },
                    {
                        "id": "ev/legacy",
                        "type": "evidence",
                        "title": "Legacy",
                        "observation": "An observation.",
                    },
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "hyp/scoped",
                        "target": "ev/legacy",
                        "relation": "supports",
                    }
                ],
            },
        ],
    )
    admission = validate_patch(GraphState(), patch, ["repo"])
    assert {
        "missing-evidence-origin",
        "ungrounded-hypothesis-scope",
        "relation-type-mismatch",
    } <= _codes(admission)
    replay = validate_patch(GraphState(), patch, ["repo"], mode="replay")
    assert not replay.rejected
    assert not replay.flags


def test_rejected_admission_is_skipped_and_later_revision_applies() -> None:
    rejection = ValidationMessage(
        level="reject",
        code="authoring-rejected",
        message="Stored admission rejection.",
        patch_revision=1,
    )
    rejected = _patch(
        1,
        [
            {
                "op": "update_nodes",
                "nodes": [{"id": "rq/missing", "changes": {"title": "Missing target"}}],
            }
        ],
        admission="rejected",
        admission_messages=[rejection],
    )
    accepted = _patch(
        2,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/accepted",
                        "type": "research_question",
                        "title": "Accepted",
                        "question": "Did it apply?",
                    }
                ],
            }
        ],
    )
    result = materialize_patches([rejected, accepted], ["repo"])
    assert result.state.revision == 2
    assert result.state.replay_status == "complete"
    assert "rq/accepted" in result.state.nodes
    assert result.state.validation_messages == [rejection]


def test_structural_replay_failure_halts_without_cascade() -> None:
    first = _patch(
        1,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/first",
                        "type": "research_question",
                        "title": "First",
                        "question": "First?",
                    }
                ],
            }
        ],
    )
    corrupt = _patch(
        2,
        [
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/missing",
                        "target": "rq/first",
                        "relation": "has_subquestion",
                    }
                ],
            }
        ],
    )
    later = _patch(
        3,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/later",
                        "type": "research_question",
                        "title": "Later",
                        "question": "Should never apply?",
                    }
                ],
            }
        ],
    )
    result = materialize_patches([first, corrupt, later], ["repo"])
    assert result.state.revision == 1
    assert result.state.replay_status == "degraded"
    assert result.state.replay_failure is not None
    assert result.state.replay_failure.revision == 2
    assert result.state.replay_failure.code == "unknown-edge-source"
    assert "rq/first" in result.state.nodes
    assert "rq/later" not in result.state.nodes
    assert 3 not in result.reports


def test_belief_history_is_derived_from_accepted_patch_log() -> None:
    create = _patch(
        1,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "hyp/main",
                        "type": "hypothesis",
                        "title": "Main",
                        "statement": "The intervention helps.",
                    }
                ],
            }
        ],
    )
    proposal_update = proposal_operation(
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/main",
                    "changes": {"status": "active"},
                    "cause": {"kind": "proposal_resolution", "ref_id": "prop/activate"},
                }
            ],
        }
    )
    propose = _patch(
        2,
        [
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/activate",
                        "title": "Activate the hypothesis",
                        "card": {},
                        "ops": [proposal_update],
                        "related_node_ids": ["hyp/main"],
                        "base_rev": 1,
                    }
                ],
            }
        ],
    )
    approve = _patch(
        3,
        [
            graph_operation_from_proposal(proposal_update),
            {
                "op": "resolve_proposals",
                "resolutions": [{"id": "prop/activate", "status": "approved"}],
            },
        ],
        kind="approval",
        author="human",
        run_truth_scope=[],
        repositories_read=[],
    )
    result = materialize_patches([create, propose, approve], ["repo"])
    assert result.state.replay_status == "complete"
    assert result.state.nodes["hyp/main"].status == "active"
    assert [item.model_dump() for item in result.state.belief_transitions] == [
        {
            "hypothesis_id": "hyp/main",
            "from_status": "proposed",
            "to_status": "active",
            "revision": 3,
            "cause": {"kind": "proposal_resolution", "ref_id": "prop/activate"},
        }
    ]


def test_experiment_attempts_remain_nested_records() -> None:
    patch = _patch(
        1,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "exp/run",
                        "type": "experiment",
                        "title": "Run",
                        "objective": "Measure the outcome.",
                        "attempts": [
                            {
                                "id": "attempt-1",
                                "sequence": 1,
                                "purpose": "First measurement",
                            }
                        ],
                    }
                ],
            }
        ],
    )
    result = materialize_patches([patch], ["repo"])
    experiment = result.state.nodes["exp/run"]
    assert experiment.attempts[0].id == "attempt-1"
    assert set(result.state.nodes) == {"exp/run"}


def test_remove_nodes_keeps_a_dependent_proposal_pending_and_makes_it_stale() -> None:
    state = _approval_state(
        [
            {
                "op": "update_nodes",
                "intent": "content_change",
                "nodes": [{"id": "hyp/main", "changes": {"status": "active"}}],
            }
        ]
    )
    proposal_op = proposal_operation(
        {"op": "remove_nodes", "intent": "removal", "node_ids": ["hyp/main"]}
    )
    operation = graph_operation_from_proposal(proposal_op)
    patch = _patch(
        4,
        [operation],
        kind="approval",
        author="human",
        run_truth_scope=[],
        repositories_read=[],
    )

    assert proposal_dependencies(state, [proposal_op]) == (["hyp/main"], [], [])
    assert not validate_patch(state, patch, ["repo"]).rejected

    updated = apply_valid_operation(state, patch, operation)
    proposal = updated.proposals["prop/change-belief"]
    assert proposal.status == "pending"
    assert proposal_is_stale(updated, proposal)


def test_research_md_renders_hypothesis_scope() -> None:
    hypothesis = _hypothesis().model_copy(
        update={"standing": Standing.ACCEPTED, "scope": "On datasets A and B."}
    )
    rendered = render_research_md(GraphState(revision=1, nodes={hypothesis.id: hypothesis}))
    assert "Scope: On datasets A and B." in rendered


def _layer_of(source: dict[str, object], target: dict[str, object], relation: str) -> str:
    patch = _patch(
        1,
        [
            {"op": "create_nodes", "nodes": [source, target]},
            {
                "op": "create_edges",
                "edges": [{"source": source["id"], "target": target["id"], "relation": relation}],
            },
        ],
    )
    state = materialize_patches([patch], ["repo"]).state
    return state.edges[f"{source['id']}::{relation}::{target['id']}"].layer


_RQ = {"id": "rq/q", "type": "research_question", "title": "Q", "question": "Why?"}
_DEC = {"id": "dec/d", "type": "decision", "title": "D", "question": "Which target?"}
_EXP = {"id": "exp/e", "type": "experiment", "title": "E", "objective": "Measure it."}
_EV = {"id": "ev/e", "type": "evidence", "title": "E", "observation": "It ran."}
_BLK = {"id": "blk/b", "type": "blocker", "title": "B", "description": "State is missing."}


def test_edge_layer_is_derived_from_the_endpoints_not_the_relation_name() -> None:
    # has_decision and blocked_by both declare "action" in RELATION_SPEC, but a
    # research_question is epistemic, so from that source they cross and are seams.
    assert _layer_of(_RQ, _DEC, "has_decision") == "seam"
    assert _layer_of(_RQ, _BLK, "blocked_by") == "seam"

    # The same relation stays inside the action layer from an action-layer source.
    assert _layer_of(_EXP, _DEC, "governed_by") == "action"
    assert _layer_of(_EXP, _BLK, "blocked_by") == "action"

    # Action evidence crosses from the epistemic layer into the action layer.
    assert _layer_of(_EV, _DEC, "informs") == "seam"
    assert _layer_of(_EV, _BLK, "addresses") == "seam"


def test_declared_seam_relations_still_resolve_to_seam() -> None:
    assert _layer_of(_EXP, _hypothesis().model_dump(mode="json"), "tests") == "seam"


def test_meta_relations_keep_their_layer_regardless_of_endpoints() -> None:
    # supersedes joins two same-type nodes, so endpoint derivation would call it
    # epistemic. Meta describes what the edge says about the graph, not where its
    # endpoints sit, so it is preserved.
    other = dict(_RQ) | {"id": "rq/older"}
    assert _layer_of(other, _RQ, "supersedes") == "meta"


def _dependency_state() -> GraphState:
    """A hypothesis and a decision joined by an edge, so removals pull the edge in."""
    return GraphState(
        revision=3,
        nodes={
            "hyp/main": _hypothesis(),
            "dec/scope": Decision.model_validate(
                {
                    "id": "dec/scope",
                    "type": "decision",
                    "title": "Scope decision",
                    "question": "Which task families are in scope?",
                }
            ),
        },
        edges={
            "edge/informs": Edge(
                id="edge/informs",
                source="dec/scope",
                target="hyp/main",
                relation="informs",
                layer="seam",
            )
        },
    )


def test_proposal_dependencies_walks_every_operation_not_just_the_first() -> None:
    # Both typed operations contribute dependencies. Stopping after the first
    # would silently drop the removal and its incident edge.
    state = _dependency_state()
    ops = [
        proposal_operation(
            {
                "op": "update_nodes",
                "intent": "content_change",
                "nodes": [{"id": "dec/scope", "changes": {"title": "Updated scope"}}],
            }
        ),
        proposal_operation({"op": "remove_nodes", "intent": "removal", "node_ids": ["hyp/main"]}),
    ]

    assert proposal_dependencies(state, ops) == (
        ["dec/scope", "hyp/main"],
        ["edge/informs"],
        [],
    )


def test_proposal_dependencies_records_the_decision_a_status_change_cites() -> None:
    # A status change caused by a Decision depends on that Decision node; one
    # caused by an evidence edge depends on the edge instead.
    state = _dependency_state()
    by_decision = proposal_operation(
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/main",
                    "changes": {"status": "active"},
                    "cause": {"kind": "decision", "ref_id": "dec/scope"},
                }
            ],
        }
    )
    by_edge = proposal_operation(
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/main",
                    "changes": {"status": "active"},
                    "cause": {"kind": "evidence_edge", "ref_id": "edge/informs"},
                }
            ],
        }
    )

    assert proposal_dependencies(state, [by_decision]) == (["dec/scope", "hyp/main"], [], [])
    assert proposal_dependencies(state, [by_edge]) == (["hyp/main"], ["edge/informs"], [])


def test_proposal_dependencies_rejects_operations_outside_the_typed_contract() -> None:
    state = _dependency_state()
    with pytest.raises(ValidationError):
        proposal_operation({"op": "invent_nodes", "nodes": []})
    with pytest.raises(ValidationError):
        proposal_operation({"op": "upsert_glossary", "terms": []})

    removal = proposal_operation(
        {
            "op": "remove_edges",
            "intent": "protected_relation_change",
            "edge_ids": ["edge/informs"],
        }
    )
    assert proposal_dependencies(state, [removal]) == (
        ["dec/scope", "hyp/main"],
        ["edge/informs"],
        [],
    )
