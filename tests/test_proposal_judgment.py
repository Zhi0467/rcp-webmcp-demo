from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from rcp.config import Manifest
from rcp.core.models import GraphState, Patch
from rcp.history import HistoryManager
from rcp.paper import PaperService
from rcp.service import (
    GraphSyncRequest,
    NodeEditConflict,
    ProjectService,
    ProposalDecisionRequest,
    ReviewRequest,
)
from rcp.storage import AppStore
from tests.helpers import seed_patch

QUESTION_ID = "rq/learning-after-shift"
HYPOTHESIS_ID = "hyp/replanning-restores-plasticity"
OTHER_HYPOTHESIS_ID = "hyp/alternate-mechanism"
STATUS_EVIDENCE_ID = "ev/replanning-status"
STATUS_EDGE_ID = "edge/replanning-status"
PROTECTED_EDGE_ID = "rq/learning-after-shift::has_hypothesis::hyp/replanning-restores-plasticity"


INTENT_OPERATIONS: dict[str, dict[str, Any]] = {
    "content_change": {
        "op": "update_nodes",
        "intent": "content_change",
        "nodes": [
            {
                "id": QUESTION_ID,
                "changes": {
                    "question": "Can adaptation remain plastic after repeated task shifts?"
                },
            }
        ],
    },
    "removal": {
        "op": "remove_nodes",
        "intent": "removal",
        "node_ids": [HYPOTHESIS_ID],
    },
    "supersede": {
        "op": "supersede_nodes",
        "intent": "supersede",
        "nodes": [{"id": HYPOTHESIS_ID, "superseded_by": OTHER_HYPOTHESIS_ID}],
    },
    "merge": {
        "op": "merge_nodes",
        "intent": "merge",
        "merges": [{"duplicate": HYPOTHESIS_ID, "canonical": OTHER_HYPOTHESIS_ID}],
    },
    "protected_relation_change": {
        "op": "remove_edges",
        "intent": "protected_relation_change",
        "edge_ids": [PROTECTED_EDGE_ID],
    },
    "status_change": {
        "op": "update_nodes",
        "intent": "status_change",
        "nodes": [
            {
                "id": HYPOTHESIS_ID,
                "changes": {"status": "supported"},
                "cause": {"kind": "evidence_edge", "ref_id": STATUS_EDGE_ID},
            }
        ],
    },
}


def _service(manifest: Manifest, tmp_path) -> ProjectService:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Prepared Proposal judgment targets.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": OTHER_HYPOTHESIS_ID,
                            "type": "hypothesis",
                            "title": "An alternate plasticity mechanism",
                            "statement": "A separate mechanism restores future learning ability.",
                            "rationale": "It provides a distinct canonical target.",
                            "predictions": ["Held-out adaptation recovers."],
                            "status": "proposed",
                        },
                        {
                            "id": STATUS_EVIDENCE_ID,
                            "type": "evidence",
                            "title": "Replanning status evidence",
                            "observation": "Held-out adaptation recovered.",
                            "origin": "analytic",
                        },
                    ],
                },
                {
                    "op": "create_edges",
                    "edges": [
                        {
                            "id": STATUS_EDGE_ID,
                            "source": STATUS_EVIDENCE_ID,
                            "target": HYPOTHESIS_ID,
                            "relation": "supports",
                            "assessment": {
                                "relevance": "direct",
                                "weight": "moderate",
                            },
                        }
                    ],
                },
            ],
        )
    )
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )
    for node_id in (QUESTION_ID, HYPOTHESIS_ID, OTHER_HYPOTHESIS_ID):
        service.review_node(node_id, ReviewRequest(standing="contested"))
    return service


def _append_proposal(service: ProjectService, intent: str) -> str:
    proposal_id = f"prop/{intent.replace('_', '-')}"
    service.history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary=f"Proposed one {intent} change.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": proposal_id,
                            "title": f"Judge {intent.replace('_', ' ')}",
                            "card": {"decision_needed": "Approve or reject this belief change."},
                            "ops": [deepcopy(INTENT_OPERATIONS[intent])],
                        }
                    ],
                }
            ],
        )
    )
    return proposal_id


def _standing(state: GraphState, node_id: str) -> str | None:
    node = state.nodes.get(node_id)
    return None if node is None else node.standing.value


@pytest.mark.parametrize("intent", list(INTENT_OPERATIONS))
def test_proposal_approval_uses_the_declared_intent_standing_target(
    manifest: Manifest, tmp_path, intent: str
) -> None:
    service = _service(manifest, tmp_path)
    if intent == "removal":
        service.review_node(HYPOTHESIS_ID, ReviewRequest(standing="accepted"))
    proposal_id = _append_proposal(service, intent)

    state = service.decide_proposal(
        proposal_id,
        ProposalDecisionRequest(decision="approved"),
    )

    assert state.proposals[proposal_id].status == "approved"
    expected_standing = {
        "content_change": ("accepted", "contested", "contested"),
        "removal": ("contested", None, "contested"),
        "supersede": ("contested", "accepted", "contested"),
        "merge": ("contested", "accepted", "contested"),
        "protected_relation_change": ("contested", "contested", "contested"),
        "status_change": ("contested", "accepted", "contested"),
    }[intent]
    assert (
        _standing(state, QUESTION_ID),
        _standing(state, HYPOTHESIS_ID),
        _standing(state, OTHER_HYPOTHESIS_ID),
    ) == expected_standing

    if intent == "content_change":
        assert state.nodes[QUESTION_ID].question.endswith("repeated task shifts?")
    elif intent == "removal":
        assert HYPOTHESIS_ID not in state.nodes
        assert STATUS_EDGE_ID not in state.edges
    elif intent == "supersede":
        assert state.nodes[HYPOTHESIS_ID].status == "superseded"
        assert f"{HYPOTHESIS_ID}::supersedes::{OTHER_HYPOTHESIS_ID}" in state.edges
    elif intent == "merge":
        assert state.nodes[HYPOTHESIS_ID].status == "superseded"
        assert f"{HYPOTHESIS_ID}::duplicate_of::{OTHER_HYPOTHESIS_ID}" in state.edges
    elif intent == "protected_relation_change":
        assert PROTECTED_EDGE_ID not in state.edges
    else:
        assert state.nodes[HYPOTHESIS_ID].status == "supported"


@pytest.mark.parametrize("intent", list(INTENT_OPERATIONS))
def test_proposal_rejection_resolves_only_and_leaves_graph_exactly_unchanged(
    manifest: Manifest, tmp_path, intent: str
) -> None:
    service = _service(manifest, tmp_path)
    proposal_id = _append_proposal(service, intent)
    before = service.history.state()

    state = service.decide_proposal(
        proposal_id,
        ProposalDecisionRequest(decision="rejected"),
    )

    assert state.proposals[proposal_id].status == "rejected"
    assert state.nodes == before.nodes
    assert state.edges == before.edges


def test_sync_rejects_content_proposal_without_changing_the_question(
    manifest: Manifest, tmp_path
) -> None:
    service = _service(manifest, tmp_path)
    proposal_id = _append_proposal(service, "content_change")
    before = service.history.state()

    state = service.sync_graph(
        GraphSyncRequest(
            base_revision=before.revision,
            proposals=[{"proposal_id": proposal_id, "decision": "rejected"}],
        ),
        active_control_node_ids=set(),
    )

    assert state.proposals[proposal_id].status == "rejected"
    assert state.nodes == before.nodes
    assert state.edges == before.edges


def test_sync_approves_removal_without_reviewing_the_deleted_node(
    manifest: Manifest, tmp_path
) -> None:
    service = _service(manifest, tmp_path)
    service.review_node(HYPOTHESIS_ID, ReviewRequest(standing="accepted"))
    proposal_id = _append_proposal(service, "removal")
    before = service.history.state()

    state = service.sync_graph(
        GraphSyncRequest(
            base_revision=before.revision,
            proposals=[{"proposal_id": proposal_id, "decision": "approved"}],
        ),
        active_control_node_ids=set(),
    )

    assert HYPOTHESIS_ID not in state.nodes
    assert state.proposals[proposal_id].status == "approved"


def test_removal_approval_withdraws_if_a_new_incident_relation_was_not_judged(
    manifest: Manifest,
    tmp_path,
) -> None:
    service = _service(manifest, tmp_path)
    proposal_id = _append_proposal(service, "removal")
    proposal = service.history.state().proposals[proposal_id]
    assert PROTECTED_EDGE_ID in proposal.related_edge_ids
    service.history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Added a later incident relation.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_edges",
                    "edges": [
                        {
                            "id": "edge/later-incident",
                            "source": STATUS_EVIDENCE_ID,
                            "target": HYPOTHESIS_ID,
                            "relation": "weakens",
                            "assessment": {
                                "relevance": "direct",
                                "weight": "moderate",
                            },
                        }
                    ],
                }
            ],
        )
    )

    state = service.decide_proposal(
        proposal_id,
        ProposalDecisionRequest(decision="approved"),
    )

    assert state.proposals[proposal_id].status == "withdrawn"
    assert HYPOTHESIS_ID in state.nodes
    assert "edge/later-incident" in state.edges


def test_sync_stages_multiple_proposal_judgments_against_evolving_state(
    manifest: Manifest,
    tmp_path,
) -> None:
    service = _service(manifest, tmp_path)
    first_id = "prop/first-question-wording"
    second_id = "prop/second-question-wording"
    independent_id = "prop/independent-status"
    first_operation = deepcopy(INTENT_OPERATIONS["content_change"])
    second_operation = deepcopy(INTENT_OPERATIONS["content_change"])
    second_operation["nodes"][0]["changes"]["question"] = (
        "Can a different wording survive repeated shifts?"
    )
    service.history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Raised overlapping and independent judgments.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": proposal_id,
                            "title": title,
                            "card": {"decision_needed": "Approve or reject this belief change."},
                            "ops": [operation],
                        }
                        for proposal_id, title, operation in (
                            (first_id, "Use the first wording", first_operation),
                            (second_id, "Use the second wording", second_operation),
                            (
                                independent_id,
                                "Judge the independent status",
                                deepcopy(INTENT_OPERATIONS["status_change"]),
                            ),
                        )
                    ],
                }
            ],
        )
    )
    before = service.history.state()

    state = service.sync_graph(
        GraphSyncRequest(
            base_revision=before.revision,
            proposals=[
                {"proposal_id": first_id, "decision": "approved"},
                {"proposal_id": second_id, "decision": "approved"},
                {"proposal_id": independent_id, "decision": "rejected"},
            ],
        ),
        active_control_node_ids=set(),
    )

    assert state.nodes[QUESTION_ID].question == (
        "Can adaptation remain plastic after repeated task shifts?"
    )
    assert state.nodes[QUESTION_ID].standing == "accepted"
    assert state.nodes[HYPOTHESIS_ID].standing == "contested"
    assert state.proposals[first_id].status == "approved"
    assert state.proposals[second_id].status == "withdrawn"
    assert state.proposals[second_id].resolution_reason == (
        f"The proposal “{state.proposals[second_id].title}” was stale and was withdrawn without "
        "applying changes."
    )
    assert state.proposals[independent_id].status == "rejected"


def test_sync_withdraws_the_second_proposal_that_creates_the_same_edge(
    manifest: Manifest,
    tmp_path,
) -> None:
    service = _service(manifest, tmp_path)
    first_id = "prop/first-new-protected-edge"
    second_id = "prop/second-new-protected-edge"
    operation = {
        "op": "create_edges",
        "intent": "protected_relation_change",
        "edges": [
            {
                "source": QUESTION_ID,
                "target": OTHER_HYPOTHESIS_ID,
                "relation": "has_hypothesis",
            }
        ],
    }
    service.history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Raised two proposals for the same protected edge.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": proposal_id,
                            "title": title,
                            "card": {"decision_needed": "Approve creating this relation."},
                            "ops": [deepcopy(operation)],
                        }
                        for proposal_id, title in (
                            (first_id, "Create the first relation"),
                            (second_id, "Create the second relation"),
                        )
                    ],
                }
            ],
        )
    )
    before = service.history.state()

    state = service.sync_graph(
        GraphSyncRequest(
            base_revision=before.revision,
            proposals=[
                {"proposal_id": first_id, "decision": "approved"},
                {"proposal_id": second_id, "decision": "approved"},
            ],
        ),
        active_control_node_ids=set(),
    )

    edge_id = f"{QUESTION_ID}::has_hypothesis::{OTHER_HYPOTHESIS_ID}"
    assert edge_id in state.edges
    assert state.proposals[first_id].status == "approved"
    assert state.proposals[second_id].status == "withdrawn"


def test_sync_refuses_approving_a_proposal_and_directly_editing_its_node(
    manifest: Manifest,
    tmp_path,
) -> None:
    service = _service(manifest, tmp_path)
    proposal_id = _append_proposal(service, "content_change")
    before = service.history.state()
    question = before.nodes[QUESTION_ID]

    with pytest.raises(NodeEditConflict, match="approve a Proposal and directly change"):
        service.sync_graph(
            GraphSyncRequest(
                base_revision=before.revision,
                proposals=[{"proposal_id": proposal_id, "decision": "approved"}],
                nodes=[
                    {
                        "node_id": QUESTION_ID,
                        "base_updated_rev": question.updated_rev,
                        "changes": {"question": "A conflicting direct human wording."},
                    }
                ],
            ),
            active_control_node_ids=set(),
        )

    unchanged = service.history.state()
    assert unchanged.revision == before.revision
    assert unchanged.proposals[proposal_id].status == "pending"
    assert unchanged.nodes[QUESTION_ID] == question


@pytest.mark.parametrize(
    ("intent", "relation"),
    [("supersede", "supersedes"), ("merge", "duplicate_of")],
)
def test_lifecycle_proposal_withdraws_if_its_implicit_edge_appears_first(
    manifest: Manifest,
    tmp_path,
    intent: str,
    relation: str,
) -> None:
    service = _service(manifest, tmp_path)
    proposal_id = _append_proposal(service, intent)
    edge_id = f"{HYPOTHESIS_ID}::{relation}::{OTHER_HYPOTHESIS_ID}"
    competing_id = f"prop/competing-{intent}"
    service.history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Raised a competing lifecycle Proposal.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": competing_id,
                            "title": "Apply the competing lifecycle change",
                            "card": {"decision_needed": "Approve this lifecycle change."},
                            "ops": [deepcopy(INTENT_OPERATIONS[intent])],
                        }
                    ],
                }
            ],
        )
    )
    service.decide_proposal(
        competing_id,
        ProposalDecisionRequest(decision="approved"),
    )

    state = service.decide_proposal(
        proposal_id,
        ProposalDecisionRequest(decision="approved"),
    )

    assert state.proposals[proposal_id].status == "withdrawn"
    assert state.nodes[HYPOTHESIS_ID].status == "superseded"
    assert edge_id in state.edges
