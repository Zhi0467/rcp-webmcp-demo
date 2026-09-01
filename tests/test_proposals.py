from __future__ import annotations

import pytest

from rcp.config import load_manifest, write_project_scope
from rcp.core.materialize import prepare_patch_bookkeeping
from rcp.core.models import Patch
from rcp.core.operations import adapt_persisted_patch_document, graph_operations_from_proposal
from rcp.core.validation import validate_patch
from rcp.history import HistoryManager, PatchRejected
from rcp.paper import PaperService
from rcp.service import (
    ProjectService,
    ProposalDecisionRequest,
    ReviewRequest,
)
from rcp.storage import AppStore
from tests.helpers import seed_patch


def ontology_payload() -> dict[str, object]:
    return {
        "types": [
            {
                "name": "mechanism_hypothesis",
                "definition": "A hypothesis about the mechanism responsible for an effect.",
                "base_type": "hypothesis",
                "layer": "epistemic",
                "deprecated": False,
            }
        ],
        "fields": [],
        "relations": [],
    }


def extension_ontology_payload(definition: str) -> dict[str, object]:
    return {
        "types": [],
        "fields": [
            {
                "owner_type": "research_question",
                "name": "review_note",
                "definition": definition,
                "kind": "text",
                "agent_writable": False,
            }
        ],
        "relations": [],
    }


def proposal_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Proposed activating the replanning hypothesis.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/activate-replanning-hypothesis",
                        "title": "Treat replanning as the active hypothesis",
                        "card": {
                            "situation_cold": "The project has a proposed causal explanation but no active one.",
                            "why_human_now": "Activating it changes what experiments are interpreted against.",
                            "consequences": "Future evidence will be organized around this prediction.",
                            "decision_needed": "Decide whether the hypothesis is ready to become active.",
                        },
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "status_change",
                                "nodes": [
                                    {
                                        "id": "hyp/replanning-restores-plasticity",
                                        "changes": {"status": "active"},
                                        "cause": {
                                            "kind": "evidence_edge",
                                            "ref_id": "edge/replanning-activation",
                                        },
                                    }
                                ],
                            }
                        ],
                        "related_node_ids": ["hyp/replanning-restores-plasticity"],
                        "base_rev": 1,
                    }
                ],
            },
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "ev/replanning-activation",
                        "type": "evidence",
                        "title": "Replanning activation evidence",
                        "observation": "The observed behavior warrants testing this as active.",
                        "origin": "analytic",
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": "edge/replanning-activation",
                        "source": "ev/replanning-activation",
                        "target": "hyp/replanning-restores-plasticity",
                        "relation": "supports",
                        "assessment": {
                            "relevance": "direct",
                            "weight": "moderate",
                            "scope": "Analytic activation evidence.",
                        },
                    }
                ],
            },
        ],
    )


def content_proposal_patch() -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Proposed clarifying the research question.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/clarify-learning-question",
                        "title": "Clarify the learning question",
                        "card": {
                            "situation_cold": "The current wording hides the repeated-shift case.",
                            "why_human_now": "The question itself is human-authoritative.",
                            "consequences": "Future work will answer the narrower wording.",
                            "decision_needed": "Approve or reject the clarified wording.",
                        },
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "content_change",
                                "nodes": [
                                    {
                                        "id": "rq/learning-after-shift",
                                        "changes": {
                                            "question": "Can adaptation remain plastic after repeated shifts?"
                                        },
                                    }
                                ],
                            }
                        ],
                        "base_rev": 1,
                    }
                ],
            }
        ],
    )


def ontology_proposal_patch() -> Patch:
    return Patch(
        schema_generation=1,
        kind="refresh",
        author="agent",
        summary="Proposed a project-specific ontology type.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/add-mechanism-hypothesis",
                        "title": "Add mechanism hypothesis type",
                        "card": {
                            "situation_cold": "Mechanism hypotheses need a distinct vocabulary.",
                            "why_human_now": "Ontology changes govern future graph authoring.",
                            "consequences": "New hypotheses may use this custom semantic type.",
                            "decision_needed": "Decide whether to activate the custom type.",
                        },
                        "ops": [
                            {
                                "op": "set_ontology",
                                "intent": "legacy_ontology_change",
                                "ontology": ontology_payload(),
                            }
                        ],
                        "related_node_ids": [],
                        "related_config_keys": ["ontology"],
                        "base_rev": 1,
                    }
                ],
            }
        ],
    )


def evidence_proposal_patch(*, remove_cause: bool = False) -> Patch:
    ops: list[dict[str, object]] = [
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "ev/replanning-result",
                    "type": "evidence",
                    "title": "Replanning result",
                    "observation": "The held-out learning curve recovered.",
                    "origin": "internal_run",
                }
            ],
        },
        {
            "op": "create_edges",
            "edges": [
                {
                    "id": "edge/replanning-support",
                    "source": "ev/replanning-result",
                    "target": "hyp/replanning-restores-plasticity",
                    "relation": "supports",
                    "assessment": {
                        "relevance": "direct",
                        "weight": "moderate",
                        "scope": "Held-out learning curve recovery.",
                    },
                }
            ],
        },
        {
            "op": "create_proposals",
            "proposals": [
                {
                    "id": "prop/support-replanning-hypothesis",
                    "title": "Mark the replanning hypothesis supported",
                    "card": {
                        "situation_cold": "The held-out result supports the hypothesis.",
                        "why_human_now": "Only a human may move the belief status.",
                        "consequences": "The hypothesis will be marked supported.",
                        "decision_needed": "Approve or reject the belief transition.",
                    },
                    "ops": [
                        {
                            "op": "update_nodes",
                            "intent": "status_change",
                            "nodes": [
                                {
                                    "id": "hyp/replanning-restores-plasticity",
                                    "changes": {"status": "supported"},
                                    "cause": {
                                        "kind": "evidence_edge",
                                        "ref_id": "edge/replanning-support",
                                    },
                                }
                            ],
                        }
                    ],
                    "related_node_ids": ["hyp/replanning-restores-plasticity"],
                    "base_rev": 1,
                }
            ],
        },
    ]
    if remove_cause:
        ops.append({"op": "remove_edges", "edge_ids": ["edge/replanning-support"]})
    return Patch(
        kind="refresh",
        author="agent",
        summary="Recorded evidence and proposed its belief transition.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=ops,
    )


def test_approval_replays_exact_ops_and_accepts_node(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(proposal_patch())
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )

    state = service.decide_proposal(
        "prop/activate-replanning-hypothesis",
        ProposalDecisionRequest(decision="approved"),
    )

    node = state.nodes["hyp/replanning-restores-plasticity"]
    assert node.status == "active"
    assert node.standing == "accepted"
    assert state.proposals["prop/activate-replanning-hypothesis"].status == "approved"


def test_content_intent_survives_materialization_and_applies_without_evidence(
    manifest, tmp_path
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(content_proposal_patch())
    stored = history.state().proposals["prop/clarify-learning-question"]
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )

    assert stored.ops[0].intent == "content_change"
    assert stored.ops[0].nodes[0].cause is None

    state = service.decide_proposal(
        stored.id,
        ProposalDecisionRequest(decision="approved"),
    )

    assert (
        state.nodes["rq/learning-after-shift"].question
        == "Can adaptation remain plastic after repeated shifts?"
    )
    assert state.proposals[stored.id].status == "approved"


def test_approved_removal_intent_can_remove_an_accepted_hypothesis(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    node_id = "hyp/replanning-restores-plasticity"
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Accepted the hypothesis.",
            ops=[{"op": "set_standing", "node_id": node_id, "standing": "accepted"}],
        )
    )
    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Proposed removing the accepted hypothesis.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": "prop/remove-replanning-hypothesis",
                            "title": "Remove the replanning hypothesis",
                            "card": {"decision_needed": "Approve or reject this removal."},
                            "ops": [
                                {
                                    "op": "remove_nodes",
                                    "intent": "removal",
                                    "node_ids": [node_id],
                                }
                            ],
                            "base_rev": history.state().revision,
                        }
                    ],
                }
            ],
        )
    )
    proposal = history.state().proposals["prop/remove-replanning-hypothesis"]
    approval = Patch(
        kind="approval",
        author="human",
        summary="Approved removal of the accepted hypothesis.",
        ops=[
            *graph_operations_from_proposal(proposal.ops),
            {
                "op": "resolve_proposals",
                "resolutions": [{"id": proposal.id, "status": "approved"}],
            },
        ],
    )

    history.append(approval)
    state = history.state()

    assert node_id not in state.nodes
    assert state.proposals[proposal.id].status == "approved"


def test_evidence_grounded_proposal_can_be_approved(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(evidence_proposal_patch())
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )

    state = service.decide_proposal(
        "prop/support-replanning-hypothesis",
        ProposalDecisionRequest(decision="approved"),
    )

    node = state.nodes["hyp/replanning-restores-plasticity"]
    assert node.status == "supported"
    assert node.standing == "accepted"
    assert state.proposals["prop/support-replanning-hypothesis"].status == "approved"


def test_same_patch_removal_of_a_belief_cause_rejects_the_proposal(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())

    with pytest.raises(PatchRejected):
        history.append(evidence_proposal_patch(remove_cause=True))

    assert history.state().proposals == {}


def test_later_removal_of_a_belief_cause_withdraws_the_stale_proposal(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(evidence_proposal_patch())
    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Removed the invalidated evidence relation.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[{"op": "remove_edges", "edge_ids": ["edge/replanning-support"]}],
        )
    )
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )

    state = service.decide_proposal(
        "prop/support-replanning-hypothesis",
        ProposalDecisionRequest(decision="approved"),
    )

    assert state.proposals["prop/support-replanning-hypothesis"].status == "withdrawn"
    assert state.nodes["hyp/replanning-restores-plasticity"].status == "proposed"


def test_agent_withdraws_a_pending_proposal_with_provenance(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(proposal_patch().model_copy(update={"source_operation_id": "task-create"}))
    withdrawal = Patch(
        kind="refresh",
        author="agent",
        summary="Withdrew an obsolete proposal.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        source_operation_id="task-withdraw",
        ops=[
            {
                "op": "withdraw_proposals",
                "proposals": [
                    {
                        "id": "prop/activate-replanning-hypothesis",
                        "reason": "A revised proposal replaces this one.",
                    }
                ],
            }
        ],
    )

    history.append(withdrawal)

    proposal = history.state().proposals["prop/activate-replanning-hypothesis"]
    assert proposal.status == "withdrawn"
    assert proposal.created_by == "agent"
    assert proposal.created_by_operation_id == "task-create"
    assert proposal.resolved_by == "agent"
    assert proposal.resolved_by_operation_id == "task-withdraw"
    assert proposal.resolution_reason == "A revised proposal replaces this one."


def test_human_created_proposal_records_operation_provenance(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    state = history.state()
    patch = proposal_patch().model_copy(
        update={
            "kind": "approval",
            "author": "human",
            "run_truth_scope": [],
            "repositories_read": [],
            "source_operation_id": "human-create-proposal",
        }
    )

    prepared = prepare_patch_bookkeeping(state, patch)

    proposal = prepared.ops[0].proposals[0]
    assert proposal.created_by == "human"
    assert proposal.created_by_operation_id == "human-create-proposal"


def test_agent_cannot_resolve_a_pending_proposal_with_human_decision_status(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(proposal_patch())
    resolution = Patch(
        kind="refresh",
        author="agent",
        summary="Tried to approve a pending human judgment.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "resolve_proposals",
                "resolutions": [
                    {"id": "prop/activate-replanning-hypothesis", "status": "approved"}
                ],
            }
        ],
    )

    with pytest.raises(PatchRejected) as caught:
        history.append(resolution)

    assert any(
        message.code == "graph-action-refused" and "resolve_proposal" in message.message
        for message in caught.value.report.messages
    )
    assert history.state().proposals["prop/activate-replanning-hypothesis"].status == "pending"


def test_agent_ontology_proposal_is_rejected(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())

    with pytest.raises(PatchRejected):
        history.append(ontology_proposal_patch())

    assert history.state().ontology.types == []
    assert history.state().proposals == {}


def test_historical_ontology_proposal_remains_replayable(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    raw = ontology_proposal_patch().model_dump(mode="python", exclude_unset=True)
    raw.pop("schema_generation", None)
    raw["ops"][0]["proposals"][0]["ops"][0].pop("intent", None)
    raw["revision"] = 2
    patch = Patch.model_validate(adapt_persisted_patch_document(raw))

    report = validate_patch(
        history.state(),
        patch,
        manifest.project.truth_scope,
        mode="replay",
    )

    assert not report.rejected


def test_historical_non_evidence_belief_cause_remains_replayable(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    raw = proposal_patch().model_dump(mode="python", exclude_unset=True)
    raw["revision"] = 2
    update = raw["ops"][0]["proposals"][0]["ops"][0]["nodes"][0]
    update["cause"] = {
        "kind": "proposal_resolution",
        "ref_id": "prop/activate-replanning-hypothesis",
    }
    patch = Patch.model_validate(raw)

    report = validate_patch(
        history.state(),
        patch,
        manifest.project.truth_scope,
        mode="replay",
    )

    assert not report.rejected


def test_replay_does_not_recheck_rcp_owned_proposal_base_revision(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    raw = proposal_patch().model_dump(mode="python", exclude_unset=True)
    raw["revision"] = 2
    raw["ops"][0]["proposals"][0]["base_rev"] = 999
    patch = Patch.model_validate(raw)

    report = validate_patch(
        history.state(),
        patch,
        manifest.project.truth_scope,
        mode="replay",
    )

    assert not report.rejected


def test_stale_proposal_is_withdrawn_without_replay(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(proposal_patch())
    node = history.state().nodes["hyp/replanning-restores-plasticity"]
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Clarified the rationale in the human editor.",
            ops=[
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "hyp/replanning-restores-plasticity",
                            "base_updated_rev": node.updated_rev,
                            "changes": {"rationale": "The mechanism is now framed more narrowly."},
                        }
                    ],
                }
            ],
        )
    )
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )

    state = service.decide_proposal(
        "prop/activate-replanning-hypothesis",
        ProposalDecisionRequest(decision="approved"),
    )

    assert state.proposals["prop/activate-replanning-hypothesis"].status == "withdrawn"
    assert state.nodes["hyp/replanning-restores-plasticity"].status == "proposed"


def test_extension_field_proposal_withdraws_after_ontology_moves(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Defined the project review note.",
            ops=[
                {
                    "op": "set_ontology",
                    "ontology": extension_ontology_payload("A human review note."),
                }
            ],
        )
    )
    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Proposed recording the review note.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": "prop/add-review-note",
                            "title": "Add the review note",
                            "card": {"decision_needed": "Approve this ontology-backed field."},
                            "ops": [
                                {
                                    "op": "update_nodes",
                                    "intent": "content_change",
                                    "nodes": [
                                        {
                                            "id": "rq/learning-after-shift",
                                            "changes": {
                                                "extension_fields": {
                                                    "review_note": "Human review requested."
                                                }
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        )
    )
    proposed = history.state().proposals["prop/add-review-note"]
    assert proposed.related_config_keys == ["ontology"]
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Clarified the ontology field definition.",
            ops=[
                {
                    "op": "set_ontology",
                    "ontology": extension_ontology_payload("A reviewed project note."),
                }
            ],
        )
    )
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )

    state = service.decide_proposal(
        "prop/add-review-note", ProposalDecisionRequest(decision="approved")
    )

    assert state.proposals["prop/add-review-note"].status == "withdrawn"
    assert state.nodes["rq/learning-after-shift"].extension_fields == {}


@pytest.mark.parametrize("legacy_bookkeeping", ["missing", "provider-supplied"])
def test_rcp_overwrites_legacy_proposal_bookkeeping_before_admission(
    manifest, legacy_bookkeeping: str
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    raw = proposal_patch().model_dump(mode="python", exclude_unset=True)
    raw["source_operation_id"] = "agent-create-proposal"
    proposal = raw["ops"][0]["proposals"][0]
    bookkeeping = {
        "related_node_ids": ["rq/learning-after-shift"],
        "related_edge_ids": ["provider/supplied"],
        "related_config_keys": ["ontology"],
        "base_rev": 999,
        "status": "approved",
        "created_by": "human",
        "created_by_operation_id": "provider-create",
        "raised_rev": 999,
        "resolved_rev": 999,
        "resolved_by": "human",
        "resolved_by_operation_id": "provider-resolve",
        "resolution_reason": "Provider-owned resolution metadata must not survive.",
        "rejection_reason": "Provider-owned bookkeeping must not survive.",
    }
    if legacy_bookkeeping == "missing":
        for field in bookkeeping:
            proposal.pop(field, None)
    else:
        proposal.update(bookkeeping)
    patch = Patch.model_validate(raw)

    canonical, result = history.append(patch)
    prepared = canonical.ops[0].proposals[0]
    admitted = result.state.proposals["prop/activate-replanning-hypothesis"]

    assert canonical.admission == "accepted"
    assert prepared.base_rev == 1
    assert prepared.related_node_ids == ["hyp/replanning-restores-plasticity"]
    assert prepared.related_edge_ids == ["edge/replanning-activation"]
    assert prepared.related_config_keys == []
    assert prepared.status == "pending"
    assert prepared.created_by == "agent"
    assert prepared.created_by_operation_id == "agent-create-proposal"
    assert prepared.raised_rev == 0
    assert prepared.resolved_rev is None
    assert prepared.resolved_by is None
    assert prepared.resolved_by_operation_id is None
    assert prepared.resolution_reason is None
    assert prepared.rejection_reason is None
    assert admitted.base_rev == 1
    assert admitted.related_node_ids == ["hyp/replanning-restores-plasticity"]
    assert admitted.related_edge_ids == ["edge/replanning-activation"]
    assert admitted.related_config_keys == []
    assert admitted.status == "pending"
    assert admitted.created_by == "agent"
    assert admitted.created_by_operation_id == "agent-create-proposal"
    assert admitted.raised_rev == 2
    assert admitted.resolved_rev is None
    assert admitted.resolved_by is None
    assert admitted.resolved_by_operation_id is None
    assert admitted.resolution_reason is None
    assert admitted.rejection_reason is None


def test_proposal_bookkeeping_derives_removed_edge_dependencies_from_state(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    state = history.state()
    edge_id = "rq/learning-after-shift::has_hypothesis::hyp/replanning-restores-plasticity"
    raw = proposal_patch().model_dump(mode="python", exclude_unset=True)
    proposal = raw["ops"][0]["proposals"][0]
    proposal["ops"] = [
        {
            "op": "remove_edges",
            "intent": "protected_relation_change",
            "edge_ids": [edge_id],
        }
    ]
    proposal["related_node_ids"] = ["provider/supplied"]
    patch = Patch.model_validate(raw)

    prepared = prepare_patch_bookkeeping(state, patch)

    prepared_proposal = prepared.ops[0].proposals[0]
    assert prepared_proposal.related_node_ids == [
        "hyp/replanning-restores-plasticity",
        "rq/learning-after-shift",
    ]
    assert prepared_proposal.related_edge_ids == [edge_id]


def test_canonical_patch_persists_dependencies_seen_after_earlier_same_patch_operations(
    manifest,
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    edge_id = "edge/same-patch-removal-dependency"
    proposal_id = "prop/remove-after-same-patch-edge"
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Added evidence and proposed removing its belief target.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "ev/same-patch-removal-dependency",
                        "type": "evidence",
                        "title": "Same-patch removal evidence",
                        "observation": "The relation was created before the Proposal.",
                        "origin": "analytic",
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": edge_id,
                        "source": "ev/same-patch-removal-dependency",
                        "target": "hyp/replanning-restores-plasticity",
                        "relation": "weakens",
                        "assessment": {
                            "relevance": "direct",
                            "weight": "limited",
                            "scope": "Same-patch evidence fixture.",
                        },
                    }
                ],
            },
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": proposal_id,
                        "title": "Remove the belief and its new evidence relation",
                        "card": {"decision_needed": "Approve removing this belief."},
                        "ops": [
                            {
                                "op": "remove_nodes",
                                "intent": "removal",
                                "node_ids": ["hyp/replanning-restores-plasticity"],
                            }
                        ],
                    }
                ],
            },
        ],
    )

    canonical, result = history.append(patch)

    stored = canonical.ops[2].proposals[0]
    assert edge_id in stored.related_edge_ids
    assert edge_id in result.state.proposals[proposal_id].related_edge_ids
    assert edge_id in history.state().proposals[proposal_id].related_edge_ids


def test_agent_edge_touching_accepted_node_applies_directly(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    service = ProjectService(
        manifest,
        history,
        PaperService(manifest, AppStore(tmp_path / "app.sqlite3")),
    )
    service.review_node(
        "rq/learning-after-shift",
        ReviewRequest(standing="accepted"),
    )
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Added a decision and linked it to accepted content.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "dec/evaluation-shape",
                        "type": "decision",
                        "title": "Evaluation shape",
                        "question": "Which held-out evaluation should we use?",
                        "options": ["matched", "shifted"],
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/learning-after-shift",
                        "target": "dec/evaluation-shape",
                        "relation": "has_decision",
                    }
                ],
            },
        ],
    )

    history.append(patch)

    state = history.state()
    assert "rq/learning-after-shift::has_decision::dec/evaluation-shape" in state.edges
    assert state.nodes["rq/learning-after-shift"].standing == "accepted"


def test_proposal_with_unknown_repository_machine_is_rejected_at_creation(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    before = manifest.path.read_text(encoding="utf-8")
    patch = Patch(
        schema_generation=1,
        kind="refresh",
        author="agent",
        summary="Proposed a repository on an unknown machine.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/add-invalid-repository",
                        "title": "Add an invalid repository",
                        "card": {
                            "situation_cold": "A repository might contain relevant evidence.",
                            "why_human_now": "Repository membership is guarded.",
                            "consequences": "Future agents would read this repository.",
                            "decision_needed": "Decide whether to add the repository.",
                        },
                        "ops": [
                            {
                                "op": "set_project_truth_scope",
                                "intent": "legacy_project_truth_scope_change",
                                "truth_scope": ["repo-a", "repo-b", "repo-c"],
                                "repository": {
                                    "alias": "repo-c",
                                    "machine": "missing-machine",
                                    "path": "/research/repo-c",
                                },
                            }
                        ],
                        "related_config_keys": ["project_truth_scope"],
                        "base_rev": 1,
                    }
                ],
            }
        ],
    )

    with pytest.raises(PatchRejected) as caught:
        history.append(patch)

    assert any(message.code == "invalid-proposal-ops" for message in caught.value.report.messages)
    assert manifest.path.read_text(encoding="utf-8") == before
    assert load_manifest(manifest.path).project.truth_scope == ["repo-a", "repo-b"]
    assert len(list((manifest.research_dir / "patches").glob("*.json"))) == 2


def test_manifest_scope_write_validates_before_replacing_file(manifest) -> None:
    before = manifest.path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="unknown machine"):
        write_project_scope(
            manifest,
            ["repo-a", "repo-b", "repo-c"],
            repository_descriptor={
                "alias": "repo-c",
                "machine": "missing-machine",
                "path": "/research/repo-c",
            },
        )

    assert manifest.path.read_text(encoding="utf-8") == before
    assert load_manifest(manifest.path).project.truth_scope == ["repo-a", "repo-b"]


def test_proposal_replay_is_dry_run_materialized_at_creation(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    raw = proposal_patch().model_dump(mode="python", exclude_unset=True)
    raw_proposal = raw["ops"][0]["proposals"][0]
    raw_proposal["ops"] = [
        {
            "op": "create_edges",
            "intent": "protected_relation_change",
            "edges": [
                {
                    "source": "rq/learning-after-shift",
                    "target": "hyp/replanning-restores-plasticity",
                    "relation": "unknown_relation",
                }
            ],
        }
    ]
    raw_proposal["related_node_ids"] = [
        "hyp/replanning-restores-plasticity",
        "rq/learning-after-shift",
    ]
    patch = Patch.model_validate(raw)

    with pytest.raises(PatchRejected) as caught:
        history.append(patch)

    assert any(message.code == "invalid-proposal-ops" for message in caught.value.report.messages)
    assert len(list((manifest.research_dir / "patches").glob("*.json"))) == 2


def test_agent_cannot_propose_a_project_scope_change(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    patch = Patch(
        schema_generation=1,
        kind="refresh",
        author="agent",
        summary="Proposed adding a valid repository.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/add-valid-repository",
                        "title": "Add a valid repository",
                        "card": {
                            "situation_cold": "A repository contains relevant evidence.",
                            "why_human_now": "Repository membership is guarded.",
                            "consequences": "Future agents may read this repository.",
                            "decision_needed": "Decide whether to add the repository.",
                        },
                        "ops": [
                            {
                                "op": "set_project_truth_scope",
                                "intent": "legacy_project_truth_scope_change",
                                "truth_scope": ["repo-a", "repo-b", "repo-c"],
                                "repository": {
                                    "alias": "repo-c",
                                    "machine": "laptop",
                                    "path": str(tmp_path / "repo-c"),
                                },
                            }
                        ],
                        "related_config_keys": ["project_truth_scope"],
                        "base_rev": 1,
                    }
                ],
            }
        ],
    )

    with pytest.raises(PatchRejected):
        history.append(patch)

    assert history.state().project_truth_scope == ["repo-a", "repo-b"]
    assert history.state().proposals == {}
    assert load_manifest(manifest.path).project.truth_scope == ["repo-a", "repo-b"]
