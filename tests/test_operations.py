from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from rcp.core.models import Evidence, Experiment, GraphState, Patch
from rcp.core.operations import CreateNodesOperation, CreateProposalsOperation
from rcp.history import HistoryManager


def _operation_examples() -> list[dict[str, object]]:
    return [
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "rq/typed-contract",
                    "type": "research_question",
                    "title": "Typed contract",
                    "question": "Does every operation retain its JSON shape?",
                }
            ],
        },
        {
            "op": "update_nodes",
            "nodes": [{"id": "rq/typed-contract", "changes": {"scope": "core"}}],
        },
        {
            "op": "create_edges",
            "edges": [
                {
                    "source": "rq/typed-contract",
                    "target": "hyp/typed-contract",
                    "relation": "has_hypothesis",
                }
            ],
        },
        {"op": "remove_edges", "edge_ids": ["edge/typed-contract"]},
        {"op": "remove_nodes", "node_ids": ["rq/typed-contract"]},
        {
            "op": "supersede_nodes",
            "nodes": [{"id": "rq/typed-contract", "superseded_by": "rq/current-contract"}],
        },
        {
            "op": "merge_nodes",
            "merges": [{"duplicate": "rq/typed-contract", "canonical": "rq/current-contract"}],
        },
        {
            "op": "create_ambiguities",
            "ambiguities": [
                {
                    "id": "amb/typed-contract",
                    "question": "Which contract applies?",
                    "why_it_matters": "Replay must be deterministic.",
                }
            ],
        },
        {
            "op": "resolve_ambiguities",
            "resolutions": [{"id": "amb/typed-contract", "status": "resolved"}],
        },
        {
            "op": "create_proposals",
            "proposals": [
                {
                    "id": "prop/typed-contract",
                    "title": "Clarify the contract",
                    "card": {},
                    "ops": [
                        {
                            "op": "update_nodes",
                            "intent": "content_change",
                            "nodes": [
                                {
                                    "id": "rq/typed-contract",
                                    "changes": {"scope": "typed core"},
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "op": "resolve_proposals",
            "resolutions": [{"id": "prop/typed-contract", "status": "approved"}],
        },
        {
            "op": "withdraw_proposals",
            "proposals": [{"id": "prop/typed-contract"}],
        },
        {
            "op": "upsert_glossary",
            "terms": [{"term": "typed_contract", "plain_definition": "One strict union."}],
        },
        {"op": "set_coverage", "coverage": {"repositories_seen": ["repo-a"]}},
        {"op": "set_standing", "node_id": "rq/typed-contract", "standing": "accepted"},
        {"op": "set_project_truth_scope", "truth_scope": ["repo-a"]},
        {"op": "set_ontology", "ontology": {"types": [], "fields": [], "relations": []}},
    ]


def _patch(operations: list[dict[str, object]]) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Exercise the typed operation contract.",
        ops=operations,
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
    )


def test_every_persisted_operation_round_trips_without_shape_changes() -> None:
    operations = _operation_examples()

    serialized = _patch(deepcopy(operations)).model_dump(mode="json")["ops"]

    assert serialized == operations
    assert (
        Patch.model_validate(_patch(operations).model_dump(mode="json")).model_dump(mode="json")[
            "ops"
        ]
        == operations
    )


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "unknown_operation"},
        {"op": "remove_edges", "edge_ids": ["edge/typed"], "extra": True},
        {"op": "remove_edges", "edge_ids": "edge/typed"},
    ],
)
def test_current_operation_decoding_is_strict(operation: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as caught:
        _patch([operation])

    assert "ops.0" in str(caught.value)


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": "rq/typed-contract",
                    "changes": {"scope": "core"},
                    "base_updated_rev": "1",
                }
            ],
        },
        {
            "op": "create_nodes",
            "nodes": [
                {
                    "id": "exp/typed-contract",
                    "type": "experiment",
                    "title": "Typed contract",
                    "objective": "Reject nested numeric coercion.",
                    "invocation_ceiling": "5",
                }
            ],
        },
        {
            "op": "create_proposals",
            "proposals": [
                {
                    "id": "prop/typed-contract",
                    "title": "Clarify the contract",
                    "card": {},
                    "base_rev": "0",
                    "ops": [
                        {
                            "op": "update_nodes",
                            "intent": "content_change",
                            "nodes": [
                                {
                                    "id": "rq/typed-contract",
                                    "changes": {"scope": "typed core"},
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "op": "set_ontology",
            "ontology": {
                "types": [
                    {
                        "name": "mechanism",
                        "definition": "A proposed causal mechanism.",
                        "base_type": "hypothesis",
                        "layer": "epistemic",
                        "deprecated": "false",
                    }
                ]
            },
        },
    ],
)
def test_current_operation_decoding_rejects_scalar_coercion_at_nested_boundaries(
    operation: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _patch([operation])


def test_strict_operations_preserve_persisted_datetime_wire_compatibility() -> None:
    raw = {
        "schema_generation": 1,
        "revision": 7,
        "kind": "refresh",
        "author": "agent",
        "producer": "agent",
        "summary": "Historical coverage timestamp.",
        "ops": [
            {
                "op": "set_coverage",
                "coverage": {"earliest_timestamp": "2026-08-17T09:30:00Z"},
            }
        ],
    }

    decoded = HistoryManager._decode_persisted_patch(json.dumps(raw))

    assert decoded.ops[0].coverage.earliest_timestamp is not None  # type: ignore[union-attr]
    assert decoded.ops[0].coverage.earliest_timestamp.year == 2026  # type: ignore[union-attr]


def test_nested_proposal_error_names_outer_operation_and_proposal() -> None:
    operation = deepcopy(_operation_examples()[9])
    operation["proposals"][0]["ops"][0]["unexpected"] = True  # type: ignore[index]

    with pytest.raises(ValidationError) as caught:
        _patch([operation])

    message = str(caught.value)
    assert "ops.0.create_proposals" in message
    assert "prop/typed-contract" in message
    assert "unexpected" in message


def test_persisted_rejected_unknown_operation_keeps_receipt_without_poisoning_replay() -> None:
    raw = {
        "revision": 7,
        "kind": "refresh",
        "author": "agent",
        "summary": "Rejected by an older decoder.",
        "ops": [{"op": "invent_nodes", "nodes": []}],
        "admission": "rejected",
        "admission_messages": [
            {
                "level": "reject",
                "code": "unknown-operation",
                "message": "The operation was never admitted.",
                "patch_revision": 7,
            }
        ],
    }

    decoded = HistoryManager._decode_persisted_patch(json.dumps(raw))

    assert decoded.schema_generation == 1
    assert decoded.admission == "rejected"
    assert decoded.ops == []
    assert decoded.admission_messages[0].code == "unknown-operation"


def test_persisted_current_generation_does_not_adapt_retired_evidence_strength() -> None:
    raw = {
        "schema_generation": 2,
        "revision": 7,
        "kind": "refresh",
        "author": "agent",
        "producer": "agent",
        "summary": "Invalid current Evidence shape.",
        "ops": [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "ev/current-invalid",
                        "type": "evidence",
                        "title": "Current invalid",
                        "observation": "Current schemas cannot author node-level strength.",
                        "strength": "supporting",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValidationError):
        HistoryManager._decode_persisted_patch(json.dumps(raw))


def test_persisted_current_generation_cannot_author_legacy_evidence_metadata() -> None:
    raw = {
        "schema_generation": 2,
        "revision": 7,
        "kind": "refresh",
        "author": "agent",
        "producer": "agent",
        "summary": "Invalid current compatibility metadata.",
        "ops": [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "ev/current-legacy",
                        "type": "evidence",
                        "title": "Current legacy",
                        "observation": "Current writes cannot mint historical metadata.",
                        "role": "result",
                        "legacy_strength": "supporting",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="schema-generation 2.*legacy_strength"):
        HistoryManager._decode_persisted_patch(json.dumps(raw))


def test_explicit_generation_one_uses_the_same_no_write_compatibility_adapter() -> None:
    raw = {
        "schema_generation": 1,
        "revision": 7,
        "kind": "refresh",
        "author": "agent",
        "producer": "agent",
        "summary": "Explicit prior generation.",
        "ops": [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "ev/explicit-prior",
                        "type": "evidence",
                        "title": "Explicit prior",
                        "observation": "An older Evidence assertion.",
                        "strength": "confirmatory",
                    },
                    {
                        "id": "exp/explicit-prior",
                        "type": "experiment",
                        "title": "Explicit prior experiment",
                        "objective": "Exercise compatibility.",
                        "status": "blocked",
                        "current_summary": "Waiting on a historical gate.",
                    },
                ],
            }
        ],
    }

    decoded = HistoryManager._decode_persisted_patch(json.dumps(raw))
    operation = decoded.ops[0]
    assert isinstance(operation, CreateNodesOperation)
    evidence, experiment = operation.nodes
    assert isinstance(evidence, Evidence)
    assert isinstance(experiment, Experiment)
    assert evidence.role == "result"
    assert evidence.legacy_strength == "confirmatory"
    assert experiment.status == "unspecified"
    assert experiment.current_summary_stale is True


def test_legacy_proposal_snapshot_round_trips_through_current_graph_state() -> None:
    raw = {
        "kind": "refresh",
        "author": "agent",
        "summary": "Historical proposal.",
        "ops": [
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/legacy-snapshot",
                        "title": "Legacy snapshot",
                        "card": {},
                        "ops": [
                            {
                                "op": "update_nodes",
                                "nodes": [
                                    {
                                        "id": "rq/typed-contract",
                                        "changes": {"scope": "historical"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    decoded = HistoryManager._decode_persisted_patch(json.dumps(raw))
    operation = decoded.ops[0]
    assert isinstance(operation, CreateProposalsOperation)
    proposal = operation.proposals[0]
    dumped = GraphState(proposals={proposal.id: proposal}).model_dump(mode="json")
    assert "intent" not in dumped["proposals"][proposal.id]["ops"][0]

    restored = GraphState.model_validate(dumped)

    assert restored.proposals[proposal.id].ops[0].intent == "legacy_content_change"


def test_current_generation_missing_nested_proposal_intent_keeps_precise_error_path() -> None:
    raw = {
        "schema_generation": 2,
        "kind": "refresh",
        "author": "agent",
        "producer": "agent",
        "summary": "Malformed current Proposal.",
        "ops": [
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/current-malformed",
                        "title": "Current malformed",
                        "card": {},
                        "ops": [
                            {
                                "op": "update_nodes",
                                "nodes": [{"id": "rq/typed-contract", "changes": {"scope": "new"}}],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValidationError) as caught:
        HistoryManager._decode_persisted_patch(json.dumps(raw))

    message = str(caught.value)
    assert "ops.0.create_proposals" in message
    assert "prop/current-malformed" in message
    assert "intent" in message


@pytest.mark.parametrize(
    ("nested_operation", "legacy_intent"),
    [
        (
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/legacy-proposal",
                        "type": "research_question",
                        "title": "Legacy Proposal",
                        "question": "Can old Proposal operations still be decoded?",
                    }
                ],
            },
            "legacy_create_nodes",
        ),
        (
            {
                "op": "create_ambiguities",
                "ambiguities": [
                    {
                        "id": "amb/legacy-proposal",
                        "question": "Which legacy shape?",
                        "why_it_matters": "Replay compatibility.",
                    }
                ],
            },
            "legacy_create_ambiguities",
        ),
        (
            {
                "op": "resolve_ambiguities",
                "resolutions": [{"id": "amb/legacy-proposal", "status": "resolved"}],
            },
            "legacy_resolve_ambiguities",
        ),
        (
            {
                "op": "upsert_glossary",
                "terms": [{"term": "legacy", "plain_definition": "Historical."}],
            },
            "legacy_upsert_glossary",
        ),
        (
            {"op": "set_coverage", "coverage": {"repositories_seen": ["repo-a"]}},
            "legacy_set_coverage",
        ),
    ],
)
def test_persisted_legacy_proposal_supports_every_previously_replayable_operation(
    nested_operation: dict[str, object],
    legacy_intent: str,
) -> None:
    raw = {
        "kind": "refresh",
        "author": "agent",
        "summary": "Historical Proposal operation.",
        "ops": [
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/legacy-vocabulary",
                        "title": "Legacy vocabulary",
                        "card": {},
                        "ops": [nested_operation],
                    }
                ],
            }
        ],
    }

    decoded = HistoryManager._decode_persisted_patch(json.dumps(raw))
    operation = decoded.ops[0]
    assert isinstance(operation, CreateProposalsOperation)
    assert operation.proposals[0].ops[0].intent == legacy_intent
    assert operation.proposals[0].model_dump(mode="json")["ops"] == [nested_operation]


@pytest.mark.parametrize(
    "semantic_operation",
    [
        {
            "op": "update_nodes",
            "intent": "content_change",
            "nodes": [{"id": "rq/legacy-approval", "changes": {"scope": "approved"}}],
        },
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [{"id": "hyp/legacy-approval", "changes": {"status": "supported"}}],
        },
        {
            "op": "remove_nodes",
            "intent": "removal",
            "node_ids": ["rq/legacy-approval"],
        },
        {
            "op": "supersede_nodes",
            "intent": "supersede",
            "nodes": [{"id": "rq/legacy-approval", "superseded_by": "rq/current"}],
        },
        {
            "op": "merge_nodes",
            "intent": "merge",
            "merges": [{"duplicate": "rq/legacy-approval", "canonical": "rq/current"}],
        },
        {
            "op": "create_edges",
            "intent": "protected_relation_change",
            "edges": [
                {
                    "source": "exp/legacy-approval",
                    "target": "dec/legacy-approval",
                    "relation": "governed_by",
                }
            ],
        },
        {
            "op": "remove_edges",
            "intent": "protected_relation_change",
            "edge_ids": ["edge/legacy-approval"],
        },
    ],
)
def test_persisted_legacy_approval_strips_proposal_only_top_level_intent(
    semantic_operation: dict[str, object],
) -> None:
    raw = {
        "revision": 9,
        "kind": "approval",
        "author": "human",
        "summary": "Historical approved Proposal semantics.",
        "ops": [semantic_operation],
    }

    decoded = HistoryManager._decode_persisted_patch(json.dumps(raw))

    assert decoded.schema_generation == 1
    assert "intent" not in decoded.model_dump(mode="json")["ops"][0]


def test_current_generation_does_not_strip_top_level_proposal_intent() -> None:
    raw = {
        "schema_generation": 2,
        "revision": 9,
        "kind": "approval",
        "author": "human",
        "producer": "human",
        "summary": "Malformed current approval.",
        "ops": [
            {
                "op": "update_nodes",
                "intent": "content_change",
                "nodes": [{"id": "rq/current", "changes": {"scope": "invalid"}}],
            }
        ],
    }

    with pytest.raises(ValidationError, match="intent"):
        HistoryManager._decode_persisted_patch(json.dumps(raw))
