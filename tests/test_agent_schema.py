from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rcp.agents import (
    AgentPatch,
    agent_output_schema,
    prepare_agent_patch,
    validate_agent_patch_shape,
)
from rcp.core.models import Patch, Proposal
from rcp.core.operations import graph_operations_from_proposal
from tests.helpers import graph_operation, seed_patch


def test_agent_patch_schema_accepts_the_canonical_seed_shape() -> None:
    patch = seed_patch()
    validate_agent_patch_shape(
        patch.model_copy(
            update={"ops": [operation for operation in patch.ops if operation.op != "set_coverage"]}
        )
    )


def test_agent_patch_schema_rejects_invented_node_fields_and_slug_formats() -> None:
    with pytest.raises(ValidationError, match="state|asserted|Extra inputs") as caught:
        Patch(
            kind="seed",
            author="agent",
            summary="Used an invented graph vocabulary.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "hyp-invented-shape",
                            "type": "hypothesis",
                            "title": "Invented shape",
                            "statement": "The schema should reject this before graph validation.",
                            "state": "supported",
                            "asserted": True,
                        }
                    ],
                }
            ],
        )

    assert "hyp-invented-shape" in str(caught.value) or "Extra inputs" in str(caught.value)


def test_agent_output_schema_describes_operations_instead_of_arbitrary_objects() -> None:
    schema = agent_output_schema()
    rendered = json.dumps(schema)

    assert '"create_nodes"' in rendered
    assert '"remove_nodes"' in rendered
    assert '"set_coverage"' not in rendered
    assert schema["$defs"]["NewEdge"]["properties"]["relation"]["pattern"].startswith("^")
    assert '"additionalProperties": false' in rendered
    assert "source_id" not in rendered
    assert "admission" not in schema["properties"]
    assert "admission_messages" not in schema["properties"]
    assert "experiment_control_node_id" not in schema["properties"]
    assert "experiment_decision_bundle" not in schema["properties"]
    assert "ValidationMessage" not in schema["$defs"]
    assert "layer" not in schema["$defs"]["NewEdge"]["properties"]
    for definition in ("NodeUpdate", "SupersedeNode", "NodeMerge"):
        assert "cause" in schema["$defs"][definition]["properties"]
    assert "cause" in schema["$defs"]["ProposalNodeUpdate"]["required"]


def test_agent_patch_schema_accepts_remove_nodes() -> None:
    validate_agent_patch_shape(
        Patch(
            kind="refresh",
            author="agent",
            summary="Removed stale asserted nodes.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "remove_nodes",
                    "node_ids": ["rq/obsolete-question", "hyp/obsolete-hypothesis"],
                }
            ],
        )
    )


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "remove_nodes", "node_ids": []},
        {"op": "remove_nodes", "node_ids": ["rq/obsolete"], "reason": "No longer useful"},
        {"op": "remove_nodes", "nodes": ["rq/obsolete"]},
        {"op": "delete_nodes", "node_ids": ["rq/obsolete"]},
    ],
)
def test_agent_remove_nodes_operation_is_strict(operation: dict[str, object]) -> None:
    values = {
        "kind": "refresh",
        "author": "agent",
        "summary": "Tried a malformed node removal.",
        "run_truth_scope": ["repo-a"],
        "repositories_read": ["repo-a"],
        "ops": [operation],
    }
    if operation == {"op": "remove_nodes", "node_ids": []}:
        patch = Patch.model_validate(values)
        with pytest.raises(ValueError, match="graph operation schema"):
            validate_agent_patch_shape(patch)
    else:
        with pytest.raises(ValidationError):
            Patch.model_validate(values)


def test_agent_patch_is_a_semantic_model_not_a_canonical_patch() -> None:
    schema = agent_output_schema()

    assert not issubclass(AgentPatch, Patch)
    assert set(schema["properties"]) == {
        "summary",
        "ops",
        "repositories_read",
        "change_summary",
    }


@pytest.mark.parametrize(
    "experiment_fields",
    [
        {"invocation_ceiling": "5"},
        {
            "attempts": [
                {
                    "id": "attempt/one",
                    "sequence": "1",
                    "purpose": "Reject nested coercion.",
                }
            ]
        },
        {
            "attempts": [
                {
                    "id": "attempt/one",
                    "sequence": 1,
                    "purpose": "Reject nested decision-pin coercion.",
                    "decision_bundle": [
                        {
                            "decision_id": "dec/budget",
                            "decision_revision": "2",
                            "selected_option": "small",
                        }
                    ],
                }
            ]
        },
    ],
)
def test_agent_schema_rejects_nested_scalar_coercion(
    experiment_fields: dict[str, object],
) -> None:
    experiment = {
        "id": "exp/strict-agent-payload",
        "type": "experiment",
        "title": "Strict agent payload",
        "objective": "Reject values with the wrong JSON scalar type.",
        **experiment_fields,
    }

    with pytest.raises(ValidationError):
        AgentPatch.model_validate(
            {
                "summary": "Tried to submit a coercible nested scalar.",
                "ops": [{"op": "create_nodes", "nodes": [experiment]}],
            }
        )


def test_agent_schema_accepts_datetime_strings_as_the_json_wire_form() -> None:
    patch = AgentPatch.model_validate(
        {
            "summary": "Recorded a source timestamp.",
            "ops": [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "exp/strict-agent-payload",
                            "type": "experiment",
                            "title": "Strict agent payload",
                            "objective": "Preserve the JSON datetime representation.",
                            "source_refs": [
                                {
                                    "machine": "local",
                                    "truth_repository": "repo-a",
                                    "source": "codex",
                                    "session_id": "session-1",
                                    "record_uuid": "record-1",
                                    "timestamp": "2026-08-17T09:30:00Z",
                                    "excerpt": "Observed strict validation.",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    assert patch.ops[0].nodes[0].source_refs[0].timestamp.year == 2026  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "field",
    [
        "kind",
        "author",
        "revision",
        "created_at",
        "run_truth_scope",
        "processed_cursors",
        "admission",
        "admission_messages",
        "experiment_control_node_id",
        "experiment_decision_bundle",
    ],
)
def test_raw_agent_patch_rejects_rcp_owned_top_level_metadata(field: str) -> None:
    raw: dict[str, object] = {"summary": "Recorded semantic graph changes.", "ops": []}
    raw[field] = "provider-supplied"

    with pytest.raises(ValidationError, match=field):
        AgentPatch.model_validate(raw)


def test_agent_output_schema_omits_nested_rcp_bookkeeping() -> None:
    definitions = agent_output_schema()["$defs"]
    node_definitions = {
        "AgentResearchQuestion",
        "AgentHypothesis",
        "AgentDecision",
        "AgentExperiment",
        "AgentEvidence",
        "AgentBlocker",
    }

    for definition in node_definitions:
        assert {"standing", "created_rev", "updated_rev"}.isdisjoint(
            definitions[definition]["properties"]
        )
    assert "created_rev" not in definitions["NewEdge"]["properties"]
    assert "AgentAmbiguity" not in definitions
    assert "CreateAmbiguitiesOperation" not in definitions
    assert "ResolveAmbiguitiesOperation" not in definitions
    assert "AgentGlossaryTerm" not in definitions
    assert "UpsertGlossaryOperation" not in definitions
    for definition in ("AgentSourceRef", "AgentExperimentAttempt", "AgentGatedCard"):
        assert definitions[definition]["additionalProperties"] is False
    assert {
        "base_rev",
        "related_node_ids",
        "related_edge_ids",
        "related_config_keys",
        "status",
        "raised_rev",
        "resolved_rev",
        "rejection_reason",
    }.isdisjoint(definitions["AgentProposal"]["properties"])


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "create_ambiguities",
            "ambiguities": [
                {
                    "id": "amb/missing-boundary",
                    "question": "What is the scope?",
                    "why_it_matters": "The Hypothesis needs a boundary.",
                }
            ],
        },
        {
            "op": "resolve_ambiguities",
            "resolutions": [{"id": "amb/missing-boundary", "status": "resolved"}],
        },
    ],
)
def test_agent_schema_has_no_ambiguity_operations(operation: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentPatch.model_validate({"summary": "Tried a retired operation.", "ops": [operation]})

    rendered = json.dumps(agent_output_schema())
    assert '"create_ambiguities"' not in rendered
    assert '"resolve_ambiguities"' not in rendered


def test_agent_schema_does_not_advertise_unpermitted_glossary_writes() -> None:
    operation = {
        "op": "upsert_glossary",
        "terms": [{"term": "RCP", "plain_definition": "Research Control Panel"}],
    }

    with pytest.raises(ValidationError):
        AgentPatch.model_validate({"summary": "Tried a glossary write.", "ops": [operation]})
    assert '"upsert_glossary"' not in json.dumps(agent_output_schema())


def test_agent_schema_allows_a_new_ready_decision() -> None:
    patch = AgentPatch.model_validate(
        {
            "summary": "Queued a makeable research choice.",
            "ops": [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "dec/evaluation-budget",
                            "type": "decision",
                            "title": "Evaluation budget",
                            "question": "Which evaluation budget should the study use?",
                            "options": ["small", "large"],
                            "status": "ready",
                        }
                    ],
                }
            ],
        }
    )

    assert patch.ops[0].nodes[0].status == "ready"  # type: ignore[union-attr]


def test_agent_schema_rejects_a_decision_proposal() -> None:
    raw = {
        "summary": "Tried to propose a Decision outcome.",
        "ops": [
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/select-budget",
                        "title": "Select the larger budget",
                        "card": {
                            "situation_cold": "The evaluation budget is ready to choose.",
                            "why_human_now": "The choice affects experiment cost.",
                            "consequences": "The larger evaluation will run.",
                            "decision_needed": "Select the large budget option.",
                        },
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "status_change",
                                "nodes": [
                                    {
                                        "id": "dec/evaluation-budget",
                                        "changes": {
                                            "selected_option": "large",
                                            "status": "decided",
                                        },
                                        "cause": {
                                            "kind": "evidence_edge",
                                            "ref_id": "ev/result::informs::dec/evaluation-budget",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValidationError, match="Hypothesis status"):
        AgentPatch.model_validate(raw)


def test_agent_schema_requires_an_evidence_cause_on_a_hypothesis_proposal() -> None:
    raw = {
        "summary": "Proposed a belief transition without evidence.",
        "ops": [
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/promote-claim",
                        "title": "Promote the claim",
                        "card": {
                            "situation_cold": "The claim has changed.",
                            "why_human_now": "Belief status is human-authoritative.",
                            "consequences": "The claim becomes active.",
                            "decision_needed": "Approve or reject active status.",
                        },
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "status_change",
                                "nodes": [
                                    {
                                        "id": "hyp/claim",
                                        "changes": {"status": "active"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValidationError, match="cause"):
        AgentPatch.model_validate(raw)


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "update_nodes",
            "intent": "content_change",
            "nodes": [{"id": "rq/claim", "changes": {"question": "A clearer question?"}}],
        },
        {
            "op": "update_nodes",
            "intent": "content_change",
            "nodes": [{"id": "rq/claim", "changes": {"status": "answered"}}],
        },
        {
            "op": "remove_nodes",
            "intent": "removal",
            "node_ids": ["hyp/claim"],
        },
        {
            "op": "supersede_nodes",
            "intent": "supersede",
            "nodes": [{"id": "hyp/claim", "superseded_by": "hyp/replacement"}],
        },
        {
            "op": "merge_nodes",
            "intent": "merge",
            "merges": [{"duplicate": "hyp/claim", "canonical": "hyp/replacement"}],
        },
        {
            "op": "create_edges",
            "intent": "protected_relation_change",
            "edges": [
                {
                    "source": "rq/claim",
                    "target": "hyp/claim",
                    "relation": "has_hypothesis",
                }
            ],
        },
        {
            "op": "update_nodes",
            "intent": "status_change",
            "nodes": [
                {
                    "id": "hyp/claim",
                    "changes": {"status": "active"},
                    "cause": {"kind": "evidence_edge", "ref_id": "edge/support"},
                }
            ],
        },
    ],
)
def test_agent_schema_accepts_each_declared_proposal_intent(operation) -> None:
    patch = AgentPatch.model_validate(
        {
            "summary": "Raised one protected change for human judgment.",
            "ops": [
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": "prop/review-belief",
                            "title": "Review the belief change",
                            "card": {"decision_needed": "Approve or reject this change."},
                            "ops": [operation],
                        }
                    ],
                }
            ],
        }
    )

    prepared = prepare_agent_patch(patch, kind="work", run_truth_scope=["repo-a"])

    assert prepared.ops[0].proposals[0].ops[0].intent == operation["intent"]


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "update_nodes",
            "nodes": [{"id": "rq/claim", "changes": {"question": "Missing intent?"}}],
        },
        {
            "op": "update_nodes",
            "intent": "content_change",
            "nodes": [
                {
                    "id": "rq/claim",
                    "changes": {"question": "No machine cause is needed."},
                    "cause": {"kind": "evidence_edge", "ref_id": "edge/support"},
                }
            ],
        },
        {
            "op": "remove_nodes",
            "intent": "removal",
            "node_ids": ["hyp/one", "hyp/two"],
        },
    ],
)
def test_agent_schema_rejects_undeclared_or_bundled_proposal_intent(operation) -> None:
    raw = {
        "summary": "Tried an invalid protected change shape.",
        "ops": [
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/invalid-belief-change",
                        "title": "Invalid belief change",
                        "card": {"decision_needed": "Do not admit this shape."},
                        "ops": [operation],
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValidationError):
        AgentPatch.model_validate(raw)


def test_agent_experiment_schema_uses_only_the_invocation_ceiling_name() -> None:
    experiment = {
        "id": "experiment/bounded-loop",
        "type": "experiment",
        "title": "Bounded loop",
        "objective": "Run a bounded sequence of agent invocations.",
        "attempt_ceiling": 5,
    }

    with pytest.raises(ValidationError, match="attempt_ceiling"):
        AgentPatch.model_validate(
            {
                "summary": "Created an Experiment with a legacy budget field.",
                "ops": [{"op": "create_nodes", "nodes": [experiment]}],
            }
        )

    assert "invocation_ceiling" in agent_output_schema()["$defs"]["AgentExperiment"]["properties"]


def test_new_agent_evidence_requires_an_explicit_origin() -> None:
    evidence = {
        "id": "ev/observed-recovery",
        "type": "evidence",
        "title": "Observed recovery",
        "observation": "The held-out learning curve recovered after replanning.",
    }
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Recorded evidence.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[{"op": "create_nodes", "nodes": [evidence]}],
    )

    with pytest.raises(ValueError, match="origin"):
        validate_agent_patch_shape(patch)

    evidence["origin"] = "internal_run"
    validate_agent_patch_shape(
        patch.model_copy(
            update={"ops": [graph_operation({"op": "create_nodes", "nodes": [evidence]})]}
        )
    )


def test_agent_belief_causes_allow_only_a_strict_evidence_edge_shape() -> None:
    cause = {"kind": "evidence_edge", "ref_id": "ev/result::supports::hyp/claim"}
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Changed a belief with a structured cause.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "changes": {"status": "supported"},
                        "cause": cause,
                    }
                ],
            }
        ],
    )

    validate_agent_patch_shape(patch)

    rendered = json.dumps(agent_output_schema())
    assert '"evidence_edge"' in rendered
    assert '"DecisionCause"' not in rendered
    assert '"proposal_resolution"' not in rendered


@pytest.mark.parametrize(
    "cause",
    [
        {"kind": "evidence_edge"},
        {"kind": "decision", "ref_id": "dec/evaluation-rule"},
        {"kind": "decision", "ref_id": "dec/evaluation-rule", "note": "extra"},
        {"kind": "proposal_resolution", "ref_id": "prop/revise-claim"},
        {"kind": "proposal_resolution", "ref_id": 7},
        {"kind": "human_edit"},
        {"kind": "human_edit", "ref_id": "human"},
        {"kind": "unknown"},
    ],
)
def test_agent_belief_causes_reject_missing_extra_or_unknown_fields(
    cause: dict[str, object],
) -> None:
    values = {
        "kind": "refresh",
        "author": "agent",
        "summary": "Used a malformed belief cause.",
        "run_truth_scope": ["repo-a"],
        "repositories_read": ["repo-a"],
        "ops": [
            {
                "op": "supersede_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "cause": cause,
                    }
                ],
            }
        ],
    }
    core_valid_causes = (
        {"kind": "decision", "ref_id": "dec/evaluation-rule"},
        {"kind": "proposal_resolution", "ref_id": "prop/revise-claim"},
        {"kind": "human_edit"},
    )
    if cause in core_valid_causes:
        patch = Patch.model_validate(values)
        with pytest.raises(ValueError, match="graph operation schema"):
            validate_agent_patch_shape(patch)
    else:
        with pytest.raises(ValidationError):
            Patch.model_validate(values)


def test_agent_edge_layer_is_backend_owned() -> None:
    raw = {
        "summary": "Tried to assign an edge layer.",
        "ops": [
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/question",
                        "target": "hyp/claim",
                        "relation": "has_hypothesis",
                        "layer": "action",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValidationError, match="layer"):
        AgentPatch.model_validate(raw)


def test_agent_schema_accepts_the_generic_extension_namespace() -> None:
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Recorded an active project-specific construct.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "mechanism_claim/optimizer-memory",
                        "type": "hypothesis",
                        "extension_type": "mechanism_claim",
                        "extension_fields": {
                            "mechanism_family": "optimizer state",
                            "directly_testable": True,
                            "alternative_explanations": ["data order", "parameter drift"],
                        },
                        "title": "Optimizer state carries task history",
                        "statement": "Optimizer state retains information about earlier tasks.",
                    }
                ],
            }
        ],
    )

    validate_agent_patch_shape(patch)


def test_agent_extension_fields_cannot_escape_the_namespace() -> None:
    with pytest.raises(ValidationError, match="mechanism_family|Extra inputs"):
        Patch(
            kind="refresh",
            author="agent",
            summary="Put a custom field at the node top level.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "mechanism_claim/optimizer-memory",
                            "type": "hypothesis",
                            "extension_type": "mechanism_claim",
                            "extension_fields": {},
                            "mechanism_family": "optimizer state",
                            "title": "Optimizer state carries task history",
                            "statement": "Optimizer state retains information about earlier tasks.",
                        }
                    ],
                }
            ],
        )


def test_agent_schema_accepts_custom_relation_names_without_a_layer() -> None:
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Connected two nodes with an active custom relation.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "mechanism_claim/optimizer-memory",
                        "target": "hyp/plasticity-loss",
                        "relation": "mechanistically_explains",
                    }
                ],
            }
        ],
    )

    validate_agent_patch_shape(patch)


def _ontology_proposal() -> dict[str, object]:
    return {
        "id": "prop/add-mechanism-claim",
        "title": "Add mechanism claims",
        "card": {
            "situation_cold": "The graph needs to distinguish causal mechanisms from predictions.",
            "why_human_now": "Only a human may activate project ontology changes.",
            "consequences": "Future agents may author mechanism claims and their fields.",
            "decision_needed": "Approve or reject the proposed ontology.",
        },
        "ops": [
            {
                "op": "set_ontology",
                "intent": "legacy_ontology_change",
                "ontology": {
                    "types": [
                        {
                            "name": "mechanism_claim",
                            "definition": "A causal account of an observed research result.",
                            "base_type": "hypothesis",
                            "layer": "epistemic",
                        }
                    ],
                    "fields": [
                        {
                            "owner_type": "mechanism_claim",
                            "name": "mechanism_family",
                            "definition": "The family of mechanisms under study.",
                            "kind": "text",
                            "required": True,
                            "agent_writable": True,
                        }
                    ],
                    "relations": [
                        {
                            "name": "mechanistically_explains",
                            "definition": "Connects a mechanism claim to what it explains.",
                            "source_types": ["mechanism_claim"],
                            "target_types": ["hypothesis"],
                            "layer": "epistemic",
                        }
                    ],
                },
            }
        ],
        "related_node_ids": [],
        "related_config_keys": ["ontology"],
        "base_rev": 3,
    }


def test_agent_cannot_apply_ontology_directly() -> None:
    proposal = Proposal.model_validate(_ontology_proposal())
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Tried to activate an ontology directly.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=graph_operations_from_proposal(proposal.ops),
    )

    with pytest.raises(ValueError, match="set_ontology|graph operation schema"):
        validate_agent_patch_shape(patch)


def test_agent_cannot_propose_an_ontology_change() -> None:
    patch = Patch(
        schema_generation=1,
        kind="refresh",
        author="agent",
        summary="Proposed a project ontology extension for human review.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[{"op": "create_proposals", "proposals": [_ontology_proposal()]}],
    )

    with pytest.raises(ValueError, match="set_ontology|graph operation schema"):
        validate_agent_patch_shape(patch)

    rendered = json.dumps(agent_output_schema())
    assert '"set_ontology"' not in rendered


def test_agent_cannot_resolve_or_reject_a_proposal() -> None:
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Tried to remove a pending human judgment.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "resolve_proposals",
                "resolutions": [{"id": "prop/pending", "status": "withdrawn"}],
            }
        ],
    )

    with pytest.raises(ValueError, match="resolve_proposals|graph operation schema"):
        validate_agent_patch_shape(patch)

    assert '"resolve_proposals"' not in json.dumps(agent_output_schema())


def test_agent_can_withdraw_a_pending_proposal() -> None:
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Withdrew an obsolete proposal.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "withdraw_proposals",
                "proposals": [
                    {"id": "prop/pending", "reason": "A later proposal supersedes this one."}
                ],
            }
        ],
    )

    validate_agent_patch_shape(patch)
    assert '"withdraw_proposals"' in json.dumps(agent_output_schema())


@pytest.mark.parametrize(
    "node",
    [
        {
            "id": "dec/already-chosen",
            "type": "decision",
            "title": "Already chosen",
            "question": "Which option?",
            "options": ["a", "b"],
            "selected_option": "a",
            "status": "decided",
        },
        {
            "id": "hyp/already-supported",
            "type": "hypothesis",
            "title": "Already supported",
            "statement": "The result supports this claim.",
            "status": "supported",
        },
    ],
)
def test_agent_created_decisions_and_hypotheses_start_unresolved(node: dict[str, object]) -> None:
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Tried to create a pre-resolved semantic node.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[{"op": "create_nodes", "nodes": [node]}],
    )

    with pytest.raises(ValueError, match="graph operation schema"):
        validate_agent_patch_shape(patch)


def test_rcp_prepares_canonical_metadata_and_proposal_bookkeeping() -> None:
    draft = AgentPatch.model_validate(
        {
            "summary": "Proposed promoting the hypothesis.",
            "repositories_read": ["repo-a"],
            "change_summary": ["Raised a belief transition for review."],
            "ops": [
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": "prop/promote-hypothesis",
                            "title": "Promote the hypothesis",
                            "card": {
                                "situation_cold": "New evidence supports the hypothesis.",
                                "why_human_now": "The belief transition requires human authority.",
                                "consequences": "The hypothesis will become active.",
                                "decision_needed": "Approve or reject the transition.",
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
                                                "ref_id": "edge/replanning-support",
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    patch = prepare_agent_patch(
        draft,
        kind="work",
        run_truth_scope=["repo-a"],
        source_operation_id="task-create-proposal",
    )
    proposal = patch.ops[0].proposals[0]

    assert isinstance(patch, Patch)
    assert patch.kind == "work"
    assert patch.author == "agent"
    assert patch.revision == 0
    assert patch.run_truth_scope == ["repo-a"]
    assert patch.repositories_read == ["repo-a"]
    assert patch.source_operation_id == "task-create-proposal"
    assert patch.change_summary == ["Raised a belief transition for review."]
    assert proposal.base_rev == 0
    assert proposal.related_node_ids == []
    assert proposal.related_edge_ids == []
    assert proposal.related_config_keys == []
    assert proposal.status == "pending"
    assert proposal.created_by == "agent"
    assert proposal.created_by_operation_id == "task-create-proposal"
    assert proposal.raised_rev == 0
    assert proposal.resolved_rev is None
    assert proposal.resolved_by is None
    assert proposal.resolved_by_operation_id is None
    assert proposal.resolution_reason is None
    assert proposal.rejection_reason is None


def test_a_repository_read_is_named_by_alias_whether_declared_as_alias_or_path() -> None:
    """A task contract hands out paths while run truth scope holds aliases.

    Both spellings name the same repository, so an honest declaration must not
    depend on which namespace the agent happened to copy.
    """

    repository_paths = {
        "vista": "/home/researcher/vista-followup",
        "vista-docs": "/home/researcher/vista-followup/docs",
        "sibling": "/home/researcher/vista",
    }
    declared = [
        "vista",
        "/home/researcher/vista-followup",
        "/home/researcher/vista-followup/experiments/probe/",
        "/home/researcher/vista-followup/docs/plan.md",
        "/home/researcher/vista",
        "/home/researcher/unregistered",
    ]
    draft = AgentPatch.model_validate(
        {"summary": "Record what this run read.", "ops": [], "repositories_read": declared}
    )

    prepared = prepare_agent_patch(
        draft,
        kind="work",
        run_truth_scope=["vista"],
        repository_paths=repository_paths,
    )

    # The nested repository wins over its parent, a shared path prefix is not
    # containment, and an unregistered path survives so scope validation reports it.
    assert prepared.repositories_read == [
        "vista",
        "vista-docs",
        "sibling",
        "/home/researcher/unregistered",
    ]
    unmapped = prepare_agent_patch(draft, kind="work", run_truth_scope=["vista"])
    assert unmapped.repositories_read == declared
