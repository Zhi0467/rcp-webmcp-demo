from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rcp.core.authority import (
    CREATE_EDGE,
    UPDATE_PROTECTED_EPISTEMIC,
    AgentProfile,
    operation_actions,
    permits,
)
from rcp.core.materialize import apply_valid_patch, materialize_patches
from rcp.core.models import (
    EVIDENCE_ASSESSMENT_MAX_QUALIFICATIONS,
    EVIDENCE_ASSESSMENT_QUALIFICATION_MAX_LENGTH,
    EVIDENCE_ASSESSMENT_SCOPE_MAX_LENGTH,
    Decision,
    Evidence,
    EvidenceAssessment,
    GraphState,
    Hypothesis,
    Patch,
    Standing,
)
from rcp.core.operations import (
    CreateEdgesOperation,
    CreateNodesOperation,
    GraphOperation,
    NewEdge,
    NodeUpdate,
    RemoveEdgesOperation,
    UpdateNodesOperation,
)
from rcp.core.research_md import render_research_md
from rcp.core.validation import ValidationReport, validate_patch
from rcp.history import HistoryManager
from tests.helpers import create_named_app, fabricated_authorizer

EVIDENCE_ID = "ev/transfer-result"
HYPOTHESIS_A_ID = "hyp/transfer-persists"
HYPOTHESIS_B_ID = "hyp/transfer-is-scoped"
DECISION_ID = "dec/evaluation-rule"
EDGE_A_ID = "edge/transfer-a"
EDGE_B_ID = "edge/transfer-b"


def _evidence(**changes: object) -> Evidence:
    values: dict[str, object] = {
        "id": EVIDENCE_ID,
        "type": "evidence",
        "title": "Transfer result",
        "observation": "Transfer remained stable after the shift.",
        "origin": "internal_run",
    }
    values.update(changes)
    return Evidence.model_validate(values)


def _hypothesis(
    node_id: str,
    title: str,
    *,
    standing: Standing = Standing.ACCEPTED,
) -> Hypothesis:
    return Hypothesis(
        id=node_id,
        type="hypothesis",
        title=title,
        statement="Transfer remains stable after a task shift.",
        standing=standing,
    )


def _state() -> GraphState:
    nodes = [
        _evidence(),
        _hypothesis(HYPOTHESIS_A_ID, "Transfer persists"),
        _hypothesis(HYPOTHESIS_B_ID, "Transfer is scoped"),
        Decision(
            id=DECISION_ID,
            type="decision",
            title="Evaluation rule",
            question="Which evaluation rule should be used?",
        ),
    ]
    return GraphState(
        project_truth_scope=["repo-a"],
        nodes={node.id: node for node in nodes},
    )


def _assessment(
    relevance: str = "direct",
    weight: str = "strong",
    *,
    scope: str | None = None,
    qualifications: list[str] | None = None,
) -> EvidenceAssessment:
    return EvidenceAssessment.model_validate(
        {
            "relevance": relevance,
            "weight": weight,
            "scope": scope,
            "qualifications": qualifications or [],
        }
    )


def _edge(
    *,
    edge_id: str = EDGE_A_ID,
    source: str = EVIDENCE_ID,
    target: str = HYPOTHESIS_A_ID,
    relation: str = "supports",
    assessment: EvidenceAssessment | None = None,
) -> NewEdge:
    return NewEdge(
        id=edge_id,
        source=source,
        target=target,
        relation=relation,
        assessment=assessment,
    )


def _current_patch_document(
    *operations: object,
    revision: int = 0,
    profile: AgentProfile | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_generation": 2,
        "revision": revision,
        "kind": "work" if profile is not None else "refresh",
        "author": "agent",
        "producer": "agent",
        "profile": profile,
        "summary": "Recorded current Evidence semantics.",
        "run_truth_scope": ["repo-a"],
        "repositories_read": ["repo-a"],
        "ops": list(operations),
    }
    if profile is not None:
        document.update(
            authorized_by=fabricated_authorizer("Evidence test owner"),
            task_id=f"task/evidence-{profile}",
        )
    return document


def _agent_patch(
    *operations: GraphOperation,
    revision: int = 0,
    profile: AgentProfile | None = None,
) -> Patch:
    return Patch.model_validate(
        _current_patch_document(*operations, revision=revision, profile=profile)
    )


def _create_nodes_operation() -> CreateNodesOperation:
    return CreateNodesOperation(
        op="create_nodes",
        nodes=[
            _evidence(),
            _hypothesis(
                HYPOTHESIS_A_ID,
                "Transfer persists",
                standing=Standing.ASSERTED,
            ),
            _hypothesis(
                HYPOTHESIS_B_ID,
                "Transfer is scoped",
                standing=Standing.ASSERTED,
            ),
        ],
    )


def _codes(report: ValidationReport) -> set[str]:
    return {message.code for message in report.messages}


def _legacy_document(
    *,
    revision: int = 1,
    strength: str | None = None,
    include_strength: bool = True,
    include_unassessed_edge: bool = False,
    operations: list[dict[str, object]] | None = None,
    author: str = "agent",
    kind: str = "seed",
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "id": EVIDENCE_ID,
        "type": "evidence",
        "title": "Transfer result",
        "observation": "Transfer remained stable after the shift.",
    }
    if include_strength:
        evidence["strength"] = strength
    default_operations: list[dict[str, object]] = [{"op": "create_nodes", "nodes": [evidence]}]
    if include_unassessed_edge:
        default_operations[0]["nodes"] = [
            evidence,
            {
                "id": HYPOTHESIS_A_ID,
                "type": "hypothesis",
                "title": "Transfer persists",
                "statement": "Transfer remains stable after a task shift.",
            },
        ]
        default_operations.append(
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": EDGE_A_ID,
                        "source": EVIDENCE_ID,
                        "target": HYPOTHESIS_A_ID,
                        "relation": "supports",
                    }
                ],
            }
        )
    return {
        "revision": revision,
        "kind": kind,
        "author": author,
        "summary": "Historical Evidence patch.",
        "run_truth_scope": ["repo-a"],
        "repositories_read": ["repo-a"],
        "ops": operations if operations is not None else default_operations,
    }


def test_current_evidence_fields_and_legacy_rejection() -> None:
    with pytest.raises(ValidationError, match="strength"):
        Patch.model_validate(
            _current_patch_document(
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": EVIDENCE_ID,
                            "type": "evidence",
                            "title": "Transfer result",
                            "observation": "Transfer remained stable.",
                            "origin": "internal_run",
                            "strength": "confirmatory",
                        }
                    ],
                }
            )
        )

    live_legacy = _agent_patch(
        CreateNodesOperation(
            op="create_nodes",
            nodes=[_evidence(legacy_strength="supporting")],
        )
    )
    legacy_report = validate_patch(GraphState(), live_legacy, ["repo-a"])
    assert "live-legacy-evidence-strength" in _codes(legacy_report)

    state = _state()
    strength_update = _agent_patch(
        UpdateNodesOperation(
            op="update_nodes",
            nodes=[NodeUpdate(id=EVIDENCE_ID, changes={"strength": "supporting"})],
        )
    )
    assert "invalid-node-update" in _codes(validate_patch(state, strength_update, ["repo-a"]))
    legacy_update = _agent_patch(
        UpdateNodesOperation(
            op="update_nodes",
            nodes=[NodeUpdate(id=EVIDENCE_ID, changes={"legacy_strength": "supporting"})],
        )
    )
    assert "live-legacy-evidence-strength" in _codes(
        validate_patch(state, legacy_update, ["repo-a"])
    )

    current = _agent_patch(
        CreateNodesOperation(
            op="create_nodes",
            nodes=[
                _evidence(
                    role="diagnostic",
                    validity="qualified",
                    origin="analytic",
                )
            ],
        ),
        revision=1,
    )
    report = validate_patch(GraphState(), current, ["repo-a"])
    assert not report.rejected
    evidence = apply_valid_patch(GraphState(), current).nodes[EVIDENCE_ID]
    assert isinstance(evidence, Evidence)
    assert (evidence.role, evidence.validity, evidence.origin) == (
        "diagnostic",
        "qualified",
        "analytic",
    )


@pytest.mark.parametrize(
    ("strength", "include_strength", "expected_legacy", "expected_role"),
    [
        ("diagnostic", True, "diagnostic", "diagnostic"),
        ("preliminary", True, "preliminary", "result"),
        ("supporting", True, "supporting", "result"),
        ("confirmatory", True, "confirmatory", "result"),
        (None, False, "preliminary", "result"),
    ],
)
def test_history_decoder_preserves_legacy_strength_without_rewriting_bytes(
    manifest,
    strength: str | None,
    include_strength: bool,
    expected_legacy: str,
    expected_role: str,
) -> None:
    history = HistoryManager(manifest)
    history.ensure_layout()
    patch_path = manifest.research_dir / "patches" / "000001.json"
    source_bytes = json.dumps(
        _legacy_document(
            strength=strength,
            include_strength=include_strength,
            include_unassessed_edge=True,
        ),
        separators=(",", ":"),
    ).encode("utf-8")
    patch_path.write_bytes(source_bytes)

    decoded = history.load_patches()[0]

    assert patch_path.read_bytes() == source_bytes
    assert decoded.schema_generation == 1
    operation = decoded.ops[0]
    assert isinstance(operation, CreateNodesOperation)
    evidence = operation.nodes[0]
    assert isinstance(evidence, Evidence)
    assert evidence.legacy_strength == expected_legacy
    assert evidence.role == expected_role
    dumped = evidence.model_dump(mode="json")
    assert "strength" not in dumped
    assert "weight" not in dumped
    edge_operation = decoded.ops[1]
    assert isinstance(edge_operation, CreateEdgesOperation)
    assert edge_operation.edges[0].assessment is None


@pytest.mark.parametrize(
    "relation",
    ["supports", "weakens", "refutes", "inconclusive", "contradicts"],
)
def test_new_evidence_hypothesis_relations_require_assessment(relation: str) -> None:
    state = _state()
    missing = _agent_patch(
        CreateEdgesOperation(
            op="create_edges",
            edges=[_edge(relation=relation)],
        )
    )
    assert "missing-evidence-assessment" in _codes(validate_patch(state, missing, ["repo-a"]))

    assessed = _agent_patch(
        CreateEdgesOperation(
            op="create_edges",
            edges=[_edge(relation=relation, assessment=_assessment())],
        )
    )
    assert not validate_patch(state, assessed, ["repo-a"]).rejected


def test_assessment_applicability_is_endpoint_and_relation_sensitive() -> None:
    state = _state()
    evidence_contradiction = _agent_patch(
        CreateEdgesOperation(
            op="create_edges",
            edges=[_edge(relation="contradicts", assessment=_assessment())],
        )
    )
    assert not validate_patch(state, evidence_contradiction, ["repo-a"]).rejected

    hypothesis_contradiction = _agent_patch(
        CreateEdgesOperation(
            op="create_edges",
            edges=[
                _edge(
                    source=HYPOTHESIS_A_ID,
                    target=HYPOTHESIS_B_ID,
                    relation="contradicts",
                    assessment=_assessment(),
                )
            ],
        )
    )
    assert "inapplicable-evidence-assessment" in _codes(
        validate_patch(state, hypothesis_contradiction, ["repo-a"])
    )

    evidence_to_decision = _agent_patch(
        CreateEdgesOperation(
            op="create_edges",
            edges=[
                _edge(
                    target=DECISION_ID,
                    relation="informs",
                    assessment=_assessment(),
                )
            ],
        )
    )
    assert "inapplicable-evidence-assessment" in _codes(
        validate_patch(state, evidence_to_decision, ["repo-a"])
    )

    wrong_unassessed_endpoint = _agent_patch(
        CreateEdgesOperation(
            op="create_edges",
            edges=[_edge(target=DECISION_ID, relation="supports")],
        )
    )
    wrong_report = validate_patch(state, wrong_unassessed_endpoint, ["repo-a"])
    assert "invalid-evidence-relation-endpoints" in _codes(wrong_report)
    assert wrong_report.rejected


@pytest.mark.parametrize(
    "assessment",
    [
        pytest.param({"weight": "strong"}, id="missing-relevance"),
        pytest.param({"relevance": "direct"}, id="missing-weight"),
        pytest.param({"relevance": "near", "weight": "strong"}, id="relevance"),
        pytest.param({"relevance": "direct", "weight": "decisive"}, id="weight"),
        pytest.param(
            {"relevance": "direct", "weight": "strong", "score": 1},
            id="unknown-field",
        ),
        pytest.param(
            {
                "relevance": "direct",
                "weight": "strong",
                "scope": "x" * (EVIDENCE_ASSESSMENT_SCOPE_MAX_LENGTH + 1),
            },
            id="scope-bound",
        ),
        pytest.param(
            {"relevance": "direct", "weight": "strong", "scope": "  "},
            id="blank-scope",
        ),
        pytest.param(
            {"relevance": "direct", "weight": "strong", "qualifications": [" "]},
            id="blank-qualification",
        ),
        pytest.param(
            {
                "relevance": "direct",
                "weight": "strong",
                "qualifications": ["one limit", " one limit "],
            },
            id="duplicate-qualification",
        ),
        pytest.param(
            {
                "relevance": "direct",
                "weight": "strong",
                "qualifications": ["x" * (EVIDENCE_ASSESSMENT_QUALIFICATION_MAX_LENGTH + 1)],
            },
            id="qualification-bound",
        ),
        pytest.param(
            {
                "relevance": "direct",
                "weight": "strong",
                "qualifications": [
                    f"qualification {index}"
                    for index in range(EVIDENCE_ASSESSMENT_MAX_QUALIFICATIONS + 1)
                ],
            },
            id="qualification-count",
        ),
    ],
)
def test_assessment_schema_is_strict_and_bounded(assessment: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Patch.model_validate(
            _current_patch_document(
                {
                    "op": "create_edges",
                    "edges": [
                        {
                            "id": EDGE_A_ID,
                            "source": EVIDENCE_ID,
                            "target": HYPOTHESIS_A_ID,
                            "relation": "supports",
                            "assessment": assessment,
                        }
                    ],
                }
            )
        )


def test_assessment_text_is_normalized() -> None:
    patch = Patch.model_validate(
        _current_patch_document(
            {
                "op": "create_edges",
                "edges": [
                    {
                        "id": EDGE_A_ID,
                        "source": EVIDENCE_ID,
                        "target": HYPOTHESIS_A_ID,
                        "relation": "supports",
                        "assessment": {
                            "relevance": "direct",
                            "weight": "strong",
                            "scope": "  held-out task shifts  ",
                            "qualifications": ["  one seed  ", "small sample"],
                        },
                    }
                ],
            }
        )
    )
    operation = patch.ops[0]
    assert isinstance(operation, CreateEdgesOperation)
    assessment = operation.edges[0].assessment
    assert assessment is not None
    assert assessment.scope == "held-out task shifts"
    assert assessment.qualifications == ["one seed", "small sample"]


def test_one_evidence_node_can_hold_distinct_assessments_for_two_hypotheses() -> None:
    patch = _agent_patch(
        CreateEdgesOperation(
            op="create_edges",
            edges=[
                _edge(
                    edge_id=EDGE_A_ID,
                    target=HYPOTHESIS_A_ID,
                    assessment=_assessment(
                        "direct",
                        "strong",
                        scope="task shifts",
                    ),
                ),
                _edge(
                    edge_id=EDGE_B_ID,
                    target=HYPOTHESIS_B_ID,
                    assessment=_assessment(
                        "contextual",
                        "limited",
                        qualifications=["The second claim is broader."],
                    ),
                ),
            ],
        ),
        revision=1,
    )
    state = _state()
    report = validate_patch(state, patch, ["repo-a"])
    assert not report.rejected

    updated = apply_valid_patch(state, patch)

    first = updated.edges[EDGE_A_ID].assessment
    second = updated.edges[EDGE_B_ID].assessment
    assert first is not None and second is not None
    assert (first.relevance, first.weight, first.scope) == (
        "direct",
        "strong",
        "task shifts",
    )
    assert (second.relevance, second.weight, second.qualifications) == (
        "contextual",
        "limited",
        ["The second claim is broader."],
    )


def test_historical_unassessed_evidence_relation_replays_and_renders_as_legacy() -> None:
    create = _legacy_document(
        strength="supporting",
        include_unassessed_edge=True,
    )
    accept = _legacy_document(
        revision=2,
        author="human",
        kind="approval",
        operations=[
            {
                "op": "set_standing",
                "node_id": HYPOTHESIS_A_ID,
                "standing": "accepted",
            }
        ],
    )
    accept.pop("run_truth_scope")
    accept.pop("repositories_read")
    patches = [
        HistoryManager._decode_persisted_patch(json.dumps(create)),
        HistoryManager._decode_persisted_patch(json.dumps(accept)),
    ]

    result = materialize_patches(patches, ["repo-a"], repository_aliases=["repo-a"])

    assert result.state.replay_status == "complete"
    assert not any(report.rejected for report in result.reports.values())
    assert result.state.edges[EDGE_A_ID].assessment is None
    rendered = render_research_md(result.state)
    assert "`supports`" in rendered
    assert "unassessed legacy relation" in rendered


@pytest.mark.parametrize("profile", ["ordinary", "orchestrator"])
def test_evidence_edge_authority_stays_direct_while_hypothesis_edits_stay_protected(
    profile: AgentProfile,
) -> None:
    state = _state()
    edge_patch = _agent_patch(
        CreateEdgesOperation(
            op="create_edges",
            edges=[_edge(assessment=_assessment())],
        ),
        profile=profile,
    )
    edge_action = operation_actions(state, edge_patch, edge_patch.ops[0])
    assert edge_action == frozenset({CREATE_EDGE})
    assert all(permits(edge_patch, action) for action in edge_action)
    edge_report = validate_patch(state, edge_patch, ["repo-a"])
    assert not edge_report.rejected

    hypothesis_patch = _agent_patch(
        UpdateNodesOperation(
            op="update_nodes",
            nodes=[
                NodeUpdate(
                    id=HYPOTHESIS_A_ID,
                    changes={"statement": "The agent changed protected belief content."},
                )
            ],
        ),
        profile=profile,
    )
    protected_action = operation_actions(
        state,
        hypothesis_patch,
        hypothesis_patch.ops[0],
    )
    assert protected_action == frozenset({UPDATE_PROTECTED_EPISTEMIC})
    assert not any(permits(hypothesis_patch, action) for action in protected_action)
    protected_report = validate_patch(state, hypothesis_patch, ["repo-a"])
    assert "graph-action-refused" in _codes(protected_report)
    assert "invalid-agent-attribution" not in _codes(protected_report)


def test_remove_and_create_replaces_assessment_without_rewriting_prior_patch(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(
        _agent_patch(
            _create_nodes_operation(),
            CreateEdgesOperation(
                op="create_edges",
                edges=[
                    _edge(
                        assessment=_assessment(
                            "indirect",
                            "moderate",
                            qualifications=["Initial interpretation."],
                        )
                    )
                ],
            ),
        )
    )
    original_path = manifest.research_dir / "patches" / "000001.json"
    original_bytes = original_path.read_bytes()

    history.append(
        _agent_patch(
            RemoveEdgesOperation(op="remove_edges", edge_ids=[EDGE_A_ID]),
            CreateEdgesOperation(
                op="create_edges",
                edges=[
                    _edge(
                        assessment=_assessment(
                            "direct",
                            "strong",
                            scope="held-out shifts",
                        )
                    )
                ],
            ),
        )
    )

    assert original_path.read_bytes() == original_bytes
    assert sorted(path.name for path in original_path.parent.glob("*.json")) == [
        "000001.json",
        "000002.json",
    ]
    final = history.state().edges[EDGE_A_ID].assessment
    assert final is not None
    assert (final.relevance, final.weight, final.scope) == (
        "direct",
        "strong",
        "held-out shifts",
    )


def test_research_markdown_and_graph_dump_place_assessment_on_edges() -> None:
    state = apply_valid_patch(
        _state(),
        _agent_patch(
            CreateEdgesOperation(
                op="create_edges",
                edges=[
                    _edge(
                        assessment=_assessment(
                            "direct",
                            "strong",
                            scope="held-out shifts",
                            qualifications=["one seed"],
                        )
                    )
                ],
            ),
            revision=1,
        ),
    )

    rendered = render_research_md(state)
    dumped = state.model_dump(mode="json")

    assert "**Transfer result** `supports` **Transfer persists**" in rendered
    assert "direct, strong, scope: held-out shifts, qualifications: one seed" in rendered
    assert dumped["edges"][EDGE_A_ID]["assessment"] == {
        "relevance": "direct",
        "weight": "strong",
        "scope": "held-out shifts",
        "qualifications": ["one seed"],
    }
    assert "strength" not in dumped["nodes"][EVIDENCE_ID]

    historical = HistoryManager._decode_persisted_patch(
        json.dumps(_legacy_document(strength="confirmatory"))
    )
    operation = historical.ops[0]
    assert isinstance(operation, CreateNodesOperation)
    historical_dump = operation.nodes[0].model_dump(mode="json")
    assert historical_dump["legacy_strength"] == "confirmatory"
    assert "strength" not in historical_dump


def test_graph_api_serializes_assessment_on_the_relation(manifest, tmp_path) -> None:
    history = HistoryManager(manifest)
    history.ensure_layout()
    legacy_create = _legacy_document(
        strength="confirmatory",
        include_unassessed_edge=True,
    )
    (manifest.research_dir / "patches" / "000001.json").write_text(
        json.dumps(legacy_create),
        encoding="utf-8",
    )
    history.append(
        _agent_patch(
            RemoveEdgesOperation(op="remove_edges", edge_ids=[EDGE_A_ID]),
            CreateEdgesOperation(
                op="create_edges",
                edges=[
                    _edge(
                        assessment=_assessment(
                            "direct",
                            "strong",
                            scope="held-out shifts",
                            qualifications=["one seed"],
                        )
                    )
                ],
            ),
        )
    )
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)

    response = client.get(f"/api/projects/{app.state.default_project_id}/graph")

    assert response.status_code == 200
    graph = response.json()
    assert graph["edges"][EDGE_A_ID]["assessment"] == {
        "relevance": "direct",
        "weight": "strong",
        "scope": "held-out shifts",
        "qualifications": ["one seed"],
    }
    assert "strength" not in graph["nodes"][EVIDENCE_ID]
    assert graph["nodes"][EVIDENCE_ID]["legacy_strength"] == "confirmatory"
