from __future__ import annotations

import json

from rcp.core.materialize import MaterializationResult
from rcp.core.models import (
    GatedCard,
    GraphState,
    Patch,
    Proposal,
    ResearchQuestion,
    Standing,
)
from rcp.core.validation import ValidationReport
from rcp.history import build_refresh_delta


def _patch(
    revision: int,
    kind: str,
    author: str,
    ops: list[dict[str, object]],
) -> Patch:
    return Patch.model_validate(
        {
            "revision": revision,
            "kind": kind,
            "author": author,
            "summary": f"Revision {revision}",
            "run_truth_scope": [] if author == "human" else ["repo-a"],
            "ops": ops,
        }
    )


def test_refresh_delta_preserves_human_and_chat_routing_metadata() -> None:
    patches = [
        _patch(1, "seed", "agent", []),
        _patch(
            2,
            "approval",
            "human",
            [
                {
                    "op": "set_standing",
                    "node_id": "rq/old-contested",
                    "standing": "contested",
                }
            ],
        ),
        _patch(3, "refresh", "agent", []),
        _patch(
            4,
            "approval",
            "human",
            [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "rq/current",
                            "base_updated_rev": 1,
                            "changes": {
                                "title": "Current title",
                                "question": "Sensitive prose is intentionally absent.",
                            },
                        }
                    ],
                }
            ],
        ),
        _patch(
            5,
            "work",
            "agent",
            [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "rq/current",
                            "changes": {"motivation": "Also absent from routing metadata."},
                        }
                    ],
                }
            ],
        ),
        _patch(
            6,
            "approval",
            "human",
            [
                {
                    "op": "resolve_proposals",
                    "resolutions": [{"id": "prop/choice", "status": "approved"}],
                }
            ],
        ),
    ]
    state = MaterializationResult(
        state=GraphState(
            revision=6,
            project_truth_scope=["repo-a"],
            nodes={
                "rq/old-contested": ResearchQuestion(
                    id="rq/old-contested",
                    type="research_question",
                    title="Older contested question",
                    question="Question text",
                    standing=Standing.CONTESTED,
                    created_rev=1,
                    updated_rev=2,
                ),
                "rq/current": ResearchQuestion(
                    id="rq/current",
                    type="research_question",
                    title="Current title",
                    question="Current question",
                    standing=Standing.ACCEPTED,
                    created_rev=1,
                    updated_rev=5,
                ),
            },
            proposals={
                "prop/choice": Proposal(
                    id="prop/choice",
                    title="Choose the evaluation",
                    card=GatedCard(),
                    ops=[],
                    base_rev=3,
                    status="approved",
                    raised_rev=3,
                    resolved_rev=6,
                )
            },
        ),
        reports={patch.revision: ValidationReport() for patch in patches},
    )

    delta = build_refresh_delta(patches, state)

    assert delta.after_revision == 3
    assert delta.through_revision == 6
    assert delta.omitted_count == 0
    assert delta.entries[0].category == "current_contested"
    assert delta.entries[0].target_id == "rq/old-contested"
    assert delta.entries[0].previous_standing == Standing.ASSERTED
    assert delta.entries[0].current_standing == Standing.CONTESTED
    assert {
        (entry.category, entry.target_id, tuple(entry.field_names)) for entry in delta.entries
    } >= {
        ("human_prose_edit", "rq/current", ("question", "title")),
        ("chat_graph_update", "rq/current", ("motivation",)),
        ("proposal_decision", "prop/choice", ("status",)),
    }
    encoded = delta.model_dump_json()
    assert "Sensitive prose" not in encoded
    assert "Also absent" not in encoded


def test_refresh_delta_records_explicit_and_agent_reset_standing_transitions() -> None:
    patches = [
        _patch(
            1,
            "seed",
            "agent",
            [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "rq/current",
                            "type": "research_question",
                            "title": "Current question",
                            "question": "What should the project test?",
                        }
                    ],
                }
            ],
        ),
        _patch(
            2,
            "approval",
            "human",
            [
                {
                    "op": "set_standing",
                    "node_id": "rq/current",
                    "standing": "accepted",
                }
            ],
        ),
        _patch(
            3,
            "chat",
            "agent",
            [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "rq/current",
                            "changes": {"question": "What causal comparison should be tested?"},
                        }
                    ],
                }
            ],
        ),
    ]
    materialization = MaterializationResult(
        state=GraphState(
            revision=3,
            project_truth_scope=["repo-a"],
            nodes={
                "rq/current": ResearchQuestion(
                    id="rq/current",
                    type="research_question",
                    title="Current question",
                    question="What causal comparison should be tested?",
                    standing=Standing.ASSERTED,
                    created_rev=1,
                    updated_rev=3,
                )
            },
        ),
        reports={patch.revision: ValidationReport() for patch in patches},
    )

    delta = build_refresh_delta(patches, materialization)
    transitions = [entry for entry in delta.entries if entry.category == "standing_transition"]

    assert [
        (entry.revision, entry.previous_standing, entry.current_standing)
        for entry in reversed(transitions)
    ] == [
        (2, Standing.ASSERTED, Standing.ACCEPTED),
        (3, Standing.ACCEPTED, Standing.ASSERTED),
    ]


def test_refresh_delta_records_human_and_work_node_removals_once() -> None:
    patches = [
        _patch(1, "refresh", "agent", []),
        _patch(
            2,
            "approval",
            "human",
            [{"op": "remove_nodes", "node_ids": ["rq/removed-by-human"]}],
        ),
        _patch(
            3,
            "work",
            "agent",
            [{"op": "remove_nodes", "node_ids": ["hyp/removed-by-work"]}],
        ),
    ]
    materialization = MaterializationResult(
        state=GraphState(revision=3, project_truth_scope=["repo-a"]),
        reports={patch.revision: ValidationReport() for patch in patches},
    )

    delta = build_refresh_delta(patches, materialization)
    removals = [entry for entry in delta.entries if entry.category == "node_removal"]

    assert [
        (entry.target_id, entry.author, entry.target_type, entry.field_names) for entry in removals
    ] == [
        ("hyp/removed-by-work", "agent", "node", ["removed"]),
        ("rq/removed-by-human", "human", "node", ["removed"]),
    ]


def test_refresh_delta_is_deterministically_bounded() -> None:
    patches = [_patch(1, "refresh", "agent", [])]
    nodes = {}
    for revision in range(2, 102):
        node_id = f"rq/item-{revision:03d}"
        patches.append(
            _patch(
                revision,
                "chat",
                "agent",
                [
                    {
                        "op": "create_nodes",
                        "nodes": [
                            {
                                "id": node_id,
                                "type": "research_question",
                                "title": "T" * 500,
                                "question": "Not serialized.",
                            }
                        ],
                    }
                ],
            )
        )
        nodes[node_id] = ResearchQuestion(
            id=node_id,
            type="research_question",
            title="T" * 500,
            question="Not serialized.",
            created_rev=revision,
            updated_rev=revision,
        )
    materialization = MaterializationResult(
        state=GraphState(
            revision=101,
            project_truth_scope=["repo-a"],
            nodes=nodes,
        ),
        reports={patch.revision: ValidationReport() for patch in patches},
    )

    first = build_refresh_delta(patches, materialization)
    second = build_refresh_delta(reversed(patches), materialization)
    encoded = json.dumps(
        first.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert first == second
    assert len(first.entries) <= 50
    assert len(encoded) <= 16 * 1024
    assert first.omitted_count > 0
    assert first.omitted_from_revision is not None
    assert first.omitted_through_revision is not None
    assert first.omitted_from_revision <= first.omitted_through_revision
