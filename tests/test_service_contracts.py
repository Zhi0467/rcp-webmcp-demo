from __future__ import annotations

import pytest
from pydantic import ValidationError

from rcp.core.attention import decision_awaits_choice, project_graph_attention
from rcp.core.models import Blocker, Decision, GatedCard, GraphState, Proposal
from rcp.service import (
    ChatMessage,
    GraphSyncRequest,
    GraphUpdateResult,
    RunRequest,
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("open", False),
        ("ready", True),
        ("decided", False),
        ("revisit", True),
        ("superseded", False),
    ],
)
def test_decision_attention_predicate_is_ready_or_revisit(status: str, expected: bool) -> None:
    decision = Decision(
        id=f"dec/{status}",
        type="decision",
        title="Choose an option",
        question="Which option should be used?",
        options=["first", "second"],
        selected_option="first" if status in {"decided", "revisit"} else None,
        status=status,
    )

    assert decision_awaits_choice(decision) is expected


def test_graph_attention_projection_publishes_exact_membership_ids() -> None:
    state = GraphState(
        nodes={
            "dec/revisit": Decision(
                id="dec/revisit",
                type="decision",
                title="Revisit",
                question="Which option?",
                status="revisit",
            ),
            "dec/open": Decision(
                id="dec/open",
                type="decision",
                title="Open",
                question="Which option?",
                status="open",
            ),
            "blk/asserted": Blocker(
                id="blk/asserted",
                type="blocker",
                title="Asserted",
                description="Needs a human judgment.",
                status="open",
                standing="asserted",
            ),
            "blk/accepted": Blocker(
                id="blk/accepted",
                type="blocker",
                title="Accepted",
                description="Already judged.",
                status="open",
                standing="accepted",
            ),
        },
        proposals={
            "prop/pending": Proposal(
                id="prop/pending",
                title="Pending proposal",
                card=GatedCard(decision_needed="Choose."),
                ops=[],
                status="pending",
            ),
            "prop/approved": Proposal(
                id="prop/approved",
                title="Approved proposal",
                card=GatedCard(decision_needed="Already chosen."),
                ops=[],
                status="approved",
            ),
        },
    )

    assert project_graph_attention(state).model_dump(mode="json") == {
        "pending_proposal_ids": ["prop/pending"],
        "decisions_awaiting_choice_ids": ["dec/revisit"],
        "open_blocker_ids": ["blk/asserted"],
    }


@pytest.mark.parametrize("collection", ["nodes", "proposals"])
def test_graph_attention_projection_rejects_mapping_identity_mismatches(collection: str) -> None:
    state = GraphState(
        nodes={
            "dec/canonical": Decision(
                id="dec/canonical",
                type="decision",
                title="Choose",
                question="Which option?",
                status="ready",
            )
        },
        proposals={
            "prop/canonical": Proposal(
                id="prop/canonical",
                title="Choose",
                card=GatedCard(decision_needed="Choose."),
                ops=[],
                status="pending",
            )
        },
    )
    member = (
        state.nodes.pop("dec/canonical")
        if collection == "nodes"
        else state.proposals.pop("prop/canonical")
    )
    getattr(state, collection)["wrong/key"] = member

    with pytest.raises(ValueError, match="mapping key"):
        project_graph_attention(state)


def test_graph_sync_contract_has_no_ambiguity_resolution_path() -> None:
    with pytest.raises(ValidationError, match="ambiguities"):
        GraphSyncRequest.model_validate(
            {
                "base_revision": 1,
                "ambiguities": [{"ambiguity_id": "amb/legacy", "status": "dismissed"}],
            }
        )


def test_conversation_requests_carry_mode_and_nothing_else_authorizes_the_graph() -> None:
    request = RunRequest(mode="work", message="Run the experiment.")

    assert request.mode == "work"
    assert request.model_dump(mode="json") == {
        "provider": None,
        "run_truth_scope": None,
        "model": None,
        "reasoning": None,
        "run_on": None,
        "chat_scope": "node",
        "node_id": None,
        "message": "Run the experiment.",
        "chat_id": None,
        "session_id": None,
        "mode": "work",
        "trigger": "human",
        "patch_kind": "work",
        "control_node_id": None,
        "control_revision": None,
        "control_episode_id": None,
        "control_invocation": None,
        "control_invocation_ceiling": None,
        "control_decision_bundle": [],
        "control_completion_criteria": [],
        "watcher_ids": [],
        "workflow_ids": None,
        "skill_ids": None,
        "invoked_workflow_ids": [],
        "invoked_skill_ids": [],
        "invoked_provider_skill_names": [],
        "resolved_provider_skills": [],
        "resolved_skill_packages": None,
        "attachment_set_id": None,
        "attachment_client_id": None,
        "attachment_batch_id": None,
        "attachments": [],
    }


def test_the_retired_graph_gate_grants_no_authority() -> None:
    request = RunRequest.model_validate(
        {"message": "Update the graph.", "allow_graph_change": True}
    )

    assert request.mode == "discuss"
    assert "allow_graph_change" not in request.model_dump(mode="json")


def test_graph_update_result_round_trips_through_a_chat_message() -> None:
    graph_update = GraphUpdateResult(
        status="rejected",
        change_summary=["Recorded the experiment outcome."],
        proposal_ids=["prop/review-next-run"],
        validation_messages=["Patch revision is stale."],
        correction_rounds=2,
    )
    message = ChatMessage(
        message_id="message-1",
        role="assistant",
        text="The experiment completed.",
        timestamp="2026-08-01T12:00:00+00:00",
        mode="work",
        graph_update=graph_update,
    )

    assert message.graph_update == graph_update
    assert message.model_dump(mode="json")["graph_update"]["status"] == "rejected"


def test_conversation_mode_is_closed() -> None:
    try:
        RunRequest(mode="auto")
    except ValidationError:
        pass
    else:
        raise AssertionError("conversation mode must be discuss or work")
