from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rcp.core.materialize import apply_valid_patch
from rcp.core.models import HUMAN_EDITABLE_NODE_FIELDS, Patch, ValidationMessage
from rcp.core.operations import (
    CreateNodesOperation,
    RemoveNodesOperation,
    ResolveProposalsOperation,
    SetOntologyOperation,
    SetStandingOperation,
    UpdateNodesOperation,
    graph_operations_from_proposal,
)
from rcp.core.validation import validate_patch
from rcp.history import HistoryManager
from rcp.service import GraphSyncNodeChange, GraphSyncRequest, ReviewRequest, RunRequest
from rcp.storage import AgentTaskRecord
from tests.helpers import append_fixture_patch, seed_patch
from tests.helpers import create_named_app as create_app

from .helpers import authorized_human


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
        "fields": [
            {
                "owner_type": "mechanism_hypothesis",
                "name": "mechanism",
                "definition": "The proposed causal mechanism.",
                "kind": "text",
                "required": False,
                "agent_writable": True,
                "deprecated": False,
            }
        ],
        "relations": [],
    }


def custom_hypothesis_payload() -> dict[str, object]:
    return {
        "id": "mechanism_hypothesis/custom-mechanism",
        "type": "hypothesis",
        "extension_type": "mechanism_hypothesis",
        "extension_fields": {"mechanism": "Replanning restores unused update directions."},
        "title": "Replanning mechanism",
        "statement": "Periodic replanning preserves future plasticity.",
    }


def append_decision_fixture(service, *, with_proposals: bool) -> None:
    nodes = [
        {
            "id": "dec/evaluation-rule",
            "type": "decision",
            "title": "Evaluation rule",
            "question": "Which evaluation rule should govern the experiment?",
            "options": ["matched", "shifted"],
        }
    ]
    ops: list[dict[str, object]] = [{"op": "create_nodes", "nodes": nodes}]
    if with_proposals:
        nodes.append(
            {
                "id": "exp/evaluation",
                "type": "experiment",
                "title": "Evaluation",
                "objective": "Evaluate the intervention under the chosen rule.",
            }
        )
        ops.append(
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "exp/evaluation",
                        "target": "dec/evaluation-rule",
                        "relation": "governed_by",
                    }
                ],
            }
        )
    append_fixture_patch(
        service,
        Patch(
            kind="refresh",
            author="agent",
            summary="Recorded the evaluation Decision.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=ops,
        ),
    )
    if not with_proposals:
        return
    proposals = []
    for suffix, option in (("matched", "matched"), ("shifted", "shifted")):
        proposals.append(
            {
                "id": f"prop/evaluation-{suffix}",
                "title": f"Choose {option}",
                "card": {
                    "situation_cold": "The evaluation needs a rule.",
                    "why_human_now": "Only the human can choose it.",
                    "consequences": "The experiment will use the selected rule.",
                    "decision_needed": f"Choose {option}?",
                },
                "ops": [
                    {
                        "op": "update_nodes",
                        "intent": "legacy_content_change",
                        "nodes": [
                            {
                                "id": "dec/evaluation-rule",
                                "changes": {
                                    "selected_option": option,
                                    "status": "decided",
                                },
                            }
                        ],
                    }
                ],
                "related_node_ids": ["dec/evaluation-rule"],
                "base_rev": 2,
            }
        )
    state = service.history.state()
    legacy_patch = Patch(
        schema_generation=1,
        revision=state.revision + 1,
        kind="refresh",
        author="agent",
        summary="Proposed two evaluation choices before Decision Proposals were retired.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[{"op": "create_proposals", "proposals": proposals}],
        admission="accepted",
    )
    assert not validate_patch(state, legacy_patch, ["repo-a"], mode="replay").rejected
    path = service.history.patches_dir / f"{legacy_patch.revision:06d}.json"
    path.write_text(legacy_patch.model_dump_json(indent=2) + "\n", encoding="utf-8")
    service.history.materialize(write_outputs=True)


def test_graph_sync_commits_staged_wording_and_judgment_once(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    node = service.history.state().nodes["rq/learning-after-shift"]
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 2,
            "nodes": [
                {
                    "node_id": node.id,
                    "base_updated_rev": node.updated_rev,
                    "changes": {"title": "Learning after a task shift"},
                    "standing": "accepted",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 3
    assert response.json()["nodes"][node.id]["standing"] == "accepted"
    assert response.json()["nodes"][node.id]["title"] == "Learning after a task shift"
    assert len(service.history.load_patches()) == 3
    assert "Learning after a task shift" in (manifest.research_dir / "research.md").read_text(
        encoding="utf-8"
    )


def test_graph_sync_directly_decides_an_ungoverned_decision(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=False)
    decision = service.history.state().nodes["dec/evaluation-rule"]
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 3,
            "nodes": [
                {
                    "node_id": decision.id,
                    "base_updated_rev": decision.updated_rev,
                    "changes": {
                        "selected_option": "shifted",
                        "status": "decided",
                    },
                    "standing": "accepted",
                }
            ],
        },
    )

    assert response.status_code == 200
    chosen = response.json()["nodes"][decision.id]
    assert chosen["selected_option"] == "shifted"
    assert chosen["status"] == "decided"
    assert chosen["standing"] == "accepted"
    assert "selected_option" not in HUMAN_EDITABLE_NODE_FIELDS["decision"]
    assert "status" in HUMAN_EDITABLE_NODE_FIELDS["decision"]


def test_graph_sync_atomically_edits_and_selects_a_decision_option(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=False)
    decision = service.history.state().nodes["dec/evaluation-rule"]
    revised_option = "shifted, with the human's additional rationale"
    revised_options = ["matched", revised_option]
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 3,
            "nodes": [
                {
                    "node_id": decision.id,
                    "base_updated_rev": decision.updated_rev,
                    "changes": {
                        "options": revised_options,
                        "selected_option": revised_option,
                        "status": "decided",
                    },
                    "standing": "accepted",
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    chosen = response.json()["nodes"][decision.id]
    assert chosen["options"] == revised_options
    assert chosen["selected_option"] == revised_option
    assert chosen["status"] == "decided"
    assert chosen["standing"] == "accepted"


def test_graph_sync_queues_a_decision_without_claiming_choice_authority(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=False)
    state = service.history.state()
    decision = state.nodes["dec/evaluation-rule"]
    request = GraphSyncRequest(
        base_revision=state.revision,
        nodes=[
            GraphSyncNodeChange(
                node_id=decision.id,
                base_updated_rev=decision.updated_rev,
                changes={"status": "ready"},
            )
        ],
    )

    patches = service._build_sync_patches(request, state, active_control_node_ids=set())

    assert len(patches) == 1
    assert patches[0].human_action is None
    assert not validate_patch(state, patches[0], ["repo-a"]).rejected
    queued = apply_valid_patch(state, patches[0]).nodes[decision.id]
    assert queued.status == "ready"


def test_graph_sync_direct_choice_atomically_withdraws_same_decision_proposals(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=True)
    before_sync = service.history.state()
    decision = before_sync.nodes["dec/evaluation-rule"]
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 4,
            "nodes": [
                {
                    "node_id": decision.id,
                    "base_updated_rev": decision.updated_rev,
                    "changes": {
                        "selected_option": "shifted",
                        "status": "decided",
                        "rationale": "The human chose the shifted evaluation directly.",
                    },
                    "standing": "accepted",
                }
            ],
            "proposals": [
                {
                    "proposal_id": "prop/evaluation-matched",
                    "decision": "approved",
                },
                {
                    "proposal_id": "prop/evaluation-shifted",
                    "decision": "rejected",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 5
    decision_payload = response.json()["nodes"][decision.id]
    assert decision_payload["selected_option"] == "shifted"
    assert decision_payload["status"] == "decided"
    assert decision_payload["standing"] == "accepted"
    assert {proposal["status"] for proposal in response.json()["proposals"].values()} == {
        "withdrawn"
    }
    stored = service.history.load_patches()[-1]
    assert [operation.op for operation in stored.ops] == [
        "update_nodes",
        "resolve_proposals",
        "set_standing",
    ]
    resolution_operation = stored.ops[1]
    assert isinstance(resolution_operation, ResolveProposalsOperation)
    resolutions = resolution_operation.resolutions
    assert {item.id for item in resolutions} == {
        "prop/evaluation-matched",
        "prop/evaluation-shifted",
    }
    assert all(item.status == "withdrawn" and item.reason for item in resolutions)
    assert all("human decided" in item for item in stored.change_summary if "Proposal" in item)
    assert not validate_patch(before_sync, stored, ["repo-a"], mode="replay").rejected


def test_direct_choice_withdraws_a_replay_valid_mixed_target_legacy_proposal(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=False)
    state = service.history.state()
    legacy_patch = Patch(
        schema_generation=1,
        revision=state.revision + 1,
        kind="refresh",
        author="agent",
        summary="Recorded a legacy mixed-target Proposal.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/legacy-mixed-target",
                        "title": "Choose matched and retitle the question",
                        "card": {"decision_needed": "Approve both legacy changes?"},
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "legacy_content_change",
                                "nodes": [
                                    {
                                        "id": "dec/evaluation-rule",
                                        "changes": {
                                            "selected_option": "matched",
                                            "status": "decided",
                                        },
                                    },
                                    {
                                        "id": "rq/learning-after-shift",
                                        "changes": {"title": "Retitled by a legacy Proposal"},
                                    },
                                ],
                            }
                        ],
                        "related_node_ids": [
                            "dec/evaluation-rule",
                            "rq/learning-after-shift",
                        ],
                        "base_rev": state.revision,
                    }
                ],
            }
        ],
    )
    replay_report = validate_patch(state, legacy_patch, ["repo-a"], mode="replay")
    assert not replay_report.rejected
    state = apply_valid_patch(state, legacy_patch)
    decision = state.nodes["dec/evaluation-rule"]
    request = GraphSyncRequest(
        base_revision=state.revision,
        nodes=[
            GraphSyncNodeChange(
                node_id=decision.id,
                base_updated_rev=decision.updated_rev,
                changes={"selected_option": "shifted", "status": "decided"},
                standing="accepted",
            )
        ],
        proposals=[
            {
                "proposal_id": "prop/legacy-mixed-target",
                "decision": "approved",
            }
        ],
    )

    patches = service._build_sync_patches(request, state, active_control_node_ids=set())

    assert len(patches) == 1
    resolution_operation = next(
        operation for operation in patches[0].ops if operation.op == "resolve_proposals"
    )
    assert isinstance(resolution_operation, ResolveProposalsOperation)
    assert resolution_operation.resolutions[0].id == "prop/legacy-mixed-target"
    assert resolution_operation.resolutions[0].status == "withdrawn"
    assert not validate_patch(state, patches[0], ["repo-a"]).rejected


def test_direct_choice_validator_requires_every_targeted_proposal_withdrawal(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=True)
    state = service.history.state()
    decision = state.nodes["dec/evaluation-rule"]
    patch = Patch(
        kind="approval",
        author="human",
        summary="Tried to leave one superseded Proposal pending.",
        human_action="decision_choice",
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": decision.id,
                        "base_updated_rev": decision.updated_rev,
                        "changes": {
                            "selected_option": "shifted",
                            "status": "decided",
                        },
                    }
                ],
            },
            {
                "op": "resolve_proposals",
                "resolutions": [
                    {
                        "id": "prop/evaluation-shifted",
                        "status": "withdrawn",
                        "reason": "The human decided directly.",
                    }
                ],
            },
        ],
    )

    report = validate_patch(state, patch, ["repo-a"])

    assert report.rejected
    assert any(
        message.code == "invalid-direct-decision-choice"
        and "prop/evaluation-matched" in message.message
        for message in report.messages
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"selected_option": "not-listed", "status": "decided"},
        {"selected_option": "matched"},
        {"status": "decided"},
    ],
)
def test_graph_sync_rejects_incoherent_direct_decision_choice(manifest, tmp_path, changes) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=False)
    decision = service.history.state().nodes["dec/evaluation-rule"]
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 3,
            "nodes": [
                {
                    "node_id": decision.id,
                    "base_updated_rev": decision.updated_rev,
                    "changes": changes,
                }
            ],
        },
    )

    assert response.status_code == 422
    assert decision.id in response.text
    assert service.history.state().revision == 3


def test_direct_choice_repairs_a_legacy_selected_but_open_decision(manifest, tmp_path) -> None:
    """A pre-fix approval left an option selected while status stayed open.

    Clicking that same option is the only repair available, and it stages no
    `selected_option` change because the option is already the canonical one.
    """

    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=False)
    state = service.history.state()
    legacy_proposal = Patch(
        schema_generation=1,
        revision=state.revision + 1,
        kind="refresh",
        author="agent",
        summary="Recorded a legacy Proposal that selected without deciding.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/legacy-selection",
                        "title": "Choose shifted",
                        "card": {"decision_needed": "Choose shifted?"},
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "legacy_content_change",
                                "nodes": [
                                    {
                                        "id": "dec/evaluation-rule",
                                        "changes": {"selected_option": "shifted"},
                                    }
                                ],
                            }
                        ],
                        "related_node_ids": ["dec/evaluation-rule"],
                        "base_rev": state.revision,
                    }
                ],
            }
        ],
    )
    assert not validate_patch(state, legacy_proposal, ["repo-a"], mode="replay").rejected
    state = apply_valid_patch(state, legacy_proposal)
    legacy_approval = Patch(
        revision=state.revision + 1,
        kind="approval",
        author="human",
        summary="Approved the legacy Proposal before the implied status existed.",
        ops=[
            *graph_operations_from_proposal(state.proposals["prop/legacy-selection"].ops),
            {
                "op": "resolve_proposals",
                "resolutions": [{"id": "prop/legacy-selection", "status": "approved"}],
            },
        ],
    )
    assert not validate_patch(state, legacy_approval, ["repo-a"], mode="replay").rejected
    state = apply_valid_patch(state, legacy_approval)
    decision = state.nodes["dec/evaluation-rule"]
    assert (decision.selected_option, decision.status) == ("shifted", "open")

    request = GraphSyncRequest(
        base_revision=state.revision,
        nodes=[
            GraphSyncNodeChange(
                node_id=decision.id,
                base_updated_rev=decision.updated_rev,
                # What the UI sends for a click on the already-selected option.
                changes={"status": "decided"},
                standing="accepted",
            )
        ],
    )
    patches = service._build_sync_patches(request, state, active_control_node_ids=set())

    assert len(patches) == 1
    assert patches[0].human_action == "decision_choice"
    assert not validate_patch(state, patches[0], ["repo-a"]).rejected
    repaired = apply_valid_patch(state, patches[0]).nodes["dec/evaluation-rule"]
    assert (repaired.selected_option, repaired.status) == ("shifted", "decided")
    assert repaired.standing == "accepted"


def test_graph_sync_approves_a_legacy_decision_proposal_through_decision_choice(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=False)
    state = service.history.state()
    legacy_proposal = Patch(
        schema_generation=1,
        revision=state.revision + 1,
        kind="refresh",
        author="agent",
        summary="Recorded a legacy Decision Proposal.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_proposals",
                "proposals": [
                    {
                        "id": "prop/legacy-decision-choice",
                        "title": "Choose shifted",
                        "card": {"decision_needed": "Choose shifted?"},
                        "ops": [
                            {
                                "op": "update_nodes",
                                "intent": "legacy_content_change",
                                "nodes": [
                                    {
                                        "id": "dec/evaluation-rule",
                                        "changes": {"selected_option": "shifted"},
                                    }
                                ],
                            }
                        ],
                        "related_node_ids": ["dec/evaluation-rule"],
                        "base_rev": state.revision,
                    }
                ],
            }
        ],
    )
    assert not validate_patch(state, legacy_proposal, ["repo-a"], mode="replay").rejected
    state = apply_valid_patch(state, legacy_proposal)
    request = GraphSyncRequest(
        base_revision=state.revision,
        proposals=[
            {
                "proposal_id": "prop/legacy-decision-choice",
                "decision": "approved",
            }
        ],
    )

    patches = service._build_sync_patches(request, state, active_control_node_ids=set())

    assert len(patches) == 1
    assert patches[0].human_action == "decision_choice"
    assert not validate_patch(state, patches[0], ["repo-a"]).rejected
    updated = apply_valid_patch(state, patches[0])
    assert updated.proposals["prop/legacy-decision-choice"].status == "approved"
    decision = updated.nodes["dec/evaluation-rule"]
    assert (decision.selected_option, decision.status) == ("shifted", "decided")


def test_a_decision_choice_patch_that_does_not_name_the_action_is_refused(
    manifest, tmp_path
) -> None:
    """The producer names the authority action; the validator never infers it.

    An ordinary node edit and a direct choice are both one `update_nodes` on
    one node, so shape cannot tell them apart. Without the marker this patch is
    an ordinary edit, and ordinary edits may not touch a Decision's choice.
    """

    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=False)
    state = service.history.state()
    decision = state.nodes["dec/evaluation-rule"]
    patch = Patch(
        kind="approval",
        author="human",
        summary="Chose an option without naming the action.",
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": decision.id,
                        "base_updated_rev": decision.updated_rev,
                        "changes": {"selected_option": "shifted", "status": "decided"},
                    }
                ],
            }
        ],
    )

    report = validate_patch(state, patch, ["repo-a"])

    assert report.rejected
    assert any(message.code == "non-prose-node-edit" for message in report.messages)


def test_direct_choice_refuses_a_proposal_withdrawal_without_an_id(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_decision_fixture(service, with_proposals=True)
    state = service.history.state()
    decision = state.nodes["dec/evaluation-rule"]
    with pytest.raises(ValidationError) as exc_info:
        Patch(
            kind="approval",
            author="human",
            summary="Withdrew Proposals without naming them.",
            human_action="decision_choice",
            ops=[
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": decision.id,
                            "base_updated_rev": decision.updated_rev,
                            "changes": {"selected_option": "shifted", "status": "decided"},
                        }
                    ],
                },
                {
                    "op": "resolve_proposals",
                    "resolutions": [
                        {"status": "withdrawn", "reason": "The human decided directly."},
                        {"status": "withdrawn", "reason": "The human decided directly."},
                    ],
                },
            ],
        )

    missing_id_locations = {
        tuple(error["loc"]) for error in exc_info.value.errors() if error["type"] == "missing"
    }
    assert missing_id_locations == {
        ("ops", 1, "resolve_proposals", "resolutions", 0, "id"),
        ("ops", 1, "resolve_proposals", "resolutions", 1, "id"),
    }


@pytest.mark.parametrize(
    ("initial_status", "initial_standing", "synced_status"),
    [
        ("resolved", "accepted", "open"),
        ("open", "accepted", "resolved"),
        ("open", "contested", "superseded"),
    ],
)
def test_graph_sync_updates_blocker_lifecycle_directly(
    manifest, tmp_path, initial_status, initial_standing, synced_status
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(
        service,
        Patch(
            kind="refresh",
            author="agent",
            summary="Recorded an operational blocker.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "blk/missing-capacity",
                            "type": "blocker",
                            "title": "Missing capacity",
                            "description": "The required accelerator is unavailable.",
                            "status": initial_status,
                        }
                    ],
                }
            ],
        ),
    )
    append_fixture_patch(
        service,
        Patch(
            kind="approval",
            author="human",
            summary=f"Marked the operational blocker {initial_standing}.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": "blk/missing-capacity",
                    "standing": initial_standing,
                }
            ],
        ),
    )
    blocker = service.history.state().nodes["blk/missing-capacity"]

    response = TestClient(app).post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 4,
            "nodes": [
                {
                    "node_id": blocker.id,
                    "base_updated_rev": blocker.updated_rev,
                    "changes": {"status": synced_status},
                }
            ],
        },
    )

    expected_history = f"Updated lifecycle for “Missing capacity”: status is now {synced_status}."
    expected_history_sentences = [expected_history, "“Missing capacity” is now asserted."]
    assert response.status_code == 200
    assert response.json()["nodes"][blocker.id]["status"] == synced_status
    assert response.json()["nodes"][blocker.id]["standing"] == "asserted"
    assert service.project_snapshot()["counts"]["open_blockers"] == (
        1 if synced_status == "open" else 0
    )
    stored = service.history.load_patches()[-1]
    assert stored.kind == "approval"
    assert len(stored.ops) == 2
    update_operation, standing_operation = stored.ops
    assert isinstance(update_operation, UpdateNodesOperation)
    assert len(update_operation.nodes) == 1
    assert update_operation.nodes[0].id == blocker.id
    assert update_operation.nodes[0].base_updated_rev == blocker.updated_rev
    assert update_operation.nodes[0].changes == {"status": synced_status}
    assert isinstance(standing_operation, SetStandingOperation)
    assert standing_operation.node_id == blocker.id
    assert standing_operation.standing == "asserted"
    assert stored.change_summary == expected_history_sentences
    assert service.history.revision_summaries(from_revision=5, to_revision=5)[0]["sentences"] == (
        expected_history_sentences
    )


def test_graph_sync_builds_and_commits_from_the_single_in_lock_current_replay(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    node = service.history.state().nodes["rq/learning-after-shift"]
    calls: list[tuple[bool, bool]] = []
    materialize = service.history.materialize

    def counted_materialize(*, write_outputs=True, pending_patch_paths=None):
        calls.append((write_outputs, pending_patch_paths is not None))
        return materialize(
            write_outputs=write_outputs,
            pending_patch_paths=pending_patch_paths,
        )

    monkeypatch.setattr(service.history, "materialize", counted_materialize)

    state = service.sync_graph(
        GraphSyncRequest(
            base_revision=2,
            nodes=[
                GraphSyncNodeChange(
                    node_id=node.id,
                    base_updated_rev=node.updated_rev,
                    standing="accepted",
                )
            ],
        ),
        active_control_node_ids=set(),
        authorized_by=authorized_human(app),
    )

    assert state.revision == 3
    # The transition is prepared from one current replay, then the one committed
    # patch is replayed to write every materialized output. There is no pending
    # batch replay path whose private staging files could diverge from history.
    assert calls == [(False, False), (True, False)]


def test_project_service_coalesces_concurrent_index_builds(manifest, tmp_path, monkeypatch) -> None:
    service = create_app(str(manifest.path), data_dir=tmp_path / "data").state.service
    builds = 0
    rendezvous = threading.Barrier(2)
    snapshot = object()

    def build(**_kwargs):
        nonlocal builds
        builds += 1
        with suppress(threading.BrokenBarrierError):
            rendezvous.wait(timeout=0.2)
        return snapshot

    monkeypatch.setattr(service.indexer, "build", build)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _item: service.index_snapshot(), range(2)))

    assert builds == 1
    assert results == [snapshot, snapshot]


def test_graph_sync_withdraws_to_asserted_and_rewrites_research_once(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    node = service.history.state().nodes["rq/learning-after-shift"]
    service.review_node(
        node.id,
        ReviewRequest(standing="accepted"),
        authorized_by=authorized_human(app),
    )
    accepted = service.history.state().nodes[node.id]
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 3,
            "nodes": [
                {
                    "node_id": node.id,
                    "base_updated_rev": accepted.updated_rev,
                    "standing": "asserted",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 4
    assert response.json()["nodes"][node.id]["standing"] == "asserted"
    assert (manifest.research_dir / "research.md").read_text(encoding="utf-8") == ""


def test_graph_sync_no_net_change_writes_no_patch(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    node = service.history.state().nodes["rq/learning-after-shift"]
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 2,
            "nodes": [
                {
                    "node_id": node.id,
                    "base_updated_rev": node.updated_rev,
                    "standing": "asserted",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 2
    assert len(service.history.load_patches()) == 2


def test_graph_sync_removes_node_and_its_incident_edges(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 2,
            "removed_node_ids": ["rq/learning-after-shift"],
        },
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 3
    assert "rq/learning-after-shift" not in response.json()["nodes"]
    assert response.json()["edges"] == {}
    stored = service.history.load_patches()[-1]
    assert stored.kind == "approval"
    assert stored.author == "human"
    assert len(stored.ops) == 1
    remove_operation = stored.ops[0]
    assert isinstance(remove_operation, RemoveNodesOperation)
    assert remove_operation.node_ids == ["rq/learning-after-shift"]


def test_graph_sync_removal_preserves_base_revision_conflict(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(
        service,
        Patch(
            kind="approval",
            author="human",
            summary="Moved the graph after the draft opened.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": "hyp/replanning-restores-plasticity",
                    "standing": "contested",
                }
            ],
        ),
    )
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 2,
            "removed_node_ids": ["rq/learning-after-shift"],
        },
    )

    assert response.status_code == 409
    assert "graph changed after this draft began" in response.json()["detail"]
    assert "rq/learning-after-shift" in service.history.state().nodes
    assert service.history.state().revision == 3


@pytest.mark.parametrize("same_draft", [False, True])
@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_graph_sync_staged_decision_withdraws_proposal_made_stale_by_node_removal(
    manifest, tmp_path, same_draft, decision
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(
        service,
        Patch(
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
                                "situation_cold": "The causal explanation is only proposed.",
                                "why_human_now": "Activation changes experiment interpretation.",
                                "consequences": "Evidence will be organized around it.",
                                "decision_needed": "Decide whether it should become active.",
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
                            "observation": "The observed behavior warrants activation testing.",
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
                            },
                        }
                    ],
                },
            ],
        ),
    )
    project_id = app.state.default_project_id
    client = TestClient(app)
    if same_draft:
        decided = client.post(
            f"/api/projects/{project_id}/sync",
            json={
                "base_revision": 3,
                "removed_node_ids": ["hyp/replanning-restores-plasticity"],
                "proposals": [
                    {
                        "proposal_id": "prop/activate-replanning-hypothesis",
                        "decision": decision,
                    }
                ],
            },
        )
    else:
        removed = client.post(
            f"/api/projects/{project_id}/sync",
            json={
                "base_revision": 3,
                "removed_node_ids": ["hyp/replanning-restores-plasticity"],
            },
        )
        assert removed.status_code == 200
        assert removed.json()["proposals"]["prop/activate-replanning-hypothesis"]["status"] == (
            "pending"
        )

        decided = client.post(
            f"/api/projects/{project_id}/sync",
            json={
                "base_revision": 4,
                "proposals": [
                    {
                        "proposal_id": "prop/activate-replanning-hypothesis",
                        "decision": decision,
                    }
                ],
            },
        )

    assert decided.status_code == 200
    assert decided.json()["revision"] == (4 if same_draft else 5)
    assert (
        decided.json()["proposals"]["prop/activate-replanning-hypothesis"]["status"] == "withdrawn"
    )
    assert "hyp/replanning-restores-plasticity" not in decided.json()["nodes"]
    withdrawal_reason = (
        "The proposal “Treat replanning as the active hypothesis” became stale because a related "
        "research concept was removed in this Sync."
        if same_draft
        else "The proposal “Treat replanning as the active hypothesis” was stale and was "
        "withdrawn without applying changes."
    )
    stored = service.history.load_patches()[-1]
    assert stored.transition is not None
    resolution_operation = next(
        operation for operation in stored.ops if isinstance(operation, ResolveProposalsOperation)
    )
    assert isinstance(resolution_operation, ResolveProposalsOperation)
    assert len(resolution_operation.resolutions) == 1
    resolution = resolution_operation.resolutions[0]
    assert resolution.id == "prop/activate-replanning-hypothesis"
    assert resolution.status == "withdrawn"
    assert resolution.reason == withdrawal_reason
    if same_draft:
        assert stored.change_summary == [
            "Removed “Replanning restores plasticity”.",
            withdrawal_reason,
        ]
        assert len(stored.transition.initiating_groups) == 2


def test_graph_sync_refuses_removing_an_accepted_node(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    service.review_node(
        "rq/learning-after-shift",
        ReviewRequest(standing="accepted"),
        authorized_by=authorized_human(app),
    )
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 3,
            "removed_node_ids": ["rq/learning-after-shift"],
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Accepted node rq/learning-after-shift cannot be removed; withdraw its acceptance "
        "and Sync before removing it."
    )
    accepted = service.history.state().nodes["rq/learning-after-shift"]
    combined = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 3,
            "nodes": [
                {
                    "node_id": accepted.id,
                    "base_updated_rev": accepted.updated_rev,
                    "standing": "asserted",
                }
            ],
            "removed_node_ids": [accepted.id],
        },
    )
    assert combined.status_code == 422
    assert "cannot both change and remove the same node" in combined.text
    assert service.history.state().revision == 3


def test_graph_sync_request_rejects_duplicate_and_conflicting_removals() -> None:
    with pytest.raises(ValueError, match="duplicate removed node targets"):
        GraphSyncRequest(
            base_revision=1,
            removed_node_ids=["hyp/one", "hyp/one"],
        )

    with pytest.raises(ValueError, match="both change and remove the same node: hyp/one"):
        GraphSyncRequest(
            base_revision=1,
            nodes=[
                GraphSyncNodeChange(
                    node_id="hyp/one",
                    base_updated_rev=1,
                    standing="contested",
                )
            ],
            removed_node_ids=["hyp/one"],
        )


def test_graph_sync_route_passes_active_experiment_loop_to_removal_guard(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    initial_document = seed_patch().model_dump(mode="python")
    initial_document["ops"][0]["nodes"].append(
        {
            "id": "exp/active-loop",
            "type": "experiment",
            "title": "Active loop",
            "objective": "Exercise the bounded loop removal guard.",
            "status": "running",
        }
    )
    initial = Patch.model_validate(initial_document)
    append_fixture_patch(service, initial)
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    now = datetime.now(UTC).isoformat()
    episode_id = str(uuid.uuid4())
    request = RunRequest(
        provider="codex",
        run_on="laptop",
        chat_id=str(uuid.uuid4()),
        chat_scope="node",
        node_id="exp/active-loop",
        mode="work",
        trigger="experiment_run",
        patch_kind="experiment_loop",
        control_node_id="exp/active-loop",
        control_revision=1,
        control_episode_id=episode_id,
        control_invocation=1,
        control_invocation_ceiling=5,
        control_decision_bundle=[],
        control_completion_criteria=[],
    )
    store.create_experiment_episode_with_invocation(
        AgentTaskRecord(
            operation_id="active-experiment-loop",
            project_id=project_id,
            episode_id=episode_id,
            kind="node_chat",
            status="queued",
            request=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            status_message="Queued bounded experiment loop.",
            authorized_by=authorized_human(store),
        )
    )
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{project_id}/sync",
        json={"base_revision": 2, "removed_node_ids": ["exp/active-loop"]},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Experiment exp/active-loop cannot be removed while its bounded experiment loop is active."
    )
    assert service.history.state().revision == 2


def test_graph_sync_commits_ontology_as_human_approval_patch(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={"base_revision": 2, "ontology": ontology_payload()},
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 3
    assert response.json()["ontology"] == ontology_payload()
    stored = service.history.load_patches()[-1]
    assert stored.kind == "approval"
    assert stored.author == "human"
    assert len(stored.ops) == 1
    ontology_operation = stored.ops[0]
    assert isinstance(ontology_operation, SetOntologyOperation)
    assert ontology_operation.ontology.model_dump(mode="python") == ontology_payload()


def test_graph_sync_unchanged_ontology_writes_no_patch(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)
    project_id = app.state.default_project_id
    assert (
        client.post(
            f"/api/projects/{project_id}/sync",
            json={"base_revision": 2, "ontology": ontology_payload()},
        ).status_code
        == 200
    )

    response = client.post(
        f"/api/projects/{project_id}/sync",
        json={"base_revision": 3, "ontology": ontology_payload()},
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 3
    assert len(service.history.load_patches()) == 3


def test_graph_sync_refuses_stale_ontology_draft(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    service.review_node(
        "rq/learning-after-shift",
        ReviewRequest(standing="accepted"),
        authorized_by=authorized_human(app),
    )
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={"base_revision": 2, "ontology": ontology_payload()},
    )

    assert response.status_code == 409
    assert "graph changed" in response.json()["detail"].lower()


def test_graph_sync_refuses_defining_and_using_a_type_in_one_draft(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 2,
            "ontology": ontology_payload(),
            "custom_nodes": [custom_hypothesis_payload()],
        },
    )

    assert response.status_code == 422
    assert "defines and uses a new ontology type" in response.json()["detail"]
    assert "sync the ontology first" in response.json()["detail"].lower()
    assert service.history.state().revision == 2


def test_graph_sync_does_not_offer_direct_base_node_authoring(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 2,
            "custom_nodes": [
                {
                    "id": "hyp/human-base-node",
                    "type": "hypothesis",
                    "title": "Human base node",
                    "statement": "This must not create a new base node.",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert "base-node authoring is not available" in response.json()["detail"]
    assert service.history.state().revision == 2


def test_graph_sync_creates_an_asserted_node_of_an_active_custom_type(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)
    project_id = app.state.default_project_id
    assert (
        client.post(
            f"/api/projects/{project_id}/sync",
            json={"base_revision": 2, "ontology": ontology_payload()},
        ).status_code
        == 200
    )

    node = custom_hypothesis_payload()
    node["standing"] = "accepted"
    response = client.post(
        f"/api/projects/{project_id}/sync",
        json={"base_revision": 3, "custom_nodes": [node]},
    )

    assert response.status_code == 200
    created = response.json()["nodes"]["mechanism_hypothesis/custom-mechanism"]
    assert created["extension_type"] == "mechanism_hypothesis"
    assert created["extension_fields"] == {
        "mechanism": "Replanning restores unused update directions."
    }
    assert created["standing"] == "asserted"
    stored = service.history.load_patches()[-1]
    assert stored.kind == "approval"
    assert isinstance(stored.ops[0], CreateNodesOperation)


def test_graph_sync_replaces_active_extension_fields_on_an_existing_custom_node(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)
    project_id = app.state.default_project_id
    assert (
        client.post(
            f"/api/projects/{project_id}/sync",
            json={"base_revision": 2, "ontology": ontology_payload()},
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/api/projects/{project_id}/sync",
            json={"base_revision": 3, "custom_nodes": [custom_hypothesis_payload()]},
        ).status_code
        == 200
    )

    response = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": 4,
            "nodes": [
                {
                    "node_id": "mechanism_hypothesis/custom-mechanism",
                    "base_updated_rev": 4,
                    "changes": {
                        "extension_fields": {"mechanism": "Replanning refreshes update directions."}
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["nodes"]["mechanism_hypothesis/custom-mechanism"][
        "extension_fields"
    ] == {"mechanism": "Replanning refreshes update directions."}


def test_graph_sync_preserves_an_unchanged_deprecated_extension_field(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)
    project_id = app.state.default_project_id
    ontology = ontology_payload()
    ontology["fields"].append(
        {
            "owner_type": "mechanism_hypothesis",
            "name": "legacy_note",
            "definition": "A field retained for old nodes.",
            "kind": "text",
            "required": False,
            "agent_writable": True,
            "deprecated": False,
        }
    )
    assert (
        client.post(
            f"/api/projects/{project_id}/sync",
            json={"base_revision": 2, "ontology": ontology},
        ).status_code
        == 200
    )
    node = custom_hypothesis_payload()
    node["extension_fields"]["legacy_note"] = "Keep this old value."
    assert (
        client.post(
            f"/api/projects/{project_id}/sync",
            json={"base_revision": 3, "custom_nodes": [node]},
        ).status_code
        == 200
    )
    ontology["fields"][1]["deprecated"] = True
    assert (
        client.post(
            f"/api/projects/{project_id}/sync",
            json={"base_revision": 4, "ontology": ontology},
        ).status_code
        == 200
    )

    omitted = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": 5,
            "nodes": [
                {
                    "node_id": "mechanism_hypothesis/custom-mechanism",
                    "base_updated_rev": 4,
                    "changes": {
                        "extension_fields": {"mechanism": "Replanning refreshes update directions."}
                    },
                }
            ],
        },
    )
    assert omitted.status_code == 422
    assert "legacy_note" in omitted.json()["detail"]

    response = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": 5,
            "nodes": [
                {
                    "node_id": "mechanism_hypothesis/custom-mechanism",
                    "base_updated_rev": 4,
                    "changes": {
                        "extension_fields": {
                            "mechanism": "Replanning refreshes update directions.",
                            "legacy_note": "Keep this old value.",
                        }
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["nodes"]["mechanism_hypothesis/custom-mechanism"][
        "extension_fields"
    ] == {
        "mechanism": "Replanning refreshes update directions.",
        "legacy_note": "Keep this old value.",
    }


def test_batch_overwrites_forged_admission_receipts(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    forged = ValidationMessage(level="reject", code="forged", message="forged")
    raw = Patch(
        kind="approval",
        author="human",
        summary="Agree with the question.",
        ops=[
            {
                "op": "set_standing",
                "node_id": "rq/learning-after-shift",
                "standing": "accepted",
            }
        ],
        admission="rejected",
        admission_messages=[forged],
    )

    prepared, result = history.append_batch([raw], expected_revision=1)

    assert result.state.revision == 2
    assert prepared[0].admission == "accepted"
    assert not prepared[0].admission_messages
    stored = history.load_patches()[-1]
    assert stored.admission == "accepted"
    assert not stored.admission_messages


def test_batch_commits_one_patch_then_replays_canonical_history_for_outputs(
    manifest, monkeypatch
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    raw = Patch(
        kind="approval",
        author="human",
        summary="Agree with the question.",
        ops=[
            {
                "op": "set_standing",
                "node_id": "rq/learning-after-shift",
                "standing": "accepted",
            }
        ],
    )
    calls: list[tuple[bool, bool]] = []
    materialize = history.materialize

    def counted_materialize(*, write_outputs=True, pending_patch_paths=None):
        calls.append((write_outputs, pending_patch_paths is not None))
        return materialize(
            write_outputs=write_outputs,
            pending_patch_paths=pending_patch_paths,
        )

    monkeypatch.setattr(history, "materialize", counted_materialize)

    prepared, result = history.append_batch([raw], expected_revision=1)

    assert [patch.revision for patch in prepared] == [2]
    assert result.state.revision == 2
    assert calls == [(False, False), (True, False)]
    stored = json.loads((manifest.research_dir / "graph.json").read_text(encoding="utf-8"))
    assert stored["revision"] == 2


def test_batch_builder_receives_fresh_state_under_append_lock(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    lock_attempts: list[bool] = []

    def build(state):
        assert state.revision == 1

        def contend_for_lock() -> None:
            acquired = history._process_lock.acquire(timeout=0.05)
            lock_attempts.append(acquired)
            if acquired:
                history._process_lock.release()

        contender = threading.Thread(target=contend_for_lock)
        contender.start()
        contender.join()
        return [
            Patch(
                kind="approval",
                author="human",
                summary="Agree with the question.",
                ops=[
                    {
                        "op": "set_standing",
                        "node_id": "rq/learning-after-shift",
                        "standing": "accepted",
                    }
                ],
            )
        ]

    prepared, result = history.append_batch_from_state(build, expected_revision=1)

    assert lock_attempts == [False]
    assert [patch.revision for patch in prepared] == [2]
    assert result.state.nodes["rq/learning-after-shift"].standing == "accepted"


def test_graph_sync_refuses_stale_project_draft(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    node = service.history.state().nodes["rq/learning-after-shift"]
    service.review_node(
        node.id,
        ReviewRequest(standing="accepted"),
        authorized_by=authorized_human(app),
    )
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{app.state.default_project_id}/sync",
        json={
            "base_revision": 2,
            "nodes": [
                {
                    "node_id": node.id,
                    "base_updated_rev": node.updated_rev,
                    "standing": "contested",
                }
            ],
        },
    )

    assert response.status_code == 409
    assert "graph changed" in response.json()["detail"].lower()


def test_interrupted_transition_patch_write_exposes_none_of_the_sync(manifest, monkeypatch) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    before_graph = (manifest.research_dir / "graph.json").read_bytes()
    patches = [
        Patch(
            kind="approval",
            author="human",
            summary="Agree with the question.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": "rq/learning-after-shift",
                    "standing": "accepted",
                }
            ],
        ),
        Patch(
            kind="approval",
            author="human",
            summary="Disagree with the hypothesis.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": "hyp/replanning-restores-plasticity",
                    "standing": "contested",
                }
            ],
        ),
    ]
    original_atomic_text = history._atomic_text
    commit_target = manifest.research_dir / "patches" / "000002.json"

    def fail_atomic_transition_commit(path, content):
        if path == commit_target:
            raise OSError("simulated disk failure")
        original_atomic_text(path, content)

    monkeypatch.setattr(history, "_atomic_text", fail_atomic_transition_commit)

    with pytest.raises(OSError, match="simulated disk failure"):
        history.append_batch(patches, expected_revision=1)

    assert [patch.revision for patch in history.load_patches()] == [1]
    assert history.state().revision == 1
    assert (manifest.research_dir / "graph.json").read_bytes() == before_graph
    assert not commit_target.exists()
    assert not list((manifest.research_dir / "patches").glob(".batch-*"))
    assert not list((manifest.research_dir / "patches").glob("batch-*"))


def test_replay_ignores_an_uncommitted_hidden_batch(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    hidden = manifest.research_dir / "patches" / ".batch-interrupted"
    hidden.mkdir()
    (hidden / "000002.json").write_text(
        Patch(
            revision=2,
            kind="approval",
            author="human",
            summary="This transaction never committed.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": "rq/learning-after-shift",
                    "standing": "accepted",
                }
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )

    assert [patch.revision for patch in history.load_patches()] == [1]
    assert history.state().nodes["rq/learning-after-shift"].standing == "asserted"


def decision_ontology_payload() -> dict[str, object]:
    return {
        "types": [
            {
                "name": "policy_decision",
                "definition": "A decision about project policy.",
                "base_type": "decision",
                "layer": "action",
                "deprecated": False,
            }
        ],
        "fields": [],
        "relations": [],
    }


def test_graph_sync_refuses_creating_an_already_decided_decision(manifest, tmp_path) -> None:
    """Creation must not be a second way to write a Decision outcome.

    `selected_option` and `status="decided"` belong to the human decision_choice
    action, which checks the selection against the node's own options. A custom
    node carrying them at creation would skip that check entirely.
    """

    app = create_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    client = TestClient(app)
    project = app.state.default_project_id

    ontology = client.post(
        f"/api/projects/{project}/sync",
        json={"base_revision": 2, "ontology": decision_ontology_payload()},
    )
    assert ontology.status_code == 200, ontology.text

    response = client.post(
        f"/api/projects/{project}/sync",
        json={
            "base_revision": service.history.state().revision,
            "custom_nodes": [
                {
                    "id": "policy_decision/pre-decided",
                    "type": "decision",
                    "extension_type": "policy_decision",
                    "title": "Pre-decided",
                    "question": "Which policy?",
                    "options": ["a", "b"],
                    "selected_option": "not-an-option",
                    "status": "decided",
                }
            ],
        },
    )

    assert response.status_code == 422, response.text
    assert any(
        item["code"] == "agent-created-decision-action" and "selected_option" in item["message"]
        for item in response.json()["detail"]
    )
    assert "policy_decision/pre-decided" not in service.history.state().nodes
