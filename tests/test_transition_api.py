from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rcp.core.models import Patch

from .helpers import append_fixture_patch, create_named_app, seed_patch


def _seeded_api(manifest, tmp_path: Path):
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    append_fixture_patch(app.state.service, seed_patch())
    return app, TestClient(app), app.state.default_project_id


def _decision_ontology() -> dict[str, object]:
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


def test_transition_manifest_names_the_ruleset_and_conservative_sync_triggers(
    manifest, tmp_path: Path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)

    response = client.get(f"/api/projects/{app.state.default_project_id}/transition-manifest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ruleset_tag"] == "rcp.lifecycle.v2"
    triggers = {item["operation"]: item for item in payload["triggers"]}
    assert {
        "update_nodes",
        "create_edges",
        "remove_edges",
        "create_proposals",
        "resolve_proposals",
        "withdraw_proposals",
    } <= set(triggers)
    assert {
        "status",
        "selected_option",
        "current_summary",
        "next_action",
    } <= set(triggers["update_nodes"]["node_fields"])
    assert {"blocked_by", "governed_by", "tests"} <= set(triggers["create_edges"]["relations"])


def test_sync_preview_returns_the_complete_noncanonical_candidate_without_writing(
    manifest, tmp_path: Path
) -> None:
    app, client, project_id = _seeded_api(manifest, tmp_path)
    service = app.state.service
    before = service.history.state()
    question = before.nodes["rq/learning-after-shift"]
    patch_bytes = {
        path.name: path.read_bytes() for path in service.history.patches_dir.glob("*.json")
    }
    graph_bytes = (manifest.research_dir / "graph.json").read_bytes()

    response = client.post(
        f"/api/projects/{project_id}/sync/preview",
        json={
            "base_revision": before.revision,
            "nodes": [
                {
                    "node_id": question.id,
                    "base_updated_rev": question.updated_rev,
                    "changes": {"title": "Learning after a distribution shift"},
                    "standing": "accepted",
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    projection = payload["projection"]
    transition = payload["transition"]
    candidate = projection["graph"]
    assert projection["canonical"] is False
    assert projection["ruleset_tag"] == transition["ruleset_tag"] == "rcp.lifecycle.v2"
    assert transition["pre_head"]["revision"] == before.revision
    assert projection["head"]["revision"] == candidate["revision"] == before.revision + 1
    assert (
        projection["head"]["transition_id"]
        == projection["transition_id"]
        == transition["transition_id"]
    )
    assert candidate["nodes"][question.id]["title"] == "Learning after a distribution shift"
    assert candidate["nodes"][question.id]["standing"] == "accepted"
    assert projection["experiment_control"] == {}
    assert projection["guidance_validity"] == {}
    assert transition["initiating_groups"][0]["operation_indexes"] == [0, 1]

    assert service.history.state() == before
    assert {
        path.name: path.read_bytes() for path in service.history.patches_dir.glob("*.json")
    } == patch_bytes
    assert (manifest.research_dir / "graph.json").read_bytes() == graph_bytes


def test_sync_preview_and_commit_publish_the_same_graph_attention_membership(
    manifest, tmp_path: Path
) -> None:
    app, client, project_id = _seeded_api(manifest, tmp_path)
    service = app.state.service
    append_fixture_patch(
        service,
        Patch(
            kind="refresh",
            author="agent",
            summary="Added graph-attention fixtures.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "dec/attention",
                            "type": "decision",
                            "title": "Choose a policy",
                            "question": "Which policy should be used?",
                            "options": ["first", "second"],
                            "status": "ready",
                        },
                        {
                            "id": "blk/attention",
                            "type": "blocker",
                            "title": "Needs judgment",
                            "description": "A human must judge this blocker.",
                            "status": "open",
                            "standing": "asserted",
                        },
                    ],
                },
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": "prop/attention",
                            "title": "Review the hypothesis",
                            "card": {"decision_needed": "Choose whether to apply it."},
                            "ops": [
                                {
                                    "op": "update_nodes",
                                    "intent": "content_change",
                                    "nodes": [
                                        {
                                            "id": "hyp/replanning-restores-plasticity",
                                            "changes": {
                                                "statement": "The hypothesis needs review."
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ],
        ),
    )
    before = service.history.state()
    question = before.nodes["rq/learning-after-shift"]
    body = {
        "base_revision": before.revision,
        "nodes": [
            {
                "node_id": question.id,
                "base_updated_rev": question.updated_rev,
                "changes": {"title": "Learning after attention review"},
            }
        ],
    }

    preview = client.post(f"/api/projects/{project_id}/sync/preview", json=body)
    committed = client.post(f"/api/projects/{project_id}/sync", json=body)

    assert preview.status_code == 200, preview.text
    assert committed.status_code == 200, committed.text
    expected = {
        "pending_proposal_ids": ["prop/attention"],
        "decisions_awaiting_choice_ids": ["dec/attention"],
        "open_blocker_ids": ["blk/attention"],
    }
    assert preview.json()["projection"]["attention"] == expected
    assert committed.json()["attention"] == expected


def test_experiment_control_publishes_whether_the_human_closed_the_node(
    manifest, tmp_path: Path
) -> None:
    app, client, project_id = _seeded_api(manifest, tmp_path)
    service = app.state.service
    append_fixture_patch(
        service,
        Patch(
            kind="refresh",
            author="agent",
            summary="Added an open and a closed Experiment.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "exp/open",
                            "type": "experiment",
                            "title": "Open experiment",
                            "objective": "Stay open.",
                            "status": "running",
                        },
                        {
                            "id": "exp/closed",
                            "type": "experiment",
                            "title": "Closed experiment",
                            "objective": "Be finished with.",
                            "status": "abandoned",
                        },
                    ],
                }
            ],
        ),
    )

    snapshot = client.get(f"/api/projects/{project_id}")

    assert snapshot.status_code == 200, snapshot.text
    controls = snapshot.json()["experiment_control"]
    # Overview reads this instead of restating the closed-status vocabulary.
    assert {node_id: control["node_closed"] for node_id, control in controls.items()} == {
        "exp/open": False,
        "exp/closed": True,
    }


def test_sync_response_is_one_coherent_graph_control_and_head_projection(
    manifest, tmp_path: Path
) -> None:
    app, client, project_id = _seeded_api(manifest, tmp_path)
    service = app.state.service
    append_fixture_patch(
        service,
        Patch(
            kind="refresh",
            author="agent",
            summary="Recorded an experiment.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "exp/adaptation",
                            "type": "experiment",
                            "title": "Adaptation experiment",
                            "objective": "Measure learning after a distribution shift.",
                            "current_summary": "The design is ready for review.",
                            "next_action": "Review the intervention matrix.",
                        }
                    ],
                }
            ],
        ),
    )
    before = service.history.state()
    experiment = before.nodes["exp/adaptation"]

    response = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": before.revision,
            "nodes": [
                {
                    "node_id": experiment.id,
                    "base_updated_rev": experiment.updated_rev,
                    "changes": {"current_summary": "The design review is complete."},
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["revision"] == payload["graph"]["revision"] == payload["head"]["revision"]
    assert payload["nodes"] == payload["graph"]["nodes"]
    assert payload["edges"] == payload["graph"]["edges"]
    assert payload["head"]["transition_id"] == payload["transition_id"]
    assert payload["canonical"] is True
    assert payload["ruleset_tag"] == "rcp.lifecycle.v2"
    assert set(payload["experiment_control"]) == {experiment.id}
    assert set(payload["guidance_validity"]) == {experiment.id}
    assert payload["graph"]["nodes"][experiment.id]["current_summary"] == (
        "The design review is complete."
    )


def test_empty_sync_response_preserves_the_exact_canonical_transition_head(
    manifest, tmp_path: Path
) -> None:
    app, client, project_id = _seeded_api(manifest, tmp_path)
    service = app.state.service
    before = service.history.state()
    question = before.nodes["rq/learning-after-shift"]
    committed = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": before.revision,
            "nodes": [
                {
                    "node_id": question.id,
                    "base_updated_rev": question.updated_rev,
                    "changes": {"title": "Learning after a shifted distribution"},
                }
            ],
        },
    )
    assert committed.status_code == 200, committed.text
    exact_head = committed.json()["head"]
    assert exact_head["transition_id"] is not None

    empty = client.post(
        f"/api/projects/{project_id}/sync",
        json={"base_revision": exact_head["revision"]},
    )

    assert empty.status_code == 200, empty.text
    payload = empty.json()
    assert payload["head"] == exact_head
    assert payload["transition_id"] == exact_head["transition_id"]
    assert payload["graph"]["revision"] == exact_head["revision"]


def test_sync_with_several_staged_actions_spends_exactly_one_revision(
    manifest, tmp_path: Path
) -> None:
    app, client, project_id = _seeded_api(manifest, tmp_path)
    service = app.state.service
    before = service.history.state()
    question = before.nodes["rq/learning-after-shift"]
    hypothesis = before.nodes["hyp/replanning-restores-plasticity"]
    before_patch_count = len(service.history.load_patches())

    response = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": before.revision,
            "nodes": [
                {
                    "node_id": question.id,
                    "base_updated_rev": question.updated_rev,
                    "changes": {"title": "Learning after a task distribution shift"},
                    "standing": "accepted",
                },
                {
                    "node_id": hypothesis.id,
                    "base_updated_rev": hypothesis.updated_rev,
                    "changes": {"title": "Replanning preserves plasticity"},
                    "standing": "contested",
                },
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["revision"] == before.revision + 1
    patches = service.history.load_patches()
    assert len(patches) == before_patch_count + 1
    committed = patches[-1]
    assert committed.revision == before.revision + 1
    assert committed.transition is not None
    assert len(committed.transition.initiating_groups) == 2
    assert len(committed.ops) == 4


def test_sync_preview_rejects_a_stale_head_without_writing(manifest, tmp_path: Path) -> None:
    app, client, project_id = _seeded_api(manifest, tmp_path)
    service = app.state.service
    before = service.history.state()
    question = before.nodes["rq/learning-after-shift"]
    committed = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": before.revision,
            "nodes": [
                {
                    "node_id": question.id,
                    "base_updated_rev": question.updated_rev,
                    "standing": "accepted",
                }
            ],
        },
    )
    assert committed.status_code == 200, committed.text
    committed_patch_count = len(service.history.load_patches())

    response = client.post(
        f"/api/projects/{project_id}/sync/preview",
        json={
            "base_revision": before.revision,
            "nodes": [
                {
                    "node_id": question.id,
                    "base_updated_rev": question.updated_rev,
                    "changes": {"title": "A stale edit"},
                }
            ],
        },
    )

    assert response.status_code == 409
    assert "moved from revision" in response.json()["detail"]
    assert len(service.history.load_patches()) == committed_patch_count


def test_invalid_sync_preview_returns_structured_transition_diagnostics(
    manifest, tmp_path: Path
) -> None:
    app, client, project_id = _seeded_api(manifest, tmp_path)
    service = app.state.service
    ontology = client.post(
        f"/api/projects/{project_id}/sync",
        json={
            "base_revision": service.history.state().revision,
            "ontology": _decision_ontology(),
        },
    )
    assert ontology.status_code == 200, ontology.text
    before = service.history.state()
    before_patch_count = len(service.history.load_patches())

    response = client.post(
        f"/api/projects/{project_id}/sync/preview",
        json={
            "base_revision": before.revision,
            "custom_nodes": [
                {
                    "id": "policy_decision/pre-decided",
                    "type": "decision",
                    "extension_type": "policy_decision",
                    "title": "Pre-decided policy",
                    "question": "Which policy?",
                    "options": ["a", "b"],
                    "selected_option": "not-an-option",
                    "status": "decided",
                }
            ],
        },
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
    assert {
        "level",
        "code",
        "message",
        "patch_revision",
        "related_node_ids",
        "operation_index",
        "rule_id",
        "cause_chain",
        "failed_invariant",
    } <= set(detail[0])
    assert any(item["code"] == "agent-created-decision-action" for item in detail)
    assert all(item["patch_revision"] == before.revision + 1 for item in detail)
    assert len(service.history.load_patches()) == before_patch_count
    assert "policy_decision/pre-decided" not in service.history.state().nodes
