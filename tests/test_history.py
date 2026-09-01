from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import pytest
from pydantic import ValidationError

import rcp.config as config_module
from rcp.config import load_manifest
from rcp.core.models import GraphState, Patch, ValidationMessage
from rcp.core.operations import operation_dict
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.core.transitions import GraphTransitionManager
from rcp.history import HistoryManager, PatchRejected, ReplayHalted, RevisionConflict
from rcp.history.manager import ProjectIdentityConflict
from tests.helpers import refresh_patch, seed_patch, shape_invalid_patch


def _remove_nodes_patch(
    *node_ids: str,
    kind: str = "refresh",
    author: str = "agent",
) -> Patch:
    return Patch.model_validate(
        {
            "kind": kind,
            "author": author,
            "summary": "Removed nodes from the current graph.",
            "run_truth_scope": ["repo-a"] if author == "agent" else [],
            "repositories_read": ["repo-a"] if author == "agent" else [],
            "ops": [{"op": "remove_nodes", "node_ids": list(node_ids)}],
        }
    )


def _remove_edges_patch(operation: dict[str, object]) -> Patch:
    return Patch.model_validate(
        {
            "kind": "refresh",
            "author": "agent",
            "summary": "Removed edges from the current graph.",
            "run_truth_scope": ["repo-a"],
            "repositories_read": ["repo-a"],
            "ops": [operation],
        }
    )


def _record_experiment(history: HistoryManager, attempt_status: str | None = None) -> str:
    experiment_id = "exp/bounded-loop"
    node: dict[str, object] = {
        "id": experiment_id,
        "type": "experiment",
        "title": "Bounded loop",
        "objective": "Measure future plasticity.",
    }
    if attempt_status is not None:
        node["attempts"] = [
            {
                "id": "attempt-1",
                "sequence": 1,
                "purpose": "Run the matched comparison.",
                "status": attempt_status,
            }
        ]
    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Recorded a bounded experiment.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[{"op": "create_nodes", "nodes": [node]}],
        )
    )
    return experiment_id


def test_manifest_writes_share_the_append_lock_across_manager_instances(
    manifest, monkeypatch
) -> None:
    initial = HistoryManager(manifest)
    initial.append(seed_patch())
    settings_history = HistoryManager(load_manifest(manifest.path))
    scope_history = HistoryManager(load_manifest(manifest.path))
    assert settings_history._process_lock is scope_history._process_lock

    real_atomic_write = config_module._atomic_write
    writes_ready = threading.Barrier(2)

    def synchronize_manifest_writes(path, content) -> None:
        if path == manifest.path:
            with suppress(threading.BrokenBarrierError):
                writes_ready.wait(timeout=0.25)
        real_atomic_write(path, content)

    monkeypatch.setattr(config_module, "_atomic_write", synchronize_manifest_writes)
    calls_ready = threading.Barrier(3)

    def update_provider_path() -> None:
        calls_ready.wait()
        settings_history.update_machine_provider_paths({"laptop": {"codex": "/opt/agents/codex"}})

    def change_scope() -> None:
        calls_ready.wait()
        scope_history.append(
            Patch(
                kind="approval",
                author="human",
                summary="Removed repo-b from the project truth scope.",
                ops=[
                    {
                        "op": "set_project_truth_scope",
                        "truth_scope": ["repo-a"],
                    }
                ],
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        settings_future = executor.submit(update_provider_path)
        scope_future = executor.submit(change_scope)
        calls_ready.wait()
        settings_future.result(timeout=5)
        scope_future.result(timeout=5)

    updated = load_manifest(manifest.path)
    assert updated.project.truth_scope == ["repo-a"]
    assert updated.machine_map["laptop"].provider_paths["codex"] == "/opt/agents/codex"
    assert scope_history.state().revision == 2


def test_seed_is_asserted_and_accepted_core_starts_empty(manifest) -> None:
    history = HistoryManager(manifest)
    forged = seed_patch().model_copy(
        update={
            "admission": "rejected",
            "admission_messages": [
                ValidationMessage(level="reject", code="forged", message="forged")
            ],
        }
    )
    patch, result = history.append(forged)

    assert patch.revision == 1
    assert patch.admission == "accepted"
    assert not patch.admission_messages
    assert history.load_patches()[0].admission == "accepted"
    assert result.state.revision == 1
    assert {node.standing for node in result.state.nodes.values()} == {"asserted"}
    assert (manifest.research_dir / "research.md").read_text(encoding="utf-8") == ""
    assert result.state.coverage.repositories_seen == []
    assert result.state.last_refresh_at == patch.created_at
    assert result.state.coverage.repositories_never_seen == ["repo-a", "repo-b"]


def test_successful_patch_materializes_processed_cursors(manifest) -> None:
    history = HistoryManager(manifest)
    session_key = "repo-a/laptop/codex/session-1"

    history.append(seed_patch().model_copy(update={"processed_cursors": {session_key: "record-2"}}))

    cursors = json.loads((manifest.research_dir / "cursors.json").read_text(encoding="utf-8"))
    assert cursors == {session_key: "record-2"}


def test_standalone_review_generates_research_md(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Accepted the primary question.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": "rq/learning-after-shift",
                    "standing": "accepted",
                }
            ],
        )
    )

    research = (manifest.research_dir / "research.md").read_text(encoding="utf-8")
    assert "Learning after task shift" in research
    assert "Replanning restores plasticity" not in research


@pytest.mark.parametrize("standing", ["asserted", "contested"])
def test_agent_removes_asserted_or_contested_ordinary_node_and_incident_edges(
    manifest, standing: str
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    node_id = "blk/missing-capacity"
    experiment_id = "exp/capacity-check"
    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Recorded an ordinary blocker and its dependent experiment.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": node_id,
                            "type": "blocker",
                            "title": "Missing capacity",
                            "description": "The experiment is waiting for capacity.",
                        },
                        {
                            "id": experiment_id,
                            "type": "experiment",
                            "title": "Capacity check",
                            "objective": "Measure the available capacity.",
                        },
                    ],
                },
                {
                    "op": "create_edges",
                    "edges": [
                        {
                            "source": experiment_id,
                            "target": node_id,
                            "relation": "blocked_by",
                        }
                    ],
                },
            ],
        )
    )
    if standing == "contested":
        history.append(
            Patch(
                kind="approval",
                author="human",
                summary="Contested the hypothesis before removal.",
                ops=[{"op": "set_standing", "node_id": node_id, "standing": "contested"}],
            )
        )

    history.append(_remove_nodes_patch(node_id))

    state = history.state()
    assert node_id not in state.nodes
    assert experiment_id in state.nodes
    assert all(edge.source != node_id and edge.target != node_id for edge in state.edges.values())


def test_direct_human_remove_nodes_is_a_valid_standalone_approval(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())

    history.append(
        _remove_nodes_patch(
            "hyp/replanning-restores-plasticity",
            kind="approval",
            author="human",
        )
    )

    assert "hyp/replanning-restores-plasticity" not in history.state().nodes


def test_accepted_target_rejects_the_entire_remove_nodes_operation(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    accepted_id = "hyp/replanning-restores-plasticity"
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Accepted the hypothesis.",
            ops=[{"op": "set_standing", "node_id": accepted_id, "standing": "accepted"}],
        )
    )

    with pytest.raises(PatchRejected) as caught:
        history.append(_remove_nodes_patch("rq/learning-after-shift", accepted_id))

    assert any(message.code == "accepted-node-removal" for message in caught.value.report.messages)
    state = history.state()
    assert {"rq/learning-after-shift", accepted_id} <= set(state.nodes)
    assert any(
        edge.source == accepted_id or edge.target == accepted_id for edge in state.edges.values()
    )


def test_standing_change_cannot_bypass_accepted_node_removal(manifest) -> None:
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

    with pytest.raises(PatchRejected) as caught:
        history.append(
            Patch(
                kind="approval",
                author="human",
                summary="Tried to clear and remove in one approval patch.",
                ops=[
                    {"op": "set_standing", "node_id": node_id, "standing": "asserted"},
                    {"op": "remove_nodes", "node_ids": [node_id]},
                ],
            )
        )

    codes = {message.code for message in caught.value.report.messages}
    assert {"invalid-standalone-review", "accepted-node-removal"} <= codes
    assert history.state().nodes[node_id].standing == "accepted"


def test_remove_nodes_rejects_unknown_target(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())

    with pytest.raises(PatchRejected) as caught:
        history.append(_remove_nodes_patch("rq/missing"))

    assert any(message.code == "unknown-node" for message in caught.value.report.messages)


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "remove_edges", "edge_ids": ["edge/one"], "unexpected": True},
        {"op": "remove_edges", "edge_ids": "edge/one"},
        {"op": "remove_edges", "edge_ids": [1]},
    ],
)
def test_malformed_remove_edges_is_rejected_before_history_admission(
    manifest, operation: dict[str, object]
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())

    with pytest.raises(ValidationError):
        _remove_edges_patch(operation)

    assert history.state().revision == 1
    assert history.state().replay_status == "complete"


@pytest.mark.parametrize("attempt_status", ["planned", "submitted", "running"])
def test_remove_nodes_refuses_experiment_with_active_attempt(manifest, attempt_status: str) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    experiment_id = _record_experiment(history, attempt_status)

    with pytest.raises(PatchRejected) as caught:
        history.append(_remove_nodes_patch(experiment_id))

    assert any(
        message.code == "active-experiment-removal" for message in caught.value.report.messages
    )
    assert experiment_id in history.state().nodes


def test_update_to_active_attempt_cannot_bypass_experiment_removal(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    experiment_id = _record_experiment(history)
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Tried to start and remove an Experiment in one patch.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": experiment_id,
                        "changes": {
                            "attempts": [
                                {
                                    "id": "attempt-1",
                                    "sequence": 1,
                                    "purpose": "Run the matched comparison.",
                                    "status": "planned",
                                }
                            ]
                        },
                    }
                ],
            },
            {"op": "remove_nodes", "node_ids": [experiment_id]},
        ],
    )

    with pytest.raises(PatchRejected) as caught:
        history.append(patch)

    assert any(
        message.code == "active-experiment-removal" for message in caught.value.report.messages
    )
    experiment = history.state().nodes[experiment_id]
    assert experiment.attempts == []


def test_experiment_loop_patch_cannot_remove_its_control_node(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    experiment_id = _record_experiment(history)
    patch = _remove_nodes_patch(experiment_id, kind="experiment_loop").model_copy(
        update={"experiment_control_node_id": experiment_id}
    )

    with pytest.raises(PatchRejected) as caught:
        history.append(patch)

    assert any(
        message.code == "experiment-loop-operation" for message in caught.value.report.messages
    )
    assert experiment_id in history.state().nodes


@pytest.mark.parametrize("standing", ["asserted", "accepted", "contested"])
def test_direct_human_prose_edit_preserves_node_standing(manifest, standing) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    if standing != "asserted":
        history.append(
            Patch(
                kind="approval",
                author="human",
                summary=f"Marked hypothesis {standing}.",
                ops=[
                    {
                        "op": "set_standing",
                        "node_id": "hyp/replanning-restores-plasticity",
                        "standing": standing,
                    }
                ],
            )
        )
    before = history.state().nodes["hyp/replanning-restores-plasticity"]

    patch, result = history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Clarified the hypothesis wording.",
            ops=[
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": before.id,
                            "base_updated_rev": before.updated_rev,
                            "changes": {
                                "title": "Search-time replanning may preserve future learning",
                                "statement": (
                                    "Replanning during search may help the learner remain able "
                                    "to adapt after its task changes."
                                ),
                            },
                        }
                    ],
                }
            ],
        )
    )

    edited = result.state.nodes[before.id]
    assert edited.title == "Search-time replanning may preserve future learning"
    assert edited.standing.value == standing
    assert edited.updated_rev == patch.revision
    assert [operation_dict(operation) for operation in patch.ops] == [
        {
            "op": "update_nodes",
            "nodes": [
                {
                    "id": before.id,
                    "base_updated_rev": before.updated_rev,
                    "changes": {
                        "title": "Search-time replanning may preserve future learning",
                        "statement": (
                            "Replanning during search may help the learner remain able "
                            "to adapt after its task changes."
                        ),
                    },
                }
            ],
        }
    ]


@pytest.mark.parametrize("field", ["status", "source_refs", "standing"])
def test_direct_human_edit_rejects_non_prose_fields(manifest, field) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    node = history.state().nodes["hyp/replanning-restores-plasticity"]
    value = {
        "status": "active",
        "source_refs": [],
        "standing": "accepted",
    }[field]

    with pytest.raises(PatchRejected) as caught:
        history.append(
            Patch(
                kind="approval",
                author="human",
                summary="Tried to bypass direct-edit boundaries.",
                ops=[
                    {
                        "op": "update_nodes",
                        "nodes": [
                            {
                                "id": node.id,
                                "base_updated_rev": node.updated_rev,
                                "changes": {field: value},
                            }
                        ],
                    }
                ],
            )
        )

    assert any(
        message.code in {"non-prose-node-edit", "immutable-node-field"}
        for message in caught.value.report.messages
    )
    assert history.state().nodes[node.id].model_dump() == node.model_dump()


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        (
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "rq/learning-after-shift",
                        "base_updated_rev": 1,
                        "changes": {"title": "A clearer question"},
                    },
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "base_updated_rev": 1,
                        "changes": {"title": "A clearer hypothesis"},
                    },
                ],
            },
            "invalid-direct-node-edit",
        ),
        (
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "changes": {"title": "Missing concurrency guard"},
                    }
                ],
            },
            "invalid-direct-node-edit",
        ),
        (
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "base_updated_rev": 0,
                        "changes": {"title": "Stale edit"},
                    }
                ],
            },
            "stale-node-edit",
        ),
        (
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "base_updated_rev": 1,
                        "changes": {"title": "Replanning restores plasticity"},
                    }
                ],
            },
            "empty-node-edit",
        ),
    ],
)
def test_malformed_direct_human_edit_shape_is_rejected(manifest, operation, code) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())

    with pytest.raises(PatchRejected) as caught:
        history.append(
            Patch(
                kind="approval",
                author="human",
                summary="Malformed direct edit.",
                ops=[operation],
            )
        )

    assert any(message.code == code for message in caught.value.report.messages)


def test_agent_cannot_apply_gated_hypothesis_transition(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    with pytest.raises(PatchRejected) as caught:
        history.append(
            Patch(
                kind="refresh",
                author="agent",
                summary="Changed hypothesis status directly.",
                run_truth_scope=["repo-a"],
                repositories_read=["repo-a"],
                ops=[
                    {
                        "op": "update_nodes",
                        "nodes": [
                            {
                                "id": "hyp/replanning-restores-plasticity",
                                "changes": {"status": "active"},
                            }
                        ],
                    }
                ],
            )
        )
    assert any(message.code == "graph-action-refused" for message in caught.value.report.messages)
    assert len(list((manifest.research_dir / "patches").glob("*.json"))) == 2


def test_agent_updates_blocker_lifecycle_directly_and_resets_accepted_standing(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(
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
                            "status": "open",
                        }
                    ],
                }
            ],
        )
    )
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Accepted the blocker.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": "blk/missing-capacity",
                    "standing": "accepted",
                }
            ],
        )
    )

    appended, result = history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Resolved the operational blocker.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "blk/missing-capacity",
                            "changes": {"status": "resolved"},
                        }
                    ],
                }
            ],
        )
    )

    blocker = result.state.nodes["blk/missing-capacity"]
    assert appended.admission == "accepted"
    assert blocker.status == "resolved"
    assert blocker.standing == "asserted"
    assert result.state.proposals == {}


@pytest.mark.parametrize(
    "changes",
    [
        {"standing": "accepted"},
        {"id": "hyp/renamed-behind-the-index"},
        {"type": "evidence"},
    ],
)
def test_node_updates_cannot_change_identity_or_standing(manifest, changes) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())

    with pytest.raises(PatchRejected) as caught:
        history.append(
            Patch(
                kind="refresh",
                author="agent",
                summary="Tried to change a system-owned node field.",
                run_truth_scope=["repo-a"],
                repositories_read=["repo-a"],
                ops=[
                    {
                        "op": "update_nodes",
                        "nodes": [
                            {
                                "id": "hyp/replanning-restores-plasticity",
                                "changes": changes,
                            }
                        ],
                    }
                ],
            )
        )

    assert any(message.code == "immutable-node-field" for message in caught.value.report.messages)
    assert len(list((manifest.research_dir / "patches").glob("*.json"))) == 2


@pytest.mark.parametrize(
    "operation",
    [
        (
            {
                "op": "supersede_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "superseded_by": "hyp/replanning-alternative",
                    }
                ],
            }
        ),
        (
            {
                "op": "merge_nodes",
                "merges": [
                    {
                        "duplicate": "hyp/replanning-restores-plasticity",
                        "canonical": "hyp/replanning-alternative",
                    }
                ],
            }
        ),
    ],
)
def test_agent_cannot_supersede_or_merge_a_hypothesis_directly(manifest, operation) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Recorded an alternative hypothesis.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "hyp/replanning-alternative",
                            "type": "hypothesis",
                            "title": "Replanning alternative",
                            "statement": "A separate mechanism explains recovery.",
                        }
                    ],
                }
            ],
        )
    )

    with pytest.raises(PatchRejected) as caught:
        history.append(
            Patch(
                kind="refresh",
                author="agent",
                summary="Tried to rewrite accepted graph identity.",
                run_truth_scope=["repo-a"],
                repositories_read=["repo-a"],
                ops=[operation],
            )
        )

    assert any(message.code == "graph-action-refused" for message in caught.value.report.messages)
    assert len(list((manifest.research_dir / "patches").glob("*.json"))) == 3


@pytest.mark.parametrize("operation", ["supersede_nodes", "merge_nodes"])
def test_agent_can_reconcile_accepted_nonbelief_nodes_directly(manifest, operation) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    duplicate_id = "blk/missing-capacity"
    canonical_id = "blk/canonical-capacity"
    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Recorded two capacity blockers.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": duplicate_id,
                            "type": "blocker",
                            "title": "Missing capacity",
                            "description": "The run needs more capacity.",
                        },
                        {
                            "id": canonical_id,
                            "type": "blocker",
                            "title": "Canonical capacity blocker",
                            "description": "The run is waiting for its assigned capacity.",
                        },
                    ],
                }
            ],
        )
    )
    history.append(
        Patch(
            kind="approval",
            author="human",
            summary="Accepted the capacity blocker.",
            ops=[
                {
                    "op": "set_standing",
                    "node_id": duplicate_id,
                    "standing": "accepted",
                }
            ],
        )
    )
    if operation == "supersede_nodes":
        op = {
            "op": operation,
            "nodes": [
                {
                    "id": duplicate_id,
                    "superseded_by": canonical_id,
                }
            ],
        }
    else:
        op = {
            "op": operation,
            "merges": [
                {
                    "duplicate": duplicate_id,
                    "canonical": canonical_id,
                }
            ],
        }

    history.append(
        Patch(
            kind="refresh",
            author="agent",
            summary="Reconciled duplicate capacity blockers.",
            run_truth_scope=["repo-a"],
            repositories_read=["repo-a"],
            ops=[op],
        )
    )

    node = history.state().nodes[duplicate_id]
    assert node.status == "superseded"
    assert node.standing == "asserted"


@pytest.mark.parametrize(
    "operation",
    [
        {
            "op": "resolve_ambiguities",
            "resolutions": [{"id": "amb/missing", "status": "resolved"}],
        },
        {
            "op": "resolve_proposals",
            "resolutions": [{"id": "prop/missing", "status": "withdrawn"}],
        },
    ],
)
def test_unknown_resolution_target_is_rejected_before_append(manifest, operation) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())

    with pytest.raises(PatchRejected):
        history.append(
            Patch(
                kind="refresh",
                author="agent",
                summary="Malformed resolution.",
                run_truth_scope=["repo-a"],
                repositories_read=["repo-a"],
                ops=[operation],
            )
        )

    assert len(list((manifest.research_dir / "patches").glob("*.json"))) == 2


def test_malformed_agent_patch_is_auditable_without_poisoning_replay(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    malformed = Patch(
        kind="refresh",
        author="agent",
        summary="Malformed relation.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/learning-after-shift",
                        "target": "hyp/replanning-restores-plasticity",
                        "relation": "not_a_relation",
                    }
                ],
            }
        ],
    )

    appended, result = history.append(malformed, raise_on_reject=False)

    assert result.reports[appended.revision].rejected is True
    assert any(
        message.code == "invalid-edge" for message in result.reports[appended.revision].messages
    )
    stored = history.load_patches()[-1]
    assert appended.admission == stored.admission == "rejected"
    assert [message.code for message in stored.admission_messages] == ["invalid-edge"]
    replayed = history.state()
    assert replayed.replay_status == "complete"
    assert replayed.revision == appended.revision
    assert replayed.nodes["hyp/replanning-restores-plasticity"].status == "proposed"
    later, later_result = history.append(refresh_patch("rq/after-rejection"))
    assert later.revision == appended.revision + 1
    assert later_result.state.replay_status == "complete"
    assert "rq/after-rejection" in later_result.state.nodes


def test_discarded_rejection_does_not_enter_history_or_consume_revision(manifest) -> None:
    history = HistoryManager(manifest)

    with pytest.raises(PatchRejected) as caught:
        history.append(shape_invalid_patch(), discard_on_reject=True)

    assert caught.value.report.rejected is True
    assert history.load_patches() == []
    assert list((manifest.research_dir / "patches").glob("*.json")) == []
    accepted, result = history.append(seed_patch())
    assert accepted.revision == 1
    assert result.state.revision == 1


def test_tampered_accepted_patch_halts_before_it_and_blocks_later_writes(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(refresh_patch("rq/tampered"))
    history.append(refresh_patch("rq/never-replayed"))

    path = manifest.research_dir / "patches" / "000002.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["ops"][0]["nodes"][0]["type"] = "not-a-node-type"
    path.write_text(json.dumps(raw), encoding="utf-8")

    state = history.state()

    assert state.replay_status == "degraded"
    assert state.revision == 1
    assert state.replay_failure is not None
    assert state.replay_failure.revision == 2
    assert state.replay_failure.code == "patch-schema-invalid"
    assert "rq/tampered" not in state.nodes
    assert "rq/never-replayed" not in state.nodes
    with pytest.raises(ReplayHalted, match="revision 2"):
        history.append(refresh_patch("rq/refused"))
    with pytest.raises(ReplayHalted, match="revision 2"):
        history.append_batch(
            [
                Patch(
                    kind="approval",
                    author="human",
                    summary="This write must be refused.",
                    ops=[],
                )
            ]
        )
    assert not (manifest.research_dir / "patches" / "000004.json").exists()


def test_structural_failure_after_invalid_patch_reports_the_earliest_boundary(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    history.append(refresh_patch("rq/semantic-failure"))
    history.append(refresh_patch("rq/schema-failure"))

    semantic_path = manifest.research_dir / "patches" / "000002.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["ops"][0]["nodes"][0]["type"] = "not-a-node-type"
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    structural_path = manifest.research_dir / "patches" / "000003.json"
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    structural["kind"] = "retired-patch-kind"
    structural_path.write_text(json.dumps(structural), encoding="utf-8")

    state = history.state()

    assert state.replay_status == "degraded"
    assert state.revision == 1
    assert state.replay_failure is not None
    assert state.replay_failure.revision == 2
    assert state.replay_failure.code == "patch-schema-invalid"
    assert "rq/semantic-failure" not in state.nodes
    assert "rq/schema-failure" not in state.nodes


def test_patch_failing_part_way_leaks_no_earlier_operation(manifest) -> None:
    """A patch is all-or-nothing even when an earlier op in it already applied.

    `_fork_state` shares node objects between revisions and only copies the
    containers, so this is the property that keeps that sharing safe.
    """
    history = HistoryManager(manifest)
    history.append(seed_patch())
    before = history.state()
    partial = Patch(
        kind="refresh",
        author="agent",
        summary="Valid node followed by a malformed relation.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "rq/transfer-after-shift",
                        "type": "research_question",
                        "title": "Transfer after task shift",
                        "question": "Does replanning transfer to an unseen task family?",
                        "motivation": "The seed corpus left transfer unexamined.",
                        "scope": "Matched compute across task families.",
                        "status": "open",
                    }
                ],
            },
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/transfer-after-shift",
                        "target": "hyp/replanning-restores-plasticity",
                        "relation": "not_a_relation",
                    }
                ],
            },
        ],
    )

    appended, result = history.append(partial, raise_on_reject=False)

    assert result.reports[appended.revision].rejected is True
    after = history.state()
    assert "rq/transfer-after-shift" not in after.nodes
    assert set(after.nodes) == set(before.nodes)
    assert set(after.edges) == set(before.edges)


def test_invalid_agent_patch_is_auditable_but_not_materialized(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    patch = Patch(
        kind="refresh",
        author="agent",
        summary="Invalid gated transition.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "update_nodes",
                "nodes": [
                    {
                        "id": "hyp/replanning-restores-plasticity",
                        "changes": {"status": "supported"},
                    }
                ],
            }
        ],
    )
    appended, result = history.append(patch, raise_on_reject=False)

    assert appended.revision == 2
    assert (manifest.research_dir / "patches" / "000002.json").exists()
    node = result.state.nodes["hyp/replanning-restores-plasticity"]
    assert node.status == "proposed"
    assert any(message.code == "gated-transition" for message in result.state.validation_messages)
    graph = json.loads((manifest.research_dir / "graph.json").read_text(encoding="utf-8"))
    assert graph["nodes"]["hyp/replanning-restores-plasticity"]["status"] == "proposed"


def test_append_refuses_a_patch_written_against_a_moved_revision(manifest) -> None:
    """The freshness check has to happen where the write happens.

    A caller that checked first and appended second would leave a window for any
    other writer — a human Sync takes the append lock without ever taking the
    agent run lock.
    """

    history = HistoryManager(manifest)
    history.append(seed_patch())
    stale = refresh_patch("rq/written-against-revision-1")

    history.append(refresh_patch("rq/landed-first"))

    with pytest.raises(RevisionConflict):
        history.append(stale, expected_revision=1)
    assert history.state().revision == 2
    assert not (manifest.research_dir / "patches" / "000004.json").exists()
    assert "rq/written-against-revision-1" not in history.state().nodes


@pytest.mark.parametrize("action", ["created", "adopted"])
def test_project_identity_claim_is_visible_idempotent_and_semantically_empty(
    manifest,
    action,
) -> None:
    space_id = str(uuid.uuid4())
    history = HistoryManager(manifest, expected_space_id=space_id)

    identity = history.claim_project_identity(action)
    again = history.claim_project_identity(action)
    result = history.current_materialization()

    assert again == identity
    assert uuid.UUID(identity.project_id).version == 4
    assert identity.project_id != space_id
    assert identity.home_space_id == space_id
    assert identity.action == action
    assert result.state.revision == 1
    assert result.state.nodes == {}
    assert result.state.edges == {}
    assert result.state.last_refresh_at is None
    assert len(history.load_patches()) == 1
    stored = history.load_patches()[0]
    assert stored.kind == "identity"
    assert stored.author is None
    assert stored.producer == "system"
    assert stored.ops == []
    assert stored.summary == (
        "Project created." if action == "created" else "Project identity adopted."
    )
    assert identity.home_space_id not in stored.summary
    assert result.patches == [stored]


def test_project_identity_claim_can_bind_one_prepared_id_and_refuses_another(
    manifest,
) -> None:
    space_id = str(uuid.uuid4())
    reserved_id = str(uuid.uuid4())
    history = HistoryManager(manifest, expected_space_id=space_id)

    identity = history.claim_project_identity("created", project_id=reserved_id)
    repeated = history.claim_project_identity("created", project_id=reserved_id)

    assert identity == repeated
    assert identity.project_id == reserved_id
    with pytest.raises(ProjectIdentityConflict, match="prepared project"):
        history.claim_project_identity("created", project_id=str(uuid.uuid4()))
    with pytest.raises(ProjectIdentityConflict, match="prepared project"):
        history.claim_project_identity("adopted", project_id=reserved_id)
    assert len(history.load_patches()) == 1


def test_prepared_identity_claim_refuses_any_prior_patch_history(manifest) -> None:
    HistoryManager(manifest).append(seed_patch())
    history = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))

    with pytest.raises(ProjectIdentityConflict, match="acquired Patch history"):
        history.claim_project_identity("created", project_id=str(uuid.uuid4()))

    assert len(history.load_patches()) == 1


@pytest.mark.parametrize("write_outputs", [False, True])
def test_replay_degrades_without_repairing_missing_scope_provenance(
    manifest,
    write_outputs,
) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    scope_base = manifest.research_dir / "scope-base.json"
    scope_base.unlink()

    result = HistoryManager(load_manifest(manifest.path)).materialize(write_outputs=write_outputs)

    assert result.state.replay_status == "degraded"
    assert result.state.revision == 0
    assert result.state.replay_failure is not None
    assert result.state.replay_failure.revision == 1
    assert result.state.replay_failure.code == "scope-provenance-missing"
    assert "absent while Patch history exists" in result.state.replay_failure.message
    assert not scope_base.exists()
    with pytest.raises(ReplayHalted, match="scope-provenance-missing"):
        history.head_ref(result)


def test_empty_history_may_bootstrap_scope_provenance_from_manifest(manifest) -> None:
    scope_base = manifest.research_dir / "scope-base.json"
    assert not scope_base.exists()

    result = HistoryManager(manifest).materialize(write_outputs=False)

    assert result.state.project_truth_scope == manifest.project.truth_scope
    assert result.patches == []
    assert not scope_base.exists()


def test_adopting_identity_never_mutates_prior_patches_or_research_semantics(manifest) -> None:
    legacy = HistoryManager(manifest)
    legacy.append(seed_patch())
    original_path = manifest.research_dir / "patches" / "000001.json"
    original_bytes = original_path.read_bytes()
    original_state = legacy.state()

    history = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))
    identity = history.claim_project_identity("adopted")
    adopted_state = history.state()

    assert identity.action == "adopted"
    assert original_path.read_bytes() == original_bytes
    assert adopted_state.revision == 2
    assert adopted_state.nodes == original_state.nodes
    assert adopted_state.edges == original_state.edges
    assert adopted_state.coverage == original_state.coverage
    assert adopted_state.last_refresh_at == original_state.last_refresh_at


def test_concurrent_same_home_claim_has_one_identity_revision(manifest) -> None:
    space_id = str(uuid.uuid4())
    first = HistoryManager(manifest, expected_space_id=space_id)
    second = HistoryManager(load_manifest(manifest.path), expected_space_id=space_id)
    ready = threading.Barrier(3)

    def claim(history: HistoryManager):
        ready.wait()
        return history.claim_project_identity("adopted")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(claim, first)
        second_future = executor.submit(claim, second)
        ready.wait()
        identities = [first_future.result(timeout=5), second_future.result(timeout=5)]

    assert identities[0] == identities[1]
    assert len(first.load_patches()) == 1
    assert first.state().revision == 1


def test_competing_home_claims_leave_first_winner_and_refuse_other(manifest) -> None:
    spaces = [str(uuid.uuid4()), str(uuid.uuid4())]
    managers = [HistoryManager(manifest, expected_space_id=item) for item in spaces]
    ready = threading.Barrier(3)

    def claim(history: HistoryManager):
        ready.wait()
        return history.claim_project_identity("adopted")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim, manager) for manager in managers]
        ready.wait()
        outcomes: list[object] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=5))
            except ProjectIdentityConflict as exc:
                outcomes.append(exc)

    winners = [item for item in outcomes if not isinstance(item, Exception)]
    refusals = [item for item in outcomes if isinstance(item, ProjectIdentityConflict)]
    assert len(winners) == 1
    assert len(refusals) == 1
    assert managers[0].project_identity() == winners[0]
    assert len(managers[0].load_patches()) == 1


def test_foreign_home_refuses_single_batch_and_settings_writes(manifest) -> None:
    home = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))
    identity = home.claim_project_identity("created")
    foreign = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))
    manifest_before = manifest.path.read_bytes()

    with pytest.raises(ProjectIdentityConflict, match="belongs to space"):
        foreign.append(seed_patch())
    with pytest.raises(ProjectIdentityConflict, match="belongs to space"):
        foreign.append_batch(
            [Patch(kind="approval", author="human", summary="Must not land.", ops=[])]
        )
    with pytest.raises(ProjectIdentityConflict, match="belongs to space"):
        foreign.update_machine_provider_paths({"laptop": {"codex": "/foreign/codex"}})

    assert home.project_identity() == identity
    assert len(home.load_patches()) == 1
    assert manifest.path.read_bytes() == manifest_before


def test_foreign_home_refuses_coherent_initialization(manifest) -> None:
    home = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))
    home.claim_project_identity("created")
    foreign = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))

    with pytest.raises(ProjectIdentityConflict, match="belongs to space"):
        foreign.initialize()


def test_foreign_home_forensic_replay_is_read_only(manifest) -> None:
    home = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))
    home.claim_project_identity("created")
    for name in HistoryManager._materialized_paths():
        if name.name == "scope-base.json":
            continue
        path = manifest.research_dir / name
        if path.exists():
            path.unlink()
    files_before = sorted(
        path.relative_to(manifest.research_dir)
        for path in manifest.research_dir.rglob("*")
        if path.is_file()
    )
    foreign = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))

    result = foreign.materialize(write_outputs=False)

    files_after = sorted(
        path.relative_to(manifest.research_dir)
        for path in manifest.research_dir.rglob("*")
        if path.is_file()
    )
    assert result.state.revision == 1
    assert result.state.nodes == {}
    assert files_after == files_before
    with pytest.raises(ProjectIdentityConflict, match="belongs to space"):
        foreign.initialize()
    assert (
        sorted(
            path.relative_to(manifest.research_dir)
            for path in manifest.research_dir.rglob("*")
            if path.is_file()
        )
        == files_before
    )


def test_conflicting_identity_revisions_degrade_identity_and_refuse_writes(manifest) -> None:
    low_level = HistoryManager(manifest)
    for action in ("created", "adopted"):
        low_level.append(
            Patch.model_validate(
                {
                    "kind": "identity",
                    "author": None,
                    "producer": "system",
                    "summary": "Conflicting fixture identity.",
                    "ops": [],
                    "project_identity": {
                        "project_id": str(uuid.uuid4()),
                        "home_space_id": str(uuid.uuid4()),
                        "action": action,
                    },
                }
            )
        )
    guarded = HistoryManager(manifest, expected_space_id=str(uuid.uuid4()))

    with pytest.raises(ProjectIdentityConflict, match="conflicting"):
        guarded.project_identity()
    with pytest.raises(ProjectIdentityConflict, match="conflicting"):
        guarded.append(seed_patch())
    assert low_level.materialize(write_outputs=False).state.revision == 2


def test_legacy_patch_without_producer_replays_without_rewriting_history(manifest) -> None:
    history = HistoryManager(manifest)
    history.append(seed_patch())
    path = manifest.research_dir / "patches" / "000001.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("producer")
    path.write_text(json.dumps(raw), encoding="utf-8")
    legacy_bytes = path.read_bytes()

    state = HistoryManager(load_manifest(manifest.path)).materialize(write_outputs=False).state

    assert state.revision == 1
    assert "rq/learning-after-shift" in state.nodes
    assert path.read_bytes() == legacy_bytes


def _transition_question_patch(node_id: str, *, revision: int = 0) -> Patch:
    return Patch(
        revision=revision,
        kind="refresh",
        author="agent",
        producer="agent",
        summary=f"Create {node_id}.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": node_id,
                        "type": "research_question",
                        "title": node_id,
                        "question": "Does exact transition ancestry remain intact?",
                    }
                ],
            }
        ],
    )


def test_main_replay_rejects_a_divergent_transition_history_splice(manifest) -> None:
    alternate_first = GraphTransitionManager().prepare_validated(
        GraphState(project_truth_scope=["repo-a"]),
        [_transition_question_patch("rq/alternate-first", revision=1)],
    )
    assert alternate_first.patch.transition is not None
    alternate_second = GraphTransitionManager().prepare_validated(
        alternate_first.projection.graph,
        [_transition_question_patch("rq/alternate-second", revision=2)],
        pre_head=GraphHeadRef(
            revision=1,
            transition_id=alternate_first.patch.transition.transition_id,
        ),
    )
    history = HistoryManager(manifest)
    accepted_first, _result = history.append(_transition_question_patch("rq/accepted-first"))
    assert accepted_first.transition is not None
    assert (
        alternate_second.patch.transition is not None
        and alternate_second.patch.transition.pre_head.transition_id
        == alternate_first.patch.transition.transition_id
    )
    (manifest.research_dir / "patches" / "000002.json").write_text(
        alternate_second.patch.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    replay = history.materialize(write_outputs=False).state

    assert replay.replay_status == "degraded"
    assert replay.replay_failure is not None
    assert replay.replay_failure.code == "transition-head-mismatch"
    assert replay.replay_failure.revision == 2
    assert replay.revision == 1
    assert "rq/accepted-first" in replay.nodes
    assert "rq/alternate-second" not in replay.nodes


def test_main_replay_rejects_a_transition_for_a_different_target(manifest) -> None:
    history = HistoryManager(manifest)
    appended, _result = history.append(_transition_question_patch("rq/main-target"))
    assert appended.transition is not None
    forged_trace = appended.transition.model_copy(
        update={
            "pre_head": appended.transition.pre_head.model_copy(
                update={"target": GraphTargetRef(kind="branch", branch_id=str(uuid.uuid4()))}
            )
        }
    )
    path = manifest.research_dir / "patches" / "000001.json"
    path.write_text(
        appended.model_copy(update={"transition": forged_trace}).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    replay = history.materialize(write_outputs=False).state

    assert replay.replay_status == "degraded"
    assert replay.replay_failure is not None
    assert replay.replay_failure.code == "transition-head-mismatch"
    assert replay.replay_failure.revision == 1
    assert replay.revision == 0
    assert "rq/main-target" not in replay.nodes
