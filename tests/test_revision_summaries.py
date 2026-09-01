from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import rcp.history.delta as delta_module
from rcp.agents.prompts import PromptFactory
from rcp.core.materialize import MaterializationResult, materialize_patches
from rcp.core.models import AuthorizedHuman, GraphState, Patch
from rcp.core.validation import ValidationReport
from rcp.history import build_revision_summaries
from rcp.service import ReviewRequest
from tests.helpers import (
    append_fixture_patch,
    create_named_app,
    refresh_patch,
    seed_patch,
    shape_invalid_patch,
)


def _patch(
    revision: int,
    ops: list[dict[str, object]],
    *,
    change_summary: list[str] | None = None,
) -> Patch:
    return Patch(
        revision=revision,
        kind="approval",
        author="human",
        summary=f"Revision {revision}",
        ops=ops,
        change_summary=change_summary or [],
    )


def test_revision_summaries_resolve_titles_fallback_and_quote_stored_consequences() -> None:
    patches = [
        _patch(
            1,
            [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "hyp/repeated-updates",
                            "type": "hypothesis",
                            "title": "Plasticity under repeated updates",
                            "statement": "Repeated updates may reduce future plasticity.",
                        }
                    ],
                }
            ],
            change_summary=["Added hyp/repeated-updates through create_nodes."],
        ),
        _patch(
            2,
            [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "hyp/repeated-updates",
                            "changes": {"title": "Future plasticity after repeated updates"},
                        }
                    ],
                }
            ],
        ),
        _patch(
            3,
            [
                {
                    "op": "create_proposals",
                    "proposals": [
                        {
                            "id": "prop/expand-probe-grid",
                            "title": "Expand the probe grid",
                            "card": {
                                "consequences": (
                                    "This adds two checkpoints for hyp/repeated-updates."
                                )
                            },
                            "ops": [],
                        }
                    ],
                }
            ],
            change_summary=["Recorded prop/expand-probe-grid."],
        ),
        _patch(
            4,
            [
                {
                    "op": "resolve_proposals",
                    "resolutions": [{"id": "prop/expand-probe-grid", "status": "approved"}],
                }
            ],
        ),
    ]
    materialization = MaterializationResult(
        state=GraphState(revision=4),
        reports={patch.revision: ValidationReport() for patch in patches},
    )

    summaries = build_revision_summaries(patches, materialization)

    assert summaries[0].sentences == ["Added Plasticity under repeated updates."]
    assert summaries[1].sentences == ["Updated “Future plasticity after repeated updates”."]
    assert summaries[2].sentences == [
        "Recorded Expand the probe grid.",
        (
            "The proposal “Expand the probe grid” records this consequence: "
            "“This adds two checkpoints for Future plasticity after repeated updates.”"
        ),
    ]
    assert summaries[3].sentences == [
        "Approved proposal “Expand the probe grid”.",
        (
            "The proposal “Expand the probe grid” records this consequence: "
            "“This adds two checkpoints for Future plasticity after repeated updates.”"
        ),
    ]
    rendered = " ".join(sentence for summary in summaries for sentence in summary.sentences)
    assert "hyp/repeated-updates" not in rendered
    assert "prop/expand-probe-grid" not in rendered
    assert "create_nodes" not in rendered
    assert "update_nodes" not in rendered
    assert "resolve_proposals" not in rendered


def test_revision_summaries_preserve_unresolved_slash_tokens_and_paths() -> None:
    patches = [
        _patch(
            1,
            [
                {
                    "op": "create_nodes",
                    "nodes": [
                        {
                            "id": "hyp/path-aware",
                            "type": "hypothesis",
                            "title": "Probe runner scripts/run-probe",
                            "statement": "The runner preserves source paths.",
                        }
                    ],
                }
            ],
            change_summary=[
                (
                    "Recorded hyp/path-aware from configs/routes.yaml, docs/metrics.md, "
                    "slime/train.py, and scripts/run-probe using create_nodes."
                )
            ],
        ),
        _patch(
            2,
            [
                {
                    "op": "update_nodes",
                    "nodes": [
                        {
                            "id": "hyp/path-aware",
                            "changes": {"rationale": "Keep the paths visible."},
                        }
                    ],
                }
            ],
        ),
    ]
    materialization = MaterializationResult(
        state=GraphState(revision=2),
        reports={patch.revision: ValidationReport() for patch in patches},
    )

    summaries = build_revision_summaries(patches, materialization)

    assert summaries[0].sentences == [
        (
            "Recorded Probe runner scripts/run-probe from configs/routes.yaml, docs/metrics.md, "
            "slime/train.py, and scripts/run-probe."
        )
    ]
    assert summaries[1].sentences == ["Updated “Probe runner scripts/run-probe”."]
    rendered = " ".join(sentence for summary in summaries for sentence in summary.sentences)
    for path in (
        "configs/routes.yaml",
        "docs/metrics.md",
        "slime/train.py",
        "scripts/run-probe",
    ):
        assert path in rendered
    assert "hyp/path-aware" not in rendered
    assert "create_nodes" not in rendered


def test_revision_summaries_preserve_inventory_prose_alongside_other_sentences() -> None:
    patch = _patch(
        1,
        [
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": "hyp/repeated-updates",
                        "type": "hypothesis",
                        "title": "Plasticity under repeated updates",
                        "statement": "Repeated updates may reduce future plasticity.",
                    }
                ],
            }
        ],
        change_summary=[
            "Updated 5 nodes.",
            "Clarified hyp/repeated-updates.",
        ],
    )
    materialization = MaterializationResult(
        state=GraphState(revision=1),
        reports={1: ValidationReport()},
    )

    summaries = build_revision_summaries([patch], materialization)

    assert summaries[0].sentences == [
        "Updated 5 nodes.",
        "Clarified Plasticity under repeated updates.",
    ]


def test_summary_api_is_additive_and_preserves_raw_history(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    append_fixture_patch(app.state.service, seed_patch())
    append_fixture_patch(
        app.state.service,
        refresh_patch().model_copy(update={"change_summary": []}),
    )
    client = TestClient(app)
    project_id = app.state.default_project_id

    summaries = client.get(f"/api/projects/{project_id}/history/summaries")
    raw = client.get(f"/api/projects/{project_id}/history")

    assert summaries.status_code == 200
    assert summaries.json()[-1] == {
        "from_revision": 2,
        "to_revision": 3,
        "kind": "refresh",
        "author": "agent",
        "producer": "agent",
        "authorized_by": None,
        "profile": None,
        "task_id": None,
        "episode_id": None,
        "episode": None,
        "created_at": raw.json()[-1]["created_at"],
        "sentences": ["Recorded a research question: “Transfer after task shift”."],
    }
    assert raw.status_code == 200
    assert set(raw.json()[-1]) == {
        "revision",
        "kind",
        "created_at",
        "summary",
        "change_summary",
    }
    assert raw.json()[-1]["change_summary"] == []


def test_manager_collects_range_during_one_replay_and_skips_stored_rejection(
    manifest,
    monkeypatch,
    tmp_path,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    history = app.state.service.history
    append_fixture_patch(app.state.service, seed_patch())
    rejected, _ = append_fixture_patch(
        app.state.service,
        shape_invalid_patch(),
        raise_on_reject=False,
    )
    accepted, _ = append_fixture_patch(app.state.service, refresh_patch())

    def reject_duplicate_apply(*_args) -> None:
        raise AssertionError("manager summaries must not apply accepted patches a second time")

    monkeypatch.setattr(delta_module, "apply_valid_patch", reject_duplicate_apply)

    summaries = history.revision_summaries(from_revision=3, to_revision=4)

    assert rejected.admission == "rejected"
    assert summaries == [
        {
            "from_revision": 3,
            "to_revision": accepted.revision,
            "kind": "refresh",
            "author": "agent",
            "producer": "agent",
            "authorized_by": None,
            "profile": None,
            "task_id": None,
            "episode_id": None,
            "created_at": accepted.created_at.isoformat(),
            "sentences": ["Added Transfer after task shift."],
        }
    ]


@pytest.mark.parametrize(
    ("action", "prefix"),
    [
        ("created", "Project created in"),
        ("adopted", "Project identity adopted in"),
    ],
)
def test_identity_revision_summary_uses_system_prose(action, prefix) -> None:
    space_id = str(uuid.uuid4())
    patch = Patch.model_validate(
        {
            "revision": 1,
            "kind": "identity",
            "author": None,
            "producer": "system",
            "summary": "Internal identity revision.",
            "ops": [],
            "project_identity": {
                "project_id": str(uuid.uuid4()),
                "home_space_id": space_id,
                "action": action,
            },
        }
    )
    materialization = MaterializationResult(
        state=GraphState(revision=1),
        reports={1: ValidationReport()},
    )

    summary = build_revision_summaries([patch], materialization)[0]

    assert summary.sentences == [f"{prefix} {space_id}."]
    assert summary.author is None
    assert summary.producer == "system"
    assert summary.authorized_by is None
    assert summary.profile is None
    assert summary.task_id is None


def test_revision_summary_preserves_human_snapshot_and_agent_task_attribution() -> None:
    authorized = AuthorizedHuman(
        space_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        display_name="Alex Kim",
    )
    human = _patch(1, []).model_copy(update={"authorized_by": authorized})
    agent = refresh_patch().model_copy(
        update={
            "revision": 2,
            "authorized_by": authorized,
            "profile": "ordinary",
            "task_id": "task-direct-1",
        }
    )
    materialization = MaterializationResult(
        state=GraphState(revision=2),
        reports={1: ValidationReport(), 2: ValidationReport()},
    )

    summaries = build_revision_summaries([human, agent], materialization)

    assert summaries[0].producer == "human"
    assert summaries[0].authorized_by == authorized
    assert summaries[0].profile is None
    assert summaries[0].task_id is None
    assert summaries[1].producer == "agent"
    assert summaries[1].authorized_by == authorized
    assert summaries[1].profile == "ordinary"
    assert summaries[1].task_id == "task-direct-1"


def test_legacy_revision_summary_remains_explicitly_unattributed() -> None:
    legacy = Patch.model_validate(
        {
            "revision": 1,
            "kind": "approval",
            "author": "human",
            "summary": "Legacy human revision.",
            "ops": [],
        }
    )
    materialization = MaterializationResult(
        state=GraphState(revision=1),
        reports={1: ValidationReport()},
    )

    summary = build_revision_summaries([legacy], materialization)[0]

    assert summary.producer == "human"
    assert summary.authorized_by is None
    assert summary.profile is None
    assert summary.task_id is None


def test_replay_observer_runs_only_after_successful_patch_application() -> None:
    first = seed_patch().model_copy(update={"revision": 1})
    corrupt = Patch(
        revision=2,
        kind="refresh",
        author="agent",
        summary="Referenced a missing edge source.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_edges",
                "edges": [
                    {
                        "source": "rq/missing",
                        "target": "hyp/replanning-restores-plasticity",
                        "relation": "has_hypothesis",
                    }
                ],
            }
        ],
    )
    later = refresh_patch().model_copy(update={"revision": 3})
    observations: list[tuple[int, int, int]] = []

    result = materialize_patches(
        [first, corrupt, later],
        ["repo-a"],
        repository_aliases=["repo-a"],
        accepted_patch_observer=lambda previous, patch, state: observations.append(
            (previous.revision, patch.revision, state.revision)
        ),
    )

    assert observations == [(0, 1, 1)]
    assert result.state.replay_status == "degraded"
    assert result.state.replay_failure is not None
    assert result.state.replay_failure.revision == 2
    assert 3 not in result.reports


def test_human_review_patch_uses_the_node_title(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    owner = app.state.background_tasks.store.local_owner
    assert owner is not None and owner.display_name is not None

    service.review_node(
        "rq/learning-after-shift",
        ReviewRequest(standing="accepted"),
        authorized_by=AuthorizedHuman(
            space_id=app.state.space_id,
            user_id=owner.user_id,
            display_name=owner.display_name,
        ),
    )

    patch = service.history.load_patches()[-1]
    assert patch.summary == "Marked “Learning after task shift” accepted."
    assert patch.change_summary == ["“Learning after task shift” is now accepted."]


def test_graph_and_work_contracts_require_reader_facing_change_summaries() -> None:
    graph_contract = PromptFactory.graph_task_contract(
        "refresh",
        project_name="Example",
        ontology_path="/state/graph.json#ontology",
        ontology_extensions=True,
        graph_path="/state/graph.json",
        research_path="/state/research.md",
        provider_log_roots={},
        ingestion_watermark=None,
        repositories=[],
        patch_path="/stage/patch.json",
        output_schema_path="/stage/schema.json",
        validator_command="python /stage/validator.py /stage/patch.json",
    )
    work_contract = PromptFactory.work_task_contract(
        project_name="Example",
        ontology_path="/state/graph.json#ontology",
        ontology_extensions=True,
        graph_path="/state/graph.json",
        research_path="/state/research.md",
        focused_node_id=None,
        repositories=[],
        introduction_path=None,
        human_request_path="/stage/request.txt",
        patch_path="/stage/patch.json",
        artifact_path="/stage/artifacts",
        output_schema_path="/stage/schema.json",
        validator_command="python /stage/validator.py /stage/patch.json",
    )

    for contract in (graph_contract, work_contract):
        assert "one ordinary-language sentence per meaningful" in contract
        assert "reader-facing titles, never ids or Patch operation names" in contract
        assert "do not summarize\n  with inventory counts" in contract or (
            "do not\n  use inventory counts" in contract
        )
        assert "instead of inventing a causal explanation" in contract
