"""Guards the shipped demo fixture.

`examples/demo-project/state-repo` is the documented Setup for several acceptance
scenarios (S03, S08, S10, S11, S15) — they open "a temporary copy of the demo
project", and S03 does it with no agent at all, so the graph has to already
exist. Nothing else in the suite touches it, which means the starting state those
scenarios depend on could rot silently.

These tests assert the properties the scenarios actually rely on, not the
fixture's current shape. Node counts and revision numbers are deliberately not
pinned, so the fixture can be rebuilt without editing this file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rcp.config import load_manifest
from rcp.control import derive_experiment_control_state
from rcp.history.manager import HistoryManager

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "demo-project"


@pytest.fixture
def demo_state(tmp_path: Path) -> Path:
    """A temporary copy, exactly as the scenarios describe. Never the original —
    opening the demo mutates it, and a dirty fixture is a signal, not noise."""
    shutil.copytree(FIXTURE, tmp_path / "demo-project")
    return tmp_path / "demo-project" / "state-repo"


def test_demo_fixture_replays_from_its_patch_log(demo_state: Path) -> None:
    manifest = load_manifest(demo_state / ".research" / "manifest.toml")
    state = HistoryManager(manifest).initialize().state

    assert state.replay_status == "complete"
    assert state.replay_failure is None
    assert state.revision > 0
    assert state.nodes

    # Historical flags may remain as append-only admission evidence. A *reject*
    # is different: it means replay hit something structurally broken.
    rejects = [item for item in state.validation_messages if item.level == "reject"]
    assert rejects == []


def test_demo_fixture_still_offers_the_work_the_scenarios_open_it_for(
    demo_state: Path,
) -> None:
    state = (
        HistoryManager(load_manifest(demo_state / ".research" / "manifest.toml")).initialize().state
    )

    # S08 approves a pending Proposal. S94 keeps the historical Ambiguity as a
    # replay-compatibility fixture even though Ambiguities no longer render or
    # admit new authoring operations.
    assert state.proposals, "S08 needs a proposal waiting on judgment"
    assert state.ambiguities, "S94 needs a historical Ambiguity replay fixture"

    # S11 coaches against an existing introduction.
    introduction = demo_state / ".research" / "paper" / "introduction.md"
    assert introduction.read_text().strip(), "S11 needs authored introduction text"

    # The challenge journey can discuss and run through the same normal profile
    # without launching a real provider or waiting on external work.
    manifest = load_manifest(demo_state / ".research" / "manifest.toml")
    for surface in (
        "seed",
        "refresh",
        "node_chat",
        "project_chat",
        "paper_coach",
        "orchestrator",
    ):
        assert manifest.agent_profile(surface).provider == "rcp-demo"
    decision = state.nodes["dec/match-endpoint-or-training-path"]
    assert decision.status == "decided"
    assert decision.selected_option == "Match the full update trajectory"
    experiment = state.nodes["exp/two-update-matched-trajectory"]
    assert experiment.attempts[-1].id == "attempt/04"
    assert experiment.attempts[-1].status == "planned"
    assert experiment.current_summary_stale is False
    assert experiment.next_action_stale is False
    assert "ev/external-path-match-study" not in state.nodes
    control = derive_experiment_control_state(state, "exp/two-update-matched-trajectory")
    assert control.ready is True
    assert control.reasons == []


def test_demo_study_reference_analysis_matches_the_seeded_claims(demo_state: Path) -> None:
    analysis = subprocess.run(
        [
            sys.executable,
            "study/analyze_held_out.py",
            "study/held_out_trajectory.csv",
        ],
        cwd=demo_state,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(analysis.stdout)

    assert result["synthetic"] is True
    assert result["rows"] == 30
    assert result["matching"] == {
        "max_first_shift_kl_gap": 0.002,
        "max_first_shift_return_gap": 0.01,
        "passed": True,
        "tolerance": 0.02,
    }
    assert result["second_shift_slope"]["mean"] == {
        "search_assisted": 0.178333,
        "value_only": 0.025333,
    }
    assert result["second_shift_slope"]["search_minus_value"] == 0.153


def test_checked_in_graph_json_is_what_the_patch_log_produces(demo_state: Path) -> None:
    """`graph.json` is an output (invariant 2). If the committed one describes a
    graph the log cannot produce, someone hand-edited it, or it rotted against a
    code change — either way the fixture stops being a trustworthy starting state.

    Compares node and edge identity only. Serialization details are deliberately
    not pinned, so a change to how an edge is *described* does not fail here while
    a change to which nodes and edges *exist* does.
    """
    committed = json.loads((demo_state / ".research" / "graph.json").read_text())
    replayed = HistoryManager(
        load_manifest(demo_state / ".research" / "manifest.toml")
    ).initialize()

    assert set(replayed.state.nodes) == set(committed["nodes"])
    assert set(replayed.state.edges) == set(committed["edges"])
