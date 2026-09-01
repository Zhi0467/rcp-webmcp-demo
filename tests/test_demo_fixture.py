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
from pathlib import Path

import pytest

from rcp.config import load_manifest
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

    # Flags are allowed and currently expected: the fixture keeps a deliberately
    # mistyped historical `supports` edge to demonstrate endpoint diagnostics. A
    # *reject* is different: it means replay hit something structurally broken.
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
