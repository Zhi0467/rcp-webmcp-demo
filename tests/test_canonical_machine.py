from __future__ import annotations

import pytest

from rcp.config import AgentSurfaceConfig, Manifest
from rcp.history import HistoryManager
from rcp.paper import PaperService
from rcp.service import ProjectService
from rcp.setup import ProjectSetupRequest
from rcp.storage import AppStore


def test_manifest_rejects_graph_agent_off_canonical_machine(manifest) -> None:
    data = manifest.model_dump(mode="python")
    data["machines"].append({"alias": "remote", "host": "gpu.example"})
    data["agent"]["refresh"] = AgentSurfaceConfig(
        provider="codex",
        reasoning="medium",
        run_on="remote",
    ).model_dump(mode="python")

    with pytest.raises(ValueError, match="canonical state machine"):
        Manifest.model_validate(data)


def test_setup_rejects_graph_profile_off_canonical_machine(tmp_path) -> None:
    local = tmp_path / "local"
    local.mkdir()

    with pytest.raises(ValueError, match="refresh must run beside canonical state"):
        ProjectSetupRequest.model_validate(
            {
                "name": "mixed",
                "repositories": [
                    {
                        "alias": "local",
                        "location": "local",
                        "path": str(local),
                    },
                    {
                        "alias": "remote",
                        "location": "ssh",
                        "host": "gpu.example",
                        "path": "/srv/project",
                    },
                ],
                "state_repository": "local",
                "agents": {
                    "seed": {"location": "local"},
                    "refresh": {"location": "ssh", "host": "gpu.example"},
                    "node_chat": {"location": "local"},
                    "project_chat": {"location": "local"},
                    "paper_coach": {"location": "ssh", "host": "gpu.example"},
                },
            }
        )


def test_runtime_override_cannot_move_graph_agent(manifest, tmp_path) -> None:
    manifest.machines.append(type(manifest.machines[0])(alias="remote", host="gpu.example"))
    history = HistoryManager(manifest)
    store = AppStore(tmp_path / "rcp.sqlite3")
    paper = PaperService(manifest, store, history.workspace, project_id="project")
    service = ProjectService(manifest, history, paper, data_dir=tmp_path / "data")

    with pytest.raises(ValueError, match="canonical state machine"):
        service.resolve_agent_profile("refresh", run_on="remote")

    coach = service.resolve_agent_profile("paper_coach", run_on="remote")
    assert coach.run_on == "remote"


def test_provider_override_does_not_inherit_previous_provider_model(manifest, tmp_path) -> None:
    stored = manifest.agent_profile("project_chat")
    manifest.agent.project_chat = stored.model_copy(
        update={"provider": "codex", "model": "gpt-5.6-luna"}
    )
    history = HistoryManager(manifest)
    store = AppStore(tmp_path / "rcp.sqlite3")
    paper = PaperService(manifest, store, history.workspace, project_id="project")
    service = ProjectService(manifest, history, paper, data_dir=tmp_path / "data")

    provider_default = service.resolve_agent_profile(
        "project_chat",
        provider="claude",
        model=None,
    )
    explicit = service.resolve_agent_profile(
        "project_chat",
        provider="claude",
        model="claude-opus-4-1",
    )

    assert provider_default.provider == "claude"
    assert provider_default.model == ""
    assert explicit.provider == "claude"
    assert explicit.model == "claude-opus-4-1"
