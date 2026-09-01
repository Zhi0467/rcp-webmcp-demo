from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

import rcp.projects as projects_module
from rcp.agents import ProviderReadiness
from rcp.config import load_manifest, permissions_for
from rcp.core.models import Patch
from rcp.history import HistoryManager
from rcp.providers import ProviderUsage
from rcp.storage import AgentTaskRecord

from .helpers import append_fixture_patch, create_named_app, gated_patch, seed_patch


def _experiment_fixture_patch(
    experiment_id: str = "exp/bounded-loop",
    *,
    invocation_ceiling: int = 2,
) -> Patch:
    return Patch(
        kind="refresh",
        author="agent",
        summary="Added an experiment for control-loop tests.",
        run_truth_scope=["repo-a"],
        repositories_read=["repo-a"],
        ops=[
            {
                "op": "create_nodes",
                "nodes": [
                    {
                        "id": experiment_id,
                        "type": "experiment",
                        "title": "Bounded loop",
                        "objective": "Exercise the experiment control contract.",
                        "completion_criteria": ["The detached fixture exits cleanly."],
                        "invocation_ceiling": invocation_ceiling,
                    }
                ],
            }
        ],
    )


def test_project_display_boundary_completes_all_public_snapshots(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    service = app.state.service
    append_fixture_patch(service, seed_patch())
    append_fixture_patch(service, _experiment_fixture_patch())
    project_id = app.state.default_project_id
    assert project_id is not None
    draft = service.project_snapshot()

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(draft)
    with pytest.raises(ValueError, match="__dict__"):
        jsonable_encoder(draft)

    client = TestClient(app)
    generation = app.state.catalog.reserve_cached_snapshot_generation(project_id)
    assert app.state.catalog.commit_cached_snapshot(
        project_id,
        draft,
        generation=generation,
        patch_log_head=service.history.workspace.cached_patch_log_head(),
    )
    raw_saved = app.state.catalog.cached_snapshot(project_id)
    assert raw_saved is not None
    assert "experiment_control" not in raw_saved

    saved = client.get(f"/api/projects/{project_id}/cached")
    assert saved.status_code == 200
    assert set(saved.json()["experiment_control"]) == {"exp/bounded-loop"}

    app.state.catalog._cached_snapshot_path(project_id).unlink()
    current = client.get(f"/api/projects/{project_id}")
    assert current.status_code == 200
    assert set(current.json()["experiment_control"]) == {"exp/bounded-loop"}

    body = {
        "default_run_truth_scope": current.json()["default_run_truth_scope"],
        "agent_profiles": {
            surface: {key: profile[key] for key in ("provider", "model", "reasoning", "run_on")}
            for surface, profile in current.json()["agent_profiles"].items()
        },
    }
    settings = client.put(f"/api/projects/{project_id}/settings", json=body)
    assert settings.status_code == 200
    assert set(settings.json()["experiment_control"]) == {"exp/bounded-loop"}
    raw_after_settings = app.state.catalog.cached_snapshot(project_id)
    assert raw_after_settings is not None
    assert "experiment_control" not in raw_after_settings


@pytest.mark.parametrize("corruption", ["attention", "attention_count"])
def test_cached_project_rejects_attention_that_disagrees_with_its_graph(
    manifest,
    tmp_path,
    corruption: str,
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    client = TestClient(app)
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    cache_path = app.state.catalog._cached_snapshot_path(project_id)
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    if corruption == "attention":
        envelope["snapshot"]["attention"]["open_blocker_ids"] = ["blk/not-in-graph"]
    else:
        envelope["snapshot"]["counts"]["open_blockers"] += 1
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    assert app.state.catalog.cached_snapshot_status(project_id) == ("invalid", None)


def test_project_revision_probe_is_small_and_does_not_replay_history(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    history = app.state.service.history

    assert client.get(f"/api/projects/{project_id}/revision").json() == {"revision": 1}
    append_fixture_patch(app.state.service, seed_patch())
    rejected, _ = append_fixture_patch(
        app.state.service,
        gated_patch(),
        raise_on_reject=False,
    )
    assert rejected.revision == 3
    assert rejected.admission == "rejected"

    monkeypatch.setattr(
        history,
        "materialize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("revision probe must not materialize history")
        ),
    )
    monkeypatch.setattr(
        history,
        "_replay",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("revision probe must not replay history")
        ),
    )
    monkeypatch.setattr(
        history.workspace,
        "refresh_if_stale",
        lambda: (_ for _ in ()).throw(
            AssertionError("revision probe must not refresh canonical state")
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("revision probe must not read canonical patch bodies")
        ),
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("revision probe must not open canonical patch files")
        ),
    )

    response = client.get(f"/api/projects/{project_id}/revision")

    assert response.status_code == 200
    assert response.json() == {"revision": 2}
    assert list(response.json()) == ["revision"]


def test_project_revision_probe_returns_normal_project_not_found(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    response = TestClient(app).get(f"/api/projects/{uuid.uuid4()}/revision")

    assert response.status_code == 404
    assert response.json() == {"detail": "Project not found"}


def test_cached_revision_heartbeat_is_cache_only_and_unchanged_head_starts_no_refresh(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    initial = TestClient(app).get(f"/api/projects/{project_id}").json()
    probes = 0

    def unchanged_head(requested_project_id):
        nonlocal probes
        assert requested_project_id == project_id
        probes += 1
        return "unchanged"

    monkeypatch.setattr(app.state.catalog, "probe_remote_patch_log_head", unchanged_head)
    monkeypatch.setattr(
        app.state.catalog,
        "reconcile_snapshot",
        lambda _project_id: (_ for _ in ()).throw(
            AssertionError("an unchanged head must not start a full refresh")
        ),
    )

    async def drive() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/projects/{project_id}/cached/revision")
            for _ in range(100):
                if project_id not in app.state.project_reconciliation_tasks:
                    break
                await asyncio.sleep(0.01)
            return response

    response = asyncio.run(drive())

    assert response.status_code == 200
    assert response.json() == {
        "revision": initial["revision"],
        "snapshot_freshness": "fresh",
        "last_remote_sync_at": None,
    }
    assert probes == 1


def test_cached_revision_heartbeat_enforces_three_second_probe_cooldown(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    assert TestClient(app).get(f"/api/projects/{project_id}").status_code == 200
    clock = 100.0
    probes = 0

    class FakeTime:
        @staticmethod
        def monotonic() -> float:
            return clock

    def unchanged_head(_project_id):
        nonlocal probes
        probes += 1
        return "unchanged"

    monkeypatch.setattr(projects_module, "time", FakeTime)
    monkeypatch.setattr(app.state.catalog, "probe_remote_patch_log_head", unchanged_head)

    async def wait_for_probe() -> None:
        for _ in range(100):
            if project_id not in app.state.project_reconciliation_tasks:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("background probe did not complete")

    async def drive() -> list[httpx.Response]:
        nonlocal clock
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.get(f"/api/projects/{project_id}/cached/revision")
            await wait_for_probe()
            clock = 102.999
            inside_cooldown = await client.get(f"/api/projects/{project_id}/cached/revision")
            await asyncio.sleep(0)
            assert project_id not in app.state.project_reconciliation_tasks
            clock = 103.0
            at_boundary = await client.get(f"/api/projects/{project_id}/cached/revision")
            await wait_for_probe()
            return [first, inside_cooldown, at_boundary]

    responses = asyncio.run(drive())

    assert all(response.status_code == 200 for response in responses)
    assert probes == 2


def test_moved_head_refreshes_in_background_singleflight(manifest, tmp_path, monkeypatch) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    initial = TestClient(app).get(f"/api/projects/{project_id}").json()
    append_fixture_patch(app.state.service, seed_patch())
    append_fixture_patch(app.state.service, _experiment_fixture_patch())
    entered = threading.Event()
    release = threading.Event()
    probe_calls = 0
    refresh_calls = 0
    reconcile_snapshot = app.state.catalog.reconcile_snapshot

    def moved_head(requested_project_id):
        nonlocal probe_calls
        assert requested_project_id == project_id
        probe_calls += 1
        return "moved"

    def blocked_reconcile(requested_project_id):
        nonlocal refresh_calls
        refresh_calls += 1
        entered.set()
        assert release.wait(timeout=3)
        return reconcile_snapshot(requested_project_id)

    monkeypatch.setattr(app.state.catalog, "probe_remote_patch_log_head", moved_head)
    monkeypatch.setattr(app.state.catalog, "reconcile_snapshot", blocked_reconcile)

    async def drive() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.get(f"/api/projects/{project_id}/cached/revision")
            assert await asyncio.to_thread(entered.wait, 1)
            second = await asyncio.wait_for(
                client.get(f"/api/projects/{project_id}/cached/revision"),
                timeout=1,
            )
            project = await asyncio.wait_for(
                client.get(f"/api/projects/{project_id}"),
                timeout=1,
            )
            release.set()
            for _ in range(100):
                if project_id not in app.state.project_reconciliation_tasks:
                    break
                await asyncio.sleep(0.01)
            return first, second, project

    first, second, project = asyncio.run(drive())
    refreshed = app.state.catalog.cached_snapshot(project_id)

    assert first.json()["revision"] == initial["revision"]
    assert second.status_code == 200
    assert project.status_code == 200
    assert project.json()["revision"] == initial["revision"]
    assert probe_calls == 1
    assert refresh_calls == 1
    assert refreshed is not None
    assert refreshed["revision"] == 3
    assert refreshed["snapshot_freshness"] == "fresh"
    assert set(refreshed["experiment_control"]) == {"exp/bounded-loop"}


def test_local_patch_head_refreshes_cache_without_joining_the_write_path(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    initial = TestClient(app).get(f"/api/projects/{project_id}").json()
    append_fixture_patch(app.state.service, seed_patch())

    async def drive() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/projects/{project_id}/cached/revision")
            for _ in range(100):
                if project_id not in app.state.project_reconciliation_tasks:
                    break
                await asyncio.sleep(0.01)
            return response

    response = asyncio.run(drive())
    refreshed = app.state.catalog.cached_snapshot(project_id)

    assert response.json()["revision"] == initial["revision"]
    assert refreshed is not None
    assert refreshed["revision"] == 2


def test_transient_head_probe_failure_marks_only_display_freshness_stale(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    initial = TestClient(app).get(f"/api/projects/{project_id}").json()
    monkeypatch.setattr(
        app.state.catalog,
        "probe_remote_patch_log_head",
        lambda _project_id: "unavailable",
    )

    async def drive() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get(f"/api/projects/{project_id}/cached/revision")
            for _ in range(100):
                if project_id not in app.state.project_reconciliation_tasks:
                    break
                await asyncio.sleep(0.01)

    asyncio.run(drive())
    cached = app.state.catalog.cached_snapshot(project_id)

    assert cached is not None
    assert cached["snapshot_freshness"] == "stale"
    assert cached["canonical_state"] == initial["canonical_state"]


def test_project_get_creates_then_reuses_display_snapshot_without_reopening(
    manifest, tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    app = create_named_app(str(manifest.path), data_dir=data_dir)
    client = TestClient(app)
    project_id = app.state.default_project_id

    initial = client.get(f"/api/projects/{project_id}")

    assert initial.status_code == 200
    cached_files = list((data_dir / "project-snapshots").iterdir())
    assert len(cached_files) == 1
    cache_path = cached_files[0]
    initial_envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    assert initial_envelope["schema_version"] == 3
    assert initial_envelope["canonical_patch_head"] == 1
    assert initial_envelope["project_id"] == project_id
    assert initial_envelope["snapshot"] == initial.json()
    assert initial_envelope["snapshot"]["default_auto_research_invocation_ceiling"] == 10
    assert "default_campaign_invocation_ceiling" not in initial_envelope["snapshot"]

    monkeypatch.setattr(
        app.state.catalog,
        "open_snapshot",
        lambda _project_id: (_ for _ in ()).throw(
            AssertionError("cached project navigation must not open canonical state")
        ),
    )
    monkeypatch.setattr(
        app.state.catalog,
        "probe_remote_patch_log_head",
        lambda _project_id: (_ for _ in ()).throw(
            AssertionError("cached project navigation must not issue a remote probe")
        ),
    )
    cached = client.get(f"/api/projects/{project_id}")

    assert cached.status_code == 200
    assert cached.json() == initial.json()
    assert list((data_dir / "project-snapshots").iterdir()) == [cache_path]
    assert json.loads(cache_path.read_text(encoding="utf-8")) == initial_envelope
    assert client.get(f"/api/projects/{project_id}/cached").json() == initial.json()


def test_cached_project_rejects_malformed_mismatched_and_oversize_files(
    manifest, tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    app = create_named_app(str(manifest.path), data_dir=data_dir)
    client = TestClient(app)
    project_id = app.state.default_project_id
    authoritative = client.get(f"/api/projects/{project_id}")
    assert authoritative.status_code == 200
    cache_path = next((data_dir / "project-snapshots").iterdir())

    cache_path.write_text("{", encoding="utf-8")
    assert client.get(f"/api/projects/{project_id}/cached").status_code == 404

    mismatched = {
        "schema_version": 1,
        "project_id": "different-project",
        "snapshot": authoritative.json(),
    }
    cache_path.write_text(json.dumps(mismatched), encoding="utf-8")
    assert client.get(f"/api/projects/{project_id}/cached").status_code == 404

    mismatched["project_id"] = project_id
    mismatched["snapshot"]["id"] = "different-project"
    cache_path.write_text(json.dumps(mismatched), encoding="utf-8")
    assert client.get(f"/api/projects/{project_id}/cached").status_code == 404

    legacy_snapshot = authoritative.json()
    legacy_snapshot["default_campaign_invocation_ceiling"] = legacy_snapshot.pop(
        "default_auto_research_invocation_ceiling"
    )
    legacy = {
        "schema_version": 2,
        "project_id": project_id,
        "canonical_patch_head": 1,
        "snapshot": legacy_snapshot,
    }
    cache_path.write_text(json.dumps(legacy), encoding="utf-8")
    migrated = client.get(f"/api/projects/{project_id}/cached")
    assert migrated.status_code == 200
    assert migrated.json()["default_auto_research_invocation_ceiling"] == 10
    assert "default_campaign_invocation_ceiling" not in migrated.json()

    legacy["snapshot"]["default_auto_research_invocation_ceiling"] = 11
    cache_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert client.get(f"/api/projects/{project_id}/cached").status_code == 404

    monkeypatch.setattr(projects_module, "PROJECT_DISPLAY_SNAPSHOT_MAX_BYTES", 16)
    cache_path.write_bytes(b"x" * 17)
    assert client.get(f"/api/projects/{project_id}/cached").status_code == 404


def test_cached_snapshot_names_the_runtime_on_profiles_saved_before_selection(
    manifest, tmp_path
) -> None:
    """A cached profile predating runtime selection is still readable.

    `agent_profiles` is part of the cached payload, so the first read after the
    upgrade would otherwise hand the settings form a profile with no runtime.
    """

    data_dir = tmp_path / "data"
    app = create_named_app(str(manifest.path), data_dir=data_dir)
    client = TestClient(app)
    project_id = app.state.default_project_id
    authoritative = client.get(f"/api/projects/{project_id}")
    assert authoritative.status_code == 200
    cache_path = next((data_dir / "project-snapshots").iterdir())

    legacy_snapshot = authoritative.json()
    for profile in legacy_snapshot["agent_profiles"].values():
        del profile["runtime"]
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "project_id": project_id,
                "canonical_patch_head": 1,
                "snapshot": legacy_snapshot,
            }
        ),
        encoding="utf-8",
    )

    migrated = client.get(f"/api/projects/{project_id}/cached")
    assert migrated.status_code == 200
    profiles = migrated.json()["agent_profiles"]
    assert profiles
    for surface, profile in profiles.items():
        expected = "exec" if profile["provider"] == "codex" else "stream-json"
        assert profile["runtime"] == expected, surface


def test_project_readiness_does_not_open_or_materialize_project(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    app.state.catalog._services.clear()
    monkeypatch.setattr(
        HistoryManager,
        "initialize",
        lambda _history: (_ for _ in ()).throw(
            AssertionError("readiness must not materialize project history")
        ),
    )
    calls: list[tuple[str, bool]] = []
    inventory_waits: list[tuple[str, str, str | None]] = []

    def readiness(provider: str, *, host: str = "", refresh: bool = False):
        calls.append((provider, refresh))
        return ProviderReadiness(
            provider=provider,
            installed=True,
            authenticated=True,
            version=f"{provider}-ready",
        )

    monkeypatch.setattr(app.state.catalog.launcher, "readiness", readiness)
    monkeypatch.setattr(
        app.state.provider_skills,
        "refresh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual readiness must not refresh provider skills")
        ),
    )
    monkeypatch.setattr(
        app.state.provider_skills,
        "wait",
        lambda provider, host, binary: inventory_waits.append((provider, host, binary)) or True,
    )

    response = client.get(f"/api/projects/{project_id}/readiness")
    refreshed = client.get(f"/api/projects/{project_id}/readiness?refresh=true")

    assert response.status_code == 200
    assert refreshed.status_code == 200
    assert response.json()["provider_readiness"]["laptop"]["codex"]["version"] == ("codex-ready")
    assert response.json()["providers"] == response.json()["provider_readiness"]["laptop"]
    assert response.json()["provider_skill_inventories"]["laptop"]["codex"]["status"] == (
        "unavailable"
    )
    assert set(calls) == {
        ("codex", False),
        ("claude", False),
        ("codex", True),
        ("claude", True),
    }
    assert inventory_waits == [
        ("codex", "", None),
        ("claude", "", None),
        ("codex", "", None),
        ("claude", "", None),
    ]
    assert project_id not in app.state.catalog._services


def test_project_settings_persist_agent_defaults_and_repository_reads(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    before = client.get(f"/api/projects/{project_id}").json()
    assert before["default_auto_research_invocation_ceiling"] == 10
    assert "default_campaign_invocation_ceiling" not in before
    profiles = {
        surface: {
            key: profile[key] for key in ("provider", "runtime", "model", "reasoning", "run_on")
        }
        for surface, profile in before["agent_profiles"].items()
    }
    assert set(profiles) == {
        "seed",
        "refresh",
        "node_chat",
        "project_chat",
        "paper_coach",
        "orchestrator",
    }
    profiles["seed"]["provider"] = "claude"
    profiles["seed"]["runtime"] = "stream-json"
    profiles["seed"]["model"] = "claude-seed"
    profiles["node_chat"]["runtime"] = "app-server"
    profiles["orchestrator"]["model"] = "campaign-orchestrator"

    incomplete = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            "default_run_truth_scope": ["repo-b"],
            "agent_profiles": {
                surface: profile
                for surface, profile in profiles.items()
                if surface != "orchestrator"
            },
        },
    )
    assert incomplete.status_code == 422

    mismatched_runtime = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            "default_run_truth_scope": ["repo-b"],
            "agent_profiles": {
                **profiles,
                "refresh": {**profiles["refresh"], "provider": "claude", "runtime": "app-server"},
            },
        },
    )
    assert mismatched_runtime.status_code == 422
    # The settings form shows this text, so it names the profile to fix and
    # carries none of the Pydantic envelope around the reason.
    detail = mismatched_runtime.json()["detail"]
    assert detail == "refresh: Provider 'claude' does not support runtime 'app-server'."

    invalid_budget = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            "default_run_truth_scope": ["repo-b"],
            "default_auto_research_invocation_ceiling": 0,
            "agent_profiles": profiles,
        },
    )
    assert invalid_budget.status_code == 422

    legacy_budget_key = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            "default_run_truth_scope": ["repo-b"],
            "default_campaign_invocation_ceiling": 14,
            "agent_profiles": profiles,
        },
    )
    assert legacy_budget_key.status_code == 422

    response = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            "default_run_truth_scope": ["repo-b"],
            "default_auto_research_invocation_ceiling": 14,
            "agent_profiles": profiles,
        },
    )

    assert response.status_code == 200
    assert response.json()["default_run_truth_scope"] == ["repo-b"]
    assert response.json()["default_auto_research_invocation_ceiling"] == 14
    assert "default_campaign_invocation_ceiling" not in response.json()
    assert response.json()["agent_profiles"]["seed"]["model"] == "claude-seed"
    assert response.json()["agent_profiles"]["node_chat"]["runtime"] == "app-server"
    assert response.json()["agent_profiles"]["orchestrator"]["model"] == "campaign-orchestrator"
    assert "write_path" not in response.json()["agent_profiles"]["refresh"]
    assert (
        client.get(f"/api/projects/{project_id}").json()["agent_profiles"]["seed"]["model"]
        == "claude-seed"
    )
    content = manifest.path.read_text(encoding="utf-8")
    assert "[execution]" not in content
    assert "[paper.coach]" not in content
    updated = load_manifest(manifest.path)
    assert updated.agent_profile("seed").provider == "claude"
    assert updated.agent_profile("node_chat").runtime == "app-server"
    assert updated.agent_profile("orchestrator").model == "campaign-orchestrator"
    assert updated.agent_profile("orchestrator").permissions == permissions_for("orchestrate")
    assert updated.agent.default_auto_research_invocation_ceiling == 14
    assert updated.agent_profile("paper_coach").permissions.write_graph_patch is False


def test_project_settings_merge_partial_provider_paths_and_preserve_omitted_values(
    manifest, tmp_path, monkeypatch
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    before = client.get(f"/api/projects/{project_id}").json()
    profiles = {
        surface: {key: profile[key] for key in ("provider", "model", "reasoning", "run_on")}
        for surface, profile in before["agent_profiles"].items()
    }
    base = {
        "default_run_truth_scope": before["default_run_truth_scope"],
        "agent_profiles": profiles,
    }
    invalidated: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        app.state.catalog.launcher,
        "invalidate_readiness",
        lambda provider, *, host="", binary=None: invalidated.append((provider, host, binary)),
    )

    first = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            **base,
            "machine_provider_paths": {"laptop": {"codex": "/opt/agents/codex"}},
        },
    )
    omitted = client.put(f"/api/projects/{project_id}/settings", json=base)
    partial = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            **base,
            "machine_provider_paths": {"laptop": {"claude": "/opt/agents/claude"}},
        },
    )

    assert first.status_code == omitted.status_code == partial.status_code == 200
    assert omitted.json()["machines"][0]["provider_paths"]["codex"] == "/opt/agents/codex"
    assert partial.json()["machines"][0]["provider_paths"] == {
        "codex": "/opt/agents/codex",
        "claude": "/opt/agents/claude",
    }
    updated = load_manifest(manifest.path)
    assert updated.machine_map["laptop"].host == ""
    assert updated.machine_map["laptop"].provider_paths["codex"] == "/opt/agents/codex"
    assert invalidated == [
        ("codex", "", "/opt/agents/codex"),
        ("claude", "", "/opt/agents/claude"),
    ]


def test_invalid_provider_path_update_is_atomic(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    before = client.get(f"/api/projects/{project_id}").json()
    content = manifest.path.read_text(encoding="utf-8")
    profiles = {
        surface: {key: profile[key] for key in ("provider", "model", "reasoning", "run_on")}
        for surface, profile in before["agent_profiles"].items()
    }

    response = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            "default_run_truth_scope": before["default_run_truth_scope"],
            "agent_profiles": profiles,
            "machine_provider_paths": {"laptop": {"codex": "relative/codex"}},
        },
    )

    assert response.status_code == 422
    assert manifest.path.read_text(encoding="utf-8") == content


def test_explicit_provider_resolve_discovers_then_persists(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    append_fixture_patch(app.state.service, seed_patch())
    append_fixture_patch(app.state.service, _experiment_fixture_patch())
    client = TestClient(app)
    project_id = app.state.default_project_id
    calls: list[str | None] = []

    def readiness(
        provider: str,
        *,
        host: str = "",
        binary: str | None = None,
        refresh: bool = False,
    ):
        assert provider == "codex"
        assert host == ""
        assert refresh is True
        calls.append(binary)
        path = binary or "/opt/new-agent/codex"
        return ProviderReadiness(
            provider=provider,
            installed=True,
            authenticated=True,
            binary_path=path,
            path_state="resolved" if binary else "unconfigured",
        )

    app.state.catalog.launcher.readiness = readiness

    response = client.post(f"/api/projects/{project_id}/machines/laptop/providers/codex/resolve")

    assert response.status_code == 200
    assert calls == [None, "/opt/new-agent/codex"]
    assert response.json()["binary_path"] == "/opt/new-agent/codex"
    assert response.json()["readiness"]["path_state"] == "resolved"
    assert response.json()["project"]["machines"][0]["provider_paths"]["codex"] == (
        "/opt/new-agent/codex"
    )
    assert set(response.json()["project"]["experiment_control"]) == {"exp/bounded-loop"}
    assert load_manifest(manifest.path).machine_map["laptop"].provider_paths["codex"] == (
        "/opt/new-agent/codex"
    )


def test_project_settings_reject_invalid_scope_without_changing_manifest(
    manifest, tmp_path
) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    client = TestClient(app)
    project_id = app.state.default_project_id
    before = client.get(f"/api/projects/{project_id}").json()
    content = manifest.path.read_text(encoding="utf-8")
    profiles = {
        surface: {key: profile[key] for key in ("provider", "model", "reasoning", "run_on")}
        for surface, profile in before["agent_profiles"].items()
    }

    response = client.put(
        f"/api/projects/{project_id}/settings",
        json={
            "default_run_truth_scope": ["not-a-project-repository"],
            "agent_profiles": profiles,
        },
    )

    assert response.status_code == 422
    assert manifest.path.read_text(encoding="utf-8") == content


def test_project_usage_endpoint_returns_counted_and_excluded_records(manifest, tmp_path) -> None:
    app = create_named_app(str(manifest.path), data_dir=tmp_path / "data")
    project_id = app.state.default_project_id
    store = app.state.background_tasks.store
    now = store.now()
    store.create_agent_task(
        AgentTaskRecord(
            operation_id="usage-operation",
            project_id=project_id,
            kind="node_chat",
            status="succeeded",
            request={"provider": "codex", "model": "gpt"},
            created_at=now,
            updated_at=now,
            status_message="done",
        )
    )
    usage = ProviderUsage(
        provider_profile="codex.turn.v1",
        provider_event_type="turn.completed",
        dedupe_key="turn-1",
        processed_input_tokens=2_000,
        generated_tokens=200,
        cached_input_tokens=1_000,
    )
    store.record_agent_usage("usage-operation", usage)
    store.record_agent_usage("usage-operation", usage)

    response = TestClient(app).get(f"/api/projects/{project_id}/usage")

    assert response.status_code == 200
    payload = response.json()
    assert payload["counted_records"] == 1
    assert payload["excluded_records"] == 1
    assert payload["input_processed"]["total_tokens"] == 2_000
    assert payload["input_processed"]["cache_share"] == 0.5
    assert payload["generated"]["total_tokens"] == 200
    assert {record["counted"] for record in payload["records"]} == {True, False}
