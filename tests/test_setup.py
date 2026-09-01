from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from rcp.agents import ProviderReadiness
from rcp.api import create_app
from rcp.config import load_manifest, permissions_for
from rcp.history import HistoryManager
from rcp.setup import ProjectSetupRequest, SetupAgents, render_manifest
from rcp.sources import project_cache_roots
from rcp.storage import AgentTaskRecord, AppStore
from rcp.transport import StateWorkspace

from .helpers import seed_patch


def test_setup_agent_profiles_no_longer_carry_a_write_mode() -> None:
    agents = SetupAgents()

    for surface in (
        "seed",
        "refresh",
        "node_chat",
        "project_chat",
        "paper_coach",
        "orchestrator",
    ):
        profile = agents.profile(surface)
        assert not hasattr(profile, "write_path")
        assert "write_path" not in profile.model_dump()
    assert agents.profile("orchestrator") == agents.profile("refresh")
    assert agents.profile("orchestrator") is not agents.profile("refresh")


def _local_payload(repository_path: str) -> dict[str, object]:
    return {
        "name": "wizard-paper",
        "repositories": [
            {
                "alias": "paper-repo",
                "location": "local",
                "path": repository_path,
                "host": "",
                "default_read": True,
            }
        ],
        "state_repository": "paper-repo",
        "execution": {
            "location": "local",
            "host": "",
        },
        "confirmed": False,
    }


def test_team_space_rejects_personal_setup_before_interpreting_the_path(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "team"
    store, _bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Team Lab")
    member = store.preprovision_team_member("Alice")
    app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, opened: opened.space_user(member.user_id),
    )
    submitted_path = tmp_path / "must-not-be-inspected-or-created"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("team setup reached the personal setup manager")

    monkeypatch.setattr(app.state.setup, "preflight", fail_if_called)
    monkeypatch.setattr(app.state.setup, "create", fail_if_called)
    monkeypatch.setattr(app.state.catalog, "register", fail_if_called)
    client = TestClient(app)
    payload = _local_payload(str(submitted_path))
    payload["confirmed"] = True
    payload["repositories"][0]["path"] = "/"

    for path in ("/api/project-setup/preflight", "/api/project-setup/create"):
        response = client.post(path, json=payload)
        assert response.status_code == 409
        assert response.json() == {
            "detail": (
                "Existing-checkout setup belongs to a personal space. "
                "Create a team-project provisioning request instead."
            )
        }

    registered = client.post("/api/projects", json={"locator": "/"})
    assert registered.status_code == 409
    assert registered.json() == {
        "detail": (
            "Existing-checkout setup belongs to a personal space. "
            "Create a team-project provisioning request instead."
        )
    }

    assert not submitted_path.exists()
    assert app.state.catalog.cards() == []
    assert app.state.background_tasks.store.project_provisioning_requests() == []


def test_local_wizard_preflights_without_writing_then_creates(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    payload = _local_payload(str(repository))

    invalid = client.post(
        "/api/project-setup/preflight",
        json={**payload, "default_auto_research_invocation_ceiling": 0},
    )
    assert invalid.status_code == 422

    legacy_key = client.post(
        "/api/project-setup/preflight",
        json={**payload, "default_campaign_invocation_ceiling": 10},
    )
    assert legacy_key.status_code == 422

    preview = client.post("/api/project-setup/preflight", json=payload)

    assert preview.status_code == 200
    assert preview.json()["action"] == "create"
    assert preview.json()["can_create"] is True
    assert "default_auto_research_invocation_ceiling = 10" in preview.json()["manifest_preview"]
    assert "default_campaign_invocation_ceiling" not in preview.json()["manifest_preview"]
    assert not (repository / ".research").exists()

    unconfirmed = client.post("/api/project-setup/create", json=payload)
    assert unconfirmed.status_code == 422
    assert not (repository / ".research").exists()

    payload["confirmed"] = True
    created = client.post("/api/project-setup/create", json=payload)

    assert created.status_code == 200
    assert created.json()["name"] == "wizard-paper"
    assert created.json()["revision"] == 1
    assert created.json()["home_space_id"] == app.state.space_id
    assert created.json()["reachable"] is True
    assert (repository / ".research" / "manifest.toml").is_file()
    assert (
        load_manifest(
            repository / ".research" / "manifest.toml"
        ).agent.default_auto_research_invocation_ceiling
        == 10
    )
    assert client.get("/api/projects").json()[0]["id"] == created.json()["id"]


def test_setup_records_discovered_provider_paths_in_new_manifest(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    app = create_app(data_dir=tmp_path / "data")

    class DiscoveringLauncher:
        @staticmethod
        def readiness(provider: str, *, host: str = "") -> ProviderReadiness:
            assert host == ""
            return ProviderReadiness(
                provider=provider,
                installed=True,
                authenticated=True,
                binary_path=f"/opt/rcp-test/{provider}",
                path_state="unconfigured",
            )

    app.state.setup.launcher = DiscoveringLauncher()
    client = TestClient(app)
    payload = _local_payload(str(repository))
    payload["confirmed"] = True

    created = client.post("/api/project-setup/create", json=payload)

    assert created.status_code == 200
    manifest = load_manifest(repository / ".research" / "manifest.toml")
    assert manifest.machine_map["laptop"].provider_paths == {
        "codex": "/opt/rcp-test/codex",
        "claude": "/opt/rcp-test/claude",
    }


def test_existing_local_manifest_is_connected_without_overwrite(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    payload = _local_payload(str(repository))
    payload["confirmed"] = True
    assert client.post("/api/project-setup/create", json=payload).status_code == 200
    manifest = repository / ".research" / "manifest.toml"
    original = manifest.read_text(encoding="utf-8")

    payload["name"] = "a-name-that-must-not-overwrite"
    payload["confirmed"] = False
    preview = client.post("/api/project-setup/preflight", json=payload)

    assert preview.status_code == 200
    assert preview.json()["action"] == "connect"
    assert preview.json()["can_create"] is False
    assert preview.json()["available_actions"] == ["open_existing"]
    archive_check = next(
        item for item in preview.json()["checks"] if item["label"] == "Archive existing research"
    )
    assert "registered in this RCP catalog" in archive_check["detail"]
    assert preview.json()["existing_project_name"] == "wizard-paper"
    assert manifest.read_text(encoding="utf-8") == original

    payload["existing_research_action"] = "open_existing"
    payload["confirmed"] = True
    connected = client.post("/api/project-setup/create", json=payload)

    assert connected.status_code == 200, connected.json()
    assert connected.json()["name"] == "wizard-paper"
    assert manifest.read_text(encoding="utf-8") == original


def test_existing_research_preflight_reports_exact_degraded_boundary_without_writing(
    manifest,
    tmp_path,
) -> None:
    history = HistoryManager(manifest)
    appended, _ = history.append(seed_patch())
    patch_path = manifest.research_dir / "patches" / f"{appended.revision:06d}.json"
    raw = json.loads(patch_path.read_text(encoding="utf-8"))
    raw["ops"][0]["nodes"][0]["type"] = "not-a-node-type"
    patch_path.write_text(json.dumps(raw), encoding="utf-8")
    before = {
        path.relative_to(manifest.research_dir): path.read_bytes()
        for path in manifest.research_dir.rglob("*")
        if path.is_file()
    }
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    canonical_repository = manifest.repository_map[manifest.state.repository]
    payload = _local_payload(canonical_repository.path)

    response = client.post("/api/project-setup/preflight", json=payload)

    assert response.status_code == 200, response.json()
    preview = response.json()
    assert preview["can_create"] is False
    assert preview["available_actions"] == [
        "open_degraded_read_only",
        "archive_and_create",
    ]
    existing = preview["existing_research"]
    assert existing["project_name"] == "test-paper"
    assert existing["canonical_location"] == str(manifest.research_dir)
    assert existing["retained_revision_count"] == 1
    assert existing["replay_status"] == "degraded"
    assert existing["coherent_revision"] == 0
    assert existing["replay_failure"]["revision"] == 1
    assert existing["replay_failure"]["code"] == "patch-schema-invalid"
    assert "not-a-node-type" in existing["replay_failure"]["message"]
    after = {
        path.relative_to(manifest.research_dir): path.read_bytes()
        for path in manifest.research_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert app.state.catalog.cards() == []


def test_degraded_existing_research_opens_last_coherent_state_without_claiming_home(
    manifest,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    history = HistoryManager(manifest)
    appended, _ = history.append(seed_patch())
    patch_path = manifest.research_dir / "patches" / f"{appended.revision:06d}.json"
    raw = json.loads(patch_path.read_text(encoding="utf-8"))
    raw["kind"] = "retired-patch-kind"
    patch_path.write_text(json.dumps(raw), encoding="utf-8")
    app = create_app(data_dir=data_dir)
    client = TestClient(app)
    canonical_repository = manifest.repository_map[manifest.state.repository]
    payload = _local_payload(canonical_repository.path)
    payload.update(
        {
            "confirmed": True,
            "existing_research_action": "open_degraded_read_only",
        }
    )

    created = client.post("/api/project-setup/create", json=payload)

    assert created.status_code == 200, created.json()
    assert created.json()["home_space_id"] is None
    opened = client.get(f"/api/projects/{created.json()['id']}")
    assert opened.status_code == 200, opened.json()
    snapshot = opened.json()
    assert snapshot["home_space_id"] is None
    assert snapshot["graph"]["replay_status"] == "degraded"
    assert snapshot["graph"]["revision"] == 0
    assert snapshot["graph"]["replay_failure"]["revision"] == 1
    assert snapshot["graph"]["replay_failure"]["code"] == "patch-schema-invalid"
    graph = client.get(f"/api/projects/{created.json()['id']}/graph")
    assert graph.status_code == 200, graph.json()
    assert graph.json()["revision"] == 0
    assert graph.json()["replay_failure"]["revision"] == 1
    history_slice = client.get(f"/api/projects/{created.json()['id']}/history")
    assert history_slice.status_code == 200, history_slice.json()
    assert history_slice.json() == []
    summaries = client.get(f"/api/projects/{created.json()['id']}/history/summaries")
    assert summaries.status_code == 200, summaries.json()
    assert summaries.json() == []

    restarted = TestClient(create_app(data_dir=data_dir))
    restarted_graph = restarted.get(f"/api/projects/{created.json()['id']}/graph")
    assert restarted_graph.status_code == 200, restarted_graph.json()
    assert restarted_graph.json()["replay_failure"]["code"] == "patch-schema-invalid"
    retained = json.loads(patch_path.read_text(encoding="utf-8"))
    assert retained.get("project_identity") is None
    assert list((manifest.research_dir / "patches").glob("*.json")) == [patch_path]


def test_archive_and_create_uses_new_setup_manifest_and_keeps_old_history(
    tmp_path,
) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    original_payload = _local_payload(str(repository))
    original_payload["confirmed"] = True
    original = client.post("/api/project-setup/create", json=original_payload)
    assert original.status_code == 200, original.json()
    source_cache, slice_cache = project_cache_roots(
        tmp_path / "data",
        original.json()["id"],
    )
    cached_source = source_cache / "remote" / "history.jsonl"
    cached_slice = slice_cache / "slice-one" / "records.jsonl"
    cached_source.parent.mkdir(parents=True)
    cached_slice.parent.mkdir(parents=True)
    cached_source.write_text("rebuildable", encoding="utf-8")
    cached_slice.write_text("rebuildable", encoding="utf-8")
    assert client.delete(f"/api/projects/{original.json()['id']}").status_code == 200
    assert not source_cache.parent.exists()
    old_manifest = (repository / ".research" / "manifest.toml").read_text(encoding="utf-8")
    payload = _local_payload(str(repository))
    payload.update(
        {
            "name": "fresh-paper",
            "confirmed": True,
            "existing_research_action": "archive_and_create",
        }
    )
    preview = client.post(
        "/api/project-setup/preflight",
        json={**payload, "existing_research_action": None},
    ).json()
    payload["existing_research_token"] = preview["existing_research"]["archive_token"]

    created = client.post("/api/project-setup/create", json=payload)

    assert created.status_code == 200, created.json()
    assert created.json()["name"] == "fresh-paper"
    fresh_manifest = repository / ".research" / "manifest.toml"
    assert load_manifest(fresh_manifest).name == "fresh-paper"
    archives = list(repository.glob(".research.archive-*"))
    assert len(archives) == 1
    assert (archives[0] / "manifest.toml").read_text(encoding="utf-8") == old_manifest
    assert (archives[0] / "patches" / "000001.json").is_file()


def test_archive_refuses_a_canonical_location_still_registered(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    payload = _local_payload(str(repository))
    payload["confirmed"] = True
    created = client.post("/api/project-setup/create", json=payload)
    assert created.status_code == 200, created.json()
    original_manifest = repository / ".research" / "manifest.toml"
    payload["existing_research_action"] = "archive_and_create"
    preview = client.post(
        "/api/project-setup/preflight",
        json={**payload, "existing_research_action": None},
    ).json()
    payload["existing_research_token"] = preview["existing_research"]["archive_token"]

    refused = client.post("/api/project-setup/create", json=payload)

    assert refused.status_code == 422
    assert "registered in this RCP catalog" in refused.json()["detail"]
    assert original_manifest.is_file()
    assert list(repository.glob(".research.archive-*")) == []


def test_archive_guard_canonicalizes_a_registered_repository_symlink(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    alias = tmp_path / "paper-alias"
    alias.symlink_to(repository, target_is_directory=True)
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    payload = _local_payload(str(repository))
    payload["confirmed"] = True
    created = client.post("/api/project-setup/create", json=payload)
    assert created.status_code == 200, created.json()

    aliased_payload = _local_payload(str(alias / ".." / alias.name))
    preview = client.post("/api/project-setup/preflight", json=aliased_payload)

    assert preview.status_code == 200, preview.json()
    assert preview.json()["canonical_location"] == str(repository / ".research")
    assert preview.json()["available_actions"] == ["open_existing"]
    archive_check = next(
        item for item in preview.json()["checks"] if item["label"] == "Archive existing research"
    )
    assert "registered in this RCP catalog" in archive_check["detail"]


def test_archive_refuses_when_retained_history_changed_after_preflight(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    payload = _local_payload(str(repository))
    payload["confirmed"] = True
    created = client.post("/api/project-setup/create", json=payload)
    assert created.status_code == 200, created.json()
    assert client.delete(f"/api/projects/{created.json()['id']}").status_code == 200

    preflight = client.post(
        "/api/project-setup/preflight",
        json={**payload, "confirmed": False},
    )
    assert preflight.status_code == 200, preflight.json()
    token = preflight.json()["existing_research"]["archive_token"]
    patch = repository / ".research" / "patches" / "000001.json"
    content = json.loads(patch.read_text(encoding="utf-8"))
    content["summary"] = "History moved after the human reviewed setup."
    patch.write_text(json.dumps(content), encoding="utf-8")
    payload.update(
        {
            "existing_research_action": "archive_and_create",
            "existing_research_token": token,
        }
    )

    refused = client.post("/api/project-setup/create", json=payload)

    assert refused.status_code == 503
    assert "changed since you reviewed it" in refused.json()["detail"]
    assert (repository / ".research" / "manifest.toml").is_file()
    assert list(repository.glob(".research.archive-*")) == []


def test_project_delete_refuses_symlinked_cache_root_without_touching_target(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    data_dir = tmp_path / "data"
    app = create_app(data_dir=data_dir)
    client = TestClient(app)
    payload = _local_payload(str(repository))
    payload["confirmed"] = True
    created = client.post("/api/project-setup/create", json=payload)
    assert created.status_code == 200, created.json()
    project_id = created.json()["id"]
    source_cache, _slice_cache = project_cache_roots(data_dir, project_id)
    stage = data_dir / "run-stage" / "saved-before-cache-validation"
    stage.mkdir(parents=True)
    stage_marker = stage / "patch.json"
    stage_marker.write_text("saved stage", encoding="utf-8")
    now = app.state.background_tasks.store.now()
    app.state.background_tasks.store.create_agent_task(
        AgentTaskRecord(
            operation_id="saved-before-cache-validation",
            project_id=project_id,
            kind="refresh",
            status="failed",
            request={},
            created_at=now,
            updated_at=now,
            status_message="failed",
            stage_root=str(stage),
        )
    )
    display = app.state.catalog._cached_snapshot_path(project_id)
    display.parent.mkdir(parents=True, exist_ok=True)
    display.write_text("saved display", encoding="utf-8")
    paper = app.state.catalog._paper_snapshot_path(project_id)
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("saved paper", encoding="utf-8")
    source_cache.parent.mkdir(parents=True)
    external = tmp_path / "external-cache"
    external.mkdir()
    marker = external / ".last-access.json"
    marker.write_text("must survive", encoding="utf-8")
    source_cache.symlink_to(external, target_is_directory=True)

    refused = client.delete(f"/api/projects/{project_id}")

    assert refused.status_code == 422
    assert "unsafe remote-source cache root" in refused.json()["detail"]
    assert marker.read_text(encoding="utf-8") == "must survive"
    assert stage_marker.read_text(encoding="utf-8") == "saved stage"
    assert display.read_text(encoding="utf-8") == "saved display"
    assert paper.read_text(encoding="utf-8") == "saved paper"
    assert app.state.background_tasks.store.agent_task("saved-before-cache-validation") is not None
    assert client.get("/api/projects").json()[0]["id"] == project_id


def test_connect_requires_confirmation_and_names_the_sole_writable_home(
    manifest,
    tmp_path,
) -> None:
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    canonical_repository = manifest.repository_map[manifest.state.repository]
    payload = _local_payload(canonical_repository.path)

    preview = client.post("/api/project-setup/preflight", json=payload)

    assert preview.status_code == 200
    assert preview.json()["action"] == "connect"
    canonical_check = next(
        item for item in preview.json()["checks"] if item["label"] == "Canonical manifest"
    )
    assert "active RCP space" in canonical_check["detail"]
    assert "sole writable home" in canonical_check["detail"]
    assert not (manifest.research_dir / "patches").exists()

    cancelled = client.post("/api/project-setup/create", json=payload)

    assert cancelled.status_code == 422
    assert not (manifest.research_dir / "patches").exists()

    payload["confirmed"] = True
    payload["existing_research_action"] = "open_existing"
    connected = client.post("/api/project-setup/create", json=payload)

    assert connected.status_code == 200
    assert connected.json()["home_space_id"] == app.state.space_id
    patches = HistoryManager(load_manifest(manifest.path)).load_patches()
    assert len(patches) == 1
    assert patches[0].kind == "identity"
    assert patches[0].project_identity is not None
    assert patches[0].project_identity.action == "adopted"


def test_wizard_rejects_blank_name_and_invalid_state_path(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)
    blank = _local_payload(str(repository))
    blank["name"] = "   "

    assert client.post("/api/project-setup/preflight", json=blank).status_code == 422

    (repository / ".research").write_text("not a directory", encoding="utf-8")
    preview = client.post("/api/project-setup/preflight", json=_local_payload(str(repository)))

    assert preview.status_code == 200
    assert preview.json()["can_create"] is False
    assert "not a directory" in preview.json()["checks"][1]["detail"]


def test_remote_preflight_checks_ssh_without_writing(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_ssh(host: str, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        assert host == "gpu.example"
        calls.append(arguments)
        if arguments[0] == "cat" or arguments[:2] in (["test", "-f"], ["test", "-e"]):
            return subprocess.CompletedProcess([], 1, "", "")
        return subprocess.CompletedProcess([], 0, "", "")

    class ReadyLauncher:
        @staticmethod
        def readiness(provider: str, *, host: str = "") -> ProviderReadiness:
            return ProviderReadiness(
                provider=provider,
                installed=True,
                authenticated=True,
                version=f"{provider}-test",
            )

    monkeypatch.setattr("rcp.setup._ssh", fake_ssh)
    app = create_app(data_dir=tmp_path / "data")
    app.state.setup.launcher = ReadyLauncher()
    request = ProjectSetupRequest.model_validate(
        {
            "name": "remote-paper",
            "repositories": [
                {
                    "alias": "remote-repo",
                    "location": "ssh",
                    "host": "gpu.example",
                    "path": "/srv/paper",
                    "default_read": True,
                }
            ],
            "state_repository": "remote-repo",
            "execution": {
                "location": "ssh",
                "host": "gpu.example",
            },
        }
    )

    preview = app.state.setup.preflight(request)

    assert preview.can_create is True
    assert preview.remote_write is True
    assert preview.canonical_location == "gpu.example:/srv/paper/.research"
    assert 'host = "gpu.example"' in preview.manifest_preview
    assert ["test", "-d", "/srv/paper"] in calls
    assert ["test", "-w", "/srv/paper"] in calls
    assert not any(arguments[0] in {"mkdir", "touch", "rm"} for arguments in calls)


def test_remote_first_identity_claim_persists_manifest_for_detection_after_delete(
    monkeypatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    remote_repository = tmp_path / "remote" / "paper"
    remote_repository.mkdir(parents=True)
    remote_root = remote_repository / ".research"
    request = ProjectSetupRequest.model_validate(
        {
            "name": "remote-paper",
            "repositories": [
                {
                    "alias": "remote-repo",
                    "location": "ssh",
                    "host": "gpu.example",
                    "path": "/srv/paper",
                    "default_read": True,
                }
            ],
            "state_repository": "remote-repo",
            "execution": {"location": "ssh", "host": "gpu.example"},
        }
    )
    bootstrap = data_dir / "bootstrap-manifests" / "remote-paper.toml"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text(render_manifest(request), encoding="utf-8")
    mirror = data_dir / "state-cache" / "remote-paper" / ".research"
    mirror.mkdir(parents=True)
    shutil.copyfile(bootstrap, mirror / "manifest.toml")

    class FilesystemRemoteWorkspace(StateWorkspace):
        def __init__(self, root: Path) -> None:
            super().__init__(root, "gpu.example:/srv/paper/.research")
            self.remote = True

        def _refresh_snapshot(self) -> bool:
            if not (remote_root / "manifest.toml").is_file():
                return False
            shutil.rmtree(self.root, ignore_errors=True)
            shutil.copytree(remote_root, self.root)
            self.reachable = True
            return True

        def publish_committed_patch(self, relative_paths, patch_path) -> None:
            assert Path(patch_path).as_posix().startswith("patches/")
            for raw_relative in relative_paths:
                relative = Path(raw_relative)
                source = self.root / relative
                if not source.is_file():
                    continue
                target = remote_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

    workspace = FilesystemRemoteWorkspace(mirror)
    monkeypatch.setattr(
        "rcp.projects.prepare_state_workspace",
        lambda _bootstrap, _data_dir: (load_manifest(mirror / "manifest.toml"), workspace),
    )
    app = create_app(data_dir=data_dir)
    record = app.state.catalog.register(str(bootstrap), identity_action="created")

    assert (remote_root / "manifest.toml").read_text(encoding="utf-8") == (
        mirror / "manifest.toml"
    ).read_text(encoding="utf-8")
    assert (remote_root / "patches" / "000001.json").is_file()
    app.state.catalog.delete(record.project_id)
    assert app.state.catalog.cards() == []

    def fake_ssh(host: str, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        assert host == "gpu.example"
        if arguments == ["cat", "/srv/paper/.research/manifest.toml"]:
            return subprocess.CompletedProcess(
                [],
                0,
                (remote_root / "manifest.toml").read_text(encoding="utf-8"),
                "",
            )
        return subprocess.CompletedProcess([], 0, "", "")

    probe = FilesystemRemoteWorkspace(data_dir / "probe-cache" / ".research")
    monkeypatch.setattr("rcp.setup._ssh", fake_ssh)
    monkeypatch.setattr("rcp.setup.state_workspace_for_probe", lambda *_args: probe)

    preview = app.state.setup.preflight(request)

    assert preview.action == "connect"
    assert preview.existing_research is not None
    assert preview.existing_research.project_name == "remote-paper"
    assert preview.existing_research.retained_revision_count == 1
    assert preview.existing_research.replay_status == "complete"
    assert preview.available_actions == ["open_existing", "archive_and_create"]


def test_wizard_manifest_records_each_agent_role_and_fixed_permissions(tmp_path) -> None:
    repository = tmp_path / "paper"
    repository.mkdir()
    request = ProjectSetupRequest.model_validate(
        {
            "name": "configured-paper",
            "repositories": [
                {
                    "alias": "paper-repo",
                    "location": "local",
                    "path": str(repository),
                    "default_read": True,
                }
            ],
            "state_repository": "paper-repo",
            "agents": {
                "seed": {
                    "provider": "claude",
                    "model": "claude-seed",
                    "reasoning": "high",
                    "location": "local",
                },
                "refresh": {
                    "provider": "codex",
                    "model": "codex-refresh",
                    "reasoning": "medium",
                    "location": "local",
                },
                "node_chat": {
                    "provider": "claude",
                    "model": "claude-node",
                    "reasoning": "low",
                    "location": "local",
                },
                "project_chat": {
                    "provider": "codex",
                    "runtime": "app-server",
                    "model": "codex-project",
                    "reasoning": "xhigh",
                    "location": "local",
                },
                "paper_coach": {
                    "provider": "claude",
                    "model": "claude-coach",
                    "reasoning": "medium",
                    "location": "local",
                },
            },
        }
    )
    research = repository / ".research"
    research.mkdir()
    path = research / "manifest.toml"
    rendered = render_manifest(request)
    path.write_text(rendered, encoding="utf-8")

    manifest = load_manifest(path)

    assert "[execution]" not in rendered
    assert "write_path" not in rendered
    assert manifest.agent_profile("seed").provider == "claude"
    # Setup writes the runtime it chose, and resolves an omitted one the same
    # way the manifest does rather than leaving the field out.
    assert manifest.agent_profile("seed").runtime == "stream-json"
    assert manifest.agent_profile("project_chat").runtime == "app-server"
    assert manifest.agent_profile("project_chat").reasoning == "xhigh"
    assert manifest.agent_profile("paper_coach").model == "claude-coach"
    assert manifest.agent_profile("paper_coach").permissions.write_graph_patch is False
    assert manifest.agent_profile("refresh").permissions.read_repositories == "run_scope"
    assert "[agent.orchestrator]" in rendered
    assert manifest.agent_profile("orchestrator").model == "codex-refresh"
    assert manifest.agent_profile("orchestrator").permissions == permissions_for("orchestrate")
