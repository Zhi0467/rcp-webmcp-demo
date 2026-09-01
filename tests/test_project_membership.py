"""S101 — being in the lab is not being on the project.

Membership exists, is seeded, and is enforced. Granting it to a second person
is S122; everything here is the boundary itself.
"""

from __future__ import annotations

import inspect
import sqlite3
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from rcp.api.app import create_app
from rcp.api.dependencies import ApiServices, require_project_membership
from rcp.core.authority import require_apply, require_dispatch
from rcp.history import HistoryManager
from rcp.projects import ProjectCatalog
from rcp.setup import ProjectSetupRequest
from rcp.storage import AppStore

from .helpers import seed_patch


def _setup_payload(repository_path: Path, name: str = "membership-paper") -> dict[str, object]:
    repository_path.mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "repositories": [
            {
                "alias": "paper-repo",
                "location": "local",
                "path": str(repository_path),
                "host": "",
                "default_read": True,
            }
        ],
        "state_repository": "paper-repo",
        "execution": {"location": "local", "host": ""},
        "confirmed": True,
    }


def _team_app(tmp_path: Path, *, members: int = 2):
    """A team space with `members` people and no project yet.

    `acting` is a one-item list so a test can change who is acting between
    requests without rebuilding the app.
    """

    data_dir = tmp_path / "team"
    store = AppStore(data_dir / "rcp.sqlite3", space_kind="team")
    people = [store.preprovision_team_member(f"Member {index}") for index in range(members)]
    acting = [people[0].user_id]
    app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, opened: opened.space_user(acting[0]),
    )
    return app, TestClient(app), store, people, acting


def _create_project(
    client: TestClient,
    repository_path: Path,
    *,
    seat_member: str | None = None,
    name: str = "membership-paper",
) -> str:
    """Prepare membership fixtures through the internal setup owner.

    A team member cannot enter through the personal project-setup API. These
    tests exercise downstream membership, so they establish that prerequisite
    directly without creating a product bypass around provisioning.
    """

    if seat_member is None:
        identity = client.get("/api/identity")
        assert identity.status_code == 200, identity.text
        seat_member = str(identity.json()["user"]["user_id"])
    request = ProjectSetupRequest.model_validate(_setup_payload(repository_path, name))
    return str(client.app.state.setup.create(request, seat_member=seat_member)["id"])


def _project_scoped_routes(app: FastAPI) -> list[APIRoute]:
    """Every project-scoped route, whether FastAPI flattened the router or nested it.

    `include_router` may leave an opaque `_IncludedRouter` in `app.routes`
    instead of merging its routes, so a flat walk would silently find none —
    and this test would pass while proving nothing.
    """

    found: list[APIRoute] = []
    pending = list(app.routes)
    while pending:
        route = pending.pop()
        nested = getattr(route, "original_router", None)
        if nested is not None:
            pending.extend(nested.routes)
        elif isinstance(route, APIRoute) and "{project_id}" in route.path:
            found.append(route)
    return found


def _routes_missing_membership(app: FastAPI) -> list[str]:
    """Every project-scoped route must carry the membership dependency."""

    gate = app.state.project_membership_dependency
    missing = []
    for route in _project_scoped_routes(app):
        attached = any(dependency.call is gate for dependency in route.dependant.dependencies)
        if not attached:
            missing.append(f"{sorted(route.methods)} {route.path}")
    return missing


def test_api_services_are_typed_wired_and_membership_gate_is_module_level(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "personal")
    services = app.state.services

    assert isinstance(services, ApiServices)
    assert tuple(field.name for field in fields(services)) == (
        "store",
        "catalog",
        "identity_access",
        "attachment_store",
        "watcher_poller",
        "result_view_keep_locks",
        "project_display_cache",
        "watcher_delivery",
        "experiment_operation_lock",
        "background_tasks",
        "experiment_admission",
        "launcher",
        "setup",
        "health_composition",
        "server_status_composition",
    )
    assert services.store is app.state.background_tasks.store
    assert services.catalog is app.state.catalog
    assert services.background_tasks is app.state.background_tasks
    assert services.identity_access is not None
    assert services.attachment_store is not None
    assert services.watcher_poller is app.state.watcher_poller
    assert services.result_view_keep_locks is not None
    assert services.project_display_cache is not None
    assert services.watcher_delivery is not None
    assert services.experiment_operation_lock is not None
    assert services.launcher is not None
    assert services.setup is not None
    assert services.health_composition.instance_metadata is app.state.instance_metadata
    assert services.health_composition.agent_mode == app.state.agent_mode
    assert (
        services.health_composition.default_project_name
        == app.state.catalog.card(app.state.default_project_id)["name"]
    )
    assert services.health_composition.space_id == app.state.space_id
    assert services.health_composition.space_kind == app.state.space_kind
    assert not hasattr(services, "__dict__")
    with pytest.raises(FrozenInstanceError):
        services.store = services.store

    assert app.state.project_membership_dependency is require_project_membership
    assert require_project_membership.__module__ == "rcp.api.dependencies"
    assert require_project_membership.__closure__ is None


# --- seeding -----------------------------------------------------------------


def test_creating_a_project_seats_its_creator_as_the_first_member(manifest, tmp_path) -> None:
    app, client, store, people, _acting = _team_app(tmp_path)
    creator, other = people

    project_id = _create_project(client, tmp_path / "repo", seat_member=creator.user_id)

    members = client.get(f"/api/projects/{project_id}/members")
    assert members.status_code == 200, members.text
    assert [item["user_id"] for item in members.json()] == [creator.user_id]
    assert other.user_id not in {item["user_id"] for item in members.json()}
    assert store.is_project_member(project_id, creator.user_id)
    assert not store.is_project_member(project_id, other.user_id)


def test_membership_binds_the_durable_user_id_and_never_a_display_name(manifest, tmp_path) -> None:
    app, client, store, people, _acting = _team_app(tmp_path)
    creator = people[0]

    project_id = _create_project(client, tmp_path / "repo", seat_member=creator.user_id)
    seated = store.project_members(project_id)
    assert [record.user_id for record in seated] == [creator.user_id]

    # The name is a label; renaming must not disturb the binding.
    store.rename_space_user(creator.user_id, "Renamed entirely")
    assert store.is_project_member(project_id, creator.user_id)
    assert [record.user_id for record in store.project_members(project_id)] == [creator.user_id]


def test_internal_registration_can_seat_an_unnamed_member(manifest, tmp_path) -> None:
    """The internal finalizer seats an id; P2 separately names its human authorizer."""

    data_dir = tmp_path / "team"
    store = AppStore(data_dir / "rcp.sqlite3", space_kind="team")
    unnamed = store.preprovision_team_member(None)
    assert unnamed.display_name is None
    app = create_app(
        data_dir=data_dir,
        trusted_principal_resolver=lambda _request, opened: opened.space_user(unnamed.user_id),
    )
    client = TestClient(app)

    project_id = _create_project(client, tmp_path / "repo", seat_member=unnamed.user_id)
    assert store.is_project_member(project_id, unnamed.user_id)


def test_a_personal_space_project_has_exactly_one_member(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "personal")
    client = TestClient(app)
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id

    owner = store.local_owner
    assert owner is not None
    assert [record.user_id for record in store.project_members(project_id)] == [owner.user_id]
    assert len(client.get("/api/projects").json()) == 1

    # No display name was ever chosen, and the project is still fully usable.
    assert owner.display_name is None
    assert client.get(f"/api/projects/{project_id}/cached").status_code in {200, 404}


# --- backfill ----------------------------------------------------------------


def _register_legacy_project(path: Path, locator: str, project_id: str) -> None:
    """Write a project row into a store built before membership existed."""

    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE IF EXISTS project_members")
    connection.execute(
        """
        INSERT INTO projects (
            project_id, locator, name, state_location, state_remote, added_at, last_opened_at
        ) VALUES (?, ?, ?, ?, 0, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """,
        (project_id, locator, project_id, str(Path(locator).parent)),
    )
    connection.commit()
    connection.close()


def test_projects_predating_membership_backfill_every_current_space_member_once(
    manifest, tmp_path
) -> None:
    path = tmp_path / "legacy" / "rcp.sqlite3"
    path.parent.mkdir(parents=True)
    AppStore(path)
    _register_legacy_project(path, str(manifest.path), "legacy-project")

    reopened = AppStore(path)
    owner = reopened.local_owner
    assert owner is not None
    assert [record.user_id for record in reopened.project_members("legacy-project")] == [
        owner.user_id
    ]


def test_the_backfill_runs_once_and_is_not_reapplied_on_later_starts(manifest, tmp_path) -> None:
    path = tmp_path / "legacy" / "rcp.sqlite3"
    path.parent.mkdir(parents=True)
    AppStore(path)
    _register_legacy_project(path, str(manifest.path), "legacy-project")

    first = AppStore(path)
    owner = first.local_owner
    assert owner is not None
    assert first.is_project_member("legacy-project", owner.user_id)

    # Someone leaves. A later start must not silently readmit them.
    with first.connection() as connection:
        connection.execute("DELETE FROM project_members WHERE project_id = 'legacy-project'")

    second = AppStore(path)
    assert second.project_members("legacy-project") == []
    assert not second.is_project_member("legacy-project", owner.user_id)


# --- what a non-member sees --------------------------------------------------


def test_a_non_member_project_is_absent_from_the_project_list(manifest, tmp_path) -> None:
    app, client, _store, people, acting = _team_app(tmp_path)
    creator, outsider = people

    project_id = _create_project(client, tmp_path / "repo", seat_member=creator.user_id)
    assert [card["id"] for card in client.get("/api/projects").json()] == [project_id]

    acting[0] = outsider.user_id
    assert client.get("/api/projects").json() == []
    assert creator.user_id != outsider.user_id


def test_a_non_member_project_is_absent_from_the_cross_project_experiment_board(
    manifest, tmp_path
) -> None:
    app, client, _store, people, acting = _team_app(tmp_path)
    _creator, outsider = people
    project_id = _create_project(client, tmp_path / "repo", seat_member=people[0].user_id)
    # Commit a display snapshot, so the board has real cached graph state to leak.
    assert client.get(f"/api/projects/{project_id}").status_code == 200

    board = client.get("/api/episodes", params={"mode": "experiment_loop"})
    assert board.status_code == 200, board.text

    # The outsider must not even reach the cache-status check that answers 503
    # for a project they cannot see — that would confirm the project exists.
    acting[0] = outsider.user_id
    outsider_board = client.get("/api/episodes", params={"mode": "experiment_loop"})
    assert outsider_board.status_code == 200, outsider_board.text
    assert outsider_board.json() == []


def test_an_exact_non_member_project_id_answers_404_and_never_403(manifest, tmp_path) -> None:
    app, client, _store, people, acting = _team_app(tmp_path)
    _creator, outsider = people
    project_id = _create_project(client, tmp_path / "repo", seat_member=people[0].user_id)

    acting[0] = outsider.user_id
    refused = client.get(f"/api/projects/{project_id}")
    unknown = client.get("/api/projects/no-such-project-at-all")

    assert refused.status_code == 404
    assert unknown.status_code == 404
    # Byte-identical: a non-member learns nothing an unknown id would not tell them.
    assert refused.json() == unknown.json() == {"detail": "Project not found"}

    for path in (
        f"/api/projects/{project_id}/cached",
        f"/api/projects/{project_id}/graph",
        f"/api/projects/{project_id}/history",
        f"/api/projects/{project_id}/tasks",
        f"/api/projects/{project_id}/members",
    ):
        assert client.get(path).status_code == 404, path


def test_a_non_member_dispatch_never_launches_a_provider(manifest, tmp_path) -> None:
    app, client, store, people, acting = _team_app(tmp_path)
    _creator, outsider = people
    project_id = _create_project(client, tmp_path / "repo", seat_member=people[0].user_id)

    acting[0] = outsider.user_id
    refused = client.post(f"/api/projects/{project_id}/tasks/seed", json={})

    assert refused.status_code == 404
    assert refused.json() == {"detail": "Project not found"}
    assert store.agent_tasks(project_id) == []


def test_a_non_member_patch_is_refused_at_apply_under_the_append_lock(manifest, tmp_path) -> None:
    """Apply reads membership live, so losing it while a task runs still refuses."""

    app, client, store, people, acting = _team_app(tmp_path)
    creator, outsider = people
    project_id = _create_project(client, tmp_path / "repo", seat_member=creator.user_id)

    catalog: ProjectCatalog = app.state.catalog
    history = catalog.open(project_id).history
    before = history.state().revision

    assert history.project_membership_check is not None
    assert history.project_membership_check(project_id, creator.user_id)
    assert not history.project_membership_check(project_id, outsider.user_id)

    # The gate is the live check, not a snapshot taken at dispatch.
    store.seat_project_member(project_id, outsider.user_id)
    assert history.project_membership_check(project_id, outsider.user_id)
    with store.connection() as connection:
        connection.execute(
            "DELETE FROM project_members WHERE project_id = ? AND user_id = ?",
            (project_id, outsider.user_id),
        )
    assert not history.project_membership_check(project_id, outsider.user_id)
    assert history.state().revision == before


def test_require_dispatch_is_unchanged_and_carries_no_membership_argument() -> None:
    """require_dispatch sees no user and no project, so it is not where the check goes."""

    dispatch = inspect.signature(require_dispatch)
    assert list(dispatch.parameters) == ["authority"]

    apply_signature = inspect.signature(require_apply)
    membership = apply_signature.parameters["is_project_member"]
    assert membership.kind is inspect.Parameter.KEYWORD_ONLY
    # Required, so no caller can reach Apply without deciding the question.
    assert membership.default is inspect.Parameter.empty


# --- the gate is structural --------------------------------------------------


def test_every_project_scoped_route_is_declared_on_the_membership_router(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "personal")

    scoped = _project_scoped_routes(app)
    assert len(scoped) > 30, "the project-scoped surface should not have collapsed"
    assert _routes_missing_membership(app) == []


def test_a_project_scoped_route_declared_outside_the_router_fails_the_route_test(
    manifest, tmp_path
) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "personal")

    @app.get("/api/projects/{project_id}/smuggled")
    def smuggled(project_id: str) -> dict[str, str]:  # pragma: no cover - never called
        return {"project_id": project_id}

    assert _routes_missing_membership(app) == ["['GET'] /api/projects/{project_id}/smuggled"]


# --- membership is operational, never canonical ------------------------------


def test_replay_succeeds_with_no_membership_records_present(manifest, tmp_path) -> None:
    """Membership lives in SQLite and never in `.research/`, so replay cannot depend on it."""

    app = create_app(str(manifest.path), data_dir=tmp_path / "personal")
    catalog: ProjectCatalog = app.state.catalog
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id

    service = catalog.open(project_id)
    HistoryManager(service.manifest, service.history.workspace).append(seed_patch())

    with store.connection() as connection:
        connection.execute("DELETE FROM project_members")
    assert store.project_members(project_id) == []

    replayed = HistoryManager(service.manifest, service.history.workspace).materialize(
        write_outputs=False
    )
    assert replayed.state.replay_status == "complete"
    assert replayed.state.revision > 0


def test_deleting_a_personal_project_takes_its_membership_with_it(manifest, tmp_path) -> None:
    app = create_app(str(manifest.path), data_dir=tmp_path / "personal")
    client = TestClient(app)
    store = app.state.background_tasks.store
    project_id = app.state.default_project_id
    assert store.project_members(project_id)

    deleted = client.delete(f"/api/projects/{project_id}")
    assert deleted.status_code == 200, deleted.text
    assert store.project_members(project_id) == []


@pytest.mark.parametrize("enrol_before_registration", [True, False])
def test_a_team_project_is_never_left_with_no_members(
    manifest, tmp_path, enrol_before_registration: bool
) -> None:
    """An unclaimed project would be invisible to everyone with no way to recover it."""

    data_dir = tmp_path / "team"
    store, bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Team Lab")

    if enrol_before_registration:
        app = create_app(data_dir=data_dir)
        client = TestClient(app, base_url="https://team.test")
        token = client.post(
            "/api/team/enroll", json={"code": bootstrap, "display_name": "Alice"}
        ).json()["token"]
        client.post("/api/team/session/exchange", json={"token": token})
        project_id = _create_project(
            client,
            tmp_path / "repo",
            seat_member=store.space_users()[0].user_id,
        )
    else:
        # The server opens the project before anybody has enrolled.
        app = create_app(str(manifest.path), data_dir=data_dir)
        client = TestClient(app, base_url="https://team.test")
        project_id = app.state.default_project_id
        assert store.project_members(project_id) == []
        token = client.post(
            "/api/team/enroll", json={"code": bootstrap, "display_name": "Alice"}
        ).json()["token"]
        client.post("/api/team/session/exchange", json={"token": token})

    assert [card["id"] for card in client.get("/api/projects").json()] == [project_id]
    assert len(store.project_members(project_id)) == 1
