"""S122 — someone puts you on the project, and you can leave it.

S101 makes membership exist and enforces it. This is how it *changes*, including
what becomes of an agent that was running on the authorization of someone who
has left.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rcp.api.app import create_app
from rcp.core.authority import AgentDispatchAuthority, AgentDispatchScope
from rcp.core.models import AuthorizedHuman
from rcp.core.transition_models import GraphHeadRef, GraphTargetRef
from rcp.runs.membership_fence import fence_episodes_for_departed_member
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    AutoResearchStateRecord,
    EpisodeRecord,
)

from .test_project_membership import _create_project, _team_app


def _invite(client: TestClient, project_id: str, user_id: str):
    return client.post(f"/api/projects/{project_id}/invitations", json={"user_id": user_id})


# --- invitation, acceptance, refusal -----------------------------------------


def test_a_project_invitation_appears_on_the_project_index(manifest, tmp_path) -> None:
    _app, client, _store, people, acting = _team_app(tmp_path, members=3)
    creator, invitee, _third = people
    project_id = _create_project(client, tmp_path / "repo")

    assert _invite(client, project_id, invitee.user_id).status_code == 201

    acting[0] = invitee.user_id
    # Before accepting, the project is not theirs, but the invitation is visible.
    assert client.get("/api/projects").json() == []
    pending = client.get("/api/project-invitations").json()
    assert len(pending) == 1
    assert pending[0]["project_id"] == project_id
    assert pending[0]["invited_by"] == creator.user_id
    assert pending[0]["invited_by_name"] == "Member 0"
    assert pending[0]["project_name"]


def test_accepting_an_invitation_grants_project_membership(manifest, tmp_path) -> None:
    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, invitee, _third = people
    project_id = _create_project(client, tmp_path / "repo")
    _invite(client, project_id, invitee.user_id)

    acting[0] = invitee.user_id
    invitation_id = client.get("/api/project-invitations").json()[0]["invitation_id"]
    accepted = client.post(f"/api/project-invitations/{invitation_id}/accept")

    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["response"] == "accepted"
    assert store.is_project_member(project_id, invitee.user_id)
    assert [card["id"] for card in client.get("/api/projects").json()] == [project_id]
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    assert client.get("/api/project-invitations").json() == []


def test_accepting_issues_no_token_and_does_not_change_space_membership(manifest, tmp_path) -> None:
    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, invitee, _third = people
    project_id = _create_project(client, tmp_path / "repo")

    def tokens_and_users() -> tuple[list[tuple], list[str]]:
        with store.connection() as connection:
            tokens = connection.execute(
                "SELECT token_id, user_id, token_hash, revoked_at FROM team_member_tokens"
            ).fetchall()
            users = connection.execute("SELECT user_id FROM space_users ORDER BY user_id")
            return [tuple(row) for row in tokens], [row["user_id"] for row in users]

    before = tokens_and_users()
    _invite(client, project_id, invitee.user_id)
    acting[0] = invitee.user_id
    invitation_id = client.get("/api/project-invitations").json()[0]["invitation_id"]
    body = client.post(f"/api/project-invitations/{invitation_id}/accept").json()

    assert tokens_and_users() == before
    # The record itself carries no secret of any kind.
    assert set(body) == {
        "invitation_id",
        "project_id",
        "invited_user_id",
        "invited_by",
        "created_at",
        "response",
        "responded_at",
    }


def test_every_project_member_may_invite_another_space_member(manifest, tmp_path) -> None:
    """No approval chain and no rank: a member seated by invitation may invite."""

    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, second, third = people
    project_id = _create_project(client, tmp_path / "repo")
    _invite(client, project_id, second.user_id)

    acting[0] = second.user_id
    invitation_id = client.get("/api/project-invitations").json()[0]["invitation_id"]
    client.post(f"/api/project-invitations/{invitation_id}/accept")

    # The newly seated member invites, with no elevated role anywhere.
    assert _invite(client, project_id, third.user_id).status_code == 201
    acting[0] = third.user_id
    assert len(client.get("/api/project-invitations").json()) == 1
    assert not store.is_project_member(project_id, third.user_id)


def test_project_members_have_no_ranks_and_no_owner(manifest, tmp_path) -> None:
    _app, client, store, people, _acting = _team_app(tmp_path, members=3)
    project_id = _create_project(client, tmp_path / "repo")
    _invite(client, project_id, people[1].user_id)

    members = client.get(f"/api/projects/{project_id}/members").json()
    assert members, "the creator should be seated"
    for member in members:
        assert set(member) == {"user_id", "display_name", "seated_at"}
        assert "role" not in member and "owner" not in member


def test_an_invitation_cannot_be_addressed_to_a_non_member_of_the_space(manifest, tmp_path) -> None:
    _app, client, _store, _people, _acting = _team_app(tmp_path)
    project_id = _create_project(client, tmp_path / "repo")

    stranger = "11111111-1111-4111-8111-111111111111"
    refused = _invite(client, project_id, stranger)

    assert refused.status_code == 404
    assert "space" in refused.json()["detail"]


def test_declining_leaves_no_membership_and_no_residual_access(manifest, tmp_path) -> None:
    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, invitee, _third = people
    project_id = _create_project(client, tmp_path / "repo")
    _invite(client, project_id, invitee.user_id)

    acting[0] = invitee.user_id
    invitation_id = client.get("/api/project-invitations").json()[0]["invitation_id"]
    declined = client.post(f"/api/project-invitations/{invitation_id}/decline")

    assert declined.status_code == 200
    assert declined.json()["response"] == "declined"
    assert not store.is_project_member(project_id, invitee.user_id)
    assert client.get("/api/projects").json() == []
    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert client.get("/api/project-invitations").json() == []
    # A declined invitation cannot be answered again into acceptance.
    assert client.post(f"/api/project-invitations/{invitation_id}/accept").status_code == 409
    assert not store.is_project_member(project_id, invitee.user_id)


def test_the_server_derives_membership_and_never_reads_it_from_the_request_body(
    manifest, tmp_path
) -> None:
    """The body names only the invitee; the inviter comes from the session."""

    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, outsider, target = people
    project_id = _create_project(client, tmp_path / "repo")

    acting[0] = outsider.user_id
    # A non-member cannot invite, and cannot claim to be one in the payload.
    forged = client.post(
        f"/api/projects/{project_id}/invitations",
        json={"user_id": target.user_id, "invited_by": people[0].user_id},
    )
    assert forged.status_code == 404
    assert store.pending_project_invitations(target.user_id) == []


# --- leaving ------------------------------------------------------------------


def test_leaving_removes_read_dispatch_and_apply(manifest, tmp_path) -> None:
    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, leaver, _third = people
    project_id = _create_project(client, tmp_path / "repo")
    _invite(client, project_id, leaver.user_id)

    acting[0] = leaver.user_id
    invitation_id = client.get("/api/project-invitations").json()[0]["invitation_id"]
    client.post(f"/api/project-invitations/{invitation_id}/accept")
    assert client.get(f"/api/projects/{project_id}").status_code == 200

    left = client.post(f"/api/projects/{project_id}/leave")

    assert left.status_code == 204
    assert not store.is_project_member(project_id, leaver.user_id)
    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert client.get("/api/projects").json() == []
    assert client.post(f"/api/projects/{project_id}/tasks/seed", json={}).status_code == 404
    assert store.agent_tasks(project_id) == []


def test_the_only_member_cannot_leave_the_project(manifest, tmp_path) -> None:
    _app, client, store, people, _acting = _team_app(tmp_path)
    creator = people[0]
    project_id = _create_project(client, tmp_path / "repo")
    assert len(store.project_members(project_id)) == 1

    refused = client.post(f"/api/projects/{project_id}/leave")

    assert refused.status_code == 409
    assert refused.json()["detail"] == (
        "You are the only member of this project. Add another member before leaving."
    )
    assert store.is_project_member(project_id, creator.user_id)


# --- the fence ----------------------------------------------------------------


class _RecordingStopper:
    """Stands in for Auto-research Stop, which is a module function now."""

    def __init__(self, store: AppStore) -> None:
        self.store = store
        self.stopped: list[str] = []

    def stop_auto_research(self, _tasks, episode_id: str):
        self.stopped.append(episode_id)
        return self.store.request_episode_stop(episode_id)


def _running_auto_research_episode(
    store: AppStore,
    project_id: str,
    authorizer: AuthorizedHuman,
    episode_id: str = "episode-one",
) -> EpisodeRecord:
    """One live Auto-research episode with unspent invocations."""

    now = store.now()
    episode = EpisodeRecord(
        episode_id=episode_id,
        project_id=project_id,
        mode="auto_research",
        graph_target=GraphTargetRef(kind="branch", branch_id=episode_id),
        graph_base_head=GraphHeadRef(revision=0),
        status="queued",
        invocation_ceiling=12,
        authorized_by=authorizer,
        created_at=now,
        updated_at=now,
    )
    root = AgentTaskRecord(
        operation_id=f"{episode_id}-root",
        project_id=project_id,
        episode_id=episode_id,
        graph_target=GraphTargetRef(kind="branch", branch_id=episode_id),
        kind="auto_research",
        status="queued",
        request={
            "episode_id": episode_id,
            "role": "orchestrator",
            "actor_operation_id": f"{episode_id}-root",
            "run_truth_scope": ["repo"],
        },
        created_at=now,
        updated_at=now,
        status_message="queued",
        authorized_by=authorizer,
        dispatch_authority=AgentDispatchAuthority(
            profile="orchestrator",
            task_contract="orchestrate",
            scope=AgentDispatchScope(
                run_truth_scope=["repo"],
                episode_id=episode_id,
                patch_kind="work",
            ),
        ),
    )
    store.create_auto_research_episode_with_root_task(
        episode,
        AutoResearchStateRecord(
            episode_id=episode_id,
            starting_instruction="Investigate the evidence.",
            created_at=now,
            updated_at=now,
        ),
        root,
    )
    # Spend one invocation through the real adapter, which is what moves the
    # episode to `running` and leaves the rest of the ceiling unspent.
    store.complete_agent_task(root.operation_id, applied_revision=None, result={})
    store.create_auto_research_agent_task(
        root.model_copy(
            update={
                "operation_id": f"{episode_id}-turn",
                "parent_operation_id": root.operation_id,
                "status": "queued",
                "request": {**root.request, "actor_operation_id": root.operation_id},
            }
        ),
        role="orchestrator",
    )
    running = store.episode(episode_id)
    assert running is not None and running.status == "running"
    assert running.invocations_used < running.invocation_ceiling
    return running


def _seat_second_member(client, acting, project_id: str, invitee_id: str) -> None:
    _invite(client, project_id, invitee_id)
    was = acting[0]
    acting[0] = invitee_id
    invitation_id = client.get("/api/project-invitations").json()[0]["invitation_id"]
    client.post(f"/api/project-invitations/{invitation_id}/accept")
    acting[0] = was


def test_losing_membership_fences_the_episode_like_stop(manifest, tmp_path, monkeypatch) -> None:
    """The same durable Stop request, not a second mechanism."""

    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, leaver, _third = people
    project_id = _create_project(client, tmp_path / "repo")
    _seat_second_member(client, acting, project_id, leaver.user_id)

    authorizer = AuthorizedHuman(
        space_id=store.space_id, user_id=leaver.user_id, display_name="Member 1"
    )
    episode = _running_auto_research_episode(store, project_id, authorizer)
    assert episode.stop_requested_at is None

    stopper = _RecordingStopper(store)
    monkeypatch.setattr("rcp.runs.membership_fence.stop_auto_research", stopper.stop_auto_research)
    fenced = fence_episodes_for_departed_member(store, stopper, project_id, leaver.user_id)

    assert fenced == [episode.episode_id]
    assert stopper.stopped == [episode.episode_id]
    fenced_record = store.episode(episode.episode_id)
    assert fenced_record is not None
    assert fenced_record.stop_requested_at is not None
    assert fenced_record.status in {"stopping", "stopped"}


def test_leaving_through_the_route_fences_and_survives_a_restart(manifest, tmp_path) -> None:
    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, leaver, _third = people
    project_id = _create_project(client, tmp_path / "repo")
    _seat_second_member(client, acting, project_id, leaver.user_id)

    authorizer = AuthorizedHuman(
        space_id=store.space_id, user_id=leaver.user_id, display_name="Member 1"
    )
    episode = _running_auto_research_episode(store, project_id, authorizer)

    acting[0] = leaver.user_id
    assert client.post(f"/api/projects/{project_id}/leave").status_code == 204

    reopened = AppStore(store.path)
    survived = reopened.episode(episode.episode_id)
    assert survived is not None
    assert survived.stop_requested_at is not None
    assert not reopened.is_project_member(project_id, leaver.user_id)
    # The turn that was already authorized keeps its record; the fence never
    # deletes work, it only refuses new admissions.
    assert reopened.auto_research_tasks(episode.episode_id)


def test_the_fence_touches_no_other_project_and_no_other_member(manifest, tmp_path) -> None:
    _app, client, store, people, acting = _team_app(tmp_path, members=3)
    creator, leaver, _third = people
    mine_project = _create_project(client, tmp_path / "repo-mine", name="mine")
    theirs_project = _create_project(client, tmp_path / "repo-theirs", name="theirs")
    _seat_second_member(client, acting, mine_project, leaver.user_id)

    mine = _running_auto_research_episode(
        store,
        mine_project,
        AuthorizedHuman(space_id=store.space_id, user_id=leaver.user_id, display_name="Member 1"),
        episode_id="mine",
    )
    theirs = _running_auto_research_episode(
        store,
        theirs_project,
        AuthorizedHuman(space_id=store.space_id, user_id=creator.user_id, display_name="Member 0"),
        episode_id="theirs",
    )

    fenced = fence_episodes_for_departed_member(
        store, _RecordingStopper(store), mine_project, leaver.user_id
    )

    assert fenced == [mine.episode_id]
    untouched = store.episode(theirs.episode_id)
    assert untouched is not None
    assert untouched.stop_requested_at is None
    assert untouched.status == "running"


def test_revoking_a_token_does_not_fence_running_work(manifest, tmp_path) -> None:
    """Revocation is about a credential; it must not kill a week-long episode."""

    data_dir = tmp_path / "team"
    store, bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Team Lab")
    app = create_app(data_dir=data_dir)
    client = TestClient(app, base_url="https://team.test")
    token = client.post(
        "/api/team/enroll", json={"code": bootstrap, "display_name": "Alice"}
    ).json()["token"]
    client.post("/api/team/session/exchange", json={"token": token})
    project_id = _create_project(client, tmp_path / "repo")

    alice = store.space_users()[0]
    _invitation, bob_code = store.create_team_invitation(alice.user_id)
    bob, _bob_token = store.enroll_team_member(bob_code, "Bob")
    store.seat_project_member(project_id, bob.user_id, seated_by=alice.user_id)
    episode = _running_auto_research_episode(
        store,
        project_id,
        AuthorizedHuman(space_id=store.space_id, user_id=alice.user_id, display_name="Alice"),
    )

    revoked = client.post("/api/team/credential/revoke", json={})
    assert revoked.status_code in {200, 204}, revoked.text

    still_running = store.episode(episode.episode_id)
    assert still_running is not None
    assert still_running.stop_requested_at is None
    assert still_running.status == "running"
    # Membership is untouched by revocation: it is about a credential, not a project.
    assert store.is_project_member(project_id, alice.user_id)


def test_no_agent_path_writes_a_membership_row(manifest, tmp_path) -> None:
    """Membership is project truth, so only a human action may change it."""

    agent_owned = (
        Path("src/rcp/agents"),
        Path("src/rcp/runs"),
        Path("src/rcp/background.py"),
    )
    writers = ("seat_project_member", "answer_project_invitation", "invite_to_project")
    offenders = []
    for root in agent_owned:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            text = path.read_text(encoding="utf-8")
            for writer in writers:
                if writer in text:
                    offenders.append(f"{path}: {writer}")
    assert offenders == []


# --- S100's deferred revocation drive ------------------------------------------


def test_membership_lost_between_dispatch_and_apply_is_refused_at_apply(manifest, tmp_path) -> None:
    """S100 could not demonstrate a permission that changes. Membership is one."""

    app, client, store, people, acting = _team_app(tmp_path, members=3)
    _creator, leaver, _third = people
    project_id = _create_project(client, tmp_path / "repo")
    _seat_second_member(client, acting, project_id, leaver.user_id)

    catalog = app.state.catalog
    history = catalog.open(project_id).history
    assert history.project_membership_check is not None

    # Authorized at dispatch...
    assert history.project_membership_check(project_id, leaver.user_id)

    acting[0] = leaver.user_id
    assert client.post(f"/api/projects/{project_id}/leave").status_code == 204

    # ...and refused at Apply, which reads membership live under the append lock.
    assert not history.project_membership_check(project_id, leaver.user_id)


def test_revoking_a_token_mid_run_refuses_nothing_and_fences_nothing(manifest, tmp_path) -> None:
    data_dir = tmp_path / "team"
    store, bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Team Lab")
    app = create_app(data_dir=data_dir)
    client = TestClient(app, base_url="https://team.test")
    token = client.post(
        "/api/team/enroll", json={"code": bootstrap, "display_name": "Alice"}
    ).json()["token"]
    client.post("/api/team/session/exchange", json={"token": token})
    project_id = _create_project(client, tmp_path / "repo")
    alice = store.space_users()[0]
    _invitation, bob_code = store.create_team_invitation(alice.user_id)
    bob, _bob_token = store.enroll_team_member(bob_code, "Bob")
    store.seat_project_member(project_id, bob.user_id, seated_by=alice.user_id)

    assert client.post("/api/team/credential/revoke", json={}).status_code in {200, 204}

    # Revocation is about a credential, so neither membership nor Apply moves.
    assert store.is_project_member(project_id, alice.user_id)
    history = app.state.catalog.open(project_id).history
    assert history.project_membership_check is not None
    assert history.project_membership_check(project_id, alice.user_id)


def test_the_whole_flow_works_through_a_real_browser_session(manifest, tmp_path) -> None:
    """Drive invite, accept, and leave through the session middleware.

    The tests above use a trusted principal resolver, which bypasses the team
    JSON/origin middleware entirely — so they cannot see a bodyless mutation
    being refused with 415. A real session can, and did.
    """

    data_dir = tmp_path / "team"
    store, bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Team Lab")
    app = create_app(data_dir=data_dir)
    alice = TestClient(app, base_url="https://team.test")
    alice_token = alice.post(
        "/api/team/enroll", json={"code": bootstrap, "display_name": "Alice"}
    ).json()["token"]
    alice.post("/api/team/session/exchange", json={"token": alice_token})
    invite_code = alice.post("/api/team/invitations", json={}).json()["code"]

    bob = TestClient(app, base_url="https://team.test")
    bob_token = bob.post(
        "/api/team/enroll", json={"code": invite_code, "display_name": "Bob"}
    ).json()["token"]
    bob.post("/api/team/session/exchange", json={"token": bob_token})
    bob_id = next(user.user_id for user in store.space_users() if user.display_name == "Bob")

    project_id = _create_project(alice, tmp_path / "repo")
    assert bob.get("/api/projects").json() == []

    invited = alice.post(f"/api/projects/{project_id}/invitations", json={"user_id": bob_id})
    assert invited.status_code == 201, invited.text

    invitation_id = bob.get("/api/project-invitations").json()[0]["invitation_id"]
    # Pinned because the first client shipped without it: a bodyless mutation is
    # refused, which is exactly what makes JSON-only the CSRF protection.
    bodyless = bob.post(f"/api/project-invitations/{invitation_id}/accept")
    assert bodyless.status_code == 415
    assert bodyless.json()["detail"]["code"] == "team_json_required"

    accepted = bob.post(f"/api/project-invitations/{invitation_id}/accept", json={})
    assert accepted.status_code == 200, accepted.text
    assert [card["id"] for card in bob.get("/api/projects").json()] == [project_id]

    left = bob.post(f"/api/projects/{project_id}/leave", json={})
    assert left.status_code == 204, left.text
    assert bob.get("/api/projects").json() == []
    assert not store.is_project_member(project_id, bob_id)
