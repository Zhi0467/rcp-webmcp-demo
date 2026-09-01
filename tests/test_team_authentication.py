from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from rcp.api import create_app
from rcp.core.models import AuthorizedHuman
from rcp.limits import TEAM_CODE_FAILED_ATTEMPT_LIMIT, TEAM_SESSION_IDLE_DAYS
from rcp.storage import (
    AgentTaskRecord,
    AppStore,
    ProjectRecord,
    TeamAuthenticationError,
)


def _claimed_team(tmp_path):
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    member, token = store.enroll_team_member(bootstrap, "Alice")
    return store, bootstrap, member, token


def _enroll_invited_member(store: AppStore, creator_id: str, name: str):
    invitation, code = store.create_team_invitation(creator_id)
    member, token = store.enroll_team_member(code, name)
    return invitation, code, member, token


def _sqlite_bytes(path) -> bytes:
    payload = b""
    for candidate in (path, path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm")):
        if candidate.exists():
            payload += candidate.read_bytes()
    return payload


def test_bootstrap_is_not_issued_before_late_schema_work_succeeds(tmp_path, monkeypatch) -> None:
    def fail_late_schema_work(*_args, **_kwargs) -> None:
        raise sqlite3.OperationalError("injected late schema failure")

    monkeypatch.setattr(AppStore, "_ensure_column", fail_late_schema_work)
    with pytest.raises(sqlite3.OperationalError, match="injected late schema failure"):
        AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")

    assert not (tmp_path / "rcp.sqlite3").exists()

    monkeypatch.undo()
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    assert store.space_kind == "team"
    assert bootstrap.startswith("rcp_bootstrap_")


def test_bootstrap_claim_is_atomic_and_single_use(tmp_path) -> None:
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")

    def claim(name: str):
        try:
            return store.enroll_team_member(bootstrap, name)
        except TeamAuthenticationError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, ("Alice", "Bob")))

    successes = [result for result in results if isinstance(result, tuple)]
    failures = [result for result in results if isinstance(result, str)]
    assert len(successes) == 1
    assert failures == ["enrollment_code_consumed"]
    assert len(store.space_users()) == 1
    with pytest.raises(TeamAuthenticationError) as reused:
        store.enroll_team_member(bootstrap, "Charlie")
    assert reused.value.code == "enrollment_code_consumed"


def test_invitations_are_creator_private_expiring_single_use_credentials(tmp_path) -> None:
    store, _bootstrap, alice, _alice_token = _claimed_team(tmp_path)
    alice_first, first_code = store.create_team_invitation(alice.user_id)
    _alice_second, second_code = store.create_team_invitation(alice.user_id)
    bob, _bob_token = store.enroll_team_member(first_code, "Same name")
    bob_invitation, _bob_code = store.create_team_invitation(bob.user_id)

    assert alice.user_id != bob.user_id
    assert bob.display_name == "Same name"
    assert {item.invitation_id for item in store.team_invitations(alice.user_id)} == {
        alice_first.invitation_id,
        _alice_second.invitation_id,
    }
    assert [item.invitation_id for item in store.team_invitations(bob.user_id)] == [
        bob_invitation.invitation_id
    ]
    with pytest.raises(TeamAuthenticationError) as reused:
        store.enroll_team_member(first_code, "Another person")
    assert reused.value.code == "enrollment_code_consumed"

    second_id = second_code.split(".", 1)[0].removeprefix("rcp_invite_")
    expired_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with store.connection() as connection:
        connection.execute(
            "UPDATE team_invitations SET expires_at = ? WHERE invitation_id = ?",
            (expired_at, second_id),
        )
    with pytest.raises(TeamAuthenticationError) as expired:
        store.enroll_team_member(second_code, "Expired")
    assert expired.value.code == "enrollment_code_expired"


def test_wrong_attempts_lock_only_the_target_invitation_code(tmp_path) -> None:
    store, _bootstrap, alice, _alice_token = _claimed_team(tmp_path)
    _target, target_code = store.create_team_invitation(alice.user_id)
    _other, other_code = store.create_team_invitation(alice.user_id)
    public = target_code.split(".", 1)[0]
    wrong_code = f"{public}.{'x' * 43}"

    for attempt in range(TEAM_CODE_FAILED_ATTEMPT_LIMIT):
        with pytest.raises(TeamAuthenticationError) as rejected:
            store.enroll_team_member(wrong_code, f"Guess {attempt}")
        expected = (
            "enrollment_code_locked"
            if attempt == TEAM_CODE_FAILED_ATTEMPT_LIMIT - 1
            else "enrollment_code_invalid"
        )
        assert rejected.value.code == expected

    with pytest.raises(TeamAuthenticationError) as locked:
        store.enroll_team_member(target_code, "Locked out")
    assert locked.value.code == "enrollment_code_locked"
    enrolled, _token = store.enroll_team_member(other_code, "Unaffected")
    assert enrolled.display_name == "Unaffected"


def test_member_tokens_are_prefixed_sha256_indexed_and_compared_in_constant_time(
    tmp_path, monkeypatch
) -> None:
    store, _bootstrap, member, token = _claimed_team(tmp_path)
    assert re.fullmatch(r"rcp_[A-Za-z0-9_-]{43}", token)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with store.connection() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(team_member_tokens)")}
        row = connection.execute(
            "SELECT token_hash FROM team_member_tokens WHERE user_id = ?", (member.user_id,)
        ).fetchone()
        indexed_columns = {
            column[2]
            for index in connection.execute("PRAGMA index_list(team_member_tokens)")
            for column in connection.execute(f"PRAGMA index_info({index[1]})")
        }
    assert columns == {"token_id", "user_id", "token_hash", "created_at", "revoked_at"}
    assert row["token_hash"] == token_hash
    assert re.fullmatch(r"[a-f0-9]{64}", row["token_hash"])
    assert "token_hash" in indexed_columns
    assert token.encode() not in _sqlite_bytes(store.path)

    comparisons: list[tuple[str, str]] = []
    real_compare = hmac.compare_digest

    def observed_compare(left: str, right: str) -> bool:
        comparisons.append((left, right))
        return real_compare(left, right)

    # Token comparison lives in the space/team half of the store.
    monkeypatch.setattr("rcp.storage.spaces.hmac.compare_digest", observed_compare)
    _session, resolved = store.create_team_session(token)
    assert resolved == member
    assert comparisons == [(token_hash, token_hash)]


def test_team_sessions_are_hashed_server_rows_with_fourteen_day_sliding_expiry(tmp_path) -> None:
    store, _bootstrap, member, token = _claimed_team(tmp_path)
    session, resolved = store.create_team_session(token)
    assert resolved == member
    assert session.startswith("rcp_session_")
    session_hash = hashlib.sha256(session.encode()).hexdigest()
    assert AppStore(store.path).resolve_team_session(session) == member
    forced_expiry = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    with store.connection() as connection:
        row = connection.execute(
            "SELECT session_hash, expires_at FROM team_sessions WHERE user_id = ?",
            (member.user_id,),
        ).fetchone()
        assert row["session_hash"] == session_hash
        connection.execute(
            "UPDATE team_sessions SET expires_at = ? WHERE session_hash = ?",
            (forced_expiry, session_hash),
        )
    assert session.encode() not in _sqlite_bytes(store.path)

    before_resolution = datetime.now(UTC)
    assert store.resolve_team_session(session) == member
    with store.connection() as connection:
        slid = connection.execute(
            "SELECT expires_at FROM team_sessions WHERE session_hash = ?", (session_hash,)
        ).fetchone()[0]
    slid_expiry = datetime.fromisoformat(slid)
    assert slid_expiry > datetime.fromisoformat(forced_expiry)
    assert slid_expiry >= before_resolution + timedelta(days=TEAM_SESSION_IDLE_DAYS - 1)

    with store.connection() as connection:
        connection.execute(
            "UPDATE team_sessions SET expires_at = ? WHERE session_hash = ?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), session_hash),
        )
    assert store.resolve_team_session(session) is None
    with store.connection() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM team_sessions WHERE session_hash = ?", (session_hash,)
            ).fetchone()
            is None
        )


@pytest.mark.parametrize("operations", [("rotate", "rotate"), ("revoke", "rotate")])
def test_credential_mutations_atomically_revalidate_the_presenting_session(
    tmp_path, operations
) -> None:
    store, _bootstrap, member, token = _claimed_team(tmp_path)
    _enroll_invited_member(store, member.user_id, "Other member")
    session, _resolved = store.create_team_session(token)
    barrier = threading.Barrier(2)

    def mutate(operation: str):
        barrier.wait()
        if operation == "rotate":
            return store.rotate_team_token(
                member.user_id,
                authenticating_session=session,
            )
        store.revoke_team_token(
            member.user_id,
            authenticating_session=session,
        )
        return None

    def capture(operation: str):
        try:
            return ("ok", mutate(operation))
        except TeamAuthenticationError as exc:
            return (exc.code, None)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(capture, operations))

    assert [status for status, _value in results].count("ok") == 1
    assert [status for status, _value in results].count("team_session_invalid") == 1
    assert store.resolve_team_session(session) is None
    with pytest.raises(TeamAuthenticationError):
        store.create_team_session(token)
    replacement = next((value for status, value in results if status == "ok" and value), None)
    if replacement is not None:
        _new_session, resolved = store.create_team_session(replacement)
        assert resolved == member


def test_rotation_and_revocation_are_member_scoped_and_preserve_authorized_work(tmp_path) -> None:
    store, _bootstrap, alice, alice_token = _claimed_team(tmp_path)
    _invite, _code, bob, bob_token = _enroll_invited_member(store, alice.user_id, "Bob")
    alice_session, _ = store.create_team_session(alice_token)
    bob_session, _ = store.create_team_session(bob_token)
    now = store.now()
    project_id = str(uuid.uuid4())
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name="Authorized work",
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at=now,
        )
    )
    operation_id = str(uuid.uuid4())
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="refresh",
            status="running",
            request={"safe": True},
            created_at=now,
            updated_at=now,
            status_message="Running.",
            authorized_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=bob.user_id,
                display_name="Bob",
            ),
        )
    )
    before = store.agent_task(operation_id)

    replacement = store.rotate_team_token(bob.user_id)
    assert replacement != bob_token
    assert store.resolve_team_session(bob_session) is None
    assert store.resolve_team_session(alice_session) == alice
    with pytest.raises(TeamAuthenticationError):
        store.create_team_session(bob_token)
    assert store.agent_task(operation_id) == before

    replacement_session, _ = store.create_team_session(replacement)
    store.revoke_team_token(bob.user_id)
    assert store.resolve_team_session(replacement_session) is None
    assert store.resolve_team_session(alice_session) == alice
    with pytest.raises(TeamAuthenticationError):
        store.create_team_session(replacement)
    assert store.agent_task(operation_id) == before


def test_self_service_revoke_refuses_to_strand_the_last_enrolled_member(tmp_path) -> None:
    store, _bootstrap, member, token = _claimed_team(tmp_path)
    app = create_app(data_dir=tmp_path)
    client = TestClient(app, base_url="https://team.test")
    assert client.post("/api/team/session/exchange", json={"token": token}).status_code == 200

    refused = client.post("/api/team/credential/revoke", json={})

    assert refused.status_code == 409
    assert "last enrolled member" in refused.json()["detail"]
    assert client.get("/api/identity").json()["user"]["user_id"] == member.user_id
    rotated = client.post("/api/team/credential/rotate", json={})
    assert rotated.status_code == 200
    assert rotated.json()["token"].startswith("rcp_")
    assert store.space_user(member.user_id) is not None


def test_raw_credentials_never_enter_sqlite_or_task_and_patch_fixtures(tmp_path) -> None:
    store, bootstrap, alice, alice_token = _claimed_team(tmp_path)
    _invitation, invite_code, bob, bob_token = _enroll_invited_member(store, alice.user_id, "Bob")
    session, _member = store.create_team_session(bob_token)
    rotated = store.rotate_team_token(bob.user_id)
    now = store.now()
    project_id = str(uuid.uuid4())
    store.upsert_project(
        ProjectRecord(
            project_id=project_id,
            home_space_id=store.space_id,
            locator=f"/tmp/{project_id}/research.yaml",
            name="Redaction",
            state_location=f"/tmp/{project_id}/.research",
            state_remote=False,
            added_at=now,
        )
    )
    operation_id = str(uuid.uuid4())
    store.create_agent_task(
        AgentTaskRecord(
            operation_id=operation_id,
            project_id=project_id,
            kind="refresh",
            status="running",
            request={"source": "team enrollment redaction test"},
            created_at=now,
            updated_at=now,
            status_message="Running.",
            authorized_by=AuthorizedHuman(
                space_id=store.space_id,
                user_id=alice.user_id,
                display_name="Alice",
            ),
        )
    )
    store.record_agent_task_event(operation_id, "Enrollment complete.")
    store.record_agent_task_receipt(operation_id, "test", {"status": "safe"})
    contract = "Use authenticated human attribution."
    store.record_agent_task_contract(
        operation_id,
        "test",
        contract,
        hashlib.sha256(contract.encode()).hexdigest(),
    )
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({"operations": []}), encoding="utf-8")

    durable_bytes = _sqlite_bytes(store.path)
    fixture_bytes = json.dumps(
        {
            "task": store.agent_task(operation_id).model_dump(mode="json"),
            "events": [
                item.model_dump(mode="json") for item in store.agent_task_events(operation_id)
            ],
            "receipts": [
                item.model_dump(mode="json") for item in store.agent_task_receipts(operation_id)
            ],
            "contracts": [
                item.model_dump(mode="json") for item in store.agent_task_contracts(operation_id)
            ],
            "patch": json.loads(patch.read_text(encoding="utf-8")),
        },
        sort_keys=True,
    ).encode()
    for secret in (bootstrap, invite_code, alice_token, bob_token, session, rotated):
        assert secret.encode() not in durable_bytes
        assert secret.encode() not in fixture_bytes


def test_team_authentication_middleware_keeps_only_bootstrap_boundaries_public(tmp_path) -> None:
    store, _bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    app = create_app(data_dir=tmp_path)
    client = TestClient(app, base_url="https://testserver")

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["space_name"] == "Team Lab"
    assert client.get("/").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert (
        client.post(
            "/api/team/enroll", json={"code": "invalid", "display_name": "Alice"}
        ).status_code
        == 401
    )
    assert client.post("/api/team/session/exchange", json={"token": "invalid"}).status_code == 401
    oversized = "x" * 5000
    for path, body in (
        ("/api/team/enroll", {"code": oversized, "display_name": "Alice"}),
        ("/api/team/session/exchange", {"token": oversized}),
    ):
        response = client.post(path, json=body)
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "team_auth_request_too_large"

    for path in ("/api", "/api/identity", "/api/projects", "/api/team/invitations"):
        response = client.get(path)
        assert response.status_code == 401, (path, response.text)
        assert response.json()["detail"]["code"] == "team_identity_required"
    assert store.space_users() == []


def test_authenticated_team_mutations_reject_forms_and_cross_origin_json(tmp_path) -> None:
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    client = TestClient(create_app(data_dir=tmp_path), base_url="https://team.test")
    enrollment = client.post(
        "/api/team/enroll", json={"code": bootstrap, "display_name": "Alice"}
    ).json()
    token = enrollment["token"]
    assert client.post("/api/team/session/exchange", json={"token": token}).status_code == 200

    mutation_paths = (
        "/api/team/session/logout",
        "/api/team/invitations",
        "/api/team/credential/rotate",
        "/api/team/credential/revoke",
    )
    for path in mutation_paths:
        forged_form = client.post(
            path,
            data={"forged": "true"},
            headers={"Origin": "https://team.test"},
        )
        assert forged_form.status_code == 415
        assert forged_form.json()["detail"]["code"] == "team_json_required"

        forged_json = client.post(
            path,
            json={},
            headers={"Origin": "https://attacker.test"},
        )
        assert forged_json.status_code == 403
        assert forged_json.json()["detail"]["code"] == "team_origin_invalid"

    assert client.get("/api/identity").status_code == 200
    assert client.get("/api/team/invitations").json() == []
    fresh_client = TestClient(create_app(data_dir=tmp_path), base_url="https://team.test")
    assert fresh_client.post("/api/team/session/exchange", json={"token": token}).status_code == 200
    assert store.space_kind == "team"
    assert len(store.space_users()) == 1


def test_authenticated_team_attachment_upload_keeps_its_bounded_multipart_contract(
    manifest, tmp_path
) -> None:
    data_dir = tmp_path / "data"
    store, bootstrap = AppStore.initialize_team_space(data_dir / "rcp.sqlite3", "Team Lab")
    app = create_app(str(manifest.path), data_dir=data_dir)
    client = TestClient(app, base_url="https://team.test")
    token = client.post(
        "/api/team/enroll", json={"code": bootstrap, "display_name": "Alice"}
    ).json()["token"]
    assert client.post("/api/team/session/exchange", json={"token": token}).status_code == 200
    project_id = app.state.default_project_id
    chat_id = str(uuid.uuid4())
    client_id = str(uuid.uuid4())
    path = f"/api/projects/{project_id}/chats/{chat_id}/attachments"

    forged = client.post(
        path,
        data={"client_id": client_id},
        files={"file": ("notes.txt", b"temporary input", "text/plain")},
        headers={"Origin": "https://attacker.test"},
    )
    assert forged.status_code == 403
    assert forged.json()["detail"]["code"] == "team_origin_invalid"

    uploaded = client.post(
        path,
        data={"client_id": client_id},
        files={"file": ("notes.txt", b"temporary input", "text/plain")},
        headers={"Origin": "https://team.test"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["attachment"]["name"] == "notes.txt"
    assert store.space_kind == "team"


def test_enrollment_exchange_and_session_cookie_make_the_team_api_usable(tmp_path) -> None:
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    app = create_app(data_dir=tmp_path)
    client = TestClient(app, base_url="https://testserver")

    enrolled = client.post("/api/team/enroll", json={"code": bootstrap, "display_name": "Alice"})
    assert enrolled.status_code == 200
    enrollment_payload = enrolled.json()
    token = enrollment_payload.pop("token")
    member = enrollment_payload["identity"]["user"]
    assert member["display_name"] == "Alice"
    assert bootstrap not in enrolled.text
    assert client.get("/api/identity").status_code == 401

    exchanged = client.post("/api/team/session/exchange", json={"token": token})
    assert exchanged.status_code == 200
    assert token not in exchanged.text
    cookie = exchanged.headers["set-cookie"].lower()
    assert cookie.startswith("__host-rcp_session=rcp_session_")
    for attribute in (
        "httponly",
        "secure",
        "samesite=lax",
        "path=/",
        "max-age=1209600",
    ):
        assert attribute in cookie
    # The __Host- prefix is only honoured when the cookie carries no Domain,
    # which is what keeps a team session from being scoped to a sibling host.
    assert "domain=" not in cookie

    identity = client.get("/api/identity")
    assert identity.status_code == 200
    assert identity.json()["user"]["user_id"] == member["user_id"]
    assert "max-age=1209600" in identity.headers["set-cookie"].lower()
    assert client.get("/api/projects").status_code == 200

    restarted = TestClient(create_app(data_dir=tmp_path), base_url="https://testserver")
    restarted_identity = restarted.get(
        "/api/identity",
        headers={"Cookie": f"__Host-rcp_session={client.cookies.get('__Host-rcp_session')}"},
    )
    assert restarted_identity.status_code == 200
    assert restarted_identity.json()["user"]["user_id"] == member["user_id"]

    invitation = client.post("/api/team/invitations", json={})
    assert invitation.status_code == 200
    assert invitation.json()["space_name"] == "Team Lab"
    assert invitation.json()["invitation"]["expires_at"]
    assert invitation.json()["code"].startswith("rcp_invite_")
    listed = client.get("/api/team/invitations")
    assert listed.status_code == 200
    assert [item["invitation_id"] for item in listed.json()] == [
        invitation.json()["invitation"]["invitation_id"]
    ]
    assert invitation.json()["code"] not in listed.text
    assert AppStore(store.path).space_user(member["user_id"]) is not None


def test_team_members_see_only_their_invitations_and_cannot_target_credentials(
    tmp_path,
) -> None:
    store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    app = create_app(data_dir=tmp_path)
    alice_client = TestClient(app, base_url="https://testserver")
    bob_client = TestClient(app, base_url="https://testserver")
    alice_enrollment = alice_client.post(
        "/api/team/enroll", json={"code": bootstrap, "display_name": "Same name"}
    ).json()
    alice_id = alice_enrollment["identity"]["user"]["user_id"]
    alice_token = alice_enrollment["token"]
    assert (
        alice_client.post("/api/team/session/exchange", json={"token": alice_token}).status_code
        == 200
    )
    alice_invitation = alice_client.post("/api/team/invitations", json={}).json()
    bob_enrollment = bob_client.post(
        "/api/team/enroll",
        json={"code": alice_invitation["code"], "display_name": "Same name"},
    ).json()
    bob_id = bob_enrollment["identity"]["user"]["user_id"]
    bob_token = bob_enrollment["token"]
    assert alice_id != bob_id
    assert (
        bob_client.post("/api/team/session/exchange", json={"token": bob_token}).status_code == 200
    )

    assert bob_client.get("/api/team/invitations").json() == []
    bob_invitation = bob_client.post("/api/team/invitations", json={}).json()
    assert {item["invitation_id"] for item in alice_client.get("/api/team/invitations").json()} == {
        alice_invitation["invitation"]["invitation_id"]
    }
    assert {item["invitation_id"] for item in bob_client.get("/api/team/invitations").json()} == {
        bob_invitation["invitation"]["invitation_id"]
    }
    renamed = bob_client.patch("/api/team/space", json={"name": "Renamed by Bob"})
    assert renamed.json() == {"space_name": "Renamed by Bob"}
    assert alice_client.get("/api/identity").json()["space_name"] == "Renamed by Bob"

    rotated = alice_client.post("/api/team/credential/rotate", json={})
    assert rotated.status_code == 200
    alice_replacement = rotated.json()["token"]
    assert alice_client.get("/api/identity").status_code == 401
    assert bob_client.get("/api/identity").json()["user"]["user_id"] == bob_id
    assert (
        TestClient(app, base_url="https://testserver")
        .post("/api/team/session/exchange", json={"token": bob_token})
        .status_code
        == 200
    )
    assert (
        TestClient(app, base_url="https://testserver")
        .post("/api/team/session/exchange", json={"token": alice_token})
        .status_code
        == 401
    )

    assert (
        alice_client.post(
            "/api/team/session/exchange", json={"token": alice_replacement}
        ).status_code
        == 200
    )
    revoked = alice_client.post("/api/team/credential/revoke", json={})
    assert revoked.status_code == 200
    assert alice_client.get("/api/identity").status_code == 401
    assert bob_client.get("/api/identity").json()["user"]["user_id"] == bob_id
    assert store.space_user(alice_id).identity_kind == "team_member"
    assert store.space_user(bob_id).identity_kind == "team_member"


def test_trusted_principal_resolver_remains_a_supported_team_authentication_path(
    tmp_path,
) -> None:
    store, _bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Team Lab")
    first = store.preprovision_team_member("Resolver member")
    second = store.preprovision_team_member("Other member")
    selected = [first.user_id]
    app = create_app(
        data_dir=tmp_path,
        trusted_principal_resolver=lambda _request, _store: selected[0],
    )
    client = TestClient(app)

    assert client.get("/api/identity").json()["user"]["user_id"] == first.user_id
    selected[0] = second.user_id
    assert client.get("/api/identity").json()["user"]["user_id"] == second.user_id
    assert client.post("/api/team/invitations").status_code == 200


def test_personal_space_keeps_its_local_owner_without_team_authentication(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    owner = app.state.background_tasks.store.local_owner
    assert owner is not None

    identity = client.get("/api/identity")
    assert identity.status_code == 200
    assert identity.json()["space_kind"] == "personal"
    assert identity.json()["space_name"] is None
    assert identity.json()["user"]["user_id"] == owner.user_id
    assert client.get("/api/projects").status_code == 200
    for path, body in (
        ("/api/team/enroll", {"code": "unused", "display_name": "Person"}),
        ("/api/team/session/exchange", {"token": "rcp_unused"}),
    ):
        assert client.post(path, json=body).status_code == 404
    assert app.state.background_tasks.store.local_owner == owner
