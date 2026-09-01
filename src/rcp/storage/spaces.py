from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Literal

from rcp.limits import (
    MEMBER_REMOVAL_PREVIEW_MAX_ITEMS,
    TEAM_CODE_FAILED_ATTEMPT_LIMIT,
    TEAM_INVITATION_TTL_DAYS,
    TEAM_MEMBER_TOKEN_MAX_LENGTH,
    TEAM_SESSION_IDLE_DAYS,
    TEAM_SESSION_TOKEN_MAX_LENGTH,
    WATCHER_GROUP_DIAGNOSTIC_ERROR_COUNT,
)
from rcp.storage.episodes import _LIVE_EPISODE_STATUSES
from rcp.storage.models import (  # noqa: F401
    _EXPERIMENT_EPISODE_CONTEXT_CANDIDATE_ROLE,
    _EXPERIMENT_EPISODE_PINNED_FIELDS,
    _MISSING_EXPERIMENT_EPISODE_CONTEXT_DIAGNOSTIC,
    _PROJECT_ID_TABLES,
    ACTIVE_AGENT_TASK_STATUSES,
    SPACE_NAME_MAX_LENGTH,
    AgentCommandInvocationRecord,
    AgentTaskContractRecord,
    AgentTaskEventRecord,
    AgentTaskKind,
    AgentTaskReceiptRecord,
    AgentTaskReceiptTier,
    AgentTaskRecord,
    AgentTaskStatus,
    AgentUsageCell,
    AgentUsageCountReason,
    AgentUsageMetric,
    AgentUsageRecord,
    AgentUsageSnapshot,
    ChatSessionContextRecord,
    ExperimentEpisodeRecord,
    ExperimentLoopRuntime,
    ExperimentWatcherResourceRecord,
    GraphCondition,
    GraphWatcherRecord,
    MemberRemovalPreviewRecord,
    NodeStatusGraphCondition,
    ProjectInvitationRecord,
    ProjectMemberRecord,
    ProjectRecord,
    ProjectStageRecord,
    ProposalResolvedGraphCondition,
    ProviderSkillInventoryRecord,
    ResultViewConflict,
    ResultViewRecord,
    SpaceKind,
    SpaceUserKind,
    SpaceUserRecord,
    StoredWatcherRecord,
    TeamAuthenticationError,
    TeamInvitationRecord,
    TeamMemberAuthorityRecord,
    WatcherClaimConflict,
    WatcherContinuation,
    WatcherDeliveryRecord,
    WatcherRecord,
    WatcherStatus,
    WatcherStopRequest,
    _canonical_space_id,
    _canonical_uuid4,
    _discard_failed_team_initialization,
    _experiment_pinned_value,
    _new_enrollment_code,
    _new_member_token,
    _new_session_token,
    _optional_str,
    _parse_enrollment_code,
    _plain_html_name,
    _required_timestamp,
    _result_view_html_bytes,
    _result_view_is_visible,
    _result_view_reference_time,
    _sha256,
    _stored_space_kind,
    _validated_result_view_html,
    normalize_space_name,
    watcher_next_check_at,
)


def _bounded_removal_values(rows, column: str, *, label: str) -> tuple[str, ...]:
    values = tuple(sorted({str(row[column]) for row in rows}))
    if len(values) > MEMBER_REMOVAL_PREVIEW_MAX_ITEMS:
        raise RuntimeError(f"The member-removal {label} exceeds its safe preview bound.")
    return values


class SpaceStoreMixin:
    """Space identity, team enrollment, browser sessions, and member tokens."""

    @property
    def space_id(self) -> str:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT space_id FROM space_identity WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("RCP space identity is unavailable.")
        return _canonical_space_id(row["space_id"])

    @property
    def space_kind(self) -> SpaceKind:
        with self.connection() as connection:
            return self._space_kind_from_connection(connection)

    @property
    def space_name(self) -> str | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT space_name FROM space_identity WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("RCP space identity is unavailable.")
        value = row["space_name"]
        if value is None:
            return None
        try:
            return normalize_space_name(value)
        except ValueError as exc:
            raise RuntimeError("RCP space name is invalid.") from exc

    def space_users(self) -> list[SpaceUserRecord]:
        with self.connection() as connection:
            return self._space_users_from_connection(connection)

    def space_user(self, user_id: str) -> SpaceUserRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM space_users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._space_user_record(row) if row is not None else None

    def active_team_member_authority(self) -> tuple[TeamMemberAuthorityRecord, ...]:
        """Return every active member and permanent token id in one read boundary."""

        with self.connection() as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                """
                SELECT user.user_id, user.display_name, token.token_id
                FROM space_users AS user
                JOIN team_member_tokens AS token ON token.user_id = user.user_id
                WHERE user.identity_kind = 'team_member'
                  AND user.removal_started_at IS NULL
                  AND user.removed_at IS NULL
                  AND token.revoked_at IS NULL
                ORDER BY user.user_id, token.token_id
                """
            ).fetchall()
        members: dict[str, dict[str, object]] = {}
        for row in rows:
            member_id = str(row["user_id"])
            member = members.setdefault(
                member_id,
                {"display_name": row["display_name"], "active_token_ids": []},
            )
            token_ids = member["active_token_ids"]
            assert isinstance(token_ids, list)
            token_ids.append(str(row["token_id"]))
        return tuple(
            TeamMemberAuthorityRecord(
                member_id=member_id,
                display_name=(
                    str(values["display_name"]) if values["display_name"] is not None else None
                ),
                active_token_ids=tuple(values["active_token_ids"]),
            )
            for member_id, values in members.items()
        )

    def member_removal_preview(self, user_id: str) -> MemberRemovalPreviewRecord:
        """Read one complete consequence set without changing member authority."""

        _canonical_uuid4(user_id, label="user identity")
        with self.connection() as connection:
            connection.execute("BEGIN")
            return self._member_removal_preview_from_connection(connection, user_id)

    def begin_member_removal(
        self,
        user_id: str,
        *,
        expected_boundary_sha256: str,
    ) -> MemberRemovalPreviewRecord:
        """Atomically fence access and revoke every member-owned live capability."""

        _canonical_uuid4(user_id, label="user identity")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            preview = self._member_removal_preview_from_connection(connection, user_id)
            if preview.member.removal_started_at is not None:
                return preview
            if preview.boundary_sha256 != expected_boundary_sha256:
                raise ValueError(
                    "The member-removal consequences changed after preview; rerun the command."
                )
            if preview.last_authenticating_member:
                raise ValueError(
                    "Removing this member would leave no other enrolled member who can "
                    "authenticate. Enroll another member first."
                )
            if preview.orphaned_project_ids:
                projects = self._project_labels(connection, preview.orphaned_project_ids)
                raise ValueError(
                    "Removing this member would leave these projects without an authenticating "
                    f"member: {', '.join(projects)}. Add another member first."
                )
            changed = connection.execute(
                """
                UPDATE space_users
                SET removal_started_at = ?, updated_at = ?
                WHERE user_id = ? AND removal_started_at IS NULL
                """,
                (now, now, user_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("The member-removal access fence did not commit exactly once.")
            connection.execute(
                "UPDATE team_member_tokens SET revoked_at = ? "
                "WHERE user_id = ? AND revoked_at IS NULL",
                (now, user_id),
            )
            connection.execute("DELETE FROM team_sessions WHERE user_id = ?", (user_id,))
            connection.execute(
                """
                UPDATE team_invitations
                SET revoked_at = ?
                WHERE created_by = ? AND consumed_at IS NULL AND revoked_at IS NULL
                """,
                (now, user_id),
            )
            connection.execute(
                """
                UPDATE project_invitations
                SET response = 'revoked', responded_at = ?
                WHERE response IS NULL AND (invited_by = ? OR invited_user_id = ?)
                """,
                (now, user_id, user_id),
            )
            connection.execute("DELETE FROM project_members WHERE user_id = ?", (user_id,))
            return self._member_removal_preview_from_connection(connection, user_id)

    def members_pending_removal(self) -> list[SpaceUserRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM space_users
                WHERE removal_started_at IS NOT NULL AND removed_at IS NULL
                ORDER BY removal_started_at, user_id
                """
            ).fetchall()
        return [self._space_user_record(row) for row in rows]

    def complete_member_removal(self, user_id: str) -> SpaceUserRecord:
        """Finish one tombstone only after every authorized work owner is terminal."""

        _canonical_uuid4(user_id, label="user identity")
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            member = self._space_user_from_connection(connection, user_id)
            if member is None or member.identity_kind != "team_member":
                raise KeyError(f"Unknown RCP team member {user_id}.")
            if member.removal_started_at is None:
                raise ValueError("Member removal has not committed its access fence.")
            if member.removed_at is not None:
                return member
            preview = self._member_removal_preview_from_connection(connection, user_id)
            if preview.active_task_ids or preview.active_episode_ids:
                raise RuntimeError("Member removal still has live authorized work.")
            connection.execute(
                "UPDATE space_users SET removed_at = ?, updated_at = ? WHERE user_id = ?",
                (now, now, user_id),
            )
            completed = self._space_user_from_connection(connection, user_id)
            assert completed is not None
            return completed

    @property
    def local_owner(self) -> SpaceUserRecord | None:
        if self.space_kind != "personal":
            return None
        users = self.space_users()
        if len(users) != 1 or users[0].identity_kind != "local_owner":
            raise RuntimeError("A personal RCP space must contain exactly one local owner.")
        return users[0]

    def seat_project_member(
        self,
        project_id: str,
        user_id: str,
        *,
        seated_by: str | None = None,
    ) -> ProjectMemberRecord:
        """Seat one person on one project, idempotently.

        Binds the durable ``user_id``. A display name is deliberately not
        required: a person exists before they have chosen one, and creating a
        project must keep working without one.
        """

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_active_space_user_from_connection(connection, user_id)
            connection.execute(
                """
                INSERT OR IGNORE INTO project_members (project_id, user_id, seated_at, seated_by)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, user_id, now, seated_by),
            )
            row = connection.execute(
                "SELECT * FROM project_members WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            ).fetchone()
        assert row is not None
        return ProjectMemberRecord.model_validate(dict(row))

    def project_members(self, project_id: str) -> list[ProjectMemberRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM project_members
                WHERE project_id = ?
                ORDER BY seated_at, user_id
                """,
                (project_id,),
            ).fetchall()
        return [ProjectMemberRecord.model_validate(dict(row)) for row in rows]

    def is_project_member(self, project_id: str, user_id: str) -> bool:
        with self.connection() as connection:
            return (
                connection.execute(
                    """
                    SELECT 1
                    FROM project_members AS member
                    JOIN projects AS project ON project.project_id = member.project_id
                    WHERE member.project_id = ? AND member.user_id = ?
                      AND project.retired_at IS NULL
                    """,
                    (project_id, user_id),
                ).fetchone()
                is not None
            )

    def member_project_ids(self, user_id: str) -> set[str]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT member.project_id
                FROM project_members AS member
                JOIN projects AS project ON project.project_id = member.project_id
                WHERE member.user_id = ? AND project.retired_at IS NULL
                """,
                (user_id,),
            ).fetchall()
        return {row["project_id"] for row in rows}

    def invite_to_project(
        self,
        project_id: str,
        invited_user_id: str,
        *,
        invited_by: str,
    ) -> ProjectInvitationRecord:
        """Invite one existing space member to one project.

        Any member may invite; there is no approval chain and no rank. The
        invitee must already be enrolled in the space, because a project
        invitation grants no credential and cannot be used to join the space.
        """

        now = self.now()
        invitation_id = str(uuid.uuid4())
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for user_id, label in ((invited_by, "inviter"), (invited_user_id, "invitee")):
                try:
                    self._require_team_member_from_connection(connection, user_id)
                except KeyError:
                    raise KeyError(f"The {label} is not a member of this RCP space.") from None
            if (
                connection.execute(
                    "SELECT 1 FROM project_members WHERE project_id = ? AND user_id = ?",
                    (project_id, invited_by),
                ).fetchone()
                is None
            ):
                raise ValueError("Only a project member may invite someone to it.")
            if (
                connection.execute(
                    "SELECT 1 FROM project_members WHERE project_id = ? AND user_id = ?",
                    (project_id, invited_user_id),
                ).fetchone()
                is not None
            ):
                raise ValueError("That person is already a member of this project.")
            try:
                connection.execute(
                    """
                    INSERT INTO project_invitations (
                        invitation_id, project_id, invited_user_id, invited_by, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (invitation_id, project_id, invited_user_id, invited_by, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("That person already has a pending invitation.") from exc
        return ProjectInvitationRecord(
            invitation_id=invitation_id,
            project_id=project_id,
            invited_user_id=invited_user_id,
            invited_by=invited_by,
            created_at=now,
        )

    def pending_project_invitations(self, invited_user_id: str) -> list[ProjectInvitationRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM project_invitations
                WHERE invited_user_id = ? AND response IS NULL
                ORDER BY created_at DESC, invitation_id
                """,
                (invited_user_id,),
            ).fetchall()
        return [ProjectInvitationRecord.model_validate(dict(row)) for row in rows]

    def answer_project_invitation(
        self,
        invitation_id: str,
        *,
        invited_user_id: str,
        response: Literal["accepted", "declined"],
    ) -> ProjectInvitationRecord:
        """Accept or decline. Declining leaves no membership and no residue."""

        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_team_member_from_connection(connection, invited_user_id)
            row = connection.execute(
                "SELECT * FROM project_invitations WHERE invitation_id = ?",
                (invitation_id,),
            ).fetchone()
            # An invitation addressed to somebody else is answered exactly as a
            # missing one, so nobody learns that it exists.
            if row is None or row["invited_user_id"] != invited_user_id:
                raise KeyError(invitation_id)
            if row["response"] is not None:
                raise ValueError("That invitation was already answered.")
            connection.execute(
                "UPDATE project_invitations SET response = ?, responded_at = ? "
                "WHERE invitation_id = ?",
                (response, now, invitation_id),
            )
            if response == "accepted":
                connection.execute(
                    """
                    INSERT OR IGNORE INTO project_members (
                        project_id, user_id, seated_at, seated_by
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (row["project_id"], invited_user_id, now, row["invited_by"]),
                )
            updated = connection.execute(
                "SELECT * FROM project_invitations WHERE invitation_id = ?",
                (invitation_id,),
            ).fetchone()
        return ProjectInvitationRecord.model_validate(dict(updated))

    def leave_project(self, project_id: str, user_id: str) -> None:
        """Give up your own membership of one project.

        The last member cannot leave: a memberless project would be invisible to
        everyone, and there is no administrator rank able to recover it. Add
        another member first.
        """

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            members = connection.execute(
                "SELECT user_id FROM project_members WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            member_ids = {row["user_id"] for row in members}
            if user_id not in member_ids:
                raise KeyError(f"{user_id} is not a member of {project_id}.")
            if len(member_ids) == 1:
                raise ValueError(
                    "You are the only member of this project. Add another member before leaving."
                )
            connection.execute(
                "DELETE FROM project_members WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            )
            # A pending invitation is not a claim on the project.
            connection.execute(
                "DELETE FROM project_invitations "
                "WHERE project_id = ? AND invited_user_id = ? AND response IS NULL",
                (project_id, user_id),
            )

    def rename_space(self, name: str) -> str:
        normalized = normalize_space_name(name)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._space_kind_from_connection(connection) != "team":
                raise ValueError("Only a team space has a mutable team name.")
            connection.execute(
                "UPDATE space_identity SET space_name = ? WHERE singleton = 1",
                (normalized,),
            )
        return normalized

    def enroll_team_member(self, code: str, display_name: str) -> tuple[SpaceUserRecord, str]:
        parsed = _parse_enrollment_code(code)
        if parsed is None:
            raise TeamAuthenticationError(
                "enrollment_code_invalid", "The enrollment code is invalid."
            )
        kind, code_id, supplied_hash = parsed
        now = self.now()
        error: TeamAuthenticationError | None = None
        member: SpaceUserRecord | None = None
        token: str | None = None
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._space_kind_from_connection(connection) != "team":
                raise ValueError("Only a team space accepts enrollment.")
            table = "team_bootstrap_codes" if kind == "bootstrap" else "team_invitations"
            id_column = "code_id" if kind == "bootstrap" else "invitation_id"
            row = connection.execute(
                f"SELECT * FROM {table} WHERE {id_column} = ?",  # noqa: S608
                (code_id,),
            ).fetchone()
            if row is None:
                error = TeamAuthenticationError(
                    "enrollment_code_invalid", "The enrollment code is invalid."
                )
            elif row["consumed_at"] is not None:
                error = TeamAuthenticationError(
                    "enrollment_code_consumed", "The enrollment code has already been used."
                )
            elif row["revoked_at"] is not None:
                error = TeamAuthenticationError(
                    "enrollment_code_invalid", "The enrollment code is invalid."
                )
            elif row["locked_at"] is not None:
                error = TeamAuthenticationError(
                    "enrollment_code_locked", "The enrollment code is locked."
                )
            elif kind == "invite" and row["expires_at"] <= now:
                error = TeamAuthenticationError(
                    "enrollment_code_expired", "The enrollment code has expired."
                )
            elif not hmac.compare_digest(row["code_hash"], supplied_hash):
                failed_attempts = int(row["failed_attempts"]) + 1
                locked_at = now if failed_attempts >= TEAM_CODE_FAILED_ATTEMPT_LIMIT else None
                connection.execute(
                    f"UPDATE {table} SET failed_attempts = ?, locked_at = ? "  # noqa: S608
                    f"WHERE {id_column} = ?",
                    (failed_attempts, locked_at, code_id),
                )
                error = TeamAuthenticationError(
                    "enrollment_code_locked" if locked_at else "enrollment_code_invalid",
                    "The enrollment code is locked."
                    if locked_at
                    else "The enrollment code is invalid.",
                )
            else:
                if kind == "bootstrap":
                    first_member = connection.execute(
                        "SELECT 1 FROM space_users LIMIT 1"
                    ).fetchone()
                    if first_member is not None:
                        error = TeamAuthenticationError(
                            "enrollment_code_consumed",
                            "The team space has already been claimed.",
                        )
                if error is None:
                    member = SpaceUserRecord(
                        user_id=str(uuid.uuid4()),
                        identity_kind="team_member",
                        display_name=display_name,
                        created_at=now,
                        updated_at=now,
                    )
                    token, token_hash = _new_member_token()
                    connection.execute(
                        """
                        INSERT INTO space_users (
                            user_id, identity_kind, display_name, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            member.user_id,
                            member.identity_kind,
                            member.display_name,
                            member.created_at,
                            member.updated_at,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO team_member_tokens (
                            token_id, user_id, token_hash, created_at, revoked_at
                        ) VALUES (?, ?, ?, ?, NULL)
                        """,
                        (str(uuid.uuid4()), member.user_id, token_hash, now),
                    )
                    connection.execute(
                        f"UPDATE {table} SET consumed_at = ?, consumed_by = ? "  # noqa: S608
                        f"WHERE {id_column} = ?",
                        (now, member.user_id, code_id),
                    )
                    # A project the server opened before anyone enrolled has no
                    # members, so it is invisible to everybody and nobody can
                    # invite themselves to it. Whoever enrols claims it; once a
                    # project has any member, invitations govern it instead.
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO project_members (
                            project_id, user_id, seated_at, seated_by
                        )
                        SELECT projects.project_id, ?, ?, NULL
                        FROM projects
                        WHERE NOT EXISTS (
                            SELECT 1 FROM project_members
                            WHERE project_members.project_id = projects.project_id
                        )
                        """,
                        (member.user_id, now),
                    )
        if error is not None:
            raise error
        if member is None or token is None:  # pragma: no cover - exhaustive transition above
            raise RuntimeError("RCP team enrollment did not produce a member credential.")
        return member, token

    def create_team_invitation(
        self,
        created_by: str,
    ) -> tuple[TeamInvitationRecord, str]:
        now = self.now()
        expires_at = (
            datetime.fromisoformat(now) + timedelta(days=TEAM_INVITATION_TTL_DAYS)
        ).isoformat()
        code, invitation_id, code_hash = _new_enrollment_code("invite")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_team_member_from_connection(connection, created_by)
            connection.execute(
                """
                INSERT INTO team_invitations (
                    invitation_id, code_hash, created_by, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (invitation_id, code_hash, created_by, now, expires_at),
            )
        return (
            TeamInvitationRecord(
                invitation_id=invitation_id,
                created_by=created_by,
                created_at=now,
                expires_at=expires_at,
                failed_attempts=0,
            ),
            code,
        )

    def team_invitations(self, created_by: str) -> list[TeamInvitationRecord]:
        with self.connection() as connection:
            self._require_team_member_from_connection(connection, created_by)
            rows = connection.execute(
                """
                SELECT invitation_id, created_by, created_at, expires_at,
                       consumed_at, consumed_by, failed_attempts, locked_at, revoked_at
                FROM team_invitations
                WHERE created_by = ?
                ORDER BY created_at DESC, invitation_id
                """,
                (created_by,),
            ).fetchall()
        return [TeamInvitationRecord.model_validate(dict(row)) for row in rows]

    def create_team_session(self, token: str) -> tuple[str, SpaceUserRecord]:
        if (
            not isinstance(token, str)
            or len(token) > TEAM_MEMBER_TOKEN_MAX_LENGTH
            or not token.startswith("rcp_")
        ):
            raise TeamAuthenticationError(
                "team_token_invalid", "The member token is invalid or revoked."
            )
        token_hash = _sha256(token)
        now = self.now()
        expires_at = (
            datetime.fromisoformat(now) + timedelta(days=TEAM_SESSION_IDLE_DAYS)
        ).isoformat()
        session, session_hash = _new_session_token()
        member: SpaceUserRecord | None = None
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._space_kind_from_connection(connection) != "team":
                raise ValueError("Only a team space accepts member tokens.")
            row = connection.execute(
                """
                SELECT user_id, token_hash FROM team_member_tokens
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (token_hash,),
            ).fetchone()
            if row is not None and hmac.compare_digest(row["token_hash"], token_hash):
                member = self._require_team_member_from_connection(connection, row["user_id"])
                connection.execute(
                    """
                    INSERT INTO team_sessions (
                        session_hash, user_id, created_at, last_seen_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_hash, member.user_id, now, now, expires_at),
                )
        if member is None:
            raise TeamAuthenticationError(
                "team_token_invalid", "The member token is invalid or revoked."
            )
        return session, member

    def authenticate_team_member_token(self, token: str) -> SpaceUserRecord:
        """Resolve one permanent token without creating a browser session."""

        if (
            not isinstance(token, str)
            or len(token) > TEAM_MEMBER_TOKEN_MAX_LENGTH
            or not token.startswith("rcp_")
        ):
            raise TeamAuthenticationError(
                "team_token_invalid", "The member token is invalid or revoked."
            )
        token_hash = _sha256(token)
        member: SpaceUserRecord | None = None
        with self.connection() as connection:
            if self._space_kind_from_connection(connection) != "team":
                raise ValueError("Only a team space accepts member tokens.")
            row = connection.execute(
                """
                SELECT user_id, token_hash FROM team_member_tokens
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (token_hash,),
            ).fetchone()
            if row is not None and hmac.compare_digest(row["token_hash"], token_hash):
                member = self._require_team_member_from_connection(connection, row["user_id"])
        if member is None:
            raise TeamAuthenticationError(
                "team_token_invalid", "The member token is invalid or revoked."
            )
        return member

    def resolve_team_session(self, session: str | None) -> SpaceUserRecord | None:
        if (
            not session
            or len(session) > TEAM_SESSION_TOKEN_MAX_LENGTH
            or not session.startswith("rcp_session_")
        ):
            return None
        session_hash = _sha256(session)
        now = self.now()
        expires_at = (
            datetime.fromisoformat(now) + timedelta(days=TEAM_SESSION_IDLE_DAYS)
        ).isoformat()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM team_sessions WHERE session_hash = ?",
                (session_hash,),
            ).fetchone()
            if row is None or not hmac.compare_digest(row["session_hash"], session_hash):
                return None
            if row["expires_at"] <= now:
                connection.execute(
                    "DELETE FROM team_sessions WHERE session_hash = ?", (session_hash,)
                )
                return None
            member = self._space_user_from_connection(connection, row["user_id"])
            if (
                member is None
                or member.identity_kind != "team_member"
                or member.removal_started_at is not None
                or member.removed_at is not None
            ):
                connection.execute(
                    "DELETE FROM team_sessions WHERE session_hash = ?", (session_hash,)
                )
                return None
            connection.execute(
                """
                UPDATE team_sessions SET last_seen_at = ?, expires_at = ?
                WHERE session_hash = ?
                """,
                (now, expires_at, session_hash),
            )
            return member

    def delete_team_session(self, session: str | None) -> None:
        if not session:
            return
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM team_sessions WHERE session_hash = ?", (_sha256(session),)
            )

    @staticmethod
    def detach_space_authentication_for_restore(
        connection: sqlite3.Connection,
        *,
        now: str,
    ) -> None:
        """Invalidate restored browser and unused enrollment capabilities."""

        if not connection.in_transaction:
            raise ValueError("restored space authentication detachment requires a transaction")
        _required_timestamp(now)
        connection.execute("DELETE FROM team_sessions")
        connection.execute(
            """
            UPDATE team_bootstrap_codes
            SET revoked_at = COALESCE(revoked_at, ?)
            WHERE consumed_at IS NULL AND revoked_at IS NULL
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE team_invitations
            SET revoked_at = COALESCE(revoked_at, ?)
            WHERE consumed_at IS NULL AND revoked_at IS NULL
            """,
            (now,),
        )

    def _require_authenticating_team_session(
        self,
        connection: sqlite3.Connection,
        session: str | None,
        user_id: str,
        now: str,
    ) -> None:
        if (
            not session
            or len(session) > TEAM_SESSION_TOKEN_MAX_LENGTH
            or not session.startswith("rcp_session_")
        ):
            raise TeamAuthenticationError(
                "team_session_invalid", "The browser session is invalid or expired."
            )
        session_hash = _sha256(session)
        row = connection.execute(
            "SELECT session_hash, user_id, expires_at FROM team_sessions WHERE session_hash = ?",
            (session_hash,),
        ).fetchone()
        if (
            row is None
            or not hmac.compare_digest(row["session_hash"], session_hash)
            or row["user_id"] != user_id
            or row["expires_at"] <= now
        ):
            raise TeamAuthenticationError(
                "team_session_invalid", "The browser session is invalid or expired."
            )

    def rotate_team_token(
        self,
        user_id: str,
        *,
        authenticating_session: str | None = None,
    ) -> str:
        now = self.now()
        token, token_hash = _new_member_token()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_team_member_from_connection(connection, user_id)
            if authenticating_session is not None:
                self._require_authenticating_team_session(
                    connection, authenticating_session, user_id, now
                )
            connection.execute(
                "UPDATE team_member_tokens SET revoked_at = ? "
                "WHERE user_id = ? AND revoked_at IS NULL",
                (now, user_id),
            )
            connection.execute("DELETE FROM team_sessions WHERE user_id = ?", (user_id,))
            connection.execute(
                """
                INSERT INTO team_member_tokens (
                    token_id, user_id, token_hash, created_at, revoked_at
                ) VALUES (?, ?, ?, ?, NULL)
                """,
                (str(uuid.uuid4()), user_id, token_hash, now),
            )
        return token

    def revoke_team_token(
        self,
        user_id: str,
        *,
        authenticating_session: str | None = None,
    ) -> None:
        now = self.now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_team_member_from_connection(connection, user_id)
            if authenticating_session is not None:
                self._require_authenticating_team_session(
                    connection, authenticating_session, user_id, now
                )
            self._require_token_revocation_safe(connection, user_id)
            connection.execute(
                "UPDATE team_member_tokens SET revoked_at = ? "
                "WHERE user_id = ? AND revoked_at IS NULL",
                (now, user_id),
            )
            connection.execute("DELETE FROM team_sessions WHERE user_id = ?", (user_id,))

    def preprovision_team_member(self, display_name: str | None = None) -> SpaceUserRecord:
        now = self.now()
        member = SpaceUserRecord(
            user_id=str(uuid.uuid4()),
            identity_kind="team_member",
            display_name=display_name,
            created_at=now,
            updated_at=now,
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._space_kind_from_connection(connection) != "team":
                raise ValueError("Only a team space can preprovision team members.")
            connection.execute(
                """
                INSERT INTO space_users (
                    user_id, identity_kind, display_name, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    member.user_id,
                    member.identity_kind,
                    member.display_name,
                    member.created_at,
                    member.updated_at,
                ),
            )
        return member

    def rename_space_user(
        self,
        user_id: str,
        display_name: str | None,
    ) -> SpaceUserRecord:
        _canonical_uuid4(user_id, label="user identity")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM space_users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown RCP space user {user_id}.")
            current = self._space_user_record(row)
            if current.removal_started_at is not None or current.removed_at is not None:
                raise KeyError(f"Unknown active RCP space user {user_id}.")
            updated = SpaceUserRecord.model_validate(
                {
                    **current.model_dump(),
                    "display_name": display_name,
                    "updated_at": self.now(),
                }
            )
            connection.execute(
                """
                UPDATE space_users
                SET display_name = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (updated.display_name, updated.updated_at, user_id),
            )
        return updated

    @classmethod
    def _member_removal_preview_from_connection(
        cls,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> MemberRemovalPreviewRecord:
        member = cls._space_user_from_connection(connection, user_id)
        if member is None or member.identity_kind != "team_member":
            raise KeyError(f"Unknown RCP team member {user_id}.")
        projects = _bounded_removal_values(
            connection.execute(
                "SELECT project_id FROM project_members WHERE user_id = ? ORDER BY project_id",
                (user_id,),
            ).fetchall(),
            "project_id",
            label="project inventory",
        )
        active_tasks = _bounded_removal_values(
            connection.execute(
                """
                SELECT operation_id FROM graph_runs
                WHERE authorized_user_id = ? AND status IN ('queued', 'running', 'pausing')
                ORDER BY operation_id
                """,
                (user_id,),
            ).fetchall(),
            "operation_id",
            label="task inventory",
        )
        episode_placeholders = ", ".join("?" for _ in _LIVE_EPISODE_STATUSES)
        active_episodes = _bounded_removal_values(
            connection.execute(
                f"""
                SELECT episode_id FROM episodes
                WHERE authorized_user_id = ? AND status IN ({episode_placeholders})
                ORDER BY episode_id
                """,  # noqa: S608 - placeholders come from the closed episode vocabulary
                (user_id, *_LIVE_EPISODE_STATUSES),
            ).fetchall(),
            "episode_id",
            label="episode inventory",
        )
        active_tokens = _bounded_removal_values(
            connection.execute(
                """
                SELECT token_id FROM team_member_tokens
                WHERE user_id = ? AND revoked_at IS NULL
                ORDER BY token_id
                """,
                (user_id,),
            ).fetchall(),
            "token_id",
            label="token inventory",
        )
        session_hashes = tuple(
            str(row["session_hash"])
            for row in connection.execute(
                "SELECT session_hash FROM team_sessions WHERE user_id = ? ORDER BY session_hash",
                (user_id,),
            ).fetchall()
        )
        space_invitations = _bounded_removal_values(
            connection.execute(
                """
                SELECT invitation_id FROM team_invitations
                WHERE created_by = ? AND consumed_at IS NULL AND revoked_at IS NULL
                ORDER BY invitation_id
                """,
                (user_id,),
            ).fetchall(),
            "invitation_id",
            label="space-invitation inventory",
        )
        project_invitations = _bounded_removal_values(
            connection.execute(
                """
                SELECT invitation_id FROM project_invitations
                WHERE response IS NULL AND (invited_by = ? OR invited_user_id = ?)
                ORDER BY invitation_id
                """,
                (user_id, user_id),
            ).fetchall(),
            "invitation_id",
            label="project-invitation inventory",
        )
        active_members = cls._active_enrolled_member_ids(connection)
        orphaned_projects = tuple(
            project_id
            for project_id in projects
            if not cls._active_project_member_ids(connection, project_id) - {user_id}
        )
        payload = {
            "member": member.model_dump(mode="json"),
            "last_authenticating_member": bool(active_tokens and not (active_members - {user_id})),
            "project_ids": projects,
            "orphaned_project_ids": orphaned_projects,
            "active_task_ids": active_tasks,
            "active_episode_ids": active_episodes,
            "active_token_ids": active_tokens,
            "browser_session_hashes": session_hashes,
            "space_invitation_ids": space_invitations,
            "project_invitation_ids": project_invitations,
            "active_enrolled_member_ids": tuple(sorted(active_members)),
        }
        boundary = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return MemberRemovalPreviewRecord(
            member=member,
            last_authenticating_member=bool(active_tokens and not (active_members - {user_id})),
            project_ids=projects,
            orphaned_project_ids=orphaned_projects,
            active_task_ids=active_tasks,
            active_episode_ids=active_episodes,
            active_token_ids=active_tokens,
            browser_session_count=len(session_hashes),
            space_invitation_ids=space_invitations,
            project_invitation_ids=project_invitations,
            boundary_sha256=boundary,
        )

    @staticmethod
    def _active_enrolled_member_ids(connection: sqlite3.Connection) -> set[str]:
        rows = connection.execute(
            """
            SELECT DISTINCT user.user_id
            FROM space_users AS user
            JOIN team_member_tokens AS token ON token.user_id = user.user_id
            WHERE user.identity_kind = 'team_member'
              AND user.removal_started_at IS NULL
              AND user.removed_at IS NULL
              AND token.revoked_at IS NULL
            """
        ).fetchall()
        return {str(row["user_id"]) for row in rows}

    @classmethod
    def _active_project_member_ids(
        cls,
        connection: sqlite3.Connection,
        project_id: str,
    ) -> set[str]:
        active_members = cls._active_enrolled_member_ids(connection)
        rows = connection.execute(
            "SELECT user_id FROM project_members WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        return {str(row["user_id"]) for row in rows}.intersection(active_members)

    @staticmethod
    def _project_labels(
        connection: sqlite3.Connection,
        project_ids: tuple[str, ...],
    ) -> list[str]:
        labels: list[str] = []
        for project_id in project_ids:
            row = connection.execute(
                "SELECT name FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            name = str(row["name"]) if row is not None else "Unknown project"
            labels.append(f"{name} ({project_id})")
        return labels

    @classmethod
    def _require_token_revocation_safe(
        cls,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> None:
        active_tokens = connection.execute(
            "SELECT 1 FROM team_member_tokens WHERE user_id = ? AND revoked_at IS NULL LIMIT 1",
            (user_id,),
        ).fetchone()
        if active_tokens is None:
            return
        if not (cls._active_enrolled_member_ids(connection) - {user_id}):
            raise ValueError(
                "You are the last enrolled member who can authenticate. Rotate this credential "
                "or enroll another member instead of revoking it."
            )
        projects = tuple(
            sorted(
                str(row["project_id"])
                for row in connection.execute(
                    "SELECT project_id FROM project_members WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
                if not cls._active_project_member_ids(connection, str(row["project_id"]))
                - {user_id}
            )
        )
        if projects:
            raise ValueError(
                "Revoking this credential would leave these projects without an authenticating "
                f"member: {', '.join(cls._project_labels(connection, projects))}. Add another "
                "member first."
            )

    @staticmethod
    def _space_kind_from_connection(connection: sqlite3.Connection) -> SpaceKind:
        row = connection.execute(
            "SELECT space_kind FROM space_identity WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("RCP space identity is unavailable.")
        return _stored_space_kind(row["space_kind"])

    @classmethod
    def _space_users_from_connection(
        cls,
        connection: sqlite3.Connection,
    ) -> list[SpaceUserRecord]:
        rows = connection.execute(
            "SELECT * FROM space_users ORDER BY created_at, user_id"
        ).fetchall()
        return [cls._space_user_record(row) for row in rows]

    @classmethod
    def _space_user_from_connection(
        cls,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> SpaceUserRecord | None:
        row = connection.execute(
            "SELECT * FROM space_users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return cls._space_user_record(row) if row is not None else None

    @classmethod
    def _require_team_member_from_connection(
        cls,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> SpaceUserRecord:
        if cls._space_kind_from_connection(connection) != "team":
            raise ValueError("Only a team space has team members.")
        member = cls._space_user_from_connection(connection, user_id)
        if (
            member is None
            or member.identity_kind != "team_member"
            or member.removal_started_at is not None
            or member.removed_at is not None
        ):
            raise KeyError(f"Unknown RCP team member {user_id}.")
        return member

    @classmethod
    def _require_active_space_user_from_connection(
        cls,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> SpaceUserRecord:
        user = cls._space_user_from_connection(connection, user_id)
        if user is None or user.removal_started_at is not None or user.removed_at is not None:
            raise KeyError(f"Unknown active RCP space user {user_id}.")
        return user

    @staticmethod
    def _ready_group_members(
        members: list[StoredWatcherRecord],
    ) -> list[StoredWatcherRecord] | None:
        """Return deliverable members only when a durable group is collectively ready."""

        if not members or any(item.group_id is None for item in members):
            return None
        if len({item.graph_target.key for item in members}) != 1:
            return None
        if any(item.status == "stopped" and item.stopped_by != "agent" for item in members):
            return None
        deliverable = [
            item
            for item in members
            if not (item.status == "stopped" and item.stopped_by == "agent")
        ]
        if not deliverable or any(item.notified for item in deliverable):
            return None
        if any(
            item.status == "active"
            or (
                item.status == "degraded"
                and item.consecutive_error_count < WATCHER_GROUP_DIAGNOSTIC_ERROR_COUNT
            )
            or item.status not in {"completed", "degraded"}
            for item in deliverable
        ):
            return None
        return deliverable

    def _validate_watcher_notification_members(
        self,
        connection: sqlite3.Connection,
        watchers: list[StoredWatcherRecord],
    ) -> None:
        """Require a delivery claim to contain every ready member of each group."""

        requested = {item.watcher_id for item in watchers}
        group_ids = {item.group_id for item in watchers if item.group_id is not None}
        for watcher in watchers:
            if watcher.group_id is None and watcher.status != "completed":
                raise ValueError("an ungrouped watcher must complete before delivery")
        for group_id in group_ids:
            assert group_id is not None
            rows = connection.execute(
                "SELECT * FROM watchers WHERE group_id = ? ORDER BY created_at, watcher_id",
                (group_id,),
            ).fetchall()
            ready = self._ready_group_members([self._watcher_record(row) for row in rows])
            if ready is None:
                raise ValueError("a watcher group is not ready for delivery")
            ready_ids = {item.watcher_id for item in ready}
            if ready_ids != (requested & {item.watcher_id for item in ready}):
                raise ValueError("a watcher group must be claimed as one delivery unit")
