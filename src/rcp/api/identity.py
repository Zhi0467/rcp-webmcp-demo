from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException, Request
from fastapi.responses import Response

from rcp.core.models import AuthorizedHuman
from rcp.limits import TEAM_SESSION_IDLE_DAYS
from rcp.storage import AppStore, SpaceKind, SpaceUserRecord

TEAM_SESSION_COOKIE = "__Host-rcp_session"
TEAM_SESSION_COOKIE_MAX_AGE = TEAM_SESSION_IDLE_DAYS * 24 * 60 * 60

TrustedPrincipalResolver = Callable[[Request, AppStore], SpaceUserRecord | str]


class IdentityAccess:
    """Resolve RCP identities and enforce the identity-level access contract."""

    def __init__(
        self,
        store: AppStore,
        *,
        space_id: str,
        space_kind: SpaceKind,
        trusted_principal_resolver: TrustedPrincipalResolver | None,
    ) -> None:
        self._store = store
        self._space_id = space_id
        self._space_kind = space_kind
        self._trusted_principal_resolver = trusted_principal_resolver

    def require_team_space(self) -> None:
        if self._space_kind != "team":
            raise HTTPException(status_code=404, detail="Team authentication is unavailable.")

    def set_team_session_cookie(self, response: Response, session: str) -> None:
        response.set_cookie(
            TEAM_SESSION_COOKIE,
            session,
            max_age=TEAM_SESSION_COOKIE_MAX_AGE,
            path="/",
            secure=True,
            httponly=True,
            samesite="lax",
        )

    def clear_team_session_cookie(self, response: Response) -> None:
        response.delete_cookie(
            TEAM_SESSION_COOKIE,
            path="/",
            secure=True,
            httponly=True,
            samesite="lax",
        )

    def resolve_team_user(self, request: Request) -> SpaceUserRecord:
        cached = getattr(request.state, "team_member", None)
        if isinstance(cached, SpaceUserRecord):
            return cached

        if self._trusted_principal_resolver is None:
            session = request.cookies.get(TEAM_SESSION_COOKIE)
            member = self._store.resolve_team_session(session)
            if member is None:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "code": "team_identity_required",
                        "message": "This team action requires a trusted authenticated member.",
                    },
                )
            request.state.team_session = session
        else:
            resolved = self._trusted_principal_resolver(request, self._store)
            user_id = resolved.user_id if isinstance(resolved, SpaceUserRecord) else resolved
            if not isinstance(user_id, str):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "code": "team_identity_invalid",
                        "message": "The trusted team identity is invalid for this space.",
                    },
                )
            member = self._store.space_user(user_id)

        if (
            member is None
            or member.identity_kind != "team_member"
            or member.removal_started_at is not None
            or member.removed_at is not None
        ):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "team_identity_invalid",
                    "message": "The trusted team identity is invalid for this space.",
                },
            )
        request.state.team_member = member
        return member

    def authenticating_team_session(self, request: Request) -> str | None:
        if self._trusted_principal_resolver is not None:
            return None
        session = getattr(request.state, "team_session", None)
        if not isinstance(session, str):  # pragma: no cover - middleware resolves this first
            raise HTTPException(status_code=401, detail="The browser session is unavailable.")
        return session

    def acting_user(self, request: Request) -> SpaceUserRecord:
        if self._space_kind == "personal":
            owner = self._store.local_owner
            if owner is None:  # pragma: no cover - guarded by the storage invariant
                raise HTTPException(status_code=500, detail="Personal owner identity is missing.")
            return owner
        return self.resolve_team_user(request)

    def require_patch_capable_identity(self, request: Request) -> AuthorizedHuman:
        user = self.acting_user(request)
        if user.display_name is None or not user.display_name.strip():
            raise HTTPException(
                status_code=428,
                detail={
                    "code": "identity_name_required",
                    "message": (
                        "Choose an RCP display name before this action. The name will be "
                        "copied into permanent project history as a snapshot."
                    ),
                },
            )
        return AuthorizedHuman(
            space_id=self._space_id,
            user_id=user.user_id,
            display_name=user.display_name,
        )

    def identity_payload(self, user: SpaceUserRecord) -> dict[str, object]:
        return {
            "space_id": self._space_id,
            "space_kind": self._space_kind,
            "space_name": self._store.space_name,
            "user": user.model_dump(mode="json"),
        }


__all__ = [
    "TEAM_SESSION_COOKIE",
    "IdentityAccess",
    "TrustedPrincipalResolver",
]
