from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rcp.api.dependencies import get_identity_access, get_store
from rcp.api.identity import TEAM_SESSION_COOKIE, IdentityAccess
from rcp.core.models import DISPLAY_NAME_MAX_LENGTH, normalize_display_name
from rcp.limits import (
    TEAM_ENROLLMENT_CODE_MAX_LENGTH,
    TEAM_MEMBER_TOKEN_MAX_LENGTH,
)
from rcp.storage import SPACE_NAME_MAX_LENGTH, AppStore, normalize_space_name

router = APIRouter()

StoreDependency = Annotated[AppStore, Depends(get_store)]
IdentityDependency = Annotated[IdentityAccess, Depends(get_identity_access)]


class SpaceIdentityUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(max_length=DISPLAY_NAME_MAX_LENGTH)

    @field_validator("display_name", mode="before")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        return normalize_display_name(value)


class TeamEnrollmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    code: str = Field(min_length=1, max_length=TEAM_ENROLLMENT_CODE_MAX_LENGTH)
    display_name: str = Field(max_length=DISPLAY_NAME_MAX_LENGTH)

    @field_validator("display_name", mode="before")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        return normalize_display_name(value)


class TeamSessionExchangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token: str = Field(min_length=1, max_length=TEAM_MEMBER_TOKEN_MAX_LENGTH)


class TeamSpaceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(max_length=SPACE_NAME_MAX_LENGTH)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return normalize_space_name(value)


@router.get("/api/identity")
def get_identity(
    request: Request,
    *,
    identity_access: IdentityDependency,
) -> dict[str, object]:
    return identity_access.identity_payload(identity_access.acting_user(request))


@router.patch("/api/identity")
def update_identity(
    request: Request,
    body: SpaceIdentityUpdateRequest,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, object]:
    current = identity_access.acting_user(request)
    try:
        renamed = store.rename_space_user(current.user_id, body.display_name)
    except KeyError as exc:  # pragma: no cover - resolved and renamed in one local store
        raise HTTPException(status_code=403, detail="Acting identity is no longer valid.") from exc
    return identity_access.identity_payload(renamed)


@router.post("/api/team/enroll")
def enroll_team_member(
    body: TeamEnrollmentRequest,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, object]:
    identity_access.require_team_space()
    member, token = store.enroll_team_member(body.code, body.display_name)
    return {"identity": identity_access.identity_payload(member), "token": token}


@router.post("/api/team/session/exchange")
def exchange_team_session(
    body: TeamSessionExchangeRequest,
    response: Response,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, object]:
    identity_access.require_team_space()
    session, member = store.create_team_session(body.token)
    identity_access.set_team_session_cookie(response, session)
    return identity_access.identity_payload(member)


@router.post("/api/team/session/logout")
def logout_team_session(
    request: Request,
    response: Response,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, bool]:
    identity_access.require_team_space()
    identity_access.acting_user(request)
    store.delete_team_session(request.cookies.get(TEAM_SESSION_COOKIE))
    identity_access.clear_team_session_cookie(response)
    return {"ok": True}


@router.get("/api/team/invitations")
def team_invitations(
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> list[dict[str, object]]:
    identity_access.require_team_space()
    member = identity_access.acting_user(request)
    return [
        invitation.model_dump(mode="json") for invitation in store.team_invitations(member.user_id)
    ]


@router.post("/api/team/invitations")
def create_team_invitation(
    request: Request,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, object]:
    identity_access.require_team_space()
    member = identity_access.acting_user(request)
    invitation, code = store.create_team_invitation(member.user_id)
    space_name = store.space_name
    if space_name is None:  # pragma: no cover - named team initialization is required
        raise HTTPException(status_code=500, detail="Team space name is missing.")
    return {
        "invitation": invitation.model_dump(mode="json"),
        "code": code,
        "space_name": space_name,
    }


@router.post("/api/team/credential/rotate")
def rotate_team_credential(
    request: Request,
    response: Response,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, str]:
    identity_access.require_team_space()
    member = identity_access.acting_user(request)
    token = store.rotate_team_token(
        member.user_id,
        authenticating_session=identity_access.authenticating_team_session(request),
    )
    identity_access.clear_team_session_cookie(response)
    return {"token": token}


@router.post("/api/team/credential/revoke")
def revoke_team_credential(
    request: Request,
    response: Response,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, bool]:
    identity_access.require_team_space()
    member = identity_access.acting_user(request)
    try:
        store.revoke_team_token(
            member.user_id,
            authenticating_session=identity_access.authenticating_team_session(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    identity_access.clear_team_session_cookie(response)
    return {"ok": True}


@router.patch("/api/team/space")
def update_team_space(
    request: Request,
    body: TeamSpaceUpdateRequest,
    *,
    identity_access: IdentityDependency,
    store: StoreDependency,
) -> dict[str, str]:
    identity_access.require_team_space()
    identity_access.acting_user(request)
    return {"space_name": store.rename_space(body.name)}


__all__ = [
    "SpaceIdentityUpdateRequest",
    "TeamEnrollmentRequest",
    "TeamSessionExchangeRequest",
    "TeamSpaceUpdateRequest",
    "create_team_invitation",
    "enroll_team_member",
    "exchange_team_session",
    "get_identity",
    "logout_team_session",
    "revoke_team_credential",
    "rotate_team_credential",
    "router",
    "team_invitations",
    "update_identity",
    "update_team_space",
]
