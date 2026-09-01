"""Canonical GitHub.com repository identities for server-owned workflows."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
_REPOSITORY = re.compile(r"[A-Za-z0-9._-]{1,100}")
_HTTPS_PREFIX = "https://github.com/"
_SSH_PREFIX = "git@github.com:"


class GitHubRepositoryRef(BaseModel):
    """One lowercase GitHub repository identity, never a caller-supplied URL."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    identity: str

    @field_validator("identity")
    @classmethod
    def validate_canonical_identity(cls, value: str) -> str:
        if value != value.lower() or value.count("/") != 1:
            raise ValueError("GitHub repository identity must be lowercase owner/repository")
        owner, repository = value.split("/", 1)
        if _OWNER.fullmatch(owner) is None:
            raise ValueError("GitHub repository owner is invalid")
        if _REPOSITORY.fullmatch(repository) is None or repository in {".", ".."}:
            raise ValueError("GitHub repository name is invalid")
        return value

    @property
    def owner(self) -> str:
        return self.identity.split("/", 1)[0]

    @property
    def repository(self) -> str:
        return self.identity.split("/", 1)[1]

    @property
    def https_clone_url(self) -> str:
        return f"https://github.com/{self.identity}.git"

    @property
    def ssh_clone_url(self) -> str:
        return f"git@github.com:{self.identity}.git"

    @property
    def settings_url(self) -> str:
        return f"https://github.com/{self.identity}/settings/keys"


def parse_github_repository_ref(value: str) -> GitHubRepositoryRef:
    """Parse the two accepted credential-free GitHub.com URL forms."""

    if not isinstance(value, str):
        raise ValueError("GitHub repository must be one URL string")
    if value != value.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("GitHub repository must be one trimmed URL")
    if value.startswith(_HTTPS_PREFIX):
        remainder = value[len(_HTTPS_PREFIX) :]
    elif value.startswith(_SSH_PREFIX):
        remainder = value[len(_SSH_PREFIX) :]
    else:
        raise ValueError("GitHub repository must use HTTPS or git@github.com SCP syntax")
    if remainder.count("/") != 1:
        raise ValueError("GitHub repository must contain exactly owner/repository")
    owner, repository = remainder.split("/", 1)
    if repository.endswith(".git"):
        repository = repository[:-4]
    if _OWNER.fullmatch(owner) is None:
        raise ValueError("GitHub repository owner is invalid")
    if _REPOSITORY.fullmatch(repository) is None or repository in {".", ".."}:
        raise ValueError("GitHub repository name is invalid")
    return GitHubRepositoryRef(identity=f"{owner}/{repository}".lower())


__all__ = ["GitHubRepositoryRef", "parse_github_repository_ref"]
