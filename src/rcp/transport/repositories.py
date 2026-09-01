from __future__ import annotations

from pydantic import BaseModel

from rcp.config import MachineConfig, RepositoryConfig


class RepositoryAccess(BaseModel):
    """A pointer to a repository. RCP never copies one; it only names where it lives."""

    alias: str
    machine: str
    host: str = ""
    path: str


def repository_access(repository: RepositoryConfig, machine: MachineConfig) -> RepositoryAccess:
    """Build the pointer for a repository. An empty host means "on this machine"."""

    return RepositoryAccess(
        alias=repository.alias,
        machine=repository.machine,
        host=machine.host,
        path=repository.path,
    )
