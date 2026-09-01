"""Concrete source-state capture for personal-to-team project transfer."""

from __future__ import annotations

import hashlib
import shlex
import subprocess

from rcp import __version__
from rcp.core.transition_models import GraphHeadRef
from rcp.limits import PROJECT_TRANSFER_SOURCE_PROBE_TIMEOUT_SECONDS
from rcp.server_ops.github import parse_github_repository_ref
from rcp.service import ProjectService
from rcp.storage import ProjectTransferRepositorySource, ProjectTransferSourceConfiguration
from rcp.transfer import TRANSFER_ARCHIVE_CODEC
from rcp.transport.ssh import ssh_arguments

PROJECT_TRANSFER_SCHEMA_GENERATION = 1
PROJECT_TRANSFER_ARCHIVE_CODEC = TRANSFER_ARCHIVE_CODEC


def capture_project_transfer_source(
    service: ProjectService,
) -> tuple[ProjectTransferSourceConfiguration, GraphHeadRef]:
    """Read the live manifest, repository identities, and canonical main head."""

    materialization = service.history.current_materialization()
    head = service.history.head_ref(materialization)
    manifest = service.history.manifest
    manifest_bytes = (service.history.root / "manifest.toml").read_bytes()
    repositories = tuple(
        ProjectTransferRepositorySource(
            alias=repository.alias,
            repository=parse_github_repository_ref(
                _repository_origin(
                    host=manifest.machine_map[repository.machine].host,
                    path=repository.path,
                )
            ),
            machine_alias=repository.machine,
        )
        for repository in sorted(manifest.repositories, key=lambda item: item.alias)
    )
    configuration = ProjectTransferSourceConfiguration(
        source_rcp_version=__version__,
        source_schema_generation=PROJECT_TRANSFER_SCHEMA_GENERATION,
        supported_archive_codecs=(PROJECT_TRANSFER_ARCHIVE_CODEC,),
        machine_aliases=tuple(sorted(manifest.machine_map)),
        repositories=repositories,
        state_repository=manifest.state.repository,
        project_truth_scope=tuple(manifest.project.truth_scope),
        default_run_truth_scope=tuple(manifest.agent.default_run_truth_scope),
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    return configuration, head


def _repository_origin(*, host: str, path: str) -> str:
    command = ["git", "-C", path, "remote", "get-url", "origin"]
    arguments = ssh_arguments(host, shlex.join(command)) if host else command
    try:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=PROJECT_TRANSFER_SOURCE_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("source repository GitHub origin could not be read") from exc
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != 1:
        raise ValueError("source repository must have one readable GitHub origin")
    return lines[0]


__all__ = [
    "PROJECT_TRANSFER_ARCHIVE_CODEC",
    "PROJECT_TRANSFER_SCHEMA_GENERATION",
    "capture_project_transfer_source",
]
