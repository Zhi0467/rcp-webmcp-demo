"""Fixed filesystem layout for the first source-built RCP team server."""

from __future__ import annotations

import importlib.resources
import re
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_PROJECT_ALIAS = re.compile(r"[a-z][a-z0-9-]{0,47}")


@dataclass(frozen=True)
class ServerLayout:
    service_account: str
    service_home: Path
    server_root: Path
    source_checkout: Path
    releases_root: Path
    data_dir: Path
    projects_root: Path
    credentials_root: Path
    update_checkpoints_root: Path
    restore_operations_root: Path
    codex_state_root: Path
    claude_state_root: Path
    ssh_state_root: Path
    config_path: Path
    current_release: Path
    runtime_dir: Path
    control_socket: Path
    cli_wrapper: Path
    systemd_unit: Path
    service_unit_name: str

    def __post_init__(self) -> None:
        paths = self.recorded_paths()
        for name, value in paths.items():
            path = Path(value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"server layout path {name} must be absolute and normalized")

        owned_children = (
            self.source_checkout,
            self.releases_root,
            self.data_dir,
            self.projects_root,
            self.credentials_root,
            self.update_checkpoints_root,
            self.restore_operations_root,
        )
        if any(self.server_root not in path.parents for path in owned_children):
            raise ValueError("ordinary server-owned paths must stay below the server root")
        for index, first in enumerate(owned_children):
            for second in owned_children[index + 1 :]:
                if first == second or first in second.parents or second in first.parents:
                    raise ValueError("ordinary server-owned paths must not overlap")
        if self.control_socket.parent != self.runtime_dir:
            raise ValueError("the control socket must be directly below the runtime directory")
        if any(
            path.parent != self.service_home
            for path in (self.codex_state_root, self.claude_state_root, self.ssh_state_root)
        ):
            raise ValueError("provider and SSH state must retain ordinary service-home paths")

    def recorded_paths(self) -> dict[str, str]:
        return {
            "service_home": str(self.service_home),
            "server_root": str(self.server_root),
            "source_checkout": str(self.source_checkout),
            "releases_root": str(self.releases_root),
            "data_dir": str(self.data_dir),
            "projects_root": str(self.projects_root),
            "credentials_root": str(self.credentials_root),
            "update_checkpoints_root": str(self.update_checkpoints_root),
            "restore_operations_root": str(self.restore_operations_root),
            "codex_state_root": str(self.codex_state_root),
            "claude_state_root": str(self.claude_state_root),
            "ssh_state_root": str(self.ssh_state_root),
            "config_path": str(self.config_path),
            "current_release": str(self.current_release),
            "runtime_dir": str(self.runtime_dir),
            "control_socket": str(self.control_socket),
            "cli_wrapper": str(self.cli_wrapper),
            "systemd_unit": str(self.systemd_unit),
        }

    def release_dir(self, commit: str) -> Path:
        if _FULL_GIT_COMMIT.fullmatch(commit) is None:
            raise ValueError("release commit must be a lowercase 40-character Git object id")
        return self.releases_root / commit

    def project_repository_dir(self, project_id: str, alias: str) -> Path:
        _project_credential_components(project_id, alias)
        return self.projects_root / project_id / "repositories" / alias

    def project_deploy_key_path(self, project_id: str, alias: str) -> Path:
        relative = project_deploy_key_relative_path(project_id, alias)
        return self.credentials_root / Path(relative)


DEFAULT_SERVER_LAYOUT = ServerLayout(
    service_account="rcp",
    service_home=Path("/home/rcp"),
    server_root=Path("/home/rcp/rcp-server"),
    source_checkout=Path("/home/rcp/rcp-server/source"),
    releases_root=Path("/home/rcp/rcp-server/releases"),
    data_dir=Path("/home/rcp/rcp-server/data"),
    projects_root=Path("/home/rcp/rcp-server/projects"),
    credentials_root=Path("/home/rcp/rcp-server/credentials"),
    update_checkpoints_root=Path("/home/rcp/rcp-server/update-checkpoints"),
    restore_operations_root=Path("/home/rcp/rcp-server/restore-operations"),
    codex_state_root=Path("/home/rcp/.codex"),
    claude_state_root=Path("/home/rcp/.claude"),
    ssh_state_root=Path("/home/rcp/.ssh"),
    config_path=Path("/etc/rcp/server.toml"),
    current_release=Path("/etc/rcp/current"),
    runtime_dir=Path("/run/rcp"),
    control_socket=Path("/run/rcp/control.sock"),
    cli_wrapper=Path("/usr/local/bin/rcp"),
    systemd_unit=Path("/etc/systemd/system/rcp.service"),
    service_unit_name="rcp.service",
)


def remote_credentials_root(remote_home: str) -> PurePosixPath:
    if not isinstance(remote_home, str) or any(
        ord(character) < 32 or ord(character) == 127 for character in remote_home
    ):
        raise ValueError("remote account home must be one safe line")
    home = PurePosixPath(remote_home)
    if not home.is_absolute() or home == PurePosixPath("/") or ".." in home.parts:
        raise ValueError("remote account home must be an absolute normalized non-root path")
    return home / ".local" / "share" / "rcp" / "credentials"


def remote_projects_root(remote_home: str) -> PurePosixPath:
    if not isinstance(remote_home, str) or any(
        ord(character) < 32 or ord(character) == 127 for character in remote_home
    ):
        raise ValueError("remote account home must be one safe line")
    home = PurePosixPath(remote_home)
    if not home.is_absolute() or home == PurePosixPath("/") or ".." in home.parts:
        raise ValueError("remote account home must be an absolute normalized non-root path")
    return home / ".local" / "share" / "rcp" / "projects"


def project_deploy_key_relative_path(project_id: str, alias: str) -> PurePosixPath:
    _project_credential_components(project_id, alias)
    return PurePosixPath("projects") / project_id / alias / "id_ed25519"


def remote_project_deploy_key_path(
    remote_home: str,
    project_id: str,
    alias: str,
) -> PurePosixPath:
    return remote_credentials_root(remote_home) / project_deploy_key_relative_path(
        project_id,
        alias,
    )


def server_service_unit_text() -> str:
    return (
        importlib.resources.files("rcp.server_ops")
        .joinpath("assets", "rcp.service")
        .read_text(encoding="utf-8")
    )


def _project_credential_components(project_id: str, alias: str) -> None:
    try:
        parsed = uuid.UUID(project_id)
    except (AttributeError, ValueError) as exc:
        raise ValueError("project id must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != project_id:
        raise ValueError("project id must be a lowercase, hyphenated canonical UUID4")
    if not isinstance(alias, str) or _PROJECT_ALIAS.fullmatch(alias) is None:
        raise ValueError("repository alias must be a canonical provisioning alias")


__all__ = [
    "DEFAULT_SERVER_LAYOUT",
    "ServerLayout",
    "project_deploy_key_relative_path",
    "remote_credentials_root",
    "remote_projects_root",
    "remote_project_deploy_key_path",
    "server_service_unit_text",
]
