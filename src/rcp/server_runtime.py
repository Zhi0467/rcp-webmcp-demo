from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from rcp import __version__
from rcp.limits import SERVER_WEB_BUILD_MAX_BYTES, SERVER_WEB_BUILD_MAX_FILES
from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout

SERVER_METADATA_SCHEMA_VERSION = 3
SERVER_METADATA_FILENAME = "rcp-server.json"
ServerOwnerKind = Literal["cli", "desktop", "embedded"]

_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_WEB_BUILD_ID = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_WEB_RELATIVE_PATH_BYTES = 4096


class ServerMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class RunningReleaseIdentity:
    commit: str
    web_build_id: str


@dataclass(frozen=True)
class ServerMetadata:
    schema_version: int
    instance_id: str
    pid: int
    host: str
    port: int
    app_version: str
    data_dir_id: str
    owner_kind: ServerOwnerKind
    control_socket: str | None
    running_commit: str | None
    web_build_id: str | None

    @classmethod
    def create(
        cls,
        data_dir: Path,
        *,
        host: str,
        port: int,
        owner_kind: ServerOwnerKind,
        control_socket: Path | None = None,
        running_commit: str | None = None,
        web_build_id: str | None = None,
    ) -> ServerMetadata:
        return cls(
            schema_version=SERVER_METADATA_SCHEMA_VERSION,
            instance_id=str(uuid.uuid4()),
            pid=os.getpid(),
            host=host,
            port=port,
            app_version=__version__,
            data_dir_id=data_dir_identity(data_dir),
            owner_kind=owner_kind,
            control_socket=str(control_socket) if control_socket is not None else None,
            running_commit=running_commit,
            web_build_id=web_build_id,
        )

    @classmethod
    def from_dict(cls, raw: object) -> ServerMetadata:
        if not isinstance(raw, dict):
            raise ServerMetadataError("server metadata is not a JSON object")
        expected = {
            "schema_version",
            "instance_id",
            "pid",
            "host",
            "port",
            "app_version",
            "data_dir_id",
            "owner_kind",
            "control_socket",
            "running_commit",
            "web_build_id",
        }
        if set(raw) != expected:
            raise ServerMetadataError("server metadata has an unsupported shape")
        try:
            metadata = cls(**raw)
            instance_id = str(uuid.UUID(metadata.instance_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ServerMetadataError("server metadata contains invalid values") from exc
        if metadata.schema_version != SERVER_METADATA_SCHEMA_VERSION:
            raise ServerMetadataError("server metadata has an unsupported schema version")
        if instance_id != metadata.instance_id:
            raise ServerMetadataError("server metadata has a non-canonical instance id")
        if (
            isinstance(metadata.pid, bool)
            or not isinstance(metadata.pid, int)
            or metadata.pid <= 0
            or not isinstance(metadata.host, str)
            or not metadata.host
            or isinstance(metadata.port, bool)
            or not isinstance(metadata.port, int)
            or not 1 <= metadata.port <= 65535
            or not isinstance(metadata.app_version, str)
            or not metadata.app_version
            or not isinstance(metadata.data_dir_id, str)
            or len(metadata.data_dir_id) != 64
            or not isinstance(metadata.owner_kind, str)
            or metadata.owner_kind not in {"cli", "desktop", "embedded"}
            or not _valid_control_socket(metadata.control_socket)
            or not _valid_release_identity(metadata.running_commit, metadata.web_build_id)
        ):
            raise ServerMetadataError("server metadata contains invalid values")
        return metadata

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}"


def data_dir_identity(data_dir: Path) -> str:
    canonical = str(data_dir.expanduser().resolve()).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def metadata_path(data_dir: Path) -> Path:
    return data_dir / SERVER_METADATA_FILENAME


def installed_control_socket_path(data_dir: Path) -> Path | None:
    """Return the fixed socket only for the configured installed service account."""

    from rcp.server_ops.config import load_installed_server_config
    from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT

    config_path = DEFAULT_SERVER_LAYOUT.config_path
    if not os.path.lexists(config_path):
        return None
    try:
        config = load_installed_server_config(config_path)
        service = pwd.getpwnam(config.service_account)
    except (KeyError, OSError, ValueError) as exc:
        raise ServerMetadataError("installed server configuration is invalid") from exc
    if Path(config.paths.data_dir).resolve() != data_dir.expanduser().resolve():
        return None
    if os.geteuid() != service.pw_uid or os.getegid() != service.pw_gid:
        raise ServerMetadataError(
            "the installed team service must run as its configured service account"
        )
    return Path(config.paths.control_socket)


def _valid_control_socket(value: object) -> bool:
    if value is None:
        return True
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    path = Path(value)
    return path.is_absolute() and ".." not in path.parts and str(path) == value


def _valid_release_identity(running_commit: object, web_build_id: object) -> bool:
    if running_commit is None or web_build_id is None:
        return running_commit is None and web_build_id is None
    return (
        isinstance(running_commit, str)
        and _FULL_GIT_COMMIT.fullmatch(running_commit) is not None
        and isinstance(web_build_id, str)
        and _WEB_BUILD_ID.fullmatch(web_build_id) is not None
    )


def capture_installed_release_identity(
    layout: ServerLayout = DEFAULT_SERVER_LAYOUT,
    *,
    working_dir: Path | None = None,
) -> RunningReleaseIdentity:
    """Capture the physical immutable release before the service starts."""

    try:
        physical = (working_dir or Path.cwd()).resolve(strict=True)
        current = layout.current_release.resolve(strict=True)
    except OSError as exc:
        raise ServerMetadataError("installed release identity is unavailable") from exc
    if (
        physical != current
        or physical.parent != layout.releases_root
        or _FULL_GIT_COMMIT.fullmatch(physical.name) is None
    ):
        raise ServerMetadataError("installed service is not running from one canonical release")
    return RunningReleaseIdentity(
        commit=physical.name,
        web_build_id=web_build_identity(physical / "web" / "dist"),
    )


def web_build_identity(root: Path) -> str:
    """Hash one bounded, symlink-free Web bundle including its relative paths."""

    try:
        if root.is_symlink() or not root.is_dir() or not (root / "index.html").is_file():
            raise ServerMetadataError("Web build is unavailable")
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    except OSError as exc:
        raise ServerMetadataError("Web build could not be inspected") from exc
    files: list[tuple[Path, bytes, int]] = []
    total = 0
    for path in paths:
        try:
            info = path.lstat()
        except OSError as exc:
            raise ServerMetadataError("Web build could not be inspected") from exc
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ServerMetadataError("Web build contains a non-regular entry")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if not relative or len(relative) > _MAX_WEB_RELATIVE_PATH_BYTES:
            raise ServerMetadataError("Web build contains an invalid path")
        total += info.st_size
        files.append((path, relative, info.st_size))
        if len(files) > SERVER_WEB_BUILD_MAX_FILES or total > SERVER_WEB_BUILD_MAX_BYTES:
            raise ServerMetadataError("Web build exceeds its inspection bound")
    if not files:
        raise ServerMetadataError("Web build is empty")
    digest = hashlib.sha256(b"rcp-web-build-v1\0")
    for path, relative, expected_size in files:
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(expected_size.to_bytes(8, "big"))
        observed_size = 0
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    observed_size += len(chunk)
        except OSError as exc:
            raise ServerMetadataError("Web build could not be read") from exc
        if observed_size != expected_size:
            raise ServerMetadataError("Web build changed during inspection")
    return f"sha256:{digest.hexdigest()}"


def read_server_metadata(data_dir: Path) -> ServerMetadata:
    path = metadata_path(data_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServerMetadataError("server metadata is unavailable") from exc
    return ServerMetadata.from_dict(raw)


def remove_server_metadata(data_dir: Path, *, instance_id: str) -> bool:
    """Remove discoverability only when it still names this lock-owning instance."""
    try:
        current = read_server_metadata(data_dir)
    except ServerMetadataError:
        return False
    if current.instance_id != instance_id:
        return False
    metadata_path(data_dir).unlink(missing_ok=True)
    return True


@contextmanager
def published_server_metadata(data_dir: Path, metadata: ServerMetadata) -> Iterator[None]:
    """Publish discoverability while an already-held lock remains authoritative."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_path(data_dir)
    temporary = path.with_name(f".{path.name}.{metadata.instance_id}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(metadata.as_dict(), handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        yield
    finally:
        temporary.unlink(missing_ok=True)
        remove_server_metadata(data_dir, instance_id=metadata.instance_id)
