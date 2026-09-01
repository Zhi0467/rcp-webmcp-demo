"""Strict, nonsecret configuration for one installed RCP team server."""

from __future__ import annotations

import os
import pwd
import re
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from rcp.server_ops.layout import DEFAULT_SERVER_LAYOUT, ServerLayout
from rcp.server_ops.models import SERVER_CLI_MAX_FIELD_CHARS

SERVER_CONFIG_SCHEMA_VERSION = 2
LEGACY_SERVER_CONFIG_SCHEMA_VERSION = 1
SERVER_CONFIG_MODE = 0o640
DEFAULT_BACKUP_SCHEDULE = "02:00"
DEFAULT_BACKUP_RETENTION = 30

_GITHUB_HTTPS_ORIGIN = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?")
_GITHUB_SSH_ORIGIN = re.compile(r"git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?")
_SSH_PUBLIC_FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{20,64}={0,2}")
_BACKUP_LOCAL_TIME = re.compile(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]")
_AGE_BECH32_ALPHABET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_AGE_BECH32_VALUES = {character: index for index, character in enumerate(_AGE_BECH32_ALPHABET)}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ServerSourceConfig(_StrictModel):
    origin: str
    branch: Literal["main"] = "main"
    authentication: Literal["public", "deploy_key"]
    public_key_fingerprint: str | None = None

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("source origin must be one trimmed line")
        if not (_GITHUB_HTTPS_ORIGIN.fullmatch(value) or _GITHUB_SSH_ORIGIN.fullmatch(value)):
            raise ValueError("source origin must be an HTTPS or SSH GitHub repository URL")
        return value

    @field_validator("public_key_fingerprint")
    @classmethod
    def validate_public_key_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and _SSH_PUBLIC_FINGERPRINT.fullmatch(value) is None:
            raise ValueError("source public-key fingerprint must use OpenSSH SHA256 form")
        return value

    @model_validator(mode="after")
    def validate_authentication(self) -> ServerSourceConfig:
        uses_ssh = _GITHUB_SSH_ORIGIN.fullmatch(self.origin) is not None
        if uses_ssh != (self.authentication == "deploy_key"):
            raise ValueError("HTTPS sources must be public and SSH sources must use a deploy key")
        if self.authentication == "public" and self.public_key_fingerprint is not None:
            raise ValueError("a public source must not record a source credential fingerprint")
        if self.authentication == "deploy_key" and self.public_key_fingerprint is None:
            raise ValueError("a deploy-key source requires its public fingerprint")
        return self


class ServerPathsConfig(_StrictModel):
    service_home: str
    server_root: str
    source_checkout: str
    releases_root: str
    data_dir: str
    projects_root: str
    credentials_root: str
    update_checkpoints_root: str
    restore_operations_root: str
    codex_state_root: str
    claude_state_root: str
    ssh_state_root: str
    config_path: str
    current_release: str
    runtime_dir: str
    control_socket: str
    cli_wrapper: str
    systemd_unit: str

    @model_validator(mode="after")
    def validate_accepted_layout(self) -> ServerPathsConfig:
        if self.model_dump() != DEFAULT_SERVER_LAYOUT.recorded_paths():
            raise ValueError("installed server paths must match the accepted fixed layout")
        return self

    @classmethod
    def from_layout(cls, layout: ServerLayout = DEFAULT_SERVER_LAYOUT) -> ServerPathsConfig:
        return cls.model_validate(layout.recorded_paths())


class ServerBackupConfig(_StrictModel):
    destination: str
    schedule: str = DEFAULT_BACKUP_SCHEDULE
    retention: int = DEFAULT_BACKUP_RETENTION
    age_recipient: str

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        return validate_backup_destination(value)

    @field_validator("schedule")
    @classmethod
    def validate_schedule(cls, value: str) -> str:
        return validate_backup_schedule(value)

    @field_validator("retention")
    @classmethod
    def validate_retention(cls, value: int) -> int:
        return validate_backup_retention(value)

    @field_validator("age_recipient")
    @classmethod
    def validate_age_recipient(cls, value: str) -> str:
        return validate_age_recipient(value)


class InstalledServerConfig(_StrictModel):
    schema_version: Literal[SERVER_CONFIG_SCHEMA_VERSION] = SERVER_CONFIG_SCHEMA_VERSION
    installation_id: str
    service_account: Literal["rcp"] = "rcp"
    service_unit: Literal["rcp.service"] = "rcp.service"
    source: ServerSourceConfig
    paths: ServerPathsConfig
    backup: ServerBackupConfig | None = None

    @field_validator("installation_id")
    @classmethod
    def validate_installation_id(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, ValueError) as exc:
            raise ValueError("installation id must be a canonical UUID4") from exc
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("installation id must be a lowercase, hyphenated canonical UUID4")
        return value


def create_installed_server_config(
    *,
    source: ServerSourceConfig,
    installation_id: str | None = None,
) -> InstalledServerConfig:
    return InstalledServerConfig(
        installation_id=installation_id if installation_id is not None else str(uuid.uuid4()),
        source=source,
        paths=ServerPathsConfig.from_layout(),
    )


def render_installed_server_config(config: InstalledServerConfig) -> str:
    document = tomlkit.document()
    document.add("schema_version", config.schema_version)
    document.add("installation_id", config.installation_id)
    document.add("service_account", config.service_account)
    document.add("service_unit", config.service_unit)

    source = tomlkit.table()
    source.add("origin", config.source.origin)
    source.add("branch", config.source.branch)
    source.add("authentication", config.source.authentication)
    if config.source.public_key_fingerprint is not None:
        source.add("public_key_fingerprint", config.source.public_key_fingerprint)
    document.add("source", source)

    paths = tomlkit.table()
    for name, value in config.paths.model_dump().items():
        paths.add(name, value)
    document.add("paths", paths)

    if config.backup is not None:
        backup = tomlkit.table()
        backup.add("destination", config.backup.destination)
        backup.add("schedule", config.backup.schedule)
        backup.add("retention", config.backup.retention)
        backup.add("age_recipient", config.backup.age_recipient)
        document.add("backup", backup)
    content = tomlkit.dumps(document)
    if parse_installed_server_config(content) != config:
        raise RuntimeError("rendered installed-server configuration changed meaning")
    return content


def parse_installed_server_config(content: str) -> InstalledServerConfig:
    try:
        data = tomlkit.parse(content).unwrap()
        schema_version = data.get("schema_version")
        if type(schema_version) is int and schema_version == LEGACY_SERVER_CONFIG_SCHEMA_VERSION:
            if "backup" in data:
                raise ValueError("legacy installed-server configuration cannot contain backup")
            data["schema_version"] = SERVER_CONFIG_SCHEMA_VERSION
        return InstalledServerConfig.model_validate(data)
    except (tomlkit.exceptions.ParseError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid installed-server configuration: {exc}") from exc


def validate_backup_destination(value: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("backup destination must be one trimmed line")
    if len(value) > SERVER_CLI_MAX_FIELD_CHARS or len(value.encode("utf-8")) > 4096:
        raise ValueError(
            "backup destination cannot exceed "
            f"{SERVER_CLI_MAX_FIELD_CHARS} characters or 4096 UTF-8 bytes"
        )
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or ".." in path.parts or str(path) != value:
        raise ValueError("backup destination must be an absolute normalized non-root path")
    return value


def validate_backup_schedule(value: str) -> str:
    if not isinstance(value, str) or _BACKUP_LOCAL_TIME.fullmatch(value) is None:
        raise ValueError("backup schedule must be server-local 24-hour time in HH:MM form")
    return value


def validate_backup_retention(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("backup retention must be a positive archive count")
    return value


def validate_age_recipient(value: str) -> str:
    if not isinstance(value, str) or value != value.strip() or value != value.lower():
        raise ValueError("age recipient must be one lowercase native X25519 recipient")
    if len(value) != 62 or not value.startswith("age1"):
        raise ValueError("age recipient must be one native X25519 age1 recipient")
    try:
        encoded = [_AGE_BECH32_VALUES[character] for character in value[4:]]
    except KeyError as exc:
        raise ValueError("age recipient contains an invalid bech32 character") from exc
    payload = encoded[:-6]
    if (
        len(payload) != 52
        or payload[-1] & 0x0F
        or _bech32_polymod((*_bech32_hrp_expand("age"), *encoded)) != 1
    ):
        raise ValueError("age recipient has an invalid native X25519 checksum")
    return value


def _bech32_hrp_expand(value: str) -> tuple[int, ...]:
    return (
        tuple(ord(character) >> 5 for character in value)
        + (0,)
        + tuple(ord(character) & 31 for character in value)
    )


def _bech32_polymod(values: tuple[int, ...]) -> int:
    generators = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    checksum = 1
    for value in values:
        top = checksum >> 25
        checksum = ((checksum & 0x1FFFFFF) << 5) ^ value
        for index, generator in enumerate(generators):
            if (top >> index) & 1:
                checksum ^= generator
    return checksum


def load_installed_server_config(
    path: Path = DEFAULT_SERVER_LAYOUT.config_path,
) -> InstalledServerConfig:
    return _load_installed_server_config(path, ownership=_expected_config_ownership())


def write_installed_server_config(
    config: InstalledServerConfig,
    path: Path = DEFAULT_SERVER_LAYOUT.config_path,
) -> None:
    """Atomically replace one validated config while retaining installation identity."""

    ownership = _expected_config_ownership()
    _reject_symlink_ancestry(path.parent)
    if path.is_symlink():
        raise ValueError(f"installed-server configuration cannot be a symlink: {path}")
    if path.exists():
        existing = _load_installed_server_config(path, ownership=ownership)
        if existing.installation_id != config.installation_id:
            raise ValueError("installed-server configuration cannot change installation_id")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"installed-server configuration parent is not a directory: {path.parent}")

    content = render_installed_server_config(config)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchown(descriptor, *ownership)
        os.fchmod(descriptor, SERVER_CONFIG_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        _validate_config_file(path, ownership=ownership)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _load_installed_server_config(
    path: Path, *, ownership: tuple[int, int]
) -> InstalledServerConfig:
    _reject_symlink_ancestry(path.parent)
    _validate_config_file(path, ownership=ownership)
    return parse_installed_server_config(path.read_text(encoding="utf-8"))


def _expected_config_ownership() -> tuple[int, int]:
    try:
        root = pwd.getpwnam("root")
        service = pwd.getpwnam(DEFAULT_SERVER_LAYOUT.service_account)
    except KeyError as exc:
        raise ValueError("installed-server configuration requires root and rcp accounts") from exc
    if root.pw_uid != 0:
        raise ValueError("the root account must have uid 0")
    return root.pw_uid, service.pw_gid


def _validate_config_file(path: Path, *, ownership: tuple[int, int]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"installed-server configuration is not a regular file: {path}")
    info = path.stat()
    if (info.st_uid, info.st_gid) != ownership:
        raise ValueError("installed-server configuration has the wrong owner or reader group")
    if stat.S_IMODE(info.st_mode) != SERVER_CONFIG_MODE:
        raise ValueError("installed-server configuration must have mode 0640")


def _reject_symlink_ancestry(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(
                f"installed-server configuration ancestry cannot contain a symlink: {candidate}"
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DEFAULT_BACKUP_RETENTION",
    "DEFAULT_BACKUP_SCHEDULE",
    "LEGACY_SERVER_CONFIG_SCHEMA_VERSION",
    "SERVER_CONFIG_MODE",
    "SERVER_CONFIG_SCHEMA_VERSION",
    "InstalledServerConfig",
    "ServerBackupConfig",
    "ServerPathsConfig",
    "ServerSourceConfig",
    "create_installed_server_config",
    "load_installed_server_config",
    "parse_installed_server_config",
    "render_installed_server_config",
    "validate_age_recipient",
    "validate_backup_destination",
    "validate_backup_retention",
    "validate_backup_schedule",
    "write_installed_server_config",
]
