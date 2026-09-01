from __future__ import annotations

import os
import re
import stat
import uuid
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rcp.server_ops import config as server_config
from rcp.server_ops.config import (
    SERVER_CONFIG_MODE,
    InstalledServerConfig,
    ServerPathsConfig,
    ServerSourceConfig,
    create_installed_server_config,
    load_installed_server_config,
    parse_installed_server_config,
    render_installed_server_config,
    write_installed_server_config,
)
from rcp.server_ops.layout import (
    DEFAULT_SERVER_LAYOUT,
    remote_credentials_root,
    remote_projects_root,
    server_service_unit_text,
)

PROJECT_ID = "f39c7f63-c565-46ab-bbb8-72fb4c7d4793"
COMMIT = "a" * 40


def _public_config(*, installation_id: str | None = None) -> InstalledServerConfig:
    return create_installed_server_config(
        installation_id=installation_id,
        source=ServerSourceConfig(
            origin="https://github.com/example/research-control-panel.git",
            authentication="public",
        ),
    )


@pytest.fixture
def configured_test_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rcp.server_ops.config._expected_config_ownership",
        lambda: (os.getuid(), os.getgid()),
    )


def test_fixed_layout_records_every_accepted_server_path() -> None:
    layout = DEFAULT_SERVER_LAYOUT

    assert layout.service_account == "rcp"
    assert layout.service_unit_name == "rcp.service"
    assert layout.service_home == Path("/home/rcp")
    assert layout.server_root == Path("/home/rcp/rcp-server")
    assert layout.source_checkout == Path("/home/rcp/rcp-server/source")
    assert layout.releases_root == Path("/home/rcp/rcp-server/releases")
    assert layout.data_dir == Path("/home/rcp/rcp-server/data")
    assert layout.projects_root == Path("/home/rcp/rcp-server/projects")
    assert layout.credentials_root == Path("/home/rcp/rcp-server/credentials")
    assert layout.update_checkpoints_root == Path("/home/rcp/rcp-server/update-checkpoints")
    assert layout.restore_operations_root == Path("/home/rcp/rcp-server/restore-operations")
    assert layout.codex_state_root == Path("/home/rcp/.codex")
    assert layout.claude_state_root == Path("/home/rcp/.claude")
    assert layout.ssh_state_root == Path("/home/rcp/.ssh")
    assert layout.config_path == Path("/etc/rcp/server.toml")
    assert layout.current_release == Path("/etc/rcp/current")
    assert layout.control_socket == Path("/run/rcp/control.sock")
    assert layout.cli_wrapper == Path("/usr/local/bin/rcp")
    assert layout.systemd_unit == Path("/etc/systemd/system/rcp.service")
    assert all(Path(value).is_absolute() for value in layout.recorded_paths().values())


def test_release_and_project_paths_accept_only_bounded_components() -> None:
    layout = DEFAULT_SERVER_LAYOUT

    assert layout.release_dir(COMMIT) == layout.releases_root / COMMIT
    assert layout.project_repository_dir(PROJECT_ID, "repo") == (
        layout.projects_root / PROJECT_ID / "repositories" / "repo"
    )

    for invalid_commit in ("a" * 39, "A" * 40, "../" + "a" * 40):
        with pytest.raises(ValueError):
            layout.release_dir(invalid_commit)
    for invalid_alias in ("", ".", "..", "a/b", "a\\b", "bad\nname"):
        with pytest.raises(ValueError):
            layout.project_repository_dir(PROJECT_ID, invalid_alias)
    with pytest.raises(ValueError):
        layout.project_repository_dir(str(uuid.uuid1()), "repo")


_SAFE_LINE = "remote account home must be one safe line"
_ABSOLUTE_HOME = "remote account home must be an absolute normalized non-root path"


@pytest.mark.parametrize(
    ("build", "suffix"),
    [
        (remote_credentials_root, "credentials"),
        (remote_projects_root, "projects"),
    ],
)
def test_remote_roots_follow_the_proven_home_instead_of_assuming_home_user(
    build: Callable[[str], PurePosixPath],
    suffix: str,
) -> None:
    assert str(build("/srv/research/alice")) == f"/srv/research/alice/.local/share/rcp/{suffix}"


@pytest.mark.parametrize(
    ("build", "_suffix"),
    [
        (remote_credentials_root, "credentials"),
        (remote_projects_root, "projects"),
    ],
)
@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        # Control characters cannot reach a remote shell line.
        ("/home/alice\nother", _SAFE_LINE),
        ("/home/alice\x00other", _SAFE_LINE),
        ("/home/alice\x7f", _SAFE_LINE),
        ("/home/alice\tother", _SAFE_LINE),
        # A space is not a control character, so it stays a legal home.
        ("home/alice", _ABSOLUTE_HOME),
        ("/", _ABSOLUTE_HOME),
        ("/srv/../home/alice", _ABSOLUTE_HOME),
        ("/home/alice/..", _ABSOLUTE_HOME),
    ],
)
def test_remote_roots_reject_unsafe_homes_with_the_guard_that_caught_them(
    build: Callable[[str], PurePosixPath],
    _suffix: str,
    invalid: str,
    message: str,
) -> None:
    """Both roots take the same untrusted home, so both need the same proof.

    The message is asserted because the two guards fail differently: one keeps
    control characters out of a remote command line, the other keeps the path
    absolute and inside the account.
    """

    with pytest.raises(ValueError, match=re.escape(message)):
        build(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        # Every recorded path must be absolute and normalized.
        ("data_dir", Path("home/rcp/rcp-server/data"), "must be absolute and normalized"),
        (
            "data_dir",
            Path("/home/rcp/rcp-server/../data"),
            "must be absolute and normalized",
        ),
        # Server-owned children may not escape the server root.
        ("projects_root", Path("/home/rcp/elsewhere"), "must stay below the server root"),
        # Server-owned children may not nest inside one another.
        (
            "projects_root",
            Path("/home/rcp/rcp-server/data/projects"),
            "must not overlap",
        ),
        # The control socket belongs directly below the runtime directory.
        (
            "control_socket",
            Path("/run/rcp/nested/control.sock"),
            "directly below the runtime directory",
        ),
    ],
)
def test_layout_rejects_a_path_set_that_breaks_its_own_containment(
    field: str,
    value: Path,
    message: str,
) -> None:
    """The layout validates itself on construction, so each guard needs a case.

    Only the valid default is exercised elsewhere, which leaves every branch of
    __post_init__ unproven.
    """

    with pytest.raises(ValueError, match=message):
        replace(DEFAULT_SERVER_LAYOUT, **{field: value})


def test_layout_cannot_be_rewritten_after_it_validates() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_SERVER_LAYOUT.data_dir = Path("/tmp/elsewhere")  # type: ignore[misc]


def test_remote_roots_accept_a_home_containing_a_space() -> None:
    assert str(remote_credentials_root("/srv/a b/alice")).startswith("/srv/a b/alice/")
    assert str(remote_projects_root("/srv/a b/alice")).startswith("/srv/a b/alice/")


def test_installed_config_round_trips_every_fixed_path() -> None:
    config = _public_config(installation_id="70994440-4c57-41b0-a2f6-8878856db969")
    content = render_installed_server_config(config)

    assert parse_installed_server_config(content) == config
    assert config.paths == ServerPathsConfig.from_layout()
    assert "private" not in content.lower()
    assert "password" not in content.lower()


@pytest.mark.parametrize(
    ("origin", "authentication", "fingerprint"),
    (
        ("https://github.com/example/research-control-panel", "public", None),
        ("git@github.com:example/research-control-panel.git", "deploy_key", "SHA256:" + "A" * 43),
    ),
)
def test_source_config_accepts_only_github_main_and_matching_public_auth(
    origin: str,
    authentication: str,
    fingerprint: str | None,
) -> None:
    source = ServerSourceConfig.model_validate(
        {
            "origin": origin,
            "branch": "main",
            "authentication": authentication,
            "public_key_fingerprint": fingerprint,
        }
    )

    assert source.origin == origin


@pytest.mark.parametrize(
    "source",
    (
        {
            "origin": "https://token@github.com/example/research-control-panel.git",
            "authentication": "public",
        },
        {
            "origin": "https://gitlab.com/example/research-control-panel.git",
            "authentication": "public",
        },
        {
            "origin": "https://github.com/example/research-control-panel.git?ref=main",
            "authentication": "public",
        },
        {
            "origin": "https://github.com/example/research-control-panel.git",
            "authentication": "public",
            "public_key_fingerprint": "SHA256:" + "A" * 43,
        },
        {
            "origin": "git@github.com:example/research-control-panel.git",
            "authentication": "deploy_key",
        },
        {
            "origin": "git@github.com:example/research-control-panel.git",
            "authentication": "public",
        },
        {
            "origin": "https://github.com/example/research-control-panel.git",
            "authentication": "deploy_key",
            "public_key_fingerprint": "SHA256:" + "A" * 43,
        },
    ),
)
def test_source_config_rejects_other_hosts_credentials_and_mismatched_auth(
    source: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ServerSourceConfig.model_validate(source)


def test_installed_config_is_closed_to_path_drift_and_unknown_fields() -> None:
    payload = _public_config().model_dump(mode="json")
    paths = dict(payload["paths"])
    paths["data_dir"] = "/tmp/rcp-data"
    payload["paths"] = paths

    with pytest.raises(ValidationError, match="accepted fixed layout"):
        InstalledServerConfig.model_validate(payload)

    payload = _public_config().model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        InstalledServerConfig.model_validate(payload)


def test_explicit_empty_installation_id_fails_closed() -> None:
    with pytest.raises(ValidationError, match="installation id must be a canonical UUID4"):
        _public_config(installation_id="")


def test_config_ownership_resolves_root_and_the_rcp_primary_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accounts = {
        "root": SimpleNamespace(pw_uid=0, pw_gid=0),
        "rcp": SimpleNamespace(pw_uid=901, pw_gid=902),
    }
    monkeypatch.setattr(server_config.pwd, "getpwnam", accounts.__getitem__)

    assert server_config._expected_config_ownership() == (0, 902)


def test_atomic_config_write_sets_mode_and_preserves_installation_identity(
    tmp_path: Path,
    configured_test_ownership: None,
) -> None:
    path = tmp_path / "server.toml"
    config = _public_config(installation_id="70994440-4c57-41b0-a2f6-8878856db969")

    write_installed_server_config(config, path)

    assert load_installed_server_config(path) == config
    assert stat.S_IMODE(path.stat().st_mode) == SERVER_CONFIG_MODE
    assert (path.stat().st_uid, path.stat().st_gid) == (os.getuid(), os.getgid())
    assert list(tmp_path.glob(".server.toml.*")) == []

    replacement = config.model_copy(
        update={
            "source": ServerSourceConfig(
                origin="git@github.com:example/research-control-panel.git",
                authentication="deploy_key",
                public_key_fingerprint="SHA256:" + "B" * 43,
            )
        }
    )
    write_installed_server_config(replacement, path)
    assert load_installed_server_config(path) == replacement

    changed_identity = replacement.model_copy(update={"installation_id": str(uuid.uuid4())})
    with pytest.raises(ValueError, match="cannot change installation_id"):
        write_installed_server_config(changed_identity, path)
    assert load_installed_server_config(path) == replacement


def test_config_reader_and_writer_reject_symlink_targets(
    tmp_path: Path,
    configured_test_ownership: None,
) -> None:
    target = tmp_path / "target.toml"
    target.write_text(render_installed_server_config(_public_config()), encoding="utf-8")
    link = tmp_path / "server.toml"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file"):
        load_installed_server_config(link)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        write_installed_server_config(_public_config(), link)


def test_config_reader_rejects_wrong_mode_owner_and_symlinked_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_test_ownership: None,
) -> None:
    path = tmp_path / "server.toml"
    write_installed_server_config(_public_config(), path)

    path.chmod(0o600)
    with pytest.raises(ValueError, match="mode 0640"):
        load_installed_server_config(path)
    path.chmod(SERVER_CONFIG_MODE)

    monkeypatch.setattr(
        "rcp.server_ops.config._expected_config_ownership",
        lambda: (os.getuid() + 1, os.getgid()),
    )
    with pytest.raises(ValueError, match="wrong owner or reader group"):
        load_installed_server_config(path)

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestry cannot contain a symlink"):
        write_installed_server_config(_public_config(), linked_parent / "server.toml")


def test_systemd_asset_uses_the_fixed_non_reloading_service_boundary() -> None:
    unit = server_service_unit_text()

    for directive in (
        "User=rcp",
        "Group=rcp",
        "WorkingDirectory=/etc/rcp/current",
        "Environment=RCP_DATA_DIR=/home/rcp/rcp-server/data",
        "ExecStart=/usr/local/bin/rcp serve --host 127.0.0.1 --port 8421 --web-assets prebuilt",
        "RuntimeDirectory=rcp",
        "RuntimeDirectoryMode=0700",
        "UMask=0077",
        "NoNewPrivileges=true",
    ):
        assert directive in unit
    assert "--reload" not in unit
    assert "0.0.0.0" not in unit
    assert "ProtectHome=true" not in unit
