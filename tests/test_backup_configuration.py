from __future__ import annotations

import json
import os
import stat
import subprocess
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rcp.__main__ import build_parser
from rcp.server_ops import backup_config as backup_owner
from rcp.server_ops import config as config_owner
from rcp.server_ops.backup_config import (
    BackupConfigurationReadback,
    BackupConfigurationRefused,
    LinuxBackupConfigurationMachine,
    backup_service_unit_text,
    prepare_backup_configure_command,
    render_backup_timer_unit,
)
from rcp.server_ops.cli import CallerIdentity, run_server_command
from rcp.server_ops.config import (
    SERVER_CONFIG_SCHEMA_VERSION,
    InstalledServerConfig,
    ServerBackupConfig,
    ServerSourceConfig,
    create_installed_server_config,
    parse_installed_server_config,
    render_installed_server_config,
)

AGE_RECIPIENT = "age1qypqxpq9qcrsszg2pvxq6rs0zqg3yyc5z5tpwxqergd3c8g7rusqmwn7f2"
INSTALLATION_ID = "70994440-4c57-41b0-a2f6-8878856db969"
ROOT_IDENTITY = CallerIdentity(uid=0, username="root", host="lab.example")


def _settings(**updates: object) -> ServerBackupConfig:
    values: dict[str, object] = {
        "destination": "/srv/rcp-backups",
        "schedule": "02:00",
        "retention": 30,
        "age_recipient": AGE_RECIPIENT,
    }
    values.update(updates)
    return ServerBackupConfig.model_validate(values)


def _installed(*, backup: ServerBackupConfig | None = None) -> InstalledServerConfig:
    base = create_installed_server_config(
        source=ServerSourceConfig(
            origin="https://github.com/example/research-control-panel.git",
            authentication="public",
        ),
        installation_id=INSTALLATION_ID,
    )
    return InstalledServerConfig.model_validate(
        {**base.model_dump(mode="python"), "backup": backup}
    )


def _argv(*, schedule: str = "02:00", retention: int = 30) -> list[str]:
    return [
        "server",
        "backup",
        "configure",
        "--destination",
        "/srv/rcp-backups",
        "--recipient",
        AGE_RECIPIENT,
        "--schedule",
        schedule,
        "--retention",
        str(retention),
        "--confirm",
    ]


class RecordingMachine:
    def __init__(self, *, fail_at: str | None = None) -> None:
        self.calls: list[tuple[str, ServerBackupConfig]] = []
        self.fail_at = fail_at

    def _call(self, name: str, config: ServerBackupConfig) -> None:
        self.calls.append((name, config))
        if self.fail_at == name:
            raise BackupConfigurationRefused(
                f"Focused {name} refusal; correct the machine and rerun the same command."
            )

    def validate_destination(self, config: ServerBackupConfig) -> None:
        self._call("validate_destination", config)

    def persist_and_install(self, config: ServerBackupConfig) -> None:
        self._call("persist_and_install", config)

    def readback(self, config: ServerBackupConfig) -> BackupConfigurationReadback:
        self._call("readback", config)
        return BackupConfigurationReadback(
            config=config,
            timer_active_state="active",
            timer_unit_file_state="enabled",
        )


def _run(machine: RecordingMachine, *, machine_readable: bool) -> tuple[int, str]:
    args = build_parser().parse_args(_argv())
    args.machine_readable = machine_readable

    def handler(request, identity):
        return prepare_backup_configure_command(request, identity, machine=machine)

    output = StringIO()
    exit_code = run_server_command(
        args,
        handler=handler,
        identity=ROOT_IDENTITY,
        stream=output,
    )
    return exit_code, output.getvalue()


def test_backup_configure_is_one_explicit_copyable_request() -> None:
    request = backup_owner.ServerCommandRequest.model_validate(
        {
            "command": "server backup configure",
            "backup_destination": "/srv/rcp-backups",
            "backup_schedule": "03:17",
            "backup_retention": 45,
            "backup_age_recipient": AGE_RECIPIENT,
            "backup_confirmed": True,
        }
    )

    assert request.backup_destination == "/srv/rcp-backups"
    assert request.backup_schedule == "03:17"
    assert request.backup_retention == 45
    assert request.backup_age_recipient == AGE_RECIPIENT
    assert request.backup_confirmed is True


@pytest.mark.parametrize(
    "argv",
    (
        ["server", "backup", "configure"],
        _argv()[:-1],
        [*_argv(), "--schedule", "24:00"],
        [*_argv(), "--retention", "0"],
        [
            "server",
            "backup",
            "configure",
            "--destination",
            "relative",
            "--recipient",
            AGE_RECIPIENT,
            "--confirm",
        ],
        [
            "server",
            "backup",
            "configure",
            "--destination",
            "/srv/backup",
            "--recipient",
            "AGE-SECRET-KEY-1NOTALLOWED",
            "--confirm",
        ],
    ),
)
def test_backup_configure_parser_rejects_incomplete_or_invalid_policy(
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_backup_configure_parser_rejects_a_destination_too_long_for_progress() -> None:
    argv = _argv()
    argv[argv.index("/srv/rcp-backups")] = "/" + ("a" * 2048)

    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_interactive_and_structured_renderers_execute_the_same_policy() -> None:
    interactive_machine = RecordingMachine()
    structured_machine = RecordingMachine()

    interactive_code, interactive = _run(interactive_machine, machine_readable=False)
    structured_code, structured = _run(structured_machine, machine_readable=True)

    assert interactive_code == structured_code == 0
    assert interactive_machine.calls == structured_machine.calls
    assert [name for name, _config in interactive_machine.calls] == [
        "validate_destination",
        "persist_and_install",
        "readback",
    ]
    assert "/srv/rcp-backups" in interactive
    assert AGE_RECIPIENT in interactive
    events = [json.loads(line) for line in structured.splitlines()]
    assert events[0]["command"] == "server backup configure"
    assert events[-1]["step"]["fields"] == [
        {"name": "destination", "value": "/srv/rcp-backups"},
        {"name": "schedule", "value": "02:00"},
        {"name": "retention", "value": 30},
        {"name": "age_recipient", "value": AGE_RECIPIENT},
        {"name": "timer_active_state", "value": "active"},
        {"name": "timer_unit_file_state", "value": "enabled"},
    ]


def test_known_failure_stops_before_later_configuration_steps() -> None:
    machine = RecordingMachine(fail_at="persist_and_install")

    exit_code, output = _run(machine, machine_readable=False)

    assert exit_code == 1
    assert [name for name, _config in machine.calls] == [
        "validate_destination",
        "persist_and_install",
    ]
    assert "Focused persist_and_install refusal" in output


def test_backup_section_round_trips_and_legacy_v1_loads_unconfigured() -> None:
    configured = _installed(backup=_settings(schedule="03:17", retention=45))
    rendered = render_installed_server_config(configured)

    assert parse_installed_server_config(rendered) == configured
    assert "schema_version = 2" in rendered
    assert "[backup]" in rendered
    assert 'schedule = "03:17"' in rendered
    assert "retention = 45" in rendered
    assert "AGE-SECRET-KEY" not in rendered

    legacy = render_installed_server_config(_installed()).replace(
        "schema_version = 2", "schema_version = 1", 1
    )
    migrated = parse_installed_server_config(legacy)
    assert migrated.schema_version == SERVER_CONFIG_SCHEMA_VERSION
    assert migrated.backup is None

    legacy_with_backup = rendered.replace("schema_version = 2", "schema_version = 1", 1)
    with pytest.raises(ValueError, match="legacy.*cannot contain backup"):
        parse_installed_server_config(legacy_with_backup)


@pytest.mark.parametrize(
    "updates",
    (
        {"destination": "relative"},
        {"destination": "/"},
        {"destination": "/srv/../backup"},
        {"destination": "/" + ("a" * 2048)},
        {"schedule": "2:00"},
        {"schedule": "24:00"},
        {"retention": 0},
        {"retention": True},
        {"age_recipient": AGE_RECIPIENT[:-1] + "x"},
        {"age_recipient": "AGE-SECRET-KEY-1NOTALLOWED"},
    ),
)
def test_backup_config_rejects_unsafe_or_malformed_values(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _settings(**updates)


def test_timer_and_service_assets_share_the_configured_command_boundary() -> None:
    timer = render_backup_timer_unit("03:17")
    service = backup_service_unit_text()

    assert "OnCalendar=*-*-* 03:17:00" in timer
    assert "@RCP_BACKUP_ON_CALENDAR@" not in timer
    assert "Persistent=true" in timer
    assert "Unit=rcp-backup.service" in timer
    assert "User=rcp" in service
    assert "Group=rcp" in service
    assert "ExecStart=/usr/local/bin/rcp server backup run" in service
    assert "PrivateTmp" not in service
    assert "AGE-SECRET-KEY" not in service + timer


def test_destination_probe_uses_the_rcp_account_and_bounded_internal_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    account = SimpleNamespace(pw_uid=501, pw_name="rcp", pw_dir="/srv/rcp")
    layout = SimpleNamespace(
        service_account="rcp",
        service_home=Path("/srv/rcp"),
    )
    machine = LinuxBackupConfigurationMachine(layout)
    monkeypatch.setattr(backup_owner.pwd, "getpwnam", lambda _name: account)

    def fake_run(argv, **_kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(backup_owner.subprocess, "run", fake_run)
    config = _settings(destination=str(tmp_path))

    machine.validate_destination(config)

    assert calls == [
        (
            "runuser",
            "--user",
            "rcp",
            "--",
            "env",
            "-i",
            "HOME=/srv/rcp",
            "PATH=/usr/bin:/bin",
            os.fspath(Path(backup_owner.sys.executable)),
            "-m",
            "rcp.server_ops.backup_config",
            "--probe-destination",
            str(tmp_path),
        )
    ]


def test_destination_probe_creates_and_removes_only_its_exclusive_file(tmp_path: Path) -> None:
    existing = tmp_path / "keep.txt"
    existing.write_text("keep", encoding="utf-8")

    assert backup_owner._probe_destination_as_current_user(tmp_path) == 0
    assert list(tmp_path.iterdir()) == [existing]


def test_configured_backup_file_is_written_and_reloaded_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "etc" / "rcp" / "server.toml"
    path.parent.mkdir(parents=True)
    ownership = (os.getuid(), os.getgid())
    monkeypatch.setattr(config_owner, "_expected_config_ownership", lambda: ownership)
    configured = _installed(backup=_settings(schedule="03:17", retention=45))

    config_owner.write_installed_server_config(configured, path)

    assert config_owner.load_installed_server_config(path) == configured
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_root_configuration_lock_refuses_a_concurrent_holder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "etc" / "rcp" / "server.toml"
    config_path.parent.mkdir(parents=True)
    layout = SimpleNamespace(config_path=config_path)
    monkeypatch.setattr(
        backup_owner,
        "_root_ownership",
        lambda: (os.getuid(), os.getgid()),
    )

    with (
        backup_owner.backup_configuration_lock(layout),
        pytest.raises(BackupConfigurationRefused, match="Another.*operation"),
        backup_owner.backup_configuration_lock(layout),
    ):
        raise AssertionError("the second holder must not enter")


def test_pending_configuration_recovers_before_a_new_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "etc" / "rcp" / "server.toml"
    config_path.parent.mkdir(parents=True)
    layout = SimpleNamespace(
        config_path=config_path,
        systemd_unit=tmp_path / "systemd" / "rcp.service",
    )
    ownership = (os.getuid(), os.getgid())
    monkeypatch.setattr(config_owner, "_expected_config_ownership", lambda: ownership)
    monkeypatch.setattr(
        backup_owner,
        "_root_ownership",
        lambda: ownership,
    )
    monkeypatch.setattr(
        backup_owner,
        "fence_backup_timer_before_unit_change",
        lambda: None,
    )
    config_owner.write_installed_server_config(_installed(), config_path)
    first = _settings(schedule="03:17", retention=45)
    second = _settings(schedule="04:29", retention=60)
    converged: list[ServerBackupConfig] = []

    def crash_then_converge(installed, actual_layout) -> None:
        assert installed.backup is not None
        converged.append(installed.backup)
        if len(converged) == 1:
            raise OSError("injected publication crash")
        config_owner.write_installed_server_config(installed, actual_layout.config_path)

    monkeypatch.setattr(
        backup_owner,
        "_converge_installed_backup_configuration",
        crash_then_converge,
    )
    machine = LinuxBackupConfigurationMachine(layout)

    with pytest.raises(BackupConfigurationRefused, match="could not finish publishing"):
        machine.persist_and_install(first)
    pending = backup_owner._pending_backup_configuration_path(layout)
    assert pending.is_file()
    assert config_owner.load_installed_server_config(config_path).backup is None

    machine.persist_and_install(second)

    assert converged == [first, first, second]
    assert config_owner.load_installed_server_config(config_path).backup == second
    assert not pending.exists()


def test_failed_pre_fence_mutates_neither_journal_nor_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = SimpleNamespace(
        config_path=Path("/etc/rcp/server.toml"),
        systemd_unit=Path("/etc/systemd/system/rcp.service"),
    )
    mutations: list[str] = []
    monkeypatch.setattr(
        backup_owner,
        "backup_configuration_lock",
        lambda _layout: nullcontext(),
    )
    monkeypatch.setattr(
        backup_owner,
        "recover_pending_backup_configuration",
        lambda _layout: None,
    )
    monkeypatch.setattr(
        backup_owner,
        "load_installed_server_config",
        lambda _path: _installed(),
    )

    def refuse_fence() -> None:
        raise backup_owner.InstallRefused("The existing backup timer could not be stopped.")

    monkeypatch.setattr(backup_owner, "fence_backup_timer_before_unit_change", refuse_fence)
    monkeypatch.setattr(
        backup_owner,
        "_write_pending_backup_configuration",
        lambda *_args: mutations.append("journal"),
    )
    monkeypatch.setattr(
        backup_owner,
        "_converge_installed_backup_configuration",
        lambda *_args: mutations.append("units"),
    )

    with pytest.raises(BackupConfigurationRefused, match="could not be stopped"):
        LinuxBackupConfigurationMachine(layout).persist_and_install(_settings())
    assert mutations == []


def test_persist_preserves_identity_and_activates_only_after_first_backup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _installed()
    config = _settings(schedule="03:17", retention=45)
    layout = SimpleNamespace(
        config_path=Path("/etc/rcp/server.toml"),
        systemd_unit=Path("/etc/systemd/system/rcp.service"),
    )
    machine = LinuxBackupConfigurationMachine(layout)
    writes: list[InstalledServerConfig] = []
    units: list[dict[str, object]] = []
    events: list[str] = []

    def record_journal(value, actual_layout) -> None:
        events.append("journal")
        assert value.backup == config
        assert actual_layout is layout

    monkeypatch.setattr(
        backup_owner,
        "backup_configuration_lock",
        lambda _layout: nullcontext(),
    )
    monkeypatch.setattr(
        backup_owner,
        "recover_pending_backup_configuration",
        lambda _layout: events.append("recover"),
    )
    monkeypatch.setattr(backup_owner, "load_installed_server_config", lambda _path: base)
    monkeypatch.setattr(
        backup_owner,
        "fence_backup_timer_before_unit_change",
        lambda: events.append("pre_fence"),
    )
    monkeypatch.setattr(
        backup_owner,
        "_write_pending_backup_configuration",
        record_journal,
    )
    monkeypatch.setattr(
        backup_owner,
        "write_installed_server_config",
        lambda value, _path: (events.append("config"), writes.append(value)),
    )
    monkeypatch.setattr(
        backup_owner,
        "install_backup_unit_files",
        lambda **kwargs: (events.append("units"), units.append(kwargs)),
    )
    monkeypatch.setattr(
        backup_owner,
        "reload_and_disable_backup_timer",
        lambda: events.append("post_fence"),
    )
    monkeypatch.setattr(
        backup_owner,
        "_readback_backup_configuration",
        lambda _config, _layout, *, expected_enabled=True: events.append(
            "readback_enabled" if expected_enabled else "readback_disabled"
        ),
    )
    monkeypatch.setattr(
        backup_owner,
        "run_backup_service_once",
        lambda: events.append("first_backup"),
    )
    monkeypatch.setattr(
        backup_owner,
        "enable_backup_timer",
        lambda: events.append("enable"),
    )
    monkeypatch.setattr(
        backup_owner,
        "_clear_pending_backup_configuration",
        lambda _layout: events.append("clear"),
    )

    machine.persist_and_install(config)

    assert len(writes) == 1
    assert writes[0].installation_id == INSTALLATION_ID
    assert writes[0].source == base.source
    assert writes[0].paths == base.paths
    assert writes[0].backup == config
    assert units[0]["layout"] is layout
    assert "OnCalendar=*-*-* 03:17:00" in str(units[0]["timer_content"])
    assert events == [
        "recover",
        "pre_fence",
        "journal",
        "units",
        "post_fence",
        "config",
        "readback_disabled",
        "first_backup",
        "enable",
        "readback_enabled",
        "clear",
    ]


def test_readback_requires_the_same_timer_text_and_enabled_systemd_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _settings(schedule="03:17", retention=45)
    installed = _installed(backup=config)
    layout = SimpleNamespace(
        config_path=Path("/etc/rcp/server.toml"),
        systemd_unit=Path("/etc/systemd/system/rcp.service"),
    )
    machine = LinuxBackupConfigurationMachine(layout)
    checked: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        backup_owner,
        "load_installed_server_config",
        lambda _path: installed,
    )
    monkeypatch.setattr(
        backup_owner,
        "_require_root_unit",
        lambda path, expected: checked.append((path, expected)),
    )
    monkeypatch.setattr(
        backup_owner,
        "read_systemd_unit_state",
        lambda _unit: ("active", "enabled"),
    )

    readback = machine.readback(config)

    assert readback == BackupConfigurationReadback(
        config=config,
        timer_active_state="active",
        timer_unit_file_state="enabled",
    )
    assert checked[0][0] == Path("/etc/systemd/system/rcp-backup.timer")
    assert "OnCalendar=*-*-* 03:17:00" in checked[0][1]

    monkeypatch.setattr(
        backup_owner,
        "read_systemd_unit_state",
        lambda _unit: ("inactive", "disabled"),
    )
    with pytest.raises(BackupConfigurationRefused, match="not both active and enabled"):
        machine.readback(config)


def test_failed_first_backup_fences_the_timer_and_keeps_the_pending_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _settings()
    installed = _installed(backup=config)
    layout = SimpleNamespace(
        config_path=Path("/etc/rcp/server.toml"),
        systemd_unit=Path("/etc/systemd/system/rcp.service"),
    )
    events: list[str] = []
    monkeypatch.setattr(backup_owner, "install_backup_unit_files", lambda **_kwargs: None)
    monkeypatch.setattr(backup_owner, "reload_and_disable_backup_timer", lambda: None)
    monkeypatch.setattr(backup_owner, "write_installed_server_config", lambda *_args: None)
    monkeypatch.setattr(
        backup_owner, "_readback_backup_configuration", lambda *_args, **_kwargs: None
    )

    def fail_first_backup() -> None:
        events.append("first_backup")
        raise backup_owner.InstallRefused("injected first backup failure")

    monkeypatch.setattr(backup_owner, "run_backup_service_once", fail_first_backup)
    monkeypatch.setattr(backup_owner, "enable_backup_timer", lambda: events.append("enable"))
    monkeypatch.setattr(
        backup_owner,
        "fence_backup_timer_before_unit_change",
        lambda: events.append("fence"),
    )

    with pytest.raises(backup_owner.InstallRefused, match="injected first backup failure"):
        backup_owner._converge_installed_backup_configuration(installed, layout)

    assert events == ["first_backup", "fence"]


def test_install_reactivation_runs_backup_before_enabling_a_disabled_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _settings()
    installed = _installed(backup=config)
    layout = SimpleNamespace(config_path=Path("/etc/rcp/server.toml"))
    events: list[str] = []
    monkeypatch.setattr(backup_owner, "backup_configuration_lock", lambda _layout: nullcontext())
    monkeypatch.setattr(
        backup_owner,
        "recover_pending_backup_configuration",
        lambda _layout: events.append("recover"),
    )
    monkeypatch.setattr(backup_owner, "load_installed_server_config", lambda _path: installed)

    def readback(_config, _layout, *, expected_enabled=True):
        events.append("readback_enabled" if expected_enabled else "readback_disabled")
        if expected_enabled and events.count("readback_enabled") == 1:
            raise BackupConfigurationRefused("timer is still disabled")
        return BackupConfigurationReadback(
            config=config,
            timer_active_state="active" if expected_enabled else "inactive",
            timer_unit_file_state="enabled" if expected_enabled else "disabled",
        )

    monkeypatch.setattr(backup_owner, "_readback_backup_configuration", readback)
    monkeypatch.setattr(
        backup_owner,
        "run_backup_service_once",
        lambda: events.append("first_backup"),
    )
    monkeypatch.setattr(
        backup_owner,
        "enable_backup_timer",
        lambda: events.append("enable"),
    )

    result = backup_owner.activate_configured_backup_timer(layout)

    assert result.timer_active_state == "active"
    assert events == [
        "recover",
        "readback_enabled",
        "readback_disabled",
        "first_backup",
        "enable",
        "readback_enabled",
    ]
