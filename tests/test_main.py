from __future__ import annotations

import json
import re
import signal
import stat
import sys
import urllib.error
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import replace

import pytest

from rcp import __version__
from rcp.__main__ import (
    EXIT_REFUSED_OCCUPIED,
    EXIT_REFUSED_UNAVAILABLE,
    EXIT_REFUSED_VERSION,
    EXIT_REFUSED_WRONG_DATA,
    ExistingServerUnavailable,
    InstanceLockHeld,
    LaunchRefused,
    _launch_automatically,
    _open_existing_server,
    _probe_owner,
    _replace_existing_server,
    _replacement_warning,
    _run_server,
    _serve_as_owner,
    instance_lock,
    main,
)
from rcp.api.app import inspect_installed_replacement_startup
from rcp.server_runtime import (
    ServerMetadata,
    read_server_metadata,
)
from rcp.storage import AppStore


class FakeResponse:
    def __init__(self, value: dict[str, object]) -> None:
        self.payload = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _metadata(tmp_path, *, version: str = __version__, pid: int = 4321) -> ServerMetadata:
    return replace(
        ServerMetadata.create(
            tmp_path,
            host="127.0.0.1",
            port=8421,
            owner_kind="desktop",
        ),
        app_version=version,
        pid=pid,
    )


def _serve_args(**overrides) -> Namespace:
    values = {
        "command": "serve",
        "reload": False,
        "reuse_existing": True,
        "machine_readable": True,
        "owner": "desktop",
        "web_assets": "prebuilt",
        "force": False,
        "project": None,
        "host": "127.0.0.1",
        "port": 8421,
    }
    values.update(overrides)
    return Namespace(**values)


def test_instance_lock_rejects_a_second_server_for_the_same_data(tmp_path) -> None:
    with (
        instance_lock(tmp_path),
        pytest.raises(InstanceLockHeld, match="Another RCP process"),
        instance_lock(tmp_path),
    ):
        pass


def test_main_checks_installed_replacement_before_touching_the_data_root(
    tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "missing-data"
    calls = []

    def refuse(path):
        calls.append(path)
        raise RuntimeError("Installed rollback restoration is incomplete")

    monkeypatch.setattr("rcp.__main__.default_data_dir", lambda: data_dir)
    monkeypatch.setattr("rcp.__main__.inspect_installed_replacement_startup", refuse)
    monkeypatch.setattr(
        "rcp.__main__.instance_lock",
        lambda *_args, **_kwargs: pytest.fail("startup touched the data-directory lock"),
    )
    monkeypatch.setattr(sys, "argv", ["rcp", "serve"])

    with pytest.raises(SystemExit, match="rollback restoration is incomplete"):
        main()

    assert calls == [data_dir.resolve()]
    assert not data_dir.exists()


def test_installed_replacement_check_finds_rollback_without_creating_data(
    tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "missing-data"
    update_root = tmp_path / "update-checkpoints"
    update_root.mkdir()
    journal = update_root / "operation" / "rollback-journal.json"
    layout = Namespace(
        data_dir=data_dir,
        restore_operations_root=tmp_path / "restore-operations",
        update_checkpoints_root=update_root,
    )
    monkeypatch.setattr(
        "rcp.api.app._installed_rollback_journals",
        lambda path: (journal,) if path == update_root else (),
    )

    with pytest.raises(RuntimeError, match="rollback restoration is incomplete"):
        inspect_installed_replacement_startup(data_dir, layout)

    assert not data_dir.exists()


def test_installed_replacement_check_keeps_completed_restore_stopped_until_cutover(
    tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "restored-data"
    update_root = tmp_path / "update-checkpoints"
    update_root.mkdir()
    operation = Namespace(state="rollback_restoring")
    layout = Namespace(
        data_dir=data_dir,
        restore_operations_root=tmp_path / "restore-operations",
        update_checkpoints_root=update_root,
    )
    monkeypatch.setattr("rcp.api.app._installed_rollback_journals", lambda _path: ())
    monkeypatch.setattr(
        "rcp.server_ops.update_cutover.update_operation_needing_recovery",
        lambda _path, *, expected_uid: (update_root / "operation.json", operation, "a" * 64),
    )

    with pytest.raises(RuntimeError, match="rollback cutover is incomplete"):
        inspect_installed_replacement_startup(data_dir, layout)

    assert not data_dir.exists()


def test_space_init_creates_a_named_team_without_locking_or_serving(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("rcp.__main__.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "rcp.__main__.instance_lock",
        lambda *_args, **_kwargs: pytest.fail("space init took the server singleton lock"),
    )
    monkeypatch.setattr(
        "rcp.__main__._serve_as_owner",
        lambda *_args, **_kwargs: pytest.fail("space init launched the server"),
    )
    monkeypatch.setattr(sys, "argv", ["rcp", "space", "init", "--team", "--name", "Lab"])

    main()

    output = capsys.readouterr().out
    codes = re.findall(r"rcp_bootstrap_[A-Za-z0-9_-]{16}\.[A-Za-z0-9_-]{43}", output)
    assert len(codes) == 1
    store = AppStore(tmp_path / "rcp.sqlite3")
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert store.space_kind == "team"
    assert store.space_name == "Lab"
    with store.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM team_bootstrap_codes").fetchone()[0] == 1


def test_space_init_refuses_noninteractive_output_and_an_existing_space(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("rcp.__main__.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["rcp", "space", "init", "--team", "--name", "Lab"])
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    with pytest.raises(SystemExit, match="interactive terminal"):
        main()
    assert not (tmp_path / "rcp.sqlite3").exists()
    assert capsys.readouterr().out == ""

    AppStore(tmp_path / "rcp.sqlite3")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with pytest.raises(SystemExit, match="already contains a space"):
        main()
    assert capsys.readouterr().out == ""


def test_space_init_recovers_an_unclaimed_team_after_terminal_interruption(
    tmp_path, monkeypatch, capsys
) -> None:
    store, unseen_code = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Lab")
    store.path.chmod(0o644)
    monkeypatch.setattr("rcp.__main__.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys, "argv", ["rcp", "space", "init", "--team", "--name", "Lab"])

    main()

    output = capsys.readouterr().out
    assert output.startswith("Recovered unclaimed team space")
    replacement = re.findall(r"rcp_bootstrap_[A-Za-z0-9_-]{16}\.[A-Za-z0-9_-]{43}", output)
    assert len(replacement) == 1
    assert replacement[0] != unseen_code
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    member, _token = store.enroll_team_member(replacement[0], "Alice")
    assert member.display_name == "Alice"

    AppStore(tmp_path / "rcp.sqlite3")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with pytest.raises(SystemExit, match="already contains a space"):
        main()
    assert capsys.readouterr().out == ""


def test_serve_never_emits_the_team_bootstrap_credential(tmp_path, monkeypatch, capsys) -> None:
    _store, bootstrap = AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Lab")

    class FakeSocket:
        def fileno(self) -> int:
            return 17

    @contextmanager
    def fake_socket(_host, _port):
        yield FakeSocket()

    def fake_run(_args, _metadata, *, server_fd, on_ready):
        assert server_fd == 17
        on_ready()

    monkeypatch.setattr("rcp.__main__._reserved_server_socket", fake_socket)
    monkeypatch.setattr("rcp.__main__._run_server", fake_run)
    with instance_lock(tmp_path):
        _serve_as_owner(_serve_args(), tmp_path)

    output = capsys.readouterr().out
    assert json.loads(output)["outcome"] == "owned"
    assert bootstrap not in output
    assert "rcp_bootstrap_" not in output


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.10", "example.test"])
def test_team_serve_refuses_plaintext_non_loopback_bind(tmp_path, host) -> None:
    AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Lab")

    with pytest.raises(SystemExit, match="only to a loopback host"):
        _serve_as_owner(_serve_args(host=host), tmp_path)


def test_refused_team_bind_never_reaches_the_singleton_takeover(tmp_path, monkeypatch) -> None:
    """A mistyped host must not shut down the server that is already running.

    The refusal used to live after the takeover, so a bad `--host` replaced a
    healthy owner and only then exited.
    """
    AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Lab")
    taken_over = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal taken_over
        taken_over = True
        raise AssertionError("the takeover ran before the bind was validated")

    monkeypatch.setattr("rcp.__main__.ServerMetadata.create", fail_if_called)

    with pytest.raises(SystemExit, match="only to a loopback host"):
        _serve_as_owner(_serve_args(host="0.0.0.0"), tmp_path)
    assert taken_over is False


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_team_serve_accepts_loopback_bind(tmp_path, monkeypatch, host) -> None:
    AppStore.initialize_team_space(tmp_path / "rcp.sqlite3", "Lab")

    @contextmanager
    def fake_socket(actual_host, _port):
        assert actual_host == host

        class FakeSocket:
            def fileno(self) -> int:
                return 17

        yield FakeSocket()

    monkeypatch.setattr("rcp.__main__._reserved_server_socket", fake_socket)
    monkeypatch.setattr("rcp.__main__._run_server", lambda *_args, **_kwargs: None)
    _serve_as_owner(_serve_args(host=host), tmp_path)


def test_open_reuses_the_server_that_holds_the_instance_lock(tmp_path, monkeypatch) -> None:
    opened = []
    metadata = _metadata(tmp_path)
    monkeypatch.setattr("rcp.__main__.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr("rcp.__main__._probe_owner", lambda *_: (metadata, {}))
    monkeypatch.setattr(
        "rcp.__main__._open_existing_server",
        lambda *args, **kwargs: opened.append((args, kwargs)),
    )
    monkeypatch.setattr(sys, "argv", ["rcp", "open"])

    with instance_lock(tmp_path):
        main()

    assert opened == [(("127.0.0.1", 8421, None), {"expected": metadata})]


def test_open_replaces_an_unavailable_lock_owner(tmp_path, monkeypatch) -> None:
    replaced = []

    def unavailable(*_args) -> None:
        raise ExistingServerUnavailable("not answering")

    monkeypatch.setattr("rcp.__main__.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr("rcp.__main__._probe_owner", unavailable)
    monkeypatch.setattr(
        "rcp.__main__._replace_existing_server",
        lambda args, data_dir: replaced.append((args.command, data_dir)),
    )
    monkeypatch.setattr(sys, "argv", ["rcp", "open"])

    with instance_lock(tmp_path):
        main()

    assert replaced == [("open", tmp_path)]


@pytest.mark.parametrize("reload", [False, True])
def test_serve_replaces_the_server_that_holds_the_instance_lock(
    tmp_path, monkeypatch, reload
) -> None:
    replaced = []
    monkeypatch.setattr("rcp.__main__.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "rcp.__main__._replace_existing_server",
        lambda args, data_dir: replaced.append((args.command, args.reload, data_dir)),
    )
    argv = ["rcp", "serve"]
    if reload:
        argv.append("--reload")
    monkeypatch.setattr(sys, "argv", argv)

    with instance_lock(tmp_path):
        main()

    assert replaced == [("serve", reload, tmp_path)]


def test_instance_lock_enforces_private_mode_on_an_existing_file(tmp_path) -> None:
    tmp_path.mkdir(exist_ok=True)
    lock_file = tmp_path / "rcp.lock"
    lock_file.write_text("stale\n", encoding="utf-8")
    lock_file.chmod(0o644)

    with instance_lock(tmp_path):
        assert stat.S_IMODE(lock_file.stat().st_mode) == 0o600


def test_replace_existing_server_requests_shutdown_then_runs_under_lock(
    tmp_path, monkeypatch
) -> None:
    calls = []
    lock_file = tmp_path / "rcp.lock"
    tmp_path.mkdir(exist_ok=True)
    lock_file.write_text("4321\n", encoding="utf-8")
    monkeypatch.setattr("rcp.__main__._replacement_warning", lambda _: "active work")
    monkeypatch.setattr("rcp.__main__.os.kill", lambda pid, sig: calls.append((pid, sig)))

    @contextmanager
    def fake_lock(data_dir, *, timeout=0.0):
        calls.append((data_dir, timeout))
        yield

    monkeypatch.setattr("rcp.__main__.instance_lock", fake_lock)
    monkeypatch.setattr(
        "rcp.__main__._serve_as_owner",
        lambda args, data_dir: calls.append((args.command, data_dir)),
    )

    args = _serve_args(force=True, reuse_existing=False)
    _replace_existing_server(args, tmp_path)

    assert calls[0] == (4321, signal.SIGTERM)
    assert calls[1] == (tmp_path, 45.0)
    assert calls[2] == ("serve", tmp_path)


def test_replace_refuses_noninteractive_interruption_without_force(tmp_path, monkeypatch) -> None:
    (tmp_path / "rcp.lock").write_text("4321\n", encoding="utf-8")
    monkeypatch.setattr("rcp.__main__._replacement_warning", lambda _: "active work")
    monkeypatch.setattr("rcp.__main__.os.kill", lambda *_: pytest.fail("sent a signal"))

    with pytest.raises(SystemExit, match="left running"):
        _replace_existing_server(_serve_args(reuse_existing=False), tmp_path)


def test_reload_prepares_watched_frontend_before_starting_uvicorn(tmp_path, monkeypatch) -> None:
    calls = []

    @contextmanager
    def fake_assets(*, watch, mode):
        calls.append(("assets", watch, mode))
        yield

    monkeypatch.setattr("rcp.__main__.prepared_web_assets", fake_assets)
    monkeypatch.setattr(
        "rcp.__main__.uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    _run_server(
        _serve_args(reload=True, web_assets="source"),
        _metadata(tmp_path),
    )

    assert calls[0] == ("assets", True, "source")
    assert calls[1][0] == ("rcp.__main__:reload_app",)
    assert calls[1][1]["reload"] is True


def test_owner_publishes_metadata_after_lock_and_reports_owned(
    tmp_path, monkeypatch, capsys
) -> None:
    observed = {}

    class FakeSocket:
        def fileno(self) -> int:
            return 17

    @contextmanager
    def fake_socket(_host, _port):
        yield FakeSocket()

    def fake_run(_args, metadata, *, server_fd, on_ready):
        observed["metadata"] = read_server_metadata(tmp_path)
        observed["server_fd"] = server_fd
        with pytest.raises(InstanceLockHeld), instance_lock(tmp_path):
            pass
        on_ready()

    monkeypatch.setattr("rcp.__main__._reserved_server_socket", fake_socket)
    monkeypatch.setattr("rcp.__main__._run_server", fake_run)

    with instance_lock(tmp_path):
        _serve_as_owner(_serve_args(), tmp_path)

    output = json.loads(capsys.readouterr().out)
    assert output["outcome"] == "owned"
    assert output["owned"] is True
    assert output["version"] == __version__
    assert observed["metadata"].instance_id == output["instance_id"]
    assert observed["server_fd"] == 17
    assert not (tmp_path / "rcp-server.json").exists()


def test_automatic_launch_reuses_compatible_owner_without_signalling(
    tmp_path, monkeypatch, capsys
) -> None:
    metadata = _metadata(tmp_path)

    @contextmanager
    def held(_data_dir):
        raise InstanceLockHeld("held")
        yield

    monkeypatch.setattr("rcp.__main__.instance_lock", held)
    monkeypatch.setattr(
        "rcp.__main__._probe_owner",
        lambda _: (metadata, {"agent_mode": "provider"}),
    )
    monkeypatch.setattr("rcp.__main__.os.kill", lambda *_: pytest.fail("sent a signal"))

    _launch_automatically(_serve_args(), tmp_path)

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "base_url": "http://127.0.0.1:8421",
        "instance_id": metadata.instance_id,
        "outcome": "reused",
        "owned": False,
        "version": __version__,
    }


@pytest.mark.parametrize(
    ("probe", "exit_code", "outcome"),
    [
        ("version", EXIT_REFUSED_VERSION, "refused-version"),
        ("unavailable", EXIT_REFUSED_UNAVAILABLE, "refused-unavailable"),
        ("wrong-data", EXIT_REFUSED_WRONG_DATA, "refused-wrong-data"),
    ],
)
def test_automatic_launch_refusals_have_stable_codes(
    tmp_path, monkeypatch, capsys, probe, exit_code, outcome
) -> None:
    @contextmanager
    def held(_data_dir):
        raise InstanceLockHeld("held")
        yield

    monkeypatch.setattr("rcp.__main__.instance_lock", held)
    if probe == "version":
        metadata = _metadata(tmp_path, version="99.0.0")
        monkeypatch.setattr("rcp.__main__._probe_owner", lambda _: (metadata, {}))
    elif probe == "unavailable":
        monkeypatch.setattr(
            "rcp.__main__._probe_owner",
            lambda _: (_ for _ in ()).throw(ExistingServerUnavailable("not answering")),
        )
    else:
        monkeypatch.setattr(
            "rcp.__main__._probe_owner",
            lambda _: (_ for _ in ()).throw(
                LaunchRefused(
                    "refused-wrong-data",
                    EXIT_REFUSED_WRONG_DATA,
                    "different data directory",
                )
            ),
        )
    monkeypatch.setattr("rcp.__main__.os.kill", lambda *_: pytest.fail("sent a signal"))

    with pytest.raises(SystemExit) as raised:
        _launch_automatically(_serve_args(), tmp_path)

    assert raised.value.code == exit_code
    assert json.loads(capsys.readouterr().out)["outcome"] == outcome


def test_automatic_launch_refuses_an_occupied_port_distinctly(
    tmp_path, monkeypatch, capsys
) -> None:
    @contextmanager
    def occupied(_host, _port):
        raise OSError("address already in use")
        yield

    monkeypatch.setattr("rcp.__main__._reserved_server_socket", occupied)

    with pytest.raises(SystemExit) as raised, instance_lock(tmp_path):
        _serve_as_owner(_serve_args(port=18421), tmp_path)

    assert raised.value.code == EXIT_REFUSED_OCCUPIED
    assert json.loads(capsys.readouterr().out)["outcome"] == "refused-occupied"


def test_probe_rejects_a_stranger_on_the_recorded_port(tmp_path, monkeypatch) -> None:
    metadata = _metadata(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "rcp.lock").write_text(f"{metadata.pid}\n", encoding="utf-8")
    (tmp_path / "rcp-server.json").write_text(json.dumps(metadata.as_dict()), encoding="utf-8")
    monkeypatch.setattr(
        "rcp.__main__._request_json",
        lambda _: {
            "status": "ok",
            "version": metadata.app_version,
            "instance_id": "02f65d78-e74c-43c0-aeb2-b676bd6863ab",
            "data_dir_id": metadata.data_dir_id,
        },
    )

    with pytest.raises(ExistingServerUnavailable, match="not the recorded lock owner"):
        _probe_owner(tmp_path)


def test_takeover_warning_names_desktop_owned_active_work(tmp_path, monkeypatch) -> None:
    metadata = _metadata(tmp_path)
    monkeypatch.setattr(
        "rcp.__main__._probe_owner",
        lambda _: (metadata, {"active_agent_tasks": 1}),
    )

    assert _replacement_warning(tmp_path) == "RCP.app is running 1 agent task. Replace it?"


def test_open_existing_server_registers_project_and_opens_its_route(monkeypatch) -> None:
    requests = []
    responses = iter(
        [
            FakeResponse(
                {
                    "status": "ok",
                    "instance_id": "b78a82d8-b6f8-4d52-9f71-c15ed3f1dfe1",
                    "projects": 1,
                }
            ),
            FakeResponse({"id": "paper/with spaces"}),
        ]
    )

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return next(responses)

    opened = []
    monkeypatch.setattr("rcp.__main__.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("rcp.__main__.webbrowser.open", opened.append)

    _open_existing_server("127.0.0.1", 8421, "/research/project")

    assert [request.get_method() for request, _ in requests] == ["GET", "POST"]
    assert json.loads(requests[1][0].data) == {"locator": "/research/project"}
    assert dict(requests[1][0].header_items())["X-rcp-instance-id"] == (
        "b78a82d8-b6f8-4d52-9f71-c15ed3f1dfe1"
    )
    assert opened == ["http://127.0.0.1:8421/#/projects/paper%2Fwith%20spaces"]


def test_open_existing_server_marks_an_unhealthy_lock_owner_unavailable(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("rcp.__main__.urllib.request.urlopen", unavailable)

    with pytest.raises(ExistingServerUnavailable, match="no healthy server answered"):
        _open_existing_server("127.0.0.1", 8421, None)
