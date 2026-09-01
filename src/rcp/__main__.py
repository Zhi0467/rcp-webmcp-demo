from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from rcp import __version__
from rcp.api import create_app
from rcp.api.app import default_data_dir, inspect_installed_replacement_startup
from rcp.limits import (
    BROWSER_OPEN_DELAY_SECONDS,
    SERVER_HEALTH_REQUEST_TIMEOUT_SECONDS,
    SERVER_LOCK_DEFAULT_TIMEOUT_SECONDS,
    SERVER_LOCK_OWNER_READ_ATTEMPTS,
    SERVER_LOCK_POLL_INTERVAL_SECONDS,
    SERVER_SHUTDOWN_TIMEOUT_SECONDS,
)
from rcp.server_ops.cli import add_server_parser, run_server_command
from rcp.server_runtime import (
    ServerMetadata,
    ServerMetadataError,
    capture_installed_release_identity,
    data_dir_identity,
    installed_control_socket_path,
    published_server_metadata,
    read_server_metadata,
)
from rcp.storage import AppStore
from rcp.web_assets import WebBuildError, prepared_web_assets

RELOAD_PROJECT_ENV = "RCP_RELOAD_PROJECT"
RELOAD_METADATA_ENV = "RCP_RELOAD_SERVER_METADATA"
RELOAD_ACCEPTANCE_AGENT_ENV = "RCP_RELOAD_ACCEPTANCE_AGENT"
_LOCKED_MESSAGE = "Another RCP process is already using this app data directory."

EXIT_REFUSED_VERSION = 20
EXIT_REFUSED_UNAVAILABLE = 21
EXIT_REFUSED_OCCUPIED = 22
EXIT_REFUSED_WRONG_DATA = 23


class InstanceLockHeld(RuntimeError):
    pass


class ExistingServerError(RuntimeError):
    pass


class ExistingServerUnavailable(ExistingServerError):
    pass


class LaunchRefused(ExistingServerUnavailable):
    def __init__(
        self,
        outcome: str,
        exit_code: int,
        message: str,
        *,
        metadata: ServerMetadata | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.exit_code = exit_code
        self.metadata = metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rcp")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "open"):
        command = subcommands.add_parser(name)
        command.add_argument(
            "project",
            nargs="?",
            help="Optional project root, .research directory, or manifest.toml to register and open",
        )
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=8421)
        command.add_argument(
            "--web-assets",
            choices=("source", "prebuilt"),
            default="source",
            help="Build from the checkout or require an already packaged frontend bundle",
        )
        command.add_argument(
            "--force",
            action="store_true",
            help="Replace an existing server without asking about active work",
        )
        if name == "serve":
            command.add_argument(
                "--acceptance-agent",
                action="store_true",
                help=("Use the explicit local CPU-only acceptance agent instead of a provider"),
            )
            command.add_argument(
                "--reload",
                action="store_true",
                help="Reload Python and rebuild the frontend when their sources change",
            )
            command.add_argument(
                "--reuse-existing",
                action="store_true",
                help="Reuse a compatible owner or refuse; never replace it",
            )
            command.add_argument(
                "--machine-readable",
                action="store_true",
                help="Write the launch outcome as one JSON object on stdout",
            )
            command.add_argument(
                "--owner",
                choices=("cli", "desktop"),
                default="cli",
                help=argparse.SUPPRESS,
            )
    space = subcommands.add_parser("space")
    space_commands = space.add_subparsers(dest="space_command", required=True)
    init = space_commands.add_parser("init")
    init.add_argument(
        "--team",
        action="store_true",
        required=True,
        help="Initialize an explicitly named team space",
    )
    init.add_argument("--name", required=True, help="Human-readable team space name")
    add_server_parser(subcommands)
    return parser


def reload_app() -> FastAPI:
    """Rebuild the app inside uvicorn's reloader, which imports it on every restart."""
    raw_metadata = os.environ.get(RELOAD_METADATA_ENV)
    metadata = ServerMetadata.from_dict(json.loads(raw_metadata)) if raw_metadata else None
    return create_app(
        os.environ.get(RELOAD_PROJECT_ENV) or None,
        instance_metadata=metadata,
        acceptance_agent=os.environ.get(RELOAD_ACCEPTANCE_AGENT_ENV) == "1",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "server":
        exit_code = run_server_command(args)
        if exit_code:
            raise SystemExit(exit_code)
        return
    data_dir = default_data_dir().expanduser().resolve()
    if args.command == "space":
        _run_space_command(args, data_dir)
        return
    try:
        inspect_installed_replacement_startup(data_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    _require_team_bind_is_loopback(args, data_dir)
    if args.command == "serve" and args.reuse_existing:
        if args.force:
            raise SystemExit("--reuse-existing and --force cannot be used together.")
        _launch_automatically(args, data_dir)
        return

    try:
        with instance_lock(data_dir):
            _serve_as_owner(args, data_dir)
    except InstanceLockHeld:
        if args.command == "open":
            try:
                metadata, _ = _probe_owner(data_dir)
                _open_existing_server(
                    metadata.host,
                    metadata.port,
                    args.project,
                    expected=metadata,
                )
                return
            except ExistingServerUnavailable as exc:
                print(f"The lock-owning RCP server is unavailable: {exc}", file=sys.stderr)
        _replace_existing_server(args, data_dir)


def _run_space_command(args: argparse.Namespace, data_dir: Path) -> None:
    if args.space_command != "init" or not args.team:  # pragma: no cover - argparse owns this
        raise SystemExit("Only explicit team-space initialization is supported.")
    if not sys.stdout.isatty():
        raise SystemExit(
            "Team-space initialization requires an interactive terminal so its bootstrap "
            "code cannot be redirected into a service log."
        )
    database_path = data_dir / "rcp.sqlite3"
    recovering = database_path.exists()
    try:
        store, bootstrap_code = AppStore.initialize_team_space(
            database_path,
            args.name,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        database_path.chmod(0o600)
    except OSError as exc:
        raise SystemExit(
            "Team-space initialization could not restrict the database to its owner."
        ) from exc
    action = "Recovered unclaimed" if recovering else "Initialized"
    print(f"{action} team space {store.space_name!r} ({store.space_id}).")
    print("One-time bootstrap code (shown once):")
    print(bootstrap_code)


def _launch_automatically(args: argparse.Namespace, data_dir: Path) -> None:
    try:
        with instance_lock(data_dir):
            _serve_as_owner(args, data_dir)
            return
    except InstanceLockHeld:
        pass
    except OSError as exc:
        refusal = LaunchRefused(
            "refused-occupied",
            EXIT_REFUSED_OCCUPIED,
            f"Cannot bind {_base_url(args.host, args.port)}: {exc}",
        )
        _exit_refused(args, refusal)

    try:
        metadata, health = _probe_owner(data_dir)
        if metadata.app_version != __version__:
            raise LaunchRefused(
                "refused-version",
                EXIT_REFUSED_VERSION,
                f"The running RCP version is {metadata.app_version}; this app is {__version__}.",
                metadata=metadata,
            )
        requested_agent_mode = (
            "acceptance" if getattr(args, "acceptance_agent", False) else "provider"
        )
        running_agent_mode = health.get("agent_mode")
        if running_agent_mode not in {"acceptance", "provider"}:
            raise LaunchRefused(
                "refused-unavailable",
                EXIT_REFUSED_UNAVAILABLE,
                "The running RCP server does not report a recognized agent mode.",
                metadata=metadata,
            )
        if running_agent_mode != requested_agent_mode:
            raise LaunchRefused(
                "refused-unavailable",
                EXIT_REFUSED_UNAVAILABLE,
                (
                    f"The running RCP server uses {running_agent_mode!r} agent mode; "
                    f"this launch requested {requested_agent_mode!r}."
                ),
                metadata=metadata,
            )
        if args.project:
            _register_project(
                metadata.base_url,
                args.project,
                expected_instance_id=metadata.instance_id,
            )
    except LaunchRefused as refusal:
        _exit_refused(args, refusal)
    except ExistingServerUnavailable as exc:
        _exit_refused(
            args,
            LaunchRefused(
                "refused-unavailable",
                EXIT_REFUSED_UNAVAILABLE,
                str(exc),
            ),
        )
    _emit_launch_outcome(args, "reused", metadata=metadata, owned=False)


def _require_team_bind_is_loopback(args: argparse.Namespace, data_dir: Path) -> None:
    """Refuse a routable bind for a team space.

    This runs before the singleton takeover, not inside it: a mistyped host must
    leave a healthy server running rather than shut it down and then refuse.
    """
    database_path = data_dir / "rcp.sqlite3"
    if not database_path.exists() or AppStore(database_path).space_kind != "team":
        return
    host = getattr(args, "host", None)
    if host is None:
        return
    try:
        loopback = ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        loopback = host.casefold() == "localhost"
    if not loopback:
        raise SystemExit(
            "A team space may bind only to a loopback host. Use the encrypted SSH "
            "connection for remote access; member credentials must never cross plaintext HTTP."
        )


def _serve_as_owner(args: argparse.Namespace, data_dir: Path) -> None:
    _require_team_bind_is_loopback(args, data_dir)
    control_socket = installed_control_socket_path(data_dir)
    release_identity = capture_installed_release_identity() if control_socket is not None else None
    metadata = ServerMetadata.create(
        data_dir,
        host=args.host,
        port=args.port,
        owner_kind=getattr(args, "owner", "cli"),
        control_socket=control_socket,
        running_commit=release_identity.commit if release_identity is not None else None,
        web_build_id=release_identity.web_build_id if release_identity is not None else None,
    )
    try:
        with (
            _reserved_server_socket(args.host, args.port) as server_socket,
            published_server_metadata(data_dir, metadata),
        ):
            _run_server(
                args,
                metadata,
                server_fd=server_socket.fileno(),
                on_ready=lambda: _emit_launch_outcome(args, "owned", metadata=metadata, owned=True),
            )
    except OSError as exc:
        if getattr(args, "machine_readable", False):
            _exit_refused(
                args,
                LaunchRefused(
                    "refused-occupied",
                    EXIT_REFUSED_OCCUPIED,
                    f"Cannot bind {_base_url(args.host, args.port)}: {exc}",
                ),
            )
        raise SystemExit(f"Cannot bind {_base_url(args.host, args.port)}: {exc}") from exc


def _run_server(
    args: argparse.Namespace,
    metadata: ServerMetadata,
    *,
    server_fd: int | None = None,
    on_ready: Callable[[], None] | None = None,
) -> None:
    reload = args.command == "serve" and args.reload
    try:
        with prepared_web_assets(
            watch=reload,
            mode=getattr(args, "web_assets", "source"),
        ):
            uvicorn_options: dict[str, object] = {
                "host": args.host,
                "port": args.port,
            }
            if server_fd is not None:
                uvicorn_options["fd"] = server_fd
            if reload:
                # uvicorn can only restart an app it imports itself, so reload runs
                # through the factory above and carries the project and identity in
                # the environment. Watch only the Python package; Vite owns web.
                os.environ[RELOAD_PROJECT_ENV] = args.project or ""
                os.environ[RELOAD_METADATA_ENV] = json.dumps(metadata.as_dict())
                os.environ[RELOAD_ACCEPTANCE_AGENT_ENV] = (
                    "1" if getattr(args, "acceptance_agent", False) else "0"
                )
                if on_ready:
                    on_ready()
                uvicorn.run(
                    "rcp.__main__:reload_app",
                    factory=True,
                    reload=True,
                    reload_dirs=[str(Path(__file__).resolve().parents[1])],
                    **uvicorn_options,
                )
                return
            app = create_app(
                args.project,
                instance_metadata=metadata,
                acceptance_agent=getattr(args, "acceptance_agent", False),
            )
            if args.command == "open":
                url = _project_url(args.host, args.port, app.state.default_project_id)
                threading.Timer(BROWSER_OPEN_DELAY_SECONDS, lambda: webbrowser.open(url)).start()
            if on_ready:
                on_ready()
            uvicorn.run(app, **uvicorn_options)
    except WebBuildError as exc:
        raise SystemExit(str(exc)) from exc


def _replace_existing_server(args: argparse.Namespace, data_dir: Path) -> None:
    pid = _lock_owner_pid(data_dir)
    warning = _replacement_warning(data_dir)
    if warning and not getattr(args, "force", False) and not _confirm_replacement(warning):
        raise SystemExit("The existing RCP server was left running.")
    print(
        f"Replacing the existing RCP server (PID {pid}) and preserving recoverable work...",
        file=sys.stderr,
    )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise SystemExit(f"Cannot stop the existing RCP server (PID {pid}): {exc}") from exc

    try:
        with instance_lock(data_dir, timeout=SERVER_SHUTDOWN_TIMEOUT_SECONDS):
            _serve_as_owner(args, data_dir)
    except InstanceLockHeld as exc:
        raise SystemExit(
            "The existing RCP server did not stop after a graceful shutdown request."
        ) from exc


def _replacement_warning(data_dir: Path) -> str | None:
    try:
        metadata, health = _probe_owner(data_dir)
    except ExistingServerUnavailable as exc:
        return (
            "The current RCP owner could not be verified and may still be doing work: "
            f"{exc}. Replace it?"
        )
    raw_active = health.get("active_agent_tasks", 0)
    active = raw_active if isinstance(raw_active, int) and not isinstance(raw_active, bool) else 0
    if active <= 0:
        return None
    label = "RCP.app" if metadata.owner_kind == "desktop" else "RCP"
    noun = "task" if active == 1 else "tasks"
    return f"{label} is running {active} agent {noun}. Replace it?"


def _confirm_replacement(message: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{message} [y/N] ").strip().casefold() in {"y", "yes"}
    except EOFError:
        return False


def _probe_owner(data_dir: Path) -> tuple[ServerMetadata, dict[str, object]]:
    expected_data_dir_id = data_dir_identity(data_dir)
    try:
        metadata = read_server_metadata(data_dir)
    except ServerMetadataError as exc:
        raise ExistingServerUnavailable(str(exc)) from exc
    if metadata.data_dir_id != expected_data_dir_id:
        raise LaunchRefused(
            "refused-wrong-data",
            EXIT_REFUSED_WRONG_DATA,
            "The lock owner's metadata identifies a different RCP data directory.",
            metadata=metadata,
        )
    if metadata.pid != _lock_owner_pid(data_dir):
        raise ExistingServerUnavailable("server metadata does not match the lock owner")
    try:
        health = _request_json(f"{metadata.base_url}/api/health")
    except ExistingServerError as exc:
        raise ExistingServerUnavailable(
            f"no healthy server answered at {metadata.base_url}: {exc}"
        ) from exc
    if health.get("status") != "ok":
        raise ExistingServerUnavailable("the lock owner returned an invalid health response")
    if health.get("instance_id") != metadata.instance_id:
        raise ExistingServerUnavailable("the responding server is not the recorded lock owner")
    if health.get("data_dir_id") != expected_data_dir_id:
        raise LaunchRefused(
            "refused-wrong-data",
            EXIT_REFUSED_WRONG_DATA,
            "The responding server uses a different RCP data directory.",
            metadata=metadata,
        )
    if health.get("version") != metadata.app_version:
        raise ExistingServerUnavailable("the server health and ownership metadata disagree")
    return metadata, health


def _lock_owner_pid(data_dir: Path) -> int:
    path = data_dir / "rcp.lock"
    for _ in range(SERVER_LOCK_OWNER_READ_ATTEMPTS):
        try:
            raw_pid = path.read_text(encoding="utf-8").strip()
            pid = int(raw_pid)
        except (FileNotFoundError, ValueError):
            time.sleep(SERVER_LOCK_POLL_INTERVAL_SECONDS)
            continue
        if pid > 0 and pid != os.getpid():
            return pid
        break
    raise ExistingServerUnavailable(
        "the instance lock is held, but its process identity could not be read"
    )


def _open_existing_server(
    host: str,
    port: int,
    project: str | None,
    *,
    expected: ServerMetadata | None = None,
) -> None:
    base_url = _base_url(host, port)
    try:
        health = _request_json(f"{base_url}/api/health")
    except ExistingServerError as exc:
        raise ExistingServerUnavailable(f"no healthy server answered at {base_url}: {exc}") from exc
    if health.get("status") != "ok":
        raise ExistingServerUnavailable(
            f"the server at {base_url} returned an invalid health response"
        )
    if expected and (
        health.get("instance_id") != expected.instance_id
        or health.get("version") != expected.app_version
        or health.get("data_dir_id") != expected.data_dir_id
    ):
        raise ExistingServerUnavailable("the server identity changed while opening it")

    current_instance_id = health.get("instance_id")
    if not isinstance(current_instance_id, str) or not current_instance_id:
        raise ExistingServerUnavailable("the server health omitted its instance identity")
    project_id = (
        _register_project(
            base_url,
            project,
            expected_instance_id=current_instance_id,
        )
        if project
        else None
    )
    webbrowser.open(_project_url(host, port, project_id))


def _register_project(
    base_url: str,
    project: str,
    *,
    expected_instance_id: str,
) -> str:
    try:
        card = _request_json(
            f"{base_url}/api/projects",
            payload={"locator": project},
            headers={"X-RCP-Instance-ID": expected_instance_id},
        )
    except ExistingServerError as exc:
        raise ExistingServerError(
            f"The existing RCP server could not register {project!r}: {exc}"
        ) from exc
    raw_project_id = card.get("id")
    if not isinstance(raw_project_id, str) or not raw_project_id:
        raise ExistingServerError("The existing RCP server returned an invalid project record.")
    return raw_project_id


def _project_url(host: str, port: int, project_id: str | None) -> str:
    url = _base_url(host, port)
    if project_id:
        url += f"/#/projects/{urllib.parse.quote(project_id, safe='')}"
    return url


def _base_url(host: str, port: int) -> str:
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{url_host}:{port}"


def _request_json(
    url: str,
    *,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = dict(headers or {})
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=SERVER_HEALTH_REQUEST_TIMEOUT_SECONDS
        ) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
            detail = body.get("detail") if isinstance(body, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = None
        raise ExistingServerError(str(detail or f"HTTP {exc.code}")) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExistingServerError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ExistingServerError("response was not a JSON object")
    return value


def _emit_launch_outcome(
    args: argparse.Namespace,
    outcome: str,
    *,
    metadata: ServerMetadata,
    owned: bool,
) -> None:
    if not getattr(args, "machine_readable", False):
        return
    print(
        json.dumps(
            {
                "outcome": outcome,
                "base_url": metadata.base_url,
                "instance_id": metadata.instance_id,
                "version": metadata.app_version,
                "owned": owned,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _exit_refused(args: argparse.Namespace, refusal: LaunchRefused) -> None:
    metadata = refusal.metadata
    if getattr(args, "machine_readable", False):
        print(
            json.dumps(
                {
                    "outcome": refusal.outcome,
                    "base_url": (
                        metadata.base_url if metadata else _base_url(args.host, args.port)
                    ),
                    "instance_id": metadata.instance_id if metadata else None,
                    "version": __version__,
                    "owned": False,
                    "reason": str(refusal),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(str(refusal), file=sys.stderr)
    raise SystemExit(refusal.exit_code)


@contextmanager
def _reserved_server_socket(host: str, port: int) -> Iterator[socket.socket]:
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    last_error: OSError | None = None
    for family, socktype, protocol, _, address in addresses:
        server_socket = socket.socket(family, socktype, protocol)
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(address)
            server_socket.listen()
            server_socket.set_inheritable(True)
        except OSError as exc:
            last_error = exc
            server_socket.close()
            continue
        try:
            yield server_socket
        finally:
            server_socket.close()
        return
    if last_error is not None:
        raise last_error
    raise OSError(f"No address is available for {host}:{port}.")


@contextmanager
def instance_lock(
    data_dir: Path, *, timeout: float = SERVER_LOCK_DEFAULT_TIMEOUT_SECONDS
) -> Iterator[None]:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "rcp.lock"
    with path.open("a+", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise InstanceLockHeld(_LOCKED_MESSAGE) from exc
                time.sleep(SERVER_LOCK_POLL_INTERVAL_SECONDS)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
