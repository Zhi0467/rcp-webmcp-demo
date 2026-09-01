from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from challenge.gateway import (
    GatewayHttpLimits,
    GatewayOptions,
    _positive_env_int,
    create_gateway,
)
from challenge.gateway_state import COOKIE_NAME, DemoSessionRegistry, GatewayLimits


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path in {"/", "/artifact"}:
            body = b"<!doctype html><html><body><main>RCP</main></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._echo(b"")

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        self._echo(self.rfile.read(length))

    def _echo(self, body: bytes) -> None:
        payload = json.dumps(
            {
                "method": self.command,
                "path": self.path,
                "body": body.decode("utf-8"),
                "cookie": self.headers.get("cookie"),
                "demo_header": self.headers.get("x-demo"),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Set-Cookie", "upstream_cookie=kept; Path=/; SameSite=Lax")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *args: object) -> None:
        return


@pytest.fixture
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


class FakeChild:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


class FakeLease:
    def __init__(self, base_url: str) -> None:
        self.child = FakeChild(base_url)
        self.released = 0

    def release(self) -> None:
        self.released += 1


class FakeProcesses:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.acquired = []
        self.leases: list[FakeLease] = []
        self.stopped: list[str] = []
        self.closed = 0
        self.reaped = 0

    def acquire(self, session):
        self.acquired.append(session)
        lease = FakeLease(self.base_url)
        self.leases.append(lease)
        return lease

    def active_session_ids(self) -> set[str]:
        return {session.session_id for session in self.acquired}

    def stop_quiescent(self, session_id: str) -> bool:
        self.stopped.append(session_id)
        return True

    def child_count(self) -> int:
        return len(self.active_session_ids())

    def close_all(self) -> None:
        self.closed += 1

    def reap_idle(self) -> list[str]:
        self.reaped += 1
        return []


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "state.txt").write_text("clean", encoding="utf-8")
    return fixture


def _gateway(
    tmp_path: Path,
    fixture_root: Path,
    upstream: str,
    *,
    registry_limits: GatewayLimits | None = None,
    http_limits: GatewayHttpLimits | None = None,
):
    registry = DemoSessionRegistry(
        tmp_path / "gateway",
        fixture_root,
        limits=registry_limits,
    )
    processes = FakeProcesses(upstream)
    app = create_gateway(
        registry,
        processes,  # type: ignore[arg-type]
        options=GatewayOptions(
            secure_cookie=False,
            http_limits=http_limits or GatewayHttpLimits(),
        ),
    )
    return app, registry, processes


def test_gateway_persists_one_cookie_and_proxies_without_leaking_it(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    app, _registry, processes = _gateway(tmp_path, fixture_root, upstream)
    with TestClient(app) as client:
        first = client.get("/")
        cookie = client.cookies.get(COOKIE_NAME)

        assert first.status_code == 200
        assert "Start over demo" in first.text
        assert cookie is not None
        assert first.headers["origin-agent-cluster"] == "?1"
        assert first.headers["x-content-type-options"] == "nosniff"
        assert "HttpOnly" in first.headers["set-cookie"]
        assert "SameSite=strict" in first.headers["set-cookie"]
        client.cookies.set("ordinary_cookie", "forwarded")
        echo = client.post("/api/echo?part=one", content="payload", headers={"X-Demo": "yes"})

        assert echo.status_code == 200
        assert echo.json() == {
            "method": "POST",
            "path": "/api/echo?part=one",
            "body": "payload",
            "cookie": "ordinary_cookie=forwarded",
            "demo_header": "yes",
        }
        assert client.cookies.get("upstream_cookie") == "kept"
        assert client.cookies.get(COOKIE_NAME) == cookie
        assert processes.acquired[0].session_id == processes.acquired[1].session_id
        assert all(lease.released == 1 for lease in processes.leases)
    assert processes.closed == 1


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/projects"),
        ("POST", "/api/project-setup/create"),
        ("POST", "/api/project-provisioning/requests"),
        ("POST", "/api/project-transfers/source-requests"),
        ("GET", "/api/native/project-transfers/source-requests/request-1/archive"),
        ("DELETE", "/api/projects/project-1"),
        ("POST", "/api/projects/project-1/leave"),
        ("PUT", "/api/projects/project-1/settings"),
        ("POST", "/api/projects/project-1/machines/laptop/providers/rcp-demo/resolve"),
    ],
)
def test_local_machine_management_routes_are_blocked_before_session_creation(
    tmp_path: Path,
    fixture_root: Path,
    upstream: str,
    method: str,
    path: str,
) -> None:
    app, registry, processes = _gateway(tmp_path, fixture_root, upstream)
    with TestClient(app) as client:
        refused = client.request(method, path, json={})

        assert refused.status_code == 403
        assert "unavailable in this synthetic demo" in refused.json()["detail"]
        assert client.cookies.get(COOKIE_NAME) is None
        assert processes.acquired == []
        assert list(registry.sessions_root.iterdir()) == []


def test_project_research_routes_remain_available_through_the_gateway(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    app, _registry, _processes = _gateway(tmp_path, fixture_root, upstream)
    with TestClient(app) as client:
        response = client.post(
            "/api/projects/project-1/tasks/node_chat",
            content="{}",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        assert response.json()["path"] == "/api/projects/project-1/tasks/node_chat"


def test_start_over_control_is_injected_only_into_the_top_level_page(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    app, _registry, _processes = _gateway(tmp_path, fixture_root, upstream)
    with TestClient(app) as client:
        top_level = client.get("/").text
        assert "Start over demo" in top_level
        assert 'data-rcp-public-demo="true"' in top_level
        artifact = client.get("/artifact")

        assert artifact.status_code == 200
        assert "Start over demo" not in artifact.text
        assert 'data-rcp-public-demo="true"' not in artifact.text


def test_two_browsers_receive_different_copies_and_refresh_keeps_progress(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    app, registry, processes = _gateway(tmp_path, fixture_root, upstream)
    with TestClient(app) as first, TestClient(app) as second:
        assert first.get("/").status_code == 200
        first_cookie = first.cookies.get(COOKIE_NAME)
        assert second.get("/").status_code == 200
        second_cookie = second.cookies.get(COOKIE_NAME)
        assert first_cookie and second_cookie and first_cookie != second_cookie
        first_session = registry.resolve(first_cookie)
        second_session = registry.resolve(second_cookie)
        assert first_session is not None and second_session is not None
        assert first_session.root != second_session.root
        (first_session.data_root / "progress.txt").write_text("kept", encoding="utf-8")

        assert first.get("/api/echo").status_code == 200
        resumed = registry.resolve(first_cookie)
        assert resumed is not None
        assert (resumed.data_root / "progress.txt").read_text(encoding="utf-8") == "kept"
        assert not (second_session.data_root / "progress.txt").exists()
    assert len({session.session_id for session in processes.acquired}) == 2


def test_confirmed_start_over_rotates_only_the_current_browser(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    app, registry, processes = _gateway(tmp_path, fixture_root, upstream)
    with TestClient(app) as current, TestClient(app) as other:
        current.get("/")
        other.get("/")
        old_cookie = current.cookies.get(COOKIE_NAME)
        other_cookie = other.cookies.get(COOKIE_NAME)
        assert old_cookie and other_cookie
        old_session = registry.resolve(old_cookie)
        assert old_session is not None
        (old_session.data_root / "progress.txt").write_text("delete me", encoding="utf-8")
        confirmation = current.get("/__rcp_demo/start-over")
        csrf = current.cookies.get("rcp_demo_start_over")
        assert confirmation.status_code == 200
        assert "permanently deletes only this browser" in confirmation.text
        assert csrf

        reset = current.post(
            "/__rcp_demo/start-over",
            data={"csrf": csrf, "confirm": "start-over"},
            headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
            follow_redirects=False,
        )

        new_cookie = current.cookies.get(COOKIE_NAME)
        assert reset.status_code == 303
        assert new_cookie and new_cookie != old_cookie
        assert registry.resolve(old_cookie) is None
        replacement = registry.resolve(new_cookie)
        assert replacement is not None
        assert not (replacement.data_root / "progress.txt").exists()
        assert registry.resolve(other_cookie) is not None
        assert processes.stopped == [old_session.session_id]


def test_session_over_storage_cap_stops_work_but_keeps_start_over_available(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    fixture_size = sum(path.stat().st_size for path in fixture_root.rglob("*") if path.is_file())
    app, registry, _processes = _gateway(
        tmp_path,
        fixture_root,
        upstream,
        registry_limits=GatewayLimits(max_session_bytes=fixture_size + 8),
    )
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        old_cookie = client.cookies.get(COOKIE_NAME)
        assert old_cookie
        session = registry.resolve(old_cookie)
        assert session is not None
        (session.data_root / "oversized.bin").write_bytes(b"x" * 16)

        assert client.get("/api/echo").status_code == 200
        blocked = client.get("/")
        assert blocked.status_code == 503
        assert "Start over demo" in blocked.text

        confirmation = client.get("/__rcp_demo/start-over")
        csrf = client.cookies.get("rcp_demo_start_over")
        assert confirmation.status_code == 200
        assert csrf
        reset = client.post(
            "/__rcp_demo/start-over",
            data={"csrf": csrf, "confirm": "start-over"},
            headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
            follow_redirects=False,
        )

        assert reset.status_code == 303
        assert client.cookies.get(COOKIE_NAME) != old_cookie
        assert registry.resolve(old_cookie) is None
        assert client.get("/").status_code == 200


def test_new_browser_gets_retryable_busy_without_affecting_existing_session(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    app, registry, _processes = _gateway(
        tmp_path,
        fixture_root,
        upstream,
        registry_limits=GatewayLimits(max_sessions=1),
    )
    with TestClient(app) as existing, TestClient(app) as newcomer:
        assert existing.get("/").status_code == 200
        existing_cookie = existing.cookies.get(COOKIE_NAME)
        assert existing_cookie

        busy = newcomer.get("/")

        assert busy.status_code == 503
        assert busy.headers["retry-after"] == "30"
        assert newcomer.cookies.get(COOKIE_NAME) is None
        assert registry.resolve(existing_cookie) is not None
        assert existing.get("/api/echo").status_code == 200


def test_request_size_and_start_over_origin_fail_before_mutation(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    app, registry, processes = _gateway(
        tmp_path,
        fixture_root,
        upstream,
        http_limits=GatewayHttpLimits(max_request_bytes=5),
    )
    with TestClient(app) as client:
        oversized = client.post("/api/echo", content="123456")
        assert oversized.status_code == 413
        assert client.cookies.get(COOKIE_NAME) is None
        assert processes.acquired == []
        assert list(registry.sessions_root.iterdir()) == []
        client.get("/")
        cookie = client.cookies.get(COOKIE_NAME)
        client.get("/__rcp_demo/start-over")
        csrf = client.cookies.get("rcp_demo_start_over")
        assert cookie and csrf

        refused = client.post(
            "/__rcp_demo/start-over",
            data={"csrf": csrf, "confirm": "start-over"},
            headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
        )

        assert refused.status_code == 403
        assert client.cookies.get(COOKIE_NAME) == cookie
        assert registry.resolve(cookie) is not None
        assert processes.stopped == []


def test_chunked_start_over_form_is_bounded_before_session_mutation(
    tmp_path: Path, fixture_root: Path, upstream: str
) -> None:
    app, registry, processes = _gateway(tmp_path, fixture_root, upstream)
    with TestClient(app) as client:
        client.get("/")
        cookie = client.cookies.get(COOKIE_NAME)
        client.get("/__rcp_demo/start-over")
        csrf = client.cookies.get("rcp_demo_start_over")
        assert cookie and csrf

        prefix = f"csrf={csrf}&confirm=start-over&padding=".encode()
        refused = client.post(
            "/__rcp_demo/start-over",
            content=(chunk for chunk in (prefix, b"x" * (16 * 1024))),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "http://testserver",
                "Sec-Fetch-Site": "same-origin",
            },
        )

        assert "content-length" not in refused.request.headers
        assert refused.status_code == 413
        assert client.cookies.get(COOKIE_NAME) == cookie
        assert registry.resolve(cookie) is not None
        assert processes.stopped == []


def test_deployment_limit_environment_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RCP_DEMO_LIMIT", "23")
    assert _positive_env_int("RCP_DEMO_LIMIT", 10) == 23
    monkeypatch.setenv("RCP_DEMO_LIMIT", "0")
    with pytest.raises(SystemExit, match="positive integer"):
        _positive_env_int("RCP_DEMO_LIMIT", 10)
