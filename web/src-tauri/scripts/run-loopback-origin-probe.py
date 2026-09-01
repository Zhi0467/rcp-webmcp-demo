#!/usr/bin/env python3
"""Drive two isolated HTTP origins through a real Tauri WKWebView."""

from __future__ import annotations

import argparse
import html
import http.cookies
import http.server
import ipaddress
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

COOKIE_NAME = "__Host-rcp_session"
PORT_A = 39121
PORT_B = 39122
PORT_OTHER = 39123
UUID_A = "11111111111141118111111111111111"
UUID_B = "22222222222242228222222222222222"
UUID_OTHER = "99999999999949998999999999999999"
PROBE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ProbeEvent:
    label: str
    path: str
    cookie: str | None
    expected: str | None
    allowed: bool

    @property
    def passed(self) -> bool:
        return self.allowed and self.cookie == self.expected


@dataclass
class ProbeState:
    events: list[ProbeEvent] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    failed: threading.Event = field(default_factory=threading.Event)
    completed: threading.Event = field(default_factory=threading.Event)

    def record(
        self,
        label: str,
        path: str,
        cookie: str | None,
        expected: str | None,
        *,
        allowed: bool,
    ) -> None:
        event = ProbeEvent(label, path, cookie, expected, allowed)
        with self.lock:
            self.events.append(event)
        if not event.passed:
            self.failed.set()
        elif path == "/complete":
            self.completed.set()
        expectation = expected or "<none>"
        if not allowed:
            expectation = "<blocked>"
        print(
            f"[origin-probe] {label} {path} cookie={cookie or '<none>'} expected={expectation}",
            flush=True,
        )


@dataclass(frozen=True)
class Origin:
    label: str
    host: str
    bind_hosts: tuple[str, ...]
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def cookie(self) -> str:
        return f"probe-{self.label}"


class ProbeServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        origin: Origin,
        origins: dict[str, Origin],
        state: ProbeState,
        bind_host: str,
    ) -> None:
        self.origin = origin
        self.origins = origins
        self.probe_state = state
        if ":" in bind_host:
            self.address_family = socket.AF_INET6
        super().__init__((bind_host, origin.port), ProbeHandler)

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


class ProbeHandler(http.server.BaseHTTPRequestHandler):
    server: ProbeServer

    def do_GET(self) -> None:  # noqa: N802
        parsed_path = urlsplit(self.path)
        path = parsed_path.path
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        cookie = self._session_cookie()
        expected, allowed = self._expectation(path, parsed_path.query)
        self.server.probe_state.record(
            self.server.origin.label,
            path,
            cookie,
            expected,
            allowed=allowed,
        )
        if not allowed or cookie != expected:
            self.send_error(409)
            return
        if path == "/login":
            self._redirect(
                f"{self.server.origin.url}/check-after-login",
                cookie=f"{COOKIE_NAME}={self.server.origin.cookie}; "
                "Path=/; HttpOnly; Secure; SameSite=Lax",
            )
            return
        if path == "/check-after-login":
            destination = (
                f"{self.server.origins['b'].url}/before-login"
                if self.server.origin.label == "a"
                else f"{self.server.origins['a'].url}/final"
            )
            self._redirect(destination)
            return
        if path == "/before-login":
            self._redirect(f"{self.server.origin.url}/login")
            return
        if path == "/resume":
            destination = (
                f"{self.server.origins['b'].url}/resume"
                if self.server.origin.label == "a"
                else f"{self.server.origins['a'].url}/final"
            )
            self._redirect(destination)
            return
        if path == "/final":
            self._send_status_page()
            return
        if path == "/complete":
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.send_error(404)

    def _expectation(self, path: str, query: str) -> tuple[str | None, bool]:
        label = self.server.origin.label
        if label == "other":
            return None, False
        if path == "/login":
            return None, True
        if label == "b" and path == "/before-login":
            return None, True
        if path in {"/check-after-login", "/resume"}:
            return self.server.origin.cookie, True
        if label == "a" and path == "/final":
            return self.server.origin.cookie, True
        if label == "a" and path == "/complete":
            http_only = parse_qs(query, strict_parsing=True).get("http_only")
            return self.server.origin.cookie, http_only == ["pass"]
        return None, False

    def _redirect(self, location: str, *, cookie: str | None = None) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _session_cookie(self) -> str | None:
        parsed = http.cookies.SimpleCookie()
        parsed.load(self.headers.get("Cookie", ""))
        morsel = parsed.get(COOKIE_NAME)
        return morsel.value if morsel is not None else None

    def _send_status_page(self) -> None:
        origin = self.server.origin
        other = self.server.origins["other"]
        body = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>RCP origin probe — {html.escape(origin.label.upper())}</title>
<style>
body {{ font: 20px system-ui; max-width: 760px; margin: 64px auto; line-height: 1.5; }}
.pass {{ color: #176b3a; }} .fail {{ color: #a3261e; }}
a {{ display: inline-block; margin: 8px 12px 8px 0; padding: 10px 14px; border: 1px solid; }}
code {{ font-size: 16px; }}
</style>
<h1>RCP loopback origin probe</h1>
<p class="pass"><strong>The A → B → A server checks passed.</strong></p>
<p id="http-only">Checking whether JavaScript can see the HttpOnly cookie…</p>
<p>Current host: <code>{html.escape(origin.host)}</code></p>
<p>The probe will now attempt the blocked arbitrary origin and close itself.</p>
<script>
const exposed = document.cookie.includes({COOKIE_NAME!r});
const line = document.getElementById('http-only');
line.textContent = exposed
  ? 'FAIL: JavaScript can see the HttpOnly session cookie.'
  : 'PASS: JavaScript cannot see the HttpOnly session cookie.';
line.className = exposed ? 'fail' : 'pass';
setTimeout(() => {{ window.location.assign({f"{other.url}/"!r}); }}, 100);
setTimeout(() => {{
  fetch(`/complete?http_only=${{exposed ? 'fail' : 'pass'}}`, {{ cache: 'no-store' }});
}}, 500);
</script>
</html>"""
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _origins(mode: str) -> dict[str, Origin]:
    if mode == "aliases":
        return {
            "a": Origin("a", f"rcp-{UUID_A}.localhost", ("127.0.0.1", "::1"), PORT_A),
            "b": Origin("b", f"rcp-{UUID_B}.localhost", ("127.0.0.1", "::1"), PORT_B),
            "other": Origin(
                "other",
                f"rcp-{UUID_OTHER}.localhost",
                ("127.0.0.1", "::1"),
                PORT_OTHER,
            ),
        }
    if mode == "canonical":
        return {
            "a": Origin("a", "localhost", ("127.0.0.1", "::1"), PORT_A),
            "b": Origin("b", "127.0.0.1", ("127.0.0.1",), PORT_B),
            "other": Origin(
                "other",
                f"rcp-{UUID_OTHER}.localhost",
                ("127.0.0.1", "::1"),
                PORT_OTHER,
            ),
        }
    return {
        "a": Origin("a", "127.0.0.2", ("127.0.0.2",), PORT_A),
        "b": Origin("b", "127.0.0.3", ("127.0.0.3",), PORT_A),
        "other": Origin("other", "127.0.0.4", ("127.0.0.4",), PORT_A),
    }


def _prove_resolution(origins: dict[str, Origin]) -> None:
    for origin in origins.values():
        addresses = socket.getaddrinfo(origin.host, origin.port, type=socket.SOCK_STREAM)
        resolved = {address[4][0] for address in addresses}
        if not resolved or not all(
            ipaddress.ip_address(address).is_loopback for address in resolved
        ):
            raise RuntimeError(f"{origin.host} did not resolve only to loopback: {resolved}")


def _assert_phase(phase: str, state: ProbeState) -> None:
    mismatches = [
        (event.label, event.path, event.cookie, event.expected)
        for event in state.events
        if not event.passed
    ]
    if mismatches:
        raise RuntimeError(f"probe phase {phase} observed invalid requests: {mismatches}")
    observed = {(event.label, event.path, event.cookie) for event in state.events}
    if phase == "login":
        required = {
            ("a", "/login", None),
            ("a", "/check-after-login", "probe-a"),
            ("b", "/before-login", None),
            ("b", "/login", None),
            ("b", "/check-after-login", "probe-b"),
            ("a", "/final", "probe-a"),
            ("a", "/complete", "probe-a"),
        }
    else:
        required = {
            ("a", "/resume", "probe-a"),
            ("b", "/resume", "probe-b"),
            ("a", "/final", "probe-a"),
            ("a", "/complete", "probe-a"),
        }
    missing = required.difference(observed)
    if missing:
        raise RuntimeError(
            f"probe phase {phase} did not observe required requests: {sorted(missing)}"
        )
    if any(event.label == "other" for event in state.events):
        raise RuntimeError("WKWebView reached an arbitrary loopback origin")


def _stop_probe(process: subprocess.Popen[bytes]) -> int:
    process.terminate()
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=5)


def _open_servers(origins: dict[str, Origin], state: ProbeState) -> list[ProbeServer]:
    servers: list[ProbeServer] = []
    try:
        for origin in origins.values():
            for bind_host in origin.bind_hosts:
                servers.append(ProbeServer(origin, origins, state, bind_host))
    except OSError:
        for server in servers:
            server.server_close()
        raise
    return servers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("aliases", "addresses", "canonical"), required=True)
    parser.add_argument("--phase", choices=("login", "resume"), required=True)
    args = parser.parse_args()

    origins = _origins(args.mode)
    _prove_resolution(origins)
    state = ProbeState()
    try:
        servers = _open_servers(origins, state)
    except OSError as error:
        print(
            f"[origin-probe] FAIL: could not bind the requested loopback address: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()

    crate = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["RCP_ORIGIN_PROBE_FIRST"] = origins["a"].url
    environment["RCP_ORIGIN_PROBE_SECOND"] = origins["b"].url
    if args.phase == "login":
        environment["RCP_ORIGIN_PROBE_LOGIN"] = "1"
    else:
        environment.pop("RCP_ORIGIN_PROBE_LOGIN", None)

    print(f"[origin-probe] {args.phase}: running automatic WKWebView drive", flush=True)
    process: subprocess.Popen[bytes] | None = None
    interrupted = False
    timed_out = False
    try:
        process = subprocess.Popen(
            [
                "cargo",
                "run",
                "--manifest-path",
                str(crate / "Cargo.toml"),
                "--example",
                "loopback_origin_probe",
            ],
            cwd=crate,
            env=environment,
        )
        deadline = time.monotonic() + PROBE_TIMEOUT_SECONDS
        while process.poll() is None:
            if state.failed.is_set() or state.completed.is_set():
                _stop_probe(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _stop_probe(process)
                break
            state.failed.wait(timeout=0.2)
    except KeyboardInterrupt:
        interrupted = True
        if process is not None and process.poll() is None:
            _stop_probe(process)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()

    if interrupted:
        print("[origin-probe] interrupted", file=sys.stderr, flush=True)
        return 130
    if timed_out:
        print(
            f"[origin-probe] FAIL: no terminal result within {PROBE_TIMEOUT_SECONDS} seconds",
            file=sys.stderr,
            flush=True,
        )
        return 1
    try:
        _assert_phase(args.phase, state)
    except RuntimeError as error:
        print(f"[origin-probe] FAIL: {error}", file=sys.stderr, flush=True)
        return 1
    if process is None or (process.returncode != 0 and not state.completed.is_set()):
        return process.returncode if process is not None else 1
    print(f"[origin-probe] {args.mode} {args.phase} passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
