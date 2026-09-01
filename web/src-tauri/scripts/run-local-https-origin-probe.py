#!/usr/bin/env python3
"""Drive two isolated local HTTPS origins through a real Tauri WKWebView.

Q11 experiment. The D2 probe proved a `Secure` session cookie is lost on plain
HTTP loopback origins. This runner repeats the same drive over HTTPS using a
certificate the probe generates for itself, pinned through an app-scoped trust
hook. Nothing is written to a system trust store or keychain.

A third origin presents a different, unpinned certificate while remaining inside
the WebView's navigation allowlist, so refusing it proves the pin rather than
the allowlist.
"""

from __future__ import annotations

import argparse
import html
import http.cookies
import http.server
import ipaddress
import os
import socket
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

COOKIE_NAME = "__Host-rcp_session"
# RCP sets a Max-Age on the real team session cookie, so the probe must too.
# Without it the cookie is a session cookie and cannot survive a restart, which
# would make the resume phase measure the wrong thing.
COOKIE_MAX_AGE_SECONDS = 14 * 24 * 60 * 60
PORT_A = 39131
PORT_B = 39132
PORT_OTHER = 39133
UUID_A = "11111111111141118111111111111111"
UUID_B = "22222222222242228222222222222222"
UUID_OTHER = "99999999999949998999999999999999"
PROBE_TIMEOUT_SECONDS = 180
CERT_DAYS = 30


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
    handshakes: dict[str, int] = field(default_factory=dict)

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
            f"[https-probe] {label} {path} cookie={cookie or '<none>'} expected={expectation}",
            flush=True,
        )

    def record_handshake(self, label: str) -> None:
        with self.lock:
            self.handshakes[label] = self.handshakes.get(label, 0) + 1


@dataclass(frozen=True)
class Origin:
    label: str
    host: str
    bind_hosts: tuple[str, ...]
    port: int

    @property
    def url(self) -> str:
        return f"https://{self.host}:{self.port}"

    @property
    def cookie(self) -> str:
        return f"probe-{self.label}"


def _generate_certificate(directory: Path, name: str, hosts: list[str]) -> tuple[Path, Path, str]:
    """Create one self-signed P-256 leaf certificate and return its DER SHA-256."""

    key_path = directory / f"{name}-key.pem"
    cert_path = directory / f"{name}-cert.pem"
    if key_path.exists() and cert_path.exists():
        der = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-outform", "DER"],
            check=True,
            capture_output=True,
        ).stdout
        import hashlib

        return key_path, cert_path, hashlib.sha256(der).hexdigest()
    san = ",".join(f"DNS:{host}" for host in hosts)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-nodes",
            "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256",
            "-sha256", "-days", str(CERT_DAYS),
            "-keyout", str(key_path), "-out", str(cert_path),
            "-subj", f"/CN=RCP local HTTPS probe ({name})",
            "-addext", f"subjectAltName={san}",
            "-addext", "basicConstraints=critical,CA:FALSE",
            "-addext", "keyUsage=critical,digitalSignature,keyEncipherment",
            "-addext", "extendedKeyUsage=serverAuth",
        ],
        check=True,
        capture_output=True,
    )
    der = subprocess.run(
        ["openssl", "x509", "-in", str(cert_path), "-outform", "DER"],
        check=True,
        capture_output=True,
    ).stdout
    import hashlib

    return key_path, cert_path, hashlib.sha256(der).hexdigest()


class ProbeServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        origin: Origin,
        origins: dict[str, Origin],
        state: ProbeState,
        bind_host: str,
        context: ssl.SSLContext,
    ) -> None:
        self.origin = origin
        self.origins = origins
        self.probe_state = state
        if ":" in bind_host:
            self.address_family = socket.AF_INET6
        super().__init__((bind_host, origin.port), ProbeHandler, bind_and_activate=False)
        self.socket = context.wrap_socket(self.socket, server_side=True)
        try:
            self.server_bind()
            self.server_activate()
        except OSError:
            self.server_close()
            raise

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()

    def handle_error(self, request: object, client_address: object) -> None:
        # A refused pin surfaces here as a TLS handshake error. That is the
        # expected result for the unpinned origin, so it must not be noise.
        error = sys.exc_info()[1]
        if isinstance(error, ssl.SSLError):
            print(
                f"[https-probe] {self.origin.label} TLS handshake refused: {error.reason}",
                flush=True,
            )
            return
        super().handle_error(request, client_address)


class ProbeHandler(http.server.BaseHTTPRequestHandler):
    server: ProbeServer

    def handle_one_request(self) -> None:
        self.server.probe_state.record_handshake(self.server.origin.label)
        super().handle_one_request()

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
            self.server.origin.label, path, cookie, expected, allowed=allowed
        )
        if not allowed or cookie != expected:
            self.send_error(409)
            return
        if path == "/login":
            self._redirect(
                f"{self.server.origin.url}/check-after-login",
                cookie=f"{COOKIE_NAME}={self.server.origin.cookie}; "
                f"Path=/; Max-Age={COOKIE_MAX_AGE_SECONDS}; HttpOnly; Secure; SameSite=Lax",
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
<title>RCP local HTTPS probe — {html.escape(origin.label.upper())}</title>
<style>
body {{ font: 20px system-ui; max-width: 760px; margin: 64px auto; line-height: 1.5; }}
.pass {{ color: #176b3a; }} .fail {{ color: #a3261e; }}
code {{ font-size: 16px; }}
</style>
<h1>RCP local HTTPS origin probe</h1>
<p class="pass"><strong>The A &rarr; B &rarr; A secure-cookie checks passed.</strong></p>
<p id="http-only">Checking whether JavaScript can see the HttpOnly cookie&hellip;</p>
<p>Current host: <code>{html.escape(origin.host)}</code></p>
<p>The probe will now attempt the unpinned certificate and close itself.</p>
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


def _origins() -> dict[str, Origin]:
    return {
        "a": Origin("a", f"rcp-{UUID_A}.localhost", ("127.0.0.1", "::1"), PORT_A),
        "b": Origin("b", f"rcp-{UUID_B}.localhost", ("127.0.0.1", "::1"), PORT_B),
        "other": Origin("other", f"rcp-{UUID_OTHER}.localhost", ("127.0.0.1", "::1"), PORT_OTHER),
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
        raise RuntimeError("WKWebView served a request from the unpinned certificate origin")
    if state.handshakes.get("other"):
        raise RuntimeError("the unpinned certificate completed a TLS handshake")


def _cargo() -> str:
    """Resolve cargo from PATH, then from the default rustup toolchain."""

    found = shutil.which("cargo")
    if found:
        return found
    toolchains = Path.home() / ".rustup" / "toolchains"
    for candidate in sorted(toolchains.glob("*/bin/cargo")):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("cargo is not installed; the probe needs the Rust toolchain")


def _stop_probe(process: subprocess.Popen[bytes]) -> int:
    process.terminate()
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("login", "resume"), required=True)
    parser.add_argument(
        "--cert-dir",
        help=(
            "reuse certificates in this directory across runs; without it each run "
            "generates fresh ones, which cannot distinguish a changed certificate "
            "from a cookie the store never kept"
        ),
    )
    args = parser.parse_args()

    origins = _origins()
    _prove_resolution(origins)

    import contextlib

    if args.cert_dir:
        chosen = Path(args.cert_dir).expanduser().resolve()
        chosen.mkdir(parents=True, exist_ok=True)
        holder = contextlib.nullcontext(str(chosen))
    else:
        holder = tempfile.TemporaryDirectory(prefix="rcp-https-probe-")

    with holder as raw_directory:
        directory = Path(raw_directory)
        _, pinned_cert, fingerprint = _generate_certificate(
            directory, "pinned", [origins["a"].host, origins["b"].host]
        )
        _, unpinned_cert, unpinned_fingerprint = _generate_certificate(
            directory, "unpinned", [origins["other"].host]
        )
        if fingerprint == unpinned_fingerprint:
            raise RuntimeError("the two probe certificates must differ")
        print(f"[https-probe] pinned sha256={fingerprint}", flush=True)
        print(f"[https-probe] unpinned sha256={unpinned_fingerprint}", flush=True)

        def context_for(cert: Path) -> ssl.SSLContext:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(
                certfile=str(cert), keyfile=str(cert.with_name(cert.name.replace("cert", "key")))
            )
            return context

        state = ProbeState()
        servers: list[ProbeServer] = []
        try:
            for origin in origins.values():
                cert = unpinned_cert if origin.label == "other" else pinned_cert
                for bind_host in origin.bind_hosts:
                    servers.append(
                        ProbeServer(origin, origins, state, bind_host, context_for(cert))
                    )
        except OSError as error:
            for server in servers:
                server.server_close()
            print(f"[https-probe] FAIL: could not bind: {error}", file=sys.stderr, flush=True)
            return 1

        threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
        for thread in threads:
            thread.start()

        crate = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["RCP_HTTPS_PROBE_FIRST"] = origins["a"].url
        environment["RCP_HTTPS_PROBE_SECOND"] = origins["b"].url
        environment["RCP_HTTPS_PROBE_UNPINNED"] = origins["other"].url
        environment["RCP_HTTPS_PROBE_FINGERPRINT"] = fingerprint
        cargo = _cargo()
        # A rustup toolchain resolved off PATH still needs its own rustc.
        environment["PATH"] = os.pathsep.join(
            [str(Path(cargo).parent), environment.get("PATH", "")]
        )
        if args.phase == "login":
            environment["RCP_HTTPS_PROBE_LOGIN"] = "1"
        else:
            environment.pop("RCP_HTTPS_PROBE_LOGIN", None)

        # Prove the probe's own TLS stack answers before blaming the WebView.
        preflight = subprocess.run(
            [
                "curl", "--silent", "--output", "/dev/null", "--write-out", "%{http_code}",
                "--cacert", str(pinned_cert), f"{origins['a'].url}/favicon.ico",
            ],
            capture_output=True,
            text=True,
        )
        if preflight.stdout.strip() != "204":
            print(
                "[https-probe] FAIL: the probe's own HTTPS origin did not answer curl "
                f"(got {preflight.stdout.strip() or preflight.stderr.strip()!r})",
                file=sys.stderr,
                flush=True,
            )
            for server in servers:
                server.shutdown()
                server.server_close()
            return 1
        # The whole result is void if the host already trusts this certificate,
        # because then nothing proves the trust was app-scoped.
        system_trust = subprocess.run(
            [
                "curl", "--silent", "--output", "/dev/null", "--write-out", "%{http_code}",
                f"{origins['a'].url}/favicon.ico",
            ],
            capture_output=True,
            text=True,
        )
        if system_trust.returncode == 0:
            print(
                "[https-probe] FAIL: the system already trusts the probe certificate, so "
                "an accepted connection would not prove app-scoped trust",
                file=sys.stderr,
                flush=True,
            )
            for server in servers:
                server.shutdown()
                server.server_close()
            return 1
        print(
            "[https-probe] preflight: the system does NOT trust the probe certificate "
            f"(curl exit {system_trust.returncode})",
            flush=True,
        )
        print("[https-probe] preflight: the pinned HTTPS origin answers curl", flush=True)
        # The preflight is not part of the WebView drive.
        with state.lock:
            state.handshakes.clear()

        print(f"[https-probe] {args.phase}: running automatic WKWebView drive", flush=True)
        process: subprocess.Popen[bytes] | None = None
        interrupted = False
        timed_out = False
        try:
            process = subprocess.Popen(
                [
                    cargo, "run",
                    "--manifest-path", str(crate / "Cargo.toml"),
                    "--features", "https-trust-probe",
                    "--example", "local_https_origin_probe",
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
        print("[https-probe] interrupted", file=sys.stderr, flush=True)
        return 130
    if timed_out:
        reached = sorted(state.handshakes)
        print(
            f"[https-probe] FAIL: no terminal result within {PROBE_TIMEOUT_SECONDS}s; "
            f"origins that completed a TLS handshake: {reached or '<none>'}",
            file=sys.stderr,
            flush=True,
        )
        if not reached:
            print(
                "[https-probe] no handshake at all means WKWebView refused the connection "
                "before the trust hook could answer.",
                file=sys.stderr,
                flush=True,
            )
        return 1
    try:
        _assert_phase(args.phase, state)
    except RuntimeError as error:
        print(f"[https-probe] FAIL: {error}", file=sys.stderr, flush=True)
        return 1
    if process is None or (process.returncode != 0 and not state.completed.is_set()):
        return process.returncode if process is not None else 1
    print(f"[https-probe] local HTTPS {args.phase} passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
