"""Public same-origin gateway for isolated WebMCP challenge sessions."""

from __future__ import annotations

import argparse
import asyncio
import html
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from challenge.demo_fixture import seed_demo_records
from challenge.gateway_processes import (
    ChildLease,
    ChildProcessManager,
    GatewayProcessBusyError,
    GatewayProcessCapacityError,
    GatewayProcessError,
    ProcessLimits,
)
from challenge.gateway_state import (
    COOKIE_NAME,
    SESSION_TTL,
    DemoSession,
    DemoSessionRegistry,
    GatewayCapacityError,
    GatewayLimits,
    GatewayRateLimitError,
    GatewaySessionStorageError,
    GatewaySessionUnavailableError,
    SessionBinding,
)
from rcp.api import create_app

_LOG = logging.getLogger("rcp.demo.gateway")
_START_OVER_PATH = "/__rcp_demo/start-over"
_START_OVER_CSRF_COOKIE = "rcp_demo_start_over"
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_START_OVER_MAX_BYTES = 16 * 1024


@dataclass(frozen=True)
class GatewayHttpLimits:
    max_request_bytes: int = 4 * 1024 * 1024
    max_html_injection_bytes: int = 2 * 1024 * 1024
    max_concurrent_requests: int = 128
    max_concurrent_per_session: int = 8
    max_requests_per_session_minute: int = 600


@dataclass(frozen=True)
class GatewayOptions:
    secure_cookie: bool = True
    trust_forwarded_for: bool = False
    http_limits: GatewayHttpLimits = GatewayHttpLimits()


class GatewayRequestLimitError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class GatewayRoutePolicyError(RuntimeError):
    """A local-machine management route is unavailable in the public demo."""


class RequestLease:
    def __init__(self, gate: RequestGate, session_id: str) -> None:
        self._gate = gate
        self._session_id = session_id
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._gate.release(self._session_id)


class RequestGate:
    """Reject overload rather than queueing one browser behind another."""

    def __init__(self, limits: GatewayHttpLimits, *, monotonic=time.monotonic) -> None:
        self.limits = limits
        if (
            limits.max_request_bytes < 1
            or limits.max_html_injection_bytes < 1
            or limits.max_concurrent_requests < 1
            or limits.max_concurrent_per_session < 1
            or limits.max_requests_per_session_minute < 1
        ):
            raise ValueError("Gateway HTTP limits must be positive.")
        self._monotonic = monotonic
        self._total = 0
        self._by_session: dict[str, int] = defaultdict(int)
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._resetting: set[str] = set()
        self._lock = threading.Lock()

    def acquire(self, session_id: str) -> RequestLease:
        with self._lock:
            if session_id in self._resetting:
                raise GatewayRequestLimitError(
                    "This demo session is starting over. Please retry shortly.",
                    status_code=409,
                )
            if self._total >= self.limits.max_concurrent_requests:
                raise GatewayRequestLimitError(
                    "The demo is handling its maximum number of requests. Please retry shortly.",
                    status_code=503,
                    retry_after=2,
                )
            if self._by_session[session_id] >= self.limits.max_concurrent_per_session:
                raise GatewayRequestLimitError(
                    "This demo session has too many requests in progress.",
                    status_code=429,
                    retry_after=2,
                )
            now = self._monotonic()
            recent = self._recent[session_id]
            while recent and recent[0] <= now - 60:
                recent.popleft()
            if len(recent) >= self.limits.max_requests_per_session_minute:
                retry = max(1, int(60 - (now - recent[0])) + 1)
                raise GatewayRequestLimitError(
                    "This demo session sent too many requests. Please retry shortly.",
                    status_code=429,
                    retry_after=retry,
                )
            recent.append(now)
            self._total += 1
            self._by_session[session_id] += 1
            return RequestLease(self, session_id)

    def release(self, session_id: str) -> None:
        with self._lock:
            if self._total < 1 or self._by_session[session_id] < 1:
                raise RuntimeError("The gateway request lease was released more than once.")
            self._total -= 1
            self._by_session[session_id] -= 1

    def begin_reset(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._resetting or self._by_session[session_id]:
                raise GatewayProcessBusyError(
                    "Wait for current demo requests to finish before starting over."
                )
            self._resetting.add(session_id)

    def end_reset(self, session_id: str) -> None:
        with self._lock:
            self._resetting.discard(session_id)


def create_gateway(
    registry: DemoSessionRegistry,
    processes: ChildProcessManager,
    *,
    options: GatewayOptions | None = None,
) -> FastAPI:
    """Create the challenge-only cookie boundary and complete HTTP reverse proxy."""

    configured = options or GatewayOptions()
    request_gate = RequestGate(configured.http_limits)
    http = httpx.AsyncClient(follow_redirects=False, timeout=None, trust_env=False)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def reap_children() -> None:
            while True:
                await asyncio.sleep(30)
                await asyncio.to_thread(processes.reap_idle)

        reaper = asyncio.create_task(reap_children())
        try:
            yield
        finally:
            reaper.cancel()
            with suppress(asyncio.CancelledError):
                await reaper
            await http.aclose()
            await asyncio.to_thread(processes.close_all)

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def origin_isolation(request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Origin-Agent-Cluster", "?1")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.get("/__rcp_demo/health")
    async def gateway_health() -> dict[str, object]:
        return {"status": "ok", "active_children": processes.child_count()}

    @app.get(_START_OVER_PATH)
    async def start_over_confirmation(request: Request) -> Response:
        binding = await _binding(
            request,
            registry,
            processes,
            configured,
            allow_oversized=True,
        )
        csrf = secrets.token_urlsafe(32)
        page = HTMLResponse(_start_over_page(csrf))
        _set_session_cookie(page, binding.cookie_value, configured)
        page.set_cookie(
            _START_OVER_CSRF_COOKIE,
            csrf,
            max_age=10 * 60,
            httponly=True,
            secure=configured.secure_cookie,
            samesite="strict",
            path=_START_OVER_PATH,
        )
        return page

    @app.post(_START_OVER_PATH)
    async def start_over(request: Request) -> Response:
        _require_same_origin_form(request)
        form = await _read_start_over_form(request)
        csrf = form.get("csrf", "")
        cookie_csrf = request.cookies.get(_START_OVER_CSRF_COOKIE, "")
        if (
            not csrf
            or not cookie_csrf
            or not secrets.compare_digest(csrf, cookie_csrf)
            or form.get("confirm") != "start-over"
        ):
            return _error_response(request, "Start-over confirmation is invalid or expired.", 403)
        cookie = request.cookies.get(COOKIE_NAME)
        if cookie is None:
            return _error_response(request, "The current demo session is unavailable.", 409)
        session = await asyncio.to_thread(
            registry.resolve,
            cookie,
            protected_session_ids=processes.active_session_ids(),
        )
        if session is None:
            return _error_response(request, "The current demo session is unavailable.", 409)
        request_gate.begin_reset(session.session_id)
        try:
            await asyncio.to_thread(processes.stop_quiescent, session.session_id)
            replacement = await asyncio.to_thread(
                registry.rotate,
                cookie,
                client_key=_client_key(request, configured),
            )
        finally:
            request_gate.end_reset(session.session_id)
        response = RedirectResponse(url="/", status_code=303)
        _set_session_cookie(response, replacement.cookie_value, configured)
        response.delete_cookie(_START_OVER_CSRF_COOKIE, path=_START_OVER_PATH)
        return response

    @app.api_route("/{path:path}", methods=_METHODS)
    async def proxy(path: str, request: Request) -> Response:
        _require_challenge_route(request.method, f"/{path}")
        content_length = _content_length(request)
        if content_length is not None and content_length > configured.http_limits.max_request_bytes:
            raise GatewayRequestLimitError(
                "The request exceeds the demo's size limit.", status_code=413
            )
        binding = await _binding(request, registry, processes, configured)
        try:
            request_lease = request_gate.acquire(binding.session.session_id)
        except Exception:
            if binding.created:
                await asyncio.to_thread(registry.discard_fresh, binding)
            raise
        child_lease: ChildLease | None = None
        try:
            try:
                child_lease = await asyncio.to_thread(processes.acquire, binding.session)
            except GatewayProcessCapacityError:
                if binding.created:
                    await asyncio.to_thread(registry.discard_fresh, binding)
                raise
            except GatewayProcessError as exc:
                _LOG.error("Isolated RCP startup failed: %s", exc)
                response = _error_response(
                    request,
                    "The isolated RCP session could not start.",
                    502,
                    retry_after=2,
                )
                _set_session_cookie(response, binding.cookie_value, configured)
                request_lease.release()
                return response
            try:
                upstream = await _send_upstream(
                    http,
                    child_lease,
                    path=path,
                    request=request,
                    max_request_bytes=configured.http_limits.max_request_bytes,
                )
            except GatewayProcessError as exc:
                _LOG.error("Isolated RCP request failed: %s", exc)
                response = _error_response(
                    request,
                    "The isolated RCP session could not answer this request.",
                    502,
                    retry_after=2,
                )
                _set_session_cookie(response, binding.cookie_value, configured)
                child_lease.release()
                request_lease.release()
                return response
            if _should_inject_control(request, upstream, configured.http_limits):
                response = await _injected_html_response(upstream)
                await asyncio.to_thread(registry.refresh_usage, binding.session.session_id)
                child_lease.release()
                request_lease.release()
            else:
                response = _streaming_proxy_response(
                    upstream,
                    child_lease,
                    request_lease,
                    registry,
                    binding.session,
                )
            _set_session_cookie(response, binding.cookie_value, configured)
            return response
        except GatewayRequestLimitError:
            if binding.created:
                await asyncio.to_thread(registry.discard_fresh, binding)
            if child_lease is not None:
                child_lease.release()
            request_lease.release()
            raise
        except BaseException:
            if child_lease is not None:
                child_lease.release()
            request_lease.release()
            raise

    @app.exception_handler(GatewayCapacityError)
    async def session_capacity_error(request: Request, exc: GatewayCapacityError) -> Response:
        return _error_response(request, str(exc), 503, retry_after=30)

    @app.exception_handler(GatewaySessionStorageError)
    async def session_storage_error(request: Request, exc: GatewaySessionStorageError) -> Response:
        return _error_response(request, str(exc), 503, start_over=True)

    @app.exception_handler(GatewayRateLimitError)
    async def session_rate_error(request: Request, exc: GatewayRateLimitError) -> Response:
        return _error_response(request, str(exc), 429, retry_after=exc.retry_after_seconds)

    @app.exception_handler(GatewaySessionUnavailableError)
    async def unavailable_error(request: Request, exc: GatewaySessionUnavailableError) -> Response:
        return _error_response(request, str(exc), 503)

    @app.exception_handler(GatewayProcessCapacityError)
    async def process_capacity_error(
        request: Request, exc: GatewayProcessCapacityError
    ) -> Response:
        return _error_response(request, str(exc), 503, retry_after=5)

    @app.exception_handler(GatewayProcessBusyError)
    async def process_busy_error(request: Request, exc: GatewayProcessBusyError) -> Response:
        return _error_response(request, str(exc), 409, retry_after=2)

    @app.exception_handler(GatewayProcessError)
    async def process_error(request: Request, exc: GatewayProcessError) -> Response:
        _LOG.error("Isolated RCP child failed: %s", exc)
        return _error_response(request, "The isolated RCP session could not start.", 502)

    @app.exception_handler(GatewayRequestLimitError)
    async def request_limit_error(request: Request, exc: GatewayRequestLimitError) -> Response:
        return _error_response(
            request,
            str(exc),
            exc.status_code,
            retry_after=exc.retry_after,
        )

    @app.exception_handler(GatewayRoutePolicyError)
    async def route_policy_error(request: Request, exc: GatewayRoutePolicyError) -> Response:
        return _error_response(request, str(exc), 403)

    return app


async def _binding(
    request: Request,
    registry: DemoSessionRegistry,
    processes: ChildProcessManager,
    options: GatewayOptions,
    *,
    allow_oversized: bool = False,
) -> SessionBinding:
    return await asyncio.to_thread(
        registry.get_or_create,
        request.cookies.get(COOKIE_NAME),
        client_key=_client_key(request, options),
        protected_session_ids=processes.active_session_ids(),
        allow_oversized=allow_oversized,
    )


async def _send_upstream(
    http: httpx.AsyncClient,
    lease: ChildLease,
    *,
    path: str,
    request: Request,
    max_request_bytes: int,
) -> httpx.Response:
    query = request.url.query
    suffix = f"?{query}" if query else ""
    url = f"{lease.child.base_url}/{path}{suffix}"
    upstream_request = http.build_request(
        request.method,
        url,
        headers=_upstream_request_headers(request),
        content=_bounded_request_stream(request, max_request_bytes),
    )
    try:
        return await http.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        raise GatewayProcessError(f"The isolated RCP request failed: {exc}") from exc


async def _bounded_request_stream(request: Request, limit: int) -> AsyncIterator[bytes]:
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise GatewayRequestLimitError(
                "The request exceeds the demo's size limit.", status_code=413
            )
        yield chunk


def _streaming_proxy_response(
    upstream: httpx.Response,
    child_lease: ChildLease,
    request_lease: RequestLease,
    registry: DemoSessionRegistry,
    session: DemoSession,
) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            child_lease.release()
            request_lease.release()
            try:
                await asyncio.to_thread(registry.refresh_usage, session.session_id)
            except (GatewayCapacityError, GatewaySessionUnavailableError):
                _LOG.warning("Could not refresh usage for demo session %s", session.session_id)

    response = StreamingResponse(body(), status_code=upstream.status_code)
    response.raw_headers = _upstream_response_headers(upstream)
    return response


async def _injected_html_response(upstream: httpx.Response) -> Response:
    try:
        content = await upstream.aread()
    finally:
        await upstream.aclose()
    marker = b"</body>"
    insertion = _START_OVER_CONTROL.encode("utf-8")
    if marker in content:
        content = content.replace(marker, insertion + marker, 1)
    response = Response(content=content, status_code=upstream.status_code)
    response.raw_headers = [
        header
        for header in _upstream_response_headers(upstream)
        if header[0].lower() != b"content-length"
    ]
    response.headers["content-length"] = str(len(content))
    return response


def _should_inject_control(
    request: Request,
    upstream: httpx.Response,
    limits: GatewayHttpLimits,
) -> bool:
    if (
        request.method != "GET"
        or request.url.path != "/"
        or "text/html" not in upstream.headers.get("content-type", "")
    ):
        return False
    if upstream.headers.get("content-encoding"):
        return False
    raw_length = upstream.headers.get("content-length")
    return raw_length is not None and int(raw_length) <= limits.max_html_injection_bytes


def _upstream_request_headers(request: Request) -> dict[str, str]:
    connection_tokens = {
        token.strip().lower()
        for token in request.headers.get("connection", "").split(",")
        if token.strip()
    }
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _HOP_BY_HOP | connection_tokens | {"host", "cookie"}
    }
    cookie = SimpleCookie()
    try:
        cookie.load(request.headers.get("cookie", ""))
    except CookieError:
        cookie.clear()
    forwarded = [
        f"{name}={morsel.value}"
        for name, morsel in cookie.items()
        if name not in {COOKIE_NAME, _START_OVER_CSRF_COOKIE}
    ]
    if forwarded:
        headers["cookie"] = "; ".join(forwarded)
    return headers


def _upstream_response_headers(upstream: httpx.Response) -> list[tuple[bytes, bytes]]:
    connection_tokens = {
        token.strip().lower()
        for token in upstream.headers.get("connection", "").split(",")
        if token.strip()
    }
    upstream_origin = str(upstream.request.url.copy_with(path="/", query=None)).rstrip("/")
    result: list[tuple[bytes, bytes]] = []
    for name, value in upstream.headers.multi_items():
        lowered = name.lower()
        if lowered in _HOP_BY_HOP | connection_tokens:
            continue
        if lowered == "location" and value.startswith(upstream_origin):
            value = value[len(upstream_origin) :] or "/"
        result.append((name.encode("latin-1"), value.encode("latin-1")))
    return result


def _set_session_cookie(response: Response, value: str, options: GatewayOptions) -> None:
    response.set_cookie(
        COOKIE_NAME,
        value,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        secure=options.secure_cookie,
        samesite="strict",
        path="/",
    )


def _client_key(request: Request, options: GatewayOptions) -> str:
    if options.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return forwarded
    if request.client is None:
        return "unknown-client"
    return request.client.host


def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise GatewayRequestLimitError("Invalid Content-Length header.", status_code=400) from exc
    if value < 0:
        raise GatewayRequestLimitError("Invalid Content-Length header.", status_code=400)
    return value


def _require_challenge_route(method: str, path: str) -> None:
    blocked_prefixes = (
        "/api/project-setup",
        "/api/project-provisioning",
        "/api/project-transfers",
        "/api/native/project-transfers",
    )
    blocked = any(path == prefix or path.startswith(f"{prefix}/") for prefix in blocked_prefixes)
    blocked = blocked or (path == "/api/projects" and method not in {"GET", "HEAD"})
    blocked = blocked or bool(method == "DELETE" and re.fullmatch(r"/api/projects/[^/]+", path))
    blocked = blocked or bool(method == "POST" and re.fullmatch(r"/api/projects/[^/]+/leave", path))
    blocked = blocked or bool(
        method == "PUT" and re.fullmatch(r"/api/projects/[^/]+/settings", path)
    )
    blocked = blocked or bool(
        method == "POST"
        and re.fullmatch(
            r"/api/projects/[^/]+/machines/[^/]+/providers/[^/]+/resolve",
            path,
        )
    )
    if blocked:
        raise GatewayRoutePolicyError(
            "Project setup and machine configuration are unavailable in this synthetic demo."
        )


def _require_same_origin_form(request: Request) -> None:
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site not in {None, "same-origin"}:
        raise GatewayRequestLimitError("Start over requires a same-origin form.", status_code=403)
    origin = request.headers.get("origin")
    if origin is None:
        raise GatewayRequestLimitError("Start over requires an Origin header.", status_code=403)
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != request.headers.get("host"):
        raise GatewayRequestLimitError("Start over requires the current origin.", status_code=403)


async def _read_start_over_form(request: Request) -> dict[str, str]:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/x-www-form-urlencoded":
        raise GatewayRequestLimitError(
            "Start-over confirmation must be a form.",
            status_code=415,
        )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _START_OVER_MAX_BYTES:
            raise GatewayRequestLimitError(
                "Start-over confirmation was too large.",
                status_code=413,
            )
    try:
        parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True, strict_parsing=True)
    except (UnicodeDecodeError, ValueError) as exc:
        raise GatewayRequestLimitError(
            "Start-over confirmation is invalid or expired.",
            status_code=403,
        ) from exc
    if any(len(values) != 1 for values in parsed.values()):
        raise GatewayRequestLimitError(
            "Start-over confirmation is invalid or expired.",
            status_code=403,
        )
    return {name: values[0] for name, values in parsed.items()}


def _error_response(
    request: Request,
    message: str,
    status_code: int,
    *,
    retry_after: int | None = None,
    start_over: bool = False,
) -> Response:
    headers = {"Cache-Control": "no-store"}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    if request.url.path.startswith("/api/") or "application/json" in request.headers.get(
        "accept", ""
    ):
        return JSONResponse({"detail": message}, status_code=status_code, headers=headers)
    return HTMLResponse(
        _error_page(message, start_over=start_over),
        status_code=status_code,
        headers=headers,
    )


def _start_over_page(csrf: str) -> str:
    escaped = html.escape(csrf, quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Start over · RCP Demo</title>{_GATEWAY_STYLE}</head>
<body><main><p class="eyebrow">Research Control Panel · WebMCP Challenge Demo</p>
<h1>Start this demo over?</h1>
<p>This permanently deletes only this browser's synthetic project progress. Other visitors are
not affected.</p>
<form method="post" action="{_START_OVER_PATH}">
<input type="hidden" name="csrf" value="{escaped}">
<button type="submit" name="confirm" value="start-over">Delete my progress and start over</button>
<a href="/">Keep my current progress</a></form></main></body></html>"""


def _error_page(message: str, *, start_over: bool = False) -> str:
    action = (
        f'<a href="{_START_OVER_PATH}">Start over demo</a>'
        if start_over
        else '<a href="/">Retry</a>'
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>RCP Demo unavailable</title>
{_GATEWAY_STYLE}</head><body><main><p class="eyebrow">Research Control Panel</p>
<h1>Demo temporarily unavailable</h1><p>{html.escape(message)}</p>
{action}</main></body></html>"""


def _prepare_demo_session(session: DemoSession) -> None:
    manifest = session.project_root / "state-repo" / ".research" / "manifest.toml"
    app = create_app(str(manifest), data_dir=session.data_root)
    project_id = app.state.default_project_id
    if project_id is None:
        raise GatewayProcessError("The copied demo project did not register.")
    seed_demo_records(
        app.state.background_tasks.store,
        project_id,
        session.stage_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="rcp-demo-gateway")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("RCP_DEMO_ROOT", "/var/data/rcp-demo")),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "demo-project",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8420")))
    parser.add_argument("--insecure-cookie", action="store_true")
    parser.add_argument("--trust-forwarded-for", action="store_true")
    args = parser.parse_args()
    registry = DemoSessionRegistry(
        args.root,
        args.fixture,
        limits=GatewayLimits(
            max_sessions=_positive_env_int("RCP_DEMO_MAX_SESSIONS", 500),
            max_total_bytes=_positive_env_int("RCP_DEMO_MAX_TOTAL_BYTES", 8 * 1024 * 1024 * 1024),
            max_session_bytes=_positive_env_int("RCP_DEMO_MAX_SESSION_BYTES", 32 * 1024 * 1024),
            max_creations_per_client=_positive_env_int("RCP_DEMO_MAX_CREATIONS_PER_CLIENT", 20),
        ),
    )
    processes = ChildProcessManager(
        limits=ProcessLimits(
            max_processes=_positive_env_int("RCP_DEMO_MAX_PROCESSES", 12),
            idle_seconds=_positive_env_int("RCP_DEMO_IDLE_SECONDS", 5 * 60),
            startup_seconds=_positive_env_int("RCP_DEMO_STARTUP_SECONDS", 20),
            shutdown_seconds=_positive_env_int("RCP_DEMO_CHILD_SHUTDOWN_SECONDS", 10),
        ),
        prepare_session=_prepare_demo_session,
    )
    app = create_gateway(
        registry,
        processes,
        options=GatewayOptions(
            secure_cookie=not args.insecure_cookie,
            trust_forwarded_for=args.trust_forwarded_for,
            http_limits=GatewayHttpLimits(
                max_request_bytes=_positive_env_int("RCP_DEMO_MAX_REQUEST_BYTES", 4 * 1024 * 1024),
                max_html_injection_bytes=_positive_env_int(
                    "RCP_DEMO_MAX_HTML_BYTES", 2 * 1024 * 1024
                ),
                max_concurrent_requests=_positive_env_int("RCP_DEMO_MAX_CONCURRENT_REQUESTS", 128),
                max_concurrent_per_session=_positive_env_int(
                    "RCP_DEMO_MAX_CONCURRENT_PER_SESSION", 8
                ),
                max_requests_per_session_minute=_positive_env_int(
                    "RCP_DEMO_MAX_REQUESTS_PER_SESSION_MINUTE", 600
                ),
            ),
        ),
    )
    uvicorn.run(app, host=args.host, port=args.port)


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a positive integer.") from exc
    if value < 1:
        raise SystemExit(f"{name} must be a positive integer.")
    return value


_START_OVER_CONTROL = f"""
<a href="{_START_OVER_PATH}" aria-label="Start the synthetic demo over"
style="position:fixed;right:18px;bottom:18px;z-index:2147483647;padding:9px 13px;border:1px solid
#8f8170;border-radius:9px;background:#fffaf2;color:#342e27;font:700 13px/1.2 system-ui,sans-serif;
text-decoration:none;box-shadow:0 4px 18px #332a1c22">Start over demo</a>
"""

_GATEWAY_STYLE = """<style>
body{margin:0;background:#f7f2e9;color:#2d2924;font:17px/1.55 system-ui,sans-serif}
main{max-width:650px;margin:12vh auto;padding:42px;border:1px solid #cbbda9;background:#fffdf8}
h1{font:700 42px/1.05 Georgia,serif;margin:8px 0 18px}.eyebrow{color:#913f33;font-size:12px;
font-weight:800;letter-spacing:.12em;text-transform:uppercase}form{display:flex;gap:18px;align-items:center;
margin-top:28px}button,a{font:700 15px system-ui,sans-serif}button{padding:12px 16px;border:0;
border-radius:7px;background:#913f33;color:white;cursor:pointer}a{color:#5c5144}
</style>"""


if __name__ == "__main__":
    main()
