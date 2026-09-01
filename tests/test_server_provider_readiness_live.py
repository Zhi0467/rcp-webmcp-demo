"""Explicitly gated read-only provider check through one reachable SSH route."""

from __future__ import annotations

import os

import pytest

from rcp.agents.launcher import AgentLauncher

_LIVE_GATE = "RCP_RUN_PROVIDER_READINESS_LIVE"
_HOST = "RCP_PROVIDER_READINESS_SSH_HOST"
_ACCOUNT = "RCP_PROVIDER_READINESS_SSH_ACCOUNT"
_PROVIDER = "RCP_PROVIDER_READINESS_SSH_PROVIDER"
_BINARY = "RCP_PROVIDER_READINESS_SSH_BINARY"
_LOCAL_UNAUTH_PROVIDER = "RCP_PROVIDER_READINESS_LOCAL_UNAUTH_PROVIDER"
_LOCAL_UNAUTH_BINARY = "RCP_PROVIDER_READINESS_LOCAL_UNAUTH_BINARY"

pytestmark = pytest.mark.skipif(
    os.environ.get(_LIVE_GATE) != "1",
    reason="read-only remote provider qualification is disabled",
)


def test_reachable_ssh_provider_is_ready_on_the_exact_account() -> None:
    host = _required(_HOST)
    expected_account = _required(_ACCOUNT)
    provider = _required(_PROVIDER)
    binary = _required(_BINARY)
    launcher = AgentLauncher()

    account = launcher.execution_account(host=host)

    assert account.reachable is True, account.reason
    assert account.os_account == expected_account

    readiness = launcher.readiness(
        provider,
        host=host,
        binary=binary,
        refresh=True,
    )

    assert readiness.installed is True, readiness.reason
    assert readiness.authenticated is True, readiness.reason
    assert readiness.binary_path == binary
    assert readiness.path_state == "resolved"
    assert readiness.version


def test_existing_local_unauthenticated_provider_is_reported_without_login() -> None:
    provider = _required(_LOCAL_UNAUTH_PROVIDER)
    binary = _required(_LOCAL_UNAUTH_BINARY)
    launcher = AgentLauncher()
    account = launcher.execution_account()

    assert account.reachable is True
    assert account.os_account

    readiness = launcher.readiness(
        provider,
        binary=binary,
        refresh=True,
    )

    assert readiness.installed is True, readiness.reason
    assert readiness.authenticated is False
    assert readiness.binary_path == binary
    assert readiness.path_state == "resolved"
    assert readiness.version


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        pytest.fail(f"{name} must contain one nonempty safe value")
    return value
