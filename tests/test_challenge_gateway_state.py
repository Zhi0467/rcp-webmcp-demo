from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from challenge.gateway_state import (
    SESSION_TTL,
    DemoSessionRegistry,
    GatewayCapacityError,
    GatewayLimits,
    GatewayRateLimitError,
    GatewaySessionStorageError,
    GatewaySessionUnavailableError,
    _tree_size,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 31, 18, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "manifest.toml").write_text("[project]\nname = 'Synthetic'\n", encoding="utf-8")
    (root / "state").mkdir()
    (root / "state" / "graph.json").write_text("{}\n", encoding="utf-8")
    return root


def test_cookie_mapping_persists_and_renews_without_sharing(
    fixture_root: Path, tmp_path: Path
) -> None:
    clock = Clock()
    state_root = tmp_path / "gateway"
    registry = DemoSessionRegistry(state_root, fixture_root, clock=clock)

    first = registry.get_or_create(None, client_key="client-a")
    second = registry.get_or_create(None, client_key="client-b")

    assert first.created is True
    assert len(first.cookie_value) == 43
    assert first.session.session_id != second.session.session_id
    assert first.session.root != second.session.root
    assert (first.session.project_root / "manifest.toml").is_file()
    marker = first.session.data_root / "progress.txt"
    marker.write_text("kept", encoding="utf-8")
    clock.advance(timedelta(days=2))
    restarted = DemoSessionRegistry(state_root, fixture_root, clock=clock)
    resumed = restarted.get_or_create(first.cookie_value, client_key="client-a")

    assert resumed.created is False
    assert resumed.session.session_id == first.session.session_id
    assert marker.read_text(encoding="utf-8") == "kept"
    assert resumed.session.expires_at == clock.value + SESSION_TTL
    with sqlite3.connect(restarted.database_path) as connection:
        stored = connection.execute("SELECT token_hash FROM sessions").fetchall()
    assert first.cookie_value not in {row[0] for row in stored}


def test_expiry_preserves_protected_copy_then_deletes_only_that_uuid(
    fixture_root: Path,
    tmp_path: Path,
) -> None:
    clock = Clock()
    registry = DemoSessionRegistry(tmp_path / "gateway", fixture_root, clock=clock)
    expired = registry.get_or_create(None, client_key="client-a")
    survivor = registry.get_or_create(None, client_key="client-b")
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched", encoding="utf-8")
    clock.advance(SESSION_TTL + timedelta(seconds=1))

    assert registry.delete_expired(protected_session_ids=[survivor.session.session_id]) == [
        expired.session.session_id
    ]
    assert not expired.session.root.exists()
    assert survivor.session.root.is_dir()
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_start_over_rotates_only_current_copy_even_at_capacity(
    fixture_root: Path,
    tmp_path: Path,
) -> None:
    limits = GatewayLimits(max_sessions=2)
    registry = DemoSessionRegistry(tmp_path / "gateway", fixture_root, limits=limits)
    current = registry.get_or_create(None, client_key="client-a")
    other = registry.get_or_create(None, client_key="client-b")
    (current.session.data_root / "progress.txt").write_text("old", encoding="utf-8")

    with pytest.raises(GatewayCapacityError, match="at capacity"):
        registry.get_or_create(None, client_key="client-c")
    replacement = registry.rotate(current.cookie_value, client_key="client-a")

    assert replacement.cookie_value != current.cookie_value
    assert replacement.session.session_id != current.session.session_id
    assert not current.session.root.exists()
    assert replacement.session.root.is_dir()
    assert not (replacement.session.data_root / "progress.txt").exists()
    assert registry.resolve(current.cookie_value) is None
    assert registry.resolve(other.cookie_value) is not None


def test_creation_rate_limit_is_persistent_and_client_scoped(
    fixture_root: Path,
    tmp_path: Path,
) -> None:
    clock = Clock()
    limits = GatewayLimits(
        max_creations_per_client=2,
        creation_window=timedelta(minutes=10),
    )
    state_root = tmp_path / "gateway"
    registry = DemoSessionRegistry(state_root, fixture_root, limits=limits, clock=clock)
    registry.get_or_create(None, client_key="client-a")
    registry.get_or_create(None, client_key="client-a")

    restarted = DemoSessionRegistry(state_root, fixture_root, limits=limits, clock=clock)
    with pytest.raises(GatewayRateLimitError) as error:
        restarted.get_or_create(None, client_key="client-a")
    assert error.value.retry_after_seconds == 601
    assert restarted.get_or_create(None, client_key="client-b").created is True
    clock.advance(timedelta(minutes=10, seconds=1))
    assert restarted.get_or_create(None, client_key="client-a").created is True


def test_storage_caps_fail_closed_without_leaving_a_copy(
    fixture_root: Path, tmp_path: Path
) -> None:
    limits = GatewayLimits(max_session_bytes=1)
    state_root = tmp_path / "gateway"
    registry = DemoSessionRegistry(state_root, fixture_root, limits=limits)

    with pytest.raises(GatewayCapacityError, match="fixture exceeds"):
        registry.get_or_create(None, client_key="client-a")

    assert list(registry.sessions_root.iterdir()) == []
    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_current_usage_is_measured_and_persisted(fixture_root: Path, tmp_path: Path) -> None:
    registry = DemoSessionRegistry(tmp_path / "gateway", fixture_root)
    binding = registry.get_or_create(None, client_key="client-a")
    (binding.session.data_root / "result.bin").write_bytes(b"new-result")

    measured = registry.refresh_usage(binding.session.session_id)

    assert measured == binding.session.size_bytes + len(b"new-result")
    with sqlite3.connect(registry.database_path) as connection:
        stored = connection.execute(
            "SELECT size_bytes FROM sessions WHERE session_id = ?",
            (binding.session.session_id,),
        ).fetchone()
    assert stored == (measured,)


def test_oversized_usage_is_persisted_and_blocks_resume_but_not_recovery(
    fixture_root: Path, tmp_path: Path
) -> None:
    fixture_size = _tree_size(fixture_root)
    limits = GatewayLimits(max_session_bytes=fixture_size + 8)
    registry = DemoSessionRegistry(tmp_path / "gateway", fixture_root, limits=limits)
    binding = registry.get_or_create(None, client_key="client-a")
    (binding.session.data_root / "oversized.bin").write_bytes(b"x" * 16)

    with pytest.raises(GatewaySessionStorageError, match="Start over"):
        registry.refresh_usage(binding.session.session_id)
    with pytest.raises(GatewaySessionStorageError, match="Start over"):
        registry.get_or_create(binding.cookie_value, client_key="client-a")

    recoverable = registry.get_or_create(
        binding.cookie_value,
        client_key="client-a",
        allow_oversized=True,
    )
    assert recoverable.created is False
    assert recoverable.session.size_bytes == fixture_size + 16
    assert registry.resolve(binding.cookie_value) is not None


def test_tree_size_tolerates_a_live_file_disappearing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "live-session"
    root.mkdir()
    stable = root / "stable.db"
    stable.write_bytes(b"stable")
    transient = root / "stable.db-wal"
    transient.write_bytes(b"transient")
    original_is_file = Path.is_file

    def remove_transient_before_stat(path: Path) -> bool:
        if path == transient and path.exists():
            path.unlink()
            return True
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", remove_transient_before_stat)

    assert _tree_size(root) == len(b"stable")


def test_fresh_copy_can_be_rolled_back_after_runtime_capacity_refusal(
    fixture_root: Path, tmp_path: Path
) -> None:
    registry = DemoSessionRegistry(tmp_path / "gateway", fixture_root)
    binding = registry.get_or_create(None, client_key="client-a")

    registry.discard_fresh(binding)

    assert registry.resolve(binding.cookie_value) is None
    assert not binding.session.root.exists()


def test_malformed_or_missing_state_never_becomes_a_filesystem_path(
    fixture_root: Path,
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "gateway"
    registry = DemoSessionRegistry(state_root, fixture_root)
    binding = registry.get_or_create(None, client_key="client-a")
    outside = tmp_path / "outside.txt"
    outside.write_text("safe", encoding="utf-8")

    assert registry.resolve("../../outside") is None
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            "UPDATE sessions SET session_id = ? WHERE session_id = ?",
            ("../../outside", binding.session.session_id),
        )
    with pytest.raises(GatewaySessionUnavailableError, match="Invalid stored"):
        registry.resolve(binding.cookie_value)

    assert outside.read_text(encoding="utf-8") == "safe"
    assert binding.session.root.is_dir()
