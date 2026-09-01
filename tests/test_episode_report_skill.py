from __future__ import annotations

from pathlib import Path

from rcp.skill_registry import official_registry


def test_episode_report_skill_is_versioned_visual_mode_aware_and_packaged() -> None:
    registry = official_registry()
    package = registry.package("skill", "episode-report")
    body = registry.package_body("skill", "episode-report")

    assert package.version == "1.0.0"
    assert "inherently visual" in body
    assert "Experiment-loop guide" in body
    assert "Auto-research guide" in body
    assert "epistemic movement" in body
    assert "delegated-agent" in body
    assert "compact immutable episode receipt" in body
    assert "Do not seek or rebuild graph" in body
    assert "no Patch, watcher, command, Proposal, or" in body

    root = Path(__file__).resolve().parents[1]
    wheel = (root / "pyproject.toml").read_text(encoding="utf-8")
    sidecar = (root / "packaging" / "rcp_backend.spec").read_text(encoding="utf-8")
    assert "src/rcp/skills/episode-report" in wheel
    assert 'SKILL_ROOT / "episode-report"' in sidecar
    assert '"rcp/skills/episode-report"' in sidecar
