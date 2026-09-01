from __future__ import annotations

from pathlib import Path

import pytest

from rcp.config import Manifest, load_manifest


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    research = repo_a / ".research"
    research.mkdir()
    claude_root = tmp_path / "claude"
    codex_root = tmp_path / "codex"
    claude_root.mkdir()
    codex_root.mkdir()
    path = research / "manifest.toml"
    path.write_text(
        f'''name = "test-paper"

[[machines]]
alias = "laptop"
host = ""

[[repositories]]
alias = "repo-a"
machine = "laptop"
path = "{repo_a}"

[[repositories]]
alias = "repo-b"
machine = "laptop"
path = "{repo_b}"

[project]
truth_scope = ["repo-a", "repo-b"]

[state]
repository = "repo-a"

[agent]
default_run_truth_scope = ["repo-a"]

[sources]
claude_roots = ["{claude_root}"]
codex_roots = ["{codex_root}"]

[execution]
run_on = "laptop"

[paper.coach]
default_provider = "codex"
default_model = ""
default_reasoning = "medium"
''',
        encoding="utf-8",
    )
    return load_manifest(path)
