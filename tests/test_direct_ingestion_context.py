from __future__ import annotations

from datetime import UTC, datetime

from rcp.agents import ContextAssembler
from rcp.core.models import GraphState


def test_contract_names_direct_provider_roots_and_project_watermark(manifest) -> None:
    watermark = datetime(2026, 7, 31, 14, 0, tzinfo=UTC)
    state = GraphState(
        project_truth_scope=manifest.project.truth_scope,
        last_refresh_at=watermark,
    )
    assembler = ContextAssembler(manifest)
    roots = assembler.source_roots("laptop")

    context = assembler.assemble(
        state,
        run_truth_scope=["repo-a"],
        source_roots=roots,
        source_errors=["codex root is unavailable"],
    )
    payload = context.prompt_payload()

    assert context.ingestion_watermark == watermark
    assert payload["source_roots"] == roots
    assert payload["source_errors"] == ["codex root is unavailable"]
    assert "sessions" not in payload
    assert "session_routing_index" not in payload
    assert "processed_cursors" not in payload


def test_new_project_has_no_ingestion_watermark(manifest) -> None:
    context = ContextAssembler(manifest).assemble(
        GraphState(project_truth_scope=manifest.project.truth_scope),
        source_roots={"codex": ["/logs/codex"], "claude": ["/logs/claude"]},
    )

    assert context.ingestion_watermark is None
