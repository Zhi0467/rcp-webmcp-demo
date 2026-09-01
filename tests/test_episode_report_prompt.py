from __future__ import annotations

import inspect
import re

from rcp.agents.episode_report_prompt import episode_report_task_contract


def _paths(prompt: str) -> set[str]:
    return set(re.findall(r"`(/[^`]+)`", prompt))


def test_episode_report_contract_is_a_mode_neutral_minimal_resume_envelope() -> None:
    receipt_path = "/stage/inputs/episode-receipt.json"
    skill_path = "/stage/packages/episode-report/SKILL.md"
    output_path = "/stage/workspace/episode-report.html"
    digest = "a" * 64

    prompt = episode_report_task_contract(
        project_name="Example",
        ending="completed",
        partial=False,
        receipt_path=receipt_path,
        receipt_sha256=digest,
        report_skill_path=skill_path,
        report_output_path=output_path,
    )

    assert _paths(prompt) == {receipt_path, skill_path, output_path}
    assert "exact native-session resume" in prompt
    assert "exact retained stage" in prompt
    assert "not an operational episode invocation" in prompt
    assert f"expected receipt SHA-256: `{digest}`" in prompt
    assert "verify that its exact bytes have the expected SHA-256" in prompt
    assert "Use only the retained native-session context and the supplied compact receipt" in prompt
    assert (
        "Never seek,\nrestage, rebuild, or read a graph, research rendering, transcript" in prompt
    )
    assert "do not infer, recreate, or\nsubstitute its contents" in prompt
    assert "exact official `episode-report` SKILL.md" in prompt
    assert "adds no mode-specific format or second visual rubric" in prompt
    assert "Write only the exact HTML output" in prompt
    assert (
        "Patch, watcher, command,\n  Proposal, message, repository content, canonical state"
        in prompt
    )
    assert "external scripts, images, fonts, fetches, forms, popups, or\ndownloads" in prompt
    assert "auto-research" not in prompt.casefold()
    assert "experiment loop" not in prompt.casefold()
    assert "campaign" not in prompt.casefold()
    assert "fallback" not in prompt.casefold()
    assert "current graph:" not in prompt.casefold()
    assert "current research" not in prompt.casefold()
    assert "campaign history" not in prompt.casefold()
    assert "episode history:" not in prompt.casefold()


def test_episode_report_contract_has_no_mode_or_large_context_parameters() -> None:
    parameters = set(inspect.signature(episode_report_task_contract).parameters)

    assert parameters == {
        "project_name",
        "ending",
        "partial",
        "receipt_path",
        "receipt_sha256",
        "report_skill_path",
        "report_output_path",
        "correction_diagnostic_path",
    }
    assert not parameters & {
        "mode",
        "surface",
        "graph_path",
        "research_path",
        "history_path",
        "repositories",
        "skill_pointers",
    }


def test_episode_report_correction_adds_only_its_diagnostic_pointer() -> None:
    receipt_path = "/stage/inputs/episode-receipt.json"
    skill_path = "/stage/packages/episode-report/SKILL.md"
    output_path = "/stage/workspace/episode-report.html"
    diagnostic_path = "/stage/inputs/report-diagnostic.txt"

    prompt = episode_report_task_contract(
        project_name="Example",
        ending="failed",
        partial=True,
        receipt_path=receipt_path,
        receipt_sha256="b" * 64,
        report_skill_path=skill_path,
        report_output_path=output_path,
        correction_diagnostic_path=diagnostic_path,
    )

    assert _paths(prompt) == {receipt_path, skill_path, output_path, diagnostic_path}
    assert "This is a partial ending" in prompt
    assert "correct only the HTML report" in prompt
    assert "Do not revisit or repeat\n  episode work" in prompt
    assert "produce any other output" in prompt
    assert prompt.count(diagnostic_path) == 1
