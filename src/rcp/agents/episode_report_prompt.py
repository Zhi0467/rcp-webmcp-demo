from __future__ import annotations


def episode_report_task_contract(
    *,
    project_name: str,
    ending: str,
    partial: bool,
    receipt_path: str,
    receipt_sha256: str,
    report_skill_path: str,
    report_output_path: str,
    correction_diagnostic_path: str | None = None,
) -> str:
    """Build the mode-neutral report contract for an exact episode continuation."""

    ending_rule = (
        "This is a partial ending. Make unfinished, uncertain, and unverified work explicit; never "
        "imply that incomplete work finished or that failed work succeeded."
        if partial
        else "Represent the ending faithfully without smoothing over failures or abandoned routes."
    )
    correction_pointer = (
        f"- exact report correction diagnostic: `{correction_diagnostic_path}`\n"
        if correction_diagnostic_path
        else ""
    )
    correction_rule = (
        """
Report correction:
- Read the supplied exact diagnostic and correct only the HTML report. Do not revisit or repeat
  episode work, broaden the input set, or produce any other output.
"""
        if correction_diagnostic_path
        else ""
    )

    return f"""# RCP episode report contract

This turn is an exact native-session resume in the episode's exact retained stage. It is report
generation only, not an operational episode invocation. Produce the durable report for project
`{project_name}` at ending `{ending}`.

Required read-only inputs:
- immutable compact episode receipt: `{receipt_path}`
- expected receipt SHA-256: `{receipt_sha256}`
- exact official `episode-report` SKILL.md: `{report_skill_path}`
{correction_pointer}Only permitted output:
- self-contained sandbox-safe HTML report: `{report_output_path}`

Before using the receipt, verify that its exact bytes have the expected SHA-256 above. Treat a
missing receipt or digest mismatch as a report-generation failure: do not infer, recreate, or
substitute its contents.

Use only the retained native-session context and the supplied compact receipt. Never seek,
restage, rebuild, or read a graph, research rendering, transcript, repository, event ledger, or
other episode history. Do not widen the context even if another path is remembered by the native
session.

Read and follow the exact official skill named above. The receipt and skill carry any
mode-specific report guidance; this envelope adds no mode-specific format or second visual rubric.
{ending_rule}

Report-only authority:
- Write only the exact HTML output named above. Do not create or change a Patch, watcher, command,
  Proposal, message, repository content, canonical state, or any other file or external state.
- Do not repeat completed operational work or attempt unfinished operational work while preparing
  or correcting the report.
{correction_rule}
The output must be one non-empty UTF-8 HTML document that is self-contained for RCP's sandboxed
renderer. It must not depend on external scripts, images, fonts, fetches, forms, popups, or
downloads. Your final assistant response may only confirm that the report was written.
"""
