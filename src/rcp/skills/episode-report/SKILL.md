---
id: episode-report
kind: skill
label: Episode report
version: 1.0.0
description: Create the required durable visual HTML wrap-up for an RCP episode, explaining its work, evidence, limits, ending, and next human decision without changing project or graph state.
dependencies:
---

# Episode report

Produce one self-contained, valid HTML report at the exact output path in the
episode wrap-up instruction. This is a retrospective for a researcher taking
over from the episode, not another operational research turn.

Make the report inherently visual. Use an intentional visual hierarchy plus the
charts, diagrams, timelines, matrices, annotated evidence maps, or other visual
forms that best expose the episode's structure and conclusions. Do not merely
decorate a prose memo. Keep every visual honest about missing evidence and
uncertainty. The HTML must remain useful in RCP's opaque sandbox without network
resources.

State why the episode ended and distinguish observations from interpretation.
If it exhausted its operational ceiling, failed, or paused for human authority,
make the partial boundary conspicuous and never imply unfinished work happened.
Use only the compact immutable episode receipt supplied for this continuation
and the native session's existing context. Do not seek or rebuild graph,
research, transcript, or repository context during wrap-up.

## Experiment-loop guide

For an `experiment_loop` episode, emphasize the objective, method and relevant
configuration, scientifically meaningful attempts, observations, evidence,
failure analysis, limitations, and the exact completion or human-authority pause.
End with the next falsifying test or human decision that the evidence supports.

## Auto-research guide

For an `auto_research` episode, also explain epistemic movement across the
research graph, Decisions made or awaiting authority, delegated-agent and worker
orchestration, what progressed or failed, unresolved uncertainty, and a concise
briefing that lets the researcher resume control without reconstructing the
episode chronology.

## Authority boundary

The report is descriptive only. It has no Patch, watcher, command, Proposal, or
graph-authority output. Do not write or modify any file except the exact report
output. The current graph and Patch history remain the sources of project truth.
