---
id: research-graph-audit
kind: workflow
label: Research graph audit
version: 3.0.0
description: Run a deliberate read-only audit of graph structure, experiment action causality, and load bearing Evidence provenance, ending in one prioritized report.
dependencies:
- graph-audit@3.0.0
- experiment-causality@1.0.0
- evidence-triage@3.0.0
---

# Research graph audit

Produce one read-only report. This workflow and its dependencies grant no authority beyond the outer
task, and all three passes remain report-only inside this workflow.

## Pass 1: broad structure

Apply Graph audit. Read `research.md` before `graph.json`; list the claims and action state a reader
would believe. Check support, relation reasoning, truthful node roles, duplicate identity, lifecycle
drift, and unresolved action gates. Record important paths that are sound.

## Pass 2: action causality

Apply Experiment causality to every main or next Experiment. Classify each Decision and Blocker by
its resolution source, recurse through empirical precursor Experiments, and verify complete
`precursor Experiment -> Evidence -> Decision|Blocker <- main Experiment` paths, where the last
stored edge is `governed_by` or `blocked_by` from the main Experiment to its gate. Report reversed,
prose-only, circular, self-blocking, stale, duplicate, and incomplete dependencies.

## Pass 3: narrow provenance

Apply Evidence triage only to Evidence carrying the claims and action gates identified by the first
two passes. Check source precedence, observation and interpretation boundaries, methodological
role, validity, citations, and each Evidence-to-Hypothesis edge's relation direction, relevance,
weight, scope, and qualifications. Treat historical unassessed relations as legacy uncertainty;
never infer weight from a legacy global strength label. Check separately whether `informs` or
`addresses` is being mistaken for a human decision or lifecycle transition; those action edges do
not carry a Hypothesis assessment.

## Deliver one report

Use the Graph audit report structure. Fold causal and provenance findings into the same prioritized
Concerns section. Mark checkable facts separately from judgment, name the smallest next action, and
identify who has authority to take it.

Do not edit canonical `.research` files. If the invoking task also requests graph changes, finish
and present the audit report first; any later Patch remains a distinct act under that task's explicit
graph authority.
