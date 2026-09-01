---
id: experiment-causality
kind: skill
label: Experiment causality
version: 1.0.0
description: Construct, repair, or globally check an experiment action program during Seed, Refresh, or graph-capable Work by tracing Decisions and Blockers through precursor Experiments and Evidence to main Experiments.
dependencies:
---

# Experiment causality

Trace every main or next Experiment back through the conditions that make it runnable. Preserve
existing ResearchQuestions and Hypotheses unless independent evidence requires changing them.

## Set authority first

Follow the outer task contract. In a read-only audit or when loaded by Research graph audit, report
findings only. In a graph-capable task that asks for construction or repair, contribute only to its
authorized Patch. Never select a Decision, approve an operation, set standing, or change truth
membership.

## Trace each main Experiment

1. **List its gates.** Collect Decisions reached through `governed_by`, Blockers reached through
   `blocked_by`, and prerequisites stated only in prose.
2. **Classify how each gate is settled:** human choice, external availability, ordinary operational
   work, existing Evidence, or a new empirical result.
3. **Build an empirical handoff only when needed.** For a new empirical result, create or reuse the
   smallest bounded precursor Experiment. Connect:

   ```text
   precursor Experiment -> produces -> Evidence
   Evidence -> informs -> downstream Decision
   Evidence -> addresses -> downstream Blocker
   main Experiment -> governed_by or blocked_by -> downstream gate
   ```

   Use only the applicable Evidence-to-gate edge. `informs` does not choose a Decision; `addresses`
   does not change Blocker status.
4. **Recurse through the precursor.** Identify its genuine input Decisions and Blockers, classify
   their resolution sources, and repeat only for empirical gates. Never move a gate that the
   precursor will settle backward into that precursor's inputs.
5. **Stop at a real resolution source.** Do not invent Experiments for human choices, external
   outages, repository access, implementation tasks, ordinary retries, or already sufficient
   Evidence.

## Check the complete action program

- **Reversed:** a downstream Decision governs, or downstream Blocker blocks, the precursor meant to
  inform or address it.
- **Prose-only:** a required gate or handoff exists only in summaries.
- **Circular:** following gates and empirical resolution paths returns to the same node.
- **Self-blocking:** an Experiment is blocked by the condition its own Evidence is meant to address.
- **Stale:** lifecycle text or status conflicts with later Evidence or action edges.
- **Duplicate:** parallel nodes or paths represent the same gate, Experiment, or Evidence.
- **Incomplete:** an empirical gate lacks a precursor, produced Evidence, handoff, or connection to
  the main Experiment.

Reuse existing node identities and relations whenever they express the same entity. Do not create a
second path merely to make the chain visually complete.

## Finish

Before reporting or writing a Patch, rerun the trace on the candidate graph. For every main
Experiment, state each gate, its resolution class, and the structural path that settles it. Report
unresolved human or external gates honestly; causal closure does not mean pretending they are clear.
