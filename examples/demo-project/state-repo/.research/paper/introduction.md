# Introduction

## What question we study

Continual learners can perform well immediately after one task shift yet lose
the ability to adapt when the environment changes again. This miniature study
asks whether search-assisted training preserves more second-shift learning than
a value-only baseline after both arms follow matched first-shift trajectories.

## What adjacent questions there are

The study separates later learning from immediate endpoint performance. It does
not determine whether search itself causes any difference, which internal
representation carries it, or whether the pattern generalizes beyond this
synthetic held-out table.

## Literature review

The public fixture does not assert an external literature result. Background
reading remains outside the current run scope until a researcher reviews and
admits it through RCP's ordinary Evidence workflow.

## High-level methods

Three fixed synthetic seeds are recorded for a value-only arm and a
search-assisted arm. Each seed contains five first-shift and five second-shift
updates. Before comparing second-shift learning slopes, the analysis requires
the arm-level first-shift return and policy-KL trajectories to remain within an
absolute gap of 0.02 at every update.

## Main results

No terminal result is stated here before the bounded Experiment runs. The
checked-in table and reference script make the planned analysis reproducible;
RCP will retain any completed result as scoped Evidence with its visual
artifact and qualifications.

## Why this deserves publication and communication to the community

The scientific example is deliberately small. Its purpose is to demonstrate a
research workflow in which an agent can inspect the exact evidence, execute a
bounded analysis, and preserve what changed without silently promoting a narrow
result into a broader claim.
