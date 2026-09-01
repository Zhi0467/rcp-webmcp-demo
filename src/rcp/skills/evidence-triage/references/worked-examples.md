# Worked examples

Use these generic examples to calibrate the narrow claim carried by Evidence.

## Calibration Evidence informs a Decision

> **observation** Three completed calibration runs show stable throughput through batch 16, then
> memory failures at batch 24.
> **interpretation** This bounds the feasible batch-size options for the main run. It does not
> choose among the feasible options.
> **role** result  **validity** valid  **origin** internal_run
> **relations** calibration Experiment `produces` this Evidence; this Evidence `informs` the batch
> size Decision.

The measurements establish the feasible set. A human-owned action records which option is selected.
Do not make the Decision govern the calibration Experiment when the calibration exists to inform
it, and do not put an Evidence-to-Hypothesis assessment on `informs`.

## Smoke Evidence addresses a Blocker

> **observation** The smoke run completed one end-to-end batch, wrote the expected artifact, and
> passed the schema check.
> **interpretation** This addresses the unverified-pipeline Blocker for the main run. The short run
> does not test the scientific effect targeted by the main Experiment.
> **role** diagnostic  **validity** valid  **origin** internal_run
> **relations** smoke Experiment `produces` this Evidence; this Evidence `addresses` the pipeline
> Blocker.

The Evidence can justify a later lifecycle update, but the `addresses` edge does not itself close
the Blocker. Do not attach an Evidence-to-Hypothesis assessment to it, and do not attach the
downstream Blocker as an input that blocks its own smoke test.

## An incomplete run is a snapshot

> **observation** At the cited timestamp, the job was healthy at step 117 of 120 with no recorded
> traceback or resource failure.
> **interpretation** This is a bounded runtime snapshot only; completion and final evaluation
> remain unobserved.
> **role** result  **validity** qualified  **origin** internal_run

Put the time boundary in the observation. Later readers must not mistake a monitoring snapshot for
live state or a completed result.

## One result bears differently on two Hypotheses

> **observation** In the shifted regime, replanning restored held-out accuracy for the small model;
> the large-model run ended before evaluation.
> **interpretation** The completed comparison directly tests the small-model mechanism. It gives
> only contextual information about whether the mechanism scales.
> **role** result  **validity** qualified  **origin** internal_run
> **relation 1** `supports` the small-model Hypothesis with `relevance: direct`, `weight: strong`,
> `scope: small model under the shifted regime`, and `qualifications: []`.
> **relation 2** `inconclusive` for the scaling Hypothesis with `relevance: contextual`,
> `weight: limited`, no scope, and `qualifications: ["The large-model run ended before
> evaluation."]`.

The relation supplies direction. Each assessment separately records directness, weight, scope, and
caveats for its own claim; neither assessment becomes a global property of the Evidence node.

## A citation must carry its claim

> **observation** Peak memory remained below the configured limit.
> **source excerpt** “The launch uses the approved runtime configuration.”

The excerpt does not contain a memory measurement. Cite the metric artifact or the exact record that
reports the peak. A source from the right conversation is not sufficient provenance.

## Suggestions are not Evidence

> **record** “We should use the smaller checkpoint interval.”

Frame this as a Decision when it is a lasting research choice; otherwise state the suggestion in the
answer. It is neither an empirical observation nor proof that the option was selected.
