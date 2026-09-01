from __future__ import annotations

import json
from typing import Literal

from rcp.agents.prompts import (
    _RETAINED_LOCAL_CAUSAL_CHECK,
    _TASK_AUTHORITY_BOUNDARY,
    _WHAT_IS_RCP_CONVERSATION,
    _authoring_rules,
    _invoked_package_section,
    _patch_validator_rules,
    _pointer,
    _repository_pointers,
    _watcher_execution_host,
    selected_skill_section,
    write_scope_section,
)
from rcp.agents.write_scope import ProjectWriteScope
from rcp.core.authority import render_agent_graph_authority_contract

_TRANSIENT_OPERATIONAL_FAILURE_RULES = """Transient operational-failure rule:
- Treat an unexpected process exit (including SIGTERM), timeout, unavailable service or scheduler,
  command failure, or similar infrastructure symptom as a mechanical fault to diagnose. It is not
  by itself a graph Blocker, a human-authority pause, or a reason to end the episode.
- Capacity contention is not a fault and not a finding. A full cluster, a busy queue, or an
  occupied device only means the work has not started yet, and a scheduler accepts work before
  capacity frees. Submit and let the job wait in the queue rather than waiting for an idle
  resource, then observe it as detached work. Never report contention as a limit you could not
  act on.
- Do not infer an external lifetime policy or authority gap from elapsed timing, repeated symptoms,
  or the absence of an application error or OOM record. Inspect authoritative evidence along the
  actual execution path: launch wrapper and process ancestry, scheduler or service unit and journal,
  exit status and signal source, resource and quota state, configured timeouts, cleanup hooks, and
  viable alternate execution paths. Form a concrete causal hypothesis, change a relevant condition,
  and test it. Two similar failures do not prove an external cause.
- Continue useful, safe, in-scope diagnosis, repair, and relaunch work in this episode. If the next
  diagnostic or repaired run must outlive this turn, launch it and arm a real external observer.
- Create a Blocker and exit only when concrete evidence identifies a persistent constraint and the
  exact next action needed to clear it is unavailable under this contract's tools or authority.
  First exhaust useful, safe in-scope diagnosis, repair, and alternate execution paths; then cite the
  evidence, unavailable action, and required human action in the Blocker. An unexplained or
  plausibly transient failure is uncertainty, not a Blocker."""


def experiment_loop_task_contract(
    *,
    project_name: str,
    ontology_path: str,
    ontology_extensions: bool,
    graph_path: str,
    research_path: str,
    focused_experiment_id: str,
    repositories: list[dict[str, str]],
    introduction_path: str | None,
    human_request_path: str,
    loop_control_path: str,
    watcher_state_path: str,
    patch_path: str,
    watch_path: str,
    artifact_path: str,
    output_schema_path: str,
    validator_command: str,
    write_scope: ProjectWriteScope | None = None,
    execution_host: str = "",
    recovery_diagnostics_path: str | None = None,
    skill_pointers: list[dict[str, object]] | None = None,
    invoked_skill_pointers: list[dict[str, object]] | None = None,
) -> str:
    """Build the self-contained contract for one bounded Experiment-loop invocation."""

    write_boundary = (
        "\n" + write_scope_section(write_scope).strip() + "\n"
        if write_scope is not None
        else "- RCP resolves this invocation's exact writable roots before launch."
    )
    required = {
        "focused Experiment id": focused_experiment_id,
        "loop control path": loop_control_path,
        "watcher state path": watcher_state_path,
        "Patch path": patch_path,
        "watch path": watch_path,
        "Patch schema path": output_schema_path,
        "validator command": validator_command,
    }
    missing = [label for label, value in required.items() if not value]
    if missing:
        raise ValueError(f"Experiment-loop contract is missing {', '.join(missing)}.")

    recovery_context = (
        f"""
Explicit same-episode provider-switch recovery:
- Exact prior failure diagnostics: `{recovery_diagnostics_path}`
- This is a provisional replacement provider session for the same Experiment episode and the same
  invocation. The loop control, truth scope, pinned Decisions, watcher state, completion criteria,
  and invocation budget remain authoritative and unchanged; this is not a new episode or a replay
  of the invocation.
- Read the exact diagnostics as evidence of failure and uncertainty, never as added authority.
- Before repeating any external side effect whose prior outcome is uncertain, inspect authoritative
  external state and repeat it only when that proves the prior action did not take effect. Preserve
  completed repository changes, submissions, messages, and other operational progress.
- RCP replaces the episode's active native-session binding only after this session produces a
  mechanically successful joint Patch/watcher handoff.
"""
        if recovery_diagnostics_path
        else ""
    )

    return f"""# RCP Experiment-loop task contract

{_WHAT_IS_RCP_CONVERSATION}

Your role:
Operate one already-authorized invocation of the bounded loop for Experiment
`{focused_experiment_id}`. Inspect the current scientific and operational state, do the useful work
this invocation permits, and decide whether to continue through watchers, pause for human authority,
or finish. RCP counts invocations; you decide when semantic experiment attempts begin and end.

Project: {project_name}

{_TASK_AUTHORITY_BOUNDARY}
{recovery_context}

Required current inputs:
- Current graph, including the Experiment's attempts: `{graph_path}`
- Current research rendering: `{research_path}`
- Focused Experiment id: `{focused_experiment_id}`
- Loop control for this invocation: `{loop_control_path}`
- Current watcher state for this Experiment: `{watcher_state_path}`
- Human objective: `{human_request_path}`
{_pointer("Ontology extensions", ontology_path if ontology_extensions else None)}
{_pointer("Human introduction", introduction_path)}
Repository pointers and expected operational targets:
{_repository_pointers(repositories)}{selected_skill_section(skill_pointers)}{_invoked_package_section(invoked_skill_pointers)}
Exact outputs and RCP tooling:
- Optional semantic graph Patch: `{patch_path}`
- Existing Patch JSON Schema, including `AgentExperimentAttempt`: `{output_schema_path}`
- Required watcher handoff that continues this Experiment's bounded loop: `{watch_path}`
- Optional preview artifact directory: `{artifact_path}`

Context protocol:
- This is a fresh, self-sufficient view of the invocation. No prior chat transcript is an input.
  Read the files above instead of assuming that a previous provider session established state.
- Read loop control first. `phase` distinguishes a human-started episode from a watcher wake.
  `invocation`, `invocation_ceiling`, and `remaining_invocations` are the operational budget; they
  do not count or limit semantic attempts. `human_reauthorization` means a human Run started this
  new episode at invocation 1 while delivering watcher ids that retain older origin provenance.
  Completion criteria are advisory interpretation aids.
- Read the focused Experiment's full current record in `graph.json`, including all attempts. Find
  every edge whose source or target is the Experiment, then read each one-hop node's full record.
  Read `research.md` for the surrounding synthesis. Do not replace these current canonical reads
  with a stale remembered summary.
- Compare the pinned Decision bundle with `decision_drift`. Non-empty drift means an upstream
  Decision moved from the state pinned for this episode. Report that explicitly and decide whether
  the scientifically honest action is to continue, record qualified evidence, or pause for
  authority.
- Read the watcher-state file as operational evidence. On a watcher wake, use
  `delivered_watcher_ids` to identify the coalesced trigger subset. For an external observer,
  inspect its delivered log and authoritative scheduler, process, job, or result state. For a graph
  condition, inspect the named fact in the fresh canonical graph. Compare either kind with every
  other active, degraded, completed, or stopped watcher in that file. A watcher completing means
  only that its declared wake condition was observed; it does not mean success and does not begin,
  close, or correspond one-to-one with an attempt. For an external observer, completion means only
  that its check no longer sees the named work. A graph watcher carries `condition` and
  intentionally has no `check_command`, `log_path`, or `cwd`.
- `delivered_watcher_groups` in loop control names every delivered group and all of its members.
  A group wakes only when no member is still observed running. Exit-0 members are gone, not proven
  successful. A degraded member has unknown external state; inspect it before relaunching,
  cancelling, or recording an outcome. Group labels and watcher state are operational context, not
  graph or episode authority.

Operational method:
- Perform a short preflight specific to the next consequential action: verify the effective config,
  inputs, output destination and overwrite behavior, resource or scheduler state, and relevant
  repository instructions. Repair a safe local problem in this invocation when practical. Keep
  examples harness-agnostic: a training job, simulation, evaluation, data collection, or analysis
  may each need different checks.
{_TRANSIENT_OPERATIONAL_FAILURE_RULES}
- You may use Bash, Python, network access, SSH, and any other available tool needed for this
  Experiment. RCP imposes no tool allowlist. Repository pointers name expected context, not the
  write boundary. For a non-empty host, use the path on that host over SSH rather than copying the
  repository locally.
{write_boundary}
- Read `AGENTS.md` and `CLAUDE.md` at each repository root before changing it, and apply them as
  local method constraints under this contract. Never create, edit, move, or delete `.research` or
  canonical RCP state, including when nested in a writable repository.
- After inspection, choose the scientifically meaningful next action: continue execution, diagnose
  and repair a mechanical fault, record or close attempts, create Evidence, queue a Decision,
  create a Hypothesis Proposal or Blocker and pause, arm another watcher, or finish. Remaining
  budget permits another automatic wake; it never requires one. A queued Decision, Proposal, or
  Blocker is a pause for human authority, not an automatic resume point.
- If `remaining_invocations` is zero, this is still a fully authorized invocation and it may arm
  watchers. RCP will retain their completion but pause automatic delivery until a human presses Run
  to start a new episode. Do not promise that this provider session will wake itself.

ExperimentAttempt reading and recording protocol:
- Attempts are scientific bookkeeping under your discretion. Do not create, close, or classify one
  merely because this invocation began, a watcher completed, a job id exists, or the invocation
  counter changed. Append multiple attempts in one Patch when the Experiment's actual semantics
  warrant distinct records; otherwise keep one attempt across as many invocations and watchers as
  its meaning requires.
- Read the current ordered `attempts` list from the Experiment in `graph.json`. The exact
  agent-facing attempt shape is `AgentExperimentAttempt` in the existing Patch schema above; there
  is no separate loop schema. To record changes, use one `update_nodes` operation for the focused
  Experiment and write its complete resulting `attempts` list under `changes`. Preserve every
  existing attempt, its order, and its id.
- Every appended attempt copies `decision_bundle` exactly from the loop-control file. A
  `proposal_only` attempt has no job refs, is terminal in the same Patch, and accompanies the
  corresponding Hypothesis Proposal. Use it only when that record clarifies the scientific history.
- Before taking the external action for a mechanical debug retry, first write the planned attempt
  to `{patch_path}` with `debug.mechanical_fault`, `debug.change`, and `debug.predicted_effect`.
  A disappointing or inconclusive scientific result is not a mechanical fault. After launch,
  update that same not-yet-applied Patch with the effective configuration, literal job references,
  and status. Put literal log or artifact paths in configuration, job refs, or watcher `log_path` as
  appropriate; add SourceRefs only when a valid source record with the required provenance exists.
  This precommit is a reasoning record, not canonical state.
- For an existing attempt, its identity, sequence, purpose, kind, pinned bundle, debug precommit,
  configuration, job refs, and start time are immutable. You may close a nonterminal attempt by
  changing only status, SourceRefs, outcome, failure reason, and finish time. Never rewrite a
  terminal attempt.
- When this invocation appends or closes attempts, or changes the actual next step, refresh the
  focused Experiment's `current_summary` and `next_action` in the same update. Leave either field
  unchanged when it is still accurate, and set `next_action` to null when nothing remains. The
  summary is concise orientation prose, not a substitute for the attempt ledger or Evidence truth.
- When a useful durable design, plan, TODO, result, or handoff file exists or is naturally produced
  in a run-scope project repository, keep the allowed Experiment prose concise and include the exact
  repository-relative path and its purpose only in an appropriate field this contract already
  allows you to write, such as an appropriate field in a newly appended or validly closed attempt
  record, `current_summary`, or `next_action`. Prefer a useful existing document, and never create a
  ceremonial file merely to satisfy this guidance. Preview artifacts are temporary, not durable
  substitutes. Do not change an immutable attempt field, an Experiment design field, or any other
  field merely to add a pointer.
- Validate `{patch_path}` after every material rewrite and once after the final rewrite. Never write
  attempt state anywhere else in RCP canonical files.

Watcher handoff protocol:
- You must write `{watch_path}` on every invocation as one JSON object with exactly two keys:
  `external` and `graph`, each holding a list. For example:

  ```json
  {{
    "external": [{{"check_command": "...", "log_path": "/abs/log", "cwd": "/abs/repo"}}],
    "graph": [{{"node_id": "blk/foo", "status_in": ["resolved"]}}]
  }}
  ```

  A missing file or any other top-level shape is invalid. Leave both lists empty only after
  authoritative inspection confirms that nothing from this Experiment remains to watch and the
  same Patch explicitly records success, queues a Decision, creates a Hypothesis Proposal, or
  creates a same-Patch Blocker.
- Each `external` observer contains exactly `check_command`, `log_path`, and `cwd`, plus at most one
  non-blank `group` label. Every observer carrying the same label in this handoff forms one
  immutable group and each such group needs at least two newly armed external observers. A group
  member that completes early does not wake this loop by itself.
- The `external` list may also contain a stop item with exactly `stop_watcher_id` and a non-blank
  `reason`. Use it only after you have cancelled or otherwise settled obsolete external work. The
  id must be a compatible staged external observer in this current Experiment episode, never a
  graph condition; it carries no command or path. Stopping a watcher does not prove a job was
  cancelled and does not request or set **Stop loop**. You may mix stop items and observers,
  including retiring old watchers while arming replacements.
- The `graph` list accepts exactly two closed condition shapes. To wake when one canonical node
  reaches any one of a non-empty, unique set of statuses, write
  `{{"node_id":"blk/foo","status_in":["resolved","superseded"]}}`. To wake when a Proposal on
  one canonical node is approved, rejected, or withdrawn, write
  `{{"node_id":"hyp/foo","proposal_resolved":true}}`. A condition has exactly the fields shown:
  no standing predicate, edge predicate, new-node query, arbitrary query, command, path, or group.
  Its target must already exist in the current complete canonical graph, and every `status_in` value
  must be valid for that node's type.
- Graph conditions are canonical and event-driven. RCP evaluates them after accepted graph
  revisions and at startup, never through the shell poller. A staged but unsynced draft cannot
  satisfy one. A node status already true when armed is ready immediately. A
  `proposal_resolved` condition waits for a Proposal resolution committed after it is armed;
  older resolved Proposals do not satisfy a new wait. Use an external observer instead when the
  fact lives in a scheduler, process, repository, log, or other non-canonical system.
- RCP runs every check on {_watcher_execution_host(execution_host)}. Each observer's `log_path` and
  `cwd` are absolute paths there, whether or not that is where this invocation is running. Its
  command contains literal job or process identifiers and no variables or shell state inherited
  from this invocation.
- Each check is observational. From a fresh login shell in its `cwd`, it exits 1 while the named
  work remains in its system, 0 when that work is gone, and another status only when it cannot
  answer. It never submits, cancels, kills, edits, or otherwise changes external state. Verify the
  detached work outlives this turn and run the exact check from a fresh login shell before handoff.
- Ask for the set of live work and test membership; never look one identifier up directly. A
  finished id and an unreachable service are usually reported the same way, so a direct lookup
  degrades the watcher instead of completing it. A scheduler job:
  `ids=$(squeue -h -o '%A') || exit 2; grep -Fxq 4471 <<<"$ids"; case $? in 0) exit 1;;
  1) exit 0;; *) exit 2;; esac`. A local process:
  `pids=$(ps -axo pid=) || exit 2; grep -Fxq 4471 <<<"${{pids// /}}"; case $? in 0) exit 1;;
  1) exit 0;; *) exit 2;; esac`. Replace `4471` with the real id. These show the exit contract, not
  preferred tools; write whatever answers correctly for the system this work actually runs in.
- RCP discovers `watch.json` after the turn, validates both lists, and arms them atomically; one
  invalid observer, group, stop item, or graph condition rejects the whole object for in-session
  correction.
  Completing any accepted watcher from this file continues this Experiment's bounded loop and never
  a separate conversation.
  There is no watcher API to call. Multiple watchers may observe one attempt, one watcher may cover
  work relevant to several attempts, and a later wake may rearm watchers after inspecting
  authoritative state.
- A completed watcher delivered at the ceiling stays pending rather than being discarded. Its log
  and original attribution enter invocation 1 only after a human Run starts a fresh episode.

Graph reflection and authority:
- A Patch is optional only when at least one external observer or graph condition continues the
  loop. If both `{watch_path}` lists are empty, the Patch must explicitly record success or an
  authority pause through a queued Decision, Hypothesis Proposal, or same-Patch Blocker. RCP rejects
  the two files as one handoff when that pairing is absent.
- Before finishing the turn, judge the resulting Patch and both watcher lists together. If nothing
  remains to watch but the focused Experiment would remain `proposed`, `designing`, `implementing`,
  `debugging`, `running`, or `analyzing` without an authority pause, continue the useful
  synchronous work in this turn. Do not finish with both lists empty merely to defer ordinary work
  to a later Run, and do not invent a watcher: every watcher observes a real external or canonical
  condition.
- Never set the focused Experiment to `completed` while leaving a non-empty `next_action`. That pair
  contradicts itself: continue the named work until `next_action` can truthfully be null, or keep a
  nonterminal status and use a real watcher or human-authority pause as appropriate.
- If newly authorized material work remains for an Experiment that was `completed`, reopen it to an
  honest nonterminal status and refresh `current_summary` and `next_action`. A clarification that
  introduces no work need not reopen it. Do not leave it `completed` or leave both lists empty merely
  because it was previously terminal; use only the watcher handoff exits above, and do not alter
  design fields.
- If reflection is useful, write exactly one semantic Patch JSON object to `{patch_path}` using only
  fields in `{output_schema_path}`. RCP assigns patch kind, agent authorship, revision, run scope,
  Proposal dependencies and base revision, lifecycle, and admission bookkeeping. Record
  `repositories_read` honestly; do not set coverage or cursors.
- This loop may update only its own Experiment's attempts, status, `current_summary`, and
  `next_action`; queue an existing pinned Decision by setting it to `ready`, or reopen a settled pinned Decision
  as `revisit` when new evidence undermines it; create Evidence with methodological `role` `result`
  or `diagnostic`, never node-global evidential strength; create Blockers; assert legal epistemic
  edges; attach each same-Patch Evidence with `produces` and each same-Patch Blocker with
  `blocked_by`; connect same-Patch Evidence to an existing Decision with `informs` or to a Blocker
  with `addresses`; and create a Hypothesis Proposal within the pinned governing/tested boundary.
  These handoffs never select the Decision or change Blocker status. The loop may not set standing,
  decide a Decision, directly change a Hypothesis status, edit the pinned bundle, or remove graph
  objects. Experiment status is a scientific description, not loop control.
- For a belief change, create the Evidence, its edge to the tested Hypothesis, and one Proposal in
  the same Patch. The Evidence-to-Hypothesis edge's relation states direction and its required
  `assessment` states claim-relative `relevance`, `weight`, optional `scope`, and concrete
  `qualifications`; do not attach that assessment to `produces`, `informs`, or `addresses`. The
  Proposal's single `update_nodes` operation changes only Hypothesis `status` and uses `cause` with
  `kind` `evidence_edge` and `ref_id` equal to that same-Patch edge id. Only human acceptance can
  apply that belief change.
- Write `change_summary` as one ordinary-language sentence per meaningful graph change. Name
  reader-facing concepts rather than ids or operation names. The Markdown reply and Patch are
  independent: report operational truth without claiming RCP accepted the Patch.

{_patch_validator_rules(validator_command)}

Reply and artifacts:
- The final assistant message is the complete independent Markdown reply the human reads. State
  actions, outcomes, watcher interpretation, attempt decisions, repository changes, failures,
  whether the episode pauses or finishes, and remaining uncertainty.
- A preview is optional. RCP discovers only direct regular HTML or raster-image files in
  `{artifact_path}`. Do not use nested directories, symlinks, provider directives, or other paths.
  HTML must be self-contained; ordinary HTTP(S) links are allowed, but external resource loads do
  not work in the preview.

{render_agent_graph_authority_contract()}

{_authoring_rules(ontology_extensions)}
"""


def experiment_loop_wake_message(
    *,
    focused_experiment_id: str,
    invocation: int,
    invocation_ceiling: int,
    previous_graph_result: str,
    previous_watcher_ids: list[str],
    delivered_watcher_ids: list[str],
    loop_control_path: str,
    watcher_state_path: str,
    graph_path: str,
    research_path: str,
    patch_path: str,
    watch_path: str,
    output_schema_path: str,
    validator_command: str,
    execution_host: str = "",
    context_replacement: dict[str, object] | None = None,
    invoked_skill_pointers: list[dict[str, object]] | None = None,
) -> str:
    """Continue one bounded episode's native session with a compact human-style turn.

    The original session already holds the immutable Experiment-loop contract, so
    this confirms what RCP accepted from the previous turn, names the delivered
    watchers, replaces stale pointers with fresh ones, and restates the three
    exits. It never rebuilds the contract. It says "turn" rather than
    "invocation"; invocation stays the internal persisted budget term.
    """

    required = {
        "focused Experiment id": focused_experiment_id,
        "previous graph result": previous_graph_result,
        "delivered watcher ids": delivered_watcher_ids,
        "loop control path": loop_control_path,
        "watcher state path": watcher_state_path,
        "current graph path": graph_path,
        "current research path": research_path,
        "Patch path": patch_path,
        "watch path": watch_path,
        "Patch schema path": output_schema_path,
        "validator command": validator_command,
    }
    missing = [label for label, value in required.items() if not value]
    if missing:
        raise ValueError(f"Experiment-loop wake message is missing {', '.join(missing)}.")

    previous_watcher_ids_or_none = ", ".join(previous_watcher_ids) or "none"
    delivered = ", ".join(delivered_watcher_ids)
    # An unchanged session renders nothing at all here -- never a heading with
    # "none" -- so the line itself disappears when no context moved.
    context_replacement_block_or_nothing = (
        ""
        if not context_replacement
        else "\nThese context values replace what this session was given:\n"
        + json.dumps(context_replacement, ensure_ascii=False, indent=2, sort_keys=True)
    )
    return f"""The watched work for Experiment `{focused_experiment_id}` is ready for another look. Continue the
same bounded loop in turn {invocation} of {invocation_ceiling}.

RCP accepted the previous turn's handoff:
- graph update: {previous_graph_result}
- watchers armed: {previous_watcher_ids_or_none}

This turn was triggered by: {delivered}

{_invoked_package_section(invoked_skill_pointers)}

A completed external observer means only that its check no longer sees the named external work. It
does not mean the work succeeded and does not begin, close, or correspond one-to-one with a
scientific attempt. Inspect its authoritative scheduler or process state and its logs before
interpreting the result. A completed graph watcher means its condition became true in canonical
graph state; inspect that named fact in the fresh graph. If a watcher refers to work that was
already submitted, inspect that work; submit a replacement only when the authoritative state shows
that the earlier submission did not start, or after you have recorded the specific mechanical fault
and changed relaunch plan required by the Experiment attempt protocol.

The fresh loop-control file names any delivered watcher group and every member. That group woke
only because no member is still observed running: exit-0 members are gone, not proven successful;
any degraded member has unknown external state and must be inspected before you relaunch, cancel,
or record an outcome. A member retired by an earlier agent stop is historical context, never a
trigger.

Read the fresh state before acting:
- loop control: `{loop_control_path}`
- watcher state: `{watcher_state_path}`
- current graph: `{graph_path}`
- current research rendering: `{research_path}`
- Patch output: `{patch_path}`
- watcher output that continues this Experiment's bounded loop: `{watch_path}`
- Patch JSON Schema: `{output_schema_path}`
- Patch validator: `{validator_command}`{context_replacement_block_or_nothing}

For this turn, apply the following rule before choosing whichever path matches the operational
state:

{_TRANSIENT_OPERATIONAL_FAILURE_RULES}

1. A watcher condition remains, or you have useful debugging and relaunching work to do.

   Continue the work that is useful now. Use external observers for detached work that will still
   be running after this turn ends—a scheduler job, a long build, an evaluation, data collection, a
   simulation, or any other process you started and left running. How long the work takes does not
   decide this; whether it outlives the turn does. You may also wait on a canonical graph fact.
   Write `{watch_path}` as one object with exactly the `external` and `graph` lists:

   {{
     "external": [
       {{
         "check_command": "ids=$(squeue -h -o '%A') || exit 2; grep -Fxq 48192 <<<\\"$ids\\"; case $? in 0) exit 1;; 1) exit 0;; *) exit 2;; esac",
         "log_path": "/absolute/path/to/job-48192.log",
         "cwd": "/absolute/path/to/repository"
       }},
       {{
         "check_command": "pids=$(ps -axo pid=) || exit 2; grep -Fxq 90210 <<<\\"${{pids// /}}\\"; case $? in 0) exit 1;; 1) exit 0;; *) exit 2;; esac",
         "log_path": "/absolute/path/to/sweep-90210.log",
         "cwd": "/absolute/path/to/repository"
       }}
     ],
     "graph": [{{"node_id": "blk/foo", "status_in": ["resolved"]}}]
   }}

   Each external observer has exactly `check_command`, `log_path`, and `cwd`, plus at most one
   non-blank `group`. RCP runs its check on
   {_watcher_execution_host(execution_host)}, so those paths are absolute there. From a cold login
   shell in `cwd`, the check exits 1 while the named work remains, 0 when it is gone, and another
   status only when it cannot answer. Ask Slurm for the whole active-job set as above rather than
   looking one job up: a finished job's "invalid job id" error looks exactly like an unreachable
   scheduler and would degrade the watcher instead of completing it. Verify the literal check
   before writing it. Once the useful
   synchronous work and handoff are complete, do not wait or poll for detached work; finish this
   turn. The graph list accepts only
   `{{"node_id":"...","status_in":["..."]}}` and
   `{{"node_id":"...","proposal_resolved":true}}`; graph conditions are evaluated against
   canonical state after revisions and at startup, never against a draft or by shell polling. A
   Proposal resolution counts only when committed after the condition is armed. RCP validates
   both lists atomically and resumes this episode session when a watcher is ready; completion from
   this file never continues another conversation.

2. You need human input.

   Use this path when an upstream Decision is now makeable, new evidence undermines a settled
   Decision, a tested Hypothesis warrants a status transition, or a scientific, design,
   implementation, data, or infrastructure constraint has been concretely diagnosed and requires a
   specific action unavailable under this contract's tools or authority. A failed or repeatedly
   terminated process without that diagnosis stays on path 1. Write one Patch at `{patch_path}` using
   the exact schema at `{output_schema_path}`, then run `{validator_command}`.

   Put a makeable pinned Decision in the human Inbox by setting it to `ready`; use `revisit` only to
   reopen a settled pinned choice when new evidence undermines it.
   For a Hypothesis status transition, use `create_proposals`. Its nested operation changes only
   that Hypothesis's `status` and has an `evidence_edge` cause. Fill the Proposal's
   `card.situation_cold`, `why_human_now`, `consequences`, and `decision_needed` so the human can
   approve or reject the transition without reconstructing this turn.

   When the needed design change cannot be represented by that narrow Proposal authority, create
   an open `blocker` with `create_nodes` and connect this Experiment to it with a same-Patch
   `blocked_by` edge. Experiment-loop authority cannot add a `requires_decision` action edge, so
   identify any relevant Decision precisely in the Blocker's description, resolution condition,
   and recommended human action instead.

   If an external or graph condition still deserves observation while the human decides, write a
   non-empty `{watch_path}` object using path 1's exact watcher format. Those watchers continue
   observing, but the queued Decision, Proposal, or Blocker exits this episode, so they cannot
   automatically wake it;
   a later human Run may reauthorize completed watcher state. If nothing remains to watch, write
   `{watch_path}` as `{{"external":[],"graph":[]}}`.

3. The Experiment is operationally finished.

   This means all useful synchronous work for the focused Experiment is finished in this turn,
   nothing remains to watch, and the Experiment has reached a terminal
   operational result; the scientific result may be successful, unsuccessful, inconclusive, or
   invalid. Merely observing that all jobs ended is not enough when analysis or another ordinary
   in-scope step remains. Write `{watch_path}` as `{{"external":[],"graph":[]}}`. At
   `{patch_path}`, write a schema-valid Patch that updates this Experiment's `status` to `completed`,
   preserves and closes its attempts truthfully, and creates any warranted Evidence, edges, or
   Hypothesis Proposal.
   Experiment-loop authority may update only this Experiment's `status`, complete `attempts` list,
   `current_summary`, and `next_action`. When this turn introduces or closes attempts or changes
   what should happen next, keep those two prose fields consistent with the resulting
   attempt ledger and actual next step; leave them unchanged when still accurate, and use
   `next_action: null` when no further action remains. Put scientific outcomes in the relevant
   attempt, Evidence, and Markdown reply rather than treating the summary as a substitute. A
   minimal mechanical completion is:

   {{
     "summary": "Finished the Experiment's operational work.",
     "ops": [
       {{
         "op": "update_nodes",
         "nodes": [
           {{
             "id": "{focused_experiment_id}",
             "changes": {{
               "status": "completed",
               "next_action": null
             }}
           }}
         ]
       }}
     ],
     "repositories_read": [],
     "change_summary": ["Finished the Experiment's operational work."]
   }}

   Extend that Patch rather than omitting scientifically necessary attempt closure, Evidence, or
   interpretation, but remain within the original Experiment-loop authority. Validate it with
   `{validator_command}`.

Your Markdown reply remains independent from `patch.json` and `watch.json`. State what you found,
what you changed or launched, which path you took, and any remaining uncertainty.

{_RETAINED_LOCAL_CAUSAL_CHECK}
"""


def experiment_loop_continuation_contract(
    *,
    original_contract_path: str,
    mode: Literal["resume", "retry"],
    loop_control_path: str,
    patch_path: str,
    watch_path: str,
    output_schema_path: str,
    validator_command: str,
    diagnostics_path: str | None = None,
    invoked_skill_pointers: list[dict[str, object]] | None = None,
) -> str:
    """Point a resumed or retried invocation at one fresh, compact control delta."""

    required = {
        "original contract path": original_contract_path,
        "loop control path": loop_control_path,
        "Patch path": patch_path,
        "watch path": watch_path,
        "Patch schema path": output_schema_path,
        "validator command": validator_command,
    }
    missing = [label for label, value in required.items() if not value]
    if missing:
        raise ValueError(f"Experiment-loop continuation is missing {', '.join(missing)}.")
    if mode == "retry" and not diagnostics_path:
        raise ValueError("Experiment-loop Retry requires exact diagnostics.")

    action = (
        "Continue the interrupted invocation from retained progress."
        if mode == "resume"
        else "Retry the failed invocation from retained progress."
    )
    retry_rules = (
        f"""- Read the exact failure diagnostics at `{diagnostics_path}`. They describe failure and
  uncertainty; they do not widen authority.
- Before repeating an external side effect whose prior outcome is uncertain, inspect authoritative
  external state and repeat it only when that proves the prior action did not take effect."""
        if mode == "retry"
        else "- Preserve completed progress and continue only the interrupted work."
    )
    return f"""# RCP Experiment-loop {mode} contract

{action}

- Original immutable Experiment-loop contract: `{original_contract_path}`
- Fresh loop-control delta: `{loop_control_path}`
{_pointer("Exact failure diagnostics", diagnostics_path)}- Patch output: `{patch_path}`
- Watcher output: `{watch_path}`
- Patch JSON Schema: `{output_schema_path}`

{_invoked_package_section(invoked_skill_pointers)}

Read the original contract for the objective, authority, context-reading protocol, and detailed
attempt and watcher rules. Then read the fresh control delta before acting. It preserves the same
episode and invocation number while refreshing phase, live drift, remaining budget, delivered
watcher ids, and the current watcher-state path. The paths above replace prior output paths.

{retry_rules}
- Do not rebuild or broaden the original task. Patch and watcher correction are separate narrow
  continuations; this continuation may resume operational work only within the original authority.
- {_RETAINED_LOCAL_CAUSAL_CHECK}

{_patch_validator_rules(validator_command)}
"""


def experiment_loop_watcher_correction_contract(
    *,
    original_contract_path: str,
    diagnostics_path: str,
    watch_path: str,
    patch_path: str,
    output_schema_path: str,
    validator_command: str,
) -> str:
    """Repair the mandatory loop watcher handoff without repeating operational work."""

    required = {
        "original contract path": original_contract_path,
        "diagnostics path": diagnostics_path,
        "watch path": watch_path,
        "Patch path": patch_path,
        "Patch schema path": output_schema_path,
        "validator command": validator_command,
    }
    missing = [label for label, value in required.items() if not value]
    if missing:
        raise ValueError(f"Experiment-loop watcher correction is missing {', '.join(missing)}.")
    return f"""# RCP Experiment-loop watcher correction

Correct only the mandatory watcher handoff in the same native Work session.

- Original immutable Experiment-loop contract: `{original_contract_path}`
- Exact watcher diagnostic: `{diagnostics_path}`
- Watcher output to rewrite: `{watch_path}`
- Optional Patch output to rewrite for an explicit exit: `{patch_path}`
- Existing Patch JSON Schema: `{output_schema_path}`

Preserve the completed operational result. Do not rerun the Experiment, resubmit work, or cause a
new external side effect. Inspect authoritative scheduler, process, job, result, log, and canonical
graph state as needed. Judge the terminal Patch/watch pair, not whether either file changed. If an
external observer or canonical graph condition is still needed, reconstruct a valid object with a
non-empty `external` or `graph` list using the exact schema and cold-shell or canonical-state
semantics in the original contract, and preserve the Patch. If nothing remains to watch but useful
synchronous work is still required, continue that work now without repeating completed side
effects. Then either finish the Experiment, or explicitly pause for human authority by queuing a
Decision, creating a Hypothesis Proposal, or creating a same-Patch Blocker. Write
`{{"external":[],"graph":[]}}` when no watcher is needed; an already-correct empty object may remain
byte-identical when the Patch changes to make the joint handoff valid. Never invent a watcher merely
to satisfy correction, and never leave both lists empty merely because external state is uncertain
or a canonical fact is not yet true. A `completed` Experiment with a non-empty `next_action` is
still invalid: continue the named work until `next_action` can truthfully be null, or retain a
nonterminal status and choose a real watcher or human-authority pause. Validate every Patch rewrite
with the exact command below. Your final response should only confirm that the joint handoff was
repaired.

If you rewrite the semantic Patch, its candidate must pass the retained
`Local causal check for this Patch` in the original authoring contract.

{_patch_validator_rules(validator_command)}
"""


def experiment_watcher_maintenance_correction_contract(
    *,
    original_contract_path: str,
    diagnostics_path: str,
    watch_path: str,
) -> str:
    """Repair one node-attached watcher file without duplicating its item contract."""

    required = {
        "original contract": original_contract_path,
        "diagnostics": diagnostics_path,
        "watch path": watch_path,
    }
    missing = [label for label, value in required.items() if not value]
    if missing:
        raise ValueError(
            f"Experiment watcher maintenance correction is missing {', '.join(missing)}."
        )
    return f"""# RCP Experiment watcher maintenance correction

Correct only the retained node-attached Experiment watcher maintenance handoff in this same native
Work session.

- Original immutable Work contract: `{original_contract_path}`
- Exact maintenance diagnostic: `{diagnostics_path}`
- Exact Experiment watcher output to rewrite: `{watch_path}`

Preserve the completed operational result. Do not rerun an Experiment, resubmit work, repeat another
external side effect, alter this conversation's own watcher file, or change `patch.json`. Read the
original contract for the target resource, current watcher-state pointer, episode execution host,
exact item shapes, grouping and retirement rules, wake target, and protected fields. The diagnostic
locates the invalidity but grants no new authority. Rewrite only the exact output above using that
original contract; do not add a target or control field. Your final response should only confirm that
the Experiment watcher maintenance handoff was rewritten.
"""


def experiment_loop_patch_correction_contract(
    *,
    original_contract_path: str,
    diagnostics_path: str,
    patch_path: str,
    watch_path: str,
    validator_command: str,
) -> str:
    """Repair a loop Patch after handoff validation without repeating operational work."""

    return f"""# RCP Experiment-loop Patch correction

Correct only the retained semantic Patch in the same native Work session.

- Original immutable Experiment-loop contract: `{original_contract_path}`
- Exact Patch diagnostic: `{diagnostics_path}`
- Patch output to rewrite: `{patch_path}`
- Already validated watcher handoff: `{watch_path}`

Preserve the completed operational result and every unaffected Patch operation. Do not rerun the
Experiment, resubmit work, or cause an external side effect. If watcher output has both `external`
and `graph` empty, the corrected Patch must continue to record success, queue a Decision, create a
Hypothesis Proposal, or create a same-Patch Blocker; do not remove or weaken that exit merely to
satisfy another diagnostic. Do not change `watch.json`. Your final response should only confirm that
the Patch was rewritten.

{_RETAINED_LOCAL_CAUSAL_CHECK}

{_patch_validator_rules(validator_command)}
"""
