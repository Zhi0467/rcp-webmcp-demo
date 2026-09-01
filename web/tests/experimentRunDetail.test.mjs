import assert from "node:assert/strict";
import { withExperimentControlAnswers, withTaskAnswers, withTurnAnswers } from "./taskAnswers.mjs";
import { after, test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { ExperimentRunDetail } = await server.ssrLoadModule(
  "/src/components/ExperimentRunDetail.tsx",
);
const { buildExperimentRun, experimentWatcherDisplayItems } =
  await server.ssrLoadModule("/src/runProjection.ts");
const { ExecutionView } = await server.ssrLoadModule("/src/views/GraphViews.tsx");

after(() => server.close());

function node(fields = {}) {
  return {
    id: "experiment/detail",
    type: "experiment",
    title: "Detailed bounded experiment",
    extension_fields: {},
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    status: "running",
    objective: "Measure future plasticity.",
    current_summary: "The detached evaluation is still running.",
    next_action: "Inspect the held-out evaluation.",
    invocation_ceiling: 3,
    completion_criteria: ["All held-out evaluations have mechanically exited."],
    attempts: [
      {
        id: "attempt-1",
        sequence: 1,
        purpose: "Run the held-out evaluation",
        attempt_kind: "external_run",
        decision_bundle: [],
        status: "running",
        outcome: null,
        failure_reason: null,
        job_refs: ["slurm-48192"],
      },
    ],
    ...fields,
  };
}

function operational(fields = {}) {
  return withTurnAnswers({
    task_active: false,
    detached_work_active: false,
    watcher_degraded: false,
    watcher_completion_pending: false,
    episode_exited: false,
    episode_live: false,
    stop_requested: false,
    stop_settled: false,
    chat_id: "chat-1",
    current_operation_id: null,
    current_status: null,
    current_phase: null,
    current_status_message: null,
    current_last_activity_at: "2026-08-06T04:00:00Z",
    current_invocation: 3,
    session: {
      provider: "codex",
      model: "gpt-5.6",
      reasoning: "high",
      run_on: "cluster",
      execution_host: "login.research",
      run_truth_scope: ["repo-a", "repo-b"],
      native_session_bound: true,
      diagnostic: null,
    },
    ...fields,
  });
}

function control(fields = {}, operationalFields = {}) {
  return withExperimentControlAnswers({
    ready: true,
    reasons: [],
    graph_reasons: [],
    invocations_used: 3,
    invocation_ceiling: 3,
    invocations_remaining: 0,
    episode_id: "episode-1",
    episode: null,
    paused: true,
    active: false,
    governing_decisions: [
      { decision_id: "decision/resources", decision_revision: 7, selected_option: "4xA100" },
    ],
    decision_drift: [
      {
        decision_id: "decision/data",
        pinned_option: "v1",
        pinned_revision: 4,
        current_option: "v2",
        current_status: "decided",
        proposed: false,
      },
    ],
    operational: operational(operationalFields),
    ...fields,
  });
}

function watcher(fields = {}) {
  return {
    watcher_id: "watcher-1",
    project_id: "project",
    origin_operation_id: "origin-turn",
    origin_task_kind: "node_chat",
    chat_id: "chat-1",
    node_id: "experiment/detail",
    execution_host: "login.research",
    check_command:
      "ids=$(squeue -h -o '%A') || exit 2; grep -Fxq 48192 <<<\"$ids\"; case $? in 0) exit 1;; 1) exit 0;; *) exit 2;; esac",
    log_path: "/scratch/evaluation.log",
    cwd: "/scratch/run",
    continuation: {
      provider: "codex",
      model: "gpt-5.6",
      reasoning: "high",
      run_on: "cluster",
      run_truth_scope: ["repo-a", "repo-b"],
      patch_kind: "experiment_loop",
      control_node_id: "experiment/detail",
      control_revision: 7,
      control_episode_id: "episode-1",
      control_invocation: 3,
      control_invocation_ceiling: 3,
      control_decision_bundle: [],
      control_completion_criteria: [],
      workflow_ids: [],
      skill_ids: [],
      invoked_workflow_ids: [],
      invoked_skill_ids: [],
      resolved_skill_packages: [],
    },
    status: "completed",
    created_at: "2026-08-06T01:00:00Z",
    last_checked_at: "2026-08-06T04:00:00Z",
    last_exit_code: 0,
    last_error: null,
    completed_at: "2026-08-06T04:00:00Z",
    next_check_at: null,
    consecutive_error_count: 0,
    group_id: null,
    group_label: null,
    notified: false,
    notification_operation_id: null,
    stopped_by: null,
    stop_reason: null,
    stopped_at: null,
    stop_operation_id: null,
    ...fields,
  };
}

function graphWatcher(fields = {}) {
  const external = watcher(fields);
  const {
    check_command: _checkCommand,
    log_path: _logPath,
    cwd: _cwd,
    last_checked_at: _lastCheckedAt,
    last_exit_code: _lastExitCode,
    last_error: _lastError,
    next_check_at: _nextCheckAt,
    consecutive_error_count: _consecutiveErrorCount,
    group_id: _groupId,
    group_label: _groupLabel,
    ...shared
  } = external;
  return {
    ...shared,
    condition: fields.condition ?? { node_id: "blk/upstream", status_in: ["resolved"] },
    armed_revision: fields.armed_revision ?? 1,
    last_evaluated_at: fields.last_evaluated_at ?? "2026-08-06T03:30:00Z",
  };
}

function render(run, props = {}) {
  const withWatcherItems = {
    ...run,
    watcherItems: run.watcherItems ?? experimentWatcherDisplayItems(run.watchers),
  };
  return renderToStaticMarkup(
    React.createElement(ExperimentRunDetail, {
      run: withWatcherItems,
      runBusy: false,
      runDisabled: false,
      stopBusy: false,
      recoveryBusy: false,
      watcherCheckBusyId: null,
      onRun() {},
      onStopLoop() {},
      onRecover() {},
      onSwitchProvider() {},
      onCheckWatcher() {},
      episodeReportHref: (episodeId) => `/reports/${episodeId}`,
      ...props,
    }),
  );
}

function assertDetailProjection(html, healthLabel, recommendationLabel) {
  const healthViews = [
    ...html.matchAll(/<div class="experiment-run-health[^"]*"[^>]*>(.*?)<\/div>/gs),
  ];
  assert.equal(healthViews.length, 1);
  assert.equal(healthViews[0][1], `<strong>${healthLabel}</strong>`);

  const recommendationViews = [
    ...html.matchAll(/<div class="experiment-run-recommendation[^"]*">(.*?)<\/div>/gs),
  ];
  assert.equal(recommendationViews.length, 1);
  assert.equal((html.match(/Recommended next step/g) ?? []).length, 1);
  assert.match(recommendationViews[0][1], /<span class="eyebrow">Recommended next step<\/span>/);
  assert.match(recommendationViews[0][1], new RegExp(`<strong>${recommendationLabel}</strong>`));
}

function recoveryTask(fields = {}) {
  return withTaskAnswers({
    operation_id: "wake-failed",
    project_id: "project",
    kind: "node_chat",
    status: "failed",
    request: {
      provider: "codex",
      model: "gpt-5.6",
      reasoning: "high",
      run_on: "cluster",
      patch_kind: "experiment_loop",
      control_node_id: "experiment/detail",
      control_episode_id: "episode-1",
    },
    created_at: "2026-08-06T04:00:00Z",
    updated_at: "2026-08-06T04:01:00Z",
    status_message: "Agent turn failed",
    error: "You've hit your session limit · resets 9:20am (UTC)",
    attempt: 1,
    estimate_seconds: 60,
    estimate_samples: 1,
    phase: "failed",
    elapsed_seconds: 1,
    progress: 0,
    can_pause: false,
    can_resume: false,
    can_retry: true,
    ...fields,
  });
}

function episode(fields = {}) {
  return {
    episode_id: "episode-1",
    project_id: "project",
    mode: "experiment_loop",
    control_node_id: "experiment/detail",
    root_operation_id: "turn-root",
    current_operation_id: null,
    current_orchestrator_task_id: null,
    current_control_task_id: null,
    recovery: null,
    status: "wrapping_up",
    starting_instruction: null,
    budget: {
      invocation_ceiling: 3,
      invocations_used: 3,
      invocations_remaining: 0,
      observed_input_tokens: 10,
      observed_generated_tokens: 20,
    },
    authorized_by: null,
    stop_requested_at: null,
    ending: "exhausted",
    ending_diagnostic: null,
    wrapup_state: "running",
    wrapup_error: null,
    created_at: "2026-08-06T01:00:00Z",
    updated_at: "2026-08-06T04:00:00Z",
    ended_at: null,
    tasks: [],
    report: null,
    can_stop: false,
    can_reauthorize: false,
    can_message: false,
    live: true,
    health: "wrapping_up",
    recommendation: "wait",
    task_control: null,
    run_section: "needs_action",
    ...fields,
  };
}

test("Experiment wrap-up uses the shared parent state without report recovery controls", () => {
  const run = buildExperimentRun(
    node(),
    control(
      {
        episode: episode(),
        ready: false,
        reasons: ["A previous episode is still open on this Experiment."],
        health: "wrapping_up",
        recommendation: "wait",
        run_section: "running",
        live: false,
        can_start: false,
      },
      { episode_live: true },
    ),
    [recoveryTask({ can_retry: true, can_resume: true })],
    [],
  );
  const html = render(run);

  assertDetailProjection(
    html,
    "Wrapping up visualization and report",
    "Wrapping up visualization and report",
  );
  assert.doesNotMatch(html, /Retry codex|Resume codex|Open report/);
  // Start is refused because the published gate says so, not because this card
  // recognised the parent status itself.
  assert.match(html, /experiment-run-button" disabled=""/);
});

test("a ready Experiment report opens from the singular episode URL", () => {
  const readyEpisode = episode({
    status: "needs_action",
    ending: "human_pause",
    wrapup_state: "ready",
    ended_at: "2026-08-06T04:00:00Z",
    report: {
      report_id: "report-hidden-from-url",
      ending: "human_pause",
      created_at: "2026-08-06T04:00:00Z",
    },
    can_reauthorize: true,
  });
  const html = render(
    buildExperimentRun(
      node(),
      control({
        episode: readyEpisode,
        health: "needs_action",
        recommendation: "open_report",
        can_open_report: true,
        report_episode_id: readyEpisode.episode_id,
      }),
      [],
      [],
    ),
  );

  assert.match(html, /href="\/reports\/episode-1"/);
  assert.match(html, /Open report/);
  assert.doesNotMatch(html, /report-hidden-from-url|Retry codex|Resume codex/);
  assert.doesNotMatch(html, /experiment-run-button" disabled=""/);
});

test("an Experiment the human closed stays completed whatever its last episode did", () => {
  // Regression: deriving health from the ending alone put every Experiment whose
  // last episode paused for a human decision back into Needs action, including
  // ones the human had already marked done.
  const paused = episode({
    status: "needs_action",
    ending: "human_pause",
    wrapup_state: "legacy_unavailable",
  });
  const html = render(
    buildExperimentRun(
      { ...node(), status: "completed" },
      control({
        episode: paused,
        ready: false,
        reasons: ["This Experiment is completed. Edit its status before starting a new episode."],
        health: "completed",
        recommendation: "none",
        run_section: "completed",
        can_start: false,
        can_open_report: true,
        report_episode_id: "previous-completed-episode",
        node_closed: true,
      }),
      [],
      [],
    ),
  );

  assertDetailProjection(html, "Completed", "Experiment is completed");
  assert.match(html, /href="\/reports\/previous-completed-episode"/);
  assert.match(html, /Open report/);
  assert.doesNotMatch(html, /Start (?:new )?episode/);
  assert.doesNotMatch(html, /Run requirements|Edit its status before starting a new episode/);
});

test("an open Experiment whose episode paused for a human still needs action", () => {
  const paused = episode({
    status: "needs_action",
    ending: "human_pause",
    wrapup_state: "legacy_unavailable",
  });
  const html = render(
    buildExperimentRun(
      { ...node(), status: "running" },
      control({ episode: paused, recommendation: "none" }),
      [],
      [],
    ),
  );

  assertDetailProjection(html, "Needs action", "Episode report unavailable");
});

test("a final Experiment report error is a note beside the episode's own outcome", () => {
  const reportFailed = episode({
    status: "needs_action",
    wrapup_state: "failed",
    wrapup_error: "The visual report could not be generated.",
    can_reauthorize: true,
  });
  const html = render(
    buildExperimentRun(
      node(),
      control({
        episode: reportFailed,
        health: "paused_at_limit",
        recommendation: "start_episode",
      }),
      [recoveryTask({ can_retry: true, can_resume: true })],
      [],
    ),
  );

  // The episode exhausted its invocations; the missing report never restates that
  // as the episode's own health or as the human's next step.
  assertDetailProjection(html, "Paused at invocation limit", "Start a new episode");
  assert.match(html, /Report generation error: The visual report could not be generated\./);
  assert.doesNotMatch(
    html,
    /Retry codex|Resume codex|Stop loop|experiment-run-button" disabled=""/,
  );
});

test("the reason an Experiment episode ended outranks its report error", () => {
  const failedBeforeSession = episode({
    status: "failed",
    ending: "failed",
    ending_diagnostic: "This Experiment turn failed before it started its agent session.",
    wrapup_state: "not_started",
    live: false,
    health: "failed",
    recommendation: "review",
  });
  const html = render(
    buildExperimentRun(
      node(),
      control({
        episode: failedBeforeSession,
        health: "failed",
        recommendation: "none",
        run_section: "completed",
      }),
      [recoveryTask({ can_retry: true, can_resume: true })],
      [],
    ),
  );

  assertDetailProjection(html, "Failed", "Episode ended");
  assert.match(html, /This Experiment turn failed before it started its agent session\./);
  assert.doesNotMatch(html, /Report generation error|Stop loop/);
  assert.doesNotMatch(html, /experiment-run-button" disabled=""/);
});

test("a stopped Experiment shows neither a report nor a report error", () => {
  // A stopped episode arrives with its diagnostic and report error already
  // withheld by the projection, so there is nothing here for the view to hide.
  const stopped = episode({
    status: "stopped",
    ending: "stopped",
    ending_diagnostic: null,
    wrapup_state: "skipped",
    wrapup_error: null,
    live: false,
    health: "stopped",
    recommendation: "none",
  });
  const html = render(
    buildExperimentRun(
      node(),
      control({
        episode: stopped,
        health: "human_stopped",
        recommendation: "start_episode",
      }),
      [],
      [],
    ),
  );

  assert.doesNotMatch(html, /Open report|Report generation error|hidden stop/);
});

test("a closed Experiment offers no episode start until its status is edited", () => {
  const html = render({
    node: node({ status: "completed", invocation_ceiling: 7 }),
    control: control(
      {
        ready: false,
        reasons: ["This Experiment is completed. Edit its status before starting a new episode."],
        invocations_used: 0,
        invocation_ceiling: 7,
        invocations_remaining: 7,
        episode_id: null,
        paused: false,
        health: "completed",
        recommendation: "none",
        run_section: "completed",
        can_start: false,
        node_closed: true,
      },
      { current_invocation: null },
    ),
    taskGroup: null,
    currentTask: null,
    watchers: [],
    currentWatchers: [],
    health: "completed",
  });

  assertDetailProjection(html, "Completed", "Experiment is completed");
  assert.doesNotMatch(html, /Start (?:new )?episode/);
  assert.doesNotMatch(html, /Run requirements|Edit its status before starting a new episode/);
  assert.doesNotMatch(html, /Next episode limit/);
});

test("prior invocation totals stay pinned beside the edited next episode limit", () => {
  const html = render({
    node: node({ status: "planned", invocation_ceiling: 7 }),
    control: control(
      {
        invocations_used: 4,
        invocation_ceiling: 5,
        invocations_remaining: 1,
        paused: false,
      },
      { current_invocation: null },
    ),
    taskGroup: null,
    currentTask: null,
    watchers: [],
    currentWatchers: [],
    health: "completed",
  });

  assert.match(html, /Invocation<\/span>4\s*\/ 5/);
  assert.match(html, /Next episode limit<\/span>7/);
  assert.match(html, /<button[^>]*>.*Start new episode<\/button>/s);
  assert.doesNotMatch(html, />Start episode<\/button>/);
});

test("provider-limited Experiment recovery stays on the loop detail", () => {
  const task = recoveryTask();
  const diagnostic =
    "Retry Codex to recheck availability, or switch provider to continue this episode.";
  const html = render({
    node: node(),
    control: control(
      {
        can_start: false,
        can_stop: true,
        task_control: "retry",
        can_switch_provider: true,
        recommendation: "retry",
      },
      {
        current_operation_id: task.operation_id,
        session: { ...operational().session, diagnostic },
      },
    ),
    taskGroup: { rootId: task.operation_id, root: task, latest: task, attempts: [task] },
    currentTask: task,
    watchers: [],
    currentWatchers: [],
    health: "needs_action",
  });

  assert.match(html, /Retry Codex/);
  assert.match(html, /Switch provider…/);
  assert.match(html, /Stop loop/);
  assert.doesNotMatch(html, /Open agent task/);
  assert.match(html, /Retry Codex to recheck availability/);
  assert.ok(html.indexOf("Retry Codex to recheck availability") < html.indexOf("Last task error"));
  assert.match(html, /Last task error/);
  assert.match(html, /You&#x27;ve hit your session limit/);
});

test("recovery controls stay hidden when their exact backend task is absent", () => {
  const html = render({
    node: node(),
    control: control(
      {
        can_start: false,
        can_stop: true,
        task_control: "retry",
        can_switch_provider: true,
        recommendation: "retry",
      },
      { current_operation_id: "missing-task", current_status: "failed" },
    ),
    taskGroup: null,
    currentTask: null,
    watchers: [],
    currentWatchers: [],
    health: "needs_action",
  });

  assert.doesNotMatch(html, /Retry Codex|Switch provider…/);
  assert.match(html, /Stop loop/);
});

test("detail follows the exact backend operation even when a newer retry row exists", () => {
  const failed = recoveryTask({ operation_id: "failed-attempt" });
  const retry = recoveryTask({
    operation_id: "running-retry",
    parent_operation_id: failed.operation_id,
    status: "running",
    attempt: 2,
    created_at: "2026-08-06T04:02:00Z",
    updated_at: "2026-08-06T04:02:00Z",
    can_retry: false,
    error: null,
  });
  const run = buildExperimentRun(
    node(),
    control(
      {
        ready: false,
        reasons: ["An experiment loop is already active."],
        health: "agent_active",
        recommendation: "wait",
        run_section: "running",
        live: true,
        can_start: false,
        can_stop: true,
      },
      {
        task_active: true,
        current_operation_id: failed.operation_id,
        current_status: "failed",
      },
    ),
    [failed, retry],
    [],
  );
  const html = render(run);

  assertDetailProjection(html, "Agent active", "Wait for the agent");
  assert.match(
    html,
    /Current task<\/dt><dd class="mono experiment-run-breakable">failed-attempt<\/dd>/,
  );
  assert.doesNotMatch(html, /Retry Codex|Switch provider…/);
});

test("staged graph changes disable Start until Sync", () => {
  const html = render(
    {
      node: node({ status: "planned" }),
      control: control(
        {
          episode_id: null,
          invocations_used: 0,
          invocations_remaining: 3,
          recommendation: "start_episode",
          can_start: true,
        },
        { current_invocation: null },
      ),
      taskGroup: null,
      currentTask: null,
      watchers: [],
      currentWatchers: [],
      health: "needs_action",
    },
    { startDisabled: true },
  );

  assertDetailProjection(html, "Needs action", "Sync staged changes before starting");
  assert.match(html, /experiment-run-button" disabled=""/);
});

test("a paused Experiment offers native-session resume and disables recovery while busy", () => {
  const task = recoveryTask({ status: "paused", can_resume: true });
  const html = render(
    {
      node: node(),
      control: control(
        {
          can_start: false,
          can_stop: true,
          task_control: "resume",
          can_switch_provider: true,
          recommendation: "resume",
        },
        { current_operation_id: task.operation_id },
      ),
      taskGroup: { rootId: task.operation_id, root: task, latest: task, attempts: [task] },
      currentTask: task,
      watchers: [],
      currentWatchers: [],
      health: "needs_action",
    },
    { recoveryBusy: true },
  );

  assert.match(html, /Resuming Codex…/);
  assert.match(html, /<button[^>]*disabled=""[^>]*>Resuming Codex…<\/button>/);
  assert.match(html, /Switch provider…/);
});

test("an unsettled stop enables exact paused recovery and hides the requested Stop action", () => {
  const task = recoveryTask({ status: "paused", can_resume: true, can_retry: true });
  const experimentControl = control(
    {
      ready: false,
      reasons: ["A graceful stop is finishing the current loop turn."],
      can_start: false,
      stop_pending: true,
      task_control: "resume",
      can_switch_provider: true,
      recommendation: "resume",
    },
    {
      task_active: true,
      stop_requested: true,
      stop_settled: false,
      current_operation_id: task.operation_id,
      current_status: "paused",
    },
  );
  const run = buildExperimentRun(node(), experimentControl, [task], []);
  const detail = render(run);

  assertDetailProjection(detail, "Needs action", "Resume this episode, or switch provider");
  assert.match(
    detail,
    /<button type="button" class="button primary compact experiment-recovery-button" aria-busy="false">Resume Codex<\/button>/,
  );
  assert.match(detail, /<button type="button" class="button compact">Switch provider…<\/button>/);
  assert.doesNotMatch(detail, /experiment-stop-loop|Stopping gracefully/);
  assert.match(detail, /experiment-run-button" disabled=""/);

  const row = renderToStaticMarkup(
    React.createElement(ExecutionView, {
      graph: { nodes: { [run.node.id]: run.node } },
      episodes: [
        episode({
          status: "running",
          ending: null,
          wrapup_state: "not_started",
          tasks: [task],
          current_control_task_id: task.operation_id,
          can_stop: true,
          live: true,
          health: "needs_action",
          recommendation: "resume",
          task_control: "resume",
        }),
      ],
      episodeMessages: {},
      episodeAction: null,
      tasks: [task],
      watchers: [],
      experimentControl: { [run.node.id]: experimentControl },
      selectedExperimentId: null,
      focusExperimentId: null,
      runBusy: false,
      stopBusyId: null,
      watcherCheckBusyId: null,
      taskActionId: null,
      onInspectTask() {},
      onSelectExperiment() {},
      onDetailFocused() {},
      onRunExperiment() {},
      onStopExperiment() {},
      onCheckExperimentWatcher() {},
      onRecoverExperiment() {},
      onSwitchExperimentProvider() {},
    }),
  );
  const detailRecommendation = detail.match(
    /<div class="experiment-run-recommendation[^"]*">.*?<strong>([^<]+)<\/strong><\/div>/s,
  )?.[1];
  assert.equal(detailRecommendation, "Resume this episode, or switch provider");
  assert.doesNotMatch(row, /campaign-run-summary|Resume the current turn/);
});

test("a running episode with nothing left to wake it points at Stop loop", () => {
  // Regression: the loop's turn succeeded without arming a watcher or exiting, so
  // the episode row stayed open. Admission refuses a second live parent, but the
  // control projection did not say so, so Runs recommended "Start a new episode"
  // and disabled that button while the Research node panel offered it. The reason
  // now arrives from the server, and Stop loop is the control that frees it.
  const stranded = episode({
    status: "running",
    ending: null,
    wrapup_state: "not_started",
    budget: {
      invocation_ceiling: 10,
      invocations_used: 1,
      invocations_remaining: 9,
      observed_input_tokens: 10,
      observed_generated_tokens: 20,
    },
  });
  const html = render(
    buildExperimentRun(
      node(),
      control(
        {
          episode: stranded,
          ready: false,
          reasons: ["A previous episode is still open on this Experiment."],
          invocations_used: 1,
          invocation_ceiling: 10,
          invocations_remaining: 9,
          paused: false,
          health: "needs_action",
          recommendation: "stop_and_restart",
          live: true,
          can_start: false,
          can_stop: true,
        },
        { current_status: "succeeded", current_invocation: 1, episode_live: true },
      ),
      [],
      [],
    ),
  );

  assertDetailProjection(html, "Needs action", "Stop loop, then start a new episode");
  assert.match(html, /<button[^>]*disabled=""[^>]*>.*Start new episode<\/button>/s);
  assert.match(html, /Stop loop/);
  // The reason is the server's sentence, not one this card composed.
  assert.match(html, /A previous episode is still open on this Experiment\./);
});

test("completed watcher at the ceiling leaves Start new episode enabled", () => {
  const completed = watcher();
  const html = render({
    node: node(),
    control: control(
      {
        health: "paused_at_limit",
        recommendation: "start_episode",
        live: true,
        can_stop: true,
      },
      { watcher_completion_pending: true },
    ),
    taskGroup: null,
    currentTask: null,
    watchers: [completed],
    currentWatchers: [completed],
    health: "paused_at_limit",
  });

  assertDetailProjection(html, "Paused at invocation limit", "Start a new episode");
  assert.match(html, /Start new episode/);
  assert.doesNotMatch(html, /<button[^>]*disabled=""[^>]*>.*Start new episode<\/button>/s);
  assert.match(html, /Stop loop/);
  assert.match(html, /Watchers<\/span><span class="experiment-fold-count">1<\/span>/);
  assert.doesNotMatch(html, />Pause<|>Resume<|>Retry<|Stop watching/);
});

test("a gated human-stopped loop recommends its available requirement action", () => {
  const reason = "Blocker blk/required-input is open.";
  const html = render({
    node: node(),
    control: control(
      {
        ready: false,
        reasons: [reason],
        graph_reasons: [reason],
        health: "human_stopped",
        recommendation: "resolve_requirements",
        can_start: false,
      },
      { stop_requested: true, stop_settled: true },
    ),
    taskGroup: null,
    currentTask: null,
    watchers: [],
    currentWatchers: [],
    health: "human_stopped",
  });

  assertDetailProjection(html, "Human-stopped", "Resolve the run requirements");
  assert.match(html, new RegExp(reason.replaceAll(".", "\\.")));
  assert.match(html, /<button[^>]*disabled=""[^>]*>.*Start new episode<\/button>/s);
});

test("detail keeps Experiment meaning under a neutral Research summary", () => {
  const html = render({
    node: node({ status: "debugging" }),
    control: control(
      { invocations_used: 1, invocations_remaining: 2 },
      { current_phase: "provider output" },
    ),
    taskGroup: null,
    currentTask: null,
    watchers: [],
    currentWatchers: [],
    health: "needs_action",
  });

  assert.match(html, /<h4>Research summary<\/h4>/);
  assert.match(html, /The detached evaluation is still running/);
  assert.match(html, /Inspect the held-out evaluation/);
  assertDetailProjection(html, "Needs action", "Start a new episode");
  assert.doesNotMatch(html, /Experiment state|Debugging|>Phase</);
});

test("degraded external watcher exposes backoff and Check now without recommending stop", () => {
  const degraded = watcher({
    status: "degraded",
    completed_at: null,
    last_exit_code: null,
    last_error: "ssh connection timed out",
    next_check_at: "2026-08-06T04:30:00Z",
    consecutive_error_count: 3,
  });
  const html = render({
    node: node({ status: "debugging" }),
    control: control(
      {
        ready: false,
        invocations_used: 1,
        invocations_remaining: 2,
        paused: false,
        health: "degraded",
        recommendation: "keep_loop",
        run_section: "running",
        live: true,
        can_start: false,
        can_stop: true,
      },
      { detached_work_active: true, watcher_degraded: true },
    ),
    taskGroup: null,
    currentTask: null,
    watchers: [degraded],
    currentWatchers: [degraded],
    health: "degraded",
  });

  assert.match(html, /Keep loop running; check now if needed/);
  assert.match(html, /Next check/);
  assert.match(html, /Consecutive failures/);
  assert.match(html, />3<\/dd>/);
  assert.match(html, />Check now<\/button>/);
  assertDetailProjection(html, "Watcher degraded", "Keep loop running; check now if needed");
});

test("watcher Check now renders busy and disables concurrent mutations", () => {
  const degraded = watcher({ status: "degraded", completed_at: null });
  const html = render(
    {
      node: node(),
      control: control(
        {
          ready: false,
          invocations_used: 1,
          invocations_remaining: 2,
          health: "degraded",
          recommendation: "keep_loop",
          run_section: "running",
          live: true,
          can_start: false,
          can_stop: true,
        },
        { detached_work_active: true, watcher_degraded: true },
      ),
      taskGroup: null,
      currentTask: null,
      watchers: [degraded],
      currentWatchers: [degraded],
      health: "degraded",
    },
    { watcherCheckBusyId: degraded.watcher_id, runDisabled: true },
  );

  assert.match(html, /<button[^>]*disabled=""[^>]*aria-busy="true"[^>]*>Checking…<\/button>/);
  assert.match(html, /experiment-stop-loop" disabled=""/);
  assert.match(html, /experiment-run-button" disabled=""/);
});

test("missing episode continuity recommends stop then start without parsing diagnostic text", () => {
  const task = recoveryTask({ can_retry: false, can_resume: false });
  const html = render({
    node: node(),
    control: control(
      {
        ready: false,
        invocations_used: 1,
        invocations_remaining: 2,
        recommendation: "stop_and_restart",
        can_start: false,
        can_stop: true,
      },
      {
        current_operation_id: task.operation_id,
        session: {
          ...operational().session,
          diagnostic: "The saved continuation no longer exists.",
        },
      },
    ),
    taskGroup: { rootId: task.operation_id, root: task, latest: task, attempts: [task] },
    currentTask: task,
    watchers: [],
    currentWatchers: [],
    health: "needs_action",
  });

  assertDetailProjection(html, "Needs action", "Stop loop, then start a new episode");
  assert.match(
    html,
    /<button type="button" class="button compact experiment-stop-loop">Stop loop<\/button>/,
  );
  assert.doesNotMatch(html, /Agent turn failed|>Phase</);
});

test("an unavailable Stop is neither shown nor recommended", () => {
  const task = recoveryTask({ can_retry: false, can_resume: false });
  const html = render({
    node: node(),
    control: control({ episode_id: null, ready: true }),
    taskGroup: { rootId: task.operation_id, root: task, latest: task, attempts: [task] },
    currentTask: task,
    watchers: [],
    currentWatchers: [],
    health: "needs_action",
  });

  assertDetailProjection(html, "Needs action", "Start an episode");
  assert.match(html, /<button[^>]*>.*Start episode<\/button>/s);
  assert.doesNotMatch(html, /experiment-stop-loop|Stop loop, then start a new episode/);
});

test("a succeeded legacy-attribution episode offers a fresh start without an unusable Stop", () => {
  const task = recoveryTask({
    operation_id: "legacy-attribution-succeeded",
    status: "succeeded",
    status_message: "OBSOLETE SUCCEEDED TASK STATUS",
    phase: "complete",
    can_retry: false,
    can_resume: false,
  });
  const diagnostic =
    "Automatic watcher wake stopped: an originating task predates durable human attribution, so RCP cannot prove who authorized the wake. Start a new Work turn or Experiment Run to continue.";
  const run = {
    node: node(),
    control: control(
      { ready: true, reasons: [], invocations_used: 1, invocations_remaining: 2 },
      {
        current_operation_id: task.operation_id,
        current_status: "succeeded",
        current_status_message: task.status_message,
        current_phase: task.phase,
        session: { ...operational().session, diagnostic },
      },
    ),
    taskGroup: { rootId: task.operation_id, root: task, latest: task, attempts: [task] },
    currentTask: task,
    watchers: [],
    currentWatchers: [],
    health: "needs_action",
  };

  const detail = render(run);
  assertDetailProjection(detail, "Needs action", "Start a new episode");
  assert.match(detail, /<button[^>]*>.*Start new episode<\/button>/s);
  assert.doesNotMatch(
    detail,
    /experiment-stop-loop|Stop loop, then start a new episode|OBSOLETE SUCCEEDED TASK STATUS|>Phase</,
  );

  const row = renderToStaticMarkup(
    React.createElement(ExecutionView, {
      graph: { nodes: { [run.node.id]: run.node } },
      episodes: [
        episode({
          status: "failed",
          ending: "failed",
          wrapup_state: "failed",
          tasks: [task],
          live: false,
          health: "failed",
          recommendation: "review",
          run_section: "needs_action",
        }),
      ],
      episodeMessages: {},
      episodeAction: null,
      tasks: [task],
      watchers: [],
      experimentControl: { [run.node.id]: run.control },
      selectedExperimentId: null,
      focusExperimentId: null,
      runBusy: false,
      stopBusyId: null,
      watcherCheckBusyId: null,
      taskActionId: null,
      onInspectTask() {},
      onSelectExperiment() {},
      onDetailFocused() {},
      onRunExperiment() {},
      onStopExperiment() {},
      onCheckExperimentWatcher() {},
      onRecoverExperiment() {},
      onSwitchExperimentProvider() {},
    }),
  );
  const detailRecommendation = detail.match(
    /<div class="experiment-run-recommendation[^"]*">.*?<strong>([^<]+)<\/strong><\/div>/s,
  )?.[1];
  assert.equal(detailRecommendation, "Start a new episode");
  assert.doesNotMatch(row, /campaign-run-summary|Review the episode failure/);
  assert.doesNotMatch(row, /OBSOLETE SUCCEEDED TASK STATUS/);
});

test("graph watcher detail shows its canonical condition without shell fields", () => {
  const graph = graphWatcher({
    watcher_id: "graph-watcher",
    status: "active",
    completed_at: null,
  });
  const html = render({
    node: node(),
    control: control({ invocations_used: 1, invocations_remaining: 2, paused: false }),
    taskGroup: null,
    currentTask: null,
    watchers: [graph],
    currentWatchers: [graph],
    health: "waiting_on_watchers",
  });

  assert.match(html, /blk\/upstream reaches resolved/);
  assert.match(html, /Graph condition/);
  assert.match(html, /Last evaluation/);
  assert.match(html, /graph-watcher/);
  assert.doesNotMatch(html, /Check command|Working directory|evaluation\.log/);
});

test("detail separates watcher provenance from semantic meaning and execution binding", () => {
  const stopped = watcher({
    status: "stopped",
    notified: true,
    completed_at: null,
    last_exit_code: null,
  });
  const html = render({
    node: node({ status: "planned" }),
    control: control(
      { invocations_used: 1, invocations_remaining: 2, paused: false },
      {
        stop_requested: true,
        stop_settled: true,
        current_operation_id: "authoritative-current-task",
      },
    ),
    taskGroup: null,
    currentTask: null,
    watchers: [stopped],
    currentWatchers: [stopped],
    health: "human_stopped",
  });

  assert.match(html, /role="status" aria-live="polite"/);
  assert.match(html, /Human-stopped/);
  assert.match(html, /Stopped · not delivered/);
  assert.match(html, /origin-turn/);
  assert.match(html, /episode episode-1 · invocation 3 \/ 3/);
  assert.match(html, /squeue -h -o/);
  assert.match(html, /evaluation\.log/);
  assert.match(html, /Working directory/);
  assert.match(html, /codex/);
  assert.match(html, /gpt-5\.6/);
  assert.match(html, /high/);
  assert.match(html, /cluster · login\.research/);
  assert.match(html, /repo-a, repo-b/);
  assert.match(html, /Bound/);
  assert.match(html, /authoritative-current-task/);
  assert.match(html, /Completion criteria/);
  assert.match(html, /All held-out evaluations have mechanically exited/);
  assert.match(html, /Semantic attempts/);
  assert.match(html, /Run the held-out evaluation/);
  assert.match(html, /slurm-48192/);
  assert.match(html, /Governing decisions/);
  assert.match(html, /decision\/resources/);
  assert.match(html, /Decision drift/);
  assert.match(html, /decision\/data moved to v2/);
  assert.doesNotMatch(html, />Pause<|>Resume<|>Retry<|Stop watching/);
});

test("stopped watcher history is collapsed separately from current watchers", () => {
  const active = watcher({ watcher_id: "watcher-active", status: "active" });
  const stopped = watcher({
    watcher_id: "watcher-stopped",
    status: "stopped",
    notified: true,
  });
  const html = render({
    node: node(),
    control: control(),
    taskGroup: null,
    currentTask: null,
    watchers: [active, stopped],
    currentWatchers: [active],
    health: "waiting_on_watchers",
  });

  assert.match(html, /Watchers<\/span><span class="experiment-fold-count">1<\/span>/);
  assert.match(
    html,
    /<details class="experiment-fold nested"><summary><span class="experiment-fold-title">Stopped watchers<\/span><span class="experiment-fold-count">1<\/span>/,
  );
  assert.doesNotMatch(html, /<details class="experiment-fold nested" open/);
  assert.match(html, /aria-label="Stopped experiment watchers"/);
});

test("a queued notification claim is not presented as proven provider delivery", () => {
  const claimed = watcher({
    notified: true,
    notification_operation_id: "wake-task",
  });
  const html = render({
    node: node(),
    control: control(),
    taskGroup: null,
    currentTask: null,
    watchers: [claimed],
    currentWatchers: [claimed],
    health: "needs_action",
  });

  assert.match(html, /Delivery claimed/);
  assert.doesNotMatch(html, />Delivered</);
});

test("grouped watchers show truthful operational counts and preserve member provenance", () => {
  const watchers = [
    watcher({
      watcher_id: "shard-finished",
      group_id: "group-eval-shards",
      group_label: "eval-shards",
    }),
    watcher({
      watcher_id: "shard-degraded",
      status: "degraded",
      completed_at: null,
      last_exit_code: null,
      last_error: "ssh connection timed out",
      consecutive_error_count: 5,
      group_id: "group-eval-shards",
      group_label: "eval-shards",
    }),
    watcher({
      watcher_id: "shard-running",
      status: "active",
      completed_at: null,
      last_exit_code: 1,
      group_id: "group-eval-shards",
      group_label: "eval-shards",
    }),
    watcher({
      watcher_id: "shard-retired",
      status: "stopped",
      completed_at: null,
      last_exit_code: null,
      notified: true,
      group_id: "group-eval-shards",
      group_label: "eval-shards",
      stopped_by: "agent",
      stop_reason: "Cancelled superseded external job",
      stopped_at: "2026-08-06T05:00:00Z",
      stop_operation_id: "wake-turn",
    }),
    watcher({
      watcher_id: "ungrouped-history",
      group_id: null,
      group_label: null,
      log_path: "/scratch/ungrouped.log",
    }),
  ];
  const html = render({
    node: node(),
    control: control(),
    taskGroup: null,
    currentTask: null,
    watchers,
    currentWatchers: watchers,
    health: "degraded",
  });

  assert.match(html, /<details>/);
  assert.equal((html.match(/Watcher group/g) ?? []).length, 1);
  assert.match(html, /eval-shards/);
  assert.match(html, /1 finished · 1 degraded · 1 running · 1 stopped/);
  assert.match(html, /group-eval-shards/);
  assert.match(html, /shard-finished/);
  assert.match(html, /shard-degraded/);
  assert.match(html, /shard-running/);
  assert.match(html, /shard-retired/);
  assert.match(html, /Current error/);
  assert.match(html, /ssh connection timed out/);
  assert.match(html, /Origin invocation/);
  assert.match(html, /evaluation\.log/);
  assert.match(html, /ungrouped-history/);
  assert.match(html, /ungrouped\.log/);
  assert.match(html, /Agent stopped/);
  assert.match(html, /Agent reason/);
  assert.match(html, /Cancelled superseded external job/);
  assert.doesNotMatch(html, /Stop watching/);
});
