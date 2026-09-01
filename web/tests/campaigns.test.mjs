import assert from "node:assert/strict";
import { after, test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

import {
  loadEpisodeMessages,
  loadEpisodes,
  loadExperimentEpisodes,
  mergeEpisodeToMain,
  reauthorizeEpisode,
  sendEpisodeMessage,
  startEpisode,
  stopEpisode,
} from "../src/api.ts";
import {
  episodeProjection,
  episodeReportPreviewUrl,
  episodeTaskRows,
  isLiveEpisode,
  mergeEpisode,
  runsEpisodeCards,
} from "../src/campaigns.ts";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { AutoResearchEpisodeCard } = await server.ssrLoadModule("/src/components/CampaignRuns.tsx");
const { AutoResearchDialog } = await server.ssrLoadModule("/src/components/AutoResearchDialog.tsx");

after(() => server.close());

const rootTask = {
  operation_id: "turn-root",
  project_id: "project one",
  kind: "auto_research",
  status: "running",
  request: { role: "orchestrator", actor_operation_id: "turn-root" },
  created_at: "2026-08-12T08:00:00Z",
  updated_at: "2026-08-12T08:00:00Z",
  status_message: "Reviewing the graph",
  attempt: 1,
  parent_operation_id: null,
  episode_id: "episode/alpha",
  estimate_seconds: 10,
  estimate_samples: 1,
  phase: "running",
  elapsed_seconds: 1,
  progress: 0.3,
  can_pause: true,
  can_resume: false,
  can_retry: false,
};

const episode = {
  episode_id: "episode/alpha",
  project_id: "project one",
  mode: "auto_research",
  control_node_id: null,
  graph_target: { kind: "main" },
  graph_base_head: null,
  graph_branch: null,
  root_operation_id: rootTask.operation_id,
  current_operation_id: rootTask.operation_id,
  current_orchestrator_task_id: rootTask.operation_id,
  current_control_task_id: rootTask.operation_id,
  recovery: null,
  status: "running",
  starting_instruction: "Begin with the unresolved **Blocker**.",
  budget: {
    invocation_ceiling: 8,
    invocations_used: 3,
    invocations_remaining: 5,
    observed_input_tokens: 12_345,
    observed_generated_tokens: 678,
  },
  authorized_by: { space_id: "space", user_id: "human", display_name: "Ada" },
  stop_requested_at: null,
  ending: null,
  ending_diagnostic: null,
  wrapup_state: "not_started",
  wrapup_error: null,
  created_at: "2026-08-12T08:00:00Z",
  updated_at: "2026-08-12T08:02:00Z",
  ended_at: null,
  tasks: [rootTask],
  report: null,
  can_stop: true,
  can_reauthorize: false,
  can_message: true,
  live: true,
  health: "active",
  recommendation: "continue",
  task_control: "pause",
};

test("the Auto-research dialog meters only operational invocations", () => {
  const html = renderToStaticMarkup(
    React.createElement(AutoResearchDialog, {
      open: true,
      busy: false,
      error: null,
      initialInvocationCeiling: 1,
      onClose() {},
      onAuthorize() {},
    }),
  );

  assert.match(html, /Operational invocation ceiling/);
  assert.match(html, /type="number" min="1"/);
  assert.doesNotMatch(html, /reserved for the report|Report invocation/);
  assert.doesNotMatch(html, /Start auto-research" disabled/);
});

function renderEpisodes(values, { busyAction = null } = {}) {
  return renderToStaticMarkup(
    React.createElement(
      "section",
      {},
      values.map((value, index) =>
        React.createElement(AutoResearchEpisodeCard, {
          episode: value,
          tasks: values.flatMap((episode) => episode.tasks),
          messages: [],
          initiallyExpanded: index === 0 || value.live,
          busyAction,
          taskActionId: null,
          onInspectTask() {},
          async onLoadMessages() {},
          async onStop() {},
          async onMerge() {},
          async onReauthorize() {},
          async onSendMessage() {},
          async onOperateTask() {},
          key: value.episode_id,
        }),
      ),
    ),
  );
}

test("the episode parent owns an operational-only invocation meter", () => {
  const html = renderEpisodes([episode]);

  assert.match(html, /Auto-research/);
  assert.match(html, /<time dateTime="2026-08-12T08:00:00Z">/);
  assert.doesNotMatch(html, /Episode ·|campaign-run-summary|Project episode/);
  assert.match(html, /3 \/ 8 invocations/);
  assert.match(html, /3 of 8 operational invocations used/);
  assert.doesNotMatch(html, /reserved|report unit|episode_report/i);
  assert.match(html, /12345|12,345/);
});

test("wrap-up has one exact parent state and no report task or recovery control", () => {
  const failedTask = {
    ...rootTask,
    status: "failed",
    status_message: "The operational turn ended",
    can_pause: false,
    can_retry: true,
  };
  const wrapping = {
    ...episode,
    status: "wrapping_up",
    wrapup_state: "running",
    tasks: [failedTask],
    health: "wrapping_up",
    recommendation: "wait",
    task_control: null,
  };
  const projection = episodeProjection(wrapping, wrapping.tasks);
  const html = renderEpisodes([wrapping]);

  assert.equal(projection.healthLabel, "Wrapping up visualization and report");
  assert.equal(projection.taskControl, null);
  assert.ok((html.match(/Wrapping up visualization and report/g) ?? []).length >= 2);
  assert.doesNotMatch(html, />Retry<|>Resume<|Report task|episode_report/);
});

test("a ready episode exposes one singular report URL", () => {
  const ready = {
    ...episode,
    status: "completed",
    live: false,
    ending: "completed",
    ended_at: "2026-08-12T08:04:00Z",
    wrapup_state: "ready",
    report: {
      report_id: "internal-report-id",
      ending: "completed",
      created_at: "2026-08-12T08:04:00Z",
    },
    can_stop: false,
    can_message: false,
    tasks: [{ ...rootTask, status: "succeeded", can_pause: false }],
    health: "completed",
    recommendation: "open_report",
    task_control: null,
  };
  const html = renderEpisodes([ready]);

  assert.match(html, /> Open report<|>Open report</);
  assert.match(
    html,
    /href="\/api\/projects\/project%20one\/episodes\/episode%2Falpha\/report\/viewer"/,
  );
  assert.doesNotMatch(html, /internal-report-id/);
  assert.equal(
    episodeReportPreviewUrl("project one", "episode/alpha"),
    "/api/projects/project%20one/episodes/episode%2Falpha/report/viewer",
  );
});

test("a final report error is visible, terminal, and has no task recovery control", () => {
  const failedTask = {
    ...rootTask,
    status: "failed",
    can_pause: false,
    can_resume: true,
    can_retry: true,
  };
  const reportFailed = {
    ...episode,
    status: "needs_action",
    live: false,
    ending: "exhausted",
    wrapup_state: "failed",
    wrapup_error: "The visual report could not be written.",
    tasks: [failedTask],
    can_stop: false,
    can_message: false,
    can_reauthorize: true,
    health: "needs_action",
    recommendation: "reauthorize",
    task_control: null,
  };
  const projection = episodeProjection(reportFailed, reportFailed.tasks);
  const html = renderEpisodes([reportFailed]);

  assert.equal(projection.taskControl, null);
  assert.match(html, /Report generation error: The visual report could not be written\./);
  assert.match(html, /New episode invocation ceiling/);
  assert.doesNotMatch(html, />Retry<|>Resume<|Open report/);
});

test("a report error does not downgrade a completed episode", () => {
  const reportFailed = {
    ...episode,
    status: "completed",
    live: false,
    ending: "completed",
    wrapup_state: "failed",
    wrapup_error: "The visual report could not be written.",
    tasks: [{ ...rootTask, status: "succeeded", can_pause: false }],
    can_stop: false,
    can_message: false,
    health: "completed",
    recommendation: "none",
    task_control: null,
  };
  const projection = episodeProjection(reportFailed, reportFailed.tasks);
  const html = renderEpisodes([reportFailed]);

  assert.equal(projection.health, "completed");
  assert.equal(projection.taskControl, null);
  assert.match(html, /Report generation error: The visual report could not be written\./);
  assert.doesNotMatch(html, />Retry<|>Resume<|Open report/);
});

test("Stop is the only ending that shows neither a report nor a report error", () => {
  const stopped = {
    ...episode,
    status: "stopped",
    live: false,
    ending: "stopped",
    ending_diagnostic: null,
    wrapup_state: "skipped",
    wrapup_error: null,
    report: null,
    can_stop: false,
    can_message: false,
    health: "stopped",
    recommendation: "none",
    task_control: null,
    tasks: [{ ...rootTask, status: "succeeded", can_pause: false }],
  };
  const html = renderEpisodes([stopped]);

  assert.match(html, /Stopped/);
  assert.doesNotMatch(html, /Open report|Report generation error|must stay hidden/);
});

test("reauthorization keeps the immutable old episode and inserts the fresh parent", () => {
  const oldEpisode = {
    ...episode,
    status: "needs_action",
    live: false,
    ending: "exhausted",
    wrapup_state: "ready",
    can_stop: false,
    can_reauthorize: true,
  };
  const freshEpisode = {
    ...episode,
    episode_id: "episode/fresh",
    root_operation_id: "fresh-root",
    current_operation_id: "fresh-root",
    created_at: "2026-08-12T09:00:00Z",
  };

  assert.deepEqual(
    mergeEpisode([oldEpisode], freshEpisode).map((item) => item.episode_id),
    ["episode/fresh", "episode/alpha"],
  );
  assert.equal(isLiveEpisode(oldEpisode), false);
  assert.equal(isLiveEpisode(freshEpisode), true);
});

test("Runs keeps only the backend-selected current episode for each Experiment node", () => {
  const olderExperiment = {
    ...episode,
    episode_id: "experiment/older",
    mode: "experiment_loop",
    control_node_id: "exp/shared",
    created_at: "2026-08-10T08:00:00Z",
  };
  const newerExperiment = {
    ...olderExperiment,
    episode_id: "experiment/newer",
    created_at: "2026-08-12T09:00:00Z",
  };
  const otherExperiment = {
    ...olderExperiment,
    episode_id: "experiment/other",
    control_node_id: "exp/other",
    created_at: "2026-08-11T08:00:00Z",
  };
  const autoResearch = { ...episode, episode_id: "auto/one" };

  assert.deepEqual(
    runsEpisodeCards(
      [olderExperiment, autoResearch, otherExperiment, newerExperiment],
      new Set([olderExperiment.episode_id, otherExperiment.episode_id]),
    ).map((item) => item.episode_id),
    ["auto/one", "experiment/other", "experiment/older"],
  );
});

const branchId = "8ba94d42-4d42-4ccb-9d2a-f299340dd3b8";
const baseHead = {
  target: { kind: "main" },
  revision: 4,
  transition_id: "transition-main-base-0004",
};
const branchHead = {
  target: { kind: "branch", branch_id: branchId },
  revision: 2,
  transition_id: "transition-branch-head-0002",
};

function withGraphBranch(overrides = {}) {
  return {
    ...episode,
    graph_target: { kind: "branch", branch_id: branchId },
    graph_base_head: baseHead,
    graph_branch: {
      branch_id: branchId,
      episode_id: episode.episode_id,
      base_head: baseHead,
      head: branchHead,
      merge_eligible: true,
      merge_state: "unmerged",
      latest_successful_merge: null,
      active_merge_task_id: null,
      merge_diagnostic: null,
      ...overrides,
    },
  };
}

test("an eligible episode shows its graph branch base, head, and merge action", () => {
  const html = renderEpisodes([withGraphBranch()]);

  assert.match(html, /Episode graph branch/);
  assert.match(html, /Graph branch/);
  assert.match(html, /8ba94d42\u20260dd3b8/);
  assert.match(html, /Base on main/);
  assert.match(html, />r4</);
  assert.match(html, /Branch head/);
  assert.match(html, />r2</);
  assert.match(html, /Unmerged/);
  assert.match(html, />Merge to main</);
});

test("an ineligible or running branch has no merge action, while another busy action disables it", () => {
  const running = renderEpisodes([
    withGraphBranch({
      merge_eligible: false,
      merge_state: "running",
      active_merge_task_id: "merge-task",
    }),
  ]);
  const disabled = renderEpisodes([withGraphBranch()], {
    busyAction: `stop:${episode.episode_id}`,
  });

  assert.match(running, /Merge running/);
  assert.doesNotMatch(running, />Merge to main</);
  assert.match(disabled, /<button[^>]+disabled=""[^>]*>.*Merge to main/s);
});

test("merged and failed branch summaries stay visible without branch-management controls", () => {
  const merged = renderEpisodes([
    withGraphBranch({
      merge_eligible: false,
      merge_state: "merged",
      latest_successful_merge: {
        schema_generation: 1,
        outcome: "committed",
        provenance: {
          schema_generation: 1,
          merge_id: "merge-1",
          branch_id: branchId,
          episode_id: episode.episode_id,
          branch_base_head: baseHead,
          branch_head: branchHead,
          rebased_main_head: { ...baseHead, revision: 10, transition_id: "main-before-0010" },
          merge_task_id: "merge-task",
        },
        result_main_head: { ...baseHead, revision: 11, transition_id: "main-after-0011" },
        authorized_by: { space_id: "space", user_id: "human", display_name: "Ada" },
        created_at: "2026-08-12T08:05:00Z",
      },
    }),
  ]);
  const failed = renderEpisodes([
    withGraphBranch({
      merge_state: "failed",
      merge_diagnostic: "The branch delta could not be rebased onto current main.",
    }),
  ]);

  assert.match(merged, />Merged</);
  assert.match(merged, /Merged on main/);
  assert.match(merged, />r11</);
  assert.doesNotMatch(merged, />Merge to main</);
  assert.doesNotMatch(merged, /discard|switch|conflict viewer/i);
  assert.match(failed, /Merge failed/);
  assert.match(failed, /The branch delta could not be rebased onto current main\./);
  assert.match(failed, />Merge to main</);
});

test("a paused or interrupted merge asks for action without presenting a failure", () => {
  const needsAction = renderEpisodes([
    withGraphBranch({
      merge_state: "needs_action",
      merge_diagnostic: "The merge was interrupted before it could finish.",
    }),
  ]);

  assert.match(needsAction, /campaign-graph-branch needs_action/);
  assert.match(needsAction, /Merge needs action/);
  assert.match(needsAction, /campaign-branch-diagnostic needs_action/);
  assert.match(needsAction, /The merge was interrupted before it could finish\./);
  assert.match(needsAction, />Merge to main</);
  assert.doesNotMatch(needsAction, /Merge failed|branch-failed|role="alert"/);
});

test("retries and continuations stay at their canonical actor depth", () => {
  const orchestratorRetryOne = {
    ...rootTask,
    operation_id: "turn-root-retry-1",
    parent_operation_id: rootTask.operation_id,
    created_at: "2026-08-12T08:01:00Z",
  };
  const orchestratorRetryTwo = {
    ...rootTask,
    operation_id: "turn-root-retry-2",
    parent_operation_id: orchestratorRetryOne.operation_id,
    created_at: "2026-08-12T08:02:00Z",
  };
  const worker = {
    ...rootTask,
    operation_id: "turn-worker",
    request: {
      role: "worker",
      actor_operation_id: "turn-worker",
      control_node_id: "experiment/demo",
    },
    parent_operation_id: orchestratorRetryTwo.operation_id,
    created_at: "2026-08-12T08:03:00Z",
  };
  const workerWake = {
    ...worker,
    operation_id: "turn-worker-wake",
    request: { ...worker.request, wake_cause: "message" },
    parent_operation_id: worker.operation_id,
    created_at: "2026-08-12T08:04:00Z",
  };
  const workerRetry = {
    ...worker,
    operation_id: "turn-worker-retry",
    parent_operation_id: workerWake.operation_id,
    created_at: "2026-08-12T08:05:00Z",
  };

  assert.deepEqual(
    episodeTaskRows(episode, [
      workerRetry,
      workerWake,
      worker,
      orchestratorRetryTwo,
      orchestratorRetryOne,
      rootTask,
    ]).map(({ task, role, depth }) => [task.operation_id, role, depth]),
    [
      ["turn-root", "orchestrator", 0],
      ["turn-root-retry-1", "orchestrator", 0],
      ["turn-root-retry-2", "orchestrator", 0],
      ["turn-worker", "worker", 1],
      ["turn-worker-wake", "wake", 1],
      ["turn-worker-retry", "worker", 1],
    ],
  );
});

test("episode API calls use only the generic endpoints and new-parent reauthorization body", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (path, init = {}) => {
    requests.push({ path, method: init.method ?? "GET", body: init.body ?? null });
    const payload = path.endsWith("/messages")
      ? init.method === "POST"
        ? { message_id: "m" }
        : []
      : [];
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    await loadEpisodes("/api/projects/demo", "auto_research");
    await startEpisode("/api/projects/demo", {
      mode: "auto_research",
      invocation_ceiling: 8,
      starting_instruction: "Start here",
    });
    await stopEpisode("/api/projects/demo", "episode/alpha");
    await reauthorizeEpisode("/api/projects/demo", "episode/alpha", 4);
    await mergeEpisodeToMain("/api/projects/demo", "episode/alpha");
    await loadEpisodeMessages("/api/projects/demo", "episode/alpha");
    await sendEpisodeMessage("/api/projects/demo", "episode/alpha", "Check the blocker");
    await loadExperimentEpisodes();
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.deepEqual(requests, [
    {
      path: "/api/projects/demo/episodes?mode=auto_research",
      method: "GET",
      body: null,
    },
    {
      path: "/api/projects/demo/episodes",
      method: "POST",
      body: JSON.stringify({
        mode: "auto_research",
        invocation_ceiling: 8,
        starting_instruction: "Start here",
      }),
    },
    {
      path: "/api/projects/demo/episodes/episode%2Falpha/stop",
      method: "POST",
      body: null,
    },
    {
      path: "/api/projects/demo/episodes/episode%2Falpha/reauthorize",
      method: "POST",
      body: JSON.stringify({ invocation_ceiling: 4 }),
    },
    {
      path: "/api/projects/demo/episodes/episode%2Falpha/merge",
      method: "POST",
      body: null,
    },
    {
      path: "/api/projects/demo/episodes/episode%2Falpha/messages",
      method: "GET",
      body: null,
    },
    {
      path: "/api/projects/demo/episodes/episode%2Falpha/messages",
      method: "POST",
      body: JSON.stringify({ body: "Check the blocker" }),
    },
    { path: "/api/episodes?mode=experiment_loop", method: "GET", body: null },
  ]);
  assert.equal(
    requests.some(({ path }) => path.includes("campaign")),
    false,
  );
  assert.equal(
    requests.some(({ path }) => path.includes("experiment-loops")),
    false,
  );
});
