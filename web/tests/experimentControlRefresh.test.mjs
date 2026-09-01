import assert from "node:assert/strict";
import { withTaskAnswers } from "./taskAnswers.mjs";
import { after, test } from "node:test";
import { createServer } from "vite";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const {
  activeBranchMergeTask,
  experimentControlsNeedWrapupPolling,
  failedTaskActionNeedsAuthoritativeProjectReload,
  loadExperimentWatcherPoll,
  terminalTaskNeedsAuthoritativeProjectReload,
  terminalTasksSince,
} = await server.ssrLoadModule("/src/App.tsx");
const { reconcileKnownActiveTasks } = await server.ssrLoadModule("/src/hooks/useAgentTasks.ts");

after(() => server.close());

test("terminal Experiment work refetches control state even without a graph revision", () => {
  const completedLoop = {
    status: "succeeded",
    applied_revision: null,
    request: {
      patch_kind: "experiment_loop",
      control_node_id: "experiment/demo",
    },
  };
  const pausedLoop = {
    ...completedLoop,
    status: "paused",
  };
  const ordinaryChat = {
    kind: "project_chat",
    status: "succeeded",
    applied_revision: null,
    request: { patch_kind: "work" },
  };
  const branchMerge = {
    ...ordinaryChat,
    kind: "branch_merge",
  };

  assert.equal(terminalTaskNeedsAuthoritativeProjectReload(completedLoop), true);
  assert.equal(terminalTaskNeedsAuthoritativeProjectReload(pausedLoop), true);
  assert.equal(terminalTaskNeedsAuthoritativeProjectReload(ordinaryChat), false);
  assert.equal(terminalTaskNeedsAuthoritativeProjectReload(branchMerge), true);
  assert.equal(
    terminalTaskNeedsAuthoritativeProjectReload({ ...ordinaryChat, applied_revision: 7 }),
    true,
  );
});

test("hidden Experiment report wrap-up keeps authoritative polling alive", () => {
  assert.equal(
    experimentControlsNeedWrapupPolling({
      "experiment/demo": { health: "wrapping_up" },
    }),
    true,
  );
  assert.equal(
    experimentControlsNeedWrapupPolling({
      "experiment/demo": { health: "completed" },
      "experiment/other": { health: "needs_action" },
    }),
    false,
  );
});

test("poll reconciliation observes every task that terminalized, not only the selected active task", () => {
  const previous = [
    withTaskAnswers({ operation_id: "newer", kind: "auto_research", status: "running" }),
    withTaskAnswers({ operation_id: "merge", kind: "branch_merge", status: "running" }),
    withTaskAnswers({ operation_id: "chat", kind: "project_chat", status: "succeeded" }),
  ];
  const current = [
    withTaskAnswers({ ...previous[0], status: "running" }),
    withTaskAnswers({ ...previous[1], status: "succeeded" }),
    previous[2],
  ];

  assert.deepEqual(
    terminalTasksSince(previous, current).map((task) => task.operation_id),
    ["merge"],
  );
  assert.equal(
    terminalTasksSince(previous, current).some(terminalTaskNeedsAuthoritativeProjectReload),
    true,
  );
});

test("a fast merge completion remains observable while another task keeps polling active", () => {
  const main = withTaskAnswers({ operation_id: "main", kind: "auto_research", status: "running" });
  const merge = withTaskAnswers({ operation_id: "merge", kind: "branch_merge", status: "queued" });
  const responseEpisode = {
    graph_branch: { active_merge_task_id: merge.operation_id },
    tasks: [merge],
  };
  const knownActive = new Map();

  assert.deepEqual(reconcileKnownActiveTasks(knownActive, [main]), []);
  const started = activeBranchMergeTask(responseEpisode);
  assert.equal(started, merge);
  knownActive.set(started.operation_id, started);
  // A concurrent project reload may still return the pre-commit graph here.
  assert.deepEqual(reconcileKnownActiveTasks(knownActive, [main, merge]), []);
  // Its task request may then observe the terminal merge and must force another reload.
  const terminal = reconcileKnownActiveTasks(knownActive, [
    main,
    withTaskAnswers({ ...merge, status: "succeeded" }),
  ]);

  assert.deepEqual(
    terminal.map((task) => task.operation_id),
    ["merge"],
  );
  assert.equal(terminal.some(terminalTaskNeedsAuthoritativeProjectReload), true);
});

test("failed Experiment Resume and Retry refetch authoritative stop state", () => {
  const experimentLoop = {
    request: { patch_kind: "experiment_loop" },
  };
  const ordinaryWork = {
    request: { patch_kind: "work" },
  };

  assert.equal(failedTaskActionNeedsAuthoritativeProjectReload(experimentLoop, "resume"), true);
  assert.equal(failedTaskActionNeedsAuthoritativeProjectReload(experimentLoop, "retry"), true);
  assert.equal(failedTaskActionNeedsAuthoritativeProjectReload(experimentLoop, "pause"), false);
  assert.equal(failedTaskActionNeedsAuthoritativeProjectReload(ordinaryWork, "retry"), false);
});

test("pending Experiment watcher polling always refreshes control state", async () => {
  const watcher = {
    watcher_id: "watcher-1",
    status: "active",
    notified: false,
    notification_operation_id: null,
    continuation: { patch_kind: "experiment_loop" },
  };
  const requested = [];
  const base = "/api/projects/project-1";
  const result = await loadExperimentWatcherPoll(async (path) => {
    requested.push(path);
    if (path.endsWith("/watchers")) return [watcher];
    if (path.endsWith("/tasks")) return [];
    return { experiment_control: { "experiment/demo": { operational: {} } } };
  }, base);

  assert.deepEqual(requested, [`${base}/watchers`, `${base}/tasks`, base]);
  assert.deepEqual(result.watchers, [watcher]);
  assert.ok(result.project.experiment_control);
});
