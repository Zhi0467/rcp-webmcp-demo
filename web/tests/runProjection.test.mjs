import assert from "node:assert/strict";
import { withExperimentControlAnswers, withTaskAnswers } from "./taskAnswers.mjs";
import test from "node:test";

import {
  buildExperimentRun,
  buildRunProjection,
  buildRunTaskProjection,
  experimentRecommendation,
  experimentWatcherDisplayItems,
  graphConditionLabel,
  isGraphWatcherRecord,
  visibleChatWatchers,
  watcherIsActive,
  watcherIsIndividuallyStoppable,
  watcherLastObservedAt,
} from "../src/runProjection.ts";

function task(
  operationId,
  status,
  createdAt,
  parentOperationId = null,
  { kind = "refresh", request = {} } = {},
) {
  return withTaskAnswers({
    operation_id: operationId,
    project_id: "project",
    kind,
    status,
    request,
    created_at: createdAt,
    updated_at: createdAt,
    status_message: `${operationId} ${status}`,
    attempt: 1,
    parent_operation_id: parentOperationId,
    phase: "agent",
    last_activity_at: createdAt,
  });
}

function loopTask(operationId, nodeId, episodeId, status, createdAt, parentOperationId = null) {
  return task(operationId, status, createdAt, parentOperationId, {
    kind: "node_chat",
    request: {
      patch_kind: "experiment_loop",
      control_node_id: nodeId,
      control_episode_id: episodeId,
      control_invocation: 1,
    },
  });
}

function experiment(id, status = "planned", fields = {}) {
  return {
    id,
    type: "experiment",
    title: `Experiment ${id}`,
    extension_fields: {},
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    status,
    objective: "Measure the bounded loop.",
    attempts: [],
    invocation_ceiling: 3,
    completion_criteria: [],
    ...fields,
  };
}

function control(fields = {}, operationalFields = {}) {
  return withExperimentControlAnswers({
    ready: true,
    reasons: [],
    graph_reasons: [],
    invocations_used: 0,
    invocation_ceiling: 3,
    invocations_remaining: 3,
    episode_id: null,
    paused: false,
    active: false,
    governing_decisions: [],
    decision_drift: [],
    operational: {
      task_active: false,
      detached_work_active: false,
      watcher_degraded: false,
      watcher_completion_pending: false,
      episode_exited: false,
      episode_live: false,
      stop_requested: false,
      stop_settled: false,
      chat_id: null,
      current_operation_id: null,
      current_status: null,
      current_phase: null,
      current_status_message: null,
      current_last_activity_at: null,
      current_invocation: null,
      session: {
        provider: null,
        model: null,
        reasoning: null,
        run_on: null,
        execution_host: null,
        run_truth_scope: null,
        native_session_bound: false,
        diagnostic: null,
      },
      ...operationalFields,
    },
    ...fields,
  });
}

function watcher(id, nodeId, episodeId, status, fields = {}) {
  return {
    watcher_id: id,
    project_id: "project",
    origin_operation_id: `origin-${id}`,
    origin_task_kind: "node_chat",
    chat_id: "chat",
    node_id: nodeId,
    execution_host: "",
    check_command: "true",
    log_path: `/tmp/${id}.log`,
    cwd: "/tmp",
    continuation: {
      provider: "codex",
      model: null,
      reasoning: null,
      run_on: "local",
      run_truth_scope: null,
      patch_kind: "experiment_loop",
      control_node_id: nodeId,
      control_revision: 1,
      control_episode_id: episodeId,
      control_invocation: 1,
      control_invocation_ceiling: 3,
      control_decision_bundle: [],
      control_completion_criteria: [],
      workflow_ids: [],
      skill_ids: [],
      invoked_workflow_ids: [],
      invoked_skill_ids: [],
      resolved_skill_packages: [],
    },
    status,
    created_at: "2026-08-06T01:00:00Z",
    last_checked_at: null,
    last_exit_code: null,
    last_error: null,
    completed_at: null,
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

function graphWatcher(id, nodeId, episodeId, status, condition, fields = {}) {
  const external = watcher(id, nodeId, episodeId, status, fields);
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
    condition,
    armed_revision: fields.armed_revision ?? 1,
    last_evaluated_at: fields.last_evaluated_at ?? null,
  };
}

function byId(projection) {
  return new Map([
    ...projection.running.map((entry) => [entry.id, ["running", entry]]),
    ...projection.actionable.map((entry) => [entry.id, ["actionable", entry]]),
    ...projection.completed.map((entry) => [entry.id, ["completed", entry]]),
  ]);
}

test("task retries group under their logical root and classify by the latest attempt", () => {
  const projection = buildRunTaskProjection([
    task("root", "failed", "2026-07-28T00:00:00Z"),
    task("retry-1", "failed", "2026-07-28T01:00:00Z", "root"),
    task("retry-2", "running", "2026-07-28T02:00:00Z", "retry-1"),
    task("paused", "paused", "2026-07-28T03:00:00Z"),
    task("done", "succeeded", "2026-07-28T04:00:00Z"),
  ]);

  assert.deepEqual(
    projection.running[0].attempts.map((item) => item.operation_id),
    ["root", "retry-1", "retry-2"],
  );
  assert.equal(projection.running[0].latest.operation_id, "retry-2");
  assert.deepEqual(
    projection.actionable.map((group) => group.rootId),
    ["paused"],
  );
  assert.deepEqual(
    projection.completed.map((group) => group.rootId),
    ["done"],
  );
});

test("dismissed and superseded ingestion failures leave the action queue", () => {
  const failed = task("failed", "failed", "2026-07-28T00:00:00Z");
  const laterSuccess = task("later", "succeeded", "2026-07-28T01:00:00Z");
  const dismissed = task("dismissed", "failed", "2026-07-28T02:00:00Z");
  const projection = buildRunTaskProjection(
    [laterSuccess, dismissed, failed],
    new Set(["dismissed"]),
  );
  assert.deepEqual(projection.actionable, []);
  assert.deepEqual(
    projection.completed.map((group) => group.rootId),
    ["later"],
  );
});

test("Runs includes Experiment-loop tasks once and excludes generic chat and coach tasks", () => {
  const node = experiment("experiment/include");
  const loop = loopTask("loop-task", node.id, "episode-current", "running", "2026-08-06T03:00:00Z");
  const projection = buildRunProjection({
    nodes: [node],
    tasks: [
      loop,
      task("generic-chat", "running", "2026-08-06T04:00:00Z", null, {
        kind: "node_chat",
        request: { patch_kind: "work" },
      }),
      task("coach", "failed", "2026-08-06T05:00:00Z", null, {
        kind: "paper_coach",
      }),
      task("refresh", "running", "2026-08-06T02:00:00Z"),
    ],
    experimentControl: {
      [node.id]: control(
        {
          episode_id: "episode-current",
          invocations_used: 1,
          invocations_remaining: 2,
          health: "agent_active",
          recommendation: "wait",
          run_section: "running",
          live: true,
          can_start: false,
          can_stop: true,
        },
        {
          task_active: true,
          current_operation_id: loop.operation_id,
          current_status: "running",
        },
      ),
    },
    actionableBlockerIds: new Set(),
  });

  assert.deepEqual(
    projection.running.map((entry) => [entry.kind, entry.id]),
    [
      ["experiment", node.id],
      ["task", "refresh"],
    ],
  );
  assert.equal(projection.actionable.length, 0);
});

test("backend operation identity selects the exact task even when a newer row exists", () => {
  const node = experiment("experiment/retry-active");
  const episodeId = "episode-retry-active";
  const failed = {
    ...loopTask("failed-attempt", node.id, episodeId, "failed", "2026-08-14T09:50:00Z"),
    can_retry: true,
  };
  const retry = {
    ...loopTask(
      "running-retry",
      node.id,
      episodeId,
      "running",
      "2026-08-14T09:51:00Z",
      failed.operation_id,
    ),
    attempt: 2,
  };
  const staleControl = control(
    {
      ready: false,
      reasons: ["An experiment loop is already active."],
      episode_id: episodeId,
      invocations_used: 3,
      invocations_remaining: 7,
      health: "needs_action",
      recommendation: "review",
      run_section: "actionable",
      can_start: false,
    },
    {
      task_active: true,
      current_operation_id: failed.operation_id,
      current_status: "failed",
      current_invocation: 3,
    },
  );

  const run = buildExperimentRun(node, staleControl, [failed, retry], []);
  assert.equal(run.currentTask.operation_id, failed.operation_id);
  assert.equal(run.health, "needs_action");
  assert.deepEqual(experimentRecommendation(run), {
    step: "review",
    label: "Review the loop state",
  });

  const entry = byId(
    buildRunProjection({
      nodes: [node],
      tasks: [failed, retry],
      experimentControl: { [node.id]: staleControl },
      actionableBlockerIds: new Set(),
    }),
  ).get(node.id);
  assert.equal(entry[0], "actionable");
  assert.equal(entry[1].experiment.currentTask.operation_id, failed.operation_id);
});

test("Experiment watcher projection keeps each immutable group and ungrouped history distinct", () => {
  const nodeId = "experiment/grouped";
  const episodeId = "episode-grouped";
  const run = buildExperimentRun(
    experiment(nodeId),
    control({ episode_id: episodeId }),
    [],
    [
      watcher("ungrouped", nodeId, episodeId, "completed"),
      watcher("shard-finished", nodeId, episodeId, "completed", {
        group_id: "group-eval-shards",
        group_label: "eval-shards",
      }),
      watcher("shard-degraded", nodeId, episodeId, "degraded", {
        group_id: "group-eval-shards",
        group_label: "eval-shards",
      }),
      watcher("shard-running", nodeId, episodeId, "active", {
        group_id: "group-eval-shards",
        group_label: "eval-shards",
      }),
      watcher("shard-retired", nodeId, episodeId, "stopped", {
        group_id: "group-eval-shards",
        group_label: "eval-shards",
        stopped_by: "agent",
      }),
    ],
  );

  const grouped = run.watcherItems.find((item) => item.kind === "group");
  assert.ok(grouped && grouped.kind === "group");
  assert.equal(grouped.group.groupId, "group-eval-shards");
  assert.equal(grouped.group.label, "eval-shards");
  assert.deepEqual(grouped.group.counts, {
    finished: 1,
    degraded: 1,
    running: 1,
    stopped: 1,
  });
  assert.deepEqual(
    grouped.group.watchers.map((item) => item.watcher_id),
    ["shard-degraded", "shard-finished", "shard-retired", "shard-running"],
  );
  assert.deepEqual(
    run.watcherItems
      .filter((item) => item.kind === "watcher")
      .map((item) => item.watcher.watcher_id),
    ["ungrouped"],
  );
});

test("graph watchers stay ungrouped and expose condition labels and evaluation time", () => {
  const status = graphWatcher(
    "graph-status",
    "experiment/grouped",
    "episode-grouped",
    "active",
    { node_id: "blk/upstream", status_in: ["resolved", "superseded"] },
    { last_evaluated_at: "2026-08-06T03:00:00Z" },
  );
  const proposal = graphWatcher(
    "graph-proposal",
    "experiment/grouped",
    "episode-grouped",
    "active",
    { node_id: "hyp/result", proposal_resolved: true },
  );

  assert.equal(isGraphWatcherRecord(status), true);
  assert.equal(
    graphConditionLabel(status.condition),
    "blk/upstream reaches resolved or superseded",
  );
  assert.equal(graphConditionLabel(proposal.condition), "Proposal on hyp/result is resolved");
  assert.equal(watcherLastObservedAt(status), "2026-08-06T03:00:00Z");
  assert.deepEqual(
    visibleChatWatchers([status], "new-chat", experiment("experiment/grouped")).map(
      (watcher) => watcher.watcher_id,
    ),
    ["graph-status"],
  );
  assert.deepEqual(
    experimentWatcherDisplayItems([status, proposal]).map((item) => item.kind),
    ["watcher", "watcher"],
  );
});

test("Chats project node-owned loop watchers separately from conversation self-wake watchers", () => {
  const node = experiment("experiment/shared-loop");
  const episodeId = "episode-shared-loop";
  const loopActive = watcher("loop-active", node.id, episodeId, "active", {
    chat_id: "creator-chat",
  });
  const loopDegraded = watcher("loop-degraded", node.id, episodeId, "degraded", {
    chat_id: "other-creator-chat",
  });
  const loopStopped = watcher("loop-stopped", node.id, episodeId, "stopped", {
    chat_id: "creator-chat",
  });
  const loopCompleted = watcher("loop-completed", node.id, episodeId, "completed", {
    chat_id: "creator-chat",
  });
  const otherNodeLoop = watcher("other-node-loop", "experiment/other", "episode-other", "active", {
    chat_id: "creator-chat",
  });
  const selfWake = watcher("self-wake", null, null, "active", {
    chat_id: "maintenance-chat",
    continuation: {
      ...loopActive.continuation,
      patch_kind: "work",
      control_node_id: null,
      control_episode_id: null,
    },
  });
  const otherChatSelfWake = watcher("other-chat-self-wake", null, null, "active", {
    chat_id: "other-chat",
    continuation: {
      ...loopActive.continuation,
      patch_kind: "work",
      control_node_id: null,
      control_episode_id: null,
    },
  });
  const stoppedSelfWake = watcher("stopped-self-wake", null, null, "stopped", {
    chat_id: "maintenance-chat",
    continuation: selfWake.continuation,
  });
  const watchers = [
    loopActive,
    loopDegraded,
    loopStopped,
    loopCompleted,
    otherNodeLoop,
    selfWake,
    otherChatSelfWake,
    stoppedSelfWake,
  ];

  const sameNodeChat = visibleChatWatchers(watchers, "maintenance-chat", node);
  assert.deepEqual(
    sameNodeChat.map((item) => item.watcher_id),
    ["loop-active", "loop-degraded", "self-wake"],
  );
  assert.deepEqual(
    visibleChatWatchers([...watchers, loopActive, selfWake], "maintenance-chat", node).map(
      (item) => item.watcher_id,
    ),
    ["loop-active", "loop-degraded", "self-wake"],
  );
  const run = buildExperimentRun(node, control({ episode_id: episodeId }), [], watchers);
  assert.equal(
    sameNodeChat.filter((item) => item.continuation.patch_kind === "experiment_loop").length,
    run.watchers.filter(watcherIsActive).length,
  );

  assert.deepEqual(
    visibleChatWatchers(watchers, "maintenance-chat", null).map((item) => item.watcher_id),
    ["self-wake"],
  );
  assert.deepEqual(
    visibleChatWatchers(watchers, "maintenance-chat", experiment("experiment/unrelated")).map(
      (item) => item.watcher_id,
    ),
    ["self-wake"],
  );
  assert.deepEqual(
    visibleChatWatchers(watchers, "project-chat", null).map((item) => item.watcher_id),
    [],
  );
});

test("Experiment projection places the backend-published health and section", () => {
  const nodes = [
    experiment("terminal-live", "completed"),
    experiment("stopping-live"),
    experiment("stopping-failed"),
    experiment("healthy-wait"),
    experiment("degraded-wait"),
    experiment("ceiling-pending"),
    experiment("graph-gated"),
    experiment("session-unavailable"),
    experiment("human-stopped"),
    experiment("terminal", "superseded"),
  ];
  const controls = {
    "terminal-live": control(
      {
        episode_id: "ep-terminal-live",
        invocations_used: 1,
        invocations_remaining: 2,
        health: "agent_active",
        recommendation: "wait",
        run_section: "running",
        live: true,
      },
      { task_active: true, current_operation_id: "terminal-live-task", current_status: "running" },
    ),
    "stopping-live": control(
      {
        ready: false,
        reasons: ["A graceful stop is finishing the current loop turn."],
        episode_id: "ep-stopping-live",
        invocations_used: 1,
        invocations_remaining: 2,
        health: "stopping",
        recommendation: "wait",
        run_section: "running",
        live: true,
        stop_pending: true,
      },
      {
        task_active: true,
        stop_requested: true,
        current_operation_id: "stopping-live-task",
        current_status: "pausing",
      },
    ),
    "stopping-failed": control(
      {
        ready: false,
        reasons: ["A graceful stop is finishing the current loop turn."],
        episode_id: "ep-stopping-failed",
        invocations_used: 1,
        invocations_remaining: 2,
        health: "stopping",
        recommendation: "wait",
        run_section: "actionable",
        live: true,
        stop_pending: true,
      },
      {
        task_active: true,
        stop_requested: true,
        current_operation_id: "stopping-failed-task",
        current_status: "failed",
      },
    ),
    "healthy-wait": control(
      {
        ready: false,
        reasons: ["An experiment loop is already active."],
        episode_id: "ep-healthy",
        invocations_used: 1,
        invocations_remaining: 2,
        active: true,
        health: "waiting_on_watchers",
        recommendation: "wait",
        run_section: "running",
        live: true,
      },
      { detached_work_active: true },
    ),
    "degraded-wait": control(
      {
        ready: false,
        reasons: ["An experiment loop is already active."],
        episode_id: "ep-degraded",
        invocations_used: 1,
        invocations_remaining: 2,
        active: true,
        health: "degraded",
        recommendation: "keep_loop",
        run_section: "running",
        live: true,
      },
      { detached_work_active: true },
    ),
    "ceiling-pending": control(
      {
        episode_id: "ep-ceiling",
        invocations_used: 3,
        invocations_remaining: 0,
        paused: true,
        health: "paused_at_limit",
        recommendation: "start_episode",
        run_section: "actionable",
      },
      { watcher_completion_pending: true },
    ),
    "graph-gated": control(
      {
        ready: false,
        reasons: ["Blocker blocker/upstream is open."],
        graph_reasons: ["Blocker blocker/upstream is open."],
        episode_id: "ep-gated",
        invocations_used: 1,
        invocations_remaining: 2,
        health: "needs_action",
        recommendation: "resolve_requirements",
        run_section: "actionable",
        can_start: false,
      },
      { detached_work_active: true },
    ),
    "session-unavailable": control(
      {
        episode_id: "ep-session",
        invocations_used: 1,
        invocations_remaining: 2,
        health: "needs_action",
        recommendation: "review",
        run_section: "actionable",
      },
      {
        watcher_completion_pending: true,
        session: {
          ...control().operational.session,
          diagnostic: "The bound native session is unavailable.",
        },
      },
    ),
    "human-stopped": control(
      {
        episode_id: "ep-stopped",
        invocations_used: 1,
        invocations_remaining: 2,
        health: "human_stopped",
        recommendation: "start_episode",
        run_section: "actionable",
      },
      { stop_requested: true, stop_settled: true },
    ),
    terminal: control({
      health: "completed",
      recommendation: "none",
      run_section: "completed",
    }),
  };
  const tasks = [
    loopTask(
      "terminal-live-task",
      "terminal-live",
      "ep-terminal-live",
      "running",
      "2026-08-06T01:00:00Z",
    ),
    loopTask(
      "stopping-live-task",
      "stopping-live",
      "ep-stopping-live",
      "pausing",
      "2026-08-06T02:00:00Z",
    ),
    loopTask(
      "stopping-failed-task",
      "stopping-failed",
      "ep-stopping-failed",
      "failed",
      "2026-08-06T03:00:00Z",
    ),
  ];
  const watchers = [
    watcher("healthy", "healthy-wait", "ep-healthy", "active"),
    watcher("degraded", "degraded-wait", "ep-degraded", "degraded"),
    watcher("ceiling", "ceiling-pending", "ep-ceiling", "completed"),
    watcher("gated", "graph-gated", "ep-gated", "active"),
    watcher("session", "session-unavailable", "ep-session", "completed"),
  ];
  const entries = byId(
    buildRunProjection({
      nodes,
      tasks,
      watchers,
      experimentControl: controls,
      actionableBlockerIds: new Set(),
    }),
  );

  assert.deepEqual(
    entries.get("terminal-live").map((value, index) => (index ? value.experiment.health : value)),
    ["running", "agent_active"],
  );
  assert.deepEqual(
    entries.get("stopping-live").map((value, index) => (index ? value.experiment.health : value)),
    ["running", "stopping"],
  );
  assert.deepEqual(experimentRecommendation(entries.get("stopping-live")[1].experiment), {
    step: "wait",
    label: "Wait for the current turn to finish",
  });
  assert.deepEqual(
    entries.get("stopping-failed").map((value, index) => (index ? value.experiment.health : value)),
    ["actionable", "stopping"],
  );
  assert.deepEqual(
    entries.get("healthy-wait").map((value, index) => (index ? value.experiment.health : value)),
    ["running", "waiting_on_watchers"],
  );
  assert.deepEqual(
    entries.get("degraded-wait").map((value, index) => (index ? value.experiment.health : value)),
    ["running", "degraded"],
  );
  assert.deepEqual(
    entries.get("ceiling-pending").map((value, index) => (index ? value.experiment.health : value)),
    ["actionable", "paused_at_limit"],
  );
  assert.equal(entries.get("graph-gated")[0], "actionable");
  assert.equal(entries.get("session-unavailable")[0], "actionable");
  assert.deepEqual(
    entries.get("human-stopped").map((value, index) => (index ? value.experiment.health : value)),
    ["actionable", "human_stopped"],
  );
  assert.deepEqual(
    entries.get("terminal").map((value, index) => (index ? value.experiment.health : value)),
    ["completed", "completed"],
  );
});

test("an unsettled Experiment stop exposes exact paused recovery before graceful waiting", () => {
  const node = experiment("experiment/stopping-paused");
  const paused = {
    ...loopTask(
      "stopping-paused-task",
      node.id,
      "episode-stopping-paused",
      "paused",
      "2026-08-06T03:00:00Z",
    ),
    can_resume: true,
    can_retry: true,
  };
  const experimentControl = control(
    {
      ready: false,
      reasons: ["A graceful stop is finishing the current loop turn."],
      episode_id: "episode-stopping-paused",
      invocations_used: 1,
      invocations_remaining: 2,
      health: "needs_action",
      recommendation: "resume",
      run_section: "actionable",
      live: true,
      can_start: false,
      can_stop: false,
      stop_pending: true,
      task_control: "resume",
      can_switch_provider: true,
    },
    {
      task_active: true,
      stop_requested: true,
      stop_settled: false,
      current_operation_id: paused.operation_id,
      current_status: "paused",
    },
  );

  const run = buildExperimentRun(node, experimentControl, [paused], []);
  assert.equal(run.health, "needs_action");
  assert.deepEqual(experimentRecommendation(run), {
    step: "resume",
    label: "Resume this episode, or switch provider",
  });

  const entry = byId(
    buildRunProjection({
      nodes: [node],
      tasks: [paused],
      experimentControl: { [node.id]: experimentControl },
      actionableBlockerIds: new Set(),
    }),
  ).get(node.id);
  assert.equal(entry[0], "actionable");
  assert.equal(entry[1].experiment.health, "needs_action");
});

test("historical watchers stay visible without driving current health or task selection", () => {
  const node = experiment("experiment/history");
  const current = loopTask(
    "current-task",
    node.id,
    "episode-current",
    "succeeded",
    "2026-08-06T01:00:00Z",
  );
  const newerHistory = loopTask(
    "historical-task",
    node.id,
    "episode-history",
    "failed",
    "2026-08-06T05:00:00Z",
  );
  const run = buildExperimentRun(
    node,
    control(
      { episode_id: "episode-current", invocations_used: 1, invocations_remaining: 2 },
      { current_operation_id: current.operation_id, current_status: "succeeded" },
    ),
    [newerHistory, current],
    [watcher("old-degraded", node.id, "episode-history", "degraded")],
  );

  assert.equal(run.watchers.length, 1);
  assert.equal(run.currentWatchers.length, 0);
  assert.equal(run.currentTask.operation_id, "current-task");
  assert.equal(run.health, "needs_action");
});

test("a succeeded legacy-attribution episode recommends a fresh episode directly", () => {
  const node = experiment("experiment/legacy-attribution");
  const succeeded = loopTask(
    "legacy-attribution-task",
    node.id,
    "episode-legacy-attribution",
    "succeeded",
    "2026-08-06T01:00:00Z",
  );
  const run = buildExperimentRun(
    node,
    control(
      {
        ready: true,
        reasons: [],
        episode_id: "episode-legacy-attribution",
        invocations_used: 1,
        invocations_remaining: 2,
      },
      {
        current_operation_id: succeeded.operation_id,
        current_status: "succeeded",
        session: {
          ...control().operational.session,
          diagnostic:
            "Automatic watcher wake stopped: an originating task predates durable human attribution, so RCP cannot prove who authorized the wake. Start a new Work turn or Experiment Run to continue.",
        },
      },
    ),
    [succeeded],
    [],
  );

  assert.equal(run.currentTask.status, "succeeded");
  assert.deepEqual(run.currentWatchers, []);
  assert.equal(run.health, "needs_action");
  assert.deepEqual(experimentRecommendation(run), {
    step: "start_episode",
    label: "Start a new episode",
  });
});

test("compatible adopted degraded watchers drive current health through control state", () => {
  const node = experiment("experiment/adopted-degraded");
  const current = loopTask(
    "current-task",
    node.id,
    "episode-current",
    "succeeded",
    "2026-08-06T01:00:00Z",
  );
  const run = buildExperimentRun(
    node,
    control(
      {
        ready: false,
        reasons: ["An experiment loop is already active."],
        episode_id: "episode-current",
        invocations_used: 1,
        invocations_remaining: 2,
        health: "degraded",
        recommendation: "keep_loop",
        run_section: "running",
        live: true,
        can_start: false,
      },
      {
        current_operation_id: current.operation_id,
        current_status: "succeeded",
        detached_work_active: true,
        watcher_degraded: true,
      },
    ),
    [current],
    [watcher("adopted-degraded", node.id, "episode-older", "degraded")],
  );

  assert.equal(run.currentWatchers.length, 0);
  assert.equal(run.health, "degraded");
  assert.deepEqual(experimentRecommendation(run), {
    step: "keep_loop",
    label: "Keep loop running; check now if needed",
  });
});

test("Experiment recommendation copy follows the backend recommendation enum", () => {
  const base = {
    node: experiment("experiment/recommendation"),
    control: control({
      episode_id: "episode-recommendation",
      health: "agent_active",
      recommendation: "wait",
      run_section: "running",
    }),
    taskGroup: null,
    currentTask: null,
    watchers: [],
    watcherItems: [],
    currentWatchers: [],
    health: "agent_active",
  };
  assert.equal(experimentRecommendation(base).step, "wait");
  assert.equal(
    experimentRecommendation({
      ...base,
      control: control({
        episode_id: "episode-recommendation",
        recommendation: "stop_and_restart",
      }),
    }).step,
    "stop_and_restart",
  );
  assert.equal(
    experimentRecommendation({
      ...base,
      control: control({ recommendation: "start_episode" }),
      health: "human_stopped",
    }).step,
    "start_episode",
  );
  assert.equal(
    experimentRecommendation({
      ...base,
      control: control({
        ready: false,
        reasons: ["Blocker blk/required-input is open."],
        graph_reasons: ["Blocker blk/required-input is open."],
        episode_id: "episode-recommendation",
        recommendation: "resolve_requirements",
      }),
      health: "human_stopped",
    }).step,
    "resolve_requirements",
  );
});

test("entries remain newest first within each section", () => {
  const projection = buildRunProjection({
    nodes: [],
    tasks: [
      task("older", "running", "2026-08-06T01:00:00Z"),
      task("newer", "running", "2026-08-06T02:00:00Z"),
    ],
    experimentControl: {},
    actionableBlockerIds: new Set(),
  });
  assert.deepEqual(
    projection.running.map((entry) => entry.id),
    ["newer", "older"],
  );
});

test("only a generic watcher can be stopped on its own", () => {
  assert.equal(watcherIsIndividuallyStoppable({ continuation: { patch_kind: "work" } }), true);
  assert.equal(
    watcherIsIndividuallyStoppable({ continuation: { patch_kind: "experiment_loop" } }),
    false,
  );
  assert.equal(watcherIsIndividuallyStoppable({}), true);
});
