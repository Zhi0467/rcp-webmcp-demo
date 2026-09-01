import assert from "node:assert/strict";
import { withExperimentControlAnswers, withTurnAnswers } from "./taskAnswers.mjs";
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
const {
  buildExperimentBoard,
  experimentBoardHref,
  experimentBoardRouteToken,
  experimentIndexEntryForRoute,
  experimentStopPath,
  experimentTerminalLabel,
  parseProjectHash,
  projectExperimentExecution,
  projectHashAfterViewChange,
} = await server.ssrLoadModule("/src/experimentBoard.ts");
const { ExperimentBoard } = await server.ssrLoadModule("/src/components/ExperimentBoard.tsx");
const { NodeChat } = await server.ssrLoadModule("/src/components/NodeChat.tsx");
const { ExecutionView } = await server.ssrLoadModule("/src/views/GraphViews.tsx");

after(() => server.close());

function node(id, status = "planned", updatedRev = 1) {
  return {
    id,
    type: "experiment",
    title: `Experiment ${id}`,
    extension_fields: {},
    standing: "asserted",
    created_rev: 1,
    updated_rev: updatedRev,
    source_refs: [],
    status,
    attempts: [],
  };
}

function control(fields = {}, operationalFields = {}) {
  return withExperimentControlAnswers({
    ready: true,
    reasons: [],
    invocations_used: 1,
    invocation_ceiling: 3,
    invocations_remaining: 2,
    episode_id: "episode-1",
    episode: null,
    paused: false,
    active: false,
    governing_decisions: [],
    decision_drift: [],
    operational: withTurnAnswers({
      task_active: false,
      detached_work_active: false,
      watcher_degraded: false,
      watcher_completion_pending: false,
      episode_exited: false,
      episode_live: false,
      stop_requested: false,
      stop_settled: false,
      chat_id: "chat-1",
      current_operation_id: "operation-1",
      current_status: null,
      current_phase: null,
      current_status_message: null,
      current_last_activity_at: null,
      current_invocation: 1,
      session: {
        provider: "codex",
        model: null,
        reasoning: null,
        run_on: "local",
        execution_host: "local",
        run_truth_scope: null,
        native_session_bound: true,
        diagnostic: null,
      },
      ...operationalFields,
    }),
    ...fields,
  });
}

function entry(id, nodeStatus, controlState, projectName = "Project") {
  return {
    project_id: `project-${projectName}`,
    project_name: projectName,
    project_reachable: true,
    graph_target: { kind: "main" },
    graph_head: null,
    parent_episode_id: null,
    node: node(id, nodeStatus),
    control: controlState,
    episode: controlState.episode,
  };
}

function episode(fields = {}) {
  return {
    episode_id: "episode-1",
    project_id: "project",
    mode: "experiment_loop",
    control_node_id: "wrapping",
    graph_target: { kind: "main" },
    graph_base_head: null,
    graph_branch: null,
    root_operation_id: "operation-1",
    current_operation_id: null,
    current_orchestrator_task_id: null,
    current_control_task_id: null,
    recovery: null,
    status: "wrapping_up",
    starting_instruction: null,
    budget: {
      invocation_ceiling: 3,
      invocations_used: 1,
      invocations_remaining: 2,
      observed_input_tokens: 0,
      observed_generated_tokens: 0,
    },
    authorized_by: null,
    stop_requested_at: null,
    ending: "completed",
    ending_diagnostic: null,
    wrapup_state: "running",
    wrapup_error: null,
    created_at: "2026-08-06T01:00:00Z",
    updated_at: "2026-08-06T02:00:00Z",
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

test("the board shows the shared report wrap-up as in-progress", () => {
  const wrapping = episode();
  const entryValue = entry(
    "wrapping",
    "active",
    control({
      episode: wrapping,
      health: "wrapping_up",
      recommendation: "wait",
      run_section: "running",
    }),
  );
  const board = buildExperimentBoard([entryValue]);
  const html = renderToStaticMarkup(
    React.createElement(ExperimentBoard, { entries: [entryValue], onOpen() {} }),
  );

  assert.deepEqual(
    board.inProgress.map((item) => item.health),
    ["wrapping_up"],
  );
  assert.match(html, /Wrapping up visualization and report/);
});

test("a final report error never becomes the board's episode health", () => {
  const reportFailed = episode({
    status: "completed",
    ending: "completed",
    wrapup_state: "failed",
    wrapup_error: "The visual report could not be generated.",
  });
  const entryValue = entry(
    "wrapping",
    "active",
    control({
      episode: reportFailed,
      health: "completed",
      recommendation: "none",
      run_section: "completed",
    }),
  );
  const board = buildExperimentBoard([entryValue]);
  const html = renderToStaticMarkup(
    React.createElement(ExperimentBoard, { entries: [entryValue], onOpen() {} }),
  );

  assert.deepEqual(
    board.finished.map((item) => item.health),
    ["completed"],
  );
  assert.match(html, /Completed/);
  assert.doesNotMatch(html, /Report error/);
});

test("board reuses loop health and groups current state in operational order", () => {
  const board = buildExperimentBoard([
    entry(
      "stopped",
      "active",
      control(
        {
          health: "human_stopped",
          recommendation: "start_episode",
          run_section: "actionable",
        },
        { current_status: "failed", stop_requested: true, stop_settled: true },
      ),
    ),
    entry(
      "running",
      "active",
      control(
        { health: "agent_active", recommendation: "wait", run_section: "running", live: true },
        { current_status: "running" },
      ),
    ),
    entry(
      "degraded",
      "active",
      control(
        { health: "degraded", recommendation: "keep_loop", run_section: "running", live: true },
        { watcher_degraded: true },
      ),
    ),
    entry(
      "finished",
      "completed",
      control({ health: "completed", recommendation: "none", run_section: "completed" }),
    ),
  ]);

  assert.deepEqual(
    board.needsAction.map((item) => [item.entry.node.id, item.health]),
    [["stopped", "human_stopped"]],
  );
  assert.deepEqual(
    board.inProgress.map((item) => [item.entry.node.id, item.health]),
    [
      ["degraded", "degraded"],
      ["running", "agent_active"],
    ],
  );
  assert.deepEqual(
    board.finished.map((item) => [item.entry.node.id, item.health]),
    [["finished", "completed"]],
  );
});

test("unsettled stops with actionable task states stay in Needs action exactly as Runs", () => {
  const board = buildExperimentBoard(
    ["failed", "paused", "interrupted"].map((status) =>
      entry(
        `stop-${status}`,
        "active",
        control(
          {},
          {
            task_active: true,
            current_status: status,
            stop_requested: true,
            stop_settled: false,
          },
        ),
      ),
    ),
  );

  assert.deepEqual(
    board.needsAction.map((item) => [item.entry.node.id, item.health]),
    [
      ["stop-failed", "needs_action"],
      ["stop-interrupted", "needs_action"],
      ["stop-paused", "needs_action"],
    ],
  );
  assert.equal(board.inProgress.length, 0);
});

test("each section sorts by newest activity with a deterministic fallback", () => {
  const board = buildExperimentBoard([
    entry(
      "older",
      "active",
      control(
        { health: "agent_active", recommendation: "wait", run_section: "running", live: true },
        { current_status: "running", current_last_activity_at: "2026-08-08T10:00:00Z" },
      ),
      "Zulu",
    ),
    entry(
      "newer",
      "active",
      control(
        { health: "agent_active", recommendation: "wait", run_section: "running", live: true },
        { current_status: "running", current_last_activity_at: "2026-08-09T10:00:00Z" },
      ),
      "Zulu",
    ),
    entry(
      "fallback-b",
      "active",
      control(
        { health: "agent_active", recommendation: "wait", run_section: "running", live: true },
        { current_status: "running" },
      ),
      "Beta",
    ),
    entry(
      "fallback-a",
      "active",
      control(
        { health: "agent_active", recommendation: "wait", run_section: "running", live: true },
        { current_status: "running" },
      ),
      "Alpha",
    ),
  ]);

  assert.deepEqual(
    board.inProgress.map((item) => item.entry.node.id),
    ["newer", "older", "fallback-a", "fallback-b"],
  );
});

test("finished outcome labels stay distinct", () => {
  assert.equal(experimentTerminalLabel("completed"), "Succeeded");
  assert.equal(experimentTerminalLabel("abandoned"), "Abandoned");
  assert.equal(experimentTerminalLabel("superseded"), "Superseded");
});

test("experiment links round-trip through the project hash parser", () => {
  const href = experimentBoardHref("remote project/one", "experiment/alpha beta");
  assert.equal(
    href,
    "#/projects/remote%20project%2Fone?view=runs&experiment=experiment%2Falpha%20beta",
  );
  assert.deepEqual(parseProjectHash(href), {
    projectId: "remote project/one",
    view: "execution",
    projectViewSpecified: true,
    experimentId: "experiment/alpha beta",
    experimentRoute: null,
  });
  assert.deepEqual(parseProjectHash("#/projects/remote%20project%2Fone"), {
    projectId: "remote project/one",
    view: "overview",
    projectViewSpecified: false,
    experimentId: null,
    experimentRoute: null,
  });
  assert.deepEqual(parseProjectHash("#/projects/new"), {
    projectId: null,
    view: "overview",
    projectViewSpecified: false,
    experimentId: null,
    experimentRoute: null,
  });
  assert.equal(projectHashAfterViewChange(href, "overview"), "#/projects/remote%20project%2Fone");
  assert.equal(projectHashAfterViewChange(href, "execution"), null);
  assert.equal(projectHashAfterViewChange("#/projects/project-one", "attention"), null);
});

test("branch Experiment links carry the exact child episode and target identity", () => {
  const branchId = "parent/episode";
  const childEpisode = episode({
    episode_id: "child/episode",
    control_node_id: "experiment/branch-only",
    status: "running",
    graph_target: { kind: "branch", branch_id: branchId },
  });
  const indexed = {
    ...entry(
      "experiment/branch-only",
      "active",
      control({ episode_id: childEpisode.episode_id, episode: childEpisode }),
    ),
    project_id: "project/one",
    graph_target: { kind: "branch", branch_id: branchId },
    graph_head: {
      target: { kind: "branch", branch_id: branchId },
      revision: 9,
      transition_id: "transition-nine",
    },
    parent_episode_id: branchId,
    episode: childEpisode,
  };

  const href = experimentBoardHref(indexed.project_id, experimentBoardRouteToken(indexed));
  assert.equal(
    href,
    "#/projects/project%2Fone?view=runs&experiment=experiment%2Fbranch-only&episode=child%2Fepisode&target=branch&branch=parent%2Fepisode&parent=parent%2Fepisode",
  );
  const parsed = parseProjectHash(href);
  assert.deepEqual(parsed.experimentRoute, {
    experiment_id: "experiment/branch-only",
    episode_id: "child/episode",
    graph_target: { kind: "branch", branch_id: branchId },
    parent_episode_id: branchId,
  });
  assert.equal(
    experimentIndexEntryForRoute([indexed], indexed.project_id, parsed.experimentRoute),
    indexed,
  );
  assert.equal(
    experimentStopPath("/api/projects/project%2Fone", indexed.node.id, childEpisode.episode_id),
    "/api/projects/project%2Fone/experiments/experiment%2Fbranch-only/stop?episode_id=child%2Fepisode",
  );
});

test("branch projection replaces colliding main state and filters control resources by target", () => {
  const branchId = "parent-episode";
  const childEpisode = episode({
    episode_id: "child-episode",
    control_node_id: "experiment/shared",
    status: "running",
    graph_target: { kind: "branch", branch_id: branchId },
    tasks: [],
  });
  const branchControl = control(
    { episode_id: childEpisode.episode_id, episode: childEpisode, active: true },
    { task_active: true, current_status: "running" },
  );
  const indexed = {
    ...entry("experiment/shared", "active", branchControl),
    project_id: "project-one",
    graph_target: { kind: "branch", branch_id: branchId },
    graph_head: {
      target: { kind: "branch", branch_id: branchId },
      revision: 8,
      transition_id: "branch-transition",
    },
    parent_episode_id: branchId,
    node: { ...node("experiment/shared", "active"), title: "Branch-modified title" },
    episode: childEpisode,
  };
  const route = parseProjectHash(
    experimentBoardHref(indexed.project_id, experimentBoardRouteToken(indexed)),
  ).experimentRoute;
  const mainTask = {
    operation_id: "main-task",
    graph_target: { kind: "main" },
    request: {
      patch_kind: "experiment_loop",
      control_node_id: indexed.node.id,
      control_episode_id: "main-episode",
    },
  };
  const branchTask = {
    operation_id: "branch-task",
    graph_target: indexed.graph_target,
    request: {
      patch_kind: "experiment_loop",
      control_node_id: indexed.node.id,
      control_episode_id: childEpisode.episode_id,
    },
  };
  const watcher = (watcherId, graphTarget, episodeId) => ({
    watcher_id: watcherId,
    graph_target: graphTarget,
    continuation: {
      patch_kind: "experiment_loop",
      control_node_id: indexed.node.id,
      control_episode_id: episodeId,
    },
  });
  const projection = projectExperimentExecution(
    [{ ...node(indexed.node.id, "active"), title: "Stale main title" }],
    [mainTask, branchTask],
    [
      watcher("main-watcher", { kind: "main" }, "main-episode"),
      watcher("branch-watcher", indexed.graph_target, childEpisode.episode_id),
    ],
    { [indexed.node.id]: control({ episode_id: "main-episode" }) },
    route,
    indexed,
  );

  assert.deepEqual(
    projection.nodes.map((item) => item.title),
    ["Branch-modified title"],
  );
  assert.deepEqual(
    projection.tasks.map((item) => item.operation_id),
    ["branch-task"],
  );
  assert.deepEqual(
    projection.watchers.map((item) => item.watcher_id),
    ["branch-watcher"],
  );
  assert.equal(projection.experimentControl[indexed.node.id], branchControl);
});

test("branch-created Runs detail uses index truth and never offers a main Start action", () => {
  const branchId = "parent-episode";
  const childEpisode = episode({
    episode_id: "child-episode",
    control_node_id: "experiment/branch-created",
    status: "running",
    // A running episode has no ending fence yet; only fencing one enters wrap-up.
    ending: null,
    wrapup_state: "not_started",
    graph_target: { kind: "branch", branch_id: branchId },
    tasks: [],
    can_stop: true,
    live: true,
    health: "active",
    recommendation: "continue",
    run_section: "needs_action",
  });
  const indexed = {
    ...entry(
      "experiment/branch-created",
      "active",
      control(
        {
          episode_id: childEpisode.episode_id,
          episode: childEpisode,
          active: true,
          health: "agent_active",
          recommendation: "wait",
          run_section: "running",
          live: true,
          can_start: false,
          can_stop: true,
        },
        { task_active: true, current_status: "running" },
      ),
    ),
    project_id: "project-one",
    graph_target: { kind: "branch", branch_id: branchId },
    graph_head: {
      target: { kind: "branch", branch_id: branchId },
      revision: 6,
      transition_id: "branch-transition",
    },
    parent_episode_id: branchId,
    node: {
      ...node("experiment/branch-created", "active"),
      title: "Created only on the branch",
    },
    episode: childEpisode,
  };
  const route = parseProjectHash(
    experimentBoardHref(indexed.project_id, experimentBoardRouteToken(indexed)),
  ).experimentRoute;
  const html = renderToStaticMarkup(
    React.createElement(ExecutionView, {
      graph: {
        revision: 5,
        nodes: {},
        edges: {},
        proposals: {},
        ambiguities: {},
        glossary: {},
        validation_messages: [],
        belief_transitions: [],
        replay_status: "complete",
        replay_failure: null,
        ontology: { types: [], fields: [], relations: [] },
      },
      episodes: [childEpisode],
      episodeMessages: {},
      episodeAction: null,
      tasks: [],
      watchers: [],
      experimentControl: {},
      exactExperimentRoute: route,
      exactExperimentEntry: indexed,
      selectedExperimentId: indexed.node.id,
      focusExperimentId: null,
      runBusy: false,
      stopBusyId: null,
      watcherCheckBusyId: null,
      taskActionId: null,
      onSelectNode() {},
      onInspectTask() {},
      onDismissTask() {},
      onSelectExperiment() {},
      onDetailFocused() {},
      onRunExperiment() {},
      onStopExperiment() {},
      onCheckExperimentWatcher() {},
      onRecoverExperiment() {},
      onSwitchExperimentProvider() {},
      episodeReportHref: () => "#",
    }),
  );

  assert.match(html, /Created only on the branch/);
  assert.match(html, /Stop loop/);
  assert.doesNotMatch(html, /Start new episode|Start episode/);

  const stoppedEpisode = {
    ...childEpisode,
    status: "stopped",
    ending: "stopped",
    wrapup_state: "skipped",
    can_stop: false,
    live: false,
    health: "stopped",
    recommendation: "none",
    run_section: "completed",
  };
  const terminalIndexed = {
    ...indexed,
    control: control(
      {
        episode_id: stoppedEpisode.episode_id,
        episode: stoppedEpisode,
        active: false,
      },
      {
        task_active: false,
        stop_requested: true,
        stop_settled: true,
        current_operation_id: null,
        current_status: null,
      },
    ),
    episode: stoppedEpisode,
  };
  const terminalHtml = renderToStaticMarkup(
    React.createElement(ExecutionView, {
      graph: {
        revision: 5,
        nodes: {},
        edges: {},
        proposals: {},
        ambiguities: {},
        glossary: {},
        validation_messages: [],
        belief_transitions: [],
        replay_status: "complete",
        replay_failure: null,
        ontology: { types: [], fields: [], relations: [] },
      },
      episodes: [stoppedEpisode],
      episodeMessages: {},
      episodeAction: null,
      tasks: [],
      watchers: [],
      experimentControl: {},
      exactExperimentRoute: route,
      exactExperimentEntry: terminalIndexed,
      selectedExperimentId: terminalIndexed.node.id,
      focusExperimentId: null,
      runBusy: false,
      stopBusyId: null,
      watcherCheckBusyId: null,
      taskActionId: null,
      onSelectNode() {},
      onInspectTask() {},
      onDismissTask() {},
      onSelectExperiment() {},
      onDetailFocused() {},
      onRunExperiment() {},
      onStopExperiment() {},
      onCheckExperimentWatcher() {},
      onRecoverExperiment() {},
      onSwitchExperimentProvider() {},
      episodeReportHref: () => "#",
    }),
  );
  assert.match(terminalHtml, /Review the owning Auto-research episode/);
  assert.doesNotMatch(terminalHtml, /Start new episode|Start episode/);
});

test("branch-created and branch-modified Experiment transcripts are read-only", () => {
  const project = {
    id: "project-one",
    name: "Project One",
    agent_profiles: {
      node_chat: {
        provider: "codex",
        model: null,
        reasoning: null,
        run_on: "local",
        permissions: {},
      },
    },
    provider_readiness: {
      local: {
        codex: {
          provider: "codex",
          label: "Codex",
          installed: true,
          authenticated: true,
          binary_path: "/usr/bin/codex",
          path_state: "resolved",
          models: [],
        },
      },
    },
    repositories: [{ alias: "repo", machine: "local", path: "/repo" }],
    project_truth_scope: ["repo"],
    state_repository: "repo",
    machines: [{ alias: "local", host: null }],
  };
  const branchNodes = [
    {
      ...node("experiment/branch-created", "active"),
      title: "Created only on the branch",
    },
    {
      ...node("experiment/shared", "active"),
      title: "Modified on the branch",
    },
  ];

  branchNodes.forEach((branchNode, index) => {
    const transcriptText = `Branch transcript ${index + 1} remains visible.`;
    const html = renderToStaticMarkup(
      React.createElement(NodeChat, {
        project,
        node: branchNode,
        nodes: { [branchNode.id]: branchNode },
        runScope: ["repo"],
        tasks: [],
        historyMessages: [
          {
            message_id: `message-${index}`,
            operation_id: `operation-${index}`,
            role: "assistant",
            text: transcriptText,
            timestamp: `2026-08-18T00:0${index}:00Z`,
            native_session_id: `session-${index}`,
            provider: "codex",
            model: null,
            reasoning: null,
            execution_machine: "local",
            applied_revision: null,
            mode: "work",
            graph_update: {
              status: "rejected",
              applied_revision: null,
              change_summary: [],
              proposal_ids: [],
              validation_messages: ["Branch graph update needs review."],
              correction_rounds: 0,
              repairable: true,
            },
            trigger: "human",
          },
        ],
        chatId: `branch-chat-${index}`,
        presentation: "workspace",
        fixedConversation: true,
        readOnly: true,
        onStartTask() {
          throw new Error("read-only branch transcript cannot start a task");
        },
        onInspectTask() {},
        onOpenInbox() {},
        onRepairGraphUpdate() {
          throw new Error("read-only branch transcript cannot repair a graph update");
        },
        onNewSession() {
          throw new Error("read-only branch transcript cannot start a session");
        },
        onClose() {},
        onResumeTask() {
          throw new Error("read-only branch transcript cannot resume a task");
        },
        onRetryTask() {
          throw new Error("read-only branch transcript cannot retry a task");
        },
      }),
    );

    assert.match(html, new RegExp(transcriptText.replace(".", "\\.")));
    assert.doesNotMatch(html, /chat-composer/);
    assert.doesNotMatch(html, /aria-label="Message"/);
    assert.doesNotMatch(html, /chat-send-button/);
    assert.doesNotMatch(html, /chat-mode-toggle/);
    assert.doesNotMatch(html, /aria-keyshortcuts/);
    assert.doesNotMatch(html, /type="file"/);
    assert.doesNotMatch(html, /chat-add-file/);
    assert.doesNotMatch(html, /chat-new-session|scope-trigger/);
    assert.match(html, /<button type="button" disabled="">[\s\S]*?Repair graph update<\/button>/);
  });
});

test("partial branch identity fails closed instead of selecting the same id on main", () => {
  assert.deepEqual(
    parseProjectHash(
      "#/projects/project-one?view=runs&experiment=experiment%2Fshared&episode=child&target=branch",
    ),
    {
      projectId: "project-one",
      view: "execution",
      projectViewSpecified: true,
      experimentId: null,
      experimentRoute: null,
    },
  );
});

test("the rendered board keeps finished work folded and unavailable work explicit", () => {
  const entries = [
    {
      ...entry("needs-human", "active", control({}, { current_status: "failed" })),
      project_reachable: false,
      node: {
        ...node("needs-human", "active"),
        next_action: "Choose the recovery path.",
        current_summary: "An older summary.",
      },
    },
    entry(
      "done",
      "superseded",
      control({ health: "completed", recommendation: "none", run_section: "completed" }),
    ),
  ];
  const html = renderToStaticMarkup(
    React.createElement(ExperimentBoard, { entries, onOpen: () => undefined }),
  );

  assert.match(html, /<h2 id="experiment-board-title">Experiments<\/h2>/);
  assert.match(html, /<details class="experiment-board-finished">/);
  assert.doesNotMatch(html, /<details[^>]+open/);
  assert.match(html, /Superseded/);
  assert.match(html, /Unavailable/);
  assert.match(html, /Choose the recovery path\./);
  assert.doesNotMatch(html, /An older summary\./);
  assert.doesNotMatch(html, />Run<|>Retry<|>Stop</);
});
