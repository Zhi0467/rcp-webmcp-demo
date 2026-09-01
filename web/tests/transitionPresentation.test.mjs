import assert from "node:assert/strict";
import { after, test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { withExperimentControlAnswers } from "./taskAnswers.mjs";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { ExperimentBoard } = await server.ssrLoadModule("/src/components/ExperimentBoard.tsx");
const { ExperimentRunDetail } = await server.ssrLoadModule(
  "/src/components/ExperimentRunDetail.tsx",
);
const { DetailDrawer } = await server.ssrLoadModule("/src/components/DetailDrawer.tsx");
const { buildExperimentRun } = await server.ssrLoadModule("/src/runProjection.ts");

after(() => server.close());

function experiment(fields = {}) {
  return {
    id: "experiment/stale-guidance",
    type: "experiment",
    title: "Stale guidance experiment",
    extension_fields: {},
    standing: "asserted",
    created_rev: 1,
    updated_rev: 2,
    source_refs: [],
    status: "planned",
    objective: "Measure the final gate state.",
    attempts: [],
    completion_criteria: [],
    invocation_ceiling: 2,
    ...fields,
  };
}

function control() {
  return withExperimentControlAnswers({
    ready: true,
    reasons: [],
    graph_reasons: [],
    invocations_used: 0,
    invocation_ceiling: 2,
    invocations_remaining: 2,
    episode_id: null,
    episode: null,
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
        provider: "codex",
        model: null,
        reasoning: null,
        run_on: "local",
        execution_host: "local",
        run_truth_scope: null,
        native_session_bound: false,
        diagnostic: null,
      },
    },
  });
}

test("current-flow board copy omits stale Experiment summary and next action", () => {
  const node = experiment({
    current_summary: "Blocked on the old gate.",
    current_summary_stale: true,
    next_action: "Wait for the resolved Blocker.",
    next_action_stale: true,
  });
  const html = renderToStaticMarkup(
    React.createElement(ExperimentBoard, {
      entries: [
        {
          project_id: "project",
          project_name: "Project",
          project_reachable: true,
          node,
          control: control(),
          episode: null,
        },
      ],
      onOpen() {},
    }),
  );

  assert.doesNotMatch(html, /Blocked on the old gate/);
  assert.doesNotMatch(html, /Wait for the resolved Blocker/);
});

test("Experiment detail labels retained stale guidance without presenting it as current", () => {
  const node = experiment({
    current_summary: "Blocked on the old gate.",
    current_summary_stale: true,
    next_action: "Wait for the resolved Blocker.",
    next_action_stale: true,
  });
  const run = buildExperimentRun(node, control(), [], []);
  const html = renderToStaticMarkup(
    React.createElement(ExperimentRunDetail, {
      run,
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
      episodeReportHref: () => "#report",
    }),
  );

  assert.match(html, /Measure the final gate state\./);
  assert.match(html, /Previous research summary \(stale\).*Blocked on the old gate\./s);
  assert.match(html, /Previous next action \(stale\).*Wait for the resolved Blocker\./s);
  assert.doesNotMatch(html, />Next action<\/span>Wait for the resolved Blocker/);
});

test("the node DetailDrawer labels stale guidance as historical instead of current", () => {
  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1200, innerHeight: 800 };
  try {
    const node = experiment({
      current_summary: "Blocked on the old gate.",
      current_summary_stale: true,
      next_action: "Wait for the resolved Blocker.",
      next_action_stale: true,
    });
    const html = renderToStaticMarkup(
      React.createElement(DetailDrawer, {
        node,
        edges: [],
        allNodes: { [node.id]: node },
        glossaryIndex: { entriesByInitial: new Map() },
        beliefTransitions: [],
        validationMessages: [],
        ontology: { types: [], fields: [], relations: [] },
        detailSlot: "original",
        onClose() {},
        onDock() {},
        onBeginEdit() {},
        onStanding() {},
        onStage() {},
        onOpenChat() {},
        onOpenRelatedNode() {},
        onSelectNode() {},
      }),
    );

    assert.match(html, /Previous research summary \(stale\)/);
    assert.match(html, /Previous next action \(stale\)/);
    assert.doesNotMatch(html, />Where things stand</);
    assert.doesNotMatch(html, /Current summary stale|Next action stale/);

    const editHtml = renderToStaticMarkup(
      React.createElement(DetailDrawer, {
        node,
        edges: [],
        allNodes: { [node.id]: node },
        glossaryIndex: { entriesByInitial: new Map() },
        beliefTransitions: [],
        validationMessages: [],
        ontology: { types: [], fields: [], relations: [] },
        detailSlot: "original",
        behind: true,
        canonicalNode: node,
        draftNodeChange: { base_updated_rev: node.updated_rev, changes: {} },
        onClose() {},
        onDock() {},
        onBeginEdit() {},
        onStanding() {},
        onStage() {},
        onOpenChat() {},
        onOpenRelatedNode() {},
        onSelectNode() {},
      }),
    );
    assert.match(editHtml, /Previous research summary \(stale\)/);
    assert.match(editHtml, /Previous next action \(stale\)/);
    assert.doesNotMatch(editHtml, /Current summary/);
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
});
