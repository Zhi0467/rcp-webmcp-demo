import assert from "node:assert/strict";
import { withTaskAnswers } from "./taskAnswers.mjs";
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
const { AttentionOverview, ExecutionView } = await server.ssrLoadModule(
  "/src/views/GraphViews.tsx",
);
const {
  decisionsAwaitingChoice,
  humanAttentionBlockers,
  shouldShowCoverageBoundaryWarning,
  taskRetryRequestBody,
} = await server.ssrLoadModule("/src/App.tsx");
const { AttentionRail } = await server.ssrLoadModule("/src/components/AttentionRail.tsx");
const { ProjectSettings } = await server.ssrLoadModule("/src/views/ProjectSettings.tsx");
const { decodeGraphAttentionProjection } = await server.ssrLoadModule("/src/types.ts");

after(() => server.close());

test("Experiment provider-switch retry overrides never submit run_on", () => {
  const task = {
    request: { patch_kind: "experiment_loop" },
  };
  const config = {
    provider: "claude",
    model: "claude-sonnet-4-5",
    reasoning: "high",
    run_on: "cluster",
  };

  assert.deepEqual(taskRetryRequestBody(task, config), {
    provider: "claude",
    model: "claude-sonnet-4-5",
    reasoning: "high",
  });
});

function graph(overrides = {}) {
  return {
    revision: 1,
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
    ...overrides,
  };
}

function blocker(id, blockerType, status = "open", standing = "asserted") {
  return {
    id,
    type: "blocker",
    title: id,
    blocker_type: blockerType,
    status,
    standing,
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
}

function decision(id, status) {
  return {
    id,
    type: "decision",
    title: id,
    question: `Choose ${id}?`,
    options: ["First", "Second"],
    selected_option: status === "decided" || status === "revisit" ? "First" : null,
    status,
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
}

test("attention decoding validates shape and referenced graph member types", () => {
  const state = graph({
    nodes: {
      decision: decision("decision", "decided"),
      blocker: blocker("blocker", "scientific", "resolved", "accepted"),
    },
    proposals: { proposal: { id: "proposal" } },
  });
  const attention = {
    pending_proposal_ids: ["proposal"],
    decisions_awaiting_choice_ids: ["decision"],
    open_blocker_ids: ["blocker"],
  };

  assert.deepEqual(decodeGraphAttentionProjection(attention, state), attention);
  assert.throws(
    () => decodeGraphAttentionProjection({ ...attention, extra: [] }, state),
    /missing or malformed/,
  );
  assert.throws(
    () =>
      decodeGraphAttentionProjection(
        { ...attention, open_blocker_ids: ["blocker", "blocker"] },
        state,
      ),
    /duplicate open_blocker_ids/,
  );
  assert.throws(
    () =>
      decodeGraphAttentionProjection({ ...attention, pending_proposal_ids: ["missing"] }, state),
    /missing Proposal missing/,
  );
  assert.throws(
    () => decodeGraphAttentionProjection({ ...attention, open_blocker_ids: ["decision"] }, state),
    /is not a Blocker/,
  );
});

function task(operationId, kind, status, statusMessage, updatedAt = "2026-08-03T00:00:00Z") {
  return withTaskAnswers({
    operation_id: operationId,
    project_id: "project",
    kind,
    status,
    request: {},
    created_at: updatedAt,
    updated_at: updatedAt,
    status_message: statusMessage,
    attempt: 1,
    estimate_seconds: 60,
    estimate_samples: 1,
    phase: status,
    elapsed_seconds: 1,
    progress: 0.1,
    can_pause: false,
    can_resume: false,
    can_retry: false,
  });
}

function runsEpisode(id, mode, runSection, createdAt, controlNodeId = null) {
  const completed = runSection === "completed";
  return {
    episode_id: id,
    project_id: "project",
    mode,
    control_node_id: controlNodeId,
    graph_target: { kind: "main" },
    graph_base_head: null,
    graph_branch: null,
    root_operation_id: null,
    current_operation_id: null,
    current_orchestrator_task_id: null,
    current_control_task_id: null,
    recovery: null,
    status: completed ? "completed" : "running",
    starting_instruction: null,
    budget: {
      invocation_ceiling: 3,
      invocations_used: completed ? 3 : 1,
      invocations_remaining: completed ? 0 : 2,
      observed_input_tokens: 10,
      observed_generated_tokens: 20,
    },
    authorized_by: null,
    stop_requested_at: null,
    ending: completed ? "completed" : null,
    ending_diagnostic: null,
    wrapup_state: completed ? "ready" : "not_started",
    wrapup_error: null,
    created_at: createdAt,
    updated_at: createdAt,
    ended_at: completed ? createdAt : null,
    tasks: [],
    report: null,
    can_stop: !completed,
    can_reauthorize: false,
    can_message: mode === "auto_research" && !completed,
    live: !completed,
    health: completed ? "completed" : "active",
    recommendation: completed ? "none" : "continue",
    task_control: null,
    run_section: runSection,
  };
}

function experimentNode(id) {
  return {
    id,
    type: "experiment",
    title: id,
    objective: "Exercise the run projection.",
    design: "",
    expected_outcomes: [],
    interpretation_rules: [],
    completion_criteria: [],
    invocation_ceiling: 3,
    attempts: [],
    current_summary: "Finished",
    next_action: null,
    status: "completed",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
}

function completedExperimentControl(episode) {
  return {
    ready: true,
    reasons: [],
    graph_reasons: [],
    invocations_used: episode.budget.invocations_used,
    invocation_ceiling: episode.budget.invocation_ceiling,
    invocations_remaining: episode.budget.invocations_remaining,
    episode_id: episode.episode_id,
    episode,
    paused: false,
    active: false,
    governing_decisions: [],
    decision_drift: [],
    operational: {},
    health: "completed",
    recommendation: "none",
    run_section: "completed",
    live: false,
    can_start: true,
    can_stop: false,
    stop_pending: false,
    task_control: null,
    can_switch_provider: false,
    can_open_report: false,
    node_closed: true,
  };
}

test("Inbox counts pending proposals, queued Decisions, and only asserted open blockers", () => {
  const asserted = blocker("ASSERTED OPEN", "scientific");
  const accepted = blocker("ACCEPTED OPEN", "infrastructure", "open", "accepted");
  const contested = blocker("CONTESTED OPEN", "design", "open", "contested");
  const resolved = blocker("ASSERTED RESOLVED", "design", "resolved");
  const decisionOpen = decision("OPEN DECISION", "open");
  const decisionReady = decision("READY DECISION", "ready");
  const decisionRevisit = decision("REVISIT DECISION", "revisit");
  const nodes = Object.fromEntries(
    [asserted, accepted, contested, resolved, decisionOpen, decisionReady, decisionRevisit].map(
      (node) => [node.id, node],
    ),
  );
  assert.deepEqual(
    humanAttentionBlockers(["ASSERTED OPEN"], nodes).map((node) => node.id),
    ["ASSERTED OPEN"],
  );

  const pending = {
    id: "pending",
    title: "Pending",
    card: {},
    ops: [],
    related_node_ids: [],
    base_rev: 1,
    status: "pending",
  };

  const html = renderToStaticMarkup(
    React.createElement(AttentionOverview, {
      proposals: [pending],
      decisions: [decisionReady, decisionRevisit],
      blockers: [asserted],
      onSelectNode() {},
    }),
  );

  assert.match(html, /4 open/);
  assert.match(html, /Decisions awaiting choice<\/span><strong>2<\/strong>/);
  assert.match(html, /Blockers awaiting judgment<\/span><strong>1<\/strong>/);
  assert.doesNotMatch(html, /Open ambiguities|Open blockers|Scientific blockers|Resolve “/);
});

test("Decision rows render supplied backend membership with staged presentation fields", () => {
  const statuses = ["open", "ready", "decided", "revisit", "superseded"];
  const canonicalNodes = Object.fromEntries(
    statuses.map((status) => {
      const node = decision(status.toUpperCase(), status);
      return [node.id, node];
    }),
  );
  const presentedNodes = Object.fromEntries(
    Object.entries(canonicalNodes).map(([status, node]) => [
      node.id,
      {
        ...node,
        title: `STAGED ${status.toUpperCase()}`,
        status: status === "ready" || status === "revisit" ? "decided" : node.status,
        draft_touched: true,
      },
    ]),
  );

  assert.deepEqual(
    decisionsAwaitingChoice(["READY", "REVISIT"], canonicalNodes, presentedNodes).map((node) => [
      node.id,
      node.title,
      node.status,
      node.draft_touched,
    ]),
    [
      ["READY", "STAGED READY", "ready", true],
      ["REVISIT", "STAGED REVISIT", "revisit", true],
    ],
  );
  assert.deepEqual(decisionsAwaitingChoice([], presentedNodes, presentedNodes), []);
});

test("Decision attention rows show only title and state and open the existing node card", () => {
  const selected = [];
  const decisions = [decision("READY ROW", "ready"), decision("REVISIT ROW", "revisit")];
  const props = {
    decisions,
    blockers: [],
    onSelectNode(nodeId) {
      selected.push(nodeId);
    },
  };
  const html = renderToStaticMarkup(React.createElement(AttentionRail, props));

  assert.match(html, /READY ROW/);
  assert.match(html, /REVISIT ROW/);
  assert.match(html, />Ready<\/span>/);
  assert.match(html, />Revisit<\/span>/);
  assert.doesNotMatch(html, /First|Second|Resolve|Dismiss|ambiguity/i);

  const tree = AttentionRail(props);
  const readyRow = findElement(tree, (element) => element.key === "READY ROW");
  assert.ok(readyRow);
  readyRow.props.onClick();
  assert.deepEqual(selected, ["READY ROW"]);
});

test("a successful Seed or Refresh suppresses the unseeded coverage warning", () => {
  const coverage = {
    repositories_never_seen: ["repo-a"],
    sessions_skipped: [],
  };

  assert.equal(shouldShowCoverageBoundaryWarning({ coverage, last_refresh_at: null }), true);
  assert.equal(
    shouldShowCoverageBoundaryWarning({
      coverage: { ...coverage, note: "No seed has completed." },
      last_refresh_at: "2026-08-06T10:00:00Z",
    }),
    false,
  );
  assert.equal(
    shouldShowCoverageBoundaryWarning({
      coverage: {
        ...coverage,
        note: "One source thread was skipped.",
        sessions_skipped: ["repo-a/session-1"],
      },
      last_refresh_at: "2026-08-06T10:00:00Z",
    }),
    true,
  );
});

test("Blocker rows render exactly the supplied backend preview membership", () => {
  const canonicalNodes = {
    agree: blocker("STAGED AGREE", "scientific"),
    contest: blocker("STAGED CONTEST", "design"),
  };
  const presentedNodes = {
    "STAGED AGREE": blocker("STAGED AGREE", "scientific", "open", "accepted"),
    "STAGED CONTEST": blocker("STAGED CONTEST", "design", "open", "contested"),
  };

  assert.deepEqual(
    humanAttentionBlockers(["STAGED AGREE", "STAGED CONTEST"], presentedNodes).map((node) => [
      node.id,
      node.standing,
    ]),
    [
      ["STAGED AGREE", "accepted"],
      ["STAGED CONTEST", "contested"],
    ],
  );
  assert.deepEqual(humanAttentionBlockers([], presentedNodes), []);
});

test("Runs is episode-first while Experiment placement and status stay control-authoritative", () => {
  const newestExperiment = runsEpisode(
    "experiment-newest",
    "experiment_loop",
    "needs_action",
    "2026-08-03T04:00:00Z",
    "EXP NEWEST",
  );
  const activeAutoResearch = runsEpisode(
    "auto-active",
    "auto_research",
    "needs_action",
    "2026-08-03T03:00:00Z",
  );
  const completedExperiment = runsEpisode(
    "experiment-complete",
    "experiment_loop",
    "completed",
    "2026-08-03T02:00:00Z",
    "EXP COMPLETE",
  );
  const completedAutoResearch = runsEpisode(
    "auto-complete",
    "auto_research",
    "completed",
    "2026-08-03T01:00:00Z",
  );
  const olderSameExperiment = runsEpisode(
    "experiment-older",
    "experiment_loop",
    "completed",
    "2026-08-03T00:30:00Z",
    "EXP NEWEST",
  );
  const html = renderToStaticMarkup(
    React.createElement(ExecutionView, {
      graph: graph({
        nodes: {
          [newestExperiment.control_node_id]: experimentNode(newestExperiment.control_node_id),
          [completedExperiment.control_node_id]: experimentNode(
            completedExperiment.control_node_id,
          ),
        },
      }),
      episodes: [
        olderSameExperiment,
        completedAutoResearch,
        completedExperiment,
        activeAutoResearch,
        newestExperiment,
      ],
      episodeMessages: {},
      episodeAction: null,
      tasks: [
        task("refresh", "refresh", "failed", "REFRESH FAILURE", "2026-08-03T01:00:00Z"),
        task("seed", "seed", "succeeded", "SEED COMPLETE"),
        task("running-refresh", "refresh", "running", "REFRESH RUNNING"),
        task("node-chat", "node_chat", "failed", "NODE CHAT TRACEBACK"),
        task("project-chat", "project_chat", "running", "PROJECT CHAT RUNNING"),
        task("coach", "paper_coach", "failed", "PAPER COACH FAILURE"),
      ],
      watchers: [],
      experimentControl: {
        [newestExperiment.control_node_id]: completedExperimentControl(newestExperiment),
        [completedExperiment.control_node_id]: completedExperimentControl(completedExperiment),
      },
      selectedExperimentId: null,
      focusExperimentId: null,
      runBusy: false,
      stopBusyId: null,
      watcherCheckBusyId: null,
      taskActionId: null,
      onInspectTask() {},
      async onLoadEpisodeMessages() {},
      async onStopEpisode() {},
      async onMergeEpisode() {},
      async onReauthorizeEpisode() {},
      async onSendEpisodeMessage() {},
      async onOperateEpisodeTask() {},
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

  assert.ok(html.indexOf(">Needs Action<") < html.indexOf(">Completed<"));
  assert.match(html, /Needs Action<\/h2><span>1<\/span>/);
  assert.match(html, /Completed<\/h2><span>3<\/span>/);
  assert.match(html, /campaign-run-title.*?<span>EXP NEWEST<\/span>/);
  assert.match(html, /<time dateTime="2026-08-03T04:00:00Z">/);
  assert.doesNotMatch(html, /2026-08-03T00:30:00Z/);
  assert.match(
    html,
    /campaign-run-title.*?<span>EXP NEWEST<\/span><\/strong><span class="campaign-run-meta"><span class="status-pill completed">Completed<\/span><time/,
  );
  assert.doesNotMatch(html, /Episode ·|campaign-run-summary|No action needed|Project episode/);
  assert.equal(html.match(/>Experiment loop<\/strong>/g)?.length, 1);
  assert.match(html, /<details class="episode-type-group"><summary><strong>Experiment loop/);
  assert.match(html, /<details class="episode-type-group"><summary><strong>Auto-research/);
  assert.ok(
    html.indexOf("<strong>Experiment loop</strong>") <
      html.lastIndexOf("<strong>Auto-research</strong>"),
  );
  assert.doesNotMatch(
    html,
    /REFRESH RUNNING|REFRESH FAILURE|SEED COMPLETE|NODE CHAT TRACEBACK|PROJECT CHAT RUNNING|PAPER COACH FAILURE/,
  );
});

test("Runs fails loudly when a cached Experiment control lacks backend lifecycle answers", () => {
  const experiment = {
    id: "exp/legacy-cache",
    type: "experiment",
    title: "Legacy cached experiment",
    objective: "Keep old cached snapshots renderable.",
    design: "",
    expected_outcomes: [],
    interpretation_rules: [],
    completion_criteria: [],
    invocation_ceiling: 2,
    attempts: [],
    current_summary: "Cached summary",
    next_action: null,
    status: "planned",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
  assert.throws(
    () =>
      renderToStaticMarkup(
        React.createElement(ExecutionView, {
          graph: graph({ nodes: { [experiment.id]: experiment } }),
          episodes: [],
          episodeMessages: {},
          episodeAction: null,
          tasks: [],
          watchers: [],
          experimentControl: {
            [experiment.id]: {
              ready: true,
              reasons: [],
              invocations_used: 0,
              invocation_ceiling: 2,
              invocations_remaining: 2,
              episode_id: null,
              paused: false,
              active: false,
              governing_decisions: [],
              decision_drift: [],
            },
          },
          selectedExperimentId: null,
          focusExperimentId: null,
          runBusy: false,
          stopBusyId: null,
          watcherCheckBusyId: null,
          taskActionId: null,
          onInspectTask() {},
          async onLoadEpisodeMessages() {},
          async onStopEpisode() {},
          async onMergeEpisode() {},
          async onReauthorizeEpisode() {},
          async onSendEpisodeMessage() {},
          async onOperateEpisodeTask() {},
          onSelectExperiment() {},
          onDetailFocused() {},
          onRunExperiment() {},
          onStopExperiment() {},
        }),
      ),
    /incomplete backend control projection/,
  );
});

test("Project Settings supports legacy profiles without an ontology authoring surface", () => {
  const storage = new Map();
  const previousLocalStorage = globalThis.localStorage;
  globalThis.localStorage = {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
  };
  try {
    const permissions = {
      read_graph: true,
      read_research_md: true,
      read_introduction: false,
      read_repositories: "run_scope",
      read_conversations: "none",
      write_graph_patch: false,
      write_project_files: false,
      write_paper: false,
    };
    const profile = {
      provider: "codex",
      model: "",
      reasoning: "medium",
      run_on: "local",
      permissions,
    };
    const metric = {
      bytes: 0,
      count: 0,
      limits: { max_bytes: 1, max_count: 1, ttl_seconds: 1 },
      reclaimable_bytes: 0,
      reclaimable_count: 0,
    };
    const html = renderToStaticMarkup(
      React.createElement(ProjectSettings, {
        apiBase: "/api/projects/project",
        project: {
          id: "project",
          name: "Project",
          state_repository: "repo",
          run_on: "local",
          default_run_truth_scope: ["repo"],
          repositories: [{ alias: "repo", machine: "local", path: "/repo" }],
          machines: [{ alias: "local", host: "", provider_paths: { codex: "codex" } }],
          agent_profiles: {
            seed: profile,
            refresh: { ...profile, model: "legacy-refresh" },
            node_chat: profile,
            project_chat: profile,
            paper_coach: profile,
          },
          providers: {},
          provider_readiness: {},
          cache_metrics: { remote_sources: metric, session_slices: metric },
        },
        usage: null,
        onRefreshUsage: async () => {},
        cacheClearDisabled: false,
        onSaved() {},
        onCacheMetricsChange() {},
        onRefreshReadiness: async () => {},
        showDisplaySettings: false,
        textScale: 100,
        onTextScaleChange() {},
      }),
    );

    assert.match(html, /Project boundary/);
    assert.match(html, /Agent defaults/);
    assert.equal(html.match(/<strong>Orchestrator<\/strong>/g)?.length, 1);
    assert.match(html.slice(html.indexOf("<strong>Orchestrator</strong>")), /legacy-refresh/);
    assert.doesNotMatch(html, /Your identity|Save name/);
    assert.doesNotMatch(html, /Ontology|Add node type|Add field|Add relation/);
  } finally {
    globalThis.localStorage = previousLocalStorage;
  }
});

function findElement(node, predicate) {
  if (Array.isArray(node)) {
    for (const child of node) {
      const match = findElement(child, predicate);
      if (match) return match;
    }
    return null;
  }
  if (!node || typeof node !== "object") return null;
  if (node.props && predicate(node)) return node;
  const children = node.props?.children;
  for (const child of Array.isArray(children) ? children : [children]) {
    const match = findElement(child, predicate);
    if (match) return match;
  }
  return null;
}
