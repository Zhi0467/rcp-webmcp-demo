import assert from "node:assert/strict";
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
const { RunDialog, agentSelectionChanged } = await server.ssrLoadModule(
  "/src/components/RunDialog.tsx",
);
const { AgentTaskInspector } = await server.ssrLoadModule("/src/components/AgentTaskInspector.tsx");
const { ProposalJudgmentSection } = await server.ssrLoadModule("/src/components/AttentionRail.tsx");
const { DetailDrawer } = await server.ssrLoadModule("/src/components/DetailDrawer.tsx");
const { shouldStartWindowDrag } = await server.ssrLoadModule("/src/components/DraggableWindow.tsx");
const { NodeChat } = await server.ssrLoadModule("/src/components/NodeChat.tsx");
const { providerPathPresentation } = await server.ssrLoadModule("/src/views/ProjectSettings.tsx");
const { ChatsWorkspace } = await server.ssrLoadModule("/src/views/ChatsWorkspace.tsx");

after(() => server.close());

const project = {
  agent_profiles: {
    seed: {
      provider: "codex",
      model: "",
      reasoning: "medium",
      run_on: "local",
      permissions: {},
    },
    node_chat: {
      provider: "codex",
      model: "",
      reasoning: "medium",
      run_on: "local",
      permissions: {},
    },
    project_chat: {
      provider: "codex",
      model: "",
      reasoning: "medium",
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
        models: [],
      },
    },
  },
  repositories: [{ alias: "repo", machine: "local", path: "/repo" }],
  project_truth_scope: ["repo"],
  state_repository: "repo",
  machines: [{ alias: "local", host: null }],
  id: "project",
  name: "Project",
};

test("seed and refresh runs offer one empty, labelled additional-message field", () => {
  const html = renderToStaticMarkup(
    React.createElement(RunDialog, {
      open: true,
      kind: "seed",
      project,
      initialScope: ["repo"],
      busy: false,
      onClose() {},
      onRun() {},
    }),
  );

  assert.match(
    html,
    /<label[^>]*>\s*<span>Additional message \(optional\)<\/span>\s*<textarea rows="4"><\/textarea>/,
  );
  assert.equal(html.match(/<textarea/g)?.length, 1);
  assert.doesNotMatch(html, /placeholder=/);
});

test("a closed run dialog renders no message field", () => {
  const html = renderToStaticMarkup(
    React.createElement(RunDialog, {
      open: false,
      kind: "seed",
      project,
      initialScope: ["repo"],
      busy: false,
      onClose() {},
      onRun() {},
    }),
  );

  assert.equal(html, "");
});

test("chat history exposes one explicit end-of-list page control", () => {
  const common = {
    project,
    conversations: [],
    selectedChatId: null,
    nodes: {},
    runScope: [],
    tasks: [],
    activeTask: null,
    watchers: [],
    graphChangesDisabled: false,
    unreadTaskIds: new Set(),
    chatTranscripts: new Map(),
    onSelect() {},
    onLoadMore() {},
    onStartTask() {},
  };
  const ready = renderToStaticMarkup(
    React.createElement(ChatsWorkspace, {
      ...common,
      hasMore: true,
      loadingMore: false,
    }),
  );
  const complete = renderToStaticMarkup(
    React.createElement(ChatsWorkspace, {
      ...common,
      hasMore: false,
      loadingMore: false,
    }),
  );
  assert.match(ready, /<button class="button primary compact" type="button">Load more<\/button>/);
  assert.doesNotMatch(complete, /Load more/);
});

test("experiment detail hides attempt history, shows the exact gate, and keeps Ask available", () => {
  const node = {
    id: "experiment/demo",
    type: "experiment",
    title: "Demo run",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
    objective: "Test the mechanism",
    invocation_ceiling: 7,
    attempts: [
      {
        id: "attempt-1",
        sequence: 1,
        purpose: "Train the ablation",
        attempt_kind: "external_run",
        decision_bundle: [
          { decision_id: "decision/resource", decision_revision: 3, selected_option: "4xA100" },
        ],
        status: "running",
        job_refs: ["4471"],
      },
    ],
  };
  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1440, innerHeight: 900 };
  let html;
  try {
    html = renderToStaticMarkup(
      React.createElement(DetailDrawer, {
        node,
        edges: [],
        allNodes: { [node.id]: node },
        glossary: {},
        beliefTransitions: [],
        validationMessages: [],
        ontology: { types: [], fields: [], relations: [] },
        experimentControl: {
          ready: false,
          reasons: [
            "Decision decision/data is still open.",
            "An experiment loop is already active.",
          ],
          invocations_used: 2,
          invocation_ceiling: 3,
          invocations_remaining: 1,
          episode_id: "episode-1",
          paused: false,
          active: true,
          governing_decisions: [],
          decision_drift: [
            {
              decision_id: "decision/resource",
              pinned_option: "4xA100",
              pinned_revision: 3,
              current_option: "8xA100",
              current_status: "decided",
              proposed: false,
            },
          ],
        },
        onClose() {},
        onBeginEdit() {},
        onStanding() {},
        onStage() {},
        onRunExperiment() {},
        onOpenChat() {},
        onExploreRelations() {},
        onSelectNode() {},
      }),
    );
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
  assert.match(html, /Episode invocations/);
  assert.match(html, /2 \/ 3/);
  assert.match(html, /1 remaining/);
  assert.match(html, /Next episode limit/);
  assert.match(html, /Next episode limit<\/span><strong>7<\/strong>/);
  assert.match(html, /Active loop/);
  assert.doesNotMatch(html, /Paused at limit/);
  assert.match(html, /Decision decision\/data is still open\./);
  assert.match(html, /decision\/resource moved to 8xA100 after this episode was pinned to 4xA100/);
  assert.match(html, /<button[^>]*disabled=""[^>]*>.*Start new episode<\/button>/s);
  // Semantic attempt history belongs in Runs detail, not the node drawer.
  assert.doesNotMatch(html, /Train the ablation/);
  assert.doesNotMatch(html, /aria-label="Attempts"/);
  assert.doesNotMatch(html, /Stop attempt/);
  assert.doesNotMatch(html, /Stop watcher/);
  assert.match(html, /Ask about this node/);
});

test("a never-run Experiment shows only its next episode limit", () => {
  const node = {
    id: "experiment/new",
    type: "experiment",
    title: "New experiment",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
    objective: "Test the mechanism",
    invocation_ceiling: 6,
    attempts: [],
  };
  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1440, innerHeight: 900 };
  let html;
  try {
    html = renderToStaticMarkup(
      React.createElement(DetailDrawer, {
        node,
        edges: [],
        allNodes: { [node.id]: node },
        glossaryIndex: { entriesByInitial: new Map() },
        beliefTransitions: [],
        validationMessages: [],
        ontology: { types: [], fields: [], relations: [] },
        experimentControl: {
          ready: true,
          reasons: [],
          invocations_used: 0,
          invocation_ceiling: 6,
          invocations_remaining: 6,
          episode_id: null,
          paused: false,
          active: false,
          governing_decisions: [],
          decision_drift: [],
        },
        onClose() {},
        onBeginEdit() {},
        onStanding() {},
        onStage() {},
        onRunExperiment() {},
        onOpenChat() {},
        onExploreRelations() {},
        onSelectNode() {},
      }),
    );
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }

  assert.doesNotMatch(html, /Episode invocations/);
  assert.doesNotMatch(html, /0 \/ 6/);
  assert.match(html, /Next episode limit<\/span><strong>6<\/strong>/);
  assert.match(html, /<button[^>]*>.*Start episode<\/button>/s);
  assert.doesNotMatch(html, /Start new episode/);
});

test("an invocation-limited episode offers a new episode for its pending watcher", () => {
  const node = {
    id: "experiment/paused",
    type: "experiment",
    title: "Paused run",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
    objective: "Test the mechanism",
    invocation_ceiling: 3,
    attempts: [
      {
        id: "attempt-open",
        sequence: 1,
        purpose: "Interpret the pending run",
        attempt_kind: "external_run",
        decision_bundle: [],
        status: "running",
        job_refs: ["job-1"],
      },
    ],
  };
  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1440, innerHeight: 900 };
  let html;
  try {
    const props = {
      node,
      edges: [],
      allNodes: { [node.id]: node },
      glossary: {},
      beliefTransitions: [],
      validationMessages: [],
      ontology: { types: [], fields: [], relations: [] },
      experimentControl: {
        ready: true,
        reasons: [],
        invocations_used: 3,
        invocation_ceiling: 3,
        invocations_remaining: 0,
        episode_id: "episode-1",
        paused: true,
        active: false,
        governing_decisions: [],
        decision_drift: [],
      },
      onClose() {},
      onBeginEdit() {},
      onStanding() {},
      onStage() {},
      onRunExperiment() {},
      onOpenChat() {},
      onExploreRelations() {},
      onSelectNode() {},
    };
    html = renderToStaticMarkup(React.createElement(DetailDrawer, props));
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
  assert.match(html, /Paused at limit/);
  assert.match(html, /0 remaining/);
  assert.doesNotMatch(html, /Interpret the pending run/);
  assert.match(html, /<button[^>]*>.*Start new episode<\/button>/s);
  assert.doesNotMatch(html, /<button[^>]*disabled=""[^>]*>.*Start new episode<\/button>/s);
  assert.doesNotMatch(html, /Stop watcher/);
});

test("node standing presents Contest and Agree as independent three-state toggles", () => {
  const baseNode = {
    id: "hyp/demo",
    type: "hypothesis",
    title: "Demo hypothesis",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
  const renderNode = (standing, props = {}) =>
    renderToStaticMarkup(
      React.createElement(DetailDrawer, {
        node: { ...baseNode, standing },
        edges: [],
        allNodes: { [baseNode.id]: { ...baseNode, standing } },
        glossary: {},
        beliefTransitions: [],
        validationMessages: [],
        ontology: { types: [], fields: [], relations: [] },
        onClose() {},
        onDock() {},
        onBeginEdit() {},
        onStanding() {},
        onStage() {},
        onOpenChat() {},
        onExploreRelations() {},
        onSelectNode() {},
        ...props,
      }),
    );

  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1440, innerHeight: 900 };
  try {
    const asserted = renderNode("asserted");
    assert.match(asserted, /data-text-selectable="true"/);
    assert.match(
      asserted,
      /class="button judgment node-standing-toggle contest"[^>]*aria-pressed="false"/,
    );
    assert.match(
      asserted,
      /class="button judgment node-standing-toggle agree"[^>]*aria-pressed="false"/,
    );
    assert.match(asserted, />Contest<\/button>/);
    assert.match(asserted, />Agree<\/button>/);

    const accepted = renderNode("accepted");
    assert.match(
      accepted,
      /class="button judgment node-standing-toggle contest"[^>]*aria-pressed="false"/,
    );
    assert.match(
      accepted,
      /class="button judgment node-standing-toggle agree selected agree"[^>]*aria-pressed="true"/,
    );
    assert.match(accepted, />Contest<\/button>/);
    assert.match(accepted, />Agree<\/button>/);

    const stagedAccepted = renderNode("accepted", { canonicalStanding: "asserted" });
    assert.match(stagedAccepted, />accepted · staged<\/span>/);

    const contested = renderNode("contested");
    assert.match(
      contested,
      /class="button judgment node-standing-toggle contest selected disagree"[^>]*aria-pressed="true"/,
    );
    assert.match(
      contested,
      /class="button judgment node-standing-toggle agree"[^>]*aria-pressed="false"/,
    );
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
});

test("proposal decisions present Reject and Approve as independent three-state toggles", () => {
  const proposal = {
    id: "prop/example",
    title: "Choose the next experiment",
    base_rev: 4,
    card: {
      situation_cold: "Two experiment paths remain plausible.",
      why_human_now: "The next run depends on this choice.",
      consequences: "One path becomes the active plan.",
      decision_needed: "Approve or reject the proposed path.",
    },
  };
  const renderProposal = (decision) =>
    renderToStaticMarkup(
      React.createElement(ProposalJudgmentSection, {
        proposals: [proposal],
        graph: { proposals: { [proposal.id]: proposal } },
        draft: decision ? { proposals: { [proposal.id]: { decision } } } : null,
        onDecision() {},
      }),
    );

  const undecided = renderProposal(null);
  assert.match(
    undecided,
    /class="button judgment proposal-decision-toggle reject"[^>]*aria-pressed="false"/,
  );
  assert.match(
    undecided,
    /class="button judgment proposal-decision-toggle approve"[^>]*aria-pressed="false"/,
  );
  assert.match(undecided, />Reject<\/button>/);
  assert.match(undecided, />Approve<\/button>/);

  const approved = renderProposal("approved");
  assert.match(
    approved,
    /class="button judgment proposal-decision-toggle reject"[^>]*aria-pressed="false"/,
  );
  assert.match(
    approved,
    /class="button judgment proposal-decision-toggle approve selected agree"[^>]*aria-pressed="true"/,
  );
  assert.match(approved, /Pending · staged approved/);

  const rejected = renderProposal("rejected");
  assert.match(
    rejected,
    /class="button judgment proposal-decision-toggle reject selected disagree"[^>]*aria-pressed="true"/,
  );
  assert.match(
    rejected,
    /class="button judgment proposal-decision-toggle approve"[^>]*aria-pressed="false"/,
  );
  assert.match(rejected, /Pending · staged rejected/);
});

test("node removal is separate, guarded by canonical truth and active loops, and remains visible", () => {
  const node = {
    id: "hyp/remove",
    type: "hypothesis",
    title: "Remove this hypothesis",
    standing: "contested",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
  const renderNode = (props = {}) =>
    renderToStaticMarkup(
      React.createElement(DetailDrawer, {
        node,
        edges: [{ id: "edge/1", source: node.id, target: "hyp/other" }],
        allNodes: { [node.id]: node },
        glossary: {},
        beliefTransitions: [],
        validationMessages: [],
        ontology: { types: [], fields: [], relations: [] },
        onClose() {},
        onDock() {},
        onBeginEdit() {},
        onStanding() {},
        onStage() {},
        onRemove() {},
        onUndoRemoval() {},
        onOpenChat() {},
        onExploreRelations() {},
        onSelectNode() {},
        ...props,
      }),
    );

  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1440, innerHeight: 900 };
  try {
    const removable = renderNode();
    assert.match(removable, /Remove node…<\/button>/);
    assert.match(removable, /class="node-removal-confirmation" role="alert" hidden=""/);
    assert.match(removable, /Remove <strong>“Remove this hypothesis”<\/strong>\?/);
    assert.match(removable, /Sync will remove it and 1 connected relation\./);
    assert.match(removable, />Cancel<\/button>/);
    assert.match(removable, />Confirm remove<\/button>/);

    const accepted = renderNode({ canonicalStanding: "accepted" });
    assert.match(accepted, /Clear or contest this accepted node and Sync before removing it\./);
    assert.match(accepted, /<button[^>]*disabled=""[^>]*>.*Remove node…<\/button>/s);

    const active = renderNode({ experimentControl: { active: true } });
    assert.match(active, /bounded experiment loop is active/);

    const staged = renderNode({ stagedForRemoval: true });
    assert.match(staged, /Removal staged\./);
    assert.match(staged, /Sync will remove this node and 1 connected relation\./);
    assert.match(staged, />Undo<\/button>/);
    assert.match(staged, /aria-pressed="true" disabled=""/);
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
});

test("selectable node banner text never starts floating-window drag", () => {
  const target = (...matches) => ({
    closest(selector) {
      return matches.includes(selector) ? {} : null;
    },
  });

  assert.equal(shouldStartWindowDrag(target("[data-drag-handle]")), true);
  assert.equal(
    shouldStartWindowDrag(target("[data-drag-handle]", "[data-text-selectable]")),
    false,
  );
  assert.equal(
    shouldStartWindowDrag(target("[data-drag-handle]", "button, input, select, textarea, a")),
    false,
  );
});

test("conversation watcher status and wake attribution stay operational", () => {
  const watcher = {
    watcher_id: "watcher-1",
    chat_id: "chat",
    status: "degraded",
    log_path: "/tmp/train.log",
    last_checked_at: "2026-08-01T04:00:00Z",
    last_error: "SSH exited 255",
  };
  const props = {
    project,
    node: null,
    runScope: ["repo"],
    tasks: [],
    activeTask: null,
    historyMessages: [
      {
        message_id: "message-1",
        operation_id: "wake-1",
        role: "assistant",
        text: "The watched work finished.",
        timestamp: "2026-08-01T04:01:00Z",
        native_session_id: null,
        provider: "codex",
        model: null,
        reasoning: null,
        execution_machine: "local",
        applied_revision: null,
        mode: "work",
        graph_update: null,
        trigger: "watcher",
      },
    ],
    chatId: "chat",
    onStartTask() {},
    onInspectTask() {},
    onOpenInbox() {},
    onRepairGraphUpdate() {},
    onStopWatcher() {},
    onClose() {},
  };
  const html = renderToStaticMarkup(
    React.createElement(NodeChat, {
      ...props,
      watchers: [
        { ...watcher, continuation: { patch_kind: "work" } },
        {
          ...watcher,
          watcher_id: "watcher-stopped",
          status: "stopped",
          continuation: { patch_kind: "work" },
        },
      ],
    }),
  );
  const experimentHtml = renderToStaticMarkup(
    React.createElement(NodeChat, {
      ...props,
      watchers: [{ ...watcher, continuation: { patch_kind: "experiment_loop" } }],
    }),
  );
  // The watcher list is disclosed by the count control, so it is absent until opened.
  assert.match(html, /class="chat-watcher-count"[^>]*aria-expanded="false"/);
  assert.match(html, /aria-label="1 active watcher"/);
  assert.doesNotMatch(html, /train\.log/);
  assert.doesNotMatch(html, /SSH exited 255/);
  assert.doesNotMatch(html, /chat-watchers/);
  assert.equal(html.match(/chat-watcher-count/g).length, 1);
  assert.doesNotMatch(experimentHtml, /chat-watcher-count/);
  assert.match(html, /chat-turn-trigger watcher[^>]*>Watcher/);
  assert.doesNotMatch(html, /node-chat-line human/);
});

test("a new Experiment chat sees the node loop and only its own generic watcher", () => {
  const node = {
    id: "experiment/shared-loop",
    type: "experiment",
    title: "Shared loop",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
  const watcher = {
    watcher_id: "loop-active",
    chat_id: "creator-chat",
    status: "active",
    log_path: "/tmp/loop-active.log",
    last_checked_at: null,
    last_error: null,
    continuation: {
      patch_kind: "experiment_loop",
      control_node_id: node.id,
    },
  };
  const props = {
    project,
    node,
    runScope: ["repo"],
    tasks: [],
    historyMessages: [],
    chatId: "new-session-chat",
    onStartTask() {},
    onInspectTask() {},
    onOpenInbox() {},
    onRepairGraphUpdate() {},
    onStopWatcher() {},
    onNewSession() {},
    onClose() {},
    onResumeTask() {},
    onRetryTask() {},
  };
  const html = renderToStaticMarkup(
    React.createElement(NodeChat, {
      ...props,
      watchers: [
        watcher,
        { ...watcher, watcher_id: "loop-degraded", status: "degraded" },
        { ...watcher, watcher_id: "loop-stopped", status: "stopped" },
        { ...watcher, watcher_id: "loop-completed", status: "completed" },
        {
          ...watcher,
          watcher_id: "other-node-loop",
          continuation: {
            patch_kind: "experiment_loop",
            control_node_id: "experiment/other",
          },
        },
        {
          ...watcher,
          watcher_id: "self-wake",
          chat_id: "new-session-chat",
          continuation: { patch_kind: "work", control_node_id: null },
        },
        {
          ...watcher,
          watcher_id: "other-chat-self-wake",
          continuation: { patch_kind: "work", control_node_id: null },
        },
      ],
    }),
  );
  const projectChatHtml = renderToStaticMarkup(
    React.createElement(NodeChat, {
      ...props,
      node: null,
      watchers: [watcher],
    }),
  );

  assert.match(html, /aria-label="3 active watchers"/);
  assert.match(html, /<svg[^>]*>.*<\/svg> 3<\/button>/s);
  assert.doesNotMatch(projectChatHtml, /chat-watcher-count/);
});

test("long human chat messages render a bounded preview control", () => {
  const longMessage = "Please compare each of these constraints carefully. ".repeat(8);
  const html = renderToStaticMarkup(
    React.createElement(NodeChat, {
      project,
      node: null,
      runScope: ["repo"],
      tasks: [],
      activeTask: null,
      historyMessages: [
        {
          message_id: "message-human-long",
          operation_id: "task-human-long",
          role: "user",
          text: longMessage,
          timestamp: "2026-08-01T04:00:00Z",
          native_session_id: null,
          provider: null,
          model: null,
          reasoning: null,
          execution_machine: null,
          applied_revision: null,
          mode: "discuss",
          graph_update: null,
          trigger: "human",
        },
        {
          message_id: "message-agent-long",
          operation_id: "task-agent-long",
          role: "assistant",
          text: longMessage,
          timestamp: "2026-08-01T04:01:00Z",
          native_session_id: null,
          provider: "codex",
          model: null,
          reasoning: null,
          execution_machine: "local",
          applied_revision: null,
          mode: "discuss",
          graph_update: null,
          trigger: "human",
        },
      ],
      chatId: "chat-long-message",
      onStartTask() {},
      onInspectTask() {},
      onOpenInbox() {},
      onRepairGraphUpdate() {},
      onClose() {},
    }),
  );

  assert.match(html, /chat-human-message collapsed/);
  assert.match(html, /class="chat-message-toggle"[^>]*aria-expanded="false"[^>]*>See more/);
  assert.match(html, /chat-markdown/);
  assert.equal(html.match(/chat-human-message collapsed/g)?.length, 1);
});

test("retry keeps the original task boundary and exposes provider configuration", () => {
  const html = renderToStaticMarkup(
    React.createElement(RunDialog, {
      open: true,
      kind: "seed",
      mode: "retry",
      project,
      initialScope: ["repo"],
      initialConfig: {
        provider: "codex",
        model: "",
        reasoning: "medium",
        run_on: "local",
      },
      busy: false,
      onClose() {},
      onRun() {},
    }),
  );

  assert.match(html, /Retry seed/);
  assert.doesNotMatch(html, /Truth input subset/);
  assert.doesNotMatch(html, /Additional message/);
  assert.match(html, />\s*Retry<\/button>/);
  assert.doesNotMatch(html, /<button class="button primary" disabled=""[^>]*>.*Retry<\/button>/s);
});

test("Experiment provider switch exposes provider controls but locks the execution machine", () => {
  const html = renderToStaticMarkup(
    React.createElement(RunDialog, {
      open: true,
      kind: "node_chat",
      mode: "retry",
      project,
      initialScope: ["repo"],
      initialConfig: {
        provider: "codex",
        model: "",
        reasoning: "medium",
        run_on: "local",
      },
      busy: false,
      onClose() {},
      onRun() {},
    }),
  );

  assert.match(html, /Switch Experiment provider/);
  assert.match(html, /<span>Provider<\/span>/);
  assert.match(html, /<span>Model<\/span>/);
  assert.match(html, /<span>Run on <svg/);
  assert.match(html, /<select disabled=""><option value="local" selected="">local · local/);
  assert.match(
    html,
    /<button class="button primary" disabled=""[^>]*>.*Switch provider<\/button>/s,
  );
  assert.doesNotMatch(html, /Truth input subset|Additional message/);
});

test("Experiment provider switch requires an agent selection change and ignores run_on", () => {
  const initial = {
    provider: "codex",
    model: "gpt-5.6",
    reasoning: "high",
    run_on: "cluster",
  };

  assert.equal(agentSelectionChanged({ ...initial, run_on: "local" }, initial), false);
  assert.equal(agentSelectionChanged({ ...initial, provider: "claude" }, initial), true);
  assert.equal(agentSelectionChanged({ ...initial, model: "gpt-5.6-mini" }, initial), true);
  assert.equal(agentSelectionChanged({ ...initial, reasoning: "medium" }, initial), true);
});

test("task inspector names every provider launch by its continuation cause", () => {
  const now = new Date().toISOString();
  const promptReceipt = (receipt_id, continuation_cause) => ({
    receipt_id,
    operation_id: "task-1",
    created_at: now,
    tier: "diagnostic",
    category: "agent_prompt",
    payload: { prompt: "Open the contract.\n", continuation_cause, line_count: 1 },
  });
  const task = {
    operation_id: "task-1",
    project_id: "project-1",
    kind: "seed",
    status: "running",
    request: {
      provider: "codex",
      run_on: "local",
      resolved_provider_skills: [
        {
          provider: "codex",
          machine: "local",
          provider_version: "0.146.1",
          inventory_hash: "inventory-hash",
          name: "frontend-design:frontend-design",
          label: "Frontend design",
          description: "Shape a distinctive interface.",
          stale: true,
        },
      ],
    },
    created_at: now,
    updated_at: now,
    status_message: "Correcting the graph update.",
    attempt: 1,
    estimate_seconds: 60,
    estimate_samples: 0,
    phase: "agent",
    elapsed_seconds: 10,
    progress: 0.2,
    can_pause: true,
    can_resume: false,
    can_retry: false,
    debug_receipts: [
      promptReceipt(1, "fresh"),
      promptReceipt(2, "correction"),
      promptReceipt(3, "resume"),
      promptReceipt(4, "handoff"),
    ],
    contracts: [
      {
        operation_id: "task-1",
        role: "base",
        created_at: now,
        sha256: "contract-digest",
        content: "# RCP seed task contract\n\nInstruction and trust boundary.",
      },
    ],
    events: [],
  };
  const html = renderToStaticMarkup(
    React.createElement(AgentTaskInspector, {
      tasks: [task],
      task,
      loading: false,
      actionBusy: false,
      onSelect() {},
      onPause() {},
      onResume() {},
      onRetry() {},
      onClose() {},
    }),
  );

  assert.match(html, /First attempt · 1/);
  assert.match(html, /Correcting prior failure · 2/);
  assert.match(html, /Continuing after interruption · 3/);
  assert.match(html, /Continuing in a new session · 4/);
  assert.match(html, /Task contracts/);
  assert.match(html, />Base</);
  assert.match(html, /contract-digest/);
  assert.match(html, /# RCP seed task contract/);
  assert.match(html, /Provider-native guidance/);
  assert.match(html, /Frontend design \(frontend-design:frontend-design\)/);
  assert.match(html, /codex · local · CLI 0\.146\.1 · stale inventory/);
});

test("provider path state distinguishes a stale recorded executable", () => {
  assert.deepEqual(
    providerPathPresentation({ path_state: "missing" }, "/old/codex", "/old/codex"),
    { label: "Recorded path missing", kind: "error" },
  );
  assert.deepEqual(
    providerPathPresentation({ path_state: "resolved" }, "/new/codex", "/old/codex"),
    { label: "Unsaved", kind: "pending" },
  );
  assert.deepEqual(
    providerPathPresentation({ path_state: "denied" }, "/protected/codex", "/protected/codex"),
    { label: "Recorded path unusable", kind: "error" },
  );
});
