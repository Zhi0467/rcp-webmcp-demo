import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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
  WEBMCP_RESULT_MAX_CHARS,
  createWebMcpToolRegistry,
  currentWebMcpContext,
  inspectProjectConversation,
  inspectProjectExperiment,
  inspectProjectNode,
  listProjectsForWebMcp,
  listProjectArtifacts,
  modelContextFromDocument,
  openProjectFromIndex,
  openProjectArtifact,
  projectArtifactToolDefinitions,
  projectConversationSendToolDefinitions,
  projectConversationToolDefinitions,
  projectExperimentToolDefinitions,
  projectExperimentStopToolDefinitions,
  projectIndexToolDefinitions,
  projectOverview,
  projectReadToolDefinitions,
  registerWebMcpTools,
  sendProjectConversationMessage,
  startProjectExperiment,
  stopProjectExperimentEpisode,
  webMcpTextResult,
} = await server.ssrLoadModule("/src/webmcp.ts");

after(() => server.close());

function definition(name = "rcp_probe") {
  return {
    name,
    description: "Return a local probe.",
    inputSchema: { type: "object", additionalProperties: false },
    annotations: { readOnlyHint: true },
    execute: (input) => webMcpTextResult(input),
  };
}

function projectFixture() {
  const nodes = {
    "rq-1": {
      id: "rq-1",
      type: "research_question",
      title: "Which drift predicts retained plasticity?",
      question: "Which measurable change predicts later held-out learning?",
      extension_fields: {},
      standing: "asserted",
      created_rev: 1,
      updated_rev: 2,
      source_refs: [],
    },
    "hyp-1": {
      id: "hyp-1",
      type: "hypothesis",
      title: "Functional drift is predictive",
      statement: "Functional drift should outperform weight distance.",
      extension_fields: {},
      standing: "asserted",
      created_rev: 1,
      updated_rev: 3,
      source_refs: [],
    },
    "exp-1": {
      id: "exp-1",
      type: "experiment",
      title: "Held-out plasticity replicate",
      objective: "Run one bounded synthetic replicate.",
      status: "planned",
      extension_fields: {},
      standing: "asserted",
      created_rev: 1,
      updated_rev: 4,
      source_refs: [],
    },
    "blocker-1": {
      id: "blocker-1",
      type: "blocker",
      title: "Calibration reliability",
      description: "Confirm the fixed probe remains reliable.",
      status: "open",
      extension_fields: {},
      standing: "asserted",
      created_rev: 1,
      updated_rev: 5,
      source_refs: [],
    },
  };
  return {
    id: "project-1",
    name: "Synthetic plasticity project",
    revision: 5,
    snapshot_freshness: "fresh",
    primary_question: nodes["rq-1"],
    counts: {
      pending_proposals: 0,
      decisions_awaiting_choice: 0,
      open_blockers: 1,
      asserted: 4,
      accepted: 0,
      contested: 0,
    },
    attention: {
      open_blocker_ids: ["blocker-1"],
      decisions_awaiting_choice_ids: [],
      pending_proposal_ids: [],
    },
    graph: {
      revision: 5,
      nodes,
      edges: {
        "edge-1": {
          id: "edge-1",
          source: "hyp-1",
          target: "rq-1",
          relation: "answers",
          layer: "epistemic",
          explanation: "The hypothesis proposes one answer.",
        },
      },
    },
    experiment_control: { "exp-1": { can_start: true, can_stop: false } },
    default_run_truth_scope: ["plasticity"],
    agent_profiles: {
      node_chat: {
        provider: "codex",
        runtime: "exec",
        model: "",
        reasoning: "high",
        run_on: "local",
      },
      project_chat: {
        provider: "codex",
        runtime: "exec",
        model: "",
        reasoning: "high",
        run_on: "local",
      },
    },
    provider_readiness: {
      local: {
        codex: {
          provider: "codex",
          label: "Codex",
          installed: true,
          authenticated: true,
          default_runtime: "exec",
        },
      },
    },
    skill_catalog: [
      {
        id: "analyze",
        kind: "workflow",
        version: "1",
        label: "Analyze",
        description: "Analyze one result.",
        dependencies: [],
      },
      {
        id: "plot",
        kind: "skill",
        version: "1",
        label: "Plot",
        description: "Build one plot.",
        dependencies: [],
      },
    ],
    skill_defaults: { workflow_ids: ["analyze"], skill_ids: ["plot"] },
    provider_skill_inventories: {
      local: {
        codex: {
          status: "fresh",
          stale: false,
          skills: [
            {
              name: "browser",
              label: "Browser",
              description: "Inspect a web page.",
              enabled: true,
            },
          ],
        },
      },
    },
  };
}

function projectCardFixtures() {
  return [
    {
      id: "project-1",
      home_space_id: null,
      name: "Synthetic plasticity project",
      locator: "/private/research/plasticity",
      state_location: "/private/research/plasticity/.research",
      remote: false,
      revision: 5,
      primary_question: "Which drift predicts retained plasticity?",
      attention_count: 1,
      reachable: true,
      error: null,
      can_delete: true,
      delete_unavailable_reason: null,
    },
    {
      id: "project-2",
      home_space_id: null,
      name: "Remote calibration study",
      locator: "researcher@example.test:/secret/calibration",
      state_location: "/secret/calibration/.research",
      remote: true,
      revision: 8,
      primary_question: "Does calibration transfer?",
      attention_count: 0,
      reachable: false,
      error: "SSH host details must remain private.",
      can_delete: false,
      delete_unavailable_reason: "Remote project",
    },
  ];
}

function conversationFixtures() {
  const task = {
    operation_id: "task-chat-1",
    project_id: "project-1",
    kind: "node_chat",
    status: "succeeded",
    request: {
      node_id: "hyp-1",
      chat_id: "chat-1",
      provider: "codex",
      model: null,
      reasoning: "high",
      run_on: "local",
      run_truth_scope: ["plasticity"],
      mode: "discuss",
    },
    result: {
      messages: ["The held-out curve remains stable."],
      graph_update: {
        status: "none",
        applied_revision: null,
        change_summary: [],
        proposal_ids: [],
        validation_messages: [],
        correction_rounds: 0,
        repairable: false,
      },
      artifacts: [
        {
          artifact_id: "artifact-1",
          name: "calibration.html",
          media_type: "text/html",
          kept_filename: null,
          available: true,
          unavailable_reason: null,
          can_open: true,
          can_download: true,
          can_keep: true,
          can_revise: true,
        },
      ],
    },
    created_at: "2026-08-31T10:00:00Z",
    updated_at: "2026-08-31T10:01:00Z",
    status_message: "Completed",
    runtime_id: "exec",
    runtime_label: "Codex exec",
    native_session_id: "session-1",
    graph_target: { kind: "main" },
    active: false,
    awaiting_human: false,
    paused: false,
    failed: false,
    settled: true,
    status_label: "Finished",
  };
  const transcript = {
    chat_id: "chat-1",
    kind: "node_chat",
    node_id: "hyp-1",
    title: "Functional drift is predictive",
    updated_at: "2026-08-31T10:01:00Z",
    message_count: 2,
    last_message_preview: "The held-out curve remains stable.",
    messages: [
      {
        message_id: "message-1",
        operation_id: "task-chat-1",
        role: "user",
        text: "Does the calibration remain stable?",
        timestamp: "2026-08-31T10:00:00Z",
        native_session_id: "session-1",
        provider: "codex",
        model: "provider-default",
        reasoning: "high",
        execution_machine: "local",
        mode: "discuss",
        trigger: "human",
        graph_update: null,
        attachments: [],
      },
      {
        message_id: "message-2",
        operation_id: "task-chat-1",
        role: "assistant",
        text: "The held-out curve remains stable.",
        timestamp: "2026-08-31T10:01:00Z",
        native_session_id: "session-1",
        provider: "codex",
        model: "provider-default",
        reasoning: "high",
        execution_machine: "local",
        mode: "discuss",
        trigger: "human",
        graph_update: null,
        attachments: [],
      },
    ],
  };
  return { task, transcript };
}

function artifactFixtures() {
  const tasks = [
    {
      operation_id: "task-1",
      project_id: "project-1",
      request: { node_id: "hyp-1", chat_id: "chat-1" },
      result: {
        artifacts: [
          {
            artifact_id: "artifact-1",
            name: "calibration.html",
            media_type: "text/html",
            kept_filename: "calibration.html",
            available: true,
            unavailable_reason: null,
            can_open: true,
            can_download: true,
            can_keep: false,
            can_revise: true,
          },
        ],
      },
      created_at: "2026-08-30T10:00:00Z",
      updated_at: "2026-08-30T10:01:00Z",
    },
    {
      operation_id: "task-2",
      project_id: "project-1",
      request: { control_node_id: "exp-1", chat_id: "chat-2" },
      episode_id: "episode-1",
      result: {
        artifacts: [
          {
            artifact_id: "artifact-2",
            name: "expired.png",
            media_type: "image/png",
            available: false,
            unavailable_reason: "Artifact bytes expired.",
            can_open: false,
            can_download: false,
            can_keep: false,
            can_revise: false,
          },
        ],
      },
      created_at: "2026-08-31T10:00:00Z",
      updated_at: "2026-08-31T10:01:00Z",
    },
  ];
  const episodes = [
    {
      episode_id: "episode-1",
      project_id: "project-1",
      control_node_id: "exp-1",
      wrapup_state: "ready",
      ending: "completed",
      updated_at: "2026-08-31T10:02:00Z",
      report: {
        report_id: "report-1",
        ending: "completed",
        created_at: "2026-08-31T10:03:00Z",
      },
    },
  ];
  return { tasks, episodes };
}

test("feature detection rejects absent and partial hosts", () => {
  assert.equal(modelContextFromDocument(null), null);
  assert.equal(modelContextFromDocument({}), null);
  assert.equal(modelContextFromDocument({ modelContext: {} }), null);
  assert.equal(currentWebMcpContext(), null);
});

test("one controller owns every registration and abort removes them together", () => {
  const registered = new Map();
  const context = {
    registerTool(tool, options) {
      registered.set(tool.name, tool);
      options.signal.addEventListener("abort", () => registered.delete(tool.name), { once: true });
    },
  };
  assert.equal(modelContextFromDocument({ modelContext: context }), context);

  const registration = registerWebMcpTools([definition("one"), definition("two")], context);
  assert.deepEqual([...registered.keys()], ["one", "two"]);
  assert.equal(registration.controller.signal.aborted, false);

  registration.dispose();
  assert.equal(registration.controller.signal.aborted, true);
  assert.deepEqual([...registered.keys()], []);
  registration.dispose();
});

test("live tool updates do not abort an in-flight WebMCP call", async () => {
  const registered = new Map();
  const signals = new Map();
  const context = {
    registerTool(tool, options) {
      registered.set(tool.name, tool);
      signals.set(tool.name, options.signal);
      options.signal.addEventListener("abort", () => registered.delete(tool.name), { once: true });
    },
  };
  let finish;
  const pendingResult = new Promise((resolve) => {
    finish = resolve;
  });
  const first = {
    ...definition("one"),
    execute: () => pendingResult,
  };
  const registry = createWebMcpToolRegistry([definition("one")], context);
  const registeredProxy = registered.get("one");
  registry.update([first]);
  assert.strictEqual(registered.get("one"), registeredProxy);
  assert.equal(signals.get("one").aborted, false);
  const call = registeredProxy.execute({});

  registry.update([definition("two")]);
  assert.equal(signals.get("one").aborted, false);
  assert.ok(registered.has("two"));
  assert.throws(() => registeredProxy.execute({}), /is not currently available/);
  finish(webMcpTextResult({ accepted: true }));
  assert.deepEqual(
    await call.then((result) => {
      assert.equal(signals.get("one").aborted, false);
      return result;
    }),
    webMcpTextResult({ accepted: true }),
  );
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(signals.get("one").aborted, true);
  assert.equal(registered.has("one"), false);

  registry.update([
    {
      ...definition("one"),
      execute: () => webMcpTextResult({ version: 2 }),
    },
    definition("two"),
  ]);
  const replacementProxy = registered.get("one");
  assert.notStrictEqual(replacementProxy, registeredProxy);
  assert.deepEqual(replacementProxy.execute({}), webMcpTextResult({ version: 2 }));
  assert.ok(registered.has("two"));

  registry.dispose();
  assert.equal(signals.get("one").aborted, true);
  assert.equal(signals.get("two").aborted, true);
  assert.deepEqual([...registered.keys()], []);
});

test("registration promises consume expected aborts and report other failures", () => {
  const rejectionHandlers = [];
  const context = {
    registerTool() {
      return {
        catch(handler) {
          rejectionHandlers.push(handler);
          return Promise.resolve();
        },
      };
    },
  };
  const errors = [];
  const originalError = console.error;
  console.error = (...args) => errors.push(args);
  try {
    const registration = registerWebMcpTools([definition("one"), definition("two")], context);
    assert.equal(rejectionHandlers.length, 2);

    registration.dispose();
    rejectionHandlers[0]({ name: "AbortError" });
    assert.deepEqual(errors, []);

    const failure = new Error("late registration failure");
    rejectionHandlers[1](failure);
    assert.deepEqual(errors, [["WebMCP tool registration failed.", failure]]);
  } finally {
    console.error = originalError;
  }
});

test("unsupported hosts and empty definitions register nothing", () => {
  assert.equal(registerWebMcpTools([definition()], null), null);
  assert.equal(
    registerWebMcpTools([], {
      registerTool() {
        assert.fail("empty registration must not call the host");
      },
    }),
    null,
  );
});

test("a partial registration is aborted when the host rejects a later tool", () => {
  let firstSignal;
  const context = {
    registerTool(tool, options) {
      if (tool.name === "two") throw new Error("duplicate tool");
      firstSignal = options.signal;
    },
  };

  assert.throws(
    () => registerWebMcpTools([definition("one"), definition("two")], context),
    /duplicate tool/,
  );
  assert.equal(firstSignal.aborted, true);
});

test("tool results are compact JSON and fail loudly beyond the bound", () => {
  assert.deepEqual(webMcpTextResult({ ok: true }), {
    content: [{ type: "text", text: '{"ok":true}' }],
  });
  assert.throws(
    () => webMcpTextResult({ text: "x".repeat(WEBMCP_RESULT_MAX_CHARS) }),
    /exceeds 1500 characters/,
  );
  assert.throws(() => webMcpTextResult(undefined), /not JSON serializable/);
});

test("project listing is bounded, searchable, and never exposes storage locators", () => {
  const cards = projectCardFixtures();
  const listed = listProjectsForWebMcp(cards, {});
  assert.equal(listed.total, 2);
  assert.equal(listed.matched, 2);
  assert.equal(listed.truncated, false);
  assert.deepEqual(listed.projects[1], {
    id: "project-2",
    name: "Remote calibration study",
    primary_question: "Does calibration transfer?",
    remote: true,
    reachable: false,
    revision: 8,
    attention_count: 0,
  });
  assert.equal(JSON.stringify(listed).includes("/secret/calibration"), false);
  assert.equal(JSON.stringify(listed).includes("SSH host details"), false);
  assert.deepEqual(
    listProjectsForWebMcp(cards, { query: "plasticity" }).projects.map((project) => project.id),
    ["project-1"],
  );
  const crowded = Array.from({ length: 10 }, (_, index) => ({
    ...cards[0],
    id: `project-${index}`,
  }));
  assert.equal(listProjectsForWebMcp(crowded, {}).projects.length, 8);
  assert.equal(listProjectsForWebMcp(crowded, {}).truncated, true);
  assert.throws(
    () => listProjectsForWebMcp(cards, { query: " " }),
    /query must be a non-blank string/,
  );
});

test("project opening revalidates the exact current id and uses the shared navigation owner", async () => {
  const cards = projectCardFixtures();
  const opened = [];
  const result = await openProjectFromIndex(cards, { project_id: "project-1" }, (projectId) => {
    opened.push(projectId);
    return true;
  });
  assert.deepEqual(opened, ["project-1"]);
  assert.equal(result.navigation_requested, true);
  assert.equal(result.target_view, "project");
  assert.equal(result.project.id, "project-1");
  await assert.rejects(
    openProjectFromIndex(cards, { project_id: "missing" }, () => assert.fail()),
    /not present in the current project index/,
  );
  await assert.rejects(
    openProjectFromIndex(cards, { project_id: "project-2" }, () => false),
    /requires the existing desktop access review/,
  );
});

test("the stable project-index tools read the current cards at invocation time", async () => {
  let cards = projectCardFixtures();
  const tools = projectIndexToolDefinitions(
    () => cards,
    () => true,
  );
  assert.deepEqual(
    tools.map((tool) => [
      tool.name,
      tool.annotations?.readOnlyHint,
      tool.annotations?.untrustedContentHint,
    ]),
    [
      ["rcp_list_projects", true, true],
      ["rcp_open_project", true, true],
    ],
  );
  assert.deepEqual(Object.keys(tools[0].inputSchema.properties), ["query"]);
  assert.deepEqual(tools[1].inputSchema.required, ["project_id"]);
  assert.equal(tools[1].inputSchema.additionalProperties, false);
  cards = cards.slice(1);
  const listed = JSON.parse((await tools[0].execute({})).content[0].text);
  assert.deepEqual(
    listed.projects.map((project) => project.id),
    ["project-2"],
  );
});

test("project overview returns bounded saved facts without an AI summary", () => {
  const overview = projectOverview(projectFixture());
  assert.deepEqual(overview.project, {
    id: "project-1",
    name: "Synthetic plasticity project",
    revision: 5,
    freshness: "fresh",
  });
  assert.equal(overview.primary_question.id, "rq-1");
  assert.deepEqual(overview.suggested_node_ids, ["blocker-1"]);
  assert.deepEqual(
    overview.recent.experiments.map((node) => node.id),
    ["exp-1"],
  );
  assert.equal("summary" in overview, false);
  assert.doesNotThrow(() => webMcpTextResult(overview));
});

test("node inspection returns exact saved content and direct relation identities", () => {
  const project = projectFixture();
  const inspected = inspectProjectNode(project, { node_id: "hyp-1" });
  assert.equal(inspected.node, project.graph.nodes["hyp-1"]);
  assert.equal(inspected.relation_count, 1);
  assert.equal(inspected.relations_truncated, false);
  assert.deepEqual(inspected.related_node_ids, ["rq-1"]);
  assert.equal(inspected.related_node_ids_truncated, false);
  assert.deepEqual(inspected.relations, [
    {
      edge_id: "edge-1",
      source_id: "hyp-1",
      target_id: "rq-1",
      relation: "answers",
      layer: "epistemic",
    },
  ]);
  assert.equal("experiment_control" in inspected, false);
});

test("node inspection bounds high-degree relation indices without losing the exact node", () => {
  const project = projectFixture();
  project.graph.edges = Object.fromEntries(
    Array.from({ length: 40 }, (_, index) => [
      `edge-${String(index).padStart(2, "0")}`,
      {
        id: `edge-${String(index).padStart(2, "0")}`,
        source: "hyp-1",
        target: `evidence-${index}`,
        relation: "supported_by",
        layer: "epistemic",
      },
    ]),
  );

  const inspected = inspectProjectNode(project, { node_id: "hyp-1" });

  assert.equal(inspected.node, project.graph.nodes["hyp-1"]);
  assert.equal(inspected.relation_count, 40);
  assert.equal(inspected.relations.length, 32);
  assert.equal(inspected.relations_truncated, true);
  assert.equal(inspected.related_node_ids.length, 32);
  assert.equal(inspected.related_node_ids_truncated, true);
});

test("node inspection refuses stale ids and includes backend Experiment control", () => {
  const project = projectFixture();
  assert.throws(
    () => inspectProjectNode(project, { node_id: "missing" }),
    /not present in the current project graph/,
  );
  assert.throws(() => inspectProjectNode(project, {}), /node_id must be a non-blank string/);
  assert.equal(
    inspectProjectNode(project, { node_id: "exp-1" }).experiment_control,
    project.experiment_control["exp-1"],
  );
});

test("the project read surface registers exactly the two confirmed tools", () => {
  const tools = projectReadToolDefinitions(projectFixture());
  assert.deepEqual(
    tools.map((tool) => [
      tool.name,
      tool.annotations?.readOnlyHint,
      tool.annotations?.untrustedContentHint,
    ]),
    [
      ["rcp_get_project_overview", true, true],
      ["rcp_inspect_node", true, true],
    ],
  );
  assert.equal(tools[1].inputSchema.additionalProperties, false);
  assert.deepEqual(tools[1].inputSchema.required, ["node_id"]);
});

test("artifact listing returns current task artifacts, kept state, and episode reports", () => {
  const project = projectFixture();
  const { tasks, episodes } = artifactFixtures();
  const listed = listProjectArtifacts(project, tasks, episodes, {});
  assert.equal(listed.total, 3);
  assert.equal(listed.truncated, false);
  assert.deepEqual(
    listed.artifacts.map((artifact) => artifact.viewer_id),
    ["report:episode-1", "task:task-2:artifact-2", "task:task-1:artifact-1"],
  );
  assert.equal(listed.artifacts[2].kept_filename, "calibration.html");
  assert.equal("viewer_url" in listed.artifacts[0], false);
});

test("artifact listing narrows to one exact current owner and rejects stale filters", () => {
  const project = projectFixture();
  const { tasks, episodes } = artifactFixtures();
  assert.deepEqual(
    listProjectArtifacts(project, tasks, episodes, { node_id: "hyp-1" }).artifacts.map(
      (artifact) => artifact.viewer_id,
    ),
    ["task:task-1:artifact-1"],
  );
  assert.deepEqual(
    listProjectArtifacts(project, tasks, episodes, { episode_id: "episode-1" }).artifacts.map(
      (artifact) => artifact.viewer_id,
    ),
    ["report:episode-1", "task:task-2:artifact-2"],
  );
  assert.throws(
    () => listProjectArtifacts(project, tasks, episodes, { task_id: "missing" }),
    /Task missing is not present/,
  );
  assert.throws(
    () => listProjectArtifacts(project, tasks, episodes, { chat_id: "missing" }),
    /Conversation missing is not present/,
  );
  assert.throws(
    () => listProjectArtifacts(project, tasks, episodes, { node_id: "hyp-1", chat_id: "chat-1" }),
    /at most one artifact filter/,
  );
});

test("artifact opening revalidates availability and opens only the existing viewer URL", async () => {
  const project = projectFixture();
  const { tasks, episodes } = artifactFixtures();
  const opened = [];
  assert.deepEqual(
    await openProjectArtifact(
      project,
      tasks,
      episodes,
      { viewer_id: "task:task-1:artifact-1" },
      (viewerUrl, contentUrl) => {
        opened.push([viewerUrl, contentUrl]);
        return true;
      },
    ),
    {
      project_id: "project-1",
      viewer_id: "task:task-1:artifact-1",
      kind: "task_artifact",
      opened: true,
    },
  );
  assert.deepEqual(opened, [
    [
      "/api/projects/project-1/tasks/task-1/artifacts/artifact-1/viewer",
      "/api/projects/project-1/tasks/task-1/artifacts/artifact-1/content",
    ],
  ]);
  await assert.rejects(
    openProjectArtifact(
      project,
      tasks,
      episodes,
      { viewer_id: "task:task-2:artifact-2" },
      () => true,
    ),
    /Artifact bytes expired/,
  );
  await assert.rejects(
    openProjectArtifact(project, tasks, episodes, { viewer_id: "report:episode-1" }, () => false),
    /could not be shown/,
  );
});

test("artifact tools expose only list and visual-open schemas", () => {
  const project = projectFixture();
  const { tasks, episodes } = artifactFixtures();
  const tools = projectArtifactToolDefinitions(project, tasks, episodes, () => true);
  assert.deepEqual(
    tools.map((tool) => [
      tool.name,
      tool.annotations?.readOnlyHint,
      tool.annotations?.untrustedContentHint,
    ]),
    [
      ["rcp_list_artifacts", true, true],
      ["rcp_open_artifact", true, true],
    ],
  );
  assert.deepEqual(Object.keys(tools[0].inputSchema.properties), [
    "node_id",
    "chat_id",
    "task_id",
    "episode_id",
  ]);
  assert.deepEqual(tools[1].inputSchema.required, ["viewer_id"]);
});

test("conversation inspection returns bounded transcript, latest result, and current Send options", async () => {
  const project = projectFixture();
  const { task, transcript } = conversationFixtures();
  const inspected = await inspectProjectConversation(
    project,
    [task],
    { chat_id: "chat-1" },
    async () => transcript,
  );
  assert.equal(inspected.chat_id, "chat-1");
  assert.equal(inspected.node_id, "hyp-1");
  assert.equal(inspected.transcript_truncated, false);
  assert.equal(inspected.recent_messages.length, 2);
  assert.deepEqual(inspected.latest_task, {
    task_id: "task-chat-1",
    status: "Finished",
    status_message: "Completed",
    active: false,
    awaiting_human: false,
    paused: false,
    failed: false,
    settled: true,
    runtime_used: "exec",
    final_answer: "The held-out curve remains stable.",
    graph_update: {
      status: "none",
      applied_revision: null,
      change_summary: [],
      proposal_ids: [],
      validation_messages: [],
      correction_rounds: 0,
      repairable: false,
    },
    artifacts: [
      {
        viewer_id: "task:task-chat-1:artifact-1",
        name: "calibration.html",
        media_type: "text/html",
        available: true,
        can_open: true,
        kept_filename: null,
      },
    ],
  });
  assert.equal(inspected.send_options.can_send, true);
  assert.equal(inspected.send_options.provider, "codex");
  assert.equal(inspected.send_options.configured_runtime, "exec");
  assert.equal(inspected.send_options.stable_session_id, "session-1");
  assert.deepEqual(
    inspected.send_options.workflows.map((item) => item.id),
    ["analyze"],
  );
  assert.deepEqual(
    inspected.send_options.skills.map((item) => item.id),
    ["plot"],
  );
  assert.deepEqual(
    inspected.send_options.provider_skills.items.map((item) => item.name),
    ["browser"],
  );
  assert.equal(inspected.send_options.workflows_total, 1);
  assert.equal(inspected.send_options.workflows_truncated, false);
  assert.equal(inspected.send_options.skills_total, 1);
  assert.equal(inspected.send_options.skills_truncated, false);
  assert.equal(inspected.send_options.run_truth_scope_truncated, false);
});

test("conversation inspection stays within its result budget at every declared list bound", async () => {
  const project = projectFixture();
  const workflows = Array.from({ length: 20 }, (_, index) => ({
    id: `workflow-${index}`,
    kind: "workflow",
    version: "1",
    label: `Workflow ${index} ${"w".repeat(180)}`,
    description: "d".repeat(500),
    dependencies: [],
  }));
  const skills = Array.from({ length: 20 }, (_, index) => ({
    id: `skill-${index}`,
    kind: "skill",
    version: "1",
    label: `Skill ${index} ${"s".repeat(180)}`,
    description: "d".repeat(500),
    dependencies: [],
  }));
  project.skill_catalog = [...workflows, ...skills];
  project.skill_defaults = {
    workflow_ids: workflows.map((item) => item.id),
    skill_ids: skills.map((item) => item.id),
  };
  project.provider_skill_inventories.local.codex.skills = Array.from(
    { length: 20 },
    (_, index) => ({
      name: `provider-skill-${index}-${"n".repeat(180)}`,
      label: `Provider skill ${index} ${"l".repeat(180)}`,
      description: "unused",
      enabled: true,
    }),
  );
  const { task, transcript } = conversationFixtures();
  task.result.messages = ["a".repeat(4_000)];
  task.result.graph_update.change_summary = Array.from({ length: 10 }, () => "c".repeat(500));
  task.result.graph_update.validation_messages = Array.from({ length: 10 }, () => "v".repeat(500));
  transcript.messages = Array.from({ length: 12 }, (_, index) => ({
    ...transcript.messages[index % 2],
    message_id: `message-${index}`,
    role: index % 2 ? "assistant" : "user",
    text: "m".repeat(2_000),
    timestamp: `2026-08-31T10:${String(index).padStart(2, "0")}:00Z`,
  }));
  transcript.message_count = transcript.messages.length;

  const tool = projectConversationToolDefinitions(project, [task], async () => transcript)[0];
  const result = await tool.execute({ chat_id: "chat-1" });
  const parsed = JSON.parse(result.content[0].text);

  assert.ok(result.content[0].text.length <= 12_000);
  assert.equal(parsed.transcript_truncated, true);
  assert.equal(parsed.recent_messages.length, 6);
  assert.ok(parsed.recent_messages.every((message) => message.text_truncated));
  assert.equal(parsed.send_options.workflows_total, 20);
  assert.equal(parsed.send_options.workflows.length, 4);
  assert.equal(parsed.send_options.workflows_truncated, true);
  assert.equal(parsed.send_options.skills_total, 20);
  assert.equal(parsed.send_options.skills.length, 4);
  assert.equal(parsed.send_options.skills_truncated, true);
  assert.equal(parsed.send_options.provider_skills.total, 20);
  assert.equal(parsed.send_options.provider_skills.items.length, 6);
  assert.equal(parsed.send_options.provider_skills.truncated, true);
});

test("conversation inspection names branch, active, and stale-transcript refusals", async () => {
  const project = projectFixture();
  const { task, transcript } = conversationFixtures();
  const branch = await inspectProjectConversation(
    project,
    [{ ...task, graph_target: { kind: "branch", branch_id: "branch-1" } }],
    { chat_id: "chat-1" },
    async () => transcript,
  );
  assert.equal(branch.send_options.can_send, false);
  assert.match(branch.send_options.refusal_reason, /branch conversation is read-only/);
  const active = await inspectProjectConversation(
    project,
    [{ ...task, active: true, settled: false, status_message: "Running provider turn" }],
    { chat_id: "chat-1" },
    async () => transcript,
  );
  assert.equal(active.send_options.refusal_reason, "Running provider turn");
  await assert.rejects(
    inspectProjectConversation(project, [task], { chat_id: "chat-1" }, async () => ({
      ...transcript,
      chat_id: "different",
    })),
    /mismatched transcript/,
  );
});

test("the conversation inspection tool has one exact read-only schema", async () => {
  const project = projectFixture();
  const { task, transcript } = conversationFixtures();
  const tools = projectConversationToolDefinitions(project, [task], async () => transcript);
  assert.deepEqual(
    tools.map((tool) => [
      tool.name,
      tool.annotations?.readOnlyHint,
      tool.annotations?.untrustedContentHint,
    ]),
    [["rcp_inspect_conversation", true, true]],
  );
  assert.deepEqual(tools[0].inputSchema.required, ["chat_id"]);
  const result = await tools[0].execute({ chat_id: "chat-1" });
  assert.match(result.content[0].text, /"chat_id":"chat-1"/);
});

test("conversation Send resumes the exact saved route and returns after durable task acceptance", async () => {
  const project = projectFixture();
  const { task, transcript } = conversationFixtures();
  const created = [];
  const submissions = [];
  const receipt = await sendProjectConversationMessage(
    project,
    [task],
    {
      chat_id: "chat-1",
      message: "  Compare the next replicate.  ",
      mode: "work",
      workflow_ids: ["analyze"],
      skill_ids: ["plot"],
      provider_skill_names: ["browser"],
    },
    async () => transcript,
    false,
    (kind, node) => {
      created.push([kind, node]);
      return "new-chat";
    },
    async (submission) => {
      submissions.push(submission);
      return {
        operation_id: "task-chat-2",
        status_label: "Queued",
        active: true,
        queued: true,
      };
    },
  );
  assert.deepEqual(created, []);
  assert.deepEqual(submissions, [
    {
      kind: "node_chat",
      config: { provider: "codex", model: "", reasoning: "high", run_on: "local" },
      runTruthScope: ["plasticity"],
      nodeId: "hyp-1",
      message: "Compare the next replicate.",
      chatId: "chat-1",
      sessionId: "session-1",
      mode: "work",
      skills: { workflow_ids: ["analyze"], skill_ids: ["plot"] },
      providerSkillNames: ["browser"],
    },
  ]);
  assert.deepEqual(receipt, {
    project_id: "project-1",
    chat_id: "chat-1",
    task_id: "task-chat-2",
    kind: "node_chat",
    node_id: "hyp-1",
    mode: "work",
    accepted: true,
    status: "Queued",
    active: true,
    queued: true,
  });
});

test("conversation Send creates only one fresh project or node conversation after validation", async () => {
  const project = projectFixture();
  const created = [];
  const submissions = [];
  const send = (input) =>
    sendProjectConversationMessage(
      project,
      [],
      input,
      async () => {
        throw new Error("fresh Send must not load a transcript");
      },
      false,
      (kind, node) => {
        const chatId = `new-${created.length + 1}`;
        created.push([kind, node?.id ?? null, chatId]);
        return chatId;
      },
      async (submission) => {
        submissions.push(submission);
        return {
          operation_id: `task-${submissions.length}`,
          status_label: "Queued",
          active: true,
          queued: true,
        };
      },
    );
  await send({ message: "Discuss the project.", mode: "discuss" });
  await send({ message: "Work on this node.", mode: "work", node_id: "hyp-1" });
  assert.deepEqual(created, [
    ["project_chat", null, "new-1"],
    ["node_chat", "hyp-1", "new-2"],
  ]);
  assert.deepEqual(
    submissions.map((item) => [item.kind, item.nodeId, item.chatId, item.sessionId]),
    [
      ["project_chat", null, "new-1", null],
      ["node_chat", "hyp-1", "new-2", null],
    ],
  );
});

test("conversation Send fails before dispatch for ambiguous targets, busy state, and disabled skills", async () => {
  const project = projectFixture();
  const { task, transcript } = conversationFixtures();
  const unreachable = () => {
    throw new Error("dispatch must not be reached");
  };
  await assert.rejects(
    sendProjectConversationMessage(
      project,
      [task],
      {
        chat_id: "chat-1",
        node_id: "hyp-1",
        message: "Ambiguous.",
        mode: "discuss",
      },
      async () => transcript,
      false,
      unreachable,
      unreachable,
    ),
    /cannot be supplied together/,
  );
  await assert.rejects(
    sendProjectConversationMessage(
      project,
      [{ ...task, active: true, settled: false, status_message: "Still running" }],
      { chat_id: "chat-1", message: "Too soon.", mode: "discuss" },
      async () => transcript,
      false,
      unreachable,
      unreachable,
    ),
    /Still running/,
  );
  await assert.rejects(
    sendProjectConversationMessage(
      project,
      [],
      { message: "Use it.", mode: "work", skill_ids: ["not-enabled"] },
      async () => transcript,
      false,
      unreachable,
      unreachable,
    ),
    /Unknown or disabled RCP skill ids/,
  );
});

test("the one Send tool exposes only semantic target, mode, message, and skill inputs", () => {
  const project = projectFixture();
  const tools = projectConversationSendToolDefinitions(
    project,
    [],
    async () => conversationFixtures().transcript,
    false,
    () => "chat-1",
    async () => ({ operation_id: "task-1" }),
  );
  assert.deepEqual(
    tools.map((tool) => [tool.name, tool.annotations?.readOnlyHint]),
    [["rcp_send_conversation_message", false]],
  );
  assert.deepEqual(Object.keys(tools[0].inputSchema.properties), [
    "message",
    "mode",
    "chat_id",
    "node_id",
    "workflow_ids",
    "skill_ids",
    "provider_skill_names",
  ]);
  assert.deepEqual(tools[0].inputSchema.required, ["message", "mode"]);
  assert.equal("provider" in tools[0].inputSchema.properties, false);
  assert.equal("attachment_ids" in tools[0].inputSchema.properties, false);
});

test("Experiment inspection compacts backend decisions and scopes work and watchers exactly", () => {
  const project = projectFixture();
  project.experiment_control["exp-1"] = {
    can_start: true,
    can_stop: false,
    reasons: [],
    can_open_report: false,
    report_episode_id: null,
  };
  const { task } = conversationFixtures();
  const experimentTask = {
    ...task,
    operation_id: "experiment-task-1",
    request: {
      control_node_id: "exp-1",
      control_episode_id: "episode-1",
      control_invocation: 1,
      patch_kind: "experiment_loop",
    },
    episode_id: "episode-1",
  };
  const watcher = {
    watcher_id: "watcher-1",
    continuation: { control_node_id: "exp-1" },
    episode_id: "episode-1",
    status: "active",
    created_at: "2026-08-31T10:02:00Z",
    completed_at: null,
    stop_reason: null,
    check_command: "true",
    last_error: null,
  };
  const inspected = inspectProjectExperiment(project, [experimentTask, task], [watcher], {
    experiment_id: "exp-1",
  });
  assert.equal(inspected.experiment.id, "exp-1");
  assert.equal(inspected.experiment.objective, undefined);
  assert.equal(inspected.control.can_start, true);
  assert.deepEqual(inspected.control.reasons, []);
  assert.equal(inspected.start_available, true);
  assert.deepEqual(
    inspected.tasks.map((item) => item.task_id),
    ["experiment-task-1"],
  );
  assert.deepEqual(inspected.watchers, [
    {
      watcher_id: "watcher-1",
      episode_id: "episode-1",
      status: "active",
      kind: "external",
      next_check_at: null,
      stop_reason: null,
      last_error: null,
    },
  ]);
  assert.ok(JSON.stringify(inspected).length <= 4_000);
  assert.equal(
    inspectProjectExperiment(project, [experimentTask], [], { experiment_id: "exp-1" }, true)
      .page_start_refusal,
    "Another task start is already being submitted.",
  );
});

test("Experiment Start revalidates the exact node and returns durable task identity", async () => {
  const project = projectFixture();
  const calls = [];
  const receipt = await startProjectExperiment(
    project,
    { experiment_id: "exp-1" },
    async (node) => {
      calls.push(node.id);
      return {
        operation_id: "experiment-task-1",
        episode_id: "episode-1",
        request: {},
        status_label: "Queued",
        active: true,
        queued: true,
      };
    },
  );
  assert.deepEqual(calls, ["exp-1"]);
  assert.deepEqual(receipt, {
    project_id: "project-1",
    experiment_id: "exp-1",
    task_id: "experiment-task-1",
    episode_id: "episode-1",
    accepted: true,
    status: "Queued",
    active: true,
    queued: true,
  });
  await assert.rejects(
    startProjectExperiment(project, { experiment_id: "hyp-1" }, async () => {
      throw new Error("must not dispatch");
    }),
    /Experiment hyp-1 is not present/,
  );
});

test("Experiment tools expose one exact read and one exact Start schema", () => {
  const project = projectFixture();
  const tools = projectExperimentToolDefinitions(
    project,
    [],
    [],
    false,
    false,
    false,
    false,
    async () => ({ operation_id: "task-1" }),
  );
  assert.deepEqual(
    tools.map((tool) => [
      tool.name,
      tool.annotations?.readOnlyHint,
      tool.annotations?.untrustedContentHint,
    ]),
    [
      ["rcp_inspect_experiment", true, true],
      ["rcp_start_experiment", false, undefined],
    ],
  );
  assert.deepEqual(tools[0].inputSchema, tools[1].inputSchema);
  assert.deepEqual(tools[0].inputSchema.required, ["experiment_id"]);
});

test("Experiment Start is discoverable only while an exact Start can succeed or is returning", () => {
  const project = projectFixture();
  project.experiment_control["exp-1"].can_start = false;
  const definitions = (
    taskStartPending,
    startReturning = false,
    mutationsDisabled = false,
    startRequiresSync = false,
  ) =>
    projectExperimentToolDefinitions(
      project,
      [],
      [],
      taskStartPending,
      startReturning,
      mutationsDisabled,
      startRequiresSync,
      async () => ({ operation_id: "task-1" }),
    ).map((tool) => tool.name);
  assert.deepEqual(definitions(false), ["rcp_inspect_experiment"]);
  assert.deepEqual(definitions(true), ["rcp_inspect_experiment"]);
  assert.deepEqual(definitions(true, true), ["rcp_inspect_experiment", "rcp_start_experiment"]);
  project.experiment_control["exp-1"].can_start = true;
  assert.deepEqual(definitions(false), ["rcp_inspect_experiment", "rcp_start_experiment"]);
  assert.deepEqual(definitions(false, false, true), ["rcp_inspect_experiment"]);
  assert.deepEqual(definitions(false, false, false, true), ["rcp_inspect_experiment"]);
});

test("Experiment Stop validates the exact live episode and requests only the graceful fence", async () => {
  const project = projectFixture();
  project.experiment_control["exp-1"] = {
    episode_id: "episode-1",
    can_stop: true,
    reasons: [],
  };
  const calls = [];
  const receipt = await stopProjectExperimentEpisode(
    project,
    { experiment_id: "exp-1", episode_id: "episode-1" },
    async (experimentId, episodeId) => calls.push([experimentId, episodeId]),
  );
  assert.deepEqual(calls, [["exp-1", "episode-1"]]);
  assert.deepEqual(receipt, {
    project_id: "project-1",
    experiment_id: "exp-1",
    episode_id: "episode-1",
    stop_requested: true,
    graceful: true,
  });
  await assert.rejects(
    stopProjectExperimentEpisode(
      project,
      { experiment_id: "exp-1", episode_id: "episode-stale" },
      async () => {
        throw new Error("must not dispatch");
      },
    ),
    /is not the live episode/,
  );
});

test("Experiment Stop refuses backend can_stop false and hides until available", async () => {
  const project = projectFixture();
  project.experiment_control["exp-1"] = {
    episode_id: "episode-1",
    can_stop: false,
    reasons: ["The episode is already stopping."],
  };
  await assert.rejects(
    stopProjectExperimentEpisode(
      project,
      { experiment_id: "exp-1", episode_id: "episode-1" },
      async () => {
        throw new Error("must not dispatch");
      },
    ),
    /already stopping/,
  );
  assert.deepEqual(
    projectExperimentStopToolDefinitions(project, async () => undefined),
    [],
  );
  const tools = projectExperimentStopToolDefinitions(project, async () => undefined, true);
  assert.deepEqual(
    tools.map((tool) => [tool.name, tool.annotations?.readOnlyHint]),
    [["rcp_stop_episode", false]],
  );
  assert.deepEqual(tools[0].inputSchema.required, ["experiment_id", "episode_id"]);
});

function evalToolDefinitions(state) {
  if (state === "landing") {
    return projectIndexToolDefinitions(
      () => projectCardFixtures(),
      () => true,
    );
  }
  const project = projectFixture();
  project.experiment_control["exp-1"] = {
    can_start: state === "project_ready",
    can_stop: state === "project_live",
  };
  const { task, transcript } = conversationFixtures();
  const { tasks, episodes } = artifactFixtures();
  return [
    ...projectReadToolDefinitions(project),
    ...projectArtifactToolDefinitions(project, tasks, episodes, () => true),
    ...projectConversationToolDefinitions(project, [task], async () => transcript),
    ...projectConversationSendToolDefinitions(
      project,
      [task],
      async () => transcript,
      false,
      () => "chat-1",
      async () => ({ operation_id: "task-next" }),
    ),
    ...projectExperimentToolDefinitions(project, [], [], false, false, false, false, async () => ({
      operation_id: "experiment-task",
    })),
    ...projectExperimentStopToolDefinitions(project, async () => undefined),
  ];
}

function assertEvalArguments(tool, input, label) {
  assert.equal(input !== null && typeof input === "object" && !Array.isArray(input), true, label);
  const properties = tool.inputSchema.properties ?? {};
  for (const name of Object.keys(input)) {
    assert.equal(name in properties, true, `${label}: unexpected argument ${name}`);
    const schema = properties[name];
    const value = input[name];
    if (schema.type === "string") {
      assert.equal(typeof value, "string", `${label}: ${name} must be a string`);
      if (schema.minLength) assert.ok(value.length >= schema.minLength, label);
      if (schema.maxLength) assert.ok(value.length <= schema.maxLength, label);
      if (schema.enum) assert.ok(schema.enum.includes(value), label);
    }
    if (schema.type === "array")
      assert.ok(Array.isArray(value), `${label}: ${name} must be an array`);
  }
  for (const name of tool.inputSchema.required ?? []) {
    assert.equal(name in input, true, `${label}: missing required argument ${name}`);
  }
}

test("all WebMCP metadata stays descriptive and within model-facing budgets", () => {
  const definitions = ["landing", "project_ready", "project_live", "project_completed"].flatMap(
    evalToolDefinitions,
  );
  for (const tool of definitions) {
    assert.ok(tool.name.length <= 30, `${tool.name}: tool name exceeds 30 characters`);
    assert.ok(tool.description.length > 0 && tool.description.length <= 500, tool.name);
    for (const [name, schema] of Object.entries(tool.inputSchema.properties ?? {})) {
      assert.ok(name.length <= 30, `${tool.name}.${name}: parameter name exceeds 30 characters`);
      assert.equal(
        typeof schema.description,
        "string",
        `${tool.name}.${name}: missing description`,
      );
      assert.ok(
        schema.description.length > 0 && schema.description.length <= 150,
        `${tool.name}.${name}: description exceeds 150 characters`,
      );
    }
  }
});

test("the challenge eval set matches live state inventories and current schemas", async () => {
  const evals = JSON.parse(
    await readFile(new URL("../../challenge/evals/webmcp_journeys.json", import.meta.url), "utf8"),
  );
  assert.equal(evals.schemaVersion, 1);
  assert.ok(evals.cases.length >= 8);
  const definitionsByState = new Map();
  for (const [state, expectedNames] of Object.entries(evals.states)) {
    const definitions = evalToolDefinitions(state);
    assert.deepEqual(
      definitions.map((tool) => tool.name),
      expectedNames,
      `${state} tool inventory drifted`,
    );
    definitionsByState.set(state, new Map(definitions.map((tool) => [tool.name, tool])));
  }
  const validateExpectedCalls = (state, calls, label) => {
    const definitions = definitionsByState.get(state);
    assert.ok(definitions, `${label}: unknown state ${state}`);
    for (const [index, call] of calls.entries()) {
      const tool = definitions.get(call.functionName);
      assert.ok(tool, `${label}: ${call.functionName} is unavailable in ${state}`);
      assertEvalArguments(tool, call.arguments, `${label} call ${index + 1}`);
    }
  };
  for (const evalCase of evals.cases) {
    assert.ok(evalCase.messages.some((message) => message.role === "user" && message.content));
    validateExpectedCalls(evalCase.state, evalCase.expectedCall, evalCase.id);
  }
  for (const journey of evals.journeys) {
    assert.ok(journey.steps.length >= 3);
    for (const [index, step] of journey.steps.entries()) {
      validateExpectedCalls(step.state, step.expectedCall, `${journey.id} step ${index + 1}`);
    }
  }
});
