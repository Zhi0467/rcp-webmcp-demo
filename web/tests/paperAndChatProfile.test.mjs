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
const { NodeChat } = await server.ssrLoadModule("/src/components/NodeChat.tsx");
const { loadPaperSnapshot, PaperWorkspace, swapPaperBuffers } = await server.ssrLoadModule(
  "/src/views/PaperWorkspace.tsx",
);

after(() => server.close());

const profile = (provider) => ({
  provider,
  model: provider === "codex" ? "gpt-codex" : "claude-opus",
  reasoning: "medium",
  run_on: "local",
  permissions: {},
});

const readiness = (provider, label) => ({
  provider,
  label,
  installed: true,
  authenticated: true,
  binary_path: `/usr/bin/${provider}`,
  path_state: "resolved",
  models: [],
});

const project = {
  id: "project",
  name: "Project",
  revision: 3,
  agent_profiles: {
    node_chat: profile("codex"),
    project_chat: profile("codex"),
    paper_coach: profile("codex"),
  },
  provider_readiness: {
    local: {
      codex: readiness("codex", "Codex"),
      claude: readiness("claude", "Claude"),
    },
  },
  repositories: [{ alias: "repo", machine: "local", path: "/repo" }],
  project_truth_scope: ["repo"],
  state_repository: "repo",
  machines: [{ alias: "local", host: null }],
};

const chatProps = {
  project,
  node: null,
  runScope: ["repo"],
  tasks: [],
  activeTask: null,
  chatId: "chat",
  onStartTask() {},
  onInspectTask() {},
  onOpenInbox() {},
  onRepairGraphUpdate() {},
  onNewSession() {},
  onClose() {},
};

test("chat shows passive provider identity, New session, and a self-labelling scope picker", () => {
  const fresh = renderToStaticMarkup(React.createElement(NodeChat, chatProps));
  assert.match(fresh, /aria-label="Chat provider: Codex"[^>]*>Codex</);
  assert.match(fresh, /class="chat-new-session"[\s\S]*?New session/);
  assert.match(fresh, /class="eyebrow">Run reads</);
  assert.doesNotMatch(fresh, /Raw truth inputs/);
  assert.doesNotMatch(fresh, /agent-config-summary/);
  assert.doesNotMatch(fresh, />Model</);
  assert.doesNotMatch(fresh, />Reasoning</);
  assert.doesNotMatch(fresh, />Run on/);

  const resumed = renderToStaticMarkup(
    React.createElement(NodeChat, {
      ...chatProps,
      historyMessages: [
        {
          message_id: "answer",
          operation_id: "operation",
          role: "assistant",
          text: "Answer",
          timestamp: "2026-08-03T10:00:00Z",
          native_session_id: "native-session",
          provider: "claude",
          model: "claude-opus",
          reasoning: "high",
          execution_machine: "local",
          applied_revision: null,
          mode: "discuss",
          graph_update: null,
          trigger: "human",
        },
      ],
    }),
  );
  assert.match(resumed, /aria-label="Chat provider: Claude"[^>]*>Claude</);
});

test("paper preview renders unsaved Markdown in the editor pane and keeps status", () => {
  const previousStorage = globalThis.localStorage;
  const storageKeys = [];
  globalThis.localStorage = {
    getItem(key) {
      storageKeys.push(key);
      return key === "rcp:paper-view:project" ? "preview" : null;
    },
  };
  let html;
  try {
    html = renderToStaticMarkup(
      React.createElement(PaperWorkspace, {
        apiBase: "/api/projects/project",
        project,
        initialPaper: {
          content: "## Methods\n\nDraft words",
          sync_state: "synced",
          canonical_available: true,
        },
        tasks: [],
        activeTask: null,
        onStartTask() {},
        onPaperChange() {},
      }),
    );
  } finally {
    if (previousStorage === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = previousStorage;
  }

  assert.deepEqual(storageKeys, ["rcp:paper-view:project"]);
  assert.match(html, /aria-label="Paper view"/);
  assert.match(html, /aria-pressed="false"[^>]*>Write</);
  assert.match(html, /aria-pressed="true"[^>]*>Preview</);
  assert.match(html, /aria-label="Paper introduction preview"/);
  assert.match(html, /<h2>Methods<\/h2>/);
  assert.match(html, /4 words/);
  assert.match(html, /sync-state synced/);
  assert.doesNotMatch(html, /Paper introduction Markdown/);
  assert.match(html, /aria-label="Writing coach provider: Codex"[^>]*>Codex</);
  assert.doesNotMatch(html, /agent-config-summary/);
  assert.doesNotMatch(html, />Model</);
  assert.doesNotMatch(html, />Reasoning</);
  assert.doesNotMatch(html, />Run on/);
});

test("behind paper exposes one reversible Incoming swap without destructive controls", () => {
  const previousStorage = globalThis.localStorage;
  globalThis.localStorage = {
    getItem(key) {
      return key === "rcp:paper-view:project" ? "incoming" : null;
    },
  };
  let html;
  try {
    html = renderToStaticMarkup(
      React.createElement(PaperWorkspace, {
        apiBase: "/api/projects/project",
        project,
        initialPaper: {
          content: "## Local draft\n\nTyped words",
          sync_state: "behind",
          base_hash: "old-base",
          canonical_hash: "incoming-hash",
          incoming_content: "## Incoming canonical\n\nRemote words",
          canonical_available: true,
        },
        tasks: [],
        activeTask: null,
        onStartTask() {},
        onPaperChange() {},
      }),
    );
  } finally {
    if (previousStorage === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = previousStorage;
  }

  assert.match(html, /aria-pressed="false"[^>]*>Write</);
  assert.match(html, /aria-pressed="false"[^>]*>Preview</);
  assert.match(html, /aria-pressed="true"[^>]*>Incoming</);
  assert.match(html, /aria-label="Swap editor and incoming introduction"[^>]*>Apply</);
  assert.match(html, /paper-markdown-preview chat-markdown/);
  assert.match(html, /<h2>Incoming canonical<\/h2>/);
  assert.match(html, /sync-state behind/);
  assert.doesNotMatch(html, /Use canonical|Overwrite canonical|conflict-banner/);

  const first = swapPaperBuffers("local", "canonical");
  assert.deepEqual(first, ["canonical", "local"]);
  assert.deepEqual(swapPaperBuffers(...first), ["local", "canonical"]);
});

test("paper freshness checks use the paper snapshot endpoint", async () => {
  const requested = [];
  const expected = { content: "draft", sync_state: "synced", canonical_available: true };
  const snapshot = await loadPaperSnapshot(async (path) => {
    requested.push(path);
    return expected;
  }, "/api/projects/project");

  assert.strictEqual(snapshot, expected);
  assert.deepEqual(requested, ["/api/projects/project/paper"]);
});
