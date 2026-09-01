import assert from "node:assert/strict";
import { after, test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

import { buildGlossaryIndex } from "../src/glossary.ts";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { DetailDrawer } = await server.ssrLoadModule("/src/components/DetailDrawer.tsx");
const { ProposalJudgmentSection } = await server.ssrLoadModule("/src/components/AttentionRail.tsx");
const { NodeChat } = await server.ssrLoadModule("/src/components/NodeChat.tsx");

after(() => server.close());

const glossaryIndex = buildGlossaryIndex({
  plasticity: {
    term: "plasticity",
    plain_definition: "Capacity to continue adapting.",
  },
  "plasticity-loss": {
    term: "plasticity loss",
    plain_definition: "A reduction in that capacity.",
  },
});

test("node prose renders focusable inline definitions without a detached glossary section", () => {
  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1200, innerHeight: 900 };
  const node = {
    id: "hyp/plasticity",
    type: "hypothesis",
    title: "Plasticity claim",
    statement: "Plasticity loss is measurable.",
    motivation: "Plasticity matters for future learning.",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
  let html;
  try {
    html = renderToStaticMarkup(
      React.createElement(DetailDrawer, {
        node,
        edges: [],
        allNodes: { [node.id]: node },
        glossaryIndex,
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
      }),
    );
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }

  assert.match(html, /<dfn[^>]*tabindex="0"[^>]*>Plasticity loss<\/dfn>/);
  assert.match(html, /data-definition="A reduction in that capacity\."/);
  assert.doesNotMatch(html, /<dfn[^>]*\stitle=/);
  assert.match(html, /<h3>Context<\/h3>/);
  assert.doesNotMatch(html, /Terms used here|node-glossary/);
});

test("proposal prose uses the shared inline glossary treatment", () => {
  const html = renderToStaticMarkup(
    React.createElement(ProposalJudgmentSection, {
      proposals: [
        {
          id: "prop/plasticity",
          title: "Plasticity decision",
          base_rev: 3,
          card: {
            situation_cold: "Plasticity loss remains plausible.",
            why_human_now: "Choose the next measurement.",
            consequences: "The plasticity test becomes active.",
            decision_needed: "Approve or reject.",
          },
          ops: [
            {
              op: "update_nodes",
              nodes: [{ id: "dec/plasticity-test", changes: { selected_option: "matched" } }],
            },
          ],
        },
      ],
      glossaryIndex,
      draft: null,
      onDecision() {},
    }),
  );

  assert.equal(html.match(/<dfn/g)?.length, 3);
  assert.match(html, /<h3><dfn[^>]*>Plasticity<\/dfn> decision<\/h3>/);
  assert.match(html, /<dfn[^>]*>Plasticity loss<\/dfn> remains plausible\./);
  assert.match(html, /Proposed action/);
  assert.match(html, /Approve or reject\./);
});

test("node chat passes the prebuilt index into Markdown answers", () => {
  const profile = {
    provider: "codex",
    model: "gpt-codex",
    reasoning: "medium",
    run_on: "local",
    permissions: {},
  };
  const project = {
    id: "project",
    name: "Project",
    revision: 3,
    agent_profiles: { node_chat: profile, project_chat: profile },
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
  const html = renderToStaticMarkup(
    React.createElement(NodeChat, {
      project,
      glossaryIndex,
      runScope: ["repo"],
      tasks: [],
      activeTask: null,
      historyMessages: [
        {
          message_id: "answer",
          operation_id: "operation",
          role: "assistant",
          text: "Plasticity loss is the result.",
          timestamp: "2026-08-03T10:00:00Z",
          native_session_id: null,
          provider: "codex",
          model: "gpt-codex",
          reasoning: "medium",
          execution_machine: "local",
          applied_revision: null,
          mode: "discuss",
          graph_update: null,
          trigger: "human",
        },
      ],
      chatId: "chat",
      onStartTask() {},
      onInspectTask() {},
      onOpenInbox() {},
      onRepairGraphUpdate() {},
      onClose() {},
    }),
  );

  assert.match(html, /<dfn[^>]*>Plasticity loss<\/dfn> is the result\./);
});
