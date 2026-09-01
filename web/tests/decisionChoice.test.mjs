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
const { DetailDrawer } = await server.ssrLoadModule("/src/components/DetailDrawer.tsx");
const { presentNode } = await server.ssrLoadModule("/src/nodePresentation.ts");
const { changedNodeFields, editableNodeFields, nodeEditDraft } =
  await server.ssrLoadModule("/src/nodeEditing.ts");

after(() => server.close());

const decision = {
  id: "dec/resource",
  type: "decision",
  title: "Choose resource level",
  question: "Which resource level should the experiment use?",
  options: ["Small", "Medium", "Medium", "Large"],
  selected_option: "Medium",
  status: "decided",
  rationale: "Balance iteration speed against signal quality.",
  consequences: "This choice governs the next experiment.",
  standing: "accepted",
  created_rev: 2,
  updated_rev: 4,
  source_refs: [],
  extension_fields: {},
  draft_touched: true,
};

const commonProps = {
  node: decision,
  edges: [],
  allNodes: { [decision.id]: decision },
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
  onDecisionChoice() {},
  onOpenChat() {},
  onOpenRelatedNode() {},
  onSelectNode() {},
};

function renderDrawer(props = {}) {
  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1440, innerHeight: 900 };
  try {
    return renderToStaticMarkup(React.createElement(DetailDrawer, { ...commonProps, ...props }));
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
}

test("Decision detail renders one accessible staged ballot above Context", () => {
  const html = renderDrawer({
    decisionChoiceStaged: true,
  });

  assert.match(html, /<section class="decision-choice-section">/);
  assert.match(html, /<span class="decision-choice-status decided">Decided · staged<\/span>/);
  assert.match(
    html,
    /<legend id="decision-question-dec\/resource">Which resource level should the experiment use\?<\/legend>/,
  );
  assert.equal(html.match(/type="radio"/g)?.length, 3);
  assert.equal(html.match(/checked=""/g)?.length, 1);
  assert.match(
    html,
    /class="decision-choice-option selected staged"[\s\S]*value="Medium"[\s\S]*Staged selection/,
  );
  assert.doesNotMatch(html, /pending proposals? target this decision/);
  assert.ok(html.indexOf("decision-choice-section") < html.indexOf("node-context"));
  assert.equal(html.match(/>Medium</g)?.length, 1);
  assert.doesNotMatch(html, /Options considered|Selected option/);

  const contextKeys = presentNode(decision).context.map((item) => item.key);
  assert.deepEqual(contextKeys, ["rationale", "consequences"]);
});

test("a behind node opens its staged editor with reversible incoming field controls", () => {
  const canonicalNode = {
    ...decision,
    title: "Incoming canonical title",
    updated_rev: 5,
  };
  const stagedNode = {
    ...canonicalNode,
    title: "My staged title",
  };
  const html = renderDrawer({
    node: stagedNode,
    allNodes: { [stagedNode.id]: stagedNode },
    canonicalNode,
    canonicalStanding: canonicalNode.standing,
    behind: true,
    draftNodeChange: {
      base_updated_rev: 4,
      changes: { title: "My staged title" },
    },
    onApplyField() {},
  });

  assert.match(html, /class="detail-drawer node-detail-drawer[^"]* draft-behind"/);
  assert.match(html, /class="node-draft-behind">behind<\/span>/);
  assert.match(html, /value="My staged title"/);
  assert.match(html, /class="node-edit-incoming-value">Incoming canonical title<\/span>/);
  assert.match(html, />Apply<\/button>/);
});

test("Decision editor exposes queue status only, including the ready to open path", () => {
  const readyDecision = { ...decision, status: "ready", selected_option: null };
  const statusField = editableNodeFields(readyDecision).find((field) => field.key === "status");

  assert.deepEqual(statusField?.options, [
    { value: "open", label: "Open" },
    { value: "ready", label: "Ready" },
    { value: "revisit", label: "Revisit" },
  ]);
  assert.equal(
    editableNodeFields(readyDecision).some((field) => field.key === "selected_option"),
    false,
  );

  const draft = nodeEditDraft(readyDecision);
  assert.equal(draft.status, "ready");
  assert.deepEqual(changedNodeFields(readyDecision, { ...draft, status: "open" }), {
    status: "open",
  });
  assert.equal(
    statusField.options.some((option) => option.value === "decided"),
    false,
  );
});

test("Decision choices disable for superseded, globally disabled, and removal-staged records", () => {
  for (const props of [
    { node: { ...decision, status: "superseded" } },
    { mutationsDisabled: true },
    { stagedForRemoval: true },
  ]) {
    const html = renderDrawer(props);
    assert.match(html, /<fieldset disabled="">/);
  }
});
