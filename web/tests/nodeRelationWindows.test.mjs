import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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
const { relatedNodeWindowAction } = await server.ssrLoadModule("/src/App.tsx");
const { DetailDrawer } = await server.ssrLoadModule("/src/components/DetailDrawer.tsx");

after(() => server.close());

function node(id, title) {
  return {
    id,
    type: "hypothesis",
    title,
    statement: `${title} statement`,
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
    extension_fields: {},
  };
}

test("related-node clicks target the other slot and focus an existing node", () => {
  assert.deepEqual(relatedNodeWindowAction("original", "beta", "alpha", null), {
    kind: "open",
    slot: "companion",
  });
  assert.deepEqual(relatedNodeWindowAction("companion", "gamma", "alpha", "beta"), {
    kind: "open",
    slot: "original",
  });
  assert.deepEqual(relatedNodeWindowAction("original", "beta", "alpha", "beta"), {
    kind: "focus",
    slot: "companion",
  });
  assert.deepEqual(relatedNodeWindowAction("companion", "alpha", "alpha", "beta"), {
    kind: "focus",
    slot: "original",
  });
});

test("App renders stable original and companion detail slots", async () => {
  const source = await readFile(new URL("../src/App.tsx", import.meta.url), "utf8");
  assert.match(source, /slot: "original" as const, selected: selectedNode/);
  assert.match(source, /slot: "companion" as const, selected: companionNode/);
  assert.match(source, /detailSlot=\{slot\}/);
  assert.match(source, /focusRequestToken=\{detailFocusTokens\[slot\]\}/);
});

test("DetailDrawer renders the relation map with slot-unique dialog labels", () => {
  const previousWindow = globalThis.window;
  globalThis.window = { innerWidth: 1200, innerHeight: 800 };
  try {
    const focus = node("focus", "Focused node");
    const peer = node("peer", "Related node");
    const commonProps = {
      node: focus,
      edges: [
        {
          id: "edge-1",
          source: "peer",
          target: "focus",
          relation: "supports",
          layer: "epistemic",
          explanation: "",
        },
      ],
      allNodes: { focus, peer },
      glossaryIndex: { entriesByInitial: new Map() },
      beliefTransitions: [],
      validationMessages: [],
      ontology: { types: [], fields: [], relations: [] },
      onClose() {},
      onDock() {},
      onBeginEdit() {},
      onStanding() {},
      onStage() {},
      onOpenChat() {},
      onOpenRelatedNode() {},
      onSelectNode() {},
    };
    const original = renderToStaticMarkup(
      React.createElement(DetailDrawer, { ...commonProps, detailSlot: "original" }),
    );
    const companion = renderToStaticMarkup(
      React.createElement(DetailDrawer, { ...commonProps, detailSlot: "companion" }),
    );

    assert.match(original, /aria-labelledby="drawer-title-original-focus"/);
    assert.match(companion, /aria-labelledby="drawer-title-companion-focus"/);
    assert.match(original, /class="relation-map relation-map-compact"/);
    assert.match(original, /Incoming relations/);
    assert.match(original, /Supports/);
    assert.doesNotMatch(original, /relation-row|Open DAG focused/);
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
});
