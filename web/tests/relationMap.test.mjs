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
const {
  RelationMap,
  evidenceAssessmentPresentation,
  groupIncidentRelations,
  makeRelationModalBackgroundInert,
  relationOverlayHost,
  trapRelationModalTab,
} = await server.ssrLoadModule("/src/components/RelationMap.tsx");

after(() => server.close());

function node(id, title) {
  return {
    id,
    type: "hypothesis",
    title,
    statement: `${title} statement`,
    extension_fields: {},
    standing: "asserted",
    created_rev: 1,
    updated_rev: 1,
    source_refs: [],
  };
}

const nodes = {
  focus: node("focus", "Focused node"),
  alpha: node("alpha", "Alpha peer"),
  beta: node("beta", "Beta peer"),
};

test("groups peers deterministically while retaining every parallel edge", () => {
  const edges = [
    { id: "z", source: "focus", target: "beta", relation: "tests" },
    { id: "b", source: "alpha", target: "focus", relation: "supports" },
    { id: "a", source: "alpha", target: "focus", relation: "contradicts" },
    { id: "ignored", source: "alpha", target: "beta", relation: "supports" },
  ];

  const grouped = groupIncidentRelations("focus", nodes, edges);

  assert.deepEqual(
    grouped.incoming.map((group) => ({
      nodeId: group.nodeId,
      edges: group.edges.map((edge) => edge.id),
    })),
    [{ nodeId: "alpha", edges: ["a", "b"] }],
  );
  assert.deepEqual(
    grouped.outgoing.map((group) => ({
      nodeId: group.nodeId,
      edges: group.edges.map((edge) => edge.id),
    })),
    [{ nodeId: "beta", edges: ["z"] }],
  );
});

test("renders incoming peers above the focus, outgoing peers below, and edge warnings", () => {
  const incidentEdges = [
    {
      id: "incoming-support",
      source: "alpha",
      target: "focus",
      relation: "supports",
      layer: "epistemic",
      explanation: "",
    },
    {
      id: "incoming-contradiction",
      source: "alpha",
      target: "focus",
      relation: "contradicts",
      layer: "epistemic",
      explanation: "",
    },
    {
      id: "outgoing-test",
      source: "focus",
      target: "beta",
      relation: "tests",
      layer: "epistemic",
      explanation: "",
    },
  ];
  const html = renderToStaticMarkup(
    React.createElement(RelationMap, {
      focusedNode: nodes.focus,
      allNodes: nodes,
      incidentEdges,
      validationMessages: [
        {
          level: "flag",
          code: "relation-type-mismatch",
          message: "Relation does not match the ontology.",
          related_node_ids: [],
          related_edge_ids: ["incoming-support"],
        },
      ],
      onOpenNodeWindow() {},
    }),
  );

  const incomingAt = html.indexOf('aria-label="Incoming relations"');
  const focusAt = html.indexOf('class="relation-map-node is-focused"');
  const outgoingAt = html.indexOf('aria-label="Outgoing relations"');
  assert.ok(incomingAt >= 0 && incomingAt < focusAt);
  assert.ok(focusAt < outgoingAt);
  assert.equal((html.match(/Alpha peer/g) ?? []).length, 2); // label and aria-label, one card
  assert.match(html, /Supports/);
  assert.match(html, /Contradicts/);
  assert.match(html, /Relation does not match the ontology\./);
  assert.match(html, /Expand relation map for Focused node/);
});

test("renders claim-relative Evidence assessment separately and labels legacy unassessed edges", () => {
  const evidence = {
    ...node("ev/result", "Held-out result"),
    type: "evidence",
    observation: "The held-out score improved.",
    role: "result",
  };
  const decision = {
    ...node("dec/route", "Route choice"),
    type: "decision",
    question: "Which route?",
  };
  const allNodes = { ...nodes, [evidence.id]: evidence, [decision.id]: decision };
  const assessed = {
    id: "assessed-support",
    source: evidence.id,
    target: nodes.focus.id,
    relation: "supports",
    layer: "epistemic",
    explanation: "The result bears on the focused claim.",
    assessment: {
      relevance: "direct",
      weight: "strong",
      scope: "Shifted small-model regime",
      qualifications: ["The large model was not evaluated."],
    },
  };
  const legacy = {
    ...assessed,
    id: "legacy-weakening",
    relation: "weakens",
    assessment: null,
  };

  assert.deepEqual(evidenceAssessmentPresentation(assessed, allNodes), assessed.assessment);
  assert.equal(evidenceAssessmentPresentation(legacy, allNodes), "legacy");
  const html = renderToStaticMarkup(
    React.createElement(RelationMap, {
      focusedNode: nodes.focus,
      allNodes,
      incidentEdges: [assessed, legacy],
      validationMessages: [],
      onOpenNodeWindow() {},
    }),
  );

  assert.match(html, /Supports/);
  assert.match(html, /Assessment · Direct relevance · Strong weight/);
  assert.match(html, /Scope · Shifted small-model regime/);
  assert.match(html, /Qualifications · The large model was not evaluated\./);
  assert.match(html, /Legacy unassessed relation/);

  const nonApplicable = {
    ...assessed,
    id: "decision-information",
    target: decision.id,
    relation: "informs",
  };
  assert.equal(evidenceAssessmentPresentation(nonApplicable, allNodes), null);
  const actionHtml = renderToStaticMarkup(
    React.createElement(RelationMap, {
      focusedNode: evidence,
      allNodes,
      incidentEdges: [nonApplicable],
      validationMessages: [],
      onOpenNodeWindow() {},
    }),
  );
  assert.match(actionHtml, /Informs/);
  assert.doesNotMatch(actionHtml, /Evidence assessment|Legacy unassessed|Strong weight/);
});

test("server rendering does not access the document for the closed overlay", () => {
  assert.doesNotThrow(() =>
    renderToStaticMarkup(
      React.createElement(RelationMap, {
        focusedNode: nodes.focus,
        allNodes: nodes,
        incidentEdges: [],
        validationMessages: [],
        onOpenNodeWindow() {},
      }),
    ),
  );
});

test("expanded relation map uses the active fullscreen element as its portal host", () => {
  const body = {};
  const fullscreenElement = {};
  assert.equal(relationOverlayHost({ body, fullscreenElement: null }), body);
  assert.equal(relationOverlayHost({ body, fullscreenElement }), fullscreenElement);
});

test("modal Tab handling wraps focus and recaptures focus from outside", () => {
  const focused = [];
  const first = focusTarget("first", focused);
  const middle = focusTarget("middle", focused);
  const last = focusTarget("last", focused);
  const overlay = {
    contains(element) {
      return [first, middle, last].includes(element);
    },
    focus() {
      focused.push("overlay");
    },
    querySelectorAll() {
      return [first, middle, last];
    },
  };

  const forward = keyEvent(false);
  assert.equal(trapRelationModalTab(forward, overlay, last), true);
  assert.equal(forward.prevented, true);
  assert.deepEqual(focused, ["first"]);

  const backward = keyEvent(true);
  assert.equal(trapRelationModalTab(backward, overlay, first), true);
  assert.equal(backward.prevented, true);
  assert.deepEqual(focused, ["first", "last"]);

  const escaped = keyEvent(false);
  assert.equal(trapRelationModalTab(escaped, overlay, {}), true);
  assert.deepEqual(focused, ["first", "last", "first"]);

  const internal = keyEvent(false);
  assert.equal(trapRelationModalTab(internal, overlay, middle), false);
  assert.equal(internal.prevented, false);
});

test("modal background inerting excludes the overlay branch and restores prior state", () => {
  const background = treeElement(false);
  const overlay = treeElement(false);
  const fullscreenHost = treeElement(false, [background, overlay]);
  const outsideFullscreen = treeElement(true);
  const body = treeElement(false, [fullscreenHost, outsideFullscreen]);
  fullscreenHost.parentElement = body;

  const restore = makeRelationModalBackgroundInert(overlay);
  assert.equal(background.inert, true);
  assert.equal(outsideFullscreen.inert, true);
  assert.equal(overlay.inert, false);
  assert.equal(fullscreenHost.inert, false);

  restore();
  assert.equal(background.inert, false);
  assert.equal(outsideFullscreen.inert, true);
});

function focusTarget(name, focused) {
  return {
    tabIndex: 0,
    focus() {
      focused.push(name);
    },
    getAttribute() {
      return null;
    },
    hasAttribute() {
      return false;
    },
  };
}

function keyEvent(shiftKey) {
  return {
    key: "Tab",
    shiftKey,
    prevented: false,
    preventDefault() {
      this.prevented = true;
    },
  };
}

function treeElement(inert, children = []) {
  const element = { inert, children, parentElement: null };
  for (const child of children) child.parentElement = element;
  return element;
}
