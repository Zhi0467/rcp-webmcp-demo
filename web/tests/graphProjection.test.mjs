import assert from "node:assert/strict";
import test from "node:test";

import {
  buildNodeProjectionEmphasis,
  buildDagProjection,
  edgeProjectionEmphasis,
  projectNodes,
  relationFocus,
} from "../src/graphProjection.ts";

const nodes = [
  { id: "accepted", type: "research_question", standing: "accepted" },
  { id: "contested", type: "hypothesis", standing: "contested" },
  { id: "asserted", type: "decision", standing: "asserted" },
  { id: "evidence", type: "evidence", standing: "asserted" },
  { id: "open-blocker", type: "blocker", status: "open", standing: "asserted" },
  { id: "resolved-blocker", type: "blocker", status: "resolved", standing: "asserted" },
];

test("working graph keeps accepted, asserted, and contested research visible", () => {
  assert.deepEqual(
    projectNodes(nodes, "working").map((node) => node.id),
    ["accepted", "contested", "asserted", "evidence", "open-blocker"],
  );
  assert.deepEqual(
    projectNodes(nodes, "accepted").map((node) => node.id),
    ["accepted"],
  );
});

test("active flow hides resolved Blockers while an explicit history projection retains them", () => {
  assert.deepEqual(
    projectNodes(nodes, "working", { includeResolvedBlockers: true }).map((node) => node.id),
    ["accepted", "contested", "asserted", "evidence", "open-blocker", "resolved-blocker"],
  );
});

test("relation focus temporarily projects every node and marks one-hop neighbors", () => {
  const graph = {
    nodes: Object.fromEntries(nodes.map((node) => [node.id, node])),
    edges: {
      first: {
        id: "first",
        source: "evidence",
        target: "contested",
        relation: "supports",
        layer: "epistemic",
        explanation: "",
      },
      second: {
        id: "second",
        source: "accepted",
        target: "asserted",
        relation: "has_decision",
        layer: "action",
        explanation: "",
      },
    },
  };
  const projection = buildDagProjection(graph, "accepted", "contested");
  assert.deepEqual(
    projection.nodes.map((node) => node.id),
    ["accepted", "contested", "asserted", "evidence", "open-blocker", "resolved-blocker"],
  );

  const focused = relationFocus("contested", projection.edges);
  assert.deepEqual([...focused.nodeIds], ["contested", "evidence"]);
  assert.deepEqual([...focused.edgeIds], ["first"]);
});

test("ontology projection changes emphasis without changing the projected graph", () => {
  const edges = [
    {
      id: "belief",
      source: "evidence",
      target: "contested",
      relation: "supports",
      layer: "epistemic",
      explanation: "",
    },
    {
      id: "seam",
      source: "asserted",
      target: "contested",
      relation: "tests",
      layer: "seam",
      explanation: "",
    },
    {
      id: "action",
      source: "accepted",
      target: "asserted",
      relation: "has_decision",
      layer: "action",
      explanation: "",
    },
    {
      id: "meta",
      source: "contested",
      target: "accepted",
      relation: "duplicate_of",
      layer: "meta",
      explanation: "",
    },
  ];
  assert.equal(edgeProjectionEmphasis(edges[0], "belief"), "emphasized");
  assert.equal(edgeProjectionEmphasis(edges[1], "belief"), "emphasized");
  assert.equal(edgeProjectionEmphasis(edges[2], "belief"), "dimmed");
  assert.equal(edgeProjectionEmphasis(edges[3], "belief"), "neutral");
  assert.equal(edgeProjectionEmphasis(edges[0], "action"), "dimmed");
  assert.equal(edgeProjectionEmphasis(edges[2], "action"), "emphasized");
  const beliefNodes = buildNodeProjectionEmphasis(edges, "belief");
  const allNodes = buildNodeProjectionEmphasis(edges, "all");
  assert.equal(beliefNodes.get("evidence"), "emphasized");
  assert.equal(beliefNodes.get("accepted"), "dimmed");
  assert.equal(allNodes.get("accepted"), "emphasized");
});
