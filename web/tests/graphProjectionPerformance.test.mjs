import assert from "node:assert/strict";
import test from "node:test";

import { buildNodeProjectionEmphasis } from "../src/graphProjection.ts";

test("node projection classifies each edge once instead of rescanning per node", () => {
  const edges = Array.from({ length: 500 }, (_, index) => ({
    id: `edge-${index}`,
    source: `node-${index}`,
    target: `node-${index + 1}`,
    relation: "supports",
    layer: index % 2 ? "epistemic" : "action",
    explanation: "",
  }));
  let classified = 0;
  const countedEdges = {
    *[Symbol.iterator]() {
      for (const edge of edges) {
        classified += 1;
        yield edge;
      }
    },
  };

  const emphasis = buildNodeProjectionEmphasis(countedEdges, "belief");

  assert.equal(classified, edges.length);
  assert.equal(emphasis.size, edges.length + 1);
});
