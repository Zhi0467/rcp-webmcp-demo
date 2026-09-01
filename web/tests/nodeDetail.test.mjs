import assert from "node:assert/strict";
import test from "node:test";

import {
  beliefCausePresentation,
  edgeValidationFlags,
  nodeBeliefTransitions,
} from "../src/nodeDetail.ts";

const nodes = {
  "hyp/main": { id: "hyp/main", type: "hypothesis", title: "Main claim" },
  "ev/result": {
    id: "ev/result",
    type: "evidence",
    title: "Held-out result",
    origin: "internal_run",
  },
  "dec/metric": { id: "dec/metric", type: "decision", title: "Choose retention metric" },
};
const edges = [
  {
    id: "ev/result::supports::hyp/main",
    source: "ev/result",
    target: "hyp/main",
    relation: "supports",
    layer: "epistemic",
  },
];
const transitions = [
  {
    hypothesis_id: "hyp/other",
    from_status: "active",
    to_status: "supported",
    revision: 8,
    cause: { kind: "human_edit" },
  },
  {
    hypothesis_id: "hyp/main",
    from_status: "active",
    to_status: "supported",
    revision: 7,
    cause: { kind: "evidence_edge", ref_id: edges[0].id },
  },
  {
    hypothesis_id: "hyp/main",
    from_status: "proposed",
    to_status: "active",
    revision: 3,
    cause: { kind: "decision", ref_id: "dec/metric" },
  },
];

test("node detail selects its belief history and resolves navigable causes", () => {
  assert.deepEqual(
    nodeBeliefTransitions("hyp/main", transitions).map((item) => item.revision),
    [7, 3],
  );
  assert.deepEqual(beliefCausePresentation(transitions[1], edges, nodes), {
    label: "Evidence: Held-out result",
    nodeId: "ev/result",
  });
  assert.deepEqual(beliefCausePresentation(transitions[2], edges, nodes), {
    label: "Decision: Choose retention metric",
    nodeId: "dec/metric",
  });
  assert.deepEqual(beliefCausePresentation(transitions[0], edges, nodes), {
    label: "Human edit",
  });
});

test("relation flags stay attached to the implicated edge", () => {
  const messages = [
    {
      level: "flag",
      code: "relation-type-mismatch",
      message: "Evidence cannot block a decision.",
      related_node_ids: [],
      related_edge_ids: ["bad-edge"],
    },
    {
      level: "flag",
      code: "other",
      message: "Other warning",
      related_node_ids: [],
      related_edge_ids: ["bad-edge"],
    },
  ];
  assert.deepEqual(
    edgeValidationFlags("bad-edge", messages).map((item) => item.message),
    ["Evidence cannot block a decision."],
  );
  assert.deepEqual(edgeValidationFlags("good-edge", messages), []);
});
