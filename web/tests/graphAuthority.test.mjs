import assert from "node:assert/strict";
import test from "node:test";

import {
  graphMutationsDisabled,
  replayFailureLabel,
  taskMayMutateGraph,
} from "../src/graphAuthority.ts";

test("degraded replay blocks graph authority and names the last coherent state", () => {
  const graph = {
    replay_status: "degraded",
    replay_failure: {
      revision: 6,
      code: "invalid-edge",
      message: "The accepted patch no longer validates.",
    },
  };
  assert.equal(graphMutationsDisabled(graph), true);
  assert.equal(
    replayFailureLabel(graph),
    "Replay stopped at revision 6 (invalid-edge): The accepted patch no longer validates. This is the last coherent graph.",
  );
  assert.equal(graphMutationsDisabled({ replay_status: "complete", replay_failure: null }), false);
});

test("only graph-writing task continuations are blocked by degraded replay", () => {
  assert.equal(taskMayMutateGraph({ kind: "refresh", request: {} }), true);
  assert.equal(taskMayMutateGraph({ kind: "project_chat", request: { mode: "work" } }), false);
  assert.equal(taskMayMutateGraph({ kind: "project_chat", request: { mode: "discuss" } }), false);
  assert.equal(taskMayMutateGraph({ kind: "node_chat", request: { mode: "work" } }), false);
  assert.equal(taskMayMutateGraph({ kind: "paper_coach", request: {} }), false);
});
