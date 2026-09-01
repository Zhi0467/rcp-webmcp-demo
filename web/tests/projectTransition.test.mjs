import assert from "node:assert/strict";
import test from "node:test";

import {
  decodeTransitionTriggerManifest,
  emptyProjectTransitionCoordinator,
  reduceProjectTransitionCoordinator,
  reduceProjectTransitionProjection,
  transitionPreviewRouting,
  transitionSnapshotRefusal,
  transitionSyncCompletionDisposition,
} from "../src/projectTransition.ts";

const transitionOne = "1".repeat(64);
const transitionTwo = "2".repeat(64);

function head(revision, transitionId, target = { kind: "main" }) {
  return { target, revision, transition_id: transitionId };
}

function projection(fields = {}) {
  return {
    head: head(1, transitionOne),
    graph: { revision: 1, marker: "graph-one" },
    attention: {
      pending_proposal_ids: [],
      decisions_awaiting_choice_ids: [],
      open_blocker_ids: [],
    },
    experiment_control: { marker: "control-one" },
    ruleset_tag: "rcp.lifecycle.v1",
    transition_id: transitionOne,
    canonical: true,
    ...fields,
  };
}

test("a coherent canonical response atomically replaces graph, control, and head", () => {
  const current = projection();
  const incoming = projection({
    head: head(2, transitionTwo),
    graph: { revision: 2, marker: "graph-two" },
    experiment_control: { marker: "control-two" },
    transition_id: transitionTwo,
  });

  const replaced = reduceProjectTransitionProjection(current, {
    kind: "canonical",
    snapshot: incoming,
  });

  assert.strictEqual(replaced, incoming);
  assert.deepEqual(
    [replaced.graph.marker, replaced.experiment_control.marker, replaced.head.revision],
    ["graph-two", "control-two", 2],
  );
});

test("transition replacement refuses missing or malformed attention", () => {
  const current = projection();
  const malformed = [
    projection({ attention: undefined }),
    projection({
      attention: {
        pending_proposal_ids: [],
        decisions_awaiting_choice_ids: [],
      },
    }),
    projection({
      attention: {
        pending_proposal_ids: ["duplicate", "duplicate"],
        decisions_awaiting_choice_ids: [],
        open_blocker_ids: [],
      },
    }),
  ];

  for (const snapshot of malformed) {
    const action = { kind: "canonical", snapshot };
    assert.equal(transitionSnapshotRefusal(current, action), "attention_projection_invalid");
    assert.strictEqual(reduceProjectTransitionProjection(current, action), current);
  }
});

test("mismatched or regressing canonical responses retain the exact prior projection", () => {
  const current = projection();
  const cases = [
    projection({ graph: { revision: 9, marker: "wrong graph head" } }),
    projection({ head: head(1, transitionTwo) }),
    projection({
      head: head(2, transitionTwo, { kind: "branch", branch_id: "other" }),
      graph: { revision: 2 },
      transition_id: transitionTwo,
    }),
    projection({
      head: head(0, transitionTwo),
      graph: { revision: 0 },
      transition_id: transitionTwo,
    }),
    projection({ canonical: false }),
  ];

  for (const candidate of cases) {
    assert.strictEqual(
      reduceProjectTransitionProjection(current, { kind: "canonical", snapshot: candidate }),
      current,
    );
  }
});

test("preview replacement requires the exact canonical base head and current manifest ruleset", () => {
  const current = projection();
  const preview = projection({
    head: head(2, transitionTwo),
    graph: { revision: 2, marker: "preview-graph" },
    experiment_control: { marker: "preview-control" },
    transition_id: transitionTwo,
    canonical: false,
    base_head: current.head,
  });
  const action = {
    kind: "preview",
    snapshot: preview,
    expected_base_head: current.head,
    manifest_ruleset_tag: "rcp.lifecycle.v1",
  };

  assert.equal(transitionSnapshotRefusal(current, action), null);
  assert.strictEqual(reduceProjectTransitionProjection(current, action), preview);

  const wrongBase = {
    ...action,
    expected_base_head: head(0, null),
  };
  assert.equal(transitionSnapshotRefusal(current, wrongBase), "preview_base_head_mismatch");
  assert.strictEqual(reduceProjectTransitionProjection(current, wrongBase), current);

  const wrongPreviewHead = {
    ...action,
    snapshot: {
      ...preview,
      head: head(3, transitionTwo),
      graph: { revision: 3 },
    },
  };
  assert.equal(
    transitionSnapshotRefusal(current, wrongPreviewHead),
    "preview_head_revision_mismatch",
  );
  assert.strictEqual(reduceProjectTransitionProjection(current, wrongPreviewHead), current);

  const wrongRuleset = { ...action, manifest_ruleset_tag: "rcp.lifecycle.v2" };
  assert.equal(transitionSnapshotRefusal(current, wrongRuleset), "preview_ruleset_mismatch");
  assert.strictEqual(reduceProjectTransitionProjection(current, wrongRuleset), current);
});

test("a derived head with an unknown transition id accepts an authoritative matching base", () => {
  const current = projection({
    head: head(1, null),
    transition_id: null,
    ruleset_tag: null,
  });
  const preview = projection({
    head: head(2, transitionTwo),
    graph: { revision: 2 },
    transition_id: transitionTwo,
    canonical: false,
    base_head: head(1, transitionOne),
  });
  const action = {
    kind: "preview",
    snapshot: preview,
    expected_base_head: current.head,
    manifest_ruleset_tag: null,
  };

  assert.equal(transitionSnapshotRefusal(current, action), null);
  assert.strictEqual(reduceProjectTransitionProjection(current, action), preview);
});

test("manifest routing is conservative but never computes transition outcomes", () => {
  const manifest = {
    ruleset_tag: "rcp.lifecycle.v1",
    triggers: [
      {
        operation: "update_nodes",
        node_types: ["blocker", "decision", "experiment"],
        node_fields: ["status", "selected_option", "current_summary", "next_action"],
        relations: [],
      },
      {
        operation: "remove_edges",
        node_types: [],
        node_fields: [],
        relations: ["blocked_by", "governed_by"],
      },
    ],
  };

  assert.deepEqual(
    transitionPreviewRouting(manifest, "rcp.lifecycle.v1", {
      operation: "update_nodes",
      node_types: ["blocker"],
      node_fields: ["status"],
      relations: [],
    }),
    { route: "backend_preview", reason: "possible_trigger" },
  );
  assert.deepEqual(
    transitionPreviewRouting(manifest, "rcp.lifecycle.v1", {
      operation: "update_nodes",
      node_types: ["evidence"],
      node_fields: ["status"],
      relations: [],
    }),
    { route: "local_draft", reason: "no_manifest_trigger" },
  );
  assert.deepEqual(
    transitionPreviewRouting(manifest, "rcp.lifecycle.v1", {
      operation: "remove_edges",
    }),
    { route: "backend_preview", reason: "possible_trigger" },
  );
  assert.deepEqual(
    transitionPreviewRouting(manifest, "rcp.lifecycle.v1", {
      operation: "create_nodes",
      node_types: ["evidence"],
    }),
    { route: "local_draft", reason: "no_manifest_trigger" },
  );
});

test("missing or mismatched routing tags always fall back to backend preview", () => {
  const manifest = { ruleset_tag: "rcp.lifecycle.v1", triggers: [] };
  assert.deepEqual(
    transitionPreviewRouting(null, "rcp.lifecycle.v1", { operation: "create_nodes" }),
    {
      route: "backend_preview",
      reason: "missing_manifest",
    },
  );
  assert.deepEqual(transitionPreviewRouting(manifest, null, { operation: "create_nodes" }), {
    route: "backend_preview",
    reason: "missing_ruleset_tag",
  });
  assert.deepEqual(
    transitionPreviewRouting(manifest, "rcp.lifecycle.v2", { operation: "create_nodes" }),
    { route: "backend_preview", reason: "ruleset_mismatch" },
  );
  assert.deepEqual(transitionPreviewRouting(manifest, "rcp.lifecycle.v1", {}), {
    route: "backend_preview",
    reason: "possible_trigger",
  });
});

test("only runtime-valid trigger manifests can authorize local draft routing", () => {
  const manifest = {
    ruleset_tag: "rcp.lifecycle.v1",
    triggers: [
      {
        operation: "update_nodes",
        node_types: ["blocker"],
        node_fields: ["status"],
        relations: [],
      },
    ],
  };
  assert.deepEqual(decodeTransitionTriggerManifest(manifest), manifest);
  assert.equal(decodeTransitionTriggerManifest({ ...manifest, triggers: null }), null);
  assert.equal(
    decodeTransitionTriggerManifest({
      ...manifest,
      triggers: [{ ...manifest.triggers[0], node_fields: "status" }],
    }),
    null,
  );
  assert.equal(decodeTransitionTriggerManifest({ ...manifest, ruleset_tag: "" }), null);
  assert.equal(decodeTransitionTriggerManifest(manifest, "rcp.lifecycle.v2"), null);
});

test("a delayed Sync response for an inactive project is reconciled instead of applied", async () => {
  const projectHead = head(3, transitionOne);
  const fence = {
    project_id: "project-a",
    request_id: 1,
    expected_head: projectHead,
    draft_generation: 0,
  };
  let coordinator = emptyProjectTransitionCoordinator();
  coordinator = reduceProjectTransitionCoordinator(coordinator, {
    kind: "activate",
    project_id: "project-a",
  });
  coordinator = reduceProjectTransitionCoordinator(coordinator, {
    kind: "observe_head",
    project_id: "project-a",
    head: projectHead,
  });
  coordinator = reduceProjectTransitionCoordinator(coordinator, { kind: "sync_started", fence });

  let releaseResponse;
  const delayedResponse = new Promise((resolve) => {
    releaseResponse = resolve;
  });
  const disposition = delayedResponse.then(() =>
    transitionSyncCompletionDisposition(coordinator, fence),
  );
  coordinator = reduceProjectTransitionCoordinator(coordinator, {
    kind: "activate",
    project_id: "project-b",
  });
  releaseResponse();

  assert.equal(await disposition, "reload_inactive");
});

test("edits staged while Sync is in flight force committed response reconciliation", async () => {
  const projectHead = head(3, transitionOne);
  const fence = {
    project_id: "project-a",
    request_id: 1,
    expected_head: projectHead,
    draft_generation: 4,
  };
  let coordinator = emptyProjectTransitionCoordinator();
  for (const action of [
    { kind: "activate", project_id: "project-a" },
    { kind: "observe_head", project_id: "project-a", head: projectHead },
    { kind: "observe_draft_generation", project_id: "project-a", generation: 4 },
    { kind: "sync_started", fence },
  ]) {
    coordinator = reduceProjectTransitionCoordinator(coordinator, action);
  }

  let releaseResponse;
  const delayedResponse = new Promise((resolve) => {
    releaseResponse = resolve;
  });
  const disposition = delayedResponse.then(() =>
    transitionSyncCompletionDisposition(coordinator, fence),
  );
  coordinator = reduceProjectTransitionCoordinator(coordinator, {
    kind: "observe_draft_generation",
    project_id: "project-a",
    generation: 5,
  });
  releaseResponse();

  assert.equal(await disposition, "reload_active");
});

test("Sync completion refuses a superseded request, advanced head, or draft generation", () => {
  const projectHead = head(3, transitionOne);
  const fence = {
    project_id: "project-a",
    request_id: 1,
    expected_head: projectHead,
    draft_generation: 0,
  };
  let coordinator = emptyProjectTransitionCoordinator();
  for (const action of [
    { kind: "activate", project_id: "project-a" },
    { kind: "observe_head", project_id: "project-a", head: projectHead },
    { kind: "sync_started", fence },
  ]) {
    coordinator = reduceProjectTransitionCoordinator(coordinator, action);
  }
  assert.equal(transitionSyncCompletionDisposition(coordinator, fence), "apply");

  const newerHead = head(4, transitionTwo);
  const advanced = reduceProjectTransitionCoordinator(coordinator, {
    kind: "observe_head",
    project_id: "project-a",
    head: newerHead,
  });
  assert.equal(transitionSyncCompletionDisposition(advanced, fence), "reload_active");

  const restaged = reduceProjectTransitionCoordinator(coordinator, {
    kind: "observe_draft_generation",
    project_id: "project-a",
    generation: 1,
  });
  assert.equal(transitionSyncCompletionDisposition(restaged, fence), "reload_active");

  const replacementFence = { ...fence, request_id: 2 };
  const superseded = reduceProjectTransitionCoordinator(coordinator, {
    kind: "sync_started",
    fence: replacementFence,
  });
  assert.equal(transitionSyncCompletionDisposition(superseded, fence), "reload_active");
  assert.equal(transitionSyncCompletionDisposition(superseded, replacementFence), "apply");
  const oldCompletion = reduceProjectTransitionCoordinator(superseded, {
    kind: "sync_finished",
    fence,
  });
  assert.equal(oldCompletion.sync_requests["project-a"].request_id, 2);
});
