import assert from "node:assert/strict";
import { after, test } from "node:test";
import { createServer } from "vite";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const {
  attentionGraphForProjection,
  canonicalGraphHead,
  experimentStartNeedsSync,
  humanAttentionBlockers,
  humanDraftTransitionRouting,
  primaryQuestionForGraph,
  projectWithGraph,
} = await server.ssrLoadModule("/src/App.tsx");

after(() => server.close());

test("only a backend candidate requires Sync before an Experiment start", () => {
  const localDraft = { base_head: null };
  const backendCandidate = { base_head: canonicalGraphHead(4) };

  assert.equal(experimentStartNeedsSync(null), false);
  assert.equal(experimentStartNeedsSync(localDraft), false);
  assert.equal(experimentStartNeedsSync(backendCandidate), true);
});

const manifest = {
  ruleset_tag: "rcp.lifecycle.v1",
  triggers: [
    {
      operation: "update_nodes",
      node_types: ["blocker", "decision", "experiment", "hypothesis"],
      node_fields: ["status", "selected_option", "current_summary", "next_action"],
      relations: [],
    },
  ],
};

const graph = {
  revision: 4,
  nodes: {
    blocker: { id: "blocker", type: "blocker", updated_rev: 4 },
    evidence: { id: "evidence", type: "evidence", updated_rev: 4 },
    experiment: { id: "experiment", type: "experiment", updated_rev: 4 },
  },
};

function draft(fields = {}) {
  return {
    version: 1,
    base_revision: 4,
    nodes: {},
    removed_node_ids: [],
    proposals: {},
    ontology: null,
    custom_nodes: {},
    ...fields,
  };
}

test("App derives a conservative canonical main head from an ordinary project snapshot", () => {
  assert.deepEqual(canonicalGraphHead(4), {
    target: { kind: "main" },
    revision: 4,
    transition_id: null,
  });
});

test("Blocker lifecycle edits route to preview while unrelated Evidence wording stays local", () => {
  const blockerStatus = draft({
    nodes: { blocker: { base_updated_rev: 4, changes: { status: "resolved" } } },
  });
  const evidenceTitle = draft({
    nodes: { evidence: { base_updated_rev: 4, changes: { title: "Clearer title" } } },
  });

  assert.deepEqual(
    humanDraftTransitionRouting(blockerStatus, graph, manifest, "rcp.lifecycle.v1"),
    { route: "backend_preview", reason: "possible_trigger" },
  );
  assert.deepEqual(
    humanDraftTransitionRouting(evidenceTitle, graph, manifest, "rcp.lifecycle.v1"),
    { route: "local_draft", reason: "no_manifest_trigger" },
  );
});

test("node removals and Proposal decisions stay backend-owned even without manifest tags", () => {
  assert.deepEqual(
    humanDraftTransitionRouting(
      draft({ removed_node_ids: ["blocker"] }),
      graph,
      manifest,
      "rcp.lifecycle.v1",
    ),
    { route: "backend_preview", reason: "possible_trigger" },
  );
  assert.deepEqual(
    humanDraftTransitionRouting(
      draft({ proposals: { proposal: { decision: "approved" } } }),
      graph,
      manifest,
      "rcp.lifecycle.v1",
    ),
    { route: "backend_preview", reason: "possible_trigger" },
  );
});

test("Experiment ceiling edits and custom Experiment creation always request coherent preview", () => {
  assert.deepEqual(
    humanDraftTransitionRouting(
      draft({
        nodes: {
          experiment: { base_updated_rev: 4, changes: { invocation_ceiling: 8 } },
        },
      }),
      graph,
      manifest,
      "rcp.lifecycle.v1",
    ),
    { route: "backend_preview", reason: "possible_trigger" },
  );
  assert.deepEqual(
    humanDraftTransitionRouting(
      draft({
        custom_nodes: {
          custom: { id: "custom", type: "experiment", title: "Custom Experiment" },
        },
      }),
      graph,
      manifest,
      "rcp.lifecycle.v1",
    ),
    { route: "backend_preview", reason: "possible_trigger" },
  );
});

test("a missing manifest fails every staged edit to backend preview", () => {
  assert.deepEqual(
    humanDraftTransitionRouting(
      draft({
        nodes: { evidence: { base_updated_rev: 4, changes: { title: "Clearer title" } } },
      }),
      graph,
      null,
      null,
    ),
    { route: "backend_preview", reason: "missing_manifest" },
  );
});

test("attention membership follows a backend preview candidate, not canonical blockers", () => {
  const canonical = {
    revision: 4,
    nodes: {
      blocker: {
        id: "blocker",
        type: "blocker",
        status: "open",
        standing: "asserted",
      },
    },
  };
  const previewGraph = {
    ...canonical,
    revision: 5,
    nodes: {
      blocker: { ...canonical.nodes.blocker, status: "resolved" },
    },
  };
  const projection = {
    head: canonicalGraphHead(5, "2".repeat(64)),
    graph: previewGraph,
    attention: {
      pending_proposal_ids: [],
      decisions_awaiting_choice_ids: [],
      open_blocker_ids: [],
    },
    experiment_control: {},
    ruleset_tag: "rcp.lifecycle.v1",
    transition_id: "2".repeat(64),
    canonical: false,
    base_head: canonicalGraphHead(4, "1".repeat(64)),
  };

  const attentionGraph = attentionGraphForProjection(canonical, projection);
  assert.strictEqual(attentionGraph, previewGraph);
  assert.deepEqual(
    humanAttentionBlockers(projection.attention.open_blocker_ids, attentionGraph.nodes),
    [],
  );
});

test("attention content follows the same local projection snapshot as the rest of staging", () => {
  const canonical = {
    revision: 4,
    nodes: {
      blocker: {
        id: "blocker",
        type: "blocker",
        title: "Canonical title",
        status: "open",
        standing: "asserted",
      },
    },
  };
  const projected = {
    ...canonical,
    nodes: {
      blocker: { ...canonical.nodes.blocker, title: "Staged title" },
    },
  };
  const projection = {
    head: canonicalGraphHead(4),
    graph: projected,
    attention: {
      pending_proposal_ids: [],
      decisions_awaiting_choice_ids: [],
      open_blocker_ids: ["blocker"],
    },
    experiment_control: {},
    ruleset_tag: "rcp.lifecycle.v1",
    transition_id: null,
    canonical: false,
    base_head: canonicalGraphHead(4),
  };

  assert.strictEqual(attentionGraphForProjection(canonical, projection), projected);
});

test("candidate snapshots rederive the primary question by backend standing and id order", () => {
  const questions = {
    revision: 5,
    nodes: {
      "rq/contested": {
        id: "rq/contested",
        type: "research_question",
        standing: "contested",
      },
      "rq/asserted-b": {
        id: "rq/asserted-b",
        type: "research_question",
        standing: "asserted",
      },
      "rq/asserted-a": {
        id: "rq/asserted-a",
        type: "research_question",
        standing: "asserted",
      },
    },
    proposals: {},
  };
  assert.equal(primaryQuestionForGraph(questions).id, "rq/asserted-a");

  const project = {
    primary_question: { id: "rq/removed", type: "research_question", standing: "accepted" },
    attention: {
      pending_proposal_ids: [],
      decisions_awaiting_choice_ids: [],
      open_blocker_ids: [],
    },
    counts: {
      pending_proposals: 0,
      decisions_awaiting_choice: 0,
      open_blockers: 0,
      asserted: 0,
      accepted: 1,
      contested: 0,
    },
  };
  assert.equal(projectWithGraph(project, questions).primary_question.id, "rq/asserted-a");
  assert.equal(projectWithGraph(project, { ...questions, nodes: {} }).primary_question, null);
});
