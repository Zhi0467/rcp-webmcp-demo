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
const { ProposalJudgmentSection } = await server.ssrLoadModule("/src/components/AttentionRail.tsx");
const { decodeProposal, proposalSemantics } = await server.ssrLoadModule("/src/types.ts");
const { emptyHumanDraft, stageProposalDecision } = await server.ssrLoadModule("/src/humanDraft.ts");

after(() => server.close());

const nodes = {
  "rq/plasticity": {
    id: "rq/plasticity",
    type: "research_question",
    title: "Plasticity after shifts",
    question: "Can plasticity survive a task shift?",
    scope: "One task shift",
    status: "open",
  },
  "hyp/replanning": {
    id: "hyp/replanning",
    type: "hypothesis",
    title: "Replanning preserves plasticity",
    statement: "Replanning prevents plasticity loss.",
    status: "proposed",
  },
  "hyp/rehearsal": {
    id: "hyp/rehearsal",
    type: "hypothesis",
    title: "Rehearsal preserves plasticity",
    statement: "Rehearsal prevents plasticity loss.",
    status: "active",
  },
  "ev/evaluation": {
    id: "ev/evaluation",
    type: "evidence",
    title: "Evaluation result",
  },
};

const graph = {
  revision: 9,
  nodes,
  edges: {
    "edge/question-hypothesis": {
      id: "edge/question-hypothesis",
      source: "rq/plasticity",
      target: "hyp/replanning",
      relation: "has_hypothesis",
    },
    "edge/evidence-hypothesis": {
      id: "edge/evidence-hypothesis",
      source: "ev/evaluation",
      target: "hyp/replanning",
      relation: "supports",
    },
    "edge/unrelated": {
      id: "edge/unrelated",
      source: "rq/plasticity",
      target: "hyp/rehearsal",
      relation: "has_hypothesis",
    },
  },
  proposals: {},
  ambiguities: {},
  glossary: {},
  ontology: { types: [], fields: [], relations: [] },
  validation_messages: [],
  belief_transitions: [],
  replay_status: "complete",
};

function proposal(operation, decisionNeeded = "Use the stored card fallback.") {
  return {
    id: `prop/${operation.intent}`,
    title: "Judge the proposed belief change",
    base_rev: 9,
    card: {
      situation_cold: "A protected belief may change.",
      why_human_now: "Only the human may make this change.",
      consequences: "The graph changes if approved.",
      decision_needed: decisionNeeded,
    },
    ops: [operation],
  };
}

function renderProposal(operation, currentGraph = graph, decisionNeeded) {
  return renderToStaticMarkup(
    React.createElement(ProposalJudgmentSection, {
      proposals: [proposal(operation, decisionNeeded)],
      graph: currentGraph,
      draft: null,
      onDecision() {},
    }),
  );
}

test("content-change proposals compare every changed field with current graph wording", () => {
  const html = renderProposal({
    op: "update_nodes",
    intent: "content_change",
    nodes: [
      {
        id: "rq/plasticity",
        changes: {
          question: "Can plasticity survive repeated task shifts?",
          scope: "Repeated task shifts",
        },
      },
    ],
  });

  assert.match(html, /Node: <\/strong>Plasticity after shifts/);
  assert.match(html, /Current question: <\/strong>“Can plasticity survive a task shift\?”/);
  assert.match(
    html,
    /Proposed question: <\/strong>“Can plasticity survive repeated task shifts\?”/,
  );
  assert.match(html, /Current scope: <\/strong>“One task shift”/);
  assert.match(html, /Proposed scope: <\/strong>“Repeated task shifts”/);

  const lifecycleHtml = renderProposal({
    op: "update_nodes",
    intent: "content_change",
    nodes: [{ id: "rq/plasticity", changes: { status: "answered" } }],
  });
  assert.match(lifecycleHtml, /Current status: <\/strong>“open”/);
  assert.match(lifecycleHtml, /Proposed status: <\/strong>“answered”/);
});

test("removal proposals name the node and every incident relation", () => {
  const html = renderProposal({
    op: "remove_nodes",
    intent: "removal",
    node_ids: ["hyp/replanning"],
  });

  assert.match(html, /Remove: <\/strong>Replanning preserves plasticity/);
  assert.match(
    html,
    /Also removes: <\/strong>Evaluation result — supports → Replanning preserves plasticity/,
  );
  assert.match(
    html,
    /Also removes: <\/strong>Plasticity after shifts — has hypothesis → Replanning preserves plasticity/,
  );
  assert.doesNotMatch(html, /Rehearsal preserves plasticity/);
});

test("supersede and merge proposals show both involved nodes", () => {
  const supersede = renderProposal({
    op: "supersede_nodes",
    intent: "supersede",
    nodes: [{ id: "hyp/replanning", superseded_by: "hyp/rehearsal" }],
  });
  assert.match(supersede, /Supersede: <\/strong>Replanning preserves plasticity/);
  assert.match(supersede, /With: <\/strong>Rehearsal preserves plasticity/);

  const merge = renderProposal({
    op: "merge_nodes",
    intent: "merge",
    merges: [{ duplicate: "hyp/replanning", canonical: "hyp/rehearsal" }],
  });
  assert.match(merge, /Merge: <\/strong>Replanning preserves plasticity/);
  assert.match(merge, /Into: <\/strong>Rehearsal preserves plasticity/);
});

test("protected relation proposals show the compact relation and both endpoints", () => {
  const creation = renderProposal({
    op: "create_edges",
    intent: "protected_relation_change",
    edges: [
      {
        source: "rq/plasticity",
        target: "hyp/rehearsal",
        relation: "has_hypothesis",
      },
    ],
  });
  assert.match(
    creation,
    /Add relation: <\/strong>Plasticity after shifts — has hypothesis → Rehearsal preserves plasticity/,
  );

  const removal = renderProposal({
    op: "remove_edges",
    intent: "protected_relation_change",
    edge_ids: ["edge/question-hypothesis"],
  });
  assert.match(
    removal,
    /Remove relation: <\/strong>Plasticity after shifts — has hypothesis → Replanning preserves plasticity/,
  );
});

test("status-change proposals show a clear current-to-proposed transition", () => {
  const html = renderProposal({
    op: "update_nodes",
    intent: "status_change",
    nodes: [
      {
        id: "hyp/replanning",
        changes: { status: "supported" },
        cause: { kind: "evidence_edge", ref_id: "edge/evidence-hypothesis" },
      },
    ],
  });

  assert.match(html, /Node: <\/strong>Replanning preserves plasticity/);
  assert.match(html, /Status: <\/strong>proposed → supported/);
});

test("legacy or stale proposals use the existing card fallback instead of inferred intent", () => {
  const missingNodeGraph = { ...graph, nodes: { ...nodes, "rq/plasticity": undefined } };
  const html = renderProposal(
    {
      op: "update_nodes",
      intent: "content_change",
      nodes: [{ id: "rq/plasticity", changes: { question: "A stale proposal" } }],
    },
    missingNodeGraph,
    "Compare this proposal from its stored card.",
  );

  assert.match(html, /Compare this proposal from its stored card\./);
  assert.doesNotMatch(html, /Current question/);

  const undeclared = renderProposal(
    {
      op: "update_nodes",
      nodes: [{ id: "hyp/replanning", changes: { status: "supported" } }],
    },
    graph,
    "Review this legacy proposal.",
  );
  assert.match(undeclared, /Review this legacy proposal\./);
  assert.doesNotMatch(undeclared, /Status: <\/strong>/);
});

test("the API decoder distinguishes the closed Proposal contract from legacy drift", () => {
  const canonical = decodeProposal({
    ...proposal({
      op: "remove_nodes",
      intent: "removal",
      node_ids: ["hyp/replanning"],
    }),
    status: "pending",
    related_node_ids: ["hyp/replanning"],
    related_edge_ids: ["edge/evidence-hypothesis", "edge/question-hypothesis"],
    related_config_keys: ["ontology"],
    raised_rev: 9,
    resolved_rev: null,
  });
  assert.equal(canonical.semantics, "canonical");
  assert.equal(canonical.ops[0].intent, "removal");
  assert.deepEqual(proposalSemantics(canonical).resourceKeys, [
    "edge:edge/evidence-hypothesis",
    "edge:edge/question-hypothesis",
    "node:hyp/replanning",
  ]);

  const drifted = decodeProposal({
    ...canonical,
    semantics: undefined,
    ops: [{ ...canonical.ops[0], unexpected: true }],
  });
  assert.equal(drifted.semantics, "legacy");
  assert.equal(proposalSemantics(drifted).operation, null);
  assert.deepEqual(proposalSemantics(drifted).resourceKeys, [
    "config:ontology",
    "edge:edge/evidence-hypothesis",
    "edge:edge/question-hypothesis",
    "node:hyp/replanning",
  ]);

  const researchQuestionLifecycle = decodeProposal({
    ...canonical,
    semantics: undefined,
    ops: [
      {
        op: "update_nodes",
        intent: "content_change",
        nodes: [{ id: "rq/question", changes: { status: "answered" } }],
      },
    ],
  });
  assert.equal(researchQuestionLifecycle.semantics, "canonical");
});

test("an overlapping second approval is visibly blocked while its rejection remains available", () => {
  const proposalWithResources = (id, title) =>
    decodeProposal({
      ...proposal({
        op: "update_nodes",
        intent: "content_change",
        nodes: [{ id: "hyp/replanning", changes: { title } }],
      }),
      id,
      title,
      status: "pending",
      related_node_ids: ["hyp/replanning"],
      related_edge_ids: [],
      related_config_keys: [],
      raised_rev: 9,
      resolved_rev: null,
    });
  const first = proposalWithResources("prop/first", "First staged change");
  const second = proposalWithResources("prop/second", "Second staged change");
  const proposalGraph = {
    ...graph,
    proposals: { [first.id]: first, [second.id]: second },
  };
  const draft = stageProposalDecision(emptyHumanDraft(9), proposalGraph, first.id, "approved");
  const html = renderToStaticMarkup(
    React.createElement(ProposalJudgmentSection, {
      proposals: [first, second],
      graph: proposalGraph,
      draft,
      onDecision() {},
    }),
  );

  assert.match(html, /Approval conflicts with staged approval: First staged change\./);
  assert.match(
    html,
    /class="button judgment proposal-decision-toggle approve"[^>]*disabled=""[^>]*title="Approval conflicts with staged approval: First staged change\./,
  );
  assert.match(
    html,
    /class="button judgment proposal-decision-toggle reject"[^>]*aria-pressed="false"/,
  );
});
