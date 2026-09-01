import assert from "node:assert/strict";
import { after, test } from "node:test";
import { createServer } from "vite";

import { ApiError } from "../src/api.ts";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const {
  applyHumanDraft,
  deserializeHumanDraft,
  emptyHumanDraft,
  humanDraftChangeCount,
  humanDraftBehindCount,
  humanDraftCommittableCount,
  humanDraftOntologyIsStale,
  humanDraftStorageKey,
  humanSyncFailure,
  normalizeHumanDraft,
  proposalApprovalConflict,
  reconcileHumanDraft,
  retainBehindDraftAfterSync,
  proposalTargetsNode,
  serializeHumanDraft,
  stageDecisionChoice,
  stageNodeEdit,
  stageNodeEditStart,
  stageNodeRemoval,
  stageNodeStanding,
  stageProposalDecision,
  stageCustomNode,
  stageOntology,
  unstageCustomNode,
  unstageNodeRemoval,
  toHumanSyncRequest,
} = await server.ssrLoadModule("/src/humanDraft.ts");

after(() => server.close());

const graph = {
  revision: 4,
  nodes: {
    "hyp/example": {
      id: "hyp/example",
      type: "hypothesis",
      title: "Existing title",
      standing: "accepted",
      created_rev: 2,
      updated_rev: 4,
      source_refs: [],
      extension_fields: {},
      statement: "Existing statement",
    },
  },
  edges: {},
  proposals: {},
  ambiguities: {},
  glossary: {},
  ontology: { types: [], fields: [], relations: [] },
  validation_messages: [],
  belief_transitions: [],
  replay_status: "complete",
  replay_failure: null,
};

test("normalization drops node fields and standing that match canonical state", () => {
  const draft = {
    ...emptyHumanDraft(4),
    nodes: {
      "hyp/example": {
        base_updated_rev: 4,
        changes: { title: "Existing title" },
        standing: "accepted",
        standing_origin: "judgment",
      },
    },
  };
  const normalized = normalizeHumanDraft(draft, graph);
  assert.deepEqual(normalized.nodes, {});
  assert.equal(humanDraftChangeCount(normalized), 0);
});

test("wording edits clear an existing judgment and disappear when fully reverted", () => {
  const editing = stageNodeEditStart(emptyHumanDraft(4), graph, "hyp/example");
  assert.equal(editing.nodes["hyp/example"].standing, "asserted");
  const edited = stageNodeEdit(editing, graph, "hyp/example", { title: "Revised" });
  assert.equal(edited.nodes["hyp/example"].standing, "asserted");
  assert.equal(applyHumanDraft(graph, edited).nodes["hyp/example"].draft_touched, true);
  assert.equal(applyHumanDraft(graph, edited).nodes["hyp/example"].title, "Revised");

  const reverted = stageNodeEdit(edited, graph, "hyp/example", { title: "Existing title" });
  assert.equal(humanDraftChangeCount(reverted), 0);
});

test("Blocker lifecycle edits invalidate prior judgment and can reopen attention", () => {
  const acceptedOpen = {
    id: "blocker/accepted-open",
    type: "blocker",
    title: "Accepted open blocker",
    standing: "accepted",
    status: "open",
    blocker_type: "scientific",
    description: "A result is missing.",
    resolution_condition: "Record the result.",
    recommended_action: null,
    created_rev: 2,
    updated_rev: 4,
    source_refs: [],
    extension_fields: {},
  };
  const resolvedAsserted = {
    ...acceptedOpen,
    id: "blocker/resolved-asserted",
    title: "Resolved asserted blocker",
    standing: "asserted",
    status: "resolved",
  };
  const lifecycleGraph = {
    ...graph,
    nodes: {
      ...graph.nodes,
      [acceptedOpen.id]: acceptedOpen,
      [resolvedAsserted.id]: resolvedAsserted,
    },
  };

  const resolvingStart = stageNodeEditStart(emptyHumanDraft(4), lifecycleGraph, acceptedOpen.id);
  const resolving = stageNodeEdit(resolvingStart, lifecycleGraph, acceptedOpen.id, {
    status: "resolved",
  });
  assert.deepEqual(resolving.nodes[acceptedOpen.id].changes, { status: "resolved" });
  assert.equal(resolving.nodes[acceptedOpen.id].standing, "asserted");
  assert.equal(
    applyHumanDraft(lifecycleGraph, resolving).nodes[acceptedOpen.id].status,
    "resolved",
  );

  const reopeningStart = stageNodeEditStart(
    emptyHumanDraft(4),
    lifecycleGraph,
    resolvedAsserted.id,
  );
  assert.equal(reopeningStart.nodes[resolvedAsserted.id].standing, "asserted");
  const reopening = stageNodeEdit(reopeningStart, lifecycleGraph, resolvedAsserted.id, {
    status: "open",
  });
  assert.deepEqual(reopening.nodes[resolvedAsserted.id].changes, { status: "open" });
  const reopened = applyHumanDraft(lifecycleGraph, reopening).nodes[resolvedAsserted.id];
  assert.equal(reopened.status, "open");
  assert.equal(reopened.standing, "asserted");
});

test("judgments and proposal decisions are reversible", () => {
  let draft = stageNodeStanding(emptyHumanDraft(4), graph, "hyp/example", "contested");
  draft = stageProposalDecision(draft, graph, "proposal/1", "approved");
  assert.equal(humanDraftChangeCount(draft), 2);

  draft = stageNodeStanding(draft, graph, "hyp/example", "accepted");
  draft = stageProposalDecision(draft, graph, "proposal/1", null);
  assert.equal(humanDraftChangeCount(draft), 0);
});

test("authoritative reconciliation clears only proposal choices that are no longer pending", () => {
  const pending = {
    id: "proposal/pending",
    status: "pending",
    related_node_ids: ["hyp/example"],
    related_edge_ids: [],
    related_config_keys: [],
    ops: [],
  };
  const withdrawn = { ...pending, id: "proposal/withdrawn", status: "withdrawn" };
  const movedGraph = {
    ...graph,
    revision: 5,
    proposals: { [pending.id]: pending, [withdrawn.id]: withdrawn },
  };
  const draft = {
    ...emptyHumanDraft(4),
    nodes: {
      "hyp/example": {
        base_updated_rev: 4,
        changes: { title: "My staged title" },
        standing: "asserted",
        standing_origin: "edit",
      },
    },
    proposals: {
      [pending.id]: { decision: "approved" },
      [withdrawn.id]: { decision: "rejected" },
      "proposal/missing": { decision: "approved" },
    },
  };

  const reconciliation = reconcileHumanDraft(draft, movedGraph);

  assert.deepEqual(reconciliation.discardedProposalIds, ["proposal/missing", "proposal/withdrawn"]);
  assert.deepEqual(reconciliation.draft.proposals, {
    [pending.id]: { decision: "approved" },
  });
  assert.deepEqual(reconciliation.draft.nodes["hyp/example"].changes, {
    title: "My staged title",
  });
  assert.equal(reconciliation.draft.base_revision, 5);
});

test("staging prevents overlapping approvals but permits rejections and independent approvals", () => {
  const proposal = (id, nodeId, edgeIds = []) => ({
    id,
    title: id,
    card: { situation_cold: "", why_human_now: "", consequences: "", decision_needed: "" },
    ops: [
      {
        op: "update_nodes",
        intent: "content_change",
        nodes: [{ id: nodeId, changes: { title: `Change from ${id}` } }],
      },
    ],
    related_node_ids: [nodeId],
    related_edge_ids: edgeIds,
    related_config_keys: [],
    base_rev: 4,
    raised_rev: 4,
    resolved_rev: null,
    status: "pending",
  });
  const first = proposal("proposal/first", "hyp/example", ["edge/shared"]);
  const overlapping = proposal("proposal/overlap", "hyp/example", ["edge/shared"]);
  const independent = proposal("proposal/independent", "hyp/other");
  const proposalGraph = {
    ...graph,
    proposals: {
      [first.id]: first,
      [overlapping.id]: overlapping,
      [independent.id]: independent,
    },
  };

  const firstApproved = stageProposalDecision(
    emptyHumanDraft(4),
    proposalGraph,
    first.id,
    "approved",
  );
  assert.deepEqual(proposalApprovalConflict(firstApproved, proposalGraph, overlapping.id), {
    proposalIds: [first.id],
    resourceKeys: ["node:hyp/example"],
  });
  assert.strictEqual(
    stageProposalDecision(firstApproved, proposalGraph, overlapping.id, "approved"),
    firstApproved,
  );

  const withRejection = stageProposalDecision(
    firstApproved,
    proposalGraph,
    overlapping.id,
    "rejected",
  );
  assert.equal(withRejection.proposals[overlapping.id].decision, "rejected");
  const withIndependentApproval = stageProposalDecision(
    withRejection,
    proposalGraph,
    independent.id,
    "approved",
  );
  assert.equal(withIndependentApproval.proposals[independent.id].decision, "approved");
});

test("approval collisions use semantic effects instead of shared dependency context", () => {
  const protectedCreation = (id, targetId) => ({
    id,
    title: id,
    card: { situation_cold: "", why_human_now: "", consequences: "", decision_needed: "" },
    ops: [
      {
        op: "create_edges",
        intent: "protected_relation_change",
        edges: [
          {
            source: "rq/shared",
            target: targetId,
            relation: "has_hypothesis",
          },
        ],
      },
    ],
    related_node_ids: ["rq/shared", targetId],
    related_edge_ids: [],
    related_config_keys: [],
    base_rev: 4,
    raised_rev: 4,
    resolved_rev: null,
    status: "pending",
  });
  const first = protectedCreation("proposal/first-edge", "hyp/first");
  const independent = protectedCreation("proposal/second-edge", "hyp/second");
  const duplicate = protectedCreation("proposal/duplicate-edge", "hyp/first");
  const removal = {
    ...protectedCreation("proposal/removal", "hyp/unused"),
    ops: [
      {
        op: "remove_nodes",
        intent: "removal",
        node_ids: ["hyp/remove"],
      },
    ],
    related_node_ids: ["hyp/remove"],
    related_edge_ids: ["edge/incident"],
  };
  const incidentRemoval = {
    ...protectedCreation("proposal/remove-incident", "hyp/unused"),
    ops: [
      {
        op: "remove_edges",
        intent: "protected_relation_change",
        edge_ids: ["edge/incident"],
      },
    ],
    related_node_ids: ["hyp/remove", "hyp/other"],
    related_edge_ids: ["edge/incident"],
  };
  const proposalGraph = {
    ...graph,
    proposals: Object.fromEntries(
      [first, independent, duplicate, removal, incidentRemoval].map((item) => [item.id, item]),
    ),
  };

  const firstApproved = stageProposalDecision(
    emptyHumanDraft(4),
    proposalGraph,
    first.id,
    "approved",
  );
  assert.equal(proposalApprovalConflict(firstApproved, proposalGraph, independent.id), null);
  const bothApproved = stageProposalDecision(
    firstApproved,
    proposalGraph,
    independent.id,
    "approved",
  );
  assert.equal(bothApproved.proposals[independent.id].decision, "approved");
  assert.deepEqual(proposalApprovalConflict(bothApproved, proposalGraph, duplicate.id), {
    proposalIds: [first.id],
    resourceKeys: ["edge:rq/shared::has_hypothesis::hyp/first"],
  });

  const removalApproved = stageProposalDecision(
    emptyHumanDraft(4),
    proposalGraph,
    removal.id,
    "approved",
  );
  assert.deepEqual(proposalApprovalConflict(removalApproved, proposalGraph, incidentRemoval.id), {
    proposalIds: [removal.id],
    resourceKeys: ["edge:edge/incident"],
  });
});

test("direct Decision choices merge with wording edits and supersede targeted proposal resolutions", () => {
  const decision = {
    id: "dec/resource",
    type: "decision",
    title: "Choose resource level",
    question: "Which resource level should the experiment use?",
    options: ["Small", "Medium", "Large"],
    selected_option: null,
    status: "open",
    standing: "asserted",
    created_rev: 2,
    updated_rev: 4,
    source_refs: [],
    extension_fields: {},
  };
  const targeted = {
    id: "proposal/targeted",
    title: "Use Medium",
    card: { situation_cold: "", why_human_now: "", consequences: "", decision_needed: "" },
    ops: [
      {
        op: "update_nodes",
        nodes: [{ id: decision.id, changes: { selected_option: "Medium" } }],
      },
    ],
    related_node_ids: [decision.id],
    base_rev: 4,
    status: "pending",
  };
  const relatedOnly = {
    ...targeted,
    id: "proposal/related-only",
    title: "Revise the hypothesis",
    ops: [
      {
        op: "update_nodes",
        nodes: [{ id: "hyp/example", changes: { statement: "Revised by proposal" } }],
      },
    ],
  };
  const decisionGraph = {
    ...graph,
    nodes: { ...graph.nodes, [decision.id]: decision },
    proposals: { [targeted.id]: targeted, [relatedOnly.id]: relatedOnly },
  };

  let draft = stageNodeEdit(emptyHumanDraft(4), decisionGraph, decision.id, {
    title: "Choose the initial resource level",
  });
  draft = stageProposalDecision(draft, decisionGraph, targeted.id, "approved");
  draft = stageProposalDecision(draft, decisionGraph, relatedOnly.id, "rejected");
  draft = stageDecisionChoice(draft, decisionGraph, decision.id, "Medium");

  assert.deepEqual(draft.nodes[decision.id], {
    base_updated_rev: 4,
    changes: {
      title: "Choose the initial resource level",
      selected_option: "Medium",
      status: "decided",
    },
    standing: "accepted",
    standing_origin: "judgment",
  });
  assert.deepEqual(draft.proposals, {
    [relatedOnly.id]: { decision: "rejected" },
  });
  assert.equal(proposalTargetsNode(targeted, decision.id), true);
  assert.equal(proposalTargetsNode(relatedOnly, decision.id), false);
  assert.equal(humanDraftChangeCount(draft), 5);

  const presented = applyHumanDraft(decisionGraph, draft).nodes[decision.id];
  assert.equal(presented.selected_option, "Medium");
  assert.equal(presented.status, "decided");
  assert.equal(presented.standing, "accepted");
  assert.equal(presented.draft_touched, true);

  const restored = deserializeHumanDraft(serializeHumanDraft(draft));
  assert.deepEqual(restored, draft);
  assert.deepEqual(toHumanSyncRequest(restored, decisionGraph), {
    base_revision: 4,
    removed_node_ids: [],
    nodes: [
      {
        node_id: decision.id,
        base_updated_rev: 4,
        changes: {
          title: "Choose the initial resource level",
          selected_option: "Medium",
          status: "decided",
        },
        standing: "accepted",
      },
    ],
    proposals: [{ proposal_id: relatedOnly.id, decision: "rejected" }],
    ontology: null,
    custom_nodes: [],
  });

  const revisedMedium = "Medium, with more time for validation";
  let editedOptionDraft = stageNodeEdit(emptyHumanDraft(4), decisionGraph, decision.id, {
    options: ["Small", revisedMedium, "Large"],
  });
  editedOptionDraft = stageDecisionChoice(
    editedOptionDraft,
    decisionGraph,
    decision.id,
    revisedMedium,
  );
  assert.deepEqual(editedOptionDraft.nodes[decision.id].changes, {
    options: ["Small", revisedMedium, "Large"],
    selected_option: revisedMedium,
    status: "decided",
  });
  assert.equal(
    toHumanSyncRequest(editedOptionDraft, decisionGraph).nodes[0].changes.selected_option,
    revisedMedium,
  );

  const replaced = stageDecisionChoice(restored, decisionGraph, decision.id, "Large");
  assert.equal(replaced.nodes[decision.id].changes.selected_option, "Large");
  assert.equal(stageDecisionChoice(replaced, decisionGraph, decision.id, "Unlisted"), replaced);

  const editedAfterChoice = stageNodeEdit(replaced, decisionGraph, decision.id, {
    rationale: "Use the larger run after the pilot.",
  });
  assert.equal(editedAfterChoice.nodes[decision.id].changes.selected_option, "Large");
  assert.equal(editedAfterChoice.nodes[decision.id].changes.status, "decided");
  assert.equal(editedAfterChoice.nodes[decision.id].standing_origin, undefined);

  let reverseOrder = stageDecisionChoice(emptyHumanDraft(4), decisionGraph, decision.id, "Small");
  reverseOrder = stageProposalDecision(reverseOrder, decisionGraph, targeted.id, "approved");
  assert.deepEqual(reverseOrder.proposals, {});

  const restoredWithTargetedResolution = deserializeHumanDraft(
    serializeHumanDraft({
      ...reverseOrder,
      proposals: { [targeted.id]: { decision: "rejected" } },
    }),
  );
  assert.deepEqual(
    normalizeHumanDraft(restoredWithTargetedResolution, decisionGraph).proposals,
    {},
  );
});

test("serialization survives localStorage round trips and request conversion strips editor metadata", () => {
  let draft = stageNodeEdit(emptyHumanDraft(4), graph, "hyp/example", {
    title: "Revised",
    statement: "Sharper statement",
  });
  draft = stageProposalDecision(draft, graph, "proposal/1", "rejected");
  const restored = deserializeHumanDraft(serializeHumanDraft(draft));
  assert.deepEqual(restored, draft);
  assert.equal("ambiguities" in emptyHumanDraft(4), false);
  assert.equal(humanDraftStorageKey("project one"), "rcp:human-draft:project one");
  assert.equal(deserializeHumanDraft("not json"), null);

  const legacyStoredDraft = JSON.parse(serializeHumanDraft(draft));
  legacyStoredDraft.ambiguities = {
    "ambiguity/1": { status: "dismissed" },
  };
  const restoredLegacyDraft = deserializeHumanDraft(JSON.stringify(legacyStoredDraft));
  assert.deepEqual(restoredLegacyDraft, draft);
  assert.equal("ambiguities" in restoredLegacyDraft, false);

  assert.deepEqual(toHumanSyncRequest(restored, graph), {
    base_revision: 4,
    removed_node_ids: [],
    nodes: [
      {
        node_id: "hyp/example",
        base_updated_rev: 4,
        changes: { title: "Revised", statement: "Sharper statement" },
        standing: "asserted",
      },
    ],
    proposals: [{ proposal_id: "proposal/1", decision: "rejected" }],
    ontology: null,
    custom_nodes: [],
  });
});

test("ontology and custom nodes round trip, count, present, and serialize through project Sync", () => {
  const ontology = {
    types: [
      {
        name: "mechanism_hypothesis",
        definition: "A causal mechanism claim.",
        base_type: "hypothesis",
        layer: "epistemic",
        deprecated: false,
      },
    ],
    fields: [
      {
        owner_type: "mechanism_hypothesis",
        name: "mechanism",
        definition: "The causal mechanism.",
        kind: "text",
        required: true,
        agent_writable: false,
        deprecated: false,
      },
    ],
    relations: [],
  };
  const customNode = {
    id: "mechanism_hypothesis/replanning",
    type: "hypothesis",
    extension_type: "mechanism_hypothesis",
    extension_fields: { mechanism: "Replanning restores update directions." },
    title: "Replanning mechanism",
    statement: "Replanning preserves plasticity.",
    standing: "asserted",
    created_rev: 0,
    updated_rev: 0,
    source_refs: [],
  };
  let draft = stageOntology(emptyHumanDraft(4), graph, ontology);
  draft = stageCustomNode(draft, customNode);
  assert.equal(humanDraftChangeCount(draft), 2);
  assert.deepEqual(deserializeHumanDraft(serializeHumanDraft(draft)), draft);
  const presented = applyHumanDraft(graph, draft);
  assert.deepEqual(presented.ontology, ontology);
  assert.equal(presented.nodes[customNode.id].draft_touched, true);
  assert.deepEqual(toHumanSyncRequest(draft, graph), {
    base_revision: 4,
    removed_node_ids: [],
    nodes: [],
    proposals: [],
    ontology,
    custom_nodes: [customNode],
  });
  const ontologyOnly = unstageCustomNode(draft, customNode.id);
  assert.equal(humanDraftChangeCount(ontologyOnly), 1);
  assert.deepEqual(toHumanSyncRequest(ontologyOnly, graph).custom_nodes, []);
});

test("node removal is persistent, reversible, normalized, and mutually exclusive with node changes", () => {
  const contested = {
    ...graph.nodes["hyp/example"],
    id: "hyp/removable",
    title: "Removable hypothesis",
    standing: "contested",
  };
  const experiment = {
    ...contested,
    id: "exp/running",
    type: "experiment",
    title: "Running experiment",
    standing: "asserted",
    attempts: [],
  };
  const removalGraph = {
    ...graph,
    nodes: { ...graph.nodes, [contested.id]: contested, [experiment.id]: experiment },
    edges: {
      "edge/incident": {
        id: "edge/incident",
        source: contested.id,
        target: "hyp/example",
      },
    },
  };

  let draft = stageNodeRemoval(emptyHumanDraft(4), removalGraph, contested.id);
  assert.deepEqual(draft.removed_node_ids, [contested.id]);
  assert.equal(humanDraftChangeCount(draft), 1);
  assert.equal(applyHumanDraft(removalGraph, draft).nodes[contested.id].draft_touched, true);
  assert.deepEqual(toHumanSyncRequest(draft, removalGraph).removed_node_ids, [contested.id]);

  assert.equal(stageNodeStanding(draft, removalGraph, contested.id, "accepted"), draft);
  assert.equal(stageNodeEdit(draft, removalGraph, contested.id, { title: "Ignored" }), draft);

  draft = unstageNodeRemoval(draft, contested.id);
  assert.equal(humanDraftChangeCount(draft), 0);

  const changed = stageNodeStanding(emptyHumanDraft(4), removalGraph, contested.id, "asserted");
  assert.equal(stageNodeRemoval(changed, removalGraph, contested.id), changed);
  assert.deepEqual(
    stageNodeRemoval(emptyHumanDraft(4), removalGraph, "hyp/example").removed_node_ids,
    [],
  );
  assert.deepEqual(
    stageNodeRemoval(emptyHumanDraft(4), removalGraph, experiment.id, true).removed_node_ids,
    [],
  );

  const vanished = { ...draft, removed_node_ids: ["hyp/missing"] };
  assert.deepEqual(normalizeHumanDraft(vanished, removalGraph).removed_node_ids, []);

  const legacy = JSON.parse(serializeHumanDraft(emptyHumanDraft(4)));
  delete legacy.removed_node_ids;
  assert.deepEqual(deserializeHumanDraft(JSON.stringify(legacy)).removed_node_ids, []);
});

test("Sync conflicts preserve exact removal guards and rewrite only revision conflicts", () => {
  const accepted =
    "Accepted node hyp/example cannot be removed; withdraw its acceptance and Sync before removing it.";
  assert.deepEqual(humanSyncFailure(new ApiError(accepted, 409)), {
    text: accepted,
    revisionConflict: false,
  });

  const active =
    "Experiment exp/running cannot be removed while its bounded experiment loop is active.";
  assert.deepEqual(humanSyncFailure(new ApiError(active, 409)), {
    text: active,
    revisionConflict: false,
  });

  assert.deepEqual(
    humanSyncFailure(new ApiError("The graph changed after this draft began.", 409)),
    {
      text: "The project moved again before Sync. Your staged changes were kept and refreshed.",
      revisionConflict: true,
    },
  );
});

test("canonical movement rebases the draft and quarantines only nodes that moved", () => {
  const unchanged = {
    ...graph.nodes["hyp/example"],
    id: "hyp/unchanged",
    title: "Unchanged canonical title",
  };
  const before = { ...graph, nodes: { ...graph.nodes, [unchanged.id]: unchanged } };
  let draft = stageNodeEdit(emptyHumanDraft(4), before, "hyp/example", {
    title: "My changed-node title",
  });
  draft = stageNodeEdit(draft, before, unchanged.id, { title: "My unchanged-node title" });

  const after = {
    ...before,
    revision: 5,
    nodes: {
      ...before.nodes,
      "hyp/example": {
        ...before.nodes["hyp/example"],
        title: "Incoming canonical title",
        updated_rev: 5,
      },
    },
  };
  const rebased = normalizeHumanDraft(draft, after);

  assert.equal(rebased.base_revision, 5);
  assert.equal(rebased.nodes["hyp/example"].base_updated_rev, 4);
  assert.equal(rebased.nodes[unchanged.id].base_updated_rev, 4);
  assert.equal(humanDraftBehindCount(rebased, after), 1);
  assert.equal(humanDraftCommittableCount(rebased, after), 2);
  assert.deepEqual(
    toHumanSyncRequest(rebased, after).nodes.map((entry) => entry.node_id),
    [unchanged.id],
  );
  assert.equal(applyHumanDraft(after, rebased).nodes["hyp/example"].title, "My changed-node title");
});

test("editing a behind node re-pins the entry and makes it committable", () => {
  const draft = stageNodeEdit(emptyHumanDraft(4), graph, "hyp/example", {
    title: "My staged title",
  });
  const moved = {
    ...graph,
    revision: 5,
    nodes: {
      ...graph.nodes,
      "hyp/example": {
        ...graph.nodes["hyp/example"],
        title: "Incoming title",
        updated_rev: 5,
      },
    },
  };
  const behind = normalizeHumanDraft(draft, moved);
  const touched = stageNodeEdit(behind, moved, "hyp/example", {
    statement: "Touched after seeing incoming",
  });

  assert.equal(touched.nodes["hyp/example"].base_updated_rev, 5);
  assert.equal(humanDraftBehindCount(touched, moved), 0);
  assert.equal(toHumanSyncRequest(touched, moved).nodes.length, 1);
});

test("Apply can swap an incoming field in and restore the displaced staged value", () => {
  const staged = stageNodeEdit(emptyHumanDraft(4), graph, "hyp/example", {
    title: "My staged title",
  });
  const moved = {
    ...graph,
    revision: 5,
    nodes: {
      ...graph.nodes,
      "hyp/example": {
        ...graph.nodes["hyp/example"],
        title: "Incoming title",
        updated_rev: 5,
      },
    },
  };
  const behind = normalizeHumanDraft(staged, moved);
  const applied = stageNodeEdit(behind, moved, "hyp/example", {}, ["title"]);
  const restored = stageNodeEdit(applied, moved, "hyp/example", { title: "My staged title" }, [
    "title",
  ]);

  assert.deepEqual(applied.nodes, {});
  assert.equal(restored.nodes["hyp/example"].base_updated_rev, 5);
  assert.equal(restored.nodes["hyp/example"].changes.title, "My staged title");
});

test("Sync retains quarantined node edits while clearing what it committed", () => {
  const unchanged = {
    ...graph.nodes["hyp/example"],
    id: "hyp/unchanged",
    title: "Unchanged canonical title",
  };
  const moved = {
    ...graph,
    revision: 5,
    nodes: {
      ...graph.nodes,
      "hyp/example": {
        ...graph.nodes["hyp/example"],
        title: "Incoming title",
        updated_rev: 5,
      },
      [unchanged.id]: unchanged,
    },
  };
  let draft = stageNodeEdit(emptyHumanDraft(4), graph, "hyp/example", {
    title: "My staged title",
  });
  draft = {
    ...normalizeHumanDraft(draft, moved),
    nodes: {
      ...normalizeHumanDraft(draft, moved).nodes,
      [unchanged.id]: {
        base_updated_rev: 4,
        changes: { title: "Committed title" },
        standing: "asserted",
      },
    },
  };
  const afterSync = {
    ...moved,
    revision: 6,
    nodes: {
      ...moved.nodes,
      [unchanged.id]: { ...unchanged, title: "Committed title", updated_rev: 6 },
    },
  };

  const retained = retainBehindDraftAfterSync(draft, moved, afterSync);
  assert.deepEqual(Object.keys(retained.nodes), ["hyp/example"]);
  assert.equal(retained.base_revision, 6);
  assert.equal(humanDraftBehindCount(retained, afterSync), 1);
});

test("ontology keeps its own stale gate after ordinary draft rebasing", () => {
  const ontologyDraft = stageOntology(emptyHumanDraft(4), graph, {
    types: [
      {
        name: "mechanism",
        definition: "A mechanism.",
        base_type: "hypothesis",
        layer: "epistemic",
        deprecated: false,
      },
    ],
    fields: [],
    relations: [],
  });
  const moved = { ...graph, revision: 5 };
  const rebased = normalizeHumanDraft(ontologyDraft, moved);

  assert.equal(rebased.base_revision, 5);
  assert.equal(rebased.ontology_base_revision, 4);
  assert.equal(humanDraftOntologyIsStale(rebased, moved), true);
});
