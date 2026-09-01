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
  ACTIVE_PROJECT_CACHE_OBSERVE_INTERVAL_MS,
  OPEN_PROJECT_HEARTBEAT_INTERVAL_MS,
  cacheProjectTabState,
  cachedSnapshotCanReplace,
  canonicalRevisionNeedsReload,
  inactiveProjectTabState,
  humanSyncSuccessNotice,
  latestSnapshotRequestCanApply,
  loadCanonicalRevision,
  persistProjectHumanDraft,
  proposalChoicesClearedNotice,
  projectIdsForCacheHeartbeat,
  projectIsStillReadable,
  projectTabStateForOpen,
  reconcileInactiveProjectTabState,
  singleFlightProjectCacheHeartbeat,
  startProjectCachePolling,
} = await server.ssrLoadModule("/src/App.tsx");

after(() => server.close());

test("canonical revision polling uses only the lightweight project endpoint", async () => {
  const requested = [];
  const revision = await loadCanonicalRevision(async (path) => {
    requested.push(path);
    return { revision: 12 };
  }, "/api/projects/project-1");

  assert.equal(revision, 12);
  assert.deepEqual(requested, ["/api/projects/project-1/cached/revision"]);
});

test("the per-project display cache is bounded and refreshed as an LRU", () => {
  const cache = new Map();
  cacheProjectTabState(cache, "alpha", { revision: 1 }, 2);
  cacheProjectTabState(cache, "beta", { revision: 2 }, 2);
  cacheProjectTabState(cache, "alpha", { revision: 3 }, 2);
  cacheProjectTabState(cache, "gamma", { revision: 4 }, 2);

  assert.deepEqual([...cache.keys()], ["alpha", "gamma"]);
  assert.deepEqual(cache.get("alpha"), { revision: 3 });
});

test("a cached response cannot move the rendered project backwards", () => {
  const snapshot = (id, revision) => ({ id, graph: { revision } });
  assert.equal(cachedSnapshotCanReplace("alpha", 8, snapshot("alpha", 7)), false);
  assert.equal(cachedSnapshotCanReplace("alpha", 8, snapshot("alpha", 8)), true);
  assert.equal(cachedSnapshotCanReplace("alpha", 8, snapshot("beta", 2)), true);
});

test("only the latest started snapshot request may update one project", () => {
  assert.equal(latestSnapshotRequestCanApply(12, 12), true);
  assert.equal(latestSnapshotRequestCanApply(12, 11), false);
  assert.equal(latestSnapshotRequestCanApply(undefined, 1), false);
});

test("returning to a cached tab restores its complete render state without an empty loading frame", () => {
  const retained = {
    project: { id: "alpha", graph: { revision: 8 }, paper: { content: "draft" } },
    tasks: [{ operation_id: "task-1" }],
    watchers: [{ watcher_id: "watcher-1" }],
    chatSummaries: [{ chat_id: "chat-1" }],
    chatTranscripts: new Map([["chat-1", { chat_id: "chat-1", messages: [] }]]),
    historyRevisionSummaries: [{ to_revision: 8 }],
    viewState: {
      view: "chats",
      panelScroll: [["chats", 420]],
      researchSubview: "dag",
      dagViewport: { zoom: 1.2, scrollLeft: 10, scrollTop: 20 },
    },
  };
  const cache = new Map([
    ["alpha", retained],
    ["beta", { project: { id: "beta" } }],
  ]);

  const open = projectTabStateForOpen(cache, "alpha");
  assert.equal(open.loading, false);
  assert.strictEqual(open.state, retained);
  assert.equal(open.state.project.graph.revision, 8);
  assert.equal(open.state.tasks[0].operation_id, "task-1");
  assert.equal(open.state.watchers[0].watcher_id, "watcher-1");
  assert.equal(open.state.chatSummaries[0].chat_id, "chat-1");
  assert.equal(open.state.chatTranscripts.get("chat-1").chat_id, "chat-1");
  assert.equal(open.state.historyRevisionSummaries[0].to_revision, 8);
  assert.deepEqual(open.state.viewState.panelScroll, [["chats", 420]]);
  assert.deepEqual([...cache.keys()], ["beta", "alpha"]);
  assert.equal(projectTabStateForOpen(cache, "missing"), null);
});

test("canonical state reloads only after the accepted revision advances", () => {
  assert.equal(canonicalRevisionNeedsReload(8, 7), true);
  assert.equal(canonicalRevisionNeedsReload(7, 7), false);
  assert.equal(canonicalRevisionNeedsReload(6, 7), false);
});

test("visible cache polling schedules every open tab and observes the active tab every second", () => {
  const intervals = new Map();
  const cleared = [];
  let nextIntervalId = 1;
  let hidden = false;
  let visibilityListener = null;
  let listenerRemoved = false;
  let sweeps = 0;
  let activeObservations = 0;
  const stop = startProjectCachePolling(
    {
      setInterval(callback, delay) {
        const id = nextIntervalId++;
        intervals.set(id, { callback, delay });
        return id;
      },
      clearInterval(id) {
        cleared.push(id);
      },
    },
    {
      isHidden: () => hidden,
      listen(callback) {
        visibilityListener = callback;
        return () => {
          listenerRemoved = true;
        };
      },
    },
    () => {
      sweeps += 1;
    },
    () => {
      activeObservations += 1;
    },
  );

  const scheduled = [...intervals.values()];
  assert.deepEqual(
    scheduled.map((entry) => entry.delay).sort((left, right) => left - right),
    [ACTIVE_PROJECT_CACHE_OBSERVE_INTERVAL_MS, OPEN_PROJECT_HEARTBEAT_INTERVAL_MS],
  );
  assert.equal(OPEN_PROJECT_HEARTBEAT_INTERVAL_MS, 3_000);
  assert.equal(ACTIVE_PROJECT_CACHE_OBSERVE_INTERVAL_MS, 1_000);
  assert.equal(sweeps, 0, "installing or switching must not synchronously heartbeat");
  scheduled.find((entry) => entry.delay === OPEN_PROJECT_HEARTBEAT_INTERVAL_MS).callback();
  scheduled.find((entry) => entry.delay === ACTIVE_PROJECT_CACHE_OBSERVE_INTERVAL_MS).callback();
  assert.equal(sweeps, 1);
  assert.equal(activeObservations, 1);

  hidden = true;
  scheduled.forEach((entry) => entry.callback());
  assert.equal(sweeps, 1);
  assert.equal(activeObservations, 1);
  hidden = false;
  visibilityListener();
  assert.equal(sweeps, 2, "visibility resume immediately sweeps every open tab");
  assert.equal(activeObservations, 1);

  stop();
  assert.deepEqual(cleared.sort(), [1, 2]);
  assert.equal(listenerRemoved, true);
});

test("heartbeat targets are unique and stale inactive results cannot update closed or active tabs", () => {
  const tabs = [
    { id: "alpha", name: "Alpha" },
    { id: "beta", name: "Beta" },
    { id: "alpha", name: "Alpha duplicate" },
  ];
  const alpha = { project: { id: "alpha" } };
  const beta = { project: { id: "beta" } };
  const cache = new Map([
    ["alpha", alpha],
    ["beta", beta],
  ]);

  assert.deepEqual(projectIdsForCacheHeartbeat(tabs), ["alpha", "beta"]);
  assert.strictEqual(inactiveProjectTabState(cache, tabs, "alpha", "beta"), beta);
  assert.equal(inactiveProjectTabState(cache, tabs, "alpha", "alpha"), null);
  assert.equal(inactiveProjectTabState(cache, tabs.slice(0, 1), "alpha", "beta"), null);
});

test("overlapping active and all-tab heartbeats are single-flight per project", async () => {
  const inFlight = new Map();
  let calls = 0;
  let finish;
  const run = () => {
    calls += 1;
    return new Promise((resolve) => {
      finish = resolve;
    });
  };

  const first = singleFlightProjectCacheHeartbeat(inFlight, "alpha", run);
  const overlapping = singleFlightProjectCacheHeartbeat(inFlight, "alpha", run);
  assert.strictEqual(overlapping, first);
  assert.equal(calls, 1);
  finish();
  await first;
  assert.equal(inFlight.has("alpha"), false);

  const next = singleFlightProjectCacheHeartbeat(inFlight, "alpha", async () => {
    calls += 1;
  });
  await next;
  assert.equal(calls, 2);
});

test("inactive advancement rebases only snapshot and draft while retaining the tab workspace", () => {
  const node = {
    id: "hyp/example",
    type: "hypothesis",
    title: "Canonical before",
    statement: "Statement",
    standing: "accepted",
    created_rev: 2,
    updated_rev: 4,
    source_refs: [],
    extension_fields: {},
  };
  const graph = {
    revision: 4,
    nodes: { [node.id]: node },
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
  const draft = {
    version: 1,
    base_revision: 4,
    nodes: {
      [node.id]: {
        base_updated_rev: 4,
        changes: { title: "My staged title" },
        standing: "asserted",
        standing_origin: "edit",
      },
    },
    removed_node_ids: [],
    proposals: {},
    ontology: null,
    custom_nodes: {},
  };
  const retained = {
    project: { id: "alpha", graph },
    humanDraft: draft,
    tasks: [{ operation_id: "task-1" }],
    watchers: [{ watcher_id: "watcher-1" }],
    chatSummaries: [{ chat_id: "chat-1" }],
    chatTranscripts: new Map([["chat-1", { chat_id: "chat-1" }]]),
    historyRevisionSummaries: [{ to_revision: 4 }],
    viewState: { view: "chats", panelScroll: [["chats", 420]] },
  };
  const movedNode = {
    ...node,
    title: "Incoming canonical title",
    updated_rev: 5,
  };
  const snapshot = {
    id: "alpha",
    snapshot_freshness: "fresh",
    attention: {
      pending_proposal_ids: [],
      decisions_awaiting_choice_ids: [],
      open_blocker_ids: [],
    },
    graph: { ...graph, revision: 5, nodes: { [node.id]: movedNode } },
  };

  const next = reconcileInactiveProjectTabState(retained, snapshot);
  assert.notStrictEqual(next, retained);
  assert.strictEqual(next.tasks, retained.tasks);
  assert.strictEqual(next.watchers, retained.watchers);
  assert.strictEqual(next.chatSummaries, retained.chatSummaries);
  assert.strictEqual(next.chatTranscripts, retained.chatTranscripts);
  assert.strictEqual(next.historyRevisionSummaries, retained.historyRevisionSummaries);
  assert.strictEqual(next.viewState, retained.viewState);
  assert.equal(next.project.graph.revision, 5);
  assert.equal(next.humanDraft.base_revision, 5);
  assert.equal(next.humanDraft.nodes[node.id].base_updated_rev, 4);
  assert.strictEqual(
    reconcileInactiveProjectTabState(next, { ...snapshot, graph }),
    next,
    "a lower cached revision must never replace retained state",
  );

  const stored = new Map();
  const storage = {
    setItem(key, value) {
      stored.set(key, value);
    },
    removeItem(key) {
      stored.delete(key);
    },
  };
  persistProjectHumanDraft(storage, "alpha", next.humanDraft);
  assert.equal(JSON.parse(stored.get("rcp:human-draft:alpha")).base_revision, 5);
  persistProjectHumanDraft(storage, "alpha", null);
  assert.equal(stored.has("rcp:human-draft:alpha"), false);
});

test("authoritative inactive snapshots prune resolved choices and clear missing node targets", () => {
  const node = {
    id: "hyp/retained",
    type: "hypothesis",
    title: "Retained node",
    statement: "Retained statement",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 4,
    source_refs: [],
    extension_fields: {},
  };
  const removedNode = { ...node, id: "hyp/removed", title: "Removed node" };
  const proposal = (id, status) => ({
    id,
    title: id,
    card: { situation_cold: "", why_human_now: "", consequences: "", decision_needed: "" },
    ops: [],
    related_node_ids: [node.id],
    related_edge_ids: [],
    related_config_keys: [],
    base_rev: 4,
    raised_rev: 4,
    resolved_rev: status === "pending" ? null : 5,
    status,
  });
  const pending = proposal("proposal/pending", "pending");
  const withdrawn = proposal("proposal/withdrawn", "withdrawn");
  const oldGraph = {
    revision: 4,
    nodes: { [node.id]: node, [removedNode.id]: removedNode },
    edges: {},
    proposals: { [pending.id]: pending, [withdrawn.id]: { ...withdrawn, status: "pending" } },
    ambiguities: {},
    glossary: {},
    ontology: { types: [], fields: [], relations: [] },
    validation_messages: [],
    belief_transitions: [],
    replay_status: "complete",
    replay_failure: null,
  };
  const draft = {
    version: 1,
    base_revision: 4,
    nodes: {
      [node.id]: {
        base_updated_rev: 4,
        changes: { title: "Retained staged title" },
        standing: "asserted",
        standing_origin: "edit",
      },
    },
    removed_node_ids: [],
    proposals: {
      [pending.id]: { decision: "rejected" },
      [withdrawn.id]: { decision: "approved" },
      "proposal/missing": { decision: "rejected" },
    },
    ontology: null,
    custom_nodes: {},
  };
  const retained = {
    project: { id: "alpha", graph: oldGraph },
    humanDraft: draft,
    selectedNodeId: removedNode.id,
    companionNodeId: node.id,
    floatingChat: { chatId: "chat/removed", nodeId: removedNode.id },
  };
  const snapshot = {
    id: "alpha",
    snapshot_freshness: "fresh",
    attention: {
      pending_proposal_ids: [pending.id],
      decisions_awaiting_choice_ids: [],
      open_blocker_ids: [],
    },
    graph: {
      ...oldGraph,
      revision: 5,
      nodes: { [node.id]: node },
      proposals: { [pending.id]: pending, [withdrawn.id]: withdrawn },
    },
  };

  const next = reconcileInactiveProjectTabState(retained, snapshot);

  assert.equal(next.selectedNodeId, null);
  assert.equal(next.companionNodeId, node.id);
  assert.equal(next.floatingChat, null);
  assert.deepEqual(next.humanDraft.proposals, {
    [pending.id]: { decision: "rejected" },
  });
  assert.deepEqual(next.humanDraft.nodes[node.id].changes, { title: "Retained staged title" });
  assert.deepEqual(next.draftReconciliationDiscardedProposalIds, [
    "proposal/missing",
    "proposal/withdrawn",
  ]);

  const stored = new Map();
  persistProjectHumanDraft(
    {
      setItem(key, value) {
        stored.set(key, value);
      },
      removeItem(key) {
        stored.delete(key);
      },
    },
    "alpha",
    next.humanDraft,
  );
  assert.deepEqual(Object.keys(JSON.parse(stored.get("rcp:human-draft:alpha")).proposals), [
    pending.id,
  ]);
  assert.equal(
    proposalChoicesClearedNotice(next.draftReconciliationDiscardedProposalIds),
    "Externally resolved proposal choices were cleared: proposal/missing, proposal/withdrawn.",
  );

  const stale = reconcileInactiveProjectTabState(retained, {
    ...snapshot,
    snapshot_freshness: "stale",
  });
  assert.strictEqual(stale, retained);
});

test("Sync reports stale withdrawals without claiming their proposed changes applied", () => {
  const nextGraph = {
    proposals: {
      "proposal/stale": { status: "withdrawn" },
      "proposal/applied": { status: "approved" },
    },
  };
  const submitted = [
    { proposal_id: "proposal/applied", decision: "approved" },
    { proposal_id: "proposal/stale", decision: "rejected" },
  ];

  assert.equal(
    humanSyncSuccessNotice(9, submitted, nextGraph),
    "Synced revision 9. Stale proposals were withdrawn and their proposed changes were not applied: proposal/stale.",
  );
  assert.equal(humanSyncSuccessNotice(9, submitted.slice(0, 1), nextGraph), "Synced revision 9.");
});

test("a project still on the filtered index keeps its tab open", async () => {
  const readable = await projectIsStillReadable(
    async () => [{ id: "alpha" }, { id: "beta" }],
    "alpha",
  );

  assert.equal(readable, true);
});

test("a project absent from the filtered index is no longer readable", async () => {
  const requested = [];
  const readable = await projectIsStillReadable(async (path) => {
    requested.push(path);
    return [{ id: "beta" }];
  }, "alpha");

  assert.equal(readable, false);
  assert.deepEqual(requested, ["/api/projects"]);
});

test("an index that cannot be reached is not evidence that a project is gone", async () => {
  const readable = await projectIsStillReadable(async () => {
    throw new Error("offline");
  }, "alpha");

  assert.equal(readable, true);
});
