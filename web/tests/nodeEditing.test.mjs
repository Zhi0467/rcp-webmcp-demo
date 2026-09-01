import assert from "node:assert/strict";
import { after, test } from "node:test";
import { createServer } from "vite";

import { changedNodeFields, editableNodeFields, nodeEditDraft } from "../src/nodeEditing.ts";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { emptyHumanDraft, humanDraftChangeCount, stageAttemptRelease, toHumanSyncRequest } =
  await server.ssrLoadModule("/src/humanDraft.ts");

after(() => server.close());

const hypothesis = {
  id: "hyp/example",
  type: "hypothesis",
  title: "Existing title",
  standing: "accepted",
  created_rev: 2,
  updated_rev: 4,
  source_refs: [],
  extension_fields: {},
  statement: "Existing statement",
  rationale: "",
  predictions: ["First prediction", "Second prediction"],
  scope: "Single-domain evaluation",
  status: "active",
};

test("editable fields mirror the human-editable allowlist for every node type", () => {
  const keys = (type) =>
    editableNodeFields({
      ...hypothesis,
      type,
      status: type === "decision" ? "open" : hypothesis.status,
    }).map((field) => field.key);
  assert.deepEqual(keys("research_question"), ["title", "question", "motivation", "scope"]);
  assert.deepEqual(keys("hypothesis"), ["title", "statement", "rationale", "predictions", "scope"]);
  assert.deepEqual(keys("decision"), [
    "title",
    "question",
    "options",
    "status",
    "rationale",
    "consequences",
  ]);
  assert.deepEqual(keys("experiment"), [
    "title",
    "objective",
    "design",
    "expected_outcomes",
    "interpretation_rules",
    "completion_criteria",
    "invocation_ceiling",
    "current_summary",
    "next_action",
  ]);
  assert.deepEqual(keys("evidence"), ["title", "observation", "interpretation"]);
  assert.deepEqual(keys("blocker"), [
    "title",
    "status",
    "description",
    "resolution_condition",
    "recommended_action",
  ]);
});

test("only queued Decisions expose a human-editable queue status", () => {
  const decision = {
    ...hypothesis,
    type: "decision",
    question: "Choose a method",
    options: ["A", "B"],
    rationale: null,
    consequences: [],
  };

  for (const status of ["open", "ready", "revisit"]) {
    const field = editableNodeFields({ ...decision, status }).find((item) => item.key === "status");
    assert.deepEqual(field?.options, [
      { value: "open", label: "Open" },
      { value: "ready", label: "Ready" },
      { value: "revisit", label: "Revisit" },
    ]);
  }
  for (const status of ["decided", "superseded"]) {
    assert.equal(
      editableNodeFields({ ...decision, status }).some((item) => item.key === "status"),
      false,
    );
  }
});

test("active base and custom extension fields are editable as one complete object", () => {
  const ontology = {
    types: [
      {
        name: "mechanism_hypothesis",
        definition: "Mechanism claim",
        base_type: "hypothesis",
        layer: "epistemic",
        deprecated: false,
      },
    ],
    fields: [
      {
        owner_type: "hypothesis",
        name: "prior",
        definition: "Prior",
        kind: "number",
        required: false,
        agent_writable: false,
        deprecated: false,
      },
      {
        owner_type: "mechanism_hypothesis",
        name: "mechanism",
        definition: "Mechanism",
        kind: "text",
        required: true,
        agent_writable: true,
        deprecated: false,
      },
      {
        owner_type: "mechanism_hypothesis",
        name: "old_note",
        definition: "Old note",
        kind: "text",
        required: false,
        agent_writable: true,
        deprecated: true,
      },
    ],
    relations: [],
  };
  const node = {
    ...hypothesis,
    extension_type: "mechanism_hypothesis",
    extension_fields: { prior: 0.4, mechanism: "Old mechanism", old_note: "Still readable" },
  };
  const fields = editableNodeFields(node, ontology);
  assert.deepEqual(
    fields.slice(-2).map((field) => field.key),
    ["extension_fields.prior", "extension_fields.mechanism"],
  );
  assert.equal(
    fields.some((field) => field.key === "extension_fields.old_note"),
    false,
  );
  const draft = nodeEditDraft(node, ontology);
  draft["extension_fields.mechanism"] = "Updated mechanism";
  assert.deepEqual(changedNodeFields(node, draft, ontology), {
    extension_fields: {
      prior: 0.4,
      mechanism: "Updated mechanism",
      old_note: "Still readable",
    },
  });
  draft["extension_fields.prior"] = "";
  assert.deepEqual(changedNodeFields(node, draft, ontology), {
    extension_fields: {
      mechanism: "Updated mechanism",
      old_note: "Still readable",
    },
  });
});

test("node drafts render lists one item per line and submit only normalized changes", () => {
  const draft = nodeEditDraft(hypothesis);
  assert.equal(draft.predictions, "First prediction\nSecond prediction");
  assert.equal(draft.scope, "Single-domain evaluation");
  assert.equal("confidence" in hypothesis, false);
  draft.title = "  Existing title  ";
  draft.predictions = "First prediction\n\n Revised second prediction  ";
  assert.deepEqual(changedNodeFields(hypothesis, draft), {
    predictions: ["First prediction", "Revised second prediction"],
  });
});

test("blank nullable prose becomes null while an existing null is unchanged", () => {
  const decision = {
    ...hypothesis,
    type: "decision",
    question: "Choose a method",
    options: ["A", "B"],
    rationale: "Current reason",
    consequences: [],
  };
  const draft = nodeEditDraft(decision);
  draft.rationale = "   ";
  assert.deepEqual(changedNodeFields(decision, draft), { rationale: null });
  const alreadyBlank = { ...decision, rationale: null };
  assert.deepEqual(changedNodeFields(alreadyBlank, nodeEditDraft(alreadyBlank)), {});
});

test("Blocker status is a closed human-labelled choice and stages a normalized change", () => {
  const blocker = {
    ...hypothesis,
    id: "blocker/example",
    type: "blocker",
    standing: "asserted",
    blocker_type: "scientific",
    status: "open",
    description: "The measurement is missing.",
    resolution_condition: "Record the missing measurement.",
    recommended_action: null,
  };
  const field = editableNodeFields(blocker).find((item) => item.key === "status");
  assert.deepEqual(field, {
    key: "status",
    label: "Status",
    kind: "select",
    options: [
      { value: "open", label: "Open" },
      { value: "resolved", label: "Resolved" },
      { value: "superseded", label: "Superseded" },
    ],
  });

  const draft = nodeEditDraft(blocker);
  assert.equal(draft.status, "open");
  draft.status = "  resolved  ";
  assert.deepEqual(changedNodeFields(blocker, draft), { status: "resolved" });
});

test("the experiment invocation ceiling is a positive integer field in the human draft", () => {
  const experiment = {
    ...hypothesis,
    type: "experiment",
    objective: "Run the benchmark",
    design: "One controlled run",
    expected_outcomes: [],
    interpretation_rules: [],
    completion_criteria: [],
    invocation_ceiling: 5,
    current_summary: "Ready",
    next_action: null,
  };
  const field = editableNodeFields(experiment).find((item) => item.key === "invocation_ceiling");
  assert.deepEqual(field, {
    key: "invocation_ceiling",
    label: "Invocation ceiling",
    kind: "number",
    min: 1,
    integer: true,
  });
  const draft = nodeEditDraft(experiment);
  draft.invocation_ceiling = "7";
  assert.deepEqual(changedNodeFields(experiment, draft), { invocation_ceiling: 7 });
});

test("releasing an attempt stages only that attempt and drops once it closes", () => {
  const experiment = {
    id: "exp/stuck",
    type: "experiment",
    title: "Stuck",
    standing: "asserted",
    created_rev: 1,
    updated_rev: 4,
    source_refs: [],
    extension_fields: {},
    objective: "Train",
    attempts: [
      {
        id: "attempt-1",
        sequence: 1,
        purpose: "Train",
        attempt_kind: "external_run",
        decision_bundle: [],
        status: "running",
        job_refs: [],
      },
      {
        id: "attempt-2",
        sequence: 2,
        purpose: "Retry",
        attempt_kind: "external_run",
        decision_bundle: [],
        status: "failed",
        job_refs: [],
      },
    ],
  };
  const graph = { revision: 4, nodes: { "exp/stuck": experiment }, edges: {}, ontology: null };

  const staged = stageAttemptRelease(emptyHumanDraft(4), graph, "exp/stuck", "attempt-1");

  assert.deepEqual(toHumanSyncRequest(staged, graph).nodes, [
    { node_id: "exp/stuck", base_updated_rev: 4, changes: {}, cancel_attempt_ids: ["attempt-1"] },
  ]);
  assert.equal(humanDraftChangeCount(staged), 1);

  // Releasing a finished attempt is meaningless, and so is re-releasing one the
  // graph already closed underneath the draft.
  const closed = {
    ...graph,
    nodes: {
      "exp/stuck": {
        ...experiment,
        attempts: [{ ...experiment.attempts[0], status: "cancelled" }, experiment.attempts[1]],
      },
    },
  };
  assert.deepEqual(
    toHumanSyncRequest(
      stageAttemptRelease(emptyHumanDraft(4), closed, "exp/stuck", "attempt-1"),
      closed,
    ).nodes,
    [],
  );
  assert.deepEqual(
    toHumanSyncRequest(
      stageAttemptRelease(emptyHumanDraft(4), graph, "exp/stuck", "attempt-2"),
      graph,
    ).nodes,
    [],
  );
});
