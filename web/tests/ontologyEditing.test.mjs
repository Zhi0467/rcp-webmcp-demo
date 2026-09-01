import assert from "node:assert/strict";
import test from "node:test";

import {
  activeCustomTypes,
  activeFieldsForNode,
  canRemoveOntologyType,
  makeCustomNode,
  removeOntologyType,
  upsertOntologyType,
} from "../src/ontologyEditing.ts";

const ontology = {
  types: [
    {
      name: "mechanism_hypothesis",
      definition: "A mechanism claim.",
      base_type: "hypothesis",
      layer: "epistemic",
      deprecated: false,
    },
    {
      name: "old_type",
      definition: "Old type.",
      base_type: "evidence",
      layer: "epistemic",
      deprecated: true,
    },
  ],
  fields: [
    {
      owner_type: "hypothesis",
      name: "prior",
      definition: "Prior.",
      kind: "number",
      required: false,
      agent_writable: false,
      deprecated: false,
    },
    {
      owner_type: "mechanism_hypothesis",
      name: "mechanism",
      definition: "Mechanism.",
      kind: "text",
      required: true,
      agent_writable: true,
      deprecated: false,
    },
    {
      owner_type: "mechanism_hypothesis",
      name: "old_note",
      definition: "Old note.",
      kind: "text",
      required: false,
      agent_writable: true,
      deprecated: true,
    },
  ],
  relations: [
    {
      name: "explains",
      definition: "Explains.",
      source_types: ["mechanism_hypothesis"],
      target_types: ["evidence", "old_type"],
      layer: "epistemic",
      deprecated: false,
    },
    {
      name: "only_old",
      definition: "Old.",
      source_types: ["old_type"],
      target_types: ["evidence"],
      layer: "epistemic",
      deprecated: true,
    },
  ],
};

test("active authoring excludes deprecated types and fields but includes base-owned fields", () => {
  assert.deepEqual(
    activeCustomTypes(ontology).map((item) => item.name),
    ["mechanism_hypothesis"],
  );
  assert.deepEqual(
    activeFieldsForNode(ontology, "hypothesis", "mechanism_hypothesis").map((item) => item.name),
    ["prior", "mechanism"],
  );
});

test("a deprecated type is removable only after all dependent definitions are deprecated", () => {
  assert.equal(canRemoveOntologyType(ontology, "old_type"), false);
  const removable = {
    ...ontology,
    relations: ontology.relations.map((item) =>
      item.source_types.includes("old_type") || item.target_types.includes("old_type")
        ? { ...item, deprecated: true }
        : item,
    ),
  };
  assert.equal(canRemoveOntologyType(removable, "old_type"), true);

  const removed = removeOntologyType(removable, "old_type");
  assert.equal(
    removed.types.some((item) => item.name === "old_type"),
    false,
  );
  assert.equal(
    removed.relations.some((item) => item.name === "explains"),
    false,
  );
  assert.equal(
    removed.relations.some((item) => item.name === "only_old"),
    false,
  );
});

test("renaming a type updates owned fields and relation endpoints", () => {
  const renamed = upsertOntologyType(
    ontology,
    { ...ontology.types[0], name: "causal_hypothesis" },
    "mechanism_hypothesis",
  );
  assert.equal(
    renamed.fields.some((item) => item.owner_type === "causal_hypothesis"),
    true,
  );
  assert.deepEqual(renamed.relations[0].source_types, ["causal_hypothesis"]);
});

test("custom node payload is full, asserted, slugged, and preserves the base semantic type", () => {
  const node = makeCustomNode(
    ontology,
    "mechanism_hypothesis",
    "Periodic Replanning",
    "Replanning mechanism",
    "Periodic replanning preserves plasticity.",
    undefined,
    { prior: 0.4, mechanism: "Refreshes update directions." },
  );
  assert.equal(node.id, "mechanism_hypothesis/periodic-replanning");
  assert.equal(node.type, "hypothesis");
  assert.equal(node.extension_type, "mechanism_hypothesis");
  assert.equal(node.standing, "asserted");
  assert.equal(node.status, "proposed");
  assert.deepEqual(node.predictions, []);
  assert.deepEqual(node.extension_fields, {
    prior: 0.4,
    mechanism: "Refreshes update directions.",
  });
});

test("custom Evidence defaults to a result role without authoring compatibility strength", () => {
  const evidenceOntology = {
    ...ontology,
    types: [
      ...ontology.types,
      {
        name: "evaluation_result",
        definition: "A project-specific evaluation observation.",
        base_type: "evidence",
        layer: "epistemic",
        deprecated: false,
      },
    ],
  };
  const node = makeCustomNode(
    evidenceOntology,
    "evaluation_result",
    "Held-out Gain",
    "Held-out gain",
    "The held-out score increased.",
    "internal_run",
    {},
  );

  assert.equal(node.type, "evidence");
  assert.equal(node.role, "result");
  assert.equal(node.origin, "internal_run");
  assert.equal("strength" in node, false);
  assert.equal("legacy_strength" in node, false);
});
