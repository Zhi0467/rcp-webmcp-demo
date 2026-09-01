import assert from "node:assert/strict";
import test from "node:test";

import { buildResearchPaths } from "../src/researchProjection.ts";

const node = (id, type) => ({ id, type, title: id, standing: "asserted", source_refs: [] });

test("research paths ignore edge direction and assign each record once", () => {
  const nodes = [
    node("question/a", "research_question"),
    node("question/b", "research_question"),
    node("idea", "hypothesis"),
    node("experiment", "experiment"),
    node("evidence", "evidence"),
  ];
  const edges = [
    { id: "1", source: "idea", target: "question/a" },
    { id: "2", source: "experiment", target: "idea" },
    { id: "3", source: "evidence", target: "experiment" },
    { id: "4", source: "question/b", target: "evidence" },
  ];

  const forward = buildResearchPaths(nodes, edges);
  const reversed = buildResearchPaths(
    nodes,
    edges.map((edge) => ({ ...edge, source: edge.target, target: edge.source })),
  );
  const assigned = forward.paths.flatMap((path) => [
    ...path.ideas,
    ...path.experiments,
    ...path.evidence,
  ]);

  assert.deepEqual(forward, reversed);
  assert.equal(new Set(assigned.map((item) => item.id)).size, assigned.length);
  assert.deepEqual(
    forward.paths.find((path) => path.question.id === "question/a")?.ideas.map((item) => item.id),
    ["idea"],
  );
  assert.deepEqual(
    forward.paths
      .find((path) => path.question.id === "question/b")
      ?.evidence.map((item) => item.id),
    ["evidence"],
  );
});

test("unconnected records stay explicit and implementation blockers are excluded", () => {
  const projection = buildResearchPaths(
    [
      node("question", "research_question"),
      node("orphan", "decision"),
      { ...node("implementation", "blocker"), blocker_type: "implementation" },
    ],
    [],
  );

  assert.deepEqual(
    projection.paths.map((path) => path.question.id),
    ["question"],
  );
  assert.deepEqual(
    projection.unconnected.map((item) => item.id),
    ["orphan"],
  );
});
