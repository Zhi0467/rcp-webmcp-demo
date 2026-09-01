import type { GraphNode } from "./types";

const primaryKey: Record<GraphNode["type"], string> = {
  research_question: "question",
  hypothesis: "statement",
  decision: "question",
  experiment: "objective",
  evidence: "observation",
  blocker: "description",
};

export const humanFieldLabels: Record<string, string> = {
  question: "Question",
  statement: "Claim",
  objective: "Objective",
  observation: "What was observed",
  description: "What is blocked",
  motivation: "Why this matters",
  rationale: "Reasoning",
  interpretation: "What it means",
  scope: "Scope",
  design: "How it will be tested",
  current_summary: "Where things stand",
  next_action: "Next action",
  predictions: "What should happen if this is right",
  expected_outcomes: "Expected outcomes",
  interpretation_rules: "How results will be read",
  completion_criteria: "What counts as complete",
  options: "Options considered",
  selected_option: "Selected option",
  consequences: "What changes because of this",
  role: "Evidence role",
  legacy_strength: "Legacy strength (historical)",
  validity: "Validity",
  origin: "Origin",
  status: "Status",
  blocker_type: "Blocker type",
  resolution_condition: "What would unblock this",
  owner: "Owner",
  artifact_refs: "Artifacts",
};

const contextOrder = [
  "motivation",
  "rationale",
  "interpretation",
  "role",
  "legacy_strength",
  "scope",
  "design",
  "current_summary",
  "next_action",
  "predictions",
  "expected_outcomes",
  "interpretation_rules",
  "completion_criteria",
  "options",
  "selected_option",
  "consequences",
  "resolution_condition",
];

export function presentNode(node: GraphNode) {
  const key = primaryKey[node.type];
  const value = readableValue(node[key]) ? node[key] : node.title;
  const context = contextOrder.flatMap((field) => {
    if (node.type === "decision" && (field === "options" || field === "selected_option")) {
      return [];
    }
    return readableValue(node[field])
      ? [{ key: field, label: humanFieldLabels[field] ?? humanize(field), value: node[field] }]
      : [];
  });
  return { key, label: humanFieldLabels[key], value, context };
}

export function nodeTypeLabel(node: GraphNode): string {
  return humanize(node.extension_type ?? node.type);
}

export function humanize(key: string): string {
  return key.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

function readableValue(value: unknown): boolean {
  return (
    value !== null &&
    value !== undefined &&
    value !== "" &&
    (!Array.isArray(value) || value.length > 0)
  );
}
