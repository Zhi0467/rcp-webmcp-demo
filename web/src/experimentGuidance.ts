export type ExperimentGuidanceField = "current_summary" | "next_action";

export interface ExperimentGuidanceNode {
  current_summary?: unknown;
  next_action?: unknown;
  current_summary_stale?: unknown;
  next_action_stale?: unknown;
}

export interface ExperimentGuidanceDetail {
  field: ExperimentGuidanceField;
  status: "empty" | "current" | "stale";
  text: string | null;
  label: string;
}

const guidanceLabels: Record<ExperimentGuidanceField, string> = {
  current_summary: "Research summary",
  next_action: "Next action",
};

export function experimentGuidanceDetail(
  node: ExperimentGuidanceNode,
  field: ExperimentGuidanceField,
): ExperimentGuidanceDetail {
  const text = authoredText(node[field]);
  if (!text) return { field, status: "empty", text: null, label: guidanceLabels[field] };
  const staleField = field === "current_summary" ? "current_summary_stale" : "next_action_stale";
  const stale = node[staleField] === true;
  return {
    field,
    status: stale ? "stale" : "current",
    text,
    label: stale
      ? `Previous ${guidanceLabels[field].toLowerCase()} (stale)`
      : guidanceLabels[field],
  };
}

/** Returns only guidance that the current gate state still authorizes the UI to present. */
export function currentExperimentGuidance(
  node: ExperimentGuidanceNode,
  field: ExperimentGuidanceField,
): string | null {
  const detail = experimentGuidanceDetail(node, field);
  return detail.status === "current" ? detail.text : null;
}

/** Board/list copy prefers a current next action, then a current summary. */
export function activeExperimentGuidanceText(node: ExperimentGuidanceNode): string | null {
  return (
    currentExperimentGuidance(node, "next_action") ??
    currentExperimentGuidance(node, "current_summary")
  );
}

function authoredText(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}
