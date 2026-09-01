import type { AgentTask, GraphState } from "./types";

export function graphMutationsDisabled(graph: GraphState): boolean {
  return graph.replay_status === "degraded";
}

export function replayFailureLabel(graph: GraphState): string | null {
  if (!graphMutationsDisabled(graph)) return null;
  const failure = graph.replay_failure;
  if (!failure) return "Replay is degraded. This is the last coherent graph.";
  return `Replay stopped at revision ${failure.revision} (${failure.code}): ${failure.message} This is the last coherent graph.`;
}

export function taskMayMutateGraph(task: AgentTask): boolean {
  // Only ingestion is gated on graph health. Work's operational result is
  // independent of its optional graph reflection: a degraded graph may reject
  // the patch, but must not strand Resume or Retry for repository work that can
  // still complete safely.
  return task.kind === "seed" || task.kind === "refresh";
}
