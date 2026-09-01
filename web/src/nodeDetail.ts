import type { BeliefTransition, Edge, GraphNode, ValidationMessage } from "./types";

export function nodeBeliefTransitions(
  nodeId: string,
  transitions: BeliefTransition[],
): BeliefTransition[] {
  return transitions
    .filter((transition) => transition.hypothesis_id === nodeId)
    .sort((left, right) => right.revision - left.revision);
}

export function edgeValidationFlags(
  edgeId: string,
  messages: ValidationMessage[],
): ValidationMessage[] {
  return messages.filter(
    (message) =>
      message.code === "relation-type-mismatch" && message.related_edge_ids.includes(edgeId),
  );
}

export function beliefCausePresentation(
  transition: BeliefTransition,
  edges: Edge[],
  nodes: Record<string, GraphNode>,
): { label: string; nodeId?: string } {
  const { kind } = transition.cause;
  if (kind === "human_edit") return { label: "Human edit" };
  const refId = transition.cause.ref_id;
  if (kind === "evidence_edge") {
    const edge = edges.find((candidate) => candidate.id === refId);
    const evidenceId = edge
      ? [edge.source, edge.target].find((nodeId) => nodes[nodeId]?.type === "evidence")
      : undefined;
    return {
      label: evidenceId
        ? `Evidence: ${nodes[evidenceId]?.title ?? evidenceId}`
        : `Evidence relation: ${refId}`,
      nodeId: evidenceId,
    };
  }
  if (kind === "decision") {
    return {
      label: `Decision: ${nodes[refId]?.title ?? refId}`,
      nodeId: nodes[refId] ? refId : undefined,
    };
  }
  return { label: `Proposal resolution: ${refId}` };
}
