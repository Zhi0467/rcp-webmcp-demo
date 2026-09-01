import type { Edge, GraphNode, GraphState, TrustView } from "./types";

export type DagOntologyProjection = "all" | "belief" | "action";
export type ProjectionEmphasis = "emphasized" | "neutral" | "dimmed";

export interface RelationFocus {
  nodeIds: Set<string>;
  edgeIds: Set<string>;
}

export interface ActiveFlowProjectionOptions {
  includeResolvedBlockers?: boolean;
}

export function projectNodes(
  nodes: GraphNode[],
  trustView: TrustView,
  options: ActiveFlowProjectionOptions = {},
): GraphNode[] {
  const trusted =
    trustView === "review"
      ? nodes
      : trustView === "accepted"
        ? nodes.filter((node) => node.standing === "accepted")
        : nodes;
  if (options.includeResolvedBlockers) return trusted;
  return trusted.filter((node) => node.type !== "blocker" || node.status !== "resolved");
}

export function buildDagProjection(
  graph: GraphState,
  trustView: TrustView,
  relationFocusNodeId?: string | null,
  options: ActiveFlowProjectionOptions = {},
) {
  const nodes = relationFocusNodeId
    ? Object.values(graph.nodes)
    : projectNodes(Object.values(graph.nodes), trustView, options);
  const visible = new Set(nodes.map((node) => node.id));
  const edges = Object.values(graph.edges).filter(
    (edge) => visible.has(edge.source) && visible.has(edge.target),
  );
  return { nodes, edges };
}

export function relationFocus(nodeId: string, edges: Edge[]): RelationFocus {
  const nodeIds = new Set([nodeId]);
  const edgeIds = new Set<string>();
  edges.forEach((edge) => {
    if (edge.source !== nodeId && edge.target !== nodeId) return;
    nodeIds.add(edge.source);
    nodeIds.add(edge.target);
    edgeIds.add(edge.id);
  });
  return { nodeIds, edgeIds };
}

export function edgeProjectionEmphasis(
  edge: Edge,
  projection: DagOntologyProjection,
): ProjectionEmphasis {
  if (projection === "all") return "emphasized";
  if (edge.layer === "meta") return "neutral";
  if (edge.layer === "seam") return "emphasized";
  const selectedLayer = projection === "belief" ? "epistemic" : "action";
  return edge.layer === selectedLayer ? "emphasized" : "dimmed";
}

export function buildNodeProjectionEmphasis(
  edges: Iterable<Edge>,
  projection: DagOntologyProjection,
): ReadonlyMap<string, ProjectionEmphasis> {
  const emphasis = new Map<string, ProjectionEmphasis>();
  for (const edge of edges) {
    const edgeEmphasis = edgeProjectionEmphasis(edge, projection);
    mergeNodeEmphasis(emphasis, edge.source, edgeEmphasis);
    mergeNodeEmphasis(emphasis, edge.target, edgeEmphasis);
  }
  return emphasis;
}

function mergeNodeEmphasis(
  emphasis: Map<string, ProjectionEmphasis>,
  nodeId: string,
  incoming: ProjectionEmphasis,
): void {
  const current = emphasis.get(nodeId);
  if (current === "emphasized" || current === incoming) return;
  if (incoming === "emphasized" || !current || (current === "neutral" && incoming === "dimmed")) {
    emphasis.set(nodeId, incoming);
  }
}
