import type { Edge, GraphNode } from "./types";

const researchTypes = new Set<GraphNode["type"]>([
  "research_question",
  "hypothesis",
  "decision",
  "experiment",
  "evidence",
]);

export interface ResearchPath {
  question: GraphNode;
  ideas: GraphNode[];
  experiments: GraphNode[];
  evidence: GraphNode[];
}

export interface ResearchPathProjection {
  paths: ResearchPath[];
  unconnected: GraphNode[];
}

export function buildResearchPaths(nodes: GraphNode[], edges: Edge[]): ResearchPathProjection {
  const eligible = [...nodes].filter((node) => researchTypes.has(node.type));
  const byId = new Map(eligible.map((node) => [node.id, node]));
  const adjacency = new Map(eligible.map((node) => [node.id, new Set<string>()]));
  edges.forEach((edge) => {
    if (!byId.has(edge.source) || !byId.has(edge.target)) return;
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  });

  const questions = eligible.filter((node) => node.type === "research_question").sort(compareNodes);
  const distances = new Map(
    questions.map((question) => [question.id, distancesFrom(question.id, adjacency)]),
  );
  const assignments = new Map<string, string>();

  eligible.forEach((node) => {
    if (node.type === "research_question") return;
    let closestQuestion: string | null = null;
    let closestDistance = Number.POSITIVE_INFINITY;
    questions.forEach((question) => {
      const distance = distances.get(question.id)?.get(node.id);
      if (distance === undefined) return;
      if (
        distance < closestDistance ||
        (distance === closestDistance && question.id < String(closestQuestion))
      ) {
        closestQuestion = question.id;
        closestDistance = distance;
      }
    });
    if (closestQuestion) assignments.set(node.id, closestQuestion);
  });

  const paths = questions.map((question) => {
    const assigned = eligible.filter((node) => assignments.get(node.id) === question.id);
    return {
      question,
      ideas: assigned
        .filter((node) => node.type === "hypothesis" || node.type === "decision")
        .sort(compareNodes),
      experiments: assigned.filter((node) => node.type === "experiment").sort(compareNodes),
      evidence: assigned.filter((node) => node.type === "evidence").sort(compareNodes),
    };
  });
  const unconnected = eligible
    .filter((node) => node.type !== "research_question" && !assignments.has(node.id))
    .sort(compareNodes);
  return { paths, unconnected };
}

function distancesFrom(start: string, adjacency: Map<string, Set<string>>): Map<string, number> {
  const distances = new Map([[start, 0]]);
  const queue = [start];
  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index];
    const nextDistance = (distances.get(current) ?? 0) + 1;
    [...(adjacency.get(current) ?? [])].sort().forEach((neighbor) => {
      if (distances.has(neighbor)) return;
      distances.set(neighbor, nextDistance);
      queue.push(neighbor);
    });
  }
  return distances;
}

function compareNodes(left: GraphNode, right: GraphNode): number {
  return left.id.localeCompare(right.id);
}
