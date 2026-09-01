export interface TopologyNode {
  id: string;
}

export interface TopologyEdge {
  source: string;
  target: string;
  relation?: string;
}

export interface TopologyLayout {
  layers: string[][];
  rankById: Record<string, number>;
}

export const RESEARCH_STAGE_BY_NODE_TYPE = {
  research_question: 0,
  hypothesis: 1,
  decision: 1,
  experiment: 2,
  blocker: 2,
  evidence: 3,
} as const;

export interface SemanticLaneNode extends TopologyNode {
  type: keyof typeof RESEARCH_STAGE_BY_NODE_TYPE;
}

export interface SemanticLaneLayout {
  lanes: string[][];
}

export interface RectangleCollisionNode {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

const BARYCENTER_SWEEPS = 4;

/**
 * Produces stable left-to-right ranks from directed topology. Cycles are
 * condensed into one strongly connected component before longest-path ranks
 * are assigned, so every edge outside a cycle points to a later rank.
 */
export function buildTopologyLayout(nodes: TopologyNode[], edges: TopologyEdge[]): TopologyLayout {
  const nodeIds = [...new Set(nodes.map((node) => node.id))].sort(compareIds);
  const visible = new Set(nodeIds);
  const outgoing = new Map(nodeIds.map((id) => [id, [] as string[]]));
  const incoming = new Map(nodeIds.map((id) => [id, [] as string[]]));

  edges.forEach((edge) => {
    if (!visible.has(edge.source) || !visible.has(edge.target)) return;
    outgoing.get(edge.source)?.push(edge.target);
    incoming.get(edge.target)?.push(edge.source);
  });
  outgoing.forEach(sortAndDeduplicate);
  incoming.forEach(sortAndDeduplicate);

  const components = stronglyConnectedComponents(nodeIds, outgoing);
  const componentById = new Map<string, number>();
  components.forEach((members, component) =>
    members.forEach((id) => componentById.set(id, component)),
  );

  const componentChildren = components.map(() => new Set<number>());
  const componentParents = components.map(() => new Set<number>());
  nodeIds.forEach((source) => {
    const sourceComponent = componentById.get(source);
    if (sourceComponent === undefined) return;
    outgoing.get(source)?.forEach((target) => {
      const targetComponent = componentById.get(target);
      if (targetComponent === undefined || targetComponent === sourceComponent) return;
      componentChildren[sourceComponent].add(targetComponent);
      componentParents[targetComponent].add(sourceComponent);
    });
  });

  const componentRanks = longestPathRanks(components, componentChildren, componentParents);
  const rankById: Record<string, number> = {};
  nodeIds.forEach((id) => {
    rankById[id] = componentRanks[componentById.get(id) ?? 0] ?? 0;
  });

  const rankCount = nodeIds.length === 0 ? 0 : Math.max(...Object.values(rankById)) + 1;
  const layers = Array.from({ length: rankCount }, () => [] as string[]);
  nodeIds.forEach((id) => layers[rankById[id]].push(id));
  orderLayersByBarycenter(layers, rankById, incoming, outgoing);

  return { layers, rankById };
}

/** Places question hierarchy levels before the fixed later research stages. */
export function buildSemanticLaneLayout(
  nodes: SemanticLaneNode[],
  edges: TopologyEdge[],
): SemanticLaneLayout {
  const topology = buildTopologyLayout(nodes, edges);
  const topologyOrder = new Map<string, number>();
  let nextOrder = 0;
  topology.layers.forEach((layer) => {
    layer.forEach((id) => {
      topologyOrder.set(id, nextOrder);
      nextOrder += 1;
    });
  });

  const questionNodes = nodes.filter((node) => node.type === "research_question");
  const questionIds = new Set(questionNodes.map((node) => node.id));
  const questionLayout = buildTopologyLayout(
    questionNodes,
    edges.filter(
      (edge) =>
        edge.relation === "has_subquestion" &&
        questionIds.has(edge.source) &&
        questionIds.has(edge.target),
    ),
  );
  const deepestQuestionRank = Math.max(0, ...Object.values(questionLayout.rankById));
  const lanes = Array.from({ length: deepestQuestionRank + 4 }, () => [] as string[]);
  nodes.forEach((node) => {
    const lane =
      node.type === "research_question"
        ? (questionLayout.rankById[node.id] ?? 0)
        : RESEARCH_STAGE_BY_NODE_TYPE[node.type] + deepestQuestionRank;
    lanes[lane].push(node.id);
  });
  lanes.forEach((lane) =>
    lane.sort(
      (left, right) =>
        (topologyOrder.get(left) ?? Number.MAX_SAFE_INTEGER) -
          (topologyOrder.get(right) ?? Number.MAX_SAFE_INTEGER) || compareIds(left, right),
    ),
  );
  return { lanes };
}

/** Returns the spatial-hash candidates that can contain overlapping centers. */
export function rectangleCollisionCandidates(
  nodes: RectangleCollisionNode[],
  collisionWidth: number,
  collisionHeight: number,
): Array<[number, number]> {
  const projected = nodes.map((node) => ({
    x: (node.x ?? 0) + (node.vx ?? 0),
    y: (node.y ?? 0) + (node.vy ?? 0),
  }));
  const buckets = new Map<string, number[]>();
  projected.forEach((point, index) => {
    const cellX = Math.floor(point.x / collisionWidth);
    const cellY = Math.floor(point.y / collisionHeight);
    const key = cellKey(cellX, cellY);
    const bucket = buckets.get(key);
    if (bucket) bucket.push(index);
    else buckets.set(key, [index]);
  });

  const candidates: Array<[number, number]> = [];
  projected.forEach((point, leftIndex) => {
    const centerCellX = Math.floor(point.x / collisionWidth);
    const centerCellY = Math.floor(point.y / collisionHeight);
    for (let cellY = centerCellY - 1; cellY <= centerCellY + 1; cellY += 1) {
      for (let cellX = centerCellX - 1; cellX <= centerCellX + 1; cellX += 1) {
        buckets.get(cellKey(cellX, cellY))?.forEach((rightIndex) => {
          if (rightIndex > leftIndex) candidates.push([leftIndex, rightIndex]);
        });
      }
    }
  });
  return candidates;
}

export function resolveRectangleCollisions(
  nodes: RectangleCollisionNode[],
  collisionWidth: number,
  collisionHeight: number,
  strength: number,
  iterations: number,
): void {
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    rectangleCollisionCandidates(nodes, collisionWidth, collisionHeight).forEach(
      ([leftIndex, rightIndex]) => {
        const left = nodes[leftIndex];
        const right = nodes[rightIndex];
        const dx = (right.x ?? 0) + (right.vx ?? 0) - ((left.x ?? 0) + (left.vx ?? 0));
        const dy = (right.y ?? 0) + (right.vy ?? 0) - ((left.y ?? 0) + (left.vy ?? 0));
        const overlapX = collisionWidth - Math.abs(dx);
        const overlapY = collisionHeight - Math.abs(dy);
        if (overlapX <= 0 || overlapY <= 0) return;

        const leftPinned = left.fx !== null && left.fx !== undefined;
        const rightPinned = right.fx !== null && right.fx !== undefined;
        if (leftPinned && rightPinned) return;
        const split = leftPinned || rightPinned ? 1 : 0.5;

        if (overlapX < overlapY) {
          const direction = dx === 0 ? (leftIndex % 2 === 0 ? 1 : -1) : Math.sign(dx);
          const displacement = overlapX * strength * split;
          if (!leftPinned) left.vx = (left.vx ?? 0) - direction * displacement;
          if (!rightPinned) right.vx = (right.vx ?? 0) + direction * displacement;
        } else {
          const direction = dy === 0 ? (leftIndex % 2 === 0 ? 1 : -1) : Math.sign(dy);
          const displacement = overlapY * strength * split;
          if (!leftPinned) left.vy = (left.vy ?? 0) - direction * displacement;
          if (!rightPinned) right.vy = (right.vy ?? 0) + direction * displacement;
        }
      },
    );
  }
}

function stronglyConnectedComponents(
  nodeIds: string[],
  outgoing: Map<string, string[]>,
): string[][] {
  let nextIndex = 0;
  const indices = new Map<string, number>();
  const lowLinks = new Map<string, number>();
  const stack: string[] = [];
  const onStack = new Set<string>();
  const components: string[][] = [];

  const visit = (id: string) => {
    const index = nextIndex;
    nextIndex += 1;
    indices.set(id, index);
    lowLinks.set(id, index);
    stack.push(id);
    onStack.add(id);

    outgoing.get(id)?.forEach((target) => {
      if (!indices.has(target)) {
        visit(target);
        lowLinks.set(id, Math.min(lowLinks.get(id) ?? index, lowLinks.get(target) ?? index));
      } else if (onStack.has(target)) {
        lowLinks.set(id, Math.min(lowLinks.get(id) ?? index, indices.get(target) ?? index));
      }
    });

    if (lowLinks.get(id) !== indices.get(id)) return;
    const component: string[] = [];
    while (stack.length > 0) {
      const member = stack.pop();
      if (member === undefined) break;
      onStack.delete(member);
      component.push(member);
      if (member === id) break;
    }
    components.push(component.sort(compareIds));
  };

  nodeIds.forEach((id) => {
    if (!indices.has(id)) visit(id);
  });
  components.sort((left, right) => compareIds(left[0], right[0]));
  return components;
}

function longestPathRanks(
  components: string[][],
  children: Set<number>[],
  parents: Set<number>[],
): number[] {
  const ranks = components.map(() => 0);
  const indegrees = parents.map((items) => items.size);
  const ready = components
    .map((_, component) => component)
    .filter((component) => indegrees[component] === 0)
    .sort((left, right) => compareIds(components[left][0], components[right][0]));

  while (ready.length > 0) {
    const component = ready.shift();
    if (component === undefined) break;
    [...children[component]]
      .sort((left, right) => compareIds(components[left][0], components[right][0]))
      .forEach((child) => {
        ranks[child] = Math.max(ranks[child], ranks[component] + 1);
        indegrees[child] -= 1;
        if (indegrees[child] === 0) {
          ready.push(child);
          ready.sort((left, right) => compareIds(components[left][0], components[right][0]));
        }
      });
  }
  return ranks;
}

function orderLayersByBarycenter(
  layers: string[][],
  rankById: Record<string, number>,
  incoming: Map<string, string[]>,
  outgoing: Map<string, string[]>,
): void {
  const verticalPosition = new Map<string, number>();
  layers.forEach((layer) => updateVerticalPositions(layer, verticalPosition));

  for (let sweep = 0; sweep < BARYCENTER_SWEEPS; sweep += 1) {
    for (let rank = 1; rank < layers.length; rank += 1) {
      reorderLayer(layers[rank], incoming, rankById, verticalPosition, rank);
    }
    for (let rank = layers.length - 2; rank >= 0; rank -= 1) {
      reorderLayer(layers[rank], outgoing, rankById, verticalPosition, rank);
    }
  }
}

function reorderLayer(
  layer: string[],
  neighborsById: Map<string, string[]>,
  rankById: Record<string, number>,
  verticalPosition: Map<string, number>,
  rank: number,
): void {
  const previousIndex = new Map(layer.map((id, index) => [id, index]));
  const scored = layer.map((id) => {
    const neighbors = (neighborsById.get(id) ?? []).filter(
      (neighbor) => rankById[neighbor] !== rank,
    );
    const fallback = verticalPosition.get(id) ?? 0.5;
    const score =
      neighbors.length === 0
        ? fallback
        : neighbors.reduce(
            (sum, neighbor) => sum + (verticalPosition.get(neighbor) ?? fallback),
            0,
          ) / neighbors.length;
    return { id, score };
  });
  scored.sort(
    (left, right) =>
      left.score - right.score ||
      (previousIndex.get(left.id) ?? 0) - (previousIndex.get(right.id) ?? 0) ||
      compareIds(left.id, right.id),
  );
  scored.forEach((item, index) => {
    layer[index] = item.id;
  });
  updateVerticalPositions(layer, verticalPosition);
}

function updateVerticalPositions(layer: string[], positions: Map<string, number>): void {
  layer.forEach((id, index) => positions.set(id, (index + 0.5) / Math.max(1, layer.length)));
}

function sortAndDeduplicate(items: string[]): void {
  items.sort(compareIds);
  let writeIndex = 0;
  items.forEach((item, index) => {
    if (index === 0 || item !== items[index - 1]) {
      items[writeIndex] = item;
      writeIndex += 1;
    }
  });
  items.length = writeIndex;
}

function cellKey(x: number, y: number): string {
  return `${x}:${y}`;
}

function compareIds(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
