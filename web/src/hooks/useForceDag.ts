import {
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Edge, GraphNode } from "../types";
import {
  RESEARCH_STAGE_BY_NODE_TYPE,
  buildSemanticLaneLayout,
  resolveRectangleCollisions,
  type SemanticLaneLayout,
} from "./dagLayout";

export const DAG_NODE_WIDTH = 236;
export const DAG_NODE_HEIGHT = 96;

export interface DagPosition {
  x: number;
  y: number;
  pinned: boolean;
}

export type DagLayoutMode = "force" | "flow";

interface ForceNode extends SimulationNodeDatum {
  id: string;
  lane: number;
  node: GraphNode;
}

interface ForceEdge extends SimulationLinkDatum<ForceNode> {
  id: string;
  relation: string;
}

interface StoredPosition {
  x: number;
  y: number;
}

interface ForceDagOptions {
  nodes: GraphNode[];
  edges: Edge[];
  projectId: string;
  repulsion: number;
  mode: DagLayoutMode;
}

const HORIZONTAL_PADDING = 54;
const VERTICAL_PADDING = 56;
const FLOW_ROW_GAP = 44;
const FLOW_COLUMN_GAP = 120;
const FORCE_CANVAS_MIN_WIDTH = 2400;
const FORCE_CANVAS_MIN_HEIGHT = 1300;
const FORCE_LANE_GUTTER = 360;
const REPULSION_MIN = 350;
const REPULSION_MAX = 1900;

export function forceTuning(repulsion: number) {
  const spread = clamp((repulsion - REPULSION_MIN) / (REPULSION_MAX - REPULSION_MIN), 0, 1);
  return {
    chargeStrength: -lerp(160, 2100, spread),
    chargeDistanceMax: lerp(620, 1700, spread),
    linkDistance: lerp(170, 470, spread),
    linkStrength: lerp(0.26, 0.06, spread),
    laneStrength: lerp(0.22, 0.045, spread),
    centerlineStrength: lerp(0.045, 0.007, spread),
    collisionPadding: lerp(12, 76, spread),
  };
}

export function useForceDag({ nodes, edges, projectId, repulsion, mode }: ForceDagOptions) {
  const flowLayout = useMemo(() => buildSemanticLaneLayout(nodes, edges), [edges, nodes]);
  const metrics = useMemo(() => canvasMetrics(nodes, mode, flowLayout), [flowLayout, mode, nodes]);
  const storageKey =
    mode === "force" ? `rcp:dag-layout:v2:${projectId}` : `rcp:dag-layout:flow:v2:${projectId}`;
  const simulationRef = useRef<Simulation<ForceNode, ForceEdge> | null>(null);
  const nodesRef = useRef<ForceNode[]>([]);
  const activeModeRef = useRef<DagLayoutMode | null>(null);
  const [positions, setPositions] = useState<Record<string, DagPosition>>({});
  const [pinCount, setPinCount] = useState(0);
  const [resetGeneration, setResetGeneration] = useState(0);

  const publish = useCallback(() => {
    const next: Record<string, DagPosition> = {};
    nodesRef.current.forEach((node) => {
      next[node.id] = constrainNodeToCanvas(node, metrics.width, metrics.height);
    });
    setPositions(next);
  }, [metrics.height, metrics.width]);

  useEffect(() => {
    const previous =
      activeModeRef.current === mode
        ? new Map(nodesRef.current.map((node) => [node.id, node]))
        : new Map<string, ForceNode>();
    activeModeRef.current = mode;
    const stored = readStoredPositions(storageKey);
    const initial =
      mode === "flow"
        ? flowPositions(flowLayout, metrics.width, metrics.height)
        : initialPositions(nodes, metrics.width, metrics.height);
    const forceNodes: ForceNode[] = nodes.map((node) => {
      const prior = previous.get(node.id);
      const saved = stored[node.id];
      const fallback = initial[node.id];
      const priorPinned = prior?.fx !== null && prior?.fx !== undefined;
      const adjustment = mode === "flow" && !priorPinned ? saved : prior;
      const x = adjustment?.x ?? saved?.x ?? fallback.x;
      const y = adjustment?.y ?? saved?.y ?? fallback.y;
      const pinned = Boolean(priorPinned) || Boolean(saved);
      return {
        id: node.id,
        lane: RESEARCH_STAGE_BY_NODE_TYPE[node.type],
        node,
        x,
        y,
        fx: pinned ? x : null,
        fy: pinned ? y : null,
      };
    });
    const forceEdges: ForceEdge[] = edges.map((edge) => ({
      id: edge.id,
      relation: edge.relation,
      source: edge.source,
      target: edge.target,
    }));

    nodesRef.current = forceNodes;
    setPinCount(forceNodes.filter((node) => node.fx !== null && node.fx !== undefined).length);

    if (mode === "flow") {
      simulationRef.current = null;
      publish();
      return;
    }

    const tuning = forceTuning(repulsion);
    const simulation = forceSimulation<ForceNode>(forceNodes)
      .force(
        "links",
        forceLink<ForceNode, ForceEdge>(forceEdges)
          .id((node) => node.id)
          .distance(tuning.linkDistance)
          .strength(tuning.linkStrength),
      )
      .force(
        "repulsion",
        forceManyBody<ForceNode>()
          .strength(tuning.chargeStrength)
          .distanceMin(90)
          .distanceMax(tuning.chargeDistanceMax),
      )
      .force(
        "lanes",
        forceX<ForceNode>((node) => forceLaneX(node.lane, metrics.width)).strength(
          tuning.laneStrength,
        ),
      )
      .force(
        "centerline",
        forceY<ForceNode>(metrics.height / 2).strength(tuning.centerlineStrength),
      )
      .force("collision", forceRectangleCollide(tuning.collisionPadding, 0.94))
      .alpha(0.9)
      .alphaDecay(0.035)
      .velocityDecay(0.34);

    simulationRef.current = simulation;
    let animationFrame = 0;
    const schedulePublish = () => {
      if (animationFrame) return;
      animationFrame = window.requestAnimationFrame(() => {
        animationFrame = 0;
        publish();
      });
    };

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      simulation.stop();
      for (let tick = 0; tick < 180; tick += 1) {
        simulation.tick();
        forceNodes.forEach((node) => constrainNodeToCanvas(node, metrics.width, metrics.height));
      }
      publish();
    } else {
      simulation.on("tick", schedulePublish);
      publish();
    }

    return () => {
      simulation.stop();
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      if (simulationRef.current === simulation) simulationRef.current = null;
    };
  }, [
    edges,
    flowLayout,
    metrics.height,
    metrics.width,
    mode,
    nodes,
    publish,
    repulsion,
    resetGeneration,
    storageKey,
  ]);

  const beginDrag = useCallback((nodeId: string) => {
    const node = nodesRef.current.find((item) => item.id === nodeId);
    if (!node) return;
    node.fx = node.x;
    node.fy = node.y;
    simulationRef.current?.alphaTarget(0.16).restart();
  }, []);

  const moveDrag = useCallback(
    (nodeId: string, x: number, y: number) => {
      const node = nodesRef.current.find((item) => item.id === nodeId);
      if (!node) return;
      node.fx = clamp(
        x,
        HORIZONTAL_PADDING + DAG_NODE_WIDTH / 2,
        metrics.width - HORIZONTAL_PADDING - DAG_NODE_WIDTH / 2,
      );
      node.fy = clamp(
        y,
        VERTICAL_PADDING + DAG_NODE_HEIGHT / 2,
        metrics.height - VERTICAL_PADDING - DAG_NODE_HEIGHT / 2,
      );
      node.x = node.fx;
      node.y = node.fy;
      publish();
    },
    [metrics.height, metrics.width, publish],
  );

  const endDrag = useCallback(() => {
    simulationRef.current?.alphaTarget(0);
    persistPinnedPositions(storageKey, nodesRef.current);
    setPinCount(
      nodesRef.current.filter((node) => node.fx !== null && node.fx !== undefined).length,
    );
    publish();
  }, [publish, storageKey]);

  const releasePins = useCallback(() => {
    removeStoredPositions(storageKey);
    if (mode === "flow") {
      nodesRef.current = [];
      setPinCount(0);
      setPositions({});
      setResetGeneration((generation) => generation + 1);
      return;
    }
    nodesRef.current.forEach((node) => {
      node.fx = null;
      node.fy = null;
    });
    setPinCount(0);
    simulationRef.current?.alpha(0.72).alphaTarget(0).restart();
    publish();
  }, [mode, publish, storageKey]);

  const releasePin = useCallback(
    (nodeId: string) => {
      const node = nodesRef.current.find((item) => item.id === nodeId);
      if (!node) return;
      node.fx = null;
      node.fy = null;

      if (mode === "flow") {
        const flowPosition = flowPositions(flowLayout, metrics.width, metrics.height)[nodeId];
        if (flowPosition) {
          node.x = flowPosition.x;
          node.y = flowPosition.y;
          node.vx = 0;
          node.vy = 0;
        }
      } else {
        simulationRef.current?.alpha(0.72).alphaTarget(0).restart();
      }

      persistPinnedPositions(storageKey, nodesRef.current);
      setPinCount(
        nodesRef.current.filter((item) => item.fx !== null && item.fx !== undefined).length,
      );
      publish();
    },
    [flowLayout, metrics.height, metrics.width, mode, publish, storageKey],
  );

  const resetLayout = useCallback(() => {
    removeStoredPositions(storageKey);
    simulationRef.current?.stop();
    nodesRef.current = [];
    setPinCount(0);
    setPositions({});
    setResetGeneration((generation) => generation + 1);
  }, [storageKey]);

  return {
    ...metrics,
    positions,
    pinCount,
    beginDrag,
    moveDrag,
    endDrag,
    releasePin,
    releasePins,
    resetLayout,
  };
}

function canvasMetrics(nodes: GraphNode[], mode: DagLayoutMode, flowLayout: SemanticLaneLayout) {
  if (mode === "force") return forceCanvasMetrics(nodes);
  const largestLane = Math.max(1, ...flowLayout.lanes.map((lane) => lane.length));
  const flowHeight =
    VERTICAL_PADDING * 2 + largestLane * DAG_NODE_HEIGHT + (largestLane - 1) * FLOW_ROW_GAP;
  const flowWidth =
    HORIZONTAL_PADDING * 2 +
    DAG_NODE_WIDTH +
    Math.max(0, flowLayout.lanes.length - 1) * (DAG_NODE_WIDTH + FLOW_COLUMN_GAP);
  return {
    width: Math.max(1220, flowWidth),
    height: Math.max(760, flowHeight),
  };
}

export function forceCanvasMetrics(nodes: Pick<GraphNode, "type">[]) {
  const laneSizes = new Map<number, number>();
  nodes.forEach((node) => {
    const lane = RESEARCH_STAGE_BY_NODE_TYPE[node.type];
    laneSizes.set(lane, (laneSizes.get(lane) ?? 0) + 1);
  });
  const largestLane = Math.max(1, ...laneSizes.values());
  return {
    width: FORCE_CANVAS_MIN_WIDTH,
    height: Math.max(
      FORCE_CANVAS_MIN_HEIGHT,
      300 + Math.ceil(nodes.length / 4) * 150,
      300 + largestLane * 150,
    ),
  };
}

function initialPositions(
  nodes: GraphNode[],
  width: number,
  height: number,
): Record<string, StoredPosition> {
  const lanes = new Map<number, GraphNode[]>();
  nodes.forEach((node) => {
    const lane = RESEARCH_STAGE_BY_NODE_TYPE[node.type];
    lanes.set(lane, [...(lanes.get(lane) ?? []), node]);
  });
  lanes.forEach((items) => items.sort((left, right) => left.id.localeCompare(right.id)));

  const positions: Record<string, StoredPosition> = {};
  lanes.forEach((items, lane) => {
    items.forEach((node, index) => {
      const availableHeight = height - (VERTICAL_PADDING * 2 + DAG_NODE_HEIGHT);
      positions[node.id] = {
        x: forceLaneX(lane, width),
        y:
          VERTICAL_PADDING +
          DAG_NODE_HEIGHT / 2 +
          ((index + 1) * availableHeight) / (items.length + 1),
      };
    });
  });
  return positions;
}

function flowPositions(
  flowLayout: SemanticLaneLayout,
  width: number,
  height: number,
): Record<string, StoredPosition> {
  const positions: Record<string, StoredPosition> = {};
  flowLayout.lanes.forEach((nodeIds, lane) => {
    const occupiedHeight =
      nodeIds.length * DAG_NODE_HEIGHT + Math.max(0, nodeIds.length - 1) * FLOW_ROW_GAP;
    const firstY = Math.max(VERTICAL_PADDING, (height - occupiedHeight) / 2) + DAG_NODE_HEIGHT / 2;
    nodeIds.forEach((nodeId, index) => {
      positions[nodeId] = {
        x: rankX(lane, flowLayout.lanes.length, width),
        y: firstY + index * (DAG_NODE_HEIGHT + FLOW_ROW_GAP),
      };
    });
  });
  return positions;
}

function rankX(rank: number, rankCount: number, width: number): number {
  if (rankCount <= 1) return width / 2;
  const left = HORIZONTAL_PADDING + DAG_NODE_WIDTH / 2;
  const right = width - HORIZONTAL_PADDING - DAG_NODE_WIDTH / 2;
  return left + (rank * (right - left)) / (rankCount - 1);
}

export function forceLaneX(lane: number, width: number): number {
  const left = HORIZONTAL_PADDING + DAG_NODE_WIDTH / 2 + FORCE_LANE_GUTTER;
  const right = width - HORIZONTAL_PADDING - DAG_NODE_WIDTH / 2 - FORCE_LANE_GUTTER;
  return left + (lane * (right - left)) / 3;
}

function readStoredPositions(storageKey: string): Record<string, StoredPosition> {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(storageKey) ?? "{}") as Record<
      string,
      StoredPosition
    >;
    return Object.fromEntries(
      Object.entries(parsed).filter(
        ([, position]) => Number.isFinite(position?.x) && Number.isFinite(position?.y),
      ),
    );
  } catch {
    return {};
  }
}

function persistPinnedPositions(storageKey: string, nodes: ForceNode[]): void {
  const pinned = Object.fromEntries(
    nodes
      .filter(
        (node) =>
          node.fx !== null && node.fx !== undefined && node.fy !== null && node.fy !== undefined,
      )
      .map((node) => [node.id, { x: Number(node.fx), y: Number(node.fy) }]),
  );
  try {
    window.localStorage.setItem(storageKey, JSON.stringify(pinned));
  } catch {
    // Layout persistence is an enhancement; interaction remains available without storage.
  }
}

function removeStoredPositions(storageKey: string): void {
  try {
    window.localStorage.removeItem(storageKey);
  } catch {
    // Ignore unavailable storage and still reset the in-memory layout.
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function lerp(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}

function constrainNodeToCanvas(node: ForceNode, width: number, height: number): DagPosition {
  const pinned = node.fx !== null && node.fx !== undefined;
  const rawX = node.fx ?? node.x ?? width / 2;
  const rawY = node.fy ?? node.y ?? height / 2;
  const x = clamp(
    rawX,
    HORIZONTAL_PADDING + DAG_NODE_WIDTH / 2,
    width - HORIZONTAL_PADDING - DAG_NODE_WIDTH / 2,
  );
  const y = clamp(
    rawY,
    VERTICAL_PADDING + DAG_NODE_HEIGHT / 2,
    height - VERTICAL_PADDING - DAG_NODE_HEIGHT / 2,
  );
  if (x !== rawX) node.vx = 0;
  if (y !== rawY) node.vy = 0;
  node.x = x;
  node.y = y;
  if (pinned) {
    node.fx = x;
    node.fy = y;
  }
  return { x, y, pinned };
}

function forceRectangleCollide(padding: number, strength: number) {
  let nodes: ForceNode[] = [];
  const force = () => {
    resolveRectangleCollisions(
      nodes,
      DAG_NODE_WIDTH + padding,
      DAG_NODE_HEIGHT + padding,
      strength,
      4,
    );
  };
  force.initialize = (forceNodes: ForceNode[]) => {
    nodes = forceNodes;
  };
  return force;
}
