import {
  ChevronDown,
  CircleDot,
  Eye,
  EyeOff,
  FlaskConical,
  Focus,
  Gauge,
  GitBranch,
  Maximize2,
  Minimize2,
  Orbit,
  Pin,
  PinOff,
  RotateCcw,
  Search,
  Workflow,
  X,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MutableRefObject,
  type PointerEvent as ReactPointerEvent,
  type Ref,
  type ReactNode,
} from "react";
import {
  buildNodeProjectionEmphasis,
  buildDagProjection,
  edgeProjectionEmphasis,
  projectNodes,
  relationFocus,
  type DagOntologyProjection,
} from "../graphProjection";
import { zoomDagAtPoint, type DagViewport, type DagZoomResult } from "../hooks/dagZoom";
import {
  DAG_NODE_HEIGHT,
  DAG_NODE_WIDTH,
  useForceDag,
  type DagLayoutMode,
  type DagPosition,
} from "../hooks/useForceDag";
import { buildResearchPaths } from "../researchProjection";
import { buildExperimentRun, type ExperimentRun } from "../runProjection";
import {
  ExperimentRunDetail,
  experimentHealthLabel,
  experimentHealthTone,
} from "../components/ExperimentRunDetail";
import { NewCustomNode } from "../components/NewCustomNode";
import { AutoResearchEpisodeCard, EpisodeBudgetMeter } from "../components/CampaignRuns";
import { runsEpisodeCards } from "../campaigns";
import { projectExperimentExecution, type ExperimentRouteIdentity } from "../experimentBoard";
import type {
  AgentTask,
  Edge,
  Episode,
  EpisodeMessage,
  ExperimentControlState,
  ExperimentLoopIndexEntry,
  GraphNode,
  GraphState,
  Proposal,
  TrustView,
  WatcherRecord,
} from "../types";
import { nodeTypeLabel } from "../nodePresentation";

interface Props {
  graph: GraphState;
  trustView: TrustView;
  onSelectNode: (node: GraphNode) => void;
}

interface ScientificProps extends Props {
  mutationsDisabled?: boolean;
  onStageCustomNode: (node: GraphNode) => void;
}

const scienceOrder: GraphNode["type"][] = [
  "research_question",
  "hypothesis",
  "decision",
  "experiment",
  "evidence",
  "blocker",
];
const dagTypes = scienceOrder;
const dagTypeMeta: Record<GraphNode["type"], { label: string; color: string }> = {
  research_question: { label: "Questions", color: "#54718c" },
  hypothesis: { label: "Hypotheses", color: "#7a4166" },
  decision: { label: "Decisions", color: "#d7ae48" },
  experiment: { label: "Experiments", color: "#2f6f70" },
  evidence: { label: "Evidence", color: "#616b3d" },
  blocker: { label: "Blockers", color: "#bc5545" },
};

export function ScientificView({
  graph,
  trustView,
  onSelectNode,
  mutationsDisabled = false,
  onStageCustomNode,
}: ScientificProps) {
  const nodes = projectNodes(Object.values(graph.nodes), trustView);
  const projection = buildResearchPaths(nodes, Object.values(graph.edges));
  const hidden = Object.values(graph.nodes).length - nodes.length;
  return (
    <section className="view-panel research-view">
      <ViewHeading
        title="Research paths"
        aside={
          hidden > 0
            ? `${hidden} hidden`
            : `${projection.paths.length} question${projection.paths.length === 1 ? "" : "s"}`
        }
        action={
          <NewCustomNode
            ontology={graph.ontology}
            disabled={mutationsDisabled}
            existingNodeIds={new Set(Object.keys(graph.nodes))}
            onStage={onStageCustomNode}
          />
        }
      />
      {projection.paths.length === 0 && projection.unconnected.length === 0 ? (
        <EmptyState icon={<Search size={20} />} title="No research structure" />
      ) : (
        <>
          <div className="research-path-list">
            {projection.paths.map((path) => (
              <article className="research-path" key={path.question.id}>
                <div className="research-path-stage question-stage">
                  <span className="research-stage-label">Question</span>
                  <ResearchNodeCard node={path.question} onSelectNode={onSelectNode} />
                </div>
                <ResearchStage
                  label="Ideas & decisions"
                  nodes={path.ideas}
                  onSelectNode={onSelectNode}
                />
                <ResearchStage
                  label="Experiments"
                  nodes={path.experiments}
                  onSelectNode={onSelectNode}
                />
                <ResearchStage label="Evidence" nodes={path.evidence} onSelectNode={onSelectNode} />
              </article>
            ))}
          </div>
          {projection.unconnected.length > 0 && (
            <section className="research-unconnected">
              <header>
                <strong>Not yet connected</strong>
                <span>{projection.unconnected.length}</span>
              </header>
              <div>
                {projection.unconnected.map((node) => (
                  <ResearchNodeCard node={node} onSelectNode={onSelectNode} compact key={node.id} />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </section>
  );
}

interface DagProps extends Props {
  projectId: string;
  /** Session-scoped pan and zoom, owned by the shell so it survives leaving the view. */
  viewportRef: MutableRefObject<DagViewport | null>;
  relationFocusNodeId?: string | null;
  onClearRelationFocus?: () => void;
}

interface DragState {
  nodeId: string;
  pointerId: number;
  startX: number;
  startY: number;
  moved: boolean;
}

export function DagView({
  graph,
  trustView,
  onSelectNode,
  projectId,
  viewportRef,
  relationFocusNodeId,
  onClearRelationFocus,
}: DagProps) {
  const projection = useMemo(
    () => buildDagProjection(graph, trustView, relationFocusNodeId),
    [graph, relationFocusNodeId, trustView],
  );
  const naturalFocusNodeId = useMemo(
    () => dagFocusNode(projection.nodes, projection.edges),
    [projection],
  );
  const focusNodeId =
    relationFocusNodeId && graph.nodes[relationFocusNodeId]
      ? relationFocusNodeId
      : naturalFocusNodeId;
  const focusedRelations = useMemo(
    () => (relationFocusNodeId ? relationFocus(relationFocusNodeId, projection.edges) : null),
    [projection.edges, relationFocusNodeId],
  );
  const [brightTypes, setBrightTypes] = useState<Set<GraphNode["type"]>>(() => new Set(dagTypes));
  const [ontologyProjection, setOntologyProjection] = useState<DagOntologyProjection>("all");
  const nodeProjectionEmphasis = useMemo(
    () => buildNodeProjectionEmphasis(projection.edges, ontologyProjection),
    [ontologyProjection, projection.edges],
  );
  const [repulsion, setRepulsion] = useState(readRepulsion);
  const [layoutMode, setLayoutMode] = useState<DagLayoutMode>(readDagLayoutMode);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [zoom, setZoom] = useState(() => viewportRef.current?.zoom ?? 1);
  // An explicit relation focus outranks the remembered viewport: it is a request to look somewhere.
  const restoringViewportRef = useRef(relationFocusNodeId ? null : viewportRef.current);
  const shellRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef(zoom);
  const pendingZoomScrollRef = useRef<DagZoomResult | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const dragWatchdogRef = useRef<number | null>(null);
  const suppressClickRef = useRef<string | null>(null);
  const framedLayoutRef = useRef<string | null>(null);
  const commitViewport = useCallback(() => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    viewportRef.current = {
      zoom: zoomRef.current,
      scrollLeft: scroller.scrollLeft,
      scrollTop: scroller.scrollTop,
    };
  }, [viewportRef]);
  const layout = useForceDag({
    nodes: projection.nodes,
    edges: projection.edges,
    projectId,
    repulsion,
    mode: layoutMode,
  });
  const typeCounts = useMemo(
    () =>
      Object.fromEntries(
        dagTypes.map((type) => [
          type,
          projection.nodes.filter((node) => node.type === type).length,
        ]),
      ) as Record<GraphNode["type"], number>,
    [projection.nodes],
  );
  const layoutReady = Object.keys(layout.positions).length === projection.nodes.length;

  useEffect(() => {
    try {
      window.localStorage.setItem("rcp:dag-repulsion", String(repulsion));
    } catch {
      // Keep the control usable when browser storage is unavailable.
    }
  }, [repulsion]);

  useEffect(() => {
    try {
      window.localStorage.setItem("rcp:dag-layout-mode", layoutMode);
    } catch {
      // Keep the layout switch usable when browser storage is unavailable.
    }
  }, [layoutMode]);

  useEffect(
    () => () => {
      if (dragWatchdogRef.current !== null) window.clearTimeout(dragWatchdogRef.current);
    },
    [],
  );

  useEffect(() => {
    const updateFullscreenState = () =>
      setIsFullscreen(document.fullscreenElement === shellRef.current);
    document.addEventListener("fullscreenchange", updateFullscreenState);
    return () => document.removeEventListener("fullscreenchange", updateFullscreenState);
  }, []);

  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    const pinchZoom = (event: WheelEvent) => {
      if (!event.ctrlKey) return;
      event.preventDefault();
      const rect = scroller.getBoundingClientRect();
      const pending = pendingZoomScrollRef.current;
      const deltaScale =
        event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? scroller.clientHeight : 1;
      const next = zoomDagAtPoint({
        zoom: zoomRef.current,
        deltaY: event.deltaY * deltaScale,
        focalX: event.clientX - rect.left - scroller.clientLeft,
        focalY: event.clientY - rect.top - scroller.clientTop,
        scrollLeft: pending?.scrollLeft ?? scroller.scrollLeft,
        scrollTop: pending?.scrollTop ?? scroller.scrollTop,
      });
      if (next.zoom === zoomRef.current) return;
      pendingZoomScrollRef.current = next;
      viewportRef.current = next;
      zoomRef.current = next.zoom;
      setZoom(next.zoom);
    };
    scroller.addEventListener("wheel", pinchZoom, { passive: false });
    return () => scroller.removeEventListener("wheel", pinchZoom);
  }, [projection.nodes.length, viewportRef]);

  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    commitViewport();
    scroller.addEventListener("scroll", commitViewport, { passive: true });
    return () => scroller.removeEventListener("scroll", commitViewport);
  }, [commitViewport, projection.nodes.length]);

  useLayoutEffect(() => {
    const pending = pendingZoomScrollRef.current;
    const scroller = scrollRef.current;
    if (!pending || pending.zoom !== zoom || !scroller) return;
    scroller.scrollLeft = pending.scrollLeft;
    scroller.scrollTop = pending.scrollTop;
    pendingZoomScrollRef.current = null;
  }, [zoom]);

  useLayoutEffect(() => {
    const restored = restoringViewportRef.current;
    const scroller = scrollRef.current;
    if (!restored || !scroller) return;
    scroller.scrollLeft = restored.scrollLeft;
    scroller.scrollTop = restored.scrollTop;
  }, []);

  // Read the viewport while the canvas is still mounted; a scroll event may never arrive in time.
  useLayoutEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    return () => {
      viewportRef.current = {
        zoom: zoomRef.current,
        scrollLeft: scroller.scrollLeft,
        scrollTop: scroller.scrollTop,
      };
    };
  }, [projection.nodes.length, viewportRef]);

  useEffect(() => {
    if (!layoutReady || !focusNodeId) return;
    const frameKey = `${layoutMode}:${focusNodeId}:${projection.nodes.length}:${projection.edges.length}`;
    if (framedLayoutRef.current === frameKey) return;
    framedLayoutRef.current = frameKey;
    if (restoringViewportRef.current) {
      restoringViewportRef.current = null;
      return;
    }
    const timer = window.setTimeout(
      () => {
        const scroller = scrollRef.current;
        const focusNode = [
          ...(canvasRef.current?.querySelectorAll<HTMLElement>(".dag-node") ?? []),
        ].find((node) => node.dataset.nodeId === focusNodeId);
        if (!scroller || !focusNode) return;
        const currentZoom = zoomRef.current;
        scroller.scrollLeft = Math.max(0, focusNode.offsetLeft * currentZoom - 32);
        scroller.scrollTop = Math.max(
          0,
          focusNode.offsetTop * currentZoom -
            Math.max(32, (scroller.clientHeight - focusNode.offsetHeight * currentZoom) / 2),
        );
      },
      layoutMode === "force" ? 550 : 80,
    );
    return () => window.clearTimeout(timer);
  }, [focusNodeId, layoutMode, layoutReady, projection.edges.length, projection.nodes.length]);

  const toggleType = (type: GraphNode["type"]) => {
    setBrightTypes((current) => {
      const next = new Set(current);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const pointerDown = (event: ReactPointerEvent<HTMLDivElement>, nodeId: string) => {
    if (event.button !== 0) return;
    if (dragWatchdogRef.current !== null) window.clearTimeout(dragWatchdogRef.current);
    dragRef.current = {
      nodeId,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    };
  };

  const finishDrag = (pointerId: number) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== pointerId) return;
    if (dragWatchdogRef.current !== null) window.clearTimeout(dragWatchdogRef.current);
    dragWatchdogRef.current = null;
    if (drag.moved) {
      layout.endDrag();
      suppressClickRef.current = drag.nodeId;
      window.requestAnimationFrame(() => {
        if (suppressClickRef.current === drag.nodeId) suppressClickRef.current = null;
      });
    }
    dragRef.current = null;
    setDraggingId(null);
  };

  const pointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (!drag.moved && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 5)
      return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!drag.moved) {
      drag.moved = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      layout.beginDrag(drag.nodeId);
      setDraggingId(drag.nodeId);
    }
    const rect = canvas.getBoundingClientRect();
    layout.moveDrag(
      drag.nodeId,
      (event.clientX - rect.left) / zoomRef.current,
      (event.clientY - rect.top) / zoomRef.current,
    );
    if (dragWatchdogRef.current !== null) window.clearTimeout(dragWatchdogRef.current);
    dragWatchdogRef.current = window.setTimeout(() => finishDrag(drag.pointerId), 3000);
    event.preventDefault();
  };

  const pointerEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    finishDrag(event.pointerId);
  };

  const inspectNode = (node: GraphNode) => {
    if (suppressClickRef.current === node.id) {
      suppressClickRef.current = null;
      return;
    }
    onSelectNode(node);
  };

  const toggleFullscreen = async () => {
    const shell = shellRef.current;
    if (!shell || !document.fullscreenEnabled) return;
    try {
      if (document.fullscreenElement === shell) await document.exitFullscreen();
      else await shell.requestFullscreen();
    } catch {
      setIsFullscreen(document.fullscreenElement === shell);
    }
  };

  const allBright = brightTypes.size === dagTypes.length;
  const allDim = brightTypes.size === 0;
  const fullscreenSupported =
    typeof document !== "undefined" &&
    document.fullscreenEnabled &&
    "requestFullscreen" in HTMLElement.prototype;
  const repulsionLabel =
    repulsion < 650
      ? "Gentle"
      : repulsion < 1050
        ? "Balanced"
        : repulsion < 1500
          ? "Strong"
          : "Wide";
  return (
    <section className="view-panel dag-panel">
      <ViewHeading
        title="DAG view"
        aside={`${projection.nodes.length} nodes · ${projection.edges.length} edges`}
      />
      {relationFocusNodeId && graph.nodes[relationFocusNodeId] && (
        <div className="dag-relation-focus" role="status">
          <Focus size={15} />
          <span>
            <strong>Relation focus:</strong> {graph.nodes[relationFocusNodeId].title}. This node and
            its directly connected neighbors stay bright; all other graph context is dimmed.
          </span>
          <button className="button compact secondary" onClick={onClearRelationFocus}>
            <X size={13} /> Clear focus
          </button>
        </div>
      )}
      {projection.nodes.length === 0 ? (
        <EmptyState icon={<GitBranch size={20} />} title="No graph" />
      ) : (
        <div className="dag-shell" ref={shellRef}>
          <div className="dag-controls">
            <div className="dag-layout-controls">
              <span className="dag-control-label">
                <GitBranch size={14} /> Layout
              </span>
              <div className="dag-layout-switch" role="group" aria-label="DAG layout">
                <button
                  className={layoutMode === "force" ? "is-active" : ""}
                  type="button"
                  aria-pressed={layoutMode === "force"}
                  onClick={() => setLayoutMode("force")}
                >
                  <Orbit size={13} /> Force-directed
                </button>
                <button
                  className={layoutMode === "flow" ? "is-active" : ""}
                  type="button"
                  aria-pressed={layoutMode === "flow"}
                  onClick={() => setLayoutMode("flow")}
                >
                  <Workflow size={13} /> Research flow
                </button>
              </div>
              <span className="dag-control-label dag-projection-label">Projection</span>
              <div
                className="dag-layout-switch dag-projection-switch"
                role="group"
                aria-label="DAG ontology projection"
              >
                {(["all", "belief", "action"] as const).map((item) => (
                  <button
                    className={ontologyProjection === item ? "is-active" : ""}
                    type="button"
                    aria-pressed={ontologyProjection === item}
                    onClick={() => setOntologyProjection(item)}
                    key={item}
                  >
                    {item[0].toUpperCase() + item.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            <div className="dag-lenses" role="group" aria-label="Node brightness by type">
              <span className="dag-control-label">
                <Eye size={14} /> Brightness
              </span>
              {dagTypes.map((type) => {
                const active = brightTypes.has(type);
                const meta = dagTypeMeta[type];
                return (
                  <button
                    className={`dag-lens ${active ? "is-bright" : "is-dim"}`}
                    style={{ "--lens-color": meta.color } as CSSProperties}
                    type="button"
                    aria-pressed={active}
                    aria-label={`${active ? "Dim" : "Brighten"} ${meta.label.toLowerCase()} (${typeCounts[type]})`}
                    onClick={() => toggleType(type)}
                    key={type}
                  >
                    {active ? <Eye size={12} /> : <EyeOff size={12} />}
                    <span>{meta.label}</span>
                    <small>{typeCounts[type]}</small>
                  </button>
                );
              })}
              <button
                className="dag-tool-button"
                type="button"
                disabled={allBright}
                onClick={() => setBrightTypes(new Set(dagTypes))}
              >
                <Eye size={13} /> Brighten all
              </button>
              <button
                className="dag-tool-button"
                type="button"
                disabled={allDim}
                onClick={() => setBrightTypes(new Set())}
              >
                <EyeOff size={13} /> Dim all
              </button>
            </div>
            <div className="dag-physics-controls">
              {layoutMode === "force" && (
                <label className="dag-force-control">
                  <span>
                    <Gauge size={14} /> Repulsion
                  </span>
                  <input
                    aria-label="Node repulsion"
                    type="range"
                    min="350"
                    max="1900"
                    step="50"
                    value={repulsion}
                    onChange={(event) => setRepulsion(Number(event.target.value))}
                  />
                  <output>{repulsionLabel}</output>
                </label>
              )}
              <button
                className="dag-tool-button"
                type="button"
                disabled={layout.pinCount === 0}
                onClick={layout.releasePins}
              >
                <PinOff size={13} /> Release all pins
              </button>
              <button
                className="dag-tool-button"
                type="button"
                onClick={() => {
                  framedLayoutRef.current = null;
                  layout.resetLayout();
                }}
              >
                <RotateCcw size={13} /> Reset layout
              </button>
              {fullscreenSupported && (
                <button
                  aria-label={isFullscreen ? "Exit DAG full screen" : "Enter DAG full screen"}
                  aria-pressed={isFullscreen}
                  className="dag-tool-button"
                  type="button"
                  onClick={toggleFullscreen}
                >
                  {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
                  {isFullscreen ? "Exit full screen" : "Full screen"}
                </button>
              )}
            </div>
          </div>
          <div className="dag-scroll" ref={scrollRef}>
            <div
              className="dag-zoom-plane"
              style={{ width: layout.width * zoom, height: layout.height * zoom }}
            >
              <div
                className="dag-canvas"
                ref={canvasRef}
                style={{ width: layout.width, height: layout.height, transform: `scale(${zoom})` }}
              >
                <svg width={layout.width} height={layout.height} aria-hidden="true">
                  <defs>
                    <marker
                      id="dag-arrow"
                      viewBox="0 0 10 10"
                      refX="9"
                      refY="5"
                      markerWidth="6"
                      markerHeight="6"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 0 L 10 5 L 0 10 z" />
                    </marker>
                  </defs>
                  {projection.edges.map((edge, edgeIndex) => {
                    const source = layout.positions[edge.source];
                    const target = layout.positions[edge.target];
                    if (!source || !target) return null;
                    const geometry = edgeGeometry(source, target, edgeIndex);
                    const projectionEmphasis = edgeProjectionEmphasis(edge, ontologyProjection);
                    const dimmed =
                      projectionEmphasis === "dimmed" ||
                      !brightTypes.has(graph.nodes[edge.source]?.type) ||
                      !brightTypes.has(graph.nodes[edge.target]?.type) ||
                      Boolean(focusedRelations && !focusedRelations.edgeIds.has(edge.id));
                    return (
                      <g
                        className={`dag-edge-group ${dimmed ? "is-dim" : ""} ${projectionEmphasis === "neutral" ? "is-neutral" : ""}`}
                        key={edge.id}
                      >
                        <path className="dag-edge" d={geometry.path} markerEnd="url(#dag-arrow)" />
                        <text className="dag-edge-label" x={geometry.labelX} y={geometry.labelY}>
                          {edge.relation.replaceAll("_", " ")}
                        </text>
                      </g>
                    );
                  })}
                </svg>
                {projection.nodes.map((node) => {
                  const position = layout.positions[node.id];
                  if (!position) return null;
                  const projectionEmphasis =
                    nodeProjectionEmphasis.get(node.id) ??
                    (ontologyProjection === "all" ? "emphasized" : "neutral");
                  const dimmed =
                    projectionEmphasis === "dimmed" ||
                    !brightTypes.has(node.type) ||
                    Boolean(focusedRelations && !focusedRelations.nodeIds.has(node.id));
                  return (
                    <div
                      className={`dag-node ${node.type} ${node.standing} ${node.draft_touched ? "draft-touched" : ""} ${dimmed ? "is-dim" : ""} ${projectionEmphasis === "neutral" ? "is-layer-neutral" : ""} ${position.pinned ? "is-pinned" : ""} ${draggingId === node.id ? "is-dragging" : ""}`}
                      data-node-id={node.id}
                      style={
                        {
                          left: position.x - DAG_NODE_WIDTH / 2,
                          top: position.y - DAG_NODE_HEIGHT / 2,
                          "--node-accent": dagTypeMeta[node.type].color,
                        } as CSSProperties
                      }
                      key={node.id}
                      onDragStart={(event) => event.preventDefault()}
                      onPointerDown={(event) => pointerDown(event, node.id)}
                      onPointerMove={pointerMove}
                      onPointerUp={pointerEnd}
                      onPointerCancel={pointerEnd}
                      onLostPointerCapture={pointerEnd}
                    >
                      <button
                        aria-label={`${node.title}. Inspect node. Drag this card to pin it.`}
                        className="dag-node-inspect"
                        type="button"
                        onClick={() => inspectNode(node)}
                      >
                        <span className="eyebrow">{nodeTypeLabel(node)}</span>
                        <strong>{node.title}</strong>
                        <small>
                          <span className={`standing ${node.standing}`}>{node.standing}</span>
                          <span>{node.status || node.validity || ""}</span>
                        </small>
                      </button>
                      {position.pinned && (
                        <button
                          aria-label={`Release pin from ${node.title}`}
                          className="dag-pin-state"
                          title="Release this pin"
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            layout.releasePin(node.id);
                          }}
                          onPointerDown={(event) => event.stopPropagation()}
                        >
                          <Pin size={9} /> pinned
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

interface ExecutionProps {
  graph: GraphState;
  episodes: Episode[];
  episodeMessages: Readonly<Record<string, EpisodeMessage[] | undefined>>;
  episodeAction: string | null;
  tasks: AgentTask[];
  watchers: WatcherRecord[];
  experimentControl: Record<string, ExperimentControlState>;
  exactExperimentRoute?: ExperimentRouteIdentity | null;
  exactExperimentEntry?: ExperimentLoopIndexEntry | null;
  selectedExperimentId: string | null;
  focusExperimentId: string | null;
  runBusy: boolean;
  stopBusyId: string | null;
  watcherCheckBusyId: string | null;
  taskActionId: string | null;
  selectedExperimentConversation?: ReactNode;
  providerLabels?: Record<string, string>;
  mutationsDisabled?: boolean;
  experimentStartsDisabled?: boolean;
  onInspectTask: (operationId: string) => void;
  onLoadEpisodeMessages: (episodeId: string) => Promise<void>;
  onStopEpisode: (episodeId: string) => Promise<void>;
  onMergeEpisode: (episodeId: string) => Promise<void>;
  onReauthorizeEpisode: (episodeId: string, invocationCeiling: number) => Promise<void>;
  onSendEpisodeMessage: (episodeId: string, body: string) => Promise<void>;
  onOperateEpisodeTask: (task: AgentTask, action: "pause" | "resume" | "retry") => Promise<void>;
  onSelectExperiment: (nodeId: string | null) => void;
  onDetailFocused: () => void;
  onRunExperiment: (node: GraphNode) => void;
  onStopExperiment: (nodeId: string, episodeId?: string) => void;
  onCheckExperimentWatcher: (watcherId: string) => void;
  onRecoverExperiment: (task: AgentTask, action: "resume" | "retry") => void;
  onSwitchExperimentProvider: (task: AgentTask) => void;
  episodeReportHref: (episodeId: string) => string;
}

export function ExecutionView({
  graph,
  episodes,
  episodeMessages,
  episodeAction,
  tasks,
  watchers,
  experimentControl,
  exactExperimentRoute = null,
  exactExperimentEntry = null,
  selectedExperimentId,
  focusExperimentId,
  runBusy,
  stopBusyId,
  watcherCheckBusyId,
  taskActionId,
  selectedExperimentConversation,
  providerLabels = {},
  mutationsDisabled = false,
  experimentStartsDisabled = false,
  onInspectTask,
  onLoadEpisodeMessages,
  onStopEpisode,
  onMergeEpisode,
  onReauthorizeEpisode,
  onSendEpisodeMessage,
  onOperateEpisodeTask,
  onSelectExperiment,
  onDetailFocused,
  onRunExperiment,
  onStopExperiment,
  onCheckExperimentWatcher,
  onRecoverExperiment,
  onSwitchExperimentProvider,
  episodeReportHref,
}: ExecutionProps) {
  const selectedDetailRef = useRef<HTMLDivElement>(null);
  const exactProjection = projectExperimentExecution(
    Object.values(graph.nodes),
    tasks,
    watchers,
    experimentControl,
    exactExperimentRoute,
    exactExperimentEntry,
  );
  const experimentNodes = new Map(
    exactProjection.nodes
      .filter((node) => node.type === "experiment")
      .map((node) => [node.id, node]),
  );
  const experimentRuns = new Map<string, ExperimentRun>();
  experimentNodes.forEach((node, nodeId) => {
    const control = exactProjection.experimentControl[nodeId];
    if (!control) {
      throw new Error(`Experiment ${nodeId} is missing its backend control projection.`);
    }
    if (!control.health || !control.recommendation || !control.run_section) {
      throw new Error(`Experiment ${nodeId} has an incomplete backend control projection.`);
    }
    if (!control?.episode_id) return;
    experimentRuns.set(
      control.episode_id,
      buildExperimentRun(node, control, exactProjection.tasks, exactProjection.watchers),
    );
  });
  const orderedEpisodes = runsEpisodeCards(episodes, new Set(experimentRuns.keys()));
  const needsAction = orderedEpisodes.filter(
    (episode) => episodeRunSection(episode) === "needs_action",
  );
  const completed = orderedEpisodes.filter((episode) => episodeRunSection(episode) === "completed");
  const completedGroups = [
    {
      mode: "experiment_loop" as const,
      title: "Experiment loop",
      episodes: completed.filter((episode) => episode.mode === "experiment_loop"),
    },
    {
      mode: "auto_research" as const,
      title: "Auto-research",
      episodes: completed.filter((episode) => episode.mode === "auto_research"),
    },
  ];

  useEffect(() => {
    if (!focusExperimentId || focusExperimentId !== selectedExperimentId) return;
    selectedDetailRef.current?.focus();
    onDetailFocused();
  }, [focusExperimentId, onDetailFocused, selectedExperimentId]);

  return (
    <section className="view-panel runs-view" aria-label="Runs">
      <div className="operating-sections episode-ledger-sections">
        <section className="operating-section episode-ledger-section needs-action">
          <header>
            <h2>Needs Action</h2>
            <span>{needsAction.length}</span>
          </header>
          <div className="campaign-run-list">
            {needsAction.map((episode, index) => renderEpisodeCard(episode, index === 0))}
          </div>
        </section>
        <section className="operating-section episode-ledger-section completed">
          <header>
            <h2>Completed</h2>
            <span>{completed.length}</span>
          </header>
          <div className="episode-type-groups">
            {completedGroups.map((group) => (
              <details className="episode-type-group" key={group.mode}>
                <summary>
                  <strong>{group.title}</strong>
                  <span>{group.episodes.length}</span>
                </summary>
                <div className="campaign-run-list">
                  {group.episodes.map((episode) => renderEpisodeCard(episode, false))}
                </div>
              </details>
            ))}
          </div>
        </section>
      </div>
    </section>
  );

  function renderEpisodeCard(episode: Episode, initiallyExpanded: boolean) {
    if (episode.mode === "auto_research") {
      return (
        <AutoResearchEpisodeCard
          episode={episode}
          tasks={tasks}
          messages={episodeMessages[episode.episode_id] ?? []}
          initiallyExpanded={initiallyExpanded}
          busyAction={episodeAction}
          taskActionId={taskActionId}
          onInspectTask={onInspectTask}
          onLoadMessages={onLoadEpisodeMessages}
          onStop={onStopEpisode}
          onMerge={onMergeEpisode}
          onReauthorize={onReauthorizeEpisode}
          onSendMessage={onSendEpisodeMessage}
          onOperateTask={onOperateEpisodeTask}
          key={episode.episode_id}
        />
      );
    }
    const run = experimentRuns.get(episode.episode_id);
    if (!run) {
      throw new Error(
        `Experiment episode ${episode.episode_id} is missing its backend control projection.`,
      );
    }
    return (
      <ExperimentEpisodeCard
        episode={episode}
        run={run}
        initiallyExpanded={initiallyExpanded || selectedExperimentId === run.node.id}
        selected={selectedExperimentId === run.node.id}
        detailRef={selectedExperimentId === run.node.id ? selectedDetailRef : undefined}
        runBusy={runBusy}
        stopBusy={stopBusyId === run.node.id}
        watcherCheckBusyId={watcherCheckBusyId}
        taskActionId={taskActionId}
        experimentConversation={selectedExperimentConversation}
        exactBranchEntry={exactProjection.exactBranchEntry}
        providerLabels={providerLabels}
        experimentStartsDisabled={experimentStartsDisabled}
        mutationsDisabled={
          mutationsDisabled ||
          runBusy ||
          Boolean(stopBusyId) ||
          Boolean(taskActionId) ||
          Boolean(watcherCheckBusyId)
        }
        onSelectExperiment={onSelectExperiment}
        onRunExperiment={onRunExperiment}
        onStopExperiment={onStopExperiment}
        onCheckExperimentWatcher={onCheckExperimentWatcher}
        onRecoverExperiment={onRecoverExperiment}
        onSwitchExperimentProvider={onSwitchExperimentProvider}
        episodeReportHref={episodeReportHref}
        key={episode.episode_id}
      />
    );
  }

  function episodeRunSection(episode: Episode): "needs_action" | "completed" {
    if (episode.mode === "auto_research") return episode.run_section;
    const run = experimentRuns.get(episode.episode_id);
    if (!run) {
      throw new Error(
        `Experiment episode ${episode.episode_id} is missing its backend control projection.`,
      );
    }
    return run.control.run_section === "completed" ? "completed" : "needs_action";
  }
}

function ExperimentEpisodeCard({
  episode,
  run,
  initiallyExpanded,
  selected,
  detailRef,
  runBusy,
  stopBusy,
  watcherCheckBusyId,
  taskActionId,
  experimentConversation,
  exactBranchEntry,
  providerLabels,
  experimentStartsDisabled,
  mutationsDisabled,
  onSelectExperiment,
  onRunExperiment,
  onStopExperiment,
  onCheckExperimentWatcher,
  onRecoverExperiment,
  onSwitchExperimentProvider,
  episodeReportHref,
}: {
  episode: Episode;
  run: ExperimentRun;
  initiallyExpanded: boolean;
  selected: boolean;
  detailRef?: Ref<HTMLDivElement>;
  runBusy: boolean;
  stopBusy: boolean;
  watcherCheckBusyId: string | null;
  taskActionId: string | null;
  experimentConversation?: ReactNode;
  exactBranchEntry: ExperimentLoopIndexEntry | null;
  providerLabels: Record<string, string>;
  experimentStartsDisabled: boolean;
  mutationsDisabled: boolean;
  onSelectExperiment: (nodeId: string | null) => void;
  onRunExperiment: (node: GraphNode) => void;
  onStopExperiment: (nodeId: string, episodeId?: string) => void;
  onCheckExperimentWatcher: (watcherId: string) => void;
  onRecoverExperiment: (task: AgentTask, action: "resume" | "retry") => void;
  onSwitchExperimentProvider: (task: AgentTask) => void;
  episodeReportHref: (episodeId: string) => string;
}) {
  const detailId = useId();
  const [expanded, setExpanded] = useState(initiallyExpanded);
  const tone = experimentHealthTone(run.health);
  const title = run.node.title;
  const episodeTimestamp = formatEpisodeTimestamp(episode.created_at);
  const isExactBranchEpisode = exactBranchEntry?.episode?.episode_id === episode.episode_id;
  const exactEpisodeId = isExactBranchEpisode
    ? (exactBranchEntry?.episode?.episode_id ?? exactBranchEntry?.control.episode_id)
    : null;

  useEffect(() => {
    if (selected) setExpanded(true);
  }, [selected]);

  return (
    <article className={`campaign-run experiment-episode-card ${tone}`}>
      <span className="campaign-state-rail" aria-hidden="true" />
      <div className="campaign-run-heading">
        <button
          className="campaign-run-toggle"
          type="button"
          aria-label={`${expanded ? "Collapse" : "Expand"} Experiment loop episode ${title}`}
          aria-expanded={expanded}
          aria-controls={detailId}
          onClick={() => {
            const next = !expanded;
            setExpanded(next);
            onSelectExperiment(next ? run.node.id : null);
          }}
        />
        <span className="campaign-run-identity">
          <strong className="campaign-run-title">
            <FlaskConical size={14} aria-hidden="true" />
            <span>{title}</span>
          </strong>
          <span className="campaign-run-meta">
            <span className={`status-pill ${tone}`}>{experimentHealthLabel(run.health)}</span>
            <time dateTime={episode.created_at}>{episodeTimestamp}</time>
          </span>
        </span>
        <EpisodeBudgetMeter episode={episode} />
        <span className="campaign-run-time">
          <ChevronDown size={15} aria-hidden="true" />
        </span>
      </div>
      {expanded && (
        <div className="campaign-run-detail" id={detailId} tabIndex={-1} ref={detailRef}>
          <ExperimentRunDetail
            run={run}
            runBusy={runBusy}
            runDisabled={mutationsDisabled}
            startDisabled={experimentStartsDisabled}
            stopBusy={stopBusy}
            recoveryBusy={Boolean(run.currentTask && taskActionId === run.currentTask.operation_id)}
            watcherCheckBusyId={watcherCheckBusyId}
            providerLabel={experimentProviderLabel(run, providerLabels)}
            conversation={experimentConversation}
            allowStart={!isExactBranchEpisode}
            onRun={() => onRunExperiment(run.node)}
            onStopLoop={() => onStopExperiment(run.node.id, exactEpisodeId ?? episode.episode_id)}
            onCheckWatcher={onCheckExperimentWatcher}
            onRecover={(action) => {
              if (run.currentTask) onRecoverExperiment(run.currentTask, action);
            }}
            onSwitchProvider={() => {
              if (run.currentTask) onSwitchExperimentProvider(run.currentTask);
            }}
            episodeReportHref={episodeReportHref}
          />
        </div>
      )}
    </article>
  );
}

function formatEpisodeTimestamp(value: string): string {
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
}

function experimentProviderLabel(
  experiment: ExperimentRun,
  providerLabels: Record<string, string>,
): string | undefined {
  const provider =
    experiment.currentTask?.request.provider || experiment.control?.operational?.session.provider;
  return provider ? providerLabels[provider] || provider : undefined;
}

interface AttentionOverviewProps {
  proposals: Proposal[];
  decisions: GraphNode[];
  blockers: GraphNode[];
  onSelectNode: (node: GraphNode) => void;
}

export function AttentionOverview({
  proposals,
  decisions,
  blockers,
  onSelectNode,
}: AttentionOverviewProps) {
  return (
    <section className="view-panel">
      <ViewHeading
        title="Attention view"
        aside={`${proposals.length + decisions.length + blockers.length} open`}
      />
      <div className="attention-overview-grid">
        <OverviewCard label="Pending proposals" value={proposals.length} />
        <OverviewCard label="Decisions awaiting choice" value={decisions.length} />
        <OverviewCard label="Blockers awaiting judgment" value={blockers.length} />
      </div>
      <h3 className="section-label">Recommended next action</h3>
      {proposals[0] ? (
        <div className="recommended-action">
          <CircleDot size={17} />
          <strong>Understand and decide “{proposals[0].title}”</strong>
        </div>
      ) : decisions[0] ? (
        <button className="recommended-action" onClick={() => onSelectNode(decisions[0])}>
          <CircleDot size={17} />
          <strong>Choose “{decisions[0].title}”</strong>
        </button>
      ) : blockers[0] ? (
        <button className="recommended-action" onClick={() => onSelectNode(blockers[0])}>
          <CircleDot size={17} />
          <strong>Inspect “{blockers[0].title}”</strong>
        </button>
      ) : (
        <div className="quiet-empty compact">
          <CircleDot size={17} />
          <strong>No judgment queued</strong>
        </div>
      )}
    </section>
  );
}

function ResearchStage({
  label,
  nodes,
  onSelectNode,
}: {
  label: string;
  nodes: GraphNode[];
  onSelectNode: (node: GraphNode) => void;
}) {
  return (
    <div className="research-path-stage">
      <span className="research-stage-label">{label}</span>
      <div className="research-stage-cards">
        {nodes.length > 0 ? (
          nodes.map((node) => (
            <ResearchNodeCard node={node} onSelectNode={onSelectNode} key={node.id} />
          ))
        ) : (
          <span className="research-stage-empty">—</span>
        )}
      </div>
    </div>
  );
}

function ResearchNodeCard({
  node,
  onSelectNode,
  compact = false,
}: {
  node: GraphNode;
  onSelectNode: (node: GraphNode) => void;
  compact?: boolean;
}) {
  return (
    <button
      className={`research-node-card ${node.standing} ${node.draft_touched ? "draft-touched" : ""} ${compact ? "compact" : ""}`}
      onClick={() => onSelectNode(node)}
    >
      <span className="research-node-topline">
        <span>{nodeTypeLabel(node)}</span>
        <span className={`standing ${node.standing}`}>{node.standing}</span>
      </span>
      <strong>{node.title}</strong>
    </button>
  );
}

function dagFocusNode(nodes: GraphNode[], edges: Edge[]): string | undefined {
  const incoming = new Set(edges.map((edge) => edge.target));
  const outgoing = new Map<string, number>();
  edges.forEach((edge) => outgoing.set(edge.source, (outgoing.get(edge.source) ?? 0) + 1));
  const questions = nodes.filter((node) => node.type === "research_question");
  const roots = questions.filter((node) => !incoming.has(node.id));
  return (
    [...(roots.length > 0 ? roots : questions)].sort(
      (left, right) =>
        (outgoing.get(right.id) ?? 0) - (outgoing.get(left.id) ?? 0) ||
        left.id.localeCompare(right.id),
    )[0]?.id ?? nodes[0]?.id
  );
}

function edgeGeometry(source: DagPosition, target: DagPosition, edgeIndex: number) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const ux = dx / distance;
  const uy = dy / distance;
  const horizontalReach = DAG_NODE_WIDTH / 2 / Math.max(0.001, Math.abs(ux));
  const verticalReach = DAG_NODE_HEIGHT / 2 / Math.max(0.001, Math.abs(uy));
  const reach = Math.min(horizontalReach, verticalReach);
  const startX = source.x + ux * reach;
  const startY = source.y + uy * reach;
  const endX = target.x - ux * reach;
  const endY = target.y - uy * reach;
  const curve = ((edgeIndex % 5) - 2) * 9;
  const controlX = (startX + endX) / 2 - uy * curve;
  const controlY = (startY + endY) / 2 + ux * curve;
  return {
    path: `M ${startX} ${startY} Q ${controlX} ${controlY}, ${endX} ${endY}`,
    labelX: (startX + 2 * controlX + endX) / 4,
    labelY: (startY + 2 * controlY + endY) / 4 - 6,
  };
}

function readRepulsion(): number {
  try {
    const stored = Number(window.localStorage.getItem("rcp:dag-repulsion"));
    if (Number.isFinite(stored) && stored >= 350 && stored <= 1900) return stored;
  } catch {
    // Use the balanced default when browser storage is unavailable.
  }
  return 950;
}

function readDagLayoutMode(): DagLayoutMode {
  try {
    return window.localStorage.getItem("rcp:dag-layout-mode") === "flow" ? "flow" : "force";
  } catch {
    return "force";
  }
}

function ViewHeading({
  title,
  aside,
  action,
}: {
  title: string;
  aside: string;
  action?: React.ReactNode;
}) {
  return (
    <header className="view-heading">
      <h2>{title}</h2>
      <span className="view-aside">{aside}</span>
      {action}
    </header>
  );
}

function EmptyState({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="empty-state">
      {icon}
      <strong>{title}</strong>
    </div>
  );
}

function OverviewCard({ label, value }: { label: string; value: number }) {
  return (
    <article className="overview-card">
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
    </article>
  );
}
