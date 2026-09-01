import { useCallback, useEffect, useLayoutEffect, useReducer, useRef, useState } from "react";
import {
  experimentBoardHref,
  projectHashAfterViewChange,
  type ExperimentRouteIdentity,
  type ProjectHashRoute,
} from "../experimentBoard";
import type { DetailWindowSlot } from "../floatingWindow";
import { projectViewportRef, type ProjectViewState, type ProjectViewportRef } from "../projectTabs";
import type {
  AppView,
  GraphNode,
  GraphState,
  PaperSnapshot,
  ProjectSnapshot,
  TrustView,
} from "../types";
import type { DagViewport } from "./dagZoom";

const emptyGraph: GraphState = {
  revision: 0,
  nodes: {},
  edges: {},
  proposals: {},
  ambiguities: {},
  glossary: {},
  validation_messages: [],
  belief_transitions: [],
  replay_status: "complete",
  replay_failure: null,
  ontology: { types: [], fields: [], relations: [] },
};

export interface GraphSelectionTabSnapshot {
  runScope: string[];
  selectedNodeId: string | null;
  companionNodeId: string | null;
  detailFocusTokens: Record<DetailWindowSlot, number>;
  selectedExperimentRunId: string | null;
  focusExperimentRunId: string | null;
  selectedExperimentRoute: ExperimentRouteIdentity | null;
  dockedNodeIds: string[];
  dagRelationFocusId: string | null;
  viewState: ProjectViewState;
}

type SelectionSnapshot = Omit<GraphSelectionTabSnapshot, "viewState">;

interface UseGraphSelectionOptions {
  initialView: AppView;
  initialExperimentId: string | null;
  initialExperimentRoute: ExperimentRouteIdentity | null;
  projectId: string | null;
  loadedProjectId: string | null;
  loading: boolean;
}

export interface ExperimentSelectionState {
  selectedExperimentRunId: string | null;
  focusExperimentRunId: string | null;
  selectedExperimentRoute: ExperimentRouteIdentity | null;
}

export type ExperimentSelectionAction =
  | {
      kind: "route";
      experimentId: string | null;
      experimentRoute: ExperimentRouteIdentity | null;
    }
  | {
      kind: "restore";
      experimentId: string | null;
      focusExperimentId: string | null;
      experimentRoute: ExperimentRouteIdentity | null;
    }
  | { kind: "select"; experimentId: string | null }
  | { kind: "show"; experimentId: string }
  | { kind: "clear_focus" }
  | { kind: "view_changed" };

export function reduceExperimentSelection(
  state: ExperimentSelectionState,
  action: ExperimentSelectionAction,
): ExperimentSelectionState {
  if (action.kind === "view_changed") return state;
  if (action.kind === "clear_focus") {
    return state.focusExperimentRunId === null ? state : { ...state, focusExperimentRunId: null };
  }
  if (action.kind === "select") {
    const retainsExactRoute =
      action.experimentId === null ||
      action.experimentId === state.selectedExperimentRoute?.experiment_id;
    return {
      ...state,
      selectedExperimentRunId: action.experimentId,
      selectedExperimentRoute: retainsExactRoute ? state.selectedExperimentRoute : null,
    };
  }
  if (action.kind === "show") {
    return {
      selectedExperimentRunId: action.experimentId,
      focusExperimentRunId: action.experimentId,
      selectedExperimentRoute: null,
    };
  }
  return {
    selectedExperimentRunId: action.experimentId,
    focusExperimentRunId:
      action.kind === "restore" ? action.focusExperimentId : action.experimentId,
    selectedExperimentRoute: copyExperimentRoute(action.experimentRoute),
  };
}

export function relatedNodeWindowAction(
  sourceSlot: DetailWindowSlot,
  targetNodeId: string,
  originalNodeId: string | null,
  companionNodeId: string | null,
): { kind: "focus" | "open"; slot: DetailWindowSlot } {
  if (targetNodeId === originalNodeId) return { kind: "focus", slot: "original" };
  if (targetNodeId === companionNodeId) return { kind: "focus", slot: "companion" };
  return { kind: "open", slot: sourceSlot === "original" ? "companion" : "original" };
}

export function useGraphSelection({
  initialView,
  initialExperimentId,
  initialExperimentRoute,
  projectId,
  loadedProjectId,
  loading,
}: UseGraphSelectionOptions) {
  const [graph, setGraph] = useState<GraphState>(emptyGraph);
  const [paper, setPaper] = useState<PaperSnapshot | null>(null);
  const [view, setView] = useState<AppView>(initialView);
  const [trustView, setTrustView] = useState<TrustView>(readTrustView);
  const [runScope, setRunScope] = useState<string[]>([]);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [companionNode, setCompanionNode] = useState<GraphNode | null>(null);
  const [detailFocusTokens, setDetailFocusTokens] = useState<Record<DetailWindowSlot, number>>({
    original: 0,
    companion: 0,
  });
  const [experimentSelection, dispatchExperimentSelection] = useReducer(reduceExperimentSelection, {
    selectedExperimentRunId: initialExperimentId,
    focusExperimentRunId: initialExperimentId,
    selectedExperimentRoute: copyExperimentRoute(initialExperimentRoute),
  });
  const { selectedExperimentRunId, focusExperimentRunId, selectedExperimentRoute } =
    experimentSelection;
  const [experimentStopId, setExperimentStopId] = useState<string | null>(null);
  const [watcherCheckId, setWatcherCheckId] = useState<string | null>(null);
  const [dockedNodeIds, setDockedNodeIds] = useState<string[]>([]);
  const [dagRelationFocusId, setDagRelationFocusId] = useState<string | null>(null);
  const panelRef = useRef<HTMLElement>(null);
  const panelScrollRef = useRef(new Map<AppView, number>());
  const viewRef = useRef<AppView>(view);
  const researchSubviewRef = useRef<AppView>("scientific");
  const dagViewportRefsRef = useRef(new Map<string, ProjectViewportRef<DagViewport>>());
  const selectionSnapshotRef = useRef<SelectionSnapshot>({
    runScope,
    selectedNodeId: selectedNode?.id ?? null,
    companionNodeId: companionNode?.id ?? null,
    detailFocusTokens,
    selectedExperimentRunId,
    focusExperimentRunId,
    selectedExperimentRoute,
    dockedNodeIds,
    dagRelationFocusId,
  });
  selectionSnapshotRef.current = {
    runScope,
    selectedNodeId: selectedNode?.id ?? null,
    companionNodeId: companionNode?.id ?? null,
    detailFocusTokens,
    selectedExperimentRunId,
    focusExperimentRunId,
    selectedExperimentRoute,
    dockedNodeIds,
    dagRelationFocusId,
  };

  const activeDagViewportRef = projectId
    ? projectViewportRef(dagViewportRefsRef.current, projectId)
    : null;

  const captureProjectSelection = useCallback((id: string): GraphSelectionTabSnapshot => {
    const panelScroll = new Map(panelScrollRef.current);
    const dagViewport = dagViewportRefsRef.current.get(id)?.current ?? null;
    if (panelRef.current) panelScroll.set(viewRef.current, panelRef.current.scrollTop);
    const current = selectionSnapshotRef.current;
    return {
      ...current,
      runScope: [...current.runScope],
      detailFocusTokens: { ...current.detailFocusTokens },
      selectedExperimentRoute: copyExperimentRoute(current.selectedExperimentRoute),
      dockedNodeIds: [...current.dockedNodeIds],
      viewState: {
        view: viewRef.current,
        panelScroll: [...panelScroll.entries()],
        researchSubview: researchSubviewRef.current,
        dagViewport: dagViewport ? { ...dagViewport } : null,
      },
    };
  }, []);

  const restoreProjectSelection = useCallback(
    (
      id: string,
      project: ProjectSnapshot,
      presentedNodes: GraphState["nodes"],
      snapshot: GraphSelectionTabSnapshot,
      requestedRoute?: ProjectHashRoute,
    ) => {
      const nextGraph = project.graph;
      setGraph(nextGraph);
      setPaper(project.paper);
      setRunScope([...snapshot.runScope]);
      setSelectedNode(
        snapshot.selectedNodeId ? (presentedNodes[snapshot.selectedNodeId] ?? null) : null,
      );
      setCompanionNode(
        snapshot.companionNodeId ? (presentedNodes[snapshot.companionNodeId] ?? null) : null,
      );
      setDetailFocusTokens({ ...snapshot.detailFocusTokens });
      dispatchExperimentSelection(
        requestedRoute
          ? {
              kind: "route",
              experimentId: requestedRoute.experimentId,
              experimentRoute: requestedRoute.experimentRoute,
            }
          : {
              kind: "restore",
              experimentId: snapshot.selectedExperimentRunId,
              focusExperimentId: snapshot.focusExperimentRunId,
              experimentRoute: snapshot.selectedExperimentRoute ?? null,
            },
      );
      setExperimentStopId(null);
      setWatcherCheckId(null);
      setDockedNodeIds(snapshot.dockedNodeIds.filter((nodeId) => Boolean(nextGraph.nodes[nodeId])));
      setDagRelationFocusId(snapshot.dagRelationFocusId);
      panelScrollRef.current = new Map(snapshot.viewState.panelScroll);
      researchSubviewRef.current = snapshot.viewState.researchSubview;
      const viewportRef = projectViewportRef(dagViewportRefsRef.current, id);
      viewportRef.current = snapshot.viewState.dagViewport
        ? { ...snapshot.viewState.dagViewport }
        : null;
      setView(requestedRoute?.view ?? snapshot.viewState.view);
    },
    [],
  );

  const resetProjectSelection = useCallback(
    (
      nextView: AppView,
      experimentId: string | null,
      experimentRoute: ExperimentRouteIdentity | null,
    ) => {
      setGraph(emptyGraph);
      setPaper(null);
      setSelectedNode(null);
      setCompanionNode(null);
      dispatchExperimentSelection({ kind: "route", experimentId, experimentRoute });
      setExperimentStopId(null);
      setWatcherCheckId(null);
      setDockedNodeIds([]);
      setDagRelationFocusId(null);
      setRunScope([]);
      panelScrollRef.current = new Map();
      researchSubviewRef.current = "scientific";
      setView(nextView);
    },
    [],
  );

  const applyCanonicalProject = useCallback(
    (nextProject: ProjectSnapshot, authoritative: boolean) => {
      const nextGraph = nextProject.graph;
      setGraph(nextGraph);
      setPaper(nextProject.paper);
      setSelectedNode((current) =>
        current ? (nextGraph.nodes[current.id] ?? (authoritative ? null : current)) : null,
      );
      setCompanionNode((current) =>
        current ? (nextGraph.nodes[current.id] ?? (authoritative ? null : current)) : null,
      );
      setDockedNodeIds((current) => current.filter((nodeId) => nextGraph.nodes[nodeId]));
      setRunScope((current) =>
        current.length
          ? current.filter((item) => nextProject.project_truth_scope.includes(item))
          : nextProject.default_run_truth_scope,
      );
    },
    [],
  );

  const applySyncedGraph = useCallback((nextGraph: GraphState) => {
    setGraph(nextGraph);
    setSelectedNode((current) => (current ? (nextGraph.nodes[current.id] ?? null) : null));
    setCompanionNode((current) => (current ? (nextGraph.nodes[current.id] ?? null) : null));
  }, []);

  const replacePaper = useCallback((nextPaper: PaperSnapshot) => {
    setPaper(nextPaper);
  }, []);
  const replaceRunScope = useCallback((nextScope: string[]) => {
    setRunScope(nextScope);
  }, []);
  const applyRouteSelection = useCallback(
    (
      nextView: AppView,
      experimentId: string | null,
      experimentRoute: ExperimentRouteIdentity | null,
    ) => {
      setView(nextView);
      dispatchExperimentSelection({ kind: "route", experimentId, experimentRoute });
    },
    [],
  );

  // Capture the outgoing scroll synchronously while its view is still mounted.
  const changeView = useCallback((next: AppView) => {
    const panel = panelRef.current;
    if (panel) panelScrollRef.current.set(viewRef.current, panel.scrollTop);
    const replacementHash = projectHashAfterViewChange(window.location.hash, next);
    if (replacementHash) window.history.replaceState(null, "", replacementHash);
    dispatchExperimentSelection({ kind: "view_changed" });
    setView(next);
  }, []);
  const openLastResearchView = useCallback(() => {
    changeView(researchSubviewRef.current);
  }, [changeView]);

  const changeTrustView = useCallback((next: TrustView) => {
    setTrustView(next);
  }, []);
  const openNode = useCallback(
    (node: GraphNode | null) => {
      if (!node) return;
      setDockedNodeIds((current) => current.filter((nodeId) => nodeId !== node.id));
      if (selectedNode?.id === node.id) {
        setDetailFocusTokens((current) => ({ ...current, original: current.original + 1 }));
        return;
      }
      if (companionNode?.id === node.id) {
        setDetailFocusTokens((current) => ({ ...current, companion: current.companion + 1 }));
        return;
      }
      setSelectedNode(node);
      setCompanionNode(null);
      setDetailFocusTokens((current) => ({ ...current, original: current.original + 1 }));
    },
    [companionNode?.id, selectedNode?.id],
  );
  const openRelatedNode = useCallback(
    (sourceSlot: DetailWindowSlot, node: GraphNode | null) => {
      if (!node) return;
      setDockedNodeIds((current) => current.filter((id) => id !== node.id));
      const action = relatedNodeWindowAction(
        sourceSlot,
        node.id,
        selectedNode?.id ?? null,
        companionNode?.id ?? null,
      );
      if (action.kind === "focus") {
        setDetailFocusTokens((current) => ({
          ...current,
          [action.slot]: current[action.slot] + 1,
        }));
        return;
      }
      if (action.slot === "original") setSelectedNode(node);
      else setCompanionNode(node);
      setDetailFocusTokens((current) => ({
        ...current,
        [action.slot]: current[action.slot] + 1,
      }));
    },
    [companionNode?.id, selectedNode?.id],
  );
  const closeDetailSlot = useCallback((slot: DetailWindowSlot) => {
    if (slot === "original") setSelectedNode(null);
    else setCompanionNode(null);
  }, []);
  const clearNodeSelections = useCallback(() => {
    setSelectedNode(null);
    setCompanionNode(null);
  }, []);
  const dockNode = useCallback(
    (nodeId: string, slot: DetailWindowSlot) => {
      setDockedNodeIds((current) => (current.includes(nodeId) ? current : [...current, nodeId]));
      closeDetailSlot(slot);
    },
    [closeDetailSlot],
  );
  const restoreDockedNode = useCallback(
    (nodeId: string, node: GraphNode | null) => {
      setDockedNodeIds((current) => current.filter((id) => id !== nodeId));
      openNode(node);
    },
    [openNode],
  );

  const selectExperiment = useCallback(
    (nodeId: string | null) => {
      if (
        nodeId &&
        selectedExperimentRoute &&
        nodeId !== selectedExperimentRoute.experiment_id &&
        projectId
      ) {
        window.history.replaceState(null, "", experimentBoardHref(projectId, nodeId));
      }
      dispatchExperimentSelection({ kind: "select", experimentId: nodeId });
    },
    [projectId, selectedExperimentRoute],
  );
  const clearExperimentFocus = useCallback(() => {
    dispatchExperimentSelection({ kind: "clear_focus" });
  }, []);
  const showExperiment = useCallback(
    (nodeId: string) => {
      dispatchExperimentSelection({ kind: "show", experimentId: nodeId });
      setSelectedNode(null);
      setCompanionNode(null);
      changeView("execution");
    },
    [changeView],
  );
  const beginExperimentStop = useCallback((nodeId: string) => {
    setExperimentStopId(nodeId);
    return () => setExperimentStopId(null);
  }, []);
  const beginWatcherCheck = useCallback((watcherId: string) => {
    setWatcherCheckId(watcherId);
    return () => setWatcherCheckId(null);
  }, []);
  const clearDagRelationFocus = useCallback(() => {
    setDagRelationFocusId(null);
  }, []);
  const forgetProjectViewport = useCallback((id: string) => {
    dagViewportRefsRef.current.delete(id);
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("rcp:trust-view", trustView);
    } catch {
      // The chosen view is a convenience; storage failures must not affect the project.
    }
  }, [trustView]);

  useLayoutEffect(() => {
    viewRef.current = view;
    if (view === "scientific" || view === "dag") researchSubviewRef.current = view;
    const panel = panelRef.current;
    if (panel) panel.scrollTop = panelScrollRef.current.get(view) ?? 0;
  }, [loading, loadedProjectId, view]);

  return {
    graph,
    paper,
    view,
    trustView,
    runScope,
    selectedNode,
    companionNode,
    detailFocusTokens,
    selectedExperimentRunId,
    focusExperimentRunId,
    selectedExperimentRoute,
    experimentStopId,
    watcherCheckId,
    dockedNodeIds,
    dagRelationFocusId,
    panelRef,
    activeDagViewportRef,
    captureProjectSelection,
    restoreProjectSelection,
    resetProjectSelection,
    applyCanonicalProject,
    applySyncedGraph,
    replacePaper,
    replaceRunScope,
    applyRouteSelection,
    changeView,
    openLastResearchView,
    changeTrustView,
    openNode,
    openRelatedNode,
    closeDetailSlot,
    clearNodeSelections,
    dockNode,
    restoreDockedNode,
    selectExperiment,
    clearExperimentFocus,
    showExperiment,
    beginExperimentStop,
    beginWatcherCheck,
    clearDagRelationFocus,
    forgetProjectViewport,
  };
}

function copyExperimentRoute(
  route: ExperimentRouteIdentity | null | undefined,
): ExperimentRouteIdentity | null {
  if (!route) return null;
  return {
    ...route,
    graph_target:
      route.graph_target.kind === "branch"
        ? { kind: "branch", branch_id: route.graph_target.branch_id }
        : { kind: "main" },
  };
}

function readTrustView(): TrustView {
  try {
    return (localStorage.getItem("rcp:trust-view") as TrustView) || "working";
  } catch {
    return "working";
  }
}
