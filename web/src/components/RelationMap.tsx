import { AlertTriangle, ExternalLink, Maximize2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { edgeValidationFlags } from "../nodeDetail";
import { humanize, nodeTypeLabel, presentNode } from "../nodePresentation";
import type { Edge, GraphNode, ValidationMessage } from "../types";

export interface RelationMapProps {
  focusedNode: GraphNode;
  allNodes: Record<string, GraphNode>;
  incidentEdges: Edge[];
  validationMessages: ValidationMessage[];
  onOpenNodeWindow: (nodeId: string) => void;
}

export interface RelationPeerGroup {
  nodeId: string;
  edges: Edge[];
}

export interface OneHopRelationGroups {
  incoming: RelationPeerGroup[];
  outgoing: RelationPeerGroup[];
}

type EvidenceAssessmentPresentation = NonNullable<Edge["assessment"]> | "legacy";

const assessedEvidenceRelations = new Set([
  "supports",
  "weakens",
  "refutes",
  "inconclusive",
  "contradicts",
]);

export function evidenceAssessmentPresentation(
  edge: Edge,
  allNodes: Record<string, GraphNode>,
): EvidenceAssessmentPresentation | null {
  if (
    allNodes[edge.source]?.type !== "evidence" ||
    allNodes[edge.target]?.type !== "hypothesis" ||
    !assessedEvidenceRelations.has(edge.relation)
  ) {
    return null;
  }
  return edge.assessment ?? "legacy";
}

const MODAL_FOCUSABLE_SELECTOR =
  'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])';

export function relationOverlayHost(
  targetDocument: Pick<Document, "body" | "fullscreenElement">,
): Element {
  return targetDocument.fullscreenElement ?? targetDocument.body;
}

export function trapRelationModalTab(
  event: Pick<KeyboardEvent, "key" | "preventDefault" | "shiftKey">,
  overlay: HTMLElement,
  activeElement: Element | null,
): boolean {
  if (event.key !== "Tab") return false;
  const focusable = [...overlay.querySelectorAll<HTMLElement>(MODAL_FOCUSABLE_SELECTOR)].filter(
    (element) =>
      !element.hasAttribute("disabled") &&
      element.getAttribute("aria-hidden") !== "true" &&
      element.tabIndex >= 0,
  );
  if (focusable.length === 0) {
    event.preventDefault();
    overlay.focus();
    return true;
  }

  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  const activeInside = activeElement !== null && overlay.contains(activeElement);
  const destination = event.shiftKey
    ? !activeInside || activeElement === first
      ? last
      : null
    : !activeInside || activeElement === last
      ? first
      : null;
  if (!destination) return false;
  event.preventDefault();
  destination.focus();
  return true;
}

export function makeRelationModalBackgroundInert(overlay: HTMLElement): () => void {
  const previous = new Map<HTMLElement, boolean>();
  let branch: HTMLElement = overlay;
  while (branch.parentElement) {
    const parent = branch.parentElement;
    for (const sibling of parent.children) {
      if (sibling === branch || !("inert" in sibling)) continue;
      const element = sibling as HTMLElement;
      previous.set(element, element.inert);
      element.inert = true;
    }
    branch = parent;
  }
  return () => {
    for (const [element, wasInert] of previous) element.inert = wasInert;
  };
}

/** Build stable one-hop rows while retaining every edge between the same pair of nodes. */
export function groupIncidentRelations(
  focusedNodeId: string,
  allNodes: Record<string, GraphNode>,
  edges: Edge[],
): OneHopRelationGroups {
  const incoming = new Map<string, Edge[]>();
  const outgoing = new Map<string, Edge[]>();

  for (const edge of edges) {
    if (edge.source === focusedNodeId) {
      appendEdge(outgoing, edge.target, edge);
    } else if (edge.target === focusedNodeId) {
      appendEdge(incoming, edge.source, edge);
    }
  }

  const comparePeers = (left: RelationPeerGroup, right: RelationPeerGroup) => {
    const leftTitle = allNodes[left.nodeId]?.title ?? left.nodeId;
    const rightTitle = allNodes[right.nodeId]?.title ?? right.nodeId;
    return leftTitle.localeCompare(rightTitle) || left.nodeId.localeCompare(right.nodeId);
  };
  const finish = (groups: Map<string, Edge[]>) =>
    [...groups.entries()]
      .map(([nodeId, groupedEdges]) => ({
        nodeId,
        edges: [...groupedEdges].sort(
          (left, right) =>
            left.relation.localeCompare(right.relation) || left.id.localeCompare(right.id),
        ),
      }))
      .sort(comparePeers);

  return { incoming: finish(incoming), outgoing: finish(outgoing) };
}

function appendEdge(groups: Map<string, Edge[]>, nodeId: string, edge: Edge) {
  const groupedEdges = groups.get(nodeId);
  if (groupedEdges) groupedEdges.push(edge);
  else groups.set(nodeId, [edge]);
}

export function RelationMap({
  focusedNode,
  allNodes,
  incidentEdges,
  validationMessages,
  onOpenNodeWindow,
}: RelationMapProps) {
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [overlayTarget, setOverlayTarget] = useState<Element | null>(null);
  const [inspectedNodeId, setInspectedNodeId] = useState<string | null>(null);
  const expandButtonRef = useRef<HTMLButtonElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const groups = useMemo(
    () => groupIncidentRelations(focusedNode.id, allNodes, incidentEdges),
    [allNodes, focusedNode.id, incidentEdges],
  );

  const closeOverlay = () => {
    setOverlayOpen(false);
    setInspectedNodeId(null);
  };

  const openOverlay = () => {
    if (typeof document !== "undefined") setOverlayTarget(relationOverlayHost(document));
    setOverlayOpen(true);
  };

  useEffect(() => {
    if (!overlayOpen || typeof document === "undefined") return;
    const syncOverlayTarget = () => setOverlayTarget(relationOverlayHost(document));
    syncOverlayTarget();
    document.addEventListener("fullscreenchange", syncOverlayTarget);
    return () => document.removeEventListener("fullscreenchange", syncOverlayTarget);
  }, [overlayOpen]);

  useEffect(() => {
    if (!overlayOpen) return;
    const trigger = expandButtonRef.current;
    return () => {
      queueMicrotask(() => {
        if (trigger?.isConnected) trigger.focus();
      });
    };
  }, [overlayOpen]);

  useEffect(() => {
    if (!overlayOpen || !overlayTarget) return;
    (closeButtonRef.current ?? overlayRef.current)?.focus();
    const overlayElement = overlayRef.current;
    return overlayElement ? makeRelationModalBackgroundInert(overlayElement) : undefined;
  }, [overlayOpen, overlayTarget]);

  useEffect(() => {
    if (!overlayOpen || typeof window === "undefined") return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeOverlay();
      else if (overlayRef.current) {
        trapRelationModalTab(event, overlayRef.current, document.activeElement);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [overlayOpen]);

  useEffect(() => {
    setInspectedNodeId(null);
  }, [focusedNode.id]);

  const compactMap = (
    <div
      className="relation-map relation-map-compact"
      aria-label={`Relations for ${focusedNode.title}`}
    >
      <div className="relation-map-toolbar">
        <span>
          {incidentEdges.length} relation{incidentEdges.length === 1 ? "" : "s"}
        </span>
        <button
          ref={expandButtonRef}
          type="button"
          className="icon-button relation-map-expand"
          aria-label={`Expand relation map for ${focusedNode.title}`}
          title="Expand relation map"
          onClick={openOverlay}
        >
          <Maximize2 size={15} />
        </button>
      </div>
      <RelationFlow
        focusedNode={focusedNode}
        allNodes={allNodes}
        groups={groups}
        validationMessages={validationMessages}
        onSelectNode={onOpenNodeWindow}
      />
    </div>
  );

  const inspectedNode = inspectedNodeId ? allNodes[inspectedNodeId] : undefined;
  const overlay =
    overlayOpen && overlayTarget
      ? createPortal(
          <div
            ref={overlayRef}
            className="relation-map-overlay"
            role="dialog"
            aria-modal="true"
            aria-label={`Expanded relations for ${focusedNode.title}`}
            tabIndex={-1}
          >
            <header className="relation-map-overlay-header">
              <div>
                <span className="eyebrow">One-hop relation map</span>
                <h2>{focusedNode.title}</h2>
              </div>
              <button
                ref={closeButtonRef}
                type="button"
                className="icon-button"
                aria-label="Close expanded relation map"
                onClick={closeOverlay}
              >
                <X size={18} />
              </button>
            </header>
            <div className={`relation-map-overlay-body${inspectedNode ? " has-inspection" : ""}`}>
              <div className="relation-map relation-map-fullscreen">
                <RelationFlow
                  focusedNode={focusedNode}
                  allNodes={allNodes}
                  groups={groups}
                  validationMessages={validationMessages}
                  onSelectNode={setInspectedNodeId}
                  onSelectFocusedNode={() => setInspectedNodeId(focusedNode.id)}
                />
              </div>
              {inspectedNode && (
                <NodeInspectionCard
                  node={inspectedNode}
                  onOpenNodeWindow={() => onOpenNodeWindow(inspectedNode.id)}
                  onClose={() => setInspectedNodeId(null)}
                />
              )}
            </div>
          </div>,
          overlayTarget,
        )
      : null;

  return (
    <>
      {compactMap}
      {overlay}
    </>
  );
}

interface RelationFlowProps {
  focusedNode: GraphNode;
  allNodes: Record<string, GraphNode>;
  groups: OneHopRelationGroups;
  validationMessages: ValidationMessage[];
  onSelectNode: (nodeId: string) => void;
  onSelectFocusedNode?: () => void;
}

function RelationFlow({
  focusedNode,
  allNodes,
  groups,
  validationMessages,
  onSelectNode,
  onSelectFocusedNode,
}: RelationFlowProps) {
  return (
    <div className="relation-flow">
      {groups.incoming.length > 0 && (
        <div className="relation-flow-level relation-flow-incoming" aria-label="Incoming relations">
          {groups.incoming.map((group) => (
            <RelationPeer
              key={`incoming-${group.nodeId}`}
              group={group}
              node={allNodes[group.nodeId]}
              allNodes={allNodes}
              validationMessages={validationMessages}
              direction="incoming"
              onSelectNode={onSelectNode}
            />
          ))}
        </div>
      )}
      <NodeCard
        node={focusedNode}
        focused
        onSelectNode={onSelectFocusedNode ? () => onSelectFocusedNode() : undefined}
      />
      {groups.outgoing.length > 0 && (
        <div className="relation-flow-level relation-flow-outgoing" aria-label="Outgoing relations">
          {groups.outgoing.map((group) => (
            <RelationPeer
              key={`outgoing-${group.nodeId}`}
              group={group}
              node={allNodes[group.nodeId]}
              allNodes={allNodes}
              validationMessages={validationMessages}
              direction="outgoing"
              onSelectNode={onSelectNode}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface RelationPeerProps {
  group: RelationPeerGroup;
  node?: GraphNode;
  allNodes: Record<string, GraphNode>;
  validationMessages: ValidationMessage[];
  direction: "incoming" | "outgoing";
  onSelectNode: (nodeId: string) => void;
}

function RelationPeer({
  group,
  node,
  allNodes,
  validationMessages,
  direction,
  onSelectNode,
}: RelationPeerProps) {
  const relationLabels = (
    <div className="relation-map-edges">
      {group.edges.map((edge) => {
        const flags = edgeValidationFlags(edge.id, validationMessages);
        const assessment = evidenceAssessmentPresentation(edge, allNodes);
        return (
          <div className={`relation-map-edge${flags.length > 0 ? " has-flag" : ""}`} key={edge.id}>
            <span className="relation-map-edge-arrow" aria-hidden="true">
              ↓
            </span>
            <span className="relation-map-edge-label">{humanize(edge.relation)}</span>
            {assessment === "legacy" && (
              <span className="relation-map-edge-warning">
                <AlertTriangle size={12} />
                Legacy unassessed relation
              </span>
            )}
            {assessment && assessment !== "legacy" && (
              <>
                <span
                  className="relation-map-edge-label"
                  aria-label={`Evidence assessment: ${assessment.relevance} relevance, ${assessment.weight} weight`}
                >
                  Assessment · {humanize(assessment.relevance)} relevance ·{" "}
                  {humanize(assessment.weight)} weight
                </span>
                {assessment.scope && (
                  <span className="relation-map-edge-label">Scope · {assessment.scope}</span>
                )}
                {assessment.qualifications.length > 0 && (
                  <span className="relation-map-edge-warning">
                    <AlertTriangle size={12} />
                    Qualifications · {assessment.qualifications.join(" · ")}
                  </span>
                )}
              </>
            )}
            {flags.map((flag, index) => (
              <span
                className="relation-map-edge-warning"
                role="status"
                key={`${flag.code}-${flag.message}-${index}`}
              >
                <AlertTriangle size={12} />
                {flag.message}
              </span>
            ))}
          </div>
        );
      })}
    </div>
  );

  return (
    <div className="relation-map-peer">
      {direction === "outgoing" && relationLabels}
      <NodeCard node={node} fallbackId={group.nodeId} onSelectNode={onSelectNode} />
      {direction === "incoming" && relationLabels}
    </div>
  );
}

interface NodeCardProps {
  node?: GraphNode;
  fallbackId?: string;
  focused?: boolean;
  onSelectNode?: (nodeId: string) => void;
}

function NodeCard({ node, fallbackId, focused = false, onSelectNode }: NodeCardProps) {
  const content = (
    <>
      {node && <span className="eyebrow">{nodeTypeLabel(node)}</span>}
      <strong>{node?.title ?? fallbackId}</strong>
      {node && <span className={`standing ${node.standing}`}>{node.standing}</span>}
    </>
  );
  if (!node || !onSelectNode) {
    return <div className={`relation-map-node${focused ? " is-focused" : ""}`}>{content}</div>;
  }
  return (
    <button
      type="button"
      className={`relation-map-node${focused ? " is-focused" : ""}`}
      aria-label={`Open ${node.title}`}
      onClick={() => onSelectNode(node.id)}
    >
      {content}
    </button>
  );
}

interface NodeInspectionCardProps {
  node: GraphNode;
  onOpenNodeWindow: () => void;
  onClose: () => void;
}

function NodeInspectionCard({ node, onOpenNodeWindow, onClose }: NodeInspectionCardProps) {
  const presentation = presentNode(node);
  return (
    <aside className="relation-map-inspection" aria-label={`Inspect ${node.title}`}>
      <header>
        <div>
          <span className="eyebrow">{nodeTypeLabel(node)}</span>
          <h3>{node.title}</h3>
          <span className={`standing ${node.standing}`}>{node.standing}</span>
        </div>
        <button
          type="button"
          className="icon-button"
          aria-label="Close node card"
          onClick={onClose}
        >
          <X size={15} />
        </button>
      </header>
      <dl className="relation-map-inspection-fields">
        <div>
          <dt>{presentation.label}</dt>
          <dd>{formatInspectionValue(presentation.value)}</dd>
        </div>
        {presentation.context.map((item) => (
          <div key={item.key}>
            <dt>{item.label}</dt>
            <dd>{formatInspectionValue(item.value)}</dd>
          </div>
        ))}
      </dl>
      <button type="button" className="button compact" onClick={onOpenNodeWindow}>
        <ExternalLink size={14} /> Open node window
      </button>
    </aside>
  );
}

function formatInspectionValue(value: unknown): string {
  if (Array.isArray(value)) return value.map(String).join(" · ");
  return String(value);
}
