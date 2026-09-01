import { AlertTriangle, ArrowRight, Check, X } from "lucide-react";
import type { GlossaryIndex } from "../glossary";
import { proposalApprovalConflict, type HumanDraft, type ProposalDecision } from "../humanDraft";
import {
  proposalSemantics,
  type GraphNode,
  type GraphState,
  type Proposal,
  type ProposalContentChangeOperation,
  type ProposalMergeOperation,
  type ProposalRemovalOperation,
  type ProposalStatusChangeOperation,
  type ProposalSupersedeOperation,
  type ProposalCreateProtectedRelationOperation,
  type ProposalRemoveProtectedRelationOperation,
} from "../types";
import { GlossaryText } from "./GlossaryText";

interface ProposalJudgmentSectionProps {
  proposals: Proposal[];
  graph: GraphState;
  glossaryIndex: GlossaryIndex;
  draft: HumanDraft | null;
  mutationsDisabled?: boolean;
  onDecision: (proposal: Proposal, decision: ProposalDecision | null) => void;
}

interface AttentionRailProps {
  decisions: GraphNode[];
  blockers: GraphNode[];
  onSelectNode: (nodeId: string) => void;
}

export function ProposalJudgmentSection({
  proposals,
  graph,
  glossaryIndex,
  draft,
  mutationsDisabled = false,
  onDecision,
}: ProposalJudgmentSectionProps) {
  if (proposals.length === 0) return null;

  return (
    <section className="proposal-judgment-section" aria-label="Pending proposals">
      <header className="rail-heading proposal-section-heading">
        <h2>Pending proposals</h2>
        <span className="count-badge">{proposals.length}</span>
      </header>

      {proposals.map((proposal) => {
        const decision = draft?.proposals[proposal.id]?.decision;
        const approved = decision === "approved";
        const rejected = decision === "rejected";
        const proposedAction = proposalAction(proposal, graph);
        const approvalConflict = proposalApprovalConflict(draft, graph, proposal.id);
        const conflictingTitles = approvalConflict?.proposalIds.map(
          (proposalId) => graph.proposals[proposalId]?.title ?? proposalId,
        );
        const approvalConflictText = conflictingTitles?.length
          ? `Approval conflicts with staged approval: ${conflictingTitles.join(", ")}.`
          : undefined;
        return (
          <article className={`proposal-card${decision ? " draft-touched" : ""}`} key={proposal.id}>
            <div className="proposal-topline">
              <span className="eyebrow">
                {decision ? `Pending · staged ${decision}` : "Pending proposal"}
              </span>
              <span className="mono">rev {proposal.base_rev}</span>
            </div>
            <h3>
              <GlossaryText text={proposal.title} glossaryIndex={glossaryIndex} />
            </h3>
            <dl className="card-brief">
              <div>
                <dt>The situation, cold</dt>
                <dd>
                  <GlossaryText
                    text={
                      proposal.card.situation_cold ||
                      "The agent did not supply a cold-readable summary."
                    }
                    glossaryIndex={glossaryIndex}
                  />
                </dd>
              </div>
              <div>
                <dt>Why you, why now</dt>
                <dd>
                  <GlossaryText
                    text={
                      proposal.card.why_human_now || "Human authority is required by the gate set."
                    }
                    glossaryIndex={glossaryIndex}
                  />
                </dd>
              </div>
              <div>
                <dt>If accepted</dt>
                <dd>
                  <GlossaryText
                    text={proposal.card.consequences || "Consequences were not made explicit."}
                    glossaryIndex={glossaryIndex}
                  />
                </dd>
              </div>
              <div>
                <dt>Proposed action</dt>
                <dd>
                  {proposedAction.map((line, index) => (
                    <div key={`${line.label ?? "action"}-${index}`}>
                      {line.label && <strong>{line.label}: </strong>}
                      <GlossaryText text={line.text} glossaryIndex={glossaryIndex} />
                    </div>
                  ))}
                </dd>
              </div>
            </dl>
            <div className="card-actions">
              <button
                className={`button judgment proposal-decision-toggle reject${rejected ? " selected disagree" : ""}`}
                aria-pressed={rejected}
                disabled={mutationsDisabled}
                onClick={() => onDecision(proposal, rejected ? null : "rejected")}
              >
                {rejected ? <Check size={14} /> : <X size={14} />}
                Reject
              </button>
              <button
                className={`button judgment proposal-decision-toggle approve${approved ? " selected agree" : ""}`}
                aria-pressed={approved}
                aria-describedby={approvalConflict ? `proposal-conflict-${proposal.id}` : undefined}
                disabled={mutationsDisabled || (!approved && Boolean(approvalConflict))}
                title={approvalConflictText}
                onClick={() => onDecision(proposal, approved ? null : "approved")}
              >
                <Check size={14} />
                Approve
              </button>
              {approvalConflictText && (
                <span className="eyebrow" id={`proposal-conflict-${proposal.id}`} role="status">
                  {approvalConflictText}
                </span>
              )}
            </div>
          </article>
        );
      })}
    </section>
  );
}

interface ProposalActionLine {
  label?: string;
  text: string;
}

function proposalAction(proposal: Proposal, graph: GraphState): ProposalActionLine[] {
  const fallback = [
    { text: proposal.card.decision_needed || "Review the stored proposal action." },
  ];
  const operation = proposalSemantics(proposal).operation;
  if (!operation) return fallback;

  const action = (() => {
    switch (operation.intent) {
      case "content_change":
        return contentChangeAction(operation, graph);
      case "removal":
        return removalAction(operation, graph);
      case "supersede":
        return supersedeAction(operation, graph);
      case "merge":
        return mergeAction(operation, graph);
      case "protected_relation_change":
        return protectedRelationAction(operation, graph);
      case "status_change":
        return statusChangeAction(operation, graph);
      default:
        return null;
    }
  })();
  return action ?? fallback;
}

function contentChangeAction(
  operation: ProposalContentChangeOperation,
  graph: GraphState,
): ProposalActionLine[] | null {
  const update = operation.nodes[0];
  const node = graph.nodes[update.id];
  const title = nodeTitle(graph, update.id);
  if (!node || !title) return null;

  return [
    { label: "Node", text: title },
    ...Object.entries(update.changes).flatMap(([field, proposed]) => [
      { label: `Current ${compactLabel(field)}`, text: displayValue(node[field]) },
      { label: `Proposed ${compactLabel(field)}`, text: displayValue(proposed) },
    ]),
  ];
}

function removalAction(
  operation: ProposalRemovalOperation,
  graph: GraphState,
): ProposalActionLine[] | null {
  const nodeId = operation.node_ids[0];
  const title = nodeTitle(graph, nodeId);
  if (!title) return null;

  const incidentEdges = Object.values(graph.edges)
    .filter((edge) => edge.source === nodeId || edge.target === nodeId)
    .sort((left, right) => left.id.localeCompare(right.id));
  const relationLines: string[] = [];
  for (const edge of incidentEdges) {
    const text = relationText(graph, edge);
    if (!text) return null;
    relationLines.push(text);
  }

  return [
    { label: "Remove", text: title },
    ...(relationLines.length > 0
      ? relationLines.map((text) => ({ label: "Also removes", text }))
      : [{ label: "Incident relations", text: "None" }]),
  ];
}

function supersedeAction(
  operation: ProposalSupersedeOperation,
  graph: GraphState,
): ProposalActionLine[] | null {
  const item = operation.nodes[0];
  const predecessorTitle = nodeTitle(graph, item.id);
  const successorTitle = nodeTitle(graph, item.superseded_by);
  if (!predecessorTitle || !successorTitle) return null;
  return [
    { label: "Supersede", text: predecessorTitle },
    { label: "With", text: successorTitle },
  ];
}

function mergeAction(
  operation: ProposalMergeOperation,
  graph: GraphState,
): ProposalActionLine[] | null {
  const item = operation.merges[0];
  const duplicateTitle = nodeTitle(graph, item.duplicate);
  const canonicalTitle = nodeTitle(graph, item.canonical);
  if (!duplicateTitle || !canonicalTitle) return null;
  return [
    { label: "Merge", text: duplicateTitle },
    { label: "Into", text: canonicalTitle },
  ];
}

function protectedRelationAction(
  operation: ProposalCreateProtectedRelationOperation | ProposalRemoveProtectedRelationOperation,
  graph: GraphState,
): ProposalActionLine[] | null {
  if (operation.op === "create_edges") {
    const text = relationText(graph, operation.edges[0]);
    return text ? [{ label: "Add relation", text }] : null;
  }
  const edge = graph.edges[operation.edge_ids[0]];
  const text = edge && relationText(graph, edge);
  return text ? [{ label: "Remove relation", text }] : null;
}

function statusChangeAction(
  operation: ProposalStatusChangeOperation,
  graph: GraphState,
): ProposalActionLine[] | null {
  const update = operation.nodes[0];
  const node = graph.nodes[update.id];
  const title = nodeTitle(graph, update.id);
  const currentStatus = typeof node?.status === "string" ? node.status : null;
  const proposedStatus = update.changes.status;
  if (!node || !title || !currentStatus || !proposedStatus) return null;
  return [
    { label: "Node", text: title },
    { label: "Status", text: `${currentStatus} → ${proposedStatus}` },
  ];
}

function relationText(
  graph: GraphState,
  edge: { source: string; target: string; relation: string },
): string | null {
  const sourceTitle = nodeTitle(graph, edge.source);
  const targetTitle = nodeTitle(graph, edge.target);
  if (!sourceTitle || !targetTitle) return null;
  return `${sourceTitle} — ${compactLabel(edge.relation)} → ${targetTitle}`;
}

function nodeTitle(graph: GraphState, nodeId: string): string | null {
  return graph.nodes[nodeId]?.title ?? null;
}

function compactLabel(value: string): string {
  return value.replaceAll("_", " ");
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined) return "Not set";
  if (typeof value === "string") return `“${value}”`;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map(displayValue).join(", ");
  return JSON.stringify(value);
}

export function AttentionRail({ decisions, blockers, onSelectNode }: AttentionRailProps) {
  const total = decisions.length + blockers.length;
  return (
    <aside className="attention-rail" aria-label="Needs your judgment">
      <header className="rail-heading">
        <h2>Needs your judgment</h2>
        <span className="count-badge">{total}</span>
      </header>

      {total === 0 && (
        <div className="quiet-empty">
          <Check size={16} />
          <strong>No other judgment queued</strong>
        </div>
      )}

      {decisions.map((decision) => (
        <button
          className={`attention-item decision${decision.draft_touched ? " draft-touched" : ""}`}
          key={decision.id}
          onClick={() => onSelectNode(decision.id)}
        >
          <strong>{decision.title}</strong>
          <span className={`decision-attention-status ${decision.status}`}>
            {decision.status === "revisit" ? "Revisit" : "Ready"}
          </span>
        </button>
      ))}

      {blockers.map((blocker) => (
        <button
          className={`attention-item blocker${blocker.draft_touched ? " draft-touched" : ""}`}
          key={blocker.id}
          onClick={() => onSelectNode(blocker.id)}
        >
          <AlertTriangle size={15} />
          <strong>{blocker.title}</strong>
          <ArrowRight size={14} />
        </button>
      ))}
    </aside>
  );
}
