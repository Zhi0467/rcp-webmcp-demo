import {
  proposalSemantics,
  type GraphNode,
  type GraphState,
  type OntologyState,
  type Standing,
} from "./types";

export type DraftNodeValue =
  | string
  | number
  | boolean
  | string[]
  | Record<string, string | number | boolean | string[]>
  | null;
export type ProposalDecision = "approved" | "rejected";

export interface DraftNodeChange {
  base_updated_rev: number;
  changes: Record<string, DraftNodeValue>;
  standing?: Standing;
  standing_origin?: "edit" | "judgment";
  cancel_attempt_ids?: string[];
}

export interface HumanDraft {
  version: 1;
  base_revision: number;
  ontology_base_revision?: number;
  nodes: Record<string, DraftNodeChange>;
  removed_node_ids: string[];
  proposals: Record<string, { decision: ProposalDecision; reason?: string }>;
  ontology: OntologyState | null;
  custom_nodes: Record<string, GraphNode>;
}

export interface HumanSyncRequest {
  base_revision: number;
  removed_node_ids: string[];
  nodes: Array<{
    node_id: string;
    base_updated_rev: number;
    changes: Record<string, DraftNodeValue>;
    standing?: Standing;
    cancel_attempt_ids?: string[];
  }>;
  proposals: Array<{ proposal_id: string; decision: ProposalDecision; reason?: string }>;
  ontology: OntologyState | null;
  custom_nodes: GraphNode[];
}

export interface HumanDraftReconciliation {
  draft: HumanDraft;
  discardedProposalIds: string[];
}

export interface ProposalApprovalConflict {
  proposalIds: string[];
  resourceKeys: string[];
}

export function emptyHumanDraft(baseRevision: number): HumanDraft {
  return {
    version: 1,
    base_revision: baseRevision,
    nodes: {},
    removed_node_ids: [],
    proposals: {},
    ontology: null,
    custom_nodes: {},
  };
}

export function normalizeHumanDraft(draft: HumanDraft, graph: GraphState): HumanDraft {
  const next = cloneDraft(draft);
  const removedNodeIds = draft.removed_node_ids.filter((nodeId) => Boolean(graph.nodes[nodeId]));
  const nodes = Object.fromEntries(
    Object.entries(draft.nodes).flatMap(([nodeId, entry]) => {
      const node = graph.nodes[nodeId];
      if (!node) return [];
      if (entry.base_updated_rev !== node.updated_rev) return [[nodeId, entry]];
      const changes = Object.fromEntries(
        Object.entries(entry.changes).filter(([key, value]) => !sameValue(node[key], value)),
      );
      let standing = entry.standing === node.standing ? undefined : entry.standing;
      let standingOrigin = standing ? entry.standing_origin : undefined;
      if (Object.keys(changes).length === 0 && standingOrigin === "edit") {
        standing = undefined;
        standingOrigin = undefined;
      }
      // An attempt already closed canonically no longer needs releasing.
      const openIds = new Set(
        (node.attempts ?? [])
          .filter((attempt) => ["planned", "submitted", "running"].includes(attempt.status))
          .map((attempt) => attempt.id),
      );
      const cancelAttemptIds = (entry.cancel_attempt_ids ?? []).filter((id) => openIds.has(id));
      if (
        Object.keys(changes).length === 0 &&
        standing === undefined &&
        cancelAttemptIds.length === 0
      )
        return [];
      return [
        [
          nodeId,
          {
            base_updated_rev: entry.base_updated_rev,
            changes,
            ...(standing ? { standing } : {}),
            ...(standingOrigin ? { standing_origin: standingOrigin } : {}),
            ...(cancelAttemptIds.length > 0 ? { cancel_attempt_ids: cancelAttemptIds } : {}),
          } satisfies DraftNodeChange,
        ],
      ];
    }),
  );
  const ontology = sameValue(draft.ontology, graph.ontology) ? null : draft.ontology;
  const normalized = {
    ...next,
    base_revision: graph.revision,
    nodes,
    removed_node_ids: removedNodeIds,
    ontology,
    ...(ontology
      ? { ontology_base_revision: draft.ontology_base_revision ?? draft.base_revision }
      : { ontology_base_revision: undefined }),
  };
  return {
    ...normalized,
    proposals: proposalDecisionsWithoutDirectChoices(normalized, graph),
  };
}

export function reconcileHumanDraft(
  draft: HumanDraft,
  graph: GraphState,
): HumanDraftReconciliation {
  const normalized = normalizeHumanDraft(draft, graph);
  const discardedProposalIds = Object.keys(normalized.proposals)
    .filter((proposalId) => graph.proposals[proposalId]?.status !== "pending")
    .sort();
  if (discardedProposalIds.length === 0) return { draft: normalized, discardedProposalIds };
  const discarded = new Set(discardedProposalIds);
  return {
    draft: {
      ...normalized,
      proposals: Object.fromEntries(
        Object.entries(normalized.proposals).filter(([proposalId]) => !discarded.has(proposalId)),
      ),
    },
    discardedProposalIds,
  };
}

export function stageNodeEdit(
  draft: HumanDraft,
  graph: GraphState,
  nodeId: string,
  changes: Record<string, DraftNodeValue>,
  replacedFields: string[] = [],
): HumanDraft {
  const node = graph.nodes[nodeId];
  if (!node || draft.removed_node_ids.includes(nodeId)) return draft;
  const existing = draft.nodes[nodeId];
  const effectiveStanding = existing?.standing ?? node.standing;
  const next = cloneDraft(draft);
  const existingChanges = { ...existing?.changes };
  for (const field of replacedFields) {
    delete existingChanges[field.startsWith("extension_fields.") ? "extension_fields" : field];
  }
  next.nodes[nodeId] = {
    base_updated_rev:
      existing?.base_updated_rev === node.updated_rev
        ? existing.base_updated_rev
        : node.updated_rev,
    changes: { ...existingChanges, ...changes },
    ...(existing?.standing ? { standing: existing.standing } : {}),
    ...(existing?.standing_origin ? { standing_origin: existing.standing_origin } : {}),
  };
  if (Object.keys(changes).length > 0 && effectiveStanding !== "asserted") {
    next.nodes[nodeId].standing = "asserted";
    next.nodes[nodeId].standing_origin = "edit";
  }
  return normalizeHumanDraft(next, graph);
}

export function stageNodeEditStart(
  draft: HumanDraft,
  graph: GraphState,
  nodeId: string,
): HumanDraft {
  const node = graph.nodes[nodeId];
  if (!node || draft.removed_node_ids.includes(nodeId)) return draft;
  const existing = draft.nodes[nodeId];
  const next = cloneDraft(draft);
  next.nodes[nodeId] = {
    base_updated_rev: existing?.base_updated_rev ?? node.updated_rev,
    changes: { ...existing?.changes },
    standing: "asserted",
    standing_origin: "edit",
  };
  return next;
}

export function stageNodeStanding(
  draft: HumanDraft,
  graph: GraphState,
  nodeId: string,
  standing: Standing,
): HumanDraft {
  const node = graph.nodes[nodeId];
  if (!node || draft.removed_node_ids.includes(nodeId)) return draft;
  const existing = draft.nodes[nodeId];
  const next = cloneDraft(draft);
  next.nodes[nodeId] = {
    base_updated_rev:
      existing?.base_updated_rev === node.updated_rev
        ? existing.base_updated_rev
        : node.updated_rev,
    changes: { ...existing?.changes },
    standing,
    standing_origin: "judgment",
  };
  return normalizeHumanDraft(next, graph);
}

export function stageDecisionChoice(
  draft: HumanDraft,
  graph: GraphState,
  nodeId: string,
  selectedOption: string,
): HumanDraft {
  const node = graph.nodes[nodeId];
  const options = draft.nodes[nodeId]?.changes.options ?? node?.options;
  if (
    !node ||
    node.type !== "decision" ||
    node.status === "superseded" ||
    draft.removed_node_ids.includes(nodeId) ||
    !Array.isArray(options) ||
    !options.includes(selectedOption)
  )
    return draft;

  const existing = draft.nodes[nodeId];
  const next = cloneDraft(draft);
  next.nodes[nodeId] = {
    base_updated_rev:
      existing?.base_updated_rev === node.updated_rev
        ? existing.base_updated_rev
        : node.updated_rev,
    changes: {
      ...existing?.changes,
      selected_option: selectedOption,
      status: "decided",
    },
    standing: "accepted",
    standing_origin: "judgment",
    ...(existing?.cancel_attempt_ids?.length
      ? { cancel_attempt_ids: existing.cancel_attempt_ids }
      : {}),
  };
  return normalizeHumanDraft(next, graph);
}

export function proposalTargetsNode(
  proposal: GraphState["proposals"][string],
  nodeId: string,
): boolean {
  return proposal.ops.some((raw) => {
    if (!isRecord(raw) || raw.op !== "update_nodes" || !Array.isArray(raw.nodes)) return false;
    return raw.nodes.some((update) => isRecord(update) && update.id === nodeId);
  });
}

export function stageAttemptRelease(
  draft: HumanDraft,
  graph: GraphState,
  nodeId: string,
  attemptId: string,
): HumanDraft {
  const node = graph.nodes[nodeId];
  if (!node || draft.removed_node_ids.includes(nodeId)) return draft;
  const existing = draft.nodes[nodeId];
  const next = cloneDraft(draft);
  const already = existing?.cancel_attempt_ids ?? [];
  next.nodes[nodeId] = {
    ...existing,
    base_updated_rev:
      existing?.base_updated_rev === node.updated_rev
        ? existing.base_updated_rev
        : node.updated_rev,
    changes: { ...existing?.changes },
    cancel_attempt_ids: already.includes(attemptId) ? already : [...already, attemptId],
  };
  return normalizeHumanDraft(next, graph);
}

export function stageNodeRemoval(
  draft: HumanDraft,
  graph: GraphState,
  nodeId: string,
  experimentLoopActive = false,
): HumanDraft {
  const node = graph.nodes[nodeId];
  if (
    !node ||
    node.standing === "accepted" ||
    experimentLoopActive ||
    draft.nodes[nodeId] ||
    draft.removed_node_ids.includes(nodeId)
  )
    return draft;
  const next = cloneDraft(draft);
  next.removed_node_ids.push(nodeId);
  return next;
}

export function unstageNodeRemoval(draft: HumanDraft, nodeId: string): HumanDraft {
  const next = cloneDraft(draft);
  next.removed_node_ids = next.removed_node_ids.filter((id) => id !== nodeId);
  return next;
}

export function stageProposalDecision(
  draft: HumanDraft,
  graph: GraphState,
  proposalId: string,
  decision: ProposalDecision | null,
): HumanDraft {
  if (decision === "approved" && proposalApprovalConflict(draft, graph, proposalId)) return draft;
  const next = cloneDraft(draft);
  if (decision) next.proposals[proposalId] = { decision };
  else delete next.proposals[proposalId];
  return normalizeHumanDraft(next, graph);
}

export function proposalApprovalConflict(
  draft: HumanDraft | null,
  graph: GraphState,
  proposalId: string,
): ProposalApprovalConflict | null {
  if (!draft) return null;
  const proposal = graph.proposals[proposalId];
  if (!proposal || proposal.status !== "pending") return null;
  const targetResources = new Set(proposalSemantics(proposal).resourceKeys);
  if (targetResources.size === 0) return null;

  const proposalIds: string[] = [];
  const resourceKeys = new Set<string>();
  for (const [stagedProposalId, judgment] of Object.entries(draft.proposals)) {
    if (stagedProposalId === proposalId || judgment.decision !== "approved") continue;
    const stagedProposal = graph.proposals[stagedProposalId];
    if (!stagedProposal || stagedProposal.status !== "pending") continue;
    const overlap = proposalSemantics(stagedProposal).resourceKeys.filter((resourceKey) =>
      targetResources.has(resourceKey),
    );
    if (overlap.length === 0) continue;
    proposalIds.push(stagedProposalId);
    overlap.forEach((resourceKey) => resourceKeys.add(resourceKey));
  }
  return proposalIds.length > 0
    ? { proposalIds: proposalIds.sort(), resourceKeys: [...resourceKeys].sort() }
    : null;
}

export function stageOntology(
  draft: HumanDraft,
  graph: GraphState,
  ontology: OntologyState,
): HumanDraft {
  return normalizeHumanDraft(
    { ...cloneDraft(draft), ontology, ontology_base_revision: graph.revision },
    graph,
  );
}

export function stageCustomNode(draft: HumanDraft, node: GraphNode): HumanDraft {
  const next = cloneDraft(draft);
  next.custom_nodes[node.id] = { ...node, extension_fields: { ...node.extension_fields } };
  return next;
}

export function unstageCustomNode(draft: HumanDraft, nodeId: string): HumanDraft {
  const next = cloneDraft(draft);
  delete next.custom_nodes[nodeId];
  return next;
}

export function applyHumanDraft(graph: GraphState, draft: HumanDraft | null): GraphState {
  if (!draft) return graph;
  const nodes = { ...graph.nodes };
  for (const [nodeId, entry] of Object.entries(draft.nodes)) {
    const node = nodes[nodeId];
    if (!node) continue;
    nodes[nodeId] = {
      ...node,
      ...entry.changes,
      ...(entry.standing ? { standing: entry.standing } : {}),
      draft_touched: true,
    };
  }
  for (const [nodeId, node] of Object.entries(draft.custom_nodes)) {
    nodes[nodeId] = { ...node, draft_touched: true };
  }
  for (const nodeId of draft.removed_node_ids) {
    const node = nodes[nodeId];
    if (node) nodes[nodeId] = { ...node, draft_touched: true };
  }
  return { ...graph, nodes, ontology: draft.ontology ?? graph.ontology };
}

export function humanDraftChangeCount(draft: HumanDraft | null): number {
  if (!draft) return 0;
  const nodeChanges = Object.values(draft.nodes).reduce(
    (count, entry) =>
      count +
      Object.keys(entry.changes).length +
      (entry.standing ? 1 : 0) +
      (entry.cancel_attempt_ids?.length ?? 0),
    0,
  );
  return (
    nodeChanges +
    draft.removed_node_ids.length +
    Object.keys(draft.proposals).length +
    (draft.ontology ? 1 : 0) +
    Object.keys(draft.custom_nodes).length
  );
}

export function humanDraftCommittableCount(draft: HumanDraft | null, graph: GraphState): number {
  if (!draft) return 0;
  const behindNodeIds = new Set(
    Object.entries(draft.nodes).flatMap(([nodeId, entry]) =>
      draftNodeIsBehind(entry, graph.nodes[nodeId]) ? [nodeId] : [],
    ),
  );
  return humanDraftChangeCount({
    ...draft,
    nodes: Object.fromEntries(
      Object.entries(draft.nodes).filter(([nodeId]) => !behindNodeIds.has(nodeId)),
    ),
  });
}

export function humanDraftBehindCount(draft: HumanDraft | null, graph: GraphState): number {
  if (!draft) return 0;
  return Object.entries(draft.nodes).reduce(
    (count, [nodeId, entry]) => count + (draftNodeIsBehind(entry, graph.nodes[nodeId]) ? 1 : 0),
    0,
  );
}

export function humanDraftOntologyIsStale(draft: HumanDraft | null, graph: GraphState): boolean {
  return Boolean(
    draft?.ontology && (draft.ontology_base_revision ?? draft.base_revision) !== graph.revision,
  );
}

export function retainBehindDraftAfterSync(
  draft: HumanDraft,
  previousGraph: GraphState,
  nextGraph: GraphState,
): HumanDraft | null {
  const nodes = Object.fromEntries(
    Object.entries(draft.nodes).filter(([nodeId, entry]) =>
      draftNodeIsBehind(entry, previousGraph.nodes[nodeId]),
    ),
  );
  if (Object.keys(nodes).length === 0) return null;
  return normalizeHumanDraft({ ...emptyHumanDraft(nextGraph.revision), nodes }, nextGraph);
}

export function draftNodeIsBehind(
  entry: DraftNodeChange | undefined,
  node: GraphNode | undefined,
): boolean {
  return Boolean(entry && node && entry.base_updated_rev !== node.updated_rev);
}

export function toHumanSyncRequest(draft: HumanDraft, graph: GraphState): HumanSyncRequest {
  return {
    base_revision: graph.revision,
    removed_node_ids: [...draft.removed_node_ids],
    nodes: Object.entries(draft.nodes).flatMap(([nodeId, entry]) =>
      draftNodeIsBehind(entry, graph.nodes[nodeId])
        ? []
        : [
            {
              node_id: nodeId,
              base_updated_rev: entry.base_updated_rev,
              changes: entry.changes,
              ...(entry.standing ? { standing: entry.standing } : {}),
              ...(entry.cancel_attempt_ids?.length
                ? { cancel_attempt_ids: entry.cancel_attempt_ids }
                : {}),
            },
          ],
    ),
    proposals: Object.entries(draft.proposals).map(([proposalId, entry]) => ({
      proposal_id: proposalId,
      ...entry,
    })),
    ontology: draft.ontology,
    custom_nodes: Object.values(draft.custom_nodes),
  };
}

export function humanDraftStorageKey(projectId: string): string {
  return `rcp:human-draft:${projectId}`;
}

export function humanSyncFailure(error: unknown): {
  text: string;
  revisionConflict: boolean;
} {
  const revisionConflict =
    isRecord(error) &&
    error.status === 409 &&
    typeof error.message === "string" &&
    error.message.includes("graph changed after this draft began");
  return {
    text: revisionConflict
      ? "The project moved again before Sync. Your staged changes were kept and refreshed."
      : error instanceof Error
        ? error.message
        : String(error),
    revisionConflict,
  };
}

export function serializeHumanDraft(draft: HumanDraft): string {
  return JSON.stringify(draft);
}

export function deserializeHumanDraft(value: string | null): HumanDraft | null {
  if (!value) return null;
  try {
    const parsed: unknown = JSON.parse(value);
    if (!isRecord(parsed) || parsed.version !== 1 || !Number.isInteger(parsed.base_revision))
      return null;
    if (!isRecord(parsed.nodes) || !isRecord(parsed.proposals)) return null;
    return {
      version: 1,
      base_revision: parsed.base_revision as number,
      ontology_base_revision: Number.isInteger(parsed.ontology_base_revision)
        ? (parsed.ontology_base_revision as number)
        : undefined,
      nodes: parsed.nodes as unknown as HumanDraft["nodes"],
      removed_node_ids: Array.isArray(parsed.removed_node_ids)
        ? parsed.removed_node_ids.filter((id): id is string => typeof id === "string")
        : [],
      proposals: parsed.proposals as unknown as HumanDraft["proposals"],
      ontology: isRecord(parsed.ontology) ? (parsed.ontology as unknown as OntologyState) : null,
      custom_nodes: isRecord(parsed.custom_nodes)
        ? (parsed.custom_nodes as Record<string, GraphNode>)
        : {},
    };
  } catch {
    return null;
  }
}

function cloneDraft(draft: HumanDraft): HumanDraft {
  return {
    ...draft,
    removed_node_ids: [...draft.removed_node_ids],
    nodes: Object.fromEntries(
      Object.entries(draft.nodes).map(([id, entry]) => [
        id,
        {
          ...entry,
          changes: { ...entry.changes },
        },
      ]),
    ),
    proposals: Object.fromEntries(
      Object.entries(draft.proposals).map(([id, entry]) => [id, { ...entry }]),
    ),
    ontology: draft.ontology
      ? {
          types: draft.ontology.types.map((item) => ({ ...item })),
          fields: draft.ontology.fields.map((item) => ({ ...item })),
          relations: draft.ontology.relations.map((item) => ({
            ...item,
            source_types: [...item.source_types],
            target_types: [...item.target_types],
          })),
        }
      : null,
    custom_nodes: Object.fromEntries(
      Object.entries(draft.custom_nodes).map(([id, node]) => [
        id,
        {
          ...node,
          extension_fields: { ...node.extension_fields },
          source_refs: [...node.source_refs],
        },
      ]),
    ),
  };
}

function sameValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function proposalDecisionsWithoutDirectChoices(
  draft: HumanDraft,
  graph: GraphState,
): HumanDraft["proposals"] {
  const decisionIds = Object.entries(draft.nodes).flatMap(([nodeId, entry]) => {
    const node = graph.nodes[nodeId];
    return node?.type === "decision" &&
      (entry.changes.selected_option !== undefined || entry.changes.status === "decided")
      ? [nodeId]
      : [];
  });
  if (decisionIds.length === 0) return draft.proposals;
  return Object.fromEntries(
    Object.entries(draft.proposals).filter(([proposalId]) => {
      const proposal = graph.proposals[proposalId];
      return (
        !proposal ||
        proposal.status !== "pending" ||
        !decisionIds.some((nodeId) => proposalTargetsNode(proposal, nodeId))
      );
    }),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
