import type { GraphAttentionProjection } from "./types";

export type TransitionGraphTarget =
  { kind: "main"; branch_id?: null } | { kind: "branch"; branch_id: string };

export interface TransitionGraphHead {
  target: TransitionGraphTarget;
  revision: number;
  transition_id: string | null;
}

export interface RevisionedTransitionGraph {
  revision: number;
}

/**
 * A local structural view of the backend transition response. Shared API types can extend this
 * interface without making the reducer depend on the rest of ProjectSnapshot.
 */
export interface ProjectTransitionProjection<Graph extends RevisionedTransitionGraph, Control> {
  head: TransitionGraphHead;
  graph: Graph;
  attention: GraphAttentionProjection;
  experiment_control: Control;
  ruleset_tag: string | null;
  transition_id: string | null;
  canonical: boolean;
  base_head?: TransitionGraphHead | null;
}

export interface TransitionTrigger {
  operation: string;
  node_types: string[];
  node_fields: string[];
  relations: string[];
}

export interface TransitionTriggerManifest {
  ruleset_tag: string;
  triggers: TransitionTrigger[];
}

/** Undefined dimensions are unknown, while an empty array means the edit has no such tags. */
export interface StagedTransitionEdit {
  operation?: string | null;
  node_types?: readonly string[];
  node_fields?: readonly string[];
  relations?: readonly string[];
}

export type TransitionPreviewRouting =
  | { route: "local_draft"; reason: "no_manifest_trigger" }
  | {
      route: "backend_preview";
      reason: "missing_manifest" | "missing_ruleset_tag" | "ruleset_mismatch" | "possible_trigger";
    };

export type ProjectTransitionReplaceAction<Graph extends RevisionedTransitionGraph, Control> =
  | {
      kind: "canonical";
      snapshot: ProjectTransitionProjection<Graph, Control>;
    }
  | {
      kind: "preview";
      snapshot: ProjectTransitionProjection<Graph, Control>;
      expected_base_head: TransitionGraphHead;
      manifest_ruleset_tag: string | null;
    };

export type TransitionSnapshotRefusal =
  | "attention_projection_invalid"
  | "graph_head_revision_mismatch"
  | "head_transition_mismatch"
  | "target_mismatch"
  | "revision_regression"
  | "canonicality_mismatch"
  | "preview_base_head_missing"
  | "preview_base_head_mismatch"
  | "preview_head_revision_mismatch"
  | "preview_ruleset_missing"
  | "preview_ruleset_mismatch";

export interface TransitionSyncFence {
  project_id: string;
  request_id: number;
  expected_head: TransitionGraphHead;
  draft_generation: number;
}

export interface ProjectTransitionCoordinatorState {
  active_project_id: string | null;
  canonical_heads: Record<string, TransitionGraphHead>;
  draft_generations: Record<string, number>;
  sync_requests: Record<string, TransitionSyncFence>;
}

export type ProjectTransitionCoordinatorAction =
  | { kind: "activate"; project_id: string | null }
  | { kind: "observe_head"; project_id: string; head: TransitionGraphHead }
  | { kind: "observe_draft_generation"; project_id: string; generation: number }
  | { kind: "sync_started"; fence: TransitionSyncFence }
  | { kind: "sync_finished"; fence: TransitionSyncFence };

export type TransitionSyncCompletionDisposition = "apply" | "reload_active" | "reload_inactive";

export function emptyProjectTransitionCoordinator(): ProjectTransitionCoordinatorState {
  return { active_project_id: null, canonical_heads: {}, draft_generations: {}, sync_requests: {} };
}

export function reduceProjectTransitionCoordinator(
  state: ProjectTransitionCoordinatorState,
  action: ProjectTransitionCoordinatorAction,
): ProjectTransitionCoordinatorState {
  switch (action.kind) {
    case "activate":
      return state.active_project_id === action.project_id
        ? state
        : { ...state, active_project_id: action.project_id };
    case "observe_head":
      return transitionHeadsEqual(state.canonical_heads[action.project_id], action.head)
        ? state
        : {
            ...state,
            canonical_heads: { ...state.canonical_heads, [action.project_id]: action.head },
          };
    case "observe_draft_generation":
      return state.draft_generations[action.project_id] === action.generation
        ? state
        : {
            ...state,
            draft_generations: {
              ...state.draft_generations,
              [action.project_id]: action.generation,
            },
          };
    case "sync_started":
      return {
        ...state,
        sync_requests: { ...state.sync_requests, [action.fence.project_id]: action.fence },
      };
    case "sync_finished": {
      if (state.sync_requests[action.fence.project_id]?.request_id !== action.fence.request_id) {
        return state;
      }
      const syncRequests = { ...state.sync_requests };
      delete syncRequests[action.fence.project_id];
      return { ...state, sync_requests: syncRequests };
    }
  }
}

export function transitionSyncCompletionDisposition(
  state: ProjectTransitionCoordinatorState,
  fence: TransitionSyncFence,
): TransitionSyncCompletionDisposition {
  if (state.active_project_id !== fence.project_id) return "reload_inactive";
  if (state.sync_requests[fence.project_id]?.request_id !== fence.request_id) {
    return "reload_active";
  }
  if ((state.draft_generations[fence.project_id] ?? 0) !== fence.draft_generation) {
    return "reload_active";
  }
  const currentHead = state.canonical_heads[fence.project_id];
  return transitionHeadsEqual(currentHead, fence.expected_head) ? "apply" : "reload_active";
}

export function decodeTransitionTriggerManifest(
  value: unknown,
  expectedRulesetTag: string | null = null,
): TransitionTriggerManifest | null {
  if (!isRecord(value) || !nonEmptyString(value.ruleset_tag) || !Array.isArray(value.triggers)) {
    return null;
  }
  if (expectedRulesetTag && value.ruleset_tag !== expectedRulesetTag) return null;
  const triggers: TransitionTrigger[] = [];
  for (const valueTrigger of value.triggers) {
    if (
      !isRecord(valueTrigger) ||
      !nonEmptyString(valueTrigger.operation) ||
      !stringArray(valueTrigger.node_types) ||
      !stringArray(valueTrigger.node_fields) ||
      !stringArray(valueTrigger.relations)
    ) {
      return null;
    }
    triggers.push({
      operation: valueTrigger.operation,
      node_types: [...valueTrigger.node_types],
      node_fields: [...valueTrigger.node_fields],
      relations: [...valueTrigger.relations],
    });
  }
  return { ruleset_tag: value.ruleset_tag, triggers };
}

export function transitionPreviewRouting(
  manifest: TransitionTriggerManifest | null | undefined,
  projectRulesetTag: string | null | undefined,
  edit: StagedTransitionEdit,
): TransitionPreviewRouting {
  if (!manifest) return { route: "backend_preview", reason: "missing_manifest" };
  if (!projectRulesetTag) return { route: "backend_preview", reason: "missing_ruleset_tag" };
  if (manifest.ruleset_tag !== projectRulesetTag) {
    return { route: "backend_preview", reason: "ruleset_mismatch" };
  }
  if (!edit.operation) return { route: "backend_preview", reason: "possible_trigger" };
  const possible = manifest.triggers
    .filter((trigger) => trigger.operation === edit.operation)
    .some((trigger) => triggerCouldMatch(trigger, edit));
  return possible
    ? { route: "backend_preview", reason: "possible_trigger" }
    : { route: "local_draft", reason: "no_manifest_trigger" };
}

/**
 * Replaces the projection as one value. A rejected async response retains the exact prior object,
 * preventing a new graph from being spliced into old control or head state.
 */
export function reduceProjectTransitionProjection<Graph extends RevisionedTransitionGraph, Control>(
  current: ProjectTransitionProjection<Graph, Control>,
  action: ProjectTransitionReplaceAction<Graph, Control>,
): ProjectTransitionProjection<Graph, Control> {
  return transitionSnapshotRefusal(current, action) ? current : action.snapshot;
}

export function transitionSnapshotRefusal<Graph extends RevisionedTransitionGraph, Control>(
  current: ProjectTransitionProjection<Graph, Control>,
  action: ProjectTransitionReplaceAction<Graph, Control>,
): TransitionSnapshotRefusal | null {
  const incoming = action.snapshot;
  if (!validAttentionProjection(incoming.attention)) return "attention_projection_invalid";
  if (incoming.graph.revision !== incoming.head.revision) return "graph_head_revision_mismatch";
  if (incoming.head.transition_id !== incoming.transition_id) return "head_transition_mismatch";
  if (!sameTarget(current.head.target, incoming.head.target)) return "target_mismatch";

  if (action.kind === "canonical") {
    if (!incoming.canonical) return "canonicality_mismatch";
    if (incoming.head.revision < current.head.revision) return "revision_regression";
    return null;
  }

  if (incoming.canonical) return "canonicality_mismatch";
  if (!incoming.base_head) return "preview_base_head_missing";
  if (!compatibleHead(incoming.base_head, action.expected_base_head)) {
    return "preview_base_head_mismatch";
  }
  if (!sameHead(current.head, action.expected_base_head)) return "preview_base_head_mismatch";
  if (incoming.head.revision !== action.expected_base_head.revision + 1) {
    return "preview_head_revision_mismatch";
  }
  if (!incoming.ruleset_tag) return "preview_ruleset_missing";
  if (
    action.manifest_ruleset_tag &&
    (incoming.ruleset_tag !== action.manifest_ruleset_tag ||
      current.ruleset_tag !== action.manifest_ruleset_tag)
  ) {
    return "preview_ruleset_mismatch";
  }
  return null;
}

function validAttentionProjection(value: unknown): value is GraphAttentionProjection {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const payload = value as Record<string, unknown>;
  const fields = ["pending_proposal_ids", "decisions_awaiting_choice_ids", "open_blocker_ids"];
  if (
    Object.keys(payload).some((key) => !fields.includes(key)) ||
    fields.some((field) => !Object.hasOwn(payload, field))
  ) {
    return false;
  }
  return fields.every((field) => {
    const ids = payload[field];
    return (
      Array.isArray(ids) &&
      ids.every((id) => typeof id === "string" && id.length > 0) &&
      new Set(ids).size === ids.length
    );
  });
}

function triggerCouldMatch(trigger: TransitionTrigger, edit: StagedTransitionEdit): boolean {
  return !(
    dimensionCannotMatch(trigger.node_types, edit.node_types) ||
    dimensionCannotMatch(trigger.node_fields, edit.node_fields) ||
    dimensionCannotMatch(trigger.relations, edit.relations)
  );
}

function dimensionCannotMatch(
  triggerValues: readonly string[],
  editValues: readonly string[] | undefined,
): boolean {
  if (triggerValues.length === 0 || editValues === undefined) return false;
  const edits = new Set(editValues);
  return !triggerValues.some((value) => edits.has(value));
}

function sameHead(left: TransitionGraphHead, right: TransitionGraphHead): boolean {
  return (
    left.revision === right.revision &&
    left.transition_id === right.transition_id &&
    sameTarget(left.target, right.target)
  );
}

function compatibleHead(
  authoritative: TransitionGraphHead,
  expected: TransitionGraphHead,
): boolean {
  return (
    authoritative.revision === expected.revision &&
    sameTarget(authoritative.target, expected.target) &&
    (expected.transition_id === null || authoritative.transition_id === expected.transition_id)
  );
}

export function transitionHeadsEqual(
  left: TransitionGraphHead | null | undefined,
  right: TransitionGraphHead | null | undefined,
): boolean {
  return Boolean(
    left &&
    right &&
    left.revision === right.revision &&
    left.transition_id === right.transition_id &&
    sameTarget(left.target, right.target),
  );
}

function sameTarget(left: TransitionGraphTarget, right: TransitionGraphTarget): boolean {
  return left.kind === right.kind && (left.kind === "main" || left.branch_id === right.branch_id);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}
