import { Check, FlaskConical, MessageCircle, Minus, PencilLine, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { experimentGuidanceDetail } from "../experimentGuidance";
import type { GlossaryIndex } from "../glossary";
import { DraggableWindow } from "./DraggableWindow";
import { GlossaryText } from "./GlossaryText";
import {
  changedNodeFields,
  editableNodeFields,
  nodeEditDraft,
  type NodeEditField,
} from "../nodeEditing";
import type { DraftNodeChange, DraftNodeValue } from "../humanDraft";
import { beliefCausePresentation, nodeBeliefTransitions } from "../nodeDetail";
import { humanFieldLabels, humanize, nodeTypeLabel, presentNode } from "../nodePresentation";
import type {
  BeliefTransition,
  Edge,
  ExperimentControlState,
  GraphNode,
  OntologyState,
  ValidationMessage,
} from "../types";
import { RelationMap } from "./RelationMap";

interface Props {
  node: GraphNode;
  edges: Edge[];
  allNodes: Record<string, GraphNode>;
  glossaryIndex: GlossaryIndex;
  beliefTransitions: BeliefTransition[];
  validationMessages: ValidationMessage[];
  ontology: OntologyState;
  sizeStorageKey?: string;
  detailSlot: "original" | "companion";
  focusRequestToken?: string | number;
  mutationsDisabled?: boolean;
  stagedNewNode?: boolean;
  stagedForRemoval?: boolean;
  hasStagedNodeChange?: boolean;
  draftNodeChange?: DraftNodeChange;
  canonicalNode?: GraphNode;
  behind?: boolean;
  canonicalStanding?: GraphNode["standing"];
  experimentControl?: ExperimentControlState | null;
  experimentRunDisabled?: boolean;
  experimentRunBusy?: boolean;
  decisionChoiceStaged?: boolean;
  onUnstage?: () => void;
  onRemove?: () => void;
  onUndoRemoval?: () => void;
  onClose: () => void;
  onDock: () => void;
  onBeginEdit: () => void;
  onStanding: (standing: GraphNode["standing"]) => void;
  onStage: (changes: Record<string, DraftNodeValue>) => void;
  onApplyField?: (changes: Record<string, DraftNodeValue>, fieldKey: string) => void;
  onDecisionChoice?: (selectedOption: string) => void;
  onRunExperiment?: () => void;
  onOpenChat: () => void;
  onOpenRelatedNode: (nodeId: string) => void;
  onSelectNode: (nodeId: string) => void;
}

const ignored = new Set([
  "id",
  "type",
  "title",
  "standing",
  "created_rev",
  "updated_rev",
  "source_refs",
  "attempts",
  "current_summary_stale",
  "next_action_stale",
  "draft_touched",
  "origin",
  "extension_type",
  "extension_fields",
]);

const originLabels: Record<NonNullable<GraphNode["origin"]>, string> = {
  internal_run: "Internal run",
  external_publication: "External publication",
  external_instance: "External instance",
  analytic: "Analytic",
  unknown: "Unknown",
};

export function DetailDrawer({
  node,
  edges,
  allNodes,
  glossaryIndex,
  beliefTransitions,
  validationMessages,
  ontology,
  sizeStorageKey,
  detailSlot,
  focusRequestToken,
  mutationsDisabled = false,
  stagedNewNode = false,
  stagedForRemoval = false,
  hasStagedNodeChange = false,
  draftNodeChange,
  canonicalNode,
  behind = false,
  canonicalStanding = node.standing,
  experimentControl = null,
  experimentRunDisabled = false,
  experimentRunBusy = false,
  decisionChoiceStaged = false,
  onUnstage,
  onRemove,
  onUndoRemoval,
  onClose,
  onDock,
  onBeginEdit,
  onStanding,
  onStage,
  onApplyField,
  onDecisionChoice,
  onRunExperiment,
  onOpenChat,
  onOpenRelatedNode,
  onSelectNode,
}: Props) {
  const [editing, setEditing] = useState(behind);
  const [removalConfirmationOpen, setRemovalConfirmationOpen] = useState(false);
  const [editBase, setEditBase] = useState(node);
  const [draft, setDraft] = useState<Record<string, string>>(() => nodeEditDraft(node, ontology));
  const [referenceDraft, setReferenceDraft] = useState<Record<string, string>>(() =>
    canonicalNode ? nodeEditDraft(canonicalNode, ontology) : {},
  );
  const [referenceFieldKeys, setReferenceFieldKeys] = useState<Set<string>>(() =>
    stagedFieldKeys(draftNodeChange, editableNodeFields(node, ontology)),
  );
  const standingBeforeEdit = useRef<GraphNode["standing"] | null>(null);
  const editFields = useMemo(
    () =>
      editableNodeFields(editBase, ontology).map((field) => {
        if (
          editBase.type !== "experiment" ||
          (field.key !== "current_summary" && field.key !== "next_action")
        ) {
          return field;
        }
        return { ...field, label: experimentGuidanceDetail(editBase, field.key).label };
      }),
    [editBase, ontology],
  );
  const changes = useMemo(
    () => changedNodeFields(editBase, draft, ontology),
    [draft, editBase, ontology],
  );
  const changeCount = Object.keys(changes).length;
  const editErrors = useMemo(
    () =>
      Object.fromEntries(
        editFields.flatMap((field) => {
          const error = nodeEditFieldError(field, draft[field.key] ?? "");
          return error ? [[field.key, error]] : [];
        }),
      ),
    [draft, editFields],
  );
  const editInvalid = Object.keys(editErrors).length > 0;
  const nodeMutationDisabled = mutationsDisabled || stagedForRemoval;
  // Whether a loop is active is the projection's answer, not a second one
  // assembled here from the operational flags underneath it. Composing those
  // flags locally let this panel and Runs disagree about the same Experiment.
  const experimentControlActive = Boolean(experimentControl?.active);
  const experimentPausedAtLimit = Boolean(experimentControl?.paused && !experimentControlActive);

  useEffect(() => {
    if (!editing) {
      setEditBase(node);
      setDraft(nodeEditDraft(node, ontology));
    }
  }, [editing, node, ontology]);

  useEffect(() => {
    if (!behind || !canonicalNode || !draftNodeChange) return;
    setReferenceDraft(nodeEditDraft(canonicalNode, ontology));
    setReferenceFieldKeys(stagedFieldKeys(draftNodeChange, editableNodeFields(node, ontology)));
    setEditing(true);
  }, [behind, canonicalNode, draftNodeChange, node, ontology]);

  useEffect(() => {
    if (nodeMutationDisabled) setEditing(false);
  }, [nodeMutationDisabled]);

  const beginEditing = () => {
    if (nodeMutationDisabled) return;
    standingBeforeEdit.current = node.standing;
    if (node.standing !== "asserted") onBeginEdit();
    setEditBase(node);
    setDraft(nodeEditDraft(node, ontology));
    setEditing(true);
  };
  const cancelEditing = () => {
    if (standingBeforeEdit.current && standingBeforeEdit.current !== "asserted") {
      onStanding(standingBeforeEdit.current);
    }
    standingBeforeEdit.current = null;
    setEditBase(node);
    setDraft(nodeEditDraft(node, ontology));
    setEditing(false);
  };
  const stage = () => {
    if (changeCount === 0 || nodeMutationDisabled || editInvalid) return;
    onStage(changes);
    standingBeforeEdit.current = null;
    setEditing(false);
  };
  const applyReference = (field: NodeEditField) => {
    const nextDraft = {
      ...draft,
      [field.key]: referenceDraft[field.key] ?? "",
    };
    setReferenceDraft((current) => ({
      ...current,
      [field.key]: draft[field.key] ?? "",
    }));
    setDraft(nextDraft);
    if (canonicalNode && onApplyField) {
      onApplyField(changedNodeFields(canonicalNode, nextDraft, ontology), field.key);
    }
  };
  const close = () => {
    if (editing && standingBeforeEdit.current && standingBeforeEdit.current !== "asserted") {
      onStanding(standingBeforeEdit.current);
    }
    standingBeforeEdit.current = null;
    onClose();
  };

  const relations = edges.filter((edge) => edge.source === node.id || edge.target === node.id);
  const removalBlockedReason = stagedForRemoval
    ? null
    : canonicalStanding === "accepted"
      ? "Clear or contest this accepted node and Sync before removing it."
      : experimentControl?.active
        ? "This node cannot be removed while its bounded experiment loop is active."
        : hasStagedNodeChange
          ? "Sync or reset this node's staged changes before removing it."
          : null;
  useEffect(() => {
    setRemovalConfirmationOpen(false);
  }, [
    canonicalStanding,
    experimentControl?.active,
    hasStagedNodeChange,
    mutationsDisabled,
    node.id,
    stagedForRemoval,
  ]);
  const confirmRemoval = () => {
    if (mutationsDisabled || removalBlockedReason || !onRemove) return;
    setRemovalConfirmationOpen(false);
    onRemove();
  };
  const transitions = nodeBeliefTransitions(node.id, beliefTransitions);
  const rawPresentation = presentNode(node);
  const presentation =
    node.type === "experiment"
      ? {
          ...rawPresentation,
          context: rawPresentation.context.map((item) => {
            if (item.key !== "current_summary" && item.key !== "next_action") return item;
            const guidance = experimentGuidanceDetail(node, item.key);
            return { ...item, label: guidance.label };
          }),
        }
      : rawPresentation;
  const presentedKeys = new Set([
    presentation.key,
    ...presentation.context.map((item) => item.key),
  ]);
  const details = Object.entries(node).filter(
    ([key, value]) =>
      !ignored.has(key) &&
      !presentedKeys.has(key) &&
      !(node.type === "decision" && ["options", "selected_option", "status"].includes(key)) &&
      hasValue(value),
  );
  const decisionOptions =
    node.type === "decision" && Array.isArray(node.options)
      ? [...new Set(node.options.filter((option): option is string => typeof option === "string"))]
      : [];
  const selectedDecisionOption =
    node.type === "decision" && typeof node.selected_option === "string"
      ? node.selected_option
      : null;
  const decisionChoiceDisabled =
    nodeMutationDisabled || node.status === "superseded" || !onDecisionChoice;
  const fullscreenTarget = typeof document === "undefined" ? null : document.fullscreenElement;
  const drawerTitleId = `drawer-title-${detailSlot}-${node.id}`;
  const drawer = (
    <DraggableWindow
      className="node-detail-window"
      kind="detail"
      resizable
      sizeStorageKey={sizeStorageKey}
      detailSlot={detailSlot}
      focusRequestToken={focusRequestToken}
    >
      <aside
        className={`detail-drawer node-detail-drawer${node.draft_touched ? " draft-touched" : ""}${behind ? " draft-behind" : ""}`}
        role="dialog"
        aria-modal="false"
        aria-labelledby={drawerTitleId}
      >
        <header data-drag-handle="true">
          <div data-text-selectable="true">
            <span className="eyebrow">{nodeTypeLabel(node)}</span>
            <h2 id={drawerTitleId}>
              <GlossaryText text={node.title} glossaryIndex={glossaryIndex} />
            </h2>
            <div className="node-meta">
              <span className="mono">{node.id}</span>
              <span className={`standing ${node.standing}`}>
                {node.standing}
                {canonicalStanding !== node.standing && " · staged"}
              </span>
              {behind && <span className="node-draft-behind">behind</span>}
              {node.type === "evidence" && node.origin && (
                <span className="node-origin">{originLabels[node.origin]}</span>
              )}
            </div>
          </div>
          <div className="window-actions">
            <button
              className="icon-button"
              aria-label="Dock node window"
              title="Dock node window"
              onClick={onDock}
            >
              <Minus size={18} />
            </button>
            <button className="icon-button" aria-label="Close detail" onClick={close}>
              <X size={18} />
            </button>
          </div>
        </header>

        <div className={`drawer-content${editing ? " editing" : ""}`}>
          {stagedForRemoval && (
            <section className="node-removal-staged" role="status">
              <Trash2 size={16} />
              <span>
                <strong>Removal staged.</strong> Sync will remove this node and {relations.length}{" "}
                connected relation{relations.length === 1 ? "" : "s"}.
              </span>
              <button
                className="button compact"
                type="button"
                disabled={mutationsDisabled || !onUndoRemoval}
                onClick={onUndoRemoval}
              >
                Undo
              </button>
            </section>
          )}
          {editing ? (
            <form
              className="node-edit-form"
              onSubmit={(event) => {
                event.preventDefault();
                stage();
              }}
            >
              {editFields.map((field) => (
                <label className="node-edit-field" key={field.key}>
                  <span>
                    {field.label}
                    {field.kind === "list" ? " · one item per line" : ""}
                    {field.nullable ? <span className="node-field-optional">Optional</span> : null}
                  </span>
                  {field.kind === "text" || field.kind === "number" ? (
                    <>
                      <input
                        type={field.kind === "number" ? "number" : "text"}
                        min={field.min}
                        step={field.kind === "number" ? (field.integer ? 1 : "any") : undefined}
                        aria-invalid={Boolean(editErrors[field.key])}
                        autoFocus={field.key === "title"}
                        value={draft[field.key] ?? ""}
                        onChange={(event) =>
                          setDraft((current) => ({ ...current, [field.key]: event.target.value }))
                        }
                      />
                      {editErrors[field.key] && (
                        <small className="node-edit-error" role="alert">
                          {editErrors[field.key]}
                        </small>
                      )}
                    </>
                  ) : field.kind === "select" ? (
                    <select
                      value={draft[field.key] ?? ""}
                      onChange={(event) =>
                        setDraft((current) => ({ ...current, [field.key]: event.target.value }))
                      }
                    >
                      {field.options?.map((option) => (
                        <option value={option.value} key={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  ) : field.kind === "boolean" ? (
                    <select
                      value={draft[field.key] ?? ""}
                      onChange={(event) =>
                        setDraft((current) => ({ ...current, [field.key]: event.target.value }))
                      }
                    >
                      {field.nullable && <option value="">—</option>}
                      <option value="true">Yes</option>
                      <option value="false">No</option>
                    </select>
                  ) : (
                    <textarea
                      rows={field.kind === "list" ? 4 : 5}
                      value={draft[field.key] ?? ""}
                      onChange={(event) =>
                        setDraft((current) => ({ ...current, [field.key]: event.target.value }))
                      }
                    />
                  )}
                  {referenceFieldKeys.has(field.key) && (
                    <span className="node-edit-incoming" role="status">
                      <span className="node-edit-incoming-heading">
                        <span>Incoming</span>
                        <button
                          className="button compact"
                          type="button"
                          onClick={() => applyReference(field)}
                        >
                          Apply
                        </button>
                      </span>
                      <span className="node-edit-incoming-value">
                        {referenceDraft[field.key] || "—"}
                      </span>
                    </span>
                  )}
                </label>
              ))}
            </form>
          ) : (
            <>
              {node.type === "decision" ? (
                <section className="decision-choice-section">
                  <div className="decision-choice-heading">
                    <span className="eyebrow">Decision</span>
                    <span className={`decision-choice-status ${node.status ?? "open"}`}>
                      {humanize(node.status ?? "open")}
                      {decisionChoiceStaged ? " · staged" : ""}
                    </span>
                  </div>
                  <fieldset disabled={decisionChoiceDisabled}>
                    <legend id={`decision-question-${node.id}`}>
                      {formatValue(presentation.value, glossaryIndex)}
                    </legend>
                    <div className="decision-choice-options">
                      {decisionOptions.map((option) => {
                        const selected = option === selectedDecisionOption;
                        return (
                          <label
                            className={`decision-choice-option${selected ? " selected" : ""}${selected && decisionChoiceStaged ? " staged" : ""}`}
                            key={option}
                          >
                            <input
                              type="radio"
                              name={`decision-choice-${node.id}`}
                              value={option}
                              checked={selected}
                              readOnly
                              // Not onChange: a click on the already-checked
                              // option fires no change event, and that is
                              // exactly the click that decides a Decision
                              // carrying an option it never moved to decided.
                              onClick={() => onDecisionChoice?.(option)}
                            />
                            <span className="decision-choice-option-label">
                              <GlossaryText text={option} glossaryIndex={glossaryIndex} />
                            </span>
                            {selected && (
                              <span className="decision-choice-option-state">
                                <Check size={13} />
                                {decisionChoiceStaged ? "Staged selection" : "Selected"}
                              </span>
                            )}
                          </label>
                        );
                      })}
                    </div>
                  </fieldset>
                </section>
              ) : (
                <section className="node-lead">
                  <span className="eyebrow">{presentation.label}</span>
                  <p>{formatValue(presentation.value, glossaryIndex)}</p>
                </section>
              )}

              {node.type === "experiment" && experimentControl && onRunExperiment && (
                <section
                  className={`experiment-control${experimentControlActive ? " active" : ""}${experimentPausedAtLimit ? " paused" : ""}`}
                >
                  <div className="experiment-control-heading">
                    <div>
                      {experimentControl.episode_id && (
                        <>
                          <span className="eyebrow">Episode invocations</span>
                          <strong>
                            {experimentControl.invocations_used} /{" "}
                            {experimentControl.invocation_ceiling}
                          </strong>
                          <span className="experiment-invocations-remaining">
                            {experimentControl.invocations_remaining} remaining
                          </span>
                        </>
                      )}
                      <span className="eyebrow">Next episode limit</span>
                      <strong>{node.invocation_ceiling}</strong>
                    </div>
                    {experimentControlActive && (
                      <span className="experiment-loop-marker">Active loop</span>
                    )}
                    {experimentPausedAtLimit && (
                      <span className="experiment-loop-marker paused">Paused at limit</span>
                    )}
                    <button
                      className="button primary compact experiment-run-button"
                      type="button"
                      disabled={
                        nodeMutationDisabled ||
                        experimentRunDisabled ||
                        experimentRunBusy ||
                        !experimentControl.ready
                      }
                      onClick={onRunExperiment}
                    >
                      <FlaskConical size={13} />{" "}
                      {experimentRunBusy
                        ? "Starting"
                        : experimentControl.episode_id
                          ? "Start new episode"
                          : "Start episode"}
                    </button>
                  </div>
                  {experimentControl.reasons.length > 0 && (
                    <ul className="experiment-gate-reasons" aria-label="Run requirements">
                      {experimentControl.reasons.map((reason) => (
                        <li key={reason}>{reason}</li>
                      ))}
                    </ul>
                  )}
                  {experimentControl.decision_drift.length > 0 && (
                    <ul className="experiment-decision-drift" aria-label="Decision drift">
                      {experimentControl.decision_drift.map((drift) => (
                        <li key={drift.decision_id}>
                          {`${drift.decision_id} moved to ${drift.current_option ?? drift.current_status ?? "an unavailable state"} after this episode was pinned to ${drift.pinned_option}.`}
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              )}

              {presentation.context.length > 0 && (
                <section className="node-context">
                  <h3>Context</h3>
                  <dl className="context-list">
                    {presentation.context.map((item) => (
                      <div key={item.key}>
                        <dt>{item.label}</dt>
                        <dd>{formatValue(item.value, glossaryIndex)}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              )}

              {transitions.length > 0 && (
                <section className="belief-history">
                  <h3>Status history</h3>
                  <ol>
                    {transitions.map((transition) => {
                      const cause = beliefCausePresentation(transition, edges, allNodes);
                      return (
                        <li
                          key={`${transition.revision}-${transition.from_status}-${transition.to_status}`}
                        >
                          <span className="belief-transition">
                            <strong>{transition.from_status}</strong>
                            <span>→</span>
                            <strong>{transition.to_status}</strong>
                          </span>
                          <span className="mono">rev {transition.revision}</span>
                          {cause.nodeId ? (
                            <button type="button" onClick={() => onSelectNode(cause.nodeId!)}>
                              {cause.label}
                            </button>
                          ) : (
                            <span>{cause.label}</span>
                          )}
                        </li>
                      );
                    })}
                  </ol>
                </section>
              )}

              {Object.keys(node.extension_fields).length > 0 && (
                <section>
                  <h3>Extension fields</h3>
                  <dl className="detail-list">
                    {Object.entries(node.extension_fields).map(([key, value]) => (
                      <div key={key}>
                        <dt>{humanize(key)}</dt>
                        <dd>{formatValue(value, glossaryIndex)}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              )}

              <section>
                <div className="relations-control-heading">
                  <strong>Relations</strong>
                  <small>{relations.length}</small>
                </div>
                <RelationMap
                  focusedNode={node}
                  allNodes={allNodes}
                  incidentEdges={relations.filter((edge) => typeof edge.relation === "string")}
                  validationMessages={validationMessages}
                  onOpenNodeWindow={onOpenRelatedNode}
                />
              </section>

              {details.length > 0 && (
                <section>
                  <h3>Record details</h3>
                  <dl className="detail-list">
                    {details.map(([key, value]) => (
                      <div key={key}>
                        <dt>{humanFieldLabels[key] ?? humanize(key)}</dt>
                        <dd>{formatValue(value, glossaryIndex)}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              )}

              <section>
                <h3>Conversation evidence</h3>
                {node.source_refs.length === 0 ? (
                  <p className="muted">No source excerpts attached.</p>
                ) : (
                  node.source_refs.map((source) => (
                    <blockquote key={`${source.session_id}-${source.record_uuid}`}>
                      <p>{source.excerpt}</p>
                      <footer className="mono">
                        {source.source} · {source.truth_repository} ·{" "}
                        {new Date(source.timestamp).toLocaleString()}
                      </footer>
                    </blockquote>
                  ))
                )}
              </section>
            </>
          )}
        </div>

        <footer className="drawer-actions">
          {editing ? (
            <>
              <span className="node-edit-status">
                {behind && changeCount === 0
                  ? "Behind"
                  : changeCount > 0
                    ? `${changeCount} field${changeCount === 1 ? "" : "s"}`
                    : "No changes"}
              </span>
              <div>
                <button className="button ghost" type="button" onClick={cancelEditing}>
                  Cancel
                </button>
                <button
                  className="button primary compact"
                  type="button"
                  disabled={nodeMutationDisabled || changeCount === 0 || editInvalid}
                  onClick={stage}
                >
                  <Check size={14} /> Done
                </button>
              </div>
            </>
          ) : stagedNewNode ? (
            <>
              <span className="node-edit-status">Staged node</span>
              <button
                className="button secondary compact"
                type="button"
                disabled={mutationsDisabled}
                onClick={onUnstage}
              >
                <Trash2 size={14} /> Remove
              </button>
            </>
          ) : (
            <>
              <button className="button ghost" onClick={onOpenChat}>
                <MessageCircle size={15} /> Ask about this node
              </button>
              <div className="node-detail-actions">
                <div className="node-judgment-actions">
                  <button
                    className="button secondary"
                    disabled={nodeMutationDisabled}
                    onClick={beginEditing}
                  >
                    <PencilLine size={14} /> Edit node
                  </button>
                  <button
                    className={`button judgment node-standing-toggle contest${node.standing === "contested" ? " selected disagree" : ""}`}
                    aria-pressed={node.standing === "contested"}
                    disabled={nodeMutationDisabled}
                    onClick={() =>
                      onStanding(node.standing === "contested" ? "asserted" : "contested")
                    }
                  >
                    {node.standing === "contested" ? <Check size={14} /> : <X size={14} />}
                    Contest
                  </button>
                  <button
                    className={`button judgment node-standing-toggle agree${node.standing === "accepted" ? " selected agree" : ""}`}
                    aria-pressed={node.standing === "accepted"}
                    disabled={nodeMutationDisabled}
                    onClick={() =>
                      onStanding(node.standing === "accepted" ? "asserted" : "accepted")
                    }
                  >
                    <Check size={14} />
                    Agree
                  </button>
                </div>
                <div className="node-removal-action">
                  <button
                    className="button danger"
                    type="button"
                    hidden={removalConfirmationOpen}
                    disabled={
                      mutationsDisabled ||
                      stagedForRemoval ||
                      Boolean(removalBlockedReason) ||
                      !onRemove
                    }
                    title={removalBlockedReason ?? undefined}
                    onClick={() => setRemovalConfirmationOpen(true)}
                  >
                    <Trash2 size={14} /> {stagedForRemoval ? "Removal staged" : "Remove node…"}
                  </button>
                  <div
                    className="node-removal-confirmation"
                    role="alert"
                    hidden={!removalConfirmationOpen}
                  >
                    <span>
                      Remove <strong>“{node.title}”</strong>? Sync will remove it and{" "}
                      {relations.length} connected relation{relations.length === 1 ? "" : "s"}.
                    </span>
                    <div>
                      <button
                        className="button compact"
                        type="button"
                        onClick={() => setRemovalConfirmationOpen(false)}
                      >
                        Cancel
                      </button>
                      <button
                        className="button danger compact"
                        type="button"
                        onClick={confirmRemoval}
                      >
                        Confirm remove
                      </button>
                    </div>
                  </div>
                  {removalBlockedReason && <small>{removalBlockedReason}</small>}
                </div>
              </div>
            </>
          )}
        </footer>
      </aside>
    </DraggableWindow>
  );
  return fullscreenTarget ? createPortal(drawer, fullscreenTarget) : drawer;
}

function hasValue(value: unknown): boolean {
  return (
    value !== null &&
    value !== undefined &&
    value !== "" &&
    !(Array.isArray(value) && value.length === 0)
  );
}

function stagedFieldKeys(entry: DraftNodeChange | undefined, fields: NodeEditField[]): Set<string> {
  if (!entry) return new Set();
  return new Set(
    fields.flatMap((field) => {
      if (!field.extensionName) return field.key in entry.changes ? [field.key] : [];
      const extensions = entry.changes.extension_fields;
      return extensions &&
        typeof extensions === "object" &&
        !Array.isArray(extensions) &&
        field.extensionName in extensions
        ? [field.key]
        : [];
    }),
  );
}

function nodeEditFieldError(field: NodeEditField, value: string): string | null {
  if (field.kind !== "number") return null;
  const number = Number(value);
  if (!Number.isFinite(number)) return "Enter a number.";
  if (field.integer && !Number.isInteger(number)) return "Enter a whole number.";
  if (field.min !== undefined && number < field.min) {
    return field.integer
      ? `Enter a whole number of at least ${field.min}.`
      : `Enter a number of at least ${field.min}.`;
  }
  return null;
}

function formatValue(value: unknown, glossaryIndex: GlossaryIndex): React.ReactNode {
  if (Array.isArray(value))
    return (
      <ul>
        {value.map((item, index) => (
          <li key={index}>{formatValue(item, glossaryIndex)}</li>
        ))}
      </ul>
    );
  if (typeof value === "object" && value !== null)
    return <pre>{JSON.stringify(value, null, 2)}</pre>;
  return <GlossaryText text={String(value)} glossaryIndex={glossaryIndex} />;
}
