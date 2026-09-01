import type { GraphNode, OntologyFieldDefinition, OntologyState } from "./types";

export interface NodeEditField {
  key: string;
  label: string;
  kind: "text" | "multiline" | "list" | "number" | "boolean" | "select";
  options?: { value: string; label: string }[];
  nullable?: boolean;
  min?: number;
  integer?: boolean;
  extensionName?: string;
}

const title: NodeEditField = { key: "title", label: "Title", kind: "text" };

const fieldsByType: Record<GraphNode["type"], NodeEditField[]> = {
  research_question: [
    title,
    { key: "question", label: "Question", kind: "multiline" },
    { key: "motivation", label: "Motivation", kind: "multiline" },
    { key: "scope", label: "Scope", kind: "multiline" },
  ],
  hypothesis: [
    title,
    { key: "statement", label: "Statement", kind: "multiline" },
    { key: "rationale", label: "Rationale", kind: "multiline" },
    { key: "predictions", label: "Predictions", kind: "list" },
    { key: "scope", label: "Scope", kind: "multiline" },
  ],
  decision: [
    title,
    { key: "question", label: "Question", kind: "multiline" },
    { key: "options", label: "Options", kind: "list" },
    {
      key: "status",
      label: "Status",
      kind: "select",
      options: [
        { value: "open", label: "Open" },
        { value: "ready", label: "Ready" },
        { value: "revisit", label: "Revisit" },
      ],
    },
    { key: "rationale", label: "Rationale", kind: "multiline", nullable: true },
    { key: "consequences", label: "Consequences", kind: "list" },
  ],
  experiment: [
    title,
    { key: "objective", label: "Objective", kind: "multiline" },
    { key: "design", label: "Design", kind: "multiline" },
    { key: "expected_outcomes", label: "Expected outcomes", kind: "list" },
    { key: "interpretation_rules", label: "Interpretation rules", kind: "list" },
    { key: "completion_criteria", label: "Completion criteria", kind: "list" },
    {
      key: "invocation_ceiling",
      label: "Invocation ceiling",
      kind: "number",
      min: 1,
      integer: true,
    },
    { key: "current_summary", label: "Current summary", kind: "multiline" },
    { key: "next_action", label: "Next action", kind: "multiline", nullable: true },
  ],
  evidence: [
    title,
    { key: "observation", label: "Observation", kind: "multiline" },
    { key: "interpretation", label: "Interpretation", kind: "multiline" },
  ],
  blocker: [
    title,
    {
      key: "status",
      label: "Status",
      kind: "select",
      options: [
        { value: "open", label: "Open" },
        { value: "resolved", label: "Resolved" },
        { value: "superseded", label: "Superseded" },
      ],
    },
    { key: "description", label: "Description", kind: "multiline" },
    { key: "resolution_condition", label: "Resolution condition", kind: "multiline" },
    { key: "recommended_action", label: "Recommended action", kind: "multiline", nullable: true },
  ],
};

export function editableNodeFields(node: GraphNode, ontology?: OntologyState): NodeEditField[] {
  const ownerTypes = new Set([node.type, ...(node.extension_type ? [node.extension_type] : [])]);
  const baseFields =
    node.type === "decision" &&
    node.status !== "open" &&
    node.status !== "ready" &&
    node.status !== "revisit"
      ? fieldsByType.decision.filter((field) => field.key !== "status")
      : fieldsByType[node.type];
  const extensionFields = ontology
    ? ontology.fields
        .filter((field) => ownerTypes.has(field.owner_type) && !field.deprecated)
        .map(toEditField)
    : [];
  return [...baseFields, ...extensionFields];
}

export function nodeEditDraft(node: GraphNode, ontology?: OntologyState): Record<string, string> {
  return Object.fromEntries(
    editableNodeFields(node, ontology).map((field) => [
      field.key,
      draftValue(
        field,
        field.extensionName ? node.extension_fields[field.extensionName] : node[field.key],
      ),
    ]),
  );
}

export function changedNodeFields(
  node: GraphNode,
  draft: Record<string, string>,
  ontology?: OntologyState,
): Record<
  string,
  string | number | boolean | string[] | Record<string, string | number | boolean | string[]> | null
> {
  const fields = editableNodeFields(node, ontology);
  const baseChanges = Object.fromEntries(
    fields
      .filter((field) => !field.extensionName)
      .flatMap((field) => {
        const next = normalizeField(field, draft[field.key] ?? "");
        const current = normalizeCurrent(field, node[field.key]);
        return equalValues(current, next) ? [] : [[field.key, next]];
      }),
  );
  const extensionDefinitions = fields.filter((field) => field.extensionName);
  const extensionChanged = extensionDefinitions.some((field) => {
    const next = normalizeField(field, draft[field.key] ?? "");
    return !equalValues(normalizeCurrent(field, node.extension_fields[field.extensionName!]), next);
  });
  if (!extensionChanged) return baseChanges;
  const extension_fields = { ...node.extension_fields };
  for (const field of extensionDefinitions) {
    const value = normalizeField(field, draft[field.key] ?? "");
    if (value === null || value === "" || (Array.isArray(value) && value.length === 0)) {
      delete extension_fields[field.extensionName!];
    } else {
      extension_fields[field.extensionName!] = value;
    }
  }
  return { ...baseChanges, extension_fields };
}

function normalizeCurrent(
  field: NodeEditField,
  value: unknown,
): string | number | boolean | string[] | null {
  if (field.kind === "list")
    return arrayValue(value)
      .map((item) => item.trim())
      .filter(Boolean);
  if (field.kind === "number") return typeof value === "number" ? value : null;
  if (field.kind === "boolean") return typeof value === "boolean" ? value : null;
  return normalizeField(field, stringValue(value));
}

function normalizeField(
  field: NodeEditField,
  value: string,
): string | number | boolean | string[] | null {
  if (field.kind === "list") {
    return value
      .split(/\r?\n/)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  if (field.nullable && value.trim() === "") return null;
  if (field.kind === "number") return Number(value);
  if (field.kind === "boolean") return value === "true";
  const normalized = value.trim();
  return field.nullable && normalized === "" ? null : normalized;
}

function arrayValue(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function stringValue(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}

function equalValues(
  left: string | number | boolean | string[] | null,
  right: string | number | boolean | string[] | null,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function draftValue(field: NodeEditField, value: unknown): string {
  if (field.kind === "list") return arrayValue(value).join("\n");
  if (field.kind === "boolean") return typeof value === "boolean" ? String(value) : "";
  return stringValue(value);
}

function toEditField(field: OntologyFieldDefinition): NodeEditField {
  return {
    key: `extension_fields.${field.name}`,
    label: field.name.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase()),
    kind: field.kind === "text_list" ? "list" : field.kind === "text" ? "multiline" : field.kind,
    nullable: !field.required,
    extensionName: field.name,
  };
}
