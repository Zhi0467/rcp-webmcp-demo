import { Check, Plus, X } from "lucide-react";
import { useMemo, useState } from "react";
import {
  activeCustomTypes,
  activeFieldsForNode,
  baseOntologyTypes,
  makeCustomNode,
} from "../ontologyEditing";
import type {
  ExtensionFieldValue,
  GraphNode,
  OntologyFieldDefinition,
  OntologyState,
} from "../types";

interface Props {
  ontology: OntologyState;
  disabled?: boolean;
  existingNodeIds: Set<string>;
  onStage: (node: GraphNode) => void;
}

export function NewCustomNode({ ontology, disabled = false, existingNodeIds, onStage }: Props) {
  const types = activeCustomTypes(ontology);
  const [open, setOpen] = useState(false);
  const [extensionType, setExtensionType] = useState(types[0]?.name ?? "");
  const [slug, setSlug] = useState("");
  const [title, setTitle] = useState("");
  const [primaryText, setPrimaryText] = useState("");
  const [origin, setOrigin] = useState<GraphNode["origin"] | "">("");
  const [values, setValues] = useState<Record<string, ExtensionFieldValue>>({});

  const definition = types.find((item) => item.name === extensionType) ?? types[0];
  const base = baseOntologyTypes.find((item) => item.name === definition?.base_type);
  const fields = useMemo(
    () => (definition ? activeFieldsForNode(ontology, definition.base_type, definition.name) : []),
    [definition, ontology],
  );
  const normalizedSlug = slug
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  const nodeId = definition && normalizedSlug ? `${definition.name}/${normalizedSlug}` : "";
  const requiredFieldsPresent = fields
    .filter((item) => item.required)
    .every((item) => hasFieldValue(values[item.name], item));
  const valid = Boolean(
    definition &&
    normalizedSlug &&
    title.trim() &&
    primaryText.trim() &&
    requiredFieldsPresent &&
    !existingNodeIds.has(nodeId) &&
    (definition.base_type !== "evidence" || origin),
  );

  const reset = () => {
    setSlug("");
    setTitle("");
    setPrimaryText("");
    setOrigin("");
    setValues({});
  };
  const submit = () => {
    if (!definition || !valid) return;
    const extensionFields = Object.fromEntries(
      fields.flatMap((field) => {
        const value = values[field.name];
        return hasFieldValue(value, field) ? [[field.name, value]] : [];
      }),
    );
    onStage(
      makeCustomNode(
        ontology,
        definition.name,
        normalizedSlug,
        title,
        primaryText,
        origin || undefined,
        extensionFields,
      ),
    );
    reset();
    setOpen(false);
  };

  if (!open) {
    return (
      <button
        className="button secondary compact"
        type="button"
        disabled={disabled || types.length === 0}
        onClick={() => {
          setExtensionType(
            types.find((item) => item.name === extensionType)?.name ?? types[0]?.name ?? "",
          );
          setOpen(true);
        }}
      >
        <Plus size={14} /> New node
      </button>
    );
  }

  return (
    <form
      className="new-custom-node"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <header>
        <strong>New node</strong>
        <button
          className="icon-button"
          type="button"
          aria-label="Close new node"
          onClick={() => setOpen(false)}
        >
          <X size={14} />
        </button>
      </header>
      <div className="new-custom-node-grid">
        <label>
          <span>Type</span>
          <select
            value={definition?.name ?? ""}
            disabled={disabled}
            onChange={(event) => {
              setExtensionType(event.target.value);
              setValues({});
              setOrigin("");
            }}
          >
            {types.map((item) => (
              <option value={item.name} key={item.name}>
                {typeLabel(item.name)}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>ID slug</span>
          <input
            value={slug}
            disabled={disabled}
            onChange={(event) => setSlug(event.target.value)}
          />
        </label>
        <label>
          <span>Title</span>
          <input
            value={title}
            disabled={disabled}
            onChange={(event) => setTitle(event.target.value)}
          />
        </label>
        {base && (
          <label className="wide">
            <span>{base.primaryLabel}</span>
            <textarea
              rows={3}
              value={primaryText}
              disabled={disabled}
              onChange={(event) => setPrimaryText(event.target.value)}
            />
          </label>
        )}
        {definition?.base_type === "evidence" && (
          <label>
            <span>Origin</span>
            <select
              value={origin}
              disabled={disabled}
              onChange={(event) => setOrigin(event.target.value as GraphNode["origin"])}
            >
              <option value="" disabled>
                —
              </option>
              <option value="internal_run">Internal run</option>
              <option value="external_publication">External publication</option>
              <option value="external_instance">External instance</option>
              <option value="analytic">Analytic</option>
            </select>
          </label>
        )}
        {fields.map((field) => (
          <ExtensionFieldInput
            field={field}
            value={values[field.name]}
            disabled={disabled}
            onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))}
            key={`${field.owner_type}.${field.name}`}
          />
        ))}
      </div>
      <footer>
        <span className="mono">{nodeId}</span>
        <button className="button primary compact" type="submit" disabled={disabled || !valid}>
          <Check size={14} /> Stage
        </button>
      </footer>
    </form>
  );
}

function ExtensionFieldInput({
  field,
  value,
  disabled,
  onChange,
}: {
  field: OntologyFieldDefinition;
  value: ExtensionFieldValue | undefined;
  disabled: boolean;
  onChange: (value: ExtensionFieldValue) => void;
}) {
  const label = `${typeLabel(field.name)}${field.required ? " *" : ""}`;
  if (field.kind === "boolean")
    return (
      <label>
        <span>{label}</span>
        <select
          value={value === true ? "true" : value === false ? "false" : ""}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value === "true")}
        >
          <option value="" disabled>
            —
          </option>
          <option value="true">Yes</option>
          <option value="false">No</option>
        </select>
      </label>
    );
  if (field.kind === "text_list")
    return (
      <label className="wide">
        <span>{label}</span>
        <textarea
          rows={3}
          value={Array.isArray(value) ? value.join("\n") : ""}
          disabled={disabled}
          onChange={(event) =>
            onChange(
              event.target.value
                .split(/\r?\n/)
                .map((item) => item.trim())
                .filter(Boolean),
            )
          }
        />
      </label>
    );
  return (
    <label>
      <span>{label}</span>
      <input
        type={field.kind === "number" ? "number" : "text"}
        value={typeof value === "string" || typeof value === "number" ? value : ""}
        disabled={disabled}
        onChange={(event) =>
          onChange(field.kind === "number" ? Number(event.target.value) : event.target.value)
        }
      />
    </label>
  );
}

function hasFieldValue(
  value: ExtensionFieldValue | undefined,
  field: OntologyFieldDefinition,
): boolean {
  if (field.kind === "boolean") return typeof value === "boolean";
  if (field.kind === "number") return typeof value === "number" && Number.isFinite(value);
  if (field.kind === "text_list") return Array.isArray(value) && value.length > 0;
  return typeof value === "string" && value.trim().length > 0;
}

function typeLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}
