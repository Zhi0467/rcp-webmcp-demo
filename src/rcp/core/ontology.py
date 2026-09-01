from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from rcp.core.models import (
    ALL_NODE_TYPES,
    RELATION_SPEC,
    BaseNode,
    Blocker,
    Decision,
    Evidence,
    Experiment,
    GraphState,
    Hypothesis,
    OntologyFieldDefinition,
    OntologyRelationDefinition,
    OntologyState,
    ProjectNode,
    ResearchQuestion,
)
from rcp.core.operations import SetOntologyOperation

if TYPE_CHECKING:
    from rcp.core.validation.report import ValidationReport

BASE_TYPE_LAYERS = {
    "research_question": "epistemic",
    "hypothesis": "epistemic",
    "evidence": "epistemic",
    "decision": "action",
    "experiment": "action",
    "blocker": "action",
}
BASE_FIELD_NAMES = frozenset(
    field
    for model in (
        BaseNode,
        ResearchQuestion,
        Hypothesis,
        Decision,
        Experiment,
        Evidence,
        Blocker,
    )
    for field in model.model_fields
)


def parse_ontology_operation(
    operation: SetOntologyOperation,
    report: ValidationReport,
    revision: int | None,
) -> OntologyState:
    ontology = operation.ontology
    validate_ontology_structure(ontology, report, revision)
    return ontology


def validate_ontology_structure(
    ontology: OntologyState, report: ValidationReport, revision: int | None
) -> None:
    custom_types = {item.name: item for item in ontology.types}
    known_types = ALL_NODE_TYPES | custom_types.keys()

    for item in ontology.types:
        if item.name in ALL_NODE_TYPES:
            report.reject(
                "base-ontology-collision",
                f"Custom type {item.name!r} collides with an immutable base type.",
                revision,
            )
        expected_layer = BASE_TYPE_LAYERS[item.base_type]
        if item.layer != expected_layer:
            report.reject(
                "invalid-base-mapping",
                f"Custom type {item.name!r} specializes {item.base_type!r} and must use "
                f"its {expected_layer!r} layer.",
                revision,
            )

    for item in ontology.fields:
        if item.owner_type not in known_types:
            report.reject(
                "unknown-ontology-owner",
                f"Field {item.owner_type}.{item.name} has no known owner type.",
                revision,
            )
        if item.name in BASE_FIELD_NAMES:
            report.reject(
                "base-ontology-collision",
                f"Custom field {item.owner_type}.{item.name} collides with an immutable base field.",
                revision,
            )

    fields_by_owner: dict[str, set[str]] = {}
    for item in ontology.fields:
        fields_by_owner.setdefault(item.owner_type, set()).add(item.name)
    for item in ontology.types:
        inherited_collisions = fields_by_owner.get(item.name, set()) & fields_by_owner.get(
            item.base_type, set()
        )
        if inherited_collisions:
            report.reject(
                "ontology-field-collision",
                f"Custom type {item.name!r} redeclares inherited project fields: "
                f"{sorted(inherited_collisions)}.",
                revision,
            )

    for item in ontology.relations:
        if item.name in RELATION_SPEC:
            report.reject(
                "base-ontology-collision",
                f"Custom relation {item.name!r} collides with an immutable base relation.",
                revision,
            )
        if len(item.source_types) != len(set(item.source_types)) or len(item.target_types) != len(
            set(item.target_types)
        ):
            report.reject(
                "duplicate-relation-endpoint-type",
                f"Custom relation {item.name!r} repeats an endpoint type.",
                revision,
            )
        unknown = (set(item.source_types) | set(item.target_types)) - set(known_types)
        if unknown:
            report.reject(
                "unknown-relation-type",
                f"Custom relation {item.name!r} names unknown endpoint types: {sorted(unknown)}.",
                revision,
            )


def validate_ontology_change(
    state: GraphState,
    desired: OntologyState,
    report: ValidationReport,
    revision: int | None,
) -> None:
    old_types = {item.name: item for item in state.ontology.types}
    new_types = {item.name: item for item in desired.types}
    old_fields = {(item.owner_type, item.name): item for item in state.ontology.fields}
    new_fields = {(item.owner_type, item.name): item for item in desired.fields}
    old_relations = {item.name: item for item in state.ontology.relations}
    new_relations = {item.name: item for item in desired.relations}

    _require_deprecation_before_removal("type", old_types, new_types, report, revision)
    _require_deprecation_before_removal("field", old_fields, new_fields, report, revision)
    _require_deprecation_before_removal("relation", old_relations, new_relations, report, revision)

    for name, previous in old_types.items():
        current = new_types.get(name)
        if current and (current.base_type, current.layer) != (previous.base_type, previous.layer):
            blockers = sorted(
                node.id for node in state.nodes.values() if node.extension_type == name
            )
            report.reject(
                "ontology-type-remap",
                f"Custom type {name!r} cannot change its base mapping; implicated nodes: "
                f"{blockers or ['none']}.",
                revision,
                related_node_ids=blockers,
            )

    for key, current in new_fields.items():
        previous = old_fields.get(key)
        applicable = [node for node in state.nodes.values() if field_applies(node, current)]
        if current.required and (previous is None or not previous.required):
            blockers = sorted(
                node.id for node in applicable if current.name not in node.extension_fields
            )
            if blockers:
                report.reject(
                    "required-field-breaks-existing-nodes",
                    f"Field {current.owner_type}.{current.name} cannot become required; missing on "
                    f"nodes: {blockers}.",
                    revision,
                    related_node_ids=blockers,
                )
        if previous is not None and previous.kind != current.kind:
            blockers = sorted(
                node.id
                for node in applicable
                if current.name in node.extension_fields
                and not field_value_matches(current, node.extension_fields[current.name])
            )
            if blockers:
                report.reject(
                    "field-kind-breaks-existing-nodes",
                    f"Field {current.owner_type}.{current.name} cannot change to {current.kind}; "
                    f"implicated nodes: {blockers}.",
                    revision,
                    related_node_ids=blockers,
                )

    for name, previous in old_relations.items():
        current = new_relations.get(name)
        if current is None:
            continue
        if previous.layer != current.layer:
            blockers = [edge for edge in state.edges.values() if edge.relation == name]
            if blockers:
                node_ids = sorted(
                    {node for edge in blockers for node in (edge.source, edge.target)}
                )
                report.reject(
                    "relation-layer-breaks-existing-edges",
                    f"Relation {name!r} cannot change layer while used by edges: "
                    f"{sorted(edge.id for edge in blockers)}.",
                    revision,
                    related_node_ids=node_ids,
                    related_edge_ids=sorted(edge.id for edge in blockers),
                )
        blockers = [
            edge
            for edge in state.edges.values()
            if edge.relation == name
            and not edge_matches_relation(state, edge.source, edge.target, current)
        ]
        if blockers:
            node_ids = sorted({node for edge in blockers for node in (edge.source, edge.target)})
            report.reject(
                "relation-narrowing-breaks-existing-edges",
                f"Relation {name!r} cannot exclude existing edges "
                f"{sorted(edge.id for edge in blockers)}; implicated nodes: {node_ids}.",
                revision,
                related_node_ids=node_ids,
                related_edge_ids=sorted(edge.id for edge in blockers),
            )


def validate_new_node_extensions(
    state: GraphState,
    raw: dict[str, Any],
    report: ValidationReport,
    revision: int | None,
    *,
    authoring: bool,
    agent_authored: bool,
) -> None:
    base_type = raw.get("type")
    extension_type = raw.get("extension_type")
    fields = raw.get("extension_fields", {})
    type_definition = ontology_type(state.ontology, extension_type)

    if extension_type is not None:
        if not authoring and type_definition is None:
            report.reject(
                "unknown-extension-type",
                f"Node {raw.get('id')!r} uses unknown custom type {extension_type!r}.",
                revision,
            )
        elif (
            not authoring and type_definition is not None and type_definition.base_type != base_type
        ):
            report.reject(
                "invalid-extension-base",
                f"Custom type {extension_type!r} maps to {type_definition.base_type!r}, not "
                f"{base_type!r}.",
                revision,
            )
        elif authoring and type_definition is not None and type_definition.deprecated:
            report.reject(
                "deprecated-extension-type",
                f"Custom type {extension_type!r} is deprecated and cannot author new nodes.",
                revision,
            )

    if not isinstance(fields, dict):
        return
    definitions = applicable_field_definitions(state.ontology, base_type, extension_type)
    _validate_extension_fields(
        raw.get("id"),
        fields,
        definitions,
        report,
        revision,
        authoring=authoring,
        agent_authored=agent_authored,
        structural=not authoring,
    )


def validate_updated_extension_fields(
    state: GraphState,
    node: ProjectNode,
    changes: dict[str, Any],
    report: ValidationReport,
    revision: int | None,
    *,
    authoring: bool,
    agent_authored: bool,
) -> None:
    if "extension_fields" not in changes:
        return
    fields = changes.get("extension_fields")
    if not isinstance(fields, dict):
        return
    definitions = applicable_field_definitions(state.ontology, node.type, node.extension_type)
    known = {item.name: item for item in definitions}
    changed = {
        name
        for name in set(node.extension_fields) | set(fields)
        if node.extension_fields.get(name, object()) != fields.get(name, object())
    }
    for name in sorted(changed):
        definition = known.get(name)
        if definition is None and not authoring:
            report.reject(
                "unknown-extension-field",
                f"Node {node.id!r} cannot change undefined extension field {name!r}.",
                revision,
                related_node_ids=[node.id],
            )
            continue
        if name in fields:
            if definition is None:
                continue
            _validate_one_field(
                node.id,
                definition,
                fields[name],
                report,
                revision,
                authoring=authoring,
                agent_authored=agent_authored,
                check_kind=not authoring,
            )
    missing = (
        sorted(item.name for item in definitions if item.required and item.name not in fields)
        if not authoring
        else []
    )
    if missing:
        report.reject(
            "missing-required-extension-field",
            f"Node {node.id!r} is missing required extension fields: {missing}.",
            revision,
            related_node_ids=[node.id],
        )


def custom_relation(ontology: OntologyState, name: Any) -> OntologyRelationDefinition | None:
    if not isinstance(name, str):
        return None
    return next((item for item in ontology.relations if item.name == name), None)


def edge_layer(
    state: GraphState,
    source_id: Any,
    target_id: Any,
    declared: str,
) -> str:
    """Resolve the layer of one edge from the nodes it actually connects.

    A layer describes an edge, not a relation name. `blocked_by` is the proof:
    from an experiment or a decision it stays inside the action layer, but from
    a research question it crosses into it, and a single declared value on the
    relation cannot be right for both. So the layer is derived per edge — same
    layer at both ends keeps that layer, different ends are a `seam`.

    `meta` is the exception. `supersedes` and `duplicate_of` are meta because of
    what they say about the graph, not because of where their endpoints sit, so
    a declared `meta` is preserved.

    `declared` is the fallback for an endpoint whose type is not resolvable yet
    — an edge naming a node created later in the same patch, for instance.
    """
    if declared == "meta":
        return declared
    layers: list[str] = []
    for node_id in (source_id, target_id):
        node = state.nodes.get(node_id) if isinstance(node_id, str) else None
        base_type = getattr(node, "type", None)
        layer = BASE_TYPE_LAYERS.get(base_type) if isinstance(base_type, str) else None
        if layer is None:
            return declared
        layers.append(layer)
    return layers[0] if layers[0] == layers[1] else "seam"


def semantic_type(node: ProjectNode | dict[str, Any]) -> str | None:
    if isinstance(node, dict):
        extension = node.get("extension_type")
        base = node.get("type")
    else:
        extension = node.extension_type
        base = node.type
    return extension if isinstance(extension, str) else base if isinstance(base, str) else None


def edge_matches_relation(
    state: GraphState,
    source_id: str,
    target_id: str,
    relation: OntologyRelationDefinition,
    *,
    created_nodes: Iterable[dict[str, Any]] = (),
) -> bool:
    created = {raw.get("id"): raw for raw in created_nodes}
    source = state.nodes.get(source_id) or created.get(source_id)
    target = state.nodes.get(target_id) or created.get(target_id)
    return (
        source is not None
        and target is not None
        and semantic_type(source) in relation.source_types
        and semantic_type(target) in relation.target_types
    )


def ontology_type(ontology: OntologyState, name: Any):
    if not isinstance(name, str):
        return None
    return next((item for item in ontology.types if item.name == name), None)


def applicable_field_definitions(
    ontology: OntologyState, base_type: Any, extension_type: Any
) -> list[OntologyFieldDefinition]:
    owners = {base_type}
    if isinstance(extension_type, str):
        owners.add(extension_type)
    return [item for item in ontology.fields if item.owner_type in owners]


def field_applies(node: ProjectNode, field: OntologyFieldDefinition) -> bool:
    return field.owner_type in {node.type, node.extension_type}


def field_value_matches(field: OntologyFieldDefinition, value: Any) -> bool:
    if field.kind == "text":
        return isinstance(value, str)
    if field.kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field.kind == "boolean":
        return isinstance(value, bool)
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_extension_fields(
    node_id: Any,
    fields: dict[str, Any],
    definitions: list[OntologyFieldDefinition],
    report: ValidationReport,
    revision: int | None,
    *,
    authoring: bool,
    agent_authored: bool,
    structural: bool,
) -> None:
    known = {item.name: item for item in definitions}
    unknown = sorted(set(fields) - set(known)) if structural else []
    if unknown:
        report.reject(
            "unknown-extension-field",
            f"Node {node_id!r} uses undefined extension fields: {unknown}.",
            revision,
            related_node_ids=[node_id] if isinstance(node_id, str) else [],
        )
    for name, value in fields.items():
        definition = known.get(name)
        if definition is not None:
            _validate_one_field(
                node_id,
                definition,
                value,
                report,
                revision,
                authoring=authoring,
                agent_authored=agent_authored,
                check_kind=structural,
            )
    if structural:
        missing = sorted(
            item.name for item in definitions if item.required and item.name not in fields
        )
        if missing:
            report.reject(
                "missing-required-extension-field",
                f"Node {node_id!r} is missing required extension fields: {missing}.",
                revision,
                related_node_ids=[node_id] if isinstance(node_id, str) else [],
            )


def _validate_one_field(
    node_id: Any,
    definition: OntologyFieldDefinition,
    value: Any,
    report: ValidationReport,
    revision: int | None,
    *,
    authoring: bool,
    agent_authored: bool,
    check_kind: bool,
) -> None:
    if check_kind and not field_value_matches(definition, value):
        report.reject(
            "invalid-extension-field-kind",
            f"Field {definition.owner_type}.{definition.name} on {node_id!r} must be "
            f"{definition.kind}.",
            revision,
            related_node_ids=[node_id] if isinstance(node_id, str) else [],
        )
    if authoring and definition.deprecated:
        report.reject(
            "deprecated-extension-field",
            f"Field {definition.owner_type}.{definition.name} is deprecated and cannot be written.",
            revision,
            related_node_ids=[node_id] if isinstance(node_id, str) else [],
        )
    if authoring and agent_authored and not definition.agent_writable:
        report.reject(
            "extension-field-human-only",
            f"Field {definition.owner_type}.{definition.name} may only be written by a human.",
            revision,
            related_node_ids=[node_id] if isinstance(node_id, str) else [],
        )


def _require_deprecation_before_removal(
    label: str,
    previous: dict[Any, Any],
    desired: dict[Any, Any],
    report: ValidationReport,
    revision: int | None,
) -> None:
    for key in sorted(set(previous) - set(desired), key=str):
        if not previous[key].deprecated:
            report.reject(
                "ontology-removal-without-deprecation",
                f"Ontology {label} {key!r} must be deprecated in an earlier revision before removal.",
                revision,
            )
