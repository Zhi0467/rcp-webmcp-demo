from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from rcp.core.models import GraphState, Patch
from rcp.core.operations import GraphOperation, ProposalOperation
from rcp.core.validation.report import ValidationReport


@dataclass
class OpContext:
    """Everything an operation rule needs to judge one operation of a patch.

    ``initial_state`` retains the graph before any operation is staged, while
    ``state`` advances after each valid operation. ``repositories`` is deliberately
    mutable and shared across the operations of a single patch: a
    ``set_project_truth_scope`` operation that introduces a new repository descriptor
    makes that alias available to the operations after it.
    """

    state: GraphState
    initial_state: GraphState
    patch: Patch
    report: ValidationReport
    revision: int | None
    project_truth_scope: set[str]
    repositories: set[str]
    machines: set[str] | None
    default_run_truth_scope: set[str]
    state_repository: str | None
    mode: Literal["admission", "replay"]
    experiment_control_node_id: str | None = None
    reference_patch: Patch | None = None


#: Validates one operation, reporting into ``ctx.report``. Returns the oldest
#: source-reference timestamp the operation cited, or ``None`` when it cites none.
OpValidator = Callable[[GraphOperation, OpContext], Any]

#: Returns the graph nodes and project-config keys one operation depends on, as
#: ``(candidate node ids, config keys)``. Node ids are retained even when the
#: outer patch creates them, because RCP derives Proposal bookkeeping before the
#: staged patch has materialized those nodes.
OpDependencies = Callable[
    [GraphOperation | ProposalOperation, GraphState], tuple[list[Any], list[str]]
]


@dataclass(frozen=True)
class OpRule:
    """One entry of the operation vocabulary.

    An operation with no ``validate`` carries no operation-level checks; one with
    no ``dependencies`` cannot make a proposal depend on existing graph state.
    Per-operation metadata belongs here as further fields.
    """

    structural_validate: OpValidator | None = None
    authoring_validate: OpValidator | None = None
    dependencies: OpDependencies | None = None
    legacy_only: bool = False
