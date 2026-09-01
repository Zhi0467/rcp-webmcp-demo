"""Closed task contracts for the graph-only branch merge agent."""

from __future__ import annotations


def branch_merge_task_contract(
    *,
    context_path: str,
    context_id: str,
    patch_path: str,
    validator_command: str,
) -> str:
    """Describe one fresh semantic rebase without exposing repositories."""

    _require_inputs(context_path, context_id, patch_path, validator_command)
    return f"""# RCP graph-branch merge

You are the dedicated graph-only merge agent for one human-dispatched Auto-research branch
merge. This task carries orchestrator graph authority, but it carries no repository authority
and no authority over project configuration, ontology, membership, or Proposal approval.

Exact immutable inputs:
- merge context: `{context_path}`
- merge context id: `{context_id}`
- candidate Patch output: `{patch_path}`
- live validator command: `{validator_command}`

The merge context contains the immutable branch-base graph, exact branch-head graph, current
main graph, a typed base-to-branch semantic delta, branch Patch summaries, transition-manager
contracts, and deterministic three-way conflicts. Treat those files and heads as exact. Do not
infer a different base, inspect canonical state directories, or inspect any repository.

Produce one semantic Patch that carries the branch's intended graph change onto the supplied
current main graph. Preserve compatible main-side changes. Resolve every listed conflict
explicitly from the supplied graph semantics; never resolve one by silently preferring an entire
branch or main object. If the intended outcome cannot be represented legally, leave a precise
diagnostic in your final response and do not invent authority.

Only permitted file output:
- Write exactly one JSON object to `{patch_path}` matching the orchestrator agent Patch schema in
  the merge context.
- Include only `summary`, semantic `ops`, `repositories_read` (which must be `[]`),
  `change_summary`, and `agent_action` only when the operation actually chooses a Decision.
- Do not include revisions, graph heads, merge ids, branch provenance, authorizers, task ids,
  transition traces, admission fields, or other RCP bookkeeping. RCP supplies all of them.
- Do not write repository files, watcher files, artifacts, or canonical `.research` files.

Before finishing, run the exact validator command. Exit 0 means the candidate is semantically
valid against the live current main graph, exit 1 supplies a correction diagnostic, and exit 2
means validation is unavailable rather than semantically invalid. A successful self-check does
not commit anything; RCP revalidates and commits atomically or commits nothing.
"""


def branch_merge_correction_contract(
    *,
    original_contract_path: str,
    context_path: str,
    context_id: str,
    patch_path: str,
    diagnostics_path: str,
    validator_command: str,
) -> str:
    """Request one bounded scratch-only correction in the same native session."""

    _require_inputs(context_path, context_id, patch_path, validator_command)
    if not original_contract_path or not diagnostics_path:
        raise ValueError("branch merge correction requires exact contract and diagnostic paths")
    return f"""# RCP graph-branch merge Patch correction

Continue the exact native session and scratch stage from `{original_contract_path}`.

Exact current inputs:
- merge context: `{context_path}`
- merge context id: `{context_id}`
- validation diagnostic: `{diagnostics_path}`
- candidate Patch output to replace: `{patch_path}`
- live validator command: `{validator_command}`

Correct only the semantic candidate Patch described by the original contract. The branch and
main heads have not changed. Read the diagnostic, rewrite `{patch_path}` with a different valid
orchestrator semantic Patch, and run the validator command before finishing. Do not repeat or
perform any operational side effect. Do not add RCP bookkeeping or branch provenance, inspect
repositories, write watcher/artifact files, or write canonical state.
"""


def branch_merge_rebase_contract(
    *,
    original_contract_path: str,
    previous_context_id: str,
    context_path: str,
    context_id: str,
    patch_path: str,
    validator_command: str,
) -> str:
    """Replace a discarded candidate after main moved, preserving the native session."""

    _require_inputs(context_path, context_id, patch_path, validator_command)
    if not original_contract_path or not previous_context_id:
        raise ValueError("branch merge rebase requires the original and previous context ids")
    if previous_context_id == context_id:
        raise ValueError("branch merge rebase requires a newly resolved main context")
    return f"""# RCP graph-branch merge rebase

Continue the exact native session and scratch stage from `{original_contract_path}`. RCP
discarded the previous candidate because main advanced; nothing from that candidate was
committed.

Stale merge context id: `{previous_context_id}`
Replacement immutable merge context: `{context_path}`
Replacement merge context id: `{context_id}`
Candidate Patch output to rewrite: `{patch_path}`
Live validator command: `{validator_command}`

Recompute the semantic merge against the replacement current-main graph. Preserve compatible
new main changes and explicitly resolve the replacement context's conflicts. Rewrite the Patch;
do not reuse the stale candidate unchanged. Run the validator command before finishing. This is
still graph-only: perform no operational side effects, inspect no repositories, and write no
watcher, artifact, or canonical-state files. RCP supplies all provenance and commits atomically
or commits nothing.
"""


def _require_inputs(
    context_path: str,
    context_id: str,
    patch_path: str,
    validator_command: str,
) -> None:
    if not context_path or not patch_path or not validator_command:
        raise ValueError("branch merge contract requires exact context, output, and validator")
    if len(context_id) != 64 or any(
        character not in "0123456789abcdef" for character in context_id
    ):
        raise ValueError("branch merge context id must be a SHA-256 digest")
