from __future__ import annotations

from collections import defaultdict

from rcp.core.models import Decision, Evidence, GraphState, Hypothesis, ResearchQuestion, Standing

EVIDENCE_HYPOTHESIS_RELATIONS = frozenset(
    {"supports", "weakens", "refutes", "inconclusive", "contradicts"}
)


def render_research_md(state: GraphState) -> str:
    accepted = [node for node in state.nodes.values() if node.standing == Standing.ACCEPTED]
    sections: dict[str, list[str]] = defaultdict(list)

    for node in sorted(accepted, key=lambda item: item.id):
        if isinstance(node, ResearchQuestion):
            details = node.question
            if node.scope:
                details += f" Scope: {node.scope}"
            sections["Research questions"].append(f"- **{node.title}** — {details}")
        elif isinstance(node, Hypothesis):
            scope = f" Scope: {node.scope}" if node.scope else ""
            sections["Hypotheses"].append(
                f"- **{node.title}** (`{node.status}`) — {node.statement}{scope}"
            )
        elif isinstance(node, Decision):
            if node.status == "decided":
                selected = node.selected_option or "Decision recorded without a selected option"
                rationale = f" {node.rationale}" if node.rationale else ""
                line = f"- **{node.title}** — Decided: {selected}.{rationale}"
            else:
                line = f"- **{node.title}** — **Open:** {node.question}"
            sections["Decisions"].append(line)

    for edge in sorted(state.edges.values(), key=lambda item: item.id):
        source = state.nodes.get(edge.source)
        target = state.nodes.get(edge.target)
        if (
            not isinstance(source, Evidence)
            or not isinstance(target, Hypothesis)
            or target.standing != Standing.ACCEPTED
            or edge.relation not in EVIDENCE_HYPOTHESIS_RELATIONS
        ):
            continue
        if edge.assessment is None:
            assessment = "unassessed legacy relation"
        else:
            details = [edge.assessment.relevance, edge.assessment.weight]
            if edge.assessment.scope:
                details.append(f"scope: {edge.assessment.scope}")
            if edge.assessment.qualifications:
                details.append("qualifications: " + "; ".join(edge.assessment.qualifications))
            assessment = ", ".join(details)
        sections["Evidence assessments"].append(
            f"- **{source.title}** `{edge.relation}` **{target.title}** "
            f"({assessment}) — {source.observation}"
        )

    if not sections:
        return ""

    lines = ["# Accepted research", "", f"Generated from graph revision {state.revision}.", ""]
    for heading in ("Research questions", "Hypotheses", "Decisions", "Evidence assessments"):
        entries = sections.get(heading)
        if not entries:
            continue
        lines.extend((f"## {heading}", "", *entries, ""))
    return "\n".join(lines).rstrip() + "\n"
