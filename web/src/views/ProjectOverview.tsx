import { ArrowUpRight } from "lucide-react";
import { currentExperimentGuidance } from "../experimentGuidance";
import type {
  AppView,
  GraphNode,
  GraphState,
  ProjectSnapshot,
  Proposal,
  RevisionSummary,
} from "../types";

interface Props {
  project: ProjectSnapshot;
  graph: GraphState;
  pendingProposals: Proposal[];
  decisionsAwaitingChoice: GraphNode[];
  latestRevisionSummary?: RevisionSummary | null;
  onNavigate: (view: AppView) => void;
}

export function ProjectOverview({
  project,
  graph,
  pendingProposals,
  decisionsAwaitingChoice,
  latestRevisionSummary,
  onNavigate,
}: Props) {
  const nodes = Object.values(graph.nodes);
  const activeExperiments = nodes.filter(
    (node) => node.type === "experiment" && !project.experiment_control[node.id]?.node_closed,
  );
  const latestNode = [...nodes].sort((left, right) => right.updated_rev - left.updated_rev)[0];
  const blockers = nodes.filter((node) => node.type === "blocker" && node.status === "open");
  const nextExperiment = activeExperiments.find((node) =>
    currentExperimentGuidance(node, "next_action"),
  );
  const nextExperimentAction = nextExperiment
    ? currentExperimentGuidance(nextExperiment, "next_action")
    : null;
  const question =
    project.primary_question?.question ||
    project.primary_question?.title ||
    "No primary research question has been seeded.";
  const latestRevisionText = latestRevisionSummary?.sentences.slice(0, 2).join(" ").trim();
  const latestRevisionDetail =
    latestRevisionSummary && latestRevisionText
      ? `Revision ${latestRevisionSummary.from_revision} to revision ${latestRevisionSummary.to_revision}`
      : null;

  const rows: Array<{
    number: string;
    prompt: string;
    answer: string;
    detail: string;
    view: AppView;
    node?: GraphNode;
  }> = [
    {
      number: "01",
      prompt: "What are we asking?",
      answer: String(question),
      detail: `${nodes.filter((node) => node.type === "hypothesis").length} hypotheses · ${project.counts.accepted} accepted nodes`,
      view: "scientific",
    },
    {
      number: "02",
      prompt: "Where are we?",
      answer: activeExperiments.length
        ? `${activeExperiments.length} active experiment${activeExperiments.length === 1 ? "" : "s"}`
        : !project.last_refresh_at
          ? "The project has not been seeded yet."
          : "Understanding and review",
      detail: `Project revision ${graph.revision}`,
      view: "execution",
    },
    {
      number: "03",
      prompt: "What changed?",
      answer: latestRevisionText || (latestNode ? latestNode.title : "No graph changes yet."),
      detail: latestRevisionDetail
        ? latestRevisionDetail
        : project.last_refresh_at
          ? `Last refresh ${new Date(project.last_refresh_at).toLocaleString()}`
          : "Never refreshed",
      view: "scientific",
    },
    {
      number: "04",
      prompt: "What is blocked?",
      answer: blockers[0]?.title || "No open blocker is recorded.",
      detail: blockers.length
        ? `${blockers.length} open blocker${blockers.length === 1 ? "" : "s"}`
        : "No open blocker in the graph",
      view: "dag",
    },
    {
      number: "05",
      prompt: "What needs you?",
      answer:
        pendingProposals[0]?.title ||
        decisionsAwaitingChoice[0]?.title ||
        "Nothing currently requires human judgment.",
      detail: `${pendingProposals.length} proposals · ${decisionsAwaitingChoice.length} decisions awaiting choice`,
      view: "attention",
    },
    {
      number: "06",
      prompt: "What happens next?",
      answer: String(
        nextExperimentAction ||
          (!project.last_refresh_at
            ? "Seed the graph from the selected truth repositories."
            : "Refresh when new research work lands."),
      ),
      detail: nextExperiment ? nextExperiment.title : "Project-level next action",
      view: nextExperiment ? "execution" : "attention",
    },
  ];

  return (
    <section className="overview-page">
      <header className="overview-heading">
        <div className="overview-revision">
          <span>Project revision · {project.canonical_state.remote ? "remote" : "local"}</span>
          <strong>{String(graph.revision).padStart(3, "0")}</strong>
        </div>
      </header>
      <div className="overview-questions">
        {rows.map((row) => (
          <button key={row.number} onClick={() => onNavigate(row.view)}>
            <span className="overview-number">{row.number}</span>
            <span className="overview-question">
              <small>{row.prompt}</small>
              <strong>{row.answer}</strong>
            </span>
            <span className="overview-detail">{row.detail}</span>
            <ArrowUpRight size={18} />
          </button>
        ))}
      </div>
    </section>
  );
}
