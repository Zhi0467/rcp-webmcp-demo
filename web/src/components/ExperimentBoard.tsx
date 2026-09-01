import { ChevronRight, FlaskConical, WifiOff } from "lucide-react";
import { useMemo } from "react";
import {
  buildExperimentBoard,
  experimentBoardRouteToken,
  experimentTerminalLabel,
  type ExperimentBoardItem,
} from "../experimentBoard";
import { activeExperimentGuidanceText } from "../experimentGuidance";
import { experimentHealthLabel, experimentHealthTone } from "./ExperimentRunDetail";
import type { ExperimentLoopIndexEntry } from "../types";

interface Props {
  entries: ExperimentLoopIndexEntry[];
  onOpen: (projectId: string, experimentRoute: string) => void;
}

export function ExperimentBoard({ entries, onOpen }: Props) {
  const board = useMemo(() => buildExperimentBoard(entries), [entries]);
  const count = entries.length;

  return (
    <section className="experiment-board" aria-labelledby="experiment-board-title">
      <header className="experiment-board-header">
        <div>
          <span className="experiment-board-kicker">
            <FlaskConical size={13} aria-hidden="true" /> Loop register
          </span>
          <h2 id="experiment-board-title">Experiments</h2>
        </div>
        <p className="experiment-board-counts" aria-label="Current experiment loop totals">
          <strong>{board.needsAction.length}</strong> needs action
          <span aria-hidden="true">·</span>
          <strong>{board.inProgress.length}</strong> in progress
          <span aria-hidden="true">·</span>
          <strong>{board.finished.length}</strong> finished
        </p>
      </header>

      {count === 0 ? (
        <div className="experiment-board-empty">No Experiment loops have been launched.</div>
      ) : (
        <div className="experiment-board-register">
          <ExperimentSection title="Needs action" items={board.needsAction} onOpen={onOpen} />
          <ExperimentSection title="In progress" items={board.inProgress} onOpen={onOpen} />
          {board.finished.length > 0 && (
            <details className="experiment-board-finished">
              <summary>
                <span>Finished</span>
                <span className="experiment-board-section-count">{board.finished.length}</span>
                <ChevronRight size={14} className="experiment-board-fold-icon" aria-hidden="true" />
              </summary>
              <ExperimentRows items={board.finished} onOpen={onOpen} />
            </details>
          )}
        </div>
      )}
    </section>
  );
}

function ExperimentSection({
  title,
  items,
  onOpen,
}: {
  title: string;
  items: ExperimentBoardItem[];
  onOpen: Props["onOpen"];
}) {
  if (items.length === 0) return null;
  return (
    <section className="experiment-board-section" aria-label={title}>
      <header>
        <h3>{title}</h3>
        <span className="experiment-board-section-count">{items.length}</span>
      </header>
      <ExperimentRows items={items} onOpen={onOpen} />
    </section>
  );
}

function ExperimentRows({
  items,
  onOpen,
}: {
  items: ExperimentBoardItem[];
  onOpen: Props["onOpen"];
}) {
  return (
    <ul className="experiment-board-rows">
      {items.map((item) => {
        const { entry, health, lastActivityAt } = item;
        const tone = experimentHealthTone(health);
        const statusLabel =
          health === "completed"
            ? experimentTerminalLabel(entry.node.status)
            : experimentHealthLabel(health);
        const summary = activeExperimentGuidanceText(entry.node);
        return (
          <li
            className={`experiment-board-row ${tone}`}
            key={`${entry.project_id}:${entry.node.id}:${entry.control.episode_id ?? "unbound"}`}
          >
            <button
              type="button"
              onClick={() => onOpen(entry.project_id, experimentBoardRouteToken(entry))}
            >
              <span className="experiment-board-status-rail" aria-hidden="true" />
              <span className="experiment-board-row-copy">
                <span className="experiment-board-row-heading">
                  <strong>{entry.node.title}</strong>
                  <span className="experiment-board-project">{entry.project_name}</span>
                </span>
                {summary && <span className="experiment-board-summary">{summary}</span>}
              </span>
              <span className="experiment-board-row-meta">
                <span className={`status-pill ${tone}`}>{statusLabel}</span>
                {entry.project_reachable === false && (
                  <span className="experiment-board-unavailable">
                    <WifiOff size={11} aria-hidden="true" /> Unavailable
                  </span>
                )}
                <time dateTime={lastActivityAt ?? undefined}>{formatActivity(lastActivityAt)}</time>
              </span>
              <ChevronRight className="experiment-board-row-arrow" size={15} aria-hidden="true" />
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function formatActivity(timestamp: string | null): string {
  if (!timestamp) return "Activity time unavailable";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "Activity time unavailable";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}
