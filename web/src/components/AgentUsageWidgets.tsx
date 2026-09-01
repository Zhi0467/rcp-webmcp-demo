import { useMemo, useState, type CSSProperties } from "react";
import {
  pickScreenStoryComparison,
  projectUsageTokens,
  screenStoryComparisonCopy,
} from "../screenStoryComparisons";
import type {
  AgentTaskKind,
  AgentUsageCell,
  AgentUsageMetric,
  AgentUsageRecord,
  AgentUsageSnapshot,
  ProviderReadiness,
} from "../types";

const taskLabels: Record<AgentTaskKind, string> = {
  seed: "Seed",
  refresh: "Refresh",
  node_chat: "Node chat",
  project_chat: "Project chat",
  paper_coach: "Paper coach",
  auto_research: "Auto-research",
  branch_merge: "Branch merge",
};

const taskOrder: AgentTaskKind[] = [
  "seed",
  "refresh",
  "node_chat",
  "project_chat",
  "paper_coach",
  "auto_research",
  "branch_merge",
];

interface Props {
  usage: AgentUsageSnapshot | null;
  providers: Record<string, ProviderReadiness>;
}

interface SelectedCell {
  taskKind: AgentTaskKind;
  provider: string;
}

export function AgentUsageWidgets({ usage, providers }: Props) {
  const [selectedCell, setSelectedCell] = useState<SelectedCell | null>(null);
  const [screenStoryComparison] = useState(() => pickScreenStoryComparison());
  const providerIds = useMemo(() => {
    const ids = new Set(Object.keys(providers));
    usage?.input_processed.cells.forEach((cell) => ids.add(cell.provider));
    usage?.generated.cells.forEach((cell) => ids.add(cell.provider));
    return [...ids].sort((left, right) =>
      providerLabel(providers, left).localeCompare(providerLabel(providers, right)),
    );
  }, [providers, usage]);

  if (!usage) {
    return <section className="settings-section agent-usage-settings" aria-busy="true" />;
  }

  const selectedRecords = selectedCell
    ? usage.records.filter(
        (record) =>
          record.task_kind === selectedCell.taskKind && record.provider === selectedCell.provider,
      )
    : [];
  const comparisonCopy = screenStoryComparisonCopy(
    projectUsageTokens(usage.input_processed.total_tokens, usage.generated.total_tokens),
    screenStoryComparison,
  );

  return (
    <section className="settings-section agent-usage-settings">
      <div className="agent-usage-widgets">
        <UsageWidget
          className="input"
          title="Input context"
          metric={usage.input_processed}
          providerIds={providerIds}
          providers={providers}
          selectedCell={selectedCell}
          onSelect={setSelectedCell}
          valueForCell={(cell) => cell.processed_input_tokens}
          meta={`${formatTokens(usage.input_processed.cached_tokens)} cached · ${Math.round(usage.input_processed.cache_share * 100)}%`}
        />
        <UsageWidget
          className="generated"
          title="Generated"
          metric={usage.generated}
          providerIds={providerIds}
          providers={providers}
          selectedCell={selectedCell}
          onSelect={setSelectedCell}
          valueForCell={(cell) => cell.generated_tokens}
        />
      </div>
      {comparisonCopy && <p className="agent-usage-comparison">{comparisonCopy}</p>}
      {selectedCell && (
        <UsageDetails
          taskKind={selectedCell.taskKind}
          provider={selectedCell.provider}
          providers={providers}
          records={selectedRecords}
          onClose={() => setSelectedCell(null)}
        />
      )}
    </section>
  );
}

function UsageWidget({
  className,
  title,
  metric,
  providerIds,
  providers,
  selectedCell,
  onSelect,
  valueForCell,
  meta,
}: {
  className: "input" | "generated";
  title: string;
  metric: AgentUsageMetric;
  providerIds: string[];
  providers: Record<string, ProviderReadiness>;
  selectedCell: SelectedCell | null;
  onSelect: (cell: SelectedCell) => void;
  valueForCell: (cell: AgentUsageCell) => number;
  meta?: string;
}) {
  return (
    <article className={`agent-usage-widget ${className}`}>
      <header>
        <strong>{title}</strong>
        <span>{formatTokens(metric.total_tokens)} tokens</span>
      </header>
      <div className="agent-usage-widget-meta">
        <span>1 square = {formatTokens(metric.block_tokens)}</span>
        {meta && <span>{meta}</span>}
      </div>
      <div
        className="agent-usage-grid"
        role="grid"
        aria-label={`${title} by task and provider`}
        style={{ "--usage-provider-count": providerIds.length } as CSSProperties}
      >
        <div className="agent-usage-grid-corner" />
        {providerIds.map((provider) => (
          <span className="agent-usage-provider" key={provider}>
            {providerLabel(providers, provider)}
          </span>
        ))}
        {taskOrder.map((taskKind) => (
          <UsageRow
            key={taskKind}
            taskKind={taskKind}
            providerIds={providerIds}
            providers={providers}
            metric={metric}
            selectedCell={selectedCell}
            onSelect={onSelect}
            valueForCell={valueForCell}
          />
        ))}
      </div>
    </article>
  );
}

function UsageRow({
  taskKind,
  providerIds,
  providers,
  metric,
  selectedCell,
  onSelect,
  valueForCell,
}: {
  taskKind: AgentTaskKind;
  providerIds: string[];
  providers: Record<string, ProviderReadiness>;
  metric: AgentUsageMetric;
  selectedCell: SelectedCell | null;
  onSelect: (cell: SelectedCell) => void;
  valueForCell: (cell: AgentUsageCell) => number;
}) {
  return (
    <>
      <span className="agent-usage-task">{taskLabels[taskKind]}</span>
      {providerIds.map((provider) => {
        const cell = metric.cells.find(
          (candidate) => candidate.task_kind === taskKind && candidate.provider === provider,
        ) ?? {
          task_kind: taskKind,
          provider,
          processed_input_tokens: 0,
          generated_tokens: 0,
          cached_input_tokens: 0,
          counted_records: 0,
        };
        const tokens = valueForCell(cell);
        const selected = selectedCell?.taskKind === taskKind && selectedCell.provider === provider;
        return (
          <button
            className={`agent-usage-cell${selected ? " selected" : ""}${tokens ? " has-value" : ""}`}
            key={provider}
            type="button"
            disabled={!tokens}
            aria-label={`${taskLabels[taskKind]}, ${providerLabel(providers, provider)}: ${formatTokens(tokens)} tokens`}
            onClick={() => onSelect({ taskKind, provider })}
          >
            <UsageSquares tokens={tokens} blockTokens={metric.block_tokens} />
            <span>{tokens ? formatTokens(tokens) : "—"}</span>
          </button>
        );
      })}
    </>
  );
}

function UsageSquares({ tokens, blockTokens }: { tokens: number; blockTokens: number }) {
  if (!tokens || !blockTokens) return <span className="agent-usage-squares empty" />;
  const equivalents = tokens / blockTokens;
  const full = Math.floor(equivalents);
  const remainder = equivalents - full;
  return (
    <span className="agent-usage-squares" aria-hidden="true">
      {Array.from({ length: full }, (_, index) => (
        <i className="full" key={`full-${index}`} />
      ))}
      {remainder > 0 && (
        <i
          className="partial"
          key="partial"
          style={{ "--usage-fill": `${Math.max(2, remainder * 100)}%` } as CSSProperties}
        />
      )}
    </span>
  );
}

function UsageDetails({
  taskKind,
  provider,
  providers,
  records,
  onClose,
}: {
  taskKind: AgentTaskKind;
  provider: string;
  providers: Record<string, ProviderReadiness>;
  records: AgentUsageRecord[];
  onClose: () => void;
}) {
  return (
    <div className="agent-usage-details">
      <header>
        <strong>
          {taskLabels[taskKind]} · {providerLabel(providers, provider)}
        </strong>
        <button className="button secondary compact" type="button" onClick={onClose}>
          Close
        </button>
      </header>
      {records.length === 0 ? (
        <p className="agent-usage-empty">No usage records.</p>
      ) : (
        <div className="agent-usage-records">
          {records.map((record) => (
            <article key={record.usage_id} className={record.counted ? "counted" : "excluded"}>
              <header>
                <strong>{record.counted ? "Counted" : "Excluded"}</strong>
                <span>{new Date(record.created_at).toLocaleString()}</span>
              </header>
              <dl>
                <div>
                  <dt>Operation</dt>
                  <dd>{record.operation_id}</dd>
                </div>
                <div>
                  <dt>Profile</dt>
                  <dd>{record.provider_profile}</dd>
                </div>
                <div>
                  <dt>Input</dt>
                  <dd>{formatTokens(record.processed_input_tokens)}</dd>
                </div>
                <div>
                  <dt>Generated</dt>
                  <dd>{formatTokens(record.generated_tokens)}</dd>
                </div>
                {!record.counted && (
                  <div>
                    <dt>Reason</dt>
                    <dd>{record.count_reason}</dd>
                  </div>
                )}
              </dl>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function providerLabel(providers: Record<string, ProviderReadiness>, provider: string): string {
  return providers[provider]?.label || provider;
}

function formatTokens(tokens: number): string {
  if (!tokens) return "0";
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(2)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(tokens >= 100_000 ? 0 : 1)}k`;
  return String(Math.round(tokens));
}
