import {
  ArchiveRestore,
  DatabaseBackup,
  GitCompareArrows,
  RefreshCw,
  ServerCog,
  ShieldCheck,
  Terminal,
  TriangleAlert,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { loadServerStatus } from "../api";
import type { ServerStatus, ServerStatusSummary } from "../types";

interface Props {
  loadStatus?: () => Promise<ServerStatus>;
}

export function shortCommit(commit: string | null): string {
  return commit?.slice(0, 10) ?? "Not available";
}

export function formatServerTimestamp(value: string | null): string {
  if (!value) return "Not recorded";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function formatServerBytes(value: number | null): string {
  if (value === null) return "Not recorded";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024;
    index += 1;
  }
  return `${amount >= 10 ? amount.toFixed(0) : amount.toFixed(1)} ${units[index]}`;
}

export function formatServerProjectCounts(
  protectedProjects: number | null,
  uncapturedProjects: number | null,
): string {
  if (protectedProjects === null || uncapturedProjects === null) return "Not recorded";
  return `${protectedProjects} protected · ${uncapturedProjects} uncaptured`;
}

function StatusMark({ summary }: { summary: ServerStatusSummary }) {
  return <span className={`server-status-mark ${summary.tone}`}>{summary.label}</span>;
}

function CommitRow({ label, commit }: { label: string; commit: string | null }) {
  return (
    <div className="server-commit-row">
      <dt>{label}</dt>
      <dd title={commit ?? undefined}>{shortCommit(commit)}</dd>
    </div>
  );
}

export function ServerSettings({ loadStatus = loadServerStatus }: Props) {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setStatus(await loadStatus());
    } catch (failure) {
      setStatus(null);
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setLoading(false);
    }
  }, [loadStatus]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return (
    <section className="settings-section server-settings" aria-labelledby="server-settings-title">
      <header>
        <span>
          <ServerCog size={16} />
        </span>
        <h2 id="server-settings-title">Team server</h2>
        {status ? <StatusMark summary={status.overall} /> : null}
        <button
          className="icon-button"
          type="button"
          aria-label="Refresh server status"
          disabled={loading}
          onClick={() => void reload()}
        >
          <RefreshCw className={loading ? "spin" : undefined} size={15} />
        </button>
      </header>

      {error ? (
        <div className="server-settings-error" role="alert">
          <TriangleAlert size={16} />
          <span>{error}</span>
        </div>
      ) : null}

      {!status && !error ? (
        <div className="server-settings-loading" aria-live="polite">
          Reading server status…
        </div>
      ) : null}

      {status ? (
        <div className="server-settings-body">
          <section className="server-release-ledger" aria-label="Source and release commits">
            <div className="server-settings-section-title">
              <GitCompareArrows size={15} />
              <h3>Source and release</h3>
              <StatusMark summary={status.releases.status} />
            </div>
            <dl className="server-commit-rail">
              <CommitRow label="Running" commit={status.releases.running_commit} />
              <CommitRow label="Installed" commit={status.releases.current_release_commit} />
              <CommitRow label="Managed main" commit={status.releases.managed_source_commit} />
              <CommitRow
                label="Last fetched origin/main"
                commit={status.releases.upstream_commit}
              />
              {status.releases.candidate_commit ? (
                <CommitRow label="Candidate" commit={status.releases.candidate_commit} />
              ) : null}
            </dl>
            {status.releases.last_update_failure ? (
              <p className="server-status-problem">{status.releases.last_update_failure}</p>
            ) : null}
            <code>{status.releases.command}</code>
          </section>

          <div className="server-status-card-grid">
            <article className="server-status-card">
              <header>
                <DatabaseBackup size={15} />
                <h3>Protected backup</h3>
              </header>
              <StatusMark summary={status.backup.status} />
              <dl>
                <div>
                  <dt>Last protected</dt>
                  <dd>{formatServerTimestamp(status.backup.last_protected_at)}</dd>
                </div>
                <div>
                  <dt>Last attempt</dt>
                  <dd>{formatServerTimestamp(status.backup.last_attempt_at)}</dd>
                </div>
                <div>
                  <dt>Projects</dt>
                  <dd>
                    {formatServerProjectCounts(
                      status.backup.protected_projects,
                      status.backup.uncaptured_projects,
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Captured</dt>
                  <dd>{formatServerBytes(status.backup.captured_bytes)}</dd>
                </div>
                <div>
                  <dt>Schedule</dt>
                  <dd>
                    {status.backup.schedule ?? "Not configured"}
                    {status.backup.retention !== null ? ` · keep ${status.backup.retention}` : ""}
                  </dd>
                </div>
                <div className="wide">
                  <dt>Destination</dt>
                  <dd className="mono">{status.backup.destination ?? "Not configured"}</dd>
                </div>
              </dl>
              {status.backup.last_failure ? (
                <p className="server-status-problem">{status.backup.last_failure}</p>
              ) : null}
              <div className="server-status-commands">
                <code>{status.backup.configure_command}</code>
                <code>{status.backup.run_command}</code>
              </div>
            </article>

            <article className="server-status-card">
              <header>
                <ArchiveRestore size={15} />
                <h3>Restore drill</h3>
              </header>
              <StatusMark summary={status.restore.status} />
              <dl>
                <div>
                  <dt>Completed</dt>
                  <dd>{formatServerTimestamp(status.restore.last_completed_at)}</dd>
                </div>
                <div>
                  <dt>Age</dt>
                  <dd>
                    {status.restore.drill_age_days === null
                      ? "Not recorded"
                      : `${status.restore.drill_age_days} days`}
                  </dd>
                </div>
              </dl>
              <code>{status.restore.command}</code>
            </article>

            <article className="server-status-card">
              <header>
                <ShieldCheck size={15} />
                <h3>Execution readiness</h3>
              </header>
              <div className="server-readiness-list">
                <StatusMark summary={status.execution.machine} />
                <StatusMark summary={status.execution.provider_checks} />
              </div>
              <p className="server-dependency-versions mono">
                {status.execution.dependency_versions}
              </p>
              <code>{status.execution.provider_command}</code>
            </article>
          </div>

          {status.problems.length > 0 ? (
            <section className="server-problem-ledger" aria-labelledby="server-problems-title">
              <div className="server-settings-section-title">
                <TriangleAlert size={15} />
                <h3 id="server-problems-title">Needs attention</h3>
              </div>
              <ul>
                {status.problems.map((problem) => (
                  <li key={problem}>{problem}</li>
                ))}
              </ul>
            </section>
          ) : null}

          <section className="server-command-ledger" aria-labelledby="server-commands-title">
            <div className="server-settings-section-title">
              <Terminal size={15} />
              <h3 id="server-commands-title">Console operations</h3>
            </div>
            <div className="server-command-list">
              {status.operator_commands.map((item) => (
                <div className="server-command-row" key={item.command}>
                  <strong>{item.name}</strong>
                  <code>{item.command}</code>
                  <span>{item.purpose}</span>
                </div>
              ))}
            </div>
          </section>
        </div>
      ) : null}
    </section>
  );
}
