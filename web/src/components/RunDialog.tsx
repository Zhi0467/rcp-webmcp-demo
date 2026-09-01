import { AlertTriangle, Play, X } from "lucide-react";
import { useEffect, useState } from "react";
import type { AgentRunConfig, ProjectSnapshot } from "../types";
import { AgentConfigControls, profileRunConfig } from "./AgentConfigControls";
import { RepositoryScope } from "./RepositoryScope";

interface Props {
  open: boolean;
  kind: "seed" | "refresh" | "node_chat";
  project: ProjectSnapshot;
  initialScope: string[];
  initialConfig?: AgentRunConfig;
  mode?: "start" | "retry";
  busy: boolean;
  onClose: () => void;
  onRun: (config: AgentRunConfig, scope: string[], message: string | null) => void;
}

export function RunDialog({
  open,
  kind,
  project,
  initialScope,
  initialConfig,
  mode = "start",
  busy,
  onClose,
  onRun,
}: Props) {
  const [scope, setScope] = useState(initialScope);
  const [config, setConfig] = useState(() => profileRunConfig(project.agent_profiles[kind]));
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!open) return;
    setScope(initialScope);
    setConfig(initialConfig || profileRunConfig(project.agent_profiles[kind]));
  }, [open, initialConfig, initialScope, kind, project.id]);

  useEffect(() => {
    if (open) setMessage("");
  }, [open]);

  if (!open) return null;
  const switchingExperimentProvider = mode === "retry" && kind === "node_chat";
  const switchSelectionUnchanged = Boolean(
    switchingExperimentProvider && initialConfig && !agentSelectionChanged(config, initialConfig),
  );
  const readiness = project.provider_readiness[config.run_on]?.[config.provider];
  // Which runtime this run will use. A request cannot override the profile's
  // runtime, and the backend swaps in the provider default only when the run
  // overrides the provider — so a provider override silently moves the runtime
  // too. The override is a draft this dialog owns, which is why the answer is
  // assembled here rather than exported.
  const profileRuntime =
    config.provider === project.agent_profiles[kind].provider
      ? project.agent_profiles[kind].runtime
      : readiness?.default_runtime;
  const providerReady =
    readiness === undefined || Boolean(readiness.installed && readiness.authenticated);
  const crossMachineRepositories = project.repositories.filter(
    (repository) => scope.includes(repository.alias) && repository.machine !== config.run_on,
  );
  const machinesByAlias = new Map(project.machines.map((machine) => [machine.alias, machine]));
  const hostlessRepositories = crossMachineRepositories.filter(
    (repository) => !machinesByAlias.get(repository.machine)?.host.trim(),
  );
  const sshRepositories = crossMachineRepositories.filter((repository) =>
    Boolean(machinesByAlias.get(repository.machine)?.host.trim()),
  );

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
    >
      <section
        className="run-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="run-dialog-title"
      >
        <header>
          <h2 id="run-dialog-title">
            {switchingExperimentProvider
              ? "Switch Experiment provider"
              : mode === "retry"
                ? `Retry ${kind === "seed" ? "seed" : "refresh"}`
                : kind === "seed"
                  ? "Seed the project graph"
                  : "Refresh project understanding"}
          </h2>
          <button className="icon-button" onClick={onClose} disabled={busy} aria-label="Close">
            <X size={17} />
          </button>
        </header>
        {mode === "start" && (
          <>
            <div className="run-dialog-section">
              <span className="field-label">Truth input subset</span>
              <RepositoryScope
                repositories={project.repositories}
                projectScope={project.project_truth_scope}
                stateRepository={project.state_repository}
                selected={scope}
                onChange={setScope}
              />
            </div>
            <div className="run-dialog-section">
              <label className="node-edit-field">
                <span>Additional message (optional)</span>
                <textarea
                  rows={4}
                  value={message}
                  disabled={busy}
                  onChange={(event) => setMessage(event.target.value)}
                />
              </label>
            </div>
          </>
        )}
        <AgentConfigControls
          project={project}
          value={config}
          onChange={setConfig}
          runtime={profileRuntime ? { value: profileRuntime, locked: true } : undefined}
          runOnLocked
          collapsible
        />
        {hostlessRepositories.length > 0 && (
          <div className="run-staging-warning">
            <AlertTriangle size={15} />
            <span>
              <strong>
                {hostlessRepositories.map((repository) => repository.alias).join(", ")} cannot be
                read from {config.run_on}.
              </strong>
              {" Their machines have no SSH host; remove them from this run."}
            </span>
          </div>
        )}
        {sshRepositories.length > 0 && (
          <div className="run-staging-warning">
            <AlertTriangle size={15} />
            <span>
              <strong>
                {sshRepositories.map((repository) => repository.alias).join(", ")} will be read over
                SSH at their declared paths.
              </strong>
              {" Repositories are never copied."}
            </span>
          </div>
        )}
        <footer>
          <button className="button secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="button primary"
            disabled={
              busy ||
              (mode === "start" && scope.length === 0) ||
              switchSelectionUnchanged ||
              !providerReady ||
              hostlessRepositories.length > 0
            }
            onClick={() => onRun(config, scope, message.trim() || null)}
          >
            <Play size={14} />{" "}
            {mode === "retry"
              ? busy
                ? switchingExperimentProvider
                  ? "Switching…"
                  : "Retrying…"
                : switchingExperimentProvider
                  ? "Switch provider"
                  : "Retry"
              : busy
                ? kind === "seed"
                  ? "Seeding…"
                  : "Refreshing…"
                : kind === "seed"
                  ? "Start seed"
                  : "Start refresh"}
          </button>
        </footer>
      </section>
    </div>
  );
}

export function agentSelectionChanged(current: AgentRunConfig, initial: AgentRunConfig): boolean {
  return (
    current.provider !== initial.provider ||
    current.model !== initial.model ||
    current.reasoning !== initial.reasoning
  );
}
