import {
  CheckCircle2,
  ChevronDown,
  LoaderCircle,
  LockKeyhole,
  RefreshCw,
  TriangleAlert,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import {
  modelChange,
  modelOptions,
  providerChange,
  providerOptions,
  reasoningOptions,
  runtimeOptions,
} from "../providers";
import type { AgentProfile, AgentRunConfig, ProjectSnapshot } from "../types";

interface Props {
  project: ProjectSnapshot;
  value: AgentRunConfig;
  onChange: (value: AgentRunConfig) => void;
  locked?: boolean;
  runOnLocked?: boolean;
  compact?: boolean;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  /** Re-probe the provider CLIs. Rendered as a control only where passed. */
  onRefreshReadiness?: () => Promise<void>;
  /**
   * The profile runtime this configuration runs on. A run request cannot
   * override it, so a surface that only builds a request passes `locked` and
   * shows what the profile will use.
   */
  runtime?: {
    value: string;
    locked?: boolean;
    onChange?: (value: string) => void;
  };
  children?: ReactNode;
}

export function profileRunConfig(profile: AgentProfile): AgentRunConfig {
  return {
    provider: profile.provider,
    model: profile.model,
    reasoning: profile.reasoning,
    run_on: profile.run_on,
  };
}

export function AgentConfigControls({
  project,
  value,
  onChange,
  locked = false,
  runOnLocked = false,
  compact = false,
  collapsible = false,
  defaultCollapsed = false,
  onRefreshReadiness,
  runtime,
  children,
}: Props) {
  const [expanded, setExpanded] = useState(!defaultCollapsed);
  const [reprobing, setReprobing] = useState(false);
  const onMachine = project.provider_readiness[value.run_on] ?? {};
  const readiness = onMachine[value.provider];
  const machine = project.machines.find((item) => item.alias === value.run_on);
  const update = (patch: Partial<AgentRunConfig>) => onChange({ ...value, ...patch });
  const className = compact ? "agent-config compact" : "agent-config";
  const providerName = readiness?.label || value.provider;

  // Everything offered below comes from the backend registry's probe of this
  // machine: which providers exist, what they are called, which models they
  // accept, and which reasoning efforts each of those models accepts.
  const models = readiness?.models ?? [];
  const providers = providerOptions(Object.values(onMachine), value.provider);
  const modelChoices = modelOptions(models, value.model);
  const reasoningChoices = reasoningOptions(models, value.model, value.reasoning);
  const runtimeChoices = runtimeOptions(readiness, runtime?.value ?? "");

  const contents = (
    <>
      <div className="agent-config-fields">
        <label>
          <span>Provider</span>
          <select
            value={value.provider}
            disabled={locked}
            onChange={(event) => {
              const next = onMachine[event.target.value];
              update(providerChange(next?.models ?? [], event.target.value, value.reasoning));
              // The runtime belongs to the provider, so it moves with it. Only
              // this machine's readiness can name the new provider's default,
              // and the list above cannot offer a provider without it.
              if (next) runtime?.onChange?.(next.default_runtime);
            }}
          >
            {providers.map(({ id, label }) => (
              <option value={id} key={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        {runtime && (
          <label className={runtime.locked ? "agent-machine-fixed" : undefined}>
            <span>
              Runtime {runtime.locked ? <LockKeyhole size={10} aria-hidden="true" /> : null}
            </span>
            <select
              value={runtime.value}
              // One choice is not a choice, and a request cannot override the
              // profile's runtime. Both read as the locked control they are.
              disabled={locked || runtime.locked || runtimeChoices.length < 2}
              onChange={(event) => runtime.onChange?.(event.target.value)}
            >
              {runtimeChoices.map(({ id, label }) => (
                <option value={id} key={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        )}
        <label>
          <span>Model</span>
          <select
            value={value.model}
            disabled={locked}
            onChange={(event) => update(modelChange(models, event.target.value, value.reasoning))}
          >
            {modelChoices.map(({ id, label }) => (
              <option value={id} key={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        {reasoningChoices.length > 0 && (
          <label>
            <span>Reasoning</span>
            <select
              value={value.reasoning}
              disabled={locked}
              onChange={(event) => update({ reasoning: event.target.value })}
            >
              {reasoningChoices.map(({ id, label }) => (
                <option value={id} key={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className={runOnLocked ? "agent-machine-fixed" : undefined}>
          <span>Run on {runOnLocked ? <LockKeyhole size={10} aria-hidden="true" /> : null}</span>
          <select
            value={value.run_on}
            disabled={locked || runOnLocked}
            onChange={(event) => update({ run_on: event.target.value })}
          >
            {project.machines.map((item) => (
              <option value={item.alias} key={item.alias}>
                {item.alias}
                {item.host ? ` · ${item.host}` : " · local"}
              </option>
            ))}
          </select>
        </label>
      </div>
      {!compact && (
        <>
          <div
            className={
              readiness === undefined
                ? "agent-readiness pending"
                : readiness.authenticated
                  ? "agent-readiness ready"
                  : "agent-readiness warning"
            }
          >
            {readiness === undefined ? (
              <LoaderCircle className="spin" size={14} />
            ) : readiness.authenticated ? (
              <CheckCircle2 size={14} />
            ) : (
              <TriangleAlert size={14} />
            )}
            <span>
              {readiness === undefined
                ? `Checking ${value.provider} on ${machine?.host || "this machine"}…`
                : readiness.authenticated
                  ? `${readiness.version || value.provider} ready on ${machine?.host || "this machine"}`
                  : readiness?.reason ||
                    `${value.provider} is not ready on ${machine?.host || "this machine"}`}
            </span>
            {onRefreshReadiness && (
              <button
                type="button"
                className="icon-button compact"
                disabled={reprobing}
                aria-label="Re-check provider CLIs"
                onClick={() => {
                  setReprobing(true);
                  void onRefreshReadiness().finally(() => setReprobing(false));
                }}
              >
                {reprobing ? <LoaderCircle className="spin" size={13} /> : <RefreshCw size={13} />}
              </button>
            )}
          </div>
        </>
      )}
      {children}
    </>
  );

  if (!collapsible) return <div className={className}>{contents}</div>;

  return (
    <details
      className={`${className} collapsible`}
      open={expanded}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary className="agent-config-summary" aria-label={`${providerName} agent settings`}>
        <span>{providerName}</span>
        <ChevronDown size={12} aria-hidden="true" />
      </summary>
      <div className="agent-config-body">{contents}</div>
    </details>
  );
}
