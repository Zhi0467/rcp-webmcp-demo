import {
  ArrowLeft,
  ArrowRight,
  Check,
  CircleAlert,
  Clipboard,
  FolderGit2,
  LoaderCircle,
  LockKeyhole,
  RefreshCw,
  Server,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api";
import {
  advanceDesktopProjectTransfer,
  discardDesktopProjectTransferExport,
  establishDesktopTeamSession,
  exportDesktopProjectTransfer,
  finishDesktopProjectTransfer,
  isDesktopRuntime,
  listDesktopTeamConnections,
  loadDesktopProjectTransfer,
  openDesktopProjectTransferTerminal,
  prepareDesktopProjectTransfer,
  readDesktopTargetProjectProvisioningOptions,
  runDesktopIncomingProjectProvision,
  selectDesktopProjectTransferExport,
  type ProjectTransferBundle,
  type ProjectTransferCommandEvent,
  type ProjectTransferMachineIntent,
  type ProjectTransferProviderIntent,
  type ProjectTransferRunResult,
  type ServerCommandEvent,
  type TargetProviderSetupProjection,
  type TeamConnectionMetadata,
} from "../desktopRuntime";
import {
  modelChange,
  modelOptions,
  providerChange,
  providerOptions,
  reasoningOptions,
  runtimeOptions,
} from "../providers";
import { formatCommandArgv, projectMoveSetupHash, type ProjectSetupRoute } from "../projectSetup";
import type {
  AgentExecutionProfile,
  AgentTask,
  Episode,
  ProjectSnapshot,
  ProviderReadiness,
  SetupAgentProfile,
} from "../types";

type MoveSetupRoute = Extract<ProjectSetupRoute, { kind: "move" }>;

export interface TransferSourceData {
  project: ProjectSnapshot;
  tasks: AgentTask[];
  episodes: Episode[];
}

export interface TransferActiveWorkSummary {
  activeTaskCount: number;
  liveEpisodeCount: number;
  totalCount: number;
}

interface TransferMachineDraft extends ProjectTransferMachineIntent {
  central_root: string | null;
}

const agentProfiles: Array<{ id: AgentExecutionProfile; label: string }> = [
  { id: "seed", label: "Seed" },
  { id: "refresh", label: "Refresh" },
  { id: "node_chat", label: "Node chat" },
  { id: "project_chat", label: "Project chat" },
  { id: "paper_coach", label: "Paper coach" },
  { id: "orchestrator", label: "Orchestrator" },
];

/** Use only backend-published lifecycle answers. */
export function transferActiveWorkSummary(
  tasks: AgentTask[],
  episodes: Episode[],
): TransferActiveWorkSummary {
  const activeTaskCount = tasks.filter((task) => task.active).length;
  const liveEpisodeCount = episodes.filter((episode) => episode.live).length;
  return {
    activeTaskCount,
    liveEpisodeCount,
    totalCount: activeTaskCount + liveEpisodeCount,
  };
}

export function transferTargetIsReady(
  connection: Pick<
    TeamConnectionMetadata,
    "connection_id" | "expected_space_id" | "local_origin" | "operator_route"
  > | null,
): boolean {
  return Boolean(
    connection?.connection_id &&
    connection.expected_space_id &&
    connection.local_origin &&
    connection.operator_route?.ssh_target.trim(),
  );
}

export function transferFinished(bundle: ProjectTransferBundle | null): boolean {
  return Boolean(bundle?.finished);
}

export function transferRelayFailure(relay: ProjectTransferRunResult | null): string | null {
  if (!relay) return null;
  if (relay.exit_code === 0 && relay.proof_verified && relay.cleanup_acknowledged) return null;
  const outcome =
    relay.exit_code !== 0
      ? `the server command exited with code ${relay.exit_code}`
      : !relay.proof_verified
        ? "the target activation proof was not verified"
        : "the target cleanup acknowledgment was not recorded";
  return `Automatic relay failed: ${outcome}. The same transfer remains retryable; use the explicit Manual relay section if the saved operator route needs interaction.`;
}

function asProviderReadiness(providers: TargetProviderSetupProjection[]): ProviderReadiness[] {
  return providers.map((provider) => ({
    ...provider,
    path_state: provider.path_state as ProviderReadiness["path_state"],
  }));
}

function initialMachines(project: ProjectSnapshot): TransferMachineDraft[] {
  return project.machines.map((machine) => ({
    alias: machine.alias,
    location: machine.host ? "ssh" : "local",
    host: machine.host,
    os_account: machine.host ? machine.os_account : "rcp",
    central_root: null,
  }));
}

function canonicalMachine(project: ProjectSnapshot): string {
  return (
    project.repositories.find((repository) => repository.alias === project.state_repository)
      ?.machine ??
    project.machines[0]?.alias ??
    ""
  );
}

function resolvedProfile(
  current: ProjectSnapshot["agent_profiles"][AgentExecutionProfile],
  providers: ProviderReadiness[],
): SetupAgentProfile {
  const selected =
    providers.find(
      (provider) =>
        provider.provider === current.provider && provider.installed && provider.authenticated,
    ) ??
    providers.find((provider) => provider.installed && provider.authenticated) ??
    providers[0];
  if (!selected) {
    return {
      provider: current.provider,
      runtime: current.runtime,
      model: current.model,
      reasoning: current.reasoning,
      location: "local",
      host: "",
    };
  }
  const runtime = selected.runtimes.some((item) => item.id === current.runtime)
    ? current.runtime
    : selected.default_runtime;
  const model = selected.models.some((item) => item.id === current.model)
    ? current.model
    : (selected.models[0]?.id ?? current.model);
  const modelInfo = selected.models.find((item) => item.id === model);
  const reasoning = modelInfo?.reasoning.includes(current.reasoning)
    ? current.reasoning
    : (modelInfo?.default_reasoning ?? current.reasoning);
  return {
    provider: selected.provider,
    runtime,
    model,
    reasoning,
    location: "local",
    host: "",
  };
}

function initialProviderChecks(
  project: ProjectSnapshot,
  providers: TargetProviderSetupProjection[],
): ProjectTransferProviderIntent[] {
  const readiness = asProviderReadiness(providers);
  const canonical = canonicalMachine(project);
  return agentProfiles.map(({ id }) => {
    const resolved = resolvedProfile(project.agent_profiles[id], readiness);
    const requestedMachine = project.agent_profiles[id].run_on;
    return {
      profile: id,
      provider: resolved.provider,
      runtime_id: resolved.runtime,
      model: resolved.model,
      reasoning: resolved.reasoning,
      machine_alias:
        id === "paper_coach" &&
        project.machines.some((machine) => machine.alias === requestedMachine)
          ? requestedMachine
          : canonical,
    };
  });
}

export function TransferProjectSetup({
  route,
  intentChooser,
  onCancel,
}: {
  route: MoveSetupRoute;
  intentChooser: ReactNode;
  onCancel: () => void;
}) {
  const desktop = isDesktopRuntime();
  const routeHasRequestPair = route.sourceRequestId !== null && route.targetRequestId !== null;
  const [step, setStep] = useState(routeHasRequestPair ? 2 : 0);
  const [source, setSource] = useState<TransferSourceData | null>(null);
  const [connections, setConnections] = useState<TeamConnectionMetadata[]>([]);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [targetProviders, setTargetProviders] = useState<TargetProviderSetupProjection[]>([]);
  const [targetName, setTargetName] = useState("");
  const [targetCeiling, setTargetCeiling] = useState(10);
  const [machines, setMachines] = useState<TransferMachineDraft[]>([]);
  const [providerChecks, setProviderChecks] = useState<ProjectTransferProviderIntent[]>([]);
  const [bundle, setBundle] = useState<ProjectTransferBundle | null>(null);
  const [events, setEvents] = useState<Array<ServerCommandEvent | ProjectTransferCommandEvent>>([]);
  const [manualArchivePath, setManualArchivePath] = useState<string | null>(null);
  const [sourceLoading, setSourceLoading] = useState(desktop);
  const [connectionsLoading, setConnectionsLoading] = useState(desktop);
  const [busy, setBusy] = useState<string | null>(routeHasRequestPair ? "resume" : null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!desktop) {
      setSourceLoading(false);
      return;
    }
    let stopped = false;
    const base = `/api/projects/${encodeURIComponent(route.sourceProjectId)}`;
    void Promise.all([
      api<ProjectSnapshot>(base),
      api<AgentTask[]>(`${base}/tasks`),
      api<Episode[]>(`${base}/episodes`),
    ])
      .then(([project, tasks, episodes]) => {
        if (stopped) return;
        if (project.id !== route.sourceProjectId) {
          throw new Error(
            "The personal backend returned a different project than the pinned source.",
          );
        }
        setSource({ project, tasks, episodes });
        setTargetName(project.name);
        setTargetCeiling(project.default_auto_research_invocation_ceiling);
        setMachines(initialMachines(project));
      })
      .catch((caught) => {
        if (!stopped && !routeHasRequestPair) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      })
      .finally(() => {
        if (!stopped) setSourceLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [desktop, route.sourceProjectId, routeHasRequestPair]);

  useEffect(() => {
    if (!desktop) {
      setConnectionsLoading(false);
      return;
    }
    let stopped = false;
    void listDesktopTeamConnections()
      .then((saved) => {
        if (!stopped) setConnections(saved);
      })
      .catch((caught) => {
        if (!stopped) setError(caught instanceof Error ? caught.message : String(caught));
      })
      .finally(() => {
        if (!stopped) setConnectionsLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [desktop]);

  useEffect(() => {
    if (!desktop || !routeHasRequestPair || !route.sourceRequestId) return;
    let stopped = false;
    setBusy("resume");
    void loadDesktopProjectTransfer(route.sourceRequestId)
      .then((loaded) => {
        if (stopped) return;
        setBundle(loaded);
        setTargetProviders(loaded.target_provider_setup);
        setTargetName(loaded.incoming_provisioning.name ?? "");
        setTargetCeiling(loaded.incoming_provisioning.default_auto_research_invocation_ceiling);
        setMachines(
          loaded.incoming_provisioning.machines.map((machine) => ({
            alias: machine.alias,
            location: machine.location,
            host: machine.host,
            os_account: machine.os_account,
            central_root: machine.intended_central_root,
          })),
        );
        setProviderChecks(loaded.incoming_provisioning.provider_checks);
        setStep(2);
      })
      .catch((caught) => {
        if (!stopped) setError(caught instanceof Error ? caught.message : String(caught));
      })
      .finally(() => {
        if (!stopped) setBusy(null);
      });
    return () => {
      stopped = true;
    };
  }, [desktop, route.sourceRequestId, routeHasRequestPair]);

  useEffect(() => {
    if (!bundle || connections.length === 0 || selectedConnectionId) return;
    const target = connections.find(
      (connection) => connection.expected_space_id === bundle.target.target_space_id,
    );
    if (target) setSelectedConnectionId(target.connection_id);
  }, [bundle, connections, selectedConnectionId]);

  const selectedConnection = useMemo(
    () =>
      connections.find((connection) => connection.connection_id === selectedConnectionId) ?? null,
    [connections, selectedConnectionId],
  );
  const targetReady = transferTargetIsReady(selectedConnection);
  const activeWork = source ? transferActiveWorkSummary(source.tasks, source.episodes) : null;
  const providers = asProviderReadiness(targetProviders);
  const complete = transferFinished(bundle);

  async function chooseTarget(): Promise<void> {
    if (!source || !selectedConnection || !targetReady) {
      setError("Select one saved team target with an established operator route.");
      return;
    }
    if (activeWork?.totalCount) {
      setError("Settle the active tasks and episodes before preparing this transfer.");
      return;
    }
    setBusy("target");
    setError(null);
    try {
      const established = await establishDesktopTeamSession(selectedConnection.connection_id);
      if (
        established.connection.connection_id !== selectedConnection.connection_id ||
        established.connection.expected_space_id !== selectedConnection.expected_space_id ||
        established.identity.space_id !== selectedConnection.expected_space_id
      ) {
        throw new Error("The established session did not match the selected team space.");
      }
      const known = await readDesktopTargetProjectProvisioningOptions(
        selectedConnection.connection_id,
      );
      setTargetProviders(known);
      setProviderChecks(initialProviderChecks(source.project, known));
      setStep(1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  function validateTargetConfiguration(): string | null {
    if (!source || !selectedConnection) return "The source and target must be loaded.";
    if (!targetName.trim()) return "Give the team project a name.";
    if (!Number.isSafeInteger(targetCeiling) || targetCeiling < 1) {
      return "Set the auto-research ceiling to at least 1 operational invocation.";
    }
    if (machines.length !== source.project.machines.length) {
      return "Every historical machine alias needs one target placement.";
    }
    if (machines.some((machine) => !machine.os_account.trim())) {
      return "Every target machine needs its exact Linux account.";
    }
    if (machines.some((machine) => machine.location === "ssh" && !machine.host?.trim())) {
      return "Every SSH target machine needs its exact host.";
    }
    if (providerChecks.length !== agentProfiles.length) {
      return "Every agent role needs a target provider configuration.";
    }
    if (providerChecks.some((profile) => !profile.provider || !profile.runtime_id)) {
      return "Choose an available provider and runtime for every agent role.";
    }
    return null;
  }

  async function prepareTransfer(): Promise<void> {
    const problem = validateTargetConfiguration();
    if (problem || !selectedConnection || !source) {
      setError(problem ?? "The transfer configuration is incomplete.");
      return;
    }
    const sourceRequestId = crypto.randomUUID();
    const targetRequestId = crypto.randomUUID();
    const hash = projectMoveSetupHash({
      sourceProjectId: route.sourceProjectId,
      sourceRequestId,
      targetRequestId,
    });
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}${hash}`,
    );
    setBusy("prepare");
    setError(null);
    try {
      const prepared = await prepareDesktopProjectTransfer({
        sourceRequestId,
        targetRequestId,
        connectionId: selectedConnection.connection_id,
        sourceProjectId: route.sourceProjectId,
        targetProvisioning: {
          name: targetName.trim(),
          default_auto_research_invocation_ceiling: targetCeiling,
          machines: machines.map((machine) => ({
            ...machine,
            host: machine.location === "ssh" ? machine.host?.trim() : "",
            os_account: machine.location === "local" ? "rcp" : machine.os_account.trim(),
            central_root: machine.central_root?.trim() || null,
          })),
          provider_checks: providerChecks,
        },
      });
      setBundle(prepared);
      setStep(2);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function refreshTransfer(): Promise<void> {
    const sourceRequestId = bundle?.source.request_id ?? route.sourceRequestId;
    if (!sourceRequestId) return;
    setBusy("refresh");
    setError(null);
    try {
      setBundle(await loadDesktopProjectTransfer(sourceRequestId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function runTargetSetup(): Promise<void> {
    if (!bundle) return;
    setBusy("setup");
    setError(null);
    setEvents([]);
    try {
      await runDesktopIncomingProjectProvision(bundle.source.request_id, (event) =>
        setEvents((current) => [...current, event]),
      );
      setBundle(await loadDesktopProjectTransfer(bundle.source.request_id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function advanceTransfer(): Promise<void> {
    if (!bundle?.can_advance) return;
    setBusy("advance");
    setError(null);
    setEvents([]);
    try {
      const advanced = await advanceDesktopProjectTransfer(bundle.source.request_id, (event) =>
        setEvents((current) => [...current, event]),
      );
      setBundle(advanced.bundle);
      setError(transferRelayFailure(advanced.relay));
    } catch (caught) {
      const transitionError = caught instanceof Error ? caught.message : String(caught);
      try {
        setBundle(await loadDesktopProjectTransfer(bundle.source.request_id));
        setError(transitionError);
      } catch (refreshCaught) {
        const refreshError =
          refreshCaught instanceof Error ? refreshCaught.message : String(refreshCaught);
        setError(`${transitionError} State refresh also failed: ${refreshError}`);
      }
    } finally {
      setBusy(null);
    }
  }

  async function startManualRelay(): Promise<void> {
    if (!bundle?.can_manual_relay) return;
    setBusy("manual-export");
    setError(null);
    try {
      const exported = await exportDesktopProjectTransfer(bundle.source.request_id);
      if (!exported.saved) return;
      if (!exported.path) {
        throw new Error("The protected transfer export did not return its saved path.");
      }
      setManualArchivePath(exported.path);
      await openDesktopProjectTransferTerminal(bundle.source.request_id, exported.path);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function resumeManualRelay(): Promise<void> {
    if (!bundle?.can_manual_relay) return;
    setBusy("manual-select");
    setError(null);
    try {
      const selected = await selectDesktopProjectTransferExport(bundle.source.request_id);
      if (!selected.selected) return;
      if (!selected.path) {
        throw new Error("The selected transfer export did not return its verified path.");
      }
      setManualArchivePath(selected.path);
      await openDesktopProjectTransferTerminal(bundle.source.request_id, selected.path);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function reopenManualRelay(): Promise<void> {
    if (!bundle || !manualArchivePath) return;
    setBusy("manual-terminal");
    setError(null);
    try {
      await openDesktopProjectTransferTerminal(bundle.source.request_id, manualArchivePath);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function finishManualRelay(): Promise<void> {
    if (!bundle || !manualArchivePath) return;
    setBusy("manual-finish");
    setError(null);
    try {
      await finishDesktopProjectTransfer(bundle.source.request_id, manualArchivePath);
      setManualArchivePath(null);
      setBundle(await loadDesktopProjectTransfer(bundle.source.request_id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function discardManualRelay(): Promise<void> {
    if (!bundle || !manualArchivePath) return;
    setBusy("manual-discard");
    setError(null);
    try {
      await discardDesktopProjectTransferExport(bundle.source.request_id, manualArchivePath);
      setManualArchivePath(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function copyOperatorCommand(): Promise<void> {
    if (!bundle?.incoming_provisioning.operator_argv.length) return;
    try {
      await navigator.clipboard.writeText(
        formatCommandArgv(bundle.incoming_provisioning.operator_argv),
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  function updateProvider(
    profile: AgentExecutionProfile,
    patch: Partial<ProjectTransferProviderIntent>,
  ) {
    setProviderChecks((current) =>
      current.map((item) => (item.profile === profile ? { ...item, ...patch } : item)),
    );
    setError(null);
  }

  if (!desktop) {
    return (
      <main className="setup-layout transfer-setup-layout">
        <section className="setup-sheet transfer-unavailable-sheet">
          <div className="setup-section transfer-unavailable" role="alert">
            <CircleAlert size={22} aria-hidden="true" />
            <SectionHeading
              eyebrow="Move to team space"
              title="This move is unavailable in a browser."
            />
            <p>
              Moving a personal project needs the source-built desktop app and its protected saved
              team connection.
            </p>
            <button className="button secondary" type="button" onClick={onCancel}>
              <ArrowLeft size={15} /> Return to projects
            </button>
          </div>
        </section>
      </main>
    );
  }

  const loading = sourceLoading || connectionsLoading || busy === "resume";
  const sourceName = source?.project.name ?? "Personal project";

  return (
    <main className="setup-layout transfer-setup-layout">
      <nav className="setup-steps" aria-label="Project move progress">
        {[
          ["01", "Source & target"],
          ["02", "Target configuration"],
          ["03", "Setup & transfer"],
        ].map(([number, label], index) => (
          <button
            className={
              index === step
                ? "setup-step active"
                : index < step
                  ? "setup-step complete"
                  : "setup-step"
            }
            type="button"
            disabled={index > step || bundle !== null}
            key={number}
            onClick={() => index < step && !bundle && setStep(index)}
          >
            <span>{index < step ? <Check size={13} /> : number}</span>
            <strong>{label}</strong>
          </button>
        ))}
      </nav>

      <section className="setup-sheet">
        {(loading || step === 0) && (
          <div className="setup-section transfer-route-context">
            {intentChooser}
            <div className="transfer-source-pin">
              <LockKeyhole size={14} aria-hidden="true" />
              <span>Source project pinned</span>
              <code>{route.sourceProjectId}</code>
            </div>
          </div>
        )}
        {loading && (
          <div className="setup-section setup-resume-loading" role="status">
            <LoaderCircle className="spin" size={18} />
            <strong>Reading the personal project and transfer state</strong>
          </div>
        )}

        {!loading && step === 0 && (
          <div className="setup-section transfer-section">
            <SectionHeading
              eyebrow="Move an existing personal project to a team"
              title={`Choose the new home for ${sourceName}.`}
            />
            {source && (
              <>
                <section className="transfer-card" aria-labelledby="transfer-source-title">
                  <header>
                    <FolderGit2 size={16} aria-hidden="true" />
                    <h2 id="transfer-source-title">Personal working copies stay in place</h2>
                  </header>
                  <div className="transfer-path-list">
                    {source.project.repositories.map((repository) => {
                      const machine = source.project.machines.find(
                        (candidate) => candidate.alias === repository.machine,
                      );
                      return (
                        <div
                          className="transfer-path"
                          key={`${repository.alias}:${repository.path}`}
                        >
                          <span>
                            {repository.alias} · {machine?.os_account ?? "current user"}
                          </span>
                          <code>{repository.path}</code>
                        </div>
                      );
                    })}
                  </div>
                </section>
                <section
                  className="transfer-card transfer-work-card"
                  aria-labelledby="transfer-work-title"
                >
                  <header>
                    <TriangleAlert size={16} aria-hidden="true" />
                    <h2 id="transfer-work-title">Active work must settle before release</h2>
                  </header>
                  <div className="transfer-work-counts">
                    <div>
                      <strong>{activeWork?.activeTaskCount ?? 0}</strong>
                      <span>active agent tasks</span>
                    </div>
                    <div>
                      <strong>{activeWork?.liveEpisodeCount ?? 0}</strong>
                      <span>live episodes</span>
                    </div>
                  </div>
                </section>
              </>
            )}

            <fieldset className="transfer-target-picker">
              <legend>Saved team target</legend>
              {connections.map((connection) => {
                const ready = transferTargetIsReady(connection);
                return (
                  <label className="transfer-target-option" key={connection.connection_id}>
                    <input
                      type="radio"
                      name="project-transfer-target"
                      checked={selectedConnectionId === connection.connection_id}
                      onChange={() => setSelectedConnectionId(connection.connection_id)}
                    />
                    <span className="transfer-target-copy">
                      <strong>{connection.display_name}</strong>
                      <span>{connection.ssh_target}</span>
                      <span className={ready ? "transfer-target-ready" : "transfer-target-missing"}>
                        {ready ? "Authenticated operator route ready" : "Operator route required"}
                      </span>
                    </span>
                  </label>
                );
              })}
              {!connections.length && (
                <div className="transfer-target-empty" role="alert">
                  No saved team target is available.
                </div>
              )}
            </fieldset>
          </div>
        )}

        {!loading && step === 1 && source && (
          <div className="setup-section transfer-section">
            <SectionHeading
              eyebrow="Team-owned execution"
              title="Review the central machines and provider accounts."
            />
            <label className="setup-field">
              <span>Team project name</span>
              <input value={targetName} onChange={(event) => setTargetName(event.target.value)} />
            </label>
            <label className="agent-auto-research-default">
              <span>Default auto-research ceiling</span>
              <input
                type="number"
                min={1}
                value={targetCeiling}
                onChange={(event) => setTargetCeiling(Number(event.target.value))}
              />
            </label>
            <div className="team-machine-stack">
              {machines.map((machine) => (
                <article className="team-machine-editor" key={machine.alias}>
                  <header>
                    <Server size={16} />
                    <strong>{machine.alias}</strong>
                  </header>
                  <div className="team-machine-fields">
                    <label>
                      Location
                      <select
                        value={machine.location}
                        onChange={(event) =>
                          setMachines((current) =>
                            current.map((item) =>
                              item.alias === machine.alias
                                ? {
                                    ...item,
                                    location: event.target.value as "local" | "ssh",
                                    host: event.target.value === "local" ? "" : item.host,
                                    os_account:
                                      event.target.value === "local" ? "rcp" : item.os_account,
                                  }
                                : item,
                            ),
                          )
                        }
                      >
                        <option value="local">RCP server</option>
                        <option value="ssh">SSH machine</option>
                      </select>
                    </label>
                    {machine.location === "ssh" && (
                      <label>
                        SSH host
                        <input
                          value={machine.host ?? ""}
                          onChange={(event) =>
                            setMachines((current) =>
                              current.map((item) =>
                                item.alias === machine.alias
                                  ? { ...item, host: event.target.value }
                                  : item,
                              ),
                            )
                          }
                        />
                      </label>
                    )}
                    <label>
                      Linux account
                      <input
                        value={machine.os_account}
                        disabled={machine.location === "local"}
                        onChange={(event) =>
                          setMachines((current) =>
                            current.map((item) =>
                              item.alias === machine.alias
                                ? { ...item, os_account: event.target.value }
                                : item,
                            ),
                          )
                        }
                      />
                    </label>
                    <label className="wide">
                      Central root
                      <input
                        value={machine.central_root ?? ""}
                        placeholder={
                          machine.location === "local"
                            ? "/var/lib/rcp/projects"
                            : "Absolute target path"
                        }
                        onChange={(event) =>
                          setMachines((current) =>
                            current.map((item) =>
                              item.alias === machine.alias
                                ? { ...item, central_root: event.target.value }
                                : item,
                            ),
                          )
                        }
                      />
                    </label>
                  </div>
                </article>
              ))}
            </div>
            <div className="agent-role-stack">
              {agentProfiles.map(({ id, label }) => {
                const profile = providerChecks.find((item) => item.profile === id);
                if (!profile) return null;
                const readiness = providers.find((item) => item.provider === profile.provider);
                const models = readiness?.models ?? [];
                return (
                  <article className="agent-role-card" key={id}>
                    <header>
                      <strong>{label}</strong>
                      <span className="role-permission">{profile.machine_alias}</span>
                    </header>
                    <div className="agent-role-fields">
                      <label>
                        Provider
                        <select
                          value={profile.provider}
                          onChange={(event) => {
                            const next = providers.find(
                              (item) => item.provider === event.target.value,
                            );
                            updateProvider(id, {
                              ...providerChange(
                                next?.models ?? [],
                                event.target.value,
                                profile.reasoning,
                              ),
                              runtime_id: next?.default_runtime ?? "",
                            });
                          }}
                        >
                          {providerOptions(providers, profile.provider).map((option) => (
                            <option value={option.id} key={option.id}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Runtime
                        <select
                          value={profile.runtime_id}
                          onChange={(event) =>
                            updateProvider(id, { runtime_id: event.target.value })
                          }
                        >
                          {runtimeOptions(readiness, profile.runtime_id).map((option) => (
                            <option value={option.id} key={option.id}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Model
                        <select
                          value={profile.model}
                          onChange={(event) =>
                            updateProvider(
                              id,
                              modelChange(models, event.target.value, profile.reasoning),
                            )
                          }
                        >
                          {modelOptions(models, profile.model).map((option) => (
                            <option value={option.id} key={option.id}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Reasoning
                        <select
                          value={profile.reasoning}
                          onChange={(event) =>
                            updateProvider(id, { reasoning: event.target.value })
                          }
                        >
                          {reasoningOptions(models, profile.model, profile.reasoning).map(
                            (option) => (
                              <option value={option.id} key={option.id}>
                                {option.label}
                              </option>
                            ),
                          )}
                        </select>
                      </label>
                    </div>
                  </article>
                );
              })}
            </div>
          </div>
        )}

        {!loading && step === 2 && bundle && (
          <div className="setup-section transfer-section">
            <SectionHeading
              eyebrow="Durable transfer"
              title={bundle.incoming_provisioning.name ?? sourceName}
            />
            <div className="provisioning-status-banner">
              <strong>
                {complete ? "Transfer complete" : bundle.incoming_provisioning.status_label}
              </strong>
              {!complete && bundle.target.next_action && <span>{bundle.target.next_action}</span>}
            </div>
            {bundle.incoming_provisioning.diagnostic && (
              <div className="setup-error" role="alert">
                <TriangleAlert size={16} />
                <span>{bundle.incoming_provisioning.diagnostic}</span>
              </div>
            )}
            <dl className="provisioning-summary">
              <div>
                <dt>Target project</dt>
                <dd>{bundle.target.project_id}</dd>
              </div>
              <div>
                <dt>Authorized by</dt>
                <dd>{bundle.incoming_provisioning.authorized_by.display_name}</dd>
              </div>
              <div>
                <dt>Readiness</dt>
                <dd>
                  {bundle.incoming_provisioning.readiness.repositories_ready}/
                  {bundle.incoming_provisioning.readiness.repositories_total} repositories ·{" "}
                  {bundle.incoming_provisioning.readiness.providers_ready}/
                  {bundle.incoming_provisioning.readiness.providers_total} provider roles
                </dd>
              </div>
            </dl>
            <section className="transfer-card">
              <header>
                <Server size={16} />
                <h2>Team central paths</h2>
              </header>
              <div className="transfer-path-list">
                {bundle.incoming_provisioning.repositories.map((repository) => (
                  <div className="transfer-path" key={repository.alias}>
                    <span>
                      {repository.alias} · {repository.machine_alias}
                    </span>
                    <code>
                      {repository.resolved_path ??
                        repository.intended_path ??
                        "Pending server setup"}
                    </code>
                  </div>
                ))}
              </div>
            </section>
            {bundle.incoming_provisioning.final_review && (
              <section className="provisioning-final-review">
                <h2>Final review</h2>
                <p>
                  This one confirmation admits the prepared team copy, makes the personal project
                  read-only, relays the sealed history, and activates the team project.
                </p>
                <dl>
                  <div>
                    <dt>Project id</dt>
                    <dd>{bundle.incoming_provisioning.final_review.proposed_project_id}</dd>
                  </div>
                  <div>
                    <dt>Review binding</dt>
                    <dd>
                      <code>{bundle.incoming_provisioning.final_review.digest}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Prepared for</dt>
                    <dd>{bundle.incoming_provisioning.final_review.authorized_by.display_name}</dd>
                  </div>
                </dl>
              </section>
            )}
            <section className="transfer-card">
              <header>
                <FolderGit2 size={16} />
                <h2>Archive boundary</h2>
              </header>
              <div className="transfer-archive-policy">
                <p>
                  <strong>Included:</strong> canonical history, finished RCP activity, Paper state,
                  facts, referenced kept files, and best-effort matched complete provider histories.
                </p>
                <p>
                  <strong>Excluded:</strong> source checkout bytes, credentials, provider
                  authentication, live work, reusable stages, caches, temporary inputs, and unkept
                  artifact bytes.
                </p>
              </div>
            </section>
            {bundle.incoming_provisioning.operator_action && (
              <section className="operator-route-card">
                <header>
                  <strong>{bundle.incoming_provisioning.operator_action.title}</strong>
                  <span>{bundle.incoming_provisioning.operator_action.performed_by}</span>
                </header>
                <p>{bundle.incoming_provisioning.operator_action.message}</p>
                <p>
                  <strong>Success:</strong>{" "}
                  {bundle.incoming_provisioning.operator_action.expected_success}
                </p>
              </section>
            )}
            {bundle.can_manual_relay && (
              <section className="operator-route-card">
                <header>
                  <strong>Manual relay</strong>
                  <span>Explicit fallback</span>
                </header>
                <p>
                  Save the exact protected archive and open the fixed import command in Terminal.
                  RCP will not switch to this path automatically.
                </p>
                {manualArchivePath && <code>{manualArchivePath}</code>}
                <div className="provisioning-controls">
                  {!manualArchivePath ? (
                    <>
                      <button
                        className="button secondary"
                        type="button"
                        disabled={busy !== null}
                        onClick={() => void startManualRelay()}
                      >
                        {busy === "manual-export" ? (
                          <LoaderCircle className="spin" size={14} />
                        ) : (
                          <Server size={14} />
                        )}{" "}
                        Save archive and open Terminal
                      </button>
                      <button
                        className="button secondary"
                        type="button"
                        disabled={busy !== null}
                        onClick={() => void resumeManualRelay()}
                      >
                        {busy === "manual-select" ? (
                          <LoaderCircle className="spin" size={14} />
                        ) : (
                          <RefreshCw size={14} />
                        )}{" "}
                        Resume saved archive
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        className="button secondary"
                        type="button"
                        disabled={busy !== null}
                        onClick={() => void reopenManualRelay()}
                      >
                        Open command again
                      </button>
                      <button
                        className="button primary"
                        type="button"
                        disabled={busy !== null}
                        onClick={() => void finishManualRelay()}
                      >
                        {busy === "manual-finish" ? (
                          <LoaderCircle className="spin" size={14} />
                        ) : (
                          <Check size={14} />
                        )}{" "}
                        Check target and finish
                      </button>
                      <button
                        className="button secondary"
                        type="button"
                        disabled={busy !== null}
                        onClick={() => void discardManualRelay()}
                      >
                        Delete saved copy
                      </button>
                    </>
                  )}
                </div>
              </section>
            )}
            {events.length > 0 && (
              <div className="provisioning-event-log" role="log">
                {events.map((event, index) => (
                  <div key={`${event.event}-${index}`}>
                    <strong>{event.event}</strong>
                    <span>{"step" in event ? event.step.message : "Server plan received"}</span>
                  </div>
                ))}
              </div>
            )}
            <div className="provisioning-controls">
              <button
                className="button secondary"
                type="button"
                disabled={busy !== null}
                onClick={() => void refreshTransfer()}
              >
                {busy === "refresh" ? (
                  <LoaderCircle className="spin" size={14} />
                ) : (
                  <RefreshCw size={14} />
                )}{" "}
                Refresh
              </button>
              {bundle.incoming_provisioning.operator_argv.length > 0 && (
                <button
                  className="button secondary"
                  type="button"
                  onClick={() => void copyOperatorCommand()}
                >
                  <Clipboard size={14} /> Copy server command
                </button>
              )}
              {bundle.incoming_provisioning.can_run_setup && (
                <button
                  className="button primary"
                  type="button"
                  disabled={busy !== null}
                  onClick={() => void runTargetSetup()}
                >
                  {busy === "setup" ? (
                    <LoaderCircle className="spin" size={14} />
                  ) : (
                    <Server size={14} />
                  )}{" "}
                  Run target setup
                </button>
              )}
              {bundle.can_advance && (
                <button
                  className="button primary"
                  type="button"
                  disabled={busy !== null}
                  onClick={() => void advanceTransfer()}
                >
                  {busy === "advance" ? (
                    <LoaderCircle className="spin" size={14} />
                  ) : (
                    <ShieldCheck size={14} />
                  )}{" "}
                  {bundle.advance_label ?? "Continue project move"}
                </button>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="setup-error" role="alert">
            <TriangleAlert size={16} />
            <span>{error}</span>
          </div>
        )}
        {!loading && (
          <footer className="setup-actions">
            <button
              className="button secondary"
              type="button"
              disabled={busy !== null}
              onClick={() => (step === 0 || bundle ? onCancel() : setStep(step - 1))}
            >
              <ArrowLeft size={15} /> Back
            </button>
            {step === 0 && (
              <button
                className="button primary"
                type="button"
                disabled={
                  busy !== null || !source || !targetReady || Boolean(activeWork?.totalCount)
                }
                onClick={() => void chooseTarget()}
              >
                {busy === "target" ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <ArrowRight size={15} />
                )}{" "}
                Continue
              </button>
            )}
            {step === 1 && (
              <button
                className="button primary"
                type="button"
                disabled={busy !== null}
                onClick={() => void prepareTransfer()}
              >
                {busy === "prepare" ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <ShieldCheck size={15} />
                )}{" "}
                Prepare team target
              </button>
            )}
          </footer>
        )}
      </section>

      {!bundle && (
        <aside className="boundary-ledger transfer-ledger">
          <span className="eyebrow">Move boundary</span>
          <h2>Ownership after transfer</h2>
          <LedgerItem number="A" label="Personal" value="Original working copies stay put" />
          <LedgerItem number="B" label="Team" value="Project identity and canonical history" />
          <LedgerItem number="C" label="Git" value="Fresh central checkouts from GitHub" />
          <LedgerItem number="D" label="Providers" value="Existing auth on target accounts" />
        </aside>
      )}
    </main>
  );
}

function SectionHeading({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <header className="setup-section-heading">
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
    </header>
  );
}

function LedgerItem({ number, label, value }: { number: string; label: string; value: string }) {
  return (
    <div className="ledger-item">
      <span>{number}</span>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
      </div>
    </div>
  );
}
