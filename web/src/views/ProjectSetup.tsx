import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  FileCode2,
  FolderGit2,
  FolderOpen,
  LoaderCircle,
  LockKeyhole,
  Plus,
  ShieldCheck,
  Trash2,
  TriangleAlert,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../api";
import { chooseDesktopRepositoryFolder, isDesktopRuntime } from "../desktopRuntime";
import {
  modelChange,
  modelOptions,
  providerChange,
  providerOptions,
  readinessFor,
  reasoningOptions,
  runtimeOptions,
} from "../providers";
import {
  assertSupportedProjectCreationIntent,
  repositoryPickerPresentation,
  selectedProjectCreationIntent,
  stateRepositoryAfterRemoval,
  type ProjectSetupRoute,
} from "../projectSetup";
import { TeamProjectSetup } from "./TeamProjectSetup";
import { TransferProjectSetup } from "./TransferProjectSetup";
import type {
  AgentExecutionProfile,
  ExistingResearchAction,
  ProviderReadiness,
  ProjectCard,
  ProjectCreationControl,
  ProjectCreationIntent,
  ProjectSetupRequest,
  SetupAgentProfile,
  SetupAgents,
  SetupPreview,
  SetupRepository,
} from "../types";

interface Props {
  projectCreation: ProjectCreationControl;
  onCancel: () => void;
  onCreated: (projectId: string) => void;
  setupRoute: ProjectSetupRoute;
}

interface DraftRepository extends SetupRepository {
  id: number;
}

const steps = [
  ["01", "Project"],
  ["02", "Truth boundary"],
  ["03", "Agent roles"],
  ["04", "Verify"],
] as const;

let repositorySequence = 1;

const agentExecutionProfiles: Array<{ id: AgentExecutionProfile; label: string }> = [
  { id: "seed", label: "Seed" },
  { id: "refresh", label: "Refresh" },
  { id: "node_chat", label: "Node chat" },
  { id: "project_chat", label: "Project chat" },
  { id: "paper_coach", label: "Paper coach" },
  { id: "orchestrator", label: "Orchestrator" },
];

// The provider is filled from the registry once it answers; the backend lists
// its default first. Hardcoding one here is what this whole change removes.
const defaultAgentProfile = (model = ""): SetupAgentProfile => ({
  provider: "",
  runtime: "",
  model,
  reasoning: "medium",
  location: "local",
  host: "",
});

export function ProjectSetup({
  projectCreation,
  onCancel,
  onCreated,
  setupRoute = { kind: "create", requestId: null },
}: Props) {
  const routeIntent =
    setupRoute.kind === "move" ? ("move_personal_project_to_team" as const) : null;
  const [selectedIntent, setSelectedIntent] = useState(() => {
    if (routeIntent) return routeIntent;
    try {
      return selectedProjectCreationIntent(projectCreation);
    } catch {
      return projectCreation.intents[0]?.intent ?? "use_existing_checkout_personally";
    }
  });
  const intent = routeIntent ?? selectedIntent;
  let setupError: string | null = null;
  if (setupRoute.kind === "invalid") {
    setupError =
      setupRoute.reason === "invalid_move_route"
        ? "This move setup link is invalid. It needs one pinned source project and, when resumed, one complete request pair."
        : "This project setup link has an invalid provisioning request identity.";
  } else {
    try {
      assertSupportedProjectCreationIntent(projectCreation, intent);
    } catch (caught) {
      setupError = caught instanceof Error ? caught.message : String(caught);
    }
  }
  const intentChooser = (
    <ProjectIntentChooser
      control={projectCreation}
      selected={intent}
      locked={routeIntent !== null}
      onSelect={setSelectedIntent}
    />
  );
  const shellClass =
    intent === "create_shared_team_project"
      ? "setup-shell team-project-setup"
      : intent === "move_personal_project_to_team"
        ? "setup-shell transfer-project-setup"
        : "setup-shell";
  return (
    <div className={shellClass}>
      <header className="setup-header">
        <button
          className="rcp-mark setup-brand"
          onClick={onCancel}
          aria-label="Return to project index"
        >
          <span className="rcp-wordmark" aria-hidden="true">
            RCP
          </span>
        </button>
        <span className="setup-header-title">
          {intent === "move_personal_project_to_team" ? "Move project" : "Add project"}
        </span>
        <button className="button ghost" onClick={onCancel}>
          Cancel
        </button>
      </header>
      {setupError ? (
        <SetupRouteFailure message={setupError} onCancel={onCancel} />
      ) : intent === "move_personal_project_to_team" ? (
        <TransferProjectSetup
          route={setupRoute as Extract<ProjectSetupRoute, { kind: "move" }>}
          intentChooser={intentChooser}
          onCancel={onCancel}
        />
      ) : intent === "create_shared_team_project" ? (
        <TeamProjectSetup intentChooser={intentChooser} onCancel={onCancel} onCreated={onCreated} />
      ) : (
        <PersonalProjectSetup
          intentChooser={intentChooser}
          onCancel={onCancel}
          onCreated={onCreated}
        />
      )}
    </div>
  );
}

function PersonalProjectSetup({
  intentChooser,
  onCancel,
  onCreated,
}: Omit<Props, "projectCreation" | "setupRoute"> & { intentChooser: ReactNode }) {
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [repositories, setRepositories] = useState<DraftRepository[]>([
    {
      id: repositorySequence,
      alias: "research",
      location: "local",
      path: "",
      host: "",
      default_read: true,
    },
  ]);
  const [stateRepository, setStateRepository] = useState("research");
  const [agents, setAgents] = useState<SetupAgents>({
    seed: defaultAgentProfile(),
    refresh: defaultAgentProfile(),
    node_chat: defaultAgentProfile(),
    project_chat: defaultAgentProfile(),
    paper_coach: defaultAgentProfile("gpt-5.6-luna"),
    orchestrator: defaultAgentProfile(),
  });
  const [defaultAutoResearchInvocationCeiling, setDefaultAutoResearchInvocationCeiling] =
    useState(10);
  const [preview, setPreview] = useState<SetupPreview | null>(null);
  const [existingResearchOpen, setExistingResearchOpen] = useState(false);
  // Agent defaults are chosen before any manifest exists, so there is no
  // per-machine readiness to read yet; ask the registry what this machine has.
  const [providers, setProviders] = useState<ProviderReadiness[]>([]);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState<"preflight" | "create" | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api<ProviderReadiness[]>("/api/providers")
      .then((known) => {
        setProviders(known);
        const fallback = known[0]?.provider;
        if (!fallback) return;
        setAgents(
          (current) =>
            Object.fromEntries(
              Object.entries(current).map(([surface, profile]) => [
                surface,
                known.some((item) => item.provider === profile.provider)
                  ? profile
                  : {
                      ...profile,
                      provider: fallback,
                      runtime: readinessFor(known, fallback)?.default_runtime ?? "",
                    },
              ]),
            ) as SetupAgents,
        );
      })
      .catch(() => setProviders([]));
  }, []);

  const remoteHosts = useMemo(
    () => [
      ...new Set(
        repositories
          .filter((repo) => repo.location === "ssh")
          .map((repo) => repo.host.trim())
          .filter(Boolean),
      ),
    ],
    [repositories],
  );
  const stateRepo = repositories.find((repo) => repo.alias === stateRepository) ?? repositories[0];
  const defaultRead = repositories.filter((repo) => repo.default_read);
  const canonicalExecution = {
    location: stateRepo.location,
    host: stateRepo.location === "ssh" ? stateRepo.host.trim() : "",
  } as const;
  const payload = (): ProjectSetupRequest => ({
    name: name.trim(),
    repositories: repositories.map(({ id: _id, ...repo }) => ({
      ...repo,
      alias: repo.alias.trim(),
      path: repo.path.trim(),
      host: repo.location === "ssh" ? repo.host.trim() : "",
    })),
    state_repository: stateRepository,
    default_auto_research_invocation_ceiling: defaultAutoResearchInvocationCeiling,
    execution: canonicalExecution,
    agents: Object.fromEntries(
      agentExecutionProfiles.map(({ id }) => [
        id,
        {
          ...agents[id],
          ...(id === "paper_coach"
            ? {
                host: agents[id].location === "ssh" ? agents[id].host.trim() : "",
              }
            : canonicalExecution),
        },
      ]),
    ) as SetupAgents,
    confirmed: false,
  });

  const updateRepository = (id: number, patch: Partial<SetupRepository>) => {
    const currentRepository = repositories.find((repo) => repo.id === id);
    if (patch.alias !== undefined && currentRepository?.alias === stateRepository) {
      setStateRepository(patch.alias);
    }
    setRepositories((current) =>
      current.map((repo) => {
        if (repo.id !== id) return repo;
        const next = { ...repo, ...patch };
        if (patch.location === "local") next.host = "";
        return next;
      }),
    );
    setError(null);
  };

  const validate = (targetStep: number): string | null => {
    if (!name.trim()) return "Give this paper-project a name.";
    if (!repositories[0].path.trim()) return "Enter the first repository's absolute path.";
    if (repositories[0].location === "ssh" && !repositories[0].host.trim()) {
      return "Enter the SSH host for the first repository.";
    }
    if (targetStep < 1) return null;
    const aliases = repositories.map((repo) => repo.alias.trim());
    if (aliases.some((alias) => !/^[a-z][a-z0-9-]{0,47}$/.test(alias))) {
      return "Aliases must start with a lowercase letter and use only lowercase letters, numbers, or hyphens.";
    }
    if (new Set(aliases).size !== aliases.length) return "Each repository needs a unique alias.";
    if (repositories.some((repo) => !repo.path.trim()))
      return "Every repository needs an absolute path.";
    if (repositories.some((repo) => repo.location === "ssh" && !repo.host.trim())) {
      return "Every SSH repository needs a host.";
    }
    if (!repositories.some((repo) => repo.default_read)) {
      return "Select at least one repository for default agent reads.";
    }
    if (!aliases.includes(stateRepository)) return "Choose a canonical state repository.";
    if (targetStep < 2) return null;
    if (
      !Number.isSafeInteger(defaultAutoResearchInvocationCeiling) ||
      defaultAutoResearchInvocationCeiling < 1
    ) {
      return "Set the default auto-research ceiling to at least 1 operational invocation.";
    }
    const invalidAgent = agentExecutionProfiles.find(
      ({ id }) =>
        id === "paper_coach" &&
        agents[id].location === "ssh" &&
        !remoteHosts.includes(agents[id].host),
    );
    if (invalidAgent) {
      return `Choose one of the repository hosts for the ${invalidAgent.label} agent.`;
    }
    return null;
  };

  const updateAgent = (surface: AgentExecutionProfile, patch: Partial<SetupAgentProfile>) => {
    setAgents((current) => {
      const next = { ...current[surface], ...patch };
      if (patch.location === "local") next.host = "";
      return { ...current, [surface]: next };
    });
    setError(null);
  };

  const advance = async () => {
    const problem = validate(step);
    if (problem) {
      setError(problem);
      return;
    }
    if (step < 2) {
      setStep((current) => current + 1);
      setError(null);
      return;
    }
    setBusy("preflight");
    setError(null);
    setPreview(null);
    try {
      const result = await api<SetupPreview>("/api/project-setup/preflight", {
        method: "POST",
        body: JSON.stringify(payload()),
      });
      setPreview(result);
      setStep(3);
      setExistingResearchOpen(Boolean(result.existing_research));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  };

  const create = async (existingResearchAction?: ExistingResearchAction) => {
    if (!preview || busy) return;
    if (preview.existing_research) {
      if (!existingResearchAction || !preview.available_actions.includes(existingResearchAction)) {
        return;
      }
    } else if (!preview.can_create || !confirmed) {
      return;
    }
    setBusy("create");
    setError(null);
    try {
      const created = await api<ProjectCard>("/api/project-setup/create", {
        method: "POST",
        body: JSON.stringify({
          ...payload(),
          confirmed: true,
          ...setupExistingResearchSelection(preview, existingResearchAction),
        }),
      });
      onCreated(created.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setConfirmed(false);
      if (existingResearchAction === "archive_and_create") {
        setExistingResearchOpen(false);
        setPreview(null);
        setStep(2);
      }
    } finally {
      setBusy(null);
    }
  };

  const addRepository = () => {
    repositorySequence += 1;
    const alias = `repository-${repositorySequence}`;
    setRepositories((current) => [
      ...current,
      {
        id: repositorySequence,
        alias,
        location: "local",
        path: "",
        host: "",
        default_read: true,
      },
    ]);
  };

  const removeRepository = (id: number) => {
    setRepositories((current) => {
      const next = current.filter((repo) => repo.id !== id);
      setStateRepository((selected) => stateRepositoryAfterRemoval(current, id, selected));
      return next;
    });
  };

  const goBack = () => {
    if (step === 0) {
      onCancel();
      return;
    }
    setStep((current) => current - 1);
    setPreview(null);
    setExistingResearchOpen(false);
    setConfirmed(false);
    setError(null);
  };

  return (
    <>
      <main className="setup-layout">
        <nav className="setup-steps" aria-label="Project setup progress">
          <span className="eyebrow">Configuration route</span>
          {steps.map(([number, label], index) => (
            <button
              className={
                index === step
                  ? "setup-step active"
                  : index < step
                    ? "setup-step complete"
                    : "setup-step"
              }
              disabled={index > step}
              aria-current={index === step ? "step" : undefined}
              key={number}
              onClick={() => {
                if (index < step) {
                  setStep(index);
                  setPreview(null);
                  setConfirmed(false);
                  setError(null);
                }
              }}
            >
              <span>{index < step ? <Check size={13} /> : number}</span>
              <strong>{label}</strong>
            </button>
          ))}
        </nav>

        <section className="setup-sheet">
          {step === 0 && (
            <div className="setup-section">
              {intentChooser}
              <SectionHeading
                eyebrow="Project identity"
                title="Start with the paper and one repository."
              />
              <label className="setup-field">
                <span>Project name</span>
                <input
                  autoFocus
                  value={name}
                  onChange={(event) => {
                    setName(event.target.value);
                    setError(null);
                  }}
                  placeholder="Continual RL Plasticity"
                />
              </label>
              <div className="setup-rule" />
              <RepositoryEditor
                repository={repositories[0]}
                canonical={stateRepository === repositories[0].alias}
                only
                onCanonical={() => setStateRepository(repositories[0].alias)}
                onChange={(patch) => updateRepository(repositories[0].id, patch)}
              />
            </div>
          )}

          {step === 1 && (
            <div className="setup-section">
              <SectionHeading
                eyebrow="Guarded truth boundary"
                title="Which repositories belong to this project?"
              />
              <div className="repository-stack">
                {repositories.map((repository) => (
                  <RepositoryEditor
                    key={repository.id}
                    repository={repository}
                    canonical={stateRepository === repository.alias}
                    only={repositories.length === 1}
                    onCanonical={() => setStateRepository(repository.alias)}
                    onChange={(patch) => updateRepository(repository.id, patch)}
                    onRemove={() => removeRepository(repository.id)}
                  />
                ))}
              </div>
              <button className="add-repository" onClick={addRepository}>
                <Plus size={16} />
                <strong>Add another repository</strong>
              </button>
            </div>
          )}

          {step === 2 && (
            <div className="setup-section">
              <SectionHeading eyebrow="Agent roles" title="Choose the agent behind each surface." />
              <label className="agent-auto-research-default">
                <span>
                  Default auto-research ceiling
                  <small>Operational invocations per newly authorized episode</small>
                </span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  inputMode="numeric"
                  value={defaultAutoResearchInvocationCeiling}
                  onChange={(event) => {
                    setDefaultAutoResearchInvocationCeiling(Number(event.target.value));
                    setError(null);
                  }}
                />
              </label>
              <div className="agent-role-stack">
                {agentExecutionProfiles.map(({ id, label }) => {
                  const profile = agents[id];
                  const readiness = readinessFor(providers, profile.provider);
                  const models = readiness?.models ?? [];
                  const runtimes = runtimeOptions(readiness, profile.runtime);
                  const execution = id === "paper_coach" ? profile : canonicalExecution;
                  const machineValue =
                    execution.location === "local" ? "local" : `ssh:${execution.host}`;
                  return (
                    <article className="agent-role-card" key={id}>
                      <header>
                        <strong>{label}</strong>
                        <span className="role-permission">
                          {id === "paper_coach"
                            ? "read-only coach"
                            : id === "orchestrator"
                              ? "auto-research"
                              : "graph patch only"}
                        </span>
                      </header>
                      <div className="agent-role-fields">
                        <label>
                          Provider
                          <select
                            value={profile.provider}
                            onChange={(event) => {
                              const next = readinessFor(providers, event.target.value);
                              updateAgent(id, {
                                ...providerChange(
                                  next?.models ?? [],
                                  event.target.value,
                                  profile.reasoning,
                                ),
                                // The runtime belongs to the provider and moves
                                // with it, the same rule Project Settings applies.
                                ...(next ? { runtime: next.default_runtime } : {}),
                              });
                            }}
                          >
                            {providerOptions(providers, profile.provider).map((option) => (
                              <option key={option.id} value={option.id}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Runtime
                          <select
                            value={profile.runtime}
                            disabled={runtimes.length < 2}
                            onChange={(event) => updateAgent(id, { runtime: event.target.value })}
                          >
                            {runtimes.map((option) => (
                              <option key={option.id} value={option.id}>
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
                              updateAgent(
                                id,
                                modelChange(models, event.target.value, profile.reasoning),
                              )
                            }
                          >
                            {modelOptions(models, profile.model).map((option) => (
                              <option key={option.id} value={option.id}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Reasoning
                          <select
                            value={profile.reasoning}
                            onChange={(event) => updateAgent(id, { reasoning: event.target.value })}
                          >
                            {reasoningOptions(models, profile.model, profile.reasoning).map(
                              (option) => (
                                <option key={option.id} value={option.id}>
                                  {option.label}
                                </option>
                              ),
                            )}
                          </select>
                        </label>
                        <label className={id === "paper_coach" ? undefined : "agent-machine-fixed"}>
                          Run on{" "}
                          {id === "paper_coach" ? null : (
                            <LockKeyhole size={10} aria-hidden="true" />
                          )}
                          <select
                            value={machineValue}
                            disabled={id !== "paper_coach"}
                            onChange={(event) => {
                              const value = event.target.value;
                              updateAgent(
                                id,
                                value === "local"
                                  ? { location: "local", host: "" }
                                  : { location: "ssh", host: value.slice(4) },
                              );
                            }}
                          >
                            <option value="local">This machine</option>
                            {remoteHosts.map((host) => (
                              <option key={host} value={`ssh:${host}`}>
                                {host}
                              </option>
                            ))}
                          </select>
                        </label>
                      </div>
                      <div className="role-contract">
                        <LockKeyhole size={13} />{" "}
                        {id === "paper_coach"
                          ? "Introduction and project inputs are read-only · no writes"
                          : id === "orchestrator"
                            ? "Project-wide research · protected beliefs stay human-controlled"
                            : "Project and run-scope inputs are read-only · graph patch output only"}
                      </div>
                    </article>
                  );
                })}
              </div>
            </div>
          )}

          {step === 3 && preview && (
            <div className="setup-section review-section">
              <SectionHeading
                eyebrow="Read-only preflight complete"
                title={
                  preview.action === "connect"
                    ? "Connect the existing RCP project."
                    : "The project is ready to initialize."
                }
              />
              {preview.action === "connect" && (
                <div className="existing-manifest">
                  <FileCode2 size={19} />
                  <span>
                    Existing manifest: <strong>{preview.existing_project_name}</strong> · choose
                    whether to open it or archive it intact
                  </span>
                </div>
              )}
              <div className="preflight-checks">
                {preview.checks.map((check, index) => (
                  <div
                    className={`preflight-check ${check.status}`}
                    key={`${check.label}-${index}`}
                  >
                    {check.status === "pass" && <CheckCircle2 size={17} />}
                    {check.status === "warn" && <TriangleAlert size={17} />}
                    {check.status === "fail" && <XCircle size={17} />}
                    <strong>{check.label}</strong>
                    <span className="preflight-check-detail">{check.detail}</span>
                  </div>
                ))}
              </div>
              <details className="manifest-preview">
                <summary>
                  <FileCode2 size={15} /> Manifest{" "}
                  {preview.action === "connect" ? "to connect" : "preview"}
                </summary>
                <pre>{preview.manifest_preview}</pre>
              </details>
              {preview.existing_research ? (
                <button
                  className="existing-research-review"
                  type="button"
                  onClick={() => setExistingResearchOpen(true)}
                >
                  <ShieldCheck size={16} /> Review existing research choices
                </button>
              ) : (
                <label
                  className={
                    preview.can_create ? "final-confirmation" : "final-confirmation disabled"
                  }
                >
                  <input
                    type="checkbox"
                    checked={confirmed}
                    disabled={!preview.can_create}
                    onChange={(event) => setConfirmed(event.target.checked)}
                  />
                  <span>{setupFinalConfirmation(preview)}</span>
                </label>
              )}
            </div>
          )}

          {error && (
            <div className="setup-error" role="alert">
              <TriangleAlert size={16} />
              <span>{error}</span>
            </div>
          )}
          <footer className="setup-actions">
            <button className="button secondary" onClick={goBack}>
              <ArrowLeft size={15} /> Back
            </button>
            {step < 3 && (
              <button
                className="button primary"
                disabled={busy !== null}
                onClick={() => void advance()}
              >
                {busy === "preflight" ? (
                  <LoaderCircle className="spin" size={15} />
                ) : step === 2 ? (
                  <ShieldCheck size={15} />
                ) : null}
                {busy === "preflight"
                  ? "Checking"
                  : step === 2
                    ? "Run read-only preflight"
                    : "Continue"}
                {!busy && step < 2 && <ArrowRight size={15} />}
              </button>
            )}
            {step === 3 &&
              preview &&
              (preview.existing_research ? (
                <button
                  className="button primary"
                  disabled={busy !== null}
                  onClick={() => setExistingResearchOpen(true)}
                >
                  <ShieldCheck size={15} /> Choose how to continue
                </button>
              ) : (
                <button
                  className="button primary"
                  disabled={!preview.can_create || !confirmed || busy !== null}
                  onClick={() => void create()}
                >
                  {busy === "create" ? (
                    <LoaderCircle className="spin" size={15} />
                  ) : (
                    <Check size={15} />
                  )}
                  {busy === "create" ? "Opening project" : "Create and open"}
                </button>
              ))}
          </footer>
        </section>

        <aside className="boundary-ledger">
          <span className="eyebrow">Live boundary ledger</span>
          <h2>What this setup means</h2>
          <LedgerItem
            number="A"
            label="Canonical state"
            value={stateRepo ? repositoryLocation(stateRepo) : "Not selected"}
          />
          <LedgerItem
            number="B"
            label="Project graph"
            value={`${repositories.length} truth repositor${repositories.length === 1 ? "y" : "ies"}`}
          />
          <LedgerItem
            number="C"
            label="Raw prompt inputs"
            value={
              defaultRead.length
                ? defaultRead.map((repo) => repo.alias || "unnamed").join(", ")
                : "None selected"
            }
          />
          <LedgerItem
            number="D"
            label="Agent roles"
            value={agentExecutionProfiles
              .map(({ id, label }) => {
                const execution = id === "paper_coach" ? agents[id] : canonicalExecution;
                return `${label}: ${agents[id].provider} @ ${execution.location === "ssh" ? execution.host || "remote" : "local"}`;
              })
              .join(" · ")}
          />
        </aside>
      </main>

      {existingResearchOpen && preview?.existing_research && (
        <ExistingResearchDialog
          preview={preview}
          busy={busy === "create"}
          error={error}
          onCancel={() => {
            if (!busy) setExistingResearchOpen(false);
          }}
          onChoose={(action) => void create(action)}
        />
      )}
    </>
  );
}

function SetupRouteFailure({ message, onCancel }: { message: string; onCancel: () => void }) {
  return (
    <main className="setup-layout setup-route-failure-layout">
      <section className="setup-sheet">
        <div className="setup-section setup-route-failure" role="alert">
          <TriangleAlert size={21} aria-hidden="true" />
          <SectionHeading eyebrow="Project setup" title="This setup route cannot continue." />
          <p>{message}</p>
          <button className="button secondary" type="button" onClick={onCancel}>
            <ArrowLeft size={15} /> Return to projects
          </button>
        </div>
      </section>
    </main>
  );
}

function ProjectIntentChooser({
  control,
  selected,
  locked = false,
  onSelect,
}: {
  control: ProjectCreationControl;
  selected: ProjectCreationIntent;
  locked?: boolean;
  onSelect: (intent: ProjectCreationIntent) => void;
}) {
  const labels: Record<ProjectCreationIntent, string> = {
    use_existing_checkout_personally: "Use an existing checkout personally",
    create_shared_team_project: "Create a shared team project",
    move_personal_project_to_team: "Move an existing personal project to a team",
  };
  return (
    <div className="project-intent-choices" role="group" aria-label="Project setup kind">
      {control.intents.map((intent) => (
        <button
          className={intent.intent === selected ? "active" : ""}
          type="button"
          key={intent.intent}
          aria-pressed={intent.intent === selected}
          disabled={!intent.eligible || locked}
          title={intent.unavailable_reason ?? undefined}
          onClick={() => onSelect(intent.intent)}
        >
          <strong>{labels[intent.intent]}</strong>
          {!intent.eligible && intent.unavailable_reason && (
            <span>{intent.unavailable_reason}</span>
          )}
        </button>
      ))}
    </div>
  );
}

export function setupExistingResearchSelection(
  preview: SetupPreview,
  action?: ExistingResearchAction,
): Pick<ProjectSetupRequest, "existing_research_action" | "existing_research_token"> {
  return {
    existing_research_action: action ?? null,
    existing_research_token:
      action === "archive_and_create" ? (preview.existing_research?.archive_token ?? null) : null,
  };
}

function ExistingResearchDialog({
  preview,
  busy,
  error,
  onCancel,
  onChoose,
}: {
  preview: SetupPreview;
  busy: boolean;
  error: string | null;
  onCancel: () => void;
  onChoose: (action: ExistingResearchAction) => void;
}) {
  const existing = preview.existing_research;
  if (!existing) return null;
  const degraded = existing.replay_status === "degraded";
  const failure = existing.replay_failure;
  const openAction: ExistingResearchAction = degraded ? "open_degraded_read_only" : "open_existing";
  const canOpen = preview.available_actions.includes(openAction);
  const canArchive = preview.available_actions.includes("archive_and_create");

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onCancel();
      }}
    >
      <section
        className={`existing-research-dialog ${degraded ? "degraded" : "compatible"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="existing-research-title"
        aria-describedby="existing-research-warning"
      >
        <header>
          {degraded ? (
            <TriangleAlert size={20} aria-hidden="true" />
          ) : (
            <ShieldCheck size={20} aria-hidden="true" />
          )}
          <div>
            <span className="eyebrow">Canonical state detected</span>
            <h2 id="existing-research-title">Existing RCP research found</h2>
          </div>
        </header>

        <div className="existing-research-body">
          <div className="existing-research-status">
            <strong>{degraded ? "Replay stopped" : "Replay compatible"}</strong>
            <span>
              {degraded
                ? `Revision ${failure?.revision ?? "unknown"} cannot be replayed. RCP can safely show revision ${existing.coherent_revision} read-only.`
                : `All ${existing.retained_revision_count} retained revisions replay successfully.`}
            </span>
          </div>
          <dl className="existing-research-ledger">
            <div>
              <dt>Research</dt>
              <dd>{existing.project_name}</dd>
            </div>
            <div>
              <dt>Canonical location</dt>
              <dd>{existing.canonical_location}</dd>
            </div>
            <div>
              <dt>Retained revisions</dt>
              <dd>{existing.retained_revision_count}</dd>
            </div>
            <div>
              <dt>Last coherent revision</dt>
              <dd>{existing.coherent_revision}</dd>
            </div>
          </dl>
          {degraded && failure && (
            <div className="existing-research-failure" role="status">
              <code>{failure.code}</code>
              <span>{failure.message}</span>
            </div>
          )}
          <p id="existing-research-warning">
            Starting fresh will move the complete <code>.research</code> directory to a timestamped
            archive beside the repository, then create new canonical state from this setup. Nothing
            in the retained patch history will be edited or overwritten.
          </p>
        </div>

        {error && (
          <div className="project-delete-error" role="alert">
            {error}
          </div>
        )}
        <footer>
          <button
            className="button secondary"
            type="button"
            autoFocus
            disabled={busy}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            className="button secondary"
            type="button"
            disabled={busy || !canOpen}
            onClick={() => onChoose(openAction)}
          >
            {busy
              ? "Opening…"
              : degraded
                ? "Open last coherent state (read-only)"
                : "Open existing research"}
          </button>
          <button
            className="button danger"
            type="button"
            disabled={busy || !canArchive}
            onClick={() => onChoose("archive_and_create")}
          >
            {busy ? "Working…" : "Archive existing research and start fresh"}
          </button>
        </footer>
      </section>
    </div>
  );
}

export function setupFinalConfirmation(preview: SetupPreview): string {
  const action =
    preview.action === "connect"
      ? "Open this project's retained canonical state without replacing its manifest"
      : "Create the project manifest";
  const location = preview.remote_write
    ? `RCP may write canonical project state over SSH at ${preview.canonical_location}`
    : `RCP may initialize canonical project state at ${preview.canonical_location}`;
  return `${action} · ${location}`;
}

function SectionHeading({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <header className="setup-section-heading">
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
    </header>
  );
}

export function RepositoryEditor({
  repository,
  canonical,
  only,
  onCanonical,
  onChange,
  onRemove,
}: {
  repository: DraftRepository;
  canonical: boolean;
  only: boolean;
  onCanonical: () => void;
  onChange: (patch: Partial<SetupRepository>) => void;
  onRemove?: () => void;
}) {
  const [pickerBusy, setPickerBusy] = useState(false);
  const [pickerError, setPickerError] = useState<string | null>(null);
  const picker = repositoryPickerPresentation(repository.location, isDesktopRuntime());
  const pathInputId = `repository-path-${repository.id}`;

  const changeRepository = (patch: Partial<SetupRepository>) => {
    if (patch.location !== undefined || patch.path !== undefined) setPickerError(null);
    onChange(patch);
  };

  const chooseFolder = async () => {
    setPickerBusy(true);
    setPickerError(null);
    try {
      const path = await chooseDesktopRepositoryFolder();
      if (path !== null) changeRepository({ path });
    } catch (error) {
      setPickerError(error instanceof Error ? error.message : String(error));
    } finally {
      setPickerBusy(false);
    }
  };

  return (
    <article className={canonical ? "repository-editor canonical" : "repository-editor"}>
      <header>
        <span className="repository-number">
          <FolderGit2 size={16} />
        </span>
        <strong>{repository.alias || "Unnamed repository"}</strong>
        {canonical && <span className="repository-state">Canonical state</span>}
        {!only && onRemove && (
          <button
            className="icon-button compact"
            aria-label={`Remove ${repository.alias}`}
            onClick={onRemove}
          >
            <Trash2 size={14} />
          </button>
        )}
      </header>
      <div className="repository-fields">
        <label>
          <span>Alias</span>
          <input
            value={repository.alias}
            onChange={(event) => onChange({ alias: event.target.value })}
            placeholder="research-code"
          />
        </label>
        <div className="location-toggle" aria-label="Repository location">
          <button
            className={repository.location === "local" ? "active" : ""}
            onClick={() => changeRepository({ location: "local" })}
          >
            Local
          </button>
          <button
            className={repository.location === "ssh" ? "active" : ""}
            onClick={() => changeRepository({ location: "ssh" })}
          >
            SSH
          </button>
        </div>
        {repository.location === "ssh" && (
          <label>
            <span>SSH host</span>
            <input
              value={repository.host}
              onChange={(event) => onChange({ host: event.target.value })}
              placeholder="gpu.example.edu"
            />
          </label>
        )}
        <div className={`repository-path-field ${repository.location === "ssh" ? "" : "wide"}`}>
          <label htmlFor={pathInputId}>
            <span>Absolute repository path</span>
          </label>
          <div className="repository-path-input">
            <input
              id={pathInputId}
              value={repository.path}
              onChange={(event) => changeRepository({ path: event.target.value })}
              aria-describedby={
                pickerError
                  ? `${pathInputId}-error`
                  : picker.hint
                    ? `${pathInputId}-hint`
                    : undefined
              }
              placeholder={
                repository.location === "ssh"
                  ? "/home/user/research/project"
                  : "/Users/you/research/project"
              }
            />
            {picker.showPicker && (
              <button type="button" onClick={() => void chooseFolder()} disabled={pickerBusy}>
                {pickerBusy ? (
                  <LoaderCircle className="spin" size={14} />
                ) : (
                  <FolderOpen size={14} />
                )}
                Choose folder…
              </button>
            )}
          </div>
          {picker.hint && (
            <small id={`${pathInputId}-hint`} className="repository-path-hint">
              {picker.hint}
            </small>
          )}
          {pickerError && (
            <small id={`${pathInputId}-error`} className="repository-path-error" role="alert">
              {pickerError}
            </small>
          )}
        </div>
      </div>
      <footer>
        <label className="check-control">
          <input
            type="checkbox"
            checked={repository.default_read}
            onChange={(event) => onChange({ default_read: event.target.checked })}
          />
          <strong>Default raw input</strong>
        </label>
        <label className="radio-control">
          <input
            type="radio"
            name="canonical-repository"
            checked={canonical}
            onChange={onCanonical}
          />
          <span>Canonical state</span>
        </label>
      </footer>
    </article>
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

function repositoryLocation(repository: DraftRepository): string {
  if (!repository.path) return "Repository path not entered";
  return repository.location === "ssh"
    ? `${repository.host || "host"}:${repository.path}`
    : repository.path;
}
