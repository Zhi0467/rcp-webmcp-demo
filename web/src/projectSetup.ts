import type {
  ProjectCreationControl,
  ProjectCreationIntent,
  ProjectProvisioningCreateRequest,
} from "./types";

const UUID4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

export interface ProjectMoveSetupRouteInput {
  sourceProjectId: string;
  sourceRequestId?: string | null;
  targetRequestId?: string | null;
}

export type ProjectSetupRoute =
  | { kind: "none" }
  | { kind: "create"; requestId: string | null }
  | ({ kind: "move"; intent: "move_personal_project_to_team" } & ProjectMoveSetupRouteInput & {
        sourceRequestId: string | null;
        targetRequestId: string | null;
      })
  | { kind: "invalid"; reason: "invalid_provisioning_route" | "invalid_move_route" };

function isCanonicalUuid4(value: string): boolean {
  return UUID4_PATTERN.test(value);
}

function requireCanonicalUuid4(value: string, label: string): string {
  if (!isCanonicalUuid4(value)) {
    throw new Error(`${label} must be a canonical UUID4.`);
  }
  return value;
}

function optionalCanonicalUuid4(value: string | null | undefined, label: string): string | null {
  if (value === null || value === undefined) return null;
  return requireCanonicalUuid4(value, label);
}

function oneQueryValue(params: URLSearchParams, name: string): string | null | undefined {
  const values = params.getAll(name);
  return values.length > 1 ? undefined : (values[0] ?? null);
}

/**
 * Parse the one visible project-setup route. The move variant is deliberately
 * separate from ordinary provisioning so a source project can never be
 * inferred from a generic `request` link.
 */
export function parseProjectSetupRoute(hash: string): ProjectSetupRoute {
  if (hash === "#/projects/new") return { kind: "create", requestId: null };
  if (!hash.startsWith("#/projects/new?")) return { kind: "none" };

  const params = new URLSearchParams(hash.slice(hash.indexOf("?") + 1));
  const intent = oneQueryValue(params, "intent");
  if (intent === "move_personal_project_to_team") {
    const sourceProjectId = oneQueryValue(params, "source_project_id");
    const sourceRequestId = oneQueryValue(params, "source_request_id");
    const targetRequestId = oneQueryValue(params, "target_request_id");
    const allowed = new Set([
      "intent",
      "source_project_id",
      "source_request_id",
      "target_request_id",
    ]);
    const hasUnknown = [...params.keys()].some((name) => !allowed.has(name));
    if (
      hasUnknown ||
      sourceProjectId === null ||
      sourceProjectId === undefined ||
      !isCanonicalUuid4(sourceProjectId) ||
      sourceRequestId === undefined ||
      targetRequestId === undefined
    ) {
      return { kind: "invalid", reason: "invalid_move_route" };
    }
    if ((sourceRequestId === null) !== (targetRequestId === null)) {
      return { kind: "invalid", reason: "invalid_move_route" };
    }
    if (
      (sourceRequestId !== null && !isCanonicalUuid4(sourceRequestId)) ||
      (targetRequestId !== null && !isCanonicalUuid4(targetRequestId))
    ) {
      return { kind: "invalid", reason: "invalid_move_route" };
    }
    return {
      kind: "move",
      intent,
      sourceProjectId,
      sourceRequestId,
      targetRequestId,
    };
  }

  // Preserve the existing create/resume behavior. Its old parser ignored
  // unrelated query keys, so keep accepting them for compatibility.
  const requestId = oneQueryValue(params, "request");
  return requestId !== null && requestId !== undefined && isCanonicalUuid4(requestId)
    ? { kind: "create", requestId }
    : { kind: "invalid", reason: "invalid_provisioning_route" };
}

export function projectMoveSetupHash(input: ProjectMoveSetupRouteInput): string {
  const sourceProjectId = requireCanonicalUuid4(input.sourceProjectId, "Source project identity");
  const sourceRequestId = optionalCanonicalUuid4(input.sourceRequestId, "Source request identity");
  const targetRequestId = optionalCanonicalUuid4(input.targetRequestId, "Target request identity");
  if ((sourceRequestId === null) !== (targetRequestId === null)) {
    throw new Error("Source and target request identities must be created as one pair.");
  }
  const params = new URLSearchParams({
    intent: "move_personal_project_to_team",
    source_project_id: sourceProjectId,
  });
  if (sourceRequestId) params.set("source_request_id", sourceRequestId);
  if (targetRequestId) params.set("target_request_id", targetRequestId);
  return `#/projects/new?${params.toString()}`;
}

export function stateRepositoryAfterRemoval(
  repositories: Array<{ id: number; alias: string }>,
  removedId: number,
  stateRepository: string,
): string {
  const removed = repositories.find((repository) => repository.id === removedId);
  if (removed?.alias !== stateRepository) return stateRepository;
  return repositories.find((repository) => repository.id !== removedId)?.alias ?? "";
}

export function repositoryPickerPresentation(location: "local" | "ssh", desktop: boolean) {
  return {
    showPicker: location === "local" && desktop,
    hint:
      location === "local" && !desktop
        ? "Paste an absolute path. Finder selection is available in the desktop app."
        : null,
  };
}

export function selectedProjectCreationIntent(
  control: ProjectCreationControl,
): ProjectCreationIntent {
  const preselected = control.intents.filter((intent) => intent.eligible && intent.preselected);
  if (preselected.length === 1) return preselected[0].intent;
  const eligible = control.intents.filter((intent) => intent.eligible);
  if (eligible.length === 1) return eligible[0].intent;
  throw new Error("The backend did not select one available project setup intent.");
}

export function assertSupportedProjectCreationIntent(
  control: ProjectCreationControl,
  intent: ProjectCreationIntent,
): void {
  const selected = control.intents.find((item) => item.intent === intent);
  if (!selected?.eligible) {
    throw new Error("The selected project setup intent is not available from this backend.");
  }
  const expectedFields: Record<ProjectCreationIntent, string[]> = {
    use_existing_checkout_personally: [
      "name",
      "repositories",
      "state_repository",
      "execution",
      "confirmed",
    ],
    create_shared_team_project: ["machines", "repositories", "provider_checks"],
    move_personal_project_to_team: ["source_project", "team_connection"],
  };
  const actualFields = new Set(selected.required_fields);
  if (
    actualFields.size !== selected.required_fields.length ||
    actualFields.size !== expectedFields[intent].length ||
    expectedFields[intent].some((field) => !actualFields.has(field))
  ) {
    throw new Error("This build does not support the backend's project setup field contract.");
  }
  if (intent !== "move_personal_project_to_team" && selected.pinned_source_project_id !== null) {
    throw new Error("Only a personal-to-team move may pin an existing source project.");
  }
}

export function projectCreationPrimaryLabel(control: ProjectCreationControl): string {
  const selected = selectedProjectCreationIntent(control);
  return control.intents.find((intent) => intent.intent === selected)?.primary_action_label ?? "";
}

export function projectProvisioningRequestId(hash: string): string | null {
  const route = parseProjectSetupRoute(hash);
  return route.kind === "create" ? route.requestId : null;
}

export function invalidProjectProvisioningHash(hash: string): boolean {
  return hash.startsWith("#/projects/new?") && parseProjectSetupRoute(hash).kind === "invalid";
}

export function projectProvisioningHash(requestId: string): string {
  if (!isCanonicalUuid4(requestId)) {
    throw new Error("Project provisioning request identity must be a canonical UUID4.");
  }
  return `#/projects/new?request=${encodeURIComponent(requestId)}`;
}

export function formatCommandArgv(argv: string[]): string {
  return argv
    .map((value) =>
      value && /^[A-Za-z0-9_@%+=:,./-]+$/.test(value)
        ? value
        : `'${value.replaceAll("'", "'\\''")}'`,
    )
    .join(" ");
}

export function buildTeamProvisioningRequest({
  name,
  stateRepository,
  defaultAutoResearchInvocationCeiling,
  machines,
  repositories,
  providerChecks,
}: {
  name: string;
  stateRepository: string;
  defaultAutoResearchInvocationCeiling: number;
  machines: ProjectProvisioningCreateRequest["machines"];
  repositories: Array<
    ProjectProvisioningCreateRequest["repositories"][number] & { default_read: boolean }
  >;
  providerChecks: ProjectProvisioningCreateRequest["provider_checks"];
}): ProjectProvisioningCreateRequest {
  const aliases = repositories.map((repository) => repository.alias);
  return {
    name: name.trim(),
    state_repository: stateRepository,
    project_truth_scope: aliases,
    default_run_truth_scope: repositories
      .filter((repository) => repository.default_read)
      .map((repository) => repository.alias),
    default_auto_research_invocation_ceiling: defaultAutoResearchInvocationCeiling,
    machines,
    repositories: repositories.map(({ default_read: _defaultRead, ...repository }) => repository),
    provider_checks: providerChecks,
  };
}
