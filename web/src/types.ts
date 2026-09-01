export type Standing = "asserted" | "accepted" | "contested";
export type TrustView = "working" | "accepted" | "review";
export type AppView =
  "overview" | "attention" | "scientific" | "dag" | "execution" | "paper" | "settings" | "chats";
export type AgentSurface = "seed" | "refresh" | "node_chat" | "project_chat" | "paper_coach";
export type AgentExecutionProfile = AgentSurface | "orchestrator";
export type AgentTaskKind = AgentSurface | "auto_research" | "branch_merge";
/**
 * Opaque on purpose, like `EpisodeStatus`. A task's lifecycle reaches the client
 * already answered, as `active`, `awaiting_human`, `settled`, `status_label`, and
 * the `can_*` controls.
 */
declare const OPAQUE_TASK_STATUS: unique symbol;
export type AgentTaskStatus = { readonly [OPAQUE_TASK_STATUS]: "AgentTaskStatus" };
export type ConversationMode = "discuss" | "work";
export type TaskTrigger = "human" | "orchestrator" | "experiment_run" | "watcher";
export type GraphPatchKind = "work" | "experiment_loop";
export type AgentCapability =
  "discuss" | "work_auto" | "orchestrate" | "scratch_patch" | "paper_readonly";

export const DISPLAY_NAME_MAX_LENGTH = 120;
export const SPACE_NAME_MAX_LENGTH = 120;

export interface Health {
  status: string;
  agent_mode: "provider" | "acceptance";
  version: string;
  space_id: string;
  space_kind: "personal" | "team";
  space_name: string | null;
  instance_id: string;
  data_dir_id: string;
  owner_kind: string;
  running_commit: string | null;
  web_build_id: string | null;
  active_agent_tasks: number;
  pid: number;
  projects?: number;
  project?: string | null;
  project_creation: ProjectCreationControl;
}

export type ServerStatusTone = "good" | "attention" | "bad" | "neutral";

export interface ServerStatusSummary {
  label: string;
  tone: ServerStatusTone;
}

export interface ServerReleaseStatus {
  status: ServerStatusSummary;
  managed_source_commit: string | null;
  current_release_commit: string | null;
  running_commit: string | null;
  upstream_commit: string | null;
  candidate_commit: string | null;
  update_available: boolean;
  last_update_failure: string | null;
  command: "rcp server update";
}

export interface ServerBackupStatus {
  status: ServerStatusSummary;
  configured: boolean;
  destination: string | null;
  schedule: string | null;
  retention: number | null;
  last_attempt_at: string | null;
  last_protected_at: string | null;
  captured_bytes: number | null;
  protected_projects: number | null;
  uncaptured_projects: number | null;
  last_failure: string | null;
  configure_command: "rcp server backup configure";
  run_command: "rcp server backup run";
}

export interface ServerRestoreStatus {
  status: ServerStatusSummary;
  last_completed_at: string | null;
  drill_age_days: number | null;
  command: "rcp server restore";
}

export interface ServerExecutionReadiness {
  machine: ServerStatusSummary;
  provider_checks: ServerStatusSummary;
  dependency_versions: string;
  provider_command: "rcp server provider check";
}

export interface ServerOperatorCommand {
  name: string;
  command: string;
  purpose: string;
}

export interface ServerStatus {
  overall: ServerStatusSummary;
  releases: ServerReleaseStatus;
  backup: ServerBackupStatus;
  restore: ServerRestoreStatus;
  execution: ServerExecutionReadiness;
  operator_commands: ServerOperatorCommand[];
  problems: string[];
}

export interface AuthorizedHuman {
  space_id: string;
  user_id: string;
  display_name: string;
}

export type ProjectCreationIntent =
  | "use_existing_checkout_personally"
  | "create_shared_team_project"
  | "move_personal_project_to_team";

export interface ProjectCreationIntentControl {
  intent: ProjectCreationIntent;
  eligible: boolean;
  preselected: boolean;
  primary_action_label: string;
  required_fields: string[];
  pinned_source_project_id: string | null;
  unavailable_reason: string | null;
}

export interface ProjectCreationControl {
  intents: ProjectCreationIntentControl[];
  requires_authenticated_member: boolean;
}

/**
 * Opaque because the backend publishes every display and action decision for
 * this lifecycle. The browser renders those answers instead of interpreting a
 * persisted status value.
 */
declare const OPAQUE_PROJECT_PROVISIONING_STATUS: unique symbol;
export type ProjectProvisioningStatus = {
  readonly [OPAQUE_PROJECT_PROVISIONING_STATUS]: "ProjectProvisioningStatus";
};

declare const OPAQUE_PROJECT_PROVISIONING_CHECK_STATUS: unique symbol;
export type ProjectProvisioningCheckStatus = {
  readonly [OPAQUE_PROJECT_PROVISIONING_CHECK_STATUS]: "ProjectProvisioningCheckStatus";
};

declare const OPAQUE_SERVER_STEP_STATE: unique symbol;
export type ServerStepState = { readonly [OPAQUE_SERVER_STEP_STATE]: "ServerStepState" };

export interface ProjectProvisioningMachineRequest {
  alias: string;
  location: "local" | "ssh";
  host?: string;
  os_account: string;
  central_root?: string | null;
}

export interface ProjectProvisioningRepositoryRequest {
  alias: string;
  source: string;
  machine_alias: string;
}

export interface ProjectProvisioningProviderRequest {
  profile: AgentExecutionProfile;
  provider: ProviderId;
  runtime_id: string;
  model: string;
  reasoning: string;
  machine_alias: string;
}

export interface ProjectProvisioningCreateRequest {
  name: string;
  state_repository: string;
  project_truth_scope: string[];
  default_run_truth_scope: string[];
  default_auto_research_invocation_ceiling: number;
  machines: ProjectProvisioningMachineRequest[];
  repositories: ProjectProvisioningRepositoryRequest[];
  provider_checks: ProjectProvisioningProviderRequest[];
}

/** Native transfer setup uses the same public provisioning intent shapes. */
export type ProjectTransferMachineIntent = ProjectProvisioningMachineRequest;
export type ProjectTransferProviderIntent = ProjectProvisioningProviderRequest;

export interface ProjectTransferTargetProvisioningIntent {
  name: string;
  default_auto_research_invocation_ceiling: number;
  machines: ProjectTransferMachineIntent[];
  provider_checks: ProjectTransferProviderIntent[];
}

export interface ProjectTransferPrepareRequest {
  sourceRequestId: string;
  targetRequestId: string;
  connectionId: string;
  sourceProjectId: string;
  targetProvisioning: ProjectTransferTargetProvisioningIntent;
}

export interface ServerMachineTarget {
  kind: "machine";
  host: string;
  os_account: string;
}

export interface ServerExternalServiceTarget {
  kind: "external_service";
  service: string;
  resource: string;
  destination_url: string;
  required_authority_role: string;
}

export type ServerStepTarget = ServerMachineTarget | ServerExternalServiceTarget;

export interface ServerCommandAction {
  kind: "command";
  argv: string[];
}

export interface ServerExternalAction {
  kind: "external";
  instruction: string;
}

export type ServerOperatorAction = ServerCommandAction | ServerExternalAction;

export interface ServerNonsecretField {
  name: string;
  value: string | number | boolean;
}

export interface ServerStep {
  number: number;
  title: string;
  purpose: string;
  performed_by: "system" | "human";
  target: ServerStepTarget;
  phase: string;
  state: ServerStepState;
  expected_success: string;
  message: string;
  actions: ServerOperatorAction[];
  fields: ServerNonsecretField[];
  resume_argv: string[];
}

export interface ProjectProvisioningMachineProjection {
  alias: string;
  location: "local" | "ssh";
  host: string;
  os_account: string;
  intended_central_root: string | null;
  resolved_central_root: string | null;
  ready: boolean;
  status_label: string;
}

export interface GitHubRepositoryRef {
  identity: string;
}

export interface ProjectProvisioningRepositoryProjection {
  alias: string;
  repository: GitHubRepositoryRef;
  https_clone_url: string;
  ssh_clone_url: string;
  settings_url: string;
  machine_alias: string;
  intended_path: string | null;
  resolved_path: string | null;
  checkout_disposition: "request_created" | "reused_existing" | null;
  status: ProjectProvisioningCheckStatus;
  status_label: string;
  ready: boolean;
  commit: string | null;
  write_verified: boolean;
  deploy_key_label: string | null;
  public_key_fingerprint: string | null;
  checked_at: string | null;
  diagnostic: string | null;
}

export interface ProjectProvisioningProviderProjection {
  profile: AgentExecutionProfile;
  provider: ProviderId;
  runtime_id: string;
  model: string;
  reasoning: string;
  machine_alias: string;
  status: ProjectProvisioningCheckStatus;
  status_label: string;
  ready: boolean;
  binary_path: string | null;
  version: string | null;
  resolved_runtime_id: string | null;
  execution_account: string | null;
  checked_at: string | null;
  diagnostic: string | null;
}

export interface ProjectProvisioningReadinessProjection {
  machines_ready: number;
  machines_total: number;
  repositories_ready: number;
  repositories_total: number;
  providers_ready: number;
  providers_total: number;
  all_ready: boolean;
}

export interface ProjectProvisioningFinalReview {
  digest: string;
  proposed_project_id: string;
  authorized_by: AuthorizedHuman;
  ready_at: string;
}

export type ProjectProvisioningCancellationDisposition =
  | "nothing_to_remove"
  | "request_owned_state_removed"
  | "prepared_state_preserved"
  | "operator_cleanup_confirmed";

export interface ProjectProvisioningResponse {
  request_id: string;
  kind: "create_team_project" | "incoming_transfer";
  status: ProjectProvisioningStatus;
  status_label: string;
  next_action: string | null;
  can_run_setup: boolean;
  can_review: boolean;
  can_cancel: boolean;
  target_space_id: string;
  proposed_project_id: string;
  name: string | null;
  state_repository: string | null;
  project_truth_scope: string[];
  default_run_truth_scope: string[];
  default_auto_research_invocation_ceiling: number;
  authorized_by: AuthorizedHuman;
  machines: ProjectProvisioningMachineProjection[];
  repositories: ProjectProvisioningRepositoryProjection[];
  provider_checks: ProjectProvisioningProviderProjection[];
  readiness: ProjectProvisioningReadinessProjection;
  diagnostic: string | null;
  operator_action: ServerStep | null;
  operator_argv: string[];
  final_review: ProjectProvisioningFinalReview | null;
  cancellation_disposition: ProjectProvisioningCancellationDisposition | null;
  revision: number;
  created_at: string;
  updated_at: string;
  setup_started_at: string | null;
  completed_at: string | null;
  cancelled_at: string | null;
}

/**
 * Opaque because the backend publishes transfer labels, next actions, and every
 * `can_*` decision. Web code must not reconstruct the cross-space state machine.
 */
declare const OPAQUE_PROJECT_TRANSFER_PHASE: unique symbol;
export type ProjectTransferPhase = {
  readonly [OPAQUE_PROJECT_TRANSFER_PHASE]: "ProjectTransferPhase";
};

declare const OPAQUE_PROJECT_TRANSFER_PROOF_STATE: unique symbol;
export type ProjectTransferProofState = {
  readonly [OPAQUE_PROJECT_TRANSFER_PROOF_STATE]: "ProjectTransferProofState";
};

export interface ProjectTransferRepositorySource {
  alias: string;
  repository: GitHubRepositoryRef;
  machine_alias: string;
}

export interface ProjectTransferSourceConfiguration {
  source_rcp_version: string;
  source_schema_generation: number;
  supported_archive_codecs: string[];
  machine_aliases: string[];
  repositories: ProjectTransferRepositorySource[];
  state_repository: string;
  project_truth_scope: string[];
  default_run_truth_scope: string[];
  source_manifest_sha256: string;
}

export interface ProjectTransferRepositoryBinding {
  alias: string;
  repository: GitHubRepositoryRef;
}

export interface ProjectTransferResolvedPath {
  repository_alias: string;
  machine_alias: string;
  path: string;
}

export interface ProjectTransferLinkReceipt {
  source_request_id: string;
  target_request_id: string;
  project_id: string;
  source_space_id: string;
  target_space_id: string;
  source_configuration_sha256: string;
  target_repositories: ProjectTransferRepositoryBinding[];
  accepted_schema_generation: number;
  accepted_archive_codec: string;
  source_release_proof_sha256: string;
  target_activation_proof_sha256: string;
  created_at: string;
}

export interface ProjectTransferTargetAdmissionReceipt {
  source_request_id: string;
  target_request_id: string;
  project_id: string;
  source_space_id: string;
  target_space_id: string;
  admitted_by: AuthorizedHuman;
  source_configuration_sha256: string;
  target_preparation_revision: number;
  target_preparation_sha256: string;
  resolved_paths: ProjectTransferResolvedPath[];
  accepted_schema_generation: number;
  accepted_archive_codec: string;
  source_release_proof_sha256: string;
  target_activation_proof_sha256: string;
  created_at: string;
}

export interface ProjectTransferSourceReleaseReceipt {
  source_request_id: string;
  target_request_id: string;
  project_id: string;
  source_space_id: string;
  target_space_id: string;
  released_by: AuthorizedHuman;
  source_configuration_sha256: string;
  target_admission_sha256: string;
  target_preparation_revision: number;
  target_preparation_sha256: string;
  source_head: GraphHeadRef;
  accepted_schema_generation: number;
  accepted_archive_codec: string;
  source_release_proof_sha256: string;
  target_activation_proof_sha256: string;
  created_at: string;
}

export interface ProjectTransferResponse {
  request_id: string;
  side: "source" | "target";
  phase: ProjectTransferPhase;
  linked_request_id: string | null;
  project_id: string;
  source_space_id: string;
  target_space_id: string;
  initiated_by: AuthorizedHuman;
  source_configuration: ProjectTransferSourceConfiguration;
  source_configuration_sha256: string;
  accepted_schema_generation: number | null;
  accepted_archive_codec: string | null;
  source_release_proof_sha256: string;
  target_activation_proof_sha256: string | null;
  link_receipt: ProjectTransferLinkReceipt | null;
  proof_state: ProjectTransferProofState;
  proof_acknowledgement_sha256: string | null;
  target_admission_receipt: ProjectTransferTargetAdmissionReceipt | null;
  source_release_receipt: ProjectTransferSourceReleaseReceipt | null;
  source_fence_head: GraphHeadRef | null;
  archive_sha256: string | null;
  archive_size_bytes: number | null;
  restore_resume_phase: ProjectTransferPhase | null;
  restore_diagnostic: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
  phase_label: string;
  next_action: string | null;
  can_link: boolean;
  can_run_setup: boolean;
  can_review: boolean;
  can_admit: boolean;
  can_accept_admission: boolean;
  can_release: boolean;
  can_accept_release: boolean;
  can_relay: boolean;
  can_restore_reentry: boolean;
  can_complete: boolean;
  finished: boolean;
}

export interface ProjectTransferSourceBoundaryResponse {
  source_configuration: ProjectTransferSourceConfiguration;
  source_configuration_sha256: string;
  source_head: GraphHeadRef;
}

export interface ProjectTransferProjection {
  request_id: string;
  side: "source" | "target";
  phase: ProjectTransferPhase;
  phase_label: string | null;
  next_action: string | null;
  linked_request_id: string | null;
  project_id: string;
  source_space_id: string;
  target_space_id: string;
  source_configuration: ProjectTransferSourceConfiguration;
  source_configuration_sha256: string;
  accepted_schema_generation: number | null;
  accepted_archive_codec: string | null;
  source_release_proof_sha256: string;
  target_activation_proof_sha256: string | null;
  archive_sha256: string | null;
  archive_size_bytes: number | null;
  can_link: boolean;
  can_run_setup: boolean;
  can_review: boolean;
  can_admit: boolean;
  can_accept_admission: boolean;
  can_release: boolean;
  can_accept_release: boolean;
  can_relay: boolean;
  can_restore_reentry: boolean;
  can_complete: boolean;
  finished: boolean;
  revision: number;
}

export type ProjectTransferIncomingProvisioningProjection = ProjectProvisioningResponse & {
  kind: "incoming_transfer";
  final_review_digest: string | null;
};

export interface TargetProviderModelProjection {
  id: string;
  label: string;
  reasoning: string[];
  default_reasoning: string;
}

export interface TargetProviderRuntimeProjection {
  id: string;
  label: string;
}

export interface TargetProviderSetupProjection {
  provider: string;
  label: string;
  installed: boolean;
  authenticated: boolean;
  version: string | null;
  reason: string | null;
  binary_path: string | null;
  path_state: string;
  models: TargetProviderModelProjection[];
  runtimes: TargetProviderRuntimeProjection[];
  default_runtime: string;
}

export interface ProjectTransferBundle {
  source: ProjectTransferProjection;
  target: ProjectTransferProjection;
  incoming_provisioning: ProjectTransferIncomingProvisioningProjection;
  target_provider_setup: TargetProviderSetupProjection[];
  can_advance: boolean;
  advance_label: string | null;
  can_manual_relay: boolean;
  finished: boolean;
}

export interface SpaceUser {
  user_id: string;
  display_name: string | null;
  identity_kind: "local_owner" | "team_member";
  created_at: string;
  updated_at: string;
  removal_started_at: string | null;
  removed_at: string | null;
}

export interface IdentityResponse {
  space_id: string;
  space_kind: "personal" | "team";
  space_name: string | null;
  user: SpaceUser;
}

export interface TeamEnrollmentResponse {
  identity: IdentityResponse;
  token: string;
}

export interface TeamInvitation {
  invitation_id: string;
  created_by: string;
  created_at: string;
  expires_at: string;
  consumed_at: string | null;
  consumed_by: string | null;
  failed_attempts: number;
  locked_at: string | null;
  revoked_at: string | null;
}

export interface TeamInvitationIssue {
  invitation: TeamInvitation;
  code: string;
  space_name: string;
}

export interface SourceRef {
  machine: string;
  truth_repository: string;
  source: string;
  session_id: string;
  record_uuid: string;
  timestamp: string;
  excerpt: string;
}

export interface GraphNode {
  id: string;
  type: "research_question" | "hypothesis" | "decision" | "experiment" | "evidence" | "blocker";
  title: string;
  extension_type?: string | null;
  extension_fields: Record<string, string | number | boolean | string[]>;
  standing: Standing;
  created_rev: number;
  updated_rev: number;
  source_refs: SourceRef[];
  status?: string;
  question?: string;
  statement?: string;
  scope?: string;
  objective?: string;
  observation?: string;
  description?: string;
  current_summary?: string;
  next_action?: string | null;
  current_summary_stale?: boolean;
  next_action_stale?: boolean;
  blocker_type?: string;
  validity?: string;
  origin?: "internal_run" | "external_publication" | "external_instance" | "analytic" | "unknown";
  role?: "result" | "diagnostic";
  legacy_strength?: "diagnostic" | "preliminary" | "supporting" | "confirmatory" | null;
  attempts?: ExperimentAttempt[];
  invocation_ceiling?: number;
  completion_criteria?: string[];
  draft_touched?: boolean;
  [key: string]: unknown;
}

export type ExtensionFieldValue = string | number | boolean | string[];

export interface ExperimentAttempt {
  id: string;
  sequence: number;
  purpose: string;
  attempt_kind: "external_run" | "proposal_only";
  decision_bundle: ExperimentDecisionPin[];
  debug?: ExperimentAttemptDebug | null;
  status: string;
  outcome?: string | null;
  failure_reason?: string | null;
  job_refs: string[];
}

export interface ExperimentDecisionPin {
  decision_id: string;
  decision_revision: number;
  selected_option: string;
}

export interface ExperimentAttemptDebug {
  mechanical_fault: string;
  change: string;
  predicted_effect: string;
}

export interface DecisionDrift {
  decision_id: string;
  pinned_option: string;
  pinned_revision: number;
  current_option: string | null;
  current_status: string | null;
}

export interface ExperimentSessionBinding {
  provider: string | null;
  model: string | null;
  reasoning: string | null;
  run_on: string | null;
  execution_host: string | null;
  run_truth_scope: string[] | null;
  native_session_bound: boolean;
  diagnostic: string | null;
}

export interface ExperimentOperationalState {
  task_active: boolean;
  detached_work_active: boolean;
  watcher_degraded: boolean;
  watcher_completion_pending: boolean;
  episode_exited: boolean;
  episode_live: boolean;
  stop_requested: boolean;
  stop_settled: boolean;
  chat_id: string | null;
  current_operation_id: string | null;
  current_status: AgentTaskStatus | null;
  current_queued: boolean;
  current_active: boolean;
  current_awaiting_human: boolean;
  current_phase: string | null;
  current_status_message: string | null;
  current_last_activity_at: string | null;
  current_invocation: number | null;
  session: ExperimentSessionBinding;
}

export type ExperimentLoopHealth =
  | "starting"
  | "agent_active"
  | "waiting_on_watchers"
  | "degraded"
  | "stopping"
  | "wrapping_up"
  | "failed"
  | "human_stopped"
  | "paused_at_limit"
  | "needs_action"
  | "completed";
export type ExperimentRecommendedStep =
  | "wait"
  | "resume"
  | "retry"
  | "keep_loop"
  | "start_episode"
  | "stop_and_restart"
  | "resolve_requirements"
  | "open_report"
  | "review"
  | "none";
export type ExperimentRunSection = "running" | "actionable" | "completed";
export type ExperimentTaskControlKind = "resume" | "retry";

export interface ExperimentControlState {
  ready: boolean;
  reasons: string[];
  graph_reasons: string[];
  invocations_used: number;
  invocation_ceiling: number;
  invocations_remaining: number;
  episode_id: string | null;
  episode: Episode | null;
  paused: boolean;
  active: boolean;
  governing_decisions: ExperimentDecisionPin[];
  decision_drift: DecisionDrift[];
  operational: ExperimentOperationalState;
  health: ExperimentLoopHealth;
  recommendation: ExperimentRecommendedStep;
  run_section: ExperimentRunSection;
  live: boolean;
  can_start: boolean;
  can_stop: boolean;
  stop_pending: boolean;
  task_control: ExperimentTaskControlKind | null;
  can_switch_provider: boolean;
  can_open_report: boolean;
  report_episode_id: string | null;
  node_closed: boolean;
}

export interface ExperimentLoopIndexEntry {
  project_id: string;
  project_name: string;
  project_reachable: boolean | null;
  graph_target: GraphTargetRef;
  graph_head: GraphHeadRef | null;
  parent_episode_id: string | null;
  node: GraphNode;
  control: ExperimentControlState;
  episode: Episode | null;
}

export interface WatcherContinuation {
  provider: string;
  model: string | null;
  reasoning: string | null;
  run_on: string;
  run_truth_scope: string[] | null;
  patch_kind: "work" | "experiment_loop";
  control_node_id: string | null;
  control_revision: number | null;
  control_episode_id: string | null;
  control_invocation: number | null;
  control_invocation_ceiling: number | null;
  control_decision_bundle: Record<string, unknown>[];
  control_completion_criteria: string[];
  workflow_ids: string[];
  skill_ids: string[];
  invoked_workflow_ids: string[];
  invoked_skill_ids: string[];
  resolved_skill_packages: SkillReference[];
}

export type GraphCondition =
  { node_id: string; status_in: string[] } | { node_id: string; proposal_resolved: true };

interface WatcherDeliveryRecord {
  watcher_id: string;
  project_id: string;
  origin_operation_id: string;
  origin_task_kind: "node_chat" | "project_chat" | "auto_research";
  chat_id: string;
  node_id: string | null;
  episode_id: string | null;
  graph_target: GraphTargetRef;
  execution_host: string;
  continuation: WatcherContinuation;
  status: "active" | "degraded" | "completed" | "stopped";
  created_at: string;
  completed_at: string | null;
  notified: boolean;
  notification_operation_id: string | null;
  stopped_by: "human" | "loop" | "agent" | null;
  stop_reason: string | null;
  stopped_at: string | null;
  stop_operation_id: string | null;
}

export interface ExternalWatcherRecord extends WatcherDeliveryRecord {
  check_command: string;
  log_path: string;
  cwd: string;
  last_checked_at: string | null;
  last_exit_code: number | null;
  last_error: string | null;
  next_check_at: string | null;
  consecutive_error_count: number;
  group_id: string | null;
  group_label: string | null;
}

export interface GraphWatcherRecord extends WatcherDeliveryRecord {
  condition: GraphCondition;
  armed_revision: number | null;
  last_evaluated_at: string | null;
  status: "active" | "completed" | "stopped";
}

export type WatcherRecord = ExternalWatcherRecord | GraphWatcherRecord;

export interface Edge {
  id: string;
  source: string;
  target: string;
  relation: string;
  layer: "epistemic" | "action" | "seam" | "meta";
  explanation: string;
  assessment?: EvidenceAssessment | null;
}

export interface EvidenceAssessment {
  relevance: "direct" | "indirect" | "contextual";
  weight: "limited" | "moderate" | "strong";
  scope?: string | null;
  qualifications: string[];
}

export type BaseNodeType = GraphNode["type"];
export type OntologyLayer = "epistemic" | "action";
export type OntologyFieldKind = "text" | "number" | "boolean" | "text_list";

export interface OntologyTypeDefinition {
  name: string;
  definition: string;
  base_type: BaseNodeType;
  layer: OntologyLayer;
  deprecated: boolean;
}

export interface OntologyFieldDefinition {
  owner_type: string;
  name: string;
  definition: string;
  kind: OntologyFieldKind;
  required: boolean;
  agent_writable: boolean;
  deprecated: boolean;
}

export interface OntologyRelationDefinition {
  name: string;
  definition: string;
  source_types: string[];
  target_types: string[];
  layer: OntologyLayer;
  deprecated: boolean;
}

export interface OntologyState {
  types: OntologyTypeDefinition[];
  fields: OntologyFieldDefinition[];
  relations: OntologyRelationDefinition[];
}

export interface BeliefTransition {
  hypothesis_id: string;
  from_status: string;
  to_status: string;
  revision: number;
  cause:
    | { kind: "evidence_edge" | "decision" | "proposal_resolution"; ref_id: string }
    | { kind: "human_edit" };
}

export interface ReplayFailure {
  revision: number;
  created_at: string;
  code: string;
  message: string;
}

export interface ProposalCard {
  situation_cold: string;
  why_human_now: string;
  consequences: string;
  decision_needed: string;
}

export interface ProposalContentChangeOperation {
  op: "update_nodes";
  intent: "content_change";
  nodes: [{ id: string; changes: Record<string, unknown> }];
}

export interface ProposalStatusChangeOperation {
  op: "update_nodes";
  intent: "status_change";
  nodes: [
    {
      id: string;
      changes: { status: string };
      cause: { kind: "evidence_edge"; ref_id: string };
    },
  ];
}

export interface ProposalRemovalOperation {
  op: "remove_nodes";
  intent: "removal";
  node_ids: [string];
}

export interface ProposalSupersedeOperation {
  op: "supersede_nodes";
  intent: "supersede";
  nodes: [{ id: string; superseded_by: string; explanation?: string }];
}

export interface ProposalMergeOperation {
  op: "merge_nodes";
  intent: "merge";
  merges: [{ duplicate: string; canonical: string; explanation?: string }];
}

export interface ProposalCreateProtectedRelationOperation {
  op: "create_edges";
  intent: "protected_relation_change";
  edges: [
    {
      id?: string | null;
      source: string;
      target: string;
      relation: string;
      explanation?: string;
    },
  ];
}

export interface ProposalRemoveProtectedRelationOperation {
  op: "remove_edges";
  intent: "protected_relation_change";
  edge_ids: [string];
}

export type CanonicalProposalOperation =
  | ProposalContentChangeOperation
  | ProposalStatusChangeOperation
  | ProposalRemovalOperation
  | ProposalSupersedeOperation
  | ProposalMergeOperation
  | ProposalCreateProtectedRelationOperation
  | ProposalRemoveProtectedRelationOperation;

interface ProposalRecord {
  id: string;
  title: string;
  card: ProposalCard;
  related_node_ids: string[];
  related_edge_ids: string[];
  related_config_keys: string[];
  base_rev: number;
  status: "pending" | "approved" | "rejected" | "withdrawn";
  created_by?: "agent" | "human";
  created_by_operation_id?: string | null;
  raised_rev: number;
  resolved_rev: number | null;
  resolved_by?: "agent" | "human" | null;
  resolved_by_operation_id?: string | null;
  resolution_reason?: string | null;
  rejection_reason?: string | null;
}

export interface CanonicalProposal extends ProposalRecord {
  semantics: "canonical";
  ops: [CanonicalProposalOperation];
}

export interface LegacyProposal extends ProposalRecord {
  semantics: "legacy";
  ops: unknown[];
}

export type Proposal = CanonicalProposal | LegacyProposal;

export interface ProposalSemantics {
  operation: CanonicalProposalOperation | null;
  resourceKeys: string[];
}

export function decodeProposal(raw: Proposal): Proposal {
  const payload = raw as Proposal & { semantics?: unknown; ops?: unknown };
  const rawOps = Array.isArray(payload.ops) ? payload.ops : [];
  const operation = rawOps.length === 1 ? decodeCanonicalProposalOperation(rawOps[0]) : null;
  const common = {
    ...payload,
    related_node_ids: stringList(payload.related_node_ids),
    related_edge_ids: stringList(payload.related_edge_ids),
    related_config_keys: stringList(payload.related_config_keys),
    raised_rev: Number.isInteger(payload.raised_rev) ? payload.raised_rev : 0,
    resolved_rev: Number.isInteger(payload.resolved_rev) ? payload.resolved_rev : null,
  };
  return operation
    ? { ...common, semantics: "canonical", ops: [operation] }
    : { ...common, semantics: "legacy", ops: rawOps };
}

export function decodeGraphState(graph: GraphState): GraphState {
  return {
    ...graph,
    proposals: Object.fromEntries(
      Object.entries(graph.proposals).map(([proposalId, proposal]) => [
        proposalId,
        decodeProposal(proposal),
      ]),
    ),
  };
}

export function decodeGraphAttentionProjection(
  value: unknown,
  graph: GraphState,
): GraphAttentionProjection {
  const keys = ["pending_proposal_ids", "decisions_awaiting_choice_ids", "open_blocker_ids"];
  if (!isPlainRecord(value) || !hasExactKeys(value, keys)) {
    throw new Error("Project attention projection is missing or malformed.");
  }
  const readIds = (field: (typeof keys)[number]): string[] => {
    const raw = value[field];
    if (!Array.isArray(raw) || raw.some((item) => !isNonEmptyString(item))) {
      throw new Error(`Project attention projection has invalid ${field}.`);
    }
    const ids = raw as string[];
    if (new Set(ids).size !== ids.length) {
      throw new Error(`Project attention projection has duplicate ${field}.`);
    }
    return [...ids];
  };
  const attention: GraphAttentionProjection = {
    pending_proposal_ids: readIds("pending_proposal_ids"),
    decisions_awaiting_choice_ids: readIds("decisions_awaiting_choice_ids"),
    open_blocker_ids: readIds("open_blocker_ids"),
  };
  for (const proposalId of attention.pending_proposal_ids) {
    if (!graph.proposals[proposalId]) {
      throw new Error(`Project attention references missing Proposal ${proposalId}.`);
    }
  }
  for (const decisionId of attention.decisions_awaiting_choice_ids) {
    if (graph.nodes[decisionId]?.type !== "decision") {
      throw new Error(`Project attention member ${decisionId} is not a Decision.`);
    }
  }
  for (const blockerId of attention.open_blocker_ids) {
    if (graph.nodes[blockerId]?.type !== "blocker") {
      throw new Error(`Project attention member ${blockerId} is not a Blocker.`);
    }
  }
  return attention;
}

export function decodeProjectSnapshot(snapshot: ProjectSnapshot): ProjectSnapshot {
  const graph = decodeGraphState(snapshot.graph);
  return {
    ...snapshot,
    graph,
    attention: decodeGraphAttentionProjection(
      (snapshot as ProjectSnapshot & { attention?: unknown }).attention,
      graph,
    ),
  };
}

export function decodeProjectTransitionResponse(
  response: ProjectTransitionResponse,
): ProjectTransitionResponse {
  const graph = decodeGraphState(response.graph);
  return {
    ...response,
    graph,
    attention: decodeGraphAttentionProjection(
      (response as ProjectTransitionResponse & { attention?: unknown }).attention,
      graph,
    ),
  };
}

export function proposalSemantics(proposal: Proposal): ProposalSemantics {
  const decoded = proposal.semantics ? proposal : decodeProposal(proposal);
  const operation = decoded.semantics === "canonical" ? decoded.ops[0] : null;
  return {
    operation,
    resourceKeys: proposalResourceKeys(decoded, operation),
  };
}

function proposalResourceKeys(
  proposal: Proposal,
  operation: CanonicalProposalOperation | null,
): string[] {
  if (!operation) {
    return [
      ...proposal.related_node_ids.map((nodeId) => `node:${nodeId}`),
      ...proposal.related_edge_ids.map((edgeId) => `edge:${edgeId}`),
      ...proposal.related_config_keys.map((configKey) => `config:${configKey}`),
    ].sort();
  }

  let resourceKeys: string[];
  switch (operation.op) {
    case "update_nodes":
      resourceKeys = [`node:${operation.nodes[0].id}`];
      break;
    case "remove_nodes":
      resourceKeys = [
        `node:${operation.node_ids[0]}`,
        ...proposal.related_edge_ids.map((edgeId) => `edge:${edgeId}`),
      ];
      break;
    case "supersede_nodes": {
      const item = operation.nodes[0];
      resourceKeys = [`node:${item.id}`, `edge:${item.id}::supersedes::${item.superseded_by}`];
      break;
    }
    case "merge_nodes": {
      const item = operation.merges[0];
      resourceKeys = [
        `node:${item.duplicate}`,
        `edge:${item.duplicate}::duplicate_of::${item.canonical}`,
      ];
      break;
    }
    case "create_edges": {
      const edge = operation.edges[0];
      resourceKeys = [`edge:${edge.id ?? `${edge.source}::${edge.relation}::${edge.target}`}`];
      break;
    }
    case "remove_edges":
      resourceKeys = [`edge:${operation.edge_ids[0]}`];
      break;
  }
  return [...new Set(resourceKeys)].sort();
}

function decodeCanonicalProposalOperation(raw: unknown): CanonicalProposalOperation | null {
  if (!isPlainRecord(raw)) return null;
  if (raw.op === "update_nodes" && raw.intent === "content_change") {
    const update = oneRecord(raw.nodes);
    if (
      !hasExactKeys(raw, ["op", "intent", "nodes"]) ||
      !update ||
      !hasExactKeys(update, ["id", "changes"]) ||
      !isNonEmptyString(update.id) ||
      !isPlainRecord(update.changes) ||
      Object.keys(update.changes).length === 0
    )
      return null;
    return raw as unknown as ProposalContentChangeOperation;
  }
  if (raw.op === "update_nodes" && raw.intent === "status_change") {
    const update = oneRecord(raw.nodes);
    const changes = update && isPlainRecord(update.changes) ? update.changes : null;
    const cause = update && isPlainRecord(update.cause) ? update.cause : null;
    if (
      !hasExactKeys(raw, ["op", "intent", "nodes"]) ||
      !update ||
      !hasExactKeys(update, ["id", "changes", "cause"]) ||
      !isNonEmptyString(update.id) ||
      !changes ||
      !hasExactKeys(changes, ["status"]) ||
      !isNonEmptyString(changes.status) ||
      !cause ||
      !hasExactKeys(cause, ["kind", "ref_id"]) ||
      cause.kind !== "evidence_edge" ||
      !isNonEmptyString(cause.ref_id)
    )
      return null;
    return raw as unknown as ProposalStatusChangeOperation;
  }
  if (raw.op === "remove_nodes" && raw.intent === "removal") {
    return hasExactKeys(raw, ["op", "intent", "node_ids"]) && oneString(raw.node_ids)
      ? (raw as unknown as ProposalRemovalOperation)
      : null;
  }
  if (raw.op === "supersede_nodes" && raw.intent === "supersede") {
    const item = oneRecord(raw.nodes);
    if (
      !hasExactKeys(raw, ["op", "intent", "nodes"]) ||
      !item ||
      !hasOnlyKeys(item, ["id", "superseded_by", "explanation"], ["id", "superseded_by"]) ||
      !isNonEmptyString(item.id) ||
      !isNonEmptyString(item.superseded_by) ||
      (item.explanation !== undefined && typeof item.explanation !== "string")
    )
      return null;
    return raw as unknown as ProposalSupersedeOperation;
  }
  if (raw.op === "merge_nodes" && raw.intent === "merge") {
    const item = oneRecord(raw.merges);
    if (
      !hasExactKeys(raw, ["op", "intent", "merges"]) ||
      !item ||
      !hasOnlyKeys(item, ["duplicate", "canonical", "explanation"], ["duplicate", "canonical"]) ||
      !isNonEmptyString(item.duplicate) ||
      !isNonEmptyString(item.canonical) ||
      (item.explanation !== undefined && typeof item.explanation !== "string")
    )
      return null;
    return raw as unknown as ProposalMergeOperation;
  }
  if (raw.op === "create_edges" && raw.intent === "protected_relation_change") {
    const edge = oneRecord(raw.edges);
    if (
      !hasExactKeys(raw, ["op", "intent", "edges"]) ||
      !edge ||
      !hasOnlyKeys(
        edge,
        ["id", "source", "target", "relation", "explanation"],
        ["source", "target", "relation"],
      ) ||
      !isNonEmptyString(edge.source) ||
      !isNonEmptyString(edge.target) ||
      !isNonEmptyString(edge.relation) ||
      (edge.id !== undefined && edge.id !== null && !isNonEmptyString(edge.id)) ||
      (edge.explanation !== undefined && typeof edge.explanation !== "string")
    )
      return null;
    return raw as unknown as ProposalCreateProtectedRelationOperation;
  }
  if (raw.op === "remove_edges" && raw.intent === "protected_relation_change") {
    return hasExactKeys(raw, ["op", "intent", "edge_ids"]) && oneString(raw.edge_ids)
      ? (raw as unknown as ProposalRemoveProtectedRelationOperation)
      : null;
  }
  return null;
}

function oneRecord(value: unknown): Record<string, unknown> | null {
  return Array.isArray(value) && value.length === 1 && isPlainRecord(value[0]) ? value[0] : null;
}

function oneString(value: unknown): string | null {
  return Array.isArray(value) && value.length === 1 && isNonEmptyString(value[0]) ? value[0] : null;
}

function hasExactKeys(value: Record<string, unknown>, keys: string[]): boolean {
  return hasOnlyKeys(value, keys, keys);
}

function hasOnlyKeys(
  value: Record<string, unknown>,
  allowedKeys: string[],
  requiredKeys: string[],
): boolean {
  const allowed = new Set(allowedKeys);
  return (
    requiredKeys.every((key) => key in value) && Object.keys(value).every((key) => allowed.has(key))
  );
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export interface Ambiguity {
  id: string;
  question: string;
  why_it_matters: string;
  related_node_ids: string[];
  status: string;
}

export interface GraphState {
  revision: number;
  nodes: Record<string, GraphNode>;
  edges: Record<string, Edge>;
  proposals: Record<string, Proposal>;
  ambiguities: Record<string, Ambiguity>;
  glossary: Record<string, GlossaryTerm>;
  ontology: OntologyState;
  validation_messages: ValidationMessage[];
  belief_transitions: BeliefTransition[];
  replay_status: "complete" | "degraded";
  replay_failure?: ReplayFailure | null;
}

export interface GlossaryTerm {
  term: string;
  plain_definition: string;
  where_defined?: string | null;
}

export interface ValidationMessage {
  level: "flag" | "reject";
  code: string;
  message: string;
  patch_revision?: number | null;
  related_node_ids: string[];
  related_edge_ids: string[];
  operation_index?: number | null;
  rule_id?: string | null;
  cause_chain?: TransitionCauseRef[];
  failed_invariant?: string | null;
}

export type GraphTargetRef =
  { kind: "main"; branch_id?: null } | { kind: "branch"; branch_id: string };

export interface GraphHeadRef {
  target: GraphTargetRef;
  revision: number;
  transition_id: string | null;
}

export interface BranchMergeProvenance {
  schema_generation: 1;
  merge_id: string;
  branch_id: string;
  episode_id: string;
  branch_base_head: GraphHeadRef;
  branch_head: GraphHeadRef;
  rebased_main_head: GraphHeadRef;
  merge_task_id: string;
}

export interface BranchMergeReceipt {
  schema_generation: 1;
  outcome: "committed" | "no_change";
  provenance: BranchMergeProvenance;
  result_main_head: GraphHeadRef;
  authorized_by: AuthorizedHuman;
  created_at: string;
}

export interface GraphBranchSummary {
  branch_id: string;
  episode_id: string;
  base_head: GraphHeadRef;
  head: GraphHeadRef;
  merge_eligible: boolean;
  merge_state: "unmerged" | "running" | "merged" | "needs_action" | "failed";
  latest_successful_merge: BranchMergeReceipt | null;
  active_merge_task_id: string | null;
  merge_diagnostic: string | null;
}

export type TransitionCauseRef =
  | { kind: "action"; action_index: number; event_id?: null }
  | { kind: "event"; action_index?: null; event_id: string };

export interface GuidanceFieldValidity {
  status: "empty" | "current" | "stale";
  invalidated_by_event_id?: string | null;
}

export interface ExperimentGuidanceValidity {
  current_summary: GuidanceFieldValidity;
  next_action: GuidanceFieldValidity;
}

export interface ProjectTransitionResponse {
  head: GraphHeadRef;
  graph: GraphState;
  attention: GraphAttentionProjection;
  experiment_control: Record<string, ExperimentControlState>;
  guidance_validity: Record<string, ExperimentGuidanceValidity>;
  ruleset_tag: string;
  transition_id: string | null;
  canonical: boolean;
  base_head?: GraphHeadRef | null;
}

export interface GraphAttentionProjection {
  pending_proposal_ids: string[];
  decisions_awaiting_choice_ids: string[];
  open_blocker_ids: string[];
}

export interface TransitionTrigger {
  operation: string;
  node_types: string[];
  node_fields: string[];
  relations: string[];
}

export interface TransitionTriggerManifest {
  ruleset_tag: string;
  triggers: TransitionTrigger[];
}

export interface TransitionPreviewResponse {
  projection: ProjectTransitionResponse;
  transition: {
    transition_id: string;
    pre_head: GraphHeadRef;
    ruleset_tag: string;
  };
}

export interface Repository {
  alias: string;
  machine: string;
  path: string;
}

export interface Machine {
  alias: string;
  host: string;
  os_account: string;
  provider_paths: Record<ProviderId, string>;
}

export interface AgentPermissions {
  read_graph: boolean;
  read_research_md: boolean;
  read_introduction: boolean;
  read_repositories: "none" | "run_scope" | "project_scope";
  read_conversations: "none" | "run_scope";
  write_graph_patch: boolean;
  write_project_files: boolean;
  write_paper: boolean;
}

/**
 * A provider id. The backend registry in `src/rcp/providers.py` is the only
 * place providers are enumerated; the frontend never hardcodes one, and reads
 * ids, labels, models, and reasoning efforts out of `provider_readiness`.
 */
export type ProviderId = string;

/** One model a provider accepts, with the reasoning efforts that model supports. */
export interface ModelChoice {
  id: string;
  label: string;
  reasoning: string[];
  default_reasoning: string;
}

export interface ProviderRuntimeChoice {
  id: string;
  label: string;
}

export interface AgentProfile {
  provider: ProviderId;
  runtime: string;
  model: string;
  reasoning: string;
  run_on: string;
  permissions: AgentPermissions;
}

export interface AgentRunConfig {
  provider: ProviderId;
  model: string;
  reasoning: string;
  run_on: string;
}

export interface AgentProfileSettings extends AgentRunConfig {
  runtime: string;
}

export interface AgentTaskReceipt {
  receipt_id: number;
  operation_id: string;
  created_at: string;
  tier: "summary" | "diagnostic" | "trace";
  category: string;
  payload: Record<string, unknown>;
}

export interface AgentTaskContract {
  operation_id: string;
  role: string;
  created_at: string;
  sha256: string;
  content: string;
}

export interface AgentTaskRequest {
  provider?: ProviderId | null;
  model?: string | null;
  reasoning?: string | null;
  run_on?: string | null;
  run_truth_scope?: string[] | null;
  chat_scope?: "node" | "project";
  node_id?: string | null;
  message?: string | null;
  chat_id?: string | null;
  attachment_set_id?: string | null;
  attachment_client_id?: string | null;
  attachments?: ChatAttachmentDescriptor[];
  session_id?: string | null;
  mode?: ConversationMode;
  artifact_context?: ArtifactContextRequest | null;
  trigger?: TaskTrigger;
  patch_kind?: GraphPatchKind;
  control_node_id?: string | null;
  control_revision?: number | null;
  control_decision_bundle?: ExperimentDecisionPin[];
  control_completion_criteria?: string[];
  watcher_ids?: string[];
  workflow_ids?: string[] | null;
  skill_ids?: string[] | null;
  invoked_workflow_ids?: string[];
  invoked_skill_ids?: string[];
  invoked_provider_skill_names?: string[];
  resolved_provider_skills?: ProviderSkillReference[];
  resolved_skill_packages?: SkillReference[] | null;
  [key: string]: unknown;
}

export type SkillKind = "skill" | "workflow";

export interface SkillReference {
  id: string;
  kind: SkillKind;
  version: string;
}

export interface SkillCatalogEntry extends SkillReference {
  label: string;
  description: string;
  dependencies: SkillReference[];
}

export interface SkillPackageDetail extends SkillCatalogEntry {
  body: string;
}

export interface SkillDefaults {
  workflow_ids: string[];
  skill_ids: string[];
}

export interface ProviderSkill {
  name: string;
  label: string;
  description: string;
  scope?: string | null;
  path?: string | null;
  enabled: boolean;
}

export interface ProviderSkillInventory {
  provider: ProviderId;
  machine: string;
  host: string;
  configured_binary?: string | null;
  resolved_binary?: string | null;
  provider_version?: string | null;
  inventory_hash?: string | null;
  refreshed_at?: string | null;
  command: string[];
  protocol?: "jsonrpc" | "jsonl" | null;
  status: "refreshing" | "fresh" | "stale" | "unavailable";
  stale: boolean;
  diagnostic?: string | null;
  skills: ProviderSkill[];
}

export interface ProviderSkillReference {
  provider: ProviderId;
  machine: string;
  provider_version: string;
  inventory_hash: string;
  name: string;
  label: string;
  description: string;
  stale: boolean;
}

export interface GraphUpdateResult {
  status: "none" | "applied" | "rejected";
  applied_revision: number | null;
  change_summary: string[];
  proposal_ids: string[];
  validation_messages: string[];
  correction_rounds: number;
  repairable: boolean;
}

export interface RevisionSummary {
  from_revision: number;
  to_revision: number;
  kind: "seed" | "refresh" | "chat" | "work" | "experiment_loop" | "approval" | "identity";
  author: "agent" | "human" | null;
  producer: "agent" | "human" | "system";
  authorized_by: AuthorizedHuman | null;
  profile: "ordinary" | "orchestrator" | null;
  task_id: string | null;
  episode_id: string | null;
  episode: HistoryEpisodeDecoration | null;
  created_at: string;
  sentences: string[];
}

export interface HistoryEpisodeDecoration {
  mode: EpisodeMode;
  state_label: string;
  ending: EpisodeEnding | null;
  wrapup_state: EpisodeWrapupState;
  report: EpisodeReportSummary | null;
}

export interface AgentTaskResult {
  messages?: string[];
  artifacts?: AgentArtifactDescriptor[];
  graph_update?: GraphUpdateResult;
  graph_updates?: GraphUpdateResult[];
  [key: string]: unknown;
}

export type AgentArtifactMediaType =
  "text/html" | "image/png" | "image/jpeg" | "image/gif" | "image/webp" | "image/svg+xml";

export interface AgentArtifactDescriptor {
  artifact_id: string;
  name: string;
  media_type: AgentArtifactMediaType;
  size_bytes?: number | null;
  kept_filename?: string | null;
  kept_at?: string | null;
  available: boolean;
  unavailable_reason: string | null;
  can_open: boolean;
  can_download: boolean;
  can_keep: boolean;
  can_revise: boolean;
}

export type ArtifactSelection =
  | {
      kind: "text";
      text: string;
      surrounding_text: string;
      comment: string;
    }
  | {
      kind: "box";
      rect: { x: number; y: number; width: number; height: number };
      viewport: { width: number; height: number };
      labels: string;
      comment: string;
    };

export interface ArtifactContextRequest {
  source?: "task" | "episode_report";
  operation_id: string;
  artifact_id: string;
  episode_id?: string | null;
  selections: ArtifactSelection[];
}

export interface AgentTask {
  operation_id: string;
  project_id: string;
  kind: AgentTaskKind;
  status: AgentTaskStatus;
  request: AgentTaskRequest;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  status_message: string;
  error?: string | null;
  applied_revision?: number | null;
  result?: AgentTaskResult | null;
  attempt: number;
  parent_operation_id?: string | null;
  episode_id?: string | null;
  runtime_id: string;
  runtime_label: string;
  native_session_id?: string | null;
  history_only: boolean;
  stage_host?: string | null;
  stage_root?: string | null;
  graph_target: GraphTargetRef;
  estimate_seconds: number;
  estimate_samples: number;
  phase: string;
  last_activity_at?: string | null;
  authorized_by?: AuthorizedHuman | null;
  elapsed_seconds: number;
  progress: number;
  can_pause: boolean;
  can_resume: boolean;
  can_retry: boolean;
  active: boolean;
  queued: boolean;
  pausing: boolean;
  awaiting_human: boolean;
  paused: boolean;
  failed: boolean;
  settled: boolean;
  finished: boolean;
  status_label: string;
  events?: AgentTaskEvent[];
  debug_receipts?: AgentTaskReceipt[];
  contracts?: AgentTaskContract[];
}

export type EpisodeHealth =
  | "starting"
  | "active"
  | "recovering"
  | "needs_action"
  | "stopping"
  | "wrapping_up"
  | "completed"
  | "stopped"
  | "failed";
export type EpisodeRecommendationKind =
  "continue" | "wait" | "resume" | "retry" | "reauthorize" | "open_report" | "review" | "none";
export type EpisodeTaskControlKind = "pause" | "resume" | "retry";
export type EpisodeRunSection = "needs_action" | "completed";
export type EpisodeMode = "auto_research" | "experiment_loop";

/**
 * Opaque on purpose. Episode lifecycle is decided by the server and reaches the
 * client already decided, as `health`, `recommendation`, `live`, and `can_*`. A
 * comparison against a status literal does not compile, so no surface can reach
 * its own conclusion about a lifecycle the backend owns.
 */
declare const OPAQUE_EPISODE_STATUS: unique symbol;
export type EpisodeStatus = { readonly [OPAQUE_EPISODE_STATUS]: "EpisodeStatus" };

export type EpisodeEnding = "completed" | "exhausted" | "stopped" | "failed" | "human_pause";

export type EpisodeWrapupState =
  "not_started" | "pending" | "running" | "ready" | "failed" | "skipped" | "legacy_unavailable";

export interface EpisodeBudgetMeter {
  invocation_ceiling: number;
  invocations_used: number;
  invocations_remaining: number;
  observed_input_tokens: number;
  observed_generated_tokens: number;
}

export interface EpisodeReportSummary {
  report_id: string;
  ending: EpisodeEnding;
  created_at: string;
}

export interface AutoResearchRecoverySummary {
  purpose: "task";
  status: "pending" | "admitted" | "exhausted" | "blocked";
  retry_mode: "exact" | "clean" | "blocked";
  operation_id: string | null;
  attempts: number;
  max_attempts: number;
  next_attempt_at: string | null;
}

export interface Episode {
  episode_id: string;
  project_id: string;
  mode: EpisodeMode;
  control_node_id: string | null;
  graph_target: GraphTargetRef;
  graph_base_head: GraphHeadRef | null;
  graph_branch: GraphBranchSummary | null;
  root_operation_id: string | null;
  current_operation_id: string | null;
  current_orchestrator_task_id: string | null;
  current_control_task_id: string | null;
  recovery: AutoResearchRecoverySummary | null;
  status: EpisodeStatus;
  starting_instruction: string | null;
  budget: EpisodeBudgetMeter;
  authorized_by: AuthorizedHuman | null;
  stop_requested_at: string | null;
  ending: EpisodeEnding | null;
  ending_diagnostic: string | null;
  wrapup_state: EpisodeWrapupState;
  wrapup_error: string | null;
  created_at: string;
  updated_at: string;
  ended_at: string | null;
  tasks: AgentTask[];
  report: EpisodeReportSummary | null;
  can_stop: boolean;
  can_reauthorize: boolean;
  can_message: boolean;
  live: boolean;
  health: EpisodeHealth;
  recommendation: EpisodeRecommendationKind;
  task_control: EpisodeTaskControlKind | null;
  run_section: EpisodeRunSection;
}

export interface EpisodeMessage {
  message_id: string;
  episode_id: string;
  sender_role: "human" | "orchestrator" | "worker";
  sender_task_id: string | null;
  authorized_by: AuthorizedHuman | null;
  recipient_task_id: string;
  control_node_id: string | null;
  body: string;
  created_at: string;
  delivered_at: string | null;
  delivery_operation_id: string | null;
}

export interface StartEpisodeRequest {
  mode: "auto_research";
  invocation_ceiling: number;
  starting_instruction?: string | null;
}

export interface AgentUsageRecord {
  usage_id: string;
  project_id: string;
  operation_id: string;
  task_kind: AgentTaskKind;
  provider: string;
  model?: string | null;
  provider_profile: string;
  provider_event_type: string;
  dedupe_key: string;
  counted: boolean;
  count_reason: "counted" | "duplicate" | "invalid";
  created_at: string;
  processed_input_tokens: number;
  generated_tokens: number;
  cached_input_tokens: number;
  cache_creation_input_tokens: number;
  cache_write_input_tokens: number;
  reasoning_output_tokens: number;
  reported_input_tokens?: number | null;
  reported_output_tokens?: number | null;
  reported_total_tokens?: number | null;
  provider_fields: Record<string, unknown>;
}

export interface AgentUsageCell {
  task_kind: AgentTaskKind;
  provider: string;
  processed_input_tokens: number;
  generated_tokens: number;
  cached_input_tokens: number;
  counted_records: number;
}

export interface AgentUsageMetric {
  total_tokens: number;
  cached_tokens: number;
  cache_share: number;
  block_percent: number;
  block_tokens: number;
  cells: AgentUsageCell[];
}

export interface AgentUsageSnapshot {
  project_id: string;
  input_processed: AgentUsageMetric;
  generated: AgentUsageMetric;
  counted_records: number;
  excluded_records: number;
  records: AgentUsageRecord[];
}

export type StartAgentTask = (kind: AgentTaskKind, request: AgentTaskRequest) => Promise<AgentTask>;

export interface ChatSummary {
  chat_id: string;
  kind: "node_chat" | "project_chat";
  node_id: string | null;
  title: string;
  updated_at: string;
  message_count: number;
  last_message_preview: string;
}

export interface ChatSummaryPage {
  items: ChatSummary[];
  total: number;
  offset: number;
  limit: number;
}

export interface ChatMessage {
  message_id: string;
  operation_id?: string | null;
  role: "user" | "assistant";
  text: string;
  timestamp: string;
  native_session_id: string | null;
  provider: string | null;
  model: string | null;
  reasoning: string | null;
  execution_machine: string | null;
  applied_revision: number | null;
  mode: ConversationMode | null;
  graph_update: GraphUpdateResult | null;
  trigger: TaskTrigger;
  attachments: ChatAttachmentDescriptor[];
}

export interface ChatAttachmentDescriptor {
  attachment_id: string;
  name: string;
  media_type: string;
  size: number;
  expires_at: string;
}

export interface ChatTranscript extends ChatSummary {
  messages: ChatMessage[];
}

export interface AgentTaskEvent {
  event_id: number;
  operation_id: string;
  created_at: string;
  level: "info" | "warning" | "error";
  message: string;
  event_kind: "message" | "command";
  command_id: string | null;
  episode_id: string | null;
  command_verb: string | null;
  command_phase: "start" | "exit" | null;
  idempotency_key: string | null;
  payload: Record<string, unknown> | null;
}

export interface ProviderReadiness {
  provider: ProviderId;
  label: string;
  installed: boolean;
  authenticated: boolean;
  binary_path: string | null;
  path_state: "resolved" | "missing" | "denied" | "unconfigured" | "unreachable";
  version?: string | null;
  reason?: string | null;
  models: ModelChoice[];
  runtimes: ProviderRuntimeChoice[];
  /** The runtime an omitted manifest value resolves to. Never derive this. */
  default_runtime: string;
}

export interface PaperSnapshot {
  content: string;
  sync_state: "not_created" | "synced" | "unsynced" | "behind";
  base_hash?: string | null;
  canonical_hash?: string | null;
  incoming_content?: string | null;
  updated_at?: string | null;
  canonical_available: boolean;
}

export interface ProjectSnapshot {
  id: string;
  home_space_id: string | null;
  name: string;
  revision: number;
  snapshot_freshness: "fresh" | "reconciling" | "stale";
  last_remote_sync_at: string | null;
  state_repository: string;
  canonical_state: {
    remote: boolean;
    reachable: boolean;
    location: string;
    last_synced_at?: string | null;
    error?: string | null;
  };
  run_on: string;
  project_truth_scope: string[];
  default_run_truth_scope: string[];
  default_auto_research_invocation_ceiling: number;
  repositories: Repository[];
  machines: Machine[];
  primary_question?: GraphNode | null;
  last_refresh_at?: string | null;
  experiment_control: Record<string, ExperimentControlState>;
  attention: GraphAttentionProjection;
  counts: {
    pending_proposals: number;
    decisions_awaiting_choice: number;
    open_blockers: number;
    asserted: number;
    accepted: number;
    contested: number;
  };
  coverage: {
    repositories_seen: string[];
    repositories_never_seen: string[];
    sessions_read: string[];
    sessions_skipped: string[];
    earliest_timestamp?: string | null;
    note: string;
  };
  graph: GraphState;
  paper: PaperSnapshot;
  paper_coach: {
    default_provider: ProviderId;
    default_model: string;
    default_reasoning: string;
  };
  agent_profiles: Record<AgentExecutionProfile, AgentProfile>;
  skill_catalog: SkillCatalogEntry[];
  skill_defaults: SkillDefaults;
  provider_skill_inventories: Record<
    string,
    Partial<Record<ProviderId, ProviderSkillInventory | null>>
  >;
  provider_readiness: Record<string, Record<ProviderId, ProviderReadiness>>;
  providers: Record<ProviderId, ProviderReadiness>;
  cache_metrics: ProjectCacheMetrics;
  validation_messages: ValidationMessage[];
}

export interface CacheMetric {
  bytes: number;
  count: number;
  limits: {
    max_bytes: number;
    max_count: number;
    ttl_seconds: number;
  };
  oldest_accessed_at?: string | null;
  reclaimable_bytes: number;
  reclaimable_count: number;
}

export interface ProjectCacheMetrics {
  remote_sources: CacheMetric;
  session_slices: CacheMetric;
}

export interface ProjectCard {
  id: string;
  home_space_id: string | null;
  name: string;
  locator: string;
  state_location: string;
  remote: boolean;
  last_opened_at?: string | null;
  revision?: number | null;
  primary_question?: string | null;
  attention_count: number;
  last_refresh_at?: string | null;
  reachable?: boolean | null;
  error?: string | null;
  can_delete: boolean;
  delete_unavailable_reason: string | null;
}

export interface ProjectMember {
  user_id: string;
  display_name: string | null;
  seated_at: string;
}

export interface ProjectInvitation {
  invitation_id: string;
  project_id: string;
  project_name: string;
  space_name: string | null;
  invited_by: string;
  invited_by_name: string | null;
  created_at: string;
}

export interface SetupRepository {
  alias: string;
  location: "local" | "ssh";
  path: string;
  host: string;
  default_read: boolean;
}

export interface SetupExecution {
  location: "local" | "ssh";
  host: string;
}

export interface SetupAgentProfile {
  provider: ProviderId;
  runtime: string;
  model: string;
  reasoning: string;
  location: "local" | "ssh";
  host: string;
}

export type SetupAgents = Record<AgentExecutionProfile, SetupAgentProfile>;

export type ExistingResearchAction =
  "open_existing" | "open_degraded_read_only" | "archive_and_create";

export interface ProjectSetupRequest {
  name: string;
  repositories: SetupRepository[];
  state_repository: string;
  default_auto_research_invocation_ceiling: number;
  execution: SetupExecution;
  agents: SetupAgents;
  confirmed: boolean;
  existing_research_action?: ExistingResearchAction | null;
  existing_research_token?: string | null;
}

export interface ProjectSettingsRequest {
  default_run_truth_scope: string[];
  default_auto_research_invocation_ceiling: number;
  agent_profiles: Record<AgentExecutionProfile, AgentProfileSettings>;
  skill_defaults: SkillDefaults;
  machine_provider_paths?: Record<string, Record<ProviderId, string>>;
}

export interface ProviderPathResolution {
  machine: string;
  provider: ProviderId;
  binary_path: string | null;
  readiness: ProviderReadiness;
  project: ProjectSnapshot;
}

export interface SetupCheck {
  label: string;
  status: "pass" | "warn" | "fail";
  detail: string;
}

export interface ExistingResearchPreview {
  project_name: string;
  canonical_location: string;
  retained_revision_count: number;
  replay_status: "complete" | "degraded";
  coherent_revision: number;
  archive_token: string;
  replay_failure?: {
    revision: number | null;
    code: string;
    message: string;
  } | null;
}

export type SetupAvailableAction = "create" | ExistingResearchAction;

export interface SetupPreview {
  checks: SetupCheck[];
  can_create: boolean;
  action: "create" | "connect";
  canonical_location: string;
  existing_project_name?: string | null;
  existing_research?: ExistingResearchPreview | null;
  available_actions: SetupAvailableAction[];
  manifest_preview: string;
  remote_write: boolean;
  providers: Record<ProviderId, ProviderReadiness>;
  agent_readiness: Record<AgentExecutionProfile, ProviderReadiness>;
}

export interface WritingSession {
  provider: ProviderId;
  runtime_id: string;
  runtime_label: string;
  native_session_id: string;
  execution_machine: string;
  project_id: string;
  title?: string | null;
  model: string;
  reasoning?: string | null;
  created_at: string;
  last_resumed_at: string;
  introduction_hash_examined: string;
  graph_revision_examined: number;
  research_md_hash_examined: string;
}
