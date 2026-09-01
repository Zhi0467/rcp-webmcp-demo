//! Native personal-to-team project transfer relay.
//!
//! This module deliberately keeps credentials, archive bytes, and proof bytes
//! on the Rust side of the desktop boundary. JavaScript supplies only public
//! request/project identities and target placement intent; it never receives
//! the archive or either transition proof.

use std::{
    collections::HashSet,
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use reqwest::{
    header::{HeaderValue, CONTENT_LENGTH, CONTENT_TYPE},
    Client, Response,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tauri::{ipc::Channel, AppHandle, Manager};
use uuid::{Uuid, Version as UuidVersion};

use crate::{
    backend::{self, BackendState},
    lifecycle::DesktopStatus,
    server_commands::{
        self, ProjectProvisionReadback, ServerCommandRunResult, TerminalLaunchResult,
    },
    team_connections::{TeamConnectionMetadata, TeamConnectionState},
    team_session::{ProjectTransferTargetReadback, TeamSessionState},
    team_tunnel::TeamTunnelState,
};

const PERSONAL_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const TRANSFER_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_PROOF_BYTES: usize = 32;
const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const TRANSFER_ARCHIVE_CONTENT_TYPE: &str = "application/octet-stream";
const SOURCE_REQUEST_PATH_PREFIX: &str = "/api/project-transfers/requests/";
const SOURCE_ARCHIVE_PATH_PREFIX: &str = "/api/native/project-transfers/source-requests/";
const SOURCE_CREATE_PATH: &str = "/api/project-transfers/source-requests";
const SOURCE_LINK_PATH_PREFIX: &str = "/api/project-transfers/source-requests/";
const COORDINATOR_VERSION: u32 = 1;
const COORDINATOR_FILENAME: &str = "project-transfer-coordinator.json";
const MAX_COORDINATOR_BYTES: u64 = 1024 * 1024;
const MAX_ADVANCE_TRANSITIONS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
struct PinnedPersonalBackend {
    base_url: String,
    instance_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceTransferRequest {
    request_id: String,
    phase: String,
    target_request_id: String,
    project_id: String,
    source_space_id: String,
    target_space_id: String,
    target_activation_proof_sha256: String,
    archive_sha256: String,
    archive_size_bytes: u64,
}

/// Public, browser-supplied intent for the one target provisioning request.
///
/// This deliberately contains target placement and public setup choices only.
/// Repository identities and truth scopes come from the source record.
/// Credentials, deploy keys, proofs, archives, and operator argv are not part
/// of the native input type.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferTargetProvisioningIntent {
    pub name: String,
    pub default_auto_research_invocation_ceiling: u64,
    pub machines: Vec<ProjectTransferMachineIntent>,
    pub provider_checks: Vec<ProjectTransferProviderIntent>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferMachineIntent {
    pub alias: String,
    pub location: String,
    pub host: String,
    pub os_account: String,
    pub central_root: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferProviderIntent {
    pub profile: String,
    pub provider: String,
    pub runtime_id: String,
    pub model: String,
    pub reasoning: String,
    pub machine_alias: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferPrepareRequest {
    pub source_request_id: String,
    pub target_request_id: String,
    pub connection_id: String,
    pub source_project_id: String,
    pub target_provisioning: ProjectTransferTargetProvisioningIntent,
}

/// The native-only record that joins the two backend request identities while
/// the transfer is being prepared. This is deliberately not a third backend
/// request: it is local crash-recovery state, keyed by the source request.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProjectTransferCoordinatorRecord {
    source_request_id: String,
    target_request_id: String,
    connection_id: String,
    source_project_id: String,
    target_space_id: String,
    target_provisioning: ProjectTransferTargetProvisioningIntent,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProjectTransferCoordinatorFile {
    version: u32,
    records: Vec<ProjectTransferCoordinatorRecord>,
}

impl Default for ProjectTransferCoordinatorFile {
    fn default() -> Self {
        Self {
            version: COORDINATOR_VERSION,
            records: Vec::new(),
        }
    }
}

/// Durable state for one or more interrupted native transfer preparations.
/// The file is written with the same temp-file, fsync, rename, and directory
/// fsync boundary used by the desktop connection registry.
pub struct ProjectTransferCoordinatorState {
    coordinator_path: PathBuf,
    lock: Mutex<()>,
}

impl ProjectTransferCoordinatorState {
    pub fn for_app(app: &AppHandle) -> Result<Self, String> {
        let config_dir = app
            .path()
            .app_config_dir()
            .map_err(|error| format!("cannot locate RCP desktop configuration: {error}"))?;
        Ok(Self::new(config_dir.join(COORDINATOR_FILENAME)))
    }

    fn new(coordinator_path: PathBuf) -> Self {
        Self {
            coordinator_path,
            lock: Mutex::new(()),
        }
    }

    fn save_or_validate(
        &self,
        request: &ProjectTransferPrepareRequest,
        target_space_id: &str,
    ) -> Result<ProjectTransferPrepareRequest, String> {
        validate_prepare_request(request)?;
        validate_uuid4(target_space_id, "target transfer space identity")?;
        let candidate = ProjectTransferCoordinatorRecord {
            source_request_id: request.source_request_id.clone(),
            target_request_id: request.target_request_id.clone(),
            connection_id: request.connection_id.clone(),
            source_project_id: request.source_project_id.clone(),
            target_space_id: target_space_id.to_string(),
            target_provisioning: request.target_provisioning.clone(),
        };
        let _guard = self.acquire()?;
        let mut file = self.read_file()?;
        if let Some(existing) = file
            .records
            .iter()
            .find(|record| record.source_request_id == candidate.source_request_id)
        {
            if existing != &candidate {
                return Err(
                    "an interrupted transfer already binds this source request to another target intent"
                        .into(),
                );
            }
            return Ok(existing.as_request());
        }
        file.records.push(candidate.clone());
        self.write_file(&file)?;
        Ok(candidate.as_request())
    }

    fn load(
        &self,
        source_request_id: &str,
    ) -> Result<Option<ProjectTransferCoordinatorRecord>, String> {
        validate_uuid4(source_request_id, "source transfer request identity")?;
        let _guard = self.acquire()?;
        Ok(self
            .read_file()?
            .records
            .into_iter()
            .find(|record| record.source_request_id == source_request_id))
    }

    fn remove(
        &self,
        request: &ProjectTransferPrepareRequest,
        target_space_id: &str,
    ) -> Result<(), String> {
        validate_prepare_request(request)?;
        validate_uuid4(target_space_id, "target transfer space identity")?;
        let expected = ProjectTransferCoordinatorRecord {
            source_request_id: request.source_request_id.clone(),
            target_request_id: request.target_request_id.clone(),
            connection_id: request.connection_id.clone(),
            source_project_id: request.source_project_id.clone(),
            target_space_id: target_space_id.to_string(),
            target_provisioning: request.target_provisioning.clone(),
        };
        let _guard = self.acquire()?;
        let mut file = self.read_file()?;
        let Some(index) = file
            .records
            .iter()
            .position(|record| record.source_request_id == expected.source_request_id)
        else {
            return Ok(());
        };
        if file.records[index] != expected {
            return Err("the durable transfer coordinator record changed unexpectedly".into());
        }
        file.records.remove(index);
        self.write_file(&file)
    }

    fn acquire(&self) -> Result<MutexGuard<'_, ()>, String> {
        self.lock
            .lock()
            .map_err(|_| "the project transfer coordinator lock is unavailable".to_string())
    }

    fn read_file(&self) -> Result<ProjectTransferCoordinatorFile, String> {
        match fs::symlink_metadata(&self.coordinator_path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    return Err("the project transfer coordinator is not a regular file".into());
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "cannot inspect the project transfer coordinator: {error}"
                ));
            }
        }
        let bytes = match fs::read(&self.coordinator_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ProjectTransferCoordinatorFile::default());
            }
            Err(error) => {
                return Err(format!(
                    "cannot read the project transfer coordinator: {error}"
                ));
            }
        };
        if bytes.len() as u64 > MAX_COORDINATOR_BYTES {
            return Err("the project transfer coordinator is too large".into());
        }
        let file: ProjectTransferCoordinatorFile = serde_json::from_slice(&bytes)
            .map_err(|error| format!("the project transfer coordinator is invalid: {error}"))?;
        if file.version != COORDINATOR_VERSION {
            return Err(format!(
                "project transfer coordinator version {} is unsupported",
                file.version
            ));
        }
        if file.records.len() > 64 {
            return Err("the project transfer coordinator contains too many records".into());
        }
        let mut source_request_ids = HashSet::new();
        for record in &file.records {
            record.validate()?;
            if !source_request_ids.insert(&record.source_request_id) {
                return Err(
                    "the project transfer coordinator has duplicate source identities".into(),
                );
            }
        }
        Ok(file)
    }

    fn write_file(&self, file: &ProjectTransferCoordinatorFile) -> Result<(), String> {
        let mut bytes = serde_json::to_vec_pretty(file).map_err(|error| {
            format!("cannot serialize the project transfer coordinator: {error}")
        })?;
        bytes.push(b'\n');
        if bytes.len() as u64 > MAX_COORDINATOR_BYTES {
            return Err("the project transfer coordinator is too large".into());
        }
        let parent = self.coordinator_path.parent().ok_or_else(|| {
            "the project transfer coordinator has no parent directory".to_string()
        })?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create RCP desktop configuration: {error}"))?;
        if let Ok(metadata) = fs::symlink_metadata(&self.coordinator_path) {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err("the project transfer coordinator is not a regular file".into());
            }
        }
        let mut temporary = tempfile::Builder::new()
            .prefix(".project-transfer-coordinator-")
            .tempfile_in(parent)
            .map_err(|error| format!("cannot create the project transfer coordinator: {error}"))?;
        secure_coordinator_permissions(temporary.as_file())?;
        temporary
            .write_all(&bytes)
            .and_then(|()| temporary.as_file().sync_all())
            .map_err(|error| format!("cannot save the project transfer coordinator: {error}"))?;
        temporary.persist(&self.coordinator_path).map_err(|error| {
            format!(
                "cannot publish the project transfer coordinator: {}",
                error.error
            )
        })?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                format!("cannot finish saving the project transfer coordinator: {error}")
            })?;
        Ok(())
    }
}

impl ProjectTransferCoordinatorRecord {
    fn as_request(&self) -> ProjectTransferPrepareRequest {
        ProjectTransferPrepareRequest {
            source_request_id: self.source_request_id.clone(),
            target_request_id: self.target_request_id.clone(),
            connection_id: self.connection_id.clone(),
            source_project_id: self.source_project_id.clone(),
            target_provisioning: self.target_provisioning.clone(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        validate_prepare_request(&self.as_request())?;
        validate_uuid4(&self.target_space_id, "target transfer space identity")
    }
}

#[cfg(unix)]
fn secure_coordinator_permissions(file: &File) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|error| format!("cannot secure the project transfer coordinator: {error}"))
}

#[cfg(not(unix))]
fn secure_coordinator_permissions(_file: &File) -> Result<(), String> {
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferRepositorySource {
    pub alias: String,
    pub repository: ProjectTransferRepositoryIdentity,
    pub machine_alias: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferRepositoryIdentity {
    pub identity: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferSourceConfiguration {
    pub source_rcp_version: String,
    pub source_schema_generation: u64,
    pub supported_archive_codecs: Vec<String>,
    pub machine_aliases: Vec<String>,
    pub repositories: Vec<ProjectTransferRepositorySource>,
    pub state_repository: String,
    pub project_truth_scope: Vec<String>,
    pub default_run_truth_scope: Vec<String>,
    pub source_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferRepositoryBinding {
    pub alias: String,
    pub repository: ProjectTransferRepositoryIdentity,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferLinkReceipt {
    pub source_request_id: String,
    pub target_request_id: String,
    pub project_id: String,
    pub source_space_id: String,
    pub target_space_id: String,
    pub source_configuration_sha256: String,
    pub target_repositories: Vec<ProjectTransferRepositoryBinding>,
    pub accepted_schema_generation: u64,
    pub accepted_archive_codec: String,
    pub source_release_proof_sha256: String,
    pub target_activation_proof_sha256: String,
    pub created_at: String,
}

/// Public receipt written by the target admission transition.  This is kept
/// native-only: the browser may render the lifecycle decisions, but it never
/// supplies or forwards the receipt that authorizes the next mutation.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProjectTransferResolvedPath {
    repository_alias: String,
    machine_alias: String,
    path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProjectTransferTargetAdmissionReceipt {
    source_request_id: String,
    target_request_id: String,
    project_id: String,
    source_space_id: String,
    target_space_id: String,
    admitted_by: ProjectProvisioningAuthorizedHuman,
    source_configuration_sha256: String,
    target_preparation_revision: u64,
    target_preparation_sha256: String,
    resolved_paths: Vec<ProjectTransferResolvedPath>,
    accepted_schema_generation: u64,
    accepted_archive_codec: String,
    source_release_proof_sha256: String,
    target_activation_proof_sha256: String,
    created_at: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProjectTransferSourceReleaseReceipt {
    source_request_id: String,
    target_request_id: String,
    project_id: String,
    source_space_id: String,
    target_space_id: String,
    released_by: ProjectProvisioningAuthorizedHuman,
    source_configuration_sha256: String,
    target_admission_sha256: String,
    target_preparation_revision: u64,
    target_preparation_sha256: String,
    source_head: TransferGraphHead,
    accepted_schema_generation: u64,
    accepted_archive_codec: String,
    source_release_proof_sha256: String,
    target_activation_proof_sha256: String,
    created_at: String,
}

/// The source-side read-only boundary is the only input accepted by the
/// source release mutation.  The coordinator captures it immediately before
/// release and echoes it exactly; Web code never sees a release receipt.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProjectTransferSourceBoundaryResponse {
    source_configuration: ProjectTransferSourceConfiguration,
    source_configuration_sha256: String,
    source_head: TransferGraphHead,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectTransferProjection {
    pub request_id: String,
    pub side: String,
    pub phase: String,
    pub phase_label: Option<String>,
    pub next_action: Option<String>,
    pub linked_request_id: Option<String>,
    pub project_id: String,
    pub source_space_id: String,
    pub target_space_id: String,
    pub source_configuration: ProjectTransferSourceConfiguration,
    pub source_configuration_sha256: String,
    pub accepted_schema_generation: Option<u64>,
    pub accepted_archive_codec: Option<String>,
    pub source_release_proof_sha256: String,
    pub target_activation_proof_sha256: Option<String>,
    pub archive_sha256: Option<String>,
    pub archive_size_bytes: Option<u64>,
    pub can_link: bool,
    pub can_run_setup: bool,
    pub can_review: bool,
    pub can_admit: bool,
    pub can_accept_admission: bool,
    pub can_release: bool,
    pub can_accept_release: bool,
    pub can_relay: bool,
    pub can_restore_reentry: bool,
    pub can_complete: bool,
    pub finished: bool,
    pub revision: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectTransferBundle {
    pub source: ProjectTransferProjection,
    pub target: ProjectTransferProjection,
    pub incoming_provisioning: ProjectProvisioningProjection,
    pub target_provider_setup: Vec<TargetProviderSetupProjection>,
    /// Aggregate decisions are computed only from the backend-published
    /// `can_*` answers.  They let the browser render one final-review action
    /// without interpreting transfer phases.
    pub can_advance: bool,
    pub advance_label: Option<String>,
    pub can_manual_relay: bool,
    pub finished: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectProvisioningProjection {
    pub request_id: String,
    pub kind: String,
    pub status: String,
    pub status_label: String,
    pub next_action: Option<String>,
    pub can_run_setup: bool,
    pub can_review: bool,
    pub can_cancel: bool,
    pub target_space_id: String,
    pub proposed_project_id: String,
    pub name: Option<String>,
    pub state_repository: Option<String>,
    pub project_truth_scope: Vec<String>,
    pub default_run_truth_scope: Vec<String>,
    pub default_auto_research_invocation_ceiling: u64,
    pub authorized_by: ProjectProvisioningAuthorizedHuman,
    pub machines: Vec<ProjectProvisioningMachineProjection>,
    pub repositories: Vec<ProjectProvisioningRepositoryProjection>,
    pub provider_checks: Vec<ProjectProvisioningProviderProjection>,
    pub readiness: ProjectProvisioningReadinessProjection,
    pub diagnostic: Option<String>,
    pub operator_action: Option<Value>,
    pub operator_argv: Vec<String>,
    pub final_review: Option<ProjectProvisioningFinalReviewProjection>,
    pub final_review_digest: Option<String>,
    pub cancellation_disposition: Option<String>,
    pub revision: u64,
    pub created_at: String,
    pub updated_at: String,
    pub setup_started_at: Option<String>,
    pub completed_at: Option<String>,
    pub cancelled_at: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectProvisioningAuthorizedHuman {
    pub space_id: String,
    pub user_id: String,
    pub display_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectProvisioningFinalReviewProjection {
    pub digest: String,
    pub proposed_project_id: String,
    pub authorized_by: ProjectProvisioningAuthorizedHuman,
    pub ready_at: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectProvisioningMachineProjection {
    pub alias: String,
    pub location: String,
    pub host: String,
    pub os_account: String,
    pub intended_central_root: Option<String>,
    pub resolved_central_root: Option<String>,
    pub ready: bool,
    pub status_label: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectProvisioningRepositoryProjection {
    pub alias: String,
    pub repository: ProjectTransferRepositoryIdentity,
    pub https_clone_url: String,
    pub ssh_clone_url: String,
    pub settings_url: String,
    pub machine_alias: String,
    pub intended_path: Option<String>,
    pub resolved_path: Option<String>,
    pub checkout_disposition: Option<String>,
    pub status: String,
    pub status_label: String,
    pub ready: bool,
    pub commit: Option<String>,
    pub write_verified: bool,
    pub deploy_key_label: Option<String>,
    pub public_key_fingerprint: Option<String>,
    pub checked_at: Option<String>,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectProvisioningProviderProjection {
    pub profile: String,
    pub provider: String,
    pub runtime_id: String,
    pub model: String,
    pub reasoning: String,
    pub machine_alias: String,
    pub status: String,
    pub status_label: String,
    pub ready: bool,
    pub binary_path: Option<String>,
    pub version: Option<String>,
    pub resolved_runtime_id: Option<String>,
    pub execution_account: Option<String>,
    pub checked_at: Option<String>,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectProvisioningReadinessProjection {
    pub machines_ready: u64,
    pub machines_total: u64,
    pub repositories_ready: u64,
    pub repositories_total: u64,
    pub providers_ready: u64,
    pub providers_total: u64,
    pub all_ready: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct TargetProviderSetupProjection {
    pub provider: String,
    pub label: String,
    pub installed: bool,
    pub authenticated: bool,
    pub version: Option<String>,
    pub reason: Option<String>,
    pub binary_path: Option<String>,
    pub path_state: String,
    pub models: Vec<TargetProviderModelProjection>,
    pub runtimes: Vec<TargetProviderRuntimeProjection>,
    pub default_runtime: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct TargetProviderModelProjection {
    pub id: String,
    pub label: String,
    pub reasoning: Vec<String>,
    pub default_reasoning: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct TargetProviderRuntimeProjection {
    pub id: String,
    pub label: String,
}

#[derive(Clone, Debug)]
struct ProjectTransferRecord {
    request_id: String,
    side: String,
    phase: String,
    phase_label: Option<String>,
    next_action: Option<String>,
    linked_request_id: Option<String>,
    project_id: String,
    source_space_id: String,
    target_space_id: String,
    source_configuration: Option<ProjectTransferSourceConfiguration>,
    source_configuration_sha256: Option<String>,
    target_admission_receipt: Option<ProjectTransferTargetAdmissionReceipt>,
    source_release_receipt: Option<ProjectTransferSourceReleaseReceipt>,
    accepted_schema_generation: Option<u64>,
    accepted_archive_codec: Option<String>,
    source_release_proof_sha256: Option<String>,
    target_activation_proof_sha256: Option<String>,
    archive_sha256: Option<String>,
    archive_size_bytes: Option<u64>,
    can_link: bool,
    can_run_setup: bool,
    can_review: bool,
    can_admit: bool,
    can_accept_admission: bool,
    can_release: bool,
    can_accept_release: bool,
    can_relay: bool,
    can_restore_reentry: bool,
    can_complete: bool,
    finished: bool,
    revision: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectTransferRunResult {
    pub request_id: String,
    pub target_request_id: String,
    pub target_space_id: String,
    pub connection_id: String,
    pub archive_sha256: String,
    pub archive_size_bytes: u64,
    pub exit_code: i32,
    pub event_count: usize,
    pub proof_verified: bool,
    pub cleanup_acknowledged: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectTransferFinishResult {
    pub request_id: String,
    pub target_request_id: String,
    pub target_space_id: String,
    pub connection_id: String,
    pub proof_verified: bool,
    pub cleanup_acknowledged: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectTransferExportResult {
    pub saved: bool,
    pub request_id: String,
    pub target_request_id: Option<String>,
    pub target_space_id: Option<String>,
    pub archive_sha256: Option<String>,
    pub archive_size_bytes: Option<u64>,
    pub path: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectTransferExportSelectionResult {
    pub selected: bool,
    pub request_id: String,
    pub target_request_id: Option<String>,
    pub target_space_id: Option<String>,
    pub archive_sha256: Option<String>,
    pub archive_size_bytes: Option<u64>,
    pub path: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectTransferExportCleanupResult {
    pub request_id: String,
    pub removed: bool,
    pub path: String,
}

impl ProjectTransferExportResult {
    pub fn cancelled(request_id: &str) -> Self {
        Self {
            saved: false,
            request_id: request_id.to_string(),
            target_request_id: None,
            target_space_id: None,
            archive_sha256: None,
            archive_size_bytes: None,
            path: None,
        }
    }
}

impl ProjectTransferExportSelectionResult {
    pub fn cancelled(request_id: &str) -> Self {
        Self {
            selected: false,
            request_id: request_id.to_string(),
            target_request_id: None,
            target_space_id: None,
            archive_sha256: None,
            archive_size_bytes: None,
            path: None,
        }
    }
}

/// The only cross-space value returned by the source proof endpoint.
///
/// It is parsed and re-serialized as this strict type before being sent to the
/// target cleanup endpoint.  In particular, the target never receives an
/// arbitrary JSON object supplied by the browser.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectTransferCleanupAcknowledgment {
    pub source_request_id: String,
    pub target_request_id: String,
    pub project_id: String,
    pub source_space_id: String,
    pub target_space_id: String,
    pub source_release_proof_sha256: String,
    pub target_activation_proof_sha256: String,
    pub archive_sha256: String,
    pub source_fence_head: TransferGraphHead,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TransferGraphHead {
    pub target: TransferGraphTarget,
    pub revision: u64,
    #[serde(default)]
    pub transition_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TransferGraphTarget {
    pub kind: String,
    #[serde(default)]
    pub branch_id: Option<String>,
}

impl ProjectTransferCleanupAcknowledgment {
    pub(crate) fn validate(&self) -> Result<(), String> {
        for (value, label) in [
            (&self.source_request_id, "source transfer request identity"),
            (&self.target_request_id, "target transfer request identity"),
            (&self.project_id, "transfer project identity"),
            (&self.source_space_id, "source transfer space identity"),
            (&self.target_space_id, "target transfer space identity"),
        ] {
            validate_uuid4(value, label)?;
        }
        if self.source_space_id == self.target_space_id {
            return Err("transfer cleanup acknowledgment must cross spaces".into());
        }
        for (value, label) in [
            (
                &self.source_release_proof_sha256,
                "source release proof commitment",
            ),
            (
                &self.target_activation_proof_sha256,
                "target activation proof commitment",
            ),
            (&self.archive_sha256, "transfer archive digest"),
        ] {
            validate_digest(value, label)?;
        }
        if self.source_fence_head.target.kind != "main"
            || self.source_fence_head.target.branch_id.is_some()
        {
            return Err("transfer cleanup acknowledgment must bind the fenced main head".into());
        }
        if let Some(transition_id) = &self.source_fence_head.transition_id {
            validate_digest(transition_id, "transfer fence transition identity")?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
struct SourceCreateBody<'a> {
    request_id: &'a str,
    project_id: &'a str,
    target_space_id: &'a str,
}

#[derive(Debug, Serialize)]
struct IncomingProvisioningCreateBody<'a> {
    request_id: &'a str,
    source_project_id: &'a str,
    name: &'a str,
    state_repository: &'a str,
    project_truth_scope: &'a [String],
    default_run_truth_scope: &'a [String],
    default_auto_research_invocation_ceiling: u64,
    machines: &'a [ProjectTransferMachineIntent],
    repositories: Vec<DerivedRepositoryIntent>,
    provider_checks: &'a [ProjectTransferProviderIntent],
}

#[derive(Debug, Serialize)]
struct DerivedRepositoryIntent {
    alias: String,
    source: String,
    machine_alias: String,
}

#[derive(Debug, Serialize)]
struct TargetTransferCreateBody<'a> {
    provisioning_request_id: &'a str,
    source_request_id: &'a str,
    source_project_id: &'a str,
    source_space_id: &'a str,
    source_configuration: &'a ProjectTransferSourceConfiguration,
    source_configuration_sha256: &'a str,
    source_release_proof_sha256: &'a str,
    accepted_schema_generation: u64,
    accepted_archive_codec: &'a str,
}

#[derive(Debug, Serialize)]
struct SourceLinkBody<'a> {
    receipt: &'a ProjectTransferLinkReceipt,
}

#[derive(Debug, Serialize)]
struct TargetAdmissionBody<'a> {
    receipt: &'a ProjectTransferTargetAdmissionReceipt,
}

#[derive(Debug, Serialize)]
struct SourceReleaseBody<'a> {
    expected_source_configuration_sha256: &'a str,
    expected_source_head: &'a TransferGraphHead,
}

/// Create or validate the exact two-request transfer bundle and its target
/// provisioning reservation.  Every network mutation is idempotent on the
/// caller-provided UUID4, and the final response is re-read from both spaces
/// before it crosses the desktop boundary.
pub async fn prepare(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    coordinator: &ProjectTransferCoordinatorState,
    request: ProjectTransferPrepareRequest,
) -> Result<ProjectTransferBundle, String> {
    validate_prepare_request(&request)?;
    let target = resolve_prepare_target(&request, connections, sessions)?;
    let request = coordinator.save_or_validate(&request, &target.expected_space_id)?;
    let status = lifecycle.status()?;
    let pinned = pin_personal_backend(lifecycle, &status).await?;

    let source_value = create_source_request(&pinned, &request, &target.expected_space_id).await?;
    let source =
        parse_transfer_mutation_record(&source_value, &request.source_request_id, "source")?;
    validate_source_for_prepare(&source, &request, &target.expected_space_id)?;

    let configuration = source
        .source_configuration
        .as_ref()
        .ok_or_else(|| "the source transfer omitted its public configuration".to_string())?;
    let source_configuration_sha256 = source
        .source_configuration_sha256
        .as_deref()
        .ok_or_else(|| "the source transfer omitted its configuration commitment".to_string())?;
    let source_release_proof_sha256 = source
        .source_release_proof_sha256
        .as_deref()
        .ok_or_else(|| "the source transfer omitted its release commitment".to_string())?;
    validate_target_machine_intent(&request.target_provisioning, configuration)?;
    let derived_repositories = derive_repository_intents(configuration);
    let incoming_body = IncomingProvisioningCreateBody {
        request_id: &request.target_request_id,
        source_project_id: &request.source_project_id,
        name: &request.target_provisioning.name,
        state_repository: &configuration.state_repository,
        project_truth_scope: &configuration.project_truth_scope,
        default_run_truth_scope: &configuration.default_run_truth_scope,
        default_auto_research_invocation_ceiling: request
            .target_provisioning
            .default_auto_research_invocation_ceiling,
        machines: &request.target_provisioning.machines,
        repositories: derived_repositories,
        provider_checks: &request.target_provisioning.provider_checks,
    };
    let incoming_value = sessions
        .create_incoming_transfer_provisioning(
            connections,
            &target.connection_id,
            &serde_json::to_value(incoming_body)
                .map_err(|_| "could not encode target provisioning intent".to_string())?,
        )
        .await?;
    let incoming = parse_project_provisioning_projection(
        &incoming_value,
        &request.target_request_id,
        &target.expected_space_id,
    )?;
    if incoming.kind != "incoming_transfer"
        || incoming.proposed_project_id != request.source_project_id
    {
        return Err("the target provisioning response does not match the source project".into());
    }
    let accepted_schema_generation = configuration.source_schema_generation;
    let accepted_archive_codec = choose_archive_codec(configuration)?;
    let target_body = TargetTransferCreateBody {
        provisioning_request_id: &request.target_request_id,
        source_request_id: &request.source_request_id,
        source_project_id: &request.source_project_id,
        source_space_id: &source.source_space_id,
        source_configuration: configuration,
        source_configuration_sha256,
        source_release_proof_sha256,
        accepted_schema_generation,
        accepted_archive_codec,
    };
    let target_value = sessions
        .create_target_project_transfer(
            connections,
            &target.connection_id,
            &serde_json::to_value(target_body)
                .map_err(|_| "could not encode target transfer intent".to_string())?,
        )
        .await?;
    let target_record =
        parse_transfer_mutation_record(&target_value, &request.target_request_id, "target")?;
    validate_target_for_prepare(&target_record, &source, &incoming, &request)?;
    let receipt = parse_link_receipt(
        target_value
            .get("link_receipt")
            .ok_or_else(|| "the target transfer omitted its link receipt".to_string())?,
    )?;
    validate_link_receipt(&receipt, &source, &target_record)?;

    post_source_link(&pinned, &request.source_request_id, &receipt).await?;
    // The mutation response is not the lifecycle contract. Re-read all three
    // safe projections after the source link and only then clear local
    // recovery state.
    let linked_source_value =
        read_personal_transfer_value(&pinned, &request.source_request_id).await?;
    let linked_source =
        parse_transfer_record(&linked_source_value, &request.source_request_id, "source")?;
    let linked_target_value = sessions
        .read_project_transfer_value(
            connections,
            &target.connection_id,
            &request.target_request_id,
        )
        .await?;
    let linked_target =
        parse_transfer_record(&linked_target_value, &request.target_request_id, "target")?;
    let linked_incoming_value = sessions
        .read_incoming_transfer_provisioning_value(
            connections,
            &target.connection_id,
            &request.target_request_id,
        )
        .await?;
    let linked_incoming = parse_project_provisioning_projection(
        &linked_incoming_value,
        &request.target_request_id,
        &target.expected_space_id,
    )?;
    let bundle = assemble_bundle(
        &linked_source,
        &linked_target,
        &linked_incoming,
        read_target_provider_setup(connections, sessions, &target.connection_id).await?,
        target.operator_route.is_some(),
    )?;
    validate_bundle(&bundle)?;
    coordinator.remove(&request, &target.expected_space_id)?;
    Ok(bundle)
}

/// Read and validate the same source/target/provisioning bundle after an
/// interrupted wizard or native prepare call.
pub async fn load_bundle(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    coordinator: &ProjectTransferCoordinatorState,
    source_request_id: &str,
) -> Result<ProjectTransferBundle, String> {
    validate_uuid4(source_request_id, "source transfer request identity")?;
    let _ = ensure_transfer_request_session(
        lifecycle,
        connections,
        sessions,
        tunnels,
        coordinator,
        source_request_id,
    )
    .await?;
    if let Some(record) = coordinator.load(source_request_id)? {
        // A crash before source link leaves only the native coordinator record
        // and perhaps an idempotently-created source request. Re-enter the
        // exact saved intent; never ask the browser to reconstruct it.
        let request = record.as_request();
        return prepare(lifecycle, connections, sessions, coordinator, request).await;
    }
    let (_pinned, source_value) = load_source_record(lifecycle, source_request_id).await?;
    let source = parse_transfer_record(&source_value, source_request_id, "source")?;
    let target_request_id = source
        .linked_request_id
        .as_deref()
        .ok_or_else(|| "the source transfer has no target request link".to_string())?;
    let target =
        resolve_target_connection_for_space(&source.target_space_id, connections, sessions)?;
    let target_value = sessions
        .read_project_transfer_value(connections, &target.connection_id, target_request_id)
        .await?;
    let target_record = parse_transfer_record(&target_value, target_request_id, "target")?;
    let incoming_value = sessions
        .read_incoming_transfer_provisioning_value(
            connections,
            &target.connection_id,
            target_request_id,
        )
        .await?;
    let incoming = parse_project_provisioning_projection(
        &incoming_value,
        target_request_id,
        &source.target_space_id,
    )?;
    let bundle = assemble_bundle(
        &source,
        &target_record,
        &incoming,
        read_target_provider_setup(connections, sessions, &target.connection_id).await?,
        target.operator_route.is_some(),
    )?;
    validate_bundle(&bundle)?;
    Ok(bundle)
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectTransferAdvanceResult {
    pub bundle: ProjectTransferBundle,
    pub relay: Option<ProjectTransferRunResult>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectTransferAdvanceAction {
    AdmitTarget,
    AcceptAdmission,
    ReleaseSource,
    AcceptRelease,
    RestoreReentry,
    Relay,
    CompletedRetry,
}

/// Drive the human-confirmation and native relay protocol from the durable
/// source request identity. Web supplies no receipt, proof, release boundary,
/// or target configuration. Each iteration rereads the three authoritative
/// projections, then chooses only an action explicitly published by a
/// backend `can_*` decision.
#[allow(clippy::too_many_arguments)]
pub async fn advance(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    coordinator: &ProjectTransferCoordinatorState,
    source_request_id: &str,
    on_event: &Channel<Value>,
    ssh_program: PathBuf,
) -> Result<ProjectTransferAdvanceResult, String> {
    validate_uuid4(source_request_id, "source transfer request identity")?;

    // Resolve the target before prepare/load so the tunnel is live for every
    // authenticated team request. A transfer cannot enter the final-review
    // protocol without the saved native relay route; manual relay still uses
    // that same configured route interactively.
    let target = ensure_transfer_request_session(
        lifecycle,
        connections,
        sessions,
        tunnels,
        coordinator,
        source_request_id,
    )
    .await?;
    server_commands::configured_route(&target)?;

    // This repairs an interrupted prepare using the exact native coordinator
    // record before any final-review mutation is considered.
    let _ = load_bundle(
        lifecycle,
        connections,
        sessions,
        tunnels,
        coordinator,
        source_request_id,
    )
    .await?;
    let status = lifecycle.status()?;
    let pinned = pin_personal_backend(lifecycle, &status).await?;
    let mut state =
        read_advance_state(&pinned, connections, sessions, &target, source_request_id).await?;

    for _ in 0..MAX_ADVANCE_TRANSITIONS {
        let Some(action) = next_advance_action(&state.source, &state.target, true) else {
            return Ok(ProjectTransferAdvanceResult {
                bundle: state.bundle,
                relay: None,
            });
        };

        match action {
            ProjectTransferAdvanceAction::AdmitTarget => {
                let value = sessions
                    .admit_target_project_transfer(
                        connections,
                        &target.connection_id,
                        &state.target.request_id,
                    )
                    .await?;
                let mutation =
                    parse_transfer_mutation_record(&value, &state.target.request_id, "target")?;
                let receipt = mutation.target_admission_receipt.as_ref().ok_or_else(|| {
                    "target admission returned no durable admission receipt".to_string()
                })?;
                validate_target_admission_matches(receipt, &state.source, &mutation)?;
            }
            ProjectTransferAdvanceAction::AcceptAdmission => {
                let receipt = state
                    .target
                    .target_admission_receipt
                    .as_ref()
                    .ok_or_else(|| {
                        "target admission decision has no durable target receipt".to_string()
                    })?;
                validate_target_admission_matches(receipt, &state.source, &state.target)?;
                let value =
                    post_target_admission(&pinned, &state.source.request_id, receipt).await?;
                let mutation =
                    parse_transfer_mutation_record(&value, &state.source.request_id, "source")?;
                let accepted = mutation.target_admission_receipt.as_ref().ok_or_else(|| {
                    "target admission acceptance returned no durable source receipt".to_string()
                })?;
                validate_target_admission_matches(accepted, &mutation, &state.target)?;
            }
            ProjectTransferAdvanceAction::ReleaseSource => {
                let boundary =
                    read_source_release_boundary(&pinned, &state.source.request_id).await?;
                if boundary.source_configuration_sha256
                    != state
                        .source
                        .source_configuration_sha256
                        .as_deref()
                        .unwrap_or_default()
                {
                    return Err("source release boundary changed its reviewed configuration".into());
                }
                let value =
                    post_source_release(&pinned, &state.source.request_id, &boundary).await?;
                let mutation =
                    parse_transfer_mutation_record(&value, &state.source.request_id, "source")?;
                let receipt = mutation.source_release_receipt.as_ref().ok_or_else(|| {
                    "source release returned no durable release receipt".to_string()
                })?;
                validate_source_release_matches(receipt, &mutation, &state.target)?;
            }
            ProjectTransferAdvanceAction::AcceptRelease => {
                let receipt = state
                    .source
                    .source_release_receipt
                    .as_ref()
                    .ok_or_else(|| {
                        "source release decision has no durable source receipt".to_string()
                    })?;
                validate_source_release_matches(receipt, &state.source, &state.target)?;
                let value = sessions
                    .accept_source_project_transfer_release(
                        connections,
                        &target.connection_id,
                        &state.target.request_id,
                        receipt,
                    )
                    .await?;
                let mutation =
                    parse_transfer_mutation_record(&value, &state.target.request_id, "target")?;
                let accepted = mutation.source_release_receipt.as_ref().ok_or_else(|| {
                    "source release acceptance returned no durable target receipt".to_string()
                })?;
                validate_source_release_matches(accepted, &state.source, &mutation)?;
            }
            ProjectTransferAdvanceAction::RestoreReentry => {
                let digest = state
                    .bundle
                    .incoming_provisioning
                    .final_review_digest
                    .as_deref()
                    .ok_or_else(|| {
                        "restored target transfer has no final-review digest to re-enter"
                            .to_string()
                    })?;
                let value = sessions
                    .restore_target_project_transfer_reentry(
                        connections,
                        &target.connection_id,
                        &state.target.request_id,
                        state.target.revision,
                        digest,
                    )
                    .await?;
                let _ = parse_transfer_mutation_record(&value, &state.target.request_id, "target")?;
            }
            ProjectTransferAdvanceAction::Relay | ProjectTransferAdvanceAction::CompletedRetry => {
                let relay = run(
                    lifecycle,
                    connections,
                    sessions,
                    tunnels,
                    source_request_id,
                    on_event,
                    ssh_program,
                )
                .await?;
                let final_state =
                    read_advance_state(&pinned, connections, sessions, &target, source_request_id)
                        .await?;
                return Ok(ProjectTransferAdvanceResult {
                    bundle: final_state.bundle,
                    relay: Some(relay),
                });
            }
        }
        state =
            read_advance_state(&pinned, connections, sessions, &target, source_request_id).await?;
    }
    Err("the project transfer advance exceeded its bounded transition count".into())
}

struct ProjectTransferAdvanceState {
    source: ProjectTransferRecord,
    target: ProjectTransferRecord,
    bundle: ProjectTransferBundle,
}

async fn read_advance_state(
    pinned: &PinnedPersonalBackend,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    target: &TeamConnectionMetadata,
    source_request_id: &str,
) -> Result<ProjectTransferAdvanceState, String> {
    let source_value = read_personal_transfer_value(pinned, source_request_id).await?;
    let source = parse_transfer_record(&source_value, source_request_id, "source")?;
    let target_request_id = source
        .linked_request_id
        .as_deref()
        .ok_or_else(|| "the source transfer has no target request link".to_string())?;
    if target.expected_space_id != source.target_space_id {
        return Err("the saved target connection does not match the source transfer space".into());
    }
    let target_value = sessions
        .read_project_transfer_value(connections, &target.connection_id, target_request_id)
        .await?;
    let target_record = parse_transfer_record(&target_value, target_request_id, "target")?;
    let incoming_value = sessions
        .read_incoming_transfer_provisioning_value(
            connections,
            &target.connection_id,
            target_request_id,
        )
        .await?;
    let incoming = parse_project_provisioning_projection(
        &incoming_value,
        target_request_id,
        &source.target_space_id,
    )?;
    let bundle = assemble_bundle(
        &source,
        &target_record,
        &incoming,
        read_target_provider_setup(connections, sessions, &target.connection_id).await?,
        target.operator_route.is_some(),
    )?;
    validate_bundle(&bundle)?;
    Ok(ProjectTransferAdvanceState {
        source,
        target: target_record,
        bundle,
    })
}

fn next_advance_action(
    source: &ProjectTransferRecord,
    target: &ProjectTransferRecord,
    operator_route_available: bool,
) -> Option<ProjectTransferAdvanceAction> {
    if target.can_admit {
        Some(ProjectTransferAdvanceAction::AdmitTarget)
    } else if source.can_accept_admission && target.target_admission_receipt.is_some() {
        Some(ProjectTransferAdvanceAction::AcceptAdmission)
    } else if source.can_release {
        Some(ProjectTransferAdvanceAction::ReleaseSource)
    } else if target.can_accept_release {
        Some(ProjectTransferAdvanceAction::AcceptRelease)
    } else if target.can_restore_reentry {
        Some(ProjectTransferAdvanceAction::RestoreReentry)
    } else if operator_route_available && (source.can_relay || target.can_relay) {
        Some(ProjectTransferAdvanceAction::Relay)
    } else if operator_route_available && source.finished && target.finished {
        Some(ProjectTransferAdvanceAction::CompletedRetry)
    } else {
        None
    }
}

fn validate_target_admission_matches(
    receipt: &ProjectTransferTargetAdmissionReceipt,
    source: &ProjectTransferRecord,
    target: &ProjectTransferRecord,
) -> Result<(), String> {
    let source_configuration_sha256 = source
        .source_configuration_sha256
        .as_deref()
        .ok_or_else(|| "source transfer has no configuration commitment".to_string())?;
    let accepted_schema_generation = target
        .accepted_schema_generation
        .ok_or_else(|| "target transfer has no accepted schema generation".to_string())?;
    let accepted_archive_codec = target
        .accepted_archive_codec
        .as_deref()
        .ok_or_else(|| "target transfer has no accepted archive codec".to_string())?;
    let target_activation_proof_sha256 = target
        .target_activation_proof_sha256
        .as_deref()
        .ok_or_else(|| "target transfer has no activation commitment".to_string())?;
    if receipt.source_request_id != source.request_id
        || receipt.target_request_id != target.request_id
        || receipt.project_id != source.project_id
        || receipt.source_space_id != source.source_space_id
        || receipt.target_space_id != source.target_space_id
        || receipt.source_configuration_sha256 != source_configuration_sha256
        || receipt.accepted_schema_generation != accepted_schema_generation
        || receipt.accepted_archive_codec != accepted_archive_codec
        || receipt.source_release_proof_sha256
            != source
                .source_release_proof_sha256
                .as_deref()
                .unwrap_or_default()
        || receipt.target_activation_proof_sha256 != target_activation_proof_sha256
    {
        return Err("target admission receipt does not match the transfer boundary".into());
    }
    Ok(())
}

fn validate_source_release_matches(
    receipt: &ProjectTransferSourceReleaseReceipt,
    source: &ProjectTransferRecord,
    target: &ProjectTransferRecord,
) -> Result<(), String> {
    let source_configuration_sha256 = source
        .source_configuration_sha256
        .as_deref()
        .ok_or_else(|| "source transfer has no configuration commitment".to_string())?;
    let accepted_schema_generation = target
        .accepted_schema_generation
        .ok_or_else(|| "target transfer has no accepted schema generation".to_string())?;
    let accepted_archive_codec = target
        .accepted_archive_codec
        .as_deref()
        .ok_or_else(|| "target transfer has no accepted archive codec".to_string())?;
    let target_activation_proof_sha256 = target
        .target_activation_proof_sha256
        .as_deref()
        .ok_or_else(|| "target transfer has no activation commitment".to_string())?;
    if receipt.source_request_id != source.request_id
        || receipt.target_request_id != target.request_id
        || receipt.project_id != source.project_id
        || receipt.source_space_id != source.source_space_id
        || receipt.target_space_id != source.target_space_id
        || receipt.source_configuration_sha256 != source_configuration_sha256
        || receipt.accepted_schema_generation != accepted_schema_generation
        || receipt.accepted_archive_codec != accepted_archive_codec
        || receipt.source_release_proof_sha256
            != source
                .source_release_proof_sha256
                .as_deref()
                .unwrap_or_default()
        || receipt.target_activation_proof_sha256 != target_activation_proof_sha256
    {
        return Err("source release receipt does not match the transfer boundary".into());
    }
    if let Some(admission) = target.target_admission_receipt.as_ref() {
        if receipt.target_preparation_revision != admission.target_preparation_revision
            || receipt.target_preparation_sha256 != admission.target_preparation_sha256
        {
            return Err("source release receipt does not match target admission".into());
        }
    }
    Ok(())
}

/// Run the fixed target-side provisioning command for the already prepared
/// bundle, then read its lifecycle from the incoming-transfer endpoint.
#[allow(clippy::too_many_arguments)]
pub async fn run_incoming_provision(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    coordinator: &ProjectTransferCoordinatorState,
    source_request_id: &str,
    on_event: &Channel<Value>,
    ssh_program: PathBuf,
) -> Result<ServerCommandRunResult, String> {
    let bundle = load_bundle(
        lifecycle,
        connections,
        sessions,
        tunnels,
        coordinator,
        source_request_id,
    )
    .await?;
    let target =
        resolve_target_connection_for_space(&bundle.target.target_space_id, connections, sessions)?;
    server_commands::configured_route(&target)?;
    let (exit_code, event_count) = server_commands::run_project_provision(
        &target,
        &bundle.target.request_id,
        on_event,
        ssh_program,
    )
    .await?;
    let incoming_value = sessions
        .read_incoming_transfer_provisioning_value(
            connections,
            &target.connection_id,
            &bundle.target.request_id,
        )
        .await?;
    let incoming = parse_project_provisioning_projection(
        &incoming_value,
        &bundle.target.request_id,
        &bundle.target.target_space_id,
    )?;
    Ok(ServerCommandRunResult {
        connection_id: target.connection_id,
        request_id: bundle.target.request_id,
        exit_code,
        event_count,
        readback: ProjectProvisionReadback {
            request_id: incoming.request_id,
            target_space_id: incoming.target_space_id,
            status: incoming.status,
            revision: incoming.revision,
        },
    })
}

pub async fn read_target_provider_setup(
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    connection_id: &str,
) -> Result<Vec<TargetProviderSetupProjection>, String> {
    validate_uuid4(connection_id, "team connection identity")?;
    let _ = sessions.established(connection_id)?;
    let value = sessions
        .read_target_provider_setup_value(connections, connection_id)
        .await?;
    parse_provider_setup_projection(&value)
}

fn validate_prepare_request(request: &ProjectTransferPrepareRequest) -> Result<(), String> {
    validate_uuid4(
        &request.source_request_id,
        "source transfer request identity",
    )?;
    validate_uuid4(
        &request.target_request_id,
        "target transfer request identity",
    )?;
    if request.source_request_id == request.target_request_id {
        return Err("source and target transfer request identities must be distinct".into());
    }
    validate_uuid4(&request.connection_id, "team connection identity")?;
    validate_uuid4(
        &request.source_project_id,
        "source transfer project identity",
    )?;
    let intent = &request.target_provisioning;
    validate_safe_line(&intent.name, "target provisioning project name", 120, true)?;
    if intent.default_auto_research_invocation_ceiling == 0 {
        return Err("target provisioning invocation ceiling must be positive".into());
    }
    if intent.machines.is_empty() || intent.machines.len() > 32 {
        return Err("target provisioning must contain a bounded machine list".into());
    }
    if intent.provider_checks.is_empty() || intent.provider_checks.len() > 32 {
        return Err("target provisioning must contain a bounded provider list".into());
    }
    let mut machine_aliases = Vec::with_capacity(intent.machines.len());
    for machine in &intent.machines {
        validate_alias(&machine.alias, "target provisioning machine alias")?;
        if !matches!(machine.location.as_str(), "local" | "ssh") {
            return Err("target provisioning machine location is invalid".into());
        }
        validate_safe_line(
            &machine.host,
            "target provisioning machine host",
            255,
            false,
        )?;
        validate_safe_line(
            &machine.os_account,
            "target provisioning machine operating-system account",
            128,
            true,
        )?;
        if let Some(root) = &machine.central_root {
            validate_absolute_path(root, "target provisioning central root")?;
        }
        if machine_aliases.contains(&machine.alias) {
            return Err("target provisioning machine aliases must be unique".into());
        }
        machine_aliases.push(machine.alias.clone());
    }
    let mut profiles = Vec::with_capacity(intent.provider_checks.len());
    for provider in &intent.provider_checks {
        validate_safe_line(
            &provider.profile,
            "target provisioning provider profile",
            120,
            true,
        )?;
        validate_safe_line(
            &provider.provider,
            "target provisioning provider",
            120,
            true,
        )?;
        validate_safe_line(
            &provider.runtime_id,
            "target provisioning provider runtime",
            120,
            true,
        )?;
        validate_safe_line(
            &provider.model,
            "target provisioning provider model",
            200,
            false,
        )?;
        validate_safe_line(
            &provider.reasoning,
            "target provisioning provider reasoning",
            80,
            true,
        )?;
        validate_alias(
            &provider.machine_alias,
            "target provisioning provider machine alias",
        )?;
        if !machine_aliases
            .iter()
            .any(|alias| alias == &provider.machine_alias)
        {
            return Err("target provisioning provider names an unknown machine".into());
        }
        if profiles.contains(&provider.profile) {
            return Err("target provisioning provider profiles must be unique".into());
        }
        profiles.push(provider.profile.clone());
    }
    Ok(())
}

fn validate_source_for_prepare(
    source: &ProjectTransferRecord,
    request: &ProjectTransferPrepareRequest,
    target_space_id: &str,
) -> Result<(), String> {
    if source.project_id != request.source_project_id
        || source.target_space_id != target_space_id
        || source.source_space_id == source.target_space_id
    {
        return Err(
            "the source transfer does not match the selected project and target space".into(),
        );
    }
    if let Some(linked_request_id) = &source.linked_request_id {
        if linked_request_id != &request.target_request_id {
            return Err("the source transfer is already linked to another target request".into());
        }
    }
    let configuration = source
        .source_configuration
        .as_ref()
        .ok_or_else(|| "the source transfer omitted its public configuration".to_string())?;
    validate_source_configuration(configuration)?;
    let configuration_digest = source
        .source_configuration_sha256
        .as_deref()
        .ok_or_else(|| "the source transfer omitted its configuration commitment".to_string())?;
    validate_digest(configuration_digest, "source configuration commitment")?;
    let release_digest = source
        .source_release_proof_sha256
        .as_deref()
        .ok_or_else(|| "the source transfer omitted its release commitment".to_string())?;
    validate_digest(release_digest, "source release proof commitment")?;
    Ok(())
}

fn resolve_prepare_target(
    request: &ProjectTransferPrepareRequest,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
) -> Result<TeamConnectionMetadata, String> {
    let target = resolve_prepare_target_metadata(request, connections)?;
    sessions.established(&target.connection_id)?;
    Ok(target)
}

fn resolve_prepare_target_metadata(
    request: &ProjectTransferPrepareRequest,
    connections: &TeamConnectionState,
) -> Result<TeamConnectionMetadata, String> {
    let matches = connections
        .list()?
        .into_iter()
        .filter(|connection| connection.connection_id == request.connection_id)
        .collect::<Vec<_>>();
    let [target] = matches.as_slice() else {
        return Err("the selected target connection is not saved on this desktop".into());
    };
    Ok(target.clone())
}

fn resolve_target_connection_for_space(
    target_space_id: &str,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
) -> Result<TeamConnectionMetadata, String> {
    let target = resolve_target_connection_metadata_for_space(target_space_id, connections)?;
    sessions.established(&target.connection_id)?;
    Ok(target)
}

fn resolve_target_connection_metadata_for_space(
    target_space_id: &str,
    connections: &TeamConnectionState,
) -> Result<TeamConnectionMetadata, String> {
    validate_uuid4(target_space_id, "target transfer space identity")?;
    let matches = connections
        .list()?
        .into_iter()
        .filter(|connection| connection.expected_space_id == target_space_id)
        .collect::<Vec<_>>();
    let [target] = matches.as_slice() else {
        return Err(
            "the source transfer target space has no unique saved desktop connection".into(),
        );
    };
    Ok(target.clone())
}

async fn ensure_transfer_request_session(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    coordinator: &ProjectTransferCoordinatorState,
    source_request_id: &str,
) -> Result<TeamConnectionMetadata, String> {
    let target = if let Some(record) = coordinator.load(source_request_id)? {
        resolve_prepare_target_metadata(&record.as_request(), connections)?
    } else {
        let status = lifecycle.status()?;
        let pinned = pin_personal_backend(lifecycle, &status).await?;
        let source_value = read_personal_transfer_value(&pinned, source_request_id).await?;
        let source = parse_transfer_record(&source_value, source_request_id, "source")?;
        resolve_target_connection_metadata_for_space(&source.target_space_id, connections)?
    };
    sessions
        .ensure_native_session(connections, tunnels, lifecycle, &target.connection_id)
        .await?;
    Ok(target)
}

async fn create_source_request(
    pinned: &PinnedPersonalBackend,
    request: &ProjectTransferPrepareRequest,
    target_space_id: &str,
) -> Result<Value, String> {
    let client = personal_client(&pinned.base_url)?;
    let body = SourceCreateBody {
        request_id: &request.source_request_id,
        project_id: &request.source_project_id,
        target_space_id,
    };
    let response = client
        .post(format!("{}{SOURCE_CREATE_PATH}", pinned.base_url))
        .header(
            "X-RCP-Instance-ID",
            HeaderValue::from_str(&pinned.instance_id)
                .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
        )
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("could not create the personal transfer request: {error}"))?;
    personal_json_response(response, "personal transfer request creation").await
}

async fn post_source_link(
    pinned: &PinnedPersonalBackend,
    source_request_id: &str,
    receipt: &ProjectTransferLinkReceipt,
) -> Result<Value, String> {
    validate_uuid4(source_request_id, "source transfer request identity")?;
    let client = personal_client(&pinned.base_url)?;
    let body = SourceLinkBody { receipt };
    let response = client
        .post(format!(
            "{}{SOURCE_LINK_PATH_PREFIX}{source_request_id}/link",
            pinned.base_url
        ))
        .header(
            "X-RCP-Instance-ID",
            HeaderValue::from_str(&pinned.instance_id)
                .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
        )
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("could not link the personal transfer request: {error}"))?;
    personal_json_response(response, "personal transfer link").await
}

async fn post_target_admission(
    pinned: &PinnedPersonalBackend,
    source_request_id: &str,
    receipt: &ProjectTransferTargetAdmissionReceipt,
) -> Result<Value, String> {
    validate_uuid4(source_request_id, "source transfer request identity")?;
    validate_target_admission_receipt(receipt)?;
    if receipt.source_request_id != source_request_id {
        return Err("target admission receipt does not match the source request".into());
    }
    let client = personal_client(&pinned.base_url)?;
    let body = TargetAdmissionBody { receipt };
    let response = client
        .post(format!(
            "{}{SOURCE_LINK_PATH_PREFIX}{source_request_id}/target-admission",
            pinned.base_url
        ))
        .header(
            "X-RCP-Instance-ID",
            HeaderValue::from_str(&pinned.instance_id)
                .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
        )
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("could not accept the target transfer admission: {error}"))?;
    personal_json_response(response, "target transfer admission acceptance").await
}

async fn read_source_release_boundary(
    pinned: &PinnedPersonalBackend,
    source_request_id: &str,
) -> Result<ProjectTransferSourceBoundaryResponse, String> {
    validate_uuid4(source_request_id, "source transfer request identity")?;
    let client = personal_client(&pinned.base_url)?;
    let response = client
        .get(format!(
            "{}{SOURCE_LINK_PATH_PREFIX}{source_request_id}/release-boundary",
            pinned.base_url
        ))
        .header(
            "X-RCP-Instance-ID",
            HeaderValue::from_str(&pinned.instance_id)
                .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
        )
        .send()
        .await
        .map_err(|error| format!("could not read the source release boundary: {error}"))?;
    let value = personal_json_response(response, "source release boundary readback").await?;
    let boundary = serde_json::from_value::<ProjectTransferSourceBoundaryResponse>(value)
        .map_err(|_| "the source release boundary returned an invalid response".to_string())?;
    validate_source_configuration(&boundary.source_configuration)?;
    validate_digest(
        &boundary.source_configuration_sha256,
        "source configuration commitment",
    )?;
    validate_transfer_graph_head(&boundary.source_head, true, "source release boundary head")?;
    Ok(boundary)
}

async fn post_source_release(
    pinned: &PinnedPersonalBackend,
    source_request_id: &str,
    boundary: &ProjectTransferSourceBoundaryResponse,
) -> Result<Value, String> {
    validate_uuid4(source_request_id, "source transfer request identity")?;
    let client = personal_client(&pinned.base_url)?;
    let body = SourceReleaseBody {
        expected_source_configuration_sha256: &boundary.source_configuration_sha256,
        expected_source_head: &boundary.source_head,
    };
    let response = client
        .post(format!(
            "{}{SOURCE_LINK_PATH_PREFIX}{source_request_id}/release",
            pinned.base_url
        ))
        .header(
            "X-RCP-Instance-ID",
            HeaderValue::from_str(&pinned.instance_id)
                .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
        )
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("could not release the source transfer request: {error}"))?;
    personal_json_response(response, "source transfer release").await
}

async fn load_source_record(
    lifecycle: &BackendState,
    request_id: &str,
) -> Result<(PinnedPersonalBackend, Value), String> {
    validate_uuid4(request_id, "source transfer request identity")?;
    let status = lifecycle.status()?;
    let pinned = pin_personal_backend(lifecycle, &status).await?;
    let value = read_personal_transfer_value(&pinned, request_id).await?;
    Ok((pinned, value))
}

async fn read_personal_transfer_value(
    pinned: &PinnedPersonalBackend,
    request_id: &str,
) -> Result<Value, String> {
    validate_uuid4(request_id, "source transfer request identity")?;
    let client = personal_client(&pinned.base_url)?;
    let response = client
        .get(format!(
            "{}{}",
            pinned.base_url,
            source_request_path(request_id)
        ))
        .header(
            "X-RCP-Instance-ID",
            HeaderValue::from_str(&pinned.instance_id)
                .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
        )
        .send()
        .await
        .map_err(|error| format!("could not read the personal transfer request: {error}"))?;
    personal_json_response(response, "personal transfer request readback").await
}

async fn personal_json_response(response: Response, description: &str) -> Result<Value, String> {
    if !response.status().is_success() {
        return Err(format!(
            "{description} was rejected (HTTP {})",
            response.status().as_u16()
        ));
    }
    response
        .json::<Value>()
        .await
        .map_err(|_| format!("{description} returned an invalid response"))
}

fn assemble_bundle(
    source: &ProjectTransferRecord,
    target: &ProjectTransferRecord,
    incoming: &ProjectProvisioningProjection,
    target_provider_setup: Vec<TargetProviderSetupProjection>,
    operator_route_available: bool,
) -> Result<ProjectTransferBundle, String> {
    let route_capable = operator_route_available;
    let (can_advance, advance_label, finished) =
        aggregate_transfer_decisions(source, target, route_capable);
    Ok(ProjectTransferBundle {
        source: source.to_projection()?,
        target: target.to_projection()?,
        incoming_provisioning: incoming.clone(),
        target_provider_setup,
        can_advance,
        advance_label,
        can_manual_relay: route_capable && (source.can_relay || target.can_relay),
        finished,
    })
}

/// Select the one native action exposed by the two backend projections.  The
/// ordering mirrors the cross-space protocol, while every gate is a backend
/// `can_*` answer rather than a client-side phase/status interpretation.
fn aggregate_transfer_decisions(
    source: &ProjectTransferRecord,
    target: &ProjectTransferRecord,
    operator_route_available: bool,
) -> (bool, Option<String>, bool) {
    let actions = [
        (
            target.can_admit,
            Some("Confirm target admission and continue"),
        ),
        (
            source.can_accept_admission && target.target_admission_receipt.is_some(),
            Some("Record target admission and continue"),
        ),
        (
            source.can_release,
            Some("Release the personal project and continue"),
        ),
        (
            target.can_accept_release,
            Some("Record source release and continue"),
        ),
        (
            target.can_restore_reentry,
            Some("Re-enter the restored transfer and continue"),
        ),
        (
            operator_route_available && (source.can_relay || target.can_relay),
            Some("Relay the sealed project archive and continue"),
        ),
    ];
    let advance_label = actions
        .iter()
        .find_map(|(enabled, label)| enabled.then(|| label.map(str::to_string)).flatten());
    let can_advance = actions.iter().any(|(enabled, _)| *enabled);
    (
        can_advance,
        advance_label,
        source.finished && target.finished,
    )
}

impl ProjectTransferRecord {
    fn to_projection(&self) -> Result<ProjectTransferProjection, String> {
        Ok(ProjectTransferProjection {
            request_id: self.request_id.clone(),
            side: self.side.clone(),
            phase: self.phase.clone(),
            phase_label: self.phase_label.clone(),
            next_action: self.next_action.clone(),
            linked_request_id: self.linked_request_id.clone(),
            project_id: self.project_id.clone(),
            source_space_id: self.source_space_id.clone(),
            target_space_id: self.target_space_id.clone(),
            source_configuration: self
                .source_configuration
                .clone()
                .ok_or_else(|| "the transfer omitted its public configuration".to_string())?,
            source_configuration_sha256: self
                .source_configuration_sha256
                .clone()
                .ok_or_else(|| "the transfer omitted its configuration commitment".to_string())?,
            accepted_schema_generation: self.accepted_schema_generation,
            accepted_archive_codec: self.accepted_archive_codec.clone(),
            source_release_proof_sha256: self
                .source_release_proof_sha256
                .clone()
                .ok_or_else(|| "the transfer omitted its release commitment".to_string())?,
            target_activation_proof_sha256: self.target_activation_proof_sha256.clone(),
            archive_sha256: self.archive_sha256.clone(),
            archive_size_bytes: self.archive_size_bytes,
            can_link: self.can_link,
            can_run_setup: self.can_run_setup,
            can_review: self.can_review,
            can_admit: self.can_admit,
            can_accept_admission: self.can_accept_admission,
            can_release: self.can_release,
            can_accept_release: self.can_accept_release,
            can_relay: self.can_relay,
            can_restore_reentry: self.can_restore_reentry,
            can_complete: self.can_complete,
            finished: self.finished,
            revision: self.revision,
        })
    }
}

fn parse_transfer_record(
    value: &Value,
    expected_request_id: &str,
    expected_side: &str,
) -> Result<ProjectTransferRecord, String> {
    parse_transfer_record_inner(value, expected_request_id, expected_side, true)
}

/// Parse a mutation response that intentionally uses the durable record shape
/// rather than the safe lifecycle projection. Every externally visible read
/// still goes through `parse_transfer_record` above.
fn parse_transfer_mutation_record(
    value: &Value,
    expected_request_id: &str,
    expected_side: &str,
) -> Result<ProjectTransferRecord, String> {
    parse_transfer_record_inner(value, expected_request_id, expected_side, false)
}

fn parse_transfer_record_inner(
    value: &Value,
    expected_request_id: &str,
    expected_side: &str,
    require_projection: bool,
) -> Result<ProjectTransferRecord, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "the project transfer returned an invalid response".to_string())?;
    let text = |field: &str| {
        object
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("the project transfer has no valid {field}"))
    };
    let request_id = text("request_id")?.to_string();
    if request_id != expected_request_id {
        return Err("the project transfer identity does not match the requested id".into());
    }
    validate_uuid4(&request_id, "project transfer request identity")?;
    if text("side")? != expected_side {
        return Err(format!(
            "the project transfer is not a {expected_side} request"
        ));
    }
    let phase = text("phase")?;
    if !matches!(
        phase,
        "awaiting_link"
            | "linked"
            | "target_admitted"
            | "source_released"
            | "source_fenced"
            | "archive_bound"
            | "target_activated"
            | "cleanup_acknowledged"
            | "completed"
            | "operator_action_needed"
    ) {
        return Err("the project transfer has an invalid durable phase".into());
    }
    let project_id = text("project_id")?.to_string();
    let source_space_id = text("source_space_id")?.to_string();
    let target_space_id = text("target_space_id")?.to_string();
    validate_uuid4(&project_id, "transfer project identity")?;
    validate_uuid4(&source_space_id, "source transfer space identity")?;
    validate_uuid4(&target_space_id, "target transfer space identity")?;
    if source_space_id == target_space_id {
        return Err("the project transfer must cross spaces".into());
    }
    let linked_request_id = optional_text(object, "linked_request_id")?;
    if let Some(linked_request_id) = &linked_request_id {
        validate_uuid4(linked_request_id, "linked transfer request identity")?;
    }
    let source_configuration = object
        .get("source_configuration")
        .map(|configuration| {
            serde_json::from_value::<ProjectTransferSourceConfiguration>(configuration.clone())
                .map_err(|_| "the project transfer has an invalid source configuration".to_string())
        })
        .transpose()?;
    if let Some(configuration) = &source_configuration {
        validate_source_configuration(configuration)?;
    }
    let source_configuration_sha256 = optional_digest(
        object,
        "source_configuration_sha256",
        "source configuration commitment",
    )?;
    let target_admission_receipt = object
        .get("target_admission_receipt")
        .filter(|value| !value.is_null())
        .map(parse_target_admission_receipt)
        .transpose()?;
    let source_release_receipt = object
        .get("source_release_receipt")
        .filter(|value| !value.is_null())
        .map(parse_source_release_receipt)
        .transpose()?;
    let source_release_proof_sha256 = optional_digest(
        object,
        "source_release_proof_sha256",
        "source release proof commitment",
    )?;
    let target_activation_proof_sha256 = optional_digest(
        object,
        "target_activation_proof_sha256",
        "target activation proof commitment",
    )?;
    let archive_sha256 = optional_digest(object, "archive_sha256", "transfer archive digest")?;
    let archive_size_bytes = optional_u64(object, "archive_size_bytes")?;
    if archive_size_bytes == Some(0) {
        return Err("the project transfer archive size must be positive".into());
    }
    let accepted_schema_generation = optional_u64(object, "accepted_schema_generation")?;
    if accepted_schema_generation == Some(0) {
        return Err("the project transfer schema generation must be positive".into());
    }
    let phase_label = if require_projection {
        Some(required_text(object, "phase_label")?.to_string())
    } else {
        optional_text(object, "phase_label")?
    };
    let next_action = if require_projection {
        required_nullable_text(object, "next_action")?
    } else {
        optional_text(object, "next_action")?
    };
    let decision = |field: &str| {
        if require_projection {
            required_bool(object, field)
        } else {
            Ok(optional_bool(object, field)?.unwrap_or(false))
        }
    };
    let revision = if require_projection {
        required_u64(object, "revision")?
    } else {
        optional_u64(object, "revision")?.unwrap_or(0)
    };
    Ok(ProjectTransferRecord {
        request_id,
        side: expected_side.to_string(),
        phase: phase.to_string(),
        phase_label,
        next_action,
        linked_request_id,
        project_id,
        source_space_id,
        target_space_id,
        source_configuration,
        source_configuration_sha256,
        target_admission_receipt,
        source_release_receipt,
        accepted_schema_generation,
        accepted_archive_codec: optional_text(object, "accepted_archive_codec")?,
        source_release_proof_sha256,
        target_activation_proof_sha256,
        archive_sha256,
        archive_size_bytes,
        can_link: decision("can_link")?,
        can_run_setup: decision("can_run_setup")?,
        can_review: decision("can_review")?,
        can_admit: decision("can_admit")?,
        can_accept_admission: decision("can_accept_admission")?,
        can_release: decision("can_release")?,
        can_accept_release: decision("can_accept_release")?,
        can_relay: decision("can_relay")?,
        can_restore_reentry: decision("can_restore_reentry")?,
        can_complete: decision("can_complete")?,
        finished: decision("finished")?,
        revision,
    })
}

fn parse_source_request(
    value: &Value,
    expected_request_id: &str,
    require_archive: bool,
) -> Result<SourceTransferRequest, String> {
    let record = parse_transfer_mutation_record(value, expected_request_id, "source")?;
    if !matches!(
        record.phase.as_str(),
        "archive_bound" | "cleanup_acknowledged" | "completed"
    ) {
        return Err("the personal transfer is not at a relayable phase".into());
    }
    let target_request_id = record
        .linked_request_id
        .ok_or_else(|| "the personal transfer has no linked target request".to_string())?;
    let target_activation_proof_sha256 = record
        .target_activation_proof_sha256
        .ok_or_else(|| "the personal transfer has no target activation commitment".to_string())?;
    let archive_sha256 = record
        .archive_sha256
        .ok_or_else(|| "the personal transfer has no archive digest".to_string())?;
    let archive_size_bytes = record
        .archive_size_bytes
        .ok_or_else(|| "the personal transfer has no bounded archive size".to_string())?;
    if require_archive && record.phase != "archive_bound" {
        return Err("the personal transfer archive is no longer available for local relay".into());
    }
    Ok(SourceTransferRequest {
        request_id: record.request_id,
        phase: record.phase,
        target_request_id,
        project_id: record.project_id,
        source_space_id: record.source_space_id,
        target_space_id: record.target_space_id,
        target_activation_proof_sha256,
        archive_sha256,
        archive_size_bytes,
    })
}

fn optional_text(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<String>, String> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(str::to_string)
            .map(Some)
            .ok_or_else(|| format!("the project transfer has no valid {field}")),
    }
}

fn required_nullable_text(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<String>, String> {
    match object.get(field) {
        None => Err(format!("the project transfer has no {field} decision")),
        Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(str::to_string)
            .map(Some)
            .ok_or_else(|| format!("the project transfer has no valid {field}")),
    }
}

fn optional_bool(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<bool>, String> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| format!("the project transfer has no valid {field}")),
    }
}

fn optional_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, String> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("the project transfer has no valid {field}")),
    }
}

fn optional_digest(
    object: &serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<Option<String>, String> {
    let value = optional_text(object, field)?;
    if let Some(value) = &value {
        validate_digest(value, label)?;
    }
    Ok(value)
}

fn validate_source_configuration(
    configuration: &ProjectTransferSourceConfiguration,
) -> Result<(), String> {
    validate_safe_line(
        &configuration.source_rcp_version,
        "source RCP version",
        120,
        true,
    )?;
    if configuration.source_schema_generation == 0
        || configuration.supported_archive_codecs.is_empty()
        || configuration.machine_aliases.is_empty()
        || configuration.repositories.is_empty()
    {
        return Err("the source transfer configuration is incomplete".into());
    }
    for codec in &configuration.supported_archive_codecs {
        validate_safe_line(codec, "source archive codec", 120, true)?;
    }
    for alias in &configuration.machine_aliases {
        validate_alias(alias, "source machine alias")?;
    }
    let mut aliases = Vec::with_capacity(configuration.repositories.len());
    let mut identities = Vec::with_capacity(configuration.repositories.len());
    for repository in &configuration.repositories {
        validate_alias(&repository.alias, "source repository alias")?;
        validate_alias(&repository.machine_alias, "source repository machine alias")?;
        if !configuration
            .machine_aliases
            .iter()
            .any(|machine| machine == &repository.machine_alias)
        {
            return Err("the source repository names an unknown machine alias".into());
        }
        if aliases.contains(&repository.alias) {
            return Err("source repository aliases must be unique".into());
        }
        validate_repository_identity(&repository.repository.identity)?;
        if identities.contains(&repository.repository.identity) {
            return Err("source repository identities must be unique".into());
        }
        aliases.push(repository.alias.clone());
        identities.push(repository.repository.identity.clone());
    }
    validate_scopes(
        &configuration.state_repository,
        &configuration.project_truth_scope,
        &configuration.default_run_truth_scope,
        &aliases,
    )?;
    validate_digest(
        &configuration.source_manifest_sha256,
        "source manifest commitment",
    )?;
    Ok(())
}

fn validate_scopes(
    state_repository: &str,
    project_truth_scope: &[String],
    default_run_truth_scope: &[String],
    repository_aliases: &[String],
) -> Result<(), String> {
    validate_alias(state_repository, "state repository alias")?;
    if project_truth_scope.is_empty() || default_run_truth_scope.is_empty() {
        return Err("transfer scopes must not be empty".into());
    }
    for alias in project_truth_scope.iter().chain(default_run_truth_scope) {
        validate_alias(alias, "transfer scope repository alias")?;
        if !repository_aliases
            .iter()
            .any(|candidate| candidate == alias)
        {
            return Err("transfer scope names an unknown repository alias".into());
        }
    }
    if !project_truth_scope
        .iter()
        .any(|alias| alias == state_repository)
        || !default_run_truth_scope.iter().all(|alias| {
            project_truth_scope
                .iter()
                .any(|candidate| candidate == alias)
        })
    {
        return Err("transfer scopes do not preserve the state repository subset".into());
    }
    Ok(())
}

fn validate_safe_line(
    value: &str,
    label: &str,
    max_bytes: usize,
    nonempty: bool,
) -> Result<(), String> {
    if value.len() > max_bytes
        || value != value.trim()
        || value.chars().any(|character| {
            let code = character as u32;
            code < 32 || code == 127
        })
        || (nonempty && value.is_empty())
    {
        return Err(format!("{label} is invalid"));
    }
    Ok(())
}

fn validate_alias(value: &str, label: &str) -> Result<(), String> {
    validate_safe_line(value, label, 120, true)?;
    if value.bytes().enumerate().any(|(index, byte)| {
        !(byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || byte == b'.')
            || (index == 0 && !byte.is_ascii_alphanumeric())
    }) {
        return Err(format!("{label} is invalid"));
    }
    Ok(())
}

fn validate_absolute_path(value: &str, label: &str) -> Result<(), String> {
    validate_safe_line(value, label, 4096, true)?;
    if !value.starts_with('/') || value.split('/').any(|component| component == "..") {
        return Err(format!("{label} is invalid"));
    }
    Ok(())
}

fn validate_repository_identity(value: &str) -> Result<(), String> {
    let (owner, repository) = value
        .split_once('/')
        .ok_or_else(|| "GitHub repository identity is invalid".to_string())?;
    if value != value.to_ascii_lowercase()
        || owner.is_empty()
        || repository.is_empty()
        || owner.len() > 39
        || repository.len() > 100
        || owner
            .bytes()
            .any(|byte| !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'))
        || repository.bytes().any(|byte| {
            !(byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || byte == b'-'
                || byte == b'_'
                || byte == b'.')
        })
    {
        return Err("GitHub repository identity is invalid".into());
    }
    Ok(())
}

fn choose_archive_codec(
    configuration: &ProjectTransferSourceConfiguration,
) -> Result<&str, String> {
    configuration
        .supported_archive_codecs
        .first()
        .map(String::as_str)
        .ok_or_else(|| "the source transfer offers no archive codec".to_string())
}

fn validate_target_machine_intent(
    intent: &ProjectTransferTargetProvisioningIntent,
    configuration: &ProjectTransferSourceConfiguration,
) -> Result<(), String> {
    for source_alias in &configuration.machine_aliases {
        if !intent
            .machines
            .iter()
            .any(|machine| &machine.alias == source_alias)
        {
            return Err(format!(
                "target provisioning is missing source machine alias {source_alias}"
            ));
        }
    }
    Ok(())
}

fn derive_repository_intents(
    configuration: &ProjectTransferSourceConfiguration,
) -> Vec<DerivedRepositoryIntent> {
    configuration
        .repositories
        .iter()
        .map(|repository| DerivedRepositoryIntent {
            alias: repository.alias.clone(),
            source: format!("https://github.com/{}.git", repository.repository.identity),
            machine_alias: repository.machine_alias.clone(),
        })
        .collect()
}

fn parse_link_receipt(value: &Value) -> Result<ProjectTransferLinkReceipt, String> {
    let receipt = serde_json::from_value::<ProjectTransferLinkReceipt>(value.clone())
        .map_err(|_| "the target transfer returned an invalid link receipt".to_string())?;
    validate_uuid4(
        &receipt.source_request_id,
        "source transfer request identity",
    )?;
    validate_uuid4(
        &receipt.target_request_id,
        "target transfer request identity",
    )?;
    validate_uuid4(&receipt.project_id, "transfer project identity")?;
    validate_uuid4(&receipt.source_space_id, "source transfer space identity")?;
    validate_uuid4(&receipt.target_space_id, "target transfer space identity")?;
    if receipt.source_space_id == receipt.target_space_id {
        return Err("the transfer link must cross spaces".into());
    }
    validate_digest(
        &receipt.source_configuration_sha256,
        "source configuration commitment",
    )?;
    validate_digest(
        &receipt.source_release_proof_sha256,
        "source release proof commitment",
    )?;
    validate_digest(
        &receipt.target_activation_proof_sha256,
        "target activation proof commitment",
    )?;
    if receipt.target_repositories.is_empty() {
        return Err("the transfer link has no target repositories".into());
    }
    let mut aliases = Vec::with_capacity(receipt.target_repositories.len());
    for repository in &receipt.target_repositories {
        validate_alias(&repository.alias, "linked target repository alias")?;
        validate_repository_identity(&repository.repository.identity)?;
        if aliases
            .last()
            .is_some_and(|previous| previous >= &repository.alias)
        {
            return Err("linked target repositories must be ordered by alias".into());
        }
        aliases.push(repository.alias.clone());
    }
    validate_safe_line(
        &receipt.accepted_archive_codec,
        "accepted archive codec",
        120,
        true,
    )?;
    validate_safe_line(&receipt.created_at, "transfer link timestamp", 120, true)?;
    Ok(receipt)
}

fn parse_target_admission_receipt(
    value: &Value,
) -> Result<ProjectTransferTargetAdmissionReceipt, String> {
    let receipt = serde_json::from_value::<ProjectTransferTargetAdmissionReceipt>(value.clone())
        .map_err(|_| "the target transfer returned an invalid admission receipt".to_string())?;
    validate_target_admission_receipt(&receipt)?;
    Ok(receipt)
}

fn parse_source_release_receipt(
    value: &Value,
) -> Result<ProjectTransferSourceReleaseReceipt, String> {
    let receipt = serde_json::from_value::<ProjectTransferSourceReleaseReceipt>(value.clone())
        .map_err(|_| "the source transfer returned an invalid release receipt".to_string())?;
    validate_source_release_receipt(&receipt)?;
    Ok(receipt)
}

fn validate_receipt_identities(
    source_request_id: &str,
    target_request_id: &str,
    project_id: &str,
    source_space_id: &str,
    target_space_id: &str,
) -> Result<(), String> {
    for (value, label) in [
        (source_request_id, "source transfer request identity"),
        (target_request_id, "target transfer request identity"),
        (project_id, "transfer project identity"),
        (source_space_id, "source transfer space identity"),
        (target_space_id, "target transfer space identity"),
    ] {
        validate_uuid4(value, label)?;
    }
    if source_space_id == target_space_id {
        return Err("the transfer receipt must cross spaces".into());
    }
    Ok(())
}

fn validate_receipt_actor(
    actor: &ProjectProvisioningAuthorizedHuman,
    expected_space_id: &str,
    label: &str,
) -> Result<(), String> {
    validate_uuid4(&actor.space_id, &format!("{label} space identity"))?;
    validate_uuid4(&actor.user_id, &format!("{label} user identity"))?;
    if actor.space_id != expected_space_id {
        return Err(format!("{label} belongs to another transfer space"));
    }
    validate_safe_line(&actor.display_name, label, 200, true)
}

fn validate_transfer_graph_head(
    head: &TransferGraphHead,
    require_main: bool,
    label: &str,
) -> Result<(), String> {
    validate_safe_line(&head.target.kind, &format!("{label} target kind"), 20, true)?;
    if !matches!(head.target.kind.as_str(), "main" | "branch") {
        return Err(format!("{label} target kind is invalid"));
    }
    match (head.target.kind.as_str(), head.target.branch_id.as_deref()) {
        ("main", None) => {}
        ("main", Some(_)) => return Err(format!("{label} main target has a branch id")),
        ("branch", Some(branch_id)) => {
            validate_uuid4(branch_id, &format!("{label} branch identity"))?
        }
        ("branch", None) => return Err(format!("{label} branch target has no branch id")),
        _ => unreachable!(),
    }
    if require_main && head.target.kind != "main" {
        return Err(format!("{label} must bind the main canonical head"));
    }
    if let Some(transition_id) = &head.transition_id {
        validate_digest(transition_id, &format!("{label} transition identity"))?;
    }
    Ok(())
}

fn validate_target_admission_receipt(
    receipt: &ProjectTransferTargetAdmissionReceipt,
) -> Result<(), String> {
    validate_receipt_identities(
        &receipt.source_request_id,
        &receipt.target_request_id,
        &receipt.project_id,
        &receipt.source_space_id,
        &receipt.target_space_id,
    )?;
    validate_receipt_actor(
        &receipt.admitted_by,
        &receipt.target_space_id,
        "admission actor",
    )?;
    for (value, label) in [
        (
            &receipt.source_configuration_sha256,
            "source configuration commitment",
        ),
        (
            &receipt.target_preparation_sha256,
            "target preparation commitment",
        ),
        (
            &receipt.source_release_proof_sha256,
            "source release proof commitment",
        ),
        (
            &receipt.target_activation_proof_sha256,
            "target activation proof commitment",
        ),
    ] {
        validate_digest(value, label)?;
    }
    if receipt.target_preparation_revision == 0 || receipt.accepted_schema_generation == 0 {
        return Err("the target admission receipt has an invalid revision".into());
    }
    validate_safe_line(
        &receipt.accepted_archive_codec,
        "accepted archive codec",
        120,
        true,
    )?;
    validate_safe_line(&receipt.created_at, "target admission timestamp", 120, true)?;
    if receipt.resolved_paths.is_empty() || receipt.resolved_paths.len() > 64 {
        return Err("the target admission receipt has an invalid resolved path list".into());
    }
    let mut aliases = HashSet::new();
    for path in &receipt.resolved_paths {
        validate_alias(&path.repository_alias, "resolved repository alias")?;
        validate_alias(&path.machine_alias, "resolved machine alias")?;
        validate_absolute_path(&path.path, "resolved repository path")?;
        if !aliases.insert(&path.repository_alias) {
            return Err("the target admission receipt repeats a repository alias".into());
        }
    }
    Ok(())
}

fn validate_source_release_receipt(
    receipt: &ProjectTransferSourceReleaseReceipt,
) -> Result<(), String> {
    validate_receipt_identities(
        &receipt.source_request_id,
        &receipt.target_request_id,
        &receipt.project_id,
        &receipt.source_space_id,
        &receipt.target_space_id,
    )?;
    validate_receipt_actor(
        &receipt.released_by,
        &receipt.source_space_id,
        "release actor",
    )?;
    for (value, label) in [
        (
            &receipt.source_configuration_sha256,
            "source configuration commitment",
        ),
        (
            &receipt.target_admission_sha256,
            "target admission commitment",
        ),
        (
            &receipt.target_preparation_sha256,
            "target preparation commitment",
        ),
        (
            &receipt.source_release_proof_sha256,
            "source release proof commitment",
        ),
        (
            &receipt.target_activation_proof_sha256,
            "target activation proof commitment",
        ),
    ] {
        validate_digest(value, label)?;
    }
    if receipt.target_preparation_revision == 0 || receipt.accepted_schema_generation == 0 {
        return Err("the source release receipt has an invalid revision".into());
    }
    validate_safe_line(
        &receipt.accepted_archive_codec,
        "accepted archive codec",
        120,
        true,
    )?;
    validate_transfer_graph_head(&receipt.source_head, true, "source release head")?;
    validate_safe_line(&receipt.created_at, "source release timestamp", 120, true)
}

fn validate_target_for_prepare(
    target: &ProjectTransferRecord,
    source: &ProjectTransferRecord,
    incoming: &ProjectProvisioningProjection,
    request: &ProjectTransferPrepareRequest,
) -> Result<(), String> {
    if target.linked_request_id.as_deref() != Some(request.source_request_id.as_str())
        || target.project_id != request.source_project_id
        || target.source_space_id != source.source_space_id
        || target.target_space_id != source.target_space_id
        || incoming.request_id != request.target_request_id
        || incoming.target_space_id != target.target_space_id
        || incoming.proposed_project_id != target.project_id
    {
        return Err(
            "the target transfer does not match the source and provisioning requests".into(),
        );
    }
    Ok(())
}

fn validate_link_receipt(
    receipt: &ProjectTransferLinkReceipt,
    source: &ProjectTransferRecord,
    target: &ProjectTransferRecord,
) -> Result<(), String> {
    if receipt.source_request_id != source.request_id
        || receipt.target_request_id != target.request_id
        || receipt.project_id != source.project_id
        || receipt.source_space_id != source.source_space_id
        || receipt.target_space_id != source.target_space_id
        || receipt.source_configuration_sha256
            != source
                .source_configuration_sha256
                .as_deref()
                .unwrap_or_default()
        || receipt.source_release_proof_sha256
            != source
                .source_release_proof_sha256
                .as_deref()
                .unwrap_or_default()
        || receipt.target_activation_proof_sha256
            != target
                .target_activation_proof_sha256
                .as_deref()
                .unwrap_or_default()
    {
        return Err("the target transfer link receipt does not match the source request".into());
    }
    if receipt.accepted_schema_generation != target.accepted_schema_generation.unwrap_or(0)
        || receipt.accepted_archive_codec
            != target.accepted_archive_codec.as_deref().unwrap_or_default()
    {
        return Err("the target transfer link receipt does not match target negotiation".into());
    }
    let configuration = source
        .source_configuration
        .as_ref()
        .ok_or_else(|| "the source transfer omitted its configuration".to_string())?;
    let expected = configuration
        .repositories
        .iter()
        .map(|repository| (&repository.alias, &repository.repository.identity))
        .collect::<Vec<_>>();
    if receipt.target_repositories.len() != expected.len()
        || receipt.target_repositories.iter().any(|repository| {
            !expected.iter().any(|(alias, identity)| {
                &repository.alias == *alias && &repository.repository.identity == *identity
            })
        })
    {
        return Err("the target transfer link receipt changed repository provenance".into());
    }
    Ok(())
}

fn validate_bundle(bundle: &ProjectTransferBundle) -> Result<(), String> {
    let source = &bundle.source;
    let target = &bundle.target;
    if source.side != "source"
        || target.side != "target"
        || source.linked_request_id.as_deref() != Some(target.request_id.as_str())
        || target.linked_request_id.as_deref() != Some(source.request_id.as_str())
        || source.project_id != target.project_id
        || source.source_space_id != target.source_space_id
        || source.target_space_id != target.target_space_id
        || source.source_space_id == source.target_space_id
        || source.source_configuration != target.source_configuration
        || source.source_configuration_sha256 != target.source_configuration_sha256
        || source.source_release_proof_sha256 != target.source_release_proof_sha256
        || target.target_activation_proof_sha256.is_none()
        || target.accepted_schema_generation.is_none()
        || target.accepted_archive_codec.is_none()
    {
        return Err("the transfer bundle has mismatched request, project, space, or configuration identities".into());
    }
    if bundle.incoming_provisioning.request_id != target.request_id
        || bundle.incoming_provisioning.target_space_id != target.target_space_id
        || bundle.incoming_provisioning.proposed_project_id != target.project_id
        || bundle.incoming_provisioning.state_repository.as_deref()
            != Some(source.source_configuration.state_repository.as_str())
        || bundle.incoming_provisioning.project_truth_scope
            != source.source_configuration.project_truth_scope
        || bundle.incoming_provisioning.default_run_truth_scope
            != source.source_configuration.default_run_truth_scope
    {
        return Err("the transfer bundle has mismatched incoming provisioning identity".into());
    }
    if bundle.incoming_provisioning.repositories.len()
        != source.source_configuration.repositories.len()
        || bundle
            .incoming_provisioning
            .repositories
            .iter()
            .any(|repository| {
                !source
                    .source_configuration
                    .repositories
                    .iter()
                    .any(|source_repository| {
                        source_repository.alias == repository.alias
                            && source_repository.repository.identity
                                == repository.repository.identity
                            && source_repository.machine_alias == repository.machine_alias
                    })
            })
    {
        return Err("the incoming provisioning repository provenance changed".into());
    }
    if target.accepted_schema_generation
        != Some(source.source_configuration.source_schema_generation)
        || !source
            .source_configuration
            .supported_archive_codecs
            .iter()
            .any(|codec| Some(codec) == target.accepted_archive_codec.as_ref())
    {
        return Err("the transfer bundle has an unsupported target negotiation".into());
    }
    Ok(())
}

fn parse_project_provisioning_projection(
    value: &Value,
    expected_request_id: &str,
    expected_space_id: &str,
) -> Result<ProjectProvisioningProjection, String> {
    let object = value.as_object().ok_or_else(|| {
        "the incoming provisioning request returned an invalid response".to_string()
    })?;
    let request_id = required_text(object, "request_id")?.to_string();
    if request_id != expected_request_id {
        return Err("the incoming provisioning request identity does not match".into());
    }
    validate_uuid4(&request_id, "incoming provisioning request identity")?;
    let kind = required_text(object, "kind")?.to_string();
    if kind != "incoming_transfer" {
        return Err("the target provisioning request is not an incoming transfer".into());
    }
    let status = required_text(object, "status")?.to_string();
    if !matches!(
        status.as_str(),
        "waiting_for_server_setup"
            | "setup_in_progress"
            | "operator_action_needed"
            | "ready_for_review"
            | "completed"
            | "cancelled"
    ) {
        return Err("the incoming provisioning request has an invalid status".into());
    }
    let target_space_id = required_text(object, "target_space_id")?.to_string();
    let proposed_project_id = required_text(object, "proposed_project_id")?.to_string();
    validate_uuid4(&target_space_id, "target transfer space identity")?;
    validate_uuid4(&proposed_project_id, "incoming transfer project identity")?;
    if target_space_id != expected_space_id {
        return Err("the incoming provisioning request belongs to another team space".into());
    }
    let machines = object_array(object, "machines")?
        .iter()
        .map(parse_machine_projection)
        .collect::<Result<Vec<_>, _>>()?;
    let repositories = object_array(object, "repositories")?
        .iter()
        .map(parse_repository_projection)
        .collect::<Result<Vec<_>, _>>()?;
    let provider_checks = object_array(object, "provider_checks")?
        .iter()
        .map(parse_provider_projection)
        .collect::<Result<Vec<_>, _>>()?;
    let readiness = parse_readiness_projection(object.get("readiness").ok_or_else(|| {
        "the incoming provisioning request has no readiness projection".to_string()
    })?)?;
    let default_ceiling = required_u64(object, "default_auto_research_invocation_ceiling")?;
    if default_ceiling == 0 {
        return Err("incoming provisioning invocation ceiling must be positive".into());
    }
    let authorized_by =
        parse_authorized_human(object.get("authorized_by").ok_or_else(|| {
            "the incoming provisioning request has no authorized_by".to_string()
        })?)?;
    let final_review = parse_final_review(object)?;
    let final_review_digest = final_review.as_ref().map(|review| review.digest.clone());
    Ok(ProjectProvisioningProjection {
        request_id,
        kind,
        status,
        status_label: required_text(object, "status_label")?.to_string(),
        next_action: required_nullable_text(object, "next_action")?,
        can_run_setup: required_bool(object, "can_run_setup")?,
        can_review: required_bool(object, "can_review")?,
        can_cancel: required_bool(object, "can_cancel")?,
        target_space_id,
        proposed_project_id,
        name: optional_text(object, "name")?,
        state_repository: optional_text(object, "state_repository")?,
        project_truth_scope: string_array(object, "project_truth_scope")?,
        default_run_truth_scope: string_array(object, "default_run_truth_scope")?,
        default_auto_research_invocation_ceiling: default_ceiling,
        authorized_by,
        machines,
        repositories,
        provider_checks,
        readiness,
        diagnostic: optional_text(object, "diagnostic")?,
        operator_action: required_nullable_object(object, "operator_action")?,
        operator_argv: string_array(object, "operator_argv")?,
        final_review,
        final_review_digest,
        cancellation_disposition: required_nullable_text(object, "cancellation_disposition")?,
        revision: required_u64(object, "revision")?,
        created_at: required_text(object, "created_at")?.to_string(),
        updated_at: required_text(object, "updated_at")?.to_string(),
        setup_started_at: required_nullable_text(object, "setup_started_at")?,
        completed_at: required_nullable_text(object, "completed_at")?,
        cancelled_at: required_nullable_text(object, "cancelled_at")?,
    })
}

fn parse_authorized_human(value: &Value) -> Result<ProjectProvisioningAuthorizedHuman, String> {
    let human: ProjectProvisioningAuthorizedHuman = serde_json::from_value(value.clone())
        .map_err(|_| "the provisioning response has an invalid authorized_by".to_string())?;
    validate_uuid4(&human.space_id, "provisioning authorized space identity")?;
    validate_uuid4(&human.user_id, "provisioning authorized user identity")?;
    validate_safe_line(
        &human.display_name,
        "provisioning authorized display name",
        120,
        true,
    )?;
    Ok(human)
}

fn parse_final_review(
    object: &serde_json::Map<String, Value>,
) -> Result<Option<ProjectProvisioningFinalReviewProjection>, String> {
    match object.get("final_review") {
        None => Err("the incoming provisioning request has no final_review field".into()),
        Some(Value::Null) => Ok(None),
        Some(value) => {
            let review: ProjectProvisioningFinalReviewProjection =
                serde_json::from_value(value.clone()).map_err(|_| {
                    "the incoming provisioning request has an invalid final review".to_string()
                })?;
            validate_digest(
                &review.digest,
                "incoming provisioning final-review commitment",
            )?;
            validate_uuid4(
                &review.proposed_project_id,
                "incoming provisioning final-review project identity",
            )?;
            Ok(Some(review))
        }
    }
}

fn required_nullable_object(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<Value>, String> {
    match object.get(field) {
        None => Err(format!(
            "the incoming provisioning request has no {field} field"
        )),
        Some(Value::Null) => Ok(None),
        Some(value) if value.is_object() => Ok(Some(value.clone())),
        Some(_) => Err(format!(
            "the incoming provisioning request has an invalid {field}"
        )),
    }
}

fn parse_machine_projection(value: &Value) -> Result<ProjectProvisioningMachineProjection, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "the target machine projection is invalid".to_string())?;
    Ok(ProjectProvisioningMachineProjection {
        alias: required_text(object, "alias")?.to_string(),
        location: required_text(object, "location")?.to_string(),
        host: required_text(object, "host")?.to_string(),
        os_account: required_text(object, "os_account")?.to_string(),
        intended_central_root: optional_text(object, "intended_central_root")?,
        resolved_central_root: optional_text(object, "resolved_central_root")?,
        ready: required_bool(object, "ready")?,
        status_label: required_text(object, "status_label")?.to_string(),
    })
}

fn parse_repository_projection(
    value: &Value,
) -> Result<ProjectProvisioningRepositoryProjection, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "the target repository projection is invalid".to_string())?;
    let repository_object = object
        .get("repository")
        .and_then(Value::as_object)
        .ok_or_else(|| "the target repository projection has no repository identity".to_string())?;
    let identity = required_text(repository_object, "identity")?.to_string();
    validate_repository_identity(&identity)?;
    Ok(ProjectProvisioningRepositoryProjection {
        alias: required_text(object, "alias")?.to_string(),
        repository: ProjectTransferRepositoryIdentity { identity },
        https_clone_url: required_text(object, "https_clone_url")?.to_string(),
        ssh_clone_url: required_text(object, "ssh_clone_url")?.to_string(),
        settings_url: required_text(object, "settings_url")?.to_string(),
        machine_alias: required_text(object, "machine_alias")?.to_string(),
        intended_path: optional_text(object, "intended_path")?,
        resolved_path: optional_text(object, "resolved_path")?,
        checkout_disposition: optional_text(object, "checkout_disposition")?,
        status: required_text(object, "status")?.to_string(),
        status_label: required_text(object, "status_label")?.to_string(),
        ready: required_bool(object, "ready")?,
        commit: optional_text(object, "commit")?,
        write_verified: required_bool(object, "write_verified")?,
        deploy_key_label: optional_text(object, "deploy_key_label")?,
        public_key_fingerprint: optional_text(object, "public_key_fingerprint")?,
        checked_at: optional_text(object, "checked_at")?,
        diagnostic: optional_text(object, "diagnostic")?,
    })
}

fn parse_provider_projection(
    value: &Value,
) -> Result<ProjectProvisioningProviderProjection, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "the target provider projection is invalid".to_string())?;
    Ok(ProjectProvisioningProviderProjection {
        profile: required_text(object, "profile")?.to_string(),
        provider: required_text(object, "provider")?.to_string(),
        runtime_id: required_text(object, "runtime_id")?.to_string(),
        model: required_text(object, "model")?.to_string(),
        reasoning: required_text(object, "reasoning")?.to_string(),
        machine_alias: required_text(object, "machine_alias")?.to_string(),
        status: required_text(object, "status")?.to_string(),
        status_label: required_text(object, "status_label")?.to_string(),
        ready: required_bool(object, "ready")?,
        binary_path: optional_text(object, "binary_path")?,
        version: optional_text(object, "version")?,
        resolved_runtime_id: optional_text(object, "resolved_runtime_id")?,
        execution_account: optional_text(object, "execution_account")?,
        checked_at: optional_text(object, "checked_at")?,
        diagnostic: optional_text(object, "diagnostic")?,
    })
}

fn parse_readiness_projection(
    value: &Value,
) -> Result<ProjectProvisioningReadinessProjection, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "the target provisioning readiness projection is invalid".to_string())?;
    Ok(ProjectProvisioningReadinessProjection {
        machines_ready: required_u64(object, "machines_ready")?,
        machines_total: required_u64(object, "machines_total")?,
        repositories_ready: required_u64(object, "repositories_ready")?,
        repositories_total: required_u64(object, "repositories_total")?,
        providers_ready: required_u64(object, "providers_ready")?,
        providers_total: required_u64(object, "providers_total")?,
        all_ready: required_bool(object, "all_ready")?,
    })
}

fn parse_provider_setup_projection(
    value: &Value,
) -> Result<Vec<TargetProviderSetupProjection>, String> {
    let providers = value.as_array().ok_or_else(|| {
        "the target provider setup endpoint returned an invalid response".to_string()
    })?;
    providers
        .iter()
        .map(|value| {
            let object = value.as_object().ok_or_else(|| {
                "the target provider setup endpoint returned an invalid provider".to_string()
            })?;
            let models = object_array(object, "models")?
                .iter()
                .map(|value| {
                    let model = value.as_object().ok_or_else(|| {
                        "the target provider setup endpoint returned an invalid model".to_string()
                    })?;
                    Ok(TargetProviderModelProjection {
                        id: required_text(model, "id")?.to_string(),
                        label: required_text(model, "label")?.to_string(),
                        reasoning: string_array(model, "reasoning")?,
                        default_reasoning: required_text(model, "default_reasoning")?.to_string(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let runtimes = object_array(object, "runtimes")?
                .iter()
                .map(|value| {
                    let runtime = value.as_object().ok_or_else(|| {
                        "the target provider setup endpoint returned an invalid runtime".to_string()
                    })?;
                    Ok(TargetProviderRuntimeProjection {
                        id: required_text(runtime, "id")?.to_string(),
                        label: required_text(runtime, "label")?.to_string(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(TargetProviderSetupProjection {
                provider: required_text(object, "provider")?.to_string(),
                label: required_text(object, "label")?.to_string(),
                installed: required_bool(object, "installed")?,
                authenticated: required_bool(object, "authenticated")?,
                version: optional_text(object, "version")?,
                reason: optional_text(object, "reason")?,
                binary_path: optional_text(object, "binary_path")?,
                path_state: required_text(object, "path_state")?.to_string(),
                models,
                runtimes,
                default_runtime: required_text(object, "default_runtime")?.to_string(),
            })
        })
        .collect()
}

fn required_text<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("the response has no valid {field}"))
}

fn required_bool(object: &serde_json::Map<String, Value>, field: &str) -> Result<bool, String> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("the response has no valid {field}"))
}

fn required_u64(object: &serde_json::Map<String, Value>, field: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("the response has no valid {field}"))
}

fn string_array(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Vec<String>, String> {
    object_array(object, field)?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("the response has no valid {field} item"))
        })
        .collect()
}

fn object_array<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a Vec<Value>, String> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("the response has no valid {field} list"))
}

pub async fn run(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    request_id: &str,
    on_event: &Channel<Value>,
    ssh_program: PathBuf,
) -> Result<ProjectTransferRunResult, String> {
    let (pinned, source) = load_source_request(lifecycle, request_id, false).await?;
    let target =
        resolve_target_connection(lifecycle, connections, sessions, tunnels, &source).await?;
    let target_readback = read_target_transfer(sessions, connections, &source, &target).await?;

    if matches!(source.phase.as_str(), "cleanup_acknowledged" | "completed") {
        if target_readback.phase == "completed" {
            return Ok(completed_run_result(&source, &target));
        }
        if target_readback.phase != "target_activated" {
            return Err(format!(
                "the source transfer is {} but the target transfer is {}; retry after the target reaches activation",
                source.phase, target_readback.phase
            ));
        }
        let archive_sha256 = source.archive_sha256.clone();
        let archive_size_bytes = source.archive_size_bytes;
        let finish =
            finish_loaded(lifecycle, connections, sessions, tunnels, source, target).await?;
        return Ok(ProjectTransferRunResult {
            request_id: finish.request_id,
            target_request_id: finish.target_request_id,
            target_space_id: finish.target_space_id,
            connection_id: finish.connection_id,
            archive_sha256,
            archive_size_bytes,
            exit_code: 0,
            event_count: 0,
            proof_verified: finish.proof_verified,
            cleanup_acknowledged: finish.cleanup_acknowledged,
        });
    }
    if source.phase != "archive_bound" {
        return Err("the source transfer is not at a relayable archive boundary".into());
    }
    if target_readback.phase == "completed" {
        return Err(concat!(
            "the target transfer is complete while the source remains archive-bound; ",
            "inspect both durable requests before retrying"
        )
        .into());
    }
    if target_readback.phase == "target_activated" {
        let archive_sha256 = source.archive_sha256.clone();
        let archive_size_bytes = source.archive_size_bytes;
        let finish =
            finish_loaded(lifecycle, connections, sessions, tunnels, source, target).await?;
        return Ok(ProjectTransferRunResult {
            request_id: finish.request_id,
            target_request_id: finish.target_request_id,
            target_space_id: finish.target_space_id,
            connection_id: finish.connection_id,
            archive_sha256,
            archive_size_bytes,
            exit_code: 0,
            event_count: 0,
            proof_verified: finish.proof_verified,
            cleanup_acknowledged: finish.cleanup_acknowledged,
        });
    }
    let archive = fetch_source_archive(&pinned, &source).await?;
    let (exit_code, event_count) = server_commands::run_project_transfer_import(
        &target,
        &source.target_request_id,
        archive,
        &source.archive_sha256,
        source.archive_size_bytes,
        on_event,
        ssh_program,
    )
    .await?;

    if exit_code != 0 {
        return Ok(ProjectTransferRunResult {
            request_id: source.request_id,
            target_request_id: source.target_request_id,
            target_space_id: source.target_space_id,
            connection_id: target.connection_id,
            archive_sha256: source.archive_sha256,
            archive_size_bytes: source.archive_size_bytes,
            exit_code,
            event_count,
            proof_verified: false,
            cleanup_acknowledged: false,
        });
    }

    let activated = read_target_transfer(sessions, connections, &source, &target).await?;
    if activated.phase != "target_activated" {
        return Err(format!(
            "the transfer command exited successfully but the target transfer is {}",
            activated.phase
        ));
    }
    let archive_sha256 = source.archive_sha256.clone();
    let archive_size_bytes = source.archive_size_bytes;
    let finish = finish_loaded(lifecycle, connections, sessions, tunnels, source, target).await?;
    Ok(ProjectTransferRunResult {
        request_id: finish.request_id,
        target_request_id: finish.target_request_id,
        target_space_id: finish.target_space_id,
        connection_id: finish.connection_id,
        archive_sha256,
        archive_size_bytes,
        exit_code,
        event_count,
        proof_verified: finish.proof_verified,
        cleanup_acknowledged: finish.cleanup_acknowledged,
    })
}

pub async fn finish(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    request_id: &str,
    archive_path: PathBuf,
) -> Result<ProjectTransferFinishResult, String> {
    let (_pinned, source) = load_source_request(lifecycle, request_id, false).await?;
    let target =
        resolve_target_connection(lifecycle, connections, sessions, tunnels, &source).await?;
    let target_readback = read_target_transfer(sessions, connections, &source, &target).await?;
    if matches!(source.phase.as_str(), "cleanup_acknowledged" | "completed")
        && target_readback.phase == "completed"
    {
        let cleanup_source = source.clone();
        let result = ProjectTransferFinishResult {
            request_id: source.request_id,
            target_request_id: source.target_request_id,
            target_space_id: source.target_space_id,
            connection_id: target.connection_id,
            proof_verified: true,
            cleanup_acknowledged: true,
        };
        remove_local_export(&archive_path, &cleanup_source)?;
        return Ok(result);
    }
    if target_readback.phase != "target_activated" {
        return Err(format!(
            "the target transfer is {}; finish-proof requires durable target activation",
            target_readback.phase
        ));
    }
    let cleanup_source = source.clone();
    let result = finish_loaded(lifecycle, connections, sessions, tunnels, source, target).await?;
    remove_local_export(&archive_path, &cleanup_source)?;
    Ok(result)
}

pub async fn discard_export(
    lifecycle: &BackendState,
    request_id: &str,
    archive_path: PathBuf,
) -> Result<ProjectTransferExportCleanupResult, String> {
    let (_pinned, source) = load_source_request(lifecycle, request_id, false).await?;
    remove_local_export(&archive_path, &source)?;
    Ok(ProjectTransferExportCleanupResult {
        request_id: source.request_id,
        removed: true,
        path: archive_path.display().to_string(),
    })
}

pub async fn export(
    lifecycle: &BackendState,
    request_id: &str,
    destination: PathBuf,
) -> Result<ProjectTransferExportResult, String> {
    let (pinned, source) = load_source_request(lifecycle, request_id, true).await?;
    validate_export_destination(&destination)?;
    let archive = fetch_source_archive(&pinned, &source).await?;
    let path = write_local_archive(archive, &source, &destination).await?;
    Ok(ProjectTransferExportResult {
        saved: true,
        request_id: source.request_id,
        target_request_id: Some(source.target_request_id),
        target_space_id: Some(source.target_space_id),
        archive_sha256: Some(source.archive_sha256),
        archive_size_bytes: Some(source.archive_size_bytes),
        path: Some(path),
    })
}

/// Verify a manually exported archive selected after a desktop restart. The
/// source request is loaded above this boundary so the browser cannot choose
/// which transfer metadata to trust. Selection itself is read-only: it neither
/// copies the archive nor advances either transfer request.
pub async fn select_export(
    lifecycle: &BackendState,
    request_id: &str,
    archive_path: PathBuf,
) -> Result<ProjectTransferExportSelectionResult, String> {
    let (_pinned, source) = load_source_request(lifecycle, request_id, true).await?;
    select_verified_export(&source, archive_path)
}

fn select_verified_export(
    source: &SourceTransferRequest,
    archive_path: PathBuf,
) -> Result<ProjectTransferExportSelectionResult, String> {
    verify_local_archive(&archive_path, source)?;
    Ok(ProjectTransferExportSelectionResult {
        selected: true,
        request_id: source.request_id.clone(),
        target_request_id: Some(source.target_request_id.clone()),
        target_space_id: Some(source.target_space_id.clone()),
        archive_sha256: Some(source.archive_sha256.clone()),
        archive_size_bytes: Some(source.archive_size_bytes),
        path: Some(archive_path.display().to_string()),
    })
}

pub async fn terminal(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    request_id: &str,
    archive_path: PathBuf,
) -> Result<TerminalLaunchResult, String> {
    let (_pinned, source) = load_source_request(lifecycle, request_id, true).await?;
    let target =
        resolve_target_connection(lifecycle, connections, sessions, tunnels, &source).await?;
    verify_local_archive(&archive_path, &source)?;
    let argv =
        server_commands::terminal_transfer_argv(&target, &source.target_request_id, &archive_path)?;
    server_commands::open_terminal(argv).await
}

async fn finish_loaded(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    source: SourceTransferRequest,
    target: TeamConnectionMetadata,
) -> Result<ProjectTransferFinishResult, String> {
    let established = sessions.established(&target.connection_id)?;
    let ready = tunnels
        .connect_saved(connections, lifecycle, &target.connection_id)
        .await?;
    if ready.local_origin != established.connection.local_origin {
        return Err("the established team session is not pinned to its saved tunnel".into());
    }
    let proof = sessions
        .retrieve_target_activation_proof(
            connections,
            &target.connection_id,
            &source.target_request_id,
        )
        .await?;

    // Reverify immediately before the proof crosses back to the source.  The
    // original source backend identity is not allowed to drift during relay.
    let status = lifecycle.status()?;
    let pinned = pin_personal_backend(lifecycle, &status).await?;
    let acknowledgment = post_target_activation_proof(&pinned, &source, proof.as_slice()).await?;
    sessions
        .post_cleanup_acknowledgment(
            connections,
            &target.connection_id,
            &source.target_request_id,
            &acknowledgment,
        )
        .await?;
    Ok(ProjectTransferFinishResult {
        request_id: source.request_id,
        target_request_id: source.target_request_id,
        target_space_id: source.target_space_id,
        connection_id: target.connection_id,
        proof_verified: true,
        cleanup_acknowledged: true,
    })
}

async fn load_source_request(
    lifecycle: &BackendState,
    request_id: &str,
    require_archive: bool,
) -> Result<(PinnedPersonalBackend, SourceTransferRequest), String> {
    validate_uuid4(request_id, "project transfer request identity")?;
    let status = lifecycle.status()?;
    let pinned = pin_personal_backend(lifecycle, &status).await?;
    let client = personal_client(&pinned.base_url)?;
    let response = client
        .get(format!(
            "{}{}",
            pinned.base_url,
            source_request_path(request_id)
        ))
        .header(
            "X-RCP-Instance-ID",
            HeaderValue::from_str(&pinned.instance_id)
                .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
        )
        .send()
        .await
        .map_err(|error| format!("could not read the personal transfer request: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "the personal transfer request was rejected (HTTP {})",
            response.status().as_u16()
        ));
    }
    let value = response
        .json::<Value>()
        .await
        .map_err(|_| "the personal transfer request returned an invalid response".to_string())?;
    let source = parse_source_request(&value, request_id, require_archive)?;
    Ok((pinned, source))
}

async fn pin_personal_backend(
    lifecycle: &BackendState,
    status: &DesktopStatus,
) -> Result<PinnedPersonalBackend, String> {
    let health = backend::reverify_identity(lifecycle, status).await?;
    if health.instance_id != status.instance_id {
        return Err("the personal backend identity changed before transfer relay".into());
    }
    Ok(PinnedPersonalBackend {
        base_url: status.base_url.trim_end_matches('/').to_string(),
        instance_id: health.instance_id,
    })
}

fn personal_client(base_url: &str) -> Result<Client, String> {
    Client::builder()
        .timeout(PERSONAL_REQUEST_TIMEOUT)
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("could not create the personal transfer client: {error}"))
        .and_then(|client| {
            let url = url::Url::parse(base_url)
                .map_err(|_| "the personal backend origin is invalid".to_string())?;
            if url.scheme() != "http" && url.scheme() != "https" {
                return Err("the personal backend origin is invalid".into());
            }
            Ok(client)
        })
}

fn personal_stream_client(base_url: &str) -> Result<Client, String> {
    Client::builder()
        .connect_timeout(PERSONAL_REQUEST_TIMEOUT)
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("could not create the personal transfer client: {error}"))
        .and_then(|client| {
            let url = url::Url::parse(base_url)
                .map_err(|_| "the personal backend origin is invalid".to_string())?;
            if url.scheme() != "http" && url.scheme() != "https" {
                return Err("the personal backend origin is invalid".into());
            }
            Ok(client)
        })
}

fn source_request_path(request_id: &str) -> String {
    format!("{SOURCE_REQUEST_PATH_PREFIX}{request_id}")
}

fn source_archive_path(request_id: &str) -> String {
    format!("{SOURCE_ARCHIVE_PATH_PREFIX}{request_id}/archive")
}

async fn fetch_source_archive(
    pinned: &PinnedPersonalBackend,
    source: &SourceTransferRequest,
) -> Result<Response, String> {
    let client = personal_stream_client(&pinned.base_url)?;
    let response = tokio::time::timeout(
        PERSONAL_REQUEST_TIMEOUT,
        client
            .get(format!(
                "{}{}",
                pinned.base_url,
                source_archive_path(&source.request_id)
            ))
            .header(
                "X-RCP-Instance-ID",
                HeaderValue::from_str(&pinned.instance_id)
                    .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
            )
            .send(),
    )
    .await
    .map_err(|_| "the personal transfer archive response headers timed out".to_string())?
    .map_err(|error| format!("could not stream the personal transfer archive: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "the personal transfer archive was rejected (HTTP {})",
            response.status().as_u16()
        ));
    }
    validate_archive_headers(&response, source)?;
    Ok(response)
}

fn validate_archive_headers(
    response: &Response,
    source: &SourceTransferRequest,
) -> Result<(), String> {
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or_default().trim());
    if content_type != Some(TRANSFER_ARCHIVE_CONTENT_TYPE) {
        return Err("the personal transfer archive returned an invalid content type".into());
    }
    let content_length = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| "the personal transfer archive omitted its bounded size".to_string())?;
    if content_length != source.archive_size_bytes {
        return Err("the personal transfer archive size differs from its durable receipt".into());
    }
    let digest = response
        .headers()
        .get("X-RCP-Archive-SHA256")
        .and_then(|value| value.to_str().ok());
    if digest != Some(source.archive_sha256.as_str()) {
        return Err("the personal transfer archive digest differs from its durable receipt".into());
    }
    Ok(())
}

async fn resolve_target_connection(
    lifecycle: &BackendState,
    connections: &TeamConnectionState,
    sessions: &TeamSessionState,
    tunnels: &TeamTunnelState,
    source: &SourceTransferRequest,
) -> Result<TeamConnectionMetadata, String> {
    let matches = connections
        .list()?
        .into_iter()
        .filter(|connection| connection.expected_space_id == source.target_space_id)
        .collect::<Vec<_>>();
    let [target] = matches.as_slice() else {
        return Err(
            "the source transfer target space has no unique saved desktop connection".into(),
        );
    };
    server_commands::configured_route(target)?;
    sessions
        .ensure_native_session(connections, tunnels, lifecycle, &target.connection_id)
        .await?;
    Ok(target.clone())
}

async fn read_target_transfer(
    sessions: &TeamSessionState,
    connections: &TeamConnectionState,
    source: &SourceTransferRequest,
    target: &TeamConnectionMetadata,
) -> Result<ProjectTransferTargetReadback, String> {
    let readback = sessions
        .read_project_transfer(
            connections,
            &target.connection_id,
            &source.target_request_id,
        )
        .await?;
    if readback.linked_request_id != source.request_id
        || readback.target_space_id != source.target_space_id
    {
        return Err("the target transfer readback does not match the source request".into());
    }
    Ok(readback)
}

fn completed_run_result(
    source: &SourceTransferRequest,
    target: &TeamConnectionMetadata,
) -> ProjectTransferRunResult {
    ProjectTransferRunResult {
        request_id: source.request_id.clone(),
        target_request_id: source.target_request_id.clone(),
        target_space_id: source.target_space_id.clone(),
        connection_id: target.connection_id.clone(),
        archive_sha256: source.archive_sha256.clone(),
        archive_size_bytes: source.archive_size_bytes,
        exit_code: 0,
        event_count: 0,
        proof_verified: true,
        cleanup_acknowledged: true,
    }
}

async fn post_target_activation_proof(
    pinned: &PinnedPersonalBackend,
    source: &SourceTransferRequest,
    proof: &[u8],
) -> Result<ProjectTransferCleanupAcknowledgment, String> {
    if proof.len() != MAX_PROOF_BYTES {
        return Err("the target activation proof has an invalid size".into());
    }
    let digest = hex_digest(proof);
    if digest != source.target_activation_proof_sha256 {
        return Err("the target activation proof does not match its commitment".into());
    }
    let client = personal_client(&pinned.base_url)?;
    let content_type = HeaderValue::from_static(TRANSFER_ARCHIVE_CONTENT_TYPE);
    let proof_path = format!(
        "{SOURCE_ARCHIVE_PATH_PREFIX}{}/target-activation-proof",
        source.request_id
    );
    let response = client
        .post(format!("{}{proof_path}", pinned.base_url))
        .header(
            "X-RCP-Instance-ID",
            HeaderValue::from_str(&pinned.instance_id)
                .map_err(|_| "the personal backend instance identity is invalid".to_string())?,
        )
        .header(CONTENT_TYPE, content_type)
        .body(proof.to_vec())
        .send()
        .await
        .map_err(|error| {
            format!("could not return target activation proof to the personal backend: {error}")
        })?;
    if !response.status().is_success() {
        return Err(format!(
            "the personal backend rejected target activation proof (HTTP {})",
            response.status().as_u16()
        ));
    }
    let acknowledgment = response
        .json::<ProjectTransferCleanupAcknowledgment>()
        .await
        .map_err(|_| {
            "the personal backend returned an invalid cleanup acknowledgment".to_string()
        })?;
    acknowledgment.validate()?;
    if acknowledgment.source_request_id != source.request_id
        || acknowledgment.target_request_id != source.target_request_id
        || acknowledgment.project_id != source.project_id
        || acknowledgment.source_space_id != source.source_space_id
        || acknowledgment.target_space_id != source.target_space_id
        || acknowledgment.archive_sha256 != source.archive_sha256
        || acknowledgment.target_activation_proof_sha256 != digest
    {
        return Err("the personal backend returned a mismatched cleanup acknowledgment".into());
    }
    Ok(acknowledgment)
}

async fn write_local_archive(
    mut response: Response,
    source: &SourceTransferRequest,
    destination: &Path,
) -> Result<String, String> {
    let parent = destination
        .parent()
        .ok_or_else(|| "the local transfer archive has no parent directory".to_string())?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".rcp-transfer-")
        .tempfile_in(parent)
        .map_err(|error| format!("could not create a protected local transfer export: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        temporary
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("could not protect the local transfer export: {error}"))?;
    }
    let mut hasher = Sha256::new();
    let mut size = 0_u64;
    while let Some(chunk) = tokio::time::timeout(TRANSFER_STREAM_IDLE_TIMEOUT, response.chunk())
        .await
        .map_err(|_| "the personal transfer archive stream made no progress".to_string())?
        .map_err(|error| format!("the personal transfer archive stream failed: {error}"))?
    {
        let chunk_size = u64::try_from(chunk.len())
            .map_err(|_| "the local transfer archive size is too large".to_string())?;
        size = size
            .checked_add(chunk_size)
            .ok_or_else(|| "the local transfer archive size overflowed".to_string())?;
        if size > source.archive_size_bytes {
            return Err("the personal transfer archive exceeded its durable size".into());
        }
        hasher.update(&chunk);
        temporary.write_all(&chunk).map_err(|error| {
            format!("could not write the protected local transfer export: {error}")
        })?;
    }
    if size != source.archive_size_bytes
        || format_digest(&hasher.finalize()) != source.archive_sha256
    {
        return Err("the local transfer export differs from its durable archive receipt".into());
    }
    temporary.as_file().sync_all().map_err(|error| {
        format!("could not finish the protected local transfer export: {error}")
    })?;
    temporary.persist_noclobber(destination).map_err(|error| {
        format!(
            "could not publish the protected local transfer export: {}",
            error.error
        )
    })?;
    Ok(destination.display().to_string())
}

fn validate_export_destination(destination: &Path) -> Result<(), String> {
    if !destination.is_absolute()
        || destination == Path::new("/")
        || destination
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("the local transfer export must be one specific absolute path".into());
    }
    let value = destination
        .to_str()
        .ok_or_else(|| "the local transfer export path is not valid UTF-8".to_string())?;
    if value.len() > 4096 || value.chars().any(char::is_control) {
        return Err("the local transfer export path is not bounded and safe".into());
    }
    if destination.exists() {
        return Err("the selected local transfer export already exists".into());
    }
    Ok(())
}

fn verify_local_archive(path: &Path, source: &SourceTransferRequest) -> Result<(), String> {
    validate_export_destination_for_read(path)?;
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("could not inspect the local transfer export: {error}"))?;
    if !metadata.file_type().is_file() || metadata.len() != source.archive_size_bytes {
        return Err("the local transfer export does not match the source archive size".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o777 != 0o600 {
            return Err("the local transfer export must be mode 0600".into());
        }
    }
    let mut file = File::open(path)
        .map_err(|error| format!("could not open the local transfer export: {error}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("could not read the local transfer export: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    if format_digest(&hasher.finalize()) != source.archive_sha256 {
        return Err("the local transfer export differs from its durable archive digest".into());
    }
    Ok(())
}

fn remove_local_export(path: &Path, source: &SourceTransferRequest) -> Result<(), String> {
    verify_local_archive(path, source)?;
    fs::remove_file(path).map_err(|error| {
        format!("could not remove the completed local transfer export: {error}")
    })?;
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        let directory = File::open(parent).map_err(|error| {
            format!("could not open the local transfer export directory: {error}")
        })?;
        directory.sync_all().map_err(|error| {
            format!("could not finish removing the local transfer export: {error}")
        })?;
    }
    Ok(())
}

fn validate_export_destination_for_read(path: &Path) -> Result<(), String> {
    if !path.is_absolute()
        || path == Path::new("/")
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("the local transfer export must be one specific absolute path".into());
    }
    let value = path
        .to_str()
        .ok_or_else(|| "the local transfer export path is not valid UTF-8".to_string())?;
    if value.len() > 4096 || value.chars().any(char::is_control) {
        return Err("the local transfer export path is not bounded and safe".into());
    }
    Ok(())
}

fn validate_uuid4(value: &str, label: &str) -> Result<(), String> {
    let parsed = Uuid::parse_str(value).map_err(|_| format!("{label} is invalid"))?;
    if parsed.get_version() != Some(UuidVersion::Random) || parsed.to_string() != value {
        return Err(format!("{label} is invalid"));
    }
    Ok(())
}

pub(crate) fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(format!("{label} is invalid"));
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format_digest(&digest)
}

fn format_digest(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    const REQUEST_ID: &str = "11111111-1111-4111-8111-111111111111";
    const TARGET_ID: &str = "22222222-2222-4222-8222-222222222222";
    const PROJECT_ID: &str = "33333333-3333-4333-8333-333333333333";
    const SOURCE_SPACE_ID: &str = "44444444-4444-4444-8444-444444444444";
    const TARGET_SPACE_ID: &str = "55555555-5555-4555-8555-555555555555";

    fn acknowledgment() -> ProjectTransferCleanupAcknowledgment {
        ProjectTransferCleanupAcknowledgment {
            source_request_id: REQUEST_ID.into(),
            target_request_id: TARGET_ID.into(),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            source_release_proof_sha256: "a".repeat(64),
            target_activation_proof_sha256: "b".repeat(64),
            archive_sha256: "c".repeat(64),
            source_fence_head: TransferGraphHead {
                target: TransferGraphTarget {
                    kind: "main".into(),
                    branch_id: None,
                },
                revision: 3,
                transition_id: Some("d".repeat(64)),
            },
        }
    }

    fn prepare_request() -> ProjectTransferPrepareRequest {
        ProjectTransferPrepareRequest {
            source_request_id: REQUEST_ID.into(),
            target_request_id: TARGET_ID.into(),
            connection_id: "66666666-6666-4666-8666-666666666666".into(),
            source_project_id: PROJECT_ID.into(),
            target_provisioning: ProjectTransferTargetProvisioningIntent {
                name: "Moved project".into(),
                default_auto_research_invocation_ceiling: 3,
                machines: vec![ProjectTransferMachineIntent {
                    alias: "server".into(),
                    location: "local".into(),
                    host: String::new(),
                    os_account: "rcp".into(),
                    central_root: Some("/srv/rcp/projects".into()),
                }],
                provider_checks: vec![ProjectTransferProviderIntent {
                    profile: "discuss".into(),
                    provider: "codex".into(),
                    runtime_id: "exec".into(),
                    model: "model".into(),
                    reasoning: "medium".into(),
                    machine_alias: "server".into(),
                }],
            },
        }
    }

    #[test]
    fn prepare_input_is_strict_and_contains_no_secret_or_archive_fields() {
        let request = prepare_request();
        let value = serde_json::to_value(&request).unwrap();
        assert!(!value.to_string().contains("archive_bytes"));
        assert!(!value.to_string().contains("proof_bytes"));
        assert!(!value.to_string().contains("member_token"));
        let mut extra = value.clone();
        extra["target_provisioning"]["repositories"] = Value::Array(Vec::new());
        assert!(serde_json::from_value::<ProjectTransferPrepareRequest>(extra).is_err());
    }

    #[test]
    fn archive_codec_selection_follows_the_source_configuration() {
        let configuration = ProjectTransferSourceConfiguration {
            source_rcp_version: "0.3.2".into(),
            source_schema_generation: 9,
            supported_archive_codecs: vec!["source-preferred-codec".into()],
            machine_aliases: vec!["server".into()],
            repositories: vec![ProjectTransferRepositorySource {
                alias: "state".into(),
                repository: ProjectTransferRepositoryIdentity {
                    identity: "example/state".into(),
                },
                machine_alias: "server".into(),
            }],
            state_repository: "state".into(),
            project_truth_scope: vec!["state".into()],
            default_run_truth_scope: vec!["state".into()],
            source_manifest_sha256: "a".repeat(64),
        };
        assert_eq!(
            choose_archive_codec(&configuration).unwrap(),
            "source-preferred-codec"
        );
        assert_eq!(configuration.source_schema_generation, 9);
    }

    fn safe_transfer_payload() -> Value {
        serde_json::json!({
            "request_id": REQUEST_ID,
            "side": "source",
            "phase": "linked",
            "phase_label": "Linked",
            "next_action": null,
            "linked_request_id": TARGET_ID,
            "project_id": PROJECT_ID,
            "source_space_id": SOURCE_SPACE_ID,
            "target_space_id": TARGET_SPACE_ID,
            "source_configuration": {
                "source_rcp_version": "0.3.2",
                "source_schema_generation": 1,
                "supported_archive_codecs": ["tar-zstd-v1"],
                "machine_aliases": ["server"],
                "repositories": [{
                    "alias": "state",
                    "repository": {"identity": "example/state"},
                    "machine_alias": "server"
                }],
                "state_repository": "state",
                "project_truth_scope": ["state"],
                "default_run_truth_scope": ["state"],
                "source_manifest_sha256": "a".repeat(64)
            },
            "source_configuration_sha256": "b".repeat(64),
            "accepted_schema_generation": 1,
            "accepted_archive_codec": "tar-zstd-v1",
            "source_release_proof_sha256": "c".repeat(64),
            "target_activation_proof_sha256": "d".repeat(64),
            "archive_sha256": null,
            "archive_size_bytes": null,
            "can_link": false,
            "can_run_setup": false,
            "can_review": false,
            "can_admit": false,
            "can_accept_admission": false,
            "can_release": false,
            "can_accept_release": false,
            "can_relay": false,
            "can_restore_reentry": false,
            "can_complete": false,
            "finished": false,
            "revision": 2
        })
    }

    fn decision_record(side: &str, request_id: &str) -> ProjectTransferRecord {
        ProjectTransferRecord {
            request_id: request_id.into(),
            side: side.into(),
            phase: "linked".into(),
            phase_label: Some("Linked".into()),
            next_action: Some("Continue transfer".into()),
            linked_request_id: Some(if side == "source" {
                TARGET_ID.into()
            } else {
                REQUEST_ID.into()
            }),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            source_configuration: Some(transfer_configuration()),
            source_configuration_sha256: Some("a".repeat(64)),
            target_admission_receipt: None,
            source_release_receipt: None,
            accepted_schema_generation: Some(1),
            accepted_archive_codec: Some("tar-zstd-v1".into()),
            source_release_proof_sha256: Some("b".repeat(64)),
            target_activation_proof_sha256: Some("c".repeat(64)),
            archive_sha256: Some("d".repeat(64)),
            archive_size_bytes: Some(1),
            can_link: false,
            can_run_setup: false,
            can_review: false,
            can_admit: false,
            can_accept_admission: false,
            can_release: false,
            can_accept_release: false,
            can_relay: false,
            can_restore_reentry: false,
            can_complete: false,
            finished: false,
            revision: 1,
        }
    }

    fn transfer_configuration() -> ProjectTransferSourceConfiguration {
        ProjectTransferSourceConfiguration {
            source_rcp_version: "0.3.2".into(),
            source_schema_generation: 1,
            supported_archive_codecs: vec!["tar-zstd-v1".into()],
            machine_aliases: vec!["server".into()],
            repositories: vec![ProjectTransferRepositorySource {
                alias: "state".into(),
                repository: ProjectTransferRepositoryIdentity {
                    identity: "example/state".into(),
                },
                machine_alias: "server".into(),
            }],
            state_repository: "state".into(),
            project_truth_scope: vec!["state".into()],
            default_run_truth_scope: vec!["state".into()],
            source_manifest_sha256: "a".repeat(64),
        }
    }

    fn authorized_human(space_id: &str, user_id: &str) -> ProjectProvisioningAuthorizedHuman {
        ProjectProvisioningAuthorizedHuman {
            space_id: space_id.into(),
            user_id: user_id.into(),
            display_name: "RCP test user".into(),
        }
    }

    fn target_admission_receipt() -> ProjectTransferTargetAdmissionReceipt {
        ProjectTransferTargetAdmissionReceipt {
            source_request_id: REQUEST_ID.into(),
            target_request_id: TARGET_ID.into(),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            admitted_by: authorized_human(TARGET_SPACE_ID, "88888888-8888-4888-8888-888888888888"),
            source_configuration_sha256: "a".repeat(64),
            target_preparation_revision: 4,
            target_preparation_sha256: "b".repeat(64),
            resolved_paths: vec![ProjectTransferResolvedPath {
                repository_alias: "state".into(),
                machine_alias: "server".into(),
                path: "/srv/rcp/projects/state".into(),
            }],
            accepted_schema_generation: 1,
            accepted_archive_codec: "tar-zstd-v1".into(),
            source_release_proof_sha256: "c".repeat(64),
            target_activation_proof_sha256: "d".repeat(64),
            created_at: "2026-08-31T00:00:00Z".into(),
        }
    }

    fn source_release_receipt() -> ProjectTransferSourceReleaseReceipt {
        ProjectTransferSourceReleaseReceipt {
            source_request_id: REQUEST_ID.into(),
            target_request_id: TARGET_ID.into(),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            released_by: authorized_human(SOURCE_SPACE_ID, "99999999-9999-4999-8999-999999999999"),
            source_configuration_sha256: "a".repeat(64),
            target_admission_sha256: "e".repeat(64),
            target_preparation_revision: 4,
            target_preparation_sha256: "b".repeat(64),
            source_head: TransferGraphHead {
                target: TransferGraphTarget {
                    kind: "main".into(),
                    branch_id: None,
                },
                revision: 7,
                transition_id: Some("f".repeat(64)),
            },
            accepted_schema_generation: 1,
            accepted_archive_codec: "tar-zstd-v1".into(),
            source_release_proof_sha256: "c".repeat(64),
            target_activation_proof_sha256: "d".repeat(64),
            created_at: "2026-08-31T00:00:01Z".into(),
        }
    }

    #[test]
    fn native_advance_action_sequence_follows_backend_decisions() {
        let mut source = decision_record("source", REQUEST_ID);
        let mut target = decision_record("target", TARGET_ID);

        // Fresh final review: target admission is the only eligible action.
        target.can_admit = true;
        assert_eq!(
            next_advance_action(&source, &target, true),
            Some(ProjectTransferAdvanceAction::AdmitTarget)
        );

        // A crash after target admission resumes at the source acceptance.
        target.can_admit = false;
        source.can_accept_admission = true;
        assert_eq!(next_advance_action(&source, &target, true), None);
        assert!(!aggregate_transfer_decisions(&source, &target, true).0);
        target.target_admission_receipt = Some(target_admission_receipt());
        assert_eq!(
            next_advance_action(&source, &target, true),
            Some(ProjectTransferAdvanceAction::AcceptAdmission)
        );
        assert!(aggregate_transfer_decisions(&source, &target, true).0);

        // The source release is its own bounded transition after admission
        // acceptance; a crash here must not skip it or imply a relay.
        source.can_accept_admission = false;
        source.can_release = true;
        assert_eq!(
            next_advance_action(&source, &target, true),
            Some(ProjectTransferAdvanceAction::ReleaseSource)
        );

        // A crash after source release resumes at target receipt acceptance.
        source.can_release = false;
        target.can_accept_release = true;
        assert_eq!(
            next_advance_action(&source, &target, true),
            Some(ProjectTransferAdvanceAction::AcceptRelease)
        );

        // A restored target is explicitly re-entered before another relay.
        target.can_accept_release = false;
        target.can_restore_reentry = true;
        assert_eq!(
            next_advance_action(&source, &target, true),
            Some(ProjectTransferAdvanceAction::RestoreReentry)
        );

        target.can_restore_reentry = false;
        source.can_relay = true;
        assert_eq!(
            next_advance_action(&source, &target, true),
            Some(ProjectTransferAdvanceAction::Relay)
        );

        // A completed retry remains an explicit idempotent native action.
        source.can_relay = false;
        source.finished = true;
        target.finished = true;
        assert_eq!(
            next_advance_action(&source, &target, true),
            Some(ProjectTransferAdvanceAction::CompletedRetry)
        );
    }

    #[test]
    fn final_review_receipts_are_strict_and_round_trip_without_browser_fields() {
        let admission = target_admission_receipt();
        let admission_value = serde_json::to_value(&admission).unwrap();
        let parsed_admission = parse_target_admission_receipt(&admission_value).unwrap();
        assert_eq!(parsed_admission, admission);

        let mut admission_with_unknown = admission_value;
        admission_with_unknown["unexpected"] = Value::Bool(true);
        assert!(parse_target_admission_receipt(&admission_with_unknown).is_err());

        let release = source_release_receipt();
        let release_value = serde_json::to_value(&release).unwrap();
        let parsed_release = parse_source_release_receipt(&release_value).unwrap();
        assert_eq!(parsed_release, release);

        let mut release_with_invalid_head = release_value;
        release_with_invalid_head["source_head"]["target"]["kind"] = Value::String("branch".into());
        assert!(parse_source_release_receipt(&release_with_invalid_head).is_err());

        // These native-only receipts never become fields on the safe bundle.
        let bundle_fields = serde_json::to_value(ProjectTransferBundle {
            source: decision_record("source", REQUEST_ID)
                .to_projection()
                .unwrap(),
            target: decision_record("target", TARGET_ID)
                .to_projection()
                .unwrap(),
            incoming_provisioning: parse_project_provisioning_projection(
                &incoming_projection_payload("ready_for_review", Value::Null, Value::Null),
                TARGET_ID,
                TARGET_SPACE_ID,
            )
            .unwrap(),
            target_provider_setup: Vec::new(),
            can_advance: true,
            advance_label: Some("Continue".into()),
            can_manual_relay: false,
            finished: false,
        })
        .unwrap();
        assert!(bundle_fields.get("target_admission_receipt").is_none());
        assert!(bundle_fields.get("source_release_receipt").is_none());
    }

    #[test]
    fn advance_transition_budget_is_explicitly_bounded() {
        assert_eq!(MAX_ADVANCE_TRANSITIONS, 8);
    }

    #[test]
    fn native_advance_does_not_silently_fallback_without_a_backend_decision() {
        let mut source = decision_record("source", REQUEST_ID);
        let target = decision_record("target", TARGET_ID);
        source.can_relay = true;
        assert_eq!(next_advance_action(&source, &target, false), None);

        source.can_relay = false;
        assert_eq!(next_advance_action(&source, &target, true), None);
    }

    #[test]
    fn safe_transfer_projection_requires_every_published_decision() {
        let fields = [
            "phase_label",
            "next_action",
            "can_link",
            "can_run_setup",
            "can_review",
            "can_admit",
            "can_accept_admission",
            "can_release",
            "can_accept_release",
            "can_relay",
            "can_restore_reentry",
            "can_complete",
            "finished",
            "revision",
        ];
        for field in fields {
            let mut missing = safe_transfer_payload();
            missing.as_object_mut().unwrap().remove(field);
            assert!(
                parse_transfer_record(&missing, REQUEST_ID, "source").is_err(),
                "safe projection unexpectedly accepted missing {field}"
            );
        }
        assert!(parse_transfer_record(&safe_transfer_payload(), REQUEST_ID, "source").is_ok());
    }

    #[test]
    fn coordinator_record_round_trips_exactly_and_can_be_removed_after_recovery() {
        let directory = tempfile::tempdir().unwrap();
        let state =
            ProjectTransferCoordinatorState::new(directory.path().join(COORDINATOR_FILENAME));
        let request = prepare_request();
        let saved = state.save_or_validate(&request, TARGET_SPACE_ID).unwrap();
        assert_eq!(saved, request);
        let bytes = fs::read(directory.path().join(COORDINATOR_FILENAME)).unwrap();
        let file: ProjectTransferCoordinatorFile = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(file.records.len(), 1);
        assert_eq!(file.records[0].source_request_id, REQUEST_ID);
        assert_eq!(file.records[0].target_request_id, TARGET_ID);
        assert_eq!(file.records[0].connection_id, request.connection_id);
        assert_eq!(file.records[0].source_project_id, PROJECT_ID);
        assert_eq!(file.records[0].target_space_id, TARGET_SPACE_ID);
        assert_eq!(
            file.records[0].target_provisioning,
            request.target_provisioning
        );
        assert_eq!(
            state.load(REQUEST_ID).unwrap().unwrap().as_request(),
            request
        );

        state.remove(&request, TARGET_SPACE_ID).unwrap();
        assert!(state.load(REQUEST_ID).unwrap().is_none());
    }

    #[test]
    fn coordinator_rejects_a_conflicting_retry_for_the_same_source() {
        let directory = tempfile::tempdir().unwrap();
        let state =
            ProjectTransferCoordinatorState::new(directory.path().join(COORDINATOR_FILENAME));
        let request = prepare_request();
        state.save_or_validate(&request, TARGET_SPACE_ID).unwrap();
        let mut changed = request.clone();
        changed.target_request_id = "77777777-7777-4777-8777-777777777777".into();
        assert!(state.save_or_validate(&changed, TARGET_SPACE_ID).is_err());
        assert_eq!(
            state.load(REQUEST_ID).unwrap().unwrap().as_request(),
            request
        );
    }

    fn incoming_projection_payload(
        status: &str,
        operator_action: Value,
        final_review: Value,
    ) -> Value {
        serde_json::json!({
            "request_id": TARGET_ID,
            "kind": "incoming_transfer",
            "status": status,
            "status_label": "Target setup",
            "next_action": null,
            "can_run_setup": status == "operator_action_needed",
            "can_review": status == "ready_for_review",
            "can_cancel": false,
            "target_space_id": TARGET_SPACE_ID,
            "proposed_project_id": PROJECT_ID,
            "name": "Moved project",
            "state_repository": "state",
            "project_truth_scope": ["state"],
            "default_run_truth_scope": ["state"],
            "default_auto_research_invocation_ceiling": 3,
            "authorized_by": {
                "space_id": TARGET_SPACE_ID,
                "user_id": "88888888-8888-4888-8888-888888888888",
                "display_name": "Z"
            },
            "machines": [{
                "alias": "server", "location": "local", "host": "", "os_account": "rcp",
                "intended_central_root": "/srv/rcp/projects", "resolved_central_root": null,
                "ready": true, "status_label": "Ready"
            }],
            "repositories": [{
                "alias": "state", "repository": {"identity": "example/state"},
                "https_clone_url": "https://github.com/example/state.git",
                "ssh_clone_url": "git@github.com:example/state.git",
                "settings_url": "https://github.com/example/state/settings",
                "machine_alias": "server", "intended_path": "/srv/rcp/projects/state",
                "resolved_path": null, "checkout_disposition": null, "status": "ready",
                "status_label": "Ready", "ready": true, "commit": "a".repeat(40),
                "write_verified": true, "deploy_key_label": null,
                "public_key_fingerprint": null, "checked_at": null, "diagnostic": null
            }],
            "provider_checks": [{
                "profile": "discuss", "provider": "codex", "runtime_id": "exec",
                "model": "model", "reasoning": "medium", "machine_alias": "server",
                "status": "ready", "status_label": "Ready", "ready": true,
                "binary_path": "/usr/bin/codex", "version": "1", "resolved_runtime_id": "exec",
                "execution_account": "rcp", "checked_at": null, "diagnostic": null
            }],
            "readiness": {
                "machines_ready": 1, "machines_total": 1,
                "repositories_ready": 1, "repositories_total": 1,
                "providers_ready": 1, "providers_total": 1, "all_ready": true
            },
            "diagnostic": null,
            "operator_action": operator_action,
            "operator_argv": ["rcp", "server", "project", "provision"],
            "final_review": final_review,
            "cancellation_disposition": null,
            "revision": 4,
            "created_at": "2026-08-31T00:00:00Z",
            "updated_at": "2026-08-31T00:00:01Z",
            "setup_started_at": null,
            "completed_at": null,
            "cancelled_at": null
        })
    }

    #[test]
    fn incoming_projection_preserves_operator_action_and_final_review() {
        let operator_action = serde_json::json!({
            "number": 1,
            "title": "Install provider",
            "purpose": "Install the required provider.",
            "performed_by": "human",
            "target": {"kind": "machine", "host": "gpu0", "os_account": "rcp"},
            "phase": "provider_setup",
            "state": "operator_action_needed",
            "expected_success": "The provider is installed.",
            "message": "Run the command on the server.",
            "actions": [{"kind": "command", "argv": ["sudo", "apt", "install", "codex"]}],
            "fields": [],
            "resume_argv": ["rcp", "server", "project", "provision"]
        });
        let action = parse_project_provisioning_projection(
            &incoming_projection_payload(
                "operator_action_needed",
                operator_action.clone(),
                Value::Null,
            ),
            TARGET_ID,
            TARGET_SPACE_ID,
        )
        .unwrap();
        assert_eq!(action.operator_action, Some(operator_action));
        assert!(action.final_review.is_none());
        assert_eq!(
            action.operator_argv,
            vec!["rcp", "server", "project", "provision"]
        );

        let final_review = serde_json::json!({
            "digest": "e".repeat(64),
            "proposed_project_id": PROJECT_ID,
            "authorized_by": {
                "space_id": TARGET_SPACE_ID,
                "user_id": "88888888-8888-4888-8888-888888888888",
                "display_name": "Z"
            },
            "ready_at": "2026-08-31T00:00:02Z"
        });
        let review = parse_project_provisioning_projection(
            &incoming_projection_payload("ready_for_review", Value::Null, final_review.clone()),
            TARGET_ID,
            TARGET_SPACE_ID,
        )
        .unwrap();
        assert_eq!(review.final_review.as_ref().unwrap().digest, "e".repeat(64));
        assert_eq!(review.final_review_digest, Some("e".repeat(64)));
        assert_eq!(review.authorized_by.display_name, "Z");
    }

    #[test]
    fn cleanup_acknowledgment_is_strict_and_public_only() {
        let value = serde_json::to_value(acknowledgment()).unwrap();
        let parsed: ProjectTransferCleanupAcknowledgment = serde_json::from_value(value).unwrap();
        parsed.validate().unwrap();
        let mut extra = serde_json::to_value(parsed).unwrap();
        extra["target_activation_proof"] = Value::String("secret".into());
        assert!(serde_json::from_value::<ProjectTransferCleanupAcknowledgment>(extra).is_err());
    }

    #[test]
    fn proof_commitment_is_checked_before_source_post() {
        let source = SourceTransferRequest {
            request_id: REQUEST_ID.into(),
            phase: "archive_bound".into(),
            target_request_id: TARGET_ID.into(),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            target_activation_proof_sha256: hex_digest(&[7; 32]),
            archive_sha256: "c".repeat(64),
            archive_size_bytes: 1,
        };
        assert_eq!(hex_digest(&[7; 32]), source.target_activation_proof_sha256);
        assert_ne!(hex_digest(&[8; 32]), source.target_activation_proof_sha256);
    }

    #[test]
    fn source_phase_is_preserved_for_idempotent_relay_retries() {
        let archive = "c".repeat(64);
        let payload = |phase: &str| {
            serde_json::json!({
                "request_id": REQUEST_ID,
                "side": "source",
                "phase": phase,
                "linked_request_id": TARGET_ID,
                "project_id": PROJECT_ID,
                "source_space_id": SOURCE_SPACE_ID,
                "target_space_id": TARGET_SPACE_ID,
                "target_activation_proof_sha256": "b".repeat(64),
                "archive_sha256": archive.clone(),
                "archive_size_bytes": 17,
            })
        };
        let completed = parse_source_request(&payload("completed"), REQUEST_ID, false).unwrap();
        assert_eq!(completed.phase, "completed");
        assert_eq!(completed.archive_size_bytes, 17);
        assert!(parse_source_request(&payload("completed"), REQUEST_ID, true).is_err());
        let bound = parse_source_request(&payload("archive_bound"), REQUEST_ID, false).unwrap();
        assert_eq!(bound.phase, "archive_bound");
        assert_eq!(bound.archive_sha256, archive);
    }

    #[test]
    fn completed_source_and_target_retry_returns_metadata_without_relay_activity() {
        let source = SourceTransferRequest {
            request_id: REQUEST_ID.into(),
            phase: "completed".into(),
            target_request_id: TARGET_ID.into(),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            target_activation_proof_sha256: "b".repeat(64),
            archive_sha256: "c".repeat(64),
            archive_size_bytes: 17,
        };
        let target = TeamConnectionMetadata {
            connection_id: "66666666-6666-4666-8666-666666666666".into(),
            display_name: "Vision lab".into(),
            ssh_target: "member@lab-server".into(),
            remote_loopback_port: 8421,
            expected_space_id: TARGET_SPACE_ID.into(),
            local_origin: "https://rcp-66666666666646668666666666666666.localhost:18421".into(),
            minimum_shell_version: "0.3.2".into(),
            last_known_cards: Vec::new(),
            operator_route: None,
        };
        let result = completed_run_result(&source, &target);
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.event_count, 0);
        assert!(result.proof_verified);
        assert!(result.cleanup_acknowledged);
        assert_eq!(result.archive_sha256, source.archive_sha256);
        assert_eq!(result.archive_size_bytes, source.archive_size_bytes);
    }

    fn source_for_archive(bytes: &[u8]) -> SourceTransferRequest {
        SourceTransferRequest {
            request_id: REQUEST_ID.into(),
            phase: "archive_bound".into(),
            target_request_id: TARGET_ID.into(),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            target_activation_proof_sha256: "b".repeat(64),
            archive_sha256: hex_digest(bytes),
            archive_size_bytes: bytes.len() as u64,
        }
    }

    #[test]
    fn cancelled_manual_export_selection_returns_only_empty_metadata() {
        let value =
            serde_json::to_value(ProjectTransferExportSelectionResult::cancelled(REQUEST_ID))
                .unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "selected": false,
                "request_id": REQUEST_ID,
                "target_request_id": null,
                "target_space_id": null,
                "archive_sha256": null,
                "archive_size_bytes": null,
                "path": null
            })
        );
    }

    #[cfg(unix)]
    #[test]
    fn manual_export_cleanup_removes_only_the_exact_verified_copy() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("manual.rcp-transfer");
        let bytes = b"manual transfer archive";
        fs::write(&archive_path, bytes).unwrap();
        fs::set_permissions(&archive_path, fs::Permissions::from_mode(0o600)).unwrap();
        let source = SourceTransferRequest {
            request_id: REQUEST_ID.into(),
            phase: "archive_bound".into(),
            target_request_id: TARGET_ID.into(),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            target_activation_proof_sha256: "b".repeat(64),
            archive_sha256: hex_digest(bytes),
            archive_size_bytes: bytes.len() as u64,
        };

        remove_local_export(&archive_path, &source).unwrap();
        assert!(!archive_path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn manual_export_cleanup_leaves_a_mismatched_copy_untouched() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("manual.rcp-transfer");
        fs::write(&archive_path, b"different archive").unwrap();
        fs::set_permissions(&archive_path, fs::Permissions::from_mode(0o600)).unwrap();
        let source = SourceTransferRequest {
            request_id: REQUEST_ID.into(),
            phase: "archive_bound".into(),
            target_request_id: TARGET_ID.into(),
            project_id: PROJECT_ID.into(),
            source_space_id: SOURCE_SPACE_ID.into(),
            target_space_id: TARGET_SPACE_ID.into(),
            target_activation_proof_sha256: "b".repeat(64),
            archive_sha256: "c".repeat(64),
            archive_size_bytes: 17,
        };

        assert!(remove_local_export(&archive_path, &source).is_err());
        assert!(archive_path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn manual_export_selection_accepts_the_exact_verified_copy() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("manual.rcp-transfer");
        let bytes = b"manual transfer archive";
        fs::write(&archive_path, bytes).unwrap();
        fs::set_permissions(&archive_path, fs::Permissions::from_mode(0o600)).unwrap();
        let source = source_for_archive(bytes);

        let result = select_verified_export(&source, archive_path.clone()).unwrap();
        assert!(result.selected);
        assert_eq!(result.request_id, REQUEST_ID);
        assert_eq!(result.target_request_id.as_deref(), Some(TARGET_ID));
        assert_eq!(result.target_space_id.as_deref(), Some(TARGET_SPACE_ID));
        assert_eq!(
            result.archive_sha256.as_deref(),
            Some(source.archive_sha256.as_str())
        );
        assert_eq!(result.archive_size_bytes, Some(bytes.len() as u64));
        assert_eq!(result.path.as_deref(), archive_path.to_str());
        assert_eq!(fs::read(&archive_path).unwrap(), bytes);
    }

    #[cfg(unix)]
    #[test]
    fn manual_export_selection_rejects_wrong_digest_without_mutating_the_copy() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("manual.rcp-transfer");
        let bytes = b"manual transfer archive";
        fs::write(&archive_path, bytes).unwrap();
        fs::set_permissions(&archive_path, fs::Permissions::from_mode(0o600)).unwrap();
        let source = source_for_archive(b"different archive");

        assert!(select_verified_export(&source, archive_path.clone()).is_err());
        assert_eq!(fs::read(&archive_path).unwrap(), bytes);
        assert_eq!(
            fs::symlink_metadata(&archive_path).unwrap().len(),
            bytes.len() as u64
        );
    }

    #[cfg(unix)]
    #[test]
    fn manual_export_selection_rejects_wrong_mode_without_mutating_the_copy() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("manual.rcp-transfer");
        let bytes = b"manual transfer archive";
        fs::write(&archive_path, bytes).unwrap();
        fs::set_permissions(&archive_path, fs::Permissions::from_mode(0o644)).unwrap();
        let source = source_for_archive(bytes);

        assert!(select_verified_export(&source, archive_path.clone()).is_err());
        assert_eq!(fs::read(&archive_path).unwrap(), bytes);
        assert_eq!(
            fs::symlink_metadata(&archive_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o644
        );
    }

    #[cfg(unix)]
    #[test]
    fn manual_export_selection_rejects_a_nonfile_without_mutating_it() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("manual.rcp-transfer");
        fs::create_dir(&archive_path).unwrap();
        let source = source_for_archive(b"manual transfer archive");

        assert!(select_verified_export(&source, archive_path.clone()).is_err());
        assert!(archive_path.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn manual_export_selection_rejects_an_archive_bound_to_another_request() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("manual.rcp-transfer");
        let bytes = b"request one archive";
        fs::write(&archive_path, bytes).unwrap();
        fs::set_permissions(&archive_path, fs::Permissions::from_mode(0o600)).unwrap();
        let mut other_request = source_for_archive(b"request two archive");
        other_request.request_id = "77777777-7777-4777-8777-777777777777".into();
        other_request.target_request_id = "88888888-8888-4888-8888-888888888888".into();

        assert!(select_verified_export(&other_request, archive_path.clone()).is_err());
        assert_eq!(fs::read(&archive_path).unwrap(), bytes);
    }
}
