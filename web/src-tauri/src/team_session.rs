use std::{
    collections::HashMap,
    net::{Ipv4Addr, SocketAddr},
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use reqwest::{
    header::{
        HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, COOKIE, SET_COOKIE,
    },
    Client, Response,
};
use semver::Version;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use tauri::WebviewWindow;
use url::Url;
use uuid::{Uuid, Version as UuidVersion};
use zeroize::Zeroizing;

use crate::{
    backend::BackendState,
    lifecycle::DesktopStatus,
    local_https::{install_team_session_cookie, LocalHttpsIdentity},
    project_transfer::{
        validate_digest, ProjectTransferCleanupAcknowledgment, ProjectTransferSourceReleaseReceipt,
    },
    server_commands::ProjectProvisionReadback,
    team_connections::{CachedTeamProjectCard, TeamConnectionMetadata, TeamConnectionState},
    team_tunnel::{TeamTunnelReady, TeamTunnelState},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_SESSION_COOKIE_BYTES: usize = 4 * 1024;
const SESSION_COOKIE_PREFIX: &str = "__Host-rcp_session=";
// The installed systemd unit executes `rcp serve`; that production CLI path
// publishes `cli`, while desktop and embedded owners are never team services.
const EXPECTED_SERVER_OWNER: &str = "cli";
const TARGET_ACTIVATION_PROOF_BYTES: usize = 32;
const TRANSFER_BINARY_CONTENT_TYPE: &str = "application/octet-stream";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollTeamConnectionRequest {
    pub ssh_target: String,
    pub remote_loopback_port: u16,
    pub enrollment_code: String,
    pub member_display_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExistingTeamConnectionRequest {
    pub ssh_target: String,
    pub remote_loopback_port: u16,
    pub member_token: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TeamUserIdentity {
    pub user_id: String,
    pub display_name: Option<String>,
    pub identity_kind: String,
    pub created_at: String,
    pub updated_at: String,
    pub removal_started_at: Option<String>,
    pub removed_at: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TeamIdentity {
    pub space_id: String,
    pub space_kind: String,
    pub space_name: Option<String>,
    pub user: TeamUserIdentity,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct TeamHealth {
    status: String,
    version: String,
    space_id: String,
    space_kind: String,
    space_name: Option<String>,
    instance_id: String,
    data_dir_id: String,
    owner_kind: String,
    running_commit: Option<String>,
    web_build_id: Option<String>,
    active_agent_tasks: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnrollmentResponse {
    identity: TeamIdentity,
    token: String,
}

#[derive(Debug, Serialize)]
struct EnrollmentBody<'a> {
    code: &'a str,
    display_name: &'a str,
}

#[derive(Debug, Serialize)]
struct ExchangeBody<'a> {
    token: &'a str,
}

#[derive(Debug, Deserialize)]
struct TeamProjectCard {
    id: String,
    name: String,
    primary_question: Option<String>,
    attention_count: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct EstablishedTeamSession {
    pub connection: TeamConnectionMetadata,
    pub identity: TeamIdentity,
    pub status: DesktopStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectTransferTargetReadback {
    pub request_id: String,
    pub phase: String,
    pub linked_request_id: String,
    pub target_space_id: String,
}

pub struct TeamSessionState {
    certificate_der: Vec<u8>,
    established: Mutex<HashMap<String, EstablishedTeamSession>>,
}

impl TeamSessionState {
    pub fn new(identity: &LocalHttpsIdentity) -> Self {
        Self {
            certificate_der: identity.certificate_der().to_vec(),
            established: Mutex::new(HashMap::new()),
        }
    }

    pub fn status_for_origin(&self, origin: &Url) -> Result<Option<DesktopStatus>, String> {
        let established = self.acquire()?;
        Ok(established
            .values()
            .find(|session| {
                Url::parse(&session.connection.local_origin)
                    .is_ok_and(|candidate| candidate.origin() == origin.origin())
            })
            .map(|session| session.status.clone()))
    }

    pub fn established(&self, connection_id: &str) -> Result<EstablishedTeamSession, String> {
        self.acquire()?
            .get(connection_id)
            .cloned()
            .ok_or_else(|| "the team browser session has not been established".to_string())
    }

    pub fn forget(&self, connection_id: &str) -> Result<(), String> {
        self.acquire()?.remove(connection_id);
        Ok(())
    }

    pub async fn read_project_provisioning(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
    ) -> Result<ProjectProvisionReadback, String> {
        let connection = connections
            .list()?
            .into_iter()
            .find(|connection| connection.connection_id == connection_id)
            .ok_or_else(|| "the team connection is not saved on this desktop".to_string())?;
        let client = self.client(&connection.local_origin)?;
        let health = read_health(&client, &connection.local_origin).await?;
        validate_health(&health, Some(&connection))?;
        let token = connections.load_member_token(connection_id)?;
        let (identity, set_cookie) =
            exchange_session(&client, &connection.local_origin, &token).await?;
        validate_identity(&identity, &health)?;
        let cookie = request_cookie(&set_cookie)?;
        let header = HeaderValue::from_str(&cookie)
            .map_err(|_| "team session exchange returned an invalid cookie".to_string())?;
        let response = client
            .get(endpoint(
                &connection.local_origin,
                &format!("/api/project-provisioning/requests/{request_id}"),
            )?)
            .header(COOKIE, header)
            .send()
            .await
            .map_err(|error| {
                format!("could not read back the project provisioning request: {error}")
            })?;
        let readback: ProjectProvisionReadback =
            response_json(response, "project provisioning request readback").await?;
        if readback.request_id != request_id
            || readback.target_space_id != connection.expected_space_id
            || !matches!(
                readback.status.as_str(),
                "waiting_for_server_setup"
                    | "setup_in_progress"
                    | "operator_action_needed"
                    | "ready_for_review"
                    | "completed"
                    | "cancelled"
            )
        {
            return Err("the project provisioning readback does not match this team space".into());
        }
        Ok(readback)
    }

    /// Read only the public target transfer lifecycle through a Rust-owned
    /// authenticated session. This is used to make native relay retries
    /// idempotent without exposing any proof or archive bytes to Web code.
    pub(crate) async fn read_project_transfer_value(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
    ) -> Result<Value, String> {
        validate_uuid4(request_id, "target transfer request identity")?;
        let (connection, client, cookie) = self
            .authenticated_request_context(connections, connection_id)
            .await?;
        let response = client
            .get(endpoint(
                &connection.local_origin,
                &format!("/api/project-transfers/requests/{request_id}"),
            )?)
            .header(COOKIE, cookie)
            .send()
            .await
            .map_err(|error| format!("could not read the target transfer request: {error}"))?;
        response_json(response, "target transfer request readback").await
    }

    /// Admit the prepared target request using the currently authenticated
    /// member. The actor is resolved by the server session; the browser sends
    /// neither an actor identity nor an admission receipt.
    pub(crate) async fn admit_target_project_transfer(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
    ) -> Result<Value, String> {
        validate_uuid4(request_id, "target transfer request identity")?;
        let (connection, client, cookie) = self
            .authenticated_request_context(connections, connection_id)
            .await?;
        let response = client
            .post(endpoint(
                &connection.local_origin,
                &format!("/api/project-transfers/target-requests/{request_id}/admit"),
            )?)
            .header(COOKIE, cookie)
            .send()
            .await
            .map_err(|error| format!("could not admit the target transfer request: {error}"))?;
        response_json(response, "target transfer admission").await
    }

    /// Carry the public source release receipt to the target backend. This is
    /// deliberately separate from the native proof endpoint used later by the
    /// archive relay.
    pub(crate) async fn accept_source_project_transfer_release(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
        receipt: &ProjectTransferSourceReleaseReceipt,
    ) -> Result<Value, String> {
        validate_uuid4(request_id, "target transfer request identity")?;
        let (connection, client, cookie) = self
            .authenticated_request_context(connections, connection_id)
            .await?;
        let body = TransferReceiptBody { receipt };
        let response = client
            .post(endpoint(
                &connection.local_origin,
                &format!("/api/project-transfers/target-requests/{request_id}/source-release"),
            )?)
            .header(COOKIE, cookie)
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("could not accept the source transfer release: {error}"))?;
        response_json(response, "source transfer release acceptance").await
    }

    /// Re-enter a restored target transfer using the exact durable revision
    /// and final-review digest shown by the target projection. The actor is
    /// inferred by the authenticated target session.
    pub(crate) async fn restore_target_project_transfer_reentry(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
        expected_restored_revision: u64,
        expected_final_review_digest: &str,
    ) -> Result<Value, String> {
        validate_uuid4(request_id, "target transfer request identity")?;
        validate_digest(expected_final_review_digest, "target final-review digest")?;
        let (connection, client, cookie) = self
            .authenticated_request_context(connections, connection_id)
            .await?;
        let body = RestoreReentryBody {
            expected_restored_revision,
            expected_resume_phase: "archive_bound",
            expected_final_review_digest,
        };
        let response = client
            .post(endpoint(
                &connection.local_origin,
                &format!("/api/project-transfers/target-requests/{request_id}/restore-reentry"),
            )?)
            .header(COOKIE, cookie)
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("could not re-enter the restored target transfer: {error}"))?;
        response_json(response, "restored target transfer re-entry").await
    }

    pub async fn read_project_transfer(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
    ) -> Result<ProjectTransferTargetReadback, String> {
        let (connection, _session) = self.saved_established(connections, connection_id)?;
        let value = self
            .read_project_transfer_value(connections, connection_id, request_id)
            .await?;
        parse_target_transfer_readback(&value, request_id, &connection.expected_space_id)
    }

    /// Create or validate the target's incoming-transfer provisioning request.
    /// The body is assembled by the native coordinator from strict public
    /// input; this method only posts the one fixed endpoint through the saved
    /// established team session.
    pub(crate) async fn create_incoming_transfer_provisioning(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        body: &Value,
    ) -> Result<Value, String> {
        let request_id = body
            .get("request_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "incoming provisioning intent has no request id".to_string())?;
        validate_uuid4(request_id, "target transfer request identity")?;
        let (connection, client, cookie) = self
            .authenticated_request_context(connections, connection_id)
            .await?;
        let response = client
            .post(endpoint(
                &connection.local_origin,
                "/api/project-transfers/incoming-provisioning-requests",
            )?)
            .header(COOKIE, cookie)
            .json(body)
            .send()
            .await
            .map_err(|error| {
                format!("could not create the incoming transfer provisioning request: {error}")
            })?;
        response_json(response, "incoming transfer provisioning request creation").await
    }

    /// Create or validate the target-side transfer request from the source's
    /// public configuration and commitments.
    pub(crate) async fn create_target_project_transfer(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        body: &Value,
    ) -> Result<Value, String> {
        let request_id = body
            .get("provisioning_request_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "target transfer intent has no request id".to_string())?;
        validate_uuid4(request_id, "target transfer request identity")?;
        let (connection, client, cookie) = self
            .authenticated_request_context(connections, connection_id)
            .await?;
        let response = client
            .post(endpoint(
                &connection.local_origin,
                "/api/project-transfers/target-requests",
            )?)
            .header(COOKIE, cookie)
            .json(body)
            .send()
            .await
            .map_err(|error| format!("could not create the target transfer request: {error}"))?;
        response_json(response, "target transfer request creation").await
    }

    /// Read the target provisioning lifecycle from the incoming-transfer
    /// endpoint. This must remain distinct from the create-team path because
    /// incoming requests have different ownership and cancellation semantics.
    pub(crate) async fn read_incoming_transfer_provisioning_value(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
    ) -> Result<Value, String> {
        validate_uuid4(request_id, "target transfer request identity")?;
        let (connection, client, cookie) = self
            .authenticated_request_context(connections, connection_id)
            .await?;
        let response = client
            .get(endpoint(
                &connection.local_origin,
                &format!("/api/project-transfers/incoming-provisioning-requests/{request_id}"),
            )?)
            .header(COOKIE, cookie)
            .send()
            .await
            .map_err(|error| {
                format!("could not read the incoming transfer provisioning request: {error}")
            })?;
        response_json(response, "incoming transfer provisioning readback").await
    }

    /// Read target-side provider/setup choices from the fixed public registry
    /// endpoint. Authentication remains Rust-owned even though the response is
    /// public readiness information.
    pub(crate) async fn read_target_provider_setup_value(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
    ) -> Result<Value, String> {
        let (connection, client, cookie) = self
            .authenticated_request_context(connections, connection_id)
            .await?;
        let response = client
            .get(endpoint(&connection.local_origin, "/api/providers")?)
            .header(COOKIE, cookie)
            .send()
            .await
            .map_err(|error| format!("could not read target provider setup choices: {error}"))?;
        response_json(response, "target provider setup choices").await
    }

    /// Retrieve the target's raw activation proof through the pinned native
    /// route. The proof remains in Rust-owned zeroizing storage; callers must
    /// post it directly to the pinned personal backend without serializing it
    /// into a Web or Tauri value.
    pub async fn retrieve_target_activation_proof(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
    ) -> Result<Zeroizing<Vec<u8>>, String> {
        validate_uuid4(request_id, "target transfer request identity")?;
        let (connection, session) = self.saved_established(connections, connection_id)?;
        let (client, _health) = self.native_client(&connection, &session).await?;
        let token = connections.load_member_token(connection_id)?;
        let authorization = bearer_header(&token)?;
        let response = client
            .get(endpoint(
                &connection.local_origin,
                &format!(
                    "/api/native/project-transfers/target-requests/{request_id}/activation-proof"
                ),
            )?)
            .header(AUTHORIZATION, authorization)
            .send()
            .await
            .map_err(|error| format!("could not retrieve the target activation proof: {error}"))?;
        if !response.status().is_success() {
            return Err(format!(
                "the target activation proof was rejected (HTTP {})",
                response.status().as_u16()
            ));
        }
        validate_binary_response_headers(&response)?;
        let mut proof = Vec::with_capacity(TARGET_ACTIVATION_PROOF_BYTES);
        let mut response = response;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("the target activation proof stream failed: {error}"))?
        {
            if proof.len().saturating_add(chunk.len()) > TARGET_ACTIVATION_PROOF_BYTES {
                return Err("the target activation proof exceeded its bounded size".into());
            }
            proof.extend_from_slice(&chunk);
        }
        if proof.len() != TARGET_ACTIVATION_PROOF_BYTES {
            return Err("the target activation proof has an invalid size".into());
        }
        Ok(Zeroizing::new(proof))
    }

    /// Acknowledge cleanup using only the typed public receipt returned by the
    /// personal source endpoint. The response body is intentionally discarded
    /// so no target-side record can become a second IPC payload.
    pub async fn post_cleanup_acknowledgment(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
        request_id: &str,
        acknowledgment: &ProjectTransferCleanupAcknowledgment,
    ) -> Result<(), String> {
        validate_uuid4(request_id, "target transfer request identity")?;
        acknowledgment.validate()?;
        if acknowledgment.target_request_id != request_id {
            return Err("the cleanup acknowledgment does not match the target request".into());
        }
        let (connection, session) = self.saved_established(connections, connection_id)?;
        let (client, _health) = self.native_client(&connection, &session).await?;
        let token = connections.load_member_token(connection_id)?;
        let authorization = bearer_header(&token)?;
        let body = CleanupAcknowledgmentBody { acknowledgment };
        let path = format!(
            "/api/native/project-transfers/target-requests/{request_id}/cleanup-acknowledgment"
        );
        let response = client
            .post(endpoint(&connection.local_origin, &path)?)
            .header(AUTHORIZATION, authorization)
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                format!("could not post the target transfer cleanup acknowledgment: {error}")
            })?;
        if !response.status().is_success() {
            return Err(format!(
                "the target transfer cleanup acknowledgment was rejected (HTTP {})",
                response.status().as_u16()
            ));
        }
        Ok(())
    }

    pub async fn enroll(
        &self,
        window: &WebviewWindow,
        connections: &TeamConnectionState,
        tunnels: &TeamTunnelState,
        lifecycle: &BackendState,
        request: EnrollTeamConnectionRequest,
    ) -> Result<EstablishedTeamSession, String> {
        let enrollment_code = Zeroizing::new(request.enrollment_code);
        let connection_id = Uuid::new_v4().to_string();
        let ready = tunnels
            .connect_candidate(
                lifecycle,
                &connection_id,
                &request.ssh_target,
                request.remote_loopback_port,
            )
            .await?;
        let result = self
            .enroll_over_tunnel(
                window,
                connections,
                &ready,
                request.ssh_target,
                request.remote_loopback_port,
                &enrollment_code,
                request.member_display_name,
            )
            .await;
        if result.is_err() && !is_saved(connections, &connection_id) {
            stop_candidate_after_failure(tunnels, &connection_id).await;
        }
        result
    }

    pub async fn add_existing(
        &self,
        window: &WebviewWindow,
        connections: &TeamConnectionState,
        tunnels: &TeamTunnelState,
        lifecycle: &BackendState,
        request: ExistingTeamConnectionRequest,
    ) -> Result<EstablishedTeamSession, String> {
        let token = Zeroizing::new(request.member_token);
        let connection_id = Uuid::new_v4().to_string();
        let ready = tunnels
            .connect_candidate(
                lifecycle,
                &connection_id,
                &request.ssh_target,
                request.remote_loopback_port,
            )
            .await?;
        let result = self
            .add_existing_over_tunnel(
                window,
                connections,
                &ready,
                request.ssh_target,
                request.remote_loopback_port,
                token,
            )
            .await;
        if result.is_err() && !is_saved(connections, &connection_id) {
            stop_candidate_after_failure(tunnels, &connection_id).await;
        }
        result
    }

    pub async fn reconnect(
        &self,
        window: &WebviewWindow,
        connections: &TeamConnectionState,
        tunnels: &TeamTunnelState,
        lifecycle: &BackendState,
        connection_id: &str,
    ) -> Result<EstablishedTeamSession, String> {
        let mut connection = connections
            .list()?
            .into_iter()
            .find(|connection| connection.connection_id == connection_id)
            .ok_or_else(|| "the team connection is not saved on this desktop".to_string())?;
        let ready = tunnels
            .connect_saved(connections, lifecycle, connection_id)
            .await?;
        let client = self.client(&ready.local_origin)?;
        let health = read_health(&client, &ready.local_origin).await?;
        validate_health(&health, Some(&connection))?;
        let token = connections.load_member_token(connection_id)?;
        let (identity, cookie) =
            exchange_session(&client, &connection.local_origin, &token).await?;
        validate_identity(&identity, &health)?;
        connection.last_known_cards =
            read_project_cards(&client, &ready.local_origin, &cookie).await?;
        connections.save_metadata(connection.clone())?;
        install_team_session_cookie(window, &ready.local_origin, cookie).await?;
        self.record(connection, identity, &health)
    }

    /// Establish the saved member session for a native-only operation. This
    /// deliberately does not touch the WebView cookie or browser navigation;
    /// the native request methods exchange the saved member token again for
    /// each authenticated request. It lets a resumed transfer recover after
    /// a desktop restart while keeping the browser session optional.
    pub(crate) async fn ensure_native_session(
        &self,
        connections: &TeamConnectionState,
        tunnels: &TeamTunnelState,
        lifecycle: &BackendState,
        connection_id: &str,
    ) -> Result<EstablishedTeamSession, String> {
        if let Ok(existing) = self.established(connection_id) {
            return Ok(existing);
        }
        let connection = connections
            .list()?
            .into_iter()
            .find(|connection| connection.connection_id == connection_id)
            .ok_or_else(|| "the team connection is not saved on this desktop".to_string())?;
        let ready = tunnels
            .connect_saved(connections, lifecycle, connection_id)
            .await?;
        let client = self.client(&ready.local_origin)?;
        let health = read_health(&client, &ready.local_origin).await?;
        validate_health(&health, Some(&connection))?;
        let token = connections.load_member_token(connection_id)?;
        let (identity, _cookie) = exchange_session(&client, &ready.local_origin, &token).await?;
        validate_identity(&identity, &health)?;
        self.record(connection, identity, &health)
    }

    #[allow(clippy::too_many_arguments)]
    async fn enroll_over_tunnel(
        &self,
        window: &WebviewWindow,
        connections: &TeamConnectionState,
        ready: &TeamTunnelReady,
        ssh_target: String,
        remote_loopback_port: u16,
        enrollment_code: &str,
        member_display_name: String,
    ) -> Result<EstablishedTeamSession, String> {
        let client = self.client(&ready.local_origin)?;
        let health = read_health(&client, &ready.local_origin).await?;
        validate_health(&health, None)?;
        let connection = new_connection(ready, ssh_target, remote_loopback_port, &health);
        connections.save_metadata(connection.clone())?;
        let enrolled: EnrollmentResponse = match post_json(
            &client,
            &ready.local_origin,
            "/api/team/enroll",
            &EnrollmentBody {
                code: enrollment_code,
                display_name: &member_display_name,
            },
            "team enrollment",
        )
        .await
        {
            Ok(enrolled) => enrolled,
            Err(error) => {
                let _ = connections.remove_metadata(&ready.connection_id);
                return Err(error);
            }
        };
        let token = Zeroizing::new(enrolled.token);
        if let Err(error) = connections.store_member_token(&ready.connection_id, token.to_string())
        {
            let _ = connections.remove_metadata(&ready.connection_id);
            return Err(error);
        }
        validate_identity(&enrolled.identity, &health)?;
        self.finish_saved_connection(window, connections, connection, client, health, token)
            .await
    }

    async fn add_existing_over_tunnel(
        &self,
        window: &WebviewWindow,
        connections: &TeamConnectionState,
        ready: &TeamTunnelReady,
        ssh_target: String,
        remote_loopback_port: u16,
        token: Zeroizing<String>,
    ) -> Result<EstablishedTeamSession, String> {
        let client = self.client(&ready.local_origin)?;
        let health = read_health(&client, &ready.local_origin).await?;
        validate_health(&health, None)?;
        let connection = new_connection(ready, ssh_target, remote_loopback_port, &health);
        connections.save_metadata(connection.clone())?;
        if let Err(error) = connections.store_member_token(&ready.connection_id, token.to_string())
        {
            let _ = connections.remove_metadata(&ready.connection_id);
            return Err(error);
        }
        self.finish_saved_connection(window, connections, connection, client, health, token)
            .await
    }

    async fn finish_saved_connection(
        &self,
        window: &WebviewWindow,
        connections: &TeamConnectionState,
        mut connection: TeamConnectionMetadata,
        client: Client,
        health: TeamHealth,
        token: Zeroizing<String>,
    ) -> Result<EstablishedTeamSession, String> {
        let (identity, cookie) =
            exchange_session(&client, &connection.local_origin, &token).await?;
        validate_identity(&identity, &health)?;
        connection.last_known_cards =
            read_project_cards(&client, &connection.local_origin, &cookie).await?;
        connections.save_metadata(connection.clone())?;
        install_team_session_cookie(window, &connection.local_origin, cookie).await?;
        self.record(connection, identity, &health)
    }

    fn record(
        &self,
        connection: TeamConnectionMetadata,
        identity: TeamIdentity,
        health: &TeamHealth,
    ) -> Result<EstablishedTeamSession, String> {
        let established = EstablishedTeamSession {
            status: desktop_status(&connection.local_origin, health),
            connection,
            identity,
        };
        self.acquire()?.insert(
            established.connection.connection_id.clone(),
            established.clone(),
        );
        Ok(established)
    }

    fn client(&self, origin: &str) -> Result<Client, String> {
        build_client(origin, &self.certificate_der)
    }

    fn saved_established(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
    ) -> Result<(TeamConnectionMetadata, EstablishedTeamSession), String> {
        validate_uuid4(connection_id, "team connection identity")?;
        let connection = connections
            .list()?
            .into_iter()
            .find(|connection| connection.connection_id == connection_id)
            .ok_or_else(|| "the team connection is not saved on this desktop".to_string())?;
        let session = self.established(connection_id)?;
        if session.connection.connection_id != connection.connection_id
            || session.connection.expected_space_id != connection.expected_space_id
            || session.connection.local_origin != connection.local_origin
            || session.status.base_url != connection.local_origin
        {
            return Err(
                "the established team session is not pinned to its saved connection".into(),
            );
        }
        Ok((connection, session))
    }

    async fn native_client(
        &self,
        connection: &TeamConnectionMetadata,
        session: &EstablishedTeamSession,
    ) -> Result<(Client, TeamHealth), String> {
        let client = self.client(&connection.local_origin)?;
        let health = read_health(&client, &connection.local_origin).await?;
        validate_health(&health, Some(connection))?;
        if health.instance_id != session.status.instance_id
            || health.data_dir_id != session.status.data_dir_id
            || health.version != session.status.version
            || health.owner_kind != session.status.owner_kind
        {
            return Err("the team server identity changed; explicit reconnect is required".into());
        }
        Ok((client, health))
    }

    async fn authenticated_request_context(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
    ) -> Result<(TeamConnectionMetadata, Client, HeaderValue), String> {
        let (connection, session) = self.saved_established(connections, connection_id)?;
        let (client, health) = self.native_client(&connection, &session).await?;
        let token = connections.load_member_token(connection_id)?;
        let (identity, set_cookie) =
            exchange_session(&client, &connection.local_origin, &token).await?;
        validate_identity(&identity, &health)?;
        let cookie = request_cookie(&set_cookie)?;
        let header = HeaderValue::from_str(&cookie)
            .map_err(|_| "team session exchange returned an invalid cookie".to_string())?;
        Ok((connection, client, header))
    }

    fn acquire(&self) -> Result<MutexGuard<'_, HashMap<String, EstablishedTeamSession>>, String> {
        self.established
            .lock()
            .map_err(|_| "the established team session state is unavailable".to_string())
    }
}

#[derive(Debug, Serialize)]
struct CleanupAcknowledgmentBody<'a> {
    acknowledgment: &'a ProjectTransferCleanupAcknowledgment,
}

#[derive(Debug, Serialize)]
struct TransferReceiptBody<'a, T> {
    receipt: &'a T,
}

#[derive(Debug, Serialize)]
struct RestoreReentryBody<'a> {
    expected_restored_revision: u64,
    expected_resume_phase: &'static str,
    expected_final_review_digest: &'a str,
}

fn bearer_header(token: &str) -> Result<HeaderValue, String> {
    HeaderValue::from_str(&format!("Bearer {token}"))
        .map_err(|_| "the saved team member credential is invalid".to_string())
}

fn validate_binary_response_headers(response: &Response) -> Result<(), String> {
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or_default().trim());
    if content_type != Some(TRANSFER_BINARY_CONTENT_TYPE) {
        return Err("the target activation proof returned an invalid content type".into());
    }
    if let Some(length) = response.headers().get(CONTENT_LENGTH) {
        let length = length
            .to_str()
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| "the target activation proof returned an invalid size".to_string())?;
        if length != TARGET_ACTIVATION_PROOF_BYTES {
            return Err("the target activation proof returned an invalid size".into());
        }
    }
    Ok(())
}

fn new_connection(
    ready: &TeamTunnelReady,
    ssh_target: String,
    remote_loopback_port: u16,
    health: &TeamHealth,
) -> TeamConnectionMetadata {
    TeamConnectionMetadata {
        connection_id: ready.connection_id.clone(),
        display_name: health
            .space_name
            .clone()
            .expect("validated team health must carry a space name"),
        ssh_target,
        remote_loopback_port,
        expected_space_id: health.space_id.clone(),
        local_origin: ready.local_origin.clone(),
        minimum_shell_version: health.version.clone(),
        last_known_cards: Vec::new(),
        operator_route: None,
    }
}

fn build_client(origin: &str, certificate_der: &[u8]) -> Result<Client, String> {
    let origin = Url::parse(origin).map_err(|_| "the local team origin is invalid".to_string())?;
    if origin.scheme() != "https" || origin.port().is_none() {
        return Err("the local team origin is not an explicit HTTPS origin".into());
    }
    let host = origin
        .host_str()
        .ok_or_else(|| "the local team origin has no hostname".to_string())?;
    let certificate = reqwest::Certificate::from_der(certificate_der)
        .map_err(|_| "the desktop local HTTPS certificate is invalid".to_string())?;
    Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .https_only(true)
        .no_proxy()
        .tls_built_in_root_certs(false)
        .add_root_certificate(certificate)
        .redirect(reqwest::redirect::Policy::none())
        // The pinned desktop certificate has a *.localhost SAN, while each
        // connection uses an immediate rcp-<id>.localhost host. rustls does not
        // apply wildcard matching to localhost, so the exact private root is
        // the authentication boundary and hostname matching is disabled here.
        .danger_accept_invalid_hostnames(true)
        .resolve(
            host,
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), origin.port().unwrap()),
        )
        .build()
        .map_err(|error| format!("could not create the team connection client: {error}"))
}

async fn read_health(client: &Client, origin: &str) -> Result<TeamHealth, String> {
    let response = client
        .get(endpoint(origin, "/api/health")?)
        .send()
        .await
        .map_err(|error| format!("could not reach the team server health endpoint: {error}"))?;
    response_json(response, "team server health").await
}

async fn exchange_session(
    client: &Client,
    origin: &str,
    token: &str,
) -> Result<(TeamIdentity, Zeroizing<String>), String> {
    let response = client
        .post(endpoint(origin, "/api/team/session/exchange")?)
        .json(&ExchangeBody { token })
        .send()
        .await
        .map_err(|error| format!("could not reach team session exchange: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "team session exchange was rejected (HTTP {})",
            response.status().as_u16()
        ));
    }
    let cookie = session_cookie(response.headers())?;
    let identity = response
        .json::<TeamIdentity>()
        .await
        .map_err(|_| "team session exchange returned an invalid identity".to_string())?;
    Ok((identity, cookie))
}

async fn read_project_cards(
    client: &Client,
    origin: &str,
    set_cookie: &str,
) -> Result<Vec<CachedTeamProjectCard>, String> {
    let cookie = request_cookie(set_cookie)?;
    let header = HeaderValue::from_str(&cookie)
        .map_err(|_| "team session exchange returned an invalid cookie".to_string())?;
    let response = client
        .get(endpoint(origin, "/api/projects")?)
        .header(COOKIE, header)
        .send()
        .await
        .map_err(|error| format!("could not read team projects: {error}"))?;
    let cards: Vec<TeamProjectCard> = response_json(response, "team project index").await?;
    Ok(cards
        .into_iter()
        .map(|card| CachedTeamProjectCard {
            id: card.id,
            name: card.name,
            primary_question: card.primary_question,
            attention_count: card.attention_count,
        })
        .collect())
}

async fn post_json<T: DeserializeOwned, B: Serialize + ?Sized>(
    client: &Client,
    origin: &str,
    path: &str,
    body: &B,
    description: &str,
) -> Result<T, String> {
    let response = client
        .post(endpoint(origin, path)?)
        .json(body)
        .send()
        .await
        .map_err(|error| format!("could not reach {description}: {error}"))?;
    response_json(response, description).await
}

async fn response_json<T: DeserializeOwned>(
    response: Response,
    description: &str,
) -> Result<T, String> {
    if !response.status().is_success() {
        return Err(format!(
            "{description} was rejected (HTTP {})",
            response.status().as_u16()
        ));
    }
    response
        .json::<T>()
        .await
        .map_err(|_| format!("{description} returned an invalid response"))
}

fn parse_target_transfer_readback(
    value: &Value,
    expected_request_id: &str,
    expected_space_id: &str,
) -> Result<ProjectTransferTargetReadback, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "the target transfer request returned an invalid response".to_string())?;
    let text = |field: &str| {
        object
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("the target transfer request has no valid {field}"))
    };
    let request_id = text("request_id")?.to_string();
    if request_id != expected_request_id {
        return Err("the target transfer request identity does not match the requested id".into());
    }
    validate_uuid4(&request_id, "target transfer request identity")?;
    if text("side")? != "target" {
        return Err("the target transfer request is not a target request".into());
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
        return Err("the target transfer request has an invalid durable phase".into());
    }
    let linked_request_id = text("linked_request_id")?.to_string();
    validate_uuid4(&linked_request_id, "source transfer request identity")?;
    let target_space_id = text("target_space_id")?.to_string();
    validate_uuid4(&target_space_id, "target transfer space identity")?;
    if target_space_id != expected_space_id {
        return Err("the target transfer request belongs to another team space".into());
    }
    Ok(ProjectTransferTargetReadback {
        request_id,
        phase: phase.to_string(),
        linked_request_id,
        target_space_id,
    })
}

fn endpoint(origin: &str, path: &str) -> Result<Url, String> {
    let mut url = Url::parse(origin).map_err(|_| "the local team origin is invalid".to_string())?;
    url.set_path(path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

fn validate_health(
    health: &TeamHealth,
    expected: Option<&TeamConnectionMetadata>,
) -> Result<(), String> {
    if health.status != "ok"
        || health.space_kind != "team"
        || health.owner_kind != EXPECTED_SERVER_OWNER
    {
        return Err("the SSH destination is not an installed RCP team service".into());
    }
    validate_uuid4(&health.space_id, "team space identity")?;
    validate_uuid4(&health.instance_id, "team server instance identity")?;
    if health.data_dir_id.len() != 64 || !is_lower_hex(&health.data_dir_id) {
        return Err("the team server data directory identity is invalid".into());
    }
    if health.space_name.as_deref().is_none_or(str::is_empty) {
        return Err("the team server has no display name".into());
    }
    let server_version = canonical_version(&health.version, "team server version")?;
    let shell_version = canonical_version(env!("CARGO_PKG_VERSION"), "desktop shell version")?;
    if server_version > shell_version {
        return Err(format!(
            "this team server requires RCP desktop {} or newer",
            health.version
        ));
    }
    match (&health.running_commit, &health.web_build_id) {
        (Some(commit), Some(build))
            if commit.len() == 40
                && is_lower_hex(commit)
                && build
                    .strip_prefix("sha256:")
                    .is_some_and(|digest| digest.len() == 64 && is_lower_hex(digest)) => {}
        _ => return Err("the team server does not report an installed source build".into()),
    }
    if let Some(expected) = expected {
        if health.space_id != expected.expected_space_id {
            return Err(
                "the saved team space identity changed; explicit reconnect is required".into(),
            );
        }
        let minimum = canonical_version(
            &expected.minimum_shell_version,
            "saved minimum desktop shell version",
        )?;
        if shell_version < minimum {
            return Err(format!(
                "this team connection requires RCP desktop {} or newer",
                expected.minimum_shell_version
            ));
        }
    }
    Ok(())
}

fn validate_identity(identity: &TeamIdentity, health: &TeamHealth) -> Result<(), String> {
    if identity.space_id != health.space_id
        || identity.space_kind != "team"
        || identity.space_name != health.space_name
        || identity.user.identity_kind != "team_member"
        || identity.user.removal_started_at.is_some()
        || identity.user.removed_at.is_some()
    {
        return Err("the team member identity does not match the verified team service".into());
    }
    validate_uuid4(&identity.user.user_id, "team member identity")?;
    Ok(())
}

fn session_cookie(headers: &HeaderMap) -> Result<Zeroizing<String>, String> {
    let values = headers.get_all(SET_COOKIE).iter().collect::<Vec<_>>();
    if values.len() != 1 {
        return Err("team session exchange did not return exactly one session cookie".into());
    }
    let value = values[0]
        .to_str()
        .map_err(|_| "team session exchange returned an invalid cookie".to_string())?;
    if value.len() > MAX_SESSION_COOKIE_BYTES
        || !value.starts_with(SESSION_COOKIE_PREFIX)
        || value.contains(['\r', '\n', ','])
    {
        return Err("team session exchange returned an invalid session cookie".into());
    }
    let attributes = value.split(';').skip(1).map(str::trim).collect::<Vec<_>>();
    let has_secure = attributes
        .iter()
        .any(|attribute| attribute.eq_ignore_ascii_case("Secure"));
    let has_http_only = attributes
        .iter()
        .any(|attribute| attribute.eq_ignore_ascii_case("HttpOnly"));
    let has_root_path = attributes
        .iter()
        .any(|attribute| attribute.eq_ignore_ascii_case("Path=/"));
    if !has_secure || !has_http_only || !has_root_path {
        return Err("team session cookie is missing required browser protections".into());
    }
    Ok(Zeroizing::new(value.to_string()))
}

fn request_cookie(set_cookie: &str) -> Result<Zeroizing<String>, String> {
    let value = set_cookie
        .split(';')
        .next()
        .filter(|value| {
            value.starts_with(SESSION_COOKIE_PREFIX) && value.len() > SESSION_COOKIE_PREFIX.len()
        })
        .ok_or_else(|| "team session exchange returned an invalid session cookie".to_string())?;
    Ok(Zeroizing::new(value.to_string()))
}

fn desktop_status(origin: &str, health: &TeamHealth) -> DesktopStatus {
    DesktopStatus {
        desktop: true,
        version: health.version.clone(),
        base_url: origin.to_string(),
        instance_id: health.instance_id.clone(),
        data_dir_id: health.data_dir_id.clone(),
        owner_kind: health.owner_kind.clone(),
        active_agent_tasks: health.active_agent_tasks,
        owned: false,
    }
}

fn validate_uuid4(value: &str, label: &str) -> Result<(), String> {
    let parsed = Uuid::parse_str(value).map_err(|_| format!("{label} is invalid"))?;
    if parsed.get_version() != Some(UuidVersion::Random) || parsed.to_string() != value {
        return Err(format!("{label} is invalid"));
    }
    Ok(())
}

fn canonical_version(value: &str, label: &str) -> Result<Version, String> {
    let version = Version::parse(value).map_err(|_| format!("{label} is invalid"))?;
    if version.to_string() != value {
        return Err(format!("{label} is invalid"));
    }
    Ok(version)
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn is_saved(connections: &TeamConnectionState, connection_id: &str) -> bool {
    connections.list().is_ok_and(|connections| {
        connections
            .iter()
            .any(|connection| connection.connection_id == connection_id)
    })
}

async fn stop_candidate_after_failure(tunnels: &TeamTunnelState, connection_id: &str) {
    if let Err(error) = tunnels.stop_candidate(connection_id).await {
        eprintln!("[rcp] could not stop a failed pending team tunnel: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    fn health() -> TeamHealth {
        TeamHealth {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            space_id: "11111111-1111-4111-8111-111111111111".into(),
            space_kind: "team".into(),
            space_name: Some("Vision lab".into()),
            instance_id: "22222222-2222-4222-8222-222222222222".into(),
            data_dir_id: "3".repeat(64),
            owner_kind: EXPECTED_SERVER_OWNER.into(),
            running_commit: Some("4".repeat(40)),
            web_build_id: Some(format!("sha256:{}", "5".repeat(64))),
            active_agent_tasks: 2,
        }
    }

    #[test]
    fn installed_team_health_is_strictly_bound() {
        assert!(validate_health(&health(), None).is_ok());
        for mut invalid in [
            {
                let mut value = health();
                value.space_kind = "personal".into();
                value
            },
            {
                let mut value = health();
                value.owner_kind = "desktop".into();
                value
            },
            {
                let mut value = health();
                value.running_commit = None;
                value
            },
            {
                let mut value = health();
                value.web_build_id = Some("NOT-HEX".into());
                value
            },
        ] {
            assert!(validate_health(&invalid, None).is_err());
            invalid.status.clear();
        }
    }

    #[test]
    fn saved_space_identity_and_shell_floor_are_enforced() {
        let verified = health();
        let mut connection = TeamConnectionMetadata {
            connection_id: "66666666-6666-4666-8666-666666666666".into(),
            display_name: "Vision lab".into(),
            ssh_target: "rcp@lab".into(),
            remote_loopback_port: 8421,
            expected_space_id: verified.space_id.clone(),
            local_origin: "https://rcp-66666666666646668666666666666666.localhost:18421".into(),
            minimum_shell_version: env!("CARGO_PKG_VERSION").into(),
            last_known_cards: Vec::new(),
            operator_route: None,
        };
        assert!(validate_health(&verified, Some(&connection)).is_ok());
        connection.expected_space_id = "77777777-7777-4777-8777-777777777777".into();
        assert!(validate_health(&verified, Some(&connection)).is_err());
        connection.expected_space_id = verified.space_id.clone();
        connection.minimum_shell_version = "999.0.0".into();
        assert!(validate_health(&verified, Some(&connection)).is_err());
    }

    #[test]
    fn session_cookie_requires_host_cookie_browser_protections() {
        let mut headers = HeaderMap::new();
        headers.insert(
            SET_COOKIE,
            HeaderValue::from_static(
                "__Host-rcp_session=rcp_session_abcdefghijklmnopqrstuvwxyz0123456789abcdefg; Path=/; Max-Age=1209600; Secure; HttpOnly; SameSite=lax",
            ),
        );
        let cookie = session_cookie(&headers).unwrap();
        assert!(cookie.starts_with(SESSION_COOKIE_PREFIX));

        for invalid in [
            "rcp_session=value; Path=/; Secure; HttpOnly",
            "__Host-rcp_session=value; Path=/; HttpOnly",
            "__Host-rcp_session=value; Path=/; Secure",
            "__Host-rcp_session=value; Path=/wrong; Secure; HttpOnly",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(SET_COOKIE, HeaderValue::from_str(invalid).unwrap());
            assert!(session_cookie(&headers).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn identity_must_match_the_verified_team() {
        let verified = health();
        let mut identity = TeamIdentity {
            space_id: verified.space_id.clone(),
            space_kind: "team".into(),
            space_name: verified.space_name.clone(),
            user: TeamUserIdentity {
                user_id: "88888888-8888-4888-8888-888888888888".into(),
                display_name: Some("Alice".into()),
                identity_kind: "team_member".into(),
                created_at: "2026-08-30T00:00:00Z".into(),
                updated_at: "2026-08-30T00:00:00Z".into(),
                removal_started_at: None,
                removed_at: None,
            },
        };
        assert!(validate_identity(&identity, &verified).is_ok());
        identity.space_id = "99999999-9999-4999-8999-999999999999".into();
        assert!(validate_identity(&identity, &verified).is_err());
    }

    #[test]
    fn target_transfer_readback_is_bound_to_request_and_saved_space() {
        let request_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
        let source_request_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
        let space_id = health().space_id;
        let value = serde_json::json!({
            "request_id": request_id,
            "side": "target",
            "phase": "target_activated",
            "linked_request_id": source_request_id,
            "target_space_id": space_id.clone(),
        });
        let readback = parse_target_transfer_readback(&value, request_id, &space_id).unwrap();
        assert_eq!(readback.phase, "target_activated");
        assert!(parse_target_transfer_readback(&value, source_request_id, &space_id).is_err());

        let mut wrong_space = value;
        wrong_space["target_space_id"] =
            Value::String("cccccccc-cccc-4ccc-8ccc-cccccccccccc".into());
        assert!(parse_target_transfer_readback(&wrong_space, request_id, &space_id).is_err());
    }

    #[test]
    fn restore_reentry_body_binds_the_fixed_archive_boundary() {
        let digest = "a".repeat(64);
        let body = RestoreReentryBody {
            expected_restored_revision: 9,
            expected_resume_phase: "archive_bound",
            expected_final_review_digest: &digest,
        };
        assert_eq!(
            serde_json::to_value(body).unwrap(),
            serde_json::json!({
                "expected_restored_revision": 9,
                "expected_resume_phase": "archive_bound",
                "expected_final_review_digest": "a".repeat(64),
            })
        );
    }
}
