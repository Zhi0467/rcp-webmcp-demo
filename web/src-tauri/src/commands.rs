use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::Serialize;
use serde_json::Value;
use tauri::{ipc::Channel, AppHandle, Emitter, State, WebviewWindow};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;
use url::Url;

use crate::{
    backend::{self, BackendState},
    dictation,
    lifecycle::DesktopStatus,
    navigation,
    project_transfer::{
        self, ProjectTransferAdvanceResult, ProjectTransferBundle, ProjectTransferCoordinatorState,
        ProjectTransferExportCleanupResult, ProjectTransferExportResult,
        ProjectTransferExportSelectionResult, ProjectTransferFinishResult,
        ProjectTransferPrepareRequest, ProjectTransferRunResult, TargetProviderSetupProjection,
    },
    server_commands::{
        self, ConfigureServerOperatorRouteRequest, ServerCommandRunResult, ServerOperatorProbe,
        TerminalLaunchResult,
    },
    team_connections::{RemovalResult, TeamConnectionMetadata, TeamConnectionState},
    team_session::{
        EnrollTeamConnectionRequest, EstablishedTeamSession, ExistingTeamConnectionRequest,
        TeamSessionState,
    },
    team_tunnel::{TeamTunnelReady, TeamTunnelState},
    updates, windows,
};

const ARTIFACT_AVAILABILITY_TIMEOUT: Duration = Duration::from_secs(5);
const REPOSITORY_PREVIEW_AVAILABILITY_TIMEOUT: Duration = Duration::from_secs(35);

#[derive(Serialize)]
pub struct ShowResult {
    shown: bool,
}

#[derive(Serialize)]
pub struct OpenResult {
    opened: bool,
}

#[derive(Serialize)]
pub struct DownloadResult {
    saved: bool,
    path: Option<String>,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct FolderSelectionResult {
    selected: bool,
    path: Option<String>,
}

#[derive(Serialize)]
pub struct QuitResult {
    quitting: bool,
}

#[derive(Serialize)]
pub struct ApplyUpdateResult {
    started: bool,
}

#[tauri::command]
pub fn desktop_list_team_connections(
    state: State<'_, TeamConnectionState>,
) -> Result<Vec<TeamConnectionMetadata>, String> {
    state.list()
}

#[tauri::command]
pub fn desktop_configure_server_operator_route(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    lifecycle: State<'_, BackendState>,
    request: ConfigureServerOperatorRouteRequest,
) -> Result<TeamConnectionMetadata, String> {
    let saved = connections.list()?;
    let caller = window
        .url()
        .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?;
    authorize_connection_repair_origin(
        &caller,
        &lifecycle.status()?.base_url,
        &saved,
        &request.connection_id,
        cfg!(debug_assertions),
    )?;
    connections.set_operator_route(&request.connection_id, request.route)
}

#[tauri::command]
pub async fn desktop_probe_server_operator(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
) -> Result<ServerOperatorProbe, String> {
    let saved = connections.list()?;
    authorize_team_tunnel_origin(
        &window
            .url()
            .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?,
        &lifecycle.status()?.base_url,
        &saved,
        &connection_id,
        cfg!(debug_assertions),
    )?;
    let connection = saved_connection(&saved, &connection_id)?;
    server_commands::probe(connection, PathBuf::from("/usr/bin/ssh")).await
}

#[tauri::command]
pub async fn desktop_run_project_provision(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
    request_id: String,
    on_event: Channel<Value>,
) -> Result<ServerCommandRunResult, String> {
    let saved = connections.list()?;
    authorize_team_tunnel_origin(
        &window
            .url()
            .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?,
        &lifecycle.status()?.base_url,
        &saved,
        &connection_id,
        cfg!(debug_assertions),
    )?;
    sessions.established(&connection_id)?;
    let connection = saved_connection(&saved, &connection_id)?;
    let (exit_code, event_count) = server_commands::run_project_provision(
        connection,
        &request_id,
        &on_event,
        PathBuf::from("/usr/bin/ssh"),
    )
    .await?;
    let readback = sessions
        .read_project_provisioning(&connections, &connection_id, &request_id)
        .await?;
    Ok(ServerCommandRunResult {
        connection_id,
        request_id,
        exit_code,
        event_count,
        readback,
    })
}

#[tauri::command]
pub async fn desktop_open_project_provision_terminal(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
    request_id: String,
) -> Result<TerminalLaunchResult, String> {
    let saved = connections.list()?;
    authorize_team_tunnel_origin(
        &window
            .url()
            .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?,
        &lifecycle.status()?.base_url,
        &saved,
        &connection_id,
        cfg!(debug_assertions),
    )?;
    let argv =
        server_commands::terminal_argv(saved_connection(&saved, &connection_id)?, &request_id)?;
    server_commands::open_terminal(argv).await
}

#[tauri::command]
pub async fn desktop_run_project_transfer(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    lifecycle: State<'_, BackendState>,
    request_id: String,
    on_event: Channel<Value>,
) -> Result<ProjectTransferRunResult, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::run(
        &lifecycle,
        &connections,
        &sessions,
        &tunnels,
        &request_id,
        &on_event,
        PathBuf::from("/usr/bin/ssh"),
    )
    .await
}

#[tauri::command]
pub async fn desktop_prepare_project_transfer(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    coordinator: State<'_, ProjectTransferCoordinatorState>,
    lifecycle: State<'_, BackendState>,
    request: ProjectTransferPrepareRequest,
) -> Result<ProjectTransferBundle, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::prepare(&lifecycle, &connections, &sessions, &coordinator, request).await
}

#[tauri::command]
pub async fn desktop_load_project_transfer(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    coordinator: State<'_, ProjectTransferCoordinatorState>,
    lifecycle: State<'_, BackendState>,
    source_request_id: String,
) -> Result<ProjectTransferBundle, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::load_bundle(
        &lifecycle,
        &connections,
        &sessions,
        &tunnels,
        &coordinator,
        &source_request_id,
    )
    .await
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn desktop_advance_project_transfer(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    coordinator: State<'_, ProjectTransferCoordinatorState>,
    lifecycle: State<'_, BackendState>,
    source_request_id: String,
    on_event: Channel<Value>,
) -> Result<ProjectTransferAdvanceResult, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::advance(
        &lifecycle,
        &connections,
        &sessions,
        &tunnels,
        &coordinator,
        &source_request_id,
        &on_event,
        PathBuf::from("/usr/bin/ssh"),
    )
    .await
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn desktop_run_incoming_project_provision(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    coordinator: State<'_, ProjectTransferCoordinatorState>,
    lifecycle: State<'_, BackendState>,
    source_request_id: String,
    on_event: Channel<Value>,
) -> Result<ServerCommandRunResult, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::run_incoming_provision(
        &lifecycle,
        &connections,
        &sessions,
        &tunnels,
        &coordinator,
        &source_request_id,
        &on_event,
        PathBuf::from("/usr/bin/ssh"),
    )
    .await
}

#[tauri::command]
pub async fn desktop_read_target_project_provisioning_options(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
) -> Result<Vec<TargetProviderSetupProjection>, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::read_target_provider_setup(&connections, &sessions, &connection_id).await
}

#[tauri::command]
pub async fn desktop_export_project_transfer(
    app: AppHandle,
    window: WebviewWindow,
    lifecycle: State<'_, BackendState>,
    request_id: String,
) -> Result<ProjectTransferExportResult, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    server_commands::validate_uuid4(&request_id, "project transfer request identity")?;
    let Some(chosen) = app
        .dialog()
        .file()
        .set_title("Save RCP project transfer archive")
        .set_file_name(format!("{request_id}.rcp-transfer"))
        .blocking_save_file()
    else {
        return Ok(ProjectTransferExportResult::cancelled(&request_id));
    };
    let destination = chosen
        .into_path()
        .map_err(|error| format!("selected destination is not a local file: {error}"))?;
    project_transfer::export(&lifecycle, &request_id, destination).await
}

#[tauri::command]
pub async fn desktop_select_project_transfer_export(
    app: AppHandle,
    window: WebviewWindow,
    lifecycle: State<'_, BackendState>,
    request_id: String,
) -> Result<ProjectTransferExportSelectionResult, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    server_commands::validate_uuid4(&request_id, "project transfer request identity")?;
    let Some(chosen) = app
        .dialog()
        .file()
        .set_title("Select RCP project transfer archive")
        .blocking_pick_file()
    else {
        return Ok(ProjectTransferExportSelectionResult::cancelled(&request_id));
    };
    let archive_path = chosen
        .into_path()
        .map_err(|error| format!("selected archive is not a local file: {error}"))?;
    project_transfer::select_export(&lifecycle, &request_id, archive_path).await
}

#[tauri::command]
pub async fn desktop_open_project_transfer_terminal(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    lifecycle: State<'_, BackendState>,
    request_id: String,
    archive_path: String,
) -> Result<TerminalLaunchResult, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::terminal(
        &lifecycle,
        &connections,
        &sessions,
        &tunnels,
        &request_id,
        PathBuf::from(archive_path),
    )
    .await
}

#[tauri::command]
pub async fn desktop_finish_project_transfer(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    lifecycle: State<'_, BackendState>,
    request_id: String,
    archive_path: String,
) -> Result<ProjectTransferFinishResult, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::finish(
        &lifecycle,
        &connections,
        &sessions,
        &tunnels,
        &request_id,
        PathBuf::from(archive_path),
    )
    .await
}

#[tauri::command]
pub async fn desktop_discard_project_transfer_export(
    window: WebviewWindow,
    lifecycle: State<'_, BackendState>,
    request_id: String,
    archive_path: String,
) -> Result<ProjectTransferExportCleanupResult, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    project_transfer::discard_export(&lifecycle, &request_id, PathBuf::from(archive_path)).await
}

#[tauri::command]
pub async fn desktop_remove_team_connection_metadata(
    window: WebviewWindow,
    state: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
) -> Result<RemovalResult, String> {
    let saved = state.list()?;
    authorize_connection_repair_origin(
        &window
            .url()
            .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?,
        &lifecycle.status()?.base_url,
        &saved,
        &connection_id,
        cfg!(debug_assertions),
    )?;
    let result = tunnels
        .remove_saved_connection(&state, &connection_id)
        .await?;
    sessions.forget(&connection_id)?;
    Ok(result)
}

#[tauri::command]
pub fn desktop_remove_team_member_token(
    window: WebviewWindow,
    state: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
) -> Result<RemovalResult, String> {
    let saved = state.list()?;
    authorize_connection_repair_origin(
        &window
            .url()
            .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?,
        &lifecycle.status()?.base_url,
        &saved,
        &connection_id,
        cfg!(debug_assertions),
    )?;
    let result = state.remove_member_token(&connection_id)?;
    sessions.forget(&connection_id)?;
    Ok(result)
}

#[tauri::command]
pub async fn desktop_enroll_team_connection(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    lifecycle: State<'_, BackendState>,
    request: EnrollTeamConnectionRequest,
) -> Result<EstablishedTeamSession, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    sessions
        .enroll(&window, &connections, &tunnels, &lifecycle, request)
        .await
}

#[tauri::command]
pub async fn desktop_add_existing_team_connection(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    lifecycle: State<'_, BackendState>,
    request: ExistingTeamConnectionRequest,
) -> Result<EstablishedTeamSession, String> {
    authorize_personal_origin(&window, &lifecycle)?;
    sessions
        .add_existing(&window, &connections, &tunnels, &lifecycle, request)
        .await
}

#[tauri::command]
pub async fn desktop_establish_team_session(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
) -> Result<EstablishedTeamSession, String> {
    let saved = connections.list()?;
    authorize_team_tunnel_origin(
        &window
            .url()
            .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?,
        &lifecycle.status()?.base_url,
        &saved,
        &connection_id,
        cfg!(debug_assertions),
    )?;
    sessions
        .reconnect(&window, &connections, &tunnels, &lifecycle, &connection_id)
        .await
}

#[tauri::command]
pub fn desktop_navigate_team(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
    project_id: Option<String>,
) -> Result<(), String> {
    let saved = connections.list()?;
    authorize_team_tunnel_origin(
        &window
            .url()
            .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?,
        &lifecycle.status()?.base_url,
        &saved,
        &connection_id,
        cfg!(debug_assertions),
    )?;
    let established = sessions.established(&connection_id)?;
    if let Some(project_id) = &project_id {
        if !established
            .connection
            .last_known_cards
            .iter()
            .any(|card| card.id == *project_id)
        {
            return Err("the selected project is not in this verified team connection".into());
        }
    }
    let fragment = project_id
        .as_deref()
        .map(|project_id| format!("/projects/{project_id}"));
    windows::navigate_main_route(
        &window,
        &established.connection.local_origin,
        fragment.as_deref(),
    )
}

#[tauri::command]
pub fn desktop_return_to_personal(
    window: WebviewWindow,
    lifecycle: State<'_, BackendState>,
) -> Result<(), String> {
    let target = windows::personal_root(&lifecycle.status()?.base_url)?;
    windows::navigate_main(&window, target.as_str())
}

#[tauri::command]
pub async fn desktop_connect_team_tunnel(
    window: WebviewWindow,
    connections: State<'_, TeamConnectionState>,
    tunnels: State<'_, TeamTunnelState>,
    lifecycle: State<'_, BackendState>,
    connection_id: String,
) -> Result<TeamTunnelReady, String> {
    let saved = connections.list()?;
    let caller = window
        .url()
        .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?;
    let personal_base_url = lifecycle.status()?.base_url;
    authorize_team_tunnel_origin(
        &caller,
        &personal_base_url,
        &saved,
        &connection_id,
        cfg!(debug_assertions),
    )?;
    tunnels
        .connect_saved(&connections, &lifecycle, &connection_id)
        .await
}

fn authorize_team_tunnel_origin(
    caller: &Url,
    personal_base_url: &str,
    connections: &[TeamConnectionMetadata],
    requested_connection_id: &str,
    allow_dev: bool,
) -> Result<(), String> {
    if !connections
        .iter()
        .any(|connection| connection.connection_id == requested_connection_id)
    {
        return Err("the team connection is not saved on this desktop".into());
    }
    let personal = Url::parse(personal_base_url)
        .map_err(|_| "the personal RCP origin is invalid".to_string())?;
    if same_origin(caller, &personal)
        || (allow_dev
            && caller.scheme() == "http"
            && caller.host_str() == Some("127.0.0.1")
            && caller.port_or_known_default() == Some(5173))
    {
        return Ok(());
    }
    let caller_connection = connections.iter().find(|connection| {
        Url::parse(&connection.local_origin).is_ok_and(|origin| same_origin(caller, &origin))
    });
    if caller_connection
        .is_some_and(|connection| connection.connection_id == requested_connection_id)
    {
        Ok(())
    } else {
        Err("this desktop origin cannot connect the requested team space".into())
    }
}

fn authorize_connection_repair_origin(
    caller: &Url,
    personal_base_url: &str,
    connections: &[TeamConnectionMetadata],
    requested_connection_id: &str,
    allow_dev: bool,
) -> Result<(), String> {
    if is_personal_origin(caller, personal_base_url, allow_dev)? {
        return Ok(());
    }
    if connections.iter().any(|connection| {
        connection.connection_id == requested_connection_id
            && Url::parse(&connection.local_origin).is_ok_and(|origin| same_origin(caller, &origin))
    }) {
        Ok(())
    } else {
        Err("this desktop origin cannot repair the requested team connection".into())
    }
}

fn is_personal_origin(
    caller: &Url,
    personal_base_url: &str,
    allow_dev: bool,
) -> Result<bool, String> {
    let personal = Url::parse(personal_base_url)
        .map_err(|_| "the personal RCP origin is invalid".to_string())?;
    Ok(same_origin(caller, &personal)
        || (allow_dev
            && caller.scheme() == "http"
            && caller.host_str() == Some("127.0.0.1")
            && caller.port_or_known_default() == Some(5173)))
}

fn saved_connection_for_origin<'a>(
    caller: &Url,
    connections: &'a [TeamConnectionMetadata],
) -> Option<&'a TeamConnectionMetadata> {
    connections.iter().find(|connection| {
        Url::parse(&connection.local_origin).is_ok_and(|origin| same_origin(caller, &origin))
    })
}

fn saved_connection<'a>(
    connections: &'a [TeamConnectionMetadata],
    connection_id: &str,
) -> Result<&'a TeamConnectionMetadata, String> {
    connections
        .iter()
        .find(|connection| connection.connection_id == connection_id)
        .ok_or_else(|| "the team connection is not saved on this desktop".to_string())
}

fn same_origin(left: &Url, right: &Url) -> bool {
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn authorize_personal_origin(
    window: &WebviewWindow,
    lifecycle: &BackendState,
) -> Result<(), String> {
    let caller = window
        .url()
        .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?;
    if is_personal_origin(
        &caller,
        &lifecycle.status()?.base_url,
        cfg!(debug_assertions),
    )? {
        Ok(())
    } else {
        Err("team spaces can be added only from the personal RCP index".into())
    }
}

#[tauri::command]
pub fn desktop_start_dictation(app: AppHandle, session_id: String) -> Result<(), String> {
    dictation::start(&app, &session_id)
}

#[tauri::command]
pub fn desktop_stop_dictation(session_id: String) -> Result<(), String> {
    dictation::stop(&session_id)
}

#[tauri::command]
pub async fn desktop_status(
    window: WebviewWindow,
    state: State<'_, BackendState>,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
) -> Result<DesktopStatus, String> {
    let current = window
        .url()
        .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?;
    let mut status = state.status()?;
    if !is_personal_origin(&current, &status.base_url, cfg!(debug_assertions))? {
        if let Some(team_status) = sessions.status_for_origin(&current)? {
            return Ok(team_status);
        }
        let saved = connections.list()?;
        return if saved_connection_for_origin(&current, &saved).is_some() {
            Err("the displayed team space has no verified browser session".into())
        } else {
            Err("the displayed desktop origin is not a saved RCP space".into())
        };
    }
    if let Ok(health) = backend::health(&status).await {
        if status.matches_health(&health) {
            state.update_health(&health);
            status.active_agent_tasks = health.active_agent_tasks;
            status.owner_kind = health.owner_kind;
        }
    }
    Ok(status)
}

#[tauri::command]
pub async fn desktop_reconnect_backend(
    app: AppHandle,
    window: WebviewWindow,
    state: State<'_, BackendState>,
    connections: State<'_, TeamConnectionState>,
    sessions: State<'_, TeamSessionState>,
    tunnels: State<'_, TeamTunnelState>,
) -> Result<DesktopStatus, String> {
    let current = window
        .url()
        .map_err(|error| format!("cannot inspect the current desktop origin: {error}"))?;
    let personal_status = state.status()?;
    if is_personal_origin(&current, &personal_status.base_url, cfg!(debug_assertions))? {
        return backend::connect(&app, &state, "Leave it running").await;
    }
    let saved = connections.list()?;
    if let Some(connection) = saved_connection_for_origin(&current, &saved) {
        return Ok(sessions
            .reconnect(
                &window,
                &connections,
                &tunnels,
                &state,
                &connection.connection_id,
            )
            .await?
            .status);
    }
    Err("the displayed desktop origin is not a saved RCP space".into())
}

#[tauri::command]
pub async fn desktop_show_ready(
    app: AppHandle,
    state: State<'_, BackendState>,
) -> Result<ShowResult, String> {
    let status = state.status()?;
    let health = backend::health(&status).await?;
    if !status.matches_health(&health) {
        let message = "backend identity changed before the desktop window was shown";
        app.emit_to(
            "main",
            "rcp://backend-mismatch",
            serde_json::json!({"message": message}),
        )
        .map_err(|error| error.to_string())?;
        return Err(message.into());
    }
    state.update_health(&health);
    windows::show_main(&app)?;
    Ok(ShowResult { shown: true })
}

#[tauri::command]
pub async fn choose_repository_folder(app: AppHandle) -> Result<FolderSelectionResult, String> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Choose repository folder")
        .pick_folder(move |folder| {
            let _ = sender.send(folder);
        });
    let selected = receiver
        .await
        .map_err(|_| "repository folder dialog closed unexpectedly".to_string())?
        .map(|folder| {
            folder
                .into_path()
                .map_err(|error| format!("selected repository is not a local folder: {error}"))
        })
        .transpose()?;
    folder_selection_result(selected)
}

fn folder_selection_result(path: Option<PathBuf>) -> Result<FolderSelectionResult, String> {
    let Some(path) = path else {
        return Ok(FolderSelectionResult {
            selected: false,
            path: None,
        });
    };
    if !path.is_absolute() {
        return Err("selected repository folder is not an absolute path".into());
    }
    let path = path
        .to_str()
        .ok_or_else(|| "selected repository folder path is not valid UTF-8".to_string())?;
    Ok(FolderSelectionResult {
        selected: true,
        path: Some(path.to_string()),
    })
}

#[tauri::command]
pub async fn open_artifact_preview(
    app: AppHandle,
    state: State<'_, BackendState>,
    project_id: String,
    task_id: String,
    artifact_id: String,
) -> Result<OpenResult, String> {
    let status = state.status()?;
    let url = artifact_url(
        &status.base_url,
        &project_id,
        &task_id,
        &artifact_id,
        "viewer",
    )?;
    ensure_available(&url, "artifact", ARTIFACT_AVAILABILITY_TIMEOUT).await?;
    backend::reverify_identity(&state, &status).await?;
    windows::open_preview(&app, url, status.base_url)?;
    Ok(OpenResult { opened: true })
}

#[tauri::command]
pub async fn open_episode_report_preview(
    app: AppHandle,
    state: State<'_, BackendState>,
    project_id: String,
    episode_id: String,
) -> Result<OpenResult, String> {
    let status = state.status()?;
    let url = episode_report_preview_url(&status.base_url, &project_id, &episode_id)?;
    if !navigation::is_loopback_rcp_url(&url, &status.base_url, false) {
        return Err("episode report preview URL is outside the RCP backend".into());
    }
    backend::reverify_identity(&state, &status).await?;
    windows::open_preview(&app, url, status.base_url)?;
    Ok(OpenResult { opened: true })
}

#[tauri::command]
pub async fn open_repository_file_preview(
    app: AppHandle,
    state: State<'_, BackendState>,
    project_id: String,
    path: String,
    line: Option<u64>,
) -> Result<OpenResult, String> {
    let status = state.status()?;
    let url = repository_file_preview_url(&status.base_url, &project_id, &path, line)?;
    ensure_available(
        &url,
        "repository file",
        REPOSITORY_PREVIEW_AVAILABILITY_TIMEOUT,
    )
    .await?;
    backend::reverify_identity(&state, &status).await?;
    windows::open_preview(&app, url, status.base_url)?;
    Ok(OpenResult { opened: true })
}

#[tauri::command]
pub async fn download_artifact(
    app: AppHandle,
    state: State<'_, BackendState>,
    project_id: String,
    task_id: String,
    artifact_id: String,
    suggested_name: String,
) -> Result<DownloadResult, String> {
    let status = state.status()?;
    let url = artifact_url(
        &status.base_url,
        &project_id,
        &task_id,
        &artifact_id,
        "download",
    )?;
    let name = safe_filename(&suggested_name);
    let Some(chosen) = app
        .dialog()
        .file()
        .set_title("Save RCP artifact")
        .set_file_name(name)
        .blocking_save_file()
    else {
        return Ok(DownloadResult {
            saved: false,
            path: None,
        });
    };
    let path = chosen
        .into_path()
        .map_err(|error| format!("selected destination is not a local file: {error}"))?;
    backend::reverify_identity(&state, &status).await?;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|error| error.to_string())?
        .get(url)
        .send()
        .await
        .map_err(|error| format!("artifact download failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("artifact download failed: {error}"))?;
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("artifact download was interrupted: {error}"))?;
    let parent = path
        .parent()
        .ok_or_else(|| "selected destination has no parent directory".to_string())?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".rcp-download-")
        .tempfile_in(parent)
        .map_err(|error| format!("cannot create a temporary download file: {error}"))?;
    temporary
        .write_all(&bytes)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| format!("cannot save artifact: {error}"))?;
    temporary
        .persist(&path)
        .map_err(|error| format!("cannot finish artifact download: {}", error.error))?;
    Ok(DownloadResult {
        saved: true,
        path: Some(path.display().to_string()),
    })
}

#[tauri::command]
pub fn open_external(app: AppHandle, url: String) -> Result<OpenResult, String> {
    let url = Url::parse(&url).map_err(|error| format!("invalid reference URL: {error}"))?;
    if !navigation::is_external_reference(&url) {
        return Err("only HTTP or HTTPS references may open externally".into());
    }
    app.opener()
        .open_url(url.as_str(), None::<&str>)
        .map_err(|error| format!("could not open system browser: {error}"))?;
    Ok(OpenResult { opened: true })
}

#[tauri::command]
pub fn request_quit(app: AppHandle) -> Result<QuitResult, String> {
    let quitting = crate::request_app_quit(app);
    Ok(QuitResult { quitting })
}

#[tauri::command]
pub async fn check_for_update(
    app: AppHandle,
    state: State<'_, BackendState>,
) -> Result<updates::UpdateStatus, String> {
    updates::check(&app, &state).await
}

#[tauri::command]
pub async fn apply_update(
    app: AppHandle,
    state: State<'_, BackendState>,
    tunnels: State<'_, TeamTunnelState>,
    confirm_active_work: bool,
) -> Result<ApplyUpdateResult, String> {
    updates::apply(&app, &state, &tunnels, confirm_active_work).await?;
    Ok(ApplyUpdateResult { started: true })
}

fn artifact_url(
    base_url: &str,
    project_id: &str,
    task_id: &str,
    artifact_id: &str,
    action: &str,
) -> Result<Url, String> {
    let mut url = Url::parse(base_url).map_err(|error| format!("invalid backend URL: {error}"))?;
    url.path_segments_mut()
        .map_err(|_| "backend URL cannot contain path segments".to_string())?
        .extend([
            "api",
            "projects",
            project_id,
            "tasks",
            task_id,
            "artifacts",
            artifact_id,
            action,
        ]);
    Ok(url)
}

fn episode_report_preview_url(
    base_url: &str,
    project_id: &str,
    episode_id: &str,
) -> Result<Url, String> {
    let mut url = Url::parse(base_url).map_err(|error| format!("invalid backend URL: {error}"))?;
    url.path_segments_mut()
        .map_err(|_| "backend URL cannot contain path segments".to_string())?
        .extend([
            "api", "projects", project_id, "episodes", episode_id, "report", "viewer",
        ]);
    Ok(url)
}

fn repository_file_preview_url(
    base_url: &str,
    project_id: &str,
    path: &str,
    line: Option<u64>,
) -> Result<Url, String> {
    validate_repository_path(path)?;
    if line == Some(0) {
        return Err("repository file line must be a positive integer".into());
    }

    let mut url = Url::parse(base_url).map_err(|error| format!("invalid backend URL: {error}"))?;
    url.path_segments_mut()
        .map_err(|_| "backend URL cannot contain path segments".to_string())?
        .extend([
            "api",
            "projects",
            project_id,
            "repositories",
            "files",
            "preview",
        ]);
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("path", path);
        if let Some(line) = line {
            query.append_pair("line", &line.to_string());
        }
    }
    Ok(url)
}

fn validate_repository_path(path: &str) -> Result<(), String> {
    if !path.starts_with('/')
        || path.contains('\\')
        || path.contains('\0')
        || path
            .split('/')
            .skip(1)
            .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
    {
        return Err(
            "repository file path must be an absolute POSIX path without empty or dot segments"
                .into(),
        );
    }
    Ok(())
}

async fn ensure_available(url: &Url, description: &str, timeout: Duration) -> Result<(), String> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|error| error.to_string())?
        .head(url.clone())
        .send()
        .await
        .map_err(|error| format!("{description} is unavailable: {error}"))?
        .error_for_status()
        .map_err(|error| format!("{description} is unavailable: {error}"))?;
    Ok(())
}

fn safe_filename(suggested: &str) -> &str {
    Path::new(suggested)
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("artifact")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn saved_connection(connection_id: &str, port: u16) -> TeamConnectionMetadata {
        TeamConnectionMetadata {
            connection_id: connection_id.into(),
            display_name: "Lab".into(),
            ssh_target: "rcp@lab-server".into(),
            remote_loopback_port: 8421,
            expected_space_id: "33333333-3333-4333-8333-333333333333".into(),
            local_origin: format!(
                "https://rcp-{}.localhost:{port}",
                connection_id.replace('-', "")
            ),
            minimum_shell_version: "0.3.2".into(),
            last_known_cards: Vec::new(),
            operator_route: None,
        }
    }

    #[test]
    fn tunnel_command_is_limited_to_personal_or_the_same_team_origin() {
        let first = saved_connection("11111111-1111-4111-8111-111111111111", 18421);
        let mut second = saved_connection("22222222-2222-4222-8222-222222222222", 19421);
        second.expected_space_id = "44444444-4444-4444-8444-444444444444".into();
        let saved = vec![first.clone(), second.clone()];

        for personal in [
            "http://127.0.0.1:8421/#/projects/a",
            "http://127.0.0.1:5173/#/projects/a",
        ] {
            assert!(authorize_team_tunnel_origin(
                &Url::parse(personal).unwrap(),
                "http://127.0.0.1:8421",
                &saved,
                &second.connection_id,
                true,
            )
            .is_ok());
        }
        assert!(authorize_team_tunnel_origin(
            &Url::parse(&format!("{}/#/projects/a", first.local_origin)).unwrap(),
            "http://127.0.0.1:8421",
            &saved,
            &first.connection_id,
            false,
        )
        .is_ok());
        assert!(authorize_team_tunnel_origin(
            &Url::parse(&first.local_origin).unwrap(),
            "http://127.0.0.1:8421",
            &saved,
            &second.connection_id,
            false,
        )
        .is_err());
        assert!(authorize_team_tunnel_origin(
            &Url::parse("https://example.com").unwrap(),
            "http://127.0.0.1:8421",
            &saved,
            &first.connection_id,
            false,
        )
        .is_err());
    }

    #[test]
    fn connection_repair_is_idempotent_from_personal_and_space_bound_from_team() {
        let first = saved_connection("11111111-1111-4111-8111-111111111111", 18421);
        let second = saved_connection("22222222-2222-4222-8222-222222222222", 19421);
        let saved = vec![first.clone(), second.clone()];

        assert!(authorize_connection_repair_origin(
            &Url::parse("http://127.0.0.1:8421").unwrap(),
            "http://127.0.0.1:8421",
            &[],
            "33333333-3333-4333-8333-333333333333",
            false,
        )
        .is_ok());
        assert!(authorize_connection_repair_origin(
            &Url::parse(&first.local_origin).unwrap(),
            "http://127.0.0.1:8421",
            &saved,
            &first.connection_id,
            false,
        )
        .is_ok());
        assert!(authorize_connection_repair_origin(
            &Url::parse(&first.local_origin).unwrap(),
            "http://127.0.0.1:8421",
            &saved,
            &second.connection_id,
            false,
        )
        .is_err());
    }

    #[test]
    fn folder_selection_result_preserves_cancel_and_path() {
        assert_eq!(
            folder_selection_result(None).unwrap(),
            FolderSelectionResult {
                selected: false,
                path: None,
            }
        );
        assert_eq!(
            folder_selection_result(Some(PathBuf::from("/Users/example/research project")))
                .unwrap(),
            FolderSelectionResult {
                selected: true,
                path: Some("/Users/example/research project".into()),
            }
        );
        assert!(folder_selection_result(Some(PathBuf::from("relative/repository"))).is_err());
    }

    #[test]
    fn repository_preview_url_encodes_identifiers_path_and_optional_line() {
        let url = repository_file_preview_url(
            "http://127.0.0.1:8421",
            "project id",
            "/Users/example/origin repo/src/a file.rs",
            Some(27),
        )
        .unwrap();

        assert_eq!(
            url.as_str(),
            "http://127.0.0.1:8421/api/projects/project%20id/repositories/files/preview?path=%2FUsers%2Fexample%2Forigin+repo%2Fsrc%2Fa+file.rs&line=27"
        );
    }

    #[test]
    fn episode_report_preview_url_is_same_origin_and_encodes_identifiers() {
        let base = "http://127.0.0.1:8421";
        let url = episode_report_preview_url(base, "project id", "episode/id").unwrap();

        assert_eq!(
            url.as_str(),
            "http://127.0.0.1:8421/api/projects/project%20id/episodes/episode%2Fid/report/viewer"
        );
        assert!(navigation::is_loopback_rcp_url(&url, base, false));
    }

    #[test]
    fn repository_preview_url_omits_absent_line() {
        let url = repository_file_preview_url(
            "http://127.0.0.1:8421",
            "project",
            "/Users/example/repo/README.md",
            None,
        )
        .unwrap();

        assert_eq!(
            url.query(),
            Some("path=%2FUsers%2Fexample%2Frepo%2FREADME.md")
        );
    }

    #[test]
    fn repository_preview_rejects_unsafe_paths_and_zero_line() {
        for path in [
            "",
            "relative/path",
            "src\\main.rs",
            "/Users/example/repo/bad\0name",
            "/",
            "/Users/example/repo/./main.rs",
            "/Users/example/repo/../main.rs",
            "/Users/example/repo//main.rs",
            "/Users/example/repo/",
        ] {
            assert!(
                repository_file_preview_url("http://127.0.0.1:8421", "project", path, None,)
                    .is_err(),
                "accepted unsafe path {path:?}"
            );
        }
        assert!(repository_file_preview_url(
            "http://127.0.0.1:8421",
            "project",
            "/Users/example/repo/src/main.rs",
            Some(0),
        )
        .is_err());
    }
}
