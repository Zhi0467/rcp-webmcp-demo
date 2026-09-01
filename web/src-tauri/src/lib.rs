mod backend;
mod commands;
mod dictation;
#[cfg(target_os = "macos")]
mod keychain;
mod lifecycle;
mod local_https;
mod navigation;
mod project_transfer;
mod server_commands;
mod team_connections;
mod team_session;
mod team_tunnel;
mod updates;
mod windows;

use std::time::Duration;

use backend::{BackendState, QuitRequest};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem, Submenu},
    Emitter, Manager, RunEvent, WindowEvent,
};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};

const QUIT_MENU_ID: &str = "rcp-quit";

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let backend_state = BackendState::default();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            verify_then_prepare_show(app.clone(), "second-launch");
        }))
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_opener::Builder::new()
                .open_js_links_on_click(false)
                .build(),
        )
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(backend_state)
        .menu(|app| {
            let quit = MenuItem::with_id(app, QUIT_MENU_ID, "Quit RCP", true, Some("CmdOrCtrl+Q"))?;
            let application = Submenu::with_items(app, "RCP", true, &[&quit])?;
            let edit = Submenu::with_items(
                app,
                "Edit",
                true,
                &[
                    &PredefinedMenuItem::undo(app, None)?,
                    &PredefinedMenuItem::redo(app, None)?,
                    &PredefinedMenuItem::separator(app)?,
                    &PredefinedMenuItem::cut(app, None)?,
                    &PredefinedMenuItem::copy(app, None)?,
                    &PredefinedMenuItem::paste(app, None)?,
                    &PredefinedMenuItem::select_all(app, None)?,
                ],
            )?;
            Menu::with_items(app, &[&application, &edit])
        })
        .on_menu_event(|app, event| {
            if event.id() == QUIT_MENU_ID {
                request_app_quit(app.clone());
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::desktop_status,
            commands::desktop_reconnect_backend,
            commands::desktop_show_ready,
            commands::desktop_list_team_connections,
            commands::desktop_configure_server_operator_route,
            commands::desktop_probe_server_operator,
            commands::desktop_run_project_provision,
            commands::desktop_open_project_provision_terminal,
            commands::desktop_run_project_transfer,
            commands::desktop_prepare_project_transfer,
            commands::desktop_load_project_transfer,
            commands::desktop_advance_project_transfer,
            commands::desktop_run_incoming_project_provision,
            commands::desktop_read_target_project_provisioning_options,
            commands::desktop_export_project_transfer,
            commands::desktop_select_project_transfer_export,
            commands::desktop_open_project_transfer_terminal,
            commands::desktop_finish_project_transfer,
            commands::desktop_discard_project_transfer_export,
            commands::desktop_remove_team_connection_metadata,
            commands::desktop_remove_team_member_token,
            commands::desktop_connect_team_tunnel,
            commands::desktop_enroll_team_connection,
            commands::desktop_add_existing_team_connection,
            commands::desktop_establish_team_session,
            commands::desktop_navigate_team,
            commands::desktop_return_to_personal,
            commands::choose_repository_folder,
            commands::desktop_start_dictation,
            commands::desktop_stop_dictation,
            commands::open_artifact_preview,
            commands::open_episode_report_preview,
            commands::open_repository_file_preview,
            commands::download_artifact,
            commands::open_external,
            commands::request_quit,
            commands::check_for_update,
            commands::apply_update,
        ])
        .setup(|app| {
            let local_https = local_https::LocalHttpsIdentity::load_or_create(app.handle())
                .map_err(std::io::Error::other)?;
            let team_connections = team_connections::TeamConnectionState::for_app(app.handle())
                .map_err(std::io::Error::other)?;
            if !app.manage(team_connections) {
                return Err(std::io::Error::other(
                    "RCP desktop team connection state was already registered",
                )
                .into());
            }
            let team_tunnels =
                team_tunnel::TeamTunnelState::new(&local_https).map_err(std::io::Error::other)?;
            if !app.manage(team_tunnels) {
                return Err(std::io::Error::other(
                    "RCP desktop team tunnel state was already registered",
                )
                .into());
            }
            let team_sessions = team_session::TeamSessionState::new(&local_https);
            if !app.manage(team_sessions) {
                return Err(std::io::Error::other(
                    "RCP desktop team session state was already registered",
                )
                .into());
            }
            let project_transfer_coordinator =
                project_transfer::ProjectTransferCoordinatorState::for_app(app.handle())
                    .map_err(std::io::Error::other)?;
            if !app.manage(project_transfer_coordinator) {
                return Err(std::io::Error::other(
                    "RCP desktop project transfer coordinator was already registered",
                )
                .into());
            }
            windows::create_main(app.handle(), &local_https)?;
            if !app.manage(local_https) {
                return Err(std::io::Error::other(
                    "RCP desktop local HTTPS identity was already registered",
                )
                .into());
            }
            start_backend(app.handle().clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main" {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    dictation::stop_active();
                    api.prevent_close();
                    windows::cancel_pending_show();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building RCP desktop app");

    app.run(|app, event| match event {
        RunEvent::ExitRequested {
            code: None, api, ..
        } => {
            api.prevent_exit();
            request_app_quit(app.clone());
        }
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => verify_then_prepare_show(app.clone(), "dock-reopen"),
        _ => {}
    });
}

fn start_backend(app: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        let state = app.state::<BackendState>().inner().clone();
        eprintln!("[rcp] connecting to a backend");
        match backend::connect(&app, &state, "Quit").await {
            Ok(status) => {
                if state.is_terminal() {
                    return;
                }
                eprintln!(
                    "[rcp] backend ready at {} (owned={})",
                    status.base_url, status.owned
                );
                finish_startup(&app, status)
            }
            Err(error) => {
                eprintln!("[rcp] backend connection failed: {error}");
                if error == backend::CONNECT_CANCELLED {
                    request_app_quit(app);
                    return;
                }
                if error == backend::CONNECT_ABORTED_BY_TERMINAL || state.is_terminal() {
                    return;
                }
                state.set_error(error.clone());
                app.dialog()
                    .message(format!("RCP could not start.\n\n{error}"))
                    .title("RCP could not start")
                    .buttons(MessageDialogButtons::Ok)
                    .blocking_show();
                if state.has_unconfirmed_children() {
                    request_app_quit_with_code(app, 1);
                } else {
                    app.exit(1);
                }
            }
        }
    });
}

fn finish_startup(app: &tauri::AppHandle, status: lifecycle::DesktopStatus) {
    if let Some(target) = windows::navigation_target(&status.base_url) {
        if let Some(window) = app.get_webview_window("main") {
            eprintln!("[rcp] main window navigating to {target}");
            if let Err(error) = window.navigate(target) {
                eprintln!("[rcp] the main window could not navigate: {error}");
            }
        }
    }
    let _ = windows::prepare_show(app, &status, "startup");
    windows::show_when_handshake_does_not_arrive(app);
}

fn verify_then_prepare_show(app: tauri::AppHandle, reason: &'static str) {
    tauri::async_runtime::spawn(async move {
        let state = app.state::<BackendState>().inner().clone();
        let status = match state.status() {
            Ok(status) => status,
            Err(error) => {
                let _ = app.emit_to(
                    "main",
                    "rcp://backend-mismatch",
                    serde_json::json!({"message": error}),
                );
                return;
            }
        };
        match backend::health(&status).await {
            Ok(health)
                if health.instance_id == status.instance_id
                    && health.data_dir_id == status.data_dir_id =>
            {
                state.update_health(&health);
                if let Err(error) = windows::recover_then_prepare_show(&app, &status, reason) {
                    eprintln!("[rcp] the main window could not be prepared: {error}");
                }
            }
            Ok(_) => {
                let _ = app.emit_to(
                    "main",
                    "rcp://backend-mismatch",
                    serde_json::json!({"message": "the backend identity changed"}),
                );
            }
            Err(error) => {
                let _ = app.emit_to(
                    "main",
                    "rcp://backend-mismatch",
                    serde_json::json!({"message": error}),
                );
            }
        }
    });
}

fn request_app_quit(app: tauri::AppHandle) -> bool {
    request_app_quit_with_code(app, 0)
}

fn request_app_quit_with_code(app: tauri::AppHandle, clean_exit_code: i32) -> bool {
    let state = app.state::<BackendState>().inner().clone();
    let team_tunnels = app.state::<team_tunnel::TeamTunnelState>().inner().clone();
    match state.begin_quit() {
        Ok(QuitRequest::Started) => {}
        Ok(QuitRequest::AlreadyQuitting) => return true,
        Ok(QuitRequest::Updating) => {
            tauri::async_runtime::spawn(async move {
                show_shutdown_problem(
                    &app,
                    "RCP is installing an update and will restart when it finishes. Quit was not started.",
                );
            });
            return false;
        }
        Err(error) => {
            tauri::async_runtime::spawn(async move {
                show_shutdown_problem(&app, &format!("Quit did not complete.\n\n{error}"));
            });
            return false;
        }
    }
    dictation::stop_active();
    tauri::async_runtime::spawn(async move {
        if let Err(error) = team_tunnels.stop_all_for_lifecycle().await {
            show_shutdown_problem(&app, &format!("Quit did not complete.\n\n{error}"));
            state.abort_quit().await;
            team_tunnels.resume_after_lifecycle_failure().await;
            return;
        }
        let shutdown = backend::stop_for_quit(&state).await;
        if let Some(message) = shutdown.problem() {
            let message = if shutdown.may_exit() {
                message.to_string()
            } else {
                format!("Quit did not complete.\n\n{message}")
            };
            show_shutdown_problem(&app, &message);
        }
        eprintln!("[rcp] desktop shutdown: {shutdown:?}");
        if !shutdown.may_exit() {
            state.abort_quit().await;
            team_tunnels.resume_after_lifecycle_failure().await;
            return;
        }
        let exit_code = if shutdown.is_clean() {
            clean_exit_code
        } else {
            1
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        app.exit(exit_code);
    });
    true
}

fn show_shutdown_problem(app: &tauri::AppHandle, message: &str) {
    app.dialog()
        .message(message)
        .title("RCP shutdown")
        .buttons(MessageDialogButtons::Ok)
        .blocking_show();
}
