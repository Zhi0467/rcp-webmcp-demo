use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock,
    },
    time::Duration,
};

use tauri::{
    webview::{NewWindowResponse, PageLoadEvent, WebviewWindow, WebviewWindowBuilder},
    AppHandle, Emitter, Manager, WebviewUrl,
};
use tauri_plugin_opener::OpenerExt;
use url::Url;

use crate::backend::BackendState;
use crate::{
    lifecycle::DesktopStatus,
    local_https::{self, LocalHttpsIdentity},
    navigation,
    team_connections::TeamConnectionState,
};

static PREVIEW_SEQUENCE: AtomicU64 = AtomicU64::new(1);
static SHOW_REQUEST_GENERATION: AtomicU64 = AtomicU64::new(0);
static INITIAL_URL: OnceLock<Url> = OnceLock::new();
static PENDING_RECOVERY: OnceLock<Mutex<Option<PendingRecovery>>> = OnceLock::new();

/// How long the hidden window waits for the frontend handshake before showing
/// itself anyway. A window that only ever appears on success turns any failure
/// into an app that silently does not open.
const HANDSHAKE_SHOW_TIMEOUT: Duration = Duration::from_secs(8);

/// The placeholder the main window holds while a cold start waits for its
/// backend. It is never the resting state: `finish_startup` navigates away from
/// it, and the handshake timeout still reveals the window if that never happens.
const BLANK_URL: &str = "about:blank";
const BACKEND_URL: &str = "http://127.0.0.1:8421";
const FRONTEND_URL_VARIABLE: &str = "RCP_DESKTOP_FRONTEND_URL";

#[derive(Debug, PartialEq)]
enum InitialNavigation {
    Eager(Url),
    AfterBackendReady(Url),
}

#[derive(Debug)]
struct PendingRecovery {
    generation: u64,
    base_url: String,
    instance_id: String,
    reason: String,
}

pub fn create_main(
    app: &AppHandle,
    local_https_identity: &LocalHttpsIdentity,
) -> Result<(), String> {
    let configured_url = std::env::var(FRONTEND_URL_VARIABLE).ok();
    let initial_navigation = initial_navigation(uses_vite_dev_server(), configured_url.as_deref())?;
    let start_url = match initial_navigation {
        InitialNavigation::Eager(url) => {
            eprintln!("[rcp] main window loading {url}");
            let _ = INITIAL_URL.set(url.clone());
            url
        }
        InitialNavigation::AfterBackendReady(url) => {
            eprintln!("[rcp] main window waiting for the backend at {url}");
            Url::parse(BLANK_URL).map_err(|error| format!("unusable blank page: {error}"))?
        }
    };
    let app_for_navigation = app.clone();
    let app_for_popup = app.clone();
    let window = WebviewWindowBuilder::new(app, "main", WebviewUrl::External(start_url))
        .title("RCP")
        .inner_size(1320.0, 860.0)
        .min_inner_size(880.0, 600.0)
        .visible(false)
        .zoom_hotkeys_enabled(false)
        .on_page_load(|window, payload| {
            finish_recovered_page_load(&window, payload.url(), payload.event());
        })
        .on_navigation(move |url| {
            if url.as_str() == BLANK_URL {
                return true;
            }
            let current_base = app_for_navigation
                .state::<BackendState>()
                .status()
                .ok()
                .map(|status| status.base_url);
            let team_origins = match app_for_navigation.state::<TeamConnectionState>().list() {
                Ok(connections) => connections
                    .into_iter()
                    .map(|connection| connection.local_origin)
                    .collect::<Vec<_>>(),
                Err(error) => {
                    eprintln!(
                        "[rcp] rejecting team navigation because saved connections are unreadable: {error}"
                    );
                    Vec::new()
                }
            };
            if navigation::is_main_window_url(
                url,
                current_base.as_deref(),
                &team_origins,
                uses_vite_dev_server(),
            )
            {
                true
            } else {
                if navigation::is_external_reference(url) {
                    let _ = app_for_navigation
                        .opener()
                        .open_url(url.as_str(), None::<&str>);
                }
                false
            }
        })
        .on_new_window(move |url, _features| {
            open_main_popup(&app_for_popup, url);
            NewWindowResponse::Deny
        })
        .build()
        .map_err(|error| format!("could not create the RCP window: {error}"))?;
    local_https::install_webview_trust(&window, local_https_identity)?;
    Ok(())
}

fn open_main_popup(app: &AppHandle, url: Url) {
    let current_base = app
        .state::<BackendState>()
        .status()
        .ok()
        .map(|status| status.base_url);
    if let Some(base_url) = current_base {
        if is_same_origin_popup(&url, &base_url) {
            if let Err(error) = open_preview(app, url, base_url) {
                eprintln!("[rcp] the preview window could not open: {error}");
            }
            return;
        }
    }
    if navigation::is_external_reference(&url) {
        let _ = app.opener().open_url(url.as_str(), None::<&str>);
    }
}

/// Whether a popup targets RCP's own backend origin. The episode report is what
/// prompted this, but the rule is deliberately about origin rather than that one
/// path: any same-origin popup belongs in a preview window instead of being
/// silently dropped. `open_preview` re-checks the origin before it builds.
fn is_same_origin_popup(url: &Url, base_url: &str) -> bool {
    url.scheme() != "about" && navigation::is_loopback_rcp_url(url, base_url, false)
}

pub fn prepare_show(app: &AppHandle, status: &DesktopStatus, reason: &str) -> Result<(), String> {
    emit_prepare_show(app, &status.instance_id, reason)
}

fn emit_prepare_show(app: &AppHandle, instance_id: &str, reason: &str) -> Result<(), String> {
    app.emit_to(
        "main",
        "rcp://prepare-show",
        serde_json::json!({"reason": reason, "instanceId": instance_id}),
    )
    .map_err(|error| format!("could not request a window refresh: {error}"))
}

/// Repair a hidden main window before asking the frontend to refresh. The
/// window stays hidden across the navigation so an API response or another
/// same-origin error document is never flashed as the RCP surface.
pub fn recover_then_prepare_show(
    app: &AppHandle,
    status: &DesktopStatus,
    reason: &str,
) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "the RCP window is unavailable".to_string())?;
    let current = window
        .url()
        .map_err(|error| format!("could not inspect the RCP window: {error}"))?;
    if let Some(target) =
        reopen_navigation_target(&current, &status.base_url, uses_vite_dev_server())?
    {
        eprintln!("[rcp] recovering the main window from {current} to {target}");
        let generation = arm_handshake_fallback(app);
        stage_pending_recovery(PendingRecovery {
            generation,
            base_url: status.base_url.clone(),
            instance_id: status.instance_id.clone(),
            reason: reason.to_string(),
        })?;
        window
            .hide()
            .map_err(|error| format!("could not hide the invalid RCP window: {error}"))?;
        window
            .navigate(target)
            .map_err(|error| format!("could not recover the RCP window: {error}"))?;
        return Ok(());
    }

    let result = prepare_show(app, status, reason);
    show_when_handshake_does_not_arrive(app);
    result
}

/// The backend origin the window must move to, or `None` when it is already
/// there — or when `tauri dev` is serving the frontend from Vite.
pub fn navigation_target(base_url: &str) -> Option<Url> {
    if uses_vite_dev_server() {
        return None;
    }
    let target = Url::parse(base_url).ok()?;
    match INITIAL_URL.get() {
        Some(initial) if *initial == target => None,
        _ => Some(target),
    }
}

pub fn personal_root(base_url: &str) -> Result<Url, String> {
    if uses_vite_dev_server() {
        return INITIAL_URL
            .get()
            .cloned()
            .or_else(|| Url::parse("http://127.0.0.1:5173").ok())
            .ok_or_else(|| "the personal development frontend URL is invalid".to_string());
    }
    let mut target = Url::parse(base_url)
        .map_err(|error| format!("the personal RCP URL is invalid: {error}"))?;
    target.set_path("/");
    target.set_query(None);
    target.set_fragment(None);
    Ok(target)
}

pub fn navigate_main(window: &WebviewWindow, origin: &str) -> Result<(), String> {
    navigate_main_route(window, origin, None)
}

pub fn navigate_main_route(
    window: &WebviewWindow,
    origin: &str,
    fragment: Option<&str>,
) -> Result<(), String> {
    let mut target = Url::parse(origin).map_err(|_| "the RCP navigation origin is invalid")?;
    target.set_path("/");
    target.set_query(None);
    target.set_fragment(fragment);
    window
        .navigate(target)
        .map_err(|error| format!("could not navigate the RCP window: {error}"))
}

pub fn show_when_handshake_does_not_arrive(app: &AppHandle) {
    let _ = arm_handshake_fallback(app);
}

fn arm_handshake_fallback(app: &AppHandle) -> u64 {
    let generation = begin_show_request();
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(HANDSHAKE_SHOW_TIMEOUT).await;
        if !show_request_is_current(generation) {
            return;
        }
        let Some(window) = app.get_webview_window("main") else {
            return;
        };
        let action = if window.is_visible().unwrap_or(false) {
            "focusing"
        } else {
            "showing"
        };
        eprintln!("[rcp] the frontend handshake did not arrive; {action} the window anyway");
        if let Err(error) = show_main(&app) {
            eprintln!("[rcp] the window could not be shown: {error}");
        }
    });
    generation
}

pub fn show_main(app: &AppHandle) -> Result<(), String> {
    cancel_pending_show();
    eprintln!("[rcp] showing the RCP window");
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "the RCP window is unavailable".to_string())?;
    window.show().map_err(|error| error.to_string())?;
    window.unminimize().map_err(|error| error.to_string())?;
    window.set_focus().map_err(|error| error.to_string())
}

/// Invalidate a timeout belonging to an earlier show request. In particular,
/// closing the window after a Dock reopen must not let that request's fallback
/// show it again several seconds later.
pub fn cancel_pending_show() {
    SHOW_REQUEST_GENERATION.fetch_add(1, Ordering::SeqCst);
    if let Ok(mut pending) = pending_recovery().lock() {
        *pending = None;
    }
}

pub fn open_preview(app: &AppHandle, url: Url, base_url: String) -> Result<(), String> {
    if !navigation::is_loopback_rcp_url(&url, &base_url, false) {
        return Err("artifact preview URL is outside the RCP backend".into());
    }
    let label = format!(
        "artifact-preview-{}",
        PREVIEW_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    let app_for_navigation = app.clone();
    let app_for_popup = app.clone();
    WebviewWindowBuilder::new(app, label, WebviewUrl::External(url))
        .title("RCP artifact preview")
        .inner_size(1040.0, 760.0)
        .min_inner_size(520.0, 400.0)
        .on_navigation(move |candidate| {
            if navigation::is_loopback_rcp_url(candidate, &base_url, false) {
                true
            } else {
                if navigation::is_external_reference(candidate) {
                    let _ = app_for_navigation
                        .opener()
                        .open_url(candidate.as_str(), None::<&str>);
                }
                false
            }
        })
        .on_new_window(move |candidate, _features| {
            if navigation::is_external_reference(&candidate) {
                let _ = app_for_popup
                    .opener()
                    .open_url(candidate.as_str(), None::<&str>);
            }
            NewWindowResponse::Deny
        })
        .build()
        .map_err(|error| format!("could not open artifact preview: {error}"))?;
    Ok(())
}

pub fn uses_vite_dev_server() -> bool {
    cfg!(debug_assertions) && !crate::backend::is_bundled_dev_app()
}

fn reopen_navigation_target(
    current: &Url,
    base_url: &str,
    allow_dev: bool,
) -> Result<Option<Url>, String> {
    if navigation::is_rcp_app_document_url(current, base_url, allow_dev) {
        return Ok(None);
    }
    let mut target =
        Url::parse(base_url).map_err(|error| format!("invalid verified backend URL: {error}"))?;
    target.set_path("/");
    target.set_query(None);
    target.set_fragment(None);
    Ok(Some(target))
}

fn pending_recovery() -> &'static Mutex<Option<PendingRecovery>> {
    PENDING_RECOVERY.get_or_init(|| Mutex::new(None))
}

fn stage_pending_recovery(pending: PendingRecovery) -> Result<(), String> {
    *pending_recovery()
        .lock()
        .map_err(|_| "the recovered-window handshake state is unavailable".to_string())? =
        Some(pending);
    Ok(())
}

fn finish_recovered_page_load(window: &WebviewWindow, url: &Url, event: PageLoadEvent) {
    let pending = {
        let Ok(mut slot) = pending_recovery().lock() else {
            eprintln!("[rcp] the recovered-window handshake state is unavailable");
            return;
        };
        let Some(pending) = slot.as_ref() else {
            return;
        };
        if !is_finished_recovered_document(event, url, &pending.base_url) {
            return;
        }
        slot.take()
    };
    let Some(pending) = pending else {
        return;
    };
    if !show_request_is_current(pending.generation) {
        return;
    }
    if let Err(error) =
        emit_prepare_show(window.app_handle(), &pending.instance_id, &pending.reason)
    {
        eprintln!("[rcp] the recovered window could not request a refresh: {error}");
    }
}

fn is_finished_recovered_document(event: PageLoadEvent, url: &Url, base_url: &str) -> bool {
    event == PageLoadEvent::Finished && navigation::is_rcp_app_document_url(url, base_url, false)
}

fn begin_show_request() -> u64 {
    let generation = SHOW_REQUEST_GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    if let Ok(mut pending) = pending_recovery().lock() {
        *pending = None;
    }
    generation
}

fn show_request_is_current(generation: u64) -> bool {
    SHOW_REQUEST_GENERATION.load(Ordering::SeqCst) == generation
}

/// Resolve the requested frontend without reading process-global state, then
/// decide whether it is safe to navigate before backend readiness. A backend
/// URL is always deferred, including when it was explicitly configured.
fn initial_navigation(
    uses_vite_dev_server: bool,
    configured_url: Option<&str>,
) -> Result<InitialNavigation, String> {
    let default = if uses_vite_dev_server {
        "http://127.0.0.1:5173"
    } else {
        BACKEND_URL
    };
    let raw = configured_url.unwrap_or(default);
    let url =
        Url::parse(raw).map_err(|error| format!("invalid {FRONTEND_URL_VARIABLE}: {error}"))?;
    if !navigation::is_loopback_rcp_url(&url, BACKEND_URL, cfg!(debug_assertions)) {
        return Err("RCP_DESKTOP_FRONTEND_URL must be an approved loopback RCP origin".into());
    }
    let backend = Url::parse(BACKEND_URL).expect("the built-in backend URL must be valid");
    if url.origin() == backend.origin() {
        Ok(InitialNavigation::AfterBackendReady(url))
    } else {
        Ok(InitialNavigation::Eager(url))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(raw: &str) -> Url {
        Url::parse(raw).unwrap()
    }

    #[test]
    fn initial_navigation_defers_only_the_backend_origin() {
        assert_eq!(
            initial_navigation(false, None).unwrap(),
            InitialNavigation::AfterBackendReady(url("http://127.0.0.1:8421")),
        );
        assert_eq!(
            initial_navigation(true, None).unwrap(),
            InitialNavigation::Eager(url("http://127.0.0.1:5173")),
        );
        assert_eq!(
            initial_navigation(false, Some("http://127.0.0.1:5173")).unwrap(),
            InitialNavigation::Eager(url("http://127.0.0.1:5173")),
        );
        assert_eq!(
            initial_navigation(true, Some("http://127.0.0.1:8421")).unwrap(),
            InitialNavigation::AfterBackendReady(url("http://127.0.0.1:8421")),
        );
    }

    #[test]
    fn main_window_routes_same_origin_report_popups_to_a_preview_window() {
        let base = "http://127.0.0.1:18421";
        let report =
            url("http://127.0.0.1:18421/api/projects/project/episodes/episode/report/preview");
        assert!(navigation::is_external_reference(&report));
        assert!(is_same_origin_popup(&report, base));
        assert!(!is_same_origin_popup(
            &url("http://127.0.0.1:19421/api/report/preview"),
            base,
        ));
        assert!(!is_same_origin_popup(
            &url("https://example.com/report"),
            base
        ));
        assert!(!is_same_origin_popup(&url("about:blank"), base));
    }

    #[test]
    fn reopen_keeps_app_root_and_repairs_same_origin_error_documents() {
        let base = "http://127.0.0.1:18421";
        assert_eq!(
            reopen_navigation_target(&url("http://127.0.0.1:18421/#/projects/a"), base, false)
                .unwrap(),
            None,
        );
        assert_eq!(
            reopen_navigation_target(
                &url("http://127.0.0.1:18421/api/projects/missing"),
                base,
                false,
            )
            .unwrap(),
            Some(url("http://127.0.0.1:18421/")),
        );
        assert_eq!(
            reopen_navigation_target(&url("about:blank"), base, false).unwrap(),
            Some(url("http://127.0.0.1:18421/")),
        );
    }

    #[test]
    fn cancelling_a_show_request_invalidates_its_fallback() {
        let generation = begin_show_request();
        assert!(show_request_is_current(generation));
        cancel_pending_show();
        assert!(!show_request_is_current(generation));
    }

    #[test]
    fn recovered_handshake_waits_for_the_backend_root_to_finish_loading() {
        let base = "http://127.0.0.1:18421";
        let root = url("http://127.0.0.1:18421/");
        assert!(!is_finished_recovered_document(
            PageLoadEvent::Started,
            &root,
            base,
        ));
        assert!(is_finished_recovered_document(
            PageLoadEvent::Finished,
            &root,
            base,
        ));
        assert!(!is_finished_recovered_document(
            PageLoadEvent::Finished,
            &url("http://127.0.0.1:18421/api/health"),
            base,
        ));
    }
}
