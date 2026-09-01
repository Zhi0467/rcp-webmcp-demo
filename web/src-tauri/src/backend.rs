use std::{
    env,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use tauri::AppHandle;
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};
use tauri_plugin_shell::{
    process::{CommandEvent, TerminatedPayload},
    ShellExt,
};
use tokio::{sync::Notify, time};

use crate::lifecycle::{DesktopStatus, Health, LaunchOutcome};

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: &str = "8421";
const HEALTH_READY_TIMEOUT: Duration = Duration::from_secs(12);
// The dev app runs the checkout backend with `--web-assets source`, and that mode
// rebuilds the frontend unconditionally before the server reports its launch
// result. A full `npm run build` takes far longer than a packaged backend's
// startup, so the launch deadline has to cover it. Keeping the packaged 12s here
// is what made every cold dev start time out while a warm one succeeded.
const LAUNCH_RESULT_TIMEOUT: Duration = if cfg!(debug_assertions) {
    Duration::from_secs(180)
} else {
    HEALTH_READY_TIMEOUT
};
// The packaged one-file supervisor can report SIGTERM after Uvicorn completed
// teardown. packaging/smoke-backend.py establishes this exact stderr receipt.
const SHUTDOWN_ACKNOWLEDGEMENT: &[u8] = b"Application shutdown complete.";
// Lifespan teardown sequentially stops the watcher poller for up to 16 seconds,
// the graph-watcher retry worker for up to 16 seconds, and background workers
// for up to 7 seconds. Keep margin beyond that 39-second window.
const GRACEFUL_STOP_TIMEOUT_SECONDS: u64 = 45;
const GRACEFUL_STOP_TIMEOUT: Duration = Duration::from_secs(GRACEFUL_STOP_TIMEOUT_SECONDS);
const FORCED_STOP_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Default)]
pub struct BackendState {
    inner: Arc<Mutex<BackendRuntime>>,
    coordinator: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Default)]
struct BackendRuntime {
    status: Option<DesktopStatus>,
    // A receipt for the exact child this shell started, not cached ownership of
    // whatever currently answers at the backend address. Quit compares its id
    // with live health before this process record grants shutdown authority.
    owned_backend: Option<OwnedBackend>,
    terminal: TerminalState,
    deferred_quit_shutdown: Option<ShutdownResult>,
    unconfirmed_children: Vec<UnconfirmedChild>,
    startup_error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TerminalState {
    #[default]
    Idle,
    Quitting,
    Updating,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuitRequest {
    Started,
    AlreadyQuitting,
    Updating,
}

impl BackendState {
    fn set_ready(&self, status: DesktopStatus, process: BackendProcess) -> bool {
        let mut inner = self.inner.lock().expect("backend state poisoned");
        if inner.terminal != TerminalState::Idle {
            return false;
        }
        if status.owned {
            inner.owned_backend = Some(OwnedBackend {
                instance_id: status.instance_id.clone(),
                base_url: status.base_url.clone(),
                process,
            });
        } else if inner
            .owned_backend
            .as_ref()
            .is_some_and(|backend| backend.instance_id != status.instance_id)
        {
            inner.owned_backend = None;
        }
        inner.status = Some(status);
        inner.startup_error = None;
        true
    }

    pub fn set_error(&self, message: String) {
        self.inner
            .lock()
            .expect("backend state poisoned")
            .startup_error = Some(message);
    }

    pub fn status(&self) -> Result<DesktopStatus, String> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| "backend state is unavailable")?;
        inner.status.clone().ok_or_else(|| {
            inner
                .startup_error
                .clone()
                .unwrap_or_else(|| "RCP is still starting".into())
        })
    }

    fn owned_backend(&self) -> Result<Option<OwnedBackend>, String> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| "backend state is unavailable")?;
        Ok(inner.owned_backend.clone())
    }

    pub fn begin_quit(&self) -> Result<QuitRequest, String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "backend lifecycle state is unavailable".to_string())?;
        match inner.terminal {
            TerminalState::Idle => {
                inner.terminal = TerminalState::Quitting;
                Ok(QuitRequest::Started)
            }
            TerminalState::Quitting => Ok(QuitRequest::AlreadyQuitting),
            TerminalState::Updating => Ok(QuitRequest::Updating),
        }
    }

    #[cfg(test)]
    fn is_quitting(&self) -> bool {
        self.inner
            .lock()
            .map_or(true, |inner| inner.terminal == TerminalState::Quitting)
    }

    pub fn is_terminal(&self) -> bool {
        self.inner
            .lock()
            .map_or(true, |inner| inner.terminal != TerminalState::Idle)
    }

    fn defer_quit_shutdown(&self, shutdown: ShutdownResult) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.deferred_quit_shutdown.get_or_insert(shutdown);
        }
    }

    fn take_deferred_quit_shutdown(&self) -> Option<ShutdownResult> {
        self.inner.lock().ok()?.deferred_quit_shutdown.take()
    }

    fn retain_unpublished_owned(&self, started: &StartedBackend) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.owned_backend = Some(OwnedBackend {
                instance_id: started.status.instance_id.clone(),
                base_url: started.status.base_url.clone(),
                process: started.process.clone(),
            });
        }
    }

    fn retain_unconfirmed_child(&self, child: UnconfirmedChild) {
        if let Ok(mut inner) = self.inner.lock() {
            if !inner
                .unconfirmed_children
                .iter()
                .any(|current| current.process.same_process(&child.process))
            {
                inner.unconfirmed_children.push(child);
            }
        }
    }

    fn unconfirmed_children(&self) -> Result<Vec<UnconfirmedChild>, String> {
        self.inner
            .lock()
            .map(|inner| inner.unconfirmed_children.clone())
            .map_err(|_| "backend lifecycle state is unavailable".to_string())
    }

    fn remove_unconfirmed_child(&self, process: &BackendProcess) {
        if let Ok(mut inner) = self.inner.lock() {
            inner
                .unconfirmed_children
                .retain(|child| !child.process.same_process(process));
        }
    }

    pub fn has_unconfirmed_children(&self) -> bool {
        match self.inner.lock() {
            Ok(inner) => !inner.unconfirmed_children.is_empty(),
            Err(_) => true,
        }
    }

    pub async fn abort_quit(&self) {
        let _guard = self.coordinator.lock().await;
        if let Ok(mut inner) = self.inner.lock() {
            if inner.terminal == TerminalState::Quitting {
                inner.terminal = TerminalState::Idle;
            }
        }
    }

    pub async fn begin_update(&self) -> Result<UpdateGuard, String> {
        let coordinator = self.coordinator.clone().lock_owned().await;
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "backend lifecycle state is unavailable".to_string())?;
        match inner.terminal {
            TerminalState::Idle => inner.terminal = TerminalState::Updating,
            TerminalState::Quitting => return Err("RCP is already quitting".into()),
            TerminalState::Updating => return Err("an RCP update is already in progress".into()),
        }
        drop(inner);
        Ok(UpdateGuard {
            state: self.clone(),
            _coordinator: coordinator,
        })
    }

    pub fn update_health(&self, health: &Health) {
        if let Ok(mut inner) = self.inner.lock() {
            if let Some(status) = inner.status.as_mut() {
                if status.matches_health(health) {
                    status.active_agent_tasks = health.active_agent_tasks;
                    status.owner_kind = health.owner_kind.clone();
                }
            }
        }
    }

    pub(crate) fn reset_connection_for_recovery(&self) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "backend lifecycle state is unavailable".to_string())?;
        inner.status = None;
        inner.owned_backend = None;
        inner.startup_error = None;
        Ok(())
    }
}

pub struct UpdateGuard {
    state: BackendState,
    _coordinator: tokio::sync::OwnedMutexGuard<()>,
}

impl UpdateGuard {
    pub async fn stop_backend(&self) -> ShutdownResult {
        graceful_stop_locked(&self.state).await
    }
}

impl Drop for UpdateGuard {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.state.inner.lock() {
            if inner.terminal == TerminalState::Updating {
                inner.terminal = TerminalState::Idle;
            }
        }
    }
}

#[derive(Clone)]
struct OwnedBackend {
    instance_id: String,
    base_url: String,
    process: BackendProcess,
}

#[derive(Clone)]
struct UnconfirmedChild {
    process: BackendProcess,
    identity: Option<BackendIdentity>,
}

#[derive(Clone)]
struct BackendIdentity {
    instance_id: String,
    base_url: String,
}

impl BackendIdentity {
    fn from_owned_outcome(outcome: &LaunchOutcome) -> Option<Self> {
        if outcome.outcome != "owned" || !outcome.owned {
            return None;
        }
        Some(Self {
            instance_id: outcome.instance_id.clone()?,
            base_url: outcome.base_url.clone(),
        })
    }
}

impl UnconfirmedChild {
    fn signal_pid(&self, health: Option<&Health>) -> u32 {
        match (&self.identity, health) {
            (Some(identity), Some(health)) if identity.instance_id == health.instance_id => {
                health.pid
            }
            _ => self.process.pid(),
        }
    }
}

impl OwnedBackend {
    fn matches_live_instance(&self, health: &Health) -> bool {
        self.instance_id == health.instance_id
    }
}

#[derive(Clone)]
pub struct BackendProcess {
    pid: u32,
    exit: Arc<ProcessExit>,
}

#[derive(Default)]
struct ProcessExit {
    state: Mutex<ProcessExitState>,
    changed: Notify,
}

#[derive(Default)]
struct ProcessExitState {
    result: Option<TerminatedPayload>,
    sigterm_attempt_active: bool,
    sigterm_requested: bool,
    shutdown_acknowledged: bool,
    acknowledgement_tail: Vec<u8>,
}

#[derive(Clone, Debug)]
struct ProcessTermination {
    code: Option<i32>,
    signal: Option<i32>,
    sigterm_requested: bool,
    shutdown_acknowledged: bool,
}

impl ProcessExit {
    fn observe_output(&self, bytes: &[u8]) {
        let mut state = self.state.lock().expect("process exit state poisoned");
        if !state.sigterm_attempt_active || state.shutdown_acknowledged {
            return;
        }

        let marker = SHUTDOWN_ACKNOWLEDGEMENT;
        let boundary_prefix_len = bytes.len().min(marker.len().saturating_sub(1));
        let mut boundary =
            Vec::with_capacity(state.acknowledgement_tail.len() + boundary_prefix_len);
        boundary.extend_from_slice(&state.acknowledgement_tail);
        boundary.extend_from_slice(&bytes[..boundary_prefix_len]);
        state.shutdown_acknowledged = bytes.windows(marker.len()).any(|window| window == marker)
            || boundary
                .windows(marker.len())
                .any(|window| window == marker);
        if state.shutdown_acknowledged {
            state.acknowledgement_tail.clear();
            return;
        }

        let retained = marker.len().saturating_sub(1);
        if bytes.len() >= retained {
            state.acknowledgement_tail.clear();
            state
                .acknowledgement_tail
                .extend_from_slice(&bytes[bytes.len() - retained..]);
        } else {
            state.acknowledgement_tail.extend_from_slice(bytes);
            if state.acknowledgement_tail.len() > retained {
                let excess = state.acknowledgement_tail.len() - retained;
                state.acknowledgement_tail.drain(..excess);
            }
        }
    }

    fn record_termination(&self, payload: TerminatedPayload) {
        self.state
            .lock()
            .expect("process exit state poisoned")
            .result = Some(payload);
        self.changed.notify_waiters();
    }

    fn begin_sigterm_attempt(&self) {
        let mut state = self.state.lock().expect("process exit state poisoned");
        state.sigterm_attempt_active = true;
        state.sigterm_requested = false;
        state.shutdown_acknowledged = false;
        state.acknowledgement_tail.clear();
    }

    fn finish_sigterm_delivery(&self, delivered: bool) {
        let mut state = self.state.lock().expect("process exit state poisoned");
        if delivered {
            state.sigterm_requested = state.sigterm_attempt_active;
        } else {
            state.sigterm_attempt_active = false;
            state.sigterm_requested = false;
            state.shutdown_acknowledged = false;
            state.acknowledgement_tail.clear();
        }
    }

    fn termination(&self) -> Option<ProcessTermination> {
        let state = self.state.lock().ok()?;
        let result = state.result.as_ref()?;
        Some(ProcessTermination {
            code: result.code,
            signal: result.signal,
            sigterm_requested: state.sigterm_requested,
            shutdown_acknowledged: state.shutdown_acknowledged,
        })
    }
}

impl BackendProcess {
    pub fn pid(&self) -> u32 {
        self.pid
    }

    fn termination(&self) -> Option<ProcessTermination> {
        self.exit.termination()
    }

    fn same_process(&self, other: &Self) -> bool {
        self.pid == other.pid && Arc::ptr_eq(&self.exit, &other.exit)
    }

    fn begin_sigterm_attempt(&self) {
        self.exit.begin_sigterm_attempt();
    }

    fn finish_sigterm_delivery(&self, delivered: bool) {
        self.exit.finish_sigterm_delivery(delivered);
    }

    async fn wait(&self, timeout: Duration) -> Option<ProcessTermination> {
        let deadline = time::Instant::now() + timeout;
        loop {
            if let Some(termination) = self.termination() {
                return Some(termination);
            }
            let changed = self.exit.changed.notified();
            if let Some(termination) = self.termination() {
                return Some(termination);
            }
            if time::timeout_at(deadline, changed).await.is_err() {
                return self.termination();
            }
        }
    }
}

pub struct StartedBackend {
    pub status: DesktopStatus,
    pub process: BackendProcess,
}

struct StartFailure {
    message: String,
    unconfirmed_child: Option<UnconfirmedChild>,
}

#[derive(Debug, PartialEq, Eq)]
enum LaunchWaitFailure {
    Ended,
    TimedOut,
}

impl StartFailure {
    fn confirmed(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            unconfirmed_child: None,
        }
    }

    fn unconfirmed(message: impl Into<String>, child: UnconfirmedChild) -> Self {
        Self {
            message: message.into(),
            unconfirmed_child: Some(child),
        }
    }
}

pub const CONNECT_CANCELLED: &str = "the existing RCP backend was left running";
pub const CONNECT_ABORTED_BY_TERMINAL: &str =
    "backend connection stopped because RCP is quitting or updating";

pub async fn connect(
    app: &AppHandle,
    state: &BackendState,
    cancel_label: &str,
) -> Result<DesktopStatus, String> {
    let _guard = state.coordinator.lock().await;
    if state.is_terminal() {
        return Err(CONNECT_ABORTED_BY_TERMINAL.into());
    }
    if let Ok(mut status) = state.status() {
        if let Ok(current) = health(&status).await {
            if status.matches_health(&current) {
                state.update_health(&current);
                status.active_agent_tasks = current.active_agent_tasks;
                status.owner_kind = current.owner_kind;
                return Ok(status);
            }
        }
    }
    let started = match start_and_record(app, state, false).await {
        Ok(started) => started,
        Err(error) => {
            if state.is_terminal() {
                return Err(CONNECT_ABORTED_BY_TERMINAL.into());
            }
            let refusal = serde_json::from_str::<LaunchOutcome>(&error).ok();
            let Some(refusal) = refusal else {
                return Err(error);
            };
            let reason = refusal
                .reason
                .unwrap_or_else(|| "the current backend cannot be reused".into());
            let replace = app
                .dialog()
                .message(format!(
                    "RCP could not use the current backend.\n\n{reason}"
                ))
                .title("RCP backend needs attention")
                .buttons(MessageDialogButtons::OkCancelCustom(
                    "Replace gracefully".into(),
                    cancel_label.into(),
                ))
                .blocking_show();
            if !replace {
                return Err(CONNECT_CANCELLED.into());
            }
            if state.is_terminal() {
                return Err(CONNECT_ABORTED_BY_TERMINAL.into());
            }
            let started = start_and_record(app, state, true).await?;
            if state.is_terminal() {
                stop_unpublished_backend(state, &started).await;
                return Err(CONNECT_ABORTED_BY_TERMINAL.into());
            }
            started
        }
    };
    let status = started.status.clone();
    if state.set_ready(started.status.clone(), started.process.clone()) {
        Ok(status)
    } else {
        stop_unpublished_backend(state, &started).await;
        Err(CONNECT_ABORTED_BY_TERMINAL.into())
    }
}

async fn start_and_record(
    app: &AppHandle,
    state: &BackendState,
    force: bool,
) -> Result<StartedBackend, String> {
    match start(app, force).await {
        Ok(started) => Ok(started),
        Err(failure) => {
            if let Some(child) = failure.unconfirmed_child {
                state.retain_unconfirmed_child(child);
            }
            Err(failure.message)
        }
    }
}

async fn stop_unpublished_backend(state: &BackendState, started: &StartedBackend) {
    if started.status.owned {
        state.retain_unpublished_owned(started);
        let shutdown = stop_owned_backend_locked(state).await;
        state.defer_quit_shutdown(shutdown);
    }
}

async fn start(app: &AppHandle, force: bool) -> Result<StartedBackend, StartFailure> {
    let (mut events, child) = backend_command(app, force)
        .map_err(StartFailure::confirmed)?
        .spawn()
        .map_err(|error| {
            StartFailure::confirmed(format!("could not start the RCP backend process: {error}"))
        })?;
    let process = BackendProcess {
        pid: child.pid(),
        exit: Arc::new(ProcessExit::default()),
    };
    drop(child);

    let (startup_tx, mut startup_rx) = tokio::sync::mpsc::unbounded_channel();
    let exit = process.exit.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Terminated(payload) => {
                    exit.record_termination(payload);
                    break;
                }
                CommandEvent::Stdout(bytes) => {
                    let _ = startup_tx.send(CommandEvent::Stdout(bytes));
                }
                CommandEvent::Stderr(bytes) => {
                    exit.observe_output(&bytes);
                    let _ = startup_tx.send(CommandEvent::Stderr(bytes));
                }
                event => {
                    let _ = startup_tx.send(event);
                }
            }
        }
    });

    let mut stderr = Vec::new();
    let mut stdout_pending = Vec::new();
    let launch_deadline = time::Instant::now() + LAUNCH_RESULT_TIMEOUT;
    let outcome = match wait_for_launch_outcome(
        &mut startup_rx,
        &mut stderr,
        &mut stdout_pending,
        launch_deadline,
    )
    .await
    {
        Ok(outcome) => outcome,
        Err(failure) => {
            let error = match failure {
                LaunchWaitFailure::Ended => {
                    launch_error(&stderr, "backend ended before reporting launch status")
                }
                LaunchWaitFailure::TimedOut => {
                    "timed out waiting for the backend launch result".to_string()
                }
            };
            let child = UnconfirmedChild {
                process: process.clone(),
                identity: None,
            };
            return Err(start_failure_after_cleanup(
                error,
                child,
                stop_process(&process, process.pid()).await,
                Some("the unidentified backend launcher required forced termination"),
                "its launcher could not be cleaned up",
            ));
        }
    };

    if outcome.is_refusal() {
        let error = serde_json::to_string(&outcome).unwrap_or_else(|_| {
            outcome
                .reason
                .clone()
                .unwrap_or_else(|| "backend refused to start".into())
        });
        let child = UnconfirmedChild {
            process: process.clone(),
            identity: None,
        };
        return Err(start_failure_after_cleanup(
            error,
            child,
            stop_process(&process, process.pid()).await,
            None,
            "the refusing launcher could not be cleaned up",
        ));
    }

    let health = match wait_for_health(&outcome).await {
        Ok(health) => health,
        Err(error) => {
            let child = UnconfirmedChild {
                process: process.clone(),
                identity: BackendIdentity::from_owned_outcome(&outcome),
            };
            let cleanup =
                stop_process_and_verify_instance(&process, process.pid(), child.identity.as_ref())
                    .await;
            return Err(start_failure_after_cleanup(
                error,
                child,
                cleanup,
                Some("the unready backend launcher required forced termination"),
                "its spawned process could not be cleaned up",
            ));
        }
    };
    let status = match DesktopStatus::from_ready(&outcome, &health) {
        Ok(status) => status,
        Err(error) => {
            let child = UnconfirmedChild {
                process: process.clone(),
                identity: BackendIdentity::from_owned_outcome(&outcome),
            };
            let signal_pid = failed_start_cleanup_pid(&outcome, &health, process.pid());
            let cleanup =
                stop_process_and_verify_instance(&process, signal_pid, child.identity.as_ref())
                    .await;
            return Err(start_failure_after_cleanup(
                error,
                child,
                cleanup,
                Some("the mismatched backend launcher required forced termination"),
                "the mismatched spawned process could not be cleaned up",
            ));
        }
    };
    Ok(StartedBackend { status, process })
}

fn failed_start_cleanup_pid(outcome: &LaunchOutcome, health: &Health, launcher_pid: u32) -> u32 {
    if outcome.outcome == "owned"
        && outcome.owned
        && outcome.instance_id.as_deref() == Some(health.instance_id.as_str())
    {
        health.pid
    } else {
        launcher_pid
    }
}

fn start_failure_after_cleanup(
    error: String,
    child: UnconfirmedChild,
    cleanup: ProcessStop,
    forced_suffix: Option<&str>,
    failed_prefix: &str,
) -> StartFailure {
    match cleanup {
        ProcessStop::Stopped | ProcessStop::UnexpectedExit { .. } => StartFailure::confirmed(error),
        ProcessStop::Forced { .. } => StartFailure::confirmed(match forced_suffix {
            Some(suffix) => format!("{error}; {suffix}"),
            None => error,
        }),
        ProcessStop::StillServing { message } | ProcessStop::Failed { message } => {
            StartFailure::unconfirmed(format!("{error}; {failed_prefix}: {message}"), child)
        }
    }
}

async fn wait_for_launch_outcome(
    startup_rx: &mut tokio::sync::mpsc::UnboundedReceiver<CommandEvent>,
    stderr: &mut Vec<u8>,
    stdout_pending: &mut Vec<u8>,
    deadline: time::Instant,
) -> Result<LaunchOutcome, LaunchWaitFailure> {
    loop {
        // Check before polling the channel so a continuously ready stream cannot
        // win over an already elapsed launch deadline.
        if time::Instant::now() >= deadline {
            return Err(LaunchWaitFailure::TimedOut);
        }
        let event = match time::timeout_at(deadline, startup_rx.recv()).await {
            Ok(Some(event)) => event,
            Ok(None) => return Err(LaunchWaitFailure::Ended),
            Err(_) => return Err(LaunchWaitFailure::TimedOut),
        };
        match event {
            CommandEvent::Stdout(bytes) => {
                if let Some(outcome) = parse_launch_stdout(stdout_pending, stderr, &bytes) {
                    return Ok(outcome);
                }
            }
            CommandEvent::Stderr(bytes) => stderr.extend_from_slice(&bytes),
            CommandEvent::Error(message) => stderr.extend_from_slice(message.as_bytes()),
            _ => {}
        }
    }
}

fn parse_launch_stdout(
    pending: &mut Vec<u8>,
    diagnostics: &mut Vec<u8>,
    bytes: &[u8],
) -> Option<LaunchOutcome> {
    pending.extend_from_slice(bytes);
    while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
        let line = pending.drain(..=newline).collect::<Vec<_>>();
        let text = String::from_utf8_lossy(&line);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if let Ok(outcome) = LaunchOutcome::parse(text) {
            return Some(outcome);
        }
        diagnostics.extend_from_slice(b"[stdout] ");
        diagnostics.extend_from_slice(text.as_bytes());
        diagnostics.push(b'\n');
    }
    None
}

fn backend_command(
    app: &AppHandle,
    force: bool,
) -> Result<tauri_plugin_shell::process::Command, String> {
    let command = if cfg!(debug_assertions) {
        let bundled = dev_bundle_settings()?;
        let (checkout, uv) = match bundled {
            Some(settings) => (
                canonical_directory(&settings.checkout, "RCPDevCheckout in Info.plist")?,
                canonical_file(&settings.uv, "RCPDevUvExecutable in Info.plist")?,
            ),
            None => (dev_checkout()?, dev_uv()?),
        };
        app.shell()
            .command(uv)
            .current_dir(checkout)
            .args(["run", "rcp", "serve"])
    } else {
        app.shell()
            .sidecar("rcp-backend")
            .map_err(|error| format!("packaged backend is unavailable: {error}"))?
            .args(["serve"])
    };

    let mut args = vec![
        "--machine-readable",
        "--owner",
        "desktop",
        "--web-assets",
        if cfg!(debug_assertions) {
            "source"
        } else {
            "prebuilt"
        },
        "--host",
        BACKEND_HOST,
        "--port",
        BACKEND_PORT,
    ];
    args.push(if force { "--force" } else { "--reuse-existing" });
    Ok(command.args(args).env("PATH", repaired_path()?))
}

fn repaired_path() -> Result<std::ffi::OsString, String> {
    let mut entries: Vec<PathBuf> = env::var_os("PATH")
        .map(|value| env::split_paths(&value).collect())
        .unwrap_or_default();
    if let Some(home) = env::var_os("HOME") {
        entries.push(Path::new(&home).join(".local/bin"));
        entries.push(Path::new(&home).join(".npm-global/bin"));
    }
    entries.extend(
        [
            "/opt/homebrew/bin",
            "/opt/homebrew/sbin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    let mut deduplicated = Vec::with_capacity(entries.len());
    for entry in entries {
        if !entry.as_os_str().is_empty() && !deduplicated.contains(&entry) {
            deduplicated.push(entry);
        }
    }
    env::join_paths(deduplicated).map_err(|error| format!("cannot construct backend PATH: {error}"))
}

struct DevBundleSettings {
    checkout: PathBuf,
    uv: PathBuf,
}

/// A bundled source-built `RCP.app` serves its frontend from the backend it
/// launches; only `tauri dev` has a Vite server to load.
pub fn is_bundled_dev_app() -> bool {
    matches!(dev_bundle_settings(), Ok(Some(_)))
}

fn dev_bundle_settings() -> Result<Option<DevBundleSettings>, String> {
    let executable =
        env::current_exe().map_err(|error| format!("cannot locate the RCP executable: {error}"))?;
    let Some(contents) = executable.parent().and_then(Path::parent) else {
        return Ok(None);
    };
    let info_plist = contents.join("Info.plist");
    if !info_plist.is_file() {
        return Ok(None);
    }
    let value = plist::Value::from_file(&info_plist)
        .map_err(|error| format!("cannot read the RCP Info.plist: {error}"))?;
    let dictionary = value
        .as_dictionary()
        .ok_or_else(|| "The RCP Info.plist is not a dictionary".to_string())?;
    let checkout = dictionary
        .get("RCPDevCheckout")
        .and_then(plist::Value::as_string);
    let uv = dictionary
        .get("RCPDevUvExecutable")
        .and_then(plist::Value::as_string);
    match (checkout, uv) {
        (Some(checkout), Some(uv)) => Ok(Some(DevBundleSettings {
            checkout: PathBuf::from(checkout),
            uv: PathBuf::from(uv),
        })),
        (None, None) => Ok(None),
        _ => {
            Err("The RCP Info.plist must record both RCPDevCheckout and RCPDevUvExecutable".into())
        }
    }
}

fn dev_checkout() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("RCP_DEV_CHECKOUT") {
        return canonical_directory(Path::new(&path), "RCP_DEV_CHECKOUT");
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    canonical_directory(
        manifest
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| "cannot resolve the RCP checkout from the Tauri manifest".to_string())?,
        "compiled RCP checkout",
    )
}

fn dev_uv() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("RCP_DEV_UV") {
        return canonical_file(Path::new(&path), "RCP_DEV_UV");
    }
    if let Some(path) = find_on_path("uv") {
        return canonical_file(&path, "uv on PATH");
    }
    if let Some(home) = env::var_os("HOME") {
        let candidate = Path::new(&home).join(".local/bin/uv");
        if candidate.is_file() {
            return canonical_file(&candidate, "uv in ~/.local/bin");
        }
    }
    Err("RCP.app cannot find uv; set RCP_DEV_UV to its absolute path".into())
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH")?
        .to_string_lossy()
        .split(':')
        .map(Path::new)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

fn canonical_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    let resolved = path
        .canonicalize()
        .map_err(|error| format!("cannot resolve {label}: {error}"))?;
    if !resolved.is_file() {
        return Err(format!("{label} is not a file: {}", resolved.display()));
    }
    Ok(resolved)
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    let resolved = path
        .canonicalize()
        .map_err(|error| format!("cannot resolve {label}: {error}"))?;
    if !resolved.is_dir() {
        return Err(format!(
            "{label} is not a directory: {}",
            resolved.display()
        ));
    }
    Ok(resolved)
}

async fn wait_for_health(outcome: &LaunchOutcome) -> Result<Health, String> {
    let expected = outcome
        .instance_id
        .as_deref()
        .ok_or_else(|| "backend launch omitted its instance id".to_string())?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| format!("cannot build health client: {error}"))?;
    let deadline = time::Instant::now() + HEALTH_READY_TIMEOUT;
    let mut last_error = "backend has not answered yet".to_string();
    while time::Instant::now() < deadline {
        match client
            .get(format!("{}/api/health", outcome.base_url))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => match response.json::<Health>().await
            {
                Ok(health) if health.instance_id == expected => return Ok(health),
                Ok(_) => last_error = "another process answered at the backend address".into(),
                Err(error) => last_error = format!("health response was invalid: {error}"),
            },
            Ok(response) => last_error = format!("health returned HTTP {}", response.status()),
            Err(error) => last_error = error.to_string(),
        }
        time::sleep(Duration::from_millis(120)).await;
    }
    Err(format!("backend did not become ready: {last_error}"))
}

pub async fn health(status: &DesktopStatus) -> Result<Health, String> {
    health_at(&status.base_url).await
}

async fn health_at(base_url: &str) -> Result<Health, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .map_err(|error| error.to_string())?
        .get(format!("{base_url}/api/health"))
        .send()
        .await
        .map_err(|error| format!("backend is unreachable: {error}"))?
        .error_for_status()
        .map_err(|error| format!("backend health failed: {error}"))?
        .json::<Health>()
        .await
        .map_err(|error| format!("backend health was invalid: {error}"))
}

pub(crate) async fn reverify_identity(
    state: &BackendState,
    status: &DesktopStatus,
) -> Result<Health, String> {
    let current = health(status).await?;
    if !status.matches_health(&current) {
        return Err("backend identity changed; reconnect before continuing".into());
    }
    state.update_health(&current);
    Ok(current)
}

#[derive(Clone, Debug, serde::Serialize, PartialEq, Eq)]
#[serde(tag = "classification", rename_all = "kebab-case")]
pub enum ShutdownResult {
    SafeNoOp,
    Stopped,
    Forced { message: String },
    OwnershipUnverified { message: String },
    UnexpectedExit { message: String },
    Failed { message: String },
}

impl ShutdownResult {
    pub fn problem(&self) -> Option<&str> {
        match self {
            Self::SafeNoOp | Self::Stopped => None,
            Self::Forced { message }
            | Self::OwnershipUnverified { message }
            | Self::UnexpectedExit { message }
            | Self::Failed { message } => Some(message),
        }
    }

    pub fn is_clean(&self) -> bool {
        matches!(self, Self::SafeNoOp | Self::Stopped)
    }

    pub fn may_exit(&self) -> bool {
        matches!(
            self,
            Self::SafeNoOp | Self::Stopped | Self::Forced { .. } | Self::UnexpectedExit { .. }
        )
    }
}

pub async fn stop_for_quit(state: &BackendState) -> ShutdownResult {
    let _guard = state.coordinator.lock().await;
    if let Some(shutdown) = state.take_deferred_quit_shutdown() {
        return combine_shutdown(stop_unconfirmed_children_locked(state).await, shutdown);
    }
    graceful_stop_locked(state).await
}

async fn graceful_stop_locked(state: &BackendState) -> ShutdownResult {
    let unconfirmed = stop_unconfirmed_children_locked(state).await;
    if !unconfirmed.may_exit() {
        return unconfirmed;
    }
    combine_shutdown(unconfirmed, stop_owned_backend_locked(state).await)
}

async fn stop_unconfirmed_children_locked(state: &BackendState) -> ShutdownResult {
    let children = match state.unconfirmed_children() {
        Ok(children) => children,
        Err(error) => {
            return ShutdownResult::OwnershipUnverified {
                message: format!(
                    "RCP could not read its spawned-child receipts. Quit did not continue: {error}"
                ),
            };
        }
    };
    let mut combined = ShutdownResult::SafeNoOp;
    for child in children {
        let live_health = match child.identity.as_ref() {
            Some(identity) => health_at(&identity.base_url).await.ok(),
            None => None,
        };
        let signal_pid = child.signal_pid(live_health.as_ref());
        let shutdown = shutdown_from_process_stop(
            stop_process_and_verify_instance(&child.process, signal_pid, child.identity.as_ref())
                .await,
        );
        if shutdown.may_exit() {
            state.remove_unconfirmed_child(&child.process);
        }
        combined = combine_shutdown(combined, shutdown);
    }
    combined
}

async fn stop_owned_backend_locked(state: &BackendState) -> ShutdownResult {
    let owned_backend = match state.owned_backend() {
        Ok(Some(backend)) => backend,
        Ok(None) => return ShutdownResult::SafeNoOp,
        Err(error) => {
            return ShutdownResult::OwnershipUnverified {
                message: format!(
                "RCP could not read its owned-backend receipt, so the backend was left running. \
                     Work may not have been paused: {error}"
            ),
            }
        }
    };
    let current = match health_at(&owned_backend.base_url).await {
        Ok(current) => current,
        Err(error) => {
            return shutdown_without_health(&owned_backend, error);
        }
    };
    if !owned_backend.matches_live_instance(&current) {
        return ShutdownResult::SafeNoOp;
    }
    let identity = BackendIdentity {
        instance_id: owned_backend.instance_id.clone(),
        base_url: owned_backend.base_url.clone(),
    };
    shutdown_from_process_stop(
        stop_process_and_verify_instance(&owned_backend.process, current.pid, Some(&identity))
            .await,
    )
}

fn combine_shutdown(left: ShutdownResult, right: ShutdownResult) -> ShutdownResult {
    let message = [left.problem(), right.problem()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n\n");
    if matches!(&left, ShutdownResult::Failed { .. })
        || matches!(&right, ShutdownResult::Failed { .. })
    {
        ShutdownResult::Failed { message }
    } else if matches!(&left, ShutdownResult::OwnershipUnverified { .. })
        || matches!(&right, ShutdownResult::OwnershipUnverified { .. })
    {
        ShutdownResult::OwnershipUnverified { message }
    } else if matches!(&left, ShutdownResult::Forced { .. })
        || matches!(&right, ShutdownResult::Forced { .. })
    {
        ShutdownResult::Forced { message }
    } else if matches!(&left, ShutdownResult::UnexpectedExit { .. })
        || matches!(&right, ShutdownResult::UnexpectedExit { .. })
    {
        ShutdownResult::UnexpectedExit { message }
    } else if matches!(&left, ShutdownResult::Stopped) || matches!(&right, ShutdownResult::Stopped)
    {
        ShutdownResult::Stopped
    } else {
        ShutdownResult::SafeNoOp
    }
}

fn shutdown_without_health(owned_backend: &OwnedBackend, error: String) -> ShutdownResult {
    match owned_backend.process.termination() {
        Some(termination) if clean_termination(&termination) => ShutdownResult::Stopped,
        Some(termination) => ShutdownResult::UnexpectedExit {
            message: format!(
                "The owned backend ended unexpectedly ({}). Work may not have been paused: {error}",
                termination_description(&termination),
            ),
        },
        None => ShutdownResult::OwnershipUnverified {
            message: format!(
                "RCP could not re-verify its owned backend, so it was left running. \
                 Work may not have been paused: {error}"
            ),
        },
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ProcessStop {
    Stopped,
    Forced { message: String },
    UnexpectedExit { message: String },
    StillServing { message: String },
    Failed { message: String },
}

fn shutdown_from_process_stop(result: ProcessStop) -> ShutdownResult {
    match result {
        ProcessStop::Stopped => ShutdownResult::Stopped,
        ProcessStop::Forced { message } => ShutdownResult::Forced { message },
        ProcessStop::UnexpectedExit { message } => ShutdownResult::UnexpectedExit { message },
        ProcessStop::StillServing { message } => ShutdownResult::OwnershipUnverified { message },
        ProcessStop::Failed { message } => ShutdownResult::Failed { message },
    }
}

async fn stop_process_and_verify_instance(
    process: &BackendProcess,
    signal_pid: u32,
    identity: Option<&BackendIdentity>,
) -> ProcessStop {
    let stopped = stop_process(process, signal_pid).await;
    if matches!(&stopped, ProcessStop::Failed { .. }) {
        return stopped;
    }
    let Some(identity) = identity else {
        return stopped;
    };
    let current = health_at(&identity.base_url).await.ok();
    process_stop_after_health_check(stopped, process, identity, current.as_ref())
}

fn process_stop_after_health_check(
    stopped: ProcessStop,
    process: &BackendProcess,
    identity: &BackendIdentity,
    current: Option<&Health>,
) -> ProcessStop {
    if !current.is_some_and(|health| health.instance_id == identity.instance_id) {
        return stopped;
    }
    let process_state = process
        .termination()
        .as_ref()
        .map(termination_description)
        .unwrap_or_else(|| "no process exit was observed".into());
    ProcessStop::StillServing {
        message: format!(
            "The backend launcher {} ended ({process_state}), but its owned backend instance \
             still answers health. RCP cannot confirm that cleanup completed, and work may not \
             have been paused.",
            process.pid(),
        ),
    }
}

async fn stop_process(process: &BackendProcess, signal_pid: u32) -> ProcessStop {
    if let Some(termination) = process.termination() {
        return classify_graceful_termination(&termination);
    }
    process.begin_sigterm_attempt();
    match send_signal(signal_pid, libc::SIGTERM) {
        Ok(delivered) => process.finish_sigterm_delivery(delivered),
        Err(error) => {
            process.finish_sigterm_delivery(false);
            return ProcessStop::Failed {
                message: format!(
                    "The backend process RCP started could not be asked to stop gracefully. \
                     Work may not have been paused: {error}"
                ),
            };
        }
    }
    if let Some(termination) = process.wait(GRACEFUL_STOP_TIMEOUT).await {
        return classify_graceful_termination(&termination);
    }
    if let Err(error) = send_signal(signal_pid, libc::SIGKILL) {
        return ProcessStop::Failed {
            message: format!(
                "Graceful shutdown timed out after {GRACEFUL_STOP_TIMEOUT_SECONDS} seconds and \
                 forced termination could not be requested. Work may not have been paused: {error}"
            ),
        };
    }
    match process.wait(FORCED_STOP_TIMEOUT).await {
        Some(termination) => ProcessStop::Forced {
            message: format!(
                "Graceful shutdown timed out after {GRACEFUL_STOP_TIMEOUT_SECONDS} seconds. \
                 Work may not have been paused; the backend process RCP started required forced termination ({}).",
                termination_description(&termination),
            ),
        },
        None => ProcessStop::Failed {
            message: format!(
                "Graceful shutdown timed out after {GRACEFUL_STOP_TIMEOUT_SECONDS} seconds and \
                 the owned backend did not exit after forced termination. \
                 Work may not have been paused."
            ),
        },
    }
}

fn classify_graceful_termination(termination: &ProcessTermination) -> ProcessStop {
    if clean_termination(termination) {
        ProcessStop::Stopped
    } else {
        ProcessStop::UnexpectedExit {
            message: format!(
                "The backend process RCP started ended non-cleanly during graceful shutdown ({}). \
                 Work may not have been paused.",
                termination_description(termination),
            ),
        }
    }
}

fn clean_termination(termination: &ProcessTermination) -> bool {
    (termination.code == Some(0) && termination.signal.is_none())
        || (termination.code == Some(128 + libc::SIGTERM)
            && termination.signal.is_none()
            && termination.sigterm_requested
            && termination.shutdown_acknowledged)
        || (termination.code.is_none()
            && termination.signal == Some(libc::SIGTERM)
            && termination.sigterm_requested
            && termination.shutdown_acknowledged)
}

fn termination_description(termination: &ProcessTermination) -> String {
    match (termination.code, termination.signal) {
        (Some(code), None) => format!("exit code {code}"),
        (code, Some(signal)) => match code {
            Some(code) => format!("exit code {code}, signal {signal}"),
            None => format!("signal {signal}"),
        },
        (None, None) => "no exit code or signal".into(),
    }
}

fn send_signal(pid: u32, signal: libc::c_int) -> Result<bool, String> {
    let pid = valid_signal_pid(pid)?;
    let result = unsafe { libc::kill(pid, signal) };
    if result == 0 {
        Ok(true)
    } else {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
        {
            Ok(false)
        } else {
            Err(format!("could not signal backend process {pid}: {error}"))
        }
    }
}

fn valid_signal_pid(pid: u32) -> Result<libc::pid_t, String> {
    let pid = libc::pid_t::try_from(pid)
        .map_err(|_| format!("backend process id {pid} is outside the signed pid_t range"))?;
    if pid <= 0 {
        return Err(format!(
            "backend process id {pid} is not a positive process id"
        ));
    }
    Ok(pid)
}

fn launch_error(stderr: &[u8], fallback: &str) -> String {
    let detail = String::from_utf8_lossy(stderr).trim().to_string();
    if detail.is_empty() {
        fallback.to_string()
    } else {
        format!("{fallback}: {detail}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch_outcome(outcome: &str, owned: bool) -> LaunchOutcome {
        LaunchOutcome {
            outcome: outcome.into(),
            base_url: "http://127.0.0.1:8421".into(),
            instance_id: Some("instance-a".into()),
            version: "0.3.0".into(),
            owned,
            reason: None,
        }
    }

    fn ready_health(pid: u32) -> Health {
        Health {
            status: "ok".into(),
            pid,
            version: "0.3.0".into(),
            instance_id: "instance-a".into(),
            data_dir_id: "data-a".into(),
            owner_kind: "desktop".into(),
            active_agent_tasks: 0,
        }
    }

    fn ready_status(instance_id: &str, owned: bool) -> DesktopStatus {
        DesktopStatus {
            desktop: true,
            version: "0.3.0".into(),
            base_url: "http://127.0.0.1:8421".into(),
            instance_id: instance_id.into(),
            data_dir_id: "data-a".into(),
            owner_kind: if owned { "desktop" } else { "cli" }.into(),
            active_agent_tasks: 0,
            owned,
        }
    }

    fn backend_process(pid: u32) -> BackendProcess {
        BackendProcess {
            pid,
            exit: Arc::new(ProcessExit::default()),
        }
    }

    fn terminated_process(pid: u32, code: Option<i32>, signal: Option<i32>) -> BackendProcess {
        terminated_process_with_ack(pid, code, signal, false, false)
    }

    fn terminated_process_with_ack(
        pid: u32,
        code: Option<i32>,
        signal: Option<i32>,
        sigterm_requested: bool,
        shutdown_acknowledged: bool,
    ) -> BackendProcess {
        let process = backend_process(pid);
        if sigterm_requested {
            process.begin_sigterm_attempt();
            process.finish_sigterm_delivery(true);
        }
        if shutdown_acknowledged {
            process.exit.observe_output(SHUTDOWN_ACKNOWLEDGEMENT);
        }
        process
            .exit
            .record_termination(TerminatedPayload { code, signal });
        process
    }

    fn termination(
        code: Option<i32>,
        signal: Option<i32>,
        sigterm_requested: bool,
        shutdown_acknowledged: bool,
    ) -> ProcessTermination {
        ProcessTermination {
            code,
            signal,
            sigterm_requested,
            shutdown_acknowledged,
        }
    }

    #[test]
    fn launch_stdout_skips_build_output_before_the_machine_result() {
        let mut pending = Vec::new();
        let mut diagnostics = Vec::new();
        let first = b"Building frontend...\n{\"outcome\":\"owned\",\"base_url\":\"http://127.0.0.1:8421\",\"instance_id\":\"instance-a\",\"version\":\"0.3.0\",";
        assert!(parse_launch_stdout(&mut pending, &mut diagnostics, first).is_none());

        let outcome = parse_launch_stdout(
            &mut pending,
            &mut diagnostics,
            b"\"owned\":true,\"reason\":null}\n",
        )
        .expect("machine result should be parsed after the build output");

        assert_eq!(outcome.outcome, "owned");
        assert_eq!(outcome.instance_id.as_deref(), Some("instance-a"));
        assert_eq!(diagnostics, b"[stdout] Building frontend...\n");
    }

    #[test]
    fn shutdown_marker_before_sigterm_attempt_is_ignored() {
        let exit = ProcessExit::default();
        exit.observe_output(SHUTDOWN_ACKNOWLEDGEMENT);
        exit.begin_sigterm_attempt();
        exit.finish_sigterm_delivery(true);
        exit.record_termination(TerminatedPayload {
            code: None,
            signal: Some(libc::SIGTERM),
        });

        let termination = exit.termination().expect("termination should be recorded");
        assert!(!termination.shutdown_acknowledged);
        assert!(matches!(
            classify_graceful_termination(&termination),
            ProcessStop::UnexpectedExit { .. }
        ));
    }

    #[test]
    fn successful_sigterm_attempt_accepts_a_split_bounded_acknowledgement() {
        let exit = ProcessExit::default();
        exit.begin_sigterm_attempt();
        exit.finish_sigterm_delivery(true);
        exit.observe_output(&vec![b'x'; SHUTDOWN_ACKNOWLEDGEMENT.len() * 4]);
        assert_eq!(
            exit.state.lock().unwrap().acknowledgement_tail.len(),
            SHUTDOWN_ACKNOWLEDGEMENT.len() - 1
        );

        exit.observe_output(b"noise: Application shutdown ");
        exit.observe_output(b"complete.\n");
        exit.record_termination(TerminatedPayload {
            code: None,
            signal: Some(libc::SIGTERM),
        });

        let termination = exit.termination().expect("termination should be recorded");
        assert!(termination.shutdown_acknowledged);
        assert!(exit.state.lock().unwrap().acknowledgement_tail.is_empty());
        assert_eq!(
            classify_graceful_termination(&termination),
            ProcessStop::Stopped
        );
    }

    #[test]
    fn failed_or_esrch_sigterm_delivery_cannot_leave_a_reusable_attempt() {
        let process = backend_process(4114);
        let failed = tauri::async_runtime::block_on(stop_process(&process, 0));
        assert!(matches!(failed, ProcessStop::Failed { .. }));
        process.exit.observe_output(SHUTDOWN_ACKNOWLEDGEMENT);
        {
            let state = process.exit.state.lock().unwrap();
            assert!(!state.sigterm_attempt_active);
            assert!(!state.sigterm_requested);
            assert!(!state.shutdown_acknowledged);
            assert!(state.acknowledgement_tail.is_empty());
        }

        let esrch = ProcessExit::default();
        esrch.begin_sigterm_attempt();
        esrch.observe_output(SHUTDOWN_ACKNOWLEDGEMENT);
        esrch.finish_sigterm_delivery(false);
        esrch.observe_output(SHUTDOWN_ACKNOWLEDGEMENT);
        let state = esrch.state.lock().unwrap();
        assert!(!state.sigterm_attempt_active);
        assert!(!state.sigterm_requested);
        assert!(!state.shutdown_acknowledged);
        assert!(state.acknowledgement_tail.is_empty());
    }

    #[test]
    fn queued_startup_noise_cannot_extend_an_elapsed_launch_deadline() {
        tauri::async_runtime::block_on(async {
            let (startup_tx, mut startup_rx) = tokio::sync::mpsc::unbounded_channel();
            for _ in 0..32 {
                startup_tx
                    .send(CommandEvent::Stderr(b"still starting\n".to_vec()))
                    .unwrap();
                startup_tx
                    .send(CommandEvent::Stdout(b"Building frontend...\n".to_vec()))
                    .unwrap();
            }
            startup_tx
                .send(CommandEvent::Stdout(
                    b"{\"outcome\":\"owned\",\"base_url\":\"http://127.0.0.1:8421\",\"instance_id\":\"instance-a\",\"version\":\"0.3.0\",\"owned\":true,\"reason\":null}\n"
                        .to_vec(),
                ))
                .unwrap();

            let mut stderr = Vec::new();
            let mut stdout_pending = Vec::new();
            let deadline = time::Instant::now();
            let result = wait_for_launch_outcome(
                &mut startup_rx,
                &mut stderr,
                &mut stdout_pending,
                deadline,
            )
            .await;

            assert!(matches!(result, Err(LaunchWaitFailure::TimedOut)));
            assert!(stderr.is_empty());
            assert!(stdout_pending.is_empty());
        });
    }

    #[test]
    fn failed_start_cleanup_uses_health_pid_only_for_a_matching_owned_identity() {
        let mut health = ready_health(7331);
        assert_eq!(
            failed_start_cleanup_pid(&launch_outcome("owned", true), &health, 4114),
            7331
        );
        health.instance_id = "instance-b".into();
        assert_eq!(
            failed_start_cleanup_pid(&launch_outcome("owned", true), &health, 4114),
            4114
        );
        assert_eq!(
            failed_start_cleanup_pid(&launch_outcome("reused", false), &health, 4114),
            4114
        );
        assert_eq!(
            failed_start_cleanup_pid(&launch_outcome("reused", true), &health, 4114),
            4114
        );
    }

    #[test]
    fn retryable_child_targets_only_its_exact_process_when_health_mismatches() {
        let child = UnconfirmedChild {
            process: backend_process(4114),
            identity: BackendIdentity::from_owned_outcome(&launch_outcome("owned", true)),
        };
        let mut health = ready_health(7331);
        assert_eq!(child.signal_pid(Some(&health)), 7331);

        health.instance_id = "instance-b".into();
        assert_eq!(child.signal_pid(Some(&health)), 4114);
        assert_eq!(child.signal_pid(None), 4114);
    }

    #[test]
    fn pre_outcome_cleanup_failure_preserves_the_exact_child_until_retry() {
        let state = BackendState::default();
        let process = backend_process(4114);
        let child = UnconfirmedChild {
            process: process.clone(),
            identity: None,
        };
        let failure = start_failure_after_cleanup(
            "launch timed out".into(),
            child,
            ProcessStop::Failed {
                message: "still live".into(),
            },
            Some("required forced termination"),
            "cleanup failed",
        );
        state.retain_unconfirmed_child(failure.unconfirmed_child.unwrap());

        let retained = state.unconfirmed_children().unwrap();
        assert_eq!(retained.len(), 1);
        assert!(retained[0].process.same_process(&process));
        assert!(state.has_unconfirmed_children());
    }

    #[test]
    fn confirmed_failed_start_cleanup_does_not_leave_a_retry_receipt() {
        for cleanup in [
            ProcessStop::Stopped,
            ProcessStop::Forced {
                message: "forced".into(),
            },
            ProcessStop::UnexpectedExit {
                message: "nonzero".into(),
            },
        ] {
            let failure = start_failure_after_cleanup(
                "launch failed".into(),
                UnconfirmedChild {
                    process: backend_process(4114),
                    identity: None,
                },
                cleanup,
                Some("required forced termination"),
                "cleanup failed",
            );
            assert!(failure.unconfirmed_child.is_none());
        }
    }

    #[test]
    fn later_shutdown_consumes_a_retry_receipt_only_after_termination() {
        let state = BackendState::default();
        state.retain_unconfirmed_child(UnconfirmedChild {
            process: terminated_process(4114, Some(0), None),
            identity: None,
        });

        let shutdown = tauri::async_runtime::block_on(stop_unconfirmed_children_locked(&state));

        assert_eq!(shutdown, ShutdownResult::Stopped);
        assert!(!state.has_unconfirmed_children());
    }

    #[test]
    fn quit_ownership_is_rederived_from_the_live_instance_id() {
        let state = BackendState::default();
        assert!(state.set_ready(ready_status("instance-a", true), backend_process(4114)));

        let owned = state
            .owned_backend()
            .unwrap()
            .expect("the exact live instance should still be owned");
        assert_eq!(owned.instance_id, "instance-a");
        assert_eq!(owned.process.pid(), 4114);
    }

    #[test]
    fn live_ownership_is_only_the_recorded_instance_id_comparison() {
        let state = BackendState::default();
        assert!(state.set_ready(ready_status("instance-a", true), backend_process(4114)));
        let owned = state.owned_backend().unwrap().unwrap();
        let mut current = ready_health(4114);

        assert!(owned.matches_live_instance(&current));
        current.instance_id = "instance-b".into();
        assert!(!owned.matches_live_instance(&current));
    }

    #[test]
    fn same_instance_reuse_preserves_the_exact_owned_receipt() {
        let state = BackendState::default();
        assert!(state.set_ready(ready_status("instance-a", true), backend_process(4114)));
        let mut reused = ready_status("instance-a", false);
        reused.base_url = "http://127.0.0.1:9000".into();
        assert!(state.set_ready(reused, backend_process(5225)));

        let owned = state.owned_backend().unwrap().unwrap();
        assert_eq!(owned.instance_id, "instance-a");
        assert_eq!(owned.base_url, "http://127.0.0.1:8421");
        assert_eq!(owned.process.pid(), 4114);
    }

    #[test]
    fn reuse_transitions_preserve_only_the_matching_owned_receipt() {
        let state = BackendState::default();
        assert!(state.set_ready(ready_status("instance-a", false), backend_process(4114)));
        assert!(state.owned_backend().unwrap().is_none());

        assert!(state.set_ready(ready_status("instance-b", true), backend_process(5225)));
        assert_eq!(
            state
                .owned_backend()
                .unwrap()
                .expect("a sidecar started after reuse should be owned")
                .process
                .pid(),
            5225
        );

        assert!(state.set_ready(ready_status("instance-c", false), backend_process(6336)));
        assert!(state.owned_backend().unwrap().is_none());
    }

    #[test]
    fn a_failed_unpublished_stop_retains_the_child_receipt_for_retry() {
        let state = BackendState::default();
        let mut status = ready_status("instance-b", true);
        status.base_url = "http://127.0.0.1:9000".into();
        let started = StartedBackend {
            status,
            process: backend_process(5225),
        };

        state.retain_unpublished_owned(&started);

        let owned = state.owned_backend().unwrap().unwrap();
        assert_eq!(owned.instance_id, "instance-b");
        assert_eq!(owned.base_url, "http://127.0.0.1:9000");
        assert_eq!(owned.process.pid(), 5225);
    }

    #[test]
    fn quit_is_terminal_single_flight_and_blocks_late_publication() {
        let state = BackendState::default();
        assert!(state.set_ready(ready_status("instance-a", true), backend_process(4114)));
        assert_eq!(state.begin_quit().unwrap(), QuitRequest::Started);
        assert_eq!(state.begin_quit().unwrap(), QuitRequest::AlreadyQuitting);
        assert!(state.is_quitting());
        assert!(!state.set_ready(ready_status("instance-b", true), backend_process(5225)));
        assert_eq!(state.status().unwrap().instance_id, "instance-a");
        assert_eq!(state.owned_backend().unwrap().unwrap().process.pid(), 4114);
    }

    #[test]
    fn lifecycle_operations_share_one_coordinator() {
        let state = BackendState::default();
        let guard = state.coordinator.try_lock().unwrap();
        assert!(state.coordinator.try_lock().is_err());
        drop(guard);
        assert!(state.coordinator.try_lock().is_ok());
    }

    #[test]
    fn zero_exit_and_acknowledged_sigterm_are_clean() {
        for payload in [
            termination(Some(0), None, false, false),
            termination(Some(0), None, false, true),
            termination(Some(128 + libc::SIGTERM), None, true, true),
            termination(None, Some(libc::SIGTERM), true, true),
        ] {
            assert_eq!(
                classify_graceful_termination(&payload),
                ProcessStop::Stopped
            );
        }
    }

    #[test]
    fn sigterm_requires_both_request_and_shutdown_acknowledgement() {
        for payload in [
            termination(Some(1), None, false, true),
            termination(Some(128 + libc::SIGTERM), None, true, false),
            termination(Some(128 + libc::SIGTERM), None, false, true),
            termination(None, Some(libc::SIGTERM), true, false),
            termination(None, Some(libc::SIGTERM), false, true),
            termination(Some(0), Some(libc::SIGTERM), true, true),
            termination(None, None, false, true),
        ] {
            assert!(matches!(
                classify_graceful_termination(&payload),
                ProcessStop::UnexpectedExit { .. }
            ));
        }
    }

    #[test]
    fn a_supervisor_exit_is_not_clean_while_its_owned_instance_is_live() {
        let process = terminated_process_with_ack(4114, None, Some(libc::SIGTERM), true, true);
        let identity = BackendIdentity {
            instance_id: "instance-a".into(),
            base_url: "http://127.0.0.1:8421".into(),
        };
        let stopped = process_stop_after_health_check(
            ProcessStop::Stopped,
            &process,
            &identity,
            Some(&ready_health(7331)),
        );

        assert!(matches!(stopped, ProcessStop::StillServing { .. }));
        let shutdown = shutdown_from_process_stop(stopped);
        assert!(matches!(
            shutdown,
            ShutdownResult::OwnershipUnverified { .. }
        ));
        assert!(!shutdown.is_clean());
        assert!(!shutdown.may_exit());
    }

    #[test]
    fn signal_pid_must_fit_a_positive_signed_pid_t() {
        assert!(valid_signal_pid(0).is_err());
        assert_eq!(
            valid_signal_pid(i32::MAX as u32),
            Ok(i32::MAX as libc::pid_t)
        );
        assert!(valid_signal_pid((i32::MAX as u32) + 1).is_err());
    }

    #[test]
    fn nonclean_shutdown_classes_are_visible_and_not_clean() {
        let forced = ShutdownResult::Forced {
            message: "forced".into(),
        };
        assert_eq!(forced.problem(), Some("forced"));
        assert!(!forced.is_clean());
        assert!(forced.may_exit());

        let unverified = ShutdownResult::OwnershipUnverified {
            message: "unverified".into(),
        };
        assert_eq!(unverified.problem(), Some("unverified"));
        assert!(!unverified.is_clean());
        assert!(!unverified.may_exit());

        let failed = ShutdownResult::Failed {
            message: "still live".into(),
        };
        assert_eq!(failed.problem(), Some("still live"));
        assert!(!failed.is_clean());
        assert!(!failed.may_exit());

        let unexpected = ShutdownResult::UnexpectedExit {
            message: "signal 15".into(),
        };
        assert_eq!(unexpected.problem(), Some("signal 15"));
        assert!(!unexpected.is_clean());
        assert!(unexpected.may_exit());
    }

    #[test]
    fn unverified_live_receipt_is_distinct_from_an_observed_exit() {
        let live = OwnedBackend {
            instance_id: "instance-a".into(),
            base_url: "http://127.0.0.1:8421".into(),
            process: backend_process(4114),
        };
        assert!(matches!(
            shutdown_without_health(&live, "health timeout".into()),
            ShutdownResult::OwnershipUnverified { .. }
        ));

        let stopped = OwnedBackend {
            instance_id: "instance-a".into(),
            base_url: "http://127.0.0.1:8421".into(),
            process: terminated_process(4114, Some(0), None),
        };
        assert_eq!(
            shutdown_without_health(&stopped, "connection closed".into()),
            ShutdownResult::Stopped
        );

        let unexpected = OwnedBackend {
            instance_id: "instance-a".into(),
            base_url: "http://127.0.0.1:8421".into(),
            process: terminated_process(4114, None, Some(libc::SIGTERM)),
        };
        assert!(matches!(
            shutdown_without_health(&unexpected, "connection closed".into()),
            ShutdownResult::UnexpectedExit { .. }
        ));
    }

    #[test]
    fn updating_owns_the_coordinator_and_rejects_user_quit_until_drop() {
        let state = BackendState::default();
        let guard = tauri::async_runtime::block_on(state.begin_update()).unwrap();

        assert!(state.is_terminal());
        assert!(!state.is_quitting());
        assert_eq!(state.begin_quit().unwrap(), QuitRequest::Updating);
        assert!(!state.set_ready(ready_status("instance-a", true), backend_process(4114)));
        assert!(state.coordinator.try_lock().is_err());

        drop(guard);

        assert!(!state.is_terminal());
        assert_eq!(state.begin_quit().unwrap(), QuitRequest::Started);
        assert!(tauri::async_runtime::block_on(state.begin_update()).is_err());
    }

    #[test]
    fn update_recovery_discards_cached_status_and_ownership() {
        let state = BackendState::default();
        assert!(state.set_ready(ready_status("instance-a", true), backend_process(4114)));

        state.reset_connection_for_recovery().unwrap();

        assert!(state.status().is_err());
        assert!(state.owned_backend().unwrap().is_none());
    }

    #[test]
    fn an_unconfirmed_quit_stays_latched_until_the_warning_is_dismissed() {
        let state = BackendState::default();
        assert_eq!(state.begin_quit().unwrap(), QuitRequest::Started);
        assert_eq!(state.begin_quit().unwrap(), QuitRequest::AlreadyQuitting);
        state.defer_quit_shutdown(ShutdownResult::Failed {
            message: "the child may still be running".into(),
        });

        let shutdown = tauri::async_runtime::block_on(stop_for_quit(&state));

        assert!(!shutdown.may_exit());
        assert!(state.is_quitting());
        assert_eq!(state.begin_quit().unwrap(), QuitRequest::AlreadyQuitting);

        tauri::async_runtime::block_on(state.abort_quit());

        assert!(!state.is_quitting());
        assert_eq!(state.begin_quit().unwrap(), QuitRequest::Started);
    }

    #[test]
    fn graceful_wait_exceeds_the_backend_shutdown_window() {
        assert!(GRACEFUL_STOP_TIMEOUT >= Duration::from_secs(45));
        assert!(GRACEFUL_STOP_TIMEOUT > Duration::from_secs(16 + 16 + 7));
    }
}
