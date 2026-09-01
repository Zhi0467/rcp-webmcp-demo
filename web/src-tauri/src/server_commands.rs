use std::{path::PathBuf, process::Stdio, time::Duration};

use reqwest::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tauri::ipc::Channel;
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    time,
};
use uuid::{Uuid, Version as UuidVersion};

use crate::team_connections::{ServerOperatorMode, ServerOperatorRoute, TeamConnectionMetadata};

const SYSTEM_SSH: &str = "/usr/bin/ssh";
const INSTALLED_RCP: &str = "/usr/local/bin/rcp";
const SYSTEM_SUDO: &str = "/usr/bin/sudo";
const PROBE_REQUEST_ID: &str = "00000000-0000-0000-0000-000000000000";
const PROBE_DIAGNOSTIC: &str = "request id must be a lowercase, hyphenated canonical UUID4";
const PROBE_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_EVENTS: usize = 1_025;
const MAX_STEPS: usize = 256;
const MAX_STDOUT_BYTES: usize = 1024 * 1024;
const MAX_STDERR_BYTES: usize = 64 * 1024;
const TRANSFER_COPY_BUFFER_BYTES: usize = 1024 * 1024;
const TRANSFER_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
const PROVISION_COMMAND: &str = "server project provision";
const TRANSFER_IMPORT_COMMAND: &str = "server project transfer-import";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigureServerOperatorRouteRequest {
    pub connection_id: String,
    pub route: Option<ServerOperatorRoute>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ServerOperatorProbe {
    pub connection_id: String,
    pub available: bool,
    pub route: ServerOperatorRoute,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProjectProvisionReadback {
    pub request_id: String,
    pub target_space_id: String,
    pub status: String,
    pub revision: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerCommandRunResult {
    pub connection_id: String,
    pub request_id: String,
    pub exit_code: i32,
    pub event_count: usize,
    pub readback: ProjectProvisionReadback,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct TerminalLaunchResult {
    pub opened: bool,
    pub argv: Vec<String>,
    pub command: String,
}

pub fn configured_route(
    connection: &TeamConnectionMetadata,
) -> Result<&ServerOperatorRoute, String> {
    connection
        .operator_route
        .as_ref()
        .ok_or_else(|| "this team space has no saved server operator route".to_string())
}

pub async fn probe(
    connection: &TeamConnectionMetadata,
    ssh_program: PathBuf,
) -> Result<ServerOperatorProbe, String> {
    let route = configured_route(connection)?.clone();
    let argv = ssh_argv(&route, PROBE_REQUEST_ID, true);
    let output = time::timeout(
        PROBE_TIMEOUT,
        run_bounded(&ssh_program, &argv, MAX_STDOUT_BYTES),
    )
    .await
    .map_err(|_| "the saved server operator route probe timed out".to_string())??;
    let available = output.exit_code == 2 && output.stderr.contains(PROBE_DIAGNOSTIC);
    Ok(ServerOperatorProbe {
        connection_id: connection.connection_id.clone(),
        available,
        route,
        diagnostic: (!available).then(|| probe_failure_message(output.exit_code)),
    })
}

pub async fn run_project_provision(
    connection: &TeamConnectionMetadata,
    request_id: &str,
    on_event: &Channel<Value>,
    ssh_program: PathBuf,
) -> Result<(i32, usize), String> {
    validate_uuid4(request_id, "project provisioning request identity")?;
    let route = configured_route(connection)?;
    let argv = ssh_argv(route, request_id, true);
    let mut child = command(&ssh_program, &argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not start the saved server operator route: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "the server command stdout pipe is unavailable".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "the server command stderr pipe is unavailable".to_string())?;
    let stderr_task = tauri::async_runtime::spawn(read_capped(stderr, MAX_STDERR_BYTES));
    let events = match stream_events(stdout, on_event).await {
        Ok(events) => events,
        Err(error) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(error);
        }
    };
    let status = child
        .wait()
        .await
        .map_err(|error| format!("could not wait for the server command: {error}"))?;
    let stderr_bytes = stderr_task
        .await
        .map_err(|_| "the server command diagnostic reader stopped".to_string())??;
    let stderr_text = safe_diagnostic(&stderr_bytes);
    if events.is_empty() {
        return Err(format!(
            "the server command returned no structured progress{}",
            diagnostic_suffix(&stderr_text)
        ));
    }
    validate_terminal_events(&events, status.code())?;
    let exit_code = status
        .code()
        .ok_or_else(|| "the server command ended without an exit code".to_string())?;
    Ok((exit_code, events.len()))
}

/// Run the one fixed stdin-only target transfer command.
///
/// The response body is consumed in bounded chunks and is never represented as
/// a Tauri value, command argument, or shell string. The target-side CLI owns
/// digest/size validation and the request-derived inbox path.
pub async fn run_project_transfer_import(
    connection: &TeamConnectionMetadata,
    request_id: &str,
    archive: Response,
    expected_archive_sha256: &str,
    expected_archive_size_bytes: u64,
    on_event: &Channel<Value>,
    ssh_program: PathBuf,
) -> Result<(i32, usize), String> {
    run_project_transfer_import_with_idle_timeout(
        connection,
        request_id,
        archive,
        expected_archive_sha256,
        expected_archive_size_bytes,
        on_event,
        ssh_program,
        TRANSFER_IDLE_TIMEOUT,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_project_transfer_import_with_idle_timeout(
    connection: &TeamConnectionMetadata,
    request_id: &str,
    archive: Response,
    expected_archive_sha256: &str,
    expected_archive_size_bytes: u64,
    on_event: &Channel<Value>,
    ssh_program: PathBuf,
    idle_timeout: Duration,
) -> Result<(i32, usize), String> {
    validate_uuid4(request_id, "project transfer request identity")?;
    validate_archive_receipt(expected_archive_sha256, expected_archive_size_bytes)?;
    if !archive.status().is_success() {
        return Err(format!(
            "the personal transfer archive was rejected (HTTP {})",
            archive.status().as_u16()
        ));
    }
    validate_archive_stream_headers(
        &archive,
        expected_archive_sha256,
        expected_archive_size_bytes,
    )?;
    let route = configured_route(connection)?;
    let argv = transfer_import_ssh_argv(route, request_id, true);
    let mut child = Command::new(&ssh_program)
        .args(&argv)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not start the saved server operator route: {error}"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "the transfer command stdin pipe is unavailable".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "the transfer command stdout pipe is unavailable".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "the transfer command stderr pipe is unavailable".to_string())?;
    let archive_future = pipe_archive(
        archive,
        stdin,
        expected_archive_sha256.to_string(),
        expected_archive_size_bytes,
        idle_timeout,
    );
    let events_future = stream_transfer_events(stdout, on_event);
    tokio::pin!(archive_future);
    tokio::pin!(events_future);
    let stderr_task = tauri::async_runtime::spawn(read_capped(stderr, MAX_STDERR_BYTES));
    enum FirstResult {
        Archive(Result<(), String>),
        Events(Result<Vec<Value>, String>),
    }
    let first = tokio::select! {
        result = &mut archive_future => FirstResult::Archive(result),
        result = &mut events_future => FirstResult::Events(result),
    };
    let events = match first {
        FirstResult::Archive(result) => {
            if let Err(error) = result {
                stop_child(&mut child).await;
                stderr_task.abort();
                let _ = stderr_task.await;
                return Err(error);
            }
            match time::timeout(idle_timeout, &mut events_future).await {
                Ok(Ok(events)) => events,
                Ok(Err(error)) => {
                    stop_child(&mut child).await;
                    stderr_task.abort();
                    let _ = stderr_task.await;
                    return Err(error);
                }
                Err(_) => {
                    stop_child(&mut child).await;
                    stderr_task.abort();
                    let _ = stderr_task.await;
                    return Err(
                        "the transfer command made no progress after receiving the archive".into(),
                    );
                }
            }
        }
        FirstResult::Events(result) => {
            let events = match result {
                Ok(events) => events,
                Err(error) => {
                    stop_child(&mut child).await;
                    stderr_task.abort();
                    let _ = stderr_task.await;
                    return Err(error);
                }
            };
            match time::timeout(idle_timeout, &mut archive_future).await {
                Ok(Ok(())) => events,
                Ok(Err(error)) => {
                    stop_child(&mut child).await;
                    stderr_task.abort();
                    let _ = stderr_task.await;
                    return Err(error);
                }
                Err(_) => {
                    stop_child(&mut child).await;
                    stderr_task.abort();
                    let _ = stderr_task.await;
                    return Err("the transfer archive stream made no progress".into());
                }
            }
        }
    };
    let status = match time::timeout(idle_timeout, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            stop_child(&mut child).await;
            stderr_task.abort();
            let _ = stderr_task.await;
            return Err(format!("could not wait for the transfer command: {error}"));
        }
        Err(_) => {
            stop_child(&mut child).await;
            stderr_task.abort();
            let _ = stderr_task.await;
            return Err("the transfer command did not exit after its final progress event".into());
        }
    };
    let stderr_bytes = match time::timeout(idle_timeout, stderr_task).await {
        Ok(Ok(Ok(stderr))) => stderr,
        Ok(Ok(Err(error))) => return Err(error),
        Ok(Err(_)) => return Err("the transfer command diagnostic reader stopped".into()),
        Err(_) => return Err("the transfer command diagnostic reader did not finish".into()),
    };
    let stderr_text = safe_diagnostic(&stderr_bytes);
    if events.is_empty() {
        return Err(format!(
            "the transfer command returned no structured progress{}",
            diagnostic_suffix(&stderr_text)
        ));
    }
    validate_transfer_terminal_events(&events, status.code())?;
    let exit_code = status
        .code()
        .ok_or_else(|| "the transfer command ended without an exit code".to_string())?;
    Ok((exit_code, events.len()))
}

pub fn terminal_argv(
    connection: &TeamConnectionMetadata,
    request_id: &str,
) -> Result<Vec<String>, String> {
    validate_uuid4(request_id, "project provisioning request identity")?;
    Ok(interactive_ssh_argv(
        configured_route(connection)?,
        request_id,
    ))
}

pub fn terminal_transfer_argv(
    connection: &TeamConnectionMetadata,
    request_id: &str,
    archive_path: &std::path::Path,
) -> Result<Vec<String>, String> {
    validate_uuid4(request_id, "project transfer request identity")?;
    validate_terminal_archive_path(archive_path)?;
    let remote = interactive_transfer_ssh_argv(configured_route(connection)?, request_id);
    let path = archive_path
        .to_str()
        .ok_or_else(|| "the local transfer archive path is not valid UTF-8".to_string())?;
    let pipeline = format!(
        "/bin/cat -- {} | {}",
        shell_quote(path),
        shell_join(&remote)
    );
    Ok(vec!["/bin/sh".into(), "-c".into(), pipeline])
}

#[cfg(target_os = "macos")]
pub async fn open_terminal(argv: Vec<String>) -> Result<TerminalLaunchResult, String> {
    let command_text = shell_join(&argv);
    let script_command = command_text.replace('\\', "\\\\").replace('"', "\\\"");
    let script = format!(
        "tell application \"Terminal\"\nactivate\ndo script \"{script_command}\"\nend tell"
    );
    let status = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map_err(|error| format!("could not open the server command in Terminal: {error}"))?;
    if !status.success() {
        return Err("Terminal refused to open the server command".into());
    }
    Ok(TerminalLaunchResult {
        opened: true,
        argv,
        command: command_text,
    })
}

#[cfg(not(target_os = "macos"))]
pub async fn open_terminal(_argv: Vec<String>) -> Result<TerminalLaunchResult, String> {
    Err("opening an interactive server command is supported only by the macOS desktop".into())
}

fn ssh_argv(route: &ServerOperatorRoute, request_id: &str, noninteractive: bool) -> Vec<String> {
    let mut argv = fixed_ssh_prefix(route, noninteractive);
    argv.extend([
        INSTALLED_RCP.into(),
        "server".into(),
        "project".into(),
        "provision".into(),
        request_id.into(),
        "--machine-readable".into(),
    ]);
    argv
}

fn transfer_import_ssh_argv(
    route: &ServerOperatorRoute,
    request_id: &str,
    noninteractive: bool,
) -> Vec<String> {
    let mut argv = fixed_ssh_prefix(route, noninteractive);
    argv.extend([
        INSTALLED_RCP.into(),
        "server".into(),
        "project".into(),
        "transfer-import".into(),
        request_id.into(),
        "--machine-readable".into(),
    ]);
    argv
}

fn fixed_ssh_prefix(route: &ServerOperatorRoute, noninteractive: bool) -> Vec<String> {
    let mut argv = Vec::new();
    if noninteractive {
        argv.extend([
            "-o".into(),
            "BatchMode=yes".into(),
            "-o".into(),
            "ConnectTimeout=12".into(),
        ]);
    }
    argv.extend(["--".into(), route.ssh_target.clone()]);
    if route.mode == ServerOperatorMode::SudoRcp {
        argv.push(SYSTEM_SUDO.into());
        if noninteractive {
            argv.push("-n".into());
        }
        argv.extend(["-u".into(), "rcp".into(), "-H".into()]);
    }
    argv
}

fn interactive_ssh_argv(route: &ServerOperatorRoute, request_id: &str) -> Vec<String> {
    let mut argv = vec![SYSTEM_SSH.into()];
    argv.extend(ssh_argv(route, request_id, false));
    argv
}

fn interactive_transfer_ssh_argv(route: &ServerOperatorRoute, request_id: &str) -> Vec<String> {
    let mut argv = vec![SYSTEM_SSH.into()];
    argv.extend(transfer_import_ssh_argv(route, request_id, false));
    argv
}

fn command(program: &PathBuf, argv: &[String]) -> Command {
    let mut command = Command::new(program);
    command.args(argv).stdin(Stdio::null());
    command
}

async fn pipe_archive(
    mut response: Response,
    mut stdin: ChildStdin,
    expected_digest: String,
    expected_size: u64,
    idle_timeout: Duration,
) -> Result<(), String> {
    let mut hasher = Sha256::new();
    let mut size = 0_u64;
    while let Some(chunk) = time::timeout(idle_timeout, response.chunk())
        .await
        .map_err(|_| "the personal transfer archive stream made no progress".to_string())?
        .map_err(|error| format!("the personal transfer archive stream failed: {error}"))?
    {
        let chunk_size = u64::try_from(chunk.len())
            .map_err(|_| "the transfer archive size is too large".to_string())?;
        size = size
            .checked_add(chunk_size)
            .ok_or_else(|| "the transfer archive size overflowed".to_string())?;
        if size > expected_size {
            return Err("the personal transfer archive exceeded its durable size".into());
        }
        hasher.update(&chunk);
        // Keep writes bounded even if an HTTP implementation hands us a larger
        // body chunk. The bytes remain in Rust-owned response state only.
        for part in chunk.chunks(TRANSFER_COPY_BUFFER_BYTES) {
            time::timeout(idle_timeout, stdin.write_all(part))
                .await
                .map_err(|_| {
                    "the target transfer command made no progress accepting the archive".to_string()
                })?
                .map_err(|error| {
                    format!("the target transfer command stopped accepting the archive: {error}")
                })?;
        }
    }
    time::timeout(idle_timeout, stdin.shutdown())
        .await
        .map_err(|_| {
            "the target transfer command did not finish accepting the archive".to_string()
        })?
        .map_err(|error| format!("could not close the target transfer archive stream: {error}"))?;
    if size != expected_size || hex_digest(&hasher.finalize()) != expected_digest {
        return Err("the personal transfer archive differs from its durable receipt".into());
    }
    Ok(())
}

async fn stop_child(child: &mut Child) {
    let _ = child.kill().await;
    let _ = child.wait().await;
}

struct BoundedOutput {
    exit_code: i32,
    stderr: String,
}

async fn run_bounded(
    program: &PathBuf,
    argv: &[String],
    stdout_limit: usize,
) -> Result<BoundedOutput, String> {
    let mut child = command(program, argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not start the saved server operator route: {error}"))?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let (stdout_result, stderr_result) = tokio::join!(
        read_capped(stdout, stdout_limit),
        read_capped(stderr, MAX_STDERR_BYTES)
    );
    if let Err(error) = stdout_result {
        let _ = child.kill().await;
        let _ = child.wait().await;
        return Err(error);
    }
    let stderr = stderr_result?;
    let status = child
        .wait()
        .await
        .map_err(|error| format!("could not wait for the server operator probe: {error}"))?;
    Ok(BoundedOutput {
        exit_code: status.code().unwrap_or(-1),
        stderr: safe_diagnostic(&stderr),
    })
}

async fn read_capped(mut reader: impl AsyncRead + Unpin, limit: usize) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        let read = reader
            .read(&mut chunk)
            .await
            .map_err(|error| format!("could not read the server command output: {error}"))?;
        if read == 0 {
            return Ok(output);
        }
        if output.len().saturating_add(read) > limit {
            return Err("the server command exceeded its bounded output limit".into());
        }
        output.extend_from_slice(&chunk[..read]);
    }
}

async fn stream_events(
    reader: impl AsyncRead + Unpin,
    on_event: &Channel<Value>,
) -> Result<Vec<Value>, String> {
    stream_events_inner(reader, on_event, validate_event_prefix).await
}

async fn stream_transfer_events(
    reader: impl AsyncRead + Unpin,
    on_event: &Channel<Value>,
) -> Result<Vec<Value>, String> {
    stream_events_inner(reader, on_event, validate_transfer_event_prefix).await
}

async fn stream_events_inner(
    reader: impl AsyncRead + Unpin,
    on_event: &Channel<Value>,
    validate_prefix: fn(&[Value]) -> Result<(), String>,
) -> Result<Vec<Value>, String> {
    let mut lines = BufReader::new(reader.take((MAX_STDOUT_BYTES + 1) as u64)).lines();
    let mut events = Vec::new();
    let mut bytes = 0_usize;
    while let Some(line) = lines
        .next_line()
        .await
        .map_err(|error| format!("could not read server command progress: {error}"))?
    {
        bytes = bytes.saturating_add(line.len() + 1);
        if bytes > MAX_STDOUT_BYTES {
            return Err("the server command exceeded its bounded output limit".into());
        }
        if line.is_empty() {
            return Err("the server command returned an empty progress event".into());
        }
        reject_credential_shape(&line)?;
        let event: Value = serde_json::from_str(&line)
            .map_err(|_| "the server command returned invalid structured progress".to_string())?;
        events.push(event);
        if events.len() > MAX_EVENTS {
            return Err("the server command returned too many progress events".into());
        }
        validate_prefix(&events)?;
        on_event
            .send(events.last().unwrap().clone())
            .map_err(|_| "the server command progress receiver closed".to_string())?;
    }
    Ok(events)
}

fn validate_event_prefix(events: &[Value]) -> Result<(), String> {
    validate_event_prefix_inner(events, validate_provision_common)
}

fn validate_transfer_event_prefix(events: &[Value]) -> Result<(), String> {
    validate_event_prefix_inner(events, validate_transfer_common)
}

fn validate_event_prefix_inner(
    events: &[Value],
    validate_common: fn(&Value, &str) -> Result<(), String>,
) -> Result<(), String> {
    let Some(first) = events.first() else {
        return Err("the server command returned no structured progress".into());
    };
    validate_common(first, "plan")?;
    let steps = first
        .get("steps")
        .and_then(Value::as_array)
        .ok_or_else(|| "the server command plan has no steps".to_string())?;
    if steps.is_empty() || steps.len() > MAX_STEPS {
        return Err("the server command plan has an invalid step count".into());
    }
    for (index, step) in steps.iter().enumerate() {
        validate_step(step, index + 1, "pending")?;
    }
    let mut latest = vec![None; steps.len()];
    let mut last_number = 0_usize;
    let mut terminated = false;
    for event in &events[1..] {
        validate_common(event, "step")?;
        let step = event
            .get("step")
            .ok_or_else(|| "the server command step event has no step".to_string())?;
        let number = step
            .get("number")
            .and_then(Value::as_u64)
            .ok_or_else(|| "the server command step has no number".to_string())?;
        if number == 0 || number as usize > steps.len() {
            return Err("the server command step is outside its plan".into());
        }
        if terminated {
            return Err("the server command continued after a terminal step".into());
        }
        let number = number as usize;
        if number < last_number {
            return Err("the server command steps moved backwards".into());
        }
        if number > last_number
            && latest[..number - 1]
                .iter()
                .any(|state| state.as_deref() != Some("succeeded"))
        {
            return Err("the server command began a step before earlier steps succeeded".into());
        }
        let state = step
            .get("state")
            .and_then(Value::as_str)
            .ok_or_else(|| "the server command step has no state".to_string())?;
        if state == "pending" {
            return Err("a server command progress step cannot remain pending".into());
        }
        validate_step(step, number, state)?;
        let planned = &steps[number - 1];
        for field in ["title", "purpose", "target", "phase", "expected_success"] {
            if step.get(field) != planned.get(field) {
                return Err(format!("the server command changed its planned {field}"));
            }
        }
        let planned_actor = planned.get("performed_by").and_then(Value::as_str);
        let actor = step.get("performed_by").and_then(Value::as_str);
        if actor != planned_actor
            && !(planned_actor == Some("system")
                && actor == Some("human")
                && state == "operator_action_needed")
        {
            return Err(
                "only an operator-action pause may transfer a server step to a human".into(),
            );
        }
        let previous = latest[number - 1].as_deref();
        if state == "running" && previous.is_some() {
            return Err("the server command started one step more than once".into());
        }
        if state == "succeeded" && previous != Some("running") {
            return Err("the server command succeeded a step before starting it".into());
        }
        if matches!(
            previous,
            Some("succeeded" | "failed" | "operator_action_needed" | "unavailable")
        ) {
            return Err("the server command changed a completed step".into());
        }
        latest[number - 1] = Some(state.to_string());
        last_number = number;
        terminated = matches!(state, "failed" | "operator_action_needed" | "unavailable");
    }
    Ok(())
}

fn validate_terminal_events(events: &[Value], exit_code: Option<i32>) -> Result<(), String> {
    validate_terminal_events_inner(events, exit_code, validate_event_prefix)
}

fn validate_transfer_terminal_events(
    events: &[Value],
    exit_code: Option<i32>,
) -> Result<(), String> {
    validate_terminal_events_inner(events, exit_code, validate_transfer_event_prefix)
}

fn validate_terminal_events_inner(
    events: &[Value],
    exit_code: Option<i32>,
    validate_prefix: fn(&[Value]) -> Result<(), String>,
) -> Result<(), String> {
    validate_prefix(events)?;
    let exit_code = exit_code
        .filter(|code| (0..=125).contains(code))
        .ok_or_else(|| "the server command ended without a valid exit code".to_string())?;
    let final_state = events
        .last()
        .and_then(|event| event.get("step"))
        .and_then(|step| step.get("state"))
        .and_then(Value::as_str)
        .ok_or_else(|| "the server command did not end with a step event".to_string())?;
    if !matches!(
        final_state,
        "succeeded" | "failed" | "operator_action_needed" | "unavailable"
    ) {
        return Err("the server command did not end in a durable terminal state".into());
    }
    let steps = events[0]["steps"].as_array().unwrap();
    let mut latest = vec![None; steps.len()];
    for event in &events[1..] {
        let step = &event["step"];
        let number = step["number"].as_u64().unwrap() as usize;
        latest[number - 1] = step["state"].as_str();
    }
    let all_succeeded = latest.iter().all(|state| *state == Some("succeeded"));
    if exit_code == 0 && !all_succeeded {
        return Err("the server command exited successfully before every step succeeded".into());
    }
    if exit_code != 0 && all_succeeded {
        return Err("the server command completed every step with a failing exit code".into());
    }
    Ok(())
}

fn validate_provision_common(event: &Value, expected: &str) -> Result<(), String> {
    if event.get("version").and_then(Value::as_u64) != Some(1)
        || event.get("event").and_then(Value::as_str) != Some(expected)
        || event.get("command").and_then(Value::as_str) != Some(PROVISION_COMMAND)
        || event
            .get("timestamp")
            .and_then(Value::as_str)
            .is_none_or(|timestamp| timestamp.is_empty() || timestamp.len() > 64)
    {
        return Err("the server command returned an incompatible progress event".into());
    }
    Ok(())
}

fn validate_transfer_common(event: &Value, expected: &str) -> Result<(), String> {
    if event.get("version").and_then(Value::as_u64) != Some(1)
        || event.get("event").and_then(Value::as_str) != Some(expected)
        || event.get("command").and_then(Value::as_str) != Some(TRANSFER_IMPORT_COMMAND)
        || event
            .get("timestamp")
            .and_then(Value::as_str)
            .is_none_or(|timestamp| timestamp.is_empty() || timestamp.len() > 64)
    {
        return Err("the server command returned an incompatible progress event".into());
    }
    Ok(())
}

fn validate_step(step: &Value, expected_number: usize, expected_state: &str) -> Result<(), String> {
    let object = step
        .as_object()
        .ok_or_else(|| "the server command returned an invalid step".to_string())?;
    if object.get("number").and_then(Value::as_u64) != Some(expected_number as u64)
        || object.get("state").and_then(Value::as_str) != Some(expected_state)
        || !matches!(
            expected_state,
            "pending"
                | "running"
                | "succeeded"
                | "failed"
                | "operator_action_needed"
                | "unavailable"
        )
        || !matches!(
            object.get("performed_by").and_then(Value::as_str),
            Some("system" | "human")
        )
    {
        return Err("the server command returned a mismatched step".into());
    }
    for field in [
        "title",
        "purpose",
        "performed_by",
        "phase",
        "expected_success",
        "message",
    ] {
        if object
            .get(field)
            .and_then(Value::as_str)
            .is_none_or(|value| value.is_empty() || value.len() > 4096 || has_control(value))
        {
            return Err("the server command returned invalid step text".into());
        }
    }
    if !object.get("target").is_some_and(Value::is_object)
        || !object.get("actions").is_some_and(Value::is_array)
        || !object.get("fields").is_some_and(Value::is_array)
        || !object.get("resume_argv").is_some_and(Value::is_array)
    {
        return Err("the server command returned an incomplete step".into());
    }
    let actions_empty = object["actions"].as_array().unwrap().is_empty();
    let resume_empty = object["resume_argv"].as_array().unwrap().is_empty();
    if expected_state == "operator_action_needed" {
        if object["performed_by"] != "human" || actions_empty || resume_empty {
            return Err("the server command returned an incomplete operator action".into());
        }
    } else if !actions_empty || !resume_empty {
        return Err("only an operator action may carry actions or a resume command".into());
    }
    Ok(())
}

fn reject_credential_shape(value: &str) -> Result<(), String> {
    for shape in [
        "-----BEGIN ",
        "Bearer ",
        "rcp_bootstrap_",
        "rcp_member_",
        "github_pat_",
        "AGE-SECRET-KEY-1",
    ] {
        if value.contains(shape) {
            return Err("the server command progress contained credential-shaped text".into());
        }
    }
    Ok(())
}

fn safe_diagnostic(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .collect::<String>()
        .trim()
        .to_string()
}

fn diagnostic_suffix(value: &str) -> String {
    if value.is_empty() {
        String::new()
    } else {
        format!(": {value}")
    }
}

fn probe_failure_message(exit_code: i32) -> String {
    format!(
        "The saved SSH operator route could not invoke the fixed RCP command noninteractively (exit {exit_code})."
    )
}

fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|value| shell_quote(value))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&byte))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn has_control(value: &str) -> bool {
    value.chars().any(char::is_control)
}

fn validate_archive_receipt(expected_digest: &str, expected_size: u64) -> Result<(), String> {
    if expected_size == 0
        || expected_digest.len() != 64
        || !expected_digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err("the transfer archive receipt is invalid".into());
    }
    Ok(())
}

fn validate_archive_stream_headers(
    response: &Response,
    expected_digest: &str,
    expected_size: u64,
) -> Result<(), String> {
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or_default().trim());
    if content_type != Some("application/octet-stream") {
        return Err("the personal transfer archive returned an invalid content type".into());
    }
    let content_length = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    if content_length != Some(expected_size) {
        return Err("the personal transfer archive size differs from its durable receipt".into());
    }
    let digest = response
        .headers()
        .get("X-RCP-Archive-SHA256")
        .and_then(|value| value.to_str().ok());
    if digest != Some(expected_digest) {
        return Err("the personal transfer archive digest differs from its durable receipt".into());
    }
    Ok(())
}

fn hex_digest(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_terminal_archive_path(path: &std::path::Path) -> Result<(), String> {
    if !path.is_absolute()
        || path == std::path::Path::new("/")
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err("the local transfer archive path must be one specific absolute path".into());
    }
    let value = path
        .to_str()
        .ok_or_else(|| "the local transfer archive path is not valid UTF-8".to_string())?;
    if value.len() > 4096 || has_control(value) {
        return Err("the local transfer archive path is not bounded and safe".into());
    }
    Ok(())
}

pub fn validate_uuid4(value: &str, label: &str) -> Result<(), String> {
    let parsed =
        Uuid::parse_str(value).map_err(|_| format!("{label} must be a canonical UUID4"))?;
    if parsed.get_version() != Some(UuidVersion::Random) || parsed.to_string() != value {
        return Err(format!(
            "{label} must be a lowercase, hyphenated canonical UUID4"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::{Arc, Mutex},
    };

    use tauri::ipc::InvokeResponseBody;

    use super::*;

    const REQUEST_ID: &str = "11111111-1111-4111-8111-111111111111";

    fn route(mode: ServerOperatorMode) -> ServerOperatorRoute {
        ServerOperatorRoute {
            ssh_target: match mode {
                ServerOperatorMode::DirectRcp => "rcp@lab-server",
                ServerOperatorMode::SudoRcp => "alice@lab-server",
            }
            .into(),
            mode,
        }
    }

    fn connection(mode: ServerOperatorMode) -> TeamConnectionMetadata {
        TeamConnectionMetadata {
            connection_id: "22222222-2222-4222-8222-222222222222".into(),
            display_name: "Vision lab".into(),
            ssh_target: "member@lab-server".into(),
            remote_loopback_port: 8421,
            expected_space_id: "33333333-3333-4333-8333-333333333333".into(),
            local_origin: "https://rcp-22222222222242228222222222222222.localhost:18421".into(),
            minimum_shell_version: "0.3.2".into(),
            last_known_cards: Vec::new(),
            operator_route: Some(route(mode)),
        }
    }

    #[cfg(unix)]
    fn executable_script(body: &str) -> (tempfile::TempDir, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fake-ssh");
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        (directory, path)
    }

    async fn archive_response(
        body: Vec<u8>,
        digest: String,
        declared_size: usize,
        chunk_delay: Duration,
    ) -> (Response, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await.unwrap();
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {declared_size}\r\nX-RCP-Archive-SHA256: {digest}\r\nConnection: close\r\n\r\n"
            );
            stream.write_all(headers.as_bytes()).await.unwrap();
            let split = body.len().div_ceil(2);
            stream.write_all(&body[..split]).await.unwrap();
            if !chunk_delay.is_zero() {
                time::sleep(chunk_delay).await;
            }
            stream.write_all(&body[split..]).await.unwrap();
            let _ = stream.shutdown().await;
        });
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(format!("http://{address}/archive"))
            .send()
            .await
            .unwrap();
        (response, server)
    }

    #[cfg(unix)]
    fn transfer_script(
        plan: &Value,
        running: &Value,
        succeeded: &Value,
        hang_after_stdin: bool,
    ) -> (tempfile::TempDir, PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let capture = directory.path().join("captured-archive");
        let path = directory.path().join("fake-ssh");
        let finish = if hang_after_stdin {
            "/bin/sleep 5".to_string()
        } else {
            format!("printf '%s\\n' {}", shell_quote(&succeeded.to_string()))
        };
        fs::write(
            &path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' {}\nprintf '%s\\n' {}\n/bin/cat > {}\n{finish}\n",
                shell_quote(&plan.to_string()),
                shell_quote(&running.to_string()),
                shell_quote(capture.to_str().unwrap()),
            ),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        (directory, path, capture)
    }

    #[test]
    fn direct_and_sudo_argv_are_fixed_and_noninteractive_only_in_app() {
        assert_eq!(
            ssh_argv(&route(ServerOperatorMode::DirectRcp), REQUEST_ID, true),
            vec![
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=12",
                "--",
                "rcp@lab-server",
                INSTALLED_RCP,
                "server",
                "project",
                "provision",
                REQUEST_ID,
                "--machine-readable",
            ]
        );
        let sudo = ssh_argv(&route(ServerOperatorMode::SudoRcp), REQUEST_ID, true);
        assert_eq!(
            &sudo[6..13],
            [
                SYSTEM_SUDO,
                "-n",
                "-u",
                "rcp",
                "-H",
                INSTALLED_RCP,
                "server"
            ]
        );
        let interactive = ssh_argv(&route(ServerOperatorMode::SudoRcp), REQUEST_ID, false);
        assert!(!interactive.iter().any(|value| value == "BatchMode=yes"));
        assert!(!interactive.iter().any(|value| value == "-n"));
    }

    #[test]
    fn transfer_argv_has_only_the_fixed_stdin_command() {
        assert_eq!(
            transfer_import_ssh_argv(&route(ServerOperatorMode::DirectRcp), REQUEST_ID, true),
            vec![
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=12",
                "--",
                "rcp@lab-server",
                INSTALLED_RCP,
                "server",
                "project",
                "transfer-import",
                REQUEST_ID,
                "--machine-readable",
            ]
        );
        let interactive =
            interactive_transfer_ssh_argv(&route(ServerOperatorMode::SudoRcp), REQUEST_ID);
        assert_eq!(
            shell_join(&interactive),
            "/usr/bin/ssh -- alice@lab-server /usr/bin/sudo -u rcp -H /usr/local/bin/rcp server project transfer-import 11111111-1111-4111-8111-111111111111 --machine-readable"
        );
    }

    #[test]
    fn request_identity_and_terminal_text_cannot_be_shell_input() {
        assert!(validate_uuid4(REQUEST_ID, "request").is_ok());
        assert!(validate_uuid4("$(touch /tmp/no)", "request").is_err());
        let argv = interactive_ssh_argv(&route(ServerOperatorMode::SudoRcp), REQUEST_ID);
        assert_eq!(
            shell_join(&argv),
            "/usr/bin/ssh -- alice@lab-server /usr/bin/sudo -u rcp -H /usr/local/bin/rcp server project provision 11111111-1111-4111-8111-111111111111 --machine-readable"
        );
    }

    #[test]
    fn structured_progress_requires_one_plan_and_terminal_step() {
        let plan = serde_json::json!({
            "version": 1,
            "event": "plan",
            "command": "server project provision",
            "timestamp": "2026-08-30T00:00:00Z",
            "steps": [step(1, "pending")],
        });
        let running = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": "server project provision",
            "timestamp": "2026-08-30T00:00:01Z",
            "step": step(1, "running"),
        });
        let final_event = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": "server project provision",
            "timestamp": "2026-08-30T00:00:02Z",
            "step": step(1, "operator_action_needed"),
        });
        let events = vec![plan, running, final_event];
        validate_terminal_events(&events, Some(75)).unwrap();
        assert!(validate_terminal_events(&events[..1], Some(0)).is_err());
    }

    #[test]
    fn structured_progress_preserves_plan_order_state_and_exit_meaning() {
        let plan = serde_json::json!({
            "version": 1,
            "event": "plan",
            "command": "server project provision",
            "timestamp": "2026-08-30T00:00:00Z",
            "steps": [step(1, "pending"), step(2, "pending")],
        });
        let event = |number, state| {
            serde_json::json!({
                "version": 1,
                "event": "step",
                "command": "server project provision",
                "timestamp": "2026-08-30T00:00:01Z",
                "step": step(number, state),
            })
        };

        assert!(validate_event_prefix(&[plan.clone(), event(2, "running")]).is_err());
        let mut changed = event(1, "running");
        changed["step"]["target"]["host"] = Value::String("other".into());
        assert!(validate_event_prefix(&[plan.clone(), changed]).is_err());
        let complete = vec![
            plan,
            event(1, "running"),
            event(1, "succeeded"),
            event(2, "running"),
            event(2, "succeeded"),
        ];
        validate_terminal_events(&complete, Some(0)).unwrap();
        assert!(validate_terminal_events(&complete, Some(1)).is_err());
    }

    #[test]
    fn transfer_progress_accepts_only_the_fixed_transfer_command() {
        let plan = serde_json::json!({
            "version": 1,
            "event": "plan",
            "command": "server project transfer-import",
            "timestamp": "2026-08-30T00:00:00Z",
            "steps": [step(1, "pending")],
        });
        let running = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": "server project transfer-import",
            "timestamp": "2026-08-30T00:00:01Z",
            "step": step(1, "running"),
        });
        let succeeded = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": "server project transfer-import",
            "timestamp": "2026-08-30T00:00:02Z",
            "step": step(1, "succeeded"),
        });
        let events = vec![plan, running, succeeded];
        validate_transfer_terminal_events(&events, Some(0)).unwrap();
        let mut wrong = events[0].clone();
        wrong["command"] = Value::String(PROVISION_COMMAND.into());
        assert!(validate_transfer_event_prefix(&[wrong]).is_err());
    }

    #[test]
    fn transfer_archive_receipt_is_bounded_and_lowercase() {
        assert!(validate_archive_receipt(&"a".repeat(64), 1).is_ok());
        assert!(validate_archive_receipt(&"A".repeat(64), 1).is_err());
        assert!(validate_archive_receipt(&"a".repeat(63), 1).is_err());
        assert!(validate_archive_receipt(&"a".repeat(64), 0).is_err());
    }

    #[tokio::test]
    async fn one_unterminated_progress_line_is_bounded_before_parsing() {
        let input = vec![b'x'; MAX_STDOUT_BYTES + 1];
        let channel = Channel::<Value>::new(|_| Ok(()));
        let error = stream_events(input.as_slice(), &channel).await.unwrap_err();
        assert!(error.contains("bounded output limit"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn probe_requires_the_fixed_parser_refusal_from_the_configured_route() {
        let (_directory, program) = executable_script(&format!(
            "printf '%s\\n' 'rcp server project provision: error: argument request_id: {PROBE_DIAGNOSTIC}' >&2\nexit 2"
        ));
        let result = probe(&connection(ServerOperatorMode::DirectRcp), program)
            .await
            .unwrap();

        assert!(result.available);
        assert_eq!(result.route.mode, ServerOperatorMode::DirectRcp);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_streams_only_bounded_project_provision_events() {
        let plan = serde_json::json!({
            "version": 1,
            "event": "plan",
            "command": "server project provision",
            "timestamp": "2026-08-30T00:00:00Z",
            "steps": [step(1, "pending")],
        });
        let running = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": "server project provision",
            "timestamp": "2026-08-30T00:00:01Z",
            "step": step(1, "running"),
        });
        let final_event = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": "server project provision",
            "timestamp": "2026-08-30T00:00:02Z",
            "step": step(1, "operator_action_needed"),
        });
        let (_directory, program) = executable_script(&format!(
            "printf '%s\\n' '{}'\nprintf '%s\\n' '{}'\nprintf '%s\\n' '{}'\nexit 75",
            plan, running, final_event
        ));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let channel_capture = captured.clone();
        let channel = Channel::<Value>::new(move |body| {
            if let InvokeResponseBody::Json(json) = body {
                channel_capture.lock().unwrap().push(json);
            }
            Ok(())
        });

        let result = run_project_provision(
            &connection(ServerOperatorMode::SudoRcp),
            REQUEST_ID,
            &channel,
            program,
        )
        .await
        .unwrap();

        assert_eq!(result, (75, 3));
        assert_eq!(captured.lock().unwrap().len(), 3);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn transfer_relay_streams_exact_bytes_and_accepts_progressive_body_chunks() {
        let body = b"request-bound transfer archive".to_vec();
        let digest = hex_digest(&Sha256::digest(&body));
        let (response, server) = archive_response(
            body.clone(),
            digest.clone(),
            body.len(),
            Duration::from_millis(40),
        )
        .await;
        let plan = serde_json::json!({
            "version": 1,
            "event": "plan",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:00Z",
            "steps": [step(1, "pending")],
        });
        let running = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:01Z",
            "step": step(1, "running"),
        });
        let succeeded = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:02Z",
            "step": step(1, "succeeded"),
        });
        let (_directory, program, capture) = transfer_script(&plan, &running, &succeeded, false);
        let channel = Channel::<Value>::new(|_| Ok(()));

        let result = run_project_transfer_import_with_idle_timeout(
            &connection(ServerOperatorMode::DirectRcp),
            REQUEST_ID,
            response,
            &digest,
            body.len() as u64,
            &channel,
            program,
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(result, (0, 3));
        assert_eq!(fs::read(capture).unwrap(), body);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn transfer_relay_kills_a_target_that_stalls_after_stdin() {
        let body = b"request-bound transfer archive".to_vec();
        let digest = hex_digest(&Sha256::digest(&body));
        let (response, server) =
            archive_response(body.clone(), digest.clone(), body.len(), Duration::ZERO).await;
        let plan = serde_json::json!({
            "version": 1,
            "event": "plan",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:00Z",
            "steps": [step(1, "pending")],
        });
        let running = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:01Z",
            "step": step(1, "running"),
        });
        let succeeded = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:02Z",
            "step": step(1, "succeeded"),
        });
        let (_directory, program, _capture) = transfer_script(&plan, &running, &succeeded, true);
        let channel = Channel::<Value>::new(|_| Ok(()));

        let error = run_project_transfer_import_with_idle_timeout(
            &connection(ServerOperatorMode::DirectRcp),
            REQUEST_ID,
            response,
            &digest,
            body.len() as u64,
            &channel,
            program,
            Duration::from_millis(50),
        )
        .await
        .unwrap_err();
        server.await.unwrap();

        assert!(error.contains("no progress"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn transfer_relay_rejects_a_truncated_archive_stream() {
        let body = b"truncated transfer archive".to_vec();
        let expected_size = body.len() + 1;
        let digest = hex_digest(&Sha256::digest(&body));
        let (response, server) =
            archive_response(body.clone(), digest.clone(), expected_size, Duration::ZERO).await;
        let plan = serde_json::json!({
            "version": 1,
            "event": "plan",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:00Z",
            "steps": [step(1, "pending")],
        });
        let running = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:01Z",
            "step": step(1, "running"),
        });
        let succeeded = serde_json::json!({
            "version": 1,
            "event": "step",
            "command": TRANSFER_IMPORT_COMMAND,
            "timestamp": "2026-08-30T00:00:02Z",
            "step": step(1, "succeeded"),
        });
        let (_directory, program, _capture) = transfer_script(&plan, &running, &succeeded, false);
        let channel = Channel::<Value>::new(|_| Ok(()));

        let error = run_project_transfer_import_with_idle_timeout(
            &connection(ServerOperatorMode::DirectRcp),
            REQUEST_ID,
            response,
            &digest,
            expected_size as u64,
            &channel,
            program,
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        server.await.unwrap();

        assert!(error.contains("stream failed") || error.contains("durable receipt"));
    }

    fn step(number: usize, state: &str) -> Value {
        let operator_action = state == "operator_action_needed";
        serde_json::json!({
            "number": number,
            "title": "Prepare project",
            "purpose": "Prepare the checkout.",
            "performed_by": if operator_action { "human" } else { "system" },
            "target": {"kind": "machine", "host": "lab", "os_account": "rcp"},
            "phase": "prepare",
            "state": state,
            "expected_success": "The checkout is ready.",
            "message": "RCP is preparing the checkout.",
            "actions": if operator_action {
                serde_json::json!([{"kind": "command", "instruction": "Repair the server."}])
            } else {
                serde_json::json!([])
            },
            "fields": [],
            "resume_argv": if operator_action {
                serde_json::json!(["rcp", "server", "project", "provision", REQUEST_ID])
            } else {
                serde_json::json!([])
            },
        })
    }
}
