use std::{
    collections::{HashMap, HashSet},
    io,
    net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use rustls::{pki_types::PrivateKeyDer, ServerConfig};
use serde::Serialize;
use socket2::{Domain, Protocol, Socket, Type};
use tokio::{
    io::copy_bidirectional,
    net::{TcpListener, TcpStream},
    process::{Child, Command},
    sync::{oneshot, watch, Mutex},
    task::{JoinHandle, JoinSet},
    time::{self, Instant},
};
use tokio_rustls::TlsAcceptor;
use url::Url;

use crate::{
    backend::BackendState,
    local_https::LocalHttpsIdentity,
    team_connections::{
        allocate_local_origin, validate_ssh_target, RemovalResult, TeamConnectionMetadata,
        TeamConnectionState,
    },
};

const SYSTEM_SSH: &str = "/usr/bin/ssh";
const SSH_READY_TIMEOUT: Duration = Duration::from_secs(12);
const CHILD_STOP_TIMEOUT: Duration = Duration::from_secs(2);
const RECONNECT_BACKOFF: Duration = Duration::from_secs(2);
const FORWARD_POLL_INTERVAL: Duration = Duration::from_millis(75);

#[derive(Clone)]
pub struct TeamTunnelState {
    runtime: Arc<Mutex<TunnelRuntime>>,
    tls_config: Arc<ServerConfig>,
    ssh_program: PathBuf,
}

struct TunnelRuntime {
    active: HashMap<String, ActiveTunnel>,
    retry_after: HashMap<String, Instant>,
    retired: HashSet<String>,
    accepting: bool,
}

impl Default for TunnelRuntime {
    fn default() -> Self {
        Self {
            active: HashMap::new(),
            retry_after: HashMap::new(),
            retired: HashSet::new(),
            accepting: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TeamTunnelReady {
    pub connection_id: String,
    pub local_origin: String,
    pub reused: bool,
}

struct ActiveTunnel {
    ready: TeamTunnelReady,
    route: TunnelRoute,
    child: OwnedChild,
    proxy_tasks: Vec<JoinHandle<()>>,
}

struct StartTunnelFailure {
    message: String,
    retained: Option<ActiveTunnel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TunnelRoute {
    local_origin: String,
    ssh_target: String,
    remote_loopback_port: u16,
}

impl From<&TeamConnectionMetadata> for TunnelRoute {
    fn from(connection: &TeamConnectionMetadata) -> Self {
        Self {
            local_origin: connection.local_origin.clone(),
            ssh_target: connection.ssh_target.clone(),
            remote_loopback_port: connection.remote_loopback_port,
        }
    }
}

impl TeamTunnelState {
    pub fn new(identity: &LocalHttpsIdentity) -> Result<Self, String> {
        Self::with_ssh_program(identity, PathBuf::from(SYSTEM_SSH))
    }

    fn with_ssh_program(
        identity: &LocalHttpsIdentity,
        ssh_program: PathBuf,
    ) -> Result<Self, String> {
        Ok(Self {
            runtime: Arc::new(Mutex::new(TunnelRuntime::default())),
            tls_config: tls_server_config(identity)?,
            ssh_program,
        })
    }

    pub async fn connect_saved(
        &self,
        connections: &TeamConnectionState,
        lifecycle: &BackendState,
        connection_id: &str,
    ) -> Result<TeamTunnelReady, String> {
        let mut runtime = self.runtime.lock().await;
        if !runtime.accepting || lifecycle.is_terminal() {
            return Err("RCP cannot connect a team space while quitting or updating".into());
        }
        if runtime.retired.contains(connection_id) {
            return Err("the team connection is being removed from this desktop".into());
        }
        let connection = connections
            .list()?
            .into_iter()
            .find(|connection| connection.connection_id == connection_id)
            .ok_or_else(|| "the team connection is not saved on this desktop".to_string())?;
        self.connect_locked(&mut runtime, lifecycle, connection)
            .await
    }

    pub(crate) async fn connect_candidate(
        &self,
        lifecycle: &BackendState,
        connection_id: &str,
        ssh_target: &str,
        remote_loopback_port: u16,
    ) -> Result<TeamTunnelReady, String> {
        validate_ssh_target(ssh_target)?;
        if remote_loopback_port == 0 {
            return Err("remote team server port must be a positive integer".into());
        }
        let mut runtime = self.runtime.lock().await;
        if !runtime.accepting || lifecycle.is_terminal() {
            return Err("RCP cannot connect a team space while quitting or updating".into());
        }
        if runtime.retired.contains(connection_id) || runtime.active.contains_key(connection_id) {
            return Err("the pending team connection identity is already in use".into());
        }
        let listeners = bind_local_https(0)?;
        let port = listeners[0]
            .local_addr()
            .map_err(|error| format!("could not inspect the pending team origin: {error}"))?
            .port();
        let route = TunnelRoute {
            local_origin: allocate_local_origin(connection_id, port)?,
            ssh_target: ssh_target.to_string(),
            remote_loopback_port,
        };
        match self
            .start_tunnel(connection_id, &route, Some(listeners))
            .await
        {
            Ok(active) => {
                let ready = active.ready.clone();
                runtime.active.insert(connection_id.to_string(), active);
                if lifecycle.is_terminal() {
                    let active = runtime
                        .active
                        .get_mut(connection_id)
                        .expect("pending team tunnel disappeared before lifecycle fencing");
                    let stopped = active.stop().await;
                    if stopped.is_ok() {
                        runtime.active.remove(connection_id);
                    }
                    return Err(stopped.err().unwrap_or_else(|| {
                        "RCP began quitting or updating while the team tunnel connected".into()
                    }));
                }
                Ok(ready)
            }
            Err(error) => {
                let StartTunnelFailure { message, retained } = *error;
                if let Some(active) = retained {
                    runtime.active.insert(connection_id.to_string(), active);
                }
                Err(message)
            }
        }
    }

    pub(crate) async fn stop_candidate(&self, connection_id: &str) -> Result<(), String> {
        let mut runtime = self.runtime.lock().await;
        runtime.retry_after.remove(connection_id);
        if let Some(active) = runtime.active.get_mut(connection_id) {
            active.stop().await?;
            runtime.active.remove(connection_id);
        }
        Ok(())
    }

    async fn connect_locked(
        &self,
        runtime: &mut TunnelRuntime,
        lifecycle: &BackendState,
        connection: TeamConnectionMetadata,
    ) -> Result<TeamTunnelReady, String> {
        let route = TunnelRoute::from(&connection);
        if let Some(active) = runtime.active.get_mut(&connection.connection_id) {
            if active.healthy() && active.route == route {
                let mut ready = active.ready.clone();
                ready.reused = true;
                return Ok(ready);
            }
        }

        if let Some(stale) = runtime.active.get_mut(&connection.connection_id) {
            stale.stop().await?;
            runtime.active.remove(&connection.connection_id);
            time::sleep(RECONNECT_BACKOFF).await;
        }
        if let Some(retry_after) = runtime.retry_after.get(&connection.connection_id) {
            if *retry_after > Instant::now() {
                return Err("the team tunnel is waiting briefly before reconnecting".into());
            }
        }
        runtime.retry_after.remove(&connection.connection_id);

        match self
            .start_tunnel(&connection.connection_id, &route, None)
            .await
        {
            Ok(active) => {
                let ready = active.ready.clone();
                runtime
                    .active
                    .insert(connection.connection_id.clone(), active);
                if lifecycle.is_terminal() {
                    let active = runtime
                        .active
                        .get_mut(&connection.connection_id)
                        .expect("new team tunnel disappeared before lifecycle fencing");
                    let stopped = active.stop().await;
                    if stopped.is_ok() {
                        runtime.active.remove(&connection.connection_id);
                    }
                    return Err(stopped.err().unwrap_or_else(|| {
                        "RCP began quitting or updating while the team tunnel connected".into()
                    }));
                }
                Ok(ready)
            }
            Err(error) => {
                let StartTunnelFailure { message, retained } = *error;
                if let Some(active) = retained {
                    runtime
                        .active
                        .insert(connection.connection_id.clone(), active);
                }
                runtime
                    .retry_after
                    .insert(connection.connection_id, Instant::now() + RECONNECT_BACKOFF);
                Err(message)
            }
        }
    }

    pub async fn remove_saved_connection(
        &self,
        connections: &TeamConnectionState,
        connection_id: &str,
    ) -> Result<RemovalResult, String> {
        let mut runtime = self.runtime.lock().await;
        if !connections
            .list()?
            .iter()
            .any(|connection| connection.connection_id == connection_id)
        {
            return connections.remove_metadata(connection_id);
        }
        runtime.retired.insert(connection_id.to_string());
        runtime.retry_after.remove(connection_id);
        if let Some(active) = runtime.active.get_mut(connection_id) {
            active.stop().await?;
            runtime.active.remove(connection_id);
        }
        connections.remove_metadata(connection_id)
    }

    pub async fn stop_all_for_lifecycle(&self) -> Result<(), String> {
        let mut runtime = self.runtime.lock().await;
        runtime.accepting = false;
        stop_all_locked(&mut runtime).await
    }

    pub async fn resume_after_lifecycle_failure(&self) {
        self.runtime.lock().await.accepting = true;
    }

    #[cfg(test)]
    async fn stop_all_for_test(&self) -> Result<(), String> {
        let mut runtime = self.runtime.lock().await;
        stop_all_locked(&mut runtime).await
    }

    async fn start_tunnel(
        &self,
        connection_id: &str,
        route: &TunnelRoute,
        listeners: Option<Vec<TcpListener>>,
    ) -> Result<ActiveTunnel, Box<StartTunnelFailure>> {
        let listeners = match listeners {
            Some(listeners) => listeners,
            None => {
                let local_https_port = local_https_port(&route.local_origin)
                    .map_err(|error| Box::new(StartTunnelFailure::from(error)))?;
                bind_local_https(local_https_port)
                    .map_err(|error| Box::new(StartTunnelFailure::from(error)))?
            }
        };
        let forward_port =
            reserve_forward_port().map_err(|error| Box::new(StartTunnelFailure::from(error)))?;
        let arguments = ssh_arguments(route, forward_port);
        let mut child = spawn_owned_child(&self.ssh_program, &arguments)
            .map_err(|error| Box::new(StartTunnelFailure::from(error)))?;
        if let Err(error) = wait_for_forward(forward_port, &mut child.child_exited).await {
            return Err(Box::new(
                failed_start(connection_id, route, child, error).await,
            ));
        }
        if *child.child_exited.borrow() {
            return Err(Box::new(
                failed_start(
                    connection_id,
                    route,
                    child,
                    "the SSH tunnel ended before its local HTTPS proxy started".into(),
                )
                .await,
            ));
        }
        let proxy_tasks = start_tls_proxies(
            listeners,
            self.tls_config.clone(),
            forward_port,
            child.child_exited.clone(),
        );
        Ok(ActiveTunnel {
            ready: TeamTunnelReady {
                connection_id: connection_id.to_string(),
                local_origin: route.local_origin.clone(),
                reused: false,
            },
            route: route.clone(),
            child,
            proxy_tasks,
        })
    }
}

impl From<String> for StartTunnelFailure {
    fn from(message: String) -> Self {
        Self {
            message,
            retained: None,
        }
    }
}

async fn failed_start(
    connection_id: &str,
    route: &TunnelRoute,
    mut child: OwnedChild,
    start_error: String,
) -> StartTunnelFailure {
    match child.stop().await {
        Ok(()) => StartTunnelFailure {
            message: start_error,
            retained: None,
        },
        Err(cleanup_error) => StartTunnelFailure {
            message: format!("{start_error}; {cleanup_error}"),
            retained: Some(ActiveTunnel {
                ready: TeamTunnelReady {
                    connection_id: connection_id.to_string(),
                    local_origin: route.local_origin.clone(),
                    reused: false,
                },
                route: route.clone(),
                child,
                proxy_tasks: Vec::new(),
            }),
        },
    }
}

async fn stop_all_locked(runtime: &mut TunnelRuntime) -> Result<(), String> {
    runtime.retry_after.clear();
    let mut problems = Vec::new();
    let connection_ids = runtime.active.keys().cloned().collect::<Vec<_>>();
    for connection_id in connection_ids {
        let tunnel = runtime
            .active
            .get_mut(&connection_id)
            .expect("team tunnel disappeared during stop");
        if let Err(error) = tunnel.stop().await {
            problems.push(error);
        } else {
            runtime.active.remove(&connection_id);
        }
    }
    if !problems.is_empty() {
        return Err(problems.join("; "));
    }
    Ok(())
}

impl ActiveTunnel {
    fn healthy(&mut self) -> bool {
        !*self.child.child_exited.borrow_and_update()
            && self.child.stop_failure.is_none()
            && self.child.supervisor.is_some()
            && !self.proxy_tasks.is_empty()
            && self.proxy_tasks.iter().all(|task| !task.is_finished())
    }

    async fn stop(&mut self) -> Result<(), String> {
        for task in &self.proxy_tasks {
            task.abort();
        }
        for task in self.proxy_tasks.drain(..) {
            let _ = task.await;
        }
        self.child.stop().await
    }
}

struct OwnedChild {
    pid: u32,
    child_exited: watch::Receiver<bool>,
    stop_child: Option<oneshot::Sender<()>>,
    supervisor: Option<JoinHandle<Result<(), String>>>,
    stop_failure: Option<String>,
}

impl OwnedChild {
    async fn stop(&mut self) -> Result<(), String> {
        if let Some(error) = &self.stop_failure {
            return Err(error.clone());
        }
        if let Some(stop) = self.stop_child.take() {
            let _ = stop.send(());
        }
        let Some(supervisor) = self.supervisor.as_mut() else {
            return Ok(());
        };
        match time::timeout(CHILD_STOP_TIMEOUT * 2, supervisor).await {
            Ok(Ok(Ok(()))) => {
                self.supervisor.take();
                Ok(())
            }
            Ok(Ok(Err(error))) => {
                self.supervisor.take();
                let error = format!("SSH tunnel process group {}: {error}", self.pid);
                self.stop_failure = Some(error.clone());
                Err(error)
            }
            Ok(Err(error)) => {
                self.supervisor.take();
                let error = format!(
                    "SSH tunnel process group {} supervisor failed: {error}",
                    self.pid
                );
                self.stop_failure = Some(error.clone());
                Err(error)
            }
            Err(_) => Err("the desktop could not confirm that its SSH tunnel stopped".into()),
        }
    }
}

fn ssh_arguments(route: &TunnelRoute, forward_port: u16) -> Vec<String> {
    vec![
        "-N".into(),
        "-T".into(),
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "ExitOnForwardFailure=yes".into(),
        "-o".into(),
        "ConnectTimeout=10".into(),
        "-o".into(),
        "ServerAliveInterval=15".into(),
        "-o".into(),
        "ServerAliveCountMax=3".into(),
        "-o".into(),
        "ControlMaster=no".into(),
        "-o".into(),
        "ControlPath=none".into(),
        "-o".into(),
        "ControlPersist=no".into(),
        "-o".into(),
        "ForkAfterAuthentication=no".into(),
        "-L".into(),
        format!(
            "127.0.0.1:{forward_port}:127.0.0.1:{}",
            route.remote_loopback_port
        ),
        route.ssh_target.clone(),
    ]
}

fn spawn_owned_child(program: &Path, arguments: &[String]) -> Result<OwnedChild, String> {
    let mut command = Command::new(program);
    command
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .process_group(0);
    let child = command
        .spawn()
        .map_err(|error| format!("could not start the system SSH client: {error}"))?;
    let pid = child
        .id()
        .ok_or_else(|| "the SSH tunnel process has no identity".to_string())?;
    let (stop_child, stop_request) = oneshot::channel();
    let (child_exit, child_exited) = watch::channel(false);
    let supervisor = tokio::spawn(supervise_child(child, stop_request, child_exit));
    Ok(OwnedChild {
        pid,
        child_exited,
        stop_child: Some(stop_child),
        supervisor: Some(supervisor),
        stop_failure: None,
    })
}

async fn supervise_child(
    mut child: Child,
    mut stop_request: oneshot::Receiver<()>,
    child_exit: watch::Sender<bool>,
) -> Result<(), String> {
    let pid = child
        .id()
        .ok_or_else(|| "the SSH tunnel process has no identity".to_string())?;
    let result = tokio::select! {
        status = child.wait() => status
            .map(|_| ())
            .map_err(|error| format!("could not observe the SSH tunnel process: {error}")),
        _ = &mut stop_request => stop_process_group(&mut child, pid).await,
    };
    let _ = child_exit.send(true);
    result
}

async fn stop_process_group(child: &mut Child, pid: u32) -> Result<(), String> {
    signal_process_group(pid, libc::SIGTERM)?;
    match time::timeout(CHILD_STOP_TIMEOUT, child.wait()).await {
        Ok(Ok(_)) => return Ok(()),
        Ok(Err(error)) => {
            return Err(format!("could not reap the SSH tunnel process: {error}"));
        }
        Err(_) => {}
    }
    signal_process_group(pid, libc::SIGKILL)?;
    time::timeout(CHILD_STOP_TIMEOUT, child.wait())
        .await
        .map_err(|_| "the SSH tunnel process did not stop after SIGKILL".to_string())?
        .map_err(|error| format!("could not reap the SSH tunnel process: {error}"))?;
    Ok(())
}

fn signal_process_group(pid: u32, signal: libc::c_int) -> Result<(), String> {
    let pid = i32::try_from(pid).map_err(|_| "the SSH tunnel process identity is invalid")?;
    let result = unsafe { libc::kill(-pid, signal) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(format!("could not stop the SSH tunnel process: {error}"))
    }
}

async fn wait_for_forward(
    forward_port: u16,
    child_exited: &mut watch::Receiver<bool>,
) -> Result<(), String> {
    let deadline = Instant::now() + SSH_READY_TIMEOUT;
    loop {
        if *child_exited.borrow_and_update() {
            return Err("the SSH tunnel ended before its local forward became ready".into());
        }
        if TcpStream::connect((Ipv4Addr::LOCALHOST, forward_port))
            .await
            .is_ok()
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err("timed out waiting for the SSH tunnel's local forward".into());
        }
        tokio::select! {
            _ = time::sleep(FORWARD_POLL_INTERVAL) => {}
            changed = child_exited.changed() => {
                if changed.is_err() || *child_exited.borrow_and_update() {
                    return Err("the SSH tunnel ended before its local forward became ready".into());
                }
            }
        }
    }
}

fn tls_server_config(identity: &LocalHttpsIdentity) -> Result<Arc<ServerConfig>, String> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let private_key = PrivateKeyDer::try_from(identity.private_key_der())
        .map_err(|_| "the local HTTPS private key is invalid".to_string())?
        .clone_key();
    let mut config = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| format!("could not configure local HTTPS: {error}"))?
        .with_no_client_auth()
        .with_single_cert(
            vec![rustls::pki_types::CertificateDer::from(
                identity.certificate_der().to_vec(),
            )],
            private_key,
        )
        .map_err(|error| format!("could not configure the local HTTPS identity: {error}"))?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

fn local_https_port(origin: &str) -> Result<u16, String> {
    Url::parse(origin)
        .ok()
        .and_then(|url| url.port())
        .ok_or_else(|| "the saved team origin has no explicit local HTTPS port".to_string())
}

fn bind_local_https(port: u16) -> Result<Vec<TcpListener>, String> {
    let ipv6 = bind_listener(
        SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::LOCALHOST, port, 0, 0)),
        true,
    )?;
    let selected_port = ipv6
        .local_addr()
        .map_err(|error| format!("could not inspect the local HTTPS listener: {error}"))?
        .port();
    let ipv4 = bind_listener(
        SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, selected_port)),
        false,
    )?;
    Ok(vec![ipv4, ipv6])
}

fn bind_listener(address: SocketAddr, ipv6_only: bool) -> Result<TcpListener, String> {
    let domain = if address.is_ipv6() {
        Domain::IPV6
    } else {
        Domain::IPV4
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))
        .map_err(|error| format!("could not create the local HTTPS listener: {error}"))?;
    if address.is_ipv6() {
        socket
            .set_only_v6(ipv6_only)
            .map_err(|error| format!("could not isolate the IPv6 HTTPS listener: {error}"))?;
    }
    socket
        .bind(&address.into())
        .map_err(|error| format!("could not bind local HTTPS at {address}: {error}"))?;
    socket
        .listen(128)
        .map_err(|error| format!("could not listen for local HTTPS at {address}: {error}"))?;
    socket
        .set_nonblocking(true)
        .map_err(|error| format!("could not configure local HTTPS at {address}: {error}"))?;
    let listener: std::net::TcpListener = socket.into();
    TcpListener::from_std(listener)
        .map_err(|error| format!("could not activate local HTTPS at {address}: {error}"))
}

fn reserve_forward_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|error| format!("could not allocate an SSH forward port: {error}"))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| format!("could not inspect the SSH forward port: {error}"))
}

fn start_tls_proxies(
    listeners: Vec<TcpListener>,
    config: Arc<ServerConfig>,
    forward_port: u16,
    child_exited: watch::Receiver<bool>,
) -> Vec<JoinHandle<()>> {
    listeners
        .into_iter()
        .map(|listener| {
            tokio::spawn(run_tls_proxy(
                listener,
                config.clone(),
                forward_port,
                child_exited.clone(),
            ))
        })
        .collect()
}

async fn run_tls_proxy(
    listener: TcpListener,
    config: Arc<ServerConfig>,
    forward_port: u16,
    mut child_exited: watch::Receiver<bool>,
) {
    let acceptor = TlsAcceptor::from(config);
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            changed = child_exited.changed() => {
                if changed.is_err() || *child_exited.borrow_and_update() {
                    break;
                }
            }
            accepted = listener.accept() => match accepted {
                Ok((socket, _)) => {
                    let acceptor = acceptor.clone();
                    connections.spawn(async move {
                        let mut downstream = acceptor.accept(socket).await?;
                        let mut upstream = TcpStream::connect((Ipv4Addr::LOCALHOST, forward_port)).await?;
                        copy_bidirectional(&mut downstream, &mut upstream).await?;
                        Ok::<(), io::Error>(())
                    });
                }
                Err(error) => {
                    eprintln!("[rcp] local HTTPS proxy stopped accepting connections: {error}");
                    break;
                }
            },
            completed = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = completed {
                    eprintln!("[rcp] local HTTPS proxy connection task failed: {error}");
                }
            }
        }
    }
    connections.abort_all();
    while connections.join_next().await.is_some() {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_rustls::TlsConnector;

    const CONNECTION_ID: &str = "11111111-1111-4111-8111-111111111111";
    const HOSTNAME: &str = "rcp-11111111111141118111111111111111.localhost";

    fn connection() -> TeamConnectionMetadata {
        TeamConnectionMetadata {
            connection_id: CONNECTION_ID.into(),
            display_name: "Vision lab".into(),
            ssh_target: "rcp@lab-server".into(),
            remote_loopback_port: 8421,
            expected_space_id: "33333333-3333-4333-8333-333333333333".into(),
            local_origin: format!("https://{HOSTNAME}:18421"),
            minimum_shell_version: "0.3.2".into(),
            last_known_cards: Vec::new(),
            operator_route: None,
        }
    }

    #[test]
    fn ssh_argv_uses_one_explicit_owned_forward_and_no_shell() {
        let route = TunnelRoute::from(&connection());
        assert_eq!(
            ssh_arguments(&route, 19421),
            vec![
                "-N",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
                "-o",
                "ControlPersist=no",
                "-o",
                "ForkAfterAuthentication=no",
                "-L",
                "127.0.0.1:19421:127.0.0.1:8421",
                "rcp@lab-server",
            ]
        );
    }

    #[test]
    fn route_identity_changes_with_target_or_remote_port() {
        let original = connection();
        let mut changed_target = original.clone();
        changed_target.ssh_target = "rcp@other-server".into();
        let mut changed_port = original.clone();
        changed_port.remote_loopback_port = 9421;

        assert_ne!(
            TunnelRoute::from(&original),
            TunnelRoute::from(&changed_target)
        );
        assert_ne!(
            TunnelRoute::from(&original),
            TunnelRoute::from(&changed_port)
        );
    }

    #[test]
    fn owned_child_supervisor_forces_and_reaps_a_stubborn_process_group() {
        tauri::async_runtime::block_on(async {
            let mut child = spawn_owned_child(
                Path::new("/bin/sh"),
                &[
                    "-c".into(),
                    "trap '' TERM; /bin/sh -c \"trap '' TERM; sleep 30\" & wait".into(),
                ],
            )
            .unwrap();
            let pid = child.pid;
            time::sleep(Duration::from_millis(100)).await;
            assert_eq!(unsafe { libc::kill(pid as i32, 0) }, 0);
            let stop_started = Instant::now();
            child.stop().await.unwrap();
            assert!(stop_started.elapsed() >= CHILD_STOP_TIMEOUT);
            assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
            assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
            assert_eq!(unsafe { libc::kill(-(pid as i32), 0) }, -1);
            assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
        });
    }

    #[test]
    fn failed_cleanup_retains_the_tunnel_ownership_record() {
        tauri::async_runtime::block_on(async {
            let identity = LocalHttpsIdentity::generated_for_test();
            let tunnels =
                TeamTunnelState::with_ssh_program(&identity, PathBuf::from(SYSTEM_SSH)).unwrap();
            let (_exit_tx, child_exited) = watch::channel(false);
            let route = TunnelRoute::from(&connection());
            let failed_start = failed_start(
                CONNECTION_ID,
                &route,
                OwnedChild {
                    pid: 1,
                    child_exited,
                    stop_child: None,
                    supervisor: Some(tokio::spawn(async {
                        Err("synthetic unconfirmed cleanup".into())
                    })),
                    stop_failure: None,
                },
                "synthetic startup failure".into(),
            )
            .await;
            assert!(failed_start.message.contains("synthetic startup failure"));
            let failed = failed_start.retained.unwrap();
            tunnels
                .runtime
                .lock()
                .await
                .active
                .insert(CONNECTION_ID.into(), failed);

            assert!(tunnels.stop_all_for_test().await.is_err());
            let runtime = tunnels.runtime.lock().await;
            let retained = runtime.active.get(CONNECTION_ID).unwrap();
            assert_eq!(
                retained.child.stop_failure.as_deref(),
                Some("SSH tunnel process group 1: synthetic unconfirmed cleanup")
            );
        });
    }

    #[test]
    fn lifecycle_pause_rejects_new_tunnels_before_starting_ssh() {
        tauri::async_runtime::block_on(async {
            let identity = LocalHttpsIdentity::generated_for_test();
            let tunnels =
                TeamTunnelState::with_ssh_program(&identity, PathBuf::from("/usr/bin/false"))
                    .unwrap();
            let directory = tempfile::tempdir().unwrap();
            let connections = TeamConnectionState::new(directory.path().join("connections.json"));
            connections.save_metadata(connection()).unwrap();
            let lifecycle = BackendState::default();

            tunnels.stop_all_for_lifecycle().await.unwrap();
            let error = tunnels
                .connect_saved(&connections, &lifecycle, CONNECTION_ID)
                .await
                .unwrap_err();
            assert_eq!(
                error,
                "RCP cannot connect a team space while quitting or updating"
            );
        });
    }

    #[test]
    fn local_https_proxy_terminates_tls_and_forwards_plain_bytes() {
        tauri::async_runtime::block_on(async {
            let identity = LocalHttpsIdentity::generated_for_test();
            let server_config = tls_server_config(&identity).unwrap();
            let upstream = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let upstream_port = upstream.local_addr().unwrap().port();
            let upstream_task = tokio::spawn(async move {
                let (mut socket, _) = upstream.accept().await.unwrap();
                let mut request = [0_u8; 4];
                socket.read_exact(&mut request).await.unwrap();
                assert_eq!(&request, b"ping");
                socket.write_all(b"pong").await.unwrap();
            });

            let listeners = bind_local_https(0).unwrap();
            let proxy_port = listeners[0].local_addr().unwrap().port();
            let (child_exit, child_exited) = watch::channel(false);
            let proxies = start_tls_proxies(listeners, server_config, upstream_port, child_exited);

            let mut roots = rustls::RootCertStore::empty();
            roots
                .add(rustls::pki_types::CertificateDer::from(
                    identity.certificate_der().to_vec(),
                ))
                .unwrap();
            let client_config = rustls::ClientConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();
            let connector = TlsConnector::from(Arc::new(client_config));
            let tcp = TcpStream::connect((Ipv4Addr::LOCALHOST, proxy_port))
                .await
                .unwrap();
            // The production WKWebView authenticates the exact pinned leaf for
            // each generated alias. This ordinary rustls client instead uses
            // the certificate's exact localhost SAN to exercise termination
            // and byte forwarding without duplicating that native pin policy.
            let server_name = rustls::pki_types::ServerName::try_from("localhost")
                .unwrap()
                .to_owned();
            let mut tls = connector.connect(server_name, tcp).await.unwrap();
            tls.write_all(b"ping").await.unwrap();
            let mut response = [0_u8; 4];
            tls.read_exact(&mut response).await.unwrap();
            assert_eq!(&response, b"pong");
            drop(tls);
            upstream_task.await.unwrap();
            child_exit.send(true).unwrap();
            for proxy in proxies {
                proxy.await.unwrap();
            }
        });
    }

    #[test]
    #[ignore = "requires RCP_LIVE_SSH_TARGET and an authenticated system SSH configuration"]
    fn live_system_ssh_child_forwards_through_owned_tls_proxy() {
        tauri::async_runtime::block_on(async {
            let ssh_target = std::env::var("RCP_LIVE_SSH_TARGET")
                .expect("RCP_LIVE_SSH_TARGET is required for the ignored live test");
            let identity = LocalHttpsIdentity::generated_for_test();
            let reserved = bind_local_https(0).unwrap();
            let proxy_port = reserved[0].local_addr().unwrap().port();
            drop(reserved);

            let mut connection = connection();
            connection.ssh_target = ssh_target;
            connection.remote_loopback_port = 22;
            connection.local_origin = format!("https://{HOSTNAME}:{proxy_port}");
            let tunnels =
                TeamTunnelState::with_ssh_program(&identity, PathBuf::from(SYSTEM_SSH)).unwrap();
            let directory = tempfile::tempdir().unwrap();
            let connections = TeamConnectionState::new(directory.path().join("connections.json"));
            connections.save_metadata(connection.clone()).unwrap();
            let lifecycle = BackendState::default();
            let ready = tunnels
                .connect_saved(&connections, &lifecycle, CONNECTION_ID)
                .await
                .unwrap();
            assert!(!ready.reused);
            assert!(
                tunnels
                    .connect_saved(&connections, &lifecycle, CONNECTION_ID)
                    .await
                    .unwrap()
                    .reused
            );

            let mut roots = rustls::RootCertStore::empty();
            roots
                .add(rustls::pki_types::CertificateDer::from(
                    identity.certificate_der().to_vec(),
                ))
                .unwrap();
            let client_config = rustls::ClientConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();
            let connector = TlsConnector::from(Arc::new(client_config));
            let tcp = TcpStream::connect((Ipv4Addr::LOCALHOST, proxy_port))
                .await
                .unwrap();
            let server_name = rustls::pki_types::ServerName::try_from("localhost")
                .unwrap()
                .to_owned();
            let mut tls = connector.connect(server_name, tcp).await.unwrap();
            let mut banner = [0_u8; 255];
            let length = time::timeout(Duration::from_secs(5), tls.read(&mut banner))
                .await
                .unwrap()
                .unwrap();
            assert!(banner[..length].starts_with(b"SSH-"));
            drop(tls);
            tunnels.stop_all_for_test().await.unwrap();
        });
    }
}
