use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
};

use semver::Version;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use url::{Host, Url};
use uuid::{Uuid, Version as UuidVersion};
use zeroize::Zeroizing;

const REGISTRY_VERSION: u32 = 2;
const REGISTRY_FILENAME: &str = "team-connections.json";
const MAX_REGISTRY_BYTES: u64 = 1024 * 1024;
const MAX_CONNECTIONS: usize = 64;
const MAX_CACHED_CARDS: usize = 256;
const MAX_DISPLAY_NAME_BYTES: usize = 120;
const MAX_PROJECT_NAME_BYTES: usize = 120;
const MAX_PRIMARY_QUESTION_BYTES: usize = 2_000;
const MAX_SSH_TARGET_BYTES: usize = 255;
const MAX_VERSION_BYTES: usize = 64;
const MEMBER_TOKEN_PREFIX: &[u8] = b"rcp_";
const MEMBER_TOKEN_RANDOM_BYTES: usize = 43;
const MEMBER_TOKEN_BYTES: usize = MEMBER_TOKEN_PREFIX.len() + MEMBER_TOKEN_RANDOM_BYTES;
const SESSION_TOKEN_PREFIX: &[u8] = b"rcp_session_";
const ENROLLMENT_CODE_ID_BYTES: usize = 16;
const BOOTSTRAP_CODE_PREFIX: &[u8] = b"rcp_bootstrap_";
const INVITATION_CODE_PREFIX: &[u8] = b"rcp_invite_";
const KEYCHAIN_SERVICE: &str = "app.researchcontrolpanel.rcp.team-member-token.source-v1";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CachedTeamProjectCard {
    pub id: String,
    pub name: String,
    pub primary_question: Option<String>,
    pub attention_count: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TeamConnectionMetadata {
    pub connection_id: String,
    pub display_name: String,
    pub ssh_target: String,
    pub remote_loopback_port: u16,
    pub expected_space_id: String,
    pub local_origin: String,
    pub minimum_shell_version: String,
    pub last_known_cards: Vec<CachedTeamProjectCard>,
    #[serde(default)]
    pub operator_route: Option<ServerOperatorRoute>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ServerOperatorMode {
    DirectRcp,
    SudoRcp,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ServerOperatorRoute {
    pub ssh_target: String,
    pub mode: ServerOperatorMode,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct RemovalResult {
    pub removed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CredentialReference {
    pub service: &'static str,
    pub account: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TeamConnectionRegistry {
    version: u32,
    connections: Vec<TeamConnectionMetadata>,
}

impl Default for TeamConnectionRegistry {
    fn default() -> Self {
        Self {
            version: REGISTRY_VERSION,
            connections: Vec::new(),
        }
    }
}

pub struct TeamConnectionState {
    registry_path: PathBuf,
    lock: Mutex<()>,
}

impl TeamConnectionState {
    pub fn for_app(app: &AppHandle) -> Result<Self, String> {
        let config_dir = app
            .path()
            .app_config_dir()
            .map_err(|error| format!("cannot locate RCP desktop configuration: {error}"))?;
        Ok(Self::new(config_dir.join(REGISTRY_FILENAME)))
    }

    pub(crate) fn new(registry_path: PathBuf) -> Self {
        Self {
            registry_path,
            lock: Mutex::new(()),
        }
    }

    pub fn list(&self) -> Result<Vec<TeamConnectionMetadata>, String> {
        let _guard = self.acquire()?;
        Ok(self.read_registry()?.connections)
    }

    /// Only verified native connection/session flows may write routing identity.
    /// Web commands deliberately do not expose this method.
    #[allow(dead_code)]
    pub(crate) fn save_metadata(
        &self,
        connection: TeamConnectionMetadata,
    ) -> Result<TeamConnectionMetadata, String> {
        connection.validate()?;
        let _guard = self.acquire()?;
        let mut registry = self.read_registry()?;

        if let Some(existing) = registry
            .connections
            .iter_mut()
            .find(|saved| saved.connection_id == connection.connection_id)
        {
            if existing.expected_space_id != connection.expected_space_id {
                return Err(
                    "a saved team connection cannot change its expected space identity".into(),
                );
            }
            if existing.local_origin != connection.local_origin {
                return Err("a saved team connection cannot change its local origin".into());
            }
            *existing = connection.clone();
        } else {
            if registry.connections.len() >= MAX_CONNECTIONS {
                return Err(format!(
                    "RCP desktop supports at most {MAX_CONNECTIONS} saved team connections"
                ));
            }
            registry.connections.push(connection.clone());
        }

        registry.validate()?;
        self.write_registry(&registry)?;
        Ok(connection)
    }

    pub fn remove_metadata(&self, connection_id: &str) -> Result<RemovalResult, String> {
        validate_uuid4(connection_id, "team connection identity")?;
        let _guard = self.acquire()?;
        let mut registry = self.read_registry()?;
        let before = registry.connections.len();
        registry
            .connections
            .retain(|connection| connection.connection_id != connection_id);
        let removed = registry.connections.len() != before;
        if removed {
            self.write_registry(&registry)?;
        }
        Ok(RemovalResult { removed })
    }

    pub fn set_operator_route(
        &self,
        connection_id: &str,
        route: Option<ServerOperatorRoute>,
    ) -> Result<TeamConnectionMetadata, String> {
        validate_uuid4(connection_id, "team connection identity")?;
        if let Some(route) = &route {
            route.validate()?;
        }
        let _guard = self.acquire()?;
        let mut registry = self.read_registry()?;
        let connection = registry
            .connections
            .iter_mut()
            .find(|connection| connection.connection_id == connection_id)
            .ok_or_else(|| "the team connection is not saved on this desktop".to_string())?;
        connection.operator_route = route;
        let updated = connection.clone();
        registry.validate()?;
        self.write_registry(&registry)?;
        Ok(updated)
    }

    pub fn store_member_token(&self, connection_id: &str, token: String) -> Result<(), String> {
        let token = Zeroizing::new(token);
        validate_member_token(&token)?;
        let reference = credential_reference(connection_id)?;
        let _guard = self.acquire()?;
        if !self
            .read_registry()?
            .connections
            .iter()
            .any(|connection| connection.connection_id == connection_id)
        {
            return Err("team connection metadata must be saved before its credential".into());
        }
        store_keychain_password(&reference, token.as_bytes())
    }

    pub(crate) fn load_member_token(
        &self,
        connection_id: &str,
    ) -> Result<Zeroizing<String>, String> {
        let reference = credential_reference(connection_id)?;
        let _guard = self.acquire()?;
        if !self
            .read_registry()?
            .connections
            .iter()
            .any(|connection| connection.connection_id == connection_id)
        {
            return Err("team connection metadata must exist before reading its credential".into());
        }
        let bytes = load_keychain_password(&reference)?
            .ok_or_else(|| "this team connection has no saved member credential".to_string())?;
        let token = Zeroizing::new(
            String::from_utf8(bytes.to_vec())
                .map_err(|_| "the saved team member credential is invalid".to_string())?,
        );
        validate_member_token(&token)?;
        Ok(token)
    }

    pub fn remove_member_token(&self, connection_id: &str) -> Result<RemovalResult, String> {
        let reference = credential_reference(connection_id)?;
        let _guard = self.acquire()?;
        remove_keychain_password(&reference).map(|removed| RemovalResult { removed })
    }

    fn acquire(&self) -> Result<MutexGuard<'_, ()>, String> {
        self.lock
            .lock()
            .map_err(|_| "the team connection registry lock is unavailable".to_string())
    }

    fn read_registry(&self) -> Result<TeamConnectionRegistry, String> {
        let file = match open_registry_file(&self.registry_path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(TeamConnectionRegistry::default());
            }
            Err(error) => {
                return Err(format!("cannot open saved team connections: {error}"));
            }
        };
        let metadata = file
            .metadata()
            .map_err(|error| format!("cannot inspect saved team connections: {error}"))?;
        if !metadata.is_file() {
            return Err("the saved team connection registry is not a regular file".into());
        }
        if metadata.len() > MAX_REGISTRY_BYTES {
            return Err("the saved team connection registry is too large".into());
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_REGISTRY_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("cannot read saved team connections: {error}"))?;
        if bytes.len() as u64 > MAX_REGISTRY_BYTES {
            return Err("the saved team connection registry is too large".into());
        }
        let registry: TeamConnectionRegistry = serde_json::from_slice(&bytes)
            .map_err(|error| format!("saved team connections are invalid: {error}"))?;
        registry.validate()?;
        Ok(registry)
    }

    fn write_registry(&self, registry: &TeamConnectionRegistry) -> Result<(), String> {
        registry.validate()?;
        let mut bytes = serde_json::to_vec_pretty(registry)
            .map_err(|error| format!("cannot serialize saved team connections: {error}"))?;
        bytes.push(b'\n');
        if bytes.len() as u64 > MAX_REGISTRY_BYTES {
            return Err("the saved team connection registry is too large".into());
        }
        if contains_rcp_credential(&bytes) {
            return Err("saved team connection metadata cannot contain an RCP credential".into());
        }

        let parent = self
            .registry_path
            .parent()
            .ok_or_else(|| "the team connection registry has no parent directory".to_string())?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create RCP desktop configuration: {error}"))?;
        reject_non_regular_destination(&self.registry_path)?;

        let mut temporary = tempfile::Builder::new()
            .prefix(".team-connections-")
            .tempfile_in(parent)
            .map_err(|error| format!("cannot create a team connection registry: {error}"))?;
        secure_file_permissions(temporary.as_file())?;
        temporary
            .write_all(&bytes)
            .and_then(|()| temporary.as_file().sync_all())
            .map_err(|error| format!("cannot save team connections: {error}"))?;
        temporary
            .persist(&self.registry_path)
            .map_err(|error| format!("cannot publish saved team connections: {}", error.error))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("cannot finish saving team connections: {error}"))?;
        Ok(())
    }
}

impl TeamConnectionRegistry {
    fn validate(&self) -> Result<(), String> {
        if self.version != REGISTRY_VERSION {
            return Err(format!(
                "saved team connection registry version {} is unsupported",
                self.version
            ));
        }
        if self.connections.len() > MAX_CONNECTIONS {
            return Err(format!(
                "RCP desktop supports at most {MAX_CONNECTIONS} saved team connections"
            ));
        }

        let mut connection_ids = HashSet::new();
        let mut space_ids = HashSet::new();
        let mut origins = HashSet::new();
        for connection in &self.connections {
            connection.validate()?;
            if !connection_ids.insert(&connection.connection_id) {
                return Err("saved team connection identities must be unique".into());
            }
            if !space_ids.insert(&connection.expected_space_id) {
                return Err("a team space may be saved only once".into());
            }
            if !origins.insert(&connection.local_origin) {
                return Err("saved team connections must use distinct local origins".into());
            }
        }
        Ok(())
    }
}

impl TeamConnectionMetadata {
    fn validate(&self) -> Result<(), String> {
        validate_uuid4(&self.connection_id, "team connection identity")?;
        validate_text(
            &self.display_name,
            "team connection display name",
            MAX_DISPLAY_NAME_BYTES,
            false,
        )?;
        validate_ssh_target(&self.ssh_target)?;
        if self.remote_loopback_port == 0 {
            return Err("remote team server port must be a positive integer".into());
        }
        validate_uuid4(&self.expected_space_id, "expected team space identity")?;
        validate_local_origin(&self.local_origin, &self.connection_id)?;
        validate_shell_version(&self.minimum_shell_version)?;
        if self.last_known_cards.len() > MAX_CACHED_CARDS {
            return Err(format!(
                "a team connection may cache at most {MAX_CACHED_CARDS} project cards"
            ));
        }
        let mut card_ids = HashSet::new();
        for card in &self.last_known_cards {
            card.validate()?;
            if !card_ids.insert(&card.id) {
                return Err("cached team project identities must be unique".into());
            }
        }
        if let Some(route) = &self.operator_route {
            route.validate()?;
        }
        Ok(())
    }
}

impl ServerOperatorRoute {
    fn validate(&self) -> Result<(), String> {
        validate_ssh_target(&self.ssh_target)?;
        if self.mode == ServerOperatorMode::DirectRcp && !self.ssh_target.starts_with("rcp@") {
            return Err("a direct RCP operator route must explicitly use rcp@host".into());
        }
        Ok(())
    }
}

impl CachedTeamProjectCard {
    fn validate(&self) -> Result<(), String> {
        validate_uuid4(&self.id, "cached team project identity")?;
        validate_text(
            &self.name,
            "cached team project name",
            MAX_PROJECT_NAME_BYTES,
            false,
        )?;
        if let Some(question) = &self.primary_question {
            validate_text(
                question,
                "cached primary question",
                MAX_PRIMARY_QUESTION_BYTES,
                false,
            )?;
        }
        Ok(())
    }
}

fn validate_uuid4(value: &str, label: &str) -> Result<(), String> {
    let parsed =
        Uuid::parse_str(value).map_err(|_| format!("{label} must be a canonical UUID4"))?;
    if parsed.get_version() != Some(UuidVersion::Random) || parsed.to_string() != value {
        return Err(format!(
            "{label} must be a lowercase, hyphenated canonical UUID4"
        ));
    }
    Ok(())
}

fn validate_text(
    value: &str,
    label: &str,
    max_bytes: usize,
    allow_empty: bool,
) -> Result<(), String> {
    if (!allow_empty && value.is_empty()) || value.len() > max_bytes {
        return Err(format!("{label} must contain 1 to {max_bytes} bytes"));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(format!("{label} must be one trimmed line"));
    }
    if contains_rcp_credential(value.as_bytes()) {
        return Err(format!("{label} cannot contain an RCP credential"));
    }
    Ok(())
}

pub(crate) fn validate_ssh_target(value: &str) -> Result<(), String> {
    validate_text(value, "SSH target", MAX_SSH_TARGET_BYTES, false)?;
    if !value.is_ascii()
        || value.starts_with('-')
        || value.contains('/')
        || value.contains('\\')
        || value.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        return Err("SSH target must be one host alias or user@host argument".into());
    }
    Ok(())
}

pub(crate) fn allocate_local_origin(connection_id: &str, port: u16) -> Result<String, String> {
    validate_uuid4(connection_id, "team connection identity")?;
    if port == 0 || port == 443 {
        return Err("local team HTTPS port must be positive and non-default".into());
    }
    Ok(format!(
        "https://rcp-{}.localhost:{port}",
        connection_id.replace('-', "")
    ))
}

fn validate_local_origin(value: &str, connection_id: &str) -> Result<(), String> {
    validate_text(value, "local team origin", 255, false)?;
    let url = Url::parse(value).map_err(|_| "local team origin must be a URL".to_string())?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_none()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(
            "local team origin must be the connection's canonical HTTPS localhost origin".into(),
        );
    }
    let expected = allocate_local_origin(connection_id, url.port().unwrap())?;
    if !matches!(url.host(), Some(Host::Domain(_)))
        || url.origin().ascii_serialization() != value
        || value != expected
    {
        return Err(
            "local team origin must be the connection's canonical HTTPS localhost origin".into(),
        );
    }
    Ok(())
}

fn validate_shell_version(value: &str) -> Result<(), String> {
    validate_text(
        value,
        "minimum desktop shell version",
        MAX_VERSION_BYTES,
        false,
    )?;
    let parsed = Version::parse(value)
        .map_err(|_| "minimum desktop shell version must be semantic versioning".to_string())?;
    if parsed.to_string() != value {
        return Err("minimum desktop shell version must be canonical semantic versioning".into());
    }
    Ok(())
}

fn validate_member_token(token: &str) -> Result<(), String> {
    if token.len() != MEMBER_TOKEN_BYTES || !is_member_token(token.as_bytes()) {
        return Err("team member credential is not a permanent RCP member token".into());
    }
    Ok(())
}

fn contains_rcp_credential(value: &[u8]) -> bool {
    contains_fixed_credential(value, MEMBER_TOKEN_PREFIX, MEMBER_TOKEN_RANDOM_BYTES)
        || contains_fixed_credential(value, SESSION_TOKEN_PREFIX, MEMBER_TOKEN_RANDOM_BYTES)
        || contains_enrollment_code(value, BOOTSTRAP_CODE_PREFIX)
        || contains_enrollment_code(value, INVITATION_CODE_PREFIX)
}

fn is_member_token(value: &[u8]) -> bool {
    value.len() == MEMBER_TOKEN_BYTES
        && value.starts_with(MEMBER_TOKEN_PREFIX)
        && value[MEMBER_TOKEN_PREFIX.len()..]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn contains_fixed_credential(value: &[u8], prefix: &[u8], random_bytes: usize) -> bool {
    value.windows(prefix.len() + random_bytes).any(|candidate| {
        candidate.starts_with(prefix) && candidate[prefix.len()..].iter().all(is_urlsafe_byte)
    })
}

fn contains_enrollment_code(value: &[u8], prefix: &[u8]) -> bool {
    let public_bytes = prefix.len() + ENROLLMENT_CODE_ID_BYTES;
    let credential_bytes = public_bytes + 1 + MEMBER_TOKEN_RANDOM_BYTES;
    value.windows(credential_bytes).any(|candidate| {
        candidate.starts_with(prefix)
            && candidate[prefix.len()..public_bytes]
                .iter()
                .all(is_urlsafe_byte)
            && candidate[public_bytes] == b'.'
            && candidate[public_bytes + 1..].iter().all(is_urlsafe_byte)
    })
}

fn is_urlsafe_byte(byte: &u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
}

fn credential_reference(connection_id: &str) -> Result<CredentialReference, String> {
    validate_uuid4(connection_id, "team connection identity")?;
    Ok(CredentialReference {
        service: KEYCHAIN_SERVICE,
        account: format!("team-connection/{connection_id}"),
    })
}

fn reject_non_regular_destination(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err("the saved team connection registry is not a regular file".into())
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot inspect saved team connections: {error}")),
    }
}

#[cfg(unix)]
fn open_registry_file(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
}

#[cfg(not(unix))]
fn open_registry_file(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

#[cfg(unix)]
fn secure_file_permissions(file: &File) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|error| format!("cannot secure saved team connections: {error}"))
}

#[cfg(not(unix))]
fn secure_file_permissions(_file: &File) -> Result<(), String> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn store_keychain_password(reference: &CredentialReference, token: &[u8]) -> Result<(), String> {
    crate::keychain::set(reference.service, &reference.account, token)
        .map_err(|error| format!("could not store the team member credential in Keychain: {error}"))
}

#[cfg(target_os = "macos")]
fn load_keychain_password(
    reference: &CredentialReference,
) -> Result<Option<Zeroizing<Vec<u8>>>, String> {
    crate::keychain::get(reference.service, &reference.account).map_err(|error| {
        format!("could not read the team member credential from Keychain: {error}")
    })
}

#[cfg(not(target_os = "macos"))]
fn store_keychain_password(_reference: &CredentialReference, _token: &[u8]) -> Result<(), String> {
    Err("team member credential storage is supported only by the macOS desktop app".into())
}

#[cfg(not(target_os = "macos"))]
fn load_keychain_password(
    _reference: &CredentialReference,
) -> Result<Option<Zeroizing<Vec<u8>>>, String> {
    Err("team member credential storage is supported only by the macOS desktop app".into())
}

#[cfg(target_os = "macos")]
fn remove_keychain_password(reference: &CredentialReference) -> Result<bool, String> {
    crate::keychain::remove(reference.service, &reference.account).map_err(|error| {
        format!("could not remove the team member credential from Keychain: {error}")
    })
}

#[cfg(not(target_os = "macos"))]
fn remove_keychain_password(_reference: &CredentialReference) -> Result<bool, String> {
    Err("team member credential storage is supported only by the macOS desktop app".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONNECTION_ID: &str = "11111111-1111-4111-8111-111111111111";
    const OTHER_CONNECTION_ID: &str = "22222222-2222-4222-8222-222222222222";
    const SPACE_ID: &str = "33333333-3333-4333-8333-333333333333";
    const OTHER_SPACE_ID: &str = "44444444-4444-4444-8444-444444444444";
    const PROJECT_ID: &str = "55555555-5555-4555-8555-555555555555";

    fn sample_connection() -> TeamConnectionMetadata {
        TeamConnectionMetadata {
            connection_id: CONNECTION_ID.into(),
            display_name: "Vision lab".into(),
            ssh_target: "rcp@lab-server".into(),
            remote_loopback_port: 8421,
            expected_space_id: SPACE_ID.into(),
            local_origin: allocate_local_origin(CONNECTION_ID, 18421).unwrap(),
            minimum_shell_version: "0.3.2".into(),
            last_known_cards: vec![CachedTeamProjectCard {
                id: PROJECT_ID.into(),
                name: "Abu Dhabi".into(),
                primary_question: Some("Which intervention works?".into()),
                attention_count: 2,
            }],
            operator_route: None,
        }
    }

    fn state(directory: &Path) -> TeamConnectionState {
        TeamConnectionState::new(directory.join(REGISTRY_FILENAME))
    }

    #[test]
    fn registry_serialization_contains_only_bounded_nonsecret_metadata() {
        let registry = TeamConnectionRegistry {
            version: REGISTRY_VERSION,
            connections: vec![sample_connection()],
        };
        registry.validate().unwrap();
        let value = serde_json::to_value(&registry).unwrap();
        assert_eq!(value["version"], REGISTRY_VERSION);
        assert_eq!(value["connections"][0]["connection_id"], CONNECTION_ID);
        assert_eq!(value["connections"][0]["expected_space_id"], SPACE_ID);
        assert_eq!(
            value["connections"][0]["last_known_cards"][0]["id"],
            PROJECT_ID
        );
        let serialized = serde_json::to_string(&registry).unwrap();
        assert!(!serialized.contains("credential"));
        assert!(!serialized.contains("token"));
        assert!(!serialized.contains(KEYCHAIN_SERVICE));
    }

    #[test]
    fn save_is_atomic_and_preserves_immutable_connection_identity() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        let original = sample_connection();
        assert_eq!(state.save_metadata(original.clone()).unwrap(), original);
        assert_eq!(state.list().unwrap(), vec![original.clone()]);

        let mut updated = original.clone();
        updated.display_name = "Vision lab shared".into();
        updated.ssh_target = "operator@lab-server".into();
        updated.last_known_cards[0].attention_count = 3;
        state.save_metadata(updated.clone()).unwrap();
        assert_eq!(state.list().unwrap(), vec![updated]);

        let mut changed_space = original.clone();
        changed_space.expected_space_id = OTHER_SPACE_ID.into();
        assert!(state.save_metadata(changed_space).is_err());
        let mut changed_origin = original;
        changed_origin.local_origin = allocate_local_origin(CONNECTION_ID, 19421).unwrap();
        assert!(state.save_metadata(changed_origin).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&state.registry_path)
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn registry_rejects_duplicate_spaces_origins_and_cards() {
        let first = sample_connection();
        let mut duplicate_space = first.clone();
        duplicate_space.connection_id = OTHER_CONNECTION_ID.into();
        duplicate_space.local_origin = allocate_local_origin(OTHER_CONNECTION_ID, 19421).unwrap();
        let registry = TeamConnectionRegistry {
            version: REGISTRY_VERSION,
            connections: vec![first.clone(), duplicate_space],
        };
        assert!(registry.validate().is_err());

        let mut duplicate_origin = first.clone();
        duplicate_origin.connection_id = OTHER_CONNECTION_ID.into();
        duplicate_origin.expected_space_id = OTHER_SPACE_ID.into();
        let registry = TeamConnectionRegistry {
            version: REGISTRY_VERSION,
            connections: vec![first.clone(), duplicate_origin],
        };
        assert!(registry.validate().is_err());

        let mut duplicate_card = first;
        duplicate_card
            .last_known_cards
            .push(duplicate_card.last_known_cards[0].clone());
        assert!(duplicate_card.validate().is_err());
    }

    #[test]
    fn metadata_and_credential_references_have_independent_lifecycles() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        state.save_metadata(sample_connection()).unwrap();

        assert_eq!(
            credential_reference(CONNECTION_ID).unwrap(),
            CredentialReference {
                service: KEYCHAIN_SERVICE,
                account: format!("team-connection/{CONNECTION_ID}"),
            }
        );
        assert!(state.remove_metadata(CONNECTION_ID).unwrap().removed);
        assert!(!state.remove_metadata(CONNECTION_ID).unwrap().removed);
        assert!(state.list().unwrap().is_empty());
        assert_eq!(
            credential_reference(CONNECTION_ID).unwrap().account,
            format!("team-connection/{CONNECTION_ID}")
        );
    }

    #[test]
    fn operator_route_is_explicit_nonsecret_metadata_and_can_be_cleared() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        state.save_metadata(sample_connection()).unwrap();
        let route = ServerOperatorRoute {
            ssh_target: "alice@lab-server".into(),
            mode: ServerOperatorMode::SudoRcp,
        };

        let configured = state
            .set_operator_route(CONNECTION_ID, Some(route.clone()))
            .unwrap();
        assert_eq!(configured.operator_route, Some(route));
        let serialized = fs::read_to_string(&state.registry_path).unwrap();
        assert!(serialized.contains("sudo_rcp"));
        assert!(!serialized.contains("password"));
        assert!(!serialized.contains("private_key"));

        let cleared = state.set_operator_route(CONNECTION_ID, None).unwrap();
        assert_eq!(cleared.operator_route, None);

        assert!(state
            .set_operator_route(
                CONNECTION_ID,
                Some(ServerOperatorRoute {
                    ssh_target: "alice@lab-server".into(),
                    mode: ServerOperatorMode::DirectRcp,
                }),
            )
            .is_err());
    }

    #[test]
    fn permanent_token_shape_is_narrow_and_metadata_rejects_every_rcp_credential() {
        let token = format!("rcp_{}", "A".repeat(MEMBER_TOKEN_RANDOM_BYTES));
        validate_member_token(&token).unwrap();
        for rejected in [
            "rcp_too-short".to_string(),
            format!("rcp_session_{}", "A".repeat(MEMBER_TOKEN_RANDOM_BYTES)),
            format!("rcp_{}!", "A".repeat(MEMBER_TOKEN_RANDOM_BYTES - 1)),
        ] {
            assert!(validate_member_token(&rejected).is_err());
        }

        let credentials = [
            token,
            format!("rcp_session_{}", "B".repeat(MEMBER_TOKEN_RANDOM_BYTES)),
            format!(
                "rcp_bootstrap_{}.{}",
                "C".repeat(ENROLLMENT_CODE_ID_BYTES),
                "D".repeat(MEMBER_TOKEN_RANDOM_BYTES)
            ),
            format!(
                "rcp_invite_{}.{}",
                "E".repeat(ENROLLMENT_CODE_ID_BYTES),
                "F".repeat(MEMBER_TOKEN_RANDOM_BYTES)
            ),
        ];
        for credential in credentials {
            let mut connection = sample_connection();
            connection.display_name = format!("Lab {credential}");
            assert!(
                connection.validate().is_err(),
                "saved credential-shaped metadata: {credential}"
            );
        }
    }

    #[test]
    fn local_origin_is_connection_bound_https_and_uses_an_explicit_port() {
        let expected = allocate_local_origin(CONNECTION_ID, 18421).unwrap();
        validate_local_origin(&expected, CONNECTION_ID).unwrap();
        assert!(allocate_local_origin(CONNECTION_ID, 0).is_err());
        assert!(allocate_local_origin(CONNECTION_ID, 443).is_err());
        for rejected in [
            "http://rcp-11111111111141118111111111111111.localhost:18421",
            "https://localhost:18421",
            "https://rcp-11111111111141118111111111111111.localhost",
            "https://rcp-22222222222242228222222222222222.localhost:18421",
            "https://rcp-11111111111141118111111111111111.localhost:18421/path",
        ] {
            assert!(
                validate_local_origin(rejected, CONNECTION_ID).is_err(),
                "accepted {rejected}"
            );
        }
    }

    #[test]
    fn malformed_unknown_or_unbounded_registry_data_fails_closed() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        fs::write(
            &state.registry_path,
            r#"{"version":1,"connections":[],"unexpected":true}"#,
        )
        .unwrap();
        assert!(state.list().is_err());

        let oversized = File::create(&state.registry_path).unwrap();
        oversized.set_len(MAX_REGISTRY_BYTES + 1).unwrap();
        assert!(state.list().is_err());

        let mut connection = sample_connection();
        connection.last_known_cards = (0..=MAX_CACHED_CARDS)
            .map(|index| CachedTeamProjectCard {
                id: format!("55555555-5555-4555-8555-{index:012}"),
                name: "project".into(),
                primary_question: None,
                attention_count: 0,
            })
            .collect();
        assert!(connection.validate().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn registry_refuses_a_symlink_destination() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("elsewhere.json");
        fs::write(&target, "untouched").unwrap();
        let state = state(directory.path());
        symlink(&target, &state.registry_path).unwrap();
        assert!(state.save_metadata(sample_connection()).is_err());
        assert_eq!(fs::read_to_string(target).unwrap(), "untouched");
    }
}
