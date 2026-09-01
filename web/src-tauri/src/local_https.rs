use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

#[cfg(target_os = "macos")]
use ring::{
    aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM, NONCE_LEN},
    rand::{SecureRandom, SystemRandom},
};
#[cfg(target_os = "macos")]
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};
#[cfg(target_os = "macos")]
use tauri::Manager;

#[cfg(target_os = "macos")]
const COOKIE_INSTALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

const IDENTITY_VERSION: u8 = 1;
const IDENTITY_HEADER: &[u8] = b"RCP-LOCAL-HTTPS\0";
const MAX_CERTIFICATE_BYTES: usize = 32 * 1024;
const MAX_PRIVATE_KEY_BYTES: usize = 16 * 1024;
const KEYCHAIN_SERVICE: &str = "app.researchcontrolpanel.rcp.local-https";
const KEYCHAIN_ACCOUNT: &str = "desktop-identity-sealing-key/source-v1";
#[cfg(target_os = "macos")]
const SEALED_IDENTITY_FILENAME: &str = "local-https-identity-v1.sealed";
#[cfg(target_os = "macos")]
const SEALED_IDENTITY_PREFIX: &[u8] = b"RCP-LOCAL-HTTPS-SEALED\0\x01";
#[cfg(target_os = "macos")]
const SEALING_KEY_BYTES: usize = 32;
#[cfg(target_os = "macos")]
const AUTH_TAG_BYTES: usize = 16;
#[cfg(target_os = "macos")]
const MAX_SEALED_IDENTITY_BYTES: u64 = (IDENTITY_HEADER.len()
    + 1
    + 8
    + MAX_CERTIFICATE_BYTES
    + MAX_PRIVATE_KEY_BYTES
    + SEALED_IDENTITY_PREFIX.len()
    + NONCE_LEN
    + AUTH_TAG_BYTES) as u64;

pub struct LocalHttpsIdentity {
    certificate_der: Vec<u8>,
    private_key_der: Zeroizing<Vec<u8>>,
    fingerprint_sha256: String,
}

impl LocalHttpsIdentity {
    #[cfg(target_os = "macos")]
    pub fn load_or_create(app: &tauri::AppHandle) -> Result<Self, String> {
        let path = sealed_identity_path(app)?;
        let sealed = read_sealed_identity(&path)?;
        let key = crate::keychain::get(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
            .map_err(|error| format!("could not read the local HTTPS sealing key: {error}"))?;
        match (sealed, key) {
            (Some(sealed), Some(key)) => {
                let encoded = open_identity(&sealed, &key)?;
                decode_identity(&encoded)
            }
            (None, None) => create_and_store_identity(&path),
            (Some(_), None) => Err(
                "the encrypted local HTTPS identity exists but its Keychain sealing key is missing"
                    .into(),
            ),
            (None, Some(_)) => Err(
                "the local HTTPS Keychain sealing key exists but its encrypted identity is missing"
                    .into(),
            ),
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn load_or_create(_app: &tauri::AppHandle) -> Result<Self, String> {
        Err("local HTTPS identity storage is supported only by the macOS desktop app".into())
    }

    pub fn fingerprint_sha256(&self) -> &str {
        &self.fingerprint_sha256
    }

    pub(crate) fn certificate_der(&self) -> &[u8] {
        &self.certificate_der
    }

    pub(crate) fn private_key_der(&self) -> &[u8] {
        &self.private_key_der
    }

    #[cfg(test)]
    pub(crate) fn generated_for_test() -> Self {
        generate_identity().expect("test local HTTPS identity")
    }
}

fn generate_identity() -> Result<LocalHttpsIdentity, String> {
    let rcgen::CertifiedKey { cert, signing_key } = rcgen::generate_simple_self_signed(vec![
        "localhost".to_string(),
        "*.localhost".to_string(),
    ])
    .map_err(|error| format!("could not generate the local HTTPS identity: {error}"))?;
    identity_from_parts(cert.der().to_vec(), signing_key.serialize_der())
}

fn identity_from_parts(
    certificate_der: Vec<u8>,
    private_key_der: Vec<u8>,
) -> Result<LocalHttpsIdentity, String> {
    let private_key_der = Zeroizing::new(private_key_der);
    if certificate_der.is_empty() || certificate_der.len() > MAX_CERTIFICATE_BYTES {
        return Err("the local HTTPS certificate has an invalid size".into());
    }
    if private_key_der.is_empty() || private_key_der.len() > MAX_PRIVATE_KEY_BYTES {
        return Err("the local HTTPS private key has an invalid size".into());
    }
    let private_key = rustls::pki_types::PrivateKeyDer::try_from(private_key_der.as_slice())
        .map_err(|_| "the local HTTPS private key is invalid".to_string())?;
    let signing_key = rustls::crypto::ring::sign::any_supported_type(&private_key)
        .map_err(|_| "the local HTTPS private key is invalid".to_string())?;
    rustls::sign::CertifiedKey::new(
        vec![rustls::pki_types::CertificateDer::from(
            certificate_der.clone(),
        )],
        signing_key,
    )
    .keys_match()
    .map_err(|_| "the local HTTPS certificate and private key do not match".to_string())?;
    let fingerprint_sha256 = lowercase_sha256(&certificate_der);
    Ok(LocalHttpsIdentity {
        certificate_der,
        private_key_der,
        fingerprint_sha256,
    })
}

fn encode_identity(identity: &LocalHttpsIdentity) -> Result<Zeroizing<Vec<u8>>, String> {
    let certificate_length = u32::try_from(identity.certificate_der.len())
        .map_err(|_| "the local HTTPS certificate is too large".to_string())?;
    let key_length = u32::try_from(identity.private_key_der.len())
        .map_err(|_| "the local HTTPS private key is too large".to_string())?;
    let mut encoded = Zeroizing::new(Vec::with_capacity(
        IDENTITY_HEADER.len()
            + 1
            + 8
            + identity.certificate_der.len()
            + identity.private_key_der.len(),
    ));
    encoded.extend_from_slice(IDENTITY_HEADER);
    encoded.push(IDENTITY_VERSION);
    encoded.extend_from_slice(&certificate_length.to_be_bytes());
    encoded.extend_from_slice(&key_length.to_be_bytes());
    encoded.extend_from_slice(&identity.certificate_der);
    encoded.extend_from_slice(&identity.private_key_der);
    Ok(encoded)
}

fn decode_identity(encoded: &[u8]) -> Result<LocalHttpsIdentity, String> {
    let prefix = IDENTITY_HEADER.len() + 1 + 8;
    if encoded.len() < prefix || !encoded.starts_with(IDENTITY_HEADER) {
        return Err("the local HTTPS identity record has an unsupported shape".into());
    }
    if encoded[IDENTITY_HEADER.len()] != IDENTITY_VERSION {
        return Err("the local HTTPS identity record has an unsupported version".into());
    }
    let lengths = &encoded[IDENTITY_HEADER.len() + 1..prefix];
    let certificate_length = u32::from_be_bytes(lengths[..4].try_into().unwrap()) as usize;
    let key_length = u32::from_be_bytes(lengths[4..].try_into().unwrap()) as usize;
    if certificate_length > MAX_CERTIFICATE_BYTES
        || key_length > MAX_PRIVATE_KEY_BYTES
        || prefix
            .checked_add(certificate_length)
            .and_then(|value| value.checked_add(key_length))
            != Some(encoded.len())
    {
        return Err("the local HTTPS identity record has invalid lengths".into());
    }
    let certificate_end = prefix + certificate_length;
    identity_from_parts(
        encoded[prefix..certificate_end].to_vec(),
        encoded[certificate_end..].to_vec(),
    )
}

fn lowercase_sha256(value: &[u8]) -> String {
    Sha256::digest(value)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(target_os = "macos")]
fn sealed_identity_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map(|directory| directory.join(SEALED_IDENTITY_FILENAME))
        .map_err(|error| format!("cannot locate RCP desktop configuration: {error}"))
}

#[cfg(target_os = "macos")]
fn read_sealed_identity(path: &Path) -> Result<Option<Vec<u8>>, String> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "could not open the encrypted local HTTPS identity: {error}"
            ));
        }
    };
    let metadata = file.metadata().map_err(|error| {
        format!("could not inspect the encrypted local HTTPS identity: {error}")
    })?;
    if !metadata.is_file()
        || metadata.len() > MAX_SEALED_IDENTITY_BYTES
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(
            "the encrypted local HTTPS identity has an invalid type, size, or permissions".into(),
        );
    }
    let mut sealed = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_SEALED_IDENTITY_BYTES + 1)
        .read_to_end(&mut sealed)
        .map_err(|error| format!("could not read the encrypted local HTTPS identity: {error}"))?;
    if sealed.len() as u64 > MAX_SEALED_IDENTITY_BYTES {
        return Err("the encrypted local HTTPS identity exceeds its size limit".into());
    }
    Ok(Some(sealed))
}

#[cfg(target_os = "macos")]
fn create_and_store_identity(path: &Path) -> Result<LocalHttpsIdentity, String> {
    let identity = generate_identity()?;
    let encoded = encode_identity(&identity)?;
    let key = generate_sealing_key()?;
    let sealed = seal_identity(&encoded, &key[..])?;
    crate::keychain::set(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, &key[..])
        .map_err(|error| format!("could not store the local HTTPS sealing key: {error}"))?;
    if let Err(write_error) = write_sealed_identity(path, &sealed) {
        let cleanup = crate::keychain::remove(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT);
        return match cleanup {
            Ok(_) => Err(write_error),
            Err(cleanup_error) => Err(format!(
                "{write_error}; could not remove the incomplete Keychain sealing key: {cleanup_error}"
            )),
        };
    }
    Ok(identity)
}

#[cfg(target_os = "macos")]
fn generate_sealing_key() -> Result<Zeroizing<[u8; SEALING_KEY_BYTES]>, String> {
    let mut key = Zeroizing::new([0_u8; SEALING_KEY_BYTES]);
    SystemRandom::new()
        .fill(key.as_mut())
        .map_err(|_| "could not generate the local HTTPS sealing key".to_string())?;
    Ok(key)
}

#[cfg(target_os = "macos")]
fn aead_key(key: &[u8]) -> Result<LessSafeKey, String> {
    if key.len() != SEALING_KEY_BYTES {
        return Err("the local HTTPS Keychain sealing key has an invalid size".into());
    }
    UnboundKey::new(&AES_256_GCM, key)
        .map(LessSafeKey::new)
        .map_err(|_| "the local HTTPS Keychain sealing key is invalid".to_string())
}

#[cfg(target_os = "macos")]
fn seal_identity(encoded: &[u8], key: &[u8]) -> Result<Vec<u8>, String> {
    let key = aead_key(key)?;
    let mut nonce = [0_u8; NONCE_LEN];
    SystemRandom::new()
        .fill(&mut nonce)
        .map_err(|_| "could not generate the local HTTPS identity nonce".to_string())?;
    let mut ciphertext = Zeroizing::new(encoded.to_vec());
    key.seal_in_place_append_tag(
        Nonce::assume_unique_for_key(nonce),
        Aad::from(SEALED_IDENTITY_PREFIX),
        &mut *ciphertext,
    )
    .map_err(|_| "could not encrypt the local HTTPS identity".to_string())?;
    let mut sealed =
        Vec::with_capacity(SEALED_IDENTITY_PREFIX.len() + NONCE_LEN + ciphertext.len());
    sealed.extend_from_slice(SEALED_IDENTITY_PREFIX);
    sealed.extend_from_slice(&nonce);
    sealed.extend_from_slice(&ciphertext);
    Ok(sealed)
}

#[cfg(target_os = "macos")]
fn open_identity(sealed: &[u8], key: &[u8]) -> Result<Zeroizing<Vec<u8>>, String> {
    let prefix = SEALED_IDENTITY_PREFIX.len() + NONCE_LEN;
    if sealed.len() < prefix + AUTH_TAG_BYTES || !sealed.starts_with(SEALED_IDENTITY_PREFIX) {
        return Err("the encrypted local HTTPS identity has an unsupported shape".into());
    }
    let nonce: [u8; NONCE_LEN] = sealed[SEALED_IDENTITY_PREFIX.len()..prefix]
        .try_into()
        .unwrap();
    let mut plaintext = Zeroizing::new(sealed[prefix..].to_vec());
    let plaintext_length = aead_key(key)?
        .open_in_place(
            Nonce::assume_unique_for_key(nonce),
            Aad::from(SEALED_IDENTITY_PREFIX),
            &mut plaintext,
        )
        .map_err(|_| "the encrypted local HTTPS identity failed authentication".to_string())?
        .len();
    plaintext.truncate(plaintext_length);
    Ok(plaintext)
}

#[cfg(target_os = "macos")]
fn write_sealed_identity(path: &Path, sealed: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "the encrypted local HTTPS identity path has no parent".to_string())?;
    fs::create_dir_all(parent).map_err(|error| {
        format!("could not create the RCP desktop configuration directory: {error}")
    })?;
    let temporary = parent.join(format!(
        ".{SEALED_IDENTITY_FILENAME}.{}.tmp",
        uuid::Uuid::new_v4()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)
            .map_err(|error| {
                format!("could not create the encrypted local HTTPS identity: {error}")
            })?;
        file.write_all(sealed)
            .and_then(|()| file.sync_all())
            .map_err(|error| {
                format!("could not persist the encrypted local HTTPS identity: {error}")
            })?;
        fs::rename(&temporary, path).map_err(|error| {
            format!("could not activate the encrypted local HTTPS identity: {error}")
        })?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                format!("could not persist the local HTTPS identity directory: {error}")
            })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(target_os = "macos")]
pub fn install_webview_trust(
    window: &tauri::WebviewWindow,
    identity: &LocalHttpsIdentity,
) -> Result<(), String> {
    use std::{
        ffi::CString,
        os::raw::c_char,
        sync::{
            atomic::{AtomicI32, Ordering},
            Arc,
        },
    };

    #[link(name = "rcp_https_trust", kind = "static")]
    extern "C" {
        fn rcp_https_trust_install_pin(
            fingerprint_hex: *const c_char,
            webview: *mut libc::c_void,
        ) -> libc::c_int;
    }

    let fingerprint = CString::new(identity.fingerprint_sha256())
        .map_err(|_| "the local HTTPS fingerprint contains an invalid byte".to_string())?;
    let result = Arc::new(AtomicI32::new(-1));
    let callback_result = result.clone();
    window
        .with_webview(move |platform| {
            callback_result.store(
                unsafe { rcp_https_trust_install_pin(fingerprint.as_ptr(), platform.inner()) },
                Ordering::SeqCst,
            );
        })
        .map_err(|error| format!("could not access the RCP WebView for local HTTPS: {error}"))?;
    match result.load(Ordering::SeqCst) {
        0 => Ok(()),
        code => Err(format!(
            "could not install app-scoped local HTTPS trust (native code {code})"
        )),
    }
}

#[cfg(target_os = "macos")]
pub async fn install_team_session_cookie(
    window: &tauri::WebviewWindow,
    origin: &str,
    set_cookie: Zeroizing<String>,
) -> Result<(), String> {
    use std::{
        ffi::{c_void, CString},
        os::raw::{c_char, c_int},
    };

    type CookieCompletion = extern "C" fn(*mut c_void, c_int);

    #[link(name = "rcp_https_trust", kind = "static")]
    extern "C" {
        fn rcp_https_trust_set_team_cookie(
            webview: *mut c_void,
            origin: *const c_char,
            set_cookie: *const c_char,
            completion: CookieCompletion,
            context: *mut c_void,
        ) -> c_int;
    }

    extern "C" fn cookie_installed(context: *mut c_void, code: c_int) {
        let sender =
            unsafe { Box::from_raw(context.cast::<tokio::sync::oneshot::Sender<c_int>>()) };
        let _ = sender.send(code);
    }

    let origin =
        CString::new(origin).map_err(|_| "the team origin contains an invalid byte".to_string())?;
    if set_cookie.as_bytes().contains(&0) {
        return Err("the team session cookie contains an invalid byte".into());
    }
    let mut cookie_bytes = Zeroizing::new(set_cookie.as_bytes().to_vec());
    cookie_bytes.push(0);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let access = window.with_webview(move |platform| {
        let context = Box::into_raw(Box::new(sender)).cast::<c_void>();
        let code = unsafe {
            rcp_https_trust_set_team_cookie(
                platform.inner(),
                origin.as_ptr(),
                cookie_bytes.as_ptr().cast(),
                cookie_installed,
                context,
            )
        };
        if code != 0 {
            cookie_installed(context, code);
        }
    });
    if let Err(error) = access {
        return Err(format!(
            "could not access the RCP WebView for its team session: {error}"
        ));
    }
    match tokio::time::timeout(COOKIE_INSTALL_TIMEOUT, receiver).await {
        Ok(Ok(0)) => Ok(()),
        Ok(Ok(callback_code)) => Err(format!(
            "could not install the team browser session (completion code {callback_code})"
        )),
        Ok(Err(_)) => Err("the team browser session installer stopped unexpectedly".into()),
        Err(_) => Err("timed out installing the team browser session".into()),
    }
}

#[cfg(not(target_os = "macos"))]
pub fn install_webview_trust(
    _window: &tauri::WebviewWindow,
    _identity: &LocalHttpsIdentity,
) -> Result<(), String> {
    Err("app-scoped local HTTPS trust is supported only by the macOS desktop app".into())
}

#[cfg(not(target_os = "macos"))]
pub async fn install_team_session_cookie(
    _window: &tauri::WebviewWindow,
    _origin: &str,
    _set_cookie: Zeroizing<String>,
) -> Result<(), String> {
    Err("team browser sessions are supported only by the macOS desktop app".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_identity_round_trips_as_one_bounded_payload() {
        let identity = generate_identity().unwrap();
        assert_eq!(identity.fingerprint_sha256.len(), 64);
        assert!(identity
            .fingerprint_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
        let encoded = encode_identity(&identity).unwrap();
        let decoded = decode_identity(&encoded).unwrap();
        assert_eq!(decoded.certificate_der, identity.certificate_der);
        assert_eq!(decoded.private_key_der, identity.private_key_der);
        assert_eq!(decoded.fingerprint_sha256(), identity.fingerprint_sha256());
    }

    #[test]
    fn identity_decoder_rejects_truncation_versions_and_length_drift() {
        let identity = generate_identity().unwrap();
        let encoded = encode_identity(&identity).unwrap();
        let truncated = encoded[..encoded.len() - 1].to_vec();
        let mut unsupported_version = encoded.to_vec();
        unsupported_version[IDENTITY_HEADER.len()] = IDENTITY_VERSION + 1;
        let mut length_drift = encoded.to_vec();
        length_drift[IDENTITY_HEADER.len() + 1] ^= 1;
        for rejected in [truncated, unsupported_version, length_drift] {
            assert!(decode_identity(&rejected).is_err());
        }
    }

    #[test]
    fn identity_rejects_a_certificate_and_private_key_from_different_pairs() {
        let first = generate_identity().unwrap();
        let second = generate_identity().unwrap();
        assert!(
            identity_from_parts(first.certificate_der, second.private_key_der.to_vec()).is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sealed_identity_round_trips_and_rejects_tampering() {
        let identity = generate_identity().unwrap();
        let encoded = encode_identity(&identity).unwrap();
        let key = [7_u8; SEALING_KEY_BYTES];
        let sealed = seal_identity(&encoded, &key).unwrap();
        assert_eq!(open_identity(&sealed, &key).unwrap(), encoded);

        let mut tampered = sealed;
        *tampered.last_mut().unwrap() ^= 1;
        assert!(open_identity(&tampered, &key).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sealed_identity_reader_rejects_symlinks_and_oversized_files() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let regular = directory.path().join("identity.sealed");
        fs::write(&regular, b"sealed").unwrap();
        fs::set_permissions(&regular, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(
            read_sealed_identity(&regular).unwrap(),
            Some(b"sealed".to_vec())
        );

        let link = directory.path().join("identity-link.sealed");
        symlink(&regular, &link).unwrap();
        assert!(read_sealed_identity(&link).is_err());

        let oversized = directory.path().join("oversized.sealed");
        fs::write(
            &oversized,
            vec![0_u8; MAX_SEALED_IDENTITY_BYTES as usize + 1],
        )
        .unwrap();
        fs::set_permissions(&oversized, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_sealed_identity(&oversized).is_err());
    }
}
