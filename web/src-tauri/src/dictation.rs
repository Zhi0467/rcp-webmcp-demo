use std::ffi::{c_char, c_int, CStr, CString};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

const MAX_SESSION_ID_BYTES: usize = 128;

#[derive(Clone, Serialize)]
struct DictationResult<'a> {
    session_id: &'a str,
    text: &'a str,
    is_final: bool,
}

#[derive(Clone, Serialize)]
struct DictationState<'a> {
    session_id: &'a str,
    state: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

fn validate_session_id(session_id: &str) -> Result<(), String> {
    if session_id.trim().is_empty() {
        return Err("dictation session id cannot be empty".into());
    }
    if session_id.len() > MAX_SESSION_ID_BYTES {
        return Err("dictation session id is too long".into());
    }
    if session_id.contains('\0') {
        return Err("dictation session id contains an invalid character".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod platform {
    use std::sync::OnceLock;

    use super::*;

    static APP: OnceLock<AppHandle> = OnceLock::new();

    extern "C" {
        fn rcp_dictation_start(
            session_id: *const c_char,
            callback: extern "C" fn(
                *const c_char,
                *const c_char,
                *const c_char,
                c_int,
                *const c_char,
                *const c_char,
            ),
        ) -> c_int;
        fn rcp_dictation_stop(session_id: *const c_char) -> c_int;
        fn rcp_dictation_stop_active();
    }

    extern "C" fn handle_event(
        session_id: *const c_char,
        kind: *const c_char,
        text: *const c_char,
        is_final: c_int,
        state: *const c_char,
        error: *const c_char,
    ) {
        let Some(app) = APP.get() else {
            return;
        };
        let session_id = string_from_ptr(session_id);
        let kind = string_from_ptr(kind);
        match kind.as_str() {
            "result" => {
                let text = string_from_ptr(text);
                let _ = app.emit(
                    "rcp://dictation-result",
                    DictationResult {
                        session_id: &session_id,
                        text: &text,
                        is_final: is_final != 0,
                    },
                );
            }
            "state" => {
                let state = string_from_ptr(state);
                let error = optional_string_from_ptr(error);
                let _ = app.emit(
                    "rcp://dictation-state",
                    DictationState {
                        session_id: &session_id,
                        state: &state,
                        error: error.as_deref(),
                    },
                );
            }
            _ => {}
        }
    }

    fn string_from_ptr(value: *const c_char) -> String {
        if value.is_null() {
            return String::new();
        }
        // SAFETY: The Objective-C bridge invokes the callback synchronously with
        // null-terminated UTF-8 strings that remain alive for the duration of it.
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned()
    }

    fn optional_string_from_ptr(value: *const c_char) -> Option<String> {
        let value = string_from_ptr(value);
        (!value.is_empty()).then_some(value)
    }

    pub(super) fn start(app: &AppHandle, session_id: &str) -> Result<(), String> {
        APP.get_or_init(|| app.clone());
        let session_id = CString::new(session_id).map_err(|error| error.to_string())?;
        // SAFETY: The bridge copies the session id before returning and retains
        // the function pointer for synchronous event callbacks.
        let result = unsafe { rcp_dictation_start(session_id.as_ptr(), handle_event) };
        if result == 0 {
            Ok(())
        } else {
            Err("could not start Apple speech recognition".into())
        }
    }

    pub(super) fn stop(session_id: &str) -> Result<(), String> {
        let session_id = CString::new(session_id).map_err(|error| error.to_string())?;
        // SAFETY: The bridge only reads the session id during this call.
        let result = unsafe { rcp_dictation_stop(session_id.as_ptr()) };
        if result == 0 {
            Ok(())
        } else {
            Err("dictation session is no longer active".into())
        }
    }

    pub(super) fn stop_active() {
        // SAFETY: The bridge serializes all state changes on the main queue.
        unsafe { rcp_dictation_stop_active() };
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub(super) fn start(_app: &AppHandle, _session_id: &str) -> Result<(), String> {
        Err("dictation is available only in the macOS desktop app".into())
    }

    pub(super) fn stop(_session_id: &str) -> Result<(), String> {
        Err("dictation is available only in the macOS desktop app".into())
    }

    pub(super) fn stop_active() {}
}

pub fn start(app: &AppHandle, session_id: &str) -> Result<(), String> {
    validate_session_id(session_id)?;
    platform::start(app, session_id)
}

pub fn stop(session_id: &str) -> Result<(), String> {
    validate_session_id(session_id)?;
    platform::stop(session_id)
}

pub fn stop_active() {
    platform::stop_active();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_or_oversized_session_ids() {
        assert!(validate_session_id("  ").is_err());
        assert!(validate_session_id(&"x".repeat(MAX_SESSION_ID_BYTES + 1)).is_err());
    }

    #[test]
    fn accepts_normal_session_ids() {
        assert!(validate_session_id("dictation-52f3").is_ok());
    }

    #[test]
    fn macos_bundles_explain_both_permissions() {
        for plist in [
            include_str!("../Info.plist"),
            include_str!("../Info.dev.template.plist"),
        ] {
            assert!(plist.contains("NSMicrophoneUsageDescription"));
            assert!(plist.contains("NSSpeechRecognitionUsageDescription"));
        }
    }
}
