//! Q11 experiment: can WKWebView keep a `Secure` session cookie on a local
//! HTTPS origin whose certificate the app generated for itself?
//!
//! The D2 probe proved plain HTTP loses the cookie. This example repeats that
//! drive over HTTPS and pins exactly one certificate through an app-scoped
//! trust hook, so nothing is added to a system-wide trust store. The third
//! origin is deliberately inside the navigation allowlist while presenting an
//! unpinned certificate, so a refusal there proves the pin and not the
//! allowlist.

use std::env;
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};

use tauri::{webview::WebviewWindowBuilder, WebviewUrl};
use url::Url;

const FIRST_ORIGIN_VARIABLE: &str = "RCP_HTTPS_PROBE_FIRST";
const SECOND_ORIGIN_VARIABLE: &str = "RCP_HTTPS_PROBE_SECOND";
const UNPINNED_ORIGIN_VARIABLE: &str = "RCP_HTTPS_PROBE_UNPINNED";
const FINGERPRINT_VARIABLE: &str = "RCP_HTTPS_PROBE_FINGERPRINT";
const LOGIN_VARIABLE: &str = "RCP_HTTPS_PROBE_LOGIN";

// Build-script link directives do not reach example targets, so the probe names
// its own native dependencies.
#[link(name = "Security", kind = "framework")]
extern "C" {}

#[link(name = "rcp_https_trust", kind = "static")]
extern "C" {
    fn rcp_https_trust_install(
        fingerprint_hex: *const c_char,
        webview: *mut c_void,
        start_url: *const c_char,
        reset_cookies: c_int,
    ) -> c_int;
    fn rcp_https_trust_stats(accepted: *mut c_int, refused: *mut c_int, delegated: *mut c_int);
}

fn main() {
    let first = origin_from_environment(FIRST_ORIGIN_VARIABLE);
    let second = origin_from_environment(SECOND_ORIGIN_VARIABLE);
    let unpinned = origin_from_environment(UNPINNED_ORIGIN_VARIABLE);
    if !has_distinct_cookie_hosts(&first, &second) {
        panic!("the local HTTPS probe requires two distinct cookie hosts");
    }
    let fingerprint = env::var(FINGERPRINT_VARIABLE)
        .unwrap_or_else(|_| panic!("{FINGERPRINT_VARIABLE} is required"));
    if fingerprint.len() != 64 || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        panic!("{FINGERPRINT_VARIABLE} must be one SHA-256 hex digest");
    }

    let start = if env::var_os(LOGIN_VARIABLE).is_some() {
        first.join("login").expect("login path")
    } else {
        first.join("resume").expect("resume path")
    };
    // The unpinned origin stays allowed here on purpose: only the certificate
    // pin may keep the WebView away from it.
    let allowed = [
        first.origin().ascii_serialization(),
        second.origin().ascii_serialization(),
        unpinned.origin().ascii_serialization(),
    ];

    tauri::Builder::default()
        .setup(move |app| {
            let allowed = allowed.clone();
            let window = WebviewWindowBuilder::new(
                app,
                "main",
                WebviewUrl::External(Url::parse("about:blank").expect("blank start page")),
            )
            .title("RCP local HTTPS origin probe")
            .inner_size(920.0, 720.0)
            .on_navigation(move |candidate| {
                if candidate.scheme() == "about" {
                    return true;
                }
                let origin = candidate.origin().ascii_serialization();
                let accepted = allowed.contains(&origin);
                eprintln!(
                    "[https-probe] navigation {} {candidate}",
                    if accepted { "allowed" } else { "rejected" }
                );
                accepted
            })
            .build()?;

            // The hook is added to the delegate of the live WKWebView, then the
            // first HTTPS navigation starts. Both happen on the main thread in
            // this closure, so no request can precede the pin.
            let pin = fingerprint.clone();
            let destination = start.to_string();
            // The login phase proves a cookie is established; the resume phase
            // proves the stored one survives a restart, so only login resets.
            let reset = c_int::from(env::var_os(LOGIN_VARIABLE).is_some());
            window.with_webview(move |platform| {
                let pinned = CString::new(pin.clone()).expect("fingerprint has no NUL");
                let target = CString::new(destination.clone()).expect("URL has no NUL");
                let installed = unsafe {
                    rcp_https_trust_install(
                        pinned.as_ptr(),
                        platform.inner(),
                        target.as_ptr(),
                        reset,
                    )
                };
                if installed != 0 {
                    eprintln!("[https-probe] FATAL: trust hook install returned {installed}");
                    std::process::exit(20 + installed);
                }
                eprintln!("[https-probe] app-scoped trust hook installed, pin={pin}");
            })?;
            Ok(())
        })
        .build(tauri::generate_context!("https-origin-probe.conf.json"))
        .expect("the local HTTPS origin probe could not start")
        .run(|_app, _event| {});

    let (mut accepted, mut refused, mut other) = (0, 0, 0);
    unsafe { rcp_https_trust_stats(&mut accepted, &mut refused, &mut other) };
    eprintln!("[https-probe] trust decisions: accepted={accepted} refused={refused} other={other}");
}

fn origin_from_environment(variable: &str) -> Url {
    let raw = env::var(variable).unwrap_or_else(|_| panic!("{variable} is required"));
    let parsed = Url::parse(&raw).unwrap_or_else(|error| panic!("invalid {variable}: {error}"));
    if parsed.scheme() != "https"
        || parsed.username() != ""
        || parsed.password().is_some()
        || parsed.port().is_none()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed.origin().ascii_serialization() != raw
    {
        panic!("{variable} must be one canonical HTTPS origin with an explicit port");
    }
    parsed
}

fn has_distinct_cookie_hosts(first: &Url, second: &Url) -> bool {
    first.host_str() != second.host_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_https_aliases_have_distinct_cookie_hosts() {
        let first = Url::parse("https://rcp-a.localhost:39131").unwrap();
        let second = Url::parse("https://rcp-b.localhost:39132").unwrap();
        assert!(has_distinct_cookie_hosts(&first, &second));
    }

    #[test]
    fn different_ports_on_one_https_host_are_rejected() {
        let first = Url::parse("https://127.0.0.1:39131").unwrap();
        let second = Url::parse("https://127.0.0.1:39132").unwrap();
        assert!(!has_distinct_cookie_hosts(&first, &second));
    }
}
