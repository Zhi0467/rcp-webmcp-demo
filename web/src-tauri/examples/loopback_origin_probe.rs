use std::env;

use tauri::{webview::WebviewWindowBuilder, WebviewUrl};
use url::Url;

const FIRST_ORIGIN_VARIABLE: &str = "RCP_ORIGIN_PROBE_FIRST";
const SECOND_ORIGIN_VARIABLE: &str = "RCP_ORIGIN_PROBE_SECOND";
const LOGIN_VARIABLE: &str = "RCP_ORIGIN_PROBE_LOGIN";

fn main() {
    let first = origin_from_environment(FIRST_ORIGIN_VARIABLE);
    let second = origin_from_environment(SECOND_ORIGIN_VARIABLE);
    if !has_distinct_cookie_hosts(&first, &second) {
        panic!("the loopback origin probe requires two distinct cookie hosts");
    }
    let start = if env::var_os(LOGIN_VARIABLE).is_some() {
        first
            .join("login")
            .expect("the first probe origin must accept a login path")
    } else {
        first
            .join("resume")
            .expect("the first probe origin must accept a resume path")
    };
    let allowed = [
        first.origin().ascii_serialization(),
        second.origin().ascii_serialization(),
    ];

    tauri::Builder::default()
        .setup(move |app| {
            let allowed = allowed.clone();
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(start.clone()))
                .title("RCP loopback origin probe")
                .inner_size(920.0, 720.0)
                .on_navigation(move |candidate| {
                    let origin = candidate.origin().ascii_serialization();
                    let accepted = allowed.contains(&origin);
                    if !accepted {
                        eprintln!("[origin-probe] rejected navigation to {candidate}");
                    }
                    accepted
                })
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!("origin-probe.conf.json"))
        .expect("the loopback origin probe could not run");
}

fn origin_from_environment(variable: &str) -> Url {
    let raw = env::var(variable).unwrap_or_else(|_| panic!("{variable} is required"));
    let parsed = Url::parse(&raw).unwrap_or_else(|error| panic!("invalid {variable}: {error}"));
    if parsed.scheme() != "http"
        || parsed.username() != ""
        || parsed.password().is_some()
        || parsed.port().is_none()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed.origin().ascii_serialization() != raw
    {
        panic!("{variable} must be one canonical HTTP origin with an explicit port");
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
    fn distinct_origin_comparison_includes_the_cookie_host() {
        let first = Url::parse("http://rcp-a.localhost:39121").unwrap();
        let second = Url::parse("http://rcp-b.localhost:39122").unwrap();
        assert_ne!(first.origin(), second.origin());
        assert!(has_distinct_cookie_hosts(&first, &second));
    }

    #[test]
    fn different_ports_on_one_cookie_host_are_rejected() {
        let first = Url::parse("http://127.0.0.1:39121").unwrap();
        let second = Url::parse("http://127.0.0.1:39122").unwrap();
        assert_ne!(first.origin(), second.origin());
        assert!(!has_distinct_cookie_hosts(&first, &second));
    }
}
