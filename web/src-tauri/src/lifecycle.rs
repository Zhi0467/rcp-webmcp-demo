use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LaunchOutcome {
    pub outcome: String,
    pub base_url: String,
    pub instance_id: Option<String>,
    pub version: String,
    pub owned: bool,
    pub reason: Option<String>,
}

impl LaunchOutcome {
    pub fn parse(line: &str) -> Result<Self, String> {
        let value: Self = serde_json::from_str(line)
            .map_err(|error| format!("backend returned invalid launch JSON: {error}"))?;
        match value.outcome.as_str() {
            "owned" if value.owned && value.instance_id.is_some() => Ok(value),
            "reused" if !value.owned && value.instance_id.is_some() => Ok(value),
            outcome
                if outcome.starts_with("refused-") && !value.owned && value.reason.is_some() =>
            {
                Ok(value)
            }
            _ => Err(format!(
                "backend returned unsupported launch outcome: {}",
                value.outcome
            )),
        }
    }

    pub fn is_refusal(&self) -> bool {
        self.outcome.starts_with("refused-")
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Health {
    pub status: String,
    pub pid: u32,
    pub version: String,
    pub instance_id: String,
    pub data_dir_id: String,
    pub owner_kind: String,
    #[serde(default)]
    pub active_agent_tasks: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DesktopStatus {
    pub desktop: bool,
    pub version: String,
    pub base_url: String,
    pub instance_id: String,
    pub data_dir_id: String,
    pub owner_kind: String,
    pub active_agent_tasks: u64,
    pub owned: bool,
}

impl DesktopStatus {
    pub fn from_ready(outcome: &LaunchOutcome, health: &Health) -> Result<Self, String> {
        let instance_id = outcome
            .instance_id
            .as_deref()
            .ok_or_else(|| "backend launch omitted its instance id".to_string())?;
        if health.status != "ok"
            || health.instance_id != instance_id
            || health.version != outcome.version
        {
            return Err("backend health does not match the launched instance".to_string());
        }
        Ok(Self {
            desktop: true,
            version: health.version.clone(),
            base_url: outcome.base_url.clone(),
            instance_id: instance_id.to_string(),
            data_dir_id: health.data_dir_id.clone(),
            owner_kind: health.owner_kind.clone(),
            active_agent_tasks: health.active_agent_tasks,
            owned: outcome.owned,
        })
    }

    pub(crate) fn matches_health(&self, health: &Health) -> bool {
        health.status == "ok"
            && health.instance_id == self.instance_id
            && health.data_dir_id == self.data_dir_id
            && health.version == self.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn health() -> Health {
        Health {
            status: "ok".into(),
            pid: 4242,
            version: "0.3.0".into(),
            instance_id: "instance-a".into(),
            data_dir_id: "data-a".into(),
            owner_kind: "desktop".into(),
            active_agent_tasks: 2,
        }
    }

    #[test]
    fn parses_owned_and_reused_launches() {
        for (outcome, owned) in [("owned", true), ("reused", false)] {
            let raw = format!(
                r#"{{"outcome":"{outcome}","base_url":"http://127.0.0.1:8421","instance_id":"instance-a","version":"0.3.0","owned":{owned}}}"#
            );
            let parsed = LaunchOutcome::parse(&raw).unwrap();
            assert_eq!(parsed.outcome, outcome);
            assert_eq!(parsed.owned, owned);
        }
    }

    #[test]
    fn refusal_requires_a_reason() {
        let raw = r#"{"outcome":"refused-version","base_url":"http://127.0.0.1:8421","instance_id":null,"version":"0.3.0","owned":false}"#;
        assert!(LaunchOutcome::parse(raw).is_err());
    }

    #[test]
    fn launch_outcome_rejects_contradictory_ownership() {
        for raw in [
            r#"{"outcome":"owned","base_url":"http://127.0.0.1:8421","instance_id":"instance-a","version":"0.3.0","owned":false}"#,
            r#"{"outcome":"reused","base_url":"http://127.0.0.1:8421","instance_id":"instance-a","version":"0.3.0","owned":true}"#,
            r#"{"outcome":"refused-version","base_url":"http://127.0.0.1:8421","instance_id":null,"version":"0.3.0","owned":true,"reason":"wrong version"}"#,
        ] {
            assert!(LaunchOutcome::parse(raw).is_err(), "accepted {raw}");
        }
    }

    #[test]
    fn cached_status_matches_only_the_full_health_identity() {
        let outcome = LaunchOutcome::parse(
            r#"{"outcome":"owned","base_url":"http://127.0.0.1:8421","instance_id":"instance-a","version":"0.3.0","owned":true}"#,
        )
        .unwrap();
        let status = DesktopStatus::from_ready(&outcome, &health()).unwrap();
        assert!(status.matches_health(&health()));

        let mut mismatches = Vec::new();
        let mut unhealthy = health();
        unhealthy.status = "starting".into();
        mismatches.push(unhealthy);
        let mut replaced = health();
        replaced.instance_id = "instance-b".into();
        mismatches.push(replaced);
        let mut other_data_dir = health();
        other_data_dir.data_dir_id = "data-b".into();
        mismatches.push(other_data_dir);
        let mut other_version = health();
        other_version.version = "0.4.0".into();
        mismatches.push(other_version);

        for mismatch in mismatches {
            assert!(!status.matches_health(&mismatch));
        }
    }

    #[test]
    fn reused_backend_is_never_owned() {
        let outcome = LaunchOutcome::parse(
            r#"{"outcome":"reused","base_url":"http://127.0.0.1:8421","instance_id":"instance-a","version":"0.3.0","owned":false}"#,
        )
        .unwrap();
        let status = DesktopStatus::from_ready(&outcome, &health()).unwrap();
        assert!(!status.owned);
    }
}
