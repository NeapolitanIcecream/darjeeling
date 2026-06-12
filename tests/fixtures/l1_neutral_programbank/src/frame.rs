use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frame {
    pub intent: String,
    #[serde(default)]
    pub slots: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_abstain: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Deserialize)]
pub struct Request {
    pub request_id: String,
    pub utterance: String,
}

#[derive(Debug, Serialize)]
pub struct L1Result {
    pub request_id: String,
    pub accepted: bool,
    pub frame: Option<Frame>,
    pub program_path: String,
    pub native_latency_us: u64,
    pub reason: String,
}

impl L1Result {
    pub fn abstain(
        request_id: impl Into<String>,
        reason: impl Into<String>,
        latency_us: u64,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            accepted: false,
            frame: None,
            program_path: "abstain".to_string(),
            native_latency_us: latency_us,
            reason: reason.into(),
        }
    }
}
