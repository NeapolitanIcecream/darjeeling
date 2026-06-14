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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FramePatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_intent: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub accepted_slots: BTreeMap<String, String>,
    #[serde(default = "l1_source_layer")]
    pub source_layer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub complete: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

fn l1_source_layer() -> String {
    "L1".to_string()
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch: Option<FramePatch>,
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
            patch: None,
            program_path: "abstain".to_string(),
            native_latency_us: latency_us,
            reason: reason.into(),
        }
    }
}
