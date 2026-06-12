use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Deserialize)]
pub struct Request {
    pub request_id: String,
    #[serde(default)]
    pub input: Value,
}

#[derive(Debug, Serialize)]
pub struct L1Result {
    pub request_id: String,
    pub accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    pub program_path: String,
    pub native_latency_us: u64,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub metadata: Map<String, Value>,
}

impl L1Result {
    pub fn abstain(request_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            accepted: false,
            output: None,
            program_path: "abstain".to_string(),
            native_latency_us: 0,
            reason: reason.into(),
            metadata: Map::new(),
        }
    }
}
