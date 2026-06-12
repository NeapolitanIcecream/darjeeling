pub mod frame;

use crate::frame::{L1Result, Request};
use std::time::Instant;

pub fn try_answer(request: &Request) -> L1Result {
    let started_at = Instant::now();
    let latency_us = started_at
        .elapsed()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX);
    L1Result::abstain(&request.request_id, "no native program configured", latency_us)
}

#[cfg(test)]
mod tests {
    use super::try_answer;
    use crate::frame::Request;

    #[test]
    fn abstains_without_target_programs() {
        let result = try_answer(&Request {
            request_id: "r1".to_string(),
            utterance: "any request".to_string(),
        });

        assert!(!result.accepted);
        assert!(result.frame.is_none());
        assert_eq!(result.reason, "no native program configured");
    }
}
