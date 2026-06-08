pub mod frame;
pub mod normalize;
pub mod programs;

use crate::frame::{L1Result, Request};
use crate::normalize::normalize;
use crate::programs::alarm::try_alarm_set;
use crate::programs::weather::try_weather_query;
use crate::programs::Candidate;
use std::time::Instant;

pub fn try_answer(request: &Request) -> L1Result {
    let started_at = Instant::now();
    let q = normalize(&request.utterance);
    let candidates = collect_candidates(&q);
    let latency_us = elapsed_us(started_at);

    if candidates.is_empty() {
        return L1Result::abstain(&request.request_id, "no native program matched", latency_us);
    }

    let first = &candidates[0];
    if candidates
        .iter()
        .all(|candidate| candidate.frame == first.frame)
    {
        let paths = candidates
            .iter()
            .map(|candidate| candidate.program_path)
            .collect::<Vec<_>>()
            .join(",");
        return L1Result {
            request_id: request.request_id.clone(),
            accepted: true,
            frame: Some(first.frame.clone()),
            program_path: paths,
            native_latency_us: latency_us,
            reason: "matched native program".to_string(),
        };
    }

    L1Result::abstain(
        &request.request_id,
        "conflicting native programs matched",
        latency_us,
    )
}

fn collect_candidates(q: &str) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    if let Some(candidate) = try_alarm_set(q) {
        candidates.push(candidate);
    }
    if let Some(candidate) = try_weather_query(q) {
        candidates.push(candidate);
    }
    candidates
}

fn elapsed_us(started_at: Instant) -> u64 {
    started_at
        .elapsed()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::try_answer;
    use crate::frame::Request;

    #[test]
    fn accepts_alarm_program() {
        let result = try_answer(&Request {
            request_id: "r1".to_string(),
            utterance: "Set an alarm for seven tomorrow morning".to_string(),
        });

        assert!(result.accepted);
        assert_eq!(result.frame.unwrap().intent, "alarm_set");
        assert!(result.native_latency_us < 10_000);
    }

    #[test]
    fn abstains_on_unknown_request() {
        let result = try_answer(&Request {
            request_id: "r2".to_string(),
            utterance: "play some jazz".to_string(),
        });

        assert!(!result.accepted);
        assert!(result.frame.is_none());
    }
}
