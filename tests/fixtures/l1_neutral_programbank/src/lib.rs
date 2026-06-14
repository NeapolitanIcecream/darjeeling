pub mod frame;
pub mod normalize;
pub mod programs;

use crate::frame::{FramePatch, L1Result, Request};
use crate::normalize::normalize;
use crate::programs::alpha::try_alpha_accept;
use crate::programs::Candidate;
use std::time::Instant;

pub fn try_answer(request: &Request) -> L1Result {
    let started_at = Instant::now();
    let q = normalize(&request.utterance);
    if q == "alpha intent" {
        return L1Result {
            request_id: request.request_id.clone(),
            accepted: true,
            frame: None,
            patch: Some(FramePatch {
                accepted_intent: Some("intent_alpha".to_string()),
                accepted_slots: Default::default(),
                source_layer: "L1".to_string(),
                confidence: Some(1.0),
                complete: false,
                metadata: Default::default(),
            }),
            program_path: "programs/alpha::try_alpha_intent_patch".to_string(),
            native_latency_us: elapsed_us(started_at),
            reason: "matched native intent patch".to_string(),
        };
    }
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
            patch: None,
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
    if let Some(candidate) = try_alpha_accept(q) {
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
    fn accepts_neutral_alpha_program() {
        let result = try_answer(&Request {
            request_id: "r1".to_string(),
            utterance: "Alpha accept red".to_string(),
        });

        assert!(result.accepted);
        let frame = result.frame.unwrap();
        assert_eq!(frame.intent, "intent_alpha");
        assert_eq!(
            frame.slots.get("slot_alpha").map(String::as_str),
            Some("red")
        );
        assert!(result.native_latency_us < 10_000);
    }

    #[test]
    fn accepts_neutral_alpha_intent_patch() {
        let result = try_answer(&Request {
            request_id: "r4".to_string(),
            utterance: "alpha intent".to_string(),
        });

        assert!(result.accepted);
        assert!(result.frame.is_none());
        let patch = result.patch.unwrap();
        assert_eq!(patch.accepted_intent.as_deref(), Some("intent_alpha"));
        assert!(!patch.complete);
    }

    #[test]
    fn abstains_on_out_of_contract_alpha_request() {
        let result = try_answer(&Request {
            request_id: "r3".to_string(),
            utterance: "Alpha accept one two three four".to_string(),
        });

        assert!(!result.accepted);
        assert!(result.frame.is_none());
    }

    #[test]
    fn abstains_on_unknown_request() {
        let result = try_answer(&Request {
            request_id: "r2".to_string(),
            utterance: "beta request".to_string(),
        });

        assert!(!result.accepted);
        assert!(result.frame.is_none());
    }
}
