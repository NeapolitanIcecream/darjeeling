pub mod worker;

pub use worker::{L1Result, Request};

pub fn try_answer(request: &Request) -> L1Result {
    L1Result::abstain(&request.request_id, "no native program configured")
}

#[cfg(test)]
mod tests {
    use super::{try_answer, Request};
    use serde_json::json;

    #[test]
    fn abstains_without_target_programs() {
        let result = try_answer(&Request {
            request_id: "r1".to_string(),
            input: json!({"text": "any request"}),
        });

        assert!(!result.accepted);
        assert!(result.output.is_none());
        assert_eq!(result.reason, "no native program configured");
    }
}
