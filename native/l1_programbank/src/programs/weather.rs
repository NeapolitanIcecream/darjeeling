use crate::frame::Frame;
use crate::programs::Candidate;
use std::collections::BTreeMap;

pub fn try_weather_query(q: &str) -> Option<Candidate> {
    if !(q.contains("weather") || q.contains("forecast")) {
        return None;
    }
    if q.contains("whether") {
        return None;
    }
    if !is_generic_no_slot_weather_query(q) {
        return None;
    }

    Some(Candidate {
        frame: Frame {
            intent: "weather_query".to_string(),
            slots: BTreeMap::new(),
            is_abstain: false,
        },
        program_path: "programs/weather::try_weather_query",
    })
}

fn is_generic_no_slot_weather_query(q: &str) -> bool {
    matches!(
        q,
        "weather"
            | "the weather"
            | "what is the weather"
            | "what's the weather"
            | "tell me the weather"
            | "show me the weather"
            | "what is weather"
    )
}

#[cfg(test)]
mod tests {
    use super::try_weather_query;

    #[test]
    fn accepts_generic_no_slot_weather_query() {
        let result = try_weather_query("tell me the weather").unwrap();

        assert_eq!(result.frame.intent, "weather_query");
        assert!(result.frame.slots.is_empty());
    }

    #[test]
    fn rejects_weather_queries_that_need_slots() {
        assert!(try_weather_query("weather in leisure city").is_none());
        assert!(try_weather_query("what will the weather be on friday").is_none());
        assert!(try_weather_query("give me the weekly weather near me").is_none());
    }
}
