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

    Some(Candidate {
        frame: Frame {
            intent: "weather_query".to_string(),
            slots: BTreeMap::new(),
            is_abstain: false,
        },
        program_path: "programs/weather::try_weather_query",
    })
}
