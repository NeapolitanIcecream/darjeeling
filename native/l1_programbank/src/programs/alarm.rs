use crate::frame::Frame;
use crate::programs::Candidate;
use std::collections::BTreeMap;

pub fn try_alarm_set(q: &str) -> Option<Candidate> {
    if !(q.contains("alarm") || q.contains("wake me")) {
        return None;
    }
    if q.contains("timer") || q.contains("stopwatch") {
        return None;
    }

    let time = extract_time(q)?;
    if !valid_short_slot(time) {
        return None;
    }

    let mut slots = BTreeMap::new();
    slots.insert("time".to_string(), time.to_string());
    Some(Candidate {
        frame: Frame {
            intent: "alarm_set".to_string(),
            slots,
            is_abstain: false,
        },
        program_path: "programs/alarm::try_alarm_set",
    })
}

fn extract_time(q: &str) -> Option<&str> {
    for marker in [" for ", " at "] {
        if let Some(index) = q.rfind(marker) {
            let value = q[index + marker.len()..].trim();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn valid_short_slot(value: &str) -> bool {
    let word_count = value.split_whitespace().count();
    word_count > 0 && word_count <= 10
}

#[cfg(test)]
mod tests {
    use super::try_alarm_set;

    #[test]
    fn extracts_alarm_time() {
        let result = try_alarm_set("set an alarm for seven tomorrow morning").unwrap();

        assert_eq!(result.frame.intent, "alarm_set");
        assert_eq!(
            result.frame.slots.get("time").map(String::as_str),
            Some("seven tomorrow morning")
        );
    }

    #[test]
    fn rejects_timer_collision() {
        assert!(try_alarm_set("set a timer alarm for seven minutes").is_none());
    }
}
