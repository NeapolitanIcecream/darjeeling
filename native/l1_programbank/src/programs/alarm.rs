use crate::frame::Frame;
use crate::programs::Candidate;
use std::collections::BTreeMap;

pub fn try_alarm_set(q: &str) -> Option<Candidate> {
    if !(q.contains("alarm") || q.contains("wake me")) {
        return None;
    }
    if is_alarm_non_set_intent(q) || has_date_or_timeofday(q) {
        return None;
    }

    let time = extract_time(q)?;
    if !valid_time_only_slot(time) {
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

fn is_alarm_non_set_intent(q: &str) -> bool {
    q.contains("timer")
        || q.contains("stopwatch")
        || q.contains("remove")
        || q.contains("delete")
        || q.contains("cancel")
        || q.contains("stop ")
        || q.starts_with("stop")
        || q.contains("active alarm")
        || q.contains("active alarms")
        || q.contains("alarm times")
        || q.contains("alarms")
        || q.contains("show alarm")
        || q.contains("show me alarm")
        || q.contains("list alarm")
        || q.contains("do i have")
        || q.contains("reminder")
}

fn has_date_or_timeofday(q: &str) -> bool {
    [
        "tomorrow",
        "today",
        "tonight",
        "morning",
        "afternoon",
        "evening",
        "night",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "next ",
        "this week",
        "the week",
    ]
    .iter()
    .any(|term| q.contains(term))
}

fn valid_time_only_slot(value: &str) -> bool {
    let word_count = value.split_whitespace().count();
    word_count > 0
        && word_count <= 4
        && value
            .split_whitespace()
            .all(|token| is_time_token(token.trim_matches(|ch: char| ch == ',')))
}

fn is_time_token(token: &str) -> bool {
    matches!(
        token,
        "zero"
            | "oh"
            | "one"
            | "two"
            | "three"
            | "four"
            | "five"
            | "six"
            | "seven"
            | "eight"
            | "nine"
            | "ten"
            | "eleven"
            | "twelve"
            | "am"
            | "a.m."
            | "pm"
            | "p.m."
            | "o'clock"
            | "oclock"
    ) || token.chars().all(|ch| ch.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::try_alarm_set;

    #[test]
    fn extracts_alarm_time() {
        let result = try_alarm_set("set an alarm for seven").unwrap();

        assert_eq!(result.frame.intent, "alarm_set");
        assert_eq!(
            result.frame.slots.get("time").map(String::as_str),
            Some("seven")
        );
    }

    #[test]
    fn rejects_timer_collision() {
        assert!(try_alarm_set("set a timer alarm for seven minutes").is_none());
    }

    #[test]
    fn rejects_alarm_queries_and_removals() {
        assert!(try_alarm_set("show me what alarm times i've set for the week").is_none());
        assert!(try_alarm_set("give me the alarm times for the next two days").is_none());
        assert!(try_alarm_set(
            "kickball is over i do not need the alarm for kickball on wednesday evening any longer"
        )
        .is_none());
    }

    #[test]
    fn rejects_alarm_sets_that_need_date_or_timeofday_slots() {
        assert!(try_alarm_set("please set an alarm at seven am tomorrow morning").is_none());
        assert!(try_alarm_set("please set a reminder alarm for three p. m. on saturday").is_none());
        assert!(try_alarm_set("set the alarm for five o'clock in the morning").is_none());
    }
}
