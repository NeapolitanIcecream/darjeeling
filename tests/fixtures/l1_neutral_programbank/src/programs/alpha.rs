use crate::frame::Frame;
use crate::programs::Candidate;
use std::collections::BTreeMap;

pub fn try_alpha_accept(q: &str) -> Option<Candidate> {
    let value = q.strip_prefix("alpha accept ")?;
    let value = clean_slot_value(value)?;
    if !is_supported_value(value) {
        return None;
    }

    let mut slots = BTreeMap::new();
    slots.insert("slot_alpha".to_string(), value.to_string());
    Some(Candidate {
        frame: Frame {
            intent: "intent_alpha".to_string(),
            slots,
            is_abstain: false,
        },
        program_path: "programs/alpha::try_alpha_accept",
    })
}

fn clean_slot_value(value: &str) -> Option<&str> {
    let trimmed = value
        .trim()
        .trim_matches(|ch: char| matches!(ch, '.' | ',' | '!' | '?'));
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn is_supported_value(value: &str) -> bool {
    let tokens = value.split_whitespace().collect::<Vec<_>>();
    !tokens.is_empty()
        && tokens.len() <= 3
        && tokens.iter().all(|token| {
            token
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        })
}

#[cfg(test)]
mod tests {
    use super::try_alpha_accept;

    #[test]
    fn extracts_neutral_alpha_slot() {
        let result = try_alpha_accept("alpha accept red").unwrap();

        assert_eq!(result.frame.intent, "intent_alpha");
        assert_eq!(
            result.frame.slots.get("slot_alpha").map(String::as_str),
            Some("red")
        );
    }

    #[test]
    fn rejects_values_outside_fixture_contract() {
        assert!(try_alpha_accept("alpha accept").is_none());
        assert!(try_alpha_accept("alpha accept one two three four").is_none());
        assert!(try_alpha_accept("alpha accept red/blue").is_none());
    }
}
