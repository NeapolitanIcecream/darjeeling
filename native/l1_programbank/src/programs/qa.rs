use crate::frame::Frame;
use crate::programs::Candidate;
use std::collections::BTreeMap;

pub fn try_qa_person_age(q: &str) -> Option<Candidate> {
    let person = q.strip_prefix("how old is ")?;
    let person = clean_person(person)?;
    if !is_safe_person_name(person) {
        return None;
    }

    let mut slots = BTreeMap::new();
    slots.insert("person".to_string(), person.to_string());
    Some(Candidate {
        frame: Frame {
            intent: "qa_factoid".to_string(),
            slots,
            is_abstain: false,
        },
        program_path: "programs/qa::try_qa_person_age",
    })
}

fn clean_person(value: &str) -> Option<&str> {
    let trimmed = value
        .trim()
        .trim_matches(|ch: char| matches!(ch, '?' | '.' | ',' | '!'));
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn is_safe_person_name(value: &str) -> bool {
    let tokens = value.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() || tokens.len() > 4 {
        return false;
    }
    let first = tokens[0];
    if matches!(
        first,
        "a" | "an" | "the" | "my" | "your" | "our" | "this" | "that" | "it" | "he" | "she"
    ) {
        return false;
    }
    if tokens.iter().any(|token| is_non_person_object_token(token)) {
        return false;
    }
    tokens.iter().all(|token| {
        token
            .chars()
            .all(|ch| ch.is_ascii_alphabetic() || matches!(ch, '\'' | '-'))
    })
}

fn is_non_person_object_token(token: &str) -> bool {
    matches!(
        token,
        "earth"
            | "world"
            | "universe"
            | "country"
            | "city"
            | "company"
            | "building"
            | "bridge"
            | "planet"
            | "dog"
            | "cat"
    )
}

#[cfg(test)]
mod tests {
    use super::try_qa_person_age;

    #[test]
    fn extracts_person_age_question() {
        let result = try_qa_person_age("how old is carrie underwood").unwrap();

        assert_eq!(result.frame.intent, "qa_factoid");
        assert_eq!(
            result.frame.slots.get("person").map(String::as_str),
            Some("carrie underwood")
        );
    }

    #[test]
    fn strips_trailing_question_mark() {
        let result = try_qa_person_age("how old is david bowie?").unwrap();

        assert_eq!(
            result.frame.slots.get("person").map(String::as_str),
            Some("david bowie")
        );
    }

    #[test]
    fn rejects_non_person_objects() {
        assert!(try_qa_person_age("how old is the earth").is_none());
        assert!(try_qa_person_age("how old is my dog").is_none());
        assert!(try_qa_person_age("how old is it").is_none());
    }
}
