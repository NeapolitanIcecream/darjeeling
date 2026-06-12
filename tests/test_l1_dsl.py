import pytest

from darjeeling.layers.l1_program_bank import ProgramRule, render_rule


def test_l1_dsl_rule_matches_and_extracts_slots() -> None:
    rule = ProgramRule.model_validate(
        {
            "rule_id": "intent_alpha_001",
                "description": "alpha requests with explicit slot value",
                "condition": {
                    "and": [
                        {"contains_any": ["alpha request", "alpha wake"]},
                        {
                            "regex_extract": {
                                "pattern": "(?:for|at) (?P<slot_alpha>.+)$",
                                "slot_map": {"slot_alpha": "slot_alpha"},
                            }
                        },
                ]
            },
            "action": {
                "accept": {
                    "intent": "intent_alpha",
                    "slots_from_regex": True,
                }
            },
        }
    )

    frame = rule.try_frame("Alpha request for value alpha extended")

    assert frame is not None
    assert frame.intent == "intent_alpha"
    assert frame.slots == {"slot_alpha": "value alpha extended"}
    assert "intent_alpha_001" in render_rule(rule)


def test_l1_dsl_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="unsupported L1 operator"):
        ProgramRule.model_validate(
            {
                "rule_id": "bad_001",
                "condition": {"decision_tree": {"depth": 3}},
                "action": {"accept": {"intent": "intent_alpha"}},
            }
        )
