import pytest

from darjeeling.layers.l1_program_bank import ProgramRule, render_rule


def test_l1_dsl_rule_matches_and_extracts_slots() -> None:
    rule = ProgramRule.model_validate(
        {
            "rule_id": "alarm_set_001",
            "description": "alarm commands with explicit time",
            "condition": {
                "and": [
                    {"contains_any": ["alarm", "wake me"]},
                    {
                        "regex_extract": {
                            "pattern": "(?:for|at) (?P<time>.+)$",
                            "slot_map": {"time": "time"},
                        }
                    },
                ]
            },
            "action": {
                "accept": {
                    "intent": "alarm_set",
                    "slots_from_regex": True,
                }
            },
        }
    )

    frame = rule.try_frame("Set an alarm for seven tomorrow morning")

    assert frame is not None
    assert frame.intent == "alarm_set"
    assert frame.slots == {"time": "seven tomorrow morning"}
    assert "alarm_set_001" in render_rule(rule)


def test_l1_dsl_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="unsupported L1 operator"):
        ProgramRule.model_validate(
            {
                "rule_id": "bad_001",
                "condition": {"decision_tree": {"depth": 3}},
                "action": {"accept": {"intent": "alarm_set"}},
            }
        )
