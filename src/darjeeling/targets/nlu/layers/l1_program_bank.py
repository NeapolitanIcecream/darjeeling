from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from darjeeling.runtime.timing import elapsed_ms
from darjeeling.schemas import LayerResult
from darjeeling.targets.nlu.data import normalize_utterance
from darjeeling.targets.nlu.schemas import Frame

ALLOWED_OPERATORS = {
    "contains",
    "contains_all",
    "contains_any",
    "starts_with",
    "regex",
    "regex_extract",
    "not",
    "and",
    "or",
}


class ProgramRule(BaseModel):
    rule_id: str = Field(pattern=r"^[a-zA-Z0-9_]+$")
    description: str = ""
    condition: dict[str, Any]
    action: dict[str, Any]

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_condition_node(value)
        return value

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: dict[str, Any]) -> dict[str, Any]:
        if set(value) == {"abstain"}:
            return value
        if set(value) != {"accept"}:
            raise ValueError("action must be accept or abstain")
        accept = value["accept"]
        if not isinstance(accept, dict) or not accept.get("intent"):
            raise ValueError("accept action requires an intent")
        return value

    def try_frame(self, utterance: str) -> Frame | None:
        qn = normalize_utterance(utterance)
        matched, captures = _eval_condition(self.condition, qn)
        if not matched or "accept" not in self.action:
            return None

        accept = self.action["accept"]
        slots = dict(accept.get("slots") or {})
        if accept.get("slots_from_regex"):
            slots.update(captures)
        return Frame(intent=accept["intent"], slots=slots)


class ProgramBankLayer:
    def __init__(self, rules: list[ProgramRule] | None = None) -> None:
        self.rules = rules or []

    def try_answer(self, utterance: str) -> LayerResult:
        with elapsed_ms() as ms:
            for rule in self.rules:
                frame = rule.try_frame(utterance)
                if frame is not None:
                    return LayerResult(
                        layer="L1",
                        accepted=True,
                        frame=frame,
                        confidence=1.0,
                        reason=f"matched {rule.rule_id}",
                        latency_ms=ms(),
                    )
            return LayerResult(
                layer="L1",
                accepted=False,
                reason="no program matched",
                latency_ms=ms(),
            )


def _validate_condition_node(node: Any) -> None:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError("condition node must contain exactly one operator")
    op, value = next(iter(node.items()))
    if op not in ALLOWED_OPERATORS:
        raise ValueError(f"unsupported L1 operator: {op}")
    if op in {"and", "or"}:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{op} requires a non-empty condition list")
        for child in value:
            _validate_condition_node(child)
    elif op == "not":
        _validate_condition_node(value)
    elif op == "regex_extract":
        if not isinstance(value, dict) or "pattern" not in value or "slot_map" not in value:
            raise ValueError("regex_extract requires pattern and slot_map")


def _eval_condition(node: dict[str, Any], qn: str) -> tuple[bool, dict[str, str]]:
    op, value = next(iter(node.items()))
    if op == "contains":
        return str(value).lower() in qn, {}
    if op == "contains_all":
        return all(str(term).lower() in qn for term in value), {}
    if op == "contains_any":
        return any(str(term).lower() in qn for term in value), {}
    if op == "starts_with":
        return any(qn.startswith(str(prefix).lower()) for prefix in value), {}
    if op == "regex":
        return re.search(str(value), qn) is not None, {}
    if op == "regex_extract":
        match = re.search(str(value["pattern"]), qn)
        if match is None:
            return False, {}
        captures = {
            slot_name: match.group(group_name)
            for group_name, slot_name in value["slot_map"].items()
            if group_name in match.groupdict() and match.group(group_name) is not None
        }
        return True, captures
    if op == "not":
        matched, _captures = _eval_condition(value, qn)
        return not matched, {}
    if op == "and":
        captures: dict[str, str] = {}
        for child in value:
            matched, child_captures = _eval_condition(child, qn)
            if not matched:
                return False, {}
            captures.update(child_captures)
        return True, captures
    if op == "or":
        for child in value:
            matched, child_captures = _eval_condition(child, qn)
            if matched:
                return True, child_captures
        return False, {}
    raise ValueError(f"unsupported L1 operator: {op}")


def render_rule(rule: ProgramRule) -> str:
    return (
        f"# Generated from validated DSL rule {rule.rule_id}\n"
        f"RULE = {rule.model_dump(mode='json')!r}\n"
    )
