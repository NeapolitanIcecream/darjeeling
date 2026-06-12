from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class Frame(BaseModel):
    intent: str
    slots: dict[str, str] = Field(default_factory=dict)
    is_abstain: bool = False


@dataclass(frozen=True)
class TaskSchema:
    intent_names: list[str]
    slot_names: list[str]
    schema_version: str = "task-schema-v1"

    @classmethod
    def from_payload(cls, payload: dict) -> TaskSchema:
        return cls(
            intent_names=[str(intent) for intent in payload.get("intent_names", [])],
            slot_names=[str(slot) for slot in payload.get("slot_names", [])],
            schema_version=str(payload.get("schema_version", "task-schema-v1")),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "intent_names": list(self.intent_names),
            "slot_names": list(self.slot_names),
            "schema_version": self.schema_version,
        }
