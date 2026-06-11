from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from darjeeling.schemas import Frame


class DataRecord(BaseModel):
    request_id: str
    locale: str
    split: str
    utterance: str
    annotated_utterance: str
    template: str
    gold_frame: Frame
    metadata: dict[str, Any] = Field(default_factory=dict)
