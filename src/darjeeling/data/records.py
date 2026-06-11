from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from darjeeling.schemas import Frame


class DataRecord(BaseModel):
    request_id: str
    utterance: str
    gold_frame: Frame
    split: str = "train"
    locale: str = ""
    workload_group_key: str | None = None
    annotated_utterance: str | None = None
    template: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
