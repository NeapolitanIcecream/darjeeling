from __future__ import annotations

from darjeeling.data.frames import normalize_utterance
from darjeeling.schemas import Frame, TeacherTrace


def exact_cache_from_teacher_traces(traces: list[TeacherTrace]) -> dict[str, Frame]:
    return {
        normalize_utterance(trace.utterance): trace.teacher_frame
        for trace in traces
        if trace.teacher_frame is not None
    }
