from pathlib import Path


def test_nlu_schema_terms_do_not_leak_into_core_python() -> None:
    repo = Path(__file__).resolve().parents[3]
    source_root = repo / "src" / "darjeeling"
    forbidden_terms = [
        "Frame",
        "intent",
        "slot",
        "utterance",
        "teacher_frame",
        "gold_frame",
        "MASSIVE",
        "NLU",
    ]
    offenders = []
    for path in source_root.rglob("*.py"):
        if "targets" in path.relative_to(source_root).parts:
            continue
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in text:
                offenders.append(f"{path.relative_to(repo)} contains {term!r}")
    assert offenders == []
