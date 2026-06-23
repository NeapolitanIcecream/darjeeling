import json
from types import SimpleNamespace

import pytest

import darjeeling.targets.nlu.clinc150_phase1 as clinc150_phase1
from darjeeling.targets.nlu.clinc150_phase1 import (
    Clinc150IntentTeacher,
    build_clinc150_gate_records,
    build_clinc150_label_cards,
    build_clinc150_stratified_records,
    clinc150_metrics_from_teacher_rows,
    compare_repeated_teacher_rows,
    evaluate_clinc150_l2,
    load_teacher_rows,
    run_clinc150_teacher_live_eval,
    select_l2_threshold,
    train_clinc150_l2,
    training_examples_from_gold_records,
    training_examples_from_teacher_rows,
    write_clinc150_l2_eval_artifacts,
    write_clinc150_l2_train_artifacts,
)
from darjeeling.targets.nlu.data import DataRecord
from darjeeling.targets.nlu.layers.l4_cloud_llm import TeacherCallResult
from darjeeling.targets.nlu.schemas import Frame, TaskSchema
from darjeeling.targets.nlu.settings import load_settings


def test_clinc150_gate_sample_is_intent_stratified_with_oos_tail() -> None:
    records = [
        _record("r1", "alpha one", "alpha", split="validation"),
        _record("r2", "alpha two", "alpha", split="validation"),
        _record("r3", "beta one", "beta", split="validation"),
        _record("r4", "beta two", "beta", split="validation"),
        _record("r5", "oos one", "out_of_scope", split="validation", abstain=True),
        _record("r6", "oos two", "out_of_scope", split="validation", abstain=True),
    ]

    sample = build_clinc150_gate_records(records, per_intent=1, oos_requests=1)

    assert [record.request_id for record in sample] == ["r1", "r3", "r5"]


def test_clinc150_label_cards_use_train_examples_only() -> None:
    records = [
        _record("train-1", "alpha train", "alpha"),
        _record("train-2", "alpha second", "alpha"),
        _record("train-3", "alpha third", "alpha"),
        _record("train-4", "not supported", "out_of_scope", abstain=True),
    ]

    cards = build_clinc150_label_cards(records, examples_per_label=2)

    assert cards == [
        {
            "intent": "alpha",
            "description": "alpha",
            "examples": ["alpha train", "alpha second"],
        },
        {
            "intent": "out_of_scope",
            "description": "unsupported or out-of-scope request",
            "examples": ["not supported"],
        },
    ]


def test_clinc150_stratified_sample_round_robins_intents() -> None:
    records = [
        _record("a1", "alpha one", "alpha"),
        _record("a2", "alpha two", "alpha"),
        _record("b1", "beta one", "beta"),
        _record("b2", "beta two", "beta"),
        _record("o1", "unsupported one", "out_of_scope", abstain=True),
    ]

    sample = build_clinc150_stratified_records(records, max_requests=5)

    assert [record.request_id for record in sample] == ["a1", "b1", "o1", "a2", "b2"]


def test_clinc150_teacher_metrics_split_in_scope_oos_and_gate() -> None:
    rows = [
        _teacher_row("r1", "alpha", "alpha"),
        _teacher_row("r2", "beta", "alpha"),
        _teacher_row("r3", "out_of_scope", "out_of_scope", abstain=True),
        _teacher_row("r4", "out_of_scope", "alpha", abstain=True),
    ]

    metrics = clinc150_metrics_from_teacher_rows(
        rows,
        min_overall_accuracy=0.5,
        min_in_scope_accuracy=0.5,
        max_parse_failure_rate=0.0,
    )

    assert metrics["overall_accuracy"] == pytest.approx(0.5)
    assert metrics["in_scope_accuracy"] == pytest.approx(0.5)
    assert metrics["oos_precision"] == pytest.approx(1.0)
    assert metrics["oos_recall"] == pytest.approx(0.5)
    assert metrics["passed_teacher_gate"] is True


def test_clinc150_teacher_uses_configured_completion_budget() -> None:
    settings = load_settings().model_copy(
        update={
            "teacher_prompt_version": "clinc150-intent-v1",
            "teacher_max_tokens": 192,
            "openai_model": "test-model",
        }
    )
    fake_client = _FakeClincClient()
    schema = TaskSchema(intent_names=["balance", "out_of_scope"], slot_names=[])

    result = Clinc150IntentTeacher(settings, client=fake_client).answer(
        "what is my balance",
        schema,
    )

    assert result.frame.intent == "balance"
    assert fake_client.completions.calls[0]["max_completion_tokens"] == 192
    assert fake_client.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_clinc150_repeat_consistency_compares_parsed_teacher_frames() -> None:
    first = [
        _teacher_row("r1", "alpha", "alpha"),
        _teacher_row("r2", "beta", "beta"),
    ]
    second = [
        _teacher_row("r1", "alpha", "alpha"),
        _teacher_row("r2", "beta", "alpha"),
    ]

    result = compare_repeated_teacher_rows(first, second)

    assert result["comparable_requests"] == 2
    assert result["consistent_requests"] == 1
    assert result["consistency"] == pytest.approx(0.5)


def test_clinc150_teacher_rows_convert_to_l2_examples_and_skip_failures() -> None:
    rows = [
        _teacher_row("r1", "alpha", "alpha"),
        {
            **_teacher_row("r2", "beta", "beta"),
            "parse_failure": True,
        },
        {
            **_teacher_row("r3", "gamma", "gamma"),
            "teacher_frame": None,
        },
    ]

    examples = training_examples_from_teacher_rows(rows)

    assert len(examples) == 1
    assert examples[0].utterance == "r1"
    assert examples[0].teacher_frame == Frame(intent="alpha", slots={}, is_abstain=False)


def test_clinc150_l2_eval_selects_high_precision_threshold() -> None:
    train_records = [
        _record("t1", "alpha train one", "alpha"),
        _record("t2", "alpha train two", "alpha"),
        _record("t3", "beta train one", "beta"),
        _record("t4", "beta train two", "beta"),
        _record("t5", "unsupported thing", "out_of_scope", abstain=True),
        _record("t6", "not in supported intents", "out_of_scope", abstain=True),
    ]
    eval_records = [
        _record("e1", "alpha train one", "alpha", split="validation"),
        _record("e2", "beta train two", "beta", split="validation"),
        _record(
            "e3",
            "not in supported intents",
            "out_of_scope",
            split="validation",
            abstain=True,
        ),
    ]
    bundle = train_clinc150_l2(
        training_examples_from_gold_records(train_records),
        accept_threshold=0.0,
    )

    result = evaluate_clinc150_l2(bundle=bundle, records=eval_records)

    assert result["requests"] == 3
    assert result["accuracy"] is not None
    selected = select_l2_threshold(
        [
            {
                "threshold": 0.5,
                "accepted_precision": 0.5,
                "accepted_coverage": 1.0,
                "lower_layer_oos_false_accept_rate": 0.0,
            },
            {
                "threshold": 0.9,
                "accepted_precision": 1.0,
                "accepted_coverage": 0.5,
                "lower_layer_oos_false_accept_rate": 0.0,
            },
        ]
    )
    assert selected is not None
    assert selected["threshold"] == 0.9


def test_clinc150_l2_eval_reports_fallback_cost_latency_and_artifacts(tmp_path) -> None:
    train_records = [
        _record("t1", "alpha train one", "alpha"),
        _record("t2", "alpha train two", "alpha"),
        _record("t3", "beta train one", "beta"),
        _record("t4", "beta train two", "beta"),
        _record("t5", "unsupported thing", "out_of_scope", abstain=True),
        _record("t6", "not in supported intents", "out_of_scope", abstain=True),
    ]
    eval_records = [
        _record("e1", "alpha train one", "alpha", split="validation"),
        _record("e2", "beta train two", "beta", split="validation"),
        _record(
            "e3",
            "not in supported intents",
            "out_of_scope",
            split="validation",
            abstain=True,
        ),
    ]
    teacher_rows = [
        _teacher_row("e1", "alpha", "alpha", tokens=20, cost_usd=0.02, latency_ms=100),
        _teacher_row("e2", "beta", "beta", tokens=30, cost_usd=0.03, latency_ms=200),
        _teacher_row(
            "e3",
            "out_of_scope",
            "out_of_scope",
            abstain=True,
            tokens=40,
            cost_usd=0.04,
            latency_ms=300,
        ),
    ]
    bundle = train_clinc150_l2(
        training_examples_from_gold_records(train_records),
        accept_threshold=0.0,
    )

    train_artifact = write_clinc150_l2_train_artifacts(
        bundle=bundle,
        examples=training_examples_from_gold_records(train_records),
        out_dir=tmp_path / "train",
        training_source="gold",
        split="train",
    )
    result = evaluate_clinc150_l2(
        bundle=bundle,
        records=eval_records,
        teacher_rows=teacher_rows,
        thresholds=(0.0,),
        include_prediction_rows=True,
    )
    eval_artifact = write_clinc150_l2_eval_artifacts(
        result=result,
        out_dir=tmp_path / "eval",
    )

    threshold = result["thresholds"][0]
    assert train_artifact.summary["examples"] == 6
    assert train_artifact.bundle_path.exists()
    assert result["all_l4_baseline"]["cost_usd_per_request"] == pytest.approx(0.03)
    assert threshold["all_l4_calls_per_100_requests"] == pytest.approx(100.0)
    assert threshold["l4_call_reduction_rate"] == pytest.approx(1.0)
    assert threshold["l4_cost_reduction_rate"] == pytest.approx(1.0)
    assert eval_artifact.summary_path.exists()
    assert eval_artifact.cost_latency_path.exists()
    assert eval_artifact.details_jsonl_path is not None
    assert eval_artifact.details_jsonl_path.exists()


def test_clinc150_teacher_eval_writes_manifest_details_and_cost_ledger(
    tmp_path,
    monkeypatch,
) -> None:
    _FakeSequenceClincTeacher.responses = [
        _clinc_call("alpha", tokens=18),
        _clinc_call("beta", tokens=20),
    ]
    _FakeSequenceClincTeacher.utterances = []
    monkeypatch.setattr(clinc150_phase1, "Clinc150IntentTeacher", _FakeSequenceClincTeacher)
    records = [
        _record("r1", "alpha request", "alpha", split="validation"),
        _record("r2", "beta request", "beta", split="validation"),
    ]

    result = run_clinc150_teacher_live_eval(
        records=records,
        task_schema=TaskSchema(intent_names=["alpha", "beta"], slot_names=[]),
        settings=load_settings(),
        split="validation",
        stream="sequential",
        prompt_version="clinc150-intent-v1",
        out_dir=tmp_path,
    )

    rows = load_teacher_rows(result.artifacts.details_jsonl_path)
    ledger = json.loads(result.artifacts.cost_ledger_path.read_text(encoding="utf-8"))
    assert _FakeSequenceClincTeacher.utterances == ["alpha request", "beta request"]
    assert [row["request_id"] for row in rows] == ["r1", "r2"]
    assert (tmp_path / "teacher_live_vs_gold.run.json").exists()
    assert ledger["observed_attempt_cost_usd"] == result.artifacts.summary["full_l4"]["cost_usd"]
    assert result.clinc_metrics["attempt_count"] == 2


def test_clinc150_teacher_eval_resume_skips_completed_rows(tmp_path, monkeypatch) -> None:
    _FakeSequenceClincTeacher.responses = [
        _clinc_call("alpha", tokens=18),
        _clinc_call("beta", tokens=20),
    ]
    _FakeSequenceClincTeacher.utterances = []
    monkeypatch.setattr(clinc150_phase1, "Clinc150IntentTeacher", _FakeSequenceClincTeacher)
    records = [
        _record("r1", "alpha request", "alpha", split="validation"),
        _record("r2", "beta request", "beta", split="validation"),
    ]
    schema = TaskSchema(intent_names=["alpha", "beta"], slot_names=[])

    initial = run_clinc150_teacher_live_eval(
        records=records,
        task_schema=schema,
        settings=load_settings(),
        split="validation",
        stream="sequential",
        prompt_version="clinc150-intent-v1",
        out_dir=tmp_path,
    )
    rows = load_teacher_rows(initial.artifacts.details_jsonl_path)
    initial.artifacts.details_jsonl_path.write_text(
        json.dumps(rows[0], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _FakeSequenceClincTeacher.responses = [_clinc_call("beta", tokens=20)]
    _FakeSequenceClincTeacher.utterances = []

    resumed = run_clinc150_teacher_live_eval(
        records=records,
        task_schema=schema,
        settings=load_settings(),
        split="validation",
        stream="sequential",
        prompt_version="clinc150-intent-v1",
        out_dir=tmp_path,
        resume_existing=True,
    )

    resumed_rows = load_teacher_rows(resumed.artifacts.details_jsonl_path)
    assert _FakeSequenceClincTeacher.utterances == ["beta request"]
    assert [row["request_id"] for row in resumed_rows] == ["r1", "r2"]


def test_clinc150_teacher_eval_resume_rejects_mismatched_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    _FakeSequenceClincTeacher.responses = [
        _clinc_call("alpha", tokens=18),
        _clinc_call("beta", tokens=20),
    ]
    monkeypatch.setattr(clinc150_phase1, "Clinc150IntentTeacher", _FakeSequenceClincTeacher)
    records = [
        _record("r1", "alpha request", "alpha", split="validation"),
        _record("r2", "beta request", "beta", split="validation"),
    ]
    schema = TaskSchema(intent_names=["alpha", "beta"], slot_names=[])
    run_clinc150_teacher_live_eval(
        records=records,
        task_schema=schema,
        settings=load_settings(),
        split="validation",
        stream="sequential",
        prompt_version="clinc150-intent-v1",
        out_dir=tmp_path,
    )
    manifest_path = tmp_path / "teacher_live_vs_gold.run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = "different-model"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model"):
        run_clinc150_teacher_live_eval(
            records=records,
            task_schema=schema,
            settings=load_settings(),
            split="validation",
            stream="sequential",
            prompt_version="clinc150-intent-v1",
            out_dir=tmp_path,
            resume_existing=True,
        )


def _record(
    request_id: str,
    utterance: str,
    intent: str,
    *,
    split: str = "train",
    abstain: bool = False,
) -> DataRecord:
    return DataRecord(
        request_id=request_id,
        utterance=utterance,
        split=split,
        gold_frame=Frame(intent=intent, slots={}, is_abstain=abstain),
    )


def _teacher_row(
    request_id: str,
    gold_intent: str,
    teacher_intent: str,
    *,
    abstain: bool = False,
    tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
) -> dict:
    gold_frame = Frame(intent=gold_intent, slots={}, is_abstain=abstain).model_dump(mode="json")
    teacher_frame = Frame(
        intent=teacher_intent,
        slots={},
        is_abstain=teacher_intent == "out_of_scope",
    ).model_dump(mode="json")
    return {
        "request_id": request_id,
        "utterance": request_id,
        "gold_frame": gold_frame,
        "teacher_frame": teacher_frame,
        "parse_failure": False,
        "frame_exact": gold_frame == teacher_frame,
        "intent_correct": gold_intent == teacher_intent,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }


class _FakeClincCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"intent": "balance"}))
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )


class _FakeClincClient:
    def __init__(self) -> None:
        self.completions = _FakeClincCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def _clinc_call(
    intent: str,
    *,
    tokens: int,
    attempt_diagnostics: list[dict] | None = None,
) -> TeacherCallResult:
    frame = Frame(intent=intent, slots={}, is_abstain=intent == "out_of_scope")
    return TeacherCallResult(
        frame=frame,
        raw_response=json.dumps({"intent": intent}),
        usage={
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens - tokens // 2,
            "total_tokens": tokens,
        },
        model="fake-clinc-teacher",
        context_hash="",
        prompt_cache_key="fake-cache-key",
        attempt_diagnostics=attempt_diagnostics or [],
    )


class _FakeSequenceClincTeacher:
    responses: list[TeacherCallResult | Exception] = []
    utterances: list[str] = []

    def __init__(self, settings, *, label_cards=None) -> None:
        del settings, label_cards

    def answer(self, utterance: str, task_schema: TaskSchema):
        del task_schema
        self.__class__.utterances.append(utterance)
        response = self.__class__.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
