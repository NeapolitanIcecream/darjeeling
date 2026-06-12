import json
from pathlib import Path

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore
from darjeeling.layers.l2_student import L2StudentConfig, L2TrainingExample, train_l2_student
from darjeeling.layers.l3_local_slm import L3PromptArtifact
from darjeeling.runtime.replay import run_replay
from darjeeling.schemas import Frame
from darjeeling.settings import load_settings
from darjeeling.targets.nlu.data import DataRecord


def test_run_replay_writes_traces_with_rust_l1_and_l4_cache(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()

    records = [
        DataRecord(
            request_id="r1",
            utterance="alpha request",
            gold_frame=Frame(intent="intent_alpha"),
        ),
        DataRecord(
            request_id="r2",
            utterance="beta sample request",
            gold_frame=Frame(intent="intent_beta"),
        ),
        DataRecord(
            request_id="r3",
            utterance="gamma request",
            gold_frame=Frame(intent="intent_gamma"),
        ),
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "teacher_cache.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "utterance": "alpha request",
                        "teacher_frame": {"intent": "intent_alpha", "slots": {}},
                    }
                ),
                json.dumps(
                    {
                        "utterance": "beta sample request",
                        "teacher_frame": {"intent": "intent_beta", "slots": {}},
                    }
                ),
                json.dumps(
                    {
                        "utterance": "gamma request",
                        "teacher_frame": {"intent": "intent_gamma", "slots": {}},
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings()

    summary = run_replay(
        stream="sequential",
        max_requests=3,
        teacher_mode="cache",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=settings,
    )

    assert summary.requests == 3
    traces = [
        json.loads(line)
        for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chosen_layers = {trace["utterance"]: trace["chosen_layer"] for trace in traces}
    assert chosen_layers["alpha request"] == "L4"
    assert chosen_layers["gamma request"] == "L4"
    assert chosen_layers["beta sample request"] == "L4"
    beta_layers = [result["layer"] for result in traces[1]["layer_results"]]
    assert beta_layers == ["L0", "L1", "L3", "L4"]
    assert traces[1]["layer_results"][2]["metadata"]["actual_mode"] == "disabled"
    assert traces[0]["gold_frame"] is not None


def test_run_replay_uses_l2_artifact_between_l1_and_l4(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    l2_dir = run_dir / "artifacts" / "generations" / "gen_001" / "l2"
    data_dir.mkdir()
    l2_dir.mkdir(parents=True)

    records = [
        DataRecord(
            request_id="r1",
            utterance="beta sample request",
            gold_frame=Frame(intent="intent_beta"),
        )
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    examples = [
        L2TrainingExample(
            utterance="beta sample request",
            teacher_frame=Frame(intent="intent_beta"),
        ),
        L2TrainingExample(utterance="beta request", teacher_frame=Frame(intent="intent_beta")),
        L2TrainingExample(
            utterance="beta alternate request",
            teacher_frame=Frame(intent="intent_beta"),
        ),
        L2TrainingExample(
            utterance="alpha request value alpha",
            teacher_frame=Frame(intent="intent_alpha"),
        ),
        L2TrainingExample(
            utterance="alpha variant value beta",
            teacher_frame=Frame(intent="intent_alpha"),
        ),
        L2TrainingExample(
            utterance="alpha variant value gamma",
            teacher_frame=Frame(intent="intent_alpha"),
        ),
    ]
    bundle = train_l2_student(
        examples,
        L2StudentConfig(accept_threshold=0.0, min_examples=4),
    )
    bundle.save(l2_dir / "l2_student.joblib")
    ArtifactStore(run_dir / "artifacts").promote(
        ArtifactManifest(
            artifact_set_id="gen_001_l2",
            generation=1,
            artifact_paths={"l2_student": "generations/gen_001/l2/l2_student.joblib"},
            promotion_reason="test fixture",
        )
    )

    summary = run_replay(
        stream="sequential",
        max_requests=1,
        teacher_mode="cache",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=load_settings(),
    )

    assert summary.layer_counts["L2"] == 1
    trace = json.loads((run_dir / "traces.jsonl").read_text(encoding="utf-8"))
    assert trace["chosen_layer"] == "L2"
    assert trace["final_frame"]["intent"] == "intent_beta"


def test_run_replay_uses_l2_target_artifact(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    l2_dir = run_dir / "artifacts" / "generations" / "gen_001" / "l2"
    target_dir = l2_dir / "target"
    data_dir.mkdir()
    target_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text(
        DataRecord(
            request_id="r1",
            utterance="beta sample request",
            gold_frame=Frame(intent="intent_beta", slots={"slot_beta": "value beta"}),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "teacher_cache.jsonl").write_text(
        json.dumps(
            {
                "utterance": "beta sample request",
                "teacher_frame": {
                    "intent": "intent_beta",
                    "slots": {"slot_beta": "value beta"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = [
        L2TrainingExample(
            utterance="beta sample request",
            teacher_frame=Frame(intent="intent_beta"),
        ),
        L2TrainingExample(utterance="beta request", teacher_frame=Frame(intent="intent_beta")),
        L2TrainingExample(
            utterance="beta alternate request",
            teacher_frame=Frame(intent="intent_beta"),
        ),
        L2TrainingExample(
            utterance="alpha request value alpha",
            teacher_frame=Frame(intent="intent_alpha"),
        ),
        L2TrainingExample(
            utterance="alpha variant value beta",
            teacher_frame=Frame(intent="intent_alpha"),
        ),
        L2TrainingExample(
            utterance="alpha variant value gamma",
            teacher_frame=Frame(intent="intent_alpha"),
        ),
    ]
    bundle = train_l2_student(
        examples,
        L2StudentConfig(accept_threshold=0.0, min_examples=4),
    )
    bundle.save(l2_dir / "l2_student.joblib")
    (target_dir / "target_l2.py").write_text(
        """
def postprocess_frame(utterance, frame, metadata):
    del metadata
    if utterance == "beta sample request":
        updated = dict(frame)
        updated["slots"] = {"slot_beta": "value beta"}
        return updated
    return frame
""",
        encoding="utf-8",
    )
    ArtifactStore(run_dir / "artifacts").promote(
        ArtifactManifest(
            artifact_set_id="gen_001_l2_target",
            generation=1,
            artifact_paths={
                "l2_student": "generations/gen_001/l2/l2_student.joblib",
                "l2_target": "generations/gen_001/l2/target/target_l2.py",
            },
            promotion_reason="test fixture",
        )
    )

    summary = run_replay(
        stream="sequential",
        max_requests=1,
        teacher_mode="cache",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=load_settings(),
    )

    assert summary.layer_counts["L2"] == 1
    trace = json.loads((run_dir / "traces.jsonl").read_text(encoding="utf-8"))
    assert trace["chosen_layer"] == "L2"
    assert trace["final_frame"] == {
        "intent": "intent_beta",
        "slots": {"slot_beta": "value beta"},
        "is_abstain": False,
    }
    assert trace["layer_results"][2]["metadata"]["target_postprocessed"] is True


def test_run_replay_loads_promoted_l3_prompt_artifact(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    l3_dir = run_dir / "artifacts" / "generations" / "gen_001" / "l3"
    data_dir.mkdir()
    l3_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text(
        DataRecord(
            request_id="r1",
            utterance="beta sample request",
            gold_frame=Frame(intent="intent_beta"),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "teacher_cache.jsonl").write_text(
        json.dumps(
            {
                "utterance": "beta sample request",
                "teacher_frame": {"intent": "intent_beta", "slots": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (l3_dir / "l3_prompt.json").write_text(
        L3PromptArtifact(
            prompt_version="fixture-l3-prompt",
            system_prompt="Return JSON only.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    ArtifactStore(run_dir / "artifacts").promote(
        ArtifactManifest(
            artifact_set_id="gen_001_l3",
            generation=1,
            artifact_paths={"l3_prompt": "generations/gen_001/l3/l3_prompt.json"},
            promotion_reason="test fixture",
        )
    )

    run_replay(
        stream="sequential",
        max_requests=1,
        teacher_mode="cache",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=load_settings(),
    )

    trace = json.loads((run_dir / "traces.jsonl").read_text(encoding="utf-8"))
    l3_result = next(result for result in trace["layer_results"] if result["layer"] == "L3")
    assert l3_result["metadata"]["prompt_version"] == "fixture-l3-prompt"
