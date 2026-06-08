import json
from pathlib import Path

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore
from darjeeling.data.massive import DataRecord
from darjeeling.layers.l2_student import L2StudentConfig, L2TrainingExample, train_l2_student
from darjeeling.layers.l3_local_slm import L3PromptArtifact
from darjeeling.runtime.replay import run_replay
from darjeeling.schemas import Frame
from darjeeling.settings import load_settings


def test_run_replay_writes_traces_with_rust_l1_and_l4_cache(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()

    records = [
        DataRecord(
            request_id="r1",
            locale="en-US",
            split="train",
            utterance="set an alarm for seven",
            annotated_utterance="set an alarm for [time : seven]",
            template="set an alarm for [time]",
            gold_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        ),
        DataRecord(
            request_id="r2",
            locale="en-US",
            split="train",
            utterance="play some jazz",
            annotated_utterance="play some jazz",
            template="play some jazz",
            gold_frame=Frame(intent="music_play"),
        ),
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "teacher_cache.jsonl").write_text(
        json.dumps(
            {
                "utterance": "play some jazz",
                "teacher_frame": {"intent": "music_play", "slots": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    settings = load_settings()
    settings.l1_rust_crate_dir = Path("native/l1_programbank")

    summary = run_replay(
        stream="sequential",
        max_requests=2,
        teacher_mode="cache",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=settings,
    )

    assert summary.requests == 2
    traces = [
        json.loads(line)
        for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chosen_layers = {trace["utterance"]: trace["chosen_layer"] for trace in traces}
    assert chosen_layers["set an alarm for seven"] == "L1"
    assert chosen_layers["play some jazz"] == "L4"
    jazz_layers = [result["layer"] for result in traces[1]["layer_results"]]
    assert jazz_layers == ["L0", "L1", "L3", "L4"]
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
            locale="en-US",
            split="train",
            utterance="play some jazz",
            annotated_utterance="play some jazz",
            template="play some jazz",
            gold_frame=Frame(intent="music_play"),
        )
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    examples = [
        L2TrainingExample(utterance="play some jazz", teacher_frame=Frame(intent="music_play")),
        L2TrainingExample(utterance="play music", teacher_frame=Frame(intent="music_play")),
        L2TrainingExample(utterance="start playlist", teacher_frame=Frame(intent="music_play")),
        L2TrainingExample(utterance="set alarm for seven", teacher_frame=Frame(intent="alarm_set")),
        L2TrainingExample(utterance="wake me at eight", teacher_frame=Frame(intent="alarm_set")),
        L2TrainingExample(utterance="alarm at nine", teacher_frame=Frame(intent="alarm_set")),
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
    assert trace["final_frame"]["intent"] == "music_play"


def test_run_replay_loads_promoted_l3_prompt_artifact(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    l3_dir = run_dir / "artifacts" / "generations" / "gen_001" / "l3"
    data_dir.mkdir()
    l3_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text(
        DataRecord(
            request_id="r1",
            locale="en-US",
            split="train",
            utterance="play some jazz",
            annotated_utterance="play some jazz",
            template="play some jazz",
            gold_frame=Frame(intent="music_play"),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "teacher_cache.jsonl").write_text(
        json.dumps(
            {
                "utterance": "play some jazz",
                "teacher_frame": {"intent": "music_play", "slots": {}},
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
