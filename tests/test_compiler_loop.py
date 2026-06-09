import json
from pathlib import Path
from types import SimpleNamespace

from darjeeling.artifacts.store import ArtifactStore
from darjeeling.compiler.l2_tuner import L2TuneResult
from darjeeling.compiler.loop import run_compiler_generation
from darjeeling.data.massive import DataRecord
from darjeeling.runtime.replay import load_l0_layer_from_manifest, run_replay
from darjeeling.schemas import Frame, LayerResult, TraceRecord
from darjeeling.settings import load_settings


def test_compiler_generation_promotes_l0_cache_without_gold_leakage(tmp_path: Path) -> None:
    train_trace = TraceRecord(
        request_id="r1",
        utterance="play some jazz",
        gold_frame=Frame(intent="music_play"),
        teacher_frame=Frame(intent="music_play"),
        chosen_layer="L4",
        final_frame=Frame(intent="music_play"),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="music_play"),
                latency_ms=1.0,
            )
        ],
    )
    holdout_trace = train_trace.model_copy(update={"request_id": "r2"})

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=[train_trace, holdout_trace],
        settings=load_settings(),
    )

    assert result.promoted
    manifest = ArtifactStore(tmp_path / "artifacts").load_current_manifest()
    assert manifest is not None
    l0_path = tmp_path / "artifacts" / manifest.artifact_paths["l0_cache"]
    hard_buffer_path = tmp_path / "artifacts" / manifest.artifact_paths["hard_buffer"]
    metrics_path = tmp_path / "artifacts" / manifest.artifact_paths["candidate_metrics_csv"]
    promotion_path = tmp_path / "artifacts" / manifest.artifact_paths["promotion_record"]
    l0_payload = l0_path.read_text(encoding="utf-8")
    assert "gold_frame" not in l0_payload
    assert "music_play" in l0_payload
    assert "gold_frame" not in hard_buffer_path.read_text(encoding="utf-8")
    assert "teacher_train_size" in metrics_path.read_text(encoding="utf-8")
    assert manifest.candidate_metrics["hard_buffer_size"] == 2
    assert manifest.candidate_metrics["hard_buffer_visibility_counts"] == {
        "replay_only": 1,
        "train_visible": 1,
    }
    assert manifest.candidate_metrics["hard_buffer_agent_context_size"] == 1
    assert manifest.candidate_metrics["promotion_eval_hard_buffer_size"] == 2
    promotion_payload = json.loads(promotion_path.read_text(encoding="utf-8"))
    assert promotion_payload["promoted"] is True
    assert promotion_payload["promotion_reason"] == "objective improved within gates"

    l0_layer = load_l0_layer_from_manifest(tmp_path)
    l0_result = l0_layer.try_answer("play some jazz")
    assert l0_result.accepted
    assert l0_result.frame == Frame(intent="music_play")


def test_compiler_generation_rejects_candidate_without_replay_coverage(
    tmp_path: Path,
) -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="play some jazz",
        gold_frame=Frame(intent="music_play"),
        teacher_frame=Frame(intent="music_play"),
        chosen_layer="L4",
        final_frame=Frame(intent="music_play"),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="music_play"),
                latency_ms=1.0,
            )
        ],
    )

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=[trace],
        settings=load_settings(),
    )

    assert not result.promoted
    assert result.reason == "promotion replay coverage is empty"
    assert ArtifactStore(tmp_path / "artifacts").load_current_manifest() is None
    generation_dir = tmp_path / "artifacts" / "generations" / "gen_001"
    assert (generation_dir / "manifest.json").exists()
    promotion_payload = json.loads((generation_dir / "promotion.json").read_text(encoding="utf-8"))
    assert promotion_payload["promoted"] is False
    assert promotion_payload["promotion_reason"] == "promotion replay coverage is empty"
    assert (generation_dir / "candidate_metrics.csv").exists()
    manifest_payload = json.loads((generation_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "hard_buffer" in manifest_payload["artifact_paths"]
    assert manifest_payload["candidate_metrics"]["hard_buffer_size"] == 1
    assert manifest_payload["candidate_metrics"]["hard_buffer_visibility_counts"] == {
        "train_visible": 1
    }
    assert manifest_payload["candidate_metrics"]["promotion_eval_hard_buffer_size"] == 0


def test_compiler_generation_records_l2_guard_threshold_search(tmp_path: Path) -> None:
    settings = load_settings()
    traces = [
        _teacher_trace("m1", "play jazz", "music_play"),
        _teacher_trace("m2", "play music", "music_play"),
        _teacher_trace("m3", "start playlist", "music_play"),
        _teacher_trace("m4", "play songs", "music_play"),
        _teacher_trace("a1", "set alarm for seven", "alarm_set"),
        _teacher_trace("a2", "wake me at eight", "alarm_set"),
        _teacher_trace("a3", "alarm at nine", "alarm_set"),
        _teacher_trace("a4", "set morning alarm", "alarm_set"),
    ]

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=traces,
        settings=settings,
    )

    assert result.manifest is not None
    metrics = result.manifest.candidate_metrics
    assert metrics["l2_trained"] is True
    assert metrics["l4_proposal_mode"] == "disabled"
    assert "l2_guard_threshold" in metrics
    assert "l2_guard_search" in metrics
    assert "l2_unguarded_train" in metrics
    assert metrics["l2_unguarded_train"]["threshold"] == 0.0
    assert metrics["l2_unguarded_train"]["accepted"] == metrics["l2_unguarded_train"]["total"]
    assert metrics["l2_runtime_enabled"] is False
    assert metrics["l2_min_runtime_examples"] == settings.l2_min_runtime_examples
    assert metrics["l2_guard_search"]["selected"]["threshold"] == metrics["l2_guard_threshold"]


def test_compiler_generation_uses_optuna_l2_tuning_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_tune_l2_student(traces, *, base_config, spec):
        tuned_config = base_config.model_copy(
            update={
                "slot_model_family": "none",
                "max_features": 1234,
                "word_ngram_range": (1, 3),
            }
        )
        captured["trace_ids"] = [trace.request_id for trace in traces]
        captured["spec"] = spec
        return L2TuneResult(
            train_size=4,
            validation_size=2,
            split_policy=spec.split_policy,
            n_trials_requested=spec.n_trials,
            n_trials_completed=1,
            best_trial_number=0,
            best_value=1.5,
            best_config=tuned_config.model_dump(mode="json"),
            best_metrics={
                "unguarded": {"accepted_accuracy": 1.0, "total": 2},
                "guarded": {"coverage": 0.5, "wrong_accept_rate": 0.0},
                "p95_latency_ms": 1.0,
            },
            trials=[],
        )

    monkeypatch.setattr("darjeeling.compiler.loop.tune_l2_student", fake_tune_l2_student)
    settings = load_settings().model_copy(
        update={
            "l2_tuning_mode": "optuna",
            "l2_tuning_trials": 3,
            "l2_tuning_min_examples": 2,
            "l2_tuning_search_space": "wide",
        }
    )
    traces = [
        _teacher_trace("m1", "play jazz", "music_play"),
        _teacher_trace("a1", "set alarm for seven", "alarm_set"),
        _teacher_trace("m2", "play music", "music_play"),
        _teacher_trace("a2", "wake me at eight", "alarm_set"),
        _teacher_trace("m3", "start playlist", "music_play"),
        _teacher_trace("a3", "alarm at nine", "alarm_set"),
        _teacher_trace("m4", "play songs", "music_play"),
        _teacher_trace("a4", "set morning alarm", "alarm_set"),
    ]

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=traces,
        settings=settings,
    )

    assert result.manifest is not None
    metrics = result.manifest.candidate_metrics
    assert captured["trace_ids"] == ["m1", "a1", "m2", "a2", "m3", "a3"]
    assert captured["spec"].n_trials == 3
    assert captured["spec"].search_space == "wide"
    assert captured["spec"].split_policy == "chronological"
    assert metrics["l2_tuning_mode"] == "optuna"
    assert metrics["l2_tuning"]["split_policy"] == "chronological"
    assert metrics["l2_tuning_succeeded"] is True
    assert metrics["l2_tuning"]["best_value"] == 1.5
    assert metrics["l2_config"]["slot_model_family"] == "none"
    assert metrics["l2_config"]["max_features"] == 1234
    assert "l2_tuning" in result.manifest.artifact_paths


def test_compiler_generation_can_train_l2_on_observed_lower_misses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_tune_l2_student(traces, *, base_config, spec):
        captured["trace_ids"] = [trace.request_id for trace in traces]
        return L2TuneResult(
            train_size=4,
            validation_size=2,
            split_policy=spec.split_policy,
            n_trials_requested=spec.n_trials,
            n_trials_completed=1,
            best_trial_number=0,
            best_value=1.0,
            best_config=base_config.model_dump(mode="json"),
            best_metrics={"unguarded": {"accepted_accuracy": 1.0}},
            trials=[],
        )

    monkeypatch.setattr("darjeeling.compiler.loop.tune_l2_student", fake_tune_l2_student)
    settings = load_settings().model_copy(
        update={
            "l2_training_scope": "lower_miss",
            "l2_tuning_mode": "optuna",
            "l2_tuning_trials": 2,
            "l2_tuning_min_examples": 2,
        }
    )
    traces = [
        _trace_with_lower_result("m0", "play cached jazz", "music_play", lower_layer="L0"),
        _trace_with_lower_result("a0", "cached alarm", "alarm_set", lower_layer="L1"),
        _trace_with_lower_result("m1", "play jazz", "music_play", lower_layer=None),
        _trace_with_lower_result("m2", "play music", "music_play", lower_layer=None),
        _trace_with_lower_result("a1", "set alarm for seven", "alarm_set", lower_layer=None),
        _trace_with_lower_result("a2", "wake me at eight", "alarm_set", lower_layer=None),
        _trace_with_lower_result("m3", "start playlist", "music_play", lower_layer=None),
        _trace_with_lower_result("a3", "alarm at nine", "alarm_set", lower_layer=None),
    ]

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=traces,
        settings=settings,
    )

    assert result.manifest is not None
    metrics = result.manifest.candidate_metrics
    assert captured["trace_ids"] == ["m1", "m2", "a1", "a2"]
    assert metrics["l2_training_scope"] == "lower_miss"
    assert metrics["l2_teacher_train_traces"] == 6
    assert metrics["l2_lower_miss_train_traces"] == 4
    assert metrics["l2_training_traces"] == 4
    assert metrics["l2_examples"] == 4
    assert "l2_unguarded_teacher_train" in metrics


def test_compiler_generation_uses_live_l4_l2_proposal_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeProposalAdapter:
        def __init__(self, settings) -> None:
            self.settings = settings

        def propose(self, **kwargs):
            if kwargs["role"] == "l3":
                return SimpleNamespace(
                    proposal={
                        "system_prompt": "Return strict JSON for local L3.",
                        "confidence_threshold": 0.82,
                        "few_shot_trace_ids": ["m1"],
                    },
                    context_hash="l3-ctx-hash",
                    prompt_cache_key="l3-cache-key",
                    source_trace_ids=["m1"],
                )
            if kwargs["role"] == "guard":
                return SimpleNamespace(
                    proposal={
                        "threshold_grid_start": 0.5,
                        "threshold_grid_stop": 0.9,
                        "threshold_grid_steps": 5,
                        "max_wrong_accept_rate": 0.04,
                        "rationale": "widen search for this window",
                    },
                    context_hash="guard-ctx-hash",
                    prompt_cache_key="guard-cache-key",
                    source_trace_ids=["m1", "a1"],
                )
            assert kwargs["role"] == "l2"
            return SimpleNamespace(
                proposal={"slot_model_family": "none", "max_features": 1234},
                context_hash="ctx-hash",
                prompt_cache_key="cache-key",
                source_trace_ids=["m1", "m2"],
            )

    monkeypatch.setattr("darjeeling.compiler.loop.L4ProposalAdapter", FakeProposalAdapter)
    settings = load_settings()
    settings.l4_proposal_mode = "live"

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=[
            _teacher_trace("m1", "play jazz", "music_play"),
            _teacher_trace("m2", "play music", "music_play"),
            _teacher_trace("m3", "start playlist", "music_play"),
            _teacher_trace("m4", "play songs", "music_play"),
            _teacher_trace("a1", "set alarm for seven", "alarm_set"),
            _teacher_trace("a2", "wake me at eight", "alarm_set"),
            _teacher_trace("a3", "alarm at nine", "alarm_set"),
            _teacher_trace("a4", "set morning alarm", "alarm_set"),
        ],
        settings=settings,
    )

    assert result.manifest is not None
    metrics = result.manifest.candidate_metrics
    assert metrics["l4_proposal_mode"] == "live"
    assert metrics["l4_l2_proposal_succeeded"] is True
    assert metrics["l4_l2_proposal"] == {"slot_model_family": "none", "max_features": 1234}
    assert metrics["l4_l2_proposal_context_hash"] == "ctx-hash"
    assert metrics["l4_guard_proposal_succeeded"] is True
    assert metrics["guard_search_spec"]["threshold_grid_start"] == 0.5
    assert metrics["guard_search_spec"]["threshold_grid_stop"] == 0.9
    assert metrics["guard_search_spec"]["threshold_grid_steps"] == 5
    assert metrics["guard_search_spec"]["max_wrong_accept_rate"] == 0.04
    assert "guard_candidate" in result.manifest.artifact_paths
    guard_path = tmp_path / "artifacts" / result.manifest.artifact_paths["guard_candidate"]
    guard_payload = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guard_payload["rationale"] == "widen search for this window"
    searched_thresholds = [
        candidate["threshold"] for candidate in metrics["l2_guard_search"]["candidates"]
    ]
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        assert threshold in searched_thresholds
    assert len(searched_thresholds) > 5
    assert metrics["l4_l3_prompt_proposal_succeeded"] is True
    assert metrics["l3_prompt_candidate_runtime_promoted"] is False
    assert "l3_prompt_candidate" in result.manifest.artifact_paths
    assert "l3_prompt" not in result.manifest.artifact_paths
    l3_prompt_path = tmp_path / "artifacts" / result.manifest.artifact_paths["l3_prompt_candidate"]
    l3_prompt_payload = json.loads(l3_prompt_path.read_text(encoding="utf-8"))
    assert l3_prompt_payload["system_prompt"] == "Return strict JSON for local L3."
    assert l3_prompt_payload["few_shot_examples"][0]["trace_id"] == "m1"
    assert "gold_frame" not in l3_prompt_path.read_text(encoding="utf-8")


def test_compiler_generation_respects_no_l2_ablation(tmp_path: Path) -> None:
    settings = load_settings()
    settings.l2_enabled = False

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=_two_intent_traces(),
        settings=settings,
    )

    assert result.manifest is not None
    metrics = result.manifest.candidate_metrics
    assert metrics["l2_enabled"] is False
    assert metrics["l2_trained"] is False
    assert metrics["l2_training_error"] == "L2 disabled by settings"
    assert "l2_student" not in result.manifest.artifact_paths


def test_compiler_generation_respects_no_guard_ablation(tmp_path: Path) -> None:
    settings = load_settings()
    settings.l2_guard_mode = "always_accept"

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=_two_intent_traces(),
        settings=settings,
    )

    assert result.manifest is not None
    metrics = result.manifest.candidate_metrics
    assert metrics["l2_guard_mode"] == "always_accept"
    assert metrics["l2_guard_threshold"] == 0.0
    assert metrics["l2_runtime_enabled"] is True
    assert metrics["l2_guard_search"]["mode"] == "always_accept"


def test_compiler_generation_records_l2_agent_patch_artifacts(tmp_path: Path) -> None:
    patch_path = tmp_path / "l2_agent.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/tests/test_l2_agent_marker.py b/tests/test_l2_agent_marker.py",
                "new file mode 100644",
                "--- /dev/null",
                "+++ b/tests/test_l2_agent_marker.py",
                "@@ -0,0 +1 @@",
                "+MARKER = 'compiler dry run marker'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings()
    settings.l2_agent_mode = "dry-run"
    settings.l2_agent_dry_run_patch = patch_path
    settings.l2_agent_run_validation = False

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=_two_intent_traces(),
        settings=settings,
    )

    assert result.manifest is not None
    manifest = result.manifest
    metrics = manifest.candidate_metrics
    assert metrics["l2_agent_mode"] == "dry-run"
    assert metrics["l2_agent_succeeded"] is True
    assert metrics["l2_agent_patch_runtime_applied"] is False
    assert "outer process apply/restart" in metrics["l2_agent_patch_runtime_reason"]
    assert "l2_agent_diff" in manifest.artifact_paths
    assert "l2_agent_transcript" in manifest.artifact_paths
    assert "l2_agent_provenance" in manifest.artifact_paths
    l2_agent_dir = tmp_path / "artifacts" / manifest.artifact_paths["l2_agent_dir"]
    context_families = json.loads(
        (l2_agent_dir / "contexts" / "l2_context_families.json").read_text(
            encoding="utf-8"
        )
    )
    assert context_families["schema_version"] == "l2-context-families-v1"
    assert "gold_frame" not in json.dumps(context_families)
    provenance = json.loads(
        (tmp_path / "artifacts" / manifest.artifact_paths["l2_agent_provenance"]).read_text(
            encoding="utf-8"
        )
    )
    assert provenance["runtime_patch_applied"] is False


def test_compiler_generation_uses_l2_mlp_settings(tmp_path: Path) -> None:
    settings = load_settings()
    settings.l2_intent_model_family = "mlp"
    settings.l2_mlp_hidden_layer_sizes = (8,)
    settings.l2_max_iter = 300

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=_two_intent_traces(),
        settings=settings,
    )

    assert result.manifest is not None
    config_payload = result.manifest.candidate_metrics["l2_config"]
    assert config_payload["intent_model_family"] == "mlp"
    assert config_payload["mlp_hidden_layer_sizes"] == [8]
    assert config_payload["max_iter"] == 300


def test_compiler_generation_force_promote_records_original_reason(tmp_path: Path) -> None:
    settings = load_settings()
    settings.force_promote_artifacts = True

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=_two_intent_traces(),
        settings=settings,
    )

    assert result.promoted
    assert result.reason.startswith("force promoted by settings after:")
    manifest = ArtifactStore(tmp_path / "artifacts").load_current_manifest()
    assert manifest is not None
    assert manifest.candidate_metrics["force_promote_artifacts"] is True
    assert "force_promote_original_reason" in manifest.candidate_metrics


def test_run_replay_compile_every_promotes_l0_for_repeated_teacher_trace(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()
    records = [
        DataRecord(
            request_id="r1",
            locale="en-US",
            split="train",
            utterance="play some jazz",
            annotated_utterance="play some jazz",
            template="play some jazz",
            gold_frame=Frame(intent="music_play"),
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
        DataRecord(
            request_id="r3",
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

    summary = run_replay(
        stream="sequential",
        max_requests=3,
        teacher_mode="cache",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=load_settings(),
        compile_every=2,
    )

    assert summary.layer_counts["L4"] == 2
    assert summary.layer_counts["L0"] == 1
    manifest = ArtifactStore(run_dir / "artifacts").load_current_manifest()
    assert manifest is not None
    assert manifest.candidate_metrics["l0_cache_lines"] == 1

    traces = [
        json.loads(line)
        for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert traces[0]["chosen_layer"] == "L4"
    assert traces[1]["chosen_layer"] == "L4"
    assert traces[2]["chosen_layer"] == "L0"


def test_run_replay_promotes_l1_agent_candidate_for_next_window(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()
    patch_path = tmp_path / "music_l1.patch"
    patch_path.write_text(_music_l1_patch(), encoding="utf-8")
    records = [
        DataRecord(
            request_id="r1",
            locale="en-US",
            split="train",
            utterance="play smooth jazz",
            annotated_utterance="play smooth jazz",
            template="play smooth jazz",
            gold_frame=Frame(intent="music_play"),
        ),
        DataRecord(
            request_id="r2",
            locale="en-US",
            split="train",
            utterance="start smooth jazz",
            annotated_utterance="start smooth jazz",
            template="start smooth jazz",
            gold_frame=Frame(intent="music_play"),
        ),
        DataRecord(
            request_id="r3",
            locale="en-US",
            split="train",
            utterance="please play smooth jazz",
            annotated_utterance="please play smooth jazz",
            template="please play smooth jazz",
            gold_frame=Frame(intent="music_play"),
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
                        "utterance": "play smooth jazz",
                        "teacher_frame": {"intent": "music_play", "slots": {}},
                    }
                ),
                json.dumps(
                    {
                        "utterance": "start smooth jazz",
                        "teacher_frame": {"intent": "music_play", "slots": {}},
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings()
    settings.l1_agent_mode = "dry-run"
    settings.l1_agent_dry_run_patch = patch_path
    settings.l1_agent_timeout_s = 120.0

    summary = run_replay(
        stream="sequential",
        max_requests=3,
        teacher_mode="cache",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=settings,
        compile_every=2,
    )

    assert summary.layer_counts["L4"] == 2
    assert summary.layer_counts["L1"] == 1
    manifest = ArtifactStore(run_dir / "artifacts").load_current_manifest()
    assert manifest is not None
    assert "l1_crate_dir" in manifest.artifact_paths
    assert "l1_agent_diff" in manifest.artifact_paths
    assert "l1_agent_transcript" in manifest.artifact_paths
    assert "l1_agent_provenance" in manifest.artifact_paths
    assert "l1_benchmark" in manifest.artifact_paths
    assert "hard_buffer" in manifest.artifact_paths
    assert "candidate_metrics_csv" in manifest.artifact_paths
    assert "promotion_record" in manifest.artifact_paths
    assert manifest.candidate_metrics["l1_agent_succeeded"] is True
    assert manifest.candidate_metrics["l1_benchmark_status"] == "success"
    assert manifest.candidate_metrics["hard_buffer_size"] == 2
    assert manifest.candidate_metrics["hard_buffer_visibility_counts"] == {
        "replay_only": 1,
        "train_visible": 1,
    }
    assert manifest.candidate_metrics["hard_buffer_agent_context_size"] == 1
    assert manifest.candidate_metrics["candidate_layer_counts"]["L1"] == 1
    l1_agent_dir = run_dir / "artifacts" / manifest.artifact_paths["l1_agent_dir"]
    context_families = json.loads(
        (l1_agent_dir / "contexts" / "context_families.json").read_text(encoding="utf-8")
    )
    assert context_families["schema_version"] == "l1-context-families-v1"
    assert context_families["family_count"] >= 1
    assert context_families["families"][0]["intent"] == "music_play"
    assert "gold_frame" not in json.dumps(context_families)
    l1_agent_hard_cases = (l1_agent_dir / "contexts" / "hard_cases.jsonl").read_text(
        encoding="utf-8"
    )
    assert "play smooth jazz" in l1_agent_hard_cases
    assert "start smooth jazz" not in l1_agent_hard_cases

    traces = [
        json.loads(line)
        for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert traces[0]["chosen_layer"] == "L4"
    assert traces[1]["chosen_layer"] == "L4"
    assert traces[2]["chosen_layer"] == "L1"
    assert traces[2]["final_frame"]["intent"] == "music_play"


def _music_l1_patch() -> str:
    return "\n".join(
        [
            "diff --git a/src/lib.rs b/src/lib.rs",
            "--- a/src/lib.rs",
            "+++ b/src/lib.rs",
            "@@ -49,10 +49,21 @@ fn collect_candidates(q: &str) -> Vec<Candidate> {",
            " fn collect_candidates(q: &str) -> Vec<Candidate> {",
            "     let mut candidates = Vec::new();",
            "     if let Some(candidate) = try_alarm_set(q) {",
            "         candidates.push(candidate);",
            "     }",
            "     if let Some(candidate) = try_weather_query(q) {",
            "         candidates.push(candidate);",
            "     }",
            '+    if q.contains("smooth jazz") {',
            "+        let slots = std::collections::BTreeMap::new();",
            "+        candidates.push(Candidate {",
            "+            frame: crate::frame::Frame {",
            '+                intent: "music_play".to_string(),',
            "+                slots,",
            "+                is_abstain: false,",
            "+            },",
            '+            program_path: "programs/music::dry_run",',
            "+        });",
            "+    }",
            "     candidates",
            " }",
            "",
        ]
    )


def _trace_with_lower_result(
    request_id: str,
    utterance: str,
    intent: str,
    *,
    lower_layer: str | None,
) -> TraceRecord:
    frame = Frame(intent=intent)
    layer_results = []
    if lower_layer is not None:
        layer_results.append(
            LayerResult(
                layer=lower_layer,
                accepted=True,
                frame=frame,
                latency_ms=1.0,
            )
        )
    else:
        layer_results.extend(
            [
                LayerResult(
                    layer="L0",
                    accepted=False,
                    frame=None,
                    latency_ms=1.0,
                ),
                LayerResult(
                    layer="L1",
                    accepted=False,
                    frame=None,
                    latency_ms=1.0,
                ),
                LayerResult(
                    layer="L4",
                    accepted=True,
                    frame=frame,
                    latency_ms=1.0,
                ),
            ]
        )
    return TraceRecord(
        request_id=request_id,
        utterance=utterance,
        gold_frame=frame,
        teacher_frame=frame,
        chosen_layer=lower_layer or "L4",
        final_frame=frame,
        layer_results=layer_results,
    )


def _teacher_trace(request_id: str, utterance: str, intent: str) -> TraceRecord:
    frame = Frame(intent=intent)
    return TraceRecord(
        request_id=request_id,
        utterance=utterance,
        gold_frame=frame,
        teacher_frame=frame,
        chosen_layer="L4",
        final_frame=frame,
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=frame,
                latency_ms=1.0,
            )
        ],
    )


def _two_intent_traces() -> list[TraceRecord]:
    return [
        _teacher_trace("m1", "play jazz", "music_play"),
        _teacher_trace("m2", "play music", "music_play"),
        _teacher_trace("m3", "start playlist", "music_play"),
        _teacher_trace("m4", "play songs", "music_play"),
        _teacher_trace("a1", "set alarm for seven", "alarm_set"),
        _teacher_trace("a2", "wake me at eight", "alarm_set"),
        _teacher_trace("a3", "alarm at nine", "alarm_set"),
        _teacher_trace("a4", "set morning alarm", "alarm_set"),
    ]
