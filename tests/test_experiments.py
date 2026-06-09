import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darjeeling.artifacts.store import ArtifactStore
from darjeeling.cli import (
    _execute_experiment_run,
    _experiment_preflight_payload,
    _promote_l3_prompt_artifact,
    _run_settings_payload,
)
from darjeeling.compiler.l3_prompt_optimizer import l3_prompt_artifact_hash
from darjeeling.eval.experiments import apply_experiment_settings, experiment_spec
from darjeeling.layers.l3_local_slm import L3PromptArtifact
from darjeeling.settings import load_settings


def test_experiment_settings_apply_l2_ablations() -> None:
    settings = load_settings()

    no_guard = apply_experiment_settings(settings, experiment_spec("no-guard"))
    no_l2 = apply_experiment_settings(settings, experiment_spec("no-l2"))
    l2_mlp = apply_experiment_settings(settings, experiment_spec("l2-mlp"))
    l2_tuned = apply_experiment_settings(settings, experiment_spec("l2-tuned"))
    l2_tuned_lower_miss = apply_experiment_settings(
        settings,
        experiment_spec("l2-tuned-lower-miss"),
    )

    assert no_guard.l2_guard_mode == "always_accept"
    assert no_guard.l2_max_wrong_accept_rate == 1.0
    assert no_guard.promotion_accuracy_epsilon == 1.0
    assert no_guard.force_promote_artifacts is True
    assert no_l2.l2_enabled is False
    assert l2_mlp.l2_intent_model_family == "mlp"
    assert l2_mlp.l2_mlp_hidden_layer_sizes == (64,)
    assert l2_mlp.l2_max_iter == 300
    assert l2_tuned.l2_tuning_mode == "optuna"
    assert l2_tuned.l2_tuning_trials == 12
    assert l2_tuned.l2_tuning_min_examples == 200
    assert l2_tuned.l2_tuning_search_space == "compact"
    assert l2_tuned.l2_training_scope == "teacher_train"
    assert l2_tuned_lower_miss.l2_training_scope == "lower_miss"
    assert l2_tuned_lower_miss.l2_tuning_mode == "optuna"
    assert settings.l2_guard_mode == "learned"
    assert settings.l2_intent_model_family == "sgd_logreg"
    assert settings.l2_max_wrong_accept_rate < 1.0
    assert settings.force_promote_artifacts is False
    assert settings.l2_enabled is True


def test_run_settings_payload_records_non_secret_settings_only(tmp_path: Path) -> None:
    settings = load_settings().model_copy(
        update={
            "openai_api_key": "secret",
            "openai_model": "model-from-test",
            "l4_input_usd_per_million": 2.0,
        }
    )

    payload = _run_settings_payload(
        stream="zipf-heavy",
        max_requests=3,
        compile_every=2,
        teacher="cache",
        data_dir=tmp_path / "data",
        settings=settings,
    )

    assert "openai_api_key" not in payload
    assert payload["openai_api_key_present"] is True
    assert payload["openai_model"] == "model-from-test"
    assert payload["l4_input_usd_per_million"] == 2.0


def test_execute_experiment_run_writes_metadata_and_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {}
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "manifest.current.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "summary.md").write_text("old report\n", encoding="utf-8")
    (tmp_path / "traces.jsonl").write_text("old trace\n", encoding="utf-8")
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "teacher_cache.jsonl").write_text("cached teacher\n", encoding="utf-8")

    def fake_execute_replay_run(**kwargs):
        calls["settings"] = kwargs["settings"]
        calls["stream"] = kwargs["stream"]
        calls["old_artifacts_present"] = (tmp_path / "artifacts").exists()
        calls["teacher_cache_present"] = (tmp_path / "teacher_cache.jsonl").exists()
        return SimpleNamespace(
            requests=3,
            traces_path=tmp_path / "traces.jsonl",
            layer_counts={"L4": 3},
        )

    def fake_generate_run_report(run_dir: Path):
        calls["report_run_dir"] = run_dir
        return SimpleNamespace(report_dir=run_dir / "reports")

    monkeypatch.setattr("darjeeling.cli._execute_replay_run", fake_execute_replay_run)
    monkeypatch.setattr("darjeeling.cli.generate_run_report", fake_generate_run_report)

    _execute_experiment_run(
        experiment_spec("no-l2"),
        run_dir=tmp_path,
        stream="zipf-heavy",
        max_requests=3,
        compile_every=2,
        teacher="cache",
        data_dir=tmp_path / "data",
    )

    metadata = json.loads((tmp_path / "experiment.json").read_text(encoding="utf-8"))
    assert metadata["experiment"] == "no-l2"
    assert metadata["settings_overrides"] == {"l2_enabled": False}
    assert calls["settings"].l2_enabled is False
    assert calls["stream"] == "zipf-heavy"
    assert calls["old_artifacts_present"] is False
    assert calls["teacher_cache_present"] is True
    assert calls["report_run_dir"] == tmp_path


def test_workload_locality_spec_runs_all_streams() -> None:
    spec = experiment_spec("workload-locality")

    assert spec.substreams == ("uniform", "zipf-mild", "zipf-heavy")


def test_experiment_preflight_passes_with_data_cache_and_l1_crate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"request_id":"r1"}\n', encoding="utf-8")
    (run_dir / "teacher_cache.jsonl").write_text(
        json.dumps(
            {
                "utterance": "play music",
                "teacher_frame": {"intent": "music_play", "slots": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = _experiment_preflight_payload(
        run_dir=run_dir,
        data_dir=data_dir,
        teacher="cache",
        settings=load_settings(),
    )

    assert payload["schema_version"] == "experiment-preflight-v1"
    assert payload["status"] == "pass"
    assert {check["name"]: check["status"] for check in payload["checks"]} == {
        "data.train_split": "pass",
        "teacher": "pass",
        "l1.rust_crate": "pass",
        "l1.agent": "warn",
        "l3.local_slm": "pass",
    }


def test_experiment_preflight_fails_when_required_inputs_are_missing(tmp_path: Path) -> None:
    payload = _experiment_preflight_payload(
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "missing-data",
        teacher="cache",
        settings=load_settings(),
    )

    assert payload["status"] == "fail"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["data.train_split"]["status"] == "fail"
    assert checks["teacher"]["status"] == "fail"


def test_l3_prompt_promotion_requires_successful_replay_artifact(tmp_path: Path) -> None:
    prompt_path = tmp_path / "candidate_prompt.json"
    replay_path = tmp_path / "candidate_replay.json"
    prompt_artifact = L3PromptArtifact(
        prompt_version="candidate-v1",
        system_prompt="Return JSON only.",
        confidence_threshold=0.8,
    )
    prompt_path.write_text(
        prompt_artifact.model_dump_json(),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps(
            {
                "schema_version": "l3-prompt-replay-v1",
                "status": "success",
                "prompt_version": "candidate-v1",
                "prompt_sha256": l3_prompt_artifact_hash(prompt_artifact),
                "requests": 2,
                "would_accept_count": 2,
                "accepted_accuracy": 1.0,
                "wrong_accept_rate": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = _promote_l3_prompt_artifact(
        run_dir=tmp_path / "run",
        prompt_path=prompt_path,
        replay_path=replay_path,
        min_accepted_accuracy=0.9,
        max_wrong_accept_rate=0.05,
    )

    assert manifest.promoted is True
    assert "l3_prompt" in manifest.artifact_paths
    assert "l3_prompt_replay" in manifest.artifact_paths
    assert manifest.candidate_metrics["l3_prompt_runtime_promoted"] is True
    current = ArtifactStore(tmp_path / "run" / "artifacts").load_current_manifest()
    assert current is not None
    assert current.artifact_paths["l3_prompt"] == manifest.artifact_paths["l3_prompt"]


def test_l3_prompt_promotion_rejects_replay_for_different_prompt(tmp_path: Path) -> None:
    prompt_artifact = L3PromptArtifact(
        prompt_version="candidate-v1",
        system_prompt="Return JSON only.",
        confidence_threshold=0.8,
    )
    prompt_path = tmp_path / "candidate_prompt.json"
    replay_path = tmp_path / "candidate_replay.json"
    prompt_path.write_text(prompt_artifact.model_dump_json(), encoding="utf-8")
    replay_path.write_text(
        json.dumps(
            {
                "schema_version": "l3-prompt-replay-v1",
                "status": "success",
                "prompt_version": "candidate-v1",
                "prompt_sha256": "wrong",
                "requests": 2,
                "would_accept_count": 2,
                "accepted_accuracy": 1.0,
                "wrong_accept_rate": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash does not match"):
        _promote_l3_prompt_artifact(
            run_dir=tmp_path / "run",
            prompt_path=prompt_path,
            replay_path=replay_path,
            min_accepted_accuracy=0.9,
            max_wrong_accept_rate=0.05,
        )
