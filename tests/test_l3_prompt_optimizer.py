import json
import subprocess
import sys
from pathlib import Path

import pytest

from darjeeling.compiler.l3_prompt_optimizer import (
    L3PromptEvolutionConfig,
    calibrate_l3_confidence_threshold,
    l3_prompt_artifact_from_proposal,
    l3_prompt_artifact_hash,
    replay_l3_prompt_artifact,
    run_l3_prompt_evolution,
)
from darjeeling.layers.l3_local_slm import L3PromptArtifact, LocalSLMConfig
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.schemas import Frame, LayerResult, TraceRecord, traces_to_teacher_view


class FakeL3Backend:
    def generate(self, prompt: str, config: LocalSLMConfig) -> str:
        assert "strict JSON" in prompt
        return '{"intent": "music_play", "slots": {}, "confidence": 0.93}'

    def status(self) -> dict:
        return {"model_name": "fake", "actual_device": "fake-device", "loaded": True}


def test_l3_prompt_artifact_from_proposal_expands_teacher_visible_examples() -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="set alarm for seven",
        gold_frame=Frame(intent="alarm_set", slots={"time": "gold-seven"}),
        teacher_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        chosen_layer="L4",
        final_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="alarm_set", slots={"time": "seven"}),
                latency_ms=1.0,
            )
        ],
    )

    artifact = l3_prompt_artifact_from_proposal(
        {
            "system_prompt": "Return strict JSON only.",
            "confidence_threshold": 0.81,
            "few_shot_trace_ids": ["r1"],
        },
        traces=traces_to_teacher_view([trace]),
        prompt_version="candidate-v1",
    )

    assert artifact.prompt_version == "candidate-v1"
    assert artifact.confidence_threshold == 0.81
    assert artifact.few_shot_examples == [
        {
            "trace_id": "r1",
            "utterance": "set alarm for seven",
            "frame": {
                "intent": "alarm_set",
                "slots": {"time": "seven"},
                "is_abstain": False,
            },
        }
    ]
    assert "gold-seven" not in artifact.model_dump_json()


def test_l3_prompt_artifact_rejects_unknown_few_shot_trace_id() -> None:
    with pytest.raises(ValueError, match="not teacher-visible"):
        l3_prompt_artifact_from_proposal(
            {
                "system_prompt": "Return JSON.",
                "few_shot_trace_ids": ["missing"],
            },
            traces=[],
            prompt_version="candidate-v1",
        )


def test_l3_guard_calibration_selects_safe_confidence_threshold() -> None:
    traces = [
        _l3_shadow_trace("r1", confidence=0.9, predicted=Frame(intent="music_play")),
        _l3_shadow_trace("r2", confidence=0.8, predicted=Frame(intent="alarm_set")),
        _l3_shadow_trace("r3", confidence=0.4, predicted=Frame(intent="music_play")),
    ]

    result = calibrate_l3_confidence_threshold(traces, max_wrong_accept_rate=0.0)

    assert result is not None
    assert result.threshold == 0.9
    assert result.accepted_count == 1
    assert result.wrong_accept_count == 0
    assert result.accepted_accuracy == 1.0


def test_l3_prompt_replay_scores_generated_shadow_outputs() -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="play jazz",
        teacher_frame=Frame(intent="music_play"),
        chosen_layer="L4",
        final_frame=Frame(intent="music_play"),
        layer_results=[],
    )

    prompt_artifact = L3PromptArtifact(
        prompt_version="candidate-v1",
        system_prompt="Return strict JSON.",
        confidence_threshold=0.8,
    )
    payload = replay_l3_prompt_artifact(
        prompt_artifact=prompt_artifact,
        traces=[trace],
        task_schema=TaskSchema(intent_names=["music_play"], slot_names=[]),
        config=LocalSLMConfig(mode="shadow", confidence_threshold=0.8),
        backend=FakeL3Backend(),
    )

    assert payload["schema_version"] == "l3-prompt-replay-v1"
    assert payload["status"] == "success"
    assert payload["prompt_version"] == "candidate-v1"
    assert payload["prompt_sha256"] == l3_prompt_artifact_hash(prompt_artifact)
    assert payload["requests"] == 1
    assert payload["would_accept_count"] == 1
    assert payload["correct_accept_count"] == 1
    assert payload["accepted_accuracy"] == 1.0
    assert payload["wrong_accept_rate"] == 0.0
    assert payload["request_results"][0]["predicted_frame"]["intent"] == "music_play"


def test_l3_prompt_evolution_agent_session_replays_private_gates(
    tmp_path: Path,
) -> None:
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import pathlib",
                "import sys",
                "workspace = pathlib.Path(sys.argv[sys.argv.index('--cd') + 1])",
                "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])",
                "prompt = sys.stdin.read()",
                "payload = {",
                "  'prompt_version': 'candidate-v1',",
                "  'system_prompt': 'Return strict JSON for music requests.',",
                "  'confidence_threshold': 0.8,",
                "  'few_shot_examples': []",
                "}",
                "(workspace / 'prompt' / 'l3_prompt.json').write_text(",
                "  json.dumps(payload) + '\\n', encoding='utf-8'",
                ")",
                "(workspace / 'runs' / 'note.txt').write_text('done\\n', encoding='utf-8')",
                "out.write_text('fake L3 report\\n', encoding='utf-8')",
                "print(json.dumps({'event': 'done', 'autonomous': 'autonomous L3' in prompt}))",
            ]
        ),
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    summary = run_l3_prompt_evolution(
        config=L3PromptEvolutionConfig(
            job_dir=tmp_path / "job",
            codex_command=str(fake_codex),
            codex_model=None,
            prompt_version="candidate-v1",
        ),
        traces=_music_traces(10),
        task_schema=TaskSchema(intent_names=["music_play"], slot_names=[]),
        local_slm_config=LocalSLMConfig(mode="shadow", confidence_threshold=0.8),
        backend=FakeL3Backend(),
    )

    workspace = tmp_path / "job" / "workspace" / "l3_prompt"
    transcript = (tmp_path / "job" / "transcripts" / "agent_session.jsonl").read_text(
        encoding="utf-8"
    )
    commands = [
        json.loads(line)
        for line in (tmp_path / "job" / "commands.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["mode"] == "agent-session"
    assert summary["stop_reason"] == "agent_session_completed"
    assert summary["agent_session"]["agent_sessions_started"] == 1
    assert summary["agent_session"]["agent_sessions_succeeded"] == 1
    assert summary["selection_decision"]["selected"] is True
    assert summary["adoption_decision"]["adopted"] is True
    assert summary["candidate"]["visible_validation"]["passes_gate"] is True
    assert summary["candidate"]["selection_holdout"]["passes_gate"] is True
    assert summary["candidate"]["promotion_holdout"]["passes_gate"] is True
    assert (workspace / "prompt" / "l3_prompt.json").exists()
    assert (workspace / "runs" / "note.txt").exists()
    assert (workspace / "contexts" / "local_slm_config.json").exists()
    assert (workspace / "tools" / "evaluate_prompt.py").exists()
    assert (workspace / "tools" / "bench_prompt.py").exists()
    assert (workspace / "tools" / "latency_cost_eval.py").exists()
    assert not (workspace / "contexts" / "selection_holdout.jsonl").exists()
    assert not (workspace / "contexts" / "promotion_holdout.jsonl").exists()
    assert (tmp_path / "job" / "private" / "selection_holdout.jsonl").exists()
    assert (tmp_path / "job" / "private" / "promotion_holdout.jsonl").exists()
    assert '"autonomous": true' in transcript
    assert Path(commands[0]["command"][commands[0]["command"].index("--cd") + 1]) == (
        workspace.resolve()
    )
    manifest = json.loads((workspace / "workspace_manifest.json").read_text(encoding="utf-8"))
    assert "..." not in json.dumps(manifest["commands"])
    assert "evaluate_visible_prompt" in manifest["commands"]
    assert "bench_prompt" in manifest["commands"]
    assert "latency_cost_eval" in manifest["commands"]

    validation = subprocess.run(
        [sys.executable, str(workspace / "tools" / "validate_prompt.py")],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validation.returncode == 0
    latency_cost = subprocess.run(
        [
            sys.executable,
            str(workspace / "tools" / "latency_cost_eval.py"),
            "--out",
            "runs/latency_cost_eval.json",
        ],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    assert latency_cost.returncode == 0
    latency_payload = json.loads(
        (workspace / "runs" / "latency_cost_eval.json").read_text(encoding="utf-8")
    )
    assert latency_payload["estimated_local_eval_cost_usd"] == 0.0
    assert latency_payload["workspace_tool"]["private_data_visible"] is False


def test_l3_prompt_evolution_agent_session_rejects_protected_context_edits(
    tmp_path: Path,
) -> None:
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import pathlib",
                "import sys",
                "workspace = pathlib.Path(sys.argv[sys.argv.index('--cd') + 1])",
                "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])",
                "(workspace / 'contexts' / 'train.jsonl').write_text(",
                "  'tampered\\n', encoding='utf-8'",
                ")",
                "out.write_text('fake report\\n', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    summary = run_l3_prompt_evolution(
        config=L3PromptEvolutionConfig(
            job_dir=tmp_path / "job",
            codex_command=str(fake_codex),
            codex_model=None,
            skip_replay=True,
        ),
        traces=_music_traces(10),
        task_schema=TaskSchema(intent_names=["music_play"], slot_names=[]),
        local_slm_config=LocalSLMConfig(mode="shadow"),
    )
    commands = [
        json.loads(line)
        for line in (tmp_path / "job" / "commands.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["stop_reason"] == "workspace_scope_violation"
    assert summary["agent_session"]["agent_sessions_succeeded"] == 0
    assert summary["selection_decision"]["selected"] is False
    assert commands[-1]["command"] == ["l3-workspace-scope-check"]
    assert "contexts/train.jsonl" in commands[-1]["stderr"]


def _l3_shadow_trace(
    request_id: str,
    *,
    confidence: float,
    predicted: Frame,
) -> TraceRecord:
    return TraceRecord(
        request_id=request_id,
        utterance=f"{request_id} utterance",
        teacher_frame=Frame(intent="music_play"),
        chosen_layer="L4",
        final_frame=Frame(intent="music_play"),
        layer_results=[
            LayerResult(
                layer="L3",
                accepted=False,
                reason="shadow local SLM would accept",
                latency_ms=1.0,
                confidence=confidence,
                metadata={
                    "confidence": confidence,
                    "shadow_frame": predicted.model_dump(mode="json"),
                    "validation_errors": [],
                },
            )
        ],
    )


def _music_traces(count: int) -> list[TraceRecord]:
    return [
        TraceRecord(
            request_id=f"m{index}",
            utterance=f"play music {index}",
            teacher_frame=Frame(intent="music_play"),
            chosen_layer="L4",
            final_frame=Frame(intent="music_play"),
            layer_results=[],
        )
        for index in range(count)
    ]
