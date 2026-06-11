import json
import subprocess
import sys
from pathlib import Path

import pytest

from darjeeling.compiler.l2_coding_agent import (
    L2CodingAgentAdapter,
    L2CodingAgentError,
    L2CodingAgentJobConfig,
    run_l2_coding_agent_job,
)
from darjeeling.schemas import Frame, LayerResult, TraceRecord, traces_to_teacher_view
from darjeeling.settings import load_settings


def _teacher_trace():
    trace = TraceRecord(
        request_id="r1",
        utterance="alpha request for seven",
        gold_frame=Frame(intent="intent_alpha", slots={"time": "gold-seven"}),
        teacher_frame=Frame(intent="intent_alpha", slots={"time": "seven"}),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_alpha", slots={"time": "seven"}),
        layer_results=[
            LayerResult(
                layer="L2",
                accepted=False,
                frame=None,
                latency_ms=4.0,
            ),
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="intent_alpha", slots={"time": "seven"}),
                latency_ms=1.0,
            ),
        ],
    )
    return traces_to_teacher_view([trace])[0]


def _slot_wrong_hard_case():
    trace = TraceRecord(
        request_id="r-slot",
        utterance="alpha request with red value",
        gold_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "red"}),
        teacher_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "red"}),
        chosen_layer="L2",
        final_frame=Frame(intent="intent_alpha", slots={}),
        layer_results=[
            LayerResult(
                layer="L2",
                accepted=True,
                frame=Frame(intent="intent_alpha", slots={}),
                confidence=0.95,
                latency_ms=2.0,
                metadata={
                    "guard_probability": 0.95,
                    "frame_source": "student",
                    "predicted_slot_count": 0.0,
                    "predicted_signature_frame_accuracy": 0.72,
                },
            )
        ],
    )
    return traces_to_teacher_view([trace])[0]


def test_l2_coding_agent_dry_run_packages_workspace_and_context(
    tmp_path: Path,
) -> None:
    patch_path = tmp_path / "dry_run.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/candidate/tests/test_l2_agent_marker.py "
                "b/candidate/tests/test_l2_agent_marker.py",
                "new file mode 100644",
                "--- /dev/null",
                "+++ b/candidate/tests/test_l2_agent_marker.py",
                "@@ -0,0 +1 @@",
                "+MARKER = 'dry run marker'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = run_l2_coding_agent_job(
        config=L2CodingAgentJobConfig(
            mode="dry-run",
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            dry_run_patch=patch_path,
            run_validation=False,
        ),
        teacher_train=[_teacher_trace()],
        hard_cases=[_slot_wrong_hard_case()],
        current_metrics={"l2_runtime_coverage": 0.001},
        objective={"wrong_accept_limit": 0.05},
    )

    assert result.succeeded
    candidate_dir = result.workspace_repo_dir / "candidate"
    system_dir = result.workspace_repo_dir / "system" / "darjeeling"
    data_dir = result.workspace_repo_dir / "data"
    assert (result.workspace_repo_dir / "program.md").exists()
    assert (result.workspace_repo_dir / "workspace_manifest.json").exists()
    assert (result.workspace_repo_dir / "tools" / "run_checks.py").exists()
    assert (candidate_dir / "src/darjeeling/layers/l2_student.py").exists()
    assert (system_dir / "src/darjeeling/layers/l2_student.py").exists()
    assert (candidate_dir / "tests/test_l2_agent_marker.py").read_text(
        encoding="utf-8"
    ) == "MARKER = 'dry run marker'\n"
    assert result.prompt_path.exists()
    assert result.transcript_path.exists()
    assert result.report_path.exists()
    assert result.provenance_path.exists()
    assert "test_l2_agent_marker.py" in result.diff_path.read_text(encoding="utf-8")
    prompt_text = result.prompt_path.read_text(encoding="utf-8")
    assert prompt_text == (
        "Read `program.md` in this workspace and complete one bounded L2 research iteration."
    )
    assert "agent_contexts" not in prompt_text
    assert "l2_context_families" not in prompt_text
    assert "r-slot" not in prompt_text
    program_text = (result.workspace_repo_dir / "program.md").read_text(encoding="utf-8")
    assert "`candidate/` is the only editable research code area" in program_text
    assert "tools/run_checks.py" in program_text
    context_families = json.loads(
        (result.context_dir / "l2_context_families.json").read_text(encoding="utf-8")
    )
    workspace_context_families = json.loads((data_dir / "l2_context_families.json").read_text())
    assert context_families["schema_version"] == "l2-context-families-v1"
    assert workspace_context_families == context_families
    assert context_families["families"][0]["family_id"] == "intent_alpha|time"
    slot_error_summary = json.loads(
        (data_dir / "slot_error_summary.json").read_text(encoding="utf-8")
    )
    assert slot_error_summary["schema_version"] == "l2-slot-error-summary-v1"
    assert slot_error_summary["l2_wrong_accept_count"] == 1
    assert slot_error_summary["l2_intent_correct_slot_mismatch_count"] == 1
    assert slot_error_summary["missing_slot_counts"] == {"slot_alpha": 1}
    assert slot_error_summary["examples"][0]["l2_metadata"]["guard_probability"] == 0.95
    manifest = json.loads(
        (result.workspace_repo_dir / "workspace_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "l2-research-workspace-v1"
    assert manifest["candidate_dir"] == "candidate"
    assert manifest["system_repo_dir"] == "system/darjeeling"
    assert manifest["data_dir"] == "data"
    assert "src/darjeeling/layers/l2_student.py" in manifest["candidate_paths"]
    assert "slot_error_summary.json" in manifest["data_files"]
    assert manifest["commands"] == {
        "inspect_context": "python3 tools/inspect_context.py",
        "run_checks": "python3 tools/run_checks.py",
    }
    assert "uv run --project" not in program_text
    commands_text = (data_dir / "commands.md").read_text(encoding="utf-8")
    assert "`python3 tools/inspect_context.py`" in commands_text
    assert "`python3 tools/run_checks.py`" in commands_text
    assert "uv run --project" not in commands_text
    inspect_result = subprocess.run(
        [sys.executable, "tools/inspect_context.py"],
        cwd=result.workspace_repo_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    assert inspect_result.returncode == 0, inspect_result.stderr
    assert "objective:" in inspect_result.stdout
    assert "slot error summary:" in inspect_result.stdout
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "l2-agent-provenance-v1"
    assert provenance["mode"] == "dry-run"
    assert provenance["runtime_patch_applied"] is False
    assert "tests/test_l2_agent_marker.py" in provenance["diff"]["changed_files"]

    context_text = "\n".join(
        path.read_text(encoding="utf-8") for path in result.context_dir.iterdir()
    )
    assert "gold_frame" not in context_text
    assert "gold-seven" not in context_text
    assert "seven" in context_text


def test_l2_coding_agent_adapter_respects_disabled_mode(tmp_path: Path) -> None:
    settings = load_settings()
    settings.l2_agent_mode = "disabled"
    adapter = L2CodingAgentAdapter(settings)

    with pytest.raises(L2CodingAgentError):
        adapter.run_l2_job(
            job_dir=tmp_path / "job",
            source_repo_dir=Path.cwd(),
            teacher_train=[_teacher_trace()],
            run_validation=False,
        )


def test_l2_coding_agent_codex_cli_mode_records_transcript_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import pathlib",
                "import sys",
                "prompt = sys.stdin.read()",
                "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])",
                "out.write_text('fake L2 agent report\\n')",
                "print(json.dumps({'event': 'done', 'prompt_seen': 'program.md' in prompt}))",
            ]
        ),
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    result = run_l2_coding_agent_job(
        config=L2CodingAgentJobConfig(
            mode="codex-cli",
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            codex_command=str(fake_codex),
            codex_model="test-model",
            run_validation=False,
        ),
        teacher_train=[_teacher_trace()],
        hard_cases=[],
        current_metrics={},
        objective={},
    )

    assert result.succeeded
    assert "fake L2 agent report" in result.report_path.read_text(encoding="utf-8")
    transcript = result.transcript_path.read_text(encoding="utf-8")
    assert '"event": "done"' in transcript
    assert '"prompt_seen": true' in transcript
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["transcript"]["event_types"] == {"done": 1}
    assert provenance["commands"][0]["return_code"] == 0
    assert provenance["ignore_user_config"] is True
    assert provenance["ignore_rules"] is True
    assert provenance["ephemeral"] is True
    command = provenance["commands"][0]["command"]
    assert command[:8] == [
        str(fake_codex),
        "--model",
        "test-model",
        "--sandbox",
        "workspace-write",
        "-a",
        "never",
        "exec",
    ]
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "--ephemeral" in command
    assert "--skip-git-repo-check" in command
    assert Path(command[command.index("--cd") + 1]).is_absolute()
    assert Path(command[command.index("-o") + 1]).is_absolute()


def test_l2_coding_agent_codex_timeout_writes_serializable_artifacts(
    tmp_path: Path,
) -> None:
    slow_codex = tmp_path / "slow_codex.py"
    slow_codex.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "sys.stdout.write('partial output')",
                "sys.stdout.flush()",
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    slow_codex.chmod(0o755)

    result = run_l2_coding_agent_job(
        config=L2CodingAgentJobConfig(
            mode="codex-cli",
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            codex_command=str(slow_codex),
            timeout_s=0.1,
            run_validation=False,
        ),
        teacher_train=[_teacher_trace()],
        hard_cases=[],
        current_metrics={},
        objective={},
    )

    assert not result.succeeded
    commands = [
        json.loads(line)
        for line in result.commands_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert commands[0]["return_code"] == 124
    assert isinstance(commands[0]["stdout"], str)
    assert isinstance(commands[0]["stderr"], str)
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["return_code"] == 124
    assert provenance["commands"][0]["return_code"] == 124
