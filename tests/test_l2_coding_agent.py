import json
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
        utterance="set alarm for seven",
        gold_frame=Frame(intent="alarm_set", slots={"time": "gold-seven"}),
        teacher_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        chosen_layer="L4",
        final_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
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
                frame=Frame(intent="alarm_set", slots={"time": "seven"}),
                latency_ms=1.0,
            ),
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
                "diff --git a/tests/test_l2_agent_marker.py b/tests/test_l2_agent_marker.py",
                "new file mode 100644",
                "--- /dev/null",
                "+++ b/tests/test_l2_agent_marker.py",
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
        hard_cases=[],
        current_metrics={"l2_runtime_coverage": 0.001},
        objective={"wrong_accept_limit": 0.05},
    )

    assert result.succeeded
    assert (result.workspace_repo_dir / "src/darjeeling/layers/l2_student.py").exists()
    assert (result.workspace_repo_dir / "tests/test_l2_agent_marker.py").read_text(
        encoding="utf-8"
    ) == "MARKER = 'dry run marker'\n"
    assert result.prompt_path.exists()
    assert result.transcript_path.exists()
    assert result.report_path.exists()
    assert result.provenance_path.exists()
    assert "test_l2_agent_marker.py" in result.diff_path.read_text(encoding="utf-8")
    context_families = json.loads(
        (result.context_dir / "l2_context_families.json").read_text(encoding="utf-8")
    )
    assert context_families["schema_version"] == "l2-context-families-v1"
    assert context_families["families"][0]["family_id"] == "alarm_set|time"
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "l2-agent-provenance-v1"
    assert provenance["mode"] == "dry-run"
    assert provenance["runtime_patch_applied"] is False
    assert provenance["diff"]["changed_file_count"] == 1

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
                "print(json.dumps({'event': 'done', 'prompt_seen': 'L2 Python' in prompt}))",
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
    assert Path(command[command.index("--cd") + 1]).is_absolute()
    assert Path(command[command.index("-o") + 1]).is_absolute()
