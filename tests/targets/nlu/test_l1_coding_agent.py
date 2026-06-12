import json
from pathlib import Path

import pytest

from darjeeling.schemas import LayerResult, TraceRecord, traces_to_teacher_view
from darjeeling.settings import load_settings
from darjeeling.targets.nlu.compiler.l1_program_compiler import (
    L1CodingAgentError,
    L1CodingAgentJobConfig,
    L4CodingAgentAdapter,
    run_l1_coding_agent_job,
)
from darjeeling.targets.nlu.schemas import Frame


def _teacher_trace():
    trace = TraceRecord(
        request_id="r1",
        utterance="alpha request value alpha",
        gold_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "gold-value-alpha"}),
        teacher_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
                latency_ms=1.0,
            )
        ],
    )
    return traces_to_teacher_view([trace])[0]


def test_legacy_l1_program_compiler_module_reexports_nlu_target_adapter() -> None:
    from darjeeling.compiler.l1_program_compiler import (
        L4CodingAgentAdapter as LegacyL4CodingAgentAdapter,
    )

    assert LegacyL4CodingAgentAdapter is L4CodingAgentAdapter


def test_l1_coding_agent_dry_run_packages_workspace_and_context(
    tmp_path: Path,
) -> None:
    patch_path = tmp_path / "dry_run.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/DRY_RUN_MARKER b/DRY_RUN_MARKER",
                "new file mode 100644",
                "--- /dev/null",
                "+++ b/DRY_RUN_MARKER",
                "@@ -0,0 +1 @@",
                "+dry run marker",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = run_l1_coding_agent_job(
        config=L1CodingAgentJobConfig(
            mode="dry-run",
            source_crate_dir=Path("native/l1_empty_programbank"),
            job_dir=tmp_path / "job",
            dry_run_patch=patch_path,
            run_validation=False,
        ),
        teacher_train=[_teacher_trace()],
        hard_cases=[],
        current_metrics={"coverage": 0.1},
        objective={"wrong_accept_limit": 0.05},
    )

    assert result.succeeded
    assert (result.workspace_crate_dir / "Cargo.toml").exists()
    assert (result.workspace_crate_dir / "DRY_RUN_MARKER").read_text(
        encoding="utf-8"
    ) == "dry run marker\n"
    assert result.prompt_path.exists()
    assert result.transcript_path.exists()
    assert result.report_path.exists()
    assert result.provenance_path.exists()
    assert "DRY_RUN_MARKER" in result.diff_path.read_text(encoding="utf-8")
    context_families = json.loads(
        (result.context_dir / "context_families.json").read_text(encoding="utf-8")
    )
    assert context_families["schema_version"] == "l1-context-families-v1"
    assert context_families["families"][0]["family_id"] == "intent_alpha|slot_alpha"
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "l1-agent-provenance-v1"
    assert provenance["mode"] == "dry-run"
    assert provenance["diff"]["changed_file_count"] == 1

    context_text = "\n".join(
        path.read_text(encoding="utf-8") for path in result.context_dir.iterdir()
    )
    assert "gold_frame" not in context_text
    assert "gold-value-alpha" not in context_text
    assert "value alpha" in context_text

    commands = [
        json.loads(line)
        for line in result.commands_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert commands[0]["command"] == ["git", "apply", str(patch_path)]
    assert commands[0]["return_code"] == 0


def test_l1_coding_agent_adapter_respects_disabled_mode(tmp_path: Path) -> None:
    settings = load_settings()
    settings.l1_agent_mode = "disabled"
    adapter = L4CodingAgentAdapter(settings)

    with pytest.raises(L1CodingAgentError):
        adapter.run_l1_job(
            job_dir=tmp_path / "job",
            source_crate_dir=Path("native/l1_empty_programbank"),
            teacher_train=[_teacher_trace()],
            run_validation=False,
        )


def test_l1_coding_agent_codex_cli_mode_records_transcript_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
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
                "out.write_text('fake agent report\\n')",
                "print(json.dumps({'event': 'done', 'prompt_seen': 'L1 Rust' in prompt}))",
            ]
        ),
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    result = run_l1_coding_agent_job(
        config=L1CodingAgentJobConfig(
            mode="codex-cli",
            source_crate_dir=repo_root / "native/l1_empty_programbank",
            job_dir=Path("job"),
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
    assert "fake agent report" in result.report_path.read_text(encoding="utf-8")
    transcript = result.transcript_path.read_text(encoding="utf-8")
    assert '"event": "done"' in transcript
    assert '"prompt_seen": true' in transcript
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["transcript"]["event_types"] == {"done": 1}
    assert provenance["commands"][0]["return_code"] == 0
    commands = [
        json.loads(line)
        for line in result.commands_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    command = commands[0]["command"]
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


def test_l1_agent_session_uses_workspace_root_and_records_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
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
                "(workspace / 'l1_programbank' / 'AGENT_SESSION_MARKER').write_text(",
                "    'agent session marker\\n', encoding='utf-8'",
                ")",
                "(workspace / 'runs' / 'agent_note.txt').write_text(",
                "    'agent session completed\\n', encoding='utf-8'",
                ")",
                "out.write_text('fake L1 agent-session report\\n', encoding='utf-8')",
                "print(json.dumps({'event': 'done', 'autonomous': 'autonomous L1' in prompt}))",
            ]
        ),
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    result = run_l1_coding_agent_job(
        config=L1CodingAgentJobConfig(
            mode="agent-session",
            source_crate_dir=repo_root / "native/l1_empty_programbank",
            job_dir=Path("job"),
            codex_command=str(fake_codex),
            codex_model=None,
            run_validation=False,
        ),
        teacher_train=[_teacher_trace()],
        hard_cases=[],
        current_metrics={},
        objective={},
    )

    workspace_root = result.job_dir / "workspace"
    commands = [
        json.loads(line)
        for line in result.commands_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    command = commands[0]["command"]
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    manifest = json.loads((workspace_root / "workspace_manifest.json").read_text(encoding="utf-8"))

    assert result.succeeded
    assert Path(command[command.index("--cd") + 1]) == workspace_root.resolve()
    assert (result.workspace_crate_dir / "AGENT_SESSION_MARKER").exists()
    assert (workspace_root / "runs" / "agent_note.txt").exists()
    assert provenance["agent_session"]["applies_to_mode"] is True
    assert provenance["agent_session"]["internal_loop_control"] == (
        "agent_decides_edit_compile_test_bench_replay_stop"
    )
    assert provenance["workspace_scope_policy"]["candidate_code_writable_roots"] == [
        "l1_programbank/"
    ]
    assert manifest["agent_session_policy"]["applies_to_mode"] is True
    assert "AGENT_SESSION_MARKER" in result.diff_path.read_text(encoding="utf-8")


def test_l1_agent_session_rejects_protected_workspace_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import pathlib",
                "import sys",
                "workspace = pathlib.Path(sys.argv[sys.argv.index('--cd') + 1])",
                "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])",
                "(workspace / 'program.md').write_text('tampered\\n', encoding='utf-8')",
                "out.write_text('fake report\\n', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    result = run_l1_coding_agent_job(
        config=L1CodingAgentJobConfig(
            mode="agent-session",
            source_crate_dir=repo_root / "native/l1_empty_programbank",
            job_dir=Path("job"),
            codex_command=str(fake_codex),
            codex_model=None,
            run_validation=False,
        ),
        teacher_train=[_teacher_trace()],
        hard_cases=[],
        current_metrics={},
        objective={},
    )

    commands = [
        json.loads(line)
        for line in result.commands_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))

    assert not result.succeeded
    assert result.return_code == 1
    assert commands[-1]["command"] == ["l1-workspace-scope-check"]
    assert commands[-1]["return_code"] == 1
    assert "program.md" in commands[-1]["stderr"]
    assert provenance["succeeded"] is False
