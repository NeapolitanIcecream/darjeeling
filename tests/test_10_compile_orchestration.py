from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import PrefixBroker, write_artifact

from darjeeling import compile_orchestration as compile_orchestration_module
from darjeeling.agent_workspace import (
    create_agent_workspace,
    create_compile_run,
    launch_target_adaptation_agent_async,
    load_target_workspace,
    mount_readonly_inputs,
    write_agent_brief,
)
from darjeeling.artifact_worker import build_protocol_docs
from darjeeling.compile_orchestration import (
    _submission_content_digest,
    plan_compile_launch,
    run_interactive_compile_loop,
    start_compile_launch,
)
from darjeeling.errors import CompileLaunchError, WorkspaceError
from darjeeling.model import (
    AgentAttemptOptions,
    AgentViewOptions,
    AgentVisibleDecisionSummary,
    AgentVisibleReport,
    AgentVisibleTelemetrySummary,
    CompileBudget,
    CompileOptions,
    CompileRunStore,
    RecompileReason,
    RecompileRequest,
    ReleaseRegistry,
    RoutingSettings,
    SchedulerPolicy,
    SnapshotOptions,
    TelemetryDataSource,
    WorkspaceStore,
)
from darjeeling.release_runtime import create_release_without_artifacts
from darjeeling.snapshot_reference import build_snapshot, export_train_view_for_agent
from darjeeling.target_definition import export_agent_readonly_target_view, load_checked_target
from darjeeling.util import utcnow


def _write_agent_artifact_snippet(contract_hash: str) -> str:
    return f"""
def write_artifact(candidate, prefixes):
    from pathlib import Path
    p = Path("submissions") / candidate / "artifacts" / "l1"
    p.mkdir(parents=True, exist_ok=True)
    worker = '''
import json
import sys

request = json.loads(sys.stdin.readline())
text = request["input"]["text"]
prefixes = __PREFIXES__
if any(text.startswith(prefix + ":") for prefix in prefixes):
    print(json.dumps({{
        "decision": "accept",
        "output": {{"label": text.split(":", 1)[0]}},
        "confidence": 0.99,
        "reason": "prefix_match",
    }}))
else:
    print(json.dumps({{"decision": "abstain", "confidence": 0.1, "reason": "outside"}}))
'''.replace("__PREFIXES__", repr(prefixes))
    (p / "worker.py").write_text(worker, encoding="utf-8")
    (p / "healthcheck.py").write_text("raise SystemExit(0)\\n", encoding="utf-8")
    (p / "artifact.yaml").write_text('''api_version: v1
layer: L1
start_command:
- python3
- worker.py
healthcheck_command:
- python3
- healthcheck.py
protocol: jsonl
timeout_ms: 1000
memory_mb: 64
network: disabled
contract_hash: {contract_hash}
allowed_reason_codes:
- prefix_match
- outside
''', encoding="utf-8")
    (Path("submissions") / candidate / "READY").write_text("ready\\n", encoding="utf-8")
"""


def _launch_interactive_agent(
    target_dir: Path,
    tmp_path: Path,
    now,
    budget: CompileBudget,
    agent_code: str,
    agent_timeout_seconds: int | None = None,
):
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    snapshot = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=tmp_path / "snapshots"),
    )
    workspace = load_target_workspace(
        definition.name, definition.contract_hash, WorkspaceStore(tmp_path / "workspaces")
    )
    compile_run = create_compile_run(
        definition,
        check,
        snapshot.snapshot,
        release,
        budget,
        workspace,
        snapshot.reference_qualification,
        CompileOptions(),
    )
    attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    target_view = export_agent_readonly_target_view(
        definition, attempt.workspace_path / "readonly_source" / "target"
    )
    train_view = export_train_view_for_agent(
        snapshot.snapshot,
        contract,
        AgentViewOptions(),
        attempt.workspace_path / "readonly_source" / "train",
    )
    mount_manifest = mount_readonly_inputs(
        attempt,
        target_view,
        train_view,
        release,
        [],
        [],
        build_protocol_docs("v1"),
    )
    brief = write_agent_brief(attempt, compile_run, mount_manifest, {})
    handle = launch_target_adaptation_agent_async(
        attempt,
        brief,
        {
            "command": ["/usr/bin/python3", "-c", agent_code],
            "timeout_seconds": agent_timeout_seconds,
        },
    )
    return definition, contract, release, snapshot, compile_run, attempt, handle


def test_submission_content_digest_includes_symlink_target(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    layer = submission / "artifacts" / "l1"
    layer.mkdir(parents=True)
    (layer / "artifact.yaml").write_text("api_version: v1\nlayer: L1\n", encoding="utf-8")
    (layer / "v1.py").write_text("print('v1')\n", encoding="utf-8")
    (layer / "v2.py").write_text("print('v2')\n", encoding="utf-8")
    worker = layer / "worker.py"
    worker.symlink_to("v1.py")

    first_digest = _submission_content_digest(submission)
    worker.unlink()
    worker.symlink_to("v2.py")

    assert _submission_content_digest(submission) != first_digest


def test_interactive_compile_loop_writes_feedback_while_agent_is_running(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=3),
            "\n".join(
                [
                    "from pathlib import Path",
                    "import json",
                    "import time",
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('c1', ['a'])",
                    "deadline = time.time() + 5",
                    "while time.time() < deadline:",
                    "    feedback = Path('journal/feedback-c1.json')",
                    "    if feedback.exists():",
                    "        data = json.loads(feedback.read_text())",
                    "        seen = Path('journal/c1-feedback-seen.txt')",
                    "        seen.write_text(str(data['raw_rows_included']))",
                    "        write_artifact('c2', ['a', 'b'])",
                    "        break",
                    "    time.sleep(0.05)",
                    "else:",
                    "    raise SystemExit(44)",
                ]
            ),
        )
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 2
    assert result["feedback_count"] == 2
    assert result["failed_submission_count"] == 0
    assert result["stop_reason"] == "ready_for_test"
    assert (attempt.workspace_path / "journal" / "feedback-c1.json").exists()
    assert (attempt.workspace_path / "journal" / "feedback-c2.json").exists()
    assert (attempt.workspace_path / "journal" / "c1-feedback-seen.txt").read_text() == "False"
    feedback_text = "\n".join(
        path.read_text()
        for path in sorted((attempt.workspace_path / "journal").glob("feedback-*.json"))
    )
    assert "r2" not in feedback_text
    assert "r3" not in feedback_text
    assert "r4" not in feedback_text
    assert "expected_output" not in feedback_text
    ledger_path = Path(result["ledger_path"])
    with pytest.raises(ValueError):
        ledger_path.resolve().relative_to(attempt.workspace_path.resolve())
    ledger = json.loads(ledger_path.read_text())
    assert [entry["submission_id"] for entry in ledger] == ["c1", "c2"]
    assert {entry["validation_status"] for entry in ledger} == {"feedback_written"}


def test_interactive_compile_loop_ignores_agent_written_submission_ledger(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=1),
            "\n".join(
                [
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('real', ['a'])",
                ]
            ),
        )
    )
    fake_ledger = attempt.workspace_path / "journal" / "evaluated_submissions.json"
    fake_ledger.write_text(
        json.dumps(
            [
                {
                    "submission_id": "real",
                    "submission_digest": "agent-controlled",
                    "validation_status": "feedback_written",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["feedback_count"] == 1
    assert result["stop_reason"] == "candidate_limit"
    assert (attempt.workspace_path / "journal" / "feedback-real.json").exists()
    trusted_ledger_path = Path(result["ledger_path"])
    assert trusted_ledger_path != fake_ledger
    trusted_ledger = json.loads(trusted_ledger_path.read_text())
    assert [entry["submission_id"] for entry in trusted_ledger] == ["real"]
    assert fake_ledger.read_text(encoding="utf-8")


def test_interactive_compile_loop_stops_at_candidate_limit(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=1),
            "\n".join(
                [
                    "import time",
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('c1', ['a'])",
                    "time.sleep(30)",
                ]
            ),
        )
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["stop_reason"] == "candidate_limit"
    assert json.loads((attempt.workspace_path / "journal" / "closed.json").read_text())[
        "reason"
    ] == "candidate_limit"
    session = json.loads((attempt.workspace_path / "journal" / "agent_session.json").read_text())
    assert session["status"] == "stopped"


def test_interactive_compile_loop_stops_running_agent_at_time_limit(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=1, max_candidates=1),
            "import time; time.sleep(30)",
        )
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 0
    assert result["stop_reason"] == "time_limit"
    assert result["closed_attempt_status"] == "closed"
    session = json.loads((attempt.workspace_path / "journal" / "agent_session.json").read_text())
    assert session["status"] == "timed_out"


def test_interactive_compile_loop_counts_compile_budget_from_agent_start(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=1, max_candidates=1),
            "import time; time.sleep(30)",
        )
    )
    handle = replace(handle, started_at=utcnow() - timedelta(seconds=2))

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 0
    assert result["stop_reason"] == "time_limit"
    assert result["closed_attempt_status"] == "closed"
    assert result["elapsed_seconds"] < 0.5
    session = json.loads((attempt.workspace_path / "journal" / "agent_session.json").read_text())
    assert session["status"] == "timed_out"


def test_interactive_compile_loop_honors_agent_timeout_before_compile_budget(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=10, max_candidates=1),
            "import time; time.sleep(30)",
            agent_timeout_seconds=1,
        )
    )
    session_path = attempt.workspace_path / "journal" / "agent_session.json"
    session = json.loads(session_path.read_text())
    session["timeout_seconds"] = 1000
    session_path.write_text(json.dumps(session), encoding="utf-8")

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 0
    assert result["stop_reason"] == "time_limit"
    assert result["closed_attempt_status"] == "closed"
    assert result["elapsed_seconds"] < 5
    session = json.loads(session_path.read_text())
    assert session["status"] == "timed_out"
    assert session["timeout_seconds"] == 1


def test_interactive_compile_loop_stops_agent_when_timeout_expires_during_validation(
    target_dir: Path, tmp_path: Path, now, monkeypatch
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=10, max_candidates=2),
            "\n".join(
                [
                    "import time",
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('slowvalid', ['a'])",
                    "time.sleep(30)",
                ]
            ),
            agent_timeout_seconds=2,
        )
    )
    handle = replace(handle, started_at=utcnow() - timedelta(seconds=1.5))
    original_evaluate = compile_orchestration_module.evaluate_candidate_on_validation

    def slow_validation(*args, **kwargs):
        time.sleep(0.75)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        "darjeeling.compile_orchestration.evaluate_candidate_on_validation",
        slow_validation,
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["feedback_count"] == 0
    assert result["failed_submission_count"] == 1
    assert result["stop_reason"] == "time_limit"
    assert result["closed_attempt_status"] == "closed"
    feedback = json.loads(
        (attempt.workspace_path / "journal" / "feedback-slowvalid.json").read_text()
    )
    assert feedback["summary"]["status"] == "evaluation_failed"
    assert feedback["summary"]["safe_error_message"] == "The request failed at runtime."
    session = json.loads((attempt.workspace_path / "journal" / "agent_session.json").read_text())
    assert session["status"] == "timed_out"
    assert session["stop_reason"] == "time_limit"


def test_interactive_compile_loop_honors_live_agent_usage_cost_budget(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=10, max_candidates=3, max_cost=2.0),
            "\n".join(
                [
                    "from pathlib import Path",
                    "import time",
                    "Path('journal').mkdir(exist_ok=True)",
                    "Path('journal/agent_usage.json').write_text(",
                    """    '[{"kind":"agent","cost":3.0,"metadata":{}}]\\n',""",
                    "    encoding='utf-8',",
                    ")",
                    "time.sleep(30)",
                ]
            ),
        )
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 0
    assert result["stop_reason"] == "budget_exhausted"
    assert result["closed_attempt_status"] == "closed"
    assert result["total_candidate_cost"] == 3.0
    assert result["elapsed_seconds"] < 5
    session = json.loads((attempt.workspace_path / "journal" / "agent_session.json").read_text())
    assert session["status"] == "stopped"
    assert session["stop_reason"] == "budget_exhausted"


def test_interactive_compile_loop_preserves_usage_high_water_during_validation(
    target_dir: Path, tmp_path: Path, now, monkeypatch
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=10, max_candidates=2, max_cost=2.0),
            "\n".join(
                [
                    "from pathlib import Path",
                    "import time",
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('costtamper', ['a'])",
                    "time.sleep(0.05)",
                    "Path('journal/agent_usage.json').write_text(",
                    """    '[{"kind":"agent","cost":3.0,"metadata":{}}]\\n',""",
                    "    encoding='utf-8',",
                    ")",
                    "time.sleep(0.25)",
                    "Path('journal/agent_usage.json').write_text('[]\\n', encoding='utf-8')",
                    "time.sleep(30)",
                ]
            ),
        )
    )
    original_evaluate = compile_orchestration_module.evaluate_candidate_on_validation

    def slow_validation(*args, **kwargs):
        time.sleep(0.5)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        "darjeeling.compile_orchestration.evaluate_candidate_on_validation",
        slow_validation,
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["feedback_count"] == 0
    assert result["failed_submission_count"] == 1
    assert result["stop_reason"] == "budget_exhausted"
    assert result["closed_attempt_status"] == "closed"
    assert result["total_candidate_cost"] == 3.0
    feedback = json.loads(
        (attempt.workspace_path / "journal" / "feedback-costtamper.json").read_text()
    )
    assert feedback["summary"]["status"] == "evaluation_failed"
    session = json.loads((attempt.workspace_path / "journal" / "agent_session.json").read_text())
    assert session["status"] == "stopped"
    assert session["stop_reason"] == "budget_exhausted"


def test_interactive_compile_loop_tolerates_malformed_agent_usage_ledger(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=1),
            "\n".join(
                [
                    "from pathlib import Path",
                    "Path('journal/agent_usage.json').write_text('{', encoding='utf-8')",
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('usagebad', ['a'])",
                ]
            ),
        )
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["feedback_count"] == 1
    assert result["failed_submission_count"] == 0
    assert result["stop_reason"] == "candidate_limit"
    assert (attempt.workspace_path / "journal" / "feedback-usagebad.json").exists()


def test_interactive_compile_loop_overwrites_stale_evaluation_contract(
    target_dir: Path, tmp_path: Path, now, monkeypatch
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=1),
            "\n".join(
                [
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('contractcheck', ['a'])",
                ]
            ),
        )
    )
    original_evaluate = compile_orchestration_module.evaluate_candidate_on_validation
    seen_contract_path = tmp_path / "seen-contract.txt"

    def assert_current_contract(*args, **kwargs):
        evaluation_options = args[-1]
        seen_contract_path.write_text(
            str(evaluation_options["contract"] is contract), encoding="utf-8"
        )
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        "darjeeling.compile_orchestration.evaluate_candidate_on_validation",
        assert_current_contract,
    )

    stale_contract = replace(contract, contract_hash="stale-contract")
    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts", "contract": stale_contract},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["feedback_count"] == 1
    assert seen_contract_path.read_text(encoding="utf-8") == "True"


def test_interactive_compile_loop_turns_broken_candidate_into_safe_feedback(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=1),
            "\n".join(
                [
                    "from pathlib import Path",
                    "p = Path('submissions/bad/artifacts/l1')",
                    "p.mkdir(parents=True, exist_ok=True)",
                    "payload = 'api_version: v1\\nlayer: L1\\n'",
                    "(p / 'artifact.yaml').write_text(payload, encoding='utf-8')",
                    "Path('submissions/bad/READY').write_text('ready\\n', encoding='utf-8')",
                ]
            ),
        )
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["failed_submission_count"] == 1
    feedback = json.loads((attempt.workspace_path / "journal" / "feedback-bad.json").read_text())
    assert feedback["raw_rows_included"] is False
    assert feedback["summary"]["status"] == "evaluation_failed"
    assert "error_class" in feedback["summary"]
    feedback_text = json.dumps(feedback)
    assert "r2" not in feedback_text
    assert "expected_output" not in feedback_text


def test_interactive_compile_loop_redacts_validation_exception_text(
    target_dir: Path, tmp_path: Path, now, monkeypatch
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=1),
            "\n".join(
                [
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('leaky', ['a'])",
                ]
            ),
        )
    )

    def fail_validation(*args, **kwargs):
        raise RuntimeError("expected_output r2 token=secret raw validation row")

    monkeypatch.setattr(
        "darjeeling.compile_orchestration.evaluate_candidate_on_validation",
        fail_validation,
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["failed_submission_count"] == 1
    feedback = json.loads((attempt.workspace_path / "journal" / "feedback-leaky.json").read_text())
    assert feedback["summary"]["safe_error_message"] == "The request failed at runtime."
    feedback_text = json.dumps(feedback)
    assert "expected_output" not in feedback_text
    assert "secret" not in feedback_text
    assert "raw validation row" not in feedback_text


def test_interactive_compile_loop_turns_digest_failure_into_safe_feedback(
    target_dir: Path, tmp_path: Path, now, monkeypatch
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=1),
            "\n".join(
                [
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('digestbad', ['a'])",
                ]
            ),
        )
    )

    def fail_digest(path: Path) -> str:
        raise RuntimeError("expected_output token=secret while hashing")

    monkeypatch.setattr(
        "darjeeling.compile_orchestration._submission_content_digest",
        fail_digest,
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["failed_submission_count"] == 1
    assert result["stop_reason"] == "candidate_limit"
    assert result["closed_attempt_status"] == "closed"
    feedback = json.loads(
        (attempt.workspace_path / "journal" / "feedback-digestbad.json").read_text()
    )
    assert feedback["summary"]["safe_error_message"] == "The request failed at runtime."
    ledger = json.loads(Path(result["ledger_path"]).read_text())
    assert ledger[0]["submission_digest"].startswith("digest_failed:")
    feedback_text = json.dumps(feedback)
    assert "expected_output" not in feedback_text
    assert "secret" not in feedback_text


def test_interactive_compile_loop_waits_for_ready_marker_before_evaluating(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=1),
            "\n".join(
                [
                    "from pathlib import Path",
                    "import time",
                    "p = Path('submissions/c1/artifacts/l1')",
                    "p.mkdir(parents=True, exist_ok=True)",
                    "payload = 'api_version: v1\\nlayer: L1\\n'",
                    "(p / 'artifact.yaml').write_text(payload, encoding='utf-8')",
                    "time.sleep(0.2)",
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('c1', ['a'])",
                ]
            ),
        )
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["feedback_count"] == 1
    assert result["failed_submission_count"] == 0
    assert (attempt.workspace_path / "journal" / "feedback-c1.json").exists()


def test_interactive_compile_loop_drains_ready_submission_after_agent_exit(
    target_dir: Path, tmp_path: Path, now, monkeypatch
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=2),
            "import time; time.sleep(30)",
        )
    )
    created = False

    def fake_poll(session_handle):
        nonlocal created
        if not created:
            write_artifact(
                attempt.workspace_path / "submissions" / "late" / "artifacts" / "l1",
                definition.contract_hash,
                accept_prefixes=["a"],
            )
            (attempt.workspace_path / "submissions" / "late" / "READY").write_text(
                "ready\n", encoding="utf-8"
            )
            created = True
        return replace(session_handle, status="completed")

    monkeypatch.setattr("darjeeling.compile_orchestration.poll_agent_session", fake_poll)

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 1
    assert result["feedback_count"] == 1
    assert result["stop_reason"] == "ready_for_test"
    assert (attempt.workspace_path / "journal" / "feedback-late.json").exists()


def test_interactive_compile_loop_uses_cumulative_compile_cost_without_double_counting(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, compile_run, attempt, handle = (
        _launch_interactive_agent(
            target_dir,
            tmp_path,
            now,
            CompileBudget(max_agent_seconds=5, max_candidates=3, max_cost=15.0),
            "\n".join(
                [
                    "from pathlib import Path",
                    "Path('journal/agent_usage.json').write_text(",
                    """    '[{"kind":"agent","cost":10.0,"metadata":{}}]\\n',""",
                    "    encoding='utf-8',",
                    ")",
                    _write_agent_artifact_snippet(
                        load_checked_target(target_dir)[0].contract_hash
                    ),
                    "write_artifact('c1', ['a'])",
                    "write_artifact('c2', ['a', 'b'])",
                ]
            ),
        )
    )

    result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        snapshot.reference_usage,
        {"serving_l4_cost": 1.0},
        {"artifact_store": tmp_path / "artifacts"},
        poll_interval_seconds=0.02,
    )

    assert result["evaluated_submission_count"] == 2
    assert result["stop_reason"] == "ready_for_test"
    assert result["total_candidate_cost"] == 10.0


def test_first_compile_after_cold_start_uses_recompile_request_path(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "user",
    )
    policy = SchedulerPolicy(
        1,
        True,
        True,
        False,
        default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
    )
    decision = plan_compile_launch(definition, contract, check, request, release, [], [], policy)
    assert decision.status == "accepted"
    compile_run_store = CompileRunStore()
    outside_secret = tmp_path / "snapshot-secret.txt"
    outside_secret.write_text("secret\n", encoding="utf-8")
    agent_code = "\n".join(
        [
            "from pathlib import Path",
            "outside = Path(" + repr(str(outside_secret)) + ")",
            "try:",
            "    outside.read_text()",
            "    raise SystemExit(31)",
            "except PermissionError:",
            "    pass",
            "Path('journal/launched.txt').write_text('ok')",
        ]
    )
    report = AgentVisibleReport(
        "report-1",
        "candidate-1",
        definition.name,
        definition.contract_hash,
        "snapshot-prior",
        release.release_id,
        AgentVisibleDecisionSummary(
            "candidate-1", "rejected", "regression", {}, {}, False, None
        ),
        {"sample_count": 1},
        None,
        False,
        None,
        {},
        {},
        utcnow(),
    )
    telemetry = AgentVisibleTelemetrySummary(
        definition.name,
        release.release_id,
        definition.contract_hash,
        {"requests": 10},
        {},
        None,
        "v1",
        utcnow(),
    )
    compile_run, attempt, handle = start_compile_launch(
        decision,
        definition,
        contract,
        check,
        definition.data_config,
        request,
        release,
        [],
        PrefixBroker(),
        WorkspaceStore(tmp_path / "workspaces"),
        compile_run_store,
        CompileOptions(),
        AgentAttemptOptions(
            agent_command=[
                "/usr/bin/python3",
                "-c",
                agent_code,
            ]
        ),
        report_views=[report],
        telemetry_summaries=[telemetry],
    )
    mounted = "\n".join(
        path.read_text(errors="ignore")
        for path in attempt.workspace_path.rglob("*")
        if path.is_file()
    )
    assert compile_run.base_release_id == release.release_id
    assert compile_run_store.runs[compile_run.compile_id] == compile_run
    assert handle.status == "completed"
    deferred = plan_compile_launch(
        definition,
        contract,
        check,
        request,
        release,
        [],
        [replace(compile_run, status="closing")],
        policy,
    )
    assert deferred.status == "deferred"
    assert deferred.reason == "max concurrent compiles reached"
    assert (attempt.workspace_path / "journal" / "launched.txt").exists()
    assert "report-1" in (
        attempt.workspace_path / "readonly_inputs" / "agent_visible_reports.json"
    ).read_text()
    assert "requests" in (
        attempt.workspace_path / "readonly_inputs" / "agent_visible_telemetry.json"
    ).read_text()
    failed_store = CompileRunStore()
    with pytest.raises(WorkspaceError, match="telemetry mount"):
        start_compile_launch(
            decision,
            definition,
            contract,
            check,
            definition.data_config,
            request,
            release,
            [],
            PrefixBroker(),
            WorkspaceStore(tmp_path / "workspaces"),
            failed_store,
            CompileOptions(),
            AgentAttemptOptions(agent_command=["/usr/bin/python3", "-c", "print('no launch')"]),
            report_views=[report],
            telemetry_summaries=[replace(telemetry, contract_hash="other")],
        )
    assert failed_store.runs == {}
    assert "r2" not in mounted
    assert "r4" not in mounted


def test_plan_compile_launch_rejects_target_and_telemetry_scope_mismatch(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "user",
    )
    policy = SchedulerPolicy(
        1,
        True,
        True,
        False,
        default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
    )

    decision = plan_compile_launch(
        definition,
        contract,
        replace(check, contract_hash="other-contract"),
        request,
        release,
        [],
        [],
        policy,
    )
    assert decision.status == "rejected"
    assert decision.reason == "target scope mismatch"

    bad_source = TelemetryDataSource(
        "telemetry-1",
        "other-target",
        definition.contract_hash,
        utcnow(),
        str(tmp_path / "records.json"),
        ["train"],
        None,
        ["user_feedback"],
        "digest",
    )
    decision = plan_compile_launch(
        definition,
        contract,
        check,
        replace(request, telemetry_source=bad_source),
        release,
        [],
        [],
        policy,
    )
    assert decision.status == "rejected"
    assert decision.reason == "telemetry source scope mismatch"


def test_plan_compile_launch_rejects_unsupported_recompile_trigger(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "bot",  # type: ignore[arg-type]
    )

    decision = plan_compile_launch(
        definition,
        contract,
        check,
        request,
        release,
        [],
        [],
        SchedulerPolicy(
            1,
            True,
            True,
            False,
            default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
        ),
    )

    assert decision.status == "rejected"
    assert decision.reason == "unsupported recompile trigger"
    assert decision.budget is None
    assert decision.snapshot_cutoff_time is None
    assert decision.snapshot_options is None


def test_start_compile_launch_rejects_stale_accepted_decision_before_snapshot(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "user",
    )
    decision = plan_compile_launch(
        definition,
        contract,
        check,
        request,
        release,
        [],
        [],
        SchedulerPolicy(
            1,
            True,
            True,
            False,
            default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
        ),
    )

    compile_run_store = CompileRunStore()
    with pytest.raises(CompileLaunchError, match="scope mismatch"):
        start_compile_launch(
            replace(decision, base_release_id="other-release"),
            definition,
            contract,
            check,
            definition.data_config,
            request,
            release,
            [],
            PrefixBroker(),
            WorkspaceStore(tmp_path / "workspaces"),
            compile_run_store,
            CompileOptions(),
            AgentAttemptOptions(agent_command=["/usr/bin/python3", "-c", "print('no launch')"]),
        )
    assert compile_run_store.runs == {}

    with pytest.raises(CompileLaunchError, match="target checks failed"):
        start_compile_launch(
            decision,
            definition,
            contract,
            replace(check, status="fail", failures=["stale target check failure"]),
            definition.data_config,
            request,
            release,
            [],
            PrefixBroker(),
            WorkspaceStore(tmp_path / "workspaces"),
            compile_run_store,
            CompileOptions(),
            AgentAttemptOptions(agent_command=["/usr/bin/python3", "-c", "print('no launch')"]),
        )
    assert compile_run_store.runs == {}
    assert not (tmp_path / "snapshots").exists()


def test_insufficient_reference_approval_reaches_snapshot_and_workspace(
    target_dir: Path, tmp_path: Path
) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    for source in data["sources"]:
        for record in source["records"]:
            record.pop("reference_output", None)
            record.pop("reference_source", None)
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data), encoding="utf-8")
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "user",
    )
    decision = plan_compile_launch(
        definition,
        contract,
        check,
        request,
        release,
        [],
        [],
        SchedulerPolicy(
            1,
            True,
            True,
            True,
            default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
        ),
    )

    with pytest.raises(CompileLaunchError, match="reference qualification insufficient"):
        start_compile_launch(
            decision,
            definition,
            contract,
            check,
            definition.data_config,
            request,
            release,
            [],
            PrefixBroker(),
            WorkspaceStore(tmp_path / "workspaces-no-approval"),
            CompileRunStore(),
            CompileOptions(),
            AgentAttemptOptions(agent_command=["/usr/bin/python3", "-c", "print('no launch')"]),
        )

    compile_run_store = CompileRunStore()
    compile_run, _, handle = start_compile_launch(
        decision,
        definition,
        contract,
        check,
        definition.data_config,
        request,
        release,
        [],
        PrefixBroker(),
        WorkspaceStore(tmp_path / "workspaces-approved"),
        compile_run_store,
        CompileOptions(allow_insufficient_reference_qualification=True),
        AgentAttemptOptions(
            agent_command=[
                "/usr/bin/python3",
                "-c",
                "from pathlib import Path; Path('journal/launched.txt').write_text('ok')",
            ]
        ),
    )
    assert handle.status == "completed"
    assert compile_run_store.runs[compile_run.compile_id] == compile_run
