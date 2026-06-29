from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from darjeeling.agent_workspace import (
    close_agent_attempt,
    create_agent_workspace,
    create_compile_run,
    load_target_workspace,
    receive_candidate_submission,
)
from darjeeling.candidate_evaluation import (
    compare_candidates,
    evaluate_candidate_on_test,
    evaluate_candidate_on_validation,
    finalize_report,
)
from darjeeling.model import (
    AgentAttemptOptions,
    AgentUsageLedger,
    ApprovalRecord,
    CompileBudget,
    CompileOptions,
    L4BaselineSummary,
    ReferenceContext,
    ReferenceResponse,
    ReleaseBaseline,
    ReleaseRegistry,
    ResultCache,
    RoutingSettings,
    RuntimeRequest,
    SnapshotOptions,
    WorkerPool,
    WorkspaceStore,
)
from darjeeling.release_runtime import (
    create_release,
    create_release_without_artifacts,
    serve_request,
    set_channel,
)
from darjeeling.runtime_trace_metrics import aggregate_runtime_metrics, write_trace
from darjeeling.snapshot_reference import build_snapshot
from darjeeling.target_definition import load_checked_target
from darjeeling.util import new_id, utcnow


@dataclass(frozen=True)
class DemoRequestResult:
    name: str
    request_text: str
    path: str
    output: str | None


@dataclass(frozen=True)
class ThinTargetDemoReport:
    cold_start_path: str
    request_results: tuple[DemoRequestResult, ...]
    precision: float | None
    local_coverage: float
    fallback_share: float
    avg_latency_ms: float
    p95_latency_ms: float
    estimated_cost_per_1000: float
    estimated_saving_per_1000: float


class ToyReferenceBroker:
    reference_version = "thin-target-reference-v1"
    cost_per_call = 0.01

    def call(self, request: dict[str, Any], context: ReferenceContext) -> ReferenceResponse:
        text = request["input"]["text"]
        return ReferenceResponse(
            payload={"label": text.split(":", 1)[0]},
            reference_source="versioned_l4",
            reference_version=self.reference_version,
            cost=self.cost_per_call,
            latency_ms=3.0,
        )


def run_thin_target_demo() -> ThinTargetDemoReport:
    with tempfile.TemporaryDirectory(prefix="darjeeling-demo-") as root_raw:
        root = Path(root_raw)
        now = datetime(2026, 6, 28, 12, 0, tzinfo=UTC)
        target_dir = _write_toy_target(root / "target", now)
        definition, contract, check = load_checked_target(target_dir)
        broker = ToyReferenceBroker()
        registry = ReleaseRegistry()
        cold = create_release_without_artifacts(
            definition, contract, check, broker, RoutingSettings(), registry
        )
        set_channel(definition.name, "stable", cold.release_id, {}, registry)
        traces = []

        def record_trace(*args: Any):
            trace = write_trace(*args)
            traces.append(trace)
            return trace

        worker_pool = WorkerPool()
        cache = ResultCache()
        cold_response = serve_request(
            RuntimeRequest("demo-cold", definition.name, {"text": "a:cold-start"}),
            registry,
            lambda _name: contract,
            worker_pool,
            lambda: new_id("trace"),
            record_trace,
            None,
            {},
            {},
            [],
            definition.runtime_config.telemetry_privacy_policy,
            [],
            broker,
            cache,
        )
        snapshot = build_snapshot(
            definition,
            contract,
            definition.data_config,
            None,
            [],
            broker,
            now,
            SnapshotOptions(storage_root=root / "snapshots"),
        )
        workspace = load_target_workspace(
            definition.name,
            definition.contract_hash,
            WorkspaceStore(root / "workspaces"),
        )
        compile_run = create_compile_run(
            definition,
            check,
            snapshot.snapshot,
            cold,
            CompileBudget(),
            workspace,
            snapshot.reference_qualification,
            CompileOptions(),
        )
        attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
        artifact_dir = attempt.workspace_path / "submissions" / "c1" / "artifacts" / "l1"
        _write_prefix_artifact(artifact_dir, definition.contract_hash, ["a", "b"])
        submission = receive_candidate_submission(
            attempt, attempt.workspace_path / "submissions" / "c1"
        )
        validation = evaluate_candidate_on_validation(
            submission,
            definition,
            snapshot.snapshot,
            cold,
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "artifact_store": root / "artifacts"},
        )
        closed = close_agent_attempt(attempt, "ready_for_test")
        test = evaluate_candidate_on_test(
            validation["candidate"],
            closed,
            definition,
            snapshot.snapshot,
            cold,
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {
                "contract": contract,
                "test_results_visible": True,
                "validation_report": validation["report"],
            },
        )
        decision = compare_candidates(
            [test["report"]],
            ReleaseBaseline(
                cold,
                l4_baseline=L4BaselineSummary(
                    cold.release_id,
                    definition.name,
                    definition.contract_hash,
                    {},
                    [],
                    snapshot.reference_qualification,
                    {},
                    {},
                ),
            ),
            {"requirements": definition.requirements},
        )
        final = finalize_report(test["report"], decision, validation["report"])
        approval = ApprovalRecord(
            new_id("approval"),
            validation["candidate"].candidate_id,
            final.report_id,
            definition.name,
            definition.contract_hash,
            snapshot.snapshot.snapshot_id,
            utcnow(),
            "user",
        )
        compiled = create_release(
            validation["candidate"], snapshot.snapshot, cold, final, approval, root / "artifacts"
        )
        registry.releases[compiled.release_id] = compiled
        set_channel(definition.name, "stable", compiled.release_id, {}, registry)

        compiled_results = []
        for request_id, text in [
            ("demo-local-a", "a:known-local"),
            ("demo-local-b", "b:known-local"),
            ("demo-fallback", "z:unfamiliar"),
        ]:
            response = serve_request(
                RuntimeRequest(request_id, definition.name, {"text": text}),
                registry,
                lambda _name: contract,
                worker_pool,
                lambda: new_id("trace"),
                record_trace,
                None,
                {},
                {},
                [],
                definition.runtime_config.telemetry_privacy_policy,
                [],
                broker,
                cache,
            )
            compiled_results.append(
                DemoRequestResult(
                    name=request_id,
                    request_text=text,
                    path=_public_path_name(response.chosen_layer),
                    output=(response.output or {}).get("label") if response.output else None,
                )
            )

        compiled_traces = [trace for trace in traces if trace.release_id == compiled.release_id]
        metrics = aggregate_runtime_metrics(
            compiled_traces,
            [],
            (utcnow().replace(year=2020), utcnow().replace(year=2030)),
            compiled,
        )
        baseline_cost_per_1000 = broker.cost_per_call * 1000.0
        estimated_cost_per_1000 = float(metrics.cost["serving_cost_per_1000"])
        precision = final.metrics["local"]["precision"]
        return ThinTargetDemoReport(
            cold_start_path=_public_path_name(cold_response.chosen_layer),
            request_results=tuple(compiled_results),
            precision=precision,
            local_coverage=metrics.local_coverage,
            fallback_share=metrics.l4_fallback_rate,
            avg_latency_ms=float(metrics.latency["avg_ms"]),
            p95_latency_ms=float(metrics.latency["p95_latency_ms"]),
            estimated_cost_per_1000=estimated_cost_per_1000,
            estimated_saving_per_1000=baseline_cost_per_1000 - estimated_cost_per_1000,
        )


def format_thin_target_report(report: ThinTargetDemoReport) -> str:
    lines = [
        "Darjeeling thin-target demo",
        "",
        "This uses toy data and a simulated reference LLM. It does not call external APIs.",
        "",
        "Cold start",
        f"  demo-cold -> {report.cold_start_path}",
        "",
        "Compiled local artifact",
    ]
    for result in report.request_results:
        lines.append(
            f"  {result.name} ({result.request_text}) -> {result.path}; output={result.output}"
        )
    lines.extend(
        [
            "",
            "Report",
            f"  precision: {_format_percent(report.precision)} on the toy test sample",
            f"  local coverage: {_format_percent(report.local_coverage)} on demo requests",
            f"  fallback share: {_format_percent(report.fallback_share)}",
            f"  average latency: {report.avg_latency_ms:.2f} ms",
            f"  p95 latency: {report.p95_latency_ms:.2f} ms",
            f"  estimated serving cost: ${report.estimated_cost_per_1000:.4f} / 1,000 requests",
            f"  estimated saving: ${report.estimated_saving_per_1000:.4f} / 1,000 requests",
        ]
    )
    return "\n".join(lines)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _public_path_name(layer: str | None) -> str:
    if layer in {"L1", "L2", "L3"}:
        return "local artifact accepted"
    if layer == "L4":
        return "simulated reference LLM fallback"
    if layer == "cache":
        return "result cache"
    return "runtime error"


def _write_toy_target(root: Path, now: datetime) -> Path:
    (root / "schemas").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "schemas" / "input.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            }
        ),
        encoding="utf-8",
    )
    (root / "schemas" / "output.json").write_text(
        json.dumps(
            {"type": "object", "required": ["label"], "properties": {"label": {"type": "string"}}}
        ),
        encoding="utf-8",
    )
    (root / "target.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "thin-target",
                "version": "1",
                "schemas": {"input": "schemas/input.json", "output": "schemas/output.json"},
                "contract": "contract.py",
                "reference": "reference.py",
                "requirements": {
                    "precision_min": 0.9,
                    "min_accepted_samples": 1,
                    "wrong_accept_rate_max": 0.05,
                },
                "runtime": {
                    "telemetry_privacy_policy": {
                        "policy_version": "demo",
                        "allowed_sources": ["l4_fallback"],
                        "default_approved_for_by_source": {"l4_fallback": ["train"]},
                        "raw_payload_allowed": True,
                        "canonicalization_required": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    rows = []
    split_roles = [
        ["train"],
        ["train"],
        ["validation_candidate"],
        ["validation_candidate"],
        ["test_candidate"],
        ["test_candidate"],
    ]
    for index, role in enumerate(split_roles):
        label = "a" if index % 2 == 0 else "b"
        rows.append(
            {
                "record_id": f"toy-{index}",
                "input": {"text": f"{label}:sample-{index}"},
                "reference_output": {"label": label},
                "reference_source": "gold" if index < 2 else "versioned_l4",
                "split_eligibility": role,
                "source_timestamp": (now - timedelta(days=1)).isoformat(),
            }
        )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "default_split_eligibility": ["train"],
                "sources": [{"name": "toy-inline", "records": rows}],
            }
        ),
        encoding="utf-8",
    )
    (root / "contract.py").write_text(
        """
def validate_input(value):
    return dict(value)

def validate_output(value):
    return dict(value)

def is_correct(output, reference):
    return output == reference

def normalize_input(input_value):
    return input_value["text"].lower()

def split_group(record):
    return record.input["text"].lower()

def slice_tags(record):
    return [record.input["text"].split(":", 1)[0]]

def redact_for_trace(value):
    result = dict(value)
    if "text" in result:
        result["text"] = "<redacted>"
    return result

def bucket_runtime_metadata(metadata):
    return {}
""".lstrip(),
        encoding="utf-8",
    )
    (root / "reference.py").write_text(
        """
def build_reference_request(input_value, reference_context):
    return {"input": input_value, "purpose": reference_context.purpose}

def parse_reference_response(response):
    return response.payload
""".lstrip(),
        encoding="utf-8",
    )
    (root / "tests" / "contract_cases.yaml").write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "name": "valid_prefix",
                        "input": {"text": "a:case"},
                        "output": {"label": "a"},
                        "reference_output": {"label": "a"},
                        "correct": True,
                        "normalized_input": "a:case",
                    },
                    {"name": "invalid_input", "invalid_input": {"wrong": "missing text"}},
                    {"name": "invalid_output", "invalid_output": {"wrong": "shape"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_prefix_artifact(path: Path, contract_hash: str, accept_prefixes: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "worker.py").write_text(
        f"""
import json
import sys

request = json.loads(sys.stdin.readline())
text = request["input"]["text"]
prefixes = {accept_prefixes!r}
if any(text.startswith(prefix + ":") for prefix in prefixes):
    print(json.dumps({{
        "decision": "accept",
        "output": {{"label": text.split(":", 1)[0]}},
        "confidence": 0.99,
        "reason": "prefix_match",
    }}))
else:
    print(json.dumps({{"decision": "abstain", "confidence": 0.1, "reason": "outside"}}))
""".lstrip(),
        encoding="utf-8",
    )
    (path / "healthcheck.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (path / "artifact.yaml").write_text(
        yaml.safe_dump(
            {
                "api_version": "v1",
                "layer": "L1",
                "start_command": ["python3", "worker.py"],
                "healthcheck_command": ["python3", "healthcheck.py"],
                "protocol": "jsonl",
                "timeout_ms": 1000,
                "memory_mb": 64,
                "network": "disabled",
                "contract_hash": contract_hash,
                "allowed_reason_codes": ["prefix_match", "outside"],
            }
        ),
        encoding="utf-8",
    )
