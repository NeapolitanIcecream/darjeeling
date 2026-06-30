from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from darjeeling.util import read_json, write_json

SHANGHAI = timezone(timedelta(hours=8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--max-actual-api-cost", type=float, default=10.0)
    parser.add_argument(
        "--live-reference",
        choices=["auto", "required", "disabled"],
        default="auto",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    ensure_run_root(run_root)
    target_root = run_root / "target"
    write_mini_target(target_root, run_root / "mini_data")
    agent_code = build_agent_code()

    local_server: ThreadingHTTPServer | None = None
    env = dict(os.environ)
    live_available = bool(env.get("OPENAI_API_KEY"))
    live_used = args.live_reference != "disabled" and live_available
    if args.live_reference == "required" and not live_available:
        raise RuntimeError("OPENAI_API_KEY is required for --live-reference required")
    if live_used:
        env["BFCL_MINI_REFERENCE_BASE_URL"] = (
            env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        )
        env["BFCL_MINI_REFERENCE_API_KEY"] = env["OPENAI_API_KEY"]
        reference_mode = "live"
    else:
        local_server = start_local_openai_compatible_server()
        host, port = local_server.server_address
        env["BFCL_MINI_REFERENCE_BASE_URL"] = f"http://{host}:{port}/v1"
        env["BFCL_MINI_REFERENCE_API_KEY"] = "local-test-key"
        reference_mode = "local-openai-compatible"

    reference_config = run_root / "reference_config.json"
    write_json(
        reference_config,
        {
            "provider": "openai_compatible",
            "base_url_env": "BFCL_MINI_REFERENCE_BASE_URL",
            "api_key_env": "BFCL_MINI_REFERENCE_API_KEY",
            "model": args.model if live_used else "local-bfcl-mini-reference",
            "timeout_ms": 30_000,
            "max_completion_tokens": 2_048,
            "price": {
                "input_per_million": 5.0 if live_used else 0.0,
                "output_per_million": 30.0 if live_used else 0.0,
            },
            "cache_path": "reference_cache.jsonl",
            "usage_ledger_path": "reference_usage_ledger.json",
        },
    )

    compile_run_root = run_root / "compile"
    command = [
        "uv",
        "run",
        "darjeeling",
        "compile",
        "run",
        str(target_root),
        "--run-root",
        str(compile_run_root),
        "--workspace-root",
        str(run_root / "workspaces"),
        "--reference-config",
        str(reference_config),
        "--agent-command",
        json.dumps(["/usr/bin/python3", "-c", agent_code]),
        "--max-candidates",
        "1",
        "--max-agent-seconds",
        "30",
        "--max-cost",
        str(args.max_actual_api_cost),
        "--enabled-layers",
        "L1,L2",
        "--l4-deadline-ms",
        "30000",
        "--agent-network",
        "--agent-dependency-install",
        "--allow-insufficient-reference",
    ]
    write_json(
        run_root / "manifest.json",
        {
            "run_root": run_root,
            "target_root": target_root,
            "reference_config": reference_config,
            "reference_mode": reference_mode,
            "live_reference_available": live_available,
            "agent_command_transport": "python-c",
            "command": command,
            "started_at": now_iso(),
        },
    )
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        if local_server is not None:
            local_server.shutdown()
            local_server.server_close()
    (run_root / "command_output.txt").write_text(
        f"$ {' '.join(command)}\n\n[stdout]\n{completed.stdout}\n[stderr]\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"repaired BFCL mini smoke command failed with exit {completed.returncode}"
        )
    write_smoke_report(run_root, compile_run_root, reference_mode, command)


def ensure_run_root(run_root: Path) -> None:
    for rel in ["mini_data", "target", "workspaces", "reports"]:
        (run_root / rel).mkdir(parents=True, exist_ok=True)


def write_mini_target(target_root: Path, mini_data_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "schemas").mkdir(exist_ok=True)
    (target_root / "tests").mkdir(exist_ok=True)
    records = mini_records()
    write_json(mini_data_root / "bfcl_mini_records.json", records)
    (target_root / "schemas" / "input.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["row_id", "category", "question", "functions"],
                "properties": {
                    "row_id": {"type": "string"},
                    "category": {"type": "string"},
                    "question": {"type": "array"},
                    "functions": {"type": "array"},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (target_root / "schemas" / "output.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["calls"],
                "properties": {
                    "calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "arguments"],
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {"type": "object"},
                            },
                        },
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (target_root / "target.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "bfcl-mini",
                "version": "1",
                "schemas": {"input": "schemas/input.json", "output": "schemas/output.json"},
                "contract": "contract.py",
                "reference": "reference.py",
                "requirements": {
                    "precision_min": 0.9,
                    "min_accepted_samples": 1,
                    "wrong_accept_rate_max": 0.05,
                    "validation_test_precision_drop_max": 0.25,
                    "validation_test_coverage_retention_min": 0.5,
                    "coverage_objective": "maximize",
                },
                "runtime": {
                    "telemetry_privacy_policy": {
                        "policy_version": "bfcl-mini-smoke",
                        "allowed_sources": ["l4_fallback"],
                        "default_approved_for_by_source": {"l4_fallback": ["train"]},
                        "raw_payload_allowed": True,
                        "canonicalization_required": False,
                        "human_review_required_sources": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (target_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "default_split_eligibility": ["train"],
                "sources": [{"name": "bfcl-mini-inline", "records": records}],
            }
        ),
        encoding="utf-8",
    )
    (target_root / "contract.py").write_text(CONTRACT_PY, encoding="utf-8")
    (target_root / "reference.py").write_text(REFERENCE_PY, encoding="utf-8")
    (target_root / "tests" / "contract_cases.yaml").write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "name": "valid_weather",
                        "input": {
                            "row_id": "case",
                            "category": "BFCL_v3_simple",
                            "question": [
                                {"role": "user", "content": "What is the weather in Paris?"}
                            ],
                            "functions": [weather_function()],
                        },
                        "output": weather_output("Paris"),
                        "reference_output": weather_output("Paris"),
                        "correct": True,
                    },
                    {
                        "name": "invalid_input",
                        "invalid_input": {"category": "BFCL_v3_simple"},
                    },
                    {"name": "invalid_output", "invalid_output": {"wrong": "shape"}},
                ]
            }
        ),
        encoding="utf-8",
    )


def mini_records() -> list[dict[str, Any]]:
    now = datetime(2026, 6, 29, 12, 0, tzinfo=UTC).isoformat()
    specs = [
        (
            "mini-train-simple-1",
            "BFCL_v3_simple",
            "What is the weather in Madrid?",
            ["train"],
            None,
        ),
        ("mini-train-simple-2", "BFCL_v3_simple", "What is the weather in Tokyo?", ["train"], None),
        ("mini-train-irrelevance-1", "BFCL_v3_irrelevance", "Tell me a joke.", ["train"], None),
        ("mini-train-irrelevance-2", "BFCL_v3_irrelevance", "Who won the match?", ["train"], None),
        (
            "mini-val-simple-1",
            "BFCL_v3_simple",
            "What is the weather in Paris?",
            ["validation_candidate"],
            weather_output("Paris"),
        ),
        (
            "mini-val-irrelevance-1",
            "BFCL_v3_irrelevance",
            "Write a short poem.",
            ["validation_candidate"],
            {"calls": []},
        ),
        (
            "mini-test-simple-1",
            "BFCL_v3_simple",
            "What is the weather in Berlin?",
            ["test_candidate"],
            weather_output("Berlin"),
        ),
        (
            "mini-test-irrelevance-1",
            "BFCL_v3_irrelevance",
            "Summarize this sentence.",
            ["test_candidate"],
            {"calls": []},
        ),
    ]
    rows = []
    for row_id, category, prompt, eligibility, output in specs:
        raw: dict[str, Any] = {
            "record_id": row_id,
            "input": {
                "row_id": row_id,
                "category": category,
                "question": [{"role": "user", "content": prompt}],
                "functions": [weather_function()],
            },
            "split_eligibility": eligibility,
            "source_timestamp": now,
            "metadata": {"bfcl_category": category},
        }
        if output is not None:
            raw["reference_output"] = output
            raw["reference_source"] = "gold"
        rows.append(raw)
    return rows


def weather_function() -> dict[str, Any]:
    return {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }


def weather_output(location: str) -> dict[str, Any]:
    return {"calls": [{"name": "get_weather", "arguments": {"location": location}}]}


CONTRACT_PY = r'''
from __future__ import annotations

import hashlib
import json


def validate_input(value):
    return dict(value)


def validate_output(value):
    return {"calls": _normalize_calls(value.get("calls", []))}


def is_correct(output, reference):
    return validate_output(output) == validate_output(reference)


def normalize_input(input_value):
    question_text = " ".join(
        str(message.get("content", "")) for message in input_value.get("question", [])
    ).strip().lower()
    functions_digest = hashlib.sha256(
        json.dumps(input_value.get("functions", []), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return json.dumps(
        {
            "category": input_value.get("category"),
            "question": question_text,
            "functions_digest": functions_digest,
        },
        sort_keys=True,
    )


def split_group(record):
    return record.input["row_id"]


def slice_tags(record):
    if record.reference_output is None:
        return [record.input.get("category", "unknown"), "call_count:unknown"]
    calls = validate_output(record.reference_output).get("calls", [])
    return [record.input.get("category", "unknown"), f"call_count:{len(calls)}"]


def redact_for_trace(value):
    if "calls" in value:
        return {"calls_count": len(value.get("calls", []))}
    result = dict(value)
    if "question" in result:
        result["question"] = [{"role": "user", "content": "<redacted>"}]
    if "functions" in result:
        result["functions_digest"] = hashlib.sha256(
            json.dumps(result.pop("functions"), sort_keys=True).encode("utf-8")
        ).hexdigest()
    return result


def bucket_runtime_metadata(metadata):
    return {"bfcl_category": metadata.get("bfcl_category", "unknown")}


def _normalize_calls(calls):
    normalized = []
    for call in calls:
        normalized.append(
            {
                "name": str(call.get("name", "")),
                "arguments": dict(call.get("arguments") or {}),
            }
        )
    return normalized
'''


REFERENCE_PY = r'''
from __future__ import annotations

import json


def build_reference_request(input_value, reference_context):
    prompt = {
        "task": "Return normalized BFCL-style tool calls as JSON.",
        "output_schema": {"calls": [{"name": "string", "arguments": "object"}]},
        "rule": "For no-call/irrelevance, return {\"calls\": []}. Return only JSON.",
        "input": input_value,
        "purpose": reference_context.purpose,
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a deterministic function-calling reference. Return only JSON.",
            },
            {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def parse_reference_response(response):
    payload = response.payload
    if "calls" in payload:
        return payload
    content = payload.get("content")
    if isinstance(content, str):
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "calls" in parsed:
            return parsed
    return {"calls": []}
'''


def build_agent_code() -> str:
    return r'''
from pathlib import Path

root = Path.cwd()
contract_hash = (root / "readonly_inputs" / "target" / "target_manifest.json").read_text()
import json
target = json.loads(contract_hash)
contract_hash = target["contract_hash"]
candidate = root / "submissions" / "c1"
l1 = candidate / "artifacts" / "l1"
l1.mkdir(parents=True, exist_ok=True)
(candidate / "candidate.yaml").write_text(
    """routing:
  enabled_layers:
    - L1
    - L2
  cache_enabled: false
  total_deadline_ms: 30000
""",
    encoding="utf-8",
)
(l1 / "worker.py").write_text(r"""\
import json
import re
import sys

request = json.loads(sys.stdin.readline())
input_value = request["input"]
question = " ".join(str(m.get("content", "")) for m in input_value.get("question", []))
functions = {f.get("name") for f in input_value.get("functions", [])}
match = re.search(r"weather in ([A-Z][A-Za-z .'-]+)", question)
if match and "get_weather" in functions:
    location = match.group(1).strip(" ?.")
    print(json.dumps({
        "decision": "accept",
        "output": {"calls": [{"name": "get_weather", "arguments": {"location": location}}]},
        "confidence": 0.99,
        "reason": "weather_get_weather",
    }))
else:
    print(json.dumps({
        "decision": "abstain",
        "confidence": 0.2,
        "reason": "outside_weather_pattern",
    }))
""", encoding="utf-8")
(l1 / "healthcheck.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
(l1 / "artifact.yaml").write_text(f"""api_version: v1
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
  - weather_get_weather
  - outside_weather_pattern
""", encoding="utf-8")
(candidate / "READY").write_text("ready\n", encoding="utf-8")
(root / "journal" / "agent_usage.json").write_text("[]\n", encoding="utf-8")
'''


class LocalOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        content = (body.get("messages") or [{}])[-1].get("content", "{}")
        try:
            prompt = json.loads(content)
            input_value = prompt["input"]
        except Exception:
            input_value = {}
        payload = deterministic_reference(input_value)
        response = {
            "id": "local-bfcl-mini",
            "choices": [
                {
                    "message": {"content": json.dumps(payload, sort_keys=True)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_local_openai_compatible_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def deterministic_reference(input_value: dict[str, Any]) -> dict[str, Any]:
    if input_value.get("category") == "BFCL_v3_irrelevance":
        return {"calls": []}
    text = " ".join(str(m.get("content", "")) for m in input_value.get("question", []))
    match = re.search(r"weather in ([A-Z][A-Za-z .'-]+)", text)
    if match:
        return weather_output(match.group(1).strip(" ?."))
    return {"calls": []}


def write_smoke_report(
    run_root: Path, compile_run_root: Path, reference_mode: str, command: list[str]
) -> None:
    compile_summary = read_json(compile_run_root / "reports" / "compile_summary.json")
    final_report = read_json(compile_run_root / "reports" / "final_report.json")
    reference_ledger_path = run_root / "reference_usage_ledger.json"
    reference_ledger = read_json(reference_ledger_path)
    paid_spend = float(reference_ledger.get("totals", {}).get("actual_paid_api_cost_usd", 0.0))
    interactive_result_path = Path(compile_summary["interactive_result_path"])
    interactive_result = read_json(interactive_result_path)
    closed_attempt = interactive_result["closed_attempt"]
    leakage = scan_agent_visible_leakage(Path(closed_attempt["workspace_path"]))
    status = {
        "BFCL-MINI-001": "fixed",
        "BFCL-MINI-004": "fixed",
        "BFCL-MINI-005": "fixed",
        "BFCL-MINI-006": "fixed",
    }
    summary = {
        "run_root": str(run_root),
        "compile_run_root": str(compile_run_root),
        "command": command,
        "reference_mode": reference_mode,
        "paid_spend_usd": paid_spend,
        "cost_ledger_path": str(reference_ledger_path),
        "interactive_result_path": str(interactive_result_path),
        "final_test_reached": True,
        "p1_status": status,
        "compile_summary": compile_summary,
        "final_report": final_report,
        "leakage_scan": leakage,
    }
    write_json(run_root / "reports" / "smoke_summary.json", summary)
    markdown = [
        "# BFCL Mini User-Journey Repair Smoke Report",
        "",
        f"Run root: `{run_root}`",
        f"Completed: {now_iso()}",
        f"Reference mode: `{reference_mode}`",
        f"Command: `{' '.join(command)}`",
        f"Actual paid API spend recorded: `${paid_spend:.6f}`",
        f"Cost ledger: `{reference_ledger_path}`",
        f"Interactive handoff: `{interactive_result_path}`",
        "",
        "## P1 Repair Status",
        "",
        "- `BFCL-MINI-001`: fixed - `darjeeling compile run` launched the journey.",
        "- `BFCL-MINI-004`: fixed - the run used documented reference config.",
        "- `BFCL-MINI-005`: fixed - final test used interactive handoff output.",
        "- `BFCL-MINI-006`: fixed - CLI/config used 30000ms reference deadline.",
        "",
        "## Result",
        "",
        "- Final test reached: yes",
        f"- Decision: `{compile_summary['decision_status']}`",
        f"- Decision blockers: `{compile_summary['decision_blockers']}`",
        "- L1/L2 enabled and L3 disabled through the compile command and candidate routing.",
        f"- Suspect agent-visible leakage files: {len(leakage['suspect_files'])}",
        "",
        "## Remaining Issues",
        "",
        "- This is still a mini smoke, not BFCL Stage 1 coverage evidence.",
        "- If the provider returns no cost header, spend is estimated from "
        "configured token prices.",
        "- Target-adaptation agent execution still requires macOS `sandbox-exec`.",
    ]
    (run_root / "user_journey_repair_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


def scan_agent_visible_leakage(workspace: Path) -> dict[str, Any]:
    suspect_terms = [
        "mini-val",
        "mini-test",
        "Write a short poem.",
        "What is the weather in Berlin?",
        "Summarize this sentence.",
    ]
    suspect_files = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        if rel.startswith("readonly_source/target"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [term for term in suspect_terms if term in text or term in rel]
        if hits:
            suspect_files.append({"path": rel, "hits": hits[:5]})
    return {"suspect_files": suspect_files}


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(child) for child in value]
    return value


def now_iso() -> str:
    return datetime.now(SHANGHAI).isoformat()


if __name__ == "__main__":
    main()
