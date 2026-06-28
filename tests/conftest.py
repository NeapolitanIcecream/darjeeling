from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from darjeeling.model import ReferenceContext, ReferenceResponse


class PrefixBroker:
    reference_version = "prefix-v1"

    def __init__(self, fail: bool = False):
        self.fail = fail

    def call(self, request: dict[str, Any], context: ReferenceContext) -> ReferenceResponse:
        if self.fail:
            raise RuntimeError("provider secret raw failure text")
        text = request["input"]["text"]
        return ReferenceResponse(
            payload={"label": text.split(":", 1)[0]},
            reference_source="versioned_l4",
            reference_version=self.reference_version,
            cost=0.01,
            latency_ms=3.0,
        )


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


@pytest.fixture
def target_dir(tmp_path: Path, now: datetime) -> Path:
    root = tmp_path / "target"
    (root / "schemas").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "schemas" / "input.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "opaque_target_field": {"type": "string"},
                },
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
                "name": "prefix",
                "version": "1",
                "schemas": {"input": "schemas/input.json", "output": "schemas/output.json"},
                "contract": "contract.py",
                "reference": "reference.py",
                "requirements": {
                    "precision_min": 0.9,
                    "min_accepted_samples": 1,
                    "wrong_accept_rate_max": 0.05,
                    "random_audit_reference_failure_rate_max": 0.2,
                },
                "runtime": {
                    "telemetry_privacy_policy": {
                        "policy_version": "p1",
                        "allowed_sources": [
                            "l4_fallback",
                            "random_audit",
                            "risk_audit",
                            "user_feedback",
                        ],
                        "default_approved_for_by_source": {
                            "l4_fallback": ["train"],
                            "random_audit": ["validation_candidate"],
                            "risk_audit": ["train"],
                            "user_feedback": ["train", "validation_candidate"],
                        },
                        "raw_payload_allowed": True,
                        "canonicalization_required": False,
                        "human_review_required_sources": [],
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
                "record_id": f"r{index}",
                "input": {"text": f"{label}:sample-{index}", "opaque_target_field": "kept-opaque"},
                "reference_output": {"label": label},
                "reference_source": "gold" if index < 2 else "versioned_l4",
                "split_eligibility": role,
                "source_timestamp": (now - timedelta(days=1)).isoformat(),
                "metadata": {"tenant": f"tenant-{index}"},
            }
        )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "default_split_eligibility": ["train"],
                "sources": [{"name": "inline", "records": rows}],
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
    return {"tenant_bucket": metadata.get("tenant", "none").split("-", 1)[0]}
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
                    {
                        "name": "invalid_input",
                        "invalid_input": {"opaque_target_field": "missing text"},
                    },
                    {"name": "invalid_output", "invalid_output": {"wrong": "shape"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    return root


def write_artifact(
    path: Path,
    contract_hash: str,
    accept_prefixes: list[str] | None = None,
    bad_output: bool = False,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    prefixes = accept_prefixes or ["a"]
    (path / "worker.py").write_text(
        f"""
import json
import sys

request = json.loads(sys.stdin.readline())
text = request["input"]["text"]
prefixes = {prefixes!r}
if any(text.startswith(prefix + ":") for prefix in prefixes):
    output = {{"bad": "shape"}} if {bad_output!r} else {{"label": text.split(":", 1)[0]}}
    print(json.dumps({{
        "decision": "accept",
        "output": output,
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
    return path
