from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from darjeeling.errors import TargetDefinitionError, ValidationError
from darjeeling.model import (
    ContractModule,
    DataConfig,
    ReferenceContext,
    ReferenceModule,
    ReferenceResponse,
    RuntimeConfig,
    SourceRecord,
    TargetCheckOptions,
    TargetCheckReport,
    TargetDefinition,
    TargetDefinitionDraft,
    TargetRequirements,
    TargetRuntimeContract,
    TargetViewManifest,
    TelemetryPrivacyPolicy,
)
from darjeeling.util import (
    file_digest,
    read_json,
    stable_hash,
    stable_json,
    tree_digest,
    write_json,
)

_REQUIRED_CONTRACT_CALLABLES = [
    "validate_input",
    "validate_output",
    "is_correct",
    "normalize_input",
    "split_group",
    "slice_tags",
    "redact_for_trace",
    "bucket_runtime_metadata",
]

_ALLOWED_TARGET_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "hashlib",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "typing",
    "typing_extensions",
}
_BANNED_TARGET_CALLS = {"__import__", "compile", "eval", "exec", "open"}
_BANNED_TARGET_ATTRIBUTE_CALLS = {"__import__", "eval", "exec", "open"}
_ALLOWED_REFERENCE_SOURCES = {"gold", "human", "versioned_l4", "verified_l4", "user_feedback"}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TargetDefinitionError(f"missing required file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TargetDefinitionError(f"{path} must contain a mapping")
    return value


def _load_json_schema(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        raise TargetDefinitionError(f"missing required schema: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TargetDefinitionError(f"schema must be an object: {path}")
    _validate_json_schema_document(value, str(path))
    return value


def _resolve_target_file(target_path: Path, path_value: str, field_name: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise TargetDefinitionError(f"{field_name} path must be non-empty text")
    resolved_root = target_path.resolve()
    resolved = (resolved_root / path_value).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise TargetDefinitionError(f"{field_name} path escapes target root: {path_value}") from exc
    return resolved


def _target_relative_path(target_path: Path, resolved_file: Path) -> Path:
    resolved_root = target_path.resolve()
    resolved = resolved_file.resolve()
    try:
        return resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise TargetDefinitionError(f"target file path escapes target root: {resolved}") from exc


def _copy_target_view_file(
    target_path: Path,
    src: Path,
    output_dir: Path,
    included: list[str],
) -> None:
    resolved_root = target_path.resolve()
    resolved_src = src.resolve()
    try:
        resolved_src.relative_to(resolved_root)
    except ValueError as exc:
        raise TargetDefinitionError(f"target view file escapes target root: {src}") from exc
    try:
        rel = src.relative_to(resolved_root)
    except ValueError:
        rel = _target_relative_path(target_path, src)
    dst = output_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    included.append(rel.as_posix())


def _parse_requirements(value: dict[str, Any]) -> TargetRequirements:
    allowed = set(TargetRequirements.__dataclass_fields__)
    unknown = set(value) - allowed
    if unknown:
        raise TargetDefinitionError(f"unknown target requirement fields: {sorted(unknown)}")
    requirements = _normalize_requirement_types(TargetRequirements(**value))
    if not 0.0 <= requirements.precision_min <= 1.0:
        raise TargetDefinitionError("precision_min must be between 0 and 1")
    if requirements.wrong_accept_rate_max is not None and not (
        0.0 <= requirements.wrong_accept_rate_max <= 1.0
    ):
        raise TargetDefinitionError("wrong_accept_rate_max must be between 0 and 1")
    bounded_optional = [
        ("validation_test_precision_drop_max", requirements.validation_test_precision_drop_max),
        (
            "validation_test_coverage_retention_min",
            requirements.validation_test_coverage_retention_min,
        ),
        ("cohort_precision_floor_min", requirements.cohort_precision_floor_min),
        ("critical_slice_precision_min", requirements.critical_slice_precision_min),
        ("critical_slice_coverage_min", requirements.critical_slice_coverage_min),
        (
            "random_audit_reference_failure_rate_max",
            requirements.random_audit_reference_failure_rate_max,
        ),
    ]
    for name, item in bounded_optional:
        if item is not None and not 0.0 <= item <= 1.0:
            raise TargetDefinitionError(f"{name} must be between 0 and 1")
    if requirements.coverage_objective not in {"maximize", "hold", "none"}:
        raise TargetDefinitionError("coverage_objective must be maximize, hold, or none")
    if requirements.min_accepted_samples < 0 or requirements.min_slice_samples < 0:
        raise TargetDefinitionError("sample minimums must be non-negative")
    if (
        requirements.candidate_rank_stability_min_shards is not None
        and requirements.candidate_rank_stability_min_shards < 0
    ):
        raise TargetDefinitionError("candidate_rank_stability_min_shards must be non-negative")
    for name, item in [
        ("p95_latency_ms_max", requirements.p95_latency_ms_max),
        ("memory_mb_max", requirements.memory_mb_max),
    ]:
        if item is not None and item < 0:
            raise TargetDefinitionError(f"{name} must be non-negative")
    for name, item in [
        ("throughput_per_second_min", requirements.throughput_per_second_min),
        ("serving_cost_per_1000_max", requirements.serving_cost_per_1000_max),
    ]:
        if item is not None and item < 0:
            raise TargetDefinitionError(f"{name} must be non-negative")
    if not isinstance(requirements.future_audit_required_for_auto_release, bool):
        raise TargetDefinitionError("future_audit_required_for_auto_release must be boolean")
    if not isinstance(requirements.manual_approval_required, bool):
        raise TargetDefinitionError("manual_approval_required must be boolean")
    if not isinstance(requirements.critical_slices, list) or any(
        not isinstance(item, str) for item in requirements.critical_slices
    ):
        raise TargetDefinitionError("critical_slices must be list[str]")
    if not 0.0 <= requirements.random_audit_rate <= 1.0:
        raise TargetDefinitionError("random_audit_rate must be between 0 and 1")
    return requirements


def _require_float_requirement(name: str, value: Any, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TargetDefinitionError(f"{name} must be a number")
    return float(value)


def _require_int_requirement(name: str, value: Any, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetDefinitionError(f"{name} must be an integer")
    return value


def _normalize_requirement_types(requirements: TargetRequirements) -> TargetRequirements:
    if not isinstance(requirements.future_audit_required_for_auto_release, bool):
        raise TargetDefinitionError("future_audit_required_for_auto_release must be boolean")
    if not isinstance(requirements.manual_approval_required, bool):
        raise TargetDefinitionError("manual_approval_required must be boolean")
    if not isinstance(requirements.coverage_objective, str):
        raise TargetDefinitionError("coverage_objective must be text")
    if not isinstance(requirements.critical_slices, list) or any(
        not isinstance(item, str) for item in requirements.critical_slices
    ):
        raise TargetDefinitionError("critical_slices must be list[str]")
    return TargetRequirements(
        precision_min=_require_float_requirement("precision_min", requirements.precision_min),
        wrong_accept_rate_max=_require_float_requirement(
            "wrong_accept_rate_max", requirements.wrong_accept_rate_max, allow_none=True
        ),
        validation_test_precision_drop_max=_require_float_requirement(
            "validation_test_precision_drop_max",
            requirements.validation_test_precision_drop_max,
            allow_none=True,
        ),
        validation_test_coverage_retention_min=_require_float_requirement(
            "validation_test_coverage_retention_min",
            requirements.validation_test_coverage_retention_min,
            allow_none=True,
        ),
        cohort_precision_floor_min=_require_float_requirement(
            "cohort_precision_floor_min", requirements.cohort_precision_floor_min, allow_none=True
        ),
        candidate_rank_stability_min_shards=_require_int_requirement(
            "candidate_rank_stability_min_shards",
            requirements.candidate_rank_stability_min_shards,
            allow_none=True,
        ),
        future_audit_required_for_auto_release=requirements.future_audit_required_for_auto_release,
        min_accepted_samples=_require_int_requirement(
            "min_accepted_samples", requirements.min_accepted_samples
        ),
        min_slice_samples=_require_int_requirement(
            "min_slice_samples", requirements.min_slice_samples
        ),
        critical_slices=list(requirements.critical_slices),
        critical_slice_precision_min=_require_float_requirement(
            "critical_slice_precision_min",
            requirements.critical_slice_precision_min,
            allow_none=True,
        ),
        critical_slice_coverage_min=_require_float_requirement(
            "critical_slice_coverage_min", requirements.critical_slice_coverage_min, allow_none=True
        ),
        coverage_objective=requirements.coverage_objective,
        p95_latency_ms_max=_require_int_requirement(
            "p95_latency_ms_max", requirements.p95_latency_ms_max, allow_none=True
        ),
        memory_mb_max=_require_int_requirement(
            "memory_mb_max", requirements.memory_mb_max, allow_none=True
        ),
        throughput_per_second_min=_require_float_requirement(
            "throughput_per_second_min", requirements.throughput_per_second_min, allow_none=True
        ),
        serving_cost_per_1000_max=_require_float_requirement(
            "serving_cost_per_1000_max", requirements.serving_cost_per_1000_max, allow_none=True
        ),
        random_audit_rate=_require_float_requirement(
            "random_audit_rate", requirements.random_audit_rate
        ),
        random_audit_reference_failure_rate_max=_require_float_requirement(
            "random_audit_reference_failure_rate_max",
            requirements.random_audit_reference_failure_rate_max,
            allow_none=True,
        ),
        manual_approval_required=requirements.manual_approval_required,
    )


def _parse_privacy_policy(value: dict[str, Any]) -> TelemetryPrivacyPolicy:
    allowed = set(TelemetryPrivacyPolicy.__dataclass_fields__)
    unknown = set(value) - allowed
    if unknown:
        raise TargetDefinitionError(f"unknown telemetry privacy policy fields: {sorted(unknown)}")
    policy = TelemetryPrivacyPolicy(**value)
    allowed_sources = {"l4_fallback", "random_audit", "risk_audit", "user_feedback"}
    if not isinstance(policy.allowed_sources, list) or any(
        not isinstance(source, str) for source in policy.allowed_sources
    ):
        raise TargetDefinitionError("allowed_sources must be list[str]")
    if any(source not in allowed_sources for source in policy.allowed_sources):
        raise TargetDefinitionError("telemetry privacy policy contains an invalid source")
    if not isinstance(policy.default_approved_for_by_source, dict):
        raise TargetDefinitionError("default_approved_for_by_source must be a mapping")
    for roles in policy.default_approved_for_by_source.values():
        _validate_split_eligibility(roles)
    if not isinstance(policy.raw_payload_allowed, bool):
        raise TargetDefinitionError("raw_payload_allowed must be boolean")
    if not isinstance(policy.canonicalization_required, bool):
        raise TargetDefinitionError("canonicalization_required must be boolean")
    if not isinstance(policy.human_review_required_sources, list) or any(
        source not in allowed_sources for source in policy.human_review_required_sources
    ):
        raise TargetDefinitionError("human_review_required_sources must list valid sources")
    return policy


def _parse_data_config(value: dict[str, Any], target_path: Path) -> DataConfig:
    allowed = set(DataConfig.__dataclass_fields__)
    unknown = set(value) - allowed
    if unknown:
        raise TargetDefinitionError(f"unknown data config fields: {sorted(unknown)}")
    config = DataConfig(**value)
    _validate_split_eligibility(config.default_split_eligibility)
    for source in config.sources:
        source_unknown = set(source) - {
            "name",
            "records",
            "path",
            "split_eligibility",
        }
        if source_unknown:
            raise TargetDefinitionError(f"unknown data source fields: {sorted(source_unknown)}")
        if "split_eligibility" in source:
            _validate_split_eligibility(source["split_eligibility"])
        if "path" in source:
            _resolve_target_file(target_path, source["path"], "data source")
        for record in source.get("records", []):
            if "split_eligibility" in record:
                _validate_split_eligibility(record["split_eligibility"])
            _validate_reference_source(record.get("reference_source"))
    return config


def _validate_split_eligibility(value: list[str]) -> None:
    allowed = {"train", "validation_candidate", "test_candidate"}
    if not isinstance(value, list) or not value or any(role not in allowed for role in value):
        raise TargetDefinitionError("split eligibility must be a non-empty list of known roles")


def _validate_reference_source(value: Any) -> None:
    if value is None:
        return
    if value not in _ALLOWED_REFERENCE_SOURCES:
        raise TargetDefinitionError(
            f"reference_source must be one of {sorted(_ALLOWED_REFERENCE_SOURCES)}"
        )


def _validate_json_schema_document(schema: dict[str, Any], path: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except JsonSchemaSchemaError as exc:
        raise TargetDefinitionError(f"invalid JSON schema at {path}: {exc.message}") from exc


def load_target_definition(target_path: Path) -> TargetDefinitionDraft:
    target_path = target_path.resolve()
    target_yaml = _load_yaml(target_path / "target.yaml")
    data_yaml = _load_yaml(target_path / "data.yaml")
    schemas = target_yaml.get("schemas", {})
    input_schema_path = _resolve_target_file(
        target_path, schemas.get("input", "schemas/input.json"), "input schema"
    )
    output_schema_path = _resolve_target_file(
        target_path, schemas.get("output", "schemas/output.json"), "output schema"
    )
    contract_path = _resolve_target_file(
        target_path, target_yaml.get("contract", "contract.py"), "contract"
    )
    reference_value = target_yaml.get("reference")
    reference_path = (
        _resolve_target_file(target_path, reference_value, "reference")
        if reference_value
        else None
    )
    runtime_value = target_yaml.get("runtime", {})
    privacy_value = runtime_value.get("telemetry_privacy_policy", {})
    return TargetDefinitionDraft(
        name=str(target_yaml["name"]),
        version=str(target_yaml["version"]),
        target_path=target_path,
        input_schema=_load_json_schema(input_schema_path),
        output_schema=_load_json_schema(output_schema_path),
        input_schema_path=input_schema_path,
        output_schema_path=output_schema_path,
        contract_path=contract_path.resolve(),
        reference_path=reference_path.resolve() if reference_path else None,
        requirements=_parse_requirements(target_yaml.get("requirements", {})),
        data_config=_parse_data_config(data_yaml, target_path),
        runtime_config=RuntimeConfig(telemetry_privacy_policy=_parse_privacy_policy(privacy_value)),
        metadata={
            k: v for k, v in target_yaml.items() if k not in {"schemas", "requirements", "runtime"}
        },
    )


def _import_module(path: Path, name_hint: str) -> ModuleType:
    if not path.exists():
        raise TargetDefinitionError(f"missing module: {path}")
    _assert_restricted_import_safe(path)
    module_name = f"_darjeeling_target_{name_hint}_{stable_hash(str(path))[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TargetDefinitionError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - exact target error type is user-owned
        raise TargetDefinitionError(f"failed to import {path}: {exc}") from exc
    return module


def _assert_restricted_import_safe(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            disallowed = roots - _ALLOWED_TARGET_IMPORT_ROOTS
            if disallowed:
                raise TargetDefinitionError(
                    f"disallowed import in target module: {sorted(disallowed)}"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_TARGET_IMPORT_ROOTS:
                raise TargetDefinitionError(f"disallowed import in target module: {root}")
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in _BANNED_TARGET_CALLS:
                raise TargetDefinitionError(
                    f"disallowed call in target module: {function.id}"
                )
            if (
                isinstance(function, ast.Attribute)
                and function.attr in _BANNED_TARGET_ATTRIBUTE_CALLS
            ):
                raise TargetDefinitionError(
                    f"disallowed call in target module: {function.attr}"
                )
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(node, ast.Import):
            continue
        if isinstance(node, ast.ImportFrom):
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            if value is None:
                continue
            try:
                ast.literal_eval(value)
                continue
            except Exception:
                pass
        raise TargetDefinitionError(
            "target modules may contain definitions, safe imports, and constant declarations only"
        )


def load_contract_module(contract_path: Path) -> ContractModule:
    module = _import_module(contract_path.resolve(), "contract")
    missing = [
        name for name in _REQUIRED_CONTRACT_CALLABLES if not callable(getattr(module, name, None))
    ]
    if missing:
        raise TargetDefinitionError(f"contract missing required callables: {', '.join(missing)}")
    return ContractModule(
        path=contract_path.resolve(), module=module, digest=file_digest(contract_path)
    )


def load_reference_module(reference_path: Path | None) -> ReferenceModule | None:
    if reference_path is None:
        return None
    module = _import_module(reference_path.resolve(), "reference")
    missing = [
        name
        for name in ["build_reference_request", "parse_reference_response"]
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise TargetDefinitionError(f"reference adapter missing callables: {', '.join(missing)}")
    return ReferenceModule(
        path=reference_path.resolve(), module=module, digest=file_digest(reference_path)
    )


def _schema_error_location(error: JsonSchemaValidationError, label: str) -> str:
    location = label
    for part in error.absolute_path:
        if isinstance(part, int):
            location += f"[{part}]"
        else:
            location += f".{part}"
    return location


def _validate_with_schema(
    validator: Draft202012Validator, value: Any, label: str
) -> None:
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        error = errors[0]
        raise ValidationError(f"{_schema_error_location(error, label)}: {error.message}")


def build_runtime_contract(
    definition_draft: TargetDefinitionDraft,
    contract_module: ContractModule,
    reference_module: ReferenceModule | None,
    contract_hash: str | None = None,
) -> TargetRuntimeContract:
    module = contract_module.module
    reference = reference_module.module if reference_module else None
    input_validator = Draft202012Validator(definition_draft.input_schema)
    output_validator = Draft202012Validator(definition_draft.output_schema)

    def validate_input_fn(value: dict[str, Any]) -> dict[str, Any]:
        _validate_with_schema(input_validator, value, "input")
        try:
            result = module.validate_input(dict(value))
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"validate_input failed: {exc}") from exc
        if not isinstance(result, dict):
            raise ValidationError("validate_input must return a dict")
        _validate_with_schema(input_validator, result, "input")
        return result

    def validate_output_fn(value: dict[str, Any]) -> dict[str, Any]:
        _validate_with_schema(output_validator, value, "output")
        try:
            result = module.validate_output(dict(value))
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"validate_output failed: {exc}") from exc
        if not isinstance(result, dict):
            raise ValidationError("validate_output must return a dict")
        _validate_with_schema(output_validator, result, "output")
        return result

    build_reference = None
    parse_reference = None
    if reference is not None:
        build_reference = reference.build_reference_request
        parse_reference = reference.parse_reference_response

    return TargetRuntimeContract(
        input_schema=definition_draft.input_schema,
        output_schema=definition_draft.output_schema,
        validate_input_fn=validate_input_fn,
        validate_output_fn=validate_output_fn,
        is_correct_fn=module.is_correct,
        normalize_input_fn=module.normalize_input,
        split_group_fn=module.split_group,
        slice_tags_fn=module.slice_tags,
        redact_for_trace_fn=module.redact_for_trace,
        bucket_runtime_metadata_fn=module.bucket_runtime_metadata,
        build_reference_request_fn=build_reference,
        parse_reference_response_fn=parse_reference,
        contract_hash=contract_hash,
    )


def validate_input(contract: TargetRuntimeContract, value: dict[str, Any]) -> dict[str, Any]:
    return contract.validate_input(value)


def validate_output(contract: TargetRuntimeContract, value: dict[str, Any]) -> dict[str, Any]:
    return contract.validate_output(value)


def is_correct(
    contract: TargetRuntimeContract, output: dict[str, Any], reference: dict[str, Any]
) -> bool:
    return contract.is_correct(output, reference)


def normalize_input(contract: TargetRuntimeContract, input_value: dict[str, Any]) -> str:
    return contract.normalize_input(input_value)


def split_group(contract: TargetRuntimeContract, record: Any) -> str:
    return contract.split_group(record)


def slice_tags(contract: TargetRuntimeContract, record: Any) -> list[str]:
    return contract.slice_tags(record)


def redact_for_trace(contract: TargetRuntimeContract, value: dict[str, Any]) -> dict[str, Any]:
    return contract.redact_for_trace(value)


def bucket_runtime_metadata(
    contract: TargetRuntimeContract, metadata: dict[str, Any]
) -> dict[str, Any]:
    return contract.bucket_runtime_metadata(metadata)


def build_reference_request(
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    reference_context: ReferenceContext,
) -> dict[str, Any]:
    return contract.build_reference_request(input_value, reference_context)


def parse_reference_response(contract: TargetRuntimeContract, response: Any) -> dict[str, Any]:
    return contract.parse_reference_response(response)


def compute_contract_hash(definition: TargetDefinition) -> str:
    tests_path = definition.target_path / "tests"
    tests_digest = tree_digest(tests_path) if tests_path.exists() else None
    return stable_hash(
        {
            "name": definition.name,
            "version": definition.version,
            "input_schema": definition.input_schema,
            "output_schema": definition.output_schema,
            "contract_module_digest": definition.contract_module_digest,
            "reference_module_digest": definition.reference_module_digest,
            "requirements": definition.requirements,
            "data_config": definition.data_config,
            "metadata": definition.metadata,
            "reference_version": definition.reference_version,
            "privacy_policy": definition.runtime_config.telemetry_privacy_policy,
            "contract_tests_digest": tests_digest,
        }
    )


def materialize_definition(
    draft: TargetDefinitionDraft,
    contract_module: ContractModule,
    reference_module: ReferenceModule | None,
) -> TargetDefinition:
    provisional = TargetDefinition(
        name=draft.name,
        version=draft.version,
        target_path=draft.target_path,
        input_schema=draft.input_schema,
        output_schema=draft.output_schema,
        contract_module_digest=contract_module.digest,
        reference_module_digest=reference_module.digest if reference_module else None,
        requirements=draft.requirements,
        data_config=draft.data_config,
        runtime_config=draft.runtime_config,
        contract_hash="",
        metadata=draft.metadata,
        reference_version=draft.metadata.get("reference_version"),
    )
    contract_hash = compute_contract_hash(provisional)
    return TargetDefinition(**{**provisional.__dict__, "contract_hash": contract_hash})


def _iter_target_check_records(
    draft: TargetDefinitionDraft,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for source in draft.data_config.sources:
        for record in source.get("records", []):
            records.append((source, record))
        if source.get("path"):
            source_path = _resolve_target_file(draft.target_path, source["path"], "data source")
            loaded = read_json(source_path)
            if not isinstance(loaded, list):
                raise TargetDefinitionError("data source path must contain a list of records")
            for record in loaded:
                if not isinstance(record, dict):
                    raise TargetDefinitionError("data source path records must be objects")
                if "split_eligibility" in record:
                    _validate_split_eligibility(record["split_eligibility"])
                _validate_reference_source(record.get("reference_source"))
                records.append((source, record))
    return records


def check_target_definition(
    target_path: Path, check_options: TargetCheckOptions
) -> TargetCheckReport:
    failures: list[str] = []
    warnings: list[str] = []
    contract_hash: str | None = None
    target_name = target_path.name
    try:
        draft = load_target_definition(target_path)
        target_name = draft.name
        contract_module = load_contract_module(draft.contract_path)
        reference_module = load_reference_module(draft.reference_path)
        if check_options.require_reference and reference_module is None:
            failures.append("reference adapter is required")
        definition = materialize_definition(draft, contract_module, reference_module)
        contract_hash = definition.contract_hash
        contract = build_runtime_contract(
            draft, contract_module, reference_module, contract_hash=contract_hash
        )
        normalized_seen: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
        split_groups: dict[str, list[list[str]]] = {}
        for source, record in _iter_target_check_records(draft):
            source_split_eligibility = list(
                source.get("split_eligibility", draft.data_config.default_split_eligibility)
            )
            try:
                inp = contract.validate_input(record["input"])
                reference_output = None
                if record.get("reference_output") is not None:
                    reference_output = contract.validate_output(record["reference_output"])
                normalized = contract.normalize_input(inp)
                record_split_eligibility = list(
                    record.get("split_eligibility", source_split_eligibility)
                )
                source_record = SourceRecord(
                    record_id=str(record.get("record_id", normalized)),
                    input=inp,
                    reference_output=reference_output,
                    reference_source=record.get("reference_source"),
                    split_eligibility=record_split_eligibility,
                    source_name=str(source.get("name", "inline")),
                )
                group = contract.split_group(source_record)
                if not isinstance(group, str) or not group:
                    record_id = record.get("record_id", "<unknown>")
                    failures.append(f"record {record_id} split_group must be non-empty text")
                contract.slice_tags(source_record)
                split_groups.setdefault(group, []).append(record_split_eligibility)
                reference_identity = (
                    reference_output,
                    record.get("reference_source"),
                )
                previous = normalized_seen.get(normalized)
                if previous is not None and previous != reference_identity:
                    failures.append(
                        f"normalized input collision has conflicting references: {normalized}"
                    )
                normalized_seen[normalized] = reference_identity
                redacted = contract.redact_for_trace(inp)
                metadata_buckets = contract.bucket_runtime_metadata(record.get("metadata", {}))
                if not isinstance(redacted, dict) or len(stable_json(redacted)) > 2000:
                    record_id = record.get("record_id", "<unknown>")
                    failures.append(f"record {record_id} redaction output is invalid")
                if (
                    not isinstance(metadata_buckets, dict)
                    or len(stable_json(metadata_buckets)) > 2000
                ):
                    record_id = record.get("record_id", "<unknown>")
                    failures.append(f"record {record_id} metadata buckets are invalid")
            except Exception as exc:
                failures.append(f"record {record.get('record_id', '<unknown>')} failed: {exc}")
        for group, eligibilities in split_groups.items():
            common = set(eligibilities[0]) if eligibilities else set()
            for eligibility in eligibilities[1:]:
                common &= set(eligibility)
            if not common:
                failures.append(f"split group has no common split eligibility: {group}")
        if reference_module is not None:
            sample_input = None
            sample_reference = None
            for _, sample_record in _iter_target_check_records(draft):
                sample_input = contract.validate_input(sample_record["input"])
                if sample_record.get("reference_output") is not None:
                    sample_reference = contract.validate_output(sample_record["reference_output"])
                break
            if sample_input is not None:
                contract.build_reference_request(
                    sample_input, ReferenceContext(purpose="target_check")
                )
                if sample_reference is not None:
                    contract.parse_reference_response(ReferenceResponse(payload=sample_reference))
            else:
                warnings.append(
                    "reference adapter not exercised because no sample records were declared"
                )
        _run_contract_tests(draft, contract, failures)
    except Exception as exc:
        failures.append(str(exc))
    return TargetCheckReport(
        target_name=target_name,
        contract_hash=contract_hash,
        status="fail" if failures else "pass",
        failures=failures,
        warnings=warnings,
    )


def _run_contract_tests(
    draft: TargetDefinitionDraft,
    contract: TargetRuntimeContract,
    failures: list[str],
) -> None:
    tests_path = draft.target_path / "tests" / "contract_cases.yaml"
    if not tests_path.exists():
        return
    raw = yaml.safe_load(tests_path.read_text(encoding="utf-8")) or {}
    cases = raw.get("cases", [])
    if not isinstance(cases, list):
        failures.append("tests/contract_cases.yaml must contain a cases list")
        return
    for index, case in enumerate(cases):
        name = case.get("name", f"case-{index}")
        try:
            if "invalid_input" in case:
                try:
                    contract.validate_input(case["invalid_input"])
                except Exception:
                    continue
                failures.append(f"contract test {name} expected invalid_input to fail")
                continue
            if "invalid_output" in case:
                try:
                    contract.validate_output(case["invalid_output"])
                except Exception:
                    continue
                failures.append(f"contract test {name} expected invalid_output to fail")
                continue
            input_value = contract.validate_input(case["input"])
            output = contract.validate_output(case["output"])
            expected_normalized = case.get("normalized_input")
            if (
                expected_normalized is not None
                and contract.normalize_input(input_value) != expected_normalized
            ):
                failures.append(f"contract test {name} normalized input mismatch")
            if "reference_output" in case:
                reference = contract.validate_output(case["reference_output"])
                expected_correct = bool(case.get("correct", True))
                if contract.is_correct(output, reference) is not expected_correct:
                    failures.append(f"contract test {name} correctness mismatch")
        except Exception as exc:
            failures.append(f"contract test {name} failed: {exc}")


def load_checked_target(
    target_path: Path,
) -> tuple[TargetDefinition, TargetRuntimeContract, TargetCheckReport]:
    draft = load_target_definition(target_path)
    contract_module = load_contract_module(draft.contract_path)
    reference_module = load_reference_module(draft.reference_path)
    definition = materialize_definition(draft, contract_module, reference_module)
    contract = build_runtime_contract(
        draft, contract_module, reference_module, contract_hash=definition.contract_hash
    )
    report = check_target_definition(target_path, TargetCheckOptions(require_reference=False))
    if report.status != "pass":
        raise TargetDefinitionError("; ".join(report.failures))
    return definition, contract, report


def export_agent_readonly_target_view(
    definition: TargetDefinition, output_dir: Path
) -> TargetViewManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    draft = load_target_definition(definition.target_path)
    contract_module = load_contract_module(draft.contract_path)
    reference_module = load_reference_module(draft.reference_path)
    current_definition = materialize_definition(draft, contract_module, reference_module)
    if current_definition.contract_hash != definition.contract_hash:
        raise TargetDefinitionError("target files changed since definition was materialized")
    files = [
        definition.target_path / "target.yaml",
        draft.contract_path,
        draft.input_schema_path,
        draft.output_schema_path,
    ]
    if draft.reference_path is not None:
        files.append(draft.reference_path)
    included: list[str] = []
    for src in files:
        _copy_target_view_file(definition.target_path, src, output_dir, included)
    contract_cases = definition.target_path / "tests" / "contract_cases.yaml"
    if contract_cases.exists():
        _copy_target_view_file(definition.target_path, contract_cases, output_dir, included)
    write_json(
        output_dir / "target_manifest.json",
        {
            "target_name": definition.name,
            "contract_hash": definition.contract_hash,
            "included_files": included,
        },
    )
    write_json(
        output_dir / "data_policy.json",
        {
            "source_names": [source.get("name") for source in definition.data_config.sources],
            "default_split_eligibility": definition.data_config.default_split_eligibility,
            "rows_included": False,
        },
    )
    return TargetViewManifest(
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        view_path=output_dir,
        included_files=[*included, "data_policy.json"],
    )
