from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from darjeeling.errors import ValidationError
from darjeeling.target_definition import (
    TargetCheckOptions,
    check_target_definition,
    export_agent_readonly_target_view,
    load_checked_target,
    load_contract_module,
    load_reference_module,
    load_target_definition,
    materialize_definition,
)


def test_contract_hash_changes_when_contract_source_changes(target_dir: Path) -> None:
    draft = load_target_definition(target_dir)
    first = materialize_definition(
        draft,
        load_contract_module(draft.contract_path),
        load_reference_module(draft.reference_path),
    )
    (target_dir / "contract.py").write_text(
        (target_dir / "contract.py").read_text() + "\n# semantic owner changed\n"
    )
    changed_draft = load_target_definition(target_dir)
    second = materialize_definition(
        changed_draft,
        load_contract_module(changed_draft.contract_path),
        load_reference_module(changed_draft.reference_path),
    )
    assert first.contract_hash != second.contract_hash


def test_contract_hash_changes_when_data_config_or_contract_tests_change(
    target_dir: Path,
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    data = yaml.safe_load((target_dir / "data.yaml").read_text())
    data["default_split_eligibility"] = ["train", "validation_candidate"]
    (target_dir / "data.yaml").write_text(yaml.safe_dump(data))
    data_changed, _, _ = load_checked_target(target_dir)
    assert data_changed.contract_hash != definition.contract_hash
    tests = yaml.safe_load((target_dir / "tests" / "contract_cases.yaml").read_text())
    tests["cases"].append(
        {
            "name": "new_case",
            "input": {"text": "b:new"},
            "output": {"label": "b"},
            "reference_output": {"label": "b"},
            "correct": True,
        }
    )
    (target_dir / "tests" / "contract_cases.yaml").write_text(yaml.safe_dump(tests))
    tests_changed, _, _ = load_checked_target(target_dir)
    assert tests_changed.contract_hash != data_changed.contract_hash
    target = yaml.safe_load((target_dir / "target.yaml").read_text())
    target["reference_version"] = "reference-v2"
    (target_dir / "target.yaml").write_text(yaml.safe_dump(target))
    metadata_changed, _, _ = load_checked_target(target_dir)
    assert metadata_changed.contract_hash != tests_changed.contract_hash


def test_invalid_target_definition_fails_early(target_dir: Path) -> None:
    text = (
        (target_dir / "contract.py")
        .read_text()
        .replace("def normalize_input", "def missing_normalize_input")
    )
    (target_dir / "contract.py").write_text(text)
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("normalize_input" in failure for failure in report.failures)


def test_standard_json_schema_keywords_are_accepted_and_enforced(target_dir: Path) -> None:
    import json

    schema = json.loads((target_dir / "schemas" / "input.json").read_text())
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["description"] = "Standard JSON Schema annotations are allowed."
    schema["properties"]["text"] = {
        "description": "Prefix-coded input",
        "oneOf": [{"type": "string", "pattern": "^[ab]:"}],
    }
    (target_dir / "schemas" / "input.json").write_text(json.dumps(schema))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "pass"

    data = yaml.safe_load((target_dir / "data.yaml").read_text())
    data["sources"][0]["records"][0]["input"]["text"] = "c:bad"
    (target_dir / "data.yaml").write_text(yaml.safe_dump(data))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any(
        "input.text" in failure and "not valid under any" in failure
        for failure in report.failures
    )


def test_invalid_json_schema_document_fails_target_loading(target_dir: Path) -> None:
    import json

    schema = json.loads((target_dir / "schemas" / "input.json").read_text())
    schema["properties"] = []
    (target_dir / "schemas" / "input.json").write_text(json.dumps(schema))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("invalid JSON schema" in failure for failure in report.failures)


def test_invalid_config_fails_target_loading(target_dir: Path) -> None:
    target = yaml.safe_load((target_dir / "target.yaml").read_text())
    target["requirements"]["precision_mni"] = 0.9
    (target_dir / "target.yaml").write_text(yaml.safe_dump(target))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("unknown target requirement" in failure for failure in report.failures)


def test_target_file_paths_must_stay_under_target_root(target_dir: Path) -> None:
    target = yaml.safe_load((target_dir / "target.yaml").read_text())
    target["schemas"]["input"] = "../input.json"
    (target_dir / "target.yaml").write_text(yaml.safe_dump(target))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("path escapes target root" in failure for failure in report.failures)


def test_unknown_inline_reference_source_fails_target_check(target_dir: Path) -> None:
    data = yaml.safe_load((target_dir / "data.yaml").read_text())
    data["sources"][0]["records"][0]["reference_source"] = "teacher"
    (target_dir / "data.yaml").write_text(yaml.safe_dump(data))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("reference_source must be one of" in failure for failure in report.failures)


def test_unknown_path_loaded_reference_source_fails_target_check(target_dir: Path) -> None:
    import json

    data = yaml.safe_load((target_dir / "data.yaml").read_text())
    record = dict(data["sources"][0]["records"][0])
    record["reference_source"] = "teacher"
    (target_dir / "records.json").write_text(json.dumps([record]))
    data["sources"] = [{"name": "file-backed", "path": "records.json"}]
    (target_dir / "data.yaml").write_text(yaml.safe_dump(data))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("reference_source must be one of" in failure for failure in report.failures)


def test_invalid_requirement_types_fail_target_loading(target_dir: Path) -> None:
    target = yaml.safe_load((target_dir / "target.yaml").read_text())
    target["requirements"]["precision_min"] = True
    (target_dir / "target.yaml").write_text(yaml.safe_dump(target))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("precision_min must be a number" in failure for failure in report.failures)

    target["requirements"]["precision_min"] = 0.9
    target["requirements"]["min_accepted_samples"] = 1.5
    (target_dir / "target.yaml").write_text(yaml.safe_dump(target))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("min_accepted_samples must be an integer" in failure for failure in report.failures)


def test_target_owned_contract_tests_fail_target_check(target_dir: Path) -> None:
    tests = yaml.safe_load((target_dir / "tests" / "contract_cases.yaml").read_text())
    tests["cases"][0]["normalized_input"] = "wrong"
    (target_dir / "tests" / "contract_cases.yaml").write_text(yaml.safe_dump(tests))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("contract test valid_prefix" in failure for failure in report.failures)


def test_target_callable_return_types_are_enforced(target_dir: Path) -> None:
    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        contract_text.replace("return output == reference", "return 'yes'")
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("is_correct must return bool" in failure for failure in report.failures)


def test_target_check_rejects_invalid_slice_tags_return_type(target_dir: Path) -> None:
    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        contract_text.replace(
            'return [record.input["text"].split(":", 1)[0]]',
            'return "not-a-list"',
        )
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("slice_tags must return list[str]" in failure for failure in report.failures)


def test_target_check_honors_source_level_split_eligibility(target_dir: Path) -> None:
    data = yaml.safe_load((target_dir / "data.yaml").read_text())
    base_record = {
        "input": {"text": "a:shared", "opaque_target_field": "kept-opaque"},
        "reference_output": {"label": "a"},
        "reference_source": "gold",
    }
    data["sources"] = [
        {
            "name": "train-only",
            "split_eligibility": ["train"],
            "records": [{**base_record, "record_id": "train-r"}],
        },
        {
            "name": "validation-only",
            "split_eligibility": ["validation_candidate"],
            "records": [{**base_record, "record_id": "validation-r"}],
        },
    ]
    (target_dir / "data.yaml").write_text(yaml.safe_dump(data))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any(
        "split group has no common split eligibility" in failure for failure in report.failures
    )


def test_reference_request_adapter_must_return_dict(target_dir: Path) -> None:
    reference_text = (target_dir / "reference.py").read_text()
    (target_dir / "reference.py").write_text(
        reference_text.replace(
            'return {"input": input_value, "purpose": reference_context.purpose}',
            'return "not-a-dict"',
        )
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("build_reference_request must return dict" in failure for failure in report.failures)


def test_agent_readonly_target_view_exports_resolved_target_paths(
    target_dir: Path,
    tmp_path: Path,
) -> None:
    custom = target_dir / "custom"
    custom.mkdir()
    (custom / "contract_custom.py").write_text((target_dir / "contract.py").read_text())
    (custom / "reference_custom.py").write_text((target_dir / "reference.py").read_text())
    (custom / "input_custom.json").write_text((target_dir / "schemas" / "input.json").read_text())
    (custom / "output_custom.json").write_text(
        (target_dir / "schemas" / "output.json").read_text()
    )
    target = yaml.safe_load((target_dir / "target.yaml").read_text())
    target["schemas"] = {
        "input": "custom/input_custom.json",
        "output": "custom/output_custom.json",
    }
    target["contract"] = "custom/contract_custom.py"
    target["reference"] = "custom/reference_custom.py"
    (target_dir / "target.yaml").write_text(yaml.safe_dump(target))

    definition, _, report = load_checked_target(target_dir)
    assert report.status == "pass"
    manifest = export_agent_readonly_target_view(definition, tmp_path / "target-view")

    expected = {
        "target.yaml",
        "custom/contract_custom.py",
        "custom/reference_custom.py",
        "custom/input_custom.json",
        "custom/output_custom.json",
        "tests/contract_cases.yaml",
        "data_policy.json",
    }
    assert expected.issubset(set(manifest.included_files))
    for rel in expected:
        assert (manifest.view_path / rel).exists()
    assert not (manifest.view_path / "contract.py").exists()
    assert not (manifest.view_path / "schemas" / "input.json").exists()


def test_agent_readonly_target_view_excludes_disallowed_test_files(
    target_dir: Path,
    tmp_path: Path,
) -> None:
    (target_dir / "tests" / "validation_rows.yaml").write_text("rows: []\n")
    (target_dir / "tests" / "production_secret.txt").write_text("secret\n")
    outside_secret = tmp_path / "outside-secret.txt"
    outside_secret.write_text("secret\n")
    (target_dir / "tests" / "escaping_secret_link.txt").symlink_to(outside_secret)

    definition, _, report = load_checked_target(target_dir)
    assert report.status == "pass"
    manifest = export_agent_readonly_target_view(definition, tmp_path / "target-view")

    assert "tests/contract_cases.yaml" in manifest.included_files
    assert "tests/validation_rows.yaml" not in manifest.included_files
    assert "tests/production_secret.txt" not in manifest.included_files
    assert "tests/escaping_secret_link.txt" not in manifest.included_files
    assert not (manifest.view_path / "tests" / "validation_rows.yaml").exists()
    assert not (manifest.view_path / "tests" / "production_secret.txt").exists()
    assert not (manifest.view_path / "tests" / "escaping_secret_link.txt").exists()


def test_runtime_contract_validation_errors_are_structured(target_dir: Path) -> None:
    definition, contract, report = load_checked_target(target_dir)
    assert report.status == "pass"
    assert definition.name == "prefix"
    with pytest.raises(ValidationError, match="input.text"):
        contract.validate_input({"text": 10})

    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        contract_text.replace(
            "def validate_input(value):\n    return dict(value)",
            """def validate_input(value):
    if value["text"] == "bad-target":
        raise RuntimeError("target input invariant")
    return dict(value)""",
        ).replace(
            "def validate_output(value):\n    return dict(value)",
            """def validate_output(value):
    if value["label"] == "bad-target":
        raise RuntimeError("target output invariant")
    return dict(value)""",
        )
    )
    _, contract, report = load_checked_target(target_dir)
    assert report.status == "pass"
    with pytest.raises(ValidationError, match="validate_input failed"):
        contract.validate_input({"text": "bad-target"})
    with pytest.raises(ValidationError, match="validate_output failed"):
        contract.validate_output({"label": "bad-target"})


def test_unbounded_redaction_and_metadata_buckets_fail_target_check(target_dir: Path) -> None:
    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        contract_text.replace(
            'result["text"] = "<redacted>"',
            'result["text"] = "x" * 5000',
        )
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any(
        "redact_for_trace" in failure or "redaction output" in failure
        for failure in report.failures
    )


def test_import_time_side_effects_are_rejected(target_dir: Path) -> None:
    (target_dir / "contract.py").write_text(
        "open('side-effect.txt', 'w').write('bad')\n" + (target_dir / "contract.py").read_text()
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any(
        "safe imports" in failure
        or "constant declarations" in failure
        or "disallowed call" in failure
        for failure in report.failures
    )


def test_banned_imports_inside_target_callables_are_rejected(target_dir: Path) -> None:
    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        contract_text.replace(
            "def validate_input(value):\n    return dict(value)",
            "def validate_input(value):\n    import urllib.request\n    return dict(value)",
        )
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("disallowed import" in failure for failure in report.failures)


def test_banned_imports_inside_reference_callables_are_rejected(target_dir: Path) -> None:
    reference_text = (target_dir / "reference.py").read_text()
    (target_dir / "reference.py").write_text(
        reference_text.replace(
            "def build_reference_request(input_value, reference_context):",
            "def build_reference_request(input_value, reference_context):\n    import requests",
        )
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("disallowed import" in failure for failure in report.failures)


def test_filesystem_calls_inside_target_callables_are_rejected(target_dir: Path) -> None:
    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        contract_text.replace(
            "def validate_input(value):\n    return dict(value)",
            "def validate_input(value):\n    open('secret.txt').read()\n    return dict(value)",
        )
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("disallowed call" in failure for failure in report.failures)


def test_hashlib_redaction_is_allowed(target_dir: Path) -> None:
    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        "import hashlib\n"
        + contract_text.replace(
            'result["text"] = "<redacted>"',
            'result["text_hash"] = hashlib.sha256(result.pop("text").encode()).hexdigest()',
        )
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "pass"


def test_re_compile_inside_target_validation_is_allowed(target_dir: Path) -> None:
    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        contract_text.replace(
            "def validate_input(value):\n    return dict(value)",
            """def validate_input(value):
    import re
    if not re.compile(r"^[ab]:").match(value["text"]):
        raise ValueError("bad prefix")
    return dict(value)""",
        )
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "pass"


def test_nested_import_time_side_effects_are_rejected(target_dir: Path) -> None:
    (target_dir / "contract.py").write_text(
        "X = {'v': open('side-effect.txt', 'w').write('bad')}\n"
        + (target_dir / "contract.py").read_text()
    )
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any(
        "safe imports" in failure
        or "constant declarations" in failure
        or "disallowed call" in failure
        for failure in report.failures
    )


def test_target_check_catches_normalization_collisions(target_dir: Path) -> None:
    contract_text = (target_dir / "contract.py").read_text()
    (target_dir / "contract.py").write_text(
        contract_text.replace('return input_value["text"].lower()', 'return "constant"')
    )
    data = yaml.safe_load((target_dir / "data.yaml").read_text())
    data["sources"][0]["records"][1]["reference_output"] = {"label": "different"}
    (target_dir / "data.yaml").write_text(yaml.safe_dump(data))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("normalized input collision" in failure for failure in report.failures)


def test_target_check_catches_identical_input_with_conflicting_reference(
    target_dir: Path,
) -> None:
    data = yaml.safe_load((target_dir / "data.yaml").read_text())
    first = data["sources"][0]["records"][0]
    second = data["sources"][0]["records"][1]
    second["input"] = dict(first["input"])
    second["reference_output"] = {"label": "different"}
    (target_dir / "data.yaml").write_text(yaml.safe_dump(data))
    report = check_target_definition(target_dir, TargetCheckOptions())
    assert report.status == "fail"
    assert any("normalized input collision" in failure for failure in report.failures)


def test_target_specific_fields_remain_opaque_to_core(target_dir: Path) -> None:
    definition, contract, report = load_checked_target(target_dir)
    value = contract.validate_input({"text": "a:hello", "opaque_target_field": "intent-like"})
    assert report.status == "pass"
    assert definition.name == "prefix"
    assert contract.normalize_input(value) == "a:hello"
    assert contract.bucket_runtime_metadata({"tenant": "tenant-123"}) == {"tenant_bucket": "tenant"}
