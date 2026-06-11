from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.runtime.timing import elapsed_ms
from darjeeling.schemas import Frame, LayerResult

LocalSLMMode = Literal["disabled", "shadow", "guarded"]
LocalSLMDevicePolicy = Literal["auto", "cpu", "mps", "cuda"]

DEFAULT_L3_BENCHMARK_UTTERANCES = (
    "alpha request",
    "beta request",
    "gamma request",
)


class LocalSLMError(RuntimeError):
    pass


class LocalSLMLoadError(LocalSLMError):
    pass


class LocalSLMNotLoadedError(LocalSLMLoadError):
    pass


class LocalSLMGenerationError(LocalSLMError):
    pass


class L3ParseError(LocalSLMError):
    pass


class LocalSLMConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    mode: LocalSLMMode = "disabled"
    device_policy: LocalSLMDevicePolicy = "auto"
    max_new_tokens: int = 256
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    prompt_version: str = "l3-prompt-v1"


class L3PromptArtifact(BaseModel):
    prompt_version: str = "l3-prompt-v1"
    system_prompt: str = (
        "You are Darjeeling L3, a local virtual-assistant NLU model. Return one JSON object only."
    )
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    few_shot_examples: list[dict[str, Any]] = Field(default_factory=list)

    def render(self, utterance: str, task_schema: TaskSchema) -> str:
        return "\n".join(
            [
                self.system_prompt,
                "Output schema:",
                (
                    '{"intent": "intent_name", "slots": {"slot_name": "slot value"}, '
                    '"is_abstain": false, "confidence": 0.0}'
                ),
                "Use only these intents:",
                json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
                "Use only these slots:",
                json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
                "Few-shot examples:",
                json.dumps(self.few_shot_examples, ensure_ascii=False, sort_keys=True),
                "Current utterance:",
                json.dumps({"utterance": utterance}, ensure_ascii=False, sort_keys=True),
                f"Prompt version: {self.prompt_version}.",
            ]
        )


class L3RawOutput(BaseModel):
    intent: str
    slots: dict[str, str] = Field(default_factory=dict)
    is_abstain: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class L3ParsedOutput(BaseModel):
    frame: Frame
    confidence: float
    repair_used: bool = False


class LocalSLMBackend(Protocol):
    def generate(self, prompt: str, config: LocalSLMConfig) -> str:
        pass

    def status(self) -> dict[str, Any]:
        pass


@dataclass
class TransformersLocalSLMBackend:
    model_name: str
    device_policy: LocalSLMDevicePolicy = "auto"
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _model: Any | None = field(default=None, init=False, repr=False)
    _actual_device: str = field(default="not-loaded", init=False)
    _load_time_ms: float | None = field(default=None, init=False)

    def generate(self, prompt: str, config: LocalSLMConfig) -> str:
        self._ensure_loaded()
        if self._tokenizer is None or self._model is None:
            raise LocalSLMNotLoadedError("local SLM backend is not loaded")

        try:
            import torch

            prompt_text = self._format_prompt(prompt)
            inputs = self._tokenizer(prompt_text, return_tensors="pt")
            input_device = self._input_device()
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            input_length = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0][input_length:]
            return str(self._tokenizer.decode(generated_ids, skip_special_tokens=True))
        except LocalSLMError:
            raise
        except Exception as exc:  # pragma: no cover - hardware/model dependent
            raise LocalSLMGenerationError(str(exc)) from exc

    def status(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device_policy": self.device_policy,
            "actual_device": self._actual_device,
            "load_time_ms": self._load_time_ms,
            "loaded": self._model is not None,
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        started_at = perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            load_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
            if self.device_policy == "auto":
                load_kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            if self.device_policy != "auto":
                device = self._select_explicit_device(torch)
                self._model.to(device)
                self._actual_device = str(device)
            else:
                self._actual_device = str(self._input_device())
            self._model.eval()
            self._load_time_ms = (perf_counter() - started_at) * 1000.0
        except LocalSLMError:
            raise
        except Exception as exc:  # pragma: no cover - hardware/model dependent
            raise LocalSLMLoadError(str(exc)) from exc

    def _select_explicit_device(self, torch: Any):
        if self.device_policy == "cpu":
            return torch.device("cpu")
        if self.device_policy == "cuda":
            if not torch.cuda.is_available():
                raise LocalSLMLoadError("CUDA device requested but unavailable")
            return torch.device("cuda")
        if self.device_policy == "mps":
            if not torch.backends.mps.is_available():
                raise LocalSLMLoadError("MPS device requested but unavailable")
            return torch.device("mps")
        raise LocalSLMLoadError(f"unsupported device policy: {self.device_policy}")

    def _input_device(self):
        if self._model is None:
            raise LocalSLMNotLoadedError("local SLM model is not loaded")
        try:
            return next(self._model.parameters()).device
        except StopIteration as exc:  # pragma: no cover - impossible for real models
            raise LocalSLMLoadError("local SLM model has no parameters") from exc

    def _format_prompt(self, prompt: str) -> str:
        if self._tokenizer is None:
            raise LocalSLMNotLoadedError("local SLM tokenizer is not loaded")
        if hasattr(self._tokenizer, "apply_chat_template"):
            return str(
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return prompt


class L3LocalSLMLayer:
    def __init__(
        self,
        *,
        config: LocalSLMConfig,
        task_schema: TaskSchema,
        prompt_artifact: L3PromptArtifact | None = None,
        backend: LocalSLMBackend | None = None,
    ) -> None:
        self.config = config
        self.task_schema = task_schema
        self.prompt_artifact = prompt_artifact or L3PromptArtifact(
            prompt_version=config.prompt_version
        )
        self.backend = backend or TransformersLocalSLMBackend(
            model_name=config.model_name,
            device_policy=config.device_policy,
        )

    def try_answer(self, utterance: str) -> LayerResult:
        if self.config.mode == "disabled":
            return self._disabled_result("local SLM disabled")

        with elapsed_ms() as ms:
            prompt = self.prompt_artifact.render(utterance, self.task_schema)
            try:
                raw_output = self.backend.generate(prompt, self.config)
            except LocalSLMError as exc:
                if self.config.mode == "shadow":
                    return LayerResult(
                        layer="L3",
                        accepted=False,
                        reason="shadow local SLM failed; degraded to disabled",
                        latency_ms=ms(),
                        metadata=self._metadata(
                            requested_mode="shadow",
                            actual_mode="disabled",
                            error=str(exc),
                        ),
                    )
                raise

            try:
                parsed = parse_l3_output(raw_output)
                validation_errors = validate_l3_output(parsed, self.task_schema)
            except L3ParseError as exc:
                return LayerResult(
                    layer="L3",
                    accepted=False,
                    reason="local SLM parse failed",
                    latency_ms=ms(),
                    metadata=self._metadata(
                        requested_mode=self.config.mode,
                        actual_mode=self.config.mode,
                        raw_output=raw_output,
                        error=str(exc),
                    ),
                )

            threshold = self.prompt_artifact.confidence_threshold
            if threshold is None:
                threshold = self.config.confidence_threshold
            would_accept = (
                not parsed.frame.is_abstain
                and not validation_errors
                and parsed.confidence >= threshold
            )
            accepted = self.config.mode == "guarded" and would_accept
            reason = _l3_reason(
                mode=self.config.mode,
                would_accept=would_accept,
                validation_errors=validation_errors,
            )
            return LayerResult(
                layer="L3",
                accepted=accepted,
                frame=parsed.frame if accepted else None,
                confidence=parsed.confidence,
                reason=reason,
                latency_ms=ms(),
                metadata=self._metadata(
                    requested_mode=self.config.mode,
                    actual_mode=self.config.mode,
                    raw_output=raw_output,
                    shadow_frame=parsed.frame.model_dump(mode="json"),
                    repair_used=parsed.repair_used,
                    confidence=parsed.confidence,
                    confidence_threshold=threshold,
                    would_accept=would_accept,
                    validation_errors=validation_errors,
                ),
            )

    def _disabled_result(self, reason: str) -> LayerResult:
        return LayerResult(
            layer="L3",
            accepted=False,
            reason=reason,
            latency_ms=0.0,
            metadata=self._metadata(
                requested_mode="disabled",
                actual_mode="disabled",
                load_attempted=False,
            ),
        )

    def _metadata(self, **extra: Any) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "prompt_version": self.prompt_artifact.prompt_version,
            "device_policy": self.config.device_policy,
            "backend": self.backend.status(),
            **extra,
        }


def parse_l3_output(raw_output: str) -> L3ParsedOutput:
    candidate = _json_candidate(raw_output)
    repair_used = False
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            payload = repair_json(candidate, ensure_ascii=False, return_objects=True)
            repair_used = True
        except Exception as exc:
            raise L3ParseError(str(exc)) from exc

    if not isinstance(payload, dict):
        raise L3ParseError("local SLM output must be a JSON object")
    try:
        parsed = L3RawOutput.model_validate(payload)
    except Exception as exc:
        raise L3ParseError(str(exc)) from exc
    return L3ParsedOutput(
        frame=Frame(
            intent=parsed.intent,
            slots=parsed.slots,
            is_abstain=parsed.is_abstain,
        ),
        confidence=parsed.confidence,
        repair_used=repair_used,
    )


def validate_l3_output(parsed: L3ParsedOutput, task_schema: TaskSchema) -> list[str]:
    errors: list[str] = []
    if parsed.frame.intent not in task_schema.intent_names:
        errors.append(f"intent not allowed: {parsed.frame.intent}")
    invalid_slots = sorted(set(parsed.frame.slots) - set(task_schema.slot_names))
    if invalid_slots:
        errors.append(f"slots not allowed: {invalid_slots}")
    return errors


def build_l3_layer_from_settings(
    *,
    settings: Any,
    task_schema: TaskSchema,
    backend: LocalSLMBackend | None = None,
    prompt_artifact: L3PromptArtifact | None = None,
) -> L3LocalSLMLayer:
    return L3LocalSLMLayer(
        config=LocalSLMConfig(
            model_name=settings.local_slm_model,
            mode=settings.local_slm_mode,
            device_policy=settings.local_slm_device_policy,
            max_new_tokens=settings.local_slm_max_new_tokens,
            confidence_threshold=settings.local_slm_confidence_threshold,
            prompt_version=settings.local_slm_prompt_version,
        ),
        task_schema=task_schema,
        prompt_artifact=prompt_artifact,
        backend=backend,
    )


def benchmark_l3_layer(
    layer: L3LocalSLMLayer,
    utterances: Iterable[str],
) -> dict[str, Any]:
    request_results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    confidences: list[float] = []
    accepted = 0
    would_accept = 0
    failures = 0
    parse_failures = 0
    repair_count = 0
    total = 0
    started_at = perf_counter()

    for utterance in utterances:
        total += 1
        try:
            result = layer.try_answer(utterance)
        except LocalSLMError as exc:
            failures += 1
            request_results.append(
                {
                    "utterance": utterance,
                    "accepted": False,
                    "would_accept": False,
                    "reason": "local SLM failed",
                    "error": str(exc),
                }
            )
            continue

        metadata = result.metadata or {}
        latencies_ms.append(result.latency_ms)
        accepted += int(result.accepted)
        would_accept += int(metadata.get("would_accept") is True)
        failures += int("failed" in result.reason)
        parse_failures += int("parse failed" in result.reason)
        repair_count += int(metadata.get("repair_used") is True)
        confidence = _float_value(result.confidence)
        if confidence is None:
            confidence = _float_value(metadata.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)
        request_results.append(
            {
                "utterance": utterance,
                "accepted": result.accepted,
                "would_accept": metadata.get("would_accept", False),
                "reason": result.reason,
                "latency_ms": result.latency_ms,
                "confidence": confidence,
            }
        )

    duration_ms = (perf_counter() - started_at) * 1000.0
    return {
        "schema_version": "l3-benchmark-v1",
        "status": "success",
        "requests": total,
        "accepted": accepted,
        "would_accept": would_accept,
        "failures": failures,
        "parse_failures": parse_failures,
        "repair_count": repair_count,
        "generation_avg_ms": _avg(latencies_ms),
        "generation_p50_ms": _percentile(latencies_ms, 50),
        "generation_p95_ms": _percentile(latencies_ms, 95),
        "confidence_avg": _avg(confidences),
        "confidence_p50": _percentile(confidences, 50),
        "confidence_p95": _percentile(confidences, 95),
        "throughput_qps": total / (duration_ms / 1000.0) if duration_ms > 0 else 0.0,
        "duration_ms": duration_ms,
        "backend": layer.backend.status(),
        "request_results": request_results,
    }


def _json_candidate(raw_output: str) -> str:
    stripped = raw_output.strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last < first:
        return stripped
    return stripped[first : last + 1]


def _l3_reason(
    *,
    mode: LocalSLMMode,
    would_accept: bool,
    validation_errors: list[str],
) -> str:
    if mode == "shadow":
        return "shadow local SLM would accept" if would_accept else "shadow local SLM rejected"
    if would_accept:
        return "guard accepted"
    if validation_errors:
        return "guard rejected invalid output"
    return "guard rejected low confidence"


def _avg(values: Iterable[float | int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: Iterable[float | int], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _float_value(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
