from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import quantiles

from darjeeling.artifacts.store import ArtifactManifest, LayerDelta
from darjeeling.layers.base import RuntimeLayer
from darjeeling.runtime.cost import ReplayCostModel
from darjeeling.targets.nlu.compiler.objective import ObjectiveMetrics, objective_score
from darjeeling.targets.nlu.data import normalize_utterance
from darjeeling.targets.nlu.layers.l1_rust_programbank import RustL1Worker, build_l1_binary
from darjeeling.targets.nlu.layers.l2_experts import L2ExpertBank, L2ExpertBankLayer
from darjeeling.targets.nlu.layers.l2_student import L2StudentBundle, L2StudentLayer
from darjeeling.targets.nlu.layers.l2_target import TargetL2Layer
from darjeeling.targets.nlu.patches import (
    FrameComposer,
    frame_field_values,
    frame_patch_from_layer_result,
)
from darjeeling.targets.nlu.schemas import Frame, FramePatch, LayerResult, TeacherTrace

LAYER_LATENCY_MS = {
    "L0": 0.1,
    "L1": 1.0,
    "L2": 5.0,
    "L3": 120.0,
    "L4": 900.0,
}
RESIDUAL_L4_LATENCY_MS = 300.0
RESIDUAL_L4_MIN_COST_FRACTION = 0.25


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    current_score: float
    candidate_score: float
    per_layer_deltas: dict[str, LayerDelta] | None = None
    promoted_with_layer_regression: bool = False
    regressed_layers: list[str] | None = None


@dataclass(frozen=True)
class TeacherReplaySplit:
    teacher_train: list[TeacherTrace]
    teacher_promotion_holdout: list[TeacherTrace]
    teacher_regression_sample: list[TeacherTrace]

    @property
    def evaluation_traces(self) -> list[TeacherTrace]:
        seen: set[str] = set()
        traces: list[TeacherTrace] = []
        for trace in [*self.teacher_promotion_holdout, *self.teacher_regression_sample]:
            if trace.request_id in seen:
                continue
            seen.add(trace.request_id)
            traces.append(trace)
        return traces


@dataclass(frozen=True)
class OfflineArtifactSet:
    l0_cache: dict[str, Frame]
    l1_crate_dir: Path | None = None
    l1_worker_timeout_s: float = 5.0
    l2_bundle: L2StudentBundle | None = None
    l2_target_path: Path | None = None
    l2_expert_bank: L2ExpertBank | None = None

    @property
    def artifact_complexity(self) -> float:
        return float(
            sqrt(len(self.l0_cache))
            + (1 if self.l1_crate_dir is not None else 0)
            + (1 if self.l2_bundle is not None else 0)
            + (1 if self.l2_target_path is not None else 0)
            + (1 if self.l2_expert_bank is not None else 0)
        )


@dataclass(frozen=True)
class OfflineReplayResult:
    objective: ObjectiveMetrics
    layer_metrics: dict[str, dict[str, float]]
    layer_counts: dict[str, int]
    field_metrics: dict[str, float]
    cost_metrics: dict[str, float]
    requests: int


@dataclass(frozen=True)
class OfflineRouteResult:
    chosen_layer: str
    frame: Frame
    weak_accepted: bool
    patches: list[FramePatch]
    full_l4_call: bool = False
    residual_l4_call: bool = False
    correct_weak_fields_avoiding_full_l4: int = 0
    residual_l4_verified_fields: int = 0
    l4_conflicts: int = 0


def decide_promotion(
    current: ObjectiveMetrics,
    candidate: ObjectiveMetrics,
    *,
    accuracy_epsilon: float = 0.02,
    max_wrong_accept_rate: float = 0.05,
) -> PromotionDecision:
    current_score = objective_score(current)
    candidate_score = objective_score(candidate)
    if candidate.wrong_accept_rate > max_wrong_accept_rate:
        return PromotionDecision(
            promoted=False,
            reason="wrong_accept_rate exceeds limit",
            current_score=current_score,
            candidate_score=candidate_score,
        )
    if candidate.frame_exact_match < current.frame_exact_match - accuracy_epsilon:
        return PromotionDecision(
            promoted=False,
            reason="accuracy regression exceeds epsilon",
            current_score=current_score,
            candidate_score=candidate_score,
        )
    if candidate_score <= current_score:
        return PromotionDecision(
            promoted=False,
            reason="objective did not improve",
            current_score=current_score,
            candidate_score=candidate_score,
        )
    return PromotionDecision(
        promoted=True,
        reason="objective improved within gates",
        current_score=current_score,
        candidate_score=candidate_score,
    )


def split_teacher_traces(
    traces: list[TeacherTrace],
    *,
    holdout_fraction: float = 0.25,
    max_holdout: int = 200,
    max_regression: int = 100,
) -> TeacherReplaySplit:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    if len(labeled) < 2:
        return TeacherReplaySplit(
            teacher_train=labeled,
            teacher_promotion_holdout=[],
            teacher_regression_sample=[],
        )

    holdout_count = max(1, min(max_holdout, int(len(labeled) * holdout_fraction)))
    holdout_count = min(holdout_count, len(labeled) - 1)
    regression_count = min(max_regression, max(0, len(labeled) // 10))
    if regression_count + holdout_count >= len(labeled):
        regression_count = 0

    regression = labeled[:regression_count]
    remaining = labeled[regression_count:]
    holdout = remaining[-holdout_count:]
    train = remaining[:-holdout_count]
    return TeacherReplaySplit(
        teacher_train=train,
        teacher_promotion_holdout=holdout,
        teacher_regression_sample=regression,
    )


def load_offline_artifact_set(
    artifacts_root: Path,
    manifest: ArtifactManifest | None,
    *,
    default_l1_crate_dir: Path | None = None,
    l1_worker_timeout_s: float = 5.0,
) -> OfflineArtifactSet:
    if manifest is None:
        return OfflineArtifactSet(
            l0_cache={},
            l1_crate_dir=default_l1_crate_dir,
            l1_worker_timeout_s=l1_worker_timeout_s,
        )

    l0_cache: dict[str, Frame] = {}
    l0_path_text = manifest.artifact_paths.get("l0_cache")
    if l0_path_text:
        l0_path = _artifact_path(artifacts_root, l0_path_text)
        payload = _read_json(l0_path)
        l0_cache = {
            normalized_utterance: Frame.model_validate(frame_payload)
            for normalized_utterance, frame_payload in payload.get(
                "frames_by_normalized_utterance", {}
            ).items()
        }

    l2_bundle: L2StudentBundle | None = None
    l2_path_text = manifest.artifact_paths.get("l2_student")
    if l2_path_text:
        l2_bundle = L2StudentBundle.load(_artifact_path(artifacts_root, l2_path_text))
    l2_target_path: Path | None = None
    l2_target_path_text = manifest.artifact_paths.get("l2_target")
    if l2_target_path_text:
        l2_target_path = _artifact_path(artifacts_root, l2_target_path_text)
    l2_expert_bank: L2ExpertBank | None = None
    l2_expert_bank_path_text = manifest.artifact_paths.get("l2_expert_bank")
    if l2_expert_bank_path_text:
        l2_expert_bank = L2ExpertBank.load(_artifact_path(artifacts_root, l2_expert_bank_path_text))

    l1_crate_dir = default_l1_crate_dir
    l1_path_text = manifest.artifact_paths.get("l1_crate_dir")
    if l1_path_text:
        l1_crate_dir = _artifact_path(artifacts_root, l1_path_text)

    return OfflineArtifactSet(
        l0_cache=l0_cache,
        l1_crate_dir=l1_crate_dir,
        l1_worker_timeout_s=l1_worker_timeout_s,
        l2_bundle=l2_bundle,
        l2_target_path=l2_target_path,
        l2_expert_bank=l2_expert_bank,
    )


def evaluate_offline_artifact_set(
    traces: list[TeacherTrace],
    artifact_set: OfflineArtifactSet,
    *,
    cost_model: ReplayCostModel | None = None,
) -> OfflineReplayResult:
    cost_model = cost_model or ReplayCostModel()
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    layer_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    layer_correct_accepts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    layer_wrong_accepts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    layer_costs = {"L0": 0.0, "L1": 0.0, "L2": 0.0, "L3": 0.0, "L4": 0.0}
    layer_field_accepts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    layer_field_correct = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    layer_field_wrong = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    latencies: list[float] = []
    total_cost = 0.0
    frame_matches = 0
    wrong_accepts = 0
    total_expected_fields = 0
    weak_accepted_fields = 0
    weak_correct_fields = 0
    weak_wrong_fields = 0
    full_l4_calls = 0
    residual_l4_calls = 0
    correct_weak_fields_avoiding_full_l4 = 0
    residual_l4_verified_fields = 0
    l4_conflicts = 0

    l1_worker = _build_l1_worker(artifact_set)
    l2_layer = _build_l2_layer(artifact_set)
    try:
        for trace in labeled:
            expected = trace.teacher_frame
            assert expected is not None
            route = _offline_route(
                trace,
                artifact_set,
                expected,
                l1_worker=l1_worker,
                l2_layer=l2_layer,
            )
            chosen_layer = route.chosen_layer
            frame = route.frame
            correct = frame == expected
            frame_matches += int(correct)
            layer_counts[chosen_layer] += 1
            layer_correct_accepts[chosen_layer] += int(correct)
            if route.weak_accepted and not correct:
                wrong_accepts += 1
                layer_wrong_accepts[chosen_layer] += 1
            expected_fields = frame_field_values(expected)
            total_expected_fields += len(expected_fields)
            for patch in route.patches:
                patch_values = _patch_field_values(patch)
                layer = patch.source_layer
                layer_field_accepts[layer] += len(patch_values)
                for field_key, field_value in patch_values.items():
                    is_correct = expected_fields.get(field_key) == field_value
                    layer_field_correct[layer] += int(is_correct)
                    layer_field_wrong[layer] += int(not is_correct)
                    if layer != "L4":
                        weak_accepted_fields += 1
                        weak_correct_fields += int(is_correct)
                        weak_wrong_fields += int(not is_correct)
            full_l4_calls += int(route.full_l4_call)
            residual_l4_calls += int(route.residual_l4_call)
            correct_weak_fields_avoiding_full_l4 += (
                route.correct_weak_fields_avoiding_full_l4
            )
            residual_l4_verified_fields += route.residual_l4_verified_fields
            l4_conflicts += route.l4_conflicts
            latencies.append(
                RESIDUAL_L4_LATENCY_MS if route.residual_l4_call else LAYER_LATENCY_MS[chosen_layer]
            )
            request_cost = _offline_request_cost(
                chosen_layer=chosen_layer,
                route=route,
                expected_field_count=len(expected_fields),
                trace=trace,
                cost_model=cost_model,
            )
            layer_costs[chosen_layer] += request_cost
            total_cost += request_cost
    finally:
        if l1_worker is not None:
            l1_worker.close()

    requests = len(labeled)
    if requests == 0:
        objective = ObjectiveMetrics(
            frame_exact_match=0.0,
            wrong_accept_rate=1.0,
            cost_usd_per_100_requests=0.0,
            p95_latency_ms=0.0,
            artifact_complexity=artifact_set.artifact_complexity,
        )
        return OfflineReplayResult(
            objective=objective,
            layer_metrics={},
            layer_counts=layer_counts,
            field_metrics={
                "total_expected_fields": 0.0,
                "weak_accepted_fields": 0.0,
                "weak_field_coverage": 0.0,
                "weak_correct_fields": 0.0,
                "weak_wrong_fields": 0.0,
                "weak_field_accuracy": 1.0,
                "wrong_accepted_field_rate": 0.0,
                "correct_weak_fields_avoiding_full_l4": 0.0,
                "correct_weak_fields_avoiding_full_l4_per_100": 0.0,
                "residual_l4_verified_fields": 0.0,
                "residual_l4_verified_fields_per_100": 0.0,
                "l4_field_conflicts": 0.0,
                "l4_conflict_rate": 0.0,
            },
            cost_metrics=_empty_cost_metrics(),
            requests=0,
        )

    layer_metrics = {
        layer: {
            "coverage": count / requests,
            "accepted_accuracy": (layer_correct_accepts[layer] / count if count else 1.0),
            "wrong_accept_rate": layer_wrong_accepts[layer] / requests,
            "p95_latency_ms": LAYER_LATENCY_MS[layer] if count else 0.0,
            "cost_usd_per_100_requests": layer_costs[layer] / requests * 100.0,
            "layer_share": count / requests,
            "field_accepts_per_request": layer_field_accepts[layer] / requests,
            "field_accepted_accuracy": (
                layer_field_correct[layer] / layer_field_accepts[layer]
                if layer_field_accepts[layer]
                else 1.0
            ),
            "wrong_accepted_field_rate": layer_field_wrong[layer]
            / max(1, total_expected_fields),
        }
        for layer, count in layer_counts.items()
    }
    field_metrics = {
        "total_expected_fields": float(total_expected_fields),
        "weak_accepted_fields": float(weak_accepted_fields),
        "weak_field_coverage": weak_accepted_fields / total_expected_fields
        if total_expected_fields
        else 0.0,
        "weak_correct_fields": float(weak_correct_fields),
        "weak_wrong_fields": float(weak_wrong_fields),
        "weak_field_accuracy": weak_correct_fields / weak_accepted_fields
        if weak_accepted_fields
        else 1.0,
        "wrong_accepted_field_rate": weak_wrong_fields / total_expected_fields
        if total_expected_fields
        else 0.0,
        "correct_weak_fields_avoiding_full_l4": float(
            correct_weak_fields_avoiding_full_l4
        ),
        "correct_weak_fields_avoiding_full_l4_per_100": (
            correct_weak_fields_avoiding_full_l4 / requests * 100.0
        ),
        "residual_l4_verified_fields": float(residual_l4_verified_fields),
        "residual_l4_verified_fields_per_100": (
            residual_l4_verified_fields / requests * 100.0
        ),
        "l4_field_conflicts": float(l4_conflicts),
        "l4_conflict_rate": l4_conflicts / total_expected_fields
        if total_expected_fields
        else 0.0,
    }
    cost_metrics = {
        "serving_full_l4_calls": float(full_l4_calls),
        "serving_residual_l4_calls": float(residual_l4_calls),
        "serving_full_l4_calls_per_100": full_l4_calls / requests * 100.0,
        "serving_residual_l4_calls_per_100": residual_l4_calls / requests * 100.0,
        "serving_l4_fields_avoided": float(correct_weak_fields_avoiding_full_l4),
        "serving_l4_fields_avoided_per_100": (
            correct_weak_fields_avoiding_full_l4 / requests * 100.0
        ),
    }
    objective = ObjectiveMetrics(
        frame_exact_match=frame_matches / requests,
        wrong_accept_rate=wrong_accepts / requests,
        cost_usd_per_100_requests=total_cost / requests * 100.0,
        p95_latency_ms=_p95(latencies),
        artifact_complexity=artifact_set.artifact_complexity,
        correct_weak_fields_avoiding_full_l4_per_100=field_metrics[
            "correct_weak_fields_avoiding_full_l4_per_100"
        ],
        residual_l4_verified_fields_per_100=field_metrics[
            "residual_l4_verified_fields_per_100"
        ],
        full_l4_calls_per_100_requests=cost_metrics["serving_full_l4_calls_per_100"],
        residual_l4_calls_per_100_requests=cost_metrics[
            "serving_residual_l4_calls_per_100"
        ],
        wrong_accepted_field_rate=field_metrics["wrong_accepted_field_rate"],
        l4_conflict_rate=field_metrics["l4_conflict_rate"],
    )
    return OfflineReplayResult(
        objective=objective,
        layer_metrics=layer_metrics,
        layer_counts=layer_counts,
        field_metrics=field_metrics,
        cost_metrics=cost_metrics,
        requests=requests,
    )


def layer_deltas(
    current: OfflineReplayResult,
    candidate: OfflineReplayResult,
) -> dict[str, LayerDelta]:
    layers = sorted(set(current.layer_metrics) | set(candidate.layer_metrics))
    return {
        layer: LayerDelta(
            coverage_delta=_metric(candidate, layer, "coverage")
            - _metric(current, layer, "coverage"),
            accepted_accuracy_delta=_metric(candidate, layer, "accepted_accuracy")
            - _metric(current, layer, "accepted_accuracy"),
            wrong_accept_delta=_metric(candidate, layer, "wrong_accept_rate")
            - _metric(current, layer, "wrong_accept_rate"),
            p95_latency_ms_delta=_metric(candidate, layer, "p95_latency_ms")
            - _metric(current, layer, "p95_latency_ms"),
            cost_delta=_metric(candidate, layer, "cost_usd_per_100_requests")
            - _metric(current, layer, "cost_usd_per_100_requests"),
            layer_share_delta=_metric(candidate, layer, "layer_share")
            - _metric(current, layer, "layer_share"),
        )
        for layer in layers
    }


def decide_artifact_set_promotion(
    current: ObjectiveMetrics,
    candidate: ObjectiveMetrics,
    *,
    per_layer_deltas: dict[str, LayerDelta],
    accuracy_epsilon: float = 0.02,
    max_wrong_accept_rate: float = 0.05,
    block_layer_regressions: bool = True,
) -> PromotionDecision:
    decision = decide_promotion(
        current,
        candidate,
        accuracy_epsilon=accuracy_epsilon,
        max_wrong_accept_rate=max_wrong_accept_rate,
    )
    regressed_layers = detect_layer_regressions(per_layer_deltas)
    promoted = decision.promoted
    reason = decision.reason
    if decision.promoted and block_layer_regressions and regressed_layers:
        promoted = False
        reason = f"per-layer regression gate failed: {', '.join(sorted(regressed_layers))}"
    return PromotionDecision(
        promoted=promoted,
        reason=reason,
        current_score=decision.current_score,
        candidate_score=decision.candidate_score,
        per_layer_deltas=per_layer_deltas,
        promoted_with_layer_regression=promoted and bool(regressed_layers),
        regressed_layers=regressed_layers,
    )


def detect_layer_regressions(
    per_layer_deltas: dict[str, LayerDelta],
    *,
    accepted_accuracy_tolerance: float = 0.01,
    wrong_accept_tolerance: float = 0.01,
    p95_latency_ms_tolerance: float = 25.0,
) -> list[str]:
    regressed = []
    for layer, delta in per_layer_deltas.items():
        layer_usage_did_not_drop = delta.layer_share_delta >= 0.0
        if (
            layer_usage_did_not_drop
            and delta.accepted_accuracy_delta < -accepted_accuracy_tolerance
        ):
            regressed.append(layer)
            continue
        if layer_usage_did_not_drop and delta.wrong_accept_delta > wrong_accept_tolerance:
            regressed.append(layer)
            continue
        if layer_usage_did_not_drop and delta.p95_latency_ms_delta > p95_latency_ms_tolerance:
            regressed.append(layer)
            continue
    return regressed


def _offline_route(
    trace: TeacherTrace,
    artifact_set: OfflineArtifactSet,
    fallback_frame: Frame,
    *,
    l1_worker: RustL1Worker | None,
    l2_layer: RuntimeLayer | None,
) -> OfflineRouteResult:
    composer = FrameComposer()
    patches: list[FramePatch] = []
    normalized = normalize_utterance(trace.utterance)
    if normalized in artifact_set.l0_cache:
        result = LayerResult(
            layer="L0",
            accepted=True,
            frame=artifact_set.l0_cache[normalized],
            confidence=1.0,
            reason="offline exact cache hit",
            latency_ms=LAYER_LATENCY_MS["L0"],
        )
        if _apply_offline_result(composer, patches, result):
            return OfflineRouteResult("L0", composer.to_frame(), True, patches)
    if l1_worker is not None:
        l1_response = l1_worker.answer(trace.utterance)
        if l1_response.accepted and (
            l1_response.frame is not None or getattr(l1_response, "patch", None) is not None
        ):
            result = LayerResult(
                layer="L1",
                accepted=True,
                frame=l1_response.frame,
                patch=getattr(l1_response, "patch", None),
                confidence=1.0,
                reason=l1_response.reason,
                latency_ms=LAYER_LATENCY_MS["L1"],
            )
            if _apply_offline_result(composer, patches, result):
                return OfflineRouteResult("L1", composer.to_frame(), True, patches)
    if l2_layer is not None:
        l2_result = l2_layer.try_answer(trace.utterance)
        if _apply_offline_result(composer, patches, l2_result):
            return OfflineRouteResult("L2", composer.to_frame(), True, patches)
    if l3_result := _recorded_l3_accept(trace):
        result = LayerResult(
            layer="L3",
            accepted=True,
            frame=l3_result,
            confidence=1.0,
            reason="recorded L3 accept",
            latency_ms=LAYER_LATENCY_MS["L3"],
        )
        if _apply_offline_result(composer, patches, result):
            return OfflineRouteResult("L3", composer.to_frame(), True, patches)
    pre_l4_values = composer.field_values()
    expected_values = frame_field_values(fallback_frame)
    correct_pre_l4_fields = sum(
        expected_values.get(field_key) == field_value
        for field_key, field_value in pre_l4_values.items()
    )
    residual_l4_call = bool(pre_l4_values)
    patch = composer.fill_or_override_from_l4_frame(
        fallback_frame,
        source_layer="L4",
        confidence=1.0,
        metadata={
            "adapter": "offline_l4_residual_fill"
            if residual_l4_call
            else "offline_l4_full_frame",
            "l4_call_kind": "residual" if residual_l4_call else "full",
            "fields_avoided": correct_pre_l4_fields if residual_l4_call else 0,
        },
    )
    patches.append(patch)
    return OfflineRouteResult(
        "L4",
        composer.to_frame(),
        bool(pre_l4_values),
        patches,
        full_l4_call=not residual_l4_call,
        residual_l4_call=residual_l4_call,
        correct_weak_fields_avoiding_full_l4=(
            correct_pre_l4_fields if residual_l4_call else 0
        ),
        residual_l4_verified_fields=len(patch.metadata.get("verified_fields", []))
        if residual_l4_call
        else 0,
        l4_conflicts=len(patch.metadata.get("field_conflicts", [])),
    )


def _apply_offline_result(
    composer: FrameComposer,
    patches: list[FramePatch],
    result: LayerResult,
) -> bool:
    patch = frame_patch_from_layer_result(result)
    if patch is None:
        return False
    composer.apply_patch(patch)
    patches.append(patch)
    return patch.complete and composer.complete


def _recorded_l3_accept(trace: TeacherTrace) -> Frame | None:
    for result in trace.layer_results:
        if result.layer == "L3" and result.accepted and result.frame is not None:
            return result.frame
    return None


def _patch_field_values(patch: FramePatch) -> dict[str, str]:
    values: dict[str, str] = {}
    if patch.accepted_intent is not None:
        values["intent"] = patch.accepted_intent
    values.update(
        {
            f"slots.{slot_key}": slot_value
            for slot_key, slot_value in patch.accepted_slots.items()
        }
    )
    return values


def _offline_request_cost(
    *,
    chosen_layer: str,
    route: OfflineRouteResult,
    expected_field_count: int,
    trace: TeacherTrace,
    cost_model: ReplayCostModel,
) -> float:
    if chosen_layer != "L4" or not route.residual_l4_call:
        return cost_model.layer_cost_usd(chosen_layer, trace.l4_usage)
    full_cost = cost_model.layer_cost_usd("L4", trace.l4_usage)
    if expected_field_count <= 0:
        return full_cost * RESIDUAL_L4_MIN_COST_FRACTION
    missing_fraction = (
        max(0, expected_field_count - route.correct_weak_fields_avoiding_full_l4)
        / expected_field_count
    )
    return full_cost * max(RESIDUAL_L4_MIN_COST_FRACTION, missing_fraction)


def _empty_cost_metrics() -> dict[str, float]:
    return {
        "serving_full_l4_calls": 0.0,
        "serving_residual_l4_calls": 0.0,
        "serving_full_l4_calls_per_100": 0.0,
        "serving_residual_l4_calls_per_100": 0.0,
        "serving_l4_fields_avoided": 0.0,
        "serving_l4_fields_avoided_per_100": 0.0,
    }


def _metric(result: OfflineReplayResult, layer: str, name: str) -> float:
    return result.layer_metrics.get(layer, {}).get(name, 0.0)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return float(quantiles(values, n=100, method="inclusive")[94])


def _artifact_path(artifacts_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return artifacts_root / path


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _build_l1_worker(artifact_set: OfflineArtifactSet) -> RustL1Worker | None:
    if artifact_set.l1_crate_dir is None:
        return None
    binary_path = build_l1_binary(artifact_set.l1_crate_dir)
    worker = RustL1Worker(binary_path, timeout_s=artifact_set.l1_worker_timeout_s)
    worker.start()
    return worker


def _build_l2_layer(artifact_set: OfflineArtifactSet) -> RuntimeLayer | None:
    if artifact_set.l2_bundle is None:
        if artifact_set.l2_expert_bank is None:
            return None
        return L2ExpertBankLayer(artifact_set.l2_expert_bank)
    if artifact_set.l2_target_path is not None:
        fallback_layer = TargetL2Layer(artifact_set.l2_bundle, artifact_set.l2_target_path)
    else:
        fallback_layer = L2StudentLayer(artifact_set.l2_bundle)
    if artifact_set.l2_expert_bank is not None:
        return L2ExpertBankLayer(artifact_set.l2_expert_bank, fallback_layer)
    return fallback_layer
