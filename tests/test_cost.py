import pytest

from darjeeling.runtime.cost import ReplayCostModel, estimate_l4_cost_from_usage


def test_estimate_l4_cost_uses_cached_input_discount() -> None:
    cost = estimate_l4_cost_from_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
        input_usd_per_million=1.0,
        cached_input_usd_per_million=0.25,
        output_usd_per_million=4.0,
    )

    assert cost == pytest.approx((60 * 1.0 + 40 * 0.25 + 20 * 4.0) / 1_000_000)


def test_replay_cost_model_uses_default_l4_cost_without_usage() -> None:
    model = ReplayCostModel(l4_default_cost_usd_per_request=0.123)

    assert model.layer_cost_usd("L4", {}) == 0.123
