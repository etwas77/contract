from decimal import Decimal

import pytest

from candidate_reader.costs import CostBudget
from candidate_reader.errors import ConfigurationError, CostBudgetExceededError


def test_cost_budget_stops_before_limit_is_exceeded() -> None:
    budget = CostBudget("gpt-5.6-terra", Decimal("0.10"))

    with pytest.raises(CostBudgetExceededError, match="Stopped before model call"):
        budget.reserve_call("candidate text", max_output_tokens=5_000)


def test_cost_budget_accumulates_reservations() -> None:
    budget = CostBudget("gpt-5.6-luna", Decimal("0.50"))

    first = budget.reserve_call("candidate text", max_output_tokens=1_000)
    second = budget.reserve_call("constraints", max_output_tokens=1_000)

    assert first > 0
    assert budget.reserved_usd == first + second


def test_custom_model_is_allowed_with_env_prices() -> None:
    budget = CostBudget(
        "custom-provider/model",
        Decimal("0.50"),
        custom_input_per_million=Decimal("1.00"),
        custom_cache_write_per_million=Decimal("0.00"),
        custom_output_per_million=Decimal("2.00"),
    )

    assert budget.reserve_call("candidate text", max_output_tokens=1_000) > 0


def test_unknown_model_requires_complete_custom_prices() -> None:
    with pytest.raises(ConfigurationError, match="price is unknown"):
        CostBudget("custom-provider/model", Decimal("0.50"))

    with pytest.raises(ConfigurationError, match="all three"):
        CostBudget(
            "custom-provider/model",
            Decimal("0.50"),
            custom_input_per_million=Decimal("1.00"),
        )
