from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from candidate_reader.errors import ConfigurationError, CostBudgetExceededError


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    cache_write_per_million: Decimal
    output_per_million: Decimal


MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-4.1": ModelPrice(Decimal("2.00"), Decimal("0"), Decimal("8.00")),
    "gpt-4.1-mini": ModelPrice(Decimal("0.40"), Decimal("0"), Decimal("1.60")),
    "gpt-4.1-nano": ModelPrice(Decimal("0.10"), Decimal("0"), Decimal("0.40")),
    "gpt-4o": ModelPrice(Decimal("2.50"), Decimal("0"), Decimal("10.00")),
    "gpt-4o-mini": ModelPrice(Decimal("0.15"), Decimal("0"), Decimal("0.60")),
    "gpt-5-mini": ModelPrice(Decimal("0.25"), Decimal("0"), Decimal("2.00")),
    "gpt-5.3-codex": ModelPrice(Decimal("1.75"), Decimal("0"), Decimal("14.00")),
    "gpt-5.4": ModelPrice(Decimal("2.50"), Decimal("0"), Decimal("15.00")),
    "gpt-5.4-mini": ModelPrice(Decimal("0.75"), Decimal("0"), Decimal("4.50")),
    "gpt-5.4-nano": ModelPrice(Decimal("0.20"), Decimal("0"), Decimal("1.25")),
    "gpt-5.5": ModelPrice(Decimal("5.00"), Decimal("0"), Decimal("30.00")),
    "gpt-5.6-luna": ModelPrice(Decimal("0.20"), Decimal("0.25"), Decimal("1.20")),
    "gpt-5.6-terra": ModelPrice(Decimal("2.00"), Decimal("2.50"), Decimal("12.00")),
    "gpt-5.6-sol": ModelPrice(Decimal("4.00"), Decimal("5.00"), Decimal("20.00")),
    "gpt-6-luna": ModelPrice(Decimal("0.10"), Decimal("0.125"), Decimal("0.50")),
    "gpt-6-sol": ModelPrice(Decimal("2.00"), Decimal("2.50"), Decimal("10.00")),
    "gpt-6-astra": ModelPrice(Decimal("10.00"), Decimal("12.50"), Decimal("50.00")),
}

SCHEMA_AND_SDK_OVERHEAD_TOKENS = 30_000


class CostBudget:
    def __init__(
        self,
        model: str,
        limit_usd: Decimal,
        *,
        custom_input_per_million: Decimal | None = None,
        custom_cache_write_per_million: Decimal | None = None,
        custom_output_per_million: Decimal | None = None,
    ) -> None:
        custom_rates = (
            custom_input_per_million,
            custom_cache_write_per_million,
            custom_output_per_million,
        )
        if any(rate is not None for rate in custom_rates):
            if not all(rate is not None for rate in custom_rates):
                raise ConfigurationError(
                    "Set all three custom model prices: input, cache write, and output"
                )
            assert custom_input_per_million is not None
            assert custom_cache_write_per_million is not None
            assert custom_output_per_million is not None
            self._price = ModelPrice(
                custom_input_per_million,
                custom_cache_write_per_million,
                custom_output_per_million,
            )
        elif model in MODEL_PRICES:
            self._price = MODEL_PRICES[model]
        else:
            supported = ", ".join(sorted(MODEL_PRICES))
            raise ConfigurationError(
                f"Model '{model}' is allowed, but its price is unknown. Set "
                "OPENAI_INPUT_PRICE_PER_MILLION, "
                "OPENAI_CACHE_WRITE_PRICE_PER_MILLION, and "
                "OPENAI_OUTPUT_PRICE_PER_MILLION in .env. "
                f"Models with built-in rates: {supported}"
            )
        self.model = model
        self.limit_usd = limit_usd
        self.reserved_usd = Decimal("0")

    def estimate_call(self, prompt: str, max_output_tokens: int) -> Decimal:
        # UTF-8 bytes are a conservative upper bound for normal text tokenization.
        input_tokens = len(prompt.encode("utf-8")) + SCHEMA_AND_SDK_OVERHEAD_TOKENS
        input_cost = (
            Decimal(input_tokens)
            * (self._price.input_per_million + self._price.cache_write_per_million)
            / Decimal(1_000_000)
        )
        output_cost = (
            Decimal(max_output_tokens) * self._price.output_per_million / Decimal(1_000_000)
        )
        return input_cost + output_cost

    def reserve_call(self, prompt: str, max_output_tokens: int) -> Decimal:
        estimated = self.estimate_call(prompt, max_output_tokens)
        projected = self.reserved_usd + estimated
        if projected > self.limit_usd:
            raise CostBudgetExceededError(
                f"Stopped before model call: projected OpenAI cost ${projected:.4f} "
                f"exceeds the ${self.limit_usd:.2f} limit"
            )
        self.reserved_usd = projected
        return estimated
