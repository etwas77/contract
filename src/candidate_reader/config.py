from decimal import Decimal
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-terra"
    openai_input_price_per_million: Decimal | None = Field(default=None, ge=0)
    openai_cache_write_price_per_million: Decimal | None = Field(default=None, ge=0)
    openai_output_price_per_million: Decimal | None = Field(default=None, ge=0)
    openai_agents_disable_tracing: bool = True
    max_openai_cost_usd: Decimal = Field(default=Decimal("0.50"), gt=0)
    structured_data_dir: Path = Path("structured_data")
