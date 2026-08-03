"""Environment-driven configuration for the analyzer pipeline."""
from __future__ import annotations

import os

from pydantic import BaseModel, Field


def _load_dotenv() -> None:
    """Minimal .env loader so the package has no extra dependency."""
    path = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()


class Config(BaseModel):
    provider: str = Field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "openrouter"))
    base_url: str = Field(
        default_factory=lambda: os.environ.get(
            "LLM_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )
    api_key: str = Field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    default_model: str = Field(
        default_factory=lambda: os.environ.get(
            "LLM_DEFAULT_MODEL", "anthropic/claude-sonnet-5"
        )
    )
    chunk_size_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("CHUNK_SIZE_TOKENS", "12000"))
    )
    chunk_overlap_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP_TOKENS", "800"))
    )
    single_pass_token_limit: int = Field(
        default_factory=lambda: int(os.environ.get("SINGLE_PASS_TOKEN_LIMIT", "20000"))
    )
    max_concurrent_chunks: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_CHUNKS", "8"))
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_RETRIES", "5"))
    )
    max_output_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "4096"))
    )


def load_config() -> Config:
    return Config()
