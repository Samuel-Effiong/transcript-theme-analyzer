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
    max_concurrent_transcripts: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_TRANSCRIPTS", "4"))
    )
    # The real throttle. Transcript- and chunk-level concurrency multiply
    # (4 transcripts x 8 chunks = 32 in-flight), which is how a run walks
    # into provider rate limits. This caps total simultaneous API calls
    # regardless of how those two interleave.
    max_concurrent_requests: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_REQUESTS", "12"))
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_RETRIES", "5"))
    )
    max_output_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "4096"))
    )
    # Per-request ceiling. The SDK's own default is 600s, long enough for one
    # hung call to hold a concurrency slot for ten minutes.
    request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.environ.get("LLM_REQUEST_TIMEOUT", "180"))
    )
    # Retries the SDK itself performs for connection-level faults, beneath
    # our own retry loop. Cheap insurance against transient socket errors.
    sdk_max_retries: int = Field(
        default_factory=lambda: int(os.environ.get("LLM_SDK_MAX_RETRIES", "2"))
    )

    def validated(self) -> "Config":
        """Clamp nonsensical concurrency/retry values rather than letting a
        stray env var (``MAX_CONCURRENT_REQUESTS=0``) deadlock the run."""
        self.max_concurrent_chunks = max(1, self.max_concurrent_chunks)
        self.max_concurrent_transcripts = max(1, self.max_concurrent_transcripts)
        self.max_concurrent_requests = max(1, self.max_concurrent_requests)
        self.max_retries = max(1, self.max_retries)
        self.sdk_max_retries = max(0, self.sdk_max_retries)
        self.request_timeout_seconds = max(1.0, self.request_timeout_seconds)
        return self


def load_config() -> Config:
    return Config().validated()
