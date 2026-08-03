"""Model-agnostic OpenAI SDK client factory.

Works against OpenAI directly or any OpenAI-compatible endpoint (e.g.
OpenRouter) purely via `base_url` / `api_key` config -- no code changes
needed to swap providers.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from .config import Config


def make_client(config: Config) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
