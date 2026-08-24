"""Model-agnostic OpenAI SDK client factory.

Works against OpenAI directly or any OpenAI-compatible endpoint (e.g.
OpenRouter) purely via `base_url` / `api_key` config -- no code changes
needed to swap providers.
"""
from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from .config import Config


def make_client(config: Config) -> AsyncOpenAI:
    """Build one client, intended to be shared for the whole run.

    A client owns an HTTP connection pool, so reusing a single instance keeps
    TLS handshakes to a minimum and lets requests ride warm connections. The
    pool is sized to the request concurrency cap with headroom -- if
    ``max_connections`` were lower than the number of in-flight calls, the
    extra calls would queue inside httpx and could time out waiting for a
    connection rather than waiting on the model.
    """
    pool_size = max(config.max_concurrent_requests * 2, 10)
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=httpx.Timeout(
            config.request_timeout_seconds,
            # A long generation can legitimately stall between tokens, so the
            # read timeout is the full budget; connecting should be quick.
            connect=15.0,
        ),
        max_retries=config.sdk_max_retries,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size,
                keepalive_expiry=60.0,
            ),
            timeout=httpx.Timeout(config.request_timeout_seconds, connect=15.0),
        ),
    )
