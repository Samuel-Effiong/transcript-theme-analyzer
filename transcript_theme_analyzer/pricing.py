"""Call cost, provider discount and per-theme attribution.

Pure functions over numbers and ``CallLog``-shaped objects, so this module
never imports ``analyzer`` (which imports it).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from .config import Config

if TYPE_CHECKING:
    from .analyzer import CallLog

logger = logging.getLogger("transcript_theme_analyzer")


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens."""
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None = None


def call_cost(call: "CallLog", prices: Prices | None) -> tuple[float | None, str | None]:
    """OpenRouter's reported cost wins; else tokens x prices; else unknown."""
    if call.reported_cost is not None:
        return call.reported_cost, "openrouter"
    if prices is None:
        return None, None
    cached = call.cached_tokens
    cached_rate = prices.cached_input_per_m
    if cached_rate is None:
        cached_rate = prices.input_per_m
    cost = (
        (call.prompt_tokens - cached) * prices.input_per_m
        + cached * cached_rate
        + call.completion_tokens * prices.output_per_m
    ) / 1e6
    return cost, "prices"


def call_discount(call: "CallLog", prices: Prices | None) -> float | None:
    """Money saved by cached input tokens versus the full input price."""
    if prices is None:
        return None
    if call.cached_tokens == 0:
        return 0.0
    if prices.cached_input_per_m is None:
        return None
    return call.cached_tokens * (prices.input_per_m - prices.cached_input_per_m) / 1e6


def _input_fraction(call: "CallLog", prices: Prices | None) -> float:
    """Share of the call attributable to input (the rest is output).

    OpenRouter's own input/output cost split wins, then the price ratio,
    then the token ratio."""
    reported_in = getattr(call, "reported_input_cost", None)
    reported_out = getattr(call, "reported_output_cost", None)
    if reported_in is not None and reported_out is not None and min(reported_in, reported_out) >= 0:
        reported_total = reported_in + reported_out
        if reported_total > 0:
            return reported_in / reported_total
    if prices is not None:
        cached = call.cached_tokens
        cached_rate = prices.cached_input_per_m
        if cached_rate is None:
            cached_rate = prices.input_per_m
        input_cost = (
            (call.prompt_tokens - cached) * prices.input_per_m + cached * cached_rate
        )
        output_cost = call.completion_tokens * prices.output_per_m
        denom = input_cost + output_cost
        if denom > 0:
            return input_cost / denom
    return _token_input_fraction(call)


def _token_input_fraction(call: "CallLog") -> float:
    denom = call.prompt_tokens + call.completion_tokens
    return call.prompt_tokens / denom if denom else 1.0


def _output_weights(call: "CallLog") -> dict[str, float]:
    chars = {t: max(call.theme_output_chars.get(t, 0), 0) for t in call.theme_ids}
    total = sum(chars.values())
    if total <= 0:
        return {t: 1 / len(call.theme_ids) for t in call.theme_ids}
    return {t: n / total for t, n in chars.items()}


def attribute_call(
    call: "CallLog", prices: Prices | None
) -> dict[str, tuple[float, float | None]]:
    """Split one call across its themes -> ``{theme_id: (tokens, cost)}``.

    Even share of the input part plus an output share proportional to each
    theme's output size. Estimates: unrounded, round at display time.
    """
    ids = list(dict.fromkeys(call.theme_ids))
    if not ids:
        return {}
    n = len(ids)
    weights = _output_weights(call)
    token_in = _token_input_fraction(call)
    cost_in = _input_fraction(call, prices)
    out: dict[str, tuple[float, float | None]] = {}
    for t in ids:
        tokens = call.total_tokens * (token_in / n + (1 - token_in) * weights[t])
        cost = (
            None
            if call.cost is None
            else call.cost * (cost_in / n + (1 - cost_in) * weights[t])
        )
        out[t] = (tokens, cost)
    return out


def fetch_openrouter_prices(
    models: list[str],
    base_url: str,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> dict[str, Prices]:
    """Look up per-1M prices in OpenRouter's catalogue. Any failure -> ``{}``."""
    try:
        url = base_url.rstrip("/") + "/models"
        if client is not None:
            resp = client.get(url, timeout=timeout)
        else:
            resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        wanted = set(models)
        out: dict[str, Prices] = {}
        for entry in resp.json()["data"]:
            if entry.get("id") not in wanted:
                continue
            pricing = entry.get("pricing") or {}
            try:
                prompt = float(pricing["prompt"]) * 1e6
                completion = float(pricing["completion"]) * 1e6
            except (KeyError, TypeError, ValueError):
                continue
            cache_read = pricing.get("input_cache_read")
            try:
                cached = float(cache_read) * 1e6 if cache_read not in (None, "") else None
            except (TypeError, ValueError):
                cached = None
            out[entry["id"]] = Prices(prompt, completion, cached)
        return out
    except Exception as exc:  # noqa: BLE001 - pricing is best-effort
        logger.warning("Could not fetch OpenRouter prices: %s", exc)
        return {}


def resolve_prices(models: list[str], config: Config) -> dict[str, Prices | None]:
    """Env prices (both input and output set) win for every model; otherwise
    OpenRouter's catalogue when on OpenRouter; otherwise unknown."""
    if config.price_input_per_m is not None and config.price_output_per_m is not None:
        env = Prices(
            config.price_input_per_m,
            config.price_output_per_m,
            config.price_cached_input_per_m,
        )
        return {m: env for m in models}
    fetched = fetch_openrouter_prices(models, config.base_url) if config.is_openrouter else {}
    return {m: fetched.get(m) for m in models}
