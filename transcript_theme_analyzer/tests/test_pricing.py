from types import SimpleNamespace

import httpx
import pytest

from transcript_theme_analyzer.analyzer import RunStats, openrouter_extra_body
from transcript_theme_analyzer.config import Config
from transcript_theme_analyzer.pricing import (
    Prices,
    _input_fraction,
    fetch_openrouter_prices,
    resolve_prices,
)

P = Prices(input_per_m=1.0, output_per_m=4.0, cached_input_per_m=0.1)


def _stats(prompt=0, completion=0, cached=0, reported_cost=None, theme_ids=(),
           theme_output_chars=None):
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    if reported_cost is not None:
        usage.cost = reported_cost
    s = RunStats()
    s.add("call", "m", usage, theme_ids=theme_ids, theme_output_chars=theme_output_chars)
    return s


def test_reported_cost_wins():
    s = _stats(prompt=1000, completion=100, cached=0, reported_cost=0.5); s.apply_prices(P)
    assert s.summary()["cost"] == 0.5 and s.summary()["cost_source"] == "openrouter"


def test_price_cost_and_discount():
    s = _stats(prompt=1_000_000, completion=0, cached=500_000); s.apply_prices(P)
    assert s.summary()["cost"] == pytest.approx(0.55)
    assert s.summary()["discount"] == pytest.approx(0.45)
    assert s.summary()["cost_source"] == "prices"


def test_no_prices_no_report_is_none_not_zero():
    s = _stats(prompt=10, completion=5, cached=0); s.apply_prices(None)
    assert s.summary()["cost"] is None and s.summary()["discount"] is None


def test_theme_attribution_sums_to_call_and_follows_output_share():
    s = _stats(prompt=900, completion=100, cached=0, theme_ids=["a", "b", "c"],
               theme_output_chars={"a": 300, "b": 100, "c": 0}); s.apply_prices(P)
    tu = s.theme_usage()
    assert sum(v["tokens"] for v in tu.values()) == pytest.approx(1000)
    assert sum(v["cost"] for v in tu.values()) == pytest.approx(s.summary()["cost"])
    assert tu["a"]["tokens"] > tu["b"]["tokens"] > tu["c"]["tokens"]


def test_theme_usage_before_prices_has_tokens_but_no_cost():
    s = _stats(prompt=90, completion=10, theme_ids=["a", "b"])
    tu = s.theme_usage()
    assert tu["a"]["tokens"] == pytest.approx(50) and tu["a"]["cost"] is None


def test_even_output_split_when_chars_all_zero():
    s = _stats(prompt=0, completion=100, theme_ids=["a", "b"],
               theme_output_chars={"a": 0, "b": 0})
    assert s.theme_usage()["a"]["tokens"] == pytest.approx(50)


def test_reported_cost_attribution_uses_token_ratio_without_prices():
    s = _stats(prompt=900, completion=100, reported_cost=1.0, theme_ids=["a"])
    s.apply_prices(None)
    assert s.theme_usage()["a"]["cost"] == pytest.approx(1.0)


def test_summary_cost_none_if_any_call_unknown_but_discount_sums_known():
    s = _stats(prompt=1_000_000, cached=500_000)
    s.add("two", "m", SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10))
    s.apply_prices({"m": P})
    s.calls[1].cost = None  # second call has no cost
    summ = s.summary()
    assert summ["cost"] is None
    assert summ["discount"] == pytest.approx(0.45)


def test_mixed_cost_source():
    s = _stats(prompt=10, completion=5, reported_cost=0.1)
    s.add("two", "m", SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    s.apply_prices(P)
    assert s.summary()["cost_source"] == "mixed"


def test_cached_tokens_both_usage_shapes():
    s = RunStats()
    s.add("a", "m", SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11,
                                    prompt_tokens_details=SimpleNamespace(cached_tokens=7)))
    s.add("b", "m", SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11,
                                    prompt_cache_hit_tokens=4))
    s.add("c", "m", SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11))
    s.add("d", "m", None)
    assert [c.cached_tokens for c in s.calls] == [7, 4, 0, 0]


def test_reported_cost_from_model_extra():
    s = RunStats()
    s.add("a", "m", SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                    model_extra={"cost": 0.25}))
    assert s.calls[0].reported_cost == 0.25


def _usage(cost_details=None, **extra):
    return SimpleNamespace(prompt_tokens=900, completion_tokens=100, total_tokens=1000,
                           model_extra={"cost_details": cost_details, **extra})


def test_reported_cost_split_preferred_over_prices():
    s = RunStats()
    # The live OpenRouter payload shape.
    s.add("a", "m", _usage({"upstream_inference_cost": 8.04e-06,
                            "upstream_inference_prompt_cost": 7.7e-06,
                            "upstream_inference_completions_cost": 3.4e-07},
                           cost=8.04e-06))
    call = s.calls[0]
    assert call.reported_input_cost == pytest.approx(7.7e-06)
    assert call.reported_output_cost == pytest.approx(3.4e-07)
    assert _input_fraction(call, P) == pytest.approx(7.7e-06 / 8.04e-06)


def test_reported_cost_split_read_from_attribute_too():
    s = RunStats()
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=100, total_tokens=1000,
                            cost_details=SimpleNamespace(upstream_inference_prompt_cost=0.2,
                                                         upstream_inference_completions_cost=0.8))
    s.add("a", "m", usage)
    assert _input_fraction(s.calls[0], P) == pytest.approx(0.2)


def test_reported_cost_split_absent_keeps_price_then_token_ratio():
    s = RunStats()
    s.add("a", "m", _usage())
    call = s.calls[0]
    assert call.reported_input_cost is None and call.reported_output_cost is None
    assert _input_fraction(call, P) == pytest.approx(900 / (900 + 400))
    assert _input_fraction(call, None) == pytest.approx(0.9)


def test_reported_cost_split_zero_sum_falls_back():
    s = RunStats()
    s.add("a", "m", _usage({"upstream_inference_prompt_cost": 0,
                            "upstream_inference_completions_cost": 0}))
    assert _input_fraction(s.calls[0], P) == pytest.approx(900 / 1300)


def test_non_numeric_reported_costs_are_ignored_without_raising():
    s = RunStats()
    s.add("a", "m", _usage({"upstream_inference_prompt_cost": "abc",
                            "upstream_inference_completions_cost": [1]},
                           cost="not-a-number"))
    call = s.calls[0]
    assert call.reported_cost is None
    assert call.reported_input_cost is None and call.reported_output_cost is None
    assert _input_fraction(call, P) == pytest.approx(900 / 1300)
    s.add("b", "m", _usage("garbage"))
    assert s.calls[1].reported_input_cost is None
    s.add("c", "m", _usage({"upstream_inference_prompt_cost": float("nan"),
                            "upstream_inference_completions_cost": 1.0}, cost=float("inf")))
    assert s.calls[2].reported_input_cost is None and s.calls[2].reported_cost is None


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_openrouter_prices_converts_per_token_to_per_million():
    def handler(request):
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [
            {"id": "x/y", "pricing": {"prompt": "0.0000003", "completion": "0.000001",
                                       "input_cache_read": "0.00000003"}},
            {"id": "other", "pricing": {"prompt": "1", "completion": "1"}},
        ]})
    out = fetch_openrouter_prices(["x/y", "missing"], "https://openrouter.ai/api/v1",
                                  client=_client(handler))
    assert set(out) == {"x/y"}
    p = out["x/y"]
    assert p.input_per_m == pytest.approx(0.3)
    assert p.output_per_m == pytest.approx(1.0)
    assert p.cached_input_per_m == pytest.approx(0.03)


def test_fetch_openrouter_prices_failure_returns_empty():
    def handler(request):
        raise httpx.ConnectError("boom")
    assert fetch_openrouter_prices(["x"], "https://openrouter.ai/api/v1",
                                   client=_client(handler)) == {}


def test_resolve_prices_env_wins(monkeypatch):
    cfg = Config(price_input_per_m=2.0, price_output_per_m=8.0)
    out = resolve_prices(["a", "b"], cfg)
    assert out["a"] == Prices(2.0, 8.0, None) and out["b"] == out["a"]


def test_resolve_prices_non_openrouter_without_env_is_none():
    cfg = Config(provider="openai", base_url="https://api.openai.com/v1",
                 price_input_per_m=None, price_output_per_m=None)
    assert resolve_prices(["a"], cfg) == {"a": None}


def test_openrouter_extra_body():
    assert openrouter_extra_body(Config(provider="openrouter")) == {"usage": {"include": True}}
    assert openrouter_extra_body(
        Config(provider="openai", base_url="https://api.openai.com/v1")) is None
