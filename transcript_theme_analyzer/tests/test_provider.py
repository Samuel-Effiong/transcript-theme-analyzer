"""Provider-compatibility tests: strict-schema generation and the response
shapes that OpenAI-compatible aggregators return as HTTP 200."""
from __future__ import annotations

import types

import pytest

from transcript_theme_analyzer.analyzer import (
    TruncatedResponseError,
    _extract_content,
    _json_schema_for,
    _repair_parsed_json,
    _strictify,
)
from transcript_theme_analyzer.config import Config
from transcript_theme_analyzer.schema import ChunkAnalysis


def _choice(content, finish_reason="stop", extra=None):
    message = types.SimpleNamespace(content=content, model_extra=extra or {})
    return types.SimpleNamespace(message=message, finish_reason=finish_reason)


def _resp(choices, error=None):
    return types.SimpleNamespace(choices=choices, error=error, model_extra={})


# --- strict schema ---------------------------------------------------------

def test_every_property_is_required_and_objects_are_closed():
    schema = _json_schema_for(ChunkAnalysis, "chunk_analysis")["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])

    marker = schema["$defs"]["LocationMarker"]
    assert marker["additionalProperties"] is False
    # Optional fields must still be required; optionality is the null union.
    assert set(marker["required"]) == set(marker["properties"])
    assert {"type": "null"} in marker["properties"]["timestamp"]["anyOf"]


def test_unsupported_validation_keywords_are_stripped():
    schema = _json_schema_for(ChunkAnalysis, "chunk_analysis")["schema"]
    score = schema["properties"]["relevance_score"]
    assert "minimum" not in score and "maximum" not in score
    assert "default" not in schema["$defs"]["LocationMarker"]["properties"]["title"]


def test_strictify_recurses_into_nested_arrays_and_unions():
    nested = _strictify({
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {
                "type": "object", "properties": {"a": {"type": "string", "minLength": 2}},
            }},
            "either": {"anyOf": [{"type": "object", "properties": {"b": {"type": "string"}}}]},
        },
    })
    inner = nested["properties"]["items"]["items"]
    assert inner["additionalProperties"] is False and inner["required"] == ["a"]
    assert "minLength" not in inner["properties"]["a"]
    union = nested["properties"]["either"]["anyOf"][0]
    assert union["additionalProperties"] is False and union["required"] == ["b"]


def test_strictify_does_not_mutate_the_input():
    original = {"type": "object", "properties": {"a": {"type": "string", "default": "x"}}}
    _strictify(original)
    assert original["properties"]["a"]["default"] == "x"
    assert "required" not in original


# --- response shapes -------------------------------------------------------

def test_plain_content_is_returned():
    assert _extract_content(_resp([_choice(' {"a": 1} ')]), "m") == '{"a": 1}'


def test_markdown_fenced_json_is_unwrapped():
    assert _extract_content(_resp([_choice('```json\n{"a": 1}\n```')]), "m") == '{"a": 1}'


def test_error_payload_on_a_200_is_raised():
    resp = _resp([], error={"message": "upstream provider is down"})
    with pytest.raises(RuntimeError, match="upstream provider is down"):
        _extract_content(resp, "m")


def test_empty_choices_is_raised_clearly():
    with pytest.raises(RuntimeError, match="no choices"):
        _extract_content(_resp([]), "m")


def test_truncated_response_says_so_instead_of_failing_to_parse():
    resp = _resp([_choice('{"relevance_score": 4', finish_reason="length")])
    with pytest.raises(TruncatedResponseError, match="output cap"):
        _extract_content(resp, "m")


def test_reasoning_only_response_is_reported_as_truncation():
    resp = _resp([_choice("", finish_reason="length", extra={"reasoning": "thinking..."})])
    with pytest.raises(TruncatedResponseError, match="reasoning but no answer"):
        _extract_content(resp, "m")


def test_missing_finish_reason_is_tolerated():
    """Some OpenAI-compatible providers omit finish_reason entirely."""
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content='{"a": 1}'))
    assert _extract_content(_resp([choice]), "m") == '{"a": 1}'


# --- score clamping (the bound is no longer enforced on the wire) ----------

@pytest.mark.parametrize("given,expected", [(150, 100), (-5, 0), (87.6, 88), (42, 42)])
def test_out_of_range_scores_are_clamped(given, expected):
    parsed = _repair_parsed_json({"relevance_score": given}, ChunkAnalysis)
    assert parsed["relevance_score"] == expected


def test_boolean_score_is_not_treated_as_a_number():
    parsed = _repair_parsed_json({"relevance_score": True}, ChunkAnalysis)
    assert parsed["relevance_score"] is True  # left for validation to reject


# --- provider wiring -------------------------------------------------------

def test_openrouter_is_detected_from_base_url_or_provider():
    assert Config(base_url="https://openrouter.ai/api/v1", provider="x").is_openrouter
    assert Config(base_url="https://example.com", provider="openrouter").is_openrouter
    assert not Config(base_url="https://api.openai.com/v1", provider="openai").is_openrouter


def test_attribution_headers_only_sent_to_openrouter():
    from transcript_theme_analyzer.client import make_client

    router = make_client(Config(provider="openrouter", base_url="https://openrouter.ai/api/v1",
                                api_key="k", openrouter_app_name="app").validated())
    assert router.default_headers.get("X-Title") == "app"

    other = make_client(Config(provider="openai", base_url="https://api.openai.com/v1",
                               api_key="k").validated())
    assert "X-Title" not in (other.default_headers or {})
