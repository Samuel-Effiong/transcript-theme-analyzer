import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from transcript_theme_analyzer import analyzer, prompts
from transcript_theme_analyzer.aggregate import compute_aggregate_score
from transcript_theme_analyzer.analyzer import _extract_and_fill_location, _repair_parsed_json
from transcript_theme_analyzer.chunker import Chunk, chunk_transcript
from transcript_theme_analyzer.config import Config
from transcript_theme_analyzer.schema import AnalysisResult, ChunkAnalysis, LocationMarker
from transcript_theme_analyzer.themes import Theme


def make_chunk(text="...", char_start=1000, char_end=1200, timestamp="[10:00]", speaker="Host"):
    return Chunk(
        index=0,
        char_start=char_start,
        char_end=char_end,
        text=text,
        nearest_timestamp=timestamp,
        nearest_speaker=speaker,
    )


def test_extract_and_fill_location_extracts_the_passage_between_markers():
    chunk = make_chunk(text="Intro. The passage begins here and keeps going until it ends right here. Outro.")
    marker = LocationMarker(start_marker="The passage begins here", end_marker="it ends right here")
    result = _extract_and_fill_location(marker, chunk)
    assert result is not None
    assert result.excerpt == "The passage begins here and keeps going until it ends right here"


def test_extract_and_fill_location_fills_timestamp_speaker_from_chunk_when_missing():
    chunk = make_chunk(text="Some text with a start point and an end point in it.", timestamp="[05:00]", speaker="Guest")
    marker = LocationMarker(start_marker="start point", end_marker="end point")
    result = _extract_and_fill_location(marker, chunk)
    assert result is not None
    assert result.timestamp == "[05:00]"
    assert result.speaker == "Guest"


def test_extract_and_fill_location_keeps_markers_own_timestamp_speaker_when_present():
    chunk = make_chunk(text="Some text with a start point and an end point in it.", timestamp="[05:00]", speaker="Guest")
    marker = LocationMarker(start_marker="start point", end_marker="end point", timestamp="[01:00]", speaker="Host")
    result = _extract_and_fill_location(marker, chunk)
    assert result is not None
    assert result.timestamp == "[01:00]"
    assert result.speaker == "Host"


def test_extract_and_fill_location_preserves_title():
    chunk = make_chunk(text="Some text with a start point and an end point in it.")
    marker = LocationMarker(start_marker="start point", end_marker="end point", title="A Title")
    result = _extract_and_fill_location(marker, chunk)
    assert result is not None
    assert result.title == "A Title"


def test_extract_and_fill_location_returns_none_when_marker_not_found():
    chunk = make_chunk(text="This text does not contain the marker phrase at all.")
    marker = LocationMarker(start_marker="nonexistent phrase", end_marker="also nonexistent")
    assert _extract_and_fill_location(marker, chunk) is None


def test_repair_fills_missing_required_reasoning_with_empty_string():
    parsed = {"relevance_score": 5, "explicitness": "tangential", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert result.reasoning == ""
    assert result.relevance_score == 5


def test_repair_rounds_and_clamps_a_fractional_relevance_score():
    parsed = {"relevance_score": 0.3, "explicitness": "tangential", "reasoning": "r", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    assert repaired["relevance_score"] == 0

    parsed = {"relevance_score": 87.6, "explicitness": "explicit", "reasoning": "r", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    assert repaired["relevance_score"] == 88


def test_repair_drops_location_entries_missing_start_or_end_marker():
    parsed = {
        "relevance_score": 40,
        "explicitness": "explicit",
        "reasoning": "r",
        "locations": [
            {"start_marker": "a start", "end_marker": "an end"},
            {"start_marker": "only a start, should be dropped"},
            {"title": "no markers at all, should be dropped"},
        ],
    }
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert len(result.locations) == 1
    assert result.locations[0].start_marker == "a start"


def test_repair_does_not_touch_a_fully_populated_response():
    parsed = {"relevance_score": 42, "explicitness": "explicit", "reasoning": "solid reasoning", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    assert repaired == parsed


def test_repair_defaults_missing_explicitness_to_absent():
    parsed = {"relevance_score": 5, "reasoning": "r", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert result.explicitness == "absent"


def test_repair_coerces_out_of_enum_explicitness_to_absent():
    parsed = {"relevance_score": 5, "explicitness": "somewhat", "reasoning": "r", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert result.explicitness == "absent"


def test_repair_keeps_valid_explicitness_values_unchanged():
    for value in ("explicit", "tangential", "absent"):
        parsed = {"relevance_score": 5, "explicitness": value, "reasoning": "r", "locations": []}
        repaired = _repair_parsed_json(parsed, ChunkAnalysis)
        assert repaired["explicitness"] == value


def test_repair_fills_missing_reasoning_on_analysis_result_too():
    parsed = {
        "theme": "t",
        "relevance_score": 10,
        "locations": [],
        "model_used": "m1",
        "chunked": False,
    }
    repaired = _repair_parsed_json(parsed, AnalysisResult)
    result = AnalysisResult.model_validate(repaired)
    assert result.reasoning == ""


# --- Multi-theme pipeline -------------------------------------------------

_ID_LINE = re.compile(r"^- id: (\S+)", re.MULTILINE)
_CHUNK_LINE = re.compile(r"chunk (\d+) of \d+")


class _FakeCompletions:
    """Answers every theme listed as ``- id: X`` in the last user message.

    ``entry(theme_id, chunk_number)`` builds an analysis entry (chunk_number
    is None for single pass). ``omit`` theme ids are left out of analysis
    answers for the first ``omit_attempts`` calls (None = always).
    """

    def __init__(self, entry=None, omit=(), omit_attempts=None):
        self.entry = entry or (lambda tid, n: {
            "theme_id": tid, "relevance_score": 10, "explicitness": "absent",
            "reasoning": f"r {tid}", "locations": [],
        })
        self.omit = set(omit)
        self.omit_attempts = omit_attempts
        self.calls = 0
        self.requests = []

    async def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        user = [m for m in kwargs["messages"] if m["role"] == "user"][-1]["content"]
        system = kwargs["messages"][0]["content"]
        ids = _ID_LINE.findall(user)
        if system == prompts.MULTI_THEME_SYNTHESIS_SYSTEM_PROMPT_V2:
            body = {"themes": [{"theme_id": t, "reasoning": f"synth {t}"} for t in ids]}
        else:
            m = _CHUNK_LINE.search(user)
            n = int(m.group(1)) if m else None
            omitting = self.omit_attempts is None or self.calls <= self.omit_attempts
            body = {"themes": [
                self.entry(t, n) for t in ids if not (omitting and t in self.omit)
            ]}
        msg = SimpleNamespace(content=json.dumps(body), model_extra={})
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage
        )


def _fake_client(**kw):
    comp = _FakeCompletions(**kw)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    analyzer._NO_STRUCTURED_OUTPUT.clear()
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_k: real_sleep(0))


THEMES = [Theme("a", "Alpha"), Theme("b", "Beta"), Theme("c", "Gamma")]


def _cfg(**overrides):
    defaults = dict(provider="openai", base_url="https://api.openai.com/v1", max_retries=3)
    defaults.update(overrides)
    return Config(**defaults).validated()


def test_single_pass_one_call_all_themes_in_order_with_passage():
    transcript = "Intro. The passage begins here and keeps going until it ends right here. Outro."

    def entry(tid, n):
        locations = []
        if tid == "b":
            locations = [{"start_marker": "The passage begins here", "end_marker": "it ends right here"}]
        return {"theme_id": tid, "relevance_score": 50, "explicitness": "explicit",
                "reasoning": f"r {tid}", "locations": locations}

    client, comp = _fake_client(entry=entry)
    results, stats = asyncio.run(analyzer.analyze(client, "m", THEMES, transcript, _cfg()))

    assert comp.calls == 1
    assert [r.theme_id for r in results] == ["a", "b", "c"]
    assert [r.theme for r in results] == ["Alpha", "Beta", "Gamma"]
    assert all(not r.chunked and r.explicitness == "explicit" for r in results)
    assert results[0].locations == [] and results[2].locations == []
    assert results[1].locations[0].excerpt == (
        "The passage begins here and keeps going until it ends right here"
    )
    assert stats.calls[0].theme_ids == ["a", "b", "c"]


def test_chunked_one_call_per_chunk_plus_one_synthesis_and_per_theme_scores():
    transcript = "word " * 180  # 900 chars
    cfg = _cfg(single_pass_token_limit=10, chunk_size_tokens=100, chunk_overlap_tokens=10)
    chunks = chunk_transcript(transcript, cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
    assert len(chunks) == 3

    scores = {"a": [90, 10, 0], "b": [5, 40, 70]}
    explicitness = {"a": ["explicit", "tangential", "absent"], "b": ["absent", "tangential", "tangential"]}

    def entry(tid, n):
        return {"theme_id": tid, "relevance_score": scores[tid][n - 1],
                "explicitness": explicitness[tid][n - 1], "reasoning": f"{tid}{n}",
                "locations": []}

    client, comp = _fake_client(entry=entry)
    themes = THEMES[:2]
    results, stats = asyncio.run(analyzer.analyze(client, "m", themes, transcript, cfg))

    assert comp.calls == 3 + 1
    lengths = [c.char_end - c.char_start for c in chunks]
    for result in results:
        own = [
            ChunkAnalysis(relevance_score=s, explicitness=e, reasoning="")
            for s, e in zip(scores[result.theme_id], explicitness[result.theme_id])
        ]
        assert result.relevance_score == compute_aggregate_score(own, lengths)
        assert result.reasoning == f"synth {result.theme_id}"
        assert result.chunked
    assert [r.explicitness for r in results] == ["explicit", "tangential"]

    synthesis_user = comp.requests[-1]["messages"][-1]["content"]
    assert "Segment 1" in synthesis_user and "a1" in synthesis_user and "b3" in synthesis_user


def test_theme_missing_on_every_attempt_raises_after_max_retries():
    client, comp = _fake_client(omit={"b"})
    with pytest.raises(RuntimeError, match="missing themes: b"):
        asyncio.run(analyzer.analyze(client, "m", THEMES, "short transcript", _cfg(max_retries=3)))
    assert comp.calls == 3


def test_theme_missing_on_first_attempt_only_succeeds_on_retry():
    client, comp = _fake_client(omit={"b"}, omit_attempts=1)
    results, _ = asyncio.run(analyzer.analyze(client, "m", THEMES, "short transcript", _cfg()))
    assert comp.calls == 2
    assert [r.theme_id for r in results] == ["a", "b", "c"]


def test_theme_usage_has_every_theme():
    client, _ = _fake_client()
    _, stats = asyncio.run(analyzer.analyze(client, "m", THEMES, "short transcript", _cfg()))
    assert set(stats.theme_usage()) == {"a", "b", "c"}

    transcript = "word " * 180
    cfg = _cfg(single_pass_token_limit=10, chunk_size_tokens=100, chunk_overlap_tokens=10)
    client, _ = _fake_client()
    _, stats = asyncio.run(analyzer.analyze(client, "m", THEMES, transcript, cfg))
    assert set(stats.theme_usage()) == {"a", "b", "c"}
    assert all(c.theme_ids == ["a", "b", "c"] for c in stats.calls)
