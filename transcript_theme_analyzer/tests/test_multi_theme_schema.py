import asyncio
import json
from types import SimpleNamespace

import pytest

from transcript_theme_analyzer import prompts
from transcript_theme_analyzer.analyzer import (
    _call_with_retry,
    _repair_multi_theme,
    _strictify,
)
from transcript_theme_analyzer.prompts import (
    MULTI_SINGLE_PASS_USER_TEMPLATE,
    format_theme_block,
)
from transcript_theme_analyzer.schema import (
    MultiThemeChunkAnalysis,
    MultiThemeSynthesis,
    ThemeChunkAnalysis,
    ThemeSynthesis,
)
from transcript_theme_analyzer.themes import Theme
from transcript_theme_analyzer.analyzer import MissingThemesError

THEMES = [Theme("grace", "Grace"), Theme("forgiveness", "Forgiveness", "letting go")]


def _entry(theme_id, score=50):
    return {
        "theme_id": theme_id,
        "relevance_score": score,
        "explicitness": "explicit",
        "reasoning": "r",
        "locations": [],
    }


def test_theme_matched_by_name_or_case():
    parsed = {"themes": [_entry("GRACE"), _entry("Forgiveness")]}
    out = _repair_multi_theme(parsed, THEMES)
    assert [e["theme_id"] for e in out["themes"]] == ["grace", "forgiveness"]


def test_missing_theme_raises():
    with pytest.raises(MissingThemesError, match="forgiveness"):
        _repair_multi_theme({"themes": [_entry("grace")]}, THEMES)


def test_missing_themes_message_lists_all():
    with pytest.raises(MissingThemesError, match="missing themes: grace, forgiveness"):
        _repair_multi_theme({"themes": []}, THEMES)


def test_unknown_and_duplicate_entries_dropped_and_order_restored():
    parsed = {"themes": [_entry("forgiveness"), _entry("hope"), _entry("grace"), _entry("grace", score=1)]}
    out = _repair_multi_theme(parsed, THEMES)
    assert [e["theme_id"] for e in out["themes"]] == ["grace", "forgiveness"]
    assert out["themes"][0]["relevance_score"] != 1


def test_entry_score_clamped_and_rounded():
    parsed = {"themes": [_entry("grace", score=88.4), _entry("forgiveness", score=-3)]}
    out = _repair_multi_theme(parsed, THEMES)
    assert out["themes"][0]["relevance_score"] == 88
    assert out["themes"][1]["relevance_score"] == 0
    MultiThemeChunkAnalysis.model_validate(out)


def test_synthesis_entries_repaired_with_entry_cls():
    parsed = {"themes": [{"theme_id": "Forgiveness", "reasoning": "x"}, {"theme_id": "grace"}]}
    out = _repair_multi_theme(parsed, THEMES, entry_cls=ThemeSynthesis)
    assert [e["theme_id"] for e in out["themes"]] == ["grace", "forgiveness"]
    assert out["themes"][0]["reasoning"] == ""
    MultiThemeSynthesis.model_validate(out)


def test_non_dict_passes_through():
    assert _repair_multi_theme([1], THEMES) == [1]


def test_theme_block_format():
    assert format_theme_block(THEMES) == "- id: grace | Grace\n- id: forgiveness | Forgiveness — letting go"


def test_theme_block_precedes_transcript():
    u = MULTI_SINGLE_PASS_USER_TEMPLATE.format(theme_block="TB", transcript="TX")
    assert u.index("TB") < u.index("TX")
    c = prompts.MULTI_CHUNK_USER_TEMPLATE.format(
        theme_block="TB", chunk_index=1, total_chunks=2, char_start=0, char_end=9, chunk_text="TX"
    )
    assert c.index("TB") < c.index("TX")


def test_prompt_constants():
    assert prompts.PROMPT_VERSION == "multi-v2"
    assert "THEMES list" in prompts.MULTI_THEME_ANALYSIS_SYSTEM_PROMPT_V2
    assert "more than one theme" in prompts.MULTI_THEME_ANALYSIS_SYSTEM_PROMPT_V2
    assert '"themes"' in prompts.MULTI_THEME_SYNTHESIS_SYSTEM_PROMPT_V2


def test_strictify_nested_entry_has_additional_properties_false():
    schema = _strictify(MultiThemeChunkAnalysis.model_json_schema())
    entry = schema["$defs"]["ThemeChunkAnalysis"]
    assert entry["additionalProperties"] is False
    assert "theme_id" in entry["required"]


def test_theme_chunk_analysis_extends_chunk_analysis():
    t = ThemeChunkAnalysis(theme_id="a", relevance_score=1, explicitness="absent", reasoning="r")
    assert t.locations == []


class _Completions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content=json.dumps({"themes": [_entry("grace")]}), model_extra={})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None
        )


def _client():
    comp = _Completions()
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


def _run(client, **kw):
    return asyncio.run(
        _call_with_retry(
            client,
            model="m-extra-body",
            system="s",
            user="u",
            schema_cls=MultiThemeChunkAnalysis,
            schema_name="n",
            max_retries=1,
            max_output_tokens=None,
            **kw,
        )
    )


def test_extra_body_and_custom_repair_reach_call():
    client, comp = _client()
    seen = []

    def repair(p):
        seen.append(p)
        return p

    result, _ = _run(client, repair=repair, extra_body={"provider": {"x": 1}})
    assert comp.calls[0]["extra_body"] == {"provider": {"x": 1}}
    assert seen and result.themes[0].theme_id == "grace"


def test_extra_body_omitted_leaves_request_unchanged():
    client, comp = _client()
    _run(client)
    assert comp.calls[0].get("extra_body") is None
