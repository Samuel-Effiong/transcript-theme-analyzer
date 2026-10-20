import pytest

from transcript_theme_analyzer.config import load_config
from transcript_theme_analyzer.themes import (
    ThemeInputError,
    build_themes,
    check_theme_limit,
    read_themes_file,
)


def test_limit_rejects_over_cap_with_exact_message():
    themes = build_themes([f"t{i}" for i in range(7)], [])
    with pytest.raises(ThemeInputError) as e:
        check_theme_limit(themes, 5)
    assert str(e.value) == ("You entered 7 themes. The limit is 5 (MAX_THEMES). "
                            "Remove 2 theme(s) or raise MAX_THEMES.")


def test_limit_allows_at_and_under_cap():
    check_theme_limit(build_themes(["a", "b", "c", "d", "e"], []), 5)
    check_theme_limit(build_themes(["a"], []), 5)


def test_unsafe_and_colliding_names_get_safe_unique_ids():
    ids = [t.id for t in build_themes(["Faith/Works", "God's Love", "Love", "love!"], [])]
    assert ids == ["faith-works", "god-s-love", "love", "love-2"]


def test_duplicate_names_case_insensitive_raise():
    with pytest.raises(ThemeInputError, match="listed more than once"):
        build_themes(["Grace", "grace"], [])


def test_no_themes_raises():
    with pytest.raises(ThemeInputError, match="No themes given"):
        build_themes([], [])


def test_empty_name_raises():
    with pytest.raises(ThemeInputError):
        build_themes(["  "], [])


def test_cli_names_come_before_file_rows_and_descriptions_kept():
    themes = build_themes(["Zeta"], [("Alpha", "first letter")])
    assert [t.name for t in themes] == ["Zeta", "Alpha"]
    assert themes[1].description == "first letter"


def test_read_themes_file_skips_blank_rows_and_keeps_descriptions(tmp_path):
    p = tmp_path / "themes.csv"
    p.write_text("Theme,Description\nGrace,Unmerited favour\n,\nFaith,\n", encoding="utf-8")
    assert read_themes_file(str(p)) == [("Grace", "Unmerited favour"), ("Faith", "")]


def test_read_themes_file_without_description_column(tmp_path):
    p = tmp_path / "themes.csv"
    p.write_text("theme\nGrace\n", encoding="utf-8")
    assert read_themes_file(str(p)) == [("Grace", "")]


def test_read_themes_file_missing_theme_column_raises(tmp_path):
    p = tmp_path / "themes.csv"
    p.write_text("Name,Description\nGrace,x\n", encoding="utf-8")
    with pytest.raises(ThemeInputError):
        read_themes_file(str(p))


def test_max_themes_clamped_to_one(monkeypatch):
    monkeypatch.setenv("MAX_THEMES", "0")
    assert load_config().max_themes == 1


def test_max_themes_default_five(monkeypatch):
    monkeypatch.delenv("MAX_THEMES", raising=False)
    assert load_config().max_themes == 5


def test_price_empty_or_unset_is_none(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_M", "")
    monkeypatch.delenv("LLM_PRICE_CACHED_INPUT_PER_M", raising=False)
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_M", "0.5")
    cfg = load_config()
    assert cfg.price_input_per_m is None
    assert cfg.price_cached_input_per_m is None
    assert cfg.price_output_per_m == 0.5
