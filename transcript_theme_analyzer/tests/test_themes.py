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
    assert ids == ["faith-works", "gods-love", "love", "love-2"]


def test_duplicate_names_case_insensitive_raise():
    with pytest.raises(ThemeInputError, match="listed more than once"):
        build_themes(["Grace", "grace"], [])


def test_no_themes_raises():
    with pytest.raises(ThemeInputError, match="No themes given"):
        build_themes([], [])


def test_empty_name_raises():
    with pytest.raises(ThemeInputError):
        build_themes(["  "], [])





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


def test_internal_whitespace_and_newlines_collapsed():
    from transcript_theme_analyzer.prompts import format_theme_block

    themes = build_themes(["the  glory\nof God"], ["Grace is unmerited\r\n\n favour\tgiven freely to all"])
    assert themes[0].text == "the glory of God"
    assert themes[1].text == "Grace is unmerited favour given freely to all"
    assert themes[1].name == "Grace is unmerited favour given freely"
    assert len(format_theme_block(themes).splitlines()) == 2


def test_names_equal_after_collapsing_are_duplicates():
    with pytest.raises(ThemeInputError, match="listed more than once"):
        build_themes(["Grace  Alone", "grace alone"], [])


from transcript_theme_analyzer.themes import Theme, derive_label  # noqa: E402

LONG = ("Teachings about God's will for an individual's personal decisions, life direction, "
        "and calling. Exclude discussions of God's sovereign plan for historical events.")


def test_theme_text_defaults_to_name():
    t = Theme("a", "Alpha")
    assert t.text == "Alpha"
    assert not hasattr(t, "description")


def test_derive_label_first_six_words_without_trailing_punctuation():
    assert derive_label(LONG) == "Teachings about God's will for an"
    assert derive_label("will of God") == "will of God"
    assert derive_label("Grace, mercy, and peace to you all,") == "Grace, mercy, and peace to you"
    assert derive_label("Faith. Hope. Love.") == "Faith. Hope. Love"


def test_long_theme_gets_label_and_short_id():
    (t,) = build_themes([LONG], [])
    assert t.text == LONG
    assert t.name == "Teachings about God's will for an"
    assert t.id == "teachings-about-gods-will-for-an"


def test_themes_sharing_first_six_words_get_distinct_labels_and_ids():
    a, b = build_themes([LONG, LONG.replace("Exclude", "Include")], [])
    assert (a.name, b.name) == ("Teachings about God's will for an", "Teachings about God's will for an (2)")
    assert (a.id, b.id) == ("teachings-about-gods-will-for-an", "teachings-about-gods-will-for-an-2")


def test_cli_texts_come_before_file_texts():
    assert [t.text for t in build_themes(["Zeta"], ["Alpha"])] == ["Zeta", "Alpha"]


def test_read_themes_file_single_column_skips_blank_rows(tmp_path):
    p = tmp_path / "themes.csv"
    p.write_text('Theme\n"will of God"\n\n"Teachings about God\'s will,\nacross lines"\n', encoding="utf-8")
    assert read_themes_file(str(p)) == ["will of God", "Teachings about God's will,\nacross lines"]


def test_read_themes_file_rejects_description_column(tmp_path):
    p = tmp_path / "themes.csv"
    p.write_text("theme,description\nGrace,unmerited favour\n", encoding="utf-8")
    with pytest.raises(ThemeInputError) as exc:
        read_themes_file(str(p))
    assert str(exc.value) == (
        f"{p} has a 'description' column. Put the full theme text in the 'theme' column "
        "instead; a theme may be a phrase or a whole paragraph."
    )
