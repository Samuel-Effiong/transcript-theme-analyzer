import os

from transcript_theme_analyzer.cli import _discover_transcript_paths, parse_args


def test_discover_transcript_paths_recursive_and_filtering(tmp_path):
    # Setup directory structure with subfolders, hidden files, and temp lock files
    sub1 = tmp_path / "folder_a"
    sub2 = tmp_path / "folder_b" / "subfolder"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    # Valid transcripts
    (tmp_path / "root.txt").write_text("root txt")
    (sub1 / "file1.docx").write_text("docx 1")
    (sub2 / "file2.DOCX").write_text("docx 2 uppercase")
    (sub2 / "file3.Txt").write_text("txt 3 mixedcase")

    # Files that should be ignored
    (tmp_path / ".hidden.txt").write_text("hidden")
    (sub1 / "._file1.docx").write_text("mac OS metadata")
    (sub1 / "~$file1.docx").write_text("word lock file")
    (sub2 / "notes.pdf").write_text("unsupported pdf")

    discovered = _discover_transcript_paths(str(tmp_path), glob_pattern=None)

    relative_discovered = [os.path.relpath(p, str(tmp_path)) for p in discovered]

    assert relative_discovered == sorted([
        "folder_b/subfolder/file2.DOCX",
        "folder_b/subfolder/file3.Txt",
        "folder_a/file1.docx",
        "root.txt",
    ])


def test_theme_is_repeatable_and_optional():
    args = parse_args(["--transcript", "t.txt", "--theme", "Grace", "--theme", "Hope"])
    assert args.theme == ["Grace", "Hope"]
    assert args.themes_file is None
    assert args.max_themes is None

    args = parse_args(["--transcript", "t.txt", "--themes-file", "themes.csv", "--max-themes", "8"])
    assert args.theme == []
    assert args.themes_file == "themes.csv"
    assert args.max_themes == 8


def test_docx_volume_words_flag_parses_and_defaults_to_none():
    assert parse_args(["--transcript", "t.txt", "--theme", "G"]).docx_volume_words is None
    assert parse_args(["--transcript", "t.txt", "--theme", "G", "--docx-volume-words", "2000"]).docx_volume_words == 2000


def test_compare_models_rejects_over_limit_themes(tmp_path, monkeypatch, capsys):
    import pytest

    from transcript_theme_analyzer.eval import compare_models

    transcript = tmp_path / "t.txt"
    transcript.write_text("hello", encoding="utf-8")

    def boom(*_a, **_k):
        raise AssertionError("no API access before the theme limit check")

    monkeypatch.setattr(compare_models, "make_client", boom)
    monkeypatch.setattr(compare_models, "resolve_prices", boom)
    argv = ["--transcript", str(transcript), "--models", "m1", "--max-themes", "1",
            "--theme", "a", "--theme", "b"]
    with pytest.raises(SystemExit) as exc:
        compare_models.main(argv)
    assert exc.value.code == 2
    assert "You entered 2 themes. The limit is 1 (MAX_THEMES)." in capsys.readouterr().err


def test_compare_models_prints_row_per_theme_and_totals(tmp_path, monkeypatch, capsys):
    from transcript_theme_analyzer.eval import compare_models

    transcript = tmp_path / "t.txt"
    transcript.write_text("hello", encoding="utf-8")

    class FakeClient:
        closed = False

        async def close(self):
            FakeClient.closed = True

    async def fake_run_one_model(text, themes, model, config, client, label="", gate=None, prices=None):
        return {
            "themes": {
                t.id: {"theme": t.name, "relevance_score": 7, "locations": [{}, {}]}
                for t in themes
            },
            "_meta": {
                "elapsed_seconds": 1.0,
                "usage": {"total_tokens": 1234, "cost": None},
            },
        }

    monkeypatch.setattr(compare_models, "make_client", lambda config: FakeClient())
    monkeypatch.setattr(compare_models, "resolve_prices", lambda models, config: {})
    monkeypatch.setattr(compare_models, "run_one_model", fake_run_one_model)
    compare_models.main(["--transcript", str(transcript), "--models", "m1", "m2",
                         "--theme", "forgiveness", "--theme", "prayer"])
    out = capsys.readouterr().out
    assert FakeClient.closed
    assert out.count("forgiveness") >= 2 and out.count("prayer") >= 2
    assert "1,234" in out
    assert "n/a" in out
