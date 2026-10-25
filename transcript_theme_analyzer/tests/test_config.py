from transcript_theme_analyzer.config import Config


def test_docx_volume_words_defaults_to_150000(monkeypatch):
    monkeypatch.delenv("DOCX_VOLUME_WORDS", raising=False)
    assert Config().docx_volume_words == 150_000


def test_docx_volume_words_reads_env(monkeypatch):
    monkeypatch.setenv("DOCX_VOLUME_WORDS", "20000")
    assert Config().docx_volume_words == 20_000


def test_validated_clamps_docx_volume_words_to_1000():
    assert Config(docx_volume_words=0).validated().docx_volume_words == 1000
    assert Config(docx_volume_words=-5).validated().docx_volume_words == 1000
    assert Config(docx_volume_words=1000).validated().docx_volume_words == 1000
    assert Config(docx_volume_words=5000).validated().docx_volume_words == 5000
