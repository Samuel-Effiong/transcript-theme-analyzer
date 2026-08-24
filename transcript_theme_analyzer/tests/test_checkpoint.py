import json
import os

from transcript_theme_analyzer.checkpoint import (
    CheckpointWriter,
    checkpoint_path,
    compact,
    has_error,
    read_checkpoint,
)

THEME = "the glory of God"
MODELS = ["model-a"]


def _payload(score=42):
    return {"relevance_score": score, "reasoning": "r", "locations": []}


def test_round_trip(tmp_path):
    path = checkpoint_path(str(tmp_path))
    with CheckpointWriter(path, THEME, MODELS) as writer:
        writer.record("a.txt", "a", [("model-a", _payload())])
        writer.record("sub/b.txt", "sub/b", [("model-a", _payload(7))])

    done = read_checkpoint(path, THEME, MODELS)
    assert set(done) == {"a.txt", "sub/b.txt"}
    assert done["a.txt"]["display_name"] == "a"
    assert done["sub/b.txt"]["results"][0][1]["relevance_score"] == 7


def test_ignores_other_theme_and_model_set(tmp_path):
    path = checkpoint_path(str(tmp_path))
    with CheckpointWriter(path, THEME, MODELS) as writer:
        writer.record("a.txt", "a", [("model-a", _payload())])

    assert read_checkpoint(path, "a different theme", MODELS) == {}
    assert read_checkpoint(path, THEME, ["model-b"]) == {}
    # Model order must not matter -- the same set is the same work.
    with CheckpointWriter(path, THEME, ["y", "x"]) as writer:
        writer.record("b.txt", "b", [("x", _payload())])
    assert "b.txt" in read_checkpoint(path, THEME, ["x", "y"])


def test_survives_torn_final_line(tmp_path):
    """The realistic crash: the process dies mid-write. The good records
    before it must still load."""
    path = checkpoint_path(str(tmp_path))
    with CheckpointWriter(path, THEME, MODELS) as writer:
        writer.record("a.txt", "a", [("model-a", _payload())])
        writer.record("b.txt", "b", [("model-a", _payload())])

    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"signature": "x", "key": "c.txt", "resu')  # truncated

    done = read_checkpoint(path, THEME, MODELS)
    assert set(done) == {"a.txt", "b.txt"}


def test_later_record_supersedes_earlier(tmp_path):
    """A retried failure must overwrite its own error entry, not sit
    alongside it."""
    path = checkpoint_path(str(tmp_path))
    with CheckpointWriter(path, THEME, MODELS) as writer:
        writer.record("a.txt", "a", [("model-a", {"model": "model-a", "error": "boom"})])
        writer.record("a.txt", "a", [("model-a", _payload(55))])

    done = read_checkpoint(path, THEME, MODELS)
    assert len(done) == 1
    assert not has_error(done["a.txt"]["results"])
    assert done["a.txt"]["results"][0][1]["relevance_score"] == 55


def test_missing_file_is_empty(tmp_path):
    assert read_checkpoint(str(tmp_path / "nope.jsonl"), THEME, MODELS) == {}


def test_has_error():
    assert has_error([("m", {"error": "x"})])
    assert not has_error([("m", _payload())])
    assert has_error([("m", _payload()), ("n", {"error": "x"})])


def test_appends_across_sessions(tmp_path):
    """Resume relies on a second process appending to the same file."""
    path = checkpoint_path(str(tmp_path))
    with CheckpointWriter(path, THEME, MODELS) as writer:
        writer.record("a.txt", "a", [("model-a", _payload())])
    with CheckpointWriter(path, THEME, MODELS) as writer:
        writer.record("b.txt", "b", [("model-a", _payload())])
    assert set(read_checkpoint(path, THEME, MODELS)) == {"a.txt", "b.txt"}


def test_compact_dedupes_and_is_atomic(tmp_path):
    path = checkpoint_path(str(tmp_path))
    with CheckpointWriter(path, THEME, MODELS) as writer:
        for _ in range(3):
            writer.record("a.txt", "a", [("model-a", _payload())])
        writer.record("b.txt", "b", [("model-a", _payload())])

    assert compact(path, THEME, MODELS) == 2
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert len(lines) == 2
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert set(read_checkpoint(path, THEME, MODELS)) == {"a.txt", "b.txt"}


def test_unicode_survives(tmp_path):
    path = checkpoint_path(str(tmp_path))
    payload = {"relevance_score": 1, "reasoning": "café — ✝ 你好", "locations": []}
    with CheckpointWriter(path, THEME, MODELS) as writer:
        writer.record("é.txt", "é", [("model-a", payload)])
    done = read_checkpoint(path, THEME, MODELS)
    assert done["é.txt"]["results"][0][1]["reasoning"] == "café — ✝ 你好"
