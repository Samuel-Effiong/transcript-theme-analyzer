"""End-to-end batch behaviour: resume, concurrency, and rate-limit handling.

Every test here stubs the network layer -- what's under test is the
orchestration around the API calls, not the calls themselves.
"""
import asyncio
import json
import re
import types

import httpx
import openai
import pytest

from transcript_theme_analyzer import analyzer, cli, prompts
from transcript_theme_analyzer.cache import read_cache
from transcript_theme_analyzer.checkpoint import CheckpointWriter, checkpoint_path, read_checkpoint
from transcript_theme_analyzer.config import Config
from transcript_theme_analyzer.pricing import Prices
from transcript_theme_analyzer.schema import ChunkAnalysis
from transcript_theme_analyzer.themes import Theme

THEME = "the glory of God"
THEMES = [Theme("glory", THEME)]
_ID_LINE = re.compile(r"^- id: (\S+)", re.MULTILINE)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """These are process-global caches; a leak between tests would make
    results depend on test order."""
    analyzer._NO_STRUCTURED_OUTPUT.clear()
    analyzer._gate = None
    analyzer._gate_loop = None
    yield
    analyzer._NO_STRUCTURED_OUTPUT.clear()
    analyzer._gate = None
    analyzer._gate_loop = None


@pytest.fixture(autouse=True)
def _no_price_lookup(monkeypatch):
    """resolve_prices may GET OpenRouter's catalogue; tests stay offline."""
    calls = []

    def fake(models, config):
        calls.append(list(models))
        return {m: None for m in models}

    monkeypatch.setattr(cli, "resolve_prices", fake)
    return calls


def _config(**overrides):
    defaults = dict(
        max_concurrent_chunks=4,
        max_concurrent_transcripts=4,
        max_concurrent_requests=8,
        max_retries=3,
        single_pass_token_limit=10**9,  # force the single-pass path: 1 call each
    )
    defaults.update(overrides)
    return Config(**defaults).validated()


def _write_transcripts(tmp_path, names, body="Speaker: a line about the theme.\n"):
    directory = tmp_path / "transcripts"
    directory.mkdir(exist_ok=True)
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return str(directory)


class FakeClient:
    """Records calls and tracks how many ran concurrently. Answers every
    theme listed as ``- id: X`` in the user message, like a real model."""

    def __init__(self, delay=0.01, fail_for=(), score=42):
        self.delay = delay
        self.fail_for = set(fail_for)
        self.score = score
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            user = kwargs["messages"][-1]["content"]
            for marker in self.fail_for:
                if marker in user:
                    raise RuntimeError(f"synthetic failure for {marker}")
            ids = _ID_LINE.findall(user)
            if kwargs["messages"][0]["content"] == prompts.MULTI_THEME_SYNTHESIS_SYSTEM_PROMPT_V2:
                body = {"themes": [{"theme_id": t, "reasoning": "because"} for t in ids]}
            else:
                body = {"themes": [
                    {
                        "theme_id": t,
                        "relevance_score": self.score,
                        "reasoning": "because",
                        "explicitness": "tangential",
                        "locations": [],
                    }
                    for t in ids
                ]}
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=json.dumps(body), model_extra={}),
                    finish_reason="stop",
                )],
                usage=types.SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    prompt_tokens_details=types.SimpleNamespace(cached_tokens=4),
                ),
            )
        finally:
            self.in_flight -= 1

    async def close(self):
        self.closed = True


@pytest.fixture
def stub_client(monkeypatch):
    holder = {}

    def _install(**kwargs):
        client = FakeClient(**kwargs)
        holder["client"] = client
        monkeypatch.setattr(cli, "make_client", lambda config: client)
        return client

    return _install


def _run(paths_dir, out_dir, config, models=("model-a",), themes=THEMES, **kwargs):
    paths = cli._discover_transcript_paths(paths_dir, None)
    return asyncio.run(
        cli.run_batch(
            paths, themes, list(models), str(out_dir), config=config, base_dir=paths_dir, **kwargs
        )
    )


def _payload(score):
    entry = {"theme": THEME, "relevance_score": score, "explicitness": "explicit",
             "reasoning": "r", "locations": []}
    return {"model_used": "model-a", "chunked": False, "themes": {"glory": entry},
            "_meta": {"elapsed_seconds": 1.0, "usage": {}, "theme_usage": {}}}


# --------------------------------------------------------------------------
# Checkpoint / resume
# --------------------------------------------------------------------------

def test_completed_transcripts_are_not_reanalyzed(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt", "c.txt"])
    out = tmp_path / "out"

    client = stub_client()
    _run(directory, out, _config())
    assert client.calls == 3

    # Second run: everything is checkpointed, so nothing should be called.
    client2 = stub_client()
    _run(directory, out, _config())
    assert client2.calls == 0
    assert (out / "report.html").exists()


def test_resume_analyzes_only_the_remainder(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt", "c.txt"])
    out = tmp_path / "out"
    out.mkdir()

    # Pretend a previous run finished only "a.txt" before dying.
    with CheckpointWriter(checkpoint_path(str(out)), THEMES, ["model-a"]) as writer:
        writer.record("a.txt", "a", [("model-a", _payload(1))])

    client = stub_client()
    _run(directory, out, _config())
    assert client.calls == 2  # b and c only

    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert set(done) == {"a.txt", "b.txt", "c.txt"}


def test_no_resume_forces_full_reanalysis(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"

    stub_client()
    _run(directory, out, _config())

    client = stub_client()
    _run(directory, out, _config(), resume=False)
    assert client.calls == 2


def test_changing_theme_invalidates_checkpoint(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"

    stub_client()
    _run(directory, out, _config())

    client = stub_client()
    paths = cli._discover_transcript_paths(directory, None)
    asyncio.run(
        cli.run_batch(paths, [Theme("other", "a totally different theme")], ["model-a"], str(out),
                      config=_config(), base_dir=directory)
    )
    assert client.calls == 2, "stale results must not be reused for a new theme"


def test_failed_transcripts_are_retried_on_resume(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"

    # "a" fails on the first run.
    client = stub_client(fail_for=["a.txt"])
    body = (tmp_path / "transcripts" / "a.txt")
    body.write_text("a.txt marker line\n", encoding="utf-8")
    _run(directory, out, _config())
    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert "error" in done["a.txt"]["results"][0][1]

    # Second run retries only the failure.
    client2 = stub_client()
    _run(directory, out, _config())
    assert client2.calls == 1
    done2 = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert "error" not in done2["a.txt"]["results"][0][1]


def test_skip_failed_leaves_failures_alone(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    (tmp_path / "transcripts" / "a.txt").write_text("a.txt marker\n", encoding="utf-8")
    out = tmp_path / "out"

    stub_client(fail_for=["a.txt"])
    _run(directory, out, _config())

    client2 = stub_client()
    _run(directory, out, _config(), retry_failed=False)
    assert client2.calls == 0


# --------------------------------------------------------------------------
# Identity: duplicate names must not collide
# --------------------------------------------------------------------------

def test_duplicate_basenames_get_distinct_keys_and_names(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["x/sermon.txt", "y/sermon.txt", "unique.txt"])
    out = tmp_path / "out"

    client = stub_client()
    _run(directory, out, _config())
    assert client.calls == 3, "same-named files in different folders are different work"

    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert len(done) == 3
    names = {entry["display_name"] for entry in done.values()}
    assert len(names) == 3, f"display names collided: {names}"
    assert "unique" in names, "non-colliding names should stay clean"

    # And the report must contain all three, not two.
    _, transcripts = read_cache(str(out / ".raw_results.json"))
    assert len(transcripts) == 3


def test_work_items_are_unique_even_for_identical_stems():
    items = cli._build_work_items(
        ["/base/a/s.txt", "/base/b/s.txt", "/base/c/s.txt"], "/base"
    )
    assert len({i.key for i in items}) == 3
    assert len({i.display_name for i in items}) == 3


def test_report_order_is_discovery_order_not_completion_order(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt", "c.txt", "d.txt"])
    out = tmp_path / "out"

    # Make "a" the slowest so completion order differs from discovery order.
    class Skewed(FakeClient):
        async def create(self, **kwargs):
            user = kwargs["messages"][-1]["content"]
            if "SLOW" in user:
                await asyncio.sleep(0.08)
            return await super().create(**kwargs)

    (tmp_path / "transcripts" / "a.txt").write_text("SLOW line\n", encoding="utf-8")
    client = Skewed(delay=0.001)
    import transcript_theme_analyzer.cli as cli_mod
    original = cli_mod.make_client
    cli_mod.make_client = lambda config: client
    try:
        _run(directory, out, _config())
    finally:
        cli_mod.make_client = original

    _, transcripts = read_cache(str(out / ".raw_results.json"))
    assert list(transcripts) == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------

def test_transcripts_actually_run_concurrently(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, [f"t{i}.txt" for i in range(8)])
    out = tmp_path / "out"

    client = stub_client(delay=0.05)
    _run(directory, out, _config(max_concurrent_transcripts=4, max_concurrent_requests=8))
    assert client.max_in_flight > 1, "transcripts ran one at a time"
    assert client.max_in_flight <= 4


def test_request_gate_caps_total_in_flight(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, [f"t{i}.txt" for i in range(12)])
    out = tmp_path / "out"

    client = stub_client(delay=0.05)
    _run(directory, out, _config(max_concurrent_transcripts=12, max_concurrent_requests=3))
    assert client.max_in_flight <= 3, f"gate leaked: {client.max_in_flight} in flight"


def test_concurrency_is_faster_than_serial(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, [f"t{i}.txt" for i in range(8)])

    client = stub_client(delay=0.05)
    import time
    start = time.monotonic()
    _run(directory, tmp_path / "fast", _config(max_concurrent_transcripts=8, max_concurrent_requests=8))
    concurrent_elapsed = time.monotonic() - start

    client2 = stub_client(delay=0.05)
    start = time.monotonic()
    _run(directory, tmp_path / "slow", _config(max_concurrent_transcripts=1, max_concurrent_requests=1))
    serial_elapsed = time.monotonic() - start

    assert client.calls == client2.calls == 8
    assert concurrent_elapsed < serial_elapsed / 2, (
        f"concurrency bought nothing: {concurrent_elapsed:.2f}s vs serial {serial_elapsed:.2f}s"
    )


def test_client_is_shared_and_closed(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt", "c.txt"])
    client = stub_client()
    _run(directory, tmp_path / "out", _config())
    assert client.closed, "the HTTP client must be closed, not leaked"


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------

def test_unreadable_transcript_does_not_sink_the_batch(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["good.txt", "bad.txt"])
    # A .docx extension on non-docx bytes: the loader will raise.
    (tmp_path / "transcripts" / "bad.txt").unlink()
    (tmp_path / "transcripts" / "bad.docx").write_bytes(b"not a real docx")
    out = tmp_path / "out"

    client = stub_client()
    _run(directory, out, _config())
    assert client.calls == 1, "the good transcript should still be analyzed"

    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert "error" in done["bad.docx"]["results"][0][1]


def test_empty_transcript_is_recorded_as_error_not_analyzed(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["good.txt"])
    (tmp_path / "transcripts" / "empty.txt").write_text("   \n\n", encoding="utf-8")
    out = tmp_path / "out"

    client = stub_client()
    _run(directory, out, _config())
    assert client.calls == 1, "an empty transcript must not be sent to the API"
    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert "error" in done["empty.txt"]["results"][0][1]


def test_multiple_models_all_recorded(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"

    client = stub_client()
    _run(directory, out, _config(), models=("model-a", "model-b"))
    assert client.calls == 4

    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a", "model-b"])
    assert {m for m, _ in done["a.txt"]["results"]} == {"model-a", "model-b"}


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------

def _rate_limit_error(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    response = httpx.Response(
        429, request=httpx.Request("POST", "http://x"), headers=headers
    )
    return openai.RateLimitError("slow down", response=response, body=None)


def test_retry_after_header_is_honoured():
    assert analyzer._retry_after_seconds(_rate_limit_error(7)) == 7.0
    assert analyzer._retry_after_seconds(_rate_limit_error()) is None
    # An HTTP-date Retry-After is unparseable as a float and must not explode.
    assert analyzer._retry_after_seconds(_rate_limit_error("Wed, 21 Oct 2015 07:28:00 GMT")) is None
    # A hostile value must not park the run for a week.
    assert analyzer._retry_after_seconds(_rate_limit_error(999999)) == 120.0


def test_rate_limit_pauses_every_worker():
    """A 429 hit by one call must hold the others back too."""

    async def scenario():
        gate = analyzer.RequestGate(limit=4)
        loop = asyncio.get_running_loop()
        gate.pause(0.15)
        start = loop.time()

        async def worker():
            async with gate:
                return loop.time() - start

        waits = await asyncio.gather(*[worker() for _ in range(4)])
        return waits

    waits = asyncio.run(scenario())
    assert all(w >= 0.14 for w in waits), f"a worker slipped through the cooldown: {waits}"


def test_pause_only_extends():
    async def scenario():
        gate = analyzer.RequestGate(limit=1)
        gate.pause(0.2)
        first = gate._pause_until
        gate.pause(0.01)  # shorter -- must not shorten the existing cooldown
        return first, gate._pause_until

    first, second = asyncio.run(scenario())
    assert first == second


def test_gate_slot_released_on_cancellation():
    """A worker cancelled while sleeping off a cooldown must not leak its
    slot, or the run deadlocks after enough interruptions."""

    async def scenario():
        gate = analyzer.RequestGate(limit=1)
        gate.pause(10)

        task = asyncio.create_task(gate.__aenter__())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        gate._pause_until = 0
        # If the slot leaked, this acquire hangs forever.
        await asyncio.wait_for(gate.__aenter__(), timeout=1.0)
        return True

    assert asyncio.run(scenario())


def test_structured_output_rejection_is_learned_once():
    """The 400-then-200 pattern must be paid once per model, not per call."""
    counts = {"schema": 0, "plain": 0}

    async def fake_create(**kwargs):
        if "response_format" in kwargs:
            counts["schema"] += 1
            raise openai.BadRequestError(
                "unsupported",
                response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                body=None,
            )
        counts["plain"] += 1
        body = {"relevance_score": 1, "reasoning": "r", "explicitness": "absent", "locations": []}
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(body)))],
            usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create)))

    async def scenario():
        for _ in range(6):
            await analyzer._call_with_retry(
                client, model="m", system="s", user="u", schema_cls=ChunkAnalysis,
                schema_name="chunk_analysis", max_retries=3, max_output_tokens=100,
            )

    asyncio.run(scenario())
    assert counts == {"schema": 1, "plain": 6}


# --------------------------------------------------------------------------
# Interruption: the case checkpointing exists for
# --------------------------------------------------------------------------

def test_interrupt_saves_finished_work_and_resumes(tmp_path, monkeypatch):
    """Kill the run partway, then re-run: finished transcripts must be kept
    and the remainder must still be analyzed."""
    directory = _write_transcripts(tmp_path, [f"t{i}.txt" for i in range(6)])
    out = tmp_path / "out"

    class DyingClient(FakeClient):
        async def create(self, **kwargs):
            if self.calls >= 2:
                raise KeyboardInterrupt("simulated Ctrl+C")
            return await super().create(**kwargs)

    dying = DyingClient(delay=0.001)
    monkeypatch.setattr(cli, "make_client", lambda config: dying)
    with pytest.raises(KeyboardInterrupt):
        _run(directory, out, _config(max_concurrent_transcripts=1, max_concurrent_requests=1))

    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert 0 < len(done) < 6, f"expected a partial checkpoint, got {len(done)}"
    for entry in done.values():
        assert "error" not in entry["results"][0][1], (
            "an interrupt was recorded as a transcript failure; resume would never retry it"
        )

    survivors = set(done)
    healthy = FakeClient(delay=0.001)
    monkeypatch.setattr(cli, "make_client", lambda config: healthy)
    _run(directory, out, _config())

    assert healthy.calls == 6 - len(survivors), "resume re-did work it already had"
    final = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert len(final) == 6
    assert (out / "report.html").exists()


def test_keyboard_interrupt_is_not_recorded_as_a_model_error():
    """run_all must let a KeyboardInterrupt out rather than turning it into
    an error payload."""

    async def scenario():
        async def boom(*args, **kwargs):
            raise KeyboardInterrupt("ctrl-c")

        import transcript_theme_analyzer.cli as cli_mod
        original = cli_mod.run_one_model
        cli_mod.run_one_model = boom
        try:
            await cli_mod.run_all("text", THEMES, ["m"], config=_config(), client=FakeClient())
        finally:
            cli_mod.run_one_model = original

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(scenario())


def test_ordinary_model_failure_is_still_recorded(tmp_path, stub_client):
    """The flip side: a real Exception must be captured, not propagated."""
    directory = _write_transcripts(tmp_path, ["a.txt"])
    (tmp_path / "transcripts" / "a.txt").write_text("a.txt marker\n", encoding="utf-8")
    out = tmp_path / "out"

    stub_client(fail_for=["a.txt"])
    _run(directory, out, _config())  # must not raise

    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert "error" in done["a.txt"]["results"][0][1]


def test_salvage_rebuilds_report_from_checkpoint_alone(tmp_path):
    """If an interrupt escapes the loop, the report must still be
    reconstructible from disk with no in-memory state."""
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt", "c.txt"])
    out = tmp_path / "out"
    out.mkdir()

    payload = _payload(3)
    with CheckpointWriter(checkpoint_path(str(out)), THEMES, ["model-a"]) as writer:
        writer.record("a.txt", "a", [("model-a", payload)])
        writer.record("c.txt", "c", [("model-a", payload)])

    paths = cli._discover_transcript_paths(directory, None)
    cli.salvage_report(paths, THEMES, ["model-a"], str(out), directory)

    _, transcripts = read_cache(str(out / ".raw_results.json"))
    assert list(transcripts) == ["a", "c"], "salvage must keep discovery order"
    assert (out / "report.html").exists()


def test_salvage_with_empty_checkpoint_writes_nothing(tmp_path):
    directory = _write_transcripts(tmp_path, ["a.txt"])
    out = tmp_path / "out"
    out.mkdir()
    paths = cli._discover_transcript_paths(directory, None)
    cli.salvage_report(paths, THEMES, ["model-a"], str(out), directory)
    assert not (out / "report.html").exists()


def test_run_batch_reports_interruption_via_return_value(tmp_path, monkeypatch):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "make_client", lambda config: FakeClient(delay=0.001))
    assert _run(directory, out, _config()) is False


def test_bad_file_warns_but_does_not_abort_the_batch(tmp_path, capsys):
    """One unreadable file must not block a corpus -- especially not the
    resume of a batch that is already mostly done."""
    directory = _write_transcripts(tmp_path, ["good.txt"])
    (tmp_path / "transcripts" / "broken.docx").write_bytes(b"not a docx")
    paths = cli._discover_transcript_paths(directory, None)

    count = cli._validate_formats_upfront(paths)
    assert count == 1
    assert "WARNING" in capsys.readouterr().err


def test_batch_survives_a_rate_limit_storm(tmp_path, monkeypatch):
    """Every transcript must still complete when the provider throws 429s
    for a while, and the retries must not be recorded as failures."""

    class Flaky(FakeClient):
        """Rejects everything for a fixed window, then recovers -- which is
        how a real rate limit behaves. (A counter-based fake would reject the
        Nth call however long you waited, which no provider does and which no
        amount of correct backoff could survive.)"""

        def __init__(self, window=0.3):
            super().__init__(delay=0.001)
            self.window = window
            self.opens_at = None
            self.rejections = 0

        async def create(self, **kwargs):
            loop = asyncio.get_running_loop()
            if self.opens_at is None:
                self.opens_at = loop.time() + self.window
            if loop.time() < self.opens_at:
                self.rejections += 1
                raise _rate_limit_error(retry_after=0.05)
            return await super().create(**kwargs)

    flaky = Flaky()
    monkeypatch.setattr(cli, "make_client", lambda config: flaky)

    directory = _write_transcripts(tmp_path, [f"t{i}.txt" for i in range(8)])
    out = tmp_path / "out"
    _run(directory, out, _config(max_concurrent_transcripts=8, max_concurrent_requests=8))

    assert flaky.rejections > 0, "the test did not actually exercise rate limiting"
    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert len(done) == 8
    for key, entry in done.items():
        assert "error" not in entry["results"][0][1], f"{key} was given up on despite retries"


def test_exhausted_retries_become_an_error_not_a_crash(tmp_path, monkeypatch):
    """When retries genuinely run out, that transcript fails but the batch
    still finishes and reports it."""

    class AlwaysLimited(FakeClient):
        async def create(self, **kwargs):
            raise _rate_limit_error(retry_after=0.01)

    monkeypatch.setattr(cli, "make_client", lambda config: AlwaysLimited(delay=0))

    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"
    _run(directory, out, _config(max_retries=2))  # must not raise

    done = read_checkpoint(checkpoint_path(str(out)), THEMES, ["model-a"])
    assert len(done) == 2
    assert all("error" in entry["results"][0][1] for entry in done.values())
    assert (out / "report.html").exists()


# --------------------------------------------------------------------------
# Multi-theme: one call for every theme, theme input, outputs
# --------------------------------------------------------------------------

FOUR = [Theme("a", "Alpha"), Theme("b", "Beta"), Theme("c", "Gamma"), Theme("d", "Delta")]


def test_one_call_per_transcript_for_all_themes(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt", "c.txt"])
    out = tmp_path / "out"

    client = stub_client()
    _run(directory, out, _config(), themes=FOUR)
    assert client.calls == 3, "every theme must go in the same call"

    _, transcripts = read_cache(str(out / ".raw_results.json"))
    payload = transcripts["a"][0][1]
    assert list(payload["themes"]) == ["a", "b", "c", "d"]
    assert payload["themes"]["b"]["theme"] == "Beta"
    assert payload["_meta"]["usage"]["cached_tokens"] == 4
    assert set(payload["_meta"]["theme_usage"]) == {"a", "b", "c", "d"}


def test_reordered_themes_reuse_checkpoint(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"

    stub_client()
    _run(directory, out, _config(), themes=FOUR[:2])

    client = stub_client()
    _run(directory, out, _config(), themes=list(reversed(FOUR[:2])))
    assert client.calls == 0, "the same theme set in another order is the same work"


def test_changed_description_invalidates_checkpoint(tmp_path, stub_client):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"

    stub_client()
    _run(directory, out, _config(), themes=[Theme("a", "Alpha", "first meaning")])

    client = stub_client()
    _run(directory, out, _config(), themes=[Theme("a", "Alpha", "second meaning")])
    assert client.calls == 2, "a new description is a different question"


def test_outputs_written(tmp_path, stub_client, capsys):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt"])
    out = tmp_path / "out"

    stub_client()
    _run(directory, out, _config(), themes=FOUR[:2])

    assert (out / ".raw_results.json").exists()
    assert (out / "docx" / "a.docx").exists()
    assert (out / "docx" / "b.docx").exists()
    assert not (out / "report.docx").exists()
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Alpha" in html and "Beta" in html
    printed = capsys.readouterr().out
    assert "report.html" in printed and "a.docx" in printed and "b.docx" in printed


def test_prices_resolved_once_per_run_and_applied(tmp_path, stub_client, monkeypatch):
    directory = _write_transcripts(tmp_path, ["a.txt", "b.txt", "c.txt"])
    out = tmp_path / "out"
    lookups = []

    def fake(models, config):
        lookups.append(list(models))
        return {m: Prices(1.0, 2.0) for m in models}

    monkeypatch.setattr(cli, "resolve_prices", fake)
    stub_client()
    _run(directory, out, _config(), models=("model-a", "model-b"), themes=FOUR[:2])

    assert lookups == [["model-a", "model-b"]]
    _, transcripts = read_cache(str(out / ".raw_results.json"))
    meta = transcripts["a"][0][1]["_meta"]
    assert meta["usage"]["cost"] is not None
    assert meta["usage"]["cost_source"] == "prices"
    assert all(t["cost"] is not None for t in meta["theme_usage"].values())


def test_resume_with_nothing_pending_skips_price_lookup(tmp_path, stub_client, _no_price_lookup):
    directory = _write_transcripts(tmp_path, ["a.txt"])
    out = tmp_path / "out"

    stub_client()
    _run(directory, out, _config())
    assert len(_no_price_lookup) == 1
    stub_client()
    _run(directory, out, _config())
    assert len(_no_price_lookup) == 1, "nothing to analyze, nothing to price"


@pytest.fixture
def cli_env(monkeypatch):
    """Run ``cli.main`` offline with a fixed config and no logging takeover."""
    monkeypatch.setattr(cli, "load_config", lambda: _config())
    monkeypatch.setattr(cli, "_setup_logging", lambda level: None)


def test_over_limit_rejected_before_any_call(tmp_path, stub_client, cli_env, monkeypatch, capsys):
    directory = _write_transcripts(tmp_path, ["a.txt"])
    client = stub_client()

    def no_discovery(*args, **kwargs):
        raise AssertionError("transcripts were discovered before the theme check")

    monkeypatch.setattr(cli, "_discover_transcript_paths", no_discovery)
    argv = ["--transcript-dir", directory, "--out-dir", str(tmp_path / "out"), "--max-themes", "5"]
    for i in range(6):
        argv += ["--theme", f"theme {i}"]

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert (
        "You entered 6 themes. The limit is 5 (MAX_THEMES). "
        "Remove 1 theme(s) or raise MAX_THEMES."
    ) in capsys.readouterr().err
    assert client.calls == 0
    assert not (tmp_path / "out").exists()


def test_no_themes_rejected(tmp_path, stub_client, cli_env, capsys):
    directory = _write_transcripts(tmp_path, ["a.txt"])
    client = stub_client()
    with pytest.raises(SystemExit) as exc:
        cli.main(["--transcript-dir", directory, "--out-dir", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert "No themes given" in capsys.readouterr().err
    assert client.calls == 0


def test_missing_themes_file_rejected(tmp_path, stub_client, cli_env, capsys):
    directory = _write_transcripts(tmp_path, ["a.txt"])
    client = stub_client()
    with pytest.raises(SystemExit) as exc:
        cli.main(["--transcript-dir", directory, "--themes-file", str(tmp_path / "nope.csv")])
    assert exc.value.code == 2
    assert "nope.csv" in capsys.readouterr().err
    assert client.calls == 0


def test_themes_file_and_flags_combine(tmp_path, stub_client, cli_env):
    directory = _write_transcripts(tmp_path, ["a.txt"])
    csv_path = tmp_path / "themes.csv"
    csv_path.write_text(
        "theme,description\nForgiveness,letting go of wrongs\nHope,\n", encoding="utf-8"
    )
    out = tmp_path / "out"

    client = stub_client()
    cli.main([
        "--transcript-dir", directory, "--out-dir", str(out),
        "--theme", "Grace", "--themes-file", str(csv_path),
    ])
    assert client.calls == 1

    themes, transcripts = read_cache(str(out / ".raw_results.json"))
    assert [t.id for t in themes] == ["grace", "forgiveness", "hope"]
    assert themes[1].description == "letting go of wrongs"
    assert list(transcripts["a"][0][1]["themes"]) == ["grace", "forgiveness", "hope"]


def test_max_themes_flag_raises_the_limit(tmp_path, stub_client, cli_env):
    directory = _write_transcripts(tmp_path, ["a.txt"])
    argv = ["--transcript-dir", directory, "--out-dir", str(tmp_path / "out"), "--max-themes", "6"]
    for i in range(6):
        argv += ["--theme", f"theme {i}"]

    client = stub_client()
    cli.main(argv)
    assert client.calls == 1


def test_single_transcript_path(tmp_path, stub_client, cli_env, _no_price_lookup):
    directory = _write_transcripts(tmp_path, ["one.txt"])
    out = tmp_path / "out"

    client = stub_client()
    cli.main([
        "--transcript", f"{directory}/one.txt", "--out-dir", str(out),
        "--theme", "Alpha", "--theme", "Beta", "--models", "model-a", "model-b",
    ])
    assert client.calls == 2  # one per model, both themes in each
    assert _no_price_lookup == [["model-a", "model-b"]]
    assert (out / "report.html").exists()
    assert (out / "docx" / "alpha.docx").exists()
    assert (out / "docx" / "beta.docx").exists()
    _, transcripts = read_cache(str(out / ".raw_results.json"))
    assert list(transcripts) == ["one"]
