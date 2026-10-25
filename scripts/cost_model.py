"""Cost model for a full-corpus run of the transcript theme analyzer.

Grounded in real measured usage from `results/x-ai__grok-4.3.json`, which
recorded 5 calls / 36,277 prompt + 4,222 completion tokens for one chunked
transcript. The per-size profiles below reproduce that pipeline's actual
behaviour: transcripts under SINGLE_PASS_TOKEN_LIMIT take a one-call fast
path, longer ones take the map-reduce path (N chunk calls + 1 synthesis).

Run with real corpus stats once the transcripts are mounted:
    python scripts/cost_model.py --transcript-dir /path/to/corpus
Otherwise it uses the documented mix assumption and prints the sensitivity
band, so every figure in the proposal can be re-derived.
"""
from __future__ import annotations

import argparse
import os
import sys

# --- Pipeline constants, mirrored from .env / config.py ------------------
CHUNK_SIZE_TOKENS = 12_000
CHUNK_OVERLAP_TOKENS = 800
SINGLE_PASS_TOKEN_LIMIT = 20_000
CHARS_PER_TOKEN_HEURISTIC = 3.5   # chunker.py's conservative estimator
CHARS_PER_TOKEN_REAL = 4.24       # back-solved from the measured Grok run

# System prompt (CHUNK_ANALYSIS_SYSTEM_PROMPT_V1) measured at ~1,300 tokens.
SYSTEM_PROMPT_TOKENS = 1_300
SYNTHESIS_IN_TOKENS = 800
SYNTHESIS_OUT_TOKENS = 430
CHUNK_OUT_TOKENS = 950            # observed 881-1,059
SINGLE_PASS_OUT_TOKENS = 1_050

# --- Live pricing, USD per 1M tokens (fetched 2026-08-27) ----------------
MODELS = {
    # Anthropic first-party API rates
    "Claude Opus 5":            (5.00,  25.00, "Anthropic"),
    "Claude Sonnet 5":          (2.00,  10.00, "Anthropic"),
    "Claude Haiku 4.5":         (1.00,   5.00, "Anthropic"),
    # OpenRouter-routed alternatives
    "DeepSeek V4 Flash Latest": (0.03,   0.10, "OpenRouter"),
    "DeepSeek V4 Flash 0423":   (0.0679, 0.168, "OpenRouter"),
    "DeepSeek V4 Flash (std)":  (0.098,  0.196, "OpenRouter"),
    "Gemini 2.5 Flash-Lite":    (0.10,   0.40, "OpenRouter"),
    "Gemini 3.1 Flash-Lite":    (0.25,   1.50, "OpenRouter"),
    "Gemini 3 Flash":           (0.50,   3.00, "OpenRouter"),
    "Gemini 3.1 Pro":           (2.00,  12.00, "OpenRouter"),
}

OPENROUTER_CREDIT_FEE = 0.055     # 5.5% on card top-ups, $0.80 minimum


def profile_transcript(transcript_tokens: int) -> tuple[int, int, int]:
    """Returns (input_tokens, output_tokens, api_calls) for one transcript,
    replicating analyzer.analyze()'s single-pass vs. map-reduce branch."""
    chars = int(transcript_tokens * CHARS_PER_TOKEN_REAL)
    estimated = int(chars / CHARS_PER_TOKEN_HEURISTIC)

    if estimated <= SINGLE_PASS_TOKEN_LIMIT:
        return transcript_tokens + SYSTEM_PROMPT_TOKENS, SINGLE_PASS_OUT_TOKENS, 1

    chunk_chars = int(CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_HEURISTIC)
    overlap_chars = int(CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN_HEURISTIC)
    stride = chunk_chars - overlap_chars
    n_chunks = max(1, -(-(chars - overlap_chars) // stride))

    chunk_real_tokens = int(chunk_chars / CHARS_PER_TOKEN_REAL)
    inp = n_chunks * (chunk_real_tokens + SYSTEM_PROMPT_TOKENS) + SYNTHESIS_IN_TOKENS
    out = n_chunks * CHUNK_OUT_TOKENS + SYNTHESIS_OUT_TOKENS
    return inp, out, n_chunks + 1


# Corpus mix: duration -> (transcript tokens, share of corpus)
DEFAULT_MIX = {
    "Short (~30 min)":  (6_000,  0.30),
    "Medium (~1 hr)":   (13_000, 0.45),
    "Long (~2 hr)":     (28_000, 0.25),
}


def corpus_totals(n_transcripts: int, mix: dict) -> tuple[float, float, float]:
    inp = out = calls = 0.0
    for tokens, share in mix.values():
        i, o, c = profile_transcript(tokens)
        inp += i * share * n_transcripts
        out += o * share * n_transcripts
        calls += c * share * n_transcripts
    return inp, out, calls


def cost(inp: float, out: float, model: str) -> float:
    pin, pout, _ = MODELS[model]
    return inp / 1e6 * pin + out / 1e6 * pout


def scan_corpus(path: str) -> dict:
    """Build a real size mix from an actual transcript directory."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transcript_theme_analyzer.loader import load_transcript_text

    sizes = []
    for root, _, files in os.walk(path):
        for name in files:
            if not name.lower().endswith((".txt", ".docx")):
                continue
            try:
                text = load_transcript_text(os.path.join(root, name))
            except Exception:
                continue
            sizes.append(int(len(text) / CHARS_PER_TOKEN_REAL))
    if not sizes:
        raise SystemExit(f"No .txt/.docx transcripts found under {path!r}")
    sizes.sort()
    print(f"Scanned {len(sizes)} transcripts.")
    print(f"  median {sizes[len(sizes)//2]:,} tok   "
          f"mean {sum(sizes)//len(sizes):,} tok   max {sizes[-1]:,} tok")
    return {f"t{i}": (s, 1.0 / len(sizes)) for i, s in enumerate(sizes)}, len(sizes)


def report(n: int, mix: dict, themes: int) -> None:
    inp, out, calls = corpus_totals(n, mix)
    inp *= themes
    out *= themes
    calls *= themes

    label = f"{n:,} transcripts x {themes} theme(s)"
    print(f"\n{'=' * 78}\nWORKLOAD: {label}\n{'=' * 78}")
    print(f"  API calls      {calls:>15,.0f}")
    print(f"  Input tokens   {inp:>15,.0f}")
    print(f"  Output tokens  {out:>15,.0f}")

    print(f"\n{'Model':<26}{'Via':<12}{'Cost/run':>12}{'vs Opus 5':>14}")
    print("-" * 78)
    baseline = cost(inp, out, "Claude Opus 5")
    for name, (_, _, via) in MODELS.items():
        c = cost(inp, out, name)
        if via == "OpenRouter":
            c *= 1 + OPENROUTER_CREDIT_FEE
        ratio = baseline / c if c else float("inf")
        mark = "  (baseline)" if name == "Claude Opus 5" else f"{ratio:>10,.0f}x cheaper"
        print(f"{name:<26}{via:<12}${c:>11,.2f}{mark:>14}")
    print("\n  OpenRouter figures include the 5.5% credit top-up fee.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--transcript-dir", help="scan a real corpus instead of the assumed mix")
    p.add_argument("--n", type=int, default=1800, help="corpus size when not scanning")
    p.add_argument("--themes", type=int, default=1, help="themes analyzed per run")
    a = p.parse_args()

    if a.transcript_dir:
        mix, n = scan_corpus(a.transcript_dir)
    else:
        mix, n = DEFAULT_MIX, a.n
        print("Using assumed corpus mix (no --transcript-dir given):")
        for k, (tok, share) in mix.items():
            print(f"  {k:<18}{tok:>7,} tok   {share:>5.0%} of corpus")

    report(n, mix, a.themes)
    if not a.transcript_dir:
        print(f"\n{'=' * 78}\nSENSITIVITY: if the corpus averages 50% longer\n{'=' * 78}")
        report(n, {k: (int(t * 1.5), s) for k, (t, s) in mix.items()}, a.themes)
