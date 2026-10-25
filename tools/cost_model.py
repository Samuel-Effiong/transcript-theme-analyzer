#!/usr/bin/env python3
"""Cost model for a full-corpus run of the transcript theme analyzer.

Derives token volume from the pipeline's *actual* configuration (chunk size,
overlap, single-pass threshold, system prompt length) rather than a guess, then
prices that volume against a table of provider rates.

Prices are per 1M tokens and are EDITABLE -- they move. Verify against
anthropic.com/pricing and openrouter.ai/models before quoting these to anyone.

Usage:
    python tools/cost_model.py                 # default assumptions
    python tools/cost_model.py --transcripts 1800 --avg-minutes 55
"""
from __future__ import annotations

import argparse
import math

CHARS_PER_TOKEN = 3.5

# Pulled from transcript_theme_analyzer/config.py + prompts.py defaults.
CHUNK_SIZE_TOKENS = 12_000
CHUNK_OVERLAP_TOKENS = 800
SINGLE_PASS_TOKEN_LIMIT = 20_000
CHUNK_SYSTEM_PROMPT_TOKENS = 1_655
SYNTHESIS_SYSTEM_PROMPT_TOKENS = 477

# Wrapper text around the transcript in the user message (theme, chunk index,
# char offsets). Small but not zero, and it is paid once per call.
USER_TEMPLATE_TOKENS = 40

# Observed shape of a chunk_analysis response: score + explicitness + reasoning
# + a handful of location markers. Tune from a real run's usage stats.
AVG_OUTPUT_TOKENS_PER_CHUNK = 600
AVG_OUTPUT_TOKENS_SYNTHESIS = 220
# Each partial reasoning fed back into the synthesis call.
SYNTHESIS_TOKENS_PER_PARTIAL = 70

# Schema-validation and rate-limit retries re-send the full input. 8% is a
# conservative real-world allowance; set to 0 to see the floor.
RETRY_OVERHEAD = 0.08

# (name, input $/Mtok, output $/Mtok).
# Verified 2026-08-27 against platform.claude.com/docs/en/about-claude/pricing
# and the individual openrouter.ai/<model> pages. OpenRouter passes provider
# pricing through without markup, so anthropic/* costs the same either way.
# VERIFY AGAIN BEFORE QUOTING -- these move.
PRICES = [
    ("Claude Opus 5 (frontier)",            5.00, 25.00),
    ("Claude Sonnet 5 (mid)",               2.00, 10.00),
    ("Claude Haiku 4.5 (small)",            1.00,  5.00),
    ("Gemini 3.6 Flash (OpenRouter)",       0.75,  3.75),
    ("Gemini 3.5 Flash Lite (OpenRouter)",  0.30,  2.50),
    ("DeepSeek V4 Flash (OpenRouter)",      0.0679, 0.168),
]


def transcript_tokens(minutes: float, words_per_minute: float = 150.0) -> int:
    """Spoken-word transcript length -> token count.

    ~150 wpm is a normal preaching/teaching cadence; ~1.35 tokens per English
    word is the usual ratio for conversational text.
    """
    return int(minutes * words_per_minute * 1.35)


def chunk_count(tokens: int) -> int:
    if tokens <= SINGLE_PASS_TOKEN_LIMIT:
        return 1
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    return 1 + math.ceil((tokens - CHUNK_SIZE_TOKENS) / step)


def tokens_per_transcript(tokens: int) -> tuple[int, int, int]:
    """-> (api_calls, input_tokens, output_tokens) for one transcript."""
    n = chunk_count(tokens)
    if n == 1:
        calls = 1
        tin = CHUNK_SYSTEM_PROMPT_TOKENS + USER_TEMPLATE_TOKENS + tokens
        tout = AVG_OUTPUT_TOKENS_PER_CHUNK
    else:
        # Overlap means the billed input exceeds the transcript's own length.
        billed_body = n * CHUNK_SIZE_TOKENS
        calls = n + 1  # n chunk calls + 1 synthesis call
        tin = n * (CHUNK_SYSTEM_PROMPT_TOKENS + USER_TEMPLATE_TOKENS) + billed_body
        tin += SYNTHESIS_SYSTEM_PROMPT_TOKENS + n * SYNTHESIS_TOKENS_PER_PARTIAL
        tout = n * AVG_OUTPUT_TOKENS_PER_CHUNK + AVG_OUTPUT_TOKENS_SYNTHESIS

    tin = int(tin * (1 + RETRY_OVERHEAD))
    tout = int(tout * (1 + RETRY_OVERHEAD))
    return calls, tin, tout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", type=int, default=1800)
    ap.add_argument("--avg-minutes", type=float, default=55.0,
                    help="average transcript duration in minutes")
    ap.add_argument("--themes", type=int, default=1,
                    help="themes analyzed per run (each is a full pass)")
    args = ap.parse_args()

    t = transcript_tokens(args.avg_minutes)
    calls, tin, tout = tokens_per_transcript(t)
    N = args.transcripts * args.themes

    print(f"Corpus:            {args.transcripts:,} transcripts x {args.themes} theme(s)")
    print(f"Avg transcript:    {args.avg_minutes:.0f} min  ~= {t:,} tokens "
          f"({chunk_count(t)} chunk(s))")
    print(f"Per transcript:    {calls} API calls, {tin:,} in / {tout:,} out")
    print(f"Per full run:      {N * calls:,} API calls, "
          f"{N * tin / 1e6:,.1f}M in / {N * tout / 1e6:,.1f}M out")
    print()
    print(f"{'Model tier':<34}{'$/run':>12}{'$/transcript':>15}")
    print("-" * 61)
    for name, pin, pout in PRICES:
        cost = (N * tin / 1e6) * pin + (N * tout / 1e6) * pout
        print(f"{name:<34}{cost:>12,.2f}{cost / N:>15,.4f}")
    print()
    print("Prices are per 1M tokens and change. Verify at anthropic.com/pricing")
    print("and openrouter.ai/models before quoting.")


if __name__ == "__main__":
    main()
