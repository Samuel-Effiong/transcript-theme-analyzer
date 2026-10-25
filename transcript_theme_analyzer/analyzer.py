"""Core analysis pipeline: single-pass fast path + chunked map-reduce path."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, get_origin

import openai
from openai import AsyncOpenAI

from . import prompts
from .aggregate import compute_aggregate_score, merge_and_dedupe_locations
from .chunker import Chunk, chunk_transcript, estimate_tokens
from .config import Config
from .extraction import extract_passage
from .pricing import Prices, attribute_call, call_cost, call_discount
from .schema import (
    AnalysisResult,
    Location,
    LocationMarker,
    MultiThemeChunkAnalysis,
    MultiThemeSynthesis,
    ThemeChunkAnalysis,
    ThemeSynthesis,
)
from .themes import Theme

logger = logging.getLogger("transcript_theme_analyzer")


@dataclass
class Progress:
    """Emits a heartbeat as one transcript/model run works through its chunks.

    A long run is otherwise completely silent between the start of a
    transcript and its final score -- this reports each chunk as it lands so
    a stalled run is distinguishable from a slow one.
    """

    label: str
    total: int = 0
    done: int = 0

    def start(self, mode: str, total: int = 0) -> None:
        self.total = total
        self.done = 0
        if total:
            logger.info("[%s] %s: %d chunks", self.label, mode, total)
        else:
            logger.info("[%s] %s", self.label, mode)

    def step(self) -> None:
        self.done += 1
        logger.info("[%s] chunk %d/%d done", self.label, self.done, self.total)

    def note(self, message: str) -> None:
        logger.info("[%s] %s", self.label, message)


@dataclass
class CallLog:
    label: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    # Cost OpenRouter itself reported for this call (USD), if any.
    reported_cost: float | None = None
    # OpenRouter's input/output split of that cost (usage.cost_details), if any.
    reported_input_cost: float | None = None
    reported_output_cost: float | None = None
    theme_ids: list[str] = field(default_factory=list)
    theme_output_chars: dict[str, int] = field(default_factory=dict)
    # Filled by RunStats.apply_prices.
    cost: float | None = None
    cost_source: str | None = None
    discount: float | None = None


def _cached_tokens(usage) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    if cached is None:
        # DeepSeek's own field name.
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
    return cached or 0


def _usage_field(obj, name: str):
    """Read a field the SDK may hold as an attribute or in ``model_extra``
    (or, for a nested dict, as a plain key)."""
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    return value


def _as_float(value) -> float | None:
    """A cost-reporting quirk must never fail a call: non-numeric -> None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reported_cost(usage) -> float | None:
    return _as_float(_usage_field(usage, "cost"))


def _reported_cost_split(usage) -> tuple[float | None, float | None]:
    details = _usage_field(usage, "cost_details")
    if details is None:
        return None, None
    return (
        _as_float(_usage_field(details, "upstream_inference_prompt_cost")),
        _as_float(_usage_field(details, "upstream_inference_completions_cost")),
    )


@dataclass
class RunStats:
    calls: list[CallLog] = field(default_factory=list)
    _prices: Any = field(default=None, repr=False)

    def add(
        self,
        label: str,
        model: str,
        usage,
        theme_ids=(),
        theme_output_chars: dict[str, int] | None = None,
    ) -> None:
        themes = {
            "theme_ids": list(theme_ids),
            "theme_output_chars": dict(theme_output_chars or {}),
        }
        if usage is None:
            self.calls.append(CallLog(label=label, model=model, **themes))
            return
        reported_input, reported_output = _reported_cost_split(usage)
        self.calls.append(
            CallLog(
                label=label,
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                cached_tokens=_cached_tokens(usage),
                reported_cost=_reported_cost(usage),
                reported_input_cost=reported_input,
                reported_output_cost=reported_output,
                **themes,
            )
        )

    def _prices_for(self, call: CallLog) -> Prices | None:
        if isinstance(self._prices, dict):
            return self._prices.get(call.model)
        return self._prices

    def apply_prices(self, prices: "Prices | dict[str, Prices | None] | None") -> None:
        """Fill each call's cost/source/discount. Takes one Prices for every
        model, or the per-model dict from ``resolve_prices``."""
        self._prices = prices
        for call in self.calls:
            p = self._prices_for(call)
            call.cost, call.cost_source = call_cost(call, p)
            call.discount = call_discount(call, p)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    def summary(self) -> dict:
        costs = [c.cost for c in self.calls]
        # A partial sum would read as a real total, so any unknown -> unknown.
        cost = None if not costs or any(x is None for x in costs) else sum(costs)
        sources = {c.cost_source for c in self.calls if c.cost_source}
        if len(sources) > 1:
            cost_source = "mixed"
        else:
            cost_source = next(iter(sources), None)
        discounts = [c.discount for c in self.calls if c.discount is not None]
        return {
            "calls": len(self.calls),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "cached_tokens": sum(c.cached_tokens for c in self.calls),
            "total_tokens": self.total_tokens,
            "cost": cost,
            "cost_source": cost_source,
            "discount": sum(discounts) if discounts else None,
            "by_call": [c.__dict__ for c in self.calls],
        }

    def theme_usage(self) -> dict[str, dict]:
        """Estimated tokens and cost per theme, summed over all calls.
        Cost is None for a theme if any call that touched it has no cost."""
        usage: dict[str, dict] = {}
        for call in self.calls:
            shares = attribute_call(call, self._prices_for(call))
            for theme_id, (tokens, cost) in shares.items():
                entry = usage.setdefault(theme_id, {"tokens": 0.0, "cost": 0.0})
                entry["tokens"] += tokens
                if cost is None or entry["cost"] is None:
                    entry["cost"] = None
                else:
                    entry["cost"] += cost
        return usage


def openrouter_extra_body(config: Config) -> dict | None:
    """Ask OpenRouter to include usage accounting (actual cost, cache hits)."""
    return {"usage": {"include": True}} if config.is_openrouter else None


RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.APITimeoutError,
)


class RequestGate:
    """Caps total in-flight API calls and coordinates backoff across them.

    Two jobs, both about not tripping provider rate limits once transcripts
    and chunks run concurrently:

    * **A single global ceiling.** Transcript- and chunk-level concurrency
      multiply, so bounding either alone doesn't bound the request rate.
    * **Shared cooldown.** When one call is rate-limited, every other worker
      is about to be too. Backing off only the unlucky caller leaves the rest
      hammering a limit that is already tripped, which is what turns a brief
      429 into a sustained one. A 429 here pauses *all* callers.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self._pause_until = 0.0

    def pause(self, seconds: float) -> None:
        """Hold every caller back for `seconds`. Only ever extends an
        existing cooldown -- a shorter one must not cut a longer one short."""
        target = asyncio.get_running_loop().time() + seconds
        self._pause_until = max(self._pause_until, target)

    async def __aenter__(self) -> "RequestGate":
        await self._semaphore.acquire()
        try:
            # Re-check in a loop: another worker can extend the cooldown
            # while this one is sleeping through it.
            while True:
                remaining = self._pause_until - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return self
                await asyncio.sleep(remaining)
        except BaseException:
            # Cancelled mid-cooldown -- must not leak the slot.
            self._semaphore.release()
            raise

    async def __aexit__(self, *exc_info) -> None:
        self._semaphore.release()


_gate: RequestGate | None = None
_gate_loop: asyncio.AbstractEventLoop | None = None


def get_request_gate(limit: int) -> RequestGate:
    """One gate per event loop. Rebuilt if the loop or limit changes, since an
    asyncio primitive from a previous loop is unusable in the current one."""
    global _gate, _gate_loop
    loop = asyncio.get_running_loop()
    if _gate is None or _gate_loop is not loop or _gate.limit != limit:
        _gate = RequestGate(limit)
        _gate_loop = loop
    return _gate


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read the provider's own Retry-After hint. Honouring it beats guessing
    with exponential backoff -- the server is stating exactly how long the
    limit lasts."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("x-ratelimit-reset-after")
    if raw is None:
        return None
    try:
        # Retry-After may also be an HTTP date; a date fails this parse and
        # falls through to normal exponential backoff, which is fine.
        return max(0.0, min(float(raw), 120.0))
    except (TypeError, ValueError):
        return None


# Models whose endpoint rejected `response_format={"type": "json_schema"}`.
# Anthropic's OpenAI-compatible endpoint is one of them, and without this the
# pipeline pays a guaranteed-400 round trip before the fallback on *every*
# call -- doubling request count and latency for the whole run.
_NO_STRUCTURED_OUTPUT: set[str] = set()


# Validation keywords that OpenAI-style strict structured output does not
# support. Sending them gets the whole request rejected, so they are stripped
# from the wire schema -- Pydantic still enforces them when validating the
# response, which is where they actually matter.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "format",
    "minItems", "maxItems", "uniqueItems", "default",
})


def _strictify(node):
    """Rewrite a Pydantic-generated JSON Schema into the strict subset.

    Strict mode has two requirements Pydantic's output violates by default:
    every object must set ``additionalProperties: false``, and every property
    must appear in ``required`` -- optionality is expressed by a nullable type
    union, not by omission from ``required``. Pydantic already emits
    ``Optional[str]`` as an ``anyOf`` with ``null``, so promoting every
    property to required is safe: the model can still answer ``null``.

    Without this the provider rejects the request, the caller silently falls
    back to prompt-based JSON, and the reliability benefit of native
    structured output is lost on every model that actually supports it.
    """
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node

    result = {k: v for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYWORDS}
    for key, value in list(result.items()):
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            result[key] = {k: _strictify(v) for k, v in value.items()}
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            result[key] = _strictify(value)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            result[key] = [_strictify(v) for v in value]

    if result.get("type") == "object" or "properties" in result:
        properties = result.get("properties") or {}
        result["additionalProperties"] = False
        result["required"] = list(properties.keys())
    return result


def _json_schema_for(model_cls, name: str) -> dict:
    return {
        "name": name,
        "schema": _strictify(model_cls.model_json_schema()),
        "strict": True,
    }


VALID_EXPLICITNESS = {"explicit", "tangential", "absent"}


def _repair_parsed_json(parsed, schema_cls):
    """Best-effort repair of a parsed JSON object against schema_cls's
    required fields before validation.

    Models are unreliable about including every required field consistently
    even when instructed to (missing `reasoning`, a `relevance_score` sent as
    a fractional float, an `explicitness` outside the allowed enum, etc.) --
    retrying the identical call over and over just burns the retry budget on
    the same failure. This fills only safe, non-fabricating defaults (empty
    string/zero/empty list/"absent") for missing or invalid required fields,
    and coerces an out-of-range float score to the nearest valid int -- it
    never invents substantive content like reasoning or excerpts.
    """
    if not isinstance(parsed, dict):
        return parsed
    parsed = dict(parsed)

    for name, field in schema_cls.model_fields.items():
        if not field.is_required() or name in parsed:
            continue
        if field.annotation is str:
            parsed[name] = ""
        elif field.annotation is int:
            parsed[name] = 0
        elif get_origin(field.annotation) is list:
            parsed[name] = []

    # `explicitness` is a Literal enum, not a plain str/int/list, so the
    # generic loop above doesn't cover it -- handle missing AND out-of-enum
    # values here (covers both cases with one check).
    if "explicitness" in schema_cls.model_fields and parsed.get("explicitness") not in VALID_EXPLICITNESS:
        parsed["explicitness"] = "absent"

    # The 0-100 bound is enforced by Pydantic but deliberately not sent on the
    # wire (strict mode rejects minimum/maximum), so an out-of-range answer
    # reaches us intact. Clamping beats burning a retry on an off-by-a-little
    # score; a fractional score is rounded the same way.
    score = parsed.get("relevance_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        parsed["relevance_score"] = max(0, min(100, round(score)))

    locations = parsed.get("locations")
    if isinstance(locations, list):
        # start_marker/end_marker are the two things on LocationMarker that
        # can't be safely defaulted -- without them there's nothing to
        # anchor extraction on. Drop any entry missing either rather than
        # fail the whole response.
        parsed["locations"] = [
            loc for loc in locations
            if isinstance(loc, dict) and loc.get("start_marker") and loc.get("end_marker")
        ]

    return parsed


class MissingThemesError(ValueError):
    """The model's answer left out one or more of the requested themes."""


def _repair_multi_theme(parsed, themes: list[Theme], entry_cls=ThemeChunkAnalysis):
    """Repair a ``{"themes": [...]}`` answer against the requested theme list.

    Each entry's ``theme_id`` is matched case-insensitively against a theme's
    id, label or full text (models sometimes echo what they were shown). Unknown and duplicate
    entries are dropped (first wins), each kept entry gets the per-entry
    repair, and entries come back in input-theme order. A theme the model
    skipped can't be defaulted without fabricating an assessment, so it
    raises MissingThemesError and the caller retries.
    """
    if not isinstance(parsed, dict) or not isinstance(parsed.get("themes"), list):
        return parsed

    lookup: dict[str, Theme] = {}
    for theme in themes:
        lookup[theme.id.casefold()] = theme
    for theme in themes:
        lookup.setdefault(theme.name.casefold(), theme)
        lookup.setdefault(theme.text.casefold(), theme)

    by_id: dict[str, dict] = {}
    for entry in parsed["themes"]:
        if not isinstance(entry, dict):
            continue
        key = entry.get("theme_id")
        theme = lookup.get(key.strip().casefold()) if isinstance(key, str) else None
        if theme is None or theme.id in by_id:
            continue
        by_id[theme.id] = _repair_parsed_json({**entry, "theme_id": theme.id}, entry_cls)

    missing = [t.id for t in themes if t.id not in by_id]
    if missing:
        raise MissingThemesError("missing themes: " + ", ".join(missing))

    return {**parsed, "themes": [by_id[t.id] for t in themes]}


class TruncatedResponseError(Exception):
    """The model hit its output cap mid-JSON. Retrying the identical request
    truncates again, so this is never retried and carries a pointed message
    rather than presenting as a generic parse failure."""


class AnalysisFailed(RuntimeError):
    """One model's analysis of one transcript failed. ``stats`` holds every
    call that did return a response, so their cost is still reported."""

    def __init__(self, message: str, stats: "RunStats") -> None:
        super().__init__(message)
        self.stats = stats


def _extract_content(resp, model: str) -> str:
    """Pull the JSON text out of a completion, failing loudly on the response
    shapes that are easy to mistake for a parse error.

    Aggregators like OpenRouter report some upstream failures as HTTP 200 with
    an error payload and no choices, so the SDK raises nothing. Reasoning
    models add a second trap: the answer can be empty while the token budget
    went to a separate reasoning field. Both look like "invalid JSON" if read
    naively, which sends the retry loop chasing the wrong problem.
    """
    error = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"Provider returned an error for {model}: {message}")

    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError(f"Provider returned no choices for {model} (empty response)")

    choice = choices[0]
    # finish_reason is optional in practice: some OpenAI-compatible providers
    # omit it entirely, so it must never be dereferenced directly.
    finish_reason = getattr(choice, "finish_reason", None)
    content = (getattr(getattr(choice, "message", None), "content", None) or "").strip()

    if not content:
        # A reasoning model that spent its whole budget thinking leaves the
        # answer empty; say so, because the fix is a bigger output cap or a
        # non-reasoning model, not another identical attempt.
        extra = getattr(getattr(choice, "message", None), "model_extra", None) or {}
        if extra.get("reasoning") or extra.get("reasoning_content"):
            raise TruncatedResponseError(
                f"{model} returned reasoning but no answer content -- raise or unset "
                f"LLM_MAX_OUTPUT_TOKENS or use a non-reasoning model"
            )
        raise RuntimeError(f"{model} returned empty content (finish_reason={finish_reason!r})")

    if finish_reason == "length":
        raise TruncatedResponseError(
            f"{model} hit the output cap mid-response -- raise or unset LLM_MAX_OUTPUT_TOKENS "
            f"(currently the JSON is cut off and cannot be parsed)"
        )

    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    return content.strip()


async def _call_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    system: str,
    user: str,
    schema_cls,
    schema_name: str,
    max_retries: int,
    max_output_tokens: int | None,
    label: str = "",
    gate: "RequestGate | None" = None,
    repair: Callable[[Any], Any] | None = None,
    extra_body: dict | None = None,
    stats: "RunStats | None" = None,
    call_label: str = "",
    theme_ids: list[str] = (),
):
    """Call the chat completions API, enforcing the schema.

    Every attempt that got a response back is a paid call, so each is logged
    to ``stats`` (retries as ``<call_label>/retryN``). A failed attempt has no
    usable answer to split output cost by, so its themes share it evenly.

    Tries native structured output (response_format=json_schema) first; if
    the provider/model rejects that parameter, falls back to a plain call
    with a strict parse-and-retry loop. Once a model is known to reject it,
    later calls skip straight to the fallback rather than re-paying the
    failed round trip every time.
    """
    # No cap configured: omit the parameter entirely so the model's own
    # maximum applies, rather than sending some arbitrary large number.
    max_tokens = max_output_tokens or openai.NOT_GIVEN
    if repair is None:
        def repair(p):
            return _repair_parsed_json(p, schema_cls)

    async def _plain_call():
        return await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            extra_body=extra_body,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user
                    + "\n\nRespond with ONLY a single valid JSON object matching "
                    "the required schema. No markdown fences, no commentary.",
                },
            ],
        )

    def _record(attempt: int, resp, entries=()) -> None:
        if stats is None:
            return
        label_n = call_label if attempt == 0 else f"{call_label}/retry{attempt}"
        stats.add(
            label_n,
            model,
            getattr(resp, "usage", None),
            theme_ids=theme_ids,
            theme_output_chars=_theme_output_chars(entries),
        )

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        resp = None
        try:
            # The gate is held across the structured-output attempt and its
            # fallback so the pair counts as one unit of concurrency.
            async with (gate if gate is not None else contextlib.nullcontext()):
                if model in _NO_STRUCTURED_OUTPUT:
                    resp = await _plain_call()
                else:
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            extra_body=extra_body,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": _json_schema_for(schema_cls, schema_name),
                            },
                        )
                    except openai.BadRequestError:
                        # Provider doesn't support structured output mode -- fall back
                        # to plain JSON-mode prompting and a strict parse retry below.
                        resp = await _plain_call()
                        # Only remember the model as unsupported once the fallback
                        # has actually succeeded: a BadRequestError from something
                        # else (oversized input, bad params) would otherwise
                        # permanently disable structured output for no reason.
                        if model not in _NO_STRUCTURED_OUTPUT:
                            _NO_STRUCTURED_OUTPUT.add(model)
                            logger.info(
                                "%s rejected structured output (response_format=json_schema); "
                                "using JSON-mode prompting for the rest of this run",
                                model,
                            )

            content = _extract_content(resp, model)
            parsed = json.loads(content)
            parsed = repair(parsed)
            result = schema_cls.model_validate(parsed)
            _record(attempt, resp, getattr(result, "themes", ()))
            return result, resp.usage
        except TruncatedResponseError as exc:
            if getattr(resp, "usage", None) is not None:
                _record(attempt, resp)
            if max_output_tokens is None:
                # No cap of ours: the model's own output ceiling, which the
                # same request hits every time.
                raise TruncatedResponseError(
                    f"{model} ran out of output space answering {len(theme_ids)} themes "
                    f"— lower MAX_THEMES or use a model with a larger output limit"
                ) from exc
            raise
        except RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            # Exponential backoff, jittered so concurrent workers that were
            # rate-limited together don't retry in lockstep and re-trip the
            # limit. A server-supplied Retry-After overrides the guess.
            delay = min(2 ** attempt + random.uniform(0, 1), 30)
            hinted = _retry_after_seconds(exc)
            if hinted is not None:
                delay = max(delay, hinted)
            if isinstance(exc, openai.RateLimitError) and gate is not None:
                # Hold every other in-flight worker back too, not just this
                # one -- see RequestGate for why.
                gate.pause(delay)
            logger.warning(
                "[%s] Retryable error on attempt %d/%d: %s. Sleeping %.1fs",
                label, attempt + 1, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
        except openai.APIStatusError:
            # Non-retryable API error (bad request, auth, billing/quota, not
            # found, etc.) -- retrying identical input against these just
            # burns the whole budget for a guaranteed-identical failure.
            raise
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001 - parse/validation retry
            last_exc = exc
            # Only a response that reports usage can have been charged.
            if getattr(resp, "usage", None) is not None:
                _record(attempt, resp)
            logger.warning(
                "[%s] Parse/validation error on attempt %d/%d: %s",
                label, attempt + 1, max_retries, exc,
            )
            await asyncio.sleep(min(1.5 ** attempt, 10))

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}") from last_exc


_EXPLICITNESS_RANK = {"absent": 0, "tangential": 1, "explicit": 2}


def _theme_output_chars(entries) -> dict[str, int]:
    """Each theme's share of the answer, used to split output cost."""
    return {entry.theme_id: len(entry.model_dump_json()) for entry in entries}


async def _analyze_single_pass(
    client: AsyncOpenAI,
    model: str,
    themes: list[Theme],
    transcript: str,
    config: Config,
    stats: RunStats,
    progress: Progress | None = None,
    gate: RequestGate | None = None,
) -> list[AnalysisResult]:
    if progress:
        progress.start("single-pass")
    user = prompts.MULTI_SINGLE_PASS_USER_TEMPLATE.format(
        theme_block=prompts.format_theme_block(themes), transcript=transcript
    )
    result, _ = await _call_with_retry(
        client,
        model=model,
        system=prompts.MULTI_THEME_ANALYSIS_SYSTEM_PROMPT_V2,
        user=user,
        schema_cls=MultiThemeChunkAnalysis,
        schema_name="multi_theme_chunk_analysis",
        max_retries=config.max_retries,
        max_output_tokens=config.max_output_tokens,
        label=progress.label if progress else "",
        gate=gate,
        repair=lambda p: _repair_multi_theme(p, themes),
        extra_body=openrouter_extra_body(config),
        stats=stats,
        call_label="single_pass",
        theme_ids=[t.id for t in themes],
    )
    if progress:
        progress.note("single-pass call complete")

    by_id = {entry.theme_id: entry for entry in result.themes}
    results = []
    for theme in themes:
        entry = by_id[theme.id]
        locations = []
        for marker in entry.locations:
            excerpt = extract_passage(transcript, marker.start_marker, marker.end_marker)
            if excerpt is None:
                continue
            locations.append(
                Location(
                    excerpt=excerpt,
                    title=marker.title,
                    timestamp=marker.timestamp,
                    speaker=marker.speaker,
                )
            )
        results.append(
            AnalysisResult(
                theme=theme.name,
                theme_id=theme.id,
                relevance_score=entry.relevance_score,
                explicitness=entry.explicitness,
                reasoning=entry.reasoning,
                locations=locations,
                model_used=model,
                chunked=False,
            )
        )
    return results


async def _analyze_one_chunk(
    client: AsyncOpenAI,
    model: str,
    themes: list[Theme],
    chunk: Chunk,
    total_chunks: int,
    config: Config,
    stats: RunStats,
    semaphore: asyncio.Semaphore,
    progress: Progress | None = None,
    gate: RequestGate | None = None,
) -> tuple[dict[str, ThemeChunkAnalysis], Chunk]:
    """Analyze one chunk for every theme -> ``({theme_id: entry}, chunk)``."""
    async with semaphore:
        user = prompts.MULTI_CHUNK_USER_TEMPLATE.format(
            theme_block=prompts.format_theme_block(themes),
            chunk_index=chunk.index + 1,
            total_chunks=total_chunks,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            chunk_text=chunk.text,
        )
        result, _ = await _call_with_retry(
            client,
            model=model,
            system=prompts.MULTI_THEME_ANALYSIS_SYSTEM_PROMPT_V2,
            user=user,
            schema_cls=MultiThemeChunkAnalysis,
            schema_name="multi_theme_chunk_analysis",
            max_retries=config.max_retries,
            max_output_tokens=config.max_output_tokens,
            label=progress.label if progress else "",
            gate=gate,
            repair=lambda p: _repair_multi_theme(p, themes),
            extra_body=openrouter_extra_body(config),
            stats=stats,
            call_label=f"chunk_{chunk.index}",
            theme_ids=[t.id for t in themes],
        )
        if progress:
            progress.step()
        return {entry.theme_id: entry for entry in result.themes}, chunk


async def _synthesize_reasoning(
    client: AsyncOpenAI,
    model: str,
    themes: list[Theme],
    final_scores: dict[str, int],
    chunk_results: list[tuple[dict[str, ThemeChunkAnalysis], Chunk]],
    config: Config,
    stats: RunStats,
    progress: Progress | None = None,
    gate: RequestGate | None = None,
) -> dict[str, str]:
    """One call that writes every theme's final reasoning -> ``{theme_id: reasoning}``."""
    ordered = sorted(chunk_results, key=lambda pair: pair[1].index)
    sections = []
    for theme in themes:
        lines = []
        for entries, chunk in ordered:
            result = entries[theme.id]
            lines.append(
                f"- Segment {chunk.index + 1} (chars {chunk.char_start}-{chunk.char_end}, "
                f"score {result.relevance_score}, {result.explicitness}): {result.reasoning}"
            )
        sections.append(
            f"THEME id: {theme.id} | {theme.text}\n"
            f"OVERALL COMPUTED SCORE: {final_scores[theme.id]}\n"
            f"PER-SEGMENT PARTIAL ANALYSES (in transcript order):\n" + "\n".join(lines)
        )
    user = (
        f"THEMES:\n{prompts.format_theme_block(themes)}\n\n" + "\n\n".join(sections)
    )

    result, _ = await _call_with_retry(
        client,
        model=model,
        system=prompts.MULTI_THEME_SYNTHESIS_SYSTEM_PROMPT_V2,
        user=user,
        schema_cls=MultiThemeSynthesis,
        schema_name="multi_theme_synthesis",
        max_retries=config.max_retries,
        max_output_tokens=config.max_output_tokens,
        label=progress.label if progress else "",
        gate=gate,
        repair=lambda p: _repair_multi_theme(p, themes, entry_cls=ThemeSynthesis),
        extra_body=openrouter_extra_body(config),
        stats=stats,
        call_label="synthesis",
        theme_ids=[t.id for t in themes],
    )
    return {entry.theme_id: entry.reasoning for entry in result.themes}


def _extract_and_fill_location(marker: LocationMarker, chunk) -> Location | None:
    """Extracts the full passage text for one chunk-relative marker, and
    fills in timestamp/speaker from the chunk's nearest match when the
    model didn't report one itself. Returns None if the marker's boundaries
    couldn't be located in the chunk's text (dropped, same policy as a
    missing excerpt)."""
    excerpt = extract_passage(chunk.text, marker.start_marker, marker.end_marker)
    if excerpt is None:
        return None
    return Location(
        excerpt=excerpt,
        title=marker.title,
        timestamp=marker.timestamp or chunk.nearest_timestamp,
        speaker=marker.speaker or chunk.nearest_speaker,
    )


async def _analyze_map_reduce(
    client: AsyncOpenAI,
    model: str,
    themes: list[Theme],
    transcript: str,
    config: Config,
    stats: RunStats,
    progress: Progress | None = None,
    gate: RequestGate | None = None,
) -> list[AnalysisResult]:
    chunks = chunk_transcript(transcript, config.chunk_size_tokens, config.chunk_overlap_tokens)
    semaphore = asyncio.Semaphore(config.max_concurrent_chunks)
    if progress:
        progress.start(f"chunked (max {config.max_concurrent_chunks} concurrent)", len(chunks))

    tasks = [
        _analyze_one_chunk(
            client, model, themes, chunk, len(chunks), config, stats, semaphore, progress, gate
        )
        for chunk in chunks
    ]
    # Let every chunk finish before failing: a sibling still in flight is
    # paid for either way, and only a finished one has its usage recorded.
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    for outcome in gathered:
        if isinstance(outcome, BaseException):
            raise outcome
    chunk_results = gathered
    lengths = [c.char_end - c.char_start for _, c in chunk_results]

    # Per theme: score, passages and explicitness, each from that theme's own
    # entries only.
    final_scores: dict[str, int] = {}
    passages: dict[str, list[Location]] = {}
    explicitness: dict[str, str] = {}
    for theme in themes:
        analyses = [entries[theme.id] for entries, _ in chunk_results]
        final_scores[theme.id] = compute_aggregate_score(analyses, lengths)
        passages[theme.id] = merge_and_dedupe_locations([
            loc
            for entries, chunk in chunk_results
            for marker in entries[theme.id].locations
            if (loc := _extract_and_fill_location(marker, chunk)) is not None
        ])
        # Strongest across chunks: one explicit chunk makes the theme explicit.
        explicitness[theme.id] = max(
            (a.explicitness for a in analyses), key=_EXPLICITNESS_RANK.__getitem__
        )
    if progress:
        progress.note(
            f"all chunks done, {len(themes)} themes, "
            f"{sum(len(p) for p in passages.values())} passages — synthesizing reasoning"
        )

    reasonings = await _synthesize_reasoning(
        client, model, themes, final_scores, chunk_results, config, stats, progress, gate
    )

    return [
        AnalysisResult(
            theme=theme.name,
            theme_id=theme.id,
            relevance_score=final_scores[theme.id],
            explicitness=explicitness[theme.id],
            reasoning=reasonings[theme.id],
            locations=passages[theme.id],
            model_used=model,
            chunked=True,
        )
        for theme in themes
    ]


async def analyze(
    client: AsyncOpenAI,
    model: str,
    themes: list[Theme],
    transcript: str,
    config: Config,
    progress: Progress | None = None,
    gate: RequestGate | None = None,
) -> tuple[list[AnalysisResult], RunStats]:
    """Entry point: picks the single-pass fast path or the chunked map-reduce
    path. Every theme goes in the same call(s); one result per theme, in
    theme order."""
    stats = RunStats()
    estimated_tokens = estimate_tokens(transcript)
    try:
        if estimated_tokens <= config.single_pass_token_limit:
            results = await _analyze_single_pass(
                client, model, themes, transcript, config, stats, progress, gate
            )
        else:
            results = await _analyze_map_reduce(
                client, model, themes, transcript, config, stats, progress, gate
            )
    except Exception as exc:
        # Carry the calls that were made (and paid for) out with the failure.
        raise AnalysisFailed(str(exc), stats) from exc
    return results, stats
