# Brief: "Migrate the Transcript Insight Pipeline from Claude to OpenRouter"

**What this file is.** Not the proposal. This is the *instruction set* you hand to an AI
(or a writer) so it produces the proposal. It carries the verified facts, the real cost
numbers, the argument structure, the counterarguments to pre-empt, and the guardrails
against overclaiming. Everything in **Part A** is fact-checked against this repo and can
be quoted. Everything in **Part D** must be re-verified live before the document ships.

---

## Part 0 — The commission (paste this to the AI)

> Write a decision proposal for the leadership of Gospel Light House recommending that
> the Transcript Theme Extraction pipeline run on **OpenRouter** (model-agnostic, paid
> per token) rather than on **Anthropic's Claude** exclusively — and that the default
> production model be a low-cost tier (DeepSeek or Gemini Flash class) with premium
> models reserved for spot-checks and adjudication.
>
> Audience: non-engineering decision-makers who control budget. They are intelligent but
> not technical. They currently associate "Claude" with "the good AI" and will need to be
> walked from that intuition to a more accurate model of how this market works.
>
> The document must be persuasive *and* honest. It must survive a skeptical reader who
> asks "what's the catch?" — so it names the trade-offs itself rather than being caught
> hiding them.
>
> Length: 8–14 pages equivalent. Include tables. Lead with a one-page executive summary
> that a reader can act on without reading anything else.

---

## Part A — Verified facts about our system (safe to quote)

These come from reading the code, not from memory. Cite them confidently.

### A1. What the pipeline actually does

For each transcript, for a given theme, the system:

1. Estimates the transcript's token length (`chunker.py`, ~3.5 chars/token).
2. **If under 20,000 tokens** → one single API call analyses the whole transcript.
3. **If over 20,000 tokens** → splits into 12,000-token chunks with 800-token overlap,
   analyses each chunk in parallel, then makes **one extra synthesis call** to merge the
   partial findings into a final judgement.
4. Each call returns strict JSON: a 0–100 relevance score, an explicitness rating
   (explicit / tangential / absent), a reasoning paragraph, and a list of passage
   locations marked by exact verbatim start/end phrases.
5. Those markers are used to cut the real passages out of the source text, which are
   assembled into a Word report (`word_report.py`).

### A2. The system is already provider-agnostic — this is the crux

`client.py` builds a **standard OpenAI-SDK client pointed at a configurable
`base_url`**. `config.py` already defaults to:

```
LLM_PROVIDER      = openrouter
LLM_BASE_URL      = https://openrouter.ai/api/v1
LLM_DEFAULT_MODEL = deepseek/deepseek-v4-flash
```

**Switching models is a configuration change, not an engineering project.** There is no
Anthropic-specific code to rip out. This single fact defuses the biggest objection a
budget-holder has to any migration — "how much will the switch itself cost us?" The
answer is: it is already done; we are asking for permission to *use* it.

Tell the AI to make this a headline point, not a footnote.

### A3. We can prove model parity rather than assert it

`transcript_theme_analyzer/eval/compare_models.py` runs the *same* transcript and theme
across several models simultaneously and prints a side-by-side table of score, token
usage and latency. The pipeline also enforces a strict JSON schema with retries
(`analyzer.py`), so a model that cannot hold the output format fails loudly and visibly
rather than silently degrading.

This turns "trust me, DeepSeek is good enough" into "here is the comparison table."
**The proposal should include an offer to run this evaluation on a 30–50 transcript
sample before full commitment.** That is the single most credible thing in the document.

### A4. Cost-shaping controls already built in

- Concurrency is capped globally (`max_concurrent_requests`, default 12) so a run cannot
  stampede into rate limits or runaway spend.
- **Resumable checkpointing** (`checkpoint.py`) means an interrupted run resumes instead
  of re-paying for completed work.
- A response cache (`cache.py`) avoids re-billing identical calls.
- `--limit N` allows a costed pilot on a subset before committing to the full corpus.

Frame these as *fiscal governance*, not features. Leadership cares that spend is bounded
and auditable.

---

## Part B — The cost model (real numbers, reproducible)

A calculator lives at [tools/cost_model.py](../tools/cost_model.py). It derives token
volume from the pipeline's own configuration constants — chunk size, overlap, the
1,655-token system prompt, an 8% retry allowance — and prices it. **Anyone can re-run it
and get the same answer.** Include that fact; it is what makes the numbers credible
rather than rhetorical.

```
python tools/cost_model.py --transcripts 1800 --avg-minutes 55
```

### B1. Volume per full-corpus run — 1,800 transcripts, one theme

| Avg. transcript | Tokens each | API calls | Input tokens | Output tokens |
|---|---|---|---|---|
| 30 min | ~6,100 | 1,800 | 15.1 M | 1.2 M |
| 55 min | ~11,100 | 1,800 | 24.9 M | 1.2 M |
| 90 min | ~18,200 | 1,800 | 38.7 M | 1.2 M |
| 150 min *(chunks)* | ~30,400 | 7,200 | 81.2 M | 3.9 M |

Note the cliff at 150 minutes: past the 20,000-token threshold the transcript is split
into 3 overlapping chunks plus a synthesis call — **4 API calls instead of 1**, and the
overlap means we pay for some text twice. Long transcripts are disproportionately
expensive on *any* provider. This matters because it is exactly where an expensive
per-token rate hurts most.

### B2. Cost per full-corpus run, by model tier

**55-minute average transcript (our central estimate):**

| Model tier | Cost per full run | Cost per transcript |
|---|---:|---:|
| Claude Opus (frontier) | **$461.65** | $0.2565 |
| Claude Sonnet (mid) | $92.33 | $0.0513 |
| Claude Haiku (small) | $30.78 | $0.0171 |
| **DeepSeek V3-class (OpenRouter)** | **$8.27** | $0.0046 |
| Gemini Flash-class (OpenRouter) | $10.40 | $0.0058 |
| Gemini Flash-Lite-class | $2.96 | $0.0016 |

**150-minute average transcript (worst case, chunked):**

| Model tier | Cost per full run |
|---|---:|
| Claude Opus | **$1,512.49** |
| Claude Sonnet | $302.50 |
| Claude Haiku | $100.83 |
| DeepSeek V3-class | $27.06 |
| Gemini Flash-class | $34.18 |
| Gemini Flash-Lite-class | $9.69 |

### B3. The multiplier that decides the argument

**Roughly 56× between Claude Opus and DeepSeek. Roughly 11× between Claude Sonnet and
DeepSeek.**

Instruct the AI to translate this into the terms leadership actually thinks in, because
$461 does not sound frightening in isolation. The real argument is **iteration count**:

- This is not a one-time job. Every theme is a **full run**. Ten themes = ten runs.
- Every prompt refinement, every schema fix, every re-analysis after a corpus update is
  another run.
- At Opus rates, **10 themes ≈ $4,600**, and a single bad prompt discovered afterwards
  costs another $460 to correct.
- At DeepSeek rates, **10 themes ≈ $83**, and a mistake costs $8 to fix.

> **The frame to use:** the expensive model does not merely cost more — it makes us
> *afraid to run the pipeline*. A budget that punishes iteration produces worse research
> output, because the team stops exploring themes it cannot justify the invoice for. The
> cheap model buys us permission to be thorough.

This is the strongest argument in the entire proposal. It should appear in the executive
summary.

### B4. Also model these scenarios

Have the AI produce a table for:
- 1 theme × 1,800 transcripts (baseline)
- 10 themes × 1,800 transcripts (realistic first-year research programme)
- 10 themes + 3 full re-runs after prompt revisions (realistic *with* iteration)
- A hybrid: DeepSeek for all 1,800, plus Claude Sonnet re-scoring only the top 150
  highest-relevance transcripts for quality assurance

The hybrid line is important — see Part C4.

---

## Part C — The argument structure

Five movements. Do not reorder them; each one clears the objection the next one would
otherwise trip on.

### C1. "The model is not the product"

Develop this carefully — it is the intellectual core, and the user's framing of it needs
sharpening to survive scrutiny.

**Do not write:** "all AI models are basically the same intelligence." A sharp reader
knows this is false and will discard the whole document over it.

**Write instead:** capability at the *frontier* varies, but our task does not live at the
frontier. Our pipeline asks a model to do one bounded, well-specified thing: read a
segment of text, judge its relevance to a theme, and emit strictly-formatted JSON with
verbatim quote boundaries. This is squarely inside what the entire current generation of
mid-tier models does reliably. We are not asking for novel reasoning, long-horizon
planning, or original research. **We are paying frontier prices for a commodity
operation.**

Supporting points:

- The gap between the best and the tenth-best model on *structured extraction* has
  narrowed to near-irrelevance, while the price gap has widened. That divergence is the
  entire arbitrage.
- Where a task genuinely does need frontier capability, OpenRouter still gives us Claude
  Opus — **on the same API key, same code path, one config line.** Choosing OpenRouter is
  not choosing against Claude. It is refusing to be *locked into* Claude for work that
  does not need it. Make this explicit; it is disarming and it is true.
- Our JSON schema enforcement is the real quality floor. A model that can't comply fails
  visibly and we swap it. The architecture protects us from a bad model choice.

### C2. "We are currently buying the wrapper, not the engine"

Distinguish three things leadership currently conflates:

1. **The model** — the raw intelligence, sold per token.
2. **The product wrapper** — the Claude chat interface, Projects, file uploads, the
   Claude Code CLI. Sold per seat, per month.
3. **Our own infrastructure** — the chunker, the concurrency gate, the retry-and-schema
   layer, the checkpointing, the aggregation and Word report generation. **This is the
   asset we have built and the thing that actually creates the deliverable.**

The chat interface is superb at what it is for: one person, one conversation, a handful
of documents. It is structurally the wrong tool for 1,800 transcripts, because:

- It has no programmatic loop — a human must be present for every unit of work.
- Context windows and upload limits force manual chunking; a person becomes the chunker.
- No checkpointing: an interrupted session is lost work.
- Output is prose in a chat window, not validated JSON that flows into a report builder.
- No cost telemetry per unit of work — you cannot cost a run, only a subscription.
- Nothing is reproducible or auditable. You cannot re-run last quarter's analysis and get
  the same answer, or show a reviewer *how* a passage was selected.

Land the point plainly: **a subscription seat scales with people. An API key scales with
work.** Our bottleneck is work, not people.

### C3. "One key, many uses" — the platform argument

An OpenRouter key is not a line-item for this project. It is shared organisational
infrastructure. The same credential and the same balance can power, with no new
procurement, no new vendor, no new contract, and no new approval cycle:

- This transcript pipeline.
- A member-facing chatbot or Q&A assistant over the sermon archive.
- Automated summarisation, tagging, and metadata for the media library.
- Search and semantic retrieval across the corpus.
- Drafting assistance inside internal tools (the CRM, admin workflows).
- Translation and transcription support.
- Any future application — through a single OpenAI-compatible endpoint every major
  framework already speaks.

Add the governance angle, which finance and leadership will value more than engineers do:
**one vendor relationship, one prepaid balance, one dashboard showing per-application
spend.** The alternative — a separate subscription per tool per team — is both more
expensive and far harder to audit. Pre-paid credits also mean **spend cannot exceed what
was deliberately loaded.** There is no surprise invoice. Say this explicitly; it is the
sentence that closes a budget-holder.

### C4. "Freedom to choose is itself the deliverable"

The durable argument is not "DeepSeek is cheaper today." Prices and rankings move every
few months. The durable argument is **optionality**:

- Hundreds of models behind one endpoint and one key.
- Switching model is one config value — proven, because it is how the system is already
  built (A2).
- Automatic failover: if one provider has an outage or rate-limits us, the run continues
  on another. A single-provider setup simply stops.
- Price competition works *for* us continuously. When a cheaper or better model ships, we
  adopt it the same week instead of renegotiating a contract.
- **No lock-in.** If Claude becomes the right answer next year, we move back — through
  the same key, same code.

Then present the **hybrid tiering** proposal, which is what a sophisticated reader will
respect most, and which pre-empts the "you're cutting corners" accusation:

> Cheap, fast models do the bulk pass over all 1,800 transcripts. A premium model
> re-scores only the small subset that matters most — the highest-relevance transcripts
> destined for publication — as quality assurance. We get frontier-grade confidence
> exactly where it counts, at a fraction of a percent of the frontier-grade bill.

This reframes the proposal from *cost-cutting* to *intelligent allocation*. Make sure the
AI understands the difference and writes it that way.

### C5. The ask

State plainly what is being requested. Suggested:

1. Approve an initial OpenRouter credit purchase (recommend a specific figure — a $50–100
   starting balance funds several full runs and all evaluation work with wide margin).
2. Approve a **paid pilot**: run `compare_models.py` plus a `--limit 50` sample across
   DeepSeek, a Gemini Flash tier, and Claude Sonnet as control. Report back cost, latency
   and score agreement. Estimated pilot cost: single-digit dollars.
3. On satisfactory pilot results, adopt the cheap tier as the production default with the
   hybrid QA pass from C4.
4. Retain the existing Claude access for interactive/exploratory work — this is not a
   proposal to cancel it, and saying so removes the sense of threat.

---

## Part D — What the AI must verify before the document ships

**Do not let the AI publish my price table as current fact.** Prices in Part B were
correct to the best of available knowledge at the time of writing and move frequently.
Instruct it to:

1. Fetch live per-million-token input and output rates from **anthropic.com/pricing** and
   **openrouter.ai/models**.
2. Confirm the exact model IDs and their real rates, specifically:
   - `deepseek/deepseek-v4-flash` — the ID currently configured as our default in
     `config.py`. Confirm it exists and get its price; substitute the nearest current
     DeepSeek tier if not.
   - The current Gemini Flash and Flash-Lite tiers on OpenRouter.
   - Current Claude Opus / Sonnet / Haiku rates.
3. Check whether **prompt caching** is available on the chosen models. Our 1,655-token
   system prompt is identical on every single call — caching it could cut input cost
   materially on a 1,800-call run. If available, add it as a further saving and as an
   implementation to-do.
4. Note OpenRouter's routing fee/margin on top of upstream model price, so the figures are
   honest and nobody discovers it later.
5. Re-run `python tools/cost_model.py` with the verified prices edited into its `PRICES`
   table, and use *those* outputs in the document.
6. Confirm the real corpus statistics — replace the 55-minute assumption with the actual
   average transcript length once measured. The calculator takes `--avg-minutes`.

---

## Part E — Counterarguments to raise and answer *in the document*

A proposal that hides these looks naive; one that names them looks authoritative. For
each, state the objection fairly, then answer it.

| Objection | Answer to give |
|---|---|
| "Cheap models will produce worse analysis." | We do not have to speculate. `compare_models.py` measures it on our own transcripts. Approve the pilot; decide on evidence. Add the hybrid QA tier (C4). |
| "Isn't DeepSeek a Chinese model — data sovereignty?" | Address directly, do not dodge. OpenRouter exposes provider routing and data-retention policies per model, and can be constrained. Note our data is public sermon content, not personal or confidential data. If the concern persists, Gemini Flash-Lite is a Western-hosted alternative at comparable cost — that option existing *is itself the argument for OpenRouter*. |
| "Another vendor to manage." | It replaces per-tool subscriptions with one prepaid account. Fewer relationships, not more, and one dashboard for all AI spend. |
| "What if OpenRouter goes down or shuts down?" | Standard OpenAI-compatible API. Redirecting to a provider's direct endpoint is a `base_url` change. The abstraction is the protection. |
| "We already pay for Claude — this is duplicate spend." | The subscription serves interactive human work and continues. This is batch machine work it structurally cannot do. Different jobs, and the batch job is where the 56× applies. |
| "Reliability of exotic models / rate limits." | Schema validation with retries catches malformed output; a global concurrency cap and checkpointing bound the damage. Failure is visible and resumable, not silent. |
| "Quality of output formatting/JSON compliance varies by model." | True, and it is the *real* selection criterion — not raw intelligence. The pilot measures exactly this. Any model that fails is disqualified at pilot stage, at a cost of a few dollars. |

---

## Part F — Required document structure

1. **Executive summary** — one page. The ask, the number (56× / 11×), the iteration
   argument from B3, the pilot proposal, the reassurance that Claude access is retained.
2. **Current state** — what the pipeline does, corpus scale, what a "run" means.
3. **The cost analysis** — tables from Part B, methodology stated, calculator referenced.
4. **Why the price gap exists** — C1, the commodity-operation argument.
5. **Interface vs. API vs. our own infrastructure** — C2.
6. **The platform case: one key, many applications** — C3.
7. **Optionality and the hybrid tiering strategy** — C4.
8. **Risks and objections** — Part E, as a table.
9. **The recommendation and pilot plan** — C5, with figures and a timeline.
10. **Appendix** — model comparison table, pricing sources with access dates, cost model
    assumptions listed explicitly (chars-per-token, retry allowance, output-token
    average) so any assumption can be challenged individually.

---

## Part G — Guardrails (things the AI must not do)

- **No invented benchmark scores.** Do not claim "DeepSeek scores X% on MMLU" unless it is
  fetched and cited. The argument does not need benchmarks; it needs the pilot.
- **No claim that models are equally intelligent.** Use the bounded-task framing from C1.
- **No disparaging Claude.** The document is stronger for conceding that Claude is
  excellent and that its interface is genuinely best-in-class for interactive work. The
  argument is about *fit and cost*, not quality. An attack invites defence of the
  incumbent; a concession invites agreement.
- **No unlabelled estimates.** Every number is either sourced-and-dated or explicitly
  marked as a modelled estimate with its assumptions shown.
- **No hidden trade-offs.** If cheap models are slower, or have lower rate limits, or need
  more retries — say so and quantify it.
- **State the corpus assumption prominently.** If the real average transcript length
  differs materially from what was modelled, every figure moves. Make that visible rather
  than burying it.
