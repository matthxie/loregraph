# Cost-reduction plan — `kg` pipeline

> Supersedes this file's prior contents, which described the **retired** agentic query
> loop (`kg/agent.py`, since deleted) and the rev5 Wikipedia/COCO corpus. Both are gone.
> This version is grounded in `runs/micro-live/run.json` (live Claude Haiku 4.5,
> LongMemEval *micro* tier) and re-verified by a multi-agent analysis pass. The query
> path is now **PPR → RAG**: a non-LLM retriever builds the context and a *single* LLM
> call answers. There is no per-hop tool loop.

## Where the money goes (baseline)

| Path | Cost | Drivers |
|---|---|---|
| **Ingest** | **$2.80 / 100 sessions** (~78% of spend) | 94 LLM calls = **47 sections × 2 passes** (extract + reflexion). Long sessions are split into 6000-char windows; each window runs extract *and* reflexion. The static **~1,671-token** prefix (`_SYS` + `GRAPH_TOOL`) is re-sent **uncached** on every call (~$0.87/100). Output is ~15% of tokens but **~47% of dollars** (priced 5×). |
| **Query** | **$0.49 / 100 (prod)** + **$0.14 / 100 (eval-judge)** | One PPR→RAG answer call, ~96% input (6 episodes × 1200 chars + 30 fact lines). The judge is an **eval-only** grader (`kg/testrun.py:_judge`) — it certifies answer correctness during testing and is **not paid in production**. |

Framing facts that drive the plan:

1. **Cost is call-count × prefix × output on ingest**, not model choice — Haiku is already the floor.
2. **Prompt caching is a dead end as-is**: the 1,671-tok prefix silently no-ops below Haiku's
   ~4,096-tok cache floor (`cache_read = cache_write = 0`, now visible as the `cache rd/wr`
   stat on the ingest dashboard). Forcing it via prefix-padding is **negative EV** (see Rejected).
3. **The "$0.64/100 query" figure is inflated** — it sums real prod cost ($0.49) with the
   always-on eval judge ($0.14). Lever 1 makes that split permanent and legible.

---

## The levers (1–4)

### 1. Split the dashboard "query cost" into prod vs eval-judge  `[ship now — zero risk]`
- **Simple terms:** the dashboard's single "query cost" secretly bundles the real answer
  call with the eval-only grader, making queries look ~30% more expensive than production
  pays. Show them as two numbers.
- **Change:** both already exist in `run.json` (`agent_cost_usd` + `judge_cost_usd`,
  `kg/testrun.py:764`). Only the dashboard adds them blindly — at `kg/dashboard.py:750`
  replace the one stat with **"query cost (prod)" = `T.agent_cost_usd`** and
  **"eval judge" = `T.judge_cost_usd`** (`?? T.cost_usd` fallback for old runs). Also fixes
  the stale "agent / agentic ask() traversal" wording at `kg/dashboard.py:743,750`.
- **Saves:** $0 (clarity). With the judge running every round, this is what keeps the
  testing-only dollars distinct from product cost on every run.
- **Risk:** none. **No A/B.**

### 2. Widen the extraction window (`long_doc_chars` 6000 → 12000, then sweep)  `[A/B, biggest win]`
- **Simple terms:** long sessions get chopped into 6000-char chunks and the full
  extraction runs **twice on every chunk** (normal pass + a "did I miss anything?" recall
  pass), each re-sending the same ~1,671-token preamble. Bigger chunks = fewer chunks =
  the preamble is paid fewer times.
- **Why 12000 specifically (and not the model's context max):** the context window is
  **not** the binding limit — Haiku holds ~200K tokens, so any session fits whole. Three
  *other* limits bind, none of them context size:
  1. **The existing input cap.** `extract_text` already truncates a call at `text[:12000]`
     (`kg/extractors.py:345`). At window = 12000 every slice is ≤12000 → the cap is a
     **no-op, zero text dropped**. At 12001+ each slice exceeds the cap and is **silently
     truncated** (a 19,361-char session at window=24000 loses **38%** with no error). So
     12000 is the largest window reachable by changing **one config line** and nothing else.
  2. **The output cap, not the input, truncates the graph.** Each call emits the whole
     `emit_graph` payload under `max_tokens=1500` (`kg/extractors.py:304`). Feed a 50K-char
     session and it must emit every entity/relation in one 1500-token answer → it hits the
     ceiling and **silently drops graph content** (the expensive 5× output side).
  3. **Recall decays on long inputs** ("lost in the middle"): the extractor finds fewer
     entities per unit text as input grows. Sectioning exists to keep extraction sharp.
- **Change:** make **both** `config.long_doc_chars` (`kg/config.py:26`, default → 12000)
  **and** the `text[:12000]` cap (`kg/extractors.py:345`) config-driven, then **sweep**
  window = 12000 / 18000 / whole-session in the A/B and pick the knee where entity/relation
  recall starts dropping. 12000 is the zero-risk floor; the sweep finds whether bigger is
  safe instead of guessing. Keep `max_sections=6` as the safety net.
- **Saves:** **~$0.89 / 100 (~32%)** at 12000 (sections 47→30, calls 94→60); potentially
  more if the sweep shows recall holds at larger windows.
- **Risk:** low at 12000 (no text dropped), rising past it (limits #2/#3). **Gate: A/B.**

### 3. Stop re-listing known entities across chunks (output dedup)  `[A/B]`
- **Simple terms:** when a session is chopped into chunks, chunk 2 and chunk 3 re-emit the
  same people/topics chunk 1 already captured — repeated output on the expensive (5×) side.
  Tell later chunks "here's what's already found; emit only what's **new**." It still asks
  for **all relationships** even between already-known entities — we suppress re-*listing*
  the entities, never the facts connecting them (that's what protects richness).
- **Change:** in `extract_text_sectioned` (`kg/extractors.py:367`), for chunks after the
  first, thread the running entity/tag names into the prompt with an "emit only new
  entities/tags, still emit every relation" instruction (the trick reflexion already uses).
  Small new hint param on the extract call. Medium effort (touches the prompt).
- **Saves:** ~$0.12 / 100 standalone — **erodes after lever 2** (fewer chunks = less
  overlap to dedupe). Bank the *measured*, post-widen number.
- **Risk:** low. **Gate: A/B.**

### 4. Trim dead fields from the relation schema  `[A/B, contingent]`
- **Simple terms:** every relation carries a `confidence` number that nothing uses to
  discriminate (the code just defaults it to 0.8, and at temperature 0 it's a near-constant)
  plus `valid_from`/`valid_to` slots usually filled with empty placeholders. Removing them
  means less boilerplate on the expensive output side, with zero loss of real information.
- **Change:** drop `confidence` from `GRAPH_TOOL`'s relation schema (`kg/extractors.py:153`;
  parser already defaults it at `:216`) and tighten the date descriptions to "omit unless an
  explicit date is stated." Keep `source/target/labels/status` and real dates.
- **Saves:** ~$0.03 / 100 — **contingent**: `confidence`/dates are not in the schema's
  `required` list, so the model may already omit them at temp 0, in which case ≈ $0.
- **Risk:** very low. **Gate:** diff the bi-temporal fact edges on the temporal instance
  (`08f4fc43`) — `valid_at`/`invalid_at`/`status` byte-identical to baseline.

> **Removed — Lever 5 (judge sampling).** The judge is eval-only and never touches
> production cost, and it certifies answer correctness, which we want on **every** test
> round. Full `--judge` stays on always. Lever 1 makes its always-on overhead legible.

---

## The shared A/B (one harness, reused for levers 2–4)

Levers 2–4 gate on the same measurement. Run `kg testrun --mode per-instance` on the
**`sample`** tier (not just micro — micro is n=3 and unrepresentative), baseline vs. change,
and compare from `run.json`:

- **Ingestion richness:** `ingest.totals.vocab.entities / relations / tags` and
  `avg_tags_per_object` — require within ~5%.
- **Query quality:** `query.totals.recall_at_k`, `citation_grounding`, `response_accuracy`,
  `response_token_recall` — require non-regressing.
- **Cost + caching sanity:** `ingest.totals.cost_usd` down as predicted; `cache rd/wr`
  stays `0 / 0` (confirms we didn't accidentally engage caching).

Ship each change only at the **knee** where cost drops but richness/accuracy hold.

### Double-count guards (do not stack naïvely)
- **Prefix pool:** lever 2 already drains the re-sent-prefix savings — caching and batching
  cannot be added on top (their savings are mostly captured by lever 2).
- **Output pool:** levers 3 and 4 both shrink *after* lever 2 (fewer chunks / fewer raw
  relations). Measure realized, not headline, savings.

### Execution order
1. **Ship lever 1** now (zero risk).
2. Build the config-driven window + A/B harness.
3. **A/B lever 2** (window, with the sweep) — the ~32% win.
4. **A/B levers 3 & 4** on top; measure realized post-widen savings.

**Realistic result if 1–4 land:** ~**31%+ off ingest** ($2.80 → ~$1.9/100) and a permanent
prod-vs-eval query split. The judge stays on for testing.

---

## Rejected (verified traps — look like savings, aren't)

- ❌ **Prefix prompt-caching / pad `_SYS` past 4,096 tok:** +$1.31/100 cache-miss downside vs
  ~$0.52 upside, and few-shot padding risks anchoring extraction and **shrinking** entity/tag
  diversity. Negative EV. (Only defensible if the added tokens independently lift recall.)
- ❌ **Reflexion fully off (~$1.02/100):** biggest single number but removes the recall pass
  whose job is recovering missed entities/relations on a temporal-reasoning corpus.
  Quality-unsafe; a *gated* variant is possible but only behind a hard A/B (not in this plan).
- ❌ **Single un-sectioned call / window past 12000 without lifting the cap:** silently drops
  up to ~38% of long sessions at the `text[:12000]` cap.
- ❌ **Cross-session batching (pack many sessions per call):** collapses differently-dated
  episodes onto one `created_at`, corrupting the bi-temporal `valid_at/invalid_at` layer.
- ❌ **Lower `rag_context_episodes` below 6 / `rag_max_facts` 30→15:** the 2nd gold episode
  sits at rank 6; the fact list isn't relevance-ranked — both risk turning a correct answer
  wrong for little/no saving.
- ❌ **Lower `max_tokens` (ingest 1500 / query 1024):** output *ceilings* billed only on
  emitted tokens (avg ~495 / ~164) → **$0 saved**, only truncation risk.
- ❌ **Post-merge tag cap:** runs *after* tokens are billed → $0 output saved while deleting
  captured tags (richness violation) and thinning SHARED_TAG bridges.
- ❌ **Route the judge to Sonnet:** +$0.28/100 for a metric, not the product. No tier below Haiku.
