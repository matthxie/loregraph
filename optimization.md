# Cost-reduction plan — `kg` pipeline

> Supersedes this file's prior contents (the retired agentic query loop + the rev5
> Wikipedia/COCO corpus, both deleted). Grounded in live Claude Haiku 4.5 runs on the
> LongMemEval tiers and **two multi-agent A/B sweeps (2026-06-25)**. The query path is
> **PPR → RAG**: a non-LLM retriever builds the context and a *single* LLM call answers.
> There is no per-hop tool loop.

## Status (2026-06-25)

- ✅ **Lever 1 — dashboard prod-vs-eval-judge cost split — SHIPPED** (zero risk, no A/B).
  The "query cost" stat now reads as **query cost (prod)** (`agent_cost_usd`) + **eval judge**
  (`judge_cost_usd`); stale "agentic ask() traversal" wording fixed.
- ✅ **Lever 2 — ingest output cap — TESTED + PARTLY SHIPPED.** Window-widening **refuted** by
  two A/Bs; the one real win — **`extract_max_tokens` 1500→4000** — is **shipped**. See below.
- ⏸ **Levers 3–4 — UNTESTED, deferred.** Harness is built; live test runs are paused (on
  personal funds until the business account is live — see *Next steps*).

## Where the money goes (baseline)

| Path | Cost | Drivers |
|---|---|---|
| **Ingest** | **$2.80 / 100 sessions** (~78% of spend) | 94 LLM calls = **47 sections × 2 passes** (extract + reflexion). Long sessions split into 6000-char windows; each window runs extract *and* reflexion. The static **~1,671-token** prefix (`_SYS` + `GRAPH_TOOL`) is re-sent **uncached** on every call (~$0.87/100). Output is ~15% of tokens but **~47% of dollars** (priced 5×). |
| **Query** | **$0.49 / 100 (prod)** + **$0.14 / 100 (eval-judge)** | One PPR→RAG answer call, ~96% input. The judge is an **eval-only** grader (`kg/testrun.py:_judge`) — certifies answer correctness during testing, **not paid in production**. Lever 1 (shipped) makes this split legible on the dashboard. |

Framing facts that drive the plan:

1. **Cost is call-count × prefix × output on ingest**, not model choice — Haiku is the floor.
2. **Prompt caching is a dead end as-is**: the 1,671-tok prefix silently no-ops below Haiku's
   ~4,096-tok cache floor (`cache rd/wr = 0/0`, confirmed across every A/B cell). Forcing it via
   prefix-padding is **negative EV** (see Rejected).
3. The **output side is where truncation hides**, but it is *small*: only ~2–3% of calls ever hit
   the 1500-tok ceiling. The big ingest waste is the re-sent **input** prefix — which lever 2 tried
   and failed to drain (below).

---

## Lever 2 — ingest output cap & extraction window  `[TESTED]`

**TL;DR:** widening the extraction window to pay the re-sent prefix fewer times *looks* like the
biggest win (~33–50% cheaper ingest) but **destroys graph richness** and is **dead**. The only
banked win is raising the *emit cap* at the **current** window — a near-free +3–4% richness.

### ✅ Shipped: `extract_max_tokens` 1500 → 4000
The `emit_graph` call had a hardcoded `max_tokens=1500` output ceiling. On the shipped 6000-char
window, **8 of 246 extract calls hit it** and were silently truncated mid-JSON (tail entities/tags
dropped — content the model generated and we paid for, then discarded). Raising the cap to **4000**
clears all 8 and recovers **+3.2% entities / +4.4% tags / +3.7% avg-tags-per-object for +0.2% cost
(+$0.003)**. Output is billed only on emitted tokens, so the cap is a ceiling, not a target.
**4000 is the knee** — richness and output plateau there (`mt8000` recovers nothing further).
Now `kg/config.py:extract_max_tokens=4000`. Artifacts: `runs/ab-mt/*`.

### ✗ Refuted: widening `long_doc_chars`
The hypothesis: each 6000-char section re-sends the ~1,671-tok prefix on *both* its extract and
reflexion passes; bigger sections = fewer passes = less prefix cost. Two A/Bs killed it:

- **Window sweep (cap 1500), `runs/ab-l2/*`:** 6000→12000→18000→whole cut ingest cost
  **−33 / −48 / −50%** (cache 0/0) but lost **23–47% of graph richness** (entities/relations/tags/
  avg-tags-per-object), monotonically worse with width. `recall@k` flat at 1.0 (coarse,
  session-level); `response_accuracy` n=8-noisy.
- **max_tokens sweep — falsified the truncation theory, `runs/ab-mt/*`:** the suspicion was that
  the 1500 cap was truncating the larger sections. **Wrong.** At window=12000, lifting the cap
  1500→4000→8000 drove truncations 4→0→0 but recovered **almost nothing** — entities +1.8% total,
  still **−22% vs the 6000 baseline at mt8000 with ZERO truncation**; output tokens rose only 0.7%.
  The loss is therefore an **input-window recall-decay effect** ("lost in the middle" — the
  extractor surfaces fewer entities per character from a larger section), **not** output truncation.

**Conclusion: keep `long_doc_chars=6000`.** Widening is dead for two independent reasons (richness
decays with width; the cheap path can't recover it). The 33–50% cost prize is real but unreachable
by sectioning size alone — it needs a different extraction strategy (see *Next steps*).

**Caveat on the verdict's strength — it rests on a proxy.** The anti-widening case is built on
**richness**, not answers. At n=8, `response_accuracy` is noise (12.5%-granular; it swung *inversely*
to richness, which is incoherent). We have NOT proven the lost entities hurt *answers* — only that a
quarter-to-half of the graph disappears. If those entities are redundant, widening could be a free
33–50% cut. **This is the #1 open question for the larger run.**

---

## Levers 3 & 4 — output side  `[UNTESTED · harness ready · deferred]`

Both target the expensive 5× output side. The config-driven A/B harness + `truncated` metric from
lever 2 now exist, so these are a flag-flip away from being measured — but they are **deferred**
until the larger run (testing is paused, see *Next steps*).

### 3. Stop re-listing known entities across chunks (output dedup)
For chunks after the first, thread the running entity/tag names into the prompt with an "emit only
**new** entities/tags, still emit **every** relation" instruction (the trick reflexion already uses)
in `extract_text_sectioned` (`kg/extractors.py:367`). Suppresses re-*listing* entities, never the
facts connecting them. **Saves ~$0.12/100 standalone — erodes now that widening is dead** (chunks
stay at 6000, so cross-chunk overlap is modest). Bank the *measured* number. Risk: low. **Gate: A/B.**

### 4. Trim dead fields from the relation schema  `[contingent]`
Drop `confidence` from `GRAPH_TOOL`'s relation schema (`kg/extractors.py:153`; parser already
defaults it at `:216`) and tighten date descriptions to "omit unless an explicit date is stated."
Keep `source/target/labels/status` + real dates. **Saves ~$0.03/100 — contingent**: these fields
are not `required`, so the model may already omit them at temp 0 (then ≈ $0). Risk: very low.
**Gate:** diff the bi-temporal fact edges on the temporal instance (`08f4fc43`) — `valid_at`/
`invalid_at`/`status` byte-identical to baseline.

> **Removed — Lever 5 (judge sampling).** The judge is eval-only, never touches production cost, and
> certifies answer correctness on **every** test round. Full `--judge` stays on always.

---

## The A/B harness (built — reuse for any future lever)

The config-driven harness is in place. Run on a LongMemEval tier, baseline vs change, compare `run.json`:

```
kg testrun --mode per-instance --tier <tier> [--queries N] \
  --long-doc-chars <W> --extract-max-chars <C> --extract-max-tokens <M> \
  --store store/<label>.db --out runs/<dir> --label <label>
```

New knobs (all default to shipped behavior): `--long-doc-chars` (window), `--extract-max-chars`
(per-call input cap), `--extract-max-tokens` (emit cap). `run.json` ingest totals now include a
**`truncated`** count (extract calls that hit `max_tokens`) — the metric that finds the right cap
(raise until `truncated`→0 **and** richness plateaus). Dashboard: `kg dashboard --out runs/<dir>`.

Compare on:
- **Richness** (stable, aggregated): `ingest.totals.vocab.entities/relations/tags`, `avg_tags_per_object`.
- **Quality** (only meaningful at **n ≳ 50**; at n=8 treat as noise): `query.totals.recall_at_k`,
  `citation_grounding`, `response_accuracy`, `response_token_recall`.
- **Cost + caching:** `ingest.totals.cost_usd`, `truncated`, `cache_read/write` (expect 0/0).

---

## Next steps — larger training run  `[DEFERRED ⏸]`

**Blocker:** test runs are live Claude (real $). **Paused while on personal funds — resume when the
business account is live.** Everything below is queued, not running.

**Why bigger:** every conclusion above that touches *answer quality* is unproven, because `sample`
is **n=8** — `response_accuracy` moves in 12.5% steps and is pure noise. Richness and cost are stable
(aggregated over 150–250 calls / 1,000+ entities); accuracy is not. To turn the proxy arguments into
answer-level evidence you need a tier where accuracy can discriminate ~5% differences (**n ≳ 50–100**).

**The run:** LongMemEval `small` (n=100). Note `small` has **no session cap** (4,723 episodes vs
sample's 48), so a *full* per-instance pass is ~$100+/variant. Cheaper read: `--tier small --queries
40` (~40 questions at capped instance count) for a real-ish accuracy signal at a fraction of cost.

**What to test, in priority order:**
1. **Re-litigate window-widening on accuracy (highest $ value).** Does `long_doc_chars=12000`'s
   23–47% richness drop actually lower `response_accuracy`/`recall@k`, or were the lost entities
   redundant? If accuracy holds, widening unlocks a **33–50% ingest cut** — the single biggest lever.
   The n=8 "don't widen" verdict is provisional pending this.
2. **Confirm the shipped free win on accuracy.** Verify `extract_max_tokens=4000` ≥ baseline accuracy
   (today it's justified on "stop discarding truncated output" + richness, not on answers).
3. **Levers 3 & 4** (output dedup, schema trim) — A/B on the bigger tier; measure realized savings.
4. **Reflexion ablation (gated).** The recall pass is ~$1/100 — the biggest single ingest line. Test
   conditional/off against accuracy; quality-unsafe to decide at n=8, decidable at n=100.

---

## Rejected (verified traps — look like savings, aren't)

- ❌ **Widen `long_doc_chars`** (was lever 2's headline): **empirically refuted** — −33/48/50% cost
  but −23/47% richness, and the cost path can't recover it. See Lever 2.
- ❌ **Prefix prompt-caching / pad `_SYS` past 4,096 tok:** +$1.31/100 cache-miss downside vs ~$0.52
  upside, and padding risks anchoring extraction and **shrinking** entity/tag diversity. Negative EV.
- ❌ **Reflexion fully off (~$1.02/100):** biggest single number but removes the recall pass that
  recovers missed entities/relations. Quality-unsafe; only behind a hard accuracy A/B (queued, #4 above).
- ❌ **Single un-sectioned call / window past the input cap without lifting it:** silently drops the
  tail of long sessions (now config-driven via `--extract-max-chars`, but widening is dead anyway).
- ❌ **Cross-session batching (pack many sessions per call):** collapses differently-dated episodes
  onto one `created_at`, corrupting the bi-temporal `valid_at/invalid_at` layer.
- ❌ **Lower `rag_context_episodes` below 6 / `rag_max_facts` 30→15:** the 2nd gold episode sits at
  rank 6 and the fact list isn't relevance-ranked — both risk turning a correct answer wrong for ~$0.
- ❌ **Lower ingest `max_tokens`:** billed only on emitted tokens → **$0 saved**, only truncation risk.
  (Conversely, **raising** it 1500→4000 IS the shipped win — it recovers truncated graph; see Lever 2.)
- ❌ **Post-merge tag cap:** runs *after* tokens are billed → $0 output saved while deleting captured
  tags (richness violation) and thinning SHARED_TAG bridges.
- ❌ **Route the judge to Sonnet:** +$0.28/100 for a metric, not the product. No tier below Haiku.
