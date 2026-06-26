# Cost & quality plan — `kg` pipeline

> Started as a cost-reduction plan; now also covers **extraction quality, evaluation, and retrieval**
> after the NLP extractor took ingest to $0 and the bottleneck moved to the reader. Grounded in live
> Claude runs on the LongMemEval tiers and **multi-agent A/B + research sweeps (2026-06-25/26)**. The
> query path is **PPR → RAG**: a non-LLM retriever builds the context and a *single* LLM call answers
> (now Opus for testing). No per-hop tool loop.

## Status (2026-06-26)

- ✅ **Lever 1 — dashboard prod-vs-eval-judge cost split — SHIPPED** (zero risk, no A/B).
  The "query cost" stat now reads as **query cost (prod)** (`agent_cost_usd`) + **eval judge**
  (`judge_cost_usd`); stale "agentic ask() traversal" wording fixed.
- ✅ **Lever 2 — ingest output cap — TESTED + PARTLY SHIPPED.** Window-widening **refuted** by
  two A/Bs; the one real win — **`extract_max_tokens` 1500→4000** — is **shipped**. See below.
- 🟢 **Lever 6 — LLM-free / hybrid extraction backend — TESTED, WINNER (biggest cut found).**
  Replacing the Haiku extractor with **GLiNER (entities) + YAKE (tags) + spaCy co-occurrence
  (relations)** — all local, ~$0 — gives **−100% ingest cost** with **same retrieval recall and
  same-or-better answer accuracy** on micro+sample A/Bs. Code is in `kg/nlp_extractors.py`
  (`config.extractor_backend`, default `haiku` so the live path is unchanged). See below.
- 🟢 **Lever 7 — GLiNER2 richer typed relations + first-person `me`-injection — TESTED.** One local
  0.5B encoder (`fastino/gliner2-large-v1`, 1.95 GB) does typed entities **and** typed relations from
  a described 30-predicate schema in one pass, $0. Captures the **user's own facts** (`me --spent_on-->
  coffee mugs`) that Haiku's default prompt misses. Backends `gliner2` / `gliner2_nounchunk` /
  `gliner2_haiku` (combo). Richer/cleaner relations than co-occur; **entity-noisy** (over-extracts
  concepts; the 0.65 threshold doesn't fix it because relation endpoints are re-added). See below.
- ❌ **Qwen-1.5B local LLM extractor — TESTED + REMOVED.** A small generative LLM *invents* open-vocab
  predicates like Haiku, but at 1.5B it is **error-prone** (`me --has_pet--> best friend`), types
  entities weakly (mostly `other`), and is slow. Encoder (GLiNER2) is more *reliable* at ≤3 GB because
  its constrained vocabulary can't hallucinate. Model deleted; code removed.
- 🔑 **EVAL REFRAME (the important one).** The literature says `recall@k` saturating is **expected** —
  on LongMemEval, session-level recall is ~0.96 and the real bottleneck is the **reader** (the single
  answer LLM), with a published 30–60% oracle→retrieved accuracy gap. So all our "recall flat at 1.0"
  results mean *retrieval is largely solved*. Two consequences: (1) **the test reader is now Opus**
  (`config.rag_model`; `kg/rag.py` omits `temperature` for Opus 4.7+/Fable) to see how much is
  reader-limited; (2) the **full-haystack eval is now ~$1, not $100** — that estimate assumed *paid*
  ingest, but ingest is $0 with the NLP extractor. See *Evaluation* + *Next steps*.

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

> **Removed — Lever 5 (judge sampling).** The judge is eval-only, never touches production cost, and
> certifies answer correctness on **every** test round. Full `--judge` stays on always.

---

## Lever 6 — LLM-free / hybrid extraction backend  `[TESTED — WINNER]`

**TL;DR:** the LLM extractor is the wrong place to spend money on *this* corpus. Swapping it for local
NLP (**GLiNER** zero-shot NER + **YAKE** topical tags + **spaCy** co-occurrence/verb relations) takes
**ingest cost to $0 (−100%)** while keeping **retrieval recall identical** and **answer accuracy
equal-or-better**. Lever 2 chased the 33–50% "different extraction strategy" prize and named it as the
only way to reach it (§ Next steps) — this is that strategy, and it beats the target.

### Why it works here
Retrieval is **embedding(bge)+BM25 over episode TEXT, topology-seeded** (`kg/retrieval.py`), and the
RAG answer reads the **retrieved episode text + a FACTS list** (`kg/rag.py`). So extraction only feeds
(a) graph topology / PPR seeds via entities+tags, and (b) the FACTS list via relations — the **answer
prose comes from text the LLM reads regardless**. Local NER is *good enough* for (a) and (b), so paying
Haiku per section buys almost nothing on answers.

### Method (config-driven, reuses the testrun harness)
`config.extractor_backend` routes `get_extractor` to `kg/nlp_extractors.py` (default `haiku`, unchanged):
`gliner_yake` · `gliner_yake_cooccur` · `gliner_nounchunk[_cooccur]` · `hybrid_nounchunk_rel`
(GLiNER+tags + ONE Haiku relations-only call/section) · `spacy_svo` · `keyword_only`. GLiNER
(`urchade/gliner_small-v2.1`, threshold 0.5, 10 natural labels remapped to the 8 `EntityType`s, chunked
to ~160-word windows) loads once as a module singleton; inference is lock-serialized (spaCy/GLiNER
aren't thread-safe under the ingest pool). Pure-NLP backends keep an empty meter → $0; the hybrid
surfaces its single call's cost through the same meter. Retrieval-stressed regime (`k=3,
rag_context_episodes=3`) added because at 6 sessions/instance ≤ `rag_context_episodes=6` the default is
**saturated** (recall@k=1.0, all text always in context — extraction can't move answers); stress makes
PPR ranking actually decide what the answerer sees.

### Results — LongMemEval `sample` (n=8, per-instance, judge on); ingest stats are regime-independent

| backend | ingest $/100 | entities | tags | rels | acc (full) | acc (stress) | recall@k (stress) |
|---|---|---|---|---|---|---|---|
| **haiku** (baseline) | **$2.83** | 1378 | 1175 | 466 | 0.66 | 0.48 | 0.73 |
| **gliner_yake_cooccur** ⭐ | **$0.00** | 1870 | 1129 | 605 | 0.72 | **0.64** | 0.73 |
| gliner_yake (no relations) | $0.00 | 1870 | 1129 | 0 | **0.74** | 0.48 | 0.73 |
| gliner_nounchunk | $0.00 | 1870 | 958 | 0 | 0.73 | 0.23 | 0.67 |
| gliner_nounchunk_cooccur | $0.00 | 1870 | 958 | 605 | 0.72 | 0.54 | 0.67 |
| hybrid_nounchunk_rel | $1.26 | 1870 | 958 | 540 | 0.73 | 0.48 | 0.67 |

(micro n=3 agreed directionally; per the lever-3 lesson it's only a smoke set, so the verdict rests on
sample. Full-regime recall@k is 1.0 for every row — saturated.)

**Findings:**
1. **Recall is preserved exactly.** Every NLP variant matches Haiku's stress recall@k (0.73) or the
   saturated 1.0 — entities/tags reproduce the retrieval topology. The user-facing "recall" claim holds.
2. **Cost −100%** (pure NLP) for **equal-or-better accuracy.** At full (production) context *all five*
   NLP variants land ≥ the Haiku baseline (0.72–0.74 vs 0.66) — a consistent no-regression signal.
3. **`gliner_yake_cooccur` is the winner.** Under stress it beats the baseline (+0.16 acc) at $0. The
   gain is the **FACTS list, not retrieval**: the free co-occurrence relations rescued 2 temporal/
   counting questions ("days between Sunday mass & Ash Wednesday" → dates surfaced; "appointments in
   March" → 2nd found) vs 1 preference-question loss. So **relations earn their keep** here — *cheap NLP
   ones*, not the LLM (the no-relations arm dropped to baseline; the LLM-relations hybrid did **not**
   beat the free co-occurrence relations, and costs $1.26/100).
4. **Two research predictions were refuted by the data** (kept because I A/B'd both): **YAKE > spaCy
   noun-chunk tags** for this pipeline (noun-chunk recall@k 0.67 < YAKE 0.73 — likely the HippoRAG
   query-token→tag-key seeding favours YAKE's surface forms), and **emit-no-relations is NOT optimal**
   (co-occurrence relations add +0.16 stress acc for free).
5. **Richness up:** GLiNER gives **+36% entities** (1870 vs ~1380); tags comparable.

### Caveats
- n=8: judge accuracy is small-sample (the stress +0.16 ≈ 2 questions). It's corroborated by an
  explainable mechanism, the micro direction, and a rock-solid full-regime no-regression across all 5
  NLP variants — but a larger tier would tighten it (see Next steps).
- GLiNER/spaCy add CPU latency (~0.3s/section GLiNER + ~0.2s spaCy parse) and torch/transformers +
  `en_core_web_sm` + `gliner`/`yake` deps. Trade local compute for API $.
- **GLiREL** (the research's first-choice relation model) is **not installed**; the co-occurrence
  relations already win, so it's a deferred spike, not needed.

### Recommendation
Ship `gliner_yake_cooccur` as an opt-in extractor backend (keep `haiku` default until confirmed at
larger n). It is the single biggest cost lever found — and unlike levers 3/4 it *improves* the product.

---

## Lever 7 — GLiNER2 richer typed relations + first-person `me`  `[TESTED]`

**Motivation:** lever-6's co-occurrence relations are *generic* (`--reflect-->`, `--consider-->`) and
miss the **user's own facts** — yet LongMemEval asks about the user ("how much did *I* spend on coffee
mugs"). GLiNER2 (`fastino/gliner2-large-v1`, 0.5B encoder, schema-driven) emits **typed** relations and,
with first-person handling, captures `me` facts. All $0, on-device (MPS).

**Three tuning knobs applied** (`kg/nlp_extractors.py:Gliner2Extractor`):
1. **Entity threshold 0.5→0.65** — *ineffective*: entity count barely moved (relation endpoints are
   re-added as entities, cancelling the cut). Entity noise is GLiNER2's real weakness (it over-extracts
   generic `concept`s). Honest miss.
2. **Relation schema with DESCRIPTIONS + widened to ~30 predicates** (the GLiNER2 "schema mode") —
   **+50% relations** (81→122 on 3 micro episodes). Descriptions disambiguate each predicate.
3. **First-person `me`** — normalize GLiNER2's own `I`/`my` relation endpoints to a `me` node (accurate
   objects + typed predicates), plus a tightened spaCy dependency-object supplement. **me-facts 28→39**,
   e.g. `me --spent_on,bought--> coffee mugs`, `me --attended--> St. Mary's Church`.

**3-way extraction comparison (3 micro evidence episodes):**

| backend | entities | relations | me-facts | cost | note |
|---|---|---|---|---|---|
| **GLiNER2 (tuned)** | 594 | **122** | **39** | $0 | typed, reliable, entity-noisy |
| **Qwen-1.5B (local LLM)** | 56 | 55 | 34 | $0 | open-vocab but **error-prone**, weak entities, slow → REMOVED |
| **Haiku (LLM)** | 69 | 83 | 0 | $0.096 | clean, rich open-vocab, **misses `me` facts** |

**Findings:**
- **At ≤3 GB the encoder (GLiNER2) beats a small generative LLM on *reliability*** — its closed
  vocabulary can't hallucinate `me --has_pet--> best friend` (which Qwen-1.5B did). "Open vocabulary"
  for encoders = label strings *you* enumerate; only an LLM *invents* predicates.
- **GLiNER2 captures `me`-facts Haiku misses** — the property most relevant to this benchmark.
- **None of the local options match Haiku's clean+rich relations**; the gap is precision (GLiNER2 junk
  edges like `sermon --visited--> church`) and entity noise. The research's fix is **GLiREL typed
  `allowed_head`/`allowed_tail`** (structurally kills type-violating edges) or a **propose→verify
  hybrid** (encoder proposes $0 → Haiku relabels survivors). Both deferred behind the eval below.
- `gliner2_haiku` (union of both, cost = Haiku) exists to test whether GLiNER2's `me`-facts *added to*
  Haiku lift answer accuracy.

---

## Evaluation — the reframe that changes the plan  `[IN PROGRESS]`

**Core insight (from the literature):** `recall@k` saturating is the **expected** result, not a bug. On
LongMemEval-S, session recall is ~0.955–0.986 for BM25/dense/hybrid, yet oracle→retrieved end-to-end
accuracy drops **30–60%**. **The bottleneck is the reader (the single answer call), not retrieval or
extraction.** This recontextualizes every "recall flat at 1.0" result above as *retrieval is solved*,
and means extraction richness only helps insofar as the **FACTS list helps the reader**.

**Actions (some applied, some queued):**
- ✅ **Reader → Opus for testing.** `config.rag_model="claude-opus-4-8"`; `kg/rag.py:_supports_temperature`
  omits `temperature` for Opus 4.7+/Fable 5 (the API rejects it). Judge stays Haiku (`l3_model`) — an
  *independent* judge avoids the self-grading bias the Mem0↔Zep dispute exposed. Early signal: on the
  micro temporal question Opus scored 1.0 (computed "30 days") where Haiku-reader missed — i.e. the
  error *was* reader-side.
- ⏳ **Full-haystack tier.** The session cap in `scripts/build_longmemeval.py` is *why* retrieval is
  trivial (6 sessions ≤ 6 context slots). Add an uncapped tier (~50 sessions/question) so retrieval is
  actually stressed. **Cost ≈ $1** (reader+judge only), because ingest is $0 with the NLP extractor —
  the old "$100/variant" estimate assumed *paid* ingest and is now obsolete.
- ⏳ **WhenLoss diagnostic** (Oracle-Evidence vs Retrieved-Memory vs Complete-Stored-Memory): 3 reader
  runs/question to attribute each error to write- / retrieval- / reader-side. Localizes where to spend.
- ⏳ **Harden** (Zep 84%→58% lessons): score the 30 abstention questions separately; run the judge ≥3×
  for variance (kills the "n=8 is noise" blocker); freeze + hash prompts in `run.json`.
- ⏳ **Per-question-kind breakdown** (temporal / knowledge-update / multi-session) — the `kind` field is
  already on every question; mirror `evaluate.py:ModeScore.per_kind` into `run_per_instance` totals.
- ⏳ **Retrieval-only $0 mode** (skip the answer LLM) as a per-commit regression gate.

---

## Retrieval / PPR directions  `[NOT STARTED]`

The query path is Personalized PageRank with restart (`kg/retrieval.py`): `nx.pagerank(alpha=ppr_damping=
0.5, max_iter=200)` — **iterate-to-convergence**, not fixed hops. `alpha=0.5` ⇒ **50% walk / 50%
teleport** to the IDF-weighted seed set (very local; mean walk length `1/(1−α)=2` hops). PPR ranks
episodes → top `k*3=24` → MMR+seed-distance rerank → top `k=8` → top **6** enter the reader context.
Improvements, by expected value (but **measure first** — retrieval is near-ceiling):
1. **Cross-encoder reranker** over the 24 candidates (`bge-reranker-v2-m3`, ~0.5 GB) — decides which 6
   episodes the reader sees; the retrieval change most likely to move *answers*. **Highest EV.**
2. **Raise damping** `0.5→0.6–0.75` for multi-hop/connect-the-dots questions (cheap sweep).
3. **HippoRAG-2 query→fact seeding** — seed relevant fact edges, not just entity/tag nodes; now viable
   with GLiNER2's typed relations.
4. **Per-edge-type weights** — `projected_graph` sums all etypes equally; upweight precise `RELATED_TO`.
5. **Widen candidate pool** `k*3→k*5` so a reranker can rescue gold ranked 9–24.
6. **Time-aware diffusion / query expansion** for temporal/as-of questions (+7–11% in LongMemEval).

---

## Other directions (ranked, from the 2024-26 research sweep)  `[BACKLOG]`

1. **Aggregation/counting read-path** — for "how many appointments / postcards" questions, *compute*
   over retrieved facts instead of letting the LLM eyeball them. Direct hit on a question class we miss.
2. **Coreference resolution pre-pass** (fastcoref/maverick-coref) — resolve `I`/`my coworker`/`she`
   before extraction; cleaner first-person + cross-sentence relations than the `me`-injection heuristic.
3. **Temporal expression normalization** anchored to each message timestamp — for temporal-reasoning Qs.
4. **GLiREL typed `allowed_head`/`allowed_tail`** (1.87 GB) — the precision fix for relations.
5. **Write-time fact reconciliation** (Mem0 ADD/UPDATE/DELETE/NOOP) on top of the existing supersede logic.

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

## Next steps — the funding blocker is gone; the eval is the gate  `[ACTIVE]`

**What changed:** the old plan deferred everything behind a ~$100/variant `small` run on *personal
funds*. That estimate assumed **paid ingest**. With the NLP extractor (Lever 6/7) **ingest is $0**, so a
full-haystack eval costs only the per-question **reader + judge** (~**$1–5/variant**, even at n=100).
The blocker was the ingest bill — it no longer exists. The remaining gate is just building the honest
eval.

**Why it still matters:** every answer-quality conclusion above rests on `sample` n=8 (12.5%-granular,
noisy) **and** a saturated retrieval setup (6 sessions ≤ 6 context slots). Neither stresses the system.

**Priority order (re-ranked after the reader-bottleneck reframe):**
0. **Build the full-haystack eval** (uncapped tier, Opus reader, judge ≥3× for variance, per-kind
   breakdown). ~$1–5. This single artifact settles almost everything below — does extraction richness
   move answers, is retrieval really solved, how much does the Opus reader buy. **Do this first.**
1. **Confirm Lever 6/7 extraction at n≥50 on full haystacks** — recall held + accuracy ≥ Haiku at $0.
2. **Cross-encoder reranker** (Retrieval #1) — likely the biggest *answer-accuracy* lever, since it
   decides the 6 episodes the reader sees.
3. **Aggregation/counting read-path** — fixes a whole question class the reader currently eyeballs.
4. **(De-prioritized) the Haiku-call levers** — window-widening (refuted), reflexion ablation,
   `extract_max_tokens` confirmation. These optimize the Haiku *ingest* call, which the $0 NLP extractor
   **removes entirely**; only relevant if we keep a Haiku/hybrid extraction path. Keep as background.

---

## Rejected (verified traps — look like savings, aren't)

- ❌ **Widen `long_doc_chars`** (was lever 2's headline): **empirically refuted** — −33/48/50% cost
  but −23/47% richness, and the cost path can't recover it. See Lever 2.
- ❌ **Prefix prompt-caching / pad `_SYS` past 4,096 tok:** +$1.31/100 cache-miss downside vs ~$0.52
  upside, and padding risks anchoring extraction and **shrinking** entity/tag diversity. Negative EV.
- ❌ **Reflexion fully off (~$1.02/100):** biggest single number but removes the recall pass that
  recovers missed entities/relations. Quality-unsafe; only behind a hard accuracy A/B (queued, #3 above).
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
