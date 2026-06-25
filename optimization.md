# Optimization plan — `kg` pipeline

Grounded in the **`rev5-demo`** dashboard run (`runs/rev5-demo/run.json`): 200 mixed
docs ingested + 12 eval questions, Haiku 4.5 extractor/agent + local `bge-small`
embedder. Every item below cites a run metric and a code location, and carries an
**adversarially-verified** impact estimate (a separate pass re-checked each number
against the data and code; several headline claims were corrected downward — those
corrections are baked in here).

> Scope note: this is a *plan only*. No code changes are proposed to be made here.

---

## The numbers this is built on

| | tokens | cost | share of $ | notes |
|---|---|---|---|---|
| **Ingest** | 997,740 | **$1.81** | **78%** | 2.0 LLM calls/doc (extract + reflexion); 80% input *tokens* but **56% of ingest *dollars* is output** (output priced 5×) |
| **Query** | 479,746 | **$0.52** | **22%** | **98% input tokens** — the agent re-sends the whole growing conversation every step; 6.42 steps & 7.25 calls/query |
| Communities / judge | — | $0 | 0% | local / judge was off |
| **Total** | 1,477,486 | **$2.33** | | wall-clock ingest 1207s (≈6.0s/doc, **sequential** in the testrun harness) |

Three framing facts that drive everything:

1. **Cost is an ingest problem** (78%), and within ingest the two levers are *reflexion*
   (the 2nd LLM call) and the *uncached, re-sent prompt prefix*. **No prompt caching
   exists anywhere** (`cache_read=0`, `cache_write=0`).
2. **Query cost is almost entirely re-sent input** (98%) — caching the agent's
   conversation prefix is the single cleanest query lever, but query is only 22% of total
   spend, so it's a *recurring* win of modest absolute size.
3. **Speed (1207s) is mostly a harness artifact.** The dashboard `testrun` ingests
   **one doc at a time**, forfeiting the production pipeline's 5-wide concurrency.
   `kg ingest` is already ~3–5× faster. Don't optimize a number production doesn't pay.

And one accuracy fact that must not be misread:

4. **`recall@k = 0.167` is a slice artifact, not a tuning failure.** Only 2 of the 12
   gold articles were in the 200-doc slice; both were retrieved (recall **2/2 = 100%**
   conditioned on the gold being present). The other 10 are physically absent from the
   graph. **Do not tune retrieval against this run.**

---

## Priority shortlist (start here)

| # | Lever | Category | Verified impact | Effort | Confidence |
|---|---|---|---|---|---|
| 1 | **Conditional / off reflexion** | cost + speed | ~−31–45% ingest $ (~$3.5–5.0 at full 1243-doc scale); halves LLM calls | S | High mechanism, **benefit unmeasured — needs A/B** |
| 2 | **Prompt-cache the agent's conversation prefix** | cost | ~−57–61% *query* $ (~$1.7–1.8 per full 68-q eval, recurring); lower latency | S | High |
| 3 | **Bump ingest concurrency w/ retry+backoff** (full-corpus runs) | speed | modest single-digit-minute win on full ingest; **OTPM-limited** | S | Medium |
| 4 | **Trim agent `tool_result` payload + cap steps** | cost | ~−15–25% query *input* (~3–4% of total run) | S | Medium |
| 5 | **gzip run.json + stop inlining it into dashboard.html** | tooling | ~9× smaller artifact; removes ~1.65MB/run duplication | M | High |
| 6 | **Ingest prompt caching** — only as *grow-to-4096 + warm reuse* | cost | ~−29% of ingest prefix (~−$3.2–3.3 full) **iff** done right | M | Medium, **easy to make it a net loss** |

Everything else is either at-scale-only, free hygiene, or a do-not-do. Details below.

---

## A. Cost — ingest (78% of spend)

### A1. Reflexion is the #1 cost lever — make it conditional `[HIGH]`
- **What:** every doc makes exactly **2** LLM calls; call #2 is the reflexion recall
  pass. It re-sends the doc text (`text[:4000]`) *and* re-pays the full ~1.8k-token
  prefix, for an **unmeasured** recall gain.
- **Evidence:** `run.json` every step `llm_calls=2` (set `{2}`), total 400 calls / 200
  docs; `kg/extractors.py:272-296` (`_reflexion` re-sends content at :277, re-calls
  `_call` at :288); `kg/config.py:25` `reflexion=True`.
- **Proposal:** gate reflexion on signal — e.g. only run it when the first pass found
  `< N` entities, or skip it for short docs (median real content is ~324 tokens, where a
  single pass already saturates). Default it off if the A/B shows no recall gain.
- **Verified impact:** ~**$0.56–0.81 of the $1.81** 200-doc ingest (~31–45%), i.e.
  **~$3.5–5.0 at full 1243-doc scale**, and **halves LLM calls** (→ wall-clock down on
  the concurrent production path too). Input savings (~45% of ingest input) are solid;
  the output-savings half is softer.
- **Risk / gate:** reflexion exists to catch missed entities/relations. The recall cost
  is **unmeasured here** — A/B `reflexion=False` on the test-graph eval (full corpus)
  before flipping the default. This is the only lever that needs no caching.

### A2. Ingest prompt caching — a trap unless done deliberately `[MEDIUM]`
- **What:** the ~1,783-token static prefix (system + `emit_graph` schema) is re-sent on
  both calls of every doc — ~84–90% of ingest input tokens.
- **Verified reality (live-probed on Haiku 4.5):** `cache_control` **silently no-ops
  below ~4096 tokens** (probed: 1783/2239/.../3943 → no cache; 4663 → caches). So:
  - Adding `cache_control` to today's prefix saves **$0**.
  - Caching only across the extract→reflexion *pair* is a **net loss** (~$6.9 vs $4.4
    full-run: the inflated 4.1k write at 1.25× exceeds 2× the small prefix).
  - It only pays as **grow-the-prefix-past-~4663-then-cache + keep the cache warm across
    docs** (5-min TTL vs ~6s/doc makes cross-doc reuse viable).
- **Proposal:** *if* pursued — add ~2.9k+ tokens of few-shot examples to `_SYS` to cross
  the threshold (target ≥~4663 *written*, not the ~2300 a naive estimate suggests), set
  `cache_control` on the system block, and order ingest so the prefix is written once per
  5-min window and read by all subsequent calls.
- **Verified impact:** ~−$3.2–3.3 on full-corpus ingest (~−29% of the ~$11 prefix),
  slightly worse than ideal because production fans out 5-wide (≈5 cold writes per TTL
  window). **Pair-only caching is a $2.5 loss — warm cross-doc reuse is load-bearing.**
- **Risk:** net loss if the cache doesn't stay warm; added few-shots enlarge every
  *uncached* miss and may shift extraction behavior — eval-gate before merge.

### A3. Don't chase output tokens by lowering `max_tokens` `[MEDIUM]`
- **What:** output is only ~20% of ingest *tokens* but **56% of ingest dollars** (5×
  price). The output is the `emit_graph` payload — the product, not padding.
- **Evidence:** input 794,762 ($0.79, 44%) vs output 202,978 ($1.01, 56%); only 5 docs
  near the 2×1500 cap; `kg/extractors.py:254` `max_tokens=1500`.
- **Verdict:** **do not lower `max_tokens`** — saves ≤~$0.024 (~1% of ingest) while
  risking truncated entities/relations on ~7–19% of docs (a recall regression). Keep
  1500 as a safety ceiling. The real output-side win is A1 (reflexion removal removes its
  whole output contribution).

### A4. (Framing) The static prefix is the gross input, not a standalone saving `[LOW]`
- The 1,783-token prefix × 400 calls = ~90% of ingest input (~$4.4 gross), but the
  *deliverable* lever is A2/A1, not this line on its own. Use it to **deprioritize
  input-only micro-tweaks** in favor of A1 (which cuts a whole call) and A2 (done right).

---

## B. Cost — query / agent (22% of spend)

### B1. Prompt-cache the agent's conversation prefix `[MEDIUM, top pick]`
- **What:** the tool-use loop re-sends the entire growing conversation on every step →
  **98% input tokens** (468,901 in / 10,845 out over 87 calls). System + tools are
  byte-stable (temperature=0, fixed tool list), so the whole prefix is cacheable.
- **Evidence:** `run.json` query totals input 468,901 / output 10,845, `cache_*` all 0;
  `kg/agent.py:658-660` (loop `messages.create`) and `:725-728` (`_force_submit`) pass
  **no** `cache_control`.
- **Proposal:** put a `cache_control:{type:ephemeral}` breakpoint on the last block of
  the most-recently-appended turn before each `messages.create` (and reuse the same
  prefix in `_force_submit`). The breakpoint must ride the *growing message list* —
  caching only the ~1.9k static header no-ops (below the 4096 floor).
- **Verified impact:** ~**−57–61% of query cost** (~$0.52 → ~$0.20 on 12 q; ~$2.96 →
  ~$1.16–1.29 on 68 q ⇒ **~$1.7–1.8 saved per full eval re-run, recurring**). Also cuts
  query latency. **Zero answer-quality change** (byte-identical replay). Caveat: query is
  only 22% of total run cost — hence *medium*, not *high*, despite the clean ~60% win.

### B2. Trim re-sent `tool_result` payloads + tighten the step cap `[MEDIUM]`
- **What:** per-step prefix grows ~1.4k tokens/step, almost all accumulated
  `tool_result` JSON carried forward every step. 4/12 queries exhaust the 8-step cap and
  are exactly the most expensive (q008 = 63.3k input). Cost grows ~quadratically in steps.
- **Evidence:** `kg/agent.py:684` truncates each result to `agent_result_chars=2000`;
  `AGENT_MAX_HITS=12`, `AGENT_MAX_SNIPPET=160` (`:40`); `kg/config.py:76,78`; stop
  distribution `{prose:6, answered:2, step_cap:4}`.
- **Proposal:** (a) drop `agent_result_chars` 2000 → ~1000–1200 and `AGENT_MAX_HITS`
  12 → ~6–8 (search stubs need id+name+score, not full snippets every row); (b) lower
  `agent_max_steps` 8 → **7** — trims only the 4 already-failing step_cap queries
  (**do not go below 7**: it would force-submit q003, the only fully-grounded correct
  answer). Compounds with B1.
- **Verified impact:** ~−15–25% of *query input* (~$0.09 on this slice, ~$0.49 over 68 q
  ⇒ ~3–4% of total run). Step-cap trim saves less than it looks (~$0.18 at cap 5, but
  cap 5 is too aggressive). `agent_result_chars` binds the dominant `seed_and_spread`
  payloads; `AGENT_MAX_HITS` binds few calls.
- **Risk:** smaller stubs could make the model pick a worse object or re-search; A/B on
  `citation_grounding` (untested here — read_object is only 7% of calls on this slice).

---

## C. Speed — wall-clock (mostly harness, not production)

### C1. The 1207s is a sequential-harness artifact `[MEDIUM]`
- **What:** `testrun` calls `g.ingest_object(item)` per doc → `Ingestor.ingest([item])`
  with a **single-element list**, so the `ThreadPoolExecutor(max_workers=5)` only ever
  has 1 work item. Production `kg ingest` passes the whole list → up to 5 docs' extraction
  pipelines overlap.
- **Evidence:** `kg/testrun.py:354`, `kg/graph.py:50-51`, `kg/ingest.py:151-152`,
  `kg/config.py:24`; per-doc time is ~pure LLM latency (corr(seconds, output_tokens)=0.92,
  corr(seconds, doc_index)=0.05).
- **Verified impact:** production is ~**3–5×** faster than the testrun number (200 docs
  ~241–400s; full 1243 ~25–40 min vs ~2.1h sequential extrapolation). 5× is an *upper
  bound* (the 2 calls/doc are serial within a doc; a ~2s/doc non-LLM floor stays serial).
- **Action:** no production code change — just **label the dashboard ingest time as
  "sequential reference, not production"** so 1207s isn't read as the real cost.

### C2. Raise `semaphore_limit` for full-corpus production runs `[MEDIUM]`
- Ingest is LLM-latency-bound and graph mutation is serialized, so wall-clock falls with
  in-flight requests — but **not near-linearly**. `5→10` roughly halves the *extract
  phase* only; serial embed/write/derive phases are fixed.
- **Verified impact:** expect a **modest single-digit-minute** improvement on the full
  ingest from `sl ~8–10`. The `5→20 ≈ 4×` scenario is **not achievable** on standard
  Anthropic tiers — `sl≥10` needs ~100–150k output-tokens/min vs Tier-4's 80k cap.
- **Risk / gate:** a failed item degrades to an *empty* Extraction (`kg/ingest.py:149`),
  so over-pushing **silently drops graph content**. Add retry/backoff before cranking.

### C3. `VectorIndex.add` re-`vstack`s the whole matrix per insert `[MEDIUM, at scale]`
- **What:** every vector insert rebuilds the kind's full `(n×384)` matrix
  (`kg/vectors.py:42-43`). At 200 docs that's ~1.89M row-copies; at full 1243 docs
  (~12k entities) ~**73M** copies — **and the same penalty on every store load**
  (`store.py`), which compounds.
- **Proposal:** amortized growth (doubling capacity buffer + live row count) or a single
  bulk `vstack` per ingest batch (surfaces already arrive grouped).
- **Verified impact:** speed-only, $0. Negligible at 200 docs (~0.2s) but ~**17s** at
  full scale + on every load. Real value is at-scale and on-load.

### C4. Testrun re-derives the whole graph after every doc `[LOW — fidelity, not speed]`
- `_derive_object_edges` runs over the full object set on every `ingest()` →
  in testrun it runs 200× over a growing graph (cumulative ~O(n²)/O(n³)).
- **Verified impact:** only ~**4s of the 1207s** today (~0.3%) — the LLM calls dominate.
  The win is **testrun fidelity** (match `kg ingest`'s derive-once semantics), not speed.
  It would matter if the dashboard run grew an order of magnitude. Production already
  derives once (`kg/ingest.py:135`), so this is harness-only.

### C5. Canonicalization & embedding micro-ops — free hygiene, sub-1% `[LOW]`
Bundle these only when already touching the files; none is a measurable win:
- **`search()` full `argsort` → `argpartition`** for top-k (`kg/vectors.py:67`): ~5–40µs/call, ~0.01% of ingest.
- **Single search reused for merge + SIMILAR_TO** in `resolve_entity` (avoids a 2nd brute-force pass per new entity, `canonicalize.py:445/461→333`): ~70ms total ingest.
- **DATE vectors don't belong in the synonymy-searched `entity` index** (~12% dead rows, `canonicalize.py:433`): clarity/correctness, ~0% speed.
- **Embedding hygiene** (`kg/embedders.py:51`): explicit `batch_size`, clean device fallback (mps is *already* default here), optional surface→vector cache. All ~0.3–1% of ingest wall — **do not over-invest** (LLM latency dominates).

---

## D. Accuracy / retrieval — measure on a full run, not this slice

### D1. `recall@k=0.167` is a missing-document artifact `[document this loudly]`
- 10/10 misses are gold-absent (83% of all queries); present-gold recall = **2/2 =
  100%**. **Tunable retrieval headroom on this slice is exactly 0.** Tuning `seed_k` /
  `mmr_lambda` against it is fitting to noise. (Minor: the chunk→article collapse lives
  in `kg/testrun.py:_article`, not `evaluate.py`.)

### D2. Full-corpus ablation plan for the real levers `[for a future full run]`
On a run where **all gold articles are ingested**, grid these independently and report
**per query-kind** (not aggregate):
- `ppr_damping` 0.5 / 0.7 / 0.85 — higher spreads farther (helps multi-hop, may hurt lookup precision).
- `mmr_lambda` 0.6 / 0.8 — higher = more relevance/less diversity (helps single-gold lookups).
- `seed_k` 10 / 15 — more entry points (recall up, mrr maybe down).
- `top_k` — leave at 8 (gold cardinality ≈ 1).
- **`seed_and_spread`'s `seeds[:8]` hard cap** (`kg/agent.py:408`) hits on 100% of calls
  while the 80-node budget never binds — raising it toward `seed_k=10` is a *likelier*
  recall lever than the budget. Gate on a full-run delta.
- All speculative until a full run exists; **don't let this slice's stop-reason pattern
  bias the grid** (answered=gold-present is fully confounded with gold presence here).

### D3. Do **not** raise `agent_node_budget` `[LOW]`
Never binds — per-query `n_touched` max 47, mean 34 vs budget 80. No-op change.

---

## E. Tooling / artifact size (offline only — no compute/$ impact)

### E1. `run.json` is 72% graph payload — gzip it & stop inlining `[the real win]`
- **What:** `_full_graph` serializes every node/edge with up to 500-char snippets →
  1.19MB of the 1.65MB `run.json`; `dashboard.html` embeds the *same* JSON inline →
  ~1.65MB duplicated.
- **Verified impact:** snippet/edge trimming is only ~−2 to −17% (the over-claimed
  −40–55% is wrong — 28% of the file is non-graph). The defensible win is **gzip the
  run.json (~9× → ~183KB)** and have the static export *fetch* run.json instead of
  inlining it (removes the ~1.65MB duplication).
- Evidence: `kg/testrun.py:118-195` (`_full_graph`), `:179` (500-char snippet),
  `kg/dashboard.py:27-28` (inline replace), `:488-492` (written twice).

### E2. A `--lite` mode for many-query cost runs `[LOW]`
Per-query subgraph replays are ~184KB/12-q (~11% of file). A `--lite` switch that
omits/down-samples them (keep answer+citations+metrics) helps at 68+ queries. Zero effect
on cost/accuracy numbers. Keep full subgraphs as the interactive default.

### E3. Flip representative-run defaults: judge + communities off `[LOW, ergonomic]`
- `--no-judge` / `--no-communities` **already exist** (`kg/cli.py:351-354`). For a cheap
  representative run, default both off and document it. Judge at full scale is only
  ~$0.11 (68 × ~$0.0016); communities are local ($0). The value is ergonomic, not cost.
- Keep judge on when grading response accuracy; keep communities on when evaluating
  global/theme queries.

---

## Do **not** do (verified rejects / over-claims corrected)

- ❌ Lower `max_tokens` to save output tokens — truncates the graph payload (A3).
- ❌ Add `cache_control` to today's ~1.8k prefix expecting savings — silently no-ops; the
  pair-only variant is a *net loss* (A2).
- ❌ Expect 5×/4× from concurrency — production is already ~3–5×; further bumps are
  OTPM-capped (C1/C2).
- ❌ Raise `agent_node_budget` — never binds (D3).
- ❌ Tune `ppr_damping` / `mmr_lambda` / `seed_k` against `rev5-demo` — slice artifact (D1).
- ❌ Treat derive/canon/embedding micro-ops as performance wins — all sub-1% (C4/C5).
- ❌ Lower `agent_max_steps` below 7 — force-submits the one correct grounded answer (B2).

Also explicitly **dropped** by the verification pass as non-issues at this scale: long-doc
sectioning prefix cost (no long docs here), the SHARED_ENTITY pairwise loop & object-kNN
re-search (sub-second batch), `_reindex` full-scan on open, the "search-heavy/read-light"
loop (a slice artifact — gold docs absent so the agent re-searches), and SentenceTransformer
reloading per subprocess (load is amortized, not per-call).

---

## Suggested sequence

1. **A/B reflexion off** on a full-corpus eval (A1) — biggest single $ + speed lever; the
   only blocker is measuring the recall delta.
2. **Agent prefix caching** (B1) — clean, recurring, zero-quality-risk; ship regardless.
3. **Full-corpus run** (all 1243 docs) — unlocks the *real* recall numbers and makes the
   D2 retrieval ablation and any reflexion/caching A/B meaningful.
4. **Concurrency + retry/backoff** (C2) and **gzip/de-inline artifacts** (E1) as
   independent quality-of-life wins.
5. Revisit ingest prompt caching (A2) **only** as a deliberate grow-to-4096 + warm-reuse
   change, gated on an eval check — otherwise skip it.
