# Configuration reference

Every tunable lives in one flat dataclass: [`kg/config.py`](../kg/config.py). Fields are
ordered in three tiers:

- **Tier 1 — primary.** The knobs most users tweak: which models run and the basic shape
  of ingestion/retrieval.
- **Tier 2 — behavior toggles.** Feature switches whose defaults are the banked A/B
  winners. Safe to flip; each is designed so the "off" value reproduces the pre-feature
  behavior byte-for-byte.
- **Tier 3 — experimental / internals.** Thresholds, caps, and weights tuned against the
  eval sets. Changing these silently shifts eval results; don't touch unless you know why.

Override any field from the CLI with `--set key=value`; common ones have dedicated flags
(`--model`, `--chunking`, `--l3`, `--self`, …). This document holds the *why* behind the
defaults — the rationale, tiering, and A/B provenance that used to live inline in the
config comments. Architecture context is in [ARCHITECTURE.md](ARCHITECTURE.md) (section
references below, e.g. §3, point there). Defaults are deliberately conservative /
link-biased (under-merge) per §9 risk 1.

---

## Models (§0)

- `llm_model` — the extraction LLM (and default for the L3 adjudicator via `--model`).
  Default `None` = the active provider's default (openai → `gpt-4o-mini`; codex/claude →
  the CLI's own model). An explicit value always wins, even one equal to a provider
  default — resolution is in `kg/llm_client.py:resolve_model`.
- `rag_model` — the answerer used by `kg ask`. Default `None` = the answerer's
  per-provider default (openai → `gpt-5-mini`, `RAG_OPENAI_DEFAULT`); an explicit value
  always wins.
- `embed_model` / `embed_dim` — sentence-transformers embedder. bge-small and MiniLM are
  both 384-dim; the hashing fallback matches, so `embed_dim` only changes if the model does.
- `judge_model` — LLM-as-judge grading for `testrun`/`ablate`/`guard_eval` scoring only.
  Never used for extraction/canon/RAG, so bumping it doesn't touch prod cost.

## Extractor backends

`extractor_backend` selects the ingestion extraction strategy:

- `"cue_gated"` (default, the production strategy) — a free local NLP floor
  (`local_backend`, default `gliner_yake_cooccur`) runs on every entry, plus ONE
  `llm_model` call ONLY on entries carrying a termination / relative-date / identity cue
  (`kg/cues.py`). `cue_escalate` controls the LLM half (needs an API key).
- `"llm"` — full `llm_model` extraction on everything.
- Pure LLM-free / hybrid NLP backends: `gliner_yake_cooccur`, `gliner2`, `keyword_only`,
  … (see `kg/nlp_extractors.py`).

GLiNER knobs: `gliner_model` / `gliner_threshold` for the entity tagger; GLiNER2
(fastino) is ONE local encoder that emits typed entities AND typed relations in batched
forward passes per section (`Gliner2Extractor`) — $0, on-device. Higher
`gliner2_entity_threshold` trims generic-concept over-extraction; higher
`gliner2_relation_threshold` = cleaner, fewer spurious edges.

Back-compat: `extractor` / `embedder` are LIVE-ONLY leftovers — the offline
heuristic/hashing backends were removed, so the only meaningful values are
`extractor="llm"`, `embedder="st"`. Kept as fields so old invocations don't break.

## Ingestion (§6)

- `semaphore_limit` — bounded LLM concurrency (graphiti SEMAPHORE_LIMIT).
- `reflexion` — one extra recall pass after extraction.
- `long_doc_chars` — above this, extract section-by-section. `lead_chars` — embed the
  lead section for very long docs.
- `extract_max_chars` — per-call input cap inside `Extractor.extract_text`. MUST be
  >= `long_doc_chars` so a window-sized section is never silently truncated (a section is
  exactly `long_doc_chars` wide); the CLI enforces this when `--long-doc-chars` is raised.
  Swept together with `long_doc_chars` in the lever-2 window A/B.
- `extract_max_tokens` — output ceiling (max_tokens) on the `emit_graph` tool call. The
  model emits the WHOLE entity/tag/relation payload under this cap; a section whose graph
  exceeds it is silently truncated mid-JSON (tail dropped — content already generated and
  paid for). Raised 1500→4000 after the max_tokens A/B (2026-06-25, lever 2): the old
  1500 truncated ~8/246 calls on the shipped 6000 window; 4000 clears all truncation and
  is the knee (richness plateaus — 8000 adds nothing). Billed only on emitted tokens,
  so ~+0.2% cost.
- `ingest_flush_every` — the ingest loop checkpoints the store to SQLite every N episodes
  (and always at the end), so a crash mid-ingest loses at most one window. 0 disables
  mid-loop flushes.

## Tag drift control — L1/L2/L3 canonicalization (§3)

Open-vocabulary tags/relations are merged in layers (`kg/canonicalize.py`):

- **L1** — deterministic string normalization; merge on exact match.
- **L2** — embedding cosine: > `syn_merge_threshold` (0.93) is a candidate hard merge;
  > `syn_link_threshold` (0.85) only gets a SIMILAR_TO link (don't merge). The entropy
  guard (`entropy_min_chars`, `entropy_min_bits`) restricts short / low-entropy tags to
  exact-match merging only.
- **L3** — selective LLM tie-breaker for the gray zone (cosine between `rel_gray_floor`
  = 0.90 and the merge threshold, where embeddings can't decide): one small `l3_model`
  call adjudicates "same or not". A deterministic guard list of known-confusable pairs is
  never allowed to merge regardless of the LLM's verdict. **Shipped disabled**
  (`l3_enabled=False`) — L1/L2 were good enough and this adds per-pair API cost — but
  fully wired: `kg ingest --l3` / `kg testrun --l3` enable it, and every verdict is
  logged (`Canonicalizer.l3_log`) for the eval gate.

Relationship-tag consolidation (§3b): `rel_syn_merge_threshold` (0.95) for
relation-label merges; `max_relation_labels` caps labels carried by one connection.

## Derived edges (§2 / §6.5)

- `episode_knn_k` / `episode_knn_floor` — SIMILAR_TO neighbours per episode and the min
  cosine for an episode↔episode edge.
- `shared_min_overlap` — min shared tags/entities for a SHARED_* edge.
- `shared_hub_cap` — SHARED_TAG / SHARED_ENTITY derivation is incremental (only pairs
  involving a NEW episode are computed), but a popular hub still pairs each new episode
  against every member; this bounds that at the `cap` most recent members. A hub big
  enough to hit the cap has high df and near-zero IDF weight anyway.
- `shared_edges=False` skips SHARED_* entirely — the removal A/B: the tag/entity star
  already connects the same episodes at path length 2, so PPR may not need shortcuts.

## Episode chunking (`kg/chunkers.py`)

`chunking` splits each text entry along its natural structure into several small
episodes, keeps the full original on an un-rankable SOURCE parent, and links chunk→parent
(PART_OF) + chunk→next-sibling (NEXT). Retrieval then ranks statement-grained chunks
instead of whole multi-thousand-char blobs.

- `"turns"` (default) — chat turns, else paragraphs; oversized units fall back to
  sentence packing.
- `"markdown"` — heading sections + breadcrumb prefixes; `"prose"` — paragraph→sentence
  packing; `"code"` — top-level blank-line blocks.
- `"auto"` — sniffs the format PER ENTRY (`sniff_format`) and routes to the right
  chunker; chat text still gets exactly the `"turns"` behavior.
- `"none"` — legacy one-episode-per-entry (byte-for-byte unchanged).

Knobs: `chunk_target_chars` greedy-packs natural units up to ~this per chunk;
`chunk_max_chars` is both the "stays unchunked" threshold and the hard ceiling one packed
unit may reach. `part_of_weight` keeps the parent from becoming a sibling super-hub in
traversal; `next_weight` weights the sequence edge.

## Retrieval — PPR (§5)

HippoRAG-style seed-and-spread: seeds from fused embedding+BM25 (`seed_k`), Personalized
PageRank diffusion (`ppr_damping`), `top_k` episodes returned.

- `ppr_backend="global"` (default) — exact power iteration over the whole projection,
  O(N+E) per query; byte-identical to the original implementation. `"push"` — local
  forward push (Andersen–Chung–Lang), explores only the seed neighborhood so work is
  independent of graph size; eps-approximate scores (`ppr_push_eps`, per unit of weighted
  degree), same fixed point and same ranking at retrieval resolution.
- `mmr_lambda` — MMR relevance↔diversity tradeoff (1.0 = pure relevance).
- `inferred_confidence_floor` — drop INFERRED edges below this in traversal.

## Answer-path pipeline (the production read strategy; used by `kg ask`)

On top of the PPR pool the answer path adds a 4-lane query router (`kg/route.py`:
recency | state | multihop | single), a fact-bearing-episode augment on state/evolution
lanes (`fact_lane_augment`), and a cross-encoder rerank of the candidate pool — but ONLY
on the hard lanes (`rerank_lanes`), since the cross-encoder can demote the gold on easy
single-fact lookups. All default ON.

- `rerank_keep_ppr_top` — PPR guarantee under the cross-encoder: the top-N episodes of
  the RAW PPR pool are always kept in the final top-k even if the reranker pushes them
  below k (the CE can demote a gold the graph ranked #1 completely out of the context).
  0 disables.
- `seed_reserve` — reserved seats: up to N final top-k slots go to the highest
  raw-seed-score episodes (embedding+BM25, the Seeder's own ranking) whose SESSION won no
  slot from the PPR→MMR→CE pipeline. The containment eval showed 13/14 missing gold
  sessions were already in seeds and were ranked out downstream. Displaces only the tail
  of the final ranking; 0 (default) disables.
- `date_window_boost` / `date_window_slots` — resolve a relative-date phrase in the query
  ("last Saturday", "two months ago") against as_of, then reserve up to N slots for
  episodes inside the window.
- `rag_session_dedup` — session diversity in the context-eligible prefix: reorder the
  final ranking so the first `rag_context_episodes` slots hold chunks from DISTINCT
  sessions (first chunk of each new session in rank order; duplicates follow). With
  `mmr_lambda=1.0` there is no diversity pressure at all, so 2 of 5 reader slots
  routinely go to a second chunk of a session already present while a gold session sits
  at rank 6. Off by default.

## RAG answer flow (§5)

The query path is retrieve-then-read: PPR (or as-of-T PPR) assembles a context blob of
the top episodes + the currently-valid facts among the touched entities, then a SINGLE
LLM call answers over it with citations. No per-hop LLM walking. LIVE-ONLY: the offline
answerer was removed, so `kg ask` requires a key (or an injected client). `rag_backend`
currently has one live implementation ("openai"; "auto" resolves to it).

Context assembly knobs:

- `rag_context_episodes` — episodes whose text enters the context blob. Set == `top_k`'s
  effective reader budget so nothing recall@k finds gets truncated before the reader sees
  it (was 6; see the baseline notes).
- `rag_episode_chars`, `rag_max_facts`, `rag_max_tokens` — per-episode text budget,
  currently-valid facts surfaced, and the answer output cap.
- `rag_chunks_per_source` — with chunking on, ranks 1..n can all be chunks of ONE source;
  cap how many context slots a single source may occupy so the reader still sees other
  sessions. 0 = off.
- `rag_parent_expand` — sibling-chunk expansion (query-side only, context-only): after
  top-n selection, pull in each selected chunk's #cNNN neighbours within this radius so a
  chunked session's answer-bearing sibling isn't left out of context even when it didn't
  rank into the top-n itself. 0 = off, byte-identical to pre-expansion behavior.
- `rag_expand_budget_chars` — hard cap on total episode text after expansion; once hit,
  stop adding siblings. All originally selected chunks are always kept — only expansion
  siblings are capped.
- `rag_retarget` — chunk-level retargeting (query-side only): the right SOURCE can win a
  seat in selection while the wrong CHUNK of it gets picked (PPR ranks by diffused chunk
  score, not by which chunk actually answers the question). Modes:
  - `"off"` — byte-identical to pre-retargeting behavior.
  - `"seed"` — within each source that won seats, refill its slots with that source's
    best chunks by raw embedding seed rank (`RetrievalResult.seed_scores`) instead of PPR
    chunk order. Same slot count per source, swaps only.
  - `"seed+lex"` — seed retarget, then a lexical-overlap swap pass: a same-source chunk
    NOT selected may swap in for a selected one if it strictly beats it on question
    content-word / digit-token overlap. The best-ranked (incumbent) chunk of each source
    is never swapped out.
  - `"ce"` (default; banked from the reader4o/reader5 A/Bs — see `runs/reader5-honest-1`)
    — within each source that won seats, rank its chunks by local cross-encoder
    question↔chunk relevance (`rerank_model`, $0) instead of seed/PPR order; falls back
    to seed order if the model can't load.
  - `"ce+seed"` — blend: a source's slots split across the CE ranking and the seed
    ranking (best-rank-of-either), since the two signals miss on disjoint questions
    (CE: relevance ≠ answer-bearing; seed: synonymy).
- `rag_provenance_promote` — a FACT's episode_id (the chunk it was extracted from) is
  pulled into context if the fact's src/dst entity names overlap the question terms and
  that chunk isn't already present — displacing only the lowest-ranked expansion sibling
  (never an originally selected chunk). Runs after sibling expansion.
- `rag_answer_events` / `rag_answer_events_lanes` — structured enumeration in the answer
  tool (reader-facing): on the configured lanes the `submit_answer` schema REQUIRES an
  `events` array (date + description + quantity) filled BEFORE the answer, turning "count
  in your head" into "fill the list, then count" — the reader's dominant failure on
  aggregation questions is stating a total without enumerating the events behind it. Same
  single LLM call, a few dozen extra output tokens. `"off"` = plain schema; `"lanes"`
  (default, banked) applies it on `rag_answer_events_lanes` (aggregation +
  temporal/state); `"all"` applies it everywhere.
- `rag_resolve_reldates` — in-text relative-date resolution (render-time only): annotate
  relative phrases in episode text with the absolute date they resolve to against the
  EPISODE's own date ("last week [≈ 2023-03-20] I attended …"), because the event's date
  is not the episode's date and the header delta can't express that. Off (default) =
  context byte-identical to today.

## Extraction-completeness audit

`completeness_tier2` / `completeness_tier2_model` — testrun/dashboard only (see
`kg/completeness.py`); never used for extraction/canon/RAG. Tier 1 (regex occurrence
audit) is always on; tier 2 adds an LLM occurrence audit.

## Personal-web mode & self-guard

`self_entity` — when on, first-person references ("i"/"me"/"my"/…) resolve to ONE stable
canonical "self" anchor (`self_name`) so the narrator's relationships form on a single
node. OFF by default — the default path is byte-for-byte unchanged when off.

`self_guard` — PPR hub guard, only bites when `self_entity` is on. In personal-web mode
every first-person reference resolves to ONE node, so on a deep single-user graph that
node becomes incident to a large fraction of RELATED_TO fact edges — a high-degree hub.
IDF down-weights it as a SEED but does NOT stop PageRank routing activation THROUGH it
during the diffusion, over-spreading mass to weakly-related episodes. Modes:

- `"none"` (default, safe) — no guard; self diffuses like any other entity.
- `"exclude"` — drop self + its RESOLVES_TO star from the projection entirely (like
  IN_COMMUNITY); self carries no discriminating signal.
- `"cap"` — keep self but cap its incident edge weights at `self_guard_cap` per edge.
- `"seed"` — keep self in the graph but never SEED it (no personalization mass).

## Misc

- `community_seed` — pin for reproducible community ids (§ phase 3).
- `directed` — graph direction (§2).
- `random_seed` — global seed for everything else.
- `extra` — free-form dict for experiment-specific values that don't merit a field.
