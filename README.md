# you

An **episodic, bi-temporal knowledge graph** over multimodal content (text, links, photos, files),
in the HippoRAG / HippoRAG 2 / Graphiti lineage.

Each ingested entry becomes an immutable **Episode**; an LLM extracts the entities it mentions and
the **directed relationships** between them. Entity occurrences become immutable **Mentions** that
point (in a *star*, not a clique) at a lean canonical **Entity** anchor — so identity is stable and
embeddings live only on the immutable layer (episodes + mentions), never re-embedded on an update.
Facts live on **edges with bi-temporal validity** (`valid_at` / `invalid_at` + belief): a state
change *closes* the old fact's window and *opens* a new one ("Becky lives in Toronto" → "…Berlin" is
an `invalid_at` plus a new edge, not an overwrite). Retrieval is *graph-first* — a non-LLM diffusion
(Personalized PageRank over the temporally-filtered, symmetrized projection) does the multi-hop work;
embeddings are an **entry-point index**, not the main path.

## Status

**Built** — the `kg/` package implements the design end-to-end (episodic ingestion → typed,
bi-temporal directed graph → routed hybrid retrieval → cross-encoder rerank → graph-RAG answers →
eval). The default **production pipeline**: cue-gated extraction (a free local `gliner_yake_cooccur`
NLP floor on every entry + a gpt-4o-mini call only on entries carrying a termination/date/identity
cue), a 4-lane query router, a fact-bearing-episode augment on state/evolution questions, and a
cross-encoder reranker on the hard lanes. Embeddings use local `bge-small`. Extraction runs keyless
on the local floor; the escalation and the single answer call need `OPENAI_API_KEY`.

```bash
pip install -r requirements.txt
python -m kg demo                       # ingest a synthetic evolving stream; show current vs as-of
python -m kg ingest --reset && python -m kg communities
python -m kg query "what are the main themes across the collection"      # algorithmic retrieval
python -m kg ask   "where does Becky live and who does she work with?"   # PPR → context → 1 answer
python -m kg ask   "where did Becky live?" --as-of 2022                  # point-in-time retrieval
python -m kg forget "my address is 42 Elm Street" --dry-run              # erasure: preview only
python -m kg forget "my address is 42 Elm Street"                        # erasure: execute + save
python -m kg eval        # recall@k ablation: PPR vs BFS vs flat vector (+ rag)
python -m kg serve       # browser viewer: watch the graph build + trace queries
python -m pytest -q
```

Two query surfaces, and **for neither does the LLM traverse the graph**: **`query`** runs the
algorithmic retrievers directly (PPR / BFS / vector / community) and returns ranked episodes;
**`ask`** is *retrieve-then-read* — the hybrid retriever routes the question, augments
state/evolution lanes with fact-bearing episodes, reranks the hard lanes with a cross-encoder, and
a **single** LLM call answers over the assembled context (top episodes + currently-valid, or as-of-T,
facts; plus the full closed+open history on evolution questions) with citations. The answer path is
live-only (needs `OPENAI_API_KEY`); a deterministic extractive synthesis survives only as an
internal crash-guard. Pass `--as-of <date>` to either surface to read the world as it was then.
To A/B the full pipeline vs the raw PPR-RAG engine: `python -m kg.ablate --tier sample --k 3 --ctx 3`.

**Forgetting** is a first-class surface next to ingesting and querying: `g.forget("…")` /
`python -m kg forget "…"` **erases** information, as distinct from superseding it (a fact whose
window closed is history and stays queryable as-of-T; an erased fact is gone from every view).
The erase is query-and-trace-back: an **exhaustive** sweep of every chunk (dense cosine + lexical —
never top-k, deletion needs recall, and the fixpoint loop re-sweeps until nothing is found), a
confirmation gate per candidate, then **sentence-level redaction in place** — the matched sentences
are removed (marker: `[redacted]`), the rest of the turn survives, the chunk is re-embedded locally
and keeps its id, and the facts/mentions/tags derived from the removed text are retracted with the
usual orphan cascade (an entity mentioned elsewhere keeps its other edges; one supported only by
erased text goes with it). Text is only ever *removed*, never LLM-rewritten. With `OPENAI_API_KEY`
set, three LLM escalations sharpen the result (~$0.01–0.05/request, `--no-escalate` to disable):
a paraphrase judge for fuzzy hits, a single-chunk re-extract diff for artifact attribution, and a
final **inference audit** — the model is asked to reconstruct the secret from what retrieval still
returns, and a successful guess escalates the contributing chunks to whole-chunk tombstones.
`--dry-run` previews the full action list without mutating. Two honest limits: erasure covers what
the *store* can reach — **ingest caches (`store/cache/`) and raw session logs must be purged
separately** — and redaction leaves a `[redacted]` marker, so the *existence* of a secret is not
hidden, only its content.

A plain-HTML viewer (no build step, no CDN — vanilla JS + SVG) shows the episode graph, animates it
**being built** in ingestion order, and **traces the path a query takes** (seeds → tag hubs → ranked
results, BFS hops animated). `python -m kg serve` for live typed queries, or `python -m kg viz
--query "…" --out kg_viz.html` for a self-contained file.

## Decided stack

- **Pipeline LLM:** gpt-4o-mini, vision-capable for images.
- **Embeddings:** local `sentence-transformers` (`BAAI/bge-small-en-v1.5`), fully offline.
- **Graph:** NetworkX `MultiDiGraph` (directed, in-memory, persisted) — PageRank/BFS/Leiden
  out of the box; traversal runs over a symmetrized projection so direction never costs recall.
- **Vectors:** NumPy brute-force cosine to start; SQLite for metadata + SHA256 cache.
- **Python:** 3.14 native (torch + sentence-transformers both ship 3.14 wheels).

See [docs/ARCHITECTURE.md §0](docs/ARCHITECTURE.md) for the full decision table.

## Getting started from scratch

```bash
pip install -r requirements.txt
cp .env.example .env               # then paste in your key: OPENAI_API_KEY=sk-...
```

`OPENAI_API_KEY` is optional. Without it, ingestion still runs (the default `cue_gated`
extractor backend has a keyless local NLP floor), but escalation to gpt-4o-mini, the live
`haiku`/`auto` extractor backends, and every `ask`/answer call are live-only and will raise
without a key. `kg/__init__.py` auto-loads `.env` on import, so you never need to `export` it
yourself; the file is gitignored.

The toy `dataset/longmemeval/sample` tier ships committed so `python -m kg ingest` works with
zero setup. Larger tiers must be built first:

```bash
python scripts/build_longmemeval.py                 # sample + small + med (downloads ~277 MB)
python scripts/build_longmemeval.py --tier large     # heavier (~2.74 GB)
python scripts/build_longmemeval.py --tier all       # everything
```

Every `kg` subcommand takes a top-level `--store <path>` (default `store/kg.db`) pointing at the
SQLite graph file to read/write.

### Unit tests

```bash
python -m pytest -q             # offline-safe: no API key needed, ~150 tests
```

### Running a benchmark (`kg testrun`)

This is the harness for measuring cost/quality/latency changes on the LongMemEval dataset — the
tool to reach for when you change extraction, chunking, canonicalization, retrieval, or reranking
and want to know what it actually did to accuracy and $ cost.

```bash
python -m kg testrun --tier micro                        # quick live smoke test (3 instances, ~$0.02)
python -m kg testrun --tier small --label my_change       # a real A/B data point
python -m kg dashboard --out runs                         # browse every run at localhost:8050
```

Each run writes `runs/<run_id>/run.json` + a static `dashboard.html`, and registers itself in
`runs/index.json`. `--label baseline` / `--label my_change` on two runs, then diff their
`run.json`, is the standard A/B workflow (see `optimization.md`'s "A/B harness" section for what
to compare on).

**All `testrun` flags:**

| flag | default | meaning |
|---|---|---|
| `--mode` | `per-instance` | `per-instance` = the dataset's native protocol, a fresh graph per question (no cross-persona pooling — the only mode whose accuracy is a real LongMemEval score). `shared` pools the whole tier into one graph (scale/structure smoke view + ingest cost/token metering only). |
| `--tier` | `micro` | `micro` (3 instances, committed, for fast live smoke tests) / `sample` / `small` / `med` / `large` — see `dataset/longmemeval/README.md`. |
| `--queries N` | all | cap the number of eval questions (= instances, in per-instance mode). |
| `--limit N` | all | **shared mode only**: cap session episodes ingested (use `--queries` for per-instance). |
| `--k N` | `8` | episodes retrieved per query (recall@k). |
| `--backend auto` | live if key set | answerer backend for the query half. |
| `--model ID` | — | override the LLM model id (extractor + L3 + answerer all at once). |
| `--extractor {auto,haiku}` | `auto` | back-compat display switch; see `--extractor-backend` for the real knob. |
| `--extractor-backend NAME` | `cue_gated` (config default) | extraction strategy: `cue_gated` (free local NLP floor + gpt-4o-mini only on cue-bearing entries), `haiku`/`auto` (full LLM on everything, paid), or an LLM-free NLP backend (`gliner2`, `gliner2_nounchunk`, `gliner_yake_cooccur`, …) — $0 ingest, runs locally. See `kg/nlp_extractors.py` `NLP_BACKENDS`. |
| `--embedder {auto,st}` | `auto` | embedding backend (local `sentence-transformers`). |
| `--long-doc-chars W` | `6000` | section size above which a doc is split for extraction; raises the per-call input cap to match. |
| `--extract-max-chars C` | `12000` | per-call input cap inside `extract_text`; normally tracks `--long-doc-chars`. |
| `--extract-max-tokens M` | `4000` | output cap on the extraction call; a section whose graph exceeds it is silently truncated (`run.json`'s `truncated` count finds the right value). |
| `--chunking {none,turns,markdown,prose,code,auto}` | `none` | natural-boundary chunking: split one big entry into several small, retrieval-grained episodes (`turns` = chat turns/paragraphs + a SOURCE parent + `PART_OF`/`NEXT` edges; `auto` sniffs format per entry). |
| `--l3` | off | enable the L3 LLM canonicalization tie-breaker. |
| `--no-judge` | judge on | skip the LLM response-accuracy judge (keep the deterministic proxy score only — cheaper, no judge-model cost). |
| `--no-completeness` | audit on | skip the tier-2 LLM occurrence-completeness audit (per-instance mode only; the $0 tier-1 regex capture-rate always runs). |
| `--no-communities` | communities on | **shared mode only**: skip community detection after ingest (faster; per-instance mode never builds communities). |
| `--no-ingest-cache` | cache on | **per-instance mode only**: bypass the ingest-store cache (see below) and always re-run extraction. |
| `--label NAME` | timestamp | run id / label shown in the dashboard index. |
| `--out DIR` | `runs` | directory dashboard runs are written to. |

**The ingest-store cache** (per-instance mode, default ON): extraction is ~93% of a run's cost, so
re-paying it when a change is query-side-only (retrieval/rerank/context/reader/judge) is pure
waste. Each instance's ingested store is cached at `store/cache/<instance_id>-<key12>.db`, keyed
off a hash of the instance's session content + every ingest-relevant config field (model,
chunking, canonicalization thresholds, extractor prompt text — see `INGEST_RELEVANT_FIELDS` in
`kg/ingest_cache.py`) — query-side fields never bust it. `run.json` reports
`ingest.totals.cached_instances`/`fresh_instances` and a per-instance `ingest_cached` flag so a
cached run is never misread as "ingest got cheaper" instead of "didn't run."

- Force a fresh run: `--no-ingest-cache`.
- Clear it: no CLI (no auto-eviction, by design) — `rm -rf store/cache` (everything) or
  `rm store/cache/<instance_id>-*.db` (one instance).
- There's no "pick a cache" flag: whichever entry matches your *current* config is used
  automatically; an ingest-relevant config change just misses and re-ingests under a new key.

### Other commands

| command | what it does |
|---|---|
| `ingest [--tier T] [--question-id ID] [--synthetic] [--limit N] [--reset]` | build/extend the graph from a LongMemEval tier, or `--synthetic` for the deterministic Becky/Alex demo stream. |
| `communities` | detect communities + summaries (global/breadth queries). |
| `query TEXT [--mode {auto,ppr,bfs,vector,community}] [--k N] [--as-of DATE]` | algorithmic retrieval only — the LLM never traverses. |
| `ask TEXT [--k N] [--as-of DATE] [--show-context]` | PPR retrieves a context, one LLM call answers, with citations. |
| `demo [--personal]` | ingest the synthetic evolving stream; prints current-view vs as-of answers. |
| `stats` | node/edge counts. |
| `inspect NODE_ID` | dump one node + its neighbours (fact validity windows for `RELATED_TO`). |
| `viz [--out FILE] [--query TEXT] [--mode {bfs,ppr,vector}]` | write a self-contained HTML graph viewer. |
| `serve [--port N]` | live browser viewer: watch the graph build + trace queries. |
| `eval [--k N] [--modes ppr,bfs,vector] [--single N] [--cross N] [--questions FILE]` | recall@k / MRR ablation across retrieval modes. |
| `extract-dump [--tier T] [--limit N] [--out FILE]` | dump per-item extractions for one extractor/model, no graph build (for inspecting extraction quality directly). |
| `eval-canon [--l3]` | canonicalization gate: synonyms must merge, antonyms/inverses must not. |

Run `python -m kg <command> --help` for any command's exact flags (the table above is the
practical subset — `--help` is always the source of truth).

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the design: graph model, tag-drift control,
  embeddings, traversal/retrieval, ingestion pipeline, storage choice, phased build plan.
- [docs/LITERATURE.md](docs/LITERATURE.md) — per-source notes (GraphRAG, HippoRAG, RAPTOR,
  A-MEM, Graphify, TnT-LLM, Chain-of-Layer, TaxoGen, TaxoCom, Leiden).
- [optimization.md](optimization.md) — the A/B benchmark harness: what's been tried, what moved
  the needle, and the full ingest-cache rationale.

## Core design bets (from the literature)

0. **Episodic / semantic split + bi-temporal facts (HippoRAG / Graphiti).** Immutable
   Episodes and Mentions (append-only, embedded once) feed lean canonical Entity anchors in a
   *star* topology; facts are *time-bounded edges* that supersede rather than overwrite. This
   designs the re-embedding problem out (cost is proportional to new data) and makes
   "coworker → ex-coworker" / "Toronto → Berlin" resolve correctly regardless of document order.
   See [docs/TEMPORAL.md](docs/TEMPORAL.md).
1. **Diffusion is the primary retrieval path; embeddings are a seed index; the LLM does not
   traverse.** Embed to *find* entry nodes, then let graph structure (Personalized PageRank over the
   temporally-filtered projection) do the multi-hop work in one step — no per-hop LLM call. `ask`
   feeds that retrieved context to a single answering LLM call (retrieve-then-read).
2. **Tags *and* relationships are first-class, open-vocabulary, and consolidated.** Topical
   tags are first-class nodes; relationship labels (`is_friend_of`, `works_with`, …) are
   LLM-generated per connection as **parallel directed edges — one per relation** (the
   KG-triple / property-graph shape, so each carries its own provenance/confidence), then
   consolidated into canonical `RelationTagNode`s by the same drift-control pipeline as tags.
   This is *open relation extraction + relation canonicalization* (Galárraga et al., CIKM 2014;
   CESI, WWW 2018) — the consolidation step is what stops free-form predicates from collapsing
   into a vague `related_to`, so we get expressivity without the drift the fixed-enum was
   guarding against.
3. **Drift control is layered**, not a single hash: exact/normalized hash → embedding synonymy
   *link* (not merge) → periodic taxonomy reconciliation. Bias toward *linking* near-duplicates,
   not hard-merging them. Relationships consolidate on a **content key** (drop function words +
   singularize): `is_friend_of` ≈ `is_friends_with` merge, but `is_enemy_of` and the passive
   inverse `managed_by` stay distinct — because embedding cosine alone can't tell synonyms from
   antonyms.
4. **Embed summaries (primary) + tag/entity strings (for synonymy linking)** — bare-tag
   embeddings alone are too lossy as the main retrieval vector.
5. **Edges carry provenance + confidence** (`EXTRACTED` / `INFERRED` / `SIMILAR` / `DERIVED`) so
   retrieval can down-weight or drop low-confidence relationships, and **fact edges carry a
   bi-temporal window + belief state** so closed/superseded/retracted facts drop out of the
   current-view (or as-of-T) traversal automatically.

## Test corpus

**LongMemEval** (Wu et al., ICLR'25; Hugging Face `xiaowu0162/longmemeval-cleaned`, MIT) — a
long-term-memory benchmark of dated, multi-session chat histories. Each instance is one user's
haystack of timestamped sessions + a question + answer + the evidence sessions; every session
becomes a dated episode, so it directly exercises the temporal / knowledge-update machinery.
Built into tiers (`sample`/`small`/`med`/`large`) by `scripts/build_longmemeval.py`. See
[dataset/longmemeval/README.md](dataset/longmemeval/README.md). *(Replaced the earlier frozen
Wikipedia+COCO corpus — see the History note in [dataset/README.md](dataset/README.md).)*
