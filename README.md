# you

A **temporal knowledge-graph memory engine**. Ingest a stream of dated content (chat
sessions, notes, documents, images) and query it as it is now — or as it was at any point
in the past.

- **Bi-temporal facts.** A state change ("Becky lives in Toronto" → "…Berlin") closes the
  old fact's validity window and opens a new one, instead of overwriting. History stays
  queryable with `--as-of <date>`.
- **Graph-first retrieval.** Multi-hop reasoning is done by graph diffusion (Personalized
  PageRank over the temporally-filtered graph), not by an LLM walking the graph.
  Embeddings serve as the entry-point index.
- **Single-call answers.** `ask` is retrieve-then-read: the retriever assembles a context
  of top episodes plus currently-valid facts, and one LLM call answers with citations.
- **First-class forgetting.** `forget` erases information from every view — distinct from
  superseding it — with sentence-level redaction and an exhaustive, recall-oriented sweep.
- **Low-cost by default.** Extraction runs on a free local NLP floor with LLM escalation
  only on entries that need it; embeddings are local. An `OPENAI_API_KEY` is required
  only for LLM escalation and the answer call.

## Results

On **LongMemEval-S** (100-question type-stratified subset, the dataset's native
one-graph-per-question protocol), judged by gpt-4o. Extraction used the default cue-gated
pipeline with **gpt-4o-mini** in both runs; only the answerer model differs:

| Question type | gpt-4o-mini answerer | gpt-5-mini answerer |
|---|---|---|
| Single-session (user) | 92.9% (13/14) | 100% (14/14) |
| Single-session (assistant) | 100% (11/11) | 100% (11/11) |
| Knowledge update | 75.0% (12/16) | 87.5% (14/16) |
| Multi-session | 63.0% (17/27) | 81.5% (22/27) |
| Temporal reasoning | 65.4% (17/26) | 80.8% (21/26) |
| Single-session (preference) | 50.0% (3/6) | 66.7% (4/6) |
| **Overall** | **73.0% (73/100)** | **86.0% (86/100)** |

Retrieval is identical in both runs: recall@8 93.5%, MRR 0.945.

Retrieval puts the right evidence in front of the answerer almost every time: 98/100
questions have a gold evidence session in the assembled context, and the reranking stack
places it at **rank 1 for 92/100** — so most residual errors are LLM answering failures, not
retrieval failures. On the answering side, the schema-constrained answer call (see
[How it works](#how-it-works)) was one of the largest single accuracy improvements on
multi-session and temporal questions.

One known inefficiency: the constructed context is **fixed-size** (top-5 episodes plus
sibling expansion) even though the gold evidence usually sits at rank 1. Sizing the
context dynamically — fewer episodes when the reranker is confident — is a planned
optimization that would cut input tokens per query without touching accuracy.

### Cost and latency

What it costs to run, measured on the same benchmark (~4,700 chat sessions ingested,
100 questions asked):

| | gpt-4o-mini answerer | gpt-5-mini answerer |
|---|---|---|
| **Ingestion** (one-time, gpt-4o-mini extraction) | $6.35 (~$1.34 / 1k sessions) | same |
| **Cost per query** | ~$0.0013 | ~$0.0048 |
| **Tokens per query** (input / output) | ~8.3k / ~170 | ~8.4k / ~1.4k |
| **Latency per query** (mean / p95) | ~5 s / ~8 s | ~16 s / ~31 s |
| **Latency excluding the LLM API call** | ~3 s | ~3 s |

Each query makes exactly one LLM call; the input tokens are the assembled context
(top episodes + currently-valid facts) and gpt-5-mini's larger output is mostly
reasoning tokens. The last row is the time spent in local processing only — embedding
the query, PPR diffusion, cross-encoder rerank, and context assembly, all on-device —
i.e. what latency would look like with a faster answering model or endpoint. The
remainder of the total is waiting on the OpenAI API response (~3 s for gpt-4o-mini,
~14 s for gpt-5-mini).

Reproduce with `python -m kg testrun --tier small`
(see [Benchmarking](#benchmarking-kg-testrun)).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env               # optional: paste in OPENAI_API_KEY=sk-...

python -m kg demo                       # ingest a synthetic evolving stream; show current vs as-of
python -m kg ingest --reset && python -m kg communities
python -m kg query "what are the main themes across the collection"      # algorithmic retrieval
python -m kg ask   "where does Becky live and who does she work with?"   # graph-RAG answer
python -m kg ask   "where did Becky live?" --as-of 2022                  # point-in-time retrieval
python -m kg forget "my address is 42 Elm Street" --dry-run              # erasure: preview only
python -m kg serve       # browser viewer: watch the graph build + trace queries
python -m pytest -q      # offline-safe: no API key needed
```

`OPENAI_API_KEY` is optional. Without it, ingestion still runs on the keyless local
extraction floor; LLM escalation and every `ask`/answer call require a key.
`kg/__init__.py` auto-loads `.env` on import (the file is gitignored). Every `kg`
subcommand takes a top-level `--store <path>` (default `store/kg.db`) pointing at the
SQLite graph file to read/write.

## How it works

Each ingested entry becomes an immutable **Episode**. The extractor identifies the
entities it mentions and the directed relationships between them; entity occurrences
become immutable **Mentions** that point (in a star, not a clique) at a lean canonical
**Entity** anchor. Identity stays stable and embeddings live only on the immutable layer,
so nothing is ever re-embedded on an update. Facts live on **edges with bi-temporal
validity** (`valid_at` / `invalid_at` + belief state), and closed/superseded/retracted
facts drop out of the current-view (or as-of-T) traversal automatically.

The default pipeline: **cue-gated extraction** (a free local NLP floor on every entry,
with a single LLM call only on entries carrying a termination/date/identity cue), a
**4-lane query router**, a fact-bearing-episode augment on state/evolution questions, and
a **cross-encoder reranker** on the hard lanes.

There are two query surfaces, and for neither does the LLM traverse the graph:

- **`query`** runs the algorithmic retrievers directly (PPR / BFS / vector / community)
  and returns ranked episodes.
- **`ask`** is retrieve-then-read: the hybrid retriever routes the question, augments
  state/evolution lanes with fact-bearing episodes, reranks the hard lanes, and a
  **single** LLM call answers over the assembled context (top episodes + currently-valid,
  or as-of-T, facts; plus the full closed+open history on evolution questions).

The answer call is **schema-constrained**, not free-form: the model must fill a
structured tool call whose schema requires citations, and on aggregation and temporal
lanes a dated `events` array it must enumerate *before* stating the answer — turning
"count in your head" into "fill the list, then count", the dominant failure mode on
multi-session counting questions.

Pass `--as-of <date>` to either surface to read the world as it was then.

### Stack

| Concern | Choice |
|---|---|
| Extraction | Cue-gated: local NLP floor (GLiNER/YAKE) + gpt-4o-mini escalation |
| Embeddings | Local `sentence-transformers` (`BAAI/bge-small-en-v1.5`), fully offline |
| Graph | Directed multigraph persisted to SQLite; retrieval traverses an immutable CSR (NumPy/SciPy) projection |
| Vectors | NumPy cosine; SQLite for metadata + content-hash cache |
| Reranking | Local cross-encoder (`ms-marco-MiniLM-L-6-v2`) |

Every tunable lives in a single tiered config — see
[docs/CONFIG.md](docs/CONFIG.md) for the reference and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Forgetting

`python -m kg forget "…"` (or `g.forget("…")`) **erases** information, as distinct from
superseding it: a fact whose validity window closed is history and stays queryable
as-of-T; an erased fact is gone from every view.

The erase is query-and-trace-back: an **exhaustive** sweep of every chunk (dense cosine +
lexical — never top-k, since deletion needs recall, and the fixpoint loop re-sweeps until
nothing is found), a confirmation gate per candidate, then **sentence-level redaction in
place** — matched sentences are removed (marker: `[redacted]`), the rest of the turn
survives, the chunk is re-embedded locally and keeps its id, and the facts/mentions/tags
derived from the removed text are retracted with an orphan cascade (an entity mentioned
elsewhere keeps its other edges; one supported only by erased text goes with it). Text is
only ever *removed*, never LLM-rewritten.

With `OPENAI_API_KEY` set, three LLM escalations sharpen the result
(~$0.01–0.05/request, `--no-escalate` to disable): a paraphrase judge for fuzzy hits, a
single-chunk re-extract diff for artifact attribution, and a final **inference audit** —
the model is asked to reconstruct the secret from what retrieval still returns, and a
successful guess escalates the contributing chunks to whole-chunk tombstones. `--dry-run`
previews the full action list without mutating.

Two scope limits to be aware of: erasure covers what the *store* can reach — ingest
caches (`store/cache/`) and raw session logs must be purged separately — and redaction
leaves a `[redacted]` marker, so the *existence* of a secret is not hidden, only its
content.

## Graph viewer

A dependency-free HTML viewer (vanilla JS + SVG, no build step, no CDN) shows the episode
graph, animates it being built in ingestion order, and traces the path a query takes
(seeds → tag hubs → ranked results, BFS hops animated). `python -m kg serve` for live
typed queries, or `python -m kg viz --query "…" --out kg_viz.html` for a self-contained
file.
Warning: the viewer may lag your browser considerably if the graph is large.

## Commands

| command | what it does |
|---|---|
| `ingest [--tier T] [--question-id ID] [--synthetic] [--limit N] [--reset]` | build/extend the graph from a LongMemEval tier, or `--synthetic` for the deterministic demo stream. |
| `communities` | detect communities + summaries (global/breadth queries). |
| `query TEXT [--mode {auto,ppr,bfs,vector,community}] [--k N] [--as-of DATE]` | algorithmic retrieval only — the LLM never traverses. |
| `ask TEXT [--k N] [--as-of DATE] [--show-context]` | retrieve a context, one LLM call answers, with citations. |
| `forget TEXT [--dry-run] [--no-escalate]` | erase information from every view (see above). |
| `demo [--personal]` | ingest the synthetic evolving stream; prints current-view vs as-of answers. |
| `stats` | node/edge counts. |
| `inspect NODE_ID` | dump one node + its neighbours (fact validity windows for `RELATED_TO`). |
| `viz [--out FILE] [--query TEXT] [--mode {bfs,ppr,vector}]` | write a self-contained HTML graph viewer. |
| `serve [--port N]` | live browser viewer: watch the graph build + trace queries. |
| `eval [--k N] [--modes ppr,bfs,vector] [--single N] [--cross N] [--questions FILE]` | recall@k / MRR ablation across retrieval modes. |
| `extract-dump [--tier T] [--limit N] [--out FILE]` | dump per-item extractions for one extractor/model, no graph build. |
| `eval-canon [--l3]` | canonicalization gate: synonyms must merge, antonyms/inverses must not. |

Run `python -m kg <command> --help` for any command's exact flags (the tables in this
README are the practical subset — `--help` is always the source of truth).

## Benchmarking (`kg testrun`)

The harness for measuring cost/quality/latency changes on the LongMemEval dataset — the
tool to reach for when you change extraction, chunking, canonicalization, retrieval, or
reranking and want to know what it actually did to accuracy and cost.

```bash
python -m kg testrun --tier micro                        # quick live smoke test (3 instances, ~$0.02)
python -m kg testrun --tier small --label my_change      # a real A/B data point
python -m kg dashboard --out runs                        # browse every run at localhost:8050
```

Each run writes `runs/<run_id>/run.json` + a static `dashboard.html`, and registers
itself in `runs/index.json`. `--label baseline` / `--label my_change` on two runs, then
diffing their `run.json`, is the standard A/B workflow.

To A/B the full answer pipeline against the raw PPR-RAG engine:
`python -m kg.ablate --tier sample --k 3 --ctx 3`.

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
| `--extractor {auto,llm}` | `auto` | back-compat display switch; see `--extractor-backend` for the real knob. |
| `--extractor-backend NAME` | `cue_gated` (config default) | extraction strategy: `cue_gated` (free local NLP floor + LLM only on cue-bearing entries), `llm`/`auto` (full LLM on everything, paid), or an LLM-free NLP backend (`gliner2`, `gliner2_nounchunk`, `gliner_yake_cooccur`, …) — $0 ingest, runs locally. See `kg/nlp_extractors.py` `NLP_BACKENDS`. |
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

**The ingest-store cache** (per-instance mode, default ON): extraction is ~93% of a
run's cost, so re-paying it when a change is query-side-only
(retrieval/rerank/context/reader/judge) is pure waste. Each instance's ingested store is
cached at `store/cache/<instance_id>-<key12>.db`, keyed off a hash of the instance's
session content + every ingest-relevant config field (model, chunking, canonicalization
thresholds, extractor prompt text — see `INGEST_RELEVANT_FIELDS` in
`kg/ingest_cache.py`) — query-side fields never bust it. `run.json` reports
`ingest.totals.cached_instances`/`fresh_instances` and a per-instance `ingest_cached`
flag so a cached run is never misread as "ingest got cheaper" instead of "didn't run."

- Force a fresh run: `--no-ingest-cache`.
- Clear it: no CLI (no auto-eviction, by design) — `rm -rf store/cache` (everything) or
  `rm store/cache/<instance_id>-*.db` (one instance).
- There's no "pick a cache" flag: whichever entry matches your *current* config is used
  automatically; an ingest-relevant config change just misses and re-ingests under a new
  key.

## Test corpus

**LongMemEval** (Wu et al., ICLR'25; Hugging Face `xiaowu0162/longmemeval-cleaned`, MIT)
— a long-term-memory benchmark of dated, multi-session chat histories. Each instance is
one user's haystack of timestamped sessions + a question + answer + the evidence
sessions; every session becomes a dated episode, so it directly exercises the temporal /
knowledge-update machinery.

The toy `dataset/longmemeval/sample` tier ships committed so `python -m kg ingest` works
with zero setup. Larger tiers must be built first:

```bash
python scripts/build_longmemeval.py                  # sample + small + med (downloads ~277 MB)
python scripts/build_longmemeval.py --tier large     # heavier (~2.74 GB)
python scripts/build_longmemeval.py --tier all       # everything
```

See [dataset/longmemeval/README.md](dataset/longmemeval/README.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the design: graph model, tag-drift
  control, embeddings, traversal/retrieval, ingestion pipeline, storage.
- [docs/CONFIG.md](docs/CONFIG.md) — every tunable, tiered by how likely you are to need
  it, with the rationale behind each default.
- [docs/TEMPORAL.md](docs/TEMPORAL.md) — the bi-temporal fact model:
  supersede-not-overwrite, validity windows, belief states.

## Design notes

The architecture draws on published work in graph-based retrieval and agent memory —
HippoRAG / HippoRAG 2 (PPR seed-and-spread retrieval), Graphiti (episodic/semantic split,
bi-temporal edges), GraphRAG (community summaries), and open relation extraction /
canonicalization (Galárraga et al., CIKM 2014; CESI, WWW 2018). The core positions:

1. **Immutable episodic layer, lean semantic layer.** Episodes and Mentions are
   append-only and embedded once; Entities are lean canonical anchors. This designs the
   re-embedding problem out — cost is proportional to new data — and makes
   "coworker → ex-coworker" / "Toronto → Berlin" resolve correctly regardless of document
   order.
2. **Diffusion is the primary retrieval path; embeddings are a seed index; the LLM does
   not traverse.** Embed to *find* entry nodes, then let graph structure do the multi-hop
   work in one step — no per-hop LLM calls.
3. **Tags and relationships are first-class, open-vocabulary, and consolidated.**
   Relationship labels are generated per connection as parallel directed edges (one per
   relation, each with its own provenance/confidence), then consolidated into canonical
   relation nodes by the same drift-control pipeline as tags — expressivity without
   free-form predicates collapsing into a vague `related_to`.
4. **Drift control is layered**: exact/normalized match → embedding synonymy *link* (not
   merge) → selective LLM adjudication. Bias toward linking near-duplicates, not
   hard-merging them; relationships consolidate on a content key so `is_friend_of` ≈
   `is_friends_with` merge while `is_enemy_of` and the passive inverse `managed_by` stay
   distinct — embedding cosine alone can't tell synonyms from antonyms.
5. **Edges carry provenance + confidence** (`EXTRACTED` / `INFERRED` / `SIMILAR` /
   `DERIVED`) so retrieval can down-weight or drop low-confidence relationships, and fact
   edges carry a bi-temporal window + belief state so stale facts drop out of traversal
   automatically.
