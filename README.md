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

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the design: graph model, tag-drift control,
  embeddings, traversal/retrieval, ingestion pipeline, storage choice, phased build plan.
- [docs/LITERATURE.md](docs/LITERATURE.md) — per-source notes (GraphRAG, HippoRAG, RAPTOR,
  A-MEM, Graphify, TnT-LLM, Chain-of-Layer, TaxoGen, TaxoCom, Leiden).

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
