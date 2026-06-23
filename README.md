# you

An **LLM-traversable knowledge graph** over multimodal content (text, links, photos, files).

Each ingested object becomes a node: an LLM extracts a single open-vocabulary set of
**entities & concepts** — named things *and* the topical themes the content is about, plus
the **dates** tied to stated facts — and the **directed relationships** between them, then
links it into a **directed graph**. There is one unified vocabulary: a topical theme is just
a `concept`-typed entity (no separate "tag" node type), so the same surface can never be
duplicated as both a tag and an entity. Both the entity/concept names *and* the relationship
labels are open-vocabulary (LLM-generated) and **consolidated over time** into a canonical
vocabulary. Retrieval is *graph-first* — an LLM (or a
PageRank-style diffusion) traverses the graph rather than relying on chunk-embedding
similarity alone. Embeddings exist mainly as an **entry-point index** to find seed nodes.

## Status

**MVP built** — the `kg/` package implements the design end-to-end (ingestion → typed
directed graph → 2-path retrieval → communities → eval). It runs fully offline by
default (pluggable backends fall back when no API key / model is present) and upgrades to
Claude Haiku 4.5 + `bge-small` embeddings when they are. See [docs/MVP.md](docs/MVP.md)
for the feature→design map, run commands, and the thesis-validating eval results. The
design is grounded in a literature review of 10 GraphRAG / graph-retrieval /
taxonomy-induction sources — see the docs below.

```bash
pip install -r requirements.txt
python -m kg ingest --reset && python -m kg communities
python -m kg query "what are the main themes across the collection"   # algorithmic retrieval
python -m kg ask   "how is Alan Turing connected to the Enigma machine?"  # LLM traverses via tools
python -m kg eval        # recall@k ablation: PPR vs BFS vs flat vector (+ agent)
python -m kg serve       # browser viewer: watch the graph build + trace queries
python -m kg testrun     # one test run: ingest the temporal stream doc-by-doc + ask all
                         #   eval questions, tracking cost/tokens/tags/temporal/accuracy
python -m kg dashboard   # browse runs: Input view (structure forms) ⇄ Query view (traversal)
python -m pytest -q
```

A **test-run dashboard** instruments the pipeline end-to-end (`kg testrun` → `runs/<id>/`):
the *Input* view animates the graph forming as the `dataset/mixed/` temporal stream is
ingested one document at a time — with synchronized charts for cost, tokens, avg tags
per node, vocabulary growth and the `doc_frequency` of the top tags over time — while the
*Query* view replays the agent's traversal for every `dataset/retrieval` question with
recall@k / MRR / citation-grounding and an optional LLM-judge response score. It runs live
(real Haiku cost/tokens) or fully offline ($0, deterministic). Trigger it in Claude Code
with *"test the input and query on the full dataset"* (the `test-graph` skill).

Two query surfaces: **`query`** runs the *algorithmic* retrievers directly (PPR / BFS / vector /
community) and returns ranked objects; **`ask`** is the *agentic* path — an LLM is handed
read-only graph tools (`seed_and_spread`/PPR, `keyword_search`, `vector_search`, `neighbors`,
`find_path`, `read_object`, `browse_themes`) and **traverses the graph itself** across tool-use
turns, then answers with citations to the objects it read. Like the rest of the stack it runs
offline: with no API key, a deterministic agent executes the same tools (`--backend offline`).
See [docs/MVP.md](docs/MVP.md#agentic-retrieval-ask).

A plain-HTML viewer (no build step, no CDN — vanilla JS + SVG) shows the object
graph, animates it **being built** in ingestion order, and **traces the path a query
takes** through the graph (seeds → tag/entity hubs → ranked results, with BFS hops
animated). `python -m kg serve` for live typed queries, or `python -m kg viz --query
"…" --out kg_viz.html` for a self-contained file.

## Decided stack

- **Pipeline LLM:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), vision-capable for images.
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
- [docs/FRAMEWORKS.md](docs/FRAMEWORKS.md) — review of production memory frameworks
  (cognee, mem0, graphiti): where they confirm the design, what to adopt, and a build-vs-borrow call.

## Core design bets (from the literature)

1. **Traversal is the primary retrieval path; embeddings are a seed index.** Embed to *find*
   entry nodes, then let graph structure (PageRank diffusion / BFS) do the work.
2. **Entities/concepts *and* relationships are first-class, open-vocabulary, and
   consolidated.** Named things and topical themes share **one** entity vocabulary (a theme is
   a `concept`-typed entity — there is no separate tag node type, so nothing is stored twice);
   relationship labels (`is_friend_of`, `works_with`, `born_on`, …) are LLM-generated per
   connection as **parallel directed edges — one per relation** (the KG-triple / property-graph
   shape, so each carries its own provenance/confidence), then consolidated into canonical
   `RelationTagNode`s by the same drift-control pipeline as the entities. This is *open relation
   extraction + relation canonicalization* (Galárraga et al., CIKM 2014; CESI, WWW 2018) — the
   consolidation step is what stops free-form predicates from collapsing into a vague
   `related_to`, so we get expressivity without the drift the fixed-enum was guarding against.
3. **Drift control is layered**, not a single hash: exact/normalized hash → embedding synonymy
   *link* (not merge) → periodic taxonomy reconciliation. Bias toward *linking* near-duplicates,
   not hard-merging them. Relationships consolidate on a **content key** (drop function words +
   singularize): `is_friend_of` ≈ `is_friends_with` merge, but `is_enemy_of` and the passive
   inverse `managed_by` stay distinct — because embedding cosine alone can't tell synonyms from
   antonyms.
4. **Embed summaries (primary) + tag/entity strings (for synonymy linking)** — bare-tag
   embeddings alone are too lossy as the main retrieval vector.
5. **Edges carry provenance + confidence** (`EXTRACTED` / `INFERRED` / `SIMILAR`) so the LLM can
   down-weight or drop low-confidence relationships.
6. A per-node **`seen` flag with bulk-clear** is the visited-set primitive for BFS / debug
   traversal.

## Test corpus

~100 page nodes streamed from the **`wikimedia/wit_base`** Hugging Face dataset — each with text
(page summary + section context) and an **embedded image**, CC BY-SA 4.0. Frozen, reproducible,
bytes-in-hand (no scraping). Because the dataset has no wikilinks/categories, ObjectNode↔ObjectNode
edges are *derived* from shared tags/entities + embedding similarity (with optional offline
Wikipedia-API enrichment for ground-truth edges). See [docs/DATASET.md](docs/DATASET.md).
