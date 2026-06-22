# you

An **LLM-traversable knowledge graph** over multimodal content (text, links, photos, files).

Each ingested object becomes a node: an LLM summarizes it, extracts tags/entities/concepts
from the summary, and links it into a **bidirectional graph**. Retrieval is *graph-first* — an
LLM (or a PageRank-style diffusion) traverses the graph rather than relying on chunk-embedding
similarity alone. Embeddings exist mainly as an **entry-point index** to find seed nodes.

## Status

**MVP built** — the `kg/` package implements the design end-to-end (ingestion → typed
bidirectional graph → 2-path retrieval → communities → eval). It runs fully offline by
default (pluggable backends fall back when no API key / model is present) and upgrades to
Claude Haiku 4.5 + `bge-small` embeddings when they are. See [docs/MVP.md](docs/MVP.md)
for the feature→design map, run commands, and the thesis-validating eval results. The
design is grounded in a literature review of 10 GraphRAG / graph-retrieval /
taxonomy-induction sources — see the docs below.

```bash
pip install -r requirements.txt
python -m kg ingest --reset && python -m kg communities
python -m kg query "what are the main themes across the collection"
python -m kg eval        # recall@k ablation: PPR vs BFS vs flat vector
python -m pytest -q
```

## Decided stack

- **Pipeline LLM:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), vision-capable for images.
- **Embeddings:** local `sentence-transformers` (`BAAI/bge-small-en-v1.5`), fully offline.
- **Graph:** NetworkX (in-memory, persisted) — PageRank/BFS/Leiden out of the box.
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
2. **Tags are first-class nodes**, not just attributes — so they can be clustered and traversed.
3. **Drift control is layered**, not a single hash: exact/normalized hash → embedding synonymy
   *link* (not merge) → periodic taxonomy reconciliation. Bias toward *linking* near-duplicates,
   not hard-merging them.
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
