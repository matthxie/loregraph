# MVP — the `kg` package

This is the implementation of the design in [ARCHITECTURE.md](ARCHITECTURE.md). It
turns the 200-item frozen corpus (`dataset/`) into a persistent, typed, directed
knowledge graph and answers queries by combining vector entry-points with graph
traversal — exactly the system class (HippoRAG / GraphRAG / A-MEM) the design targets.

## What's built (feature → design section)

| Feature | Where | Design ref |
|---|---|---|
| Graph model: Object/Entity/Tag/Community nodes; typed edges w/ provenance + confidence | `kg/models.py` | §2 |
| Timestamps + `valid`/`superseded_by` soft-invalidation; `seen` debug flag | `kg/models.py`, `kg/store.py` | §2 rev 2 |
| Open-vocab, multi-label relationship tags consolidated like tags (rev 3) | `kg/canonicalize.py` `resolve_relation`, `kg/models.py` `RelationTagNode` | §2, §3b |
| Directed `MultiDiGraph` + symmetrized traversal projection (rev 3) + SQLite + NumPy cosine | `kg/store.py`, `kg/retrieval.py`, `kg/vectors.py` | §0, §2, §7 |
| Structured-output extraction direct from raw content (+ reflexion) | `kg/extractors.py` `HaikuExtractor` | §6.4, §10 |
| Vision path for images (tags + one-line description) | `kg/extractors.py` | §2 multimodal |
| L1 hash + L2 embedding-synonymy **link** (not merge) + entropy guard | `kg/canonicalize.py` | §3 |
| Node-specificity / IDF weighting | `kg/canonicalize.py` `idf_weight` | §3 |
| Raw-text-primary embeddings; tag/entity strings for synonymy | `kg/ingest.py`, `kg/canonicalize.py` | §4 |
| Bounded-concurrency ingest; per-object SHA256 cache (skip/supersede) | `kg/ingest.py` | §6 |
| Derived Object↔Object edges: SHARED_TAG / SHARED_ENTITY (overlap×IDF) + kNN SIMILAR_TO | `kg/ingest.py` | §2, §6.5 |
| **Path A** retriever: fused seed (embedding + BM25 + query-entity link) → PPR seed-and-spread → MMR + node-distance rerank | `kg/retrieval.py` `PPRRetriever` | §5 |
| A/B baselines: plain BFS (uses `seen`), flat vector top-k | `kg/retrieval.py` | §5, §10 |
| INFERRED-edge confidence floor + skip superseded edges in traversal | `kg/retrieval.py` `projected_graph` | §2, §9.5 |
| **Path B**: Louvain communities + summaries + breadth-query router | `kg/communities.py` | phase 3 |
| Eval harness: recall@k / MRR ablation (PPR vs BFS vs vector) | `kg/evaluate.py` | phase 4 |
| CLI | `kg/cli.py` | — |

This covers Phases 0–4 of the §8 build plan. Deferred per the design: L3 batch
taxonomy reconciliation, the selective LLM ADD/UPDATE/MERGE/NOOP tie-breaker, and
optional offline Wikipedia-API `HYPERLINKS_TO` enrichment.

## Key engineering decision: pluggable backends, offline by default

The pipeline LLM (Haiku) and the embedder are both **pluggable with offline
fallbacks**, auto-selected at runtime:

- **Extractor** — `HaikuExtractor` (real, needs `ANTHROPIC_API_KEY`) ⇄
  `HeuristicExtractor` (offline: proper-noun + keyword extraction; **images use the
  COCO manifest label as the VLM stand-in**, which is a clean ground-truth proxy for
  the vision call).
- **Embedder** — `SentenceTransformerEmbedder` (real, `BAAI/bge-small-en-v1.5`) ⇄
  `HashingEmbedder` (offline, deterministic).

Why: the whole system runs end-to-end with **no API key and no network**, tests are
deterministic, and it upgrades to the full-quality path the moment a key / model is
present (`--extractor haiku --embedder st`). Set `ANTHROPIC_API_KEY` to use Haiku.

## Run it

```bash
pip install -r requirements.txt          # torch+ST optional; falls back to hashing
python -m kg ingest --reset              # build the graph from dataset/ (~12s offline)
python -m kg communities                 # detect communities for breadth queries
python -m kg query "Canadian football player who won the Grey Cup"
python -m kg query "what are the main themes across the collection"   # → global path
python -m kg inspect obj_wiki_000        # node + neighbours (provenance/confidence)
python -m kg eval --single 40 --cross 40 # the thesis ablation
python -m pytest -q                       # 26 offline tests
```

Use `--extractor haiku --embedder st` on `ingest` for the full-quality pipeline.

## Viewer (plain HTML)

A dependency-free browser viewer (`kg/viz.py` + `kg/serve.py`; vanilla JS + SVG, no
CDN) does two things:

- **Watch the graph build** — the object-level graph (200 object nodes + the derived
  object↔object edges), with a *Play build* animation that reveals nodes in ingestion
  order. Articles and images visibly form separate clusters — a direct read on §9's
  open question about whether image nodes integrate or island.
- **Trace a query's traversal** — enter a query and see the focused subgraph it
  retrieves over: seed nodes (gold), the tag/entity **hubs the traversal hops
  through**, and the ranked result objects (red, numbered), with the BFS expansion
  animated hop-by-hop. The ranked list is in the sidebar.

```bash
python -m kg serve                       # http://127.0.0.1:8000 — live typed queries
python -m kg viz --query "Canadian football Grey Cup champion" --out kg_viz.html
```

Note: retrieval here is graph diffusion (PPR) or BFS, not an LLM literally walking —
the BFS/“seen”-flag path is the design's explainability view (§5), and that is what
the traversal animation shows.

## Results — the thesis ablation

Recall@8 / MRR over auto-generated ground-truth questions on the 100-article corpus
(offline backends: heuristic extractor + bge-small embeddings):

| mode | recall@8 (overall) | **cross-article recall@8** | single-article recall@8 |
|---|---|---|---|
| **PPR** (graph) | 0.706 | **0.537** | 0.875 |
| **BFS** (graph) | 0.736 | **0.546** | 0.925 |
| vector (flat, no graph) | 0.560 | **0.145** | 0.975 |

The central bet holds: on **cross-article / multi-hop** questions (gold = every
article sharing an entity), flat vector retrieval collapses to **0.145** while the
graph paths reach **~0.54 — ~3.7× better**. On single-article lookups flat vector is
best (expected: the answer's text matches the query directly), which is precisely why
the design keeps embeddings as the *seed index* and lets the graph do cross-document
work.

It also answers the design's own open question (§5 rev 2, §9.2): **at 200 nodes plain
BFS is competitive with PPR** (BFS slightly higher recall, PPR higher MRR / ranking
quality). PPR does not yet decisively earn its complexity at this scale — an honest,
measured finding, and the reason the retriever is pluggable and A/B-able.

## Review & hardening

The code was put through a multi-agent adversarial review (correctness, design
fidelity, robustness, quality), and the confirmed findings were fixed, including:

- **Crash fixes:** empty/whitespace query no longer crashes PPR (zero-mass
  personalization `ZeroDivisionError`); a clear error replaces a cryptic matmul
  failure on embedder-dim mismatch.
- **Fidelity:** PPR/BFS diffusion is scoped to the design's edge set — `IN_COMMUNITY`
  edges are excluded so community hubs can't distort the spread (PPR is now invariant
  to whether communities are built).
- **Correctness:** re-ingest now retracts the superseded object's `doc_frequency` and
  frees its content hash; the IDF denominator counts only valid objects.
- **Robustness:** extraction errors are surfaced in the `IngestReport` (a bad/absent
  API key no longer silently yields an empty graph); active backends are shown in
  `stats()`.
- **Efficiency:** cached object count for IDF; single multi-source BFS for
  node-distance reranking; precomputed vectors in MMR; single node lookup per PPR
  candidate.

Regression tests for each of these live in `tests/test_kg.py` (33 tests, all offline).
