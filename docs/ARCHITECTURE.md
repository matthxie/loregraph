<!-- Generated from a literature-review workflow over 10 sources. See LITERATURE.md for per-source notes. -->

# Knowledge Graph Architecture & Decisions

A concrete design for an LLM-traversable, bidirectional knowledge graph over multimodal content, synthesized from 10 sources.

---

## 0. Decided stack (locked 2026-06-21)

These choices override the option menus in §4 and §7 below.

| Concern | Decision | Notes |
|---|---|---|
| **Pipeline LLM** (summary, tag/entity extraction, image captioning) | **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) | Vision-capable; cheapest tier for the 100-article batch. Needs `ANTHROPIC_API_KEY`. |
| **Embeddings** | **Local `sentence-transformers`** | Default model `BAAI/bge-small-en-v1.5` (384-dim, strong quality/size); `all-MiniLM-L6-v2` as a lighter fallback (what A-MEM used). Fully offline, no API key. |
| **Embedding grain** | **Summary embeddings = primary** retrieval surface; tag/entity-string embeddings = synonymy linking only | Per §4; bare-tag embeddings alone are too lossy. |
| **Graph store** | **NetworkX** in-memory, persisted to disk | PageRank/BFS/Leiden out of the box. Migration target if it outgrows memory: Kùzu. |
| **Vector store** | **NumPy brute-force cosine** to start | ~100 articles → milliseconds. Add `sqlite-vec`/FAISS only if it grows. |
| **Metadata/cache** | **SQLite** (stdlib) | SHA256 content-hash cache + node metadata. |
| **Python** | **3.14 (system, native)** | Verified: `torch` 2.12.1 ships a macOS arm64 `cp314` wheel and `sentence-transformers` 5.6 supports ≥3.10 — no 3.12 venv required. Still use a project venv for isolation. |
| **Drift control scope (MVP)** | L1 hash + L2 embedding synonymy **link** (not merge); L3 batch reconciliation deferred | Per §3; bias toward linking near-duplicates over hard-merging. |
| **Test corpus** | **`wikimedia/wit_base`** (HF, streamed) — ~100 deduped page nodes, text + embedded image bytes, CC BY-SA 4.0 | Replaces live Wikipedia scraping. See [DATASET.md](DATASET.md). Consequence: **no free hyperlink/category edges** — edges are *derived* from shared tags/entities + embedding kNN (optional offline API enrichment for ground-truth edges). |

---

## 1. Problem framing

**What this is.** You are building a *graph-RAG memory system*: an offline indexing pipeline that turns arbitrary objects (text, links, photos, files) into a persistent, typed, bidirectional graph, plus an online retrieval path where an LLM answers queries by combining vector entry-points with graph traversal. This is the same system class as **HippoRAG**, **GraphRAG (Microsoft)**, **A-MEM**, and **Graphify** — not classic chunk-and-embed RAG.

**The 2-3 big lessons the literature agrees on:**

1. **An entity/concept graph beats chunk-embedding RAG for cross-document and multi-hop reasoning.** Every source that compared (HippoRAG, GraphRAG, Graphify) found that explicit nodes + typed edges integrate knowledge across passage boundaries that isolated chunk embeddings cannot. This validates the core bet of your project.

2. **Traversal/diffusion should be the primary retrieval path; embeddings are an *entry-point index*, not the main retrieval mechanism.** Graphify uses no embeddings at all (pure BFS/DFS). HippoRAG uses embeddings only to *seed* Personalized PageRank, then diffuses over the graph — and explicitly found that naive 1-hop neighbor expansion *hurt* vs. principled diffusion. RAPTOR found flat similarity over *all* abstraction levels beats level-by-level walking. The convergent lesson: **embed to find seeds, then let graph structure do the work — don't LLM-walk hop-by-hop as the default.**

3. **Don't over-engineer hard tag normalization; semantic consolidation is what matters.** GraphRAG uses naive exact-string entity matching and lets clustering/summarization absorb duplicates. HippoRAG keeps near-duplicates as *distinct nodes linked by synonymy edges* rather than merging. A-MEM and TnT-LLM re-harmonize tags semantically (LLM evolution / a maintained taxonomy) rather than relying on a string hash. A hash table is a cheap *first* layer, not the load-bearing solution.

A fourth, weaker consensus: **summarize-first.** TnT-LLM, RAPTOR, A-MEM, and Graphify all reduce each object to an LLM summary *before* extraction, because the summary normalizes length and noise. Your pipeline already does this — keep it.

---

## 2. Graph model

### Node types

| Node type | Source / object | Notes |
|---|---|---|
| **ObjectNode** | one per ingested item (article, link, photo, file) | Holds raw ref, LLM summary, content hash (SHA256), modality. The "document" unit. |
| **EntityNode** | named entities/concepts extracted from the summary | Typed (person, place, org, concept…). The traversal backbone. |
| **TagNode** | canonical tags from the taxonomy | First-class nodes (see below). |
| **CommunityNode** *(phase 3)* | Leiden communities | Holds a precomputed cluster summary for breadth queries. |

### Edge types (all carry provenance + confidence)

- `MENTIONS` (ObjectNode → EntityNode)
- `TAGGED_AS` (ObjectNode → TagNode)
- `RELATED_TO` / typed relations (EntityNode ↔ EntityNode) — from extraction
- `SIMILAR_TO` (any ↔ any) — embedding cosine above threshold, the synonymy edge
- `SHARED_TAG` / `SHARED_ENTITY` (ObjectNode ↔ ObjectNode) — **derived, overlap-weighted**; the primary ObjectNode↔ObjectNode edge now that the corpus has no wikilinks
- `IN_COMMUNITY` (node → CommunityNode)
- `HYPERLINKS_TO` (ObjectNode → ObjectNode) — *optional enrichment*: deterministic Wikipedia links/categories fetched offline per `page_url` (the `wit_base` corpus does not include them; see [DATASET.md](DATASET.md))

**Tag the edges, not just nodes.** Per **Graphify** and **Chain-of-Layer**, every extracted edge stores `provenance ∈ {EXTRACTED, INFERRED, SIMILAR}` and a `confidence ∈ [0,1]`. This lets the LLM weight or discard low-confidence relationships during traversal and is your single best hallucination-control lever. Validate each edge before commit (Chain-of-Layer's per-layer filter; A-MEM's LLM link gate).

### Tags: first-class nodes vs attributes

**Recommendation: make tags first-class TagNodes, *and* keep a denormalized tag list as an ObjectNode attribute for cheap filtering.**

- **For nodes** (so they can be traversed/clustered): **HippoRAG** (nodes are noun-phrases/entities), **GraphRAG** (typed entity nodes), **TaxoGen/TaxoCom** (a node = a coherent term cluster). **Leiden (Louvain→Leiden)** is explicit: *"if tags are mere attributes (not connected nodes) clustering gets nothing to cluster — you must build tag-cooccurrence / node-node edges first."* That settles it for any system wanting topical structure.
- **Tags as a flat label layer** is also endorsed by **TnT-LLM** (tags are abstract labels that connect docs sharing no literal entity) — but that's about the *vocabulary*, not storage. Keep the denormalized attribute purely as a fast pre-filter.

Net: TagNodes participate in the graph; the attribute copy is an index optimization.

### Bidirectionality

Represent every edge **once, undirected-in-effect**. Concretely: store a directed row but **always insert/traverse both directions** (an adjacency that returns neighbors regardless of stored direction). This matches HippoRAG's PPR (runs over undirected edges), GraphRAG's degree-weighted undirected edges, and Graphify's BFS over bidirectional links. A-MEM's "confirmed links are mutual" is the same idea. Do *not* model direction semantics except where they're real (`HYPERLINKS_TO`, `is-a`).

### Multimodal (image) nodes

Images become **ObjectNodes with modality=image**. Per Graphify (Claude Vision) and the unanimous text-only papers' gap: **caption-then-treat-as-text.** A VLM (Claude with vision) produces a summary/caption; from there the image flows through the *identical* extraction → tag → embed path as text. The image's semantics are only as good as the captioner (RAPTOR's caution about the captioner being the ceiling). Store the original image ref and the generated caption on the node.

---

## 3. Tagging & drift control

### How the literature handles normalization/synonymy

- **HippoRAG (node resolution):** does **not merge**. Adds `SIMILAR_TO` synonymy edges between nodes whose embedding cosine > τ=0.8, and lets PPR flow across them. Tunable, preserves nuance, avoids over-merging.
- **GraphRAG (entity dedup):** naive **exact-string** matching; relies on downstream Leiden clustering + community summarization to absorb near-duplicates. Explicitly says perfect normalization is "less critical than feared."
- **A-MEM:** **memory evolution** — on insert, the LLM rewrites neighbors' tags to re-harmonize them. Semantic, not lexical.
- **TnT-LLM:** maintain a **global tag taxonomy state** with a Generate-Update-Review loop and a cardinality cap; catches `automobile`/`cars`/`vehicles` that a hash misses, and stores a *description* per tag.
- **TaxoCom:** drift is **context-dependent** — the same tag is generic at one node, discriminative at another. Mint a new tag only if its softmax membership novelty vs. existing sibling tags exceeds an adaptive threshold.
- **TaxoGen:** score tags by **popularity × concentration**; diffuse tags (appearing everywhere) are generic noise, concentrated tags are real concepts.

### Recommended canonicalization design (concrete, layered)

Three cheap layers, in order — implement L1+L2 for MVP, L3 later:

**L1 — Exact/normalized hash table (cheap, deterministic).**
Key = `lower(strip_punct(lemmatize(tag)))`. Collapses `Natural Language Processing` / `natural-language processing`. This is your hash table — keep it, but treat it as only the first pass (every source that uses a hash treats it as non-load-bearing).

**L2 — Embedding synonymy gate (semantic, the HippoRAG move).**
On a new tag, embed it, find nearest existing TagNodes (cosine). If cosine > **0.85**: **don't auto-merge** — create a `SIMILAR_TO` edge and flag for optional merge. Merge only above a higher bar (e.g. > 0.93) or via LLM confirmation. This preserves nuance and is tunable. Compute similarity within the *local candidate neighborhood*, not one global threshold (TaxoGen/TaxoCom: global thresholds both over- and under-merge).

**L3 — Periodic taxonomy reconciliation (TnT-LLM, batch).**
Once the corpus stabilizes, run TnT-LLM's Generate-Update-Review loop in minibatches over the tag set to merge semantic near-duplicates and assign each canonical tag a **short description** (descriptions are what let the LLM decide merges and disambiguate during traversal). Cap cardinality.

Store per canonical tag: `{id, canonical_name, description, aliases[], embedding, doc_frequency}`.

### Tags-as-nodes vs hash-table tradeoff

They are not competitors — they operate at different layers. The **hash table is a write-time dedup mechanism**; **tags-as-nodes is the storage/traversal model**. The real tradeoff is **merge vs. link**: hard-merging (hash) is simple but risks collapsing distinct concepts and is lossy; soft-linking (`SIMILAR_TO` + community detection) preserves nuance but shifts the burden to clustering quality (Graphify's noted pitfall: near-dupes proliferate). **Recommendation: link by default (L2), merge only on high-confidence exact/near-exact (L1 + cosine>0.93), reconcile in batch (L3).** This is the HippoRAG + TnT-LLM hybrid and is the safest for a solo dev.

Add **node specificity (HippoRAG): weight each tag/entity by inverse document-frequency** (1/|docs containing it|) so generic tags get downranked during retrieval. Cheap, and it improved HippoRAG's results. This is the same signal as TaxoGen's "concentration."

---

## 4. Embeddings

### What the papers found

| System | What it embeds | Verdict |
|---|---|---|
| HippoRAG | **short node strings** (entities/phrases) | for *linking + synonymy only*, not primary retrieval |
| RAPTOR | **summaries + raw chunks** | dense summary embeddings are the retrieval surface; flat over all levels |
| A-MEM | **note text** (content + context), *not tags alone* | bare-tag embeddings lose disambiguating context |
| TnT-LLM | nothing (LLM reasoning over summaries) | mild challenge to "embed tags" |
| GraphRAG | summaries (local mode); **none** for global | embeddings don't serve whole-corpus questions |
| Graphify | nothing | topology only |
| TaxoCom/TaxoGen | terms, but **re-fit locally** | one global tag embedding lacks fine discrimination |

**The clear convergent finding:** **bare-tag embeddings are too lossy as your primary retrieval vector.** A-MEM, RAPTOR, and TaxoCom all say so directly. Tags are short and context-free.

### Concrete recommendation

**Embed two things, at two granularities:**

1. **Summary embeddings (primary retrieval surface).** Embed each ObjectNode's LLM summary. This is your main query→seed entry index (RAPTOR, A-MEM). Optionally embed `summary + canonical_tags` concatenated to inject symbolic signal.
2. **Tag/entity string embeddings (for synonymy + linking only).** Embed canonical TagNode/EntityNode strings purely to power the L2 synonymy gate and tag→node linking (HippoRAG). Not the primary query path.

**Do not** rely on embeddings for whole-corpus/"what are the themes" questions — GraphRAG showed those need the **community-summary layer** (Section 5), not vector search.

**Granularity & model family:**
- Use **summary-level** as the retrieval grain (cleaner than raw chunks for ~100 articles; chunks optional later for fine recall).
- **Model:** a strong general sentence-embedding model. For a solo prototype, pick *one* and move on: `text-embedding-3-small` (OpenAI, hosted, trivial) **or** a local `bge`/`gte`/`all-MiniLM` family model if you want zero API cost (A-MEM used `all-MiniLM-L6-v2`; RAPTOR used `multi-qa-mpnet`). For ~100 articles either is instant. **Use cosine / normalize to unit length** (TaxoCom: spherical/vMF aligns training with usage and supports softmax membership scoring).
- You likely get TaxoGen's "local discrimination" benefit *for free* from modern contextual embeddings — skip per-subtree re-embedding.

---

## 5. Traversal & retrieval

### The options the literature offers

- **HippoRAG — Personalized PageRank seed-and-spread.** Embed query entities → link to seed nodes → run PPR (damping 0.5) → single-step multi-hop. 10-30× cheaper than iterative LLM walking; naive neighbor expansion measurably *hurt*. Seeds reweighted by node specificity (IDF).
- **GraphRAG — community summaries + map-reduce.** For *global/breadth* questions: precompute Leiden communities, summarize each, map-reduce over summaries. Local traversal misses corpus-wide themes.
- **RAPTOR — collapsed-tree flat retrieval.** Flatten all abstraction levels into one pool, cosine-rank, greedy fill a token budget. Beats level-by-level walking — let the retriever pick granularity.
- **Graphify — BFS/DFS minimal subgraph.** Pure traversal, returns the minimal subgraph; needs the visited-set (your "seen" flag).

### Recommended traversal for THIS design (a 2-path retriever)

**Path A — Local/specific queries (default):**
1. Embed query → cosine over **summary embeddings** → top-k **seed nodes** (the entry-point index, *not* the answer).
2. Also extract query entities → link to EntityNodes/TagNodes as additional seeds (HippoRAG).
3. Run **Personalized PageRank** seeded at those nodes over the bidirectional graph (`RELATED_TO` + `SIMILAR_TO` + `HYPERLINKS_TO` edges), seeds reweighted by **node specificity (IDF)**. Weight edges by their `confidence`; downweight/skip `INFERRED` low-confidence edges.
4. Take the top-ranked nodes' ObjectNodes as the minimal subgraph; hand summaries (+ raw content on demand) to the LLM to answer.

This makes **graph diffusion the primary path and embeddings the seed index** — the unanimous lesson. PPR over a ~100-article graph is milliseconds (`networkx.pagerank` with `personalization=`). It's far cheaper and more robust than LLM hop-by-hop.

**Path B — Global/breadth queries** ("main themes across everything"):
Route to the **GraphRAG community-summary layer** (Section 6 / phase 3): Leiden communities, each with a precomputed summary, map-reduced. Detect breadth queries with a cheap classifier or an LLM router prompt.

**LLM-guided traversal — reserve it.** Live LLM walking (the seen-flag loop) is the *debug/explainability* path and the **path-finding** fallback when PPR returns a weak/disconnected subgraph, *not* the default. When you do walk, serialize the current subgraph compactly into the prompt (Chain-of-Layer's Hierarchical Format) and respect the ~80-node-per-prompt ceiling — chunk the graph into bounded sub-views.

### Role of the "seen" flag

**Validated as the correct visited-set primitive** (Graphify: "BFS over a bidirectional graph needs exactly that visited-set semantics"). Its real jobs:
- **Cycle avoidance** during any BFS/DFS or LLM-guided local expansion over the bidirectional (hence cycle-rich) graph.
- **Debug/traceability** — inspect exactly which nodes the LLM touched answering a query.
- **Bulk-clear** resets it between queries in one call. Keep this; it's cheap and correct.

Note: the "seen" flag is **orthogonal to PPR** (diffusion doesn't need it). It matters specifically for the traversal/debug path — which is exactly where you spec'd it.

---

## 6. Ingestion pipeline

### Step-by-step for one object

```
1. INTAKE        Accept text / link / image / file. Compute SHA256(content).
2. CACHE CHECK   If hash in cache → skip (Graphify incremental rebuild). Else continue.
3. NORMALIZE     Links → fetch + extract main text. Files → extract text. Images → step 4.
4. SUMMARIZE     LLM (vision-capable for images) → concise summary.
                 Images: Claude Vision caption, then identical downstream path.
5. EXTRACT       From the SUMMARY (not raw text — TnT-LLM): LLM emits
                 entities (typed), concepts, candidate tags, and typed
                 relations between entities. Use a "gleaning" 2nd pass
                 ("did you miss any?") to lift recall (GraphRAG).
6. DERIVED EDGES Corpus = wit_base (no free wikilinks/categories — see
                 DATASET.md). Derive edges instead: shared tags/entities
                 (overlap-weighted) + embedding-kNN SIMILAR_TO. OPTIONAL:
                 one-time offline Wikipedia API call per page_url to fetch
                 real categories/links as ground-truth deterministic edges.
7. CANONICALIZE  Each candidate tag → L1 hash → L2 embedding synonymy gate
                 (Section 3). Link or merge. Update doc_frequency.
8. EMBED         Summary embedding (primary). Tag/entity-string embeddings
                 (for synonymy/linking).
9. WRITE GRAPH   Create ObjectNode; create/link EntityNodes & TagNodes;
                 insert edges with {provenance, confidence}, both directions.
                 Gate each LLM-inferred edge (A-MEM LLM link gate / CoL filter).
10. CACHE        Store hash → node id.
```

(Defer A-MEM-style "memory evolution" rewrites of neighbors — it's O(k) LLM calls per insert and causes non-deterministic drift. Use batch L3 reconciliation instead.)

### Test corpus & harness

- **Corpus:** ~100 deduped page nodes from **`wikimedia/wit_base`** (HF, streamed) — each with text (page summary + section context) and an embedded image, CC BY-SA 4.0. Frozen + reproducible, bytes-in-hand, no scraping. **Tradeoff:** image-anchored (not full-article) text and **no free hyperlink/category edges** — edges are derived (shared tags/entities + embedding kNN), with optional offline Wikipedia-API enrichment per `page_url` for ground-truth edges. Full rationale, loader, and caveats in [DATASET.md](DATASET.md).
- **Eval, two tiers:**
  1. **Retrieval correctness (objective):** author ~30-50 questions — a mix of single-article, multi-hop cross-article, and 2-3 global/theme questions. For multi-hop, you know the ground-truth article set, so measure **recall@k of the retrieved subgraph**. Reuse GraphRAG's **Claimify claim-extraction + clustering** for comprehensiveness/diversity if you want rigor.
  2. **Answer quality (subjective):** **LLM-as-judge head-to-head** on comprehensiveness/diversity/empowerment with a **directness control** to catch verbosity bias (GraphRAG's exact protocol — directly reusable).
- **Baselines to beat:** (a) flat chunk-embedding RAG, (b) summary-embedding top-k with no graph. Showing the graph/PPR path wins on multi-hop is the whole thesis.
- **Ablations worth running:** PPR vs. naive 1-hop expansion (HippoRAG predicts PPR wins); with vs. without `HYPERLINKS_TO` deterministic edges; with vs. without node-specificity reweighting.

---

## 7. Storage choice

**Recommendation: NetworkX (in-memory) + a flat vector index, persisted to disk. Use sqlite-vec (or a single FAISS file) for vectors and a single `graph.pkl`/`graph.json` for the graph.**

Justification for a solo prototype at ~100 nodes (low thousands of nodes/edges):

- **NetworkX** gives you PPR (`pagerank(personalization=…)`), BFS/DFS, and `python-louvain`/`graspologic` Leiden **out of the box** — this is *exactly* Graphify's and effectively GraphRAG's stack. No query language to learn, trivial to set the "seen" flag as a node attribute and bulk-clear it. The literature's own implementations live here.
- **Vectors:** `sqlite-vec` keeps everything in one SQLite file (graph metadata *and* vectors co-located, easy backup, zero server). FAISS is fine too but adds a second artifact. For ~100 articles a brute-force cosine in NumPy is honestly enough — don't over-build.
- **Persistence:** pickle/JSON the graph with a **SHA256 content-hash cache** so re-ingesting the test set only reprocesses changed inputs (Graphify).

**Why not the alternatives (now):**
- **Neo4j / Kùzu:** real graph DBs with Cypher, but you pay setup + a query language, and you'd *still* drop to Python for PPR + embeddings + LLM calls. Overkill below ~10k nodes. **Kùzu** is the right migration target *if* you outgrow memory (embedded, columnar, fast graph queries, no server) — note it but don't start there.
- **DuckDB:** great analytical SQL + `vss` extension, but weak for iterative graph algorithms (PPR/Leiden) — you'd reimplement them.
- **Pure vector DB (Chroma/Pinecone):** wrong primitive — no traversal, which is your primary path.

**Known ceiling (accept it):** NetworkX is single-process, in-memory, not concurrent (Graphify's noted pitfall). Fine for the prototype. Migration path if it grows: **Kùzu for the graph + keep sqlite-vec/FAISS for vectors.**

---

## 8. Concrete build plan

Each milestone is independently shippable and demoable.

**Phase 0 — Skeleton + ingestion (MVP-of-MVP).**
- `wit_base` loader (stream ~100 page nodes: text + embedded image → `corpus/`). See [DATASET.md](DATASET.md).
- SHA256 cache. NetworkX graph + SQLite (metadata) + NumPy cosine (vectors).
- Claude Haiku 4.5 summarize (vision for the image) → extract entities/tags (single pass, no gleaning yet).
- L1 hash normalization only. Write ObjectNodes + TagNodes + `TAGGED_AS` + derived `SHARED_TAG` edges.
- **Ship:** "ingest 100 nodes, dump the graph, eyeball it in a notebook."

**Phase 1 — Drift control + edges.**
- L2 embedding synonymy gate (`SIMILAR_TO`). Node specificity (IDF).
- Typed `RELATED_TO` extraction with `{provenance, confidence}`; LLM link gate on inferred edges.
- Gleaning second-pass for recall.
- **Ship:** "tags no longer drift; edges carry confidence."

**Phase 2 — Embeddings + retrieval (the core thesis).**
- Summary embeddings as the seed index; tag/entity embeddings for linking.
- **PPR seed-and-spread retriever** (Path A): query → embed → seed → personalized PageRank → minimal subgraph → LLM answers.
- "Seen" flag + bulk-clear wired into an optional BFS/LLM-walk debug path.
- **Ship:** "ask multi-hop questions, get cross-article answers."

**Phase 3 — Communities + breadth queries.**
- Leiden (graspologic, **not** Louvain — connectivity guarantee) over the tag/entity graph. CommunityNodes + precomputed community summaries.
- Query router: local → PPR, global → community map-reduce (Path B).
- **Ship:** "ask 'what are the main themes' and get a real answer."

**Phase 4 — Eval harness + hardening.**
- The Wikipedia-100 question set; recall@k + LLM-as-judge-with-directness-control.
- Ablations (PPR vs 1-hop; ±deterministic edges; ±IDF).
- Optional: TnT-LLM batch taxonomy reconciliation (L3) with per-tag descriptions.
- **Ship:** "numbers showing the graph beats flat RAG on multi-hop."

---

## 9. Risks & open questions

**Genuinely uncertain / needs your decision:**

1. **Merge vs. link threshold for tags.** The whole drift-control design hinges on cosine cutoffs (link >0.85, merge >0.93 are starting guesses). These are corpus-dependent (TaxoCom). *Decision needed:* tune on the 100-article set, or accept defaults and revisit. I recommend **link-biased** (under-merge) — over-merging silently destroys distinctions and is hard to detect.

2. **Does PPR actually beat LLM-walking on YOUR corpus?** HippoRAG's gains were largest on *entity-centric* data (2WikiMultiHop) and weaker on HotpotQA's "concept-context tradeoff." Wikipedia is mixed. This is the central empirical bet — the Phase 4 ablation answers it. If PPR underperforms, fall back to RAPTOR-style collapsed-tree flat retrieval before reaching for expensive LLM traversal.

3. **Image semantics ceiling.** Every image's value is capped by the VLM caption quality (RAPTOR). Wikipedia images are often decorative/captioned-already. *Open question:* are images even worth ingesting as full nodes, or just as attributes on their article's ObjectNode? Cheap experiment: try both, measure if image nodes ever appear in correct retrieval subgraphs.

4. **Extraction quality on long articles** is the dominant error source (HippoRAG, Chain-of-Layer's ~80-entity ceiling). Wikipedia articles are long. *Mitigation:* extract from the *summary* not raw text (already planned), use gleaning, and possibly section-wise summarization for very long articles.

5. **Hallucinated `INFERRED` edges** persist as "facts" in a stored graph (RAPTOR: ~4% build-time hallucination *ossifies*). Confidence tags *surface* but don't *fix* this, and you have no human-review loop (Graphify's AMBIGUOUS assumes one). *Decision:* set a confidence floor below which inferred edges are dropped, or stored-but-excluded-from-traversal.

6. **Staleness of communities & taxonomy.** Both Leiden communities (Louvain→Leiden) and the TnT-LLM taxonomy are batch artifacts that go stale on every insert. For a growing store this needs a re-index cadence. At 100 articles, ignore; flag before you scale.

7. **Determinism.** Leiden is stochastic (seed-dependent cluster IDs). Pin the seed if you need reproducible community IDs across runs.

**Things the literature does *not* cover (you own them):** all multimodal/image handling (every paper except Graphify is text-only), incremental updates for most methods, and concurrent/scalable storage. These are extensions, not solved problems you can lift.