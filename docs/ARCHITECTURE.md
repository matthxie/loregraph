<!-- Generated from a literature-review workflow over 10 sources. See LITERATURE.md for per-source notes. -->

# Knowledge Graph Architecture & Decisions

A concrete design for an LLM-traversable, directed knowledge graph over multimodal content, synthesized from 10 sources. (Rev 3: open-vocabulary, consolidated, multi-label relationship tags on a directed graph — see §2.)

> **Revision 2 (2026-06-21):** (a) **No summary step** — extract tags/entities *directly* from raw content; the object's raw text (not a generated summary) is the primary embedding/retrieval surface. Images are the exception (no text → the vision model still emits tags + a one-line description as its only searchable text). (b) **Per-node timestamps** `created_at` / `last_modified`, plus a `valid` / `superseded_by` soft-invalidation flag. (c) Refinements adopted from production frameworks (cognee, mem0, graphiti) — see new **§10** and [FRAMEWORKS.md](FRAMEWORKS.md). Sections below are updated in place; older summary-first prose is superseded by §10 and this banner.

---

## 0. Decided stack (locked 2026-06-21)

These choices override the option menus in §4 and §7 below.

| Concern | Decision | Notes |
|---|---|---|
| **Pipeline LLM** (tag/entity extraction, image description) | **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) | Vision-capable; cheapest tier for the batch. **API key** auth (`ANTHROPIC_API_KEY`, Console pay-per-token) — full corpus costs ~$1 (§ cost note). `ant auth login` OAuth/subscription is the alt; don't pass `output_config.effort` (400s on Haiku). |
| **Embeddings** | **Local `sentence-transformers`** | Default model `BAAI/bge-small-en-v1.5` (384-dim, strong quality/size); `all-MiniLM-L6-v2` as a lighter fallback (what A-MEM used). Fully offline, no API key. |
| **Embedding grain** | **Raw object-text embeddings = primary** retrieval surface (no summary step); tag/entity-string embeddings = synonymy linking only | Per §4 (rev 2). Articles: embed the raw text (lead section for very long docs). Images: embed the VLM's one-line description. Bare-tag embeddings alone are too lossy. |
| **No summary** | Extract tags/entities **directly** from raw content | rev 2. Drops one LLM call + a failure point. The summary was only a proxy for the text — embed the text instead. |
| **Timestamps + validity** | Every node carries `created_at`, `last_modified`; nodes/edges carry `valid` + `superseded_by` | rev 2. Re-ingestion *supersedes* (soft) rather than overwriting/duplicating (graphiti-lite). |
| **Graph store** | **NetworkX** in-memory, persisted to disk | PageRank/BFS/Leiden out of the box. Migration target if it outgrows memory: Kùzu. |
| **Vector store** | **NumPy brute-force cosine** to start | ~100 articles → milliseconds. Add `sqlite-vec`/FAISS only if it grows. |
| **Metadata/cache** | **SQLite** (stdlib) | SHA256 content-hash cache + node metadata. |
| **Python** | **3.14 (system, native)** | Verified: `torch` 2.12.1 ships a macOS arm64 `cp314` wheel and `sentence-transformers` 5.6 supports ≥3.10 — no 3.12 venv required. Still use a project venv for isolation. |
| **Drift control scope (MVP)** | L1 hash + L2 embedding synonymy **link** (not merge); L3 batch reconciliation deferred | Per §3; bias toward linking near-duplicates over hard-merging. |
| **Test corpus** | **Decoupled (rev): 100 full Wikipedia articles** (`wikimedia/wikipedia`, text) **+ 100 random COCO photos** (`detection-datasets/coco`, images) — text and images are *not* paired | Built & on disk via `scripts/build_dataset.py` → `dataset/`. Consequence: **no free hyperlink/category edges** — edges are *derived* from shared tags/entities + embedding kNN (optional offline API enrichment). See [DATASET.md](DATASET.md). |

---

## 1. Problem framing

**What this is.** You are building a *graph-RAG memory system*: an offline indexing pipeline that turns arbitrary objects (text, links, photos, files) into a persistent, typed, directed graph, plus an online retrieval path where an LLM answers queries by combining vector entry-points with graph traversal. This is the same system class as **HippoRAG**, **GraphRAG (Microsoft)**, **A-MEM**, and **Graphify** — not classic chunk-and-embed RAG.

**The 2-3 big lessons the literature agrees on:**

1. **An entity/concept graph beats chunk-embedding RAG for cross-document and multi-hop reasoning.** Every source that compared (HippoRAG, GraphRAG, Graphify) found that explicit nodes + typed edges integrate knowledge across passage boundaries that isolated chunk embeddings cannot. This validates the core bet of your project.

2. **Traversal/diffusion should be the primary retrieval path; embeddings are an *entry-point index*, not the main retrieval mechanism.** Graphify uses no embeddings at all (pure BFS/DFS). HippoRAG uses embeddings only to *seed* Personalized PageRank, then diffuses over the graph — and explicitly found that naive 1-hop neighbor expansion *hurt* vs. principled diffusion. RAPTOR found flat similarity over *all* abstraction levels beats level-by-level walking. The convergent lesson: **embed to find seeds, then let graph structure do the work — don't LLM-walk hop-by-hop as the default.**

3. **Don't over-engineer hard tag normalization; semantic consolidation is what matters.** GraphRAG uses naive exact-string entity matching and lets clustering/summarization absorb duplicates. HippoRAG keeps near-duplicates as *distinct nodes linked by synonymy edges* rather than merging. A-MEM and TnT-LLM re-harmonize tags semantically (LLM evolution / a maintained taxonomy) rather than relying on a string hash. A hash table is a cheap *first* layer, not the load-bearing solution.

A fourth, weaker consensus the literature offered — **summarize-first** (TnT-LLM, RAPTOR, A-MEM, Graphify reduce each object to a summary before extraction to normalize length/noise) — has been **dropped in rev 2.** At ~200 clean docs the noise-normalization benefit is marginal and not worth an extra LLM call + failure point; extract tags directly from raw text and embed the raw text. (Re-introduce summaries only if you later scale to a large/noisy corpus.)

---

## 2. Graph model

### Node types

> **Update (rev 5) — tags retired into one entity/concept vocabulary.** What were "topical
> tags" are now `concept`-typed **EntityNode**s: the extractor emits a single open-vocabulary
> entity list (named things *and* themes *and* dates), so the same surface is never stored
> twice as both a tag and an entity. **TagNode / `TAGGED_AS` / `SHARED_TAG` are legacy** —
> nothing mints them anymore (the enum members are kept only so an old store still
> deserializes); objects connect to concepts via `MENTIONS`, and ObjectNode↔ObjectNode
> overlap is `SHARED_ENTITY`. The drift-control machinery below applies unchanged to the
> unified entity vocabulary. A new `date` entity type carries concrete dates (canonicalized by
> `normalize_date`), so temporal facts become edges like `born_on` / `died_on` / `founded_in`.

| Node type | Source / object | Notes |
|---|---|---|
| **ObjectNode** | one per ingested item (article, link, photo, file) | Holds raw ref + raw text (the embedding surface), content hash (SHA256), modality, `created_at`, `last_modified`, `valid`/`superseded_by`. Images also store a one-line VLM description (their only text). The "document" unit. *(No `summary` field — rev 2.)* |
| **EntityNode** | named entities/concepts extracted **directly from the raw content** (rev 2) | Typed (person, place, org, concept…). The traversal backbone. |
| **TagNode** | canonical tags from the taxonomy | First-class nodes (see below). |
| **RelationTagNode** *(rev 3)* | canonical relationship labels (`is_friend_of`, `works_with`, `founded`…) | Open-vocabulary predicate vocabulary, consolidated like TagNodes (aliases + `doc_frequency`). Each directed `RELATED_TO` edge carries ONE canonical id (`Edge.rel_tag`); a pair with several relations = several **parallel edges** (rev 4). |
| **CommunityNode** *(phase 3)* | Leiden communities | Holds a precomputed cluster summary for breadth queries. |

### Edge types (all carry provenance + confidence)

- `MENTIONS` (ObjectNode → EntityNode)
- `TAGGED_AS` (ObjectNode → TagNode)
- `RELATED_TO` (EntityNode **→** EntityNode, **directed**) — from extraction; **one parallel edge per canonical relationship** (rev 4), each carrying a single `rel_tag` id plus its own provenance/confidence/timestamp (so `[is_friend_of, works_with]` = two edges A→B)
- `SIMILAR_TO` (any ↔ any) — embedding cosine above threshold, the synonymy edge
- `SHARED_TAG` / `SHARED_ENTITY` (ObjectNode ↔ ObjectNode) — **derived, overlap-weighted**; the primary ObjectNode↔ObjectNode edge now that the corpus has no wikilinks
- `IN_COMMUNITY` (node → CommunityNode)
- `HYPERLINKS_TO` (ObjectNode → ObjectNode) — *optional enrichment*: deterministic Wikipedia links/categories fetched offline per `page_url` (the `wit_base` corpus does not include them; see [DATASET.md](DATASET.md))

**Tag the edges, not just nodes.** Per **Graphify** and **Chain-of-Layer**, every extracted edge stores `provenance ∈ {EXTRACTED, INFERRED, SIMILAR}` and a `confidence ∈ [0,1]`. This lets the LLM weight or discard low-confidence relationships during traversal and is your single best hallucination-control lever. Validate each edge before commit (Chain-of-Layer's per-layer filter; A-MEM's LLM link gate).

**Open-vocabulary relationships + consolidation (rev 3 — supersedes the rev-2 fixed enum).** Earlier revisions used a small fixed relation enum, on the cognee/graphiti/mem0 observation that *uncontrolled* free-form relations collapse into a vague `RELATED_TO`. Rev 3 keeps the expressivity of free-form relations **without** that collapse by adding the missing half of the recipe — **canonicalization**:

- Extraction emits, per directed connection, a **multi-label set** of natural-language relationship labels read source→target (`founded`, `works_with`, `member_of`, `parent_of`, …). No enum.
- Each label is consolidated by `Canonicalizer.resolve_relation` into a canonical **RelationTagNode** — the *same two-layer drift control as topical tags* (normalized-key exact hash → high-bar embedding-synonymy merge), with `doc_frequency` for IDF. Each canonical relation becomes **its own directed `RELATED_TO` edge** (rev 4 — parallel edges keyed by the rel-node id), so per-relation provenance/confidence/timestamp survive; this is the RDF-triple / Neo4j-LPG shape, and is equivalent to a multiplex / edge-colored multigraph (Kivelä et al. 2014). The diffusion projection combines parallel relations between a pair by max (not sum) so a verbose extraction can't inflate PPR weight.
- This is exactly **open relation extraction + relation canonicalization** (Galárraga et al., *Canonicalizing Open Knowledge Bases*, CIKM 2014; **CESI**, WWW 2018), the literature's standard answer to predicate sprawl. The fixed enum was guarding against *un-consolidated* free-form; with consolidation the guard is unnecessary and we gain real relationship semantics (`is_friend_of` ≠ `manages`).
- **Consolidation keys on the content word, not the embedding** (the key design choice). Embedding cosine alone *cannot* separate synonyms from antonyms — `is_friend_of`/`is_friends_with` and `is_friend_of`/`is_enemy_of` are equally close — so L1 uses a **content key** (`relation_content_key`): drop relational function words (`is`, `of`, `with`, …) and singularize, leaving the content lemma. `is_friend_of` / `is_friends_with` → `friend` (**merge**); `is_enemy_of` → `enemy` (**distinct**, different content word); `managed_by` keeps the passive `by` marker so it never collapses into `manages` (**distinct inverse**). The node keeps a readable canonical name (`is_friend_of`); variants become aliases. The L2 embedding gate (high bar, 0.95, no `SIMILAR_TO` linking) only adds the cross-lexical synonym case (`collaborates_with` ↔ `works_with`). Looser semantic merges are deferred to the batch L3 reconciliation pass (with LLM confirmation), same as tags.

**Provenance back-pointer (rev 2 — graphiti lesson).** Every EntityNode/TagNode and extracted edge stores a pointer back to the ObjectNode (and char span where cheap) it came from. Enables audit ("why is this tag here?") and re-extraction.

**Node validity (rev 2).** Beyond the `seen` debug flag, nodes/edges carry a real-data `valid` flag + optional `superseded_by`. Re-ingesting a changed object soft-invalidates the old node/edges instead of overwriting or duplicating — graphiti's edge-invalidation idea without the bi-temporal quad (unjustified for a stable Wikipedia snapshot).

### Tags: first-class nodes vs attributes

**Recommendation: make tags first-class TagNodes, *and* keep a denormalized tag list as an ObjectNode attribute for cheap filtering.**

- **For nodes** (so they can be traversed/clustered): **HippoRAG** (nodes are noun-phrases/entities), **GraphRAG** (typed entity nodes), **TaxoGen/TaxoCom** (a node = a coherent term cluster). **Leiden (Louvain→Leiden)** is explicit: *"if tags are mere attributes (not connected nodes) clustering gets nothing to cluster — you must build tag-cooccurrence / node-node edges first."* That settles it for any system wanting topical structure.
- **Tags as a flat label layer** is also endorsed by **TnT-LLM** (tags are abstract labels that connect docs sharing no literal entity) — but that's about the *vocabulary*, not storage. Keep the denormalized attribute purely as a fast pre-filter.

Net: TagNodes participate in the graph; the attribute copy is an index optimization.

### Directionality (rev 3 — supersedes "bidirectional-in-effect")

The store is a NetworkX **`MultiDiGraph`**: every edge keeps its real direction (`src→dst`), because relationship semantics are now first-class and many predicates are inherently directed (`manages`, `founded`, `located_in`, `parent_of`). This is the **store-directed / symmetrize-for-diffusion** split that the graph-RAG literature uses:

- **Storage & semantics are directed.** `RELATED_TO`, `MENTIONS`, `TAGGED_AS`, `IN_COMMUNITY`, `HYPERLINKS_TO` all have a meaningful source→target. `inspect`/viz render the arrow; `neighbors(..., direction="out"|"in")` honour it.
- **Traversal is symmetrized.** PPR/BFS run over an **undirected** weighted projection (`retrieval.projected_graph`) that collapses `src→dst` and `dst→src` into one weighted edge. This is non-negotiable for recall: **HippoRAG's PPR runs over undirected edges**, GraphRAG uses degree-weighted undirected edges, and "find content related to X" must be able to traverse a relationship *both ways*. Running PPR directly on the directed graph would create sink/dangling-node mass leaks and halve reachability. So `neighbors()` defaults to walking **both** successors and predecessors — the pre-rev-3 bidirectional contract every retriever relies on is preserved exactly; only the *stored* graph gained direction.
- A-MEM's "confirmed links are mutual" still holds at the *diffusion* layer; we simply no longer throw away the direction at the *storage* layer.

### Multimodal (image) nodes

Images become **ObjectNodes with modality=image**. An image has no text, so the VLM (Claude Haiku 4.5 with vision) is the *only* way to get a textual handle — this is the one place a description survives the rev-2 "no summary" rule. The VLM emits **tags/entities directly + one short description line** in a single structured call; the description is the image's embedding/retrieval surface (you can't embed a photo as text otherwise), and the tags flow through the identical canonicalization path as text tags. The image's semantics are only as good as the VLM (RAPTOR's caution: the describer is the ceiling). Store the original image ref + the one-line description on the node.

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

**Embed two things, at two granularities:** *(rev 2 — no summary)*

1. **Raw object-text embeddings (primary retrieval surface).** Embed each ObjectNode's **raw text directly** (for very long docs, the lead/first section — Wikipedia leads are already summary-like). For images, embed the VLM's one-line description. This is your main query→seed entry index. *Why this is safe without a summary:* the summary was only ever a length-normalized proxy for the text; at ~200 clean docs, embedding the text itself loses almost nothing and removes an LLM call + a failure point. The summary's real benefit (noise normalization) matters at scale, not here. Optionally embed `text + canonical_tags` to inject symbolic signal.
2. **Tag/entity string embeddings (for synonymy + linking only).** Embed canonical TagNode/EntityNode strings purely to power the L2 synonymy gate and tag→node linking (HippoRAG). Not the primary query path.

**Do not** rely on embeddings for whole-corpus/"what are the themes" questions — GraphRAG showed those need the **community-summary layer** (Section 5), not vector search.

**Granularity & model family:**
- Use **object-text level** as the retrieval grain (raw text / lead section; chunks optional later for fine recall on long docs).
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
1. Seed by **fused signals (rev 2)**: embed query → cosine over **object-text embeddings**, *and* run **BM25/keyword** over the raw text (mem0 v3 + graphiti both fuse lexical + semantic) → union into top-k **seed nodes** (the entry-point index, *not* the answer).
2. Also extract query entities → link to EntityNodes/TagNodes as additional seeds (HippoRAG).
3. Run **Personalized PageRank** seeded at those nodes over the **symmetrized traversal projection** of the directed graph (`RELATED_TO` + `SIMILAR_TO` + `SHARED_TAG` (+ optional `HYPERLINKS_TO`) edges), seeds reweighted by **node specificity (IDF)**. Weight edges by their `confidence`; downweight/skip `INFERRED` low-confidence and `valid=false` superseded edges.
4. **Rerank** the PPR-scored nodes with MMR (diversity) + node-distance-to-seed (query-anchored relevance) — graphiti's rerankers on top of raw scores. Take the top nodes' ObjectNodes as the minimal subgraph; hand their **raw text** (on demand) to the LLM to answer.

This makes **graph diffusion the primary path and embeddings the seed index** — the unanimous lesson. PPR over a ~200-node graph is milliseconds (`networkx.pagerank` with `personalization=`). **PPR is one mode, not the only path (rev 2):** none of cognee/mem0/graphiti use PPR (they use vector+BM25+BFS), so make the retriever pluggable and **A/B PPR-spread vs. plain BFS** — prove PPR earns its place at this scale before relying on it.

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
1. INTAKE        Accept text / link / image / file. Compute SHA256(content),
                 stamp created_at.
2. CACHE CHECK   If hash in cache → skip (Graphify incremental rebuild).
                 If same id but changed content → soft-invalidate old node
                 (valid=false, set superseded_by), bump last_modified.
3. NORMALIZE     Links → fetch + extract main text. Files → extract text.
                 Keep the raw text (it IS the embedding surface — no summary).
4. EXTRACT       DIRECTLY from raw content (rev 2 — no summary step), in ONE
                 structured-output call (cognee): Haiku returns a typed
                 {entities[], tags[], relations[]} object — each relation is a
                 DIRECTED connection with open-vocab labels[] (rev 3), consolidated
                 downstream like tags. Each item carries a provenance back-pointer
                 to this object. Images: vision call
                 emits tags/entities + one description line in the same shot.
                 Then ONE reflexion pass ("did you miss any concept?") to lift
                 recall (graphiti) — cheaper than a separate summarize call.
5. DERIVED EDGES Corpus = wit_base (no free wikilinks/categories — see
                 DATASET.md). Derive edges: shared tags/entities
                 (overlap-weighted) + embedding-kNN SIMILAR_TO. OPTIONAL:
                 one-time offline Wikipedia API call per page_url for
                 real categories/links as ground-truth deterministic edges.
6. CANONICALIZE  Each candidate tag → L1 hash → L2 embedding synonymy gate
                 (§3), with an entropy guard so short/common tags ("AI","US")
                 aren't wrongly merged (graphiti). Each relationship label runs the
                 SAME path → canonical RelationTagNode (higher merge bar; rev 3).
                 Ambiguous-only → LLM ADD/UPDATE/MERGE/NOOP tie-breaker (mem0).
                 Update doc_frequency.
7. EMBED         Raw object-text embedding (primary; lead section if long).
                 Tag/entity-string embeddings (synonymy/linking). Images:
                 embed the VLM description line.
8. WRITE GRAPH   Create ObjectNode (created_at, last_modified, valid=true);
                 create/link EntityNodes & TagNodes with provenance pointers;
                 insert DIRECTED edges — one parallel RELATED_TO edge per canonical
                 relation, each {rel_tag, provenance, confidence} (rev 4). Gate each
                 LLM-inferred edge (A-MEM/CoL filter).
9. CACHE         Store hash → node id.
```

Run ingestion under a **bounded-concurrency semaphore** (graphiti's `SEMAPHORE_LIMIT`) — the per-object Haiku calls fan out, so cap parallelism to avoid 429s.

(Defer A-MEM-style "memory evolution" rewrites of neighbors — it's O(k) LLM calls per insert and causes non-deterministic drift. Use batch L3 reconciliation instead. The mem0 tie-breaker in step 6 is the *selective* alternative — only on ambiguous merges, not every write.)

### Test corpus & harness

- **Corpus (decoupled, rev):** **100 full Wikipedia articles** (`wikimedia/wikipedia` `20231101.en`, streamed, seed 42) **+ 100 random COCO photos** (`detection-datasets/coco`), text and images **not paired**. Already built on disk in `dataset/` via `scripts/build_dataset.py` (reproducible). **No free hyperlink/category edges** — edges are derived (shared tags/entities + embedding kNN), with optional offline Wikipedia-API enrichment for ground-truth edges. Full rationale, loader, caveats in [DATASET.md](DATASET.md).
- **Eval, two tiers:**
  1. **Retrieval correctness (objective):** author ~30-50 questions — a mix of single-article, multi-hop cross-article, and 2-3 global/theme questions. For multi-hop, you know the ground-truth article set, so measure **recall@k of the retrieved subgraph**. Reuse GraphRAG's **Claimify claim-extraction + clustering** for comprehensiveness/diversity if you want rigor.
  2. **Answer quality (subjective):** **LLM-as-judge head-to-head** on comprehensiveness/diversity/empowerment with a **directness control** to catch verbosity bias (GraphRAG's exact protocol — directly reusable).
- **Baselines to beat:** (a) flat chunk-embedding RAG, (b) raw-text-embedding top-k with no graph. Showing the graph/PPR path wins on multi-hop is the whole thesis.
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
- Load corpus from `dataset/` (100 article texts + 100 images, decoupled — see [DATASET.md](DATASET.md)). *(Done: `scripts/build_dataset.py`.)*
- SHA256 cache. NetworkX graph + SQLite (metadata) + NumPy cosine (vectors).
- Claude Haiku 4.5 **structured-output extraction directly from raw content** (no summary; rev 2): typed `{entities[], tags[], relations∈enum[]}` in one call (+ vision call for images → tags + description). Provenance back-pointers.
- L1 hash normalization only. Write ObjectNodes (with `created_at`/`last_modified`/`valid`) + TagNodes + `TAGGED_AS` + derived `SHARED_TAG` edges. Bounded-concurrency semaphore.
- **Ship:** "ingest 200 nodes, dump the graph, eyeball it in a notebook."

**Phase 1 — Drift control + edges.**
- L2 embedding synonymy gate (`SIMILAR_TO`) **+ entropy guard** for short/common tags. Node specificity (IDF).
- Typed (enum) relation extraction with `{provenance, confidence}`; LLM link gate on inferred edges; **reflexion pass** for recall.
- **Selective LLM tie-breaker** (ADD/UPDATE/MERGE/NOOP) only on ambiguous merges. **Soft-invalidation** (`valid`/`superseded_by`) on re-ingest.
- **Ship:** "tags no longer drift; edges carry confidence; re-ingest supersedes cleanly."

**Phase 2 — Embeddings + retrieval (the core thesis).**
- Raw object-text embeddings as the seed index; tag/entity embeddings for linking. **BM25 keyword index** as a parallel seed signal.
- **PPR seed-and-spread retriever** (Path A) **+ MMR/node-distance reranking** — but make it pluggable and **A/B against plain BFS** to justify PPR at 200 nodes (rev 2).
- "Seen" flag + bulk-clear wired into an optional BFS/LLM-walk debug path.
- **Ship:** "ask multi-hop questions, get cross-article answers — with numbers comparing PPR vs BFS."

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

3. **Image semantics ceiling.** Every image's value is capped by the VLM description quality (RAPTOR). The corpus is now **decoupled** — 100 standalone COCO photos, not article-attached — so images are independent ObjectNodes whose only handle is the VLM's tags + one-line description. *Open question:* do these image nodes connect into the same graph as the text nodes via shared tags (e.g. a "dog" photo linking to a dog article), or do they form an isolated island? Cheap experiment: check whether image nodes ever appear in correct cross-modal retrieval subgraphs; if not, the image path is a separate demo, not integrated memory.

4. **Extraction quality on long articles** is the dominant error source (HippoRAG, Chain-of-Layer's ~80-entity ceiling). Wikipedia articles are long. *Mitigation (rev 2 — no summary):* extract directly from raw text with a reflexion pass; for very long docs, extract section-by-section (per-section extract, union the tags) rather than truncating. This replaces the old "extract from the summary" mitigation.

5. **Hallucinated `INFERRED` edges** persist as "facts" in a stored graph (RAPTOR: ~4% build-time hallucination *ossifies*). Confidence tags *surface* but don't *fix* this, and you have no human-review loop (Graphify's AMBIGUOUS assumes one). *Decision:* set a confidence floor below which inferred edges are dropped, or stored-but-excluded-from-traversal.

6. **Staleness of communities & taxonomy.** Both Leiden communities (Louvain→Leiden) and the TnT-LLM taxonomy are batch artifacts that go stale on every insert. For a growing store this needs a re-index cadence. At 100 articles, ignore; flag before you scale.

7. **Determinism.** Leiden is stochastic (seed-dependent cluster IDs). Pin the seed if you need reproducible community IDs across runs.

**Things the literature does *not* cover (you own them):** all multimodal/image handling (every paper except Graphify is text-only), incremental updates for most methods, and concurrent/scalable storage. These are extensions, not solved problems you can lift.

---

## 10. Refinements from production frameworks (cognee · mem0 · graphiti)

Three production memory frameworks were reviewed *after* the original paper-based design — full analysis in [FRAMEWORKS.md](FRAMEWORKS.md). They **confirmed the foundation** (local NetworkX+SQLite+local-embeddings stack, summarize-or-text-first→tags, layered cheap-first dedup, append+batch-reconcile over per-write mutation, no-LLM-at-query-time retrieval). They did **not** change the architecture — they refined three stages. **Build-vs-borrow verdict: keep building on NetworkX, borrow patterns, vendor none** — NetworkX uniquely gives free PPR (none of the three implement it), every borrowed idea is a ~10–50 line pattern not a subsystem, and mem0 v3 itself *removed* its graph as questionable ROI.

**Adopted (folded into §2/§4/§5/§6/§8 above):**

| Refinement | Stage | Source | Effort |
|---|---|---|---|
| Structured-output extraction (typed `Node[]/Edge[]`, not free-text parse) | extraction §6.4 | cognee | cheap |
| ~~Fixed relation-type enum~~ → **open-vocab relationship tags + canonicalization** (rev 3) | graph model §2 | Galárraga CIKM'14 / CESI WWW'18; multi-label = multiplex networks | medium |
| Directed store (`MultiDiGraph`) + symmetrized traversal projection (rev 3) | graph model §2 | HippoRAG (PPR undirected) / GraphRAG | cheap |
| Provenance back-pointer (tag/edge → source node) | graph model §2 | graphiti | cheap |
| Reflexion self-critique pass after extraction | extraction §6.4 | graphiti | cheap |
| Entropy guard so short/common tags aren't mis-merged | dedup §6.6 | graphiti | cheap |
| `valid`/`superseded_by` soft-invalidation + `created_at`/`last_modified` | graph model §2 | graphiti/mem0 | cheap |
| Per-node/edge `confidence` for trust-filtered retrieval | graph model §2 | cognee | cheap |
| Bounded-concurrency semaphore during ingest | pipeline §6 | graphiti | cheap |
| Selective LLM tie-breaker (ADD/UPDATE/MERGE/NOOP) on ambiguous merges only | dedup §6.6 | mem0 | medium |
| BM25 keyword seed fused with embedding seed | retrieval §5 | graphiti/mem0 v3 | medium |
| MMR + node-distance reranking on PPR scores | retrieval §5 | graphiti | medium |
| PPR as one *comparable* mode, A/B vs BFS (prove it at 200 nodes) | retrieval §5 | all three (none use PPR) | medium |

**Explicitly NOT adopted:** cognee's full RDF/OWL ontology engine and DataPoint "type-hints become edges" magic (schema rigidity, real maintenance — take the confidence flag + *canonicalized* open relations instead); mem0's per-write LLM reconciliation as the default (expensive; mem0 removed it in v3 — selective tie-breaker only); hard-delete on LLM-judged contradiction (always soft-invalidate); graphiti's full bi-temporal quad + LLM date-parsing (unjustified for a stable snapshot — take the soft-invalidation flag, not the quad); a server graph DB (Neo4j/FalkorDB) — loses free PPR; `difflib`-fuzzy in place of embedding synonymy (weaker). Borrow patterns, not dependency weight.