# Scaling "you" — Plan for an Enterprise-Grade MVP

*Written for readers with a CS background but no experience in the AI-memory / retrieval
space. Every specialized term is explained the first time it appears.*

---

## 1. What this project is today

**"you" is a memory system for AI.** You feed it a stream of content — chat logs, notes,
documents, photos — and it builds a **knowledge graph** out of them: a network where the
*nodes* are things (people, places, organizations, concepts) and the *edges* are
relationships between them ("Becky **lives in** Berlin", "Alan Turing **worked at**
Bletchley Park"). Later, you ask questions in plain English and the system finds the right
evidence in the graph and has a language model write an answer with citations.

Three design choices make it different from a plain search index:

1. **Episodes are immutable.** Every piece of content you ingest becomes an "episode" —
   a permanent, never-edited record. New information never overwrites old information.

2. **Facts are *bi-temporal*.** "Bi-temporal" means every fact carries two timelines:
   *when it was true in the world* (Becky lived in Toronto until 2023, then Berlin) and
   *when the system learned it*. When a fact changes, the old fact isn't deleted — its
   validity window is closed and a new fact is opened. This lets you ask **as-of
   questions**: "where did Becky live in 2022?" gets a different (correct) answer than
   "where does Becky live?". It also gives you an audit trail: every answer can cite the
   exact source episode it came from. Most competing systems cannot do this.

3. **Retrieval is graph-first, not LLM-first.** "Retrieval" = finding the relevant
   evidence before answering. Many systems make repeated calls to a large language model
   (LLM) to "walk" through data step by step — slow and expensive. Here, a classic graph
   algorithm (described in §4) does the multi-hop work in milliseconds with **zero LLM
   calls**, and exactly **one** LLM call at the end writes the answer.

### The pipeline in one diagram

```
INGESTION (write path)                          QUERY (read path)
─────────────────────                           ─────────────────
content in                                      question in
   │                                               │
   ▼                                               ▼
extract entities/relationships                  find "seed" nodes matching the question
(LLM and/or local models)                          │
   │                                               ▼
   ▼                                            spread out from the seeds through the
merge duplicates                                graph to score every episode
("NYC" = "New York City")                       (PageRank — no LLM involved)
   │                                               │
   ▼                                               ▼
write episode + facts into the graph            re-rank the top results, build a context
with time windows                               package of episodes + currently-valid facts
                                                   │
                                                   ▼
                                                ONE LLM call writes the answer,
                                                citing episode IDs
```

---

## 2. Why we're confident about where the time and money go

We built a **profiler** into the test harness — instrumentation that times every stage of
the pipeline and attributes every dollar of LLM spend to the call site that incurred it.
Every benchmark run now produces a dashboard showing time-by-stage and cost-by-stage.

Real numbers from a recent benchmark run (18 chat sessions, 3 questions):

| Stage | Time | Cost | Share of total cost |
|---|---|---|---|
| Entity extraction — local models | 348s (CPU) | $0 in API fees (but CPU isn't free) | — |
| Entity extraction — LLM calls | 174s | $0.016 | ~90% |
| Embedding (see §4) | 10s | $0 (runs locally) | — |
| Graph writes | 0.4s | $0 | — |
| Answering a query (everything) | ~2–5s each | $0.0005/query | ~8% |

Two conclusions that drive this whole plan:

- **Ingestion — the write path — is where all the money goes** (~90%+). Whoever ingests
  cheaper wins on cost.
- **The query path is already fast at small scale** but contains a hidden time bomb (§4).

---

## 3. The market, and the gap we can own

The "memory for AI agents" space has real competitors. What a new entrant needs is a
**differential**: something measurably better that incumbents can't trivially copy.

| Competitor | What they do | Weakness |
|---|---|---|
| **Zep (Graphiti)** | Temporal knowledge graph memory, sold as a cloud API | Write path makes **3–6 LLM calls per message** ingested — slow and expensive at scale |
| **Mem0** | Extracts "memories" from conversations, sold as an API | Shallow time-reasoning; not really a graph |
| **Letta (MemGPT)** | An "operating system" for agent memory | Different layer of the stack; not a graph store |
| **Academic systems (HippoRAG)** | Graph-based retrieval research — our retrieval is in this family | Research code, not products |

Note what is **not** a differential: "we have a temporal knowledge graph." Zep has one
too. The claims nobody in the market can currently make, and that this codebase is
positioned to make:

1. **An order of magnitude cheaper ingestion.** Our design only escalates to an LLM for
   entries that contain "cues" (dates, endings, identity statements — the things cheap
   models get wrong). Competitors pay for multiple LLM calls on *every* message.
   ⚠️ *This advantage only survives if the planned extractor changes keep some form of
   gating or batching — replacing the cheap path with a per-entry LLM call would erase
   the exact cost edge we'd be pitching.*

2. **Sub-100ms retrieval at millions of episodes, with no LLM in the loop.** Requires
   the engineering in §4 — that's the core of this MVP.

3. **Self-hostable.** Enterprises with sensitive data (legal, health, finance) often
   cannot ship it to a third-party API. Zep and Mem0 are primarily hosted APIs. A version
   a customer can run inside their own network is a real enterprise selling point.

4. **Audit-grade answers.** Bi-temporal facts + citations = "what did we believe on
   March 3rd, and based on which document?" Compliance-heavy industries pay for this.

**The pitch artifact is a benchmark chart**: accuracy, ingestion cost per 1,000 episodes,
ingestion throughput, and query latency at the 50th/99th percentile — our numbers next to
Zep's and Mem0's, at one million episodes. We already own the harness that generates it.

---

## 4. What breaks at scale, in the order it breaks

Measured density: ~150 graph nodes per chat session ingested. So one enterprise customer
with 1M documents ≈ a **150M-node graph**. Today's implementation hits four walls before
that, and we know exactly where because the code documents its own placeholders.

### Wall 1 — the retrieval algorithm is pure Python *(hits first, ~100k nodes)*

Retrieval uses **Personalized PageRank (PPR)** — the same idea as Google's original
PageRank. Ordinary PageRank asks "which nodes are important overall?"; the *personalized*
variant asks "which nodes are important *relative to these starting points*?" We start it
from the nodes matching the question, let the score "diffuse" through the network, and
the episodes that accumulate the most score are the most relevant evidence — even ones
several relationship-hops away from the question's keywords. This is how the system
answers multi-hop questions without any LLM-driven searching.

Today PPR runs on **NetworkX**, a Python library where every node and edge is a Python
object. It's wonderful for prototyping and hopeless at scale: on our small test graphs
PPR takes 3 milliseconds; the math is O(edges × iterations) in *interpreted Python*, so
at millions of edges that becomes seconds-to-minutes per query.

**Fix:** represent the graph as a **sparse matrix** — a compressed numerical format
(CSR: "compressed sparse row") storing only the edges that exist — and run PageRank as
matrix-vector multiplication in `scipy` (compiled C, not interpreted Python). Same math,
same answers, typically 10–100× faster, and the matrix is cached and reused across
queries. This is the single highest-leverage change in the plan.

### Wall 2 — the whole graph lives in Python memory *(~1M nodes, a few GB)*

Every node/edge being a Python object costs hundreds of bytes of overhead each. The
codebase already names its escape hatch (an embedded graph database); the near-term MVP
version of the fix is the same CSR representation from Wall 1 plus the existing SQLite
storage.

### Wall 3 — vector search checks every vector *(~1M vectors)*

An **embedding** is a list of numbers (a vector) representing a text's meaning; two texts
with similar meaning get nearby vectors. We embed every episode, and "find episodes
similar to this question" = "find nearby vectors." Today that comparison checks **every
stored vector one by one** (brute force). Fine below ~1M vectors; past that we swap in an
**approximate nearest-neighbor index** (e.g. FAISS, sqlite-vec) — a data structure that
finds near-matches without exhaustive scanning, the standard tool for this.

### Wall 4 — ingestion is single-machine *(throughput, not correctness)*

Measured: ~8 seconds of wall-clock per chat session. At 1M documents that's ~3 months on
one machine. The good news is architectural: extraction (the expensive part) is
**stateless** — each document is processed independently, so it parallelizes across as
many worker machines as we want. All *graph mutation* is already serialized per graph, by
design, so correctness doesn't change. The MVP wraps the existing code in a work queue.

---

## 5. The two-tier product, one codebase

The strategy: a **free consumer tier** and an **optimized self-hosted enterprise tier** —
built as *configurations of the same pipeline*, never as a fork.

This works because the codebase already defines swap points ("interfaces" /
"protocols") for its components:

| Component | Consumer (exists today) | Enterprise (this MVP) |
|---|---|---|
| Extraction | cheap local path, LLM only on cue-bearing entries | same gating + batched LLM calls (bulk-discounted) |
| Embeddings | small local model, free, private | GPU-batched for throughput |
| Graph storage | NetworkX + SQLite (fine to ~50k episodes/user) | sparse-matrix (CSR) representation |
| Retrieval (PPR) | NetworkX PageRank | scipy sparse PageRank |
| Vector search | brute force | approximate-nearest-neighbor index |
| Ingestion | in-process | queue + parallel workers |

Two seams don't exist yet and must be created (that's part of the work): storage and
retrieval currently call NetworkX directly. We define a `StoreBackend` interface so the
two tiers differ only in which backend a config file names.

**Why "one codebase" is a strategy and not a slogan:** every improvement ships to both
tiers; the consumer tier doubles as the enterprise tier's correctness test (same test
suite, same benchmark harness, two configs, diff the dashboards); and there is no
"community edition rots while cloud edition advances" trap.

**Multi-tenancy note** ("multi-tenant" = many customers on shared infrastructure): each
user/tenant gets their **own isolated graph** — which is already exactly how the
benchmark protocol works. Tenants never share memory, which sidesteps the hardest
enterprise problems (per-fact access control, cross-user identity merging) for now.
Those are deliberately **out of scope** for the MVP — see §7.

---

## 6. The MVP, concretely

Five workstreams, roughly in dependency order. Rough effort assumes one experienced
engineer.

| # | Workstream | What it involves | Effort |
|---|---|---|---|
| 1 | **Sparse PPR** | CSR matrix build + cached power-iteration PageRank; NetworkX version kept as the reference implementation the tests compare against | ~3–5 days |
| 2 | **Vector index** | Swap brute-force cosine for sqlite-vec or FAISS behind the existing `VectorIndex` interface | ~1–2 days |
| 3 | **StoreBackend seam** | Define the storage interface; NetworkX/SQLite and CSR-native become configs | ~3–5 days |
| 4 | **Parallel ingestion** | Work queue around the existing stateless extraction; N workers feeding one sequential graph writer per tenant | ~3–5 days |
| 5 | **The 1M-episode benchmark** | Generate/load a 1M-episode corpus, run the existing harness, produce the dashboard: ingestion $/1k episodes, episodes/sec, query p50/p99 latency, accuracy — beside Zep/Mem0 published numbers | ~1 week |

("p50/p99" = the 50th and 99th percentile of query latency. Enterprises contract on p99 —
"even your slowest 1% of queries return within X ms" — so we measure it from day one.)

Workstream 5 is the point of the whole exercise: **the fundraising artifact is the
benchmark dashboard**, and every other workstream exists to make its numbers good.

---

## 7. Explicitly out of scope (and why that's fine)

- **Distributed graph storage** — per-tenant sharding covers the demo and early
  customers; one tenant's graph fitting on one machine holds far past 1M episodes once
  CSR replaces Python objects.
- **Per-fact access control (ACLs)** — only matters for *org-shared* graphs; per-tenant
  isolation defers it.
- **Cross-user entity resolution** — "is Alice's 'Bob' the same person as Carol's
  'Bob'?" Hard, and unnecessary while graphs are isolated.
- **Rewriting extraction quality** — the extractor is being iterated separately; this
  plan only cares that its *cost structure* (gating/batching) preserves the write-path
  cost advantage.

---

## 8. Risks, honestly

1. **The cost differential is self-inflicted-fragile.** Our cheapest-ingestion claim
   depends on not paying an LLM for every entry. The planned extractor changes must be
   benchmarked with the profiler *before* being locked in.
2. **Competitors can lower prices.** The durable moats are the self-hosted deployment
   and the audit/as-of story; the cost edge is a wedge, not a wall.
3. **Benchmark credibility.** Comparing against Zep/Mem0 must use their published
   numbers or reproducible public setups, or the chart invites easy attack.
4. **Accuracy at scale is unproven.** Retrieval quality on a 3-instance test says little
   about a 1M-episode graph; workstream 5 measures accuracy, not just speed, for exactly
   this reason.

---

## Glossary (one-liners)

- **LLM** — large language model (GPT, Claude…); costs money per token processed.
- **Token** — the unit LLMs bill by; roughly ¾ of a word.
- **Knowledge graph** — data stored as things (nodes) + relationships (edges).
- **Episode** — one immutable ingested item (a chat session, a note, a photo).
- **Entity extraction** — pulling "who/what/where" out of raw text.
- **Bi-temporal** — facts track both when they were true and when we learned them.
- **As-of query** — "answer as of date T," using only facts valid at T.
- **Retrieval / RAG** — finding relevant evidence first, then having an LLM answer from
  it ("retrieval-augmented generation").
- **Embedding / vector** — a numeric representation of meaning; similar meaning ⇒ nearby
  vectors.
- **PPR (Personalized PageRank)** — PageRank started from chosen seed nodes; scores the
  graph *relative to the question*.
- **CSR / sparse matrix** — compressed numeric storage for graphs that makes the math
  fast (compiled code instead of Python objects).
- **ANN index** — approximate-nearest-neighbor structure; fast similarity search without
  scanning everything.
- **Multi-tenant** — many customers on shared infrastructure, isolated from each other.
- **p50 / p99** — median / 99th-percentile latency; enterprises contract on p99.
- **Ingestion / write path** — everything that happens when content enters the system.
- **Query / read path** — everything that happens when a question is asked.
