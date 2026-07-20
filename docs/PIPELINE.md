# PIPELINE.md — end-to-end reference

Personal quick-reference for the full kg pipeline, with the load-bearing details inline.
(Deeper rationale: ARCHITECTURE.md, TEMPORAL.md, CONFIG.md.)

```
CorpusItem → hash/dedup → chunk → extract (LLM) → embed → canonicalize → apply_fact → derived edges
                                                                              ↓
question → route (4 lanes) → seed (BM25+emb) → PPR over projection → MMR/CE rerank
        → context builder (episodes + facts + history) → ONE answer LLM call
```

---

## 1. Intake (`kg/ingest.py`)

- **Dedup/versioning**: SHA256 of `(modality, content, created_at, id)`. Same hash → skip.
  Same id, new content → append-only version `ep_X_v1` (old episode kept — both are history).
- **Chunking** (`chunkers.py`, default `turns`): big text → chunk episodes `<id>#cNNN` +
  a SOURCE parent node with `PART_OF` (w=0.3) and `NEXT` (w=0.5) edges.
- **Extraction fans out** concurrently (semaphore 40); all graph writes are sequential.
  Failed extractions are dropped *without* recording the hash, so retry reprocesses.

## 2. Extraction (`kg/extractors.py`)

One `emit_graph` tool call per entry returns:
- **entities** (salience-filtered), **tags/concepts**, **facts[]** (every stated
  amount/count — exempt from salience filter), and **relations**:
  `source, target, labels[≤3], status, valid_from, valid_to, confidence`.
- `status`: `asserted` | `ended` ("former", "no longer") | `retracted` ("never true").
- `valid_from`/`valid_to`: **only if the text states a date; never guessed.** Empty
  defaults to "as of this content". ⚠ The extractor is *not* given the episode date, so
  relative dates ("last March") stay unresolved.
- **First person**: only when `config.self_entity` is on, the narrator is extracted as
  entity `me` → the single self anchor. **Off (the default) = first-person relations are
  dropped entirely.** The Engine facade (app/daemon path) forces it ON (`engine.py:187`).
- Keyless floor: cue-gated local NLP (`nlp_extractors.py`); `reflexion` adds one recall pass.

## 3. Embedding (`kg/ingest.py` §3)

Embeddings go **only on immutable nodes** — episode text and mention surfaces. Canonical
entity anchors are never embedded, so nothing is ever re-embedded on rename/merge.

## 4. Canonicalization (`kg/canonicalize.py`)

Link-biased (prefer under-merge):
- **L1** exact/normalized key match.
- **L2** embedding gate: cosine > link τ (.85 ent / .80 tag) → `SIMILAR_TO` link;
  > merge τ (.93 ent / .88 tag) → hard merge. Entropy guard blocks fuzzy merges of
  short/low-entropy strings ("AI").
- **L3** LLM tie-breaker: shipped disabled.
- **Relations**: `relation_content_key` strips interior function words + stems tense
  (`lives_in/lived_in → liv_in`) but keeps trailing markers (`works_at ≠ works_with`,
  `managed_by ≠ manages`). Cardinality is **lexicon-based, no LLM**:
  `predicate_cardinality()` → `(functional, symmetric)` stamped on the RelationNode.
- `doc_frequency` per node feeds IDF weighting at seed time.

## 5. Facts: bi-temporal storage (`kg/temporal.py`, `kg/store.py`)

A fact is a `RELATED_TO` edge: `src --rel_tag--> dst` carrying
`valid_at`, `invalid_at`, `belief`, `confidence` (=`weight`), `episode_id`, `created_at`,
plus audit fields (`closed_by_episode`, `confirmed_by[]`, `disputed_by[]`, `retracted_at`).

- **valid time** = `valid_at → invalid_at`; `invalid_at == ""` means **open (∞)**.
- **transaction time** = `created_at` / `retracted_at`; `belief` ∈ asserted|retracted.
- `apply_fact(status, at, valid_from, valid_to)` — `at` is the episode event time.
  For `asserted`: `start = valid_from or at`. Exactly one action fires:

| action | trigger | effect |
|---|---|---|
| open | brand-new fact | new edge `[start, ∞]` |
| confirm | same open (src,dst,rel) again | max() confidence, widen `valid_at` to earliest, append `confirmed_by` |
| close | `status=ended` | set `invalid_at = valid_to or at` on open edges |
| supersede | functional predicate, different dst | close old value at new start, open new |
| backfill | start arrives after an end-first edge | fill unknown `valid_at` |
| retract | `status=retracted` | flip `belief` — leaves *every* view |
| dispute | overturning claim > 0.3 below stored confidence | recorded in `disputed_by`, edge unchanged |

- **Repeatable** (= not functional, not symmetric): a re-assertion with a *different
  explicit date* opens a new occurrence instead of confirm-collapsing.
- **Open-world rule**: absence of mention never closes an edge.
- Symmetric facts stored once, pinned `src < dst`. functional+symmetric (spouse_of)
  supersede scans both directions of both endpoints.
- ⚠ **Events** ("went to the park") have no representation of their own: they're stored
  as open states `[event_date, ∞)` and never close.

## 6. Derived edges (post-ingest, incremental)

Episode↔episode `SIMILAR_TO` (kNN k=6, cosine ≥ .55), `SHARED_TAG`/`SHARED_ENTITY`
shortcuts, communities (Leiden-ish, for global/theme questions only).

---

## 7. Temporal views: `fact_active(data, as_of)` (`store.py:43`)

The single gate used by context building AND the PPR projection:

- `as_of=None` (**default for every question**) → "current view": active iff
  `belief=asserted` **and** `invalid_at == ""` (open). Closed facts drop out.
- `as_of=T` → active iff `valid_at <= T < invalid_at` (half-open; ISO strings compare
  lexically, bare years work).
- **Nothing extracts `as_of` from question text** — it's caller-supplied only (eval
  datasets pass it; the app generally doesn't). `resolve_relative_window` only parses
  "last week"-style phrases into an *episode* boost window, never a fact filter.

## 8. Query routing (`kg/route.py`) — regex, $0, text-only

| lane | example | effect |
|---|---|---|
| RECENCY | "yesterday", "lately" | recency emphasis |
| STATE | "now", "in 2022", evolution, date arithmetic | fact-lane augment + HISTORY block |
| MULTIHOP | "who did I talk to about…", "how many/much" (aggregation) | wider PPR pool |
| SINGLE | everything else | plain hybrid |

Misroutes degrade gracefully — same machinery answers every lane.

## 9. Retrieval (`kg/retrieval.py`)

- **Seeding**: BM25 over composite episode docs (raw text + title + description +
  entity/tag surfaces + media tokens) fused with embedding sim → top `seed_k=10` nodes.
  Seed mass is **IDF-weighted** (`canon.idf_weight`).
- **Projection** (cached per store version): undirected; excludes `IN_COMMUNITY`;
  **RELATED_TO edges are filtered by `fact_active(as_of)` before diffusion**; edge weight
  `= conf × weight`, **MAX within an etype per node-pair** ("parallel facts count once" —
  fact-edge frequency is deliberately flattened), SUM across etypes.
  Frequency still leaks in via topology: N visit episodes = N parallel
  `me ↔ episode_i ↔ park` mention corridors.
- **Self-hub guard** (`self_guard`: none|exclude|cap|seed): throttles edges incident to
  the self anchor. ⚠ Default `none`, and the Engine doesn't set it — app runs unguarded.
- **PPR** (α=.5, exact or local-push) → episode candidates → distance-to-seed boost →
  **MMR** (λ=1.0 = pure relevance) → **cross-encoder rerank on hard lanes only**
  (`rerank_lanes=(state, multihop)`; CE demotes gold on easy lookups), raw-PPR top-3
  always kept. Optional: session dedup, seed-reserve slots, date-window slots.

## 10. Context building (`kg/rag.py` ContextBuilder.build)

The blob the answer LLM reads, in order:
1. `QUESTION:` + `AS-OF:` (or "now (current view)").
2. **EPISODES** — top `rag_context_episodes=5` (chunk retargeting, sibling expansion,
   provenance promotion pulls a fact's source chunk in; `since/until` window is a hard
   bound on episodes only, **never on facts**). Each shows event date + delta vs as_of.
3. **FACTS "currently valid"** — `facts_for(entities, as_of)`: walks anchor entities'
   RELATED_TO edges both directions, filtered by `fact_active`.
   ⚠ Truncated at `rag_max_facts=30` in **arbitrary order** (hub-entity lottery).
   Rendered `src --rel--> dst (since D | from D until D | mentioned D) [episode]`.
   ⚠ Anything with `invalid_at` renders/reports as `status: ended`.
4. **HISTORY** (STATE lane only, and only if ended history exists) — full closed+open
   trajectory from `FactIndex.history()`.

Anchor entities = PPR subgraph nodes + entities mentioned by the top episodes.

## 11. Answering (`kg/rag.py`)

One LLM call, forced `submit_answer` tool: `{answer, citations[], events[]}` — events
enumeration is on for state/multihop lanes (`rag_answer_events="lanes"`). The LLM never
traverses the graph. Offline fallback: extractive answer from the same context.

---

## Known sharp edges (as of 2026-07)

1. **Events stored as `[date, ∞)` states** — temporally wrong; survives retrieval only
   *because* it never closes. A closed event (`[d,d]`) would vanish from every view
   (current view needs open; as-of is half-open so zero-width matches nothing).
   **ADDRESSED 2026-07-16** (docs/OFFLINE_EVAL.md Round 2), behind default-off knobs:
   `event_facts` writes lexicon/bounded event predicates as closed `[d,d]` occurrence
   edges (`event=True` on RelationNode + edge; confirm-on-closed dedup) and
   `history_all_lanes` serves the closed-fact delta on every lane so they stay findable
   (`fact_active` deliberately untouched). Rendered `(on d)` / `(d1 -> d2)`, status
   `occurred`. The two knobs must ship AS A PAIR; old `[d, ∞)` edges coexist unmigrated.
2. **Engine forces `self_entity=True` but leaves `self_guard="none"`** — unthrottled
   "me" super-hub in the app path.
3. **`facts_for` truncation is unranked** — rank fact lines by query similarity.
4. **Extractor lacks a reference date** — relative dates can't become `valid_from`.
5. **`as_of` never derived from question text** — "in 2022" is matched by the STATE
   regex, then discarded.
6. **Abstract/disposition questions** ("what do I like to do?") have no lane and no
   seeding path onto concrete evidence; distilled frequency lines
   (`went_to → park, 5×, Mar–Jul`) embedded as retrieval targets is the structural fix.
7. `disputed_by` is stored but never surfaced in context.
