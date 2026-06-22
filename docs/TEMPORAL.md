# Temporal, Evolving Nodes & Relationships — Design Plan

> **Status: PLANNING (not built).** The current corpus is a *static* Wikipedia snapshot
> that doesn't exercise any of this. Build it when you move to evolving data (agent
> memory, a live feed about real people/entities). This doc is the blueprint for when
> that happens. No code here — design + phasing only.
>
> Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (§2 graph model, §6 ingestion). References
> live code by name: `Edge`, `RelationTagNode`, `supersede_node`, `resolve_relation`,
> `canonicalize.py`, `ingest.py`, the rev-4 parallel-edge model.

---

## 0. The one core idea

**Everything that can change about an entity becomes a *time-bounded edge*. A node's
identity stays fixed forever; what evolves is the set of edges currently valid around it,
plus a derived snapshot computed from them.** "Former coworker" is not a new relationship
— it's a `works_with` edge whose validity *ended*.

A person's state at any time = **the entity node + every edge whose valid-window contains
that time.** "Current" = edges with `valid_to = ∞`. Evolution is edges opening and closing.

---

## 1. What we have today, and the three gaps

We are currently **uni-temporal**: nodes/edges carry `created_at`, `last_modified`, a
`valid` flag, and `superseded_by` — but no notion of *when a fact was true in the world*.

Concrete gaps (why coworker→ex-coworker breaks today):
1. **Relations are append-only across documents.** A new doc never retracts an old edge;
   it only adds. "Alice works_with Bob" persists forever, even after she leaves.
2. **`supersede_node` is too coarse and object-scoped.** It only fires on re-ingest of the
   *same object id*, and it invalidates **all** of a node's edges at once — it cannot
   express "only this one relationship ended."
3. **`coworker` and `ex_coworker` canonicalize to different relation tags** (different
   content words), so they become two parallel A→B edges, **both marked valid** — a
   permanent, unresolved contradiction.

**Good news:** the rev-4 parallel-edge model (one `RELATED_TO` edge per canonical relation)
is the *right substrate* for this. Each relationship is already its own edge, so we can give
each its own valid-window and expire `works_with` without touching `mentored` or `friend_of`
between the same pair.

---

## 2. Node evolution model (does Alice evolve? — yes)

An **entity node (Alice) has a stable identity for its entire life** — one node, never
overwritten, never re-versioned. Two distinct things evolve around it:

| What evolves | Mechanism |
|---|---|
| **Knowledge growth** — we learn *more* about Alice (new job, birthday, new friend) | New attribute/relationship **edges accrete** onto the stable node |
| **State change** — the world changes (Paris→London; coworker→ex-coworker) | Old edges **close** (`valid_to` set); new edges open |
| **Refinement** — better name form / alias discovered | Existing canonicalization (alias onto the stable node) |
| **Correction** — a recorded fact was wrong | **Transaction-time** retraction (see §4) |

**Stable parts of a node** (never mutated): `id`, `type`, canonical name + aliases,
provenance pointers.
**Evolving parts**: the surrounding edge set, and a **derived "snapshot"** of current
state.

> **The snapshot is *derived*, not a mutable blob.** Alice's one-line description /
> current-state summary is computed on demand from her currently-valid edges — *not* a
> field we hand-edit on each update. (Mutating one summary blob over many updates causes
> "telephone-game" drift; graphiti hit this.) Optionally **cache/materialize dated
> snapshots** ("Alice as of 2018", "Alice now") for performance and time-travel, but the
> edges remain the source of truth so any snapshot can be recomputed.

**Entity nodes vs object nodes:** entity nodes (Alice) get stable identity + edge
evolution as above. **Object/document nodes still version on re-ingest** (`superseded_by`,
as today) — that's a separate, unchanged mechanism.

---

## 3. Two kinds of change — both needed (the "as we gain more info" crux)

| Reason a fact changes | Time axis | Action |
|---|---|---|
| **The world changed** — they really stopped being coworkers in 2020 | **valid-time** (`valid_from`/`valid_to`) | Close the old edge's window; keep its history |
| **Our knowledge improved** — we recorded "married" from a weak source; a better source says "never married" | **transaction-time** (`created_at` / belief state) | Mark the recorded belief retracted; it was never *valid* |

Bi-temporality lets us answer both *"true 2015–2020"* and *"we believed this until we
corrected it on date Y."* We already have transaction-time; this plan adds valid-time and
the logic to choose **close** vs **correct**.

---

## 4. Data model changes

- **Edges gain a valid-window:** `valid_from`, `valid_to` (`valid_to = ∞` ⇒ currently
  true). Support **unknown/open bounds** (we may learn the end before the start).
- **Edges gain a belief state:** `asserted` vs `retracted`, alongside the existing
  `confidence`, so a low-confidence rumor can't hard-kill a high-confidence fact.
- **Keep the provenance pointer** on every edge (already present) — it's what lets us
  reconstruct "what we knew when" and recompute a snapshot after a correction.
- **Model evolving node attributes AS edges**, not mutable fields: job → `employed_by`,
  role → `has_role`, location → `lives_in`. Then attribute-evolution and
  relationship-evolution are the *same* mechanism (time-bounded edges).
- **Entity nodes are never re-versioned;** identity is the anchor.

---

## 5. Per-predicate cardinality & symmetry flags (prerequisite)

Add to each `RelationTagNode` (pairs with the per-predicate symmetry flag already on the
roadmap):
- **Functional / single-valued** (`lives_in`, `ceo_of`, `spouse_of`): a new value
  **supersedes** the old (you can't live in two cities) → auto-close the prior edge.
- **Multi-valued** (`works_with`, `friend_of`): many coexist → never auto-supersede.
- **Symmetric** (`works_with`, `married_to`, `sibling_of`): store once; A→B and B→A are the
  same fact (today nothing reconciles them).

Without cardinality we can't tell "moved to London" (supersede Paris) from "also works with
Carol" (coexists with Bob).

---

## 6. Extraction changes (the prompt/schema gains a temporal dimension)

Per relationship, the extractor should emit:
- **Polarity** — is the source asserting the relationship *holds*, or that it *ended /
  never held*?
- **Optional time bounds** — "until 2020", "since 2015", "from 2010 to 2014".
- **Termination detection** — "former," "ex-," "no longer," "left," "until X" → emit the
  **base relationship with status = ended** (and an end date if stated), *not* a new
  `ex_*` predicate.

Stay conservative: only emit bounds the text actually states; everything else defaults to
"as of this document." (LLM date-parsing is fragile — see §11.)

---

## 7. Canonicalization rule (new, deterministic)

In `resolve_relation`: normalize tense/aspect wrappers — `former_X` / `ex_X` /
`used_to_X` / `no_longer_X` → **base predicate `X` + ended status.** This prevents
predicate sprawl into tense. Antonyms (`friend`/`enemy`) stay distinct exactly as today —
only the *temporal wrappers* get folded onto the base predicate.

---

## 8. Ingest decision logic — the phase that actually fixes coworker→ex-coworker

For each incoming relationship fact, choose exactly one action:

| Action | When | Effect |
|---|---|---|
| **Open** | relationship newly asserted | new edge, `valid_from` = stated/ingest, `valid_to` = ∞ |
| **Confirm / extend** | same fact seen again | bump confidence; maybe extend `valid_to` |
| **Close** | termination signal or explicit end date | set the existing edge's `valid_to`; the edge stays in the graph, just not "current" |
| **Supersede** | functional predicate, new value | close the old, open the new (Paris→London) |
| **Correct** | better source contradicts a *recorded* fact | retract the old belief (transaction-time); no valid-window |

**The open-world rule (critical):** *absence of mention never closes an edge.* A new doc
that doesn't mention an old coworker must **not** end that relationship — closure requires
**positive evidence**. (This is the temporal analog of the link-biased under-merge stance:
prefer keeping a fact over silently dropping it.)

**Order independence:** because resolution is by valid-time, not ingest order, learning
"ex-coworker (ended 2020)" *before* "worked together 2015–2020" still resolves correctly —
the bounds merge into one edge `[2015 … 2020]`. Unknown starts stay open (`valid_from =
unknown`), never fabricated.

---

## 9. Query / retrieval

- **Default = current view:** PPR/BFS run only over edges valid *now* (extend the existing
  `valid=false` filter to `valid_to = ∞`; the symmetrized traversal projection inherits it).
- **As-of T mode:** keep only edges whose window contains T, then retrieve normally → "who
  did Alice work with in 2018?"
- **Relationship history:** all parallel edges between a pair with their windows → the
  timeline of Alice↔Bob.
- **Person snapshot:** gather the entity + its currently-valid (or as-of-T) edges into a
  view; render/cache the derived summary from them.

---

## 10. Worked example — the Alice timeline

1. Doc A (2016): "Alice works at Acme with Bob." → open `Alice —works_with→ Bob`
   `[valid_from≈2016, ∞]`; open `Alice —employed_by→ Acme [2016, ∞]`.
2. Doc B (2021): "Alice, formerly of Acme, now at Globex." → `employed_by` is **functional**
   → **supersede**: close `employed_by Acme` at 2021, open `employed_by Globex [2021, ∞]`.
3. Doc C (2021): "Alice and Bob are former colleagues." → termination signal → **close**
   `works_with Bob` at 2021. (No new `ex_coworker` node.)
4. Query "are Alice and Bob coworkers?" → no *currently-valid* `works_with` edge → "they
   used to be (until 2021)."
5. Query "Alice as of 2018" → snapshot view: `works_with Bob`, `employed_by Acme` (both
   valid in 2018).
6. Query "Alice now" → `employed_by Globex`; Bob relationship shown as past.

Throughout, **one stable Alice node**; only her edge-set and derived snapshot change.

---

## 11. Guardrails / risks

- **Date-parsing fragility** (graphiti's documented failure): mis-parsed bounds silently
  corrupt the timeline. Default to ingest-time validity; trust explicit dates only when
  high-confidence.
- **Confidence-gated closure:** closing should require evidence at least as strong as what
  opened the fact — don't let a low-confidence rumor retract a strong fact.
- **Open-world, not closed-world:** never infer "ended" from silence.
- **Never fabricate a start** for an "ended" fact you learned end-first: store
  `valid_from = unknown`.
- **Storage growth:** history accumulates → need archival/compaction of long-closed edges.
- **Derived-snapshot freshness:** when an entity's edges change materially, its cached
  snapshot (and the snapshot's embedding, if used for retrieval) needs incremental refresh.
- **Don't over-build:** a static snapshot corpus needs none of this. Gate the work on having
  genuinely evolving data.

---

## 12. Phasing (smallest useful increments first)

1. **Valid-time on edges** + current-view query filter. *(Non-breaking; everything defaults
   to `[ingest, ∞]`.)*
2. **Per-predicate cardinality / functional / symmetry flags.**
3. **Extraction emits polarity + termination + optional dates**, and the `former_/ex_` →
   base-predicate + ended normalization rule.
4. **Ingest decision logic** (open / confirm / **close** / supersede / correct) + the
   open-world rule. ← *fixes coworker→ex-coworker.*
5. **As-of-T + relationship-history queries.**
6. **Person-snapshot view** (derive current state from valid edges; optional dated cache).
7. **Belief-revision (transaction-time corrections)** + **compaction/archival.**

Phases 1–4 deliver clean coworker→ex-coworker (resolved correctly regardless of document
order). Phases 5–6 deliver the time-traveling, evolving person snapshot. Phase 7 handles
"we got better information and must rewrite what we believed."

---

## 13. Eval (extend the `eval-canon` discipline to time)

Build a temporal test set with labeled expected outcomes before trusting any of this:
- coworker → ex-coworker resolves to **one** relationship with a closed window (not two
  live edges, not an `ex_` predicate).
- **Silence preserves** an existing edge (open-world).
- **End-first then start** still merges into one window (order independence).
- **Functional supersession** (Paris→London) closes the old, opens the new.
- **As-of-2018 ≠ as-of-now** for an entity whose state changed between.
- A **correction** retracts a belief without leaving a phantom valid-window.
