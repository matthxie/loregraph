# Spike: can a local encoder (GLiNER-Relex) replace or pre-filter the gpt-4o-mini extractor?

**Date:** 2026-07-04 · **Author:** research spike (timeboxed) · **Verdict up front:
(a) gate / pre-filter — yes; full replacement — no.** Entities: near-complete recall of
what the LLM extracts, drowning in over-extraction. Relations: structurally incapable of
matching the LLM's open vocabulary (≤42 % pair coverage even with an oracle label set).
Temporal fields (status=ended, valid_from/to): categorically absent from the local model.

---

## 1. Setup

| | Local | LLM (ground truth) |
|---|---|---|
| Model | `knowledgator/gliner-relex-large-v1.0` (joint zero-shot NER + RE, arXiv 2605.10108) | `OpenAIExtractor` — gpt-4o-mini, reflexion ON, sectioned at 6 000 chars (exactly the production ingest path) |
| Hardware | RTX 3060 Ti (CUDA), fp32 | OpenAI API |
| Labels | Entity: the 10 natural-word labels from `kg/nlp_extractors.GLINER_LABELS`, remapped to the 8 `EntityType`s. Relations: the 30-predicate `GLINER2_REL_SCHEMA` keys | Open vocabulary (prompt-defined) |
| Chunking | paragraph packs ≤ 140 words (DeBERTa 512-token window) | 6 000-char sections |

**No new packages were installed.** The already-pinned `gliner 0.2.27` loads the Relex
checkpoint directly (`UniEncoderSpanRelexGLiNER`, `predict_relations()` API); torch /
transformers / sentence-transformers pins untouched.

**Data:** 20 sessions from the LongMemEval `sample` tier (every k-th of 48, spanning 8
instances; 2.5 k–21 k chars, real chat text). Ground truth cost **$0.045** (112 calls,
209 k in / 23 k out tokens). Note: the org's 10 k requests/day budget was exhausted
during the spike, so the LLM pass had to be paced at 1 call / 9.5 s — LLM wall-times
below are nominal, not today's measurements.

Artifacts: `sessions.json` (inputs), `llm_extractions.json` + `llm/*.json` (ground
truth), `local_extractions.json` / `local_oracle.json` / `local_hithr.json` (local
runs), `scores*.json`, `side_by_side*.txt` (per-session diff for eyeballing),
`run_llm.py` / `run_local.py` / `score.py`.

## 2. Numbers

Micro-averaged over 20 sessions. "Recall" = share of LLM entities the local model also
found (exact = normalized-name match; +fuzzy adds containment / difflib ≥ 0.85 alias
near-misses). "Precision" = share of local entities with any LLM counterpart. Relation
coverage = share of LLM (source, target) pairs connected by *any* local edge, either
direction, any label, fuzzy endpoints. Narrator aliases (`User` ≡ `me` ≡ `I`) unified;
chat-role pseudo-entities excluded from entity scoring.

| Variant | Ent recall (exact) | Ent recall (+fuzzy) | Ent precision | Rel pair coverage | local ents | local rels |
|---|---|---|---|---|---|---|
| **Relex, thr .45, 30-pred schema** | **0.856** | **0.935** | **0.225** | **0.290** | 2 822 | 1 728 |
| Relex, oracle rel-vocab (top-60 labels the LLM actually used) | 0.859 | 0.954 | 0.161 | 0.418 | 4 258 | 2 909 |
| Relex, high thresholds (ent .7, rel .6) | 0.796 | 0.877 | 0.272 | 0.188 | 2 025 | 768 |

LLM totals on the same sessions: 432 entities, 517 relations.

### Entities — recall excellent, precision hopeless
The local model finds **86 % of LLM entities exactly, 94 % with fuzzy aliasing** — on
12/20 sessions it missed *zero*. What it misses is categorical, not random:
money amounts and ranges (`$60,000`, `$250,000 to $300,000`, `$297` — 14 of the ~40
misses), niche medical/abstract concepts (`carpal tunnel syndrome`, `confirmation bias`,
`astrological fatalism`), and era terms (`Jomon period`). All are outside the
natural-word label set; a `medical condition` / `amount of money` label would recover
most (untested — timebox).

The cost: it emits **6.4× more entities than the LLM** (2 822 vs 432), so only 22.5 %
of local entities have an LLM counterpart. The extras are generic-noun sludge —
`purchases`, `everything else`, `3% cashback`, `regular APR`, `daily essentials` — plus
every commodity noun the assistant's listicles mention. Thresholding barely helps
(0.7 → precision 27 %, while recall drops 6 pts): **salience filtering, not recognition,
is the LLM's actual value on the entity side.** Feeding this stream to the graph
unfiltered would bloat nodes ~6× and poison SHARED_ENTITY bridges with generic hubs.

### Relations — structurally not competitive
With the product's 30-predicate schema, only **29 %** of LLM relation pairs get any
local edge. The root cause is vocabulary: across just 517 relations the LLM coined
**224 distinct labels** (`offers`, `styled_with`, `allocated_to`, `graduated_from`,
`reverts_to`, …) — the top-60 cover only 67 % of uses. Fixed-label zero-shot RE cannot
express most of these; giving Relex those top-60 labels as an *oracle* lifts coverage
to only **42 %**, while over-generating 5.6× (2 909 edges, most shoehorned or wrong —
e.g. `Fetch Rewards -[spent_on]-> grocery purchases`). Whole relation families score
~0 %: attribute/value edges (`Chase Freedom Unlimited -[offers]-> 3% cashback`),
quantitative allocations (`$718 -[allocated_to]-> birthdays`), and product-spec facts.
What it does get reliably: classic person-centric edges — `me -[interested_in]->
Chase Freedom Unlimited card`, `NOVA LU works_at`-type pairs, `sibling_of`, `has_pet`,
`Rakuten -[owns]-> Ebates`.

### What the local model misses categorically (eyeballed 5 sessions)
1. **Temporal fields — a hard zero.** Relex has no notion of `status=ended` or
   `valid_from/to`. Example: the LLM emitted `NOVA LU -[worked_for]-> Goldman Sachs
   (status=ended)` and `graduated_from (from=2013-05-01)`; Relex found both *pairs* but
   as timeless edges — precisely the information the bi-temporal layer exists to store.
   Honest caveat: only **2 of 517** LLM relations in this sample carried temporal
   fields, so most temporal signal in this corpus comes from session `created_at`, not
   extraction — but the 2 that exist are exactly the answer-bearing kind.
2. **First-person state & quantities.** `User -[earns]-> $60,000`, `-[saves]->
   $20,000`, `-[spent]-> coffee mugs` — the memory-critical facts. Relex catches the
   *interest/possession* flavor (`me -[interested_in]-> …`) but nothing quantitative,
   since amounts aren't in its entity set and the predicates aren't in any fixed schema.
3. **Salience.** No sense of what a *session is about*; it transcribes every noun the
   assistant's boilerplate mentions. (Also: no tags — the product's tag lane would
   still need YAKE/noun-chunks.)
4. **Termination cues.** "formerly known as Ebates", "no longer" — polarity is lost
   even when the pair is found.

## 3. Throughput & cost

| | GLiNER-Relex (RTX 3060 Ti) | gpt-4o-mini extractor |
|---|---|---|
| Wall time / session | **1.7 s** (0.60 sessions/s; 33.5 s for 20, + 7 s one-time model load) | ~4 s nominal for the 2-call short-session case; this corpus averaged **5.6 calls/session** (sectioning + reflexion), so realistically 10–20 s serial (pipeline parallelizes) |
| Cost / session | $0 | ~$0.0023 (=$2.26 / 1 000 sessions) |
| Failure modes | none observed | 429 storms when org RPD is exhausted (bit us today: sessions took 4–10 min) |

Chunks were run serially through `predict_relations`; batching the chunks (as
`Gliner2Extractor` already does) should give another ~3–5×.

## 4. Recommendation — (a): gate / pre-filter, not replace

Full replacement (option b) is out: relations are not close (≤42 % ceiling with oracle
labels, at 5.6× over-generation), and even "entities only" would need a salience filter
the local model doesn't have. But the recall profile is exactly what a **tiered/lazy
extraction** gate wants — the local pass almost never misses something the LLM would
have named, so a cheap tier-0 can safely decide *what deserves an LLM call*:

- **Tier 0 (every session, $0):** Relex entities + relations. Use for (i) retrieval
  seeding / SHARED_ENTITY candidates after a stop-list + score cut, (ii) an
  entity-density / novelty signal for the gate, (iii) provisional person-centric edges
  (`interested_in`, `works_at`, `has_pet`, `owns` are reliable).
- **Escalate to gpt-4o-mini** for: open-vocabulary relations, all temporal fields, and
  salience-filtered canonical entities. Note the repo *already has this shape*:
  `CueGatedExtractor` (cue-triggered escalation) with a GLiNER floor. Relex is a strict
  upgrade over the current `gliner_yake_cooccur` floor — same entity engine, but real
  typed relations instead of the co-occurrence hack, at the same ~1.7 s/session. A
  lazy variant (extract locally at ingest, run the LLM only when a session is first
  *retrieved*) would move ~93 % of ingest cost to first-touch time.
- **Do not** feed raw Relex entities into node creation without a salience/stop-list
  layer — 6.4× over-extraction would hub-poison the graph.

**Not good enough (option c) applies to one sub-goal:** if the tier's purpose were to
produce dated fact statements locally, stop — no encoder model in this family emits
validity intervals or polarity; that stays LLM-only.

### If this graduates from spike to feature, next questions
1. Add `amount of money` / `medical condition` entity labels — likely closes most of
   the entity-recall gap (the 14 %) for free.
2. Batch chunks through Relex (mirror `Gliner2Extractor`'s batched passes) — target
   ~0.5 s/session.
3. The real accuracy question is downstream: run the `test-graph` harness with
   `local_backend` swapped to a Relex floor and compare answer accuracy / recall@k,
   not extraction-level agreement.
