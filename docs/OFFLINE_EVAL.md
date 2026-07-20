# OFFLINE_EVAL.md — read-side retrieval A/B (no paid LLM calls)

*2026-07-16. All numbers from `scripts/offline_eval.py` on local models only (bge-small
embedder + ms-marco cross-encoder, CPU). No OPENAI_API_KEY was used: the script strips
the key on import and stubs the extractor, so no paid call can fire. Re-run with
`.venv/bin/python scripts/offline_eval.py --stores synth,pilot --out <dir>`; the run
backing this doc is preserved at `runs/offline_eval/` (results.json, ppr_mass.json,
context snapshots).*

## 1. What was tested and why

The paid LLM sits only at ingest-time extraction and the final answer call. Everything
between is offline, and `KnowledgeGraph.search().context` is the **exact prompt blob**
`ask()`'s LLM would read — so "did the needed evidence reach the context" is the offline
proxy for answer quality. Variants, each behind a new Config knob whose default
reproduces current behavior exactly:

| variant | knob (kg/config.py) | mechanism |
|---|---|---|
| **A** | `history_all_lanes` (new, default False) | Serve the HISTORY (closed+open) block on **every** routed lane, not just STATE (kg/rag.py `ContextBuilder.build`). Keeps the "only when ended history exists" condition and `history()`'s limit=40 cap. |
| **B** | `projection_closed_facts` (new, default `"off"`; `"full"`/`"half"`) | Keep closed RELATED_TO edges in the PPR projection (kg/retrieval.py `_build_projection`), filtered by **belief** only; as-of views still exclude facts with `valid_at > as_of`. `"half"` diffuses closed edges at half weight. The knob is part of the projection cache key. |
| **C** | `ranked_facts` (new, default False) | `facts_for` (kg/rag.py) gathers **all** valid facts among the anchor entities, scores each line's semantic core (`"src rel dst"`) by embedding similarity to the query, keeps the top `rag_max_facts=30` — instead of truncating in arbitrary iteration order. |
| **D** | `self_guard="cap"` (knob pre-existing) | Cap every projection edge incident to the `me` self-anchor at `self_guard_cap=0.05` — the throttle the Engine's forced `self_entity=True` never turns on. Config-only. |
| **E** | `seed_reserve=2` (knob pre-existing; added mid-eval) | Reserve 2 context seats for the seeder's top episodes whose session won nothing downstream. Added after tracing probe b2 (§5.4). |

Combos: `AC`, `ABCD`, `AE`. Code touched (all default-off): `kg/config.py` (+3 knobs),
`kg/rag.py` (history gate, ranked facts, `ContextBuilder` now takes the embedder),
`kg/retrieval.py` (projection filter + cache key). No write path or `apply_fact`
semantics were modified. Full pytest: **312 passed, 1 failed** —
`test_search_and_answer_carry_structured_fact_rows`, which also fails on a clean
checkout of HEAD (the "facts store frequency" commit added `mentions`/`last_mentioned`
to `FactLine.to_row()` without updating the test). Pre-existing, unrelated.

## 2. Stores

**synth** — a synthetic first-person store built fresh per run by
`scripts/offline_eval.py` via `ScriptedExtractor` (the offline pattern from
`tests/test_temporal.py`), with `self_entity=True, self_name="me"`. 69 episodes,
344 nodes, 62 first-person fact assertions. Scenario coverage:

- repeated undated events: "went to the park" ×5 → confirm-collapses into one open edge
  rendered `mentioned 5x (2025-01-05 -> 2025-05-18)`
- explicitly-dated occurrences: yoga class 2024-03-10 / 04-14 / 06-02 → three separate
  open edges (repeatable-predicate path)
- superseded functional fact: `lives_in` Seattle → Denver (2023-05-01)
- "used to X" closures: volunteers_at animal shelter (ended 2024-01-20); employed_by
  Acme (ended 2022-08-15) → Globex (2022-09-06)
- bounded interval: Japan trip `[2023-11-01, 2023-11-14]` — a closed edge, invisible in
  the current view by construction
- ordinary states (cat Luna, sister Mia in Boston, Subaru, peanut allergy, …)
- ~35 filler me-facts (the me-hub carries ~60 facts, so the `rag_max_facts=30` cap and
  the PPR hub effect are both real) and 20 third-party distractors that lexically shadow
  the canaries (Becky's cat Whiskers, Raj's dog park, Dana's ended Denver residency,
  Becky visiting Seattle, …).

**pilot** — a **copy** (the original on disk is never opened) of
`store/events_pilot.db`: a real ingested chat store (302 episode chunks, 2,932 fact
edges, 48 closed facts; extraction cost already sunk). It has first-person *text* but no
`me` anchor (ingested with `self_entity` off), so D is inert there; it is the real-data
canary/bloat check for A/B/C. Notably its 48 closed facts are extraction noise
(`kefir --ferment--> shrubs (ended)`), which makes it a stress test for A's
noise-dredging risk.

## 3. Probes and metrics

30 questions on synth (3 preference, 6 history, 4 as-of + 1 as-of-from-text-only,
4 counting, 2 projection-sensitive, 10 canaries), 8 content-grounded canaries on pilot —
`SYNTH_PROBES` / `PILOT_PROBES` in `scripts/offline_eval.py`, each with gold episode ids
and gold answer substrings. Recorded per question × variant:

- **hit** (headline): any gold substring present anywhere in `.context`
- **coverage**: fraction of gold substrings present (hit saturates on stores this size;
  coverage is the discriminating number)
- **found_in**: which section carried each substring (episodes / facts / history) — this
  is what makes the win/loss mechanisms traceable
- **ep_recall**: fraction of gold episodes among the context episodes
- context chars, fact lines, history lines, routed lane, latency

Artifacts per run: `results.json`, `ppr_mass.json`, and 2–4 full `.context` snapshots
per variant under `contexts/<store>/<variant>/<qid>.txt`.

## 4. Results matrix

### synth (n=30)

| variant | hit | coverage | ep_recall | ctx chars | hist lines | note |
|---|---|---|---|---|---|---|
| baseline | 0.933 | 0.917 | 0.771 | 3,520 | 16.0 | |
| **A_history** | 0.933 | **0.933** | 0.771 | 5,111 (+45%) | 40.0 | fixes h2; no canary loss |
| B_full | 0.933 | 0.917 | 0.771 | 3,520 | 16.0 | **no change anywhere** |
| B_half | 0.933 | 0.917 | 0.771 | 3,520 | 16.0 | no change |
| C_ranked | 0.933 | **0.900** | 0.771 | 3,517 | 16.0 | loses p3; 1.5–4× latency |
| D_selfcap | 0.933 | 0.917 | 0.771 | 3,520 | 16.0 | no ranking change (§5.6) |
| AC | 0.933 | 0.933 | 0.771 | 5,108 | 40.0 | = A ∪ C, no interaction |
| ABCD | 0.933 | 0.933 | 0.771 | 5,108 | 40.0 | = A ∪ C |
| E_seedres | 0.933 | 0.917 | **0.657** | 3,521 | 16.0 | displaces gold eps (§5.5) |
| AE | 0.933 | 0.933 | 0.657 | 5,112 | 40.0 | |

Coverage by category (baseline → best variant): asof 1.00 (all), canary 1.00 (all),
counting 1.00 (all), history **0.92 → 1.00 under A**, preference 1.00 (**0.83 under
C** — the one C regression), projection 0.00 under *every* variant (§5.3–5.4).

### pilot (n=8)

| variant | hit | coverage | ep_recall | ctx chars | hist lines |
|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 1.000 | 22,229 | 0 |
| A_history | 1.000 | 1.000 | 1.000 | 22,229 (+0) | 0 |
| B_full / B_half | 1.000 | 1.000 | 1.000 | 22,229 | 0 |
| C_ranked | 1.000 | 1.000 | 1.000 | 21,799 (−2%) | 0 |
| D_selfcap | 1.000 | 1.000 | 1.000 | 22,229 | 0 |
| E_seedres / AE | 1.000 | 1.000 | 1.000 | **26,636 (+20%)** | 0 |

No pilot probe changed outcome under any variant; zero canary regressions on either
store for A/B/C/D.

## 5. Win/loss traces

### 5.1 A's win — h2 "Which companies have I worked for?" (SINGLE lane)

Baseline: episode retrieval is pure noise (the seeder latches onto "work" and returns
the banking/journal/Garmin episodes), the FACTS section shows only the open
`me --employed_by--> Globex`, and because the lane is SINGLE the HISTORY block is gated
off — **"Acme" never reaches the context** (coverage 0.5). Under A the block fires and
the first line is `me --employed_by--> Acme Corp (since 2019-02-01; until 2022-08-15;
ended)` → coverage 1.0. Same flip in every A-containing combo. This is exactly sharp
edge #1: closed facts invisible outside STATE. Note the STATE-phrasing sibling h1
("Where did I *use to* work?") already hit at baseline because "used to" routes STATE —
A's value is confined to history questions *phrased without* STATE cue words (also
h5 "Tell me about my time at Acme" already hits via lexical seeding). Real but narrow.

Bloat cost: on synth A adds ~1.6 KB (+45%) to **every** question, because the me-hub's
history (40-line cap) fires everywhere. Two structural issues found while reading the
blocks (snapshot `contexts/synth/A_history/h2.txt`):

- **HISTORY duplicates FACTS**: `history()` returns closed+open, so ~36 of the 40 lines
  restate open facts already listed in FACTS. The information A adds is only the closed
  lines (4 here). A "closed-only delta block" would deliver the same win for ~10% of the
  added chars.
- **`history()`'s cap keeps the *oldest* 40** (`kg/facts.py:163-165` sorts ascending by
  valid-time and takes `[:limit]`). On a store whose hub has >40 facts, a *recent*
  closure would be silently cut in favor of 2019 filler. On a real aged store this
  inverts the block's usefulness — worth fixing (rank by recency or query relevance)
  before or alongside promoting A.

On pilot, A added **zero** lines/chars: none of the probe anchors carry closed facts, so
the "only when ended history exists" condition keeps it dormant — i.e. on real data A is
free until it has something to say. The noise-closed facts (`kefir --ferment--> shrubs`)
never surfaced because their entities were never anchors for these probes; a probe
whose anchors DO touch noise closures would dredge them, so the delta-block/compactness
fix above doubles as the noise limiter.

### 5.2 C's loss — p3 "What do I usually do on weekends?" (preference)

Baseline keeps `me --hiked--> Mount Si` in FACTS by **insertion-order luck**; C's
embedding ranking scores "me hiked Mount Si" low against "what do I usually do on
weekends" (bge-small has no idea Mount Si is a weekend activity; the episode text that
says "this weekend" is not part of the fact line's semantic core) and cuts it →
coverage 1.0 → 0.5. Meanwhile C's *intended* win case never materialized: on both
stores, gold facts survive baseline truncation anyway, because `facts_for` iterates
anchor entities in `relevant_entities` order and **query-seeded entities come first** —
the gold entity is nearly always seeded, so its facts are emitted before the cap. The
hub-lottery only bites hub-mediated questions (preference/disposition), which is exactly
where the embedding ranking is weakest. Net on this evidence: C is neutral-to-negative,
plus 1.5–4× search latency (it embeds up to a few hundred fact lines per query,
uncached).

### 5.3 b1 — why no projection variant can fix it

"Who put together the get-together for former employees of my old company?" (gold: the
`Priya organized the Acme alumni picnic` episode). Trace: gold is absent from the
**entire 32-episode PPR pool** under baseline *and* B_full. The intended bridge
`me --employed_by(ended)--> Acme Corp → picnic episode` doesn't exist because the picnic
episode's extracted entity is **"Acme alumni picnic"**, which never canonicalizes onto
"Acme Corp" (cosine below link τ). B restores the me→Acme edge but the path dead-ends
one hop later. Read-side knobs can't fix an entity-resolution miss — that's
extraction/canonicalization work.

### 5.4 b2 — the failure nobody fixes (and the real shape of the gap)

"Which cities did I visit on my trip abroad?" (gold: the Japan-trip episode, whose
`traveled_to` fact is closed `[2023-11-01, 2023-11-14]`, so it's absent from
current-view FACTS). Trace: the trip episode is **seed #1** (bge finds it fine), but PPR
diffusion demotes it to **rank 7** and the context reads only the top 5 → "Tokyo/Kyoto"
never arrive. Identical under B (the closed edge's diffusion weight is negligible next
to the timeless mention corridors), under D (see 5.6), and — surprisingly — under E:
`_reserve_slots` only injects episodes whose *session isn't already represented in the
top-k*, and the trip episode IS in the top-k (rank 7), just below the 5-episode context
prefix. So the observed dominant failure is precisely **"in pool, below prefix"**, which
no current knob addresses. A cheap fix worth trying: extend seed-reserve (or a new knob)
to *promote* an in-k seed-top episode into the context prefix, not just inject uncovered
sessions. (A on this question also does nothing *for the cities* — HISTORY carries
`me --traveled_to--> Japan (…ended)`, which answers "did I travel" (c3 hits this way)
but not "which cities", which live only in the episode text.)

### 5.5 E's regressions — prefix displacement

E flipped ep_recall down on p1/h6/c1/c4/n6/n7: the two reserved seats are spliced into
the end of the 5-episode context prefix, and on these questions they **displaced gold
episodes** (n6 "Where do I live now?" lost the Denver episode itself; the answer
survived only via the FACTS line). On pilot E added +20% context chars with zero metric
change. As configured, E is net-negative on this evidence — don't promote without the
promote-in-k change from §5.4.

### 5.6 D — PPR mass, measured (`ppr_mass.json`)

| query | self mass none→cap | top3 share | entropy |
|---|---|---|---|
| What is my cat's name? | .0323 → .0260 (−20%) | .157 → .156 | 3.873 → 3.876 |
| Where do I work now? | .0368 → .0293 (−20%) | .198 → .197 | 3.735 → 3.739 |
| How many times did I go to the park? | .0251 → .0152 (−39%) | .203 → .203 | 3.706 → 3.708 |
| What do I like to do for fun? | .0314 → .0244 (−22%) | .153 → .152 | 3.853 → 3.856 |

The cap drains 20–40% of the self-anchor's own mass, but episode-level concentration
(top-1/3/5 share, entropy) moves in the 3rd decimal and **no probe's retrieval changed
at all**. Why so weak: (i) `me` sits on ~60 fact edges, so even capped at 0.05 each it keeps
substantial aggregate pull, and (ii) mass reaches episodes mainly through the
episode↔mention↔entity corridors, which the guard doesn't touch. D is safe to flip on
(free, marginally saner mass distribution) but on this evidence it isn't the lever the
"unthrottled super-hub" framing suggests — at least not at this store size.

### 5.7 B — a genuine null, with a mechanism

B changed **nothing on either store** (contexts byte-identical). Reason: every closed
`src --rel--> dst` fact edge is topologically shadowed by timeless structural paths
(`src ↔ mention ↔ episode ↔ mention ↔ dst`) built at ingest; dropping or restoring the
fact edge barely moves diffusion. The only case where B should matter — an entity pair
connected *only* by a closed fact, with no shared episode — cannot exist in this
pipeline, because the fact was extracted *from* an episode that mentions both endpoints.
B (incl. half-weight) can be shelved.

### 5.8 Other observations

- **a5** ("Where did I live in 2022?", `as_of` NOT passed — sharp edge #5): the text
  matches the STATE regex, so the baseline HISTORY block already rescues "Seattle" even
  though FACTS shows only Denver. The as-of gap is real but masked whenever the phrasing
  routes STATE; with A it is masked on every lane. A is therefore also a partial
  mitigation for edge #5.
- **pilot e5**: "light fixture" reaches the context **only** through a FACTS line (the
  episode never ranks in) — the facts section does rescue real questions, which is C's
  premise; C just needs to *keep* golds better than the insertion-order accident, and on
  current evidence it doesn't.
- **Counting questions**: c1 works at baseline via BOTH the 5 park episodes and the
  `mentioned 5x (2025-01-05 -> 2025-05-18)` fact rendering — the confirm-collapse
  frequency line is already a good count carrier. c2's three dated yoga occurrences all
  surface as separate FACTS lines. The park/yoga machinery is healthier than expected;
  the broken counting case is the *closed-interval* one (b2/c3 via episodes).
- Latency: A/B/D are free; C costs 1.5–4× per search (uncached per-query embedding of
  every candidate fact line). If C is ever promoted it needs a fact-line embedding cache
  keyed by edge (fact lines are immutable per (edge, valid_at)).

## 6. Recommendation

Promote to a single paid confirmation run: **A (`history_all_lanes=True`), amended** —
ideally with the two small fixes that fell out of the traces (they're in the same
function): render only the **closed** lines outside the STATE lane (kills the 90%
duplication and the noise-dredging risk on real stores) and rank `history()`'s cap by
recency instead of oldest-first. A is the only variant that flipped a question to
correct, it costs nothing when no closure exists (pilot: +0 chars), and it had zero
canary regressions on both stores.

Not worth a paid run on this evidence: **B** (provably null here — shadowed by mention
corridors), **C** (one regression, zero wins, 1.5–4× latency; revisit only with an
embedding cache and a better line representation, e.g. including the source episode
snippet), **D** (safe config hygiene, but changed no retrieval outcome — flip it on for
free if desired, don't spend a run on it), **E** (net-negative as-is; displacement
outweighs injection).

The most valuable follow-up isn't any of the four variants: it's the **"in pool, below
prefix" promotion gap** (§5.4) — the one reproducible hard failure (b2) had the gold
episode as seed #1 and still lost, and every knob tested walked past it — plus the
**entity-granularity miss** (§5.3), which is ingest-side.

---

# Round 2: event representation

*2026-07-16. Same harness, local models only, no key (`scripts/offline_eval.py`,
now defaulting to `--out runs/offline_eval_round2`; the run backing this section is
preserved there — results.json, edge_stats.json, context snapshots). Implements the
Round-1 recommendation (amended A) plus the write-side event fix it unlocks.*

## 1. What changed (code)

| piece | where | knob |
|---|---|---|
| **Amended A**: HISTORY block on every lane; outside STATE only the **closed** lines render (the delta — open lines measured ~90% duplication of FACTS in Round 1). STATE keeps its full closed+open block byte-identical. The non-STATE delta is also filtered by `as_of` (lines with `valid_at > T` dropped) so a pre-event as-of view can't show the future. | kg/rag.py `ContextBuilder.build` | `history_all_lanes` (default False) |
| **history() cap fix**: was sort-ascending-keep-oldest-40; now the capped selection is recency-ranked **with closed rows kept preferentially** — plain most-recent-40 would have cut the 2019 Acme closure on the synth me-hub (>40 facts) in favor of recent open filler, regressing exactly the h2 win the block exists for. Kept rows still render in ascending time order. | kg/facts.py `FactIndex.history` | none (strict improvement) |
| **Event classification, no LLM**: (a) any asserted relation arriving with BOTH `valid_from` and `valid_to` is event-shaped by construction; (b) an event-verb lexicon (`went_to, visited, attended, traveled_to, bought, purchased, ate, watched, hiked, tried, met_up, met_with, flew_to`) matched via `relation_content_key` stems, `event=True` stamped on the RelationNode alongside functional/symmetric. **`played` is pointedly excluded**: the stem collides with habitual `plays` ("plays tennis on Tuesdays"), and a state misclassified as an event is worse than a missed event. | kg/canonicalize.py (`_EVENT_SURFACES`, `predicate_is_event`), kg/models.py | stamped always; inert unless `event_facts` |
| **Event write semantics**: event-shaped asserted facts write `valid_to = start` → closed `[d,d]` (explicit bounded `[d1,d2]` passes through as before) with `event=True` on the edge; runs AFTER supersede so a bounded functional fact still displaces a standing value exactly as today. **Confirm-on-closed dedup**: for event-shaped assertions the confirm lookup matches closed edges with the same `valid_at` (dedup + `confirmed_by`); a different date opens a new closed occurrence edge. `fact_active`, retract/close/belief logic, and the PPR projection untouched (Round 1 proved B null). | kg/temporal.py `apply_fact` | `event_facts` (default False) |
| **Occurrence rendering**: edges with `event=True` render `(on 2025-01-05)` for `[d,d]`, `(2023-11-01 -> 2023-11-14)` for bounded — never since/until/ended; same-day repeats keep the `mentioned Nx` counter. `status` in `to_row()`/`_fact_row` is `"occurred"`, not `"ended"`. Old-format `[d,∞)` edges have no flag and render exactly as before — both representations coexist, no migration; the `event` column round-trips through the SQLite migration path (`ALTER TABLE` no-op pattern). | kg/facts.py `FactLine`, kg/engine.py `_fact_row`, kg/store.py | keyed off the edge flag |

Tests: `tests/test_event_facts.py` (14 new: lexicon stamping, [d,d] write, bounded
pass-through, event-shaped-by-construction, confirm-on-closed dedup, distinct-date
occurrences, occurrence rendering + `occurred` status, knob-off byte-compat, old-format
coexistence + SQLite round-trip, recency cap, closure-priority cap, delta block on/off,
delta as-of filter). Full suite: **327 passed, 0 failed** — the Round-1 pre-existing
failure (`test_search_and_answer_carry_structured_fact_rows`) was trivial (the
"facts store frequency" commit added `mentions`/`last_mentioned` to `to_row()` without
updating the test's expected key set) and is now fixed.

## 2. Round-2 configurations

The synth corpus is built TWICE: `synth` (legacy write, `event_facts=False`) and
`synth_ev` (new write path). Pilot is the same read-only copy (its edges are all
old-format — the coexistence check at scale).

| config | store | knobs |
|---|---|---|
| baseline | synth / pilot | none (must reproduce the Round-1 recorded baseline — it does, per-probe) |
| A_amended | synth / pilot | `history_all_lanes=True` |
| EV | synth_ev | `event_facts=True` (write side only; read side baseline) |
| **EV_A** (promoted) | synth_ev | `event_facts=True, history_all_lanes=True` |

The script now hard-gates: per-probe hit/coverage under the promoted configs must be
`>=` the Round-1 recorded baseline (`runs/offline_eval/results.json`), and the new
negative-substring probes must hold; any violation exits 1. This run: **38 probes
checked, no regressions, no neg violations.**

## 3. Results matrix (synth probes, n=30 shared)

| config | hit | coverage | ctx chars | hist lines/probe |
|---|---|---|---|---|
| Round-1 recorded baseline | 0.933 | 0.917 | 3,520 | 16.0 |
| baseline (this run) | 0.933 | 0.917 | 3,526 | 16.0 |
| A_amended | 0.933 | **0.933** | 3,788 (**+7%**) | 18.4 |
| EV alone | 0.900 | 0.867 | 3,467 | 16.0 |
| **EV_A** | 0.933 | **0.933** | 4,038 (**+14.5%**) | 24.2 |

Pilot (n=8): baseline and A_amended both 1.000/1.000; A_amended adds **+38 chars**
(0.2 hist lines/probe — see §5.3).

Per-probe changes vs the Round-1 baseline under **EV_A** (everything not listed is
unchanged at 1.0, including every canary n1–n10, all as-of probes, h1–h6, c1–c4):

- **h2 "Which companies have I worked for?" 0.5 → 1.0** — the Round-1 variant-A win,
  now delivered by the closed-only delta on the SINGLE lane (`Acme` arrives via
  HISTORY, `Globex` via FACTS).
- **b1 / b2** still 0.0 — the entity-granularity miss and the "in pool, below prefix"
  gap (Round 1 §5.3–5.4); read-side and write-side event work were never going to move
  these, and didn't.
- New event probes (no Round-1 baseline): **ev1–ev5 all pass under EV_A**, zero
  negative-substring violations: past events render as occurrences never state grammar
  (ev1/ev2); an as-of AFTER a point event finds it while later occurrences are filtered
  from the delta (ev3); an as-of BEFORE the trip shows no `me --traveled_to--> Japan`
  in any fact-derived section (ev4); all five park visits surface as five dated
  occurrence rows (ev5).

Counting probes kept their carriers in the new representation: c1 now reads five
`me --went_to--> the park (on …)` rows (plus the five episodes) instead of the single
`mentioned 5x` line; c2's three yoga dates all render as `(on 2024-03-10)` etc. in the
delta; c3's Japan interval renders `(2023-11-01 -> 2023-11-14)`.

## 4. Edge-count deltas (synth corpus, legacy vs event write)

| store | fact edges | open | closed | event-flagged |
|---|---|---|---|---|
| synth (legacy) | 78 | 72 | 6 | 0 |
| synth_ev | 82 (+4) | 61 | 21 | 16 |

The +4 is exactly the park: five dated occurrences instead of one confirm-collapsed
open edge. The 16 event edges are the 8 `went_to` (park ×5, climbing gym, and two
third-party), 3 `attended` yoga, 2 `traveled_to` (mine bounded + Raj's), 2 `hiked`,
1 `visited` — i.e. the lexicon caught everything intended and nothing else; 11 of the
15 new closures are mine, and every previously-open wrong-information event
(`me --went_to--> the park (mentioned 5x)` rendered as if still true) is now a dated
occurrence.

## 5. Surprises / notes

1. **EV without A is a regression, not a neutral step** (hit 0.933→0.900, coverage
   0.917→0.867; p3 "What do I usually do on weekends?" drops to 0.0 because
   `me --hiked--> Mount Si` becomes a closed occurrence and leaves current-view FACTS
   with no block to serve it). The write fix and the delta block are a package —
   `event_facts` must never ship without `history_all_lanes`.
2. **Plain recency for the history cap would have been a bug**: on the synth me-hub
   (>40 rows) most-recent-40 cuts the 2019 Acme closure — the exact line h2 needs.
   Hence the closure-priority cap (closed rows keep their seats; open filler competes
   for what's left). This deviates from "rank by recency" as literally specified,
   deliberately.
3. **The as-of delta filter is edge-granular, which is correct but trips naive probes**:
   the first version of ev4 asserted no "japan" in fact sections as-of 2023-06-01 and
   failed — because `Raj --traveled_to--> Japan (on 2023-05-30)` legitimately predates
   T. The probe now scopes to `me --traveled_to--> japan`.
4. **On pilot the delta block is no longer strictly free**: Round-1 A added +0 chars;
   the amended delta fires for one probe's anchors (+38 chars, 0.2 lines/probe, noise
   closures of the `kefir --ferment--> shrubs` kind) with zero metric change. The
   closed-only rendering keeps the dredging bounded to exactly the closed lines.
5. **`played` had to be excluded from the event lexicon** — `relation_content_key`
   folds `plays`/`played` onto one stem, so the habitual state would be misclassified.
   Same check killed `stayed_at` (`stays_at` collision). Any future lexicon addition
   needs the collision test: does the habitual/present form share the content key?
6. STATE-lane output is byte-identical under every Round-2 config except the history
   cap fix (which only changes stores whose anchors carry >40 fact rows), and
   `baseline` reproduced the Round-1 recorded per-probe numbers exactly.

## 6. Recommendation

Promote **`event_facts=True` + `history_all_lanes=True` as a pair** to a paid
confirmation run. On this evidence the pair: fixes the open-events wrong-information
class at write time (16 edges on synth), delivers the Round-1 variant-A findability win
(h2), keeps every canary and counting probe at parity or better, holds all five new
event-semantics probes with zero state-grammar leaks, and costs +14.5% context on synth
(vs +45% for Round-1's unamended A) and ~nothing on real data. Old stores need no
migration; both representations render correctly side by side.

Still open, unchanged from Round 1: the "in pool, below prefix" promotion gap (b2) and
the entity-granularity miss (b1) — both out of scope here and unmoved.

---

# Round 3: seed-score fusion

*2026-07-17. New harness: `scripts/offline_eval_round3.py` (+
`scripts/round3_flip_contexts.py` for the flip-question context dumps), run against
the CACHED per-instance stores of the latest paid run (`runs/sample-datefix-events-1`,
LongMemEval small, 100 questions) — extraction cost already sunk, $0 spent here. Local
models only (bge-small + ms-marco CE); the scripts strip any API key on import and stub
the extractor. Results preserved at `runs/offline_eval_round3/` (results.json with
per-question rankings/pools/context membership at every alpha, contexts/<qid>/ flip
snapshots, offline_eval_round3.log).*

## 1. Hypothesis and change under test

The Seeder computes per-episode query-relevance scores (BM25+embedding fused), uses
them only to seed PPR personalization, then discards them; diffusion replaces query
relevance with topical mass. Documented casualties in the paid run: 06f04340 (gold
session out of context), 1c549ce4 / 2ce6a0f2 (gold evidence lost marginal seats), plus
Round 1's b2 ("seed #1, PPR rank 7"). Hypothesis: blending the seed score back into
the base score that feeds MMR restores those seats.

**Code** (fusion only — no change to `_reserve_slots`, retargeting, or any write path):

| piece | where | knob |
|---|---|---|
| `fused = α·norm(ppr·dist_boost) + (1−α)·norm(seed_score)`, `norm` = per-query min-max over the candidate set (PPR mass ~1e-6..1e-2 and seed scores ~0..1 are incomparable raw); `seed_score = 0.0` for unseeded candidates. Computed in `PPRRetriever._rerank` on the base score MMR consumes, so `HybridRetriever` (ask/search path) inherits it and MMR / CE-rerank / `_reserve_slots` all see the fused ordering. The branch is skipped entirely at α=1.0 — the default is byte-identical to pre-knob behavior by construction. | kg/retrieval.py `PPRRetriever._rerank` | `seed_fusion_alpha` (default **1.0**; QUERY-SIDE — deliberately NOT in `INGEST_RELEVANT_FIELDS`, so cached stores stay valid) |

Tests: `tests/test_seed_fusion.py` — α=1.0 output (ids AND rounded scores) checked
against a frozen copy of the pre-change `_rerank` on three queries; an α<1 re-ordering
case with exact blend values; a no-episode-seeds degenerate case (pure-PPR order kept).
Full suite: **349 passed, 0 failed**.

## 2. Harness: driving search() offline on the paid run's exact stores

- **Stores**: `store/cache/*.db` per-instance ingest caches. Verified, not assumed: the
  exact `ingest_cache_key` was recomputed per instance and matched against the cache —
  the run ingested with `extractor_backend=cue_gated, event_facts=true,
  ingest_date_filter=true` (all 100 instances hit; every hit is also the newest cache
  entry for its instance; the 06f04340 store carries 43 event-flagged / 87 closed fact
  edges as expected for the event-facts write path). Stores are **copied** to a temp
  path before opening — the cache is never touched.
- **Question set**: rebuilt from `runs/sample-datefix-events-1/run.json` (id, query,
  gold session ids, `answer_expected`, `question_date` passed as `as_of`, exactly as
  `run_per_instance` passes it to `ask()`).
- **Config**: the run's query-side knobs (`rag_retarget=ce, rag_provenance_promote=true,
  mmr_lambda=1.0, rag_parent_expand=2, rag_chunks_per_source=2, history_all_lanes=true,
  event_facts=true`), k=8.
- **Path driven**: `HybridRetriever.retrieve` → `ContextBuilder.build` — exactly what
  `KnowledgeGraph.search()` runs; `.context` is the blob the answer LLM would read.
- **Fidelity gate**: at α=1.0 the harness reproduces the paid run's recorded
  gold-in-context marks **184/184** — the offline baseline IS the paid run's retrieval.

Per question × alpha: gold rank in the final top-8 and in the fused pool,
gold-in-context at session level and at answer-chunk level (a gold-session chunk whose
raw text contains `answer_expected`), `answer_expected`-substring-in-context, context
chars, and Kendall tau of the pool ordering vs α=1.0.

## 3. Alpha sweep matrix (n=100)

| α | all-gold-in-ctx | any-gold-in-ctx | ans-chunk-in-ctx | ans-substr-in-ctx | ctx chars/q | pool τ vs 1.0 |
|---|---|---|---|---|---|---|
| **1.0** (baseline) | 85 | 97 | 43 | 43 | 27,851 | 1.000 |
| 0.9 | 85 | 97 | 43 | 43 | 27,577 | 0.980 |
| 0.8 | 85 | 97 | 43 | 43 | 27,450 | 0.967 |
| 0.7 | 85 | 97 | 43 | 43 | 27,532 | 0.958 |
| 0.5 | 85 | 97 | 43 | 43 | 27,763 | 0.940 |

Not just equal counts — the **question sets behind every context-level metric are
identical at every alpha** (same 85, same 97, same 43/43). Context-outcome regressions
vs α=1.0: **zero**. Context-outcome fixes: **zero**. Per-category effects at the
context level: none — every `kind` (multi-session, temporal-reasoning,
knowledge-update, single-session-{user,preference,assistant}) is outcome-unchanged
across the sweep.

**Blast radius under the flat surface** (why the τ column matters): the fused pool
order changes for 86–100 of 100 questions, the final top-8 changes for 21 (α=.9) → 45
(α=.5), the 5-episode context prefix changes for 9 → 23 — all of it swaps among
non-gold / same-session-redundant chunks. Per-lane τ at α=0.5: multihop .983, state
.930, single .903 (single moves most: no CE re-rank downstream of the fusion there).

Gold **rank** movement (within "still in context" outcomes):

| α | pool rank up/down | final rank up/down | worsened (q, gold, 1.0→α) |
|---|---|---|---|
| 0.9 | +9 / −5 | +1 / −0 | — (0977f2af gold_1 *enters* top-8 at rank 8) |
| 0.8 | +12 / −12 | +1 / −2 | 0862e8bf_abs 3→4; 71017277 6→7 |
| 0.7 | +14 / −18 | +1 / −2 | 0862e8bf_abs 3→4; 71017277 6→8 |
| 0.5 | +29 / −29 | +3 / −6 | 0862e8bf_abs 3→4; **06f04340 6→8**; 07741c44 g2 1→2; 71017277 6→8; 27016adc g2 1→3; 37f165cf g2 1→2 |

Mean pool-rank delta for gold: +0.01 (α=.9) → −0.08 (α=.5). The movement is symmetric
noise trending slightly *against* gold as alpha drops. None of these rank moves crossed
the 5-episode context boundary in either direction (the one entry, 0977f2af at rank 8,
lands below the prefix — in-k but unread, the b2 shape again).

## 4. Flip-question outcomes — none recover, and each shows why

| q | paid-run failure | outcome across α ∈ {1.0…0.5} |
|---|---|---|
| **06f04340** "what should I serve for dinner… homegrown ingredients" | gold session out of context (rank 6, prefix reads 5) | Unchanged at α≥0.7; **worse at α=0.5 (rank 6→8)**. Trace: the gold session's chunks are NOT among the top episode seeds — the seed list is dominated by other cooking/dinner sessions (91223fd5_1, 6e6fbb6b, b459f888_3 score 0.91–1.0; gold's best chunk is lower). The premise inverts: here the *seeder* prefers the distractors and PPR was what held gold at 6; blending seed score in pushes gold down. Notably the evidence substrings (cherry tomato / basil / mint) are in the context at every alpha via FACTS lines and overlapping chunks — the paid answer's miss wasn't purely a retrieval gap. |
| **1c549ce4** "total cost of car cover + detailing spray" | gold chunk lost retarget/sibling seats | Both gold sessions rank 1–2 and are in context at every alpha, and the price evidence ("car cover", "$120", "$20") is present in the α=1.0 context **on the current working tree** — the chunk-seat loss documented from the paid run does not reproduce at today's HEAD+knobs baseline. Nothing for fusion to recover; nothing regresses. |
| **2ce6a0f2** "how many art-related events" (needs 4 sessions) | gold _3 absent | Gold _3 is absent from the **entire fused pool at every alpha** — it never becomes a candidate, so no re-weighting of candidates can seat it. 3/4 golds in context at every alpha, unchanged. This is a seeding/extraction recall miss, not a ranking miss. |

## 5. Why fusion is inert here (mechanism)

1. **No headroom in the signal**: the run's retrieval already scores recall@pool 0.955
   / MRR 0.946 — where gold is findable, seed and PPR agree, so the blend is a no-op on
   the outcome. The seed score is the same signal that *personalized* the PPR walk;
   fusing it back mostly re-concentrates mass PPR had deliberately diffused.
2. **Where they disagree, the seeder is not reliably righter** (06f04340): distractor
   sessions can out-seed gold on lexical/semantic surface. Fusion assumes
   seed-disagreement = PPR error; on this data it is ~50/50 (pool moves +29/−29 at
   α=0.5), i.e. noise.
3. **73 of 100 questions route state/multihop**, where the cross-encoder re-ranks the
   pool downstream — the fused ordering survives only as pool membership (which the
   candidate trim keeps alpha-independent) and the `rerank_keep_ppr_top=3` guarantee.
   Fusion's direct leverage is confined to the single lane (26 q), and there it moved
   redundant chunks, not gold.
4. The remaining hard failures are **not rank-blend-shaped**: below-prefix (0977f2af
   enters at rank 8, unread; 06f04340 stuck at 6–8), never-in-pool (2ce6a0f2 _3), or
   already-solved-at-baseline (1c549ce4).

## 6. Recommendation

**No alpha dominates — do not promote; do not spend a paid run.** Keep
`seed_fusion_alpha=1.0` (the shipped default; byte-identical to pre-knob behavior,
guarded by test). Every α<1.0 delivers exactly zero context-level wins on 100 real
questions while perturbing the final ranking of up to 45 of them and mildly degrading
gold final ranks at α≤0.8 (net −1 to −3). If a paid confirmation of the *null* is ever
wanted anyway, the cheapest probe is
`--set seed_fusion_alpha=0.8` on the same tier — but the offline evidence says the
money is better spent elsewhere.

The knob stays in the codebase as cheap, tested infrastructure: it is the right
5-line hook if a future seeder (e.g. distilled preference lines, better composite
docs) produces a score genuinely orthogonal to PPR.

What the failures actually ask for, consistent with Rounds 1–2: a **context-prefix
promotion** rule for in-k-below-prefix golds (b2, 0977f2af, 06f04340 — Round 1 §5.4's
suggestion, still unbuilt), and **seeding/extraction recall** work for never-in-pool
golds (2ce6a0f2 _3, b1). Score fusion addresses neither.

---

# Round 4: context-prefix promotion

*2026-07-18. The rule Rounds 1–3 kept pointing at, built and measured:
`scripts/offline_eval_round4.py` (same harness pattern as Round 3 — cached
per-instance stores of `runs/sample-datefix-events-1` copied to temp paths, exact
`search()` read path driven offline, API keys stripped on import, $0 spent). Results
preserved at `runs/offline_eval_round4/` (results.json with per-question rankings /
context membership / promotion records at every setting, offline_eval_round4.log).*

## 1. What changed (code)

| piece | where | knob |
|---|---|---|
| **Prefix promotion**: after the final ranking is fully assembled (post session-dedup, post `_reserve_slots`, post §7.3 window bound), take the Seeder's top-N scored valid EPISODES. Each that is already in the final top-k but ranked BELOW the context prefix (`rag_context_episodes`) is moved into the last prefix seat(s); displaced episodes slide down from the bottom of the prefix and stay in the top-k (pure reorder — nothing enters or leaves the ranking). The head (rank-1) is never displaced (promotions cap at prefix−1). NOT gated on session coverage (the exact case §5.4 showed `seed_reserve` skips), and never INJECTS an out-of-k episode (that stays `seed_reserve`'s territory, still off). Promoted ids are recorded on `RetrievalResult.promoted`. | kg/retrieval.py `HybridRetriever._promote_seed_top` | `seed_promote` (default **0** = off, byte-identical; QUERY-SIDE — deliberately NOT in `INGEST_RELEVANT_FIELDS`) |
| **Retarget seat-lock**: a promoted chunk is exempt from `_retarget_chunks` swaps — both the seed/CE refill pass and the `seed+lex` eviction loop treat `result.promoted` ids as locked seats (evicting the chunk there would undo the promotion one stage later). With no promotions the lock set is empty and both passes are provably unchanged. Sibling expansion and provenance promotion never evict an originally-selected chunk, so no further exemption is needed. | kg/rag.py `ContextBuilder._retarget_source` / `_retarget_chunks` | keyed off `result.promoted` |

`seed_reserve`, `rerank_keep_ppr_top`, MMR, the CE, and every write path are untouched —
any outcome delta is attributable to promotion alone.

Tests: `tests/test_seed_promote.py` (12 new: default off; knob-off unit no-op; full
retrieve→build pipeline byte-identical vs a reference with the promotion hook
structurally removed; below-prefix seed-top promoted into the last prefix seat with the
displaced episode kept in-k; two promotions in seed order; already-in-prefix costs
nothing; in-prefix seed-top consumes an N slot; out-of-k never injected; head never
displaced (prefix−1 cap, 1-seat prefix promotes nothing); `res.promoted` wiring;
retarget seed-mode and lex-mode each demonstrably evict the chunk when unprotected and
keep it when promoted). Full suite: **361 passed, 0 failed** (349 + 12).

## 2. Sweep (n=100, Round-3 config: `rag_retarget=ce, rag_provenance_promote=true,
mmr_lambda=1.0, rag_parent_expand=2, rag_chunks_per_source=2, history_all_lanes=true,
event_facts=true`, k=8)

**Fidelity gate first**: at `seed_promote=0` the harness reproduces the paid run's
recorded gold in_context marks **184/184** (same gate Round 3 passed).

| N | all-gold-in-ctx | any-gold-in-ctx | ans-chunk-in-ctx | ans-substr-in-ctx | ctx chars/q | fired | of which raw-PPR-top-3 |
|---|---|---|---|---|---|---|---|
| **0** (baseline) | 85 | 97 | 44* | 44* | 27,856 | — | — |
| 1 | 85 | 97 | 43 | 43 | 27,824 | 8 | 7 |
| 2 | 85 | 97 | 43 | 43 | 27,786 | 24 | 20 |

*\*One-question caveat: 27016adc's answer-chunk mark flips across PROCESSES (44 here vs
Round 3's 43) on an identical retrieval ranking — a CE near-tie in the retarget pick
(`_1#c003` vs `_1#c005`) that is stable within a run (4/4 identical on re-execution,
including with the promotion hook structurally removed) but not across environments.
Pre-existing builder wobble, independent of this knob; all comparisons below are
within-run.*

**Wins: zero, at every level, at both settings.** Session-level gold-in-context is
outcome-identical to baseline (same 85, same 97). Regressions:

- **36b9f61e** (ans-chunk AND ans-substr, at BOTH N=1 and N=2 — the only metric-level
  regression, named per the no-ship rule): at baseline all three gold sessions hold
  final ranks 1–3 and the answer chunk region (`_1#c004/#c005`) is in context via
  selection + sibling expansion. Promotion moves `_1#c001` (the seeder's top chunk —
  of a GOLD session, but the wrong chunk of it) into the prefix; under
  `rag_chunks_per_source=2` it takes `_1#c004`'s seat, sibling expansion now covers
  `c000–c003` instead of `c002–c006`, and the new retarget seat-lock prevents the CE
  from swapping `#c001` back toward the answer chunk. The answer leaves the context.
  Mechanism: **seed-top chunk ≠ answer chunk within the same gold session** —
  promotion + per-source cap + seat-lock can evict the right chunk of the right
  session. The question was already fully covered; promotion was pure downside.
- **2788b940** (gold-session displacement, both N; invisible in the headline counts
  only because `all_gold` was already False via a different missing gold): promotion
  seats `ultrachat_410014#c004` — a NON-gold distractor the seeder scored top — and
  gold session `_4`'s chunks fall out of the context entirely. The 06f04340 lesson
  from Round 3 §5.2 again: where seeder and pipeline disagree, the seeder is not
  reliably righter.

Promotion mechanics observed: N=1 fired 8/100 (8 promotions: 4 gold-session chunks,
4 distractors); N=2 fired 24/100 (26 promotions: 15 gold, 11 distractors). Context
size is flat (mean −390 / −290 chars on fired questions; extremes ±5 KB are sibling-
window shifts). **Promoted-but-never-seated leak**: 3 (N=1) / 5 (N=2) promoted chunks
never reached the built context — `_select_episodes`' per-source cap skips a promoted
chunk whose source already holds `rag_chunks_per_source` seats above it, so the
promotion silently no-ops (worth knowing if this rule is ever revisited).

**Interaction with `rerank_keep_ppr_top=3`**: 7/8 (N=1) and 20/24 (N=2) fired
questions promoted an episode sitting in the raw-PPR top-3 — i.e. promotion mostly
"rescues" exactly the episodes the keep-ppr guarantee re-inserts at the tail. But the
rescue is hollow: those episodes' sessions are almost always already represented in
the prefix by another chunk, so the move swaps chunks within covered sessions
(no metric change) — or worse (36b9f61e).

## 3. The regression set — why the expected wins don't materialize

| q | expectation | outcome, and the mechanism |
|---|---|---|
| **0977f2af** | expected WIN (rank-8 gold enters the prefix) | **No change — the premise doesn't hold at the shipped config.** Gold `_1`'s pool rank at α=1.0 is **9**: "enters top-8 at rank 8" was the α=0.9 fusion artifact (Round 3 §3), and the shipped default is α=1.0. Out-of-k → promotion (correctly, by design) never injects. Gold `_2` is seed #1 and final rank 1 — already read; its seed dominance also means N=1's one slot is consumed by an already-seated episode. This failure is out-of-pool territory (seeding/recall), not below-prefix. |
| **06f04340** | possible win (gold final rank 6, prefix 5) | **No change — fails the seed gate.** The shape is exactly right (in-k, below prefix) but promotion is seed-gated and the gold's best chunk is only **seed rank 11** (0.706) behind ten distractor chunks at 0.85–1.0 (`91223fd5_1`, `6e6fbb6b`, `b459f888_3` — the same distractor dominance Round 3 §4 documented). N≥11 would promote ten distractors first. A seed-gated rule cannot fix a case where the seed signal itself prefers the wrong sessions. |
| **2ce6a0f2** | expected NO change (gold `_3` never in pool) | **Correct no-op, verified.** Gold `_3` absent from the entire pool at every N — never injected. At N=2 the rule promotes gold `_2`'s `#c002` (session already in context at rank 5): outcomes unchanged, no harm. |
| **canaries** (10 paid-run judge-correct, gold-in-context: 001be529, 00ca467f, 0100672e, 01493427, 031748ae, 031748ae_abs, 06878be2, 06db6396, 07741c44, 07741c45) | no change | N=1: zero changes of any kind. N=2: two questions (001be529, 00ca467f) swap context chunk composition within already-covered gold sessions; zero metric changes. |

## 4. Recommendation

**No-ship, at N=1 and N=2. Keep `seed_promote=0`** (the shipped default;
byte-identical, test-guarded). On 100 real questions the rule delivers **zero**
context-level wins at any granularity and produces one answer-chunk regression
(36b9f61e) plus one gold-session displacement (2788b940) — it breaks more than it
fixes, which was the pre-declared no-ship line. Do not spend a paid run; if a paid
confirmation of the *null* is ever wanted anyway, the probe is `--set seed_promote=1`
on top of the Round-3 config (N=2 fires 3× more often for the same zero wins).

Why the gap the rule was built for turned out to be empty: the two live failure cases
each fail one of its gates — 0977f2af's gold is out-of-k at α=1.0 (the "rank 8" was
Round 3's α=0.9 artifact), 06f04340's gold is out-of-seed-top (rank 11) — and where
both gates DO pass, the seed-top episode's session is nearly always already
represented in the prefix, so promotion just reshuffles chunks of covered sessions.
The session-level "in pool, below prefix" failure essentially does not exist at k=8 /
prefix=5 on this benchmark with today's ranking stack; what remains is (i)
**chunk-level identity** within covered sessions — retargeting's job, and 36b9f61e
shows a seed-score prior makes it worse, not better, when the seed-top chunk isn't the
answer chunk — and (ii) **out-of-pool recall** (2ce6a0f2 `_3`, 0977f2af `_1`, b1),
which no post-ranking reorder can touch. Round 1's b2 (synthetic, seed #1 demoted to
rank 7) remains the one clean specimen of the shape, but it did not survive contact
with the real benchmark's failure distribution.

The knob stays as tested, default-off infrastructure: `_promote_seed_top` +
`RetrievalResult.promoted` + the retarget seat-lock are the right seams if a future
seeder (distilled preference lines, better composite docs) makes seed-top a
trustworthy gold signal — the current one isn't.

---

# Round 5: out-of-pool recall diagnosis

*2026-07-18. The failure class Rounds 3–4 kept pointing at — gold that is absent from
the entire candidate pool, so no post-ranking reorder can touch it — traced funnel by
funnel. `scripts/offline_eval_round5.py` (same harness pattern: cached per-instance
stores of `runs/sample-datefix-events-1` copied to temp paths, the exact
`HybridRetriever.retrieve → ContextBuilder.build` read path driven offline, API keys
stripped on import, local models only — bge-small + ms-marco CE — $0 spent). Results at
`runs/offline_eval_round5/results.json` (per-question, per-gold-session, per-chunk funnel:
BM25/embedding/seed/PPR ranks+scores, composite-doc surfaces, entity overlap, and the
smaller-chunks sub-split measurement). No behavior change to `kg/` — instrumentation
reconstructs the funnel by calling the public seeder/projection/PPR entry points; the
Round-3/4 knobs (`seed_fusion_alpha`, `seed_promote`) are gone from the tree and this
round does not use them.*

## 1. Fidelity gate — and a pool-level shift the gate cannot see

**Context-level fidelity holds: 184/184.** At the baseline config the harness reproduces
the paid run's recorded `gold_marks.in_context` on all 184 marks (same gate Rounds 3–4
passed). One cross-process wobble: a standalone re-run scored 183/184, flipping
`0bc8ad93 _1` (a state-lane gold at pool rank 11 / final rank None that only reaches
context via sibling/provenance expansion — a CE near-tie at the context boundary, the
same nondeterminism class Round 4 flagged for `27016adc`). In-process the number is
184/184; the flip is environment noise, not a revert effect.

**But the POOL-level population is NOT reproduced, and the fidelity gate is blind to it.**
`in_context` cannot distinguish "gold absent from the pool" from "gold in the pool but
ranked below context" — both render as out-of-context. The paid run's own `gold_marks`
carry a separate `hit` flag (= gold in the retrieved pool, `recall_at_pool`). Comparing
that to today's HEAD:

| | paid run (`hit=False`) | at today's HEAD |
|---|---|---|
| out-of-pool gold sessions | **13** | **2** |
| of those 13, now in-pool (rank 3–31) | — | 11 |
| of those 11, reached context | — | 1 (`6e984302 _1`) |

The three specimens the brief named were out-of-pool **in the paid run** and are all
in-pool at HEAD: `2ce6a0f2 _3` (paid `hit=False` → pool rank 20), `0977f2af _1` (→ rank 9),
`1a8a66a6 _4` (→ rank 31). The working-tree state since Round 4 (the revert, and/or
intervening edits) shifted pool composition — it *raised* `recall_at_pool` on 11 golds —
while changing **zero** `in_context` outcomes, which is exactly why the context-level gate
still reads 184/184. Per the brief's instruction, **all Round-5 findings are anchored to
today's measured baseline**, and the discrepancy with Rounds 3–4's out-of-pool prose is
attributable to this pool-level shift, not to a measurement difference (pool = `base.objects`
= `res.ppr_pool`, identical to Rounds 3–4's definition; verified airtight against the full
`cand_ids` including the STATE fact-augment).

**Consequence for the study:** at HEAD, "gold absent from the entire candidate pool" is
nearly extinct — **2 gold sessions across 2 questions**. The 11 paid-run out-of-pool golds
that moved into the pool did *not* thereby reach context (10 of 11 are now in-pool-below-
context — Round 4's null territory); pool recall improved without answer-recall improving.

## 2. The enumerated population (today's baseline)

| q | gold | lane | Q / answer | in cand pool | reached PPR | in seed dict | best emb rank | best BM25 rank | best PPR rank | seeded-entity overlap | paid-run judge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `4dfccbf8` | `_2` | state | "What did I do with Rachel on the Wednesday two months ago?" / *"I started taking ukulele lessons with Rachel."* | **no** | yes (45) | no | 202 | 29 | 45 | **0** | ✅ correct |
| `1a8a66a6` | `_2` | multihop | "How many magazine subscriptions do I currently have?" / *"2"* | **no** | yes (201) | no | 59 | 18† | 201 | **0** | ❌ incorrect |

†BM25 rank 18 is a *false-friend* chunk (`#c006`, plant-styling text matching "currently/have"),
not the magazine chunk; the actual magazine mention (`#c000`/`#c001`) sits at BM25 162 / emb 107.

Both were also out-of-pool in the paid run — they are the residue that the HEAD pool
shift did **not** rescue.

## 3. Per-case cause, with funnel evidence

**`4dfccbf8 _2` — genuine content mismatch (benchmark multi-gold artifact). HARMLESS.**
The question's answer ("ukulele lessons with Rachel") lives in gold `_1` (its `#c002–c005`
carry "Rachel"), and `_1` is in context — the paid run answered **correctly**. Gold `_2`
is a *keyboards/amps/pedals shopping* session (Korg, Roland, Fender, Taylor GS Mini,
compressor pedals): it contains none of the query anchors — no "Rachel", no "ukulele", no
"Wednesday" — and its 8–14 extracted entities per chunk (guitar/amp models) have **zero**
overlap with the query's seeded entity neighborhood (only 1 entity seeded at all). Best
embedding rank 202/309, BM25 29. It reached PPR (rank 45) only through generic corridors,
never through a query-relevant entity. This is not a retrieval defect — the session is
correctly deprioritized; it is gold only by the benchmark's session-level labeling.

**`1a8a66a6 _2` — composite-doc poverty / entity-bridge gap, compounded by dilution. REAL.**
The two subscriptions the count needs are **Architectural Digest** (only in `_2`) and
**National Geographic** (only in `_4`). In `_2`, "Architectural Digest" is mentioned once —
*"by the way, I'm also getting Architectural Digest, which I love for home decor
inspiration"* — inside a Mid-Century-Modern / Snake-Plant / Pothos decorating session whose
composite doc is dominated by interior-design vocabulary.
- **LEXICAL**: the query term "magazine"/"subscription" is **absent** from the answer
  chunk `#c000` ("Architectural Digest, which I love for home decor" — no "magazine"),
  so BM25 over the composite doc can't bridge it (`#c000` BM25 rank 162).
- **COMPOSITE-DOC POVERTY**: "Architectural Digest" *is* extracted as an entity surface,
  but nothing types it as a *magazine* — no tag/concept bridges the proper noun to the
  query's category term. The surfaces on the answer chunk are `['Architectural Digest',
  'living room']`; none relate to "magazine subscription".
- **DILUTION/SEMANTIC**: the decor-dominated embedding puts the best `_2` chunk at rank 59;
  PPR rank 201 (outside the multihop trim of 192) because zero seeded-entity overlap gives
  it no corridor.
- **Compounding**: gold `_4` (National Geographic) fails *identically* — proper-noun
  magazine title, no "magazine" bridge, buried in a groceries/ziplock-bags session; it
  reached pool rank 31 but not context. The question needs both `_2` and `_4` and gets
  neither → wrong count.

## 4. Contrast group — what saves the near-misses (n=168 in-context golds)

Seeding is the gate. **166 of 168** in-context gold sessions had at least one chunk inside
the top-`seed_k` (=10) of embedding **or** BM25 — i.e. they were directly seeded. Only
**2** reached context on graph structure alone, and both were *close* on seeding and were
carried by strong PPR corridors:

| q | gold | lane | best emb rank | best BM25 rank | best PPR rank | what saved it |
|---|---|---|---|---|---|---|
| `157a136e` | `_1` | state | 16 | 44 | **3** | PPR corridors from seeded entities lifted it to pool rank 3 despite missing the seed cutoff |
| `129d1232` | `_3` | multihop | 22 | 39 | **24** | wide multihop pool (64) + PPR rank 24 kept it, then CE/sibling pulled it into context |

The two out-of-pool failures fail *both* gates the survivors passed: their best raw-source
ranks are 59/202 (emb) and 18‡/29 (BM25) — far outside seed_k — **and** they have zero
seeded-entity overlap, so PPR has no corridor to rescue them (PPR rank 201/45). The
survivor pattern that they lack is: *either* clear the seed cutoff on one raw source, *or*
have an entity corridor to a seed. Out-of-pool = neither.

## 5. Would smaller chunks have fixed it? — measured, no

For every out-of-pool gold chunk carrying the answer substring (all 11 chunks of
`1a8a66a6 _2`; `4dfccbf8 _2` has none — its session doesn't contain the answer, so
chunking is irrelevant there), the chunk's raw text was re-split at natural (turn)
boundaries into ~500-char sub-chunks and each embedded with the same bge-small; the best
sub-chunk cosine was compared to the current embedding seed cutoff (the rank-`seed_k`
episode score, 0.591).

| chunk | len | n sub | best sub cosine | cutoff | clears? |
|---|---|---|---|---|---|
| `_2#c001` (the "subscribing to Architectural Digest" sentence) | 2719 | 6 | **0.570** | 0.591 | no |
| `_2#c006` | 264 | 1 | 0.543 | 0.591 | no |
| `_2#c000` (answer chunk) | 293 | 1 | 0.512 | 0.591 | no |
| …all 11 chunks | — | — | 0.449–0.570 | 0.591 | no |

**Aggregate: 0/11 sub-chunks clear the seed cutoff.** The closest (0.570) is the isolated
"subscribing to Architectural Digest" sentence — still under. The miss is semantic, not
geometric: at *any* granularity, "Architectural Digest … home decor" is simply not near
"magazine subscriptions" in embedding space, because the sentence never says "magazine".
Smaller chunks do not fix this case, and there is no other DILUTION-classifiable case to
fix.

**Query-side seed_k also ruled out (free probe on cached stores):** sweeping
`seed_k ∈ {10,16,20,30,50}` brings **neither** gold into the pool at any value — the emb
ranks (59/202) never clear even a seed_k=50 gate, and the BM25-18 false-friend chunk
personalizes PPR toward plant-styling text, not the magazine mention (PPR rank stays 201).

## 6. Bucketed fix menu, by cause and cost

| bucket | cases | count | cheapest fix | cost class | expected recovery |
|---|---|---|---|---|---|
| **Content mismatch / benchmark artifact** | `4dfccbf8 _2` | 1 | none — correctly deprioritized, Q already correct | — | 0 questions |
| **Composite-doc poverty / entity-bridge** | `1a8a66a6 _2` (+ `_4`) | 1 q (2 sessions) | extraction-side: emit a typed concept/tag (`magazine`/`periodical`) on periodical entities, or a distilled surface ("Architectural Digest — magazine subscription"), so BM25 and the embedding can bridge the proper noun to the category term | **full re-ingest** (extraction change) | ≤1 question, *and only if* `_4` also clears the same fix and then survives to context (it is currently in-pool-below-context) — payoff uncertain |
| **Chunking geometry (dilution)** | — | 0 | — (measured 0/11) | — | 0 questions |
| **Query-side (seed_k / pool trim / expansion)** | — | 0 | — (seed_k→50 recovers neither; pool trim already includes the paid-run specimens at HEAD) | free to test, tested | 0 questions |

## 7. Recommendation (ranked by questions-recovered ÷ cost)

1. **Do not pursue an out-of-pool fix as a recall lever.** At HEAD the class is 2 sessions
   / 1 question, and that one question (`1a8a66a6`) needs a *compound* fix (both `_2`
   out-of-pool **and** `_4` in-pool-below-context must be corrected). The only relevant
   fix is extraction-side and requires a full re-ingest for a ceiling of one question.
   Cheapest-per-question is **zero** — nothing here justifies a paid run.
2. **The recall action has moved.** Out-of-pool was 13 golds in the paid run and is 2 now;
   the pool improved on its own. What remains out-of-*context* concentrates in the
   **in-pool-below-context** class (14 gold sessions across the 15 missing-context
   questions) — which Round 4 already built a rule for (`seed_promote`) and found **null**.
   The residual is chunk-level identity within already-covered sessions (retargeting's job),
   not pool recall.
3. **If extraction is ever revisited for another reason**, the entity-typing idea
   (type periodicals/products with a category concept so proper nouns bridge to category
   query terms) is a *general* composite-doc-poverty remedy worth folding in — but spec it
   as part of a re-ingest that also serves distilled preference/frequency lines (Round 1
   §5.4, sharp-edge #6), **measured before promotion**, not run speculatively for this one
   question.

**Spec (not run — needs re-ingest):** in `kg/extractors.py`, when an entity is recognized
as a periodical/product/brand, emit a category tag (`magazine`, `newspaper`, …) onto its
mention surface so it enters `Seeder._episode_doc` (raw text + title + description +
entity/**tag** surfaces). Re-ingest the 100 LongMemEval instances (`extractor_backend=cue_gated,
event_facts=true, ingest_date_filter=true`, KG_LLM=openai) — this re-runs extraction (paid)
— then re-run `scripts/offline_eval_round5.py` against the new stores and check whether
`1a8a66a6 _2`/`_4` clear the seed cutoff and reach context. Do not promote on the offline
pool-membership signal alone; confirm the answer flips on a single paid probe of that one
question before shipping.

---

# Round 6a: answer-time aggregation

*2026-07-19. Not an offline A/B sweep — this round SHIPS two default-off, query-side
knobs in the answer flow (kg/rag.py only) and lands them behind tests; the paid
confirmation command is printed at the end (NOT run). No paid calls were made building
this; all behavior is exercised with scripted fake OpenAI clients (tests/test_agg.py),
keys stripped exactly as the Round-3 harness does.*

## 1. Motivation

Six questions in the paid run (`runs/sample-datefix-events-1/run.json`) fail with ALL
gold evidence in context (`gold_marks[*].in_context == true`, recall 1.0) — the single
reader call miscounts. Enumerated from the run (aggregate-shaped ∧ all-gold-in-context ∧
`judge.correct == false`):

| id | kind | lane | n_sess | expected | question |
|---|---|---|---|---|---|
| `0a995998` | multi-session | multihop | 44 | 3 | how many items of clothing to pick up/return |
| `1c549ce4` | multi-session | multihop | 45 | $140 | total cost of car cover + detailing spray |
| `2318644b` | multi-session | multihop | 41 | $270 | how much more per night, Hawaii vs Tokyo |
| `09ba9854_abs` | multi-session | multihop | 50 | *not enough info* | how much will I save taking the bus vs taxi |
| `370a8ff4` | temporal-reasoning | **state** | 46 | 15 | how many weeks since flu when I did my 10th jog |
| `982b5123` | temporal-reasoning | **state** | 47 | *five months ago* | how many months ago did I book the SF Airbnb |

The fix moves the arithmetic out of the model and into CODE — extract the items from the
context (which is complete), count/sum in Python, and let the model only phrase. Never
state a number the code didn't compute or the enumeration doesn't support.

The last two are **date arithmetic**, not occurrence tallies: they match
`is_aggregate_question` via "how many weeks/months" but route STATE (route.py splits them
off with `_DATE_ARITH` *before* the aggregate check). Both knobs deliberately exclude them
(§2), so the mechanism targets the **four multihop count/sum** questions. `2318644b` is a
*difference*, not a plain sum — the computed item table feeds the reduce, but the final
subtraction is still the model's; count it as a partial-fit case. `09ba9854_abs` is the
abstention canary — the saving can't be computed (bus fare never stated), so the correct
behavior is "not enough information", which the empty-list path must preserve.

## 2. What changed (code) — kg/rag.py answer flow only

| piece | where | knob |
|---|---|---|
| **events[] reconciliation**: after the normal single answer call, when the question is aggregate-shaped (`completeness.is_aggregate_question`), recompute the aggregate from the reader's OWN enumerated `events[]` (already returned on multihop/state lanes by `rag_answer_events="lanes"`): `question_shape=="count"` → `len` of events deduped on `(date, normalized description tokens)`; `"sum"` → parse amounts (`completeness.find_amounts_in_text` + an explicit amount/quantity field) and add in Python. Compare to the first numeric stated in the answer text; on mismatch **append** a correction sentence citing the enumeration ("…the enumerated events list N matching item(s): …"). Match / empty events / non-aggregate → untouched. **Date-arithmetic questions are skipped** by reusing `route._DATE_ARITH` (no regex duplication) so a "how many weeks" difference is never miscounted as a tally. | `OpenAIAnswerer._reconcile_answer`, called at the tail of `answer()` | `agg_reconcile` (default False) |
| **map-reduce aggregation lane**: when the routed lane is MULTIHOP ∧ `is_aggregate_question`, before the answer call: **(i) MAP** — one forced `list_items` call PER SOURCE SESSION (context chunks grouped by `eid.split("#")[0]`), returning `items[]={date,description,amount?,verbatim_quote}`, instructed to list only THIS session's matches (empty if none); **(ii) CODE** — merge, dedup `(date, normalized description tokens)`, count or sum in Python; **(iii) REDUCE** — the normal answer call, with the original context blob PLUS a `--- COMPUTED AGGREGATION ---` table + computed number appended to the user message, instructed to answer from the computed aggregate unless the episodes contradict it. Map calls reuse the answerer client/backoff and are metered (`meter.record("rag.map", …)`); the reduce inherits all of the existing length-retry / extractive-fallback machinery (it IS the single answer call, only the user content differs). **Abstention safety**: an empty merged list appends an *insufficiency* note, not a fabricated number — a computed 0 is offered as an answer only for "how many" (count) shape; sum/other are told to say the info is insufficient. | `OpenAIAnswerer._agg_map_reduce` / `_agg_map_call` / `_session_text`, wired into `answer()` before the answer call | `agg_map_reduce` (default False) |

Both knobs are **query-side** — added to `kg/config.py` Tier 2, and deliberately NOT in
`kg/ingest_cache.INGEST_RELEVANT_FIELDS`, so every cached store stays valid. No retrieval,
context-building, `kg/store.py`, or write-path code was touched.

**Design decisions / notes**
- The reduce reuses `self._answer_tool(result)`, which on the MULTIHOP lane is the
  events-enabled schema — so when both knobs are on, reconcile can still audit the reduce
  call's own enumeration. The two knobs are independent and compose.
- Dedup normalization (`_norm_desc_tokens`): lowercase → drop `_RETARGET_STOP` stopwords →
  singularize a single trailing 's' (`visits`→`visit`, leaving `class` alone) → drop ≤2-char
  tokens. Catches the same occurrence re-mentioned in a later session (very common in
  LongMemEval's dated logs) without a stemmer.
- `_stated_count` takes the first integer in the answer, `_stated_sum` the first monetary
  amount (`find_amounts_in_text`) — reconcile is a no-op when the answer states no number to
  audit (nothing to contradict), which is the conservative choice.
- Byte-identical when off is structural: `reduce_addendum` is `""` unless
  `agg_map_reduce` fires, so the user message is `blob + "" == blob`; reconcile early-returns
  the answer unchanged unless `agg_reconcile` fires on an aggregate question with events.

## 3. Tests (tests/test_agg.py — fake-client, offline, $0)

15 tests, all green:
- **reconcile**: corrects a count mismatch; leaves a matching count untouched; dedups a
  re-mentioned occurrence before counting; corrects a sum mismatch; **skips date
  arithmetic** ("how many weeks…"); end-to-end `ask()` appends the correction and a note;
  **leaves a non-aggregate question untouched**.
- **map-reduce**: MAP fires exactly once per source session and every map call carries the
  `list_items` schema; a re-mentioned item dedups to 1; the Python sum matches the expected
  $140; the **empty-map abstention path** emits no fabricated count and an insufficiency
  escape hatch; end-to-end `ask()` feeds the `COMPUTED AGGREGATION` table into the reduce
  call's context.
- **pure CODE**: `_dedup_events` (date + normalized tokens), `_sum_events` (amount from
  field or description).
- **both knobs off = byte-identical**: an end-to-end multihop-aggregate `ask()` makes
  exactly ONE LLM call (no map calls), the reader sees exactly `context_text`, and the
  answer is the model's verbatim output with no correction and no `agg_*` note.

Test files run: `tests/test_agg.py` (15 passed), `tests/test_rag.py` +
`tests/test_event_facts.py` (59 passed). Full suite: **360 passed, 0 failed** (up from 361
in Round 4 — the count differs because the suite has evolved since; no failures, and the
Round-1/2 pre-existing failure remains fixed).

## 4. Open risks

- **Map-call cost scales with sessions-in-context**: one MAP call per distinct source in
  the context (≤ `rag_context_episodes`=5 sources in practice, but expansion can add
  siblings of the same source — those collapse to one call since grouping is by base id).
  Worst case ≈ 5 small calls + 1 reduce per aggregate multihop question. Metered under
  `rag.map`; watch the per-question token line on the paid run.
- **`2318644b` is a difference, not a sum** — map-reduce computes the total; the final
  subtraction is still the reduce model's job. The item table should help, but this one may
  not flip.
- **`_stated_sum`/`_stated_count` are first-numeric heuristics** — an answer that leads with
  a year/date before the count could be audited against the wrong number. Mitigated for
  sums (money-shaped match only); the count path is genuinely first-integer.
- **Reduce can still be overruled by the model** — the reduce prompt says "unless the
  episodes contradict it", by design (so a bad map extraction can't force a wrong number),
  which also means a correct computed number can be talked out of. If the paid run shows the
  reduce ignoring correct computations, tighten the reduce instruction.

## 5. Paid validation command (NOT run)

Sample tier, cached stores (query-side knobs don't change the ingest-cache key, so no
re-extraction), the run's usual query-side `--set` string plus the two new knobs:

```
python -m kg testrun --mode per-instance --tier sample \
  --set history_all_lanes=true --set event_facts=true \
  --set rag_retarget=ce --set rag_provenance_promote=true \
  --set mmr_lambda=1.0 --set rag_parent_expand=2 --set rag_chunks_per_source=2 \
  --set agg_reconcile=true --set agg_map_reduce=true \
  --label agg-round6a-1 --out runs
```

(Most of the query-side knobs above are already the config defaults; they are set
explicitly to match `runs/sample-datefix-events-1` exactly. `KG_LLM=openai` /
`OPENAI_API_KEY` must be live. Expected cost ≈ **$0.05–0.15** for the sample tier — the
map calls add a handful of small calls to the ~4 multihop aggregate questions.)

**Six question ids to watch** (compare `judge.correct` against the paid baseline):
`0a995998`, `1c549ce4`, `2318644b`, `09ba9854_abs` (the four multihop count/sum targets —
`09ba9854_abs` must STAY abstained), and `370a8ff4`, `982b5123` (the two date-arithmetic
STATE questions — both knobs exclude them, so they must be UNCHANGED, i.e. no new
regression). Do not run this here.

---

# Round 6b: facts projection + tally evidence

*2026-07-19. Two default-off, query-side knobs that make the graph's fact/occurrence data
(a) queryable as SQL and (b) visible to the reader as **clearly caveated corroboration** —
without ever becoming a second source of truth. Occurrence capture at ingest is ~47%
complete (completeness tier2), so graph-side counts must be EVIDENCE, never an oracle. No
paid calls building this; both knobs exercised on scripted stores and on the CACHED stores
of `runs/sample-datefix-events-1` (keys stripped on import, `$0`). Touches `kg/store.py` and
`ContextBuilder` only — no answer-flow, retrieval, or write-path semantics (independent of,
and non-conflicting with, Round 6a's `agg_reconcile`/`agg_map_reduce`).*

## 1. What changed (code)

| piece | where | knob |
|---|---|---|
| **Relational projection**: on `GraphStore.flush()`/`save()`, (re)generate two derived SQLite tables in the same db file, rebuilt **WHOLESALE** from the RELATED_TO edges every flush (no incremental maintenance). `facts_view(src_name, rel_name, dst_name, valid_from, valid_to, event, confidence, belief, episode_id, mentions)` — one row per RELATED_TO edge (open, closed, or retracted; `belief` distinguishes), names resolved through node payloads, `mentions = 1 + len(confirmed_by)`. `agg_view(src_name, rel_name, dst_name, n_occurrences, first_date, last_date)` — grouped over BELIEVED (non-retracted) rows only, `n_occurrences = SUM(mentions)` across the parallel edges of a pair, first/last = earliest/latest non-empty `valid_from`. The LOAD path (`_load_rows`) never reads either table. | `kg/store.py` `GraphStore._rebuild_facts_projection`, called at the tail of `save()` | `facts_projection` (default False) |
| **Graph-tally context lines**: when on and the question is aggregate-shaped (REUSES `completeness.is_aggregate_question`), `ContextBuilder.build` appends `"GRAPH TALLIES (may be incomplete; verify against the episodes):"` + per-pair occurrence tallies for the question's anchor entities — `me --went_to--> the park: 5 occurrences (2025-01-05 -> 2025-05-18)` — capped at 10 lines, ordered by count desc. Computed IN-MEMORY from the believed RELATED_TO edges among the anchors (group parallel edges by `(src, rel_tag, dst)`, count `= Σ 1+len(confirmed_by)`), so it works WITHOUT the projection tables and on read-only stores. | `kg/rag.py` `ContextBuilder._graph_tallies` + `build` | `agg_evidence` (default False) |

Both knobs are added to `kg/config.py` Tier 2 and deliberately **NOT** in
`kg/ingest_cache.INGEST_RELEVANT_FIELDS`, so every cached store stays valid (test asserts
both are absent). Byte-identical when off is structural: `save()` early-returns the same
no-op when nothing is dirty and `facts_projection` is off (no extra tables ever appear); the
tally block only appends, and only when `agg_evidence` fires on an aggregate question with
≥1 anchor tally.

## 2. Schema and design decisions

- **Wholesale rebuild, not incremental.** Every flush DROPs and recreates both tables from
  the live edge set. Rationale: (i) a fact's identity is bi-temporal and mutates in place
  (confirm widens `valid_at`, close sets `invalid_at`, supersede/retract flip state,
  canonicalization renames endpoints) — an incremental projection would need to mirror every
  one of `apply_fact`'s seven actions and every canonical rename, doubling the write-path
  surface it exists to *observe*; (ii) the projection is a derived read-model, so a
  from-scratch rebuild is the one implementation that can never drift or leave a stale row.
  The **stale-row test** proves it: flush pair A→park, then add A→gym and reflush — the
  tables show exactly the two current edges, no ghost of the first flush. Cost is O(fact
  edges) per flush; on these stores (≤ a few thousand fact edges) that is a handful of ms,
  dwarfed by extraction, and the ingest loop already flushes on a coarse cadence
  (`ingest_flush_every`).
- **`mentions` vs `n_occurrences`.** A single `(src,rel,dst)` pair can have several *parallel*
  edges (distinct dated occurrences — the repeatable-predicate / event path) each carrying its
  own confirm count. `facts_view.mentions` is per-edge (`1 + len(confirmed_by)` — same-key
  re-mentions); `agg_view.n_occurrences` sums `mentions` across the pair's parallel edges, so
  five dated park visits read as `n_occurrences=5` whether they are five `[d,d]` event edges
  or one confirm-collapsed edge with four `confirmed_by`.
- **belief in `facts_view`, believed-only in `agg_view`.** `facts_view` keeps retracted rows
  (with `belief='retracted'`) so the SQL view is complete/auditable; `agg_view` filters to
  `belief='asserted'` so a never-true correction can never inflate a tally. Mirrors
  `facts.py::_believed` (retracted ≠ ended).
- **Load-path independence.** `_load_rows` selects only `nodes/edges/vectors/cache` — never
  the views. Tested directly: build with the knob on, DROP both tables, reload → graph
  snapshot (nodes + edge keys) byte-identical to a reload that kept them. A restored cache
  store (which never had the tables, since it was built knob-off) gets them created cleanly on
  its first knob-on flush — tested both with a fresh mutation and in the nothing-dirty
  reflush path (the knob-on idle `save()` still materializes the tables, unlike the knob-off
  idle no-op).
- **Tallies computed in-memory, not read from `agg_view`.** The reader-facing tallies
  deliberately re-derive from edges rather than query `agg_view`, so `agg_evidence` is
  independent of `facts_projection` and works on a read-only copy. The two paths share the
  same grouping definition (parallel edges → `Σ 1+len(confirmed_by)`, believed-only) so the
  numbers agree, but neither depends on the other.

## 3. Tests (`tests/test_facts_projection.py` — 14, offline, `$0`)

Projection: absent when off; created on flush when on (facts_view 1-row-per-edge, agg_view
grouped with correct `n_occurrences`/first/last); `mentions` counts confirmations; **wholesale
rebuild drops stale rows**; **retracted edges present in `facts_view` but excluded from
`agg_view`**; **load-path independence** (delete tables → reload → identical graph);
**restored-cache first-flush** creates tables cleanly; **projection-only reflush with nothing
dirty** still materializes them; knobs absent from `INGEST_RELEVANT_FIELDS`. Tallies: present
+ correctly ordered (park 5 before gym 2) when on; **absent when off and the on-blob is
append-only** (`on.startswith(off)`); absent for a non-aggregate question; **capped at 10
lines** (15 destinations → top-10 by count, `place00: 15 occurrences`, `place14` cut);
retracted pair never tallied. Full suite: **375 passed, 0 failed** (composes with Round 6a's
`tests/test_agg.py`; no pre-existing failures).

## 4. Harness — `scripts/offline_eval_round6b.py` (Round-3 pattern, cached stores)

Same offline harness as Rounds 3–5: the 100 cached per-instance stores of
`runs/sample-datefix-events-1` copied to temp paths, the exact
`HybridRetriever.retrieve → ContextBuilder.build` read path driven with the run's query-side
config, keys stripped on import. Each question's context is built TWICE — `agg_evidence`
OFF (baseline) and ON — recording tally presence/lines, the gold session in-context marks
under each, and OFF-vs-run.json fidelity.

**Fidelity gate:** the OFF baseline reproduces the paid run's recorded `gold_marks.in_context`
**184/184** — the tally feature's baseline IS the paid run's retrieval.

| metric (n=100) | value |
|---|---|
| tally section present, `agg_evidence` **OFF** | **0** (byte-identical-off holds on real data) |
| tally section present, `agg_evidence` **ON** | 56 (the aggregate-shaped questions with ≥1 anchor edge; 44 non-aggregate/anchorless get nothing) |
| questions whose **gold-in-context marks changed** vs OFF | **0** (append-only: no canary regresses) |
| questions whose **context episode set changed** vs OFF | **0** |
| mean context chars added (all 100 / the 56 that fire) | +349 / ≈+625 |
| mean tally lines when present | 8.6 (cap 10) |

**All six Round-6a failure questions gain a tally section** (each hits the 10-line cap;
`marks_same=True` for every one — the section is purely additive):

- **0a995998** (+693 ch) "how many items of clothing to pick up/return": tallies are
  reselling-advice relations (`eBay --selling--> items`, `items --focus--> music boxes`,
  `notes app --tracks--> pickups and returns`) — none is the clothing count.
- **1c549ce4** (+718 ch) "total cost of car cover + detailing spray":
  `detailing spray --purchased_from--> Amazon: 2×`, `waterproof car cover --claim--> car
  cover`, `Chemical Guys Car Care Kit --included--> detail spray` — the *entities* are right,
  but the $120/$20 amounts live on QUANTITY nodes, not as tally counts.
- **2318644b** (+676 ch) "how much more per night, Hawaii vs Tokyo":
  `Tokyo --located_in--> Shibuya Crossing: 2×`, `Tokyo --explore--> Walking`,
  `Tokyo --has--> transportation` — travelogue relations, no nightly rate.
- **09ba9854_abs** (+691 ch, the abstention canary): `Airport Limousine Bus --cost-->
  $10-$20`, `--cost--> 2000 JPY`, `--cost--> 3000 JPY` — the assistant's *generic* fares,
  exactly the figures the correct answer must NOT compute a saving from. The caveat header +
  the answer prompt's user-vs-assistant rule are what keep this abstained.
- **370a8ff4** (+641 ch, date-arith STATE): `flu --recovered_from--> exercise: 2×`,
  `User --completed--> 10th jog (2023-04-10)`, `jogging --introduce--> walking` — the tally
  section fires (it is "how many weeks"-shaped) but carries no week count; the arithmetic is
  Round 6a's job and both its knobs exclude date-arith.
- **982b5123** (+708 ch, date-arith STATE): `San Francisco --located_in-->
  Haight-Ashbury: 2×`, `Airbnb --located_in--> San Francisco`, `San Francisco --help-->
  wine` — Airbnb booking relations, no month count.

## 5. Open risks — why the caveat header is load-bearing

The six diffs above are the whole argument. At ~47% occurrence capture the tally lines are
**dominated by extraction-noise relations** (`flu --recovered_from--> exercise`,
`Tokyo --explore--> Walking`, `San Francisco --help--> wine`) — predicate mislabelings and
generic-fact captures that read as confident counts but answer a *different* question than
the one asked. In several cases the tallies are not just unhelpful but adversarial: 09ba9854's
top tallies are the assistant's generic bus fares, the exact operands the abstention answer
must refuse. So:

- The header **"GRAPH TALLIES (may be incomplete; verify against the episodes)"** is not
  decoration — it is the contract that keeps the reader treating these as corroboration to
  check against the EPISODES, never as the count. It must convey incompleteness; wording that
  reads as authoritative would actively harm the abstention and difference cases.
- **A tally is never an oracle here.** Where a count *is* right (e.g. the park/yoga
  occurrence machinery on the synth store) the tally corroborates; where capture is partial or
  mislabeled it is visibly one of ten hedged lines the reader can discount. The value is the
  *distribution* (which pairs recur, over what span), not any single number.
- **Ordering by raw count amplifies whatever capture produced.** A frequently-mislabeled
  predicate can top the list. The 10-line cap bounds the blast radius (mean +625 chars) but
  does not clean the signal; a future improvement is to rank tallies by anchor/query relevance
  (as Round 1 §5.2 argued for `facts_for`) rather than raw frequency — deferred, since it needs
  the same query-similarity scoring C measured as neutral-to-negative.

Net: knob 1 (projection) is safe SQL infrastructure with zero read-path effect and a
byte-identical-off guarantee; knob 2 (tallies) is append-only, regresses no canary on 100 real
questions, and is honestly framed as incomplete. Neither is promoted here.

## 6. Paid validation (rides along with Round 6a; NOT run)

No paid validation is needed for knob 1 (no read-path effect; fully offline-testable). Knob 2
rides Round 6a's command — add `--set agg_evidence=true` to the block in §Round-6a-5:

```
python -m kg testrun --mode per-instance --tier sample \
  --set history_all_lanes=true --set event_facts=true \
  --set rag_retarget=ce --set rag_provenance_promote=true \
  --set mmr_lambda=1.0 --set rag_parent_expand=2 --set rag_chunks_per_source=2 \
  --set agg_reconcile=true --set agg_map_reduce=true \
  --set agg_evidence=true \
  --label agg-round6a6b-1 --out runs
```

Watch the same six ids: with `agg_evidence` on, their contexts gain the tally sections
measured above and NO canary's gold-in-context regresses (offline: 0/100). The paid question is
only whether the reader is *helped* by corroboration or *misled* by noise — which is why the
header wording is the thing to check first if any of the six regress. Do not run this here.

---

# Round 7a: fact-line vectors

*2026-07-19. The STORAGE layer for statement-granularity retrieval — one default-off,
ingest-side knob (`fact_vectors`) that embeds each fact as a first-class retrieval target.
No paid calls: the embedder is the local bge-small (kg/embedders.py); this round adds no
LLM call site. Touches kg/fact_vectors.py (new), kg/config.py, kg/ingest_cache.py,
kg/store.py, kg/ingest.py, kg/graph.py, kg/cli.py; tests in tests/test_fact_vectors.py.
No retrieval behavior changes — the Seeder is untouched; a follow-up builds the lane that
consumes these vectors.*

## 1. Motivation (Round 4 §3–4, the 06f04340 trace)

Retrieval seeds on composite-episode topical density; gold evidence is often a single
answer-bearing statement made in passing, structurally unfindable at chunk granularity and
diluted by its surrounding text. Round 5 §5 measured the failure geometrically: for
`1a8a66a6 _2`, **0/11 sub-chunks at any granularity clear the seed cutoff** — "Architectural
Digest … home decor" is simply not near "magazine subscriptions" in embedding space because
the sentence never says "magazine". The dilution is the *company the statement keeps*. A
fact line's text — `me subscribes_to Architectural Digest` — is the content undiluted, so
making the fact itself an embeddable target is the structural fix (sharp edge #6). This
round builds the vectors; it does not yet seed on them.

## 2. What changed (code)

| piece | where | knob |
|---|---|---|
| **Surface rendering**: `statement_surface(store, src, dst, data)` = `"<src> <rel> <dst>"`, names resolved by REUSING `FactLine.from_edge` (the same resolver context/CLI render through) — no dates, ids, or window grammar. One surface per believed RELATED_TO edge; parallel dated occurrences of a pair render the SAME text and dedupe. `current_surfaces(store)` returns the statement set plus the distilled aggregates. | kg/fact_vectors.py | — |
| **Distilled aggregates**: per `(src, rel_tag, dst)` group with `n_occurrences > 1`, a second surface `"<src> <rel> <dst> N times from <first> to <last>"` (span omitted when the group is undated). REUSES Round 6b's tally grouping definition verbatim: `n = Σ (1 + len(confirmed_by))` across parallel edges, believed-only, first/last = earliest/latest non-empty `valid_at`. | kg/fact_vectors.py `current_surfaces` | — |
| **Ingest-time batch embed**: after fact writes + canonical renames + derived edges (new step 5b, once per batch), `sync_fact_vectors(prune=True)` embeds the MISSING surfaces in one `embedder.embed` batch (the §3 pattern) and reconciles the kind="fact" index. Incremental (embeds only new/changed surfaces), and prunes surfaces orphaned by a rename/retraction. `IngestReport.fact_vectors` counts embeds. | kg/ingest.py | `fact_vectors` (default False) |
| **Ingest-cache membership**: `fact_vectors` ADDED to `INGEST_RELEVANT_FIELDS` (it changes what ingest WRITES), hashed **only when ON** via the generalized `_HASH_ONLY_WHEN_ON` set (the same back-compat pattern `event_facts`/`ingest_date_filter` use). Off = digest byte-identical to pre-knob → every existing cached store stays valid. | kg/ingest_cache.py | — |
| **Vector-row deletion**: `GraphStore.remove_vector(kind, node_id)` drops the in-memory vector AND schedules the SQL row for deletion on flush (new `_deleted_vectors` dirty set; there is no on_remove hook). Byte-identical when unused: fact-off never calls it, so the set stays empty and `save()` is unchanged. | kg/store.py | — |
| **Backfill**: `KnowledgeGraph.backfill_fact_vectors()` / `kg backfill-fact-vectors` CLI = `sync_fact_vectors(prune=False)` — additive, idempotent, incremental (only missing surfaces). Touches ONLY the vectors table, so it CANNOT change the ingest-cache key: the 100 cached benchmark stores gain fact vectors in place, $0, no re-extraction. | kg/graph.py, kg/cli.py | — |

## 3. Vector-kind design (the keying choice, documented)

**One vector kind, `"fact"`; keyed by SURFACE HASH; two families by id namespace.**

- The `vectors(node_id, kind)` table takes any stable string id, with no flag column. So
  the two families share `kind="fact"` and are distinguished by the id PREFIX:
  `fact:<sha256(surface)[:16]>` for statements, `factagg:<…>` for aggregates ("same vector
  kind, flagged distinct"). The seeder can partition hits by prefix.
- **Hash the surface text, not the edge.** This is what the prompt's dedup rule
  ("parallel dated occurrences share a surface") *requires*: five park-visit edges produce
  one surface → one id → one vector. Keying by edge would store five identical vectors.
  The surface→id map is deterministic and re-derivable from the graph at query time
  (`current_surfaces`), so a vector hit maps back to its fact(s) with no stored side-table.
- **Rename handling falls out for free.** A surface's text changes only when
  canonicalization renames an endpoint. Because the id IS the surface hash, a rename yields
  a NEW id (embedded) and orphans the OLD id (pruned by `sync_fact_vectors(prune=True)` on
  the next ingest flush). This is the honest cost the design rule names; the reconciliation
  counts it (`removed`). Backfill deliberately does NOT prune (additive-only is what makes
  it safe to run against a cache), so a renamed-then-backfilled store keeps a stale vector
  until its next ingest-side sync — acceptable, since backfill's job is to fill gaps, not
  garbage-collect.
- **Load-path independence.** `_load_rows` reads all vectors uniformly; fact vectors carry
  synthetic ids that match no `nodes` row, so node deletion (`WHERE node_id=?`) never
  touches them. Off = no `"fact"` kind ever appears.

## 4. Backfill cost (measured, one real cached store)

`kg backfill-fact-vectors` on a **copy** of `store/cache/001be529-*.db` (19,929 edges;
2,845 open + 39 closed = 2,884 believed facts), real bge-small, model warm:

| metric | value |
|---|---|
| surfaces embedded | **2,931** (2,870 statement + 61 aggregate) |
| statement dedup | 2,884 believed edges → 2,870 statement surfaces (14 parallel-occurrence collapses) |
| aggregates | 61 recurring `(src,rel,dst)` pairs (`n_occurrences > 1`) |
| wall time (warm embedder) | **4.18 s** ($0 — local CPU) |
| second run (reopen) | +0 / −0 (idempotent) |

Extrapolated: ~7 min of local CPU to backfill all 100 cached stores, one-time, no paid
calls, no cache invalidation.

## 5. Tests (tests/test_fact_vectors.py — 16, offline, $0)

Surfaces deterministic + name-resolved + dateless; parallel occurrences dedupe to one
statement surface; aggregates only for `n_occurrences > 1` (dated span + undated no-span +
confirm-count cases); sync stores both families under kind="fact" by id prefix; **batch
embed at ingest only when the knob is on** (becky_stream end-to-end, on vs off) with
`report.fact_vectors` set; **rename-on-merge re-embeds** the new surface and prunes the old
(persisted as a DELETE); **backfill idempotent + incremental + purely additive (no prune)**;
**backfill on a COPY of a real cached store** (added>0, idempotent on reopen; skips if no
cache present); **knob off = no "fact" kind and no extra writes**; **cache-key back-compat**
(`fact_vectors` in `INGEST_RELEVANT_FIELDS`, off digest unchanged, on re-keys); load
round-trip preserves fact vectors. Full suite: **391 passed, 0 failed** (375 + 16).

## 6. Open risks

- **No retrieval evidence yet.** This round only proves the vectors are correct, deduped,
  and cheap to produce/maintain. Whether seeding on them recovers the dilution class
  (Round 5's `1a8a66a6`, the 06f04340 trace) is the FOLLOW-UP's measurement — the Seeder is
  untouched here, so there is deliberately zero A/B delta to report.
- **Aggregate surfaces inherit the tally-noise risk (Round 6b §5).** At ~47% occurrence
  capture, a distilled `"X rel Y N times …"` can be a confidently-worded mislabel
  (`flu recovered_from exercise 2 times`). As a *retrieval target* that is less dangerous
  than as a reader-facing count (a bad target costs a wasted seed slot, not a wrong answer),
  but the seeder that consumes these should treat aggregate hits as evidence, not oracle —
  the same contract as `agg_evidence`.
- **Surface ≠ query vocabulary.** The fix presumes the fact's *predicate/endpoint* names
  bridge to the query where the episode text didn't. For `1a8a66a6` specifically the bridge
  still fails at the entity layer ("Architectural Digest" is never typed a *magazine*), so
  fact-line vectors alone won't seat it — that case needs the Round 5 §6 extraction-side
  entity-typing too. Fact vectors are necessary infrastructure for the disposition/passing-
  mention class, not a complete fix for every out-of-pool gold.
- **Backfill doesn't prune.** A store canonicalized-then-backfilled (not re-ingested) keeps
  stale renamed-surface vectors. Harmless for recall (extra targets, never wrong answers),
  but a store that has drifted far should be re-synced ingest-side, not just backfilled.

---

# Round 7b: fact lane

*2026-07-19. The retrieval lane that CONSUMES the Round-7a vectors: statement-granularity
seeding feeding the existing episode pipeline. One default-off, QUERY-SIDE knob (`fact_lane`).
Scores the query against the kind="fact" vectors, maps the top hits back to their provenance
CHUNKS + endpoint ENTITIES, and merges those ADDITIVELY into the seed set — a fact's asserting
chunk enters the PPR pool because its CLAIM matched, not because its surrounding prose did.
Touches kg/config.py, kg/fact_vectors.py (`fact_provenance`), kg/retrieval.py (Seeder.fact_seed
+ PPRRetriever._merge_fact_lane), kg/rag.py (FACTS [matched] marking + provenance-promote
`force_ids`); tests in tests/test_fact_lane.py. Harness scripts/offline_eval_round7b.py drives
the same cached-store read path as Round 3, plus a synthetic needle store. No paid calls, no
commits. Verdict up front: **SHIP-candidate — 4 wins / 0 regressions on 100 real questions,
fidelity 184/184, knob-off byte-identical.***

## 1. Motivation (Round 4 §3, the 06f04340/0977f2af traces)

Round 4 built a post-ranking reorder (`seed_promote`) for the "gold in pool, below prefix"
class and found it EMPTY at the shipped config: the two live failures each failed a gate —
`0977f2af`'s gold `_1` is **out-of-k** (pool rank 9 at α=1.0), `06f04340`'s gold is
**out-of-seed-top** (seed rank 11 behind ten distractor chunks). Both are RECALL failures a
reorder cannot touch: the gold is a single answer-bearing statement diluted by off-topic prose,
so the chunk never seeds. Round 7a made each fact an embeddable target; this round seeds on
them, so the *claim* — `me used_for Air Fryer sweet potato fries` — carries its chunk into the
pool even when the chunk text is about something else.

## 2. What changed (code)

| piece | where | knob |
|---|---|---|
| **Vector→graph resolver**: `fact_provenance(store, hit_ids)` maps a `fact:`/`factagg:` vector id back to `{episodes, entities, stmt_surface}` in one walk of the believed edges, recomputing each surface's id exactly as `current_surfaces` does (no side-table). A statement hit unions ALL its deduped parallel occurrences' provenance; an aggregate hit unions its whole `(src,rel,dst)` group's. | kg/fact_vectors.py | — |
| **Seeder fact lane**: `Seeder.fact_seed(query)` scores the query embedding against the kind="fact" index (top `fact_lane_k=10`), resolves the hits, and returns `{node_id: cosine}` over each hit's provenance chunks + endpoint entities (max cosine per node) plus the matched statement surfaces. Embedding-only — BM25 over 3-token surfaces is degenerate (idf collapses on near-duplicate short strings), so it is deliberately omitted. Empty on an un-backfilled store (lane no-ops). | kg/retrieval.py | `fact_lane_k` |
| **Additive merge**: `PPRRetriever._merge_fact_lane` adds ONLY nodes the episode lane did not already seed, so every episode-lane seed keeps its EXACT score; total added mass is scaled to ≤ `fact_lane_weight=0.5` × the episode-lane mass so noisy ~gpt-4o-mini facts cannot out-vote the episode lane in the PPR personalization. **SEED-MASS route, not pool injection** (judgment call, §3). | kg/retrieval.py | `fact_lane` (default **off**, QUERY-SIDE, NOT in INGEST_RELEVANT_FIELDS), `fact_lane_weight` |
| **Context**: matched lines render in FACTS marked `[matched]`; the matches' provenance chunks ride the EXISTING `rag_provenance_promote` path via a new `force_ids` argument — promoted UNCONDITIONALLY (the lane already judged them relevant by cosine), because the term-overlap gate would miss a fact whose match lives in the PREDICATE ("blood type") not an endpoint name. Promotion still only ever displaces an expansion sibling, never an originally-selected chunk. | kg/rag.py | keyed off `result.fact_matched` |

Knob off ⇒ `_merge_fact_lane` never runs, `result.fact_matched` is `{}`, `force_ids` is None,
and FACTS renders no mark: seeds / pool / context are byte-identical (tested).

## 3. Design decision — seed-mass route vs explicit pool injection

The provenance chunk enters the pool by being **seeded** (it gets PPR restart mass ≥ (1−α)·p_i,
so it always surfaces in the candidate pool), NOT by being force-appended to `cand_ids`. Reasons:
(i) it reuses the whole existing PPR→MMR→CE stack, so the needle competes on merit and the CE
can still demote a bad match — where Round 4's *forced* prefix injection produced the 36b9f61e
answer-chunk regression and the 2788b940 gold displacement; (ii) it populates `seed_scores`, so
chunk-retargeting can also use the signal; (iii) seeding the endpoint ENTITIES lets diffusion
corroborate. The mass cap + idf-weighting (the `me` hub gets low idf) throttle noisy/hub facts.
The cost: on a SINGLE-CHUNK store (no expansion sibling to displace) provenance-promote cannot
seat the needle and the lane relies on seed-rank alone — see §5.

## 4. REAL sweep (n=100, cached stores, Round-3 config + fact vectors backfilled)

Each cached store is COPIED and `backfill_fact_vectors()`'d in place ($0 local bge, additive —
the ingest-cache key is untouched), then the read path is run `fact_lane` OFF vs ON.

**Fidelity gate first**: at `fact_lane=0` the harness reproduces the paid run's recorded gold
in_context marks **184/184** — the backfill and the off code path change nothing.

| metric | OFF (baseline) | ON | Δ |
|---|---|---|---|
| all-gold-in-context | 85 | **88** | +3 |
| any-gold-in-context | 97 | **99** | +2 |
| answer-chunk-in-context | 43 | **46** | +3 |
| lane fired | — | 100/100 | — |
| **regressions** | — | **0** | — |

**Wins (4), traced:**

| q | OFF → ON | mechanism |
|---|---|---|
| **0977f2af** | all-gold **False→True**, ans-chunk **False→True** | The marquee Round-4 case. Gold `_1` was pool rank **9** (out-of-k — the exact failure Round 4 §3 declared unfixable by reorder). The lane matched `Air Fryer used_for sweet potato fries` / `sweet potato fries cooked_in Air Fryer` and seeded BOTH gold chunks → gold `_1` climbs to pool rank **4**, enters the prefix. Recall fix, not a reorder. |
| **09d032c9** | all/any **False→True** | Gold pool rank **3→1** (fact-seeded); enters context. Largest context growth (+12.9 KB) — the lane pulled the gold session's chunks in. |
| **71017277** | all/any **False→True** | Gold pool rank **6→4** (fact-seeded); enters context. |
| **27016adc** | ans-chunk **False→True** (session-gold already in) | Pool rank unchanged (2, 1); the fact-seeding tipped the CE retarget pick toward the answer chunk. **Fragile**: this is the very question Round 4 §2 / Round 5 flagged for a cross-process CE near-tie on the ans-chunk mark — within-process the flip is attributable to the lane, but it sits on a known tie, so count it a half-win. |

**Named targets that did NOT move (honest negatives):**

- **06f04340**: gold pool rank **6→6**, `gold_seeded_by_fact=[]`. The brief hoped the gold's
  "homegrown cherry tomatoes/basil/mint" statements would now match directly. They did not: the
  top-10 fact hits for this query were OTHER food statements (`mixed greens topped_with grilled
  chicken`, the garbled `family dinner try my mom`) — the right statements ranked below the cut.
  This is Round 7a open-risk #3 realized: at ~gpt-4o-mini extraction quality the RIGHT statement
  competes with many sibling food statements even at fact granularity; fact vectors are
  necessary but not sufficient here. No change (correctly — the lane didn't seat a wrong chunk).
- **2ce6a0f2**: gold `_3` pool rank **20→21** (a 1-slot tail reshuffle from the added seeds),
  NOT in `gold_seeded_by_fact` (the lane seeded `_1`/`_2`/`_4`, already high). Answer to the
  brief's question — *does gold `_3` enter the pool via any extracted fact?* — **no**: `_3`'s
  content ("upcoming events/exhibitions") was matched only to already-seated chunks. Outcome
  unchanged (any-gold True via `_1` at rank 1). Confirms the lane does not manufacture recall
  where no fact bridges to the missing chunk.

**Canaries**: the fidelity gate (184/184) IS the canary check — every recorded gold
in_context mark is unchanged at knob-off, and no question lost any-gold at knob-on (0
regressions).

## 5. Noise analysis (the top-10 matched facts)

The lane fired on all 100 questions (a personal store always has *some* top-10 facts). Across
800 matched surfaces a crude garble heuristic (>7 tokens or heavy word repetition) flags ~5%
as clear extraction junk (`Alternative Therapies in Health and Medicine published_in binaural
beats`, `group of engineers and a manager total_count 6`), and eyeballing the "clean" 95% shows
plenty more that are merely loose (`therapists help clients`, `David met workshop`) — squarely
the ~gpt-4o-mini quality the brief anticipated. **This is why the design is additive-capped and
promotion is displacement-conservative**: a noisy fact costs a wasted seed slot and (at worst) a
wasted `[matched]` tag, never a displaced gold chunk — which is exactly what the 0-regression
column buys. Contrast Round 4, where a seed-score *prior* on chunk identity produced regressions.

## 6. SYNTHETIC needle probes (scripts/offline_eval_round7b.py --synth-only)

The Round-1/2 synth store (71 episodes) extended with off-topic needles: a blood-type fact
stated while booking a flight, a dentist referral inside a scheduling chat, plus a preference
probe. Findings, honestly: on this SINGLE-CHUNK, self-anchored store the `me`-hub makes every
me-episode reachable by diffusion, so **object-level pool entry is not a discriminator** and the
unchunked episodes give provenance-promote **no sibling to displace** — so the needle is *seeded
and marked* but not *seated* (ctx unchanged on↔off). What the synth DOES show cleanly: (a) the
`[matched]` mark appears only with the lane on, and the top matched surface is the right fact
for each probe (`me has_blood_type O-negative`, `me sees_dentist Dr. Nguyen`); (b) the
preference probe "what do I like to do for fun?" matches the distilled AGGREGATE surfaces
(`me went_to the park`, `me plays tennis`) — the disposition-retrieval target Round 7a built.
The recall WIN needs chunked sessions with displaceable siblings, which is why it shows on the
REAL benchmark (§4) and not here. The unit test
`test_provenance_promotion_seats_needle_when_sibling_displaceable` proves the seating mechanism
directly.

## 7. Tests (tests/test_fact_lane.py — 11, offline, $0)

`fact_provenance` resolves a statement hit to its provenance chunk + both endpoints and an
aggregate hit to its whole group; `fact_seed` catches an off-topic needle the episode lane
misses and returns empty on an un-backfilled store; **additive-only** — merging adds only new
nodes and a node the episode lane already seeded keeps the EPISODE score (not the fact score);
**mass cap** — Σ fact-lane mass == `fact_lane_weight` × episode mass when raw exceeds it;
**knob-off byte-identical** seeds and no `[matched]` mark; end-to-end the lane seeds the needle
and marks its line while the episode lane doesn't; provenance-promote seats a needle via
`force_ids` when a sibling is displaceable and honours the `rag_provenance_promote` gate; the
`[matched]` tag lands only on the hit line (`fact_lane_k=1`). Full suite: **402 passed, 0
failed** (391 + 11).

## 8. Recommendation & paid validation (NOT run)

**Ship-candidate.** 4 wins / 0 regressions on 100 real questions, fidelity 184/184, knob-off
byte-identical, and the mechanism is a RECALL fix (fixes `0977f2af`, the case Round 4 proved no
reorder could) rather than a reshuffle. The one caveat is `27016adc`'s fragile CE-tie half-win;
the three clean wins (`0977f2af`, `09d032c9`, `71017277`) are unambiguous pool-rank → context
gains. Keep `fact_lane_k=10` / `fact_lane_weight=0.5` (unswept — a sweep is the obvious paid
follow-up; the cap is conservative by construction and the 0-regression result gives no signal
that it is too tight). Confirm with ONE paid answer run before flipping the default:

```
# 1. one-time $0 LOCAL backfill of the cached benchmark stores (additive; cache key unchanged)
for db in store/cache/*.db; do python -m kg --store "$db" backfill-fact-vectors; done
# 2. paid answer run — query-side fact_lane on top of the run's usual --set string
python -m kg testrun --mode per-instance --tier sample \
  --set history_all_lanes=true --set event_facts=true \
  --set rag_retarget=ce --set rag_provenance_promote=true \
  --set mmr_lambda=1.0 --set rag_parent_expand=2 --set rag_chunks_per_source=2 \
  --set fact_lane=true \
  --label fact-lane-round7b-1 --out runs
```

(`fact_lane` is query-side and does NOT change the ingest-cache key, so step 1's backfill is the
only prerequisite and there is NO paid re-extraction. `KG_LLM=openai` / `OPENAI_API_KEY` must be
live. Expected cost ≈ **$0.05–0.15** for the sample tier — the context grows a few KB on the
fired questions, no new call sites.) **Four question ids to watch** (compare `judge.correct`
against the paid baseline): `0977f2af`, `09d032c9`, `71017277` (expected to FLIP correct — gold
now in context), and `06f04340` (expected UNCHANGED — the lane didn't seat its gold). Do not run
this here.

---

# Round 8: speaker attribution

*2026-07-20. Who SAID a fact's evidence — so the reader can refuse to compute an answer from
figures it can't attribute to the user. One default-off, QUERY-SIDE knob
(`speaker_attribution`) plus an always-on, additive, $0 ingest/backfill stamp. Speaker is
DERIVABLE from raw text (chat chunks carry inline "User:"/"Assistant:" turn markers), so the
whole ingest side is a local, no-LLM backfill; the reader change rides the same knob. Touches
kg/speakers.py (new), kg/models.py, kg/store.py, kg/ingest.py, kg/config.py, kg/facts.py,
kg/rag.py, kg/engine.py, kg/graph.py, kg/cli.py; tests in tests/test_speakers.py. Harness
scripts/offline_eval_round8.py drives the same cached-store read path as Rounds 3–7b. No paid
calls, no commits. Verdict up front: **ship-candidate on the offline evidence — fidelity
184/184, soft-not-hard 100/100 append-only, and every abstention target now carries the
[assistant] marker on its offending fact lines; the reader FLIP is the paid question.***

## 1. Motivation (Round 6b §5, the abstention canaries)

Round 6b already named the disease: on the abstention questions the top graph signal is the
ASSISTANT's generic figures — `09ba9854_abs`'s top tallies are the assistant's example bus/taxi
fares, "the exact operands the abstention answer must refuse." The reader miscomputes because it
cannot tell WHO stated a figure: the assistant's generic price ranges / typical fares / example
head counts sit in the same conversations as the user's own stated facts, and a
"what is true of me / what did I spend" question that reads an assistant figure as a user fact
answers with a number that was never the user's. Three failures make this concrete
(`runs/sample-datefix-events-1/run.json`): `031748ae_abs` (engineer head counts the assistant
suggested), `19b5f2b3_abs` ("how long in Korea" — the assistant's Seoul itinerary, no user stay),
`09ba9854_abs` ("saving by taking the bus" — the assistant's generic fares). The fix is
provenance the reader can weigh, derived at $0 from markers already in the text.

## 2. Data model (decided; NOT graph edges, NOT stored on fact edges)

Speaker provenance is METADATA you filter by, never a diffusion signal and never a stored
attribute of a fact. Two pieces, plus a read-time derivation:

| piece | where | shape |
|---|---|---|
| **`speakers` registry** — reference data, one row per speaker | kg/store.py (`speakers` table; `upsert_speaker`/`get_speaker`; loaded tolerantly so a pre-feature db with no table still opens) | `(speaker_id, kind∈{human,assistant,mixed}, canonical_name, aliases[])`. `speaker_id = "sp_"+sha256(canonical_name)[:12]` (stable, content-addressed). `aliases[]` is forward-scaffolding for multi-human — stored, NO identity-resolution logic. Every store has exactly the ~3 canonical rows. |
| **`speaker_id` field** on each immutable EPISODE/chunk node | kg/models.py `Node.speaker_id`; stamped in kg/ingest.py `_write_episode` and by backfill | ONE stamp per chunk, parsed from its inline turn markers. `None` for an unmarked chunk (plain note / described media). "chunks by speaker" is a QUERY (`nodes where speaker_id=X`), never a stored list. |
| **attribution** — DERIVED at read time, never stored | kg/speakers.py `asserted_by(store, data)` | a fact's speakers = the kinds of its provenance episodes (`episode_id ∪ confirmed_by`) resolved through the registry. Derivable → cannot go stale on retract / forget / merge / canonical-rename. |

**Why NOT graph edges** (a `SAID_BY` edge per chunk): that mints a per-speaker super-hub (every
chunk incident to `assistant`), the exact `self_guard` pathology PPR already fights. **Why NOT
an `asserted_by` column on the fact edge**: attribution is derivable from provenance, so a stored
copy would go stale the moment a fact is confirmed by a new turn, retracted, or has an endpoint
renamed. Both are avoided by keeping speaker on the immutable chunk and deriving the rest.

**Reduction — ANY-USER.** A fact is USER-GROUNDED if ANY provenance turn is a human (a user fact
echoed back by the assistant is still a user fact). The reader marker `[assistant]` fires ONLY
when a fact rests EXCLUSIVELY on assistant turns (`is_assistant_only`: ≥1 known speaker, all
`assistant`). Unknown provenance (no stamp) blocks the mark too — conservative, never discount.

**Mixed chunks.** Chunking is not guaranteed single-turn: the `turns` chunker packs adjacent
turns up to `chunk_target_chars`, so a chunk often holds BOTH roles. Such a chunk is stamped
`kind='mixed'` and counts as CONTAINING-HUMAN for attribution — so a fact grounded in a mixed
chunk is never wrongly discounted to [assistant]. This is not a corner case: on the sample store
below, **110 of 311 chunks (~35%) are mixed**.

## 3. What changed (code)

| piece | where | knob |
|---|---|---|
| **Deterministic parse**: `parse_speaker(raw_text)` → `(speaker_id, kind)` from `^User:/Assistant:` (also Human/AI) markers; both roles → `mixed`; no marker → `(None,None)`. Regex only, no LLM. | kg/speakers.py | — |
| **Ingest stamping**: each episode is stamped + the registry upserted at write time. ALWAYS ON (additive, $0) — but every CONSUMER is gated, so a knob-off run is byte-identical downstream. Stamping writes only `Node.speaker_id` + the registry table, so it does NOT enter `INGEST_RELEVANT_FIELDS`. | kg/ingest.py `_write_episode`, kg/store.py | — |
| **Backfill** `KnowledgeGraph.backfill_speakers()` / `kg backfill-speakers` (mirrors `backfill-fact-vectors`): stamp existing chunks + build the registry — idempotent, incremental (only re-touches a node whose stamp changed), $0. Touches only node payloads + the speakers table, so it CANNOT change the ingest-cache key (which hashes config/sessions/prompt, never db contents): the 100 cached benchmark stores gain provenance in place with no paid re-extraction. | kg/graph.py, kg/cli.py | — |
| **Context marker**: a FACTS line resting exclusively on assistant turns gets a trailing `[assistant]` (composes after any `[matched]`); off ⇒ marker always `""` ⇒ rendering byte-identical. | kg/rag.py `ContextBuilder.build` | `speaker_attribution` |
| **Reader prompt rule** (`_SPEAKER_RULE`, appended to `_RAG_SYS` only when on): "Lines marked [assistant] rest only on the assistant's turns … for questions about what is true of the user / what they did or spent, rely on user-stated facts; do not compute from [assistant] figures alone — if only [assistant] material bears on the question, say the information isn't available." Off ⇒ system prompt byte-identical. | kg/rag.py `OpenAIAnswerer.answer` | `speaker_attribution` |
| **Structured `asserted_by`**: the derived list of kinds is added to every fact row (kg/facts.py `FactLine.to_row`, kg/engine.py `_fact_row`) so programmatic/agent callers can filter user- from assistant-material. Derived at row-build, not stored; `[]` = unknown (treat as user-grounded). | kg/facts.py, kg/engine.py | — (always present) |

`speaker_attribution` is Tier-2, QUERY-SIDE, and deliberately NOT in `INGEST_RELEVANT_FIELDS`
(test asserts the cache key is unchanged when it flips). Knob off ⇒ context AND system prompt
byte-identical (both tested).

**SOFT SIGNAL, NEVER A HARD FILTER.** Speaker is a MARKER the reader weighs, not a retrieval
exclusion. Assistant chunks are NOT removed from seeding / PPR / `facts_for`: ~13% of benchmark
answers live ONLY in assistant turns, and some questions specifically WANT assistant content
(`06878be2` "what accessories would complement my camera" — the assistant's recommendations ARE
the answer). Filtering would break those; marking lets the reader keep them and weigh them.

## 4. Backfill cost (measured, one real cached store)

`kg backfill-speakers` on a COPY of `store/cache/001be529-*.db` (311 episodes), no LLM:

| metric | value |
|---|---|
| chunks stamped | **309** (2 unmarked — a chunk with no turn marker) |
| chunk kind split | 112 assistant / 110 mixed / 87 human / 2 none |
| registry rows | **3** (user, assistant, mixed) |
| wall time | **0.006 s** ($0 — pure regex, no embed, no LLM) |
| second run (reopen) | 0 changed (idempotent) |

Extrapolated: the whole 100-store benchmark backfills in a few seconds of local CPU, one-time,
no paid calls, no cache invalidation.

## 5. Offline validation — `scripts/offline_eval_round8.py` (Round-3 pattern, cached stores)

Same harness as Rounds 3–7b: the 100 cached per-instance stores copied to temp paths, speakers
backfilled into each copy ($0), the exact `HybridRetriever.retrieve → ContextBuilder.build` read
path driven with the run's query-side config, keys stripped on import. Each question's context is
built TWICE (`speaker_attribution` OFF, then ON).

| metric (n=100) | value |
|---|---|
| speakers backfilled | 30,676 chunk stamps; 3 registry rows/store (avg 307 chunks/store) |
| **fidelity (knob OFF) vs run.json** | **184/184** gold in_context marks agree — backfill + knob-off change no retrieval/context |
| **append-only (ON == OFF + markers)** | **100/100** — the ON blob equals the OFF blob after stripping " [assistant]"; NO fact line is ever removed (soft-not-hard, proven on real data) |
| context episode set unchanged, ON | 100/100 |
| questions with ≥1 [assistant] mark | 99/100 (a personal chat store almost always has some assistant-only fact) |

**The three abstention targets — offending fact lines now carry `[assistant]`** (append-only,
context unchanged in all three):

- **031748ae_abs** (engineer counts): 8 assistant-only lines — the assistant's suggested
  city/landmark/rooftop relations, none a user-stated head count.
- **19b5f2b3_abs** ("Korea"): 8 assistant-only lines — `Seoul --include--> Gyeongbokgung Palace`,
  `Myeong-dong --stay--> Seoul`, `N Seoul Tower --include--> Seoul` … the assistant's Seoul
  itinerary, not a user stay-length.
- **09ba9854_abs** (bus fares): 21 assistant-only lines including
  `Taxi --cost--> 15000 JPY`, `--cost--> 20000 JPY`, `--cost--> 6000 JPY`,
  `Taxi --approximate_travel_time--> 120 minutes` — exactly the generic fares the correct answer
  must NOT compute a saving from. The marker + the prompt rule are what keep this abstained.

**The keep-it target — assistant lines MARKED but NOT removed:**

- **06878be2** ("accessories for my camera"): 12 assistant-only lines
  (`Anker PowerCore 20000 PD --charges--> Sony A7R IV`, `Mophie Powerstation XXL --charges-->
  Sony A7R IV`, …) — the assistant's recommendations, which ARE the answer. `append_only=True`,
  `ctx_same=True`: every one stays in context, just flagged. This is the soft-not-hard contract
  made visible: marking ≠ filtering.

## 6. Tests (tests/test_speakers.py — 21, offline, $0)

Parse (user / assistant / mixed / none / Human-AI aliases / header-tolerant; stable+distinct
ids); registry upsert idempotent + persists/reloads + **load tolerates a pre-feature db with no
speakers table**; attribution any-user reduction over `episode_id ∪ confirmed_by` (assistant-only
= assistant-grounded; user-echoed = user-grounded; mixed counts as human; unknown unmarked);
**marker on the assistant-only line only, user-echoed line UNmarked, only when the knob is on**;
**knob-off context byte-identical** (on == off after stripping markers) AND **knob-off system
prompt byte-identical** (== `_RAG_SYS`, via a fake answer client); `asserted_by` on the fact row;
backfill idempotent + incremental on a scripted chat store AND on a COPY of a real cached store;
**cache-key unchanged by the knob**. Full suite: **422 passed** (the 21 new tests included), 1
pre-existing environment failure unrelated to this work (`test_fact_vectors.py::test_backfill_on_real_cache_copy`
— `added=0` because the local `store/cache/*.db` were already fact-vector-backfilled; fails on the
base commit too). `tests/test_rag.py` updated for the additive `asserted_by` row key.

## 7. The MEASURABLE lever, and the paid question

Reader BEHAVIOR can't be judged offline (it is an LLM prompt+context change). But this is the
measurable lever the fact-lane noise never was: the *_abs questions fail CONSISTENTLY on
identical stores (they are abstention canaries, not retrieval-rank noise), and the offending
operands are now unambiguously flagged on the SAME context the paid baseline read (fidelity
184/184). So the offline result is a clean setup: same retrieval, same episodes, the assistant's
figures marked, the user's unmarked, and a prompt rule that says do-not-compute-from-[assistant].

## 8. Paid validation (NOT run)

Small tier, cached stores (the knob is query-side → the ingest-cache key is unchanged → no
re-extraction). One-time $0 local backfill first, then the run's usual `--set` string plus the
new knob:

```
# 1. one-time $0 LOCAL backfill of the cached benchmark stores (additive; cache key unchanged)
for db in store/cache/*.db; do python -m kg --store "$db" backfill-speakers; done
# 2. paid answer run — query-side speaker_attribution on top of the run's usual --set string
python -m kg testrun --mode per-instance --tier sample \
  --set history_all_lanes=true --set event_facts=true \
  --set rag_retarget=ce --set rag_provenance_promote=true \
  --set mmr_lambda=1.0 --set rag_parent_expand=2 --set rag_chunks_per_source=2 \
  --set speaker_attribution=true \
  --label speaker-attribution-round8-1 --out runs
```

(`KG_LLM=openai` / `OPENAI_API_KEY` must be live. Expected cost ≈ **$0.05–0.15** for the sample
tier — the context grows only the marker suffixes, no new call sites.) **Watch four ids** (compare
`judge.correct` against the paid baseline): `031748ae_abs`, `19b5f2b3_abs`, `09ba9854_abs`
(expect FLIP toward abstained/correct — their offending fact lines are now marked and the prompt
forbids computing from [assistant] figures alone), and `06878be2` (expect UNCHANGED — the
assistant's camera-accessory recommendations are marked but still in context and still usable).
Do not run this here.
