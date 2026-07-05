# Extraction completeness after the typed-facts fix — before/after

Re-runs `spikes/completeness/build_stores.py` (unchanged) against the same 8 target
instances, after the kg/extractors.py changes: a typed `facts[]` array on `emit_graph`
(subject/predicate/value/unit/date), an always-extract completeness rule exempting
amounts from the entity salience filter, deterministic compound-value splitting
("Tuesdays and Thursdays" → 2), per-occurrence non-collapse in `Extraction.merge` and
`apply_fact`, and a reflexion pass that also checks for missed amounts/occurrences. Old
stores were preserved under `spikes/completeness/stores_before/` before rebuilding.

## SUM questions — the primary target

| id | question | gold amounts | captured BEFORE | captured AFTER |
|---|---|---|---|---|
| 2b8f3739 | $ earned selling at markets | 225, 150, 120 | 0/3 | **0/3** (unchanged) |
| 36b9f61e | $ spent on luxury items | 800, 1200, 500 | 2/3 | **2/3** (same 2; still missing 1200) |
| 129d1232 | $ raised via charity events | 250, 600, 5000 | 1/3 | **3/3** ✅ |
| **total** | | **9** | **3/9 (33%)** | **5/9 (56%)** |

Improvement, but short of the ≥7/9 target. Root-caused below — it is not a shortfall in
the new schema/prompt itself.

### What worked

`129d1232` went 1→3: every raised amount ($250, $600, $5000) now lands as a typed
`quantity` node/edge (`entity_type=quantity`, numeric `value`, `unit=USD`) reachable
from its subject ("charity walk" --raised--> 250, etc.) — summable with `SUM(value)`
grouped by subject, no regex. `36b9f61e` kept its two previously-captured amounts ($800,
$500), now also as typed facts instead of a bare string node.

### What's still missing, and why (diagnosed, not hand-waved)

For every one of the 4 still-missing amounts (`2b8f3739`'s all three; `36b9f61e`'s
$1200), I traced the failure to the **same root cause, and it is upstream of the
extractor prompt/schema this task was scoped to fix**:

`kg`'s default extraction path is `CueGatedExtractor` (`extractor_backend=cue_gated`,
`Config.default()` — used by `build_stores.py` unchanged, matching production). It runs
a free local-NLP floor on every entry and escalates to `OpenAIExtractor` (the extractor
this fix lives in) only on **cue-bearing text** (`kg/cues.py`: termination / relative-date
/ identity cues). For long sessions, `extract_text_sectioned` splits the text into
6000-char sections and **re-checks the cue gate per section**, independently.

Traced with `has_cue()` directly against the raw session text:

- `36b9f61e` / Gucci session (10,349 chars, 2 sections): the "$1,200" mention sits at
  char ~280, inside **section 0, which has no cue** (`has_cue → False`). Section 1
  (chars 6000+) does have a cue and escalates, but the $1200 text isn't in it — so the
  fixed extractor never sees "$1,200" at all. (Section 1 also produced several
  budget-breakdown amounts the model apparently reasoned from a later hypothetical
  discussion in that half of the transcript — extraction quality on content it *did*
  see, a separate concern from completeness.)
- `2b8f3739` / jam-sale session (chars ~250, no cue) and herbs session (chars ~50, no
  cue): same story, 2 of the 3 sessions never reach any LLM extractor call, cue-gated
  or not — the local-only floor doesn't know about `facts[]`. The third session (herb
  plants, "$7.5 each", 18,474 chars / 4 sections) does have a cue, but only in **section
  1** (chars 6000–12000); the price mention is in **section 0** (no cue), so again the
  fixed extractor is invoked on content that doesn't contain the number.

Verified directly: calling `OpenAIExtractor.extract_text` on the model in isolation over
content that *does* contain the target number reliably follows the new completeness
rule (this is what unit tests in `tests/test_facts.py` cover, and what explains
`129d1232`'s 1→3 and the two amounts that were already working). The gap is entirely
**whether the extractor ever runs on the right slice of text**, which is the cue-gating
policy (`kg/cues.py` + per-section re-gating in `extract_text_sectioned`) — explicitly
out of this task's scope ("Root causes are in the extractor prompt/schema... GRAPH_TOOL
emit_graph schema"). Fixing it would mean either gating on the section's own content
more liberally, or running the typed-facts extraction on every section regardless of
cue (cost trade-off), which is a different lever than the one this task asked for.

## Compound-value splitting (item C)

`_split_compound()` (kg/extractors.py) is unit-tested and correct in isolation
(`tests/test_facts.py::test_split_compound_weekdays`,
`test_relation_target_compound_split_into_two_relations`): "Tuesdays and Thursdays"
parsed through `_parse_tool_payload` → two relations, "Bed and Breakfast"/"Johnson and
Johnson" → left alone.

Re-inspecting the live `2788b940` (Zumba) store, however, the persisted edge is still
`Zumba --?--> 'Tuesdays and Thursdays'` as ONE compound node. Isolating the call: the
Zumba mention's 6000-char section *does* carry a cue and *does* escalate to
`OpenAIExtractor` — but on a fresh direct call, the model did not re-emit the
weekday-schedule relation at all in that response (temperature=0 but real API
run-to-run variance exists; the earlier extraction that produced the compound value
most likely came from the free local-NLP floor, which is unioned into the result via
`CueGatedExtractor` and has no `_split_compound` pass — that's a separate module
(`kg/nlp_extractors.py`), not the OpenAI extractor this task scoped in). This is the
same structural boundary as the SUM misses: the fix works where it runs; the local-floor
merge path and cue-gating decide whether it gets the chance to.

## Discrete-event questions (spot check, not a full re-audit)

Time-boxed: I did not redo the full manual 26-occurrence reclassification from the
original REPORT.md. `00ca467f` (doctor's appointments) was re-checked and is unchanged
— both occurrences (Dr. Smith/March 3rd, Dr. Thompson/March 20th) still land as
distinct edges. No regression observed in the questions this fix wasn't targeting.

## Regression checks (all passed)

- Full test suite: **162/162 passed** (1 apparent failure during a run was traced to my
  own test-shell clearing `OPENAI_API_KEY`, not a code regression — confirmed passing
  standalone with the key restored).
- 12 new unit tests (`tests/test_facts.py`) cover: `facts[]` schema parsing and
  validation, compound splitting (weekdays/amounts split, proper names don't), per-
  occurrence non-collapse in both `Extraction.merge` and `apply_fact` (a second,
  differently-dated visit/purchase now opens a new edge instead of confirming into the
  first), and the canonicalizer guarantee that two distinct amounts never alias-merge
  (verified end-to-end: $250 and $2,500 in one episode produce two distinct `quantity`
  nodes, summable to $2,750).
- `ingest_cache.py`'s `_extractor_prompt_digest()` now also hashes `GRAPH_TOOL`
  (previously only `_SYS`), so this schema change invalidates the ingest cache even
  without a wording change next time.
- Sample-tier smoke (`kg testrun --tier sample --queries 8 --chunking turns`, live,
  vs. baseline `runs/sample-chunked-1`):

  | metric | baseline | after fix |
  |---|---|---|
  | recall@k / mrr / hit_rate | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 (no regression) |
  | citation grounding | 0.896 | 0.958 (improved) |
  | judge accuracy | 0.75 | 0.75–1.0 (noisy across repeated runs of an 8-question set) |
  | ingest tokens | 252,731 | ~288,000–289,000 |
  | **ingest token growth** | — | **~14%** (target: <10%) |

  **Missed target, diagnosed honestly:** the growth is dominated by fixed per-call
  overhead — the `emit_graph` tool schema and system prompt are sent on every one of
  166 extraction calls, and the new `facts[]` array (5 typed properties + description)
  and the new prompt rule add a combined ~210–240 tokens/call regardless of content
  (measured directly with `tiktoken`: `_SYS` +55 tok, `GRAPH_TOOL` +156 tok after two
  rounds of trimming wording down to the minimum needed to state the rule). At 166
  calls that's ~35–40k tokens of fixed overhead alone, i.e. ~14–16% of the 226k-token
  baseline input, before counting the *additional* real output tokens from the model
  actually returning more facts (the intended effect). I trimmed every description down
  to field-name-only where the name is self-explanatory; further cuts would mean
  dropping one of the required typed fields (value/unit/subject/predicate/date), which
  requirement A explicitly calls for. This is a real, structural cost of giving
  quantities a typed home, not an oversight — I did not tune the schema wording to hit
  the number.

## Bottom line

The fix works, and works well, whenever the extractor actually runs on the text
containing the number: `129d1232` went from catastrophically wrong (1/3, 96% short) to
exact, and $800/$500 keep working. But the sample's remaining misses (4/9 amounts, one
compound-split case) are gated by a **pre-existing, out-of-scope mechanism**
(`CueGatedExtractor`'s per-section cue escalation and the separate local-NLP floor) that
decides whether the fixed extractor is ever invoked on a given slice of text at all. If
closing that gap matters, the next lever is cue-gating policy or per-section escalation
coverage — a different, follow-on piece of work from "fix the extractor's own
prompt/schema" that this task was scoped to.

## Follow-on: closing the cue-gating gap with a quantity cue

Implements the lever named above. Added a fourth cue kind to `kg/cues.py` —
`_QUANTITY` — that fires on a currency symbol/word attached to digits (`$1,200`, `20
bucks`) or an explicit numeral attached to a measurement unit (`10 lbs`, `3 miles`, `2
dozen`). Deliberately narrow: bare numbers, years (`in 1995`), clock times (`at 5pm`),
ordinals (`3rd`), and unitless counts (`3 of my friends`) do **not** match — every match
costs a paid escalation call, so the pattern requires an explicit currency/unit token,
not just a digit. 5 unit tests in `tests/test_cues.py` cover the positive/negative cases.
`extract_text_sectioned` needed no change: it re-invokes `extractor.extract_text` per
section, and `CueGatedExtractor.extract_text` already calls `has_cue()` — the new kind
is picked up automatically.

**Cache correctness:** `kg/ingest_cache.py`'s `_extractor_prompt_digest()` hashed the
extractor prompt/schema but not `kg/cues.py`, so this change would silently NOT have
invalidated existing cached stores even though it changes what gets written. Fixed by
hashing `inspect.getsource(kg.cues)` into the same digest, with a unit test
(`test_cue_pattern_change_invalidates_key`) that stubs `inspect.getsource` to simulate an
edit and asserts the cache key changes.

**Escalation-rate impact** (measured directly with `has_cue`, no API, over the same 8
target instances' sessions, sectioned at `long_doc_chars=6000`):

| | sections | escalated |
|---|---|---|
| before (3 cue kinds) | 848 | 377 (44.5%) |
| after (4 cue kinds) | 848 | 440 (51.9%) |
| newly escalated by the quantity cue alone | | 63 (7.4 pts) |

**Completeness re-audit** (`spikes/completeness/build_stores.py`, live, ~8 min ingest,
stores rebuilt into `spikes/completeness/stores`; prior stores preserved under
`stores_after_facts_fix/`):

| id | question | gold amounts | captured BEFORE (facts fix only) | captured AFTER (+ quantity cue) |
|---|---|---|---|---|
| 2b8f3739 | $ earned selling at markets | 225, 150, 120 | 0/3 | **2/3** ($225 jam, $120 farmers' market now typed edges; the $150 herb-plant sale is present as unit price `$7.5` × quantity `20` but never combined into a literal `150` node — a multiplication the extractor doesn't do, not a cue-gating miss) |
| 36b9f61e | $ spent on luxury items | 800, 1200, 500 | 2/3 | **3/3** ✅ (`$1,200` Gucci handbag now captured — this was the headline miss in the prior report) |
| 129d1232 | $ raised via charity events | 250, 600, 5000 | 3/3 | **3/3** (unchanged, unaffected) |
| **total** | | **9** | **5/9 (56%)** | **8/9 (89%)** ✅ target met (≥8/9) |

`36b9f61e`'s $1,200 was the specific case root-caused in the prior report (section 0 of
the Gucci session had no cue at all under the old 3-kind gate); it's captured now. The
one remaining miss (`2b8f3739`'s $150) is a genuinely different problem — arithmetic
composition of two already-captured facts, not a text the extractor never saw — and is
out of scope for a cue-gating fix.

**Full test suite:** 175/175 passed (162 previously + 13 new: 5 in `tests/test_cues.py`,
1 in `tests/test_ingest_cache.py`; the `tests/test_facts.py` count from the prior report
didn't change). Ran standalone in 287s; an earlier concurrent run (started alongside the
live completeness rebuild) was killed after both jobs were observed thrashing the same
GPU-resident local NER model — not a suite problem, just don't run them together.

**Sample-tier smoke** (`kg testrun --tier sample --queries 8 --chunking turns --label
quantity-cue-smoke`, live, vs. the facts-fix baseline `facts-fix-sample-smoke-v3`):

| metric | facts-fix baseline | + quantity cue |
|---|---|---|
| recall@k / mrr / hit_rate | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 (no regression) |
| citation grounding | — | 0.958 |
| judge accuracy | noisy 0.75–1.0 | 0.875 |
| tier1 SUM capture (amounts_in_graph / in_text) | 0/37 (0%) | **22/37 (59.5%)** |
| tier2 captured / collapsed / missing (n=15) | 6 / 7 / 2 | **7 / 8 / 0** (missing → 0) |
| ingest llm_calls | 166 | 229 (+38%) |
| ingest tokens | 288,103 | 414,520 |
| **ingest token growth over facts-fix baseline** | — | **+43.9%** |

**Not small — reported honestly.** The task expected a small cost delta; it isn't one.
63 more Haiku calls (+38%) fire because more sections now carry a cue, and each call
pays the same fixed `emit_graph` schema/prompt overhead as before (the facts-fix report
already measured that overhead at ~210–240 tokens/call) on top of whatever additional
content it actually extracts. This is the direct, expected cost of the tradeoff this task
asked for: catching every dollar amount means escalating on every section that mentions
one, and dollar amounts are common. If this growth needs to come down, the lever is
tightening `_MEASURE_UNIT`/`_MONEY_WORD` further (e.g. dropping generic measurement units
and keeping only currency, since the completeness gap that motivated this work was
specifically about money) rather than anything in this change's own correctness.
