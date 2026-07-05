# Extraction completeness for counting/aggregation questions

**Question:** when the true answer to "how many times did X happen?" (or "how much
in total?") is N, how many of those N occurrences actually exist as edges in the
graph after ingestion? This gates a planned deterministic SQL-aggregation layer.

**Method:** 8 aggregate-shaped questions from `runs/small-triage-1/run.json`
(longmemeval:small, per-instance mode) were re-ingested into fresh, single-instance
stores under `spikes/completeness/stores/`. Ground truth per question was enumerated
by an LLM pass (gpt-4o-mini) over each instance's gold-evidence sessions
(`ground_truth.py` / `ground_truth.json`), then spot-checked by hand against the raw
session text for 2 questions (`00ca467f`, `2b8f3739` — see
`spot_check_00ca467f.txt`, `spot_check_2b8f3739.txt`). For each occurrence, the
instance's `edges` table was queried by `episode_id = ep_<qid>__<session_id>` and the
connected node payloads inspected (`dump_edges.py`) to classify it CAPTURED /
COLLAPSED / MISSING.

Selection: 6 "reader"-bucket failures (gold evidence was in context but the LLM
reader still answered wrong), 1 "join_miss" failure (retrieval didn't pull all gold
sessions), and 1 "ok" passing case, to see both failure and success modes.

## Per-question results

| id | question (short) | gold | occurrences | captured | collapsed | missing | naive SQL vs gold |
|---|---|---|---|---|---|---|---|
| 00ca467f | doctor's appointments in March | 2 | 2 | 2 | 0 | 0 | 2 vs 2 — **match** |
| 2788b940 | fitness classes in a typical week | 5 | 5 | 3 | 2 | 0 | 4 vs 5 — **under by 1** |
| 2e6d26dc | babies born to friends/family | 5 | 5 | 5 | 0 | 0 | 5 vs 5 — **match** |
| 21d02d0d | fun runs missed in March | 2 | 2 | 2 | 0 | 0 | 2 vs 2 — **match** |
| 0a995998 | clothing items to pick up/return | 3 | 3† | 2 | 0 | 1† | 2 vs 3 — **under by 1** |
| 2b8f3739 | $ earned selling at markets (SUM) | $495 | 3 amounts | 0 | 0 | 3 | $0 vs $495 — **total miss** |
| 36b9f61e | $ spent on luxury items (SUM) | $2,500 | 3 amounts | 2 | 0 | 1 | $1,300 vs $2,500 — **48% short** |
| 129d1232 | $ raised via charity events (SUM) | $5,850 | 3 amounts | 1 | 0 | 2 | $250 vs $5,850 — **96% short** |

† `0a995998`'s third gold item ("winter clothes I haven't touched... need to sort
through") is ambiguous — it's not clearly framed as a "pick up/return from a store"
item the way the dry-cleaned blazer and the Zara boots exchange are. Counted as
missing/unclear rather than resolved either way; didn't chase further (timeboxed).

### Notable per-question detail

- **00ca467f** (doctor's appointments): both occurrences are individually captured
  as edges with distinct, semantically meaningful `rel_tag`s (`diagnose` for
  Dr. Smith/bronchitis, `follow` for Dr. Thompson/March 20th), each on the correct
  `episode_id`. Extraction is **not** the problem here — this was a "reader" failure
  (LLM synthesis), not a completeness failure. Confirmed by direct read of all 3
  evidence sessions.
- **2788b940** (fitness classes/week): BodyPump→Mondays, Hip Hop Abs→Saturday,
  Yoga→Sundays are each a separate edge (captured). But Zumba's two weekly slots
  are extracted as **one** edge to a single compound node, `'Tuesdays and Thursdays'`,
  instead of two separate day facts — a genuine schema-flattening COLLAPSE of 2 true
  occurrences into 1 edge. This is the one clean confirmation of the original
  "repeated events collapse to one edge" hypothesis in this sample.
- **2e6d26dc** (babies born): all 5 named babies (Jasper, Charlotte, Max, Ava, Lily)
  are captured as distinct entities, each linked via a `parent_of`-labeled relation
  to their parent. `COUNT(DISTINCT dst)` over that relation type would return
  exactly 5 — extraction is complete and in principle aggregatable, again pointing to
  a reader-side synthesis failure (the run.json answer said "three babies").
- **21d02d0d** (fun runs missed, passing case): both misses are captured as separate
  `miss`-labeled edges (`User→March 5th`, `Run→March 26th`). This is the clean
  success case — extraction, retrieval, and reading all worked, matching its "ok"
  triage bucket.
- **2b8f3739 / 36b9f61e / 129d1232** (all SUM/money questions): this is the sample's
  dominant failure. Across the 3 SUM questions, 9 distinct dollar amounts were
  enumerated in ground truth; only **3 of 9 (33%)** show up anywhere in the graph as
  a node name or edge target — the rest are simply never extracted as usable
  numbers, even though the surrounding entities (market names, dates, item names)
  *are* captured. `129d1232` is also flagged `join_miss` in the original eval
  (retrieval, not reading, was blamed) — but even with perfect retrieval, a SQL SUM
  here would have been badly wrong regardless, because the amounts themselves were
  never captured at ingestion time.

## Aggregate numbers

- True occurrences enumerated across all 8 questions: **26**
- CAPTURED: **17** (65%)
- COLLAPSED (extractor saw it, schema flattened it — fixable by prompt/schema change): **2** (8%)
- MISSING (extractor never saw it — harder problem): **7** (27%)

Split by question type:
- **Discrete-event counting** (00ca467f, 2788b940, 2e6d26dc, 21d02d0d, 0a995998 — 17
  occurrences): 14 captured, 2 collapsed, 1 missing/unclear. Naive `COUNT` ties gold
  exactly on 3/5 questions and undercounts by exactly 1 on the other 2 — close, but
  not reliable enough to trust blindly (an off-by-one in a "how many" answer is still
  a wrong answer).
- **SUM/amount questions** (2b8f3739, 36b9f61e, 129d1232 — 9 amounts): 3 captured, 0
  collapsed, 6 missing. Naive `SUM` is catastrophically wrong on all 3 (total miss,
  48% short, 96% short) — this is not a rounding problem, it's amounts not making it
  into the graph as parseable, addressable facts at all.

## Are quantities parseable when present?

Where a dollar amount *does* appear (`$800`, `$500`, `$250`), it's stored as a plain
`entity` node whose `name` is the literal string `"$800"` (not a typed/numeric
field), linked to its context via an ordinary `RELATED_TO` edge. It's parseable with
a regex, but there's no schema guarantee an amount node exists at all for a given
event — as shown above, most don't.

## Verdict

Shipping `COUNT`/`SUM` over today's edges would be a **mixed-to-bad bet**, and the
two question shapes should not be treated the same:

- For **discrete "how many distinct X" questions**, the graph is often complete
  enough that a naive count ties the gold answer, but not reliably — one of the two
  failure cases in this sample (2788b940) came from a real, fixable schema gap
  (compound multi-value nodes collapsing 2 facts into 1 edge). A SQL layer here
  would sometimes tie the current LLM reader and sometimes silently return an
  off-by-one — better than nothing only if paired with a fix for compound-value
  nodes and a way to select the "right" relation type per question.
- For **SUM/amount questions**, SQL aggregation would lose badly and consistently —
  3/3 in this sample. This isn't an aggregation-logic problem, it's an extraction
  problem: numeric amounts are being dropped well before they'd ever reach a SQL
  layer. Shipping SUM today would produce exactly the "confidently wrong number"
  scenario this spike was meant to catch. This needs an extractor/schema fix (e.g.
  forcing amount mentions into a typed, always-extracted field) before SUM over the
  graph is trustworthy at all.

**Bottom line:** don't ship a SQL aggregation layer yet. COUNT-style aggregation is
close enough to be worth revisiting after fixing the compound-value collapse; SUM
is not close and needs the extractor to stop dropping numeric quantities first.

## Artifacts

- `build_stores.py` — ingests the 8 target instances into `stores/<qid>.db`
- `ground_truth.py` / `ground_truth.json` — LLM-enumerated occurrences per question
- `dump_edges.py` — dumps non-structural edges for a given episode for manual review
- `spot_check_00ca467f.txt`, `spot_check_2788b940.txt`, `spot_check_2b8f3739.txt` —
  raw session text dumps used for manual verification
