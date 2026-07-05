# Query-side reader fixes: sibling expansion, honest fact dates, abstention/supersede discipline, numeric judge_suspect triage

**Goal:** the last 100-instance run (`runs/small-fixes-1/run.json`, 55% accuracy)
showed retrieval was nearly solved (hit_rate 0.99) but answers died during context
assembly and reading. Three hand-verified failure patterns drove this work:
sibling-chunk gaps, facts-block overconfidence (a session date dressed up as a
place/date fact), and cumulative-vs-increment miscounts. All changes below are
**query-side only** — nothing under `kg/extractors.py`, `kg/cues.py`, `kg/ingest.py`,
or any `INGEST_RELEVANT_FIELDS` config field changed, so the 100 cached ingest
stores under `store/cache/` are untouched.

## What changed

### Fix 1 — sibling-chunk (parent) expansion, new flag, default OFF
`kg/rag.py` `ContextBuilder._expand_siblings` (called from `build()` right after
`_select_episodes`): for each selected chunk `<source>#cNNN`, pulls in its
`#cNNN±w` siblings from the store, in document order (contiguous by chunk index
within a source), deduped, capped by a hard character budget. Sources are expanded
in their original rank order, so when the budget is hit it's the lowest-ranked
source's siblings that get cut — the originally-selected top-n chunks are always
kept regardless of budget.

New `Config` fields (`kg/config.py`):
- `rag_parent_expand: int = 0` — sibling radius; **0 = off, byte-identical context**
  (verified: `_expand_siblings` returns the input list unchanged when 0).
- `rag_expand_budget_chars: int = 60000` — hard cap on total episode text after
  expansion.

Expansion siblings are context-only: they enter the `EPISODES` block and are legal
citation targets (`_validate` checks against the same `context_episodes` list), and
`gold_marks[].in_context` (testrun diagnostics) correctly picks them up because it's
computed from `ans.context_episodes` — but retrieval metrics (`recall@k`, `gold_ranks`,
`rank`, `hit`) are computed from `ans.object_ids` (the raw PPR-ranked list), which
`_expand_siblings` never touches. Confirmed by reading `kg/testrun.py` `_score_query`
and `_diagnose` — no changes needed there.

### Fix 2 — generic `--set field=value` config override
No CLI override mechanism existed for arbitrary `Config` fields, so added
`--set FIELD=VALUE` (repeatable) to `python -m kg testrun` (`kg/cli.py`
`_apply_config_override`), coercing the value to the field's existing type
(bool/int/float/str). `rag_chunks_per_source` (already existed, default 4) is now
overridable this way too, e.g. `--set rag_chunks_per_source=2`.

### Fix 3 — honest fact dates + abstention discipline
**3a** `kg/facts.py` `FactLine.render`: an open fact's `valid_at` is usually just the
asserting session's `created_at`, not a confirmed event date. Now renders
`(mentioned YYYY-MM-DD)` when `valid_at` is present with no `invalid_at`; `since
YYYY-MM-DD; until YYYY-MM-DD` is reserved for a real closed bi-temporal window
(`invalid_at` present) — the STATE-lane HISTORY block (`kg/rag.py:210-215`, closed+open
trajectory) is unaffected since closed facts keep the old wording.

**3b** `kg/rag.py` `_RAG_SYS`: added "the FACTS lines are machine-extracted and may be
wrong or mis-dated; the EPISODES text is the ground truth" and extended the
subject-verification clause to explicitly cover place names and dates ("if asked about
city/venue X and the context only covers a different city/venue Y, say the information
is not available").

### Fix 4 — knowledge-update supersede rule
Appended to `_RAG_SYS`: when the same running total is restated at different dates,
the most recent statement supersedes earlier ones — report the latest total, never sum
restatements, only sum amounts that are explicitly separate events.

### Fix 5 — tighten judge_suspect triage
`kg/testrun.py` `_response_proxy`: for a bare-numeric reference (single all-digit
token, e.g. `"25"`), `contains` no longer does plain substring matching — it now
checks that the reference equals the **first number the answer states** (its
asserted value). This kills the false positive where an answer like "...50 new
postcards...includes 17...and 25..." was flagged `judge_suspect` for containing "25"
verbatim while actually asserting 50. Non-numeric references are unchanged (still
plain substring match on the normalized text). Dashboard legend text
(`kg/dashboard.py:891`) updated to describe the new definition.

## Tests

Added to `tests/test_rag.py` (182 passing total, up from 175):
- `test_parent_expand_off_is_noop` — `rag_parent_expand=0` returns the input unchanged.
- `test_parent_expand_pulls_in_sibling_window` — radius-1 window pulls in exactly the
  immediate siblings, contiguous order.
- `test_parent_expand_respects_budget_and_keeps_selected` — a tight budget still keeps
  the originally-selected chunk and stops expanding.
- `test_parent_expand_lowest_ranked_source_cut_first` — under a shared budget, the
  better-ranked source's sibling survives and the worse-ranked source's is cut.
- `test_factline_render_mentioned_vs_since_until` — open fact renders `mentioned`,
  closed fact keeps `since`/`until`.
- `test_response_proxy_numeric_reference_requires_asserted_value` — the
  postcards-miscount false positive no longer triggers `contains=True`; a genuinely
  correct numeric answer still does.
- `test_response_proxy_non_numeric_reference_unchanged` — substring behavior preserved
  for non-numeric references.

`.\.venv\Scripts\python.exe -m pytest -q` → **182 passed**, no regressions.

## Live smoke (tier=sample, 8 instances, chunking=turns)

Commands run (`OPENAI_API_KEY` cleared first so the key loads from the repo's `.env`):

```
python -m kg testrun --tier sample --chunking turns --label queryside-smoke-1
python -m kg testrun --tier sample --chunking turns \
    --set rag_parent_expand=2 --set rag_chunks_per_source=2 --label queryside-smoke-2
```

| | smoke-1 (baseline, flags at default) | smoke-2 (`rag_parent_expand=2`, `rag_chunks_per_source=2`) |
|---|---|---|
| ingest cache | **8/8 cached, 0 fresh** | **8/8 cached, 0 fresh** |
| ingest cost | $0.0000 | $0.0000 |
| query tokens | 40,739 | 88,178 |
| query cost | $0.0067 | $0.0138 |
| total cost | **$0.0067** | **$0.0138** |
| recall@k / mrr / hit_rate | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| judge_acc | 0.75 (6/8) | 0.75 (6/8) |
| context_episodes per query | 6–8 | 12–30 |
| dropped citations (total, 8 q) | 1 | 3 |
| crashes | none | none |

Both runs confirm the ingest cache hit on all 8 instances (the config digest excludes
`rag_*` fields, per `kg/ingest_cache.py` `INGEST_RELEVANT_FIELDS`), context sizes grew
as expected under expansion while staying well under the 60,000-char budget, and
citation validation kept working (dropped citations are tracked, not crashes). Total
combined smoke spend: **$0.0205** — comfortably inside the ~$0.05–0.10 target. `judge_acc`
is identical on this 8-question sample; a meaningful accuracy delta needs the full
small-tier (100-instance) run below.

## Commands for the user: small-tier 2×2 (~$0.31 each, thanks to the ingest cache)

```
python -m kg testrun --tier small --chunking turns --label queryside-baseline
python -m kg testrun --tier small --chunking turns --set rag_parent_expand=2 --label queryside-expand
python -m kg testrun --tier small --chunking turns --set rag_chunks_per_source=2 --label queryside-percap
python -m kg testrun --tier small --chunking turns --set rag_parent_expand=2 --set rag_chunks_per_source=2 --label queryside-both
```

Clear `OPENAI_API_KEY` first (`$env:OPENAI_API_KEY = $null`) so the key loads from the
repo's `.env`, and verify `ingest.totals.cached_instances == 100` in each `run.json`
before trusting the cost figure.
