# Chunk-level retargeting (query-side)

## Problem

`ContextBuilder._select_episodes` (kg/rag.py) picks context chunks in PPR chunk-rank
order, then `_expand_siblings` radiates ±radius around whatever chunk ranked. The right
SOURCE session reliably wins seats, but the specific CHUNK that carries the answer can
still be left out — e.g. session wins 2 of its chunks, but the decisive sentence lives in
a chunk 4+ positions away that the PPR diffusion score never surfaced and the sibling
radius never reaches.

Verified against `runs/reader5-queryside-both/run.json`:
- `25e5aa4f` — "Where did I complete my Bachelor's degree in CS?" (UCLA) — evidence session
  won seats, but the "UCLA" chunk (`answer_986de8c3#c004`) wasn't in context.
- `37f165cf` — page-count question — the `416-page novel` chunk (`answer_6b9b2b1e_1#c000`)
  was excluded even though its source won 8 of the 21 context slots.
- `099778bb` — women-in-leadership % — `answer_80d6d664_2#c002` ("women occupy 20 of the
  leadership positions") was the one chunk of that source left out of context.

## What changed (query-side only)

- `kg/retrieval.py`: added `RetrievalResult.seed_scores` (the raw embedding+BM25+link
  seed mass per node id, already computed by `Seeder.seed`) and threaded it through both
  `PPRRetriever.retrieve` and `HybridRetriever.retrieve`. Additive field, default `{}` —
  no existing behavior changed.
- `kg/config.py`: two new `rag_*`-prefixed fields, both off by default:
  - `rag_retarget: str = "off"` — `"off" | "seed" | "seed+lex"`
  - `rag_provenance_promote: bool = False`
- `kg/rag.py`: `ContextBuilder.build()` now runs, between `_select_episodes` and
  `_expand_siblings`, a new `_retarget_chunks` step, and after `_expand_siblings` a new
  `_promote_provenance` step:
  - **seed retarget** — for each source that won seats, refill its slots with that
    source's chunks ranked by raw embedding seed score (`RetrievalResult.seed_scores`)
    instead of PPR chunk order. Same slot count, swaps only, never adds seats.
  - **lexical retarget** (`seed+lex`) — on top of the seed pick, a chunk of the same
    source not currently picked swaps in if it strictly beats a picked chunk on
    question-content-word / digit-token overlap (`node.raw_text` vs. the query, digit
    matches weighted 3x since they're the decisive signal for count/percentage
    questions). Ties never swap, so the top-ranked incumbent chunk survives ties but can
    still be displaced by a strictly better lexical match.
  - **provenance promotion** (`rag_provenance_promote`) — after expansion, pulls in a
    fact's source chunk (`FactLine.episode_id`) when the fact's src/dst names overlap the
    question terms and the chunk isn't already in context, displacing only the
    lowest-ranked expansion sibling (never an originally selected chunk).
  - Swaps/promotions are recorded on `ContextBuilder.last_retargeted` and surfaced on
    `RagAnswer.retargeted` for inspection — bookkeeping only, nothing reads it.
  - All defaults are off/`"off"`, so context is byte-identical to today unless
    `rag_retarget`/`rag_provenance_promote` are set.

## Tests

`tests/test_rag.py` — 4 new focused tests (plus the 20 pre-existing pass unchanged):
- `test_retarget_off_is_noop`
- `test_retarget_seed_swaps_by_embedding_rank`
- `test_retarget_lexical_swap_beats_seed_pick`
- `test_provenance_promote_displaces_expansion_sibling_only`

```
.venv/Scripts/python.exe -m pytest tests/test_rag.py -q
24 passed
```//
(the repo's system Python lacks `sentence-transformers`; the project `.venv` has it — use
`.venv/Scripts/python.exe`.)

## Probe results (`spikes/retarget/probe.py`)

Opens the three cached per-instance stores directly (`store/cache/<qid>-*.db`), runs
`HybridRetriever.retrieve` + `ContextBuilder.build()` with `rag_parent_expand=2,
rag_chunks_per_source=2`, no LLM calls, `off` vs `seed+lex` + `rag_provenance_promote=True`:

```
env -u OPENAI_API_KEY .venv/Scripts/python.exe spikes/retarget/probe.py
```

| instance  | needle              | before | after | result |
|-----------|---------------------|--------|-------|--------|
| 25e5aa4f  | "UCLA"              | True   | True  | already present — see note below |
| 37f165cf  | "440 pages"         | True   | True  | no change (already present) |
| 37f165cf  | "416-page"          | False  | True  | **FIXED** |
| 099778bb  | "women occupy 20"   | False  | True  | **FIXED** |
| 099778bb  | "20%"               | False  | False | not fixed — no chunk contains the literal computed "20%"; the reader would still need to divide 20/100 itself (see below) |

**25e5aa4f note**: under these probe parameters (`rag_parent_expand=2`), "UCLA" already
surfaces in the `off` baseline — but via the **FACTS** section (`CS --position--> UCLA`,
a graph-extracted relationship), not the episode text. The chunk-retargeting fix targets
the EPISODES section specifically; this instance's episode text still doesn't include the
`#c004` "UCLA" chunk under `off`, it's just that the fact edge happens to carry the same
keyword here. Confirmed by locating the match offset in the blob (`query.build_context`
FACTS block, not EPISODES). Retargeting is still exercised correctly for this instance
(chunk selection does change between `off` and `seed+lex`, see printed chunk-id diffs in
probe output) — the needle simply isn't a clean episode-text signal for this one case.

**099778bb "20%" note**: no single chunk states the literal string "20%" — the number
requires combining "100 leadership positions total" (`answer_80d6d664_1#c000`) with
"women occupy 20 of the leadership positions" (`answer_80d6d664_2#c002`). The fix gets
the second (previously-missing) chunk into context; the first was already present in both
`off` and `on`. Getting both source facts into context is the retrieval-side job; doing
the arithmetic is the reader's.

Full chunk-id before/after diffs are printed by the probe for all three instances.

## Suggested full-run cell

```
kg testrun --set rag_retarget=seed+lex --set rag_provenance_promote=true \
           --set rag_parent_expand=2 --set rag_chunks_per_source=2 \
           --label reader5-retarget-seedlex-promote
```

Compare against `runs/reader5-queryside-both` (same `rag_parent_expand`/
`rag_chunks_per_source`, `rag_retarget=off`) to isolate the retargeting delta. Ingest is
untouched (no `INGEST_RELEVANT_FIELDS` field changed), so this reuses the existing
`store/cache/` ingest cache — the run should cost ~query-only tokens, no re-extraction.
