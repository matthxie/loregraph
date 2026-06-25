# dataset/longmemeval/

The test corpus for the knowledge-graph prototype: three sized tiers (plus a tiny
committed `sample`) derived from **LongMemEval**, a benchmark for the long-term memory of
chat assistants. This replaced the earlier frozen Wikipedia/COCO corpus (the old
`dataset/{wikipedia,images,mixed}/`), which couldn't exercise the thing this system is
actually built for: dated, multi-session memory with updates over time.

## Where the data comes from

| | |
|---|---|
| **Benchmark** | LongMemEval — *Benchmarking Chat Assistants on Long-Term Interactive Memory* (Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu), ICLR 2025 · [arXiv:2410.10813](https://arxiv.org/abs/2410.10813) |
| **Project / repo** | https://github.com/xiaowu0162/longmemeval · https://xiaowu0162.github.io/long-mem-eval/ |
| **Source release** | Hugging Face dataset [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (files: `longmemeval_s_cleaned.json` 277 MB, `longmemeval_m_cleaned.json` 2.74 GB, `longmemeval_oracle.json` 15 MB) |
| **License** | **MIT** |
| **What we keep** | Only what `scripts/build_longmemeval.py` derives into the tiers below. The multi-GB raw download is fetched **transiently** into `.cache/longmemeval/` (gitignored) and deleted after the build — we don't store the whole thing. |

LongMemEval ships **exactly 500 hand-curated question instances**. The `S`, `M`, and
`oracle` variants are the *same* 500 questions; they differ only in how many distractor
sessions pad each history (`S` ≈ 50 sessions / ~115k tokens, `M` ≈ 500 sessions / ~1.5M
tokens, `oracle` = evidence sessions only). It targets the five abilities this episodic,
bi-temporal graph is designed for: information extraction, multi-session reasoning,
**temporal reasoning**, **knowledge updates**, and **abstention**. It is, in effect, the
real graded version of the hand-built `kg.synthetic` Becky stream.

## What one datapoint is

One **instance** = one user's long chat *history* + a question asked later + the answer +
a pointer to the evidence:

- a **haystack** of dated chat sessions (`haystack_sessions`, each a list of
  `{role, content}` turns, each session timestamped via `haystack_dates`). Most sessions
  are distractors; a few hold the evidence.
- a **question** + `question_date` (ask "as of" that time).
- the gold **answer**.
- `answer_session_ids` — which sessions actually contain the answer (the retrieval target).

## The tiers

Only 500 distinct questions exist, so we honour `small=100` / `med=500` and let **large**
scale the way LongMemEval itself does — by history **depth** (the `M` variant), not by
question count (there is no set of 1000 distinct questions to draw from).

| tier | instances (questions) | haystack/instance | source variant | episodes | committed? |
|---|---|---|---|---|---|
| **sample** | 8 | ≤6 sessions (capped) | `longmemeval_s` | 48 | ✅ episodes too (offline fixture) |
| **small** | 100 | ~50 sessions | `longmemeval_s` | 4,723 | only `questions.jsonl` + `manifest.json` |
| **med** | 500 (all) | ~50 sessions | `longmemeval_s` | 23,867 | only `questions.jsonl` + `manifest.json` |
| **large** | 500 (all) | ~500 sessions | `longmemeval_m` | ~250k | build-on-demand (pulls the 2.74 GB `M`) |

Episode bodies are **heavy and regenerable**, so they're gitignored (`.gitignore`:
`dataset/longmemeval/*/episodes.jsonl`, except `sample`). Only the lean `questions.jsonl`
(the graded ground truth) and `manifest.json` are version-controlled. Rebuild any tier
with the script below.

## Ordering — "you can't randomize ordered conversations"

Two different orders, handled deliberately:

- **Within an instance, sessions are time-ordered and must not be shuffled** — temporal
  and knowledge-update questions depend on it. We emit each haystack's sessions **sorted
  by their timestamp** (34/500 instances ship them out of order in the raw file), so each
  instance reads as a clean chronological episode stream the bi-temporal layer can order.
- **Across instances the 500 are independent.** Tier membership is chosen by a
  **deterministic, RNG-free stratified order** (round-robin over the six question types,
  tie-broken by `question_id`), so every tier preserves the type mix and the tiers
  **nest**: `small`'s 100 `question_id`s are a strict prefix of `med`'s/`large`'s 500.

## Files per tier

### `episodes.jsonl` — ingestible episodes (one row per chat *session*)
```json
{"id": "00ca467f__answer_39900a0a_2", "question_id": "00ca467f",
 "session_id": "answer_39900a0a_2", "created_at": "2023-03-27T08:05:00+00:00",
 "date": "2023/03/27 (Mon) 08:05", "modality": "text", "is_evidence": true,
 "n_turns": 12, "text": "[chat session — …]\nUser: …\nAssistant: …"}
```
- `id` = `<question_id>__<session_id>` — session ids are **not** globally unique (they
  collide across instances), so they're namespaced by `question_id`.
- `created_at` is in `kg.store.now_iso()` format and is threaded into the graph's
  `created_at` / `valid_at` machinery as the session's chat time.

### `questions.jsonl` — graded queries (one row per *instance*)
```json
{"id": "00ca467f", "query": "How many doctor's appointments did I go to in March?",
 "kind": "multi-session", "question_date": "2023-03-27T23:35:00+00:00",
 "question_date_raw": "2023/03/27 (Mon) 23:35",
 "gold": ["obj_00ca467f__answer_39900a0a_2", "obj_00ca467f__answer_39900a0a_3"],
 "answer": "2", "abstention": false, "n_evidence": 3, "n_sessions": 47, "source": "longmemeval_s"}
```
- `kind` is the LongMemEval `question_type`: `single-session-user`,
  `single-session-assistant`, `single-session-preference`, `temporal-reasoning`,
  `knowledge-update`, `multi-session` (mix ≈ 14 / 11 / 6 / 27 / 16 / 27 %).
- `abstention` (the 30 `_abs` questions, ~6%): the right answer is "not enough
  information." Their evidence sessions still exist, so recall@k still applies; the
  **answer judge** is what actually scores the abstention.
- `gold` lists the evidence sessions as `obj_<id>`. The harness collapses both `obj_<id>`
  and the ingested `ep_<id>` to the same key (`kg.testrun._article`), so recall@k / MRR
  line up with no special-casing. `answer` is always a string (32/500 are integer counts,
  coerced).

### `manifest.json`
Tier metadata + full provenance + the ordered `question_id` list + build parameters.

## Consumption — run it **per instance**

LongMemEval questions are all first-person ("I/me") about *the user*. Ingesting all 500
instances into one shared graph would pool 500 different users' lives into a single memory
and scramble every knowledge-update — so the correct protocol is **a fresh graph per
question, ingesting only that instance's haystack**:

```python
from kg import Config, KnowledgeGraph
from kg.corpus import iter_lme_instances

cfg = Config.default()
for q, sessions in iter_lme_instances("small"):     # (question, its haystack)
    g = KnowledgeGraph.open(":memory:", cfg)         # fresh memory per instance
    g.ingest(sessions)
    ans = g.ask(q["query"])                          # score ans.answer vs q["answer"],
    ...                                              # ans.object_ids vs q["gold"]
```

`python -m kg testrun --tier small` instead ingests the **whole tier into one shared
graph** (the dashboard's scale/structure view, and a smoke test). That's fine for watching
the graph form and for cost/token metering, but it cross-contaminates personas — don't
read its accuracy as a LongMemEval score. (The first-person `self_entity` anchor is **off**
by default, which avoids the worst-case collapse, but generic-name entities can still
collide across instances.) Per-instance accuracy harness wiring is the natural next step.

## Build / regenerate

```bash
python scripts/build_longmemeval.py                 # sample + small + med (downloads S, ~277 MB)
python scripts/build_longmemeval.py --tier large    # large (downloads M, ~2.74 GB — heavy)
python scripts/build_longmemeval.py --tier all      # everything
```

Deterministic: same inputs → identical tiers. The raw HF download lands in
`.cache/longmemeval/` and is deleted afterward unless you pass `--keep-cache`.

## Attribution

If you publish results, cite the LongMemEval paper (arXiv:2410.10813) and link the
[repo](https://github.com/xiaowu0162/longmemeval). Data © its authors under the MIT
license; this folder only stores derived subsets + a build recipe.
