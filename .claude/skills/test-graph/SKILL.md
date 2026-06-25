---
name: test-graph
description: Run the knowledge-graph test harness and open its dashboard. Use whenever the user asks to "test the input and query on the full dataset", "test the input and query", "run the graph test", "test the graph", "test ingest and query", "run a test run", or otherwise wants to exercise the kg pipeline end-to-end (ingest the temporal dataset + run the eval queries) and see cost / tokens / tags / accuracy in the dashboard. Do NOT use for unit tests (`pytest`) or the recall@k ablation (`kg eval`).
---

# Test the graph: input + query → dashboard

This runs `python -m kg testrun`, which ingests a **LongMemEval tier**
(`dataset/longmemeval/<tier>/`, default `--tier micro`) one chat session at a time (the
Input view: watch the structure form, with cost/tokens/tags/temporal charts) and runs that
tier's `questions.jsonl` through the `ask` flow (the Query view: PPR→RAG trace +
recall@k/MRR/citation-grounding + an LLM-judge response score). It writes `runs/<id>/run.json`
+ a self-contained `dashboard.html`, then the dashboard server lets you browse every run and
toggle Input ⇄ Query.

**The pipeline is LIVE-ONLY** (the offline heuristic/hashing/offline-answerer backends were
removed). Every run calls Claude Haiku for extraction + answering and uses a local bge
embedder, so a run needs `ANTHROPIC_API_KEY` (kg auto-reads the project-root `.env`) and
costs real (small) money. The default `micro` tier is sized so a full live run is cheap and
finishes in well under a minute.

> **NB on protocol:** `testrun` defaults to `--mode per-instance` — the dataset's native
> protocol (a fresh graph per question, no cross-user pooling), so accuracy is meaningful.
> `--mode shared` pools the whole tier into ONE graph (a scale/structure smoke view) and
> cross-contaminates instances — use it only for the structure visual, not for accuracy.

## Steps

1. **Pick the interpreter.** Use the project venv (it has the deps + the API key):
   `.venv/bin/python` if it exists, else `python3`. Call it `$PY`. A live run needs the key —
   if `ANTHROPIC_API_KEY` is unset and there's no `.env`, stop and tell the user.

   **Tier build:** `micro` and `sample` ship their episodes (committed) — no build needed.
   For `small`/`med`/`large`, if `dataset/longmemeval/<tier>/episodes.jsonl` is missing run
   `$PY scripts/build_longmemeval.py` (downloads ~277 MB once). Rebuild `micro` itself with
   `$PY scripts/build_micro.py`.

2. **Scope from the user's words.**
   - tier: default / "quick" / "smoke" → `--tier micro` (3 instances / 18 sessions, committed —
     the cheap default). "sample" → `--tier sample` (8 instances / 48 sessions). "small" →
     `--tier small` (100 Q / ~4.7k sessions). "full"/"med" → `--tier med` (500 Q / ~24k sessions).
   - a number / cap → add `--queries M` (per-instance: caps instances) and/or `--limit N`
     (shared mode: caps session episodes).

3. **Heads-up on cost before a bigger run.** Every run is live (~one Haiku extraction per
   session + the queries + judge). `micro` ≈ 18 sessions (cents, seconds). `sample` ≈ 48.
   `small` ≈ 4.7k sessions, `med` ≈ 24k — real money and many minutes; confirm before `small`+
   and especially `med`. Default to `micro`; mention the cost of anything larger, then proceed.

4. **Run the harness** (long-running — run it in the background and stream progress):
   ```
   $PY -m kg testrun --tier micro        # add flags from step 2/3
   ```
   Useful flags: `--tier {micro,sample,small,med,large}`, `--mode {per-instance,shared}`,
   `--queries`, `--limit`, `--no-judge` (skip the LLM judge), `--no-communities` (faster),
   `--label my-run`, `--model claude-sonnet-4-6`, `--out runs`. It prints a summary
   (sessions → nodes/edges, avg tags/obj, tokens, $, then recall@k / MRR / hit-rate /
   grounding / judge-accuracy).

5. **Open the dashboard.** Start the server in the background and give the user the URL:
   ```
   $PY -m kg dashboard --out runs --port 8050   # http://127.0.0.1:8050
   ```
   The index lists every run; clicking one opens the Input/Query toggle. (Each run also
   has a standalone `runs/<id>/dashboard.html` you can open directly without the server.)

6. **Report** the printed summary (cost, tokens, node/edge counts, avg tags per node,
   recall@k, response accuracy) and the dashboard URL.
