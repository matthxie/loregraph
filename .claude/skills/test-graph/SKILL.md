---
name: test-graph
description: Run the knowledge-graph test harness and open its dashboard. Use whenever the user asks to "test the input and query on the full dataset", "test the input and query", "run the graph test", "test the graph", "test ingest and query", "run a test run", or otherwise wants to exercise the kg pipeline end-to-end (ingest the temporal dataset + run the eval queries) and see cost / tokens / tags / accuracy in the dashboard. Do NOT use for unit tests (`pytest`) or the recall@k ablation (`kg eval`).
---

# Test the graph: input + query → dashboard

This runs `python -m kg testrun`, which ingests a **LongMemEval tier**
(`dataset/longmemeval/<tier>/`, default `--tier small`) one chat session at a time (the
Input view: watch the structure form, with cost/tokens/tags/temporal charts) and runs that
tier's `questions.jsonl` through the agentic `ask` (the Query view: traversal replay +
recall@k/MRR/citation-grounding + an LLM-judge response score). It writes `runs/<id>/run.json`
+ a self-contained `dashboard.html`, then the dashboard server lets you browse every run and
toggle Input ⇄ Query.

> **NB on accuracy:** `testrun` pools the whole tier into ONE shared graph (a scale/structure
> view), which cross-contaminates LongMemEval's per-user instances — so treat its accuracy as
> a smoke signal, not a real LongMemEval score (the correct protocol is per-instance; see
> `dataset/longmemeval/README.md`). The Input view + cost/token/structure panels are fully valid.

## Steps

1. **Pick the interpreter.** Prefer the project venv (it has the deps + the API key):
   use `.venv/bin/python` if `.venv/bin/python` exists, else `python3`. Call it `$PY`.

   **The tier must be built first** (episode bodies are gitignored): if
   `dataset/longmemeval/<tier>/episodes.jsonl` is missing, run
   `$PY scripts/build_longmemeval.py` (builds `sample`+`small`+`med`; downloads ~277 MB once).
   The committed `sample` tier needs no build.

2. **Scope from the user's words.**
   - tier: "full"/"med" → `--tier med` (500 Q / ~24k sessions); default/"small" → `--tier small`
     (100 Q / ~4.7k sessions); "quick"/"smoke" → `--tier sample` (8 Q / 48 sessions, committed).
   - a number / cap → add `--limit N` (session episodes) and/or `--queries M`.
   - "offline" / "free" / "deterministic" → add `--backend offline --extractor heuristic --embedder hashing` (no API calls, $0; cost panels read zero).

3. **Heads-up on cost before a *live* run.** Live ingest is ~one Haiku extraction per
   session: `small` ≈ 4.7k calls, `med` ≈ 24k calls (+ the queries +judge) — real money and
   many minutes. For a live run prefer `--tier sample` or `--tier small --limit N`; confirm
   before `--tier med` live. Offline is free. Mention it, then proceed.

4. **Run the harness** (long-running — run it in the background and stream progress):
   ```
   $PY -m kg testrun --tier small        # add flags from step 2/3
   ```
   Useful flags: `--tier {sample,small,med,large}`, `--limit`, `--queries`,
   `--no-judge` (skip the LLM judge), `--no-communities` (faster), `--label my-run`,
   `--model claude-sonnet-4-6`, `--out runs`. It prints a summary (sessions → nodes/edges,
   avg tags/obj, tokens, $, then recall@k / MRR / hit-rate / grounding / judge-accuracy).

5. **Open the dashboard.** Start the server in the background and give the user the URL:
   ```
   $PY -m kg dashboard --out runs --port 8050   # http://127.0.0.1:8050
   ```
   The index lists every run; clicking one opens the Input/Query toggle. (Each run also
   has a standalone `runs/<id>/dashboard.html` you can open directly without the server.)

6. **Report** the printed summary (cost, tokens, node/edge counts, avg tags per node,
   recall@k, response accuracy) and the dashboard URL.
