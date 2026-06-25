---
name: test-graph
description: Run the knowledge-graph test harness and open its dashboard. Use whenever the user asks to "test the input and query on the full dataset", "test the input and query", "run the graph test", "test the graph", "test ingest and query", "run a test run", or otherwise wants to exercise the kg pipeline end-to-end (ingest the temporal dataset + run the eval queries) and see cost / tokens / tags / accuracy in the dashboard. Do NOT use for unit tests (`pytest`) or the recall@k ablation (`kg eval`).
---

# Test the graph: input + query → dashboard

This runs `python -m kg testrun`, which ingests **all of `dataset/mixed/`** one document
at a time (the Input view: watch the structure form, with cost/tokens/tags/temporal
charts) and runs **all `dataset/retrieval/questions.jsonl`** through the agentic `ask`
(the Query view: traversal replay + recall@k/MRR/citation-grounding + an LLM-judge
response score). It writes `runs/<id>/run.json` + a self-contained `dashboard.html`, then
the dashboard server lets you browse every run and toggle Input ⇄ Query.

## Steps

1. **Pick the interpreter.** Prefer the project venv (it has the deps + the API key):
   use `.venv/bin/python` if `.venv/bin/python` exists, else `python3`. Call it `$PY`.

2. **Scope from the user's words.**
   - "full dataset" / no qualifier → run the whole thing (all 1343 docs + 68 queries).
   - a number / "quick" / "small" / "smoke" → add `--limit N` (docs) and/or `--queries M`.
   - "offline" / "free" / "deterministic" → add `--backend offline --extractor heuristic --embedder hashing` (no API calls, $0; cost panels read zero).

3. **Heads-up on cost before a full *live* run.** A full live run makes ~2500+ Haiku
   calls for ingest plus the 68 queries (+judge) — a few dollars and many minutes. If the
   user said "full" and a key is present, that's expected; otherwise confirm or suggest
   `--limit`. Mention it, then proceed.

4. **Run the harness** (long-running — run it in the background and stream progress):
   ```
   $PY -m kg testrun            # full live run; add flags from step 2/3
   ```
   Useful flags: `--limit`, `--queries`, `--no-judge` (skip the LLM judge),
   `--no-communities` (faster), `--label my-run`, `--model claude-sonnet-4-6`,
   `--out runs`. It prints a summary (docs → nodes/edges, avg tags/obj, tokens, $, then
   recall@k / MRR / hit-rate / grounding / judge-accuracy).

5. **Open the dashboard.** Start the server in the background and give the user the URL:
   ```
   $PY -m kg dashboard --out runs --port 8050   # http://127.0.0.1:8050
   ```
   The index lists every run; clicking one opens the Input/Query toggle. (Each run also
   has a standalone `runs/<id>/dashboard.html` you can open directly without the server.)

6. **Report** the printed summary (cost, tokens, node/edge counts, avg tags per node,
   recall@k, response accuracy) and the dashboard URL.
