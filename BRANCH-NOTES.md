# `staging` — the app-integration branch

This branch is what the [`foca-apps/brainbrain`](https://github.com/foca-apps/brainbrain)
app's vendored engine wheel is built from (pairing with brainbrain's `staging` branch).
It absorbed the retired `new-approach` experiment branch on 2026-08-04 — that branch's
gpt-5-family compatibility, billing fixes, and reflexion gating all live here now.

**Not aimed at `main`.** Merging `main` INTO this branch is fine and expected — take
upstream's fixes. Landing app-specific pins or the version suffix on `main` is not.

## What this branch carries beyond `main`, and why the app needs it

The app runs every model call on OpenAI's gpt-5 families via its baked app key
(`gpt-5.4-nano` extraction, `gpt-5.4-mini` answering — `APP_KEY_MODELS` in
brainbrain's `electron/src/shared/bridge.ts`):

- **Extraction and answering send gpt-5-family params.** `max_completion_tokens` instead
  of `max_tokens`, no `temperature`, and `reasoning_effort: "none"` on 5.4/5.6. Without
  the extractor half, *every* live extraction fails with `invalid_request_error` — this
  is not a tuning change, it is the difference between working and not.
- **`Config` honors the app's model pins.** brainbrain sets `KG_LLM_MODEL` /
  `KG_RAG_MODEL` in the daemon child's env; `Config.__post_init__` fills unset
  `llm_model` / `rag_model` from them (explicit values win).
- **The reflexion recall pass is gated** to entries over `reflexion_min_chars` (400).
  Short notes have no room to hide an omission, so the second call was doubling spend
  for nothing.
- **`kg/metering.py` prices the 5.4/5.6 models** and bills cache-write tokens at their
  real rate. The app bills one daily USD budget and cannot see engine token counts, so
  it consumes this meter's `cost_usd` over the wire. An unpriced model reads as free
  there; keep the table in lockstep with brainbrain's `electron/src/main/pricing.ts`.
- **`kg.BRANCH = "staging"`** — carried in code (not package metadata) so the paired
  app can show which engine line is loaded even under a stale editable install.

## Version

`pyproject.toml` carries a **local version segment** — `0.1.19+staging`. That is
deliberate: a `0.1.19` released from `main` can never collide with a wheel built here,
so pip cannot silently keep one when the other was meant. The base version tracks the
last `main` merged in; bump the local segment's trailing number (`.2`, `.3`, …) for
wheel iterations between merges.

## Rebuilding the app's wheel after a change here

The app vendors a wheel; editing this repo alone does not update it (an editable dev
install hides that, and a packaged build then fails):

```sh
cd /Users/Projects/loregraph && .venv/bin/pip wheel --no-deps -w /tmp/w .
cp /tmp/w/you_kg-*.whl <brainbrain>/engine/vendor/    # then update the pin in engine/requirements.txt
```
