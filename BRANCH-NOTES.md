# `new-approach` — paired experimental branch

This branch is **half of a two-repo experiment**. Its other half is the `new-approach`
branch of [`foca-apps/brainbrain`](https://github.com/foca-apps/brainbrain), and the two
only work together.

**Not aimed at `main`.** Do not merge, rebase, or reconcile this branch with `main` by
reflex — it may never land there. `main` moving on is expected and fine.

## What changed here, and why the app needs it

The paired app runs every model call on OpenAI's gpt-5 families (`gpt-5.4-nano` extraction,
`gpt-5.6-luna` answering). Three things follow:

- **Extraction and answering send gpt-5-family params.** `max_completion_tokens` instead of
  `max_tokens`, no `temperature`, and `reasoning_effort: "none"` on 5.4/5.6. Without the
  extractor half, *every* live extraction fails with `invalid_request_error` — this is not a
  tuning change, it is the difference between working and not.
- **The reflexion recall pass is gated** to entries over `reflexion_min_chars` (400). Short
  notes have no room to hide an omission, so the second call was doubling spend for nothing.
- **`kg/metering.py` prices the 5.4/5.6 models.** The app bills one daily USD budget and
  cannot see engine token counts, so it consumes this meter's `cost_usd` over the wire
  (reported per request by the app's daemon on drain / chat / import). An unpriced model
  reads as free there, which is why these rows matter beyond dashboards.

## Version

`pyproject.toml` carries a **local version segment** — `0.1.17+new.approach`. That is
deliberate: a `0.1.17` released from `main` can never collide with a wheel built here, so
pip cannot silently keep one when the other was meant. Bump the base version only if this
work merges.

## Rebuilding the app's wheel after a change here

The app vendors a wheel; editing this repo alone does not update it (an editable dev install
hides that, and a packaged build then fails):

```sh
cd /Users/Projects/loregraph && .venv/bin/pip wheel --no-deps -w /tmp/w .
cp /tmp/w/you_kg-*.whl <brainbrain>/engine/vendor/    # then update the pin in engine/requirements.txt
```

## Known inherited failure

`tests/test_engine.py::test_unsupported_provider_kinds_raise` fails on this branch **and on
`main`**: commit `4a5f287` added `"gemini"` to `SUPPORTED_KINDS` without updating the test
that asserts it is unsupported. Left alone here on purpose — it is main's to fix.
