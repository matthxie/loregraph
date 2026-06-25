# dataset/

Test corpus for the knowledge-graph prototype.

- **[`longmemeval/`](longmemeval/)** — the current corpus: sized tiers (`sample` /
  `small` / `med` / `large`) derived from **LongMemEval**, a long-term-memory benchmark
  over dated, multi-session chats (Wu et al., ICLR'25; Hugging Face
  `xiaowu0162/longmemeval-cleaned`, MIT). This is what `kg ingest` / `kg testrun` read.
  Build it with `python scripts/build_longmemeval.py`. See
  [`longmemeval/README.md`](longmemeval/README.md) for provenance, schema, the
  ordering policy, and how to consume it (per-instance vs. shared-graph).

- **`retrieval/`** — ⚠️ **legacy.** A hand-authored 68-question recall@k set graded
  against the *old* Wikipedia/COCO corpus (`gold` = `obj_wiki_NNN` / `obj_img_NNN`). That
  corpus was removed (see below), so this file is **orphaned** — kept only for reference;
  the live harness no longer uses it.

## History

The earlier corpus — 100 full Wikipedia articles + 100 COCO photos, plus the
per-paragraph temporal `mixed/` stream — lived in `dataset/{wikipedia,images,mixed}/`
(built by the now-removed `scripts/build_dataset.py` + `build_mixed.py`). It was a frozen
*snapshot* and couldn't exercise evolving, multi-session memory, so it was replaced by
LongMemEval.
