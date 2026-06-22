# dataset/

Test corpus for the knowledge-graph prototype. Text and images are **independent** (not paired). Built by `scripts/build_dataset.py` via streaming (reproducible, fixed seed 42).

## Contents

- `wikipedia/articles.jsonl` — 100 random full Wikipedia articles (`{id, title, url, text, char_len}` per line). Source: `wikimedia/wikipedia` `20231101.en`, license **CC BY-SA 4.0** (attribution + share-alike).
- `images/*.jpg` + `images/manifest.jsonl` — 100 random real photos, ~640px, varied everyday subjects (`{id, file, source, label}` per line, `label` = COCO object categories). Source: `detection-datasets/coco`.
- `mixed/` — the text and image items **combined into one folder** with hashed filenames. Each item is `<sha1(orig_id)[:16]>.txt` (article raw text) or `<hash>.jpg` (copied photo), so a directory listing **interleaves text and images in a basically-random order** instead of grouping by type. `mixed/manifest.jsonl` (sorted by hash) preserves the mapping: `{id (hash), file, modality, orig_id, title, url, label}`. Derived from the two folders above (which are left untouched).

Regenerate: `python scripts/build_dataset.py` (the two source folders), then `python scripts/build_mixed.py` (the combined `mixed/` folder).
