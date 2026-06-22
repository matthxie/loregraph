# dataset/

Test corpus for the knowledge-graph prototype. Text and images are **independent** (not paired). Built by `scripts/build_dataset.py` via streaming (reproducible, fixed seed 42).

## Contents

- `wikipedia/articles.jsonl` — 100 random full Wikipedia articles (`{id, title, url, text, char_len}` per line). Source: `wikimedia/wikipedia` `20231101.en`, license **CC BY-SA 4.0** (attribution + share-alike).
- `images/*.jpg` + `images/manifest.jsonl` — 100 random real photos, ~640px, varied everyday subjects (`{id, file, source, label}` per line, `label` = COCO object categories). Source: `detection-datasets/coco`.

Regenerate: `python scripts/build_dataset.py`
