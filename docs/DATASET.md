# Test Corpus Dataset

**Decision (2026-06-21):** use a ready-made multimodal Wikipedia dataset instead of live scraping.

## Pick: `wikimedia/wit_base`

Hugging Face dataset id `wikimedia/wit_base` (`split="train"`, `streaming=True`).

It is the only vetted candidate that ships **embedded image bytes + real page/section-level
text in one row**, under a clear **CC BY-SA 4.0** license, and **streams ~100 examples in
seconds** without touching the 308 GB full corpus. Every other text-rich option (WikiWeb2M,
Encyclopedic-VQA, M-BEIR) delivers images as **URLs/paths only** and requires multi-GB or
multi-store joins — disqualifying for a solo dev who wants bytes-in-hand with minimal friction.

**Runner-up — WikiWeb2M:** choose only if true full-article body text at whole-article
granularity is non-negotiable, and you accept installing TensorFlow, streaming multi-GB
TFRecord shards, and fetching image bytes yourself over HTTP.

## Comparison

| Name | Unit | Text | Images | Size | License | Load-100 | Fit |
|---|---|---|---|---|---|---|---|
| **wikimedia/wit_base** | per-image (+ its page/section context) | page summary + section context (nested, multilingual) | **bytes, embedded, 300px** | 308 GB / stream a slice | CC BY-SA 4.0 | **easy** | **8/10** |
| WikiWeb2M | per-page (full article) | full article body + sections | URL only | multi-GB TFRecord | CC BY-SA 3.0 | hard (TF, no HF) | 4/10 |
| Encyclopedic-VQA KB | per-article | full article body | URL only (+174 GB AToMiC join for bytes) | 4.9 GB + 174 GB | unclear/mixed | hard | 3/10 |
| M-BEIR | retrieval candidate | short entity snippet | path in 169 GB tar.gz | 176 GB | MIT (pkg) | hard | 2/10 |

## The tradeoff (read this)

`wit_base` rows are **image-anchored, not whole-article**. One row = one image + the page/section
text context around it. Dedupe by `page_url` to get ~one node per page, but each node carries only
that page's captured context (summary + a section), **not the full article body**. We accept this
for ease + reproducibility + bytes-in-hand.

**What we lose vs. the live Wikipedia API** (and how we compensate):

| Lost (free from the API) | Compensation |
|---|---|
| `[[wikilinks]]` → natural directed edges between articles | **Derive edges**: shared tags/entities (overlap-weighted) + embedding-kNN `SIMILAR_TO`. This makes the *tag graph the primary edge source* — which is more on-mission for an LLM-traversable-via-tagging graph anyway. |
| Categories / infobox templates → free typing & clustering | Same derived-edge approach; optionally enrich (below). |
| Full article body | Section + page summary is enough for summarize→tag→graph; switch to WikiWeb2M if downstream quality is thin. |
| Whole-article unit | `page_url` dedupe recovers ~one node per page. |

**Optional enrichment (best of both):** each node keeps `page_url`, so a *one-time offline*
Wikipedia API call per node can pull real categories/links and add them as **ground-truth edges**
— keeping the corpus frozen/deterministic while restoring the deterministic edge types, useful for
eval ablations (±deterministic edges).

## Loader

```bash
pip install "datasets>=2.18" pillow
```

```python
import json, os
from datasets import load_dataset

OUT = "corpus"; os.makedirs(f"{OUT}/images", exist_ok=True)

# Stream so we never pull the full ~308 GB.
ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)

records, seen_pages = [], set()
for row in ds:
    feats = row["wit_features"]              # column-oriented: dict of parallel lists
    langs = feats.get("language") or []
    i = langs.index("en") if "en" in langs else 0   # prefer English

    def f(key):
        col = feats.get(key)
        return col[i] if col and i < len(col) and col[i] else None

    page_url = f("page_url")
    if not page_url or page_url in seen_pages:       # dedupe to one row per page
        continue

    text = "\n".join(t for t in [
        f("page_title"),
        f("context_page_description"),    # article summary
        f("context_section_description"), # section body
    ] if t)
    if len(text) < 80:                    # skip near-empty context rows
        continue

    img = row["image"]                              # PIL.Image, embedded bytes
    img_path = f"{OUT}/images/node_{len(records):03d}.jpg"
    img.convert("RGB").save(img_path, "JPEG", quality=90)

    seen_pages.add(page_url)
    records.append({
        "id": f"node_{len(records):03d}",
        "title": f("page_title"),
        "text": text,
        "image_path": img_path,
        "page_url": page_url,
        "image_attribution": row.get("caption_attribution_description"),
    })
    if len(records) >= 100:
        break

with open(f"{OUT}/nodes.jsonl", "w") as out:
    for r in records:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} nodes -> {OUT}/nodes.jsonl (+ images/)")
```

## Caveats

- **License — CC BY-SA 4.0 (share-alike).** Attribution required; the loader keeps `page_url` +
  `image_attribution`. Redistributed derivatives of the *content* inherit share-alike. Fine for an
  internal test corpus; note it if you publish. Individual images carry their own per-file licenses.
- **URL rot — avoided** (pixels are embedded).
- **Resolution:** images downscaled to 300px width — fine for tagging/embedding.
- **Language:** rows are multilingual (108 langs); loader filters to English and skips thin context.
- **NSFW/filtering:** unfiltered Wikimedia Commons content — eyeball the 100 JPEGs; add a filter for
  larger pulls.
- **Text depth:** section/page context, not full article body.
