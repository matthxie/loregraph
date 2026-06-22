#!/usr/bin/env python3
"""Build a third corpus folder that *combines* the text + image datasets.

Leaves dataset/wikipedia/ and dataset/images/ untouched. Reads both and writes
dataset/mixed/, where every item's filename is a hash of its original id — so a
plain directory listing interleaves articles and photos in a basically-random
order instead of grouping them by type.

  dataset/mixed/<hash>.txt        one Wikipedia article (raw text)
  dataset/mixed/<hash>.jpg        one photo (copied bytes)
  dataset/mixed/manifest.jsonl    one record per item, sorted by hash:
                                  {id, file, modality, orig_id, title, url, label}

Deterministic: the hash is sha1(orig_id), so re-running reproduces identical
names. The manifest preserves every field needed to ingest the pile (titles for
text nodes, COCO labels as the offline image-description stand-in).

Usage:  python scripts/build_mixed.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dataset")
WIKI = os.path.join(DATASET_DIR, "wikipedia", "articles.jsonl")
IMG_MANIFEST = os.path.join(DATASET_DIR, "images", "manifest.jsonl")
IMG_DIR = os.path.join(DATASET_DIR, "images")
OUT_DIR = os.path.join(DATASET_DIR, "mixed")


def _hash(orig_id: str, taken: set[str]) -> str:
    """sha1(orig_id) → 16 hex chars; extend on the (astronomically unlikely) collision."""
    full = hashlib.sha1(orig_id.encode("utf-8")).hexdigest()
    n = 16
    while full[:n] in taken and n < len(full):
        n += 1
    h = full[:n]
    taken.add(h)
    return h


def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)        # rebuild cleanly
    os.makedirs(OUT_DIR, exist_ok=True)

    taken: set[str] = set()
    records = []

    # text → <hash>.txt
    for r in _read_jsonl(WIKI):
        h = _hash(r["id"], taken)
        fname = f"{h}.txt"
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as out:
            out.write(r.get("text", ""))
        records.append({
            "id": h, "file": fname, "modality": "text", "orig_id": r["id"],
            "title": r.get("title"), "url": r.get("url"), "label": None,
        })

    # images → <hash>.jpg (copied bytes)
    for r in _read_jsonl(IMG_MANIFEST):
        h = _hash(r["id"], taken)
        fname = f"{h}.jpg"
        src = os.path.join(IMG_DIR, os.path.basename(r["file"]))
        shutil.copyfile(src, os.path.join(OUT_DIR, fname))
        records.append({
            "id": h, "file": fname, "modality": "image", "orig_id": r["id"],
            "title": None, "url": None, "label": r.get("label"),
        })

    # sort by hash so the manifest order is randomized too (no type grouping)
    records.sort(key=lambda x: x["id"])
    with open(os.path.join(OUT_DIR, "manifest.jsonl"), "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_text = sum(1 for r in records if r["modality"] == "text")
    n_img = sum(1 for r in records if r["modality"] == "image")
    print(f"wrote {len(records)} items ({n_text} text + {n_img} image) -> {OUT_DIR}")
    print(f"first 6 by listing order: "
          f"{[r['file'] for r in records[:6]]}")


if __name__ == "__main__":
    main()
