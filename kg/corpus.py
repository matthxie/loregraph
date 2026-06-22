"""Load the frozen test corpus from dataset/ (docs/DATASET.md).

Text and images are independent (not paired): 100 full Wikipedia articles +
100 COCO photos. Each yields a normalized record the ingestion pipeline consumes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dataset")


@dataclass
class CorpusItem:
    id: str
    modality: str          # "text" | "image"
    source_ref: str        # url / file path
    title: str = ""
    text: str | None = None
    image_path: str | None = None
    label_hint: str | None = None  # COCO labels — offline VLM stand-in
    created_at: str | None = None  # corpus item's own time (mixed stream); None → wall clock


def load_articles(path: str | None = None, limit: int | None = None) -> list[CorpusItem]:
    path = path or os.path.join(DATASET_DIR, "wikipedia", "articles.jsonl")
    items: list[CorpusItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items.append(CorpusItem(
                id=r["id"], modality="text", source_ref=r.get("url") or r["id"],
                title=r.get("title", ""), text=r.get("text", "")))
            if limit and len(items) >= limit:
                break
    return items


def load_images(manifest: str | None = None, limit: int | None = None) -> list[CorpusItem]:
    manifest = manifest or os.path.join(DATASET_DIR, "images", "manifest.jsonl")
    items: list[CorpusItem] = []
    base = os.path.dirname(manifest)
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items.append(CorpusItem(
                id=r["id"], modality="image",
                source_ref=os.path.join(base, os.path.basename(r["file"])),
                image_path=os.path.join(base, os.path.basename(r["file"])),
                label_hint=r.get("label")))
            if limit and len(items) >= limit:
                break
    return items


def load_mixed(manifest: str | None = None, limit: int | None = None) -> list[CorpusItem]:
    """Load the per-paragraph temporal stream from dataset/mixed/manifest.jsonl.

    Built by scripts/build_mixed.py: each Wikipedia paragraph (and each COCO photo)
    is its own record stamped with a synthetic `created_at`. Carrying that timestamp
    through lets the graph's created_at/valid/superseded_by machinery see real
    spread-out times instead of identical wall-clock stamps. Text records read their
    paragraph body from the sibling `.txt` file; image records mirror load_images
    (photo path + COCO label hint). Provenance is `orig_id#pNNN` (orig article/image
    + paragraph index); since every paragraph is a distinct fact, this is an append
    stream, not a supersession stream.
    """
    manifest = manifest or os.path.join(DATASET_DIR, "mixed", "manifest.jsonl")
    base = os.path.dirname(manifest)
    items: list[CorpusItem] = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pidx = r.get("para_index")
            cid = r["orig_id"] if pidx is None else f"{r['orig_id']}#p{pidx:03d}"
            path = os.path.join(base, os.path.basename(r["file"]))
            if r["modality"] == "image":
                items.append(CorpusItem(
                    id=cid, modality="image", source_ref=path,
                    title=r.get("title") or "", image_path=path,
                    label_hint=r.get("label"), created_at=r.get("created_at")))
            else:
                with open(path, encoding="utf-8") as tf:
                    text = tf.read()
                items.append(CorpusItem(
                    id=cid, modality="text", source_ref=r.get("url") or path,
                    title=r.get("title") or "", text=text,
                    created_at=r.get("created_at")))
            if limit and len(items) >= limit:
                break
    return items


def load_corpus(n_text: int | None = None, n_image: int | None = None) -> list[CorpusItem]:
    return load_articles(limit=n_text) + load_images(limit=n_image)
