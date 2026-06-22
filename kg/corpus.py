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


def load_corpus(n_text: int | None = None, n_image: int | None = None) -> list[CorpusItem]:
    return load_articles(limit=n_text) + load_images(limit=n_image)
