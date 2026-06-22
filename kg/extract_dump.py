"""Per-model extraction dump — run ONLY the extraction step (no graph build, no
canonicalization, no embeddings) over the corpus and serialize what each
extractor/model produces, so different modes (heuristic, Haiku, Sonnet, …) can be
compared side by side.

    python -m kg extract-dump --extractor haiku --model claude-haiku-4-5-20251001 --out store/dump_haiku.jsonl
    python -m kg extract-dump --extractor haiku --model claude-sonnet-4-6        --out store/dump_sonnet.jsonl
    python -m kg extract-dump --extractor heuristic                              --out store/dump_heuristic.jsonl

Each line of the .jsonl is one item's raw Extraction; a companion <out>.summary.json
holds aggregate vocabulary stats (tag/entity/relation-label histograms). See CLAUDE.md.
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .corpus import CorpusItem
from .extractors import Extraction, Extractor, extract_text_sectioned


def _record(item: CorpusItem, ext: Extraction) -> dict:
    return {
        "id": item.id,
        "title": item.title,
        "modality": item.modality,
        "entities": [{"name": e.name, "type": e.type.value} for e in ext.entities],
        "tags": list(ext.tags),
        "relations": [{"source": r.source, "target": r.target,
                       "labels": list(r.labels), "confidence": r.confidence}
                      for r in ext.relations],
        "description": ext.description,
    }


def extract_corpus(extractor: Extractor, items: list[CorpusItem],
                   config: Config) -> tuple[list[dict], list[str]]:
    """Run extraction over every item under the same bounded-concurrency semaphore the
    ingest pipeline uses. A failed item degrades to an empty record (with an `error`)
    so the batch survives; returns (records, error_messages)."""
    def work(item: CorpusItem) -> tuple[dict, str | None]:
        try:
            if item.modality == "image":
                ext = extractor.extract_image(item.image_path, item.label_hint)
            else:
                ext = extract_text_sectioned(extractor, item.text or "", item.title,
                                             config.long_doc_chars)
            return _record(item, ext), None
        except Exception as e:  # noqa: BLE001 — keep the batch alive, record the error
            rec = {"id": item.id, "title": item.title, "modality": item.modality,
                   "entities": [], "tags": [], "relations": [], "description": None,
                   "error": repr(e)}
            return rec, f"{item.id}: {e!r}"

    with ThreadPoolExecutor(max_workers=config.semaphore_limit) as pool:
        pairs = list(pool.map(work, items))
    records = [p[0] for p in pairs]
    errors = [p[1] for p in pairs if p[1]]
    return records, errors


def summarize(records: list[dict], label: str) -> dict:
    """Aggregate vocabulary stats for one mode — the comparable artifact across models."""
    tag_counts: Counter[str] = Counter()
    ent_types: Counter[str] = Counter()
    rel_labels: Counter[str] = Counter()
    uniq_ents: set[str] = set()
    n_ents = n_rels = n_failed = 0
    for r in records:
        if r.get("error"):
            n_failed += 1
        for t in r["tags"]:
            tag_counts[t] += 1
        for e in r["entities"]:
            ent_types[e["type"]] += 1
            uniq_ents.add(e["name"].lower())
            n_ents += 1
        for rel in r["relations"]:
            n_rels += 1
            for lab in rel["labels"]:
                rel_labels[lab] += 1
    return {
        "label": label,
        "items": len(records),
        "failed": n_failed,
        "entities_total": n_ents,
        "unique_entities": len(uniq_ents),
        "entity_types": dict(ent_types.most_common()),
        "tags_total": sum(tag_counts.values()),
        "unique_tags": len(tag_counts),
        "top_tags": tag_counts.most_common(25),
        "relations_total": n_rels,
        "unique_relation_labels": len(rel_labels),
        "top_relation_labels": rel_labels.most_common(25),
    }


def write_dump(records: list[dict], summary: dict, out_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_path + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
