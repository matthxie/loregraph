#!/usr/bin/env python3
"""Build a third corpus folder that *combines* the text + image datasets, with the
Wikipedia articles **exploded into per-paragraph chunks stamped over a timeline**.

Leaves dataset/wikipedia/ and dataset/images/ untouched. Reads both and writes
dataset/mixed/, where every item's filename is a hash so a plain directory listing
interleaves articles and photos in a basically-random order instead of grouping
them by type.

Unlike a 1:1 article→file copy, each Wikipedia article is split into its
paragraphs and **every paragraph becomes its own independent entry** with a fresh
hashed name and its own `created_at` timestamp. The timestamps fan an article's
paragraphs out across a window (paragraph 0 oldest, later ones stamped later — so
within an article `created_at` follows document order), and different articles
start at different points, so a listing sorted by `created_at` reads as a stream
of paragraph-entries arriving over time instead of a frozen snapshot.

Scope: this is a *chronological append stream* — it spreads `created_at` over a
timeline. It is NOT an update/contradiction stream: every entry is a distinct,
never-restated fact (`orig_id#pNNN`), so nothing supersedes anything. And note the
kg pipeline does not currently ingest dataset/mixed/ at all (kg/corpus.py reads
wikipedia/articles.jsonl + images/manifest.jsonl, and kg/ingest.py stamps nodes
with now_iso() at ingest) — this folder is a standalone artifact. The `created_at`
format deliberately matches kg.store.now_iso() so a future load_mixed() *could*
thread these times into the graph's created_at/valid machinery, but no code does so
today.

  dataset/mixed/<hash>.txt        ONE paragraph of a Wikipedia article (raw text,
                                  its section heading kept as the first line)
  dataset/mixed/<hash>.jpg        one photo (copied bytes)
  dataset/mixed/manifest.jsonl    one record per item, sorted by created_at (a
                                  chronological stream):
                                  {id, file, modality, orig_id, title, url, label,
                                   para_index, para_count, created_at}

Deterministic: paragraph hashes are sha1(orig_id#pNNN) and image hashes are
sha1(orig_id), and all timestamps come from a seeded RNG, so re-running reproduces
identical names and times. para_index / para_count let you regroup an article's
paragraphs and recover their order (within an article created_at is monotonic in
para_index). The manifest records the provenance a future loader would need (titles
for text nodes, COCO labels as the offline image-description stand-in).

Usage:  python scripts/build_mixed.py [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from datetime import datetime, timedelta, timezone

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dataset")
WIKI = os.path.join(DATASET_DIR, "wikipedia", "articles.jsonl")
IMG_MANIFEST = os.path.join(DATASET_DIR, "images", "manifest.jsonl")
IMG_DIR = os.path.join(DATASET_DIR, "images")
OUT_DIR = os.path.join(DATASET_DIR, "mixed")

# Timeline the synthetic stream is spread across. Anchored at the Wikipedia dump
# date (20231101) and running up to ~the present, so the corpus reads as facts
# accumulating from the snapshot forward.
WINDOW_START = datetime(2023, 11, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 6, 1, tzinfo=timezone.utc)
_SPAN_S = (WINDOW_END - WINDOW_START).total_seconds()


def _hash(key: str, taken: set[str]) -> str:
    """sha1(key) → 16 hex chars; extend on the (astronomically unlikely) collision."""
    full = hashlib.sha1(key.encode("utf-8")).hexdigest()
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


# Wikipedia end-matter: everything from the first of these standalone headings to the
# end of the article is page apparatus, not prose paragraphs — reference lists,
# navigation, and the trailing category dump ("1980 births / Living people / ..."). We
# cut it so an article's entries are its actual paragraphs, not boilerplate stamped as
# its own "fact". (Anything genuinely encyclopedic lives above these in the body.)
_FOOTER_HEADINGS = {
    "references", "external links", "see also", "further reading", "notes",
    "bibliography", "citations", "sources", "footnotes", "works cited",
    "notes and references", "explanatory notes",
}


def _is_heading(block: str) -> bool:
    """A bare section heading: a single short line that isn't a sentence and isn't a
    list-intro (e.g. 'History', 'Early years'). These get folded onto the paragraph
    that follows so a chunk keeps its section context instead of stranding the heading.
    A trailing ':' or ',' means a list-intro ('Directors:'), handled by forward-merge."""
    if "\n" in block:
        return False
    s = block.strip()
    return bool(s) and len(s) <= 70 and s[-1] not in ".?!:,;"


def _strip_footer(blocks: list[str]) -> list[str]:
    """Drop everything from the first Wikipedia footer heading onward. Matches both a
    standalone heading block ('References') and a block whose first line is the heading
    with its list glued on ('See also\\n Foo\\n Bar') — these words are always end-matter."""
    for i, b in enumerate(blocks):
        first = b.split("\n", 1)[0].strip().lower().rstrip(":")
        if first in _FOOTER_HEADINGS:
            return blocks[:i]
    return blocks


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def split_paragraphs(text: str) -> list[str]:
    """Split an article into self-contained paragraph chunks (the article's "entries").

    Pipeline: split on blank lines → drop the reference/category footer → fold a
    list-intro line ('… the following:') forward onto its list items → fold a bare
    section heading backward onto the paragraph it titles → drop degenerate stubs (a
    heading with no real body, e.g. 'Honours\\n.'). Each surviving chunk is one
    paragraph (optionally led by its heading), self-contained.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # any non-newline whitespace can fill a "blank" separator line (NBSP, form-feed…)
    blocks = [b.strip() for b in re.split(r"\n[^\S\n]*\n", text) if b.strip()]
    blocks = _strip_footer(blocks)

    # forward-merge: a block ending in ':' is a list-intro → pull in the items that
    # were split off into the following block(s), so the intro and its list stay one chunk
    merged: list[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        while b.rstrip().endswith(":") and i + 1 < len(blocks):
            i += 1
            b = f"{b}\n{blocks[i]}"
        merged.append(b)
        i += 1

    # backward-merge bare headings onto their paragraph; drop empty/degenerate chunks
    chunks: list[str] = []
    pending: str | None = None  # accumulated heading(s) awaiting their paragraph
    for b in merged:
        if _is_heading(b):
            pending = f"{pending}\n{b}" if pending else b
            continue
        chunk = f"{pending}\n{b}" if pending else b
        pending = None
        # a chunk needs a real body: more than its heading line(s) plus a token of prose
        body = chunk.split("\n", 1)[1] if "\n" in chunk and _is_heading(chunk.split("\n")[0]) else chunk
        if _word_count(body) >= 3:
            chunks.append(chunk)
    # a dangling trailing heading (no body followed) is simply dropped
    return chunks


def _article_times(rng: random.Random, n: int) -> list[datetime]:
    """`n` strictly-increasing arrival times for one article's paragraphs, all
    inside [WINDOW_START, WINDOW_END]. Each article gets its own editing window of
    duration D placed somewhere in the timeline: usually a short same-session burst
    (minutes–days), occasionally a long-running edit history (weeks–months). The
    paragraphs are sorted random points within that window, so paragraph 0 is the
    oldest and later paragraphs are appended over time."""
    if rng.random() < 0.7:
        D = rng.uniform(5 * 60, 3 * 24 * 3600)          # 5 min – 3 d (burst)
    else:
        D = rng.uniform(3 * 24 * 3600, 120 * 24 * 3600)  # 3 – 120 d (edited over time)
    D = min(D, _SPAN_S - n)                               # leave room inside the window
    base = rng.uniform(0, max(0.0, _SPAN_S - D - n))
    offsets = sorted(rng.uniform(0, D) for _ in range(n))
    # Output precision is whole seconds (see _iso), so enforce strict monotonicity
    # on *integer* seconds — otherwise two sub-second-apart paragraphs format equal.
    times: list[datetime] = []
    prev = -1
    for off in offsets:
        s = int(base + off)
        if s <= prev:
            s = prev + 1
        prev = s
        times.append(WINDOW_START + timedelta(seconds=s))
    return times


def _iso(dt: datetime) -> str:
    """Match kg.store.now_iso(): UTC, seconds precision, '+00:00' offset."""
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for timestamps")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # Validate sources BEFORE the destructive rebuild, so a missing input never wipes
    # the previous good dataset/mixed/ and leaves a half-written folder behind.
    for src in (WIKI, IMG_MANIFEST):
        if not os.path.exists(src):
            raise SystemExit(f"missing source {src} — run scripts/build_dataset.py first")

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)        # rebuild cleanly
    os.makedirs(OUT_DIR, exist_ok=True)

    taken: set[str] = set()
    records = []
    n_articles = 0
    n_dropped = 0

    # text → one <hash>.txt per paragraph, each its own timestamped entry
    for r in _read_jsonl(WIKI):
        orig_id = r["id"]
        paras = split_paragraphs(r.get("text", ""))
        if not paras:
            n_dropped += 1
            continue
        n_articles += 1
        n_para = len(paras)
        times = _article_times(rng, n_para)
        for idx, (para, t) in enumerate(zip(paras, times)):
            h = _hash(f"{orig_id}#p{idx:03d}", taken)
            fname = f"{h}.txt"
            with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as out:
                out.write(para)
            records.append({
                "id": h, "file": fname, "modality": "text", "orig_id": orig_id,
                "title": r.get("title"), "url": r.get("url"), "label": None,
                "para_index": idx, "para_count": n_para, "created_at": _iso(t),
            })

    # images → <hash>.jpg (copied bytes), one timestamped entry each
    for r in _read_jsonl(IMG_MANIFEST):
        h = _hash(r["id"], taken)
        fname = f"{h}.jpg"
        src = os.path.join(IMG_DIR, os.path.basename(r["file"]))
        shutil.copyfile(src, os.path.join(OUT_DIR, fname))
        t = WINDOW_START + timedelta(seconds=rng.uniform(0, _SPAN_S))
        records.append({
            "id": h, "file": fname, "modality": "image", "orig_id": r["id"],
            "title": None, "url": None, "label": r.get("label"),
            "para_index": None, "para_count": None, "created_at": _iso(t),
        })

    # sort chronologically so the manifest reads as a stream of arriving updates
    records.sort(key=lambda x: (x["created_at"], x["id"]))
    with open(os.path.join(OUT_DIR, "manifest.jsonl"), "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_text = sum(1 for r in records if r["modality"] == "text")
    n_img = sum(1 for r in records if r["modality"] == "image")
    print(f"wrote {len(records)} entries "
          f"({n_text} paragraph chunks from {n_articles} articles + {n_img} images) "
          f"-> {OUT_DIR}")
    if n_dropped:
        print(f"  ({n_dropped} articles had no usable paragraphs, skipped)")
    print(f"  timeline: {records[0]['created_at']} … {records[-1]['created_at']}")
    print(f"  first 6 by arrival: {[r['file'] for r in records[:6]]}")


if __name__ == "__main__":
    main()
