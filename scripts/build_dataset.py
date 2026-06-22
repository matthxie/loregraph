#!/usr/bin/env python3
"""Build the test corpus into ./dataset.

Two independent pulls (text and images are NOT paired):
  - 100 random *full* Wikipedia articles  -> dataset/wikipedia/articles.jsonl
  - 100 random real photos                -> dataset/images/*.jpg (+ manifest.jsonl)

Everything is streamed (HF `datasets`, streaming=True) so we never download the
full multi-GB datasets; we pull lazily and stop at 100. Shuffles use a fixed seed
so the corpus is reproducible.

Usage:  python scripts/build_dataset.py [--n 100] [--seed 42]
"""
from __future__ import annotations
import argparse, json, os, sys

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"
WIKI_MIN_CHARS = 1500          # skip stubs / disambiguation / redirects-ish
WIKI_SHUFFLE_BUFFER = 10000

# Tried in order; first one that loads + yields PIL images wins.
# COCO = varied everyday scenes (~80 categories); the rest are varied fallbacks.
IMAGE_CANDIDATES = [
    {"path": "detection-datasets/coco", "name": None, "split": "val"},
    {"path": "Maysee/tiny-imagenet", "name": None, "split": "valid"},
    {"path": "uoft-cs/cifar100", "name": None, "split": "train"},
]
IMAGE_KEYS = ["image", "img", "jpg", "png"]
IMAGE_LABEL_KEYS = ["label", "fine_label", "coarse_label", "classes"]
IMAGE_SHUFFLE_BUFFER = 600


def _find_pil(example):
    """Return (key, PIL.Image) for the first image-like field, else (None, None)."""
    from PIL import Image
    for k in IMAGE_KEYS:
        v = example.get(k)
        if isinstance(v, Image.Image):
            return k, v
    # fall back: any value that looks like a PIL image
    for k, v in example.items():
        if isinstance(v, Image.Image):
            return k, v
    return None, None


def _coco_subjects(example, features):
    """For COCO: turn the per-image object categories into a readable subject list."""
    objs = example.get("objects")
    if not isinstance(objs, dict) or "category" not in objs:
        return None
    cats = objs.get("category") or []
    names = None
    try:  # objects is a dict; objects["category"] is a List(ClassLabel(...))
        cl = features["objects"]["category"].feature
        names = [cl.int2str(int(c)) for c in cats]
    except Exception:
        names = [str(c) for c in cats]
    # dedupe, preserve order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return ", ".join(out) if out else None


def _label_of(example, features=None):
    coco = _coco_subjects(example, features)
    if coco:
        return coco
    for k in IMAGE_LABEL_KEYS:
        if k in example:
            v = example[k]
            if features is not None:
                try:  # int -> class name when schema has a ClassLabel
                    return features[k].int2str(int(v))
                except Exception:
                    pass
            return v if isinstance(v, str) else str(v)
    return None


def build_wikipedia(n: int, seed: int) -> int:
    from datasets import load_dataset
    out_dir = os.path.join(DATASET_DIR, "wikipedia")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "articles.jsonl")

    print(f"[wikipedia] streaming {WIKI_DATASET}:{WIKI_CONFIG} (shuffle seed={seed}) ...")
    ds = load_dataset(WIKI_DATASET, WIKI_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=WIKI_SHUFFLE_BUFFER)

    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in ds:
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            if len(text) < WIKI_MIN_CHARS:
                continue
            if title.lower().endswith("(disambiguation)"):
                continue
            rec = {
                "id": f"wiki_{written:03d}",
                "title": title,
                "url": row.get("url"),
                "text": text,
                "char_len": len(text),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % 20 == 0:
                print(f"[wikipedia] {written}/{n}")
            if written >= n:
                break
    print(f"[wikipedia] wrote {written} articles -> {out_path}")
    return written


def build_images(n: int, seed: int) -> int:
    from datasets import load_dataset
    out_dir = os.path.join(DATASET_DIR, "images")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    last_err = None
    for cand in IMAGE_CANDIDATES:
        try:
            label = f"{cand['path']}" + (f":{cand['name']}" if cand["name"] else "")
            print(f"[images] trying {label} (shuffle seed={seed}) ...")
            ds = load_dataset(cand["path"], cand["name"], split=cand["split"], streaming=True)
            features = getattr(ds, "features", None)
            ds = ds.shuffle(seed=seed, buffer_size=IMAGE_SHUFFLE_BUFFER)

            written = 0
            with open(manifest_path, "w", encoding="utf-8") as man:
                for example in ds:
                    key, img = _find_pil(example)
                    if img is None:
                        continue
                    fname = f"img_{written:03d}.jpg"
                    fpath = os.path.join(out_dir, fname)
                    img.convert("RGB").save(fpath, "JPEG", quality=90)
                    man.write(json.dumps({
                        "id": f"img_{written:03d}",
                        "file": f"images/{fname}",
                        "source": label,
                        "label": _label_of(example, features),
                    }, ensure_ascii=False) + "\n")
                    written += 1
                    if written % 20 == 0:
                        print(f"[images] {written}/{n}")
                    if written >= n:
                        break
            if written >= n:
                print(f"[images] wrote {written} images from {label} -> {out_dir}")
                return written
            print(f"[images] {label} only yielded {written}; trying next candidate")
        except Exception as e:  # noqa: BLE001 - want to fall through to next source
            last_err = e
            print(f"[images] {cand['path']} failed: {e!r}; trying next candidate")
    print(f"[images] ERROR: no image source worked. last error: {last_err!r}", file=sys.stderr)
    return 0


def write_readme(n_text: int, n_img: int, img_source: str | None):
    path = os.path.join(DATASET_DIR, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# dataset/\n\n"
            "Test corpus for the knowledge-graph prototype. Text and images are **independent** "
            "(not paired). Built by `scripts/build_dataset.py` via streaming (reproducible, fixed seed).\n\n"
            "## Contents\n\n"
            f"- `wikipedia/articles.jsonl` — {n_text} random full Wikipedia articles "
            "(`{id, title, url, text, char_len}` per line). Source: `wikimedia/wikipedia` `20231101.en`, "
            "license **CC BY-SA 4.0** (attribution + share-alike).\n"
            f"- `images/*.jpg` + `images/manifest.jsonl` — {n_img} random real photos "
            f"(`{{id, file, source, label}}` per line). Source: `{img_source or 'n/a'}`.\n\n"
            "Regenerate: `python scripts/build_dataset.py`\n"
        )
    print(f"[readme] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    os.makedirs(DATASET_DIR, exist_ok=True)
    n_text = 0 if args.skip_text else build_wikipedia(args.n, args.seed)
    n_img = 0 if args.skip_images else build_images(args.n, args.seed)

    img_source = None
    man = os.path.join(DATASET_DIR, "images", "manifest.jsonl")
    if os.path.exists(man):
        with open(man) as f:
            first = f.readline()
            if first:
                img_source = json.loads(first).get("source")
    write_readme(n_text, n_img, img_source)
    print(f"\nDone: {n_text} articles, {n_img} images -> {DATASET_DIR}")


if __name__ == "__main__":
    main()
