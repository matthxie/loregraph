"""Build the committed `micro` LongMemEval tier — a tiny LIVE smoke set.

`micro` is a deterministic 3-instance subset of the committed `sample` tier, chosen to
exercise the graph's differentiators in a run that finishes in well under a minute on live
Haiku + bge:

  * 08f4fc43  temporal-reasoning  — bi-temporal fact windows / as-of
  * 0100672e  multi-session       — cross-session multi-hop retrieval
  * 01493427  knowledge-update    — the supersede path (a fact changes over time)

Each LongMemEval instance is a multi-session haystack, so "micro" is 3 instances ≈ 18
sessions. It is the default tier for `python -m kg testrun` (live-only). Episode bodies are
committed (like `sample`) so it needs no download. Rebuild with:

    python scripts/build_micro.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "dataset", "longmemeval", "sample")
DST = os.path.join(ROOT, "dataset", "longmemeval", "micro")

# Instances chosen to cover temporal / multi-hop / update in the fewest sessions.
KEEP = ["08f4fc43", "0100672e", "01493427"]


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    keep = set(KEEP)

    episodes = [e for e in _read_jsonl(os.path.join(SRC, "episodes.jsonl"))
                if e["question_id"] in keep]
    questions = [q for q in _read_jsonl(os.path.join(SRC, "questions.jsonl"))
                 if q["id"] in keep]
    # preserve the requested instance order, then chronological within an instance
    order = {qid: i for i, qid in enumerate(KEEP)}
    episodes.sort(key=lambda e: (order[e["question_id"]], e.get("created_at", "")))
    questions.sort(key=lambda q: order[q["id"]])

    _write_jsonl(os.path.join(DST, "episodes.jsonl"), episodes)
    _write_jsonl(os.path.join(DST, "questions.jsonl"), questions)

    kinds: dict[str, int] = {}
    for q in questions:
        kinds[q.get("kind", "")] = kinds.get(q.get("kind", ""), 0) + 1
    manifest = {
        "tier": "micro",
        "n_instances": len(questions),
        "n_episodes": len(episodes),
        "question_types": kinds,
        "instance_ids": KEEP,
        "selection": "committed 3-instance LIVE smoke subset of `sample` "
                     "(temporal / multi-session / knowledge-update)",
        "ordering": "by instance, then chronological by created_at",
        "source": "longmemeval sample tier",
        "note": "committed (episode bodies included); default tier for `kg testrun` (live).",
    }
    with open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {DST}: {len(episodes)} episodes across {len(questions)} instances "
          f"({kinds})")


if __name__ == "__main__":
    main()
