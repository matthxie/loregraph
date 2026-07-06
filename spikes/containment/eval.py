"""Offline containment eval — replays retrieval + context-building against the cached
per-instance stores (store/cache/<qid>-*.db) with NO LLM calls, so query-side config
changes (rag_retarget, rag_parent_expand, ...) can be scored for free before paying for
a full reader run.

Reuses the open-cached-store-and-run-HybridRetriever/ContextBuilder path shown in
spikes/retarget/probe.py, but over all 100 questions from an existing run file instead of
three hand-picked cases, and prints per-kind summary stats instead of before/after diffs.

Scores, per question:
  containment   — fraction of answer_expected's content tokens present in the context blob
  gold_coverage — fraction of gold sessions with >=1 chunk in the context
  context_chars / context_chunks — how much context this config produced

Run (offline, no key needed):
  OPENAI_API_KEY= .venv/bin/python spikes/containment/eval.py --label baseline
  .venv/bin/python spikes/containment/eval.py --label retarget-off --set rag_retarget=off
  .venv/bin/python spikes/containment/eval.py --only 25e5aa4f --only 37f165cf
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kg.config import Config
from kg.graph import KnowledgeGraph
from kg.rag import ContextBuilder
from kg.retrieval import HybridRetriever

RUN_PATH = "runs/reader5-queryside-both-retarget-1/run.json"
CACHE_DIR = "store/cache"
OUT_DIR = "spikes/containment/out"
_WORD = re.compile(r"[a-z0-9]+")


def _article(oid: str) -> str:
    """Collapse a chunk id to its session id: ep_<qid>__<session>#c000 -> <qid>__<session>,
    obj_<qid>__<session>_1 -> <qid>__<session>_1. Mirrors kg/testrun.py's _article so gold
    ids (obj_-prefixed) and context chunk ids (ep_-prefixed) compare on the same key."""
    base = oid.split("#", 1)[0]
    for prefix in ("ep_", "obj_"):
        if base.startswith(prefix):
            return base[len(prefix):]
    return base


def _load_questions(run_path: str) -> list[dict]:
    with open(run_path, encoding="utf-8") as f:
        run = json.load(f)
    return run["query"]["queries"]


def _cache_path(qid: str) -> str | None:
    """A qid can have several cache entries from earlier ingest-side experiments (the
    cache key hashes ingest-relevant config, not query-side); take the most recently
    written one, i.e. the latest ingest."""
    matches = sorted(glob.glob(os.path.join(CACHE_DIR, f"{qid}-*.db")), key=os.path.getmtime)
    return matches[-1] if matches else None


def _open_store(cache_path: str, cfg: Config) -> KnowledgeGraph:
    tmp = tempfile.mktemp(suffix=".db")
    shutil.copy(cache_path, tmp)
    return KnowledgeGraph.open(tmp, cfg)


def _apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    cfg = replace(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        key = "top_k" if key == "k" else key
        if not hasattr(cfg, key):
            raise SystemExit(f"--set: unknown config field {key!r}")
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            val = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        else:
            val = raw
        setattr(cfg, key, val)
    return cfg


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 1}


def _score(q: dict, cfg: Config) -> dict | None:
    cache = _cache_path(q["id"])
    if cache is None:
        return {"id": q["id"], "kind": q.get("kind", ""), "error": "no cached store"}
    g = _open_store(cache, cfg)
    retriever = HybridRetriever(g.store, g.embedder, g.canon, cfg)
    builder = ContextBuilder(g.store, cfg)
    result = retriever.retrieve(q["query"], k=cfg.top_k, as_of=q.get("as_of"),
                                kind=q.get("kind"))
    ctx_ids, _facts, blob = builder.build(result)

    e_toks = _content_tokens(q.get("answer_expected", ""))
    blob_toks = _content_tokens(blob)
    containment = (round(sum(1 for t in e_toks if t in blob_toks) / len(e_toks), 3)
                   if e_toks else None)

    gold_sessions = {_article(gid) for gid in q.get("gold", [])}
    ctx_sessions = {_article(c) for c in ctx_ids}
    covered = sum(1 for s in gold_sessions if s in ctx_sessions)
    gold_coverage = round(covered / len(gold_sessions), 3) if gold_sessions else None

    return {"id": q["id"], "kind": q.get("kind", ""), "containment": containment,
            "gold_coverage": gold_coverage, "n_gold": len(gold_sessions),
            "n_gold_covered": covered, "context_chars": len(blob),
            "context_chunks": len(ctx_ids)}


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _print_summary(records: list[dict]) -> None:
    by_kind: dict[str, list[dict]] = {}
    for r in records:
        by_kind.setdefault(r["kind"], []).append(r)
    hdr = f"{'kind':<28}{'n':>4}{'containment':>13}{'gold_cov':>10}{'chars':>12}{'chunks':>8}"
    print(hdr)
    print("-" * len(hdr))
    for kind in sorted(by_kind):
        rs = by_kind[kind]
        print(f"{kind:<28}{len(rs):>4}{_mean([r.get('containment') for r in rs]):>13}"
              f"{_mean([r.get('gold_coverage') for r in rs]):>10}"
              f"{_mean([r.get('context_chars') for r in rs]):>12}"
              f"{_mean([r.get('context_chunks') for r in rs]):>8}")
    print("-" * len(hdr))
    print(f"{'OVERALL':<28}{len(records):>4}"
          f"{_mean([r.get('containment') for r in records]):>13}"
          f"{_mean([r.get('gold_coverage') for r in records]):>10}"
          f"{_mean([r.get('context_chars') for r in records]):>12}"
          f"{_mean([r.get('context_chunks') for r in records]):>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN_PATH, help="run.json to source questions/gold from")
    ap.add_argument("--label", default="containment", help="output file stem")
    ap.add_argument("--only", action="append", default=[], help="restrict to this qid (repeatable)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="key=value", help="Config field override (repeatable)")
    args = ap.parse_args()

    cfg = Config.default()
    cfg.top_k = 8
    cfg.rag_parent_expand = 2
    cfg.rag_chunks_per_source = 2
    cfg = _apply_overrides(cfg, args.overrides)

    questions = _load_questions(args.run)
    if args.only:
        wanted = set(args.only)
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            print(f"warning: qids not found in {args.run}: {sorted(missing)}")

    records = [_score(q, cfg) for q in questions]
    for r in records:
        print(json.dumps(r))
    _print_summary([r for r in records if "error" not in r])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "overrides": args.overrides, "records": records}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
