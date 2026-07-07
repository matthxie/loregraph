"""Pilot re-ingest for the extraction-side event/category changes.

Re-ingests ONLY the given question ids (default: every question judged incorrect in
--run) under the CURRENT extractor prompt/schema/cues, writing fresh entries into
store/cache/ under the new cache key. The old cache entries are left untouched; the
containment eval and testrun both pick the newest db per qid by mtime/key, so after
this runs the $0 containment eval scores the new stores directly:

  .venv/bin/python spikes/events_pilot/reingest.py                     # wrong qids
  .venv/bin/python spikes/events_pilot/reingest.py --only 1a8a66a6 ...  # explicit
  env -u OPENAI_API_KEY .venv/bin/python spikes/containment/eval.py \
      --label events-pilot --set mmr_lambda=1.0 --set rag_retarget=ce \
      --only <qids...>

Needs OPENAI_API_KEY (cue-escalation calls are the whole point of the pilot).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kg.config import Config
from kg.corpus import iter_lme_instances
from kg.graph import KnowledgeGraph
from kg.ingest_cache import cache_path, ingest_cache_key, save
from kg.metering import totals_of

DEFAULT_RUN = "runs/reader5-ce-prompts-1/run.json"
STORE_PATH = os.path.join("store", "events_pilot.db")


def _wrong_qids(run_path: str) -> list[str]:
    with open(run_path, encoding="utf-8") as f:
        run = json.load(f)
    return [q["id"] for q in run["query"]["queries"]
            if (q.get("judge") or {}).get("correct") is False]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--tier", default="small")
    ap.add_argument("--only", action="append", default=[],
                    help="explicit qid (repeatable); default = all wrong in --run")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY required — the pilot exists to run escalations")

    # MUST mirror the benchmark's ingest-relevant config (testrun ran with defaults
    # plus --chunking turns; every other override in the run command was query-side).
    cfg = Config.default()
    cfg.chunking = "turns"

    qids = args.only or _wrong_qids(args.run)
    print(f"re-ingesting {len(qids)} instance(s): {' '.join(qids)}")

    total_cost = 0.0
    t0 = time.time()
    done = 0
    for q, sessions in iter_lme_instances(args.tier):
        if q["id"] not in qids:
            continue
        key = ingest_cache_key(q["id"], sessions, cfg)
        dst = cache_path(STORE_PATH, q["id"], key)
        if os.path.exists(dst):
            print(f"  {q['id']}: already cached under new key — skip")
            done += 1
            continue
        if os.path.exists(STORE_PATH):
            os.remove(STORE_PATH)
        g = KnowledgeGraph.open(STORE_PATH, cfg)
        g.extractor.meter.drain()
        g.canon.meter.drain()
        rep = g.ingest(sessions)
        tok = totals_of(g.extractor.meter.drain() + g.canon.meter.drain())
        save(STORE_PATH, q["id"], key)
        total_cost += tok["cost_usd"]
        done += 1
        esc = (g.extractor.escalation_summary()
               if hasattr(g.extractor, "escalation_summary") else {})
        print(f"  {q['id']}: {rep.ingested} eps, {rep.facts} facts, "
              f"${tok['cost_usd']:.4f}, escalation {esc.get('escalation_rate', '?')} "
              f"({done}/{len(qids)})")

    print(f"\ndone: {done} instances, ${total_cost:.3f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
