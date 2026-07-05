"""Ingest selected LongMemEval `small` instances into per-instance stores for the
extraction-completeness spike. Mirrors kg/testrun.py::run_per_instance's ingest loop.

Usage: .venv\\Scripts\\python.exe spikes\\completeness\\build_stores.py
"""
import os
import sys
import time

os.environ.pop("OPENAI_API_KEY", None)  # drop any stale key; kg loads the real one from .env on import

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kg.config import Config
from kg.corpus import iter_lme_instances
from kg.graph import KnowledgeGraph

TARGET_IDS = {
    "00ca467f",   # doctor's appointments in March (gold 2, reader)
    "2788b940",   # fitness classes in a typical week (gold 5, reader)
    "2e6d26dc",   # babies born to friends/family (gold 5, reader)
    "2b8f3739",   # total earned selling products at markets (gold $495, reader, SUM)
    "36b9f61e",   # total spent on luxury items (gold $2500, reader, SUM)
    "129d1232",   # total raised via charity events (gold $5850, join_miss, SUM)
    "21d02d0d",   # fun runs missed in March (gold 2, ok/passing)
    "0a995998",   # clothing items to pick up/return (gold 3, reader)
}

STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stores")
os.makedirs(STORE_DIR, exist_ok=True)


def main() -> None:
    cfg = Config.default()
    instances = list(iter_lme_instances("small", limit=None))
    todo = [(q, sessions) for q, sessions in instances if q["id"] in TARGET_IDS]
    found_ids = {q["id"] for q, _ in todo}
    missing = TARGET_IDS - found_ids
    if missing:
        print(f"WARNING: ids not found in dataset: {missing}")

    for i, (q, sessions) in enumerate(todo):
        qid = q["id"]
        store_path = os.path.join(STORE_DIR, f"{qid}.db")
        if os.path.exists(store_path):
            print(f"[{i+1}/{len(todo)}] {qid}: store exists, skipping")
            continue
        print(f"[{i+1}/{len(todo)}] {qid}: ingesting {len(sessions)} sessions ...")
        t0 = time.time()
        g = KnowledgeGraph.open(store_path, cfg)
        rep = g.ingest(sessions)
        g.save()
        stats = g.store.stats()
        print(f"  done in {time.time()-t0:.1f}s -> nodes={stats['nodes']} edges={stats['edges']}")
        time.sleep(2)  # be polite to the RPD budget between instances

    print("\nAll target stores built (or already present) in", STORE_DIR)


if __name__ == "__main__":
    main()
