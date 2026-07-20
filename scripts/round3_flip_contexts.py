"""Round-3 helper: dump full context blobs for the three documented flip questions
(06f04340, 1c549ce4, 2ce6a0f2) at each alpha, plus targeted evidence-substring checks
(the answer-bearing strings the paid run's answers actually missed). Offline, $0.

Run:  .venv/bin/python scripts/round3_flip_contexts.py [--out runs/offline_eval_round3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_eval_round3 import (FLIP_QIDS, INGEST_OVERRIDES, K,  # noqa: E402
                                 QUERY_OVERRIDES, RUN_JSON, locate_stores, sess,
                                 gold_sessions)

import kg.graph as kg_graph                                       # noqa: E402
from kg import Config, KnowledgeGraph                             # noqa: E402
from kg.extractors import ScriptedExtractor                       # noqa: E402
from kg.ingest_cache import _sqlite_copy                          # noqa: E402
from kg.rag import ContextBuilder                                 # noqa: E402
from kg.retrieval import HybridRetriever                          # noqa: E402

for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)
kg_graph.get_extractor = lambda config: ScriptedExtractor({})

# evidence the paid run's wrong answers were missing (from run.json judge reasons)
EVIDENCE = {
    "06f04340": ["cherry tomato", "basil", "mint"],
    "1c549ce4": ["car cover", "$120", "$20"],
    "2ce6a0f2": ["art"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1.0,0.9,0.8,0.7,0.5")
    ap.add_argument("--out", default="runs/offline_eval_round3")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    run = json.load(open(RUN_JSON))
    questions = {q["id"]: q for q in run["query"]["queries"] if q["id"] in FLIP_QIDS}
    stores = locate_stores(list(questions.values()))
    import tempfile
    from dataclasses import replace
    work = tempfile.mkdtemp(prefix="round3-flips-")

    base_cfg = Config.default()
    for k, v in {**INGEST_OVERRIDES, **QUERY_OVERRIDES}.items():
        setattr(base_cfg, k, v)

    for qid, q in questions.items():
        wp = os.path.join(work, f"{qid}.db")
        _sqlite_copy(stores[qid], wp)
        g = KnowledgeGraph.open(wp, base_cfg)
        golds = gold_sessions(q)
        print(f"\n== {qid}: {q['query']!r}  gold={golds}")
        for alpha in alphas:
            cfg = replace(base_cfg, seed_fusion_alpha=alpha)
            retriever = HybridRetriever(g.store, g.embedder, g.canon, cfg)
            builder = ContextBuilder(g.store, cfg)
            res = retriever.retrieve(q["query"], k=K, as_of=q.get("question_date"))
            ep_ids, _facts, blob = builder.build(res)
            snap = os.path.join(args.out, "contexts", qid)
            os.makedirs(snap, exist_ok=True)
            with open(os.path.join(snap, f"alpha_{alpha}.txt"), "w") as f:
                f.write(blob)
            low = blob.lower()
            ev = {s: (s.lower() in low) for s in EVIDENCE[qid]}
            ctx_sess = {sess(e) for e in ep_ids}
            print(f"  a={alpha}: gold_in_ctx={[g_ in ctx_sess for g_ in golds]} "
                  f"evidence={ev} ranked_head={[e for e in res.object_ids[:5]]}")
        del g


if __name__ == "__main__":
    main()
