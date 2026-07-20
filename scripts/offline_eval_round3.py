"""Offline Round-3 eval: seed-score fusion alpha sweep (docs/OFFLINE_EVAL.md Round 3).

NO paid LLM calls. Drives the exact read path (HybridRetriever.retrieve →
ContextBuilder.build — what KnowledgeGraph.search() runs) against the CACHED
per-instance benchmark stores from runs/sample-datefix-events-1 (store/cache/*.db,
extraction cost already sunk). Each cached store is COPIED to a temp path first;
the cache is never opened read-write.

For each alpha in --alphas (default 1.0,0.9,0.8,0.7,0.5) and each of the run's ~100
questions, records:
  * gold session rank in the final ranked list and in the fused PPR pool
  * gold-in-context at session level (any gold-session chunk in the context) and at
    answer-chunk level (a gold chunk whose raw text contains answer_expected)
  * answer_expected substring present in the context blob; context size
  * Kendall tau vs the alpha=1.0 pool ordering (blast radius)
alpha=1.0 rows double as a harness-fidelity check against the run.json gold_marks.

Run:  .venv/bin/python scripts/offline_eval_round3.py [--out runs/offline_eval_round3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kg.graph as kg_graph                                    # noqa: E402
from kg import Config, KnowledgeGraph                          # noqa: E402
from kg.extractors import ScriptedExtractor                    # noqa: E402
from kg.ingest_cache import cache_path, ingest_cache_key       # noqa: E402
from kg.ingest_cache import _sqlite_copy                       # noqa: E402
from kg.models import NodeType                                 # noqa: E402
from kg.rag import ContextBuilder                              # noqa: E402
from kg.retrieval import HybridRetriever                       # noqa: E402

# ---- hard no-LLM guard (same as offline_eval.py): kg auto-loads .env on import.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)
kg_graph.get_extractor = lambda config: ScriptedExtractor({})

RUN_JSON = "runs/sample-datefix-events-1/run.json"
FLIP_QIDS = ("06f04340", "1c549ce4", "2ce6a0f2")

# The run's config: ingest-side (must match the cache key exactly — verified against
# store/cache before this script existed) + query-side (the testrun --set values).
INGEST_OVERRIDES = {"extractor_backend": "cue_gated", "event_facts": True,
                    "ingest_date_filter": True}
QUERY_OVERRIDES = {"rag_retarget": "ce", "rag_provenance_promote": True,
                   "mmr_lambda": 1.0, "rag_parent_expand": 2,
                   "rag_chunks_per_source": 2, "history_all_lanes": True}
K = 8   # run.json config.k


def sess(eid: str) -> str:
    return eid.split("#", 1)[0]


def gold_sessions(q: dict) -> list[str]:
    # run.json gold ids are "obj_<...>"; store episode/session ids are "ep_<...>"
    return ["ep_" + g[4:] if g.startswith("obj_") else g for g in q["gold"]]


def kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    """Kendall tau over the ids common to both orderings (1.0 = identical order)."""
    common = [x for x in order_a if x in set(order_b)]
    n = len(common)
    if n < 2:
        return None
    pos_b = {x: i for i, x in enumerate(order_b)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = pos_b[common[i]] - pos_b[common[j]]
            if d < 0:
                conc += 1
            elif d > 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0


def locate_stores(questions: list[dict]) -> dict[str, str]:
    """Map question id -> cached store path via the exact ingest cache key."""
    from kg.corpus import iter_lme_instances
    os.environ.setdefault("KG_LLM", "openai")   # pin: llm_model hashes as gpt-4o-mini,
    #                                             matching the paid run's digest
    cfg = Config.default()
    for k, v in INGEST_OVERRIDES.items():
        setattr(cfg, k, v)
    want = {q["id"] for q in questions}
    out = {}
    for q, sessions in iter_lme_instances("small"):
        if q["id"] not in want:
            continue
        p = cache_path("store/lme_instance.db", q["id"],
                       ingest_cache_key(q["id"], sessions, cfg))
        if os.path.exists(p):
            out[q["id"]] = p
    return out


def eval_question(store, embedder, canon, base_cfg: Config, q: dict,
                  alpha: float) -> dict:
    cfg = replace(base_cfg, seed_fusion_alpha=alpha)
    retriever = HybridRetriever(store, embedder, canon, cfg)
    builder = ContextBuilder(store, cfg)
    as_of = q.get("question_date")
    res = retriever.retrieve(q["query"], k=K, as_of=as_of)
    ep_ids, facts, blob = builder.build(res)

    golds = gold_sessions(q)
    ranked = list(res.object_ids)                       # final top-k
    pool = [eid for eid, _s in getattr(res, "ppr_pool", [])]  # fused-base pool order

    def first_rank(order: list[str], gold: str) -> int | None:
        for i, eid in enumerate(order):
            if sess(eid) == gold:
                return i + 1
        return None

    ctx_sessions = {sess(e) for e in ep_ids}
    ans = (q.get("answer_expected") or "").strip().lower()
    # answer chunks: gold-session chunks whose raw text carries the expected answer
    ans_chunks = []
    if ans:
        for gold in golds:
            base_node = store.get_node(gold)
            cands = [gold] if base_node is not None else []
            for n in store.nodes_of_type(NodeType.EPISODE):
                if n.id.startswith(gold + "#"):
                    cands.append(n.id)
            for cid in cands:
                n = store.get_node(cid)
                if n and ans in (n.raw_text or "").lower():
                    ans_chunks.append(cid)
    return {
        "qid": q["id"], "alpha": alpha, "lane": getattr(res, "lane", ""),
        "gold_rank_final": {g: first_rank(ranked, g) for g in golds},
        "gold_rank_pool": {g: first_rank(pool, g) for g in golds},
        "gold_in_ctx_session": {g: (g in ctx_sessions) for g in golds},
        "all_gold_in_ctx": all(g in ctx_sessions for g in golds),
        "any_gold_in_ctx": any(g in ctx_sessions for g in golds),
        "ans_chunk_in_ctx": (any(c in ep_ids for c in ans_chunks)
                             if ans_chunks else None),
        "ans_substr_in_ctx": (ans in blob.lower()) if ans else None,
        "ctx_chars": len(blob), "n_ctx_eps": len(ep_ids),
        "ranked": ranked, "pool": pool[:64], "ctx_eps": ep_ids,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1.0,0.9,0.8,0.7,0.5")
    ap.add_argument("--out", default="runs/offline_eval_round3")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    os.makedirs(args.out, exist_ok=True)

    run = json.load(open(RUN_JSON))
    questions = run["query"]["queries"][: args.limit]
    stores = locate_stores(questions)
    missing = [q["id"] for q in questions if q["id"] not in stores]
    if missing:
        sys.exit(f"no cache hit for {len(missing)} instances: {missing[:5]} …")
    print(f"{len(questions)} questions, all cached; alphas={alphas}", flush=True)

    base_cfg = Config.default()
    for k, v in {**INGEST_OVERRIDES, **QUERY_OVERRIDES}.items():
        setattr(base_cfg, k, v)

    work = tempfile.mkdtemp(prefix="round3-")
    rows = []
    t0 = time.time()
    for i, q in enumerate(questions):
        wp = os.path.join(work, f"{q['id']}.db")
        _sqlite_copy(stores[q["id"]], wp)          # copy — never mutate the cache
        g = KnowledgeGraph.open(wp, base_cfg)      # one store/embedder/canon per instance
        for alpha in alphas:
            rows.append(eval_question(g.store, g.embedder, g.canon, base_cfg, q, alpha))
        del g
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(wp + suf):
                os.remove(wp + suf)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(questions)}  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)

    # ---------------- summary ----------------
    by = {}
    for r in rows:
        by.setdefault(r["qid"], {})[r["alpha"]] = r
    base_alpha = alphas[0]
    print("\n== alpha sweep (n=%d questions) ==" % len(by))
    hdr = f"{'alpha':>6} {'allgold_ctx':>11} {'anygold_ctx':>11} {'ans_chunk':>9} " \
          f"{'ans_substr':>10} {'ctx_chars':>9} {'tau_vs_1.0':>10}"
    print(hdr)
    for alpha in alphas:
        rs = [by[qid][alpha] for qid in by]
        taus = [kendall_tau(by[qid][base_alpha]["pool"], by[qid][alpha]["pool"])
                for qid in by]
        taus = [t for t in taus if t is not None]
        chunk = [r["ans_chunk_in_ctx"] for r in rs if r["ans_chunk_in_ctx"] is not None]
        sub = [r["ans_substr_in_ctx"] for r in rs if r["ans_substr_in_ctx"] is not None]
        print(f"{alpha:>6} {sum(r['all_gold_in_ctx'] for r in rs):>11} "
              f"{sum(r['any_gold_in_ctx'] for r in rs):>11} "
              f"{sum(chunk):>9} {sum(sub):>10} "
              f"{sum(r['ctx_chars'] for r in rs)//len(rs):>9} "
              f"{(sum(taus)/len(taus) if taus else 1.0):>10.3f}")

    # regressions / fixes vs alpha=1.0 (session-level all-gold-in-context)
    for alpha in alphas[1:]:
        fixed = [q for q in by if by[q][alpha]["all_gold_in_ctx"]
                 and not by[q][base_alpha]["all_gold_in_ctx"]]
        broke = [q for q in by if not by[q][alpha]["all_gold_in_ctx"]
                 and by[q][base_alpha]["all_gold_in_ctx"]]
        print(f"alpha={alpha}: fixed {len(fixed)} {fixed}  broke {len(broke)} {broke}")

    # flip questions detail
    for qid in FLIP_QIDS:
        if qid not in by:
            continue
        print(f"\n-- flip {qid} --")
        for alpha in alphas:
            r = by[qid][alpha]
            print(f"  a={alpha}: rank_final={r['gold_rank_final']} "
                  f"in_ctx={r['gold_in_ctx_session']} ans_chunk={r['ans_chunk_in_ctx']} "
                  f"substr={r['ans_substr_in_ctx']}")

    # harness fidelity: alpha=1.0 vs the run's recorded gold_marks.in_context
    agree = tot = 0
    for q in questions:
        r = by[q["id"]][base_alpha]
        for gm in q.get("gold_marks", []):
            gid = "ep_" + gm["id"][4:]
            tot += 1
            agree += (r["gold_in_ctx_session"].get(gid) == gm["in_context"])
    print(f"\nfidelity vs run.json (alpha={base_alpha}): "
          f"{agree}/{tot} gold in_context marks agree")
    print("done ->", args.out)


if __name__ == "__main__":
    main()
