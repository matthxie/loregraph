"""Offline Round-4 eval: context-prefix promotion sweep (docs/OFFLINE_EVAL.md Round 4).

NO paid LLM calls. Same harness pattern as scripts/offline_eval_round3.py: drives the
exact read path (HybridRetriever.retrieve → ContextBuilder.build — what
KnowledgeGraph.search() runs) against the CACHED per-instance benchmark stores from
runs/sample-datefix-events-1 (store/cache/*.db, extraction cost already sunk). Each
cached store is COPIED to a temp path first; the cache is never opened read-write.

Sweeps config.seed_promote in --settings (default 0,1,2). Per (question, N) records:
  * gold session rank in the final ranked list and in the raw PPR pool
  * gold-in-context at session level and at answer-chunk level; answer substring
  * which episodes promotion moved into the prefix (res.promoted), whether each is a
    gold session, whether it landed in the built context, whether it sits in the raw
    PPR top-`rerank_keep_ppr_top` (the tail re-insertion promotion is meant to rescue)
  * context size
Displacement (who lost a prefix seat, and was it gold) is derived in the summary by
diffing each question's context episodes against its own N=0 row.

N=0 rows double as the harness-fidelity gate against run.json's recorded gold_marks
(Round 3 reproduced 184/184); the script aborts if the gate fails.

Run:  .venv/bin/python scripts/offline_eval_round4.py [--out runs/offline_eval_round4]
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
# Round-3 regression set: 0977f2af enters top-8 at rank 8 (below prefix — expected
# WIN); 06f04340 gold rank 6 (possible win); 2ce6a0f2 gold never in pool (expected
# NO change — the rule must correctly do nothing).
FLIP_QIDS = ("0977f2af", "06f04340", "2ce6a0f2")

INGEST_OVERRIDES = {"extractor_backend": "cue_gated", "event_facts": True,
                    "ingest_date_filter": True}
QUERY_OVERRIDES = {"rag_retarget": "ce", "rag_provenance_promote": True,
                   "mmr_lambda": 1.0, "rag_parent_expand": 2,
                   "rag_chunks_per_source": 2, "history_all_lanes": True}
K = 8   # run.json config.k


def sess(eid: str) -> str:
    return eid.split("#", 1)[0]


def gold_sessions(q: dict) -> list[str]:
    return ["ep_" + g[4:] if g.startswith("obj_") else g for g in q["gold"]]


def locate_stores(questions: list[dict]) -> dict[str, str]:
    from kg.corpus import iter_lme_instances
    os.environ.setdefault("KG_LLM", "openai")
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


def eval_question(store, embedder, canon, base_cfg: Config, q: dict, n: int) -> dict:
    cfg = replace(base_cfg, seed_promote=n)
    retriever = HybridRetriever(store, embedder, canon, cfg)
    builder = ContextBuilder(store, cfg)
    as_of = q.get("question_date")
    res = retriever.retrieve(q["query"], k=K, as_of=as_of)
    ep_ids, facts, blob = builder.build(res)

    golds = gold_sessions(q)
    ranked = list(res.object_ids)
    pool = [eid for eid, _s in getattr(res, "ppr_pool", [])]

    def first_rank(order: list[str], gold: str) -> int | None:
        for i, eid in enumerate(order):
            if sess(eid) == gold:
                return i + 1
        return None

    ctx_sessions = {sess(e) for e in ep_ids}
    ans = (q.get("answer_expected") or "").strip().lower()
    ans_chunks = []
    if ans:
        for gold in golds:
            base_node = store.get_node(gold)
            cands = [gold] if base_node is not None else []
            for nd in store.nodes_of_type(NodeType.EPISODE):
                if nd.id.startswith(gold + "#"):
                    cands.append(nd.id)
            for cid in cands:
                nd = store.get_node(cid)
                if nd and ans in (nd.raw_text or "").lower():
                    ans_chunks.append(cid)

    keep_n = int(getattr(cfg, "rerank_keep_ppr_top", 0))
    raw_top = {sess(e) for e in pool[:keep_n]} if keep_n else set()
    promoted = list(getattr(res, "promoted", []) or [])
    return {
        "qid": q["id"], "n": n, "lane": getattr(res, "lane", ""),
        "gold_rank_final": {g: first_rank(ranked, g) for g in golds},
        "gold_rank_pool": {g: first_rank(pool, g) for g in golds},
        "gold_in_ctx_session": {g: (g in ctx_sessions) for g in golds},
        "all_gold_in_ctx": all(g in ctx_sessions for g in golds),
        "any_gold_in_ctx": any(g in ctx_sessions for g in golds),
        "ans_chunk_in_ctx": (any(c in ep_ids for c in ans_chunks)
                             if ans_chunks else None),
        "ans_substr_in_ctx": (ans in blob.lower()) if ans else None,
        "ctx_chars": len(blob), "n_ctx_eps": len(ep_ids),
        "promoted": promoted,
        "promoted_gold": [e for e in promoted if sess(e) in set(golds)],
        "promoted_in_ctx": [e for e in promoted if e in ep_ids],
        "promoted_from_ppr_top": [e for e in promoted if sess(e) in raw_top],
        "ranked": ranked, "pool": pool[:64], "ctx_eps": ep_ids,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="0,1,2")
    ap.add_argument("--out", default="runs/offline_eval_round4")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    settings = [int(s) for s in args.settings.split(",")]
    assert settings[0] == 0, "first setting must be the N=0 baseline (fidelity gate)"
    os.makedirs(args.out, exist_ok=True)

    run = json.load(open(RUN_JSON))
    questions = run["query"]["queries"][: args.limit]
    stores = locate_stores(questions)
    missing = [q["id"] for q in questions if q["id"] not in stores]
    if missing:
        sys.exit(f"no cache hit for {len(missing)} instances: {missing[:5]} …")
    print(f"{len(questions)} questions, all cached; seed_promote={settings}", flush=True)

    base_cfg = Config.default()
    for k, v in {**INGEST_OVERRIDES, **QUERY_OVERRIDES}.items():
        setattr(base_cfg, k, v)

    work = tempfile.mkdtemp(prefix="round4-")
    rows = []
    t0 = time.time()
    for i, q in enumerate(questions):
        wp = os.path.join(work, f"{q['id']}.db")
        _sqlite_copy(stores[q["id"]], wp)
        g = KnowledgeGraph.open(wp, base_cfg)
        for n in settings:
            rows.append(eval_question(g.store, g.embedder, g.canon, base_cfg, q, n))
        del g
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(wp + suf):
                os.remove(wp + suf)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(questions)}  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)

    by = {}
    for r in rows:
        by.setdefault(r["qid"], {})[r["n"]] = r

    # ---------------- fidelity gate FIRST (N=0 vs run.json gold_marks) ----------
    agree = tot = 0
    disagreements = []
    for q in questions:
        r = by[q["id"]][0]
        for gm in q.get("gold_marks", []):
            gid = "ep_" + gm["id"][4:]
            tot += 1
            if r["gold_in_ctx_session"].get(gid) == gm["in_context"]:
                agree += 1
            else:
                disagreements.append((q["id"], gid))
    print(f"\nFIDELITY GATE (seed_promote=0 vs run.json in_context): {agree}/{tot}")
    if disagreements:
        print("  disagreements:", disagreements[:10])
        sys.exit("fidelity gate FAILED — offline baseline does not reproduce the "
                 "paid run; investigate before trusting the sweep")

    # ---------------- sweep matrix ----------------
    print("\n== seed_promote sweep (n=%d questions) ==" % len(by))
    print(f"{'N':>3} {'allgold_ctx':>11} {'anygold_ctx':>11} {'ans_chunk':>9} "
          f"{'ans_substr':>10} {'ctx_chars':>9} {'fired':>5} {'from_ppr_top':>12}")
    for n in settings:
        rs = [by[qid][n] for qid in by]
        chunk = [r["ans_chunk_in_ctx"] for r in rs if r["ans_chunk_in_ctx"] is not None]
        sub = [r["ans_substr_in_ctx"] for r in rs if r["ans_substr_in_ctx"] is not None]
        fired = sum(1 for r in rs if r["promoted"])
        ppr_rescue = sum(1 for r in rs if r["promoted_from_ppr_top"])
        print(f"{n:>3} {sum(r['all_gold_in_ctx'] for r in rs):>11} "
              f"{sum(r['any_gold_in_ctx'] for r in rs):>11} "
              f"{sum(chunk):>9} {sum(sub):>10} "
              f"{sum(r['ctx_chars'] for r in rs)//len(rs):>9} {fired:>5} "
              f"{ppr_rescue:>12}")

    # ---------------- wins / regressions vs N=0 ----------------
    for n in settings[1:]:
        for metric in ("all_gold_in_ctx", "any_gold_in_ctx", "ans_chunk_in_ctx",
                       "ans_substr_in_ctx"):
            fixed = [q for q in by if by[q][n][metric] is True
                     and by[q][0][metric] is False]
            broke = [q for q in by if by[q][n][metric] is False
                     and by[q][0][metric] is True]
            if fixed or broke:
                print(f"N={n} {metric}: fixed {fixed}  broke {broke}")

    # ---------------- promotion fire + displacement analysis ----------------
    for n in settings[1:]:
        fired = [q for q in by if by[q][n]["promoted"]]
        print(f"\nN={n}: promotion fired on {len(fired)}/{len(by)} questions")
        disp_gold = []
        for qid in fired:
            r0, rn = by[qid][0], by[qid][n]
            golds = set(r0["gold_in_ctx_session"])
            lost = [s for s in {sess(e) for e in r0["ctx_eps"]}
                    if s not in {sess(e) for e in rn["ctx_eps"]}]
            lost_gold = [s for s in lost if s in golds]
            if lost_gold:
                disp_gold.append((qid, lost_gold))
        print(f"  displaced-a-gold-session on {len(disp_gold)} questions: "
              f"{disp_gold[:10]}")
        not_landed = [(q, [e for e in by[q][n]["promoted"]
                           if e not in by[q][n]["promoted_in_ctx"]])
                      for q in fired
                      if len(by[q][n]["promoted_in_ctx"]) < len(by[q][n]["promoted"])]
        print(f"  promoted-but-not-in-context on {len(not_landed)}: {not_landed[:10]}")
        gold_promos = [q for q in fired if by[q][n]["promoted_gold"]]
        print(f"  promoted a GOLD session on {len(gold_promos)}: {gold_promos}")
        rescue = [q for q in fired if by[q][n]["promoted_from_ppr_top"]]
        print(f"  promoted episode was in raw-PPR top-{base_cfg.rerank_keep_ppr_top} "
              f"(keep_ppr_top tail re-insertion rescue) on {len(rescue)}: {rescue[:15]}")

    # ---------------- flip questions detail ----------------
    for qid in FLIP_QIDS:
        if qid not in by:
            continue
        print(f"\n-- flip {qid} --")
        for n in settings:
            r = by[qid][n]
            print(f"  N={n}: rank_final={r['gold_rank_final']} "
                  f"in_ctx={r['gold_in_ctx_session']} ans_chunk={r['ans_chunk_in_ctx']} "
                  f"substr={r['ans_substr_in_ctx']} promoted={r['promoted']}")

    # ---------------- canaries: stable-correct paid-run questions ----------------
    canaries = sorted(q["id"] for q in questions
                      if (q.get("judge") or {}).get("correct")
                      and all(gm["in_context"] for gm in q.get("gold_marks", [])))[:10]
    print(f"\ncanaries (paid-run judge-correct, gold in ctx): {canaries}")
    for n in settings[1:]:
        changed = []
        for qid in canaries:
            r0, rn = by[qid][0], by[qid][n]
            delta = [m for m in ("all_gold_in_ctx", "any_gold_in_ctx",
                                 "ans_chunk_in_ctx", "ans_substr_in_ctx")
                     if r0[m] != rn[m]]
            ctx_changed = r0["ctx_eps"] != rn["ctx_eps"]
            if delta or ctx_changed:
                changed.append((qid, delta, "ctx_eps_changed" if ctx_changed else ""))
        print(f"N={n} canary changes: {changed or 'none'}")

    print("\ndone ->", args.out)


if __name__ == "__main__":
    main()
