"""Offline Round-5 eval: OUT-OF-POOL recall diagnosis (docs/OFFLINE_EVAL.md Round 5).

NO paid LLM calls. Same harness pattern as scripts/offline_eval_round3.py /
_round4.py: drives the exact read path (HybridRetriever.retrieve -> ContextBuilder.build,
what KnowledgeGraph.search() runs) against the CACHED per-instance benchmark stores from
runs/sample-datefix-events-1 (store/cache/*.db, extraction cost already sunk). Each
cached store is COPIED to a temp path first; the cache is never opened read-write.

This round does NOT sweep a knob (the Round-3/4 seed_fusion_alpha / seed_promote knobs
were reverted from the tree). It runs ONE baseline retrieval per question and, for every
GOLD session, traces the retrieval funnel stage by stage so out-of-pool failures can be
assigned a primary cause:

  Stage 1 SEEDING     - per gold chunk: BM25 rank+score over the composite episode docs
                        (Seeder._episode_doc), embedding-sim rank+score, whether it makes
                        the fused seed dict (Seeder.seed), and its raw-source ranks vs the
                        seed_k cutoff.  Composite-doc surfaces (mention/tag names) dumped.
  Stage 2 GRAPH REACH - full PPR mass per gold chunk (exact PPR reconstructed the same way
                        PPRRetriever.retrieve builds it: seed*idf personalization over the
                        as_of-filtered projection), its PPR rank, whether the pool trim
                        (cand[:max(k*3,k)]) or the MMR pool cut it; entity overlap between
                        the gold chunk's entities and the seeded entity neighborhood.
  Stage 3 LANE/CE     - lane, whether gold sits in the final pool (base.objects) / top-k /
                        context (the CE only trims via rerank_pool, captured downstream).

Also runs a SMALLER-CHUNKS measurement (no re-ingest, no writes): for every gold chunk
that carries the answer substring, the chunk's raw text is re-split at natural boundaries
(kg/chunkers._turn_units / _pack) and each sub-chunk embedded with the SAME embedder; the
best sub-chunk cosine is compared to the current embedding-seed cutoff (the score of the
rank-seed_k episode) to decide whether smaller chunks would have cleared the seed gate.

N=0 baseline rows double as the harness-fidelity gate against run.json's recorded
gold_marks (Rounds 3-4 reproduced 184/184); the script reports the number.

Run:  .venv/bin/python scripts/offline_eval_round5.py [--out runs/offline_eval_round5]
      [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402

import kg.graph as kg_graph                                     # noqa: E402
from kg import Config, KnowledgeGraph                           # noqa: E402
from kg.chunkers import _pack, _turn_units                      # noqa: E402
from kg.extractors import ScriptedExtractor                     # noqa: E402
from kg.ingest_cache import cache_path, ingest_cache_key        # noqa: E402
from kg.ingest_cache import _sqlite_copy                        # noqa: E402
from kg.models import EdgeType, NodeType                        # noqa: E402
from kg.rag import ContextBuilder                               # noqa: E402
from kg.retrieval import (HybridRetriever, Seeder,              # noqa: E402
                          projected_graph, personalized_pagerank,
                          local_push_ppr, self_like_ids)

# ---- hard no-LLM guard (same as offline_eval.py): kg auto-loads .env on import.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)
kg_graph.get_extractor = lambda config: ScriptedExtractor({})

RUN_JSON = "runs/sample-datefix-events-1/run.json"
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


def chunk_entities(store, cid: str) -> tuple[set[str], list[str]]:
    """(canonical entity ids, mention/entity surface names) reachable from a chunk via
    MENTIONED_IN -> RESOLVES_TO — the same corridor Seeder._episode_doc walks."""
    ent_ids: set[str] = set()
    surfaces: list[str] = []
    for mid, _d in store.neighbors(cid, etypes={EdgeType.MENTIONED_IN}, direction="in"):
        mnode = store.get_node(mid)
        if mnode is not None and (mnode.name or ""):
            surfaces.append(mnode.name)
        for eid, _d2 in store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                        direction="out"):
            ent = store.get_node(eid)
            if ent is not None and ent.valid:
                ent_ids.add(eid)
                if ent.name:
                    surfaces.append(ent.name)
    return ent_ids, sorted(set(surfaces))


def full_ppr(store, embedder, canon, cfg, query, as_of):
    """Reconstruct the exact PPR mass PPRRetriever.retrieve computes, but keep the FULL
    (untrimmed) mass dict so a gold chunk's true PPR rank is visible even below the
    cand[:max(k*3,k)] trim. Mirrors PPRRetriever.retrieve lines 539-564."""
    seeder = Seeder(store, embedder, canon, cfg)
    seeds = seeder.seed(query)
    if not seeds:
        return {}, {}, seeder
    G = projected_graph(store, cfg, as_of=as_of)
    skip_self = getattr(cfg, "self_guard", "none") in ("exclude", "seed")
    self_ids = self_like_ids(store, cfg) if skip_self else set()
    pers = {}
    for nid, s in seeds.items():
        if skip_self and nid in self_ids:
            continue
        if nid in G and s > 0:
            pers[nid] = s * canon.idf_weight(nid)
    if not pers or sum(pers.values()) <= 0:
        return {}, dict(seeds), seeder
    if getattr(cfg, "ppr_backend", "global") == "push":
        ppr = local_push_ppr(G, alpha=cfg.ppr_damping, personalization=pers,
                             eps=getattr(cfg, "ppr_push_eps", 1e-6))
    else:
        ppr = personalized_pagerank(G, alpha=cfg.ppr_damping, personalization=pers,
                                    max_iter=200)
    return ppr, dict(seeds), seeder


def rank_in(order: list[str], target: str):
    for i, x in enumerate(order):
        if x == target:
            return i + 1
    return None


def smaller_chunks_test(embedder, raw_text: str, qv, emb_cutoff: float,
                        target: int) -> dict:
    """Re-split raw_text at natural (turn) boundaries into ~target-char sub-chunks, embed
    each with the SAME embedder, and report the best sub-chunk cosine vs the query and
    whether it clears emb_cutoff (the current rank-seed_k episode-embedding score). No
    ingest, no writes."""
    _hdr, units = _turn_units(raw_text or "")
    if not units:
        units = [raw_text or ""]
    subs = _pack(units, target=target, max_chars=max(target * 2, 800))
    subs = [s for s in subs if s.strip()]
    if not subs:
        return {"n_sub": 0, "best_cos": None, "clears": False}
    vecs = embedder.embed(subs)
    best = -1.0
    best_i = -1
    for i, v in enumerate(vecs):
        v = np.asarray(v, dtype=float)
        nv = v / (np.linalg.norm(v) or 1.0)
        cos = float(np.dot(nv, qv))
        if cos > best:
            best, best_i = cos, i
    return {"n_sub": len(subs), "best_cos": best,
            "best_sub_len": len(subs[best_i]) if best_i >= 0 else None,
            "emb_cutoff": emb_cutoff, "clears": bool(best >= emb_cutoff),
            "sub_lens": [len(s) for s in subs]}


def eval_question(g, base_cfg, q, sub_target):
    store, embedder, canon = g.store, g.embedder, g.canon
    as_of = q.get("question_date")
    query = q["query"]

    # -- real retrieval (the exact search() read path) -------------------------
    retriever = HybridRetriever(store, embedder, canon, base_cfg)
    builder = ContextBuilder(store, base_cfg)
    res = retriever.retrieve(query, k=K, as_of=as_of)
    ep_ids, _facts, blob = builder.build(res)
    pool = [e for e, _ in getattr(res, "ppr_pool", [])]     # base.objects (MMR pool)
    pool_sess = {sess(e) for e in pool}
    final = list(res.object_ids)
    final_sess = {sess(e) for e in final}
    ctx_sess = {sess(e) for e in ep_ids}
    lane = getattr(res, "lane", "")

    # -- funnel internals (reconstructed exactly as PPRRetriever builds them) ---
    ppr, seeds, seeder = full_ppr(store, embedder, canon, base_cfg, query, as_of)
    qv = np.asarray(embedder.embed([query])[0], dtype=float)
    qv = qv / (np.linalg.norm(qv) or 1.0)

    # full embedding ranking over episodes, full BM25 ranking, full ppr ordering
    n_eps = len(list(store.nodes_of_type(NodeType.EPISODE)))
    emb_hits = store.vectors.search("episode", qv, k=n_eps, floor=-1.0)
    emb_rank = {oid: i + 1 for i, (oid, _c) in enumerate(emb_hits)}
    emb_score = {oid: float(c) for oid, c in emb_hits}
    # embedding seed cutoff = score of the rank-seed_k episode (what a chunk must beat)
    seed_k = int(base_cfg.seed_k)
    emb_cutoff = float(emb_hits[seed_k - 1][1]) if len(emb_hits) >= seed_k else (
        float(emb_hits[-1][1]) if emb_hits else 0.0)
    bm = seeder.bm25_search(query, k=n_eps, normalized=True)
    bm_rank = {oid: i + 1 for i, (oid, _s) in enumerate(bm)}
    bm_score = {oid: float(s) for oid, s in bm}
    ppr_order = sorted(ppr.items(), key=lambda x: (-x[1], x[0]))
    # PPR rank among EPISODES only (matches PPRRetriever's cand filter)
    ppr_ep_order = [nid for nid, _m in ppr_order
                    if (nd := store.get_node(nid)) and nd.ntype == NodeType.EPISODE
                    and nd.valid]
    ppr_rank = {nid: i + 1 for i, nid in enumerate(ppr_ep_order)}
    # the pool passed to PPRRetriever by HybridRetriever, and its internal trim
    from kg.route import MULTIHOP
    base_pool = max(int(getattr(base_cfg, "rerank_pool", 32)), K * 3)
    pool_arg = base_pool * 2 if lane == MULTIHOP else base_pool
    trim_n = max(pool_arg * 3, pool_arg)             # PPRRetriever cand[:max(k*3,k)]

    # seeded entity neighborhood (entities that carry personalization mass)
    seeded_entities = {nid for nid in seeds
                       if (nd := store.get_node(nid)) and nd.ntype == NodeType.ENTITY
                       and nd.valid}

    golds = gold_sessions(q)
    ans = (q.get("answer_expected") or "").strip().lower()

    gold_out = {}
    for gs in golds:
        chunks = [gs] if store.get_node(gs) is not None else []
        for nd in store.nodes_of_type(NodeType.EPISODE):
            if nd.id.startswith(gs + "#"):
                chunks.append(nd.id)
        chunk_rows = []
        for cid in chunks:
            nd = store.get_node(cid)
            raw = (nd.raw_text or "") if nd else ""
            ent_ids, surfaces = chunk_entities(store, cid)
            overlap = ent_ids & seeded_entities
            has_ans = bool(ans) and ans in raw.lower()
            row = {
                "id": cid, "len": len(raw),
                "emb_rank": emb_rank.get(cid), "emb_score": emb_score.get(cid),
                "bm25_rank": bm_rank.get(cid), "bm25_score": bm_score.get(cid),
                "seed_score": seeds.get(cid), "in_seed_dict": cid in seeds,
                "ppr_rank": ppr_rank.get(cid), "ppr_mass": ppr.get(cid),
                "within_trim": (ppr_rank.get(cid) is not None
                                and ppr_rank[cid] <= trim_n),
                "in_pool": cid in set(pool),
                "n_entities": len(ent_ids),
                "n_seeded_overlap": len(overlap),
                "seeded_overlap_names": sorted(
                    (store.get_node(e).name for e in overlap
                     if store.get_node(e)), key=str)[:12],
                "surfaces": surfaces[:20],
                "has_answer_substr": has_ans,
            }
            # smaller-chunks test only where it matters: chunk carries the answer but
            # is not in the pool (a candidate dilution/lexical fix)
            if has_ans and cid not in set(pool):
                row["subchunk"] = smaller_chunks_test(
                    embedder, raw, qv, emb_cutoff, sub_target)
            chunk_rows.append(row)

        in_pool = any(r["in_pool"] for r in chunk_rows)
        gold_out[gs] = {
            "n_chunks": len(chunk_rows),
            "in_pool": in_pool,
            "in_final": gs in final_sess,
            "in_context": gs in ctx_sess,
            "any_reached_ppr": any(r["ppr_rank"] is not None for r in chunk_rows),
            "any_in_seed_dict": any(r["in_seed_dict"] for r in chunk_rows),
            "ans_chunks": [r["id"] for r in chunk_rows if r["has_answer_substr"]],
            "chunks": chunk_rows,
        }

    return {
        "qid": q["id"], "query": query, "lane": lane, "as_of": as_of,
        "kind": q.get("kind"),
        "n_episodes": n_eps, "pool_size": len(pool), "pool_arg": pool_arg,
        "trim_n": trim_n, "seed_k": seed_k, "emb_cutoff": emb_cutoff,
        "n_seeded_entities": len(seeded_entities),
        "answer_expected": q.get("answer_expected"),
        "gold": gold_out,
        "pool_ids": pool, "final_ids": final, "ctx_ids": ep_ids,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/offline_eval_round5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sub-target", type=int, default=500,
                    help="target chars for the smaller-chunks re-split measurement")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    run = json.load(open(RUN_JSON))
    questions = run["query"]["queries"][: args.limit]
    stores = locate_stores(questions)
    missing = [q["id"] for q in questions if q["id"] not in stores]
    if missing:
        sys.exit(f"no cache hit for {len(missing)} instances: {missing[:5]} ...")
    print(f"{len(questions)} questions, all cached", flush=True)

    base_cfg = Config.default()
    for k, v in {**INGEST_OVERRIDES, **QUERY_OVERRIDES}.items():
        setattr(base_cfg, k, v)

    work = tempfile.mkdtemp(prefix="round5-")
    rows = []
    t0 = time.time()
    for i, q in enumerate(questions):
        wp = os.path.join(work, f"{q['id']}.db")
        _sqlite_copy(stores[q["id"]], wp)
        g = KnowledgeGraph.open(wp, base_cfg)
        rows.append(eval_question(g, base_cfg, q, args.sub_target))
        del g
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(wp + suf):
                os.remove(wp + suf)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(questions)}  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)

    # ---------------- fidelity gate (baseline vs run.json gold_marks) ----------
    by = {r["qid"]: r for r in rows}
    agree = tot = 0
    disagreements = []
    for q in questions:
        r = by[q["id"]]
        ctx_sess = {sess(e) for e in r["ctx_ids"]}
        for gm in q.get("gold_marks", []):
            gid = "ep_" + gm["id"][4:]
            tot += 1
            if (gid in ctx_sess) == gm["in_context"]:
                agree += 1
            else:
                disagreements.append((q["id"], gid, gid in ctx_sess, gm["in_context"]))
    print(f"\nFIDELITY GATE (baseline vs run.json in_context): {agree}/{tot}")
    for d in disagreements[:20]:
        print("  DISAGREE", d)

    # ---------------- out-of-pool population -----------------------------------
    oop = []   # (qid, gold_session)
    for r in rows:
        for gs, gd in r["gold"].items():
            if not gd["in_pool"]:
                oop.append((r["qid"], gs))
    print(f"\nOUT-OF-POOL gold sessions: {len(oop)} across "
          f"{len({q for q, _ in oop})} questions")
    for qid, gs in oop:
        r = by[qid]
        gd = r["gold"][gs]
        best_seed = max((c["seed_score"] or 0) for c in gd["chunks"]) \
            if gd["chunks"] else 0
        best_ppr = min((c["ppr_rank"] for c in gd["chunks"]
                        if c["ppr_rank"] is not None), default=None)
        best_emb = min((c["emb_rank"] for c in gd["chunks"]
                        if c["emb_rank"] is not None), default=None)
        best_bm = min((c["bm25_rank"] for c in gd["chunks"]
                       if c["bm25_rank"] is not None), default=None)
        print(f"  {qid} {gs[-24:]} lane={r['lane']} chunks={gd['n_chunks']} "
              f"seed_dict={gd['any_in_seed_dict']} best_seed={best_seed:.3f} "
              f"emb_rank={best_emb} bm25_rank={best_bm} ppr_rank={best_ppr} "
              f"reached_ppr={gd['any_reached_ppr']} ans_chunks={len(gd['ans_chunks'])}")

    print("\ndone ->", args.out)


if __name__ == "__main__":
    main()
