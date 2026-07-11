"""Self-anchor PPR hub guard — validation on DEEP single-user graphs.

The guard only matters when `self_entity` is on AND the graph is deep enough that
the single SELF_ENTITY_ID node accumulates many fact edges. micro/sample instances
cap at <=6 sessions (too shallow), and the local extractors DROP first-person edges,
so this must run on deep `small`-tier instances with the LLM extractor + self_entity.

For each of the N deepest instances:
  1. ingest once (LLM extractor + self_entity, no reflexion) — the only paid step;
  2. measure the self node as a hub: its degree, its share of RELATED_TO edges, and
     its Personalized-PageRank mass on the instance's own query;
  3. A/B recall@k (FREE — retrieval only, no LLM) under self_guard none vs exclude;
  4. A/B answer accuracy (LLM judge) under none vs exclude.

Reports whether excluding the self anchor from the diffusion changes retrieval/answers.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import networkx as nx

from .config import Config
from .corpus import iter_lme_instances
from .graph import KnowledgeGraph
from .metering import UsageMeter, totals_of
from .models import SELF_ENTITY_ID
from .retrieval import projected_graph
from .testrun import _article, _build_judge_client, _dedup, _judge


def _recall_at_k(object_ids, gold, k):
    gold_art = {_article(x) for x in gold}
    if not gold_art:
        return None
    ranked_art = _dedup(_article(o) for o in object_ids)
    topk = set(ranked_art[:k])
    return len(topk & gold_art) / len(gold_art)


def _self_hub_stats(store, query, embedder, canon, cfg) -> dict:
    """Structural hub measurement on the un-guarded projection."""
    G = projected_graph(store, replace(cfg, self_guard="none"))
    if SELF_ENTITY_ID not in G:
        return {"present": False}
    deg = G.degree(SELF_ENTITY_ID)
    n_edges = G.number_of_edges()
    self_inc = sum(1 for _ in G.edges(SELF_ENTITY_ID))
    # PPR mass on the self node for THIS query's seeds (what retrieval would route)
    from .retrieval import Seeder
    seeds = Seeder(store, embedder, canon, cfg).seed(query)
    pers = {nid: s * canon.idf_weight(nid) for nid, s in seeds.items()
            if nid in G and s > 0}
    self_mass = None
    if pers and sum(pers.values()) > 0:
        ppr = nx.pagerank(G, alpha=cfg.ppr_damping, personalization=pers,
                          weight="weight", max_iter=200)
        self_mass = ppr.get(SELF_ENTITY_ID, 0.0)
        # rank of self among all nodes by mass
        ranked = sorted(ppr.values(), reverse=True)
        self_rank = 1 + sum(1 for v in ranked if v > (self_mass or 0))
    else:
        self_rank = None
    return {"present": True, "degree": deg, "self_incident_edges": self_inc,
            "total_edges": n_edges,
            "edge_share": round(self_inc / n_edges, 3) if n_edges else 0.0,
            "ppr_mass": round(self_mass, 5) if self_mass is not None else None,
            "ppr_rank": self_rank}


def run(*, tier: str = "small", deepest: int = 6, k: int = 3, ctx_episodes: int = 3,
        judge: bool = True, model: str | None = None, out: str = "pm/results",
        store_path: str = "store/guard_eval.db", log=print) -> dict:
    # stressed regime: small k and a tight context window (the task's spec)
    all_inst = list(iter_lme_instances(tier))
    all_inst.sort(key=lambda qi: len(qi[1]), reverse=True)
    instances = all_inst[:deepest]
    log(f"guard eval: {len(instances)} deepest of {len(all_inst)} {tier} instances "
        f"(sessions: {[len(s) for _, s in instances]}), k={k}, ctx={ctx_episodes}\n")

    cfg = Config.default()
    cfg = replace(cfg, self_entity=True, self_name="me", reflexion=False,
                  top_k=k, rag_context_episodes=ctx_episodes)
    if model:
        cfg = replace(cfg, llm_model=model, rag_model=model, l3_model=model)

    modes = ("none", "exclude", "cap", "seed")          # all guard strategies the task proposed
    jclient = _build_judge_client(cfg.judge_model) if judge else None
    judge_meter = UsageMeter()
    rows = []
    ingest_cost = 0.0
    t0 = time.time()

    for i, (q, sessions) in enumerate(instances):
        if os.path.exists(store_path):
            os.remove(store_path)
        g = KnowledgeGraph.open(store_path, cfg)
        g.extractor.meter.drain()
        g.ingest(sessions)
        recs = g.extractor.meter.drain() + g.canon.meter.drain()
        ingest_cost += totals_of(recs)["cost_usd"]

        hub = _self_hub_stats(g.store, q["query"], g.embedder, g.canon, cfg)

        # FREE: retrieval-only recall@k under each guard mode
        rec_row = {"id": q["id"], "n_sessions": len(sessions), "kind": q.get("kind", ""),
                   "query": q["query"], "gold": q.get("gold", []), "self_hub": hub,
                   "recall": {}, "accuracy": {}}
        for mode in modes:
            g.config = replace(g.config, self_guard=mode)
            res = g.query(q["query"], mode="ppr", k=k)
            rec_row["recall"][mode] = _recall_at_k(res.object_ids, q.get("gold", []), k)

        # PAID: answer accuracy under each guard mode
        if judge and jclient is not None:
            for mode in modes:
                g.config = replace(g.config, self_guard=mode)
                ans = g.ask(q["query"], k=k)
                jr = _judge(jclient, cfg.judge_model, q, ans.answer, judge_meter)
                rec_row["accuracy"][mode] = (jr or {}).get("score")
        rows.append(rec_row)
        rstr = " ".join(f"{m}={rec_row['recall'][m]}" for m in modes)
        log(f"  {i+1}/{len(instances)} {q['id']:>16} sess={len(sessions):>3} "
            f"hub_deg={hub.get('degree','-')} edge_share={hub.get('edge_share','-')} "
            f"ppr_rank={hub.get('ppr_rank','-')}  recall[{rstr}]")

    if os.path.exists(store_path):
        os.remove(store_path)

    def _avg(key, mode):
        vals = [r[key][mode] for r in rows if r[key].get(mode) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    present = [r for r in rows if r["self_hub"].get("present")]
    npres = max(1, len(present))
    judged = totals_of(judge_meter.drain())
    summary = {
        "tier": tier, "deepest": deepest, "k": k, "ctx_episodes": ctx_episodes,
        "n_instances": len(rows), "modes": list(modes),
        "recall": {m: _avg("recall", m) for m in modes},
        "accuracy": {m: _avg("accuracy", m) for m in modes},
        "avg_self_degree": round(sum(r["self_hub"].get("degree", 0) for r in present) / npres, 1),
        "avg_self_edge_share": round(sum(r["self_hub"].get("edge_share", 0) for r in present) / npres, 3),
        "avg_self_ppr_rank": round(sum(r["self_hub"].get("ppr_rank", 0) or 0 for r in present) / npres, 1),
        "ingest_cost_usd": round(ingest_cost, 6),
        "judge_cost_usd": round(judged["cost_usd"], 6),
        "total_cost_usd": round(ingest_cost + judged["cost_usd"], 6),
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"guard_eval_{tier}_d{deepest}_k{k}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log("\n" + render(summary))
    log(f"\nwrote {path}")
    return summary


def render(s: dict) -> str:
    def f(v):
        return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    modes = s.get("modes", ["none", "exclude"])
    lines = [
        f"self-anchor guard A/B  ({s['n_instances']} deep instances, k={s['k']}, ctx={s['ctx_episodes']})",
        f"  avg self-node degree        {f(s['avg_self_degree'])}",
        f"  avg self edge-share         {f(s['avg_self_edge_share'])}   "
        f"(fraction of all projection edges touching the self node)",
        f"  avg self PPR rank           {f(s.get('avg_self_ppr_rank'))}   "
        f"(1 = self is the single highest-mass node)",
        "  " + "  ".join(f"recall[{m}]={f(s['recall'][m])}" for m in modes),
        "  " + "  ".join(f"acc[{m}]={f(s['accuracy'][m])}" for m in modes),
        f"  cost: ingest ${f(s['ingest_cost_usd'])} + judge ${f(s['judge_cost_usd'])} "
        f"= ${f(s['total_cost_usd'])}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="kg.guard_eval",
                                 description="self-anchor PPR guard A/B on deep instances")
    ap.add_argument("--tier", default="small")
    ap.add_argument("--deepest", type=int, default=6)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=3, help="rag_context_episodes (stressed)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run(tier=a.tier, deepest=a.deepest, k=a.k, ctx_episodes=a.ctx,
        judge=not a.no_judge, model=a.model, out=a.out)
