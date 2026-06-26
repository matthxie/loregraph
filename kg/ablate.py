"""Ablation harness: the FULL production pipeline vs the RAW PPR-RAG engine.

Both systems are the same `KnowledgeGraph` — only config flags differ, so the comparison
isolates exactly what the production strategy adds:

  raw  : extractor_backend=gliner_yake_cooccur, rerank=off, route=off  (the engine baseline)
  full : the defaults — cue-gated extraction + 4-lane routing + conditional cross-encoder
         rerank + fact-lane augmentation + evolution-aware context

Runs the LongMemEval per-instance protocol (a fresh graph per question) and scores both with
the identical `kg.testrun._score_query`. Reports cost, speed, recall@k, and answer accuracy.

    python -m kg.ablate --tier sample --k 3 --ctx 3
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace

from .config import Config
from .corpus import iter_lme_instances
from .graph import KnowledgeGraph
from .metering import UsageMeter, totals_of
from .store import GraphStore
from .testrun import _build_judge_client, _query_totals, _score_query


def _system_config(system: str, *, k: int, model: str | None,
                   ctx_episodes: int | None) -> Config:
    cfg = replace(Config.default(), top_k=k, reflexion=False)
    if ctx_episodes is not None:
        cfg = replace(cfg, rag_context_episodes=ctx_episodes)
    if model:
        cfg = replace(cfg, llm_model=model, rag_model=model, l3_model=model)
    if system == "raw":                       # the engine baseline: flags off, local extraction
        cfg = replace(cfg, extractor_backend="gliner_yake_cooccur",
                      route=False, rerank=False, fact_lane_augment=False)
    return cfg                                  # "full" keeps all the production defaults on


def run_system(system: str, instances, *, k: int, judge: bool, model: str | None,
               ctx_episodes: int | None, store_path: str, log=print) -> dict:
    cfg = _system_config(system, k=k, model=model, ctx_episodes=ctx_episodes)
    jclient = _build_judge_client(cfg.l3_model) if judge else None
    judge_meter = UsageMeter()

    qrecords: list[dict] = []
    ingest_cost = ingest_tokens = ingest_llm = 0.0
    ingest_seconds = query_seconds = 0.0
    agg_nodes = agg_edges = agg_related = 0
    escal = {"seen": 0, "escalated": 0, "cue_counts": {}}
    rerank_active = None
    lanes: dict[str, int] = {}

    for i, (q, sessions) in enumerate(instances):
        if os.path.exists(store_path):
            os.remove(store_path)
        g = KnowledgeGraph.open(store_path, cfg)
        g.extractor.meter.drain()
        t0 = time.time()
        g.ingest(sessions)
        ingest_seconds += time.time() - t0
        tok = totals_of(g.extractor.meter.drain() + g.canon.meter.drain())
        ingest_cost += tok["cost_usd"]
        ingest_tokens += tok["tokens"]
        ingest_llm += tok["llm_calls"]
        stats = g.store.stats()
        agg_nodes += stats["nodes"]
        agg_edges += stats["edges"]
        agg_related += stats["by_edge_type"].get("RELATED_TO", 0)
        if hasattr(g.extractor, "escalation_summary"):
            es = g.extractor.escalation_summary()
            escal["seen"] += es["seen"]
            escal["escalated"] += es["escalated"]
            for kk, vv in es["cue_counts"].items():
                escal["cue_counts"][kk] = escal["cue_counts"].get(kk, 0) + vv

        t1 = time.time()
        ans = g.ask(q["query"], k=k, kind=q.get("kind"))
        query_seconds += time.time() - t1
        if getattr(ans, "rerank_active", None) is not None:
            rerank_active = ans.rerank_active
        rec = _score_query(q, ans, k, g.store, jclient, cfg, judge_meter)
        lane = getattr(ans, "lane", "")
        rec["lane"] = lane
        if lane:
            lanes[lane] = lanes.get(lane, 0) + 1
        qrecords.append(rec)
        log(f"  [{system:4}] {i+1}/{len(instances)} {rec['id']:>16} lane={lane or '-':8} "
            f"recall@{k}={rec['recall_at_k']:.2f} hit={'Y' if rec['hit'] else '.'} "
            f"judge={(rec['judge'] or {}).get('score', '-')}  ${ingest_cost:.4f}")

    if os.path.exists(store_path):
        os.remove(store_path)
    qt = _query_totals(qrecords, judge_meter, k)
    n = max(1, len(instances))
    return {
        "system": system, "n_instances": len(instances),
        "recall_at_k": qt["recall_at_k"], "mrr": qt["mrr"],
        "answer_accuracy": qt.get("response_accuracy"),
        "citation_grounding": qt.get("citation_grounding"),
        "ingest_cost_usd": round(ingest_cost, 6), "query_cost_usd": round(qt["cost_usd"], 6),
        "total_cost_usd": round(ingest_cost + qt["cost_usd"], 6),
        "ingest_llm_calls": int(ingest_llm),
        "ingest_seconds": round(ingest_seconds, 1), "query_seconds": round(query_seconds, 1),
        "avg_related_to": round(agg_related / n, 1), "avg_edges": round(agg_edges / n, 1),
        "rerank_active": rerank_active, "lanes": lanes, "escalation": escal,
        "queries": qrecords,
    }


def _fmt(v):
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render_table(results: list[dict], k: int) -> str:
    rows = [("recall@%d" % k, "recall_at_k"), ("mrr", "mrr"),
            ("answer accuracy (judge)", "answer_accuracy"),
            ("citation grounding", "citation_grounding"),
            ("ingest $", "ingest_cost_usd"), ("query $", "query_cost_usd"),
            ("total $", "total_cost_usd"), ("ingest LLM calls", "ingest_llm_calls"),
            ("ingest seconds", "ingest_seconds"), ("query seconds", "query_seconds"),
            ("avg RELATED_TO/graph", "avg_related_to"), ("avg edges/graph", "avg_edges")]
    by = {r["system"]: r for r in results}
    order = [s for s in ("raw", "full") if s in by]
    w = 26
    head = "metric".ljust(w) + "".join(s.rjust(16) for s in order)
    lines = [head, "-" * len(head)]
    for label, key in rows:
        lines.append(label.ljust(w) + "".join(_fmt(by[s].get(key)).rjust(16) for s in order))
    full = by.get("full", {})
    esc = full.get("escalation", {})
    if esc.get("seen"):
        rate = esc["escalated"] / esc["seen"]
        lines.append("")
        lines.append(f"full cue-escalation: {esc['escalated']}/{esc['seen']} sections "
                     f"({rate:.1%}) called Haiku  cues={esc.get('cue_counts', {})}")
    ra = full.get("rerank_active")
    flag = "YES" if ra else ("NO — DEGRADED to PPR order!" if ra is False else "?")
    lines.append(f"full cross-encoder active: {flag}   lanes routed: {full.get('lanes', {})}")
    return "\n".join(lines)


def main(tier: str = "sample", k: int = 8, queries: int | None = None,
         judge: bool = True, model: str | None = None, ctx_episodes: int | None = None,
         out: str = "results", store_path: str = "store/ablate.db") -> dict:
    instances = list(iter_lme_instances(tier, limit=queries))
    reg = f"k={k}" + (f", ctx={ctx_episodes}" if ctx_episodes is not None else "")
    print(f"ablation on longmemeval:{tier}  ({len(instances)} instances, per-instance, {reg})\n")
    results = []
    for system in ("raw", "full"):
        print(f"=== {system} ===")
        results.append(run_system(system, instances, k=k, judge=judge, model=model,
                                   ctx_episodes=ctx_episodes, store_path=store_path))
        print()
    table = render_table(results, k)
    print(table)
    os.makedirs(out, exist_ok=True)
    suffix = f"_k{k}" + (f"_ctx{ctx_episodes}" if ctx_episodes is not None else "")
    path = os.path.join(out, f"ablate_{tier}{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tier": tier, "k": k, "ctx_episodes": ctx_episodes, "model": model,
                   "results": results, "table": table}, f, indent=2)
    print(f"\nwrote {path}")
    return results


if __name__ == "__main__":
    import argparse
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="kg.ablate",
                                 description="full production pipeline vs raw PPR-RAG engine")
    ap.add_argument("--tier", default="sample")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=None, help="rag_context_episodes (stressed, e.g. 3)")
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(tier=a.tier, k=a.k, queries=a.queries, judge=not a.no_judge, model=a.model,
         ctx_episodes=a.ctx, out=a.out)
