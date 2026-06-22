"""Command-line interface:  python -m kg <command> [options]

    ingest       build the graph from dataset/ (or re-ingest; cache skips unchanged)
    communities  detect communities + summaries (Path B / breadth queries)
    query        retrieve for a question (auto-routes local↔global)
    stats        print node/edge counts
    inspect      dump one node + its neighbours
    eval         recall@k / MRR ablation across retrieval modes
"""
from __future__ import annotations

import argparse
import json
import os

from .config import Config
from .corpus import load_articles, load_images
from .graph import KnowledgeGraph

DEFAULT_STORE = os.path.join("store", "kg.db")


def _config(args) -> Config:
    cfg = Config.default()
    if getattr(args, "extractor", None):
        cfg.extractor = args.extractor
    if getattr(args, "embedder", None):
        cfg.embedder = args.embedder
    return cfg


def _open(args) -> KnowledgeGraph:
    return KnowledgeGraph.open(args.store, _config(args))


def cmd_ingest(args):
    if args.reset and os.path.exists(args.store):
        os.remove(args.store)
    g = _open(args)
    items = []
    if not args.no_text:
        items += load_articles(limit=args.n_text)
    if not args.no_images:
        items += load_images(limit=args.n_image)
    print(f"ingesting {len(items)} items into {args.store} ...")
    report = g.ingest(items)
    print(report)
    if report.notes:
        print("notes:", *report.notes[:5], sep="\n  ")
    g.save()
    print("saved.", json.dumps(g.stats(), indent=2))


def cmd_communities(args):
    g = _open(args)
    n = g.build_communities()
    g.save()
    print(f"built {n} communities; saved.")


def cmd_query(args):
    g = _open(args)
    res = g.query(args.text, mode=args.mode, k=args.k)
    if isinstance(res, list):  # community / global
        print(f"query: {args.text!r}  mode=community (global/breadth)")
        for c in res:
            print(f"  [{c['score']:.4f}] {c['community']} (size {c['size']}): {c['summary']}")
        return
    print(g.explain(res, max_objects=args.k))


def cmd_stats(args):
    g = _open(args)
    print(json.dumps(g.stats(), indent=2))


def cmd_inspect(args):
    g = _open(args)
    n = g.store.get_node(args.node_id)
    if not n:
        print(f"no such node: {args.node_id}")
        return
    print(json.dumps({"id": n.id, "type": n.ntype.value, "name": n.name,
                      "valid": n.valid, "tags": n.tags[:15],
                      "doc_frequency": n.doc_frequency,
                      "description": n.description,
                      "summary": n.summary}, indent=2, ensure_ascii=False))
    print("neighbours:")
    out = {id(d) for _n, d in g.store.neighbors(args.node_id, direction="out")}
    for nbr, data in list(g.store.neighbors(args.node_id))[:25]:
        nbr_node = g.store.get_node(nbr)
        name = nbr_node.name if nbr_node else "?"
        # rev 4: one relationship label per (parallel) edge; legacy class as fallback
        rn = g.store.get_node(data["rel_tag"]) if data.get("rel_tag") else None
        if rn:
            rel = f" [{rn.name}]"
        elif data.get("relation"):
            rel = f"/{data['relation']}"
        else:
            rel = ""
        arrow = "->" if id(data) in out else "<-"   # honour the real edge direction
        print(f"  {data.get('etype','?')}{rel} {arrow} {nbr} ({name})  "
              f"conf={data.get('confidence', 0.0):.2f} prov={data.get('provenance','?')}")


def cmd_viz(args):
    from .viz import graph_payload, query_trace, render_html
    g = _open(args)
    graph = graph_payload(g.store)
    trace = None
    if args.query:
        trace = query_trace(g, args.query, mode=args.mode)
    html = render_html(graph, trace=trace, server=False)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}  ({graph['stats']['by_node_type'].get('object', 0)} objects). "
          f"Open it in a browser.")
    if not args.query:
        print("tip: `python -m kg serve` for live, typed queries with traversal animation.")


def cmd_serve(args):
    from .serve import serve
    serve(args.store, port=args.port, config=_config(args))


def cmd_eval(args):
    from .evaluate import (cross_article_questions, evaluate, load_questions,
                           single_article_questions)
    g = _open(args)
    if args.questions:
        questions = load_questions(args.questions)
    else:
        questions = (single_article_questions(g, limit=args.single)
                     + cross_article_questions(g, limit=args.cross))
    print(f"evaluating {len(questions)} questions "
          f"(modes={args.modes}, k={args.k}) ...")
    modes = tuple(m.strip() for m in args.modes.split(","))
    scores = evaluate(g, questions, modes=modes, k=args.k)
    print("\n=== results ===")
    for s in scores:
        print(s)
        for kind, v in s.per_kind.items():
            print(f"            {kind:8s} recall@k={v['recall_at_k']:.3f} "
                  f"mrr={v['mrr']:.3f} (n={v['n']})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m kg")
    p.add_argument("--store", default=DEFAULT_STORE, help="path to the SQLite graph store")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="build/extend the graph from dataset/")
    pi.add_argument("--n-text", type=int, default=None)
    pi.add_argument("--n-image", type=int, default=None)
    pi.add_argument("--no-text", action="store_true")
    pi.add_argument("--no-images", action="store_true")
    pi.add_argument("--extractor", choices=["auto", "haiku", "heuristic"], default="auto")
    pi.add_argument("--embedder", choices=["auto", "st", "hashing"], default="auto")
    pi.add_argument("--reset", action="store_true", help="delete the store first")
    pi.set_defaults(func=cmd_ingest)

    pc = sub.add_parser("communities", help="detect communities + summaries")
    pc.set_defaults(func=cmd_communities)

    pq = sub.add_parser("query", help="retrieve for a question")
    pq.add_argument("text")
    pq.add_argument("--mode", default="auto",
                    choices=["auto", "ppr", "bfs", "vector", "community"])
    pq.add_argument("--k", type=int, default=8)
    pq.set_defaults(func=cmd_query)

    ps = sub.add_parser("stats", help="node/edge counts")
    ps.set_defaults(func=cmd_stats)

    pn = sub.add_parser("inspect", help="dump a node + neighbours")
    pn.add_argument("node_id")
    pn.set_defaults(func=cmd_inspect)

    pv = sub.add_parser("viz", help="write a self-contained HTML graph viewer")
    pv.add_argument("--out", default="kg_viz.html")
    pv.add_argument("--query", default=None, help="embed a query's traversal trace")
    pv.add_argument("--mode", default="bfs", choices=["bfs", "ppr", "vector"])
    pv.set_defaults(func=cmd_viz)

    pse = sub.add_parser("serve", help="live graph viewer with typed queries")
    pse.add_argument("--port", type=int, default=8000)
    pse.set_defaults(func=cmd_serve)

    pe = sub.add_parser("eval", help="recall@k / MRR ablation")
    pe.add_argument("--k", type=int, default=8)
    pe.add_argument("--modes", default="ppr,bfs,vector")
    pe.add_argument("--single", type=int, default=40)
    pe.add_argument("--cross", type=int, default=40)
    pe.add_argument("--questions", default=None, help="JSONL of authored questions")
    pe.set_defaults(func=cmd_eval)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
