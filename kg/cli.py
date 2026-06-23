"""Command-line interface:  python -m kg <command> [options]

    ingest         build the graph from dataset/ (or re-ingest; cache skips unchanged)
    extract-dump   dump per-item extractions for one extractor/model (no graph build)
    eval-canon     canonicalization gate (synonyms merge; antonyms/inverses must not)
    communities    detect communities + summaries (Path B / breadth queries)
    query          retrieve for a question (auto-routes local↔global)
    ask            answer a question with an LLM that traverses the graph via tools (§5)
    stats          print node/edge counts
    inspect        dump one node + its neighbours
    eval           recall@k / MRR ablation across retrieval modes
"""
from __future__ import annotations

import argparse
import json
import os

from .config import Config
from .corpus import load_articles, load_images, load_mixed
from .graph import KnowledgeGraph

DEFAULT_STORE = os.path.join("store", "kg.db")


def _config(args) -> Config:
    cfg = Config.default()
    if getattr(args, "extractor", None):
        cfg.extractor = args.extractor
    if getattr(args, "embedder", None):
        cfg.embedder = args.embedder
    if getattr(args, "model", None):          # override the LLM model (extractor + L3 + agent)
        cfg.llm_model = args.model
        cfg.l3_model = args.model
        cfg.agent_model = args.model
    if getattr(args, "backend", None):        # agent backend override (ask)
        cfg.agent_backend = args.backend
    if getattr(args, "l3", False):            # enable the L3 canonicalization tie-breaker
        cfg.l3_enabled = True
    return cfg


def _open(args) -> KnowledgeGraph:
    return KnowledgeGraph.open(args.store, _config(args))


def cmd_ingest(args):
    if args.reset and os.path.exists(args.store):
        os.remove(args.store)
    g = _open(args)
    items = []
    if args.mixed:
        # per-paragraph temporal stream: each item carries its own created_at
        items = load_mixed(limit=args.limit)
    else:
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


def cmd_extract_dump(args):
    from .extract_dump import extract_corpus, summarize, write_dump
    from .extractors import get_extractor
    cfg = _config(args)
    ext = get_extractor(cfg)
    items = []
    if not args.no_text and args.n_text != 0:          # n==0 means "none of this modality"
        items += load_articles(limit=args.n_text)
    if not args.no_images and args.n_image != 0:       # (the loader treats limit=0 as "all")
        items += load_images(limit=args.n_image)
    label = args.label or (cfg.llm_model if ext.name == "haiku" else ext.name)
    print(f"extracting {len(items)} items  (extractor={ext.name}, model={cfg.llm_model}, "
          f"label={label!r}) ...")
    records, errors = extract_corpus(ext, items, cfg)
    summary = summarize(records, label)
    write_dump(records, summary, args.out)
    if errors:
        print(f"  ⚠ {len(errors)} extraction error(s); first: {errors[0]}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}  (+ {args.out}.summary.json)")


def cmd_eval_canon(args):
    from .eval_canon import run_gate
    report = run_gate(_config(args))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nGATE:", "PASS ✅" if report["gate_pass"] else "FAIL ❌",
          "(no antonym/inverse or distinct-sense false-merges)")


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


def cmd_ask(args):
    g = _open(args)
    ans = g.ask(args.text, backend=args.backend, k=args.k, max_steps=args.max_steps)
    print(f"ask: {args.text!r}  backend={ans.backend}  steps={ans.steps}  "
          f"stopped={ans.stopped}")
    print(f"\nanswer:\n{ans.answer}\n")
    if ans.citations:
        print("citations:")
        for cid in ans.citations:
            n = g.store.get_node(cid)
            print(f"  [{cid}] {n.name if n else '?'}")
    else:
        print("citations: (none)")
    if ans.dropped_citations:
        print(f"⚠ dropped {len(ans.dropped_citations)} unvalidated citation(s): "
              f"{', '.join(ans.dropped_citations)}")
    if args.show_trace:
        print("\ntrace:")
        for s in ans.trace:
            inp = {kk: vv for kk, vv in s["input"].items() if kk != "etypes" or vv}
            print(f"  {s['step'] + 1} {s['tool']:16s} {json.dumps(inp, ensure_ascii=False)}"
                  f"  -> {s['result_summary']}")
    if args.trace_out:
        with open(args.trace_out, "w", encoding="utf-8") as f:
            json.dump({"query": ans.query, "backend": ans.backend, "answer": ans.answer,
                       "citations": ans.citations, "trace": ans.trace}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nwrote trace -> {args.trace_out}")
    if args.viz:
        from .viz import agent_trace_payload, graph_payload, render_html
        html = render_html(graph_payload(g.store),
                           trace=agent_trace_payload(ans, g.store), server=False)
        with open(args.viz, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"wrote viewer -> {args.viz}  (open it in a browser)")


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


def cmd_testrun(args):
    from .testrun import run_testrun, summarize
    cfg = _config(args)
    # keep the dashboard's corpus out of the main graph store unless overridden
    store_path = (args.store if args.store != DEFAULT_STORE
                  else os.path.join("store", "testrun.db"))
    run = run_testrun(
        store_path=store_path, limit=args.limit, n_queries=args.queries,
        backend=args.backend, k=args.k, max_steps=args.max_steps,
        judge=not args.no_judge, communities=not args.no_communities,
        label=args.label, out_dir=args.out, config=cfg, progress=print)
    print("\n" + summarize(run))
    print(f"\nview it:  python -m kg dashboard --out {args.out}"
          f"   (or open {args.out}/{run['run_id']}/dashboard.html)")


def cmd_dashboard(args):
    from .dashboard import serve
    serve(out_dir=args.out, port=args.port)


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
    pi.add_argument("--mixed", action="store_true",
                    help="ingest the per-paragraph temporal stream from dataset/mixed/ "
                         "(each item carries its own created_at)")
    pi.add_argument("--limit", type=int, default=None,
                    help="cap the number of items when using --mixed")
    pi.add_argument("--extractor", choices=["auto", "haiku", "heuristic"], default="auto")
    pi.add_argument("--embedder", choices=["auto", "st", "hashing"], default="auto")
    pi.add_argument("--model", default=None,
                    help="override the LLM extractor model id (e.g. claude-sonnet-4-6)")
    pi.add_argument("--l3", action="store_true",
                    help="enable the L3 LLM canonicalization tie-breaker (needs a key; off by default)")
    pi.add_argument("--reset", action="store_true", help="delete the store first")
    pi.set_defaults(func=cmd_ingest)

    pd = sub.add_parser("extract-dump",
                        help="dump per-item extractions for an extractor/model (no graph build)")
    pd.add_argument("--extractor", choices=["auto", "haiku", "heuristic"], default="auto")
    pd.add_argument("--model", default=None,
                    help="LLM model id to extract with, e.g. claude-haiku-4-5-20251001 / claude-sonnet-4-6")
    pd.add_argument("--n-text", type=int, default=20)
    pd.add_argument("--n-image", type=int, default=0)
    pd.add_argument("--no-text", action="store_true")
    pd.add_argument("--no-images", action="store_true")
    pd.add_argument("--label", default=None, help="name for this mode in the summary")
    pd.add_argument("--out", default=os.path.join("store", "extract_dump.jsonl"))
    pd.set_defaults(func=cmd_extract_dump)

    pg = sub.add_parser("eval-canon",
                        help="canonicalization gate: synonyms merge, antonyms/inverses must NOT")
    pg.add_argument("--extractor", choices=["auto", "haiku", "heuristic"], default="heuristic")
    pg.add_argument("--embedder", choices=["auto", "st", "hashing"], default="auto")
    pg.add_argument("--model", default=None, help="L3 adjudicator model (with --l3)")
    pg.add_argument("--l3", action="store_true",
                    help="exercise the L3 LLM tie-breaker during the gate (needs a key)")
    pg.set_defaults(func=cmd_eval_canon)

    pc = sub.add_parser("communities", help="detect communities + summaries")
    pc.set_defaults(func=cmd_communities)

    pq = sub.add_parser("query", help="retrieve for a question")
    pq.add_argument("text")
    pq.add_argument("--mode", default="auto",
                    choices=["auto", "ppr", "bfs", "vector", "community"])
    pq.add_argument("--k", type=int, default=8)
    pq.set_defaults(func=cmd_query)

    pa = sub.add_parser("ask", help="answer a question with an LLM that traverses the graph "
                                    "via tools (§5 agentic retrieval)")
    pa.add_argument("text")
    pa.add_argument("--backend", choices=["auto", "claude", "offline"], default="auto",
                    help="auto = Claude if a key is present, else the offline deterministic agent")
    pa.add_argument("--model", default=None, help="override the agent LLM model id")
    pa.add_argument("--k", type=int, default=8, help="default objects per search tool")
    pa.add_argument("--max-steps", type=int, default=None, help="tool-call budget (default 8)")
    pa.add_argument("--show-trace", action="store_true", help="print the per-step tool calls")
    pa.add_argument("--trace-out", default=None, help="write the agent trace as JSON")
    pa.add_argument("--viz", default=None, help="write the HTML viewer with this agent's trace")
    pa.set_defaults(func=cmd_ask)

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

    pt = sub.add_parser("testrun", help="run the input+query test on the temporal dataset "
                                        "and write a dashboard run (cost/tokens/accuracy)")
    pt.add_argument("--limit", type=int, default=None,
                    help="cap the number of mixed/temporal documents (default: all 1343)")
    pt.add_argument("--queries", type=int, default=None,
                    help="cap the number of eval questions (default: all 68)")
    pt.add_argument("--backend", choices=["auto", "claude", "offline"], default=None,
                    help="agent backend for the query half (auto = live if a key is set)")
    pt.add_argument("--extractor", choices=["auto", "haiku", "heuristic"], default="auto")
    pt.add_argument("--embedder", choices=["auto", "st", "hashing"], default="auto")
    pt.add_argument("--model", default=None, help="override the LLM model id")
    pt.add_argument("--k", type=int, default=8, help="objects per search / recall@k")
    pt.add_argument("--max-steps", type=int, default=None, help="agent tool-call budget")
    pt.add_argument("--l3", action="store_true", help="enable the L3 canonicalization tie-breaker")
    pt.add_argument("--no-judge", action="store_true",
                    help="skip the LLM response-accuracy judge (deterministic proxy only)")
    pt.add_argument("--no-communities", action="store_true",
                    help="skip community detection after ingest (faster)")
    pt.add_argument("--label", default=None, help="run id / label (default: timestamp)")
    pt.add_argument("--out", default="runs", help="directory of dashboard runs")
    pt.set_defaults(func=cmd_testrun)

    pdash = sub.add_parser("dashboard", help="serve the test-run dashboard (run index + "
                                             "Input/Query drill-down)")
    pdash.add_argument("--out", default="runs", help="directory of dashboard runs")
    pdash.add_argument("--port", type=int, default=8050)
    pdash.set_defaults(func=cmd_dashboard)

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
