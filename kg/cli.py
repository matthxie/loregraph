"""Command-line interface:  python -m kg <command> [options]

    ingest         build the graph from dataset/ (or --synthetic for the Becky stream)
    extract-dump   dump per-item extractions for one extractor/model (no graph build)
    eval-canon     canonicalization gate (synonyms merge; antonyms/inverses must not)
    communities    detect communities + summaries (Path B / breadth queries)
    query          retrieve for a question (auto-routes local↔global; --as-of for time travel)
    ask            answer a question: PPR retrieves a context, ONE LLM call answers (§5)
    demo           ingest the synthetic Becky/Alex stream and show temporal evolution
    stats          print node/edge counts
    inspect        dump one node + its neighbours (fact windows for RELATED_TO)
    eval           recall@k / MRR ablation across retrieval modes
"""
from __future__ import annotations

import argparse
import json
import os

from .config import Config
from .corpus import load_articles, load_images, load_mixed
from .graph import KnowledgeGraph
from .models import EdgeType

DEFAULT_STORE = os.path.join("store", "kg.db")


def _config(args) -> Config:
    cfg = Config.default()
    if getattr(args, "extractor", None):
        cfg.extractor = args.extractor
    if getattr(args, "embedder", None):
        cfg.embedder = args.embedder
    if getattr(args, "model", None):          # override the LLM model (extractor + L3 + answerer)
        cfg.llm_model = args.model
        cfg.l3_model = args.model
        cfg.rag_model = args.model
    if getattr(args, "backend", None):        # answerer backend override (ask)
        cfg.rag_backend = args.backend
    if getattr(args, "l3", False):
        cfg.l3_enabled = True
    if getattr(args, "self", None):           # personal-web: --self NAME enables the anchor
        cfg.self_entity = True
        cfg.self_name = args.self
    elif getattr(args, "personal", False):    # demo --personal: enable with the default name
        cfg.self_entity = True
    return cfg


def _open(args) -> KnowledgeGraph:
    return KnowledgeGraph.open(args.store, _config(args))


def cmd_ingest(args):
    if args.reset and os.path.exists(args.store):
        os.remove(args.store)
    g = _open(args)
    if args.synthetic:
        from .extractors import ScriptedExtractor
        from .synthetic import becky_stream
        items, table = becky_stream()
        g.extractor = ScriptedExtractor(table)   # deterministic prose→facts for the demo
    elif args.mixed:
        items = load_mixed(limit=args.limit)
    else:
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


def cmd_extract_dump(args):
    from .extract_dump import extract_corpus, summarize, write_dump
    from .extractors import get_extractor
    cfg = _config(args)
    ext = get_extractor(cfg)
    items = []
    if not args.no_text and args.n_text != 0:
        items += load_articles(limit=args.n_text)
    if not args.no_images and args.n_image != 0:
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
    res = g.query(args.text, mode=args.mode, k=args.k, as_of=args.as_of)
    if isinstance(res, list):  # community / global
        print(f"query: {args.text!r}  mode=community (global/breadth)")
        for c in res:
            print(f"  [{c['score']:.4f}] {c['community']} (size {c['size']}): {c['summary']}")
        return
    print(g.explain(res, max_objects=args.k))


def cmd_ask(args):
    g = _open(args)
    ans = g.ask(args.text, backend=args.backend, k=args.k, as_of=args.as_of, model=args.model)
    print(f"ask: {args.text!r}  backend={ans.backend}  as_of={ans.as_of or 'now'}")
    print(f"\nanswer:\n{ans.answer}\n")
    if ans.citations:
        print("citations:")
        for cid in ans.citations:
            n = g.store.get_node(cid)
            print(f"  [{cid}] {n.name if n else '?'}")
    else:
        print("citations: (none)")
    if args.show_context:
        print("\ncontext episodes:", ", ".join(ans.context_episodes) or "(none)")
        print("facts in context:")
        for f in ans.facts or ["(none)"]:
            print(f"  {f}")
    if ans.dropped_citations:
        print(f"⚠ dropped {len(ans.dropped_citations)} uncontextual citation(s): "
              f"{', '.join(ans.dropped_citations)}")
    if ans.notes:
        print("notes:", *ans.notes, sep="\n  ")


def _print_self_facts(g):
    """Print the self anchor's currently-valid facts. Walks BOTH directions (a symmetric
    predicate like works_with is stored in one orientation) and dedupes."""
    from .models import SELF_ENTITY_ID
    from .store import fact_active
    print("\n=== self entity facts (current) ===")
    node = g.store.get_node(SELF_ENTITY_ID)
    if node is None:
        print("  (no self anchor — was --personal/--self enabled?)")
        return
    seen, any_fact = set(), False
    for direction in ("out", "in"):
        for nbr, d in g.store.neighbors(SELF_ENTITY_ID, etypes={EdgeType.RELATED_TO},
                                        direction=direction):
            if not fact_active(d, None):
                continue
            src, dst = (SELF_ENTITY_ID, nbr) if direction == "out" else (nbr, SELF_ENTITY_ID)
            fkey = (src, d.get("rel_tag"), dst)
            if fkey in seen:
                continue
            seen.add(fkey)
            rel = g.store.get_node(d.get("rel_tag"))
            sn, tn = g.store.get_node(src), g.store.get_node(dst)
            print(f"  {sn.name if sn else src} --{rel.name if rel else '?'}--> "
                  f"{tn.name if tn else dst}")
            any_fact = True
    if not any_fact:
        print("  (none)")


def cmd_demo(args):
    """Ingest the synthetic evolving stream and show the graph's temporal evolution."""
    from .extractors import ScriptedExtractor
    from .synthetic import becky_stream, personal_stream
    if os.path.exists(args.store):
        os.remove(args.store)
    g = _open(args)
    if args.personal:
        items, table = personal_stream()
        g.extractor = ScriptedExtractor(table)
        print(f"ingesting the personal-web first-person stream ({len(items)} episodes) ...")
        print(g.ingest(items))
        g.save()
        print("\n" + json.dumps(g.stats()["facts"], indent=2))
        _print_self_facts(g)
        return
    items, table = becky_stream()
    g.extractor = ScriptedExtractor(table)
    print(f"ingesting the synthetic Becky/Alex stream ({len(items)} episodes) ...")
    print(g.ingest(items))
    g.save()
    print("\n" + json.dumps(g.stats()["facts"], indent=2))

    def ask(q, as_of=None):
        ans = g.ask(q, backend="offline", as_of=as_of)
        tag = f"  (as of {as_of})" if as_of else ""
        print(f"\nQ: {q}{tag}")
        for f in ans.facts:
            print(f"   fact: {f}")
        print(f"   answer: {ans.answer.splitlines()[0] if ans.answer else ''}")

    print("\n=== current view ===")
    ask("Where does Becky live and who does she work with?")
    print("\n=== point-in-time (as-of) ===")
    ask("Where does Becky live and who does she work with?", as_of="2022")


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
                      "valid": n.valid, "created_at": n.created_at,
                      "tags": n.tags[:15], "doc_frequency": n.doc_frequency,
                      "functional": n.functional, "symmetric": n.symmetric,
                      "description": n.description, "summary": n.summary},
                     indent=2, ensure_ascii=False))
    print("neighbours:")
    out = {id(d) for _n, d in g.store.neighbors(args.node_id, direction="out")}
    for nbr, data in list(g.store.neighbors(args.node_id))[:25]:
        nbr_node = g.store.get_node(nbr)
        name = nbr_node.name if nbr_node else "?"
        rn = g.store.get_node(data["rel_tag"]) if data.get("rel_tag") else None
        rel = f" [{rn.name}]" if rn else ""
        arrow = "->" if id(data) in out else "<-"
        win = ""
        if data.get("etype") == EdgeType.RELATED_TO.value:
            v, iv = data.get("valid_at", ""), data.get("invalid_at", "")
            win = f"  valid[{v[:10] or '?'}..{iv[:10] or '∞'}] {data.get('belief', '')}"
        print(f"  {data.get('etype','?')}{rel} {arrow} {nbr} ({name})  "
              f"conf={data.get('confidence', 0.0):.2f} prov={data.get('provenance','?')}{win}")


def cmd_viz(args):
    from .viz import graph_payload, query_trace, render_html
    g = _open(args)
    graph = graph_payload(g.store)
    trace = query_trace(g, args.query, mode=args.mode) if args.query else None
    html = render_html(graph, trace=trace, server=False)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}  ({graph['stats']['by_node_type'].get('episode', 0)} episodes). "
          f"Open it in a browser.")
    if not args.query:
        print("tip: `python -m kg serve` for live, typed queries with traversal animation.")


def cmd_serve(args):
    from .serve import serve
    serve(args.store, port=args.port, config=_config(args))


def cmd_testrun(args):
    from .testrun import run_testrun, summarize
    cfg = _config(args)
    # keep the test-run's chunked corpus out of the main graph store unless overridden
    store_path = (args.store if args.store != DEFAULT_STORE
                  else os.path.join("store", "testrun.db"))
    run = run_testrun(
        store_path=store_path, limit=args.limit, n_queries=args.queries,
        backend=args.backend, k=args.k, judge=not args.no_judge,
        communities=not args.no_communities, label=args.label, out_dir=args.out,
        config=cfg, progress=print)
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

    pi = sub.add_parser("ingest", help="build/extend the graph from dataset/ (or --synthetic)")
    pi.add_argument("--n-text", type=int, default=None)
    pi.add_argument("--n-image", type=int, default=None)
    pi.add_argument("--no-text", action="store_true")
    pi.add_argument("--no-images", action="store_true")
    pi.add_argument("--mixed", action="store_true",
                    help="ingest the per-paragraph temporal stream from dataset/mixed/")
    pi.add_argument("--synthetic", action="store_true",
                    help="ingest the synthetic evolving Becky/Alex stream (deterministic facts)")
    pi.add_argument("--limit", type=int, default=None, help="cap items when using --mixed")
    pi.add_argument("--extractor", choices=["auto", "haiku", "heuristic"], default="auto")
    pi.add_argument("--embedder", choices=["auto", "st", "hashing"], default="auto")
    pi.add_argument("--model", default=None, help="override the LLM extractor model id")
    pi.add_argument("--l3", action="store_true", help="enable the L3 canonicalization tie-breaker")
    pi.add_argument("--self", default=None, metavar="NAME",
                    help="personal-web: resolve first-person refs (i/me/my) to one stable "
                         "self anchor with this display name")
    pi.add_argument("--reset", action="store_true", help="delete the store first")
    pi.set_defaults(func=cmd_ingest)

    pd = sub.add_parser("extract-dump",
                        help="dump per-item extractions for an extractor/model (no graph build)")
    pd.add_argument("--extractor", choices=["auto", "haiku", "heuristic"], default="auto")
    pd.add_argument("--model", default=None, help="LLM model id to extract with")
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
    pg.add_argument("--l3", action="store_true", help="exercise the L3 LLM tie-breaker")
    pg.set_defaults(func=cmd_eval_canon)

    pc = sub.add_parser("communities", help="detect communities + summaries")
    pc.set_defaults(func=cmd_communities)

    pq = sub.add_parser("query", help="retrieve for a question")
    pq.add_argument("text")
    pq.add_argument("--mode", default="auto",
                    choices=["auto", "ppr", "bfs", "vector", "community"])
    pq.add_argument("--k", type=int, default=8)
    pq.add_argument("--as-of", default=None, dest="as_of",
                    help="retrieve the world as of this ISO date/year (point-in-time)")
    pq.set_defaults(func=cmd_query)

    pa = sub.add_parser("ask", help="answer a question: PPR retrieves a context, one LLM "
                                    "call answers (the LLM does not traverse)")
    pa.add_argument("text")
    pa.add_argument("--backend", choices=["auto", "claude", "offline"], default="auto",
                    help="auto = Claude if a key is present, else the offline answerer")
    pa.add_argument("--model", default=None, help="override the answerer LLM model id")
    pa.add_argument("--k", type=int, default=8, help="episodes retrieved for the context")
    pa.add_argument("--as-of", default=None, dest="as_of",
                    help="answer as of this ISO date/year (point-in-time retrieval)")
    pa.add_argument("--show-context", action="store_true",
                    help="print the context episodes + facts the answer was grounded in")
    pa.set_defaults(func=cmd_ask)

    pde = sub.add_parser("demo", help="ingest the synthetic Becky/Alex stream and show "
                                      "temporal evolution (current view vs as-of)")
    pde.add_argument("--embedder", choices=["auto", "st", "hashing"], default="auto")
    pde.add_argument("--personal", action="store_true",
                     help="use the first-person personal-web stream (self anchor) instead "
                          "of the Becky/Alex stream")
    pde.set_defaults(func=cmd_demo)

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
                    help="cap the number of mixed/temporal documents (default: all)")
    pt.add_argument("--queries", type=int, default=None,
                    help="cap the number of eval questions (default: all)")
    pt.add_argument("--backend", choices=["auto", "claude", "offline"], default=None,
                    help="answerer backend for the query half (auto = live if a key is set)")
    pt.add_argument("--extractor", choices=["auto", "haiku", "heuristic"], default="auto")
    pt.add_argument("--embedder", choices=["auto", "st", "hashing"], default="auto")
    pt.add_argument("--model", default=None, help="override the LLM model id")
    pt.add_argument("--k", type=int, default=8, help="episodes per query / recall@k")
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
