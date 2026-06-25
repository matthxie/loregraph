"""Test-run harness for the graph dashboard.

One invocation = one *test run*, split into the two halves the dashboard toggles
between:

  INPUT  — ingest a LongMemEval tier's chat sessions (`dataset/longmemeval/<tier>/`,
           default `sample`) **one episode at a time** (`KnowledgeGraph.ingest_object`),
           snapshotting after each: node/edge
           counts by type, tokens + USD cost (drained from the extractor meter),
           avg tags-per-object, vocabulary growth, and per-document tag
           `doc_frequency` — the "temporal tag change" signal. The final object
           graph (with a layout + build order) is captured so the dashboard can
           animate the structure forming.

  QUERY  — run every `dataset/longmemeval/<tier>/questions.jsonl` question through
           `KnowledgeGraph.ask` (PPR retrieves a context, ONE LLM call answers — the
           LLM does not traverse), capturing the retrieval subgraph
           (`viz.rag_trace_payload`), the nodes PPR touched, tokens + cost, retrieval
           accuracy (recall@k / MRR / citation-grounding) and an optional LLM-judge score.

The result is written to `runs/<run_id>/run.json` (consumed by the dashboard
server) plus a self-contained `runs/<run_id>/dashboard.html` static export, and
`runs/<run_id>` is registered in `runs/index.json`.

Everything is **offline-safe**: with no API key the heuristic extractor + offline
agent run the same flow deterministically, and the cost/token panels read $0 / 0
(the meters stay empty). Triggered by the `test-graph` skill / `python -m kg
testrun`.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

from .config import Config
from .corpus import load_longmemeval
from .evaluate import _mrr, _recall_at_k
from .graph import KnowledgeGraph
from .metering import UsageMeter, totals_of
from .models import EdgeType, NodeType
from .viz import rag_trace_payload

DEFAULT_TIER = "micro"   # tiny committed LIVE smoke set (3 instances); see scripts/build_micro.py
DEFAULT_OUT = "runs"


def _tier_questions_path(tier: str) -> str:
    return os.path.join("dataset", "longmemeval", tier, "questions.jsonl")


QUESTIONS_PATH = _tier_questions_path(DEFAULT_TIER)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Graph-state probes (all read-only, cheap)
# --------------------------------------------------------------------------- #
def _vocab(stats: dict) -> dict:
    bt = stats["by_node_type"]
    # "objects" = episodes (the raw-entry count); key kept for the dashboard's chart labels
    return {"objects": bt.get("episode", 0), "tags": bt.get("tag", 0),
            "entities": bt.get("entity", 0), "relations": bt.get("relation", 0),
            "communities": bt.get("community", 0)}


def _avg_tags_per_object(stats: dict) -> float:
    n_ep = stats["by_node_type"].get("episode", 0)
    tagged = stats["by_edge_type"].get("TAGGED_AS", 0)
    return round(tagged / n_ep, 3) if n_ep else 0.0


def _episode_entities(store, ep_id: str):
    """The entities an episode references, via its mention star
    (episode ← MENTIONED_IN ← mention → RESOLVES_TO → entity)."""
    out, seen = [], set()
    for mid, _d in store.neighbors(ep_id, etypes={EdgeType.MENTIONED_IN}, direction="in"):
        for eid, _d2 in store.neighbors(mid, etypes={EdgeType.RESOLVES_TO}, direction="out"):
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
    return out


def _doc_footprint(store, obj_id: str, max_rel: int = 8) -> dict:
    """What this episode contributes/touches: its canonical tags, the entities it mentions
    (via the mention star), and the directed relation labels among those entities."""
    node = store.get_node(obj_id)
    tags = list(node.tags)[:14] if node else []
    ents, rel_tags = [], []
    for eid in _episode_entities(store, obj_id):
        en = store.get_node(eid)
        if not en:
            continue
        ents.append(en.name)
        for _t, rd in store.neighbors(eid, etypes={EdgeType.RELATED_TO}, direction="out"):
            rid = rd.get("rel_tag")
            rn = store.get_node(rid) if rid else None
            if rn and rn.name not in rel_tags:
                rel_tags.append(rn.name)
                if len(rel_tags) >= max_rel:
                    break
    return {"tags": tags, "entities": ents[:16], "rel_tags": rel_tags}


def _tag_df(store, obj_id: str) -> list[dict]:
    """doc_frequency of each of this object's tags, AFTER ingesting it — the temporal
    drift signal. The dashboard forward-fills these to chart any tag's df over time."""
    out = []
    for nbr, _d in store.neighbors(obj_id, etypes={EdgeType.TAGGED_AS}, direction="out"):
        tn = store.get_node(nbr)
        if tn:
            out.append({"name": tn.name, "df": tn.doc_frequency})
    return out


def _rel_pair_stats(store) -> dict:
    """Relation-tags per entity PAIR (the meaningful aggregate — each RELATED_TO edge
    carries exactly one rel_tag by rev-4 design, so per-edge is trivially ~1)."""
    pairs: dict[tuple[str, str], set] = {}
    for u, v, d in store.all_edges():
        if d.get("etype") == EdgeType.RELATED_TO.value and d.get("rel_tag"):
            pairs.setdefault((u, v), set()).add(d["rel_tag"])
    n_pairs = len(pairs)
    n_edges = sum(len(s) for s in pairs.values())
    return {"pairs": n_pairs, "rel_edges": n_edges,
            "avg_rel_tags_per_pair": round(n_edges / n_pairs, 3) if n_pairs else 0.0}


# The collapsed Episode→Entity link drawn in place of the (mention → episode/entity) star,
# so the dashboard graph stays Episode/Entity/Tag-only. Labelled "MENTIONS" so the viewer's
# existing edge styling/legend applies unchanged.
_MENTIONS_ETYPE = "MENTIONS"
_DIRECTED_ETYPES = {EdgeType.TAGGED_AS.value, _MENTIONS_ETYPE,
                    EdgeType.RELATED_TO.value, EdgeType.HYPERLINKS_TO.value}


def _full_graph(store) -> dict:
    """Obsidian-style graph payload for the Input view's force graph: EPISODE nodes are the
    *raw entries* (green), TAG/ENTITY nodes are *created* (blue), plus the edges between
    them. The immutable mention layer is COLLAPSED into a direct Episode→Entity 'MENTIONS'
    edge (the mention nodes themselves aren't drawn), keeping the graph to things that
    connect. Relation-tag and community nodes are not drawn (relations ride on RELATED_TO
    edges as labels; communities are out of scope).

    Each node carries `appear` — the ingestion step at which it first shows up — so the
    dashboard can *grow* the graph in generation order, `deg`/`indeg` (the client sizes
    nodes by in-degree), plus `meta` for the click-to-inspect panel."""
    NT, ET = NodeType, EdgeType
    objs = sorted(store.nodes_of_type(NT.EPISODE), key=lambda n: (n.created_at or "", n.id))
    build_order = [n.id for n in objs]
    appear: dict[str, int] = {oid: i for i, oid in enumerate(build_order)}
    drawn = (NT.EPISODE, NT.ENTITY, NT.TAG)

    deg: dict[str, int] = {}
    indeg: dict[str, int] = {}  # edges pointing AT a node (sizes the node in the view)
    edges, seen = [], set()

    def add_edge(u, v, et, rel, directed):
        key = (u, v, et, rel)
        if key in seen:
            return
        seen.add(key)
        edges.append({"s": u, "t": v, "etype": et, "directed": directed, "rel": rel})
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
        if directed:                          # only directed edges "point at" a node;
            indeg[v] = indeg.get(v, 0) + 1    # symmetric shared/similar edges don't.

    # real Episode↔Episode / Episode→Tag / Entity→Entity edges (skip mention/community)
    skip = {ET.IN_COMMUNITY.value, ET.MENTIONED_IN.value, ET.RESOLVES_TO.value}
    for u, v, d in store.all_edges():
        if not d.get("valid", True) or d.get("etype") in skip:
            continue
        un, vn = store.get_node(u), store.get_node(v)
        if not un or not vn or un.ntype not in drawn or vn.ntype not in drawn:
            continue
        et = d.get("etype")
        rel = ""
        if et == ET.RELATED_TO.value and d.get("rel_tag"):
            rn = store.get_node(d["rel_tag"])
            rel = rn.name if rn else ""
        add_edge(u, v, et, rel, et in _DIRECTED_ETYPES)

    # collapsed Episode→Entity "MENTIONS" edges (the mention star, drawn as one hop)
    for ep in build_order:
        for eid in _episode_entities(store, ep):
            if store.get_node(eid):
                add_edge(ep, eid, _MENTIONS_ETYPE, "", True)

    # propagate appear to created nodes: the step of the earliest episode they touch
    for i, ep in enumerate(build_order):
        for eid in _episode_entities(store, ep):
            appear.setdefault(eid, i)
        for nbr, d in store.neighbors(ep, etypes={ET.TAGGED_AS}, direction="out"):
            appear.setdefault(nbr, i)

    def meta(n) -> dict:
        if n.ntype == NT.EPISODE:
            ents = [store.get_node(e).name for e in _episode_entities(store, n.id)
                    if store.get_node(e)]
            return {"modality": n.modality.value if n.modality else "text",
                    "created_at": n.created_at, "source_ref": n.source_ref,
                    "n_tags": len(n.tags), "tags": list(n.tags)[:20], "entities": ents[:20],
                    # full episode text (the click panel scrolls) — not truncated
                    "snippet": (n.raw_text or n.description or "")}
        if n.ntype == NT.ENTITY:
            return {"entity_type": n.entity_type.value if n.entity_type else "other",
                    "df": n.doc_frequency}
        return {"df": n.doc_frequency, "aliases": list(n.aliases or [])[:10]}  # tag

    nodes = []
    for n in store.nodes.values():
        if n.ntype not in drawn or not n.valid:
            continue
        nodes.append({"id": n.id, "type": n.ntype.value, "raw": n.ntype == NT.EPISODE,
                      "label": (n.name or n.id), "deg": deg.get(n.id, 0),
                      "indeg": indeg.get(n.id, 0),
                      "appear": appear.get(n.id, len(build_order) - 1), "meta": meta(n)})

    return {"nodes": nodes, "edges": edges, "build_order": build_order,
            "stats": store.stats(), "kind": "full"}


# --------------------------------------------------------------------------- #
# Response-accuracy scoring (deterministic proxy + optional LLM judge)
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")


def _article(oid: str) -> str:
    """Collapse a mixed-stream chunk id to its source article/image id AND strip the node
    prefix, so retrieval is scored at the article level and is robust across the
    object→episode rename: `ep_wiki_062#p003` -> `wiki_062`, and a gold id authored under
    the old prefix (`obj_wiki_010`) -> `wiki_010`, so the two still match. A question is
    satisfied when retrieval surfaces ANY chunk derived from the gold article."""
    base = oid.split("#", 1)[0]
    for prefix in ("ep_", "obj_"):
        if base.startswith(prefix):
            return base[len(prefix):]
    return base


def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _norm(s: str) -> str:
    return " ".join(_WORD.findall((s or "").lower()))


def _response_proxy(answer: str, expected) -> dict:
    """Key-free signals: does the short reference answer appear in the response, and
    what fraction of the reference's content tokens does the response contain."""
    expected = "" if expected is None else str(expected)   # some gold answers are ints
    answer = "" if answer is None else str(answer)
    if not expected:
        return {"contains": None, "token_recall": None}
    a_norm = _norm(answer)
    e_toks = [t for t in _WORD.findall(expected.lower()) if len(t) > 1]
    if not e_toks:
        return {"contains": None, "token_recall": None}
    a_set = set(_WORD.findall(a_norm))
    present = sum(1 for t in set(e_toks) if t in a_set)
    return {"contains": _norm(expected) in a_norm,
            "token_recall": round(present / len(set(e_toks)), 3)}


_JUDGE_SYS = (
    "You are a strict grader. Given a question, a reference answer, and a model's "
    "answer, decide whether the model's answer is factually correct and actually "
    "answers the question. Be lenient about phrasing, strict about facts. Call the "
    "grade tool exactly once.")

_JUDGE_TOOL = {
    "name": "grade",
    "description": "Grade the model answer against the reference.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {"type": "boolean",
                        "description": "true if the model answer is factually correct "
                        "and answers the question"},
            "score": {"type": "number",
                      "description": "0.0-1.0 quality: 1 fully correct, 0.5 partial, 0 wrong"},
            "reason": {"type": "string", "description": "one short sentence"},
        },
        "required": ["correct", "score"],
    },
}


def _judge(client, model: str, q: dict, answer: str, meter: UsageMeter) -> dict | None:
    prompt = (f"Question: {q['query']}\n"
              f"Reference answer: {q.get('answer', '')}\n"
              f"Rationale: {q.get('rationale', '')}\n\n"
              f"Model answer:\n{answer[:1500]}\n\n"
              "Grade the model answer.")
    try:
        msg = client.messages.create(
            model=model, max_tokens=300, temperature=0, system=_JUDGE_SYS,
            tools=[_JUDGE_TOOL], tool_choice={"type": "tool", "name": "grade"},
            messages=[{"role": "user", "content": prompt}])
        meter.record("judge", model, msg, label=q.get("id", ""))
        for b in msg.content:
            if getattr(b, "type", None) == "tool_use" and b.name == "grade":
                d = b.input or {}
                return {"correct": bool(d.get("correct")),
                        "score": round(float(d.get("score", 0.0)), 3),
                        "reason": str(d.get("reason", ""))[:300]}
    except Exception as e:  # noqa: BLE001 — judge is best-effort, never crash the run
        return {"error": f"{e!r}"}
    return None


# --------------------------------------------------------------------------- #
# Question loading
# --------------------------------------------------------------------------- #
def load_questions(path: str = QUESTIONS_PATH, limit: int | None = None) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def _build_judge_client(model: str):
    """A separate Anthropic client for the response-accuracy judge, or None offline."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:  # noqa: BLE001
        return None


def _score_query(q: dict, ans, kk: int, store, jclient, cfg: Config,
                 judge_meter: UsageMeter) -> dict:
    """Score one answered question into a dashboard query-record. Shared by the shared-graph
    `run_testrun` and the per-instance `run_per_instance`: `store` is whichever graph the
    answer was produced over (the one shared graph, or this instance's own graph), so the
    captured `subgraph` is always the graph the query actually ran against."""
    ranked = ans.object_ids
    gold = set(q.get("gold", []))
    # _article collapses obj_<id> (gold) and ep_<id> (ingested) to the same key
    gold_art = {_article(x) for x in gold}
    ranked_art = _dedup(_article(o) for o in ranked)
    topk_art = set(ranked_art[:kk])
    recall = _recall_at_k(ranked_art, gold_art, kk)
    mrr = _mrr(ranked_art, gold_art)
    rank = next((idx + 1 for idx, a in enumerate(ranked_art) if a in gold_art), None)
    cited_art = {_article(c) for c in ans.citations}
    grounding = (round(len(cited_art & gold_art) / len(gold_art), 3) if gold_art else 0.0)
    hit = bool(topk_art & gold_art)
    gold_marks = [{"id": x, "hit": _article(x) in topk_art} for x in sorted(gold)]
    proxy = _response_proxy(ans.answer, q.get("answer", ""))
    jres = _judge(jclient, cfg.l3_model, q, ans.answer, judge_meter) if jclient else None
    return {
        "id": q.get("id", ""), "query": q["query"], "kind": q.get("kind", ""),
        "difficulty": q.get("difficulty", ""), "gold": sorted(gold),
        "gold_marks": gold_marks,
        "answer_expected": q.get("answer", ""), "rationale": q.get("rationale", ""),
        "abstention": q.get("abstention", False),
        "answer": ans.answer, "citations": ans.citations,
        "dropped_citations": ans.dropped_citations, "object_ids": ranked,
        "recall_at_k": round(recall, 3), "mrr": round(mrr, 3),
        "hit": hit, "rank": rank,
        "citation_grounding": grounding,
        "response_contains": proxy["contains"], "response_token_recall": proxy["token_recall"],
        "judge": jres,
        "steps": ans.steps, "stopped": ans.stopped, "backend": ans.backend,
        "trace": ans.trace, "seeds": ans.seeds, "touched": ans.touched,
        "n_touched": len(ans.touched),
        "subgraph": rag_trace_payload(ans, store),
        "llm_calls": ans.usage.get("llm_calls", 0),
        "input_tokens": ans.usage.get("input_tokens", 0),
        "output_tokens": ans.usage.get("output_tokens", 0),
        "tokens": ans.usage.get("tokens", 0),
        "cost_usd": ans.usage.get("cost_usd", 0.0),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_testrun(*, store_path: str = os.path.join("store", "testrun.db"),
                tier: str = DEFAULT_TIER,
                limit: int | None = None, n_queries: int | None = None,
                questions_path: str | None = None,
                backend: str | None = None, k: int | None = None,
                max_steps: int | None = None, judge: bool = True,
                label: str | None = None, out_dir: str = DEFAULT_OUT,
                config: Config | None = None, agent_client=None,
                judge_client=None, communities: bool = True,
                progress=None) -> dict:
    """Run the full input+query test and write the dashboard artifact. Returns the run
    dict. `agent_client` / `judge_client` inject (possibly fake) Anthropic clients for
    deterministic tests; left None they auto-select live-vs-offline like the rest of kg."""
    log = progress or (lambda *_: None)
    cfg = config or Config.default()
    if backend:
        cfg.rag_backend = backend
    kk = k or cfg.top_k

    # fresh store every run so per-document deltas start from an empty graph
    if os.path.exists(store_path):
        os.remove(store_path)
    g = KnowledgeGraph.open(store_path, cfg)

    # ---------------------------------------------------------------- INPUT half
    # NB: this ingests ALL of the tier's sessions into ONE shared graph (a scale/structure
    # view of the dashboard). For correct LongMemEval accuracy use the per-instance protocol
    # (kg.corpus.iter_lme_instances) — see dataset/longmemeval/README.md §Consumption.
    items = load_longmemeval(tier, limit=limit)
    log(f"ingesting {len(items)} chat-session episodes one at a time ...")
    steps: list[dict] = []
    all_records = []
    prev_ids: set[str] = set(g.store.nodes.keys())
    prev_stats = g.store.stats()
    t0 = time.time()
    for i, item in enumerate(items):
        g.extractor.meter.drain()        # reset per-document attribution
        g.canon.meter.drain()
        rep = g.ingest_object(item)
        recs = g.extractor.meter.drain() + g.canon.meter.drain()
        all_records.extend(recs)
        after = g.store.stats()
        cur_ids = set(g.store.nodes.keys())
        added = cur_ids - prev_ids
        obj_id = next((nid for nid in added
                       if g.store.get_node(nid).ntype == NodeType.EPISODE), None)
        tok = totals_of(recs)
        delta = {t: after["by_node_type"].get(t, 0) - prev_stats["by_node_type"].get(t, 0)
                 for t in set(after["by_node_type"]) | set(prev_stats["by_node_type"])}
        step = {
            "i": i, "doc_id": item.id, "title": (item.title or item.id)[:80],
            "modality": item.modality, "created_at": item.created_at,
            "status": ("ingested" if rep.ingested else
                       "skipped" if rep.skipped else "failed"),
            "seconds": round(rep.seconds, 3),
            "nodes": after["nodes"], "edges": after["edges"],
            "by_node_type": after["by_node_type"], "by_edge_type": after["by_edge_type"],
            "vocab": _vocab(after), "node_delta": delta,
            "avg_tags_per_object": _avg_tags_per_object(after),
            "added_nodes": len(added),
            "llm_calls": tok["llm_calls"], "input_tokens": tok["input_tokens"],
            "output_tokens": tok["output_tokens"], "tokens": tok["tokens"],
            "cost_usd": tok["cost_usd"],
        }
        if obj_id:
            step["object_id"] = obj_id
            step["footprint"] = _doc_footprint(g.store, obj_id)
            step["tag_df"] = _tag_df(g.store, obj_id)
        steps.append(step)
        prev_ids, prev_stats = cur_ids, after
        if i % 25 == 0 or i == len(items) - 1:
            log(f"  ingest {i + 1}/{len(items)}  nodes={after['nodes']} "
                f"edges={after['edges']} cost=${totals_of(all_records)['cost_usd']:.4f}")
    ingest_seconds = round(time.time() - t0, 1)

    if communities:
        log("detecting communities ...")
        g.build_communities()
    g.save()

    log("assembling the object/entity/tag graph ...")
    graph = _full_graph(g.store)
    final = g.store.stats()
    ingest_tok = totals_of(all_records)
    ingest_totals = {
        "docs": sum(1 for s in steps if s["status"] == "ingested"),
        "items": len(items),
        "nodes": final["nodes"], "edges": final["edges"],
        "by_node_type": final["by_node_type"], "by_edge_type": final["by_edge_type"],
        "vocab": _vocab(final),
        "avg_tags_per_object": _avg_tags_per_object(final),
        "seconds": ingest_seconds,
        **_rel_pair_stats(g.store),
        **ingest_tok,
    }

    # ---------------------------------------------------------------- QUERY half
    questions_path = questions_path or _tier_questions_path(tier)
    questions = load_questions(questions_path, limit=n_queries)
    log(f"running {len(questions)} queries through the PPR→RAG ask() ...")
    judge_meter = UsageMeter()
    jclient = judge_client
    if judge and jclient is None and agent_client is None:
        jclient = _build_judge_client(cfg.l3_model)
    qrecords: list[dict] = []
    for q in questions:
        ans = g.ask(q["query"], backend=backend, k=kk, client=agent_client)
        rec = _score_query(q, ans, kk, g.store, jclient, cfg, judge_meter)
        qrecords.append(rec)
        log(f"  {rec['id'] or rec['query'][:30]:32s} recall@k={rec['recall_at_k']:.2f} "
            f"hit={'Y' if rec['hit'] else '.'} steps={rec['steps']}")

    query_totals = _query_totals(qrecords, judge_meter, kk)

    # ---------------------------------------------------------------- assemble
    run_id = label or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    backends = g.stats()["backends"]
    backends.setdefault("agent", backends.get("answerer", "offline"))  # dashboard alias
    run = {
        "run_id": run_id, "label": label or run_id, "created_at": _now(),
        "backends": backends,
        "models": {"extractor": cfg.llm_model, "agent": cfg.rag_model,
                   "l3_judge": cfg.l3_model, "embedder": cfg.embed_model},
        "dataset": {"input": f"longmemeval:{tier}", "n_input": len(items),
                    "queries": os.path.basename(questions_path), "n_queries": len(questions)},
        "config": {"k": kk, "agent_backend": cfg.rag_backend, "mode": "shared",
                   "answerer": "ppr-rag", "reflexion": cfg.reflexion,
                   "l3_enabled": cfg.l3_enabled, "communities": communities,
                   "match": "session-level (gold evidence sessions vs ingested episodes)"},
        "cost_usd": round(ingest_totals["cost_usd"] + query_totals["cost_usd"], 6),
        "tokens": ingest_totals["tokens"] + query_totals["tokens"],
        "ingest": {"totals": ingest_totals, "steps": steps, "graph": graph},
        "query": {"totals": query_totals, "queries": qrecords},
    }

    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False)
    from .dashboard import render_run_html
    with open(os.path.join(run_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(render_run_html(run, server=False))
    _update_index(out_dir, run)
    log(f"wrote {run_dir}/run.json + dashboard.html")
    return run


def _write_run(run: dict, out_dir: str, log) -> dict:
    """Serialize a run dict + its static dashboard, and register it in the index."""
    run_dir = os.path.join(out_dir, run["run_id"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False)
    from .dashboard import render_run_html
    with open(os.path.join(run_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(render_run_html(run, server=False))
    _update_index(out_dir, run)
    log(f"wrote {run_dir}/run.json + dashboard.html")
    return run


def run_per_instance(*, tier: str = DEFAULT_TIER,
                     store_path: str = os.path.join("store", "lme_instance.db"),
                     n_queries: int | None = None,
                     backend: str | None = None, k: int | None = None,
                     judge: bool = True, label: str | None = None,
                     out_dir: str = DEFAULT_OUT, config: Config | None = None,
                     agent_client=None, judge_client=None,
                     communities: bool = False, progress=None) -> dict:
    """Per-instance LongMemEval eval — the dataset's NATIVE protocol: each question is
    answered against a FRESH graph built from ONLY its own haystack, so the 500 first-person
    personas never share one memory (no cross-user entity collisions). Each instance *is* the
    "one big personal graph" prod scenario at one-user scale.

    Emits the same run.json the dashboard consumes:
      - QUERY view (unchanged): one record per question, each carrying ITS OWN instance graph
        as the retrieval subgraph — so every query is shown against the graph it ran on.
      - INPUT view: per-instance ingest steps with cumulative cost/tokens/graph-size across
        the run, a REPRESENTATIVE instance's graph for the structure visual, and aggregate
        totals. (There is no single unified graph — that's the whole point of per-instance.)
    """
    log = progress or (lambda *_: None)
    cfg = config or Config.default()
    if backend:
        cfg.rag_backend = backend
    kk = k or cfg.top_k

    from .corpus import iter_lme_instances
    instances = list(iter_lme_instances(tier, limit=n_queries))
    log(f"per-instance eval: {len(instances)} instances from longmemeval:{tier} "
        f"(a fresh graph each) ...")

    judge_meter = UsageMeter()
    jclient = judge_client
    if judge and jclient is None and agent_client is None:
        jclient = _build_judge_client(cfg.l3_model)

    qrecords: list[dict] = []
    steps: list[dict] = []
    agg_nodes = agg_edges = 0
    agg_tagged = agg_episodes = 0                          # pooled tags-per-object (not mean-of-means)
    cum_vocab = {"objects": 0, "tags": 0, "entities": 0, "relations": 0, "communities": 0}
    ingest_cost = 0.0
    ingest_tokens = ingest_llm = ingest_in = ingest_out = 0
    ingest_cr = ingest_cw = 0          # prompt-cache read/write rollup (caching signal)
    ingest_trunc = 0                   # # extract calls that hit max_tokens (silent payload cut)
    total_sessions = 0
    rep_graph: dict | None = None
    rep_nodes = -1
    backends_seen: dict | None = None
    t0 = time.time()

    for i, (q, sessions) in enumerate(instances):
        total_sessions += len(sessions)
        if os.path.exists(store_path):
            os.remove(store_path)
        g = KnowledgeGraph.open(store_path, cfg)         # fresh memory per instance
        g.extractor.meter.drain()
        g.canon.meter.drain()
        rep = g.ingest(sessions)                         # only THIS instance's haystack
        recs = g.extractor.meter.drain() + g.canon.meter.drain()
        tok = totals_of(recs)
        if communities:
            g.build_communities()
        stats = g.store.stats()
        agg_nodes += stats["nodes"]
        agg_edges += stats["edges"]
        agg_tagged += stats["by_edge_type"].get("TAGGED_AS", 0)
        agg_episodes += stats["by_node_type"].get("episode", 0)
        v = _vocab(stats)
        for key in cum_vocab:
            cum_vocab[key] += v.get(key, 0)
        ingest_cost += tok["cost_usd"]
        ingest_tokens += tok["tokens"]
        ingest_llm += tok["llm_calls"]
        ingest_in += tok["input_tokens"]
        ingest_out += tok["output_tokens"]
        ingest_cr += tok["cache_read"]
        ingest_cw += tok["cache_write"]
        ingest_trunc += tok["truncated"]
        atpo = _avg_tags_per_object(stats)

        ans = g.ask(q["query"], backend=backend, k=kk, client=agent_client)
        rec = _score_query(q, ans, kk, g.store, jclient, cfg, judge_meter)
        rec["n_sessions"] = len(sessions)
        qrecords.append(rec)

        steps.append({
            "i": i, "doc_id": q["id"], "title": q["query"][:80], "modality": "text",
            "created_at": q.get("question_date"), "status": "ingested",
            "seconds": round(rep.seconds, 3),
            "nodes": agg_nodes, "edges": agg_edges,            # cumulative across the run
            "by_node_type": stats["by_node_type"], "by_edge_type": stats["by_edge_type"],
            "vocab": dict(cum_vocab), "node_delta": {},
            "avg_tags_per_object": atpo, "added_nodes": stats["nodes"],
            "llm_calls": tok["llm_calls"], "input_tokens": tok["input_tokens"],
            "output_tokens": tok["output_tokens"], "tokens": tok["tokens"],
            "cost_usd": tok["cost_usd"],                       # per-instance (JS accumulates)
            "object_id": q["id"],
            "footprint": {"kind": q.get("kind", ""), "n_sessions": len(sessions),
                          "question": q["query"], "answer": q.get("answer", ""),
                          "tags": [], "entities": [], "rel_tags": []},
            "tag_df": [],
        })
        if stats["nodes"] > rep_nodes:                         # richest graph = the visual
            rep_nodes = stats["nodes"]
            rep_graph = _full_graph(g.store)
            rep_graph["representative_of"] = q["id"]
        if backends_seen is None:
            backends_seen = g.stats()["backends"]

        if i % 10 == 0 or i == len(instances) - 1:
            log(f"  {i + 1}/{len(instances)}  {rec['id'] or '':>16}  "
                f"recall@k={rec['recall_at_k']:.2f} hit={'Y' if rec['hit'] else '.'}  "
                f"${ingest_cost:.4f}")

    ingest_seconds = round(time.time() - t0, 1)
    if os.path.exists(store_path):
        os.remove(store_path)

    n = max(1, len(instances))
    ingest_totals = {
        "docs": len(instances), "items": total_sessions,
        "nodes": agg_nodes, "edges": agg_edges,
        "by_node_type": {}, "by_edge_type": {},
        "vocab": dict(cum_vocab),
        # pooled across the whole run (total TAGGED_AS / total episodes), not a mean-of-means
        "avg_tags_per_object": round(agg_tagged / agg_episodes, 3) if agg_episodes else 0.0,
        "avg_rel_tags_per_pair": 0.0, "pairs": 0, "rel_edges": 0,
        "seconds": ingest_seconds,
        "llm_calls": ingest_llm, "input_tokens": ingest_in, "output_tokens": ingest_out,
        "cache_read": ingest_cr, "cache_write": ingest_cw,
        "truncated": ingest_trunc,
        "tokens": ingest_tokens, "cost_usd": round(ingest_cost, 6),
    }
    query_totals = _query_totals(qrecords, judge_meter, kk)

    run_id = label or datetime.now().strftime("lme_%Y%m%d_%H%M%S")
    backends = backends_seen or {"extractor": cfg.extractor, "embedder": cfg.embedder,
                                 "answerer": cfg.rag_backend}
    backends.setdefault("agent", backends.get("answerer", "offline"))
    run = {
        "run_id": run_id, "label": label or run_id, "created_at": _now(),
        "backends": backends,
        "models": {"extractor": cfg.llm_model, "agent": cfg.rag_model,
                   "l3_judge": cfg.l3_model, "embedder": cfg.embed_model},
        "dataset": {"input": f"longmemeval:{tier} (per-instance)", "n_input": total_sessions,
                    "queries": os.path.basename(_tier_questions_path(tier)),
                    "n_queries": len(instances)},
        "config": {"k": kk, "agent_backend": cfg.rag_backend, "answerer": "ppr-rag",
                   "mode": "per-instance", "reflexion": cfg.reflexion,
                   "l3_enabled": cfg.l3_enabled, "communities": communities,
                   "match": "session-level (gold evidence sessions vs ingested episodes)",
                   "note": "fresh graph per question (no cross-instance pooling); the Input "
                           "graph is a representative single instance"},
        "cost_usd": round(ingest_totals["cost_usd"] + query_totals["cost_usd"], 6),
        "tokens": ingest_totals["tokens"] + query_totals["tokens"],
        "ingest": {"totals": ingest_totals, "steps": steps,
                   "graph": rep_graph or {"nodes": [], "edges": [], "build_order": [],
                                          "stats": {}, "kind": "full"}},
        "query": {"totals": query_totals, "queries": qrecords},
    }
    return _write_run(run, out_dir, log)


def _query_totals(qrecords: list[dict], judge_meter: UsageMeter, k: int) -> dict:
    n = len(qrecords)
    if not n:
        return {"n": 0, **totals_of([])}

    def mean(key):
        return round(sum(r[key] for r in qrecords) / n, 4)

    judged = [r["judge"]["score"] for r in qrecords
              if r.get("judge") and "score" in r["judge"]]
    proxies = [r["response_token_recall"] for r in qrecords
               if r["response_token_recall"] is not None]
    kinds = sorted({r["kind"] for r in qrecords})
    by_kind = {}
    for kind in kinds:
        rs = [r for r in qrecords if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(rs),
            "recall_at_k": round(sum(r["recall_at_k"] for r in rs) / len(rs), 3),
            "mrr": round(sum(r["mrr"] for r in rs) / len(rs), 3),
            "hit_rate": round(sum(1 for r in rs if r["hit"]) / len(rs), 3),
        }
    agent_tok = totals_of([])  # sum agent usage across records
    agent_cost = round(sum(r["cost_usd"] for r in qrecords), 6)
    agent_tokens = sum(r["tokens"] for r in qrecords)
    jt = judge_meter.totals()
    return {
        "n": n,
        "recall_at_k": mean("recall_at_k"), "mrr": mean("mrr"),
        "hit_rate": round(sum(1 for r in qrecords if r["hit"]) / n, 3),
        "citation_grounding": mean("citation_grounding"),
        "response_accuracy": round(sum(judged) / len(judged), 3) if judged else None,
        "response_token_recall": round(sum(proxies) / len(proxies), 3) if proxies else None,
        "avg_steps": round(sum(r["steps"] for r in qrecords) / n, 2),
        "avg_touched": round(sum(r["n_touched"] for r in qrecords) / n, 1),
        "by_kind": by_kind,
        "llm_calls": sum(r["llm_calls"] for r in qrecords) + jt["llm_calls"],
        "input_tokens": sum(r["input_tokens"] for r in qrecords) + jt["input_tokens"],
        "output_tokens": sum(r["output_tokens"] for r in qrecords) + jt["output_tokens"],
        "tokens": agent_tokens + jt["tokens"],
        "agent_cost_usd": agent_cost,
        "judge_cost_usd": jt["cost_usd"],
        "cost_usd": round(agent_cost + jt["cost_usd"], 6),
    }


def _update_index(out_dir: str, run: dict) -> None:
    """Append/replace this run's summary in runs/index.json (newest first)."""
    idx_path = os.path.join(out_dir, "index.json")
    summary = {
        "run_id": run["run_id"], "label": run["label"], "created_at": run["created_at"],
        "mode": run.get("config", {}).get("mode", "shared"),
        "backends": run["backends"], "models": run["models"],
        "n_input": run["dataset"]["n_input"], "n_queries": run["dataset"]["n_queries"],
        "nodes": run["ingest"]["totals"]["nodes"], "edges": run["ingest"]["totals"]["edges"],
        "cost_usd": run["cost_usd"], "tokens": run["tokens"],
        "recall_at_k": run["query"]["totals"].get("recall_at_k"),
        "mrr": run["query"]["totals"].get("mrr"),
        "hit_rate": run["query"]["totals"].get("hit_rate"),
        "response_accuracy": run["query"]["totals"].get("response_accuracy"),
        "avg_tags_per_object": run["ingest"]["totals"]["avg_tags_per_object"],
    }
    runs = []
    if os.path.exists(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as f:
                runs = json.load(f)
        except (ValueError, OSError):
            runs = []
    runs = [r for r in runs if r.get("run_id") != run["run_id"]]
    runs.insert(0, summary)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)


def summarize(run: dict) -> str:
    it, qt = run["ingest"]["totals"], run["query"]["totals"]
    lines = [
        f"run {run['run_id']}  backends={run['backends']}",
        f"  INPUT  {it['docs']} docs -> {it['nodes']} nodes / {it['edges']} edges  "
        f"(avg {it['avg_tags_per_object']} tags/obj, {it['avg_rel_tags_per_pair']} "
        f"rel-tags/pair)  {it['tokens']} tok  ${it['cost_usd']:.4f}  {it['seconds']}s",
        f"  QUERY  {qt['n']} queries  recall@k={qt['recall_at_k']}  mrr={qt['mrr']}  "
        f"hit_rate={qt['hit_rate']}  grounding={qt['citation_grounding']}"
        + (f"  judge_acc={qt['response_accuracy']}" if qt.get('response_accuracy') is not None else "")
        + f"  {qt['tokens']} tok  ${qt['cost_usd']:.4f}",
        f"  TOTAL  ${run['cost_usd']:.4f}  {run['tokens']} tokens",
    ]
    return "\n".join(lines)
