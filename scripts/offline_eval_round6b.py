"""Offline Round-6b eval: graph-tally evidence lines (docs/OFFLINE_EVAL.md Round 6b).

NO paid LLM calls. Drives the exact read path (HybridRetriever.retrieve →
ContextBuilder.build — what KnowledgeGraph.search() runs) against the CACHED
per-instance benchmark stores from runs/sample-datefix-events-1 (store/cache/*.db,
extraction cost already sunk). Each cached store is COPIED to a temp path first; the
cache is never opened read-write. Same harness pattern as scripts/offline_eval_round3.py.

For each of the run's ~100 questions, builds the context TWICE — agg_evidence OFF
(baseline) and ON — and records:
  * whether a "GRAPH TALLIES" section appears (and its lines)
  * gold session in-context marks (to prove no canary regresses: the section is
    append-only, so episode selection / gold-in-context must be byte-for-byte unchanged)
  * fidelity of the OFF baseline vs the paid run's recorded gold_marks.in_context

Headline checks:
  * the six stable multi-session failure questions gain a tally section under agg_evidence
  * NO question's gold-in-context marks change vs the OFF baseline (canary safety)

Run:  .venv/bin/python scripts/offline_eval_round6b.py [--out runs/offline_eval_round6b]
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
from kg.ingest_cache import _sqlite_copy, cache_path, ingest_cache_key  # noqa: E402
from kg.rag import ContextBuilder                              # noqa: E402
from kg.retrieval import HybridRetriever                       # noqa: E402

# ---- hard no-LLM guard (same as offline_eval_round3.py): kg auto-loads .env on import.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)
kg_graph.get_extractor = lambda config: ScriptedExtractor({})

RUN_JSON = "runs/sample-datefix-events-1/run.json"
# The six aggregate-shaped, all-gold-in-context failure questions (Round 6a §1).
FAILURE_QIDS = ("0a995998", "1c549ce4", "2318644b", "09ba9854_abs",
                "370a8ff4", "982b5123")

INGEST_OVERRIDES = {"extractor_backend": "cue_gated", "event_facts": True,
                    "ingest_date_filter": True}
QUERY_OVERRIDES = {"rag_retarget": "ce", "rag_provenance_promote": True,
                   "mmr_lambda": 1.0, "rag_parent_expand": 2,
                   "rag_chunks_per_source": 2, "history_all_lanes": True}
K = 8


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


def _tally_lines(blob: str) -> list[str]:
    if "GRAPH TALLIES" not in blob:
        return []
    tail = blob.split("GRAPH TALLIES", 1)[1]
    lines = []
    for ln in tail.splitlines()[1:]:      # skip the header remainder line
        ln = ln.strip()
        if not ln:
            continue
        lines.append(ln)
    return lines


def eval_question(store, embedder, canon, base_cfg: Config, q: dict) -> dict:
    as_of = q.get("question_date")
    golds = gold_sessions(q)

    def run(agg: bool):
        cfg = replace(base_cfg, agg_evidence=agg)
        retriever = HybridRetriever(store, embedder, canon, cfg)
        builder = ContextBuilder(store, cfg)
        res = retriever.retrieve(q["query"], k=K, as_of=as_of)
        ep_ids, _facts, blob = builder.build(res)
        ctx_sessions = {sess(e) for e in ep_ids}
        return blob, ep_ids, {g: (g in ctx_sessions) for g in golds}

    off_blob, off_eps, off_marks = run(False)
    on_blob, on_eps, on_marks = run(True)
    return {
        "qid": q["id"],
        "off_marks": off_marks, "on_marks": on_marks,
        "eps_identical": off_eps == on_eps,
        "marks_identical": off_marks == on_marks,
        "has_tally": "GRAPH TALLIES" in on_blob,
        "off_has_tally": "GRAPH TALLIES" in off_blob,
        "tally_lines": _tally_lines(on_blob),
        "off_chars": len(off_blob), "on_chars": len(on_blob),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/offline_eval_round6b")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    run = json.load(open(RUN_JSON))
    questions = run["query"]["queries"][: args.limit]
    stores = locate_stores(questions)
    missing = [q["id"] for q in questions if q["id"] not in stores]
    if missing:
        sys.exit(f"no cache hit for {len(missing)} instances: {missing[:5]} …")
    print(f"{len(questions)} questions, all cached", flush=True)

    base_cfg = Config.default()
    for k, v in {**INGEST_OVERRIDES, **QUERY_OVERRIDES}.items():
        setattr(base_cfg, k, v)

    work = tempfile.mkdtemp(prefix="round6b-")
    rows = []
    t0 = time.time()
    for i, q in enumerate(questions):
        wp = os.path.join(work, f"{q['id']}.db")
        _sqlite_copy(stores[q["id"]], wp)
        g = KnowledgeGraph.open(wp, base_cfg)
        rows.append(eval_question(g.store, g.embedder, g.canon, base_cfg, q))
        del g
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(wp + suf):
                os.remove(wp + suf)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(questions)}  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    by = {r["qid"]: r for r in rows}

    # ---------------- summary ----------------
    n = len(rows)
    n_tally = sum(r["has_tally"] for r in rows)
    n_off_tally = sum(r["off_has_tally"] for r in rows)
    regressed = [r["qid"] for r in rows if not r["marks_identical"]]
    eps_changed = [r["qid"] for r in rows if not r["eps_identical"]]
    print(f"\n== Round 6b tally sweep (n={n}) ==")
    print(f"agg_evidence OFF, tally present : {n_off_tally}  (MUST be 0)")
    print(f"agg_evidence ON,  tally present : {n_tally}/{n}")
    print(f"gold-in-context marks changed   : {len(regressed)}  (MUST be 0) {regressed}")
    print(f"context episode set changed     : {len(eps_changed)}  (MUST be 0) {eps_changed}")
    added = sum(r["on_chars"] - r["off_chars"] for r in rows) / n
    print(f"mean context chars added by ON  : {added:.0f}")

    print("\n-- six failure questions (Round 6a) --")
    for qid in FAILURE_QIDS:
        r = by.get(qid)
        if not r:
            print(f"  {qid}: (no cache)")
            continue
        print(f"  {qid}: tally={r['has_tally']}  +{r['on_chars']-r['off_chars']} chars  "
              f"marks_same={r['marks_identical']}")
        for ln in r["tally_lines"][:10]:
            print(f"       {ln}")

    # fidelity: OFF baseline vs run.json recorded gold_marks.in_context
    agree = tot = 0
    for q in questions:
        r = by[q["id"]]
        for gm in q.get("gold_marks", []):
            gid = "ep_" + gm["id"][4:]
            tot += 1
            agree += (r["off_marks"].get(gid) == gm["in_context"])
    print(f"\nfidelity vs run.json (OFF baseline): {agree}/{tot} gold in_context marks agree")

    ok = (n_off_tally == 0 and not regressed and not eps_changed
          and all(by.get(x, {}).get("has_tally") for x in FAILURE_QIDS if x in by))
    print("\nRESULT:", "PASS" if ok else "CHECK", "-> ", args.out)


if __name__ == "__main__":
    main()
