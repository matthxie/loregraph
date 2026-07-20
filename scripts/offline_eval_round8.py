"""Offline Round-8 eval: SPEAKER ATTRIBUTION (docs/OFFLINE_EVAL.md Round 8).

NO paid LLM calls. Drives the exact read path (HybridRetriever.retrieve →
ContextBuilder.build — what KnowledgeGraph.search() runs) with config.speaker_attribution
OFF vs ON, on the ~100 cached per-instance benchmark stores of runs/sample-datefix-events-1.

Each store is COPIED to a temp path, then speakers are BACKFILLED into the copy ($0 local
regex, additive — never touches the cache, never changes the ingest-cache key). Two checks:

  1. FIDELITY (knob OFF) — the OFF pass reproduces the paid run's recorded gold in_context
     marks (184/184). Backfilling speakers + the knob-off code path change no retrieval/context.
  2. SOFT-NOT-HARD (knob ON) — the ON context equals the OFF context after stripping the
     " [assistant]" suffixes: the marker is purely additive, no fact line is ever removed
     (~13% of benchmark answers live only in assistant turns; speaker is a marker, not a filter).

Then it prints the watch list for the paid run (reader BEHAVIOR is an LLM change, unjudgeable
offline): the three abstention questions expected to FLIP toward abstained/correct
(031748ae_abs, 19b5f2b3_abs, 09ba9854_abs) — whose offending fact lines now carry [assistant] —
and 06878be2, whose assistant-recommendation lines are marked but NOT removed (expect UNCHANGED,
the assistant content is still used).

Run:  .venv/bin/python scripts/offline_eval_round8.py [--out runs/offline_eval_round8]
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

# ---- hard no-LLM guard (kg auto-loads .env on import) ----
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)
kg_graph.get_extractor = lambda config: ScriptedExtractor({})

RUN_JSON = "runs/sample-datefix-events-1/run.json"
# the *_abs abstention targets (expected to FLIP toward abstained) + the assistant-content
# question that must stay UNCHANGED (assistant material marked but still usable).
ABS_TARGETS = ("031748ae_abs", "19b5f2b3_abs", "09ba9854_abs")
KEEP_TARGET = "06878be2"

INGEST_OVERRIDES = {"extractor_backend": "cue_gated", "event_facts": True,
                    "ingest_date_filter": True}
QUERY_OVERRIDES = {"rag_retarget": "ce", "rag_provenance_promote": True,
                   "mmr_lambda": 1.0, "rag_parent_expand": 2,
                   "rag_chunks_per_source": 2, "history_all_lanes": True}
K = 8

_FACTS_HDR = "FACTS currently valid among the relevant entities:"


def sess(eid: str) -> str:
    return eid.split("#", 1)[0]


def locate_stores(questions: list[dict]) -> dict[str, str]:
    from kg.corpus import iter_lme_instances
    os.environ.setdefault("KG_LLM", "openai")           # so the llm_model digest matches
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


def _facts_lines(blob: str) -> list[str]:
    """The FACTS-section bullet lines of a context blob (up to the next blank/section)."""
    if _FACTS_HDR not in blob:
        return []
    tail = blob.split(_FACTS_HDR, 1)[1]
    out = []
    for ln in tail.splitlines():
        if ln.startswith("- "):
            out.append(ln)
        elif ln.strip() == "" and out:
            break
    return out


def eval_question(store, embedder, canon, base_cfg: Config, q: dict) -> dict:
    as_of = q.get("question_date")

    def run(on: bool) -> dict:
        cfg = replace(base_cfg, speaker_attribution=on)
        retriever = HybridRetriever(store, embedder, canon, cfg)
        builder = ContextBuilder(store, cfg)
        res = retriever.retrieve(q["query"], k=K, as_of=as_of)
        ep_ids, _facts, blob = builder.build(res)
        return {"ep_ids": list(ep_ids), "blob": blob,
                "ctx_sessions": sorted({sess(e) for e in ep_ids}),
                "facts": _facts_lines(blob)}

    off, on = run(False), run(True)
    marked = [ln for ln in on["facts"] if ln.endswith("[assistant]")]
    # SOFT-NOT-HARD: the ON blob is the OFF blob + " [assistant]" suffixes only.
    append_only = on["blob"].replace(" [assistant]", "") == off["blob"]
    return {
        "qid": q["id"],
        "ctx_off": off["ctx_sessions"],
        "ctx_on": on["ctx_sessions"],
        "n_facts": len(on["facts"]),
        "n_marked": len(marked),
        "marked": marked,
        "append_only": append_only,
        "ctx_same": off["ep_ids"] == on["ep_ids"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/offline_eval_round8")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    base_cfg = Config.default()
    for k, v in {**INGEST_OVERRIDES, **QUERY_OVERRIDES}.items():
        setattr(base_cfg, k, v)

    run = json.load(open(RUN_JSON))
    questions = run["query"]["queries"][: args.limit]
    stores = locate_stores(questions)
    missing = [q["id"] for q in questions if q["id"] not in stores]
    if missing:
        sys.exit(f"no cache hit for {len(missing)} instances: {missing[:5]} …")
    print(f"[real] {len(questions)} questions, all cached", flush=True)

    work = tempfile.mkdtemp(prefix="round8-")
    rows, t0 = [], time.time()
    total_stamped = total_speakers = 0
    for i, q in enumerate(questions):
        wp = os.path.join(work, f"{q['id']}.db")
        _sqlite_copy(stores[q["id"]], wp)
        g = KnowledgeGraph.open(wp, base_cfg)
        bf = g.backfill_speakers()              # $0 local; additive on the COPY
        total_stamped += bf["stamped"]
        total_speakers = max(total_speakers, bf["speakers"])
        rows.append({**eval_question(g.store, g.embedder, g.canon, base_cfg, q),
                     "backfill": bf})
        del g
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(wp + suf):
                os.remove(wp + suf)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(questions)}  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "real.json"), "w") as f:
        json.dump(rows, f, indent=1)
    by = {r["qid"]: r for r in rows}

    # ---------- fidelity gate: knob-OFF vs run.json recorded gold_marks ----------
    agree = tot = 0
    for q in questions:
        r = by[q["id"]]
        ctx_off = set(r["ctx_off"])
        for gm in q.get("gold_marks", []):
            gid = "ep_" + gm["id"][4:] if gm["id"].startswith("obj_") else gm["id"]
            tot += 1
            agree += ((sess(gid) in ctx_off) == gm["in_context"])

    n = len(rows)
    all_append_only = sum(r["append_only"] for r in rows)
    fired = sum(1 for r in rows if r["n_marked"] > 0)
    ctx_same = sum(r["ctx_same"] for r in rows)

    print(f"\n== Round 8 speaker attribution — REAL (n={n}) ==")
    print(f"speakers backfilled       : {total_stamped} episode stamps, "
          f"{total_speakers} registry rows/store (avg {total_stamped/n:.0f} chunks/store)")
    print(f"fidelity (knob OFF) vs run.json : {agree}/{tot} gold in_context marks agree")
    print(f"append-only (ON == OFF + markers) : {all_append_only}/{n}  "
          f"(soft-not-hard: no fact line removed)")
    print(f"context episode set unchanged ON  : {ctx_same}/{n}")
    print(f"questions with ≥1 [assistant] mark: {fired}/{n}")

    print("\n-- watch list (paid run behavioral targets) --")
    for qid in ABS_TARGETS + (KEEP_TARGET,):
        r = by.get(qid)
        if not r:
            print(f"  {qid}: (no cache / not in run)")
            continue
        tag = "expect FLIP→abstain/correct" if qid in ABS_TARGETS else "expect UNCHANGED (used)"
        print(f"\n  {qid}  [{tag}]")
        print(f"    facts={r['n_facts']} marked[assistant]={r['n_marked']} "
              f"append_only={r['append_only']} ctx_same={r['ctx_same']}")
        for ln in r["marked"][:6]:
            print(f"      {ln}")

    ok = (agree == tot and all_append_only == n)
    print("\nREAL RESULT:", "PASS" if ok else "CHECK", "->", args.out)

    print("\n-- PAID validation command (small tier, cached; NOT run here) --")
    print("python -m kg testrun --mode per-instance --tier sample \\")
    print("  --set history_all_lanes=true --set event_facts=true \\")
    print("  --set rag_retarget=ce --set rag_provenance_promote=true \\")
    print("  --set mmr_lambda=1.0 --set rag_parent_expand=2 --set rag_chunks_per_source=2 \\")
    print("  --set speaker_attribution=true \\")
    print("  --label speaker-attribution-round8-1 --out runs")
    print("(prereq: one-time $0 local backfill —")
    print("   for db in store/cache/*.db; do python -m kg --store \"$db\" backfill-speakers; done)")
    print(f"watch: {', '.join(ABS_TARGETS)} (expect FLIP), {KEEP_TARGET} (expect UNCHANGED)")


if __name__ == "__main__":
    main()
