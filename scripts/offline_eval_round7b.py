"""Offline Round-7b eval: the FACT LANE (docs/OFFLINE_EVAL.md Round 7b).

NO paid LLM calls. Drives the exact read path (HybridRetriever.retrieve →
ContextBuilder.build — what KnowledgeGraph.search() runs) with config.fact_lane OFF vs ON,
on two datasets:

(a) REAL — the ~100 cached per-instance benchmark stores of runs/sample-datefix-events-1
    (store/cache/*.db; extraction cost already sunk). Each store is COPIED to a temp path,
    then fact vectors are BACKFILLED into the copy ($0 local bge, additive — never touches the
    cache). The knob-OFF pass must reproduce the paid run's recorded gold in_context marks
    (fidelity gate); the knob-ON pass is compared question-by-question for wins/regressions.
    Named targets: 06f04340, 2ce6a0f2, 0977f2af (the Round 4 §3 dilution cases).

(b) SYNTH — the Round-1/2 synthetic first-person store (scripts/offline_eval._SYNTH) EXTENDED
    with needle episodes: a fact stated ONCE, off-topic (blood type while booking a flight; a
    dentist referral inside a scheduling chat) + preference probes that hit the aggregate
    surfaces. Shows seed/pool/context inclusion of the needle with the lane on vs off.

Run:  .venv/bin/python scripts/offline_eval_round7b.py [--out runs/offline_eval_round7b]
      .venv/bin/python scripts/offline_eval_round7b.py --synth-only
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
from kg.models import NodeType                                 # noqa: E402
from kg.rag import ContextBuilder                              # noqa: E402
from kg.retrieval import HybridRetriever                       # noqa: E402

# ---- hard no-LLM guard (same as offline_eval_round3.py): kg auto-loads .env on import.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)
kg_graph.get_extractor = lambda config: ScriptedExtractor({})

RUN_JSON = "runs/sample-datefix-events-1/run.json"
TARGET_QIDS = ("06f04340", "2ce6a0f2", "0977f2af")

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


def _ans_chunks(store, golds: list[str], ans: str) -> list[str]:
    out = []
    if not ans:
        return out
    for gold in golds:
        cands = [gold] if store.get_node(gold) is not None else []
        for n in store.nodes_of_type(NodeType.EPISODE):
            if n.id.startswith(gold + "#"):
                cands.append(n.id)
        for cid in cands:
            n = store.get_node(cid)
            if n and ans in (n.raw_text or "").lower():
                out.append(cid)
    return out


def eval_question(store, embedder, canon, base_cfg: Config, q: dict) -> dict:
    as_of = q.get("question_date")
    golds = gold_sessions(q)
    ans = (q.get("answer_expected") or "").strip().lower()
    ans_chunks = _ans_chunks(store, golds, ans)

    def run(fact_lane: bool) -> dict:
        cfg = replace(base_cfg, fact_lane=fact_lane)
        retriever = HybridRetriever(store, embedder, canon, cfg)
        builder = ContextBuilder(store, cfg)
        res = retriever.retrieve(q["query"], k=K, as_of=as_of)
        ep_ids, _facts, blob = builder.build(res)
        pool = [eid for eid, _s in getattr(res, "ppr_pool", [])]
        ctx_sessions = {sess(e) for e in ep_ids}

        def first_rank(order, gold):
            for i, eid in enumerate(order):
                if sess(eid) == gold:
                    return i + 1
            return None

        fm = getattr(res, "fact_matched", {}) or {}
        return {
            "gold_in_ctx": {g: (g in ctx_sessions) for g in golds},
            "all_gold": all(g in ctx_sessions for g in golds),
            "any_gold": any(g in ctx_sessions for g in golds),
            "gold_rank_pool": {g: first_rank(pool, g) for g in golds},
            "ans_chunk_in_ctx": (any(c in ep_ids for c in ans_chunks)
                                 if ans_chunks else None),
            "ans_substr_in_ctx": (ans in blob.lower()) if ans else None,
            "ctx_chars": len(blob),
            "fact_fired": bool(fm.get("surfaces")),
            "fact_surfaces": fm.get("surfaces", [])[:K],
            "fact_gold_seeded": [g for g in golds
                                 if any(sess(e) == g for e in fm.get("episodes", []))],
        }

    off, on = run(False), run(True)
    return {"qid": q["id"], "lane": "", "golds": golds, "off": off, "on": on,
            "changed": off["gold_in_ctx"] != on["gold_in_ctx"]
            or off["ans_chunk_in_ctx"] != on["ans_chunk_in_ctx"]}


def run_real(args, base_cfg: Config) -> None:
    run = json.load(open(RUN_JSON))
    questions = run["query"]["queries"][: args.limit]
    stores = locate_stores(questions)
    missing = [q["id"] for q in questions if q["id"] not in stores]
    if missing:
        sys.exit(f"no cache hit for {len(missing)} instances: {missing[:5]} …")
    print(f"[real] {len(questions)} questions, all cached", flush=True)

    work = tempfile.mkdtemp(prefix="round7b-")
    rows, t0 = [], time.time()
    for i, q in enumerate(questions):
        wp = os.path.join(work, f"{q['id']}.db")
        _sqlite_copy(stores[q["id"]], wp)
        g = KnowledgeGraph.open(wp, base_cfg)
        bf = g.backfill_fact_vectors()          # $0 local; additive on the COPY
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
        for gm in q.get("gold_marks", []):
            gid = "ep_" + gm["id"][4:]
            tot += 1
            agree += (r["off"]["gold_in_ctx"].get(gid) == gm["in_context"])
    n = len(rows)
    off_all = sum(r["off"]["all_gold"] for r in rows)
    on_all = sum(r["on"]["all_gold"] for r in rows)
    off_any = sum(r["off"]["any_gold"] for r in rows)
    on_any = sum(r["on"]["any_gold"] for r in rows)
    fired = sum(r["on"]["fact_fired"] for r in rows)

    def ansc(side):
        return sum(bool(r[side]["ans_chunk_in_ctx"]) for r in rows
                   if r[side]["ans_chunk_in_ctx"] is not None)

    print(f"\n== Round 7b fact lane — REAL (n={n}) ==")
    print(f"fidelity (knob OFF) vs run.json : {agree}/{tot} gold in_context marks agree")
    print(f"all-gold-in-ctx   OFF/ON : {off_all} / {on_all}")
    print(f"any-gold-in-ctx   OFF/ON : {off_any} / {on_any}")
    print(f"ans-chunk-in-ctx  OFF/ON : {ansc('off')} / {ansc('on')}")
    print(f"fact lane fired          : {fired}/{n} questions")

    wins = [r["qid"] for r in rows
            if (r["on"]["all_gold"] and not r["off"]["all_gold"])
            or (r["on"]["any_gold"] and not r["off"]["any_gold"])
            or (r["on"]["ans_chunk_in_ctx"] and not r["off"]["ans_chunk_in_ctx"])]
    regr = [r["qid"] for r in rows
            if (r["off"]["all_gold"] and not r["on"]["all_gold"])
            or (r["off"]["any_gold"] and not r["on"]["any_gold"])
            or (r["off"]["ans_chunk_in_ctx"] and not r["on"]["ans_chunk_in_ctx"])]
    print(f"\nWINS ({len(wins)}): {wins}")
    print(f"REGRESSIONS ({len(regr)}): {regr}")
    for qid in wins + regr:
        r = by[qid]
        print(f"  {qid}: gold OFF={r['off']['gold_in_ctx']} ON={r['on']['gold_in_ctx']} "
              f"ans_chunk {r['off']['ans_chunk_in_ctx']}->{r['on']['ans_chunk_in_ctx']}")

    print("\n-- named targets --")
    for qid in TARGET_QIDS:
        r = by.get(qid)
        if not r:
            print(f"  {qid}: (no cache)")
            continue
        print(f"  {qid}: OFF all={r['off']['all_gold']} any={r['off']['any_gold']} "
              f"pool={r['off']['gold_rank_pool']}")
        print(f"           ON all={r['on']['all_gold']} any={r['on']['any_gold']} "
              f"pool={r['on']['gold_rank_pool']} fired={r['on']['fact_fired']} "
              f"gold_seeded_by_fact={r['on']['fact_gold_seeded']}")
        for s in r["on"]["fact_surfaces"][:6]:
            print(f"             matched: {s}")

    # noise: the matched surfaces across all fired questions (manual junk characterization)
    with open(os.path.join(args.out, "matched_surfaces.json"), "w") as f:
        json.dump({r["qid"]: r["on"]["fact_surfaces"] for r in rows
                   if r["on"]["fact_fired"]}, f, indent=1)

    ok = (agree == tot and not regr)
    print("\nREAL RESULT:", "PASS" if ok else "CHECK", "->", args.out)


# --------------------------------------------------------------------------- #
# (b) SYNTHETIC needle probes
# --------------------------------------------------------------------------- #
def build_synth_needle_store(path: str) -> KnowledgeGraph:
    """The Round-1/2 synth store (scripts/offline_eval._SYNTH) EXTENDED with off-topic needle
    episodes, built with fact_vectors on so the lane has targets."""
    from scripts.offline_eval import _E, _R, _SYNTH
    from kg.corpus import CorpusItem
    from kg.extractors import Extraction

    needles = [
        # fact stated ONCE, off-topic: blood type mentioned while booking a flight. The text
        # never says "blood type"; only the extracted PREDICATE bridges to the query.
        ("nd_blood", "2024-03-01",
         "Sorted out the trip logistics today: booked the United flight to Chicago, reserved "
         "an aisle seat, printed the boarding passes for the whole family, and pre-paid the "
         "airport parking for the week.",
         ["me", "O-negative"], [("me", "O-negative", "has_blood_type")]),
        # dentist referral inside a scheduling chat (the word "dentist" is absent from text)
        ("nd_dds", "2024-05-12",
         "Rescheduling chat with the team: moved the Tuesday sync to Thursday, pushed the "
         "1:1s to next week, and Priya passed along the name of a specialist she likes, "
         "Dr. Nguyen, for that molar I mentioned.",
         ["me", "Dr. Nguyen"], [("me", "Dr. Nguyen", "sees_dentist")]),
    ]
    cfg = Config.default()
    cfg.embedder = "st"
    cfg.self_entity = True
    cfg.self_name = "me"
    cfg.event_facts = True
    cfg.fact_vectors = True
    g = KnowledgeGraph.open(path, cfg)
    items, table = [], {}
    for eid, day, text, ents, rels in _SYNTH + needles:
        items.append(CorpusItem(id=eid, modality="text", source_ref=f"synthetic/{eid}",
                                title=eid, text=text, created_at=f"{day}T12:00:00+00:00"))
        table[text] = Extraction(entities=[_E(n) for n in ents], tags=["personal"],
                                 relations=[_R(*r) for r in rels])
    g.extractor = ScriptedExtractor(table)
    g.ingest(items)
    g.save()
    return g


SYNTH_NEEDLE_PROBES = [
    # (id, question, as_of, needle_session, gold_substrings)
    ("blood", "What is my blood type?", None, "ep_nd_blood", ["O-negative"]),
    ("dentist", "Who is my dentist?", None, "ep_nd_dds", ["Nguyen"]),
    # preference probe hitting the distilled AGGREGATE surfaces (went_to the park 5x, etc.)
    ("prefs", "What do I like to do for fun?", None, None, ["park"]),
]


def run_synth(args, base_cfg: Config) -> None:
    work = tempfile.mkdtemp(prefix="round7b-synth-")
    sp = os.path.join(work, "kg.db")
    print("[synth] building needle store …", flush=True)
    g = build_synth_needle_store(sp)
    from kg.fact_vectors import FACT_KIND
    print(f"[synth] {len(list(g.store.nodes_of_type(NodeType.EPISODE)))} episodes, "
          f"{len(g.store.vectors.ids(FACT_KIND))} fact vectors", flush=True)

    rows = []
    for qid, q, as_of, needle, subs in SYNTH_NEEDLE_PROBES:
        row = {"qid": qid, "q": q, "needle": needle}
        for lane in (False, True):
            cfg = replace(base_cfg, fact_lane=lane)
            cfg = replace(cfg, self_entity=True, self_name="me")
            retriever = HybridRetriever(g.store, g.embedder, g.canon, cfg)
            builder = ContextBuilder(g.store, cfg)
            res = retriever.retrieve(q, as_of=as_of, k=K)
            ep_ids, _facts, blob = builder.build(res)
            pool = [e for e, _s in getattr(res, "ppr_pool", [])]
            fm = getattr(res, "fact_matched", {}) or {}
            key = "on" if lane else "off"
            row[key] = {
                "needle_seeded": needle in res.seeds if needle else None,
                "needle_in_pool": needle in pool if needle else None,
                "needle_in_ctx": needle in ep_ids if needle else None,
                "subs_in_ctx": [s for s in subs if s.lower() in blob.lower()],
                "matched": fm.get("surfaces", [])[:6],
                "has_matched_mark": "[matched]" in blob,
            }
        rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "synth.json"), "w") as f:
        json.dump(rows, f, indent=1)

    print(f"\n== Round 7b fact lane — SYNTH needles ==")
    for r in rows:
        print(f"\n{r['qid']}: {r['q']!r}  (needle={r['needle']})")
        for key in ("off", "on"):
            s = r[key]
            print(f"  {key:3s}: seeded={s['needle_seeded']} pool={s['needle_in_pool']} "
                  f"ctx={s['needle_in_ctx']} subs={s['subs_in_ctx']} "
                  f"marked={s['has_matched_mark']}")
        if r["on"]["matched"]:
            print(f"       matched surfaces: {r['on']['matched']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/offline_eval_round7b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--real-only", action="store_true")
    args = ap.parse_args()

    base_cfg = Config.default()
    for k, v in {**INGEST_OVERRIDES, **QUERY_OVERRIDES}.items():
        setattr(base_cfg, k, v)

    if not args.synth_only:
        run_real(args, base_cfg)
    if not args.real_only:
        run_synth(args, base_cfg)


if __name__ == "__main__":
    main()
