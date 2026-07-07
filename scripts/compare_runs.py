"""Compare two testrun run.json files: per-kind accuracy deltas, per-question flips,
persistent failures classified by gold-evidence position, and (when present) whether the
rag_answer_events enumeration scaffold was filled and helped.

Usage:
  python scripts/compare_runs.py runs/OLD/run.json runs/NEW/run.json
"""
from __future__ import annotations

import json
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    return run


def correct(q: dict):
    j = q.get("judge") or {}
    return j.get("correct") if isinstance(j, dict) else None


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    old_run, new_run = load(sys.argv[1]), load(sys.argv[2])
    old = {q["id"]: q for q in old_run["query"]["queries"]}
    new = {q["id"]: q for q in new_run["query"]["queries"]}
    shared = [qid for qid in new if qid in old]

    # ---- per-kind accuracy ------------------------------------------------- #
    print(f"OLD: {old_run['run_id']}  (acc "
          f"{old_run['query']['totals'].get('response_accuracy')}, "
          f"cost ${old_run.get('cost_usd')})")
    print(f"NEW: {new_run['run_id']}  (acc "
          f"{new_run['query']['totals'].get('response_accuracy')}, "
          f"cost ${new_run.get('cost_usd')})")
    print()
    kinds = sorted({q["kind"] for q in new.values()})
    print(f"{'kind':<28}{'n':>4}{'old':>8}{'new':>8}{'delta':>8}")
    for kind in kinds + ["OVERALL"]:
        ids = [i for i in shared if kind == "OVERALL" or new[i]["kind"] == kind]
        if not ids:
            continue
        o = sum(1 for i in ids if correct(old[i]) is True) / len(ids)
        n = sum(1 for i in ids if correct(new[i]) is True) / len(ids)
        print(f"{kind:<28}{len(ids):>4}{o:>8.3f}{n:>8.3f}{n - o:>+8.3f}")

    # ---- flips ------------------------------------------------------------- #
    ups = [i for i in shared if correct(old[i]) is False and correct(new[i]) is True]
    downs = [i for i in shared if correct(old[i]) is True and correct(new[i]) is False]
    both = [i for i in shared if correct(old[i]) is False and correct(new[i]) is False]
    print(f"\nFIXED in new ({len(ups)}):")
    for i in ups:
        print(f"  + [{new[i]['kind']}] {new[i]['query'][:80]}")
    print(f"BROKEN in new ({len(downs)}):")
    for i in downs:
        print(f"  - [{new[i]['kind']}] {new[i]['query'][:80]}")

    # ---- persistent failures, classified ----------------------------------- #
    print(f"STILL WRONG in both ({len(both)}):")
    for i in both:
        q = new[i]
        gm = q.get("gold_marks") or []
        n_gold = len(gm)
        in_ctx = sum(1 for g in gm if g.get("in_context"))
        cls = ("READER" if in_ctx == n_gold else
               "PARTIAL" if in_ctx else "RETRIEVAL")
        print(f"  = {cls:<9} ctx {in_ctx}/{n_gold} [{q['kind']}] {q['query'][:70]}")
        print(f"      judge: {(q.get('judge') or {}).get('reason', '')[:100]}")

    # ---- events scaffold efficacy (new run only, if recorded) --------------- #
    with_ev = [i for i in shared if new[i].get("events")]
    if with_ev:
        ev_right = sum(1 for i in with_ev if correct(new[i]) is True)
        print(f"\nEVENTS scaffold: filled on {len(with_ev)} questions, "
              f"{ev_right} correct ({ev_right / len(with_ev):.0%})")
        wrong_filled = [i for i in with_ev if correct(new[i]) is False]
        for i in wrong_filled:
            print(f"  filled-but-wrong [{new[i]['kind']}] {new[i]['query'][:70]} "
                  f"({len(new[i]['events'])} events listed)")
    else:
        print("\nEVENTS scaffold: not recorded in the new run "
              "(field absent — run predates persistence or feature off)")


if __name__ == "__main__":
    main()
