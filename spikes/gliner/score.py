"""Score local (GLiNER-Relex) vs LLM (gpt-4o-mini) extractions per session.

Entity agreement: precision/recall of local vs LLM entity names, exact on normalized
name; fuzzy near-misses (containment or difflib >= 0.85) tallied separately.
Relation agreement: fraction of LLM relation pairs with ANY local relation connecting
the same two entities (either direction, any label; endpoints matched fuzzily).
Temporal: how many LLM relations carry temporal signal (status=ended or valid bounds)
and whether locals ever do (they can't — no such head).
Writes scores.json + a side_by_side.txt for eyeballing.
"""
import difflib, json, os, re, string, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_FILE = sys.argv[1] if len(sys.argv) > 1 else "local_extractions.json"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
llm = json.load(open(os.path.join(HERE, "llm_extractions.json"), encoding="utf-8"))
loc = json.load(open(os.path.join(HERE, LOCAL_FILE), encoding="utf-8"))
loc_by_id = {s["id"]: s for s in loc["sessions"]}

_PUNCT = str.maketrans("", "", string.punctuation)


def norm(name: str) -> str:
    n = name.lower().strip().translate(_PUNCT)
    n = re.sub(r"^(the|a|an|my|his|her|their)\s+", "", n)
    n = " ".join(n.split())
    # the LLM names the narrator 'User'; the local runner normalizes to 'me' — same node
    return "me" if n in ("user", "i") else n


# chat-role noise excluded from ENTITY scoring on both sides (relations keep 'me')
_ROLE_ENTS = {"me", "assistant"}


def match(a: str, b: str):
    """exact -> 'exact'; containment/similar -> 'fuzzy'; else None"""
    if a == b:
        return "exact"
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return "fuzzy"
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.85:
        return "fuzzy"
    return None


def best_match(name: str, pool: list[str]):
    if name in pool:
        return name, "exact"
    for p in pool:
        m = match(name, p)
        if m:
            return p, m
    return None, None


rows, sbs = [], []
tot = dict(llm_ents=0, loc_ents=0, exact=0, fuzzy=0, loc_matched=0,
           llm_rels=0, loc_rels=0, rel_hit=0, llm_temporal=0, temporal_hit=0)

for s in llm["sessions"]:
    sid = s["id"]
    L = loc_by_id[sid]
    llm_names = {}
    for e in s["entities"]:
        n = norm(e["name"])
        if n not in _ROLE_ENTS:
            llm_names.setdefault(n, e)
    loc_names = {}
    for e in L["entities"]:
        n = norm(e["name"])
        if n not in _ROLE_ENTS:
            loc_names.setdefault(n, e)
    llm_pool, loc_pool = list(llm_names), list(loc_names)

    exact = fuzzy = 0
    missed = []
    for n in llm_pool:
        m, kind = best_match(n, loc_pool)
        if kind == "exact":
            exact += 1
        elif kind == "fuzzy":
            fuzzy += 1
        else:
            missed.append(llm_names[n]["name"])
    loc_matched = sum(1 for n in loc_pool if best_match(n, llm_pool)[1])
    extra = [loc_names[n]["name"] for n in loc_pool if not best_match(n, llm_pool)[1]]

    # ---- relations: LLM pair covered by any local edge between same two entities?
    def ep_match(llm_ep: str, loc_eps: set[str]) -> bool:
        n = norm(llm_ep)
        return any(match(n, x) for x in loc_eps) or n in loc_eps

    loc_pairs = set()
    for r in L["relations"]:
        loc_pairs.add((norm(r["source"]), norm(r["target"])))
    loc_endpoints = {a for a, _ in loc_pairs} | {b for _, b in loc_pairs}

    rel_hit = 0
    rel_missed = []
    n_temporal = temporal_hit = 0
    for r in s["relations"]:
        a, b = norm(r["source"]), norm(r["target"])
        hit = False
        for (x, y) in loc_pairs:
            if ((match(a, x) or a == x) and (match(b, y) or b == y)) or \
               ((match(a, y) or a == y) and (match(b, x) or b == x)):
                hit = True
                break
        temporal = (r.get("status") == "ended" or r.get("valid_from") or r.get("valid_to"))
        if temporal:
            n_temporal += 1
            if hit:
                temporal_hit += 1
        if hit:
            rel_hit += 1
        else:
            rel_missed.append(f'{r["source"]} -[{",".join(r["labels"])}]-> {r["target"]}'
                              + (" [TEMPORAL]" if temporal else ""))

    rows.append({
        "id": sid,
        "llm_ents": len(llm_pool), "loc_ents": len(loc_pool),
        "ent_recall_exact": round(exact / len(llm_pool), 3) if llm_pool else None,
        "ent_recall_incl_fuzzy": round((exact + fuzzy) / len(llm_pool), 3) if llm_pool else None,
        "ent_precision": round(loc_matched / len(loc_pool), 3) if loc_pool else None,
        "llm_rels": len(s["relations"]), "loc_rels": len(L["relations"]),
        "rel_pair_coverage": round(rel_hit / len(s["relations"]), 3) if s["relations"] else None,
        "llm_temporal_rels": n_temporal,
    })
    tot["llm_ents"] += len(llm_pool); tot["loc_ents"] += len(loc_pool)
    tot["exact"] += exact; tot["fuzzy"] += fuzzy; tot["loc_matched"] += loc_matched
    tot["llm_rels"] += len(s["relations"]); tot["loc_rels"] += len(L["relations"])
    tot["rel_hit"] += rel_hit
    tot["llm_temporal"] += n_temporal; tot["temporal_hit"] += temporal_hit

    sbs.append(f"\n{'='*90}\nSESSION {sid}\n"
               f"LLM entities ({len(llm_pool)}): "
               + "; ".join(f'{e["name"]}[{e["type"]}]' for e in llm_names.values())
               + f"\n\nLOCAL entities ({len(loc_pool)}): "
               + "; ".join(f'{e["name"]}[{e["type"]}]' for e in loc_names.values())
               + f"\n\nLLM entities MISSED by local ({len(missed)}): " + "; ".join(missed)
               + f"\n\nLOCAL extras (no LLM match, {len(extra)}): " + "; ".join(extra[:60])
               + f"\n\nLLM relations ({len(s['relations'])}):\n  "
               + "\n  ".join(f'{r["source"]} -[{",".join(r["labels"])}]-> {r["target"]}'
                             + (f' (status={r["status"]})' if r.get("status") == "ended" else "")
                             + (f' (from={r["valid_from"]})' if r.get("valid_from") else "")
                             + (f' (to={r["valid_to"]})' if r.get("valid_to") else "")
                             for r in s["relations"])
               + f"\n\nLLM relations with NO local pair ({len(rel_missed)}):\n  "
               + "\n  ".join(rel_missed)
               + f"\n\nLOCAL relations ({len(L['relations'])}):\n  "
               + "\n  ".join(f'{r["source"]} -[{r["label"]}]-> {r["target"]} ({r["score"]:.2f})'
                             for r in L["relations"][:80]))

summary = {
    "micro_ent_recall_exact": round(tot["exact"] / tot["llm_ents"], 3),
    "micro_ent_recall_incl_fuzzy": round((tot["exact"] + tot["fuzzy"]) / tot["llm_ents"], 3),
    "micro_ent_precision": round(tot["loc_matched"] / tot["loc_ents"], 3),
    "total_llm_ents": tot["llm_ents"], "total_loc_ents": tot["loc_ents"],
    "micro_rel_pair_coverage": round(tot["rel_hit"] / tot["llm_rels"], 3),
    "total_llm_rels": tot["llm_rels"], "total_loc_rels": tot["loc_rels"],
    "llm_temporal_rels": tot["llm_temporal"],
    "temporal_rels_with_local_pair": tot["temporal_hit"],
}
json.dump({"summary": summary, "per_session": rows},
          open(os.path.join(HERE, f"scores{SUFFIX}.json"), "w", encoding="utf-8"), indent=1)
open(os.path.join(HERE, f"side_by_side{SUFFIX}.txt"), "w", encoding="utf-8").write("\n".join(sbs))
print(json.dumps(summary, indent=1))
for r in rows:
    print(r)
