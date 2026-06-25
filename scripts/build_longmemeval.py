"""Build the LongMemEval test datasets (small / med / large) for the KG harness.

LongMemEval (Wu et al., 2024 — https://github.com/xiaowu0162/longmemeval, ICLR'25)
is a benchmark for the long-term memory of chat assistants: each of its **500
hand-curated question instances** is one user's long, multi-session chat *history*
(a "haystack" of dated sessions, mostly distractors) plus a question asked at a later
date, the gold answer, and a pointer to which sessions actually hold the evidence. It
stresses exactly the five abilities this episodic, bi-temporal graph is built for —
information extraction, multi-session reasoning, **temporal reasoning**, **knowledge
updates**, and **abstention**. It is, in effect, the real graded version of the tiny
hand-built `kg.synthetic` Becky stream.

We derive three tiers from the published, cleaned release on Hugging Face
(`xiaowu0162/longmemeval-cleaned`, **MIT license**). The S and M variants share the
*same* 500 questions; they differ only in how many distractor sessions pad each
haystack (S ≈ 50 sessions / ~115k tokens, M ≈ 500 sessions / ~1.5M tokens):

    tier    questions   haystack/instance   source variant       =
    small   100         ~50 sessions        longmemeval_s        quick-iteration subset
    med     500 (all)   ~50 sessions        longmemeval_s        the full LongMemEval_S
    large   500 (all)   ~500 sessions       longmemeval_m        the deep LongMemEval_M

Why not "1000": there are only 500 distinct questions in existence, so we honour
small=100 / med=500 and let *large* scale the way LongMemEval itself does — by history
**depth** (the M variant), not by question count.

ORDERING. Conversations are time-ordered and must not be shuffled: within an instance
the sessions are emitted **sorted by their timestamp** (34/500 haystacks ship out of
chronological order), so each instance reads as a clean episode stream the bi-temporal
layer can order. *Across* instances the 500 are independent, so tier membership is
chosen by a **deterministic, RNG-free stratified order** (round-robin over the six
question types) and tiers **nest**: small's 100 question_ids ⊂ med's/large's 500.

OUTPUT (per tier, under dataset/longmemeval/<tier>/):
  episodes.jsonl   one row per *session* = one ingestible episode (GITIGNORED — heavy,
                   regenerable from this script). Fields: id, question_id, session_id,
                   created_at (ISO, store.now_iso format), date, modality, is_evidence,
                   n_turns, text (the session's turns rendered as User/Assistant prose).
  questions.jsonl  one row per *instance* = the graded query (COMMITTED — tiny). Fields:
                   id, query, kind (question_type), question_date, gold (evidence episode
                   ids as obj_<question_id>__<session_id>, matching the harness's
                   `_article` collapse), answer, abstention, n_evidence, n_sessions, source.
  manifest.json    tier metadata + provenance + the ordered question_id list (COMMITTED).

Usage:
    python scripts/build_longmemeval.py                 # build small + med (from S, ~277MB)
    python scripts/build_longmemeval.py --tier large    # build large (downloads M, ~2.74GB)
    python scripts/build_longmemeval.py --tier all
    python scripts/build_longmemeval.py --keep-cache    # don't delete the raw HF download

Episode bodies are deliberately not committed (see .gitignore) — re-run this to
materialise them. Only the lean questions.jsonl + manifest.json are version-controlled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

HF_REPO = "xiaowu0162/longmemeval-cleaned"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
LICENSE = "MIT"
PAPER = "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (Wu et al., ICLR 2025; arXiv:2410.10813)"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, ".cache", "longmemeval")
OUT_DIR = os.path.join(ROOT, "dataset", "longmemeval")

# tier -> (source variant file, n_instances or None for all, session cap or None)
# The `sample` tier is a tiny, COMMITTED offline fixture (the others gitignore their
# episode bodies): 8 instances, each haystack capped to a few sessions (evidence always
# kept) so the whole thing is a few hundred KB — enough for `kg testrun` smoke runs and
# the offline dashboard test without the 277 MB download.
VARIANTS = {
    "sample": ("longmemeval_s_cleaned.json", 8, 6),
    "small":  ("longmemeval_s_cleaned.json", 100, None),
    "med":    ("longmemeval_s_cleaned.json", None, None),
    "large":  ("longmemeval_m_cleaned.json", None, None),
}
QUESTION_TYPES = [
    "single-session-user", "single-session-assistant", "single-session-preference",
    "temporal-reasoning", "knowledge-update", "multi-session",
]


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
def parse_lme_date(s: str) -> datetime:
    """'2023/04/10 (Mon) 23:07' -> aware UTC datetime. The '(Mon)' day-of-week token is
    redundant (the date fixes it) and is intentionally ignored, not validated."""
    head, _, tail = s.partition(" (")
    clock = tail.split(") ", 1)[1] if ") " in tail else tail
    return datetime.strptime(f"{head} {clock}", "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    """Match kg.store.now_iso(): isoformat at seconds resolution, UTC offset."""
    return dt.isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Rendering + ordering
# --------------------------------------------------------------------------- #
def render_session(turns: list[dict], date_str: str) -> str:
    """A chat session -> readable prose for the extractor. Roles are user/assistant."""
    lines = [f"[chat session — {date_str}]"]
    for t in turns:
        role = (t.get("role") or "user").strip().lower()
        who = "User" if role == "user" else "Assistant" if role == "assistant" else role.title()
        lines.append(f"{who}: {(t.get('content') or '').strip()}")
    return "\n".join(lines)


def canonical_order(instances: list[dict]) -> list[dict]:
    """Deterministic, RNG-free stratified order: round-robin over question types so any
    prefix preserves the type mix; tie-break by question_id. Tiers nest because a smaller
    tier is just a prefix of this single global order."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for inst in instances:
        by_type[inst["question_type"]].append(inst)
    keyed = []
    for qtype, group in by_type.items():
        group.sort(key=lambda i: i["question_id"])
        n = len(group)
        for rank, inst in enumerate(group):
            # evenly spaced position in [0,1) -> interleaves groups proportionally
            keyed.append(((rank + 0.5) / n, qtype, inst["question_id"], inst))
    keyed.sort(key=lambda x: (x[0], x[1], x[2]))
    return [x[3] for x in keyed]


def sorted_sessions(inst: dict, cap: int | None = None) -> list[tuple[str, str, list[dict]]]:
    """(session_id, date_str, turns) for this instance, sorted chronologically by date.

    haystack_sessions / haystack_session_ids / haystack_dates are parallel arrays; some
    instances ship them out of chronological order, so we sort here. Stable on the
    original index for any same-minute ties.

    `cap` (the tiny `sample` tier only) keeps the haystack to `cap` sessions while ALWAYS
    retaining every evidence session, so the gold answer stays reachable. These don't
    conflict for our data: max evidence/instance is 6 and the sample cap is 6, so evidence
    never exceeds the cap (kept-count = max(#evidence, cap), bounded by #sessions)."""
    rows = list(zip(inst["haystack_session_ids"], inst["haystack_dates"],
                    inst["haystack_sessions"], range(len(inst["haystack_sessions"]))))
    rows.sort(key=lambda r: (parse_lme_date(r[1]), r[3]))
    if cap is not None and len(rows) > cap:
        evidence = set(inst.get("answer_session_ids") or [])
        keep = [r for r in rows if r[0] in evidence]
        for r in rows:
            if len(keep) >= cap:
                break
            if r[0] not in evidence:
                keep.append(r)
        rows = sorted(keep, key=lambda r: (parse_lme_date(r[1]), r[3]))
    return [(sid, date, turns) for sid, date, turns, _ in rows]


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def download(filename: str) -> str:
    """Stream a variant file from Hugging Face into .cache/ (skip if present)."""
    import requests
    os.makedirs(CACHE_DIR, exist_ok=True)
    dest = os.path.join(CACHE_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cache hit: {dest} ({os.path.getsize(dest)/1e6:.0f} MB)")
        return dest
    url = f"{HF_BASE}/{filename}"
    print(f"  downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r    {got/1e6:6.0f} / {total/1e6:.0f} MB", end="", flush=True)
        print()
    return dest


def build_tier(tier: str, instances: list[dict], order: list[dict]) -> dict:
    """Emit episodes.jsonl + questions.jsonl + manifest.json for one tier from the
    canonical `order` (a prefix of it for sub-full tiers)."""
    _, n, cap = VARIANTS[tier]
    chosen = order if n is None else order[:n]
    by_id = {i["question_id"]: i for i in instances}
    tier_dir = os.path.join(OUT_DIR, tier)
    os.makedirs(tier_dir, exist_ok=True)

    n_episodes = 0
    types: Counter = Counter()
    abst = 0
    qid_list: list[str] = []
    ep_path = os.path.join(tier_dir, "episodes.jsonl")
    q_path = os.path.join(tier_dir, "questions.jsonl")
    with open(ep_path, "w", encoding="utf-8") as epf, open(q_path, "w", encoding="utf-8") as qf:
        for sel in chosen:
            inst = by_id[sel["question_id"]]
            qid = inst["question_id"]
            evidence = set(inst.get("answer_session_ids") or [])
            sessions = sorted_sessions(inst, cap=cap)
            for sid, date_str, turns in sessions:
                ep_id = f"{qid}__{sid}"
                epf.write(json.dumps({
                    "id": ep_id, "question_id": qid, "session_id": sid,
                    "created_at": iso(parse_lme_date(date_str)), "date": date_str,
                    "modality": "text", "is_evidence": sid in evidence,
                    "n_turns": len(turns), "text": render_session(turns, date_str),
                }, ensure_ascii=False) + "\n")
                n_episodes += 1
            is_abst = qid.endswith("_abs")
            abst += int(is_abst)
            types[inst["question_type"]] += 1
            qid_list.append(qid)
            qf.write(json.dumps({
                "id": qid, "query": inst["question"], "kind": inst["question_type"],
                "question_date": iso(parse_lme_date(inst["question_date"])),
                "question_date_raw": inst["question_date"],
                # gold = the evidence sessions, as obj_<id> so testrun._article collapses
                # them onto the ingested ep_<id> episode ids for recall@k / MRR.
                # dict.fromkeys = order-preserving dedup (defensive; deterministic — a set
                # would reorder). The source has no dup evidence ids today, but stay robust.
                "gold": [f"obj_{qid}__{sid}"
                         for sid in dict.fromkeys(inst["answer_session_ids"])],
                # 32/500 answers are integers (counting/temporal Qs) — keep them as strings
                "answer": str(inst["answer"]), "abstention": is_abst,
                "n_evidence": len(evidence), "n_sessions": len(sessions),
                "source": VARIANTS[tier][0].replace("_cleaned.json", ""),
            }, ensure_ascii=False) + "\n")

    manifest = {
        "tier": tier,
        "source_variant": VARIANTS[tier][0],
        "hf_repo": HF_REPO, "license": LICENSE, "paper": PAPER,
        "n_instances": len(chosen), "n_episodes": n_episodes,
        "question_types": dict(sorted(types.items())),
        "abstention": abst,
        "session_cap": cap,
        "selection": "deterministic stratified round-robin over question_type "
                     "(RNG-free); tiers nest as prefixes of one global order",
        "ordering": "sessions emitted chronologically (sorted by timestamp)"
                    + ("" if cap is None else
                       f"; haystacks capped to {cap} sessions (evidence kept)"),
        "built_at": iso(datetime.now(timezone.utc)),
        "question_ids": qid_list,
        "note": "episodes.jsonl is gitignored — regenerate via "
                f"`python scripts/build_longmemeval.py --tier {tier}`",
    }
    with open(os.path.join(tier_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  {tier:5s}: {len(chosen):3d} instances, {n_episodes:5d} episodes, "
          f"{abst} abstention -> {tier_dir}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build LongMemEval small/med/large tiers.")
    ap.add_argument("--tier", choices=["sample", "small", "med", "large", "all", "default"],
                    default="default",
                    help="default = sample+small+med (S only); large pulls the 2.74GB M")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the raw HF download in .cache/longmemeval/")
    args = ap.parse_args()

    if args.tier == "default":
        tiers = ["sample", "small", "med"]
    elif args.tier == "all":
        tiers = ["sample", "small", "med", "large"]
    else:
        tiers = [args.tier]

    # group tiers by source variant so each variant is loaded at most once
    used_files = sorted({VARIANTS[t][0] for t in tiers})
    loaded: dict[str, list[dict]] = {}
    canon: dict[str, list[dict]] = {}
    for fn in used_files:
        big = " (~2.74 GB — this is the heavy one)" if "_m_" in fn else ""
        print(f"variant {fn}{big}")
        path = download(fn)
        print(f"  loading {path} ...")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) != 500:
            print(f"  WARNING: expected 500 instances, got {len(data)}", file=sys.stderr)
        loaded[fn] = data
        canon[fn] = canonical_order(data)

    os.makedirs(OUT_DIR, exist_ok=True)
    print("building tiers ...")
    for t in tiers:
        fn = VARIANTS[t][0]
        build_tier(t, loaded[fn], canon[fn])

    if not args.keep_cache:
        for fn in used_files:
            p = os.path.join(CACHE_DIR, fn)
            if os.path.exists(p):
                os.remove(p)
                print(f"removed raw download {p}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
