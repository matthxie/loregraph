"""Load the LongMemEval test corpus from dataset/longmemeval/ (see that folder's README).

LongMemEval (Wu et al., ICLR'25) is a long-term-memory benchmark over dated, multi-session
chats. Each *instance* is one user's chat history (a haystack of dated sessions) + a
question + the gold answer + which sessions hold the evidence. We derive three tiers
(small / med / large) plus a tiny committed `sample`; build them with
`python scripts/build_longmemeval.py`. Here each *session* becomes one ingestible
episode (chronologically ordered, its timestamp threaded into `created_at`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dataset")
LME_DIR = os.path.join(DATASET_DIR, "longmemeval")


@dataclass
class CorpusItem:
    id: str
    modality: str          # "text" | "image"
    source_ref: str        # url / file path
    title: str = ""
    text: str | None = None
    image_path: str | None = None
    label_hint: str | None = None  # COCO labels — offline VLM stand-in
    created_at: str | None = None  # corpus item's own time (session date); None → wall clock


def _lme_tier_dir(tier: str) -> str:
    d = os.path.join(LME_DIR, tier)
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "episodes.jsonl")):
        raise FileNotFoundError(
            f"LongMemEval tier {tier!r} is not built (looked in {d}). Run "
            f"`python scripts/build_longmemeval.py --tier {tier}` first — episode bodies "
            f"are gitignored / regenerable (only questions.jsonl + manifest.json are "
            f"committed). The committed `sample` tier ships its episodes for offline tests.")
    return d


def _episode_item(r: dict, tier: str) -> CorpusItem:
    """One LongMemEval session row → an ingestible text episode. `created_at` carries the
    session's timestamp so the bi-temporal layer orders facts by real chat time."""
    return CorpusItem(
        id=r["id"], modality="text",
        source_ref=f"longmemeval/{tier}/{r['question_id']}/{r['session_id']}",
        title="", text=r["text"], created_at=r.get("created_at"))


def load_longmemeval(tier: str = "sample", question_id: str | None = None,
                     limit: int | None = None) -> list[CorpusItem]:
    """Load a LongMemEval tier's sessions as episodes (one CorpusItem per session, in
    chronological order). Pass `question_id` to load just one instance's haystack — the
    unit the benchmark scores against (see dataset/longmemeval/README.md §Consumption)."""
    if limit is not None and limit <= 0:                 # --limit 0 means none, not all
        return []
    path = os.path.join(_lme_tier_dir(tier), "episodes.jsonl")
    items: list[CorpusItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if question_id is not None and r["question_id"] != question_id:
                continue
            items.append(_episode_item(r, tier))
            if limit and len(items) >= limit:
                break
    return items


def load_longmemeval_questions(tier: str = "sample", limit: int | None = None) -> list[dict]:
    """Load a tier's graded queries: {id, query, kind, question_date, gold, answer,
    abstention, n_evidence, n_sessions, source}. `gold` lists evidence episode ids."""
    if limit is not None and limit <= 0:                 # --queries 0 means none, not all
        return []
    path = os.path.join(_lme_tier_dir(tier), "questions.jsonl")
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def iter_lme_instances(tier: str = "sample", limit: int | None = None):
    """Yield (question, [session CorpusItems]) per instance — the correct LongMemEval
    protocol: each question is answered against ONLY its own haystack in a fresh graph,
    so the 500 personas don't cross-contaminate one shared memory. See the dataset README."""
    by_q: dict[str, list[CorpusItem]] = {}
    path = os.path.join(_lme_tier_dir(tier), "episodes.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_q.setdefault(r["question_id"], []).append(_episode_item(r, tier))
    for q in load_longmemeval_questions(tier, limit=limit):
        yield q, by_q.get(q["id"], [])
