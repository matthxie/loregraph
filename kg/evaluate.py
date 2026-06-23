"""Evaluation harness (docs/ARCHITECTURE.md §6 / phase 4).

Two objective tiers, both with *programmatically derived* ground truth (so they run
on the random Wikipedia corpus without hand-authoring):

  * single-article  — query built from an article's lead; gold = that article.
                      Tests seed quality.
  * cross-article   — query = an entity shared by ≥2 articles; gold = every article
                      mentioning it. Tests whether traversal pulls in the connected
                      articles a flat vector search would miss (the central thesis).

Metrics: recall@k and MRR, reported per retrieval mode (PPR vs BFS vs vector) — the
key ablation. A hand-authored JSONL question file ({"query":..,"gold":[ids]}) is also
supported via `load_questions`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .graph import KnowledgeGraph
from .models import EdgeType, NodeType

_STOP = set("the a an of to in on and or for with from this that".split())
_W = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")


@dataclass
class Question:
    query: str
    gold: set[str]
    kind: str = "single"


@dataclass
class ModeScore:
    mode: str
    n: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    per_kind: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.mode:8s}  recall@k={self.recall_at_k:.3f}  mrr={self.mrr:.3f}  (n={self.n})"


# --------------------------------------------------------------------------- #
# Question generation
# --------------------------------------------------------------------------- #
def single_article_questions(g: KnowledgeGraph, limit: int | None = None) -> list[Question]:
    qs = []
    for n in g.store.nodes_of_type(NodeType.OBJECT):
        if n.modality and n.modality.value == "image":
            continue
        text = n.raw_text or ""
        lead = text[:300]
        # query = salient words from the lead, minus the title tokens (don't gift the answer)
        title_toks = {t.lower() for t in _W.findall(n.name)}
        words = [w for w in _W.findall(lead)
                 if w.lower() not in _STOP and w.lower() not in title_toks]
        if len(words) < 5:
            continue
        qs.append(Question(query=" ".join(words[:12]), gold={n.id}, kind="single"))
        if limit and len(qs) >= limit:
            break
    return qs


def cross_article_questions(g: KnowledgeGraph, limit: int | None = None) -> list[Question]:
    """Entities mentioned by ≥2 articles → multi-hop questions (gold = that set)."""
    ent_objs: dict[str, set[str]] = {}
    for n in g.store.nodes_of_type(NodeType.ENTITY):
        objs = set()
        for nbr, data in g.store.neighbors(n.id, etypes={EdgeType.MENTIONS}):
            on = g.store.get_node(nbr)
            if on and on.ntype == NodeType.OBJECT and on.modality.value == "text":
                objs.add(nbr)
        if len(objs) >= 2:
            ent_objs[n.id] = objs
    qs = []
    for eid, objs in sorted(ent_objs.items(), key=lambda kv: -len(kv[1])):
        ent = g.store.get_node(eid)
        qs.append(Question(query=ent.name, gold=set(objs), kind="cross"))
        if limit and len(qs) >= limit:
            break
    return qs


def load_questions(path: str) -> list[Question]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Question(query=r["query"], gold=set(r["gold"]),
                                kind=r.get("kind", "authored")))
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = len(set(ranked[:k]) & gold)
    return hit / len(gold)


def _mrr(ranked: list[str], gold: set[str]) -> float:
    for i, oid in enumerate(ranked):
        if oid in gold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(g: KnowledgeGraph, questions: list[Question],
             modes=("ppr", "bfs", "vector"), k: int = 8) -> list[ModeScore]:
    scores = []
    kinds = sorted({q.kind for q in questions})
    for mode in modes:
        ms = ModeScore(mode=mode)
        agg_r, agg_m = 0.0, 0.0
        per = {kind: {"n": 0, "r": 0.0, "m": 0.0} for kind in kinds}
        for q in questions:
            # "agent" routes through the §5 agentic traversal (offline backend for a
            # deterministic, key-free comparison); its .object_ids (citations ∪ surfaced
            # objects) drops into the same recall@k/MRR math as a RetrievalResult.
            res = (g.ask(q.query, backend="offline", k=k) if mode == "agent"
                   else g.query(q.query, mode=mode, k=k))
            ranked = res.object_ids
            r = _recall_at_k(ranked, q.gold, k)
            m = _mrr(ranked, q.gold)
            agg_r += r
            agg_m += m
            per[q.kind]["n"] += 1
            per[q.kind]["r"] += r
            per[q.kind]["m"] += m
        n = max(1, len(questions))
        ms.n = len(questions)
        ms.recall_at_k = agg_r / n
        ms.mrr = agg_m / n
        ms.per_kind = {kk: {"n": v["n"],
                            "recall_at_k": round(v["r"] / max(1, v["n"]), 3),
                            "mrr": round(v["m"] / max(1, v["n"]), 3)}
                       for kk, v in per.items()}
        scores.append(ms)
    return scores
