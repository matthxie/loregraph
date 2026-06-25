"""Retrieval (docs/ARCHITECTURE.md §5) — a pluggable, multi-mode retriever.

This is the **retrieve-then-read** half of the system: a cheap index picks entry nodes,
then graph structure (not an LLM) does the multi-hop work. Seeding fuses signals over the
IMMUTABLE layer — episode-text embeddings, BM25 over episode text, mention embeddings, and
query-entity/tag links — into seed nodes (the entry-point index, not the answer). Then one
traversal mode spreads from the seeds over the entity/episode subgraph:

  * PPRRetriever    — Personalized PageRank seed-and-spread + MMR/node-distance rerank
                      (the primary path; HippoRAG's no-LLM-in-the-hop-loop diffusion).
  * BFSRetriever    — plain n-hop BFS over the symmetrized projection (the A/B baseline).
  * VectorRetriever — flat episode-embedding top-k, no graph ("does the graph help?").

Fact edges are filtered to a temporal view before diffusion: `as_of=None` keeps only
currently-valid facts (the default current view); `as_of=T` keeps facts whose valid window
contained T (point-in-time retrieval). Structural edges (mention/episode/tag) are timeless.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .canonicalize import Canonicalizer, normalize_key
from .config import Config
from .embedders import Embedder
from .models import EdgeType, NodeType, Provenance
from .store import GraphStore, fact_active

_TOK = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievalResult:
    query: str
    mode: str
    objects: list[tuple[str, float]] = field(default_factory=list)  # (episode_id, score)
    seeds: list[str] = field(default_factory=list)
    subgraph: set[str] = field(default_factory=set)   # every node touched (recall@k)
    as_of: str | None = None

    @property
    def object_ids(self) -> list[str]:
        return [oid for oid, _ in self.objects]


# --------------------------------------------------------------------------- #
# Seeding (shared by every mode)
# --------------------------------------------------------------------------- #
class Seeder:
    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config):
        self.store = store
        self.embedder = embedder
        self.canon = canon
        self.config = config
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_corpus: list[list[str]] = []

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        corpus, ids = [], []
        for n in self.store.nodes_of_type(NodeType.EPISODE):
            surface = n.raw_text or n.description or n.name or ""
            corpus.append(_TOK.findall(surface.lower()))
            ids.append(n.id)
        self._bm25_ids = ids
        self._bm25_corpus = corpus
        if not corpus:
            self._bm25 = False
            return
        try:
            from rank_bm25 import BM25Okapi
        except Exception:
            self._bm25 = False
            return
        self._bm25 = BM25Okapi(corpus)

    def bm25_search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Top-k (episode_id, max-normalized score) over episode raw-text by lexical
        relevance. BM25 when `rank_bm25` is installed, else a deterministic token-overlap
        scan so lexical search still works fully offline."""
        k = k or self.config.seed_k
        self._ensure_bm25()
        toks = _TOK.findall((query or "").lower())
        if not toks or not self._bm25_ids:
            return []
        if self._bm25:
            scores = self._bm25.get_scores(toks)
        else:
            want = set(toks)
            scores = np.array([sum(1 for t in doc if t in want)
                               for doc in self._bm25_corpus], dtype=np.float64)
        peak = float(scores.max()) if len(scores) else 0.0
        if peak <= 0:
            return []
        scores = scores / peak
        out = []
        for i in np.argsort(-scores)[:k]:
            if scores[i] <= 0:
                break
            out.append((self._bm25_ids[i], float(scores[i])))
        return out

    def seed(self, query: str) -> dict[str, float]:
        """Return {node_id: seed_mass} fusing episode-text + BM25 + mention + entity/tag links."""
        scores: dict[str, float] = {}
        qv = self.embedder.embed([query])[0]

        # (a) episode-text embedding seeds (the primary retrieval surface)
        for oid, cos in self.store.vectors.search("episode", qv, k=self.config.seed_k,
                                                 floor=0.0):
            scores[oid] = max(scores.get(oid, 0.0), float(cos))

        # (b) BM25 keyword seeds (lexical recall the embedding misses)
        for oid, score in self.bm25_search(query, k=self.config.seed_k):
            scores[oid] = max(scores.get(oid, 0.0), score)

        # (c) query-entity / tag linking → seed those anchors (HippoRAG)
        keys = set()
        toks = _TOK.findall(query.lower())
        for n in range(1, 4):  # unigram..trigram surface forms
            for i in range(len(toks) - n + 1):
                keys.add(normalize_key(" ".join(toks[i:i + n])))
        for nid in (self.canon._tag_keys.get(k) for k in keys):
            if nid:
                scores[nid] = max(scores.get(nid, 0.0), 0.6)
        for nid in (self.canon._entity_keys.get(k) for k in keys):
            if nid:
                scores[nid] = max(scores.get(nid, 0.0), 0.6)
        # (d) embedding-matched mentions + tag/entity anchors (synonymy seeds)
        for nid, cos in self.store.vectors.search("mention", qv, k=5, floor=0.6):
            scores[nid] = max(scores.get(nid, 0.0), float(cos) * 0.7)
        for kind in ("tag", "entity"):
            for nid, cos in self.store.vectors.search(kind, qv, k=3, floor=0.6):
                scores[nid] = max(scores.get(nid, 0.0), float(cos) * 0.6)

        return scores


# --------------------------------------------------------------------------- #
# Weighted simple-graph projection (shared by PPR + BFS)
# --------------------------------------------------------------------------- #
# IN_COMMUNITY would turn each CommunityNode into a high-degree hub and distort the spread.
_TRAVERSAL_EXCLUDE = {EdgeType.IN_COMMUNITY.value}


def projected_graph(store: GraphStore, config: Config, *, as_of: str | None = None,
                    exclude_etypes: set[str] | None = None) -> nx.Graph:
    """Undirected, weighted projection of the directed store (HippoRAG runs PPR
    undirected). Fact (RELATED_TO) edges are filtered to the requested temporal view
    (`as_of=None` → current; `as_of=T` → as-of-T); structural edges are timeless."""
    exclude = _TRAVERSAL_EXCLUDE if exclude_etypes is None else exclude_etypes
    G = nx.Graph()
    for n in store.nodes.values():
        if n.valid:
            G.add_node(n.id)
    pair_w: dict[tuple, dict[str, float]] = {}
    for u, v, data in store.all_edges():
        if not data.get("valid", True):
            continue
        et = data["etype"]
        if et in exclude:
            continue
        if et == EdgeType.RELATED_TO.value and not fact_active(data, as_of):
            continue   # superseded / future / retracted facts don't diffuse
        if (data["provenance"] == Provenance.INFERRED.value
                and data["confidence"] < config.inferred_confidence_floor):
            continue
        if u not in G or v not in G:
            continue
        w = max(1e-4, float(data["confidence"]) * float(data["weight"]))
        pair = (u, v) if u <= v else (v, u)        # symmetrize direction
        by_etype = pair_w.setdefault(pair, {})
        if w > by_etype.get(et, 0.0):              # MAX within an etype (parallel facts
            by_etype[et] = w                       # count once); SUM across distinct etypes
    for (u, v), by_etype in pair_w.items():
        G.add_edge(u, v, weight=sum(by_etype.values()))
    return G


# --------------------------------------------------------------------------- #
# Retrievers
# --------------------------------------------------------------------------- #
class PPRRetriever:
    mode = "ppr"

    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config):
        self.store = store
        self.embedder = embedder
        self.canon = canon
        self.config = config
        self.seeder = Seeder(store, embedder, canon, config)

    def retrieve(self, query: str, k: int | None = None,
                 as_of: str | None = None) -> RetrievalResult:
        k = k or self.config.top_k
        if not query or not query.strip():
            return RetrievalResult(query=query, mode=self.mode, as_of=as_of)
        seeds = self.seeder.seed(query)
        res = RetrievalResult(query=query, mode=self.mode, seeds=list(seeds), as_of=as_of)
        if not seeds:
            return res
        G = projected_graph(self.store, self.config, as_of=as_of)
        pers = {}
        for nid, s in seeds.items():
            if nid in G and s > 0:
                pers[nid] = s * self.canon.idf_weight(nid)
        if not pers or sum(pers.values()) <= 0:
            return res
        ppr = nx.pagerank(G, alpha=self.config.ppr_damping, personalization=pers,
                          weight="weight", max_iter=200)
        cand = []
        for nid, sc in ppr.items():
            n = self.store.get_node(nid)
            if n and n.ntype == NodeType.EPISODE and n.valid:
                cand.append((nid, sc))
        cand.sort(key=lambda x: -x[1])
        cand = cand[: max(k * 3, k)]
        ranked = self._rerank(query, cand, seeds, G, k)
        res.objects = ranked
        res.subgraph = set(seeds) | {oid for oid, _ in ranked}
        return res

    def _rerank(self, query, cand, seeds, G, k):
        """MMR diversity + node-distance to seeds on top of PPR scores (graphiti)."""
        if not cand:
            return []
        seed_objs = [s for s in seeds if s in G]
        dist = self._seed_distances(G, seed_objs, cutoff=5)
        dist_boost = {oid: 1.0 + 1.0 / (1.0 + dist.get(oid, 6.0)) for oid, _ in cand}
        base = {oid: sc * dist_boost[oid] for oid, sc in cand}
        vecs = {oid: self.store.vectors.get("episode", oid) for oid, _ in cand}
        selected: list[tuple[str, float]] = []
        selected_vecs: list = []
        remaining = [oid for oid, _ in cand]
        lam = self.config.mmr_lambda
        while remaining and len(selected) < k:
            best, best_score = None, -1e9
            for oid in remaining:
                div = 0.0
                ov = vecs[oid]
                if selected_vecs and ov is not None:
                    div = max((float(np.dot(ov, sv)) for sv in selected_vecs
                               if sv is not None), default=0.0)
                mmr = lam * base[oid] - (1 - lam) * div
                if mmr > best_score:
                    best, best_score = oid, mmr
            selected.append((best, round(base[best], 6)))
            if vecs[best] is not None:
                selected_vecs.append(vecs[best])
            remaining.remove(best)
        return selected

    @staticmethod
    def _seed_distances(G, seeds, cutoff: int = 5) -> dict:
        dist: dict[str, float] = {}
        for s in seeds:
            if s not in G:
                continue
            lengths = nx.single_source_shortest_path_length(G, s, cutoff=cutoff)
            for node, d in lengths.items():
                if d < dist.get(node, float("inf")):
                    dist[node] = d
        return dist


class BFSRetriever:
    mode = "bfs"

    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config, max_hops: int = 2):
        self.store = store
        self.embedder = embedder
        self.canon = canon
        self.config = config
        self.max_hops = max_hops
        self.seeder = Seeder(store, embedder, canon, config)

    def retrieve(self, query: str, k: int | None = None,
                 as_of: str | None = None) -> RetrievalResult:
        k = k or self.config.top_k
        if not query or not query.strip():
            return RetrievalResult(query=query, mode=self.mode, as_of=as_of)
        seeds = self.seeder.seed(query)
        res = RetrievalResult(query=query, mode=self.mode, seeds=list(seeds), as_of=as_of)
        if not seeds:
            return res
        G = projected_graph(self.store, self.config, as_of=as_of)
        scored: dict[str, float] = {}
        for seed, mass in seeds.items():
            if seed not in G:
                continue
            frontier = {seed}
            seen_local = {seed}
            for hop in range(self.max_hops + 1):
                nxt = set()
                for nid in frontier:
                    node = self.store.get_node(nid)
                    if node and node.ntype == NodeType.EPISODE and node.valid:
                        scored[nid] = scored.get(nid, 0.0) + mass / (1 + hop)
                    for nbr in G.neighbors(nid):
                        if nbr not in seen_local:
                            seen_local.add(nbr)
                            nxt.add(nbr)
                frontier = nxt
                res.subgraph |= seen_local
        ranked = sorted(scored.items(), key=lambda x: -x[1])[:k]
        res.objects = ranked
        res.subgraph |= set(seeds)
        return res


class VectorRetriever:
    mode = "vector"

    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config):
        self.store = store
        self.embedder = embedder
        self.config = config

    def retrieve(self, query: str, k: int | None = None,
                 as_of: str | None = None) -> RetrievalResult:
        k = k or self.config.top_k
        if not query or not query.strip():
            return RetrievalResult(query=query, mode=self.mode, as_of=as_of)
        qv = self.embedder.embed([query])[0]
        invalid = {n.id for n in self.store.nodes_of_type(NodeType.EPISODE, valid_only=False)
                   if not n.valid}
        hits = self.store.vectors.search("episode", qv, k=k, floor=1e-6, exclude=invalid)
        res = RetrievalResult(query=query, mode=self.mode, as_of=as_of,
                             objects=[(i, float(s)) for i, s in hits],
                             seeds=[i for i, _ in hits])
        res.subgraph = set(res.object_ids)
        return res


def get_retriever(mode: str, store, embedder, canon, config):
    mode = (mode or "ppr").lower()
    if mode == "ppr":
        return PPRRetriever(store, embedder, canon, config)
    if mode == "bfs":
        return BFSRetriever(store, embedder, canon, config)
    if mode in ("vector", "flat"):
        return VectorRetriever(store, embedder, canon, config)
    raise ValueError(f"unknown retrieval mode: {mode}")
