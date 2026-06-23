"""Retrieval (docs/ARCHITECTURE.md §5) — a pluggable, multi-mode retriever.

Seeding fuses three signals (rev 2): object-text embeddings, BM25 keyword search,
and query-entity/concept linking → seed nodes (the *entry-point index*, not the answer).
Then one of three traversal modes spreads from the seeds:

  * PPRRetriever    — Personalized PageRank seed-and-spread + MMR/node-distance
                      rerank (the primary path / central bet).
  * BFSRetriever    — plain n-hop BFS over the symmetrized projection (the A/B baseline
                      PPR must beat at this scale; uses the `seen` visited-set flag).
  * VectorRetriever — flat object-embedding top-k, no graph (the "does the graph
                      even help?" baseline).

All return a RetrievalResult so the eval harness can compare them apples-to-apples.
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
from .store import GraphStore

_TOK = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievalResult:
    query: str
    mode: str
    objects: list[tuple[str, float]] = field(default_factory=list)  # (object_id, score)
    seeds: list[str] = field(default_factory=list)
    subgraph: set[str] = field(default_factory=set)   # every node touched (for recall@k)

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
        for n in self.store.nodes_of_type(NodeType.OBJECT):
            surface = n.raw_text or n.description or n.name or ""
            corpus.append(_TOK.findall(surface.lower()))
            ids.append(n.id)
        self._bm25_ids = ids
        self._bm25_corpus = corpus     # kept for the offline token-overlap fallback
        if not corpus:
            self._bm25 = False
            return
        try:
            from rank_bm25 import BM25Okapi
        except Exception:
            self._bm25 = False         # rank_bm25 absent → bm25_search uses the fallback
            return
        self._bm25 = BM25Okapi(corpus)

    def bm25_search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Top-k (object_id, max-normalized score) over object raw-text by lexical
        relevance. Shared by `seed()` step (b) and the agent's `keyword_search` tool
        so the two can't drift. Uses BM25 when `rank_bm25` is installed; otherwise
        degrades to a deterministic token-overlap scan so lexical search still works
        fully offline."""
        k = k or self.config.seed_k
        self._ensure_bm25()
        toks = _TOK.findall((query or "").lower())
        if not toks or not self._bm25_ids:
            return []
        if self._bm25:
            scores = self._bm25.get_scores(toks)
        else:                          # offline fallback: count query-token hits per doc
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
        """Return {node_id: seed_mass} fusing embedding + BM25 + entity/concept links."""
        scores: dict[str, float] = {}
        qv = self.embedder.embed([query])[0]

        # (a) object-text embedding seeds
        for oid, cos in self.store.vectors.search("object", qv, k=self.config.seed_k,
                                                 floor=0.0):
            scores[oid] = max(scores.get(oid, 0.0), float(cos))

        # (b) BM25 keyword seeds (lexical recall the embedding misses)
        for oid, score in self.bm25_search(query, k=self.config.seed_k):
            scores[oid] = max(scores.get(oid, 0.0), score)

        # (c) query-entity linking → seed those nodes too (HippoRAG). The entity vocabulary
        # is unified: named entities AND concepts (the former "tags") both live here.
        keys = set()
        toks = _TOK.findall(query.lower())
        for n in range(1, 4):  # unigram..trigram surface forms
            for i in range(len(toks) - n + 1):
                keys.add(normalize_key(" ".join(toks[i:i + n])))
        for nid in (self.canon._entity_keys.get(k) for k in keys):
            if nid:
                scores[nid] = max(scores.get(nid, 0.0), 0.6)
        # embedding-matched entities/concepts (synonymy seeds)
        for nid, cos in self.store.vectors.search("entity", qv, k=3, floor=0.6):
            scores[nid] = max(scores.get(nid, 0.0), float(cos) * 0.6)

        return scores


# --------------------------------------------------------------------------- #
# Weighted simple-graph projection (shared by PPR + BFS)
# --------------------------------------------------------------------------- #
# Edges PPR/BFS diffusion must NOT spread over. IN_COMMUNITY would turn each
# CommunityNode into a high-degree hub and distort the spread (§5 scopes traversal
# to the entity/tag/object edge set, not the community layer).
_TRAVERSAL_EXCLUDE = {EdgeType.IN_COMMUNITY.value}


def projected_graph(store: GraphStore, config: Config,
                    exclude_etypes: set[str] | None = None) -> nx.Graph:
    # Deliberately UNDIRECTED (nx.Graph): the store is directed (§2 rev 3) so
    # relationship semantics survive, but diffusion runs over a symmetrized
    # projection — HippoRAG's PPR and GraphRAG both run undirected, and "find
    # related content" wants to traverse a relationship both ways. Storing
    # direction while symmetrizing here is the standard store-directed /
    # symmetrize-for-diffusion split; collapsing src→dst and dst→src into one
    # weighted edge keeps PPR/BFS recall identical to the pre-rev-3 graph.
    exclude = _TRAVERSAL_EXCLUDE if exclude_etypes is None else exclude_etypes
    G = nx.Graph()
    for n in store.nodes.values():
        if n.valid:
            G.add_node(n.id)
    # Accumulate per undirected pair as {etype: max weight}. Taking the MAX within an
    # etype (then summing ACROSS etypes) means N parallel RELATED_TO relations between
    # a pair (rev 4 — is_friend_of + works_with) count once, not N times — a verbose
    # LLM can't inflate diffusion weight (the research's per-pair-normalization point).
    # Distinct signals (SHARED_ENTITY + SIMILAR_TO) still combine.
    pair_w: dict[tuple, dict[str, float]] = {}
    for u, v, data in store.all_edges():
        if not data.get("valid", True):
            continue
        if data["etype"] in exclude:
            continue
        if (data["provenance"] == Provenance.INFERRED.value
                and data["confidence"] < config.inferred_confidence_floor):
            continue
        if u not in G or v not in G:
            continue
        w = max(1e-4, float(data["confidence"]) * float(data["weight"]))
        pair = (u, v) if u <= v else (v, u)        # symmetrize direction
        et = data["etype"]
        by_etype = pair_w.setdefault(pair, {})
        if w > by_etype.get(et, 0.0):
            by_etype[et] = w
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

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.config.top_k
        if not query or not query.strip():
            return RetrievalResult(query=query, mode=self.mode)
        seeds = self.seeder.seed(query)
        res = RetrievalResult(query=query, mode=self.mode, seeds=list(seeds))
        if not seeds:
            return res
        G = projected_graph(self.store, self.config)
        # personalization reweighted by node specificity (IDF); drop zero-mass seeds
        # so nx.pagerank never sees an all-zero personalization vector (ZeroDivisionError)
        pers = {}
        for nid, s in seeds.items():
            if nid in G and s > 0:
                pers[nid] = s * self.canon.idf_weight(nid)
        if not pers or sum(pers.values()) <= 0:
            return res
        ppr = nx.pagerank(G, alpha=self.config.ppr_damping, personalization=pers,
                          weight="weight", max_iter=200)
        # rank OBJECT nodes (single get_node lookup per node)
        cand = []
        for nid, sc in ppr.items():
            n = self.store.get_node(nid)
            if n and n.ntype == NodeType.OBJECT and n.valid:
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
        # node-distance multiplier from ONE multi-source BFS (cutoff-bounded),
        # instead of a per-(candidate, seed) shortest-path call.
        seed_objs = [s for s in seeds if s in G]
        dist = self._seed_distances(G, seed_objs, cutoff=5)
        dist_boost = {oid: 1.0 + 1.0 / (1.0 + dist.get(oid, 6.0)) for oid, _ in cand}
        base = {oid: sc * dist_boost[oid] for oid, sc in cand}
        # precompute candidate object vectors once (fixed for the whole rerank)
        vecs = {oid: self.store.vectors.get("object", oid) for oid, _ in cand}
        # MMR selection over object embeddings
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
                    div = max(float(np.dot(ov, sv)) for sv in selected_vecs
                              if sv is not None) if any(sv is not None for sv in selected_vecs) else 0.0
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
        """Min hop-distance from any seed to every reachable node (one BFS)."""
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

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.config.top_k
        if not query or not query.strip():
            return RetrievalResult(query=query, mode=self.mode)
        seeds = self.seeder.seed(query)
        res = RetrievalResult(query=query, mode=self.mode, seeds=list(seeds))
        if not seeds:
            return res
        G = projected_graph(self.store, self.config)
        self.store.clear_seen()                       # bulk-clear the visited set (§5)
        scored: dict[str, float] = {}
        for seed, mass in seeds.items():
            if seed not in G:
                continue
            # BFS up to max_hops, decaying by hop distance
            frontier = {seed}
            seen_local = {seed}
            for hop in range(self.max_hops + 1):
                nxt = set()
                for nid in frontier:
                    node = self.store.get_node(nid)
                    if node is not None:
                        node.seen = True
                    if node and node.ntype == NodeType.OBJECT and node.valid:
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

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.config.top_k
        if not query or not query.strip():
            return RetrievalResult(query=query, mode=self.mode)
        qv = self.embedder.embed([query])[0]
        # exclude superseded objects from the flat baseline
        invalid = {n.id for n in self.store.nodes_of_type(NodeType.OBJECT, valid_only=False)
                   if not n.valid}
        hits = self.store.vectors.search("object", qv, k=k, floor=1e-6, exclude=invalid)
        res = RetrievalResult(query=query, mode=self.mode,
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
