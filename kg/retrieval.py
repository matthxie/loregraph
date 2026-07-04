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
import weakref
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .canonicalize import Canonicalizer, normalize_key
from .config import Config
from .embedders import Embedder
from .facts import FactIndex
from .models import SELF_ENTITY_ID, EdgeType, NodeType, Provenance
from .profiler import span as prof_span
from .rerank import CrossEncoderReranker
from .route import MULTIHOP, RECENCY, STATE, route
from .store import GraphStore, fact_active

_TOK = re.compile(r"[a-z0-9]+")
_WSP = re.compile(r"\s+")


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
        self._bm25_version: int = -1

    def _ensure_bm25(self):
        # Rebuild only when the EPISODE set has changed since the last build (episode
        # text is immutable, so nothing else can stale the corpus). Retrievers are
        # long-lived (hoisted onto KnowledgeGraph), so across queries this is a no-op.
        version = getattr(self.store, "episode_version", None)
        if self._bm25 is not None and version == self._bm25_version:
            return
        self._bm25_version = version
        corpus, ids = [], []
        for n in sorted(self.store.nodes_of_type(NodeType.EPISODE), key=lambda n: n.id):
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
        # ids are pre-sorted (corpus build), so argsort's stable ties are id-ordered
        for i in np.argsort(-scores, kind="stable")[:k]:
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

# Projection cache, keyed by store identity (weak: a dropped store frees its cache),
# holding {params_key: graph} for ONE store version at a time. Building the projection
# is O(all edges) — the single biggest per-query cost — but it only changes when the
# store does, so across queries this makes it O(1).
_PROJ_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def projected_graph(store: GraphStore, config: Config, *, as_of: str | None = None,
                    exclude_etypes: set[str] | None = None) -> nx.Graph:
    """Undirected, weighted projection of the directed store (HippoRAG runs PPR
    undirected). Fact (RELATED_TO) edges are filtered to the requested temporal view
    (`as_of=None` → current; `as_of=T` → as-of-T); structural edges are timeless.
    Cached per (store.version, parameters); rebuilt only after the graph mutates."""
    exclude = _TRAVERSAL_EXCLUDE if exclude_etypes is None else exclude_etypes
    version = getattr(store, "version", None)
    key = (as_of, frozenset(exclude), getattr(config, "self_guard", "none"),
           float(getattr(config, "self_guard_cap", 0.05)),
           config.inferred_confidence_floor)
    cached_version, views = _PROJ_CACHE.get(store, (None, None))
    if views is not None and cached_version == version and key in views:
        return views[key]
    G = _build_projection(store, config, as_of=as_of, exclude=exclude)
    if cached_version != version:
        views = {}                        # store moved → every cached view is stale
        _PROJ_CACHE[store] = (version, views)
    views[key] = G
    return G


def _build_projection(store: GraphStore, config: Config, *, as_of: str | None,
                      exclude: set[str]) -> nx.Graph:
    self_guard = getattr(config, "self_guard", "none")
    self_cap = float(getattr(config, "self_guard_cap", 0.05))
    # Deterministic construction (sorted nodes / sorted pair accumulation): the
    # projection — and everything downstream of it, PPR float sums included — is a
    # function of graph CONTENT, not of node/edge insertion or load order.
    G = nx.Graph()
    for nid in sorted(store.nodes):
        n = store.nodes[nid]
        if n.valid:
            if self_guard == "exclude" and n.id == SELF_ENTITY_ID:
                continue   # drop the self anchor entirely (it carries no discriminating signal)
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
        # Self-anchor hub guard: stop the single first-person node from being a
        # PPR super-hub that activation routes through (see config.self_guard).
        incident_self = self_guard != "none" and (u == SELF_ENTITY_ID or v == SELF_ENTITY_ID)
        if self_guard == "exclude" and incident_self:
            continue                               # drop self + its RESOLVES_TO star
        w = max(1e-4, float(data["confidence"]) * float(data["weight"]))
        if self_guard == "cap" and incident_self:
            w = min(w, self_cap)                   # keep self, throttle its pull
        pair = (u, v) if u <= v else (v, u)        # symmetrize direction
        by_etype = pair_w.setdefault(pair, {})
        if w > by_etype.get(et, 0.0):              # MAX within an etype (parallel facts
            by_etype[et] = w                       # count once); SUM across distinct etypes
    for (u, v) in sorted(pair_w):
        by_etype = pair_w[(u, v)]
        G.add_edge(u, v, weight=sum(w for _et, w in sorted(by_etype.items())))
    return G


# --------------------------------------------------------------------------- #
# Personalized PageRank over a cached sparse operator
# --------------------------------------------------------------------------- #
# nx.pagerank rebuilds the row-normalized CSR matrix from the graph on EVERY call —
# an O(E) conversion that dwarfs the actual power iteration. The projection is cached
# above, so the operator derived from it can be too: keyed weakly by the graph object
# (a rebuilt projection is a new object, so its operator follows automatically).
_PPR_OP_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _ppr_operator(G: nx.Graph):
    bundle = _PPR_OP_CACHE.get(G)
    if bundle is None:
        import scipy.sparse as sp
        nodelist = list(G)
        A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", dtype=float)
        S = np.asarray(A.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = sp.csr_array(sp.spdiags(S.T, 0, *A.shape))
        A = Q @ A                               # row-normalized transition operator
        is_dangling = np.where(S == 0)[0]
        index = {n: i for i, n in enumerate(nodelist)}
        bundle = (nodelist, index, A, is_dangling)
        _PPR_OP_CACHE[G] = bundle
    return bundle


def personalized_pagerank(G: nx.Graph, *, alpha: float, personalization: dict,
                          max_iter: int = 200, tol: float = 1e-6) -> dict:
    """Same math as nx.pagerank's scipy path (uniform start, dangling mass to the
    personalization vector, L1 convergence at N*tol), minus the per-call graph→CSR
    conversion. Raises nx.PowerIterationFailedConvergence like nx.pagerank does."""
    N = len(G)
    if N == 0:
        return {}
    nodelist, index, A, is_dangling = _ppr_operator(G)
    p = np.zeros(N, dtype=float)
    for n, val in personalization.items():
        i = index.get(n)
        if i is not None:
            p[i] = val
    p /= p.sum()
    x = np.repeat(1.0 / N, N)
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * p) + (1 - alpha) * p
        if np.absolute(x - xlast).sum() < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)


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
        with prof_span("query.seed"):
            seeds = self.seeder.seed(query)
        res = RetrievalResult(query=query, mode=self.mode, seeds=list(seeds), as_of=as_of)
        if not seeds:
            return res
        with prof_span("query.project_graph"):
            G = projected_graph(self.store, self.config, as_of=as_of)
        skip_self_seed = getattr(self.config, "self_guard", "none") in ("exclude", "seed")
        pers = {}
        for nid, s in seeds.items():
            if skip_self_seed and nid == SELF_ENTITY_ID:
                continue   # don't pour personalization mass into the self anchor
            if nid in G and s > 0:
                pers[nid] = s * self.canon.idf_weight(nid)
        if not pers or sum(pers.values()) <= 0:
            return res
        with prof_span("query.pagerank"):
            ppr = personalized_pagerank(G, alpha=self.config.ppr_damping,
                                        personalization=pers, max_iter=200)
        cand = []
        for nid, sc in ppr.items():
            n = self.store.get_node(nid)
            if n and n.ntype == NodeType.EPISODE and n.valid:
                cand.append((nid, sc))
        cand.sort(key=lambda x: (-x[1], x[0]))    # id tie-break: order-insensitive
        cand = cand[: max(k * 3, k)]
        with prof_span("query.mmr_rerank"):
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
        """Min hop-distance from ANY seed, as one multi-source BFS. Equivalent to (but
        one traversal instead of len(seeds)) a per-seed BFS keeping the minimum."""
        dist: dict[str, int] = {s: 0 for s in seeds if s in G}
        frontier = list(dist)
        d = 0
        while frontier and d < cutoff:
            d += 1
            nxt = []
            for u in frontier:
                for v in G.adj[u]:
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            frontier = nxt
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


class HybridRetriever:
    """The production answer-path retriever (used by `ask`, NOT by `query`): on top of the
    PPR pool it adds a 4-lane query router, a fact-bearing-episode augment on state/evolution
    questions, and a cross-encoder rerank — but ONLY on the hard lanes (config.rerank_lanes),
    since the web-search cross-encoder can demote the gold on easy single-fact lookups. Stashes
    `.lane` and `.entity_ids` on the result for the evolution-aware context builder."""
    mode = "hybrid"

    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config):
        self.store = store
        self.embedder = embedder
        self.canon = canon
        self.config = config
        self.ppr = PPRRetriever(store, embedder, canon, config)
        self.facts = FactIndex(store)
        self._reranker = (CrossEncoderReranker(config.rerank_model)
                          if getattr(config, "rerank", True) else None)

    @property
    def rerank_active(self) -> bool | None:
        return self._reranker.available if self._reranker is not None else None

    def _snippet(self, ep_id: str, n: int = 512) -> str:
        node = self.store.get_node(ep_id)
        if not node:
            return ""
        text = node.raw_text or node.description or node.name or ""
        return _WSP.sub(" ", text).strip()[:n]

    def _entity_seeds(self, seeds: list[str]) -> list[str]:
        out = []
        for nid in seeds:
            n = self.store.get_node(nid)
            if n and n.ntype == NodeType.ENTITY and n.valid:
                out.append(nid)
        return out

    def _event_time(self, ep_id: str) -> str:
        n = self.store.get_node(ep_id)
        return (n.created_at or n.ingested_at or "") if n else ""

    def retrieve(self, query: str, k: int | None = None,
                 as_of: str | None = None, kind: str | None = None) -> RetrievalResult:
        k = k or self.config.top_k
        lane = route(query, kind) if getattr(self.config, "route", True) else "single"
        base_pool = max(int(getattr(self.config, "rerank_pool", 32)), k * 3)
        pool = base_pool * 2 if lane == MULTIHOP else base_pool   # multihop widens the pool

        base = self.ppr.retrieve(query, k=pool, as_of=as_of)
        cand_ids: list[str] = list(base.object_ids)
        ent_ids = self._entity_seeds(base.seeds)

        # STATE/evolution: guarantee the fact-bearing episodes are in the pool.
        if getattr(self.config, "fact_lane_augment", True) and lane == STATE and ent_ids:
            with prof_span("query.fact_augment"):
                for ep in sorted(self.facts.fact_episodes(ent_ids)):
                    n = self.store.get_node(ep)
                    if ep not in cand_ids and n and n.valid and n.ntype == NodeType.EPISODE:
                        cand_ids.append(ep)

        # RECENCY: restrict to the most recent candidates by event time before ranking.
        if lane == RECENCY and cand_ids:
            cand_ids = sorted(cand_ids, key=self._event_time, reverse=True)[: max(k * 2, k)]

        # Conditional rerank: only the hard lanes; single/recency keep PPR / event-time order.
        rerank_lanes = set(getattr(self.config, "rerank_lanes", ("state", "multihop")))
        if self._reranker is not None and cand_ids and lane in rerank_lanes:
            with prof_span("query.cross_encoder"):
                ranked = self._reranker.rerank(
                    query, [(ep, self._snippet(ep)) for ep in cand_ids], k)
            # PPR guarantee: the raw PPR pool's top-N must survive the cross-encoder —
            # the CE can demote an episode the graph ranked #1 out of the top-k entirely.
            # Trim the reranked tail to make room; the reranker's relative order is kept.
            keep_n = int(getattr(self.config, "rerank_keep_ppr_top", 0))
            if keep_n > 0:
                missing = [ep for ep in base.object_ids[:keep_n]
                           if ep in cand_ids and ep not in ranked]
                if missing:
                    ranked = ranked[: max(k - len(missing), 0)] + missing
        else:
            ranked = cand_ids[:k]

        res = RetrievalResult(query=query, mode=self.mode, as_of=as_of,
                              seeds=list(base.seeds),
                              subgraph=set(base.subgraph) | set(ranked))
        res.objects = [(ep, float(len(ranked) - i)) for i, ep in enumerate(ranked)]
        res.lane = lane            # type: ignore[attr-defined]
        res.entity_ids = ent_ids   # type: ignore[attr-defined]
        # Raw PPR pool (episode_id, real diffusion score) BEFORE lane trims / cross-encoder
        # rerank — the failure-triage layer reads gold's rank + score margin off this to
        # split "never retrievable" from "retrievable but ranked out".
        res.ppr_pool = list(base.objects)  # type: ignore[attr-defined]
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
