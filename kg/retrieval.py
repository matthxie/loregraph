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

import os
import re
import weakref
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .canonicalize import Canonicalizer, normalize_key
from .config import Config
from .csr import CSRGraph
from .embedders import Embedder
from .facts import FactIndex
from .models import SELF_ENTITY_ID, EdgeType, NodeType, Provenance
from .profiler import span as prof_span
from .rerank import CrossEncoderReranker
from .route import MULTIHOP, RECENCY, STATE, route
from .store import GraphStore, fact_active

_TOK = re.compile(r"[a-z0-9]+")
_WSP = re.compile(r"\s+")

# --------------------------------------------------------------------------- #
# Relative-date window resolution (config.date_window_boost)
# --------------------------------------------------------------------------- #
_WEEKDAY_IDX = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6}
_NUM_WORD_VAL = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                 "twelve": 12, "couple": 2, "few": 3}
_RE_LAST_WEEKDAY = re.compile(
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_RE_N_AGO = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"couple|few)\s+(?:of\s+)?(day|week|month)s?\s+ago\b", re.I)
_RE_LAST_UNIT = re.compile(r"\blast\s+(week|month|year)\b", re.I)


def resolve_relative_window(query: str, as_of: str | None):
    """Resolve one relative-date phrase in `query` against the `as_of` anchor date to an
    inclusive (start, end) datetime.date window, widened by the phrase's natural
    granularity ("two months ago" is fuzzier than "yesterday"). None when the query has
    no such phrase or no usable anchor — callers treat None as boost-off."""
    if not as_of:
        return None
    from datetime import date, timedelta
    try:
        anchor = date.fromisoformat(str(as_of)[:10])
    except ValueError:
        return None
    q = query or ""
    m = _RE_LAST_WEEKDAY.search(q)
    if m:
        back = (anchor.weekday() - _WEEKDAY_IDX[m.group(1).lower()]) % 7 or 7
        t = anchor - timedelta(days=back)
        return t - timedelta(days=1), t + timedelta(days=1)
    m = _RE_N_AGO.search(q)
    if m:
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else _NUM_WORD_VAL[raw]
        unit = m.group(2).lower()
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        fuzz = {"day": 1, "week": 3, "month": 10}[unit]
        t = anchor - timedelta(days=days)
        return t - timedelta(days=fuzz), t + timedelta(days=fuzz)
    if re.search(r"\byesterday\b", q, re.I):
        t = anchor - timedelta(days=1)
        return t, t
    m = _RE_LAST_UNIT.search(q)
    if m:
        unit = m.group(1).lower()
        days = {"week": 7, "month": 30, "year": 365}[unit]
        fuzz = {"week": 4, "month": 12, "year": 45}[unit]
        t = anchor - timedelta(days=days)
        return t - timedelta(days=fuzz), t + timedelta(days=fuzz)
    return None


@dataclass
class RetrievalResult:
    query: str
    mode: str
    objects: list[tuple[str, float]] = field(default_factory=list)  # (episode_id, score)
    seeds: list[str] = field(default_factory=list)
    # raw embedding+BM25+link seed mass per node id (query-side chunk retargeting reads
    # this to rank a source's chunks by embedding relevance instead of PPR diffusion order —
    # see ContextBuilder._retarget_chunks in kg/rag.py). Empty unless populated below.
    seed_scores: dict = field(default_factory=dict)
    subgraph: set[str] = field(default_factory=set)   # every node touched (recall@k)
    as_of: str | None = None
    # Fact lane (config.fact_lane): what the statement-granularity lane matched, for the
    # context builder to mark [matched] and provenance-promote. Empty unless the lane fires
    # (knob off ⇒ empty ⇒ seeds/pool/context byte-identical). Shape:
    #   {"surfaces": [stmt surface, …], "episodes": [provenance chunk id, …],
    #    "hits": [(vector_id, cosine), …]}
    fact_matched: dict = field(default_factory=dict)

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

    def _episode_doc(self, n) -> str:
        """The §3.4 composite corpus doc for one episode: raw text + title + the
        analyzed media/link description + entity/concept surface names + file
        name/extension tokens from media_paths ("png", "pdf", …). Wire shapes and
        score semantics are unchanged — a wider corpus just matches more."""
        parts = [n.raw_text or "", n.title or "", n.description or ""]
        parts.extend(n.tags or [])           # denormalised canonical tag/concept names
        seen: set[str] = set()
        for mid, _d in self.store.neighbors(n.id, etypes={EdgeType.MENTIONED_IN},
                                            direction="in"):
            for eid, _d2 in self.store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                                 direction="out"):
                if eid in seen:
                    continue
                seen.add(eid)
                ent = self.store.get_node(eid)
                if ent is not None and ent.valid:
                    parts.append(ent.name)
        for p in n.media_paths or []:
            base = os.path.basename(p)
            parts.append(base)               # _TOK splits "img_2024.png" → img 2024 png
            ext = os.path.splitext(base)[1].lstrip(".")
            if ext:
                parts.append(ext)
        return " ".join(p for p in parts if p)

    def _ensure_bm25(self):
        # Rebuild when the store has mutated since the last build. The composite doc
        # (title, description, entity/tag surfaces) is sensitive to post-ingest
        # mutation — canonical renames/merges, title backfill — which bumps
        # store.version but NOT episode_version, so the wider gate is required.
        # Retrievers are long-lived (hoisted onto KnowledgeGraph), so across pure
        # read queries this stays a no-op.
        version = getattr(self.store, "version", None)
        if self._bm25 is not None and version == self._bm25_version:
            return
        self._bm25_version = version
        corpus, ids = [], []
        for n in sorted(self.store.nodes_of_type(NodeType.EPISODE), key=lambda n: n.id):
            corpus.append(_TOK.findall(self._episode_doc(n).lower()))
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

    def bm25_search(self, query: str, k: int | None = None, *,
                    normalized: bool = True) -> list[tuple[str, float]]:
        """Top-k (episode_id, score) over the composite episode docs by lexical
        relevance. BM25 when `rank_bm25` is installed, else a deterministic token-overlap
        scan so lexical search still works fully offline. Scores are max-normalized for
        seed fusion (the default); `normalized=False` serves the raw BM25 scale the
        `search` verb reports (PROTOCOL §3.4)."""
        k = k or self.config.seed_k
        self._ensure_bm25()
        toks = _TOK.findall((query or "").lower())
        if not toks or not self._bm25_ids:
            return []
        scores = self._bm25.get_scores(toks) if self._bm25 else None
        if scores is None or (float(scores.max()) if len(scores) else 0.0) <= 0:
            # rank_bm25 missing — or degenerate on a tiny corpus (a term in half or
            # more of very few docs gets idf ≤ 0, zeroing every score). Token-overlap
            # keeps exact-name search alive on small personal stores.
            want = set(toks)
            scores = np.array([sum(1 for t in doc if t in want)
                               for doc in self._bm25_corpus], dtype=np.float64)
        peak = float(scores.max()) if len(scores) else 0.0
        if peak <= 0:
            return []
        if normalized:
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

        # Tombstone filter (kg/forget.py): the vector index is append-only, so erased
        # nodes still surface here — drop them so no seed mass reaches forgotten content.
        return {nid: s for nid, s in scores.items()
                if (n := self.store.get_node(nid)) is None or n.valid}

    def fact_seed(self, query: str) -> tuple[dict[str, float], list[str], list[tuple[str, float]]]:
        """Fact lane (config.fact_lane): score the query against the kind="fact" vectors
        (statement + distilled-aggregate surfaces written by config.fact_vectors) and map the
        top `fact_lane_k` hits back through `fact_provenance` to their provenance CHUNKS +
        endpoint ENTITIES, each carrying the fact's cosine (max over the facts that name it).

        Returns (raw_scores{node_id: cosine}, matched_stmt_surfaces, hits[(vec_id, cosine)]).
        This is the RAW, uncapped, un-merged contribution — the caller merges it ADDITIVELY
        (new nodes only) under the fact_lane_weight mass cap, so this method never touches the
        episode lane. Empty when no fact vectors are present (the lane no-ops on an
        un-backfilled store)."""
        from .fact_vectors import FACT_KIND, fact_provenance
        k = int(getattr(self.config, "fact_lane_k", 10))
        qv = self.embedder.embed([query])[0]
        try:
            hits = self.store.vectors.search(FACT_KIND, qv, k=k, floor=0.0)
        except ValueError:
            hits = []                                  # dim mismatch: no usable fact index
        if not hits:
            return {}, [], []
        prov = fact_provenance(self.store, {hid for hid, _ in hits})
        raw: dict[str, float] = {}
        surfaces: list[str] = []
        seen_surf: set[str] = set()
        for hid, cos in hits:                          # hits are cosine-desc, id tie-broken
            rec = prov.get(hid)
            if not rec:
                continue
            surf = rec.get("stmt_surface") or rec["surface"]
            if surf not in seen_surf:
                seen_surf.add(surf)
                surfaces.append(surf)
            for nid in rec["episodes"] | rec["entities"]:
                n = self.store.get_node(nid)
                if n is not None and not n.valid:      # forgotten/tombstoned — no mass
                    continue
                raw[nid] = max(raw.get(nid, 0.0), float(cos))
        return raw, surfaces, [(hid, float(cos)) for hid, cos in hits]


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
                    exclude_etypes: set[str] | None = None) -> CSRGraph:
    """Undirected, weighted projection of the directed store (HippoRAG runs PPR
    undirected), as an immutable array-backed CSRGraph (kg/csr.py). Fact (RELATED_TO)
    edges are filtered to the requested temporal view (`as_of=None` → current;
    `as_of=T` → as-of-T); structural edges are timeless.
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


def self_like_ids(store: GraphStore, config: Config) -> set[str]:
    """Node ids the self-guard treats as the first-person hub: the canonical self anchor
    (personal-web mode) PLUS any regular extracted narrator entity ('me'/self_name). The
    extractor's USER ACTIONS rule mints 'me' as an ordinary entity when self_entity is
    off, and on a single-user store that node ends up incident to most action fact
    edges — exactly the PPR super-hub the guard exists for, under a different id."""
    names = {"me", str(getattr(config, "self_name", "self")).strip().lower()}
    ids = {SELF_ENTITY_ID}
    for nid, n in store.nodes.items():
        if (n.valid and n.ntype == NodeType.ENTITY
                and (n.name or "").strip().lower() in names):
            ids.add(nid)
    return ids


def _build_projection(store: GraphStore, config: Config, *, as_of: str | None,
                      exclude: set[str]) -> CSRGraph:
    self_guard = getattr(config, "self_guard", "none")
    self_cap = float(getattr(config, "self_guard_cap", 0.05))
    self_ids = self_like_ids(store, config) if self_guard != "none" else {SELF_ENTITY_ID}
    # Deterministic construction (sorted nodes / sorted pair accumulation): the
    # projection — and everything downstream of it, PPR float sums included — is a
    # function of graph CONTENT, not of node/edge insertion or load order.
    nodes: list[str] = []
    for nid in sorted(store.nodes):
        n = store.nodes[nid]
        if n.valid:
            if self_guard == "exclude" and n.id in self_ids:
                continue   # drop the self anchor entirely (it carries no discriminating signal)
            nodes.append(n.id)
    node_set = set(nodes)
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
        if u not in node_set or v not in node_set:
            continue
        # Self-anchor hub guard: stop the single first-person node from being a
        # PPR super-hub that activation routes through (see config.self_guard).
        incident_self = self_guard != "none" and (u in self_ids or v in self_ids)
        if self_guard == "exclude" and incident_self:
            continue                               # drop self + its RESOLVES_TO star
        w = max(1e-4, float(data["confidence"]) * float(data["weight"]))
        if self_guard == "cap" and incident_self:
            w = min(w, self_cap)                   # keep self, throttle its pull
        pair = (u, v) if u <= v else (v, u)        # symmetrize direction
        by_etype = pair_w.setdefault(pair, {})
        if w > by_etype.get(et, 0.0):              # MAX within an etype (parallel facts
            by_etype[et] = w                       # count once); SUM across distinct etypes
    edge_w = {pair: sum(w for _et, w in sorted(pair_w[pair].items()))
              for pair in sorted(pair_w)}
    return CSRGraph.from_pairs(nodes, edge_w)


# --------------------------------------------------------------------------- #
# Personalized PageRank over a cached sparse operator
# --------------------------------------------------------------------------- #
# nx.pagerank rebuilds the row-normalized CSR matrix from the graph on EVERY call —
# an O(E) conversion that dwarfs the actual power iteration. The projection is cached
# above, so the operator derived from it can be too: keyed weakly by the graph object
# (a rebuilt projection is a new object, so its operator follows automatically).
_PPR_OP_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _ppr_operator(G):
    bundle = _PPR_OP_CACHE.get(G)
    if bundle is None:
        import scipy.sparse as sp
        if isinstance(G, CSRGraph):
            # zero-conversion: the projection's arrays ARE the adjacency matrix
            # (canonical CSR, same structure nx.to_scipy_sparse_array produces)
            nodelist = list(G.ids)
            n = len(nodelist)
            A = sp.csr_array((G.weights, G.indices, G.indptr), shape=(n, n))
        else:
            nodelist = list(G)
            A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight",
                                         dtype=float)
        S = np.asarray(A.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = sp.csr_array(sp.spdiags(S.T, 0, *A.shape))
        A = Q @ A                               # row-normalized transition operator
        is_dangling = np.where(S == 0)[0]
        index = {n: i for i, n in enumerate(nodelist)}
        bundle = (nodelist, index, A, is_dangling)
        _PPR_OP_CACHE[G] = bundle
    return bundle


def personalized_pagerank(G, *, alpha: float, personalization: dict,
                          max_iter: int = 200, tol: float = 1e-6) -> dict:
    """Same math as nx.pagerank's scipy path (uniform start, dangling mass to the
    personalization vector, L1 convergence at N*tol), minus the per-call graph→CSR
    conversion. Accepts a CSRGraph (the projection) or an nx.Graph.
    Raises nx.PowerIterationFailedConvergence like nx.pagerank does."""
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


def local_push_ppr(G, *, alpha: float, personalization: dict,
                   eps: float = 1e-6) -> dict:
    """Approximate PPR by forward push (Andersen–Chung–Lang): solves the same fixed
    point as `personalized_pagerank` — x = (1-alpha)·s + alpha·x·W, dangling mass
    redistributed to the personalization — but explores only outward from the seeds,
    so work is O(1/((1-alpha)·eps)) pushes and INDEPENDENT of graph size. Returns only
    nodes with nonzero approximate mass; each score is within eps·weighted-degree of
    the exact solution, which preserves the ranking at the resolution retrieval uses.
    Deterministic: seeds are processed in sorted order and neighbor iteration follows
    the projection's content-sorted construction. Accepts a CSRGraph (array fast
    path) or an nx.Graph (dict path, for tests / ad-hoc graphs)."""
    if isinstance(G, CSRGraph):
        return _local_push_ppr_csr(G, alpha=alpha, personalization=personalization,
                                   eps=eps)
    if len(G) == 0:
        return {}
    total = sum(v for n, v in personalization.items() if n in G and v > 0)
    if total <= 0:
        return {}
    seeds = {n: v / total for n, v in sorted(personalization.items())
             if n in G and v > 0}

    from collections import deque
    p: dict = {}
    r: dict = dict(seeds)
    degw: dict = {}

    def deg(u) -> float:
        d = degw.get(u)
        if d is None:
            d = degw[u] = float(sum(data.get("weight", 1.0) for data in G[u].values()))
        return d

    def excess(u) -> bool:
        return r.get(u, 0.0) > eps * max(deg(u), 1.0)

    queue = deque(seeds)
    in_queue = set(seeds)
    while queue:
        u = queue.popleft()
        in_queue.discard(u)
        if not excess(u):                     # residual may have shrunk below threshold
            continue
        ru = r[u]
        p[u] = p.get(u, 0.0) + (1.0 - alpha) * ru
        r[u] = 0.0
        du = deg(u)
        if du > 0:
            spread = ((v, data.get("weight", 1.0) / du) for v, data in G[u].items())
        else:                                 # dangling: follow-mass teleports to seeds
            spread = seeds.items()
        for v, frac in spread:
            r[v] = r.get(v, 0.0) + alpha * ru * frac
            if v not in in_queue and excess(v):
                queue.append(v)
                in_queue.add(v)
    return p


def _local_push_ppr_csr(G: CSRGraph, *, alpha: float, personalization: dict,
                        eps: float) -> dict:
    """local_push_ppr over the CSR arrays: same push rule and threshold, node ids
    swapped for row indices and neighbor iteration done on array slices."""
    if len(G) == 0:
        return {}
    total = sum(v for n, v in personalization.items() if n in G and v > 0)
    if total <= 0:
        return {}
    index = G.index
    seeds = {index[n]: v / total for n, v in sorted(personalization.items())
             if n in G and v > 0}
    degw = G.weighted_degrees()

    from collections import deque
    p: dict[int, float] = {}
    r: dict[int, float] = dict(seeds)

    def excess(i) -> bool:
        return r.get(i, 0.0) > eps * max(degw[i], 1.0)

    queue = deque(seeds)
    in_queue = set(seeds)
    while queue:
        i = queue.popleft()
        in_queue.discard(i)
        if not excess(i):                     # residual may have shrunk below threshold
            continue
        ri = r[i]
        p[i] = p.get(i, 0.0) + (1.0 - alpha) * ri
        r[i] = 0.0
        di = degw[i]
        if di > 0:
            idx, wts = G.neighbor_rows(i)
            spread = zip(idx.tolist(), (wts / di).tolist())
        else:                                 # dangling: follow-mass teleports to seeds
            spread = seeds.items()
        for j, frac in spread:
            r[j] = r.get(j, 0.0) + alpha * ri * frac
            if j not in in_queue and excess(j):
                queue.append(j)
                in_queue.add(j)
    ids = G.ids
    return {ids[i]: val for i, val in p.items()}


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
                 as_of: str | None = None,
                 mmr_lambda: float | None = None) -> RetrievalResult:
        k = k or self.config.top_k
        if not query or not query.strip():
            return RetrievalResult(query=query, mode=self.mode, as_of=as_of)
        with prof_span("query.seed"):
            seeds = self.seeder.seed(query)
        fact_matched: dict = {}
        if getattr(self.config, "fact_lane", False):
            with prof_span("query.fact_lane"):
                seeds, fact_matched = self._merge_fact_lane(query, seeds)
        res = RetrievalResult(query=query, mode=self.mode, seeds=list(seeds),
                              seed_scores=dict(seeds), as_of=as_of,
                              fact_matched=fact_matched)
        if not seeds:
            return res
        with prof_span("query.project_graph"):
            G = projected_graph(self.store, self.config, as_of=as_of)
        skip_self_seed = getattr(self.config, "self_guard", "none") in ("exclude", "seed")
        self_ids = self_like_ids(self.store, self.config) if skip_self_seed else set()
        pers = {}
        for nid, s in seeds.items():
            if skip_self_seed and nid in self_ids:
                continue   # don't pour personalization mass into the self anchor
            if nid in G and s > 0:
                pers[nid] = s * self.canon.idf_weight(nid)
        if not pers or sum(pers.values()) <= 0:
            return res
        with prof_span("query.pagerank"):
            if getattr(self.config, "ppr_backend", "global") == "push":
                ppr = local_push_ppr(G, alpha=self.config.ppr_damping,
                                     personalization=pers,
                                     eps=getattr(self.config, "ppr_push_eps", 1e-6))
            else:
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
            ranked = self._rerank(query, cand, seeds, G, k, mmr_lambda=mmr_lambda)
        res.objects = ranked
        res.subgraph = set(seeds) | {oid for oid, _ in ranked}
        return res

    def _merge_fact_lane(self, query: str, seeds: dict[str, float]
                         ) -> tuple[dict[str, float], dict]:
        """ADDITIVE fact-lane merge (config.fact_lane). Seeds the top-fact hits' provenance
        chunks + endpoint entities INTO the seed set, but only nodes the episode lane did not
        already seed — so every episode-lane seed keeps its exact score (additive-only; tested
        in tests/test_fact_lane.py). Total added mass is capped at `fact_lane_weight` × the
        episode-lane mass so noisy ~gpt-4o-mini facts cannot out-vote the episode lane in the
        PPR personalization. Design (docs/OFFLINE_EVAL.md Round 7b): the SEED-MASS route (vs an
        explicit pool injection) — the provenance chunk earns its pool slot through the normal
        PPR→MMR→CE pipeline (a seeded episode gets ≥(1-α)·mass restart, so it enters the pool)
        rather than being force-injected past the ranker (Round 4's forced-injection
        regressions), and populating seed_scores lets chunk-retargeting use it too."""
        raw, surfaces, hits = self.seeder.fact_seed(query)
        if not raw:
            return seeds, {}
        add = {nid: sc for nid, sc in raw.items() if nid not in seeds}   # NEW nodes only
        prov_eps = sorted(nid for nid in raw
                          if (n := self.store.get_node(nid)) is not None
                          and n.ntype == NodeType.EPISODE and n.valid)
        matched = {"surfaces": surfaces, "episodes": prov_eps,
                   "hits": [(hid, round(c, 4)) for hid, c in hits]}
        if not add:
            return seeds, matched                       # everything already episode-seeded
        ep_mass = sum(seeds.values())
        raw_mass = sum(add.values())
        cap = float(getattr(self.config, "fact_lane_weight", 0.5)) * ep_mass
        if raw_mass > cap > 0:
            scale = cap / raw_mass
            add = {nid: sc * scale for nid, sc in add.items()}
        merged = dict(seeds)                             # episode-lane scores untouched
        merged.update(add)                              # add[*] keys are disjoint from seeds
        return merged, matched

    def _rerank(self, query, cand, seeds, G, k, mmr_lambda: float | None = None):
        """MMR diversity + node-distance to seeds on top of PPR scores (graphiti).
        `mmr_lambda` (PROTOCOL §7.3): a per-call relevance⇄diversity dial, clamped to
        [0, 1]; None keeps the engine's configured default."""
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
        lam = self.config.mmr_lambda if mmr_lambda is None \
            else max(0.0, min(1.0, float(mmr_lambda)))
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
                for v in G.neighbors(u):
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
        self._forced_reranker = None    # lazily built for per-call rerank=True only —
        #                                 never installed as self._reranker, so a forced
        #                                 call can't flip lane-gated CE on for later
        #                                 default calls when config.rerank is off

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

    def _valid_episode(self, eid: str) -> bool:
        n = self.store.get_node(eid)
        return bool(n and n.ntype == NodeType.EPISODE and n.valid)

    def _session_dedup(self, ranked: list[str], k: int) -> list[str]:
        """config.rag_session_dedup: reorder so the context-eligible prefix
        (rag_context_episodes) holds the FIRST chunk of each distinct session in rank
        order; extra chunks of already-represented sessions follow. Pure reorder — no
        episode enters or leaves the top-k, so recall_at_pool is untouched; only the
        session mix the reader actually sees changes."""
        if not getattr(self.config, "rag_session_dedup", False) or not ranked:
            return ranked
        ctx_n = min(int(getattr(self.config, "rag_context_episodes", k)) or k, k)
        prefix: list[str] = []
        rest: list[str] = []
        seen: set[str] = set()
        for eid in ranked:
            s = eid.split("#", 1)[0]
            if len(prefix) < ctx_n and s not in seen:
                prefix.append(eid)
                seen.add(s)
            else:
                rest.append(eid)
        return (prefix + rest)[:k]

    def _reserve_slots(self, query: str, ranked: list[str], base: RetrievalResult,
                       cand_ids: list[str], k: int, as_of: str | None) -> list[str]:
        """Post-ranking slot reservations (both off by default, see config):

        date_window_boost — when the query carries a relative-date phrase, episodes whose
        event date falls in the resolved window get first claim on reserved slots.
        seed_reserve — the Seeder's own top-scored episodes get slots when their session
        won none from the PPR->MMR->CE pipeline (the dominant recall failure: gold in
        seeds, ranked out downstream).

        Both displace only the TAIL of the final ranking, never the head, and only for a
        session not already represented — coverage additions, not reorderings."""
        def sess(eid: str) -> str:
            return eid.split("#", 1)[0]

        out = list(ranked)
        covered = {sess(e) for e in out}
        picks: list[str] = []

        def inject(cands, slots):
            for eid in cands:
                if slots <= 0:
                    break
                if sess(eid) in covered or eid in out or not self._valid_episode(eid):
                    continue
                picks.append(eid)
                covered.add(sess(eid))
                slots -= 1

        d_slots = int(getattr(self.config, "date_window_slots", 2)) \
            if getattr(self.config, "date_window_boost", False) else 0
        if d_slots > 0:
            win = resolve_relative_window(query, as_of)
            if win is not None:
                lo, hi = win

                def in_window(eid: str) -> bool:
                    t = self._event_time(eid)[:10]
                    return bool(t) and str(lo) <= t <= str(hi)

                pool = [ep for ep, _s in getattr(base, "objects", []) or []]
                seed_eps = sorted((nid for nid in base.seed_scores
                                   if self._valid_episode(nid)),
                                  key=lambda nid: (-base.seed_scores[nid], nid))
                inject([e for e in pool + seed_eps if in_window(e)], d_slots)

        s_slots = int(getattr(self.config, "seed_reserve", 0))
        if s_slots > 0:
            seed_eps = sorted((nid for nid in base.seed_scores
                               if self._valid_episode(nid)),
                              key=lambda nid: (-base.seed_scores[nid], nid))
            inject(seed_eps, s_slots)

        if not picks:
            return out
        # Splice the reserved picks in at the END of the context-eligible prefix
        # (rag_context_episodes), not the tail of k — ContextBuilder._select_episodes
        # only reads the top-n of this ranking, so a tail-of-k injection would win a
        # ranking slot but never reach the reader.
        ctx_n = min(int(getattr(self.config, "rag_context_episodes", k)) or k, k)
        cut = max(ctx_n - len(picks), 0)
        return (out[:cut] + picks + out[cut:])[:k]

    def retrieve(self, query: str, k: int | None = None,
                 as_of: str | None = None, rerank: bool | None = None,
                 mmr_lambda: float | None = None, since: str | None = None,
                 until: str | None = None) -> RetrievalResult:
        """Per-call knobs (PROTOCOL §3.3/§7.3), all optional and additive:
        `rerank=True` blends the cross-encoder into EVERY lane (building it lazily if
        the config left it off); None/False keep the engine's lane-gated default —
        the internal hard-lane CE is baseline ranking quality, not the wire knob.
        `mmr_lambda` dials the PPR MMR stage (clamped [0,1]; None ⇒ config default).
        `since`/`until` bound candidates to an event-time window, compared on the
        10-char date prefix, before lane trims and ranking."""
        k = k or self.config.top_k
        lane = route(query) if getattr(self.config, "route", True) else "single"
        base_pool = max(int(getattr(self.config, "rerank_pool", 32)), k * 3)
        pool = base_pool * 2 if lane == MULTIHOP else base_pool   # multihop widens the pool

        base = self.ppr.retrieve(query, k=pool, as_of=as_of, mmr_lambda=mmr_lambda)
        cand_ids: list[str] = list(base.object_ids)
        ent_ids = self._entity_seeds(base.seeds)

        # STATE/evolution: guarantee the fact-bearing episodes are in the pool.
        if getattr(self.config, "fact_lane_augment", True) and lane == STATE and ent_ids:
            with prof_span("query.fact_augment"):
                for ep in sorted(self.facts.fact_episodes(ent_ids)):
                    n = self.store.get_node(ep)
                    if ep not in cand_ids and n and n.valid and n.ntype == NodeType.EPISODE:
                        cand_ids.append(ep)

        # §7.3 event-time window: filter on the 10-char date prefix (inputs are
        # normalized upstream — bare years arrive as their Jan-1 start).
        def _in_window(eid: str) -> bool:
            d = self._event_time(eid)[:10]
            if not d:
                return False
            return (not since or d >= since[:10]) and (not until or d <= until[:10])
        if (since or until) and cand_ids:
            cand_ids = [eid for eid in cand_ids if _in_window(eid)]

        # RECENCY: restrict to the most recent candidates by event time before ranking.
        if lane == RECENCY and cand_ids:
            cand_ids = sorted(cand_ids, key=self._event_time, reverse=True)[: max(k * 2, k)]

        # Conditional rerank: only the hard lanes; single/recency keep PPR / event-time
        # order — unless the caller forces the CE everywhere (rerank=True).
        rerank_lanes = set(getattr(self.config, "rerank_lanes", ("state", "multihop")))
        reranker = self._reranker
        if rerank and reranker is None:
            if self._forced_reranker is None:
                self._forced_reranker = CrossEncoderReranker(self.config.rerank_model)
            reranker = self._forced_reranker
        if reranker is not None and cand_ids and (rerank or lane in rerank_lanes):
            with prof_span("query.cross_encoder"):
                ranked = reranker.rerank(
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

        ranked = self._session_dedup(ranked, k)
        ranked = self._reserve_slots(query, ranked, base, cand_ids, k, as_of)
        if since or until:
            # _reserve_slots can inject from the UNFILTERED pool (seed reserve /
            # date-window boost) — the window is a hard §7.3 bound, so enforce it
            # again on the final ranking.
            ranked = [eid for eid in ranked if _in_window(eid)]

        res = RetrievalResult(query=query, mode=self.mode, as_of=as_of,
                              seeds=list(base.seeds), seed_scores=dict(base.seed_scores),
                              subgraph=set(base.subgraph) | set(ranked),
                              fact_matched=getattr(base, "fact_matched", {}) or {})
        res.objects = [(ep, float(len(ranked) - i)) for i, ep in enumerate(ranked)]
        res.lane = lane            # type: ignore[attr-defined]
        res.entity_ids = ent_ids   # type: ignore[attr-defined]
        # The context builder re-injects episodes (sibling expansion, provenance
        # promotion) after this ranking; it reads the window off the result so those
        # additions honour the same §7.3 bound.
        res.window = (since, until)  # type: ignore[attr-defined]
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
