"""Communities & breadth queries (docs/ARCHITECTURE.md §6 / phase 3, Path B).

Detect communities over the symmetrized traversal projection (Louvain here — needs
an undirected graph; seedable and in
NetworkX core; Leiden/graspologic is the noted upgrade), attach a precomputed
summary to each, and route breadth queries ("what are the main themes") to a
map-reduce over those summaries instead of local PPR traversal.
"""
from __future__ import annotations

import re
from collections import Counter

import networkx as nx

from .config import Config
from .embedders import Embedder
from .models import EdgeType, Edge, NodeType, Provenance, community_node
from .retrieval import projected_graph
from .store import GraphStore, now_iso

_GLOBAL_CUES = re.compile(
    r"\b(theme|themes|overall|main topics?|broad|across (?:all|everything|the)|"
    r"summar|what kinds?|categor|in general|big picture|landscape|range of)\b", re.I)


def is_global_query(query: str) -> bool:
    """Cheap breadth-query classifier (§5 Path B router)."""
    return bool(_GLOBAL_CUES.search(query or ""))


def build_communities(store: GraphStore, embedder: Embedder, config: Config) -> int:
    """(Re)build CommunityNodes + IN_COMMUNITY edges + summaries. Returns count."""
    # drop any previous communities
    for cid in [n.id for n in store.nodes_of_type(NodeType.COMMUNITY, valid_only=False)]:
        if cid in store.g:
            store.g.remove_node(cid)
        store.nodes.pop(cid, None)

    G = projected_graph(store, config)
    if G.number_of_nodes() == 0:
        return 0
    try:
        parts = nx.community.louvain_communities(
            G, weight="weight", seed=config.community_seed)
    except Exception:
        parts = nx.community.label_propagation_communities(G)

    count = 0
    for i, members in enumerate(parts):
        objs = [m for m in members
                if store.get_node(m) and store.get_node(m).ntype == NodeType.OBJECT]
        if len(objs) < 2:                      # skip singletons / non-object clusters
            continue
        cid = f"comm_{i:03d}"
        summary = _summarize_community(store, members, objs)
        cnode = community_node(cid, members=list(objs), summary=summary, ts=now_iso())
        store.add_node(cnode)
        for m in objs:
            store.add_edge(Edge(src=m, dst=cid, etype=EdgeType.IN_COMMUNITY,
                               provenance=Provenance.DERIVED, confidence=1.0))
        store.vectors.add("community", cid, embedder.embed([summary])[0])
        count += 1
    return count


def _summarize_community(store: GraphStore, members, objs) -> str:
    """Extractive summary: dominant tags + representative titles (offline-friendly)."""
    tag_counts: Counter = Counter()
    for m in members:
        node = store.get_node(m)
        if node and node.ntype == NodeType.TAG:
            tag_counts[node.name] += 1
    for oid in objs:
        for nbr, data in store.neighbors(oid, etypes={EdgeType.TAGGED_AS}):
            tn = store.get_node(nbr)
            if tn:
                tag_counts[tn.name] += 1
    top_tags = [t for t, _ in tag_counts.most_common(8)]
    titles = [store.get_node(o).name for o in objs[:6] if store.get_node(o)]
    return (f"Theme: {', '.join(top_tags)}. "
            f"Includes: {'; '.join(titles)}.")


class CommunityRetriever:
    """Path B: map-reduce over community summaries for breadth questions."""
    mode = "community"

    def __init__(self, store: GraphStore, embedder: Embedder, config: Config):
        self.store = store
        self.embedder = embedder
        self.config = config

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        qv = self.embedder.embed([query])[0]
        hits = self.store.vectors.search("community", qv, k=k, floor=1e-6)
        out = []
        for cid, score in hits:
            node = self.store.get_node(cid)
            if node:
                out.append({"community": cid, "score": round(float(score), 4),
                            "summary": node.summary, "size": len(node.members),
                            "members": node.members})
        return out
