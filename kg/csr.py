"""CSRGraph — the immutable, array-backed projection that retrieval traverses.

The store keeps a mutable directed multigraph (ingest mutates it edge-by-edge); the
query path never walks that structure directly. `projected_graph` (kg/retrieval.py)
flattens it into this compressed-sparse-row form: node ids in one content-sorted list,
the symmetrized adjacency in three NumPy arrays (indptr / indices / weights). That is
~16 bytes per directed edge instead of a Python dict per edge, feeds the SciPy PPR
operator zero-conversion, and gives cache-friendly BFS / forward-push traversal.

Deterministic: row order is the caller's node order (the projection passes a sorted
list) and each row's neighbor indices are ascending, so every downstream float sum has
a fixed order. Duck-types the slice of the networkx.Graph API the retrieval / eval /
test code uses (len / in / iter, neighbors, degree, edges, nodes, `G[u]` row views),
so projection consumers don't care which representation they were handed.
"""
from __future__ import annotations

import numpy as np


class CSRGraph:
    __slots__ = ("ids", "index", "indptr", "indices", "weights", "_degw", "__weakref__")

    def __init__(self, ids: list, index: dict, indptr: np.ndarray,
                 indices: np.ndarray, weights: np.ndarray):
        self.ids = ids                  # row i -> node id
        self.index = index              # node id -> row i
        self.indptr = indptr            # int64, len n+1
        self.indices = indices          # int64, neighbor row per directed edge
        self.weights = weights          # float64, aligned with indices
        self._degw = None               # lazy weighted-degree array

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pairs(cls, node_ids, pair_weights: dict) -> "CSRGraph":
        """Build from undirected {(u, v): weight} pairs over `node_ids`, whose order
        becomes the row order — pass a deterministically sorted list. Each pair is
        stored in both directions; rows come out with ascending neighbor indices
        (canonical CSR), so the derived SciPy operator is bit-identical to the one
        `nx.to_scipy_sparse_array` builds from the same pairs."""
        ids = list(node_ids)
        index = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        m = len(pair_weights)
        if not m:
            return cls(ids, index, np.zeros(n + 1, dtype=np.int64),
                       np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
        src = np.empty(2 * m, dtype=np.int64)
        dst = np.empty(2 * m, dtype=np.int64)
        wts = np.empty(2 * m, dtype=np.float64)
        k = 0
        for (u, v), w in pair_weights.items():
            iu, iv = index[u], index[v]
            src[k], dst[k], wts[k] = iu, iv, w
            src[k + 1], dst[k + 1], wts[k + 1] = iv, iu, w
            k += 2
        order = np.lexsort((dst, src))          # row-major, columns ascending
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
        return cls(ids, index, indptr, dst[order], wts[order])

    @classmethod
    def from_networkx(cls, G) -> "CSRGraph":
        """Adapter for callers holding a networkx.Graph (tests, ad-hoc graphs)."""
        pairs = {}
        for u, v, data in G.edges(data=True):
            pairs[(u, v)] = float(data.get("weight", 1.0))
        return cls.from_pairs(list(G), pairs)

    # ------------------------------------------------- container protocol
    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __contains__(self, nid) -> bool:
        return nid in self.index

    def __getitem__(self, nid) -> dict:
        """nx-adjacency-style row view: {neighbor_id: {"weight": w}}."""
        i = self.index[nid]
        lo, hi = self.indptr[i], self.indptr[i + 1]
        return {self.ids[j]: {"weight": float(w)}
                for j, w in zip(self.indices[lo:hi], self.weights[lo:hi])}

    # ------------------------------------------------- nx-compatible surface
    def number_of_nodes(self) -> int:
        return len(self.ids)

    def number_of_edges(self) -> int:
        return int(len(self.indices)) // 2

    def nodes(self) -> list:
        return list(self.ids)

    def degree(self, nid) -> int:
        i = self.index[nid]
        return int(self.indptr[i + 1] - self.indptr[i])

    def neighbors(self, nid):
        i = self.index[nid]
        ids = self.ids
        for j in self.indices[self.indptr[i]:self.indptr[i + 1]]:
            yield ids[j]

    def edges(self, nid=None, data: bool = False):
        """Undirected edges, each pair once (lower row first — with sorted ids that is
        lexicographic (u, v) order, matching the old projection's insertion order).
        With `nid`, every edge incident to that node."""
        ids = self.ids
        if nid is None:
            for i in range(len(ids)):
                lo, hi = self.indptr[i], self.indptr[i + 1]
                for j, w in zip(self.indices[lo:hi], self.weights[lo:hi]):
                    if i < j:
                        yield (ids[i], ids[j], {"weight": float(w)}) if data \
                            else (ids[i], ids[j])
        else:
            i = self.index[nid]
            lo, hi = self.indptr[i], self.indptr[i + 1]
            for j, w in zip(self.indices[lo:hi], self.weights[lo:hi]):
                yield (nid, ids[j], {"weight": float(w)}) if data else (nid, ids[j])

    # ------------------------------------------------------------ fast paths
    def neighbor_rows(self, i: int):
        """Row i as raw (indices, weights) array slices — the traversal hot path."""
        lo, hi = self.indptr[i], self.indptr[i + 1]
        return self.indices[lo:hi], self.weights[lo:hi]

    def weighted_degrees(self) -> np.ndarray:
        """Per-row sum of incident edge weights (dangling rows are 0.0); cached."""
        if self._degw is None:
            cs = np.concatenate(([0.0], np.cumsum(self.weights)))
            self._degw = cs[self.indptr[1:]] - cs[self.indptr[:-1]]
        return self._degw

    # ---------------------------------------------------------------- export
    def to_networkx(self):
        """Materialize as networkx.Graph (community detection / layout only — never
        on the query path). Nodes in row order, edges in `edges()` order, so the
        result is constructed identically to the old nx projection."""
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.ids)
        for u, v, d in self.edges(data=True):
            G.add_edge(u, v, weight=d["weight"])
        return G
