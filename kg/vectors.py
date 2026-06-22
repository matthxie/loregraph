"""NumPy brute-force cosine index (docs/ARCHITECTURE.md §0/§7).

At ~200-2000 vectors this is milliseconds and needs no extra dependency. Vectors
are stored L2-normalised so cosine == dot product. Vectors are partitioned by
`kind` ("object" = raw-text/image-description retrieval surface; "tag"/"entity" =
synonymy linking) because those are searched independently (§4).
"""
from __future__ import annotations

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._ids: dict[str, list[str]] = {}      # kind -> [node_id, ...]
        self._mat: dict[str, np.ndarray] = {}     # kind -> (n, dim) normalised
        self._row: dict[str, dict[str, int]] = {}  # kind -> {node_id: row}

    # ---- mutation -----------------------------------------------------------
    def add(self, kind: str, node_id: str, vec: np.ndarray) -> None:
        vec = l2_normalize(vec).reshape(1, -1)
        ids = self._ids.setdefault(kind, [])
        rows = self._row.setdefault(kind, {})
        if node_id in rows:                       # update in place
            self._mat[kind][rows[node_id]] = vec
            return
        rows[node_id] = len(ids)
        ids.append(node_id)
        if kind not in self._mat:
            self._mat[kind] = vec
        else:
            self._mat[kind] = np.vstack([self._mat[kind], vec])

    def get(self, kind: str, node_id: str) -> np.ndarray | None:
        row = self._row.get(kind, {}).get(node_id)
        return None if row is None else self._mat[kind][row]

    def ids(self, kind: str) -> list[str]:
        return list(self._ids.get(kind, []))

    # ---- search -------------------------------------------------------------
    def search(self, kind: str, query: np.ndarray, k: int = 10,
               floor: float = -1.0, exclude: set[str] | None = None
               ) -> list[tuple[str, float]]:
        """Top-k (node_id, cosine) for one kind, cosine >= floor."""
        mat = self._mat.get(kind)
        if mat is None or len(mat) == 0:
            return []
        q = l2_normalize(query).reshape(-1)
        if q.shape[0] != mat.shape[1]:
            raise ValueError(
                f"query embedding dim {q.shape[0]} != index dim {mat.shape[1]} for "
                f"kind={kind!r}. The store was built with a different embedder; "
                f"re-ingest with the same --embedder, or rebuild the store.")
        sims = mat @ q                            # both unit-norm → cosine
        order = np.argsort(-sims)
        ids = self._ids[kind]
        out: list[tuple[str, float]] = []
        exclude = exclude or set()
        for idx in order:
            nid = ids[idx]
            if nid in exclude:
                continue
            s = float(sims[idx])
            if s < floor:
                break
            out.append((nid, s))
            if len(out) >= k:
                break
        return out

    # ---- persistence helpers ------------------------------------------------
    def iter_vectors(self):
        for kind, ids in self._ids.items():
            mat = self._mat[kind]
            for i, nid in enumerate(ids):
                yield kind, nid, mat[i]
