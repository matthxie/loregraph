"""Embedders (docs/ARCHITECTURE.md §4).

Primary: local `sentence-transformers` (BAAI/bge-small-en-v1.5), fully offline once
the model is cached. Fallback: a dependency-free hashing embedder so the pipeline
still runs (and tests stay deterministic) when torch isn't installed. Both return
L2-normalised float32 (n, dim) matrices; the VectorIndex assumes unit norm.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Config
from .vectors import l2_normalize

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """The real, semantic embedder. Model loads lazily on first use."""

    def __init__(self, model_name: str, dim: int):
        self.name = f"st:{model_name}"
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            getdim = (getattr(self._model, "get_embedding_dimension", None)
                      or self._model.get_sentence_embedding_dimension)
            self.dim = getdim()
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._ensure()
        vecs = model.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Deterministic offline fallback: signed feature hashing of words + char
    3-grams into `dim` buckets. Captures lexical overlap (enough to demo the full
    pipeline and keep tests deterministic), without torch."""

    def __init__(self, dim: int):
        self.name = "hashing"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = _WORD.findall((text or "").lower())
        feats: list[str] = list(words)
        for w in words:
            padded = f"#{w}#"
            feats.extend(padded[i:i + 3] for i in range(len(padded) - 2))
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            bucket = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[bucket] += sign
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.vstack([self._vec(t) for t in texts]))


def get_embedder(config: Config) -> Embedder:
    choice = config.embedder
    if choice in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(config.embed_model, config.embed_dim)
    if choice == "hashing":
        return HashingEmbedder(config.embed_dim)
    # auto: use the real embedder iff sentence-transformers is importable
    import importlib.util
    if importlib.util.find_spec("sentence_transformers") is not None:
        return SentenceTransformerEmbedder(config.embed_model, config.embed_dim)
    return HashingEmbedder(config.embed_dim)
