"""Embedders (docs/ARCHITECTURE.md §4).

Local `sentence-transformers` (BAAI/bge-small-en-v1.5), fully local once the model is
cached (no network at run time, no API key). Returns L2-normalised float32 (n, dim)
matrices; the VectorIndex assumes unit norm.

The dependency-free HashingEmbedder (lexical feature-hashing) was removed: it captured
only lexical overlap, not semantics, so it isn't representative of the live graph.
Embeddings are now semantic-only.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Config


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


_MODEL_CACHE: dict = {}


class SentenceTransformerEmbedder:
    """The semantic embedder. Model loads lazily on first use, once per process."""

    def __init__(self, model_name: str, dim: int):
        self.name = f"st:{model_name}"
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _ensure(self):
        if self._model is None:
            model = _MODEL_CACHE.get(self.model_name)
            if model is None:
                from sentence_transformers import SentenceTransformer
                model = _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
            self._model = model
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


def get_embedder(config: Config) -> Embedder:
    """The semantic embedder (sentence-transformers). `embedder` is accepted as
    'st'/'auto' for back-compat; both return the same SentenceTransformerEmbedder."""
    return SentenceTransformerEmbedder(config.embed_model, config.embed_dim)
