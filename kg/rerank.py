"""Cross-encoder reranker — the highest-ROI accuracy lever in the design.

A bi-encoder (bge) retrieves a candidate pool cheaply; a cross-encoder then jointly
encodes (query, passage) for each candidate and re-scores for true relevance. The
model is a CPU-friendly MS-MARCO MiniLM. It is loaded once per process and cached
across instances (the per-instance harness builds a fresh graph each question, so a
per-call reload would dominate wall-clock).

Degrades gracefully: if sentence-transformers / the model is unavailable, we keep
the input order (the bi-encoder/PPR ranking), so retrieval still works offline.
"""
from __future__ import annotations

_MODEL_CACHE: dict = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.available = True

    def _model(self):
        if self.model_name not in _MODEL_CACHE:
            from sentence_transformers import CrossEncoder
            _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
        return _MODEL_CACHE[self.model_name]

    def rerank(self, query: str, items: list[tuple[str, str]], k: int) -> list[str]:
        """items: list of (id, text). Returns the top-k ids by cross-encoder score,
        falling back to input order if the model can't load."""
        if not items:
            return []
        try:
            scores = self._model().predict([(query, text or "") for _, text in items])
        except Exception:  # noqa: BLE001 — keep bi-encoder/PPR order if the CE is unavailable
            self.available = False
            return [i for i, _ in items][:k]
        order = sorted(range(len(items)), key=lambda idx: -float(scores[idx]))
        return [items[idx][0] for idx in order[:k]]
