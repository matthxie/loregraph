"""Tag/entity canonicalization & drift control (docs/ARCHITECTURE.md §3).

Layered, link-biased (under-merge):

  L1  exact/normalized hash         — collapse "Natural Language Processing" /
                                       "natural-language processing".
  L2  embedding synonymy gate       — cosine > link τ → SIMILAR_TO *link* (not merge);
                                       cosine > merge τ → merge. An entropy guard stops
                                       short/low-entropy strings ("AI","US") from fuzzy
                                       merging (graphiti).
  (L3 batch reconciliation is deferred — see §3.)

Also maintains `doc_frequency` per tag/entity so retrieval can weight by node
specificity / inverse document frequency (HippoRAG).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .config import Config
from .embedders import Embedder
from .models import (Edge, EdgeType, EntityType, NodeType, Provenance,
                     entity_node, tag_node)
from .store import GraphStore, now_iso

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_key(s: str) -> str:
    s = _PUNCT.sub(" ", (s or "").lower())
    s = _WS.sub(" ", s).strip()
    # light singularisation of the final token (avoid mangling short words)
    toks = s.split()
    if toks and len(toks[-1]) > 4:
        w = toks[-1]
        if w.endswith("ies"):
            toks[-1] = w[:-3] + "y"
        elif w.endswith("es") and not w.endswith(("ses", "zes")):
            toks[-1] = w[:-2]
        elif w.endswith("s") and not w.endswith("ss"):
            toks[-1] = w[:-1]
    return " ".join(toks)


def char_entropy(s: str) -> float:
    s = (s or "").lower()
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


class Canonicalizer:
    def __init__(self, store: GraphStore, embedder: Embedder, config: Config):
        self.store = store
        self.embedder = embedder
        self.config = config
        self._tag_keys: dict[str, str] = {}     # normalized key -> tag node id
        self._entity_keys: dict[str, str] = {}   # normalized key -> entity node id
        self._emb_cache: dict[str, np.ndarray] = {}  # surface -> embedding (batch primed)
        self._next = Counter()
        self._reindex()

    def prime_embeddings(self, surfaces: list[str]) -> None:
        """Batch-embed unique surfaces up front (≫ faster than per-resolve calls)."""
        todo = sorted({s.strip() for s in surfaces if s and s.strip()
                       and s.strip() not in self._emb_cache})
        if not todo:
            return
        vecs = self.embedder.embed(todo)
        for s, v in zip(todo, vecs):
            self._emb_cache[s] = v

    def _embed(self, surface: str) -> np.ndarray:
        v = self._emb_cache.get(surface.strip())
        if v is None:
            v = self.embedder.embed([surface])[0]
            self._emb_cache[surface.strip()] = v
        return v

    def _reindex(self) -> None:
        """Rebuild key→id maps from a loaded store."""
        for n in self.store.nodes.values():
            if n.ntype == NodeType.TAG:
                self._tag_keys[normalize_key(n.name)] = n.id
                for a in n.aliases:
                    self._tag_keys.setdefault(normalize_key(a), n.id)
            elif n.ntype == NodeType.ENTITY:
                self._entity_keys[normalize_key(n.name)] = n.id
        self._next["tag"] = sum(1 for n in self.store.nodes.values()
                                if n.ntype == NodeType.TAG)
        self._next["entity"] = sum(1 for n in self.store.nodes.values()
                                   if n.ntype == NodeType.ENTITY)

    def _new_id(self, prefix: str) -> str:
        nid = f"{prefix}_{self._next[prefix]:04d}"
        self._next[prefix] += 1
        while self.store.has_node(nid):  # guard against collisions
            nid = f"{prefix}_{self._next[prefix]:04d}"
            self._next[prefix] += 1
        return nid

    # ------------------------------------------------------------------ shared
    def _synonymy(self, kind: str, surface: str, embedding: np.ndarray,
                  new_id: str) -> None:
        """L2: link/merge against existing same-kind nodes by cosine."""
        if not self._entropy_ok(surface):
            return
        hits = self.store.vectors.search(kind, embedding, k=3,
                                         floor=self.config.syn_link_threshold,
                                         exclude={new_id})
        for other_id, cos in hits:
            self.store.add_edge(Edge(
                src=new_id, dst=other_id, etype=EdgeType.SIMILAR_TO,
                provenance=Provenance.SIMILAR, confidence=round(cos, 3),
                weight=round(cos, 3)))

    def _entropy_ok(self, surface: str) -> bool:
        key = normalize_key(surface)
        return (len(key) >= self.config.entropy_min_chars
                and char_entropy(key) >= self.config.entropy_min_bits)

    # -------------------------------------------------------------------- tags
    def resolve_tag(self, surface: str) -> str | None:
        surface = (surface or "").strip()
        if not surface:
            return None
        key = normalize_key(surface)
        if not key:
            return None
        if key in self._tag_keys:                      # L1 hit
            tid = self._tag_keys[key]
            self._add_alias(tid, surface)
            return tid
        vec = self._embed(surface)
        # L2 merge gate (high bar) — only if entropy guard allows. NOTE: this uses a
        # single global cosine threshold; TaxoCom's local-neighborhood thresholding
        # (§3) is an accepted MVP simplification at ~1k tags.
        if self._entropy_ok(surface):
            hits = self.store.vectors.search("tag", vec, k=1,
                                            floor=self.config.syn_merge_threshold)
            if hits:
                tid = hits[0][0]
                self._add_alias(tid, surface)
                self._tag_keys[key] = tid
                return tid
        tid = self._new_id("tag")
        node = tag_node(tid, canonical=surface.lower(), ts=now_iso())
        self.store.add_node(node)
        self.store.vectors.add("tag", tid, vec)
        self._tag_keys[key] = tid
        self._synonymy("tag", surface, vec, tid)       # link (not merge)
        return tid

    def _add_alias(self, tag_id: str, surface: str) -> None:
        node = self.store.get_node(tag_id)
        if node and surface.lower() != node.name and surface.lower() not in node.aliases:
            node.aliases.append(surface.lower())
            self._tag_keys.setdefault(normalize_key(surface), tag_id)

    # ---------------------------------------------------------------- entities
    def resolve_entity(self, name: str, etype: EntityType) -> str | None:
        name = (name or "").strip()
        if not name:
            return None
        key = normalize_key(name)
        if not key:
            return None
        if key in self._entity_keys:                   # L1 hit
            return self._entity_keys[key]
        vec = self._embed(name)
        if self._entropy_ok(name):
            hits = self.store.vectors.search("entity", vec, k=1,
                                            floor=self.config.syn_merge_threshold)
            if hits:
                eid = hits[0][0]
                self._entity_keys[key] = eid
                return eid
        eid = self._new_id("entity")
        node = entity_node(eid, name=name, etype=etype, ts=now_iso())
        self.store.add_node(node)
        self.store.vectors.add("entity", eid, vec)
        self._entity_keys[key] = eid
        self._synonymy("entity", name, vec, eid)       # link (not merge)
        return eid

    # ---------------------------------------------------------- IDF / specificity
    def bump_doc_frequency(self, node_id: str) -> None:
        n = self.store.get_node(node_id)
        if n is not None:
            n.doc_frequency += 1

    def idf_weight(self, node_id: str) -> float:
        """1 / (1 + df) style specificity — generic tags downranked (HippoRAG/TaxoGen)."""
        n = self.store.get_node(node_id)
        n_objs = max(1, self.store.object_count())
        if n is None or n.doc_frequency <= 0:
            return 1.0
        return math.log(1 + n_objs / n.doc_frequency)
