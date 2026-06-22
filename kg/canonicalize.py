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
                     entity_node, relation_tag_node, tag_node)
from .store import GraphStore, now_iso

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_REL_SEP = re.compile(r"[\s\-]+")

# Relational function words stripped when computing a relation's *match key*, so
# "is_friend_of" and "is_friends_with" reduce to the same content word ("friend").
# "by" is POINTEDLY excluded — it marks the passive/inverse, so "managed_by" must
# stay distinct from "manages".
_REL_FUNCTION_WORDS = frozenset(
    "is are am was were be been being a an the of with to from in on at for as "
    "that who whom which into onto and".split())


def _singularize(w: str) -> str:
    """Light noun-oriented depluralisation of one token (shared by the tag and
    relation keys). Imperfect on verbs but deterministic; the embedding gate and
    the L3 batch pass back it up."""
    if len(w) <= 4:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("es") and not w.endswith(("ses", "zes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def normalize_relation(s: str) -> str:
    """Display/canonical form of a relation label: lowercase, depunctuate, join
    tokens with underscores. Keeps every token (stays readable), unlike the match
    key. "is friends with" / "is-friends-with" / "Is Friends With" → "is_friends_with".
    """
    s = _PUNCT.sub(" ", (s or "").lower())
    s = _REL_SEP.sub("_", s.strip())
    return re.sub(r"_+", "_", s).strip("_")


def relation_content_key(s: str) -> str:
    """Match key for relation consolidation — drop relational function words and
    singularize the remaining content tokens, so surface / inflectional variants of
    the same predicate collapse while genuinely different predicates don't:

        is_friend_of / is_friends_with / friends-with → "friend"   (merge)
        works_with   / work with                      → "work"     (merge)
        is_friend_of vs is_enemy_of                   → friend / enemy   (distinct: content word)
        manages      vs managed_by                    → manag / managed_by (distinct: "by" kept)

    If a label is *only* function words ("is_a"), fall back to the full form so it
    still resolves to a stable key.
    """
    norm = normalize_relation(s)
    if not norm:
        return ""
    content = [_singularize(t) for t in norm.split("_") if t not in _REL_FUNCTION_WORDS]
    return "_".join(content) if content else norm


def normalize_key(s: str) -> str:
    s = _PUNCT.sub(" ", (s or "").lower())
    s = _WS.sub(" ", s).strip()
    # light singularisation of the final token (avoid mangling short words)
    toks = s.split()
    if toks:
        toks[-1] = _singularize(toks[-1])
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
        self._relation_keys: dict[str, str] = {}  # normalized key -> relation-tag id
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
            elif n.ntype == NodeType.RELATION:
                self._relation_keys[relation_content_key(n.name)] = n.id
                for a in n.aliases:
                    self._relation_keys.setdefault(relation_content_key(a), n.id)
        self._next["tag"] = sum(1 for n in self.store.nodes.values()
                                if n.ntype == NodeType.TAG)
        self._next["entity"] = sum(1 for n in self.store.nodes.values()
                                   if n.ntype == NodeType.ENTITY)
        self._next["rel"] = sum(1 for n in self.store.nodes.values()
                                if n.ntype == NodeType.RELATION)

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

    # --------------------------------------------------------- relationship tags
    def resolve_relation(self, surface: str) -> str | None:
        """Consolidate an LLM-generated relationship label into a canonical
        RelationTagNode — the same two-layer move as `resolve_tag`, tuned for
        predicates:

          L1  CONTENT-KEY exact hash (`relation_content_key`): drop relational
              function words + singularize, so "is_friend_of" / "is_friends_with"
              collapse on the content word "friend" — while "is_enemy_of" (different
              content word) and "managed_by" (passive "by" kept) stay distinct.
          L2  embedding-synonymy MERGE only, at a HIGH bar (rel_syn_merge_threshold,
              default 0.95) and behind the entropy guard — catches synonyms with
              *different* content words ("collaborates_with" ↔ "works_with") that L1
              can't. Unlike tags we do NOT add SIMILAR_TO links between near-miss
              predicates: antonyms/inverses sit close in embedding space but must
              stay distinct, and relation-tag nodes aren't traversed, so a link would
              add drift risk for no retrieval gain.

        The node keeps a READABLE canonical name (the first surface's display form);
        later variants become aliases.
        """
        surface = (surface or "").strip()
        if not surface:
            return None
        display = normalize_relation(surface)
        key = relation_content_key(surface)
        if not key:
            return None
        if key in self._relation_keys:                 # L1 hit (content key)
            rid = self._relation_keys[key]
            self._add_relation_alias(rid, display)
            return rid
        vec = self._embed(surface)
        if self._entropy_ok(surface):                  # L2 high-bar merge only
            hits = self.store.vectors.search("relation", vec, k=1,
                                            floor=self.config.rel_syn_merge_threshold)
            if hits:
                rid = hits[0][0]
                self._add_relation_alias(rid, display)
                self._relation_keys[key] = rid
                return rid
        rid = self._new_id("rel")
        node = relation_tag_node(rid, canonical=display, ts=now_iso())
        self.store.add_node(node)
        self.store.vectors.add("relation", rid, vec)
        self._relation_keys[key] = rid
        return rid

    def _add_relation_alias(self, rid: str, display: str) -> None:
        node = self.store.get_node(rid)
        if node and display != node.name and display not in node.aliases:
            node.aliases.append(display)
        self._relation_keys.setdefault(relation_content_key(display), rid)

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
