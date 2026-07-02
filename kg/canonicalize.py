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

import json
import math
import os
import re
from collections import Counter

import numpy as np

from .config import Config
from .embedders import Embedder
from .metering import UsageMeter
from .profiler import span as prof_span
from .models import (SELF_ENTITY_ID, Edge, EdgeType, EntityType, NodeType,
                     Provenance, entity_node, relation_tag_node, tag_node)
from .store import GraphStore, now_iso

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_REL_SEP = re.compile(r"[\s\-]+")

# First-person pronoun forms (personal-web mode, optional). These are already the
# OUTPUTS of normalize_key() on {"I","me","my","mine","myself"} — normalize_key is
# idempotent on them (lowercases, strips punctuation, leaves these short tokens alone),
# so the resolve_entity() guard matches by normalize_key(name) ∈ _FIRST_PERSON. "we"/"us"
# are deliberately excluded; "I" colliding with a Roman-numeral entity is an accepted
# limitation of personal-web mode (best on natural, non-numeral corpora).
_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself"})

# Relational function words stripped when computing a relation's *match key*, so
# "is_friend_of" and "is_friends_with" reduce to the same content word ("friend").
# "by" is POINTEDLY excluded — it marks the passive/inverse, so "managed_by" must
# stay distinct from "manages".
_REL_FUNCTION_WORDS = frozenset(
    "is are am was were be been being a an the of with to from in on at for as "
    "that who whom which into onto and".split())


# Opposite content lemmas (compared AFTER relation_content_key, so already function-
# word-stripped + singularized) that the L3 tie-breaker must NEVER merge, even though
# they sit close in embedding space — the exact hazard resolve_relation's merge-only
# 0.95 bar exists to avoid. Symmetric pairs.
_ANTONYM_LEMMAS = (
    frozenset({"friend", "enemy"}), frozenset({"ally", "enemy"}),
    frozenset({"ally", "rival"}), frozenset({"parent", "child"}),
    frozenset({"ancestor", "descendant"}), frozenset({"predecessor", "successor"}),
    frozenset({"buyer", "seller"}), frozenset({"teacher", "student"}),
    frozenset({"employer", "employee"}), frozenset({"leader", "follower"}),
    frozenset({"owner", "owned"}), frozenset({"love", "hate"}),
)


# Per-predicate cardinality (docs/TEMPORAL.md §5), defined over READABLE predicate
# surfaces and reduced to content keys at import so they match whatever
# relation_content_key() produces (which mangles inflections deterministically).
# Functional = single-valued: a new value supersedes the old (you can't live in two
# cities). Symmetric = orientation-free: A↔B is one fact (works_with, married_to).
_FUNCTIONAL_SURFACES = ("lives_in", "located_in", "employed_by", "born_in",
                        "died_in", "based_in", "headquartered_in", "capital_of", "ceo_of",
                        "president_of", "spouse_of", "married_to", "moved_to")
_SYMMETRIC_SURFACES = ("works_with", "collaborates_with", "married_to", "spouse_of",
                       "sibling_of", "friend_of", "is_friend_of", "colleague_of",
                       "partnered_with", "co_founder_of")


def relation_merge_vetoed(a: str, b: str) -> bool:
    """Deterministic guard run BEFORE any relation pair reaches the L3 LLM: force the
    two predicates to stay DISTINCT (never even offered as a merge option) when they are
    passive/inverse voices or known opposites. This guarantees a model slip can't produce
    a wrong relation merge — the whole reason resolve_relation is merge-only-no-link."""
    na, nb = normalize_relation(a), normalize_relation(b)
    if not na or not nb:
        return True
    # passive/inverse asymmetry: exactly one side carries the "_by" direction marker
    if na.endswith("_by") != nb.endswith("_by"):
        return True
    ka = set(relation_content_key(a).split("_"))
    kb = set(relation_content_key(b).split("_"))
    for pair in _ANTONYM_LEMMAS:
        ia, ib = pair & ka, pair & kb
        if ia and ib and ia != ib:
            return True
    return False


# --- L3 adjudication prompts (only used when config.l3_enabled and a key is present) ---
_L3_SYS = (
    "You are a careful knowledge-graph curator. Decide whether a NEW label is the SAME "
    "as one of a few EXISTING canonical labels, or genuinely new. The golden rule is "
    "UNDER-MERGE: when in doubt, answer NEW. A wrong merge corrupts the graph and is hard "
    "to undo; keeping two near-synonyms separate is cheap and recoverable. Reply with ONLY "
    "a JSON object and nothing else."
)
_L3_REL_PROMPT = (
    "Decide whether this NEW relationship predicate means the SAME directed relationship "
    "as one of the existing canonical predicates.\n\n"
    "MERGE only true synonyms of the same directed relationship (e.g. works_with ≈ "
    "collaborates_with; founded ≈ established; located_in ≈ situated_in).\n"
    "Answer NEW if it is merely related, narrower, or broader (e.g. manages vs leads; "
    "works_with vs reports_to).\n\n"
    "NEW predicate: {surface}\n"
    "Existing canonical predicates (most distinctive first):\n{candidates}\n\n"
    'Reply: {{"verdict": "<existing id from the list, or NEW>", "reason": "<one short clause>"}}'
)
_L3_ENT_PROMPT = (
    "Decide whether this NEW name refers to the SAME real-world {kind} as one of the "
    "existing canonical {kind}s.\n\n"
    "MERGE only clear aliases of one and the same thing (e.g. USA ≈ United States of "
    "America; NLP ≈ natural language processing).\n"
    "Answer NEW for things that merely share a word, an acronym, or a surname, and for a "
    "specific instance vs a broader category (e.g. Apple Inc. vs apple the fruit; Paris, "
    "Texas vs Paris, France).\n\n"
    "NEW {kind}: {surface}\n"
    "Existing canonical {kind}s (most distinctive first):\n{candidates}\n\n"
    'Reply: {{"verdict": "<existing id from the list, or NEW>", "reason": "<one short clause>"}}'
)

_L3_UNSET = object()


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
    return None


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


def _verb_stem(w: str) -> str:
    """Collapse common verb inflections so TENSE variants of one predicate share a key
    (docs/TEMPORAL.md §7 — fold tense onto the base predicate): lives/lived/living → liv,
    moves/moved → mov, works/worked → work. Deterministic and conservative; the "_by"
    passive marker and distinct content words are untouched, so inverses/antonyms stay
    distinct. Falls back to noun-singularization for non-verb tokens."""
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    return _singularize(w)


def relation_content_key(s: str) -> str:
    """Match key for relation consolidation — drop relational function words and stem the
    remaining content tokens (tense + number), so surface / inflectional / tense variants
    of the same predicate collapse while genuinely different predicates don't:

        is_friend_of / is_friends_with / friends-with → "friend"   (merge)
        lives_in / lived_in / living_in               → "liv"      (merge — tense folded)
        is_friend_of vs is_enemy_of                   → friend / enemy   (distinct: content word)
        manages      vs managed_by                    → manag / manag_by (distinct: "by" kept)

    If a label is *only* function words ("is_a"), fall back to the full form so it still
    resolves to a stable key.
    """
    norm = normalize_relation(s)
    if not norm:
        return ""
    content = [_verb_stem(t) for t in norm.split("_") if t not in _REL_FUNCTION_WORDS]
    return "_".join(content) if content else norm


# Cardinality lexicons reduced to content keys (see _FUNCTIONAL_SURFACES above).
FUNCTIONAL_KEYS = frozenset(relation_content_key(s) for s in _FUNCTIONAL_SURFACES)
SYMMETRIC_KEYS = frozenset(relation_content_key(s) for s in _SYMMETRIC_SURFACES)


def predicate_cardinality(surface: str) -> tuple[bool, bool]:
    """(functional, symmetric) for a relation surface, by content key."""
    ck = relation_content_key(surface)
    return ck in FUNCTIONAL_KEYS, ck in SYMMETRIC_KEYS


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
        self._l3_client = _L3_UNSET                   # lazy OpenAI client (L3 tie-breaker)
        self.l3_log: list[dict] = []                  # every L3 verdict, for the eval gate
        self.meter = UsageMeter()                     # L3 token/cost accounting (testrun)
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
        # personal-web: a reloaded store still routes first-person forms to the persisted
        # self anchor (OFF-path never touches this — the existing offline path is unchanged).
        if self.config.self_entity and self.store.has_node(SELF_ENTITY_ID):
            self._ensure_self()

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

    # ------------------------------------------------------------ L3 tie-breaker
    def _l3(self):
        """Lazy OpenAI client for the L3 adjudicator, or None if disabled / no key
        (offline parity: with no key the whole L3 path is skipped and resolve_* keep
        their deterministic under-merge default)."""
        if not self.config.l3_enabled:
            return None
        if self._l3_client is _L3_UNSET:
            self._l3_client = None
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    import openai
                    self._l3_client = openai.OpenAI()
                except Exception:  # noqa: BLE001 — missing dep / bad env → stay disabled
                    self._l3_client = None
        return self._l3_client

    def _node_name(self, node_id: str) -> str:
        n = self.store.get_node(node_id)
        return n.name if n else node_id

    def _l3_adjudicate(self, kind: str, surface: str,
                       cands: list[tuple[str, float]]) -> str | None:
        """Ask the LLM whether `surface` MERGEs into one of the gray-band candidates.
        Returns an existing node id to merge into, or None (mint a new node — the safe
        under-merge default on any uncertainty, parse failure, or error)."""
        client = self._l3()
        if client is None or not cands:
            return None
        # IDF-rank so the most discriminative existing labels sit at the primacy
        # position (fights lost-in-the-middle); cap at 5.
        cands = sorted(cands, key=lambda c: self.idf_weight(c[0]), reverse=True)[:5]
        ids = {cid for cid, _ in cands}
        lines = "\n".join(f"  [{cid}] {self._node_name(cid)}" for cid, _ in cands)
        if kind == "relation":
            prompt = _L3_REL_PROMPT.format(surface=normalize_relation(surface), candidates=lines)
        else:
            prompt = _L3_ENT_PROMPT.format(kind=kind, surface=surface, candidates=lines)
        verdict, reason = "NEW", ""
        try:
            with prof_span("canon.l3_llm"):
                msg = client.chat.completions.create(
                    model=self.config.l3_model, max_tokens=300, temperature=0,
                    messages=[{"role": "system", "content": _L3_SYS},
                              {"role": "user", "content": prompt}])
            self.meter.record("l3", self.config.l3_model, msg, label=surface)
            text = msg.choices[0].message.content or "" if msg.choices else ""
            data = _extract_json(text) or {}
            verdict = str(data.get("verdict", "NEW")).strip()
            reason = str(data.get("reason", ""))[:200]
        except Exception as e:  # noqa: BLE001 — any failure → under-merge (NEW)
            verdict, reason = "NEW", f"error: {e!r}"
        chosen = verdict if verdict in ids else None
        self.l3_log.append({"kind": kind, "surface": surface,
                            "candidates": [(cid, round(s, 3)) for cid, s in cands],
                            "verdict": chosen or "NEW", "reason": reason})
        return chosen

    def _l3_relation(self, surface: str, hits: list[tuple[str, float]]) -> str | None:
        """Gray-band relation candidates, minus any vetoed inverse/antonym pair."""
        cands = [(cid, s) for cid, s in hits
                 if not relation_merge_vetoed(surface, self._node_name(cid))]
        return self._l3_adjudicate("relation", surface, cands)

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
            hits = self.store.vectors.search("tag", vec, k=5,
                                            floor=self.config.syn_link_threshold)
            if hits and hits[0][1] >= self.config.syn_merge_threshold:  # L2 hard merge
                tid = hits[0][0]
                self._add_alias(tid, surface)
                self._tag_keys[key] = tid
                return tid
            # L3: gray band [syn_link, syn_merge) → ask the LLM merge-or-new (no-op if disabled)
            gray = [(c, s) for c, s in hits if s < self.config.syn_merge_threshold]
            tid = self._l3_adjudicate("tag", surface, gray)
            if tid:
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
            self.store.touch_node(tag_id)
            self._tag_keys.setdefault(normalize_key(surface), tag_id)

    # -------------------------------------------------------------- self anchor
    def _ensure_self(self) -> str:
        """Idempotently create/return the canonical first-person "self" anchor
        (personal-web mode). Mints a lean PERSON entity at SELF_ENTITY_ID on first call,
        embeds its NAME once (mirroring resolve_entity), and (re)registers every
        first-person form + the display name's key so "i"/"me"/"my"/… all route here —
        including after a reload. Safe to call repeatedly. Only ever reached behind the
        config.self_entity guard, so the OFF-path is untouched."""
        node = self.store.get_node(SELF_ENTITY_ID)
        if node is None:
            node = entity_node(SELF_ENTITY_ID, name=self.config.self_name,
                               etype=EntityType.PERSON, ts=now_iso())
            self.store.add_node(node)
        elif node.name != self.config.self_name:   # honour a changed --self across sessions
            node.name = self.config.self_name
            node.last_modified = now_iso()
            self.store.touch_node(SELF_ENTITY_ID)
        # The self anchor carries NO entity embedding ON PURPOSE: it is resolved only by the
        # first-person pronoun guard + the lexical routes below, never by embedding
        # similarity. Keeping it out of the "entity" vector index is what makes a --self
        # that happens to share a real entity's NAME (another "Jude") unable to L2-merge that
        # entity into self — neither the name key nor a name embedding routes here, only the
        # pronouns. The forms persist as aliases so a reload reconstructs the routes (_reindex).
        merged = sorted(set(node.aliases) | set(_FIRST_PERSON))
        if merged != node.aliases:
            node.aliases = merged
            self.store.touch_node(SELF_ENTITY_ID)
        for key in _FIRST_PERSON:
            self._entity_keys[key] = SELF_ENTITY_ID
        return SELF_ENTITY_ID

    # ---------------------------------------------------------------- entities
    def resolve_entity(self, name: str, etype: EntityType) -> str | None:
        # personal-web: first-person references collapse onto ONE stable self anchor,
        # BEFORE the normal L1 lookup. OFF by default → this never fires and resolution
        # is byte-for-byte the existing path.
        if self.config.self_entity and normalize_key(name) in _FIRST_PERSON:
            return self._ensure_self()
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
            hits = self.store.vectors.search("entity", vec, k=5,
                                            floor=self.config.syn_link_threshold)
            if hits and hits[0][1] >= self.config.syn_merge_threshold:  # L2 hard merge
                eid = hits[0][0]
                self._entity_keys[key] = eid
                return eid
            gray = [(c, s) for c, s in hits if s < self.config.syn_merge_threshold]
            eid = self._l3_adjudicate("entity", name, gray)
            if eid:
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
            hits = self.store.vectors.search("relation", vec, k=5,
                                            floor=self.config.rel_gray_floor)
            if hits and hits[0][1] >= self.config.rel_syn_merge_threshold:  # L2 hard merge
                rid = hits[0][0]
                self._add_relation_alias(rid, display)
                self._relation_keys[key] = rid
                return rid
            # L3: gray band [rel_gray_floor, rel_syn_merge_threshold), AFTER the
            # deterministic antonym/inverse/passive veto (no-op if L3 disabled)
            gray = [(c, s) for c, s in hits if s < self.config.rel_syn_merge_threshold]
            rid = self._l3_relation(surface, gray)
            if rid:
                self._add_relation_alias(rid, display)
                self._relation_keys[key] = rid
                return rid
        rid = self._new_id("rel")
        functional, symmetric = predicate_cardinality(surface)
        node = relation_tag_node(rid, canonical=display, ts=now_iso(),
                                 functional=functional, symmetric=symmetric)
        self.store.add_node(node)
        self.store.vectors.add("relation", rid, vec)
        self._relation_keys[key] = rid
        return rid

    def _add_relation_alias(self, rid: str, display: str) -> None:
        node = self.store.get_node(rid)
        if node and display != node.name and display not in node.aliases:
            node.aliases.append(display)
            self.store.touch_node(rid)
        self._relation_keys.setdefault(relation_content_key(display), rid)

    # ---------------------------------------------------------- IDF / specificity
    def bump_doc_frequency(self, node_id: str) -> None:
        n = self.store.get_node(node_id)
        if n is not None:
            n.doc_frequency += 1
            self.store.touch_node(node_id)

    def idf_weight(self, node_id: str) -> float:
        """1 / (1 + df) style specificity — generic tags downranked (HippoRAG/TaxoGen)."""
        n = self.store.get_node(node_id)
        n_eps = max(1, self.store.episode_count())
        if n is None or n.doc_frequency <= 0:
            return 1.0
        return math.log(1 + n_eps / n.doc_frequency)
