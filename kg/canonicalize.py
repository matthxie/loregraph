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
import re
from collections import Counter

import numpy as np

from .backoff import call_with_backoff
from .config import Config
from .embedders import Embedder
from .llm_client import llm_available, make_client, resolve_model
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

# Relational function words stripped from the INTERIOR of a relation's *match key*, so
# "is_friend_of" and "friend_of" reduce to the same content key ("friend_of").
# "by" is POINTEDLY excluded — it marks the passive/inverse, so "managed_by" must
# stay distinct from "manages". A TRAILING function word is likewise kept by
# relation_content_key(): the final preposition is the predicate's argument-structure
# marker ("works_with" symmetric comitative vs "works_at"/"works_on" directional), so
# stripping it folded directional predicates onto symmetric content keys and got their
# orientation pinned ("Acme works_at me"-order facts).
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
# Surfaces reduce to content keys at import, which keep the trailing preposition — so
# "friend_of" and "friends_with" are two keys (both listed, both symmetric), and a
# directional "works_at"/"works_on" can never collide into "works_with"'s symmetric key.
_SYMMETRIC_SURFACES = ("works_with", "collaborates_with", "married_to", "spouse_of",
                       "sibling_of", "friend_of", "is_friend_of", "friends_with",
                       "colleague_of", "partnered_with", "co_founder_of")


def _trailing_marker(s: str) -> str:
    """The kept trailing function word of a content key ('' when it ends in a content token)."""
    last = relation_content_key(s).rsplit("_", 1)[-1]
    return last if last in _REL_FUNCTION_WORDS else ""


def relation_merge_vetoed(a: str, b: str) -> bool:
    """Deterministic guard run BEFORE any relation pair reaches the L3 LLM: force the
    two predicates to stay DISTINCT (never even offered as a merge option) when they are
    passive/inverse voices, argument-structure mismatches, or known opposites. This
    guarantees a model slip can't produce a wrong relation merge — the whole reason
    resolve_relation is merge-only-no-link."""
    na, nb = normalize_relation(a), normalize_relation(b)
    if not na or not nb:
        return True
    # passive/inverse asymmetry: exactly one side carries the "_by" direction marker
    if na.endswith("_by") != nb.endswith("_by"):
        return True
    # argument-structure mismatch: different trailing markers (works_at vs works_with) must
    # never merge even when embeddings sit close — same rationale as the "_by" rule
    if _trailing_marker(a) != _trailing_marker(b):
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
    """Match key for relation consolidation — drop INTERIOR relational function words and stem
    the remaining content tokens (tense + number), KEEPING a trailing function word: like the
    pointedly-unstripped "_by" (passive/inverse marker), the final preposition is the
    predicate's argument-structure marker, so dropping it folded directional predicates onto
    symmetric content keys (works_at → "work" == works_with → "work") and got their
    orientation pinned. Surface / inflectional / tense variants of the same predicate still
    collapse while genuinely different predicates don't:

        is_friend_of / friend_of / friend-of → "friend_of"  (merge — interior words dropped)
        lives_in / lived_in / living_in      → "liv_in"     (merge — tense folded)
        works_with / worked_with             → "work_with"  (merge)
        works_at  vs works_with              → work_at / work_with   (distinct: trailing marker)
        friend_of vs friends_with            → friend_of / friend_with (distinct nodes; BOTH
                                               symmetric via _SYMMETRIC_SURFACES)
        is_friend_of vs is_enemy_of          → friend_of / enemy_of  (distinct: content word)
        manages      vs managed_by           → manag / manag_by      (distinct: "by" kept)

    If a label is *only* function words ("is_a"), fall back to the full form so it still
    resolves to a stable key.
    """
    norm = normalize_relation(s)
    if not norm:
        return ""
    toks = [t for t in norm.split("_") if t]
    content = [_verb_stem(t) for t in toks if t not in _REL_FUNCTION_WORDS]
    if not content:
        return norm
    if toks[-1] in _REL_FUNCTION_WORDS:
        content.append(toks[-1])       # trailing argument-structure marker kept as-is
    return "_".join(content)


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


# Typed proper-noun entity kinds: their surfaces are NAMES, not common nouns, so the
# last-token depluralisation in normalize_key() is unwanted — it makes "Socks"→"sock"
# collide with "Sock" and "Paris"→"pari" with "Pari". Skip singularization for these.
_PROPER_TYPES = frozenset({EntityType.PERSON, EntityType.PLACE, EntityType.ORG})


def _is_proper_surface(name: str, etype: EntityType | None) -> bool:
    """A capitalized proper noun of a person/place/org type — a NAME, not a common noun."""
    return (etype in _PROPER_TYPES and bool(name)
            and name.lstrip()[:1].isupper())


def entity_key(name: str, etype: EntityType | None = None) -> str:
    """L1 key for an ENTITY surface. Same as normalize_key, but for a capitalized proper-noun
    person/place/org it does NOT singularize the last token — so two distinct names differing
    only by a trailing 's' ("Socks"/"Sock", "Paris"/"Pari") stay distinct instead of L1-merging
    on a bogus depluralisation. Common-noun / term / other surfaces keep singularization."""
    if _is_proper_surface(name, etype):
        s = _PUNCT.sub(" ", (name or "").lower())
        return _WS.sub(" ", s).strip()
    return normalize_key(name)


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
        self._l3_client = _L3_UNSET                   # lazy LLM client (L3 tie-breaker)
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
                self._entity_keys[entity_key(n.name, n.entity_type)] = n.id
                for a in n.aliases:
                    self._entity_keys.setdefault(entity_key(a, n.entity_type), n.id)
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
                  new_id: str, link_threshold: float | None = None) -> None:
        """L2: link/merge against existing same-kind nodes by cosine. `link_threshold`
        overrides the (entity) default so the TAG path can link at a looser bar without
        changing entity synonymy — entities call this with the default."""
        if not self._entropy_ok(surface):
            return
        floor = (self.config.syn_link_threshold if link_threshold is None
                 else link_threshold)
        hits = self.store.vectors.search(kind, embedding, k=3,
                                         floor=floor,
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
        """Lazy LLM client for the L3 adjudicator, or None if disabled / no provider
        (offline parity: with no provider the whole L3 path is skipped and resolve_* keep
        their deterministic under-merge default)."""
        if not self.config.l3_enabled:
            return None
        if self._l3_client is _L3_UNSET:
            self._l3_client = None
            if llm_available():
                try:
                    self._l3_client = make_client()
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
        # explicit config override if the user changed it, else the active provider's
        # default (None for CLI providers → the CLI's own default model).
        model = resolve_model(self.config.l3_model, "gpt-4o-mini")
        try:
            with prof_span("canon.l3_llm"):
                msg = call_with_backoff(lambda: client.chat.completions.create(
                    model=model, max_tokens=300, temperature=0,
                    messages=[{"role": "system", "content": _L3_SYS},
                              {"role": "user", "content": prompt}]))
            self.meter.record("l3", model or "cli-default", msg, label=surface)
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
        """Create-or-reuse a canonical TAG node: exact-key (L1) reuse, else embedding
        merge (L2, at the looser TAG bar) into a near-identical existing tag, else mint
        a new node and LINK it to similar tags (SIMILAR_TO). Uses the tag-specific
        thresholds so it can merge/link more freely than entity resolution."""
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
        # L2 merge gate — uses the TAG thresholds (looser than entity), only if the entropy
        # guard allows. NOTE: this uses a single global cosine threshold; TaxoCom's
        # local-neighborhood thresholding (§3) is an accepted MVP simplification at ~1k tags.
        if self._entropy_ok(surface):
            hits = self.store.vectors.search("tag", vec, k=5,
                                            floor=self.config.tag_syn_link_threshold)
            if hits and hits[0][1] >= self.config.tag_syn_merge_threshold:  # L2 hard merge
                tid = hits[0][0]
                self._add_alias(tid, surface)
                self._tag_keys[key] = tid
                return tid
            # L3: gray band [link, merge) → ask the LLM merge-or-new (no-op if disabled)
            gray = [(c, s) for c, s in hits if s < self.config.tag_syn_merge_threshold]
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
        self._synonymy("tag", surface, vec, tid,       # link (not merge), at the looser bar
                       link_threshold=self.config.tag_syn_link_threshold)
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

    # ------------------------------------------------------- entity type guards
    def _types_compatible(self, existing: EntityType | None, incoming: EntityType) -> bool:
        """Two entity types may share one node only if they don't CONTRADICT. OTHER (the
        catch-all) is compatible with anything — a specific type just refines it. Two DIFFERENT
        specific types (person vs place, org vs person) contradict → keep them apart rather than
        silently collapsing a "Jordan the person" onto "Jordan the country"."""
        if existing is None or existing == EntityType.OTHER or incoming == EntityType.OTHER:
            return True
        return existing == incoming

    def _reconcile_type(self, node_id: str, incoming: EntityType) -> None:
        """UPGRADE an existing OTHER-typed anchor to a specific type when a typed mention
        resolves onto it (the reverse — a specific node meeting an OTHER mention — is a no-op;
        the specific type already wins). Only ever narrows OTHER → specific, never re-types a
        node that already carries a specific type."""
        node = self.store.get_node(node_id)
        if node is None:
            return
        if node.entity_type in (None, EntityType.OTHER) and incoming != EntityType.OTHER:
            node.entity_type = incoming
            node.last_modified = now_iso()
            self.store.touch_node(node_id)

    def _add_entity_alias(self, entity_id: str, surface: str) -> None:
        """Persist a merged surface as an alias on the entity node (mirrors the tag path's
        _add_alias) so identity survives a save/reload — _reindex rebuilds keys from name AND
        aliases. Register the alias key too, so an intra-session repeat of the surface L1-hits."""
        node = self.store.get_node(entity_id)
        if node is None:
            return
        low = surface.strip().lower()
        if low and low != node.name.lower() and low not in node.aliases:
            node.aliases.append(low)
            self.store.touch_node(entity_id)
        self._entity_keys.setdefault(entity_key(surface, node.entity_type), entity_id)

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
        key = entity_key(name, etype)
        if not key:
            return None
        if key in self._entity_keys:                   # L1 hit (normalized name / proper-noun)
            eid = self._entity_keys[key]
            existing = self.store.get_node(eid)
            # type reconciliation: an L1 name collision across INCOMPATIBLE specific types
            # (person vs place) is two different real-world things sharing a name — fall
            # through and mint a distinct anchor instead of collapsing them. Compatible / OTHER
            # → reuse (and upgrade an OTHER anchor to the specific type the mention supplies).
            if existing is None or self._types_compatible(existing.entity_type, etype):
                self._reconcile_type(eid, etype)
                return eid
        vec = self._embed(name)
        if self._entropy_ok(name):
            hits = self.store.vectors.search("entity", vec, k=5,
                                            floor=self.config.syn_link_threshold)
            if hits and hits[0][1] >= self.config.syn_merge_threshold:  # L2 hard merge
                eid = hits[0][0]
                # only merge onto a type-compatible node — a fuzzy name match across
                # contradictory specific types is a false merge, so mint instead.
                cand = self.store.get_node(eid)
                if cand is None or self._types_compatible(cand.entity_type, etype):
                    self._add_entity_alias(eid, name)     # persist identity across reloads
                    self._entity_keys.setdefault(key, eid)
                    self._reconcile_type(eid, etype)
                    return eid
            else:
                gray = [(c, s) for c, s in hits if s < self.config.syn_merge_threshold]
                eid = self._l3_adjudicate("entity", name, gray)
                if eid:
                    cand = self.store.get_node(eid)
                    if cand is None or self._types_compatible(cand.entity_type, etype):
                        self._add_entity_alias(eid, name)
                        self._entity_keys.setdefault(key, eid)
                        self._reconcile_type(eid, etype)
                        return eid
        eid = self._new_id("entity")
        node = entity_node(eid, name=name, etype=etype, ts=now_iso())
        self.store.add_node(node)
        self.store.vectors.add("entity", eid, vec)
        self._entity_keys.setdefault(key, eid)   # an incompatible-type L1 collision keeps the
        #                                          first node's key; this distinct anchor is
        #                                          still reachable by embedding / its own name.
        self._synonymy("entity", name, vec, eid)       # link (not merge)
        return eid

    # --------------------------------------------------------- relationship tags
    def resolve_relation(self, surface: str) -> str | None:
        """Consolidate an LLM-generated relationship label into a canonical
        RelationTagNode — the same two-layer move as `resolve_tag`, tuned for
        predicates:

          L1  CONTENT-KEY exact hash (`relation_content_key`): drop interior relational
              function words + stem, keeping the trailing argument-structure marker, so
              "is_friend_of" / "friend_of" collapse on "friend_of" — while "is_enemy_of"
              (different content word), "friends_with" (different trailing marker) and
              "managed_by" (passive "by" kept) stay distinct.
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

    # ------------------------------------------------------------ merge (L3 drain)
    def _readd_edge(self, src: str, dst: str, data: dict) -> bool:
        """Re-attach an existing edge's FULL attribute dict onto a new (src, dst) — used
        when apply_merge re-points a merged node's edges. Rebuilding an Edge and routing it
        through store.add_edge preserves every attribute (incl. the closed_at / confirmed_by /
        disputed_by mutation provenance), re-pins symmetric orientation, gives a fact a fresh
        seq, and collapses onto a stronger duplicate exactly like any other add. Drops
        (returns False) a self-loop; a weaker structural duplicate collapses silently inside
        add_edge (still counted as moved — the receipt is informational)."""
        if src == dst:
            return False
        self.store.add_edge(Edge(
            src=src, dst=dst, etype=EdgeType(data["etype"]),
            provenance=Provenance(data.get("provenance", Provenance.DERIVED.value)),
            confidence=data.get("confidence", 1.0), weight=data.get("weight", 1.0),
            rel_tag=data.get("rel_tag"), valid_at=data.get("valid_at", ""),
            invalid_at=data.get("invalid_at", ""),
            belief=data.get("belief", "asserted"),
            episode_id=data.get("episode_id", ""), via=list(data.get("via") or []),
            valid=data.get("valid", True), created_at=data.get("created_at", ""),
            disputed_by=list(data.get("disputed_by") or []),
            confirmed_by=list(data.get("confirmed_by") or []),
            closed_at=data.get("closed_at", ""),
            closed_by_episode=data.get("closed_by_episode", ""),
            retracted_at=data.get("retracted_at", ""),
            retracted_by_episode=data.get("retracted_by_episode", "")))
        return True

    def apply_merge(self, survivor_id: str, loser_id: str) -> dict:
        """Execute a reconcile-ledger merge (loser → survivor) — the deferred L3 adjudication
        run as a review pass at zero API cost. Re-points every edge incident to the loser onto
        the survivor (dropping self-loops), folds the loser's name + aliases into the survivor,
        deletes the loser node, and repairs the key maps so the loser's normalized keys resolve
        to the survivor (a later mention of the loser's surface L1-hits the survivor). REFUSES
        to merge the self anchor away as the loser. Returns a receipt."""
        if loser_id == SELF_ENTITY_ID:
            return {"merged": False, "reason": "refusing to merge the self anchor away"}
        if survivor_id == loser_id:
            return {"merged": False, "reason": "same node"}
        surv, lose = self.store.get_node(survivor_id), self.store.get_node(loser_id)
        if surv is None or lose is None:
            return {"merged": False, "reason": "missing node"}
        if surv.ntype != lose.ntype:
            return {"merged": False,
                    "reason": f"type mismatch {surv.ntype.value}/{lose.ntype.value}"}
        g = self.store.g
        out_edges: list[tuple[str, dict]] = []
        in_edges: list[tuple[str, dict]] = []
        if loser_id in g:
            out_edges = [(v, dict(d)) for _u, v, d in g.out_edges(loser_id, data=True)]
            in_edges = [(u, dict(d)) for u, _v, d in g.in_edges(loser_id, data=True)]
        # remove_node drops the loser + its incident edges from the graph AND flush-deletes
        # their persisted rows; the survivor's re-added edges are marked dirty by add_edge.
        self.store.remove_node(loser_id)
        moved = dropped = 0
        for v, d in out_edges:
            ok = self._readd_edge(survivor_id, survivor_id if v == loser_id else v, d)
            moved, dropped = (moved + 1, dropped) if ok else (moved, dropped + 1)
        for u, d in in_edges:
            ok = self._readd_edge(survivor_id if u == loser_id else u, survivor_id, d)
            moved, dropped = (moved + 1, dropped) if ok else (moved, dropped + 1)
        for alias in [lose.name] + list(lose.aliases or []):
            a = (alias or "").strip().lower()
            if a and a != surv.name.lower() and a not in surv.aliases:
                surv.aliases.append(a)
        surv.doc_frequency = max(surv.doc_frequency, lose.doc_frequency)
        surv.last_modified = now_iso()
        self.store.touch_node(survivor_id)
        # Drop the loser's in-memory vector so it is never a search candidate again; the
        # persisted row is deleted by remove_node's flush, so it never comes back on reload.
        kind = {NodeType.ENTITY: "entity", NodeType.TAG: "tag"}.get(lose.ntype)
        if kind is not None:
            self.store.vectors.remove(kind, loser_id)
        # repair the key maps: every key that resolved to the loser now hits the survivor.
        keymap = self._entity_keys if lose.ntype == NodeType.ENTITY else self._tag_keys
        for k, nid in list(keymap.items()):
            if nid == loser_id:
                keymap[k] = survivor_id
        return {"merged": True, "survivor": survivor_id, "loser": loser_id,
                "edges_moved": moved, "edges_dropped": dropped}

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
