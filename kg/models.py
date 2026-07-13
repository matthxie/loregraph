"""Graph data model — Episode / Mention / Entity (docs/ARCHITECTURE.md §2).

The store follows the HippoRAG / Graphiti episodic-semantic split:

  * **Episode** (immutable, append-only) — one per ingested entry (article paragraph,
    note, image). Holds the raw text/description (the embedding & retrieval surface),
    content hash, modality, source ref, and a bi-temporal stamp (`created_at` = event
    time, `ingested_at` = transaction time). Never edited, never re-embedded.
  * **Mention** (immutable, append-only) — one per *occurrence* of an entity inside an
    episode. Carries the exact surface form, a type guess, a char span, and a
    back-pointer to its episode. Embedded once; never edited. This is the atom that
    makes re-embedding unnecessary: new data appends mentions, it never mutates them.
  * **Entity** (lean canonical anchor) — ONE identity node per real-world thing. Holds
    only id / canonical name / type / aliases / doc-frequency — no raw text, no growing
    summary blob. Mentions point at it in a STAR (not a clique), so it stays a small,
    addressable identity rather than a high-degree hub that wrecks PPR.

Facts live on **edges with bi-temporal validity** (`valid_at` / `invalid_at` +
`belief`): a state change closes the old fact's window and opens a new one instead of
overwriting or re-embedding anything ("Becky lives in Toronto" → "…Berlin" is an
`invalid_at` on one edge plus a new edge). See docs/TEMPORAL.md.

Nodes live in GraphStore.nodes; NetworkX holds only topology + edge attributes for the
graph algorithms. Every fact edge carries `provenance` + `confidence` so traversal can
down-weight or drop low-trust relationships.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


# Special canonical node id for the first-person self anchor (personal-web mode)
SELF_ENTITY_ID = "entity_self"


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class NodeType(str, Enum):
    EPISODE = "episode"     # immutable ingested entry (the document/note/image unit)
    SOURCE = "source"       # immutable parent of chunked episodes (full original text;
                            # provenance only — never embedded, ranked, or BM25-indexed)
    MENTION = "mention"     # immutable per-episode occurrence of an entity
    ENTITY = "entity"       # lean canonical identity anchor
    TAG = "tag"             # canonical topical-tag vocabulary
    RELATION = "relation"   # canonical relationship-tag (predicate) vocabulary
    COMMUNITY = "community"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class EntityType(str, Enum):
    PERSON = "person"
    PLACE = "place"
    ORG = "org"
    CONCEPT = "concept"
    WORK = "work"
    EVENT = "event"
    DATE = "date"
    QUANTITY = "quantity"   # a stated amount/count/measurement (kg/extractors.py facts[])
    OTHER = "other"


class EntityCategory(str, Enum):
    """Broad visual identity used by graph clients, independent of semantic entity type."""
    PERSON = "person"
    PLACE = "place"
    THING = "thing"


def entity_category_for_type(t: "EntityType | None") -> "EntityCategory":
    """Collapse the fine-grained EntityType onto the three glyph categories a graph
    renderer draws — PERSON/PLACE keep their identity; org/concept/work/event/date/
    quantity/other all read as a generic THING."""
    if t == EntityType.PERSON:
        return EntityCategory.PERSON
    if t == EntityType.PLACE:
        return EntityCategory.PLACE
    return EntityCategory.THING


class Belief(str, Enum):
    """Transaction-time belief state of a fact edge (docs/TEMPORAL.md §3).

    `asserted` = we currently believe this fact held over its valid window;
    `retracted` = a better source contradicted the *recorded* belief — it was never
    actually valid (a correction, distinct from a fact whose valid window simply ended)."""
    ASSERTED = "asserted"
    RETRACTED = "retracted"


class EdgeType(str, Enum):
    MENTIONED_IN = "MENTIONED_IN"  # Mention  → Episode (provenance of an occurrence)
    RESOLVES_TO = "RESOLVES_TO"    # Mention  → Entity  (the star spoke to the anchor)
    TAGGED_AS = "TAGGED_AS"        # Episode  → Tag
    RELATED_TO = "RELATED_TO"      # Entity   → Entity  (a bi-temporal FACT edge)
    SIMILAR_TO = "SIMILAR_TO"      # immutable ↔ immutable (embedding synonymy)
    SHARED_TAG = "SHARED_TAG"      # Episode  ↔ Episode  (derived, overlap-weighted)
    SHARED_ENTITY = "SHARED_ENTITY"  # Episode ↔ Episode (share a resolved entity)
    IN_COMMUNITY = "IN_COMMUNITY"  # node → CommunityNode
    HYPERLINKS_TO = "HYPERLINKS_TO"  # optional deterministic enrichment
    PART_OF = "PART_OF"            # chunk Episode → parent Source (low traversal weight —
                                   # the parent must not become a sibling super-hub)
    NEXT = "NEXT"                  # chunk Episode → next sibling (sequence within a source)


class Provenance(str, Enum):
    EXTRACTED = "EXTRACTED"  # the LLM pulled it straight from the content
    INFERRED = "INFERRED"    # the LLM reasoned it (lower trust)
    SIMILAR = "SIMILAR"      # embedding cosine
    DERIVED = "DERIVED"      # deterministic (shared tags / kNN / hyperlinks / structure)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """Base node. Subtypes use a subset of the fields; everything round-trips through a
    single JSON payload so one SQLite table serialises every node type."""
    id: str
    ntype: NodeType
    name: str = ""
    created_at: str = ""        # event time (when the fact/content is about)
    last_modified: str = ""
    ingested_at: str = ""       # transaction time (when we recorded it)
    valid: bool = True
    superseded_by: str | None = None

    # ---- Episode ----
    modality: Modality | None = None
    raw_text: str | None = None          # the embedding / retrieval surface
    description: str | None = None        # image: the VLM one-liner
    content_hash: str | None = None
    source_ref: str | None = None         # url / file path / title
    tags: list[str] = field(default_factory=list)  # denormalised filter copy

    # ---- Mention ----
    episode_id: str | None = None         # back-pointer to the asserting episode
    char_span: list[int] | None = None    # [start, end] in the episode text (best-effort)

    # ---- Entity / Tag / Relation (canonical anchors) ----
    entity_type: EntityType | None = None
    entity_category: str | None = None     # broad glyph category for graph clients (PERSON/PLACE/THING)
    aliases: list[str] = field(default_factory=list)
    tag_description: str | None = None     # TagNode (L3)
    doc_frequency: int = 0                 # # episodes referencing it (IDF specificity)
    functional: bool = False               # RelationNode: single-valued (lives_in, employed_by)
    symmetric: bool = False                # RelationNode: orientation-free (works_with)

    # ---- Community ----
    members: list[str] = field(default_factory=list)
    summary: str | None = None

    # ---- Quantity (a typed amount/count/measurement; entity_type == QUANTITY) ----
    # Numeric home for stated amounts (docs: extraction-completeness fix) so SUM/COUNT
    # over a subject never needs to regex-parse a node name. Always minted fresh per
    # occurrence — NEVER routed through canonicalizer merge logic — so distinct amounts
    # ($250 vs $2,500) can never alias-merge, and repeated occurrences of the same
    # amount/date stay separate nodes/edges instead of collapsing into one.
    value: float | None = None
    unit: str | None = None

    def to_payload(self) -> str:
        d = asdict(self)
        d["ntype"] = self.ntype.value
        if self.modality is not None:
            d["modality"] = self.modality.value
        if self.entity_type is not None:
            d["entity_type"] = self.entity_type.value
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_payload(cls, payload: str) -> "Node":
        d = json.loads(payload)
        d["ntype"] = NodeType(d["ntype"])
        if d.get("modality") is not None:
            d["modality"] = Modality(d["modality"])
        if d.get("entity_type") is not None:
            d["entity_type"] = EntityType(d["entity_type"])
        # tolerate stores written before a field existed
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #
@dataclass
class Edge:
    src: str
    dst: str
    etype: EdgeType
    provenance: Provenance = Provenance.DERIVED
    confidence: float = 1.0
    weight: float = 1.0
    rel_tag: str | None = None            # canonical RelationNode id (RELATED_TO only)

    # ---- bi-temporal fact fields (RELATED_TO; docs/TEMPORAL.md) ----
    valid_at: str = ""                    # valid-time start ("" = unknown)
    invalid_at: str = ""                  # valid-time end   ("" = ∞ / currently true)
    belief: Belief = Belief.ASSERTED      # transaction-time belief state
    episode_id: str = ""                  # provenance: the episode that asserted this fact

    valid: bool = True                    # structural soft-invalidation
    created_at: str = ""                  # transaction time

    def key(self) -> tuple:
        """Per-edge identity in the MultiDiGraph. For a RELATED_TO fact the
        discriminator is (canonical relationship id, valid_at) so a *reopened* fact and
        its closed predecessor between the same pair are distinct parallel edges; the
        common case (one open fact per predicate) collapses naturally."""
        rel = self.rel_tag or ""
        disc = self.valid_at if self.etype == EdgeType.RELATED_TO else ""
        return (self.etype.value, rel, disc)


# --------------------------------------------------------------------------- #
# Convenience constructors
# --------------------------------------------------------------------------- #
def episode_node(node_id: str, *, modality: Modality, source_ref: str,
                 raw_text: str | None, content_hash: str, ts: str,
                 description: str | None = None, ingested_at: str = "") -> Node:
    return Node(
        id=node_id, ntype=NodeType.EPISODE, name=source_ref, modality=modality,
        source_ref=source_ref, raw_text=raw_text, description=description,
        content_hash=content_hash, created_at=ts, last_modified=ts,
        ingested_at=ingested_at or ts,
    )


def source_node(node_id: str, *, source_ref: str, raw_text: str | None,
                content_hash: str, ts: str, title: str = "",
                ingested_at: str = "") -> Node:
    """Parent of chunked episodes: keeps the full original text + provenance in ONE
    place. Deliberately NOT embedded / BM25-indexed / PPR-rankable (only NodeType.EPISODE
    is) so it can't outcompete its own chunks in retrieval."""
    return Node(id=node_id, ntype=NodeType.SOURCE, name=title or source_ref,
                modality=Modality.TEXT, source_ref=source_ref, raw_text=raw_text,
                content_hash=content_hash, created_at=ts, last_modified=ts,
                ingested_at=ingested_at or ts)


def mention_node(node_id: str, *, surface: str, etype: EntityType, episode_id: str,
                 ts: str, char_span: list[int] | None = None) -> Node:
    return Node(id=node_id, ntype=NodeType.MENTION, name=surface, entity_type=etype,
                episode_id=episode_id, char_span=char_span, created_at=ts, last_modified=ts)


def entity_node(node_id: str, *, name: str, etype: EntityType, ts: str) -> Node:
    return Node(id=node_id, ntype=NodeType.ENTITY, name=name,
                entity_type=etype, created_at=ts, last_modified=ts)


def tag_node(node_id: str, *, canonical: str, ts: str) -> Node:
    return Node(id=node_id, ntype=NodeType.TAG, name=canonical,
                created_at=ts, last_modified=ts)


def relation_tag_node(node_id: str, *, canonical: str, ts: str,
                      functional: bool = False, symmetric: bool = False) -> Node:
    """A canonical relationship-tag node (predicate vocabulary). Like TagNode it carries
    `aliases` + `doc_frequency`, plus per-predicate cardinality flags (docs/TEMPORAL.md
    §5): `functional` predicates (lives_in, employed_by) are single-valued so a new value
    supersedes the old; `symmetric` predicates (works_with) store one orientation only."""
    return Node(id=node_id, ntype=NodeType.RELATION, name=canonical,
                functional=functional, symmetric=symmetric,
                created_at=ts, last_modified=ts)


def quantity_node(node_id: str, *, display: str, value: float, unit: str, ts: str) -> Node:
    """A single stated amount/count/measurement occurrence. Deliberately minted directly
    (never through Canonicalizer.resolve_entity) so it never enters the L1/L2/L3 merge
    machinery — two occurrences of "$250" stay two distinct nodes/edges, and "$250" can
    never alias-merge with "$2,500"."""
    return Node(id=node_id, ntype=NodeType.ENTITY, name=display,
                entity_type=EntityType.QUANTITY, value=value, unit=unit,
                created_at=ts, last_modified=ts)


def community_node(node_id: str, *, members: list[str], summary: str, ts: str) -> Node:
    return Node(id=node_id, ntype=NodeType.COMMUNITY, name=node_id,
                members=members, summary=summary, created_at=ts, last_modified=ts)
