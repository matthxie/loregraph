"""Graph data model (docs/ARCHITECTURE.md §2).

Nodes are dataclasses kept in the GraphStore's `nodes` dict; NetworkX holds only
topology + edge attributes for the graph algorithms. Every node carries timestamps
and a `valid`/`superseded_by` soft-invalidation flag (rev 2) plus the `seen` debug
flag. Every edge carries `provenance` + `confidence` so traversal can down-weight or
drop low-trust relationships.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class NodeType(str, Enum):
    OBJECT = "object"
    ENTITY = "entity"
    TAG = "tag"
    RELATION = "relation"   # canonical relationship-tag node (predicate vocabulary)
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
    OTHER = "other"


class RelationType(str, Enum):
    """Legacy coarse relation vocabulary (rev 3).

    Relations are now open-vocabulary, LLM-generated, multi-label *relationship
    tags* that are consolidated over time (see RelationTagNode / NodeType.RELATION
    and Canonicalizer.resolve_relation). This enum is kept only as a coarse
    fallback / back-compat for the single `Edge.relation` slot; the live payload is
    the canonical relationship-tag id in `Edge.rel_tag` (one per parallel edge).
    """
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    CREATED_BY = "created_by"
    INSTANCE_OF = "instance_of"
    CAUSES = "causes"
    MENTIONS = "mentions"
    RELATED_TO = "related_to"  # catch-all

    @classmethod
    def coerce(cls, value: str | None) -> "RelationType":
        if not value:
            return cls.RELATED_TO
        try:
            return cls(value.strip().lower().replace(" ", "_"))
        except ValueError:
            return cls.RELATED_TO


class EdgeType(str, Enum):
    MENTIONS = "MENTIONS"          # ObjectNode  → EntityNode
    TAGGED_AS = "TAGGED_AS"        # ObjectNode  → TagNode
    RELATED_TO = "RELATED_TO"      # EntityNode  → EntityNode (directed; ONE per canonical
                                   #   relationship in `rel_tag` — parallel edges per pair)
    SIMILAR_TO = "SIMILAR_TO"      # any ↔ any   (embedding synonymy)
    SHARED_TAG = "SHARED_TAG"      # ObjectNode ↔ ObjectNode (derived, overlap-weighted)
    SHARED_ENTITY = "SHARED_ENTITY"
    IN_COMMUNITY = "IN_COMMUNITY"  # node → CommunityNode
    HYPERLINKS_TO = "HYPERLINKS_TO"  # optional deterministic enrichment


class Provenance(str, Enum):
    EXTRACTED = "EXTRACTED"  # the LLM pulled it straight from the content
    INFERRED = "INFERRED"    # the LLM reasoned it (lower trust)
    SIMILAR = "SIMILAR"      # embedding cosine
    DERIVED = "DERIVED"      # deterministic (shared tags / kNN / hyperlinks)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """Base node. Subtypes add fields; everything round-trips through a JSON payload."""
    id: str
    ntype: NodeType
    name: str = ""
    created_at: str = ""
    last_modified: str = ""
    valid: bool = True
    superseded_by: str | None = None
    seen: bool = False  # visited-set / debug flag (§5)

    # type-specific payload (kept loose so one table serialises every node type)
    modality: Modality | None = None
    raw_text: str | None = None          # ObjectNode: the embedding surface
    description: str | None = None        # ObjectNode(image): the VLM one-liner
    content_hash: str | None = None       # ObjectNode
    source_ref: str | None = None         # ObjectNode: url / file path / title
    tags: list[str] = field(default_factory=list)  # ObjectNode: denormalised filter copy

    entity_type: EntityType | None = None  # EntityNode
    aliases: list[str] = field(default_factory=list)  # TagNode
    tag_description: str | None = None     # TagNode (L3)
    doc_frequency: int = 0                 # EntityNode / TagNode (for IDF specificity)
    provenance_objs: list[str] = field(default_factory=list)  # back-pointers (§2)

    members: list[str] = field(default_factory=list)  # CommunityNode
    summary: str | None = None                        # CommunityNode

    def to_payload(self) -> str:
        d = asdict(self)
        # enums → str
        for k in ("ntype", "modality", "entity_type"):
            if d.get(k) is not None and isinstance(d[k], Enum):
                d[k] = d[k].value
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
        return cls(**d)


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
    relation: RelationType | None = None  # legacy coarse class (back-compat only)
    # canonical relationship-tag node id labelling a directed RELATED_TO edge (rev 4).
    # ONE per edge: A→B [is_friend_of] and A→B [works_with] are two PARALLEL edges in
    # the MultiDiGraph, each with its own provenance / confidence / timestamp — the
    # idiomatic KG-triple / property-graph shape (one relation per edge).
    rel_tag: str | None = None
    valid: bool = True
    created_at: str = ""

    def key(self) -> tuple:
        """Identity of an edge — the per-relation key in the MultiDiGraph. For a
        RELATED_TO edge the discriminator is the canonical relationship-tag id, so
        each relationship between a pair becomes its own parallel edge."""
        rel = self.rel_tag or (self.relation.value if self.relation else "")
        return (self.etype.value, rel)


# --------------------------------------------------------------------------- #
# Convenience constructors
# --------------------------------------------------------------------------- #
def object_node(node_id: str, *, modality: Modality, source_ref: str,
                raw_text: str | None, content_hash: str, ts: str,
                description: str | None = None) -> Node:
    return Node(
        id=node_id, ntype=NodeType.OBJECT, name=source_ref, modality=modality,
        source_ref=source_ref, raw_text=raw_text, description=description,
        content_hash=content_hash, created_at=ts, last_modified=ts,
    )


def entity_node(node_id: str, *, name: str, etype: EntityType, ts: str) -> Node:
    return Node(id=node_id, ntype=NodeType.ENTITY, name=name,
                entity_type=etype, created_at=ts, last_modified=ts)


def tag_node(node_id: str, *, canonical: str, ts: str) -> Node:
    return Node(id=node_id, ntype=NodeType.TAG, name=canonical,
                created_at=ts, last_modified=ts)


def relation_tag_node(node_id: str, *, canonical: str, ts: str) -> Node:
    """A canonical relationship-tag node (predicate vocabulary). Parallel to
    TagNode — it carries `aliases` and `doc_frequency` so the relation vocabulary
    is consolidated and IDF-weighted exactly like the topical-tag vocabulary."""
    return Node(id=node_id, ntype=NodeType.RELATION, name=canonical,
                created_at=ts, last_modified=ts)


def community_node(node_id: str, *, members: list[str], summary: str, ts: str) -> Node:
    return Node(id=node_id, ntype=NodeType.COMMUNITY, name=node_id,
                members=members, summary=summary, created_at=ts, last_modified=ts)
