"""Test suite for the kg MVP.

Runs fully offline/deterministic (hashing embedder + heuristic extractor) so it
needs no API key, no model download, and no network. Run: python -m pytest -q
"""
from __future__ import annotations

import json
import os
import tempfile
import types

import numpy as np
import pytest

from kg import Config, KnowledgeGraph
from kg.canonicalize import (Canonicalizer, char_entropy, normalize_date,
                             normalize_key, normalize_relation, relation_merge_vetoed)
from kg.corpus import CorpusItem, load_articles, load_images, load_mixed
from kg.embedders import HashingEmbedder, get_embedder
from kg.extractors import (Extraction, ExtractedEntity, HeuristicExtractor,
                           get_extractor)
from kg.ingest import Ingestor
from kg.models import (Edge, EdgeType, EntityType, Modality, Node, NodeType,
                       Provenance, RelationType, object_node)
from kg.store import GraphStore
from kg.vectors import VectorIndex


def cfg() -> Config:
    c = Config.default()
    c.embedder = "hashing"
    c.extractor = "heuristic"
    return c


def tmp_store() -> str:
    return os.path.join(tempfile.mkdtemp(), "kg.db")


def sample_items():
    return [
        CorpusItem(id="a", modality="text", source_ref="u/a", title="Alan Turing",
                   text="Alan Turing was a British mathematician and computer scientist. "
                        "Turing worked at Bletchley Park on cryptography during the war. "
                        "He is considered the father of computer science and artificial intelligence."),
        CorpusItem(id="b", modality="text", source_ref="u/b", title="Bletchley Park",
                   text="Bletchley Park was the central site for British codebreakers during "
                        "World War II. Alan Turing and many mathematicians worked there on "
                        "cryptography and the Enigma machine."),
        CorpusItem(id="c", modality="text", source_ref="u/c", title="Photosynthesis",
                   text="Photosynthesis is the process by which plants convert light energy "
                        "into chemical energy. Chlorophyll absorbs sunlight in the leaves."),
        CorpusItem(id="d", modality="image", source_ref="img/d.jpg",
                   image_path="img/d.jpg", label_hint="dog, frisbee, person"),
    ]


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_node_payload_roundtrip():
    n = object_node("obj_x", modality=Modality.TEXT, source_ref="u",
                    raw_text="hello", content_hash="h", ts="t")
    n.tags = ["a", "b"]
    back = Node.from_payload(n.to_payload())
    assert back.id == "obj_x"
    assert back.ntype == NodeType.OBJECT
    assert back.modality == Modality.TEXT
    assert back.tags == ["a", "b"]


def test_relation_coerce():
    assert RelationType.coerce("located_in") == RelationType.LOCATED_IN
    assert RelationType.coerce("Located In") == RelationType.LOCATED_IN
    assert RelationType.coerce("nonsense") == RelationType.RELATED_TO
    assert RelationType.coerce(None) == RelationType.RELATED_TO


def test_edge_key():
    e = Edge("a", "b", EdgeType.RELATED_TO, relation=RelationType.CAUSES)
    assert e.key() == ("RELATED_TO", "causes")
    # rev 4: a canonical relation-tag id is the per-edge discriminator (precedence)
    e2 = Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0007")
    assert e2.key() == ("RELATED_TO", "rel_0007")


# --------------------------------------------------------------------------- #
# vectors
# --------------------------------------------------------------------------- #
def test_vector_normalize_and_search():
    idx = VectorIndex(dim=8)
    idx.add("object", "o1", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    idx.add("object", "o2", np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    v = idx.get("object", "o1")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    hits = idx.search("object", np.array([1, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32), k=2)
    assert hits[0][0] == "o1"
    assert hits[0][1] > hits[1][1]


def test_vector_update_in_place():
    idx = VectorIndex(dim=4)
    idx.add("tag", "t", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.add("tag", "t", np.array([0, 1, 0, 0], dtype=np.float32))  # update
    assert len(idx.ids("tag")) == 1
    hit = idx.search("tag", np.array([0, 1, 0, 0], dtype=np.float32), k=1)
    assert hit[0][1] > 0.99


def test_vector_search_floor_and_exclude():
    idx = VectorIndex(dim=4)
    idx.add("object", "o1", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.add("object", "o2", np.array([1, 0, 0, 0], dtype=np.float32))
    hits = idx.search("object", np.array([1, 0, 0, 0], dtype=np.float32), k=5,
                      floor=0.9, exclude={"o1"})
    assert [h[0] for h in hits] == ["o2"]


# --------------------------------------------------------------------------- #
# canonicalize
# --------------------------------------------------------------------------- #
def test_normalize_key():
    assert normalize_key("Natural-Language Processing") == "natural language processing"
    assert normalize_key("Networks") == "network"
    assert normalize_key("studies") == "study"
    # the >4-char guard protects short words from being mangled (e.g. "lens"→"len")
    assert normalize_key("Cars") == "cars"
    assert normalize_key("  ") == ""


def test_char_entropy_short_vs_long():
    assert char_entropy("ai") < char_entropy("artificial intelligence")


def test_l1_dedup_and_entropy_guard():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    t1 = canon.resolve_entity("Machine Learning", EntityType.CONCEPT)
    t2 = canon.resolve_entity("machine-learning", EntityType.CONCEPT)  # same normalized key
    assert t1 == t2
    # short low-entropy surfaces must NOT be embedding-merged: distinct nodes
    a = canon.resolve_entity("AI", EntityType.CONCEPT)
    b = canon.resolve_entity("US", EntityType.CONCEPT)
    assert a != b


def test_normalize_relation():
    # spaces, hyphens and case all collapse to one underscore key
    assert normalize_relation("is friends with") == "is_friends_with"
    assert normalize_relation("is-friends-with") == "is_friends_with"
    assert normalize_relation("Is  Friends  With") == "is_friends_with"
    # punctuation stripped, underscores preserved/collapsed
    assert normalize_relation("works_with!") == "works_with"
    # unlike noun keys, predicate morphology is NOT singularized (direction matters)
    assert normalize_relation("manages") == "manages"
    assert normalize_relation("managed_by") == "managed_by"


def test_resolve_relation_consolidation():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    # L1: surface variants of the same predicate consolidate to one relation node
    r1 = canon.resolve_relation("works with")
    r2 = canon.resolve_relation("works-with")
    r3 = canon.resolve_relation("Works With")
    assert r1 == r2 == r3
    assert store.get_node(r1).ntype == NodeType.RELATION
    # a genuinely different predicate gets its own node
    other = canon.resolve_relation("founded")
    assert other != r1
    # the relation vocabulary is consolidated, not duplicated
    rel_nodes = store.nodes_of_type(NodeType.RELATION)
    assert len(rel_nodes) == 2


def test_relation_synonym_merges_but_antonym_and_inverse_dont():
    """rev 3: inflectional / function-word variants of a predicate consolidate via
    the content key, while antonyms and passive-inverses stay distinct."""
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    a = canon.resolve_relation("is_friend_of")
    b = canon.resolve_relation("is friends with")   # function-word + plural variant
    assert a == b, "synonymous inflections should consolidate to one relation node"
    node = store.get_node(a)
    assert node.name == "is_friend_of"               # readable canonical name kept
    assert "is_friends_with" in node.aliases         # the variant is an alias
    # antonyms (different content word) must NOT merge
    assert canon.resolve_relation("is_enemy_of") != a
    # passive inverse ("by" is preserved as a direction marker) must NOT merge
    assert canon.resolve_relation("manages") != canon.resolve_relation("managed_by")


# --------------------------------------------------------------------------- #
# date canonicalization (EntityType.DATE)
# --------------------------------------------------------------------------- #
def test_normalize_date_variants_and_granularity():
    # surface variants of the SAME instant collapse to one structured key
    for s in ("July 18, 1896", "18 July 1896", "1896-07-18", "1896/07/18", "18th July 1896"):
        assert normalize_date(s) == "1896-07-18", s
    # granularity is preserved — coarser dates get distinct keys, never colliding
    assert normalize_date("1896") == "1896"
    assert normalize_date("July 1896") == "1896-07"
    assert normalize_date("July 4th") == "--07-04"      # month-day, no year
    assert normalize_date("December 7, 1941") == "1941-12-07"
    # a stray numeral NOT adjacent to the month must not be fabricated into a day, so it
    # still dedups with the bare month+year form
    assert normalize_date("page 12 of May 1896") == "1896-05"
    assert normalize_date("on 2 separate days in May 1896") == "1896-05"
    # not a concrete date → None (falls back to generic string handling)
    for s in ("the 1890s", "early 20th century", "summer", "nonsense", ""):
        assert normalize_date(s) is None, s


def test_date_entity_consolidation_and_no_fuzzy_merge():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    d1 = canon.resolve_entity("July 18, 1896", EntityType.DATE)
    d2 = canon.resolve_entity("18 July 1896", EntityType.DATE)
    d3 = canon.resolve_entity("1896-07-18", EntityType.DATE)
    assert d1 == d2 == d3, "date surface variants must consolidate to one node"
    assert store.get_node(d1).entity_type == EntityType.DATE
    # different granularities stay distinct (no specificity loss)
    yr = canon.resolve_entity("1896", EntityType.DATE)
    md = canon.resolve_entity("July 4th", EntityType.DATE)
    assert len({d1, yr, md}) == 3
    # adjacent years sit close in embedding space but must NOT merge (structured key only)
    assert canon.resolve_entity("1897", EntityType.DATE) != yr
    # an unparseable temporal phrase still becomes a (generic) entity, not dropped
    decade = canon.resolve_entity("the 1890s", EntityType.DATE)
    assert store.get_node(decade).ntype == NodeType.ENTITY


def test_date_entity_key_survives_save_load():
    path = tmp_store()
    store = GraphStore.open(path, cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    d1 = canon.resolve_entity("July 18, 1896", EntityType.DATE)
    store.save()
    store2 = GraphStore.open(path, cfg())
    canon2 = Canonicalizer(store2, get_embedder(cfg()), cfg())  # _reindex re-registers date keys
    # a different surface for the same instant resolves to the persisted node, no duplicate
    assert canon2.resolve_entity("1896-07-18", EntityType.DATE) == d1


# --------------------------------------------------------------------------- #
# L3 selective tie-breaker (offline: veto + fake client)
# --------------------------------------------------------------------------- #
def test_relation_merge_vetoed():
    # passive/inverse asymmetry: exactly one side carries the "_by" marker
    assert relation_merge_vetoed("manages", "managed_by")
    assert relation_merge_vetoed("founded", "founded_by")
    assert relation_merge_vetoed("employs", "employed_by")
    # known opposites compared on content lemmas
    assert relation_merge_vetoed("is_friend_of", "is_enemy_of")
    assert relation_merge_vetoed("parent_of", "child_of")
    assert relation_merge_vetoed("predecessor_of", "successor_of")
    # genuine synonyms are NOT vetoed (they remain eligible to merge)
    assert not relation_merge_vetoed("works_with", "collaborates_with")
    assert not relation_merge_vetoed("founded", "established")
    assert not relation_merge_vetoed("works_with", "works_with")


class _FakeL3Client:
    """Stand-in anthropic client: messages.create returns a fixed verdict so the L3
    plumbing can be exercised offline with no API key."""
    def __init__(self, verdict: str):
        self._verdict = verdict
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        text = json.dumps({"verdict": self._verdict, "reason": "test"})
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)])


def _l3_canon(verdict: str):
    c = cfg()
    c.l3_enabled = True
    store = GraphStore(c)
    canon = Canonicalizer(store, get_embedder(c), c)
    canon._l3_client = _FakeL3Client(verdict)   # inject; bypass lazy init / API key
    return store, canon


def test_l3_adjudicate_merges_on_existing_id():
    store, canon = _l3_canon("placeholder")
    rid = canon.resolve_relation("works_with")
    canon._l3_client = _FakeL3Client(rid)        # vote to merge into the existing node
    out = canon._l3_adjudicate("relation", "collaborates_with", [(rid, 0.92)])
    assert out == rid
    assert canon.l3_log[-1]["verdict"] == rid


def test_l3_adjudicate_new_is_under_merge_default():
    store, canon = _l3_canon("NEW")
    rid = canon.resolve_relation("works_with")
    # a NEW verdict (or any id not in the candidate set) → mint new, i.e. return None
    assert canon._l3_adjudicate("relation", "reports_to", [(rid, 0.92)]) is None


def test_l3_relation_veto_blocks_inverse_before_llm():
    store, canon = _l3_canon("placeholder")
    rid = canon.resolve_relation("manages")
    fake = _FakeL3Client(rid)
    canon._l3_client = fake
    # managed_by is the passive inverse of manages → vetoed → the LLM is never consulted
    assert canon._l3_relation("managed_by", [(rid, 0.92)]) is None
    assert fake.calls == []


def test_l3_disabled_by_default_leaves_resolution_deterministic():
    # default cfg has l3_enabled False → the adjudicator is a no-op even with candidates
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    rid = canon.resolve_relation("works_with")
    assert canon._l3_adjudicate("relation", "collaborates_with", [(rid, 0.92)]) is None
    assert canon.l3_log == []


# --------------------------------------------------------------------------- #
# extract-dump + eval-canon (offline)
# --------------------------------------------------------------------------- #
def test_extract_dump_runs_and_summarizes():
    from kg.extract_dump import extract_corpus, summarize
    ext = HeuristicExtractor(cfg())
    records, errors = extract_corpus(ext, sample_items(), cfg())
    assert len(records) == 4 and not errors
    s = summarize(records, "heuristic")
    assert s["items"] == 4 and s["unique_concepts"] > 0 and s["entity_types"]
    img = next(r for r in records if r["modality"] == "image")
    assert img["entities"] and img["description"]


def test_extract_text_sectioned_unions_sections():
    from kg.extractors import extract_text_sectioned
    ext = HeuristicExtractor(cfg())
    long = "Alan Turing worked at Bletchley Park on cryptography and the Enigma. " * 200
    merged = extract_text_sectioned(ext, long, "Turing", long_doc_chars=500, max_sections=4)
    assert merged.entities  # produced a unioned extraction across sections


def test_eval_canon_gate_passes_offline():
    from kg.eval_canon import run_gate
    rep = run_gate(cfg())   # heuristic + hashing, L3 disabled
    assert rep["gate_pass"] is True
    assert rep["relation"]["wrong_antonym_inverse_merges"] == 0
    assert rep["entity"]["false_merges"] == 0
    # L1 content-key still consolidates the inflectional synonym pair
    assert rep["relation"]["counts"]["tp"] >= 1


def test_idf_weight_monotonic():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    # add some object nodes so the corpus size is > 0
    for i in range(10):
        store.add_node(object_node(f"obj_{i}", modality=Modality.TEXT, source_ref="u",
                                   raw_text="x", content_hash=str(i), ts="t"))
    common = canon.resolve_entity("common", EntityType.CONCEPT)
    rare = canon.resolve_entity("rare", EntityType.CONCEPT)
    store.get_node(common).doc_frequency = 8
    store.get_node(rare).doc_frequency = 1
    assert canon.idf_weight(rare) > canon.idf_weight(common)


# --------------------------------------------------------------------------- #
# embedders / extractors / factories
# --------------------------------------------------------------------------- #
def test_hashing_embedder_deterministic_and_unit_norm():
    e = HashingEmbedder(dim=64)
    a = e.embed(["knowledge graph"])
    b = e.embed(["knowledge graph"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5


def test_embedder_factory_fallback():
    c = cfg()
    c.embedder = "hashing"
    assert isinstance(get_embedder(c), HashingEmbedder)


def test_extractor_auto_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = Config.default()  # auto
    assert isinstance(get_extractor(c), HeuristicExtractor)


def test_heuristic_extract_text():
    ext = HeuristicExtractor(cfg())
    r = ext.extract_text("Alan Turing worked at Bletchley Park on cryptography.",
                         "Alan Turing")
    names = {e.name.lower() for e in r.entities}
    assert any("turing" in n for n in names)
    # topical content words come back as CONCEPT entities (no separate tag list)
    assert any(e.type == EntityType.CONCEPT for e in r.entities)


def test_heuristic_extract_image_uses_labels():
    ext = HeuristicExtractor(cfg())
    r = ext.extract_image("x.jpg", "dog, frisbee")
    names = {e.name for e in r.entities}
    assert "dog" in names and "frisbee" in names
    assert r.description and "dog" in r.description


def test_parse_tool_payload_tolerates_malformed_shapes():
    """Robustness: the model occasionally emits an entity as a bare string or a relation in a
    non-object shape. The parser must recover (bare string → untyped entity) or skip, never
    crash the whole extraction (the old AttributeError on e.get gave the item an empty graph)."""
    from kg.extractors import _parse_tool_payload
    payload = {
        "entities": ["Jepara", {"name": "Bali", "type": "place"}, 42, {"type": "org"}],
        "relations": ["not-an-object", {"source": "Jepara", "target": "Bali", "labels": ["near"]}],
    }
    ext = _parse_tool_payload(payload)
    names = {e.name for e in ext.entities}
    assert "Jepara" in names and "Bali" in names     # bare string recovered, dict parsed
    assert 42 not in names                            # non-str/non-dict skipped
    assert len(ext.relations) == 1 and ext.relations[0].labels == ["near"]


# --------------------------------------------------------------------------- #
# store persistence
# --------------------------------------------------------------------------- #
def test_store_bidirectional_neighbors():
    store = GraphStore(cfg())
    store.add_node(object_node("a", modality=Modality.TEXT, source_ref="u",
                               raw_text="x", content_hash="1", ts="t"))
    store.add_node(object_node("b", modality=Modality.TEXT, source_ref="u",
                               raw_text="y", content_hash="2", ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.SHARED_TAG, Provenance.DERIVED, 1.0, 1.0))
    assert any(nbr == "b" for nbr, _ in store.neighbors("a"))
    assert any(nbr == "a" for nbr, _ in store.neighbors("b"))  # both directions


def test_store_save_load_roundtrip():
    path = tmp_store()
    store = GraphStore.open(path, cfg())
    store.add_node(object_node("a", modality=Modality.TEXT, source_ref="u",
                               raw_text="x", content_hash="1", ts="t"))
    store.add_node(object_node("b", modality=Modality.TEXT, source_ref="u",
                               raw_text="y", content_hash="2", ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.SIMILAR_TO, Provenance.SIMILAR, 0.9, 0.9))
    store.add_edge(Edge("a", "b", EdgeType.SHARED_TAG, Provenance.DERIVED, 0.5, 2.0))
    store.vectors.add("object", "a", np.ones(cfg().embed_dim, dtype=np.float32))
    store.hash_cache["1"] = "a"
    store.save()

    s2 = GraphStore.open(path, cfg())
    assert s2.has_node("a") and s2.has_node("b")
    # both parallel typed edges survive
    etypes = {d["etype"] for _n, d in s2.neighbors("a")}
    assert {"SIMILAR_TO", "SHARED_TAG"} <= etypes
    assert s2.hash_cache.get("1") == "a"
    assert s2.vectors.get("object", "a") is not None


def test_directed_edges_distinct_but_neighbors_bidirectional():
    """rev 3: storage is directed (a→b ≠ b→a), but neighbors() still walks both
    directions by default so retrieval/derivation stay bidirectional."""
    store = GraphStore(cfg())
    for nid in ("a", "b"):
        store.add_node(object_node(nid, modality=Modality.TEXT, source_ref="u",
                                   raw_text="x", content_hash=nid, ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.RELATED_TO, Provenance.EXTRACTED, 1.0, 1.0))
    # a→b and b→a are independent directed edges
    assert store.g.has_edge("a", "b")
    assert not store.g.has_edge("b", "a")
    # default neighbours are bidirectional
    assert any(n == "b" for n, _ in store.neighbors("a"))
    assert any(n == "a" for n, _ in store.neighbors("b"))
    # but direction can be honoured explicitly
    assert [n for n, _ in store.neighbors("a", direction="out")] == ["b"]
    assert [n for n, _ in store.neighbors("a", direction="in")] == []
    assert [n for n, _ in store.neighbors("b", direction="in")] == ["a"]


def test_symmetric_edges_pinned_to_one_canonical_orientation():
    """SIMILAR_TO / SHARED_* are symmetric: a→b and b→a must collapse to a single
    canonical edge so the undirected projection can't double-count their weight."""
    store = GraphStore(cfg())
    for nid in ("a", "b"):
        store.add_node(object_node(nid, modality=Modality.TEXT, source_ref="u",
                                   raw_text="x", content_hash=nid, ts="t"))
    store.add_edge(Edge("b", "a", EdgeType.SIMILAR_TO, Provenance.SIMILAR, 0.9, 0.9))
    store.add_edge(Edge("a", "b", EdgeType.SIMILAR_TO, Provenance.SIMILAR, 0.9, 0.9))
    # exactly one SIMILAR_TO edge exists, in canonical (min,max) orientation
    assert store.g.number_of_edges() == 1
    assert store.g.has_edge("a", "b") and not store.g.has_edge("b", "a")
    # RELATED_TO, by contrast, is directional and keeps both directions distinct
    store.add_edge(Edge("b", "a", EdgeType.RELATED_TO, Provenance.EXTRACTED, 1.0, 1.0))
    assert store.g.has_edge("b", "a")


def test_parallel_relation_edges_one_per_tag():
    """rev 4: each relationship tag is its own directed edge between the pair."""
    store = GraphStore(cfg())
    for nid in ("a", "b"):
        store.add_node(object_node(nid, modality=Modality.TEXT, source_ref="u",
                                   raw_text="x", content_hash=nid, ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0000"))
    store.add_edge(Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0001"))
    # two PARALLEL directed edges between the same pair, one per relation
    assert store.g.number_of_edges("a", "b") == 2
    assert set(store.edge_rel_tags("a", "b")) == {"rel_0000", "rel_0001"}
    # re-asserting an existing relation does not add a duplicate edge
    store.add_edge(Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0000"))
    assert store.g.number_of_edges("a", "b") == 2
    # the reverse direction carries nothing — direction is real
    assert store.edge_rel_tags("b", "a") == []


def test_supersede_marks_invalid():
    store = GraphStore(cfg())
    store.add_node(object_node("a", modality=Modality.TEXT, source_ref="u",
                               raw_text="x", content_hash="1", ts="t"))
    store.add_node(object_node("a_v1", modality=Modality.TEXT, source_ref="u",
                               raw_text="x2", content_hash="2", ts="t"))
    store.add_edge(Edge("a", "a_v1", EdgeType.SIMILAR_TO))
    store.supersede_node("a", "a_v1")
    assert store.get_node("a").valid is False
    assert store.get_node("a").superseded_by == "a_v1"


# --------------------------------------------------------------------------- #
# end-to-end ingest + retrieval + eval
# --------------------------------------------------------------------------- #
def test_ingest_builds_graph():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    rep = g.ingest(sample_items())
    assert rep.ingested == 4
    s = g.stats()
    assert s["by_node_type"]["object"] == 4
    assert s["by_node_type"]["entity"] > 0
    # tags were retired into the entity vocabulary — no separate TAG nodes are ever minted
    assert s["by_node_type"].get("tag", 0) == 0
    # Turing & Bletchley should share an entity → a derived object-object edge exists
    assert s["by_edge_type"].get("SHARED_ENTITY", 0) + s["by_edge_type"].get("SHARED_TAG", 0) > 0


def test_store_is_directed():
    import networkx as nx
    g = KnowledgeGraph.open(tmp_store(), cfg())
    assert isinstance(g.store.g, nx.MultiDiGraph)


def test_concepts_are_entities_and_no_tag_nodes():
    """Tags were retired into one vocabulary: a theme the extractor surfaces is a CONCEPT
    entity, not a separate tag node. So there are NO tag nodes at all (duplication is
    impossible by construction), concepts are reachable via MENTIONS, and node.tags mirrors
    the object's concept entities for cheap display."""
    c = cfg()
    store = GraphStore(c)
    embedder = get_embedder(c)
    canon = Canonicalizer(store, embedder, c)
    ext = Extraction(entities=[
        ExtractedEntity(name="Ukraine", type=EntityType.PLACE),
        ExtractedEntity(name="energy policy", type=EntityType.CONCEPT),
        ExtractedEntity(name="geopolitics", type=EntityType.CONCEPT),
    ])

    class _FixedExtractor:
        name = "fixed"

        def extract_text(self, text, title=""):
            return ext

        def extract_image(self, image_path, label_hint=None):
            return Extraction()

    ingestor = Ingestor(store, _FixedExtractor(), embedder, canon, c)
    item = CorpusItem(id="x", modality="text", source_ref="u", title="",
                      text="A paragraph about Ukraine, energy policy and geopolitics.")
    rep = ingestor.ingest([item])
    assert rep.ingested == 1
    # no separate tag vocabulary exists at all
    assert store.nodes_of_type(NodeType.TAG) == []
    # the concepts AND the place are all entities, reachable via MENTIONS
    ent_names = {n.name for n in store.nodes_of_type(NodeType.ENTITY)}
    assert {"Ukraine", "energy policy", "geopolitics"} <= ent_names
    # node.tags mirrors ONLY the object's CONCEPT entities (the place is not a "tag")
    assert set(store.get_node("obj_x").tags) == {"energy policy", "geopolitics"}


def test_date_entity_ignores_embedding_collision():
    """Load-bearing guard: even if the embedder reports two different dates as IDENTICAL,
    DATE entities must NOT merge — they consolidate on the structured key, never the fuzzy
    embedding gate. (The hashing embedder alone keeps adjacent years apart, so this uses a
    constant embedder to actually force the collision the date path must withstand.)"""
    c = cfg()

    class _ConstEmbedder:
        name = "const"

        def embed(self, texts):
            return [np.ones(c.embed_dim, dtype=np.float32) for _ in texts]

    store = GraphStore(c)
    canon = Canonicalizer(store, _ConstEmbedder(), c)
    assert (canon.resolve_entity("1896", EntityType.DATE)
            != canon.resolve_entity("1897", EntityType.DATE))
    # control (fresh store): the SAME const embedder DOES fuzzy-merge two non-date concepts,
    # proving the collision is real and the date path is what prevents it
    store2 = GraphStore(c)
    canon2 = Canonicalizer(store2, _ConstEmbedder(), c)
    assert (canon2.resolve_entity("biography", EntityType.CONCEPT)
            == canon2.resolve_entity("chemistry", EntityType.CONCEPT))


def test_load_mixed_is_title_free_and_text_only(tmp_path):
    """The mixed stream is title-free body text and image-free: load_mixed must never
    inject the manifest title (no leakage into the prompt/node name) and must skip any
    non-text row."""
    (tmp_path / "aaa.txt").write_text("Some paragraph body about a topic.", encoding="utf-8")
    rows = [
        {"id": "aaa", "file": "aaa.txt", "modality": "text", "orig_id": "wiki_009",
         "title": "Some Article Title", "url": "https://en.wikipedia.org/wiki/Some",
         "label": None, "para_index": 0, "para_count": 1,
         "created_at": "2024-01-01T00:00:00+00:00"},
        {"id": "bbb", "file": "bbb.jpg", "modality": "image", "orig_id": "img_001",
         "title": None, "url": None, "label": "dog", "para_index": None,
         "para_count": None, "created_at": "2024-01-02T00:00:00+00:00"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items = load_mixed(manifest=str(manifest))
    assert len(items) == 1                       # the image row is skipped (text-only)
    it = items[0]
    assert it.modality == "text"
    assert it.title == ""                        # title is NEVER injected (no leakage)
    assert it.id == "wiki_009#p000"
    assert it.created_at == "2024-01-01T00:00:00+00:00"
    assert "Some paragraph body" in it.text


def test_ingest_builds_relation_tags_and_directed_edges():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    s = g.stats()
    # open-vocab relationship labels are consolidated into first-class RELATION nodes
    assert s["by_node_type"].get("relation", 0) > 0
    # at least one directed RELATED_TO edge is labelled with a single relation-tag
    found = False
    for _u, _v, d in g.store.all_edges():
        if d["etype"] == EdgeType.RELATED_TO.value and d.get("rel_tag"):
            found = True
            assert g.store.get_node(d["rel_tag"]).ntype == NodeType.RELATION
    assert found, "expected a labelled directed RELATED_TO edge"


def test_parallel_relation_edges_survive_save_load():
    path = tmp_store()
    store = GraphStore.open(path, cfg())
    for nid in ("a", "b"):
        store.add_node(object_node(nid, modality=Modality.TEXT, source_ref="u",
                                   raw_text="x", content_hash=nid, ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0000"))
    store.add_edge(Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0001"))
    store.save()
    s2 = GraphStore.open(path, cfg())
    assert s2.g.number_of_edges("a", "b") == 2  # both parallel edges survive
    assert set(s2.edge_rel_tags("a", "b")) == {"rel_0000", "rel_0001"}
    assert not s2.g.has_edge("b", "a")  # direction preserved across persistence


def test_rev3_set_on_edge_migrates_to_parallel_edges():
    """A pre-rev-4 store with a rel_tags SET on one RELATED_TO edge loads as N
    parallel directed edges (one per relation)."""
    import json as _json
    import sqlite3
    path = tmp_store()
    store = GraphStore.open(path, cfg())
    for nid in ("a", "b"):
        store.add_node(object_node(nid, modality=Modality.TEXT, source_ref="u",
                                   raw_text="x", content_hash=nid, ts="t"))
    store.save()
    # inject a legacy rev-3 row: a SET of rel ids stored on ONE RELATED_TO edge
    con = sqlite3.connect(path)
    con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("a", "b", "RELATED_TO", "", "EXTRACTED", 0.9, 0.9, 1, "",
                 _json.dumps(["rel_0000", "rel_0001", "rel_0002"])))
    con.commit()
    con.close()
    s2 = GraphStore.open(path, cfg())
    assert s2.g.number_of_edges("a", "b") == 3      # exploded into parallel edges
    assert set(s2.edge_rel_tags("a", "b")) == {"rel_0000", "rel_0001", "rel_0002"}


def test_relation_df_idempotent_on_reingest():
    """Re-stating the same connection must not double-count its relation frequency."""
    path = tmp_store()
    g = KnowledgeGraph.open(path, cfg())
    v1 = CorpusItem(id="p", modality="text", source_ref="u", title="People",
                    text="Alice runs daily. Alice writes books. Alice paints often. "
                         "Alice meets Bob and Carol at work.")
    g.ingest([v1])
    df_before = {r.id: r.doc_frequency for r in g.store.nodes_of_type(NodeType.RELATION)}
    assert df_before, "expected at least one consolidated relationship tag"
    v2 = CorpusItem(id="p", modality="text", source_ref="u", title="People",
                    text="Alice swims weekly. Alice cooks meals. Alice travels far. "
                         "Alice meets Bob and Carol again.")
    g.ingest([v2])  # supersedes v1; same entities + same co-occurrence relation
    for rid, df in df_before.items():
        assert g.store.get_node(rid).doc_frequency <= df


def test_ingest_cache_skips_on_rerun():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    rep2 = g.ingest(sample_items())  # identical content
    assert rep2.ingested == 0
    assert rep2.skipped == 4


def test_ingest_supersedes_on_change():
    path = tmp_store()
    g = KnowledgeGraph.open(path, cfg())
    g.ingest([sample_items()[0]])
    changed = CorpusItem(id="a", modality="text", source_ref="u/a", title="Alan Turing",
                         text="Completely different content about marine biology and coral reefs.")
    rep = g.ingest([changed])
    assert rep.superseded == 1
    assert g.store.get_node("obj_a").valid is False


def test_retrievers_run_and_find_relevant():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    for mode in ("ppr", "bfs", "vector"):
        res = g.query("cryptography codebreaking at Bletchley", mode=mode, k=3)
        assert res.object_ids, f"{mode} returned nothing"
        # the relevant articles are a (Turing) and/or b (Bletchley)
        assert {"obj_a", "obj_b"} & set(res.object_ids), f"{mode} missed the target"


def test_communities_and_global_route():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    n = g.build_communities()
    assert n >= 1
    res = g.query("what are the main themes", mode="auto")
    assert isinstance(res, list)  # routed to community path
    assert res and "summary" in res[0]


def test_empty_query_and_empty_graph_do_not_crash():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    # query on an empty graph
    assert g.query("anything", mode="ppr").object_ids == []
    assert g.query("anything", mode="bfs").object_ids == []
    assert g.query("anything", mode="vector").object_ids == []


def test_blank_query_on_populated_graph_no_crash():
    """Regression: an empty/whitespace query (zero embedding under the hashing
    embedder) must not crash PPR with a ZeroDivisionError from nx.pagerank."""
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    g.build_communities()
    for mode in ("ppr", "bfs", "vector"):
        assert g.query("   ", mode=mode).object_ids == []
        assert g.query("", mode=mode).object_ids == []
    assert g.query("  ", mode="community") == []


def test_extraction_failures_are_surfaced_not_swallowed():
    """Regression: a failing extractor must be reported, not silently swallowed."""
    g = KnowledgeGraph.open(tmp_store(), cfg())

    class BoomExtractor:
        name = "boom"
        def extract_text(self, text, title=""):
            raise RuntimeError("bad api key")
        def extract_image(self, path, hint=None):
            raise RuntimeError("bad api key")

    g.extractor = BoomExtractor()
    rep = g.ingest(sample_items())
    assert rep.extraction_failures == 4
    assert rep.notes and "extraction failed" in rep.notes[0]


def test_supersede_retracts_doc_frequency():
    """Regression: re-ingesting changed content must not double-count doc_frequency."""
    g = KnowledgeGraph.open(tmp_store(), cfg())
    item = CorpusItem(id="x", modality="text", source_ref="u",
                      title="Quantum", text="Quantum mechanics describes atoms and photons. "
                      "Quantum theory and quantum fields are central to physics.")
    g.ingest([item])
    df_before = {n.id: n.doc_frequency for n in g.store.nodes_of_type(NodeType.ENTITY)}
    changed = CorpusItem(id="x", modality="text", source_ref="u", title="Quantum",
                         text="Completely different: marine biology, coral reefs and fish.")
    g.ingest([changed])
    # the old entities/concepts must have been retracted (df back down), not double-counted
    for tid, df in df_before.items():
        assert g.store.get_node(tid).doc_frequency <= df


def test_vector_retriever_excludes_superseded():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest([sample_items()[0]])
    changed = CorpusItem(id="a", modality="text", source_ref="u/a", title="Alan Turing",
                         text="Alan Turing mathematician cryptography Bletchley computing.")
    g.ingest([changed])
    res = g.query("Alan Turing cryptography", mode="vector", k=5)
    assert "obj_a" not in res.object_ids  # the superseded original is excluded


def test_object_count_counts_valid_only():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest([sample_items()[0]])
    assert g.store.object_count() == 1
    changed = CorpusItem(id="a", modality="text", source_ref="u/a", title="X",
                         text="Totally different content about volcanoes and geology here.")
    g.ingest([changed])
    assert g.store.object_count() == 1  # old superseded object not counted


def test_vector_dim_guard():
    idx = VectorIndex(dim=8)
    idx.add("object", "o", np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        idx.search("object", np.ones(4, dtype=np.float32), k=1)


def test_ppr_excludes_community_edges():
    """Regression: building communities must not change PPR results (IN_COMMUNITY
    edges are excluded from the traversal projection)."""
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    before = g.query("cryptography Bletchley", mode="ppr", k=4).object_ids
    g.build_communities()
    after = g.query("cryptography Bletchley", mode="ppr", k=4).object_ids
    assert before == after


def test_eval_metrics():
    from kg.evaluate import (_mrr, _recall_at_k, cross_article_questions, evaluate,
                             single_article_questions)
    assert _recall_at_k(["a", "b", "c"], {"b"}, 3) == 1.0
    assert _recall_at_k(["a", "b"], {"z"}, 2) == 0.0
    assert _mrr(["a", "b"], {"b"}) == 0.5
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    qs = single_article_questions(g, limit=3) + cross_article_questions(g, limit=3)
    assert qs
    scores = evaluate(g, qs, modes=("ppr", "vector"), k=5)
    assert len(scores) == 2
    assert all(0.0 <= s.recall_at_k <= 1.0 for s in scores)


# --------------------------------------------------------------------------- #
# corpus loader (uses the real frozen dataset on disk)
# --------------------------------------------------------------------------- #
def test_viz_payloads_and_html():
    from kg.viz import graph_payload, query_trace, render_html
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    gp = graph_payload(g.store)
    assert gp["nodes"] and len(gp["build_order"]) == len(gp["nodes"])
    assert all("x" in n and "y" in n for n in gp["nodes"])
    tr = query_trace(g, "cryptography Bletchley codebreaking", mode="bfs")
    assert tr["mode"] == "bfs"
    assert tr["nodes"] and isinstance(tr["ranked"], list)
    assert all(0.0 <= n["x"] <= 1.0 for n in tr["nodes"])
    html = render_html(gp, trace=tr, server=False)
    assert "/*__DATA__*/" not in html and "<svg" in html


def test_viz_global_query_has_no_traversal():
    from kg.viz import query_trace
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    g.build_communities()
    tr = query_trace(g, "what are the main themes", mode="auto")
    # community/global queries have no node-level path to draw
    assert tr["nodes"] == [] and "note" in tr


def test_corpus_loads_from_disk():
    arts = load_articles(limit=5)
    imgs = load_images(limit=5)
    assert len(arts) == 5 and all(a.modality == "text" and a.text for a in arts)
    assert len(imgs) == 5 and all(i.modality == "image" for i in imgs)
