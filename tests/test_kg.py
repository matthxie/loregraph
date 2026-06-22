"""Test suite for the kg MVP.

Runs fully offline/deterministic (hashing embedder + heuristic extractor) so it
needs no API key, no model download, and no network. Run: python -m pytest -q
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from kg import Config, KnowledgeGraph
from kg.canonicalize import Canonicalizer, char_entropy, normalize_key
from kg.corpus import CorpusItem, load_articles, load_images
from kg.embedders import HashingEmbedder, get_embedder
from kg.extractors import HeuristicExtractor, get_extractor
from kg.models import (Edge, EdgeType, Modality, Node, NodeType,
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
    t1 = canon.resolve_tag("Machine Learning")
    t2 = canon.resolve_tag("machine-learning")  # normalizes to same key
    assert t1 == t2
    # short low-entropy tags must NOT be embedding-merged: distinct nodes
    a = canon.resolve_tag("AI")
    b = canon.resolve_tag("US")
    assert a != b


def test_idf_weight_monotonic():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    # add some object nodes so the corpus size is > 0
    for i in range(10):
        store.add_node(object_node(f"obj_{i}", modality=Modality.TEXT, source_ref="u",
                                   raw_text="x", content_hash=str(i), ts="t"))
    common = canon.resolve_tag("common")
    rare = canon.resolve_tag("rare")
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
    assert len(r.tags) > 0


def test_heuristic_extract_image_uses_labels():
    ext = HeuristicExtractor(cfg())
    r = ext.extract_image("x.jpg", "dog, frisbee")
    assert "dog" in r.tags and "frisbee" in r.tags
    assert r.description and "dog" in r.description


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
    assert s["by_node_type"]["tag"] > 0
    assert s["by_node_type"]["entity"] > 0
    # Turing & Bletchley should share an entity → a derived object-object edge exists
    assert s["by_edge_type"].get("SHARED_ENTITY", 0) + s["by_edge_type"].get("SHARED_TAG", 0) > 0


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
    df_before = {n.id: n.doc_frequency for n in g.store.nodes_of_type(NodeType.TAG)}
    changed = CorpusItem(id="x", modality="text", source_ref="u", title="Quantum",
                         text="Completely different: marine biology, coral reefs and fish.")
    g.ingest([changed])
    # the old tags must have been retracted (df back to 0), not left double-counted
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
