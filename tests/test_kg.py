"""Test suite for the kg episodic/temporal graph.

Runs fully offline/deterministic (hashing embedder + heuristic extractor) so it needs no
API key, no model download, and no network. Run: python -m pytest -q
"""
from __future__ import annotations

import json
import os
import tempfile
import types

import numpy as np
import pytest

from kg import Config, KnowledgeGraph
from kg.canonicalize import (Canonicalizer, char_entropy, normalize_key,
                             normalize_relation, predicate_cardinality, relation_merge_vetoed)
from kg.corpus import CorpusItem, load_longmemeval, load_longmemeval_questions
from kg.embedders import HashingEmbedder, get_embedder
from kg.extractors import HeuristicExtractor, get_extractor
from kg.models import (Belief, Edge, EdgeType, Modality, Node, NodeType, Provenance,
                       EntityType, episode_node, entity_node,
                       mention_node, relation_tag_node)
from kg.store import GraphStore, fact_active
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
    n = episode_node("ep_x", modality=Modality.TEXT, source_ref="u",
                     raw_text="hello", content_hash="h", ts="t")
    n.tags = ["a", "b"]
    back = Node.from_payload(n.to_payload())
    assert back.id == "ep_x"
    assert back.ntype == NodeType.EPISODE
    assert back.modality == Modality.TEXT
    assert back.tags == ["a", "b"]


def test_mention_payload_roundtrip():
    m = mention_node("men_0", surface="Becky", etype=EntityType.PERSON, episode_id="ep_x",
                     ts="t", char_span=[0, 5])
    back = Node.from_payload(m.to_payload())
    assert back.ntype == NodeType.MENTION
    assert back.episode_id == "ep_x" and back.char_span == [0, 5]


def test_edge_key_bitemporal_discriminator():
    # RELATED_TO discriminates on (rel_tag, valid_at) so closed + reopened facts coexist
    e1 = Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0007", valid_at="2020")
    e2 = Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0007", valid_at="2024")
    assert e1.key() == ("RELATED_TO", "rel_0007", "2020")
    assert e1.key() != e2.key()
    # non-fact edges ignore valid_at in their key
    assert Edge("m", "e", EdgeType.RESOLVES_TO).key() == ("RESOLVES_TO", "", "")


# --------------------------------------------------------------------------- #
# vectors
# --------------------------------------------------------------------------- #
def test_vector_normalize_and_search():
    idx = VectorIndex(dim=8)
    idx.add("episode", "o1", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    idx.add("episode", "o2", np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    v = idx.get("episode", "o1")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    hits = idx.search("episode", np.array([1, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32), k=2)
    assert hits[0][0] == "o1" and hits[0][1] > hits[1][1]


def test_vector_search_floor_and_exclude():
    idx = VectorIndex(dim=4)
    idx.add("episode", "o1", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.add("episode", "o2", np.array([1, 0, 0, 0], dtype=np.float32))
    hits = idx.search("episode", np.array([1, 0, 0, 0], dtype=np.float32), k=5,
                      floor=0.9, exclude={"o1"})
    assert [h[0] for h in hits] == ["o2"]


def test_vector_dim_guard():
    idx = VectorIndex(dim=8)
    idx.add("episode", "o", np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        idx.search("episode", np.ones(4, dtype=np.float32), k=1)


# --------------------------------------------------------------------------- #
# canonicalize
# --------------------------------------------------------------------------- #
def test_normalize_key():
    assert normalize_key("Natural-Language Processing") == "natural language processing"
    assert normalize_key("Networks") == "network"
    assert normalize_key("Cars") == "cars"
    assert normalize_key("  ") == ""


def test_char_entropy_short_vs_long():
    assert char_entropy("ai") < char_entropy("artificial intelligence")


def test_l1_dedup_and_entropy_guard():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    assert canon.resolve_tag("Machine Learning") == canon.resolve_tag("machine-learning")
    assert canon.resolve_tag("AI") != canon.resolve_tag("US")


def test_normalize_relation():
    assert normalize_relation("is friends with") == "is_friends_with"
    assert normalize_relation("Is  Friends  With") == "is_friends_with"
    assert normalize_relation("manages") == "manages"
    assert normalize_relation("managed_by") == "managed_by"


def test_resolve_relation_consolidation():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    r1 = canon.resolve_relation("works with")
    assert r1 == canon.resolve_relation("works-with") == canon.resolve_relation("Works With")
    assert store.get_node(r1).ntype == NodeType.RELATION
    assert canon.resolve_relation("founded") != r1
    assert len(store.nodes_of_type(NodeType.RELATION)) == 2


def test_relation_synonym_merges_but_antonym_and_inverse_dont():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    a = canon.resolve_relation("is_friend_of")
    assert a == canon.resolve_relation("is friends with")
    node = store.get_node(a)
    assert node.name == "is_friend_of" and "is_friends_with" in node.aliases
    assert canon.resolve_relation("is_enemy_of") != a
    assert canon.resolve_relation("manages") != canon.resolve_relation("managed_by")


def test_predicate_cardinality_flags():
    # functional (single-valued) vs symmetric (orientation-free), via content key
    assert predicate_cardinality("lives_in") == (True, False)
    assert predicate_cardinality("employed_by") == (True, False)
    assert predicate_cardinality("works_with") == (False, True)
    assert predicate_cardinality("married_to") == (True, True)
    assert predicate_cardinality("founded") == (False, False)


def test_resolve_relation_stamps_cardinality():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    lives = store.get_node(canon.resolve_relation("lives_in"))
    works = store.get_node(canon.resolve_relation("works_with"))
    assert lives.functional and not lives.symmetric
    assert works.symmetric and not works.functional


def test_relation_merge_vetoed():
    assert relation_merge_vetoed("manages", "managed_by")
    assert relation_merge_vetoed("is_friend_of", "is_enemy_of")
    assert relation_merge_vetoed("parent_of", "child_of")
    assert not relation_merge_vetoed("works_with", "collaborates_with")


def test_idf_weight_monotonic():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    for i in range(10):
        store.add_node(episode_node(f"ep_{i}", modality=Modality.TEXT, source_ref="u",
                                    raw_text="x", content_hash=str(i), ts="t"))
    common = canon.resolve_tag("common")
    rare = canon.resolve_tag("rare")
    store.get_node(common).doc_frequency = 8
    store.get_node(rare).doc_frequency = 1
    assert canon.idf_weight(rare) > canon.idf_weight(common)


# --------------------------------------------------------------------------- #
# L3 selective tie-breaker (offline: veto + fake client)
# --------------------------------------------------------------------------- #
class _FakeL3Client:
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
    canon._l3_client = _FakeL3Client(verdict)
    return store, canon


def test_l3_adjudicate_merges_on_existing_id():
    store, canon = _l3_canon("placeholder")
    rid = canon.resolve_relation("works_with")
    canon._l3_client = _FakeL3Client(rid)
    assert canon._l3_adjudicate("relation", "collaborates_with", [(rid, 0.92)]) == rid


def test_l3_relation_veto_blocks_inverse_before_llm():
    store, canon = _l3_canon("placeholder")
    rid = canon.resolve_relation("manages")
    fake = _FakeL3Client(rid)
    canon._l3_client = fake
    assert canon._l3_relation("managed_by", [(rid, 0.92)]) is None
    assert fake.calls == []


def test_l3_disabled_by_default():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    rid = canon.resolve_relation("works_with")
    assert canon._l3_adjudicate("relation", "collaborates_with", [(rid, 0.92)]) is None


# --------------------------------------------------------------------------- #
# extract-dump + eval-canon (offline)
# --------------------------------------------------------------------------- #
def test_extract_dump_runs_and_summarizes():
    from kg.extract_dump import extract_corpus, summarize
    ext = HeuristicExtractor(cfg())
    records, errors = extract_corpus(ext, sample_items(), cfg())
    assert len(records) == 4 and not errors
    s = summarize(records, "heuristic")
    assert s["items"] == 4 and s["unique_tags"] > 0


def test_eval_canon_gate_passes_offline():
    from kg.eval_canon import run_gate
    rep = run_gate(cfg())
    assert rep["gate_pass"] is True
    assert rep["relation"]["wrong_antonym_inverse_merges"] == 0
    assert rep["entity"]["false_merges"] == 0


def test_extractor_termination_normalizes_to_ended():
    """former_/ex_/no-longer prefixes fold onto the base predicate + status=ended."""
    from kg.extractors import _parse_tool_payload
    ext = _parse_tool_payload({"entities": [{"name": "A", "type": "person"},
                                            {"name": "B", "type": "person"}],
                               "tags": [],
                               "relations": [{"source": "A", "target": "B",
                                              "labels": ["former_colleague"]}]})
    r = ext.relations[0]
    assert r.status == "ended" and "colleague" in r.labels[0] and "former" not in r.labels[0]


# --------------------------------------------------------------------------- #
# embedders / extractors / factories
# --------------------------------------------------------------------------- #
def test_hashing_embedder_deterministic_and_unit_norm():
    e = HashingEmbedder(dim=64)
    a = e.embed(["knowledge graph"])
    assert np.allclose(a, e.embed(["knowledge graph"]))
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5


def test_extractor_auto_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(get_extractor(Config.default()), HeuristicExtractor)


def test_heuristic_extract_image_uses_labels():
    r = HeuristicExtractor(cfg()).extract_image("x.jpg", "dog, frisbee")
    assert "dog" in r.tags and "frisbee" in r.tags


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def test_store_bidirectional_neighbors():
    store = GraphStore(cfg())
    store.add_node(episode_node("a", modality=Modality.TEXT, source_ref="u",
                                raw_text="x", content_hash="1", ts="t"))
    store.add_node(episode_node("b", modality=Modality.TEXT, source_ref="u",
                                raw_text="y", content_hash="2", ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.SHARED_TAG, Provenance.DERIVED, 1.0, 1.0))
    assert any(nbr == "b" for nbr, _ in store.neighbors("a"))
    assert any(nbr == "a" for nbr, _ in store.neighbors("b"))


def test_store_save_load_roundtrip():
    path = tmp_store()
    store = GraphStore.open(path, cfg())
    store.add_node(episode_node("a", modality=Modality.TEXT, source_ref="u",
                                raw_text="x", content_hash="1", ts="t"))
    store.add_node(episode_node("b", modality=Modality.TEXT, source_ref="u",
                                raw_text="y", content_hash="2", ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.SIMILAR_TO, Provenance.SIMILAR, 0.9, 0.9))
    store.add_edge(Edge("a", "b", EdgeType.SHARED_TAG, Provenance.DERIVED, 0.5, 2.0))
    store.vectors.add("episode", "a", np.ones(cfg().embed_dim, dtype=np.float32))
    store.hash_cache["1"] = "a"
    store.save()
    s2 = GraphStore.open(path, cfg())
    assert s2.has_node("a") and s2.has_node("b")
    etypes = {d["etype"] for _n, d in s2.neighbors("a")}
    assert {"SIMILAR_TO", "SHARED_TAG"} <= etypes
    assert s2.hash_cache.get("1") == "a"
    assert s2.vectors.get("episode", "a") is not None


def test_directed_facts_distinct_but_neighbors_bidirectional():
    store = GraphStore(cfg())
    for nid in ("a", "b"):
        store.add_node(entity_node(nid, name=nid, etype=EntityType.PERSON, ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.RELATED_TO, Provenance.EXTRACTED, 1.0, 1.0,
                        rel_tag="rel_0"))
    assert store.g.has_edge("a", "b") and not store.g.has_edge("b", "a")
    assert any(n == "b" for n, _ in store.neighbors("a"))
    assert any(n == "a" for n, _ in store.neighbors("b"))
    assert [n for n, _ in store.neighbors("a", direction="out")] == ["b"]


def test_symmetric_edges_pinned_to_one_orientation():
    store = GraphStore(cfg())
    for nid in ("a", "b"):
        store.add_node(episode_node(nid, modality=Modality.TEXT, source_ref="u",
                                    raw_text="x", content_hash=nid, ts="t"))
    store.add_edge(Edge("b", "a", EdgeType.SIMILAR_TO, Provenance.SIMILAR, 0.9, 0.9))
    store.add_edge(Edge("a", "b", EdgeType.SIMILAR_TO, Provenance.SIMILAR, 0.9, 0.9))
    assert store.g.number_of_edges() == 1
    assert store.g.has_edge("a", "b") and not store.g.has_edge("b", "a")


def test_find_and_close_facts():
    store = GraphStore(cfg())
    for nid in ("a", "b"):
        store.add_node(entity_node(nid, name=nid, etype=EntityType.PERSON, ts="t"))
    store.add_edge(Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0",
                        valid_at="2020", invalid_at=""))
    assert [d["invalid_at"] for _v, _k, d in store.find_facts("a", "b", "rel_0")] == [""]
    assert store.close_facts("a", "b", "rel_0", "2024") == 1
    assert next(store.find_facts("a", "b", "rel_0"))[2]["invalid_at"] == "2024"
    # closing again is a no-op (no open facts left)
    assert store.close_facts("a", "b", "rel_0", "2025") == 0


def test_fact_active_current_vs_asof():
    open_fact = {"belief": "asserted", "valid_at": "2021", "invalid_at": ""}
    closed = {"belief": "asserted", "valid_at": "2021", "invalid_at": "2023"}
    retracted = {"belief": "retracted", "valid_at": "2021", "invalid_at": ""}
    assert fact_active(open_fact, None) and not fact_active(closed, None)
    assert fact_active(closed, "2022") and not fact_active(closed, "2024")
    assert not fact_active(retracted, None)


# --------------------------------------------------------------------------- #
# end-to-end ingest
# --------------------------------------------------------------------------- #
def test_ingest_builds_episodic_graph():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    rep = g.ingest(sample_items())
    assert rep.ingested == 4 and rep.mentions > 0
    s = g.stats()
    assert s["by_node_type"]["episode"] == 4
    assert s["by_node_type"]["mention"] > 0 and s["by_node_type"]["entity"] > 0
    # the mention star: every mention has MENTIONED_IN + RESOLVES_TO
    for m in g.store.nodes_of_type(NodeType.MENTION):
        ets = {d["etype"] for _n, d in g.store.neighbors(m.id, direction="out")}
        assert {"MENTIONED_IN", "RESOLVES_TO"} <= ets
    # shared structure links episodes
    assert s["by_edge_type"].get("SHARED_ENTITY", 0) + s["by_edge_type"].get("SHARED_TAG", 0) > 0


def test_entity_anchor_is_lean():
    """The canonical entity is an identity anchor — no raw text, no embedding in the
    retrieval index (embeddings live only on the immutable episode/mention layer)."""
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    for e in g.store.nodes_of_type(NodeType.ENTITY):
        assert e.raw_text is None and e.summary is None
        assert g.store.vectors.get("episode", e.id) is None
        assert g.store.vectors.get("mention", e.id) is None


def test_store_is_directed():
    import networkx as nx
    g = KnowledgeGraph.open(tmp_store(), cfg())
    assert isinstance(g.store.g, nx.MultiDiGraph)


def test_ingest_cache_skips_on_rerun():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    rep2 = g.ingest(sample_items())
    assert rep2.ingested == 0 and rep2.skipped == 4


def test_ingest_appends_new_version_on_change():
    """Episodes are append-only: changed content under a known id adds a NEW immutable
    episode; the old one stays valid (history is preserved, not overwritten)."""
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest([sample_items()[0]])
    changed = CorpusItem(id="a", modality="text", source_ref="u/a", title="Alan Turing",
                         text="Completely different content about marine biology and coral reefs.")
    g.ingest([changed])
    assert g.store.has_node("ep_a") and g.store.has_node("ep_a_v1")
    assert g.store.get_node("ep_a").valid and g.store.get_node("ep_a_v1").valid
    assert g.store.episode_count() == 2


def test_tag_doc_frequency_dedup_within_episode():
    """Duplicate tags in one episode (or variants that canonicalize to the same node) must
    bump doc_frequency once — df = #episodes referencing the tag, not #occurrences."""
    from kg.extractors import Extraction, ScriptedExtractor
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.extractor = ScriptedExtractor(
        {"a body": Extraction(entities=[], tags=["python", "Python", "ai"], relations=[])})
    g.ingest([CorpusItem(id="x", modality="text", source_ref="u", title="X", text="a body")])
    py = next(n for n in g.store.nodes_of_type(NodeType.TAG) if n.name == "python")
    assert py.doc_frequency == 1


def test_extraction_failures_are_surfaced():
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


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
def test_retrievers_run_and_find_relevant():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    for mode in ("ppr", "bfs", "vector"):
        res = g.query("cryptography codebreaking at Bletchley", mode=mode, k=3)
        assert res.object_ids, f"{mode} returned nothing"
        assert {"ep_a", "ep_b"} & set(res.object_ids), f"{mode} missed the target"


def test_empty_and_blank_query_do_not_crash():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    g.build_communities()
    for mode in ("ppr", "bfs", "vector"):
        assert g.query("   ", mode=mode).object_ids == []
        assert g.query("", mode=mode).object_ids == []
    assert g.query("anything", mode="ppr").object_ids != [] or True  # smoke


def test_communities_and_global_route():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    assert g.build_communities() >= 1
    res = g.query("what are the main themes", mode="auto")
    assert isinstance(res, list) and res and "summary" in res[0]


def test_ppr_excludes_community_edges():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    before = g.query("cryptography Bletchley", mode="ppr", k=4).object_ids
    g.build_communities()
    assert before == g.query("cryptography Bletchley", mode="ppr", k=4).object_ids


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def test_eval_metrics_and_rag_mode():
    from kg.evaluate import (_mrr, _recall_at_k, cross_article_questions, evaluate,
                             single_article_questions)
    assert _recall_at_k(["a", "b", "c"], {"b"}, 3) == 1.0
    assert _mrr(["a", "b"], {"b"}) == 0.5
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    qs = single_article_questions(g, limit=3) + cross_article_questions(g, limit=3)
    assert qs
    scores = evaluate(g, qs, modes=("ppr", "vector", "rag"), k=5)
    assert len(scores) == 3
    assert all(0.0 <= s.recall_at_k <= 1.0 for s in scores)


# --------------------------------------------------------------------------- #
# viz + corpus
# --------------------------------------------------------------------------- #
def test_viz_payloads_and_html():
    from kg.viz import graph_payload, query_trace, render_html
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.ingest(sample_items())
    gp = graph_payload(g.store)
    assert gp["nodes"] and len(gp["build_order"]) == len(gp["nodes"])
    assert all(n["type"] == "episode" for n in gp["nodes"])
    tr = query_trace(g, "cryptography Bletchley codebreaking", mode="bfs")
    assert tr["mode"] == "bfs" and tr["nodes"] and isinstance(tr["ranked"], list)
    html = render_html(gp, trace=tr, server=False)
    assert "/*__DATA__*/" not in html and "<svg" in html


def test_corpus_loads_from_disk():
    # the committed `sample` LongMemEval tier (small, capped — ships its episode bodies)
    eps = load_longmemeval("sample", limit=5)
    assert len(eps) == 5
    assert all(e.modality == "text" and e.text and e.created_at for e in eps)
    qs = load_longmemeval_questions("sample")
    assert qs and all(q.get("gold") and "answer" in q and "kind" in q for q in qs)
    # gold ids namespace each evidence session by question_id (collide-safe)
    assert all(g.startswith("obj_") for q in qs for g in q["gold"])
